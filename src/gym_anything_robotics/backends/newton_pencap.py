"""Pen-cap adapter to the existing local and authenticated HTTP interfaces."""

import base64
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import uuid
import numpy as np
from .base import RoboticsBackend, reject_unexecuted
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation


class NewtonPenCapBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-panda", config=None, device=None, cameras=("bench_rgb",),
                 output_dir=None, renderer_options=None, recording=True):
        from pencap.newton15.config import Config
        from pencap.newton15.env import PenCapEnv
        from pencap.newton15.catalog import Catalog
        from pencap.newton15.kinematics import Kinematics
        if robot != "dual-panda":
            raise UnsupportedConfigurationError("Pen-cap uses two manufacturer-description Panda arms")
        if isinstance(config, dict):
            config = Config(**config)
        config = config or Config(scene="studio")
        if device is not None:
            config = replace(config, device=device)
        if config.dt > 1/960:
            raise UnsupportedConfigurationError("Pen-cap contacts require dt <= 1/960 s; the 1/240 s migration penetrated the shoulder")
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras)-{"bench_rgb"}:
            raise ValueError("Policy camera is bench_rgb; other views are diagnostic render labels")
        if self.cameras and config.scene != "studio":
            raise ValueError("Mounted policy camera requires the studio scene")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir or Path.home()/".cache/gar/pencap"/uuid.uuid4().hex)
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = PenCapEnv(config)
        self.catalog, self.ik = Catalog(self.env), Kinematics()
        self.images = deque()
        self.trace = None
        if recording:
            from pencap.newton15.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        focal = .018/math.tan(math.radians(69/2))
        self.camera = CameraSpec("bench_rgb", "floor_stand_D435_enclosure", width, height, 1920, 1080,
            (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities("pencap", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, (self.camera,) if self.cameras else (), notes={
                "clock": "Exact requested ticks; duration must be a whole number of physics ticks. Simulated time "
                         "advances only while a command runs: between commands and after your last command the scene "
                         "is frozen, and the task is judged on the scene as your last command leaves it",
                "control": "Physical servos; IK is not a collision-free path planner",
                "objects": "Actions control the robot only; pen, cap and notebook move through contact",
                "camera": "Ideal pinhole, D435-like RGB FOV; filmback is virtual, not a measured sensor size",
                "gripper": "0..40 mm per finger; fixed baseline 8 N effort limit; no effort action",
                "joint_units": "panda_joint1..7 radians; panda_finger_joint1/2 metres"})

    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.env.config.seed):
            raise ValueError("This fixed baseline configuration has no randomization or reset options")
        return self.env.reset(), {"continuity": "reset", "seed": self.env.config.seed}

    def get_state(self):
        return self.env.get_state()

    def _render(self, names):
        from pencap.newton15.render import open_renderer
        with self.env.lock:
            from ..render_lifecycle import renderer_operation
            with renderer_operation(self.env, "renderer"):
                if self.env.renderer is None:
                    self.env.renderer = open_renderer(self.env, self.output/"render", **self.options)
                result = self.env.renderer.observe(self.env.tick, self.env.state.body_q.numpy(), names)
            frames = {}
            for name, blob in result["images"].items():
                path = self.output/f"{name}_{self.env.generation}_{self.env.tick}_{uuid.uuid4().hex}.png"
                path.write_bytes(base64.b64decode(blob))
                self.images.append(path)
                frames[name] = path
            while len(self.images) > 64:
                old = self.images.popleft()
                if old.parent != self.output:
                    raise RuntimeError("Image retention escaped session folder")
                old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and new camera frames")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], "dual-panda", {"joints": state["joints"]},
                {n: ImageFrame(n, p, True, camera=self.camera) for n, p in frames.items()})

    def _ticks(self, seconds):
        ticks = seconds/self.env.config.dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Requested duration is not exactly representable by this physics timestep")
        return int(round(ticks))

    def _fingers(self, arm, width):
        if not 0. <= width <= .04:
            raise ValueError("Panda half gap must be in [0, .04] metres")
        return {f"{arm}/panda_finger_joint{i}": width for i in (1, 2)}

    def act(self, action):
        # A rejected target (unknown joint, outside the published limits) cancels the command, not the session.
        return reject_unexecuted(lambda: self._act(action), lambda: (self.env.tick, self.env.targets.copy()))

    def _act(self, action):
        guard = None
        with self.env.lock:
            if isinstance(action, NamedJointTarget):
                targets, ticks = dict(action.targets_si), action.ticks
            elif isinstance(action, Gripper):
                if action.inward_effort_n is not None:
                    raise UnsupportedActionError("Gripper effort action unavailable; baseline limit is 8 N per finger")
                targets, ticks = self._fingers(action.arm, action.half_width_m), self._ticks(action.duration_s)
            elif isinstance(action, (CartesianPose, CartesianDelta)):
                if action.safety.stop_on_grip_loss is not None:
                    raise UnsupportedActionError("Grip-loss guard is not implemented")
                ticks = self._ticks(action.duration_s)
                if isinstance(action, CartesianPose):
                    position = action.position_m
                    orientation = action.orientation_wxyz
                    orientation = None if orientation is None else (*orientation[1:], orientation[0])
                else:
                    q = self.env.state.joint_q.numpy()
                    joints = [q[self.env.joints[f"{action.arm}/panda_joint{i}"]["q"]] for i in range(1, 8)]
                    position = self.ik.forward(joints, action.arm, self.env.config.table_top)[:3, 3]+action.delta_m
                    orientation = None
                targets = self.ik.solve(self.env, action.arm, position, orientation)
                if action.gripper_half_target_m is not None:
                    targets.update(self._fingers(action.arm, action.gripper_half_target_m))
                safety = action.safety
                def guard(e):
                    n = int(e.contacts.rigid_contact_count.numpy()[0])
                    a, b = (getattr(e.contacts, "rigid_contact_"+k).numpy()[:n] for k in ("shape0", "shape1"))
                    def arm(i):
                        body = e.builder.shape_body[int(i)]
                        label = e.builder.body_label[body] if body >= 0 else ""
                        return next((x for x in ("left", "right") if label.startswith(x)), None)
                    for x, y in zip(a, b):
                        x, y = arm(x), arm(y)
                        if safety.stop_on_arm_arm_contact and x and y and x != y:
                            return True
                        if (safety.stop_on_robot_contact or safety.stop_on_contact) and (x or y):
                            return True
                    return False
            else:
                raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta")
            before = self.env.tick
            try:
                state = self.env.act(targets, ticks=ticks, guard=guard)
            except (RuntimeError, FloatingPointError, OSError) as exc:
                raise BackendProtocolError(f"Stopped at tick {self.env.tick}: {exc}") from exc
            info = {"requested_ticks": ticks, "executed_ticks": self.env.tick-before,
                "stopped_on_contact": self.env.tick-before < ticks}
            return BackendStep(state, info, state["evaluation"]["score"], state["evaluation"]["success"])

    def policy_access(self):
        from pencap.newton15.task import DESCRIPTION
        return PolicyAccess("dual-panda", (self.camera,) if self.cameras else (),
            {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")},
            self.kinds, lambda: DESCRIPTION, {"dt_s": self.env.config.dt,
                "clock": self.capabilities.notes["clock"], "objects": self.capabilities.notes["objects"],
                "joints": {n: {**{k:v for k,v in s.items() if k not in ("q", "dof")},
                    "unit": "metre" if "finger" in n else "radian"} for n,s in self.env.joints.items()}})

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, False, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        if name not in ("room", "bench", "closeup", "bench_rgb"):
            raise ValueError("Render label must be room, bench, closeup, or bench_rgb")
        return self._render((name,))[name]

    def get_geometry_index(self):
        return self.catalog.index()

    def get_geometry(self, geometry_id):
        return self.catalog.objects[str(geometry_id)]

    def get_geometry_asset(self, asset_id):
        return self.catalog.assets[str(asset_id)]

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def close(self):
        if self.trace:
            self.trace.close()
            self.trace = None
        self.env.close()

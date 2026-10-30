"""Lock-and-key bridge to GAR's existing authenticated HTTP and local protocol."""

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


class NewtonLockKeyBackend(RoboticsBackend):
    def __init__(self, *, robot="panda", config=None, device=None, cameras=("bench_rgb",),
                 output_dir=None, renderer_options=None, recording=True, task="unlock"):
        from lockkey.newton15.config import Config
        from lockkey.newton15.env import LockKeyEnv
        from lockkey.newton15.catalog import Catalog
        from lockkey.newton15.kinematics import Kinematics
        from lockkey.task_set.catalogue import validate
        self.task = validate(task)
        if robot != "panda":
            raise UnsupportedConfigurationError("Lock-and-key uses one manufacturer-description Panda")
        config = Config(**config) if isinstance(config, dict) else (config or Config())
        if device is not None:
            config = replace(config, device=device)
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras)-{"bench_rgb"}:
            raise ValueError("Policy camera is bench_rgb; other views are privileged diagnostics")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir or Path.home()/".cache/gar/lockkey"/uuid.uuid4().hex)
        self.output.mkdir(parents=True, exist_ok=False)
        if self.task == "unlock":
            self.env = LockKeyEnv(config)
        else:
            from lockkey.task_set.adapter import TaskLockKeyEnv
            self.env = TaskLockKeyEnv(config, self.task)
        self.catalog, self.ik = Catalog(self.env), Kinematics()
        self.images, self.trace = deque(), None
        if recording:
            from lockkey.newton15.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        focal = .018/math.tan(math.radians(69/2))
        self.camera = CameraSpec("bench_rgb", "external_floor_stand", width, height, 1920, 1080,
            (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities("lockkey", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, (self.camera,) if self.cameras else (), notes={
                "qualification": ("Reference configuration qualified by assisted physical fixtures, RGB/history runtime and HTTP checks; see lockkey/REPORT.pdf"
                    if self.task == "unlock" else
                    "Task addition checked in constructed states only; settling, contacts and rendering are not yet checked in the simulator"),
                "clock": "Exact ticks; a duration must be a whole number of physics ticks",
                "control": "Physical position servos; IK is not collision-free path planning",
                "arm": "left selects the single Panda; right is unavailable",
                "camera": "Ideal pinhole, 69-degree horizontal FOV; external mounted RGB, not onboard hardware",
                "gripper": "0..40 mm per finger, 45 N effort limit; effort override unavailable",
                "joint_units": "panda/panda_joint1..7 radians; panda/panda_finger_joint1/2 metres"})

    def reset(self, *, seed=None, options=None):
        if options or seed is not None:
            raise ValueError("This fixed task has no reset randomization; select configuration when creating it")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        return self.env.get_state()

    def _render(self, names):
        from lockkey.newton15.render import open_renderer
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
                    raise RuntimeError("Image retention escaped the session folder")
                old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], "panda", {"joints": state["joints"]},
                {n: ImageFrame(n, p, True, camera=self.camera) for n, p in frames.items()})

    def _ticks(self, seconds):
        ticks = seconds/self.env.config.dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8, rel_tol=0):
            raise ValueError("Duration cannot be represented exactly by the physics timestep")
        return int(round(ticks))

    @staticmethod
    def _arm(arm):
        if arm != "left":
            raise UnsupportedActionError("This single-Panda task uses arm='left'; there is no right arm")

    @staticmethod
    def _fingers(width):
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 0. <= width <= .04:
            raise ValueError("Panda half gap must be in [0, .04] metres")
        return {f"panda/panda_finger_joint{i}": width for i in (1, 2)}

    def act(self, action):
        # A rejected target (unknown joint, outside the published limits) cancels the command, not the session.
        return reject_unexecuted(lambda: self._act(action), lambda: (self.env.tick, self.env.targets.copy()))

    def _act(self, action):
        guard = None
        with self.env.lock:
            if isinstance(action, NamedJointTarget):
                targets, ticks = dict(action.targets_si), action.ticks
            elif isinstance(action, Gripper):
                self._arm(action.arm)
                if action.inward_effort_n is not None:
                    raise UnsupportedActionError("Effort override unavailable; the gripper has its fixed 45 N limit")
                targets, ticks = self._fingers(action.half_width_m), self._ticks(action.duration_s)
            elif isinstance(action, (CartesianPose, CartesianDelta)):
                self._arm(action.arm)
                if action.safety.stop_on_grip_loss is not None:
                    raise UnsupportedActionError("Grip-loss guard is not implemented")
                ticks = self._ticks(action.duration_s)
                if isinstance(action, CartesianPose):
                    position = action.position_m
                    orientation = action.orientation_wxyz
                    orientation = None if orientation is None else (*orientation[1:], orientation[0])
                else:
                    q = self.env.state.joint_q.numpy()
                    joints = [q[self.env.joints[f"panda/panda_joint{i}"]["q"]] for i in range(1, 8)]
                    position = self.ik.forward(joints, self.env.config.table_height)[:3, 3]+action.delta_m
                    orientation = None
                targets = self.ik.solve(self.env, position, orientation)
                if action.gripper_half_target_m is not None:
                    targets.update(self._fingers(action.gripper_half_target_m))
                safety = action.safety
                def guard(env):
                    if not (safety.stop_on_robot_contact or safety.stop_on_contact):
                        return False
                    n = int(env.contacts.rigid_contact_count.numpy()[0])
                    for field in ("shape0", "shape1"):
                        for shape in getattr(env.contacts, "rigid_contact_"+field).numpy()[:n]:
                            body = env.builder.shape_body[int(shape)]
                            if body >= 0 and env.builder.body_label[body].startswith("panda"):
                                return True
                    return False
            else:
                raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta")
            before = self.env.tick
            try:
                state = self.env.act(targets, ticks=ticks, guard=guard)
            except (RuntimeError, FloatingPointError, OSError) as exc:
                raise BackendProtocolError(f"Stopped at tick {self.env.tick}: {exc}") from exc
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.tick-before,
                "stopped_on_contact": self.env.tick-before < ticks}, state["evaluation"]["score"], state["evaluation"]["success"])

    def policy_access(self):
        from lockkey.task_set.catalogue import description
        return PolicyAccess("panda", (self.camera,) if self.cameras else (),
            {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")},
            self.kinds, lambda: description(self.task), {"dt_s": self.env.config.dt,
                "joints": {n: {**{k:v for k,v in s.items() if k not in ("q", "dof")},
                    "unit": "metre" if "finger" in n else "radian"} for n,s in self.env.joints.items()}})

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, False, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        if name not in ("room", "bench", "lock", "keys", "keyway", "grasp", "bench_rgb"):
            raise ValueError("Render label must be room, bench, lock, keys, keyway, grasp, or bench_rgb")
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

"""Laundry workcell using GAR's common local/HTTP/policy/history protocol."""
import base64
from collections import deque
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import uuid
import numpy as np
from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta
from ..errors import ActionRejectedError, UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation


class NewtonLaundryBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-panda", config=None, device=None, cameras=("bench_rgb",),
                 output_dir=None, renderer_options=None, recording=True, task=None, contract=None):
        from laundry.newton15.config import Config
        from laundry.task_set_env import CatalogueLaundryEnv
        from laundry.task_set import select
        from laundry.newton15.catalog import Catalog
        from laundry.newton15.kinematics import Kinematics
        if robot != "dual-panda":
            raise UnsupportedConfigurationError("Laundry uses two manufacturer-description Panda arms")
        config = Config(**config) if isinstance(config, dict) else (config or Config())
        if device is not None:
            config = replace(config, device=device)
        select(config.garment, task, contract)
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras)-{"bench_rgb"}:
            raise ValueError("Policy camera is bench_rgb; other cameras are privileged diagnostics")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir or Path.home()/".cache/gar/laundry"/uuid.uuid4().hex)
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = CatalogueLaundryEnv(config, task=task, contract=contract)
        self.catalog, self.ik = Catalog(self.env), Kinematics(config.robot_base_y)
        self.images, self.trace = deque(), None
        self.render_recoveries = []
        if recording:
            from laundry.newton15.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        w, h = self.options.get("width", 1280), self.options.get("height", 720)
        f = .018/math.tan(math.radians(69/2))
        self.camera = CameraSpec("bench_rgb", "external_floor_stand", w, h, 1920, 1080,
            (.036, .036*h/w), (f, f))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities("laundry", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, (self.camera,) if self.cameras else (), notes={
                "clock": "Exact physics ticks; time advances only through actions. Empty named_joint_target holds all targets.",
                "control": "Physical servo targets. IK is not collision-free path planning.",
                "camera": "Mounted external ideal pinhole RGB, 69 degree horizontal FOV; no depth channel claimed.",
                "cloth": "Sewn Style3D cloth; partitioned contact reaction to finite-mass robot, one-tick delay.",
                "qualification": "Not an autonomous folding solver."})

    def reset(self, *, seed=None, options=None):
        if seed is not None or options:
            raise ActionRejectedError("Fixed constructed reset; select garment through config when creating the environment")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        try:
            state = self.env.get_state()
            state["render_recoveries"] = list(self.render_recoveries)
            return state
        except RuntimeError as exc:
            raise BackendProtocolError(str(exc)) from exc

    def _render(self, names):
        from laundry.newton15.render import Renderer
        with self.env.lock:
            from ..render_lifecycle import renderer_operation
            with renderer_operation(self.env, "renderer"):
                try:
                    for attempt in range(2):
                        if self.env.renderer is None:
                            folder = "render" if not self.render_recoveries else f"render_recovery_{len(self.render_recoveries)}"
                            self.env.renderer = Renderer(self.env, self.output/folder, **self.options)
                        renderer = self.env.renderer
                        try:
                            result = renderer.observe(self.env.tick, self.env.state.body_q.numpy(), names,
                                                      self.env.state.particle_q.numpy())
                            break
                        except (ConnectionError, OSError) as exc:
                            if attempt or len(self.render_recoveries) >= 3:
                                raise
                            # Recover only a confirmed exited render worker. A slow
                            # live renderer or an invalid request is not retried.
                            try:
                                exit_code = renderer.process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                raise exc
                            renderer.close()
                            self.env.renderer = None
                            self.render_recoveries.append(dict(tick=self.env.tick, generation=self.env.generation,
                                exit_code=exit_code, error=f"{type(exc).__name__}: {exc}",
                                physics_steps=0, reason="Exited render-only worker; same live physical state retained"))
                            (self.output/"render_recoveries.json").write_text(
                                json.dumps(self.render_recoveries, indent=2)+"\n", encoding="utf-8")
                except Exception as exc:
                    raise BackendProtocolError(f"Renderer failed at physical tick {self.env.tick}: {exc}") from exc
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
                raise RuntimeError("Cannot combine stale state and a new image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], "dual-panda", {"joints": state["joints"]},
                {n:ImageFrame(n, p, True, camera=self.camera) for n, p in frames.items()})

    def _ticks(self, seconds):
        value = seconds/self.env.config.dt
        if not math.isclose(value, round(value), abs_tol=1e-8, rel_tol=0):
            raise ActionRejectedError("Duration must be an exact whole number of physics ticks")
        return int(round(value))

    def _fingers(self, arm, width):
        if arm not in ("left", "right") or not 0 <= width <= .04:
            raise ActionRejectedError("Choose left/right and a half-width from 0 to .04 metres")
        return {f"{arm}/panda_finger_joint{i}": width for i in (1, 2)}

    def act(self, action):
        with self.env.lock:
            guard = None
            if isinstance(action, NamedJointTarget):
                targets, ticks = dict(action.targets_si), action.ticks
            elif isinstance(action, Gripper):
                if action.inward_effort_n is not None:
                    raise UnsupportedActionError("Effort override unavailable; manufacturer joint limits apply")
                targets, ticks = self._fingers(action.arm, action.half_width_m), self._ticks(action.duration_s)
            elif isinstance(action, (CartesianPose, CartesianDelta)):
                if action.safety.stop_on_grip_loss is not None:
                    raise UnsupportedActionError("Grip-loss guard is unavailable")
                ticks = self._ticks(action.duration_s)
                if isinstance(action, CartesianPose):
                    position, orientation = action.position_m, action.orientation_wxyz
                    orientation = None if orientation is None else (*orientation[1:], orientation[0])
                else:
                    q = self.env.state.joint_q.numpy()
                    joints = [q[self.env.joints[f"{action.arm}/panda_joint{i}"]["q"]] for i in range(1, 8)]
                    position = self.ik.forward(joints, action.arm, self.env.config.table_top)[:3, 3]+action.delta_m
                    orientation = None
                try:
                    targets = self.ik.solve(self.env, action.arm, position, orientation)
                except ValueError as exc:
                    # IK only reads state. A rejected pose has not touched the
                    # actuator targets or physics and must not fault the gateway.
                    raise ActionRejectedError(str(exc)) from exc
                if action.gripper_half_target_m is not None:
                    targets.update(self._fingers(action.arm, action.gripper_half_target_m))
                def guard(e):
                    s = action.safety
                    n = min(int(e.contacts.rigid_contact_count.numpy()[0]), e.contacts.rigid_contact_max)
                    def owner(i):
                        body = e.builder.shape_body[int(i)]
                        label = e.builder.body_label[body] if body >= 0 else ""
                        return next((a for a in ("left", "right") if label.startswith(a)), None)
                    a, b = (getattr(e.contacts, "rigid_contact_"+k).numpy()[:n] for k in ("shape0", "shape1"))
                    for i, j in zip(a, b):
                        first, second = owner(i), owner(j)
                        if s.stop_on_arm_arm_contact and first and second and first != second:
                            return True
                        if (s.stop_on_robot_contact or s.stop_on_contact) and (first or second):
                            return True
                    m = min(int(e.contacts.soft_contact_count.numpy()[0]), e.contacts.soft_contact_max)
                    return bool((s.stop_on_robot_contact or s.stop_on_contact) and
                                any(owner(i) for i in e.contacts.soft_contact_shape.numpy()[:m]))
            else:
                raise UnsupportedActionError("Use named_joint_target, gripper, cartesian_pose or cartesian_delta")
            before = self.env.tick
            try:
                state = self.env.act(targets, ticks=ticks, guard=guard)
            except Exception as exc:
                # A post-step evaluator/recorder error is not an invalid input.
                # Never return HTTP 400 after silently committing physical ticks.
                if self.env.tick != before or self.env.fault is not None:
                    self.env.fault = self.env.fault or f"{type(exc).__name__}: {exc}"
                    raise BackendProtocolError(f"Stopped at tick {self.env.tick}: {exc}") from exc
                if isinstance(exc, ValueError):
                    # LaundryEnv validates all joint targets before assigning
                    # them. Errors inside its stepping loop set env.fault above.
                    raise ActionRejectedError(str(exc)) from exc
                raise
            return BackendStep(state, dict(requested_ticks=ticks, executed_ticks=self.env.tick-before,
                stopped_on_contact=self.env.tick-before < ticks), state["evaluation"]["score"], state["evaluation"]["success"])

    def policy_access(self):
        return PolicyAccess("dual-panda", (self.camera,) if self.cameras else (),
            {("joints", n, f):None for n in self.env.joints for f in ("position", "velocity")}, self.kinds,
            lambda: self.env.task.spec.label + ". Use the robot arms; leave the garment released and resting on the table.",
            {"dt_s": self.env.config.dt,
             "gripper_control": {"mode": "position", "half_width_range_m": [0., .04],
                 "inward_effort_override": False,
                 "instruction": "Use half_width_m; omit inward_effort_n. Fixed joint effort limits apply."},
             "joints": {n:{**{k:v for k,v in s.items() if k not in ("q", "dof")},
             "unit": "metre" if "finger" in n else "radian"} for n,s in self.env.joints.items()}})

    def act_policy(self, action):
        r = self.act(action)
        return BackendStep(r.state, r.info, None, False, r.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        if name not in ("room", "bench", "cloth", "overhead", "bench_rgb", "bimanual",
                        "handling_detail", "fabric", "edge_wide"):
            raise ValueError("Unknown camera")
        return self._render((name,))[name]

    def get_geometry_index(self):
        return self.catalog.index()

    def get_geometry(self, geometry_id):
        return self.catalog.objects[str(geometry_id)]

    def get_geometry_asset(self, asset_id):
        return self.catalog.assets[str(asset_id)]

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available":False, "reason":"recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def close(self):
        if self.trace:
            self.trace.close()
            self.trace = None
        self.env.close()

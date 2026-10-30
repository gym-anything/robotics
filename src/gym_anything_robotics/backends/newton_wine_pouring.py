"""Wine pouring bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo rigid world, GPU DFSPH liquid,
Isaac RTX rendering of the reconstructed liquid surface).

Render workers follow barcode_scanning's lifecycle: one worker per operation, started while the action's physics runs,
kept for at most renderer_options idle_release_s seconds (default IDLE_RELEASE_S) for the next operation, then released;
close() releases it at once, and a worker watches its owner process.
"""

import math
from collections import deque
from pathlib import Path
import threading
import time
import uuid
import numpy as np

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, Keyboard
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

POLICY_CAMERAS = ("station_rgb", "left_wrist_rgb", "right_wrist_rgb", "bar_side_rgb")
DIAGNOSTIC = ("room", "guest", "side", "closeup")
SESSION_MARKER = ".gar-wine-pouring-session"
SESSION_RETENTION = 20
IDLE_RELEASE_S = math.inf             # the render worker is kept for the life of the session. With 60 s, a policy waiting
                                      # longer between commands pays a new worker inside its next command: 11 s, 1,096x
                                      # wall/sim for a 10 ms command. renderer_options
                                      # {"idle_release_s": s} releases it after s idle seconds instead


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"wine-pouring-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


MIN_POLICY_COMMAND_S = 0.010             # the floor with one 1280x720 camera at 24 samples (see min_command_s)
# observation cost model behind the floor (idle RTX 5090): one 1280x720 RTX frame at 24
# samples per pixel 0.57-0.66 s, liquid surface reconstruction 0.25-1.1 s (more while a stream is in the air), physics
# 46-62x wall/sim on the heaviest catalogue scene (100,246 particles, at rest)
FRAME_S_1280x720_24SPP = 0.66
SURFACE_S = 0.6
PHYSICS_RATIO_MAX = 70.0


def min_command_s(n_cameras, width=1280, height=720, samples=24):
    """Shortest policy command (s) whose wall time, physics plus the observation it returns, stays under 200x wall/sim:
    T * PHYSICS_RATIO_MAX + obs <= 200 T. Rounded up to whole milliseconds; never below MIN_POLICY_COMMAND_S. With three
    cameras a 10 ms command measured 195-214x (the single 10 ms floor did not depend on the camera load)."""
    if not n_cameras:
        return 0.0
    obs = SURFACE_S + n_cameras*FRAME_S_1280x720_24SPP*(width*height/(1280*720))*(samples/24.0)
    return max(MIN_POLICY_COMMAND_S, math.ceil(1000.0*obs/(200.0 - PHYSICS_RATIO_MAX))/1000.0)


class NewtonWinePouringBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset="service", task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from wine_pouring.config import Config, TASKS
        from wine_pouring.env import WineEnv
        from wine_pouring.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["wine-pouring"]
        qualified = tuple(catalog["aliases"])
        if not isinstance(robot, str):
            from wine_pouring.robot_setup import validate
            try:
                robot = validate(robot)          # a gar.robot-setup.v1 object equal to a qualified rig -> its alias
            except ValueError as exc:
                raise UnsupportedConfigurationError(str(exc)) from exc
        if robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"wine-pouring does not support {robot!r}: {catalog['unsupported'][robot]}")
        if robot not in qualified:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
        values = dict(config or {})
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        if device is not None:
            values["device"] = device
        self.config = Config.from_dict(values)
        self.cameras = tuple(cameras if cameras is not None else ("station_rgb",))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; others are privileged diagnostics")
        if robot.startswith("single-") and "right_wrist_rgb" in self.cameras:
            raise UnsupportedConfigurationError(f"{robot} has one arm (left mount): policy cameras are station_rgb, left_wrist_rgb")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("wine-pouring session (auto-created; pruned by retention)\n")
        self.env = WineEnv(self.config, output_dir=self.output)
        self.ik = ArmKinematics(self.env.world)
        self.arms = tuple(self.env.world.rig.instances)
        self.images = deque()
        self.last_timing = {}
        self._renderer = None
        self._prestart = None
        self._retained = None
        self.trace = None
        if recording and self.config.record_trace:
            from wine_pouring.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        mounted = set(self.env.world.scene.cameras)
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            if name not in mounted:
                continue
            fov = self.env.world.scene.cameras[name]["fov"]
            focal = .018/math.tan(math.radians(fov/2))
            mount = "link7_d405_envelope" if "wrist" in name else ("guest_side_bar_level" if name == "bar_side_rgb"
                                                                   else "bar_ceiling_post")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard"})
        self.min_command_s = min_command_s(len(self.cameras), width, height,
                                           self.options.get("samples", self.config.policy_samples))
        self.capabilities = Capabilities(
            "wine-pouring", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": "One tick = 1 ms (MuJoCo rigid step); the liquid advances in CFL substeps inside each tick",
                "control": "Physical position servos slewed at datasheet joint speeds; IK computes targets only",
                "arms": ",".join(self.arms),
                "gripper": "Franka Hand: one synchronized half_width_m position actuator (0..40 mm per finger); either "
                           "imported finger-joint name commands the rack, and unequal named targets project to their "
                           "mean. inward_effort_n grasps with <= 70 N total (50 mm/s until blocked, then force)",
                "liquid": "GPU DFSPH, 2 mm particles (8 uL each), two-way coupled to every container; volumes in the "
                          "evaluation come from the particle inventory",
                "keyboard": "Game-style chords held for ticks/rate_hz s (<= 10 s): W/S A/D R/F move the TCP along world "
                            "+y/-y -x/+x +z/-z at linear_speed_m_s; I/K J/L U/O rotate it about world x, y, z at "
                            "angular_speed_rad_s; ALT x0.2, SHIFT x6; Q closes, E opens the fingers (P x0.2, T x10); "
                            "H grips in force mode (35 N total; with G 10 N, with B 70 N) until E",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state (glass and wine as absorbing "
                           "dielectrics); no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions",
                "min_command_s": (f"{self.min_command_s:g}: policy commands shorter than this are refused; each returns "
                                  f"{len(self.cameras)} camera observation(s) costing ~"
                                  f"{SURFACE_S + len(self.cameras)*FRAME_S_1280x720_24SPP:.1f} s of render, so a shorter "
                                  "command exceeds 200x wall/sim. Owner commands are not floored (diagnostic stepping)"),
                "checkpoints": "In memory for this session only; a checkpoint restores only into the variation (preset, "
                               "task, seed, robot) it was saved in"})

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        """Reset to the session's task, or - with options {preset, task} and/or a seed - to another catalogue variation:
        the world is rebuilt for it in this process (about the cost of a session start's physics part, 30-45 s, but the
        server, its keys and history stay)."""
        from dataclasses import replace
        from wine_pouring.variants import resolve
        from wine_pouring.env import WineEnv
        from wine_pouring.kinematics import ArmKinematics
        values = dict(options or {})
        if set(values) - {"preset", "task"}:
            raise ActionRejectedError("Reset selectors are preset, task (options) and seed")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        preset = values.get("preset", None if "task" in values else self.config.preset)     # a task alone selects its preset
        task = values.get("task", self.config.task if preset == self.config.preset else None)
        try:
            sel = resolve(preset, task, self.config.seed if seed is None else seed)
        except (ValueError, KeyError) as exc:
            # refused before the world is touched: a pre-execution rejection, so the session stays ready
            # (a plain ValueError would latch a runtime fault in the HTTP service)
            raise ActionRejectedError(str(exc)) from exc
        if (sel["preset"], sel["task"], sel["seed"]) == (self.config.preset, self.config.task, self.config.seed):
            return self.env.reset(), {"continuity": "reset", "variation": sel}
        config = replace(self.config, **sel)
        candidate = WineEnv(config, output_dir=self.output)
        with self.env.lock:
            self._end_operation(final=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
            old, self.env, self.config = self.env, candidate, config
            candidate.generation = old.generation + 1
            self.ik = ArmKinematics(candidate.world)
            self._scene_path = None
            if self.trace:
                path = self.trace.store.path
                self.trace.close()
                from wine_pouring.trace import Trace
                self.trace = Trace(candidate, path)
            self.capabilities.notes["task"] = f"{config.preset}/{config.task} seed {config.seed}"
        del old
        return self.env.get_state(), {"continuity": "explicit_reset", "variation": sel}

    def select_variation(self, request):
        from wine_pouring.variants import resolve
        from wine_pouring.task import instruction
        if set(request) - {"preset", "task", "seed"}:
            raise ValueError("Selectors are preset, task and seed")
        sel = resolve(request.get("preset", None if "task" in request else self.config.preset), request.get("task"),
                      request.get("seed", 0))
        return dict(variation=sel, task=instruction(type(self.config)(**{**self.config.__dict__, **sel})),
                    reset_request=dict(seed=sel["seed"], options={"preset": sel["preset"], "task": sel["task"]}))

    def robot_configuration(self):
        """Shared robot handles, setup id and fingerprint (the world keeps its builder at world.scene.b)."""
        w = self.env.world
        handles = getattr(w.scene.b, "_gar_robots", {})
        if not handles:
            return {}
        for handle in handles.values():
            if handle.model is not w.model:
                handle.bind(w.model)
        result = {"instances": {name: handle.capabilities() for name, handle in handles.items()}}
        setup = getattr(w.scene, "robot_setup", None)
        if setup is not None:
            result.update(setup_id=setup.id, setup_fingerprint=setup.fingerprint, alias=self.config.robot)

        def finite(o):
            if isinstance(o, float):
                return o if math.isfinite(o) else None
            if isinstance(o, dict):
                return {k: finite(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [finite(v) for v in o]
            return o
        return finite(result)

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self.env.evaluator.verify()       # evaluate(), except split-bottle's settled read (task.py)

    # ------------------------------------------------------------------ render workers
    def _scene(self):
        from wine_pouring.render import export
        if getattr(self, "_scene_path", None) is None or not self._scene_path.is_file():
            self._scene_path, self.env.render_meta = export(self.env.world, self.output/"render", seed=self.config.seed)
        return self._scene_path

    def _new_renderer(self):
        from wine_pouring.render import Renderer
        return Renderer(self.env.world, self.output/"render", self._scene(), width=self.options.get("width", 1280),
                        height=self.options.get("height", 720), samples=self.options.get("samples", self.config.policy_samples))

    def _start_renderer_early(self):
        if self._prestart is not None or self._renderer is not None or self._retained is not None:
            return
        self._scene()
        holder = {"renderer": None, "error": None, "warm_s": None}

        def work():
            try:
                t0 = time.perf_counter()
                holder["renderer"] = self._new_renderer()
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:
                holder["error"] = exc
        holder["thread"] = threading.Thread(target=work, name="wine-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _operation_renderer(self):
        if self._renderer is not None:
            return self._renderer
        if self._retained is not None:
            r, self._retained = self._retained, None
            self._renderer = r
            self.last_timing["retained_worker"] = True
            return r
        holder, self._prestart = self._prestart, None
        if holder is not None:
            t0 = time.perf_counter()
            holder["thread"].join()
            self.last_timing.update(prestart_wait_s=time.perf_counter() - t0, prestart_warm_s=holder["warm_s"])
            if holder["error"] is not None:
                raise BackendProtocolError(f"Render worker failed to start: {holder['error']}")
            self._renderer = holder["renderer"]
            return self._renderer
        t0 = time.perf_counter()
        self._renderer = self._new_renderer()
        self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return self._renderer

    def _end_operation(self, final=False):
        idle = 0. if final else float(self.options.get("idle_release_s", IDLE_RELEASE_S) or 0.)
        if idle > 0. and self._renderer is not None and self._prestart is None:
            r, self._renderer = self._renderer, None
            old = getattr(self, "_retain_timer", None)
            if old is not None:
                old.cancel()
            self._retained = r
            if math.isfinite(idle):
                timer = threading.Timer(idle, self._release_retained, args=(r,))
                timer.daemon = True
                timer.start()
                self._retain_timer = timer
            return
        retained, self._retained = self._retained, None
        rs = [r for r in (self._renderer, (self._prestart or {}).get("renderer"), retained) if r is not None]
        holder = self._prestart
        self._renderer, self._prestart = None, None

        def reap():
            if holder is not None:
                holder["thread"].join()
                if holder["renderer"] is not None and holder["renderer"] not in rs:
                    rs.append(holder["renderer"])
            for r in rs:
                r.close()
        t = threading.Thread(target=reap, name="wine-render-release", daemon=False)
        t.start()
        self._reapers = [x for x in getattr(self, "_reapers", []) if x.is_alive()] + [t]

    def _release_retained(self, r):
        with self.env.lock:
            if self._retained is not r:
                return
            self._retained = None
        r.close()

    # ------------------------------------------------------------------ observation
    def _render(self, names):
        from wine_pouring.render import liquid_meshes
        with self.env.lock:
            r = self._operation_renderer()
            try:
                t0 = time.perf_counter()
                meshes = liquid_meshes(self.env.world)
                self.last_timing["surface_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                result = r.observe(tuple(names), self.env.world.state.body_q.numpy(), self.env.world.tick, meshes=meshes)
                self.last_timing["frame_s"] = time.perf_counter() - t0
            finally:
                self._end_operation()
            frames = {}
            for name, blob in result.items():
                path = self.output/f"{name}_{self.env.generation}_{self.env.world.tick}_{uuid.uuid4().hex}.png"
                path.write_bytes(blob)
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
            if (state["tick"], state["generation"]) != (self.env.world.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.config.robot,
                               {"joints": state["joints"], "robot_status": self._robot_status()},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

    def _robot_status(self):
        w = self.env.world
        return {"arms": {a: {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4)} for a in self.arms},
                "refused_targets": int(w.refused_targets)}

    # ------------------------------------------------------------------ actions
    def _ticks(self, seconds):
        ticks = seconds/self.config.rigid_dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8, rel_tol=0) or round(ticks) < 1:
            raise ValueError(f"Duration must be a positive multiple of {self.config.rigid_dt} s")
        return int(round(ticks))

    def _arm(self, arm):
        if arm not in self.arms:
            raise UnsupportedActionError(f"Arm must be one of {self.arms}")
        return arm

    def _keyboard(self, action):
        from ..keyboard_control import KEY_TRANSLATION, KEY_ROTATION
        from scipy.spatial.transform import Rotation
        w = self.env.world
        arm = self._arm(action.arm)
        keys = set(action.keys)
        known = set(KEY_TRANSLATION) | set(KEY_ROTATION) | {"Q", "E", "H", "G", "B", "P", "T", "ALT", "SHIFT"}
        if keys - known:
            raise UnsupportedActionError(f"Unsupported keys {sorted(keys - known)}")
        for a, b in (("A", "D"), ("W", "S"), ("R", "F"), ("I", "K"), ("J", "L"), ("U", "O"), ("Q", "E"), ("G", "B"),
                     ("P", "T"), ("ALT", "SHIFT"), ("H", "E")):
            if {a, b} <= keys:
                raise ValueError(f"Conflicting keys {a} and {b}")
        if keys & {"G", "B"} and "H" not in keys:
            raise ValueError("G/B choose the H force-grip level and need H")
        duration = action.ticks/action.rate_hz
        if duration > 10.:
            raise UnsupportedConfigurationError("A keyboard chord lasts at most 10 s; split longer holds")
        ticks = self._ticks(round(duration/self.config.rigid_dt)*self.config.rigid_dt)

        def direction(table):
            v = np.sum([table[k] for k in keys if k in table] or [np.zeros(3)], axis=0).astype(float)
            n = np.linalg.norm(v)
            return v/n if n > 1e-12 else np.zeros(3)
        mult = .2 if "ALT" in keys else (6. if "SHIFT" in keys else 1.)
        move, turn = direction(KEY_TRANSLATION), direction(KEY_ROTATION)
        targets = {}
        if move.any() or turn.any():
            T = self.ik.forward(arm)
            position = T[:3, 3] + move*action.linear_speed_m_s*mult*duration
            R = Rotation.from_rotvec(turn*action.angular_speed_rad_s*mult*duration)*Rotation.from_matrix(T[:3, :3])
            targets.update(self.ik.solve(arm, position, R.as_quat()))
        grip = {}
        if "H" in keys:
            total = 10. if "G" in keys else (70. if "B" in keys else 35.)
            grip[arm] = total/2.
            targets.update(self._fingers(arm, 0.))
        elif keys & {"Q", "E"}:
            grip[arm] = 0.
            q = w.state.joint_q.numpy()
            gap = float(np.mean([q[w.joints[j]["q"]] for j in w.gripper_joints(arm)]))
            rate = .2 if "P" in keys else (10. if "T" in keys else 1.)
            step = (1. if "E" in keys else -1.)*action.gripper_half_speed_m_s*rate*duration
            targets.update(self._fingers(arm, min(.04, max(0., gap + step))))
        return targets, ticks, grip

    def _fingers(self, arm, width):
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 0. <= width <= .04:
            raise ValueError("Franka Hand half gap must be in [0, 0.04] m")
        return {j: float(width) for j in self.env.world.gripper_joints(arm)}

    def _validate_targets(self, targets):
        w = self.env.world
        for name, value in targets.items():
            if name not in w.joints:
                raise ValueError(f"Not a robot actuator: {name}")
            spec = w.joints[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value} (limits {spec['lower']:.4f}..{spec['upper']:.4f})")

    def act(self, action):
        """Plan and validate the whole command first (no side effect; a refused command raises ActionRejectedError and
        leaves the session usable); then apply grip modes and run the physics."""
        with self.env.lock:
            try:
                targets, ticks, grip = self._plan(action)
                self._validate_targets(targets)
                for arm, f in grip.items():
                    limit = float(self.env.world.joints[self.env.world.gripper_joints(arm)[0]]["effort_limit"])
                    if not 0. <= f <= limit:
                        raise ValueError(f"inward_effort_n must be in [0, {2*limit:g}] N (both fingers)")
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise ActionRejectedError(str(exc)) from exc
            return self._execute(targets, ticks, grip)

    def _plan(self, action):
        grip = {}
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
            if type(ticks) is not int or ticks <= 0:
                raise ValueError("ticks must be a positive integer")
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                total = float(action.inward_effort_n)
                if not math.isfinite(total) or total < 0:
                    raise ValueError("inward_effort_n must be a finite non-negative total force (N)")
                targets = self._fingers(arm, 0. if total > 0 else (action.half_width_m if action.half_width_m is not None else .04))
                grip[arm] = total/2.
            else:
                targets = self._fingers(arm, action.half_width_m)
                grip[arm] = 0.
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            arm = self._arm(action.arm)
            if action.safety.stop_on_grip_loss is not None:
                raise UnsupportedActionError("Grip-loss guard is not implemented")
            ticks = self._ticks(action.duration_s)
            if isinstance(action, CartesianPose):
                position = action.position_m
                o = action.orientation_wxyz
                orientation = None if o is None else (*o[1:], o[0])
            else:
                position = self.ik.forward(arm)[:3, 3] + np.asarray(action.delta_m)
                orientation = None
            targets = self.ik.solve(arm, position, orientation)
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        elif isinstance(action, Keyboard):
            targets, ticks, grip = self._keyboard(action)
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, keyboard")
        return targets, ticks, grip

    def _execute(self, targets, ticks, grip):
        w = self.env.world
        for arm, f in grip.items():
            w.set_grip_force(arm, f)
        before = w.tick
        n_reflex, n_refused = len(w.reflexes), w.refused_targets
        self.last_timing = {}
        if self.cameras:
            self._start_renderer_early()
        try:
            t0 = time.perf_counter()
            state = self.env.act(targets, ticks=ticks)
            self.last_timing["physics_s"] = time.perf_counter() - t0
        except BaseException as exc:
            self._end_operation()
            if isinstance(exc, (RuntimeError, FloatingPointError, OSError)):
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            raise
        ev = state["evaluation"]
        return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": w.tick - before,
                                   "reflexes": [dict(r) for r in w.reflexes[n_reflex:]],
                                   "refused_targets": w.refused_targets - n_refused,
                                   "arms_stopped_s": {a: round(u - w.sim_time, 3) for a, u in w._reflex_until.items()
                                                      if u > w.sim_time},
                                   "liquid": state.get("liquid", {})},
                           ev.get("score", 0.), False)

    def policy_access(self):
        from wine_pouring.task import instruction
        from ..keyboard_control import SUPPORTED_KEYS
        proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        proprio.update({("robot_status", "arms", a, "stopped_s"): None for a in self.arms})
        proprio[("robot_status", "refused_targets")] = None
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                            self.kinds, lambda: instruction(self.config),
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()},
                             "gripper": self.capabilities.notes["gripper"],
                             "keyboard": self.capabilities.notes["keyboard"],
                             "protective_stop": f"{'Panda' if 'panda' in self.config.robot else 'FR3'} collision reflex "
                                                "(libfranka default thresholds: external joint torque over 10 N m "
                                                "nominal, 20 N m while moving) and velocity reflex stop an arm; it then "
                                                "refuses joint targets for 1 s (finger targets pass)",
                             "min_command_s": self.min_command_s,
                             # HTTP client timeout gar sessions adopt (sessions.declared_client_timeout; default 120 s):
                             # split-bottle's settled final read holds up to 3 s of simulated time, 75 s of wall time
                             # measured for a 2.3 s hold on an L40S
                             "client_timeout_s": 300},
                            keyboard_keys=tuple(SUPPORTED_KEYS))

    def act_policy(self, action):
        """The policy's command. With cameras, each command is followed by RGB observation(s) costing ~0.6 s per camera
        plus 0.25-1.1 s of surface reconstruction, so a command shorter than self.min_command_s
        (min_command_s: 10 ms with one camera, 15 ms with two, 20 ms with three) could not meet the 200x wall/sim budget
        and is refused
        (send longer commands, or a session with cameras=())."""
        if self.cameras:
            with self.env.lock:
                try:
                    _, ticks, _ = self._plan(action)
                except (ValueError, UnsupportedConfigurationError) as exc:
                    raise ActionRejectedError(str(exc)) from exc
            if ticks*self.config.rigid_dt < self.min_command_s - 1e-12:
                raise ActionRejectedError(f"Policy commands with {len(self.cameras)} camera observation(s) must last >= "
                                          f"{self.min_command_s*1e3:.0f} ms of simulated time ({ticks} ticks requested): "
                                          "each observation's render costs about 1 s per camera, so a shorter command "
                                          "exceeds the 200x wall/sim budget (capabilities notes.min_command_s)")
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        valid = tuple(c for c in POLICY_CAMERAS if c in self.camera_specs) + DIAGNOSTIC
        if name not in valid:
            raise ValueError(f"Render label must be one of {valid}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        """Privileged geometry: every glassware item (exact axisymmetric wall and liquid-space profiles in its body
        frame), the bar top and floor planes, the delivery spots, the task goal and the robot's bodies."""
        from wine_pouring import params as P
        w = self.env.world
        sc = w.scene
        q = w.state.body_q.numpy()
        items = {n: dict(kind=it["kind"], geometry=it["geometry"], body_id=int(it["body"]), role=it.get("role"),
                         liquid=it.get("liquid"), initial_volume_m3=float(it.get("volume") or 0.), spot=it.get("spot"),
                         mass_kg=float(it["mass"]), pose_xyzw=[float(v) for v in q[it["body"]]])
                 for n, it in sc.items.items()}
        return {"items": items,
                "planes": {"bar_top": dict(normal=[0., 0., 1.], height_m=float(sc.liquid_planes[0][1]),
                                           xy_lo=list(sc.liquid_planes[0][2][:2]), xy_hi=list(sc.liquid_planes[0][3])),
                           "floor": dict(normal=[0., 0., 1.], height_m=0.0)},
                "delivery_spots": {str(k): list(v) for k, v in P.DELIVERY_SPOTS.items()},
                "goal": self.env.layout["goal"],
                "robot_bodies": {w.model.body_label[b]: int(b) for b in sc.robot_bodies},
                "ids": "get_geometry(<item name>) returns the item's exact (r, z) wall and liquid-space polygons, fill "
                       "lines, rim and mass properties; geometry assets 'profile/<geometry>' return the polygons only",
                "source": "The simulator's own geometry (wine_pouring/geometry.py), not a camera estimate"}

    def get_geometry(self, geometry_id):
        from wine_pouring import geometry as GEO
        w = self.env.world
        gid = str(geometry_id)
        if gid not in w.scene.items:
            raise KeyError(f"Unknown geometry id {gid!r}; ids are the item names of /v1/geometry")
        it = w.scene.items[gid]
        geo = GEO.get(it["geometry"])
        m, com, inertia = geo.mass_properties()
        return dict(id=gid, kind=it["kind"], geometry=geo.kind, body_id=int(it["body"]),
                    pose_xyzw=[float(v) for v in w.state.body_q.numpy()[it["body"]]],
                    frame="body frame: axis of revolution = +z, base at z = 0; polygons in (r, z) metres",
                    wall_polygon=np.asarray(geo.solid).round(6).tolist(),
                    liquid_space_polygon=np.asarray(geo.interior).round(6).tolist(),
                    rim_z_m=float(geo.rim_z), mouth_inner_radius_m=float(geo.mouth_r),
                    capacity_m3=float(geo.capacity), fill_lines_z_m={k: float(v) for k, v in geo.lines.items()},
                    fill_lines_volume_m3={k: float(geo.volume_below(v)) for k, v in geo.lines.items()},
                    mass_kg=float(m), com_body_m=[float(v) for v in com], wall_material=geo.material,
                    wall_density_kg_m3=float(geo.density), notes=geo.notes)

    def get_geometry_asset(self, asset_id):
        from wine_pouring import geometry as GEO
        aid = str(asset_id)
        if not aid.startswith("profile/"):
            raise KeyError(f"Unknown asset {aid!r}; assets are profile/<geometry>")
        geo = GEO.get(aid.split("/", 1)[1])
        return dict(id=aid, wall_polygon=np.asarray(geo.solid).round(6).tolist(),
                    liquid_space_polygon=np.asarray(geo.interior).round(6).tolist(), units="metre, (r, z) body frame")

    def variation_contract(self):
        from wine_pouring.variants import contract
        return dict(**contract(), current=dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed),
                    reset="POST /v1/variations/select (or reset with options {preset, task} and a seed) rebuilds the "
                          "world for the selection in this session (30-45 s)")

    def _variation_key(self):
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, robot=c.robot)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints = getattr(self, "_checkpoints", {})
            snap = self.env.checkpoint()
            snap["variation"] = self._variation_key()
            self._checkpoints[label] = snap
            return {"label": label, "tick": self.env.world.tick, "generation": self.env.generation,
                    "variation": snap["variation"], "storage": "in_memory_session"}

    def restore_checkpoint(self, label, **kwargs):
        """A checkpoint restores only into the variation it was saved in: after a variation change the world is a
        different scene (other bodies, other liquid), and restoring into it would produce a hybrid state (for example
        decant-carafe's carafe at pour-glass's glass position holding pour-glass's wine)."""
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            if snap.get("variation") != self._variation_key():
                # refused before the world is touched (ActionRejectedError: the session stays ready)
                raise ActionRejectedError(f"Checkpoint {label!r} was saved in variation {snap.get('variation')}; the "
                                          f"session is now {self._variation_key()}. Reset to that variation first, then "
                                          "restore")
            self.env.restore(snap)
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.tick, "generation": self.env.generation,
                                          "continuity": "restored_discontinuity", "storage": "in_memory_session"}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["tick"], "variation": v.get("variation"),
                      "restorable": v.get("variation") == self._variation_key()}
                     for k, v in getattr(self, "_checkpoints", {}).items())

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def close(self):
        with self.env.lock:
            t = getattr(self, "_retain_timer", None)
            if t is not None:
                t.cancel()
            self._end_operation(final=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
        if self.trace:
            self.trace.close()
            self.trace = None

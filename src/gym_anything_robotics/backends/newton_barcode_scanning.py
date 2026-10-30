"""Barcode scanning bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + Isaac RTX imager rendering).

Render workers: one per operation.
An action's worker is started while its physics runs; the same worker renders the scanner imager's frames
(which the firmware takes only while the trigger is pulled or the scanner is in its cradle and something moves
in front of it) and then the action's policy observation, and is released right after.
"""

from collections import deque
import math
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

POLICY_CAMERAS = ("cart_rgb", "left_wrist_rgb", "right_wrist_rgb")
G1_POLICY_CAMERAS = ("head_rgb", "cart_rgb")       # the G1's head D435 (policy default) and the cart mast camera
DIAGNOSTIC = ("room", "front", "tray", "slot", "side", "imager")
SESSION_MARKER = ".gar-barcode-session"
SESSION_RETENTION = 20


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"barcode-scanning-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex



# seconds an operation's render worker is kept for the next operation before a timer releases it (bounded retention;
# 0 = release after every operation).
IDLE_RELEASE_S = 10.0

# what an action costs in wall time (the shared benchmark runner's default 120 s client timeout would end a scanning
# policy at its second action)
WALL_TIME_NOTE = ("An action that keeps a hand or an item in front of the docked scanner (or a handheld trigger session) renders "
                  "the imager's 30 Hz frames, each a path-traced 640 x 480 frame: a simulated second of that has taken "
                  "about 75 s of wall time on a quiet GPU and several minutes on a GPU shared with other jobs. Give HTTP "
                  "clients a timeout of at least 600 s per request. Actions without scanner frames have run at 3-60x "
                  "wall/sim with the default render-worker retention.")

# robot / task pairs the layouts cannot serve: the selected robot provably cannot complete the task as built (reach).
# Refused at launch rather than offered.
UNREACHABLE_TASKS = {
    ("single-fr3", "checkout-present"): "from its left mount the one arm cannot reach the bagging tote",
    ("single-fr3", "checkout-bulky"): "from its left mount the one arm cannot reach the bagging tote",
    ("single-fr3", "checkout-counter"): "from its left mount the one arm cannot reach the bagging tote",
    ("single-fr3", "library-checkin"): "from its left mount the one arm cannot reach the returns bin",
    ("g1", "checkout-bulky"): "standing at the counter the G1 cannot reach the 24-pack at the counter's far end (181 mm short)",
}

class NewtonBarcodeScanningBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset="stockroom", task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from barcode_scanning.config import Config, TASKS
        from barcode_scanning.env import ScanEnv
        from barcode_scanning.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["barcode-scanning"]
        qualified = tuple(catalog["aliases"])
        if isinstance(robot, dict):
            raise UnsupportedConfigurationError(f"barcode-scanning supports catalog aliases {', '.join(qualified)}")
        if robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"barcode-scanning does not support {robot!r}: {catalog['unsupported'][robot]}")
        if robot not in qualified:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
        values = dict(config or {})
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        why = UNREACHABLE_TASKS.get((robot, values["task"]))
        if why:
            raise UnsupportedConfigurationError(f"barcode-scanning task {values['task']!r} is not offered for {robot!r}: {why}")
        if device is not None:
            values["device"] = device
        self.config = Config.from_dict(values)
        self.g1 = robot == "g1"
        self.policy_cameras = G1_POLICY_CAMERAS if self.g1 else POLICY_CAMERAS
        self.cameras = tuple(cameras if cameras is not None else (("head_rgb",) if self.g1 else ("cart_rgb",)))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(self.policy_cameras):
            raise ValueError(f"Policy cameras are {self.policy_cameras}; others are privileged diagnostics")
        if robot == "single-fr3" and "right_wrist_rgb" in self.cameras:
            raise UnsupportedConfigurationError("single-fr3 has one arm (left mount): policy cameras are cart_rgb, left_wrist_rgb")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("barcode-scanning session (auto-created; pruned by retention)\n")
        self.env = ScanEnv(self.config, output_dir=self.output)
        self.env.renderer_factory = self._operation_renderer
        if self.g1:
            from barcode_scanning.g1 import G1HandIK
            self.ik = G1HandIK(self.env.world)
            self.arms = G1HandIK.ARMS
        else:
            self.ik = ArmKinematics(self.env.world)
            self.arms = tuple(self.env.world.rig.instances)
        self.images = deque()
        self.last_timing = {}
        self._renderer = None
        self._prestart = None
        self.trace = None
        if recording and self.config.record_trace:
            from barcode_scanning.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        mounted = set(self.env.world.scene.cameras)
        self.camera_specs = {}
        for name in self.policy_cameras:
            if ("wrist" in name or name == "head_rgb") and name not in mounted:
                continue
            fov = 87. if "wrist" in name else 69.
            focal = .018/math.tan(math.radians(fov/2))
            mount = "link7_d405_envelope" if "wrist" in name else ("g1_urdf_d435_joint" if name == "head_rgb" else "cart_mast_mount")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        if self.g1:
            self.kinds = frozenset({"named_joint_target", "cartesian_pose", "cartesian_delta"})
        else:
            self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard"})
        notes = self._g1_notes() if self.g1 else None
        self.capabilities = Capabilities(
            "barcode-scanning", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes=notes or {
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": "One tick = 1 ms (MuJoCo rigid step)",
                "control": "Physical position servos slewed at datasheet joint speeds; IK computes targets only",
                "arms": ",".join(self.arms),
                "gripper": "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode "
                           f"grasp (the two fingers' pushes summed, at most {self._grip_max_total(self.arms[0]):g} N; "
                           "50 mm/s until blocked, then force; both fingers are coupled, as on the real hand)",
                "scanner": "Handheld 1D/2D imager (DS2278 class): physical trigger (spring + microswitch), cradle seat "
                           "sensor, thin-lens imager rendered by Isaac RTX, sensor model, zxing-cpp decoder; reads only "
                           "come from decoded frames",
                "keyboard": "Game-style chords held for ticks/rate_hz s (<= 10 s): W/S A/D R/F move the TCP along world "
                            "+y/-y -x/+x +z/-z at linear_speed_m_s; I/K J/L U/O rotate it about world x, y, z at "
                            "angular_speed_rad_s; ALT x0.2, SHIFT x6; Q closes, E opens the fingers (P x0.2, T x10); "
                            f"H grips in force mode (35 N total; with G 10 N, with B {self._grip_max_total(self.arms[0]):g} N) until E. Targets come from "
                            "the same IK as cartesian_pose (unreachable: refused, no motion)",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
                "wall_time": WALL_TIME_NOTE,
                "qualification": "Assisted fixtures are not policy actions"})

    def _g1_notes(self):
        return {"task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": f"One tick = {self.config.rigid_dt*1e3:g} ms (MuJoCo rigid step of the G1 configuration)",
                "control": "43 joint position servos (robots/models/g1 'barcode_scanning' profile: the whiteboard gains, copied; URDF effort limits, targets "
                           "slewed at the URDF motor speeds) plus a model gravity feed-forward on the waist and arm joints; "
                           "no balance controller and no support: the floating G1 stands on its servos",
                "arms": "left,right (Cartesian commands move one 7-joint arm; its TCP is 80 mm along the wrist-yaw link, "
                        "the whiteboard HandIK convention); IK computes targets only and a Cartesian command is streamed "
                        "along a quintic ramp over its duration",
                "hands": "Dex3-1: 7 joints per hand commanded by name (g1/<side>_hand_<joint>_joint, radians); there is no "
                         "parallel-jaw gripper action and no grip-force mode",
                "scanner": "Handheld 1D/2D imager (DS2278 class): physical trigger (spring + microswitch), cradle seat "
                           "sensor, thin-lens imager rendered by Isaac RTX, sensor model, zxing-cpp decoder; reads only "
                           "come from decoded frames",
                "cameras": "head_rgb: the G1 head D435 mount (URDF d435_joint), 69 deg HFOV, ideal pinhole; cart_rgb: the "
                           "cart mast camera",
                "protective_stop": "none modelled: the G1 low-level joint interface has no collision or velocity reflex; "
                                   "robot_status.imu reports the pelvis attitude",
                "wall_time": WALL_TIME_NOTE,
                "qualification": "Assisted fixtures are not policy actions"}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.config.seed):
            raise ValueError("Select preset/task/seed when creating the session; reset restores that initial state")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self.env.evaluator.evaluate()

    # ------------------------------------------------------------------ render workers (one per operation)
    def _scene(self):
        from barcode_scanning.render import export
        key = (id(self.env.world), self.env.generation)
        if getattr(self, "_scene_key", None) != key or not self._scene_path.is_file():
            if getattr(self, "_scene_key", (None,))[0] != id(self.env.world) or not getattr(self, "_scene_path", None):
                self._scene_path, self.env.render_meta = export(self.env.world, self.output/"render")
            self._scene_key = key
        return self._scene_path

    def _new_renderer(self):
        from barcode_scanning.render import Renderer
        return Renderer(self.env.world, self.output/"render", self._scene(), width=self.options.get("width", 1280),
                        height=self.options.get("height", 720), samples=self.options.get("samples", self.config.policy_samples),
                        scan_samples=self.options.get("scan_samples", self.config.render_samples))

    def _start_renderer_early(self):
        if self._prestart is not None or self._renderer is not None or getattr(self, "_retained", None) is not None:
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
        holder["thread"] = threading.Thread(target=work, name="barcode-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _operation_renderer(self):
        """The worker of the current operation (early-started one, else a new one); kept until the operation ends."""
        if self._renderer is not None:
            return self._renderer
        if getattr(self, "_retained", None) is not None:
            # bounded idle retention (renderer_options idle_release_s): the previous operation's worker, still warm,
            # if it was opened on the scene exported now
            r, self._retained = self._retained, None
            if getattr(self, "_retained_scene", None) == self._scene():
                self._renderer = r
                self.last_timing["retained_worker"] = True
                return self._renderer
            threading.Thread(target=r.close, name="barcode-render-release", daemon=False).start()
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
        """Release this operation's worker (and an unused prestart) off the reply's path. The worker is kept for
        renderer_options idle_release_s seconds (default IDLE_RELEASE_S) for the next operation and then released by a
        timer; close() releases it at once, and a worker watches its owner process, so a crash cannot orphan it. 0
        releases it after every operation (no worker may be kept indefinitely after an observation). A fresh worker and
        its first frame cost 15-21 s, which alone put 0.1 s actions over the 200x wall/sim limit."""
        idle = 0. if final else float(self.options.get("idle_release_s", IDLE_RELEASE_S) or 0.)
        if idle > 0. and self._renderer is not None and self._prestart is None:
            r, self._renderer = self._renderer, None
            old = getattr(self, "_retain_timer", None)
            if old is not None:
                old.cancel()
            self._retained, self._retained_scene = r, getattr(self, "_scene_path", None)
            timer = threading.Timer(idle, self._release_retained, args=(r,))
            timer.daemon = True
            timer.start()
            self._retain_timer = timer
            return
        retained, self._retained = getattr(self, "_retained", None), None
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
        t = threading.Thread(target=reap, name="barcode-render-release", daemon=False)
        t.start()
        self._reapers = [x for x in getattr(self, "_reapers", []) if x.is_alive()] + [t]

    def _release_retained(self, r):
        with self.env.lock:
            if getattr(self, "_retained", None) is not r:
                return                                     # taken by a new operation (or already released)
            self._retained = None
        r.close()

    # ------------------------------------------------------------------ observation
    def _dynamic_view(self):
        """Light/visibility/texture state of the scanner and display at the current tick."""
        from barcode_scanning import params as P
        from barcode_scanning.device import HostApp
        env = self.env
        fw, meta = env.firmware, env.render_meta
        t = env.world.sim_time
        lights = {p: fw.led_intensity(t) for p in meta["scanner"]["leds"]}
        lights[meta["scanner"]["aim"]] = P.AIM_INTENSITY if fw.aim_on(t) else 0.
        vis = {meta["scanner"]["led_on"]: bool(fw.good_read_led(t))}
        # one file per distinct screen content: an unchanged screen keeps the same texture path, so a warm worker does
        # not reload it (a new file per tick cost ~2.5 s per frame on a retained worker)
        import hashlib
        import json as _json
        digest = hashlib.sha256(_json.dumps(env.host.screen, sort_keys=True, default=str).encode()).hexdigest()[:16]
        shot = self.output/"render"/"art"/f"display_{digest}.png"
        if not shot.is_file():
            HostApp.draw_screen(env.host.screen, shot)
        return lights, vis, {meta["display"]["texture_attr"]: str(shot)}

    def _render(self, names):
        with self.env.lock:
            r = self._operation_renderer()
            try:
                lights, vis, tex = self._dynamic_view()
                t0 = time.perf_counter()
                result = r.observe(tuple(names), self.env.world.state.body_q.numpy(), self.env.world.tick,
                                   lights=lights, visibility=vis, textures=tex)
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
        """What the arms' own controllers report (a real FR3 reports its reflex state): per arm, how long its
        protective stop still holds (0 when it accepts commands), and how many joint targets were refused so far.
        G1: its pelvis IMU attitude (roll, pitch, yaw, tilt from vertical) and the refused-target count."""
        w = self.env.world
        if self.g1:
            from scipy.spatial.transform import Rotation
            q = w.state.body_q.numpy()[w.scene.g1_hw["pelvis"]]
            R = Rotation.from_quat(q[3:])
            tilt = math.degrees(math.acos(max(-1., min(1., float(R.apply((0., 0., 1.))[2])))))
            return {"imu": {"rpy_rad": [round(float(x), 5) for x in R.as_euler("xyz")], "tilt_deg": round(tilt, 3)},
                    "refused_targets": int(w.refused_targets)}
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
        """One held chord -> one Cartesian target (world-frame translation and rotation, the shared key map of
        keyboard_control) and finger targets; the physics then runs for the chord's duration."""
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
            top = self._grip_max_total(arm)
            total = 10. if "G" in keys else (top if "B" in keys else min(35., top))
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

    def _grip_max_total(self, arm):
        """The largest force-mode grip (N, both fingers' pushes summed): twice the finger joint's effort limit in the
        robot's controller profile (FR3 35 N and Panda 20 N per finger), which World.set_grip_force enforces."""
        w = self.env.world
        return 2.*float(w.joints[w.gripper_joints(arm)[0]]["effort_limit"])

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
        if not self.g1:
            # the Franka Hand's fingers are one mechanism (one drive, mirrored): both get the same target, or use gripper
            for arm in self.arms:
                vals = {float(targets[j]) for j in w.gripper_joints(arm) if j in targets}
                if len(vals) > 1:
                    raise ValueError(f"The {arm} Franka Hand's two fingers are one mechanism: give both the same target "
                                     "(or use the gripper action)")

    def act(self, action):
        """Plan and validate the whole command first, with no side effect; a refused command raises
        ActionRejectedError (nothing changed, the session stays usable). Then apply grip modes and run the physics."""
        with self.env.lock:
            try:
                targets, ticks, grip = self._plan(action)
                self._validate_targets(targets)
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise ActionRejectedError(str(exc)) from exc
            return self._execute(targets, ticks, grip)

    def _plan(self, action):
        """(joint targets, ticks, grip-force changes {arm: per-finger N}) for one command; no side effects."""
        w = self.env.world
        grip = {}
        self._ramp = False
        if self.g1:
            return self._plan_g1(action)
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
            if type(ticks) is not int or ticks <= 0:
                raise ValueError("ticks must be a positive integer")
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                total, top = float(action.inward_effort_n), self._grip_max_total(arm)
                if not math.isfinite(total) or total < 0 or total > top:
                    raise ValueError(f"inward_effort_n must be in [0, {top:g}] N: the two fingers' pushes summed, each at "
                                     f"most its joint's effort limit ({top/2:g} N in this robot's profile)")
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
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, "
                                         "keyboard")
        return targets, ticks, grip

    def _plan_g1(self, action):
        """G1: named joint targets for any of the 43 joints (applied at once), or one arm's TCP pose/delta through the
        bounded 7-joint IK (streamed along a quintic ramp over the action). Everything else is refused before any
        side effect."""
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
            if type(ticks) is not int or ticks <= 0:
                raise ValueError("ticks must be a positive integer")
            return targets, ticks, {}
        if isinstance(action, (CartesianPose, CartesianDelta)):
            arm = action.arm
            if arm not in self.arms:
                raise ValueError(f"Arm must be one of {self.arms}")
            if action.safety.stop_on_grip_loss is not None:
                raise ValueError("Grip-loss guard is not implemented")
            if action.gripper_half_target_m is not None:
                raise ValueError("The G1's Dex3 hand has no parallel-jaw gap; command its finger joints by name")
            ticks = self._ticks(action.duration_s)
            if isinstance(action, CartesianPose):
                o = action.orientation_wxyz
                position, orientation = action.position_m, None if o is None else (*o[1:], o[0])
            else:
                position, orientation = self.ik.pose(arm)[0] + np.asarray(action.delta_m), None
            targets, _ = self.ik.solve(arm, position, orientation)
            self._ramp = True
            return targets, ticks, {}
        if isinstance(action, Gripper):
            raise ValueError("The G1's Dex3 hands have no parallel-jaw gripper action; command the 7 hand joints per side "
                             "by name (named_joint_target)")
        if isinstance(action, Keyboard):
            raise ValueError("Keyboard teleoperation is not qualified for the G1 configuration")
        raise ValueError("Supported for the G1: named_joint_target, cartesian_pose, cartesian_delta")

    def _execute(self, targets, ticks, grip):
        w = self.env.world
        for arm, f in grip.items():
            w.set_grip_force(arm, f)
        before = w.tick
        n_reflex, n_refused = len(w.reflexes), w.refused_targets
        n_frames, n_reads = len(self.env.frames), len(self.env.firmware.reads)
        self.last_timing = {}
        self._start_renderer_early()
        try:
            t0 = time.perf_counter()
            state = self.env.act(targets, ticks=ticks, ramp=bool(getattr(self, "_ramp", False)))
            self.last_timing["physics_and_frames_s"] = time.perf_counter() - t0
        except BaseException as exc:
            self._end_operation()
            if isinstance(exc, (RuntimeError, FloatingPointError, OSError)):
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            raise
        ev = state["evaluation"]
        return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": w.tick - before,
                                   "reflexes": [dict(r) for r in w.reflexes[n_reflex:]],
                                   # joint targets dropped because their arm was in its post-reflex hold, and
                                   # the hold time each stopped arm has left (s)
                                   "refused_targets": w.refused_targets - n_refused,
                                   "arms_stopped_s": {a: round(u - w.sim_time, 3)
                                                      for a, u in w._reflex_until.items() if u > w.sim_time},
                                   "scanner_frames": len(self.env.frames) - n_frames,
                                   "scanner_reads": len(self.env.firmware.reads) - n_reads},
                           ev["score"], False)

    def policy_access(self):
        from barcode_scanning.task import instruction
        from ..keyboard_control import SUPPORTED_KEYS
        if self.g1:
            proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
            proprio[("robot_status", "imu", "rpy_rad")] = 3           # roll, pitch, yaw of the pelvis IMU
            proprio[("robot_status", "imu", "tilt_deg")] = None
            proprio[("robot_status", "refused_targets")] = None
            notes = self.capabilities.notes
            return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                                self.kinds, lambda: instruction(self.config, self.env.evaluator.meta),
                                {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                                 "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                                "unit": "radian"} for n, s in self.env.joints.items()},
                                 "hands": notes["hands"], "control": notes["control"],
                                 "protective_stop": notes["protective_stop"]})
        proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        proprio.update({("robot_status", "arms", a, "stopped_s"): None for a in self.arms})
        proprio[("robot_status", "refused_targets")] = None
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                            self.kinds, lambda: instruction(self.config, self.env.evaluator.meta),
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()},
                             "gripper": self.capabilities.notes["gripper"],
                             "keyboard": self.capabilities.notes["keyboard"],
                             "protective_stop": "FR3 collision reflex (external joint torque over 10 N m nominal, 20 N m "
                                                "while moving) and velocity reflex stop an arm; it then refuses joint "
                                                "targets for 1 s (finger targets pass). proprioception.robot_status reports "
                                                "each arm's remaining stop time and the refused-target count"},
                            keyboard_keys=tuple(SUPPORTED_KEYS))

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        if self.g1:
            pc = tuple(c for c in self.policy_cameras if c in self.camera_specs) + DIAGNOSTIC[:-1] + ("g1_left", "g1_back")
            if name not in pc:
                raise ValueError(f"Render label must be one of {pc}")
        elif name not in POLICY_CAMERAS + DIAGNOSTIC or name == "imager":
            raise ValueError(f"Render label must be one of {POLICY_CAMERAS + DIAGNOSTIC[:-1]}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        sc = self.env.world.scene
        return {"items": {n: {k: v for k, v in i.items() if k not in ("inner",)} for n, i in sc.items.items()},
                "labels": [{k: (v if k != "symbol" else dict(kind=v.kind, data=v.data, x_mm=v.x_mm)) for k, v in L.items()}
                           for L in sc.labels],
                "slots": sc.slots, "task": getattr(sc, "task_meta", {})}

    def get_geometry(self, geometry_id):
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints = getattr(self, "_checkpoints", {})
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.world.tick, "generation": self.env.generation}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            self.env.restore(snap)
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.tick, "generation": self.env.generation,
                                          "continuity": "restored_discontinuity", "storage": "in_memory_session"}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["tick"]} for k, v in getattr(self, "_checkpoints", {}).items())

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

"""Classroom arrangement bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo CPU, Isaac RTX images).

One render worker per session, started on the first image and released after renderer_options idle_release_s
(default 300 s) without a request (the next image then pays a ~6 s cold start plus a ~1.8 s first frame); close() releases it at once and the worker watches its owner process.
Physics never waits for the renderer: an action steps the world and then renders
the action's observation from the final state.
"""

import math
from collections import deque
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, BaseTwist, Keyboard
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

POLICY_CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb", "front_rgb", "rear_rgb", "left_rgb", "right_rgb")
BASE_CAMERAS = ("front_rgb", "rear_rgb", "left_rgb", "right_rgb")
DIAGNOSTIC = ("room", "front", "plan", "chase", "side")
SESSION_MARKER = ".gar-classroom-session"
SESSION_RETENTION = 20
IDLE_RELEASE_S = 300.0
# keyboard: the shared hand map (keyboard_control: W/S/A/D/R/F world translation, I/K/J/L/U/O rotation, Q/E fingers,
# H force grip with G/B levels, ALT slow, SHIFT fast) plus the mobile base: arrow keys drive the base in its own frame
# (UP forward, DOWN back, LEFT/RIGHT sideways), Z/X turn it counter-/clockwise
BASE_KEYS = {"UP": (1., 0., 0.), "DOWN": (-1., 0., 0.), "LEFT": (0., 1., 0.), "RIGHT": (0., -1., 0.), "Z": (0., 0., 1.),
             "X": (0., 0., -1.)}
MAX_ACTION_S = 30.0
MIN_CARTESIAN_S = 0.02     # Cartesian / hand-key commands: IK planning (<= IK_BUDGET_S) must stay inside the 200x wall/sim limit
MIN_ACTION_S = 0.02        # every command, for one 640x360 camera or none: a rendered observation (0.16-1.4 s) and the GAR
                           # history cost per command must stay inside 200x wall/sim (1-tick commands cost
                           # 245-1354x); scaled with the render load per observation (three cameras or 1280x720 at
                           # 0.02 s cost up to 205x), see _min_action_s
IK_BUDGET_S = 0.8         # policy-path IK search budget (kinematics.ArmKinematics.solve)


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"classroom-arrangement-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def _finite(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


class NewtonClassroomArrangementBackend(RoboticsBackend):
    def __init__(self, *, robot="mobile-fr3-duo", preset="room114", task=None, seed=0, glides="nylon", contents=0,
                 reflex_profile="contact_x5", cameras=None, output_dir=None, renderer_options=None, recording=True,
                 device=None, reflex_stops_base=True, policy_samples=None, contract="v1", **unknown):
        from classroom_arrangement.config import Config
        from classroom_arrangement.layout import PRESETS
        from classroom_arrangement.env import ClassroomEnv
        from classroom_arrangement.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["classroom-arrangement"]
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown classroom-arrangement options {sorted(unknown)}; options: robot, preset, task, seed, glides, contents, "
                "reflex_profile, cameras, renderer_options, recording, device, reflex_stops_base, policy_samples, contract")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if policy_samples is not None and (type(policy_samples) is not int or not 1 <= policy_samples <= 256):
            raise UnsupportedConfigurationError("policy_samples must be an integer 1..256")
        if device not in (None, "cpu"):
            raise UnsupportedConfigurationError(f"classroom-arrangement runs MuJoCo physics on the CPU only; device={device!r} "
                                                "is not supported (the renderer picks its own GPU)")
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"classroom-arrangement does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(catalog['aliases'])}")
        extra = {} if policy_samples is None else dict(policy_samples=int(policy_samples))
        self.config = Config(preset=preset, task=task or PRESETS[preset][0], seed=int(seed), robot=robot, glides=glides,
                             contents=int(contents), reflex_profile=reflex_profile, reflex_stops_base=reflex_stops_base,
                             contract=contract, **extra)
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("classroom-arrangement session (auto-created; pruned by retention)\n")
        self.env = ClassroomEnv(self.config, output_dir=self.output)
        w = self.env.world
        self.arms = tuple(w.arms)
        self.ik = ArmKinematics(w)
        self.policy_cameras = tuple(c for c in POLICY_CAMERAS if c == "head_rgb" or c in BASE_CAMERAS or c.split("_")[0] in self.arms)
        self.cameras = tuple(cameras if cameras is not None else ("head_rgb",))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(self.policy_cameras):
            raise ValueError(f"Policy cameras are {self.policy_cameras}; others are privileged diagnostics")
        width, height = self.options.get("width", 640), self.options.get("height", 360)
        self.camera_specs = {}
        for name in self.policy_cameras:
            from classroom_arrangement import params as P
            fov = P.HEAD_HFOV if name == "head_rgb" else (P.BASE_CAMERA_HFOV if name in BASE_CAMERAS else P.WRIST_HFOV)
            focal = .018/math.tan(math.radians(fov/2))
            mount = "franka_head_v0_2 head_camera_mounting_point (ZED Mini class)" if name == "head_rgb" else \
                (f"TMR {name.split('_')[0]}_mounting_point (RealSense D455 RGB)" if name in BASE_CAMERAS else
                 f"{name.split('_')[0]}_fr3_link7 D405 envelope")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.min_action_s = self._min_action_s(len(self.cameras), width, height)
        self.images = deque()
        self._renderer, self._timer, self._scene_path = None, None, None
        self._render_lock = threading.RLock()
        self.last_timing = {}
        self.trace = None
        self.recording = bool(recording)
        if recording:
            from classroom_arrangement.trace import Trace
            self.trace = Trace(w, self.output/"physics.sqlite3")
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "base_twist", "keyboard"})
        from ..keyboard_control import SUPPORTED_KEYS
        self.keyboard_keys = tuple(sorted(set(SUPPORTED_KEYS) | set(BASE_KEYS)))
        self.capabilities = Capabilities(
            "classroom-arrangement", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())

    @staticmethod
    def _min_action_s(n_cameras, width, height):
        """Shortest legal command for this session's observation: 0.02 s for one 640x360 camera (or none), scaled with
        the number of rendered pixels per observation, in whole 0.5 ms ticks."""
        load = max(1., n_cameras*width*height/(640*360))
        return math.ceil(MIN_ACTION_S*load/0.0005 - 1e-9)*0.0005

    def _notes(self):
        from classroom_arrangement import params as P
        lim = P.BASE_LIMITS
        return {
            "min_command_s": f"every command lasts at least {self.min_action_s:g} s ({len(self.cameras)} camera(s) at "
                             f"{self.options.get('width', 640)}x{self.options.get('height', 360)}: 0.02 s per 640x360 frame of "
                             "render load) so that each command stays inside 200x wall/sim",
            "base_sensors": "robot_status.base_sensors: IMU (angular rate, specific force; base frame) and two corner LIDAR "
                            "scans (275 deg, 0.5 deg steps, 0.05-10 m); base RGB cameras front_rgb, rear_rgb, left_rgb, "
                            "right_rgb (D455, 90 deg) are selectable policy cameras",
            "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}, glides {self.config.glides}, "
                    f"contents {self.config.contents}",
            "clock": "One tick = 0.5 ms (MuJoCo rigid step); joint-target actions give ticks, others duration_s",
            "robot": "Franka Mobile FR3 Duo: TMR base (two ArgoDrive swerve units, two casters, rocker), Franka Spine lift, "
                     + ("two FR3 arms" if len(self.arms) == 2 else "one FR3 arm (right)") + " with Franka Hands",
            "base": f"base_twist: v_x, v_y (m/s) and w_z (rad/s) in the base frame, |v| <= {lim['v']}, |w_z| <= {lim['w']}; "
                    f"slewed at {lim['a']} m/s2 and {lim['alpha']} rad/s2 by the swerve controller; drive torque 16 N m "
                    "accelerating / 28.4 N m braking; the twist is held until the next base_twist (send zeros to stop; "
                    "a stopped base holds its wheels). Wheels can slip.",
            "spine": f"named_joint_target robot/franka_spine_vertical_joint, 0..{P.SPINE['stroke']} m at <= {P.SPINE['speed']} m/s",
            "arms": "named_joint_target (radians) or cartesian_pose / cartesian_delta of an arm's hand TCP in the world frame "
                    "(IK computes joint targets; the servos move at the FR3 joint speed limits)",
            "gripper": "Franka Hand: half_width_m position mode (0..0.04 m per finger) or inward_effort_n force mode (total of "
                       "both fingers, <= 140 N, 50 mm/s until blocked, then that force)",
            "protective_stop": f"FR3 collision reflexes ({self.config.reflex_profile} thresholds; joint torque and hand wrench, 10 ms filtered external "
                               "joint torque) and velocity reflex stop an arm; it then refuses joint targets for 1 s",
            "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
            "keyboard": "keyboard chords: W/S/A/D/R/F move the chosen arm's hand along world y/x/z, I/K/J/L/U/O rotate it, Q/E "
                        "close/open the fingers, H grips by force (G 10 N, default 35 N, B 70 N total), ALT slow, SHIFT fast; "
                        "UP/DOWN/LEFT/RIGHT drive the base in its frame (0.3 m/s), Z/X turn it (0.5 rad/s); a chord lasts "
                        "ticks/rate_hz s (<= 10 s), then the base stops",
            "qualification": "Assisted fixtures are not policy actions"}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        """Restore the initial state; with a different variation (options preset/task/glides/contents/contract, or seed: the
        `reset_request` of select_variation) the room is rebuilt for it first (bodies, books and floor dots differ
        between variations)."""
        from classroom_arrangement.variants import resolve
        options = dict(options or {})
        unknown = set(options) - {"preset", "task", "glides", "contents", "seed", "contract"}
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are preset, task, seed, glides, "
                                         "contents, contract; nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            cur = self._variation()
            default_contract = cur.get("contract", "v1") if options.get("task", cur["task"]) == cur["task"] else "v1"
            try:                                      # validated before anything changes: a refusal, not a fault
                want = resolve(options.get("preset", cur["preset"]), options.get("task", cur["task"]),
                               int(options.get("seed", cur["seed"])), options.get("glides", cur["glides"]),
                               int(options.get("contents", cur["contents"])), options.get("contract", default_contract))
            except (ValueError, TypeError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            if want != cur:
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def _variation(self):
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, glides=c.glides, contents=c.contents,
                    **({"contract": c.contract} if c.contract != "v1" else {}))

    def _rebuild(self, v):
        from dataclasses import replace
        from classroom_arrangement.env import ClassroomEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            self._release()
            self.config = replace(self.config, preset=v["preset"], task=v["task"], seed=int(v["seed"]), glides=v["glides"],
                                  contents=int(v["contents"]), contract=v.get("contract", "v1"))
            old = self.env
            self.env = ClassroomEnv(self.config, output_dir=self.output)
            self.ik = ArmKinematics(self.env.world)
            self._scene_path = None
            self._checkpoints = {}
            self.env.generation = old.generation + 1
            from dataclasses import replace as _replace
            self.capabilities = _replace(self.capabilities, notes=self._notes())   # the task note names the new episode
            if self.recording:
                from classroom_arrangement.trace import Trace
                self.trace = Trace(self.env.world, self.output/f"physics-{self.env.generation}.sqlite3")

    def variation_contract(self):
        from classroom_arrangement.variants import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        """Validate a variation and return the reset request that applies it (GAR's variation protocol)."""
        from classroom_arrangement.variants import resolve
        from classroom_arrangement.catalogue import description
        if set(request) - {"preset", "task", "seed", "glides", "contents", "contract"}:
            raise ValueError("Selectors are preset, task, seed, glides, contents and contract")
        cur = self._variation()
        default_contract = cur.get("contract", "v1") if request.get("task", cur["task"]) == cur["task"] else "v1"
        v = resolve(request.get("preset", cur["preset"]), request.get("task", cur["task"]), int(request.get("seed", cur["seed"])),
                    request.get("glides", cur["glides"]), int(request.get("contents", cur["contents"])),
                    request.get("contract", default_contract))
        return dict(variation=v, task=description(v["task"], v.get("contract", "v1")),
                    reset_request=dict(seed=v["seed"], options={k: v[k] for k in ("preset", "task", "glides", "contents")} |
                                       {"contract": v.get("contract", "v1")}))

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            ev = self.env.evaluator.evaluate()
            return dict(ev, success=bool(ev["succeeded"]), score=float(ev["score"]))

    # ------------------------------------------------------------------ rendering (one worker per session, idle release)
    def _scene(self):
        from classroom_arrangement.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env.world, self.output/"render")
        return self._scene_path

    def _worker(self):
        from classroom_arrangement.render import Renderer
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._renderer is None:
            t0 = time.perf_counter()
            self._renderer = Renderer(self.output/"render", self._scene(), width=self.options.get("width", 640),
                                      height=self.options.get("height", 360),
                                      samples=self.options.get("samples", self.config.policy_samples))
            self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return self._renderer

    def _release_later(self):
        idle = float(self.options.get("idle_release_s", IDLE_RELEASE_S) or 0.)
        if idle <= 0:
            self._release()
            return
        self._timer = threading.Timer(idle, self._release)
        self._timer.daemon = True
        self._timer.start()

    def _release(self):
        with self._render_lock:
            r, self._renderer = self._renderer, None
            if r is not None:
                r.close()

    def _render(self, names):
        with self._render_lock:
            r = self._worker()
            try:
                t0 = time.perf_counter()
                result = r.observe(tuple(names), self.env.world.state.body_q.numpy(), self.env.world.tick)
                self.last_timing["frame_s"] = time.perf_counter() - t0
            finally:
                self._release_later()
        frames = {}
        for name, blob in result.items():
            path = self.output/f"{name}_{self.env.generation}_{self.env.world.tick}_{uuid.uuid4().hex[:8]}.png"
            path.write_bytes(blob)
            self.images.append(path)
            frames[name] = path
        while len(self.images) > 64:
            self.images.popleft().unlink(missing_ok=True)
        return frames

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.world.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.robot_name,
                               {"joints": state["joints"], "robot_status": state["robot_status"]},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        allowed = self.policy_cameras + DIAGNOSTIC
        if name not in allowed:
            raise ValueError(f"Render label must be one of {allowed}")
        with self.env.lock:
            return self._render((name,))[name]

    # ------------------------------------------------------------------ actions
    def _ticks(self, seconds):
        from classroom_arrangement import params as P
        lo = getattr(self, "min_action_s", MIN_ACTION_S)
        if not math.isfinite(seconds) or seconds < lo - 1e-12 or seconds > MAX_ACTION_S:
            raise ValueError(f"duration_s must be in [{lo:g}, {MAX_ACTION_S}] s (this session's minimum: {lo:g} s)")
        ticks = seconds/P.DT
        if not math.isclose(ticks, round(ticks), abs_tol=1e-6, rel_tol=0) or round(ticks) < 1:
            raise ValueError(f"Duration must be a positive multiple of {P.DT} s")
        return int(round(ticks))

    def _arm(self, arm):
        if arm not in self.arms:
            raise UnsupportedActionError(f"Arm must be one of {self.arms}")
        return arm

    def _fingers(self, arm, width):
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 0. <= width <= .04:
            raise ValueError("Franka Hand half gap must be in [0, 0.04] m")
        return {j: float(width) for j in self.env.world.gripper_joints(arm)}

    def _plan(self, action):
        """(joint targets, twist or None, ticks, grip-force changes {arm: per-finger N}) without side effects."""
        w = self.env.world
        grip, twist = {}, None
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
            if type(ticks) is not int or ticks < self._ticks(self.min_action_s) or ticks > self._ticks(MAX_ACTION_S):
                raise ValueError(f"ticks must be an integer in [{self._ticks(self.min_action_s)}, {self._ticks(MAX_ACTION_S)}] "
                                 f"(0.5 ms each; at least {self.min_action_s:g} s in this session)")
            base = w.base_joint_names
            for n in targets:
                if n in base:
                    raise ValueError(f"{n} belongs to the base controller; drive the base with base_twist")
        elif isinstance(action, BaseTwist):
            ticks = self._ticks(action.duration_s)
            from classroom_arrangement import params as P
            lim = P.BASE_LIMITS
            if math.hypot(action.vx_m_s, action.vy_m_s) > lim["v"] + 1e-9 or abs(action.wz_rad_s) > lim["w"] + 1e-9:
                raise ValueError(f"base twist exceeds the limits ({lim['v']} m/s, {lim['w']} rad/s)")
            targets, twist = {}, (action.vx_m_s, action.vy_m_s, action.wz_rad_s)
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                total = float(action.inward_effort_n)
                if not math.isfinite(total) or total < 0 or total > 140.:
                    raise ValueError("inward_effort_n must be in [0, 140] N (the Franka Hand's 70 N per finger)")
                targets = self._fingers(arm, 0. if total > 0 else (action.half_width_m if action.half_width_m is not None else .04))
                grip[arm] = total/2.
            else:
                targets = self._fingers(arm, action.half_width_m)
                grip[arm] = 0.
        elif isinstance(action, Keyboard):
            return self._keyboard(action)
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            arm = self._arm(action.arm)
            if action.safety.stop_on_grip_loss is not None:
                raise UnsupportedActionError("Grip-loss guard is not implemented")
            ticks = self._ticks(action.duration_s)
            if isinstance(action, CartesianPose):
                o = action.orientation_wxyz
                position, orientation = action.position_m, None if o is None else (*o[1:], o[0])
            else:
                Tc = self.ik.forward(arm, self.ik.commanded(arm))          # from the commanded pose, not the measured
                q = Rotation.from_matrix(Tc[:3, :3]).as_quat()
                position, orientation = Tc[:3, 3] + np.asarray(action.delta_m), q
            if action.duration_s < MIN_CARTESIAN_S:
                raise ValueError(f"Cartesian commands plan IK (up to {IK_BUDGET_S} s); duration_s must be >= {MIN_CARTESIAN_S} s")
            if orientation is None:
                q = Rotation.from_matrix(self.ik.forward(arm, self.ik.commanded(arm))[:3, :3]).as_quat()
                orientation = q
            targets = self.ik.solve(arm, position, orientation, seed=self.ik.commanded(arm), budget_s=IK_BUDGET_S)
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        else:
            raise UnsupportedActionError("Supported: named_joint_target, base_twist, gripper, cartesian_pose, cartesian_delta")
        for name, value in targets.items():
            spec = w.joints.get(name)
            if spec is None:
                raise ValueError(f"Not a robot actuator: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value} (limits {spec['lower']:.4f}..{spec['upper']:.4f})")
        return targets, twist, ticks, grip

    def _keyboard(self, action):
        """One held chord -> (joint targets, base twist, ticks, grip): the shared hand map plus base keys."""
        from ..keyboard_control import KEY_TRANSLATION, KEY_ROTATION
        from scipy.spatial.transform import Rotation
        w = self.env.world
        arm = self._arm(action.arm)
        keys = set(action.keys)
        known = set(self.keyboard_keys)
        if keys - known:
            raise UnsupportedActionError(f"Unsupported keys {sorted(keys - known)}")
        for a, b in (("A", "D"), ("W", "S"), ("R", "F"), ("I", "K"), ("J", "L"), ("U", "O"), ("Q", "E"), ("G", "B"),
                     ("P", "T"), ("ALT", "SHIFT"), ("H", "E"), ("UP", "DOWN"), ("LEFT", "RIGHT"), ("Z", "X")):
            if {a, b} <= keys:
                raise ValueError(f"Conflicting keys {a} and {b}")
        if keys & {"G", "B"} and "H" not in keys:
            raise ValueError("G/B choose the H force-grip level and need H")
        duration = action.ticks/action.rate_hz
        if duration > 10.:
            raise ValueError("A keyboard chord lasts at most 10 s; split longer holds")
        if duration < self.min_action_s - 1e-12:
            raise ValueError(f"A keyboard chord lasts at least {self.min_action_s:g} s in this session")
        from classroom_arrangement import params as P
        ticks = max(1, int(round(duration/P.DT)))

        def direction(table):
            v = np.sum([table[k] for k in keys if k in table] or [np.zeros(3)], axis=0).astype(float)
            n = np.linalg.norm(v)
            return v/n if n > 1e-12 else np.zeros(3)
        mult = .2 if "ALT" in keys else (6. if "SHIFT" in keys else 1.)
        move, turn = direction(KEY_TRANSLATION), direction(KEY_ROTATION)
        targets, grip = {}, {}
        if move.any() or turn.any():
            T = self.ik.forward(arm, self.ik.commanded(arm))                  # commanded pose (see CartesianDelta)
            position = T[:3, 3] + move*action.linear_speed_m_s*mult*duration
            R = Rotation.from_rotvec(turn*action.angular_speed_rad_s*mult*duration)*Rotation.from_matrix(T[:3, :3])
            if duration < MIN_CARTESIAN_S:
                raise ValueError(f"Hand keys plan IK (up to {IK_BUDGET_S} s); duration_s must be >= {MIN_CARTESIAN_S} s")
            targets.update(self.ik.solve(arm, position, R.as_quat(), seed=self.ik.commanded(arm), budget_s=IK_BUDGET_S))
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
        base = np.sum([BASE_KEYS[k] for k in keys if k in BASE_KEYS] or [np.zeros(3)], axis=0)
        bmult = .33 if "ALT" in keys else (2.6 if "SHIFT" in keys else 1.)
        twist = (0.3*bmult*base[0], 0.3*bmult*base[1], 0.5*bmult*base[2]) if base.any() else (0., 0., 0.)
        lim = P.BASE_LIMITS
        if np.hypot(twist[0], twist[1]) > lim["v"] + 1e-9 or abs(twist[2]) > lim["w"] + 1e-9:
            raise ValueError("base keys with SHIFT exceed the base limits")
        return targets, twist, ticks, grip

    def act(self, action):
        """Plan and validate the whole command first, with no side effect (a refused command raises
        ActionRejectedError and nothing changes), then run the physics."""
        with self.env.lock:
            try:
                targets, twist, ticks, grip = self._plan(action)
                if twist is not None and any(abs(v) > 0 for v in twist) and self.env.world.base_held():
                    raise ValueError("Base command refused: an arm protective stop holds the base for 1 s; "
                                     "no command executed")
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise ActionRejectedError(str(exc)) from exc
            if self._timer is not None:          # the render worker must not be released while this action's physics runs
                self._timer.cancel()
                self._timer = None
            w = self.env.world
            before, n_reflex, n_refused = w.tick, len(w.reflexes), w.refused_targets
            t0 = time.perf_counter()
            try:
                state = self.env.act(targets, twist=twist, ticks=ticks, grip=grip)
                if isinstance(action, Keyboard) and twist is not None and any(twist):
                    w.base.set_twist(0., 0., 0.)          # a released base key stops the base (it then decelerates)
            except (RuntimeError, FloatingPointError, OSError) as exc:
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            self.last_timing["physics_s"] = time.perf_counter() - t0
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": w.tick - before,
                                       "reflexes": [dict(r) for r in w.reflexes[n_reflex:]],
                                       "refused_targets": w.refused_targets - n_refused,
                                       "arms_stopped_s": {a: round(u - w.sim_time, 3) for a, u in w._reflex_until.items()
                                                          if u > w.sim_time},
                                       "base_twist_commanded": [round(float(v), 4) for v in w.base.request]},
                               ev["score"], False)

    def policy_access(self):
        from classroom_arrangement.task import instruction
        w = self.env.world
        proprio = {("joints", n, f): None for n in w.joints for f in ("position", "velocity")}
        proprio[("robot_status", "base", "odometry")] = 3
        proprio[("robot_status", "base", "twist_measured")] = 3
        proprio[("robot_status", "base", "twist_commanded")] = 3
        for a in self.arms:
            proprio[("robot_status", "arms", a, "stopped_s")] = None
            proprio[("robot_status", "arms", a, "external_torque_nm")] = 7
            proprio[("robot_status", "arms", a, "external_wrench_hand")] = 6
            proprio[("robot_status", "arms", a, "grip_force_n")] = None
        proprio[("robot_status", "refused_targets")] = None
        notes = self.capabilities.notes
        joints = {}
        for n, s in w.joints.items():
            joints[n] = {"lower": _finite(float(s["lower"])), "upper": _finite(float(s["upper"])),
                         "velocity_limit": _finite(float(s.get("velocity_limit", float("inf")))),
                         "effort_limit": float(s["effort_limit"]), "unit": s.get("unit", "rad"),
                         "commandable": n not in w.base_joint_names}
        return PolicyAccess(self.robot_name, tuple(self.camera_specs[c] for c in self.cameras), proprio, self.kinds,
                            lambda: instruction(self.config, self.env.spec),
                            {"dt_s": 0.0005, "arms": list(self.arms), "joints": joints, "base": notes["base"],
                             "spine": notes["spine"], "gripper": notes["gripper"], "control": notes["arms"],
                             "protective_stop": notes["protective_stop"]})

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    QUALIFIED_MODELS = ("mobile-fr3-duo@1", "mobile-fr3-single@1")

    @classmethod
    def _qualified_setup(cls, robot, catalog):
        """Accept a gar.robot-setup.v1 object (or RobotSetup) when it describes a qualified configuration: one
        instance named "robot", model mobile-fr3-duo@1 or mobile-fr3-single@1, mount floor_start, controller profile
        "classroom", no attachments. Anything else is refused with the reason (unified contract: no substitution)."""
        from robots.catalog import resolve
        try:
            setup = resolve(robot)
        except (TypeError, ValueError) as exc:
            raise UnsupportedConfigurationError(f"Invalid robot setup: {exc}") from exc
        d = setup.to_dict()
        inst = d["instances"]
        problems = []
        if list(inst) != ["robot"]:
            problems.append("exactly one instance named 'robot'")
        for i in inst.values():
            if i["model"] not in cls.QUALIFIED_MODELS:
                problems.append(f"model in {cls.QUALIFIED_MODELS} (got {i['model']})")
            if i.get("mount") != "floor_start":
                problems.append("mount 'floor_start'")
            if i.get("controller_profile") != "classroom":
                problems.append("controller profile 'classroom' (the only qualified one)")
        if d.get("attachments"):
            problems.append("no attachments")
        if problems:
            raise UnsupportedConfigurationError("classroom-arrangement qualifies only the Mobile FR3 Duo / single-arm "
                                                f"configurations; this setup needs: {'; '.join(problems)}. Aliases: "
                                                f"{', '.join(catalog['aliases'])}")
        return d

    # ------------------------------------------------------------------ robot configuration, geometry, trace
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
            result.update(setup_id=setup.id, setup_fingerprint=setup.fingerprint)

        def finite(o):           # continuous joints (wheels, steering, casters) have infinite limits: JSON null
            if isinstance(o, float):
                return o if math.isfinite(o) else None
            if isinstance(o, dict):
                return {k: finite(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [finite(v) for v in o]
            return o
        return finite(result)

    def _catalog(self):
        """Exact simulator geometry (pencap's rigid catalogue over every Newton shape and body), built on first use."""
        if getattr(self, "_geometry", None) is None:
            import newton
            from types import SimpleNamespace
            from pencap.newton15.catalog import Catalog
            w = self.env.world
            self._geometry = Catalog(SimpleNamespace(builder=w.scene.b, model=w.model, newton=newton))
        return self._geometry

    def get_geometry_index(self):
        """Privileged geometry: every shape (id = Newton shape index) and body with mass/COM/inertia, plus the
        furniture items with their part lists and current poses, and the task's targets, floor dots and checks."""
        spec, w = self.env.spec, self.env.world
        q = w.state.body_q.numpy()
        index = dict(self._catalog().index())
        index.pop("pen_frame", None)
        index.update(
            items={n: dict(kind=it["kind"], body_id=int(it["body"]), start_pose=list(spec["items"][n]["pose"])
                           if n in spec["items"] else None, pose_xyzw=[float(v) for v in q[it["body"]]],
                           shape_ids=[str(s) for s in it["shapes"]]) for n, it in w.scene.items.items()},
            targets={n: list(v) for n, v in spec["targets"].items()},
            floor_dots={n: [list(p) for p in pts] for n, pts in spec["dots"].items()},
            checks={n: (v if isinstance(v, str) else list(map(str, v))) for n, v in spec["check"].items()},
            item_ids="get_geometry(<item name>) returns the item's authored parts (furniture.Part) and current pose",
            source="Actual model geometry, masses and friction; not a camera-derived estimate")
        return index

    def get_geometry(self, geometry_id):
        gid = str(geometry_id)
        w = self.env.world
        if gid in w.scene.items:
            it = w.scene.items[gid]
            q = w.state.body_q.numpy()[it["body"]]
            parts = [dict(name=p.name, kind=p.kind, size=list(p.size), pos=list(p.pos), quat_xyzw=list(p.quat),
                          mass_kg=float(p.mass), material=p.material, role=p.role, collides=bool(p.collide),
                          friction=p.mu) for p in w.scene.parts[gid]]
            return dict(id=gid, kind=it["kind"], body_id=int(it["body"]), pose_xyzw=[float(v) for v in q],
                        mass_kg=float(w.model.body_mass.numpy()[it["body"]]), parts=parts,
                        size_convention="box: half extents; capsule/cylinder: radius, half height (local z)")
        return self._catalog().objects[gid]

    def get_geometry_asset(self, asset_id):
        return self._catalog().assets[str(asset_id)]

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

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

    def close(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None
        if self._timer is not None:
            self._timer.cancel()
        self._release()

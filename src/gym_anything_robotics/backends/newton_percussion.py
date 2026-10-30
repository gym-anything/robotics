"""Percussion bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo rigid world, audio-rate modal
instruments with implicit vibro-contact, simulated room acoustics, Isaac RTX rendering).

Observations: joint state, robot status, the station's cue light, the station microphone (the robot's own measurement
microphone on the station mast, or the G1 head array: the last 0.5 s of pressure at 16 kHz plus L_AF / L_Aeq / peak) and RGB from the
policy cameras. The client's-ear microphone, contact segments and modal states are privileged (state/evaluation).
Render workers: one worker per session, kept IDLE_RELEASE_S after its last observation (renderer_options.idle_release_s)
and then released (bounded retention, as classroom_arrangement and painting); the
next observation starts a fresh worker. The minimum policy command is fixed per session (min_command_s).
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

POLICY_CAMERAS = ("station_rgb", "left_wrist_rgb", "right_wrist_rgb", "head_rgb")
DIAGNOSTIC = ("room", "client", "side", "drum_close", "xylo_close", "resbar_close", "rack_close", "shelf", "drum_stand")
SESSION_MARKER = ".gar-percussion-session"
SESSION_RETENTION = 20
IDLE_RELEASE_S = 300.0                 # s a render worker is kept after its last observation
AUDIO_OBS_S = 0.5                      # station-microphone window in each observation
AUDIO_OBS_FS = 16000                   # decimated 48 kHz -> 16 kHz (Nyquist 8 kHz: every bar's third partial)
AUDIO_OBS_N = int(AUDIO_OBS_S*AUDIO_OBS_FS)
MIN_POLICY_COMMAND_S = 0.010
FRAME_S_1280x720_24SPP = 0.66          # one RTX frame (wine_pouring measurement on the same GPU class)
PHYSICS_RATIO_START = 60.0             # reference physics wall/sim the session's fixed minimum command assumes
BUDGET = 200.0                         # wall time / simulated time (a speed target; info.timing.over_budget)


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"percussion-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def min_command_s(n_cameras, width=1280, height=720, samples=24):
    """The session's minimum policy command (s), fixed when it starts: the shortest command whose wall time - physics
    plus the observation it returns - stays under 200x wall/sim with the reference costs (physics PHYSICS_RATIO_START,
    a frame FRAME_S_1280x720_24SPP scaled by pixels and samples): T * PHYSICS_RATIO_START + obs <= 200 T, rounded up to
    whole milliseconds, never below 10 ms. It depends on the cameras' count, resolution and samples, not on this host's
    measured speed (min_command_now)."""
    if not n_cameras:
        return MIN_POLICY_COMMAND_S                     # 10 ms without cameras too (the history's disk tail)
    obs = n_cameras*FRAME_S_1280x720_24SPP*(width*height/(1280*720))*(samples/24.0)
    return max(MIN_POLICY_COMMAND_S, math.ceil(1000.0*obs/(BUDGET - PHYSICS_RATIO_START))/1000.0)


class NewtonPercussionBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset="therapy", task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from percussion.config import Config, TASKS
        from percussion.env import PercussionEnv
        from percussion.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["percussion"]
        qualified = tuple(catalog["aliases"])
        if not isinstance(robot, str) or robot not in qualified:
            from percussion.robot_setup import validate
            try:
                robot = validate(robot)
            except ValueError as exc:
                raise UnsupportedConfigurationError(str(exc)) from exc
        values = dict(config or {})
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        if device is not None:
            values["device"] = device
        try:
            self.config = Config.from_dict(values)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        g1 = robot == "g1"
        allowed = ("head_rgb",) if g1 else POLICY_CAMERAS[:3]
        self.cameras = tuple(cameras if cameras is not None else (allowed[0],))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(allowed):
            raise ValueError(f"Policy cameras of {robot} are {allowed}; others are privileged diagnostics")
        if robot.startswith("single-") and "left_wrist_rgb" in self.cameras:  # noqa: E501
            raise UnsupportedConfigurationError(f"{robot} has one arm (right mount): policy cameras are station_rgb, "
                                                "right_wrist_rgb")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("percussion session (auto-created; pruned by retention)\n")
        self.env = PercussionEnv(self.config, output_dir=self.output)
        self.g1 = g1
        if g1:
            from percussion.g1 import G1Kinematics
            self.ik = G1Kinematics(self.env.world)
            self.arms = ("left", "right")
        else:
            self.ik = ArmKinematics(self.env.world)
            self.arms = tuple(self.env.world.rig.instances)
        self.images = deque()
        self.last_timing = {}
        self._renderer = None
        self._prestart = None
        self._retained = None
        self.trace = None
        if recording and self.config.record_trace:
            from percussion.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        mounted = set(self.env.world.scene.cameras)
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            if name not in mounted:
                continue
            fov = self.env.world.scene.cameras[name]["fov"]
            focal = .018/math.tan(math.radians(fov/2))
            mount = "link7_d405_envelope" if "wrist" in name else ("g1_torso_d435" if name == "head_rgb" else
                                                                   "station_mast_d435_class")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "cartesian_pose", "cartesian_delta"} if g1 else
                               {"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard"})
        self._frame_s_seen = 0.0                # slowest recent policy render (s), decaying (reported)
        self._cmd_t0 = None
        self._cmd = None
        self.min_command_s = min_command_s(len(self.cameras), width, height,
                                           self.options.get("samples", self.config.policy_samples))
        self.capabilities = Capabilities(
            "percussion", "newton", robot, self.kinds,
            frozenset({"proprioception", "audio", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": ("One tick = 1 ms (two 0.5 ms MuJoCo steps for the G1). " if g1 else
                          "One tick = 1 ms (the FR3's FCI rate). ") + "Every instrument mode advances 48 audio samples "
                         "per tick (48 kHz); near an impact the rigid world is stepped at 48 kHz inside the tick",
                "control": ("Unitree G1 (29 body joints) + two Dex3-1 hands (7 joints each), floating, standing on its PD "
                            "servos (no balance controller, no protective stop): named joint targets (radians) slewed at "
                            "the URDF joint speed; waist/arm gravity feed-forward; Cartesian pose/delta for the 'left'/"
                            "'right' hand TCP (80 mm along the wrist-yaw link) through a bounded 7-joint IK. Dex3 "
                            "fingers by named joint targets; no gripper or keyboard command" if g1 else
                            "Joint targets pass libfranka's rate limiter (velocity, 10 rad/s^2, 5000 rad/s^3 for the FR3) "
                            "into physical position servos with velocity feed-forward; IK computes targets only"),
                "arms": ",".join(self.arms),
                "gripper": ("Dex3-1 three-finger hands: thumb_0..2, index_0..1, middle_0..1 joint targets; fingertip "
                            "pads are vibro-contact strikers" if g1 else
                            "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode "
                            "grasp (total N of both fingers, <= 70 N; 50 mm/s until blocked, then force)"),
                "audio": (f"proprioception.audio.station_mic: the G1 head microphone array (one capsule at the head), last "
                          if g1 else "proprioception.audio.station_mic: the station-mast measurement microphone (UMIK-1 "
                          "class), last ") + (
                         f"{AUDIO_OBS_S:g} s of sound pressure (Pa) at {AUDIO_OBS_FS} Hz ({AUDIO_OBS_N} samples, oldest "
                         "first, zero before t = 0) plus L_AF (dB(A), IEC 61672 fast), L_Aeq over the last 100 ms and the "
                         "peak pressure; simulated from the struck instruments' modes through the room (direct sound, "
                         "first reflections, late reverberation)"),
                "keyboard": "Game-style chords held for ticks/rate_hz s (<= 10 s): W/S A/D R/F move the TCP along world "
                            "+y/-y -x/+x +z/-z at linear_speed_m_s; I/K J/L U/O rotate it about world x, y, z at "
                            "angular_speed_rad_s; ALT x0.2, SHIFT x6; Q closes, E opens the fingers (P x0.2, T x10); "
                            "H grips in force mode (30 N total; with G 10 N, with B 70 N) until E",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions",
                "min_command_s": (f"{self.min_command_s:g} s for the whole session: policy commands shorter than this are "
                                  f"refused before execution. It is set by the cameras: each command returns "
                                  f"{len(self.cameras)} camera observation(s) costing ~"
                                  f"{len(self.cameras)*FRAME_S_1280x720_24SPP:.1f} s of render on the reference GPU, and "
                                  f"T x {PHYSICS_RATIO_START:g} (reference physics wall/sim) + that render <= 200 T, never "
                                  "below 10 ms. It does not follow the host's measured speed, so on a slower host or in "
                                  "an expensive regime (hands resting on an instrument) a command can exceed 200x "
                                  "wall/sim; each policy command's measured cost is in its info.timing (physics_ratio, "
                                  "command_wall_s, wall_per_sim, over_budget). Owner commands are not floored "
                                  "(diagnostic stepping)"),
                "render_worker": (f"one Isaac worker per session, released {IDLE_RELEASE_S:g} s after its last observation "
                                  "(renderer_options.idle_release_s; 0 releases after every observation); the next "
                                  "observation starts a fresh one (10-20 s)"),
                "checkpoints": "In memory for this session only; a checkpoint restores only into the variation (preset, "
                               "task, seed, robot) it was saved in"})

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        from dataclasses import replace
        from percussion.variants import resolve
        from percussion.env import PercussionEnv
        from percussion.kinematics import ArmKinematics
        values = dict(options or {})
        if set(values) - {"preset", "task"}:
            raise ActionRejectedError("Reset selectors are preset, task (options) and seed")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        preset = values.get("preset", None if "task" in values else self.config.preset)
        task = values.get("task", self.config.task if preset == self.config.preset else None)
        try:
            sel = resolve(preset, task, self.config.seed if seed is None else seed)
        except (ValueError, KeyError) as exc:
            raise ActionRejectedError(str(exc)) from exc
        if (sel["preset"], sel["task"], sel["seed"]) == (self.config.preset, self.config.task, self.config.seed):
            return self.env.reset(), {"continuity": "reset", "variation": sel}
        config = replace(self.config, **sel)
        candidate = PercussionEnv(config, output_dir=self.output)
        if self.g1:                    # the IK for the new world before anything is swapped (a failure leaves the
            from percussion.g1 import G1Kinematics      # session as it was)
            ik = G1Kinematics(candidate.world)
        else:
            ik = ArmKinematics(candidate.world)
        with self.env.lock:
            self._end_operation(final=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
            old, self.env, self.config = self.env, candidate, config
            candidate.generation = old.generation + 1
            self.ik = ik
            self._renders_by_worker = {}
            self._scene_path = None
            if self.trace:
                path = self.trace.store.path
                self.trace.close()
                from percussion.trace import Trace
                self.trace = Trace(candidate, path)
            self.capabilities.notes["task"] = f"{config.preset}/{config.task} seed {config.seed}"
        del old
        return self.env.get_state(), {"continuity": "explicit_reset", "variation": sel}

    def select_variation(self, request):
        from percussion.variants import resolve
        from percussion.task import instruction
        if set(request) - {"preset", "task", "seed"}:
            raise ValueError("Selectors are preset, task and seed")
        sel = resolve(request.get("preset", None if "task" in request else self.config.preset), request.get("task"),
                      request.get("seed", 0))
        return dict(variation=sel, task=instruction(type(self.config)(**{**self.config.__dict__, **sel})),
                    reset_request=dict(seed=sel["seed"], options={"preset": sel["preset"], "task": sel["task"]}))

    def robot_configuration(self):
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
            return self.env.evaluator.evaluate()

    # ------------------------------------------------------------------ render workers
    def _scene(self):
        from percussion.render import export
        if getattr(self, "_scene_path", None) is None or not self._scene_path.is_file():
            self._scene_path, self.env.render_meta = export(self.env.world, self.output/"render", seed=self.config.seed)
        return self._scene_path

    def _new_renderer(self):
        from percussion.render import Renderer
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
        holder["thread"] = threading.Thread(target=work, name="percussion-render-prestart", daemon=True)
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
        t = threading.Thread(target=reap, name="percussion-render-release", daemon=False)
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
        from percussion.render import cue_visibility
        with self.env.lock:
            r = self._operation_renderer()
            try:
                t0 = time.perf_counter()
                result = r.observe(tuple(names), self.env.world.state.body_q.numpy(), self.env.world.tick,
                                   visibility=cue_visibility(self.env))
                self.last_timing["frame_s"] = time.perf_counter() - t0
                # a decaying maximum of the measured render time (the GPU is shared: 0.4 s per frame alone, 2.5-3.5 s
                # next to other workers' jobs); a worker's first frame (scene upload,
                # shader warm-up: 2-4x a steady frame) is not counted
                seen = getattr(self, "_renders_by_worker", {})
                self._renders_by_worker = seen
                first = seen.get(id(r), 0) == 0
                seen[id(r)] = seen.get(id(r), 0) + 1
                self.last_timing["warmup_frame"] = first
                if not first:
                    self._frame_s_seen = max(self.last_timing["frame_s"], 0.9*self._frame_s_seen)
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

    def _audio(self):
        """The station microphone's last AUDIO_OBS_S seconds at 16 kHz, its L_AF now, L_Aeq(100 ms) and peak
        (percussion.acoustics.policy_audio)."""
        from percussion.acoustics import policy_audio
        return {"station_mic": policy_audio(self.env.mic_window("station_mic", AUDIO_OBS_S + 0.25), AUDIO_OBS_N)}

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.world.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            obs = Observation(sequence, state["time_s"], self.config.robot,
                              {"joints": state["joints"], "robot_status": self._robot_status(), "audio": self._audio(),
                               "station": {"cue_on": 1.0 if self.env.cue.get("on") else 0.0}},
                              {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})
            self._account_command()
            return obs

    def _account_command(self):
        """After a policy command's observation: its measured cost, timed from the server's request receipt, into
        last_timing (the command's info.timing): physics wall/sim, the whole command's wall time and wall/sim, and
        whether it exceeded the 200x speed target. A record only; the minimum command length does not use it."""
        cmd, self._cmd = self._cmd, None
        if cmd is None:
            return
        wall = time.perf_counter() - cmd["t0"]
        phys = float(self.last_timing.get("physics_s", 0.0))
        ratio = phys/cmd["sim_s"]
        self.last_timing.update(command_wall_s=round(wall, 4), physics_ratio=round(ratio, 2),
                                wall_per_sim=round(wall/cmd["sim_s"], 2), over_budget=bool(wall > BUDGET*cmd["sim_s"]),
                                command_started=cmd.get("clock", "backend"))

    def _robot_status(self):
        w = self.env.world
        out = {"arms": {a: {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4)} for a in self.arms},
               "refused_targets": int(w.refused_targets)}
        if self.g1:                                   # the G1's IMU: pelvis height is not observable, tilt is
            from percussion.g1 import pelvis_state
            out["imu_tilt_deg"] = round(pelvis_state(w)[1], 4)
        return out

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
            total = 10. if "G" in keys else (70. if "B" in keys else 30.)
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
            if self.g1:
                raise UnsupportedActionError("The G1's Dex3 hands take named joint targets (g1/<side>_hand_*_joint)")
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
                if self.g1:
                    raise UnsupportedActionError("gripper_half_target_m: the G1's Dex3 fingers take named joint targets")
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        elif isinstance(action, Keyboard):
            if self.g1:
                raise UnsupportedActionError("Keyboard chords drive the Franka arms; the G1 takes joint or Cartesian commands")
            targets, ticks, grip = self._keyboard(action)
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, keyboard")
        return targets, ticks, grip

    def _execute(self, targets, ticks, grip):
        w = self.env.world
        for arm, f in grip.items():
            w.set_grip_force(arm, f)
        before = w.tick
        n_reflex, n_refused, n_seg = len(w.reflexes), w.refused_targets, len(w.contact_log)
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
                                   "contacts": len(w.contact_log) - n_seg,
                                   # this command's measured cost (physics_s; for a policy command, once its observation
                                   # is made: frame_s, command_wall_s, physics_ratio, wall_per_sim, over_budget)
                                   "timing": self.last_timing},
                           ev.get("score", 0.), False)

    def policy_access(self):
        from percussion.task import instruction
        from ..keyboard_control import SUPPORTED_KEYS
        proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        proprio.update({("robot_status", "arms", a, "stopped_s"): None for a in self.arms})
        proprio[("robot_status", "refused_targets")] = None
        if self.g1:
            proprio[("robot_status", "imu_tilt_deg")] = None
        proprio[("audio", "station_mic", "pressure_pa_16k")] = AUDIO_OBS_N
        for k in ("laf_dba", "laeq_100ms_dba", "peak_pa_100ms"):
            proprio[("audio", "station_mic", k)] = None
        proprio[("station", "cue_on")] = None
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                            self.kinds, lambda: instruction(self.config),
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()},
                             "gripper": self.capabilities.notes["gripper"],
                             "keyboard": self.capabilities.notes["keyboard"],
                             "audio": self.capabilities.notes["audio"],
                             "cue": "station.cue_on: the cue light on the station mast (strike-and-stop) is on (1) or off (0)",
                             "judgement": ("the episode is judged when you stop sending commands or reach the action "
                                           "limit; by then the robot must have been still for 0.5 s (no new joint "
                                           "target, and no joint other than the fingers moving faster than 0.01 rad/s)"),
                             "rate_limits": ("G1: targets slewed at the URDF joint velocity limit; no acceleration "
                                             "limit" if self.g1 else
                                             "libfranka-style: joint velocity 98 % of the limit, acceleration 10 rad/s^2, "
                                             "jerk 5000 rad/s^3 (FR3; Panda per-joint datasheet values)"),
                             "protective_stop": ("none: the G1 low-level joint interface has no collision or velocity "
                                                 "reflex; a fall ends the episode (robot_status.imu_tilt_deg)" if self.g1
                                                 else f"{'Panda' if 'panda' in self.config.robot else 'FR3'} collision "
                                                 "reflex (libfranka default thresholds: external joint torque over 10 N m "
                                                 "nominal, 20 N m while moving) and velocity reflex stop an arm; it then "
                                                 "refuses joint targets for 1 s (finger targets pass)"),
                             "min_command_s": self.min_command_now(),
                             "min_command_s_note": (f"shortest accepted policy command: {self.min_command_now():g} s of "
                                                    "simulated time (ticks x dt_s: "
                                                    f"{int(round(self.min_command_now()/self.config.rigid_dt))} ticks). "
                                                    "It is fixed for this session by its cameras' count, resolution "
                                                    "and samples (never below "
                                                    f"{MIN_POLICY_COMMAND_S:g} s) and does not change with the "
                                                    "simulation's speed; a shorter command is refused before execution "
                                                    "(HTTP 400) with that reason")},
                            keyboard_keys=tuple(SUPPORTED_KEYS))

    def min_command_now(self):
        """Shortest policy command (s): the session's fixed minimum (min_command_s), the same on every host.

        It does not follow this session's measured costs. The 200x wall/sim speed target protects no physics, safety
        or scoring property, and a minimum that held it on a host with much slower physics would be 0.33-1 s (1 s
        once the physics alone is >= 190x, where no length meets the target), longer than the 0.35-0.75 s beat of the
        steady-beat tasks. The measured cost of every policy command is recorded - info.timing: physics_ratio,
        command_wall_s, wall_per_sim and over_budget (> 200x) - for the owner."""
        return self.min_command_s

    def begin_action_trace(self, context):
        now = time.perf_counter()
        # the command's clock starts at the server's request receipt when it says so (received_perf_s: before the
        # history's durable 'started' commit - 0.03-0.55 s, a 0.85 s disk tail)
        rec = (context or {}).get("received_perf_s") if hasattr(context, "get") else None
        try:
            rec = float(rec) if rec is not None else None
        except (TypeError, ValueError):
            rec = None
        self._cmd_t0 = rec if rec is not None and 0.0 <= now - rec < 30.0 else now
        self._cmd_clock = "request" if self._cmd_t0 == rec else "trace"
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def act_policy(self, action):
        now = time.perf_counter()
        t0 = self._cmd_t0 if self._cmd_t0 is not None and now - self._cmd_t0 < 30.0 else now
        clock = getattr(self, "_cmd_clock", "trace") if t0 != now else "backend"
        self._cmd_t0 = None
        with self.env.lock:
            try:
                _, ticks, _ = self._plan(action)
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise ActionRejectedError(str(exc)) from exc
        floor = self.min_command_now()
        if ticks*self.config.rigid_dt < floor - 1e-12:
            err = ActionRejectedError(f"Policy commands must last >= {floor*1e3:.0f} ms of simulated time ({ticks} ticks "
                                      f"requested); the session's minimum, fixed by its {len(self.cameras)} camera(s) "
                                      "(capabilities notes.min_command_s)")
            err.policy_reason = (f"the command lasts {ticks*self.config.rigid_dt*1e3:g} ms of simulated time; the "
                                 f"minimum is {floor*1e3:.0f} ms ({int(round(floor/self.config.rigid_dt))} ticks, "
                                 "hardware.min_command_s), fixed for this session; nothing was executed")
            raise err
        result = self.act(action)
        self._cmd = dict(t0=t0, sim_s=ticks*self.config.rigid_dt, clock=clock)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        valid = tuple(c for c in POLICY_CAMERAS if c in self.camera_specs) + DIAGNOSTIC
        if name not in valid:
            raise ValueError(f"Render label must be one of {valid}")
        if name not in self.env.world.scene.cameras:
            raise ValueError(f"Camera {name} is not in preset {self.config.preset}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        """Privileged geometry: instruments (bars with their lengths, node positions and notes; the drumhead), tools,
        the task goal and the robot's bodies."""
        from percussion import params as P
        w = self.env.world
        sc = w.scene
        q = w.state.body_q.numpy()
        items = {}
        for n, it in sc.items.items():
            if "body" not in it:
                continue
            d = dict(kind=it["kind"], body_id=int(it["body"]), pose_xyzw=[float(v) for v in q[it["body"]]])
            for k in ("note", "length", "grip_x", "mass", "tension"):
                if k in it:
                    d[k] = it[k] if not isinstance(it[k], float) else round(it[k], 6)
            items[n] = d
        surfaces = {}
        for f in sc.surfaces:
            s = dict(kind=f["kind"], body_id=int(f["body"]), material=f["material"], structure=f["struct"])
            if f["kind"] == "bar":
                s.update(note=f["note"], half_extents_m=[float(v) for v in f["half"]],
                         frame="bar body frame: x along the bar, top face z = +half_extents[2]")
            else:
                s.update(radius_m=float(f["radius"]), frame="drum body frame: head plane z = 0, centre at the origin")
            surfaces[f["name"]] = s
        return {"items": items, "surfaces": surfaces, "goal": self.env.layout["goal"],
                "microphones": {"station_mic": list(P.STATION_MIC), "client_mic (privileged)": list(P.CLIENT_EAR)},
                "robot_bodies": {w.model.body_label[b]: int(b) for b in sc.robot_bodies},
                "source": "The simulator's own geometry (percussion/scene.py), not a camera estimate"}

    def get_geometry(self, geometry_id):
        idx = self.get_geometry_index()
        gid = str(geometry_id)
        if gid in idx["surfaces"]:
            return dict(id=gid, **idx["surfaces"][gid])
        if gid in idx["items"]:
            return dict(id=gid, **idx["items"][gid])
        raise KeyError(f"Unknown geometry id {gid!r}; ids are the item and surface names of /v1/geometry")

    def get_geometry_asset(self, asset_id):
        raise KeyError(f"Unknown asset {asset_id!r}; percussion publishes geometry through /v1/geometry/<id>")

    def variation_contract(self):
        from percussion.variants import contract
        return dict(**contract(), current=dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed),
                    reset="POST /v1/variations/select (or reset with options {preset, task} and a seed) rebuilds the "
                          "world for the selection in this session (10-20 s)")

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
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            if snap.get("variation") != self._variation_key():
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

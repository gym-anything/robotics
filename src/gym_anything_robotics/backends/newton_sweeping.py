"""Sweeping bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo CPU rigid world, a Warp bristle and
debris solver on the GPU, Isaac RTX images).

The robot is the Mobile FR3 Duo of classroom-arrangement with the same controller profile, so the command planning
(named joint targets, base twist, gripper, Cartesian pose and delta, keyboard chords), the protective stops, the
checkpoints and the physics trace are that backend's, inherited unchanged. This module adds what is this
environment's: its configuration and variations, its room export and render worker (rigid poses, loose pieces and
bristles in every frame), its task instructions and its geometry index.

Render worker. One Isaac render-only worker per session. It is started inside the launch (the first observation), kept
for as long as the session is live and not paused, whatever the time between commands, and replaced inside an owner's
reset to another seed or task. So no policy command waits for a worker to start: the classroom backend this one inherits
from releases its worker after 300 s without an image, and the first command after that pays the restart (247 to 266
times real time for a 0.05 s command). The owner's pause is the idle signal instead: a paused session keeps the worker
for renderer_options idle_release_s more seconds (PAUSED_IDLE_RELEASE_S by default; 0 gives it back at the pause, null
keeps it until stop) and then gives the GPU memory back with the world untouched; the owner's resume starts a released
worker again and returns only after its first frame, so that wait is the owner's. A session without policy cameras never
needs the worker for a command, so a worker an owner's diagnostic render started there is released the same time after
its last image or command. close() releases the worker at once and the worker watches its owner process.

Faults. A worker whose process has exited is replaced, and a frame that fails is retried once on a new worker
(_render): the command that meets the fault carries that one restart. A live session also looks at its worker every
WORKER_WATCH_S and starts a new one as soon as the old one has exited, so that a worker lost between two commands is
usually back before the next. If the retry fails too the worker is dropped, the error is raised (the gateway reports
the command as committed without an observation and the session as not ready) and the next image request starts
again from nothing: the owner's POST /v1/recover is such a request.

Refusals. Launch options are checked before anything is built (unknown renderer_options keys, ranges, camera names,
an observation so heavy that no command of legal length could carry it). A refused command raises
ActionRejectedError with policy_reason, the text a restricted caller is shown by the gateway; a policy request of
several commands is checked whole before its first command runs for everything that does not depend on where the
robot will be (SweepPolicyAccess).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from collections import deque
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .newton_classroom_arrangement import (NewtonClassroomArrangementBackend, POLICY_CAMERAS, BASE_CAMERAS, SESSION_RETENTION,
                                           IDLE_RELEASE_S, BASE_KEYS, MAX_ACTION_S, _finite)
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities

DIAGNOSTIC = ("room", "front", "plan", "chase", "side", "rail", "open", "square", "alcove", "table_b1", "desk", "craft", "spill")
SESSION_MARKER = ".gar-sweeping-session"
SELECTORS = ("preset", "task", "seed", "variant", "glides", "amount")
MIN_ACTION_S = 0.05        # per 640x360 camera frame of render load at up to 16 samples per pixel: a frame carries the
                           # loose pieces and the bristles, so commands are longer than classroom's 0.02 s
RENDER_OPTIONS = ("width", "height", "samples", "idle_release_s")
WIDTH_RANGE = (160, 1920)  # pixels of a policy frame; the cameras' native format is 1280 x 720
HEIGHT_RANGE = (90, 1080)
SAMPLES_RANGE = (1, 256)   # path-traced samples per pixel of a policy frame (policy_samples or renderer_options.samples)
SAMPLES_REF = 16           # the default; the frame cost that MIN_ACTION_S was measured at
# What a frame costs against samples per pixel: measured with the worker started at 16, 32, 64, 128 and 256 samples and
# the scene changing between frames. On a shared GPU a frame at 16 samples took 0.75 to 1.2 s and at 256 samples 7.7 to
# 13.3 s, between 10 and 16 times as long. The floor takes the proportion samples/16, so that a frame is no larger a
# share of the shortest command at more samples than at 16 (measured: the frame alone is 13 to 17 times real time at the
# shortest command for 32 to 256 samples, 16 to 24 times at 16), and never credits fewer than 16 samples (sample_factor)
KEYBOARD_MAX_S = 10.0      # the longest keyboard chord (classroom backend _keyboard); the longest command is MAX_ACTION_S
PAUSED_IDLE_RELEASE_S = IDLE_RELEASE_S   # 300 s: how long a paused (or camera-less) session keeps its render worker
WORKER_WATCH_S = 1.0       # how often a live session looks at its render worker's process
# measured on the shared RTX machine (GPU 99 % busy with other jobs): a worker restart took 7.5 to 8.0 s and
# its first frame 6.2 s (later frames of an unchanged scene 0.1 s); nvidia-smi showed 3,835 MiB for the worker and
# 534 MiB for the session process (the debris solver). Only quoted in the capabilities note, never used as a limit
WORKER_RESTART_S = 14.0
WORKER_GPU_GB = 3.8
# What a long command costs in wall time. A command pays its physics for
# every simulated second and one frame at its end, so over several seconds the cost is the physics: about 10 times
# real time while nothing the debris solver sees moves (the world skips the solver, SweepWorld._sweep_tick) and about
# 20 times while the robot moves. Measured three times through `gar action` on the shared machine, in camera-less
# sessions: an 8 s hold committed after 75 s, 81 s and 72 s, 9 to 10 times; a 10 s base drive at 0.1 m/s took about
# 195 s (longer than the client's default wait), 173 s and 213 s, 17 to 21 times. Round figures are quoted. Only
# quoted in the notes, never used as a limit.
LONG_COMMAND_REST_X = 10.0
LONG_COMMAND_MOVING_X = 20.0
CLIENT_TIMEOUT_S = 120.0   # what `gar action` and `gar benchmark` wait for a reply by default (cli_parser.py, --timeout)


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"sweeping-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


class UnknownCheckpoint(KeyError):
    """A checkpoint label this session does not hold. A KeyError, so that the gateway answers 400 and leaves the
    session ready (it latches a fault for a ValueError raised by a restore), with a sentence instead of KeyError's
    quoted key."""

    def __str__(self):
        return str(self.args[0]) if self.args else "unknown checkpoint"


@dataclass(frozen=True)
class SweepPolicyAccess(PolicyAccess):
    """PolicyAccess whose decode step also runs the backend's state-independent checks on every command of a request
    of several commands. The gateway decodes a request before it runs any of it (http_server.py action()), so such a
    request with a refused command anywhere in it is refused whole and nothing has moved. (Without this a request
    whose second command is too short would run the first and leave the session waiting for the owner's recovery.)"""
    precheck: Callable | None = None

    def decode_actions(self, payload):
        try:
            actions = super().decode_actions(payload)
        except ActionRejectedError:
            raise
        except (ValueError, TypeError) as exc:
            # A malformed policy request (an arm named neither left nor right, an unknown or missing field, a value
            # that is not a number; with a TypeError: null for a number, a number for a list, a body that is not an
            # object) is stopped by the shared decoder with a plain ValueError or TypeError, and the gateway shows a
            # restricted caller no text for that ("Invalid request; consult /v1/help."). The decoder's sentence is
            # made from the request alone, so it is given as the refusal's reason. A PermissionError (a command or a
            # simulator guard the policy key may not use) is not a ValueError and passes unchanged.
            raise NewtonSweepingBackend._rejected(exc) from exc
        if self.precheck is not None and isinstance(actions, (tuple, list)) and len(actions) > 1:
            self.precheck(tuple(actions))
        return actions


def sample_factor(samples):
    """What the floor allows for a frame at `samples` per pixel, relative to the default 16: in proportion above the
    default (an upper bound of the measured cost), 1 at and below it."""
    return max(1., float(samples)/SAMPLES_REF)


def check_renderer_options(options, policy_samples=None):
    """Validated renderer options as a new dict with every key present (width, height, samples, idle_release_s).
    Unknown keys and values out of range are refused (unified contract: unknown fields fail, nothing is clamped);
    `samples` and the launch option `policy_samples` are two names for the same number and must not disagree."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise UnsupportedConfigurationError(f"renderer_options must be an object with keys from {', '.join(RENDER_OPTIONS)}")
    unknown = sorted(str(k) for k in set(options) - set(RENDER_OPTIONS))
    if unknown:
        raise UnsupportedConfigurationError(f"Unknown renderer_options {unknown}; options: {', '.join(RENDER_OPTIONS)}")

    def whole(name, value, lo, hi):
        if type(value) is not int or not lo <= value <= hi:
            raise UnsupportedConfigurationError(f"{name} must be an integer {lo}..{hi} (got {value!r})")
        return value
    out = dict(width=whole("renderer_options.width", options.get("width", 640), *WIDTH_RANGE),
               height=whole("renderer_options.height", options.get("height", 360), *HEIGHT_RANGE))
    if policy_samples is not None:
        whole("policy_samples", policy_samples, *SAMPLES_RANGE)
    if "samples" in options:
        whole("renderer_options.samples", options["samples"], *SAMPLES_RANGE)
        if policy_samples is not None and policy_samples != options["samples"]:
            raise UnsupportedConfigurationError(
                f"policy_samples ({policy_samples}) and renderer_options.samples ({options['samples']}) both set the samples "
                "per pixel of a policy frame and disagree; give one of them")
    out["samples"] = options.get("samples", policy_samples if policy_samples is not None else SAMPLES_REF)
    idle = options.get("idle_release_s", PAUSED_IDLE_RELEASE_S)
    if idle is not None and (isinstance(idle, bool) or not isinstance(idle, (int, float)) or not math.isfinite(idle) or idle < 0):
        raise UnsupportedConfigurationError("renderer_options.idle_release_s must be null (keep the render worker until stop) "
                                            f"or a number of seconds >= 0 (got {idle!r})")
    out["idle_release_s"] = None if idle is None else float(idle)
    return out


class NewtonSweepingBackend(NewtonClassroomArrangementBackend):
    QUALIFIED_MODELS = ("mobile-fr3-duo@1", "mobile-fr3-single@1")

    def __init__(self, *, robot="mobile-fr3-duo", preset="room112", task=None, seed=0, variant="default", glides="nylon",
                 amount=1.0, reflex_profile="contact_x5", cameras=None, output_dir=None, renderer_options=None,
                 recording=True, device=None, reflex_stops_base=True, policy_samples=None, **unknown):
        from sweeping.config import Config
        from sweeping.layout import PRESETS
        from sweeping.env import SweepEnv
        from classroom_arrangement.kinematics import ArmKinematics
        from classroom_arrangement import params as CP
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["sweeping"]
        # ---- everything below this line and above "build" is checked before a folder or a world exists
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown sweeping options {sorted(unknown)}; options: robot, preset, task, seed, variant, glides, amount, "
                "reflex_profile, cameras, renderer_options, recording, device, reflex_stops_base, policy_samples")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if not isinstance(recording, bool):
            raise UnsupportedConfigurationError("recording must be true or false")
        self.options = check_renderer_options(renderer_options, policy_samples)
        if device not in (None, "cuda:0", "cuda"):
            raise UnsupportedConfigurationError(
                f"sweeping runs the rigid world on the CPU and the bristle and debris solver on the first GPU; device={device!r} "
                "is not supported (at 4,000 substeps a simulated second the solver is not usable on a CPU)")
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"sweeping does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; built: {', '.join(catalog['aliases'])}")
        try:
            self.config = Config(preset=preset, task=task or PRESETS.get(preset, ("",))[0], seed=seed, robot=robot, variant=variant,
                                 glides=glides, amount=amount, reflex_profile=reflex_profile, reflex_stops_base=reflex_stops_base,
                                 policy_samples=self.options["samples"])
            from sweeping.variants import resolve
            resolve(self.config.preset, self.config.task, self.config.seed, self.config.variant, self.config.glides, self.config.amount)
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.arms = self._arms_of(robot)
        self.policy_cameras = tuple(c for c in POLICY_CAMERAS if c == "head_rgb" or c in BASE_CAMERAS or c.split("_")[0] in self.arms)
        self.cameras = self._cameras(cameras, self.policy_cameras, self.robot_name)
        width, height, samples = self.options["width"], self.options["height"], self.options["samples"]
        self.idle_release_s = self.options["idle_release_s"]
        self.min_action_s = self._min_action_s(len(self.cameras), width, height, samples)
        if self.min_action_s > KEYBOARD_MAX_S:
            raise UnsupportedConfigurationError(
                f"{len(self.cameras)} camera(s) at {width}x{height} and {samples} samples per pixel would need every command "
                f"to last at least {self.min_action_s:g} s to stay inside 200x wall/sim, and a keyboard chord lasts at most "
                f"{KEYBOARD_MAX_S:g} s (any command at most {MAX_ACTION_S:g} s); choose fewer cameras, fewer pixels or fewer "
                "samples")
        # ---- build
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("sweeping session (auto-created; pruned by retention)\n")
        self.env = SweepEnv(self.config, output_dir=self.output)
        w = self.env.world
        if tuple(w.arms) != self.arms:
            raise RuntimeError(f"robot {self.robot_name!r} was built with arms {tuple(w.arms)}, the catalogue says {self.arms}")
        self.ik = ArmKinematics(w)
        self.camera_specs = {}
        for name in self.policy_cameras:
            fov = CP.HEAD_HFOV if name == "head_rgb" else (CP.BASE_CAMERA_HFOV if name in BASE_CAMERAS else CP.WRIST_HFOV)
            focal = .018/math.tan(math.radians(fov/2))
            mount = "franka_head_v0_2 head_camera_mounting_point (ZED Mini class)" if name == "head_rgb" else \
                (f"TMR {name.split('_')[0]}_mounting_point (RealSense D455 RGB)" if name in BASE_CAMERAS else
                 f"{name.split('_')[0]}_fr3_link7 D405 envelope")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.images = deque()
        self._renderer, self._timer, self._scene_path = None, None, None
        self._render_lock = threading.RLock()
        self._paused, self._closed, self._restart_due, self._fresh = False, False, False, False
        self.worker_events = dict(starts=0, launch_or_rebuild_starts=0, resume_restarts=0, fault_restarts=0, owner_restarts=0,
                                  frames=0, frame_retries=0, released=[], last_start_s=None, last_first_frame_s=None,
                                  last_fault=None)
        self._start_reason = "launch"
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
            "sweeping", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())
        self._watch_stop = threading.Event()
        self._watcher = None
        if self.cameras:
            self._watcher = threading.Thread(target=self._watch_worker, name="sweeping-render-watch", daemon=True)
            self._watcher.start()

    @staticmethod
    def _arms_of(robot):
        """The arms of a built configuration, from the catalogue (no world is needed to know them)."""
        from robots.catalog import for_environment, model
        (_, instance), = for_environment("sweeping", robot).to_dict()["instances"].items()
        groups = model(instance["model"]).to_dict()["groups"]
        return tuple(g[:-4] for g in groups if g.endswith("_arm"))

    @staticmethod
    def _cameras(cameras, allowed, robot_name):
        """The session's policy cameras, checked before the room is built (not after the 40 s build and settle)."""
        if cameras is None:
            return ("head_rgb",)
        if isinstance(cameras, (str, bytes)) or not isinstance(cameras, (list, tuple)) or not all(isinstance(c, str) for c in cameras):
            raise UnsupportedConfigurationError(f"cameras must be a list of camera names from {', '.join(allowed)} (got {cameras!r})")
        names = tuple(cameras)
        unknown = [c for c in names if c not in allowed]
        if unknown:
            hint = " (diagnostic views such as room or plan are owner renders, not policy cameras)" \
                if any(c in DIAGNOSTIC for c in unknown) else ""
            raise UnsupportedConfigurationError(f"Unknown policy camera(s) {unknown} for robot {robot_name!r}; the policy cameras "
                                                f"are {', '.join(allowed)}{hint}")
        if len(set(names)) != len(names):
            raise UnsupportedConfigurationError(f"cameras lists a name twice: {list(names)}")
        return names

    @staticmethod
    def _min_action_s(n_cameras, width, height, samples=SAMPLES_REF):
        """Shortest legal command for this session's observation, in whole 0.5 ms ticks: 0.05 s for one 640x360
        camera at up to 16 samples per pixel (or for no camera), in proportion to the number of rendered pixels per
        observation and to the samples per pixel above 16 (sample_factor)."""
        load = max(1., n_cameras*width*height/(640*360)*sample_factor(samples))
        return math.ceil(MIN_ACTION_S*load/0.0005 - 1e-9)*0.0005

    def _worker_note(self):
        idle = self.idle_release_s
        kept = ("keeps it until the session is stopped (renderer_options idle_release_s null)" if idle is None else
                "gives it back at once (renderer_options idle_release_s 0)" if idle == 0. else
                f"keeps it for {idle:g} s more (renderer_options idle_release_s) and then gives it back")
        if not self.cameras:
            return ("This session has no policy camera, so no command needs a render worker. An owner's diagnostic render "
                    f"starts one inside that request; after its last image or command the session {kept}.")
        return ("One Isaac render worker per session, started inside the launch and kept while the session is live and not "
                "paused, however long the policy waits between commands: no policy command starts a worker. It holds "
                f"about {WORKER_GPU_GB:g} GB of GPU memory. A paused session " + kept + "; the owner's resume then starts "
                f"a new one and returns after its first frame (about {WORKER_RESTART_S:g} s, measured), so the first "
                "command after a pause costs what any "
                "other does. A reset to another seed or task replaces the worker inside the owner's reset. If the worker "
                "dies, the session starts a new one at once; a command that meets a dead worker or a failed frame "
                "restarts it and renders once more (that one command then takes about "
                f"{WORKER_RESTART_S:g} s longer), and if that fails "
                "too the command is reported as committed without an observation and the owner's recover request starts "
                "a worker again.")

    def _long_command_note(self, policy=False):
        """What a command of several seconds costs in wall time and what its client has to do about it, for the owner
        (capabilities notes) and for the policy (hardware block). The numbers are LONG_COMMAND_REST_X,
        LONG_COMMAND_MOVING_X and CLIENT_TIMEOUT_S above and what follows from them and MAX_ACTION_S; nothing here
        limits a command."""
        fits = CLIENT_TIMEOUT_S/LONG_COMMAND_MOVING_X
        lo, hi = MAX_ACTION_S*LONG_COMMAND_REST_X, MAX_ACTION_S*LONG_COMMAND_MOVING_X
        cost = (f"A command of several seconds costs about {LONG_COMMAND_REST_X:g} times its simulated length in wall "
                f"time while the robot is at rest and about {LONG_COMMAND_MOVING_X:g} times while it moves, with one "
                "frame at its end (measured through gar action on a shared machine: 9 to 10 times for "
                "an 8 s hold, 17 to 21 times for a 10 s base drive; a busier machine is slower). The longest legal "
                f"command, {MAX_ACTION_S:g} s, takes about {lo:g} to {hi:g} s. ")
        if policy:
            return cost + ("The reply comes when the command has run, so the client has to wait that long: one that "
                           f"waits {CLIENT_TIMEOUT_S:g} s (the default of gar action and gar benchmark) loses the reply of "
                           f"a moving command longer than about {fits:g} s. A command whose reply was lost is not "
                           "cancelled: it runs to its end and is committed. Do not send it again; read the clock of "
                           "the next observation.")
        return cost + (f"gar action and gar benchmark wait {CLIENT_TIMEOUT_S:g} s for a reply by default, so a moving "
                       f"command longer than about {fits:g} s outlasts them: pass --timeout SECONDS to both for any "
                       f"command longer than a few seconds ({1.5*LONG_COMMAND_MOVING_X:g} s for each simulated second "
                       f"leaves a margin of one half; --timeout {1.5*hi:g} covers every legal command). An agent under "
                       "gar benchmark that sends one long command fails its run without it. A client that gives up "
                       "first says that execution may have occurred: the command is not cancelled, it runs to its end "
                       "and is committed, gar status reads UNREACHABLE until then, and afterwards the session is ready "
                       "with its clock advanced by the whole command. Do not send it again; read the clock first "
                       "(gar state SESSION --field time_s).")

    def _notes(self):
        from classroom_arrangement import params as CP
        c = self.config
        lim = CP.BASE_LIMITS
        o = self.options
        return {
            "min_command_s": f"every command lasts at least {self.min_action_s:g} s ({len(self.cameras)} camera(s) at "
                             f"{o['width']}x{o['height']}, {o['samples']} samples per pixel: 0.05 s per 640x360 frame of "
                             "render load at up to 16 samples, more in proportion to the pixels and to the samples "
                             "above 16) so that each command stays inside 200x wall/sim; a shorter command is refused "
                             "before any physics, with the reason. A policy reads the number as hardware.min_command_s "
                             "of GET /v1/capabilities",
            # the same minimum in ticks, digits only: an owner's capabilities carry no hardware block, and the session
            # console (session_static/app.js) reads this to pre-fill a hold of legal length
            "min_command_ticks": f"{int(round(self.min_action_s/0.0005))}",
            "long_commands": self._long_command_note(),
            "render_worker": self._worker_note(),
            "refusals": "A refused command changes nothing and says why (duration below the minimum or above 30 s, not a "
                        "multiple of the 0.5 ms tick, twist over the base limits, joint target out of range, unknown "
                        "joint, arm or key, unreachable Cartesian target, base held for 1 s after an arm's protective "
                        "stop). A request of several commands is checked whole before its first command runs for "
                        "everything that does not depend on where the robot will be; only an unreachable Cartesian "
                        "target or a held base at a later command can stop such a request part-way, and the owner "
                        "then has to recover the session.",
            "task": f"{c.preset}/{c.task} seed {c.seed}, variant {c.variant}, glides {c.glides}, amount {c.amount:g}",
            "clock": "One tick = 0.5 ms (rigid step; bristles and debris take two 0.25 ms substeps inside it); joint-target "
                     "actions give ticks, others duration_s",
            "robot": "Franka Mobile FR3 Duo: TMR base (two ArgoDrive swerve units, two casters, rocker), Franka Spine lift, "
                     + ("two FR3 arms" if len(self.arms) == 2 else "one FR3 arm (right)") + " with Franka Hands",
            "base": f"base_twist: v_x, v_y (m/s) and w_z (rad/s) in the base frame, |v| <= {lim['v']}, |w_z| <= {lim['w']}; "
                    f"slewed at {lim['a']} m/s2 and {lim['alpha']} rad/s2 by the swerve controller; drive torque 16 N m "
                    "accelerating / 28.4 N m braking; the twist is held until the next base_twist (send zeros to stop; "
                    "a stopped base holds its wheels). Wheels can slip.",
            "spine": f"named_joint_target robot/franka_spine_vertical_joint, 0..{CP.SPINE['stroke']} m at <= {CP.SPINE['speed']} m/s",
            "arms": "named_joint_target (radians) or cartesian_pose / cartesian_delta of an arm's hand TCP in the world frame "
                    "(IK computes joint targets; the servos move at the FR3 joint speed limits)",
            "gripper": "Franka Hand: half_width_m position mode (0..0.04 m per finger) or inward_effort_n force mode (total of "
                       "both fingers, <= 140 N, 50 mm/s until blocked, then that force)",
            "protective_stop": f"FR3 collision reflexes ({c.reflex_profile} thresholds; joint torque and hand wrench, 10 ms "
                               "filtered external joint torque) and velocity reflex stop an arm; it then refuses joint targets "
                               "for 1 s and the base is held for that second",
            "base_sensors": "robot_status.base_sensors: IMU (angular rate, specific force; base frame) and two corner LIDAR "
                            "scans (275 deg, 0.5 deg steps, 0.05-10 m); base RGB cameras front_rgb, rear_rgb, left_rgb, "
                            "right_rgb (D455, 90 deg) are selectable policy cameras",
            "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
            "keyboard": "keyboard chords: W/S/A/D/R/F move the chosen arm's hand along world y/x/z, I/K/J/L/U/O rotate it, Q/E "
                        "close/open the fingers, H grips by force (G 10 N, default 35 N, B 70 N total), ALT slow, SHIFT fast; "
                        "UP/DOWN/LEFT/RIGHT drive the base in its frame (0.3 m/s), Z/X turn it (0.5 rad/s); a chord lasts "
                        "ticks/rate_hz s (<= 10 s), then the base stops",
            "tools": "Brooms hang by their caps on prongs on the right wall; the lobby dustpan stands under them; the counter "
                     "brush and hand dustpan lie on the back counter (on the craft table in table-scraps). Nothing is attached "
                     "to the robot: a tool is held by fingertip friction and falls if the grip is too weak.",
            "debris": "Every loose piece is a rigid body in the debris solver, moved only by contact with bristles, tools, "
                      "containers, furniture, the floor, other pieces and the robot: its fingers and hands, its arm links "
                      "and torso (each as a box around its collision mesh), its wheels and a skirt box for the chassis.",
            "qualification": "Assisted fixtures are not policy actions"}

    # ------------------------------------------------------------------ lifecycle
    def _variation(self):
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, variant=c.variant, glides=c.glides, amount=float(c.amount))

    def _resolve(self, options):
        from sweeping.variants import resolve
        cur = self._variation()
        seed = options.get("seed", cur["seed"])
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        return resolve(options.get("preset", cur["preset"]), options.get("task", cur["task"]), seed,
                       options.get("variant", cur["variant"]), options.get("glides", cur["glides"]),
                       options.get("amount", cur["amount"]))

    def reset(self, *, seed=None, options=None):
        """Restore the initial state; with a different variation the room is rebuilt for it first (tools, debris and
        chair positions differ between variations)."""
        options = dict(options or {})
        unknown = set(options) - set(SELECTORS)
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are {', '.join(SELECTORS)}; "
                                         "nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            cur = self._variation()
            try:
                want = self._resolve(options)
            except (ValueError, TypeError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            if want != cur:
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def _rebuild(self, v):
        from dataclasses import replace
        from sweeping.env import SweepEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            with self._render_lock:                   # the scene changes: the reset's observation starts the new worker
                self._drop("rebuild")
                self._restart_due = False
                self._start_reason = "rebuild"
            self.config = replace(self.config, preset=v["preset"], task=v["task"], seed=int(v["seed"]), variant=v["variant"],
                                  glides=v["glides"], amount=float(v["amount"]))
            old = self.env
            self.env = SweepEnv(self.config, output_dir=self.output)
            self.ik = ArmKinematics(self.env.world)
            self._scene_path = None
            self._checkpoints = {}
            self._geometry = None
            self.env.generation = old.generation + 1
            self.capabilities = replace(self.capabilities, notes=self._notes())
            if self.recording:
                from classroom_arrangement.trace import Trace
                self.trace = Trace(self.env.world, self.output/f"physics-{self.env.generation}.sqlite3")

    def variation_contract(self):
        from sweeping.variants import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        from sweeping.task import instruction
        if set(request) - set(SELECTORS):
            raise ValueError(f"Selectors are {', '.join(SELECTORS)}")
        v = self._resolve(request)
        return dict(variation=v, task=instruction(None, v),
                    reset_request=dict(seed=v["seed"], options={k: v[k] for k in SELECTORS if k != "seed"}))

    # ------------------------------------------------------------------ rendering
    def _scene(self):
        from sweeping.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env.world, self.output/"render")
        return self._scene_path

    @staticmethod
    def _alive(r):
        p = getattr(r, "process", None)
        return r is not None and not getattr(r, "closed", False) and (p is None or p.poll() is None)

    def _note_release(self, reason):
        self.worker_events["released"].append(reason)
        del self.worker_events["released"][:-32]

    def _drop(self, reason):
        """Give the session's worker up (caller holds _render_lock). Returns whether there was one."""
        self._cancel_idle()
        r, self._renderer = self._renderer, None
        if r is None:
            return False
        self._note_release(reason)
        try:
            r.close()
        except Exception as exc:              # a worker that is already gone cannot be closed politely
            self.worker_events["last_fault"] = f"close after {reason}: {type(exc).__name__}: {exc}"[:300]
        return True

    def _start(self, reason):
        """Start a worker (caller holds _render_lock; the scene file exists or the caller holds the world)."""
        from sweeping.render import Renderer
        t0 = time.perf_counter()
        o = self.options
        self._renderer = Renderer(self.output/"render", self._scene(), width=o["width"], height=o["height"], samples=o["samples"])
        self._fresh = True
        took = time.perf_counter() - t0
        ev = self.worker_events
        ev["starts"] += 1
        ev[{"launch": "launch_or_rebuild_starts", "rebuild": "launch_or_rebuild_starts", "resume": "resume_restarts",
            "fault": "fault_restarts"}.get(reason, "owner_restarts")] += 1
        ev["last_start_s"] = round(took, 3)
        ev["last_start_reason"] = reason
        self.last_timing["cold_start_s"] = took
        return self._renderer

    def _worker(self, reason=None):
        """The session's live worker (caller holds _render_lock). A kept worker whose process has exited is replaced.
        `reason` names who pays for a start if one is needed: "launch", "rebuild", "resume", "fault" (a worker lost
        to a fault), or "owner" (an image asked for while no worker is held: an owner's render or observation on a
        paused or camera-less session)."""
        self._cancel_idle()
        if self._renderer is not None and not self._alive(self._renderer):
            code = getattr(getattr(self._renderer, "process", None), "returncode", None)
            self.worker_events["last_fault"] = f"the render worker's process had exited (code {code})"
            self._drop("worker_exited")
            reason = "fault"
        if self._renderer is None:
            if reason is None:
                reason, self._start_reason = (self._start_reason or ("fault" if self._restart_due else "owner")), None
            self._restart_due = False
            self._start(reason)
        return self._renderer

    def _render(self, names):
        """One frame of the named cameras from the present state (caller holds the world). A frame that fails is
        retried once on a new worker; if that fails as well the worker is dropped and the error raised."""
        from sweeping.render import frame
        with self._render_lock:
            fr, tick = frame(self.env.world), self.env.world.tick
            try:
                r = self._worker()
            except Exception as exc:                 # no worker at all (no Isaac interpreter, no GPU memory, ...)
                self.worker_events["last_fault"] = f"start: {type(exc).__name__}: {exc}"[:300]
                raise BackendProtocolError(f"The render worker could not be started ({type(exc).__name__}: {exc}); no "
                                           f"image of tick {tick}. The world is unchanged by this") from exc
            try:
                fresh, self._fresh = self._fresh, False
                t0 = time.perf_counter()
                result = r.observe(tuple(names), fr, tick)
            except Exception as first:
                self.worker_events["last_fault"] = f"{type(first).__name__}: {first}"[:300]
                self.worker_events["frame_retries"] += 1
                self._drop("frame_error")
                try:
                    r = self._worker("fault")
                    fresh, self._fresh = True, False
                    t0 = time.perf_counter()
                    result = r.observe(tuple(names), fr, tick)
                except Exception as second:
                    self._drop("frame_error")
                    self._restart_due = True         # the watch brings a worker back for the next request
                    self._arm_idle()
                    raise BackendProtocolError(
                        f"The render worker failed a frame twice ({type(first).__name__}: {first}; after a restart "
                        f"{type(second).__name__}: {second}); no image of tick {tick}. The world is unchanged by this; "
                        "the next image request starts a worker again") from second
            took = time.perf_counter() - t0
            self.last_timing["frame_s"] = took
            self.worker_events["frames"] += 1
            if fresh:
                self.worker_events["last_first_frame_s"] = round(took, 3)
            self._arm_idle()
        frames = {}
        for name, blob in result.items():
            path = self.output/f"{name}_{self.env.generation}_{self.env.world.tick}_{uuid.uuid4().hex[:8]}.png"
            path.write_bytes(blob)
            self.images.append(path)
            frames[name] = path
        while len(self.images) > 64:
            self.images.popleft().unlink(missing_ok=True)
        return frames

    # ---- when the worker is given back
    def _cancel_idle(self):
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _arm_idle(self):
        """After an image: a live session with policy cameras keeps its worker. A paused session, and a session
        without policy cameras, gives it back idle_release_s after its last image (0: now; null: never)."""
        self._cancel_idle()
        if self._renderer is None or not (self._paused or not self.cameras):
            return False
        idle = self.idle_release_s
        if idle is None:
            return False
        if idle <= 0.:
            return self._drop("paused" if self._paused else "no_policy_camera")
        timer = threading.Timer(idle, self._idle_expired)
        timer.args = (timer,)
        timer.daemon = True
        self._timer = timer
        timer.start()
        return False

    def _idle_expired(self, timer):
        with self._render_lock:
            if self._timer is not timer or not (self._paused or not self.cameras):   # an image, a resume or close came first
                return
            self._timer = None
            self._drop("paused_idle" if self._paused else "no_policy_camera_idle")

    def _release_later(self):                 # the classroom backend's name for the same decision
        with self._render_lock:
            self._arm_idle()

    def _release(self):
        with self._render_lock:
            self._drop("requested")

    def release_render_worker(self):
        """Give the worker back now (an in-process owner's own choice; no HTTP route calls this). The next image
        starts one inside that request, so this is not for a live session that serves a policy: pause it instead."""
        with self._render_lock:
            self._restart_due = False
            return self._drop("requested")

    def session_paused(self, paused):
        """Session-lifecycle hook (gym_anything_robotics.session_service: POST /v1/session/pause and /resume, inside
        the owner's request).

        Pause: the worker is kept idle_release_s more seconds and then given back (0: now; null: never). Returns
        whether it was given back now.
        Resume: if the paused session gave its worker back, or the worker died meanwhile, a new one is started and
        draws its first frame here, before the resume request returns: the wait is the owner's and the first policy
        command after it meets a ready worker. Returns whether a worker had to be started. A failure is raised to the
        session layer, which logs it and resumes all the same; the next image request then starts a worker."""
        from sweeping.render import frame
        with self.env.lock, self._render_lock:
            if paused:
                self._paused = True
                return bool(self._arm_idle())
            self._paused = False
            self._cancel_idle()
            if not self.cameras or self._alive(self._renderer):
                return False
            t0 = time.perf_counter()
            r = self._worker("resume")
            self._fresh = False
            try:
                r.observe(tuple(self.cameras), frame(self.env.world), self.env.world.tick)   # the slow first frame; discarded
            except Exception:
                self._drop("frame_error")
                self._restart_due = True
                raise
            self.worker_events["frames"] += 1
            self.worker_events["last_resume_restart_s"] = round(time.perf_counter() - t0, 3)
            return True

    # ---- the session's own look at its worker
    def _watch_worker(self):
        while not self._watch_stop.wait(WORKER_WATCH_S):
            try:
                self._check_worker()
            except Exception as exc:          # a failed look or start is not a fault of the session: the next image
                self.worker_events["last_fault"] = f"watch: {type(exc).__name__}: {exc}"[:300]   # request meets it itself

    def _check_worker(self):
        """If the kept worker's process has exited, or a failed frame has cost the session its worker, start a new
        one now and have it draw its first (slow) frame, instead of inside the next command. A paused session is left
        alone (its resume starts the worker), and so is a worker given back on purpose. Returns whether a worker was
        started."""
        from sweeping.render import frame
        dead = self._renderer is not None and not self._alive(self._renderer)
        if self._closed or self._paused or not self.cameras or not (dead or self._restart_due):
            return False
        if not self._render_lock.acquire(timeout=0.05):          # a frame is being drawn; it or the next look acts
            return False
        try:
            r = self._renderer
            dead = r is not None and not self._alive(r)
            if self._closed or self._paused or not (dead or (self._restart_due and r is None)):
                return False
            if self._scene_path is None or not self._scene_path.is_file():   # a rebuild is under way: its reset starts one
                return False
            if dead:
                code = getattr(getattr(r, "process", None), "returncode", None)
                self.worker_events["last_fault"] = f"the render worker's process had exited (code {code})"
                self._drop("worker_exited")
            self._restart_due = False                            # one attempt: if it fails, the next image request tries
            r = self._start("fault")
            env = self.env
            if env.lock.acquire(timeout=0.05):                   # no command holds the world: a copy of its state
                try:
                    fr, tick = (frame(env.world), env.world.tick) if self.env is env else (None, None)
                finally:
                    env.lock.release()
                if fr is not None:
                    t0 = time.perf_counter()
                    r.observe(tuple(self.cameras), fr, tick)     # the slow first frame; the image is discarded
                    self._fresh = False
                    self.worker_events["frames"] += 1
                    self.worker_events["last_first_frame_s"] = round(time.perf_counter() - t0, 3)
            return True
        finally:
            self._render_lock.release()

    def worker_status(self):
        """What the session's render worker is doing, for the owner (state field render_worker)."""
        r = self._renderer
        ev = self.worker_events
        return dict(held=r is not None, alive=self._alive(r), pid=getattr(getattr(r, "process", None), "pid", None),
                    paused=self._paused, idle_release_s=self.idle_release_s, idle_timer_armed=self._timer is not None,
                    **{k: (list(v) if isinstance(v, list) else v) for k, v in ev.items()})

    def get_state(self):
        state = self.env.get_state()
        state["render_worker"] = self.worker_status()
        return state

    def restore_checkpoint(self, label, **kwargs):
        saved = sorted(getattr(self, "_checkpoints", {}))
        if label not in saved:
            raise UnknownCheckpoint(f"No checkpoint named {label!r} in this session; saved: "
                                    f"{', '.join(repr(k) for k in saved) or 'none'} (checkpoints are kept in memory and a "
                                    "reset to another seed or task discards them). Nothing changed.")
        return super().restore_checkpoint(label, **kwargs)

    def close(self):
        self._closed = True
        self._watch_stop.set()
        if self.trace is not None:
            self.trace.close()
            self.trace = None
        with self._render_lock:
            self._drop("close")
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=5.0)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        allowed = self.policy_cameras + DIAGNOSTIC
        if name not in allowed:
            raise ValueError(f"Render label must be one of {allowed}")
        with self.env.lock:
            return self._render((name,))[name]

    # ------------------------------------------------------------------ commands: every refusal says why
    @staticmethod
    def _rejected(exc, prefix=""):
        """ActionRejectedError for a refusal decided from the command and the robot's own limits, joints and state
        (duration, tick, minimum, twist and joint limits, finger width, grip force, arm, keys, reach, base held after
        a protective stop): that text is what a real controller tells its operator, so the gateway may show it to the
        policy (policy_reason; http_server.py _error)."""
        text = prefix + str(exc)
        err = ActionRejectedError(text)
        err.policy_reason = text
        return err

    def act(self, action):
        try:
            return super().act(action)
        except ActionRejectedError as exc:
            if getattr(exc, "policy_reason", None) is None:
                exc.policy_reason = str(exc)
            raise
        finally:
            # the inherited act stops the idle clock for the length of the command; a session that gives its worker
            # back when idle (paused, or without policy cameras) starts the clock again here
            if self._renderer is not None and self._timer is None and (self._paused or not self.cameras):
                with self._render_lock:
                    self._arm_idle()

    def act_policy(self, action):
        """As act, without the reward. A command the robot does not have (an arm it lacks, a key it does not know) is
        a refusal with its reason here too: the gateway shows a restricted caller no text for the unsupported kind."""
        try:
            result = self.act(action)
        except UnsupportedActionError as exc:
            raise self._rejected(exc) from exc
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def _precheck(self, actions):
        """Every command of a policy request of several commands, checked before the first one runs
        (SweepPolicyAccess.decode_actions): durations against the minimum, the maximum and the tick, twist limits, arm
        and key names, joint names and limits, finger widths and grip forces. Nothing is solved, written or stepped:
        the inverse kinematics of a Cartesian or hand-key command and the base hold after a protective stop depend on
        where the robot is when the command's turn comes, and are decided then."""
        n = len(actions)
        with self.env.lock:
            solve = self.ik.solve
            self.ik.solve = lambda *a, **k: {}
            try:
                for index, action in enumerate(actions):
                    try:
                        self._plan(action)
                    except (ValueError, UnsupportedActionError, UnsupportedConfigurationError) as exc:
                        raise self._rejected(exc, f"command {index + 1} of {n}: ") from exc
            except ActionRejectedError as exc:
                exc.policy_reason = exc.args[0] + " Nothing in this request was executed."
                raise
            finally:
                self.ik.solve = solve

    # ------------------------------------------------------------------ policy access
    def policy_access(self):
        from sweeping.task import instruction
        return SweepPolicyAccess(self.robot_name, tuple(self.camera_specs[c] for c in self.cameras), self._proprio(), self.kinds,
                                 lambda: instruction(self.config, self.env.spec), self._policy_info(self.capabilities.notes),
                                 precheck=self._precheck)

    def _proprio(self):
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
        return proprio

    def _policy_info(self, notes):
        w = self.env.world
        joints = {}
        for n, s in w.joints.items():
            joints[n] = {"lower": _finite(float(s["lower"])), "upper": _finite(float(s["upper"])),
                         "velocity_limit": _finite(float(s.get("velocity_limit", float("inf")))),
                         "effort_limit": float(s["effort_limit"]), "unit": s.get("unit", "rad"),
                         "commandable": n not in w.base_joint_names}
        lo = self.min_action_s
        return {"dt_s": 0.0005, "min_command_s": lo, "max_command_s": MAX_ACTION_S,
                "command_length": f"Every command lasts at least {lo:g} s and at most {MAX_ACTION_S:g} s of simulated time, in "
                                  f"whole ticks of 0.5 ms: duration_s from {lo:g} to {MAX_ACTION_S:g} in multiples of 0.0005; "
                                  f"ticks (named_joint_target) from {int(round(lo/0.0005))} to {int(round(MAX_ACTION_S/0.0005))}; a "
                                  f"keyboard chord lasts ticks/rate_hz seconds, from {lo:g} to 10. A shorter, longer or "
                                  "off-tick command is refused before anything moves, and the refusal says which limit. "
                                  f"The minimum follows this session's observation ({len(self.cameras)} camera(s) at "
                                  f"{self.options['width']}x{self.options['height']}, {self.options['samples']} samples per "
                                  "pixel) and does not change during the session.",
                "command_wall_time": self._long_command_note(policy=True),
                "refusals": notes["refusals"],
                "arms": list(self.arms), "joints": joints, "base": notes["base"], "spine": notes["spine"],
                "gripper": notes["gripper"], "control": notes["arms"], "protective_stop": notes["protective_stop"],
                "tools": notes["tools"]}

    @classmethod
    def _qualified_setup(cls, robot, catalog):
        from sweeping.robot_setup import validate
        from robots.catalog import resolve
        try:
            validate(robot)
            return resolve(robot).to_dict()
        except (TypeError, ValueError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------ geometry (privileged)
    def get_geometry_index(self):
        """Privileged geometry: every rigid shape and body with mass, centre of mass and inertia, the furniture, tools
        and containers with their part lists and current poses, the bristle banks, and the loose pieces by class."""
        spec, w = self.env.spec, self.env.world
        q = w.state.body_q.numpy()
        index = dict(self._catalog().index())
        index.pop("pen_frame", None)
        d = w.debris()
        classes = {}
        for k, c in enumerate(w.debris_classes):
            sel = d["cls"] == k
            if sel.any():
                classes[c.name] = dict(pieces=int(sel.sum()), mass_g=round(1e3*float(d["mass"][sel].sum()), 4),
                                       friction_floor=c.mu_floor, friction=c.mu, friction_bristle=c.mu_bristle,
                                       restitution=c.restitution, rolling_resistance=c.mu_roll)
        index.update(
            items={n: dict(kind=it["kind"], body_id=int(it["body"]), pose_xyzw=[float(v) for v in q[it["body"]]],
                           shape_ids=[str(s) for s in it["shapes"]]) for n, it in w.scene.items.items()},
            tools={n: dict(kind=t["kind"], body_id=int(t["body"]),
                           bristles=None if t.get("bank") is None else dict(clusters=int(len(t["bank"].roots)), nodes=int(t["bank"].nodes),
                                                                             filaments_per_cluster=float(t["bank"].filaments),
                                                                             filament_d_m=float(t["bank"].d), modulus_pa=float(t["bank"].E)))
                   for n, t in w.scene.tools.items()},
            debris=dict(pieces=int(len(d["mass"])), classes=classes, note="get_geometry('debris') returns every piece's pose"),
            zone=list(spec["zone"]), square=spec["square"],
            item_ids="get_geometry(<item or tool name>) returns its authored parts and current pose",
            source="Actual model geometry, masses and friction; not a camera-derived estimate")
        return index

    def get_geometry(self, geometry_id):
        gid = str(geometry_id)
        w = self.env.world
        if gid == "debris":
            d = w.debris()
            names = [c.name for c in w.debris_classes]
            return dict(id="debris", position_m=np.round(d["pos"], 5).tolist(), orientation_xyzw=np.round(d["quat"], 5).tolist(),
                        mass_kg=d["mass"].tolist(), cls=[names[k] for k in d["cls"]], asleep=d["asleep"].tolist())
        return super().get_geometry(gid)

"""Forklift bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 / CPU MuJoCo with the forklift's tyre,
drivetrain, steering and hydraulic models; Isaac RTX render-only workers).

The operator works a 2.5 t LPG counterbalance truck only by moving its controls by contact. Two robots are offered: the
default, a floating Unitree H1-2 with Inspire hands seated on the truck's suspension seat (Unitree low-level motor
channels per joint: q*, dq*, kp, kd, tau_ff; forklift/motors.py), and a driving-robot kit bolted into the cab
(--robot drive-kit: steering, pedal, lever, selector, hand-brake, key and horn actuators; operator.KitMotors). Commands
are held for at least MIN_COMMAND_TICKS. One tick = 0.5 ms. The evaluator is the environment's (forklift/evaluator.py).
Policy camera: the selected robot's head or mast camera. Every command of a served session is
recorded tick by tick for the owner (_ForkliftTrace; GET /v1/history/<id>/trace). The policy's capabilities carry the
measured wall time, the client timeout and the render worker's release interval (hardware).
"""

from collections import deque
import copy
import math
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from .base import RoboticsBackend
from ..actions import NamedJointTarget
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..physics_trace import PhysicsTraceStore, ValidatedJSON
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

SESSION_MARKER = ".gar-forklift-session"
SESSION_RETENTION = 20
# The Isaac render worker is kept warm this long after a request (renderer_options idle_release_s; 0 releases it after
# every request). A policy that thinks longer than this between steps pays a new worker at its next observation,
# 58-61 s each time on a busy shared host (about 3,000 wall s per simulated s for a 20 ms command). An LLM or VLA policy
# thinks seconds to minutes; 10 minutes covers it (sewing and welding use the same), and a session left idle longer
# gives the worker's GPU memory back to the shared GPU.
IDLE_RELEASE_S = 600.0
# A request that has to start a new render worker (a session's first observation is taken by `gar run` itself; after
# that, the first request after more than IDLE_RELEASE_S without one), as a client sees it, on a shared host: 71-99 s
# (five requests of a cold program, 73-99 s, and the request after a 660 s pause, 71 s; 1-minute load 14-23). Runs on
# the same host measured up to 162 s at higher load. The worker starts beside the command's physics (_prestart_worker),
# so only a command whose physics lasts as long hides the start: a 5 s command (32-34 s of physics) still waited
# 62-65 s.
COLD_WORKER_S = "71-99 s"
POLICY_SAMPLES = 8
MIN_COMMAND_TICKS = 4
MAX_COMMAND_S = 30.0
# Served runtime per robot, measured through `gar run` + HTTP with the policy key and the camera (100-request loops per
# command duration and task).
# shortest_command_within_200x_s: the shortest command none of whose requests exceeded 200 wall s per simulated s and
# from which every longer command stayed inside 200 at its 99th percentile; a longer command's rare request over 200 is
# a stall of the shared host (stalls_beyond_declared: [command s, the request's wall s]), which no command length
# absorbs.
# per_step: wall s per simulated s per request (median, p99, worst, requests over 200, sustained = sum wall / sum
# simulated); frame_render_s: what a new 30 Hz frame adds to a request (seconds); cold_request_wall_s: a request that
# starts a new render worker; camera_launch_wall_s: `gar run` with the camera; worker_gpu_mib: the render worker's GPU
# memory (nvidia-smi, per process); host_rss_mib / host_pss_mib: the server and the worker together.
RUNTIME = {
    'h1-2': {'shortest_command_within_200x_s': 0.02, 'per_step': {'0.01': {'n': 300, 'median': 31.1, 'p99': 180.8, 'max': 276.1, 'over_limit': 1, 'sustained': 62.3}, '0.02': {'n': 300, 'median': 65.7, 'p99': 109.5, 'max': 144.7, 'over_limit': 0, 'sustained': 55.3}, '0.1': {'n': 300, 'median': 25.3, 'p99': 44.3, 'max': 49.8, 'over_limit': 0, 'sustained': 26.7}}, 'stalls_beyond_declared': [], 'frame_render_s': {'median': 1.09, 'p90': 1.33, 'max': 2.48}, 'request_fixed_s': 0.158, 'load_1min': [12.0, 53.0], 'gpu_util_pct': [87, 100], 'cold_request_wall_s': [71.0, 99.0], 'camera_launch_wall_s': [91.0, 162.0], 'worker_gpu_mib': [6040, 8170], 'host_rss_mib': [6776.0, 7324.0], 'host_pss_mib': [6136.0, 6797.0], 'physics_1s': {'start-and-park': {'median': 6.3, 'max': 6.6, 'load_1min': [16.6, 19.6]}, 'aisle-cruise': {'median': 7.7, 'max': 9.9, 'load_1min': [13.9, 16.9]}, 'drive-to-pallet': {'median': 7.2, 'max': 7.5, 'load_1min': [15.8, 26.2]}, 'carry-to-floor-bay': {'median': 7.1, 'max': 7.2, 'load_1min': [17.5, 24.2]}, 'carry-loose-stack': {'median': 19.9, 'max': 23.1, 'load_1min': [10.0, 21.0]}, 'put-away-rack': {'median': 9.5, 'max': 10.9, 'load_1min': [9.0, 17.7]}, 'feed-conveyor': {'median': 8.0, 'max': 8.4, 'load_1min': [9.3, 39.6]}, 'load-trailer': {'median': 8.8, 'max': 9.6, 'load_1min': [11.9, 13.9]}, 'load-trailer-chocked': {'median': 8.4, 'max': 8.9, 'load_1min': [10.0, 10.6]}, 'load-bales': {'median': 8.2, 'max': 9.3, 'load_1min': [13.7, 15.7]}, 'unload-trailer': {'median': 7.7, 'max': 8.0, 'load_1min': [11.0, 11.8]}, 'ramp-up-loaded': {'median': 7.9, 'max': 8.2, 'load_1min': [10.8, 11.8]}, 'ramp-down-loaded': {'median': 10.1, 'max': 10.3, 'load_1min': [10.6, 11.7]}, 'yard-route': {'median': 7.3, 'max': 7.5, 'load_1min': [10.0, 11.2]}}},
    'drive-kit': {'shortest_command_within_200x_s': 0.1, 'per_step': {'0.01': {'n': 300, 'median': 36.4, 'p99': 193.3, 'max': 377.4, 'over_limit': 3, 'sustained': 69.9}, '0.02': {'n': 300, 'median': 76.8, 'p99': 151.7, 'max': 209.5, 'over_limit': 1, 'sustained': 63.0}, '0.1': {'n': 300, 'median': 24.1, 'p99': 43.5, 'max': 57.4, 'over_limit': 0, 'sustained': 26.8}}, 'stalls_beyond_declared': [], 'frame_render_s': {'median': 1.23, 'p90': 1.52, 'max': 2.35}, 'request_fixed_s': 0.127, 'load_1min': [15.5, 50.6], 'gpu_util_pct': [88, 100], 'cold_request_wall_s': None, 'camera_launch_wall_s': [67.0, 127.0], 'worker_gpu_mib': [6338, 8040], 'host_rss_mib': [6186.0, 6506.0], 'host_pss_mib': [5541.0, 5846.0], 'physics_1s': {'start-and-park': {'median': 5.0, 'max': 5.1, 'load_1min': [7.7, 26.1]}, 'carry-loose-stack': {'median': 18.7, 'max': 21.3, 'load_1min': [12.3, 46.5]}, 'put-away-rack': {'median': 8.2, 'max': 9.1, 'load_1min': [10.2, 21.5]}, 'load-trailer': {'median': 7.6, 'max': 7.8, 'load_1min': [10.3, 13.7]}, 'ramp-up-loaded': {'median': 6.7, 'max': 7.5, 'load_1min': [10.7, 11.2]}}},
}
# The per-action physics trace (on by default), per robot, measured in process: the trace's own time per 0.5 ms tick
# (it times itself), its share of a traced command's wall time on that host, payload and file bytes per simulated
# second (parked on every task, and driven by the assisted autopilot), the simulated seconds the default retention
# budget (TRACE_BYTES, counted in payload bytes) keeps, whether one maximum command fits it, and the physics alone
# with the trace on (wall s per simulated s, 0.5 s commands).
TRACE = {
    'h1-2': {'tasks': 14, 'ms_per_tick': [0.19, 0.41], 'share_of_traced_wall_pct': [4.1, 10.4], 'payload_MB_per_sim_s': [3.37, 6.17], 'file_MB_per_sim_s': [4.27, 8.41], 'retained_sim_s': [44, 80], 'max_command_fits': True, 'physics_traced': [5.9, 20.2], 'physics_slowest_task': 'carry-loose-stack', 'load_1min': [8.3, 30.38]},
    'drive-kit': {'tasks': 14, 'ms_per_tick': [0.17, 0.59], 'share_of_traced_wall_pct': [3.5, 18.5], 'payload_MB_per_sim_s': [2.81, 5.54], 'file_MB_per_sim_s': [4.27, 8.4], 'retained_sim_s': [48, 96], 'max_command_fits': True, 'physics_traced': [4.9, 18.8], 'physics_slowest_task': 'carry-loose-stack', 'load_1min': [8.56, 17.94]},
}
# The shortest command the capabilities declare for the default robot (RUNTIME carries each robot's).
SHORTEST_COMMAND_WITHIN_200X_S = RUNTIME["h1-2"]["shortest_command_within_200x_s"]
# HTTP client timeout the backend declares (hardware.client_timeout_s; the gar CLI uses it by default, else 120 s). The
# longest legal request is one 30 s command (MAX_COMMAND_S): one on carry-loose-stack with the trace on took 594 s
# (the slowest physics with the trace on, TRACE: 20.2 wall s per simulated s), plus a request that starts a render
# worker (up to 99 s here) and the request's fixed cost (seconds): about 700 s. Runs at higher load measured physics up
# to 38-47 and worker starts up to 162 s (1,300-1,600 s for such a request); 1,800 s covers them; a busier host needs
# more.
CLIENT_TIMEOUT_S = 1800
CHANNELS = ("q", "dq", "kp", "kd", "tau")
CAMERA_HZ = 30.0                  # D435-class colour stream at 848 x 480 runs 30-60 Hz; frames are captured at 30 Hz
HEAD_CAMERA = dict(width=848, height=480, hfov_deg=69.4)      # Intel RealSense D435i colour (H1-2 head camera)
RENDERER_OPTIONS = frozenset({"samples", "idle_release_s", "trace", "trace_bytes"})
# The trace's retention budget in payload bytes: at the fastest measured writing, 6.17 MB per simulated second
# (carry-loose-stack, TRACE['h1-2']), it keeps the last 44 s of simulated time, so one maximum command fits: a 30 s
# command there kept all 60,000 samples and evicted none (181 MB, 68 % of the budget). Older samples are evicted and
# counted (the trace summary's evicted_samples).
TRACE_BYTES = 256 * 1024 * 1024
ROBOT_NAMES = {"h1-2": "seated Unitree H1-2", "drive-kit": "driving-robot kit"}


def _n(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}"


def _span(lo, hi, scale=1.0, fmt="{:.1f}"):
    a, b = fmt.format(lo * scale), fmt.format(hi * scale)
    return a if a == b else f"{a}-{b}"


def wall_time_note(robot):
    """The runtime note of a robot's sessions (hardware.wall_time), written from RUNTIME and TRACE: what a policy pays
    per request, why, and what a session holds."""
    r, t = RUNTIME.get(robot), TRACE.get(robot)
    if not r:
        return f"No served-runtime measurement exists for robot {robot!r}."
    length = lambda d: f"{float(d) * 1e3:g} ms" if float(d) < 0.1 else f"{float(d):g} s"
    steps = "; ".join(f"{length(d)} commands median {_n(c['median'])}, p99 {_n(c['p99'])}, worst {_n(c['max'])} "
                      f"({c['over_limit']} of {c['n']} over 200; sustained {_n(c['sustained'])})"
                      for d, c in r["per_step"].items())
    slowest = ", ".join(f"{c['max'] * float(d):.3g} s for a {length(d)} command" for d, c in r["per_step"].items())
    fr, gpu, pss, rss = r["frame_render_s"], r["worker_gpu_mib"], r["host_pss_mib"], r["host_rss_mib"]
    stalls = (f"the slowest single requests here took {slowest}; a request that stalls for longer than 200 times its "
              "command (2 s for a 10 ms command, 20 s for a 0.1 s one) exceeds the limit, and nothing in the environment "
              "bounds such stalls")
    served = (" Physics alone through the served backend (owner key, 1 s commands): " + ", ".join(
        f"{k} {v['median']:g}" for k, v in r["physics_1s"].items()) + ".") if r.get("physics_1s") else ""
    trace = (f"costs {t['ms_per_tick'][0]:g}-{t['ms_per_tick'][1]:g} ms of wall time per 0.5 ms tick ("
             f"{t['share_of_traced_wall_pct'][0]:g}-{t['share_of_traced_wall_pct'][1]:g} % of a traced command's time "
             f"while physics ran at {t['physics_traced'][0]:g}-{t['physics_traced'][1]:g}; a larger share when physics "
             f"runs faster), writes {t['payload_MB_per_sim_s'][0]:g}-{t['payload_MB_per_sim_s'][1]:g} MB of payload "
             f"({t['file_MB_per_sim_s'][0]:g}-{t['file_MB_per_sim_s'][1]:g} MB on disk) per simulated second, and its "
             f"default {TRACE_BYTES / 2 ** 20:g} MiB budget keeps the last {t['retained_sim_s'][0]:g}-"
             f"{t['retained_sim_s'][1]:g} s of simulated time (one {MAX_COMMAND_S:g} s command "
             f"{'fits' if t['max_command_fits'] else 'does not fit'}); older samples are evicted and counted (the trace "
             "summary's evicted_samples)") if t else "was not measured for this robot"
    physics = (f" Physics alone (in process, trace on, 0.5 s commands, {t['tasks']} tasks): "
               f"{t['physics_traced'][0]:g}-{t['physics_traced'][1]:g} wall s per simulated s ({t['physics_slowest_task']} "
               f"the slowest).") if t else ""
    return (
        f"Measured for the {ROBOT_NAMES.get(robot, robot)} through gar run and HTTP with the policy key and the camera, "
        f"100-request loops per command length, on a shared host (1-minute load {r['load_1min'][0]:g}-"
        f"{r['load_1min'][1]:g}, GPU utilisation {r['gpu_util_pct'][0]:g}-{r['gpu_util_pct'][1]:g} %). "
        "Wall s per simulated s per request: "
        f"{steps}. shortest_command_within_200x_s ({length(r['shortest_command_within_200x_s'])}) is the shortest "
        "command none of whose requests exceeded 200 and from which every longer command stayed inside 200 at its 99th "
        f"percentile; it does not bound every request: on this saturated shared GPU {stalls}. Short commands cost more "
        "because a command that crosses a 30 Hz frame boundary renders a new camera frame inside its request (median "
        f"{fr['median']:g} s, p90 {fr['p90']:g} s, up to {fr['max']:g} s here): a 10 ms command crosses one about every "
        "third command, a 20 ms command two in three, a 0.1 s command always; an observation inside the current frame "
        f"reuses it. Every command also pays the action history's synced commits (median {r['request_fixed_s']:g} s)."
        f"{served}{physics} The render worker stays warm {IDLE_RELEASE_S:g} s after each request (renderer_options "
        f"idle_release_s); a request that has to start a new one (the first after a longer pause) took {COLD_WORKER_S} "
        "on this host, and since the new worker starts beside the command's physics, only a command whose physics lasts "
        f"that long hides it; gar run with the camera took {r['camera_launch_wall_s'][0]:g}-{r['camera_launch_wall_s'][1]:g} "
        f"s. A session with the camera holds a render worker of {_span(*gpu, 1 / 1024)} GiB of GPU memory, and its "
        f"server and worker {_span(*pss, 1 / 1024)} GiB of host memory (PSS; {_span(*rss, 1 / 1024)} GiB resident). "
        "The per-action physics trace (owner diagnostics, on by "
        f"default; renderer_options trace=false turns it off) {trace}. A command carries at most {MAX_COMMAND_S:g} s of "
        f"simulated time. Client timeout: {CLIENT_TIMEOUT_S} s (hardware.client_timeout_s; the gar CLI uses it); give "
        "other HTTP clients at least that.")


# the default robot's note (the robot-specific note of a session is wall_time_note(its robot))
WALL_TIME_NOTE = wall_time_note("h1-2")


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws) / ".gar" if ws else Path(tempfile.gettempdir())) / "forklift-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p / SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base / uuid.uuid4().hex


def _check_renderer_options(options):
    """renderer_options: samples (path-traced samples per pixel), idle_release_s (seconds the render worker stays warm
    after a request; 0 releases it after every request), trace (per-action physics trace, default on), trace_bytes
    (the trace's retention budget). Unknown keys and out-of-range values are refused, never replaced."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise UnsupportedConfigurationError("renderer_options must be an object")
    unknown = set(options) - RENDERER_OPTIONS
    if unknown:
        raise UnsupportedConfigurationError(f"renderer_options accepts {sorted(RENDERER_OPTIONS)}; unknown: {sorted(unknown)}")
    v = options.get("samples")
    if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 256):
        raise UnsupportedConfigurationError("renderer_options.samples must be an integer in [1, 256]")
    v = options.get("idle_release_s")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
        raise UnsupportedConfigurationError("renderer_options.idle_release_s must be a number of seconds >= 0 (0: release "
                                            "the render worker after every request)")
    v = options.get("trace")
    if v is not None and not isinstance(v, bool):
        raise UnsupportedConfigurationError("renderer_options.trace must be true or false")
    v = options.get("trace_bytes")
    if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 16 * 1024 * 1024):
        raise UnsupportedConfigurationError("renderer_options.trace_bytes must be an integer of at least 16 MiB")
    return dict(options)


def _built(setup):
    """What the operator builds from a setup: its instances and attachments (its id and description are names)."""
    d = setup.to_dict()
    return dict(instances=d["instances"], attachments=d.get("attachments", {}))


def _accepted_setup(requested, accepted):
    """The robot setup the session must run, and a refusal when the caller's own setup is not exactly it.

    `requested`: what the caller passed (an alias, a catalogue rig id, a JSON path, a RobotSetup or its dict);
    `accepted`: what forklift.robot_setup.validate_setup made of it (an alias of robots.catalog ENVIRONMENTS['forklift'],
    or a setup). The operator builds the catalogue rig of the alias, so a caller's setup whose instances or attachments
    differ from that rig (a mount, a profile, a pose, a parameter) would be replaced by the stock rig while the session
    reported the caller's request: refused instead.
    The same rule as validate_setup's (id and description aside), checked again where the backend receives the setup,
    because a validator may map a setup to an alias. Returns the RobotSetup to run; after the world is built the
    backend checks that the operator built exactly that."""
    from robots.catalog import ENVIRONMENTS, resolve
    aliases = ENVIRONMENTS["forklift"]["aliases"]
    target = resolve(aliases[accepted]) if isinstance(accepted, str) and accepted in aliases else resolve(accepted)
    if isinstance(requested, str) and requested in aliases:
        return target
    own = resolve(requested)
    a, b = _built(own), _built(target)
    if a != b:
        differs = [f"{n}.{k}" for n in sorted(set(a["instances"]) | set(b["instances"]))
                   for k in sorted(set(a["instances"].get(n, {})) | set(b["instances"].get(n, {})))
                   if a["instances"].get(n, {}).get(k) != b["instances"].get(n, {}).get(k)]
        if a["attachments"] != b["attachments"]:
            differs.append("attachments")
        alias = next((k for k, v in aliases.items() if v == target.id), target.id)
        raise UnsupportedConfigurationError(
            f"forklift builds the catalogue rig {target.id!r} for this robot exactly; the setup {own.id!r} differs from "
            f"it in {', '.join(differs) or 'its instances'} and would be replaced by it, so it is refused. Select --robot "
            f"{alias} or pass the rig unchanged (robots/rigs/{target.id}.json)")
    return target


def _robot_notes(world):
    from forklift.operator import policy_notes
    return policy_notes(world, MIN_COMMAND_TICKS, MAX_COMMAND_S)


def _f(v):
    v = float(v)
    return v if math.isfinite(v) else None


def _scalars(d):
    """The JSON scalars of a telemetry dict (numbers as float, flags as bool, names as str)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            v = float(v)
            out[k] = v if math.isfinite(v) else None
        elif isinstance(v, str):
            out[k] = v
    return out


class _ForkliftTrace:
    """Privileged per-action physics trace in GAR's shared format (gym_anything_robotics.physics_trace), served by
    /v1/history/{id}/trace, /v1/history/{id}/trace/events and `gar export --include-traces`. Solver state, not sensor
    data (the pattern of sewing/trace.py).

    One sample per 0.5 ms tick of a traced command, built from what the world already holds after the tick (no extra
    ticks; nothing in the world is written): the truck (chassis pose x y z qx qy qz qw, forward speed, yaw rate, the
    controls' signals, the drivetrain, steering and mast hydraulics telemetry scalars, the mast, carriage, side-shift,
    steering-wheel and knuckle joints, every tyre's normal load), every load's pose and velocity (in the header's
    load_names order), and the robot (pelvis pose, motor joint positions and actuator torques in motor_names order).
    Events, from the environment's own per-tick latches (forklift/events.py): strike_begin / strike_end (mover,
    structure class, onset speed, peak force, impulse), wheel_lift_begin / wheel_lift_end, overturn, load_floor_contact,
    horn, intersection, seat_switch, fault. Every tick is sampled, so the store's completeness check holds; samples are
    buffered and written in one transaction per FLUSH_TICKS ticks or FLUSH_WALL_S of wall time."""

    FLUSH_TICKS = 200
    FLUSH_WALL_S = 1.0
    DISK_RESERVE = 512 * 1024 ** 2
    JOINTS = ("mast_outer", "mast_int", "carriage", "sideshift", "steering_wheel", "knuckle_l", "knuckle_r")

    def __init__(self, env, path, max_bytes=TRACE_BYTES):
        self.env = env
        import importlib.util
        # zstd where the runtime has it (the seeds' choice; it costs less per sample than zlib), else zlib
        self.store = PhysicsTraceStore(Path(path), max_bytes=max_bytes, max_items=max(100_000, max_bytes // 256),
                                       compression="zstd" if importlib.util.find_spec("zstandard") else "zlib",
                                       synchronous="NORMAL")
        self.current = None
        self._buffer, self._flushed = [], 0.0
        self._latch = None
        self.wall_s, self._error = 0.0, None

    def _layout(self):
        w = self.env.world
        loads = sorted(w.scene.loads)
        mo = w.operator.motors if w.rig is not None else None
        return dict(loads=loads, load_bodies=np.asarray([w.scene.loads[n]["body"] for n in loads], int),
                    joints={n: (w.q_of[n], w.dof_of[n]) for n in self.JOINTS if n in w.q_of},
                    motors=mo, pelvis=(w.operator.pelvis if mo is not None else None))

    def begin(self, context):
        if self.current is not None:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < self.DISK_RESERVE:
            raise OSError("Trace disk reserve unavailable; no action executed")
        w = self.env.world
        key = str(context["action_id"])
        self._map = self._layout()
        mo = self._map["motors"]
        self.store.begin(key, str(context["request_id"]), int(w.tick), float(w.config.dt), dict(
            source="forklift/newton1.5", generation=self.env.generation, task=f"{w.config.preset}/{w.config.task}",
            seed=w.config.seed, load_names=self._map["loads"], truck_joints=list(self._map["joints"]),
            motor_names=list(mo.names) if mo is not None else [],
            sampling="Every completed 0.5 ms physics tick of the command; no additional physics steps.",
            fields="t; truck: pose (x, y, z, qx, qy, qz, qw), speed_m_s (forward), yaw_rate_rad_s, signals (the "
                   "controls), drivetrain / steering / hydraulics (telemetry scalars), joints {name: [q, qd]}, "
                   "tyre_Fz_N; loads: pose (n x 7, load_names order), velocity_m_s (n x 3); robot: pelvis (7), q (rad), "
                   "torque_nm (motor_names order)",
            events="strike_begin, strike_end, wheel_lift_begin, wheel_lift_end, overturn, load_floor_contact, horn, "
                   "intersection, seat_switch, fault (forklift/events.py latches, contact sweeps every 4 ticks)",
            missing="No measured sensors; contact forces are MuJoCo's solved contact forces as the event latches "
                    "read them (every 4 ticks), not per-tick contact lists."))
        self.current = key
        self._latch = self._latches()
        self._buffer, self._flushed = [], time.perf_counter()
        self.wall_s, self._error = 0.0, None
        self.env.trace_callback = self._on_tick
        return dict(available=True, id=key, store_path=str(self.store.path),
                    note="Privileged solver trace (truck, loads, robot, per-tick event latches), not sensor data")

    # ------------------------------------------------------------------ per tick
    def _latches(self):
        ev = self.env.world.events
        return dict(active=set(ev._active), lift={k: v[0] > 0.0 for k, v in ev.wheel_lift.items()},
                    overturn=ev.overturn is not None, floor={k: len(v) for k, v in ev.load_floor.items()},
                    horn=len(ev.horn_presses), crossings=len(ev.crossings),
                    seat=bool(self.env.world.signals.get("seat_switch", True)))

    def _events(self, tick):
        w = self.env.world
        ev, old, t = w.events, self._latch, round(w.sim_time, 6)
        new = self._latches()
        out = []
        for key in sorted(new["active"] - old["active"]):
            t0, spd, mv = ev._onset.get(key, (t, 0.0, None))
            out.append(dict(type="strike_begin", tick=tick, t=t, mover=key[0], structure=key[1], body=mv,
                            onset_speed_m_s=float(spd)))
        for key in sorted(old["active"] - new["active"]):
            imp = (ev.strikes.get(key, {}).get("impacts") or [{}])[-1]
            out.append(dict(type="strike_end", tick=tick, t=t, mover=key[0], structure=key[1],
                            peak_N=float(imp.get("peak_N", 0.0)), impulse_Ns=float(imp.get("impulse_Ns", 0.0)),
                            started_t=imp.get("t"), parts=imp.get("parts")))
        for name in sorted(set(new["lift"]) | set(old["lift"])):
            a, b = old["lift"].get(name, False), new["lift"].get(name, False)
            if a != b:
                out.append(dict(type="wheel_lift_begin" if b else "wheel_lift_end", tick=tick, t=t, wheel=name,
                                **({} if b else {"duration_s": float(ev.wheel_lift.get(name, (0.0, 0.0))[1])})))
        if new["overturn"] and not old["overturn"]:
            out.append(dict(type="overturn", tick=tick, **{k: float(v) for k, v in ev.overturn.items()}))
        for name, n in new["floor"].items():
            for at, xy in ev.load_floor[name][old["floor"].get(name, 0):n]:
                out.append(dict(type="load_floor_contact", tick=tick, t=float(at), load=name, xy=[float(xy[0]), float(xy[1])]))
        for at in ev.horn_presses[old["horn"]:new["horn"]]:
            out.append(dict(type="horn", tick=tick, t=float(at)))
        for c in ev.crossings[old["crossings"]:new["crossings"]]:
            out.append(dict(type="intersection", tick=tick, zone=c["zone"], t=float(c["t"]), speed_m_s=float(c["speed"]),
                            since_horn_s=None if c["since_horn"] is None else float(c["since_horn"])))
        if new["seat"] != old["seat"]:
            out.append(dict(type="seat_switch", tick=tick, t=t, closed=new["seat"]))
        self._latch = new
        return out

    def _sample(self, tick):
        w = self.env.world
        m = self._map
        bq, bqd = w.state.body_q.numpy(), w.state.body_qd.numpy()
        q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
        c = bq[w.chassis].astype(float)
        qx, qy, qz, qw = c[3], c[4], c[5], c[6]           # body_q: x y z qx qy qz qw; fwd = the chassis' +x axis
        fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)])
        v = bqd[w.chassis].astype(float)
        lb = m["load_bodies"]
        truck = dict(pose=[_f(e) for e in c], speed_m_s=_f(v[:3] @ fwd), yaw_rate_rad_s=_f(v[5]),
                     signals=_scalars(w.signals), drivetrain=_scalars(w.drivetrain.tele),
                     steering=_scalars(w.steering.tele), hydraulics=_scalars(w.hydraulics.tele),
                     joints={n: [_f(q[i]), _f(qd[d])] for n, (i, d) in m["joints"].items()},
                     tyre_Fz_N={k: _f(t["Fz"]) for k, t in w.tyres.telemetry.items()})
        # every number above is a finite float or None (_f, _scalars): packed as is, without the store's per-value walk
        out = dict(tick=tick, t=round(w.sim_time, 6), generation=self.env.generation, truck=ValidatedJSON(truck),
                   loads=dict(pose=bq[lb].astype(np.float32), velocity_m_s=bqd[lb, :3].astype(np.float32)))
        mo = m["motors"]
        if mo is not None:
            out["robot"] = dict(pelvis=bq[m["pelvis"]].astype(np.float32), q=q[mo.q_idx].astype(np.float32),
                                torque_nm=mo.torques().astype(np.float32))
        return out

    def _on_tick(self, env):
        if self.current is None or self._error is not None:
            return
        t0 = time.perf_counter()
        try:
            tick = int(env.world.tick)
            self._buffer.append((tick, self._sample(tick), self._events(tick)))
            if len(self._buffer) >= self.FLUSH_TICKS or t0 - self._flushed > self.FLUSH_WALL_S:
                self._flush()
        except Exception as exc:
            # a recording failure (disk reserve, store) never faults the world: the command runs on unrecorded, and the
            # trace's summary says so (capture_errors, error, the missing ticks)
            self._error = f"tick {env.world.tick}: {type(exc).__name__}: {exc}"
            self._buffer = []
        self.wall_s += time.perf_counter() - t0

    def _flush(self):
        if self._buffer:
            if shutil.disk_usage(self.store.path.parent).free < self.DISK_RESERVE:
                raise OSError("Trace disk reserve reached during the command")
            self.store.append_batch(self.current, self._buffer)
            self._buffer = []
        self._flushed = time.perf_counter()

    def finish(self):
        if self.current is None:
            return None
        key, w = self.current, self.env.world
        self.env.trace_callback = None
        error = self._error
        try:
            tail = [] if error else self._events(int(w.tick))
            if w.fault:
                tail.append(dict(type="fault", tick=int(w.tick), fault=str(w.fault)))
            tail.append(dict(type="trace_cost", tick=int(w.tick), wall_s=round(self.wall_s, 4)))
            self._buffer.append((int(w.tick), None, tail))
            self._flush()
        except Exception as exc:                  # the header is closed below whatever happened to the last batch
            error = error or f"{type(exc).__name__}: {exc}"
        self.current = None
        self._buffer = []
        fault = w.fault or error
        self.store.finish(key, int(w.tick), status="failed" if fault else "completed", error=fault,
                          capture_errors=int(error is not None))
        return self.store.summary(key)

    def close(self):
        self.finish()
        self.store.close()


class NewtonForkliftBackend(RoboticsBackend):
    def __init__(self, *, robot=None, task="start-and-park", seed=0, preset=None, cameras=None, device=None, config=None,
                 output_dir=None, renderer_options=None, contract=None):
        from forklift.config import Config, TASKS, DEFAULT_ROBOT
        from forklift.env import ForkliftEnv
        presets = {t: p for p, ts in TASKS.items() for t in ts if p != "lab"}
        if task not in presets:
            raise UnsupportedConfigurationError(f"Unknown forklift task {task!r}; tasks: {sorted(presets)}")
        if preset is not None and preset != presets[task]:
            raise UnsupportedConfigurationError(f"task {task!r} belongs to preset {presets[task]!r}, not {preset!r}")
        from forklift.catalogue import CONTRACT, resolve_contract
        try:
            self.task_contract = resolve_contract(task, contract)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        robot = robot or DEFAULT_ROBOT
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["forklift"]
        if isinstance(robot, str) and robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(catalog["unsupported"][robot])
        from forklift.robot_setup import validate_setup, describe
        self.requested_robot = describe(robot)
        requested = robot
        try:
            robot = validate_setup(robot)
            # the setup the session runs, exactly: a caller's setup that the operator would replace is refused here,
            # before a world is built
            self.accepted_setup = _accepted_setup(requested, robot)
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        values = dict(config or {})
        if "extras" in values:
            raise UnsupportedConfigurationError("config 'extras' selects assisted probe fixtures and is not available "
                                                "through the policy interface")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise UnsupportedConfigurationError("seed must be an integer 0..999")
        values.update(preset=presets[task], task=task, seed=int(seed), robot=robot)
        if device is not None:
            values["device"] = device
        try:
            self.config = Config.from_dict(values)
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        if cameras is not None and (set(cameras) - {"head"} or len(set(cameras)) != len(tuple(cameras))):
            raise UnsupportedConfigurationError("Policy cameras are ('head',); the others are owner diagnostics")
        self.options = _check_renderer_options(renderer_options)
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output / SESSION_MARKER).write_text("forklift session (auto-created; pruned by retention)\n")
        self.env = ForkliftEnv(self.config, output_dir=self.output)
        w = self.env.world
        self._task_progress = None
        if self.task_contract == CONTRACT:
            from forklift.task_progress import TaskProgress
            self._task_progress = TaskProgress(w, w.scene.task_start)
            w.on_tick.append(self._task_progress.on_tick)
        built = getattr(w.scene, "robot_setup", None)
        if built is not None and _built(built) != _built(self.accepted_setup):
            raise UnsupportedConfigurationError(
                f"the operator built robot setup {built.id!r} ({built.fingerprint[:16]}) but the session accepted "
                f"{self.accepted_setup.id!r} ({self.accepted_setup.fingerprint[:16]}); refused rather than substituted")
        self.trace = None
        if self.options.get("trace", True):
            self.trace = _ForkliftTrace(self.env, self.output / "physics_trace.sqlite3",
                                        max_bytes=int(self.options.get("trace_bytes", TRACE_BYTES)))
        wd, ht = HEAD_CAMERA["width"], HEAD_CAMERA["height"]
        f = 0.018 / math.tan(math.radians(HEAD_CAMERA["hfov_deg"] / 2))
        # the policy camera's frame from the robot the session runs (the humanoid's head camera, the kit's cab camera)
        mount = f"{w.operator.handle.instance}/head_camera"
        self.camera_specs = {"head": CameraSpec("head", mount, wd, ht, wd, ht, (0.036, 0.036 * ht / wd), (f, f))}
        allowed = ("head",)
        self.cameras = tuple(cameras) if cameras is not None else allowed
        if set(self.cameras) - set(allowed) or len(set(self.cameras)) != len(self.cameras):
            raise UnsupportedConfigurationError(f"Policy cameras are {allowed}; the others are owner diagnostics")
        from forklift.render import DIAGNOSTIC_CAMERAS
        self.diagnostic = tuple(DIAGNOSTIC_CAMERAS)
        self.images = deque()
        self.last_timing = {}
        self._renderer = self._retained = self._timer = None
        self._prestart, self._closed, self._rendered = None, False, False
        self._scene_path = None
        self.render_meta = None
        self.frame_hz = CAMERA_HZ
        self._frame = None
        self._frame_images = None
        self._checkpoints = {}
        if self.cameras:
            self.env.frame_callback = self._clock
            self._capture(w)
        self.kinds = frozenset({"named_joint_target"})
        self.capabilities = Capabilities(
            "forklift", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, False, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes=self._notes())

    def _notes(self):
        c = self.config
        return {
            "task": f"{c.preset}/{c.task} seed {c.seed}",
            "clock": f"One tick = {c.dt * 1e3:g} ms; the clock advances only when a command runs",
            "truck": "The truck is driven only through its controls: steering wheel with spinner knob, inching, brake and "
                     "accelerator pedals, direction lever (F/N/R), lift/tilt/side-shift levers, parking-brake lever, key, "
                     "horn button. There is no speed display (speed is judged from the camera); no truck state is "
                     "observable",
            # control and cameras: from the robot the session runs (forklift/operator.py policy_notes)
            **_robot_notes(self.env.world),
            "wall_time": wall_time_note(c.robot),
            "episode": self._episode_note(),
            "qualification": "Assisted fixtures and scripted operators are not policy actions",
        }

    def _time_limit_s(self):
        from forklift.task import TIME_LIMIT_S
        return float(TIME_LIMIT_S[self.config.task])

    def _episode_note(self):
        """The task's simulated-time limit as the evaluation enforces it (the instruction does not state it)."""
        limit = self._time_limit_s()
        text = (f"The task's time limit is {limit:g} s of simulated time ({self.config.task}; forklift/task.py "
                "TIME_LIMIT_S), counted from the settled start; simulated time advances only while a command runs. "
                "The evaluation's time_limit criterion requires the task to be complete within it")
        if getattr(self, "_task_progress", None) is not None:
            text += (f"; under task contract {self.task_contract} the attempt ends at the limit: a command that "
                     "would run past it is cut there, the state at that moment is scored, and further commands are "
                     "refused")
        return text + "."

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.config.seed):
            raise ActionRejectedError("Select task/seed when creating the session; reset restores that settled initial state")
        with self.env.lock:
            state = self.env.reset()
            if getattr(self, "_task_progress", None) is not None:
                self._task_progress.reset()
            return state, {"continuity": "reset"}

    def _evaluate_task(self):
        adapter = getattr(self, "_task_progress", None)
        if adapter is not None:
            return adapter.evaluate(self.env.world)
        from forklift.evaluator import evaluate
        from forklift.catalogue import resolve_contract
        out = evaluate(self.env.world, self.env.world.scene.task_start)
        out["task_contract"] = resolve_contract(self.config.task)
        return out

    def get_state(self):
        """Owner state: the robot's joints and status, the truck (speed, controls' signals, drivetrain, steering,
        hydraulics, tyres, mast and steering joints), every load's pose, the latched events and the evaluation."""
        with self.env.lock:
            st = self.env.get_state()
            if self.env.world.fault is None:
                st["world"] = self.env.owner_state()
                st["world"]["events"].pop("traj", None)
                st["evaluation"] = self._evaluate_task()
            elif getattr(self, "_task_progress", None) is not None:
                st["evaluation"] = self._evaluate_task()
            return st

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault and getattr(self, "_task_progress", None) is None:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self._evaluate_task()

    # ------------------------------------------------------------------ rendering
    def _scene(self):
        from forklift.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env.world, self.output / "render")
        return self._scene_path

    def _worker(self):
        from forklift.render import Renderer
        if self._renderer is not None:
            return self._renderer
        if self._prestart is not None:
            holder, self._prestart = self._prestart, None
            t0 = time.perf_counter()
            holder["thread"].join()
            self.last_timing.update(cold_start_s=holder["start_s"], prestart_wait_s=time.perf_counter() - t0)
            if holder["renderer"] is not None:
                self._renderer = holder["renderer"]
                return self._renderer
            # the prestart failed: start a worker here, so that a lasting failure is this request's error
            self.last_timing["prestart_error"] = holder["error"]
        if self._retained is not None:
            self._renderer, self._retained = self._retained, None
            if self._timer is not None:
                self._timer.cancel()
            self.last_timing["retained_worker"] = True
            return self._renderer
        t0 = time.perf_counter()
        scene = self._scene()
        self._renderer = Renderer(self.output / "render", scene, samples=self.options.get("samples", POLICY_SAMPLES))
        self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return self._renderer

    def _idle_release_s(self):
        """Seconds the render worker stays warm after a request (renderer_options idle_release_s, default
        IDLE_RELEASE_S; 0: released after every request)."""
        v = self.options.get("idle_release_s")
        return float(IDLE_RELEASE_S if v is None else v)

    def _end_operation(self, final=False):
        idle = 0.0 if final else self._idle_release_s()
        r, self._renderer = self._renderer, None
        if r is None:
            return
        if idle > 0.0:
            self._retained = r
            self._timer = threading.Timer(idle, self._release, args=(r,))
            self._timer.daemon = True
            self._timer.start()
        else:
            threading.Thread(target=r.close, name="forklift-render-release", daemon=False).start()

    def _release(self, r):
        with self.env.lock:
            if self._retained is not r:
                return
            self._retained = None
        r.close()

    def _prestart_worker(self, ticks):
        """Start a new render worker beside this command's physics when the session's camera has been used (a served
        session observes at its start and after every command; a script that only acts never starts one), none is warm
        (the first command after the worker was released) and the command will cross a 30 Hz frame boundary, so that
        its observation needs a new frame (welding's backend does the same).
        The worker's start then overlaps the physics: only a command whose physics lasts as long as the start (71-99 s,
        COLD_WORKER_S) hides it; a short one still waits for nearly all of it. The observation's _worker()
        takes the started worker (or, if its start failed, starts one itself, so a lasting failure is that request's error)."""
        if not self.cameras or not self._rendered or self._renderer is not None or self._retained is not None \
                or self._prestart is not None:
            return
        w, fr = self.env.world, self._frame
        index = int((w.sim_time + ticks * self.config.dt) * self.frame_hz + 1e-6)
        if fr is not None and fr["generation"] == self.env.generation and index <= fr["index"] and \
                self._frame_images is not None and all(p.is_file() for p in self._frame_images.values()):
            return                                       # the observation reuses the current frame: no worker needed
        from forklift.render import Renderer
        scene, samples, folder = self._scene(), self.options.get("samples", POLICY_SAMPLES), self.output / "render"
        holder = dict(renderer=None, error=None, start_s=None)

        def start():
            t0 = time.perf_counter()
            try:
                holder["renderer"] = Renderer(folder, scene, samples=samples)
            except BaseException as exc:
                holder["error"] = f"{type(exc).__name__}: {exc}"
            holder["start_s"] = time.perf_counter() - t0
        holder["thread"] = threading.Thread(target=start, name="forklift-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    @staticmethod
    def _close_prestarted(holder):
        holder["thread"].join()
        if holder["renderer"] is not None:
            holder["renderer"].close()

    def _discard_prestart(self):
        """A command that failed: its prestarted worker is closed once started, without making the request wait."""
        holder, self._prestart = self._prestart, None
        if holder is not None:
            threading.Thread(target=self._close_prestarted, args=(holder,), name="forklift-render-release",
                             daemon=False).start()

    def _adopt_prestart(self):
        """A prestarted worker the observation did not need (its command ended inside the current frame, as after an
        early stop): once started it is kept warm like any worker after a request (idle release), and the request does
        not wait for its start."""
        holder, self._prestart = self._prestart, None
        if holder is None:
            return

        def adopt():
            holder["thread"].join()
            r = holder["renderer"]
            if r is None:
                return
            with self.env.lock:
                if not self._closed and self._renderer is None and self._retained is None and self._prestart is None:
                    self._renderer = r
                    self._end_operation()
                    return
            r.close()
        threading.Thread(target=adopt, name="forklift-render-adopt", daemon=True).start()

    def _clock(self, world):
        fr = self._frame
        if fr is None or fr["generation"] != self.env.generation or world.tick < fr["tick"] or \
                int(world.sim_time * self.frame_hz + 1e-6) > fr["index"]:
            self._capture(world)

    def _capture(self, world):
        from forklift import render as RD
        self._frame = dict(index=int(world.sim_time * self.frame_hz + 1e-6), t=float(world.sim_time), tick=int(world.tick),
                           generation=self.env.generation, state=RD.frame_state(world))
        self._frame_images = None

    def _render(self, names, frame=None):
        from forklift import render as RD
        with self.env.lock:
            try:
                r = self._worker()
                st = RD.frame_state(self.env.world) if frame is None else frame["state"]
                tick = self.env.world.tick if frame is None else frame["tick"]
                t0 = time.perf_counter()
                imgs = r.observe(list(names), st, tick, resolutions={n: self.render_meta["cameras"][n]["resolution"]
                                                                     for n in names})
                self.last_timing["render_s"] = time.perf_counter() - t0
                self._rendered = True
                frames = {}
                for name in names:
                    path = self.output / f"{name}_{self.env.generation}_{tick}_{uuid.uuid4().hex[:8]}.png"
                    path.write_bytes(imgs[name])
                    self.images.append(path)
                    frames[name] = path
            finally:
                self._end_operation()
            keep = set((self._frame_images or {}).values())
            while len(self.images) > 64 and self.images[0] not in keep:
                old = self.images.popleft()
                if old.parent != self.output:
                    raise RuntimeError("Image retention escaped the session folder")
                old.unlink(missing_ok=True)
            return frames

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.world.tick, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            meta, frames = {}, {}
            status = state["robot_status"]
            if self.cameras:
                fr = self._frame
                if fr is None or fr["generation"] != self.env.generation or fr["tick"] > self.env.world.tick:
                    self._capture(self.env.world)
                    fr = self._frame
                if self._frame_images is None or any(not p.is_file() for p in self._frame_images.values()):
                    self.last_timing.pop("render_s", None)
                    self._frame_images = self._render(self.cameras, fr)
                    reused = False
                else:
                    reused = True
                    if self._prestart is not None:
                        self._adopt_prestart()
                frames = self._frame_images
                meta = {"camera_frame": {"rate_hz": self.frame_hz, "index": fr["index"], "capture_time_s": fr["t"],
                                         "age_s": round(state["time_s"] - fr["t"], 6), "reused": reused}}
                cf = meta["camera_frame"]
                status = dict(status, camera_frame=dict(index=cf["index"], capture_time_s=cf["capture_time_s"],
                                                        age_s=cf["age_s"]))
            return Observation(sequence, state["time_s"], self.config.robot,
                               {"joints": state["joints"], "robot_status": status},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()}, meta)

    def render(self, *, label="interactive"):
        name = "overview" if label in ("interactive", "http") else label
        allowed = tuple(self.camera_specs) + self.diagnostic
        if name not in allowed:
            raise ValueError(f"Render label must be one of {allowed}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ actions
    def _plan(self, action):
        w = self.env.world
        if isinstance(action, NamedJointTarget):
            cmd = {}
            for key, v in action.targets_si.items():
                name, _, ch = key.partition(".")
                if ch and ch not in CHANNELS:
                    raise ValueError(f"Unknown channel {ch!r} in {key!r}; channels: {CHANNELS}")
                cmd[key] = float(v)
            return cmd, action.ticks
        raise UnsupportedActionError(f"Supported: {sorted(self.kinds)}")

    def validate_action(self, action):
        with self.env.lock:
            w = self.env.world
            try:
                cmd, ticks = self._plan(action)
                w.operator.motors.parse(cmd)
                if ticks > int(round(MAX_COMMAND_S / self.config.dt)):
                    raise ValueError("one command runs at most 30 s of simulated time")
                if ticks < MIN_COMMAND_TICKS:
                    raise ValueError(f"one command runs at least {MIN_COMMAND_TICKS} ticks (2 ms)")
            except (ValueError, KeyError) as exc:
                raise ActionRejectedError(str(exc)) from exc
            return cmd, ticks

    def act(self, action):
        with self.env.lock:
            w = self.env.world
            cmd, ticks = self.validate_action(action)
            requested_ticks = ticks
            adapter = getattr(self, "_task_progress", None)
            if adapter is not None:
                from forklift.task import TIME_LIMIT_S
                adapter.on_tick(w)
                if adapter.terminal is not None:
                    raise ActionRejectedError("This task attempt has ended; reset to start another")
                ticks = min(ticks, max(1, int(round((TIME_LIMIT_S[self.config.task] - w.sim_time) / self.config.dt))))
            before = w.tick
            self._prestart_worker(ticks)
            t0 = time.perf_counter()
            try:
                state = self.env.act(cmd, ticks)
            except (RuntimeError, FloatingPointError) as exc:
                self._discard_prestart()
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            except BaseException:
                self._discard_prestart()
                raise
            wall = time.perf_counter() - t0
            self.last_timing = {"physics_s": wall}
            result = (adapter.terminal or {}) if adapter is not None else {}
            return BackendStep(state, {"requested_ticks": requested_ticks, "executed_ticks": w.tick - before,
                                       "wall_per_sim": wall / max(1e-9, (w.tick - before) * self.config.dt)}, None,
                               bool(result.get("episode_done") and result.get("end_reason") != "time_limit"),
                               result.get("end_reason") == "time_limit")

    def policy_access(self):
        from forklift.task import instruction
        w = self.env.world
        mo = w.operator.motors
        proprio = {("joints", n, f): None for n in mo.names for f in ("position", "velocity", "torque")}
        proprio[("robot_status", "imu", "rpy_rad")] = 3
        proprio[("robot_status", "imu", "gyro_rad_s")] = 3
        if self.cameras:
            for k in ("index", "capture_time_s", "age_s"):
                proprio[("robot_status", "camera_frame", k)] = None
        motors = {n: dict(lower=float(mo.lower[i]), upper=float(mo.upper[i]), unit="radian",
                          effort_limit_nm=float(mo.effort[i]), velocity_limit_rad_s=float(mo.w_nl[i]),
                          kp_max=float(mo.kp_max[i]), kd_max=float(mo.kd_max[i]), group=mo.group[i])
                  for i, n in enumerate(mo.names)}
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio, self.kinds,
                            lambda: instruction(w.scene.task_start),
                            {"dt_s": self.config.dt, "min_command_ticks": MIN_COMMAND_TICKS,
                             "max_command_s": MAX_COMMAND_S,
                             # the episode's simulated-time budget, which the evaluation enforces (the instruction does
                             # not state it)
                             "time_limit_s": self._time_limit_s(), "episode": self._episode_note(),
                             "motors": motors, "channels": list(CHANNELS),
                             "control": self.capabilities.notes["control"], "truck": self.capabilities.notes["truck"],
                             "client_timeout_s": CLIENT_TIMEOUT_S, **self._runtime_hardware()})

    def _runtime_hardware(self):
        """The measured runtime of this session's robot (RUNTIME, TRACE; the note: wall_time_note)."""
        robot = self.config.robot
        r, t = RUNTIME.get(robot) or {}, TRACE.get(robot)
        return {
            "wall_time": wall_time_note(robot),
            "shortest_command_within_200x_s": r.get("shortest_command_within_200x_s"),
            # wall s per simulated s per request, by command length (s): median, p99, worst, requests over 200, sustained
            "wall_per_sim_per_request": r.get("per_step"),
            "camera_frame_render_s": r.get("frame_render_s") if self.cameras else None,
            "render_worker_idle_release_s": self._idle_release_s() if self.cameras else None,
            "render_worker_cold_start": (f"{COLD_WORKER_S} measured on a busy shared host (the first request after more "
                                         "than render_worker_idle_release_s without one; the new worker starts beside "
                                         "the command's physics, so only a command whose physics lasts that long hides "
                                         "it)") if self.cameras else None,
            "render_worker_gpu_mib": r.get("worker_gpu_mib") if self.cameras else None,
            "session_host_memory_mib": dict(pss=r.get("host_pss_mib"), rss=r.get("host_rss_mib"))
                                       if self.cameras and r else None,
            "physics_trace": dict(t, enabled=self.trace is not None, budget_bytes=int(self.options.get(
                "trace_bytes", TRACE_BYTES))) if t else None,
        }

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    # ------------------------------------------------------------------ owner extras
    def owner_state(self):
        with self.env.lock:
            return self.env.owner_state()

    def get_geometry_index(self):
        sc = self.env.world.scene
        return {"preset": self.config.preset, "task": self.config.task,
                "zones": {k: v for k, v in sc.zones.items() if k != "rack_locations"}}

    def get_geometry(self, geometry_id):
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def robot_configuration(self):
        w = self.env.world
        setup = getattr(w.scene, "robot_setup", None)
        out = dict(requested=self.requested_robot, effective=self.config.robot)
        if setup is not None:
            out.update(setup_id=setup.id, setup_fingerprint=setup.fingerprint)
        rig = getattr(w, "rig", None)
        if rig is not None:
            out["instances"] = {name: dict(model=r.model_id, joints=len(r.joints),
                                           controller_profile=r.metadata.get("controller_profile"))
                                for name, r in rig.instances.items()}
        return out

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints[label] = self.env.checkpoint()
            if getattr(self, "_task_progress", None) is not None:
                self._checkpoints[label]["task_progress"] = copy.deepcopy(self._task_progress.terminal)
            return {"label": label, "tick": self.env.world.tick, "generation": self.env.generation}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = self._checkpoints.get(label)
            if snap is None:
                raise KeyError(label)
            self.env.restore(snap)
            if getattr(self, "_task_progress", None) is not None:
                self._task_progress.terminal = copy.deepcopy(snap["task_progress"])
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.tick,
                                          "generation": self.env.generation, "continuity": "restored_discontinuity",
                                          "storage": "in_memory_session"}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["tick"]} for k, v in self._checkpoints.items())

    # ------------------------------------------------------------------ per-action physics trace
    def begin_action_trace(self, context):
        if self.trace is None:
            return {"available": False, "reason": "trace disabled (renderer_options trace=false)"}
        with self.env.lock:
            return self.trace.begin(context)

    def end_action_trace(self):
        if self.trace is None:
            return None
        with self.env.lock:
            return self.trace.finish()

    def close(self):
        with self.env.lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
            holder, self._prestart = getattr(self, "_prestart", None), None
            if holder is not None:                   # a worker still starting: closed once its start returns
                self._close_prestarted(holder)
            for r in (self._renderer, self._retained):
                if r is not None:
                    r.close()
            self._renderer = self._retained = None
            if getattr(self, "trace", None) is not None:
                self.trace.close()
                self.trace = None

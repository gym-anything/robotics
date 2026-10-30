"""Sewing bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 / MuJoCo arms + the cloth, the lockstitch
machine and its presser foot on the GPU).

The environment (sewing/env.py SewingEnv) owns the world, its clock, the start state, checkpoints and the evaluator;
this adapter selects the deployment, maps GAR robot actions to joint targets and the machine's control-box signals
(tool_io), renders the policy cameras through a retained Isaac render-only worker (released after idle_release_s) and
exposes the audited restricted interface. Render worker lifecycle as welding's and filleting's adapters (prestart
during physics, bounded idle retention).

Owner-side extras, as welding's adapter: a per-tick trace of every action
(sewing/trace.py; GAR's /v1/history/{id}/trace) and the variation catalogue, task x seed, switched in the session
(sewing/variations.py; /v1/variations, /v1/variations/start, or reset with options {"variation_id": N}).

State: GAR reads the state before and after every command (its action history) as well as for the command's result
and for an observation; the backend builds SewingEnv's state, with its evaluation, once per tick and hands out copies.
A batch of commands is validated whole before its first command runs.
"""

from collections import deque
import copy
import gc
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, ToolIO, RawAction
from ..errors import (ActionRejectedError, BackendProtocolError, UnsupportedActionError,
                      UnsupportedConfigurationError)
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

# The render worker is kept warm this long after a request (renderer_options idle_release_s; 0 releases it after every
# request). A policy that thinks longer than this pays a fresh worker on its next request, 6.9-7.8 s more on every
# deployment, so a 50 ms command after a 15 s pause runs at 204-232x on a quiet host. A policy's think time (an LLM or
# VLA choosing its next action) is seconds to minutes; 10 minutes covers it, as in welding's adapter, and a session left
# idle longer gives the worker's ~4.3 GB of GPU memory back to the shared GPU.
IDLE_RELEASE_S = 600.0
E16_DEPLOYMENTS = (("dual-fr3", "FR3 pair"), ("dual-panda", "Panda pair"), ("single-fr3", "single FR3"), ("g1", "G1"))


def e16_result_paths(tag=""):
    """sewing/data/runtime/e16_runtime_api[_ROBOT][_TAG].json of the sewing package this process imports, by
    deployment (None if the package is not found); tag "quiet": the runs on a quiet host."""
    import importlib.util
    spec = importlib.util.find_spec("sewing")
    if spec is None or not spec.submodule_search_locations:
        return None
    d = Path(list(spec.submodule_search_locations)[0])/"data"/"runtime"
    suffix = f"_{tag}" if tag else ""
    return {r: d/(("e16_runtime_api" if r == "dual-fr3" else f"e16_runtime_api_{r}") + suffix + ".json")
            for r, _ in E16_DEPLOYMENTS}


def e16b_result_paths():
    """sewing/data/runtime/e16b_pause[_ROBOT].json (E16b: commands after pauses), by deployment; None without the
    package."""
    p = e16_result_paths()
    return None if p is None else {r: path.parent/(("e16b_pause" if r == "dual-fr3" else f"e16b_pause_{r}") + ".json")
                                   for r, path in p.items()}


def wall_time_figures(records, quiet=None, pause=None):
    """The numbers WALL_TIME_NOTE quotes, derived from E16's result files (one per deployment; quiet: the same runs on
    a quiet host, if recorded; pause: E16b's records of commands sent after pauses, if recorded). The note is built
    from those files when this module is imported, so new result files update what a policy reads."""

    def rng(values, nd=0):
        lo, hi = min(values), max(values)
        a, b = f"{lo:,.{nd}f}", f"{hi:,.{nd}f}"
        return a if a == b else f"{a}-{b}"

    def ratios(e, prefix="", within=""):
        return [r["wall_s"]/r["sim_s"] for r in e["rows"]
                if r["step"].startswith(prefix) and within in r["step"] and r["sim_s"]]

    out = {}
    for robot, _ in E16_DEPLOYMENTS:
        e = records[robot]
        if e.get("robot") != robot:
            raise ValueError(f"E16's {robot} file records {e.get('robot')}")
        out[f"{robot}_sewing"] = rng(ratios(e, "sewing"))
        out[f"{robot}_approach"] = rng(ratios(e, within="(machine idle)"))
    every = [records[r] for r, _ in E16_DEPLOYMENTS]
    short = sorted({float(L) for e in every for L in e["by_command_length"]})
    shortest = [e["shortest_command_inside_200x_s"] for e in every]
    out["shortest_s"] = "none" if None in shortest else f"{max(shortest):g}"
    out["tick_cmd_s"] = f"{short[0]:g}"
    out["shortest_cmd"] = rng([e["by_command_length"][f"{short[0]:g}"]["median"] for e in every])
    out["second_cmd_s"] = f"{short[1]:g}"
    out["second_cmd"] = rng([e["by_command_length"][f"{short[1]:g}"]["median"] for e in every])
    out["observation_s"] = rng([w for e in every for w in e["observation_only_wall_s"]], 1)
    out["tool_io_s"] = rng([w for e in every for w in e["one_tick_tool_io_wall_s"]], 1)
    out["worst"] = f"{max(e['worst_action_wall_per_sim'] for e in every):.0f}"
    out["load"] = rng([r["host"]["loadavg"][0] for e in every for r in e["rows"] if "host" in r])
    out["launch_s"] = rng([e["launch_s"] for e in every])
    if quiet:
        for robot, _ in E16_DEPLOYMENTS:
            e = quiet[robot]
            if e.get("robot") != robot:
                raise ValueError(f"E16's quiet {robot} file records {e.get('robot')}")
            out[f"quiet_{robot}_sewing"] = rng(ratios(e, "sewing"))
        qs = [quiet[r]["shortest_command_inside_200x_s"] for r, _ in E16_DEPLOYMENTS]
        out["quiet_shortest_s"] = "none" if None in qs else f"{max(qs):g}"
        out["quiet_load"] = rng([r["host"]["loadavg"][0] for r_, _ in E16_DEPLOYMENTS for r in quiet[r_]["rows"]
                                 if "host" in r])
        qpause = [quiet[r].get("pause_beyond_release_adds_s") for r, _ in E16_DEPLOYMENTS]
        if None not in qpause:
            out["pause_adds_s"] = rng(qpause, 1)
        every = every + [quiet[r] for r, _ in E16_DEPLOYMENTS]
    if pause:
        for robot, _ in E16_DEPLOYMENTS:
            if pause[robot].get("robot") != robot:
                raise ValueError(f"E16b's {robot} file records {pause[robot].get('robot')}")
        out["pause_shared_s"] = rng([pause[r]["pause_beyond_release_adds_s"] for r, _ in E16_DEPLOYMENTS], 1)
        inside = [pause[r]["pause_inside_release_adds_s"] for r, _ in E16_DEPLOYMENTS]
        out["pause_inside_s"] = f"{min(inside):+.1f} to {max(inside):+.1f}"
        out["pause_load"] = rng([x["host"]["loadavg"][0] for r, _ in E16_DEPLOYMENTS for x in pause[r]["rows"]
                                 if "host" in x])
        # E16b records whose session was launched with a short release and paused on both sides of it say so
        # (idle_release_s, pauses_s) and carry what a new render process cost an observation. Older records have none
        # of these fields: their sessions kept the default release and paused 5 s and 15 s, both inside it
        if all("idle_release_s" in pause[r] for r, _ in E16_DEPLOYMENTS):
            out["pause_release_s"] = rng([pause[r]["idle_release_s"] for r, _ in E16_DEPLOYMENTS])
            out["pause_short_s"] = rng([pause[r]["pauses_s"]["inside_release"] for r, _ in E16_DEPLOYMENTS])
            out["pause_long_s"] = rng([pause[r]["pauses_s"]["beyond_release"] for r, _ in E16_DEPLOYMENTS])
            out["fresh_render_s"] = rng([pause[r]["fresh_render_process_s"] for r, _ in E16_DEPLOYMENTS], 1)
        every = every + [pause[r] for r, _ in E16_DEPLOYMENTS]
    hashes = {e.get("physics_hash") for e in every}
    out["physics_hash"] = hashes.pop() if len(hashes) == 1 else "differs between deployments"
    return out


def _wall_time_note(f):
    # what the pause records hold. Newer E16b records measure a release (a session launched with a short
    # idle_release_s, paused on both sides of it); older records and E16's own pauses, 5 s and 15 s, lie inside the
    # default release and are quoted as that
    measured = "pause_release_s" in f
    pauses = []
    if measured:
        pauses.append(f"a 0.05 s command sent {f['pause_long_s']} s after the last, beyond the release, took "
                      f"{f['pause_shared_s']} s longer than one sent at once on the shared host (E16b, "
                      f"a session launched with idle_release_s {f['pause_release_s']}, "
                      f"load {f['pause_load']}; after a {f['pause_short_s']} s pause, inside the release, "
                      f"{f['pause_inside_s']} s), and an observation alone {f['fresh_render_s']} s longer (the new "
                      f"render worker's start, part of which a command's physics hides)")
    elif "pause_shared_s" in f:
        pauses.append(f"a 0.05 s command sent 15 s after the last took {f['pause_shared_s']} s longer than one sent at "
                      f"once on the shared host (E16b, load {f['pause_load']}; after a 5 "
                      f"s pause, inside the release, {f['pause_inside_s']} s)")
    if "pause_adds_s" in f:
        inside = ", inside the default release," if measured else ""
        pauses.append((f"{f['pause_adds_s']} s longer on the quiet host" if pauses and not measured else
                       f"a 0.05 s command sent 15 s after the last{inside} took {f['pause_adds_s']} s longer on the "
                       f"quiet host"))
    pause_text = (("; " if measured else " and ").join(pauses))
    pause_text = pause_text[:1].upper() + pause_text[1:] if measured else pause_text
    return (
        f"Measured through gar run + HTTP with the policy key and each deployment's policy cameras on the shared RTX "
        f"5090 host (physics {f['physics_hash']}): the plain "
        f"seam sewn at 600 sti/min while one hand travels with the work. Wall seconds per simulated second for 0.5 s "
        f"actions while sewing: FR3 pair {f['dual-fr3_sewing']}, Panda pair {f['dual-panda_sewing']}, single FR3 "
        f"{f['single-fr3_sewing']}, G1 {f['g1_sewing']}; the hand's approach with the machine idle: "
        f"{f['dual-fr3_approach']}, {f['dual-panda_approach']}, {f['single-fr3_approach']} and {f['g1_approach']}. "
        f"The host's 1-minute load average during the runs was {f['load']} on 16 CPUs and other jobs used the GPU; "
        f"the ratios depend on both: the cloth's step is a GPU job, and every further simulation on the same card "
        f"slows it in proportion (several at once can put any command outside the 200x limit). Every "
        f"request pays a fixed cost (the observation's camera images and the server's handling; the state and the "
        f"evaluation take milliseconds, tens of milliseconds on the load task): an observation alone "
        f"took {f['observation_s']} s and a one-tick tool_io {f['tool_io_s']} s, so short commands cost more per "
        f"simulated second: "
        + (f"the shortest command length that stayed within the 200x limit on every deployment was {f['shortest_s']} s"
           if f["shortest_s"] != "none" else "no command length stayed within the 200x limit on every deployment")
        + f" ({f['second_cmd_s']} s commands ran at a median {f['second_cmd']}x, {f['tick_cmd_s']} s commands at "
        f"{f['shortest_cmd']}x); the worst command of 0.3 s or longer ran at {f['worst']}x. Startup "
        f"{f['launch_s']} s in these runs, longer the first time after a physics change (the start state is prepared "
        f"then). "
        + (f"On a quiet host (each run started after three minutes of 1-minute load under 4 and GPU use under 30 %; "
           f"load {f['quiet_load']} during the runs) the same runs took FR3 pair "
           f"{f['quiet_dual-fr3_sewing']}, Panda pair {f['quiet_dual-panda_sewing']}, single FR3 "
           f"{f['quiet_single-fr3_sewing']}, G1 {f['quiet_g1_sewing']} while sewing, and "
           + (f"the shortest command length within 200x on every deployment was {f['quiet_shortest_s']} s. "
              if f["quiet_shortest_s"] != "none" else "no command length stayed within 200x on every deployment. ")
           if "quiet_g1_sewing" in f else "")
        + (f"The render worker is kept for renderer_options idle_release_s (default {IDLE_RELEASE_S:g} s) after a "
           f"request; the first request after a longer pause starts a new one. "
           + ("" if measured else "The recorded pauses, 5 s and 15 s, are both inside the default release, so these "
                                  "records do not hold what a new render worker costs: ")
           + pause_text
           + f". A client that pauses longer than the default between requests can raise idle_release_s, and one "
           f"that wants the worker's GPU memory back sooner can lower it (gar run --backend-options-json "
           f"'{{\"renderer_options\": {{\"idle_release_s\": {3*IDLE_RELEASE_S:g}}}}}'; 0 releases the worker after "
           f"every request); the worker keeps its GPU memory while it waits. "
           if pauses else "")
        + "A command carries at most max_command_s of simulated time. Client timeout: client_timeout_s")


WALL_TIME_FALLBACK = (
    "The runtime figures are built from E16's result files (sewing/data/runtime/), which this installation does not "
    "have or could not read: see sewing/REPORT.pdf. The cloth's implicit step runs at tens to hundreds of "
    "wall seconds per simulated second depending on the host's load, and every request adds a fixed cost (the "
    "observation's images), so short commands cost far more wall time per simulated second than long ones. A command "
    "carries at most max_command_s of simulated time. Client timeout: client_timeout_s")


def load_wall_time_note(paths=None, quiet_paths=None, pause_paths=None):
    """The policy's runtime note from E16's result files (by default the package's; quiet_paths defaults to the
    package's quiet-host records only when paths does too); WALL_TIME_FALLBACK if any shared-host record is missing or
    unreadable. The quiet-host records are optional: the note quotes them only when all four are readable."""
    import json
    errors = (AttributeError, OSError, TypeError, KeyError, IndexError, ValueError, ZeroDivisionError)
    default = paths is None
    try:
        paths = e16_result_paths() if default else paths
        records = {r: json.loads(Path(paths[r]).read_text()) for r, _ in E16_DEPLOYMENTS}
        figures = wall_time_figures(records)
    except errors:
        return WALL_TIME_FALLBACK
    quiet = pause = None
    try:
        if quiet_paths is None and default:
            quiet_paths = e16_result_paths("quiet")
        if quiet_paths:
            quiet = {r: json.loads(Path(quiet_paths[r]).read_text()) for r, _ in E16_DEPLOYMENTS}
            figures = wall_time_figures(records, quiet)
    except errors:
        quiet = None
    try:
        if pause_paths is None and default:
            pause_paths = e16b_result_paths()
        if pause_paths:
            pause = {r: json.loads(Path(pause_paths[r]).read_text()) for r, _ in E16_DEPLOYMENTS}
            figures = wall_time_figures(records, quiet, pause)
    except errors:
        pass
    return _wall_time_note(figures)


WALL_TIME_NOTE = load_wall_time_note()
# every policy camera is rendered by default (the launch's `cameras` option selects fewer)
POLICY_CAMERAS = ("operator", "overhead", "needle")
G1_POLICY_CAMERAS = ("head_rgb", "overhead", "needle")      # the humanoid looks through its head D435 (sewing/g1.py)
DIAGNOSTIC = ("hero", "macro", "seam", "machine")
MOUNTS = {"operator": "external_operator_mount", "overhead": "cell_overhead_d435_boom",
          "needle": "machine_face_plate_mount", "head_rgb": "g1_head_d435_mount"}
# the longest command one request may carry (simulated seconds): at the worst ratio E16 measured on the shared host
# (WALL_TIME_NOTE) a 2 s command returns inside the declared 900 s client timeout; an uncapped one could outlast it
# while the server kept stepping
MAX_COMMAND_S = 2.0
CLIENT_TIMEOUT_S = 900
RENDERER_OPTIONS = frozenset({"width", "height", "samples", "idle_release_s"})
POLICY_SAMPLES = 8
TOP_LEVEL = ("preset", "task", "seed", "robot", "contract")
TRACE_BYTES = 512*1024*1024     # the per-action trace's store (backend option trace_bytes; retention evicts the oldest)
# where checkpoints live and what survives
CHECKPOINT_SCOPE = ("In memory, in this server process only: a checkpoint holds the whole world (cloth, machine and "
                    "its stitch ledger, arms, the episode clock and histories) and can be restored any number of times "
                    "within the session and the variation it was saved in (a reset keeps them). A server restart, a "
                    "session close and a variation switch (/v1/variations/start, or reset with variation_id) drop "
                    "every checkpoint; nothing is written to disk.")
SESSION_MARKER = ".gar-sewing-session"
SESSION_RETENTION = 20
# the machine's control box as wired to the cell controller (an industrial head's external I/O: pedal speed command,
# needle up/down stop, presser lift, touch-back reverse, thread trimmer) [read: Juki DDL-9000C control box functions]
TOOL_OUTPUTS = {
    "speed_spm": dict(kind="number", range=(0.0, 5000.0), unit="sti/min",
                      note="pedal speed command; 0 stops at the selected needle position (<= 4000 above 4 mm stitch). "
                           "The model is converged in time at 600 sti/min; at 5,000 the outcome of unguided sewing "
                           "depends on the solver's tick, and the speeds between are not qualified"),
    "needle_stop": dict(kind="choice", choices=("up", "down"), note="needle position the machine stops at"),
    "presser": dict(kind="choice", choices=("down", "lift1", "lift2"),
                    note="presser foot: down, or lifted by the auto-lifter 5.0 / 8.5 mm (travels at ~0.2 m/s)"),
    "reverse": dict(kind="bool", note="touch-back switch: reverse feed while held (switches at the next feed stroke)"),
    "trim": dict(kind="bool", note="thread trimmer: stops needle-up and cuts both threads, leaving 3 mm tails"),
}
TOOL_FEEDBACK = ("shaft_angle_rad", "speed_spm", "needle_down", "presser_down", "reverse", "running", "fault",
                 "stitch_counter")


def _session_folder():
    import os
    import shutil
    import tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"sewing-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


_ENGINE_MODULES = ("warp", "mujoco", "newton", "newton.solvers", "newton._src.solvers.mujoco.constants", "robots",
                   "robots.catalog", "robots.backends.newton", "sewing.world", "sewing.env", "sewing.task",
                   "sewing.variants", "sewing.kinematics", "sewing.g1", "sewing.trace", "sewing.variations")
_engine_imported = False


def _import_engine():
    """Import the engine and the sewing modules once per process, from a short-lived thread of their own, before the
    first world is built. Newton's USD module keeps the ImportError of its optional newton_usd_schemas import
    (newton/_src/usd/__init__.py), and that traceback's frames reach back through every frame on the stack of the
    first `import newton`: imported inside World.__init__ (sewing.world.require_engine), it held the session's first
    SewingEnv, and the backend constructor's frame with its first world, for the life of the process, so the first
    variation switch could not release the old world (measured on rigid-only worlds: the launch environment stayed
    alive after three switches while each switched one was freed). A thread's stack
    ends at its own entry, so nothing of a session is reachable from frames kept during these imports."""
    global _engine_imported
    if _engine_imported:
        return
    import importlib
    box = {}

    def work():
        try:
            for name in _ENGINE_MODULES:
                importlib.import_module(name)
            importlib.import_module("sewing.world").require_engine()
        except BaseException as exc:               # raised in the caller (its type kept)
            box["error"] = exc
    t = threading.Thread(target=work, name="sewing-engine-import", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    _engine_imported = True


def _check_renderer_options(options):
    """The renderer options of a launch, checked before anything is built (as welding's adapter does)."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("renderer_options must be an object")
    if set(options) - RENDERER_OPTIONS:
        raise UnsupportedConfigurationError(f"renderer_options accepts {sorted(RENDERER_OPTIONS)}")
    for key, (lo, hi) in (("width", (16, 7680)), ("height", (16, 4320)), ("samples", (1, 256))):
        v = options.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi):
            raise ValueError(f"renderer_options.{key} must be an integer in [{lo}, {hi}]")
    v = options.get("idle_release_s")
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
        raise ValueError("renderer_options.idle_release_s must be a number of seconds >= 0 (0: release the render "
                         "worker after every request)")
    return {k: v for k, v in options.items() if v is not None}


def _check_robot(robot):
    from robots.catalog import ENVIRONMENTS
    catalog = ENVIRONMENTS["sewing"]
    qualified = tuple(catalog["aliases"])
    if isinstance(robot, dict):
        from sewing.robot_setup import validate_setup
        try:
            return validate_setup(robot)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
    if robot in catalog.get("unsupported", {}):
        raise UnsupportedConfigurationError(f"sewing does not support {robot!r}: {catalog['unsupported'][robot]}")
    if robot not in qualified:
        raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
    return robot


class NewtonSewingBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset=None, task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, trace=True, trace_bytes=None, contract=None):
        if not isinstance(trace, bool):
            raise ValueError("trace must be true or false")
        if trace_bytes is not None and (isinstance(trace_bytes, bool) or not isinstance(trace_bytes, int)
                                        or trace_bytes < 64*1024*1024):
            raise ValueError("trace_bytes must be an integer >= 64 MiB")
        self.options = _check_renderer_options(renderer_options)
        _import_engine()
        robot = _check_robot(robot)
        self.g1 = robot == "g1"
        self.policy_cameras = G1_POLICY_CAMERAS if self.g1 else POLICY_CAMERAS
        cameras = self.policy_cameras if cameras is None else cameras
        values = dict(config or {})
        clash = sorted(set(TOP_LEVEL) & set(values))
        if clash:
            raise UnsupportedConfigurationError(f"{clash} are top-level options, not config fields")
        if device is not None and "device" in values:
            raise UnsupportedConfigurationError("Device supplied at two levels; choose exactly one")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(self.policy_cameras):
            raise ValueError(f"Policy cameras are {self.policy_cameras}; {DIAGNOSTIC} are privileged diagnostics")
        from sewing.variants import check_robot
        from sewing.task_set import resolve, choose_contract
        from sewing.world import Config
        from sewing.task_set_env import make_environment
        try:
            chosen = resolve(preset, task, seed, contract)
            self.task_contract = choose_contract(chosen["task"], contract)
        except KeyError as exc:
            raise ValueError(f"Unknown preset {preset!r}; presets: sample-room") from exc
        try:
            check_robot(robot, chosen["task"])
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        unknown = sorted(set(values) - set(Config.__dataclass_fields__))
        if unknown:
            raise UnsupportedConfigurationError(f"Unknown config fields {unknown}")
        # kept for a variation switch, which builds the new task and seed's Config the way this constructor does
        self._config_overrides, self._device = dict(values), device
        values.update(chosen)
        values["robot"] = robot
        if device is not None:
            values["device"] = device
        self.config = Config(**{k: v for k, v in values.items() if k in Config.__dataclass_fields__})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("sewing session (auto-created; pruned by retention)\n")
        self.env = make_environment(self.config, contract=self.task_contract, output_dir=self.output)
        from sewing import variations as VR
        self.variation = VR.specification(VR.index_of(self.config.task, self.config.seed), self.task_contract)
        # the per-action trace's store is opened at the first traced action (sewing/trace.py)
        self.trace, self._trace_bytes = None, (int(trace_bytes or TRACE_BYTES) if trace else None)
        w = self.env.world
        # arm names of the deployment: the FR3/Panda instances, or the G1's two hands (with the waist leaning: +waist)
        self.arms = ("left", "right", "left+waist", "right+waist") if self.g1 else tuple(w.rig.instances)
        # what GAR's action decoding accepts as `arm` besides 'left' and 'right' (gym_anything_robotics/actions.py
        # arm_names)
        self.extra_arm_names = tuple(a for a in self.arms if a not in ("left", "right"))
        self._state_cache = None
        self._last_valid_evaluation = None
        self.images = deque()
        self.last_timing = {}
        self._prestart = None
        self._retained = None
        self._retain_timer = None
        self._reapers = []
        self._checkpoints = {}
        self._scene_path = None
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        from sewing import layout as LY
        self.camera_specs = {}
        for name in self.policy_cameras:
            if name in LY.CAMERAS:
                fov = float(LY.CAMERAS[name]["hfov"])
            else:
                from sewing.g1 import HEAD_CAMERA
                fov = float(HEAD_CAMERA["hfov_deg"])
            focal = .018/math.tan(math.radians(fov/2))
            self.camera_specs[name] = CameraSpec(name, MOUNTS[name], width, height, 1280, 720,
                                                 (.036, .036*height/width), (focal, focal))
        self.torque_joints = tuple(n for dofs in getattr(w, "_arm_mj_dofs", {}).values() for n, _ in dofs)
        # "raw" carries the arms' joint impedance (op joint_impedance): the FR3 and the
        # Panda are torque-controlled, and libfranka sets each joint's impedance; the G1 is not offered it
        self.kinds = frozenset({"named_joint_target", "cartesian_pose", "cartesian_delta", "tool_io",
                                *(() if self.g1 else ("gripper", "raw"))})
        hand = w.hand_arms[0] if w.hand_arms and w.gripper_joints(w.hand_arms[0]) else None
        self.finger_effort_limit = float(w.joints[w.gripper_joints(hand)[0]]["effort_limit"]) if hand else None
        c = self.config
        from sewing.task_set import instruction
        self.capabilities = Capabilities(
            "sewing", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[n] for n in self.cameras), notes={
                "task": self._task_note(),
                "instruction": instruction(c.task, self.config.robot, self.task_contract),
                "clock": f"One tick = {c.dt*1e3:g} ms for the arms (MuJoCo), the cloth (implicit step) and the machine",
                "control": ("Physical position servos (the standing G1 profile: legs kp 800, waist and arms 120, "
                            "wrists 80, Dex3 fingers 1.8) with a gravity feed-forward on the waist and arms and a "
                            "rate-limited motion generator; the robot stands on its servos (no balance controller); IK "
                            "on the 7 arm joints (arm 'left'/'right') or with the 3 waist joints ('left+waist'/"
                            "'right+waist'), targets only, no collision-free planning. No protective stop: the G1's "
                            "low-level joint interface has no collision or velocity reflex" if self.g1 else
                            "Physical position servos with a rate-limited motion generator (joint speed and "
                            "acceleration limits); IK computes targets only (no collision-free planning); an "
                            "unreachable IK target or out-of-limit joint target is rejected before any tick. Protective "
                            "stop: libfranka's default 10/20 N m collision thresholds (contacts and the cloth's "
                            "reactions on the hands), and a joint over its speed limit"),
                "compliance": ("not offered: the G1's low-level gains have no sourced range in this cell" if self.g1
                               else {"action": {"type": "raw", "request": {"op": "joint_impedance", "arm": "left",
                                                                           "stiffness_nm_per_rad": [600, 600, 600, 600,
                                                                                                    250, 150, 50],
                                                                           "ticks": 1}},
                                     "stiffness_nm_per_rad": "a number or one per arm joint (7), each in [0, 14250] "
                                                             "(libfranka Robot::setJointImpedance); null restores the "
                                                             "position servo",
                                     "damping_nms_per_rad": "optional, a number or 7; default keeps the damping "
                                                            "ratios of libfranka's joint impedance example",
                                     "ticks": "physics ticks to run after the change, holding the joint targets "
                                              "(default 1)",
                                     "law": "tau = K (q_target - q) - D dq per joint, the arm's gravity compensated as "
                                            "the robot's controller does; joint targets are commanded as before and "
                                            "the motion generator still rate-limits them"}),
                "machine_io": {"arm": "any arm of the deployment (the cell controller's I/O channel)",
                               "outputs": {k: dict(v) for k, v in TOOL_OUTPUTS.items()},
                               "feedback": list(TOOL_FEEDBACK),
                               "note": "the machine's control box; its needle, feed dog, presser foot, hook and "
                                       "trimmer act on the cloth only through the physics"},
                "arms": ",".join(self.arms),
                "gripper": ("Dex3-1 hands: 7 joints per hand commanded by name (g1/<side>_hand_<joint>_joint); no "
                            "gripper action" if self.g1 else
                            (f"Franka Hand per arm: half_width_m position mode or inward_effort_n force mode (<= "
                             f"{2*self.finger_effort_limit:g} N total). Force mode is the hand's grasp: the "
                             f"fingers close at 50 mm/s until something stops them, then squeeze with the commanded "
                             f"force per finger. On cloth each finger's pads are solved with the cloth, sprung to the "
                             f"finger along its closing axis (the fingertip's compliance), so the squeeze settles at "
                             f"the command (2-35 N per finger); position mode squeezes what "
                             f"the servo's stiffness gives at the plies' thickness (~1.2 N). A finger outside its "
                             f"travel ends the episode as a world fault") if hand else "no hand"),
                "physics": "Woven plies as orthotropic shells (toe-region tension law, shear, bending, each with "
                           "the fabric's measured hysteresis) in an implicit step with contact (fabric compression), "
                           "Coulomb friction with static stick (elastic-plastic, 0.1 mm pre-slip) and the air under the "
                           "cloth; each hand's pads on compliant bodies solved with the cloth (vertical; the fingers' closing axes); the presser foot a "
                           "body pressing 40 N; the drop feed's four-motion dog; needle puncture, deflection and plate "
                           "strikes; loop pickup and stitch formation from the simulated state; thread elements at "
                           "their set tension",
                "evaluator": "Physical-state evaluator (sewing/task.py, self-tested on fixtures): the stitch ledger "
                             "in the plies' material coordinates, the plies' registration where they lie stacked, "
                             "seam-band heights and the seam's length along its path",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise",
                "force": ("No force or tactile channel: the Dex3's fingertip tactile sensors are not simulated, and "
                          "the G1's joint torque estimates are not exposed" if self.g1 else
                          "Each arm joint's estimated external torque (proprioception joints.<arm>/<joint>."
                          "external_torque, N m): the torque of the contacts on the arm and the cloth's reaction on "
                          "the hand, as libfranka reports tau_ext_hat; the collision reflex judges the same quantity"),
                "wall_time": WALL_TIME_NOTE,
                "qualification": "Assisted fixtures are not policy actions",
                "trace": ("Owner: every action records each 0.5 ms tick (joints and servo set points, reflexes, "
                          "external torques, the cloth's force on the hands, the machine: shaft, needle, foot, "
                          "stitches, threads) and its events (sewing/trace.py; the cloth's vertices are not sampled); "
                          "GET /v1/history/{id}/trace" if trace else "disabled (trace=false)"),
                "variations": f"Owner: catalogue {VR.VERSION} ({VR.COUNT} entries, task x seed; sewing/variations.py); "
                              "GET /v1/variations, POST /v1/variations/start or reset with options {\"variation_id\": "
                              "N}; a switch took 4.6 s with the start state cached and no cameras (18.5-22.6 s "
                              "with the three policy cameras: the render worker "
                              "is restarted on the new scene) and 208-418 s when it was prepared (the FR3 pair on the "
                              "shared GPU; GET /v1/variations: note)",
                "checkpoints": CHECKPOINT_SCOPE})

    # ------------------------------------------------------------------ lifecycle
    def _task_note(self):
        c = self.config
        return (f"{c.preset}/{c.task} seed {c.seed} (variation {self.variation['variation_id']}); start state: "
                f"{self.env.start_state}")

    def reset(self, *, seed=None, options=None):
        if options and "variation_id" in options and set(options) <= {"variation_id", "contract"} and seed is None:
            return self._start_variation(options["variation_id"], options.get("contract"))
        if options and "variation_id" in options:
            raise ActionRejectedError("Reset options: variation_id and optional contract (the catalogue, GET /v1/variations); do not "
                                      "combine a seed or other options; nothing changed")
        if options or (seed is not None and seed != self.config.seed):
            # a refusal that changed nothing: the session stays ready (a plain ValueError would latch a runtime fault
            # and the session would then refuse every command until the owner recovered it)
            raise ActionRejectedError("reset restores the episode's initial state; to change task or seed, switch "
                                      "variation (POST /v1/variations/start, or reset with options {variation_id}; the "
                                      "catalogue: GET /v1/variations); nothing changed")
        with self.env.lock:
            self._state_cache = None
            return self._state(self.env.reset()), {"continuity": "reset"}

    # ------------------------------------------------------------------ variations
    def _variation_config(self, entry):
        """The world's Config of a catalogue entry for this session: the constructor's construction (the session's
        config overrides, then the entry's task and seed as sewing/variants.py resolves them, the robot, the device),
        so the switched world and its cached start state are those of a fresh session at that task and seed."""
        from sewing import variations as VR
        from sewing.variants import check_robot
        from sewing.world import Config
        values = dict(self._config_overrides)
        values.update(VR.config_values(entry))
        values["robot"] = self.config.robot
        if self._device is not None:
            values["device"] = self._device
        try:
            check_robot(self.config.robot, entry["task"])
        except ValueError as exc:
            raise ActionRejectedError(f"{exc}; nothing changed") from exc
        return Config(**{k: v for k, v in values.items() if k in Config.__dataclass_fields__})

    def _start_variation(self, variation_id, contract=None):
        """Replace the episode by a catalogue variation (same robot, cameras, renderer options, device, config overrides
        and session): the new environment is built -- its start state loaded from the cache, or prepared when not
        cached, as a fresh session's -- before the old one is released. A refused id changes nothing; a build that
        fails leaves the old episode in place."""
        from sewing import variations as VR
        from sewing.task_set_env import make_environment
        from sewing.task_set import instruction
        if self.trace is not None and self.trace.current:
            raise ActionRejectedError("Cannot switch the variation during an active action trace; nothing changed")
        try:
            entry = VR.specification(variation_id, contract)
        except ValueError as exc:
            raise ActionRejectedError(f"{exc}; nothing changed") from exc
        config = self._variation_config(entry)
        with self.env.lock:
            t0 = time.perf_counter()
            try:
                env = make_environment(config, contract=entry.get("contract"), output_dir=self.output)
            except Exception as exc:
                gc.collect()
                raise BackendProtocolError(f"variation {variation_id} could not be built ({type(exc).__name__}: "
                                           f"{exc}); the previous episode is unchanged") from exc
            build_s = time.perf_counter() - t0
            old = self.env
            env.generation = old.generation + 1
            # the render worker holds the old world's scene (the plies differ by task): release it; the first
            # observation of the new episode starts a worker on the new scene
            for r in (self._take_prestart()[0], self._take_retained()):
                if r is not None:
                    self._release(r)
            self._scene_path = None
            self._checkpoints = {}
            self.env, self.config, self.variation = env, config, entry
            self.task_contract = env.task_contract
            self._state_cache = None
            if self.trace is not None:
                self.trace.bind(env)
            self.capabilities.notes["task"] = self._task_note()
            self.capabilities.notes["instruction"] = instruction(config.task, config.robot, self.task_contract)
            # release the replaced world now: SewingEnv.close() frees nothing, and the env and its world (Newton's
            # model, the MuJoCo model and data, the cloth's Warp arrays) sit in reference cycles (env <-> evaluator,
            # world <-> kinematics) that only the cyclic collector frees, whenever a served session's automatic
            # collections reach them: collect once here (a test holds that the replaced env and world are gone after
            # the switch returns; with _import_engine, the first switch too)
            old.close()
            del old
            t1 = time.perf_counter()
            gc.collect()
            self.last_timing.update(switch_build_s=build_s, switch_gc_s=time.perf_counter() - t1)
            state = self._state()
        log = getattr(self.env, "start_log", None)
        return state, {"continuity": "new_variation", "variation": entry,
                       "start_state": dict(name=self.env.start_state,
                                           cached=None if log is None else bool(log.get("cached")),
                                           build_s=round(build_s, 3)),
                       "note": "checkpoints of the previous episode were dropped"}

    def variation_contract(self):
        from sewing import variations as VR
        current = dict(variation=self.variation, task=self.config.task, seed=self.config.seed,
                       robot=self.config.robot, config_overrides=sorted(self._config_overrides))
        return VR.contract(current, checkpoints=CHECKPOINT_SCOPE)

    def select_variation(self, request):
        from sewing import variations as VR
        from sewing.task_set import instruction
        entry = VR.select_request(dict(request))
        return dict(variation_id=entry["variation_id"], variation=entry, task=entry["task"],
                    description=instruction(entry["task"], self.config.robot, entry.get("contract")),
                    reset_request={"options": {"variation_id": entry["variation_id"],
                                               **({"contract": entry["contract"]} if "contract" in entry else {})}})

    # ------------------------------------------------------------------ per-action trace
    def begin_action_trace(self, context):
        if self._trace_bytes is None:
            return {"available": False, "reason": "trace disabled (trace=false)"}
        with self.env.lock:
            if self.trace is None:
                from sewing.trace import Trace
                self.trace = Trace(self.env, self.output/"physics_trace.sqlite3", max_bytes=self._trace_bytes)
            return self.trace.begin(context)

    def end_action_trace(self):
        if self.trace is None:
            return None
        with self.env.lock:
            return self.trace.finish()

    def _state_key(self):
        env = self.env
        return id(env), env.generation, env.steps, int(env.world.tick)

    def _state(self, built=None):
        """The environment's state at the current tick, built once per tick (welding's adapter does the same). GAR reads
        the state before and after every command for its action history, and the command's result and an observation use
        it too: three builds per command, each with the task's evaluation. Every command, reset, restore and variation
        switch drops the kept state before it changes the world, and the key holds the episode's tick and the world's
        own, so a state is never handed out for another tick. `built`: a state the environment has just returned for
        this tick, kept instead of building one. Callers own their copy."""
        env = self.env
        with env.lock:
            key = self._state_key()
            c = self._state_cache
            if built is not None:
                c = self._state_cache = dict(key=key, state=self._unjudged_fault(built))
            elif c is None or c["key"] != key:
                c = self._state_cache = dict(key=key, state=self._unjudged_fault(env.get_state()))
            return copy.deepcopy(c["state"])

    def _unjudged_fault(self, state):
        """A faulted world is not judged: SewingEnv.get_state gives its state {success: False, score: 0, checks: {}}, a
        zero no evaluator produced, and `evaluate` (gar verify) refuses it. Its evaluation is reported unavailable, with
        the fault and the last evaluation judged before it, marked interrupted. A state
        that is not faulted is kept as the last judged one."""
        if state.get("fault"):
            state["evaluation"] = dict(available=False, success=None, score=None, fault=state["fault"],
                                       task_contract=self.task_contract,
                                       interrupted=True, last_valid=self._last_valid_evaluation)
        elif state.get("evaluation") is not None:
            self._last_valid_evaluation = dict(tick=state.get("tick"), evaluation=state["evaluation"])
        return state

    def get_state(self):
        return self._state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self.env.evaluator.evaluate()

    # ------------------------------------------------------------------ render workers
    def _frame(self):
        from sewing.render import world_frame
        return world_frame(self.env.world, "UNASSISTED")

    def _scene(self, frame):
        from sewing.render import export_scene, scene_meta
        if self._scene_path is None or not Path(self._scene_path).is_file():
            self._scene_path = export_scene(self.output/"render", frame, scene_meta(self.env.world))
        return self._scene_path

    def _new_renderer(self, scene):
        from sewing.render import Renderer
        return Renderer(self.output/"render", scene, width=self.options.get("width", 1280),
                        height=self.options.get("height", 720), samples=self.options.get("samples", POLICY_SAMPLES))

    def _start_renderer_early(self):
        if not self.cameras or self._prestart is not None:
            return
        r = self._take_retained()
        if r is not None:
            self._prestart = {"renderer": r, "error": None, "warm_s": 0., "thread": None, "retained": True}
            return
        frame = self._frame()
        scene = self._scene(frame)
        holder = {"renderer": None, "error": None, "warm_s": None}

        def work():
            try:
                t0 = time.perf_counter()
                holder["renderer"] = self._new_renderer(scene)
                holder["renderer"].observe(frame, tuple(self.cameras))
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:
                holder["error"] = exc
        holder["thread"] = threading.Thread(target=work, name="sewing-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _take_prestart(self):
        holder, self._prestart = self._prestart, None
        if holder is None:
            return None, None
        if holder.get("thread") is not None:
            holder["thread"].join()
        self.last_timing.update(prestart_warm_s=holder["warm_s"], retained_worker=bool(holder.get("retained")))
        if holder["error"] is not None or holder["renderer"] is None:
            if holder["renderer"] is not None:
                self._release(holder["renderer"])
            return None, holder["error"]
        return holder["renderer"], None

    def _release(self, renderer, wait=False):
        self._reapers = [t for t in self._reapers if t.is_alive()]
        if wait:
            renderer.close()
            return
        t = threading.Thread(target=renderer.close, name="sewing-render-release", daemon=False)
        t.start()
        self._reapers.append(t)

    def _take_retained(self):
        r, self._retained = self._retained, None
        if self._retain_timer is not None:
            self._retain_timer.cancel()
            self._retain_timer = None
        return r

    def _end_operation(self, renderer):
        idle = float(self.options.get("idle_release_s", IDLE_RELEASE_S) or 0.)
        if idle <= 0.:
            self._release(renderer)
            return
        old = self._take_retained()
        if old is not None:
            self._release(old)
        self._retained = renderer
        timer = threading.Timer(idle, self._release_retained, args=(renderer,))
        timer.daemon = True
        timer.start()
        self._retain_timer = timer

    def _release_retained(self, renderer):
        with self.env.lock:
            if self._retained is not renderer:
                return
            self._retained = None
        renderer.close()

    def _operation_renderer(self, frame):
        renderer, _ = self._take_prestart()
        if renderer is not None:
            return renderer
        renderer = self._take_retained()
        if renderer is not None:
            self.last_timing["retained_worker"] = True
            return renderer
        scene = self._scene(frame)
        t0 = time.perf_counter()
        renderer = self._new_renderer(scene)
        self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return renderer

    def _render(self, names):
        with self.env.lock:
            frame = self._frame()
            r = self._operation_renderer(frame)
            try:
                t0 = time.perf_counter()
                result = r.observe(frame, tuple(names))
                self.last_timing["frame_s"] = time.perf_counter() - t0
            except BaseException:
                self._release(r)
                raise
            self._end_operation(r)
            frames = {}
            for name, blob in result.items():
                path = self.output/f"{name}_{self.env.generation}_{self.env.steps}_{uuid.uuid4().hex}.png"
                path.write_bytes(blob)
                self.images.append(path)
                frames[name] = path
            while len(self.images) > 64:
                old = self.images.popleft()
                if old.parent != self.output:
                    raise RuntimeError("Image retention escaped the session folder")
                old.unlink(missing_ok=True)
            return frames

    def _machine_proprioception(self, state):
        m = state.get("machine") or {}
        return {"shaft_angle_rad": float(m.get("shaft_angle_rad", 0.)), "speed_spm": float(m.get("speed_spm", 0.)),
                "needle_down": 1. if m.get("needle") == "down" else 0.,
                "presser_down": 1. if m.get("presser") == "down" else 0.,
                "reverse": 1. if m.get("reverse") else 0., "running": 1. if m.get("mode") not in (None, "idle") else 0.,
                "fault": 1. if m.get("fault") else 0., "stitch_counter": float(m.get("stitch_counter", 0))}

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.steps, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.config.robot,
                               {"joints": state["joints"], "machine": self._machine_proprioception(state)},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

    # ------------------------------------------------------------------ actions
    def _ticks(self, seconds):
        ticks = seconds/self.config.dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8, rel_tol=0) or round(ticks) < 1:
            raise ValueError(f"Duration must be a positive multiple of {self.config.dt} s")
        return int(round(ticks))

    def _arm(self, arm):
        if arm not in self.arms:
            raise UnsupportedActionError(f"Arm must be one of {self.arms}")
        return arm

    def _fingers(self, arm, width):
        w = self.env.world
        if self.g1:
            raise UnsupportedActionError("the G1's Dex3 hands take named joint targets, not gripper widths")
        if arm not in w.hand_arms:
            raise ValueError(f"Arm {arm!r} carries no gripper")
        joints = w.gripper_joints(arm)
        spec = w.joints[joints[0]]
        lo, hi = float(spec["lower"]), float(spec["upper"])
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) \
                or not lo <= width <= hi:
            raise ValueError(f"Franka Hand half gap must be in [{lo:g}, {hi:g}] m")
        return {j: float(width) for j in joints}

    def _validate(self, targets):
        joints = self.env.world.joints
        for name, value in targets.items():
            spec = joints.get(name)
            if spec is None:
                raise ValueError(f"Not a robot actuator: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Joint target {name}={value} outside [{spec['lower']:g}, {spec['upper']:g}]")

    def _machine_outputs(self, action):
        if action.arm not in self.arms:
            raise ValueError(f"tool_io: the machine's control box is wired to the cell controller; name one of the "
                             f"arms {self.arms} as the channel")
        out = {}
        for k, v in action.outputs.items():
            spec = TOOL_OUTPUTS.get(k)
            if spec is None:
                raise ValueError(f"tool_io: unknown output {k!r}; outputs: {', '.join(TOOL_OUTPUTS)}")
            if spec["kind"] == "bool":
                if not isinstance(v, bool):
                    raise ValueError(f"tool_io: {k} must be true or false")
            elif spec["kind"] == "number":
                lo, hi = spec["range"]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
                    raise ValueError(f"tool_io: {k} must be a number in [{lo:g}, {hi:g}] {spec['unit']}")
                v = float(v)
            elif v not in spec["choices"]:
                raise ValueError(f"tool_io: {k} must be one of {spec['choices']}")
            out[k] = v
        if "speed_spm" in out and self.config.stitch_length > 4e-3 and out["speed_spm"] > 4000.:
            raise ValueError("tool_io: above 4 mm stitch length the head is limited to 4000 sti/min")
        if out.get("trim") is False:
            out.pop("trim")
        return out

    def _impedance(self, request):
        """The joint_impedance raw operation, validated: (arm, stiffness or None, damping or None, ticks)."""
        allowed = {"op", "arm", "stiffness_nm_per_rad", "damping_nms_per_rad", "ticks"}
        extra = set(request) - allowed
        if extra:
            raise ValueError(f"joint_impedance: unknown fields {sorted(extra)}; fields: {sorted(allowed)}")
        if self.g1:
            raise UnsupportedActionError("joint_impedance is offered for the FR3 and Panda arms only")
        arm = self._arm(request.get("arm"))
        ticks = request.get("ticks", 1)
        if type(ticks) is not int or ticks < 1:
            raise ValueError("joint_impedance: ticks must be a positive integer")
        if request.get("stiffness_nm_per_rad") is None and request.get("damping_nms_per_rad") is not None:
            raise ValueError("joint_impedance: damping needs a stiffness; send neither to restore the position servo")
        return arm, request.get("stiffness_nm_per_rad"), request.get("damping_nms_per_rad"), ticks

    def _prepare(self, action, solve=True):
        """(joint targets, ticks, grip change or None, machine outputs or None); raises before any write or tick. A
        joint_impedance request is validated here and kept in self._pending_impedance for act to apply. solve=False
        leaves out the one check that depends on where the arms are, a Cartesian command's IK (validate_action)."""
        grip, tool = None, None
        self._pending_impedance = None
        if isinstance(action, RawAction):
            request = dict(action.request)
            if request.get("op") != "joint_impedance":
                raise UnsupportedActionError(f"raw operation {request.get('op')!r}: only joint_impedance is offered")
            arm, K, D, ticks = self._impedance(request)
            w = self.env.world
            # validate the values on the world's own checks without changing it: a dry run against the bounds
            names = [n for n, _ in w._arm_mj_dofs[arm]]
            for what, v, hi in (("stiffness_nm_per_rad", K, 14250.0), ("damping_nms_per_rad", D, None)):
                if v is None:
                    continue
                vals = [v]*len(names) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                if not isinstance(vals, (list, tuple)) or len(vals) != len(names) or any(
                        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                        or (hi is not None and x > hi) for x in vals):
                    raise ValueError(f"joint_impedance: {what} must be a number or {len(names)} numbers >= 0"
                                     + (f" and <= {hi:g}" if hi else ""))
            self._pending_impedance = (arm, K, D)
            if ticks*self.config.dt > MAX_COMMAND_S + 1e-9:
                raise ValueError(f"A command may last at most {MAX_COMMAND_S:g} s of simulated time")
            return {}, ticks, None, None
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
        elif isinstance(action, ToolIO):
            tool = self._machine_outputs(action)
            targets, ticks = {}, action.ticks
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                total = float(action.inward_effort_n)
                if self.finger_effort_limit is None or total/2. > self.finger_effort_limit:
                    raise ValueError("inward_effort_n exceeds the hand's finger limit (or no hand)")
                w = self.env.world
                width = 0. if total > 0 else (action.half_width_m if action.half_width_m is not None else
                                              float(w.joints[w.gripper_joints(arm)[0]]["upper"]))
                targets, grip = self._fingers(arm, width), (arm, total/2.)
            else:
                targets, grip = self._fingers(arm, action.half_width_m), (arm, 0.)
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            arm = self._arm(action.arm)
            s = action.safety
            if s.stop_on_grip_loss is not None or s.stop_on_contact or s.stop_on_robot_contact:
                raise UnsupportedActionError("Simulator motion guards are not implemented in sewing; no action executed")
            ticks = self._ticks(action.duration_s)
            if not solve:
                targets = {}
            elif isinstance(action, CartesianPose):
                o = action.orientation_wxyz
                targets = self.env.world.kin.solve(arm, action.position_m, None if o is None else (*o[1:], o[0]))
            else:
                kin = self.env.world.kin
                targets = kin.solve(arm, kin.forward(arm)[:3, 3] + np.asarray(action.delta_m))
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, "
                                         "tool_io, raw (op joint_impedance)")
        if ticks*self.config.dt > MAX_COMMAND_S + 1e-9:
            raise ValueError(f"A command may last at most {MAX_COMMAND_S:g} s of simulated time (it returns inside the "
                             f"{CLIENT_TIMEOUT_S} s client timeout at the slowest measured rate); send several")
        self._validate(targets)
        return targets, ticks, grip, tool

    def validate_action(self, action):
        """One command checked as act checks it, without a write or a tick, in everything that does not depend on
        where the arms are: the action kind, the arm, the guards, the duration and its cap, the machine's outputs, the
        gripper and joint targets, the joint impedance. ActionRejectedError (or UnsupportedActionError) on refusal.
        GAR calls it for every command of a batch before the first one runs (gym_anything_robotics/env.py), so
        {"actions": [valid, invalid]} executes nothing and the session stays ready.
        Not checked here: whether a Cartesian target is reachable. The IK starts from the arm's joints when the
        command's turn comes, which the earlier commands of the batch will have moved, so judging it now would refuse
        batches that are valid; an unreachable target later in a batch is still refused at its turn, after the
        commands before it have run."""
        with self.env.lock:
            try:
                self._prepare(action, solve=False)
            except (ValueError, TypeError) as exc:
                message = str(exc) if "no action executed" in str(exc) else f"{exc}; no action executed"
                raise ActionRejectedError(message) from exc
            finally:
                self._pending_impedance = None

    def act(self, action):
        with self.env.lock:
            w = self.env.world
            if w.fault:
                raise BackendProtocolError(f"Physics is faulted ({w.fault}); restore a checkpoint or reset")
            try:
                targets, ticks, grip, tool = self._prepare(action)
            except (ValueError, TypeError) as exc:
                message = str(exc) if "no action executed" in str(exc) else f"{exc}; no action executed"
                raise ActionRejectedError(message) from exc
            before = self.env.steps
            self.last_timing = {}
            self._state_cache = None
            self._start_renderer_early()
            try:
                if self._pending_impedance is not None:
                    arm, K, D = self._pending_impedance
                    w.set_joint_impedance(arm, K, D)
                if grip is not None:
                    w.set_grip_force(*grip)
                if tool:
                    self.env.machine_io(tool)
                t0 = time.perf_counter()
                self.env.act(targets, ticks=ticks, state=False)
                self.last_timing["physics_s"] = time.perf_counter() - t0
                state = self._state()
            except BaseException as exc:
                renderer, _ = self._take_prestart()
                if renderer is not None:
                    self._release(renderer)
                if isinstance(exc, (RuntimeError, FloatingPointError, OSError, ValueError)):
                    raise BackendProtocolError(f"Stopped at tick {self.env.steps}: {exc}") from exc
                raise
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.steps - before,
                                       "machine": state["machine"], "events": state["events"],
                                       "termination": "time_limit" if state["terminated"] else None,
                                       "physics_s": self.last_timing.get("physics_s")},
                               ev["score"], bool(state["terminated"]))

    def policy_access(self):
        from sewing.task_set import instruction
        prop = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        # the arms' estimated external joint torques (libfranka's tau_ext_hat; the G1 has none here)
        prop.update({("joints", n, "external_torque"): None for n in self.torque_joints})
        prop.update({("machine", k): None for k in TOOL_FEEDBACK})
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), prop,
                            self.kinds, lambda: instruction(self.config.task, self.config.robot, self.task_contract),
                            {"dt_s": self.config.dt, "arms": list(self.arms), "wall_time": WALL_TIME_NOTE,
                             "client_timeout_s": CLIENT_TIMEOUT_S, "max_command_s": MAX_COMMAND_S,
                             "machine_io": {k: dict(v) for k, v in TOOL_OUTPUTS.items()},
                             "machine_io_channel": "tool_io with any arm name of the deployment",
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()}},
                            extra_arm_names=self.extra_arm_names)

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "hero" if label in ("interactive", "http") else label
        if name not in self.policy_cameras + DIAGNOSTIC:
            raise ValueError(f"Render label must be one of {self.policy_cameras + DIAGNOSTIC}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry and checkpoints
    def get_geometry_index(self):
        from sewing import layout as LY
        w = self.env.world
        return {"items": {"table": dict(kind="box", x_m=list(LY.TABLE["x"]), y_m=list(LY.TABLE["y"]), top_z_m=0.0),
                          "needle": dict(kind="axis", position_m=[0., 0., 0.], note="needle axis meets the plate top"),
                          "presser_foot": dict(kind="foot", sole_bottom_z_m=float(w.machine.foot_z)),
                          "plies": dict(kind="cloth", count=len(w.ply_ids), width_m=w.config.ply_width,
                                        length_m=w.config.ply_length, seam_edge="u = width (right edge)")},
                "frame": "machine frame: origin where the needle axis meets the plate top, x right, y away from "
                         "the operator (feed), z up; metres", "source": "sewing/layout.py, live state"}

    def get_geometry(self, geometry_id):
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.steps, "generation": self.env.generation,
                    "variation_id": self.variation["variation_id"], "storage": "in_memory_session",
                    "scope": CHECKPOINT_SCOPE}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = self._checkpoints.get(label)
            if snap is None:
                raise KeyError(label)
            self._state_cache = None
            self.env.restore(snap)
            self.env.generation += 1
            return self._state(), {"checkpoint": label, "tick": self.env.steps,
                                          "generation": self.env.generation, "continuity": "restored_discontinuity",
                                          "storage": "in_memory_session", "scope": CHECKPOINT_SCOPE}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["steps"]} for k, v in self._checkpoints.items())

    def close(self):
        with self.env.lock:
            renderer, _ = self._take_prestart()
            if renderer is not None:
                self._release(renderer, wait=True)
            retained = self._take_retained()
            if retained is not None:
                self._release(retained, wait=True)
            for t in self._reapers:
                t.join()
            self._reapers = []
            if self.trace is not None:
                self.trace.close()
                self.trace = None
        self.env.close()

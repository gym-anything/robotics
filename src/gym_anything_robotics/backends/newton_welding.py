"""Welding bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 rigid cell + the weld process on the GPU).

The environment (welding/env.py WeldingEnv) owns the world, its clock, the start state, checkpoints and the evaluator;
this adapter selects the deployment, maps GAR robot actions to joint targets and the welding source's I/O (tool_io),
renders robot cameras through a retained Isaac render-only worker (released after idle_release_s, default 600 s)
and exposes the audited restricted interface. Render worker lifecycle as filleting's adapter (prestart during physics,
bounded idle retention).

Owner-side extras: a per-physics-tick trace of every action (welding/trace.py; GAR's /v1/history/{id}/trace) and the
frozen variation catalogue (welding/variations.py; /v1/variations, variation_id at launch or reset).

State and evaluation: GAR reads the state before and after every command (its action history) and for the command's
result and observation; the backend reads WeldingEnv's state once per tick and runs the task evaluator only on demand
(GET /v1/evaluation), unless the option state_evaluation=true puts it in every state and the owner's reward.
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
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, ToolIO
from ..errors import (ActionRejectedError, BackendProtocolError, UnsupportedActionError,
                      UnsupportedConfigurationError)
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

# HTTP client timeout the backend declares (the gar CLI uses it by default): the longest legal requests are a whole
# fillet pass as one command (~25 s simulated with the arc on: from ~1,250 s at the 50 wall s per simulated s of a
# lightly shared GPU to ~4,900 s at 195 under five other render workers, so send shorter commands) and the first start
# after a physics change, which builds the tack weld inside the request (157 s on a quiet GPU, up to 45 min on a
# saturated one)
CLIENT_TIMEOUT_S = 3600
# The render worker is kept warm this long after a request (renderer_options idle_release_s; 0 releases it after every
# request). A policy that thinks longer than this pays a fresh worker on its next request, about 12 s, so a 50 ms
# command after a 15 s pause takes 303-495 wall s per simulated s. A policy's think time (an LLM or VLA choosing its
# next action) is seconds to minutes; 10 minutes covers it, and a session left idle longer gives the worker's GPU memory
# back to the shared GPU (RENDER_WORKER_GPU).
IDLE_RELEASE_S = 600.0
# what one welding render worker holds on the GPU: 4,209-4,408 MiB (4.1-4.3 GiB) at the policy cameras' 1280x720 in
# every run (nvidia-smi per process, counting the worker's two identical rows once). Mostly fixed: 3,910 MiB at
# 640x360, 4,706 MiB at 1920x1080; 1 sample instead of 8 and three cameras instead of one change it by under 0.1 GiB
RENDER_WORKER_GPU = "about 4.2 GiB of GPU memory"
LIMIT_X = 200.
# load dependence measured independently of E16 (physics cb1ff92bd4ba68d2): quoted with the physics it ran on, beside
# the E16 figures that follow the latest run
AUDIT_LOAD_NOTE = ("a separate measurement (physics cb1ff92bd4ba68d2) found 50 ms commands at "
                   "152-175x under one other render worker and 203-233x under four (13.7-19.2 GB in use, 100 % busy), "
                   "while 100 ms commands ran at 102-173x under both")


def _rng(values, nd=0):
    lo, hi = min(values), max(values)
    a, b = f"{lo:,.{nd}f}", f"{hi:,.{nd}f}"
    return a if a == b else f"{a}-{b}"


def _load_text(e16):
    """What the shared GPU carried while E16 ran: from its sampled load or, for results without one, from the
    nvidia-smi lines read before and after each camera set."""
    sets = list(e16["sets"].values())
    loads = [s.get("load") for s in sets]
    if all(isinstance(x, dict) and x.get("used_mib") for x in loads):
        used = [x["used_mib"]["median"]/1024 for x in loads]
        util = [x["util_pct"]["median"] for x in loads]
        workers = [x["other_render_workers"]["median"] for x in loads]
        most = max(x["other_render_workers"]["max"] for x in loads)
        return (f"the GPU also carried {_rng(workers)} other Isaac render worker{'s' if max(workers) != 1 else ''} "
                f"(median of each camera set; at most {most}) and {_rng(used, 1)} GiB in use in all, {_rng(util)} % busy "
                f"(medians, nvidia-smi every 3 s)")
    busy = [float(s[k].split(",")[-1].strip().rstrip("%").strip()) for s in sets for k in ("gpu_before", "gpu_after")]
    used = [float(s[k].split(",")[0].split()[0])/1024 for s in sets for k in ("gpu_before", "gpu_after")]
    return (f"the GPU had {_rng(used, 1)} GiB in use and was {_rng(busy)} % busy (nvidia-smi before and after each "
            f"camera set, the measuring session included)")


def wall_time_figures(e16):
    """The numbers WALL_TIME_NOTE quotes, derived from E16's result (welding/data/runtime/e16_runtime_api.json).
    The note is built from that file when this module is imported, so a new E16 run updates what a policy reads. A
    result may have no paused-policy case and no sampled load; the note then says so."""
    one, three = e16["sets"]["one camera"], e16["sets"]["three cameras"]

    def walls(s, kind):
        return [r["wall_s"] for r in s["rows"] if r["kind"] == kind]

    def ratios(s, kind, step=""):
        return [r["wall_per_sim"] for r in s["rows"] if r["kind"] == kind and r["step"].startswith(step)]

    minimum = {s["minimum_command_s"] for s in (one, three)}
    if len(minimum) != 1:
        raise ValueError("E16's camera sets qualify different minimum commands")
    minimum_ms = 1e3*minimum.pop()
    out = dict(load=_load_text(e16), minimum_ms=f"{minimum_ms:g}",
               physics_hash=str(e16.get("physics_hash", "hash not recorded")),
               launch_s=_rng([one["launch_s"], three["launch_s"]]),
               physics_arc_on=_rng([s["fit_single_travel"]["slope_wall_per_sim"] for s in (one, three)]),
               physics_arc_off=_rng(ratios(one, "approach") + ratios(three, "approach")))
    for tag, s in (("one", one), ("three", three)):
        tw = s["travel_wall_per_sim"]
        gated = sorted((int(k.split()[0]) for k, v in tw.items() if v["gated"]), reverse=True)
        out[f"{tag}_observation_s"] = _rng(walls(s, "observation"), 2)
        out[f"{tag}_fixed_s"] = f"{s['fit_single_travel']['fixed_s']:.1f}"
        out[f"{tag}_longest"] = f"{tw[f'{gated[0]} ms']['median']:.0f}x at {gated[0]} ms"
        out[f"{tag}_minimum"] = f"{tw[f'{gated[-1]} ms']['median']:.0f}x at {gated[-1]} ms"
        out[f"{tag}_worst"] = f"worst {s['gated_worst']['wall_per_sim']:.0f}x"
        out[f"{tag}_within"] = "stayed within" if s.get("passed") is True else "did NOT all stay within"
        for ms in (50, 20, 10):
            out[f"{tag}_{ms}ms"] = _rng(ratios(s, "travel", f"travel {ms} ms"))
        out[f"{tag}_tool_io_alone"] = _rng(ratios(s, "tool_io_alone"))
        out[f"{tag}_batch"] = _rng(ratios(s, "batch"))
        paused = [r for r in s["rows"] if r["kind"] == "paused"]
        out[f"{tag}_paused"] = ("; ".join(
            f"{ms:g} ms {_rng([r['wall_per_sim'] for r in paused if abs(1e3*r['sim_s'] - ms) < 1e-6])}x"
            for ms in sorted({round(1e3*r["sim_s"], 6) for r in paused}, reverse=True)) if paused else None)
    pause_s = e16.get("pause_s")
    out["paused"] = (f"after a {pause_s:g} s pause before each command (a policy thinking): one camera "
                     f"{out['one_paused']}, three cameras {out['three_paused']}"
                     if out["one_paused"] and out["three_paused"] and pause_s else
                     "not measured by this E16 run (it sent its requests back to back)")
    worker = [s.get("render_worker_idle_release_s") for s in (one, three)]
    out["idle_release"] = (f"{_rng(worker)} s" if all(isinstance(w, (int, float)) for w in worker) else
                           "10 s (older E16 results do not record it)")
    del out["one_paused"], out["three_paused"]
    none = e16["sets"].get("no cameras")
    if none is None:
        out["no_cameras"] = "Without images (backend option cameras: []): not measured by this E16 run"
    else:
        tw = none["travel_wall_per_sim"]
        gated = sorted((int(k.split()[0]) for k, v in tw.items() if v["gated"]), reverse=True)
        paused = [r["wall_per_sim"] for r in none["rows"] if r["kind"] == "paused"]
        obs = walls(none, "observation")
        out["no_cameras"] = (
            f"Without images (backend option cameras: [], no render worker): an observation "
            f"{'under 0.01' if max(obs) < .01 else _rng(obs, 2)} s, a command's fixed cost "
            f"{none['fit_single_travel']['fixed_s']:.1f} "
            f"s; arc-on commands of {gated[-1]} ms or longer "
            f"{'stayed within' if none.get('passed') is True else 'did NOT all stay within'} the limit "
            f"({tw[f'{gated[0]} ms']['median']:.0f}x at {gated[0]} ms to {tw[f'{gated[-1]} ms']['median']:.0f}x at "
            f"{gated[-1]} ms, worst {none['gated_worst']['wall_per_sim']:.0f}x), 20 ms at "
            f"{_rng(ratios(none, 'travel', 'travel 20 ms'))}x, 10 ms at {_rng(ratios(none, 'travel', 'travel 10 ms'))}x"
            + (f", after a pause {_rng(paused)}x" if paused else ""))
    return out


def _wall_time_note(f):
    return (
        f"Measured through gar run + HTTP with the policy key on the shared RTX 5090 while {f['load']} "
        f"(physics {f['physics_hash']}: fillet-2f on ur10e-fr3, arc on along the seam). Every request pays a fixed "
        f"cost, mostly the observation's images (all requested cameras are rendered in one request): an observation alone "
        f"{f['one_observation_s']} s with the default 'bench' camera, {f['three_observation_s']} s with 'bench', 'side' "
        f"and 'weld_camera'; a command's fixed cost {f['one_fixed_s']} s and {f['three_fixed_s']} s. On top of it each "
        f"simulated second cost {f['physics_arc_on']} wall s with the arc burning, {f['physics_arc_off']} with the arc "
        f"off on the warm coupon (the fit's slope: the physics; other jobs on the GPU slow it as well as the renders). "
        f"Arc-on commands of {f['minimum_ms']} ms or longer against the 200x limit: with one camera they "
        f"{f['one_within']} it ({f['one_longest']} to {f['one_minimum']}, {f['one_worst']}), with three they "
        f"{f['three_within']} it ({f['three_longest']} to {f['three_minimum']}, {f['three_worst']}). Measured, not "
        f"qualified: 50 ms commands ran at {f['one_50ms']}x "
        f"with one camera and {f['three_50ms']}x with three, 20 ms at {f['one_20ms']}x and {f['three_20ms']}x, 10 ms "
        f"at {f['one_10ms']}x and {f['three_10ms']}x, a 1-tick tool_io sent alone at {f['one_tool_io_alone']}x and "
        f"{f['three_tool_io_alone']}x. Send tool_io changes in the same {{'actions': [...]}} request as the next motion "
        f"(one observation per request: with a {f['minimum_ms']} ms motion {f['one_batch']}x with one camera, "
        f"{f['three_batch']}x with three). {f['no_cameras']}. Pausing between requests: {f['paused']}; E16's "
        f"session kept its render "
        f"worker warm {f['idle_release']} after each request. The backend now keeps it {IDLE_RELEASE_S:g} s by default "
        f"(renderer_options idle_release_s; meanwhile it holds {RENDER_WORKER_GPU}); the first request after a longer "
        f"pause starts a new worker, 10-13 s more. These ratios depend on the GPU's load: {AUDIT_LOAD_NOTE}. Startup "
        f"{f['launch_s']} s with the tacked start state cached; building it (the first session after a physics "
        f"change) 157 s on a quiet GPU, up to 45 min on a saturated one. Client timeout: {CLIENT_TIMEOUT_S} s "
        f"(hardware.client_timeout_s).")


WALL_TIME_FALLBACK = (
    "The runtime figures are built from E16's result file (welding/data/runtime/e16_runtime_api.json), which this "
    "installation does not have or could not read: see welding/REPORT.pdf (Runtime). The physics runs at tens to hundreds "
    "of wall seconds per simulated second with the arc burning, depending on what else the GPU carries, and every "
    "request adds a fixed cost, mostly the observation's images, so short commands cost far more wall time per "
    "simulated second than long ones; send tool_io changes in the same {'actions': [...]} request as the next motion. "
    f"The render worker is kept warm {IDLE_RELEASE_S:g} s after each request (renderer_options idle_release_s). Client "
    f"timeout: {CLIENT_TIMEOUT_S} s (hardware.client_timeout_s).")


def e16_result_path():
    """welding/data/runtime/e16_runtime_api.json of the welding package this process imports (None if not found)."""
    import importlib.util
    spec = importlib.util.find_spec("welding")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])/"data"/"runtime"/"e16_runtime_api.json"


def load_wall_time_note(path=None):
    """The policy's runtime note from E16's result file; WALL_TIME_FALLBACK if the file is missing or unreadable."""
    import json
    try:
        path = Path(path) if path is not None else e16_result_path()
        return _wall_time_note(wall_time_figures(json.loads(path.read_text())))
    except (AttributeError, OSError, TypeError, KeyError, IndexError, ValueError):
        return WALL_TIME_FALLBACK


WALL_TIME_NOTE = load_wall_time_note()
POLICY_CAMERAS = ("bench", "side", "weld_camera")
DIAGNOSTIC = ("hero", "seam")
MOUNTS = {"bench": "external_boom_mount", "side": "external_stand_mount", "weld_camera": "torch_body_mount"}
RENDERER_OPTIONS = frozenset({"width", "height", "samples", "idle_release_s"})
POLICY_SAMPLES = 8
TOP_LEVEL = ("preset", "task", "seed", "robot")
SELECTED_CONFIG = ("coupon_offset", "draft_m_s")    # set by the seed or the catalogue variation, never by config
TRACE_BYTES = 512*1024*1024
SESSION_MARKER = ".gar-welding-session"
SESSION_RETENTION = 20
# the welding power source's robot interface: outputs and their ranges [read: Miller Auto-Continuum 350 spec sheet
# AU/10.0 (Mar. 2022) pp. 1, 8, 9: output 20-400 A, 10-44 V (rated 350 A at 31.5 V, 100 % duty; 400 A up to 50 % duty),
# wire drive 1.27-25.4 m/min; the wire feed set point here accepts 1.3-25 m/min, inside that range]; processes are the
# source's jobs
ARM_ARM_STOP_N = 1.        # the arm-arm guard's threshold: any loaded contact between the two arms [authored]
TOOL_OUTPUTS = {
    "arc_on": dict(kind="bool", note="weld start (the source energises, the wire runs in and strikes on contact)"),
    "gas_on": dict(kind="bool", note="shielding gas valve"),
    "touch_sense": dict(kind="bool", note="touch-sensing mode: with the arc off, the source reports wire-to-work contact"),
    "wire_feed_m_min": dict(kind="number", range=(1.3, 25.0), unit="m/min", note="wire feed speed set point"),
    "voltage_v": dict(kind="number", range=(10.0, 44.0), unit="V", note="constant-voltage set point"),
    "wire_inch": dict(kind="choice", choices=("forward", "backward", "stop"),
                      note="inch the wire at 2.5 m/min with the output off (it stops against metal); sets the stick-out"),
}
TOOL_FEEDBACK = ("arc_on", "current_a", "voltage_v", "wire_feed_m_min", "touch", "fault", "gas_on")


def _session_folder():
    import os
    import shutil
    import tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"welding-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def _check_robot(robot):
    from robots.catalog import ENVIRONMENTS
    catalog = ENVIRONMENTS["welding"]
    qualified = tuple(catalog["aliases"])
    if isinstance(robot, dict):
        raise UnsupportedConfigurationError(f"welding supports catalog aliases {', '.join(qualified)}")
    if robot in catalog.get("unsupported", {}):
        raise UnsupportedConfigurationError(f"welding does not support {robot!r}: {catalog['unsupported'][robot]}")
    if robot not in qualified:
        raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
    return robot


BACKEND_OPTIONS = ("preset", "task", "seed", "device", "config", "cameras", "output_dir", "renderer_options",
                   "variation_id", "trace", "trace_bytes", "state_evaluation")


def _check_renderer_options(options):
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
    return dict(options)


def check_options(robot="ur10e-fr3", **options):
    """Every check the backend makes before it builds anything, from the launch options alone (no world, no GPU, no
    session folder). The constructor runs it first, and GAR's launch check (welding/variants.py check_launch) runs it
    before a session exists. Returns the robot alias, the resolved
    preset/task/seed, the catalogue entry (or None), the world's Config values, the cameras and the renderer options."""
    unknown = sorted(set(options) - set(BACKEND_OPTIONS))
    if unknown:
        hint = ("; phantom_torch is an owner-side ASSISTED fixture, not a GAR option" if "phantom_torch" in unknown
                else "")
        raise UnsupportedConfigurationError(f"Unknown welding backend option(s) {unknown}; options: "
                                            f"{', '.join(BACKEND_OPTIONS)}{hint}")
    o = dict(preset=None, task=None, seed=None, device=None, config=None, cameras=("bench",), renderer_options=None,
             variation_id=None, trace=True, trace_bytes=None, state_evaluation=False)
    o.update(options)
    robot = _check_robot(robot)
    if o["config"] is not None and not isinstance(o["config"], dict):
        raise ValueError("config must be an object of welding.world.Config fields")
    values = dict(o["config"] or {})
    clash = sorted(set(TOP_LEVEL) & set(values))
    if clash:
        raise UnsupportedConfigurationError(f"{clash} are top-level options, not config fields")
    clash = sorted(set(SELECTED_CONFIG) & set(values))
    if clash:
        raise UnsupportedConfigurationError(f"{clash} are selected by seed or variation_id, not config fields")
    if o["variation_id"] is not None and o["seed"] is not None:
        raise UnsupportedConfigurationError("seed (legacy layouts 0-9) and variation_id are exclusive; variation 0 "
                                            "is seed 0's authored layout")
    if not isinstance(o["trace"], bool):
        raise ValueError("trace must be true or false")
    if not isinstance(o["state_evaluation"], bool):
        raise ValueError("state_evaluation must be true or false")
    tb = o["trace_bytes"]
    if tb is not None and (isinstance(tb, bool) or not isinstance(tb, int) or tb < 64*1024*1024):
        raise ValueError("trace_bytes must be an integer >= 64 MiB")
    if "phantom_torch" in values:
        raise UnsupportedConfigurationError("phantom_torch is an owner-side ASSISTED fixture, not a GAR option")
    if o["device"] is not None and "device" in values:
        raise UnsupportedConfigurationError("Device supplied at two levels; choose exactly one")
    seed = o["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer")
    cams = o["cameras"]
    if isinstance(cams, str) or not isinstance(cams, (list, tuple)) or not all(isinstance(c, str) for c in cams):
        raise ValueError(f"cameras must be a list of policy camera names {POLICY_CAMERAS}")
    cameras = tuple(cams)
    if len(set(cameras)) != len(cameras) or set(cameras) - set(POLICY_CAMERAS):
        raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; {DIAGNOSTIC} are privileged diagnostics")
    renderer = _check_renderer_options(o["renderer_options"])
    from welding.variants import resolve, check_robot
    from welding.world import Config
    from welding import variations as VR
    chosen = resolve(o["preset"], o["task"], 0 if seed is None else seed)
    try:
        check_robot(robot, chosen["task"])
    except ValueError as exc:
        raise UnsupportedConfigurationError(str(exc)) from exc
    variation = VR.specification(o["variation_id"]) if o["variation_id"] is not None else None
    values.update(preset=chosen["preset"], task=chosen["task"], seed=chosen["seed"], robot=robot,
                  coupon_offset=chosen["coupon_offset"])
    if variation is not None:
        values.update(VR.config_values(variation))
    if o["device"] is not None:
        values["device"] = o["device"]
    try:
        Config(**values)
    except TypeError as exc:
        raise UnsupportedConfigurationError(f"config: {exc}") from exc
    return robot, chosen, variation, values, cameras, renderer


class _WeldingPolicyAccess(PolicyAccess):
    """The restricted interface. A gripper command on a deployment without a Franka Hand is refused as an unsupported
    command ("Robot command is unsupported; no action executed. Consult the robot capabilities.", HTTP 422) instead of
    GAR's generic "Not permitted for this credential." (403), which reads like a key problem; the capabilities still do
    not list 'gripper' there."""

    def decode_actions(self, payload):
        items = payload.get("actions", [payload]) if isinstance(payload, dict) else []
        if isinstance(items, list) and "gripper" not in self.action_types and any(
                isinstance(i, dict) and i.get("type") == "gripper" for i in items):
            raise UnsupportedActionError("This deployment has no gripper (no Franka Hand); no action executed")
        return super().decode_actions(payload)


class NewtonWeldingBackend(RoboticsBackend):
    def __init__(self, *, robot="ur10e-fr3", preset=None, task=None, seed=None, device=None, config=None,
                 cameras=("bench",), output_dir=None, renderer_options=None, variation_id=None, trace=True,
                 trace_bytes=None, state_evaluation=False):
        robot, chosen, self.variation, values, self.cameras, self.options = check_options(
            robot, preset=preset, task=task, seed=seed, device=device, config=config, cameras=cameras,
            renderer_options=renderer_options, variation_id=variation_id, trace=trace, trace_bytes=trace_bytes,
            state_evaluation=state_evaluation)
        from welding.world import Config
        self.config = Config(**values)
        self.legacy_seed = None if self.variation is not None else chosen["seed"]
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("welding session (auto-created; pruned by retention)\n")
        self.state_evaluation = state_evaluation
        self._state_cache = None
        self.env = self._make_env(self.config, self.variation)
        self.trace = None
        if trace:
            from welding.trace import Trace
            self.trace = Trace(self.env, self.output/"physics_trace.sqlite3", max_bytes=int(trace_bytes or TRACE_BYTES))
        w = self.env.world
        # the G1 is one rig instance with two arms: Cartesian commands name the arm (right = torch, left = the Dex3
        # handling hand; '+waist' adds the three waist joints to the IK), as welding/humanoid.py G1Kinematics
        self.g1 = bool(getattr(w, "g1", False))
        self.arms = tuple(w.kin.arms) if self.g1 else tuple(w.rig.instances)
        self.torch_arm = w.torch_arm
        self.images = deque()
        self.last_timing = {}
        self._prestart = None
        self._retained = None
        self._retain_timer = None
        self._reapers = []
        self._checkpoints = {}
        self._scene_path = None
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        from welding import layout as LY
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            fov = float(LY.WELD_CAMERA["hfov"] if name == "weld_camera" else LY.CAMERAS[name]["hfov"])
            focal = .018/math.tan(math.radians(fov/2))
            self.camera_specs[name] = CameraSpec(name, MOUNTS[name], width, height, 1280, 720,
                                                 (.036, .036*height/width), (focal, focal))
        # the gripper command drives a Franka Hand; a deployment without one does not offer it
        self.gripper_arms = tuple(a for a in w.hand_arms
                                  if str(w.scene.arm_hw.get(a, {}).get("hand", "")).split("@")[0] == "franka-hand")
        self.kinds = frozenset({"named_joint_target", "cartesian_pose", "cartesian_delta", "tool_io",
                                *(["gripper"] if self.gripper_arms else [])})
        hand = self.gripper_arms[0] if self.gripper_arms else None
        self.finger_effort_limit = float(w.joints[w.gripper_joints(hand)[0]]["effort_limit"]) if hand else None
        c = self.config
        from welding.task import INSTRUCTIONS
        self.capabilities = Capabilities(
            "welding", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[n] for n in self.cameras), notes={
                "task": f"{c.preset}/{c.task} {self._layout_name()}; start state: {self.env.start_state}",
                "instruction": INSTRUCTIONS[c.task],
                "clock": f"One tick = {c.rigid_dt*1e3:g} ms rigid MuJoCo step; the weld process (thermal solver on the "
                         "GPU, arc, pool) advances the same interval in explicit substeps",
                "control": "Physical position servos with a rate-limited motion generator (joint speed and "
                           "acceleration limits); IK computes targets only (no collision-free planning); an "
                           "unreachable IK target or out-of-limit joint target is rejected before any tick. Protective "
                           "stops: the UR10e stops above its 150 N force limit (default safety configuration), the FR3 "
                           "and the Panda on libfranka's default 10/20 N m collision thresholds; all three on a joint "
                           "over its speed limit. The G1's low-level joint interface has no collision or speed reflex "
                           "(its torques saturate at the URDF limits; a gravity feed-forward is added on its waist and "
                           "arms, as Unitree's manipulation stack sends)",
                "guards": "Cartesian commands honour safety.stop_on_arm_arm_contact (default true): with two arms the "
                          "command ends at the first tick the arms touch (> 1 N) and both arms hold there; the step's "
                          "info reports guard_stop (two-arm rigs; the G1's arms are one robot and have no such "
                          "guard). stop_on_contact, stop_on_robot_contact and stop_on_grip_loss are refused before any "
                          "tick",
                "torch_tcp": "Cartesian commands to the torch arm ('right') place the torch work point -- on the wire "
                             "axis, 19 mm beyond the contact tip -- with its z axis along the wire toward the work",
                "tool_io": {"arm": self.torch_arm, "outputs": {k: dict(v) for k, v in TOOL_OUTPUTS.items()},
                            "feedback": list(TOOL_FEEDBACK),
                            "note": "the welding power source's robot interface; the arc exists only where the wire "
                                    "reaches metal that is grounded through the work clamp"},
                "arms": ",".join(self.arms),
                "gripper": (f"Franka Hand on '{hand}': half_width_m position mode or inward_effort_n force mode (<= "
                            f"{2*self.finger_effort_limit:g} N total)") if hand else
                           ("Dex3-1 hands: 7 joints per hand commanded by name (g1/<side>_hand_<joint>_joint); no "
                            "gripper action" if self.g1 else "no hand in this deployment"),
                "physics": "Rigid robots, clamps and coupon in MuJoCo; transient heat conduction with melting in the "
                           "coupon; a constant-voltage arc that self-regulates with stand-off; filler deposition and "
                           "a surface-tension pool surface; fusion, porosity from lost shielding, the crater; a tack/"
                           "weld bond whose capacity comes from the fused throat",
                "evaluator": ("Physical-state evaluator (welding/task.py): sections measured from the cells; GET "
                              "/v1/evaluation runs it on demand" + ("; every state carries it (state_evaluation=true)"
                                                                    if state_evaluation else
                                                                    "; states and command results do not (backend "
                                                                    "option state_evaluation=true adds it)")),
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; the weld camera uses a "
                           "process-camera exposure (welding filter) and sees the arc and pool; no depth or noise",
                "render_worker": (f"The images come from an Isaac render worker started with the session's first "
                                  f"observation (10-13 s) and kept warm {self._idle_release_s():g} s after each "
                                  f"request (renderer_options idle_release_s, default {IDLE_RELEASE_S:g}; 0 releases it "
                                  f"after every request). While warm it holds {RENDER_WORKER_GPU} on the shared GPU; "
                                  f"a request after a longer pause, and the first observation after a catalogue "
                                  f"variation switch, start a new one" if self.cameras else
                                  "no cameras: no render worker"),
                "wall_time": WALL_TIME_NOTE,
                "qualification": "Assisted fixtures are not policy actions",
                "trace": ("Owner: every action records each 1 ms physics tick (joints, torch work point, the welding "
                          "source, contact forces, gun contacts, bond, pool and energy ledger) and its events "
                          "(welding/trace.py); GET /v1/history/{id}/trace" if trace else "disabled (trace=false)"),
                "variations": "Owner: frozen catalogue welding-steel-cell-v1 (10,000 cell states with train/validation/"
                              "test splits); GET /v1/variations, POST /v1/variations/start"})

    # ------------------------------------------------------------------ lifecycle
    def _make_env(self, config, variation):
        """The environment for a configuration, with the task's start preset (welding/presets.py) and then a catalogue
        variation's initial conditions applied before tick 0."""
        from welding.env import WeldingEnv
        from welding import variations as VR
        from welding.presets import apply_start_preset
        env = WeldingEnv(config, output_dir=self.output)
        apply_start_preset(env)
        if variation is not None:
            VR.apply_initial(env, variation)
        return env

    def _layout_name(self):
        return (f"variation {self.variation['variation_id']} ({self.variation['split']})" if self.variation is not None
                else f"seed {self.legacy_seed}")

    def reset(self, *, seed=None, options=None):
        if self.trace is not None and self.trace.current:
            raise RuntimeError("Cannot reset during an active action trace")
        if options:
            if set(options) != {"variation_id"} or seed is not None:
                raise ValueError("Reset options: variation_id only (the frozen catalogue); do not combine a seed")
            return self._start_variation(options["variation_id"])
        if seed is not None and seed != self.legacy_seed:
            raise ValueError("Select preset/task/seed or variation_id when creating the session (or start a catalogue "
                             "variation); reset restores that initial state")
        with self.env.lock:
            self._state_cache = None
            state = self.env.reset()                      # WeldingEnv.reset reads the state once (evaluated)
            self._state_cache = dict(key=self._state_key(), state=state)
            return self._state(), {"continuity": "reset"}

    def _start_variation(self, variation_id):
        """Replace the episode by a catalogue variation (same task, robot and session): the new world is built and its
        start state made before the old one is released; a rejected id changes nothing."""
        from dataclasses import replace
        from welding import variations as VR
        entry = VR.specification(variation_id)
        config = replace(self.config, **VR.config_values(entry))
        with self.env.lock:
            env = self._make_env(config, entry)
            old = self.env
            env.generation = old.generation + 1
            for r in (self._take_prestart()[0], self._take_retained()):
                if r is not None:
                    self._release(r)
            self._scene_path = None
            self._checkpoints = {}
            self.env, self.config, self.variation, self.legacy_seed = env, config, entry, None
            self._state_cache = None
            if self.trace is not None:
                self.trace.bind(env)
            self.capabilities.notes["task"] = (f"{config.preset}/{config.task} {self._layout_name()}; start state: "
                                               f"{env.start_state}")
            # release the replaced world now (kept, it costs +256 MB of GPU and +395 MB of host memory per switch).
            # close() drops its Warp arrays at once; the rest (Newton's model, the MuJoCo
            # model and data, the solver) sits in reference cycles inside Newton that only the cyclic collector frees,
            # and a served session's automatic collections almost never reached them: collect here, once per switch
            old.close()
            del old
            t0 = time.perf_counter()
            gc.collect()
            self.last_timing["switch_gc_s"] = time.perf_counter() - t0
            state = self._state()
        return state, {"continuity": "new_variation", "variation": entry,
                       "note": "checkpoints of the previous episode were dropped"}

    def variation_contract(self):
        from welding import variations as VR
        current = dict(variation=self.variation, legacy_seed=self.legacy_seed, task=self.config.task,
                       robot=self.config.robot)
        return VR.contract(current)

    def select_variation(self, request):
        from welding import variations as VR
        from welding.task import INSTRUCTIONS
        entry = VR.select_request(dict(request))
        return dict(variation_id=entry["variation_id"], variation=entry, task=self.config.task,
                    description=INSTRUCTIONS[self.config.task],
                    reset_request={"options": {"variation_id": entry["variation_id"]}})

    def begin_action_trace(self, context):
        if self.trace is None:
            return {"available": False, "reason": "trace disabled (trace=false)"}
        with self.env.lock:
            return self.trace.begin(context)

    def end_action_trace(self):
        if self.trace is None:
            return None
        with self.env.lock:
            return self.trace.finish()

    def _state(self, evaluate=None):
        """The environment's state at the current tick, computed once per tick: GAR reads it before and after every
        command (the action history), the command's own result and the observation all use the same tick. The task
        evaluator runs only when asked (evaluate=True, GET /v1/evaluation, or the option state_evaluation): run on every
        state it would run three times per command, 0.10-0.13 s each on fillet-2f. Every command, reset, restore and
        variation drops the cached state before it changes the world."""
        env = self.env
        with env.lock:
            evaluate = self.state_evaluation if evaluate is None else evaluate
            key = self._state_key()
            c = self._state_cache
            if c is None or c["key"] != key:
                state = env.get_state(evaluate=False)
                if env.world.fault is None:
                    state["evaluation"] = None
                c = self._state_cache = dict(key=key, state=state)
            if evaluate and c["state"]["evaluation"] is None:
                c["state"]["evaluation"] = env.evaluator.evaluate()
            state = copy.deepcopy(c["state"])           # callers own their copy (observations keep the joints dict)
            if state["evaluation"] is None:
                state["evaluation_note"] = ("not evaluated with every state (backend option state_evaluation=false); "
                                            "GET /v1/evaluation evaluates this tick")
            return state

    def _state_key(self):
        env = self.env
        return id(env), env.generation, env.steps, int(env.world.tick)

    def get_state(self):
        return self._state()

    def evaluate(self):
        """The on-demand evaluation (GET /v1/evaluation, the benchmark's final one) of this tick. A weld whose pool is
        still freezing with the arc off is judged as it freezes hands-off (welding/task.py, freeze_hands_off: a copy of
        the weld process; the world is not stepped), and that evaluation is this tick's from then on."""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            ev = self._state(evaluate=True)["evaluation"]
            if (ev.get("judged") or {}).get("freeze_pending"):
                ev = self.env.evaluator.evaluate(settle=True)
                self._state_cache["state"]["evaluation"] = copy.deepcopy(ev)
            return ev

    # ------------------------------------------------------------------ render workers
    def _frame(self):
        from welding.render import world_frame
        return world_frame(self.env.world, "UNASSISTED")

    def _scene(self, frame):
        from welding.render import export_scene, scene_meta
        if self._scene_path is None or not Path(self._scene_path).is_file():
            self._scene_path = export_scene(self.output/"render", frame, scene_meta(self.env.world))
        return self._scene_path

    def _new_renderer(self, scene):
        from welding.render import Renderer
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
        holder["thread"] = threading.Thread(target=work, name="welding-render-prestart", daemon=True)
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
        t = threading.Thread(target=renderer.close, name="welding-render-release", daemon=False)
        t.start()
        self._reapers.append(t)

    def _take_retained(self):
        r, self._retained = self._retained, None
        if self._retain_timer is not None:
            self._retain_timer.cancel()
            self._retain_timer = None
        return r

    def _idle_release_s(self):
        """Seconds the render worker stays warm after a request (renderer_options idle_release_s, default
        IDLE_RELEASE_S; 0: released after every request)."""
        v = self.options.get("idle_release_s")
        return float(IDLE_RELEASE_S if v is None else v)

    def _end_operation(self, renderer):
        idle = self._idle_release_s()
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

    def _tool_proprioception(self, state):
        t = state.get("tool") or {}
        return {"arc_on": 1. if t.get("arc_on") else 0., "current_a": float(t.get("current_a", 0.)),
                "voltage_v": float(t.get("voltage_v", 0.)), "wire_feed_m_min": float(t.get("wire_feed_m_min", 0.)),
                "touch": 1. if t.get("touch") else 0., "fault": 1. if t.get("fault") else 0.,
                "gas_on": 1. if t.get("gas_on") else 0.}

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.steps, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.config.robot,
                               {"joints": state["joints"], "tool": self._tool_proprioception(state)},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

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

    def _fingers(self, arm, width):
        w = self.env.world
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

    def _tool_outputs(self, action):
        if action.arm != self.torch_arm:
            raise ValueError(f"tool_io: the welding source is on arm {self.torch_arm!r}")
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
        return out

    def _prepare(self, action):
        """(joint targets, ticks, grip change or None, tool outputs or None); raises before any write or tick."""
        grip, tool = None, None
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
        elif isinstance(action, ToolIO):
            tool = self._tool_outputs(action)
            targets, ticks = {}, action.ticks
        elif isinstance(action, Gripper):
            if "gripper" not in self.kinds:
                raise UnsupportedActionError("This deployment has no gripper (no Franka Hand); no action executed")
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
                raise UnsupportedActionError("Simulator motion guards are not implemented in welding; no action executed")
            ticks = self._ticks(action.duration_s)
            kin = self.env.world.kin
            if isinstance(action, CartesianPose):
                o = action.orientation_wxyz
                targets = kin.solve(arm, action.position_m, None if o is None else (*o[1:], o[0]))
            else:
                targets = kin.solve(arm, kin.forward(arm)[:3, 3] + np.asarray(action.delta_m))
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, "
                                         "tool_io")
        self._validate(targets)
        return targets, ticks, grip, tool

    def act(self, action, *, evaluate=None):
        """One command. The reward is the evaluator's score when the state is evaluated (option state_evaluation),
        else None (GET /v1/evaluation evaluates on demand)."""
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
            self._guard_stop = None
            self._state_cache = None
            guard = len(self.arms) > 1 and not getattr(self, "g1", False) and \
                bool(getattr(getattr(action, "safety", None), "stop_on_arm_arm_contact", False))
            self._start_renderer_early()
            try:
                if grip is not None:
                    w.set_grip_force(*grip)
                if tool:
                    self.env.tool_io(tool)
                t0 = time.perf_counter()
                if guard:
                    self._act_guarded(targets, ticks)
                else:
                    self.env.act(targets, ticks=ticks, state=False)
                self.last_timing["physics_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                state = self._state(evaluate=evaluate)
                self.last_timing["state_s"] = time.perf_counter() - t0
            except BaseException as exc:
                renderer, _ = self._take_prestart()
                if renderer is not None:
                    self._release(renderer)
                if isinstance(exc, (RuntimeError, FloatingPointError, OSError, ValueError)):
                    raise BackendProtocolError(f"Stopped at tick {self.env.steps}: {exc}") from exc
                raise
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.steps - before,
                                       "guard_stop": self._guard_stop,
                                       "tool": state["tool"], "events": state["events"],
                                       "termination": "time_limit" if state["terminated"] else None,
                                       "physics_s": self.last_timing.get("physics_s"),
                                       "state_s": self.last_timing.get("state_s")},
                               None if ev is None else ev["score"], bool(state["terminated"]))

    def _arm_arm_force(self):
        """Summed normal force (N) of the MuJoCo contacts between the two arms' bodies (tools included)."""
        w = self.env.world
        mj, m, d = w.solver._mujoco, w.solver.mj_model, w.solver.mj_data
        lab = w.model.body_label
        f6 = np.zeros(6)
        total = 0.
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(w._g2b[con.geom[0]]), int(w._g2b[con.geom[1]])
            if b1 < 0 or b2 < 0:
                continue
            a1, a2 = lab[b1].split("/")[0], lab[b2].split("/")[0]
            if a1 != a2 and a1 in self.arms and a2 in self.arms:
                mj.mj_contactForce(m, d, i, f6)
                total += abs(float(f6[0]))
        return total

    def _act_guarded(self, targets, ticks):
        """stop_on_arm_arm_contact (MotionSafety, default true): the command runs tick by tick and ends at the first tick
        on which the two arms (their tools included) touch with more than ARM_ARM_STOP_N; both arms then hold where they
        are, and the step's info reports it (guard_stop)."""
        env, w = self.env, self.env.world
        for k in range(ticks):
            env.act(targets if k == 0 else {}, ticks=1, state=False)
            f = self._arm_arm_force()
            if f > ARM_ARM_STOP_N:
                q = w.state.joint_q.numpy()
                for arm in self.arms:
                    w._stop_arm(arm, q)
                self._guard_stop = dict(guard="stop_on_arm_arm_contact", tick=env.steps, force_n=round(f, 2),
                                        executed_ticks=k + 1, requested_ticks=ticks)
                break

    def policy_access(self):
        from welding.task import INSTRUCTIONS
        prop = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        prop.update({("tool", k): None for k in TOOL_FEEDBACK})
        return _WeldingPolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), prop,
                                    self.kinds, lambda: INSTRUCTIONS[self.config.task],
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms), "torch_arm": self.torch_arm,
                             "wall_time": WALL_TIME_NOTE, "client_timeout_s": CLIENT_TIMEOUT_S,
                             "render_worker_idle_release_s": self._idle_release_s() if self.cameras else None,
                             "torch_tcp": self.capabilities.notes["torch_tcp"],
                             "tool_io": {k: dict(v) for k, v in TOOL_OUTPUTS.items()},
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()}})

    def act_policy(self, action):
        result = self.act(action, evaluate=False)          # the policy's result carries no reward
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "hero" if label in ("interactive", "http") else label
        if name not in POLICY_CAMERAS + DIAGNOSTIC:
            raise ValueError(f"Render label must be one of {POLICY_CAMERAS + DIAGNOSTIC}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry and checkpoints
    def get_geometry_index(self):
        from welding import layout as LY
        w = self.env.world
        o, R = w.weld_frame()
        c = w.coupon
        bodies = {}
        for name, b in w.scene.bodies.items():
            p, Rb = w.body_pose(b)
            from scipy.spatial.transform import Rotation
            bodies[name] = dict(position_m=p.tolist(), quat_xyzw=Rotation.from_matrix(Rb).as_quat().tolist())
        return {"items": {"table": dict(kind="box", size_m=list(LY.TABLE["size"]) + [LY.TABLE["top_thickness"]],
                                        top_z_m=LY.TABLE["top_z"]),
                          "coupon": dict(kind=c.kind, base_m=list(c.base), web_m=list(c.web) if c.web else None,
                                         weld_length_m=c.length)},
                "weld_frame": dict(origin_m=o.tolist(), axes_columns=R.tolist(),
                                   note="x along the joint, y out of the bracket's weld face, z up; root line = x axis"),
                "bodies": bodies, "frame": "world, metres, z up", "source": "welding/layout.py, live body poses"}

    def get_geometry(self, geometry_id):
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.steps, "generation": self.env.generation}

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
                                   "storage": "in_memory_session"}

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

"""Syringe handling bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo rigid world with the
equipment's liquid network on the CPU, Isaac RTX rendering with the liquid drawn from the state). The structure is
syringe_handling's backend (which is wine_pouring's), taken over with its render-worker lifecycle, its command floor
and its refusals; what this environment changes is named where it differs: no thermometer, a liquid network instead of
a continuum (its meshes cost milliseconds, so the cost constants are lower), another geometry index.

Render workers follow barcode_scanning's lifecycle: a worker is started while the first command's physics runs and is
kept between operations; after renderer_options idle_release_s seconds (default IDLE_RELEASE_S) without an operation
it is released, and the next command starts a new one beside its physics. The idle clock does not run while a command's
physics runs (otherwise a command whose physics took longer than the idle time would lose its worker and its
observation would pay a cold start, 7-9 s and a slow first frame). close() releases the worker at once, and a
worker watches its owner process.

A worker that has died is replaced. Before a kept worker is relied on its process is looked at (_alive): one that has
exited is closed and dropped, it does not count as held (so the floor in force covers the start of its replacement, as
after an idle release), and the next command starts a new worker beside its physics. A worker that fails a frame (its
process ended under the request, its socket is broken, it answers with an error or not at all) is ended and dropped at
once and the frame is asked once more of a new worker; if that fails as well the session is left without a worker, the
error is raised, and the next image request starts one. Nothing waits for the idle release, and the owner's /v1/recover
and a reset go the same way, because they ask for an observation like everything else. What the worker has been through
is in the owner's state, field render_worker (worker_status: held, alive, pid, starts, lost, frame_retries, the last
fault).

The launch option renderer_options is an object with these keys and no others (check_renderer_options; an unknown
key, a value of the wrong type or out of range is refused before anything is built, with the reason):
  width            integer 160..1920, default 1280    pixels of every policy frame (the cameras' own format is 1280x720)
  height           integer 90..1080, default 720
  samples          integer 1..256, default 16         path-traced samples per pixel of a policy frame; the same number
                                                      as config.policy_samples (given both, they must agree)
  idle_release_s   number 0..86400 or null,           how long the render worker is kept after the session's last
                   default 900                        operation. 0: not kept, every observation starts a worker; null:
                                                      kept until the session is stopped
The session's first floor grows with cameras x pixels x samples (min_command_s); a combination whose first floor would
be above FLOOR_MAX_S (2 s) is refused, because the floor could then not cover its own observation.

The command floor. Every policy command, with or without cameras, lasts at least min_command_s of simulated time, so
that its wall time stays under BUDGET (200) times its simulated time: T >= (F + O + W) / (DESIGN_RATIO - r), a
command at the floor being planned to cost DESIGN_RATIO (120) times real time, with 40 percent headroom, with
  F  the fixed cost of a command whatever its length: the two state records the gateway takes round it, the history
     and trace writes, planning, the reply. The session measures it on every command (_note_cost: the wall time from
     the request's decoding to the end of its observation, less the physics and the frames) and UNSEEN_S is added for
     what the backend cannot time (the request before it is decoded, the reply's encoding, the image downloads);
  O  the observation: the liquid's meshes and the frames (0 for a session without cameras);
  W  a render worker's start, counted only while the session holds no worker (after the idle release): the command
     that follows starts one, so the floor in force is higher until that command has run;
  r  the physics' wall seconds per simulated second.
The session's first floor comes from the constants below, or, with cameras, from what the scene's first observation
took (observation_from_state); after that it follows the dearest of the last three
commands. (Without a floor, a session without cameras ran its 1 ms commands at 176 to 2,314 times real time, and the
command after a worker release at 269.)

What the 200x limit rests on, and what the policy is told. The floor bounds what a short command adds to its physics
(F, O, W). It does nothing about r, and r is what moves with the machine's load: alone on the card the largest task
ran at 26 to 56 wall seconds per simulated second, with three sessions side by side at 73 to 78, and under the load
this environment's own overnight records were made at 82 to 205. When r
reaches DESIGN_RATIO - 5 no command length is planned inside the limit: floor_from then returns its cap FLOOR_MAX_S
(2 s) and the command is accepted and runs slower than 200x. It is NOT refused: no command could be inside the limit
then, so a refusal would leave the session unusable for as long as the machine is loaded, and a session learns that the
load has gone only by running a command. Instead the policy is told, in two places (_budget, _command_timing):
  capabilities   hardware.budget: the limit, the physics ratio this session has measured (the dearest of its last
                 three policy commands, and the last command of 20 ms or more), what a command at the floor in force
                 is expected to cost, inside_budget (true/false; null before any command has been timed), and how many
                 of the session's timed commands ran over;
  every reply    the observation's proprioception command_timing: wall_s, sim_s, wall_per_sim, physics_wall_per_sim
                 and over_budget (1 when the command that this observation follows ran over the limit, else 0).
The owner reads the same in the capability notes `budget` and `command_cost`.

Refusals leave the session usable. A policy request - one command or a batch {"actions": [...]} - is checked whole
when it is decoded, before its first command runs, for everything that does not depend on where the arms will be
(SyringePolicyAccess.decode_actions -> _precheck: each command's duration against the floor and the tick, arm names, joint
names and limits, finger widths, grip forces, keyboard chords); a refused command carries policy_reason, so the policy
is told the floor in force. What cannot be known beforehand is the inverse-kinematics solve of a Cartesian or
keyboard-move command that is not the first of its batch: it is solved from the pose the earlier commands leave, and if
it is unreachable then, those commands have run and the gateway answers partial true with the session still ready
(http_server.py recover_known_partial_batches). An owner reset whose selectors, seed, variant or robot/task pair are refused raises
ActionRejectedError before anything is touched (the drilling backend's pattern).

The recorded state. get_state() is what the gateway records before and after every command (and returns at
/v1/state); it does not hold the task evaluation (ingredient_mixing's rule). Nothing in the gateway or the trace reads an evaluation from the state: /v1/evaluation, `gar
verify` and the benchmark call evaluate().
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

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, Keyboard, RawAction
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

RAW_HELP = {
    "required": ["type", "request"],
    "request": {"op": "set_load", "arm": "left | right", "mass_kg": "0 .. the arm's payload less its hand",
                "com_m": "[x, y, z] of the load's centre of mass in the TCP frame", "duration_s": "optional"},
    "example": {"type": "raw", "request": {"op": "set_load", "arm": "left", "mass_kg": 1.0, "com_m": [0, 0, 0.05]}},
    "note": "The one raw operation: set_load tells an arm's controller what its hand carries (libfranka's setLoad). "
            "See hardware.load of the capabilities.",
}


@dataclass(frozen=True)
class SyringePolicyAccess(PolicyAccess):
    """PolicyAccess whose decode step also runs the backend's state-independent checks on every command of the
    request. The gateway decodes a request before it runs any of it (http_server.py action()), so a batch with a
    refused command anywhere in it is refused whole and nothing has moved (drilling's DrillingPolicyAccess).

    It also admits the raw action, which the shared allowlist (policy_access.ROBOT_ACTIONS) does not hold: here raw
    carries one robot-controller operation, set_load, and nothing else (the backend's _plan refuses any other op). A
    raw item is decoded by the shared codec (exactly the fields type and request); every other item goes through the
    shared allowlist and guard rules unchanged."""
    precheck: Callable | None = None

    def decode_actions(self, payload):
        from ..action_codec import actions_from_request
        raw = lambda item: isinstance(item, dict) and item.get("type") == "raw" and "raw" in self.action_types
        items = payload.get("actions") if isinstance(payload, dict) and "actions" in payload else None
        if items is None and raw(payload):
            actions = actions_from_request(dict(payload))
        elif isinstance(items, list) and items and set(payload) == {"actions"} and any(raw(i) for i in items):
            if any(not isinstance(i, dict) or "actions" in i for i in items):
                raise ValueError("Invalid action")
            actions = tuple(actions_from_request(dict(i)) if raw(i) else super(SyringePolicyAccess, self).decode_actions(i)
                            for i in items)
        else:
            actions = super().decode_actions(payload)
        if self.precheck is not None:
            self.precheck(actions if isinstance(actions, tuple) else (actions,))
        return actions

    def capabilities(self):
        out = super().capabilities()
        if "raw" in self.action_types:
            out["action_types"] = sorted(set(out["action_types"]) | {"raw"})
        return out

    def help(self):
        out = super().help()
        if "raw" in self.action_types:
            out["action_types"]["raw"] = dict(RAW_HELP)
            out["effective_capabilities"] = self.capabilities()
        return out


# bay_rgb: the ceiling recording camera, the one that sees the wall clock slow-push's instruction times the push by
POLICY_CAMERAS = ("station_rgb", "overhead_rgb", "left_wrist_rgb", "right_wrist_rgb", "bay_rgb")
DIAGNOSTIC = ("hero", "front", "front_left", "side", "tray_close", "work_close")
SESSION_MARKER = ".gar-syringe-handling-session"
SESSION_RETENTION = 20
# Bounded worker retention between operations. A policy that thinks longer than this between commands pays a worker
# start (7-11 s) in its next command, and a short command is then far over the 200x budget. 15 minutes, as gift_bags
# uses for the same reason; renderer_options idle_release_s overrides it.
IDLE_RELEASE_S = 900.0
# The launch option renderer_options: its keys and their ranges (check_renderer_options). Unknown keys fail (unified
# contract); nothing is clamped.
RENDER_OPTIONS = ("width", "height", "samples", "idle_release_s", "engine")
WIDTH_RANGE = (160, 1920)          # pixels of a policy frame; the cameras' own format is 1280 x 720 (the default)
HEIGHT_RANGE = (90, 1080)
SAMPLES_RANGE = (1, 256)           # path-traced samples per pixel of a policy frame (default: config.policy_samples, 16)
IDLE_RELEASE_MAX_S = 86400.0       # a day; for longer there is null (the worker is kept until the session is stopped)


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"syringe-handling-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


MIN_POLICY_COMMAND_S = 0.020
# Observation cost model behind the published floor, from measurements on an RTX 5090. On a quiet card one policy
# camera's observation took 0.8-0.9 s, all four 2.6-2.8 s (mixture surface included), and physics alone ran at 9-22 wall
# seconds per simulated second. With another session's jobs holding the card at 96-100 %: 1.5-2.9 s for one camera and
# 9.6-9.8 s for four on the largest task, physics at 40-94; with the quiet card's 0.8 s a frame, the largest task with
# four cameras ran at 221x at the floor. These constants are that load's: 2.4 s a frame, 1.0 s for the surface, physics
# at up to 120. With two sessions' jobs on the card they are exceeded (4.1 s a frame, physics at 149, 274x), so the
# floor in force follows the session's own timings (floor_from) and these constants only set where it starts.
# This environment: the constants of the teaching kitchen's loaded card for a frame; the liquid's meshes cost
# milliseconds (SURFACE_S); the physics is a 77-dof rigid world on one core, 1.2 to 2 wall seconds per simulated second
# alone, taken here at 20 for a container whose CPU quota is shared.
FRAME_S_1280x720_16SPP = 2.4
SURFACE_S = 0.05
PHYSICS_RATIO_MAX = 20.0
# The fixed cost of a command, whatever its length (the module docstring's F), where the session's first floor starts.
# Measured through the gateway, with the history and the trace on and four other simulations holding the card at
# 98-99 %: 0.06 to 0.15 s for most commands on every task (the two state records are 1 ms each, as they hold no
# evaluation; the rest is the history's and the trace's writes), with stalls of 0.26, 0.33, 1.09, 1.12 and 1.59 s (the
# history's synchronous writes, dearest after a long command's trace and while the container's CPU quota was reached in
# every period). The constant is the largest stall measured, rounded up: not a bound (1.74 s was measured on the first
# command of a four-camera session, which still ran at 53.5 times real time; after its first command a session's floor
# follows its own measured fixed cost). With 0.30 s a stall of 1.12 s on a 5 ms command would have cost 310 times real
# time.
FIXED_S = 1.6
# What a command costs outside the backend's own timing (the request before it is decoded, the reply's encoding and
# sending, the client's image downloads): the client's wall time less the backend's interval: 0.002 to 0.007 s
# without and with one camera (four-camera downloads: 0.012 s). A margin over those.
UNSEEN_S = 0.05
# A render worker's start with its first frame, where the estimate starts (7-9 s and a first frame of 1-3 s on a quiet
# card); the session's own latest start replaces it when it was dearer (the first image of every session is one:
# 16 and 18 s under the load above). The command after an idle release pays it.
WORKER_START_S = 12.0


BUDGET = 200.0                    # wall seconds per simulated second a command with its observation may cost
# Planned for 120x: at a design ratio of 180 a four-camera session ran at 202.40x.
# This is headroom, not a guarantee under arbitrary shared-machine contention. (A tenth on the observation alone,
# T >= 1.1 obs / (BUDGET - r), plans 194 at a physics ratio of 127 to 134 with four other simulations on the card, and a
# command ran at 185.)
DESIGN_RATIO = 120.0
FLOOR_MAX_S = 2.0                 # the floor is never raised above this (the machine cannot hold the budget then)


def floor_from(n_cameras, observation_s, physics_ratio, published, fixed_s=0.0, worker_start_s=0.0):
    """The shortest policy command (s) that keeps a command, the records round it and the observation it returns under
    BUDGET, from what the session itself has just measured: `fixed_s` wall seconds a command costs whatever its
    length, `observation_s` for an observation (not counted without cameras), `physics_ratio` wall seconds of physics
    per simulated second, and `worker_start_s` when the next command has to start a render worker. Never below the
    `published` floor (the session's first, from the constants above); planned for DESIGN_RATIO, with 40 percent headroom under the
    budget; capped at FLOOR_MAX_S. A fixed floor holds only for the load its constants were measured under: with two
    sessions' jobs on the card the largest task with four cameras ran at 274x at the published floor."""
    cost = float(fixed_s) + ((float(observation_s) + float(worker_start_s)) if n_cameras else 0.0)
    if cost <= 0.0:
        return float(published)
    if physics_ratio >= DESIGN_RATIO - 5.0:
        return FLOOR_MAX_S
    need = math.ceil(1000.0*cost/(DESIGN_RATIO - physics_ratio) - 1e-9)/1000.0
    return min(FLOOR_MAX_S, max(float(published), need))


def min_command_s(n_cameras, width=1280, height=720, samples=16):
    """Shortest policy command (s) whose wall time, the fixed cost of a command plus its physics plus the observation
    it returns, stays under 200x wall/sim at the constants above: T * PHYSICS_RATIO_MAX + FIXED_S + UNSEEN_S + obs
    <= 200 T. Rounded up to whole milliseconds. With cameras never below MIN_POLICY_COMMAND_S; without cameras the
    observation costs nothing and the floor is the fixed cost's alone (a few milliseconds)."""
    obs = SURFACE_S + n_cameras*FRAME_S_1280x720_16SPP*(width*height/(1280*720))*(samples/16.0) if n_cameras else 0.0
    need = math.ceil(1000.0*(FIXED_S + UNSEEN_S + obs)/(BUDGET - PHYSICS_RATIO_MAX) - 1e-9)/1000.0
    return max(MIN_POLICY_COMMAND_S, need) if n_cameras else need


def check_renderer_options(options, n_cameras=1, policy_samples=None, default_samples=16):
    """The launch option renderer_options, checked before anything is built. Returns a new dict with every key present
    (width, height, samples, idle_release_s); raises UnsupportedConfigurationError with the reason otherwise.

    Unknown keys fail and nothing is clamped (unified contract). `samples` and config.policy_samples are two names for
    the same number: `policy_samples` is the config's value when the launch gave one (they must then agree),
    `default_samples` the value in force when neither was given. `n_cameras` is the number of policy cameras: the
    session's first floor grows with cameras x pixels x samples, and a combination whose first floor would be above
    FLOOR_MAX_S is refused (floor_from never raises the floor above that cap, so such a session's floor could not
    cover its own observation)."""
    import difflib
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise UnsupportedConfigurationError("renderer_options must be an object with keys from "
                                            f"{', '.join(RENDER_OPTIONS)} (got {type(options).__name__})")
    unknown = sorted(str(k) for k in set(options) - set(RENDER_OPTIONS))
    if unknown:
        near = {k: difflib.get_close_matches(k, RENDER_OPTIONS, n=1, cutoff=0.6) for k in unknown}
        hint = "; ".join(f"{k!r}: did you mean {m[0]!r}?" for k, m in near.items() if m)
        raise UnsupportedConfigurationError(f"Unknown renderer_options {unknown}; the options are "
                                            f"{', '.join(RENDER_OPTIONS)}" + (f" ({hint})" if hint else "")
                                            + ". Nothing was built")

    def whole(name, value, lo, hi):
        if type(value) is not int or not lo <= value <= hi:
            raise UnsupportedConfigurationError(f"{name} must be an integer {lo}..{hi} (got {value!r})")
        return value
    out = dict(width=whole("renderer_options.width", options.get("width", 1280), *WIDTH_RANGE),
               height=whole("renderer_options.height", options.get("height", 720), *HEIGHT_RANGE))
    if policy_samples is not None:
        whole("config.policy_samples", policy_samples, *SAMPLES_RANGE)
    if "samples" in options:
        whole("renderer_options.samples", options["samples"], *SAMPLES_RANGE)
        if policy_samples is not None and policy_samples != options["samples"]:
            raise UnsupportedConfigurationError(
                f"config.policy_samples ({policy_samples}) and renderer_options.samples ({options['samples']}) both set "
                "the samples per pixel of a policy frame and disagree; give one of them")
    out["samples"] = whole("the samples per pixel of a policy frame",
                           options.get("samples", policy_samples if policy_samples is not None else default_samples),
                           *SAMPLES_RANGE)
    idle = options.get("idle_release_s", IDLE_RELEASE_S)
    if idle is not None and (isinstance(idle, bool) or not isinstance(idle, (int, float)) or not math.isfinite(idle)
                             or not 0 <= idle <= IDLE_RELEASE_MAX_S):
        raise UnsupportedConfigurationError(
            f"renderer_options.idle_release_s must be a number of seconds 0..{IDLE_RELEASE_MAX_S:g} (0: the render "
            "worker is not kept between operations) or null (it is kept until the session is stopped); got "
            f"{idle!r}")
    out["idle_release_s"] = None if idle is None else float(idle)
    out["engine"]=options.get("engine","mujoco")
    if out["engine"] not in ("mujoco","isaac"):
        raise UnsupportedConfigurationError("renderer_options.engine must be mujoco or isaac")
    first = min_command_s(n_cameras, out["width"], out["height"], out["samples"])
    if first > FLOOR_MAX_S:
        raise UnsupportedConfigurationError(
            f"renderer_options {out['width']} x {out['height']} at {out['samples']} samples per pixel with {n_cameras} "
            f"policy camera(s) would put the session's first command floor at {first:g} s, above the {FLOOR_MAX_S:g} s "
            f"the floor is ever raised to (by the cost model a command of {FLOOR_MAX_S:g} s with its observation would "
            f"be over the {BUDGET:g}x limit): choose fewer samples, fewer pixels or fewer cameras")
    return out


class NewtonSyringeHandlingBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset=None, task=None, seed=0, variant=None, contract=None, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from syringe_handling.config import Config, TASKS
        from syringe_handling.env import SyringeEnv
        from syringe_handling.kinematics import ArmKinematics
        from syringe_handling.variants import resolve
        from syringe_handling import robot_setup
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["syringe-handling"]
        qualified = tuple(catalog["aliases"])
        if isinstance(robot, str) and robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"syringe-handling does not support {robot!r}: {catalog['unsupported'][robot]}")
        if not isinstance(robot, str) or robot not in qualified:
            # a rig's own name or a gar.robot-setup.v1 object equal to an offered rig -> its alias; anything else is
            # refused with what it would need (a rig that builds and is offered for no task: with that reason)
            try:
                robot = robot_setup.validate(robot)
            except ValueError as exc:
                raise UnsupportedConfigurationError(str(exc)) from exc
        if robot not in qualified:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; offered: {', '.join(qualified)}")
        values = dict(config or {})
        if "layout" in values:
            raise UnsupportedConfigurationError("layout is an owner/probe field, not a launch option")
        # the variant is a catalogue selector (`variant`) and, as before, a field of config; both given: they must agree
        if variant is not None and values.get("variant") not in (None, "") \
                and values["variant"] != (variant if variant != "plain" else ""):
            raise UnsupportedConfigurationError(f"Variant supplied twice ({variant!r} and config.variant "
                                                f"{values['variant']!r}); choose one")
        try:
            if contract is not None and "contract" in values and values["contract"] != contract:
                raise ValueError("contract and config.contract must agree")
            sel = resolve(preset, task, int(seed), variant if variant is not None else values.get("variant", ""),
                          contract if contract is not None else values.get("contract", "v1"))
            robot_setup.check_task(robot, sel["preset"], sel["task"])
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        values.update(sel, robot=robot)
        if device not in (None, "cpu"):
            raise UnsupportedConfigurationError("syringe-handling physics supports device=cpu only; rendering engine is selected separately")
        samples_given = values.get("policy_samples")          # None unless the launch's config named it
        self.config = Config.from_dict(values)
        self.cameras = tuple(cameras if cameras is not None else ("station_rgb",))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; others are privileged diagnostics")
        rig_arms = robot_setup.arms_of(robot)                # the wrist cameras follow the rig's arms
        for cam in self.cameras:
            side = cam.split("_")[0]
            if cam.endswith("_wrist_rgb") and side not in rig_arms:
                raise UnsupportedConfigurationError(f"{robot} has no {side} arm (its arms: {', '.join(rig_arms)}): it "
                                                    f"has no {cam}")
        # every key present and in range from here on (unknown keys and values out of range are refused, before the
        # session folder is made and anything is built)
        self.options = check_renderer_options(renderer_options, len(self.cameras), samples_given,
                                              self.config.policy_samples)
        self.idle_release_s = self.options["idle_release_s"]  # seconds; 0: the worker is not kept; None: kept until stop
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("syringe-handling session (auto-created; pruned by retention)\n")
        self.env = SyringeEnv(self.config, output_dir=self.output)
        self.ik = None if self.config.robot == "g1" else ArmKinematics(self.env.world)
        self.arms = tuple(self.env.world.rig.instances)
        self.images = deque()
        self.last_timing = {}
        self._renderer = None                                # the worker of the operation under way
        self._prestart = None                                # a worker being started beside a command's physics
        self._retained = None                                # the worker kept between operations (idle clock running)
        self._retain_timer = None
        self._retain_token = None
        self._suspect = None                                 # a worker whose last frame raised a ValueError (_frame)
        self._reapers = []
        # what the session's render worker has been through, for the owner (worker_status, state field render_worker)
        self.worker_events = dict(starts=0, lost=0, frames=0, frame_retries=0, released=[], last_fault=None,
                                  last_start_s=None)
        self.trace = None
        if recording and self.config.record_trace:
            from syringe_handling.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options["width"], self.options["height"]
        mounted = set(self.env.world.scene.cameras)
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            if name not in mounted:
                continue
            fov = self.env.world.scene.cameras[name]["fov"]
            focal = .018/math.tan(math.radians(fov/2))
            mount = ("link7_d405_envelope" if "wrist" in name else "ceiling_arm" if name == "overhead_rgb" else
                     "ceiling_bracket" if name == "bay_rgb" else "services_trunking")
            self.camera_specs[name] = CameraSpec(name, mount, width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target"}) if robot == "g1" else frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard", "raw"})
        self.min_command_published = min_command_s(len(self.cameras), width, height, self.options["samples"])
        self._floor_measured = self.min_command_published    # follows the session's own timings (_note_cost)
        self._costs = []                                     # the last commands' (observation s, physics ratio, fixed s)
        self._physics_ratio = None
        self._cycle = None                                   # the command being timed (see _precheck, _note_cost)
        self._worker_start_s = WORKER_START_S                # the session's latest worker start, never below the constant
        self.cost = {}                                       # the last command's cost as the session measured it
        self._timing = dict(self._NO_TIMING)                 # what the policy is told of it (_command_timing)
        self._timed = 0                                      # policy requests timed in this session ...
        self._over = 0                                       # ... and how many of them ran over BUDGET
        self.arm_model = "G1" if robot == "g1" else "Panda" if "panda" in robot else "FR3"
        w0 = self.env.world
        self.grip_limit_n = 0. if robot == "g1" else 2.0*min(float(w0.joints[w0.gripper_joints(a)[0]]["effort_limit"]) for a in self.arms)
        self._capabilities = Capabilities(
            "syringe-handling", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": self._task_note(),
                "clock": f"One tick = {self.config.rigid_dt*1000:g} ms (MuJoCo rigid step with the liquid network solved in it); every observation "
                         "carries the simulated time (time_s), which is also what the room's wall clock shows",
                "control": f"Physical position servos following a jerk-limited joint trajectory within the {self.arm_model}'s "
                           "velocity, acceleration and jerk limits; IK computes targets only",
                "arms": ",".join(self.arms),
                "gripper": "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode "
                           f"grasp (total N of both fingers, <= {self.grip_limit_n:g} N; 50 mm/s until blocked, then force)",
                "liquid": "every syringe, vial and line holds parcels of named liquids and gas; a plunger pushes against "
                          "the pressure the network gives back (seal, needle or cannula, trapped air, a vial's headspace, "
                          "tissue); the evaluation is computed from where the liquid is and when it arrived",
                "proprioception": "joint positions, velocities and measured joint torques of each arm; finger widths; "
                                  "robot_status: each arm's protective-stop time left and the load its controller has "
                                  "been told it carries (load_kg, load_com_m), the count of refused targets; "
                                  "no liquid state, pressure or plunger position (the dose is read from the barrel's "
                                  "scale in the cameras); "
                                  "command_timing: what the last policy command cost (see the note command_timing)",
                "command_timing": (
                    "Every observation carries proprioception command_timing, the cost of the last policy request "
                    "(one command or a batch) before it as this session timed it: wall_s (wall seconds from the "
                    "request's decoding to the end of its observation), sim_s (simulated seconds it ran), wall_per_sim "
                    "(their quotient), physics_wall_per_sim (the wall time of its physics alone over sim_s) and "
                    f"over_budget (1 when wall_s plus {UNSEEN_S:g} s, allowed for what the session cannot time: the "
                    f"request before it is decoded, the reply, the image downloads, exceeds {BUDGET:g} x sim_s; else "
                    "0). All zero before the first policy command and after a reset or a checkpoint restore. It is "
                    "the interface's own clock and says nothing of the scene"),
                "keyboard": "Game-style chords held for ticks/rate_hz s (<= 10 s): W/S A/D R/F move the TCP along world "
                            "+y/-y -x/+x +z/-z at linear_speed_m_s; I/K J/L U/O rotate it about world x, y, z at "
                            "angular_speed_rad_s; ALT x0.2, SHIFT x6; Q closes, E opens the fingers (P x0.2, T x10); "
                            "H grips in force mode (35 N total; with G 10 N, with B 70 N) until E",
                "cameras": f"Ideal pinhole RGB from {self.options['engine']} (mujoco: software EGL; isaac: authored RTX appearance); no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions",
                "equipment": "the owner's state (GET /v1/state) has field equipment: each syringe's plunger, contents, "
                             "pressure and Luer joint, each needle's sheath, track and record, the vial, the lock and its "
                             "line, liquid by place; the policy's observation has none of it",
                "instruments": self._instruments_note(),
                "min_command_s": "",                         # written by the capabilities property (the floor in force)
                "budget": "",                                # written by the capabilities property (_budget_note)
                "render_worker": "",                         # written by the capabilities property (_worker_note)
                "renderer_options": (
                    "Launch option, an object with these keys and no others. engine (mujoco, default, or isaac). width (integer "
                    f"{WIDTH_RANGE[0]}..{WIDTH_RANGE[1]}, default 1280) and height (integer {HEIGHT_RANGE[0]}.."
                    f"{HEIGHT_RANGE[1]}, default 720): the pixels of every policy frame. samples (integer "
                    f"{SAMPLES_RANGE[0]}..{SAMPLES_RANGE[1]}, default 16): path-traced samples per pixel of a policy "
                    "frame for Isaac only (ignored by software EGL), the same number as config.policy_samples (given both, they must agree). idle_release_s (a "
                    f"number of seconds 0..{IDLE_RELEASE_MAX_S:g} or null, default {IDLE_RELEASE_S:g}): how long the "
                    "render worker is kept after the session's last operation; 0: not kept, every observation starts a "
                    "worker; null: kept until the session is stopped. An unknown key, a value of the wrong type or out "
                    "of range is refused at launch, before anything is built, with the reason; so is a combination of "
                    f"cameras, pixels and samples whose first command floor would be above {FLOOR_MAX_S:g} s. In force "
                    f"in this session: width {self.options['width']}, height {self.options['height']}, samples "
                    f"{self.options['samples']}, idle_release_s "
                    + ("null" if self.idle_release_s is None else f"{self.idle_release_s:g}")),
                "load": "raw action {'type': 'raw', 'request': {'op': 'set_load', 'arm': ARM, 'mass_kg': M, 'com_m': "
                        "[x, y, z]}}: tells the arm's controller what its hand carries, as libfranka's setLoad does (mass "
                        f"0..{self._load_limit_kg():g} kg, the arm's payload less its hand; centre of mass in the TCP "
                        f"frame, within {self._load_com_max_m():g} m). The controller compensates that load's weight and "
                        "leaves it out of the external torque its collision reflex judges; whatever the hand really "
                        "holds beyond it the joints feel. Nothing is configured at the start and after a reset "
                        "(0 kg); the simulator does not set it for the robot. The command changes no joint target: the "
                        "arms hold for the floor in force (min_command_s; optional duration_s in the request, a "
                        f"multiple of {self.config.rigid_dt*1000:g} ms, for longer; an owner's set_load holds one tick unless it says otherwise). "
                        "set_load is the only raw operation. The configured load is in the observation: "
                        "proprioception robot_status.arms.ARM.load_kg and load_com_m",
                "batches": "A policy request {'actions': [...]} is checked whole before its first command runs: every "
                           f"command's duration against the floor in force and the {self.config.rigid_dt*1000:g} ms tick, arm names, joint names "
                           "and limits, finger widths, grip forces and keyboard chords; a refused request has moved "
                           "nothing and the refusal names the command. Not known beforehand: whether a Cartesian or "
                           "keyboard-move command that is not the first of its batch is reachable from the pose the "
                           "earlier commands leave. If it is not, the commands before it have run, it and the ones "
                           "after it have not, and the session stays ready with no recovery needed: the response has "
                           "committed true, partial true, completed_actions, rejected_action_index, "
                           "unexecuted_action_indices and the observation after the completed commands",
                "state": "GET /v1/state and the history's before/after records hold the rigid state, joints, the "
                         "equipment and events; the task evaluation is not in them: GET /v1/evaluation or gar verify "
                         "computes it on request",
                "checkpoints": "In memory for this session only; a checkpoint restores only into the variation (preset, "
                               "task, seed, variant, robot) it was saved in",
                "trace": ("recording: every tick of every command is written to the session folder's physics.sqlite3 "
                          "(GET /v1/history/{id}/trace; body poses and velocities, joints, and every 20 ticks the "
                          "equipment's state). Launch with config record_trace false to run without it"
                          if self.trace is not None else
                          "not recording (config record_trace false, or recording false): no per-tick trace is kept")})

        if robot == "g1":
            self._capabilities.notes.update(
                control="Supported-pelvis G1; shared PD profile, URDF torque limits, 1 rad/s reference slew and 4 rad/s2 body / 8 rad/s2 hand acceleration. No walking/balance controller.",
                gripper="Dex3: seven revolute joints per hand, all commanded in radians using g1/left_hand_* and g1/right_hand_* named_joint_target. No half-width or force-grip emulation.",
                keyboard="Not offered for G1; use named joint targets.",
                load="Not offered for G1; no Franka setLoad emulation.",
                proprioception="43 body and Dex3 joint positions, velocities and actuator torques; no liquid/evaluator fields.")

    def _task_note(self):
        c = self.config
        return f"{c.preset}/{c.task} seed {c.seed}" + (f" variant {c.variant}" if c.variant else "")

    # ------------------------------------------------------------------ the command floor
    def _worker_held(self):
        """Whether the next image finds a worker: one in use or kept whose process is alive, or one being started. A
        kept worker that has died does not count (it did: the floor then stayed where a held worker puts it, and the
        command that had to start the replacement was not allowed the time for it)."""
        alive = NewtonSyringeHandlingBackend._alive
        return alive(self._renderer) or alive(self._retained) or self._prestart is not None

    @property
    def min_command_s(self):
        """The floor in force (s). While the session holds no live render worker (cameras on; after the idle release,
        or after its worker has died) the next command starts one, and the floor covers that start as well."""
        if not self.cameras or self._worker_held():
            return self._floor_measured
        obs, ratio, fixed = self._cost_now()
        return max(self._floor_measured, floor_from(len(self.cameras), obs, ratio, self.min_command_published, fixed,
                                                    self._worker_start_s))

    def _cost_now(self):
        """(observation s, physics ratio, fixed s) the floor is computed from: the dearest of the last three commands,
        or the constants before the session has measured any."""
        if not self._costs:
            width, height = self.options.get("width", 1280), self.options.get("height", 720)
            samples = self.options.get("samples", self.config.policy_samples)
            obs = (SURFACE_S + len(self.cameras)*FRAME_S_1280x720_16SPP*(width*height/(1280*720))*(samples/16.0)
                   if self.cameras else 0.0)
            return obs, PHYSICS_RATIO_MAX, FIXED_S + UNSEEN_S
        return (max(c[0] for c in self._costs), max(c[1] for c in self._costs), max(c[2] for c in self._costs) + UNSEEN_S)

    def _floor_note(self):
        floor = self.min_command_s
        n = len(self.cameras)
        obs, ratio, fixed = self._cost_now()
        held = "none (no policy camera)" if not n else ("held" if self._worker_held() else
                                                        "none held (released after the idle time, or lost); the next "
                                                        "command starts one")
        return (f"{floor:g} now ({self.min_command_published:g} at the session's start): policy commands shorter than "
                f"the floor are refused, with or without cameras, because a shorter command exceeds {BUDGET:g}x wall/sim. "
                f"Floor = (fixed cost of a command {fixed:.3f} s + observation {obs:.3f} s"
                + (f" + render worker start {self._worker_start_s:.1f} s" if n and not self._worker_held() else "")
                + f") / (design ratio {DESIGN_RATIO:g} - physics ratio {ratio:.1f}), from the dearest of this session's last three commands "
                f"(the constants before it has run any), never below the start value and at most {FLOOR_MAX_S:g} s. "
                f"Render worker: {held}. A refused command's message carries the floor in force, and so does "
                "hardware.min_command_s of the policy's capabilities. Owner commands are not floored (diagnostic "
                "stepping) and are outside the 200x guarantee. The floor bounds what a short command adds to its "
                "physics; it cannot hold the limit when the physics alone is over it (note `budget`)")

    # ------------------------------------------------------------------ the 200x limit: what the session tells
    _NO_TIMING = dict(wall_s=0.0, sim_s=0.0, wall_per_sim=0.0, physics_wall_per_sim=0.0, over_budget=0)
    BUDGET_NOTE = (
        f"A policy command with the observation it returns should cost at most {BUDGET:g} wall seconds per simulated "
        "second. What holds that is the command floor (min_command_s), and the floor bounds only what a short command "
        "adds to its physics: the fixed cost of a command, its observation, a render worker's start. It does nothing "
        "about the physics itself, whose cost per simulated second follows the machine's load. When the physics alone "
        f"costs {DESIGN_RATIO - 5.0:g} wall seconds per simulated second or more, no command length is planned inside "
        f"the limit: the floor then stands at its cap of {FLOOR_MAX_S:g} s and commands are accepted and run slower than "
        f"{BUDGET:g}x. They are not refused (no command could be inside the limit then, and the session learns that the "
        "load has gone only by running one); the session says so instead: inside_budget of hardware.budget in the "
        "policy's capabilities, and command_timing.over_budget in the proprioception of every observation, which is 1 "
        "when the command it follows ran over")

    def _budget(self):
        """What the session can say about the 200x limit at this moment (the policy's capabilities, hardware.budget):
        the physics ratio it has measured, what a command at the floor in force is expected to cost, and whether that
        is inside the limit. Before any policy command has been timed the ratio is not known and inside_budget is
        None (the floor then stands on the constants, which plan DESIGN_RATIO)."""
        n = len(self.cameras)
        obs, ratio, fixed = self._cost_now()
        floor = self.min_command_s
        start = self._worker_start_s if n and not self._worker_held() else 0.0
        overhead = fixed + ((obs + start) if n else 0.0)
        expected = ratio + overhead/floor
        measured = bool(self._costs)
        return dict(
            limit_wall_per_sim=BUDGET,
            physics_wall_per_sim=round(ratio, 1) if measured else None,
            physics_wall_per_sim_last=None if self._physics_ratio is None else round(self._physics_ratio, 1),
            min_command_s=floor, min_command_cap_s=FLOOR_MAX_S, floor_at_its_cap=bool(floor >= FLOOR_MAX_S),
            command_overhead_s=round(overhead, 3),
            at_the_floor_wall_per_sim=round(expected, 1) if measured else None,
            inside_budget=bool(expected <= BUDGET) if measured else None,
            commands_timed=self._timed, commands_over_budget=self._over,
            fields=("physics_wall_per_sim: wall seconds of physics per simulated second, the dearest of this "
                    "session's last three policy commands (null before the first); physics_wall_per_sim_last: the last "
                    "command of 20 ms or more, whoever sent it. command_overhead_s: what a command costs besides its "
                    "physics (fixed cost, observation, a render worker's start while none is held). "
                    "at_the_floor_wall_per_sim: what a command of min_command_s is expected to cost now; a longer "
                    "command costs less per simulated second, never less than the physics ratio. inside_budget: "
                    "whether that expectation is within the limit (null before the first timed command). "
                    "commands_timed / commands_over_budget: this session's policy requests so far"),
            note=self.BUDGET_NOTE)

    def _budget_note(self):
        b = self._budget()
        if b["inside_budget"] is None:
            now = (f"no policy command has been timed yet; the floor stands on the start values (physics "
                   f"{PHYSICS_RATIO_MAX:g}, planned {DESIGN_RATIO:g}x)")
        else:
            now = (f"physics {b['physics_wall_per_sim']:g} wall s per simulated s (the dearest of the last three policy "
                   f"commands); a command at the floor in force ({b['min_command_s']:g} s) is expected at "
                   f"{b['at_the_floor_wall_per_sim']:g}x; inside the limit: {'yes' if b['inside_budget'] else 'NO'}"
                   + ("" if b["inside_budget"] else " (commands are accepted and run slower than the limit)"))
        return (f"{self.BUDGET_NOTE}. Now: {now}. Of {b['commands_timed']} policy request(s) timed in this session "
                f"{b['commands_over_budget']} ran over the limit. The policy reads the same in hardware.budget of its "
                "capabilities")

    def _command_timing(self):
        """What the policy is told of the command an observation follows (proprioception command_timing; numbers only,
        as the policy's filter requires): zeros before any command and after a reset."""
        return dict(self._timing)

    def _worker_note(self):
        s = self.worker_status()
        if not self.cameras:
            now = "this session has no policy camera and holds no worker (an owner's render starts one)"
        elif s["alive"]:
            now = f"a worker is held (pid {s['pid']})"
        elif s["starting"]:
            now = "a worker is being started"
        else:
            now = "no worker is held; the next command starts one beside its physics"
        idle = self.idle_release_s
        kept = ("kept until the session is stopped (idle_release_s null)" if idle is None else
                "not kept between operations (idle_release_s 0)" if idle <= 0.0 else
                f"kept for {idle:g} s after the session's last operation (idle_release_s), then given back")
        return ("One Isaac render worker per session, started beside a command's physics and " + kept + ". A worker "
                "whose process has died is replaced: it is dropped when it is next looked at, stops counting as held "
                "(the floor in force then covers a worker start, as after an idle release), and the next command "
                "starts a new one beside its physics. A worker that fails a frame is ended and the frame is asked once "
                "more of a new worker; if that fails as well the command stands (committed, observation null), the "
                "session is left without a worker and the next image request starts one. POST /v1/recover and a reset "
                "replace a dead worker the same way; nothing waits for the idle release. The owner's state has the "
                f"worker's record (render_worker: held, alive, pid, starts, lost, frame_retries, last_fault). Now: "
                f"{now}; workers started {s['starts']}, lost {s['lost']}, frames asked a second time "
                f"{s['frame_retries']}"
                + (f"; last fault: {s['last_fault']}" if s["last_fault"] else ""))

    @property
    def capabilities(self):
        """The session's capabilities; notes.min_command_s is the floor in force at the time of the request, and
        notes.budget, command_cost and render_worker are of that moment as well."""
        notes = self._capabilities.notes
        notes["min_command_s"] = self._floor_note()
        notes["budget"] = self._budget_note()
        notes["render_worker"] = self._worker_note()
        notes["command_cost"] = (
            "the last command as this session timed it (s): " + ", ".join(f"{k} {v:g}" for k, v in self.cost.items())
            if self.cost else "no command timed yet")
        return self._capabilities

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        """Reset to the session's task, or - with options {preset, task, variant} and/or a seed - to another catalogue
        variation: the world is rebuilt for it in this process (about the cost of a session start's physics part,
        30-45 s, but the server, its keys and history stay). A variant not named stays while the task stays and is
        dropped when the task changes; variant "plain" (or "") is the task as listed."""
        from dataclasses import replace
        from syringe_handling.env import SyringeEnv
        from syringe_handling.kinematics import ArmKinematics
        # A refused selection (an unknown option, preset, task or variant, a task of another preset, a variant of
        # another task, a seed outside the catalogue, a task this robot is not offered for) is decided here, before
        # anything is touched, and raised as ActionRejectedError: the gateway then answers 400 and keeps the session
        # as it was. Any other error out of a reset is recorded by the gateway as a runtime fault that only a valid
        # reset or the owner's recovery clears.
        try:
            sel = self._reset_selection(seed, options)
        except (ValueError, TypeError) as exc:
            raise ActionRejectedError(f"Reset refused, nothing changed: {exc}") from exc
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        self._cycle = None
        self._timing = dict(self._NO_TIMING)                  # the observation of a reset follows no command
        c = self.config
        if all(sel[k] == getattr(c, k) for k in ("preset", "task", "seed", "variant", "contract")):
            return self._without_evaluation(self.env.reset()), {"continuity": "reset", "variation": sel}
        config = replace(self.config, **sel)
        candidate = SyringeEnv(config, output_dir=self.output)
        with self.env.lock:
            self._end_operation(final=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
            old, self.env, self.config = self.env, candidate, config
            candidate.generation = old.generation + 1
            self.ik = None if self.config.robot == "g1" else ArmKinematics(candidate.world)
            self._scene_path = None
            if self.trace:
                path = self.trace.store.path
                self.trace.close()
                from syringe_handling.trace import Trace
                self.trace = Trace(candidate, path)
            self._capabilities.notes["task"] = self._task_note()
            self._capabilities.notes["instruments"] = self._instruments_note()
            # another scene: what the last task's commands cost says nothing about this one's
            self._costs, self._physics_ratio, self._floor_measured = [], None, self.min_command_published
        del old
        return self.get_state(), {"continuity": "explicit_reset", "variation": sel}

    def _reset_selection(self, seed, options):
        """The variation a reset request selects, checked against the catalogue and the session's robot; ValueError
        when it is refused. Nothing is changed."""
        from syringe_handling import robot_setup
        from syringe_handling.variants import resolve, variant_name, variant_tasks
        values = dict(options or {})
        unknown = set(values) - {"preset", "task", "variant", "contract"}
        if unknown:
            raise ValueError("Reset selectors are preset, task, variant, contract (options) and seed; got "
                             f"{sorted(map(str, unknown))}")
        for key in ("preset", "task", "variant", "contract"):
            if values.get(key) is not None and not isinstance(values[key], str):
                raise ValueError(f"Reset option {key} must be a name (string)")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("Reset seed must be an integer 0..9999")
        c = self.config
        named_task, named_preset = values.get("task") is not None, values.get("preset") is not None
        preset = values["preset"] if named_preset else (None if named_task else c.preset)   # a task alone selects its preset
        task = values["task"] if named_task else (c.task if preset == c.preset else None)
        if values.get("variant") is not None:
            variant = values["variant"]
            if not named_task and not named_preset:
                # a variant alone: this task with that variant; if it belongs to one other task, that task (as in the
                # catalogue: `--select variant=subcut` is inject as a subcutaneous dose); otherwise resolve() refuses it
                only = variant_tasks(variant_name(variant))
                preset, task = (c.preset, c.task) if (c.task in only or len(only) != 1) else (None, None)
        else:                                                 # not named: it stays while the task stays
            keep = preset in (None, c.preset) and task in (None, c.task)
            variant = c.variant if keep else ""
        same_task = preset in (None, c.preset) and task in (None, c.task)
        selected_contract = values.get("contract")
        if selected_contract is None:
            selected_contract = c.contract if same_task else "v1"
        sel = resolve(preset, task, c.seed if seed is None else seed, variant,
                      selected_contract)
        robot_setup.check_task(c.robot, sel["preset"], sel["task"])
        return sel

    def select_variation(self, request):
        from dataclasses import replace
        from syringe_handling.variants import resolve
        from syringe_handling.task import instruction
        unknown = set(request) - {"preset", "task", "seed", "variant", "contract"}
        if unknown:
            raise ValueError(f"Selectors are preset, task, seed, variant and contract; got {sorted(map(str, unknown))}")
        selected_contract = request.get("contract")
        sel = resolve(request.get("preset", None if ("task" in request or "variant" in request) else self.config.preset),
                      request.get("task"), request.get("seed", 0), request.get("variant"),
                      "v1" if selected_contract is None else selected_contract)
        return dict(variation=sel, task=instruction(replace(self.config, **sel)),
                    reset_request=dict(seed=sel["seed"], options={"preset": sel["preset"], "task": sel["task"],
                                                                  "variant": sel["variant"] or "plain", "contract": sel["contract"]}))

    @staticmethod
    def _without_evaluation(state):
        state.pop("evaluation", None)
        return state

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
        """The privileged state, as the gateway records it before and after every command and returns at /v1/state.
        Without the task evaluation (ingredient_mixing's rule, kept: the evaluation is asked for, not recorded) and nothing
        in the gateway, the history or the trace reads it from the state; evaluate() computes it on request. With
        the render worker's record (worker_status), so that an owner can see whether the session holds a worker, which
        process it is and what has happened to it."""
        state = self.env.get_state(evaluate=False)
        state["render_worker"] = self.worker_status()
        return state

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self.env.evaluate()

    # ------------------------------------------------------------------ render workers
    def _scene(self):
        from syringe_handling.render import export
        if getattr(self, "_scene_path", None) is None or not self._scene_path.is_file():
            self._scene_path, self.env.render_meta = export(self.env.world, self.output/"render", seed=self.config.seed)
        return self._scene_path

    def _new_renderer(self):
        from syringe_handling.render import Renderer
        o = self.options
        t0 = time.perf_counter()
        r = Renderer(self.env.world, self.output/"render", self._scene() if o["engine"]=="isaac" else None, width=o["width"], height=o["height"],
                     samples=o["samples"],engine=o["engine"])
        self.worker_events["starts"] += 1
        self.worker_events["last_start_s"] = round(time.perf_counter() - t0, 3)
        return r

    @staticmethod
    def _alive(r):
        """Whether r is a worker that can be asked for a frame: there is one, it is not closed and its process is
        running (a stand-in without a process counts as running)."""
        if r is None or getattr(r, "closed", False):
            return False
        p = getattr(r, "process", None)
        return p is None or p.poll() is None

    def _note_fault(self, text):
        self.worker_events["last_fault"] = str(text)[:300]

    def _note_release(self, reason):
        released = self.worker_events["released"]
        released.append(reason)
        del released[:-16]

    def _end_worker(self, r, reason, wait=False):
        """End a worker that cannot be asked politely (its process has exited, or it failed a frame): TERM, then KILL,
        and reap exactly this session's own child (Renderer.close -> render_lifecycle.stop_worker; no other process is
        signalled). A process that has exited is reaped at once; one that hangs takes up to 10 s. With `wait` it is
        done before returning (the frame's second attempt starts a worker next, and the card should have the old
        one's memory back by then); otherwise in a thread that close() and a rebuild join."""
        def end():
            try:
                r.close(graceful=False)
            except Exception as exc:                          # the worker is gone either way; keep what happened
                self._note_fault(f"close after {reason}: {type(exc).__name__}: {exc}")
        if wait:
            return end()
        t = threading.Thread(target=end, name="mix-render-end", daemon=False)
        t.start()
        self._reapers = [x for x in self._reapers if x.is_alive()] + [t]

    def _drop_dead(self):
        """Close and drop a worker of this session, in use or kept, whose process has exited. Returns whether there
        was one. Called wherever a kept worker would be relied on: when a command begins (its replacement then starts
        beside the command's physics) and when an image is wanted."""
        dropped = False
        for attr in ("_renderer", "_retained"):
            r = getattr(self, attr)
            if r is None or self._alive(r):
                continue
            setattr(self, attr, None)
            if attr == "_retained":
                self._stop_idle_clock()
            code = getattr(getattr(r, "process", None), "returncode", None)
            self._note_fault(f"the render worker's process had exited (code {code})")
            self._note_release("worker_exited")
            self.worker_events["lost"] += 1
            self._end_worker(r, "worker_exited")
            dropped = True
        return dropped

    def _start_renderer_early(self):
        self._drop_dead()                                     # a kept worker that has died is replaced here
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
        holder["thread"] = threading.Thread(target=work, name="mix-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _operation_renderer(self):
        """The worker for this operation: the one in use, the kept one, the one started beside the command's physics,
        or a new one (a cold start). A worker whose process has exited is dropped first and is never handed back
        (it was: every frame of a session whose worker had died then failed until the idle release)."""
        self._drop_dead()
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
        """After an operation: keep its worker for the next one (for idle_release_s; None: until the session is
        stopped), or give everything back (final, idle_release_s 0, or a worker being started that nobody waited
        for). Only a worker whose process is alive is kept."""
        idle = self.idle_release_s
        keep = not final and (idle is None or idle > 0.)
        if keep and self._renderer is not None and self._prestart is None:
            r, self._renderer = self._renderer, None
            self._stop_idle_clock()
            if self._alive(r):
                self._retained = r
                if idle is not None:
                    self._arm_release(r, idle)
            else:                                             # it died under the operation: not kept
                self._renderer = r
                self._drop_dead()
            return
        self._stop_idle_clock()
        retained, self._retained = self._retained, None
        rs = [r for r in (self._renderer, (self._prestart or {}).get("renderer"), retained) if r is not None]
        holder = self._prestart
        self._renderer, self._prestart = None, None
        if not rs and holder is None:
            return
        self._note_release("final" if final else ("not_kept" if keep is False else "operation_failed"))

        def reap():
            if holder is not None:
                holder["thread"].join()
                if holder["renderer"] is not None and holder["renderer"] not in rs:
                    rs.append(holder["renderer"])
            for r in rs:
                r.close()
        t = threading.Thread(target=reap, name="mix-render-release", daemon=False)
        t.start()
        self._reapers = [x for x in self._reapers if x.is_alive()] + [t]

    def _arm_release(self, r, idle):
        """Start the idle clock of the kept worker r. A token makes an older clock harmless: one that has already
        fired and is waiting for the lock finds another token and does nothing."""
        self._retain_token = token = object()
        timer = threading.Timer(idle, self._release_retained, args=(r, token))
        timer.daemon = True
        timer.start()
        self._retain_timer = timer

    def _stop_idle_clock(self):
        self._retain_token = object()
        timer, self._retain_timer = self._retain_timer, None
        if timer is not None:
            timer.cancel()

    def _hold_retained(self):
        """A command begins: stop the kept worker's idle clock (the command's observation will want the worker)."""
        if self._retained is not None:
            self._stop_idle_clock()

    def _resume_idle_clock(self):
        """The command's physics has ended: the idle clock starts again (its observation normally follows at once)."""
        if self._retained is not None and self.idle_release_s is not None:
            self._arm_release(self._retained, self.idle_release_s or 1.0)

    def _release_retained(self, r, token=None):
        with self.env.lock:
            if self._retained is not r or (token is not None and token is not self._retain_token):
                return
            self._retained, self._retain_timer = None, None
            self._note_release("idle")
        r.close()

    def worker_status(self):
        """What the session's render worker is and has been through, for the owner (state field render_worker; none
        of it goes to the policy). pid is the worker process this session started and holds, a child of the session's
        own process; starts counts the workers started, lost those dropped because their process had exited or they
        failed a frame, frame_retries the frames asked a second time of a new worker."""
        r = self._renderer if self._renderer is not None else self._retained
        return dict(held=r is not None, alive=self._alive(r), pid=getattr(getattr(r, "process", None), "pid", None),
                    kept=self._retained is not None, starting=self._prestart is not None,
                    idle_release_s=self.idle_release_s, idle_clock_running=self._retain_timer is not None,
                    worker_start_s=round(float(self._worker_start_s), 3),
                    **{k: (list(v) if isinstance(v, list) else v) for k, v in self.worker_events.items()})

    # ------------------------------------------------------------------ observation
    def _frame(self, names, body_q, tick, meshes):
        """One frame of `names` from the session's worker. A worker that fails the frame (its process ended under the
        request, its socket is broken, it answers with an error or not at all) is ended and dropped, and the frame is
        asked once more of a new worker. If that fails as well, or no worker can be started, the session is left
        without a worker, BackendProtocolError is raised (the gateway then answers a command with `committed`,
        observation null), and the next image request starts a worker again.

        A ValueError is the request's fault, not the worker's: Renderer.observe raises it on this side before
        anything is sent (the liquid's meshes too large for one message, a non-finite pose). The worker is kept and
        the error raised as it is. Twice running on the same worker it is taken for the worker's fault after all (a
        reply that cannot be read leaves its stream unusable), and the worker is replaced."""
        first = None
        while True:
            try:
                r = self._operation_renderer()
            except Exception as exc:
                self._renderer = None
                self._note_fault(f"start: {type(exc).__name__}: {exc}")
                after = "" if first is None else f" after a failed frame ({type(first).__name__}: {first})"
                raise BackendProtocolError(
                    f"The render worker could not be started{after} ({type(exc).__name__}: {exc}); no image of tick "
                    f"{tick}. The world is unchanged by this; the next image request starts a worker again") from exc
            try:
                t0 = time.perf_counter()
                result = r.observe(names, body_q, tick, meshes=meshes)
                self.last_timing["frame_s"] = time.perf_counter() - t0
            except Exception as exc:
                if isinstance(exc, ValueError) and first is None and self._alive(self._renderer) \
                        and self._suspect is not self._renderer:
                    self._suspect = self._renderer
                    self._note_fault(f"request: {type(exc).__name__}: {exc}")
                    raise
                dead, self._renderer = self._renderer, None
                self._note_fault(f"frame: {type(exc).__name__}: {exc}")
                self._note_release("frame_error")
                self.worker_events["lost"] += 1
                if dead is not None:
                    self._end_worker(dead, "frame_error", wait=True)
                if first is not None:
                    raise BackendProtocolError(
                        f"The render worker failed a frame twice ({type(first).__name__}: {first}; on a new worker "
                        f"{type(exc).__name__}: {exc}); no image of tick {tick}. The world is unchanged by this; the "
                        "next image request starts a worker again") from exc
                first = exc
                self.worker_events["frame_retries"] += 1
                self.last_timing["frame_retry"] = True
                continue
            self.worker_events["frames"] += 1
            self._suspect = None
            return result

    def _render(self, names):
        from syringe_handling.render import liquid_meshes
        with self.env.lock:
            w = self.env.world
            try:
                t0 = time.perf_counter()
                meshes = liquid_meshes(w) if self.options["engine"]=="isaac" else {}
                self.last_timing["surface_s"] = time.perf_counter() - t0
                result = self._frame(tuple(names), w.state.body_q.numpy(), w.tick, meshes)
                started = max(float(self.last_timing.get(k) or 0.0) for k in ("cold_start_s", "prestart_warm_s"))
                if started > 0.0:
                    # a worker was started for this image: its start and its first frame (which loads shaders and
                    # textures) are what the command after an idle release pays. The session's latest start is the
                    # estimate of the next one, never below the constant
                    self._worker_start_s = max(WORKER_START_S, started + self.last_timing["frame_s"])
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
            t_render = time.perf_counter()
            first = self._cycle is None and not self._costs
            try:
                frames = self._render(self.cameras) if self.cameras else {}
            except BaseException:
                self._note_cost(time.perf_counter() - t_render, observed=False)
                raise
            self._note_cost(time.perf_counter() - t_render)
            if first and self.cameras:
                # The first observation of a scene (session start, a reset) follows no command, so _note_cost keeps
                # nothing of it. What it took is nevertheless the only measurement of this machine's present load the
                # session has before its first command, and the first floor is set from it and not from constants
                # measured on an idle graphics card: with the card shared with four other render jobs a command's
                # observation cost 8 s against the constants' 2.5 s, and the session's first command at its first
                # floor of 23 ms ran at 318 to 452 times real time. A session whose first image had to start the
                # render worker starts with a longer floor than it needs; the entry is displaced by the costs of its
                # first three commands. (A second frame of the same view, timed, would not help: an unchanged scene
                # renders in 0.3 s and says nothing.)
                self._costs = [(time.perf_counter() - t_render, PHYSICS_RATIO_MAX, FIXED_S)]
                obs_max, ratio_max, fixed_max = self._cost_now()
                self._floor_measured = floor_from(len(self.cameras), obs_max, ratio_max, self.min_command_published, fixed_max)
            proprio = {"joints": state["joints"], "joint_torques_nm": state.get("joint_torques_nm", {}),
                       "robot_status": self._robot_status()}
            # what the command this observation follows cost against the 200x limit (the interface's own clock; no
            # simulator state): the one place a command's reply can tell the policy that it ran over
            proprio["command_timing"] = self._command_timing()
            return Observation(sequence, state["time_s"], self.config.robot, proprio,
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

    def _has_thermometer(self):
        return False

    def _instruments_note(self):
        return ("none: no reading of the liquid, of a pressure or of a plunger is in the observation. The simulated time is "
                "(time_s of every observation), as the wall clock in the room shows it")

    def _note_cost(self, render_s, observed=True):
        """A command (or a batch) has ended with its observation: split its wall time, counted from the request's
        decoding (_precheck; from the command's start when it did not come through the gateway), into the physics
        (simulated time x the physics ratio), the observation (`render_s`) and the rest, the fixed cost per command.
        Keep the last three commands' costs and set the floor from the dearest of each part. A command whose image
        started a render worker is not kept (the start is paid once; it sets _worker_start_s in _render), and neither
        is an observation that followed no command. The command is also judged against the limit here (what the
        policy is told: _command_timing, and the session's count of commands over it). With observed=False the
        command ran and its observation failed: it is timed and judged all the same (the next observation says what
        it cost) and is not kept for the floor."""
        cycle, self._cycle = self._cycle, None
        t = self.last_timing
        if cycle is None or not cycle["n"] or cycle["sim_s"] <= 0.0:
            return
        started = any(k in t for k in ("cold_start_s", "prestart_wait_s", "prestart_warm_s"))
        total = time.perf_counter() - cycle["t0"]
        obs = (float(t.get("surface_s", 0.0)) + float(t.get("frame_s", 0.0))) if self.cameras else 0.0
        ratio = self._physics_ratio if self._physics_ratio is not None else cycle["physics_s"]/cycle["sim_s"]
        fixed = max(0.0, total - render_s - ratio*cycle["sim_s"])/cycle["n"]
        # over the limit: judged with UNSEEN_S allowed for what the backend cannot time (the request before it is
        # decoded, the reply, the image downloads), so that a command the client times at just over 200x is not called
        # inside it here
        per_sim = total/cycle["sim_s"]
        over = total + UNSEEN_S > BUDGET*cycle["sim_s"]
        self._timed += 1
        self._over += int(over)
        self.cost = dict(wall_s=round(total, 4), sim_s=round(cycle["sim_s"], 6), commands=cycle["n"],
                         physics_s=round(cycle["physics_s"], 4), physics_ratio=round(ratio, 2), render_s=round(render_s, 4),
                         observation_s=round(obs, 4), fixed_s=round(fixed, 4), worker_started=float(started),
                         floor_before_s=cycle["floor"], wall_per_sim=round(per_sim, 2), over_budget=float(over))
        self._timing = dict(wall_s=round(total, 4), sim_s=round(cycle["sim_s"], 6), wall_per_sim=round(per_sim, 2),
                            physics_wall_per_sim=round(cycle["physics_s"]/cycle["sim_s"], 2), over_budget=int(over))
        if started or not observed:
            return
        self._costs = (self._costs + [(obs, ratio, fixed)])[-3:]
        obs_max, ratio_max, fixed_max = self._cost_now()
        self._floor_measured = floor_from(len(self.cameras), obs_max, ratio_max, self.min_command_published, fixed_max)
        self.cost["floor_after_s"] = self._floor_measured

    def _robot_status(self):
        w = self.env.world
        return {"arms": {a: {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4),
                             **self._load_status(a)} for a in self.arms},
                "refused_targets": int(w.refused_targets)}

    # ------------------------------------------------------------------ the configured load (raw action set_load)
    @staticmethod
    def _load_limit_kg():
        from syringe_handling import params as P
        return float(P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG)

    @staticmethod
    def _load_com_max_m():
        from syringe_handling import params as P
        return float(P.LOAD_COM_MAX_M)

    def _load_status(self, arm):
        """What the arm's controller has been told its hand carries (World.set_load): mass and centre of mass in the
        TCP frame; nothing configured reads 0."""
        cfg = getattr(self.env.world, "load_cfg", {}).get(arm) or {}
        com = cfg.get("com_tcp")
        return {"load_kg": float(cfg.get("mass", 0.0)),
                "load_com_m": [float(x) for x in com] if com is not None else [0.0, 0.0, 0.0]}

    def _load_request(self, request):
        """(arm, mass_kg, com_m, duration_s or None) of a raw set_load request, checked against the limits World.set_load
        enforces, before anything changes. ValueError / UnsupportedActionError when it is refused."""
        if request.get("op") != "set_load":
            raise ValueError(f"The raw operation of syringe-handling is set_load; got op {str(request.get('op'))[:40]!r}")
        unknown = set(request) - {"op", "arm", "mass_kg", "com_m", "duration_s"}
        if unknown or not {"arm", "mass_kg"} <= set(request):
            raise ValueError("set_load takes arm, mass_kg and optionally com_m and duration_s"
                             + (f"; got {sorted(map(str, unknown))}" if unknown else ""))
        if not isinstance(request["arm"], str):
            raise ValueError("set_load arm must be an arm's name")
        arm = self._arm(request["arm"])
        mass, limit = request["mass_kg"], self._load_limit_kg()
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) or not 0. <= mass <= limit:
            raise ValueError(f"Load mass must be in [0, {limit:g}] kg (the arm's payload less its hand)")
        com = request.get("com_m", [0., 0., 0.])
        if not isinstance(com, (list, tuple)) or len(com) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in com) \
                or math.sqrt(sum(float(x)**2 for x in com)) > self._load_com_max_m():
            raise ValueError(f"Load centre of mass must be three finite metres within {self._load_com_max_m():g} m of the TCP")
        duration = request.get("duration_s")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                     or not math.isfinite(duration)):
            raise ValueError("set_load duration_s must be a number of seconds")
        return arm, float(mass), tuple(float(x) for x in com), duration

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

    def _keyboard(self, action, solve=True):
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
        if (move.any() or turn.any()) and solve:
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
        """The owner's command: plan and validate the whole command first (no side effect; a refused command raises
        ActionRejectedError and leaves the session usable); then apply grip modes and run the physics. Not floored."""
        self._cycle = None                                    # only policy commands are timed for the floor
        return self._act(action, 0.0)

    def _act(self, action, floor_s):
        """With floor_s (the policy path) a command shorter than that is refused; the command is planned once (a
        Cartesian or keyboard command's inverse-kinematics solve runs once per command)."""
        with self.env.lock:
            try:
                targets, ticks, grip = self._check(action, floor_s)
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise self._rejected(exc) from exc
            return self._execute(targets, ticks, grip)

    @staticmethod
    def _rejected(exc, prefix=""):
        """ActionRejectedError for a refusal decided from the command and the robot's own limits and joints (duration,
        tick, floor, joint limits, finger width, grip force, chord, reach): that text is what a real controller tells
        its operator, so the gateway may show it to the policy (policy_reason; http_server.py _error)."""
        err = ActionRejectedError(prefix + str(exc))
        err.policy_reason = prefix + str(exc)
        return err

    def _check(self, action, floor_s, solve=True):
        """Plan and validate one command without side effects; ValueError / UnsupportedConfigurationError /
        UnsupportedActionError when it is refused. With solve=False the inverse-kinematics solve of a Cartesian or
        keyboard-move command is left out (it depends on where the arm is when the command's turn comes) and
        everything else is checked exactly as when the command runs."""
        if self.config.robot == "g1" and not isinstance(action, NamedJointTarget):
            raise UnsupportedActionError("G1 uses named_joint_target for all 43 revolute body/Dex3 joints; no parallel-gripper or Franka Cartesian/load emulation")
        targets, ticks, grip = self._plan(action, solve, floor_s)
        if ticks*self.config.rigid_dt < floor_s - 1e-12:
            n = len(self.cameras)
            raise ValueError(f"Policy commands must last >= {floor_s*1e3:.0f} ms of simulated time now ({ticks} ticks of "
                             f"{self.config.rigid_dt*1e3:g} ms requested): a shorter command exceeds the {BUDGET:g}x "
                             f"wall/sim budget at what a command"
                             + (f" and its {n} camera observation(s) cost" if n else " costs") + " in this session at "
                             "present" + ("" if not n or self._worker_held() else
                                          " (no render worker is held, after the idle time or because it was lost: "
                                          "this command starts one)")
                             + "; the floor in force is hardware.min_command_s of GET /v1/capabilities")
        self._validate_targets(targets)
        for arm, f in grip.items():
            if arm == "__load__":
                continue
            limit = float(self.env.world.joints[self.env.world.gripper_joints(arm)[0]]["effort_limit"])
            if not 0. <= f <= limit:
                raise ValueError(f"inward_effort_n must be in [0, {2*limit:g}] N (both fingers)")
        return targets, ticks, grip

    def _precheck(self, actions):
        """Every command of a policy request, checked before the first one runs (SyringePolicyAccess.decode_actions): the
        floor in force and everything else that does not need the arm's pose. Nothing is solved, written or stepped.
        The request's timing starts here (_note_cost), with the floor it was admitted under."""
        self._cycle = None
        t0 = time.perf_counter()
        floor = self.min_command_s
        many = len(actions) > 1
        with self.env.lock:
            for index, action in enumerate(actions):
                where = f"command {index + 1} of {len(actions)}: " if many else ""
                try:
                    self._check(action, floor, solve=False)
                except UnsupportedActionError as exc:
                    raise UnsupportedActionError(where + str(exc)) from exc
                except (ValueError, UnsupportedConfigurationError) as exc:
                    raise self._rejected(exc, where + ("nothing in this request was executed. " if many else "")) from exc
        self._cycle = dict(t0=t0, floor=floor, physics_s=0.0, sim_s=0.0, n=0, decoded=True)

    def _plan(self, action, solve=True, floor_s=0.0):
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
            elif solve:
                position = self.ik.forward(arm)[:3, 3] + np.asarray(action.delta_m)
                orientation = None
            targets = self.ik.solve(arm, position, orientation) if solve else {}
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        elif isinstance(action, Keyboard):
            targets, ticks, grip = self._keyboard(action, solve)
        elif isinstance(action, RawAction):
            # set_load: the controller is told the load, no joint target changes, and the arms hold for the floor in
            # force (a command of no simulated time would have no wall/sim ratio at all), or for duration_s
            arm, mass, com, duration = self._load_request(action.request)
            ticks = self._ticks(duration) if duration is not None else max(1, int(math.ceil(floor_s/self.config.rigid_dt - 1e-9)))
            targets = {}
            grip = {"__load__": (arm, mass, com)}
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, "
                                         "keyboard, raw (set_load)")
        return targets, ticks, grip

    def _execute(self, targets, ticks, grip):
        w = self.env.world
        load = grip.pop("__load__", None)
        if load is not None:
            try:                                              # checked by _load_request already; the world has the last word
                w.set_load(load[0], load[1], load[2])
            except ValueError as exc:
                raise self._rejected(exc) from exc
        for arm, f in grip.items():
            w.set_grip_force(arm, f)
        before = w.tick
        n_reflex, n_refused = len(w.reflexes), w.refused_targets
        self.last_timing = {}
        if self.cameras:
            self._hold_retained()
            self._start_renderer_early()
        try:
            t0 = time.perf_counter()
            state = self.env.act(targets, ticks=ticks)
            self.last_timing["physics_s"] = time.perf_counter() - t0
            if ticks >= 20:                                   # shorter commands time their own overhead, not the physics
                self._physics_ratio = self.last_timing["physics_s"]/(ticks*self.config.rigid_dt)
            if self._cycle is not None:                       # a policy command being timed for the floor (_note_cost)
                self._cycle["physics_s"] += self.last_timing["physics_s"]
                self._cycle["sim_s"] += (w.tick - before)*self.config.rigid_dt
                self._cycle["n"] += 1
            self._resume_idle_clock()
        except BaseException as exc:
            self._cycle = None
            self._resume_idle_clock()
            self._end_operation()
            if isinstance(exc, (RuntimeError, FloatingPointError, OSError)):
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            raise
        return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": w.tick - before,
                                   "reflexes": [dict(r) for r in w.reflexes[n_reflex:]],
                                   "refused_targets": w.refused_targets - n_refused,
                                   "arms_stopped_s": {a: round(u - w.sim_time, 3) for a, u in w._reflex_until.items()
                                                      if u > w.sim_time},
                                   **({"load_set": {"arm": load[0], "mass_kg": load[1], "com_m": list(load[2])}} if load else {})},
                           0., False)

    def policy_access(self):
        from syringe_handling.task import instruction
        from ..keyboard_control import SUPPORTED_KEYS
        proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        proprio.update({("robot_status", "arms", a, "stopped_s"): None for a in self.arms})
        proprio.update({("robot_status", "arms", a, "load_kg"): None for a in self.arms})
        proprio.update({("robot_status", "arms", a, "load_com_m"): 3 for a in self.arms})
        proprio[("robot_status", "refused_targets")] = None
        proprio.update({("joint_torques_nm", n): None for n in self.env.joints if "finger" not in n})
        proprio.update({("command_timing", k): None for k in self._NO_TIMING})
        notes = self._capabilities.notes
        return SyringePolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                               self.kinds, lambda: instruction(self.config),
                               {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                                "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                               "unit": "metre" if "finger" in n else "radian"}
                                           for n, s in self.env.joints.items()},
                                "gripper": notes["gripper"],
                                "keyboard": notes["keyboard"],
                                "protective_stop": "G1 hard joint/torque limits and 1.2x URDF joint-velocity fault; pelvis support is a physical fixture" if self.config.robot == "g1" else f"{self.arm_model} collision reflex "
                                                   "(libfranka default thresholds: external joint torque over 10 N m "
                                                   "nominal, 20 N m while moving) and velocity reflex stop an arm; it then "
                                                   "refuses joint targets for 1 s (finger targets pass)",
                                "batches": notes["batches"],
                                "load": notes["load"],
                                "instruments": notes["instruments"],
                                "min_command_s": self.min_command_s,
                                "budget": self._budget(),
                                "command_timing": notes["command_timing"]},
                               keyboard_keys=tuple(SUPPORTED_KEYS), precheck=self._precheck)

    def act_policy(self, action):
        """The policy's command. Every policy command, with or without cameras, lasts at least the floor in force
        (min_command_s) of simulated time: a shorter one could not meet the 200x wall/sim budget and is refused, and
        the refusal tells the policy the floor. A request that came through the gateway was admitted, whole, by
        _precheck under the floor of that moment; its commands run under that floor (a floor that moved between the
        commands of a batch must not refuse one of them after the others have run)."""
        cycle = self._cycle
        if cycle is None:                                     # not through the gateway's decoding (local use, tests)
            cycle = self._cycle = dict(t0=time.perf_counter(), floor=self.min_command_s, physics_s=0.0, sim_s=0.0, n=0,
                                       decoded=False)
        try:
            result = self._act(action, cycle["floor"])
        except BaseException:
            self._cycle = None
            raise
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        valid = tuple(c for c in POLICY_CAMERAS if c in self.camera_specs) + DIAGNOSTIC
        if name not in valid:
            raise ValueError(f"Render label must be one of {valid}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        """Privileged geometry: the tray and its channels, the basin, the sharps container's slot and inside, the pad and
        its ring, the vial, the lock, every syringe and needle with its sizes, the counter and floor planes, the robot's
        bodies."""
        from syringe_handling import params as P
        w = self.env.world
        sc, fl = w.scene, w.fluid
        q = w.state.body_q.numpy()
        pose = lambda b: [float(x) for x in q[int(b)]]
        syringes = {n: dict(kind=s.kind, label=s.s["label"], barrel_body=int(s.barrel), plunger_body=int(s.plunger), pose_xyzw=pose(s.barrel),
                            bore_m=s.s["bore"], outside_m=s.s["od"], barrel_length_m=s.s["barrel_len"], flange_m=list(s.s["flange"]),
                            thumb_rest_m=s.s["thumb_d"], scale_ml_per_m=1e-6/(3.141592653589793*s.s["bore"]**2/4)**-1 if False else
                            (3.141592653589793*s.s["bore"]**2/4)/1e-6, division_ml=s.s["div_ml"],
                            frame="barrel frame: +x from the nozzle toward the thumb rest, origin at the chamber's front wall")
                     for n, s in fl.syringes.items()}
        needles = {n: dict(kind=d.kind, label=d.n["label"], hub_body=int(d.hub), sheath_body=None if d.sheath is None else int(d.sheath),
                           pose_xyzw=pose(d.hub), length_m=d.n["length"], outside_m=d.n["od"],
                           frame="hub frame: +x from the point toward the hub's mouth, origin at the mouth") for n, d in fl.needles.items()}
        out = {"syringes": syringes, "needles": needles,
               "tray": dict(body_id=int(sc.bodies["tray"]), pose_xyzw=pose(sc.bodies["tray"]), size_m=list(P.TRAY["size"]),
                            rib_pitch_m=P.TRAY["rib_pitch"], rib_height_m=P.TRAY["rib_h"], ribs_run_along="x of the tray"),
               "basin": dict(body_id=int(sc.bodies["basin"]), pose_xyzw=pose(sc.bodies["basin"]), size_m=list(P.BASIN["size"])),
               "sharps": dict(slot_centre_m=list(map(float, sc.sharps_slot["centre"])), slot_size_m=list(sc.sharps_slot["size"]),
                              inside_lo_m=list(map(float, sc.sharps_cavity["lo"])), inside_hi_m=list(map(float, sc.sharps_cavity["hi"])),
                              flap_body=int(sc.bodies["sharps_flap"])),
               "planes": {"counter": dict(normal=[0., 0., 1.], height_m=P.COUNTER_Z, xy_lo=list(sc.counter_plane["lo"]),
                                          xy_hi=list(sc.counter_plane["hi"])),
                          "floor": dict(normal=[0., 0., 1.], height_m=0.0)},
               "robot_bodies": {w.model.body_label[b]: int(b) for b in sc.robot_bodies},
               "ids": "get_geometry(<syringe or needle name>) returns that device's sizes and pose",
               "source": "The simulator's own geometry (syringe_handling/params.py, devices.py, scene.py), not a camera estimate"}
        if "pad" in sc.bodies:
            out["pad"] = dict(body_id=int(sc.bodies["pad"]), pose_xyzw=pose(sc.bodies["pad"]), size_m=list(P.PAD["size"]),
                              thickness_m=float(sc.pad_thick), ring_diameter_m=P.PAD["ring_d"],
                              layers_m={k: P.LAYERS[k]["thick"] for k in ("skin", "fat", "muscle")},
                              frame="pad frame: +z out of the skin, the ring's centre at (0, 0, thickness)")
        for n, v in fl.vials.items():
            out[n] = dict(body_id=int(v.body), pose_xyzw=pose(v.body), outside_m=P.VIAL["od"], height_m=P.VIAL["height"],
                          target_diameter_m=P.VIAL["target_d"], adapter=bool(v.adapter))
        for n, lk in fl.locks.items():
            out[n] = dict(connector_body=int(lk.body), connector_pose_xyzw=pose(lk.body), clamp_body=int(lk.clamp),
                          clamp_pose_xyzw=pose(lk.clamp), clamp_travel_m=P.EXTENSION["clamp_travel"],
                          frame="connector frame: +x out of its mouth, origin on the mouth plane; the clamp closes along its +x")
        return out

    def get_geometry(self, geometry_id):
        gid = str(geometry_id)
        index = self.get_geometry_index()
        for group in ("syringes", "needles"):
            if gid in index[group]:
                return dict(id=gid, **index[group][gid])
        if gid in index and isinstance(index[gid], dict) and gid not in ("planes", "robot_bodies"):
            return dict(id=gid, **index[gid])
        raise KeyError(f"Unknown geometry id {gid!r}; ids are the syringe and needle names and the top-level names of /v1/geometry")

    def get_geometry_asset(self, asset_id):
        raise KeyError(f"Unknown asset {str(asset_id)!r}; this environment has no geometry assets (sizes are in /v1/geometry)")

    def variation_contract(self):
        from syringe_handling import robot_setup
        from syringe_handling.variants import contract
        c = self.config
        offered = robot_setup.tasks_for(c.robot)
        return dict(**contract(), current=dict(preset=c.preset, task=c.task, seed=c.seed, variant=c.variant or "plain", contract=c.contract),
                    robot=dict(alias=c.robot, tasks="every task" if offered is None else list(offered)),
                    reset="POST /v1/variations/start (or reset with options {preset, task, variant, contract} and a seed) "
                          "rebuilds the world for the selection in this session (a few seconds). A variant not "
                          "named stays while the task stays and is dropped when the task changes; variant 'plain' is "
                          "the task as listed. Contract also stays while the task stays, else defaults to v1. "
                          "A refused selection (unknown name, a variant of another task, a seed "
                          "outside 0..9999, a task this session's robot is not offered for) answers 400 and changes "
                          "nothing")

    def _variation_key(self):
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, robot=c.robot, variant=c.variant)

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
        different scene (other bodies, other liquid); restoring into it would produce a hybrid state."""
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            if snap.get("variation") != self._variation_key():
                raise ValueError(f"Checkpoint {label!r} was saved in variation {snap.get('variation')}; the session is now "
                                 f"{self._variation_key()}. Reset to that variation first, then restore")
            self.env.restore(snap)
            self.env.generation += 1
            self._cycle, self._timing = None, dict(self._NO_TIMING)
            return self.get_state(), {"checkpoint": label, "tick": self.env.world.tick, "generation": self.env.generation,
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
            self._stop_idle_clock()
            self._end_operation(final=True)
            for t in self._reapers:
                t.join()
            self._reapers = []
        if self.trace:
            self.trace.close()
            self.trace = None

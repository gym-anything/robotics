"""Concrete-finishing bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo CPU rigid world, the wet
materials as 2.5-D yield-stress regions coupled through per-sole support plates, Isaac RTX images).

The robot is the Mobile FR3 Duo of classroom-arrangement with the same controller profile, so the command planning
(named joint targets, base twist, gripper, Cartesian pose and delta, keyboard chords), the protective stops, the
checkpoints and the physics trace are that backend's, and the render-worker lifecycle (one Isaac worker per session,
kept while the session is live, released on pause, restarted on a fault) is sweeping's, inherited unchanged. This
module adds what is this environment's: its configuration and variations, its room export and paste meshes in every
frame, its task instructions, its geometry index, and one command of its own:

  wait  -- {"type": "raw", "request": {"op": "wait", "duration_s": S}}: the material clock advances S seconds (1 s
           steps; strength, bleeding, evaporation, skins, slow flow) while the robot holds still, then 0.2 s of full
           ticks re-seat every resting body (concrete_finishing/env.py FinishEnv.wait). Finishing is mostly waiting
           for the concrete; a robot that holds still for an hour of simulated time in 0.5 ms ticks would cost days of
           wall time. It is refused, before anything happens and with the reason, unless the tools, the base and the
           arms have been at rest for the last 0.5 s and no arm is in a protective stop.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from collections import deque
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .newton_classroom_arrangement import POLICY_CAMERAS, BASE_CAMERAS, SESSION_RETENTION, MAX_ACTION_S, _finite
from .newton_sweeping import (NewtonSweepingBackend, SweepPolicyAccess, check_renderer_options as _shared_renderer_options, KEYBOARD_MAX_S,
                              WORKER_WATCH_S)
from ..actions import RawAction
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..types import BackendStep, CameraSpec, Capabilities

ENV_ID = "concrete-finishing"
DIAGNOSTIC = ("room", "pad", "pad_close", "patch", "infill", "joint", "wallboard", "plan", "chase")
SESSION_MARKER = ".gar-concrete-finishing-session"
SELECTORS = ("preset", "task", "seed", "variant")
WAIT_RESEAT_S = 0.2
RAW_HELP = {
    "required": ["type", "request"],
    "request": {"op": "wait", "duration_s": "0 < S <= 3600: simulated seconds of the material clock"},
    "example": {"type": "raw", "request": {"op": "wait", "duration_s": 600}},
    "note": "The one raw operation: wait advances the material clock (strength, bleeding, evaporation, sealant skin, "
            "slow flow) while the robot holds still, then 0.2 s of full ticks re-seat resting bodies. Refused unless "
            "the tools, the base and the arms have been at rest for the last 0.5 s and tools are unloaded from "
            "material regions. Lift them clear first. See hardware.wait.",
}


def check_renderer_options(options=None, policy_samples=None):
    if options is not None and not isinstance(options, dict):
        raise UnsupportedConfigurationError("renderer_options must be an object")
    values = dict(options or {})
    engine = values.pop("engine", "software")
    if engine not in ("software", "isaac"):
        raise UnsupportedConfigurationError("renderer_options.engine must be software or isaac")
    return dict(_shared_renderer_options(values, policy_samples), engine=engine)


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"concrete-finishing-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def wait_seconds(request):
    """Seconds of a raw wait request, checked before anything happens (ValueError with the reason)."""
    from concrete_finishing import params as P
    if request.get("op") != "wait":
        raise ValueError(f"The raw operation of concrete-finishing is wait; got op {str(request.get('op'))[:40]!r}")
    unknown = set(request) - {"op", "duration_s"}
    if unknown or "duration_s" not in request:
        raise ValueError("wait takes duration_s" + (f"; got {sorted(map(str, unknown))}" if unknown else ""))
    s = request["duration_s"]
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 < s <= P.WAIT_MAX:
        raise ValueError(f"wait duration_s must be a number of seconds in (0, {P.WAIT_MAX:g}]")
    return float(s)


@dataclass(frozen=True)
class FinishPolicyAccess(SweepPolicyAccess):
    """SweepPolicyAccess that also admits the raw action, which the shared allowlist (policy_access.ROBOT_ACTIONS)
    does not hold: here raw carries one operation, wait, and nothing else (the backend refuses any other op). A raw
    item is decoded by the shared codec (exactly the fields type and request); every other item goes through the shared
    allowlist and guard rules unchanged (ingredient_mixing's MixPolicyAccess)."""

    def decode_actions(self, payload):
        from ..action_codec import actions_from_request
        raw = lambda item: isinstance(item, dict) and item.get("type") == "raw" and "raw" in self.action_types
        items = payload.get("actions") if isinstance(payload, dict) and "actions" in payload else None
        try:
            if items is None and raw(payload):
                actions = actions_from_request(dict(payload))
            elif isinstance(items, list) and items and set(payload) == {"actions"} and any(raw(i) for i in items):
                if any(not isinstance(i, dict) or "actions" in i for i in items):
                    raise ValueError("Invalid action")
                actions = tuple(actions_from_request(dict(i)) if raw(i) else SweepPolicyAccess.decode_actions(self, i)
                                for i in items)
            else:
                return super().decode_actions(payload)
        except ActionRejectedError:
            raise
        except (ValueError, TypeError) as exc:
            raise NewtonConcreteFinishingBackend._rejected(exc) from exc
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


class NewtonConcreteFinishingBackend(NewtonSweepingBackend):
    QUALIFIED_MODELS = ("mobile-fr3-duo@1", "mobile-fr3-single@1")

    def __init__(self, *, robot="mobile-fr3-duo", preset="b012", task=None, seed=0, variant="default",
                 reflex_profile="contact_x5", cameras=None, output_dir=None, renderer_options=None, recording=True,
                 reflex_stops_base=True, policy_samples=None, **unknown):
        from concrete_finishing.config import Config
        from concrete_finishing.layout import PRESETS
        from concrete_finishing.env import FinishEnv
        from classroom_arrangement.kinematics import ArmKinematics
        from classroom_arrangement import params as CP
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS[ENV_ID]
        # ---- everything below this line and above "build" is checked before a folder or a world exists
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown concrete-finishing options {sorted(unknown)}; options: robot, preset, task, seed, variant, "
                "reflex_profile, cameras, renderer_options, recording, reflex_stops_base, policy_samples")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if not isinstance(recording, bool):
            raise UnsupportedConfigurationError("recording must be true or false")
        self.options = check_renderer_options(renderer_options, policy_samples)
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"concrete-finishing does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; built: {', '.join(catalog['aliases'])}")
        try:
            self.config = Config(preset=preset, task=task or PRESETS.get(preset, ("",))[0], seed=seed, robot=robot,
                                 variant=variant, reflex_profile=reflex_profile, reflex_stops_base=reflex_stops_base,
                                 policy_samples=self.options["samples"])
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.arms = self._arms_of(robot)
        self.policy_cameras = tuple(c for c in POLICY_CAMERAS if c == "head_rgb" or c in BASE_CAMERAS or c.split("_")[0] in self.arms)
        self.cameras = self._cameras(cameras, self.policy_cameras, self.robot_name)
        width, height, samples = self.options["width"], self.options["height"], self.options["samples"]
        self.idle_release_s = self.options["idle_release_s"]
        self.min_action_s = .05 if self.options["engine"] == "software" else self._min_action_s(len(self.cameras), width, height, samples)
        if self.min_action_s > KEYBOARD_MAX_S:
            raise UnsupportedConfigurationError(
                f"{len(self.cameras)} camera(s) at {width}x{height} and {samples} samples per pixel would need every command "
                f"to last at least {self.min_action_s:g} s to stay inside 200x wall/sim; choose fewer cameras, fewer pixels "
                "or fewer samples")
        # ---- build
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("concrete-finishing session (auto-created; pruned by retention)\n")
        self.env = FinishEnv(self.config, output_dir=self.output)
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
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "base_twist", "keyboard",
                                "raw"})
        from ..keyboard_control import SUPPORTED_KEYS
        from .newton_classroom_arrangement import BASE_KEYS
        self.keyboard_keys = tuple(sorted(set(SUPPORTED_KEYS) | set(BASE_KEYS)))
        self.capabilities = Capabilities(
            ENV_ID, "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())
        self._watch_stop = threading.Event()
        self._watcher = None
        if self.cameras:
            self._watcher = threading.Thread(target=self._watch_worker, name="concrete-finishing-render-watch", daemon=True)
            self._watcher.start()

    @staticmethod
    def _arms_of(robot):
        from robots.catalog import for_environment, model
        (_, instance), = for_environment(ENV_ID, robot).to_dict()["instances"].items()
        groups = model(instance["model"]).to_dict()["groups"]
        return tuple(g[:-4] for g in groups if g.endswith("_arm"))

    def _base_notes(self):
        from classroom_arrangement import params as CP
        c = self.config
        lim = CP.BASE_LIMITS
        o = self.options
        return {
            "min_command_s": f"every command lasts at least {self.min_action_s:g} s ({len(self.cameras)} camera(s) at "
                             f"{o['width']}x{o['height']}, {o['samples']} samples per pixel: 0.05 s per 640x360 frame of "
                             "render load at up to 16 samples, more in proportion to the pixels and to the samples "
                             "above 16). This is a nominal render allowance, not a measured 200x guarantee; shared load "
                             "and worker startup can exceed that budget. A shorter command is refused "
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
                        "target, a held base or a refused wait at a later command can stop such a request part-way. "
                        "The commands before it have then run, the refused command and the ones after it have not, "
                        "and the session stays ready with no recovery needed: the response has committed true, "
                        "partial true, completed_actions, rejected_action_index, unexecuted_action_indices and the "
                        "observation after the completed commands.",
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

    def _notes(self):
        notes = self._base_notes()
        notes.pop("debris", None)
        c = self.config
        site = c.site()
        from concrete_finishing.layout import TASK_SETUP
        words = dict(mag_float="the magnesium float", wood_float="the wood float", trowel="the steel trowel",
                     darby="the darby", edger="the edger", margin_trowel="the margin trowel", jointer="the jointer",
                     sealant_tool="the sealant tool", joint_knife="the joint knife", putty_knife="the putty knife",
                     penetrometer="the pocket penetrometer")
        names = [words.get(n, n.replace("_", " ")) for n, *_ in TASK_SETUP[c.task]["tools"]]
        listing = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        notes.update({
            "cameras": f"Ideal pinhole RGB, {self.options['engine']} renderer. Software uses the exported USD room, catalog visual meshes, diffuse textures and exact paste surfaces with local lighting; Isaac uses the authored RTX materials. No depth/noise model.",
            "renderer_options": "engine: software (default, CPU Mesa EGL) or isaac (RTX, retained diagnostic path). Width, height, samples and idle_release_s keep shared validation; samples only affects Isaac.",
            "task": f"{c.preset}/{c.task} seed {c.seed}, variant {c.variant} (concrete and air {site['temp_c']:g} C, air "
                    f"{site['air']}, bleed {site['bleed']:g} kg/m2/h)",
            "clock": "One tick = 0.5 ms (rigid step; the paste under every sole is cut, filled and worked after it, and the "
                     "material clock steps every 0.05 s); joint-target actions give ticks, others duration_s. The wait "
                     "command advances the material clock alone (raw op wait, up to 3600 s)",
            "wait": "raw {'op': 'wait', 'duration_s': S}, 0 < S <= 3600: strength, bleeding, evaporation, sealant skin and "
                    "slow flow advance S seconds in 1 s steps while the robot holds still; then 0.2 s of full ticks. "
                    "Refused unless the tools, the base and the arms have been at rest for the last 0.5 s, no joint target "
                    "is still being reached, the base has no twist, no arm is in a protective stop, and no tool is loaded "
                    "against a material region (0.05 N contact tolerance). Lift tools clear before accelerated wait. A wait of an hour "
                    "takes up to a few minutes of wall time on the pad (a 1 s step of its 323,000 cells)",
            "tools": f"{listing[0].upper() + listing[1:]} {'lies' if len(names) == 1 else 'lie'} on the locked utility "
                     "cart beside the station. Nothing is attached to the "
                     "robot: a tool is held by fingertip friction and falls if the grip is too weak. The paste carries a "
                     "tool by its bearing capacity (a support under the sole), drags it by its interface shear and the roll "
                     "it pushes, and holds it back when lifted off wet (tack).",
            "paste": "Concrete (pad, tub), repair mortar (floor patch), masonry mortar (block joints, mortar board), "
                     "sealant (control joint), compound (wallboard patch, hawk) as height fields on 1-4 mm cells with "
                     "their age, bleed water, surface paste layer, densification and texture; what a sole does to them is "
                     "physical (cut, fill, roll, spill, heave) and they change with time; nothing is moved by the "
                     "evaluator or the scene.",
            "qualification": "Assisted fixtures are not policy actions"})
        return notes

    def _worker_note(self):
        # The inherited sweeping timings describe another scene and load. Keep
        # this environment's lifecycle explanation without importing those claims.
        if not self.cameras:
            return ("No policy cameras are selected. An owner's diagnostic render starts the selected renderer; "
                    "its lifetime follows renderer_options.idle_release_s.")
        return (f"One {self.options['engine']} render worker supplies this session's cameras. Launch, reset or owner resume may wait "
                "for startup and the first frame. Camera commands normally reuse the live worker. If it dies, "
                "render recovery does not repeat physics; a failed observation can follow a committed action. "
                "Startup and render latency depend on scene and shared host load. The retained Isaac checks exceed 200x.")

    # ------------------------------------------------------------------ lifecycle
    def _variation(self):
        c = self.config
        return dict(preset=c.preset, task=c.task, seed=c.seed, variant=c.variant)

    def _resolve(self, options):
        from concrete_finishing.variants import resolve
        cur = self._variation()
        seed = options.get("seed", cur["seed"])
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        return resolve(options.get("preset", cur["preset"]), options.get("task", cur["task"]), seed,
                       options.get("variant", cur["variant"]))

    def reset(self, *, seed=None, options=None):
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
        from concrete_finishing.env import FinishEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            with self._render_lock:
                self._drop("rebuild")
                self._restart_due = False
                self._start_reason = "rebuild"
            self.config = replace(self.config, preset=v["preset"], task=v["task"], seed=int(v["seed"]), variant=v["variant"])
            old = self.env
            self.env = FinishEnv(self.config, output_dir=self.output)
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
        from concrete_finishing.variants import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        from concrete_finishing.task import instruction
        if set(request) - set(SELECTORS):
            raise ValueError(f"Selectors are {', '.join(SELECTORS)}")
        v = self._resolve(request)
        return dict(variation=v, task=instruction(v),
                    reset_request=dict(seed=v["seed"], options={k: v[k] for k in SELECTORS if k != "seed"}))

    # ------------------------------------------------------------------ rendering (sweeping's lifecycle, this room's frames)
    def _scene(self):
        from concrete_finishing.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env.world, self.output/"render")
        return self._scene_path

    def _start(self, reason):
        from concrete_finishing.render import Renderer
        t0 = time.perf_counter()
        o = self.options
        self._renderer = Renderer(self.output/"render", self._scene(), width=o["width"], height=o["height"], samples=o["samples"], engine=o["engine"])
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

    def _render(self, names):
        from concrete_finishing.render import frame
        with self._render_lock:
            fr, tick = frame(self.env.world), self.env.world.tick
            try:
                r = self._worker()
            except Exception as exc:
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
                    self._restart_due = True
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

    def session_paused(self, paused):
        from concrete_finishing.render import frame
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
                r.observe(tuple(self.cameras), frame(self.env.world), self.env.world.tick)
            except Exception:
                self._drop("frame_error")
                self._restart_due = True
                raise
            self.worker_events["frames"] += 1
            self.worker_events["last_resume_restart_s"] = round(time.perf_counter() - t0, 3)
            return True

    def _check_worker(self):
        from concrete_finishing.render import frame
        dead = self._renderer is not None and not self._alive(self._renderer)
        if self._closed or self._paused or not self.cameras or not (dead or self._restart_due):
            return False
        if not self._render_lock.acquire(timeout=0.05):
            return False
        try:
            r = self._renderer
            dead = r is not None and not self._alive(r)
            if self._closed or self._paused or not (dead or (self._restart_due and r is None)):
                return False
            if self._scene_path is None or not self._scene_path.is_file():
                return False
            if dead:
                code = getattr(getattr(r, "process", None), "returncode", None)
                self.worker_events["last_fault"] = f"the render worker's process had exited (code {code})"
                self._drop("worker_exited")
            self._restart_due = False
            r = self._start("fault")
            env = self.env
            if env.lock.acquire(timeout=0.05):
                try:
                    fr, tick = (frame(env.world), env.world.tick) if self.env is env else (None, None)
                finally:
                    env.lock.release()
                if fr is not None:
                    t0 = time.perf_counter()
                    r.observe(tuple(self.cameras), fr, tick)
                    self._fresh = False
                    self.worker_events["frames"] += 1
                    self.worker_events["last_first_frame_s"] = round(time.perf_counter() - t0, 3)
            return True
        finally:
            self._render_lock.release()

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        allowed = self.policy_cameras + DIAGNOSTIC
        if name not in allowed:
            raise ValueError(f"Render label must be one of {allowed}")
        with self.env.lock:
            return self._render((name,))[name]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        """The task's evaluator (concrete_finishing/task.py): `success`, `score` and every criterion with its value,
        threshold and basis. (The classroom backend's evaluate reads a `succeeded` key this evaluator does not
        write.)"""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            ev = self.env.evaluator.evaluate()
            return dict(ev, success=bool(ev["success"]), score=float(ev["score"]))

    # ------------------------------------------------------------------ commands
    def act(self, action):
        if isinstance(action, RawAction):
            return self._wait(action)
        return super().act(action)

    def _wait(self, action):
        w = self.env.world
        with self.env.lock:
            try:
                seconds = wait_seconds(action.request)
                reason = self.env.wait_refusal()
                if reason:
                    raise ValueError(f"wait refused, nothing executed: {reason}")
            except ValueError as exc:
                raise self._rejected(exc) from exc
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            before, n_reflex = w.tick, len(w.reflexes)
            clock0 = self.env.material_clock()
            t0 = time.perf_counter()
            try:
                state = self.env.wait(seconds)
            except (RuntimeError, FloatingPointError, OSError) as exc:
                raise BackendProtocolError(f"Stopped at tick {w.tick}: {exc}") from exc
            self.last_timing["physics_s"] = time.perf_counter() - t0
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": int(round(WAIT_RESEAT_S/0.0005)), "executed_ticks": w.tick - before,
                                       "waited_s": round(self.env.material_clock() - clock0 - (w.tick - before)*0.0005, 3),
                                       "reflexes": [dict(r) for r in w.reflexes[n_reflex:]]}, ev["score"], False)

    def _precheck(self, actions):
        """Every command of a policy request of several commands, checked before the first one runs. A wait is
        checked for its fields only (whether the robot is at rest is decided when its turn comes)."""
        n = len(actions)
        for index, action in enumerate(actions):
            if isinstance(action, RawAction):
                try:
                    wait_seconds(action.request)
                except ValueError as exc:
                    err = self._rejected(exc, f"command {index + 1} of {n}: ")
                    err.policy_reason = err.args[0] + " Nothing in this request was executed."
                    raise err from exc
        rest = tuple(a for a in actions if not isinstance(a, RawAction))
        if len(rest) > 1 or (rest and len(actions) > 1):
            super()._precheck(rest)

    # ------------------------------------------------------------------ policy access
    def policy_access(self):
        from concrete_finishing.task import instruction
        return FinishPolicyAccess(self.robot_name, tuple(self.camera_specs[c] for c in self.cameras), self._proprio(), self.kinds,
                                  lambda: instruction(self.config), self._policy_info(self.capabilities.notes),
                                  precheck=self._precheck)

    def _policy_info(self, notes):
        info = super()._policy_info(notes)
        info["wait"] = notes["wait"]
        info["paste"] = notes["paste"]
        info["clock"] = ("observation time_s is the simulated time of the rigid world; material_clock_s also counts the "
                         "waits (a crew's watch)")
        return info

    @classmethod
    def _qualified_setup(cls, robot, catalog):
        from concrete_finishing.robot_setup import validate
        from robots.catalog import resolve
        try:
            validate(robot)
            return resolve(robot).to_dict()
        except (TypeError, ValueError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------ geometry (privileged)
    def get_geometry_index(self):
        """Privileged geometry: every rigid shape and body (mass, centre of mass, inertia), the tools with their soles
        and current poses, and the wet regions (frame, grid, material, start state)."""
        w = self.env.world
        q = w.state.body_q.numpy()
        index = dict(self._catalog().index())
        index.pop("pen_frame", None)
        index.update(
            tools={n: dict(body_id=int(t["body"]), pose_xyzw=[float(v) for v in q[t["body"]]], kind=t["tool"].kind,
                           part=t["tool"].meta.get("part"), mass_kg=t["tool"].mass,
                           sole=dict(kind=t["tool"].sole.kind, half_extents_m=[t["tool"].sole.hx, t["tool"].sole.hy]),
                           shape_ids=[str(s) for s in t["shapes"]]) for n, t in w.scene.tools.items()},
            regions={n: dict(material=r.material.name, origin_m=r.o.tolist(), u_axis=r.U.tolist(), v_axis=r.V.tolist(),
                             normal=r.N.tolist(), cell_m=r.cell, shape=[r.nu, r.nv], summary=r.summary())
                     for n, r in w.regions.items()},
            region_ids="get_geometry('region:<name>') returns the region's fields (heights and surface states)",
            item_ids="get_geometry(<tool name>) returns its authored parts and current pose",
            source="Actual model geometry, masses, friction and material fields; not a camera-derived estimate")
        return index

    def get_geometry(self, geometry_id):
        gid = str(geometry_id)
        w = self.env.world
        if gid.startswith("region:"):
            name = gid.split(":", 1)[1]
            if name not in w.regions:
                raise KeyError(f"No region {name!r}; regions: {', '.join(w.regions)}")
            r = w.regions[name]
            return dict(id=gid, material=r.material.name, cell_m=r.cell, shape=[r.nu, r.nv], origin_m=r.o.tolist(),
                        u_axis=r.U.tolist(), v_axis=r.V.tolist(), normal=r.N.tolist(), clock_s=r.clock,
                        fields={f: np.round(getattr(r, f), 7).tolist() for f in ("b", "tb", "ts", "w", "E", "D", "S", "a")},
                        paste_mask=r.paste_mask.tolist())
        if gid in w.scene.tools:
            t = w.scene.tools[gid]
            q = w.state.body_q.numpy()[t["body"]]
            return dict(id=gid, parts=[dict(name=p.name, kind=p.kind, size=list(p.size), pos=list(p.pos), quat=list(p.quat),
                                            mass_kg=p.mass, material=p.material) for p in t["tool"].parts],
                        pose_xyzw=[float(v) for v in q])
        return super(NewtonSweepingBackend, self).get_geometry(gid)

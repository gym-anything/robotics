"""Filleting bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 rigid + filleting tissue solver).

The environment (filleting/env.py FilletingEnv) owns the world, its clock, the settled initial state, checkpoints and
the evaluator; this adapter only selects the deployment, maps GAR robot actions to joint targets, renders robot
cameras and exposes an audited restricted interface.

Render workers: an action's worker is started and warmed on a snapshot of the pre-action state while the action's
physics runs and consumed by that action's observation. After the observation the worker is kept for
renderer_options idle_release_s seconds (default IDLE_RELEASE_S) for the next operation, then released by a timer
(bounded idle retention, as barcode_scanning's adapter: no worker is kept indefinitely after an observation); 0
releases it after every observation, the strict per-operation lifecycle. A fresh worker and its warm-up frame cost
18-24 s, which alone put a 0.1 s action over the 200x wall/sim limit.
"""

from collections import deque
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta
from ..errors import (ActionRejectedError, BackendProtocolError, UnsupportedActionError,
                      UnsupportedConfigurationError)
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

# the runtime facts every client needs (owner capabilities and the restricted policy's hardware description), measured
WALL_TIME_NOTE = ("Slow: tens to a few hundred wall seconds per simulated second. On an unshared host with one camera: "
                  "holding still ~60x, cutting ~100-200x (a 0.5 s slice of the trout ~80-100 s, a 0.1 s stroke after it "
                  "9-17 s); every extra policy camera adds ~1.6 s per observation (four cameras ~6.5 s); beside other "
                  "GPU or CPU jobs two to four times longer. Startup 135-225 s. Use client timeouts of at least 600 s "
                  "(gar action/observation/state/verify --timeout 600; the default 120 s can expire on an action that "
                  "has already committed)")

POLICY_CAMERAS = ("overhead", "supervisor", "left_wrist_rgb", "right_wrist_rgb")
DIAGNOSTIC = ("hero", "hero_reverse", "bench", "workpiece", "knife_block")
MOUNTS = {"overhead": "external_overhead_mount", "supervisor": "external_supervisor_mount",
          "left_wrist_rgb": "link7_d405_envelope", "right_wrist_rgb": "link7_d405_envelope"}
RENDERER_OPTIONS = frozenset({"width", "height", "samples", "tissue_mode", "idle_release_s"})
IDLE_RELEASE_S = 10.0
# path-tracing samples per pixel of an observation (denoised); the films render 32. Four policy cameras took ~20 s per
# observation at 32 on the loaded GPU and ~4.5 s at 8; renderer_options samples=32
# restores the film quality
POLICY_SAMPLES = 8
TOP_LEVEL = ("preset", "task", "seed", "robot")

SESSION_MARKER = ".gar-filleting-session"
SESSION_RETENTION = 20


def _session_folder():
    """Auto-created session folder: $GAR_WORKSPACE/.gar/filleting-sessions/<uuid> (the workspace's managed,
    git-ignored session area; else the system temp directory), never inside the installed package. Keeps the
    SESSION_RETENTION newest auto-created sessions; only folders carrying SESSION_MARKER are ever pruned."""
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"filleting-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def _check_robot(robot):
    """The robot catalog decides; unsupported deployments fail with the catalog's reason before any physics import."""
    from robots.catalog import ENVIRONMENTS
    catalog = ENVIRONMENTS["filleting"]
    qualified = tuple(catalog["aliases"])
    if isinstance(robot, dict):
        raise UnsupportedConfigurationError(f"filleting supports catalog aliases {', '.join(qualified)}")
    if robot in catalog.get("unsupported", {}):
        raise UnsupportedConfigurationError(f"filleting does not support {robot!r}: {catalog['unsupported'][robot]}")
    if robot not in qualified:
        raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
    return robot


class NewtonFilletingBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset=None, task=None, seed=0, device=None, config=None,
                 cameras=("overhead",), output_dir=None, renderer_options=None):
        robot = _check_robot(robot)
        values = dict(config or {})
        if "phantom_rig" in values:
            raise UnsupportedConfigurationError("phantom_rig is an owner-side ASSISTED rig for demos, not a GAR option")
        clash = sorted(set(TOP_LEVEL) & set(values))
        if clash:
            raise UnsupportedConfigurationError(f"{clash} are top-level options, not config fields")
        if device is not None and "device" in values:
            raise UnsupportedConfigurationError("Device supplied at two levels; choose exactly one")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; {DIAGNOSTIC} are privileged diagnostics")
        self.options = dict(renderer_options or {})
        if set(self.options) - RENDERER_OPTIONS:
            raise UnsupportedConfigurationError(f"renderer_options accepts {sorted(RENDERER_OPTIONS)}")
        # the environment's own resolver: slugs or task codes, preset/task consistency, seed range
        from filleting.variants import resolve, PRESET_SLUG, TASK_SLUG
        from filleting.world import Config
        from filleting.env import FilletingEnv
        from filleting.task import Evaluator
        try:
            chosen = resolve(preset, task, seed)
        except KeyError as exc:                                   # the resolver's unknown-preset path
            raise ValueError(f"Unknown preset {preset!r}; presets: {', '.join(PRESET_SLUG.values())}") from exc
        from filleting.variants import check_robot
        try:
            check_robot(robot, chosen["task"])
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        values.update(preset=chosen["preset"], task=chosen["task"], seed=chosen["seed"], robot=robot)
        from filleting.variants import CONTACT_TASKS
        if chosen["task"] in CONTACT_TASKS:
            # the cutting tasks' recipe: libfranka's contact-task collision thresholds, as every cutting demonstration
            # and runtime measurement used (an explicit config value is kept)
            values.setdefault("collision_profile", "contact")
        if device is not None:
            values["device"] = device
        self.config = Config(**values)                                   # unknown fields fail (TypeError)
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("filleting session (auto-created; pruned by retention)\n")
        self.env = FilletingEnv(self.config, output_dir=self.output)
        w = self.env.world
        self.arms = tuple(w.rig.instances)
        missing = [c for c in self.cameras if c.endswith("_wrist_rgb") and c.split("_")[0] not in self.arms]
        if missing:
            raise UnsupportedConfigurationError(f"{missing}: this deployment has arms {', '.join(self.arms)} only")
        self.images = deque()
        self.last_timing = {}
        self._prestart = None
        self._retained = None
        self._retain_timer = None
        self._reapers = []
        self._checkpoints = {}
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        from filleting.render import CAMERAS, D405
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            fov = float(D405["hfov"] if "wrist" in name else CAMERAS[name]["hfov"])
            focal = .018/math.tan(math.radians(fov/2))
            self.camera_specs[name] = CameraSpec(name, MOUNTS[name], width, height, 1280, 720,
                                                 (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        # measured from the built deployment, not restated from a datasheet
        finger = w.gripper_joints(self.arms[0])[0]
        self.finger_effort_limit = float(w.joints[finger]["effort_limit"])
        finger_upper = float(w.joints[finger]["upper"])
        tissue_dt = float(w.config.tissue_dt)
        has_eval = hasattr(Evaluator, f"_eval_{self.config.task}")
        c = self.config
        self.capabilities = Capabilities(
            "filleting", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[n] for n in self.cameras), notes={
                "task": f"{PRESET_SLUG[c.preset]}/{TASK_SLUG[c.task]} ({c.task}) seed {c.seed}, workpiece "
                        f"{c.workpiece}",
                "clock": f"One tick = {c.rigid_dt*1e3:g} ms rigid MuJoCo step; the tissue FEM advances the same "
                         f"interval in explicit substeps of {tissue_dt*1e6:g} us (nominal "
                         f"{round(c.rigid_dt/tissue_dt)} per tick; the last tick's count is state tissue.substeps)",
                "control": "Physical position servos; joint targets pass a rate limiter at the joint speed and "
                           "acceleration limits (as libfranka's; fingers at the Franka Hand's 50 mm/s); FR3/Panda "
                           f"protective stops (velocity, collision torque: '{c.collision_profile}' profile, "
                           + ("libfranka's contact-task 100 N m thresholds, the cutting tasks' default" if
                              c.collision_profile == "contact" else "libfranka's default 10/20 N m thresholds")
                           + ") freeze the arm; IK computes targets only "
                           "(not collision-free planning); an unreachable IK target or out-of-limit joint target is "
                           "rejected before any tick",
                "guards": "No simulator motion guards: stop_on_contact, stop_on_robot_contact and stop_on_grip_loss "
                          "are rejected; stop_on_arm_arm_contact is not enforced, commands run their full duration",
                "arms": ",".join(self.arms),
                "gripper": f"Franka Hand: half_width_m position mode (0..{finger_upper*1e3:g} mm per finger) or "
                           f"inward_effort_n force-mode grasp (total N of both fingers, <= "
                           f"{2*self.finger_effort_limit:g} N = 2 x {self.finger_effort_limit:g} N finger limit)",
                "knife": ("The right Franka Hand carries authored knife-handle inserts and starts the episode "
                          f"holding the fillet knife (spawned in the insert grasp at {c.grip_force:g} N per finger, "
                          f"then settled hands-off for {self.env.settle_s:g} s: a disclosed authored initial "
                          "condition)") if self.env.world.knife_start == "hand" else (
                          "The fillet knife stands handle-up in the knife block's slot (right of the board); the right "
                          "Franka Hand carries authored knife-handle inserts and starts empty"
                          if self.env.world.knife_start == "block" else "Knife not in hand (config knife_in_hand=False)"),
                "physics": "Explicit FEM tissue with the knife-edge cutting law, crack insertion, cohesive "
                           "interfaces and bone mechanics; tissue reactions reach the rigid bodies one tick later "
                           "through a 5 ms filter (filleting/world.py)",
                "evaluator": ("Physical-state evaluator (filleting/task.py)" if has_eval else
                              "No evaluator implemented for this task yet: evaluation reports success=false, "
                              "score=0 with a note (filleting/task.py)"),
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state (render-only worker, "
                           "warmed during the action's physics and kept idle_release_s after an observation); no "
                           "depth or noise model",
                "wall_time": WALL_TIME_NOTE,
                "qualification": "Assisted fixtures are not policy actions"})

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.config.seed):
            raise ValueError("Select preset/task/seed when creating the session; reset restores that initial state")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            return self.env.evaluator.evaluate()

    # ------------------------------------------------------------------ render workers (one per operation)
    def _frame(self):
        """Render frame of the current world state (filleting/frames.py); never drawn with a badge here."""
        from filleting.frames import world_frame
        return world_frame(self.env.world, "UNASSISTED")

    def _scene(self, frame):
        """The exported static stage of this world (exported once from the first observed frame)."""
        from filleting.render import export_scene
        key = id(self.env.world)
        if getattr(self, "_scene_key", None) != key or not self._scene_path.is_file():
            self._scene_path = export_scene(self.output/"render", frame, self.config.preset)
            self._scene_key = key
        return self._scene_path

    def _new_renderer(self, scene):
        from filleting.render import Renderer
        kwargs = {k: self.options[k] for k in ("samples", "tissue_mode") if k in self.options}
        kwargs.setdefault("samples", POLICY_SAMPLES)
        return Renderer(self.output/"render", None, self.config.preset, width=self.options.get("width", 1280),
                        height=self.options.get("height", 720), scene=scene, **kwargs)

    def _start_renderer_early(self):
        """Start and warm this operation's worker on the pre-action frame while the physics runs (or take the
        retained worker of the previous observation: owned by this operation from now, so its idle timer cannot
        release it while the physics runs)."""
        if not self.cameras or self._prestart is not None:
            return
        r = self._take_retained()
        if r is not None:
            self._prestart = {"renderer": r, "error": None, "warm_s": 0., "thread": None, "retained": True}
            return
        frame = self._frame()                                  # in this thread: reads the world
        scene = self._scene(frame)
        holder = {"renderer": None, "error": None, "warm_s": None}

        def work():
            try:
                t0 = time.perf_counter()
                holder["renderer"] = self._new_renderer(scene)
                holder["renderer"].observe(frame, tuple(self.cameras))       # warm-up frame, discarded
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:                       # reported when consumed
                holder["error"] = exc
        holder["thread"] = threading.Thread(target=work, name="filleting-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _take_prestart(self):
        holder, self._prestart = self._prestart, None
        if holder is None:
            return None, None
        t0 = time.perf_counter()
        if holder.get("thread") is not None:
            holder["thread"].join()
        self.last_timing.update(prestart_wait_s=time.perf_counter() - t0, prestart_warm_s=holder["warm_s"],
                                retained_worker=bool(holder.get("retained")))
        if holder["error"] is not None or holder["renderer"] is None:
            if holder["renderer"] is not None:
                self._release(holder["renderer"])
            return None, holder["error"]
        return holder["renderer"], None

    def _release(self, renderer, wait=False):
        """Release a worker now; the reply does not wait for Kit's shutdown unless `wait`."""
        self._reapers = [t for t in self._reapers if t.is_alive()]
        if wait:
            renderer.close()
            return
        t = threading.Thread(target=renderer.close, name="filleting-render-release", daemon=False)
        t.start()
        self._reapers.append(t)

    def _take_retained(self):
        """The retained worker, if it was opened on the current scene (else released); cancels its idle timer."""
        r, self._retained = self._retained, None
        if self._retain_timer is not None:
            self._retain_timer.cancel()
            self._retain_timer = None
        if r is not None and getattr(self, "_retained_scene", None) != getattr(self, "_scene_path", None):
            self._release(r)
            return None
        return r

    def _end_operation(self, renderer):
        """After an observation: keep the worker for idle_release_s, then release it (0: release now)."""
        idle = float(self.options.get("idle_release_s", IDLE_RELEASE_S) or 0.)
        if idle <= 0.:
            self._release(renderer)
            return
        old = self._take_retained()
        if old is not None:
            self._release(old)
        self._retained, self._retained_scene = renderer, getattr(self, "_scene_path", None)
        timer = threading.Timer(idle, self._release_retained, args=(renderer,))
        timer.daemon = True
        timer.start()
        self._retain_timer = timer

    def _release_retained(self, renderer):
        with self.env.lock:
            if self._retained is not renderer:
                return                                    # taken by a new operation (or already released)
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

    def observation_from_state(self, state, *, sequence):
        with self.env.lock:
            if (state["tick"], state["generation"]) != (self.env.steps, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.config.robot, {"joints": state["joints"]},
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
        joints = self.env.world.gripper_joints(arm)
        spec = self.env.world.joints[joints[0]]
        lo, hi = float(spec["lower"]), float(spec["upper"])
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) \
                or not lo <= width <= hi:
            raise ValueError(f"Franka Hand half gap must be in [{lo:g}, {hi:g}] m")
        return {j: float(width) for j in joints}

    def _validate(self, targets):
        """world.set_targets' checks, before anything is written (a failure here executes nothing)."""
        joints = self.env.world.joints
        for name, value in targets.items():
            spec = joints.get(name)
            if spec is None:
                raise ValueError(f"Not a robot actuator: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Joint target {name}={value} outside [{spec['lower']:g}, {spec['upper']:g}]")

    def _prepare(self, action):
        """(joint targets, ticks, grip-force change or None); raises before any write or tick."""
        grip = None
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                # Franka Hand grasp semantics: total inward force split over the two fingers, fingers servo toward
                # closed; 0 N returns to pure position control.
                total = float(action.inward_effort_n)
                if total/2. > self.finger_effort_limit:
                    raise ValueError(f"inward_effort_n {total:g} N exceeds {2*self.finger_effort_limit:g} N "
                                     f"(2 x {self.finger_effort_limit:g} N finger limit)")
                width = 0. if total > 0 else (action.half_width_m if action.half_width_m is not None else
                                              float(self.env.world.joints[self.env.world.gripper_joints(arm)[0]]["upper"]))
                targets, grip = self._fingers(arm, width), (arm, total/2.)
            else:
                targets, grip = self._fingers(arm, action.half_width_m), (arm, 0.)
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            arm = self._arm(action.arm)
            s = action.safety
            if s.stop_on_grip_loss is not None or s.stop_on_contact or s.stop_on_robot_contact:
                raise UnsupportedActionError("Simulator motion guards are not implemented in filleting; "
                                             "no action executed")
            ticks = self._ticks(action.duration_s)
            kin = self.env.world.kin
            if isinstance(action, CartesianPose):
                position = action.position_m
                o = action.orientation_wxyz
                orientation = None if o is None else (*o[1:], o[0])
            else:
                position = kin.forward(arm)[:3, 3] + np.asarray(action.delta_m)
                orientation = None
            targets = kin.solve(arm, position, orientation)
            if action.gripper_half_target_m is not None:
                targets.update(self._fingers(arm, action.gripper_half_target_m))
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta")
        self._validate(targets)
        return targets, ticks, grip

    def act(self, action):
        with self.env.lock:
            w = self.env.world
            if w.fault:
                raise BackendProtocolError(f"Physics is faulted ({w.fault}); restore a checkpoint or reset")
            try:
                targets, ticks, grip = self._prepare(action)
            except (ValueError, TypeError) as exc:
                message = str(exc) if "no action executed" in str(exc) else f"{exc}; no action executed"
                raise ActionRejectedError(message) from exc
            before = self.env.steps
            self.last_timing = {}
            self._start_renderer_early()
            try:
                if grip is not None:
                    w.set_grip_force(*grip)
                t0 = time.perf_counter()
                state = self.env.act(targets, ticks=ticks)
                self.last_timing["physics_s"] = time.perf_counter() - t0
            except BaseException as exc:
                renderer, _ = self._take_prestart()          # the operation failed: release its worker
                if renderer is not None:
                    self._release(renderer)
                if isinstance(exc, (RuntimeError, FloatingPointError, OSError, ValueError)):
                    raise BackendProtocolError(f"Stopped at tick {self.env.steps}: {exc}") from exc
                raise
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.steps - before,
                                       "tissue_substeps_last_tick": state["tissue"]["substeps"],
                                       "knife": state["knife"], "events": state["events"],
                                       "termination": "time_limit" if state["terminated"] else None,
                                       "physics_s": self.last_timing.get("physics_s")},
                               ev["score"], bool(state["terminated"]))

    def policy_access(self):
        from filleting.task import INSTRUCTIONS, TIME_LIMIT_S
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras),
                            {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")},
                            self.kinds, lambda: INSTRUCTIONS[self.config.task],
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                             # the episode's simulated-time budget, which the evaluation's time check enforces (the
                             # instruction does not state it)
                             "time_limit_s": TIME_LIMIT_S.get(self.config.task),
                             # what a client using only the policy key must know to keep its connection alive (an
                             # action commits before its reply)
                             "wall_time": WALL_TIME_NOTE, "client_timeout_s": 600,
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()}})

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "hero" if label in ("interactive", "http") else label
        if name not in POLICY_CAMERAS + DIAGNOSTIC:
            raise ValueError(f"Render label must be one of {POLICY_CAMERAS + DIAGNOSTIC}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ checkpoints
    def get_geometry_index(self):
        """Owner geometry: the bench objects (layout dimensions, metres), the knife's bodies and the workpiece as it
        lies now (its tissue surface's bounds and size, its rigid bones' poses). Privileged; a restricted policy does
        not see it."""
        from filleting import layout as LY
        w = self.env.world
        ts = w.tissue
        preset = self.config.preset
        items = {"bench": dict(kind="box", size_m=[LY.TABLE["size"][0], LY.TABLE["size"][1], LY.TABLE["top_thickness"]],
                               top_z_m=LY.TABLE["top_z"]),
                 "board": dict(kind="box", size_m=list(LY.BOARD["size"]), center_xy_m=list(LY.BOARD["center"]),
                               top_z_m=LY.BOARD_TOP_Z),
                 "fillet_tray": dict(kind="tray", size_m=list(LY.FILLET_TRAY["size"]),
                                     center_xy_m=list(LY.fillet_tray(self.config.task)["center"])),
                 "bowl": dict(kind="bowl", diameter_m=LY.BOWL["diameter"], depth_m=LY.BOWL["depth"],
                              center_xy_m=list(LY.BOWL["center"])),
                 "knife_block": dict(kind="box", size_m=list(LY.KNIFE_BLOCK["size"]),
                                     center_xy_m=list(LY.KNIFE_BLOCK["center"]))}
        for name, spec in (("ice_pan", dict(LY.ICE_PAN, center=LY.ice_pan_center(self.config.task))),
                           ("scale", LY.SCALE)):
            if preset in spec.get("presets", ()):
                items[name] = dict(kind="box", size_m=list(spec["size"]), center_xy_m=list(spec["center"]))
        if self.config.task in LY.FRAME_PAN["tasks"]:
            items["frame_pan"] = dict(kind="tray", size_m=list(LY.FRAME_PAN["size"]),
                                      center_xy_m=list(LY.FRAME_PAN["center"]), rim_m=LY.FRAME_PAN["rim"],
                                      taper_m=LY.FRAME_PAN["taper"])
        st = ts.body_states()
        k0 = w.knife_body0
        names = {"knife_handle": k0, **{f"knife_blade_segment_{j}": k0 + 1 + j for j in range(w.scene.blade.segments)}}
        if w.assembly is not None:
            names.update(dict(w.assembly["body_names"]))
        bodies = {n: dict(position_m=[float(v) for v in st["p"][i]], quat_xyzw=[float(v) for v in st["q"][i]])
                  for n, i in names.items() if i < len(st["p"])}
        kd = ts.kind.numpy()[:ts.n]
        x = ts.x.numpy()[:ts.n][kd == 0].astype(float)
        workpiece = dict(name=self.config.workpiece, tissue_nodes=int(len(x)), tets=int(ts.nt),
                         bounds_m=[x.min(0).tolist(), x.max(0).tolist()] if len(x) else None,
                         crack_faces=len(w.cut.cut_faces))
        return {"items": items, "bodies": bodies, "workpiece": workpiece, "frame": "world, metres, z up",
                "source": "filleting/layout.py (bench objects), the tissue solver (knife, bones, workpiece)"}

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
            self.env.restore(snap)
            self.env.generation += 1
            # backend contract: (state, info); checkpoints are in-memory for the life of the session
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.steps,
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
        self.env.close()

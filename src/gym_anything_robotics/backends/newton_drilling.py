"""Drilling bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + CPU MuJoCo rigid world with the hole
module and contact-operated tools; Isaac RTX rendering of the recorded state).

Render worker: one Isaac render-only worker per session, started with the session's first image and kept for as long
as the session is live and not paused, whatever the time between commands, so no policy command waits for a worker to
start. The owner's pause is the idle signal: a paused session (session_paused, called by the session layer) keeps the
worker for renderer_options idle_release_s more seconds (IDLE_RELEASE_S by default; 0 gives it back at the pause, null
keeps it until stop) and then gives the GPU back with the physics state untouched; the owner's resume request restarts
a released worker and returns only after its first frame, so that wait (COLD_START_S, measured) is the owner's and not
a policy command's. The worker is also released on stop (close) and on a reset to another variation (the scene changes;
the new one is started inside the reset). Faults: a live session looks at its worker every WORKER_WATCH_S (_check_worker)
and, when the worker's process has exited or a frame error has cost the session its worker, starts a new one at once
and has it draw its first frame as soon as no command holds the world; only a camera command that arrives while that
worker is still starting waits for it. A worker watches its owner process (gym_anything_robotics.render_lifecycle).

Policy commands shorter than min_command_s are refused: the floor comes from the measured cost model in
drilling/runtime_cost.py (fixed per-command overhead, per-frame cost by each camera's pixel count, physics ratio). The
floor does not depend on idle_release_s, because no setting of it can put a worker start inside a policy command.

Refusals leave the session usable. A policy request - one command or a batch {"actions": [...]} - is checked whole
when it is decoded, before its first command runs, for everything that does not depend on where the arms will be
(DrillingPolicyAccess.decode_actions -> _precheck: each command's duration against the floor and the tick, arm
names, joint names and limits, finger widths, grip forces, keyboard chords). What cannot be known beforehand is the
inverse-kinematics solve of a Cartesian or keyboard-move command that is not the first of its batch: it is solved
from the pose the earlier commands leave, and if it is unreachable then, those commands have run and the gateway asks
for the owner's recovery (its rule for a partly executed batch). An owner reset whose selectors, seed or robot/task
pair are refused raises ActionRejectedError before anything is touched.
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

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, Keyboard, CartesianWrench
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation


@dataclass(frozen=True)
class DrillingPolicyAccess(PolicyAccess):
    """PolicyAccess whose decode step also runs the backend's state-independent checks on every command of the
    request. The gateway decodes a request before it runs any of it (http_server.py action()), so a batch with a
    refused command anywhere in it is refused whole and nothing has moved."""
    precheck: Callable | None = None

    def decode_actions(self, payload):
        actions = super().decode_actions(payload)
        if self.precheck is not None:
            self.precheck(actions if isinstance(actions, tuple) else (actions,))
        return actions

POLICY_CAMERAS = ("station_rgb", "overview_rgb", "left_wrist_rgb", "right_wrist_rgb")
DIAGNOSTIC = ("hero", "wall_macro", "press_side", "bench_macro")
SESSION_MARKER = ".gar-drilling-session"
SESSION_RETENTION = 20
from drilling import runtime_cost  # noqa: E402
IDLE_RELEASE_S = runtime_cost.IDLE_RELEASE_S   # authored: how long a paused session keeps its render worker
WORKER_WATCH_S = runtime_cost.WORKER_WATCH_S   # authored: how often a live session looks at its worker's process
# cost model behind the command floor (measured); the values live in drilling/runtime_cost.py


def _session_folder():
    import os
    import shutil
    import tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"drilling-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


def min_command_s(resolutions, samples=16):
    """Shortest policy command (s) for cameras of these (width, height): drilling/runtime_cost.min_command_s."""
    return runtime_cost.min_command_s(resolutions, samples)


class NewtonDrillingBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset="wall", task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from drilling.config import Config, TASKS
        from drilling.env import DrillingEnv
        from drilling.kinematics import ArmKinematics
        from drilling.robot_setup import validate, check_task
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["drilling"]
        qualified = tuple(catalog["aliases"])
        if not isinstance(robot, str):
            try:
                robot = validate(robot)
            except ValueError as exc:
                raise UnsupportedConfigurationError(str(exc)) from exc
        if robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"drilling does not support {robot!r}: {catalog['unsupported'][robot]}")
        if robot not in qualified:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
        if preset not in ("wall", "bench"):
            raise UnsupportedConfigurationError("Presets are wall and bench")
        values = dict(config or {})
        if "layout" in values:
            raise UnsupportedConfigurationError("layout is an owner/probe field, not a launch option")
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        if device is not None:
            values["device"] = device
        self.config = Config.from_dict(values)
        try:
            check_task(robot, self.config.preset, self.config.task)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        single = robot.startswith("single-")
        # the station camera frames the work point (the wall marks, the vise under the spindle); the kit trolley, the
        # shop vacuum and the parts lying on the bench are outside it, so the default observation adds the overview,
        # in which every tool and part of every task is visible (looked at for both presets)
        default_cams = ("station_rgb", "overview_rgb")
        self.cameras = tuple(cameras if cameras is not None else default_cams)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; others are privileged diagnostics")
        if single and "left_wrist_rgb" in self.cameras:
            raise UnsupportedConfigurationError(f"{robot} has one arm (right mount): no left_wrist_rgb")
        self.options = dict(renderer_options or {})
        idle = self.options.get("idle_release_s", IDLE_RELEASE_S)
        if idle is not None and (isinstance(idle, bool) or not isinstance(idle, (int, float)) or not math.isfinite(idle)
                                 or idle < 0):
            raise ValueError("renderer_options idle_release_s must be null or a finite number of seconds >= 0")
        self.idle_release_s = None if idle is None else float(idle)
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("drilling session (auto-created; pruned by retention)\n")
        self.env = DrillingEnv(self.config, output_dir=self.output)
        self.ik = ArmKinematics(self.env.world)
        self.arms = tuple(self.env.world.rig.instances)
        self.images = deque()
        self.last_timing = {}
        self._renderer = self._prestart = self._idle_timer = None
        self._reapers = []
        self._paused = False
        self._restart_due = False                # a frame error cost the live session its worker: the watch restarts it
        self._watch_stop = threading.Event()
        self._watcher = None
        self.worker_events = {"cold_starts": 0, "resume_restarts": 0, "last_resume_restart_s": None, "frames": 0,
                              "fault_restarts": 0, "released": []}
        self.trace = None
        if recording and self.config.record_trace:
            from drilling.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        self.camera_specs = self._scene_camera_specs()
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard",
                                "cartesian_wrench"})
        self.min_command_s, floor_note = self._floor()
        grip_total = 2.*float(self.env.world.joints[self.env.world.gripper_joints(self.arms[0])[0]]["effort_limit"])
        idle_text = ("keeps it until stop (renderer_options idle_release_s null)" if self.idle_release_s is None else
                     "gives it back at the pause (renderer_options idle_release_s 0)" if self.idle_release_s == 0. else
                     f"keeps it for {self.idle_release_s:g} s (renderer_options idle_release_s) and then gives it back")
        self._capabilities = Capabilities(
            "drilling", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, True, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": "One tick = 1 ms (MuJoCo rigid step); the hole module and tool mechanisms update every tick",
                "control": "Physical position servos; arm joint targets are followed by a time-synchronized "
                           "jerk-limited trajectory inside Franka's joint limits (datasheet speeds, 10 rad/s^2, "
                           "5000 rad/s^3); IK computes targets only; cartesian_wrench adds J^T(F, M) at the TCP as "
                           "joint-torque feed-forward for its duration, inside the same per-joint torque cap as the "
                           "servo and changing at most 1000 N m/s; a held tool is the arm's compensated configured load",
                "arms": ",".join(self.arms),
                "gripper": "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode "
                           f"grasp (total N of both fingers, <= {grip_total:g} N; 50 mm/s until blocked, then force)",
                "tools": "Tools are operated only by contact: a finger presses a trigger, turns a selector, pulls a feed "
                         "handle or a switch; there is no 'spindle on' action",
                "keyboard": "Game-style chords held for ticks/rate_hz s (<= 10 s): W/S A/D R/F move the TCP along world "
                            "+y/-y -x/+x +z/-z at linear_speed_m_s; I/K J/L U/O rotate it about world x, y, z at "
                            "angular_speed_rad_s; ALT x0.2, SHIFT x6; Q closes, E opens the fingers (P x0.2, T x10); "
                            "H grips in force mode (35 N total; with G 10 N, with B 70 N) until E",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; holes, dimples, burrs, "
                           "spall and dust are drawn from the hole records; no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions",
                "min_command_s": floor_note,
                "batches": "A policy request - one command or {'actions': [...]} - is checked whole before its first "
                           "command runs for everything that does not depend on where the arms will be: every "
                           "command's duration (the floor, whole ticks), arm and joint names, joint limits, finger "
                           "widths, grip forces and keyboard chords; a request with a refused command anywhere in it "
                           "runs nothing and the session stays usable. A Cartesian or keyboard-move command that is "
                           "not the first of its batch is solved when its turn comes, from the pose the earlier "
                           "commands left: if it is unreachable then, the earlier commands have run and the session "
                           "waits for the owner's recovery (POST /v1/recover). Send such commands one per request",
                "render_worker": (f"One Isaac render worker per session, kept for as long as the session is live and "
                                  f"not paused, whatever the time between commands: no policy command waits for a worker "
                                  f"to start (it holds about 4.6 GB of GPU memory: pause or stop a session that is not in "
                                  f"use). A paused session {idle_text}; the owner's resume request restarts a "
                                  f"released worker and returns after its first frame (up to about "
                                  f"{runtime_cost.COLD_START_S:g} s, measured: the owner's wait). Also released on stop "
                                  f"and on a reset to another variation (restarted inside the reset). After a crash of "
                                  f"the worker process or a frame error the session starts a new worker by itself "
                                  f"within about {WORKER_WATCH_S:g} s; a camera command that arrives while that worker "
                                  f"is still starting (up to about {runtime_cost.COLD_START_S:g} s) waits for it, and "
                                  f"that fault path alone is outside the 200x guarantee"),
                "checkpoints": "In memory for this session only; a checkpoint restores only into the variation (preset, "
                               "task, seed, robot) it was saved in"})
        if self.cameras:
            self._watcher = threading.Thread(target=self._watch_worker, name="drilling-render-watch", daemon=True)
            self._watcher.start()

    def _scene_camera_specs(self):
        """Specifications of the policy cameras mounted in the present scene. The station camera's mount depends on
        the preset (on the cart's rear mast at the wall, fixed over the drill-press bench), so this is read again
        after a reset to another variation."""
        specs = {}
        mounted = self.env.world.scene.cameras
        for name in POLICY_CAMERAS:
            if name not in mounted:
                continue
            cam = mounted[name]
            width = self.options.get("width", cam.get("width", 960))
            height = self.options.get("height", cam.get("height", 540))
            focal = .018/math.tan(math.radians(cam["fov"]/2))
            station = "cart_rear_mast" if self.config.preset == "wall" else "bench_fixed"
            mount = "link7_d405_envelope" if "wrist" in name else (station if name == "station_rgb" else "ceiling_corner")
            specs[name] = CameraSpec(name, mount, width, height, width, height, (.036, .036*height/width), (focal, focal))
        return specs

    def _floor(self):
        """(min_command_s, its capability note) for the session's cameras as the present scene mounts them."""
        samples = self.options.get("samples", self.config.policy_samples)
        resolutions = [(self.camera_specs[c].width_px, self.camera_specs[c].height_px) for c in self.cameras]
        floor = min_command_s(resolutions, samples)
        note = (f"{floor:g}: policy commands shorter than this are refused. Floor = (per-command overhead + "
                "inverse-kinematics solve + each camera's frame cost by its pixel count) / (design ratio "
                f"{runtime_cost.DESIGN_RATIO:g} - physics ratio {runtime_cost.PHYSICS_RATIO_MAX:g}), under the "
                f"{runtime_cost.LIMIT:g}x wall/sim limit (drilling/runtime_cost.py; {len(self.cameras)} camera(s)"
                f"{' ' if resolutions else ''}{', '.join(f'{w}x{h}' for w, h in resolutions)} at {samples} spp). The "
                "per-command overhead is the largest stall measured with several sessions running side by side on "
                "this machine, not a bound. The floor is the same for every command kind and every idle_release_s. "
                "Owner commands are not floored and are outside the 200x guarantee")
        return floor, note

    @property
    def capabilities(self):
        """The session's capabilities; notes.render_worker_state is the worker's state at the time of the request."""
        self._capabilities.notes["render_worker_state"] = self._worker_state()
        return self._capabilities

    def _worker_state(self):
        ev = self.worker_events
        if not self.cameras:
            state = "none (no policy camera)"
        elif self._renderer is not None and self._alive(self._renderer):
            state = "ready"
        elif self._prestart is not None:                     # a worker started ahead of the next operation
            state = "ready" if self._prestart.get("warmed") else "starting"
        else:
            state = "released"
        return (f"{state}; session {'paused' if self._paused else 'live'}; frames drawn {ev['frames']}; worker starts "
                f"inside an operation {ev['cold_starts']} (the session's first image is one), inside a resume "
                f"{ev['resume_restarts']}"
                + (f" (last {ev['last_resume_restart_s']:.1f} s)" if ev["last_resume_restart_s"] is not None else "")
                + f", by the session itself after a worker fault {ev['fault_restarts']}"
                + (f"; released: {', '.join(ev['released'][-4:])}" if ev["released"] else ""))

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        """Reset to the session's task, or - with options {preset, task} and/or a seed - to another catalogue
        variation: the world is rebuilt for it in this process."""
        from dataclasses import replace
        from drilling.variants import resolve
        from drilling.env import DrillingEnv
        from drilling.kinematics import ArmKinematics
        from drilling.robot_setup import check_task
        # A refused selection (an unknown option, preset or task, a task of another preset, a seed outside the
        # catalogue, a task this robot is not offered for) is decided here, before anything is touched, and raised as
        # ActionRejectedError: the gateway then answers 400 and keeps the session as it was. Any other error out of a
        # reset is recorded by the gateway as a runtime fault that only a valid reset or the owner's recovery clears.
        try:
            values = dict(options or {})
            if set(values) - {"preset", "task"}:
                raise ValueError("Reset selectors are preset, task (options) and seed; got "
                                 f"{sorted(map(str, set(values) - {'preset', 'task'}))}")
            for key in ("preset", "task"):
                if values.get(key) is not None and not isinstance(values[key], str):
                    raise ValueError(f"Reset option {key} must be a name (string)")
            preset = values.get("preset", None if "task" in values else self.config.preset)
            task = values.get("task", self.config.task if preset == self.config.preset else None)
            sel = resolve(preset, task, self.config.seed if seed is None else seed)
            check_task(self.config.robot, sel["preset"], sel["task"])
        except (ValueError, TypeError) as exc:
            raise ActionRejectedError(f"Reset refused, nothing changed: {exc}") from exc
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        if (sel["preset"], sel["task"], sel["seed"]) == (self.config.preset, self.config.task, self.config.seed):
            return self.env.reset(), {"continuity": "reset", "variation": sel}
        config = replace(self.config, **sel)
        candidate = DrillingEnv(config, output_dir=self.output)
        with self.env.lock:
            self._release_workers("variation_reset")             # the scene changes; the old worker cannot draw it
            for t in self._reapers:
                t.join()
            self._reapers = []
            old, self.env, self.config = self.env, candidate, config
            candidate.generation = old.generation + 1
            self.ik = ArmKinematics(candidate.world)
            self._scene_path = None
            if self.trace:
                path = self.trace.store.path
                self.trace.close()
                from drilling.trace import Trace
                self.trace = Trace(candidate, path)
            self._capabilities.notes["task"] = f"{config.preset}/{config.task} seed {config.seed}"
            # the new scene's cameras: the station camera's mount follows the preset (capabilities, the policy's
            # hardware record and every observation's camera block read self.camera_specs); the floor follows them
            self.camera_specs = self._scene_camera_specs()
            self.min_command_s, self._capabilities.notes["min_command_s"] = self._floor()
            self._capabilities = replace(self._capabilities,
                                         camera_specs=tuple(self.camera_specs[c] for c in self.cameras))
            if self.cameras:                     # the new scene's worker: the reset's own observation waits for it
                self._start_renderer_early(warm=True)
        del old
        return self.env.get_state(), {"continuity": "explicit_reset", "variation": sel}

    def select_variation(self, request):
        from drilling.variants import resolve
        from drilling.task import INSTRUCTIONS
        if set(request) - {"preset", "task", "seed"}:
            raise ValueError("Selectors are preset, task and seed")
        sel = resolve(request.get("preset", None if "task" in request else self.config.preset), request.get("task"),
                      request.get("seed", 0))
        return dict(variation=sel, task=INSTRUCTIONS[sel["task"]],
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
        from drilling.render import export
        if getattr(self, "_scene_path", None) is None or not self._scene_path.is_file():
            self._scene_path, self.env.render_meta = export(self.env.world, self.output/"render", seed=self.config.seed)
        return self._scene_path

    def _new_renderer(self):
        from drilling.render import Renderer
        return Renderer(self.env.world, self.output/"render", self._scene(), width=self.options.get("width"),
                        height=self.options.get("height"), samples=self.options.get("samples", self.config.policy_samples))

    def _watch_worker(self):
        while not self._watch_stop.wait(WORKER_WATCH_S):
            try:
                self._check_worker()
            except Exception:                    # a failed look or start is not a fault of the session: the next
                pass                             # operation meets the worker itself, as before

    def _check_worker(self):
        """The session's own look at its render worker (every WORKER_WATCH_S on a live session with cameras). If the
        kept worker's process has exited, or a frame error has cost the session its worker, a new one is started now
        and draws its first frame as soon as no command holds the world, instead of inside the next camera command.
        A paused session is left alone (its resume restarts the worker) and so is a worker released on purpose.
        Returns whether a restart was begun."""
        r = self._renderer
        dead = r is not None and not self._alive(r)
        if self._paused or not self.cameras or not (dead or self._restart_due):
            return False
        if not self.env.lock.acquire(timeout=0.05):          # a command holds the world; it or the next look acts
            return False
        try:
            r = self._renderer
            dead = r is not None and not self._alive(r)
            if self._paused or self._prestart is not None or not (dead or (self._restart_due and r is None)):
                return False
            if dead:
                self._renderer = None
                self.worker_events["released"].append("worker_exited")
                del self.worker_events["released"][:-32]
                self._reap([r], None)
            self._restart_due = False
            self.worker_events["fault_restarts"] += 1
            self._start_renderer_early(warm=True)
            if self._prestart is not None:
                self._prestart["by_watch"] = True
            return True
        finally:
            self.env.lock.release()

    def _start_renderer_early(self, warm=False):
        """Start the session's worker in a background thread. With warm, the thread also renders the policy cameras
        once (image discarded) when no command holds the world - after a variation reset - because a new worker's
        first frame costs seconds more than the next (measured 4.5 s); during a command's physics (the fault path: no
        worker after a frame error or a crash) it skips that and the command's own observation is the first frame."""
        if self._renderer is not None and not self._alive(self._renderer):
            dead, self._renderer = self._renderer, None
            self.last_timing["replaced_exited_worker"] = True
            self.worker_events["released"].append("worker_exited")
            del self.worker_events["released"][:-32]
            self._reap([dead], None)
        if self._prestart is not None or self._renderer is not None:
            return
        self._restart_due = False
        self._scene()
        holder = {"renderer": None, "error": None, "warm_s": None, "warmed": False}
        env = self.env

        def work():
            try:
                t0 = time.perf_counter()
                holder["renderer"] = self._new_renderer()
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:
                holder["error"] = exc
                return
            if not (warm and self.cameras):
                return
            while not env.lock.acquire(timeout=0.05):        # wait for the world to be free, unless a command
                if self._prestart is not holder:             # has taken this worker: its observation warms it
                    return
            try:
                if self._prestart is holder and self.env is env:
                    holder["renderer"].observe(tuple(self.cameras), env.world)
                    holder["warmed"] = True
                    self.worker_events["frames"] += 1
            except BaseException as exc:                     # the command's own frame will retry on this worker
                holder["warm_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                env.lock.release()
        holder["thread"] = threading.Thread(target=work, name="drilling-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    @staticmethod
    def _alive(r):
        p = getattr(r, "process", None)
        return not getattr(r, "closed", False) and (p is None or p.poll() is None)

    def _operation_renderer(self, count_cold=True):
        """The session's worker: the kept one, else the one started during this command's physics or by a variation
        reset, else a cold start. A kept worker whose process has exited is replaced. worker_events cold_starts counts
        the worker starts that an operation (the session's first image, a reset's or a command's observation, an
        owner's image) took or waited for; a start inside a resume is counted as resume_restarts instead."""
        self._cancel_idle()
        if self._renderer is not None and not self._alive(self._renderer):
            dead, self._renderer = self._renderer, None
            self.last_timing["replaced_exited_worker"] = True
            self._reap([dead], None)
        if self._renderer is not None:
            self.last_timing["kept_worker"] = True
            return self._renderer
        holder, self._prestart = self._prestart, None
        if holder is not None:
            t0 = time.perf_counter()
            holder["thread"].join()
            self.last_timing.update(prestart_wait_s=time.perf_counter() - t0, prestart_warm_s=holder["warm_s"],
                                    prestart_frame=holder["warmed"])
            if holder["error"] is not None:
                raise BackendProtocolError(f"Render worker failed to start: {holder['error']}")
            self._renderer = holder["renderer"]
            # a worker the session restarted by itself and that has already drawn its first frame cost this
            # operation nothing: not a start inside an operation
            if count_cold and not (holder.get("by_watch") and holder["warmed"]):
                self.worker_events["cold_starts"] += 1
            return self._renderer
        t0 = time.perf_counter()
        self._renderer = self._new_renderer()
        self.last_timing["cold_start_s"] = time.perf_counter() - t0
        if count_cold:
            self.worker_events["cold_starts"] += 1
        return self._renderer

    def _end_operation(self):
        """After an operation the worker is kept for the next one. On a live session nothing else happens: it keeps
        its worker whatever the time to the next command. An operation on a paused session (an owner's image) restarts
        the paused session's idle release."""
        if self._paused:
            self._pause_release()

    def _pause_release(self):
        """A paused session gives its worker back idle_release_s after the pause (or after its last operation while
        paused): at once for 0, never for null. Returns whether it was released now."""
        self._cancel_idle()
        if self._renderer is None and self._prestart is None:
            return False
        if self.idle_release_s is None:
            return False
        if self.idle_release_s <= 0.:
            return self._release_workers("pause")
        self._arm_idle(self.idle_release_s)
        return False

    def _arm_idle(self, idle):
        self._cancel_idle()
        timer = threading.Timer(idle, self._idle_expired)
        timer.args = (timer,)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _cancel_idle(self):
        timer, self._idle_timer = getattr(self, "_idle_timer", None), None
        if timer is not None:
            timer.cancel()

    def _idle_expired(self, timer):
        with self.env.lock:
            if self._idle_timer is not timer or not self._paused:    # an operation, a resume or close came first
                return
            self._idle_timer = None
            self._release_workers("paused_idle")

    def _reap(self, rs, holder):
        def reap():
            if holder is not None:
                holder["thread"].join()
                if holder["renderer"] is not None and holder["renderer"] not in rs:
                    rs.append(holder["renderer"])
            for r in rs:
                r.close()
        t = threading.Thread(target=reap, name="drilling-render-release", daemon=False)
        t.start()
        self._reapers = [x for x in getattr(self, "_reapers", []) if x.is_alive()] + [t]

    def _release_workers(self, reason):
        """Release the session's worker and one still starting; the physics state is untouched and the next image
        starts a worker again. Returns whether there was one."""
        self._cancel_idle()
        rs = [r for r in (self._renderer,) if r is not None]
        holder = self._prestart
        self._renderer, self._prestart = None, None
        if not rs and holder is None:
            return False
        self.worker_events["released"].append(reason)
        del self.worker_events["released"][:-32]
        self._reap(rs, holder)
        return True

    def release_render_worker(self):
        """Release the render worker now (an in-process owner's own choice; no HTTP route calls this). The next image
        restarts it inside that operation, so this is not for a live session that serves a policy: pause it instead."""
        with self.env.lock:
            self._restart_due = False
            return self._release_workers("requested")

    def session_paused(self, paused):
        """Session-lifecycle hook (pause/resume of the GAR session, called inside the owner's request).

        Pause: the worker is kept for idle_release_s more seconds and then released (0: now; null: never). Returns
        whether it was released now.
        Resume: if the paused session gave its worker back (or the worker died), a new one is started and draws its
        first frame HERE, before the resume request returns, so the owner's resume carries the wait (COLD_START_S)
        and the first policy command after it finds a ready worker. Returns whether a worker had to be started. If
        the start fails the error is raised to the session layer (which logs it and resumes all the same); the next
        camera command then starts a worker itself, as after any worker fault."""
        with self.env.lock:
            if paused:
                self._paused = True
                return self._pause_release()
            self._paused = False
            self._cancel_idle()
            if not self.cameras or (self._renderer is not None and self._alive(self._renderer)):
                return False
            t0 = time.perf_counter()
            r = self._operation_renderer(count_cold=False)
            try:
                r.observe(tuple(self.cameras), self.env.world)   # the slow first frame; the image is discarded
            except BaseException:
                self._release_workers("frame_error")
                raise
            self.worker_events["frames"] += 1
            self.worker_events["resume_restarts"] += 1
            self.worker_events["last_resume_restart_s"] = time.perf_counter() - t0
            return True

    # ------------------------------------------------------------------ observation
    def _render(self, names):
        with self.env.lock:
            r = self._operation_renderer()
            try:
                t0 = time.perf_counter()
                result = r.observe(tuple(names), self.env.world)
                self.last_timing["frame_s"] = time.perf_counter() - t0
                self.worker_events["frames"] += 1
            except BaseException:
                self._release_workers("frame_error")             # a worker that failed a frame is not kept
                self._restart_due = not self._paused             # the watch starts the live session a new one
                raise
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
        """What an FR3 reports: protective stops, and per arm the estimated external joint torques (contact forces
        mapped to the joints, payload compensated, filtered: libfranka tau_ext_hat_filtered) and gripper widths."""
        w = self.env.world
        out = {"arms": {}, "refused_targets": int(w.refused_targets)}
        for a in self.arms:
            tau = w._tau_ext.get(a)
            out["arms"][a] = {"stopped_s": round(max(0., float(w._reflex_until.get(a, 0.)) - w.sim_time), 4),
                              "tau_ext_nm": [round(float(v), 3) for v in (tau if tau is not None else np.zeros(7))],
                              "gripper_width_m": round(float(sum(w.jq[w.joints[j]["q"]] for j in w.gripper_joints(a))), 5)}
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
        if solve and (move.any() or turn.any()):
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
            gap = float(np.mean([w.jq[w.joints[j]["q"]] for j in w.gripper_joints(arm)]))
            rate = .2 if "P" in keys else (10. if "T" in keys else 1.)
            step = (1. if "E" in keys else -1.)*action.gripper_half_speed_m_s*rate*duration
            targets.update(self._fingers(arm, min(.04, max(0., gap + step))))
        return targets, ticks, grip, None

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

    def act(self, action, *, floor_s=0.):
        """Plan and validate the whole command first (no side effect; a refused command raises ActionRejectedError and
        leaves the session usable); then apply grip modes and wrench commands and run the physics. With floor_s (the
        policy path) a command shorter than that is refused; the command is planned once (a Cartesian or keyboard
        command's inverse-kinematics solve runs once per command)."""
        with self.env.lock:
            try:
                targets, ticks, grip, wrench = self._check(action, floor_s)
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise self._rejected(exc) from exc
            return self._execute(targets, ticks, grip, wrench)

    @staticmethod
    def _rejected(exc, prefix=""):
        """ActionRejectedError for a refusal decided from the command and the robot's own limits and joints (duration,
        tick, floor, joint limits, finger width, grip force, chord, reach): that text is what a real controller tells
        its operator, so the gateway may show it to the policy (policy_reason)."""
        err = ActionRejectedError(prefix + str(exc))
        err.policy_reason = prefix + str(exc)
        return err

    def _check(self, action, floor_s, solve=True):
        """Plan and validate one command without side effects; ValueError / UnsupportedConfigurationError /
        UnsupportedActionError when it is refused. With solve=False the inverse-kinematics solve of a Cartesian or
        keyboard-move command is left out (it depends on where the arm is when the command's turn comes) and
        everything else is checked exactly as when the command runs."""
        targets, ticks, grip, wrench = self._plan(action, solve)
        if ticks*self.config.rigid_dt < floor_s - 1e-12:
            raise ValueError(f"Policy commands with {len(self.cameras)} camera observation(s) must last >= "
                             f"{floor_s*1e3:.0f} ms of simulated time ({ticks} ticks requested) to stay within "
                             f"the {runtime_cost.LIMIT:g}x wall/sim budget (capabilities notes.min_command_s)")
        self._validate_targets(targets)
        for arm, f in grip.items():
            limit = float(self.env.world.joints[self.env.world.gripper_joints(arm)[0]]["effort_limit"])
            if not 0. <= f <= limit:
                raise ValueError(f"inward_effort_n must be in [0, {2*limit:g}] N (both fingers)")
        if wrench is not None:
            from drilling import params as P
            if np.linalg.norm(wrench[1]) > P.WRENCH_MAX_N or np.linalg.norm(wrench[2]) > P.WRENCH_MAX_NM:
                raise ValueError(f"Wrench beyond the accepted range (|F| <= {P.WRENCH_MAX_N} N, "
                                 f"|M| <= {P.WRENCH_MAX_NM} N m)")
        return targets, ticks, grip, wrench

    def _precheck(self, actions):
        """Every command of a policy request, checked before the first one runs (DrillingPolicyAccess.decode_actions):
        the floor and everything else that does not need the arm's pose. Nothing is solved, written or stepped."""
        many = len(actions) > 1
        for index, action in enumerate(actions):
            where = f"command {index + 1} of {len(actions)}: " if many else ""
            try:
                self._check(action, self.min_command_s, solve=False)
            except UnsupportedActionError as exc:
                raise UnsupportedActionError(where + str(exc)) from exc
            except (ValueError, UnsupportedConfigurationError) as exc:
                raise self._rejected(exc, where + ("nothing in this request was executed. " if many else "")) from exc

    def _plan(self, action, solve=True):
        grip, wrench = {}, None
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
        elif isinstance(action, CartesianWrench):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            targets = {}
            wrench = (arm, np.asarray(action.force_n, float), np.asarray(action.moment_nm, float))
        elif isinstance(action, Keyboard):
            targets, ticks, grip, wrench = self._keyboard(action, solve)
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta, "
                                         "cartesian_wrench, keyboard")
        return targets, ticks, grip, wrench

    def _execute(self, targets, ticks, grip, wrench):
        w = self.env.world
        for arm, f in grip.items():
            w.set_grip_force(arm, f)
        if wrench is not None:
            w.set_wrench(wrench[0], wrench[1], wrench[2], ticks)
        before = w.tick
        n_reflex, n_refused = len(w.reflexes), w.refused_targets
        self.last_timing = {}
        self._cancel_idle()                          # an operation is running: a paused-idle release waits for it
        if self.cameras:                             # no worker (after a frame error or a crash): start one now
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
                                                      if u > w.sim_time}},
                           ev.get("score", 0.), False)

    def policy_access(self):
        from drilling.task import INSTRUCTIONS
        from ..keyboard_control import SUPPORTED_KEYS
        proprio = {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")}
        for a in self.arms:
            proprio[("robot_status", "arms", a, "stopped_s")] = None
            proprio[("robot_status", "arms", a, "tau_ext_nm")] = 7          # per-joint vector (the policy filter
            # rejected every observation while this was declared a scalar: GAR session test)
            proprio[("robot_status", "arms", a, "gripper_width_m")] = None
        proprio[("robot_status", "refused_targets")] = None
        return DrillingPolicyAccess(
                            self.config.robot, tuple(self.camera_specs[c] for c in self.cameras), proprio,
                            self.kinds, lambda: INSTRUCTIONS[self.config.task],
                            {"dt_s": self.config.rigid_dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()},
                             "gripper": self.capabilities.notes["gripper"],
                             "tools": self.capabilities.notes["tools"],
                             "keyboard": self.capabilities.notes["keyboard"],
                             "cartesian_wrench": "force_n / moment_nm at the TCP in world axes, fed forward as joint "
                                                 "torques for duration_s; |F| <= 150 N, |M| <= 25 N m",
                             "protective_stop": f"{'Panda' if 'panda' in self.config.robot else 'FR3'} collision reflex in "
                                                "the drilling contact setting (external joint torque above 95 % of the "
                                                "joint's effort limit, held tools' weight compensated) and the velocity "
                                                "reflex stop an arm; it then refuses joint targets for 1 s",
                             "batches": self.capabilities.notes["batches"],
                             "min_command_s": self.min_command_s},
                            keyboard_keys=tuple(SUPPORTED_KEYS), precheck=self._precheck)

    def act_policy(self, action):
        """Every policy command, with or without cameras, lasts at least min_command_s of simulated time."""
        result = self.act(action, floor_s=self.min_command_s)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "hero" if label in ("interactive", "http") else label
        valid = tuple(c for c in POLICY_CAMERAS if c in self.camera_specs) + tuple(
            c for c in DIAGNOSTIC if c in self.env.world.scene.cameras)
        if name not in valid:
            raise ValueError(f"Render label must be one of {valid}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        """Privileged geometry: workpieces (boxes in their body frames with their hole records), parts, tools, marks."""
        w = self.env.world
        sc = w.scene
        return {"drillables": {n: dict(body_id=int(d.body), material=d.material, half_extents_m=[float(v) for v in d.half],
                                       pose_xyzw=[float(v) for v in w.bq[d.body]], holes=len(d.holes))
                               for n, d in w.hm.drillables.items()},
                "parts": {n: dict(kind=p["kind"], body_id=int(p["body"])) for n, p in sc.parts.items()},
                "tools": {n: type(t).__name__ for n, t in sc.tools.items()},
                "marks": self.env.layout.get("marks", []),
                "robot_bodies": {w.model.body_label[b]: int(b) for b in sc.robot_bodies},
                "ids": "get_geometry(<drillable name>) returns its box and every hole record in its body frame",
                "source": "The simulator's own geometry and hole records (drilling/holes.py), not a camera estimate"}

    def get_geometry(self, geometry_id):
        w = self.env.world
        gid = str(geometry_id)
        if gid not in w.hm.drillables:
            raise KeyError(f"Unknown geometry id {gid!r}; ids are the drillable names of /v1/geometry")
        d = w.hm.drillables[gid]
        return dict(id=gid, material=d.material, body_id=int(d.body), half_extents_m=[float(v) for v in d.half],
                    pose_xyzw=[float(v) for v in w.bq[d.body]], frame="body frame, box centred at the origin",
                    holes=[h.summary() for h in d.holes])

    def get_geometry_asset(self, asset_id):
        raise KeyError(f"Unknown asset {asset_id!r}; drilling has no separate geometry assets")

    def variation_contract(self):
        from drilling.variants import contract
        return dict(**contract(), current=dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed),
                    reset="POST /v1/variations/select (or reset with options {preset, task} and a seed) rebuilds the "
                          "world for the selection in this session")

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
                raise ValueError(f"Checkpoint {label!r} was saved in variation {snap.get('variation')}; the session is now "
                                 f"{self._variation_key()}. Reset to that variation first, then restore")
            self.env.restore(snap)
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.tick, "generation": self.env.generation,
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
        """Session end: the render worker (kept or still starting) is released and reaped before close returns."""
        self._watch_stop.set()
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=5)
        with self.env.lock:
            self._paused = False
            self._restart_due = False
            self._cancel_idle()
            self._release_workers("close")
            for t in self._reapers:
                t.join()
            self._reapers = []
        if self.trace:
            self.trace.close()
            self.trace = None

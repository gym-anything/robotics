"""Painting bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo CPU rigid world, the painting film /
applicator models on the CPU, Isaac RTX images).

The robot interface is classroom_arrangement's qualified Mobile FR3 Duo stack (base twist, Spine, joint targets,
Cartesian targets through IK, Franka Hand position/force modes, keyboard chords), unchanged. Paint moves only through
the tools' contacts and the aerosol button a fingertip presses; no paint, valve or fixture control is a policy action.
One render worker per session, started on the first image and released after renderer_options idle_release_s
(default 300 s) without an image (a refused command draws none); close() releases it at once.
While it is released a command is refused, and the policy is told why
(policy_reason) and, in its help, the rule (camera_renderer). A command the inherited planner refuses (duration or tick,
joint or finger limit, grip force, base twist limit, IK reach or self-contact, base held after a protective stop; the
G1's own checks likewise) is refused with the controller's own text as its policy_reason (_planner_refusal).
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
from .newton_classroom_arrangement import (NewtonClassroomArrangementBackend as _Classroom, POLICY_CAMERAS, BASE_CAMERAS,
                                          BASE_KEYS, MAX_ACTION_S, MIN_CARTESIAN_S, MIN_ACTION_S, IK_BUDGET_S, _finite)
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta, BaseTwist, Keyboard
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError, ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

DIAGNOSTIC = ("room", "wall", "raking", "strip", "ceiling_line", "floor_tools", "table", "paper", "spray_box", "stencil",
              "side", "chase")
SESSION_MARKER = ".gar-painting-session"
SESSION_RETENTION = 20
IDLE_RELEASE_S = 300.0


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"painting-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


class NewtonPaintingBackend(_Classroom):
    """Reuses the classroom backend's command planning (_plan, _keyboard, act), robot configuration and geometry
    helpers; replaces the environment, rendering, evaluation, variation and policy-access parts."""

    QUALIFIED_MODELS = ("mobile-fr3-duo@1", "mobile-fr3-single@1")

    def __init__(self, *, robot="mobile-fr3-duo", preset=None, task=None, seed=0, reflex_profile="contact_x5", cameras=None,
                 output_dir=None, renderer_options=None, recording=True, device=None, reflex_stops_base=True,
                 policy_samples=None, contract=None, **unknown):
        from painting.config import Config
        from painting.env import PaintingEnv
        from painting.variants import resolve
        from classroom_arrangement.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["painting"]
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown painting options {sorted(unknown)}; options: robot, preset, task, seed, reflex_profile, cameras, "
                "renderer_options, recording, device, reflex_stops_base, policy_samples, contract")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if policy_samples is not None and (type(policy_samples) is not int or not 1 <= policy_samples <= 256):
            raise UnsupportedConfigurationError("policy_samples must be an integer 1..256")
        if device not in (None, "cpu"):
            raise UnsupportedConfigurationError(f"painting runs MuJoCo and the paint fields on the CPU only; device={device!r} "
                                                "is not supported (the renderer picks its own GPU)")
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"painting does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(catalog['aliases'])}")
        try:
            v = resolve(preset, task, int(seed), contract)
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        from painting.robot_setup import check_task
        try:
            check_task(robot, v["preset"])
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        extra = {} if policy_samples is None else dict(policy_samples=int(policy_samples))
        self.config = Config(preset=v["preset"], task=v["task"], seed=v["seed"], robot=robot, reflex_profile=reflex_profile,
                             reflex_stops_base=reflex_stops_base, contract=v.get("contract"), **extra)
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("painting session (auto-created; pruned by retention)\n")
        self.env = PaintingEnv(self.config, output_dir=self.output)
        w = self.env.world
        self.arms = tuple(w.arms)
        self.ik = ArmKinematics(w)
        self.policy_cameras = tuple(c for c in POLICY_CAMERAS if c == "head_rgb" or c in BASE_CAMERAS or c.split("_")[0] in self.arms)
        self.cameras = tuple(cameras if cameras is not None else ("head_rgb",))
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(self.policy_cameras):
            raise ValueError(f"Policy cameras are {self.policy_cameras}; others are privileged diagnostics")
        width, height = self.options.get("width", 640), self.options.get("height", 360)
        from classroom_arrangement import params as CP
        self.camera_specs = {}
        for name in self.policy_cameras:
            fov = CP.HEAD_HFOV if name == "head_rgb" else (CP.BASE_CAMERA_HFOV if name in BASE_CAMERAS else CP.WRIST_HFOV)
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
            from painting.trace import Trace
            self.trace = Trace(w, self.output/"physics.sqlite3")
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "base_twist", "keyboard"})
        from ..keyboard_control import SUPPORTED_KEYS
        self.keyboard_keys = tuple(sorted(set(SUPPORTED_KEYS) | set(BASE_KEYS)))
        self.capabilities = Capabilities(
            "painting", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())

    def _notes(self):
        notes = dict(super()._notes())
        c = self.config
        notes.update(
            task=f"{c.preset}/{c.task} seed {c.seed}: {self.env.spec['instruction']}",
            paint="Paint is a physical field: films on the wall, trim, paper, crate, tray ramp and drop cloth flow (yield-stress "
                  "thin-film lubrication), level, sag, dry, soak into porous surfaces and form lap marks; tools carry paint "
                  "in their nap/bristles, load it from containers and transfer it only by contact; the aerosol sprays only "
                  "while a fingertip holds its button down. None of this is commanded directly.",
            start=f"the task's tool starts in a valid grasp ({', '.join(f'{a}: {t}' for a, t in self.env.spec['held'].items())})",
            qualification="Assisted fixtures are not policy actions")
        notes["min_command_s"] = (
            f"warm camera actions require >= {self.min_action_s:g} s; cold/released RGB actions are rejected "
            "before physics. Request an observation to establish readiness (zero ticks; restart latency included "
            "in that request). Idle GPU release remains enabled. 200x is a measured warm-workload target, "
            "not a hardware-independent bound under shared-host contention.")
        notes.pop("keyboard", None)
        notes["keyboard"] = ("keyboard chords: W/S/A/D/R/F move the chosen arm's hand along world y/x/z, I/K/J/L/U/O rotate it, "
                             "Q/E close/open the fingers, H grips by force (G 10 N, default 35 N, B 70 N total), ALT slow, "
                             "SHIFT fast; UP/DOWN/LEFT/RIGHT drive the base in its frame, Z/X turn it. The start grasp is "
                             "70 N total on a brush or the can (H+B) and 120 N on the roller (the gripper action, 60 N per "
                             "finger): a plain H re-grip halves the hold on a brush")
        return notes

    def act(self, action):
        # Lock order matches observation/render. Check before _plan or physics:
        # a released renderer requires an explicit observation/readiness request.
        # Its restart is timed as that request, never hidden inside a short action.
        with self.env.lock, self._render_lock:
            if self.cameras and self._renderer is None:
                raise self._released_refusal(
                    "RGB renderer is idle/released; no action executed. Request an observation "
                    "(HTTP GET /v1/observation) to warm the cameras, then retry. Readiness advances zero ticks.")
            try:
                return super().act(action)
            except ActionRejectedError as exc:
                if isinstance(exc.__cause__, (ValueError, UnsupportedConfigurationError)):
                    self._planner_refusal(exc)
                raise

    @staticmethod
    def _planner_refusal(err):
        """A refusal the command planner decided from the command and the robot's own limits (_Classroom.act:
        duration or tick, joint or finger limit, grip force, base twist limit, IK reach or self-contact, base held by a
        protective stop; the G1's act likewise). Its text is what a controller reports to its operator, and it is the
        reason the policy is given (policy_reason, the only backend text the gateway shows it). Without it the policy
        reads only "adjust the target or command parameters". classroom_arrangement itself is unchanged."""
        if getattr(err, "policy_reason", None) is None:
            err.policy_reason = str(err)
        return err

    def _release_rule(self):
        from .newton_classroom_arrangement import IDLE_RELEASE_S as default_s      # what _release_later uses
        idle = float(self.options.get("idle_release_s", default_s) or 0.)
        when = f"after {idle:g} s without an image" if idle > 0 else "after every image"
        return when + " (an observation, or the reply of a command that ran; a refused command draws none)"

    def _released_refusal(self, text):
        """The refusal of a command sent while the cameras' render worker is released. The gateway shows a restricted
        caller only policy_reason; without it the policy reads "adjust the target or command parameters" and can take
        the refusal for a wrong arm pose."""
        err = ActionRejectedError(text)
        err.policy_reason = (f"the cameras' render worker is not running: it is released {self._release_rule()}. "
                             "This refusal is not about the command's target or parameters, and nothing moved. "
                             "GET /v1/observation starts the worker again without advancing simulated time.")
        return err

    def _camera_renderer_note(self):
        return ("One render worker draws the RGB cameras. It starts on an image request and is released "
                f"{self._release_rule()}. While it is not running every command is refused before anything moves, "
                "with that reason; GET /v1/observation starts it again without advancing simulated time (its start-up "
                "wait is part of that request).")

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        from painting.variants import resolve
        options = dict(options or {})
        unknown = set(options) - {"preset", "task", "seed", "contract"}
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are preset, task, seed, contract; nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            cur = self._variation()
            try:
                want = resolve(options.get("preset", cur["preset"] if "task" not in options else None), options.get("task", cur["task"]),
                               int(options.get("seed", cur["seed"])), options.get("contract", cur.get("contract")
                               if options.get("task", cur["task"]) == cur["task"] else None))
            except (ValueError, TypeError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            if want != cur:
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def _variation(self):
        c = self.config
        from painting.variants import resolve
        return resolve(c.preset, c.task, c.seed, c.contract)

    def _rebuild(self, v):
        from dataclasses import replace
        from painting.env import PaintingEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            self._release()
            self.config = replace(self.config, preset=v["preset"], task=v["task"], seed=v["seed"], contract=v.get("contract"))
            old = self.env
            self.env = PaintingEnv(self.config, output_dir=self.output)
            self.ik = ArmKinematics(self.env.world)
            self._scene_path = None
            self._checkpoints = {}
            self._geometry = None
            self.env.generation = old.generation + 1
            from dataclasses import replace as _replace
            self.capabilities = _replace(self.capabilities, notes=self._notes())
            if self.recording:
                from painting.trace import Trace
                self.trace = Trace(self.env.world, self.output/f"physics-{self.env.generation}.sqlite3")

    def variation_contract(self):
        from painting.variants import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        from painting.variants import resolve
        from painting.catalogue import describe
        if set(request) - {"preset", "task", "seed", "contract"}:
            raise ValueError("Selectors are preset, task, seed and contract")
        cur = self._variation()
        v = resolve(request.get("preset", cur["preset"] if "task" not in request else None), request.get("task", cur["task"]),
                    int(request.get("seed", cur["seed"])), request.get("contract", cur.get("contract")
                    if request.get("task", cur["task"]) == cur["task"] else None))
        return dict(variation=v, task=describe(v["task"], v["seed"], v.get("contract")),
                    reset_request=dict(seed=v["seed"], options=dict(preset=v["preset"], task=v["task"],
                                                                    contract=v.get("contract"))))

    def evaluate(self):
        """Verification with the settle rule (painting/task.py): the films' deterministic future until set, measured."""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            led = self.env.paint.ledger_residual()
            if led["relative"] > 1e-9:
                raise BackendProtocolError(f"Paint ledger fault (relative residual {led['relative']:.2e}); not a task result")
            ev = self.env.evaluator.evaluate()
            return dict(ev, success=bool(ev["succeeded"]), score=float(ev["score"]), ledger=led)

    # ------------------------------------------------------------------ rendering
    def _scene(self):
        from painting.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env, self.output/"render")
        return self._scene_path

    def _worker(self):
        from painting.render import Renderer
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._renderer is None:
            t0 = time.perf_counter()
            scene = self._scene()
            self._renderer = Renderer(self.output/"render", scene, self.render_meta, width=self.options.get("width", 640),
                                      height=self.options.get("height", 360),
                                      samples=self.options.get("samples", self.config.policy_samples))
            self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return self._renderer

    def _render(self, names):
        with self._render_lock:
            r = self._worker()
            try:
                t0 = time.perf_counter()
                result = r.observe(self.env, tuple(names))
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

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        allowed = self.policy_cameras + DIAGNOSTIC
        if name not in allowed:
            raise ValueError(f"Render label must be one of {allowed}")
        with self.env.lock:
            return self._render((name,))[name]

    # ------------------------------------------------------------------ policy access
    def policy_access(self):
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
        spec = self.env.spec
        return PolicyAccess(self.robot_name, tuple(self.camera_specs[c] for c in self.cameras), proprio, self.kinds,
                            lambda: spec["instruction"],
                            {"dt_s": 0.0005, "arms": list(self.arms), "joints": joints, "base": notes["base"],
                             "spine": notes["spine"], "gripper": notes["gripper"], "control": notes["arms"],
                             "protective_stop": notes["protective_stop"], "start": notes["start"],
                             **({"camera_renderer": self._camera_renderer_note()} if self.cameras else {})})

    # ------------------------------------------------------------------ robot configuration and geometry
    @classmethod
    def _qualified_setup(cls, robot, catalog):
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
            raise UnsupportedConfigurationError("painting qualifies only the Mobile FR3 Duo / single-arm configurations; this "
                                                f"setup needs: {'; '.join(problems)}. Aliases: {', '.join(catalog['aliases'])}")
        return d

    def get_geometry_index(self):
        """Privileged geometry: every shape and body (pencap's rigid catalogue), the tools with their parts and poses,
        the paint charts (frames, sizes, cell sizes, roles) and the task's target regions."""
        w, paint = self.env.world, self.env.paint
        q = w.state.body_q.numpy()
        index = dict(self._catalog().index())
        index.pop("pen_frame", None)
        index.update(
            items={n: dict(kind=it["kind"], body_id=int(it["body"]), pose_xyzw=[float(v) for v in q[it["body"]]],
                           shape_ids=[str(s) for s in it["shapes"]]) for n, it in w.scene.items.items()},
            charts={n: dict(body=c.body, origin=c.origin.tolist(), u=c.u.tolist(), v=c.v.tolist(), size_m=list(c.size),
                            cell_m=c.dx, role=c.role, substrate=c.substrate) for n, c in paint.charts.items()},
            reservoirs={n: dict(body=r.body, volume_mL=r.volume*1e6) for n, r in paint.reservoirs.items()},
            item_ids="get_geometry(<item name>) returns the item's authored parts and current pose",
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
                        mass_kg=float(w.mjm.body_mass[w._mj_body[it["body"]]]), parts=parts,
                        size_convention="box: half extents; capsule/cylinder: radius, half height (local z)")
        return self._catalog().objects[gid]

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints = getattr(self, "_checkpoints", {})
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.world.tick, "generation": self.env.generation}

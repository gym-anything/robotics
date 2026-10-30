"""Soil-excavation bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + MuJoCo CPU rigid world, Newton 1.5
implicit MPM soil on the GPU, two-way coupled; Isaac RTX images).

The robot interface is classroom_arrangement's qualified Mobile FR3 Duo stack (base twist, Spine, joint targets,
Cartesian targets through IK, Franka Hand position/force modes, keyboard chords), unchanged. Soil moves only through
the tools', stones', root ball's, post's and the robot's own contacts with it; no dig, scoop, dump, drill or soil action
exists (the drill's trigger is squeezed by the gripper). One render worker per session, started on the first image and
released after renderer_options idle_release_s (default 300 s) without a request; close() releases it at once.
"""

import math
from collections import deque
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .newton_classroom_arrangement import (NewtonClassroomArrangementBackend as _Classroom, POLICY_CAMERAS, BASE_CAMERAS,
                                          BASE_KEYS, MAX_ACTION_S, _finite)
from ..errors import UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError
from ..policy_access import PolicyAccess
from ..types import CameraSpec, Capabilities

DIAGNOSTIC = ("yard", "work", "work_low", "hole", "top", "tarp", "front", "chase")
SESSION_MARKER = ".gar-soil-session"
SESSION_RETENTION = 20
# HTTP client timeout the backend declares (hardware.client_timeout_s; the gar CLI uses it, else 120 s). set-post's
# verdict simulates a counterfactual brace-release push test of up to ~8 s; on an A10 that took 0.5-1.4 h of wall time
# with the GPU shared, so the default 120 s would report every set-post verify as a timeout.
CLIENT_TIMEOUT_S = 10800.0
SELECTORS = ("task", "seed", "moisture", "topsoil", "stones")


def _session_folder():
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"soil-excavation-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


class NewtonSoilExcavationBackend(_Classroom):
    """Reuses the classroom backend's command planning (_plan, _keyboard, act), robot configuration and geometry helpers;
    replaces the environment, rendering, evaluation, variation and policy-access parts."""

    QUALIFIED_MODELS = ("mobile-fr3-duo@1", "mobile-fr3-single@1")

    def __init__(self, *, robot="mobile-fr3-duo", task=None, seed=0, moisture=None, topsoil=None, stones=None, clutch=8,
                 reflex_profile="contact_x5", cameras=None, output_dir=None, renderer_options=None, recording=True,
                 device=None, reflex_stops_base=True, policy_samples=None, preset=None, **unknown):
        from soil_excavation.config import Config
        from soil_excavation.env import SoilEnv
        from soil_excavation.variants import resolve
        from soil_excavation.robot_setup import check_task
        from classroom_arrangement.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["soil-excavation"]
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown soil-excavation options {sorted(unknown)}; options: robot, task, seed, moisture, topsoil, stones, "
                "clutch, reflex_profile, cameras, renderer_options, recording, device, reflex_stops_base, policy_samples")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if policy_samples is not None and (type(policy_samples) is not int or not 1 <= policy_samples <= 256):
            raise UnsupportedConfigurationError("policy_samples must be an integer 1..256")
        if device not in (None, "cuda", "cuda:0"):
            raise UnsupportedConfigurationError(f"soil-excavation runs MuJoCo on the CPU and the soil on cuda:0; device={device!r} "
                                                "is not supported")
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"soil-excavation does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(catalog['aliases'])}")
        try:
            v = resolve(task, int(seed), moisture, topsoil, stones)
            if preset is not None and preset != v["preset"]:
                raise ValueError(f"task {v['task']!r} belongs to preset {v['preset']!r}, not {preset!r}")
            check_task(robot, v["task"])
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        extra = {} if policy_samples is None else dict(policy_samples=int(policy_samples))
        try:
            self.config = Config(task=v["task"], seed=v["seed"], moisture=v["moisture"], topsoil=v["topsoil"],
                                 stones=v["stones"], clutch=clutch, robot=robot, reflex_profile=reflex_profile,
                                 reflex_stops_base=reflex_stops_base, **extra)
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("soil-excavation session (auto-created; pruned by retention)\n")
        self.env = SoilEnv(self.config, output_dir=self.output)
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
            from classroom_arrangement.trace import Trace
            self.trace = Trace(w, self.output/"physics.sqlite3")
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "base_twist", "keyboard"})
        from ..keyboard_control import SUPPORTED_KEYS
        self.keyboard_keys = tuple(sorted(set(SUPPORTED_KEYS) | set(BASE_KEYS)))
        self.capabilities = Capabilities(
            "soil-excavation", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())

    @staticmethod
    def _min_action_s(n_cameras, width, height):
        """Shortest legal command: 0.1 s for one 640x360 camera or none, scaled with the rendered pixels per
        observation, in whole 2 ms soil steps. Each observation rebuilds the soil surface from 0.65 M particles when
        any moved. This amortizes observation cost; it cannot guarantee a wall/sim ratio on a shared GPU."""
        load = max(1., n_cameras*width*height/(640*360))
        return round(math.ceil(0.1*load/0.002 - 1e-9)*0.002, 6)      # 0.10000000000000002 refused a 0.1 s chord

    def _notes(self):
        notes = dict(super()._notes())
        c = self.config
        notes.update(
            task=f"{c.preset}/{c.task} seed {c.seed} ({c.moisture}): {self.env.spec['instruction']}",
            clock="One tick = 0.5 ms (MuJoCo rigid step); the soil steps every 4 ticks (2 ms, GPU implicit MPM); "
                  "joint-target actions give ticks, others duration_s",
            soil="The ground is a two-way coupled granular continuum (Newton implicit MPM, Drucker-Prager with tension "
                 "cut-off, cohesion that is lost when soil loosens and regained when it is compacted, 2 cm voxels): it "
                 "is cut, pried, lifted, spilled, thrown, piled, poured and tamped only through the contacts of the "
                 "tools, stones, root ball, post and the robot itself. Nothing about it is commanded directly.",
            start="the task's tool starts in a valid grasp (" + ", ".join(f"{a}: {r['tool']}" for a, r in self.env.hold_record.items())
                  + "); the Spine lowers both arms together (the robot's vertical push)",
            qualification="Assisted fixtures are not policy actions")
        notes["min_command_s"] = (f"every command lasts at least {self.min_action_s:g} s ({len(self.cameras)} camera(s) at "
                                  f"{self.options.get('width', 640)}x{self.options.get('height', 360)}) to amortize "
                                  "observation cost; runtime depends on GPU contention and is not guaranteed below 200x")
        return notes

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        from soil_excavation.variants import resolve
        options = dict(options or {})
        unknown = set(options) - set(SELECTORS)
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are {', '.join(SELECTORS)}; nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            cur = self._variation()
            merged = dict(cur, **options)
            try:
                want = resolve(merged["task"], int(merged["seed"]), merged["moisture"], merged["topsoil"], merged["stones"])
                from soil_excavation.robot_setup import check_task
                check_task(self.config.robot, want["task"])
            except (ValueError, TypeError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            want = {k: want[k] for k in SELECTORS}
            if want != cur:
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def _variation(self):
        c = self.config
        return dict(task=c.task, seed=c.seed, moisture=c.moisture, topsoil=c.topsoil, stones=c.stones)

    def _rebuild(self, v):
        from dataclasses import replace
        from soil_excavation.env import SoilEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            self._release()
            self.config = replace(self.config, **v)
            old = self.env
            old.world.soil = None                         # free the GPU soil before the new one is built
            self.env = SoilEnv(self.config, output_dir=self.output)
            self.ik = ArmKinematics(self.env.world)
            self._scene_path = None
            self._checkpoints = {}
            self._geometry = None
            self.env.generation = old.generation + 1
            from dataclasses import replace as _replace
            self.capabilities = _replace(self.capabilities, notes=self._notes())
            if self.recording:
                from classroom_arrangement.trace import Trace
                self.trace = Trace(self.env.world, self.output/f"physics-{self.env.generation}.sqlite3")

    def variation_contract(self):
        from soil_excavation.variants import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        from soil_excavation.variants import resolve
        from soil_excavation.layout import TASKS
        if set(request) - set(SELECTORS):
            raise ValueError(f"Selectors are {', '.join(SELECTORS)}")
        cur = dict(self._variation(), **request)
        v = resolve(cur["task"], int(cur["seed"]), cur["moisture"], cur["topsoil"], cur["stones"])
        return dict(variation=v, task=TASKS[v["task"]]["instruction"],
                    reset_request=dict(seed=v["seed"], options={k: v[k] for k in SELECTORS if k != "seed"}))

    def evaluate(self):
        """Verification (soil_excavation/task.py): criteria read from the particles, bodies, contacts and logs."""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            from soil_excavation.task import VerdictUnavailable
            try:
                ev = self.env.evaluator.evaluate()
            except VerdictUnavailable as exc:              # an interruption, not a failed task
                raise BackendProtocolError(str(exc)) from exc
            return dict(ev, success=bool(ev["succeeded"]), score=float(ev["score"]))

    # ------------------------------------------------------------------ rendering
    def _scene(self):
        from soil_excavation.render import export
        if self._scene_path is None or not self._scene_path.is_file():
            self._scene_path, self.render_meta = export(self.env, self.output/"render")
        return self._scene_path

    def _worker(self):
        from soil_excavation.render import Renderer
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
                result = r.observe(self.env, tuple(names), cell=self.options.get("soil_cell_m"))
                self.last_timing["frame_s"] = time.perf_counter() - t0
                self.last_timing.update(r.last_timings)
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
        name = "yard" if label in ("interactive", "http") else label
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
                             "client_timeout_s": CLIENT_TIMEOUT_S})

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
            raise UnsupportedConfigurationError("soil-excavation qualifies only the Mobile FR3 Duo / single-arm configurations; "
                                                f"this setup needs: {'; '.join(problems)}. Aliases: {', '.join(catalog['aliases'])}")
        return d

    def get_geometry_index(self):
        """Privileged geometry: every shape and body, the tools with their parts and poses, the soil block, its layers
        and materials, the marks, the tarp and the task's targets."""
        w, sp = self.env.world, self.env.spec
        q = w.state.body_q.numpy()
        index = dict(self._catalog().index())
        index.pop("pen_frame", None)
        index.update(
            items={n: dict(kind=it["kind"], body_id=int(it["body"]), pose_xyzw=[float(v) for v in q[it["body"]]],
                           shape_ids=[str(s) for s in it["shapes"]]) for n, it in w.scene.items.items()},
            soil=dict(block=sp["block"], layers=sp["layers"], voxel_m=sp["voxel"], moisture=sp["moisture"],
                      particles=int(w.soil.n), materials={k: {kk: vv for kk, vv in v.items() if kk not in ("color", "basis")}
                                                          for k, v in __import__("soil_excavation.params").params.MATERIALS.items()}),
            marks=sp["marks"], tarp=sp["tarp"], targets={k: sp[k] for k in ("target", "targets", "target_line", "target_square", "pipe")
                                                         if k in sp},
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

    def close(self):
        try:
            super().close()
        finally:
            env = getattr(self, "env", None)
            if env is not None:
                env.world.soil = None

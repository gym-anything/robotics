"""Catalogue adapter; the original backend is left unchanged.

Construction and rebuild mirror that backend, changing only task config/spec and
verifier construction. Robot controls, render lifecycle and option coercion are
inherited unchanged. No process-global monkeypatches or task registration.
"""
import math
from collections import deque
from pathlib import Path
import threading

from .newton_surface_rinsing import (
    NewtonSurfaceRinsingBackend as LegacyBackend, _session_folder, SESSION_MARKER,
    POLICY_CAMERAS, BASE_CAMERAS, BASE_KEYS, CameraSpec, Capabilities,
    UnsupportedConfigurationError, UnsupportedActionError,
)
from surface_rinsing.task_set.catalogue import resolve


class NewtonSurfaceRinsingBackend(LegacyBackend):
    def __init__(self, *, robot="mobile-fr3-duo", preset=None, task=None, seed=0, reflex_profile="contact_x5", cameras=None,
                 output_dir=None, renderer_options=None, recording=True, device=None, reflex_stops_base=True,
                 policy_samples=None, contract=None, **unknown):
        from surface_rinsing.task_set.runtime import Config
        from surface_rinsing.task_set.runtime import RinsingEnv
        from surface_rinsing.task_set.catalogue import resolve
        from classroom_arrangement.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["surface-rinsing"]
        if unknown:
            raise UnsupportedConfigurationError(
                f"Unknown surface-rinsing options {sorted(unknown)}; options: robot, preset, task, seed, reflex_profile, "
                "cameras, renderer_options, recording, device, reflex_stops_base, policy_samples, contract")
        if not isinstance(reflex_stops_base, bool):
            raise UnsupportedConfigurationError("reflex_stops_base must be true or false")
        if policy_samples is not None and (type(policy_samples) is not int or not 1 <= policy_samples <= 256):
            raise UnsupportedConfigurationError("policy_samples must be an integer 1..256")
        if device not in (None, "cpu"):
            raise UnsupportedConfigurationError(f"surface-rinsing runs MuJoCo on the CPU and the water on cuda:0; "
                                                f"device={device!r} is not supported (the renderer picks its own GPU)")
        if not isinstance(robot, str):
            robot = self._qualified_setup(robot, catalog)
        elif robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"surface-rinsing does not support {robot!r}: {catalog['unsupported'][robot]}")
        if isinstance(robot, str) and robot not in catalog["aliases"]:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(catalog['aliases'])}")
        try:
            v = resolve(preset, task, int(seed), contract)
        except (ValueError, TypeError) as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        from surface_rinsing.robot_setup import check_task
        try:
            check_task(robot, v["task"])
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        extra = {} if policy_samples is None else dict(policy_samples=int(policy_samples))
        self.config = Config(task=v["task"], seed=v["seed"], robot=robot, reflex_profile=reflex_profile,
                             reflex_stops_base=reflex_stops_base, task_contract=v.get("contract", "legacy"), **extra)
        self.robot_name = robot if isinstance(robot, str) else robot["id"]
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("surface-rinsing session (auto-created; pruned by retention)\n")
        self.env = RinsingEnv(self.config, output_dir=self.output)
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
            from surface_rinsing.trace import Trace
            self.trace = Trace(w, self.output/"physics.sqlite3")
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "base_twist", "keyboard"})
        from ..keyboard_control import SUPPORTED_KEYS
        self.keyboard_keys = tuple(sorted(set(SUPPORTED_KEYS) | set(BASE_KEYS)))
        self.capabilities = Capabilities(
            "surface-rinsing", "newton", robot, self.kinds, frozenset({"proprioception", *(["rgb"] if self.cameras else [])}),
            True, True, True, self.cameras, tuple(self.camera_specs[c] for c in self.cameras), keyboard_keys=self.keyboard_keys,
            notes=self._notes())

    def _rebuild(self, v):
        from dataclasses import replace
        from surface_rinsing.task_set.runtime import RinsingEnv
        from classroom_arrangement.kinematics import ArmKinematics
        with self.env.lock:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            self._release()
            self.config = replace(self.config, task=v["task"], seed=int(v["seed"]), task_contract=v.get("contract", "legacy"))
            old = self.env
            self.env = RinsingEnv(self.config, output_dir=self.output)
            self.ik = ArmKinematics(self.env.world)
            self._scene_path = None
            self._checkpoints = {}
            self._geometry = None
            self.env.generation = old.generation + 1
            from dataclasses import replace as _replace
            self.capabilities = _replace(self.capabilities, notes=self._notes())
            if self.recording:
                from surface_rinsing.trace import Trace
                self.trace = Trace(self.env.world, self.output/f"physics-{self.env.generation}.sqlite3")

    def _variation(self):
        c = self.config
        return resolve(c.preset, c.task, c.seed, c.task_contract)

    @staticmethod
    def _requested(request, cur):
        task = request.get("task", cur["task"])
        selected_contract = request.get("contract", cur.get("contract") if task == cur["task"] else None)
        # Keep the backend's int(seed) coercion, including numeric strings/bools.
        return resolve(request.get("preset", cur["preset"] if "task" not in request else None), task,
                       int(request.get("seed", cur["seed"])), selected_contract)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        unknown = set(options) - {"preset", "task", "seed", "contract"}
        if unknown:
            raise UnsupportedActionError(f"Unknown reset options {sorted(unknown)}; selectors are preset, task, seed, contract; nothing changed")
        if seed is not None:
            options["seed"] = seed
        if options:
            try:
                want = self._requested(options, self._variation())
            except (ValueError, TypeError) as exc:
                raise UnsupportedActionError(f"Reset refused: {exc}; nothing changed") from exc
            if want != self._variation():
                self._rebuild(want)
                return self.env.get_state(), {"continuity": "reset", "variation": self._variation(), "rebuilt": True}
        return self.env.reset(), {"continuity": "reset", "variation": self._variation()}

    def variation_contract(self):
        from surface_rinsing.task_set.catalogue import contract
        return dict(**contract(), current=self._variation())

    def select_variation(self, request):
        from surface_rinsing.task_set.catalogue import instruction
        if set(request) - {"preset", "task", "seed", "contract"}:
            raise ValueError("Selectors are preset, task, seed and contract")
        v = self._requested(request, self._variation())
        # Explicit legacy on reset matters when switching a revised task back.
        opts = dict(preset=v["preset"], task=v["task"], contract=v.get("contract", "legacy"))
        return dict(variation=v, task=instruction(v), reset_request=dict(seed=v["seed"], options=opts))

    def evaluate(self):
        out = super().evaluate()
        # Preserve the pan's exact scoring/checks and its already reported
        # reward.contract; add provenance to legacy fraction-of-check results.
        out["contract"] = out.get("reward", {}).get("contract", "legacy-mean-of-checks-v1")
        return out

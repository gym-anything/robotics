"""Label application bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 + label solver)."""

from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from .base import RoboticsBackend, reject_unexecuted
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta
from ..errors import (ActionRejectedError, UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError,
                      BackendUnavailableError)
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

POLICY_CAMERAS = ("bench_rgb", "left_wrist_rgb", "right_wrist_rgb")
WRIST_CAMERA_TASKS = ("part-id-match", "tube-wrap", "gs1-case-side",     # wrist cameras on by default (see __init__)
                      "smooth-corner", "finish-securing", "tube-label")
DIAGNOSTIC = ("room", "printer", "closeup", "scanner", "carton_top", "carton_side", "tube", "part")


SESSION_MARKER = ".gar-label-session"
SESSION_RETENTION = 20


def _session_folder():
    """Auto-created session folder: $GAR_WORKSPACE/.gar/label-application-sessions/<uuid> (the workspace's
    managed, git-ignored session area; else the system temp
    directory), never inside the installed package. Keeps the SESSION_RETENTION newest auto-created sessions;
    only folders carrying SESSION_MARKER are ever pruned."""
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"label-application-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


class NewtonLabelApplicationBackend(RoboticsBackend):
    def __init__(self, *, robot="dual-fr3", preset="parcel", task=None, seed=0, device=None, config=None,
                 cameras=None, output_dir=None, renderer_options=None, recording=True):
        from label_application.config import Config, TASKS
        from label_application.env import LabelEnv
        from label_application.kinematics import ArmKinematics
        from robots.catalog import ENVIRONMENTS
        catalog = ENVIRONMENTS["label-application"]
        qualified = tuple(catalog["aliases"])                  # the catalog, not a list kept here
        if isinstance(robot, dict):
            raise UnsupportedConfigurationError(f"label-application supports catalog aliases {', '.join(qualified)}")
        if robot in catalog.get("unsupported", {}):
            raise UnsupportedConfigurationError(f"label-application does not support {robot!r}: {catalog['unsupported'][robot]}")
        if robot not in qualified:
            raise UnsupportedConfigurationError(f"Unsupported robot {robot!r}; qualified: {', '.join(qualified)}")
        values = dict(config or {})
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        if device is not None:
            values["device"] = device
        self.config = Config(**values)
        if cameras is None:
            # per-task default: the overhead camera, plus the wrist cameras where the overhead view cannot show what the
            # task asks the robot to look at: part-id-match, the plates' laser-marked part numbers and the label's
            # printed one (~3 px per character overhead); tube-wrap, the manufacturer label's edge for 0-2 mm
            # registration (the 13 mm tube is ~11 px across overhead); gs1-case-side, the long side faces (the
            # robot-facing one is not visible overhead at all); tube-label as tube-wrap; smooth-corner and
            # finish-securing, the close view of the lifted areas from the ready pose's hand above the label (a 4 x 6 in
            # label is ~90 x 60 px overhead)
            cameras = ("bench_rgb",)
            if self.config.task in WRIST_CAMERA_TASKS:
                cameras += ("left_wrist_rgb",) if robot == "single-fr3" else ("left_wrist_rgb", "right_wrist_rgb")
        self.cameras = tuple(cameras)
        self.last_timing = {}                                  # per-operation timing breakdown
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; others are privileged diagnostics")
        if robot == "single-fr3" and "right_wrist_rgb" in self.cameras:
            raise UnsupportedConfigurationError("single-fr3 has one arm, on the left pedestal beside the printer: "
                                                "policy cameras are bench_rgb, left_wrist_rgb")
        self.options = dict(renderer_options or {})
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("label-application session (auto-created; pruned by retention)\n")
        self.env = LabelEnv(self.config, output_dir=self.output)
        self.ik = ArmKinematics(self.env.world)
        self.arms = tuple(self.env.world.rig.instances)
        self.images, self.trace = deque(), None
        if recording and self.config.record_trace:
            from label_application.trace import Trace
            self.trace = Trace(self.env, self.output/"physics.sqlite3")
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        self.camera_specs = {}
        from label_application.render import cameras as fixed_cameras
        bench_focal_mm = fixed_cameras(self.env.world)["bench_rgb"][2]
        mounted = set(self.env.world.scene.cameras)          # wrist cameras exist only on built arms
        for name in POLICY_CAMERAS:
            if "wrist" in name and name not in mounted:
                continue
            fov = 87. if "wrist" in name else 2*math.degrees(math.atan(18/bench_focal_mm))
            focal = .018/math.tan(math.radians(fov/2))
            self.camera_specs[name] = CameraSpec(name, "link7_d405_envelope" if "wrist" in name else "external_bench_mount",
                                                 width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities(
            "label-application", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": f"One tick = {self.config.label_dt*1e3:.1f} ms (label step) = {self.config.label_substeps} rigid MuJoCo ticks",
                "control": "Physical position servos; IK computes targets only (not collision-free planning)",
                "arms": ",".join(self.arms),
                "gripper": "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode grasp (total N of both fingers, <= 70 N continuous = 2 x 35 N finger limit; the pad-to-pad squeeze on a held object is half the total)",
                "label_physics": "Orthotropic shell + pressure-sensitive adhesive interface",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions"})

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.config.seed):
            raise ValueError("Select preset/task/seed when creating the session; reset restores that initial state")
        if self.trace and self.trace.current:
            raise RuntimeError("Finish the current recorded command before resetting")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            optical = self._optical()
            return self.env.evaluator.evaluate(optical=optical)

    # ------------------------------------------------------------------ render workers (one per operation)
    # Requirement: render-only workers are released when their operation (one observation, one evaluation, one
    # diagnostic render) ends, including on errors; none is kept idle between operations. Speed within that rule (the
    # ~11 s worker start and ~4 s first frame dominate an observation): an action's worker is started and warmed on a
    # snapshot of the pre-action state WHILE the action's physics runs, consumed by that action's observation and
    # released right after capture on a reaper thread (off the reply's path). The static stage is exported once per
    # world and reused; every frame still sends the live body poses and label mesh.

    def _scene(self):
        """The exported static stage of the current world (re-exported after a reset)."""
        from label_application.render import export
        key = id(self.env.world)
        if getattr(self, "_scene_key", None) != key or not self._scene_path.is_file():
            self._scene_path = export(self.env.world, self.output/"render", self.env.art)
            self._scene_key = key
        return self._scene_path

    def _new_renderer(self):
        from label_application.render import Renderer
        return Renderer(self.env.world, self.output/"render", self.env.art, scene=self._scene(),
                        width=self.options.get("width", 1280), height=self.options.get("height", 720),
                        samples=self.options.get("samples", 24))

    def _start_renderer_early(self):
        """Start (and warm) this operation's render worker while its physics runs. At most one; consumed
        by the operation's observation, or released if the action fails."""
        if not self.cameras or getattr(self, "_prestart", None) is not None:
            return
        from label_application.render import label_sheets
        scene = self._scene()                              # in this thread: export touches the world
        front, back, n = label_sheets(self.env.world)
        snap = dict(tick=self.env.world.tick, body_q=self.env.world.state.body_q.numpy().copy(),
                    front=front, back=back, n=n)
        holder = {"renderer": None, "error": None, "warm_s": None}

        def work():
            try:
                t0 = time.perf_counter()
                from label_application.render import Renderer
                holder["renderer"] = Renderer(self.env.world, self.output/"render", self.env.art, scene=scene,
                                              width=self.options.get("width", 1280),
                                              height=self.options.get("height", 720),
                                              samples=self.options.get("samples", 24))
                holder["renderer"].observe(tuple(self.cameras), state=snap)     # warm-up frame, discarded
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:                       # reported when consumed
                holder["error"] = exc
        holder["thread"] = threading.Thread(target=work, name="label-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _take_prestart(self):
        holder, self._prestart = getattr(self, "_prestart", None), None
        if holder is None:
            return None, None
        t0 = time.perf_counter()
        holder["thread"].join()
        # per-action timing: how long the observation waited
        # for the early-started worker after the physics ended, and the worker's start + warm-up time
        self.last_timing.update(prestart_wait_s=time.perf_counter() - t0, prestart_warm_s=holder["warm_s"])
        if holder["error"] is not None or holder["renderer"] is None:
            if holder["renderer"] is not None:
                self._release(holder["renderer"])
            return None, holder["error"]
        return holder["renderer"], None

    def _release(self, renderer, wait=False):
        """Release a worker now; the reply does not wait for Kit's shutdown unless `wait`."""
        self._reapers = [t for t in getattr(self, "_reapers", []) if t.is_alive()]
        if wait:
            renderer.close()
            return
        t = threading.Thread(target=renderer.close, name="label-render-release", daemon=False)
        t.start()
        self._reapers.append(t)

    def _operation_renderer(self):
        """The worker for this operation: the early-started one if any, else a new one."""
        renderer, _ = self._take_prestart()
        if renderer is not None:
            return renderer
        t0 = time.perf_counter()
        renderer = self._new_renderer()
        self.last_timing["cold_start_s"] = time.perf_counter() - t0
        return renderer

    def _verification_renderer(self):
        """The verification render runs at its own fixed settings (optical.VERIFY_RENDER), whatever the session's
        policy-camera options: the verdict must not change with the agent's camera resolution or samples. The
        early-started worker is reused only when its settings are the same."""
        from label_application import optical
        from label_application.render import Renderer
        session = dict(width=self.options.get("width", 1280), height=self.options.get("height", 720),
                       samples=self.options.get("samples", 24))
        if session == optical.VERIFY_RENDER:
            return self._operation_renderer()
        renderer, _ = self._take_prestart()
        if renderer is not None:
            self._release(renderer)
        return Renderer(self.env.world, self.output/"render", self.env.art, scene=self._scene(), **optical.VERIFY_RENDER)

    def _optical(self):
        """Render the task's reader view(s) from the current state and decode with zxing-cpp. No declared reader
        frames the symbol: no read can count, nothing is rendered. The renderer cannot run: no verdict
        (BackendUnavailableError), never a task failure."""
        from label_application import optical
        world, task = self.env.world, self.env.world.config.task
        tag = f"{self.env.generation}_{self.env.world.steps}"
        if not any(r[3] for r in optical.framed_reads(world, self.env.art, task)):
            renderer, _ = self._take_prestart()             # this operation needs no worker
            if renderer is not None:
                self._release(renderer)
            res = optical.read(None, self.env.art, task, self.output, tag, world=world, diagnostic=False)
            res["note"] = "no declared reader frames the symbol with its quiet zones; nothing rendered"
            return res
        try:
            r = self._verification_renderer()
        except Exception as exc:
            raise BackendUnavailableError(f"Verification renderer could not start; no verdict: {exc}") from exc
        try:
            return optical.read(r, self.env.art, task, self.output, tag, world=world)
        except optical.VerificationUnavailable as exc:
            raise BackendUnavailableError(f"Verification render failed; no verdict: {exc}") from exc
        finally:
            self._release(r)

    # ------------------------------------------------------------------ observation
    def _render(self, names):
        """RGB observation from one per-operation render worker (see the render-worker section above):
        early-started during the action's physics when there was an action, released after capture."""
        with self.env.lock:
            r = self._operation_renderer()
            try:
                t0 = time.perf_counter()
                result = r.observe(tuple(names))
                self.last_render_s = time.perf_counter() - t0
                self.last_timing["frame_s"] = self.last_render_s
            finally:
                self._release(r)
            frames = {}
            for name, blob in result.items():
                path = self.output/f"{name}_{self.env.generation}_{self.env.world.steps}_{uuid.uuid4().hex}.png"
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
            if (state["tick"], state["generation"]) != (self.env.world.steps, self.env.generation):
                raise RuntimeError("Cannot combine stale state and a new camera image")
            frames = self._render(self.cameras) if self.cameras else {}
            return Observation(sequence, state["time_s"], self.config.robot, {"joints": state["joints"]},
                               {n: ImageFrame(n, p, True, camera=self.camera_specs[n]) for n, p in frames.items()})

    # ------------------------------------------------------------------ actions
    def _ticks(self, seconds):
        ticks = seconds/self.config.label_dt
        if not math.isclose(ticks, round(ticks), abs_tol=1e-8, rel_tol=0) or round(ticks) < 1:
            raise ValueError(f"Duration must be a positive multiple of {self.config.label_dt} s")
        return int(round(ticks))

    def _arm(self, arm):
        if arm not in self.arms:
            raise UnsupportedActionError(f"Arm must be one of {self.arms}")
        return arm

    def _fingers(self, arm, width):
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 0. <= width <= .04:
            raise ValueError("Franka Hand half gap must be in [0, 0.04] m")
        return {j: float(width) for j in self.env.world.gripper_joints(arm)}

    def act(self, action):
        # A rejected command (invalid target, grip force or duration) cancels the command, not the session.
        w = self.env.world
        return reject_unexecuted(lambda: self._act(action), lambda: (
            w.steps, w.targets.copy(), dict(w.grip_force), sorted(w.grasp_latched)))

    def _refuse_after_episode_end(self):
        """A tear ends the episode (world.label_frozen, config.stop_on_tear; REPORT §14): the command that tore
        commits as usual (`terminated` in the owner's step result), and every later one is refused before any target
        write or tick, as locomotive_driving refuses commands once its episode is over. The gateway withholds
        `terminated` from policy access, so the refusal carries the reason (policy_reason, http_server.py _error):
        the tear is a physical event in the scene, not evaluator state. A reset or a restored pre-tear checkpoint
        (owner key) clears it."""
        ended = self.env.world.label_frozen
        if ended is None:
            return
        text = (f"episode over: the label tore at tick {ended['tick']} (t = {ended['time_s']:.3f} s), which ends the "
                f"episode; no further command runs")
        err = ActionRejectedError(text)
        err.policy_reason = text
        raise err

    def _act(self, action):
        with self.env.lock:
            self._refuse_after_episode_end()
            if isinstance(action, NamedJointTarget):
                targets, ticks = dict(action.targets_si), action.ticks
            elif isinstance(action, Gripper):
                arm = self._arm(action.arm)
                ticks = self._ticks(action.duration_s)
                if action.inward_effort_n is not None:
                    # Franka Hand grasp semantics: total inward force, split over the two fingers,
                    # fingers servo toward closed; 0 N returns to pure position control.
                    total = float(action.inward_effort_n)
                    targets = self._fingers(arm, 0. if total > 0 else (action.half_width_m if action.half_width_m is not None else .04))
                    self.env.world.set_grip_force(arm, total/2.)
                else:
                    targets = self._fingers(arm, action.half_width_m)
                    self.env.world.set_grip_force(arm, 0.)
            elif isinstance(action, (CartesianPose, CartesianDelta)):
                arm = self._arm(action.arm)
                if action.safety.stop_on_grip_loss is not None:
                    raise UnsupportedActionError("Grip-loss guard is not implemented")
                ticks = self._ticks(action.duration_s)
                if isinstance(action, CartesianPose):
                    position = action.position_m
                    o = action.orientation_wxyz
                    orientation = None if o is None else (*o[1:], o[0])
                else:
                    position = self.ik.forward(arm)[:3, 3] + np.asarray(action.delta_m)
                    orientation = None
                targets = self.ik.solve(arm, position, orientation)
                if action.gripper_half_target_m is not None:
                    targets.update(self._fingers(arm, action.gripper_half_target_m))
            else:
                raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta")
            before = self.env.world.steps
            n_reflex = len(self.env.world.reflexes)
            self.last_timing = {}
            self._start_renderer_early()
            try:
                t0 = time.perf_counter()
                state = self.env.act(targets, ticks=ticks)
                self.last_timing["physics_s"] = time.perf_counter() - t0
            except BaseException as exc:
                renderer, _ = self._take_prestart()          # the operation failed: release its worker
                if renderer is not None:
                    self._release(renderer)
                if isinstance(exc, (RuntimeError, FloatingPointError, OSError)):
                    raise BackendProtocolError(f"Stopped at tick {self.env.world.steps}: {exc}") from exc
                raise
            ev = state["evaluation"]
            reflexes = self.env.world.reflexes[n_reflex:]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.world.steps - before,
                                       "velocity_reflexes": [dict(r) for r in reflexes],
                                       "note": "success here excludes the optical decode; use evaluate/gar verify",
                                       "termination": state["termination"], "label_solver": state["label_solver"]},
                               ev["score"], bool(state["terminated"]))

    def policy_access(self):
        from label_application.task import INSTRUCTIONS
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras),
                            {("joints", n, f): None for n in self.env.joints for f in ("position", "velocity")},
                            self.kinds, lambda: INSTRUCTIONS[self.config.task],
                            {"dt_s": self.config.label_dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in self.env.joints.items()}})

    def act_policy(self, action):
        result = self.act(action)
        return BackendStep(result.state, result.info, None, result.terminated, result.truncated)

    def render(self, *, label="interactive"):
        name = "room" if label in ("interactive", "http") else label
        if name not in POLICY_CAMERAS + DIAGNOSTIC:
            raise ValueError(f"Render label must be one of {POLICY_CAMERAS + DIAGNOSTIC}")
        return self._render((name,))[name]

    # ------------------------------------------------------------------ geometry, checkpoints, trace
    def get_geometry_index(self):
        w = self.env.world
        return {"items": {n: {"body": i["body"], **{k: v for k, v in i.items() if k != "body"}} for n, i in w.scene.items.items()},
                "label": dict(format=w.scene.label_spec["fmt"], stock=w.label.stock.name, length_md_m=w.label.length,
                              width_cd_m=w.label.width, vertices=w.label.n, triangles=len(w.label.tris)),
                "colliders": [dict(name=c.name, shape=c.shape, size=list(c.size), surface=c.surface.name, role=c.role)
                              for c in w.label.colliders]}

    def get_geometry(self, geometry_id):
        w = self.env.world
        if geometry_id == "label":
            return {"vertices_m": w.label.x.tolist(), "triangles": w.label.tris.tolist(), "bond": w.label.bond.tolist(),
                    "phi": w.label.phi.tolist(), "rest_uv_m": w.label.X.tolist()}
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints = getattr(self, "_checkpoints", {})
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.world.steps, "generation": self.env.generation}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = getattr(self, "_checkpoints", {}).get(label)
            if snap is None:
                raise KeyError(label)
            self.env.restore(snap)
            self.env.generation += 1
            # backend contract: (state, info); checkpoints are in-memory for the life of the session
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.steps,
                                          "generation": self.env.generation, "continuity": "restored_discontinuity",
                                          "storage": "in_memory_session"}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["steps"]} for k, v in getattr(self, "_checkpoints", {}).items())   # env ticks = label steps

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def close(self):
        with self.env.lock:
            renderer, _ = self._take_prestart()
            if renderer is not None:
                self._release(renderer, wait=True)
            for t in getattr(self, "_reapers", []):
                t.join()
            self._reapers = []
        if self.trace:
            self.trace.close()
            self.trace = None
        self.env.close()

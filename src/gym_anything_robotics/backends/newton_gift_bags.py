"""Gift-bags bridge to GAR's authenticated HTTP/local protocol (Newton 1.5 MuJoCo robots + the gift_bags
implicit contact world for paper, tape and packaging items).

Commands are planned and validated completely (duration, arm, IK, joint limits, grip width/force) before anything is
written to the robot; a refused command raises ActionRejectedError (UnsupportedActionError for an unsupported kind)
and leaves the session usable.

Render workers: one Isaac render worker per session serves every observation and diagnostic render. It is started
while the first action's physics runs (or on the first image), kept between operations and released after
renderer_options idle_release_s seconds without an image request (default IDLE_RELEASE_S = 15 min, or the environment
variable GIFT_BAGS_RENDER_IDLE_S for sessions started by `gar run`, which has no renderer options; 0 releases it after
every operation) and at close(). After a release the next image pays a worker start (10-11 s measured). A worker that has exited or fails a frame is replaced and the frame retried once. The
worker watches its owner process (gym_anything_robotics.render_lifecycle).

Each command sent through the HTTP gateway records a per-contact-step physics trace (StepTrace below) in GAR's
PhysicsTraceStore format (GET /v1/history/{id}/trace, gar export --include-traces).
"""

from collections import deque
import math
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from .base import RoboticsBackend
from ..actions import NamedJointTarget, Gripper, CartesianPose, CartesianDelta
from ..errors import ActionRejectedError, UnsupportedActionError, UnsupportedConfigurationError, BackendProtocolError
from ..policy_access import PolicyAccess
from ..types import BackendStep, CameraSpec, Capabilities, ImageFrame, Observation

POLICY_CAMERAS = ("overhead_rgb", "left_wrist_rgb", "right_wrist_rgb", "head_rgb")   # head_rgb: G1 only (torso D435)
DIAGNOSTIC = ("room", "front", "counter_wide", "bag", "mat", "gift", "side", "carton", "carton_end")
SESSION_MARKER = ".gar-gift-bags-session"
SESSION_RETENTION = 20
IDLE_RELEASE_S = 900.0     # s; a policy that thinks longer than this between actions pays a 10-11 s worker start
                           # on every action
IDLE_RELEASE_ENV = "GIFT_BAGS_RENDER_IDLE_S"
# Per-step trace bounds. A sample is written for every 4 ms contact step; sheet vertices (float32) only on a command's
# first step, every TRACE_SHEET_STRIDE steps after it and on its last step. The store keeps at most TRACE_BYTES of
# compressed payload and TRACE_ITEMS items per session and evicts the oldest items first (each trace summary reports
# what it lost), so a long command or a long session cannot grow the trace or its export without bound. Measured
# (free air): a sample is 0.9-1.1 kB compressed / 2.6-3.3 kB exported JSON, a sample with the sheets
# 11 kB / 112 kB (counter, 1,731 vertices) and 22 kB / 150 kB (unwrap, 2,286 vertices): about 0.4 MB stored and
# 2 MB exported per simulated second on the paper presets, so the store holds the latest ~160 simulated seconds and an
# export with traces stays below ~350 MB.
TRACE_BYTES = 64*1024*1024
TRACE_ITEMS = 100_000
TRACE_SHEET_STRIDE = 25
# bag-purchase's evaluation completes the 3 s hands-off window itself (gift_bags/env.py final_evaluation): told to the
# owner (capabilities note) and to the policy (hardware), and a client timeout that covers it. The counter's paper runs
# at 35-127x wall/sim at rest and above in contact, so up to 3 s of branch is minutes; 1800 s also covers the 3 s at
# 600x.
SETTLE_NOTE = ("The result is judged from the simulator state when the episode ends. If its last 3 s were not already "
               "free of robot contact with the items of the job (the bag you took and its handles, the candle, a "
               "board taken from the pile), the evaluation first advances the world for the rest of those 3 s, both "
               "arms holding their last command and nothing else acting, judges the state it reaches, and then "
               "restores the episode's state: you do not have to wait after letting go, but an item still held, or "
               "touched in those 3 s, fails the release. Waiting hands-off yourself saves that time.")
SETTLE_CLIENT_TIMEOUT_S = 1800.


class StepTrace:
    """Privileged per-command physics trace (drilling/trace.py pattern; gym_anything_robotics.physics_trace format).

    One sample per completed contact step (Config.dt, 4 ms): time, joint positions/velocities/targets and the slewed
    servo targets, each hand's mode/force/width command, item poses, the contact solver's Newton iterations and
    convergence for that step, and the contact reactions on robot links; sheet vertices on the first step, every
    TRACE_SHEET_STRIDE steps and on the last step. Events: velocity reflexes. Recording only: no extra physics steps,
    no state written."""

    def __init__(self, env, path):
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        self.env = env
        self.store = PhysicsTraceStore(path, max_bytes=TRACE_BYTES, max_items=TRACE_ITEMS, compression="zlib",
                                       synchronous="NORMAL")
        self.current = None
        self.final_tick = None
        self.outcome = None
        self._prev = env.trace_callback
        env.trace_callback = self._on_step

    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable")
        e = self.env
        w, rig, ipc = e.world, e.world.rig, e.world.ipc
        self._joints = list(rig.joints)
        self._q = np.array([rig.joints[n]["q"] for n in self._joints])
        self._dof = np.array([rig.joints[n]["dof"] for n in self._joints])
        self._items = [(i, b.name) for i, b in enumerate(ipc.bodies) if b.kind == "dynamic"]
        key = str(context["action_id"])
        self.store.begin(key, str(context["request_id"]), int(w.steps), float(e.config.dt), dict(
            source="gift_bags/newton1.5+mujoco (robots) + implicit contact world (paper, tape, items)",
            joint_names=self._joints, item_names=[n for _, n in self._items],
            sheets={s.name: int(sl.stop - sl.start) for s, sl in zip(ipc.sheets, ipc.sheet_slices)},
            sampling=f"every completed {e.config.dt*1e3:g} ms contact step ({e.config.substeps} robot ticks); sheet "
                     f"vertices on a command's first step, every {TRACE_SHEET_STRIDE} steps and on its last step; "
                     "no extra physics",
            units=dict(time="s", joints="rad (arm) / m (fingers)", items="position m + quaternion xyzw",
                       reactions="N, contact force on the robot link", sheets="m (float32)"),
            contact_solver="newton_iterations and converged are this step's; failures are this step's increments",
            retention=f"store bound {TRACE_BYTES} bytes / {TRACE_ITEMS} items per session, oldest evicted first"))
        self.current, self.final_tick, self.outcome, self._start = key, None, None, int(w.steps)
        self._marks = self._counters()
        return dict(available=True, id=key, store_path=str(self.store.path))

    def _counters(self):
        st = self.env.world.ipc.stats
        return dict(newton=st.get("newton", 0), unconverged=st.get("unconverged", 0),
                    line_search_failures=st.get("line_search_failures", 0),
                    hessian_failures=st.get("hessian_failures", 0),
                    warm_start_rejected=st.get("warm_start_rejected", 0), reflexes=len(self.env.world.rig.reflexes))

    def _on_step(self, env):
        if self._prev:
            self._prev(env)
        if self.current is None:
            return
        from scipy.spatial.transform import Rotation
        w, rig, ipc = env.world, env.world.rig, env.world.ipc
        jq, jv = rig.state.joint_q.numpy(), rig.state.joint_qd.numpy()
        servo = rig.control.joint_target_q.numpy()
        now, m = self._counters(), self._marks
        poses = np.zeros((len(self._items), 7))
        for k, (i, _) in enumerate(self._items):
            p, R = ipc.body_pose(i)
            poses[k, :3], poses[k, 3:] = p, Rotation.from_matrix(R).as_quat()
        sample = dict(
            tick=int(w.steps), time_s=float(w.steps*env.config.dt),
            joints=dict(position=jq[self._q].astype(float), velocity=jv[self._dof].astype(float),
                        target=np.asarray(rig.targets, float)[self._q], servo_target=servo[self._q].astype(float)),
            grippers={a: dict(mode=h["mode"], force_n_per_finger=float(h["force"]),
                              half_width_target_m=None if h["width"] is None else float(h["width"]),
                              **({"closure": float(h["closure"])} if "closure" in h else {})) for a, h in rig.hand.items()},
            items=poses,
            contact_solver=dict(newton_iterations=int(now["newton"] - m["newton"]),
                                converged=now["unconverged"] == m["unconverged"],
                                line_search_failures=int(now["line_search_failures"] - m["line_search_failures"]),
                                hessian_failures=int(now["hessian_failures"] - m["hessian_failures"]),
                                warm_start_rejected=int(now["warm_start_rejected"] - m["warm_start_rejected"]),
                                collisions=int(getattr(ipc, "_last_collisions", -1))),
            reactions={ipc.bodies[b].name: np.asarray(F, float) for b, (F, _) in ipc.reactions.items()})
        if (w.steps - self._start - 1) % TRACE_SHEET_STRIDE == 0 or w.steps == self.final_tick:
            x = ipc.x_free()
            sample["sheets"] = {s.name: np.asarray(x[sl], np.float32) for s, sl in zip(ipc.sheets, ipc.sheet_slices)}
        events = [{**{k: v for k, v in r.items() if isinstance(v, (int, float, str))}, "type": "velocity_reflex"}
                  for r in rig.reflexes[m["reflexes"]:]]
        self._marks = now
        self.store.append(self.current, int(w.steps), sample, events)

    def expect(self, ticks):
        """The command about to run: its last step also records the sheets."""
        self._start = int(self.env.world.steps)
        self.final_tick = self._start + int(ticks)

    def finish(self):
        if self.current is None:
            return None
        key, self.current = self.current, None
        status, error = self.outcome or ("completed", None)
        self.store.finish(key, int(self.env.world.steps), status=status, error=error)
        s = self.store.summary(key)
        return dict(id=key, store_path=str(self.store.path), status=status, complete=s["complete"],
                    retained_samples=s["retained_samples"])

    def close(self):
        if self.env.trace_callback == self._on_step:
            self.env.trace_callback = self._prev
        self.store.close()


def _session_folder():
    """$GAR_WORKSPACE/.gar/gift-bags-sessions/<uuid> (else the system temp directory); keeps the
    SESSION_RETENTION newest auto-created sessions (only folders carrying SESSION_MARKER are pruned)."""
    import os, shutil, tempfile
    ws = os.environ.get("GAR_WORKSPACE")
    base = (Path(ws)/".gar" if ws else Path(tempfile.gettempdir()))/"gift-bags-sessions"
    base.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in base.iterdir() if (p/SESSION_MARKER).is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - (SESSION_RETENTION - 1))]:
        shutil.rmtree(p, ignore_errors=True)
    return base/uuid.uuid4().hex


class NewtonGiftBagsBackend(RoboticsBackend):
    g1 = False                              # the Unitree G1 configuration (set in __init__)

    def __init__(self, *, robot="dual-fr3", preset=None, task=None, seed=0, device=None, config=None, contract=None,
                 cameras=("overhead_rgb",), output_dir=None, renderer_options=None, recording=True):
        from gift_bags.task_set_catalog import Config, TASKS, completes_window
        from gift_bags.task_set_env import GiftBagsEnv
        from gift_bags.kinematics import ArmKinematics
        from gift_bags.variants import POLICY_PRESETS, availability, default_preset
        preset = default_preset() if preset is None else preset      # the first preset whose start state is usable
        from robots.catalog import ENVIRONMENTS
        from gift_bags.robot_setup import validate, SINGLE
        try:
            robot = validate(robot)                 # alias, rig id or gar.robot-setup.v1 content -> qualified alias
        except ValueError as exc:
            raise UnsupportedConfigurationError(str(exc)) from exc
        if preset not in POLICY_PRESETS:
            raise UnsupportedConfigurationError(f"preset must be one of {POLICY_PRESETS}")
        if not availability()[preset]["available"]:
            raise UnsupportedConfigurationError(f"the {preset} preset is unavailable: {availability()[preset]['reason']}")
        if robot in SINGLE and preset == "counter":
            raise UnsupportedConfigurationError(f"{robot} cannot reach the counter's bag stack from the left pedestal "
                                                "and opening a bag is two-handed; "
                                                "use dual-fr3 or dual-panda, or the home preset")
        if robot in SINGLE and preset == "receiving":
            raise UnsupportedConfigurationError(f"{robot} is not qualified for the receiving preset: its reach study "
                                                "covers the dual cell only")
        values = dict(config or {})
        from gift_bags import g1 as G1
        from gift_bags.scene import G1_LAYOUT
        self.g1 = robot == "g1"
        if self.g1:
            if preset not in G1_LAYOUT:
                raise UnsupportedConfigurationError(f"the G1 is laid out for the {sorted(G1_LAYOUT)} presets only "
                                                    "(gift_bags/scene.py G1_LAYOUT)")
            # the floating G1 needs a 0.5 ms MuJoCo step (g1.RIGID_DT_MAX); 8 robot ticks keep the 4 ms contact step
            values.setdefault("rigid_dt", G1.RIGID_DT_MAX)
            values.setdefault("substeps", G1.SUBSTEPS)
        values.update(preset=preset, task=task or TASKS[preset][0], seed=int(seed), robot=robot)
        if contract is not None:
            if "contract" in values and values["contract"] not in (None, contract):
                raise ValueError("Conflicting task contract selectors")
            values["contract"] = contract
        if device is not None:
            values["device"] = device
        self.config = Config(**values)
        self.cameras = tuple(cameras)
        if len(set(self.cameras)) != len(self.cameras) or set(self.cameras) - set(POLICY_CAMERAS):
            raise ValueError(f"Policy cameras are {POLICY_CAMERAS}; others are privileged diagnostics")
        self.options = dict(renderer_options or {})
        if "idle_release_s" not in self.options and os.environ.get(IDLE_RELEASE_ENV):
            try:
                self.options["idle_release_s"] = float(os.environ[IDLE_RELEASE_ENV])
            except ValueError:
                raise ValueError(f"{IDLE_RELEASE_ENV} must be a number of seconds >= 0") from None
        idle = self.options.get("idle_release_s", IDLE_RELEASE_S)
        if isinstance(idle, bool) or not isinstance(idle, (int, float)) or not math.isfinite(idle) or idle < 0:
            raise ValueError("renderer_options idle_release_s must be a finite number of seconds >= 0")
        if type(recording) is not bool:
            raise TypeError("recording must be boolean")
        self.output = Path(output_dir) if output_dir else _session_folder()
        self.output.mkdir(parents=True, exist_ok=False)
        if not output_dir:
            (self.output/SESSION_MARKER).write_text("gift-bags session (auto-created; pruned by retention)\n")
        self.env = GiftBagsEnv(self.config, output_dir=self.output)
        if self.g1:
            self.ik = G1.G1Kinematics(self.env.world.rig)
            self.arms = tuple(self.env.world.rig.arms)        # the G1's left and right arm/hand
        else:
            self.ik = ArmKinematics(self.env.world.rig)
            self.arms = tuple(self.env.world.rig.rig.instances)
        if robot in SINGLE and "right_wrist_rgb" in self.cameras:
            raise UnsupportedConfigurationError(f"{robot} has one arm: policy cameras are overhead_rgb, left_wrist_rgb")
        self.images = deque()
        self.last_timing = {}
        self._renderer, self._prestart, self._idle_timer, self._reapers, self._checkpoints = None, None, None, [], {}
        self.trace = None
        width, height = self.options.get("width", 1280), self.options.get("height", 720)
        from gift_bags.render import cameras as fixed_cameras
        overhead_focal = fixed_cameras(self.env.world)["overhead_rgb"][2]
        mounted = set(self.env.world.rig.cameras)
        self.camera_specs = {}
        for name in POLICY_CAMERAS:
            if ("wrist" in name or name == "head_rgb") and name not in mounted:
                continue
            fov = 87. if "wrist" in name else self.env.world.rig.cameras[name]["fov"] if name == "head_rgb" else \
                2*math.degrees(math.atan(18/overhead_focal))
            focal = .018/math.tan(math.radians(fov/2))
            self.camera_specs[name] = CameraSpec(name, "link7_d405_envelope" if "wrist" in name else
                                                 "g1_torso_d435_head" if name == "head_rgb" else "ceiling_drop_mount",
                                                 width, height, 1280, 720, (.036, .036*height/width), (focal, focal))
        missing = set(self.cameras) - set(self.camera_specs)
        if missing:
            raise UnsupportedConfigurationError(f"cameras not mounted on this robot: {sorted(missing)}")
        self.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
        self.capabilities = Capabilities(
            "gift-bags", "newton", robot, self.kinds,
            frozenset({"proprioception", *(["rgb"] if self.cameras else [])}), True, False, True,
            self.cameras, tuple(self.camera_specs[c] for c in self.cameras), notes={
                "task": f"{self.config.preset}/{self.config.task} seed {self.config.seed}",
                "clock": f"One tick = {self.config.dt*1e3:.1f} ms contact step = {self.config.substeps} MuJoCo robot ticks",
                "control": "Physical position servos; IK computes targets only (not collision-free planning)",
                "arms": ",".join(self.arms),
                "gripper": ("Dex3-1 hand (G1): no grasp primitive. half_width_m maps to a hand POSTURE of its 7 joint servos, "
                            "closure = 1 - half_width_m/0.04 between open (fingers straight) and closed (thumb opposed to "
                            "index and middle, gift_bags/g1.py HAND_CLOSED); what it presses with is the servos' torque "
                            "against contact. inward_effort_n (force mode) is refused; name the 7 hand joints for any "
                            "other finger motion" if self.g1 else
                            "Franka Hand: half_width_m position mode (0..40 mm per finger) or inward_effort_n force-mode grasp (total N of both fingers, <= 70 N = 2 x 35 N finger effort limit)"),
                **({"robot": "Unitree G1, floating base standing on its joint servos (no weld, no balance controller); "
                             "arms: 7-joint IK to a TCP 80 mm along each wrist-yaw link (gift_bags/g1.py); every joint "
                             "(29 body + 2 x 7 Dex3) is a named joint target; head_rgb is the torso D435"} if self.g1 else {}),
                "packaging_physics": "Implicit contact world: orthotropic paper shells with creases, tearing, tape bonds, lid air film, rigid items",
                "cameras": "Ideal pinhole RGB rendered by Isaac RTX from the simulated state; no depth or noise model",
                "qualification": "Assisted fixtures are not policy actions",
                "render_worker": f"One Isaac render worker per session, released after {float(idle):g} s without an "
                                 "image request (renderer_options idle_release_s) and at close",
                "checkpoints": "In memory for this session only (GAR has no cross-session checkpoint route)",
                "trace": (f"Per-contact-step physics trace of every HTTP command; at most {TRACE_BYTES >> 20} MiB "
                          f"compressed / {TRACE_ITEMS} items per session, oldest evicted first"
                          if recording else "recording disabled"),
                **({"evaluation": SETTLE_NOTE + f" Client timeout: {SETTLE_CLIENT_TIMEOUT_S:g} s "
                                                "(hardware.client_timeout_s; the gar CLI uses it)"}
                   if completes_window(self.config.task, self.config.contract) else {})})
        if recording:
            self.trace = StepTrace(self.env, self.output/"physics_trace.sqlite3")

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options=None):
        if options or (seed is not None and seed != self.config.seed):
            # refused before anything changes (GAR's rejected-reset contract): the session stays usable
            raise ActionRejectedError(f"gift-bags selects preset/task/seed at launch (this session: "
                                      f"{self._variation()}); reset restores that initial state. Start another session "
                                      "for another variation: gar run gift-bags --select preset=P --select task=T "
                                      "--select seed=S")
        return self.env.reset(), {"continuity": "reset"}

    def get_state(self):
        return self.env.get_state()

    def evaluate(self):
        """GiftBagsEnv.final_evaluation: for bag-purchase the hands-off window is completed on a restored settle
        branch (SETTLE_NOTE); the other tasks are judged on the state and history as they are."""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError("Cannot evaluate a faulted physics state")
            try:
                return self.env.final_evaluation()
            except RuntimeError as exc:
                raise BackendProtocolError(str(exc)) from exc

    def robot_configuration(self):
        rig = self.env.world.rig
        handles = getattr(rig.b, "_gar_robots", {}) or {}
        out = {}
        for name, handle in handles.items():
            if handle.model is not rig.model:
                handle.bind(rig.model)
            out[name] = handle.capabilities()
        if not out:
            return {}
        result = {"instances": out}
        setup = getattr(rig, "setup", None)       # the resolved robot setup
        if setup is not None and hasattr(setup, "id"):
            result.update(setup_id=setup.id, setup_fingerprint=getattr(setup, "fingerprint", None), alias=self.config.robot)
        return result

    # ------------------------------------------------------------------ variations (selected at launch)
    def _variation(self):
        from gift_bags.task_set_catalog import CONTRACT, selected_contract
        out = dict(preset=self.config.preset, task=self.config.task, seed=self.config.seed)
        if selected_contract(self.config.task, getattr(self.config, "contract", None)) == CONTRACT:
            out["contract"] = CONTRACT
        return out

    def variation_contract(self):
        from gift_bags.task_set_catalog import contract
        return dict(**contract(), current=self._variation(), robot=self.config.robot, in_session_selection=False,
                    reset="The variation is fixed when the session is launched (gar run gift-bags --select "
                          "preset=P --select task=T --select seed=S; gar tasks resolve gift-bags). POST "
                          "/v1/variations/select previews any catalogue entry; POST /v1/variations/start applies only "
                          "this session's own variation (a reset to its initial state) and refuses others without "
                          "changing anything")

    def select_variation(self, request):
        from gift_bags.task_set_catalog import resolve, instruction, task_id
        if set(request) - {"preset", "task", "seed", "contract"}:
            raise ValueError("Selectors are preset, task, seed and contract")
        preset = request.get("preset", None if "task" in request else self.config.preset)
        if preset is None:
            from gift_bags.task_set_catalog import TASKS
            preset = next((p for p, ts in TASKS.items() if request["task"] in ts), None)
            if preset is None:
                raise ValueError(f"Unknown task {request['task']!r}")
        sel = resolve(preset, request.get("task"), request.get("seed", 0), request.get("contract"))
        here = sel == self._variation()
        return dict(variation=sel, task=instruction(sel["task"], sel.get("contract")),
                    task_id=task_id(sel),
                    applies_in_this_session=here,
                    reset_request=dict(seed=sel["seed"], options=None if here else
                                       {k: v for k, v in sel.items() if k != "seed"}))

    # ------------------------------------------------------------------ rendering (one worker per session)
    def _scene(self):
        """The render scene, exported once per world: apart from the start poses and sheet skins, which every
        observe() request replaces with the recorded state, it is static (gift_bags/render.py)."""
        from gift_bags.render import export
        if getattr(self, "_scene_world", None) is not self.env.world or not self._scene_path.is_file():
            self._scene_path = export(self.env.world, self.output/"render")
            self._scene_world = self.env.world
        return self._scene_path

    def _new_renderer(self, scene=None):
        from gift_bags.render import Renderer
        return Renderer(self.env.world, self.output/"render", scene=scene or self._scene(),
                        width=self.options.get("width", 1280), height=self.options.get("height", 720),
                        samples=self.options.get("samples", 24))

    @staticmethod
    def _alive(renderer):
        process = getattr(renderer, "process", None)
        return renderer is not None and not getattr(renderer, "closed", False) and process is not None \
            and process.poll() is None

    def _start_renderer_early(self):
        """No live session worker: start and warm one while this action's physics runs (taken by the observation,
        released on failure)."""
        if not self.cameras or self._prestart is not None or self._alive(self._renderer):
            return
        from gift_bags.render import snapshot
        scene = self._scene()
        snap = snapshot(self.env.world)
        holder = {"renderer": None, "error": None, "warm_s": None}

        def work():
            try:
                t0 = time.perf_counter()
                holder["renderer"] = self._new_renderer(scene)
                holder["renderer"].observe(tuple(self.cameras), state=snap)
                holder["warm_s"] = time.perf_counter() - t0
            except BaseException as exc:
                holder["error"] = exc
        holder["thread"] = threading.Thread(target=work, name="gift-render-prestart", daemon=True)
        holder["thread"].start()
        self._prestart = holder

    def _take_prestart(self):
        holder, self._prestart = self._prestart, None
        if holder is None:
            return None, None
        t0 = time.perf_counter()
        holder["thread"].join()
        self.last_timing.update(prestart_wait_s=time.perf_counter() - t0, prestart_warm_s=holder["warm_s"])
        if holder["error"] is not None or holder["renderer"] is None:
            if holder["renderer"] is not None:
                self._release(holder["renderer"])
            self.last_timing["prestart_error"] = f"{type(holder['error']).__name__}: {holder['error']}"
            return None, holder["error"]
        return holder["renderer"], None

    def _release(self, renderer, wait=False):
        self._reapers = [t for t in self._reapers if t.is_alive()]
        if wait:
            renderer.close()
            return
        t = threading.Thread(target=renderer.close, name="gift-render-release", daemon=False)
        t.start()
        self._reapers.append(t)

    def _cancel_idle(self):
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _end_operation(self):
        """Keep the session worker for the next operation; release it after idle_release_s without one."""
        idle = float(self.options.get("idle_release_s", IDLE_RELEASE_S))
        r = self._renderer
        if r is None:
            return
        self._cancel_idle()
        if idle <= 0:
            self._renderer = None
            self._release(r)
            return
        timer = threading.Timer(idle, self._release_idle)
        timer.args = (r, timer)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _release_idle(self, renderer, timer):
        with self.env.lock:
            if self._renderer is not renderer or self._idle_timer is not timer:
                return                                       # used or replaced since this timer was armed
            self._renderer, self._idle_timer = None, None
        renderer.close()

    def _worker(self):
        """The session's render worker: the live one, else the one warmed during this action's physics, else a cold
        start. A worker whose process has exited is replaced."""
        self._cancel_idle()
        if self._renderer is not None and not self._alive(self._renderer):
            dead, self._renderer = self._renderer, None
            self.last_timing["replaced_exited_worker"] = True
            self._release(dead)
        if self._renderer is not None:
            self.last_timing["reused_worker"] = True
            return self._renderer
        renderer, _ = self._take_prestart()
        if renderer is None:
            t0 = time.perf_counter()
            renderer = self._new_renderer()
            self.last_timing["cold_start_s"] = time.perf_counter() - t0
        self._renderer = renderer
        return renderer

    def _render(self, names):
        with self.env.lock:
            try:
                for attempt in (0, 1):
                    try:
                        r = self._worker()
                        t0 = time.perf_counter()
                        result = r.observe(tuple(names))
                        self.last_timing["frame_s"] = time.perf_counter() - t0
                        break
                    except Exception as exc:
                        # the worker crashed, hung or refused the frame: release it and render once more from a fresh
                        # worker (the simulated state is unchanged); a second failure is reported
                        broken, self._renderer = self._renderer, None
                        if broken is not None:
                            self._release(broken)
                        self.last_timing.setdefault("render_failures", []).append(f"{type(exc).__name__}: {exc}")
                        if attempt:
                            raise
            finally:
                self._end_operation()
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
            joints = {n: {"position": v["position"], "velocity": v["velocity"]} for n, v in state["joints"].items()}
            return Observation(sequence, state["time_s"], self.config.robot, {"joints": joints},
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

    def _check_targets(self, targets):
        """RobotRig.set_targets' checks, made before anything is written."""
        joints = self.env.world.rig.joints
        for name, value in targets.items():
            if name not in joints:
                raise ValueError(f"Not a robot actuator: {name}")
            spec = joints[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value} "
                                 f"(limits {spec['lower']:.4f}..{spec['upper']:.4f})")

    def _check_grip(self, arm, width, force):
        """RobotRig.set_grip's checks, made before anything is written."""
        h = self.env.world.rig.hand[arm]
        if force is not None and (isinstance(force, bool) or not math.isfinite(force) or not 0 <= force <= h["effort"]):
            raise ValueError(f"Grip force must be in [0, {h['effort']}] N per finger")
        if width is not None and (isinstance(width, bool) or not math.isfinite(width) or not 0. <= width <= .04):
            raise ValueError("Franka Hand half gap must be in [0, 0.04] m")

    def _guard(self, arm_arm, robot_contact):
        """MotionSafety guards (the wiping adapter's semantics): the command stops after the contact step in which an
        arm touches the other arm (stop_on_arm_arm_contact) or any robot link touches anything (stop_on_contact /
        stop_on_robot_contact). Contacts read: penetrating MuJoCo contacts between different bodies (robot against
        furniture, floor, itself or the other arm; a wrist camera's mount on its own arm excluded) and contact-world reactions on robot links (paper, tape, items: a reaction
        exists once the link is within the barrier's activation distance). Returns None when nothing is guarded."""
        if not (arm_arm or robot_contact):
            return None
        if self.g1:
            return self._g1_guard(arm_arm, robot_contact)
        side = lambda s: s.split("/")[0] if s.startswith(("left/", "right/")) else \
            s.split("_")[0] if s.startswith(("left_camera", "right_camera")) else None
        mount = lambda s: s.endswith(("_camera_case", "_camera_bracket"))

        def guard(env):
            rig, ipc = env.world.rig, env.world.ipc
            g2s = np.asarray(rig.solver.mjc_geom_to_newton_shape.numpy())[0]
            labels, body = rig.model.shape_label, rig.model.shape_body.numpy()
            d = rig.solver.mj_data
            for i in range(d.ncon):
                c = d.contact[i]
                a, b = int(g2s[c.geom1]), int(g2s[c.geom2])
                if c.dist > 0. or a < 0 or b < 0 or body[a] == body[b]:
                    continue
                sides = {side(labels[a]), side(labels[b])}
                if len(sides) == 1 and (mount(labels[a]) or mount(labels[b])):
                    continue                       # the wrist camera's mount on its own arm (bolted; overlaps 0.46 mm)
                if arm_arm and sides == {"left", "right"}:
                    self._stop_reason = f"arm-arm contact: {labels[a]} / {labels[b]}"
                    return True
                if robot_contact and sides - {None}:
                    self._stop_reason = f"robot contact: {labels[a]} / {labels[b]}"
                    return True
            if robot_contact:
                for b_i, (F, _) in ipc.reactions.items():
                    if ipc.bodies[b_i].name.startswith("robot:") and np.linalg.norm(F) > 1e-6:
                        self._stop_reason = f"robot contact in the contact world: {ipc.bodies[b_i].name}"
                        return True
            return False
        return guard

    def _g1_guard(self, arm_arm, robot_contact):
        """MotionSafety for the G1: robot contact = a penetrating MuJoCo contact of a G1 body with anything, except its feet
        (ankle-roll links) on the floor it stands on, or a contact-world reaction on a G1 link; arm-arm = a left arm/hand
        body touching a right arm/hand body."""
        def part(lab):
            leaf = lab.split("/")[-1]
            if not lab.startswith("g1/"):
                return None
            for s in ("left", "right"):
                if leaf.startswith(s + "_") and any(k in leaf for k in ("shoulder", "elbow", "wrist", "hand")):
                    return s
            return "body"

        def guard(env):
            rig, ipc = env.world.rig, env.world.ipc
            g2s = np.asarray(rig.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            sb, blab, slab = rig.model.shape_body.numpy(), rig.model.body_label, rig.model.shape_label
            d = rig.solver.mj_data
            for i in range(d.ncon):
                c = d.contact[i]
                a, b = int(g2s[c.geom1]), int(g2s[c.geom2])
                if c.dist > 0. or a < 0 or b < 0 or sb[a] == sb[b]:
                    continue
                la = blab[sb[a]] if sb[a] >= 0 else slab[a]
                lb = blab[sb[b]] if sb[b] >= 0 else slab[b]
                pa, pb = part(la), part(lb)
                if pa is None and pb is None:
                    continue
                if {la.split("/")[-1], lb.split("/")[-1]} & {"left_ankle_roll_link", "right_ankle_roll_link"} and \
                        "floor" in (la, lb):
                    continue                               # standing
                if arm_arm and {pa, pb} == {"left", "right"}:
                    self._stop_reason = f"arm-arm contact: {la} / {lb}"
                    return True
                if robot_contact:
                    self._stop_reason = f"robot contact: {la} / {lb}"
                    return True
            if robot_contact:
                for b_i, (F, _) in ipc.reactions.items():
                    if ipc.bodies[b_i].name.startswith("robot:") and np.linalg.norm(F) > 1e-6:
                        self._stop_reason = f"robot contact in the contact world: {ipc.bodies[b_i].name}"
                        return True
            return False
        return guard

    def _trajectory(self, arm, position, delta, orientation_xyzw, ticks):
        """A Cartesian command as a motion generator executes it: the TCP moves on the straight line (orientation
        slerped) from where the arm is commanded to be to the goal, over the command's duration, with a minimum-jerk
        time profile; one joint-target set per contact step. The path is solved before anything moves
        (ArmKinematics.cartesian_path: one IK branch, poses <= 1 cm / 8 deg apart) and refused if unreachable, if it
        needs a branch change, or if the duration asks a joint for more than its velocity limit (the FR3 would
        trip its velocity reflex)."""
        from scipy.spatial.transform import Rotation
        rig = self.env.world.rig
        chain, names = self.ik.arms[arm]
        q0 = np.array([rig.targets[rig.joints[n]["q"]] for n in names], float)      # the commanded state
        T0 = self.ik.forward(arm, q0)
        goal = np.asarray(position, float) if position is not None else T0[:3, 3] + np.asarray(delta, float)
        quat = orientation_xyzw if orientation_xyzw is not None else Rotation.from_matrix(T0[:3, :3]).as_quat()
        path = self.ik.cartesian_path(arm, q0, [(goal, quat)])
        Q = np.vstack([q0] + [[j[n] for n in names] for j, _ in path])
        s_path = np.linspace(0., 1., len(Q))
        tau = np.arange(1, ticks + 1)/ticks
        s = tau**3*(10. - 15.*tau + 6.*tau**2)                                      # minimum jerk
        Qt = np.column_stack([np.interp(s, s_path, Q[:, j]) for j in range(Q.shape[1])])
        dt = self.config.dt
        speed = np.abs(np.diff(np.vstack([q0, Qt]), axis=0)).max(0)/dt
        limits = np.array([rig.joints[n]["velocity_limit"] for n in names])
        worst = int(np.argmax(speed/limits))
        if speed[worst] > limits[worst]:
            need = ticks*dt*speed[worst]/limits[worst]
            raise ValueError(f"{names[worst]} would need {speed[worst]:.2f} rad/s (limit {limits[worst]:.2f}): "
                             f"give this motion at least {need:.2f} s; no action executed")
        return [dict(zip(names, map(float, q))) for q in Qt]

    def _joint_schedule(self, targets, ticks):
        """A joint-target command as a joint motion generator executes it: every commanded arm joint moves from where
        it is commanded to be to its target over the command's duration, on a minimum-jerk time profile, one target
        set per contact step; the command is refused if that asks a joint for more than its velocity limit. Finger
        joints (and every joint of the G1, which has no such interface here) are set at once.

        Set at once, a far arm target is a step: the position servo drives the joints far past their velocity limits
        (the reflex only freezes the targets), and in one abuse run (lift-lid, seed 2) the right arm tripped the reflex
        65 times in 0.5 s, its tool point moved 97 mm in one 4 ms step and joint 4 ended 0.15 rad beyond its limit. A
        real FR3 rejects such a command."""
        rig = self.env.world.rig
        if self.g1:
            return None
        names = [n for n in targets if n in rig.joints and "finger" not in n and rig.joints[n].get("velocity_limit")]
        if not names:
            return None
        self._check_targets(targets)
        q0 = np.array([rig.targets[rig.joints[n]["q"]] for n in names], float)      # the commanded state
        q1 = np.array([targets[n] for n in names], float)
        tau = np.arange(1, ticks + 1)/ticks
        s = tau**3*(10. - 15.*tau + 6.*tau**2)                                      # minimum jerk
        Qt = q0 + np.outer(s, q1 - q0)
        dt = self.config.dt
        speed = np.abs(np.diff(np.vstack([q0, Qt]), axis=0)).max(0)/dt
        limits = np.array([rig.joints[n]["velocity_limit"] for n in names])
        worst = int(np.argmax(speed/limits))
        if speed[worst] > limits[worst]:
            need = ticks*dt*speed[worst]/limits[worst]
            raise ValueError(f"{names[worst]} would need {speed[worst]:.2f} rad/s (limit {limits[worst]:.2f}): give this "
                             f"motion at least {need:.2f} s ({int(math.ceil(need/dt))} ticks); no action executed")
        rest = {n: v for n, v in targets.items() if n not in names}
        return [{**rest, **dict(zip(names, map(float, q)))} for q in Qt]

    def _plan(self, action):
        """The whole command, validated, with no side effect: (joint targets, ticks, grip command or None,
        guard or None)."""
        rig = self.env.world.rig
        grip = None                                          # (arm, half width m, force N per finger)
        guard = schedule = None
        if isinstance(action, NamedJointTarget):
            targets, ticks = dict(action.targets_si), action.ticks
            if type(ticks) is not int or ticks <= 0:
                raise ValueError("ticks must be a positive integer")
            schedule = self._joint_schedule(targets, ticks)
        elif isinstance(action, Gripper) and self.g1:
            from gift_bags import g1 as G1
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            if action.inward_effort_n is not None:
                raise UnsupportedActionError("the G1's Dex3-1 hand has no force-mode grasp here: use half_width_m (a hand "
                                             "posture, see capabilities) or named targets for its 7 finger joints")
            targets = G1.hand_posture(arm, G1.closure_of_half_width(action.half_width_m))
        elif isinstance(action, Gripper):
            arm = self._arm(action.arm)
            ticks = self._ticks(action.duration_s)
            targets = {}
            if action.inward_effort_n is not None:
                total = float(action.inward_effort_n)
                if not math.isfinite(total) or not 0. <= total <= 2*rig.hand[arm]["effort"]:
                    raise ValueError(f"inward_effort_n must be in [0, {2*rig.hand[arm]['effort']}] N")
                grip = (arm, None, total/2.) if total > 0 else \
                    (arm, action.half_width_m if action.half_width_m is not None else .04, None)
            else:
                grip = (arm, action.half_width_m, None)
        elif isinstance(action, (CartesianPose, CartesianDelta)):
            arm = self._arm(action.arm)
            safety = action.safety
            if safety.stop_on_grip_loss is not None:
                raise UnsupportedActionError("Grip-loss guard is not implemented")
            guard = self._guard(safety.stop_on_arm_arm_contact, safety.stop_on_contact or safety.stop_on_robot_contact)
            ticks = self._ticks(action.duration_s)
            o = action.orientation_wxyz if isinstance(action, CartesianPose) else None
            schedule = self._trajectory(arm, action.position_m if isinstance(action, CartesianPose) else None,
                                        None if isinstance(action, CartesianPose) else action.delta_m,
                                        None if o is None else (*o[1:], o[0]), ticks)
            targets = schedule[-1]
            if action.gripper_half_target_m is not None and self.g1:
                from gift_bags import g1 as G1
                hand = G1.hand_posture(arm, G1.closure_of_half_width(action.gripper_half_target_m))
                schedule = [{**j, **hand} for j in schedule]
                targets = schedule[-1]
            elif action.gripper_half_target_m is not None:
                grip = (arm, action.gripper_half_target_m, None)
        else:
            raise UnsupportedActionError("Supported: named_joint_target, gripper, cartesian_pose, cartesian_delta")
        self._check_targets(targets)
        if grip is not None:
            self._check_grip(*grip)
        return targets, ticks, grip, guard, schedule

    def act(self, action):
        """Plan and validate the whole command first (a refused command raises ActionRejectedError or
        UnsupportedActionError and writes nothing); then set the grip and the joint targets and run the physics."""
        with self.env.lock:
            if self.env.world.fault:
                raise BackendProtocolError(f"Physics faulted at tick {self.env.world.steps}: {self.env.world.fault}")
            try:
                targets, ticks, grip, guard, schedule = self._plan(action)
            except (ValueError, TypeError, UnsupportedActionError) as exc:
                if self.trace and self.trace.current:
                    self.trace.outcome = ("rejected", f"{type(exc).__name__}: {exc}")
                if isinstance(exc, UnsupportedActionError):
                    raise
                err = ActionRejectedError(str(exc))
                # planning refusals are computed from the command and the robot's own joints (duration, joint jump, IK,
                # limits, grip width): safe to show a restricted policy (http_server passes policy_reason through)
                err.policy_reason = str(exc)
                raise err from exc
            rig = self.env.world.rig
            if grip is not None:
                self.env.set_grip(grip[0], width=grip[1], force=grip[2])
            if self.g1 and isinstance(action, (Gripper, CartesianPose, CartesianDelta)):
                hp = action.half_width_m if isinstance(action, Gripper) else action.gripper_half_target_m
                if hp is not None:                          # the posture command, for the state/trace (targets carry it)
                    from gift_bags import g1 as G1
                    rig.hand[action.arm]["closure"] = G1.closure_of_half_width(hp)
            before = self.env.world.steps
            n_reflex = len(rig.reflexes)
            self.last_timing = {}
            self._cancel_idle()
            self._start_renderer_early()
            if self.trace and self.trace.current:
                self.trace.expect(ticks)
            try:
                t0 = time.perf_counter()
                self._stop_reason = None
                state = self.env.act(schedule[0] if schedule else targets, ticks=ticks, guard=guard, schedule=schedule)
                self.last_timing["physics_s"] = time.perf_counter() - t0
            except BaseException as exc:
                if self.trace and self.trace.current:
                    self.trace.outcome = ("failed", f"{type(exc).__name__}: {exc}")
                renderer, _ = self._take_prestart()
                if renderer is not None:
                    self._release(renderer)
                if isinstance(exc, (RuntimeError, FloatingPointError, OSError, ValueError)):
                    raise BackendProtocolError(f"Stopped at tick {self.env.world.steps}: {exc}") from exc
                raise
            ev = state["evaluation"]
            return BackendStep(state, {"requested_ticks": ticks, "executed_ticks": self.env.world.steps - before,
                                       "stopped_on_contact": self._stop_reason is not None,
                                       "stop_reason": self._stop_reason,
                                       "velocity_reflexes": [dict(r) for r in rig.reflexes[n_reflex:]],
                                       "contact_solver": state["contact_solver"],
                                       "wall_per_sim": self.last_timing["physics_s"]/max(ticks*self.config.dt, 1e-12)},
                               ev["score"], bool(state["terminated"]))

    def policy_access(self):
        from gift_bags.task_set_catalog import instruction, completes_window
        rig = self.env.world.rig
        return PolicyAccess(self.config.robot, tuple(self.camera_specs[c] for c in self.cameras),
                            {("joints", n, f): None for n in rig.joints for f in ("position", "velocity")},
                            self.kinds, lambda: instruction(self.config.task, getattr(self.config, "contract", None)),
                            {"dt_s": self.config.dt, "arms": list(self.arms),
                             "joints": {n: {**{k: v for k, v in s.items() if k not in ("q", "dof", "native_label")},
                                            "unit": "metre" if "finger" in n else "radian"}
                                        for n, s in rig.joints.items()},
                             **({"evaluation": SETTLE_NOTE, "client_timeout_s": SETTLE_CLIENT_TIMEOUT_S}
                                if completes_window(self.config.task, getattr(self.config, "contract", None)) else {})})

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
        ipc = self.env.world.ipc
        return {"items": {b.name: {"kind": b.kind, "mass_kg": b.mass} for b in ipc.bodies if b.kind == "dynamic"},
                "sheets": {s.name: dict(vertices=int(sl.stop - sl.start), triangles=len(s.tris),
                                        stocks=[m.name for m in s.mats]) for s, sl in zip(ipc.sheets, ipc.sheet_slices)},
                "furniture": [dict(name=f["name"], center_m=list(f["center"]), half_m=list(f["half"]))
                              for f in self.env.world.scene.furniture],
                "contact_caliper": {k: v for k, v in ipc.caliper.items() if k != "stock"}}

    def get_geometry(self, geometry_id):
        ipc = self.env.world.ipc
        for k, (s, sl) in enumerate(zip(ipc.sheets, ipc.sheet_slices)):
            if geometry_id == s.name:
                return {"vertices_m": ipc.x_free()[sl].tolist(), "triangles": np.asarray(s.tris).tolist(),
                        "rest_coordinates_m": np.asarray(s.X2).tolist()}
        raise KeyError(geometry_id)

    def get_geometry_asset(self, asset_id):
        raise KeyError(asset_id)

    def save_checkpoint(self, *, label):
        with self.env.lock:
            self._checkpoints[label] = self.env.checkpoint()
            return {"label": label, "tick": self.env.world.steps, "generation": self.env.generation,
                    "storage": "in_memory_session"}

    def restore_checkpoint(self, label, **kwargs):
        with self.env.lock:
            snap = self._checkpoints.get(label)
            if snap is None:
                raise KeyError(label)
            self.env.restore(snap)
            self.env.generation += 1
            return self.env.get_state(), {"checkpoint": label, "tick": self.env.world.steps,
                                          "generation": self.env.generation, "continuity": "restored_discontinuity",
                                          "storage": "in_memory_session"}

    def list_checkpoints(self):
        return tuple({"label": k, "tick": v["steps"], "storage": "in_memory_session"}
                     for k, v in self._checkpoints.items())

    def begin_action_trace(self, context):
        return self.trace.begin(context) if self.trace else {"available": False, "reason": "recording_disabled"}

    def end_action_trace(self):
        return self.trace.finish() if self.trace else None

    def close(self):
        with self.env.lock:
            self._cancel_idle()
            renderer, _ = self._take_prestart()
            for r in (renderer, self._renderer):
                if r is not None:
                    self._release(r, wait=True)
            self._renderer = None
            for t in self._reapers:
                t.join()
            self._reapers = []
        if self.trace:
            self.trace.close()
            self.trace = None
        self.env.close()

"""Welding environment wrapper (the GAR backend's engine): the coupled world, the task's start state, checkpoints,
actions (joint targets, the source's I/O), histories for the evaluator, and the state dictionary.

Start states (disclosed initial conditions):
  loose        the bracket stands on the clamped base plate, unattached (tack).
  tacked       the bracket has been tacked at both ends of the near side by an ASSISTED fixture: a phantom torch (no
               robot) runs the spray schedule at 19 mm over two 15 mm tacks with the world's own physics, then the
               part cools for TACK_COOL_S. The tacks are therefore real fused weld metal and a real bond, made by the
               same process the policy will use. The state is cached on disk by a hash of the physics source.
The robots start at their 'welding' poses; the cell then settles hands-off until it is at rest (settle_to_rest:
REST_WINDOW_S windows until nothing E0 gates moves REST_LIMIT_MM in one), before any weld state is placed.
"""

import copy
import hashlib
import math
import os
import pickle
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .world import World, Config
from .task import Evaluator, INSTRUCTIONS, TIME_LIMIT_S, START_STATE

SETTLE_S = .3
# The start state is handed over at rest. Settling for SETTLE_S only is enough for the fixed-base arms (E0: 0.00 mm),
# not for the G1 standing on its position servos with no balance controller (humanoid.py), which would be handed over
# swaying -- its torch swung 25.7 mm peak to peak in the first second and came to rest ~19 mm from its start pose ~9 s
# later. So the cell settles SETTLE_S (the clamp pads and the coupon on their contacts: 0.56 mm), then on in
# REST_WINDOW_S windows until over one whole window no free body, torch work point or handling hand's tool point (the
# points E0 gates) moves REST_LIMIT_MM from where it was at the window's start (settle_to_rest); at most REST_MAX_S,
# else the build fails. The window is E0's 1 s and longer than the standing G1's sway period (about 0.6 s: a 0.3 s
# window passed at a turning point while the sway was still 0.07 mm); the limit is half E0's gate, so the first second
# after handover, which E21 gates at REST_MM, keeps a factor-2 margin. It runs on the cold coupon before a tack weld is
# placed or built: the weld's thermal state at handover is unchanged. The fixed-base arms pass the first window
# (1e-6 mm); the G1 needs about 13-15 s, so its settled robot state is cached per physics source and rig
# (rest_<hash>_<rig>.pkl, _settle_start) and later worlds start their robot from it and settle the same way.
REST_MM = .05             # E0's gate and E21's start_at_rest gate
REST_WINDOW_S = 1.
REST_LIMIT_MM = REST_MM/2
REST_MAX_S = 30.
REST_SAMPLE_TICKS = 20
TACK_COOL_S = 8.
TACKS = ((.008, .023), (.127, .142))          # weld-frame x ranges of the two tacks (m) [authored: 15 mm each,
                                              # within 25 mm of each end, as a fitter places them]
from . import params as PR
# The fixture tacks with the joint's own spray procedure at its 19 mm contact tip to work: in this 90 deg corner at 45 deg
# the tapered 5/8 in nozzle cannot come closer than ~18 mm, where the documented short-circuit schedule (175 A at
# 12 mm) falls to ~147 A and fused nothing on this 10 mm plate (phantom technique tests); a cell doing
# short-circuit work would fit a 1/2 in nozzle, which is not modelled. Spray tacks of a 15 mm path fuse ~19 mm.
_SP = PR.v(PR.SCHEDULES)["spray_0p9_90-10"]          # Lincoln C4.200: 0.035 in ER70S-6, 90/10, 29 V spray
TACK_SCHEDULE = dict(wire_feed_m_min=_SP["wfs_m_min"], voltage_v=_SP["voltage_v"])
TACK_CTWD = _SP["ctwd_m"]
TACK_TRAVEL, TACK_WORK, TACK_PUSH = 10e-3, 45., 5.      # m/s, deg, deg [authored: a quick tack pass]
TACK_POST_FLOW_S = .5     # s of gas over each tack after the arc (Miller Auto-Deltaweld default, as the robot programs)
SOURCE_FILES = ("params.py", "thermal.py", "arc.py", "pool.py", "process.py", "bonds.py", "scene.py", "world.py",
                "layout.py", "torch.py", "env.py", "humanoid.py")


def _cache_dir():
    ws = os.environ.get("GAR_WORKSPACE")
    base = Path(ws)/".gar"/"welding-cache" if ws else Path.home()/".cache"/"gar"/"welding"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _prune_cache(current, keep_other=2):
    """Start states are ~190 MB each: keep the current physics source's and the newest `keep_other` older sources'
    (the tack welds, tacked_<hash>.pkl, and the settled robot states, rest_<hash>_<rig>.pkl, a few kB)."""
    groups = {}
    for p in list(_cache_dir().glob("tacked_*.pkl")) + list(_cache_dir().glob("rest_*.pkl")):
        groups.setdefault(p.stem.split("_")[1], []).append(p)
    old = sorted((h for h in groups if h != current), key=lambda h: -max(p.stat().st_mtime for p in groups[h]))
    for h in old[keep_other:]:
        for p in groups[h]:
            p.unlink(missing_ok=True)


def _hash_sources():
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in SOURCE_FILES:
        h.update((here/f).read_bytes())
    return h.hexdigest()[:16]


# taken once, when this module is imported with the physics modules: a process keeps the key of the code it runs (hashed
# at save time, a process started before an edit would store its old tacks under the new key)
_PHYSICS_HASH = _hash_sources()


def physics_hash():
    return _PHYSICS_HASH


def rest_points(w):
    """What a start state must hold still, the points E0 gates: every free body's origin (the coupon, the clamp pads,
    the ground clamp), the torch work point and each handling hand's tool point (world, m; one row each). The robots'
    own links are not in it: the G1's right ankle and knee moved 0.09-0.16 mm in every 0.3 s window through 60 s of
    standing (rigid cell)."""
    q = w.state.body_q.numpy()[:, :3].astype(float)
    free = getattr(w, "_free_bodies", None)
    if free is None:
        free = w._free_bodies = [i for i in range(w.model.body_count) if i not in w._robot_bodies]
    pts = [q[free], np.asarray(w.torch_pose()[2], float)[None]]
    pts += [np.asarray(w.kin.forward(a)[:3, 3], float)[None] for a in w.hand_arms]
    return np.concatenate(pts)


def settle_to_rest(w, window_s=REST_WINDOW_S, limit_mm=REST_LIMIT_MM, max_s=REST_MAX_S, every=REST_SAMPLE_TICKS):
    """Step the world hands-off (nothing commanded) in windows of window_s until, over one whole window, no rest point
    moved limit_mm or more from where it was at the window's start (sampled every `every` ticks); at most max_s.
    Returns (simulated s, each window's largest motion in mm)."""
    t0, motions = w.time, []
    n = max(every, int(round(window_s/w.config.rigid_dt)))
    while True:
        ref, worst = rest_points(w), 0.
        for k in range(n):
            w.step()
            if (k + 1) % every == 0 or k == n - 1:
                worst = max(worst, float(np.linalg.norm(rest_points(w) - ref, axis=1).max())*1e3)
        motions.append(worst)
        if worst < limit_mm or w.time - t0 >= max_s - 1e-9:
            return w.time - t0, motions


def _robot_coords(w):
    """Generalized coordinate and velocity indices of the robots' joints (their child bodies are robot bodies; the G1's
    floating base included)."""
    m = w.model
    child, qs, ds = m.joint_child.numpy(), m.joint_q_start.numpy(), m.joint_qd_start.numpy()
    qi, di = [], []
    for j in range(m.joint_count):
        if int(child[j]) in w._robot_bodies:
            qi += range(int(qs[j]), int(qs[j + 1]))
            di += range(int(ds[j]), int(ds[j + 1]))
    return np.array(qi, int), np.array(di, int)


def robot_state(w):
    qi, di = _robot_coords(w)
    return dict(q_index=qi.tolist(), qd_index=di.tolist(), q=w.state.joint_q.numpy()[qi].copy(),
                qd=w.state.joint_qd.numpy()[di].copy())


def place_robot_state(w, snap):
    """Start this world's robots from a settled robot state of the same physics source and rig (robot_state). Returns
    False, changing nothing, if the state does not fit this model."""
    qi, di = _robot_coords(w)
    if snap.get("q_index") != qi.tolist() or snap.get("qd_index") != di.tolist():
        return False
    q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
    q[qi], qd[di] = snap["q"], snap["qd"]
    w.state.joint_q.assign(q)
    w.state.joint_qd.assign(qd)
    w.newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
    for a in ("body_q", "body_qd", "joint_q", "joint_qd"):
        getattr(w.next_state, a).assign(getattr(w.state, a).numpy())
    w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
    w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
    return True


def _import_engine():
    """Import Newton before the first World is built. Newton keeps the ImportError of its optional USD schemas in a
    module global (newton/_src/usd/__init__.py, _newton_usd_schemas_import_error), and that exception's traceback keeps
    every frame on the stack at Newton's first import alive for the life of the process. When the first import ran
    inside World.__init__ it kept that frame -- the process's first world and its locals, Newton's ModelBuilder and the
    MuJoCo model among them -- after the world was closed. Imported here, the frames kept are this function's and
    WeldingEnv.__init__'s (the env is emptied by close())."""
    from .world import require_engine
    require_engine()


class WeldingEnv:
    TCP_EVERY = 50                 # ticks between TCP motion samples (settled check)

    def __init__(self, config=None, *, output_dir=None, use_cache=True):
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = output_dir
        _import_engine()
        self.world = World(c)
        self.generation = 0
        self.steps = 0
        self.start_state = START_STATE.get(c.task, "loose")
        self.assisted_log = []
        self._make_start(use_cache)
        self._reset_histories()
        self.initial = self.checkpoint()
        self.evaluator = Evaluator(self)

    # ------------------------------------------------------------------ start states
    def _make_start(self, use_cache):
        self._make_tacked(use_cache)
        self.assisted_log.append(self.rest_record)
        if not self.config.work_clamp_on_table:
            self.world.park_work_clamp()
            self.world.settle(.3)
            self.world._update_circuit()
            self.assisted_log.append(dict(fixture="ground clamp parked on the power source (circuit variant)"))
            self._clear_world_logs()

    def _make_tacked(self, use_cache):
        """The tacked start state. The tack fixture's weld (a phantom torch: the robots hold still) does not depend on
        the robots or on where the coupon sits on the table -- the cells live in the base plate's frame -- so it is built
        once per physics source and cached as the weld process and bond state only; every deployment and catalogue entry
        places it on its own freshly settled world in seconds (_place_tacked). Caching the whole world per deployment
        and per entry would cost 6-10 GPU minutes and 187 MB each for bit-identical weld states."""
        w = self.world
        self._settle_start(use_cache)
        if self.start_state != "tacked" or w.process is None:
            self._clear_world_logs()
            return
        path = _cache_dir()/f"tacked_{physics_hash()}.pkl"
        if use_cache and path.is_file():
            snap = pickle.loads(path.read_bytes())
            self._place_tacked(snap)
            self.assisted_log = list(snap.get("assisted_log", []))
            self.assisted_log.append(dict(note=f"tack fixture's weld state loaded from cache {path.name} and placed on "
                                               f"this deployment's coupon"))
            self._clear_world_logs()
            return
        t0 = time.perf_counter()
        self._tack_fixture()
        w.settle(TACK_COOL_S)
        self.assisted_log.append(dict(fixture="tacks", wall_s=time.perf_counter() - t0, sim_s=w.time,
                                      bond=dict(active=w.bond.active, capacity_n=w.bond.capacity_n)))
        if not w.bond.active:
            raise RuntimeError("the tack fixture produced no bond (the start state would be untacked)")
        if use_cache:
            b = w.bond
            snap = dict(process=w.process.snapshot(), assisted_log=self.assisted_log,
                        bond=dict(capacity_n=b.capacity_n, length_m=b.length_m, x_range=b.x_range,
                                  events=copy.deepcopy(b.events), cracked=sorted(b.cracked),
                                  sections=list(getattr(b, "sections", [])), group=list(b.group)))
            tmp = path.with_suffix(f".{os.getpid()}.tmp")         # concurrent builders do not share a temp file
            tmp.write_bytes(pickle.dumps(snap, protocol=5))
            tmp.replace(path)
            _prune_cache(physics_hash())
        self._clear_world_logs()

    def _settle_start(self, use_cache):
        """The cell settles hands-off until it is at rest (SETTLE_S, then settle_to_rest; see SETTLE_S above). A rig
        whose settle takes more than one window (the standing G1: about 13-15 simulated s at ~14 wall s each on the
        shared GPU) writes its settled robot state to the cache once per physics source; later worlds start their robot
        from it and settle to rest the same way (the coupon and clamps settle from their import poses in every world).
        Recorded in assisted_log ('settle')."""
        w = self.world
        rig = w.scene.robot_setup.to_dict()["id"]
        path = _cache_dir()/f"rest_{physics_hash()}_{rig}.pkl"
        placed = bool(use_cache and path.is_file() and place_robot_state(w, pickle.loads(path.read_bytes())))
        t0, limit = time.perf_counter(), REST_LIMIT_MM
        w.settle(SETTLE_S)
        sim_s, windows = settle_to_rest(w, limit_mm=limit)
        sim_s += SETTLE_S
        self.rest_record = dict(settle="hands-off until at rest", rig=rig, sim_s=round(sim_s, 3),
                                wall_s=round(time.perf_counter() - t0, 1), window_s=REST_WINDOW_S, limit_mm=limit,
                                window_motion_mm=[round(m, 5) for m in windows], at_rest=bool(windows[-1] < limit),
                                robot_from_cache=path.name if placed else None)
        if windows[-1] >= limit:
            raise RuntimeError(f"the start state did not come to rest in {REST_MAX_S:g} s hands-off (last window "
                               f"{windows[-1]:.4f} mm >= {limit:.4f} mm): {self.rest_record}")
        if use_cache and not placed and len(windows) > 1:
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(pickle.dumps(dict(robot_state(w), rig=rig, record=self.rest_record), protocol=5))
            tmp.replace(path)
            _prune_cache(physics_hash())

    def _place_tacked(self, snap):
        """Put the cached tack fixture's weld onto this world: the cells and source state, then the bond equality
        switched on at the parts' present (settled, fit-up) pose with the cached capacity and weld group."""
        w = self.world
        w.process.restore(snap["process"])
        b, sb = w.bond, snap["bond"]
        b.capacity_n, b.length_m, b.x_range = sb["capacity_n"], sb["length_m"], sb["x_range"]
        b.events, b.cracked = copy.deepcopy(sb["events"]), set(sb["cracked"])
        b.sections, b.group = list(sb["sections"]), list(sb["group"])
        b.active, b.broken = True, False
        w.set_bond(True)
        w._weld_colliders(every=True)
        w.settle(SETTLE_S)
        w._update_circuit()

    def _tack_fixture(self):
        """ASSISTED: two tacks with a phantom torch (welding/process.py driven along a scripted path in the weld
        frame; the robots hold still). Source commands are the same tool_io a robot program sends."""
        w = self.world
        th, wa = math.radians(TACK_PUSH), math.radians(TACK_WORK)
        axis = np.array([math.sin(th), -math.cos(th)*math.cos(wa), -math.cos(th)*math.sin(wa)])
        w.tool_io(gas_on=True, **TACK_SCHEDULE)
        for x0, x1 in TACKS:
            state = dict(x=x0)

            def path(t, state=state):
                return np.array([state["x"], 0., 0.]) - axis*TACK_CTWD, axis
            w.phantom = path
            w.tool_io(arc_on=True)
            k = 0
            while w.process.src.st.arc is False and k < 1500 and not w.process.src.st.fault:
                w.step()
                k += 1
            while state["x"] < x1:
                w.step()
                state["x"] += TACK_TRAVEL*w.config.rigid_dt
            w.tool_io(arc_on=False)
            for _ in range(int(round(TACK_POST_FLOW_S/w.config.rigid_dt))):
                w.step()
            self.assisted_log.append(dict(fixture="tack", x_mm=(x0*1e3, x1*1e3), fault=w.process.src.st.fault))
        # the program closes the gas valve after the post-flow
        w.tool_io(gas_on=False)
        w.phantom = None
        w._tip_prev = None

    def _clear_world_logs(self):
        w = self.world
        # a protective stop while the start state is built means the start state is broken (a collider overlapping
        # an arm at its home pose spun the UR10e and toppled the bracket, and clearing the logs hid it)
        if w.reflexes:
            raise RuntimeError(f"protective stop while building the start state: {w.reflexes[:3]}")
        w.events, w.reflexes, w._reflex_until, w._back_off = [], [], {}, {}
        w.timing = {k: 0. for k in w.timing}

    # ------------------------------------------------------------------ histories
    def _reset_histories(self):
        self.steps = 0
        self.history = dict(arc_s=0., wire_touch=False, max_current_a=0., gun_contact_ticks=0, strikes=0,
                            faults=[])
        self.tcp_log = deque(maxlen=100)
        self._arc_prev = False

    def _record(self):
        w = self.world
        st = w.process.src.st if w.process is not None else None
        if st is not None:
            if st.arc:
                self.history["arc_s"] += w.config.rigid_dt
                self.history["max_current_a"] = max(self.history["max_current_a"], float(st.current))
                if not self._arc_prev:
                    self.history["strikes"] += 1
            self._arc_prev = bool(st.arc)
            if st.fault and (not self.history["faults"] or self.history["faults"][-1] != st.fault):
                self.history["faults"].append(st.fault)
            if (not st.arc) and math.isfinite(st.d_tw) and st.d_tw <= st.l_s + 1e-4:
                self.history["wire_touch"] = True
        if self.steps % self.TCP_EVERY == 0:
            _, _, tcp = w.torch_pose()
            self.tcp_log.append((self.sim_time, tcp.copy()))
        if self.steps % 10 == 0 and w.gun_contacts():
            self.history["gun_contact_ticks"] += 10

    def tcp_motion_mm(self, window_s):
        """Largest TCP displacement (mm) over the last window_s from the sampled log (None if too short)."""
        pts = [(t, p) for t, p in self.tcp_log if t >= self.sim_time - window_s - 1e-9]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < window_s - 2*self.TCP_EVERY*self.config.rigid_dt:
            return None
        P = np.array([p for _, p in pts])
        return float(np.linalg.norm(P - P[-1], axis=1).max()*1e3)

    @property
    def sim_time(self):
        return self.steps*self.config.rigid_dt

    # ------------------------------------------------------------------ physical queries for the evaluator
    def bracket_alignment(self):
        """(squareness error deg, offset mm) of the bracket relative to the base plate vs the authored fit-up."""
        w = self.world
        if "bracket" not in w.scene.bodies:
            return 0., 0.
        pb, Rb = w.body_pose(w.scene.bodies["base_plate"])
        pk, Rk = w.body_pose(w.scene.bodies["bracket"])
        Rrel = Rb.T @ Rk
        ang = math.degrees(math.acos(np.clip(Rrel[2, 2], -1, 1)))
        rel = Rb.T @ (pk - pb)
        nominal = w.scene.bracket_rel_nominal
        return ang, float(np.linalg.norm(rel - nominal)*1e3)

    def joint_max_temperature(self):
        p = self.world.process
        if p is None:
            return 20.
        pool = p.pool
        T = p.s.field("T").reshape(p.g.nx, p.g.ny, p.g.nz)[:, pool.j0:pool.j1, pool.k0:pool.k1]
        return float(T.max())

    def hand_touches_bracket(self):
        w = self.world
        if "bracket" not in w.scene.bodies or not w.hand_arms:
            return False
        return w.bodies_touch(w.scene.bodies["bracket"], set(w.hand_bodies()))

    # ------------------------------------------------------------------ checkpoints
    def _world_snapshot(self):
        w = self.world
        return dict(
            rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
            targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), cmd_v=w._cmd_v.copy(),
            tick=w.tick, grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched),
            reflex_until=dict(w._reflex_until), back_off=dict(w._back_off), reflexes=copy.deepcopy(w.reflexes),
            refused=w.refused_targets,
            events=copy.deepcopy(w.events),
            process=w.process.snapshot() if w.process is not None else None,
            bond=dict(active=w.bond.active, broken=w.bond.broken, capacity_n=w.bond.capacity_n,
                      length_m=w.bond.length_m, x_range=w.bond.x_range, events=copy.deepcopy(w.bond.events),
                      cracked=sorted(w.bond.cracked), sections=list(getattr(w.bond, "sections", [])),
                      group=list(getattr(w.bond, "group", [])),
                      eq=w.bond_eq_state()))

    def _restore_world(self, snap):
        w = self.world
        for a, v in snap["rigid"].items():
            getattr(w.state, a).assign(v)
            getattr(w.next_state, a).assign(v)
        w.targets = snap["targets"].copy()
        w.control.joint_target_q.assign(snap["control"])
        w._cmd_v = snap["cmd_v"].copy()
        w.tick = snap["tick"]
        w.grip_force = dict(snap["grip_force"])
        w.grasp_latched = dict(snap["grasp_latched"])
        w._reflex_until, w.reflexes = dict(snap["reflex_until"]), copy.deepcopy(snap["reflexes"])
        w._back_off = dict(snap.get("back_off", {}))
        w.refused_targets = snap["refused"]
        w.events = copy.deepcopy(snap["events"])
        if snap["process"] is not None:
            w.process.restore(snap["process"])
            w._weld_colliders(every=True)         # the restored cells' weld metal as colliders
        b = snap["bond"]
        w.bond.active, w.bond.broken, w.bond.capacity_n = b["active"], b["broken"], b["capacity_n"]
        w.bond.length_m, w.bond.x_range, w.bond.events = b["length_m"], b["x_range"], copy.deepcopy(b["events"])
        w.bond.cracked, w.bond.sections = set(b.get("cracked", [])), list(b.get("sections", []))
        w.bond.group = list(b.get("group", []))
        if w.bond.active and not w.bond.group:
            w.bond.update()                   # a state saved before the weld group was kept: rebuild it from the cells
        w.restore_bond_eq(b["eq"])
        w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
        w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
        w._tip_prev = None
        w.fault = None
        w._update_circuit()

    def checkpoint(self):
        snap = self._world_snapshot()
        snap.update(steps=self.steps, history=copy.deepcopy(self.history), tcp_log=list(self.tcp_log),
                    arc_prev=self._arc_prev)
        return snap

    def restore(self, snap):
        self._restore_world(snap)
        self.steps = snap["steps"]
        self.history = copy.deepcopy(snap["history"])
        self.tcp_log = deque(snap["tcp_log"], maxlen=100)
        self._arc_prev = snap["arc_prev"]

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, state=True):
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            if targets:
                w.set_targets(targets)
            for _ in range(ticks):
                w.step()
                self.steps += 1
                self._record()
            return self.get_state() if state else None

    def tool_io(self, outputs):
        with self.lock:
            self.world.tool_io(**outputs)

    # ------------------------------------------------------------------ state
    def tool_feedback(self):
        """What the welding power source reports to the robot controller (a real robot interface's feedback)."""
        p = self.world.process
        if p is None:
            return None
        st, s = p.src.st, p.src.s
        return dict(arc_on=bool(st.arc), current_a=float(st.current), voltage_v=float(st.v_term),
                    wire_feed_m_min=float(st.wfs_actual*60.), touch=bool(st.touch), fault=st.fault or None,
                    gas_on=bool(st.gas_on), stickout_mm=float(st.l_s*1e3),
                    commanded=dict(arc_on=bool(st.on_cmd), touch_sense=bool(st.sensing),
                                   wire_inch={1: "forward", -1: "backward", 0: "stop"}[st.inch],
                                   wire_feed_m_min=float(s.wfs_m_min), voltage_v=float(s.voltage_v)))

    def get_state(self, evaluate=True):
        with self.lock:
            w = self.world
            jq, jv = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
            ev = self.evaluator.evaluate() if (evaluate and not w.fault) else dict(success=False, score=0., checks={})
            p = w.process
            return {
                "tick": self.steps, "generation": self.generation, "time_s": self.sim_time,
                "ready": w.fault is None, "fault": w.fault,
                "config": {k: v for k, v in asdict(self.config).items()},
                "instruction": INSTRUCTIONS.get(self.config.task),
                "start_state": self.start_state,
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                               "target": float(w.targets[s["q"]])} for n, s in w.joints.items()},
                "tool": self.tool_feedback(),
                "process": None if p is None else dict(
                    circuit_closed=bool(w.circuit_closed), arc_length_mm=float(p.src.st.l_a*1e3),
                    transfer_mode=p.src.st.mode or None, air_fraction=float(p.src.st.x_air),
                    stickout_mm=float(p.src.st.l_s*1e3), deposited_mm3=float(p.src.st.deposited_m3*1e9),
                    liquid_sections=int(sum(1 for s_ in p.pool.slices if s_.liquid)),
                    ledger_residual_rel=float(p.ledger()["residual_rel"])),
                "bond": dict(active=w.bond.active, broken=w.bond.broken, capacity_n=float(w.bond.capacity_n)),
                "robot": {"reflexes": list(w.reflexes[-10:]), "refused_targets": int(w.refused_targets),
                          "stopped_arms": sorted(a for a, u in w._reflex_until.items() if w.time < u)},
                "history": dict(self.history),
                "events": list(w.events[-20:]),
                "terminated": self.sim_time >= TIME_LIMIT_S.get(self.config.task, 1e9),
                "evaluation": ev,
            }

    def close(self):
        """Release this environment's world; the env cannot be used afterwards (closed is True, world is None).

        The world sits in reference cycles -- its Kinematics and its Bond keep a reference to it, the weld pool's
        obstacle callback is one of its bound methods, the evaluator refers to this env -- so dropping the last outside
        reference would leave it, with its Warp arrays (about 290 MB of GPU memory) and about 400 MB of host memory, to
        Python's cyclic garbage collector, which a served session hardly ever runs: every catalogue variation switch
        would keep the world it replaced (+256 MB GPU and +395 MB host per switch, no plateau). close() drops the
        references this env and its world hold, so reference counting frees the weld process and its Warp arrays (the
        GPU memory) as soon as the caller lets go of the env. The emptied world object, Newton's model, the MuJoCo model
        and data and the solver stay in reference cycles inside Newton (closures that refer back to the world among
        them) that only the cyclic collector frees: the backend collects once after a catalogue switch
        (newton_welding.py _start_variation)."""
        with self.lock:
            if getattr(self, "closed", False):
                return
            w = self.__dict__.get("world")
            if w is not None:
                for name in ("kin", "bond", "process"):         # built by this world for itself (World.__init__)
                    part = w.__dict__.get(name)
                    if part is not None and hasattr(part, "__dict__"):
                        part.__dict__.clear()
                w.__dict__.clear()
            evaluator = self.__dict__.get("evaluator")
            if evaluator is not None:
                evaluator.__dict__.clear()
            lock = self.lock
            self.__dict__.clear()
            self.lock, self.world, self.closed = lock, None, True

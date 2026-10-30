"""Wine-pouring environment: world + liquid + task layout + evaluator; state, actions, checkpoints.

reset(): build the bar scene for (preset, task, seed), let the glassware settle on the bar (rigid), fill the task's
containers with liquid (liquid.Liquid.fill_container), prepare that liquid at rest (zero-g packing relaxation,
hydrostatic pressure, damped settle) and hand over the scene at t = 0. The liquid is then near equilibrium, not at it:
over the first ~5 s particles in a deep bottle column still move up to 2 cm (mean 1 mm) as the packing relaxes, with
no change in any container's contents (E0).
"""

import copy
import threading
import time
import numpy as np

from . import params as P
from .config import Config
from .layout import task_layout


class WineEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = config or Config()
        self.output_dir = output_dir
        self.lock = threading.RLock()
        self.generation = 0
        self.trace_callback = None
        self.renderer_factory = None
        self.render_meta = None
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self):
        from .world import World
        from .task import Evaluator
        t0 = time.perf_counter()
        self.layout = task_layout(self.config)
        self.world = World(self.config)
        self.timing = dict(build_s=time.perf_counter() - t0)
        self._settle_rigid()
        if self.config.liquid:
            self._fill()
        self.evaluator = Evaluator(self)
        self._still_since = 0.
        self._last_targets = self.world.targets.copy()
        # service rule (sommelier standard): the bottle never touches the glass. Watched every 10 ticks
        # for presses above 0.5 N; split-bottle, whose instruction states the rule, every tick for any touch
        # (task.touch_rule). Read-only: MuJoCo's contact list and mj_contactForce, nothing written back
        from .task import touch_rule
        lay = self.layout
        src = next((i["name"] for i in lay["items"] if i.get("role") == "source"), None)
        tg = [i["name"] for i in lay["items"] if i.get("role") in ("target", "dump")]
        sb = self.world.scene.bodies
        self._touch_pair = ([sb[src]], [sb[n] for n in tg]) if src and tg else None
        self._touch_every, self._touch_floor_n = touch_rule(self.config.task)
        self.source_touch = dict(max_n=0., ticks=0)
        self._initial = self.checkpoint()

    def _settle_rigid(self):
        w = self.world
        t0 = time.perf_counter()
        for _ in range(int(round(self.config.settle_rigid_s/self.config.rigid_dt))):
            w.step()
        w.tick = 0
        self.timing["settle_rigid_s"] = time.perf_counter() - t0

    def _fill(self):
        """Particles into every container the layout fills, at its settled pose; then the at-rest preparation."""
        from scipy.spatial.transform import Rotation
        w = self.world
        liq = w.attach_liquid()
        q = w.state.body_q.numpy()
        xs, lids, groups, n0 = [], [], [], 0
        for item in self.layout["items"]:
            if not item.get("liquid") or not item.get("volume"):
                continue
            body = w.scene.bodies[item["name"]]
            geo = self.geometry(item["name"])
            x, lid = liq.fill_container(geo, q[body, :3], q[body, 3:], item["volume"], item["liquid"],
                                        seed=self.config.seed)
            xs.append(x); lids.append(lid)
            groups.append(np.arange(n0, n0 + len(x)))
            n0 += len(x)
        if not xs:
            return
        t0 = time.perf_counter()
        liq.set_particles(np.concatenate(xs), np.zeros((n0, 3), np.float32), np.concatenate(lids))
        liq.prepare(groups, self.config.liquid_packing_steps, self.config.liquid_settle_steps)
        self.timing["liquid_prepare_s"] = time.perf_counter() - t0
        self.timing["particles"] = int(n0)

    def geometry(self, name):
        from . import geometry as GEO
        it = self.world.scene.items[name]
        return GEO.get(it["kind"], it.get("color"))

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        snap = dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), tick=w.tick,
                    grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), grip_ref=dict(w._grip_ref),
                    reflex_until=dict(w._reflex_until), reflex_fired=dict(w._reflex_fired),
                    reflexes=copy.deepcopy(w.reflexes), refused=w.refused_targets, still_since=getattr(self, "_still_since", 0.),
                    servo=copy.deepcopy(w.servo_state()),
                    source_touch=dict(getattr(self, "source_touch", {"max_n": 0., "ticks": 0})),
                    mj=(w.solver.mj_data.qpos.copy(), w.solver.mj_data.qvel.copy(), w.solver.mj_data.qacc_warmstart.copy()))
        if getattr(w, "lumped", None) is not None:
            snap["lumped"] = w.lumped.body_inertia_state()
        if w.gas is not None and w.gas.enabled:
            g = w.gas
            snap["gas"] = dict(sealed=list(g.sealed), p=list(g.p), p0=list(g.p0), vg0=list(g.vg0), vl_f=list(g.vl_f),
                               n_atm=list(g.n_atm))
        if w.liquid is not None:
            L = w.liquid
            snap["liquid"] = dict(n=L.n, x=L.x.numpy()[:L.n].copy(), v=L.v.numpy()[:L.n].copy(), lid=L.lid.numpy()[:L.n].copy(),
                                  kd=L.kacc[True].numpy()[:L.n].copy(), kv=L.kacc[False].numpy()[:L.n].copy(),
                                  dt_prev=L.dt_prev, time=L.time)
        return snap

    def restore(self, snap):
        with self.lock:
            w = self.world
            for a, v in snap["rigid"].items():
                getattr(w.state, a).assign(v)
                getattr(w.next_state, a).assign(v)
            w.targets = snap["targets"].copy()
            w.control.joint_target_q.assign(snap["control"])
            if "servo" in snap:
                w.set_servo_state(snap["servo"])
            w.tick = snap["tick"]
            w.grip_force, w.grasp_latched, w._grip_ref = dict(snap["grip_force"]), dict(snap["grasp_latched"]), dict(snap["grip_ref"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w.reflexes, w.refused_targets = copy.deepcopy(snap["reflexes"]), snap["refused"]
            w.fault = None
            qpos, qvel, qacc = snap["mj"]
            d = w.solver.mj_data
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = qpos, qvel, qacc
            w._mujoco.mj_forward(w.solver.mj_model, d)
            self._still_since = snap.get("still_since", 0.)
            self.source_touch = dict(snap.get("source_touch", {"max_n": 0., "ticks": 0}))
            if "liquid" in snap and w.liquid is not None:
                s = snap["liquid"]
                L = w.liquid
                L.set_particles(s["x"], s["v"], s["lid"])
                L.kacc[True].assign(s["kd"]); L.kacc[False].assign(s["kv"])
                L.dt_prev, L.time = s["dt_prev"], s["time"]
                w._push_bodies()
            if "lumped" in snap and getattr(w, "lumped", None) is not None:
                w.lumped.m_l, w.lumped.a_lag = snap["lumped"]["m_l"].copy(), snap["lumped"]["a_lag"].copy()
                w.lumped.res = snap["lumped"]["res"].copy()
                w.lumped.al_lag = snap["lumped"]["al_lag"].copy()
                w.lumped.w_prev = None if snap["lumped"]["w_prev"] is None else snap["lumped"]["w_prev"].copy()
                w.lumped.v_prev = None if snap["lumped"]["v_prev"] is None else snap["lumped"]["v_prev"].copy()
                vzp, zcp = snap["lumped"].get("vz_prev"), snap["lumped"].get("zc_prev")
                w.lumped.vz_prev = None if vzp is None else vzp.copy()
                qp, qdp = snap["lumped"].get("q_prev"), snap["lumped"].get("qd_prev")
                w.lumped.q_prev = None if qp is None else qp.copy()
                w.lumped.qd_prev = None if qdp is None else qdp.copy()
                w.lumped.zc_prev = None if zcp is None else zcp.copy()
            if "gas" in snap and w.gas is not None and w.gas.enabled:
                g = w.gas
                g.sealed, g.p, g.p0, g.vg0, g.vl_f, g.n_atm = (list(snap["gas"][k]) for k in
                                                               ("sealed", "p", "p0", "vg0", "vl_f", "n_atm"))

    def reset(self):
        with self.lock:
            self.restore(self._initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, guard=None):
        """Run `ticks` physics ticks (rigid_dt each, liquid included) with new joint targets."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            w.set_targets(targets, ramp_ticks=ticks)
            for _ in range(ticks):
                w.step()
                self._track_still()
                if self._touch_pair and w.tick % self._touch_every == 0:
                    f = sum(w.pair_contact_forces(*self._touch_pair).values())
                    if f > self._touch_floor_n:
                        self.source_touch["max_n"] = max(self.source_touch["max_n"], round(float(f), 3))
                        self.source_touch["ticks"] += self._touch_every
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state()

    def _track_still(self):
        w = self.world
        qd = w.state.joint_qd.numpy()
        moving = any(abs(float(qd[s["dof"]])) > 0.01 for n, s in w.joints.items() if "finger" not in n)
        changed = not np.array_equal(w.targets, self._last_targets)
        if moving or changed:
            self._still_since = w.sim_time
        self._last_targets = w.targets.copy()

    def robot_still_s(self):
        return self.world.sim_time - self._still_since

    def robot_contacts(self):
        """Normal force (N) between the robot and each glassware item."""
        w = self.world
        items = {name: it["body"] for name, it in w.scene.items.items()}
        f = w.pair_contact_forces(w.scene.robot_bodies, items.values())
        inv = {b: n for n, b in items.items()}
        out = {n: 0. for n in items}
        for (rb, ib), v in f.items():
            out[inv[ib]] += v
        return {k: round(v, 3) for k, v in out.items() if v > P.CONTACT_FORCE_TOL_N}

    # ------------------------------------------------------------------ state
    def inventory(self):
        from .inventory import summarize
        w = self.world
        if w.liquid is None or not w.liquid.n:
            return dict(containers={}, bar={}, floor={}, elsewhere={}, film={}, total_mL=0., speed_mean_m_s=0., speed_max_m_s=0.)
        return summarize(w.liquid, w)

    def get_state(self):
        with self.lock:
            w = self.world
            q = w.state.joint_q.numpy()
            qd = w.state.joint_qd.numpy()
            joints = {n: {"position": float(q[s["q"]]), "velocity": float(qd[s["dof"]])} for n, s in w.joints.items()}
            bq = w.state.body_q.numpy()
            items = {}
            for name, it in w.scene.items.items():
                p, R = w.body_pose(name)
                items[name] = dict(kind=it["kind"], position_m=[round(float(x), 5) for x in p],
                                   tilt_deg=round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 3))
            finite = bool(np.isfinite(bq).all() and np.isfinite(q).all())
            ev = None
            if not w.fault and finite:
                try:
                    ev = self.evaluator.evaluate()
                except Exception as exc:          # never let a scoring bug hide the physical state
                    ev = dict(success=False, score=0., error=f"{type(exc).__name__}: {exc}")
            liq = w.liquid
            return dict(tick=int(w.tick), time_s=round(w.sim_time, 6), generation=self.generation, joints=joints,
                        items=items, ready=True, finite=finite, fault=w.fault,
                        liquid=dict(particles=int(liq.n) if liq is not None else 0,
                                    last_step=dict(liq.last) if liq is not None else {},
                                    headspace=w.gas.state() if w.gas is not None else {}),
                        evaluation=ev or dict(success=False, score=0.))

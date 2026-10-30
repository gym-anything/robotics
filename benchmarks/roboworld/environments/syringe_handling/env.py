"""Syringe-handling environment: world + start state + evaluator; state, actions, checkpoints.

reset(): build the bay for (task, seed, variant); let everything loose come to rest (syringes on the tray's ribs, the
vial, the trainer); hand over at t = 0 with the liquid ledger, the records of touches and every timer cleared.
"""

import copy
import threading
import time
import numpy as np

from . import params as P
from . import task as T
from .config import Config

# what the equipment writes into the compiled model at run time (fits, friction limits, armatures, lumped liquid)
MODEL_FIELDS = ("dof_armature", "dof_frictionloss", "dof_invweight0", "eq_data", "eq_solref", "geom_contype", "geom_conaffinity",
                "body_mass", "body_subtreemass", "body_ipos", "body_inertia", "jnt_actfrcrange")


class SyringeEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = config or Config()
        self.output_dir = output_dir
        self.lock = threading.RLock()
        self.generation = 0
        self.trace_callback = None
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self):
        from .world import World
        t0 = time.perf_counter()
        self.world = w = World(self.config)
        self.layout = w.scene.layout
        self.spec = T.task_spec(self.config, self.layout)
        self.timing = dict(build_s=time.perf_counter() - t0)
        self._settle()
        fl = w.fluid
        self.start = T.start_record(w)
        # the start: nothing has been touched, nothing has flowed, no timer has run
        w.tick = 0
        w.reflexes, w.events = [], []
        fl.ledger.arrivals, fl.ledger.events = {}, []
        for s in fl.syringes.values():
            s.touch, s.chamber.peak_gauge = {}, 0.0
            s.rest_s = 1e9
        for n in fl.needles.values():
            n.touch, n.moment_peak, n.records = {}, 0.0, []
        for lk in fl.locks.values():
            lk.peak_gauge, lk.max_shift = 0.0, 0.0
            lk.pose0 = (w.bq[lk.body, :3].copy(), lk.pose0[1])
        fl.initial = fl.totals()
        self._history = T.History(w)
        self._still_since = 0.
        self._last_targets = w.targets.copy()
        self._initial = self.checkpoint()

    def _settle(self):
        """Hand over at rest: every free body below 0.1 mm/s at three checks 0.1 s apart (drilling's criterion); a
        start state that has not come to rest in 6 s is handed over as it is, and its record says so."""
        w = self.world
        t0 = time.perf_counter()
        objs = w._objects
        start, quiet, first = w.tick, 0, True
        limit = self.config.settle_s + 6.0
        while True:
            for _ in range(int(round((max(0.3, self.config.settle_s) if first else 0.1)/w.dt))):
                w.step()
            first = False
            used = (w.tick - start)*w.dt
            vmax = np.linalg.norm(w.bqd[objs, :3], axis=1).max(initial=0.)
            quiet = quiet + 1 if vmax < 1e-4 else 0
            if quiet >= 3 or used >= limit:
                break
        self.settle = dict(sim_s=round(used, 3), max_body_speed_m_s=round(float(vmax), 6), at_rest=bool(quiet >= 3))
        self.timing["settle_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        d, m = w.solver.mj_data, w.solver.mj_model
        return dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), tq=w.tq.copy(), tqd=w.tqd.copy(), tick=w.tick,
                    grip_force=dict(getattr(w, "grip_force", {})), grasp_latched=dict(getattr(w, "grasp_latched", {})),
                    grip_ref=dict(getattr(w, "_grip_ref", {})), reflex_until=dict(w._reflex_until),
                    reflex_fired=dict(w._reflex_fired), reflexes=copy.deepcopy(w.reflexes), refused=w.refused_targets,
                    controller=w.controller_state() if w.rig else None, events=copy.deepcopy(w.events),
                    still_since=getattr(self, "_still_since", 0.),
                    mj=(d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.act.copy(), d.eq_active.copy()),
                    model={f: getattr(m, f).copy() for f in MODEL_FIELDS}, invweight=dict(w._invweight),
                    fluid=w.fluid.snapshot(), history=self._history.snapshot())

    def restore(self, snap):
        with self.lock:
            w = self.world
            for a, v in snap["rigid"].items():
                getattr(w.state, a).assign(v)
                getattr(w.next_state, a).assign(v)
            w.targets, w.tq, w.tqd = snap["targets"].copy(), snap["tq"].copy(), snap["tqd"].copy()
            w.tick = snap["tick"]
            if w.rig:
                w.grip_force, w.grasp_latched = dict(snap["grip_force"]), dict(snap["grasp_latched"])
                w._grip_ref = dict(snap["grip_ref"])
                w.set_controller_state(snap["controller"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w.reflexes, w.refused_targets = copy.deepcopy(snap["reflexes"]), snap["refused"]
            w.events = copy.deepcopy(snap["events"])
            w.fault = None
            w.body_wrench[:] = 0.
            d, m = w.solver.mj_data, w.solver.mj_model
            arm_changed = not np.array_equal(m.dof_armature, snap["model"]["dof_armature"])
            for f, v in snap["model"].items():
                getattr(m, f)[:] = v
            w._invweight = dict(snap["invweight"])
            if arm_changed:                      # an armature acts only after mj_setConst (world._apply_const)
                w._arm_pending = {i: float(v) for i, v in enumerate(snap["model"]["dof_armature"])}
                w._apply_const()
                m.dof_invweight0[:] = snap["model"]["dof_invweight0"]
            qpos, qvel, qacc, act, eq_active = snap["mj"]
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:], d.act[:], d.eq_active[:] = qpos, qvel, qacc, act, eq_active
            w._eq_rows_tick = -1
            w._mujoco.mj_forward(m, d)
            w.refresh()
            w.fluid.restore(w, copy.deepcopy(snap["fluid"]))
            w._arm_pending = {}
            self._history.restore(snap["history"])
            self._still_since = snap.get("still_since", 0.)

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
        """Run `ticks` physics ticks with new joint targets."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            w.set_targets(targets)
            for _ in range(ticks):
                w.step()
                self._after_tick()
                if guard and guard(self):
                    break
            return self.get_state(evaluate=False)

    def run(self, ticks):
        """Advance without changing targets (probes/demos)."""
        with self.lock:
            for _ in range(ticks):
                self.world.step()
                self._after_tick()

    def _after_tick(self):
        self._history.tick(self.world)
        self._track_still()
        if self.trace_callback:
            self.trace_callback(self)

    def _track_still(self):
        w = self.world
        if not w.rig:
            return
        moving = any(abs(float(w.jqd[s["dof"]])) > 0.01 for n, s in w.joints.items() if "finger" not in n)
        changed = not np.array_equal(w.targets, self._last_targets)
        if moving or changed:
            self._still_since = w.sim_time
        self._last_targets = w.targets.copy()

    def robot_still_s(self):
        return self.world.sim_time - self._still_since

    # ------------------------------------------------------------------ state
    def objects_state(self):
        w = self.world
        out = {}
        for name, body in w.scene.bodies.items():
            p, R = w.body_pose(body)
            out[name] = dict(position_m=[round(float(x), 5) for x in p],
                             rotation=[[round(float(v), 6) for v in row] for row in R],
                             speed_m_s=round(float(np.linalg.norm(w.bqd[body, :3])), 5))
        return out

    def joint_torques(self):
        """Measured joint torques (the FR3's joint torque sensors): actuator plus fed-forward torque per arm joint."""
        w = self.world
        if not w.rig:
            return {}
        d = w.solver.mj_data
        out = {}
        for arm, dofs in w._arm_mj_dofs.items():
            for name, mdof in dofs:
                out[name] = float(d.qfrc_actuator[mdof] + d.qfrc_applied[mdof])
        return out

    def equipment_state(self):
        """Owner-only: the liquid, the fits, the needles, the records."""
        w, fl = self.world, self.world.fluid
        led = fl.ledger
        return dict(syringes={n: s.reading(w) for n, s in fl.syringes.items()},
                    needles={n: s.reading(w) for n, s in fl.needles.items()},
                    vials={n: s.reading(w) for n, s in fl.vials.items()},
                    locks={n: s.reading(w) for n, s in fl.locks.items()},
                    places_ml={k: {sp: round(float(v[i])/1e-6, 5) for i, sp in enumerate(P.SPECIES + ("air",)) if abs(v[i]) > 1e-13}
                               for k, v in led.places.items()},
                    airborne_ml=round(fl.spray.airborne()/1e-6, 5), hanging_ml=round(fl.spray.hanging()/1e-6, 5),
                    conservation_error_ml=round(fl.conservation_error_ml(), 6),
                    sharps_flap_deg=round(float(np.degrees(w.jq[w.scene.flap_joint.q])), 2) if hasattr(w.scene, "flap_joint") else None)

    def evaluate(self):
        with self.lock:
            return T.evaluate(self)

    def get_state(self, evaluate=True):
        with self.lock:
            w = self.world
            joints = {n: {"position": float(w.jq[s["q"]]), "velocity": float(w.jqd[s["dof"]])} for n, s in w.joints.items()}
            finite = bool(np.isfinite(w.bq).all() and np.isfinite(w.jq).all())
            ev = None
            if evaluate and not w.fault and finite:
                try:
                    ev = self.evaluate()
                except Exception as exc:          # never let a scoring bug hide the physical state
                    ev = dict(success=False, error=f"{type(exc).__name__}: {exc}")
            state = dict(tick=int(w.tick), time_s=round(w.sim_time, 6), generation=self.generation, joints=joints,
                         joint_torques_nm=self.joint_torques(), objects=self.objects_state(),
                         equipment=self.equipment_state() if finite else None,
                         ready=True, finite=finite, fault=w.fault, reflexes=list(w.reflexes[-20:]),
                         events=list(w.events[-40:]), load_kg=dict(getattr(w, "load_kg", {})))
            if w.rig:
                state["grip"] = {arm: dict(force_n=w.grip_force.get(arm, 0.), applied_n=round(w.grip_applied_n(arm), 3))
                                 for arm in w.grip_force}
            if ev is not None:
                state["evaluation"] = ev
            return state

"""Drilling environment: world + task layout + evaluator; state, actions, checkpoints.

reset(): build the station for (preset, task, seed), prepare the start state (a pre-clamped vise is closed by an
ASSISTED crank torque that is then removed; the thread friction holds the clamp), let everything settle until at rest
(at least 1 s, then 0.1 s at a time until every free body is below 2 mm/s, at most 5 s) and hand
over the scene at t = 0.
"""

import copy
import threading
import time
import numpy as np

from . import params as P
from .config import Config


class DrillingEnv:
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
        from .task import Evaluator
        t0 = time.perf_counter()
        self.world = World(self.config)
        self.layout = self.world.scene.layout
        self.timing = dict(build_s=time.perf_counter() - t0)
        self._prepare()
        self._settle()
        self.evaluator = Evaluator(self)
        self._still_since = 0.
        self._last_targets = self.world.targets.copy()
        self._initial = self.checkpoint()

    def _prepare(self):
        """ASSISTED start-state preparation (never reachable from the policy API): a vise that the task starts closed
        is cranked with VISE_PRECLAMP_NM through its screw for 0.5 s; then the torque is removed."""
        w = self.world
        L = self.layout
        if L["preset"] != "bench" or L["task"] != "press-blind":
            return
        vise = w.scene.tools["press_vise"]
        dof = vise.j["screw"].dof
        t0 = time.perf_counter()

        def crank(world, wrench):
            world.jf[dof] -= P.VISE_PRECLAMP_NM          # clockwise from the crank closes (tools.Vise)
            return wrench
        for _ in range(int(0.5/w.dt)):
            w.jf[:] = 0.
            w.fixture = crank
            w.step()
        w.fixture = None
        self.timing["preclamp_s"] = time.perf_counter() - t0
        self.preclamp = dict(torque_nm=P.VISE_PRECLAMP_NM, clamp_force_n=round(vise.s_clamp_n, 1))

    def _settle(self):
        w = self.world
        t0 = time.perf_counter()
        objs = np.setdiff1d(w._objects, w._spin_exempt)
        used, quiet = 0., 0
        # hand over at rest: every free body below 0.1 mm/s (the E0 rest criterion) at three checks 0.1 s apart. A
        # single check at 2 mm/s handed over a hole brush still rocking 0.5 mm on its handle and rod
        while True:
            for _ in range(int(round((1.0 if used == 0 else 0.1)/w.dt))):
                w.step()
            used = w.sim_time
            v = np.linalg.norm(w.bqd[objs, :3], axis=1).max(initial=0.)
            quiet = quiet + 1 if v < 1e-4 else 0
            if quiet >= 3 or used >= self.config.settle_s + 4.:
                break
        self.settle = dict(settle_sim_s=round(used, 3), max_speed_m_s=round(float(v), 5))
        w.tick = 0
        w.reflexes = []
        w.events = []
        self.timing["settle_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        d = w.solver.mj_data
        snap = dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), tq=w.tq.copy(), tqd=w.tqd.copy(), tick=w.tick,
                    grip_force=dict(getattr(w, "grip_force", {})), grasp_latched=dict(getattr(w, "grasp_latched", {})),
                    grip_ref=dict(getattr(w, "_grip_ref", {})), reflex_until=dict(w._reflex_until),
                    reflex_fired=dict(w._reflex_fired), reflexes=copy.deepcopy(w.reflexes), refused=w.refused_targets,
                    controller=w.controller_state() if w.rig else None, geometry=w.geometry_state(),
                    events=copy.deepcopy(w.events), still_since=getattr(self, "_still_since", 0.),
                    over_payload_s=dict(getattr(w, "over_payload_s", {})),
                    mj=(d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.act.copy() if d.act is not None else None),
                    holes=w.hm.snapshot(), tools={n: copy.deepcopy(t.snapshot()) for n, t in w.scene.tools.items()},
                    evaluator=copy.deepcopy(self.evaluator.snapshot()) if hasattr(self, "evaluator") else None)
        return snap

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
            w.set_geometry_state(snap["geometry"])
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w.reflexes, w.refused_targets = copy.deepcopy(snap["reflexes"]), snap["refused"]
            w.events = copy.deepcopy(snap["events"])
            w.over_payload_s = dict(snap.get("over_payload_s", {}))
            w.fault = None
            w.body_wrench[:] = 0.
            qpos, qvel, qacc, act = snap["mj"]
            d = w.solver.mj_data
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = qpos, qvel, qacc
            if act is not None:
                d.act[:] = act
            w._mujoco.mj_forward(w.solver.mj_model, d)
            w.refresh()
            for n, t in w.scene.tools.items():
                t.restore(copy.deepcopy(snap["tools"][n]))
            w.hm.restore(snap["holes"])
            self._still_since = snap.get("still_since", 0.)
            if snap.get("evaluator") is not None:
                self.evaluator.restore(copy.deepcopy(snap["evaluator"]))

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
                self.evaluator.tick()
                self._track_still()
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state()

    def run(self, ticks):
        """Advance without changing targets (probes/demos)."""
        with self.lock:
            for _ in range(ticks):
                self.world.step()
                self.evaluator.tick()
                if self.trace_callback:
                    self.trace_callback(self)

    def _track_still(self):
        w = self.world
        moving = any(abs(float(w.jqd[s["dof"]])) > 0.01 for n, s in w.joints.items() if "finger" not in n)
        changed = not np.array_equal(w.targets, self._last_targets)
        if moving or changed:
            self._still_since = w.sim_time
        self._last_targets = w.targets.copy()

    def robot_still_s(self):
        return self.world.sim_time - self._still_since

    # ------------------------------------------------------------------ state
    def parts_state(self):
        w = self.world
        out = {}
        for name, info in w.scene.parts.items():
            p, R = w.body_pose(info["body"])
            out[name] = dict(kind=info["kind"], position_m=[round(float(x), 5) for x in p],
                             rotation=[[round(float(v), 6) for v in row] for row in R])
        return out

    def get_state(self):
        with self.lock:
            w = self.world
            joints = {n: {"position": float(w.jq[s["q"]]), "velocity": float(w.jqd[s["dof"]])} for n, s in w.joints.items()}
            finite = bool(np.isfinite(w.bq).all() and np.isfinite(w.jq).all())
            from .task import NO_SCORE_WHEN_UNEVALUABLE
            # a state that cannot be evaluated is no score, not score 0, where the task says so (task.py)
            unscored = None if self.layout["task"] in NO_SCORE_WHEN_UNEVALUABLE else 0.
            ev = None
            if not w.fault and finite:
                try:
                    ev = self.evaluator.evaluate()
                except Exception as exc:          # never let a scoring bug hide the physical state
                    ev = dict(success=False, score=unscored, error=f"{type(exc).__name__}: {exc}")
            elif unscored is None:
                ev = dict(success=False, score=None, evaluated=False, reason=w.fault or "non-finite state")
            return dict(tick=int(w.tick), time_s=round(w.sim_time, 6), generation=self.generation, joints=joints,
                        parts=self.parts_state(), tools={n: t.state() for n, t in w.scene.tools.items()},
                        holes=w.hm.summary(), bits=w.hm.item_summary(), ready=True, finite=finite, fault=w.fault,
                        reflexes=list(w.reflexes[-20:]), events=list(w.events[-20:]) + list(w.hm.events[-20:]),
                        evaluation=ev or dict(success=False, score=0.))

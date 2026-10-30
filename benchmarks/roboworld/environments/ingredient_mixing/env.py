"""Ingredient-mixing environment: world + start state + evaluator; state, actions, checkpoints.

reset(): build the station for (preset, task, seed, variant); let the rigid scene settle (tools come to rest in the
crock); seed the task's ingredients into the vessel where it has come to rest; let everything settle again; hand over
at t = 0.
"""

import copy
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import task as T
from .config import Config

CONT_ARRAYS = ("x", "xr", "v", "aff", "dev", "phi", "jac", "flags", "spec", "temp", "state", "work", "gdot", "visc",
               "lam_n", "lam_l", "lam_on", "mass", "vol0", "kbulk", "shear", "cons", "nidx", "tauy", "mudp", "wet",
               "heatcap", "dmicro", "dgrain", "free_liq", "dry_pore", "expo", "xout", "tout", "nuw", "fowner", "foff", "ftime", "mfilm")


class MixEnv:
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
        self.spec = T.task_spec(self.config)
        self.timing = dict(build_s=time.perf_counter() - t0)
        self._settle(rigid_only=True)
        self.info = None
        if self.config.material:
            t1 = time.perf_counter()
            st = T.start_state(self.config, w.scene)
            v = w.scene.vessel
            q = w.bq[v.body]
            pts = q[:3] + Rotation.from_quat(q[3:]).apply(st["points"])
            cont = w.attach_material(st["recipe"], pts, st["species"], temp=st["temp"])
            self.info = dict(batch=st["batch"], spacing=st["spacing"], points=int(len(pts)),
                             volume_m3=float(cont.read("vol0").sum()), start_level_m=st["level"],
                             recipe=st["recipe"].as_dict())
            self.timing["seed_s"] = time.perf_counter() - t1
            self._settle(rigid_only=False)
        w.scene.vessel_start = w.bq[w.scene.vessel.body, :3].copy()
        w.tick = 0
        w.reflexes, w.events = [], []
        if w.cont is not None:
            w.cont.time, w.cont.ticks = 0., 0
        self._still_since = 0.
        self._last_targets = w.targets.copy()
        self._initial = self.checkpoint()

    def _settle(self, rigid_only):
        """Hand over at rest: every free body below 0.1 mm/s at three checks 0.1 s apart (drilling's criterion), and,
        once the ingredients are in, their points too: the 99th percentile of their speed below 1 mm/s at the same
        three checks. Heaps and spoonfuls are put in as shapes (task._Fill) and take what their strength allows here;
        a start state that has not come to rest in 12 s is handed over as it is, and its record says so. The record
        also gives the 99.9th percentile and the fastest point: liquid soaking into powder and a spoonful of honey
        spreading go on for minutes, and "at rest" here means the 99 % only."""
        w = self.world
        t0 = time.perf_counter()
        objs = np.setdiff1d(w._objects, w._spin_exempt)
        start, quiet = w.tick, 0
        limit = (self.config.settle_s + 6.0) if rigid_only else (self.config.settle_s + 11.5)
        first = True
        pmax, p999, pfast = 0.0, 0.0, 0.0
        while True:
            for _ in range(int(round((max(0.3, self.config.settle_s) if first else 0.1)/w.dt))):
                w.step()
            first = False
            used = (w.tick - start)*w.dt
            vmax = np.linalg.norm(w.bqd[objs, :3], axis=1).max(initial=0.)
            if not rigid_only and w.cont is not None and w.cont.n:
                speed = np.linalg.norm(w.cont.read("v"), axis=1)
                pmax, p999, pfast = float(np.percentile(speed, 99)), float(np.percentile(speed, 99.9)), float(speed.max())
            quiet = quiet + 1 if (vmax < 1e-4 and pmax < 1e-3) else 0
            if quiet >= 3 or used >= limit:
                break
        key = "settle_rigid" if rigid_only else "settle_material"
        rec = dict(sim_s=round(used, 3), max_body_speed_m_s=round(float(vmax), 6))
        if not rigid_only:
            rec.update(point_speed_p99_mm_s=round(pmax*1e3, 3), point_speed_p999_mm_s=round(p999*1e3, 3),
                       point_speed_max_mm_s=round(pfast*1e3, 3), at_rest=bool(quiet >= 3))
        setattr(self, key, rec)
        self.timing[key + "_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        w = self.world
        d = w.solver.mj_data
        m = w.solver.mj_model
        snap = dict(rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
                    targets=w.targets.copy(), tq=w.tq.copy(), tqd=w.tqd.copy(), tick=w.tick,
                    grip_force=dict(getattr(w, "grip_force", {})), grasp_latched=dict(getattr(w, "grasp_latched", {})),
                    grip_ref=dict(getattr(w, "_grip_ref", {})), reflex_until=dict(w._reflex_until),
                    reflex_fired=dict(w._reflex_fired), reflexes=copy.deepcopy(w.reflexes), refused=w.refused_targets,
                    controller=w.controller_state() if w.rig else None, events=copy.deepcopy(w.events),
                    still_since=getattr(self, "_still_since", 0.),
                    mj=(d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.act.copy() if d.act is not None else None),
                    inertial=(m.body_mass.copy(), m.body_ipos.copy(), m.body_inertia.copy()),
                    tools={n: copy.deepcopy(t.snapshot()) for n, t in w.scene.tools.items()})
        if w.cont is not None:
            c = w.cont
            snap["cont"] = dict(arrays={a: getattr(c, a).numpy().copy() for a in CONT_ARRAYS}, n=c.n, time=c.time,
                                ticks=c.ticks, origin=c.origin.copy(), grid=(c.nx, c.ny, c.nz), counts=c.counts.numpy().copy(),
                                lump={k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w._lump.items()},
                                tool_res={k: v.copy() for k, v in w._tool_res.items()})
            if w.heat is not None:
                snap["heat"] = w.heat.snapshot()
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
            w._reflex_until, w._reflex_fired = dict(snap["reflex_until"]), dict(snap["reflex_fired"])
            w.reflexes, w.refused_targets = copy.deepcopy(snap["reflexes"]), snap["refused"]
            w.events = copy.deepcopy(snap["events"])
            w.fault = None
            w.body_wrench[:] = 0.
            qpos, qvel, qacc, act = snap["mj"]
            d, m = w.solver.mj_data, w.solver.mj_model
            m.body_mass[:], m.body_ipos[:], m.body_inertia[:] = snap["inertial"]
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = qpos, qvel, qacc
            if act is not None:
                d.act[:] = act
            w._mujoco.mj_forward(m, d)
            w.refresh()
            for n, t in w.scene.tools.items():
                t.restore(copy.deepcopy(snap["tools"][n]))
            if w.cont is not None and "cont" in snap:
                c, s = w.cont, snap["cont"]
                for a, v in s["arrays"].items():
                    getattr(c, a).assign(v)
                c.n, c.time, c.ticks, c.origin = s["n"], s["time"], s["ticks"], s["origin"].copy()
                c.nx, c.ny, c.nz = s["grid"]
                c.nodes = c.nx*c.ny*c.nz
                if c.nodes > c.nodes_alloc:
                    c._alloc_grid(c.nodes)
                c.counts.assign(s["counts"])
                c.gexpo.zero_()
                w._lump = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s["lump"].items()}
                w._tool_res = {k: v.copy() for k, v in s["tool_res"].items()}
                if w.heat is not None and "heat" in snap:
                    w.heat.restore(snap["heat"])
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
                self._track_still()
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state(evaluate=False)

    def run(self, ticks):
        """Advance without changing targets (probes/demos)."""
        with self.lock:
            for _ in range(ticks):
                self.world.step()
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

    def evaluate(self):
        with self.lock:
            w = self.world
            if w.cont is None or self.info is None:
                return dict(success=False, score=0.0, criteria={}, note="no mixture in this build")
            return T.evaluate(w, self.info)

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
                         tools={n: t.state() for n, t in w.scene.tools.items() if t.state()},
                         ready=True, finite=finite, fault=w.fault, reflexes=list(w.reflexes[-20:]),
                         events=list(w.events[-20:]), load_kg=dict(getattr(w, "load_kg", {})))
            if w.rig:
                state["grip"] = {arm: dict(force_n=w.grip_force.get(arm, 0.), applied_n=round(w.grip_applied_n(arm), 3))
                                 for arm in w.rig.instances}
            if w.heat is not None:
                state["hob"] = w.heat.state()
            if ev is not None:
                state["evaluation"] = ev
            return state

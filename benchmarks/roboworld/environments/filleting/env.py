"""Filleting environment wrapper (the GAR backend's engine): the coupled world, its settled initial state and
checkpoints, action execution, histories for the verifier, and the state dictionary.

Initial condition (disclosed): the knife is spawned in the right hand's insert grasp and the
world settles hands-off before the episode starts (coupon 0.5 s; anatomical workpieces 1.5 s, the first 1.2 s with
dynamic relaxation so the workpiece starts at rest on the board: `_settle`, tissue.relax).
"""

import copy
import math
import threading
import time
from collections import deque
from dataclasses import asdict

import numpy as np

from . import layout as LY
from .world import World, Config
from .task import (Evaluator, INSTRUCTIONS, TIME_LIMIT_S, WORKPIECE_OF, WORKPIECE_OF_TASK, RECEPTACLES_OF_TASK,
                   STOP_AT_TIME_LIMIT, PROGRESS_SCORED, EDGE_REACH_M, near_boxes, edge_distance)


class FilletingEnv:
    SETTLE_S = .5
    MOTION_EVERY = 100          # ticks between motion-history samples (0.1 s)
    SPEED_EVERY = 10            # ticks between samples of the fastest tissue node (thrown-tissue check)
    JMIN_EVERY = 20             # ticks between per-tet compression checks

    def __init__(self, config=None, *, output_dir=None):
        c = config or Config()
        if c.workpiece is None:
            c.workpiece = WORKPIECE_OF_TASK.get(c.task) or (WORKPIECE_OF[c.task[0]] if c.task in INSTRUCTIONS else "trout")
        self.config = c
        self.lock = threading.RLock()
        self.output = output_dir
        self.world = World(c)
        self.generation = 0
        self.trace_callback = None
        self._settle()
        self.initial = self.checkpoint()
        self.evaluator = Evaluator(self)
        if c.task in PROGRESS_SCORED:
            self.evaluator.prime()                 # the progress measure's starting connection counts, from this state

    # ------------------------------------------------------------------ histories
    def _reset_histories(self):
        ts = self.world.tissue
        self.steps = 0
        self.motion_history = deque(maxlen=20)
        self.tet_jmin_history = np.ones(ts.Tc)
        self.blade_left_hand_contacts = 0
        self.blade_robot_contacts = 0
        self.grip_history = deque(maxlen=64)       # (t, handle offset in the right grasp (3), axis angle deg) while held
        self.scale_log = deque(maxlen=4000)        # (t, bench scale reading g) every 50 ms (presets B, C)
        self.knife_held_since = None               # sim time since which the right hand has held the knife
        self.tissue_speed_max = (0., None, None)   # fastest tissue or bone-rod node over the episode (m/s, sim time, kind)
        # fastest piece over the episode (m/s of a fragment's centroid, sim time, its node count): what the thrown-tissue
        # check judges (a node of an attached sliver ringing on a cut face is not a thrown piece)
        self.piece_speed_max = (0., None, None)
        self._piece_lab = None                     # (node -> fragment labels, crack faces count, sim time) for that
        # the same, over pieces whose centroid is outside the task's product receptacles (task.RECEPTACLES_OF_TASK; plan
        # footprints): a product let go into its receptacle falls faster than the thrown-tissue limit from rim height,
        # and is not thrown. Recorded only for those tasks
        specs = [LY.task_spec(n, self.config.task) for n in RECEPTACLES_OF_TASK.get(self.config.task, ())]
        self._receptacles = [(*s["center"], s["size"][0]/2, s["size"][1]/2) for s in specs]
        self.piece_speed_max_outside = (0., None, None)
        # per-tet worst compression recorded only while the knife's EDGE is farther than EDGE_REACH_M from the tet: the
        # edge law crushes the tissue it is cutting (tissue.py), which is not handling damage (recorded for the
        # receptacle tasks only)
        self.tet_jmin_far = np.ones(ts.Tc) if self._receptacles else None

        self._left_boxes = [i for i, sh in enumerate(self.world.tissue_shapes) if sh.get("arm") == "left"]
        self._robot_boxes = [i for i, sh in enumerate(self.world.tissue_shapes) if sh.get("arm")]

    PIECE_RELABEL_S = .1          # s: fragment labels for the piece speeds are refreshed at most this often after a cut

    def _piece_speed(self, ts, v, kd):
        """Fastest fragment centroid (tissue nodes, equal weights) now; fragments by face connectivity
        (topology.CrackRecorder._fragments), relabelled after crack faces change, at most every PIECE_RELABEL_S."""
        cut = self.world.cut
        nf = len(getattr(cut, "cut_faces", ()))
        pl = self._piece_lab
        if pl is None or (pl[1] != nf and self.sim_time - pl[2] >= self.PIECE_RELABEL_S):
            nt = len(cut.orig)
            lab = cut._fragments()
            act = np.nonzero(ts.hactive[:nt])[0]
            node_lab = np.full(ts.n, -1, np.int64)
            node_lab[ts.htets[act].ravel()] = np.repeat(lab[act], 4)
            self._piece_lab = pl = (node_lab, nf, self.sim_time)
        node_lab = pl[0][:ts.n]
        if len(node_lab) < ts.n:                  # nodes split since the last labelling wait for the next one
            node_lab = np.concatenate([node_lab, np.full(ts.n - len(node_lab), -1, np.int64)])
        ok = (node_lab >= 0) & (kd[:len(node_lab)] == 0)
        if not ok.any():
            return
        lab = node_lab[ok]
        cnt = np.bincount(lab)
        vs = np.stack([np.bincount(lab, weights=v[:ts.n][ok, d], minlength=len(cnt)) for d in range(3)], 1)
        sp = np.linalg.norm(vs, axis=1)/np.maximum(cnt, 1)
        j = int(np.argmax(sp))
        if sp[j] > self.piece_speed_max[0]:
            self.piece_speed_max = (float(sp[j]), round(self.sim_time, 3), int(cnt[j]))
        if self._receptacles:
            x = ts.x.numpy()[:ts.n][ok]
            c = np.stack([np.bincount(lab, weights=x[:, d], minlength=len(cnt)) for d in (0, 1)], 1)
            c /= np.maximum(cnt, 1)[:, None]
            over = np.zeros(len(cnt), bool)
            for cx, cy, hx, hy in self._receptacles:
                over |= (np.abs(c[:, 0] - cx) <= hx) & (np.abs(c[:, 1] - cy) <= hy)
            # a piece a robot hand is touching is carried, not thrown
            held = near_boxes(self.world, self._robot_boxes, x)
            if held.any():
                over[np.unique(lab[held])] = True
            so = np.where(over, 0., sp)
            j = int(np.argmax(so))
            if so[j] > self.piece_speed_max_outside[0]:
                self.piece_speed_max_outside = (float(so[j]), round(self.sim_time, 3), int(cnt[j]))

    def _record(self):
        w = self.world
        ts = w.tissue
        k = self.steps
        if k % self.MOTION_EVERY == 0:
            self.motion_history.append((self.sim_time, ts.x.numpy()[:ts.n].copy()))
        if k % self.SPEED_EVERY == 0:
            kd = ts.kind.numpy()[:ts.n]
            sel = np.nonzero((kd == 0) | (kd == 2))[0]
            if len(sel):
                v_all = ts.v.numpy()[:ts.n].astype(float)
                sp2 = (v_all[sel]**2).sum(1)
                i = int(np.argmax(sp2))
                sp = float(np.sqrt(sp2[i]))
                if sp > self.tissue_speed_max[0]:
                    self.tissue_speed_max = (sp, round(self.sim_time, 3), "bone rod" if kd[sel[i]] == 2 else "flesh")
                self._piece_speed(ts, v_all, kd)
        if k % self.JMIN_EVERY == 0:
            nt = ts.nt
            x = ts.x.numpy()
            r = ts.htets[:nt]
            Ds = np.stack([x[r[:, i]] - x[r[:, 0]] for i in (1, 2, 3)], 2)
            J = np.linalg.det(Ds)*np.linalg.det(ts.hDminv[:nt])
            act = ts.hactive[:nt] > 0
            J[~act] = 1.
            self.tet_jmin_history[:nt] = np.minimum(self.tet_jmin_history[:nt], J)
            if self.tet_jmin_far is not None:
                far = edge_distance(w, x[r].mean(1)) >= EDGE_REACH_M
                self.tet_jmin_far[:nt] = np.minimum(self.tet_jmin_far[:nt], np.where(far, J, 1.))
        # blade proximity to the holding (left) hand: any blade edge/spine sample inside a left-hand finger box; and
        # to any robot box (task A0: the blade never touches a robot link)
        if self._left_boxes and k % 2 == 0:
            if self._blade_touches(self._left_boxes):
                self.blade_left_hand_contacts += 1
            if self._blade_touches(self._robot_boxes):
                self.blade_robot_contacts += 1
        # the knife in the right hand's grasp and the bench scale's reading (every 50 ms)
        if k % 50 == 0:
            g_ = w.scale_reading_g()
            if g_ is not None:
                self.scale_log.append((self.sim_time, g_))
            held = w.knife_mode == "held" and min(w.pinch_forces()["right"]) > .5
            if held:
                if self.knife_held_since is None:
                    self.knife_held_since = self.sim_time
                off, ang = w.knife_in_grip()
                self.grip_history.append((self.sim_time, np.asarray(off, float), float(ang)))
            else:
                self.knife_held_since = None
                self.grip_history.clear()

    def _blade_touches(self, box_ids):
        w = self.world
        ts = w.tissue
        sm = ts.smp
        body = sm["body"].numpy()
        sel = (body >= 1) & (body <= w.scene.blade.segments)
        if not sel.any():
            return False
        st = ts.body_states()
        from .workpiece import _R
        loc = sm["loc"].numpy()[sel]; b = body[sel]; rad = sm["r"].numpy()[sel]
        pts = np.array([st["p"][bb] + _R(st["q"][bb]) @ l for bb, l in zip(b, loc)])
        p0 = ts.s_p0.numpy(); q0 = ts.s_q0.numpy()
        for s in box_ids:
            R = _R(q0[s]); h = np.asarray(w.tissue_shapes[s]["half"])
            pl = (pts - p0[s]) @ R
            d = np.abs(pl) - h[None]
            out = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0)
            if (out < rad + 2e-4).any():
                return True
        return False

    @property
    def sim_time(self):
        return self.steps*self.config.rigid_dt

    def bond_cut_by_blade(self):
        ts = self.world.tissue
        if not getattr(ts, "n_bonds", 0):
            return np.zeros(0, bool)
        return ts.bonds["cut"].numpy() > 0

    # ------------------------------------------------------------------ lifecycle
    # anatomical workpieces: dynamic relaxation (velocities scaled by exp(-RELAX_RATE dt) after each tick) over the first
    # RELAX_S of the settle, then the world's own dynamics alone, so the episode starts from static equilibrium
    RELAX_S = 1.2
    RELAX_RATE = 80.

    ANATOMICAL_SETTLE_S = 1.5

    @property
    def settle_s(self):
        """Hands-off settle before the episode: 1.5 s for the anatomical workpieces, SETTLE_S for the coupon."""
        return self.ANATOMICAL_SETTLE_S if self.config.workpiece not in (None, "coupon") else self.SETTLE_S

    def _settle(self):
        # coupon: 0.5 s; anatomical workpieces settle onto the board in ~1 s (E0 rest probe), so 1.5 s hands-off. The
        # tissue's only damping is numerical (Kelvin-Voigt, tau 1e-4 s, for mesh-scale ringing): without relaxation the
        # trout's tail was still ringing at 258 um peak-to-peak when its episode began (tissue.relax)
        settle = self.settle_s
        anatomical = self.config.workpiece not in (None, "coupon")
        dt = self.config.rigid_dt
        n = int(round(settle/dt))
        n_relax = int(round(self.RELAX_S/dt)) if anatomical else 0
        factor = math.exp(-self.RELAX_RATE*dt)
        for k in range(n):
            self.world.step()
            if k < n_relax and self.world.tissue is not None:
                self.world.tissue.relax(factor)
        self.world.timing = {k: 0. for k in self.world.timing}
        self.world.events = []
        # protective stops during the hands-off settle belong to the authored initial condition, not the episode
        self.settle_reflexes = list(self.world.reflexes)
        self.world.reflexes = []
        self.world._reflex_until = {}
        self._reset_histories()

    def checkpoint(self):
        w = self.world
        return dict(
            rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
            targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), tick=w.tick,
            cmd_v=getattr(w, "_cmd_v", None).copy() if getattr(w, "_cmd_v", None) is not None else None,
            wrench=w.body_wrench.copy(), filtered=w.tissue_filtered.copy(),
            grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), knife_mode=w.knife_mode,
            no_pinch=w._no_pinch, pinch_ticks=w._pinch_ticks, reflex_until=dict(w._reflex_until), reflexes=copy.deepcopy(w.reflexes),
            refused_targets=w.refused_targets,
            tissue=w.tissue.snapshot(), cut=w.cut.snapshot(), steps=self.steps,
            hist=dict(motion=list(self.motion_history), jmin=self.tet_jmin_history.copy(),
                      blade_hand=self.blade_left_hand_contacts, blade_robot=self.blade_robot_contacts,
                      grip=list(self.grip_history), held_since=self.knife_held_since, scale=list(self.scale_log),
                      speed_max=self.tissue_speed_max, piece_speed_max=self.piece_speed_max,
                      piece_speed_max_outside=self.piece_speed_max_outside,
                      jmin_far=None if self.tet_jmin_far is None else self.tet_jmin_far.copy()),
            events=copy.deepcopy(w.events))

    def restore(self, snap):
        w = self.world
        for a, v in snap["rigid"].items():
            getattr(w.state, a).assign(v)
            getattr(w.next_state, a).assign(v)
        w.targets = snap["targets"].copy()
        w.control.joint_target_q.assign(snap["control"])
        w.tick = snap["tick"]
        w.body_wrench = snap["wrench"].copy()
        w.tissue_filtered = snap["filtered"].copy()
        w.grip_force = dict(snap["grip_force"])
        w.grasp_latched = dict(snap["grasp_latched"])
        w.knife_mode, w._no_pinch, w._pinch_ticks = snap["knife_mode"], snap["no_pinch"], snap["pinch_ticks"]
        if snap.get("cmd_v") is not None:
            w._cmd_v = snap["cmd_v"].copy()
        elif hasattr(w, "_cmd_v"):
            w._cmd_v[:] = 0.
        w._knife_mj_contacts(w.knife_mode == "held" and w.phantom_knife is None)
        w._reflex_until, w.reflexes = dict(snap["reflex_until"]), copy.deepcopy(snap["reflexes"])
        w.refused_targets = snap["refused_targets"]
        w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
        w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
        w.tissue.restore(snap["tissue"])
        w.cut.restore(snap["cut"])
        w.tissue.invalidate_graphs()
        w.fault = None
        w.events = copy.deepcopy(snap["events"])
        self.steps = snap["steps"]
        self.motion_history = deque(snap["hist"]["motion"], maxlen=20)
        self.tet_jmin_history = snap["hist"]["jmin"].copy()
        self.blade_left_hand_contacts = snap["hist"]["blade_hand"]
        self.blade_robot_contacts = snap["hist"]["blade_robot"]
        self.grip_history = deque(snap["hist"]["grip"], maxlen=64)
        self.tissue_speed_max = tuple(snap["hist"].get("speed_max", (0., None, None)))
        self.piece_speed_max = tuple(snap["hist"].get("piece_speed_max", (0., None, None)))
        self.piece_speed_max_outside = tuple(snap["hist"].get("piece_speed_max_outside", (0., None, None)))
        if snap["hist"].get("jmin_far") is not None:
            self.tet_jmin_far = snap["hist"]["jmin_far"].copy()
        self._piece_lab = None
        self.knife_held_since = snap["hist"]["held_since"]
        self.scale_log = deque(snap["hist"]["scale"], maxlen=4000)

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, guard=None, state=True):
        """Set joint targets and step `ticks`; returns the state with its evaluation, or None with state=False
        (scripted demonstrations streaming targets every 10 ms skip the evaluator between their steps)."""
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.world.fault:
                raise RuntimeError(self.world.fault)
            if self.config.task in STOP_AT_TIME_LIMIT:
                # the episode ends at its time limit: an action crossing it runs only to the limit, later ones run no
                # tick, so the evaluated state is the state within the budget (the runner never stops on time and a
                # policy is not shown `terminated`)
                left = int(round(TIME_LIMIT_S[self.config.task]/self.config.rigid_dt)) - self.steps
                if left <= 0:
                    return self.get_state() if state else None
                ticks = min(ticks, left)
            if targets:
                self.world.set_targets(targets)
            for _ in range(ticks):
                self.world.step()
                self.steps += 1
                self._record()
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state() if state else None

    def step_once(self):
        """One tick without new targets (scripted operators call world-level helpers, then this)."""
        self.world.step()
        self.steps += 1
        self._record()

    # ------------------------------------------------------------------ state
    def get_state(self, evaluate=True):
        with self.lock:
            w = self.world
            jq, jv = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
            ev = self.evaluator.evaluate() if (evaluate and not w.fault) else dict(success=False, score=0., checks={})
            off, ang = w.knife_in_grip()
            return {
                "tick": self.steps, "generation": self.generation, "time_s": self.sim_time,
                "ready": w.fault is None, "fault": w.fault,
                "config": {k: v for k, v in asdict(self.config).items()},
                "instruction": INSTRUCTIONS.get(self.config.task),
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                               "target": float(w.targets[s["q"]])} for n, s in w.joints.items()},
                "knife": {"grasp_offset_m": off.tolist(), "grasp_axis_deg": float(ang), "mode": w.knife_mode,
                          "integrated_by": {"held": "mujoco", "free": "tissue", "phantom": "phantom_rig"}.get(w.knife_mode),
                          "gripped": bool(getattr(w, "knife_gripped", False))},
                "robot": {"reflexes": list(w.reflexes[-10:]), "refused_targets": int(w.refused_targets),
                          "stopped_arms": sorted(a for a, u in w._reflex_until.items() if w.time < u)},
                "tissue": {"nodes": int(w.tissue.n), "tets": int(w.tissue.nt), "cut_faces": len(w.cut.cut_faces),
                           "substeps": int(w.tissue.last_substeps)},
                "events": list(w.events[-20:]),
                "terminated": self.sim_time >= TIME_LIMIT_S.get(self.config.task, 1e9),
                "evaluation": ev,
            }

    def close(self):
        pass

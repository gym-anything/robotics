"""Coupled world: robots (rig.py, MuJoCo 1 kHz) + packaging (ipcworld.py, implicit contact, 4 ms: config.py).

Per environment tick (one contact-world step = `substeps` robot ticks):
  1. robot ticks with the joint targets and the contact reactions from the previous contact step
     (lagged one step, first-order filtered, capped);
  2. every robot link's pose becomes the target of its driven obstacle in the contact world; each finger's
     commanded closing position and force cap come from the hand state (rig.py);
  3. one implicit contact step for paper, tape, rope, items, furniture and robot obstacles together;
  4. contact wrenches on robot obstacles -> robot bodies (finger off-axis loads to the hand); the MuJoCo
     finger joints follow the finger position the contact world resolved.

Unitree G1 (g1.py, scene.G1_LAYOUT): the floating humanoid stands where the scene's G1 layout puts it. Every G1 link,
the Dex3 finger links included, is an ordinary driven obstacle (link compliance, reaction cap) whose contact wrench goes
back to that link's own MuJoCo body; the Dex3 fingers are revolute position servos, so a pinch force is the joint servo's
torque against that reaction. Steps 2 and 4's closing-axis finger coupling is for parallel grippers only and is skipped
(the G1 has no finger_ids). Its finger links carry the friction class "dex3_finger" (params.FRICTION_DEX3).
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

from . import ipcworld as W
from . import params as Pm
from .rig import RobotRig

REACTION_TAU = 5e-3       # s, first-order filter on contact-world -> robot reactions (pad/link compliance scale)
REACTION_CAP = 250.       # N per link: no packaging interaction here carries more (a 2.5 kg arm-scale bound)
PREDICT_MIN_N = .5        # N, robot bodies loaded above this get the in-substep force prediction


class World:
    def __init__(self, scene, config):
        self.c = config
        self.scene = scene
        from .robot_setup import resolve_setup
        setup = resolve_setup(config.robot)
        g1 = getattr(scene, "g1", None)
        mounts, mount_z, ready = scene.mounts, scene.mount_z, scene.ready_pose
        if g1 is not None:
            from . import g1 as G1
            if config.rigid_dt > G1.RIGID_DT_MAX + 1e-12:
                raise ValueError(f"the floating G1 is qualified at a MuJoCo step of at most {G1.RIGID_DT_MAX*1e3:g} ms "
                                 f"(g1.RIGID_DT_MAX: its feet creep at 1 ms); use Config(robot='g1', "
                                 f"rigid_dt={G1.RIGID_DT_MAX}, substeps={G1.SUBSTEPS})")
            mounts, mount_z, ready = {G1.MOUNT: (tuple(g1["pelvis"]), g1["yaw"])}, G1.PELVIS_Z, g1.get("arms")
        self.rig = RobotRig(setup=setup, mounts=mounts, mount_z=mount_z, furniture=scene.furniture,
                            device=config.device, rigid_dt=config.rigid_dt, ready_pose=ready,
                            contact_timeconst=config.contact_timeconst)
        if bool(self.rig.humanoid) != (g1 is not None):
            raise ValueError("robot selection and scene layout disagree: build the scene with the same Config.robot")
        bodies = list(scene.bodies)
        self.item_index = {b.name: i for i, b in enumerate(bodies)}
        # robot links as driven obstacles
        self.link_body = {}
        shapes = self.rig.link_collision_shapes()
        self.finger_ids = {}
        for arm in ([] if self.rig.humanoid else self.rig.rig.instances):
            fb = self.rig.finger_bodies(arm)
            self.finger_ids[arm] = fb
        finger_set = {f for fs in self.finger_ids.values() for f in fs}
        dex3 = set()
        if self.rig.humanoid:
            from . import g1 as G1
            dex3 = {i for i in self.rig.robot_bodies if self.rig.model.body_label[i].split("/")[-1] in G1.FINGER_LINKS}
        for body, (V, F) in shapes.items():
            p, R = self.rig.body_pose(body)
            label = self.rig.model.body_label[body]
            k_lin = config.finger_k if body in finger_set else config.link_k
            bodies.append(W.Body(f"robot:{label}", V, F, kind="driven", p=p, R=R, mu=config.finger_mu if body in finger_set else .3,
                                 fmat="finger" if body in finger_set else "dex3_finger" if body in dex3 else "link",
                                 k_lin=k_lin, k_rot=k_lin*.05**2, f_cap=REACTION_CAP))
            self.link_body[body] = len(bodies) - 1
        self.ipc = W.IPCWorld(scene.sheets, scene.rods, bodies, dhat=config.dhat, mu_default=.4, contact_stock=scene.contact_stock,
                              newton_max=config.newton_max, tol=config.newton_tol, pair_mu=scene.pair_mu,
                              mu_pairs={**Pm.FRICTION, **Pm.FRICTION_DEX3} if self.rig.humanoid else Pm.FRICTION,
                              friction_strict=True, kappa=config.kappa)
        for setup_fn in scene.ipc_setup:
            setup_fn(self.ipc, self)
        self.steps = 0
        self.timing = dict(robot_s=0., contact_s=0., couple_s=0.)
        self.fault = None
        self.filtered = {}

    # ------------------------------------------------------------------ stepping
    def step(self, fixture=None):
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.c
        try:
            t0 = time.perf_counter()
            base = self.rig.body_wrench.copy()
            for _ in range(c.substeps):
                self._predict_contact_wrench(base)
                self.rig.rigid_step()
            self.rig.body_wrench = base
            for attr in ("body_q", "body_qd", "joint_q"):
                if not np.isfinite(getattr(self.rig.state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite robot {attr} at tick {self.rig.tick}")
            t1 = time.perf_counter()
            self._drive_obstacles()
            if fixture is not None:
                fixture(self)
            t2 = time.perf_counter()
            self.ipc.step(c.dt)
            t3 = time.perf_counter()
            self._apply_reactions()
            self._fingers_follow()
            self._record_reaction_peak()
            self.steps += 1
            self.timing["robot_s"] += t1 - t0
            self.timing["contact_s"] += t3 - t2
            self.timing["couple_s"] += t2 - t1 + time.perf_counter() - t3
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _drive_obstacles(self):
        rig = self.rig
        for body, b_i in self.link_body.items():
            p, R = rig.body_pose(body)
            self.ipc.set_driven(b_i, p, R)
        self.ipc.finger_cmd = {}
        for arm, fingers in self.finger_ids.items():
            h = rig.hand[arm]
            P = [rig.body_pose(f)[0] for f in fingers]
            q = rig.state.joint_q.numpy()
            fj = rig.gripper_joints(arm)
            # closing direction of the first finger = minus its link frame's y axis (checked against the finger
            # separation with the hand open: 80 mm apart along exactly that direction). The finger separation itself
            # is not used: it vanishes when the fingers are fully closed.
            a0 = -rig.body_pose(fingers[0])[1][:, 1]
            for k, f in enumerate(fingers):
                a = a0 if k == 0 else -a0                 # closing direction of this finger
                qk = float(q[rig.joints[fj[k]]["q"]])
                p_cmd = P[k] + (qk - h["cmd"][k])*a
                cap = h["force"] if h["mode"] == "force" else h["effort"]
                self.ipc.finger_cmd[self.link_body[f]] = (a, float(a @ p_cmd), max(cap, 1e-3), self.c.finger_k)

    def _apply_reactions(self):
        rig = self.rig
        w = np.zeros_like(rig.body_wrench)
        com_local = rig.model.body_com.numpy()
        q = rig.state.body_q.numpy()
        hands = {arm: rig.hand_body(arm) for arm in self.finger_ids}
        finger_of = {f: arm for arm, fs in self.finger_ids.items() for f in fs}
        # along the closing axis, the equal-and-opposite part of the two fingers' contact forces is the grasp
        # squeeze (internal to the hand); their net axial force is external (a closed hand pushing an object
        # with its outer finger face) and goes to the hand with the off-axis parts
        net_axial, axial_at = {}, {}
        for arm, fs in self.finger_ids.items():
            tot, mom, wsum = np.zeros(3), np.zeros(3), 0.
            for f in fs:
                b_i = self.link_body[f]
                if b_i in self.ipc.reactions:
                    a, *_ = self.ipc.finger_cmd[b_i]
                    F, _ = self.ipc.reactions[b_i]
                    fa = float(F @ a)
                    tot += fa*a
                    mom += abs(fa)*rig.body_pose(f)[0]; wsum += abs(fa)
            net_axial[arm] = tot
            axial_at[arm] = mom/wsum if wsum > 0 else None
        for body, b_i in self.link_body.items():
            if b_i not in self.ipc.reactions:
                continue
            F, tau = self.ipc.reactions[b_i]                    # about the obstacle frame origin (link origin)
            p, R = rig.body_pose(body)
            target = body
            if body in finger_of:
                a, *_ = self.ipc.finger_cmd[b_i]
                arm = finger_of[body]
                F = F - (F @ a)*a                                # off-axis part; the net axial push is added once below
                tau = tau.copy()
                target = hands[arm] if hands[arm] is not None else body
            qt = q[target]
            com = qt[:3] + Rotation.from_quat(qt[3:]).apply(com_local[target])
            w[target, :3] += F
            w[target, 3:] += tau + np.cross(p - com, F)
        # the hand's net push along its closing axis, once per hand, at the pushing fingers (split as half
        # per finger with a reaction, a single finger pushing with its outer face would pass only half of the force the
        # item received to the arm)
        for arm, Fa in net_axial.items():
            target = hands[arm]
            if target is None or axial_at[arm] is None or not np.any(Fa):
                continue
            qt = q[target]
            com = qt[:3] + Rotation.from_quat(qt[3:]).apply(com_local[target])
            w[target, :3] += Fa
            w[target, 3:] += np.cross(axial_at[arm] - com, Fa)
        mag = np.linalg.norm(w[:, :3], axis=1)
        scale = np.minimum(1., REACTION_CAP/np.maximum(mag, 1e-12))
        w *= scale[:, None]
        a = self.c.dt/(REACTION_TAU + self.c.dt)
        rig.body_wrench = ((1 - a)*rig.body_wrench + a*w).astype(np.float32)
        # first-order prediction of the contact force inside the next robot substeps (_predict_contact_wrench):
        # for each loaded robot body, its push direction, the coupling stiffness behind it and its pose now
        self._pred = {}
        bq = rig.state.body_q.numpy()
        kmap = {hands[a_]: self.c.finger_k for a_ in hands if hands[a_] is not None}
        for t_ in np.where(np.linalg.norm(rig.body_wrench[:, :3], axis=1) > PREDICT_MIN_N)[0]:
            F0 = rig.body_wrench[t_, :3].astype(float)
            n = F0/np.linalg.norm(F0)
            self._pred[int(t_)] = (n, float(np.linalg.norm(F0)), kmap.get(int(t_), self.c.link_k), bq[t_, :3].astype(float).copy())

    def extremes(self):
        """Run-record summary of the recorded extremes."""
        pend = [r["pending_update_m"] for r in getattr(self, "unconverged_pending", []) if r["pending_update_m"] == r["pending_update_m"]]
        out = dict(reaction_peaks=getattr(self, "reaction_peaks", [])[:5], barrier_side_peaks=getattr(self, "barrier_peaks", [])[:5],
                   item_impulse_peaks=getattr(self, "item_impulse_peaks", [])[:5],
                   unconverged_steps=len(getattr(self, "unconverged_pending", [])),
                   unconverged_pending_update_mm=dict(max=1e3*max(pend), median=1e3*float(np.median(pend))) if pend else None)
        ev = getattr(self.ipc, "tear_events", None)
        if ev is not None:
            # where and when the paper tore (vertex splits: time, sheet, rest position of the split vertex)
            # (and why: the rule that split it, its in-plane and out-of-plane indices, and how far its cohesive bond has
            # opened since: 0 = it never opened and nothing is torn there)
            tb = self.ipc.tear_bonds

            def opened(e):
                b = e.get("bond")
                return None if b is None else round(min(1., float(tb["diss"][b])/max(float(tb["gl"][b]), 1e-30)), 4)
            out["tear"] = dict(summary=self.ipc.tear_summary(),
                               events=[dict(t=round(e["t"], 4), step=int(e.get("step", -1)), sheet=self.ipc.sheets[e["sheet"]].name,
                                            rest_mm=[round(1e3*c, 1) for c in e["rest"]], kind=e.get("kind"),
                                            in_plane_index=e.get("in_plane_index"), out_of_plane_index=e.get("out_of_plane_index"),
                                            ratio=e.get("ratio"), bond_opened_fraction=opened(e),
                                            step_converged=e.get("converged")) for e in ev[:200]])
            # was a split decided on a state the solver had not converged? (splits are tested after each step)
            out["tear"]["splits_decided_after_an_unconverged_step"] = int(sum(not e.get("converged", True) for e in ev))
        return out

    def _record_extremes(self, unconv):
        """Keep (a) the 10 largest BARRIER-side readings on robot links (the readout the coupling
        side replaced; reported so the difference stays visible), (b) the 10 largest per-step impulses on dynamic items
        (|m dv|, a hit or a snatch shows here), (c) for every unconverged step the size of the update Newton left
        pending (how far from equilibrium an accepted state was)."""
        ipc = self.ipc
        rb = getattr(ipc, "reactions_barrier", {})
        if rb:
            b_i, (Fb, _) = max(rb.items(), key=lambda kv: float(np.linalg.norm(kv[1][0])))
            fb = float(np.linalg.norm(Fb))
            bp = getattr(self, "barrier_peaks", [])
            if len(bp) < 10 or fb > bp[-1]["barrier_side_n"]:
                Fc = ipc.reactions.get(b_i)
                bp.append(dict(step=self.steps, t=round(self.sim_time, 4), body=ipc.bodies[b_i].name, barrier_side_n=fb,
                               coupling_side_n=float(np.linalg.norm(Fc[0])) if Fc is not None else None,
                               unconverged=bool(unconv)))
                bp.sort(key=lambda r: -r["barrier_side_n"])
                self.barrier_peaks = bp[:10]
        prev = getattr(self, "_item_v", None)
        cur = {}
        for i, b in enumerate(ipc.bodies):
            if b.kind == "dynamic":
                o = ipc.body_dof[i]
                cur[i] = ipc.v[o:o + 3].copy()
        if prev:
            ip = getattr(self, "item_impulse_peaks", [])
            for i, v in cur.items():
                if i in prev:
                    J = float(ipc.bodies[i].mass*np.linalg.norm(v - prev[i]))
                    if len(ip) < 10 or J > ip[-1]["impulse_n_s"]:
                        ip.append(dict(step=self.steps, t=round(self.sim_time, 4), body=ipc.bodies[i].name, impulse_n_s=J,
                                       mean_force_n=J/max(self.c.dt, 1e-12)))
                        ip.sort(key=lambda r: -r["impulse_n_s"])
                        ip = ip[:10]
            self.item_impulse_peaks = ip
        self._item_v = cur
        if unconv:
            pend = getattr(self, "unconverged_pending", [])
            pend.append(dict(step=self.steps, t=round(self.sim_time, 4),
                             pending_update_m=float(ipc.stats.get("last_unconverged_update_m", float("nan")))))
            self.unconverged_pending = pend[-200:]

    def _record_reaction_peak(self):
        """Keep the 20 largest contact reactions on any robot link with the step's Newton count and whether the step
        converged (diagnostic: a reaction read from an unconverged state is not an equilibrium force)."""
        st = self.ipc.stats
        unconv = st.get("unconverged", 0) != getattr(self, "_unconv_seen", 0)
        self._unconv_seen = st.get("unconverged", 0)
        self._record_extremes(unconv)
        if not self.ipc.reactions:
            return
        b_i, (F, _) = max(self.ipc.reactions.items(), key=lambda kv: float(np.linalg.norm(kv[1][0])))
        f = float(np.linalg.norm(F))
        peaks = getattr(self, "reaction_peaks", [])
        if len(peaks) < 20 or f > peaks[-1]["force_n"]:
            fb = getattr(self.ipc, "reactions_barrier", {}).get(b_i)
            peaks.append(dict(step=self.steps, t=round(self.sim_time, 4), body=self.ipc.bodies[b_i].name, force_n=f,
                              barrier_side_n=float(np.linalg.norm(fb[0])) if fb is not None else None,
                              newton_iters=st.get("last_iters"), unconverged=bool(unconv)))
            peaks.sort(key=lambda r: -r["force_n"])
            self.reaction_peaks = peaks[:20]

    def _predict_contact_wrench(self, base):
        """Within the 4 robot substeps of one contact step the contact world is not re-solved, so a robot body
        pressing on paper or an item would feel a constant (lagged) force however far it moves into the contact.
        That one-step lag made position-servoed arms bounce on rigid items (lid squeeze: 0-116 N chatter).
        Predict the force to first order instead: along the body's push direction n the force grows
        with the coupling stiffness k as the body moves deeper (F = F0 + k (p0 - p) . n), never pulls (F >= 0)
        and stays under the reaction cap; torque scales with it."""
        rig = self.rig
        w = base.copy()
        pred = getattr(self, "_pred", {})
        if pred:
            bq = rig.state.body_q.numpy()
            for t_, (n, F0, k, p0) in pred.items():
                F = min(max(F0 + k*float((p0 - bq[t_, :3]) @ n), 0.), REACTION_CAP)
                w[t_] *= F/F0
        rig.body_wrench = w.astype(np.float32)

    def _fingers_follow(self):
        rig = self.rig
        q = rig.state.joint_q.numpy()
        for arm, fingers in self.finger_ids.items():
            fj = rig.gripper_joints(arm)
            h = rig.hand[arm]
            for k, f in enumerate(fingers):
                b_i = self.link_body[f]
                a, *_ = self.ipc.finger_cmd[b_i]
                p_ipc, _ = self.ipc.body_pose(b_i)
                p_mj, _ = rig.body_pose(f)
                qk = float(q[rig.joints[fj[k]]["q"]])
                opening = qk - float(a @ (p_ipc - p_mj))
                lo, hi = rig.joints[fj[k]]["lower"], rig.joints[fj[k]]["upper"]
                h["follow"][k] = float(np.clip(opening, lo, hi))

    # ------------------------------------------------------------------ queries
    @property
    def sim_time(self):
        return self.steps*self.c.dt

    def item_pose(self, name):
        return self.ipc.body_pose(self.item_index[name])

    def tcp(self, arm):
        return self.rig.rig.frame_pose(f"{arm}/tcp", self.rig.state)

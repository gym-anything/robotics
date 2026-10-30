"""Rigid world + soil: classroom_arrangement's qualified Mobile FR3 Duo world (Newton 1.5 + MuJoCo CPU, 0.5 ms ticks,
servo / reflex / base stack unchanged) with the tool-holding additions first qualified in painting (finger spindle
coupling, pad torsion, 1 ms tool/pad contacts, converged no-slip, force-mode grip; frozen in holding.py) and the soil
system's coupling:

  every SOIL_TICKS ticks (2 ms) the soil steps on the GPU with each coupled body's pose and its velocity rewound by
  the previous soil wrench (soil.py); the returned wrench is applied to the body (at its COM, world frame) over the
  next SOIL_TICKS ticks. Coupled bodies: tools (and the auger, the digger halves), stones, the root ball, the post,
  and the robot's distal arm links, fingers, base hull, drive units and casters (a hand or a wheel in the soil feels
  it). Soil wrenches on robot links enter the FR3 collision-reflex estimate (J^T f) like contact forces do.

Mechanisms run here per tick: the drill motor (DC-motor torque from the trigger travel and spindle speed, clutch
torque cap, reaction on the drill body), the post's braces (two spring-damper diagonals until released) and the
strike log of tool contacts with the irrigation lateral.
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement import params as CP
from .holding import HoldingWorld
from . import params as P
from . import layout as L
from .scene import SoilScene
from .soil import SoilSystem, SoilFault, block_particles, sdf, ORIGIN_BLOCK, ORIGIN_PILE
from . import materials as M



class SoilWorld(HoldingWorld):
    scene_cls = SoilScene

    def __init__(self, config, spec, device=None, particles=None):
        super().__init__(config, spec)
        self.strikes, self.strike_contacts = [], 0
        self.braces = None
        self.drill_state = None
        self._soil_device = device or getattr(config, "device", None) or "cuda:0"
        t0 = time.perf_counter()
        if particles == "none":          # render-only rebuild of a recorded episode's rigid world (video.py)
            self.soil = None
            self.particle_spec = None
            self.coupled_bodies = np.zeros(0, int)
        else:
            self._build_soil(particles)
        self.soil_build_s = time.perf_counter() - t0
        if "drill" in self.scene.bodies:
            self._init_drill()

    # ------------------------------------------------------------------ soil construction
    def _robot_shapes_for_soil(self, body):
        """Collision shapes of one robot body as soil collider shapes (body frame)."""
        import newton
        b = self.scene.b
        out = []
        C = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for i in self.scene.robot_shapes:
            if b.shape_body[i] != body or not int(b.shape_flags[i]) & C:
                continue
            tf = np.asarray(b.shape_transform[i], float)
            pos, quat = tuple(tf[:3]), tuple(tf[3:])
            t = b.shape_type[i]
            sc = np.asarray(b.shape_scale[i], float)
            G = newton.GeoType
            if t == G.BOX:
                out.append(dict(kind="box", size=tuple(sc), pos=pos, quat=quat))
            elif t == G.CAPSULE:
                out.append(dict(kind="capsule", size=(sc[0], sc[1]), pos=pos, quat=quat))
            elif t == G.CYLINDER:
                out.append(dict(kind="cylinder", size=(sc[0], sc[1]), pos=pos, quat=quat))
            elif t == G.SPHERE:
                out.append(dict(kind="sphere", size=(sc[0],), pos=pos, quat=quat))
            elif t in (G.MESH, G.CONVEX_MESH):
                src = b.shape_source[i]
                v = np.asarray(src.vertices, np.float32)*sc
                f = np.asarray(src.indices, np.int32).reshape(-1, 3)
                out.append(dict(kind="mesh", mesh=(v, f), size=(1, 1, 1), pos=pos, quat=quat))
        return out

    def _robot_soil_bodies(self):
        """The robot bodies the soil sees, read from the robot template (robots/models/<id>/robot.json), not from link
        names: every body's load-bearing wheel shape, the body of the template's `base` frame, each arm's
        `<arm>_wrist` frame body (it carries the hand) and the bodies the `gripper` groups' joints move. The chassis
        remains a collider for real bottoming contact, but is no longer a substitute for the missing wheel contact."""
        from robots.catalog import model
        sc = self.scene
        spec = model(sc.robot_setup.to_dict()["instances"][sc.robot_instance]["model"]).to_dict()
        frames = spec["frames"]
        base_row = sc.robot_body[frames["base"]["parent"]]
        names = {frames["base"]["parent"]} | {f["parent"] for k, f in frames.items() if k.endswith("_wrist")}
        out = {sc.robot_body[n] for n in names if n in sc.robot_body}
        self._robot_wheel_rows = {int(sc.b.shape_body[i]) for i in sc.wheel_shapes}
        self._robot_base_rows = self._robot_wheel_rows | {base_row}
        out.update(self._robot_wheel_rows)
        labels = list(self.model.joint_label)
        child = self.model.joint_child.numpy()
        for g in spec["groups"].values():
            if g.get("role") != "gripper":
                continue
            for j in g["joints"]:
                k = next((i for i, lab in enumerate(labels) if lab.split("/")[-1] == j), None)
                if k is not None:
                    out.add(int(child[k]))
        return sorted(out)

    def _couplers(self):
        sc = self.scene
        mass = self.model.body_mass.numpy()
        com = self.model.body_com.numpy()
        inertia = self.model.body_inertia.numpy()
        q = self.state.body_q.numpy()
        robot_soil_bodies = self._robot_soil_bodies()
        robot_rows = np.asarray(sc.robot_bodies, int)
        robot_mass = float(mass[robot_rows].sum())
        Rr = Rotation.from_quat(q[robot_rows, 3:]).as_matrix()
        cr = q[robot_rows, :3] + np.einsum("nij,nj->ni", Rr, com[robot_rows])
        base_row = next(iter(self._robot_base_rows - self._robot_wheel_rows))
        Rb = Rotation.from_quat(q[base_row, 3:]).as_matrix()
        cb = q[base_row, :3] + Rb @ com[base_row]
        d = cr - cb
        Iw = (np.einsum("nij,njk,nlk->nil", Rr, inertia[robot_rows], Rr)
              + mass[robot_rows, None, None]
              * (np.einsum("ni,nj->nij", d, d)*-1.0
                 + np.einsum("ni,ni->n", d, d)[:, None, None]*np.eye(3))).sum(0)
        robot_inertia_at_base = Rb.T @ Iw @ Rb
        out, idx = [], []
        for name, c in sc.coupled.items():
            if not c["shapes"]:
                continue
            bi = c["body"]
            out.append(dict(name=name, shapes=c["shapes"], mass=float(mass[bi]), com=com[bi], inertia=inertia[bi],
                            iface=c["iface"], pose=(q[bi, :3], q[bi, 3:]), cuts=c.get("cuts", [])))
            idx.append(bi)
        for bi in robot_soil_bodies:
            label = self.model.body_label[bi].split("/")[-1]
            shapes = self._robot_shapes_for_soil(bi)
            if not shapes:
                continue
            # The freely moving chassis enters the MPM solve with the complete robot's mass and inertia; otherwise
            # its lagged kinematic reaction can carry the robot in a non-decaying contact cycle.
            soil_mass = robot_mass if bi == base_row else 0.0
            true_mass = robot_mass if bi == base_row else float(mass[bi])
            true_inertia = robot_inertia_at_base if bi == base_row else inertia[bi]
            iface = P.INTERFACE["caster"] if label.startswith("caster_") else \
                    P.INTERFACE["wheel"] if bi in self._robot_wheel_rows else P.INTERFACE["robot"]
            out.append(dict(name=f"robot/{label}", shapes=shapes, mass=soil_mass, true_mass=true_mass, com=com[bi],
                            kinematic=bi != base_row,
                            inertia=true_inertia, iface=iface, pose=(q[bi, :3], q[bi, 3:])))
            idx.append(bi)
        return out, np.asarray(idx, int)

    def _build_soil(self, particles):
        sp = self.spec
        if particles is None:
            particles = build_particles(sp, self)
        self.particle_spec = particles
        b = sp["block"]
        lo = np.asarray(b["lo"]) - 0.1
        hi = np.asarray(b["hi"]) + np.array([0.1, 0.1, 0.6])
        (tx0, ty0), (tx1, ty1) = sp["tarp"]
        lo = np.minimum(lo, (tx0, ty0, lo[2])); hi = np.maximum(hi, (tx1, ty1, hi[2]))
        couplers, idx = self._couplers()
        self.coupled_bodies = idx
        self.coupled_names = [c["name"] for c in couplers]
        self.coupled_mass = np.array([c.get("true_mass", c["mass"]) for c in couplers], float)
        self.coupled_kinematic = np.array([c.get("kinematic", c["name"].startswith("robot/")) for c in couplers], bool)
        mjb = np.array([self._mj_body[int(b)] for b in idx], int)
        self._coupled_mj = mjb
        # articulated tools: a held drill also holds its spindle (the auger) and trigger
        self._held_alias = {}
        for k, c in enumerate(couplers):
            if c["name"] == "auger" and "drill" in self.scene.bodies:
                self._held_alias[k] = self._mj_body[self.scene.bodies["drill"]]
        self.coupled_com = np.array([c["com"] for c in couplers], float).reshape(-1, 3)
        self.coupled_invI = np.array([np.linalg.inv(np.asarray(c["inertia"], float).reshape(3, 3)) for c in couplers]).reshape(-1, 3, 3)
        spec = dict(voxel=sp["voxel"], moisture=sp["moisture"], particles=particles, statics=self.scene.statics_soil,
                    bounds=(lo, hi))
        self.soil = SoilSystem(spec, couplers, device=self._soil_device)
        self.soil_wrench = np.zeros((len(idx), 6))
        self.soil_ticks = P.SOIL_TICKS
        self._arm_link_rows = {}
        for k, n in enumerate(self.coupled_names):
            for arm in self.arms:
                if n.startswith(f"robot/{arm}_"):                # the arm's own links (template groups are prefixed by arm)
                    self._arm_link_rows.setdefault(arm, []).append(k)
        self.timing.update(soil_s=0.0)

    def settle_soil(self, seconds=None):
        """Soil-only gravity settle with every coupled body held where it is (before the coupled settle)."""
        q = self.state.body_q.numpy()[self.coupled_bodies]
        return self.soil.settle(seconds, q=q)

    # ------------------------------------------------------------------ coupling
    def held_bodies(self):
        """MuJoCo body ids a fingertip pad touches (a held or pushed tool is driven by the robot's servos)."""
        d = self.mjd
        if not d.ncon:
            return set()
        g = np.asarray(d.contact.geom[:d.ncon])
        pads = np.isin(g, self.pad_geoms)
        held = set()
        for i in np.flatnonzero(pads.any(1)):
            other = g[i, 1] if pads[i, 0] else g[i, 0]
            held.add(int(self._geom_body[other]))
        return held

    def held_bodies_by_arm(self, arm):
        """MuJoCo bodies touched by one arm's finger pads (any contact)."""
        d = self.mjd
        g = d.contact.geom[:d.ncon]
        if not hasattr(self, "pad_geoms_by_arm"):
            self.pad_geoms_by_arm = {}
            for (shape, label), geom in zip(self.scene.pad_shapes.items(), self.pad_geoms):
                a = "right" if "right_" in label else "left"
                self.pad_geoms_by_arm.setdefault(a, set()).add(int(geom))
        pads = self.pad_geoms_by_arm.get(arm, set())
        out = set()
        for g0, g1 in g:
            if int(g0) in pads:
                out.add(int(self._geom_body[g1]))
            elif int(g1) in pads:
                out.add(int(self._geom_body[g0]))
        return out

    def _soil_step(self):
        """Two-way coupling: a body the robot drives -- a robot link, or a tool a
        hand holds or an assisted fixture drives -- is kinematic in the soil (its MPM mass 0): the soil returns the force
        needed to move it along its MuJoCo motion, applied over the next 4 ticks. A free body (a dropped tool, a stone,
        the root ball, the post) is a dynamic collider with its own mass and the previous soil wrench rewound from its
        velocity. C1 measures why: at 50 rheology iterations the dynamic body-soil solve under-resolves a pushed blade's
        resistance (26-46 N against the 94-114 N of the same blade moved kinematically, the bench value); the
        kinematic-lagged scheme gave 125/149 N at 10/18 cm against the edge-bearing estimate 93/146 N."""
        q = self.state.body_q.numpy()[self.coupled_bodies].astype(float)
        qd = self.state.body_qd.numpy()[self.coupled_bodies].astype(float)
        dt = P.SOIL_DT
        held = self.held_bodies() | getattr(self, "driven_bodies", set())   # + bodies an assisted fixture drives
        kin = self.coupled_kinematic | np.array([int(b) in held or self._held_alias.get(k, -1) in held
                                                 for k, b in enumerate(self._coupled_mj)], bool)
        # every other coupled body (stones, the root ball, the post, a dropped tool) is a dynamic collider: its mass
        # is in the MPM's implicit contact solve, which is stable at any soil stiffness. Making bodies of 1.5 kg and
        # more kinematic does not stiffen a pushed post (the post test still leans 0.13 m, B10), and a resting kinematic
        # body chatters on the explicit lag: in a soil bin a buried stone so coupled bounces at +-13 mm/s, loosens
        # 21 % of the hardpan's top 4 cm 15 cm and more away from it, and sinks
        self.soil_kinematic = kin
        self.soil.set_masses(np.where(kin, 0.0, self.coupled_mass))
        F = self.soil_wrench
        R = Rotation.from_quat(q[:, 3:]).as_matrix()
        Iw_inv = np.einsum("nij,njk,nlk->nil", R, self.coupled_invI, R)
        v = qd[:, :3] - np.where(kin[:, None], 0.0, dt*F[:, :3]/np.maximum(self.coupled_mass, 1e-9)[:, None])
        w = qd[:, 3:] - np.where(kin[:, None], 0.0, dt*np.einsum("nij,nj->ni", Iw_inv, F[:, 3:]))
        t0 = time.perf_counter()
        F_new = self.soil.step(q, np.hstack([v, w]))
        # under-relaxed explicit coupling for kinematic bodies: the applied soil wrench is a running average (relaxation
        # alpha per soil step, time constant ~2 ms / alpha). Unrelaxed, the reaction on a driven spade alternated every
        # soil step (21-66 N) with 0.06 m/s / 0.4 rad/s velocity noise -- the soil entrained per step outweighs the
        # tool's rotational inertia at the blade -- and the vibrating blade remolded the soil it was to lift (D2-P v5)
        a = getattr(self, "coupling_relaxation", P.COUPLING_RELAXATION)      # probes may override
        if getattr(self, "_F_relax", None) is None or self._F_relax.shape != F_new.shape:
            self._F_relax = F_new.copy()
        self._F_relax = np.where(kin[:, None], a*F_new + (1.0 - a)*self._F_relax, F_new)
        self.soil_wrench = self._F_relax
        self.soil_wrench_raw = F_new
        self.timing["soil_s"] += time.perf_counter() - t0
        self._qd_k = qd                          # the velocities this soil step assumed (damper reference)

    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        try:
            if self.tick % self.soil_ticks == 0:
                self._soil_step()
        except SoilFault as exc:
            self.fault = f"SoilFault: {exc}"
            raise
        self.body_wrench[:] = 0.0
        self.body_wrench[self.coupled_bodies] += self.soil_wrench.astype(np.float32)
        self._coupling_damper()
        if self.drill_state is not None:
            self._drill_motor()
        if self.braces is not None:
            self._brace_forces()
        self._couple_fingers()
        self._force_mode_targets()
        from classroom_arrangement.world import World
        World.step(self)
        if "pipe" in self.scene.static_shapes and self.tick % 4 == 0:
            self._strike_check()

    def _guards(self):
        """classroom World._guards (non-finite state, objects over 20 m/s, MuJoCo warnings) plus the robot's own
        bodies, which that guard leaves out: a robot body below the ground or moving faster than any FR3 link or the TMR
        base can is a fault, not a state to continue from."""
        super()._guards()
        if self.tick % 20:
            return
        rb = getattr(self, "_robot_rows", None)
        if rb is None:
            rb = self._robot_rows = np.array(sorted(self.scene.robot_body.values()), int)
            # the mobile base's own bodies (hull, drive units, casters, battery): a hand may reach into a dug hole
            base = [i for lab, i in self.scene.robot_body.items() if not any(k in lab for k in ("fr3", "spine", "head", "mount"))]
            self._base_rows = np.array(sorted(base), int)
        G = P.ROBOT_GUARD
        bq = self.state.body_q.numpy()
        qd = self.state.body_qd.numpy()
        zb = bq[self._base_rows, 2]
        v = np.linalg.norm(qd[rb, :3], axis=1)
        if zb.min() < L.GRADE - G["below_grade_m"]:
            k = int(self._base_rows[int(np.argmin(zb))])
            raise FloatingPointError(f"Robot base below the ground at tick {self.tick}: {self.model.body_label[k]} at "
                                     f"z {float(bq[k, 2]):.3f} m")
        if v.max() > G["speed_m_s"]:
            k = int(rb[int(np.argmax(v))])
            raise FloatingPointError(f"Unphysical robot speed at tick {self.tick}: {self.model.body_label[k]} "
                                     f"{float(v.max()):.2f} m/s")

    def _coupling_damper(self):
        """Within a soil interval, damp a kinematically coupled body's deviation from the velocity the soil step
        assumed (params.COUPLING_DAMPING): the explicit 2 ms lag otherwise rings against
        the soil's elastic contact stiffness. Zero in steady motion and without soil contact; each component capped
        below MuJoCo's explicit limit (cap x mass or principal inertia / dt), rotation per principal axis (capped by
        the smallest inertia, the shaft's, every axis could not damp a blade swinging about the grip)."""
        kin = getattr(self, "soil_kinematic", None)
        if kin is None or getattr(self, "_qd_k", None) is None or not kin.any():
            return
        C = P.COUPLING_DAMPING
        idx = np.flatnonzero(kin)
        bodies = self.coupled_bodies[idx]
        qd = self.state.body_qd.numpy()[bodies].astype(float)
        q = self.state.body_q.numpy()[bodies].astype(float)
        dv, dw = qd[:, :3] - self._qd_k[idx, :3], qd[:, 3:] - self._qd_k[idx, 3:]
        F = self.soil_wrench[idx]
        dt = 0.0005
        D = np.minimum(np.linalg.norm(F[:, :3], axis=1)/C["v_ref"], C["cap"]*self.coupled_mass[idx]/dt)
        self.body_wrench[bodies, :3] -= (D[:, None]*dv).astype(np.float32)
        R = Rotation.from_quat(q[:, 3:]).as_matrix()
        Ib = np.array([np.diag(np.linalg.inv(m)) for m in self.coupled_invI[idx]])          # principal-ish (body frame)
        tq = np.linalg.norm(F[:, 3:], axis=1)
        dw_b = np.einsum("nji,nj->ni", R, dw)                                                 # world -> body
        Dr = np.minimum(tq[:, None]/C["w_ref"], C["cap"]*Ib/dt)
        self.body_wrench[bodies, 3:] -= np.einsum("nij,nj->ni", R, Dr*dw_b).astype(np.float32)

    # ------------------------------------------------------------------ reflex estimate including soil on robot links
    def _soil_joint_torque(self):
        """J^T of the soil wrenches acting on the arms' own links (MuJoCo dof space), or None."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        tau = None
        for arm, rows in self._arm_link_rows.items():
            for k in rows:
                f = self.soil_wrench[k]
                if not np.any(np.abs(f) > 1e-6):
                    continue
                bi = self._mj_body[self.coupled_bodies[k]]
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mj.mj_jac(m, d, jp, jr, d.xipos[bi], bi)
                t = jp.T @ f[:3] + jr.T @ f[3:]
                tau = t if tau is None else tau + t
        return tau

    def _torque_reflex(self):
        """classroom World._torque_reflex (same filter, thresholds, wrench estimate and stop), with the soil wrench on
        the arm's own links added to the contact-derived external joint torque: a hand pushed into the soil is an
        external force the FR3 senses. A held tool's soil load reaches the arm through the pad contacts as before."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        alpha = CP.DT/CP.REFLEX_FILTER_S
        touching = False
        if d.ncon:
            g = np.asarray(d.contact.geom[:d.ncon])
            touching = bool(self._arm_body_mask[self._geom_body[g]].any())
        tau = None
        if touching and d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            tau = np.zeros(m.nv)
            mj.mj_mulJacTVec(m, d, tau, f)
        ts = self._soil_joint_torque()
        if ts is not None:
            tau = ts if tau is None else tau + ts
        cur = self.control.joint_target_q.numpy()
        for arm, dofs in self._arm_mj_dofs.items():
            idx = [dof for _, dof in dofs]
            ext_s = tau[idx] if tau is not None else np.zeros(7)
            self._tau_f[arm] += alpha*(np.abs(ext_s) - self._tau_f[arm])
            self._tau_sf[arm] += alpha*(ext_s - self._tau_sf[arm])
            moving = any(abs(float(self.targets[self.joints[n]["q"]] - cur[self.joints[n]["q"]])) > 1e-6 for n, _ in dofs)
            if np.abs(self._tau_sf[arm]).max() > 0.2:
                b7, off, Rt7 = self._tcp_in7[arm]
                R7 = d.xmat[b7].reshape(3, 3)
                point = d.xpos[b7] + R7 @ off
                jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mj.mj_jac(m, d, jacp, jacr, point, b7)
                J = np.vstack([jacp[:, idx], jacr[:, idx]])
                wv = np.linalg.lstsq(J.T, self._tau_sf[arm], rcond=None)[0]
                Rh = R7 @ Rt7
                self._wrench_f[arm] = np.abs(np.concatenate([Rh.T @ wv[:3], Rh.T @ wv[3:]]))
            else:
                self._wrench_f[arm] *= 0.
            limits = self._reflex_limits[1 if moving else 0]
            over = self._tau_f[arm] - limits
            k = int(np.argmax(over))
            flim = self._force_limits[1 if moving else 0]
            over_f = self._wrench_f[arm] - flim
            kf = int(np.argmax(over_f))
            if over[k] > 0 or over_f[kf] > 0:
                self._reflex_until[arm] = self.sim_time + CP.REFLEX_RECOVERY_S
                self._freeze(arm)
                self._stop_base_on_reflex()
                if self._new_stop(arm):
                    if over[k] > 0:
                        self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                                  external_torque_nm=float(self._tau_f[arm][k]), limit_nm=float(limits[k]),
                                                  cause=self._reflex_cause(dofs[k][1]) if touching else "soil"))
                    else:
                        axis = ("fx", "fy", "fz", "mx", "my", "mz")[kf]
                        self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="cartesian", joint=f"{arm}_hand/{axis}",
                                                  external_force=float(self._wrench_f[arm][kf]), limit=float(flim[kf]),
                                                  hand_wrench=[round(float(v), 2) for v in self._wrench_f[arm]],
                                                  cause=self._reflex_cause(dofs[int(np.argmax(self._tau_f[arm]))][1])
                                                  if touching else "soil"))

    # ------------------------------------------------------------------ drill
    def _init_drill(self):
        sc = self.scene
        D = P.DRILL
        self.drill_state = dict(gear=0, clutch=getattr(self.config, "clutch", None), omega=0.0, torque=0.0, slip=False,
                                trigger=0.0)
        self._drill_b, self._spindle_b = sc.bodies["drill"], sc.bodies["auger"]
        jq = self.model.joint_q_start.numpy(); jd = self.model.joint_qd_start.numpy()
        lab = list(self.model.joint_label)
        self._trig_q = int(jq[lab.index("drill_trigger_joint")])
        self._spin_qd = int(jd[lab.index("drill_spindle")])
        # rotor inertia about the spindle axis (+x of the auger body): chuck + auger + the joint's armature
        I = np.asarray(self.model.body_inertia.numpy()[self._spindle_b], float).reshape(3, 3)
        self._spin_I = float(I[0, 0]) + 2.0e-4
        # the shared World's joint targets start every joint at mid-range; the trigger's rest is its outer stop (0)
        self.targets[self._trig_q] = 0.0
        ct = self.control.joint_target_q.numpy()
        ct[self._trig_q] = 0.0
        self.control.joint_target_q.assign(ct)

    def _drill_motor(self):
        """DC motor through the gearbox: tau = tau_stall (u - omega/omega_0) with u the PWM duty, electronic brake when
        released (tau opposes omega), clutch torque cap; reaction on the drill body.

        The back-EMF term is a damping of tau_stall / omega_0 ~ 2 N m s/rad on a ~5e-4 kg m2 rotor: applied at the
        start-of-tick speed it is at the explicit limit (c dt / I ~ 2): the torque flips between +- the clutch cap
        every tick with the spindle parked at the cap's speed (414.5 rpm in air). The torque is evaluated at the speed the rotor reaches by the end of the tick under it (implicit
        Euler on the motor's own law, external loads explicit): stable at any gear, free-running at u omega_0."""
        D = P.DRILL
        st = self.drill_state
        q = self.state.joint_q.numpy(); qd = self.state.joint_qd.numpy()
        # trigger return spring: preload + rate, pushing the trigger out of the grip (reaction on the housing)
        pre, k_tr = D["trigger_spring"]
        qt = float(np.clip(q[self._trig_q], 0.0, D["trigger_travel"]))
        f_tr = pre + k_tr*qt
        q_d = self.state.body_q.numpy()[self._drill_b]
        ax = Rotation.from_quat(q_d[3:]).apply((-1., 0., 0.))           # +q direction of the trigger joint
        tb = self.scene.bodies["drill_trigger"]
        self.body_wrench[tb, :3] -= (f_tr*ax).astype(np.float32)
        self.body_wrench[self._drill_b, :3] += (f_tr*ax).astype(np.float32)
        # Electrical input follows measured slider displacement. The return
        # spring and actual finger/housing contacts determine that displacement;
        # commanded grip force is not an electrical surrogate. A small switch
        # dead zone is an authored travel estimate, not a manufacturer curve.
        travel_fraction = qt/D["trigger_travel"]
        u = float(np.clip((travel_fraction - 0.10)/0.90, 0.0, 1.0))
        st.update(trigger_travel_m=qt, trigger_fraction=travel_fraction)
        om = float(qd[self._spin_qd])
        g = st["gear"]
        w0 = D["gears_rpm"][g]*2*math.pi/60
        stall = D["torque_hard"]*D["gears_rpm"][0]/D["gears_rpm"][g]
        dt, I = CP.DT, self._spin_I
        if u > 0:
            om1 = (I*om + dt*stall*u)/(I + dt*stall/w0)  # speed at the end of the tick under the motor alone
            tau = stall*(u - om1/w0)
        else:
            om1 = I*om/(I + dt*0.05)
            tau = -float(np.clip(om1*0.05, -2.0, 2.0))   # brake
        cap = P.DRILL_CLUTCH.get(st["clutch"]) if st["clutch"] is not None else None
        st["slip"] = bool(cap is not None and abs(tau) > cap)
        if cap is not None:
            tau = float(np.clip(tau, -cap, cap))
        st.update(omega=om, torque=float(tau), trigger=u)
        q_b = self.state.body_q.numpy()[self._drill_b]
        axis = Rotation.from_quat(q_b[3:]).apply((1., 0., 0.))
        self.body_wrench[self._spindle_b, 3:] += (tau*axis).astype(np.float32)
        self.body_wrench[self._drill_b, 3:] -= (tau*axis).astype(np.float32)

    # ------------------------------------------------------------------ post braces
    def set_braces(self, anchors, k=2.0e4, c=150.0):
        """Two diagonal 1x2 braces from the post (attachment points in the post frame) to stakes (world points),
        each a spring-damper of stiffness k (N/m, estimate: a nailed 1x2 brace) at its initial length."""
        b = self.scene.bodies["post"]
        p, R = self.body_pose(b)
        self.braces = [dict(local=np.asarray(loc, float), anchor=np.asarray(anc, float),
                            L0=float(np.linalg.norm(p + R @ np.asarray(loc, float) - np.asarray(anc, float))), k=k, c=c)
                       for loc, anc in anchors]

    def release_braces(self):
        self.braces = None

    def _brace_forces(self):
        b = self.scene.bodies["post"]
        q = self.state.body_q.numpy()[b]; qd = self.state.body_qd.numpy()[b]
        p, R = q[:3].astype(float), Rotation.from_quat(q[3:]).as_matrix()
        com = p + R @ self.model.body_com.numpy()[b]
        for br in self.braces:
            x = p + R @ br["local"]
            d = x - br["anchor"]
            Ln = float(np.linalg.norm(d))
            u = d/max(Ln, 1e-9)
            v = qd[:3] + np.cross(qd[3:], x - com)
            f = -(br["k"]*(Ln - br["L0"]) + br["c"]*float(v @ u))*u
            self.body_wrench[b, :3] += f.astype(np.float32)
            self.body_wrench[b, 3:] += np.cross(x - com, f).astype(np.float32)

    # ------------------------------------------------------------------ strikes on the irrigation lateral
    def _strike_check(self):
        pipe = self.scene.static_shapes["pipe"]
        g = self._shape_geom.get(pipe)
        if g is None:
            return
        m, d, mj = self.mjm, self.mjd, self._mujoco
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if g not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == g else c.geom1
            bi = int(m.geom_bodyid[other])
            mj.mj_contactForce(m, d, i, f6)
            v6 = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, bi, v6, 0)
            speed = float(np.linalg.norm(v6[3:]))
            fn = abs(float(f6[0]))
            self.strike_contacts += 1
            if fn > P.STRIKE["force_n"] or speed > P.STRIKE["speed_m_s"]:
                name = self._mujoco.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bi)
                if not self.strikes or self.tick - self.strikes[-1]["tick"] > 200:
                    self.strikes.append(dict(tick=int(self.tick), t=round(self.sim_time, 4), body=name, force_n=round(fn, 1),
                                             speed_m_s=round(speed, 3)))


# ------------------------------------------------------------------ particles of an episode
def pile_positions(n, center, radius, h, slope_deg=30.0, base_z=0.0, seed=0):
    """n particle positions of a loose mound (packing CRITICAL_FRACTION) on a flat base."""
    f = P.CRITICAL_FRACTION
    hl = h/f**(1/3)
    vol = n*hl**3
    t = math.tan(math.radians(slope_deg))
    R = radius
    vc = math.pi*R*R*R*t/3
    if vol > vc:
        R = (3*vol/(math.pi*t))**(1/3)
    Hm = R*t
    m = int(math.ceil(2*R/hl)) + 2
    g = (np.arange(m) - (m - 1)/2)*hl
    zz = (np.arange(int(math.ceil(Hm/hl)) + 2) + 0.5)*hl
    X, Y, Z = np.meshgrid(g, g, zz, indexing="ij")
    rr = np.hypot(X, Y)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    key = Z.ravel() + (rr.ravel()/max(R, 1e-6))*Hm      # fill from the bottom centre outward: a cone of the slope
    pts = pts[np.argsort(key, kind="stable")[:n]]
    rng = np.random.default_rng(seed)
    pts = pts + rng.uniform(-0.15, 0.15, pts.shape)*hl
    pts[:, 0] += center[0]; pts[:, 1] += center[1]; pts[:, 2] += base_z
    return pts.astype(np.float32)


def build_particles(spec, world=None):
    """The episode's particles: the layered block lattice minus every rigid inclusion (stones, root ball, post, pipe,
    valve box; 0.4 h margin) and minus any pre-dug excavation, whose particles are relocated loose onto the tarp."""
    voxel = spec["voxel"]
    h = voxel/P.PPC_AXIS
    b = spec["block"]
    pos, mat = block_particles(b["lo"], b["hi"], spec["layers"], h, seed=spec["seed"], undulation=0.01)
    keep = np.ones(len(pos), bool)
    shapes = []
    for name, it in spec["items"].items():
        if it["kind"] == "stone":
            x, y, z, q = it["pose"]
            shapes.append(dict(kind="ellipsoid", size=it["radii"], pos=(x, y, z), quat=q))
        elif it["kind"] == "rootball":
            x, y, z, q = it["pose"]
            R = P.ROOTBALL
            shapes.append(dict(kind="cylinder", size=(R["diameter"]/2, R["height"]/2), pos=(x, y, z + R["height"]/2), quat=q))
        elif it["kind"] == "post":
            x, y, z, q = it["pose"]
            a = P.POST["section"]
            shapes.append(dict(kind="box", size=(a/2, a/2, P.POST["length"]/2), pos=(x, y, z + P.POST["length"]/2), quat=q))
    if "pipe" in spec:
        pp = spec["pipe"]
        shapes.append(dict(kind="capsule", size=(pp["r"], 3.0), pos=(0., pp["y"], pp["z"]),
                           quat=tuple(Rotation.from_euler("y", 90, degrees=True).as_quat())))
    if "valve_box" in spec["props"]:
        x, y = spec["props"]["valve_box"]["pos"]
        shapes.append(dict(kind="cylinder", size=(0.127, 0.125), pos=(x, y, L.GRADE - 0.125), quat=(0, 0, 0, 1)))
    for s in shapes:
        keep &= sdf(pos, s) > 0.4*h
    loose = np.zeros(len(pos), bool)
    origin = np.zeros(len(pos), np.int8)
    for cut in spec.get("cuts", []):
        inside = L.dug_wall(pos, cut) if "hand_seed" in cut else (sdf(pos, cut) < 0)
        sel = inside & keep
        idx = np.flatnonzero(sel)
        if not len(idx):
            continue
        dest = cut.get("dest")
        if dest is None:
            keep[idx] = False
            continue
        pts = pile_positions(len(idx), dest["center"], dest.get("radius", 0.25), h, base_z=dest["z"],
                             slope_deg=dest.get("slope", 30.0), seed=spec["seed"] + 3)
        rng = np.random.default_rng(spec["seed"] + 11)
        perm = rng.permutation(len(idx))          # spoil is mixed as it is dug and thrown
        pos[idx] = pts[perm]
        loose[idx] = True
        origin[idx] = ORIGIN_PILE
    for extra in spec.get("fills", []):
        # loose material placed into the yard (bedding sand in a pail, a gravel base course in a hole)
        pts = extra["points"](h)
        pos = np.concatenate([pos, pts]); mat = np.concatenate([mat, np.full(len(pts), M.NAMES.index(extra["material"]), np.int32)])
        keep = np.concatenate([keep, np.ones(len(pts), bool)])
        loose = np.concatenate([loose, np.full(len(pts), not extra.get("bank", False), bool)])   # bank: tamped fill
        origin = np.concatenate([origin, np.full(len(pts), ORIGIN_PILE, np.int8)])
    out = dict(pos=pos[keep], mat=mat[keep], loose=loose[keep], origin=origin[keep])
    out["paint"] = paint_marks(out["pos"], spec, h)
    return out


def paint_marks(pos, spec, h):
    """Marking paint on the top layer of particles (render only; it travels with the soil): 1 white (proposed dig),
    2 purple (irrigation lateral, APWA)."""
    paint = np.zeros(len(pos), np.int8)
    top = pos[:, 2] > L.GRADE - 1.2*h
    xy = pos[:, :2]
    for m in spec["marks"]:
        code = 2 if m.get("color") == "purple" else 1
        if m["kind"] == "dot":
            sel = np.linalg.norm(xy - np.asarray(m["pos"]), axis=1) < m["r"]
        elif m["kind"] == "line":
            a, b = np.asarray(m["a"]), np.asarray(m["b"])
            d = b - a
            t = np.clip(((xy - a) @ d)/float(d @ d), 0, 1)
            sel = np.linalg.norm(xy - (a + t[:, None]*d), axis=1) < m["w"]/2
        elif m["kind"] == "square":
            lo, hi = np.asarray(m["lo"]), np.asarray(m["hi"])
            inner = (xy > lo + m["w"]).all(1) & (xy < hi - m["w"]).all(1)
            sel = (xy > lo).all(1) & (xy < hi).all(1) & ~inner
        else:
            continue
        paint[top & sel] = code
    return paint

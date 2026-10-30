"""Rigid world (Newton 1.5 + CPU MuJoCo) with the mixture continuum coupled into every tick.

Robot interface: the one qualified in barcode_scanning, wine_pouring and drilling on the same FR3 arms and Franka Hand,
taken here in drilling's form: arm targets are followed by a time-synchronised jerk-limited trajectory within Franka's
joint velocity, acceleration and jerk limits; the desired joint torque (servo plus any fed-forward wrench) is capped at
the joint's rating while the arm's own weight and the load its controller has been told it carries (set_load) are
compensated outside it; a joint-velocity reflex; the libfranka collision reflex; the Franka Hand's grasp (close at 50 mm/s until blocked,
then hold the commanded force); hard joint stops; MuJoCo no-slip.

Step order per tick: appliances set their drives (mixer motor, detents); the robot's motion generator and servos
advance; the continuum receives every body's pose and twist, advances through its substeps against the moving solids
and returns the wrench it put on each body; MuJoCo steps with those wrenches; reflexes and fault checks run. Every
MATERIAL_EVERY ticks the points' properties, slow chemistry, species exchange and heat advance.

Coupling (as in wine_pouring): a light vessel holding heavier contents is unstable under staggered
coupling, so the contents' mass and centre of mass are lumped into the vessel's rigid body each tick and the rigid
solver receives the residual wrench, low-pass filtered at 8 ms; tools receive the filtered wrench directly.
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization

from . import params as P
from . import materials as M
from .config import Config
from .scene import Scene

FINGER_LIMIT_MARGIN = 1e-3
MATERIAL_EVERY = 10                 # ticks between property refresh / exchange / heat (materials.REFRESH = 10 ms)
RESIDUAL_TAU_S = 0.008
ACC_TAU_S = 0.004


_ACTIVE = None                      # the world whose MuJoCo step is running (MuJoCo's control callback is process-wide)


def _mj_control(m, d):
    w = _ACTIVE
    if w is not None:
        w._contact_rule(d)
        w._pad_stick(d)


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


class World:
    scene_cls = Scene               # hook for environments built on this station (dough_cutting); unchanged here

    def __init__(self, config=None):
        self.config = c = config or Config()
        self.newton, self.wp = newton, wp = require_engine()
        t0 = time.perf_counter()
        self.scene = sc = self.scene_cls(c, newton, wp)
        b = sc.b
        self.dt = c.rigid_dt
        import mujoco
        self._mujoco = mujoco
        with wp.ScopedDevice("cpu"):
            self.model = b.finalize(device="cpu")
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (c.contact_timeconst, 1.)
            mass = self.model.body_mass.numpy()
            sb = np.asarray(b.shape_body)
            pads = {int(s) for s in sc.pad_shapes}
            imp = self.model.mujoco.geom_solimp.numpy()
            for i in range(len(sb)):
                if i in pads:
                    ref[i] = (P.PAD_TIMECONST*math.sqrt(float(mass[int(sb[i])])/P.PAD_TIMECONST_REF_MASS), 1.)
                else:
                    imp[i][:3] = c.contact_solimp
            self.model.mujoco.solref.assign(ref)
            self.model.mujoco.geom_solimp.assign(imp)
            lo = self.model.joint_limit_lower.numpy()
            labels = list(self.model.joint_label)
            qd_start = self.model.joint_qd_start.numpy()
            for j, lab in enumerate(labels):
                if "finger_joint" in lab:
                    lo[qd_start[j]] -= FINGER_LIMIT_MARGIN
            self.model.joint_limit_lower.assign(lo)
            self.rig = sc.rig_pending.bind(self.model) if hasattr(sc, "rig_pending") else None
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                      enable_multiccd=True, integrator="implicitfast", cone="elliptic",
                                                      solver="newton", iterations=c.solver_iterations, ls_iterations=20,
                                                      impratio=c.solver_impratio, nconmax=c.contact_capacity,
                                                      njmax=c.contact_capacity*3, update_data_interval=0)
            mjm = self.solver.mj_model
            for j in range(mjm.njnt):
                if not mjm.jnt_limited[j]:
                    continue
                name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                d0, dmax, mid, power = P.STOP_SOLIMP
                width = 5e-4 if ("finger_joint" in name or mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE) else 1e-3
                mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                mjm.jnt_solimp[j] = (d0, dmax, width, mid, power)
            # the hand's fingers are one mechanism: the imported mimic is a joint equality at MuJoCo's default softness
            # until it is given the hardness of a gear (params.FINGER_GEAR_SOLREF)
            self.finger_gears = []
            for e in range(mjm.neq):
                if mjm.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or ""
                         for j in (mjm.eq_obj1id[e], mjm.eq_obj2id[e])]
                if all("finger_joint" in n for n in names):
                    mjm.eq_solref[e] = P.FINGER_GEAR_SOLREF
                    mjm.eq_solimp[e] = P.FINGER_GEAR_SOLIMP
                    self.finger_gears.append(e)
            mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            mjm.opt.ls_iterations = int(c.solver_ls_iterations)
            mjm.opt.ls_tolerance = float(c.solver_ls_tolerance)
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads:
                    mjm.geom_priority[g] = 1
                    mjm.geom_condim[g] = 4
                    mjm.geom_friction[g][1] = P.PAD_TORSIONAL_M
                    # the pad's friction rows remove a slip within a tick; its normal stiffness stays the anchored one
                    # (MuJoCo's direct form: stiffness 1/timeconst^2, rate dmax/dt; params.PAD_SLIP_RATE_DT)
                    tc_ = float(mjm.geom_solref[g][0])
                    if tc_ > 0. and P.PAD_SLIP_RATE_DT > 0.:
                        mjm.geom_solref[g] = (-1./tc_**2, -P.PAD_SLIP_RATE_DT*float(mjm.geom_solimp[g][1])/c.rigid_dt)
            self._geom_pad = np.zeros(mjm.ngeom, bool)
            self._pad_stick_k = np.zeros(mjm.ngeom)
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads and float(mjm.geom_solref[g][0]) < 0.:
                    self._geom_pad[g] = True
                    self._pad_stick_k[g] = P.PAD_STICK["shear_ratio"]*(-float(mjm.geom_solref[g][0]))/float(
                        mjm.geom_solimp[g][1])**2
            self._stick, self._stick_seen = {}, set()
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        mjm = self.solver.mj_model
        j2d = np.asarray(self.solver.mjc_jnt_to_newton_dof.numpy()).reshape(-1)
        self.mj_joint_of_dof = {int(dof): jm for jm, dof in enumerate(j2d[:mjm.njnt]) if dof >= 0}
        m2n = np.asarray(self.solver.mjc_body_to_newton.numpy())[0]
        self._mj_to_newton = {j: int(n) for j, n in enumerate(m2n)}
        self._newton_to_mj = {n: j for j, n in self._mj_to_newton.items() if n >= 0}
        self.body_com = self.model.body_com.numpy().astype(np.float64)
        self.body_wrench = np.zeros((self.model.body_count, 6))
        self.tq = self.model.joint_target_q.numpy().copy()
        self.tqd = np.zeros(self.model.joint_dof_count, np.float32)
        self.jf = np.zeros(self.model.joint_dof_count, np.float32)
        self.tick, self.fault = 0, None
        self.events = []
        self.fixture = None          # ASSISTED callback (probes/demos only), never reachable from the policy API
        self.phantom_log = []
        self.reflexes, self.refused_targets = [], 0
        self._reflex_until, self._reflex_fired, self._tau_ext = {}, {}, {}
        self.timing = dict(build_s=0., step_s=0., continuum_s=0., material_s=0., rigid_s=0.)
        self.joints = dict(self.rig.joints) if self.rig else {}
        if self.rig:
            self._robot_setup(mjm)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.state.joint_q.assign(self.model.joint_q)
        self.targets = self.tq.copy()
        if self.rig:
            self._ready_pose()
        robots = set(sc.robot_bodies)
        spinning = {bb for t in sc.tools.values() for bb in getattr(t, "beaters", [])}
        self._objects = np.array([bb for bb in range(self.model.body_count) if bb not in robots], dtype=int)
        self._spin_exempt = np.array(sorted(spinning), dtype=int)
        self.sync_solver()
        self.refresh()
        for t in sc.tools.values():
            t.bind(self)
        self.cont = None
        self.heat = None
        self.cont_wrench = np.zeros((self.model.body_count, 6))
        self.tool_force = {}
        self.timing["build_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ continuum
    def attach_material(self, recipe, points, species, temp=20.0, capacity=None, headroom=0.04):
        """Create the continuum around the task's vessel and seed it. points: (n, 3) world; species: (n, NS) kg."""
        from .continuum import Continuum, PAD
        c, sc = self.config, self.scene
        v = sc.vessel
        dx = float(c.cell_m)
        shell = 5.0*dx
        # the grid region: the world-aligned box around a cylinder fixed in the vessel (its inside, its wall as the
        # mixture sees it, and room above the rim), at whatever attitude the vessel has (_set_region, every tick)
        self._region = dict(radius=v.rim_r + shell + 0.02, z_lo=-shell - 0.005, z_hi=v.depth + headroom)
        lo, hi = self._region_box()
        cells = tuple(int(math.ceil((h - l)/dx)) + 2*PAD + 1 for l, h in zip(lo, hi))
        n_markers = sum(len(mk["x"]) for t in sc.tools.values() for mk in t.markers)
        self.cont = cont = Continuum(capacity or len(points), dx=dx, sound=recipe.sound, grid_cells=cells,
                                     device=c.device, max_markers=max(8, n_markers))
        cont.set_recipe(recipe)
        self.recipe = recipe
        v.collider(cont, mu_dry=math.tan(math.radians(10.5)))
        for t in sc.tools.values():
            t.register(cont, mu_dry=math.tan(math.radians(10.5)))
        from .continuum import KIND_PLANE, KIND_BOX
        _, pos, half, _, quat = next(s for s in sc.static_boxes if s[0] == "counter_top")
        cont.counter_col = cont.add_collider(KIND_BOX, pos=pos, quat=quat, size=half, mu_dry=0.4, name="counter")
        cont.add_collider(KIND_PLANE, pos=(0., 0., sc.counter_plane["floor"]), mu_dry=0.4, name="floor")
        # the robot's fingers and hands are solids for the mixture too: a hand dipped into batter displaces it and
        # feels it. Fingers: their collision boxes; hand and wrist link: the bounding box of each collision mesh in
        # its body's frame (loose by the meshes' rounding, a few millimetres)
        self.robot_colliders = []
        b = sc.b
        for arm, fingers in getattr(self, "_arm_fingers", {}).items():
            bodies = [self._mj_to_newton[x] for x in fingers] + [self._mj_to_newton[self._arm_hand[arm]]]
            for i in range(b.shape_count):
                body = int(b.shape_body[i])
                if body not in bodies or not (int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES)):
                    continue
                tf = np.asarray(b.shape_transform[i], float)
                scale = np.asarray(b.shape_scale[i], float)
                if b.shape_type[i] == self.newton.GeoType.BOX:
                    pos, quat, half = tf[:3], tf[3:], scale
                elif b.shape_source[i] is not None and hasattr(b.shape_source[i], "vertices"):
                    vtx = np.asarray(b.shape_source[i].vertices, float)*scale
                    lo, hi = vtx.min(0), vtx.max(0)
                    pos, quat, half = tf[:3] + Rotation.from_quat(tf[3:]).apply(0.5*(lo + hi)), tf[3:], 0.5*(hi - lo)
                else:
                    continue
                k = cont.add_collider(KIND_BOX, body=body, pos=tuple(pos), quat=tuple(quat),
                                      size=tuple(np.maximum(half, dx)), mu_dry=0.5, name=f"{arm}_{b.shape_label[i]}")
                self.robot_colliders.append(k)
        # wire markers
        self._markers = []
        k0 = 0
        for name, t in sc.tools.items():
            for mk in t.markers:
                n = len(mk["x"])
                self._markers.append(dict(tool=name, body=mk["body"], x=np.asarray(mk["x"], float),
                                          t=np.asarray(mk["t"], float), sl=slice(k0, k0 + n)))
                k0 += n
        if k0:
            cont.markers = k0
            cont.mlen.assign(np.concatenate([mk["length"] for t in sc.tools.values() for mk in t.markers]).astype(np.float32))
            cont.mdia.assign(np.concatenate([mk["dia"] for t in sc.tools.values() for mk in t.markers]).astype(np.float32))
            cont.mrun.assign(np.concatenate([np.full(len(mk["x"]), mk.get("run", 0.03)) for t in sc.tools.values()
                                             for mk in t.markers]).astype(np.float32))
        self._marker_u = None
        self._set_region()
        cont.set_points(points, species, temp=temp)
        cont.refresh(0.0)
        # contents lumped into the vessel's rigid body
        mjm = self.solver.mj_model
        jb = self._newton_to_mj[v.body]
        self._lump = dict(mj=jb, mass=float(mjm.body_mass[jb]), ipos=np.array(mjm.body_ipos[jb]),
                          inertia=np.array(mjm.body_inertia[jb]), m_c=0., c_c=np.zeros(3), a_lag=np.zeros(3),
                          v_prev=None, res=np.zeros(6))
        self._tool_res = {}
        self._sums = self.wp.zeros(10, dtype=float, device=c.device)
        if sc.layout.get("hob") is not None:
            from .heat import Hob
            self.heat = Hob(self)
        self._push_bodies()
        return cont

    def _region_box(self):
        """World-aligned box around the region cylinder of the vessel at its present pose."""
        v, g = self.scene.vessel, self._region
        q = self.bq[v.body]
        axis = Rotation.from_quat(q[3:]).apply([0., 0., 1.])
        p0, p1 = q[:3] + g["z_lo"]*axis, q[:3] + g["z_hi"]*axis
        side = g["radius"]*np.sqrt(np.clip(1. - axis*axis, 0., 1.))
        return np.minimum(p0, p1) - side, np.maximum(p0, p1) + side

    def _set_region(self):
        """The grid follows the vessel in position AND attitude. (With an upright box that moved with the vessel and
        did not turn with it, tipped past 15-25 degrees, the low side of the vessel's inside would lie below the box,
        and the mixture there would leave the grid and fall through the wall as ballistic points: 75 % of the
        saucepan's sauce gone at 32 degrees.)"""
        v = self.scene.vessel
        lo, hi = self._region_box()
        self.cont.set_box(lo, hi, counter=self.scene.counter_plane)
        q = self.bq[v.body]
        self.cont.envelope = dict(pos=q[:3], quat=q[3:], radius=v.rim_r + v.render_wall + 0.5*self.cont.dx, top=v.depth,
                                  bottom=-(v.render_wall + 0.5*self.cont.dx))

    def _push_bodies(self):
        cont = self.cont
        cont.set_bodies(self.bq, self.bqd, self.body_com)
        if cont.markers:
            x = np.zeros((cont.markers, 3)); u = np.zeros_like(x); t = np.zeros_like(x)
            for mk in self._markers:
                b = mk["body"]
                R = Rotation.from_quat(self.bq[b, 3:])
                com = self.bq[b, :3] + R.apply(self.body_com[b])
                xw = self.bq[b, :3] + R.apply(mk["x"])
                x[mk["sl"]] = xw
                u[mk["sl"]] = self.bqd[b, :3] + np.cross(self.bqd[b, 3:], xw - com)
                t[mk["sl"]] = R.apply(mk["t"])
            self._marker_x = x
            cont.mx.assign(x.astype(np.float32)); cont.mu.assign(u.astype(np.float32)); cont.mt.assign(t.astype(np.float32))
            cont.mf.zero_()
            # what a film on each wire feels: gravity less the wire's own acceleration (continuum.k_film)
            g_ = np.array([0., 0., -P.GRAVITY])
            if self._marker_u is not None and len(self._marker_u) == len(u):
                acc = np.clip((u - self._marker_u)/self.dt, -2000., 2000.)
                cont.macc.assign(np.concatenate([np.linalg.norm(g_ - acc, axis=1), np.full(cont.max_markers - len(u), P.GRAVITY)])
                                 .astype(np.float32))
            self._marker_u = u.copy()

    def _couple(self, wrench):
        """Advance the continuum one tick and return the body wrenches (force at each body's centre of mass, torque
        about it) it leaves on the rigid world, filtered; the vessel's contents ride in its rigid body."""
        cont, sc, c = self.cont, self.scene, self.config
        v = sc.vessel
        self._set_region()
        self._push_bodies()
        self._lump_contents()
        cont.step()
        f, tq = cont.wrenches()
        raw = np.zeros((self.model.body_count, 6))
        for k, col in enumerate(cont._col):
            b = col["body"]
            if b < 0:
                continue
            R = Rotation.from_quat(self.bq[b, 3:])
            com = self.bq[b, :3] + R.apply(self.body_com[b])
            raw[b, :3] += f[k]
            raw[b, 3:] += tq[k] + np.cross(cont._col_pos0[k] - com, f[k])
        if cont.markers:
            mf = cont.mf.numpy().astype(np.float64)/cont.substeps
            for mk in self._markers:
                b = mk["body"]
                R = Rotation.from_quat(self.bq[b, 3:])
                com = self.bq[b, :3] + R.apply(self.body_com[b])
                fk = mf[mk["sl"]]
                raw[b, :3] += fk.sum(0)
                raw[b, 3:] += np.cross(self._marker_x[mk["sl"]] - com, fk).sum(0)
        alpha = min(1., c.rigid_dt/RESIDUAL_TAU_S)
        out = np.zeros_like(raw)
        for b in np.nonzero(np.abs(raw).sum(1))[0].tolist() + list(self._tool_res):
            if b == v.body:
                continue
            r = self._tool_res.get(b, np.zeros(6))
            r = r + alpha*(raw[b] - r)
            self._tool_res[b] = r
            out[b] = r
        # the vessel: residual about the combined centre of mass (the lumped contents' weight and inertia removed)
        L = self._lump
        mjm = self.solver.mj_model
        Rv = Rotation.from_quat(self.bq[v.body, 3:])
        c_g = self.bq[v.body, :3] + Rv.apply(self.body_com[v.body])
        c_new = self.bq[v.body, :3] + Rv.apply(np.asarray(mjm.body_ipos[L["mj"]]))
        g = np.array([0., 0., -P.GRAVITY])
        F = raw[v.body, :3]
        tau = raw[v.body, 3:] + np.cross(c_g - c_new, F)
        Fc = L["m_c"]*(L["a_lag"] - g)
        res = np.r_[F + Fc, tau + np.cross(L["c_c"] - c_new, Fc)]
        L["res"] = L["res"] + alpha*(res - L["res"])
        out[v.body] = L["res"]
        self.cont_wrench = raw
        return wrench + out

    def _lump_contents(self):
        """Mass and centre of mass of the points riding in the vessel into its rigid body (MuJoCo mass, inertial
        position and inertia), and the lagged acceleration of that centre for the residual."""
        from .continuum import k_vessel_sums
        cont, v, L = self.cont, self.scene.vessel, self._lump
        q = self.bq[v.body]
        self._sums.zero_()
        self.wp.launch(k_vessel_sums, dim=cont.capacity, inputs=[
            cont.x, cont.flags, cont.mass, self.wp.vec3(*q[:3]), self.wp.quat(*q[3:]), float(v.rim_r + 0.01),
            float(v.depth + 0.02), self._sums], device=cont.device)
        s = self._sums.numpy().astype(np.float64)
        m_c = float(s[0])
        R = Rotation.from_quat(q[3:])
        mjm = self.solver.mj_model
        if m_c > 1e-6:
            c_c = s[1:4]/m_c
            S = np.array([[s[4], s[7], s[8]], [s[7], s[5], s[9]], [s[8], s[9], s[6]]]) - m_c*np.outer(c_c, c_c)
            I_c = np.trace(S)*np.eye(3) - S
            ax = R.as_matrix()[:, 2]
            Pp = np.eye(3) - np.outer(ax, ax)          # a paste turns with its vessel only through wall friction: the
            I_c = Pp @ I_c @ Pp                        # axial moment is not lumped (as in wine_pouring)
            r_c = R.inv().apply(c_c - q[:3])
            Mt = L["mass"] + m_c
            ipos = (L["mass"]*L["ipos"] + m_c*r_c)/Mt
            I = np.diag(L["inertia"]).copy()
            for mass_, pos in ((L["mass"], L["ipos"]), (m_c, r_c)):
                d = pos - ipos
                I = I + mass_*(np.dot(d, d)*np.eye(3) - np.outer(d, d))
            I = I + R.as_matrix().T @ I_c @ R.as_matrix()
            mjm.body_mass[L["mj"]] = Mt
            mjm.body_ipos[L["mj"]] = ipos
            mjm.body_inertia[L["mj"]] = np.maximum(np.diag(I), 1e-7)
        else:
            c_c = q[:3].copy()
            mjm.body_mass[L["mj"]] = L["mass"]
            mjm.body_ipos[L["mj"]] = L["ipos"]
            mjm.body_inertia[L["mj"]] = L["inertia"]
        com = q[:3] + R.apply(self.body_com[v.body])
        v_pt = self.bqd[v.body, :3] + np.cross(self.bqd[v.body, 3:], c_c - com)
        if L["v_prev"] is not None:
            a = (v_pt - L["v_prev"])/self.dt
            L["a_lag"] = L["a_lag"] + min(1., self.dt/ACC_TAU_S)*(a - L["a_lag"])
        L["v_prev"], L["m_c"], L["c_c"] = v_pt, m_c, c_c

    # ------------------------------------------------------------------ robot
    def _robot_setup(self, mjm):
        mujoco = self._mujoco
        sc = self.scene
        mj = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self._arm_mj_dofs = {}
        for name in self.joints:
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mname, dof in mj.items() if mname and mname.startswith(arm + "_") and mname.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._finger_speed = {j: P.FRANKA_HAND_FINGER_SPEED for arm in sc.arm_hw for j in self.gripper_joints(arm)}
        mjname = lambda b: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self._arm_fingers = {arm: [b for b in range(mjm.nbody) if mjname(b).startswith(arm + "_") and "finger" in mjname(b)]
                             for arm in self._arm_mj_dofs}
        self._arm_hand = {arm: int(mjm.body_parentid[fs[0]]) for arm, fs in self._arm_fingers.items()}
        self._prev_cvel = None
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched, self._grip_ref = {}, {}
        self.grip_applied = {}
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {n: float(ke[self.joints[n]["dof"]]) for arm in sc.arm_hw for n in self.gripper_joints(arm)}
        self.wrench_cmd = {}
        self._tcp_off = {}
        self.load_kg = {}
        self.load_cfg = {}               # arm -> the load its controller is told it carries (set_load)
        self._arm = {}
        for arm, dofs in self._arm_mj_dofs.items():
            names = [n for n, _ in dofs]
            panda = names[0].split("/", 1)[1].startswith("panda_")
            inp, out = InputParameter(len(names)), OutputParameter(len(names))
            inp.max_velocity = [P.ROBOT_VELOCITY_FRACTION*float(self.joints[n]["velocity_limit"]) for n in names]
            inp.max_acceleration = list(P.PANDA_MAX_JOINT_ACC if panda else P.FR3_MAX_JOINT_ACC)
            inp.max_jerk = list(P.PANDA_MAX_JOINT_JERK if panda else P.FR3_MAX_JOINT_JERK)
            inp.synchronization = Synchronization.Time
            self._arm[arm] = dict(
                names=names, otg=Ruckig(len(names), self.config.rigid_dt), inp=inp, out=out, moving=False,
                q=np.array([self.joints[n]["q"] for n in names]), dof=np.array([self.joints[n]["dof"] for n in names]),
                jnt=np.array([int(mjm.dof_jntid[mdof]) for _, mdof in dofs]),
                lim=np.array([float(self.joints[n]["effort_limit"]) for n in names]),
                ff=np.zeros(len(names)), load=np.zeros(len(names)), load_want=np.zeros(len(names)))
            mjm.jnt_actfrcrange[self._arm[arm]["jnt"]] = np.stack([-self._arm[arm]["lim"], self._arm[arm]["lim"]], 1)

    def controller_state(self):
        out = {}
        for arm, g in self._arm.items():
            inp = g["inp"]
            out[arm] = dict(p=list(inp.current_position), v=list(inp.current_velocity), a=list(inp.current_acceleration),
                            target=list(inp.target_position), moving=g["moving"], ff=g["ff"].copy(),
                            load=g["load"].copy(), load_want=g["load_want"].copy(),
                            range=self.solver.mj_model.jnt_actfrcrange[g["jnt"]].copy())
        return dict(arms=out, wrench_cmd=dict(self.wrench_cmd), prev_cvel=dict(self._prev_cvel or {}),
                    stick={k: [v[0].copy(), v[1].copy()] for k, v in self._stick.items()}, stick_seen=set(self._stick_seen),
                    load={a: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in L.items()}
                          for a, L in self.load_cfg.items()})

    def set_controller_state(self, st):
        for arm, a in st["arms"].items():
            g = self._arm[arm]
            inp = g["inp"]
            inp.current_position, inp.current_velocity, inp.current_acceleration = a["p"], a["v"], a["a"]
            inp.target_position = a["target"]
            g["moving"], g["ff"], g["load"], g["load_want"] = a["moving"], a["ff"].copy(), a["load"].copy(), \
                a["load_want"].copy()
            self.solver.mj_model.jnt_actfrcrange[g["jnt"]] = a["range"]
        self.wrench_cmd = dict(st["wrench_cmd"])
        self._prev_cvel = dict(st["prev_cvel"])
        self._stick = {k: [v[0].copy(), v[1].copy()] for k, v in st.get("stick", {}).items()}
        self._stick_seen = set(st.get("stick_seen", ()))
        self.load_cfg = {a: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in L.items()}
                         for a, L in st.get("load", {}).items()}
        self.load_kg = {a: round(L["mass"], 4) for a, L in self.load_cfg.items() if L["mass"] > 0.}

    def _hold_command(self, arm, q):
        g = self._arm[arm]
        g["inp"].current_position = [float(v) for v in q]
        g["inp"].current_velocity = [0.]*len(q)
        g["inp"].current_acceleration = [0.]*len(q)
        g["moving"] = True
        self.tq[g["q"]] = q
        self.tqd[g["dof"]] = 0.

    def _tcp_offset(self, arm):
        """The TCP (the rig's `arm/tcp` frame) in the frame of the body that carries the hand (as in drilling)."""
        if arm not in self._tcp_off:
            nb = self._mj_to_newton[self._arm_hand[arm]]
            q = self.bq[nb]
            tcp = np.asarray(self.rig.frame_pose(f"{arm}/tcp", self.state)["position_m"], float)
            self._tcp_off[arm] = Rotation.from_quat(q[3:]).inv().apply(tcp - q[:3])
        return self._tcp_off[arm]

    def _ready_pose(self):
        q = self.model.joint_q.numpy()
        for name, hw in self.scene.arm_hw.items():
            pose = self.scene.layout.get("ready_pose", {}).get(name, P.READY_POSE)
            for joint, v in zip(hw["arm_joints"] + hw["gripper_joints"], pose):
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = self.tq[self.joints[key]["q"]] = self.targets[self.joints[key]["q"]] = v
        for arm, g in self._arm.items():
            self._hold_command(arm, q[g["q"]])
        self.model.joint_q.assign(q)
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def gripper_joints(self, arm):
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def grip_applied_n(self, arm):
        f = [abs(v) for (a, _), v in self.grip_applied.items() if a == arm]
        return float(min(f)) if len(f) == len(self.gripper_joints(arm)) and self.grip_force.get(arm) else 0.

    def set_grip_force(self, arm, force):
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]
            for key in [k for k in self._grip_ref if k[0] == arm]:
                self.tq[self.joints[key[1]]["q"]] = self.jq[self.joints[key[1]]["q"]]
                del self._grip_ref[key]
                self.grip_applied.pop(key, None)

    def set_targets(self, targets):
        """Joint position targets. An arm stopped by its collision reflex refuses new arm targets for
        P.REFLEX_RECOVERY_S (finger targets pass), as on the real robot until an error recovery."""
        latched = {arm for arm, until in self._reflex_until.items() if self.sim_time < until}
        if latched:
            kept = {n: v for n, v in targets.items() if n.split("/", 1)[0] not in latched or "finger" in n}
            self.refused_targets += len(targets) - len(kept)
            targets = kept
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Not a robot actuator: {name}")
            spec = self.joints[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value}")
        for name, value in targets.items():
            self.targets[self.joints[name]["q"]] = value
            if "finger" in name:
                for key in [k for k in self.grasp_latched if k[1] == name]:
                    del self.grasp_latched[key]

    # ------------------------------------------------------------------ module interface
    def refresh(self):
        self.bq = self.state.body_q.numpy().astype(np.float64)
        self.bqd = self.state.body_qd.numpy().astype(np.float64)
        self.jq = self.state.joint_q.numpy().astype(np.float64)
        self.jqd = self.state.joint_qd.numpy().astype(np.float64)

    def body_pose(self, body):
        body = self.scene.bodies[body] if isinstance(body, str) else int(body)
        q = self.bq[body]
        return q[:3].copy(), Rotation.from_quat(q[3:]).as_matrix()

    def add_force(self, body, F, p_w):
        q = self.bq[body]
        com = q[:3] + Rotation.from_quat(q[3:]).apply(self.body_com[body])
        self.body_wrench[body, :3] += F
        self.body_wrench[body, 3:] += np.cross(np.asarray(p_w) - com, F)

    def add_torque(self, body, tau):
        self.body_wrench[body, 3:] += tau

    def place_free(self, body, p, quat_xyzw, keep_children=True, velocity=None):
        """ASSISTED placement (probes/demos only): move a free root body to a world pose, at rest or with a linear
        `velocity`; the links jointed to it keep their joint coordinates; MuJoCo's state follows."""
        m = self.model
        jt = m.joint_type.numpy()
        child = m.joint_child.numpy()
        q0 = m.joint_q_start.numpy()
        d0 = m.joint_qd_start.numpy()
        js = [j for j in range(m.joint_count) if child[j] == body and jt[j] == int(self.newton.JointType.FREE)]
        if not js:
            raise ValueError(f"body {body} has no free joint")
        j = js[0]
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        q[q0[j]:q0[j] + 7] = [*map(float, p), *map(float, quat_xyzw)]
        qd[d0[j]:d0[j] + 6] = 0.
        if velocity is not None:
            qd[d0[j]:d0[j] + 3] = [*map(float, velocity)]
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.sync_solver()
        self.refresh()

    def log_event(self, e):
        self.events.append(e)

    # ------------------------------------------------------------------ stepping
    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            self.jf[:] = 0.
            for t in self.scene.tools.values():
                t.pre_step(self)
            if self.heat is not None:
                self.heat.scrape()
            if self.rig:
                self._servo(c)
            wrench = self.body_wrench
            t1 = time.perf_counter()
            if self.cont is not None and self.cont.n:
                wrench = self._couple(wrench)
            t2 = time.perf_counter()
            if self.fixture is not None:
                wrench = self.fixture(self, wrench)
            self.control.joint_target_q.assign(self.tq.astype(np.float32))
            self.control.joint_target_qd.assign(self.tqd)
            self.control.joint_f.assign(self.jf)
            self.state.clear_forces()
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            # MuJoCo's control callback is process-wide: it is installed for this world's step only (left installed, it
            # is called while another world's model is compiled, and that compile fails)
            global _ACTIVE
            _ACTIVE = self
            self._mujoco.set_mjcb_control(_mj_control)
            try:
                self.solver.step(self.state, self.next_state, self.control, None, c.rigid_dt)
            finally:
                self._mujoco.set_mjcb_control(None)
                _ACTIVE = None
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self.body_wrench = np.zeros_like(self.body_wrench)
            self.refresh()
            for t in self.scene.tools.values():
                t.post_step(self)
            if self.rig:
                self._velocity_reflex()
                self._torque_reflex()
            self._check_faults()
            t3 = time.perf_counter()
            if self.cont is not None and self.cont.n and self.tick % MATERIAL_EVERY == 0:
                self._material_step(MATERIAL_EVERY*c.rigid_dt)
            t4 = time.perf_counter()
            self.timing["step_s"] += t4 - t0
            self.timing["continuum_s"] += t2 - t1
            self.timing["material_s"] += t4 - t3
            self.timing["rigid_s"] += (t1 - t0) + (t3 - t2)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _material_step(self, dt):
        cont, r = self.cont, self.recipe
        if self.heat is not None:
            self.heat.step(dt)
        cont.refresh(dt)
        wet_rate = r.wet_d/cont.spacing**2
        heat_rate = (r.conductivity/(1030.0*3900.0))*dt/cont.spacing**2 if self.heat is not None else 0.0
        if self.heat is not None:
            heat_rate *= self.heat.conduction_gain
        cont.exchange(dt, wet_rate=wet_rate, heat_rate=min(heat_rate, 0.3))
        if self.tick % 100 == 0:
            if not cont.finite():
                raise FloatingPointError(f"Non-finite mixture state at tick {self.tick}")
            hits = int(cont.counts.numpy()[0])
            if hits > 0.001*cont.n*cont.substeps*100:
                raise FloatingPointError(f"Mixture speed guard tripped {hits} times by tick {self.tick}")
            cont.counts.zero_()

    def set_wrench(self, arm, force, moment, ticks):
        """Cartesian wrench at the TCP (world frame), fed forward as joint torques J^T (F, M) for `ticks` ticks, inside
        the joints' torque rating together with the servo and rate-limited (drilling)."""
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        F, Mo = np.asarray(force, float), np.asarray(moment, float)
        if F.shape != (3,) or Mo.shape != (3,) or not (np.isfinite(F).all() and np.isfinite(Mo).all()):
            raise ValueError("force_n and moment_nm must be finite 3-vectors")
        if np.linalg.norm(F) > P.WRENCH_MAX_N or np.linalg.norm(Mo) > P.WRENCH_MAX_NM:
            raise ValueError(f"Wrench beyond the accepted range (|F| <= {P.WRENCH_MAX_N} N, |M| <= {P.WRENCH_MAX_NM} N m)")
        if self.sim_time < self._reflex_until.get(arm, -1.):
            self.refused_targets += 1
            return
        self.wrench_cmd[arm] = (F, Mo, self.tick + int(ticks))

    def _arm_torques(self):
        mujoco, m, d = self._mujoco, self.solver.mj_model, self.solver.mj_data
        rate = P.FRANKA_MAX_TORQUE_RATE*self.config.rigid_dt
        for arm, g in self._arm.items():
            want = np.zeros(len(g["names"]))
            cmd = self.wrench_cmd.get(arm)
            if cmd is not None and self.tick >= cmd[2]:
                del self.wrench_cmd[arm]
                cmd = None
            if cmd is not None:
                F, Mo, _ = cmd
                hand = self._arm_hand[arm]
                tcp = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ self._tcp_offset(arm)
                jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacp, jacr, tcp, hand)
                tau = jacp.T @ F + jacr.T @ Mo
                want = np.clip(tau[[mdof for _, mdof in self._arm_mj_dofs[arm]]], -g["lim"], g["lim"])
            if not (want.any() or g["ff"].any() or g["load"].any() or g["load_want"].any()):
                continue
            g["ff"] = g["ff"] + np.clip(want - g["ff"], -rate, rate)
            g["load"] = g["load"] + np.clip(g["load_want"] - g["load"], -rate, rate)
            m.jnt_actfrcrange[g["jnt"]] = np.stack([-g["lim"] - g["ff"], g["lim"] - g["ff"]], 1)
            self.jf[g["dof"]] += (g["ff"] + g["load"]).astype(np.float32)

    def _servo(self, c):
        self._arm_torques()
        for arm, g in self._arm.items():
            target = self.targets[g["q"]]
            inp = g["inp"]
            if not g["moving"] and np.array_equal(target, inp.target_position):
                continue
            inp.target_position = target.tolist()
            res = g["otg"].update(inp, g["out"])
            if res not in (Result.Working, Result.Finished):
                raise RuntimeError(f"{arm} motion generator: {res}")
            g["out"].pass_to_input(inp)
            self.tq[g["q"]] = inp.current_position
            self.tqd[g["dof"]] = inp.current_velocity
            g["moving"] = res == Result.Working
        q_now = self.jq
        for name in self._finger_speed:
            j = self.joints[name]["q"]
            slew = min(self.joints[name]["velocity_limit"], self._finger_speed[name])*c.rigid_dt
            self.tq[j] += np.clip(self.targets[j] - self.tq[j], -slew, slew)
        for arm, force in self.grip_force.items():
            if not force:
                continue
            for name in self.gripper_joints(arm):
                spec = self.joints[name]
                j, dof = spec["q"], spec["dof"]
                key = (arm, name)
                kp = self._finger_kp[name]
                reach = force/kp
                ref = self._grip_ref.get(key, float(self.tq[j]))
                step = self._finger_speed[name]*c.rigid_dt
                short = float(self.targets[j]) - float(q_now[j])
                goal = float(self.targets[j]) + reach*float(np.clip(short/P.GRIP_FORCE_BAND_M, -1., 1.))
                ref += float(np.clip(goal - ref, -step, step))
                ref = float(np.clip(ref, q_now[j] - reach, q_now[j] + reach))
                self._grip_ref[key] = ref
                push = kp*(ref - float(q_now[j]))
                self.grip_applied[key] = push
                self.jf[dof] += push
                self.tq[j] = q_now[j]
                if abs(push) >= .999*force:
                    self.grasp_latched[key] = True
                else:
                    self.grasp_latched.pop(key, None)

    def _contact_rule(self, d):
        """A correction to MuJoCo's friction rows, made where the pad rule is made (after the rows and their reference
        accelerations exist, before the solve). MuJoCo asks of every contact that its residual acceleration lie in the
        dual of the friction cone: normal residual >= mu x tangential residual. A friction row's reference is to remove
        its sliding speed within the contact's time constant, b v_t, with b = 500 /s here; a body sliding at v_t is
        therefore asked for a normal residual of mu b v_t, which gravity cannot meet above v_t = mu g / b = 6 mm/s: the
        normal force comes out near m (g + mu b v_t)/(1 + mu^2), many times the weight, the body is thrown off the
        surface, the contact (which exists only at negative distance) is gone on the next tick, and the body falls back.
        Measured on the sauce bowl pushed along the counter: no contact on 73 % of the sliding ticks, normal forces to
        14 weights, hops of 0.3 mm at 73 mm/s (E0).

        The rule: a friction row of a contact without a finger pad asks for no more deceleration than Coulomb friction
        can give that contact, mu (g + the normal row's own reference). (Pads keep their own rule, PAD_SLIP_RATE_DT.)
        At rest and in sticking the reference is far below that and nothing changes. A body coasting on a surface is
        then asked for exactly the deceleration friction gives it, and stays in contact; a body being pushed harder
        than friction holds it still leaves the surface on a fifth of the ticks (the dual cone asks a normal residual
        of mu times its tangential acceleration as well), by micrometres and with forces to twice its weight.

        Tried with it and taken out again: contact rows existing from 0.2 mm before the surfaces meet, their reference
        put back by the same distance. It ended the remaining hops (0 % of ticks without contact), but the tools
        standing in the crock crept (the spatula 14 degrees in 3 s) and in one setting a milk carton at rest was thrown
        at 36 m/s."""
        if P.SLIDE_REFERENCE <= 0.:
            return
        n = int(d.ncon)
        if not n:
            return
        adr = np.asarray(d.contact.efc_address[:n])
        sl = (adr >= 0) & (np.asarray(d.contact.dim[:n]) >= 3) & ~self._geom_pad[np.asarray(d.contact.geom[:n])].any(axis=1)
        if not sl.any():
            return
        a = adr[sl]
        mu = np.asarray(d.contact.friction[:n])[sl, 0]
        t1, t2 = d.efc_aref[a + 1], d.efc_aref[a + 2]
        mag = np.hypot(t1, t2)
        cap = P.SLIDE_REFERENCE*mu*(P.GRAVITY + np.maximum(d.efc_aref[a], 0.))
        k = np.minimum(1., cap/np.maximum(mag, 1e-12))
        d.efc_aref[a + 1] = t1*k
        d.efc_aref[a + 2] = t2*k

    def _pad_stick(self, d):
        """Where a finger pad grips, it holds a place (params.PAD_STICK). Called by MuJoCo inside its step, after the
        contacts and their reference accelerations are made and before the solve: every pad contact's two friction
        rows (and its torsional row) get the offset of the touched body from where it was in the finger's frame when
        the two met, as the position error MuJoCo gives no friction row. The anchor is per (finger, touched body); it
        follows when the offset passes the pad's elastic reach, and is dropped when the two part."""
        m = self.solver.mj_model
        n = int(d.ncon)
        seen = set()
        if n:
            geoms = np.asarray(d.contact.geom[:n])
            pad = self._geom_pad[geoms]
            rows = {}
            for i in np.nonzero(pad[:, 0] != pad[:, 1])[0]:
                adr, dim = int(d.contact.efc_address[i]), int(d.contact.dim[i])
                if adr < 0 or dim < 3:
                    continue
                first = bool(pad[i, 0])
                gp, go = (geoms[i, 0], geoms[i, 1]) if first else (geoms[i, 1], geoms[i, 0])
                rows.setdefault((int(m.geom_bodyid[gp]), int(m.geom_bodyid[go])), []).append(
                    (adr, dim, 1. if first else -1., int(gp), int(i)))
            reach, lever = P.PAD_STICK["reach_m"], P.PAD_STICK["lever_m"]
            for key, items in rows.items():
                bf, bo = key
                xf, Rf = d.xpos[bf], d.xmat[bf].reshape(3, 3)
                xo, Ro = d.xpos[bo], d.xmat[bo].reshape(3, 3)
                seen.add(key)
                a = self._stick.get(key) if key in self._stick_seen else None
                if a is None:
                    a = self._stick[key] = [Rf.T @ (xo - xf), Rf.T @ Ro]
                # twist of the touched body in the finger's frame since the anchor; beyond the reach the anchor turns
                tw = Rotation.from_matrix((Rf.T @ Ro) @ a[1].T).as_rotvec()
                ang = float(np.linalg.norm(tw))
                if ang > reach/lever:
                    a[1] = Rotation.from_rotvec(tw*(1. - reach/lever/ang)).as_matrix() @ a[1]
                    tw = tw*(reach/lever/ang)
                tw_w = Rf @ tw
                cs = [np.array(d.contact.pos[i]) for *_, i in items]
                frames = [np.array(d.contact.frame[i]).reshape(3, 3) for *_, i in items]
                drift = [c - (xf + Rf @ (a[1] @ (Ro.T @ (c - xo)) + a[0])) for c in cs]
                tang = [dr - fr[0]*(fr[0] @ dr) for dr, fr in zip(drift, frames)]
                mean = np.mean(tang, axis=0)
                far = float(np.linalg.norm(mean))
                if far > reach:
                    shift = mean*(1. - reach/far)
                    a[0] = a[0] + Rf.T @ shift
                    tang = [t_ - shift for t_ in tang]
                for (adr, dim, sgn, gp, _), fr, t_ in zip(items, frames, tang):
                    k = self._pad_stick_k[gp]
                    d.efc_aref[adr + 1] -= k*sgn*float(fr[1] @ t_)
                    d.efc_aref[adr + 2] -= k*sgn*float(fr[2] @ t_)
                    if dim >= 4:
                        d.efc_aref[adr + 3] -= k*sgn*float(fr[0] @ tw_w)
        for key in [k_ for k_ in self._stick if k_ not in seen]:
            del self._stick[key]
        self._stick_seen = seen

    def _check_faults(self):
        if not (np.isfinite(self.jq).all() and np.isfinite(self.bq).all() and np.isfinite(self.bqd).all()):
            raise FloatingPointError(f"Non-finite state at tick {self.tick}")
        qd = self.bqd
        objs = self._objects
        v = np.abs(qd[objs, :3]).max(initial=0.)
        free = np.setdiff1d(objs, self._spin_exempt)
        om = np.abs(qd[free, 3:]).max(initial=0.)
        if v > P.MAX_OBJECT_SPEED_M_S or om > P.MAX_OBJECT_SPIN_RAD_S:
            k = int(objs[int(np.argmax(np.abs(qd[objs]).max(axis=1)))])
            raise FloatingPointError(f"Unphysical speed at tick {self.tick}: {self.model.body_label[k]} "
                                     f"{np.linalg.norm(qd[k, :3]):.1f} m/s, {np.linalg.norm(qd[k, 3:]):.0f} rad/s")
        if self.solver.mj_data.warning[self._mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise FloatingPointError(f"MuJoCo bad acceleration warning at tick {self.tick}")

    def _new_stop(self, arm):
        last = self._reflex_fired.get(arm)
        self._reflex_fired[arm] = int(self.tick)
        return last is None or not (self.tick - 50 <= last < self.tick)

    def _stop_arm(self, arm):
        g = self._arm[arm]
        q = self.jq[g["q"]].copy()
        self.targets[g["q"]] = q
        self._hold_command(arm, q)
        self.wrench_cmd.pop(arm, None)

    def _velocity_reflex(self):
        qd = self.jqd
        for arm in self.rig.instances:
            hit = None
            for name, spec in self.joints.items():
                if not name.startswith(arm + "/") or "finger" in name:
                    continue
                v = abs(float(qd[spec["dof"]]))
                if v > 1.05*spec["velocity_limit"]:
                    hit = (name, v, spec["velocity_limit"])
                    break
            if hit:
                self._stop_arm(arm)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="velocity", joint=hit[0],
                                              speed_rad_s=hit[1], limit_rad_s=hit[2]))

    def _torque_reflex(self):
        """FR3 collision reflex (libfranka reference thresholds, P.REFLEX_TORQUE_NM): contact forces on the robot mapped
        to its joints (J^T f over contact rows only), less the weight and inertia of the configured load (set_load),
        filtered; above 20 N m while the arm moves, 10 N m otherwise, the arm stops."""
        m, d = self.solver.mj_model, self.solver.mj_data
        mujoco = self._mujoco
        loads = self._payload_torques()
        if not d.nefc:
            self._tau_ext = {}
            return
        f = np.array(d.efc_force[:d.nefc])
        f[np.asarray(d.efc_type[:d.nefc]) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
        if not f.any():
            return
        tau = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, tau, f)
        alpha = min(1., self.config.rigid_dt/P.REFLEX_FILTER_S)
        for arm, dofs in self._arm_mj_dofs.items():
            raw = tau[[dof for _, dof in dofs]] - loads.get(arm, 0.)
            self._tau_ext[arm] = self._tau_ext.get(arm, raw) + alpha*(raw - self._tau_ext.get(arm, raw))
            ext = np.abs(self._tau_ext[arm])
            limit = P.REFLEX_TORQUE_NM[1 if self._arm[arm]["moving"] else 0]
            k = int(np.argmax(ext))
            if ext[k] > limit:
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                self._stop_arm(arm)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(ext[k]), limit_nm=float(limit)))

    def set_load(self, arm, mass_kg, com_m=(0., 0., 0.)):
        """The load the arm's controller is told its hand carries (libfranka's setLoad): its mass and its centre of
        mass in the TCP frame (the rig's `arm/tcp`: the point between the fingertips, the hand's axes). The controller
        compensates that load's weight and leaves it out of its external-torque estimate; 0 kg (the start) means the
        hand carries nothing it knows of. Whatever the hand really holds beyond the configured load, the joints feel:
        it droops the servo by torque/stiffness and counts toward the collision reflex."""
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        com = np.asarray(com_m, float)
        limit = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
        if isinstance(mass_kg, bool) or not isinstance(mass_kg, (int, float)) or not math.isfinite(mass_kg) \
                or not 0. <= mass_kg <= limit:
            raise ValueError(f"Load mass must be in [0, {limit:g}] kg (the arm's payload less its hand)")
        if com.shape != (3,) or not np.isfinite(com).all() or float(np.linalg.norm(com)) > P.LOAD_COM_MAX_M:
            raise ValueError(f"Load centre of mass must be a finite 3-vector within {P.LOAD_COM_MAX_M:g} m of the TCP")
        nb = self._mj_to_newton[self._arm_hand[arm]]
        Rh = Rotation.from_quat(self.bq[nb, 3:])
        q = self.rig.frame_pose(f"{arm}/tcp", self.state)["orientation_wxyz"]
        Rt = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        self.load_cfg[arm] = dict(mass=float(mass_kg), com_tcp=com.copy(),
                                  com_hand=self._tcp_offset(arm) + (Rh.inv()*Rt).apply(com))
        self.load_kg = {a: round(L["mass"], 4) for a, L in self.load_cfg.items() if L["mass"] > 0.}

    def _payload_torques(self):
        """Each arm's configured load (set_load): the joint torques of its weight and inertia, which the collision
        reflex leaves out of the external torque it judges, and the gravity torques the arm feeds forward beside its
        servo from the next tick.

        (A load taken from the simulator's contact list, any free body between both fingers of a hand at its full
        mass, the vessel's contents included, would know the mass of whatever is gripped for free, and it is wrong
        whenever something else carries the thing: holding the handle of a pan that stands on the hob, the arm would
        push up with the pan's whole weight and read 9.1-9.7 N m of the 10 at which an arm holding still is
        stopped.)"""
        m, d, mujoco = self.solver.mj_model, self.solver.mj_data, self._mujoco
        out = {}
        g = np.array(m.opt.gravity)
        want = {arm: np.zeros(len(gr["names"])) for arm, gr in self._arm.items()}
        cvel = {}
        for arm, L in self.load_cfg.items():
            if L["mass"] <= 0.:
                continue
            hand = self._arm_hand[arm]
            c = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ L["com_hand"]
            jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, jacr, c, hand)
            v = jacp @ d.qvel
            prev = (self._prev_cvel or {}).get(arm)
            acc = (v - prev)/self.config.rigid_dt if prev is not None else np.zeros(3)
            cvel[arm] = v
            cols = [dof for _, dof in self._arm_mj_dofs[arm]]
            out[arm] = (jacp.T @ (L["mass"]*(g - np.clip(acc, -50., 50.))))[cols]
            want[arm] = (jacp.T @ (-L["mass"]*g))[cols]
        for arm, gr in self._arm.items():
            gr["load_want"] = want[arm]
        self._prev_cvel = cvel
        return out

    def sync_solver(self):
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    # ------------------------------------------------------------------ queries
    def pair_contact_forces(self, bodies_a, bodies_b):
        """{(a, b): total normal force (N)} over every contact between a body in bodies_a and one in bodies_b."""
        m, d = self.solver.mj_model, self.solver.mj_data
        A, B = set(bodies_a), set(bodies_b)
        f6 = np.zeros(6)
        out = {}
        for i in range(d.ncon):
            g = d.contact.geom[i]
            b0, b1 = (self._mj_to_newton.get(int(m.geom_bodyid[x]), -1) for x in g)
            pair = (b0, b1) if b0 in A and b1 in B else (b1, b0) if b1 in A and b0 in B else None
            if pair:
                self._mujoco.mj_contactForce(m, d, i, f6)
                out[pair] = out.get(pair, 0.) + abs(float(f6[0]))
        return out

    def tcp_pose(self, arm):
        f = self.rig.frame_pose(f"{arm}/tcp", self.state)
        q = f["orientation_wxyz"]
        return np.asarray(f["position_m"], float), np.array([q[1], q[2], q[3], q[0]], float)

    @property
    def sim_time(self):
        return self.tick*self.config.rigid_dt

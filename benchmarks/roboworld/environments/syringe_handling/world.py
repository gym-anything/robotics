"""Rigid world (Newton 1.5 + CPU MuJoCo) with the liquid network of the equipment coupled into every tick.

Robot interface: the one qualified in barcode_scanning, wine_pouring, drilling and ingredient_mixing on the same FR3
arms and Franka Hand, taken here in ingredient_mixing's form: arm targets are followed by a time-synchronised
jerk-limited trajectory within Franka's joint velocity, acceleration and jerk limits; the desired joint torque (servo
plus any fed-forward wrench) is capped at the joint's rating while the arm's own weight and the load its controller
has been told it carries (set_load) are compensated outside it; a joint-velocity reflex; the libfranka collision
reflex; the Franka Hand's grasp (close at 50 mm/s until blocked, then hold the commanded force); hard joint stops;
MuJoCo no-slip.

Step order per tick: the equipment reads the last step's state and sets this tick's forces, friction limits, fits and
armatures (devices.Fluidics.pre_step: Luer joints, needle points and tracks, each chamber's implicit coefficients); the
robot's motion generator and servos advance; MuJoCo steps; the equipment moves the liquid that goes with the speeds
the solver returned (post_step); reflexes and fault checks run.

Engine facts this module carries: an armature written at run
time acts only after mj_setConst, which also returns qpos to the reference pose and recomputes every dof's
invweight0 without its armature; so armature changes are collected per tick and applied once (`_apply_const`), with
the state saved and put back and the kept invweights rewritten.
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization

from . import params as P
from .config import Config
from .scene import Scene

FINGER_LIMIT_MARGIN = 1e-3


_ACTIVE = None                      # the world whose MuJoCo step is running (MuJoCo's control callback is process-wide)


def _mj_control(m, d):
    w = _ACTIVE
    if w is not None:
        w._vial_contacts(d)
        w._track_constraints(d)
        w._device_constraints(d)
        w._contact_rule(d)
        w._pad_normal(d)
        w._glass_normal(d)
        w._pad_stick(d)
        # MuJoCo 3.11 copies position-dependent R/D into island storage BEFORE
        # mjcb_control. Keep those copies synchronized with our constitutive rows;
        # updating only efc_R/D leaves the main solve using the old stiffness.
        if d.nisland:
            ix = np.asarray(d.map_iefc2efc[:int(d.nefc)])
            d.iefc_R[:int(d.nefc)] = d.efc_R[ix]
            d.iefc_D[:int(d.nefc)] = d.efc_D[ix]
        w._mujoco.mj_projectConstraint(m, d)


def _mj_filter(m, d, g1, g2):
    """Only the puncturable stopper / geometrically clear fitting bore are absent.
    Glass walls, neck, bottom, rim encounters and all unrelated contacts remain.
    """
    if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or
            (m.geom_contype[g2] & m.geom_conaffinity[g1])):
        return 1
    w=_ACTIVE
    if w is None:return 0
    for tube, other in ((g1,g2),(g2,g1)):
        n=w._needle_geoms.get(tube)
        if n is not None:
            for v in w.fluid.vials.values():
                if other not in v.outer_geoms:continue
                if n.track is not None and n.track['pen'] is v.site:return 1
                hit=v.site.surface(w,n.tip_world(w)[0])
                if hit is not None and hit[2]:return 1
        s=w._nozzle_geoms.get(tube)
        if s is not None and s.state=='free':
            for f in w.fluid.females.values():
                if f.mate is not None or m.geom_bodyid[other]!=w._newton_to_mj[f.body]:continue
                depth,lat,cos,clear=s.mouth_clearance(w,f)
                if -.00005 <= depth <= P.LUER_CAPTURE['depth'] and clear and cos >= math.cos(math.radians(P.LUER_CAPTURE['angle_deg'])):return 1
    return 0


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


class World:
    def __init__(self, config=None):
        self.config = c = config or Config()
        self.is_g1 = c.robot == "g1"
        self.newton, self.wp = newton, wp = require_engine()
        t0 = time.perf_counter()
        # The scene is built with the processor as Warp's device. Newton's URDF import loads the robot's meshes onto
        # the default device, which is the graphics card when there is one: with the shared card full, every build
        # stopped with "Failed to acquire primary context for device cuda:0" although nothing here runs on it.
        with wp.ScopedDevice("cpu"):
            self.scene = sc = Scene(c, newton, wp)
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
            # "implicit", not "implicitfast": MuJoCo 3.11's implicitfast takes the gyroscopic term's derivative only for a
            # free tree that is one body, and steps any other as Euler does. A needle is three bodies (hub, tube, bead):
            # rolling on the sharps container's floor it spun at 540 rad/s about its axis and, free, its wobble grew by
            # sqrt(1 + (spin dt)^2) = 1.136 a tick (measured 1.1364) until the speed check stopped the world at 16.5 m/s
            # (E14). Under implicit it comes to rest; the arms move within 4.6e-5 rad of the same commands, the benches
            # within 0.1 %, at no cost a tick. implicit damps such a wobble (a free needle at that spin loses most of its
            # transverse spin in 20 ticks) where a real one keeps it: a single-body loose thing has always been stepped so.
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                      enable_multiccd=True, integrator="implicit", cone="elliptic",
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
            # Track rows use a rotating local basis and exact body Jacobians.
            # Keep the solve dense and unpartitioned so there is one authoritative J.
            mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
            mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            mjm.opt.ls_iterations = int(c.solver_ls_iterations)
            mjm.opt.ls_tolerance = float(c.solver_ls_tolerance)
            mjm.opt.tolerance = float(P.SOLVER_TOLERANCE)
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
        self.timing = dict(build_s=0., step_s=0., devices_s=0., rigid_s=0., const_n=0)
        e2n = np.asarray(self.solver.mjc_eq_to_newton_eq.numpy()).reshape(-1)
        self._eq_mj = {int(n): e for e, n in enumerate(e2n[:mjm.neq]) if n >= 0}
        self._shape_geom = {int(sh): g for g, sh in enumerate(self._geom_shape[:mjm.ngeom]) if sh >= 0}
        self._eq_soft, self._eq_rows, self._eq_rows_tick = {}, {}, -1
        self._arm_pending, self._invweight = {}, {}
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
        self._objects = np.array([bb for bb in range(self.model.body_count) if bb not in robots], dtype=int)
        self._spin_exempt = np.array([], dtype=int)
        sb = np.asarray(b.shape_body)
        self.pad_bodies = sorted({int(sb[i]) for i in sc.pad_shapes})
        self.sync_solver()
        self.refresh()
        self.fluid = sc.fluid
        if hasattr(sc, "flap_joint"):
            sc.flap_joint.bind(self)
        self.fluid.bind(self)
        # MuJoCo caches broad-phase masks per body; include the needle-only glass channel.
        for body in range(mjm.nbody):
            geoms=np.flatnonzero(mjm.geom_bodyid==body)
            if len(geoms):
                mjm.body_contype[body]=np.bitwise_or.reduce(mjm.geom_contype[geoms])
                mjm.body_conaffinity[body]=np.bitwise_or.reduce(mjm.geom_conaffinity[geoms])
        self._glass_geoms={self.geom_of_shape(sh) for v in self.fluid.vials.values() for sh in v.inner_shapes}
        self._needle_geoms={n.g_tube:n for n in self.fluid.needles.values()}
        self._nozzle_geoms={g:s for s in self.fluid.syringes.values() for g in s.g_nozzle}
        self._apply_const()
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)      # contacts and rows with the fits as bound
        self.timing["build_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ fits, equalities, armatures
    def geom_of_shape(self, shape):
        return self._shape_geom[int(shape)]

    def eq_row(self, e):
        """The data row of a Newton equality in the compiled model."""
        return self.solver.mj_model.eq_data[self._eq_mj[int(e)]]

    def eq_enable(self, e, on, soft_ticks=0):
        """Switch an equality. Switched on with `soft_ticks`, it starts with a 10 ms time constant and takes its own
        hardness after that many ticks, so that a fit that closes over a millimetre does so without a blow."""
        m, d = self.solver.mj_model, self.solver.mj_data
        k = self._eq_mj[int(e)]
        d.eq_active[k] = 1 if on else 0
        if on and soft_ticks:
            self._eq_soft[k] = [int(soft_ticks), tuple(m.eq_solref[k])]
            m.eq_solref[k] = (0.010, 1.0)
        elif k in self._eq_soft:
            m.eq_solref[k] = self._eq_soft.pop(k)[1]

    def eq_active(self, e):
        return bool(self.solver.mj_data.eq_active[self._eq_mj[int(e)]])

    def eq_force(self, e):
        """Constraint forces of an equality in the last step (N or N m, world axes), or None if it had no rows."""
        d = self.solver.mj_data
        if self._eq_rows_tick != self.tick:
            self._eq_rows, self._eq_rows_tick = {}, self.tick
            n = int(d.nefc)
            if n:
                types = np.asarray(d.efc_type[:n])
                eqs = np.flatnonzero(types == int(self._mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                ids = np.asarray(d.efc_id[:n])[eqs]
                for k in np.unique(ids):
                    self._eq_rows[int(k)] = eqs[ids == k]
        rows = self._eq_rows.get(self._eq_mj[int(e)])
        return None if rows is None else np.array(d.efc_force[rows])

    def limit_force(self, mj_joint):
        """Force of a joint's stop in the last step (N or N m; 0 if the joint was not at a stop)."""
        d = self.solver.mj_data
        n = int(d.nefc)
        if not n:
            return 0.0
        types = np.asarray(d.efc_type[:n])
        rows = np.flatnonzero((types == int(self._mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (np.asarray(d.efc_id[:n]) == int(mj_joint)))
        return float(np.abs(d.efc_force[rows]).sum()) if len(rows) else 0.0

    def set_armature(self, mj_dof, value, invweight):
        """Ask for a dof's armature (applied once per tick by _apply_const) and the invweight0 that goes with it."""
        self._arm_pending[int(mj_dof)] = float(value)
        self._invweight[int(mj_dof)] = float(invweight)

    def keep_invweight(self, mj_dof, invweight):
        self._invweight[int(mj_dof)] = float(invweight)
        self.solver.mj_model.dof_invweight0[int(mj_dof)] = float(invweight)

    def _apply_const(self):
        if not self._arm_pending:
            return
        m, d = self.solver.mj_model, self.solver.mj_data
        for dof, v in self._arm_pending.items():
            m.dof_armature[dof] = v
        self._arm_pending = {}
        keep = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.qacc_warmstart.copy(), d.eq_active.copy(), d.time)
        # mj_setConst also rewrites every CONNECT equality's second anchor from its first at qpos0 (measured: fields
        # 3:6 of eq_data, MuJoCo 3.11; welds and joint equalities are left alone). A needle's track is a CONNECT whose
        # anchor is written at the puncture: with the anchor moved to where the point was at the start, the bead was
        # pulled there in one tick and its hub left at 59 m/s (the draw-dose walkthrough, the tick after an armature
        # rewrite with the needle in a stopper).
        eq_data = m.eq_data.copy()
        self._mujoco.mj_setConst(m, d)
        m.eq_data[:] = eq_data
        d.qpos[:], d.qvel[:], d.act[:], d.qacc_warmstart[:], d.eq_active[:] = keep[:5]
        d.time = keep[5]
        for dof, v in self._invweight.items():
            m.dof_invweight0[dof] = v
        self._mujoco.mj_kinematics(m, d)
        self._mujoco.mj_comPos(m, d)
        self.timing["const_n"] += 1

    # ------------------------------------------------------------------ robot
    def _robot_setup(self, mjm):
        if self.is_g1:
            from .g1 import setup_world
            setup_world(self, mjm)
            return
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
        return dict(g1_velocity=getattr(self,"_g1_v",np.zeros(0)).copy(), arms=out, wrench_cmd=dict(self.wrench_cmd), prev_cvel=dict(self._prev_cvel or {}),
                    stick={k: [v[0].copy(), v[1].copy()] for k, v in self._stick.items()}, stick_seen=set(self._stick_seen),
                    eq_soft={k: [v[0], tuple(v[1])] for k, v in self._eq_soft.items()},
                    load={a: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in L.items()}
                          for a, L in self.load_cfg.items()})

    def set_controller_state(self, st):
        if self.is_g1:self._g1_v=np.asarray(st["g1_velocity"]).copy()
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
        self._eq_soft = {k: [v[0], tuple(v[1])] for k, v in st.get("eq_soft", {}).items()}
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
        if self.is_g1:
            from .g1 import ready
            ready(self)
            return
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

    def point_velocity(self, body, p_w):
        """World velocity of the body's material point at p_w (Newton's body_qd is the centre of mass's velocity and
        the body's angular velocity, both in the world frame)."""
        q = self.bq[body]
        com = q[:3] + Rotation.from_quat(q[3:]).apply(self.body_com[body])
        return self.bqd[body, :3] + np.cross(self.bqd[body, 3:], np.asarray(p_w, float) - com)

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
            self.fluid.pre_step(self)
            self._apply_const()
            t1 = time.perf_counter()
            if self.rig:
                self._servo(c)
            wrench = self.body_wrench
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
            self._mujoco.set_mjcb_contactfilter(_mj_filter)
            try:
                self.solver.step(self.state, self.next_state, self.control, None, c.rigid_dt)
            finally:
                self._mujoco.set_mjcb_control(None)
                self._mujoco.set_mjcb_contactfilter(None)
                _ACTIVE = None
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self.body_wrench = np.zeros_like(self.body_wrench)
            self.refresh()
            t2 = time.perf_counter()
            self.fluid.post_step(self)
            if self._eq_soft:
                m = self.solver.mj_model
                for k in list(self._eq_soft):
                    self._eq_soft[k][0] -= 1
                    if self._eq_soft[k][0] <= 0:
                        m.eq_solref[k] = self._eq_soft.pop(k)[1]
            t3 = time.perf_counter()
            if self.is_g1:
                from .g1 import check
                check(self)
            elif self.rig:
                self._velocity_reflex()
                self._torque_reflex()
            self._check_faults()
            t4 = time.perf_counter()
            self.timing["step_s"] += t4 - t0
            self.timing["devices_s"] += (t1 - t0) + (t3 - t2)
            self.timing["rigid_s"] += (t2 - t1) + (t4 - t3)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

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
        if self.is_g1:
            from .g1 import servo
            servo(self)
            return
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
        14 weights, hops of 0.3 mm at 73 mm/s.

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

    def _track_constraints(self, d):
        """A compliant track in series with the elastic needle, backward Euler.

        Six native weld rows provide storage only: replace their Jacobian with
        relative point/rotation Jacobians in the entry-axis basis. Axial bead
        attachment is stiff; lateral translation and tilt have the authored
        trainer stiffness; spin about the needle is free. J.T supplies paired
        reactions, including moments at the same contact point.
        """
        m, mj = self.solver.mj_model, self._mujoco
        J = d.efc_J.reshape((int(d.nefc), m.nv))
        for n in self.fluid.needles.values():
            if not n.track:
                continue
            tr = n.track; pn = tr["pen"]
            rows = np.flatnonzero((d.efc_type == int(mj.mjtConstraint.mjCNSTR_EQUALITY)) &
                                  (d.efc_id == self._eq_mj[tr["eq"]]))
            if len(rows) != 6:
                continue
            # pre_step may have just placed the bead at a new track entry.
            # The native step has recomputed kinematics; cached Newton body_q
            # still describes the pre-capture pose and would create a false impulse.
            mb = self._newton_to_mj[pn.body] if pn.body >= 0 else 0
            ppos = d.xpos[mb] if mb else np.zeros(3)
            prot = d.xmat[mb].reshape(3, 3) if mb else np.eye(3)
            axis = prot @ tr["axis_local"]
            trial = np.eye(3)[int(np.argmin(np.abs(axis)))]
            u = np.cross(axis, trial); u /= np.linalg.norm(u)
            B = np.array([axis, u, np.cross(axis, u)])
            point = d.xpos[self._newton_to_mj[n.bead]]
            entry = ppos + prot @ tr["entry"]
            jp, jr, pp, pr = [np.zeros((3, m.nv)) for _ in range(4)]
            mj.mj_jac(m, d, jp, jr, point, self._newton_to_mj[n.bead])
            if pn.body >= 0:
                mj.mj_jac(m, d, pp, pr, point, self._newton_to_mj[pn.body])
            J[rows[:3]] = B @ (jp-pp)
            J[rows[3:]] = B @ (jr-pr)
            vel = J[rows] @ d.qvel
            off = B @ (point-entry)
            ax_in = -d.xmat[n.mj_tube].reshape(3, 3)[:, 0]
            tilt = B @ np.cross(axis, ax_in)
            kl, reach = pn.lateral
            lateral = float(np.linalg.norm(off[1:]))
            hard = P.TRACK_LATERAL["hard"] if lateral > reach else 0.
            f = np.r_[-1e6*off[0], -kl*off[1:], -pn.pivot_k(float(d.qpos[n.a_b]))*tilt]
            if hard:
                f[1:3] -= hard*(1-reach/lateral)*off[1:]
            kr = pn.pivot_k(float(d.qpos[n.a_b]))
            angle = math.asin(min(1., float(np.linalg.norm(tilt))))
            extra = .5 if pn.tilt_limit is not None and angle > pn.tilt_limit else 0.
            if extra:
                f[4:] -= extra*(angle-pn.tilt_limit)*tilt[1:]/max(np.linalg.norm(tilt),1e-12)
            K = np.array([1e6, kl+hard, kl+hard, 0., kr+extra, kr+extra])
            C = np.array([10., P.TRACK_LATERAL["damping"], P.TRACK_LATERAL["damping"], 0., .02*kr, .02*kr])
            b = np.maximum(C + self.dt*K, 1e-12)
            R = 1/(self.dt*b)
            d.efc_R[rows], d.efc_D[rows] = R, 1/R
            d.efc_vel[rows] = vel
            d.efc_aref[rows] = R*(f-b*vel)

    def _device_constraints(self, d):
        """Implicit device forces solved together with contact, friction and fittings.

        A row with f = f0 - b*v_new has R=1/(h*b), aref=R*(f0-b*v).
        Needle springs use f0=-K*q and b=C+h*K; hydraulic coefficients already
        eliminate pressure/flow in terms of v_new. These are force constitutive
        rows, not kinematic locks. No added mass or prescribed part motion.
        """
        eq = {self._eq_mj[s.eq_hydraulic]: s for s in self.fluid.syringes.values()}
        elastic = {self._eq_mj[e]: (n, j) for n in self.fluid.needles.values() for j, e in enumerate(n.eq_elastic)}
        for a in range(int(d.nefc)):
            kind, idx = int(d.efc_type[a]), int(d.efc_id[a])
            if kind == int(self._mujoco.mjtConstraint.mjCNSTR_EQUALITY) and idx in eq:
                s = eq[idx]
                b, f0 = getattr(s, "implicit_fluid", (P.SEAL_VISCOUS, 0.0))
            elif kind == int(self._mujoco.mjtConstraint.mjCNSTR_EQUALITY) and idx in elastic:
                n, j = elastic[idx]
                k, b = n.elastic_coeff
                f0 = -k*float(d.qpos[n.elastic_q[j]])
            else:
                continue
            R = 1/(self.dt*max(b, 1e-12))
            d.efc_R[a], d.efc_D[a] = R, 1/R
            d.efc_aref[a] = R*(f0 - b*float(d.efc_vel[a]))

    def _pad_normal(self, d):
        """One implicit normal spring per pad/body pair, distributed over its contacts.

        f = K*penetration - (C+h*K)*v - h*(C+h*K)*a. Thus MuJoCo's
        f=(aref-a)/R uses R=1/[h*(C+h*K)] and the reference below. This
        is backward Euler on the spring and damper, with the actual inverse
        mass left in J M^-1 J'. There is no acceleration/stiffness cap on light
        parts. Friction remains a Coulomb cone with the same physical mu.
        """
        n = int(d.ncon)
        if not n or P.PAD_STIFFNESS <= 0.:
            return
        m = self.solver.mj_model
        geoms = np.asarray(d.contact.geom[:n])
        pad = self._geom_pad[geoms]
        pairs = {}
        for i in np.nonzero(pad[:, 0] != pad[:, 1])[0]:
            adr = int(d.contact.efc_address[i])
            if adr < 0:
                continue
            gp, go = (geoms[i, 0], geoms[i, 1]) if pad[i, 0] else (geoms[i, 1], geoms[i, 0])
            pairs.setdefault((int(m.geom_bodyid[gp]), int(m.geom_bodyid[go])), []).append((adr, int(i)))
        for rows in pairs.values():
            k_point = P.PAD_STIFFNESS/len(rows)
            for adr, i in rows:
                # Critical damping from the contact's effective inverse mass;
                # unlike stiffness, damping should depend on the carried mass.
                c_point = 2*P.PAD_DAMPING_RATIO*math.sqrt(k_point/max(float(d.efc_diagA[adr]), 1e-9))
                b = c_point + self.dt*k_point
                R = 1/(self.dt*b)
                pen = max(-(float(d.efc_pos[adr]) - float(d.efc_margin[adr])), 0.)
                d.efc_R[adr] = R
                d.efc_D[adr] = 1/R
                d.efc_aref[adr] = R*(k_point*pen - b*float(d.efc_vel[adr]))
                if int(d.contact.dim[i]) >= 3:
                    d.contact[i].mu = float(d.contact.friction[i, 0])*math.sqrt(float(d.efc_R[adr + 1])/R)

    def _vial_contacts(self, d):
        """One-sided contacts on the actual cavity, including the swept shaft.

        Convex capsule/box contact can choose the outside face after a thin wall
        is crossed by a prescribed fixture. Analytic cavity normals always point
        inward. MuJoCo solves these as unilateral contacts with paired reactions.
        No pose correction, velocity reset or fluid grant is used.
        """
        mj=self._mujoco;m=self.solver.mj_model;added=False
        for n in self.fluid.needles.values():
            if not n.track or not hasattr(n.track['pen'],'vial'):continue
            v=n.track['pen'].vial;R=d.xmat[v.mj].reshape(3,3);p=d.xpos[v.mj]
            base=R.T@(d.xpos[n.mj_tube]-p)
            tip=base-R.T@d.xmat[n.mj_tube].reshape(3,3)[:,0]*n.n['length']
            radius=n.n['od']/2;points=[tip]
            # Shaft intersections just inside each cavity section, plus its root
            # if that root itself has entered. Endpoints bound radial distance
            # on a straight segment in each cylindrical section.
            if base[2]<v.stopper_in:points.append(base)
            for z in (v.body_h-1e-5,v.body_h+1e-5,v.stopper_in-1e-5):
                if abs(base[2]-tip[2])>1e-10:
                    t=(z-tip[2])/(base[2]-tip[2])
                    if 0<t<1:points.append(tip+t*(base-tip))
            for local in points:
                if local[2]>v.stopper_in:continue
                rho=np.linalg.norm(local[:2]);ri=v.r_in if local[2]<v.body_h else P.VIAL['bore']/2
                candidates=[]
                if rho>1e-9:candidates.append((ri-radius-rho,np.r_[-local[:2]/rho,0.]))
                candidates.append((local[2]-v.z0-radius,np.array([0.,0.,1.])))
                for dist,normal in candidates:
                    if dist>0.00005:continue
                    normal=R@normal;u=np.cross(normal,np.eye(3)[int(np.argmin(abs(normal)))]);u/=np.linalg.norm(u)
                    c=mj.MjContact();c.flex[:]=-1;c.elem[:]=-1;c.vert[:]=-1;c.efc_address=-1
                    c.dist=dist;c.pos=p+R@local;c.frame=np.r_[normal,u,np.cross(normal,u)]
                    c.geom=np.array([self.geom_of_shape(v.inner_shapes[0]),n.g_tube]);c.dim=1
                    c.includemargin=.00005;c.solref=np.array([.002,1.]);c.solimp=np.array([.95,.99,.001,.5,2.])
                    if mj.mj_addContact(m,d,c)!=0:raise RuntimeError('Vial contact buffer overflow')
                    added=True
        if added:
            mj.mj_makeConstraint(m,d);mj.mj_projectConstraint(m,d)
            mj.mj_mulJacVec(m,d,d.efc_vel,d.qvel);mj.mj_referenceConstraint(m,d)

    def _glass_normal(self, d):
        """Implicit unilateral glass contact, 200 N/mm penalty stiffness.

        Native contacts supply geometry, friction, moments and paired Jacobians;
        this avoids mass-scaled soft-contact penetration of the subgram tube.
        """
        for i in range(int(d.ncon)):
            contact=d.contact[i];g1,g2=map(int,contact.geom)
            if not ((g1 in self._glass_geoms and g2 in self._needle_geoms) or
                    (g2 in self._glass_geoms and g1 in self._needle_geoms)):continue
            a=int(contact.efc_address)
            if a<0:continue
            k=2e5;c=2*math.sqrt(k/max(float(d.efc_diagA[a]),1e-9));b=c+self.dt*k;R=1/(self.dt*b)
            penetration=max(float(d.efc_margin[a]-d.efc_pos[a]),0.)
            d.efc_R[a]=R;d.efc_D[a]=1/R;d.efc_aref[a]=R*(k*penetration-b*float(d.efc_vel[a]))
            if contact.dim>=3:contact.mu=float(contact.friction[0])*math.sqrt(float(d.efc_R[a+1])/R)

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

        (The load is not taken from the simulator's contact list, any free body between both fingers of a hand at its
        full mass: that knows the mass of whatever is gripped for free, and it is wrong whenever something else carries
        the thing. Holding the handle of a pan that stands on the hob, the arm would push up with the pan's whole weight
        and read 9.1-9.7 N m of the 10 at which an arm holding still is stopped.)"""
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

"""The simulated world: Newton 1.5 + native CPU MuJoCo for the truck articulation, the robot, loads and structures; the
truck's own systems (controls, drivetrain, steering, hydraulics, tyres) evaluated once per tick. One tick = config.dt.

Order inside a tick (explicit coupling):
  1. read joint and body state;
  2. controls: spring/detent torques on the control joints and the signals the truck decodes from their angles;
  3. drivetrain (engine, converter, clutches, brakes) -> front-wheel joint torques;
  4. steering unit -> steering-wheel reaction torque and left-knuckle cylinder torque (priority flow first);
  5. mast hydraulics -> tilt, lift, free-lift and side-shift cylinder forces (remaining pump flow);
  6. tyres against the drivable surfaces -> wrenches on the wheels and the surface bodies, rolling-resistance torques;
  7. robot motors (robots catalog profile) -> joint targets / torques;
  8. any LABELLED fixture (assisted probes and demos only; never reachable from the policy interface);
  9. MuJoCo steps once; faults (non-finite state, MuJoCo warnings, speeds above params.MAX_BODY_SPEED/SPIN);
 10. per-tick event latches (events.py) and the energy ledger.
"""

import math
import time

import numpy as np

from . import params as P
from .config import Config

# MuJoCo collision groups (contype, conaffinity)
GROUPS = {
    "obj": (1, 1 | 2 | 4 | 8),        # robot, loads, structures
    "truck": (2, 1 | 8),              # truck parts: never each other, never wheels
    "wheel": (4, 1),                  # tyres: drivable surfaces are the tyre model's; still hit pallets, curbs, racks
    "drivable": (8, 1 | 2),           # floors, ramps, trailer floors: native contact with objects and truck bodies (not wheels)
    "deck": (16, 1 | 2 | 8),          # leveler deck and lip: like drivable, and resting on the trailer floor
}


def contact_material(label, material, group):
    """(contact material, ledger mu) of one collision shape from its label, scene material and collision group; mu None
    keeps the shape's own ledger mu (truck MU_TRUCK, structures MU_STRUCTURE, robot MU_ROBOT_*) at priority 0."""
    if label.startswith("sideshift:") and ("blade" in label or "shank" in label):
        return "fork", P.MU_FORK_PALLET                     # steel on wood
    if label.endswith(":coil"):
        return "coil", P.MU_COIL
    if material in ("carton", "wrapped"):
        return material, P.MU_CARTON
    if material == "bale":
        return "bale", P.BALE["mu"]
    if material in ("wood", "dunnage"):
        return material, P.MU_PALLET_FLOOR                  # wood on concrete
    if material in ("rack_beam", "rack_upright"):
        return "rack", P.MU_RACK_STEEL                      # steel on wood
    if material == "steel_roller":
        return "roller", P.MU_CONVEYOR_ROLLER               # steel on wood
    if group in ("drivable", "deck"):
        return "other", P.MU_STRUCTURE
    return "other", None


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


class World:
    def __init__(self, config=None, scene=None):
        self.config = c = config or Config(preset="lab", task="free", robot="none")
        self.newton, self.wp = newton, wp = require_engine()
        import mujoco
        self._mujoco = mujoco
        wp.set_device(c.device)
        from .scene import Scene
        with wp.ScopedDevice(c.device):
            self.scene = sc = (scene or Scene)(c, newton, wp)
            self.model = sc.b.finalize(device=c.device)
            self.rig = sc.bind_robot(self.model)
            self._contact_parameters()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=True, use_mujoco_contacts=True, enable_multiccd=True,
                integrator=str(P.INTEGRATOR), cone="elliptic", solver="newton", iterations=c.solver_iterations,
                ls_iterations=int(P.LS_ITERATIONS), impratio=c.solver_impratio, nconmax=c.contact_capacity, njmax=c.contact_capacity * 3,
                update_data_interval=0)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        self._mj_body_to_newton = np.asarray(self.solver.mjc_body_to_newton.numpy()).reshape(-1)
        self._newton_to_mj_body = {int(nb): mb for mb, nb in enumerate(self._mj_body_to_newton) if nb >= 0}
        self._mujoco_parameters()
        self.T = T = sc.truck
        qs, qds = self.model.joint_q_start.numpy(), self.model.joint_qd_start.numpy()
        self.q_of = {n: int(qs[j]) for n, j in T.joints.items()}
        self.dof_of = {n: int(qds[j]) for n, j in T.joints.items()}
        self.n_dof = int(self.model.joint_dof_count)
        self.body_com_local = self.model.body_com.numpy().astype(float)
        self.body_mass = self.model.body_mass.numpy().astype(float)
        # ---- the truck's systems
        from .controls import Controls
        from .drivetrain import Drivetrain
        from .steering import Steering
        from .hydraulics import Hydraulics
        from .tyres import TyreSet, Wheel
        op_mass = sc.robot_mass if self.rig is not None else 0.0
        self.controls = Controls(T, self.dof_of, operator_mass=op_mass if op_mass > 0 else P.OPERATOR_MASS)
        self.drivetrain = Drivetrain((self.dof_of["wheel_fl"], self.dof_of["wheel_fr"]), c.dt)
        self.steering = Steering(self.dof_of["steering_wheel"], self.dof_of["knuckle_l"], T.ackermann["left_range"], c.dt)
        m = T.masses
        mast_up = m["mast_inner"] + m["carriage"] + m["sideshift"]
        self.hydraulics = Hydraulics(
            dict(tilt=self.dof_of["mast_outer"], main=self.dof_of["mast_int"], free=self.dof_of["carriage"],
                 shift=self.dof_of["sideshift"]),
            dict(tilt=(m["mast_outer"] + m["mast_int"] + mast_up) * P.MAST_TILT_GYRATION ** 2, main=m["mast_int"] + 4 * mast_up,
                 free=m["carriage"] + m["sideshift"], shift=m["sideshift"]), c.dt)
        wheels = []
        for n, r, w, K, sp, drv in (("wheel_fl", P.FRONT_WHEEL_RADIUS, P.FRONT_TYRE_WIDTH, P.TYRE_K, P.SUPREM_FRONT, True),
                                    ("wheel_fr", P.FRONT_WHEEL_RADIUS, P.FRONT_TYRE_WIDTH, P.TYRE_K, P.SUPREM_FRONT, True),
                                    ("wheel_rl", P.REAR_WHEEL_RADIUS, P.REAR_TYRE_WIDTH, P.TYRE_K_REAR, P.SUPREM_REAR, False),
                                    ("wheel_rr", P.REAR_WHEEL_RADIUS, P.REAR_TYRE_WIDTH, P.TYRE_K_REAR, P.SUPREM_REAR, False)):
            b_ = T.bodies[n]
            J = float(self.model.body_inertia.numpy()[b_][1, 1] + self.model.joint_armature.numpy()[self.dof_of[n]])
            wheels.append(Wheel(n, b_, self.dof_of[n], r, w, K, sp, drv, J))
        self.tyres = TyreSet(wheels, sc.surfaces)
        self.chassis = T.bodies["chassis"]
        # facility machines and support forces
        self.systems = []
        if sc.trailers:
            from .dock import TrailerSupports
            self.systems.append(("wrench", TrailerSupports(self)))
        if "conveyor" in sc.machines:
            from .machines import Conveyor
            self.conveyor = Conveyor(self)
            self.systems.append(("joint", self.conveyor))
        # ---- bookkeeping
        self.tick, self.fault = 0, None
        self.solver_stats = {"ticks": 0, "max_gradient": 0.0, "over_1e-3": 0}
        self.fixture = None              # ASSISTED: callable(world) -> dict(wrench=(nb,6) array, joint_f={dof: tau})
        self.fixture_log = []
        self.on_tick = []
        self.joint_f = np.zeros(self.n_dof, np.float32)
        self.wrench = np.zeros((self.model.body_count, 6))
        self.timing = dict(systems_s=0.0, tyres_s=0.0, mujoco_s=0.0, robot_s=0.0)
        self.signals = {}
        from .events import Events
        self.events = Events(self)
        self.initial_pose()

    # ------------------------------------------------------------------ construction
    def _contact_parameters(self):
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
        c = self.config
        self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
        ref = self.model.mujoco.solref.numpy()
        ref[:] = (c.contact_timeconst, 1.0)
        self.model.mujoco.solref.assign(ref)
        imp = self.model.mujoco.geom_solimp.numpy()
        imp[:, :3] = c.contact_solimp
        self.model.mujoco.geom_solimp.assign(imp)

    def _mujoco_parameters(self):
        mujoco, c = self._mujoco, self.config
        m = self.solver.mj_model
        m.opt.noslip_iterations = int(c.solver_noslip_iterations)
        m.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)        # barcode_scanning: midphase dropped pairs
        if not P.SOLVER_ISLANDS:
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
        m.opt.ccd_tolerance = float(P.CCD_TOLERANCE)          # finer than a resting load's penetration (params)
        m.opt.ccd_iterations = int(P.CCD_ITERATIONS)
        groups = self.scene.shape_group
        self.geom_group = {}
        for g in range(m.ngeom):
            s = int(self._geom_shape[g]) if g < len(self._geom_shape) else -1
            grp = groups.get(s, "obj") if s >= 0 else "obj"
            m.geom_contype[g], m.geom_conaffinity[g] = GROUPS[grp]
            self.geom_group[g] = grp
        # contact friction by material: every geom gets its material's ledger mu and priority (params.CONTACT_PRIORITY:
        # MuJoCo takes the higher-priority geom's mu, the larger at equal priority), so every pair the ledger names takes
        # its value
        mats = self.scene.materials
        self.geom_material = {}
        for g in range(m.ngeom):
            sh = int(self._geom_shape[g]) if g < len(self._geom_shape) else -1
            if sh < 0:
                continue
            kind, mu = contact_material(str(self.model.shape_label[sh]), mats.get(sh, ""), self.geom_group.get(g))
            self.geom_material[g] = kind
            if mu is not None:
                m.geom_priority[g] = P.CONTACT_PRIORITY[kind]
                m.geom_friction[g][0] = mu
        # one condim for every surface a resting body can touch: elliptic cones, condim 3 everywhere
        # joint limits hard (barcode_scanning). This sets every hinge and slide joint's limit: the builders pass no
        # limit_ke / limit_kd (Newton's would be replaced here)
        for j in range(m.njnt):
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                m.jnt_solref[j] = (float(P.JOINT_LIMIT_TIMECONST), 1.0)
                m.jnt_solimp[j] = tuple(P.JOINT_LIMIT_SOLIMP)
        self.scene.mujoco_hook(self, m)

    def initial_pose(self):
        q = self.model.joint_q.numpy()
        self.scene.initial_q(self, q)
        self._spawn_on_tyres(q)
        self.model.joint_q.assign(q)
        self.state.joint_q.assign(q)
        self.state.joint_qd.zero_()
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.sync_solver()
        qd = np.zeros(self.n_dof)
        self.steering.reset_to(float(q[self.q_of["steering_wheel"]]), float(q[self.q_of["knuckle_l"]]))
        m = self.T.masses
        carried = sum(v["mass"] for k, v in self.scene.loads.items() if k.startswith("carried"))
        up = m["carriage"] + m["sideshift"] + carried
        self.hydraulics.init_from(self._q_by_dof(q), static=dict(
            free=P.G * up, main=P.G * (m["mast_int"] + 2 * (m["mast_inner"] + up)), tilt=self._tilt_static()))
        if self.rig is not None:
            self.scene.robot_start(self)
        self.scene.start_systems(self)

    def _carried(self):
        return [v["body"] for k, v in self.scene.loads.items() if k.startswith("carried")]

    def _spawn_on_tyres(self, q):
        """Start the truck at rest on its tyres. Built at the asset's wheel centres (tyres 14 / 16 mm into the ground),
        a laden truck would drop another 26 mm at the front onto tyres damped at about 4 % of critical (Traigo's
        C_max), and the pitch rebound can overturn it with the rated load and the mast vertical at 2.0 m (a real truck
        standing loaded is at rest). The static axle loads follow from the composite centre of mass (truck,
        carried load, operator: the H1-2's mass at the seat until it is placed); each tyre's deflection from the Traigo
        law F = K d^1.5; the chassis and the carried loads are pitched about the front axle and lowered so each axle's
        wheel centres sit at that deflection."""
        from .mathx import quat_to_R
        from scipy.spatial.transform import Rotation
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        bq = self.state.body_q.numpy().astype(float)
        com = self._com_world(bq)
        T = self.T.bodies
        bodies = list(T.values()) + self._carried()
        M = float(sum(self.body_mass[b] for b in bodies))
        c = sum(self.body_mass[b] * com[b] for b in bodies)
        if self.rig is not None:
            m_rob = float(sum(self.body_mass[b] for b in self.scene.robot_bodies))
            c = c + m_rob * com[T["seat"]]
            M += m_rob
        c = c / M
        ch = T["chassis"]
        R = quat_to_R(bq[ch:ch + 1, 3:7])[0]
        loc = lambda p: R.T @ (p - bq[ch, :3])
        pf = 0.5 * (bq[T["wheel_fl"], :3] + bq[T["wheel_fr"], :3])
        pr = 0.5 * (bq[T["wheel_rl"], :3] + bq[T["wheel_rr"], :3])
        xf, xr, xc = loc(pf)[0], loc(pr)[0], loc(c)[0]
        Fr = M * P.G * (xf - xc) / (xf - xr)
        Ff = M * P.G - Fr
        df = (max(Ff, 0.0) / 2 / P.TYRE_K) ** (1.0 / P.TYRE_EXP)
        dr = (max(Fr, 0.0) / 2 / P.TYRE_K_REAR) ** (1.0 / P.TYRE_EXP)
        # the asset's wheel centres already sit below their radius over the ground the truck stands on (built
        # deflection 14 / 16 mm): lower each axle by the rest. The ground under each axle: a floor the task gives (a
        # trailer's) is level; otherwise the static ground under each axle, so a truck built level on a grade is pitched
        # onto it (built at the front axle's height on 15 %, its rear wheels started 0.25 m in the air and the load left
        # upward at 8 m/s when they landed)
        if "truck_z" in (self.scene.task_start or {}):
            zf = zr = float(self.scene.truck_ground)
        else:
            zf, zr = self.scene.ground_height(pf[0], pf[1]), self.scene.ground_height(pr[0], pr[1])
        df0 = P.FRONT_WHEEL_RADIUS - (pf[2] - zf)
        dr0 = P.REAR_WHEEL_RADIUS - (pr[2] - zr)
        th = math.asin(((df - df0) - (dr - dr0)) / (xf - xr))
        Rp = Rotation.from_rotvec(th * R[:, 1])
        drop = np.array([0.0, 0.0, -(df - df0)])
        qs = self.model.joint_q_start.numpy()
        labels = list(self.model.joint_label)
        roots = [self.q_of["chassis"]] + [int(qs[labels.index(str(self.model.body_label[b]))]) for b in self._carried()]
        for a in roots:
            q[a:a + 3] = pf + Rp.apply(np.asarray(q[a:a + 3], float) - pf) + drop
            q[a + 3:a + 7] = (Rp * Rotation.from_quat(np.asarray(q[a + 3:a + 7], float))).as_quat()
        self.spawn = dict(mass_kg=M, axle_loads_N=(float(Ff), float(Fr)), tyre_deflection_m=(float(df), float(dr)),
                          built_deflection_m=(float(df0), float(dr0)), ground_z=(float(zf), float(zr)),
                          pitch_deg=math.degrees(th))

    def _tilt_static(self):
        """Generalized force the tilt column holds at the start: gravity's moment about the mast pivot of every body
        above it (mast stages, carriage, side-shifter, forks) and the carried load (the column otherwise starts slack
        and the mast and load drop onto it)."""
        from .mathx import quat_to_R
        bq = self.state.body_q.numpy().astype(float)
        com = self._com_world(bq)
        j = self.T.joints["mast_outer"]
        par = int(self.model.joint_parent.numpy()[j])
        Xp = self.model.joint_X_p.numpy()[j].astype(float)
        Rpar = quat_to_R(bq[par:par + 1, 3:7])[0]
        anchor = bq[par, :3] + Rpar @ Xp[:3]
        axis = Rpar @ quat_to_R(Xp[None, 3:7])[0] @ self.model.joint_axis.numpy()[int(self.model.joint_qd_start.numpy()[j])].astype(float)
        jp, jc = self.model.joint_parent.numpy(), self.model.joint_child.numpy()
        sub, grow = {int(jc[j])}, True
        while grow:
            grow = False
            for k in range(len(jc)):
                if int(jp[k]) in sub and int(jc[k]) not in sub:
                    sub.add(int(jc[k])); grow = True
        tau = 0.0
        for b in list(sub) + self._carried():
            tau += float(np.cross(com[b] - anchor, np.array([0.0, 0.0, -P.G * self.body_mass[b]])) @ axis)
        return -tau

    def _q_by_dof(self, q):
        """Joint coordinates indexed by dof for the truck's 1-dof joints."""
        if not hasattr(self, "_qidx"):
            names = [n for n in self.dof_of if n != "chassis"]
            self._qidx = np.array([self.q_of[n] for n in names]); self._didx = np.array([self.dof_of[n] for n in names])
        out = np.zeros(self.n_dof)
        out[self._didx] = q[self._qidx]
        return out

    # ------------------------------------------------------------------ stepping
    def step(self, n=1):
        for _ in range(n):
            self._step()

    def _step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            q = self.state.joint_q.numpy()
            qd_all = self.state.joint_qd.numpy()
            qv = self._q_by_dof(q)
            jf = self.joint_f
            jf[:] = 0.0
            # 2. controls
            tau, sig = self.controls.torques(qv, qd_all, c.dt)
            for d, t in tau.items():
                jf[d] += t
            seat = sig["seat_switch"] or self.scene.presence_override
            # 3. drivetrain (its pump loaded by last tick's circuit pressures: the priority valve feeds steering first, so
            # the pump works against the higher of the two, plus the open-centre standby pressure)
            self.drivetrain.pump_pressure = max(self.hydraulics.tele.get("p_pump", 0.0),
                                                abs(self.steering.tele.get("pressure_Pa", 0.0))) + P.PUMP_STANDBY_PRESSURE
            dfl, dfr = self.dof_of["wheel_fl"], self.dof_of["wheel_fr"]
            for d, t in self.drivetrain.step(sig, (qv[dfl], qv[dfr]), (qd_all[dfl], qd_all[dfr]), seat).items():
                jf[d] += t
            # 4. steering (priority flow)
            w_e = self.drivetrain.w_e if self.drivetrain.running else 0.0
            pump = P.PUMP_DISPLACEMENT * w_e / (2 * math.pi) * P.PUMP_VOL_EFF
            dsw, dkl = self.dof_of["steering_wheel"], self.dof_of["knuckle_l"]
            st_tau, st_flow = self.steering.step(qv[dsw], qd_all[dsw], qv[dkl], qd_all[dkl],
                                                 min(pump, P.STEER_PRIORITY_FLOW), seat)
            for d, t in st_tau.items():
                jf[d] += t
            # 5. hydraulics
            for d, t in self.hydraulics.step(sig, qv, qd_all, w_e, st_flow, seat).items():
                jf[d] += t
            t1 = time.perf_counter()
            # 6. tyres
            bq = self.state.body_q.numpy().astype(float)
            bqd = self.state.body_qd.numpy().astype(float)
            com_w = self._com_world(bq)
            tw, trr = self.tyres.step(bq, bqd, com_w, self.chassis, c.dt, joint_qd=qd_all)
            for d, t in trr.items():
                jf[d] += t
            wrench = tw
            for kind, sysm in self.systems:
                sysm.step(self, wrench if kind == "wrench" else jf)
            t2 = time.perf_counter()
            # 7. robot motors
            if self.rig is not None:
                self.scene.robot_write(self, jf)
                self.operator.apply_belt(wrench)
            # 8. ASSISTED fixture
            if self.fixture is not None:
                fx = self.fixture(self)
                if fx:
                    if fx.get("wrench") is not None:
                        wrench = wrench + fx["wrench"]
                    for d, t in fx.get("joint_f", {}).items():
                        jf[d] += t
            t3 = time.perf_counter()
            self.signals = sig
            self.wrench = wrench
            self.control.joint_f.assign(jf)
            self.state.clear_forces()
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            self.solver.step(self.state, self.next_state, self.control, None, c.dt)
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            t4 = time.perf_counter()
            self.timing["systems_s"] += t1 - t0
            self.timing["tyres_s"] += t2 - t1
            self.timing["robot_s"] += t3 - t2
            self.timing["mujoco_s"] += t4 - t3
            self._check()
            self.events.update(self)
            for cb in self.on_tick:
                cb(self)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def settle(self, max_s=20.0, window_s=0.5, v_tol=0.004, w_tol=0.02, min_s=1.0, d_tol=0.0005, r_tol=0.0005):
        """Construction step (not part of an episode): run physics until every movable body has stayed in place over
        window_s (as in barcode_scanning): its COM within d_tol and its orientation within r_tol rad of where the window
        began (displacement, not speed: the H1-2's fingers carry a micrometre-scale ~240 Hz micro-vibration that reads
        as up to ~25 mm/s and kept a speed test from ever passing), at most max_s. The robot holds its start command.
        Returns a report; the clock and event latches restart at zero afterwards."""
        dt = self.config.dt
        n_win = int(window_s / dt)
        movable = [i for i in range(self.model.body_count) if not str(self.model.body_label[i]).startswith("static/")]
        quiet = 0
        peak = (0.0, 0.0, None)
        t0 = self.tick
        ref_p = ref_q = None
        while (self.tick - t0) * dt < max_s:
            self.step(20)
            bq = self.state.body_q.numpy()[movable].astype(float)
            com = self._com_world(self.state.body_q.numpy().astype(float))[movable]
            qd = self.state.body_qd.numpy()[movable].astype(float)
            v = np.linalg.norm(qd[:, :3], axis=1); w = np.linalg.norm(qd[:, 3:], axis=1)
            if ref_p is None:
                ref_p, ref_q = com.copy(), bq[:, 3:7].copy()
                quiet = 0
                continue
            dp = np.linalg.norm(com - ref_p, axis=1)
            # rotation since the window began from the relative quaternion's vector part (conj(ref) * q, xyzw): the
            # 2 acos(|q . ref|) form reads up to 0.85 mrad for a still body from float32 rounding alone (above r_tol)
            a, b = ref_q, bq[:, 3:7]
            vr = a[:, 3:4] * b[:, :3] - b[:, 3:4] * a[:, :3] - np.cross(a[:, :3], b[:, :3])
            dq = 2.0 * np.arcsin(np.clip(np.linalg.norm(vr, axis=1), 0.0, 1.0))
            k = int(np.argmax(np.maximum(dp / d_tol, dq / r_tol)))
            peak = (float(v.max()), float(w.max()), str(self.model.body_label[movable[k]]))
            if dp.max() < d_tol and dq.max() < r_tol:
                quiet += 20
            else:
                ref_p, ref_q = com.copy(), bq[:, 3:7].copy()           # moved: the window restarts here
                quiet = 0
            if quiet >= n_win and (self.tick - t0) * dt >= min_s:
                break
        settled = quiet >= n_win
        report = dict(settled=bool(settled), sim_s=(self.tick - t0) * dt, last_v=peak[0], last_w=peak[1], slowest=peak[2])
        self.tick = 0
        self.events.reset()
        self.drivetrain.energy = {k: 0.0 for k in self.drivetrain.energy}
        self.tyres.energy = {k: 0.0 for k in self.tyres.energy}
        return report

    def _com_world(self, bq):
        from .mathx import rotate
        return bq[:, :3] + rotate(bq[:, 3:7], self.body_com_local)

    def _check(self):
        d = self.solver.mj_data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise FloatingPointError(f"Non-finite MuJoCo state at tick {self.tick}")
        warn = [int(d.warning[i].number) for i in range(len(d.warning))]
        if any(warn):
            names = [self._mujoco.mjtWarning(i).name for i, n in enumerate(warn) if n]
            raise FloatingPointError(f"MuJoCo warning at tick {self.tick}: {names}")
        # the constraint solver must have converged: a Newton solve that stalls returns forces that are not physics
        # (E17). The last line-searched iteration of each island carries the solve's final scaled gradient
        g_end = 0.0
        for isl in range(max(1, min(int(d.nisland), 20))):
            for k in range(min(int(d.solver_niter[isl]), 200) - 1, -1, -1):
                rec = d.solver[isl * 200 + k]
                if rec.neval > 0:
                    g_end = max(g_end, float(rec.gradient))
                    break
        st = self.solver_stats
        st["ticks"] += 1
        st["max_gradient"] = max(st["max_gradient"], g_end)
        st["over_1e-3"] += g_end > 1e-3
        if g_end > P.SOLVER_STALL_GRADIENT:
            raise FloatingPointError(f"Constraint solver did not converge at tick {self.tick}: final scaled gradient "
                                     f"{g_end:.3g}")
        if self.rig is not None:                              # a joint far above its motor's speed
            mo = self.operator.motors
            jq = np.abs(self.state.joint_qd.numpy()[mo.dof])
            cap = np.maximum(P.JOINT_SPEED_FAULT * mo.w_nl, P.JOINT_SPEED_FLOOR)
            if (jq > cap).any():
                k = int(np.argmax(jq / cap))
                raise FloatingPointError(f"Unphysical joint speed at tick {self.tick}: {mo.names[k]} {jq[k]:.0f} rad/s "
                                         f"({jq[k] / mo.w_nl[k]:.1f} x its motor's no-load speed)")
        qd = self.state.body_qd.numpy()
        v = np.linalg.norm(qd[:, :3], axis=1); om = np.linalg.norm(qd[:, 3:], axis=1)
        if not hasattr(self, "_v_cap"):                       # robot links and the truck and loads (params)
            robot = set(self.scene.robot_bodies)
            self._v_cap = np.array([P.MAX_BODY_SPEED if i in robot else P.MAX_LOAD_SPEED
                                    for i in range(len(v))], float)
        if (v > self._v_cap).any() or om.max(initial=0.0) > P.MAX_BODY_SPIN:
            k = int(np.argmax(np.maximum(v / self._v_cap, om / P.MAX_BODY_SPIN)))
            raise FloatingPointError(f"Unphysical body speed at tick {self.tick}: {self.model.body_label[k]} "
                                     f"{v[k]:.1f} m/s, {om[k]:.0f} rad/s")
        if not hasattr(self, "_not_robot"):
            self._not_robot = np.array([i not in set(self.scene.robot_bodies) for i in range(len(v))])
        rise = np.where(self._not_robot, qd[:, 2], 0.0)
        if rise.max(initial=0.0) > P.MAX_LOAD_RISE:
            k = int(np.argmax(rise))
            raise FloatingPointError(f"Unphysical upward speed at tick {self.tick}: {self.model.body_label[k]} rising at "
                                     f"{rise[k]:.1f} m/s")
        for name, t in self.tyres.telemetry.items():
            if t.get("ribs") is not None and t["Fz"] > P.TYRE_FAULT_LOAD:
                raise FloatingPointError(f"Tyre {name} load {t['Fz']:.0f} N: a tyre deep inside a surface")

    def sync_solver(self):
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    # ------------------------------------------------------------------ queries
    @property
    def sim_time(self):
        return self.tick * self.config.dt

    def body_pose(self, i):
        from scipy.spatial.transform import Rotation
        i = self.T.bodies[i] if isinstance(i, str) and i in self.T.bodies else (
            self.scene.bodies[i] if isinstance(i, str) else int(i))
        q = self.state.body_q.numpy()[i]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def joint(self, name):
        return float(self.state.joint_q.numpy()[self.q_of[name]]), float(self.state.joint_qd.numpy()[self.dof_of[name]])

    def truck_speed(self):
        """Signed forward speed of the chassis (m/s) and the chassis yaw rate."""
        p, R = self.body_pose("chassis")
        v = self.state.body_qd.numpy()[self.chassis].astype(float)
        return float(v[:3] @ R[:, 0]), float(v[5])

    def mj_geom_label(self, g):
        s = int(self._geom_shape[g]) if g < len(self._geom_shape) else -1
        if s >= 0:
            return str(self.model.shape_label[s])
        m = self.solver.mj_model
        return self._mujoco.mj_id2name(m, self._mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or f"geom{g}"

    def mj_body(self, newton_body):
        return self._newton_to_mj_body[int(newton_body)]

    def systems_checkpoint(self):
        return dict(systems=[sysm.checkpoint() for _, sysm in self.systems],
                    drivetrain=self.drivetrain.checkpoint(), steering=self.steering.checkpoint(),
                    hydraulics=self.hydraulics.checkpoint(), tyres=self.tyres.checkpoint(),
                    controls=self.controls.checkpoint(), events=self.events.checkpoint(), tick=self.tick)

    def systems_restore(self, c):
        for (_, sysm), sc in zip(self.systems, c.get("systems", [])):
            sysm.restore(sc)
        self.drivetrain.restore(c["drivetrain"]); self.steering.restore(c["steering"])
        self.hydraulics.restore(c["hydraulics"]); self.tyres.restore(c["tyres"])
        self.controls.restore(c["controls"]); self.events.restore(c["events"]); self.tick = c["tick"]

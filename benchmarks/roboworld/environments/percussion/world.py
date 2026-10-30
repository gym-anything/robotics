"""Rigid world (Newton 1.5 + MuJoCo, CPU) coupled to the audio-rate instrument models.

Clocks. The world tick is 1 ms (the FR3's FCI rate). Every instrument mode advances 48 audio samples per tick
(48 kHz, exact propagation, modal.py). While any registered striker is within WINDOW_GUARD_M + 1.5 ticks of travel of
a vibrating surface, the tick is an impact window: MuJoCo is stepped 48 times at 20.83 us directly on the solver's
float64 mj_data (update_data_interval 0) with the robot controls held (the FCI zero-order hold), and per sample the
vibro-contact force of every active pair is solved implicitly (contact.contact_force) against the struck structure's
exact modal response, the stick's own bending modes, the residual compliance of truncated membrane modes and the
rigid bodies' response. MuJoCo receives G_k = (F_k + F_{k-1})/2 on both bodies (contact.FreeStriker: positions are then
the trapezoidal ones of a force held over the sample; elastic impacts close their energy balance). Newton's state is
synchronised once at the end of the tick.

Tangential contact: every vibro-contact also carries regularised Coulomb friction on the solved normal force,
mu per material pair (params.FRICTION), opposing the slip velocity at the contact point, applied to both bodies with
its torques; below the slip speed where mu G would exceed the stable viscous bound 0.5 m_t/dt the law is viscous
(sticking). Strikers are the tools' tips, shafts and heads and every robot surface that can meet an instrument from
above (fingertip pads and Dex3 tips, the Franka hand housing, link 7 and finger boxes, the G1 palms and all Dex3 links:
scene._proxies). Rigid impacts on the instruments' non-vibrating parts (hoop, stand, pins, shell, xylophone frame,
resonator boxes) radiate a short transient into the room (_rigid_clicks), so a rim hit is heard.

Robot command path: joint targets pass a libfranka-style rate limiter (velocity, acceleration, jerk:
FR3 10 rad/s^2, 5000 rad/s^3) that generates the reference position and velocity the joint servos track (barcode
profile gains, velocity feed-forward); the Franka Hand's speed-then-force grasp; the collision reflex on the filtered
external torque (MuJoCo contact rows plus vibro-contact forces on the fingers, minus the held payload) and the
velocity reflex. Nothing in the world is actuated except through the robots.
"""

import math
import time
import numpy as np

from . import params as P
from . import contact as CT
from . import modal as MD
from .acoustics import click_signal
from .config import Config
from .scene import Scene

def _cross(a, b):
    """3-vector cross product (np.cross costs ~30 us per call in the per-sample loop)."""
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


DT_A = 1.0/P.AUDIO_FS
NS = P.SAMPLES_PER_TICK
G1_SUBSTEPS = 2                 # g1.RIGID_SUBSTEPS: 0.5 ms MuJoCo steps for the G1 (coarse ticks)
HEAD_BELOW_M = 0.03       # a point this far below the head plane is under the drum, not on its head
LATE_OVERLAP_M = 2e-4     # a contact found deeper than this at its first sample was detected late (see _window_tick)
DEEP_ENTRY_M = 0.003      # first-sample penetration beyond which a 'contact' is an approach from behind
KIN_CACHE = True          # per-sample kinematics cache in impact windows (False only for the equivalence check)


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


class Pair:
    """A striker-surface pair in an impact window."""
    __slots__ = ("si", "fi", "F_prev", "c", "w", "S", "cs", "ws", "law", "c_res", "t", "inv_m", "since", "tick_b",
                 "seg", "local", "struct", "stick", "last_n", "last_p", "u_res", "_mshape", "d_off", "inv_t", "mu",
                 "anchor", "zo")

    def __init__(self, si, fi):
        self.si, self.fi = si, fi
        self.F_prev = 0.0
        self.inv_t = 0.0
        self.mu = 0.0
        self.c = self.w = self.cs = self.ws = None
        self._mshape = None
        self.d_off = 0.0
        self.S = 0.0
        self.c_res = 0.0
        self.t = None
        self.inv_m = 0.0
        self.since = None
        self.tick_b = -1
        self.seg = None
        self.local = None
        self.struct = None
        self.stick = None
        self.last_n = None
        self.last_p = None
        self.anchor = None            # surface-frame point the contact sticks to (static friction), None when free
        self.zo = None


class World:
    def __init__(self, config=None):
        self.config = c = config or Config()
        self.newton, self.wp = newton, wp = require_engine()
        self.scene = Scene(c, newton, wp)
        b = self.scene.b
        with wp.ScopedDevice(c.device):
            self.model = b.finalize(device=c.device)
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (c.contact_timeconst, 1.)
            mass = self.model.body_mass.numpy()
            sb = np.asarray(b.shape_body)
            pads = {int(s) for s in self.scene.pad_shapes}
            imp = self.model.mujoco.geom_solimp.numpy()
            for i in range(len(sb)):
                if i in pads:
                    ref[i] = (P.PAD_SOLREF_TIMECONST*math.sqrt(float(mass[int(sb[i])])/P.PAD_SOLREF_REF_MASS), 1.)
                else:
                    imp[i][:3] = c.contact_solimp
            self.model.mujoco.solref.assign(ref)
            self.model.mujoco.geom_solimp.assign(imp)
            lo = self.model.joint_limit_lower.numpy()
            labels = list(self.model.joint_label)
            qd_start = self.model.joint_qd_start.numpy()
            for j, lab in enumerate(labels):
                if "finger_joint" in lab:
                    lo[qd_start[j]] -= 1e-3
            self.model.joint_limit_lower.assign(lo)
            self.rig = self.scene.rig_pending.bind(self.model)
            import warnings
            with warnings.catch_warnings():
                # mujoco_warp's note that some CCD pair types get one contact: expected here (the vibro-contacts carry
                # the instruments; MuJoCo contacts are rigid backstops) - it would surface as an operator 'Warning' in
                # gar status
                warnings.filterwarnings("ignore", message="MULTICCD is enabled", category=UserWarning)
                self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                          enable_multiccd=True, integrator="implicitfast", cone="elliptic",
                                                          solver="newton", iterations=c.solver_iterations,
                                                          ls_iterations=20, impratio=c.solver_impratio,
                                                          nconmax=c.contact_capacity, njmax=c.contact_capacity*3,
                                                          update_data_interval=0)
            import mujoco
            self._mujoco = mujoco
            self.mjm, self.mjd = self.solver.mj_model, self.solver.mj_data
            mjm = self.mjm
            for j in range(mjm.njnt):
                name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if any(t in name for t in ("fr3_joint", "panda_joint")) and "finger" not in name:
                    mjm.jnt_solref[j] = (2*P.RIGID_DT_S, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 1e-3, mid, power)
                elif "finger_joint" in name:
                    mjm.jnt_solref[j] = (2*P.RIGID_DT_S, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 5e-4, mid, power)
            mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)      # as in barcode_scanning: midphase dropped pairs
            # MuJoCo sleeping islands: instruments and tools at rest (every DOF below sleep_tolerance for mjMINAWAKE
            # steps) are not integrated until a contact, an applied force (the vibro-contact's xfrc) or an awake
            # neighbour wakes their whole contact island; the robots never sleep (they are under control). Without
            # it every 20.8 us window step solved the resting xylophone's 60 contacts (0.83 ms; asleep 0.05 ms)
            mjm.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            robot_trees = {int(mjm.body_treeid[self._newton_to_mj_body(bb)]) for bb in self.scene.robot_bodies
                           if self._newton_to_mj_body(bb) is not None and int(mjm.body_treeid[self._newton_to_mj_body(bb)]) >= 0}
            for t in range(mjm.ntree):
                mjm.tree_sleep_policy[t] = int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER if t in robot_trees
                                               else mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED)
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads:
                    mjm.geom_priority[g] = 1
                    mjm.geom_condim[g] = 4
                    mjm.geom_friction[g][1] = P.PAD_TORSIONAL_M
                elif g < len(self._geom_shape) and int(self._geom_shape[g]) == self.scene.floor_shape:
                    # carpet tiles: rolling resistance (a dropped stick stops rolling) and pile torsion; MuJoCo takes
                    # the larger condim and the element-wise larger friction of a contact's two geoms
                    mjm.geom_condim[g] = 6
                    mjm.geom_friction[g][1] = P.FLOOR_TORSIONAL_M
                    mjm.geom_friction[g][2] = P.FLOOR_ROLLING_M
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        m2n = np.asarray(self.solver.mjc_body_to_newton.numpy())[0]
        self._mj_to_newton = {j: int(n) for j, n in enumerate(m2n)}
        self._newton_to_mj = {n: j for j, n in self._mj_to_newton.items() if n >= 0}
        self._robot_setup()
        self._ready_pose()
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        if self.g1:                       # the robot's microphone is the G1 head array, where the head stands at build
            from scipy.spatial.transform import Rotation
            from . import g1 as G1
            tq = self.state.body_q.numpy()[self.scene.g1_hw["torso"]].astype(float)
            self.scene.mics["station_mic"] = tuple(map(float, tq[:3] + Rotation.from_quat(tq[3:]).apply(G1.HEAD_MIC["xyz"])))
        self.targets = self.control.joint_target_q.numpy().copy()
        self._ref_init()
        robots_ = set(self.scene.robot_bodies)
        self._objects = np.array([bb for bb in range(self.model.body_count) if bb not in robots_], dtype=int)
        self.sync_solver()
        self._reflex_until, self._reflex_fired, self.refused_targets = {}, {}, 0
        self.reflexes = []
        self.tick, self.fault = 0, None
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched, self._grip_ref = {}, {}
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {n: float(ke[self.joints[n]["dof"]]) for arm in self.scene.arm_hw for n in self.gripper_joints(arm)}
        self.timing = dict(rigid_s=0., window_s=0., modal_s=0., acoustic_s=0., broad_s=0., ticks=0, window_ticks=0)
        self.fixture = None          # ASSISTED callback (probes and demos only), never reachable from the policy API
        self._fixture_tick, self._fixture_substep = None, False
        self._ks = None              # per-sample kinematics cache inside an impact window (one MuJoCo state)
        self.phantom_log = []
        self.noslip_fallbacks = []
        mjm = self.mjm
        free = [int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt) if int(mjm.jnt_type[j]) == 0]
        self._free_lin = np.array([a + k for a in free for k in range(3)], int)
        self._free_ang = np.array([a + 3 + k for a in free for k in range(3)], int)
        self._audit_exclusions()
        self._vibro_setup()
        self._acoustics_setup()

    def _mj_step(self):
        """mj_step with a guarded noslip pass. MuJoCo's noslip post-processing (100 iterations, kept from barcode/wine:
        it stops grasped tools creeping in the pads) diverged once in a contact set of a drumstick slipping in the
        pinch while touching link 7 (stick at 28 km/s in one 1 ms step, in a steady-beat run); with noslip off that
        step was fine. So: snapshot qpos/qvel/act/warm-start, step, and if any free body leaves the physical range
        redo the step without noslip and log it (noslip_fallbacks)."""
        mujoco, m, d = self._mujoco, self.mjm, self.mjd
        snap = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.qacc_warmstart.copy(), d.time)
        mujoco.mj_step(m, d)
        v = d.qvel[self._free_lin]
        w = d.qvel[self._free_ang]
        if (len(v) and (np.abs(v).max() > P.MAX_OBJECT_SPEED_M_S or np.abs(w).max() > P.MAX_OBJECT_SPIN_RAD_S)) or \
                not np.isfinite(d.qvel).all():
            d.qpos[:], d.qvel[:], d.act[:], d.qacc_warmstart[:] = snap[:4]
            d.time = snap[4]
            it = m.opt.noslip_iterations
            m.opt.noslip_iterations = 0
            try:
                mujoco.mj_step(m, d)
            finally:
                m.opt.noslip_iterations = it
            self.noslip_fallbacks.append(dict(tick=int(self.tick), dt=float(m.opt.timestep)))

    def wake_all(self):
        """Mark every tree and body awake (after a restore: MuJoCo's sleeping-island links of the snapshot need not
        match the restored contacts, and a contact between two 'sleeping' bodies of different islands is fatal).
        Trees fall asleep again after mjMINAWAKE quiet steps."""
        d = self.mjd
        d.tree_asleep[:] = -(self._mujoco.mjMINAWAKE + 1)
        d.tree_awake[:] = 1
        ba = np.array(d.body_awake)
        ba[ba == 0] = 1
        d.body_awake[:] = ba

    def assisted_place(self, name, position, R):
        """ASSISTED fixture placement (probes and demos only; never reachable from the policy API): put free body `name`
        at `position` (its body origin, world m) with rotation matrix R, at rest, and recompute the kinematics of every
        body (MuJoCo skips sleeping bodies in mj_forward, so the sleep flag is lifted for this one forward pass)."""
        from scipy.spatial.transform import Rotation
        mujoco, m, d = self._mujoco, self.mjm, self.mjd
        mb = self._newton_to_mj[self.scene.bodies[name]]
        j = int(m.body_jntadr[mb])
        qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
        d.qpos[qa:qa + 7] = [*np.asarray(position, float), q[3], q[0], q[1], q[2]]
        d.qvel[va:va + 6] = 0.
        self.wake_all()
        flag = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        m.opt.enableflags &= ~flag
        try:
            mujoco.mj_forward(m, d)
        finally:
            m.opt.enableflags |= flag
        self.solver._update_newton_state(self.model, self.state, d, state_prev=self.state)
        self.phantom_log.append(dict(tick=int(self.tick), placed=name, position=[float(v) for v in position]))

    def _newton_to_mj_body(self, newton_body):
        m2n = np.asarray(self.solver.mjc_body_to_newton.numpy())[0]
        hit = np.flatnonzero(m2n == newton_body)
        return int(hit[0]) if len(hit) else None

    # ================================================================== robot command path
    def _robot_setup(self):
        mjm = self.mjm
        mujoco = self._mujoco
        mj = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self.g1 = bool(getattr(self.scene, "g1", False))
        self._arm_mj_dofs = {}
        self._prev_cvel = None
        self._tau_ext = {}
        self._vibro_tau = {}
        self.rate = {}
        if self.g1:
            # G1: position servos on every joint; targets slewed at the URDF joint speed (no libfranka acceleration or
            # jerk limits, no Franka Hand, no protective reflexes: g1.py)
            self._finger_speed, self._arm_fingers, self._arm_hand = {}, {}, {}
            for name, spec in self.joints.items():
                self.rate[name] = (float(spec["velocity_limit"]), 1e4, 1e8)
            self._g1_ff_setup()
            return
        for name in self.joints:
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mname, dof in mj.items() if mname and mname.startswith(arm + "_") and mname.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._finger_speed = {j: P.FRANKA_HAND_FINGER_SPEED for arm in self.scene.arm_hw for j in self.gripper_joints(arm)}
        mjname = lambda b: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self._arm_fingers = {arm: [b for b in range(mjm.nbody) if mjname(b).startswith(arm + "_") and "finger" in mjname(b)]
                             for arm in self._arm_mj_dofs}
        self._arm_hand = {arm: int(mjm.body_parentid[fs[0]]) for arm, fs in self._arm_fingers.items() if fs}
        # per-joint rate limits of the command path (libfranka): velocity from the controller profile, acceleration
        # and jerk from the robot's interface specification
        for name, spec in self.joints.items():
            if "finger" in name:
                continue
            arm = name.split("/", 1)[0]
            k = int(name.rsplit("joint", 1)[1]) - 1
            model_id = self.scene.arm_hw[arm]["model"]
            if model_id.startswith("panda"):
                a, j = P.PANDA_JOINT_ACC[k], P.PANDA_JOINT_JERK[k]
            else:
                a, j = P.FR3_JOINT_ACC, P.FR3_JOINT_JERK
            self.rate[name] = (0.98*float(spec["velocity_limit"]), float(a), float(j))

    def _g1_ff_setup(self):
        """G1 waist and arm servos: position PD plus the model's gravity torque at the commanded configuration as
        feed-forward (Unitree's teleoperation stack sends pinocchio rnea(q_target, 0, 0) as the motor tau; without it
        the waist sags ~4 deg: barcode_scanning section 6.5). Legs and Dex3 fingers are PD only."""
        self._g1_ff = None
        if not self.config.g1_arm_gravity_ff:
            return
        mujoco, mjm = self._mujoco, self.mjm
        names = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(mjm.njnt)}
        arm = [n for n in self.joints if any(k in n for k in ("_shoulder_", "_elbow_", "_wrist_", "/waist_"))]
        mj_j = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in arm]
        self._g1_ff = dict(names=arm, q=[self.joints[n]["q"] for n in arm], dof=[self.joints[n]["dof"] for n in arm],
                           mj_q=[int(mjm.jnt_qposadr[j]) for j in mj_j], mj_dof=[int(mjm.jnt_dofadr[j]) for j in mj_j],
                           limit=np.array([float(self.joints[n]["effort_limit"]) for n in arm]),
                           data=mujoco.MjData(mjm), tau=np.zeros(len(arm)), every=5)

    def _g1_gravity_feedforward(self, jf, cur):
        ff = self._g1_ff
        if self.tick % ff["every"] == 0:
            m, d, mujoco = self.mjm, ff["data"], self._mujoco
            d.qpos[:] = self.mjd.qpos
            d.qvel[:] = 0.
            d.qpos[ff["mj_q"]] = cur[ff["q"]]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            out = np.zeros(m.nv)
            mujoco.mj_rne(m, d, 0, out)
            ff["tau"] = np.clip(out[ff["mj_dof"]], -ff["limit"], ff["limit"])
        jf[ff["dof"]] += ff["tau"]

    def _ready_pose(self):
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        if self.g1:
            from . import g1 as G1
            for key, v in G1.ARMS_READY.items():
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        for name, hw in self.scene.arm_hw.items():
            for joint, v in zip(hw["arm_joints"] + hw["gripper_joints"], P.READY_POSE):
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        self.model.joint_q.assign(q)
        self.control.joint_target_q.assign(t)

    def _ref_init(self):
        q = self.state.joint_q.numpy()
        self.ref = {n: [float(q[self.joints[n]["q"]]), 0.0, 0.0] for n in self.rate}

    def gripper_joints(self, arm):
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def set_grip_force(self, arm, force):
        if arm not in self.scene.arm_hw:
            raise ValueError(f"No arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]
            q = self.state.joint_q.numpy()
            t = self.control.joint_target_q.numpy()
            for key in [k for k in self._grip_ref if k[0] == arm]:
                t[self.joints[key[1]]["q"]] = q[self.joints[key[1]]["q"]]
                del self._grip_ref[key]
            self.control.joint_target_q.assign(t)

    def set_targets(self, targets):
        """Joint position targets. An arm stopped by a reflex refuses arm targets for REFLEX_RECOVERY_S."""
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

    def _rate_limit(self):
        """One tick of the reference generator: move toward the target as fast as the velocity, acceleration and jerk
        limits allow and stop on it (braking distance v^2/2a + v a/2j)."""
        dt = P.RIGID_DT_S
        tq = self.control.joint_target_q.numpy()
        tqd = np.zeros(self.model.joint_dof_count, np.float32)
        for name, (vmax, amax, jmax) in self.rate.items():
            j = self.joints[name]
            st = self.ref[name]
            q, v, a = st
            e = float(self.targets[j["q"]]) - q
            if abs(e) < 1e-7 and abs(v) < 1e-5:
                st[:] = [q + e, 0.0, 0.0]
            else:
                s = math.copysign(1.0, e)
                ae = abs(e)
                vb = -amax**2/(2*jmax) + math.sqrt(amax**4/(4*jmax**2) + 2*amax*ae)
                v_des = s*min(vmax, 0.97*vb)
                a_des = max(-amax, min(amax, (v_des - v)/dt))
                a_new = a + max(-jmax*dt, min(jmax*dt, a_des - a))
                v_new = max(-vmax, min(vmax, v + a_new*dt))
                q_new = q + 0.5*(v + v_new)*dt
                if (e > 0 and q_new > q + e) or (e < 0 and q_new < q + e):     # never step past the target
                    q_new, v_new, a_new = q + e, 0.0, 0.0
                st[:] = [q_new, v_new, a_new]
            tq[j["q"]] = st[0]
            tqd[j["dof"]] = st[1]
        return tq, tqd

    # ================================================================== vibro-contact registry
    def _audit_exclusions(self):
        """Every striker tool body must be MuJoCo-excluded from every vibrating-surface body (scene.apply_exclusions);
        a missing exclude would let MuJoCo's rigid contact double the vibro-contact."""
        m = self.mjm
        sig = set(int(s) for s in np.asarray(m.exclude_signature[:m.nexclude]))
        missing = []
        for tb, sb in self.scene.exclude_bodies:
            a, b_ = self._newton_to_mj[tb], self._newton_to_mj[sb]
            if ((a << 16) + b_) not in sig and ((b_ << 16) + a) not in sig:
                missing.append((self.model.body_label[tb], self.model.body_label[sb]))
        if missing:
            raise RuntimeError(f"MuJoCo excludes missing for striker/surface bodies: {missing[:5]} ... ({len(missing)})")
        self.exclusion_audit = dict(pairs=len(self.scene.exclude_bodies), nexclude=int(m.nexclude))

    def _vibro_setup(self):
        sc = self.scene
        self.structs = {}
        lay = sc.layout
        if lay["xylophone"]:
            self.structs["xylophone"] = MD.build_xylophone(lay["xylophone"]["notes"], with_resonators=self.config.resonators)
        for rb in lay["resonator_bars"]:
            self.structs[f"resbar_{rb['note']}"] = MD.build_bar_structure(rb["note"], resonator=(True if self.config.resonators else None))
        if lay["drum"]:
            self.structs["drum"] = MD.build_drumhead(lay["drum"]["tension"])
        self.stick_modes = {}
        for t in lay["tools"]:
            if t["kind"] == "stick":
                self.stick_modes[t["name"]] = MD.build_stick_modes(f"{t['name']}_modes")
        m = self.mjm
        self.strikers = []
        for s in sc.strikers:
            mb = self._newton_to_mj[s["body"]]
            free = int(m.body_jntnum[mb]) == 1 and int(m.jnt_type[m.body_jntadr[mb]]) == 0
            d = dict(s, mj=mb, free=free)
            if free:
                j = int(m.body_jntadr[mb])
                d["qadr"], d["vadr"] = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            self.strikers.append(d)
        self.surfaces = []
        for f in sc.surfaces:
            mb = self._newton_to_mj[f["body"]]
            owner = mb
            while int(m.body_jntnum[owner]) == 0 and int(m.body_parentid[owner]) > 0:
                owner = int(m.body_parentid[owner])                       # the welded drumhead -> the drum's free joint
            j = int(m.body_jntadr[owner])
            d = dict(f, mj=mb, owner=owner, qadr=int(m.jnt_qposadr[j]), vadr=int(m.jnt_dofadr[j]))
            st = self.structs[f["struct"]]
            if f["kind"] == "bar":
                d["bar"] = st.bars[f["note"]]["data"]
                d["slice"] = st.slices[f["note"]]
            self.surfaces.append(d)
        # rigid mass properties for the analytic effective masses of free bodies
        self._body_mass = np.array(m.body_mass)
        self._body_ipos = np.array(m.body_ipos)
        self._body_inertia = np.array(m.body_inertia)
        self._body_iquat = np.array(m.body_iquat)
        self._subtree_mass = np.array(m.body_subtreemass)
        self._surf_qadr = np.array([f["qadr"] for f in self.surfaces], int)
        self._surf_owner = np.array([f["owner"] for f in self.surfaces], int)
        self._striker_mj = np.array([s["mj"] for s in self.strikers], int)
        self._surf_rad = np.array([float(np.linalg.norm(f["half"])) if f["kind"] == "bar" else f["radius"] + 0.01
                                   for f in self.surfaces])
        self._surf_box = np.array([f["half"] if f["kind"] == "bar" else (f["radius"], f["radius"], 0.02)
                                   for f in self.surfaces], float)                       # surface-frame half extents
        # striker arrays for the vectorised broadphase: two local points per striker (a sphere repeats its centre)
        self._stk_pts = np.array([[s["p"], s["p"]] if s["shape"] == "sphere" else [s["p0"], s["p1"]]
                                  for s in self.strikers], float).reshape(-1, 2, 3)
        self._stk_r = np.array([s["r"] for s in self.strikers], float)
        self._stk_half = 0.5*np.linalg.norm(self._stk_pts[:, 1] - self._stk_pts[:, 0], axis=1)
        self._stk_free = np.array([bool(s.get("free")) for s in self.strikers])
        self._stk_qadr = np.array([s.get("qadr", 0) for s in self.strikers], int)
        self._stk_vadr = np.array([s.get("vadr", 0) for s in self.strikers], int)
        self._stk_root = np.array([int(m.body_rootid[s["mj"]]) for s in self.strikers], int)
        fric = 1.0 if self.config.vibro_friction else 0.0
        self._mu = [{f["name"]: fric*P.FRICTION[s["material"]][f["material"]] for f in self.surfaces}
                    for s in self.strikers]
        self.pairs = {}
        self.contact_log = []          # closed contact segments (privileged; the evaluator's input)
        self.ring_ref = {}
        self.window_events = dict(late_open=0, max_open_penetration_m=0.0)
        self.mode_snapshots = []       # struck-bar mode energies shortly after each segment
        # rigid instrument parts that click when hit (scene.click_shapes), and the bodies that can hit them
        ng = m.ngeom
        self._click_kind = np.zeros(ng, np.int8)                     # 0 none, 1 metal, 2 wood
        for g in range(min(ng, len(self._geom_shape))):
            kind = self.scene.click_shapes.get(int(self._geom_shape[g]))
            if kind:
                self._click_kind[g] = 1 if kind == "metal" else 2
        hitters = {self._newton_to_mj[b_] for b_ in self.scene.robot_bodies if b_ in self._newton_to_mj}
        hitters |= {self._newton_to_mj[self.scene.items[t["name"]]["body"]] for t in sc.layout["tools"]}
        self._hitter_body = np.zeros(m.nbody, bool)
        self._hitter_body[list(hitters)] = True
        self._click_pairs = set()
        self._cvel_prev = None
        self.rigid_impacts = []        # clicks emitted (privileged log)

    # -------------------------------------------------------------- kinematics of registered bodies
    def _free_pose(self, qadr):
        ks = self._ks
        if ks is not None and ("p", qadr) in ks:
            return ks[("p", qadr)]
        q = self.mjd.qpos
        out = np.array(q[qadr:qadr + 3]), _quat_wxyz_to_R(q[qadr + 3:qadr + 7])
        if ks is not None:
            ks[("p", qadr)] = out
        return out

    def _free_vel(self, vadr, R):
        ks = self._ks
        if ks is not None and ("v", vadr) in ks:
            return ks[("v", vadr)]
        v = self.mjd.qvel
        out = np.array(v[vadr:vadr + 3]), R @ np.array(v[vadr + 3:vadr + 6])
        if ks is not None:
            ks[("v", vadr)] = out
        return out

    def _body_kin(self, which, tick_cache):
        """(origin, R, v_origin, omega_world) of a striker/surface body now. Free bodies: exact from qpos/qvel;
        articulated bodies (fingers): the tick-start kinematics advanced by its velocity - the origin by the
        trapezoid of its velocity, and the orientation by the trapezoid of its angular velocity. With the orientation
        held at the tick's start, a contact point off the body origin drifts from its true place by omega x r per tick
        and jumps back at the next tick's fresh kinematics: a 1 kHz staircase in the penetration of a moving pressed
        fingertip, heard as the 1000 / 3000 Hz lines (a 2 s lift of a pressed G1 hand: 88 % of the sound at
        1000 Hz)."""
        if which.get("free", True) and "qadr" in which:
            p, R = self._free_pose(which["qadr"])
            v, w = self._free_vel(which["vadr"], R)
            return p, R, v, w
        mb = which["mj"]
        d = self.mjd
        if mb not in tick_cache:
            mujoco, m = self._mujoco, self.mjm
            jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, mb)
            tick_cache[mb] = (np.array(d.xpos[mb]), np.array(d.xmat[mb]).reshape(3, 3), jacp, jacr, self._tick_elapsed,
                              jacp @ d.qvel, jacr @ d.qvel)
        ks = self._ks
        if ks is not None and ("b", mb) in ks:
            return ks[("b", mb)]
        p, R, jacp, jacr, t0, v0, w0 = tick_cache[mb]
        v = jacp @ d.qvel                  # the current joint velocities (window samples step MuJoCo every 20.8 us)
        w = jacr @ d.qvel
        tau = self._tick_elapsed - t0
        th = 0.5*(w0 + w)*tau              # rotation vector over the elapsed part of the tick (Rodrigues)
        a = float(math.sqrt(th @ th))
        if a > 1e-12:
            kx, ky, kz = th/a
            Kx = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
            R = (np.eye(3) + math.sin(a)*Kx + (1.0 - math.cos(a))*(Kx @ Kx)) @ R
        out = p + 0.5*(v0 + v)*tau, R, v, w
        if ks is not None:
            ks[("b", mb)] = out
        return out

    def _inv_mass_free(self, mb, R, p_world, n):
        """1/m_eff along n at p of a free body (subtree mass for the drum with its welded head)."""
        m = self._subtree_mass[mb]
        com = self._free_pose_com(mb, R)
        r = p_world - com
        I = self._body_inertia[mb]
        Rq = R @ _quat_wxyz_to_R(self._body_iquat[mb])
        rn = _cross(r, n)
        loc = Rq.T @ rn
        return 1.0/m + float(np.sum(loc**2/np.maximum(I, 1e-12)))

    def _free_pose_com(self, mb, R):
        j = int(self.mjm.body_jntadr[mb])
        qadr = int(self.mjm.jnt_qposadr[j])
        return np.array(self.mjd.qpos[qadr:qadr + 3]) + R @ self._body_ipos[mb]

    def _com_now(self, which):
        """World centre of mass of the body a vibro-contact force is applied to (xfrc_applied acts at the COM)."""
        mb = which["mj"]
        if "owner" in which:                                    # a surface: its own body (bar, or the welded head)
            p, R = self._free_pose(which["qadr"])
            return p + R @ (np.array(self.mjm.body_pos[mb]) + self._body_ipos[mb]) if mb != which["owner"] else \
                p + R @ self._body_ipos[mb]
        if which.get("free"):
            p, R = self._free_pose(which["qadr"])
            return p + R @ self._body_ipos[mb]
        return np.array(self.mjd.xipos[mb])

    def _inv_mass_jac(self, mb, p, n):
        mujoco, m, d = self._mujoco, self.mjm, self.mjd
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, p, mb)
        Jn = n @ jacp
        x = np.zeros(m.nv)
        mujoco.mj_solveM(m, d, x.reshape(1, -1), Jn.reshape(1, -1))
        return float(Jn @ x)

    # -------------------------------------------------------------- geometry of a pair
    def _geometry(self, pr, tick_cache, recompute_t=False):
        """(gap, point, normal (surface->striker), v_rel_n, Rs, ps, local) for the pair now, or None."""
        s, f = self.strikers[pr.si], self.surfaces[pr.fi]
        ps, Rs, vs, ws = self._body_kin(s, tick_cache)
        pf, Rf = self._free_pose(f["qadr"])                     # the welded head shares the drum's frame
        vf, wf = self._free_vel(f["vadr"], Rf)
        if s["shape"] == "sphere":
            c_w = ps + Rs @ s["p"]
        else:
            if pr.t is None or recompute_t:
                pr.t = self._capsule_t(s, f, ps, Rs, pf, Rf)
            c_w = ps + Rs @ (s["p0"] + pr.t*(s["p1"] - s["p0"]))
        c_l = Rf.T @ (c_w - pf)
        if f["kind"] == "bar":
            gap, q_l, n_l = CT.sphere_box(c_l, s["r"], f["half"])
            if s.get("top_only") and n_l[2] < 0.7:
                return None
        else:
            # the whole head out to its rim: the mode shapes vanish there, so the rim region is stiff. (A 6 mm margin
            # leaves an annulus where strikers - excluded from MuJoCo's head collider - fall through the head; moving
            # inward from it, a stick tip met the head 0.5-3 mm deep and one sample of 870 N put 1.4 J into the drum
            # in a steady-beat run.)
            h = CT.sphere_disc(c_l, s["r"], f["radius"])
            if h is None or h[0] < -HEAD_BELOW_M:
                return None                                 # outside the head, or underneath the drum
            gap, q_l, n_l = h
        n = Rf @ n_l
        p = pf + Rf @ q_l
        v_s = vs + _cross(ws, p - ps)
        v_f = vf + _cross(wf, p - pf)
        vr = v_s - v_f
        return gap, p, n, float(vr @ n), Rs, ps, q_l, n_l, Rf, pf, vr

    def _capsule_t(self, s, f, ps, Rs, pf, Rf):
        p0 = Rf.T @ (ps + Rs @ s["p0"] - pf)
        p1 = Rf.T @ (ps + Rs @ s["p1"] - pf)
        if f["kind"] == "bar":
            return CT.segment_box(p0, p1, s["r"], f["half"])[3]
        return 0.0 if p0[2] < p1[2] else 1.0

    # -------------------------------------------------------------- broadphase (per tick)
    def _broadphase(self, cache):
        """Candidate pairs for this tick: per awake striker, its points in every surface frame at once (vectorised) and
        an inflated box test; then the exact gap for the survivors. A pair joins the impact window when its gap minus
        1.5 ticks of approach falls below WINDOW_GUARD_M, or while it is in contact."""
        t0 = time.perf_counter()
        look = 1.5*P.RIGID_DT_S
        qpos = self.mjd.qpos
        pf = qpos[self._surf_qadr[:, None] + np.arange(3)[None, :]]
        qf = qpos[self._surf_qadr[:, None] + np.arange(3, 7)[None, :]]
        w_, x_, y_, z_ = qf[:, 0], qf[:, 1], qf[:, 2], qf[:, 3]
        Rf = np.stack([np.stack([1 - 2*(y_*y_ + z_*z_), 2*(x_*y_ - z_*w_), 2*(x_*z_ + y_*w_)], -1),
                       np.stack([2*(x_*y_ + z_*w_), 1 - 2*(x_*x_ + z_*z_), 2*(y_*z_ - x_*w_)], -1),
                       np.stack([2*(x_*z_ - y_*w_), 2*(y_*z_ + x_*w_), 1 - 2*(x_*x_ + y_*y_)], -1)], 1)   # (nf, 3, 3)
        awake = np.asarray(self.mjd.body_awake)
        f_aw = awake[self._surf_owner] != 0
        d = self.mjd
        # every striker at once: body poses (free tools exact from qpos/qvel; robot links from MuJoCo's last forward
        # pass, as _body_kin), a speed bound, its two points in every surface frame, an inflated box test
        mb = self._striker_mj
        P_ = np.array(d.xpos)[mb]
        R_ = np.array(d.xmat)[mb].reshape(-1, 3, 3)
        cv = np.array(d.cvel)[mb]
        V_ = cv[:, 3:] + np.cross(cv[:, :3], P_ - np.array(d.subtree_com)[self._stk_root])
        W_ = cv[:, :3]
        fr = self._stk_free
        if fr.any():
            qa = self._stk_qadr[fr]
            P_[fr] = qpos[qa[:, None] + np.arange(3)[None, :]]
            q4 = qpos[qa[:, None] + np.arange(3, 7)[None, :]]
            w_, x_, y_, z_ = q4[:, 0], q4[:, 1], q4[:, 2], q4[:, 3]
            R_[fr] = np.stack([np.stack([1 - 2*(y_*y_ + z_*z_), 2*(x_*y_ - z_*w_), 2*(x_*z_ + y_*w_)], -1),
                               np.stack([2*(x_*y_ + z_*w_), 1 - 2*(x_*x_ + z_*z_), 2*(y_*z_ - x_*w_)], -1),
                               np.stack([2*(x_*z_ - y_*w_), 2*(y_*z_ + x_*w_), 1 - 2*(x_*x_ + y_*y_)], -1)], 1)
            va = self._stk_vadr[fr]
            qv = np.asarray(d.qvel)
            V_[fr] = qv[va[:, None] + np.arange(3)[None, :]]
            W_[fr] = np.einsum("nij,nj->ni", R_[fr], qv[va[:, None] + np.arange(3, 6)[None, :]])
        s_aw = awake[mb] != 0
        keys = set()
        if s_aw.any() or f_aw.any():
            pts = P_[:, None, :] + np.einsum("nij,npj->npi", R_, self._stk_pts)                  # (ns, 2, 3)
            spd = np.linalg.norm(V_, axis=1) + np.linalg.norm(W_, axis=1)*0.45
            marg = self._stk_r + P.WINDOW_GUARD_M + spd*look                                   # (ns,)
            # bounding spheres first (most strikers are far from every instrument), then the box test for the rest
            ctr = 0.5*(pts[:, 0] + pts[:, 1])
            dist = np.sqrt(((ctr[None, :, :] - pf[:, None, :])**2).sum(-1))                       # (nf, ns)
            ok = dist < self._surf_rad[:, None] + (self._stk_half + marg)[None, :]
            ok &= s_aw[None, :] | f_aw[:, None]     # both asleep: only a contact, a force or an awake neighbour wakes them
            fi_, si_ = np.nonzero(ok)
            if len(fi_):
                loc = np.einsum("kji,kpj->kpi", Rf[fi_], pts[si_] - pf[fi_][:, None, :])        # (k, 2, 3)
                m_ = marg[si_][:, None]
                box = self._surf_box[fi_]
                hit = np.all((loc.max(1) + m_ >= -box) & (loc.min(1) - m_ <= box), axis=1)
                keys = set(zip(si_[hit].tolist(), fi_[hit].tolist()))
        active = {k for k, pr in self.pairs.items() if pr.F_prev > 0 or pr.since is not None}
        out = []
        for key in sorted(keys | active):
            pr = self.pairs.get(key) or Pair(*key)
            g = self._geometry(pr, cache, recompute_t=True)
            if g is None and pr.F_prev == 0 and pr.since is None:
                continue
            if g is None or g[0] - max(0.0, -g[3])*look < P.WINDOW_GUARD_M or pr.F_prev > 0 or pr.since is not None:
                self.pairs[key] = pr
                out.append(pr)
        keep = {(p.si, p.fi) for p in out}
        for k in list(self.pairs):
            if k not in keep and self.pairs[k].F_prev == 0.0:
                self._close_segment(self.pairs[k])
                del self.pairs[k]
        self.timing["broad_s"] += time.perf_counter() - t0
        return out

    def _prepare_pair(self, pr, g):
        """Shape vectors, compliance and effective mass of a pair for this tick (contact point fixed over the tick)."""
        s, f = self.strikers[pr.si], self.surfaces[pr.fi]
        gap, p, n, vn, Rs, ps, q_l, n_l, Rf, pf = g[:10]
        st = self.structs[f["struct"]]
        K, a, lam, _ = P.CONTACT[s["material"]][f["material"]]
        if f["kind"] == "bar":
            b = np.zeros(st.n)
            b[f["slice"]] = MD.bar_shape(f["bar"], q_l[0], q_l[1], q_l[2], n_l)
            c_res = 0.0
        else:
            # the head's shape vector at the contact point (810 Bessel modes, ~3 ms): reused while the point stays
            # within 0.5 mm of where it was computed (a resting or pressing striker)
            cache_ = pr._mshape
            if cache_ is not None and float(np.hypot(*(np.asarray(q_l[:2]) - cache_[0]))) < 5e-4:
                b, c_res = cache_[1], cache_[2]
            else:
                r = float(math.hypot(q_l[0], q_l[1]))
                th = math.atan2(q_l[1], q_l[0])
                patch = CT.patch_radius(s["material"], "mylar", s["r"], st.membrane["tension"], self.config.patch_scale)
                b = MD.membrane_shape(st, r, th, patch)
                c_res = MD.membrane_residual(st, r, patch, b)
                pr._mshape = (np.array(q_l[:2], float), b, c_res)
        pr.c, pr.w = st.out_row(b), st.in_col(b)
        pr.struct = f["struct"]
        S = st.compliance(pr.c, pr.w, half=False) + c_res
        pr.stick = None
        if s["tool"] in self.stick_modes and s["shape"] == "sphere":
            sm = self.stick_modes[s["tool"]]
            n_loc = Rs.T @ n
            bs = MD.stick_shape(sm, s["stick_x"], n_loc)
            pr.cs, pr.ws = sm.out_row(bs), sm.in_col(bs)
            S += sm.compliance(pr.cs, pr.ws, half=False)
            pr.stick = s["tool"]
        inv_s = self._inv_mass_free(s["mj"], Rs, p, n) if s.get("free") else self._inv_mass_jac(s["mj"], p, n)
        inv_f = self._inv_mass_free(f["owner"], Rf, p, n)
        pr.inv_m = inv_s + inv_f
        # the larger tangential inverse effective mass (two directions in the contact plane): bounds the viscous
        # (stick-regime) friction coefficient so the explicit per-sample friction force stays stable
        t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        t1 /= np.linalg.norm(t1)
        inv_t = 0.0
        for t in (t1, np.cross(n, t1)):
            it = (self._inv_mass_free(s["mj"], Rs, p, t) if s.get("free") else self._inv_mass_jac(s["mj"], p, t)) + \
                self._inv_mass_free(f["owner"], Rf, p, t)
            inv_t = max(inv_t, it)
        pr.inv_t = inv_t
        pr.mu = self._mu[pr.si][f["name"]]
        pr.S = S + DT_A**2*pr.inv_m/2
        pr.law = (K, a, lam)
        pr.c_res = c_res
        pr.tick_b = self.tick
        pr.local = (q_l.copy(), n_l.copy())

    # ================================================================== stepping
    def step(self):
        """One world tick (1 ms)."""
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            tq, tqd = self._rate_limit()
            jf = np.zeros(self.model.joint_dof_count, np.float32)
            q_now = self.state.joint_q.numpy()
            for name in self._finger_speed:
                j = self.joints[name]["q"]
                slew = self._finger_speed[name]*P.RIGID_DT_S
                tq[j] += np.clip(self.targets[j] - tq[j], -slew, slew)
            for arm, force in self.grip_force.items():
                if not force:
                    continue
                for name in self.gripper_joints(arm):
                    spec = self.joints[name]
                    j, dof = spec["q"], spec["dof"]
                    key = (arm, name)
                    kp = self._finger_kp[name]
                    ref = self._grip_ref.get(key, float(tq[j]))
                    stepm = self._finger_speed[name]*P.RIGID_DT_S
                    ref += float(np.clip(self.targets[j] - ref, -stepm, stepm))
                    ref = float(np.clip(ref, q_now[j] - force/kp, q_now[j] + force/kp))
                    self._grip_ref[key] = ref
                    push = kp*(ref - float(q_now[j]))
                    jf[dof] = push
                    tq[j] = q_now[j]
                    if abs(push) >= .999*force:
                        self.grasp_latched[key] = True
                    else:
                        self.grasp_latched.pop(key, None)
            if self.g1 and self._g1_ff is not None:
                self._g1_gravity_feedforward(jf, tq)
            self.control.joint_target_q.assign(tq)
            self.control.joint_target_qd.assign(tqd)
            self.control.joint_f.assign(jf)
            self.state.clear_forces()
            wrench = self.body_wrench.astype(np.float64).copy()
            self._fixture_tick = None
            if self.fixture is not None:
                fixed = self.fixture(self, wrench.copy())
                self._fixture_tick = fixed - wrench         # the fixture's share (an impact window re-evaluates it)
                wrench = fixed
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            cache = {}
            self._tick_elapsed = 0.0
            pairs = self._broadphase(cache)
            t1 = time.perf_counter()
            if pairs:
                self._window_tick(pairs, cache)
                self.timing["window_s"] += time.perf_counter() - t1
                self.timing["window_ticks"] += 1
            else:
                S = self.solver
                S._apply_mjc_control(self.model, self.state, self.control, self.mjd)
                n_sub = G1_SUBSTEPS if self.g1 else 1
                self.mjm.opt.timestep = P.RIGID_DT_S/n_sub
                for _ in range(n_sub):
                    self._mj_step()
                S._update_newton_state(self.model, self.next_state, self.mjd, state_prev=self.state)
                S._step += 1
                self.state, self.next_state = self.next_state, self.state
                self.timing["rigid_s"] += time.perf_counter() - t1
                t2 = time.perf_counter()
                for name, st in list(self.structs.items()) + list(self.stick_modes.items()):
                    if st.silent():
                        continue
                    act, Z = st.free_block()
                    self._radiate(name, st, Z, act)
                self.timing["modal_s"] += time.perf_counter() - t2
            self.tick += 1
            self.timing["ticks"] += 1
            self._rigid_clicks()
            if self.acoustics is not None:
                t3 = time.perf_counter()
                self.acoustics.advance(NS)
                self.timing["acoustic_s"] += time.perf_counter() - t3
            if not self.g1:                           # the G1 low-level joint interface has no protective stops
                self._velocity_reflex()
                self._torque_reflex()
            self._guards()
            if self.tick % 20 == 0:
                self._check_seating()
                self._update_radiation()
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _window_tick(self, pairs, cache):
        """48 MuJoCo steps of 20.83 us with the vibro-contacts solved per sample."""
        mujoco, m, d = self._mujoco, self.mjm, self.mjd
        S = self.solver
        # servo set-points and joint feed-forward change once per world tick; inside an impact window they are ramped
        # linearly over the tick's 48 samples from last tick's values (held as a 1 kHz staircase, a slewing G1 hand
        # pressed on the head drove the contact with a force staircase heard as a 1000 Hz and 3000 Hz tone; a real
        # servo's rotor, gears and current loop filter it)
        ramp = self.tick > 0
        ctrl0, qfrc0 = np.array(d.ctrl), np.array(d.qfrc_applied)
        S._apply_mjc_control(self.model, self.state, self.control, d)
        ctrl1, qfrc1 = np.array(d.ctrl), np.array(d.qfrc_applied)
        ramp = ramp and (np.any(ctrl1 != ctrl0) or np.any(qfrc1 != qfrc0))
        base = np.array(d.xfrc_applied)
        # ASSISTED fixtures (probes and demos) are spring-dampers on a tool: inside an impact window they are re-evaluated
        # every sample from the current MuJoCo state. Held over the tick's 48 samples, the damping term's 1 ms-old
        # velocity feeds energy into a tool pressed on a bar (a 3 N felt press on a silent C4 bar grew to 47-56 dB SPL;
        # with an 8 Hz spring it chattered at 0-55 N)
        fix_nb = []
        if self.fixture is not None and self._fixture_tick is not None:
            fix_nb = [int(i) for i in np.flatnonzero(np.abs(self._fixture_tick).max(axis=1) > 0)]
            base_nofix = base.copy()
            for nb in fix_nb:
                base_nofix[self._newton_to_mj[nb]] -= self._fixture_tick[nb]
        m.opt.timestep = DT_A
        if any(not self.strikers[p.si].get("free") for p in pairs):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cache.clear()                                       # finger kinematics: fresh at the window's start
        for pr in pairs:
            g = self._geometry(pr, cache)
            if g is None:
                pr.c = None
                continue
            self._prepare_pair(pr, g)
            # a pair whose true penetration (rigid gap minus the surface's and the stick's elastic displacement) is
            # already positive when its window opens was detected late (the rigid gap alone is negative whenever the
            # struck head is deflected away from the striker - that is no error)
            st = self.structs[pr.struct]
            d0 = -g[0] + 2*float(np.real(np.dot(pr.c, st.z))) + (2*float(np.real(np.dot(pr.cs, self.stick_modes[pr.stick].z)))
                                                               if pr.stick else 0.0)
            if d0 > 2e-4 and pr.F_prev == 0.0 and pr.since is None:
                self.window_events["late_open"] += 1
                self.window_events["max_open_penetration_m"] = max(self.window_events["max_open_penetration_m"], d0)
                ev = self.window_events.setdefault("late", [])
                if len(ev) < 20:
                    ev.append(dict(tick=int(self.tick), striker=self.strikers[pr.si]["name"],
                                   surface=self.surfaces[pr.fi]["name"], penetration_m=float(d0), vn=float(g[3])))
        live = [p for p in pairs if p.c is not None]
        sim = {p.struct for p in live}
        # cross-compliances between pairs on one structure (contact points fixed over the tick): the end-of-sample
        # displacement at pair i per unit force at pair j, 2 Re(c_i . G w_j)
        by_struct = {}
        for i_, p_ in enumerate(live):
            by_struct.setdefault(p_.struct, []).append(i_)
        xc = {}
        for kk, idx in by_struct.items():
            if len(idx) > 1:
                Gs = self.structs[kk].G
                for i_ in idx:
                    for j_ in idx:
                        if i_ != j_:
                            xc[(i_, j_)] = 2*float(np.real(np.dot(live[i_].c, Gs*live[j_].w)))
        sticks = {p.stick for p in live if p.stick}
        # every pair's displacement row on its structure, stacked: per sample the free displacements of all pairs on a
        # structure are two matrix-vector products with its state (not E z recomputed per pair, 810 modes)
        Cm = {kk: np.array([live[i].c for i in idx]) for kk, idx in by_struct.items()}
        row_of = {i_: r_ for kk, idx in by_struct.items() for r_, i_ in enumerate(idx)}
        Zs = {k: np.empty((self.structs[k].nc, NS), complex) for k in sim}
        Zst = {k: np.empty((self.stick_modes[k].nc, NS), complex) for k in sticks}
        vibro_robot = {}
        rows = set()
        for k in range(NS):
            self._tick_elapsed = k*DT_A
            self._ks = {} if KIN_CACHE else None               # poses/velocities of this sample's MuJoCo state
            if ramp:
                a_ = (k + 1)/NS
                d.ctrl[:] = ctrl0 + a_*(ctrl1 - ctrl0)
                d.qfrc_applied[:] = qfrc0 + a_*(qfrc1 - qfrc0)
            for r in rows:
                d.xfrc_applied[r] = base[r]
            rows = set()
            if fix_nb:
                self._fixture_substep = True
                try:
                    fw = self.fixture(self, np.zeros((self.model.body_count, 6)))
                finally:
                    self._fixture_substep = False
                for nb in fix_nb:
                    mb_ = self._newton_to_mj[nb]
                    d.xfrc_applied[mb_] = base_nofix[mb_] + fw[nb]
                    rows.add(mb_)
            win = {kk: None for kk in sim}
            wst = {kk: None for kk in sticks}
            # phase A: geometry and the free (force-less) penetration at the sample's end, per pair
            Ez = {kk: self.structs[kk].E*self.structs[kk].z for kk in sim}
            U0 = {kk: 2*np.real(Cm[kk] @ self.structs[kk].z) for kk in sim}
            UF = {kk: 2*np.real(Cm[kk] @ Ez[kk]) for kk in sim}
            pre = []
            for i_, pr in enumerate(live):
                g = self._geometry(pr, cache)
                if g is None:
                    pre.append((pr, g, None))
                    continue
                gap, p, n, vn = g[:4]
                u0 = float(U0[pr.struct][row_of[i_]]) - pr.c_res*pr.F_prev
                uf = float(UF[pr.struct][row_of[i_]])
                us0 = usf = 0.0
                if pr.stick:
                    sm = self.stick_modes[pr.stick]
                    us0 = 2*float(np.real(np.dot(pr.cs, sm.z)))
                    usf = 2*float(np.real(np.dot(pr.cs, sm.E*sm.z)))
                d0 = -gap + u0 + us0
                gap_end_free = gap + DT_A*vn + DT_A**2*pr.F_prev*pr.inv_m/2
                d1 = -gap_end_free + uf + usf
                if pr.F_prev == 0.0 and pr.seg is None and pr.d_off == 0.0 and LATE_OVERLAP_M < d0 <= DEEP_ENTRY_M:
                    # a contact found already overlapping (detected late): measure its penetration from the overlap
                    # it was found at, so the late detection cannot inject energy as a one-sample force spike
                    pr.d_off = d0
                    self.window_events["late_overlap"] = self.window_events.get("late_overlap", 0) + 1
                    self.window_events["max_late_overlap_m"] = max(self.window_events.get("max_late_overlap_m", 0.0), d0)
                d1_true = d1
                if pr.d_off:
                    d0, d1 = d0 - pr.d_off, d1 - pr.d_off
                pre.append((pr, g, (d0, d1, p, n, d1_true)))
                pr.last_p, pr.last_n = p, n
            # phase B: forces. Pairs on one structure (two finger pads on a bar, two mallets on the xylophone, a
            # stick's tip and taper on the head) see each other through the structure within the sample: block
            # Gauss-Seidel sweeps add the other pairs' current forces to each pair's free end displacement. Solving
            # them independently (lagging the cross terms a sample) pumped energy into a bar under a resting hand.
            Fs = [0.0]*len(pre)
            cand = [i for i, (pr, g, q) in enumerate(pre) if q is not None and q[1] > -1e-4]
            groups = {}
            for i in cand:
                groups.setdefault(pre[i][0].struct, []).append(i)
            shared = any(len(v) > 1 for v in groups.values())
            for sweep in range(3 if shared else 1):
                F_sweep = list(Fs)
                for i in cand:
                    pr, g, (d0, d1f, p, n, _) = pre[i]
                    K, a, lam = pr.law
                    if d0 > DEEP_ENTRY_M and pr.F_prev == 0.0:
                        # a striker that 'enters' a surface already millimetres deep came from behind or below it
                        # (a stick hanging under the drum: rigid gap -0.16 m gave 66 kN and blew the step up);
                        # the Hertz force there would be kN - no physical approach reaches it. Not a contact.
                        if sweep == 0:
                            self.window_events["deep_entry_skipped"] = self.window_events.get("deep_entry_skipped", 0) + 1
                        continue
                    if shared:
                        for j in groups[pr.struct]:
                            if j != i and Fs[j] > 0:
                                d1f -= Fs[j]*xc[(i, j)]
                    Fs[i], _ = CT.contact_force(d0, d1f, pr.S, K, a, lam, DT_A, Fs[i] if Fs[i] > 0 else pr.F_prev)
                if sweep and all(abs(Fs[i] - F_sweep[i]) <= 1e-10*max(1.0, Fs[i]) for i in cand):
                    break                                       # converged: further sweeps change nothing
            # phase C: apply
            for i, (pr, g, q) in enumerate(pre):
                F = Fs[i]
                if q is None:
                    p, n = pr.last_p, pr.last_n
                else:
                    p, n = q[2], q[3]
                if F > 0 or pr.F_prev > 0:
                    G = 0.5*(F + pr.F_prev)
                    s, f = self.strikers[pr.si], self.surfaces[pr.fi]
                    fc = G*n
                    if pr.mu > 0 and g is not None:
                        # Coulomb friction with static friction: the contact sticks to an anchor on the
                        # surface - a tangential spring-damper (Mindlin's no-slip stiffness k_t = TANGENTIAL_STIFFNESS_
                        # RATIO x the normal contact stiffness at this load, critically damped, both capped for the
                        # explicit per-sample force) - and slips with |Ft| = mu G when the spring would exceed it, the
                        # anchor then following the slip. A viscous sticking branch -c v_t carries a load only while
                        # moving: a G1 hand pressed on the head creeps 1-2 mm/s and is heard
                        vr, q_l, Rf = g[10], g[6], g[8]
                        vt = vr - float(vr @ n)*n
                        if pr.anchor is None:
                            pr.anchor = np.array(q_l, float)
                        u = Rf @ (q_l - pr.anchor)
                        u = u - float(u @ n)*n
                        K_, a_l, _ = pr.law
                        inv_t = max(pr.inv_t, 1e-9)
                        k_n = a_l*K_**(1.0/a_l)*G**((a_l - 1.0)/a_l) if G > 0 else 0.0
                        k_t = min(P.TANGENTIAL_STIFFNESS_RATIO*k_n, P.TANGENTIAL_K_DT2/(DT_A**2*inv_t))
                        c_t = min(0.5/(DT_A*inv_t), 2.0*np.sqrt(k_t/inv_t))
                        ft = -k_t*u - c_t*vt
                        cap = pr.mu*G
                        ftn = float(np.linalg.norm(ft))
                        slip = ftn > cap
                        if slip:
                            ft = ft*(cap/ftn)
                            ftn = cap
                            # the anchor follows: the spring holds the slip force (its share of the capped force)
                            pr.anchor = np.array(q_l, float) + (Rf.T @ (ft/k_t) if k_t > 0 else 0.0)
                        fc = fc + ft
                        if pr.seg is not None:
                            if ftn > pr.seg.get("ft_peak_N", 0.0):
                                pr.seg["ft_peak_N"] = ftn
                            if slip:
                                pr.seg["slip_samples"] = pr.seg.get("slip_samples", 0) + 1
                    for mb, sign, com in ((s["mj"], 1.0, self._com_now(s)), (f["mj"], -1.0, self._com_now(f))):
                        fw = sign*fc
                        d.xfrc_applied[mb, :3] += fw
                        d.xfrc_applied[mb, 3:] += _cross(p - com, fw)
                        rows.add(mb)
                    if not s.get("free"):
                        acc = vibro_robot.setdefault((s["arm"], s["mj"]), [np.zeros(3), np.zeros(3), 0])
                        acc[0] += fc
                        acc[1] += p
                        acc[2] += 1
                if F > 0:
                    wi = pr.w*(-F)
                    win[pr.struct] = wi if win[pr.struct] is None else win[pr.struct] + wi
                    if pr.stick:
                        ws_ = pr.ws*(-F)
                        wst[pr.stick] = ws_ if wst[pr.stick] is None else wst[pr.stick] + ws_
                    self._segment(pr, F, p, n, g)
                    if pr.zo is not None:
                        st_o = self.structs[pr.struct]
                        pr.zo = st_o.E*pr.zo + st_o.G*wi
                elif pr.F_prev > 0:
                    self._end_segment(pr)
                if F == 0.0 and q is not None and q[4] < 0.0:
                    pr.d_off = 0.0                      # truly separated: the next contact is measured normally
                pr.F_prev = F
            for kk in sim:                                      # = Structure.step_radiating, reusing E z
                st_, w_ = self.structs[kk], win[kk]
                st_.z = Ez[kk] if w_ is None else Ez[kk] + st_.G*w_
                Zs[kk][:, k] = st_.z if w_ is None else st_.z + w_/st_.lam
            for kk in sticks:
                Zst[kk][:, k] = self.stick_modes[kk].step(wst[kk])
            self._ks = None
            self._mj_step()
        for r in rows:
            d.xfrc_applied[r] = base[r]
        # every open contact segment keeps the largest vibration energy of its structure seen while it lasts - the
        # structure's energy without the quasi-static deflection of the forces now applied (a press stores bending
        # energy that is no vibration): an evaluation before the contact ends can tell a strike that stays on the
        # surface from a damping touch
        for kk in sim:
            st_ = self.structs[kk]
            wF = None
            for i_ in by_struct.get(kk, ()):
                pr = live[i_]
                if pr.F_prev > 0:
                    wF = pr.w*(-pr.F_prev) if wF is None else wF + pr.w*(-pr.F_prev)
            E_dyn = st_.energy(st_.z if wF is None else st_.z + wF/st_.lam)
            for i_ in by_struct.get(kk, ()):
                seg = live[i_].seg
                if seg is not None:
                    seg["E_max_dyn"] = max(seg.get("E_max_dyn", 0.0), E_dyn)
                    seg["E_dyn_now"] = E_dyn
        S._update_newton_state(self.model, self.next_state, d, state_prev=self.state)
        self.state, self.next_state = self.next_state, self.state
        S._step += 1
        for name, st in self.structs.items():
            if name in sim:
                self._radiate(name, st, Zs[name])
            elif not st.silent():
                act, Z = st.free_block()
                self._radiate(name, st, Z, act)
        for name, st in self.stick_modes.items():
            if name not in sticks and not st.silent():
                st.free_block()
        self._vibro_tau = {}
        if vibro_robot:
            for (arm, mb), (fsum, psum, cnt) in vibro_robot.items():
                fm, pm = fsum/NS, psum/cnt
                jacp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacp, None, pm, mb)
                tau = jacp.T @ fm
                if arm in self._arm_mj_dofs:
                    self._vibro_tau[arm] = self._vibro_tau.get(arm, 0.) + tau[[dof for _, dof in self._arm_mj_dofs[arm]]]

    # -------------------------------------------------------------- rigid-part clicks
    def _rigid_clicks(self):
        """A rigid MuJoCo contact that begins this tick between a tool or robot body and a sounding rigid instrument
        part (scene.click_shapes: hoop, stand, pins - metal; shell, xylophone frame, resonator boxes - wood) radiates a
        short transient from the contact point: energy RIGID_CLICK_ETA x 1/2 m v_n^2 (v_n from the bodies' velocities a
        tick earlier, i.e. before the contact acted; m: the hitter's mass, robot links capped at 1 kg) in authored
        damped partials (params.RIGID_CLICK_SPECTRA). The microphones - the policy's and the client's - hear it."""
        d, m = self.mjd, self.mjm
        cv_prev, self._cvel_prev = self._cvel_prev, np.array(d.cvel)
        if not d.ncon:
            self._click_pairs = set()
            return
        geom = np.asarray(d.contact.geom)[:d.ncon]
        k1, k2 = self._click_kind[geom[:, 0]], self._click_kind[geom[:, 1]]
        bod = np.asarray(m.geom_bodyid)[geom]
        h1, h2 = self._hitter_body[bod[:, 0]], self._hitter_body[bod[:, 1]]
        sel = np.flatnonzero(((k1 > 0) & h2) | ((k2 > 0) & h1))
        now = set()
        for i in sel:
            key = (int(geom[i, 0]), int(geom[i, 1]))
            now.add(key)
            if key in self._click_pairs or cv_prev is None:
                continue
            part, hit = (0, 1) if k1[i] > 0 else (1, 0)
            pos = np.array(d.contact.pos[i])
            nrm = np.array(d.contact.frame[i][:3])
            com = np.array(d.subtree_com)
            vel = []
            for b_ in (int(bod[i, 0]), int(bod[i, 1])):
                c = cv_prev[b_]
                vel.append(c[3:] + np.cross(c[:3], pos - com[int(m.body_rootid[b_])]))
            vn = abs(float((vel[0] - vel[1]) @ nrm))
            if vn < P.RIGID_CLICK_MIN_SPEED_M_S:
                continue
            mass = min(float(m.body_mass[int(bod[i, hit])]), 1.0)
            E = P.RIGID_CLICK_ETA*0.5*mass*vn**2
            kind = "metal" if int(self._click_kind[geom[i, part]]) == 1 else "wood"
            self.rigid_impacts.append(dict(t=round(self.sim_time, 4), part=self._geom_label(int(geom[i, part])),
                                           hitter=self._body_label(int(bod[i, hit])), kind=kind, v_n=round(vn, 3),
                                           E_rad_J=E))
            if self.acoustics is not None and E > 0:
                self.acoustics.add_point_source(pos, click_signal(kind, E), self.acoustics.n)
        self._click_pairs = now

    def _geom_label(self, g):
        s = int(self._geom_shape[g]) if g < len(self._geom_shape) else -1
        return self.model.shape_label[s] if s >= 0 else str(g)

    def _body_label(self, mb):
        nb = self._mj_to_newton.get(mb, -1)
        return self.model.body_label[nb] if nb >= 0 else str(mb)

    # -------------------------------------------------------------- contact log
    def _segment(self, pr, F, p, n, g):
        if pr.seg is None:
            s, f = self.strikers[pr.si], self.surfaces[pr.fi]
            pr.seg = dict(striker=s["name"], tool=s["tool"], surface=f["name"], kind=f["kind"], note=f.get("note"),
                          start=self.tick*NS + int(round(self._tick_elapsed/DT_A)), end=None, peak_N=0.0, impulse_Ns=0.0,
                          v_in=float(g[3]) if g is not None else 0.0, local=[float(v) for v in pr.local[0]],
                          E_before=self.structs[pr.struct].energy())
            st = self.structs[pr.struct]
            if f["kind"] == "bar":
                e = st.dof_energy()
                pr.seg["bars_before_J"] = {n: float(e[st.slices[n]].sum()) for n in st.slices}
                pr.zo = np.zeros(st.nc, complex)
            pr.since = self.tick
        pr.seg["peak_N"] = max(pr.seg["peak_N"], float(F))
        pr.seg["impulse_Ns"] += float(F)*DT_A

    def _end_segment(self, pr):
        if pr.seg is None:
            return
        seg = pr.seg
        seg["end"] = self.tick*NS + int(round(self._tick_elapsed/DT_A))
        seg["E_after"] = self.structs[pr.struct].energy()
        seg["duration_ms"] = (seg["end"] - seg["start"])*DT_A*1e3
        if seg["kind"] == "bar":
            st = self.structs[pr.struct]
            e = st.dof_energy()
            sl = self.surfaces[pr.fi]["slice"]
            kinds = st.bars[seg["note"]]["kinds"]
            eb = e[sl]
            seg["bar_energy_J"] = float(eb.sum())
            seg["mode1_share"] = float(eb[0]/max(eb.sum(), 1e-30))
            seg["all_bars_energy_J"] = {n: float(e[st.slices[n]].sum()) for n in st.slices}
            before = seg.get("bars_before_J", {})
            seg["bars_added_J"] = {n: v - before.get(n, 0.0) for n, v in seg["all_bars_energy_J"].items()}
            self._contact_split(pr, seg)
            if abs(seg["v_in"]) >= P.STRIKE_MIN_SPEED_M_S:
                self.ring_ref[seg["surface"]] = (seg["end"], self._dyn_z(pr.struct, pr))
        self.contact_log.append(seg)
        pr.seg = None
        pr.since = None
        pr.anchor = None
        pr.zo = None

    def _dyn_z(self, struct, ended=None):
        """The structure's state without the quasi-static deflection of the contact forces still applied (a press
        stores bending energy that is no vibration; as E_max_dyn). `ended`: a pair whose contact has just ended."""
        st = self.structs[struct]
        wF = sum((p.w*(-p.F_prev) for p in self.pairs.values()
                  if p is not ended and p.struct == struct and p.F_prev > 0 and p.w is not None), np.zeros(st.nc, complex))
        return st.z + wF/st.lam

    def _contact_split(self, pr, seg):
        """Bar contacts (bookkeeping for the evaluator; the simulation is unchanged). Each pair carries the structure's
        response to its own contact force alone (`zo`: by linearity the structure's state is the free continuation plus
        the sum of these responses), so a stroke is judged on what it did, not on a still-ringing bar's phase or on
        another contact's force meanwhile. Logged: per-bar energies of that response
        (bars_stroke_J); its energy in the struck bar's fundamental, the first vertical bending mode (fundamental_J), and
        the energy an impulse of the same size at the bar's centre gives the fundamental (fundamental_centre_J). The end
        of every strike on a bar (approach >= STRIKE_MIN_SPEED_M_S) also sets ring_ref, its reference for the ring."""
        if pr.zo is None:
            return
        st = self.structs[pr.struct]
        f, sl = self.surfaces[pr.fi], self.surfaces[pr.fi]["slice"]
        own = st.dof_energy(pr.zo)
        seg["bars_stroke_J"] = {n: float(own[s_].sum()) for n, s_ in st.slices.items()}
        seg["fundamental_J"] = float(own[sl][0])
        seg["fundamental_centre_J"] = 0.5*(seg["impulse_Ns"]*float(np.interp(0.0, f["bar"]["x"], f["bar"]["W_vert"][:, 0])))**2

    def open_segments(self):
        """Contacts still in progress, as segments ending now (non-destructive): their E_after is the largest
        vibration energy of the struck structure seen during the contact (E_max_dyn), so an evaluation before the
        contact ends counts a strike that stays on the surface as a strike and a damping touch as no strike."""
        out = []
        now = self.tick*NS
        for pr in self.pairs.values():
            seg = pr.seg
            if seg is None:
                continue
            s = dict(seg)
            # E_after as for a contact that ended now: the vibration energy now (the peak is E_max_dyn, kept)
            s.update(end=now, provisional=True, duration_ms=(now - seg["start"])*DT_A*1e3,
                     E_after=float(seg.get("E_dyn_now", seg.get("E_before", 0.0))))
            if seg["kind"] == "bar" and pr.struct in self.structs:
                st = self.structs[pr.struct]
                e = st.dof_energy()
                eb = e[self.surfaces[pr.fi]["slice"]]
                s["bar_energy_J"] = float(eb.sum())
                s["mode1_share"] = float(eb[0]/max(eb.sum(), 1e-30))
                s["all_bars_energy_J"] = {nm: float(e[st.slices[nm]].sum()) for nm in st.slices}
                before = seg.get("bars_before_J", {})
                s["bars_added_J"] = {nm: v - before.get(nm, 0.0) for nm, v in s["all_bars_energy_J"].items()}
                self._contact_split(pr, s)
            out.append(s)
        return out

    def _close_segment(self, pr):
        if pr.seg is not None:
            self._end_segment(pr)

    # ================================================================== acoustics
    def _acoustics_setup(self):
        self.acoustics = None
        self.radiation = {}
        if not self.config.acoustics:
            return
        from .acoustics import Acoustics
        self.acoustics = Acoustics(dict(self.scene.mics), seed=self.config.seed,
                                   noise=self.config.room_noise)
        self._rad_pose = {}
        self._update_radiation(force=True)

    def _structure_pose_key(self, name):
        sc = self.scene
        if name == "xylophone":
            bodies = [sc.bodies["xylophone_frame"]] + [b["body"] for b in sc.xylo_bars]
        elif name.startswith("resbar_"):
            n = name.split("_", 1)[1]
            bodies = [sc.bodies[f"resbox_{n}"], sc.bodies[f"rbar_{n}"]]
        else:
            bodies = [sc.bodies["drum"]]
        return np.asarray(self.state.body_q.numpy()[bodies], float)

    def _update_radiation(self, force=False):
        if self.acoustics is None:
            return
        from . import acoustics as AC
        for name, st in self.structs.items():
            key = self._structure_pose_key(name)
            old = self._rad_pose.get(name)
            if not force and old is not None:
                dp = np.abs(key[:, :3] - old[:, :3]).max()
                dq = np.abs(np.abs(np.sum(key[:, 3:]*old[:, 3:], axis=1)) - 1).max()
                if dp < 0.005 and dq < 1e-4:
                    continue
            self._rad_pose[name] = key
            st.radiators = self._build_radiation(name, st)

    def _build_radiation(self, name, st):
        from scipy.spatial.transform import Rotation
        from . import acoustics as AC
        A = self.acoustics
        q = self.state.body_q.numpy()
        sc = self.scene
        om = st.lam.imag
        if name == "drum":
            b = sc.bodies["drum"]
            p = q[b, :3].astype(float)
            R = Rotation.from_quat(q[b, 3:]).as_matrix()
            pts, mic, alpha, labels = A.rows_for(p)
            H, dist = AC.drum_transfer(st, R, p, pts, om)
            Hrows = H*(2*st.lam)[None, :]
            return A.radiation(st, Hrows, dist, mic, alpha, labels, om)
        # bars (+ resonators): per DOF transfer at the DOF's own frequency, then onto the complex modes
        if name == "xylophone":
            bars = [(bb["note"], bb["body"]) for bb in sc.xylo_bars]
            ref_body = sc.bodies["xylophone_frame"]
        else:
            n = name.split("_", 1)[1]
            bars = [(n, sc.bodies[f"rbar_{n}"])]
            ref_body = sc.bodies[f"resbox_{n}"]
        pref = q[ref_body, :3].astype(float) + np.array([0, 0, 0.1])
        pts, mic, alpha, labels = A.rows_for(pref, source_above_table=True)
        dist = np.linalg.norm(pts - pref[None, :], axis=1)
        Hdof = np.zeros((len(pts), st.n), complex)
        for note, body in bars:
            bd = st.bars[note]["data"]
            kinds = st.bars[note]["kinds"]
            sl = st.slices[note]
            p = q[body, :3].astype(float)
            R = Rotation.from_quat(q[body, 3:]).as_matrix()
            omega = st.omega[sl]
            ov, oh = omega[kinds == "v"], omega[kinds == "h"]
            Hb, _ = AC.bar_transfer(bd, R, p, pts, ov, oh)
            # bar_transfer's phase is relative to the bar's own distance; re-reference to pref's row distances
            db = np.linalg.norm(pts - p[None, :], axis=1)
            idx = np.arange(sl.start, sl.stop)
            iv, ih = idx[kinds == "v"], idx[kinds == "h"]
            kk = np.r_[ov, oh]/P.C_AIR
            Hb = Hb*np.exp(-1j*np.outer(db - dist, kk))
            Hdof[:, np.r_[iv, ih]] = Hb
        frame_R = Rotation.from_quat(q[ref_body, 3:]).as_matrix()
        for h in st.resonators:
            if name == "xylophone":
                r_i = st.resonators.index(h)
                po = q[ref_body, :3].astype(float) + frame_R @ sc.resonator_openings[r_i]
            else:
                po = q[ref_body, :3].astype(float) + frame_R @ np.array([0., 0., h["box"]["height"]])
            gain = -h["S"]/math.sqrt(h["m_a"])                  # volume velocity out of the mouth per unit DOF velocity
            Rx = np.linalg.norm(pts - po[None, :], axis=1)
            w0 = st.omega[h["dof"]]
            Hdof[:, h["dof"]] = (1j*w0*P.RHO_AIR*gain)/(4*math.pi*Rx)*np.exp(-1j*w0/P.C_AIR*(Rx - dist))
        Phi_v = (st.Phi_q*st.lam[None, :]) if st.coupled else None
        Hrows = 2*(Hdof @ Phi_v) if st.coupled else 2*Hdof*st.lam[None, :]
        return A.radiation(st, Hrows, dist, mic, alpha, labels, om)

    def _radiate(self, name, st, Z, idx=None):
        """Add a structure's sound for the samples of the current tick (Z: states at each sample's end; idx: their
        mode indices when only the active modes were propagated)."""
        if self.acoustics is None or getattr(st, "radiators", None) is None:
            return
        t0 = time.perf_counter()
        self.acoustics.add_block(st.radiators, Z, self.tick*NS, idx)
        self.timing["acoustic_s"] += time.perf_counter() - t0

    # ================================================================== bars on/off the instrument
    def _check_seating(self):
        """A xylophone bar lifted off its pins no longer pumps its resonator: the coupled structure is rebuilt without
        it (the vibration of every bar is carried over exactly)."""
        if "xylophone" not in self.structs:
            return
        sc = self.scene
        q = self.state.body_q.numpy()
        fb = sc.bodies["xylophone_frame"]
        from scipy.spatial.transform import Rotation
        Rf = Rotation.from_quat(q[fb, 3:]).as_matrix()
        pf = q[fb, :3]
        top = sc.items["xylophone_frame"]["top"]
        off = []
        for bb in sc.xylo_bars:
            pl = Rf.T @ (q[bb["body"], :3] - pf)
            seated = abs(pl[0] - bb["frame_x"]) < 0.01 and abs(pl[1]) < 0.012 and \
                abs(pl[2] - (top + P.FELT_THICK_M + P.BAR_THICK_M/2)) < 0.006
            if not seated:
                off.append(bb["note"])
        st = self.structs["xylophone"]
        if tuple(off) != tuple(getattr(st, "removed", ())):
            new = MD.build_xylophone(self.scene.layout["xylophone"]["notes"], with_resonators=self.config.resonators,
                                     removed=tuple(off))
            new.transfer_from(st)
            new.removed = tuple(off)
            new.radiators = st.radiators
            self.structs["xylophone"] = new
            for pr in self.pairs.values():
                if pr.struct == "xylophone":
                    pr.tick_b = -1
            self._rad_pose.pop("xylophone", None) if hasattr(self, "_rad_pose") else None

    # ================================================================== reflexes and guards (wine_pouring precedent)
    def _new_stop(self, arm):
        last = self._reflex_fired.get(arm)
        self._reflex_fired[arm] = int(self.tick)
        return last is None or not (self.tick - 50 <= last < self.tick)

    def _stop_arm(self, arm):
        q = self.state.joint_q.numpy()
        for name, spec in self.joints.items():
            if name.startswith(arm + "/") and "finger" not in name:
                self.targets[spec["q"]] = float(q[spec["q"]])
                self.ref[name] = [float(q[spec["q"]]), 0.0, 0.0]

    def _velocity_reflex(self):
        qd = self.state.joint_qd.numpy()
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
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="velocity", joint=hit[0],
                                              speed_rad_s=hit[1], limit_rad_s=hit[2]))

    def _torque_reflex(self):
        m, d = self.mjm, self.mjd
        mujoco = self._mujoco
        tau = np.zeros(m.nv)
        if d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            if f.any():
                mujoco.mj_mulJacTVec(m, d, tau, f)
        loads = self._payload_torques()
        alpha = min(1., P.RIGID_DT_S/P.REFLEX_FILTER_S)
        q = self.state.joint_q.numpy()
        for arm, dofs in self._arm_mj_dofs.items():
            raw = tau[[dof for _, dof in dofs]] - loads.get(arm, 0.) + self._vibro_tau.get(arm, 0.)
            self._tau_ext[arm] = self._tau_ext.get(arm, raw*0) + alpha*(raw - self._tau_ext.get(arm, raw*0))
            ext = np.abs(self._tau_ext[arm])
            moving = any(abs(self.ref[n][1]) > 1e-4 for n, _ in dofs)
            limit = P.REFLEX_TORQUE_NM[1 if moving else 0]
            k = int(np.argmax(ext))
            if ext[k] > limit:
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                self._stop_arm(arm)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(ext[k]), limit_nm=float(limit)))

    def _payload_torques(self):
        """Joint torques of the tool a hand holds (franka::Robot::setLoad), as in wine_pouring."""
        m, d, mujoco = self.mjm, self.mjd, self._mujoco
        held = {}
        if d.ncon:
            bb = m.geom_bodyid[np.asarray(d.contact.geom[:d.ncon])]
            for arm, fingers in self._arm_fingers.items():
                for col in (0, 1):
                    sel = np.isin(bb[:, col], fingers) & ~np.isin(bb[:, 1 - col], fingers)
                    for fb, other in zip(bb[sel, col], bb[sel, 1 - col]):
                        held.setdefault((arm, int(other)), set()).add(int(fb))
        out = {}
        g = np.array(m.opt.gravity)
        for (arm, body), fs in held.items():
            if len(fs) < 2 or body == self._arm_hand.get(arm):
                continue
            F = float(m.body_mass[body])*g
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, None, d.xipos[body], self._arm_hand[arm])
            t = jacp.T @ F
            out[arm] = out.get(arm, 0.) + t[[dof for _, dof in self._arm_mj_dofs[arm]]]
        return out

    def _guards(self):
        q = self.state.joint_q.numpy()
        bq = self.state.body_q.numpy()
        qd = self.state.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(bq).all() or not np.isfinite(qd).all():
            raise FloatingPointError(f"Non-finite state at tick {self.tick}")
        v, om = np.abs(qd[self._objects, :3]).max(initial=0.), np.abs(qd[self._objects, 3:]).max(initial=0.)
        if v > P.MAX_OBJECT_SPEED_M_S or om > P.MAX_OBJECT_SPIN_RAD_S:
            k = int(self._objects[int(np.argmax(np.abs(qd[self._objects]).max(axis=1)))])
            raise FloatingPointError(f"Unphysical speed at tick {self.tick}: {self.model.body_label[k]} "
                                     f"{np.linalg.norm(qd[k, :3]):.1f} m/s, {np.linalg.norm(qd[k, 3:]):.0f} rad/s")
        for name, st in self.structs.items():
            if not np.isfinite(st.z).all():
                raise FloatingPointError(f"Non-finite modal state of {name} at tick {self.tick}")

    # ================================================================== queries
    def sync_solver(self):
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    def body_pose(self, name_or_index):
        from scipy.spatial.transform import Rotation
        i = self.scene.bodies[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
        q = self.state.body_q.numpy()[i]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def pair_contact_forces(self, bodies_a, bodies_b):
        """{(a, b): total normal force (N)} over every MuJoCo contact between a Newton body in bodies_a and one in
        bodies_b (vectorised over the contact list)."""
        m, d = self.mjm, self.mjd
        out = {}
        if not d.ncon:
            return out
        if not hasattr(self, "_geom_newton_body"):
            sb = np.asarray(self.scene.b.shape_body)
            gs = self._geom_shape[:m.ngeom]
            self._geom_newton_body = np.where(gs >= 0, sb[np.clip(gs, 0, len(sb) - 1)], -1)
        geo = np.asarray(d.contact.geom[:d.ncon])
        bb = self._geom_newton_body[geo]
        A, B = np.asarray(list(bodies_a)), np.asarray(list(bodies_b))
        ab = np.isin(bb[:, 0], A) & np.isin(bb[:, 1], B)
        ba = np.isin(bb[:, 1], A) & np.isin(bb[:, 0], B)
        f6 = np.zeros(6)
        for i in np.flatnonzero(ab | ba):
            self._mujoco.mj_contactForce(m, d, int(i), f6)
            pair = (int(bb[i, 0]), int(bb[i, 1])) if ab[i] else (int(bb[i, 1]), int(bb[i, 0]))
            out[pair] = out.get(pair, 0.) + abs(float(f6[0]))
        return out

    @property
    def sim_time(self):
        return self.tick*P.RIGID_DT_S

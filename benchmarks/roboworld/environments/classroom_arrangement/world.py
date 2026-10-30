"""Rigid world: room 114, its furniture and the Mobile FR3 Duo on Newton 1.5 + MuJoCo (CPU), 0.5 ms steps.

Robot interface (each part a port of a qualified repository mechanism unless marked new):
  * arm and spine joints: position servos whose targets are slewed at the datasheet joint speeds (barcode world.py);
    FR3 link gravity and Coriolis terms compensated (MuJoCo's bias force on the FR3 joints added to the servo, inside
    their torque limits; the TMR, Spine and payloads weigh, and the base carries the arms);
  * Franka Hand: close at 50 mm/s until blocked, then hold a commanded force (barcode world.py);
  * FR3 collision reflexes (barcode's joint-torque reflex, extended): the external joint torque (contact rows only,
    J^T f), first-order filtered over 10 ms like the robot's own estimates, against the profile's upper joint-torque
    thresholds, and the hand wrench it implies (J_tcp^T w = tau_ext, hand frame) against the profile's upper Cartesian
    force/torque thresholds (params.REFLEX_PROFILES: contact_x5 default, franka_ros, libfranka_default); either stops
    the arm and refuses its joint targets for 1 s; velocity reflex as barcode;
  * base (new): swerve controller of the two ArgoDrives (base.py), drive torque clamped per tick to 16 N m
    accelerating / 28.4 N m braking (MuJoCo joint actuator-force range); casters and the rear rocker are passive.
Furniture friction (new): every glide geom has MuJoCo priority 1 (its own coefficient governs the glide-floor pair,
not MuJoCo's default maximum) and its coefficient is scheduled every tick from the glide's slip speed of the previous
tick, mu = mu_k + (mu_s - mu_k) exp(-(v/v_c)^2). Faults: non-finite state, a body faster than 20 m/s
or 1000 rad/s, or a MuJoCo solver warning stop the world.
"""

import math
import time
import numpy as np
from . import params as P
from .scene import Scene
from .base import BaseController


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


FINGER_LIMIT_MARGIN = 1e-3
PAD_REF_MASS = .0291


class World:
    scene_cls = Scene          # painting.world.PaintingWorld reuses this stack with its own scene

    def __init__(self, config, spec):
        self.config = c = config
        self.spec = spec
        self.newton, self.wp = newton, wp = require_engine()
        import mujoco
        self._mujoco = mujoco
        t_build = time.perf_counter()
        with wp.ScopedDevice("cpu"):
            self.scene = sc = self.scene_cls(c, newton, wp, spec)
            b = sc.b
            self.model = b.finalize(device="cpu")
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            imp = self.model.mujoco.geom_solimp.numpy()
            mass = self.model.body_mass.numpy()
            sb = np.asarray(b.shape_body)
            ref[:] = (P.CONTACT_TIMECONST, 1.)
            for i in range(len(sb)):
                if i in sc.pad_shapes:
                    ref[i] = (P.PAD_TIMECONST*math.sqrt(float(mass[int(sb[i])])/PAD_REF_MASS), 1.)
                else:
                    imp[i][:3] = P.CONTACT_SOLIMP
            self.model.mujoco.solref.assign(ref)
            self.model.mujoco.geom_solimp.assign(imp)
            lo = self.model.joint_limit_lower.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            labels = list(self.model.joint_label)
            for j, lab in enumerate(labels):
                if "finger_joint" in lab:
                    lo[qd_start[j]] -= FINGER_LIMIT_MARGIN
            self.model.joint_limit_lower.assign(lo)
            self.rig = sc.rig_pending.bind(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=P.CONTACT_CAPACITY, broad_phase="sap",
                                                     reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_cpu=True, use_mujoco_contacts=True, enable_multiccd=True, integrator="implicitfast",
                cone="elliptic", solver="newton", iterations=P.SOLVER_ITERATIONS, ls_iterations=20,
                impratio=P.SOLVER_IMPRATIO, nconmax=P.CONTACT_CAPACITY, njmax=P.CONTACT_CAPACITY*4, update_data_interval=0)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.build_s = time.perf_counter() - t_build
        mjm = self.solver.mj_model
        self.mjm, self.mjd = mjm, self.solver.mj_data
        # MuJoCo's midphase (per-body BVH pruning of geom pairs) drops the rear glide-floor contacts of every chair
        # after the first step (the chair's inertial frame is rotated: iquat far from identity), so all 29 chairs sink
        # through the floor at the back and fall over onto their backrests (48 deg) during the reset settle (E0,
        # same defect as barcode_scanning world.py). Every geom pair of each broadphase body pair
        # is tested instead.
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        mjm.opt.noslip_iterations = P.NOSLIP_ITERATIONS
        mjm.opt.noslip_tolerance = P.NOSLIP_TOLERANCE
        self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        shape_geom = {int(s): g for g, s in enumerate(self._geom_shape) if s >= 0}
        self._shape_geom = shape_geom
        # hard stops: arm joint limits, finger lead screws, spine ends, rocker range (as in barcode)
        for j in range(mjm.njnt):
            name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            width = 1e-3 if "fr3_joint" in name else 5e-4 if "finger_joint" in name else 1e-3 \
                if ("spine_vertical" in name or "rocker" in name) else None
            if "spine_vertical" in name:
                mjm.jnt_range[j, 1] = min(float(mjm.jnt_range[j, 1]), P.SPINE["stroke"])
            if width is not None and mjm.jnt_limited[j]:
                mjm.jnt_solref[j] = (2*P.DT, 1.)
                d0, dmax, mid, power = P.STOP_SOLIMP
                mjm.jnt_solimp[j] = (d0, dmax, width, mid, power)
        # glides, wheels and fingertip pads govern their own contact pairs (priority 1 over the floor and furniture)
        self.glide_geoms = np.array([shape_geom[s] for s in sc.glide_shapes], int)
        self.wheel_geoms = np.array([shape_geom[s] for s in sc.wheel_shapes], int)
        self.pad_geoms = np.array([shape_geom[s] for s in sc.pad_shapes], int)
        # textbooks too: book-on-steel 0.30 instead of the max with the book box's 0.35
        self.book_geoms = np.array([shape_geom[s] for n, it in sc.items.items() if it["kind"] == "book" for s in it["shapes"]], int)
        for g in (*self.glide_geoms, *self.wheel_geoms, *self.pad_geoms, *self.book_geoms):
            mjm.geom_priority[g] = 1
        self.glide_mu = np.array([P.GLIDE_MU[spec["glides"]]] * len(self.glide_geoms), float).reshape(-1, 2)   # static, kinetic
        mjm.geom_friction[self.glide_geoms, 0] = self.glide_mu[:, 0]
        # bodies: Newton index <-> MuJoCo id
        self._mj_body = {}
        nb = {self.model.body_label[i].replace("/", "_"): i for i in range(self.model.body_count)}
        for k in range(mjm.nbody):
            n = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, k)
            if n in nb:
                self._mj_body[nb[n]] = k
        self.arm_bodies = [i for i in sc.robot_bodies if "_fr3_" in self.model.body_label[i].split("/")[-1]]
        self.joints = dict(sc.joints)
        # Spine stroke: the Mobile FR3 Duo datasheet's 827 mm, not the URDF's 0.85 m range; the
        # hard stop above is moved to it as well
        for n, spec in self.joints.items():
            if "spine_vertical" in n:
                self.joints[n] = dict(spec, upper=min(spec["upper"], P.SPINE["stroke"]))
        self.arms = sc.arms
        self.groups = sc.groups
        mj = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self._arm_mj_dofs = {}
        for arm in self.arms:
            for name in self.groups[f"{arm}_arm"]:
                short = name.split("/")[-1]
                hit = [dof for n, dof in mj.items() if n and n.endswith("_" + short)]
                if len(hit) != 1:
                    raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
                self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._arm_mj_bodies = {k for k in (self._mj_body[i] for i in self.arm_bodies)}
        self._geom_body = np.asarray(mjm.geom_bodyid)
        self._arm_body_mask = np.zeros(mjm.nbody, bool)
        self._arm_body_mask[list(self._arm_mj_bodies)] = True
        self.base = BaseController(self.joints, sc.robot_instance, P.DT)
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        self._kp = self.model.joint_target_ke.numpy().copy()
        self._kd = self.model.joint_target_kd.numpy().copy()
        import mujoco as _mj
        jn = {(_mj.mj_id2name(self.mjm, _mj.mjtObj.mjOBJ_JOINT, j) or ""): j for j in range(self.mjm.njnt)}
        self._wheel_mj_joint = [next(j for n, j in jn.items() if n.endswith(d["wheel"]["native_label"].split("/")[-1]))
                                for d in self.base.drives]
        slewed = [n for n in self.joints if n not in self.base_joint_names]
        self._slew_q = np.array([self.joints[n]["q"] for n in slewed], int)
        self._slew_d = np.array([self.joints[n]["dof"] for n in slewed], int)
        self._slew_names = slewed
        # profiles cruise at ARM_SPEED_FRACTION of the joint speed limit: the servo (PD on the reference, velocity target,
        # no acceleration feed-forward) lags by M a / kp while the reference accelerates and releases it as a ~0.25 rad/s
        # velocity overshoot when the acceleration ends; at 95 % one legal stowed -> ready target reaches 1.05 x the
        # limit on joint 2 and trips the velocity reflex (E25). libfranka's motion generators keep a
        # margin of the same kind (kLimitEps, kJointVelocityLimitsTolerance)
        self._vmax = np.array([self.joints[n]["velocity_limit"]*(P.ARM_SPEED_FRACTION if "fr3_joint" in n else 1.)
                               for n in slewed], float)
        self._amax = np.array([P.SPINE["accel"] if "spine" in n else 1e3 if "finger" in n else P.ARM_ACCEL for n in slewed], float)
        self._vref = np.zeros(len(slewed))
        # acceleration feed-forward for the FR3 joints (tau = M(q) a_ref through MuJoCo's mass matrix), paid from the
        # same actuator budget: each tick the joint's actuator-force range is shifted by the feed-forward so that
        # servo + gravity compensation + feed-forward stays within +-87 / 12 N m. Without it the servo lagged the
        # accelerating reference by M a / kp and overshot the speed limit when the acceleration ended (E25)
        arm_dof = {n: d for a in self.arms for n, d in self._arm_mj_dofs[a]}
        self._ff_k = np.array([k for k, n in enumerate(slewed) if n in arm_dof], int)
        self._ff_mjdof = np.array([arm_dof[slewed[k]] for k in self._ff_k], int)
        self._ff_jnt = np.array([int(mjm.dof_jntid[d]) for d in self._ff_mjdof], int)
        self._ff_limit = np.array([float(mjm.jnt_actfrcrange[j][1]) for j in self._ff_jnt], float)
        self._ff_vec, self._ff_res = np.zeros(mjm.nv), np.zeros(mjm.nv)
        self.feedforward = np.zeros(len(self._ff_k))
        self.gravcomp = dict(method="qfrc_bias (gravity + Coriolis of the arm subtree) added to the FR3 joint servos",
                             joints=int(len(self._ff_k)), enabled=bool(P.ARM_GRAVCOMP), mujoco_ngravcomp=int(mjm.ngravcomp))
        robots = set(sc.robot_bodies)
        self._objects = np.array([i for i in range(self.model.body_count) if i not in robots], int)
        self.sync_solver()
        self.reflexes, self.warnings = [], []
        self.tick, self.fault = 0, None
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.reflex_stops_base = bool(getattr(config, "reflex_stops_base", True))
        self.fixture = None                  # ASSISTED callback (probes/demos only); never reachable from the policy API
        self.recorders = []                  # per-tick callbacks (traces, video pose recording)
        self.grip_force = {arm: 0. for arm in self.arms}
        self.grasp_latched, self._grip_ref = {}, {}
        self._reflex_until, self._reflex_fired, self.refused_targets = {}, {}, 0
        self._tau_f = {arm: np.zeros(7) for arm in self.arms}
        self._tau_sf = {arm: np.zeros(7) for arm in self.arms}          # signed filtered external joint torque
        self._wrench_f = {arm: np.zeros(6) for arm in self.arms}        # |hand-frame wrench| from _tau_sf
        prof = P.REFLEX_PROFILES[getattr(c, "reflex_profile", P.REFLEX_PROFILE)]
        self._reflex_limits = (np.array(prof["nominal"]), np.array(prof["acceleration"]))
        self._force_limits = (np.array(prof["force_nominal"]), np.array(prof["force_acceleration"]))
        # TCP pose in each arm's link7 frame (the Cartesian reflex maps the external joint torque to a hand wrench)
        self._tcp_in7 = {}
        for arm in self.arms:
            b7 = self.scene.robot_body[f"{arm}_fr3_link7"]
            p7, R7 = self.body_pose(b7)
            fr = self.rig.frame_pose(f"{self.scene.robot_instance}/{arm}_tcp", self.state)
            qw = fr["orientation_wxyz"]
            from scipy.spatial.transform import Rotation as _R
            Rt = _R.from_quat((qw[1], qw[2], qw[3], qw[0])).as_matrix()
            self._tcp_in7[arm] = (self._mj_body[b7], R7.T @ (np.asarray(fr["position_m"]) - p7), R7.T @ Rt)
        self._prev_glide = self.mjd.geom_xpos[self.glide_geoms, :2].copy()
        self.glide_speed = np.zeros(len(self.glide_geoms))
        self.timing = dict(physics_s=0., control_s=0.)
        self._warn0 = self._warning_counts()

    # ------------------------------------------------------------------ robot helpers
    def gripper_joints(self, arm):
        return self.groups[f"{arm}_gripper"]

    def set_grip_force(self, arm, force):
        """Per-finger force (N): 0 = position mode (the fingers hold their current width)."""
        if arm not in self.arms:
            raise ValueError(f"No arm {arm}")
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) \
                or not 0 <= force <= P.HAND["cont"]:
            raise ValueError(f"Grip force must be in [0, {P.HAND['cont']}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]
            q = self.state.joint_q.numpy()
            for key in [k for k in self._grip_ref if k[0] == arm]:
                self.targets[self.joints[key[1]]["q"]] = q[self.joints[key[1]]["q"]]
                del self._grip_ref[key]

    def set_targets(self, targets):
        """Joint position targets (rad / m). An arm stopped by its collision reflex refuses new targets for 1 s."""
        latched = {arm for arm, until in self._reflex_until.items() if self.sim_time < until}
        if latched:
            kept = {n: v for n, v in targets.items()
                    if not any(n in self.groups[f"{a}_arm"] for a in latched)}
            self.refused_targets += len(targets) - len(kept)
            targets = kept
        for name, value in targets.items():
            spec = self.joints.get(name)
            if spec is None or name in self.base_joint_names:
                raise ValueError(f"Not a commandable robot joint: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value}")
        for name, value in targets.items():
            self.targets[self.joints[name]["q"]] = value
            for key in [k for k in self.grasp_latched if k[1] == name]:
                del self.grasp_latched[key]

    @property
    def base_joint_names(self):
        return set(self.groups["base_steer"] + self.groups["base_drive"] + self.groups["base_passive"])

    # ------------------------------------------------------------------ stepping
    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        try:
            t0 = time.perf_counter()
            dt = P.DT
            cur = self.control.joint_target_q.numpy()
            vel = self.control.joint_target_qd.numpy()
            q_now = self.state.joint_q.numpy()
            qd_now = self.state.joint_qd.numpy()
            base = self.base_joint_names
            # joint references follow their targets along velocity- and acceleration-limited profiles (FR3: datasheet
            # joint speeds and 10 rad/s2; Spine 0.3 m/s, 1 m/s2), braking so as to arrive; the reference velocity is the
            # servo's velocity target (a joint controller tracking its trajectory). A velocity step without the
            # acceleration bound saturated the Spine's 670 N and left it 9.7 mm behind at up to 0.5 m/s (E6)
            sq, sd = self._slew_q, self._slew_d
            e = self.targets[sq] - cur[sq]
            vdes = np.sign(e)*np.minimum(self._vmax, np.sqrt(2*self._amax*np.abs(e)))
            v_old = self._vref.copy()
            self._vref += np.clip(vdes - self._vref, -self._amax*dt, self._amax*dt)
            move = self._vref*dt
            over = np.abs(move) > np.abs(e)
            move[over] = e[over]
            self._vref[over] = e[over]/dt
            cur[sq] += move
            vel[sd] = self._vref
            a_ref = np.clip((self._vref - v_old)/dt, -self._amax, self._amax)[self._ff_k]
            self.base.step(q_now, qd_now, cur, vel, self._kp, self._kd)
            for jid, cap in zip(self._wheel_mj_joint, self.base.wheel_cap):
                self.mjm.jnt_actfrcrange[jid] = (-cap, cap)
            jf = np.zeros(self.model.joint_dof_count, np.float32)
            for arm, force in self.grip_force.items():
                if not force:
                    continue
                for name in self.gripper_joints(arm):
                    spec = self.joints[name]
                    j, dof = spec["q"], spec["dof"]
                    key = (arm, name)
                    kp = float(self._kp[dof])
                    r = self._grip_ref.get(key, float(cur[j]))
                    step = P.HAND["speed"]*dt
                    r += float(np.clip(self.targets[j] - r, -step, step))
                    r = float(np.clip(r, q_now[j] - force/kp, q_now[j] + force/kp))
                    self._grip_ref[key] = r
                    push = kp*(r - float(q_now[j]))
                    jf[dof] = push
                    cur[j] = q_now[j]
                    if abs(push) >= .999*force:
                        self.grasp_latched[key] = True
                    else:
                        self.grasp_latched.pop(key, None)
            if (P.ARM_ACCEL_FEEDFORWARD or P.ARM_GRAVCOMP) and len(self._ff_k):
                ff = np.zeros(len(self._ff_k))
                if P.ARM_ACCEL_FEEDFORWARD:
                    self._ff_vec[:] = 0.
                    self._ff_vec[self._ff_mjdof] = a_ref
                    self._mujoco.mj_mulM(self.mjm, self.mjd, self._ff_res, self._ff_vec)
                    ff += self._ff_res[self._ff_mjdof]
                if P.ARM_GRAVCOMP:
                    # the arm subtree's gravity and velocity terms at the last step (one 0.5 ms tick old)
                    ff += np.asarray(self.mjd.qfrc_bias)[self._ff_mjdof]
                ff = np.clip(ff, -self._ff_limit, self._ff_limit)
                self.feedforward = ff
                jf[self._slew_d[self._ff_k]] += ff
                self.mjm.jnt_actfrcrange[self._ff_jnt, 0] = -self._ff_limit - ff
                self.mjm.jnt_actfrcrange[self._ff_jnt, 1] = self._ff_limit - ff
            self.control.joint_target_q.assign(cur)
            self.control.joint_target_qd.assign(vel)
            self.control.joint_f.assign(jf)
            self.state.clear_forces()
            wrench = self.body_wrench
            if self.fixture is not None:
                wrench = self.fixture(self)
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            t1 = time.perf_counter()
            self.solver.step(self.state, self.next_state, self.control, self.contacts, dt)
            self.timing["physics_s"] += time.perf_counter() - t1
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self._imu_update(dt)
            self._velocity_reflex()
            self._torque_reflex()
            self._friction_schedule()
            self._guards()
            for rec in self.recorders:
                rec(self)
            self.timing["control_s"] += time.perf_counter() - t0
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _friction_schedule(self):
        """mu(v) per glide from its slip speed over the last tick."""
        pos = self.mjd.geom_xpos[self.glide_geoms, :2]
        v = np.linalg.norm(pos - self._prev_glide, axis=1)/P.DT
        self._prev_glide = pos.copy()
        self.glide_speed = v
        mus, muk = self.glide_mu[:, 0], self.glide_mu[:, 1]
        self.mjm.geom_friction[self.glide_geoms, 0] = muk + (mus - muk)*np.exp(-(v/P.STRIBECK_V)**2)

    def _warning_counts(self):
        mj = self._mujoco
        return {w: int(self.mjd.warning[int(getattr(mj.mjtWarning, w))].number)
                for w in ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS", "mjWARN_CNSTRFULL", "mjWARN_CONTACTFULL")}

    def _guards(self):
        q = self.state.joint_q.numpy()
        qd = self.state.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all() or not np.isfinite(self.state.body_q.numpy()).all():
            raise FloatingPointError(f"Non-finite state at tick {self.tick}")
        objs = qd[self._objects]
        v, om = float(np.abs(objs[:, :3]).max(initial=0.)), float(np.abs(objs[:, 3:]).max(initial=0.))
        if v > P.MAX_OBJECT_SPEED_M_S or om > P.MAX_OBJECT_SPIN_RAD_S:
            k = int(self._objects[int(np.argmax(np.abs(objs).max(axis=1)))])
            raise FloatingPointError(f"Unphysical speed at tick {self.tick}: {self.model.body_label[k]} "
                                     f"{np.linalg.norm(qd[k, :3]):.1f} m/s, {np.linalg.norm(qd[k, 3:]):.0f} rad/s")
        if self.tick % 20 == 0:
            now = self._warning_counts()
            new = {k: v - self._warn0[k] for k, v in now.items() if v > self._warn0[k]}
            if new:
                self._warn0 = now
                self.warnings.append(dict(tick=int(self.tick), **new))
                raise FloatingPointError(f"MuJoCo solver warning at tick {self.tick}: {new}")

    def _new_stop(self, arm):
        last = self._reflex_fired.get(arm)
        self._reflex_fired[arm] = int(self.tick)
        return last is None or not (self.tick - 50 <= last < self.tick)

    def _stop_base_on_reflex(self):
        """An arm protective stop also stops the base (it decelerates at its limits) and base commands are refused
        during the arm's 1 s hold, as a real operator program would do: otherwise the base pushes on through the
        stopped arms as through struts, and the hand-force reflex would count events without bounding the force (E15).
        `reflex_stops_base=False` turns this off."""
        if self.reflex_stops_base:
            self.base.set_twist(0., 0., 0.)
            self._base_hold_until = self.sim_time + P.REFLEX_RECOVERY_S

    def base_held(self):
        return self.sim_time < getattr(self, "_base_hold_until", -1.)

    def _freeze(self, arm):
        """Protective stop: the arm's targets and references jump to where it is, at zero reference velocity."""
        q = self.state.joint_q.numpy()
        cur = self.control.joint_target_q.numpy()
        names = set(self.groups[f"{arm}_arm"])
        for k, n in enumerate(self._slew_names):
            if n in names:
                j = self.joints[n]["q"]
                self.targets[j] = cur[j] = float(q[j])
                self._vref[k] = 0.
        self.control.joint_target_q.assign(cur)

    def _velocity_reflex(self):
        qd = self.state.joint_qd.numpy()
        for arm in self.arms:
            for name in self.groups[f"{arm}_arm"]:
                spec = self.joints[name]
                v = abs(float(qd[spec["dof"]]))
                if v > 1.05*spec["velocity_limit"]:
                    self._freeze(arm)
                    self._stop_base_on_reflex()
                    self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                    if self._new_stop(arm):
                        self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="velocity", joint=name, speed_rad_s=v,
                                                  limit_rad_s=spec["velocity_limit"]))
                    break

    def _torque_reflex(self):
        """FR3 collision reflexes on the contact-derived external joint torque tau_ext = J^T f (contact rows only),
        first-order filtered (REFLEX_FILTER_S) like the robot's own estimates:
          joint:     |tau_ext| per joint against the profile's upper torque thresholds;
          Cartesian: the hand wrench w solving J_tcp^T w = tau_ext (the way the FR3 estimates K_F_ext_hat_K), in the
                     hand frame, |component| against the profile's upper force/torque thresholds.
        The acceleration-phase thresholds apply while the arm's reference is moving."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        alpha = P.DT/P.REFLEX_FILTER_S
        touching = False
        if d.ncon:
            g = np.asarray(d.contact.geom[:d.ncon])
            bodies = self._geom_body[g]
            touching = bool(self._arm_body_mask[bodies].any())
        if touching and d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            tau = np.zeros(m.nv)
            mj.mj_mulJacTVec(m, d, tau, f)
        else:
            tau = None
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
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                self._freeze(arm)
                self._stop_base_on_reflex()
                if self._new_stop(arm):
                    if over[k] > 0:
                        self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                                  external_torque_nm=float(self._tau_f[arm][k]), limit_nm=float(limits[k]),
                                                  cause=self._reflex_cause(dofs[k][1]) if tau is not None else None))
                    else:
                        axis = ("fx", "fy", "fz", "mx", "my", "mz")[kf]
                        self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="cartesian", joint=f"{arm}_hand/{axis}",
                                                  external_force=float(self._wrench_f[arm][kf]), limit=float(flim[kf]),
                                                  hand_wrench=[round(float(v), 2) for v in self._wrench_f[arm]],
                                                  cause=self._reflex_cause(dofs[int(np.argmax(self._tau_f[arm]))][1])
                                                  if tau is not None else None))

    def _reflex_cause(self, dof):
        """The contact contributing most to a reflex joint's external torque: body labels, torque, point."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        f = np.array(d.efc_force[:d.nefc])
        f[np.asarray(d.efc_type[:d.nefc]) < int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
        best = None
        for r in np.flatnonzero(f):
            one = np.zeros(d.nefc); one[r] = f[r]
            t1 = np.zeros(m.nv); mj.mj_mulJacTVec(m, d, t1, one)
            if best is None or abs(t1[dof]) > abs(best[0]):
                best = (float(t1[dof]), int(r))
        if best is None:
            return None
        c = d.contact[int(d.efc_id[best[1]])]
        sb = self.scene.b.shape_body
        names = []
        for g in c.geom:
            s = int(self._geom_shape[g]) if 0 <= g < len(self._geom_shape) else -1
            names.append(self.model.body_label[int(sb[s])].split("/")[-1] if s >= 0 and sb[s] >= 0 else "static")
        return dict(bodies=names, torque_nm=round(best[0], 2), point=np.round(c.pos, 3).tolist())

    def sync_solver(self):
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.mjm, self.mjd)

    # ------------------------------------------------------------------ queries
    def body_pose(self, name_or_index):
        from scipy.spatial.transform import Rotation
        i = self.scene.bodies[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
        q = self.state.body_q.numpy()[i]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def pose2d(self, name):
        """(x, y, yaw) of an item's body frame, and its tilt (deg) from upright."""
        p, R = self.body_pose(name)
        tilt = math.degrees(math.acos(max(-1., min(1., float(R[2, 2])))))
        return (float(p[0]), float(p[1]), math.atan2(R[1, 0], R[0, 0])), tilt

    def _imu_update(self, dt):
        """Base IMU (imu_mounting_point on base_link): angular rate and specific force (a - g) in a base-aligned frame
        (x forward, z up), the force low-pass filtered over IMU_FILTER_S from the per-tick change of the mount point's
        velocity. A reading, not an estimate: no bias or noise model."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        if not hasattr(self, "_imu_body"):
            self._imu_body = self._mj_body[self.scene.robot_body["base_link"]]
            self._imu_r = np.array([0.260, 0.0, 0.1478])
            self._imu_v, self._imu_f = None, np.array([0., 0., P.GRAVITY])
            self._v6 = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, self._imu_body, self._v6, 0)
        R = d.xmat[self._imu_body].reshape(3, 3)
        w = self._v6[:3].copy()
        v = self._v6[3:] + np.cross(w, R @ self._imu_r)
        if self._imu_v is not None:
            f = (v - self._imu_v)/dt + np.array([0., 0., P.GRAVITY])
            self._imu_f += min(1., dt/P.IMU_FILTER_S)*(R.T @ f - self._imu_f)
        self._imu_v = v
        self.imu = dict(angular_velocity_rad_s=R.T @ w, specific_force_m_s2=self._imu_f.copy())

    def lidar_scan(self):
        """Planar scans of the two corner nanoScan3 (params.LIDAR): ray-cast every step_deg over the scan angle in the
        base's horizontal plane at the mount height, against every collider except the robot's chassis; ranges in m,
        the cap where nothing is hit. Computed on request from the current state (no extra physics)."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        L = P.LIDAR
        base = self._mj_body[self.scene.robot_body["base_link"]]
        pb, R = d.xpos[base].copy(), d.xmat[base].reshape(3, 3)
        n = int(round(L["fov_deg"]/L["step_deg"])) + 1
        out = {}
        for k, (x, y, z, centre) in enumerate(L["mounts"]):
            ang = np.radians(centre + np.linspace(-L["fov_deg"]/2, L["fov_deg"]/2, n))
            dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1) @ R.T
            pnt = pb + R @ np.array([x, y, z])
            geomid, dist = np.zeros(n, np.int32), np.zeros(n)
            mj.mj_multiRay(m, d, np.ascontiguousarray(pnt, np.float64), np.ascontiguousarray(dirs.flatten(), np.float64), None,
                           True, int(base), geomid, dist, None, n, float(L["range_m"][1]))
            r = np.where((dist < 0) | (dist > L["range_m"][1]), L["range_m"][1], dist)
            out["front" if k == 0 else "rear"] = dict(angle_min_deg=centre - L["fov_deg"]/2, angle_step_deg=L["step_deg"],
                                                      ranges_m=np.round(np.maximum(r, L["range_m"][0]), 3))
        return out

    def base_pose(self):
        p, R = self.body_pose(self.scene.robot_body["base_link"])
        return float(p[0]), float(p[1]), math.atan2(R[1, 0], R[0, 0])

    def contact_table(self):
        """[(newton body a, newton body b, normal force N, shape a, shape b)] of the current contacts."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        out = []
        f6 = np.zeros(6)
        sb = self.scene.b.shape_body
        for i in range(d.ncon):
            g = d.contact[i].geom
            s = [int(self._geom_shape[x]) if 0 <= x < len(self._geom_shape) else -1 for x in g]
            bodies = [int(sb[x]) if x >= 0 else -1 for x in s]
            mj.mj_contactForce(m, d, i, f6)
            out.append((bodies[0], bodies[1], abs(float(f6[0])), s[0], s[1]))
        return out

    def contact_force(self, bodies_a, body_b):
        """Total contact force (world frame, N: normal and friction) that the bodies in `bodies_a` exert on `body_b`."""
        m, d, mj = self.mjm, self.mjd, self._mujoco
        f6 = np.zeros(6)
        total = np.zeros(3)
        sb = self.scene.b.shape_body
        for i in range(d.ncon):
            c = d.contact[i]
            s = [int(self._geom_shape[x]) if 0 <= x < len(self._geom_shape) else -1 for x in c.geom]
            b = [int(sb[x]) if x >= 0 else -1 for x in s]
            if b[1] == body_b and b[0] in bodies_a:
                sign = 1.         # MuJoCo's contact force acts on geom 2 (normal from geom 1 to geom 2)
            elif b[0] == body_b and b[1] in bodies_a:
                sign = -1.
            else:
                continue
            mj.mj_contactForce(m, d, i, f6)
            frame = np.asarray(c.frame).reshape(3, 3)          # rows: normal, tangent 1, tangent 2
            total += sign*(frame.T @ f6[:3])
        return total

    @property
    def sim_time(self):
        return self.tick*P.DT

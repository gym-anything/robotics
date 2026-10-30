"""Rigid world (Newton 1.5 + MuJoCo, CPU) with the GPU liquid coupled into every step.

Robot interface: barcode_scanning/world.py, which qualified it on the same FR3/Panda arms and Franka Hand (position
servos slewed at the datasheet joint speeds, a joint-velocity reflex, the libfranka collision reflex, the Franka Hand's
grasp semantics: close at 50 mm/s until blocked, then hold the commanded force; hard joint stops; MuJoCo no-slip).
Glassware rigid contact: rigid (4 ms) contact everywhere except the rubber fingertip pads (5 ms, anchored), as there.

Liquid coupling (liquid.py): before each 1 ms rigid step the liquid receives every body's pose and twist, advances
through its CFL substeps against the moving walls, and returns the time-averaged wrench it exerted on each body; that
wrench enters MuJoCo as an external force on the container for the step (explicit, staggered coupling).
"""

import math
import time
import numpy as np

from . import params as P
from .config import Config
from .scene import Scene

FRANKA_HAND_FINGER_SPEED = .05          # m/s per finger (Franka datasheet; barcode/label precedent)
PAD_SOLREF_TIMECONST = .005             # s, fingertip pad contact time constant (label_application, anchored)
PAD_SOLREF_REF_MASS = .0291             # kg, FR3 finger link the time constant was anchored on
FINGER_LIMIT_MARGIN = 1e-3


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
                    ref[i] = (PAD_SOLREF_TIMECONST*math.sqrt(float(mass[int(sb[i])])/PAD_SOLREF_REF_MASS), 1.)
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
            self.rig = self.scene.rig_pending.bind(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=c.contact_capacity,
                                                     broad_phase="sap", reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                      enable_multiccd=True, integrator="implicitfast", cone="elliptic",
                                                      solver="newton", iterations=c.solver_iterations, ls_iterations=20,
                                                      impratio=c.solver_impratio,
                                                      nconmax=c.contact_capacity, njmax=c.contact_capacity*3,
                                                      update_data_interval=int(c.solver_update_data_interval))
            import mujoco
            self._mujoco = mujoco
            mjm = self.solver.mj_model
            for j in range(mjm.njnt):
                name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if any(t in name for t in ("fr3_joint", "panda_joint")) and "finger" not in name:
                    mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 1e-3, mid, power)
                elif "finger_joint" in name:
                    mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 5e-4, mid, power)
            # barcode_scanning: MuJoCo's midphase silently dropped valid convex-hull pairs in one model; test every geom
            # pair of each broadphase body pair instead
            mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads:
                    mjm.geom_priority[g] = 1
                    if P.PAD_TORSIONAL_M > 0:
                        # torsional friction about the contact normal (the pads decide: higher priority)
                        mjm.geom_condim[g] = 4
                        mjm.geom_friction[g][1] = P.PAD_TORSIONAL_M
            # The Franka Hand has one motor and a rack. The URDF mimic is a MuJoCo equality,
            # but its default 20 ms softness lets the two 3,500 N/m drives overpower it.
            for e in range(mjm.neq):
                if int(mjm.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
                    joint = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(mjm.eq_obj1id[e])) or ""
                    if "finger_joint" in joint:
                        mjm.eq_solref[e] = (2*c.rigid_dt, 1.)
                        d0, dmax, mid, power = P.STOP_SOLIMP
                        mjm.eq_solimp[e] = (d0, dmax, 5e-4, mid, power)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        self._reflex_until, self.refused_targets, self._reflex_fired = {}, 0, {}
        mjm = self.solver.mj_model
        mj = {self._mujoco.mj_id2name(mjm, self._mujoco.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self._arm_mj_dofs = {}
        for name in self.joints:
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mname, dof in mj.items() if mname and mname.startswith(arm + "_") and mname.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._finger_speed = {j: FRANKA_HAND_FINGER_SPEED for arm in self.scene.arm_hw for j in self.gripper_joints(arm)}
        # MuJoCo bodies of each arm's fingers and hand (payload model of the collision reflex)
        mjname = lambda b: self._mujoco.mj_id2name(mjm, self._mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self._arm_fingers = {arm: [b for b in range(mjm.nbody) if mjname(b).startswith(arm + "_") and "finger" in mjname(b)]
                             for arm in self._arm_mj_dofs}
        self._arm_hand = {arm: int(mjm.body_parentid[fs[0]]) for arm, fs in self._arm_fingers.items()}
        self._prev_cvel = None
        self._tau_ext = {}
        m2n = np.asarray(self.solver.mjc_body_to_newton.numpy())[0]
        self._mj_to_newton = {j: int(n) for j, n in enumerate(m2n)}
        self._ready_pose()
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        robots = set(self.scene.robot_bodies)
        self._objects = np.array([bb for bb in range(self.model.body_count) if bb not in robots], dtype=int)
        self.sync_solver()
        self.reflexes = []
        self.tick, self.fault = 0, None
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched, self._grip_ref = {}, {}
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {n: float(ke[self.joints[n]["dof"]]) for arm in self.scene.arm_hw for n in self.gripper_joints(arm)}
        self.timing = dict(rigid_s=0., liquid_s=0.)
        self.fixture = None          # ASSISTED callback (probes/demos only), never reachable from the policy API
        self.phantom_log = []
        self.body_com = self.model.body_com.numpy().astype(np.float64)
        self.liquid = None
        self.gas = None
        self.lumped = None
        self.liquid_wrench = np.zeros((self.model.body_count, 6))
        self._init_motion()

    def _init_motion(self):
        """Each arm's motion generator (motion.py: FCI joint velocity/acceleration/jerk and TCP limits) and the arm
        joints it drives, in the kinematic chain's order."""
        from .kinematics import ArmKinematics
        from .motion import ArmMotion, family
        self.kin = ArmKinematics(self)
        self.motion, self._motion_q = {}, {}
        cur = self.control.joint_target_q.numpy()
        for arm, (chain, names) in self.kin.arms.items():
            idx = np.array([self.joints[n]["q"] for n in names], int)
            vmax = [float(self.joints[n]["velocity_limit"]) for n in names]
            self.motion[arm] = ArmMotion(family(self.scene.arm_hw[arm]["model"]), vmax, self.config.rigid_dt, cur[idx],
                                         fk=chain.fk)
            self._motion_q[arm] = (idx, names)
        self._arm_joint_q = {int(j) for idx, _ in self._motion_q.values() for j in idx}

    def servo_state(self):
        return {arm: m.state() for arm, m in self.motion.items()}

    def set_servo_state(self, s):
        for arm, st in s.items():
            self.motion[arm].set_state(st)

    def _hold_arm(self, arm, q):
        """Protective stop: the arm's goal and reference at its measured position, at rest."""
        idx, names = self._motion_q[arm]
        for n in names:
            self.targets[self.joints[n]["q"]] = float(q[self.joints[n]["q"]])
        self.motion[arm].hold(q[idx])

    # ------------------------------------------------------------------ liquid
    def attach_liquid(self):
        from .liquid import Liquid
        sc = self.scene
        boxes = [(c, h) for name, c, h, m in sc.static_boxes]
        self.liquid = Liquid(self.model.body_count, [(bd, geo) for bd, geo, _ in sc.containers], device=self.config.device,
                             planes=sc.liquid_planes, boxes=boxes, obbs=self.robot_obbs(), hulls=self.robot_hulls(),
                             surface=self.config.surface_forces)
        self._push_bodies()
        from .gas import GasModel
        from .coupling import LumpedLiquid
        self.gas = GasModel(self) if self.config.gas else None
        self.lumped = LumpedLiquid(self) if self.config.lump_liquid else None
        return self.liquid

    def robot_hulls(self, n_dirs=48, merge_cos=0.97):
        """Every robot mesh collision shape (links 1-7, the hand) as a convex polytope for the liquid guard: the planes
        of the shape's convex hull, merged where their normals are within ~14 deg and completed with fixed directions,
        each at the hull's support distance, so the polytope contains the hull and is tight on its faces (a few
        millimetres loose on rounded parts). Returns (body, centre, radius, planes (m, 4) n.x <= d) in body frames.
        Their axis-aligned bounding boxes, used first, were centimetres loose: the hand's box reached through a gripped
        bottle's glass into the wine and pinned it."""
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        b = self.scene.b
        robots = set(self.scene.robot_bodies)
        k = np.arange(n_dirs) + 0.5
        ph, th = np.arccos(1 - 2*k/n_dirs), math.pi*(1 + 5**0.5)*k
        fib = np.stack([np.cos(th)*np.sin(ph), np.sin(th)*np.sin(ph), np.cos(ph)], 1)
        out = []
        for i in range(b.shape_count):
            body = b.shape_body[i]
            src = b.shape_source[i]
            if body not in robots or not (int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            if src is None or not hasattr(src, "vertices") or "finger" in b.body_label[body]:
                continue
            tf = np.asarray(b.shape_transform[i], float)
            v = Rotation.from_quat(tf[3:]).apply(np.asarray(src.vertices, float)*np.asarray(b.shape_scale[i], float)) + tf[:3]
            hull = ConvexHull(v)
            nrm, area = hull.equations[:, :3], hull.area
            tri_area = np.array([np.linalg.norm(np.cross(v[s_[1]] - v[s_[0]], v[s_[2]] - v[s_[0]]))/2 for s_ in hull.simplices])
            dirs = []
            for j in np.argsort(-tri_area):
                if all(float(nrm[j] @ d_) < merge_cos for d_ in dirs):
                    dirs.append(nrm[j])
            dirs = np.array(dirs + [d_ for d_ in fib if all(float(d_ @ e) < merge_cos for e in dirs)])
            hv = (v @ dirs.T).max(0)
            c = v.mean(0)
            out.append((int(body), tuple(c), float(np.linalg.norm(v - c, axis=1).max()),
                        np.concatenate([dirs, hv[:, None]], 1).astype(np.float32)))
        return out

    def robot_obbs(self):
        """The finger pads and other box collision shapes of the robot as oriented boxes for the liquid guard (exact
        for box pads); the links and hand are convex polytopes (robot_hulls)."""
        b = self.scene.b
        robots = set(self.scene.robot_bodies)
        out = []
        for i in range(b.shape_count):
            body = b.shape_body[i]
            if body not in robots or not (int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            if "finger" not in b.body_label[body] and b.shape_type[i] != self.newton.GeoType.BOX:
                continue
            tf = np.asarray(b.shape_transform[i], float)
            scale = np.asarray(b.shape_scale[i], float)
            if b.shape_type[i] == self.newton.GeoType.BOX:
                out.append((int(body), tuple(tf[:3]), tuple(tf[3:]), tuple(scale)))
            elif b.shape_source[i] is not None and hasattr(b.shape_source[i], "vertices"):
                v = np.asarray(b.shape_source[i].vertices, float)*scale
                lo, hi = v.min(0), v.max(0)
                from scipy.spatial.transform import Rotation
                c = tf[:3] + Rotation.from_quat(tf[3:]).apply(0.5*(lo + hi))
                out.append((int(body), tuple(c), tuple(tf[3:]), tuple(0.5*(hi - lo))))
        return out

    def _push_bodies(self):
        q = self.state.body_q.numpy().astype(np.float64)
        qd = self.state.body_qd.numpy().astype(np.float64)
        self.liquid.set_bodies(q, qd, self.body_com)

    # ------------------------------------------------------------------ robot helpers
    def _ready_pose(self):
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        for name, hw in self.scene.arm_hw.items():
            for joint, v in zip(hw["arm_joints"] + hw["gripper_joints"], P.READY_POSE):
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        self.model.joint_q.assign(q)
        self.control.joint_target_q.assign(t)

    def gripper_joints(self, arm):
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

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
            q = self.state.joint_q.numpy()
            t = self.control.joint_target_q.numpy()
            for key in [k for k in self._grip_ref if k[0] == arm]:
                t[self.joints[key[1]]["q"]] = q[self.joints[key[1]]["q"]]
                del self._grip_ref[key]
            self.control.joint_target_q.assign(t)

    def set_targets(self, targets, ramp_ticks=1):
        """Joint position targets. An arm stopped by its collision reflex refuses new arm targets for
        P.REFLEX_RECOVERY_S (finger targets pass), as on the real robot until an error recovery. Arm targets are goals
        of the arm's motion generator (motion.py), reached by a linear ramp over `ramp_ticks` (the command's duration)
        under the FCI velocity, acceleration and jerk limits."""
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
        targets = dict(targets)
        for arm in self.rig.instances:
            fingers = self.gripper_joints(arm)
            widths = [float(targets[name]) for name in fingers if name in targets]
            if widths:
                # The nearest legal command to conflicting imported coordinates is one shared half width.
                width = sum(widths)/len(widths)
                targets.update({name: width for name in fingers})
        for name, value in targets.items():
            self.targets[self.joints[name]["q"]] = value
            if "finger" in name:
                for key in [k for k in self.grasp_latched if k[1] == name]:
                    del self.grasp_latched[key]
        for arm, (idx, names) in self._motion_q.items():
            mask = np.array([n in targets for n in names])
            if mask.any():
                self.motion[arm].command(self.targets[idx], ramp_ticks, mask)

    # ------------------------------------------------------------------ stepping
    def step(self):
        """One rigid step (rigid_dt) with the liquid advanced over the same interval."""
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            cur = self.control.joint_target_q.numpy()
            for arm, (idx, _) in self._motion_q.items():
                cur[idx] = self.motion[arm].step()
                self.motion[arm].advance_fk()
            for name, spec in self.joints.items():
                j = spec["q"]
                if j in self._arm_joint_q:
                    continue
                vlim = spec["velocity_limit"]
                if name in self._finger_speed:
                    vlim = min(vlim, self._finger_speed[name])
                slew = vlim*c.rigid_dt
                cur[j] += np.clip(self.targets[j] - cur[j], -slew, slew)
            jf = np.zeros(self.model.joint_dof_count, np.float32)
            q_now = self.state.joint_q.numpy()
            for arm, force in self.grip_force.items():
                if not force:
                    continue
                # Franka Hand grasp (barcode_scanning/world.py): a speed-limited drive whose push is capped at `force`
                for name in self.gripper_joints(arm):
                    spec = self.joints[name]
                    j, dof = spec["q"], spec["dof"]
                    key = (arm, name)
                    kp = self._finger_kp[name]
                    ref = self._grip_ref.get(key, float(cur[j]))
                    step = self._finger_speed[name]*c.rigid_dt
                    ref += float(np.clip(self.targets[j] - ref, -step, step))
                    ref = float(np.clip(ref, q_now[j] - force/kp, q_now[j] + force/kp))
                    self._grip_ref[key] = ref
                    push = kp*(ref - float(q_now[j]))
                    jf[dof] = push
                    cur[j] = q_now[j]
                    if abs(push) >= .999*force:
                        self.grasp_latched[key] = True
                    else:
                        self.grasp_latched.pop(key, None)
            self.control.joint_target_q.assign(cur)
            self.control.joint_f.assign(jf)
            self.state.clear_forces()
            wrench = self.body_wrench.astype(np.float64).copy()
            t1 = time.perf_counter()
            if self.liquid is not None and self.liquid.n:
                t1 = time.perf_counter()
                self._push_bodies()
                if self.lumped is not None:
                    self.lumped.update(c.rigid_dt)
                if self.gas is not None:
                    self.gas.update(self.sim_time)
                self.liquid_wrench = self.liquid.step(c.rigid_dt)
                if self.lumped is not None:
                    self.liquid_wrench = self.liquid_wrench + self.lumped.impulse_wrench
                wrench += self.liquid_wrench
                self.timing["liquid_s"] += time.perf_counter() - t1
            if self.lumped is not None and self.liquid is not None and self.liquid.n:
                wrench = self.body_wrench.astype(np.float64) + self.lumped.correct(self.liquid_wrench)
            if self.fixture is not None:
                wrench = self.fixture(self, wrench)
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            t2 = time.perf_counter()
            self.solver.step(self.state, self.next_state, self.control, self.contacts, c.rigid_dt)
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self._velocity_reflex()
            self._torque_reflex()
            q = self.state.joint_q.numpy()
            qd = self.state.body_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(self.state.body_q.numpy()).all() or not np.isfinite(qd).all():
                raise FloatingPointError(f"Non-finite state at tick {self.tick}")
            v, om = np.abs(qd[self._objects, :3]).max(initial=0.), np.abs(qd[self._objects, 3:]).max(initial=0.)
            if v > P.MAX_OBJECT_SPEED_M_S or om > P.MAX_OBJECT_SPIN_RAD_S:
                k = int(self._objects[int(np.argmax(np.abs(qd[self._objects]).max(axis=1)))])
                raise FloatingPointError(f"Unphysical speed at tick {self.tick}: {self.model.body_label[k]} "
                                         f"{np.linalg.norm(qd[k, :3]):.1f} m/s, {np.linalg.norm(qd[k, 3:]):.0f} rad/s")
            self.timing["step_s"] = self.timing.get("step_s", 0.) + time.perf_counter() - t0
            self.timing["rigid_s"] = self.timing["step_s"] - self.timing["liquid_s"]
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _new_stop(self, arm):
        last = self._reflex_fired.get(arm)
        self._reflex_fired[arm] = int(self.tick)
        return last is None or not (self.tick - 50 <= last < self.tick)

    def _velocity_reflex(self):
        qd = self.state.joint_qd.numpy()
        q = self.state.joint_q.numpy()
        for arm in self.rig.instances:
            hit = None
            for name, spec in self.joints.items():
                if not name.startswith(arm + "/") or "finger" in name:
                    continue
                v = abs(float(qd[spec["dof"]]))
                if v > 1.05*spec["velocity_limit"]:
                    hit = (name, v, spec["velocity_limit"]); break
            if hit:
                self._hold_arm(arm, q)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, joint=hit[0], speed_rad_s=hit[1], limit_rad_s=hit[2]))

    def _torque_reflex(self):
        """FR3/Panda collision reflex (libfranka reference thresholds, P.REFLEX_TORQUE_NM): contact forces on the robot
        mapped to its joints (J^T f over contact rows only) above 20 N m while accelerating / 10 N m otherwise stop the
        arm. The liquid's force on a held bottle reaches the arm through the grasp contacts and counts like any load."""
        m, d = self.solver.mj_model, self.solver.mj_data
        mujoco = self._mujoco
        if not d.nefc:
            self._tau_ext = {}
            return
        f = np.array(d.efc_force[:d.nefc])
        f[np.asarray(d.efc_type[:d.nefc]) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
        if not f.any():
            return
        tau = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, tau, f)
        cur = self.control.joint_target_q.numpy()
        q = self.state.joint_q.numpy()
        loads = self._payload_torques()
        alpha = min(1., self.config.rigid_dt/P.REFLEX_FILTER_S)
        for arm, dofs in self._arm_mj_dofs.items():
            raw = tau[[dof for _, dof in dofs]] - loads.get(arm, 0.)
            self._tau_ext[arm] = self._tau_ext.get(arm, raw) + alpha*(raw - self._tau_ext.get(arm, raw))
            ext = np.abs(self._tau_ext[arm])
            moving = self.motion[arm].moving()
            limit = P.REFLEX_TORQUE_NM[1 if moving else 0]
            k = int(np.argmax(ext))
            if ext[k] > limit:
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                self._hold_arm(arm, q)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(ext[k]), limit_nm=float(limit)))

    def _payload_torques(self):
        """Joint torques of the payload each hand is holding, as the FR3 controller would model a configured load
        (franka::Robot::setLoad, which a bartending application sets for the bottle it picks up): a body touched by both
        fingers of an arm is its payload; its weight and inertial force M (g - a) at its centre of mass (liquid lumped,
        coupling.py) map to the arm joints through the hand's Jacobian at that point. Without this, holding a full
        1.25 kg bottle read as 22-50 N m of 'collision' while the arm tilted it, and the reflex stopped every pour."""
        m, d, mujoco = self.solver.mj_model, self.solver.mj_data, self._mujoco
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
            if len(fs) < 2 or body == self._arm_hand[arm]:
                continue
            nb = self._mj_to_newton.get(body)
            if nb is None or nb < 0:
                continue
            qd = self.state.body_qd.numpy()[nb]
            if self._prev_cvel is not None and nb in self._prev_cvel:
                a = (qd[:3] - self._prev_cvel[nb])/self.config.rigid_dt
            else:
                a = np.zeros(3)
            F = float(m.body_mass[body])*(g - np.clip(a, -50., 50.))
            jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, jacr, d.xipos[body], self._arm_hand[arm])
            t = jacp.T @ F
            out[arm] = out.get(arm, 0.) + t[[dof for _, dof in self._arm_mj_dofs[arm]]]
        self._prev_cvel = {nb: self.state.body_qd.numpy()[nb, :3].copy() for nb in self._held_newton(held)}
        return out

    def _held_newton(self, held):
        return [self._mj_to_newton[b] for (_, b) in held if self._mj_to_newton.get(b, -1) >= 0]

    def sync_solver(self):
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    # ------------------------------------------------------------------ queries
    def body_pose(self, name_or_index):
        from scipy.spatial.transform import Rotation
        i = self.scene.bodies[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
        q = self.state.body_q.numpy()[i]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def pair_contact_forces(self, bodies_a, bodies_b):
        """{(a, b): total normal force (N)} over every contact between a body in bodies_a and one in bodies_b."""
        m, d = self.solver.mj_model, self.solver.mj_data
        A, B = set(bodies_a), set(bodies_b)
        f6 = np.zeros(6)
        out = {}
        sb = self.scene.b.shape_body
        for i in range(d.ncon):
            g = d.contact[i].geom
            s = [int(self._geom_shape[x]) if 0 <= x < len(self._geom_shape) else -1 for x in g]
            b0, b1 = [int(sb[x]) if x >= 0 else -1 for x in s]
            pair = (b0, b1) if b0 in A and b1 in B else (b1, b0) if b1 in A and b0 in B else None
            if pair:
                self._mujoco.mj_contactForce(m, d, i, f6)
                out[pair] = out.get(pair, 0.) + abs(float(f6[0]))
        return out

    @property
    def sim_time(self):
        return self.tick*self.config.rigid_dt

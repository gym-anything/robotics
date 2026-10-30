"""Rigid world: robots, scanner (with its trigger), keypad, cradle and stock on Newton 1.5 + MuJoCo (CPU).

The robot interface follows label_application/world.py, which qualified it on the same arms and hand:
position servos slewed at the datasheet joint speeds, a joint-velocity reflex, the Franka Hand's grasp
semantics (close at 50 mm/s until blocked, then hold a commanded force) and fingertip-pad contact stiffness
anchored to the pad compliance. Added here: the FR3/Panda collision reflex (external joint torque over the libfranka
reference thresholds stops the arm), hard arm joint limits, rigid contact parameters for every non-pad shape, and
MuJoCo's no-slip post-solver.

Switches are read from their joint coordinates every physics tick, with the hysteresis and debounce in
params.py: the trigger's microswitch (Omron D2F class) and the keypad's key switches (Cherry MX class).
The cradle's seat sensor needs geometric seating and persistent contact. Nothing here is actuated by an API
call: a switch changes only when a body pushes its lever or key far enough.
"""

import math
import time
import numpy as np
from . import params as P
from .config import Config
from .scene import Scene

FRANKA_HAND_FINGER_SPEED = .05          # m/s per finger (Franka datasheet; label_application)
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


class Switch:
    """Snap-action switch on a joint coordinate: closes at `close`, opens at `open` (< close), debounced."""

    def __init__(self, close, open_, debounce_s):
        self.close, self.open, self.debounce = close, open_, debounce_s
        self.raw = self.state = False
        self.since = 0.
        self.events = []

    def update(self, x, t):
        raw = (x >= self.close) if not self.raw else (x > self.open)
        if raw != self.raw:
            self.raw, self.since = raw, t
        if self.state != self.raw and t - self.since >= self.debounce - 1e-9:
            self.state = self.raw
            self.events.append((t, self.state))
            return True
        return False


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
            # rigid contact everywhere except the fingertip rubber: MuJoCo's default (20 ms, impedance 0.9-0.95) let a
            # finger housing sink 33 mm into a steel shelf upright and two arm links 28 mm into each other under the
            # arms' own torque limits (E17 random targets); 4 ms (4 steps, MuJoCo's floor is 2) and 0.95-0.99 keep
            # those under 6 mm (2 ms went unstable in one case). Only the pad shapes keep the anchored rubber value.
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
                                                      solver="newton", iterations=c.solver_iterations,
                                                      ls_iterations=int(c.solver_ls_iterations),
                                                      impratio=c.solver_impratio,
                                                      nconmax=c.contact_capacity, njmax=c.contact_capacity*3,
                                                      update_data_interval=int(c.solver_update_data_interval))
            import mujoco
            self._mujoco = mujoco
            # rigid stops for the trigger and key travel: MuJoCo's default limit softness (0.02 s) let the 1 N trigger
            # preload sink 0.93 mm past its rest stop; 2 ms = 2 physics steps is MuJoCo's stability floor
            mjm = self.solver.mj_model
            for j in range(mjm.njnt):
                name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if name == "scanner_trigger_pivot" or (name.startswith("key_") and name.endswith("_travel")):
                    mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                    # near-rigid plastic stops: impedance 0.99 -> 0.999 within 0.025 mm of blade/key travel
                    width = 1e-3 if name == "scanner_trigger_pivot" else 2.5e-5        # rad (x 25 mm arm) / m
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, width, mid, power)
                elif any(t in name for t in ("fr3_joint", "panda_joint")) and "finger" not in name:
                    # the arm's joint limits are hard stops too: under a legal full-torque jam the soft default let a
                    # joint run 0.135 rad (7.7 deg) past its limit (E17, library keypad jam)
                    mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 1e-3, mid, power)
                elif "finger_joint" in name:
                    # the Franka Hand's fingers ride a lead screw: they do not open past 40 mm when an object is forced
                    # between them (the soft default let 300-600 N spread them to 48 mm, E7)
                    mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                    d0, dmax, mid, power = P.STOP_SOLIMP
                    mjm.jnt_solimp[j] = (d0, dmax, 5e-4, mid, power)
            # MuJoCo's midphase (per-body BVH pruning of geom pairs) silently dropped every scanner-handle / cradle-cup
            # contact in the dual-panda model although the geometry, bitmasks, exclusions and both bodies' BVH boxes
            # were identical to the dual-fr3 model's (handle contacts 0 with the midphase, 30 without; the scanner tipped
            # out of its cradle at tick 0): test every geom pair of each broadphase body pair instead (no measurable
            # cost at this scene size)
            mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            # fingertip pads govern their own contacts (MuJoCo geom priority): the pair's compliance is the rubber pad's
            # (5 ms, anchored), not an average with the other geom's default 20 ms -- averaged, the pad-to-scanner
            # contact ran at 12.5 ms and a 17.5 N pinch crept and rolled the 214 g scanner out of the hand (E7)
            pads = set(int(k) for k in self.scene.pad_shapes)
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads:
                    mjm.geom_priority[g] = 1
                    # spin resistance at the pads too (condim 4, the same (2/3) mu N a law and coefficient as the free
                    # bodies', a 1.9 mm patch at the pad's mu 0.8): with point contacts only, a scanner turned 9 deg in a
                    # 35 N pinch by a legal whip kept sliding in the hand for 30 s (4.0 mm, still 0.1 mm/s); a face
                    # contact's corner points resist spin with or without it
                    mjm.geom_condim[g] = 4
                    mjm.geom_friction[g][1] = P.TORSIONAL_FRICTION_M
            # the Franka Hand's two fingers are one mechanism (one drive, both fingers mirrored): Newton's URDF importer
            # turns the URDF's mimic into a MuJoCo joint equality constraint at MuJoCo's default 20 ms softness, which
            # the 3 500 N/m finger servos overpowered (fingers driven to different targets sat 10 mm apart).
            # It is made as stiff as the hard stops (2 steps), so the hand closes symmetrically and
            # an off-centre object is pushed to the middle by the first finger to touch it, as on the real hand.
            for e in range(mjm.neq):
                if int(mjm.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
                    j1 = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(mjm.eq_obj1id[e])) or ""
                    if "finger_joint" in j1:
                        mjm.eq_solref[e] = (2*c.rigid_dt, 1.)
                        d0, dmax, mid, power = P.STOP_SOLIMP
                        mjm.eq_solimp[e] = (d0, dmax, 5e-4, mid, power)
            # spin resistance (MuJoCo condim 4, torsional friction) on every free body's contacts: a point contact
            # (condim 3) resists no spin about its normal, and a 510 g bottle standing tilted on its rim against a
            # stopped arm spun and slid back and forth at 1.75 mm/s for 30 s (with any torsional coefficient
            # from 0.5 mm up it came to rest). A real contact patch of radius a resists up to
            # (2/3) mu N a; params.TORSIONAL_FRICTION_M. The fingertip pads keep governing their own contacts
            # (priority 1, with the same torsional term, above).
            robot = set(int(b) for b in self.scene.robot_bodies)
            shape_body = self.model.shape_body.numpy()
            free = {b for b in range(mjm.nbody) if mjm.body_jntnum[b] > 0
                    and int(mjm.jnt_type[mjm.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE)}
            for g in range(mjm.ngeom):
                if int(mjm.geom_bodyid[g]) in free and g < len(self._geom_shape) and int(self._geom_shape[g]) >= 0 \
                        and int(shape_body[int(self._geom_shape[g])]) not in robot:
                    mjm.geom_condim[g] = 4
                    mjm.geom_friction[g][1] = P.TORSIONAL_FRICTION_M
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        self._reflex_until, self.refused_targets = {}, 0
        self._reflex_fired = {}              # arm -> last tick a protective stop fired (one record per stop episode)
        self._ramp = None
        # each arm's 7 joints -> MuJoCo dof address (for the collision reflex): MuJoCo names are the Newton labels
        # with '/' -> '_' (e.g. right/fr3_joint4 -> right_fr3_fr3_joint4 for the prefixed catalog link names)
        mjm = self.solver.mj_model
        mj = {self._mujoco.mj_id2name(mjm, self._mujoco.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self._arm_mj_dofs = {}
        self.g1 = bool(getattr(self.scene, "g1", False))
        self._g1_ff = None
        if self.g1 and c.g1_arm_gravity_ff:
            # G1 upper-body servos: position PD plus a gravity feed-forward torque computed from the model at the commanded
            # configuration, as Unitree's own G1 manipulation stack sends it for the arms (xr_teleoperate robot_arm_ik.py:
            # pinocchio rnea(q_target, 0, 0) -> motor_cmd.tau; web-sourced). The waist gets the same (on
            # the real robot Unitree's balance controller holds it; without it the PD waist sagged 0.074 rad = 4.2 deg
            # under the upper body's weight, measured). Legs and Dex3 fingers are PD only.
            mjm = self.solver.mj_model
            names = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(mjm.njnt)}
            arm = [n for n in self.scene.joints if any(k in n for k in ("_shoulder_", "_elbow_", "_wrist_", "/waist_"))]
            mj_j = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in arm]
            self._g1_ff = dict(names=arm, q=[self.scene.joints[n]["q"] for n in arm], dof=[self.scene.joints[n]["dof"] for n in arm],
                               mj_q=[int(mjm.jnt_qposadr[j]) for j in mj_j], mj_dof=[int(mjm.jnt_dofadr[j]) for j in mj_j],
                               limit=np.array([float(self.scene.joints[n]["effort_limit"]) for n in arm]),
                               data=mujoco.MjData(mjm), tau=np.zeros(len(arm)),
                               every=max(1, int(round(.005/c.rigid_dt))))
        for name in ([] if self.g1 else self.joints):        # the G1 has no libfranka collision reflex (g1.py)
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mname, dof in mj.items() if mname and mname.startswith(arm + "_") and mname.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._finger_speed = {j: FRANKA_HAND_FINGER_SPEED for arm in self.scene.arm_hw for j in self.gripper_joints(arm)}
        self._finger_pairs = [(arm, *(self.joints[n]["q"] for n in self.gripper_joints(arm)))
                              for arm in self.scene.arm_hw if len(self.gripper_joints(arm)) == 2]
        self._ready_pose()
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        robots = set(getattr(self.scene, "robot_bodies", ()))
        self._objects = np.array([b for b in range(self.model.body_count) if b not in robots], dtype=int)
        self.sync_solver()
        self.reflexes = []
        self.tick, self.fault = 0, None
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched = {}
        self._grip_ref = {}                  # force-mode fingers: the drive's slewed reference (see step)
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {n: float(ke[self.joints[n]["dof"]]) for arm in self.scene.arm_hw for n in self.gripper_joints(arm)}
        self.timing = dict(rigid_s=0.)
        # joint coordinates of the mechanisms
        jstart = self.model.joint_q_start.numpy()
        self._q_index = {lab: int(jstart[j]) for j, lab in enumerate(labels)}
        self.trigger_q = self._q_index["scanner_trigger_pivot"]
        self.trigger = Switch(P.TRIGGER_CLOSE_M/P.TRIGGER_ARM_M, (P.TRIGGER_CLOSE_M - P.TRIGGER_HYSTERESIS_M)/P.TRIGGER_ARM_M,
                              P.SWITCH_DEBOUNCE_S)
        self.key_q = {k: self._q_index[f"key_{k}_travel"] for k in self.scene.keys}
        self.key_switch = {k: Switch(P.KEY_OPERATE_M, P.KEY_RELEASE_M, P.SWITCH_DEBOUNCE_S) for k in self.scene.keys}
        self._spring_targets()
        self.seated, self._seat_since, self._cup_ok = False, None, None
        self.fixture = None          # ASSISTED callback (probes/demos only), never reachable from the policy API
        self.phantom_log = []

    def _spring_targets(self):
        """Preloaded return springs: drive targets beyond the rest stop (Newton initialises limited joints' targets
        to mid-range, which held the trigger 1.8 mm down with nothing touching it), joints start at rest."""
        k_t = P.TRIGGER_RATE_N_PER_M*P.TRIGGER_ARM_M**2
        k_k = (P.KEY_FORCE_OPERATE_N - P.KEY_FORCE_INITIAL_N)/P.KEY_OPERATE_M
        tq = self.control.joint_target_q.numpy()
        jq = self.state.joint_q.numpy()
        tq[self.trigger_q] = -P.TRIGGER_PRELOAD_N*P.TRIGGER_ARM_M/k_t
        jq[self.trigger_q] = 0.
        for qi in self.key_q.values():
            tq[qi] = -P.KEY_FORCE_INITIAL_N/k_k
            jq[qi] = 0.
        self.control.joint_target_q.assign(tq)
        self.state.joint_q.assign(jq)
        self.model.joint_q.assign(jq)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.sync_solver()

    # ------------------------------------------------------------------ robot helpers
    def _ready_pose(self):
        """Environment-owned tucked pose (P.READY_POSE): hands 0.2 m in front of the shelf face at 1.16 m, clear of
        the stock, the cradle riser and each other (the label_application pose reaches 0.46 m and starts both hands
        inside the shelf)."""
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        for name, hw in self.scene.arm_hw.items():
            for joint, v in zip(hw["arm_joints"] + hw["gripper_joints"], P.READY_POSE):
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        if getattr(self.scene, "g1", False):          # G1: the preset's arm posture over the profile's pose (g1.LAYOUT)
            for key, v in (self.scene.g1_hw["layout"].get("arms") or {}).items():
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
            # back to position mode: the servo resumes from where the fingers are
            q = self.state.joint_q.numpy()
            t = self.control.joint_target_q.numpy()
            for key in [k for k in self._grip_ref if k[0] == arm]:
                t[self.joints[key[1]]["q"]] = q[self.joints[key[1]]["q"]]
                del self._grip_ref[key]
            self.control.joint_target_q.assign(t)

    def set_targets(self, targets, ramp_ticks=None):
        """Joint position targets. An arm stopped by its collision reflex keeps its frozen targets (and refuses new
        ones) for P.REFLEX_RECOVERY_S: the real robot stays stopped until an error recovery; commands streamed at it
        meanwhile would otherwise keep pushing into the contact (E17: a fingertip held 9.7 mm into a keycap).
        `ramp_ticks` (G1 Cartesian commands, whiteboard/ik.py precedent): the targets move from their current values to
        the new ones along a quintic ramp over that many ticks (zero velocity and acceleration at both ends), streamed
        like a real controller's interpolated set points; any later set_targets call ends a running ramp."""
        # the Franka Hand's two fingers are one mechanism (one drive; the stiff mimic constraint, __init__): a target for
        # either finger sets both, two different ones their mean. Independent targets made the servos fight the coupling
        # and one finger was forced 19 mm past its limit (E17 random targets). The backend refuses unequal
        # finger targets from a policy before they reach here
        for arm in getattr(self.scene, "arm_hw", {}):
            fj = self.gripper_joints(arm)
            given = [float(targets[n]) for n in fj if n in targets and not isinstance(targets[n], bool)
                     and isinstance(targets[n], (int, float)) and math.isfinite(targets[n])]
            if given and len(fj) > 1:
                targets = dict(targets, **{n: sum(given)/len(given) for n in fj})
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
        if targets:
            self._ramp = None
        if ramp_ticks and targets:
            if type(ramp_ticks) is not int or ramp_ticks <= 0:
                raise ValueError("ramp_ticks must be a positive integer")
            idx = [self.joints[n]["q"] for n in targets]
            self._ramp = dict(q=idx, start=[float(self.targets[j]) for j in idx], goal=[float(v) for v in targets.values()],
                              t0=int(self.tick), n=int(ramp_ticks))
            return
        for name, value in targets.items():
            self.targets[self.joints[name]["q"]] = value
            if "finger" in name:
                for key in [k for k in self.grasp_latched if k[1] == name]:
                    del self.grasp_latched[key]

    # ------------------------------------------------------------------ stepping
    def step(self):
        """One physics tick (rigid_dt)."""
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            if self._ramp is not None:
                r = self._ramp
                u = min(1., (self.tick + 1 - r["t0"])/r["n"])
                blend = u**3*(10 - 15*u + 6*u*u)
                for j, a, b in zip(r["q"], r["start"], r["goal"]):
                    self.targets[j] = a + blend*(b - a)
                if u >= 1.:
                    self._ramp = None
            cur = self.control.joint_target_q.numpy()
            for name, spec in self.joints.items():
                j = spec["q"]
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
                # The Franka Hand's grasp: a speed-controlled drive whose push is limited to the commanded force. Each
                # finger follows a reference slewed towards its target at the hand's 50 mm/s through its own servo
                # spring (kp), the push capped at `force`: in free air the finger moves exactly as in position mode;
                # blocked, its push builds up (kp x lag) to `force` and stays there; the reference never runs further
                # ahead than the capped push (no wind-up), so a finger that loses its object does not snap shut. A
                # switch to a constant force once the finger lags 0.5 mm -- the servo's ordinary tracking lag at full
                # speed (kd v / kp = 0.57 mm) -- starts the squeeze in free air and threw the fingers at the item at
                # 0.8 m/s, 16x the hand's speed (collision reflexes at grasps)
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
                    cur[j] = q_now[j]                      # the servo's own spring off; its damping (-kd qd) stays
                    if abs(push) >= .999*force:
                        self.grasp_latched[key] = True     # at the force limit: squeezing
                    else:
                        self.grasp_latched.pop(key, None)
            self.control.joint_target_q.assign(cur)
            if self.g1 and self._g1_ff is not None:
                self._g1_gravity_feedforward(jf, cur)
            self.control.joint_f.assign(jf)
            self.state.clear_forces()
            wrench = self.body_wrench
            if self.fixture is not None:
                wrench = self.fixture(self) if callable(self.fixture) else wrench
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            self.solver.step(self.state, self.next_state, self.control, self.contacts, c.rigid_dt)
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            if not self.g1:
                # FR3/Panda protective stops (libfranka). The G1's low-level joint interface has neither: its joint
                # servos saturate at the URDF efforts and its targets are slewed at the URDF motor speeds (above)
                self._velocity_reflex()
                self._torque_reflex()
            q = self.state.joint_q.numpy()
            qd = self.state.body_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(self.state.body_q.numpy()).all() or not np.isfinite(qd).all():
                raise FloatingPointError(f"Non-finite state at tick {self.tick}")
            # an object thrown faster than anything in this room can move is a numerical failure, not an outcome:
            # stop like a non-finite state (a lip balm left a phantom hold at 631 m/s)
            v, om = np.abs(qd[self._objects, :3]).max(initial=0.), np.abs(qd[self._objects, 3:]).max(initial=0.)
            if v > P.MAX_OBJECT_SPEED_M_S or om > P.MAX_OBJECT_SPIN_RAD_S:
                k = int(self._objects[int(np.argmax(np.abs(qd[self._objects]).max(axis=1)))])
                raise FloatingPointError(f"Unphysical speed at tick {self.tick}: {self.model.body_label[k]} "
                                         f"{np.linalg.norm(qd[k, :3]):.1f} m/s, {np.linalg.norm(qd[k, 3:]):.0f} rad/s")
            # robot links are outside that bound, but a Franka Hand's two fingers are one mechanism: fingers parted
            # further than any impact can part them are a solver failure too
            for arm, a, b in self._finger_pairs:
                if abs(float(q[a]) - float(q[b])) > P.MAX_FINGER_COUPLING_M:
                    raise FloatingPointError(f"Unphysical finger state at tick {self.tick}: the {arm} Franka Hand's coupled "
                                             f"fingers at {q[a]*1e3:.1f} and {q[b]*1e3:.1f} mm")
            t = self.sim_time
            self.trigger.update(float(q[self.trigger_q]), t)
            for k, qi in self.key_q.items():
                self.key_switch[k].update(float(q[qi]), t)
            self._seat(t)
            self.timing["rigid_s"] += time.perf_counter() - t0
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _g1_gravity_feedforward(self, jf, cur):
        """Gravity torque of the G1 waist and arm joints at their commanded positions (rest of the robot measured),
        refreshed every 5 ms (Unitree's teleoperation stack streams it at 250 Hz), clipped to the URDF effort limits."""
        ff = self._g1_ff
        if self.tick % ff["every"] == 0:
            m, d, mujoco = self.solver.mj_model, ff["data"], self._mujoco
            d.qpos[:] = self.solver.mj_data.qpos
            d.qvel[:] = 0.
            d.qpos[ff["mj_q"]] = cur[ff["q"]]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            out = np.zeros(m.nv)
            mujoco.mj_rne(m, d, 0, out)
            ff["tau"] = np.clip(out[ff["mj_dof"]], -ff["limit"], ff["limit"])
        jf[ff["dof"]] += ff["tau"]

    def _new_stop(self, arm):
        """A reflex fired on `arm` this tick; True if it starts a new stop episode (no reflex on that arm in the previous 50
        ticks), which is then recorded once. The last firing tick is part of the checkpoint: comparing with the last
        record's tick instead dropped every stop after a restore (a restore rewinds the tick, and a deterministic replay
        fires at the same tick as the recorded one; E17)."""
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
                for name, spec in self.joints.items():
                    if name.startswith(arm + "/") and "finger" not in name:
                        self.targets[spec["q"]] = float(q[spec["q"]])
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, joint=hit[0], speed_rad_s=hit[1], limit_rad_s=hit[2]))

    def _torque_reflex(self):
        """FR3/Panda collision reflex (libfranka Robot::setCollisionBehavior, reference setting of the libfranka
        examples, P.REFLEX_TORQUE_NM): an arm whose external joint torque exceeds the upper threshold on any joint
        stops (its targets freeze at the current configuration) and the event is logged. External torque = the
        contact forces on the robot mapped to its joints (J^T f over contact rows only; joint limits, the gripper's
        own squeeze and gravity are not external). 20 N m while the arm's targets are still slewing (accelerating
        phase), 10 N m once they have arrived (nominal phase)."""
        m, d = self.solver.mj_model, self.solver.mj_data
        mujoco = self._mujoco
        if not d.nefc:
            return
        f = np.array(d.efc_force[:d.nefc])
        f[np.asarray(d.efc_type[:d.nefc]) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
        if not f.any():
            return
        tau = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, tau, f)
        cur = self.control.joint_target_q.numpy()
        q = self.state.joint_q.numpy()
        for arm, dofs in self._arm_mj_dofs.items():
            ext = np.abs(tau[[dof for _, dof in dofs]])
            moving = any(abs(float(self.targets[self.joints[n]["q"]] - cur[self.joints[n]["q"]])) > 1e-6 for n, _ in dofs)
            limit = P.REFLEX_TORQUE_NM[1 if moving else 0]
            k = int(np.argmax(ext))
            if ext[k] > limit:
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                for n, _ in dofs:
                    self.targets[self.joints[n]["q"]] = float(q[self.joints[n]["q"]])
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(ext[k]), limit_nm=float(limit),
                                              cause=self._reflex_cause(f, dofs[k][1])))

    def _reflex_cause(self, f, dof):
        """The contact contributing most to a reflex joint's external torque: shape labels, force, torque, point."""
        m, d, mujoco = self.solver.mj_model, self.solver.mj_data, self._mujoco
        best = None
        for r in np.flatnonzero(f):
            one = np.zeros(d.nefc); one[r] = f[r]
            t1 = np.zeros(m.nv); mujoco.mj_mulJacTVec(m, d, t1, one)
            if best is None or abs(t1[dof]) > abs(best[0]):
                best = (float(t1[dof]), int(r))
        if best is None:
            return None
        c = d.contact[int(d.efc_id[best[1]])]
        names = [self.model.shape_label[int(self._geom_shape[g])] if self._geom_shape[g] >= 0 else str(g) for g in c.geom]
        return dict(shapes=names, torque_nm=best[0], point=np.round(c.pos, 3).tolist())

    def _seat(self, t):
        """Cradle seat sensor: seat frame within tolerance of the cup and >= 2 scanner-cup contacts for 0.2 s."""
        from scipy.spatial.transform import Rotation
        cr = self.scene.cradle
        pos, R = self.body_pose("scanner")
        hc, hq = self.scene.handle_frame()
        bottom = pos + R @ (hc + Rotation.from_quat(hq).apply((0., 0., -self.scene.HANDLE_HALF[2])))
        axis_now = R @ Rotation.from_quat(hq).apply((0., 0., 1.))
        axis_seat = Rotation.from_quat(cr["handle_quat"]).apply((0., 0., 1.))
        ang = math.degrees(math.acos(max(-1., min(1., float(np.dot(axis_now, axis_seat))))))
        near = np.linalg.norm(bottom - np.asarray(cr["seat"])) <= P.SEAT_TOL_M and ang <= P.SEAT_TOL_DEG
        if near:
            if self.tick % 10 == 0 or self._cup_ok is None:          # contact count every 10 ms (a hall sensor's rate)
                self._cup_ok = self._cup_contacts() >= 2
            near = self._cup_ok
        else:
            self._cup_ok = None
        if near:
            if self._seat_since is None:
                self._seat_since = t
            if t - self._seat_since >= P.SEAT_HOLD_S:
                self.seated = True
        else:
            self._seat_since = None
            self.seated = False

    def _cup_contacts(self):
        """Active (non-excluded) contacts between the scanner body and the cradle cup, from MuJoCo's contact list."""
        d = self.solver.mj_data
        if d.ncon == 0:
            return 0
        geom = np.asarray(d.contact.geom)
        active = np.asarray(d.contact.exclude) == 0
        shapes = np.where((geom >= 0) & (geom < len(self._geom_shape)), self._geom_shape[np.clip(geom, 0, len(self._geom_shape) - 1)], -1)
        sb = np.asarray(self.scene.b.shape_body)
        bodies = np.where(shapes >= 0, sb[np.clip(shapes, 0, len(sb) - 1)], -1)
        cup = np.isin(shapes, list(self._cup_shapes()))
        scanner = bodies == self.scene.bodies["scanner"]
        pair = (scanner[:, 0] & cup[:, 1]) | (scanner[:, 1] & cup[:, 0])
        return int((pair & active).sum())

    def _cup_shapes(self):
        if not hasattr(self, "_cup"):
            self._cup = {i for i, lab in enumerate(self.scene.b.shape_label) if lab.startswith("cradle_")}
        return self._cup

    def sync_solver(self):
        """Push the Newton state into MuJoCo's data (after a direct write to state.joint_q/qd). With
        solver_update_data_interval = 0 MuJoCo carries its own float64 state between steps and this is the only way a
        write reaches it; with 1 (Newton's default) it also happens at every step."""
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    # ------------------------------------------------------------------ queries
    def body_pose(self, name_or_index):
        from scipy.spatial.transform import Rotation
        i = self.scene.bodies[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
        q = self.state.body_q.numpy()[i]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def trigger_travel_m(self):
        return float(self.state.joint_q.numpy()[self.trigger_q])*P.TRIGGER_ARM_M

    def contact_forces(self, body_a, body_b=None):
        """Total normal force (N) between two bodies (or body and anything), from MuJoCo's contact solve."""
        m, d = self.solver.mj_model, self.solver.mj_data
        f6 = np.zeros(6)
        tot = 0.
        sb = self.scene.b.shape_body
        for i in range(d.ncon):
            g = d.contact[i].geom
            s = [int(self._geom_shape[x]) if 0 <= x < len(self._geom_shape) else -1 for x in g]
            bodies = [int(sb[x]) if x >= 0 else -1 for x in s]
            if body_a in bodies and (body_b is None or body_b in bodies):
                self._mujoco.mj_contactForce(m, d, i, f6)
                tot += abs(float(f6[0]))
        return tot

    def pair_contact_forces(self, bodies_a, bodies_b):
        """{(a, b): total normal force (N)} over every contact between a body in bodies_a and one in bodies_b, in one pass
        over MuJoCo's contacts (contact_forces makes one pass per pair)."""
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

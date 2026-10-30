"""Rigid world (Newton 1.5 + CPU MuJoCo 3.11) with the hole module and the tool mechanisms in every step.

Robot interface: barcode_scanning/world.py, reused by wine_pouring, with the FR3's own limits on top: arm targets are
followed by a jerk-limited trajectory within Franka's joint velocity, acceleration and jerk limits; the desired joint
torque (servo plus any fed-forward wrench) is capped at the joint's rating, while the robot's own weight and the held
tool's weight (the configured load) are compensated outside it, as Franka specifies the limit on the desired torque
without gravity; a joint-velocity reflex; the libfranka collision reflex (here in the drilling contact setting);
the Franka Hand's grasp (close at 50 mm/s until blocked, then hold the commanded force); hard joint
stops; MuJoCo no-slip. Step order per tick: tools set their drives from their own joint states; the hole module places
its proxies and queues last tick's explicit wrenches (cutting torque, walking friction); the robot's motion generator
and servos advance; MuJoCo steps; caches refresh; the hole module reads the solved contact forces and advances holes;
reflexes and fault checks run.
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization

from . import params as P
from .config import Config
from .scene import Scene
from .holes import HoleModule

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
        t0 = time.perf_counter()
        self.scene = sc = Scene(c, newton, wp)
        b = sc.b
        self.hm = HoleModule(self)
        for d in sc.drillables.values():
            self.hm.add_drillable(d)
        for it in sc.items.values():
            self.hm.add_item(it)
        self.dt = c.rigid_dt
        import mujoco
        self._mujoco = mujoco
        with wp.ScopedDevice(c.device):
            self.model = b.finalize(device=c.device)
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (c.contact_timeconst, 1.)
            mass = self.model.body_mass.numpy()
            sb = np.asarray(b.shape_body)
            pads = {int(s) for s in sc.pad_shapes}
            imp = self.model.mujoco.geom_solimp.numpy()
            stiff = {int(s_) for s_ in sc.stiff_shapes}
            solids = {int(s_) for s_ in sc.stiff_solids}
            hold_ = {int(s_) for s_ in sc.clamp_shapes} | {int(s_) for s_ in sc.stack_shapes}
            loose = {int(s_) for s_ in sc.loose_shapes}
            for i in range(len(sb)):
                if i in loose:
                    imp[i][:3] = c.contact_solimp                 # the repo's standard contact (params.LOOSE_SOLMIX)
                elif i in stiff:
                    ref[i] = (2*c.rigid_dt, 1.)
                    imp[i][:3] = P.SOLID_SOLIMP if i in solids else P.HOLD_DOWN_SOLIMP if i in hold_ else P.STIFF_SOLIMP
                elif i in pads:
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
                name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if not mjm.jnt_limited[j]:
                    continue
                d0, dmax, mid, power = P.STOP_SOLIMP
                width = 5e-4 if ("finger_joint" in name or mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE) else 1e-3
                mjm.jnt_solref[j] = (2*c.rigid_dt, 1.)
                mjm.jnt_solimp[j] = (d0, dmax, width, mid, power)
            # the hand's fingers are one mechanism: the imported mimic is a joint equality at MuJoCo's default softness
            # until it is given the hardness of the world's other gears (params.FINGER_GEAR_SOLREF)
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
            # the main solve's line search, fine enough to finish the solve (params.SOLVER_LS_ITERATIONS)
            mjm.opt.ls_iterations = int(c.solver_ls_iterations)
            mjm.opt.ls_tolerance = float(c.solver_ls_tolerance)
            # the convex collision routine (cylinders on boxes), fine enough for parts to lie still on their rests
            # (params.SOLVER_CCD_TOLERANCE)
            mjm.opt.ccd_iterations = int(c.solver_ccd_iterations)
            mjm.opt.ccd_tolerance = float(c.solver_ccd_tolerance)
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            bearing = set() if P.HOLD_DOWN_BEARING_ROLLS else {int(s_) for s_ in sc.bearing_shapes}
            no_roll = bearing | (set() if P.HOLD_DOWN_BEARING_ROLLS else {int(s_) for s_ in sc.stack_shapes})
            for g in range(mjm.ngeom):
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in stiff:
                    shape_ = int(self._geom_shape[g])
                    # a clamp's bearing faces outrank the work they bear on: their law, their friction and their
                    # contact dimension rule (params.HOLD_DOWN_BEARING_ROLLS)
                    mjm.geom_priority[g] = 2 if shape_ in bearing else 1
                    if shape_ in no_roll:
                        continue
                    # the higher-priority geom's contact dimension and friction rule a contact, so a round part's
                    # rolling resistance (its rim's condim 6) never reached a metal face: an anchor rolled 1.82 m on
                    # the steel plate. The metal faces carry the same rolling lever arm, and a
                    # torsional coefficient at MuJoCo's floor (no added twist resistance for a clamped part). Not the
                    # load-bearing faces of the clamps and stacks, on which nothing rolls
                    mjm.geom_condim[g] = 6
                    mjm.geom_friction[g][1] = mujoco.mjMINMU
                    mjm.geom_friction[g][2] = P.ROLLING_B_M
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in loose:
                    # a small loose part ranks with the hardened faces and its own law outweighs theirs
                    mjm.geom_priority[g] = 1
                    mjm.geom_solmix[g] = P.LOOSE_SOLMIX
                if g < len(self._geom_shape) and int(self._geom_shape[g]) in pads:
                    # the rubber pads rule every contact they make: they rank above the hardened faces and the loose
                    # parts (priority 1), so their law is not averaged with those faces' laws
                    mjm.geom_priority[g] = 2
                    mjm.geom_condim[g] = 4
                    mjm.geom_friction[g][1] = P.PAD_TORSIONAL_M
                    # the pad's friction rows remove a slip within a tick; its normal stiffness stays the anchored one
                    # (MuJoCo's direct form: stiffness 1/timeconst^2, rate dmax/dt; params.PAD_SLIP_RATE_DT)
                    tc_ = float(mjm.geom_solref[g][0])
                    if tc_ > 0.:
                        mjm.geom_solref[g] = (-1./tc_**2, -P.PAD_SLIP_RATE_DT*float(mjm.geom_solimp[g][1])/c.rigid_dt)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        mjm = self.solver.mj_model
        # Newton dof -> MuJoCo joint (tools bind through it)
        j2d = np.asarray(self.solver.mjc_jnt_to_newton_dof.numpy()).reshape(-1)
        self.mj_joint_of_dof = {int(dof): jm for jm, dof in enumerate(j2d[:mjm.njnt]) if dof >= 0}
        m2n = np.asarray(self.solver.mjc_body_to_newton.numpy())[0]
        self._mj_to_newton = {j: int(n) for j, n in enumerate(m2n)}
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
        self.timing = dict(build_s=0., step_s=0., hole_s=0., tools_s=0.)
        # robot
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
        spinning = set()
        for t in sc.tools.values():
            if hasattr(t, "spindle"):
                spinning.update(t.spindle.bodies)
        self._objects = np.array([bb for bb in range(self.model.body_count) if bb not in robots], dtype=int)
        self._spin_exempt = np.array(sorted(spinning), dtype=int)
        self.sync_solver()
        self.refresh()
        for t in sc.tools.values():
            t.bind(self)
        self._initial_holes()
        self.hm.bind()
        self.hm.pre_step()
        self.body_wrench[:] = 0.
        self.timing["build_s"] = time.perf_counter() - t0

    def _initial_holes(self):
        """Start-state dimples (a punched part) and pre-drilled holes (clean-and-set) as hole records. A punched dimple
        has the depth a strike of the punch leaves in that material; a pre-drilled wall hole carries the dust column
        the dust law leaves after drilling, both authored start states."""
        from .holes import Hole
        from . import materials as M
        hm, sc = self.hm, self.scene
        for name, p_l, n_l, depth in sc.initial_dimples:
            d = hm.drillables[name]
            if depth is None:
                ta = math.tan(P.PUNCH_TIP_HALF_ANGLE)
                depth = (3*P.PUNCH_ENERGY_J/(math.pi*ta*ta*P.HARDNESS_PA[d.material]))**(1/3)
            hm.make_dimple(d, np.asarray(p_l, float), np.asarray(n_l, float), depth)
        for spec in sc.initial_holes:
            d = hm.drillables[spec["drillable"]]
            p_w = np.asarray(spec["face_point"], float)
            q = self.bq[d.body]
            R = Rotation.from_quat(q[3:])
            p0 = R.inv().apply(p_w - q[:3])
            a = R.inv().apply(np.asarray(spec["axis"], float))
            n = R.inv().apply(np.asarray(spec["normal"], float))
            h = Hole(hm.next_id, d.name, p0, a, n, thickness=hm.thickness_along(d, p0, a),
                     stages=[dict(d_mm=spec["d_mm"], s=spec["depth"], start=0., through=False)], seated_axis=a.copy(),
                     tools=["start_state"], created_tick=0)
            A = math.pi*(0.5*spec["d_mm"])**2
            dust = A*spec["dust_mm"]
            h.dust_loose_mm3, h.dust_packed_mm3 = 0.5*dust, 0.5*dust
            h.removed_mm3 = A*spec["depth"]*1e3
            # the dust the drilling made: what stays in the hole is the column above; the rest left the hole (the
            # renderer draws it at the mouth and below it)
            h.dust_produced_mm3 = max(P.DUST_BULKING*h.removed_mm3, dust)
            h.dust_evacuated_mm3 = h.dust_produced_mm3 - dust
            h.entry_spall_d_m = P.SPALL_DIAMETER_FACTOR*spec["d_mm"]*1e-3
            h.events.append(dict(tick=0, event="start_state", depth_mm=round(spec["depth"]*1e3, 2),
                                 dust_mm=round(spec["dust_mm"], 2)))
            hm.next_id += 1
            d.holes.append(h)

    # ------------------------------------------------------------------ robot setup (barcode/wine lineage)
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
        self.grip_applied = {}           # (arm, finger joint) -> force the force mode applied this tick, N (signed)
        self.grip_log = {}               # arm -> largest commanded force, largest and last force both fingers applied (N)
        World.last = self                # the record writer adds this world's grip log to what it writes
        ke = self.model.joint_target_ke.numpy()
        self._finger_kp = {n: float(ke[self.joints[n]["dof"]]) for arm in sc.arm_hw for n in self.gripper_joints(arm)}
        # drilling contact setting: each joint stops above 95 % of its effort limit
        self.reflex_limit = {n: P.REFLEX_EFFORT_FRACTION*float(self.joints[n]["effort_limit"])
                             for n in self.joints if "finger" not in n}
        self.wrench_cmd = {}
        self._tcp_off = {}
        # The arm's motion generator. A target step is followed by a time-optimal jerk-limited trajectory (Ruckig, the
        # generator of the FR3 control library franky) within Franka's joint velocity, acceleration and jerk limits,
        # the necessary conditions on any commanded FR3 trajectory. An arm's joints are
        # synchronized in time (Ruckig's default, as a joint move in franky or MoveIt): a target step moves the arm
        # along the straight joint-space segment. Run independently, small steps ended at different times per joint
        # under the acceleration limit, each hand left its path, and two hands carrying one drill fought until the
        # wrist's collision reflex tripped at every correction.
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
        """The arm controllers' internal state (checkpoints): commanded trajectory, feed-forward, configured load."""
        out = {}
        for arm, g in self._arm.items():
            inp = g["inp"]
            out[arm] = dict(p=list(inp.current_position), v=list(inp.current_velocity), a=list(inp.current_acceleration),
                            target=list(inp.target_position), moving=g["moving"], ff=g["ff"].copy(),
                            load=g["load"].copy(), load_want=g["load_want"].copy(),
                            range=self.solver.mj_model.jnt_actfrcrange[g["jnt"]].copy())
        return dict(arms=out, wrench_cmd=dict(self.wrench_cmd), prev_cvel=dict(self._prev_cvel or {}))

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

    def geometry_state(self):
        """Run-time edits of the collision model (a broken bit's shortened colliders) and the items they move."""
        m = self.solver.mj_model
        return dict(size=m.geom_size.copy(), pos=m.geom_pos.copy(), contype=m.geom_contype.copy(),
                    conaffinity=m.geom_conaffinity.copy(),
                    items={n: (it.tip.copy(), it.length, it.stickout) for n, it in self.hm.items.items()})

    def set_geometry_state(self, st):
        m = self.solver.mj_model
        m.geom_size[:], m.geom_pos[:] = st["size"], st["pos"]
        m.geom_contype[:], m.geom_conaffinity[:] = st["contype"], st["conaffinity"]
        for n, (tip, length, stickout) in st["items"].items():
            it = self.hm.items[n]
            it.tip, it.length, it.stickout = tip.copy(), length, stickout

    def _hold_command(self, arm, q):
        """The arm's commanded trajectory stops where the arm is: position q, zero velocity and acceleration (start
        state, the reflexes)."""
        g = self._arm[arm]
        g["inp"].current_position = [float(v) for v in q]
        g["inp"].current_velocity = [0.]*len(q)
        g["inp"].current_acceleration = [0.]*len(q)
        g["moving"] = True
        self.tq[g["q"]] = q
        self.tqd[g["dof"]] = 0.

    def _tcp_offset(self, arm):
        """The TCP (the rig's `arm/tcp` frame, between the fingertips) in the frame of the body that carries the hand. The
        hand is merged into fr3_link7, whose origin lies one flange length behind the flange: adding only the 0.1034 m
        flange-to-TCP offset to it had put cartesian_wrench 107 mm behind the fingertips (robot wall demonstration)."""
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
        """Force the hand's force mode applies now, N per finger: the smaller of the two fingers' pushes (0 when the
        force mode is off). The commanded value is grip_force[arm]."""
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

    # ------------------------------------------------------------------ tool / module interface
    def refresh(self):
        self.bq = self.state.body_q.numpy().astype(np.float64)
        self.bqd = self.state.body_qd.numpy().astype(np.float64)
        self.jq = self.state.joint_q.numpy().astype(np.float64)
        self.jqd = self.state.joint_qd.numpy().astype(np.float64)

    def body_pose(self, body):
        q = self.bq[body]
        return q[:3].copy(), Rotation.from_quat(q[3:]).as_matrix()

    def add_force(self, body, F, p_w):
        q = self.bq[body]
        com = q[:3] + Rotation.from_quat(q[3:]).apply(self.body_com[body])
        self.body_wrench[body, :3] += F
        self.body_wrench[body, 3:] += np.cross(np.asarray(p_w) - com, F)

    def place_free(self, body, p, quat_xyzw):
        """ASSISTED placement (probes/demos only): move a free root body to a world pose at rest, keeping the poses of
        the links welded or jointed to it relative to it; MuJoCo's state follows."""
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
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.sync_solver()
        self.refresh()

    def add_torque(self, body, tau):
        self.body_wrench[body, 3:] += tau

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
            t1 = time.perf_counter()
            self.hm.pre_step()
            t2 = time.perf_counter()
            if self.rig:
                self._servo(c)
            wrench = self.body_wrench
            if self.fixture is not None:
                # ASSISTED fixture (probes/demos): may add body wrenches and joint forces (self.jf) for this tick
                wrench = self.fixture(self, wrench)
            self.control.joint_target_q.assign(self.tq.astype(np.float32))
            self.control.joint_target_qd.assign(self.tqd)
            self.control.joint_f.assign(self.jf)
            self.state.clear_forces()
            self.state.body_f.assign(np.asarray(wrench, np.float32))
            self.solver.step(self.state, self.next_state, self.control, None, c.rigid_dt)
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            self.body_wrench = np.zeros_like(self.body_wrench)
            self.refresh()
            t3 = time.perf_counter()
            self.hm.post_step()
            t4 = time.perf_counter()
            for t in self.scene.tools.values():
                t.post_step(self)
            if self.rig:
                self._velocity_reflex()
                self._torque_reflex()
            self._check_faults()
            t5 = time.perf_counter()
            self.timing["step_s"] += t5 - t0
            self.timing["hole_s"] += (t2 - t1) + (t4 - t3)
            self.timing["tools_s"] += (t1 - t0) + (t5 - t4)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def set_wrench(self, arm, force, moment, ticks):
        """Cartesian wrench at the TCP (world frame), fed forward as joint torques J^T (F, M) for `ticks` ticks. It is
        part of the desired joint torque: servo and feed-forward together are capped at each joint's rating, and the
        feed-forward changes at most at Franka's torque rate (1000 N m/s), rising and falling."""
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        F, Mo = np.asarray(force, float), np.asarray(moment, float)
        if F.shape != (3,) or Mo.shape != (3,) or not (np.isfinite(F).all() and np.isfinite(Mo).all()):
            raise ValueError("force_n and moment_nm must be finite 3-vectors")
        if np.linalg.norm(F) > P.WRENCH_MAX_N or np.linalg.norm(Mo) > P.WRENCH_MAX_NM:
            raise ValueError(f"Wrench beyond the FR3 hand's rated range (|F| <= {P.WRENCH_MAX_N} N, "
                             f"|M| <= {P.WRENCH_MAX_NM} N m)")
        if self.sim_time < self._reflex_until.get(arm, -1.):
            self.refused_targets += 1
            return
        self.wrench_cmd[arm] = (F, Mo, self.tick + int(ticks))

    def _arm_torques(self):
        """Per arm joint: the desired torque is the servo's (MuJoCo's position/velocity actuator) plus the wrench
        feed-forward, both inside one cap of the joint's rating. MuJoCo clamps the actuator to jnt_actfrcrange, set here
        each tick to [-lim - ff, lim - ff] with the feed-forward ff itself within +-lim and applied as a joint force,
        so the sum stays within +-lim. Outside the cap, as the FR3 compensates gravity of its
        model and its configured load beside the desired torque: MuJoCo's passive compensation of the arm's own links
        and the held tool's weight (_payload_torques)."""
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
                # The hand presses with the commanded force wherever a finger is stopped short of its commanded
                # position (Franka's grasp: travel towards a width, hold the force on whatever is in between). The
                # reference therefore runs on past the commanded position by the travel the full force needs, in
                # proportion to how far short the finger is stopped and in full from GRIP_FORCE_BAND_M on; a finger
                # that arrives is held at the commanded position. (A reference that stops AT the commanded
                # position, which the gripper action sets to the closed end, gives a part narrower than 2 force/kp =
                # 20 mm kp x half its width instead of the force: 12 N on an M8 bolt, 24 N on the centre punch
                # for 35 N commanded.)
                short = float(self.targets[j]) - float(q_now[j])
                goal = float(self.targets[j]) + reach*float(np.clip(short/P.GRIP_FORCE_BAND_M, -1., 1.))
                ref += float(np.clip(goal - ref, -step, step))
                ref = float(np.clip(ref, q_now[j] - reach, q_now[j] + reach))
                self._grip_ref[key] = ref
                push = kp*(ref - float(q_now[j]))
                self.grip_applied[key] = push
                st_ = self.grip_log.setdefault(arm, dict(commanded_n=0., applied_max_n=0., applied_last_n=0.))
                st_["commanded_n"] = max(st_["commanded_n"], float(force))
                self.jf[dof] += push
                self.tq[j] = q_now[j]
                if abs(push) >= .999*force:
                    self.grasp_latched[key] = True
                else:
                    self.grasp_latched.pop(key, None)
            both = self.grip_applied_n(arm)                    # the smaller of the two fingers' pushes this tick
            if both > 0.:
                st_ = self.grip_log[arm]
                st_["applied_max_n"], st_["applied_last_n"] = max(st_["applied_max_n"], both), both

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
        """A reflex stops the arm where it is (targets and commanded trajectory) and ends its wrench feed-forward, as a
        reflex aborts the FR3's control loop; the feed-forward then falls at the torque rate."""
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
        """FR3/Panda collision reflex in the drilling contact setting: contact forces on the robot mapped
        to its joints (J^T f over contact rows only), less the held tools' weight and inertia (setLoad), filtered over
        REFLEX_FILTER_S; a joint above 95 % of its effort limit stops its arm."""
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
            lim = np.array([self.reflex_limit[n] for n, _ in dofs])
            ratio = ext/lim
            k = int(np.argmax(ratio))
            if ratio[k] > 1.:
                self._reflex_until[arm] = self.sim_time + P.REFLEX_RECOVERY_S
                self._stop_arm(arm)
                if self._new_stop(arm):
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(ext[k]), limit_nm=float(lim[k])))

    def _payload_torques(self):
        """Each arm's configured load, as the FR3 controller models one (setLoad; wine_pouring): every carried tool the
        hand pinches between both fingers, counted once per tool. Only a body that can be carried is a load: one whose
        kinematic root is free. A pinched mechanism attached to the world (the vise crank, the press's feed handle) is
        not; counting its root's subtree read 0.5 N of real knob contact as 17.7 N m on joint 3 of an arm turning the
        vise crank, the 9.3 kg vise subtracted as if held (robot press demonstration). A tool held by two hands is
        shared as statics shares it (take-drill): vertical forces at the two TCPs whose resultant passes through the
        tool's centre of mass, the moment they cannot balance split equally. Returns the load's joint torques (weight
        and inertia) that the collision reflex subtracts, and sets the gravity torques each arm compensates next tick
        beside its desired torque, as the FR3 compensates gravity of its configured load."""
        m, d, mujoco = self.solver.mj_model, self.solver.mj_data, self._mujoco
        held = {}
        if d.ncon:
            bb = m.geom_bodyid[np.asarray(d.contact.geom[:d.ncon])]
            for arm, fingers in self._arm_fingers.items():
                for col in (0, 1):
                    sel = np.isin(bb[:, col], fingers) & ~np.isin(bb[:, 1 - col], fingers)
                    for fb, other in zip(bb[sel, col], bb[sel, 1 - col]):
                        # by the touched body's ROOT: a tool is one articulated thing, and a hand that pinches a grip
                        # with one pad on its trigger (the trigger is a body of its own) holds the tool. Keyed by the
                        # touched body, that hand held nothing: no configured load for the one hand that drills with
                        # the rotary hammer, and the whole 3.23 kg charged to the other hand while both carried it
                        # (880 of 880 samples without a load while drilling)
                        held.setdefault((arm, int(m.body_rootid[int(other)])), {}).setdefault(int(fb), int(other))
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        tools = {}
        for (arm, root), fs in sorted(held.items()):
            if len(fs) < 2:
                continue
            jr = int(m.body_jntadr[root])
            if root == 0 or jr < 0 or int(m.jnt_type[jr]) != free:
                continue
            body = next((b_ for b_ in sorted(fs.values()) if self._mj_to_newton.get(b_, -1) >= 0), None)
            if body is None:
                continue
            tools.setdefault(root, {}).setdefault(arm, body)
        out = {}
        g = np.array(m.opt.gravity)
        want = {arm: np.zeros(len(gr["names"])) for arm, gr in self._arm.items()}
        cvel = {}
        carried = {}
        for root, arms in tools.items():
            # the whole articulated tool (housing and its moving parts) is the payload
            mass = float(m.body_subtreemass[root])
            c = d.subtree_com[root].copy()
            nb = self._mj_to_newton[next(iter(arms.values()))]
            qd = self.bqd[nb]
            acc = (qd[:3] - self._prev_cvel[nb])/self.config.rigid_dt \
                if self._prev_cvel is not None and nb in self._prev_cvel else np.zeros(3)
            cvel[nb] = qd[:3].copy()
            names = list(arms)
            if len(names) == 1:
                pts, shares = [c], [1.]
            else:
                pts = [d.xpos[self._arm_hand[a]] + d.xmat[self._arm_hand[a]].reshape(3, 3) @ self._tcp_offset(a)
                       for a in names]
                d0, d1 = (pts[0] - c)[:2], (pts[1] - c)[:2]
                u = d0 - d1
                s0 = float(np.clip(-(d1 @ u)/max(u @ u, 1e-12), 0., 1.))
                shares = [s0, 1. - s0]
            for a, sh in zip(names, shares):
                carried[a] = carried.get(a, 0.) + sh*mass
            for F_tot, sink in ((mass*(g - np.clip(acc, -50., 50.)), "reflex"), (-mass*g, "gravity")):
                Fs = [sh*F_tot for sh in shares]
                M = -sum(np.cross(p - c, f) for p, f in zip(pts, Fs))/len(names)
                for a, p, f in zip(names, pts, Fs):
                    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                    mujoco.mj_jac(m, d, jacp, jacr, p, self._arm_hand[a])
                    t = (jacp.T @ f + jacr.T @ M)[[dof for _, dof in self._arm_mj_dofs[a]]]
                    if sink == "reflex":
                        out[a] = out.get(a, 0.) + t
                    else:
                        want[a] = want[a] + t
        for arm, gr in self._arm.items():
            gr["load_want"] = want[arm]
        self._prev_cvel = cvel
        # rated payload: a diagnostic, not a refusal (params.FRANKA_HAND_KG)
        rated = P.FR3_PAYLOAD_KG - P.FRANKA_HAND_KG
        over = {a for a, kg in carried.items() if kg > rated}
        # seconds each arm has carried more than its rating (the evaluator's constraint reads it), and one event per
        # episode: an episode ends once the arm has been under its rating for 0.5 s (while two hands shared the hammer
        # the estimate switched ten times a second and logged 125 events in 11 s)
        acc = self.__dict__.setdefault("over_payload_s", {})
        last = self.__dict__.setdefault("_over_payload_last", {})
        for a in sorted(over):
            acc[a] = acc.get(a, 0.) + self.dt
            if self.sim_time - last.get(a, -1e9) > 0.5:
                self.log_event(dict(type="payload_over_rating", tick=int(self.tick), arm=a, load_kg=round(carried[a], 3),
                                    rated_kg=round(rated, 3)))
            last[a] = self.sim_time
        self._over_payload = over
        self.load_kg = {a: round(kg, 3) for a, kg in carried.items()}
        holders = {int(r): sorted(a) for r, a in tools.items()}                  # diagnostics: tool root -> arms
        if holders != getattr(self, "load_holders", {}):
            self.load_changes = getattr(self, "load_changes", 0) + 1
        self.load_holders = holders
        return out

    def on_bit_broken(self, it):
        """A bit that breaks loses its end: the piece the hole module leaves in the hole (it.state["stub_m"], else
        2.5 d behind the tip, holes._break) is gone from the bit too. The point stops colliding and the body cylinder
        is shortened at the tip end, so the bit in the chuck is as long as what is left of it; its tip, length and
        stickout move back with it."""
        mujoco, m = self._mujoco, self.solver.mj_model
        s = float(it.state.get("stub_m") or 2.5e-3*it.d_mm)
        if s <= 0.:
            return
        mjb = next((j for j, n in self._mj_to_newton.items() if n == it.body), None)
        a = np.asarray(it.axis, float)/np.linalg.norm(it.axis)
        labels = self.model.shape_label
        cut_cyl = 0.
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != mjb or g >= len(self._geom_shape):
                continue
            label = str(labels[int(self._geom_shape[g])])
            if label.endswith("_point"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
            elif label.endswith("_body") and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                r = float(m.geom_size[g][0])
                cone = float(np.linalg.norm(it.tip - (m.geom_pos[g] + a*m.geom_size[g][1]))) if r > 0 else 0.
                cut_cyl = min(max(0., s - cone), 2.*float(m.geom_size[g][1]) - 1e-3)
                m.geom_size[g][1] -= 0.5*cut_cyl
                m.geom_pos[g] -= a*0.5*cut_cyl
        it.tip = it.tip - a*s
        it.length = max(1e-3, it.length - s)
        it.stickout = max(1e-3, it.stickout - s)
        it.state["shortened_m"] = round(s, 5)
        self.log_event(dict(type="bit_shortened", item=it.name, tick=int(self.tick), stub_m=round(s, 5),
                            cylinder_cut_m=round(cut_cyl, 5)))

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

    @property
    def sim_time(self):
        return self.tick*self.config.rigid_dt

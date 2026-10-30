"""Coupled world of the robotic GMAW cell: UR10e + MIG gun and FR3 + Franka Hand (Newton 1.5 -> MuJoCo on the CPU,
1 ms tick) and the weld process (welding/process.py: thermal solver on the GPU, CV arc, pool) on the coupon.

Coupling, every rigid tick:
  robot -> process  the contact tip's position and the wire axis come from the UR10e's last link (MuJoCo state) and
                    the gun geometry (torch.py), expressed in the weld frame carried by the base plate body; the
                    tip's velocity in that frame is the travel speed. The welding circuit is closed only if the metal
                    the arc reaches is connected to the work clamp through touching steel (MuJoCo contacts) or a bond.
  process -> rigid  only the bond (bonds.py: a MuJoCo weld equality switched on by fusion, removed by overload).
                    The thermal solver exerts no forces; deposited mass (~25 g on 3 kg) is not added (stated).
The source is commanded through `tool_io` (the robot's I/O to the welding power source); nothing else switches it.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import layout as LY
from . import params as PR
from . import torch as TO

BOND_EVERY = 50               # ticks between bond capacity updates (50 ms)
GRID_EVERY = 10               # ticks between checks that the thermal grid's bracket is where the rigid bracket is
CIRCUIT_EVERY = 10            # ticks between contact-graph updates of the welding circuit
# the arms' protective-stop and motion-generator constants live in the parameter ledger with their provenance
# (params.py, "robots"); the names below are kept for this module's readers
UR_FORCE_LIMIT_N = PR.v(PR.UR_FORCE_LIMIT)        # N, tool and elbow force limit (UR10e default safety configuration)
UR_TOOL_LINKS = ("wrist_1_link", "wrist_2_link", "wrist_3_link")
UR_ELBOW_LINKS = ("upper_arm_link", "forearm_link")
UR_BACK_OFF_TRACE_S = PR.v(PR.UR_BACK_OFF_TRACE)  # s of its own path a UR arm retraces after a force-limit stop
UR_ACCEL = PR.v(PR.UR_ACCEL)                      # rad/s^2, UR10e motion-generator acceleration
VELOCITY_STOP = PR.v(PR.VELOCITY_STOP)            # x rated joint speed that trips a protective stop
NOSLIP_ITERATIONS = 30        # MuJoCo no-slip post-solver iterations (static friction without creep; E19)
NOSLIP_TOLERANCE = 0.         # the no-slip pass runs all NOSLIP_ITERATIONS every step [numerical]. MuJoCo's default (1e-6)
                              # ends it once an iteration's scaled improvement falls below that: on the standing G1 it
                              # stopped after 4 of the 30 (mjData.solver_niter 5 = 1 main + 4) and left the soft friction's
                              # creep -- the whole robot slid sideways at a steady 7 um/s (0.88 mm in 120 s) under the
                              # feet's 21 N internal load at 168-187 N (0.13 of mu N). Run to the
                              # end the feet hold: 2.4 um net in 119 s, 0.02-0.6 um/s in 10 s windows with no steady
                              # direction (rigid cell); the default cell's start states unchanged (E0: loose bracket
                              # 0.07 um, tacked bracket 3.45 um in 20 s, its shrinkage)
IMPRATIO = 10.                # MuJoCo's ratio of frictional to normal constraint impedance (elliptic cones): how hard
                              # friction is against the normal force [numerical]. 100 cut the standing G1's foot creep
                              # from 9.4 to 1.6 um/s at the torch, but the default cell's loose-bracket start state then
                              # never came to rest (1-10 mm per second for 30 s): global, so kept at 10
PLAN_SPEED_FRACTION = PR.v(PR.PLAN_SPEED_FRACTION)  # the motion generator's planned share of the rated joint speed
FR3_ACCEL = PR.v(PR.FR3_ACCEL)                    # rad/s^2, FR3 motion-generator acceleration
PANDA_ACCEL = PR.v(PR.PANDA_ACCEL)                # rad/s^2, Panda joints 1-7 (libfranka kMaxJointAcceleration)
G1_FF_EVERY_S = .001          # s: the G1's gravity feed-forward refresh: every tick [numerical]. At 5 ms ("Unitree's
                              # teleoperation stack streams at 250 Hz") a hold is a sampled feedback with up to 5 ms of
                              # delay, since Unitree computes it from the target configuration alone while this one
                              # reads the measured floating base: with the feet held by converged static friction
                              # (NOSLIP_TOLERANCE) and no inactive contacts on the G1 (_g1_contacts), the robot settles
                              # from its import pose into a sustained vibration (pelvis 85 um, feet 120 um peak to peak;
                              # 22.7 Hz as sampled at 50 Hz) that never passes the start state's rest window; refreshed
                              # every tick (or switched off) it settles to rest
REFLEX_TORQUE_NM = PR.v(PR.REFLEX_TORQUE)        # N m, FR3/Panda collision thresholds (nominal, accelerating)
REFLEX_RECOVERY_S = PR.v(PR.REFLEX_RECOVERY)      # s a stop refuses the arm's commands, then it is unlocked
BACK_OFF_S, BACK_OFF_MARGIN = PR.v(PR.STOP_BACK_OFF)  # s window after the unlock, margin above the held load
FRANKA_HAND_FINGER_SPEED = .05
CONTACT_TIMECONST = .004       # s, MuJoCo contact time constant of the cell's rigid contacts (see World.__init__)
MAGNET_RANGE = 1e-3           # m: the ground clamp's magnet pulls within this gap of the table top


@dataclass
class Config:
    preset: str = "steel-cell"
    task: str = "fillet-2f"
    seed: int = 0
    robot: str = None
    device: str = "cuda:0"
    rigid_dt: float = 1e-3
    solver_iterations: int = 50
    contact_capacity: int = 2000
    process: bool = True              # False: rigid cell only (layout and reach probes)
    gravity_position: str = "2F"      # 2F: horizontal fillet (z up in the weld frame)
    draft_m_s: float = 0.             # ambient cross draft at the joint (task seed may set it)
    coupon_offset: tuple = (0., 0., 0.)   # (dx m, dy m, yaw deg) of the coupon on the table (variants.py seeds)
    work_clamp_on_table: bool = True  # False: the ground clamp starts on the power source's top (no welding circuit)


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


class World:
    def __init__(self, config=None, coupon=None):
        self.config = c = config or Config()
        newton, wp = self.newton, self.wp = require_engine()
        from .process import tee_coupon, WeldProcess
        self.coupon = coupon or tee_coupon()
        from .scene import Scene
        self.scene = Scene(c, self.coupon, newton, wp)
        b = self.scene.b
        with wp.ScopedDevice("cpu"):
            self.model = b.finalize(device="cpu")
            # contact time constants (MuJoCo raw solref): steel parts, table, clamps and gun 4 ms -- the stiffest
            # stable at the 1 ms step (MuJoCo: >= 2 dt). At MuJoCo's default 20 ms the 400 N clamp pads sank ~8 mm into
            # the plate and the plate ~2 mm into the table, and their ringing walked the loose bracket 2 mm sideways
            # (settle probe). Franka finger shapes keep the pads' 5 ms scaled by sqrt(link mass) (as in
            # label_application and filleting).
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (CONTACT_TIMECONST, 1.)
            mass = self.model.body_mass.numpy()
            for i, body in enumerate(self.model.shape_body.numpy()):
                if body >= 0 and "finger" in self.model.body_label[body]:
                    ref[i] = (.005*math.sqrt(float(mass[body])/.0291), 1.)
            self.model.mujoco.solref.assign(ref)
            if any(hw["model"] == "g1" for hw in self.scene.arm_hw.values()):
                self._g1_contacts()
        self.rig = self.scene.rig_pending.bind(self.model)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=c.contact_capacity, broad_phase="sap",
                                                 reduce_contacts=True)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                  enable_multiccd=True, integrator="implicitfast", cone="elliptic",
                                                  solver="newton", iterations=c.solver_iterations, ls_iterations=20,
                                                  impratio=IMPRATIO, nconmax=c.contact_capacity,
                                                  njmax=c.contact_capacity*3)
        # MuJoCo's midphase culls candidate geom pairs with per-body bounding-volume trees built when the model was
        # compiled; Newton moves geoms afterwards without rebuilding them, and the Franka fingers' trees end 4 mm short
        # of their fingertip pads: every contact with a fingertip was dropped and the hand closed through the bracket
        # (measured: pads 4.2 mm into the bracket by mj_geomDistance, no contact; with the midphase off, the
        # contacts appear). The midphase is a culling optimisation only: off, every body pair goes to the exact
        # narrowphase.
        mjm = self.solver.mj_model
        mjm.opt.disableflags = int(mjm.opt.disableflags) | int(self.solver._mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        # MuJoCo's friction is a soft, velocity-level constraint: under a sustained tangential load far below mu*N it
        # creeps (a 7 N bracket slid 6.4 mm out of a 60 N, mu 0.8 grip in 2.5 s, E19). Its no-slip post-solver makes
        # static friction hold (slip -0.31 mm with 30 iterations, 0.94 mm with 10) when it runs them all
        # (NOSLIP_TOLERANCE)
        mjm.opt.noslip_iterations = NOSLIP_ITERATIONS
        mjm.opt.noslip_tolerance = NOSLIP_TOLERANCE
        # a Franka Hand drives both fingers from one motor through a rigid mechanism: the imported model couples finger 2
        # to finger 1 by a MuJoCo joint equality at MuJoCo's default soft time constant (20 ms), which on the Panda's 14 g
        # fingers let them part by millimetres under the grip force; one finger then pushed the loose bracket alone and
        # the arm read 12.8 N m at joint 3 (a libfranka reflex). The coupling is held at FRANKA_FINGER_COUPLING_TAU
        mj_ = self.solver._mujoco
        for k in range(int(mjm.neq)):
            if int(mjm.eq_type[k]) != int(mj_.mjtEq.mjEQ_JOINT):
                continue
            names = [mj_.mj_id2name(mjm, mj_.mjtObj.mjOBJ_JOINT, int(j)) or "" for j in (mjm.eq_obj1id[k], mjm.eq_obj2id[k])
                     if int(j) >= 0]
            if names and all("finger" in n for n in names):
                mjm.eq_solref[k] = (PR.v(PR.FRANKA_FINGER_COUPLING_TAU), 1.)
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        self.arms = tuple(self.rig.instances)
        self.torch_arm = self.scene.torch_arm
        # the G1 (welding/humanoid.py): one instance with both arms; its left Dex3 hand handles the part, commanded
        # joint by joint (no Franka gripper force mode); no libfranka reflex
        self.g1 = any(hw["model"] == "g1" for hw in self.scene.arm_hw.values())
        self.hand_arms = ("left",) if self.g1 else tuple(a for a in self.arms if self.scene.arm_hw[a].get("hand"))
        mj, m = self.solver._mujoco, self.solver.mj_model
        # body and geom maps (Newton labels -> MuJoCo, '/' -> '_')
        g2s = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
        sb = self.model.shape_body.numpy()
        self._g2s = g2s
        self._g2b = np.where(g2s >= 0, sb[np.clip(g2s, 0, None)], -1)
        labels = list(self.model.shape_label)
        # the fixture table's conductor: its top and legs, and the toggle clamps' steel bases and arms bolted to it
        # (DESTACO 225-U: steel, electro-zinc plated [search: distributor listings, 2026-10-11], so at the table's
        # potential; their neoprene spindle caps are separate bodies and insulate)
        clamp_steel = {f"{C['name']}_{part}" for C in LY.CLAMPS for part in ("base", "arm")}
        self._table_shapes = {i for i, lab in enumerate(labels) if lab.startswith("table_") or lab in clamp_steel}
        self._steel_bodies = {self.scene.bodies[k] for k in ("base_plate", "bracket", "work_clamp") if k in self.scene.bodies}
        names = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j): int(m.jnt_dofadr[j]) for j in range(m.njnt)}
        self._arm_mj_dofs = {}
        for name in ([] if self.g1 else self.joints):      # the G1 has no libfranka collision reflex
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mn, dof in names.items() if mn and mn.startswith(arm + "_") and mn.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._robot_bodies = {i for i, lab in enumerate(self.model.body_label)
                              if any(lab.startswith(a + "/") for a in self.arms)}
        if self.g1:                    # the ready posture over the profile's standing pose (humanoid.READY)
            from . import humanoid as HU
            q, t = self.model.joint_q.numpy(), self.control.joint_target_q.numpy()
            for key, v in HU.READY.items():
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
            self.model.joint_q.assign(q)
            self.control.joint_target_q.assign(t)
        # deposited weld metal's colliders (scene.weld_colliders): given the bracket's collision group, so
        # they meet the robots, the gun, the hand, the clamps and the work clamp but not the bracket they lie against
        # (the base plate is their own body, which MuJoCo never collides with itself)
        s2g = {int(sh): gi for gi, sh in enumerate(g2s) if sh >= 0}
        self._weld_geoms = [s2g[i] for i in getattr(self.scene, "weld_colliders", [])]
        if self._weld_geoms and "bracket" in self.scene.bodies:
            kg = [gi for gi in range(m.ngeom) if self._g2b[gi] == self.scene.bodies["bracket"]][0]
            for gi in self._weld_geoms:
                m.geom_contype[gi], m.geom_conaffinity[gi] = m.geom_contype[kg], m.geom_conaffinity[kg]
        # compiled at the plate's origin, MuJoCo marked these geoms as sharing the body's frame (geom_sameframe) and
        # copies the body's pose to them every step, ignoring the poses written below: without this line every
        # weld-metal box sits at the plate's centre, inside the plate, and touches nothing
        for gi in self._weld_geoms:
            m.geom_sameframe[gi] = 0
        self._weld_park = {gi: (m.geom_pos[gi].copy(), m.geom_quat[gi].copy(), m.geom_size[gi].copy())
                           for gi in self._weld_geoms}
        self.weld_collider_boxes = {}            # segment -> (center weld frame, phi rad, half extents, empty area m2
                                                 # inside the box, largest empty gap m above the metal) in use
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        self._cmd_v = np.zeros(len(self.targets))
        self._finger_speed = {j: FRANKA_HAND_FINGER_SPEED for a in self.hand_arms for j in self.gripper_joints(a)}
        self.grip_force = {a: 0. for a in self.hand_arms}
        self.grasp_latched = {}
        self.tick, self.fault = 0, None
        self.events, self.reflexes = [], []
        self._reflex_until = {}
        self._back_off = {}                      # arm -> (end of the back-off window, the load that stopped it)
        self._shrink_applied = 0.                # rad of shrinkage turn already in the bond's rest pose
        self.refused_targets = 0
        self.timing = dict(rigid_s=0., process_s=0., bond_s=0.)
        if self.g1:
            from .humanoid import G1Kinematics
            self.kin = G1Kinematics(self)
            self._g1_setup()
        else:
            from .kinematics import Kinematics
            self.kin = Kinematics(self)
        # the weld process on the coupon (GPU)
        self.process = None
        self.phantom = None                 # ASSISTED: fn(t) -> (tip, axis) in the weld frame
        if c.process:
            grav = {"2F": (0., 0., -9.81)}[c.gravity_position]
            self.process = WeldProcess(self.coupon, device=c.device, draft_m_s=c.draft_m_s, gravity=grav)
            self.process.pool.obstacle = self._pool_obstacles
        self._tip_prev = None
        self.circuit_closed = False
        self.stray_arc = None             # an arc burning on a grounded body outside the coupon's grid (_process_tick)
        self._wire_geom = None
        self._circuit_tick = -10**9
        from .bonds import Bond
        self.bond = Bond(self)
        self._mj_bad = self._mj_warnings()
        self.assist_force = None          # ASSISTED owner-side probes only: (body, world point, fn(t) -> force N)
        self.assist_log = []
        self._connected = set()
        if "work_clamp" in self.scene.bodies:
            self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
            self.solver._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)
            self._update_circuit()

    # ------------------------------------------------------------------ helpers
    @property
    def time(self):
        return self.tick*self.config.rigid_dt

    def gripper_joints(self, arm):
        if self.g1:                    # the Dex3 hands are commanded joint by joint (named targets)
            return []
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def _g1_setup(self):
        """G1 upper-body servos: position PD plus a gravity feed-forward torque from the model at the commanded
        configuration, as Unitree's G1 manipulation stack sends it for the arms (xr_teleoperate robot_arm_ik.py:
        pinocchio rnea(q_target, 0, 0) -> motor_cmd.tau; as sewing/world.py). The model includes the torch and its
        clamp on the right wrist (their masses are on that link), as an integrator sets the tool payload. The waist gets
        the same; legs and Dex3 fingers are PD only."""
        mj, mjm = self.solver._mujoco, self.solver.mj_model
        names = {mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_JOINT, j): j for j in range(mjm.njnt)}
        arm = [n for n in self.joints if any(k in n for k in ("_shoulder_", "_elbow_", "_wrist_", "/waist_"))]
        mj_j = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in arm]
        self._g1_ff = dict(names=arm, q=[self.joints[n]["q"] for n in arm], dof=[self.joints[n]["dof"] for n in arm],
                           mj_q=[int(mjm.jnt_qposadr[j]) for j in mj_j], mj_dof=[int(mjm.jnt_dofadr[j]) for j in mj_j],
                           limit=np.array([float(self.joints[n]["effort_limit"]) for n in arm]),
                           data=mj.MjData(mjm), tau=np.zeros(len(arm)),
                           every=max(1, int(round(G1_FF_EVERY_S/self.config.rigid_dt))))

    def _g1_contacts(self):
        """No inactive far contacts on the G1, as for every other shape of the cell (scene.py sets the builder's rigid_gap
        to 0). The robots package builds the G1 in its own ModelBuilder, whose default contact gap is Newton's 0.1 m, and
        merges it: MuJoCo then generated "inactive" contacts out to 100 mm for every pair of a G1 geom with anything
        (1,240-1,790 contacts a step against 84 constraint rows; the rigid G1 cell ran 12 wall s per simulated s, 0.9
        without them). At those distances MuJoCo 3.11's box-box routine returned one contact with its distance's sign
        flipped: the left thumb's box, 8.2 mm clear of the loose bracket by mj_geomDistance, got a contact 29.6 mm deep
        and a 76-106 N kick that knocked the bracket over; with no gap the same configuration makes no contact. Inactive
        contacts carry no force, and nothing here reads them."""
        sb = self.model.shape_body.numpy()
        gap = self.model.shape_gap.numpy()
        gap[np.isin(sb, list(self.scene.robot_bodies))] = 0.
        self.model.shape_gap.assign(gap)

    def _g1_gravity_feedforward(self, jf, cur):
        """Gravity torque of the G1 waist and arm joints at their commanded positions (the rest of the robot as
        measured), refreshed every G1_FF_EVERY_S, clipped to the URDF effort limits."""
        ff = self._g1_ff
        if self.tick % ff["every"] == 0:
            mj = self.solver._mujoco
            m, d = self.solver.mj_model, ff["data"]
            d.qpos[:] = self.solver.mj_data.qpos
            d.qvel[:] = 0.
            d.qpos[ff["mj_q"]] = cur[ff["q"]]
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            out = np.zeros(m.nv)
            mj.mj_rne(m, d, 0, out)
            ff["tau"] = np.clip(out[ff["mj_dof"]], -ff["limit"], ff["limit"])
        jf[ff["dof"]] += ff["tau"]

    def body_pose(self, body):
        from scipy.spatial.transform import Rotation
        q = self.state.body_q.numpy()[body].astype(float)
        return q[:3], Rotation.from_quat(q[3:]).as_matrix()

    def weld_frame(self):
        """(origin, R) of the weld frame in the world, carried by the base plate body."""
        p, R = self.body_pose(self.scene.bodies["base_plate"])
        return p + R @ self.scene.weld_origin_in_base, R

    def torch_pose(self):
        """World (contact tip point, wire axis unit vector, work point at the nominal CTWD) from the torch arm's link."""
        p, R = self.body_pose(self.scene.gun_body)
        pt, Rt = self.scene.tool0_in_body
        g = self.scene.gun_spec
        tip = p + R @ (pt + Rt @ g.contact_tip)
        axis = R @ Rt @ g.axis
        tcp = p + R @ (pt + Rt @ g.tcp)
        return tip, axis, tcp

    def _mj_warnings(self):
        mj = self.solver._mujoco
        d = self.solver.mj_data
        W = mj.mjtWarning
        return sum(int(d.warning[int(k)].number) for k in (W.mjWARN_BADQACC, W.mjWARN_BADQPOS, W.mjWARN_BADQVEL))

    # ------------------------------------------------------------------ commands
    def set_targets(self, targets):
        latched = {arm for arm, until in self._reflex_until.items() if self.time < until}
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

    def set_grip_force(self, arm, force):
        if self.g1:
            raise ValueError("the G1's Dex3 hands have no gripper force mode; command their joints by name")
        if arm not in self.hand_arms:
            raise ValueError(f"No hand on arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) \
                or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]

    # no "process" output: the transfer mode is not a source setting, it follows from wire feed, voltage, stand-off and
    # gas (arc.CVSource.transfer_mode)
    TOOL_IO = {"arc_on": "on", "gas_on": "gas_on", "touch_sense": "sensing", "wire_feed_m_min": "wfs_m_min",
               "voltage_v": "voltage_v", "wire_inch": "inch"}

    def tool_io(self, **outputs):
        """The robot's outputs to the welding power source (a robot-source interface: weld start, gas, touch-sense
        mode, wire feed and voltage set points, wire inching). Ranges are validated by the caller (backend)."""
        if self.process is None:
            raise ValueError("this world was built without the weld process")
        unknown = set(outputs) - set(self.TOOL_IO)
        if unknown:
            raise ValueError(f"unknown tool outputs {sorted(unknown)}")
        self.process.arc_command(**{self.TOOL_IO[k]: v for k, v in outputs.items()})
        self.events.append(dict(tick=self.tick, tool_io=dict(outputs)))

    # ------------------------------------------------------------------ the welding circuit
    def _update_circuit(self):
        """Steel bodies connected to the work clamp through contacts with load (or the bond), the table top counting
        as one conductor. Returns the set of connected Newton body ids."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        f6 = np.zeros(6)
        edges = []
        TABLE = -2
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom[0]), int(con.geom[1])
            n1 = TABLE if int(self._g2s[g1]) in self._table_shapes else int(self._g2b[g1])
            n2 = TABLE if int(self._g2s[g2]) in self._table_shapes else int(self._g2b[g2])
            if n1 == n2:
                continue
            mj.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) > 1e-2:
                edges.append((n1, n2))
        if self.bond.active:
            edges.append((self.scene.bodies["base_plate"], self.scene.bodies["bracket"]))
        start = self.scene.bodies["work_clamp"]
        seen, stack = {start}, [start]
        while stack:
            a = stack.pop()
            for u, v in edges:
                for x, y in ((u, v), (v, u)):
                    if x == a and y not in seen and (y in self._steel_bodies or y == TABLE):
                        seen.add(y)
                        stack.append(y)
        self._connected = seen
        return seen

    def set_bond_stiffness(self):
        """The bond equality's impedance for a stiffness proportional to the bond's capacity (PR.BOND_STIFFNESS), from
        the measured soft-constraint law k = C0/(d(1-d)); d within [0.5, 0.9999]."""
        m = self.solver.mj_model
        i = self._bond_eq()
        k_ref, c_ref = PR.v(PR.BOND_STIFFNESS)
        k = k_ref*max(self.bond.capacity_n, 0.)/c_ref
        x = 1. - 4.*PR.v(PR.BOND_SOFT_C0)/max(k, 1e-9)
        dimp = float(np.clip(.5*(1. + math.sqrt(x)) if x > 0 else .5, .5, .9999))
        m.eq_solimp[i] = (dimp, dimp, .001, .5, 2.)
        self.bond_impedance = dimp

    def _wire_ray(self, tip, axis, reach=3.):
        """(distance from the contact tip along the wire axis to the first collision geometry that is not the gun's
        own, its circuit node: a Newton body id, -2 for the table top and what is bolted to it, None if nothing within
        reach). World frame. The geometry's label is kept in self._wire_geom."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        if not hasattr(self, "_gun_mj_body"):
            gb = [int(m.geom_bodyid[g]) for g in range(m.ngeom) if int(self._g2b[g]) == self.scene.gun_body]
            self._gun_mj_body = gb[0] if gb else -1
        gid = np.full(1, -1, np.int32)
        dist = mj.mj_ray(m, d, np.asarray(tip, np.float64), np.asarray(axis, np.float64), None, 1, self._gun_mj_body, gid)
        self._wire_geom = None
        if dist < 0 or dist > reach or gid[0] < 0:
            return float("inf"), None
        g = int(gid[0])
        sh = int(self._g2s[g])
        self._wire_geom = str(self.model.shape_label[sh]) if sh >= 0 else None
        node = -2 if sh in self._table_shapes else int(self._g2b[g])
        return float(dist), node

    def _pool_obstacles(self, xs, pts, nrm):
        """Clearance for the weld pool's free surface (pool.Pool.obstacle): from each chord point of each liquid slice
        (weld frame: x = xs[s], (y, z) = pts[s, k]) along the slice's outward normal nrm[s] to the first collision
        geometry that is not the coupon's metal (the base plate with its weld-metal boxes, and the bracket while the
        thermal grid holds it: its metal is the grid's cells there), inf where nothing is within one chord length.
        Rays are cast only when some geometry's bounding box meets the box the slices' surfaces can reach (in normal
        welding the nozzle stands 15 mm off and none is cast). None when nothing is near."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        o, R = self.weld_frame()
        reach = float(np.max(np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1)))
        ends = np.concatenate([pts, pts + reach*nrm[:, None, :]], 1)
        lo = np.array([xs.min() - 5e-4, ends[..., 0].min(), ends[..., 1].min()])
        hi = np.array([xs.max() + 5e-4, ends[..., 0].max(), ends[..., 1].max()])
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        cw = o[None, :] + corners @ R.T
        wlo, whi = cw.min(0), cw.max(0)
        if not hasattr(self, "_obst_geoms"):
            plate = self.scene.bodies["base_plate"]
            self._obst_plate = int(m.geom_bodyid[[g for g in range(m.ngeom) if int(self._g2b[g]) == plate][0]])
            self._obst_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) != self._obst_plate
                                         and int(m.geom_type[g]) != int(mj.mjtGeom.mjGEOM_PLANE)], int)
        G = self._obst_geoms
        xp = d.geom_xpos[G]
        near = np.linalg.norm(xp - np.clip(xp, wlo, whi), axis=1) <= m.geom_rbound[G]
        bracket = self.scene.bodies.get("bracket")
        skip = bracket if (bracket is not None and self.bracket_in_grid) else None
        cand = 0
        for g in G[near]:
            if skip is not None and int(self._g2b[g]) == skip:
                continue
            Rg = d.geom_xmat[g].reshape(3, 3)
            loc = (cw - d.geom_xpos[g]) @ Rg                         # the reachable box's corners in the geom's frame
            c_, h_ = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            if np.all(loc.min(0) <= c_ + h_) and np.all(loc.max(0) >= c_ - h_):
                cand += 1
        if not cand:
            return None
        out = np.full(pts.shape[:2], np.inf)
        gid = np.full(1, -1, np.int32)
        for s_ in range(pts.shape[0]):
            dw = R @ np.array([0., nrm[s_, 0], nrm[s_, 1]])
            for k in range(1, pts.shape[1] - 1):
                p0 = o + R @ np.array([xs[s_], pts[s_, k, 0], pts[s_, k, 1]])
                run = 0.
                for _ in range(3):
                    dist = mj.mj_ray(m, d, p0, dw, None, 1, self._obst_plate, gid)
                    if dist < 0 or gid[0] < 0 or run + dist > reach:
                        break
                    if skip is not None and int(self._g2b[int(gid[0])]) == skip:
                        run += dist + 1e-6                            # through the bracket's box (its metal is cells)
                        p0 = p0 + dw*(dist + 1e-6)
                        continue
                    out[s_, k] = run + dist
                    break
        return out

    def _arc_body(self, cell_body):
        """Rigid body of the metal the arc reaches (thermal body id): 1 base plate, 2 bracket, 255 weld metal."""
        from .process import BODY_BASE, BODY_WEB
        if cell_body == BODY_WEB:
            return self.scene.bodies.get("bracket")
        return self.scene.bodies["base_plate"]

    # ------------------------------------------------------------------ the bond (bonds.py)
    def set_bond(self, on):
        from scipy.spatial.transform import Rotation
        m, d = self.solver.mj_model, self.solver.mj_data
        i = self._bond_eq()
        if on:
            p1, R1 = self.body_pose(self.scene.bodies["base_plate"])
            p2, R2 = self.body_pose(self.scene.bodies["bracket"])
            q = Rotation.from_matrix(R1.T @ R2).as_quat()
            a = self._bond_anchor()
            m.eq_data[i, :3] = a
            # MuJoCo's weld: relpose position is where the anchor (a point of body2) sits in body1's frame (measured:
            # with relpose = body2's origin, a nonzero anchor read as a position error equal to the anchor, and the
            # first tack bond snapped under 3 kN on its first tick)
            m.eq_data[i, 3:6] = R1.T @ (p2 + R2 @ a - p1)
            m.eq_data[i, 6:10] = (q[3], q[0], q[1], q[2])
            m.eq_data[i, 10] = 1.
            m.eq_solref[i] = (.004, 1.)
            self.set_bond_stiffness()
            # a bond made now rests at the present pose: the shrinkage so far is in that pose already
            self._shrink_applied = float(getattr(self.bond, "shrink_target_rad", 0.))
        d.eq_active[i] = 1 if on else 0

    def park_work_clamp(self):
        """Initial-condition step (before tick 0, disclosed): the ground clamp is put on the power source's steel case
        on its cart, where it closes no welding circuit (the research's circuit variant)."""
        from . import layout as L
        k = self.scene.bodies["work_clamp"]
        j = [i for i, lab in enumerate(self.model.joint_label) if self.model.joint_child.numpy()[i] == k][0]
        qs = int(self.model.joint_q_start.numpy()[j])
        ds = int(self.model.joint_qd_start.numpy()[j])
        x0, x1, y0, y1, z = L.source_top()
        q = self.state.joint_q.numpy()
        q[qs:qs + 7] = ((x0 + x1)/2, (y0 + y1)/2, z + L.WORK_CLAMP["size"][2]/2 + 1e-4, 0., 0., 0., 1.)
        qd = self.state.joint_qd.numpy()
        qd[ds:ds + 6] = 0.
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.solver._update_mjc_data(self.solver.mj_data, self.model, self.state)
        self.solver._mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)

    def _bond_anchor(self):
        """MuJoCo weld anchor (bracket frame): the bonded sections' centroid on the root line, so the constraint's
        moment is the moment on the weld (the bond's interaction rule is written for it). At the base plate's centre
        the peel test read the moment 6 % high and broke 15 % early (E7)."""
        xr = self.bond.x_range
        xc = .5*(xr[0] + xr[1]) if xr else self.coupon.length/2
        return self.scene.weld_origin_in_bracket + np.array([xc, 0., 0.])

    def update_bond_anchor(self):
        """Move the anchor along the root line as the bonded length grows, keeping the constraint exactly satisfied:
        the new anchor's place in body1 comes from the old relation (the bodies' current relative pose is the bond's
        rest pose)."""
        if not self.bond.active:
            return
        from scipy.spatial.transform import Rotation
        m = self.solver.mj_model
        i = self._bond_eq()
        a_old, rel_old = np.array(m.eq_data[i, :3]), np.array(m.eq_data[i, 3:6])
        qw = np.array(m.eq_data[i, 6:10])
        Rrel = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()       # body2 in body1
        o2 = rel_old - Rrel @ a_old                                                # body2's origin in body1
        a = self._bond_anchor()
        m.eq_data[i, :3] = a
        m.eq_data[i, 3:6] = o2 + Rrel @ a
        # shrinkage distortion: the bond's rest pose turns about the root line through the anchor, the bracket's top
        # toward the weld side (+y of the weld frame, which is the base plate's frame), as the weld metal cools
        th = float(getattr(self.bond, "shrink_target_rad", 0.))
        dth = th - self._shrink_applied
        if abs(dth) > 1e-7:
            Rrel = Rotation.from_rotvec([-dth, 0., 0.]).as_matrix() @ Rrel
            q = Rotation.from_matrix(Rrel).as_quat()
            m.eq_data[i, 6:10] = (q[3], q[0], q[1], q[2])
            self._shrink_applied = th
        self.set_bond_stiffness()                 # the bond stiffens as it grows and cools (its capacity)

    def _bond_eq(self):
        m = self.solver.mj_model
        mj = self.solver._mujoco
        if not hasattr(self, "_bond_eq_id"):
            ids = [i for i in range(m.neq) if int(m.eq_type[i]) == int(mj.mjtEq.mjEQ_WELD)]
            if len(ids) != 1:
                raise RuntimeError(f"expected one weld equality (the bond), found {len(ids)}")
            self._bond_eq_id = ids[0]
        return self._bond_eq_id

    def bond_wrench(self):
        """(force N, moment N m) the bond's equality transmits (MuJoCo constraint forces of its 6 rows)."""
        m, d = self.solver.mj_model, self.solver.mj_data
        mj = self.solver._mujoco
        i = self._bond_eq()
        rows = [r for r in range(d.nefc) if int(d.efc_type[r]) == int(mj.mjtConstraint.mjCNSTR_EQUALITY)
                and int(d.efc_id[r]) == i]
        f = np.array([d.efc_force[r] for r in rows]) if rows else np.zeros(6)
        if len(f) < 6:
            return np.zeros(3), np.zeros(3)
        # MuJoCo's weld rotational rows are conjugate to half the relative angular velocity (the quaternion
        # derivative): their constraint forces read twice the torque. Measured: 100 N at 75 mm read 14.97 N m for the
        # true 7.50; read as torque, the tack bond broke at half its predicted load.
        return f[:3], f[3:6]/(2.*max(float(m.eq_data[i, 10]), 1e-12))

    def bond_eq_state(self):
        m, d = self.solver.mj_model, self.solver.mj_data
        i = self._bond_eq()
        return dict(data=np.array(m.eq_data[i]).copy(), solref=np.array(m.eq_solref[i]).copy(),
                    solimp=np.array(m.eq_solimp[i]).copy(), active=int(d.eq_active[i]),
                    shrink_applied=float(self._shrink_applied))

    def restore_bond_eq(self, st):
        m, d = self.solver.mj_model, self.solver.mj_data
        i = self._bond_eq()
        m.eq_data[i] = st["data"]
        m.eq_solref[i] = st["solref"]
        if "solimp" in st:
            m.eq_solimp[i] = st["solimp"]
        d.eq_active[i] = st["active"]
        self._shrink_applied = float(st.get("shrink_applied", 0.))

    # ------------------------------------------------------------------ contact queries (read-only)
    def _contacts(self, min_force=1e-2):
        """(newton shape a, newton shape b, normal force N) of MuJoCo's active contacts."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        f6 = np.zeros(6)
        out = []
        for i in range(d.ncon):
            con = d.contact[i]
            mj.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) >= min_force:
                out.append((int(self._g2s[con.geom[0]]), int(self._g2s[con.geom[1]]), abs(float(f6[0]))))
        return out

    def gun_contacts(self):
        """The gun's shapes touching anything: one entry per (gun shape, other shape) pair with the summed normal force
        and the number of MuJoCo contacts (the evaluator's 'no contact'). A cylinder against a box is reported as three
        equal contact rows that share the load; listing them one by one would read as a third of the pair's force
        (E6, E8)."""
        gun = set(self.scene.gun_shapes)
        lab = self.model.shape_label
        pairs = {}
        for a, b, f in self._contacts():
            if a in gun or b in gun:
                g, o = (a, b) if a in gun else (b, a)
                e = pairs.setdefault((g, o), [0., 0])
                e[0] += f
                e[1] += 1
        return [dict(gun=lab[g], other=lab[o] if o >= 0 else "static", force_n=round(f, 3), contacts=n)
                for (g, o), (f, n) in pairs.items()]

    def bodies_touch(self, body, others):
        sb = self.model.shape_body.numpy()
        for a, b, f in self._contacts():
            ba, bb = (int(sb[a]) if a >= 0 else -1), (int(sb[b]) if b >= 0 else -1)
            if (ba == body and bb in others) or (bb == body and ba in others):
                return True
        return False

    def hand_bodies(self):
        lab = list(self.model.body_label)
        if self.g1:                    # the left Dex3: its palm (on the wrist-yaw link) and finger links
            return [i for i in self._robot_bodies if lab[i].split("/")[-1].startswith("left_hand_")
                    or lab[i].endswith("/left_wrist_yaw_link")]
        return [i for i in self._robot_bodies if any(lab[i].startswith(a + "/") for a in self.hand_arms)
                and ("finger" in lab[i] or "hand" in lab[i] or "link7" in lab[i])]

    # ------------------------------------------------------------------ protective stops
    def external_joint_torques(self):
        m, d = self.solver.mj_model, self.solver.mj_data
        mj = self.solver._mujoco
        tau = np.zeros(m.nv)
        if d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mj.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            if f.any():
                mj.mj_mulJacTVec(m, d, tau, f)
        return {arm: tau[[dof for _, dof in dofs]] for arm, dofs in self._arm_mj_dofs.items()}

    def robot_contact_force(self, arm, links=None):
        """Net contact force (N) on the arm's bodies (and its tool) from everything that is not the arm itself; with
        `links` (body label endings), only the contacts on those of its bodies."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        lab = list(self.model.body_label)
        mine = {i for i in self._robot_bodies if lab[i].startswith(arm + "/")}
        whole = mine
        if links is not None:
            mine = {i for i in mine if lab[i].endswith(tuple(links))}
        f6 = np.zeros(6)
        F = np.zeros(3)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(self._g2b[con.geom[0]]), int(self._g2b[con.geom[1]])
            if (b1 in mine) == (b2 in mine) or (b1 in whole and b2 in whole):
                continue
            mj.mj_contactForce(m, d, i, f6)
            R = np.asarray(con.frame).reshape(3, 3)
            fw = R.T @ f6[:3]
            F += fw if b2 in mine else -fw
        return F

    def _stop_arm(self, arm, q):
        cur = self.control.joint_target_q.numpy()
        for name, spec in self.joints.items():
            if name.startswith(arm + "/") and "finger" not in name:
                self.targets[spec["q"]] = cur[spec["q"]] = float(q[spec["q"]])
                self._cmd_v[spec["q"]] = 0.
        self.control.joint_target_q.assign(cur)

    def _protective_stops(self):
        if self.g1:                    # the G1's low-level joint interface has no collision or velocity reflex
            return
        q = self.state.joint_q.numpy()
        if not hasattr(self, "_q_hist"):
            from collections import deque
            self._q_hist = deque(maxlen=200)
        if self._q_hist and self._q_hist[-1][0] >= self.time:
            self._q_hist.clear()                 # a restored checkpoint or a reset: the path before it is not this one
        self._q_hist.append((self.time, q.copy()))
        qd = self.state.joint_qd.numpy()
        ext = None
        for arm in self.arms:
            model = self.scene.arm_hw[arm]["model"]
            hit = None
            for name, spec in self.joints.items():
                if name.startswith(arm + "/") and "finger" not in name and \
                        abs(float(qd[spec["dof"]])) > VELOCITY_STOP*spec["velocity_limit"]:
                    hit = dict(kind="velocity", joint=name, speed_rad_s=float(qd[spec["dof"]]))
                    break
            if hit is None and model == "ur10e":
                for where, links in (("tool", UR_TOOL_LINKS), ("elbow", UR_ELBOW_LINKS)):
                    F = self.robot_contact_force(arm, links)
                    if np.linalg.norm(F) > UR_FORCE_LIMIT_N:
                        hit = dict(kind="force_limit", where=where, force_n=float(np.linalg.norm(F)),
                                   limit_n=UR_FORCE_LIMIT_N)
                        break
            elif hit is None:
                ext = ext or self.external_joint_torques()
                cur = self.control.joint_target_q.numpy()
                dofs = self._arm_mj_dofs[arm]
                moving = any(abs(float(self.targets[self.joints[n]["q"]] - cur[self.joints[n]["q"]])) > 1e-6 for n, _ in dofs)
                lim = REFLEX_TORQUE_NM[1] if moving else REFLEX_TORQUE_NM[0]
                e = np.abs(ext[arm])
                k = int(np.argmax(e))
                if e[k] > lim:
                    hit = dict(kind="collision", joint=dofs[k][0], external_torque_nm=float(e[k]), limit_nm=lim)
            if hit:
                load = hit.get("force_n", hit.get("external_torque_nm"))
                win = self._back_off.get(arm)
                if load is not None and win is not None and self.time < self._reflex_until.get(arm, -1.):
                    # stopped: the load it holds while it stops (still rising as the arm decelerates) is the stop's own
                    self._back_off[arm] = (win[0], max(win[1], load))
                    continue
                if load is not None and win is not None and self.time < win[0] and load <= BACK_OFF_MARGIN*win[1]:
                    continue                     # the stop's own load, held or being backed off: not a new stop
                self._reflex_until[arm] = self.time + REFLEX_RECOVERY_S
                self._back_off[arm] = (self.time + REFLEX_RECOVERY_S + BACK_OFF_S, load if load is not None else 0.)
                if hit["kind"] == "force_limit":
                    # UR: stop, then back off to where the force limit is not exceeded: the arm retraces its own path
                    past = [qq for t_, qq in self._q_hist if t_ <= self.time - UR_BACK_OFF_TRACE_S]
                    self._stop_arm(arm, past[-1] if past else q)
                else:
                    self._stop_arm(arm, q)
                if not self.reflexes or self.reflexes[-1]["tick"] < self.tick - 50 or self.reflexes[-1]["arm"] != arm:
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, **hit))

    # ------------------------------------------------------------------ stepping
    def _motion_generator(self):
        c = self.config
        cur = self.control.joint_target_q.numpy()
        dt = c.rigid_dt
        for name, spec in self.joints.items():
            j = spec["q"]
            vlim = spec["velocity_limit"]
            if name in self._finger_speed:
                vlim = min(vlim, self._finger_speed[name])
                e = float(self.targets[j] - cur[j])
                cur[j] += float(np.clip(e, -vlim*dt, vlim*dt))
                continue
            vlim *= PLAN_SPEED_FRACTION
            model = self.scene.arm_hw[name.split("/")[0]]["model"]
            amax = UR_ACCEL if model == "ur10e" else FR3_ACCEL
            if model == "panda":
                amax = PANDA_ACCEL[int(name[-1]) - 1]
            e = float(self.targets[j] - cur[j])
            v0 = self._cmd_v[j]
            v_stop = math.copysign(min(vlim, math.sqrt(2.*amax*abs(e))), e) if e else 0.
            v1 = min(max(v_stop, v0 - amax*dt), v0 + amax*dt)
            step = v1*dt
            if e == 0. or (e > 0. and step >= e) or (e < 0. and step <= e):
                cur[j] = self.targets[j]
                v1 = 0.
            else:
                cur[j] += step
            self._cmd_v[j] = v1
        jf = np.zeros(self.model.joint_dof_count, np.float32)
        q_now = self.state.joint_q.numpy()
        qd_now = self.state.joint_qd.numpy()
        for arm, force in self.grip_force.items():
            if not force:
                continue
            for name in self.gripper_joints(arm):
                spec = self.joints[name]
                key = (arm, name)
                if key not in self.grasp_latched:
                    lag = float(q_now[spec["q"]] - cur[spec["q"]])
                    stalled = abs(float(qd_now[spec["dof"]])) < 5e-3
                    done = abs(float(self.targets[spec["q"]] - cur[spec["q"]])) < 1e-6
                    if (lag > 5e-4 and stalled) or done:
                        self.grasp_latched[key] = True
                if key in self.grasp_latched:
                    cur[spec["q"]] = q_now[spec["q"]]
                    jf[spec["dof"]] = -force
        self.control.joint_target_q.assign(cur)
        if self.g1:
            self._g1_gravity_feedforward(jf, cur)
        self.control.joint_f.assign(jf)

    def _external_forces(self):
        """The ground clamp's magnet: 490 N pull toward the steel table while its face is within MAGNET_RANGE."""
        bf = np.zeros((self.model.body_count, 6), np.float32)
        k = self.scene.bodies["work_clamp"]
        p, R = self.body_pose(k)
        half_z = LY.WORK_CLAMP["size"][2]/2
        bottom = p[2] - half_z*abs(R[2, 2])
        T = LY.TABLE
        on_table = abs(p[0]) < T["size"][0]/2 and abs(p[1]) < T["size"][1]/2
        if on_table and R[2, 2] > .95 and bottom - T["top_z"] < MAGNET_RANGE:
            bf[k, 2] = -LY.WORK_CLAMP["pull_n"]
        elif not on_table and R[2, 2] > .95:
            # on the power source's steel top the magnet holds too (it is steel), but the case is not in the circuit
            x0, x1, y0, y1, z = LY.source_top()
            if x0 < p[0] < x1 and y0 - .03 < p[1] < y1 + .03 and bottom - z < MAGNET_RANGE:
                bf[k, 2] = -LY.WORK_CLAMP["pull_n"]
        st = self.process.src.st if self.process is not None else None
        if st is not None and st.wire_push_n > 0:
            # a stubbed wire: the feeder drives it into the work and the work pushes back on the gun along the wire
            # axis, at the contact tip (equal and opposite on the base plate, which the clamps hold)
            tip, axis, _ = self.torch_pose()
            F = -float(st.wire_push_n)*np.asarray(axis, float)
            for body, f_ in ((self.scene.gun_body, F), (self.scene.bodies["base_plate"], -F)):
                pc = np.asarray(self.state.body_q.numpy()[body][:3], float)
                com = pc + self.body_pose(body)[1] @ self.model.body_com.numpy()[body].astype(float)
                bf[body, :3] += f_
                bf[body, 3:] += np.cross(np.asarray(tip, float) - com, f_)
        if self.assist_force is not None:
            # ASSISTED: phantom forces on bodies at world points (probes), never reachable from the backend; one
            # (body, point, fn) or a list of them
            items = self.assist_force if isinstance(self.assist_force, list) else [self.assist_force]
            for body, pt, fn in items:
                F = np.asarray(fn(self.time), float)
                pc = np.asarray(self.state.body_q.numpy()[body][:3], float)
                com = pc + self.body_pose(body)[1] @ self.model.body_com.numpy()[body].astype(float)
                bf[body, :3] += F
                bf[body, 3:] += np.cross(np.asarray(pt, float) - com, F)
        self.state.body_f.assign(bf)

    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            self._motion_generator()
            self.state.clear_forces()
            self._external_forces()
            self.solver.step(self.state, self.next_state, self.control, self.contacts, c.rigid_dt)
            bad = self._mj_warnings()
            if bad > self._mj_bad:
                raise FloatingPointError(f"MuJoCo reported an unstable state at tick {self.tick}")
            self.state, self.next_state = self.next_state, self.state
            self.tick += 1
            for attr in ("body_q", "body_qd", "joint_q"):
                if not np.isfinite(getattr(self.state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite rigid {attr} at tick {self.tick}")
            self._protective_stops()
            t1 = time.perf_counter()
            self.timing["rigid_s"] += t1 - t0
            if self.process is not None:
                if self.tick % GRID_EVERY == 0:
                    self._bracket_grid()
                self._process_tick()
                t2 = time.perf_counter()
                self.timing["process_s"] += t2 - t1
                if self.tick % BOND_EVERY == 0 and self.process.ever_hot:
                    self.bond.update()
                    self._weld_colliders()
                self.bond.check()
                self.timing["bond_s"] += time.perf_counter() - t2
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _process_tick(self):
        c = self.config
        o, Rb = self.weld_frame()
        if self.phantom is not None:
            tip_w, axis_w = self.phantom(self.time)
        else:
            tip, axis, _ = self.torch_pose()
            tip_w, axis_w = Rb.T @ (tip - o), Rb.T @ axis
        vel = np.zeros(3) if self._tip_prev is None else (tip_w - self._tip_prev)/c.rigid_dt
        self._tip_prev = tip_w.copy()
        st = self.process.src.st
        if self.tick - self._circuit_tick >= CIRCUIT_EVERY or st.on_cmd or st.sensing:
            if self.tick - self._circuit_tick >= CIRCUIT_EVERY:
                self._update_circuit()
                self._circuit_tick = self.tick
        from .arc import attach, MAX_STICKOUT
        # which body would the arc reach? (cheap re-use: the process attaches again inside tick)
        point, dist, body = attach(self.process.g, self.process.s.mat, self.process.s.body, tip_w, axis_w)
        if self.phantom is None:
            # the wire is a rod along the axis: it stops against any geometry it meets first (table, clamp, a robot link,
            # the coupon's own boxes), not only the coupon's metal cells (before: an inched wire ran through the table)
            d_ray, node = self._wire_ray(tip, axis)
            coupon = node in (self.scene.bodies["base_plate"], self.scene.bodies.get("bracket", -9))
            if coupon and not math.isfinite(dist) and d_ray <= MAX_STICKOUT + 1e-3:
                # the wire meets the coupon beyond the arc root's search reach (arc.STRIKE_REACH, 40 mm) but within the
                # wire's own reach (arc.MAX_STICKOUT): search out to where the wire meets it, so that a wire inched onto
                # the coupon from farther off strikes on the coupon like anywhere else on it
                point, dist, body = attach(self.process.g, self.process.s.mat, self.process.s.body, tip_w, axis_w,
                                           reach=d_ray + 1e-3)
            if coupon and math.isfinite(dist):
                d_ray = float("inf")            # the coupon's cells (beads, grooves, melt) rule over its collision boxes
            if d_ray < dist - 1e-4:
                # the wire faces a body outside the coupon's grid. If that body is a conductor at work potential (the
                # fixture table the ground clamp sits on, the zinc-plated toggle clamps bolted to it, the ground clamp
                # itself: _update_circuit) the circuit closes through it: touch sensing reads contact, and with the
                # output on the wire strikes a stray arc there, which the source runs and books (arc.py: stray_node,
                # stray_m3, stray_j; the grid receives neither its heat nor its metal). Anything else (painted or
                # plastic equipment, the robots, ungrounded steel) and free air close nothing: the wire stalls against
                # it or runs to its bound and the source reports arc_start_failure.
                st_ = self.process.src.st
                grounded = node is not None and node in self._connected
                # a coupon body met here has no cells along the wire (a loose bracket moved off its fit-up pose has
                # left the grid: process.detach_part), so it is a stray body too. The process must leave the coupon
                # alone while a stray arc burns (WeldProcess.books_stray_arcs)
                stray = bool(grounded and getattr(self.process, "books_stray_arcs", False))
                closed = bool(grounded and (stray or (st_.sensing and not st_.on_cmd)))
                st_.stray_node = node if stray else None
                self.circuit_closed = closed
                self.wire_stop = dict(distance_m=d_ray, node=node, geom=self._wire_geom)
                self.process.tick(c.rigid_dt, tip_w, axis_w, vel, circuit_closed=closed, attached=(None, d_ray, 0))
                self.stray_arc = (dict(node=node, geom=self._wire_geom, root_w=(np.asarray(tip, float) +
                                                                                np.asarray(axis, float)*d_ray).tolist(),
                                       current_a=float(st_.current), length_m=float(st_.l_a))
                                  if stray and st_.arc else None)
                return
        self.wire_stop = None
        self.stray_arc = None
        self.process.src.st.stray_node = None
        rb = self._arc_body(body) if math.isfinite(dist) else None
        closed = rb is not None and rb in self._connected
        if body == 255 and not closed:
            # weld metal is part of whichever fused part is grounded
            closed = (self.scene.bodies["base_plate"] in self._connected or
                      self.scene.bodies.get("bracket", -9) in self._connected)
        self.circuit_closed = bool(closed)
        self.process.tick(c.rigid_dt, tip_w, axis_w, vel, circuit_closed=self.circuit_closed,
                          attached=(point, dist, body))

    # ------------------------------------------------------------------ deposited weld metal as a collider
    def _weld_colliders(self, every=False):
        """Size and place each segment's collider box from the solid weld-metal cells of its slices. A
        segment is refitted when the pool has changed its cells and none of its slices is liquid (a pool is not a rigid
        body; its collider keeps the last frozen shape meanwhile). The box is the one, of three orientations about the
        joint line (axes along the plate's normal, along the fillet face's normal at 45 deg, along the bracket face's
        normal), that covers the segment's weld cells with the least air inside it; overlapping the plate and the
        bracket is harmless (the plate is its own body, the bracket's group is excluded)."""
        p = self.process
        if p is None or not self._weld_geoms:
            return
        from . import thermal as TH
        pool, g, s = p.pool, p.g, p.s
        segs = self.scene.weld_segments
        if every:
            todo = set(range(len(segs)))
        else:
            dirty, pool.dirty = pool.dirty, set()
            self._weld_dirty = getattr(self, "_weld_dirty", set()) | dirty
            xs = g.xc
            todo = {k for k, (x0, x1) in enumerate(segs)
                    if any(x0 <= xs[i] < x1 for i in self._weld_dirty if 0 <= i < g.nx)}
        if not todo:
            return
        m = self.solver.mj_model
        j0, j1, k0, k1 = pool.j0, pool.j1, pool.k0, pool.k1
        Y, Z = pool.YY, pool.ZZ
        hy, hz = g.dy[j0:j1, None]/2 + 0*Z, g.dz[None, k0:k1]/2 + 0*Y
        done = set()
        for k in sorted(todo):
            x0, x1 = segs[k]
            I = np.nonzero((g.xc >= x0) & (g.xc < x1))[0]
            if len(I) == 0 or any(pool.slices[i].liquid for i in I):
                continue
            done.update(int(i) for i in I)
            idx = g.index(I[:, None, None], np.arange(j0, j1)[None, :, None], np.arange(k0, k1)[None, None, :])
            mat = s.mat[idx]
            weld = (mat == TH.WELD).any(0)
            gi = self._weld_geoms[k]
            if not weld.any():
                m.geom_pos[gi], m.geom_quat[gi], m.geom_size[gi] = self._weld_park[gi]
                m.geom_rbound[gi] = float(np.linalg.norm(m.geom_size[gi]))
                m.geom_aabb[gi] = (0., 0., 0., *m.geom_size[gi])
                self.weld_collider_boxes.pop(k, None)
                continue
            air = ~(mat != TH.AIR).any(0)
            yw, zw = Y[weld], Z[weld]
            ew, ez = hy[weld], hz[weld]
            best = None
            for ny_, nz_ in ((0., 1.), (math.sqrt(.5), math.sqrt(.5)), (1., 0.)):
                ty_, tz_ = nz_, -ny_                                     # t = n rotated by -90 deg in (y, z)
                pn, pt = yw*ny_ + zw*nz_, yw*ty_ + zw*tz_
                en, et = abs(ny_)*ew + abs(nz_)*ez, abs(ty_)*ew + abs(tz_)*ez     # each cell's half extent
                n_lo, n_hi = float((pn - en).min()), float((pn + en).max())
                t_lo, t_hi = float((pt - et).min()), float((pt + et).max())
                qn, qt = Y*ny_ + Z*nz_, Y*ty_ + Z*tz_
                inside = (qn >= n_lo) & (qn <= n_hi) & (qt >= t_lo) & (qt <= t_hi)
                cost = float(np.sum((4*hy*hz)[inside & air]))
                if best is None or cost < best[0]:
                    best = (cost, (ny_, nz_), (ty_, tz_), (n_lo, n_hi), (t_lo, t_hi))
            _, (ny_, nz_), (ty_, tz_), (n_lo, n_hi), (t_lo, t_hi) = best
            nc, tc = (n_lo + n_hi)/2, (t_lo + t_hi)/2
            c_w = np.array([(x0 + x1)/2, nc*ny_ + tc*ty_, nc*nz_ + tc*tz_])
            half = np.array([(x1 - x0)/2, (t_hi - t_lo)/2, (n_hi - n_lo)/2])
            phi = math.atan2(-ny_, nz_)                    # local y -> t, local z -> n: a rotation about x by phi
            if not every:
                fit = self._weld_box_clear(k, gi, c_w, phi, half, np.array([0., ny_, nz_]))
                if fit is None:
                    done.difference_update(int(i) for i in I)    # kept as it is; refitted once the body has left
                    continue
                c_w, half = fit
                n_hi = n_lo + 2*half[2]
            m.geom_pos[gi] = c_w + self.scene.weld_origin_in_base
            m.geom_quat[gi] = (math.cos(phi/2), math.sin(phi/2), 0., 0.)
            m.geom_size[gi] = half
            m.geom_rbound[gi] = float(np.linalg.norm(half))
            m.geom_aabb[gi] = (0., 0., 0., *half)
            # the box's error: the empty space it puts above the metal (along its outward axis n, per 0.25 mm strip
            # across the joint, from the box's face down to metal present in every slice of the segment)
            qn, qt = Y*ny_ + Z*nz_, Y*ty_ + Z*tz_
            en = abs(ny_)*hy + abs(nz_)*hz
            inside = (qn >= n_lo) & (qn <= n_hi) & (qt >= t_lo) & (qt <= t_hi)
            solid = (mat != TH.AIR).all(0) & inside
            strip = np.floor((qt - t_lo)/2.5e-4).astype(int)
            gap = 0.
            for b_ in np.unique(strip[inside]):
                top = (qn + en)[solid & (strip == b_)]
                gap = max(gap, n_hi - (float(top.max()) if top.size else n_lo))
            self.weld_collider_boxes[k] = (c_w.tolist(), phi, half.tolist(), float(np.sum((4*hy*hz)[inside & ~solid])), gap)
        if not every:
            self._weld_dirty -= done

    def _weld_box_clear(self, k, gi, c_w, phi, half, n_w):
        """Refit guard for a weld-metal box (segment k, geom gi): the new box may not reach into a rigid body deeper
        than the box in use did. A box is the hull of its segment's cells, coarser than the metal; resized in one step
        under a resting body it overlapped it, and MuJoCo threw the body off (the nozzle at up to 146 mm/s and 1.1 kN).
        The pool keeps liquid out of a body's space (pool.Pool.obstacle); this keeps the box out of it too. The outer
        face (along n) is pulled in by the excess, or, if the box would then be thinner than a cell, the box in use is
        kept and the segment refitted once the body has gone. Returns (c_w, half), or None to keep the box in use. Each
        pulled or kept refit is logged."""
        mj = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        if not hasattr(self, "_box_others"):
            bracket = self.scene.bodies.get("bracket", -9)
            plate_mj = int(m.geom_bodyid[gi])
            self._box_others = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) != plate_mj
                                         and int(self._g2b[g]) != bracket
                                         and int(m.geom_type[g]) != int(mj.mjtGeom.mjGEOM_PLANE)
                                         and ((int(m.geom_contype[gi]) & int(m.geom_conaffinity[g]))
                                              or (int(m.geom_contype[g]) & int(m.geom_conaffinity[gi])))], int)
        b = int(m.geom_bodyid[gi])
        Xb, pb = d.xmat[b].reshape(3, 3), d.xpos[b].copy()
        Rl = np.array([[1., 0., 0.], [0., math.cos(phi), -math.sin(phi)], [0., math.sin(phi), math.cos(phi)]])
        G = self._box_others
        cw_new = pb + Xb @ (c_w + self.scene.weld_origin_in_base)
        reach = float(np.linalg.norm(half)) + 2e-3
        near = ((np.linalg.norm(d.geom_xpos[G] - cw_new, axis=1) <= m.geom_rbound[G] + reach)
                | (np.linalg.norm(d.geom_xpos[G] - d.geom_xpos[gi], axis=1) <= m.geom_rbound[G] + m.geom_rbound[gi] + 2e-3))
        cands = [int(g) for g in G[near]]
        if not cands:
            return c_w, half
        saved = (d.geom_xpos[gi].copy(), d.geom_xmat[gi].copy(), m.geom_size[gi].copy(), float(m.geom_rbound[gi]),
                 m.geom_aabb[gi].copy())
        ft = np.zeros(6)
        d_old = {g: mj.mj_geomDistance(m, d, gi, g, .05, ft) for g in cands}

        def place(c, hh):
            d.geom_xpos[gi] = pb + Xb @ (c + self.scene.weld_origin_in_base)
            d.geom_xmat[gi] = (Xb @ Rl).ravel()
            m.geom_size[gi] = hh
            m.geom_rbound[gi] = float(np.linalg.norm(hh))
            m.geom_aabb[gi] = (0., 0., 0., *hh)

        def restore():
            d.geom_xpos[gi], d.geom_xmat[gi], m.geom_size[gi] = saved[0], saved[1], saved[2]
            m.geom_rbound[gi], m.geom_aabb[gi] = saved[3], saved[4]

        pulled = 0.
        log = getattr(self, "weld_box_guard", None)
        if log is None:
            log = self.weld_box_guard = []
        for _ in range(4):
            place(c_w, half)
            worst, who = 0., None
            for g in cands:
                dn = mj.mj_geomDistance(m, d, gi, g, .05, ft)
                excess = min(d_old[g], 0.) - dn
                if excess > 1e-5 and excess > worst:
                    worst, who = excess, g
            if worst == 0.:
                restore()
                if pulled > 0. and len(log) < 500:
                    log.append(dict(tick=self.tick, segment=k, action="outer face pulled in", pulled_mm=pulled*1e3))
                return c_w, half
            if 2*half[2] - worst < 2.5e-4:
                break
            half = half.copy()
            half[2] -= worst/2
            c_w = c_w - n_w*worst/2
            pulled += worst
        restore()
        if len(log) < 500:
            log.append(dict(tick=self.tick, segment=k, action="kept (a body rests where the new box would reach)",
                            body=int(self._g2b[who]) if who is not None else None))
        return None

    @property
    def bracket_in_grid(self):
        """True while the thermal grid holds the bracket's cells (the bracket at or near its fit-up pose, or bonded)."""
        from .process import BODY_WEB
        return self.process is None or BODY_WEB not in self.process.detached

    def _bracket_grid(self):
        """The thermal grid is fixed in the base plate's frame and holds the bracket at its fit-up pose. While the
        bracket is not bonded and has left that pose (beyond the bond's own fit-up tolerance, bonds.FIT_MM / FIT_DEG)
        its cells are taken out of the grid -- the arc, the wire and touch sensing then meet the rigid bracket's
        collision box, and the cameras draw the rigid body -- and they are put back once it is within 80 % of the
        tolerance again. Otherwise a moved bracket leaves its lower 14 mm standing in the grid, and the arc welds to
        metal that is not there."""
        from .process import BODY_WEB
        if "bracket" not in self.scene.bodies or self.bond.active:
            return
        if self.bracket_in_grid and not self.bond._at_fit_up():
            n = self.process.detach_part(BODY_WEB)
            self.events.append(dict(tick=self.tick, event="bracket_left_fit_up", cells=n))
        elif not self.bracket_in_grid and self.bond._at_fit_up(scale=.8):
            n = self.process.attach_part(BODY_WEB)
            self.events.append(dict(tick=self.tick, event="bracket_back_at_fit_up", cells=n))

    def settle(self, seconds):
        for _ in range(int(round(seconds/self.config.rigid_dt))):
            self.step()

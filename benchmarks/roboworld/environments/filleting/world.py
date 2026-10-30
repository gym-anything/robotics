"""Coupled world: two FR3 arms and the fillet knife (Newton 1.5 -> MuJoCo, 1 kHz rigid tick) and the workpiece
(tissue.py explicit FEM with the knife-edge law, topology.py crack insertion, bodies.py rigid bones).

Coupling, every rigid tick:
  rigid -> tissue  every rigid collider that can touch the workpiece (knife blade segments and handle, all finger
                   and insert boxes, the board, pans, bowl) is a kinematic shape of the tissue solver; its pose at
                   the start of the tick and its twist over the tick come from the MuJoCo state, so within the
                   tissue substeps it moves exactly as the rigid body moved;
  tissue -> rigid  the tissue's contact, friction and edge reactions on each shape are summed per rigid body and
                   applied as an external wrench (body_f, about the body COM) during the next tick, through a
                   first-order filter with time constant REACTION_TAU standing in for pad/handle compliance (the
                   label_application precedent: an explicit one-tick coupling to 20-30 g finger links is otherwise
                   an unstable spring loop). Reported in the ledger as coupling lag.
The knife's edge samples are the blade segments' edges (knife.py) at the start of the tick with their velocities.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import layout as LY
from . import params as PR
from .scene import Scene, HANDLE_L, HANDLE_W, HANDLE_H

REACTION_TAU = 5e-3            # s, filter on tissue -> rigid reactions (label_application precedent)
PAD_SOLREF_TIMECONST = .005    # s, MuJoCo contact time constant of the fingertip pads (label_application)
PAD_SOLREF_REF_MASS = .0291    # kg, FR3 finger link that time constant was anchored on
FINGER_LIMIT_MARGIN = 1e-3     # m, finger joint lower limit below the pads' meeting point
FRANKA_HAND_FINGER_SPEED = .05  # m/s per finger (Franka Hand datasheet)
# tissue mesh edge per workpiece: 3.5 mm (E2: +3 % cut plateau at 2.5 mm) for the trout (73k tets), its fillet and the
# pork tenderloin (150k tets, 146x wall/sim contended); 5 mm for the 1 kg salmon half side (279k tets at 3.5 mm ran
# 356x wall/sim contended; ~96k at 5 mm)
MESH_H = {"coupon": 3.5e-3, "trout": 3.5e-3, "trout_fillet_rib_strip": 3.5e-3, "salmon": 5e-3, "tenderloin": 3.5e-3}
KNIFE_PINCH_N = .5              # N per finger: a hand holds the knife while both its fingers press on the handle
KNIFE_TOUCH_N = .2              # N: the robot touches the knife (MuJoCo contact normal force)
KNIFE_TISSUE_N = .05            # N: the knife lies on tissue (tissue and bone load on its colliders)
KNIFE_RELEASE_TICKS = 3         # ticks without that pinch before the knife is free
KNIFE_GRIP_TICKS = 5            # ticks of pinch before a free knife is gripped again
KNIFE_OFF_TISSUE_TICKS = 100    # ticks off tissue (in the air, on the board) before an unheld knife goes back to MuJoCo
KNIFE_GRIP_MAX_PEN = 2e-3       # m: MuJoCo's knife may overlap the robot at most this much when it takes the knife back


@dataclass
class Config:
    preset: str = "A"
    task: str = "A2"
    seed: int = 0
    robot: str = None
    device: str = "cuda:0"
    rigid_dt: float = 1e-3
    tissue_dt: float = None        # None: 2e-5 s (coupon), 1e-5 s (anatomical workpieces: the skin's in-plane stiffness
                                   # on thin surface triangles needs it; at 2e-5 s its stability mass would be ~110 g)
    mesh_h: float = None           # tissue mesh edge (m); None: MESH_H per workpiece (the element count is held near
                                   # the trout's ~73k so every preset runs within the 200x wall/sim budget)
    split_every: int = 10
    tearing: bool = True           # flesh tears where pulled apart (tearing.py); False: only blades separate tissue
    contact_capacity: int = 3000
    solver_iterations: int = 60
    solver_impratio: float = 10.       # MuJoCo friction/normal impedance ratio (barcode_scanning E7: at 1 a 17.5 N pinch
                                       # let a 214 g scanner creep 0.95 mm/s, at 10 0.05 mm/s; MuJoCo's remedy for slip
                                       # with elliptic cones)
    solver_noslip_iterations: int = 50  # MuJoCo no-slip post-solver (barcode_scanning E7: creep 0.10 -> 0.020 mm/s)
    # its convergence tolerance. At MuJoCo's default 1e-6 the post-solver stopped after ~9 iterations whatever the cap
    # (10, 50 and 200 gave bit-identical holds) and the held knife turned in the grip at 0.0012 deg/s under its own
    # weight (its tip 0.14 mm in 30 s at rest); at 1e-10 it takes ~31 iterations and the turn is 3e-6 deg in 3 s, at
    # 1e-14 none (E11). Cost: ~1 ms per tick of the rigid step
    solver_noslip_tolerance: float = 1e-10
    # s, contact time constant of the knife-handle inserts. A 4 mm TPU liner deflects micrometres under the 35 N pinch;
    # with the rubber pads' ~7 ms (0.5-0.7 mm under a 13 N contact point) the handle rocked in the channel and crept
    # 2.8 mm per fast wrist swing (J5 +-1.2 rad at 4.4 rad/s); the stiffest stable value at the
    # 1 ms step (MuJoCo: time constant >= 2 dt) leaves 1.2 mm. The friction impedance ratio and no-slip iterations
    # did not change it (10/10, 30/30, 100/50: 2.80-2.83 mm). None: the pads' value.
    insert_solref_timeconst: float = .002
    # the FR3/Panda collision reflex thresholds the deployment sets (libfranka setCollisionBehavior): 'default' = the
    # libfranka examples' general setting (10 N m nominal, 20 N m while accelerating), 'contact' = its contact-task
    # examples' 100 N m (force control, Cartesian impedance). A real cutting cell raises the thresholds: with the
    # default, the robot's own press cut through a 25 mm fish block stopped at 15 N m on joint 2
    collision_profile: str = "default"
    knife_in_hand: bool = True
    knife_start: str = None        # "hand" | "block" | None: automatic (task A0: the knife stands in the block's slot,
                                   # otherwise knife_in_hand decides)
    grip_force: float = 35.       # N per finger on the knife (70 N total: the Franka Hand's continuous rating)
    workpiece: str = None          # None (rigid only), "coupon", or a preset generator ("trout", ...)
    phantom_rig: bool = False      # ASSISTED owner-side runs only (demos/probes; the GAR backend rejects it): the knife
                                   # starts held by the phantom rig at layout.PHANTOM_KNIFE_PARK, not in the hand


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


def _quat_xyzw_from_R(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


class World:
    def __init__(self, config=None):
        self.config = c = config or Config()
        if c.task:
            from .variants import check_robot
            check_robot(c.robot, c.task)
        if c.collision_profile not in ("default", "contact"):
            raise ValueError(f"collision_profile must be 'default' or 'contact', got {c.collision_profile!r}")
        if c.tissue_dt is None:
            c.tissue_dt = 2e-5 if c.workpiece in (None, "coupon") else 1e-5
        if c.mesh_h is None:
            c.mesh_h = MESH_H.get(c.workpiece, 3.5e-3)
        self.newton, self.wp = newton, wp = require_engine()
        self.scene = Scene(c, newton, wp)
        b = self.scene.b
        with wp.ScopedDevice(c.device):
            self.model = b.finalize(device=c.device)
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy()
            ref[:] = (.02, 1.)
            mass = self.model.body_mass.numpy()
            sb = np.asarray(b.shape_body)
            finger_bodies = {fb[1] for fb in self.scene.finger_boxes}
            knife_bodies = {self.scene.bodies["knife_handle"]}
            insert_shapes = set(int(k) for k in self.scene.insert_shapes)
            for i in range(len(sb)):
                body = int(sb[i])
                if i in insert_shapes and c.insert_solref_timeconst:
                    ref[i] = (float(c.insert_solref_timeconst), 1.)
                elif body in finger_bodies:
                    # every finger collision shape gets the pad time constant scaled by sqrt(mass) (label_application)
                    ref[i] = (PAD_SOLREF_TIMECONST*math.sqrt(float(mass[body])/PAD_SOLREF_REF_MASS), 1.)
                elif body in knife_bodies:
                    ref[i] = (.005, 1.)
            self.model.mujoco.solref.assign(ref)
            lo = self.model.joint_limit_lower.numpy()
            labels = list(self.model.joint_label) if hasattr(self.model, "joint_label") else []
            qd_start = self.model.joint_qd_start.numpy()
            for j, lab_ in enumerate(labels):
                if "finger_joint" in lab_:
                    lo[qd_start[j]] -= FINGER_LIMIT_MARGIN
            self.model.joint_limit_lower.assign(lo)
            self.rig = self.scene.rig_pending.bind(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=c.contact_capacity,
                                                     broad_phase="sap", reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                      enable_multiccd=True, integrator="implicitfast",
                                                      cone="elliptic", solver="newton", iterations=c.solver_iterations,
                                                      ls_iterations=20, impratio=c.solver_impratio,
                                                      nconmax=c.contact_capacity, njmax=c.contact_capacity*3)
            mjm = self.solver.mj_model
            mjm.opt.noslip_iterations = int(c.solver_noslip_iterations)
            mjm.opt.noslip_tolerance = float(c.solver_noslip_tolerance)
            # the knife-handle inserts govern their contacts with the handle (MuJoCo geom priority): the pair's
            # compliance is the TPU liner's, not an average with the handle's (barcode_scanning: pad priority)
            g2s = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            ins = set(int(k) for k in self.scene.insert_shapes)
            for g in range(mjm.ngeom):
                if g < len(g2s) and int(g2s[g]) in ins:
                    mjm.geom_priority[g] = 1
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = {n: e for n, e in self.rig.joints.items()}
        self._reflex_until, self.refused_targets = {}, 0
        # each arm's 7 joints -> MuJoCo dof address (collision reflex); MuJoCo names are the Newton labels with
        # '/' -> '_' (barcode_scanning/world.py)
        mj_ = self.solver._mujoco
        mjm = self.solver.mj_model
        names = {mj_.mj_id2name(mjm, mj_.mjtObj.mjOBJ_JOINT, j): int(mjm.jnt_dofadr[j]) for j in range(mjm.njnt)}
        self._arm_mj_dofs = {}
        for name in self.joints:
            if "finger" in name:
                continue
            arm, jn = name.split("/", 1)
            hit = [dof for mname, dof in names.items() if mname and mname.startswith(arm + "_") and mname.endswith("_" + jn)]
            if len(hit) != 1:
                raise RuntimeError(f"collision reflex: no unique MuJoCo joint for {name}: {hit}")
            self._arm_mj_dofs.setdefault(arm, []).append((name, hit[0]))
        self._finger_speed = {j: float(hw["pads"].get("finger_speed_m_s", FRANKA_HAND_FINGER_SPEED))
                              for arm, hw in self.scene.arm_hw.items() for j in self.gripper_joints(arm)}
        self._ready_pose()
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        from .kinematics import ArmKinematics
        self.kin = ArmKinematics(self)
        self.targets = self.control.joint_target_q.numpy().copy()
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched = {}
        self.tick, self.fault = 0, None
        self.events = []
        self.assist = []                     # ASSISTED rig log (phantom knife, holds)
        self.phantom_knife = None
        self.driven_bones = {}               # ASSISTED phantom hands on bone bodies: tissue body index -> path
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.timing = dict(rigid_s=0., tissue_s=0., couple_s=0., topo_s=0., topo_record_s=0., topo_split_s=0.,
                           topo_tear_s=0.)
        self.reflexes = []
        # knife ownership: "held" (a hand grips it: MuJoCo integrates the rigid knife, the tissue solver's handle
        # follows), "free" (nothing grips it: the whole knife is integrated in the tissue solver, MuJoCo's handle
        # follows), "phantom" (ASSISTED rig)
        self.knife_mode = "free"
        self.knife_gripped = False
        self._no_pinch = 0
        self._pinch_ticks = 0
        self._blade_air = 0             # consecutive ticks MuJoCo owns the knife with no tissue load on its blade
        start = c.knife_start or ("block" if c.task == "A0" else ("hand" if c.knife_in_hand else "park"))
        if start not in ("hand", "block", "park"):
            raise ValueError(f"knife_start must be 'hand' or 'block', got {start!r}")
        self.knife_start = start
        if c.phantom_rig:
            self._phantom_park()
            self.knife_mode = "phantom"
        elif start == "hand":
            self._knife_in_hand()
            self.knife_mode = "held"
        elif start == "block":
            self._knife_in_block()
            self.knife_mode = "held"            # MuJoCo integrates it: it rests on rigid geometry, not on tissue
        self.tissue = None
        self.cut = None
        if c.workpiece:
            from .workpiece import build_tissue
            build_tissue(self, c.workpiece)
            if self.knife_mode in ("free", "held"):
                self._set_knife_mode(self.knife_mode, event=False)
            elif self.knife_mode == "phantom":
                # the ASSISTED rig holds the handle through a force-limited servo, as a hand holds a knife (a pose
                # followed with unbounded force squeezes flesh against bone and board until the solver goes
                # non-finite)
                ts = self.tissue
                fixed = ts.rb["fixed"].numpy()
                fixed[self.knife_body0] = 3
                ts.rb["fixed"].assign(fixed)
                ts.h_rb_fixed = fixed
                ts.invalidate_graphs()
        self._knife_mj_contacts(self.knife_mode == "held" and self.phantom_knife is None)
        self._mj_badqacc = self._mj_warnings()

    # ------------------------------------------------------------------ setup
    def _ready_pose(self):
        from robots.catalog import model
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        for name, hw in self.scene.arm_hw.items():
            pose = model(hw["model"]).to_dict()["poses"]["filleting"]
            for joint, v in pose.items():
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        self.model.joint_q.assign(q)
        self.control.joint_target_q.assign(t)

    def gripper_joints(self, arm):
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def joint_index(self, label):
        labels = list(self.model.joint_label)
        return labels.index(label)

    def _phantom_park(self):
        """ASSISTED runs: the knife handle is placed at layout.PHANTOM_KNIFE_PARK and the phantom rig holds it there
        (a constant pose) through the settle, so a demo path can start from exactly this pose (no teleport)."""
        from scipy.spatial.transform import Rotation
        pk = LY.PHANTOM_KNIFE_PARK
        p = np.asarray(pk["handle_com"], float)
        qk = Rotation.from_euler("z", pk["yaw_deg"], degrees=True).as_quat()
        q = self.model.joint_q.numpy()
        qs = self.model.joint_q_start.numpy()[self.joint_index("knife_free")]
        q[qs:qs + 3] = p - Rotation.from_quat(qk).as_matrix() @ self.scene.handle_ref_local
        q[qs + 3:qs + 7] = qk
        self.model.joint_q.assign(q)
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.phantom_knife = lambda t: (p, qk, np.zeros(3), np.zeros(3))
        self.assist.append(dict(tick=self.tick, rig="phantom_knife", on=True, note="parked at PHANTOM_KNIFE_PARK"))

    # ------------------------------------------------------------------ the FR3's protective stops (barcode_scanning)
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
                self._stop_arm(arm, q)
                if not self.reflexes or self.reflexes[-1]["tick"] < self.tick - 50 or self.reflexes[-1]["arm"] != arm:
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="velocity", joint=hit[0],
                                              speed_rad_s=hit[1], limit_rad_s=hit[2]))

    def external_joint_torques(self):
        """External torque on each arm joint (N m): MuJoCo's contact constraint forces mapped by J^T (joint limits,
        the grip's own squeeze and gravity are not external: the squeeze's two finger forces cancel on the arm), plus
        the tissue solver's wrenches on robot bodies (a finger or pad pressing on the fish) mapped by the bodies'
        Jacobians. Returns {arm: array(7)} in the order of self._arm_mj_dofs[arm]."""
        m, d = self.solver.mj_model, self.solver.mj_data
        mujoco = self.solver._mujoco
        tau = np.zeros(m.nv)
        if d.nefc:
            f = np.array(d.efc_force[:d.nefc])
            f[np.asarray(d.efc_type[:d.nefc]) < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0.
            if f.any():
                mujoco.mj_mulJacTVec(m, d, tau, f)
        wb = self.body_wrench
        if wb is not None and np.any(wb):
            if not hasattr(self, "_robot_mj_body"):
                self._robot_mj_body = self._robot_body_map()
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for nb_, mb_ in self._robot_mj_body.items():
                F, M = wb[nb_, :3].astype(float), wb[nb_, 3:].astype(float)
                if not (F.any() or M.any()):
                    continue
                com = np.asarray(d.xipos[mb_])
                mujoco.mj_jac(m, d, jp, jr, com, mb_)
                tau += jp.T @ F + jr.T @ M
        return {arm: tau[[dof for _, dof in dofs]] for arm, dofs in self._arm_mj_dofs.items()}

    def _robot_body_map(self):
        """Newton body index -> MuJoCo body id for the robots' bodies (Newton labels with '/' -> '_')."""
        mujoco = self.solver._mujoco
        m = self.solver.mj_model
        out = {}
        for i, lab in enumerate(self.model.body_label):
            if not any(lab.startswith(arm + "/") for arm in self.rig.instances):
                continue
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lab.replace("/", "_"))
            if bid >= 0:
                out[i] = int(bid)
        return out

    def _joint_accel_limits(self):
        """Joint acceleration limits of the arms' motion generator by robot model (params JOINT_ACCEL_*); fingers
        keep the Franka Hand's speed-limited closing."""
        out = {}
        for arm, hw in self.scene.arm_hw.items():
            model = str(hw.get("model", ""))
            lims = PR.v(PR.JOINT_ACCEL_PANDA) if model.startswith("panda") else PR.v(PR.JOINT_ACCEL_FR3)
            k = 0
            for name in self.joints:
                if name.startswith(arm + "/") and "finger" not in name:
                    out[name] = float(lims[min(k, len(lims) - 1)])
                    k += 1
        return out

    def _stop_arm(self, arm, q):
        """A protective stop: the arm's targets and its motion generator freeze at the current configuration."""
        cur = self.control.joint_target_q.numpy()
        for name, spec in self.joints.items():
            if name.startswith(arm + "/") and "finger" not in name:
                self.targets[spec["q"]] = float(q[spec["q"]])
                cur[spec["q"]] = float(q[spec["q"]])
                if hasattr(self, "_cmd_v"):
                    self._cmd_v[spec["q"]] = 0.
        self.control.joint_target_q.assign(cur)

    def _torque_reflex(self):
        """FR3/Panda collision reflex (libfranka setCollisionBehavior, reference setting of the libfranka examples):
        an arm whose external joint torque exceeds the upper threshold on any joint stops (its targets freeze at the
        current configuration) and refuses new targets for REFLEX_RECOVERY_S; 20 N m while the arm's targets are still
        slewing, 10 N m once they have arrived (barcode_scanning/world.py, extended with the tissue's wrenches)."""
        ext = self.external_joint_torques()
        cur = self.control.joint_target_q.numpy()
        q = self.state.joint_q.numpy()
        lim_nominal, lim_accel = PR.v(PR.REFLEX_TORQUE_CONTACT_NM if self.config.collision_profile == "contact"
                                      else PR.REFLEX_TORQUE_NM)
        for arm, dofs in self._arm_mj_dofs.items():
            e = np.abs(ext[arm])
            moving = any(abs(float(self.targets[self.joints[n]["q"]] - cur[self.joints[n]["q"]])) > 1e-6 for n, _ in dofs)
            limit = lim_accel if moving else lim_nominal
            k = int(np.argmax(e))
            if e[k] > limit:
                self._reflex_until[arm] = self.time + PR.v(PR.REFLEX_RECOVERY_S)
                self._stop_arm(arm, q)
                if not self.reflexes or self.reflexes[-1]["tick"] < self.tick - 50 or self.reflexes[-1]["arm"] != arm:
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, kind="collision", joint=dofs[k][0],
                                              external_torque_nm=float(e[k]), limit_nm=float(limit),
                                              culprits=self._reflex_culprits(arm)))

    def _reflex_culprits(self, arm, top=6):
        """What pushed the arm when its collision reflex fired (diagnostic record, read-only): the largest MuJoCo
        contacts on the arm's bodies (both geoms by name, normal force N, penetration mm) and the tissue solver's
        largest forces on the arm's bodies (N)."""
        mujoco = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        if not hasattr(self, "_g2b"):
            self._mj_pinch(False)
        labels = list(self.model.body_label)
        mine = {i for i in self._robot_bodies if labels[i].startswith(arm + "/")}

        def gname(g):
            b = int(self._g2b[g])
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or (labels[b] if b >= 0 else f"static geom {g}")
        f6 = np.zeros(6)
        rows = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(self._g2b[c.geom[0]]), int(self._g2b[c.geom[1]])
            if b1 not in mine and b2 not in mine:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            rows.append(dict(a=gname(c.geom[0]), b=gname(c.geom[1]), normal_n=round(abs(float(f6[0])), 2),
                             penetration_mm=round(-float(c.dist)*1e3, 3)))
        rows.sort(key=lambda r: -r["normal_n"])
        tissue = []
        wb = self.body_wrench
        if wb is not None:
            for i in mine:
                f = float(np.linalg.norm(wb[i, :3]))
                if f > 1e-3:
                    tissue.append(dict(body=labels[i], force_n=round(f, 2)))
            tissue.sort(key=lambda r: -r["force_n"])
        return dict(mujoco_contacts=rows[:top], tissue_loads=tissue[:top])

    def _free_override(self):
        """While the knife is free, MuJoCo's handle is placed on the tissue solver's handle each tick (zero velocity):
        the knife's dynamics, including every stiff blade and handle contact, stay inside the substepped solver, and
        MuJoCo still sees the handle for the fingers' contacts (the grip detector) and the render."""
        from scipy.spatial.transform import Rotation
        st = self.tissue.body_states()
        k = self.knife_body0
        p, qk = st["p"][k], st["q"][k]
        j = self.joint_index("knife_free")
        qs = self.model.joint_q_start.numpy()[j]
        ds = self.model.joint_qd_start.numpy()[j]
        q = self.state.joint_q.numpy()
        q[qs:qs + 3] = p - Rotation.from_quat(qk).as_matrix() @ self.scene.handle_ref_local
        q[qs + 3:qs + 7] = qk
        # the knife's true velocity (Newton FREE joint: COM velocity and angular velocity, world frame): MuJoCo's
        # contacts with the fingers then see the knife's real motion, and a hand-back to MuJoCo is seamless
        kh = self.scene.bodies["knife_handle"]
        R = Rotation.from_quat(qk).as_matrix()
        com_w = q[qs:qs + 3] + R @ self.model.body_com.numpy()[kh].astype(float)
        v, om = st["v"][k], st["w"][k]
        qd = self.state.joint_qd.numpy()
        qd[ds:ds + 3] = v + np.cross(om, com_w - p)
        qd[ds + 3:ds + 6] = om
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)

    def pinch_forces(self, with_penetration=False):
        """Normal force (N) each finger of each hand exerts on the knife: {arm: [left finger, right finger]}. While a
        hand holds it (MuJoCo integrates the knife), from MuJoCo's contacts on the handle; otherwise (free: the tissue
        solver integrates it) from the knife's contact loads on each finger's colliders in the tissue solver. With
        with_penetration, also the knife's deepest overlap with the robot in MuJoCo's geometry (mm-scale gate for
        handing a free knife back to MuJoCo)."""
        out, pen = self._mj_pinch(True)
        if not with_penetration:
            return out
        if self.knife_mode != "held" and any(min(v) > KNIFE_PINCH_N for v in out.values()):
            pen = max(pen, self._handback_overlap())
        return out, pen

    def _mj_pinch(self, with_penetration):
        solver = self.solver
        mj = solver._mujoco
        m, d = solver.mj_model, solver.mj_data
        if not hasattr(self, "_g2b"):
            g2s = solver.mjc_geom_to_newton_shape.numpy()[0]
            sb = self.model.shape_body.numpy()
            self._g2b = np.where(g2s >= 0, sb[np.clip(g2s, 0, None)], -1)
            self._finger_side = {}
            for i, lab in enumerate(self.model.body_label):
                for arm in self.rig.instances:
                    if lab.startswith(f"{arm}/") and lab.endswith("leftfinger"):
                        self._finger_side[i] = (arm, 0)
                    elif lab.startswith(f"{arm}/") and lab.endswith("rightfinger"):
                        self._finger_side[i] = (arm, 1)
            labels = list(self.model.body_label)
            self._robot_bodies = {i for i, lab in enumerate(labels)
                                  if any(lab.startswith(arm + "/") for arm in self.rig.instances)}
        kh = self.scene.bodies["knife_handle"]
        out = {arm: [0., 0.] for arm in self.rig.instances}
        pen = 0.
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self._g2b[c.geom[0]], self._g2b[c.geom[1]]
            other = b2 if b1 == kh else (b1 if b2 == kh else None)
            if other is None:
                continue
            pen = max(pen, -float(c.dist))                 # any overlap of the knife (robot or rigid geometry)
            if other not in self._finger_side:
                continue
            mj.mj_contactForce(m, d, i, f6)
            arm, side = self._finger_side[other]
            out[arm][side] += abs(float(f6[0]))
        return (out, pen) if with_penetration else out

    def _handback_overlap(self):
        """Deepest overlap (m) of MuJoCo's knife with everything it collides with once MuJoCo integrates it: the robot
        AND the static geometry (a rigid blade handed back while still in the block's slot can start inside the slot
        walls, and MuJoCo's state then goes non-finite). Signed geometric distances, since a
        free knife's MuJoCo copy only has robot contacts."""
        mujoco = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        others = getattr(self, "_knife_partners", None)
        if others is None:
            kset = set(int(g) for g in self._knife_geoms)
            ct, ca = self._knife_geom_masks
            c0, a0 = int(ct[0]), int(ca[0])
            others = [g for g in range(m.ngeom) if g not in kset
                      and ((c0 & int(m.geom_conaffinity[g])) or (int(m.geom_contype[g]) & a0))]
            self._knife_partners = others
        ft = np.zeros(6)
        pen = 0.
        for g1 in self._knife_geoms:
            for g2 in others:
                pen = max(pen, -float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), .01, ft)))
        return pen

    def _knife_contacts(self):
        """(normal force of the robot on MuJoCo's knife, N; normal load of tissue and bones on the knife's colliders in
        the tissue solver over the last tick, N)."""
        mujoco = self.solver._mujoco
        m, d = self.solver.mj_model, self.solver.mj_data
        if not hasattr(self, "_g2b"):
            self._mj_pinch(False)
        kh = self.scene.bodies["knife_handle"]
        f6 = np.zeros(6)
        robot = 0.
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self._g2b[c.geom[0]], self._g2b[c.geom[1]]
            other = b2 if b1 == kh else (b1 if b2 == kh else None)
            if other is None or other not in self._robot_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            robot += abs(float(f6[0]))
        tissue = 0.
        load = getattr(self.tissue, "cload_last", None)
        if load is not None:
            k0, k1 = self.blade_shape0 - 1, self.blade_shape0 + self.scene.blade.segments      # handle, blade
            tissue = float(np.sum(load[k0:k1]))
        edge = getattr(self.tissue, "edge", None)
        if edge is not None and tissue < KNIFE_TISSUE_N:
            # the edge's own load on the tissue it is cutting (anchored samples, tissue.k_edge_law): a blade cutting
            # through material that the flanks no longer press (tissue.k_straddle) is on tissue all the same
            ld = edge["load"].numpy()
            tissue += float(np.linalg.norm(ld, axis=1).sum())
        return robot, tissue

    def _update_knife_mode(self):
        """Who integrates the knife. MuJoCo ('held') whenever a hand grips it, the robot touches it, or it is not on
        tissue: it then carries the whole knife's mass, its rigid blade meets the board, block and robot in MuJoCo's
        implicit contacts, and the tissue solver's blade follows and returns the tissue's loads. The tissue solver
        ('free') only while the knife lies on tissue with no robot touching it (dropped on the fish): it then
        integrates the flexible knife against the tissue, the bones and the rigid geometry. (If any unpinched knife
        belonged to the tissue solver, a hand closing on a knife in the block would meet two copies of it, one per
        solver, flinging the fingers open to 80 mm and driving MuJoCo's state non-finite.)"""
        f, pen = self.pinch_forces(with_penetration=True)
        pinched = any(min(v) > KNIFE_PINCH_N for v in f.values())
        robot, tissue = self._knife_contacts()
        mujoco_owns = pinched or robot > KNIFE_TOUCH_N or tissue < KNIFE_TISSUE_N
        self.knife_gripped = pinched
        self._blade_air = self._blade_air + 1 if (self.knife_mode == "held" and tissue < KNIFE_TISSUE_N) else 0
        if self.knife_mode == "held":
            self._no_pinch = 0 if mujoco_owns else self._no_pinch + 1
            if self._no_pinch >= KNIFE_RELEASE_TICKS:
                self._set_knife_mode("free")
        else:
            # handed back to MuJoCo only while MuJoCo's rigid copy barely overlaps the robot and the static geometry:
            # MuJoCo would otherwise resolve a deep overlap as a blow. A hand taking the knife takes it within
            # KNIFE_GRIP_TICKS; a knife that has only left the tissue (bouncing, or lying on the board) stays with the
            # tissue solver for KNIFE_OFF_TISSUE_TICKS, which integrates it against the rigid geometry too: handed back
            # after 5 ticks, a knife dropped on the trout changed owner 11 times in 0.51 s
            taken = pinched or robot > KNIFE_TOUCH_N
            self._pinch_ticks = self._pinch_ticks + 1 if mujoco_owns else 0
            need = KNIFE_GRIP_TICKS if taken else KNIFE_OFF_TISSUE_TICKS
            if self._pinch_ticks >= need and max(pen, self._handback_overlap()) <= KNIFE_GRIP_MAX_PEN:
                self._set_knife_mode("held")

    def _set_knife_mode(self, mode, event=True):
        ts = self.tissue
        fixed = ts.rb["fixed"].numpy()
        fixed[self.knife_body0] = 2 if mode == "held" else 0
        # the blade's segments are dynamic in every mode but MuJoCo-owned-in-air (workpiece._blade_in_air)
        fixed[self.knife_body0 + 1:self.knife_body0 + 1 + self.scene.blade.segments] = 0
        self._blade_rigid = False
        self._blade_air = 0
        ts.rb["fixed"].assign(fixed)
        ts.rb["knife_held"].assign(np.array([1 if mode == "held" else 0], np.int32))
        ts.h_rb_fixed = fixed
        ts.invalidate_graphs()
        if mode == "held":
            # MuJoCo takes the knife where and as fast as the tissue solver has it (placed there this tick)
            self.tissue_filtered[self.scene.bodies["knife_handle"]] = 0.
        self.knife_mode = mode
        self._knife_mj_contacts(mode == "held" and self.phantom_knife is None)
        self._no_pinch = 0
        self._pinch_ticks = 0
        if event:
            self.events.append(dict(tick=self.tick, event="knife_to_mujoco" if mode == "held" else "knife_to_tissue"))

    def _knife_in_block(self):
        """Task A0's initial condition: the fillet knife stands in the block's slot at the slot's pose
        (scene.knife_block), not gripped. It rests on rigid geometry, not on tissue, so MuJoCo integrates it against
        the block's slot (knife_mode 'held' means MuJoCo owns it, not that a hand holds it)."""
        from scipy.spatial.transform import Rotation
        heel, Rk = self.scene.block_knife_pose
        origin = heel - Rk @ np.array([HANDLE_L/2, 0., -self.scene.blade.H])
        q = self.model.joint_q.numpy()
        qs = self.model.joint_q_start.numpy()[self.joint_index("knife_free")]
        q[qs:qs + 3] = origin
        q[qs + 3:qs + 7] = Rotation.from_matrix(Rk).as_quat()
        self.model.joint_q.assign(q)
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)

    def _knife_mj_contacts(self, dynamic):
        """MuJoCo integrates the knife only while a hand holds it. In the other modes (free: the tissue solver
        integrates it; phantom: the rig places it) MuJoCo's copy is placed kinematically each tick, and a kinematic
        body in contact with static geometry is an immovable object against an immovable object: MuJoCo's solver
        answers a blade placed into the board with unbounded accelerations and resets its whole state (in an A2
        demonstration the robots jumped to their zero pose and the tissue exploded). So outside 'held', the knife's
        MuJoCo geoms collide with the robots only (grip detection, the robot feeling the knife); its contacts with the
        board, bench and block are the tissue solver's."""
        m = self.solver.mj_model
        if not hasattr(self, "_knife_geoms"):
            g2s = self.solver.mjc_geom_to_newton_shape.numpy()[0]
            sb = self.model.shape_body.numpy()
            gb = np.where(g2s >= 0, sb[np.clip(g2s, 0, None)], -1)
            labels = list(self.model.body_label)
            kh = self.scene.bodies["knife_handle"]
            self._knife_geoms = np.nonzero(gb == kh)[0]
            self._knife_geom_masks = (m.geom_contype[self._knife_geoms].copy(), m.geom_conaffinity[self._knife_geoms].copy())
            robot = [g for g in range(m.ngeom) if gb[g] >= 0 and any(labels[gb[g]].startswith(a + "/") for a in self.rig.instances)]
            self._robot_mask = int(np.bitwise_or.reduce(m.geom_contype[robot])) if robot else 0
        if dynamic:
            m.geom_contype[self._knife_geoms], m.geom_conaffinity[self._knife_geoms] = self._knife_geom_masks
        else:
            # contacts with the robots only (implicit in MuJoCo, and how the grip is detected). No MuJoCo contacts at
            # all, the fingers meeting the free knife only in the tissue solver, is not stable: that explicit,
            # one-tick-lagged coupling of a light finger against a stiff contact built 1,057 N on a finger and drove
            # MuJoCo non-finite during a legal grasp; the robot-only contacts are stable (their cost: the copy is
            # kinematic, so a finger cannot push a free knife in MuJoCo)
            m.geom_contype[self._knife_geoms] = 0
            m.geom_conaffinity[self._knife_geoms] = self._robot_mask

    def _mj_warnings(self):
        """MuJoCo's count of unstable-state warnings (non-finite or huge acceleration, position or velocity)."""
        mujoco = self.solver._mujoco
        d = self.solver.mj_data
        W = mujoco.mjtWarning
        return sum(int(d.warning[int(k)].number) for k in (W.mjWARN_BADQACC, W.mjWARN_BADQPOS, W.mjWARN_BADQVEL))

    def _phantom_override(self):
        """While the phantom rig holds the knife, MuJoCo's handle is placed kinematically each tick (zero velocity) on
        the tissue solver's handle, which the rig's force-limited servo moves toward its path (params RIG_*), so the
        rendered handle, the robots' contacts and the servoed handle agree. Without a workpiece, on the rig's pose."""
        from scipy.spatial.transform import Rotation
        if self.tissue is not None and self.tissue.h_rb_fixed[self.knife_body0] == 3:
            st = self.tissue.body_states()
            p, qk = st["p"][self.knife_body0], st["q"][self.knife_body0]
        else:
            p, qk, _, _ = self.phantom_knife(self.tick*self.config.rigid_dt)
        com = self.scene.handle_ref_local
        j = self.joint_index("knife_free")
        qs = self.model.joint_q_start.numpy()[j]
        ds = self.model.joint_qd_start.numpy()[j]
        q = self.state.joint_q.numpy()
        q[qs:qs + 3] = np.asarray(p, float) - Rotation.from_quat(np.asarray(qk, float)).as_matrix() @ com
        q[qs + 3:qs + 7] = qk
        qd = self.state.joint_qd.numpy()
        qd[ds:ds + 6] = 0.
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)

    def _knife_in_hand(self):
        """Authored initial condition (disclosed): the knife handle is spawned in the right
        hand's insert channel, edge down, handle axis along the hand's x axis (the blade toward the board centre),
        the fingers at the contact opening and in force mode at `grip_force` per finger. world.settle() then lets
        the grasp come to rest (0.5 s) before the episode starts."""
        from scipy.spatial.transform import Rotation
        fp = self.rig.frame_pose("right/tcp", self.state)
        p = np.asarray(fp["position_m"], float)
        w, x, y, z = fp["orientation_wxyz"]
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        xh, yh, zh = R[:, 0], R[:, 1], R[:, 2]
        to_board = np.array([LY.BOARD["center"][0], LY.BOARD["center"][1], LY.BOARD_TOP_Z]) - p
        xk = xh if np.dot(xh, to_board) >= 0 else -xh
        zk = -zh
        yk = np.cross(zk, xk)
        Rk = np.column_stack([xk, yk, zk])
        qk = _quat_xyzw_from_R(Rk)
        q = self.model.joint_q.numpy()
        j = self.joint_index("knife_free")
        qs = self.model.joint_q_start.numpy()[j]
        q[qs:qs + 3] = p
        q[qs + 3:qs + 7] = qk
        # fingers at the insert contact opening
        for name in self.gripper_joints("right"):
            q[self.joints[name]["q"]] = self.scene.insert_contact_q - 2e-4
            self.targets[self.joints[name]["q"]] = 0.
        self.model.joint_q.assign(q)
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        t = self.control.joint_target_q.numpy()
        for name in self.gripper_joints("right"):
            t[self.joints[name]["q"]] = q[self.joints[name]["q"]]
        self.control.joint_target_q.assign(t)
        self.targets = t.copy()
        self.grip_force["right"] = self.config.grip_force
        for name in self.gripper_joints("right"):
            self.grasp_latched[("right", name)] = True

    # ------------------------------------------------------------------ control
    def set_targets(self, targets):
        """Joint position targets. An arm stopped by its collision reflex keeps its frozen targets (and refuses new
        ones, fingers excepted) for REFLEX_RECOVERY_S: the real FR3 stays stopped until an error recovery."""
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
        """Franka Hand force-mode grasp: constant inward force per finger (N), fingers servo to closed."""
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) \
                or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]

    def _rigid_step(self):
        c = self.config
        cur = self.control.joint_target_q.numpy()
        if not hasattr(self, "_cmd_v"):
            self._cmd_v = np.zeros(len(cur))
            self._accel = self._joint_accel_limits()
        dt = c.rigid_dt
        for name, spec in self.joints.items():
            j = spec["q"]
            vlim = spec["velocity_limit"]
            if name in self._finger_speed:
                vlim = min(vlim, self._finger_speed[name])
            e = float(self.targets[j] - cur[j])
            amax = self._accel.get(name)
            if amax is None:
                slew = vlim*dt
                cur[j] += np.clip(e, -slew, slew)
                continue
            # rate-limited motion generator (libfranka limit_rate): the commanded joint position moves toward the target
            # at no more than the joint's speed limit, changing speed by at most amax per second, and brakes to arrive
            # (the fastest speed from which it can still stop at the target). Slewing at the speed limit with an instant
            # speed step makes the position servo overshoot the limit and trip the velocity reflex (E15).
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
            if force:
                for name in self.gripper_joints(arm):
                    spec = self.joints[name]
                    key = (arm, name)
                    if key not in self.grasp_latched:
                        # Franka Hand grasp: close at the commanded speed until the finger is BLOCKED (behind its
                        # command and stalled), then squeeze with the force. Latching on the lag alone would also catch
                        # a finger still finishing its previous move, which then shoots shut at ~400 mm/s under 35 N
                        lag = float(q_now[spec["q"]] - cur[spec["q"]])
                        stalled = abs(float(qd_now[spec["dof"]])) < 5e-3
                        travel_done = abs(float(self.targets[spec["q"]] - cur[spec["q"]])) < 1e-6
                        if (lag > 5e-4 and stalled) or travel_done:
                            self.grasp_latched[key] = True
                    if key in self.grasp_latched:
                        cur[spec["q"]] = q_now[spec["q"]]
                        jf[spec["dof"]] = -force
        self.control.joint_target_q.assign(cur)
        self.control.joint_f.assign(jf)
        if self.phantom_knife is not None:
            self._phantom_override()
        elif self.knife_mode == "free" and self.tissue is not None:
            self._free_override()
        self.state.clear_forces()
        self.state.body_f.assign(self.body_wrench)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, c.rigid_dt)
        bad = self._mj_warnings()
        if bad > getattr(self, "_mj_badqacc", 0):
            # MuJoCo found a non-finite or huge acceleration and reset its state (the robots would jump to their
            # zero pose): a fault, never a silent continuation
            self._mj_badqacc = bad
            raise FloatingPointError(f"MuJoCo reported an unstable state (non-finite or huge acceleration, position or "
                                     f"velocity; MuJoCo resets) at tick {self.tick}")
        self.state, self.next_state = self.next_state, self.state
        self.tick += 1
        self._velocity_reflex()
        self._torque_reflex()

    # ------------------------------------------------------------------ stepping
    def step(self):
        """One rigid tick (rigid_dt) plus the tissue over the same interval."""
        if self.fault:
            raise RuntimeError(self.fault)
        try:
            t0 = time.perf_counter()
            q_before = self.state.body_q.numpy().astype(float)
            self._rigid_step()
            t1 = time.perf_counter()
            self.timing["rigid_s"] += t1 - t0
            if self.tissue is not None:
                from .workpiece import tissue_tick
                tissue_tick(self, q_before)
                if self.knife_mode in ("held", "free"):
                    self._update_knife_mode()
            for attr in ("body_q", "body_qd", "joint_q"):
                if not np.isfinite(getattr(self.state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite rigid {attr} at tick {self.tick}")
            if self.tissue is not None and self.tick % 10 == 0:
                ts = self.tissue
                if not np.isfinite(ts.x.numpy()[:ts.n]).all():
                    raise FloatingPointError(f"Non-finite tissue node positions at tick {self.tick}")
                if ts.nb and not np.isfinite(ts.rb["p"].numpy()[:ts.nb]).all():
                    raise FloatingPointError(f"Non-finite tissue rigid bodies at tick {self.tick}")
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    # ------------------------------------------------------------------ ASSISTED rigs (owner-side, never policy actions)
    def set_phantom_knife(self, fn):
        """ASSISTED phantom knife rig: `fn(t) -> (position (3), quat xyzw (4), velocity (3), angular velocity (3))`
        of the knife HANDLE (its grip box centre, scene.handle_ref_local) at sim time t. The tissue solver's handle body and MuJoCo's handle both
        follow it (MuJoCo kinematically, _phantom_override). A path must start at the knife's current pose (worlds built
        with Config(phantom_rig=True) start at layout.PHANTOM_KNIFE_PARK); None hands the knife back to the dynamics
        (it then falls unless a hand holds it). Recorded in world.assist."""
        self.phantom_knife = fn
        self._knife_mj_contacts(fn is None and self.knife_mode == "held")
        self.assist.append(dict(tick=self.tick, rig="phantom_knife", on=fn is not None))

    def hold_body(self, name, on=True):
        """ASSISTED phantom hold: fix a rigid bone body of the workpiece (e.g. the head) in place."""
        ts = self.tissue
        i = ts.body_names.index(name)
        fixed = ts.rb["fixed"].numpy()
        if fixed[i] == 2:
            raise ValueError("driven bodies cannot be held")
        fixed[i] = 1 if on else 0
        ts.rb["fixed"].assign(fixed)
        if on:
            v = ts.rb["v"].numpy(); w_ = ts.rb["w"].numpy(); v[i] = 0; w_[i] = 0
            ts.rb["v"].assign(v); ts.rb["w"].assign(w_)
        ts.h_rb_fixed = fixed
        ts.invalidate_graphs()
        self.assist.append(dict(tick=self.tick, rig="hold", body=name, on=on))

    def drive_body(self, name, fn):
        """ASSISTED phantom hand: a rigid bone body of the workpiece follows `fn(t) -> (position (3), quat xyzw (4),
        velocity (3), angular velocity (3))` of its reference point on the world clock (World.time); the flesh tied to
        it follows through the physics. fn=None releases it to the dynamics. Recorded in world.assist."""
        ts = self.tissue
        i = ts.body_names.index(name)
        if i < self.assembly["n_knife_bodies"]:
            raise ValueError("knife bodies are driven by the phantom knife rig, not drive_body")
        fixed = ts.rb["fixed"].numpy()
        fixed[i] = 2 if fn is not None else 0
        ts.rb["fixed"].assign(fixed)
        ts.h_rb_fixed = fixed
        if fn is None:
            self.driven_bones.pop(i, None)
        else:
            self.driven_bones[i] = fn
        ts.invalidate_graphs()
        self.assist.append(dict(tick=self.tick, rig="drive_body", body=name, on=fn is not None))

    def settle(self, seconds=.5):
        for _ in range(int(round(seconds/self.config.rigid_dt))):
            self.step()

    # ------------------------------------------------------------------ queries
    @property
    def time(self):
        """World clock (s) since the world was built, settle included: the clock the phantom rig is driven by (the
        environment's sim_time restarts at 0 after the settle)."""
        return self.tick*self.config.rigid_dt

    def body_pose(self, name):
        from scipy.spatial.transform import Rotation
        q = self.state.body_q.numpy()[self.scene.bodies[name]]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def tcp(self, arm):
        return self.rig.frame_pose(f"{arm}/tcp", self.state)

    @property
    def sim_time(self):
        return self.tick*self.config.rigid_dt

    def scale_reading_g(self):
        """Bench scale reading (g): the normal load the tissue put on the weighing platform over the last tick / g
        (presets B, C); None where there is no scale."""
        ts = self.tissue
        names = [sh.get("name") for sh in self.tissue_shapes]
        if ts is None or "scale_platform" not in names:
            return None
        return float(ts.cload_last[names.index("scale_platform")]/9.81*1e3)

    def knife_in_grip(self):
        """Knife handle pose relative to the right TCP: (offset m (3), angle between handle axis and hand x, deg)."""
        from scipy.spatial.transform import Rotation
        fp = self.tcp("right")
        p = np.asarray(fp["position_m"])
        w, x, y, z = fp["orientation_wxyz"]
        Rh = Rotation.from_quat([x, y, z, w]).as_matrix()
        ph, Rk = self.body_pose("knife_handle")
        off = Rh.T @ (ph - p)
        ang = math.degrees(math.acos(min(1., abs(float(np.dot(Rk[:, 0], Rh[:, 0]))))))
        return off, ang

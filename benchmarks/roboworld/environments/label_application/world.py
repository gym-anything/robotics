"""Coupled world: rigid robots/items (Newton 1.5 + MuJoCo Warp) and the label shell (labelsim).

Coupling per label step (every `label_substeps` rigid ticks):
  rigid -> label  collider poses from body_q; rigid normal loads on finger pads/tools
                  (sandwich rule: pressure = rigid pad contact force / label area under the pad)
  label -> rigid  label contact/bond/friction reactions applied as external wrenches
                  (body_f, world frame about the COM) during the following rigid ticks
The reaction is lagged by one label step and does not include compressive contact of
sandwiched vertices (that load is already carried by the rigid pad-substrate contact).
"""

import math
import time
import numpy as np

# Franka Hand (FR3 and Panda carry the same hand): 50 mm/s per finger, 70 N continuous grasp force
# (Franka datasheet, web-sourced). The shared control profiles give the FR3 fingers 0.2 m/s.
FRANKA_HAND_FINGER_SPEED = .05
PAD_SOLREF_TIMECONST = .005     # s, MuJoCo contact time constant of the fingertip pads on the FR3 finger (anchored below)
PAD_SOLREF_REF_MASS = .0291     # kg, the FR3 finger link that time constant was anchored on (MJCF/URDF inertial)
FINGER_LIMIT_MARGIN = 1e-3      # m, finger joint lower limit moved below the pads' meeting point
# "settled" is judged on net motion over this window (0.1 s at the 2 ms label step): the lagged
# label-rigid coupling leaves a zero-mean jitter of up to ~2 mm/s on a 10 g tube that an instantaneous
# speed threshold mistook for motion
SETTLE_WINDOW_STEPS = 50
from .config import Config
from .scene import Scene, TABLE
from . import labelsim as L, materials as M


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
            # MuJoCo contact dynamics with an explicit, stable time constant (MuJoCo default 0.02 s,
            # critical damping) instead of an approximate mapping of authored ke/kd, as in lockkey.
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy(); ref[:] = (.02, 1.)
            # fingertip pads: contact stiffness anchored to the pad compliance the label solver uses (rubber
            # 2 MPa over 2 mm = 1e9 Pa/m; over the 17.5 x 18.5 mm pad face two pads in series ~1.6e5 N/m). At the
            # 20 ms default the pads overlapped 1.46 mm under a 20 N grasp (~9 kN/m) and the fingers ran past
            # their joint limit, whose spring took ~7 N of the squeeze; MuJoCo contact stiffness scales as 1/tau^2,
            # so tau = 5 ms (>= 2 rigid steps, stable) gives the anchored pad-pair stiffness.
            # MuJoCo's contact stiffness also scales with the effective mass, so the same stiffness on a lighter
            # finger needs tau ~ sqrt(m): the Panda finger (14.1 g against 29.1 g) at 5 ms was half as stiff, its
            # URDF finger meshes met the partner finger at the 20 ms default, and the grasp oscillated between the
            # pad contact and the joint limit (0 <-> -2.3 mm, pinch 2 <-> 21 N every 32 ms).
            # Every collision shape of a finger body gets the pad's time constant (the Panda's URDF finger mesh
            # includes the fingertip and meets the partner finger at closure too): pad boxes alone at the scaled
            # time constant still oscillated (0 <-> -1.7 mm); with all finger shapes stiff the grasp holds still
            # and the squeeze splits ~16 N pads / ~4 N finger meshes on the Panda, 20 N pads on the FR3.
            self.pad_shapes = self._pad_shapes()
            mass = self.model.body_mass.numpy()
            sb = np.asarray(self.scene.b.shape_body)
            pad_bodies = {int(sb[s]) for s in self.pad_shapes}
            for i in range(len(sb)):
                body = int(sb[i])
                if body in pad_bodies:
                    ref[i] = (PAD_SOLREF_TIMECONST*math.sqrt(float(mass[body])/PAD_SOLREF_REF_MASS), 1.)
            self.model.mujoco.solref.assign(ref)
            # the Franka fingers' 0 mm lower limit is where the pads meet: with compliant pads the true stop is the
            # pad contact, so the limit is moved 1 mm lower and never carries grasp force (it did: ~7 N of 20 N)
            lo = self.model.joint_limit_lower.numpy()
            self._finger_limit_margin(lo)
            self.model.joint_limit_lower.assign(lo)
            self.rig = self.scene.rig_pending.bind(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=c.contact_capacity,
                                                     broad_phase="sap", reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            # MuJoCo-C on CPU: ~2 arms + a few items is far cheaper than kernel launches on the GPU.
            self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                                      enable_multiccd=True,   # 4-point convex contacts: cylinders rest flat
                                                      integrator="implicitfast", cone="elliptic",
                                                      solver="newton", iterations=c.solver_iterations,
                                                      ls_iterations=20, nconmax=c.contact_capacity,
                                                      njmax=c.contact_capacity*3)
            import mujoco
            self._mujoco = mujoco
            self._geom_shape = np.asarray(self.solver.mjc_geom_to_newton_shape.numpy()).reshape(-1)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = {n: e for n, e in self.rig.joints.items()}
        # hand speed per gripper joint from the hand spec (Franka Hand: 50 mm/s per finger)
        self._finger_speed = {j: float(hw["pads"].get("finger_speed_m_s", FRANKA_HAND_FINGER_SPEED))
                              for arm, hw in self.scene.arm_hw.items() for j in self.gripper_joints(arm)}
        self._ready_pose()
        self.state.joint_q.assign(self.model.joint_q)      # joint vector and body poses agree at t = 0
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        self.reflexes = []
        self.label_frozen = None             # set when a tear ends the episode (stop_on_tear)
        from collections import deque
        self.label_history = deque(maxlen=SETTLE_WINDOW_STEPS)
        self.tick, self.fault = 0, None      # tick counts rigid MuJoCo ticks
        self.steps = 0                         # environment ticks (label steps)
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.shape_body = np.asarray(b.shape_body)
        self.label = self._make_label()
        # Read-only damage record for the evaluator, kept beside label_history: the largest permanent set (|theta_rest|)
        # each hinge has carried. labelsim's set is a rest shape that bending the other way pulls back down; a crease
        # bent back is still a crease in paper. It is taken before and after every label step and changes no dynamics.
        self.set_peak = np.abs(self.label.theta_rest)
        self.timing = dict(rigid_s=0., label_s=0., couple_s=0.)
        self.fixture_log = []
        self.grip_force = {arm: 0. for arm in self.rig.instances}
        self.grasp_latched = {}          # (arm, finger joint) -> True once blocked: force-controlled hold

    def set_grip_force(self, arm, force):
        """Franka Hand force-mode grasp: constant inward force per finger (N), fingers servo to closed."""
        if arm not in self.rig.instances:
            raise ValueError(f"No arm {arm}")
        limit = float(self.joints[self.gripper_joints(arm)[0]]["effort_limit"])
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or not 0 <= force <= limit:
            raise ValueError(f"Grip force must be in [0, {limit}] N per finger")
        self.grip_force[arm] = float(force)
        if not force:
            for key in [k for k in self.grasp_latched if k[0] == arm]:
                del self.grasp_latched[key]

    # ------------------------------------------------------------------ setup
    def _ready_pose(self):
        """Ready pose from the shared robot spec (robots/models/<id>/robot.json poses.label_application)."""
        from robots.catalog import model
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        for name, hw in self.scene.arm_hw.items():
            pose = model(hw["model"]).to_dict()["poses"]["label_application"]
            for joint, v in pose.items():
                key = f"{name}/{joint}"
                q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        self.model.joint_q.assign(q)
        self.control.joint_target_q.assign(t)

    def gripper_joints(self, arm):
        """The arm's two gripper joint names (model groups.gripper)."""
        return [f"{arm}/{j}" for j in self.scene.arm_hw[arm]["gripper_joints"]]

    def _finger_limit_margin(self, lo):
        """Lower every finger joint's limit by FINGER_LIMIT_MARGIN (model DOF order, before the solver is built)."""
        labels = list(self.model.joint_label) if hasattr(self.model, "joint_label") else []
        qd_start = self.model.joint_qd_start.numpy()
        for j, lab_ in enumerate(labels):
            if "finger_joint" in lab_:
                lo[qd_start[j]] -= FINGER_LIMIT_MARGIN

    def _pad_shapes(self):
        """Collision shapes of the rubber fingertips (the hand spec's pad box, z = 45.25 mm in the Franka finger
        frame). A finger
        modelled by one collision mesh (Panda URDF) has no separate tip shape: its mesh carries the
        pinch load."""
        pads = {}
        b = self.scene.b
        by_body = {}
        for i, label in enumerate(b.shape_label):
            if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]:
                by_body.setdefault(b.shape_body[i], []).append(i)
                tf = np.asarray(b.shape_transform[i])
                if any(abs(tf[2] - hw["pads"]["finger_frame_offset_z_m"]) < 1e-4 for hw in self.scene.arm_hw.values()) \
                        and b.shape_type[i] == self.newton.GeoType.BOX:
                    pads[i] = b.body_label[b.shape_body[i]]
        for body, shapes in by_body.items():
            if not any(pads.get(i) for i in shapes):
                for i in shapes:
                    if b.shape_type[i] == self.newton.GeoType.MESH:
                        pads[i] = b.body_label[body]
        return pads

    def _make_label(self):
        spec = self.scene.label_spec
        stock = M.STOCKS[spec["stock"]]
        lab = L.LabelSim(stock, spec["md"], spec["cd"], spec["h"],
                         np.asarray(spec["origin"]) + np.array([0, 0, .5*stock.thickness]),
                         spec["axis_md"], spec["axis_cd"], colliders=self.scene.colliders,
                         newton_iters=self.config.newton_iters, newton_max=self.config.newton_max,
                         name=spec["fmt"])
        self._pose_colliders(first=True)
        self._set_pinch_faces()
        # the trailing edge is still on the printer's liner web: bonded at the liner's release energy
        liner = [i for i, col in enumerate(lab.colliders) if col.role == "liner"][0]
        col = lab.colliders[liner]
        u = lab.X[:, 0]
        trailing = np.nonzero(u <= spec["liner_bonded_md"] + 1e-9)[0]
        d, n, loc = col.sdf(lab.x[trailing])
        lab.bond[trailing] = liner
        lab.anchor[trailing] = loc + (n @ col.R)*(.5*stock.thickness)
        lab.anchor_n[trailing] = n @ col.R
        lab.phi[trailing] = 1.
        lab.bond_time[trailing] = 3600.
        return lab

    def _set_pinch_faces(self):
        """Each fingertip pad's pinching face is the local box face pointing at the partner pad."""
        pads = [c for c in self.scene.colliders if c.role == "pad"]
        for c in pads:
            arm = c.name.split("_")[0]
            other = [o for o in pads if o is not c and o.name.split("_")[0] == arm]
            if not other:
                continue
            d = c.R.T @ (other[0].p - c.p)
            c.pinch_axis = int(np.argmax(np.abs(d)))
            c.pinch_sign = float(np.sign(d[c.pinch_axis]))
            c.partner = other[0].name

    def _pose_colliders(self, first=False):
        q = self.state.body_q.numpy()
        for col in self.scene.colliders:
            if col.body >= 0:
                p = q[col.body, :3].astype(float)
                R = L.quat_to_matrix(q[col.body, 3:].astype(float))
                col.set_body_pose(R, p, first=first)
            elif first:
                col.set_body_pose(np.eye(3), np.zeros(3), first=True)

    # ------------------------------------------------------------------ stepping
    def set_targets(self, targets):
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
                # a new finger command ends a force-controlled hold (Franka Hand: move/grasp replace it)
                for key in [k for k in self.grasp_latched if k[1] == name]:
                    del self.grasp_latched[key]

    def _rigid_step(self):
        c = self.config
        cur = self.control.joint_target_q.numpy()
        for name, spec in self.joints.items():
            j = spec["q"]
            vlim = spec["velocity_limit"]
            if name in self._finger_speed:
                vlim = min(vlim, self._finger_speed[name])
            slew = vlim*c.rigid_dt
            cur[j] += np.clip(self.targets[j] - cur[j], -slew, slew)
        self.control.joint_target_q.assign(cur)
        jf = np.zeros(self.model.joint_dof_count, np.float32)
        q_now = self.state.joint_q.numpy()
        for arm, force in self.grip_force.items():
            if force:
                for name in self.gripper_joints(arm):
                    spec = self.joints[name]
                    # Franka Hand grasp(width, speed, force): the fingers close at the hand's speed (targets
                    # slew at the 50 mm/s per-finger datasheet limit) until blocked -- lagging the slewed
                    # target by > 0.5 mm (an object in the way) or at the end of the commanded travel --
                    # then the hand holds the commanded force under force control until the next command.
                    # The hold is latched: the position servo stops pulling (its target follows the finger,
                    # leaving only its damping) and the finger pushes with exactly `force`. A force scaled
                    # by finger speed (off above 20 mm/s) is a relay that chatters: the squeeze vanishes
                    # whenever the finger moves, pinch loads fall to 0 N in 12-68 % of samples and a held
                    # label never comes to rest.
                    key = (arm, name)
                    if key not in self.grasp_latched:
                        lag = float(q_now[spec["q"]] - cur[spec["q"]])
                        travel_done = abs(float(self.targets[spec["q"]] - cur[spec["q"]])) < 1e-6
                        if lag > 5e-4 or travel_done:
                            self.grasp_latched[key] = True
                    if key in self.grasp_latched:
                        cur[spec["q"]] = q_now[spec["q"]]
                        jf[spec["dof"]] = -force                                        # closing = -q
        self.control.joint_target_q.assign(cur)
        self.control.joint_f.assign(jf)
        self.state.clear_forces()
        self.state.body_f.assign(self.body_wrench)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, c.rigid_dt)
        self.state, self.next_state = self.next_state, self.state
        self.tick += 1
        self._velocity_reflex()

    def _velocity_reflex(self):
        """FR3/Panda joint-velocity reflex: a joint faster than 1.05x its datasheet limit stops its arm
        (targets frozen at the current pose), as the robot's controller does with a velocity-violation
        reflex. The targets slew at the limit, but the shared servo lags a ramping target and can
        overshoot when it catches up (measured up to 1.5x on full-range random targets)."""
        qd = self.state.joint_qd.numpy()
        q = self.state.joint_q.numpy()
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
                for name, spec in self.joints.items():
                    if name.startswith(arm + "/") and "finger" not in name:
                        self.targets[spec["q"]] = float(q[spec["q"]])
                if not self.reflexes or self.reflexes[-1]["tick"] < self.tick - 50 or self.reflexes[-1]["arm"] != arm:
                    self.reflexes.append(dict(tick=int(self.tick), arm=arm, joint=hit[0], speed_rad_s=hit[1],
                                              limit_rad_s=hit[2]))

    def _pad_loads(self):
        """Rigid normal force on each fingertip pad, per contact partner body (MuJoCo contact solve).
        Returns {pad_body: {other_body: N}}; other_body is -1 for static geometry."""
        m, d = self.solver.mj_model, self.solver.mj_data
        loads = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            shapes = [int(self._geom_shape[g]) if 0 <= g < len(self._geom_shape) else -1 for g in con.geom]
            if not any(s in self.pad_shapes for s in shapes):
                continue
            self._mujoco.mj_contactForce(m, d, i, f6)
            for k, s in enumerate(shapes):
                if s in self.pad_shapes:
                    other = shapes[1-k]
                    ob = int(self.shape_body[other]) if other >= 0 else -1
                    body = int(self.shape_body[s])
                    loads.setdefault(body, {})
                    loads[body][ob] = loads[body].get(ob, 0.) + abs(float(f6[0]))
        self.last_pad_loads = {b: sum(v.values()) for b, v in loads.items()}
        self.last_pad_pair_loads = loads
        return loads

    def step(self, fixture=None):
        """One label step = label_substeps rigid ticks. `fixture` is an ASSISTED callback only."""
        if self.fault:
            raise RuntimeError(self.fault)
        c = self.config
        try:
            t0 = time.perf_counter()
            for _ in range(c.label_substeps):
                self._rigid_step()
            t1 = time.perf_counter()
            self._pose_colliders()
            loads = self._pad_loads()
            for col in self.scene.colliders:
                if col.role == "pad":
                    col.pressure_force = dict(loads.get(col.body, {}))   # per partner body
            if fixture is not None:
                fixture(self)
            t2 = time.perf_counter()
            np.maximum(self.set_peak, np.abs(self.label.theta_rest), out=self.set_peak)   # read-only record
            if self.label_frozen is None:
                self.label.step(c.label_dt)
                if self.label.torn and c.stop_on_tear:
                    # The tear ONSET is the validated quantity (averaged orthotropic criterion); what an
                    # element-eroded sheet does afterwards is not (it fragments into element-sized pieces).
                    # The episode is decided (label_not_torn can never pass again), so the
                    # label is frozen where it tore and transmits no further load to robots or items.
                    self.label_frozen = dict(tick=self.steps + 1, time_s=(self.steps + 1)*c.label_dt,
                                             failed_triangles=int((~self.label.alive).sum()),
                                             reason="label_torn")
            self.label_history.append(self.label.x.copy())      # net-motion window for "settled"
            np.maximum(self.set_peak, np.abs(self.label.theta_rest), out=self.set_peak)   # read-only record
            self.steps += 1
            t3 = time.perf_counter()
            if self.label_frozen is None or self.label_frozen["tick"] == self.steps:
                self._apply_reactions()                         # the tearing step's own reaction still acts
            else:
                self.body_wrench[:] = 0.
                self.label.reactions = {}
            for attr in ("body_q", "body_qd", "joint_q"):
                if not np.isfinite(getattr(self.state, attr).numpy()).all():
                    raise FloatingPointError(f"Non-finite rigid {attr} at tick {self.tick}")
            self.timing["rigid_s"] += t1 - t0
            self.timing["couple_s"] += t2 - t1 + time.perf_counter() - t3
            self.timing["label_s"] += t3 - t2
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    REACTION_TAU = 5e-3        # s, first-order filter on label -> rigid reactions

    def _apply_reactions(self):
        """Label reactions as lagged external wrenches. The explicit one-step coupling is made stable
        by a first-order filter (time constant REACTION_TAU, standing in for fingertip-pad compliance)
        and each force is capped at the label's tear load (tensile strength x width): a label cannot
        transmit more without tearing, and a taut label pinned between a 30 g finger and a fixed
        printer is otherwise an unstable explicit spring loop."""
        w = np.zeros_like(self.body_wrench)
        q = self.state.body_q.numpy()
        com_local = self.model.body_com.numpy()
        for name, (f, tau, body) in self.label.reactions.items():
            if body < 0 or not np.any(f):
                continue
            col = next(cc for cc in self.scene.colliders if cc.name == name)
            R = L.quat_to_matrix(q[body, 3:].astype(float))
            com = q[body, :3] + R @ com_local[body]
            w[body, :3] += f
            w[body, 3:] += tau + np.cross(col.p - com, f)
        lab = self.label
        cap = min(lab.stock.strength_md, lab.stock.strength_cd)*lab.width
        mag = np.linalg.norm(w[:, :3], axis=1)
        scale = np.minimum(1., cap/np.maximum(mag, 1e-12))
        w *= scale[:, None]
        self.reaction_capped = int((scale < 1).sum())
        a = self.config.label_dt/(self.REACTION_TAU + self.config.label_dt)
        self.body_wrench = ((1 - a)*self.body_wrench + a*w).astype(np.float32)

    # ------------------------------------------------------------------ queries
    def body_pose(self, name):
        q = self.state.body_q.numpy()[self.scene.bodies[name]]
        return q[:3].astype(float), L.quat_to_matrix(q[3:].astype(float))

    def tcp(self, arm):
        return self.rig.frame_pose(f"{arm}/tcp", self.state) if f"{arm}/tcp" else None

    @property
    def sim_time(self):
        return self.tick*self.config.rigid_dt

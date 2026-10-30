"""Robots in Newton 1.5 + MuJoCo (CPU), from the shared robot catalog. Only the robots and the fixed
furniture they can bump into live here; every packaging item is in the implicit contact world
(ipcworld.py), where robot links appear as driven obstacles.

Hand semantics (Franka Hand, manual R50010 v1.2): fingers move at <= 50 mm/s each; `position` mode
drives each finger toward a half-width target with at most the finger effort limit; `force` mode closes
at that speed until blocked and then presses with the commanded force (grasp(width, speed, force)).
The closing coordinate is resolved in the contact world against whatever is between the fingers; the
MuJoCo finger joints then follow it (so they never close through paper the robot simulator cannot see).

Unitree G1 (g1.py): a floating humanoid with two Dex3-1 hands. It stands on its joint servos (no weld, no support), its
waist and arm servos get the gravity feed-forward torque Unitree's manipulation stack sends, and every joint (the Dex3
finger joints included) is an ordinary position servo slewed at its URDF speed: there is no parallel-gripper hand state,
no closing-axis coupling and no libfranka reflex (the G1's low-level joint interface has none).
"""

import math
import numpy as np

FRANKA_HAND_FINGER_SPEED = .05      # m/s per finger [Franka Hand manual R50010 v1.2]


def require_engine():
    import newton
    import warp as wp
    if newton.__version__ != "1.5.0":
        raise RuntimeError(f"Requires newton==1.5.0; found {newton.__version__}")
    newton.use_coord_layout_targets = True
    return newton, wp


class RobotRig:
    def __init__(self, *, setup, mounts, mount_z, furniture, device="cuda:0", rigid_dt=1e-3, solver_iterations=60,
                 ready_pose=None, contact_timeconst=.02):
        self.newton, self.wp = newton, wp = require_engine()
        import robots
        from robots.backends.newton import mount
        from newton.solvers import SolverMuJoCo
        self.rigid_dt = rigid_dt
        b = newton.ModelBuilder(gravity=(0., 0., -9.81))   # vector form (the scalar form is deprecated in Newton 1.4)
        SolverMuJoCo.register_custom_attributes(b)
        self.b = b
        self.materials = {}
        b.add_ground_plane(cfg=self._cfg(mu=.6), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        for f in furniture:
            if f.get("robot_collides", True):
                quat = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), float(f["yaw"])) if f.get("yaw") else wp.quat_identity()
                i = b.add_shape_box(-1, xform=wp.transform(f["center"], quat),
                                    hx=f["half"][0], hy=f["half"][1], hz=f["half"][2], cfg=self._cfg(mu=f.get("mu", .4)),
                                    label=f["name"])
                self.materials[i] = f.get("material", "furniture")
        self.setup = setup
        first, first_shape = b.body_count, b.shape_count
        self.pending = robots.add_to(b, setup, mounts={k: mount((x, y, mount_z), yaw) for k, ((x, y), yaw) in mounts.items()})
        self.robot_bodies = list(range(first, b.body_count))
        self.mounts = mounts
        self.mount_z = mount_z
        self.arm_hw = self._hardware(setup)
        self.humanoid = any(h.get("humanoid") for h in self.arm_hw.values())
        if self.humanoid:
            # contact detection envelope of the G1's shapes: the catalog import leaves Newton's default 0.1 m, with which
            # MuJoCo ran box-box narrowphase on pairs 10 cm apart and returned false contacts up to 0.64 m deep in
            # barcode scanning (its E17); the same rule as there: boxes and planes 0, hulls / spheres / cylinders 2 mm
            for i in range(first_shape, b.shape_count):
                box = b.shape_type[i] in (newton.GeoType.BOX, newton.GeoType.PLANE)
                b.shape_gap[i] = 0. if box else .002
        self.cameras = {}
        for name, handle in self.pending.instances.items():
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            if self.humanoid:
                from . import g1 as G1
                G1.head_camera(self)
            else:
                self._wrist_camera(name, handle)
        with wp.ScopedDevice(device):
            self.model = b.finalize(device=device)
            from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
            self.model.mujoco.solref_mode.fill_(SOLREF_MODE_RAW)
            ref = self.model.mujoco.solref.numpy(); ref[:] = (contact_timeconst, 1.)
            self.model.mujoco.solref.assign(ref)
            self.rig = self.pending.bind(self.model)
            self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=1024, broad_phase="sap",
                                                     reduce_contacts=True)
            self.contacts = self.pipeline.contacts()
            self.solver = SolverMuJoCo(self.model, use_mujoco_cpu=True, use_mujoco_contacts=True, enable_multiccd=True,
                                       integrator="implicitfast", cone="elliptic", solver="newton",
                                       iterations=solver_iterations, ls_iterations=20, nconmax=1024, njmax=3072)
            if self.humanoid:
                # MuJoCo's midphase (per-body BVH pruning of geom pairs) drops foot-sphere / floor contacts of the G1 in
                # this model: with it the robot pitched forward and fell within about 1 s of standing still at 0.5 ms; with
                # it disabled it stands. The same engine pruning dropped scanner/cradle contacts in barcode scanning,
                # which disables it too (barcode_scanning/world.py). The Franka configurations keep the setting they were qualified with.
                import mujoco
                self.solver.mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
        self.joints = dict(self.rig.joints)
        self._ready(ready_pose)
        self.state.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.targets = self.control.joint_target_q.numpy().copy()
        self.body_wrench = np.zeros((self.model.body_count, 6), np.float32)
        self.reflexes = []
        self.tick = 0
        self._g1_ff = self._g1_feedforward_setup() if self.humanoid else None
        # hand state per arm
        self.hand = {}
        if self.humanoid:
            # Dex3 hands: no gripper state machine; `closure` records the last gripper-posture command (g1.hand_posture)
            from . import g1 as G1
            self.hand = {s: dict(mode="posture", force=0., width=None, closure=0.) for s in G1.SIDES}
        for arm in ([] if self.humanoid else self.rig.instances):
            fj = self.gripper_joints(arm)
            q = self.model.joint_q.numpy()
            self.hand[arm] = dict(mode="position", force=0., width=float(q[self.joints[fj[0]]["q"]]),
                                  cmd=[float(q[self.joints[j]["q"]]) for j in fj],
                                  follow=[float(q[self.joints[j]["q"]]) for j in fj],
                                  effort=float(self.joints[fj[0]]["effort_limit"]))

    def _cfg(self, **kw):
        v = dict(mu=.45, ke=2e4, kd=100., density=1000., gap=.002, margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    @property
    def arms(self):
        """Policy arm names: the instances (FR3/Panda), or the G1's two arm/hand sides."""
        if self.humanoid:
            from . import g1 as G1
            return G1.SIDES
        return tuple(self.rig.instances)

    def _hardware(self, setup):
        from robots.catalog import model
        data = setup.to_dict()
        hw = {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            if spec.get("root", {}).get("mounting") == "floating":
                if inst["model"] != "g1@1" or data.get("attachments"):
                    raise ValueError(f"gift-bags hosts one floating robot, the Unitree G1 (g1@1) with its own Dex3-1 hands; "
                                     f"got {inst['model']} with attachments {sorted(data.get('attachments', {}))}")
                hw[name] = dict(model=inst["model"], mount=self.mounts[inst["mount"]], humanoid=True)
                continue
            hw[name] = dict(model=inst["model"], mount=self.mounts[inst["mount"]],
                            wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                            arm_joints=spec["groups"]["arm"]["joints"],
                            gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            if pads is None:
                raise ValueError(f"gift-bags needs a hand with fingertip pads; {att['model']} declares none")
            hw[owner].update(hand=att["model"], pads=pads)
        for name, h in hw.items():
            if h.get("humanoid"):
                continue
            if not h.get("gripper_joints") or "pads" not in h:
                raise ValueError(f"arm {name}: no gripper with fingertip pads")
        return hw

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist camera on link7 (the label/control-cable mount geometry)."""
        wl = self.arm_hw[arm]["wrist_link"]
        bodies = [i for i in handle.body_ids if wl and self.b.body_label[i].endswith(wl)]
        if not bodies:
            return
        body = bodies[0]
        wp = self.wp
        for nm, pos, half, dens, mat in ((f"{arm}_camera_case", (.075, 0, .07), (.021, .021, .0115), .060/(.042*.042*.023), "camera_dark"),
                                         (f"{arm}_camera_bracket", (.05, 0, .046), (.025, .013, .003), .010/(.050*.026*.006), "aluminum")):
            i = self.b.add_shape_box(body, xform=wp.transform(pos, wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2],
                                     cfg=self._cfg(density=dens), label=nm)
            self.materials[i] = mat
        # image up = +x of link7 (away from the hand axis): the fingers sit at the bottom centre of the image
        self.cameras[f"{arm}_wrist_rgb"] = dict(body=body, eye=[.075, -.012, .083], target=[.075, -.012, .283],
                                                up=[1, 0, 0], fov=87., role="policy")

    def _ready(self, ready_pose):
        """Start pose: the catalogue's label_application pose, overridden by the scene's ready pose. Scene poses are
        written with FR3 joint names (fr3_jointN); they apply by joint number to the arm actually mounted (a Panda's
        panda_jointN), and a joint that matches no arm joint or lies outside that arm's limits is an error, never
        skipped (a skipped name once left the dual-Panda setup in another start pose without a word)."""
        import re
        from robots.catalog import model
        q = self.model.joint_q.numpy()
        t = self.control.joint_target_q.numpy()
        if self.humanoid:
            # G1: the profile's initial pose (applied by the catalog import) with the scene's posture over it, written
            # with full joint names ({'g1/left_elbow_joint': v}); an unknown name or a value outside the limits is an error
            for key, v in (ready_pose or {}).items():
                if key not in self.joints:
                    raise ValueError(f"G1 ready pose: {key!r} is not a G1 joint")
                spec = self.joints[key]
                if not spec["lower"] <= v <= spec["upper"]:
                    raise ValueError(f"G1 ready pose {key} = {v} outside [{spec['lower']}, {spec['upper']}]")
                q[spec["q"]] = t[spec["q"]] = v
            self.model.joint_q.assign(q)
            self.control.joint_target_q.assign(t)
            return
        number = lambda j: (lambda m: int(m.group(1)) if m else None)(re.search(r"joint(\d+)$", j))
        for name, hw in self.arm_hw.items():
            pose = dict(model(hw["model"]).to_dict()["poses"]["label_application"])
            arm = {number(k.split("/", 1)[1]): k.split("/", 1)[1] for k in self.joints
                   if k.startswith(name + "/") and "finger" not in k and number(k) is not None}
            for joint, v in ((ready_pose or {}).get(name) or {}).items():
                n = number(joint)
                if n not in arm:
                    raise ValueError(f"ready pose joint {joint!r} matches no joint of the {name} arm ({hw['model']})")
                spec = self.joints[f"{name}/{arm[n]}"]
                if not spec["lower"] <= v <= spec["upper"]:
                    raise ValueError(f"ready pose {name}/{arm[n]} = {v} outside [{spec['lower']}, {spec['upper']}]")
                pose[arm[n]] = v
            for joint, v in pose.items():
                key = f"{name}/{joint}"
                if key in self.joints:
                    q[self.joints[key]["q"]] = t[self.joints[key]["q"]] = v
        self.model.joint_q.assign(q)
        self.control.joint_target_q.assign(t)

    def gripper_joints(self, arm):
        return [f"{arm}/{j}" for j in self.arm_hw[arm]["gripper_joints"]]

    # ------------------------------------------------------------------ commands
    def set_targets(self, targets):
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Not a robot actuator: {name}")
            spec = self.joints[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Invalid joint target {name}={value}")
        for name, value in targets.items():
            if "finger" in name:
                arm = name.split("/")[0]
                h = self.hand[arm]
                h["mode"], h["force"] = "position", 0.
                h["width"] = float(value)
            else:
                self.targets[self.joints[name]["q"]] = value

    def set_grip(self, arm, *, width=None, force=None):
        """Franka Hand: width = half-gap target per finger (m, position mode); force = per-finger force (N,
        force mode, closes until blocked)."""
        if arm not in self.hand:
            raise ValueError(f"No arm {arm}")
        h = self.hand[arm]
        if force is not None:
            if isinstance(force, bool) or not math.isfinite(force) or not 0 <= force <= h["effort"]:
                raise ValueError(f"Grip force must be in [0, {h['effort']}] N per finger")
            if force > 0:
                h["mode"], h["force"], h["width"] = "force", float(force), 0.
                return
        if width is not None:
            if isinstance(width, bool) or not math.isfinite(width) or not 0. <= width <= .04:
                raise ValueError("Franka Hand half gap must be in [0, 0.04] m")
            h["mode"], h["force"], h["width"] = "position", 0., float(width)

    # ------------------------------------------------------------------ stepping
    def rigid_step(self):
        dt = self.rigid_dt
        cur = self.control.joint_target_q.numpy()
        for name, spec in self.joints.items():
            j = spec["q"]
            if "finger" in name:
                continue
            slew = spec["velocity_limit"]*dt
            cur[j] += np.clip(self.targets[j] - cur[j], -slew, slew)
        # finger command slews at the hand's speed; the MuJoCo fingers follow the contact world's result
        for arm, h in ([] if self.humanoid else self.hand.items()):
            for k, jn in enumerate(self.gripper_joints(arm)):
                goal = h["width"]
                h["cmd"][k] += float(np.clip(goal - h["cmd"][k], -FRANKA_HAND_FINGER_SPEED*dt, FRANKA_HAND_FINGER_SPEED*dt))
                cur[self.joints[jn]["q"]] = h["follow"][k]
        self.control.joint_target_q.assign(cur)
        if self._g1_ff is not None:
            self._g1_gravity_feedforward(cur)
        self.state.clear_forces()
        self.state.body_f.assign(self.body_wrench)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, dt)
        self.state, self.next_state = self.next_state, self.state
        self.tick += 1
        if not self.humanoid:
            self._velocity_reflex()

    def _g1_feedforward_setup(self):
        """G1 upper-body servos: position PD plus a gravity feed-forward torque computed from the model at the commanded
        configuration, as Unitree's own G1 manipulation stack sends it for the arms (xr_teleoperate robot_arm_ik.py:
        pinocchio rnea(q_target, 0, 0) -> motor_cmd.tau; web-sourced); the waist
        gets the same (on the real robot Unitree's balance controller holds it; without it the PD waist sagged 4.2 deg
        under the upper body's weight in barcode scanning). Legs and Dex3 fingers are PD only. Same code as
        barcode_scanning/world.py World._g1_gravity_feedforward."""
        import mujoco
        self._mujoco = mujoco
        mjm = self.solver.mj_model
        names = {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(mjm.njnt)}
        arm = [n for n in self.joints if any(k in n for k in ("_shoulder_", "_elbow_", "_wrist_", "/waist_"))]
        mj_j = [[j for k, j in names.items() if k and k.endswith("_" + n.split("/")[-1])][0] for n in arm]
        return dict(names=arm, q=[self.joints[n]["q"] for n in arm], dof=[self.joints[n]["dof"] for n in arm],
                    mj_q=[int(mjm.jnt_qposadr[j]) for j in mj_j], mj_dof=[int(mjm.jnt_dofadr[j]) for j in mj_j],
                    limit=np.array([float(self.joints[n]["effort_limit"]) for n in arm]),
                    data=mujoco.MjData(mjm), tau=np.zeros(len(arm)), every=max(1, int(round(.005/self.rigid_dt))),
                    jf=np.zeros(self.model.joint_dof_count, np.float32))

    def _g1_gravity_feedforward(self, cur):
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
            ff["jf"][:] = 0.
            ff["jf"][ff["dof"]] = ff["tau"]
            self.control.joint_f.assign(ff["jf"])

    def _velocity_reflex(self):
        """FR3/Panda joint-velocity reflex (label_application.world): a joint faster than 1.05x its limit
        freezes its arm's targets at the current pose."""
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

    # ------------------------------------------------------------------ geometry for the contact world
    def link_collision_shapes(self):
        """{body index: (verts in body frame, faces)} of every robot link's collision shapes (convex
        hulls and the Franka finger boxes), merged per link."""
        from scipy.spatial.transform import Rotation
        from .ipcworld import box_mesh
        b = self.b
        out = {}
        for handle in self.pending.instances.values():
            for i in handle.shape_ids:
                body = b.shape_body[i]
                if body < 0 or not (b.shape_flags[i] & 2) or b.shape_label[i].endswith(("_camera_case", "_camera_bracket")):
                    continue
                kind = b.shape_type[i]
                tf = np.asarray(b.shape_transform[i], float)
                R = Rotation.from_quat(tf[3:]).as_matrix()
                scale = np.asarray(b.shape_scale[i], float)
                if kind == self.newton.GeoType.BOX:
                    V, F = box_mesh(tuple(scale))
                elif kind in (self.newton.GeoType.SPHERE, self.newton.GeoType.CYLINDER, self.newton.GeoType.CAPSULE):
                    # G1 foot spheres and shoulder cylinders (the FR3/Panda have none): their surfaces as hulls, axis z
                    import trimesh
                    if kind == self.newton.GeoType.SPHERE:
                        tm = trimesh.creation.icosphere(subdivisions=1, radius=float(scale[0]))
                    elif kind == self.newton.GeoType.CYLINDER:
                        tm = trimesh.creation.cylinder(radius=float(scale[0]), height=2*float(scale[1]), sections=16)
                    else:
                        tm = trimesh.creation.capsule(radius=float(scale[0]), height=2*float(scale[1]), count=[8, 8])
                    V, F = np.asarray(tm.vertices, float), np.asarray(tm.faces)
                elif kind in (self.newton.GeoType.MESH, self.newton.GeoType.CONVEX_MESH):
                    src = b.shape_source[i]
                    if src is None:
                        continue
                    import trimesh
                    tm = trimesh.Trimesh(np.asarray(src.vertices, float)*scale, np.asarray(src.indices).reshape(-1, 3),
                                         process=True).convex_hull
                    V, F = np.asarray(tm.vertices), np.asarray(tm.faces)
                    if self.humanoid:                    # G1 visual-resolution hulls -> inner hulls within 0.5 mm (g1.py)
                        from .g1 import inner_hull
                        V, F = inner_hull(V)
                else:
                    continue
                Vb = V @ R.T + tf[:3]
                if body in out:
                    V0, F0 = out[body]
                    out[body] = (np.vstack([V0, Vb]), np.vstack([F0, F + len(V0)]))
                else:
                    out[body] = (Vb, np.asarray(F))
        for c in self.cameras.values():
            pass
        return out

    def body_pose(self, body):
        from scipy.spatial.transform import Rotation
        q = self.state.body_q.numpy()[body]
        return q[:3].astype(float), Rotation.from_quat(q[3:].astype(float)).as_matrix()

    def finger_bodies(self, arm):
        """The two finger link bodies of an arm, in gripper-joint order."""
        out = []
        for jn in self.gripper_joints(arm):
            child = self.joints[jn].get("child_body")
            out.append(child)
        if all(o is not None for o in out):
            return out
        labels = self.model.body_label
        fingers = [i for i in self.robot_bodies if labels[i].startswith(arm + "/") and "finger" in labels[i]]
        fingers.sort(key=lambda i: ("left" not in labels[i], labels[i]))
        return fingers

    def hand_body(self, arm):
        """The rigid body that carries the hand: a separate hand body if the model has one, else the wrist link the
        hand is fixed to (the catalog's FR3 merges the Franka Hand into fr3_link7)."""
        labels = self.model.body_label
        cands = [i for i in self.robot_bodies if labels[i].startswith(arm + "/") and labels[i].endswith("hand")]
        if cands:
            return cands[0]
        wl = self.arm_hw.get(arm, {}).get("wrist_link")
        cands = [i for i in self.robot_bodies if wl and labels[i].startswith(arm + "/") and labels[i].endswith(wl)]
        return cands[0] if cands else None

    @property
    def sim_time(self):
        return self.tick*self.rigid_dt

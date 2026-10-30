"""Rigid scene of the robotic GMAW cell (Newton ModelBuilder -> MuJoCo): fixture table, the two arms from the shared
robot catalog, the MIG gun on the UR10e flange, the coupon (base plate and bracket as free steel bodies), the toggle
clamps with sprung spindle pads, the magnetic ground clamp, the welding equipment, camera mounts and screens, and the tack bond (a MuJoCo weld equality
built disabled; welding/bonds.py switches it on only where the thermal solver has fused both parts).

All placements come from welding/layout.py and the gun from welding/torch.py. Heat, melting and the bead live in the
thermal solver (welding/process.py); visual-only dressing (welding/shop.py) is outside every arm's reach and has no
collision.
"""

import math

import numpy as np

from . import layout as LY
from . import torch as TO

STEEL_RHO = 7850.
MU_STEEL = .6                  # mild steel on mild steel, dry (static ~0.74, kinetic ~0.57) [literature, rounded]
MU_PAD = .8                    # neoprene spindle cap on steel [estimate]

# Franka Hand finger collision boxes (left-finger frame) from the manufacturer's fr3_franka_hand.urdf (copied from
# filleting/scene.py, which read them from the URDF)
FRANKA_HAND_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                            ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                            ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                            ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))


WELD_SEG = 2e-3     # m: length of joint one weld-metal collider covers (4 cells along the weld)


class Scene:
    def __init__(self, config, coupon, newton, wp):
        self.c, self.coupon, self.newton, self.wp = config, coupon, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=-9.81)
        self.b.rigid_gap = 0.            # MuJoCo computes contacts; no inactive far contacts
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.bodies, self.shapes = {}, {}, {}
        self.cameras = {}
        self.room()
        self.robots()
        self.gun()
        self.workpieces()
        self.clamps()
        self.work_clamp()
        self.bond()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=MU_STEEL, ke=2e4, kd=100., density=0., gap=0., margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, pos, half, material, body=-1, rot=None, **kw):
        i = self.b.add_shape_box(body, xform=self.wp.transform(tuple(map(float, pos)), rot or self.wp.quat_identity()),
                                 hx=half[0], hy=half[1], hz=half[2], cfg=self.cfg(**kw), label=name)
        self.materials[i] = material
        self.shapes[name] = i
        return i

    # ------------------------------------------------------------------ room, table, cart
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=.6), label="floor")
        T = LY.TABLE
        z, (sx, sy), t = T["top_z"], T["size"], T["top_thickness"]
        self.box("table_top", (0., 0., z - t/2), (sx/2, sy/2, t/2), "table_steel")
        for x in (-sx/2 + .06, sx/2 - .06):
            for y in (-sy/2 + .06, sy/2 - .06):
                self.box(f"table_leg_{'p' if x > 0 else 'm'}{'p' if y > 0 else 'm'}", (x, y, (z - t)/2),
                         (T["leg"]/2, T["leg"]/2, (z - t)/2), "table_steel")
        # the welding equipment, the camera mounts and the screens: everything drawn within an arm's reach collides
        for name, c, h in LY.source_boxes():
            if name == "gas_cylinder":
                i = self.b.add_shape_cylinder(-1, xform=self.wp.transform(tuple(map(float, c)), self.wp.quat_identity()),
                                              radius=h[0], half_height=h[2], cfg=self.cfg(), label=name)
                self.materials[i], self.shapes[name] = "cylinder_paint", i
            else:
                self.box(name, c, h, "source_case")
        for name, c, h in LY.CAMERA_MOUNTS:
            self.box(name, c, h, "camera_mount")
        for name, c, h, yaw in LY.screen_boxes():
            self.box(name, c, h, "screen", rot=self.wp.quat(*map(float, LY.yaw_quat(yaw))))

    # ------------------------------------------------------------------ robots
    def robots(self):
        from robots.catalog import for_environment, model
        from robots.backends.newton import mount
        import robots
        setup = for_environment("welding", self.c.robot)
        self.robot_setup = setup
        data = setup.to_dict()
        self.arm_hw, mounts = {}, {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            if inst["model"].split("@")[0] == "g1":
                # the humanoid stands on the floor behind the table (welding/humanoid.py); one instance, both arms
                from . import humanoid as HU
                pos, yaw = HU.mount_pose()
                mounts[inst["mount"]] = mount(pos, yaw)
                self.arm_hw[name] = dict(model="g1", profile=inst["controller_profile"], mount=(tuple(pos[:2]), yaw),
                                         base_z=pos[2], arm_joints=spec["groups"]["body"]["joints"], gripper_joints=[],
                                         floating=True)
                continue
            base = LY.ROBOT_BASES[name]
            mounts[inst["mount"]] = mount(tuple(base["pos"]), math.radians(base["yaw_deg"]))
            self.arm_hw[name] = dict(model=inst["model"].split("@")[0], profile=inst["controller_profile"],
                                     mount=(tuple(base["pos"][:2]), math.radians(base["yaw_deg"])),
                                     base_z=base["pos"][2], arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            self.arm_hw[owner]["hand"] = att["model"]
        first = self.b.body_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        self.joints = {}
        for name, handle in self.rig_pending.instances.items():
            self.joints.update(handle.joints)
            for i in handle.shape_ids:
                self.materials[i] = "robot_" + self.arm_hw[name]["model"]
            if self.arm_hw[name].get("hand"):
                self._finger_colliders(name, handle)

    def _finger_colliders(self, arm, handle):
        """The manufacturer URDF's finger collision boxes on each Franka Hand finger (as filleting/label_application)."""
        from scipy.spatial.transform import Rotation
        b = self.b
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            if any(b.shape_type[i] == self.newton.GeoType.BOX for i in shapes):
                continue
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            tf = np.asarray(b.shape_transform[meshes[0]], float) if meshes else np.array([0, 0, 0, 0, 0, 0, 1.])
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            short = b.body_label[body].split("/")[-1]
            for k, (xyz, rpy, size) in enumerate(FRANKA_HAND_FINGER_BOXES):
                R = Rm*Rotation.from_euler("xyz", rpy)
                self.box(f"{arm}_{short}_box{k}", pm + Rm.apply(xyz), tuple(.5*v for v in size), "robot", body=body,
                         rot=self.wp.quat(*map(float, R.as_quat())), density=0., mu=.8)

    # ------------------------------------------------------------------ the MIG gun on the torch arm
    def gun(self):
        """The gun's shapes on the torch arm's flange (the 'right' instance of the welding rigs; UR10e tool0 or FR3
        link8 from the shared rig), with their masses: the arm carries its real gun (welding/torch.py: the 2.76 kg
        cobot gun on the UR10e, the 1.6 kg light torch on an FR3). Collision shapes: nozzle, neck, body, mount."""
        from scipy.spatial.transform import Rotation
        arm = self.torch_arm = "right" if "right" in self.arm_hw else next(iter(self.arm_hw))
        self.torch_instance = arm                 # the rig instance carrying the gun (the G1's: its right arm)
        model = self.arm_hw[arm]["model"]
        self.gun_model = model
        self.gun_spec = TO.for_model(model)
        self.gun_shapes = []
        if model in TO.MOUNTS:
            # no tool flange (the G1): the gun's flange frame on the link it is clamped to, and the clamp's collider
            M = TO.MOUNTS[model]
            self.torch_arm = "right"
            body = next(i for i in self.rig_pending.instances[arm].body_ids if self.b.body_label[i].endswith("/" + M.link))
            Rt, pt = Rotation.from_matrix(M.rotation()), np.asarray(M.origin, float)
            cx, cy, cz = M.clamp_half
            i = self.box("gun_clamp", M.clamp_center, M.clamp_half, "gun_clamp", body=body,
                         density=M.clamp_mass/(8*cx*cy*cz))
            self.gun_shapes.append(i)
        else:
            tool0 = self.rig_pending.instances[arm].frames["tool0" if model == "ur10e" else "flange"]
            body = tool0["body"]
            q = tool0["orientation_wxyz"]
            Rt = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            pt = np.asarray(tool0["position_m"], float)
        self.tool0_in_body = (pt, Rt.as_matrix())
        self.gun_body = body
        for name, kind, c, R, dims, mass in self.gun_spec.parts():
            p = pt + Rt.apply(c)
            Rb = Rt*Rotation.from_matrix(R)
            rot = self.wp.quat(*map(float, Rb.as_quat()))
            if kind == "cylinder":
                vol = math.pi*dims[0]**2*2*dims[1]
                i = self.b.add_shape_cylinder(body, xform=self.wp.transform(tuple(map(float, p)), rot), radius=dims[0],
                                              half_height=dims[1], cfg=self.cfg(density=mass/vol), label=name)
            elif kind == "cone":
                mesh, vol = self._frustum(*dims, self.gun_spec.nozzle_sides)
                i = self.b.add_shape_convex_hull(body, xform=self.wp.transform(tuple(map(float, p)), rot), mesh=mesh,
                                                 cfg=self.cfg(density=mass/vol), label=name)
            else:
                vol = math.pi*dims[0]**2*2*dims[1] + 4/3*math.pi*dims[0]**3
                i = self.b.add_shape_capsule(body, xform=self.wp.transform(tuple(map(float, p)), rot), radius=dims[0],
                                             half_height=dims[1], cfg=self.cfg(density=mass/vol), label=name)
            self.materials[i] = "gun_" + name.split("_")[1]
            self.shapes[name] = i
            self.gun_shapes.append(i)

    def _frustum(self, r_base, r_tip, half, sides):
        """A closed convex frustum mesh along local z (base ring at -half, tip ring at +half) whose polygon circumscribes
        the cone; returns (newton.Mesh, volume of the cone)."""
        k = 1./math.cos(math.pi/sides)
        a = np.arange(sides)*2*math.pi/sides
        ring = lambda r, z: np.stack([k*r*np.cos(a), k*r*np.sin(a), np.full(sides, z)], 1)
        V = np.concatenate([ring(r_base, -half), ring(r_tip, half), [[0., 0., -half], [0., 0., half]]]).astype(np.float32)
        cb, ct = 2*sides, 2*sides + 1
        tri = []
        for j in range(sides):
            n = (j + 1) % sides
            tri += [(j, n, sides + n), (j, sides + n, sides + j), (cb, n, j), (ct, sides + j, sides + n)]
        mesh = self.newton.Mesh(V, np.asarray(tri, np.int32).ravel())
        vol = math.pi*2*half/3*(r_base*r_base + r_base*r_tip + r_tip*r_tip)
        return mesh, vol

    # ------------------------------------------------------------------ coupon
    def workpieces(self):
        """Base plate and bracket as free steel bodies at their authored pose in the weld frame (welding/process.py
        coupon). The weld frame's origin relative to the base plate body is kept for the process coupling."""
        c = self.coupon
        wp = self.wp
        bx, by, bz = c.base
        base_c = np.array([c.base_x0 + bx/2, (c.base_y[0] + c.base_y[1])/2, -bz/2])          # weld frame
        b = self.b
        dx, dy, yaw = getattr(self.c, "coupon_offset", (0., 0., 0.))
        Rz = np.array([[math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)), 0.],
                       [math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0.], [0., 0., 1.]])
        origin = LY.WELD_ORIGIN + np.array([dx, dy, 0.])
        qz = wp.quat(*map(float, LY.yaw_quat(yaw)))
        place = lambda c_: tuple(map(float, origin + Rz @ c_))
        self.weld_origin_world0 = origin
        h = b.add_body(xform=wp.transform(place(base_c), qz), label="base_plate")
        self.box("base_plate_box", (0, 0, 0), (bx/2, by/2, bz/2), "steel_mill_scale", body=h, density=STEEL_RHO)
        self.bodies["base_plate"] = h
        self.weld_origin_in_base = -base_c                       # weld origin in the base plate body frame
        if c.kind == "tee":
            wx, wt, wh = c.web
            web_c = np.array([c.length/2, -wt/2, wh/2])
            k = b.add_body(xform=wp.transform(place(web_c), qz), label="bracket")
            self.box("bracket_box", (0, 0, 0), (wx/2, wt/2, wh/2), "steel_mill_scale", body=k, density=STEEL_RHO)
            self.bodies["bracket"] = k
            self.weld_origin_in_bracket = -web_c
            self.bracket_rel_nominal = web_c - base_c          # bracket centre in the base plate frame (fit-up)
        # deposited weld metal's colliders (as thermal cells only, the beads and tacks would let the gun, the hand and
        # its fingers pass through them): one box per WELD_SEG of joint on the base plate body,
        # built here with the model and parked inside the plate; world.World._weld_colliders sizes and places each from
        # the solid weld-metal cells of its segment. Massless (the cells' metal is already in the plate's mass)
        self.weld_segments, self.weld_colliders = [], []
        x0 = -3e-3
        while x0 < c.length + 3e-3 - 1e-9:
            x1 = min(x0 + WELD_SEG, c.length + 3e-3)
            i = self.box(f"weld_metal_{len(self.weld_colliders):03d}", (0., 0., 0.), (1e-4, 1e-4, 1e-4),
                         "weld_metal", body=h, density=0.)
            self.weld_segments.append((x0, x1))
            self.weld_colliders.append(i)
            x0 = x1

    def clamps(self):
        """DESTACO 225-U class hold-down clamps: a static base and arm (locked over centre) and the spindle's
        neoprene pad on a vertical slide, pressed onto the plate by the locked linkage. The linkage compliance is
        lumped into the slide's spring, preloaded to CLAMP_FORCE (k = 4e4 N/m over a 10 mm preload: a 300 g pad and
        spindle at a 1 ms step needs omega dt < 1; the static force is what the plate feels) [authored estimate]."""
        wp, b = self.wp, self.b
        z_plate = LY.WELD_ORIGIN[2]
        self.clamp_pads = {}
        for C in LY.CLAMPS:
            x, y = C["pad_xy"]
            H = LY.CLAMP_PAD["height"]
            # static base on the table in front of the plate, arm over the plate edge (locked)
            self.box(f"{C['name']}_base", (x, y - .085, LY.TABLE["top_z"] + .025), (.022, .03, .025), "clamp_paint")
            self.box(f"{C['name']}_arm", (x, y - .045, z_plate + .035), (.009, .055, .008), "clamp_paint")
            pad_z0 = z_plate + H/2                                 # pad centre resting on the plate
            # spindle + pad: 0.3 kg as a 24 mm x 60 mm cylinder (its inertia; a mass with zero inertia was
            # "corrected" by the builder at finalize, with a startup warning)
            m_, r_, h_ = .3, .012, .060
            Ixx = m_*(3*r_*r_ + h_*h_)/12
            k = b.add_link(xform=wp.transform((x, y, pad_z0 + .002), wp.quat_identity()), mass=m_,
                           inertia=wp.mat33(Ixx, 0., 0., 0., Ixx, 0., 0., 0., .5*m_*r_*r_), label=f"{C['name']}_pad")
            i = b.add_shape_cylinder(k, radius=LY.CLAMP_PAD["radius"], half_height=H/2,
                                     cfg=self.cfg(density=0., mu=MU_PAD), label=f"{C['name']}_pad_shape")
            self.materials[i] = "neoprene"
            stiff = 4e4
            pre = LY.CLAMP_FORCE/stiff
            j = b.add_joint_prismatic(-1, k, parent_xform=wp.transform((x, y, pad_z0), wp.quat_identity()),
                                      axis=(0., 0., 1.), target_pos=-pre, target_ke=stiff, target_kd=60.,
                                      limit_lower=-.02, limit_upper=.03, effort_limit=1e4, label=f"{C['name']}_slide")
            b.add_articulation([j], label=C["name"])
            self.clamp_pads[C["name"]] = k

    def work_clamp(self):
        W = LY.WORK_CLAMP
        wp, b = self.wp, self.b
        sx, sy, sz = W["size"]
        pos = (W["pos"][0], W["pos"][1], LY.TABLE["top_z"] + sz/2 + 1e-4)
        k = b.add_body(xform=wp.transform(pos, wp.quat_identity()), label="work_clamp")
        self.box("work_clamp_box", (0, 0, 0), (sx/2, sy/2, sz/2), "clamp_copper", body=k, density=W["mass"]/(sx*sy*sz))
        self.bodies["work_clamp"] = k

    def bond(self):
        """The tack/weld bond between bracket and base plate: a MuJoCo weld equality, built disabled. It is switched on
        (bonds.py) only when the thermal solver has fused weld metal into both parts, with the parts' pose at that
        moment, and removed if its load exceeds the fused section's capacity."""
        if "bracket" not in self.bodies:
            self.bond_index = None
            return
        from newton._src.solvers.mujoco.equality import _add_equality_constraint
        from newton._src.solvers.mujoco.enums import EqType
        self.bond_index = _add_equality_constraint(self.b, EqType.WELD, body1=self.bodies["base_plate"],
                                                   body2=self.bodies["bracket"], enabled=False, label="weld_bond")

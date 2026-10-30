"""Physics scene (Newton 1.5 ModelBuilder): room, fixtures, robots, scanner, cradle, keypad and stock.

All collision geometry is authored. Printed labels
and packaging artwork have no collision (they are 0.1 mm films); their poses on their bodies are recorded
in `Scene.labels` so the renderer, the verifier's attribution and the evaluator use the same geometry.

World frame: z up; the shelving's front face is the plane y = 0, its interior y > 0, the aisle y < 0.
The counter presets reuse the same frame with the counter's robot-side edge at y = 0.
"""

import math
import numpy as np
from . import params as P
from .catalog import SKUS, symbol_for, neck

# robot deployment: both arm bases on the cart top, facing +y (the shelf / counter)
ARM_MOUNTS = {"left_arm_mount": ((-P.ARM_Y_M, -P.BASE_TO_SHELF_M), math.pi/2),
              "right_arm_mount": ((P.ARM_Y_M, -P.BASE_TO_SHELF_M), math.pi/2)}
CART_FRONT_Y = -P.BASE_TO_SHELF_M + 0.12          # cart front edge (bases 0.12 m behind it)
CART_BACK_Y = CART_FRONT_Y - 0.60
DECKS = (0.1143, 0.5715, 1.0287, 1.4478, 1.8288)  # deck tops on the 38.1 mm grid (3, 15, 27, 38, 48 steps)
BAY_W, BAY_D = P.SHELF_BAY_M[0], P.SHELF_BAY_M[1]
BEAM_H, BEAM_T = 0.045, 0.035
SLOTS_PER_DECK = 3


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    s = math.sin(angle/2)
    return (float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(math.cos(angle/2)))


def quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def rotate(q, v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).apply(v)


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))     # z up
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.rng = np.random.default_rng(config.seed)
        self.materials, self.visual = {}, set()
        self.bodies, self.items, self.labels, self.cameras = {}, {}, [], {}
        self.hull_boxes = {}
        self.fixtures = {}             # static fixture frames for dressing (shelves, counter, cart ...)
        self.keys, self.slots = {}, {}
        # the G1 humanoid stands on the floor where the FR3 cart would be: its cart is parked beside it (g1.LAYOUT);
        # None keeps every FR3/Panda coordinate exactly as authored
        self.g1 = config.robot == "g1"
        self.cart_pose = None
        if self.g1:
            from . import g1 as G1
            self.cart_pose = G1.cart_transform(config.preset)
        self.room()
        self.cart()
        self.robots()
        self.scanner_and_cradle()
        getattr(self, config.preset)()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=.5, ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_M, margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, hull=False, **kw):
        """A box shape; hull=True makes it an 8-vertex convex hull of the same size and mass, which MuJoCo collides with
        its general convex collider instead of its box-box routine (used for box SKUs of BOX_AS_HULL_MIN_MASS_KG or
        more, add_item). Every free box as a hull was tried and rejected on measurement: resting stock crept and a held
        scanner slipped (a cereal box in the seed-2 basket 24 um/s against 1.0 um/s as a box; E7's head grasp slipped
        3.3 mm and turned 3.5 deg against 0.9 mm and 0.7 deg)."""
        tf = self.wp.transform(pos, rot or self.wp.quat_identity())
        if hull and collide:
            import trimesh
            tm = trimesh.creation.box(extents=2*np.asarray(half, float))
            m = self.newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())
            i = self.b.add_shape_convex_hull(body, mesh=m, xform=tf, cfg=self.cfg(**dict(dict(gap=P.MESH_CONTACT_GAP_M), **kw)),
                                             label=name)
            self.materials[i] = material
            self.hull_boxes[i] = tuple(map(float, half))           # drawn as a box (render.export)
            return i
        cfg = self.cfg(**kw) if collide else self.newton.ModelBuilder.ShapeConfig(
            density=0., has_shape_collision=False, has_particle_collision=False)
        i = self.b.add_shape_box(body, xform=tf, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        return i

    def hull_cylinder(self, name, pos, radius, half, material, body, sections=32, rot=None, **kw):
        """Cylinder as a 32-facet convex hull: MuJoCo's multi-point convex contact rests it flat
        (single-point cylinder-box contact rocks and creeps; label_application scene.cyl_hull)."""
        import trimesh
        tm = trimesh.creation.cylinder(radius=radius, height=2*half, sections=sections)
        m = self.newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())
        i = self.b.add_shape_convex_hull(body, mesh=m, xform=self.wp.transform(pos, rot or self.wp.quat_identity()),
                                         cfg=self.cfg(**dict(dict(gap=P.MESH_CONTACT_GAP_M), **kw)), label=name)
        self.materials[i] = material
        return i

    def necked_hull(self, name, s, body, mu, sections=32):
        """Jar or bottle: the straight body and its tapering shoulder as one convex hull (the profile narrows
        upward, so their union is convex), the cap as a second hull on the same body. The contents' mass sits in
        the body (98 %); set_mass then scales the total to the SKU's mass."""
        import trimesh
        g = neck(s)
        z0, z1 = g["body"]; z2 = g["shoulder"][1]; z3 = g["cap"][1]
        ring = lambda rad, z: [(rad*math.cos(a), rad*math.sin(a), z) for a in np.linspace(0, 2*math.pi, sections, endpoint=False)]
        pts = np.array(ring(g["r"], z0) + ring(g["r"], z1) + ring(g["cap_r"], z2), np.float32)
        hull = trimesh.convex.convex_hull(pts)
        cap = trimesh.creation.cylinder(radius=g["cap_r"], height=z3 - z2, sections=sections)
        cap.apply_translation((0., 0., (z2 + z3)/2))
        for part, tm, frac in (("hull", hull, .98), ("cap", cap, .02)):
            m = self.newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())
            i = self.b.add_shape_convex_hull(body, mesh=m, xform=self.wp.transform((0., 0., 0.), self.wp.quat_identity()),
                                             cfg=self.cfg(gap=P.MESH_CONTACT_GAP_M, density=s.mass*frac/float(tm.volume), mu=mu),
                                             label=f"{name}_{part}")
            self.materials[i] = f"sku_{s.id}"

    def free_body(self, name, pos, rot=None):
        body = self.b.add_body(xform=self.wp.transform(pos, rot or self.wp.quat_identity()), label=name)
        self.bodies[name] = body
        return body

    def set_mass(self, body, mass):
        m = self.b.body_mass[body]
        if m <= 0 or mass <= 0:
            raise ValueError("Dynamic solid must have positive mass")
        self.b.body_inertia[body] = self.b.body_inertia[body]*float(mass/m)
        self.b.body_mass[body] = float(mass)

    def label(self, name, body, symbol, center, normal, up, size, finish="matte", kind="product", item=None,
              curvature_radius=None, text=None):
        """A printed symbol's pose in its body frame (center, outward normal, 'up' direction in the label
        plane), its printed size (w, h in m), finish and role. Visual only."""
        self.labels.append(dict(name=name, body=body, symbol=symbol, center=list(map(float, center)),
                                normal=list(map(float, normal)), up=list(map(float, up)), size=list(map(float, size)),
                                finish=finish, kind=kind, item=item, radius=curvature_radius, text=text))

    # ------------------------------------------------------------------ room and cart
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=.6), label="floor")
        self.materials[b.shape_count - 1] = "floor"

    CART_CENTER = (0., (CART_FRONT_Y + CART_BACK_Y)/2, 0.)      # the cart's centre in the FR3 layout (the authored frame)

    def cart_point(self, p, q=None):
        """A point (and orientation) authored in the FR3 cart layout -> world. Identity (the very same objects returned)
        unless the robot configuration parks the cart elsewhere (G1: g1.LAYOUT)."""
        if self.cart_pose is None:
            return p, q
        (cx, cy), yaw = self.cart_pose
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = (float(v) - float(o) for v, o in zip(p, self.CART_CENTER))
        pw = (cx + c*x - s*y, cy + s*x + c*y, z)
        qw = quat_mul(quat_axis_angle((0, 0, 1), yaw), q if q is not None else (0., 0., 0., 1.))
        return pw, qw

    def _cart_box(self, name, pos, half, material, rot=None, **kw):
        p, q = self.cart_point(pos, rot)
        return self.box(name, p, half, material, rot=q, **kw)

    def cart(self):
        """Steel utility cart carrying the two arms (casters locked; authored). In the G1 configuration
        the same cart and tray stand beside the humanoid (cart_point)."""
        z = P.CART_TOP_M
        cx, cy = 0., (CART_FRONT_Y + CART_BACK_Y)/2
        self._cart_box("cart_top", (cx, cy, z - .012), (.50, .30, .012), "cart_steel", mu=.4)
        self._cart_box("cart_shelf", (cx, cy, .16), (.48, .28, .01), "cart_steel")
        for sx in (-.47, .47):
            for sy in (-.27, .27):
                self._cart_box(f"cart_post_{sx}_{sy}", (sx, cy + sy, (z - .024 + .06)/2), (.015, .015, (z - .024 - .06)/2), "cart_frame")
        self.fixtures["cart"] = dict(center=self.cart_point((cx, cy, z))[0], half=(.50, .30))
        if self.cart_pose is not None:
            self.fixtures["cart"]["yaw"] = self.cart_pose[1]
        # accessory tray bolted to the cart's front edge (between and ahead of the arm bases): the host display,
        # the keypad and the scanner cradle, clear of both arm bases
        self.tray_y = CART_FRONT_Y + .065
        self._cart_box("cart_tray", (0., self.tray_y, z + .02), (.32, .065, .008), "cart_frame", mu=.4)
        for sx in (-.29, .29):
            self._cart_box(f"cart_tray_arm_{sx}", (sx, CART_FRONT_Y + .03, z + .006), (.012, .04, .006), "cart_frame")
        self.riser_top = z + .028
        # host display (8 in class, 200 x 130 mm screen) on a stand at the tray's left, tilted back 50 deg to face
        # the head camera on the mast
        # low and laid back (60 deg) in the tray's left corner: out of the space the hands use under a held scanner
        # (a stand with its top edge at 0.93 m stopped the left hand reaching under the scanner)
        tilt, dx, stand = 60., -.20, .02
        if self.g1:                    # other end of the tray, turned towards the humanoid at the tray's edge (g1.TRAY)
            from . import g1 as G1
            tilt, dx, stand = G1.TRAY["display_tilt_deg"], G1.TRAY["display_x"], -.02
        tq = quat_axis_angle((1, 0, 0), math.radians(tilt))
        if self.g1:                    # laid back towards the humanoid, text upright for it: turned 180 deg about z
            tq = quat_mul(quat_axis_angle((0, 0, 1), math.pi), quat_axis_angle((1, 0, 0), math.radians(-tilt)))
        self._cart_box("display_stand", (dx, self.tray_y + stand, self.riser_top + .015), (.03, .02, .015), "cart_frame")
        self._cart_box("display", (dx, self.tray_y, self.riser_top + .06), (.11, .075, .008), "tablet_black", rot=tq)
        c_t, q_t = self.cart_point((dx, self.tray_y, self.riser_top + .06), tq)
        self.fixtures["tablet"] = dict(center=c_t, quat=q_t, screen=(.20, .13))
        # camera mast at the cart's back edge
        self._cart_box("mast", (0., CART_BACK_Y + .04, z + .44), (.015, .015, .44), "cart_frame")
        self._cart_box("mast_head", (0., CART_BACK_Y + .06, 1.62), (.03, .03, .02), "cart_frame")
        self.fixtures["mast_camera"] = dict(eye=self.cart_point((0., CART_BACK_Y + .09, 1.64))[0])
        if self.cart_pose is not None:            # the mast camera turns with the cart (render.py default otherwise)
            self.fixtures["mast_camera"]["target"] = self.cart_point((0., .30, .78))[0]

    def robots(self):
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        if self.g1:
            return self._g1_robot()
        setup = for_environment("barcode-scanning", self.c.robot)
        self.robot_setup = setup
        self._hardware(setup)
        mounts = {k: mount((x, y, P.CART_TOP_M), yaw) for k, ((x, y), yaw) in ARM_MOUNTS.items()}
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        # contact detection envelope of the catalog robots' links and fingers: their controller profile leaves Newton's
        # default 0.1 m (FR3), with which MuJoCo ran box-box narrowphase on pairs 10 cm apart; it returned false contacts
        # up to 0.64 m deep and hurled a 5 kg case at 3 m/s from a finger 9 cm away (E17 random targets). The same
        # envelope rule as the scene's own shapes (params: boxes 0, hulls 2 mm)
        for i in range(first_shape, self.b.shape_count):
            box = self.b.shape_type[i] in (self.newton.GeoType.BOX, self.newton.GeoType.PLANE)
            self.b.shape_gap[i] = P.CONTACT_GAP_M if box else P.MESH_CONTACT_GAP_M
        self.joints, self.arms = {}, {}
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            self.arms[name] = handle
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._fingertips(name, handle)
            self._wrist_camera(name, handle)

    def _g1_robot(self):
        """Unitree G1 + two Dex3-1 hands standing on the floor (g1.py; rig robots/rigs/scan-g1.json). Floating base:
        no weld, no pelvis support. The FR3 hardware tables stay empty (no parallel gripper, no fingertip pads: the Dex3
        links collide with their URDF meshes' convex hulls under the scene's rigid contact law)."""
        from robots.catalog import resolve
        from robots.backends.newton import mount
        import robots
        from . import g1 as G1
        setup = resolve(G1.SETUP)
        self.robot_setup = setup
        self.arm_hw, self.pad_shapes = {}, {}
        lay = G1.LAYOUT[self.c.preset]
        (x, y), yaw = lay["pelvis"], lay["yaw"]
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts={G1.MOUNT: mount((x, y, G1.PELVIS_Z), yaw)})
        self.robot_bodies = list(range(first, self.b.body_count))
        for i in range(first_shape, self.b.shape_count):           # the same contact envelope rule as the FR3 (E17)
            box = self.b.shape_type[i] in (self.newton.GeoType.BOX, self.newton.GeoType.PLANE)
            self.b.shape_gap[i] = P.CONTACT_GAP_M if box else P.MESH_CONTACT_GAP_M
        self.joints, self.arms = {}, {}
        handle = self.rig_pending.instances[G1.INSTANCE]
        for j, entry in handle.joints.items():
            self.joints[j] = entry
        self.arms[G1.INSTANCE] = handle
        for i in handle.shape_ids:
            self.materials[i] = "robot"
        self.g1_hw = dict(pelvis=G1.body_index(self.b, "pelvis"), torso=G1.body_index(self.b, "torso_link"),
                          wrist={s: G1.body_index(self.b, f"{s}_wrist_yaw_link") for s in ("left", "right")},
                          feet={s: G1.body_index(self.b, f"{s}_ankle_roll_link") for s in ("left", "right")},
                          layout=dict(lay))
        G1.head_camera(self, handle)

    def _hardware(self, setup):
        from robots.catalog import model
        data = setup.to_dict()
        self.arm_hw = {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            self.arm_hw[name] = dict(model=inst["model"], mount=ARM_MOUNTS[inst["mount"]],
                                     wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                                     arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)
        for name, hw in self.arm_hw.items():
            if not hw.get("pads") or not hw["gripper_joints"]:
                raise ValueError(f"arm {name}: barcode-scanning needs a hand with fingertip pads and gripper joints")

    # Franka Hand finger collision boxes from the manufacturer's fr3_franka_hand.urdf (xyz, rpy, full size); the
    # Panda URDF models its fingers as meshes, so the same boxes are added there (label_application precedent).
    FRANKA_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                           ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                           ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                           ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

    def _fingertips(self, arm, handle):
        """A finger with only a mesh collider (Panda URDF) gets the Franka Hand's URDF boxes; the rubber tip box
        (17.5 x 15.2 x 18.5 mm at z = 45.25 mm) is the pad that touches the trigger, keys and stock."""
        from scipy.spatial.transform import Rotation
        b = self.b
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        self.pad_shapes = getattr(self, "pad_shapes", {})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            boxes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.BOX]
            pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
            if boxes:
                for i in boxes:
                    if abs(float(np.asarray(b.shape_transform[i])[2]) - pad_z) < 1e-4:
                        self.pad_shapes[i] = b.body_label[body]
                continue
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            if not meshes:
                continue
            tf = np.asarray(b.shape_transform[meshes[0]], float)
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            for k, (xyz, rpy, size) in enumerate(self.FRANKA_FINGER_BOXES):
                R = Rm*Rotation.from_euler("xyz", rpy)
                p = pm + Rm.apply(xyz)
                i = self.box(f"{b.body_label[body].split('/')[-1]}_box{k}", tuple(p), tuple(np.asarray(size)/2), "robot_dark",
                             body=body, rot=tuple(R.as_quat()), density=0., mu=P.MU["pad_housing"])
                if abs(xyz[2] - pad_z) < 1e-6:
                    self.pad_shapes[i] = b.body_label[body]

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on link7 (same mount geometry as control_cable and label_application)."""
        labels = self.b.body_label
        wl = self.arm_hw[arm]["wrist_link"]
        link7 = [i for i in handle.body_ids if wl and labels[i].endswith(wl)]
        if not link7:
            return
        body = link7[0]
        self.box(f"{arm}_camera_case", (.075, 0, .07), (.021, .021, .0115), "camera_dark", body=body,
                 density=.060/(.042*.042*.023))
        self.box(f"{arm}_camera_bracket", (.05, 0, .046), (.025, .013, .003), "aluminum", body=body,
                 density=.010/(.050*.026*.006))
        self.cameras[f"{arm}_wrist_rgb"] = dict(body=body, eye=[.075, -.012, .083], target=[.075, -.012, .283],
                                                up=[0, 1, 0], fov=87., role="policy")

    # ------------------------------------------------------------------ scanner, cradle, keypad
    # scanner body frame: origin at the exit-window centre, +x out of the window (optical axis), +z up, +y left.
    # Inside the DS2278's envelope, 175 H x 66 W x 90 D mm (params.SCANNER_ENVELOPE_M, checked when the scanner is
    # built): a 72 x 66 x 68 mm head and the 45 x 35 x 105 mm handle raked 15 deg back, its top 41 mm behind the window so
    # that its rear bottom corner is the device's 90 mm depth. The head overhangs the handle's back by 9 mm.
    HEAD_HALF = (0.036, 0.033, 0.03388)
    HANDLE_HALF = (0.0225, 0.0175, 0.0525)
    HANDLE_RAKE = math.radians(15.)
    HANDLE_TOP = (-0.04109, 0., -0.03388)

    def handle_frame(self):
        """Centre and quaternion (scanner frame) of the handle box; its local z points up along the handle."""
        d = np.array([-math.sin(self.HANDLE_RAKE), 0., -math.cos(self.HANDLE_RAKE)])
        centre = np.asarray(self.HANDLE_TOP) + self.HANDLE_HALF[2]*d
        return centre, quat_axis_angle((0, 1, 0), self.HANDLE_RAKE)

    def _scanner_corners(self):
        from scipy.spatial.transform import Rotation
        hh, dh = np.asarray(self.HEAD_HALF), np.asarray(self.HANDLE_HALF)
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        hc, hq = self.handle_frame()
        return np.vstack([(-hh[0], 0., 0.) + signs*hh, hc + Rotation.from_quat(hq).apply(signs*dh)])

    def scanner_extent(self):
        """(height, width, depth) of the scanner's colliders, m, as the datasheet gives the envelope."""
        c = self._scanner_corners()
        return tuple(float(v) for v in (np.ptp(c[:, 2]), np.ptp(c[:, 1]), np.ptp(c[:, 0])))

    def scanner_com(self):
        """Centre of mass of the scanner and its trigger blade, scanner frame (m), from the colliders' masses."""
        hc, _ = self.handle_frame()
        m_head = P.SCANNER_CORDED_HEAD_FRACTION*P.SCANNER_CORDED_MASS_KG
        m_handle = P.SCANNER_MASS_KG - m_head - P.TRIGGER_MASS_KG
        r = P.TRIGGER_ARM_M
        hq = self.handle_frame()[1]
        front_top = hc + rotate(hq, (self.HANDLE_HALF[0], 0., self.HANDLE_HALF[2] - .004))
        pivot = front_top + .015*rotate(hq, (0., 0., 1.)) - .004*rotate(hq, (1., 0., 0.))
        blade = pivot + rotate(hq, (.007, 0., -r))
        return (m_head*np.array([-self.HEAD_HALF[0], 0., 0.]) + m_handle*hc + P.TRIGGER_MASS_KG*blade)/P.SCANNER_MASS_KG

    def scanner_and_cradle(self):
        """DS2278-class cordless imager (P.SCANNER_*) with a spring-return trigger, and its presentation cradle."""
        wp = self.wp
        cq = self.cradle_pose()
        seat_world, scanner_q = self._seated_scanner_pose(cq)
        body = self.b.add_link(xform=wp.transform(tuple(seat_world), scanner_q), label="scanner")
        self.bodies["scanner"] = body
        free = self.b.add_joint_free(body, label="scanner_free")
        hc, hq = self.handle_frame()
        # mass split derived from the two products' masses: the corded unit's head (60 % of 161.6 g, an estimate) and
        # everything else in the handle, the cordless unit's battery included: 45 / 55 % of 214 g. The trigger
        # blade's 3 g is part of the 214 g
        m_head = P.SCANNER_CORDED_HEAD_FRACTION*P.SCANNER_CORDED_MASS_KG
        m_handle = P.SCANNER_MASS_KG - m_head - P.TRIGGER_MASS_KG
        hh, dh = self.HEAD_HALF, self.HANDLE_HALF
        ext = self.scanner_extent()
        if any(e > lim + 5e-4 for e, lim in zip(ext, P.SCANNER_ENVELOPE_M)):
            raise ValueError(f"scanner {ext} m (H, W, D) exceeds the DS2278 envelope {P.SCANNER_ENVELOPE_M}")
        self.box("scanner_head", (-hh[0], 0., 0.), hh, "scanner_body", body=body,
                 density=m_head/(8*hh[0]*hh[1]*hh[2]), mu=P.MU["scanner_cradle"])
        self.box("scanner_handle", tuple(hc), dh, "scanner_grip", body=body, rot=hq,
                 density=m_handle/(8*dh[0]*dh[1]*dh[2]), mu=P.MU["scanner_cradle"])
        # trigger: pivot inside the head 15 mm above the blade; blade 20 mm on the handle's front face
        front_top = hc + rotate(hq, (dh[0], 0., dh[2] - .004))           # handle front face, near its top
        up = rotate(hq, (0., 0., 1.)); fwd = rotate(hq, (1., 0., 0.))
        pivot = front_top + .015*up - .004*fwd
        r = P.TRIGGER_ARM_M
        tb = self.b.add_link(xform=wp.transform(tuple(np.asarray(seat_world) + rotate(scanner_q, pivot)),
                                                quat_mul(scanner_q, hq)), label="scanner_trigger")
        self.bodies["scanner_trigger"] = tb
        blade_half = (.003, .008, .010)
        # blade centre: r below the pivot along -up (handle frame), 7 mm in front of the pivot (6 mm proud)
        self.box("trigger_blade", (.007, 0., -r), blade_half, "trigger_black", body=tb,
                 density=P.TRIGGER_MASS_KG/(8*blade_half[0]*blade_half[1]*blade_half[2]), mu=.6)
        # joint: rotation about the handle y axis; positive = blade moves back into the handle (R_y(+q) takes the
        # blade, 25 mm below the pivot, towards -x; with -y a press drives it into its rest stop)
        pre = P.TRIGGER_PRELOAD_N*r                                         # N m at the rest stop
        k = P.TRIGGER_RATE_N_PER_M*r*r                                      # N m/rad
        self.trigger_joint = self.b.add_joint_revolute(
            body, tb, parent_xform=wp.transform(tuple(pivot), hq), child_xform=wp.transform_identity(),
            axis=(0., 1., 0.), limit_lower=0., limit_upper=P.TRIGGER_STOP_M/r,
            target_pos=-pre/k, target_ke=k, target_kd=2*P.TRIGGER_DAMPING_RATIO*math.sqrt(k*(P.TRIGGER_MASS_KG*r*r + 1e-6)),
            armature=1e-6, friction=0., effort_limit=5.,
            label="scanner_trigger_pivot")
        self.b.add_articulation([free, self.trigger_joint], label="scanner")
        self.scanner = dict(body=body, trigger_body=tb, joint=self.trigger_joint, pivot=pivot, arm=r)
        # the imager's entrance pupil and light sources, scanner frame (rendering and attribution)
        self.scanner.update(pupil=(-P.PUPIL_SETBACK_M, 0., 0.), leds=((-.004, P.LED_OFFSET_M, -.012), (-.004, -P.LED_OFFSET_M, -.012)),
                            aim=(-.004, 0., P.AIM_OFFSET_M))

    def cradle_pose(self):
        """Cradle base pose (world): on the cart riser in front of the arms (stockroom), or on the counter."""
        if self.c.preset in ("counter", "library"):
            # window towards the operator (the robot), as a cashier's presentation stand faces the cashier: items are
            # passed between the robot and the window (facing the customer side made the arm reach over the scanner
            # and knock it out of its cup)
            return (0.06, 0.16, P.COUNTER_TOP_M), quat_axis_angle((0, 0, 1), -math.pi/2)
        x, yaw_deg = self.STOCKROOM_CRADLE
        if getattr(self, "g1", False):  # the tray end nearest the shelf, window towards the shelf (g1.TRAY)
            from . import g1 as G1
            return self.cart_point((G1.TRAY["cradle_x"], self.tray_y, self.riser_top),
                                   quat_axis_angle((0, 0, 1), math.radians(G1.TRAY["cradle_yaw_deg"])))
        return (x, self.tray_y, self.riser_top), quat_axis_angle((0, 0, 1), math.radians(yaw_deg))

    STOCKROOM_CRADLE = (0.19, 90.)     # tray x (m) and yaw (deg; 90 = window towards the shelf)
    # The CR2278-PC (Zebra Quick Reference Guide MN-002917-03 pp. 3, 5): a low teardrop base, the handle's pocket at its
    # raised rear (72 mm overall height), the charging contacts on the pocket's front edge; the scanner is set in
    # handle first and "naturally rotates forward to engage the contacts", its head overhanging the front with nothing
    # under it. Modelled as a socket for the handle: a floor, a front lip (the contacts), and rear and side walls that
    # stand straight for their first 8 mm, 1.5 mm from the handle, and flare out above (a moulded pocket's lead-in). The
    # head's weight turns the scanner forward until its front face rests on the lip's top edge and the handle's bottom
    # on the floor and against the rear wall; the socket is built tilted back by that lean (socket_lean), so the seated
    # pose (window 12 deg down) is the rest itself, and the straight walls make it the bottom of a well: turning further
    # forward lifts the handle up the rear wall (its centre of mass rises 5 mm per degree). Rejected pockets: a 40 mm cup
    # with 2 mm clearance wedged the scanner 14 mm off its seat, and a head rest on a post (not on the real cradle) held
    # it and blocked the G1's index and middle fingers; a 40 mm cup with 1.5 mm clearance and no head rest let the
    # scanner rest 4.5 deg forward of its seated pose, held it wedged 9 mm off when knocked and jammed it in every G1
    # lift (215-258 N on its walls, 0 of 9 lifts out), and its front wall blocked the G1's middle finger; a 15 mm lip let
    # the scanner tip out over it; walls sloping back 5 deg from the floor made the well shallow (0.85 mm per degree) and
    # a scanner let go 20 mm above its seat hung 6 mm up. Of twelve releases (upright 5-30 mm above the floor, at the
    # seated lean 2-12 mm above the seat) a 25 or 30 mm lip seated ten, a 35 mm lip all twelve
    CUP_CLEAR = 0.0015                 # handle to the rear and side walls at the floor, per side
    LIP_H = 0.035                      # front lip (the contacts), along the socket from its floor
    STRAIGHT_H = 0.008                 # the rear and side walls' straight part
    SIDE_H = 0.018                     # side walls
    REAR_H = 0.024                     # rear wall: its flare's top, 74 mm above the base's foot, is the cradle's highest point
                                       # (datasheet: 72 mm)
    REAR_DRAFT = 0.                    # the rear wall's straight part (the lean's equation allows a slope)
    REAR_FLARE = math.radians(30.)     # the rear wall above its straight part
    SIDE_FLARE = math.radians(15.)     # the side walls above their straight part

    def socket_lean(self):
        """The resting lean in the socket: angle theta (rad) by which the handle stands turned forward from the socket's
        axis, and the handle frame's origin (bottom centre) in socket coordinates (x forward, z up the socket). Front
        face through the lip's top edge (a, L), front-bottom corner on the floor, rear-bottom corner on the rear wall
        (through (-(a + c), 0), sloping back at d): 2a (1 - cos th) + c - L tan th + 2a sin th tan d = 0."""
        a, c, L, d = self.HANDLE_HALF[0], self.CUP_CLEAR, self.LIP_H, self.REAR_DRAFT
        f = lambda th: 2*a*(1 - math.cos(th)) + c - L*math.tan(th) + 2*a*math.sin(th)*math.tan(d)
        lo, hi = 0., .4
        for _ in range(80):
            mid = .5*(lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        th = .5*(lo + hi)
        return th, np.array([a - a*math.cos(th) - L*math.tan(th), 0., a*math.sin(th)])

    def _seated_scanner_pose(self, cradle):
        """Scanner pose when seated: handle bottom on the cup floor, the handle inclined as in a hand-held grip,
        the window facing the cradle's +x (towards the presentation zone / shelf)."""
        (px, py, pz), q = cradle
        base_h = .030
        # handle bottom centre in scanner frame
        hc, hq = self.handle_frame()
        bottom = hc + rotate(hq, (0., 0., -self.HANDLE_HALF[2]))
        # scanner orientation: pitch the whole scanner so the window points 12 deg down (authored presentation angle)
        pitch = quat_axis_angle((0, 1, 0), math.radians(12.))
        sq = quat_mul(q, pitch)
        # seat: the socket (tilted with the handle, 15 deg rake + 12 deg pitch) stands on the base; its floor's lowest
        # corner (half its fore-aft depth, times the sine of the tilt) at the base top (an 8 mm margin here put the
        # cradle's top at 90 mm against the datasheet's 72 mm)
        tilt = self.HANDLE_RAKE + math.radians(12.)
        lift = (self.HANDLE_HALF[0] + self.CUP_CLEAR + .004)*math.sin(tilt)
        seat = np.array([px, py, pz + base_h + lift])
        origin = seat - rotate(sq, bottom)
        self.cradle = dict(pos=(px, py, pz), quat=q, scanner_quat=sq, seat=seat, handle_quat=quat_mul(sq, hq),
                           origin=tuple(map(float, origin)))
        return origin, sq

    @staticmethod
    def cradle_base_mesh():
        """Teardrop base of the CR2278-PC (Quick Reference Guide p. 3, top view): the round end (94 mm) around the pocket at
        the origin, a narrower nose (60 mm) running forward (+x, the window's side) under the overhanging head; 124 x 94 mm,
        30 mm tall, the top 8 % inside the foot. Cradle frame, base bottom at z = 0."""
        import trimesh
        pts = [(cx + s*r*math.cos(a), s*r*math.sin(a), z) for z, s in ((0., 1.), (.030, .92))
               for cx, r in ((0., .047), (.047, .030)) for a in np.linspace(0., 2*math.pi, 48, endpoint=False)]
        return trimesh.convex.convex_hull(np.asarray(pts))

    def build_cradle(self):
        """Presentation cradle (CR2278 class, 72 x 94 x 122 mm, 151 g, fastened): the teardrop base and a
        socket for the handle (floor, front lip, side walls, raised rear wall; class constants above); nothing under
        the overhanging head (CR2278-PC QRG pp. 3, 5)."""
        c = self.cradle
        (px, py, pz) = c["pos"]
        tm = self.cradle_base_mesh()
        m = self.newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())
        i = self.b.add_shape_convex_hull(-1, mesh=m, xform=self.wp.transform((px, py, pz), c["quat"]),
                                         cfg=self.cfg(gap=P.MESH_CONTACT_GAP_M, mu=P.MU["scanner_cradle"]), label="cradle_base")
        self.materials[i] = "cradle_black"
        seat = np.asarray(c["seat"])
        dh = self.HANDLE_HALF
        a, cl, bw = dh[0], self.CUP_CLEAR, dh[1] + self.CUP_CLEAR      # +x towards the front face, +z up the socket
        w = .004
        # the socket's frame: the seated handle's, turned back by the resting lean about its y axis
        th, t_h = self.socket_lean()
        hs = quat_mul(c["handle_quat"], quat_axis_angle((0, 1, 0), -th))
        c["socket_lean_deg"] = math.degrees(th)
        at = lambda local: tuple(seat + rotate(hs, np.asarray(local, float) - t_h))
        turned = lambda axis, ang: quat_mul(hs, quat_axis_angle(axis, ang))
        # floor under the handle bottom (and the rear clearance)
        self.box("cradle_floor", at((-cl/2, 0., -w/2)), (a + cl/2 + w, bw + w, w/2), "cradle_black", rot=hs, mu=P.MU["scanner_cradle"])
        # front lip (the contacts): the seated handle's front face rests on its top edge
        self.box("cradle_lip", at((a + w/2, 0., self.LIP_H/2)), (w/2, bw + w, self.LIP_H/2), "cradle_black", rot=hs,
                 mu=P.MU["scanner_cradle"])
        # rear wall: straight, 1.5 mm behind the handle, for STRAIGHT_H, then flaring back to REAR_H
        h0, wide = self.STRAIGHT_H, bw + w + .012
        x0 = -(a + cl) - h0*math.tan(self.REAR_DRAFT)
        for name, base, h, d in (("cradle_rear", np.array([-(a + cl), 0., 0.]), h0, self.REAR_DRAFT),
                                 ("cradle_rear_flare", np.array([x0, 0., h0]), self.REAR_H - h0, self.REAR_FLARE)):
            up, out = np.array([-math.sin(d), 0., math.cos(d)]), np.array([-math.cos(d), 0., -math.sin(d)])
            self.box(name, at(base + h/2*up + w/2*out), (w/2, wide, h/2), "cradle_black", rot=turned((0, 1, 0), -d),
                     mu=P.MU["scanner_cradle"])
        # side walls: straight, 1.5 mm beside the handle, for STRAIGHT_H, then flaring out to SIDE_H, from the rear wall
        # to the lip
        for sy in (-1, 1):
            for name, base, h, s in ((f"cradle_side{sy:+d}", np.array([-cl/2, sy*bw, 0.]), h0, 0.),
                                     (f"cradle_side_flare{sy:+d}", np.array([-cl/2, sy*bw, h0]), self.SIDE_H - h0, self.SIDE_FLARE)):
                up = np.array([0., sy*math.sin(s), math.cos(s)]); out = np.array([0., sy*math.cos(s), -math.sin(s)])
                self.box(name, at(base + h/2*up + w/2*out), (a + cl/2 + w, w/2, h/2), "cradle_black",
                         rot=turned((1, 0, 0), -sy*s), mu=P.MU["scanner_cradle"])
        # the socket's frame, for robots putting the scanner back in (insertion_pose)
        c["socket_quat"] = tuple(map(float, hs))
        c["socket_axis"] = tuple(map(float, rotate(hs, (0., 0., 1.))))
        c["socket_origin"] = tuple(map(float, seat - rotate(hs, t_h)))
        # pedestal: the floor's underside down to the base's top (a hull, so nothing visible is without a collider)
        import trimesh
        corners = [np.asarray(at((x, y, -w))) for x in (-(a + cl + w), a + w) for y in (-(bw + w), bw + w)]
        base_top = c["pos"][2] + .030*.999
        foot = [np.array([q_[0], q_[1], base_top - .004]) for q_ in corners]
        hull = trimesh.convex.convex_hull(np.asarray(corners + foot))
        mp = self.newton.Mesh(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32).ravel())
        k = self.b.add_shape_convex_hull(-1, mesh=mp, cfg=self.cfg(gap=P.MESH_CONTACT_GAP_M, mu=P.MU["scanner_cradle"]),
                                         label="cradle_pedestal")
        self.materials[k] = "cradle_black"
        self.fixtures["cradle"] = dict(c)

    def insertion_pose(self, height, back=0.):
        """Scanner pose (origin, quaternion) for putting it into its cradle: upright in the socket (handle parallel to its
        walls, no lean), the handle's bottom `height` m above the socket floor and `back` m behind the middle of its
        fore-aft clearance. Set in like this and let go, the scanner drops onto the floor and its head's weight turns it
        forward onto the lip (CR2278-PC QRG p. 5: inserted straight down, it "naturally rotates forward")."""
        c = self.cradle
        hc, hq = self.handle_frame()
        bottom = hc + rotate(hq, (0., 0., -self.HANDLE_HALF[2]))
        q = quat_mul(c["socket_quat"], (-hq[0], -hq[1], -hq[2], hq[3]))
        target = np.asarray(c["socket_origin"]) + rotate(c["socket_quat"], (-self.CUP_CLEAR/2 - back, 0., height))
        return target - rotate(q, bottom), q

    def keypad(self, center, yaw=0.):
        """17-key numeric keypad with mechanical key switches (Cherry MX Red class): each keycap is a
        prismatic joint with a preloaded spring (0.30 N at rest, 0.45 N at the 2.0 mm operating point, stop at
        4.0 mm)."""
        wp = self.wp
        cx, cy, cz = center
        q = quat_axis_angle((0, 0, 1), yaw)
        self.box("keypad_case", (cx, cy, cz + .012), (.045, .07, .012), "keypad_black", rot=q, mu=.5)
        pitch = P.KEY_PITCH_M
        # standard numpad: rows top->bottom
        layout = [["NUM", "/", "*", "-"], ["7", "8", "9", "+"], ["4", "5", "6", None], ["1", "2", "3", "ENTER"], ["0", None, ".", None]]
        k_rate = (P.KEY_FORCE_OPERATE_N - P.KEY_FORCE_INITIAL_N)/P.KEY_OPERATE_M
        for r, row in enumerate(layout):
            for c, name in enumerate(row):
                if name is None:
                    continue
                w, h = 1, 1
                if name == "0":
                    w = 2
                if name in ("+", "ENTER"):
                    h = 2
                u = (c + w/2 - 2.)*pitch
                v = (2.5 - r - h/2)*pitch
                top = cz + .024 + .004 + P.KEY_TRAVEL_M + .001      # keycap centre: 1 mm clear of the case at full travel (the joint limit is the only stop; a keycap bottoming on the case at the limit rang)
                pos = (cx + math.cos(yaw)*u - math.sin(yaw)*v, cy + math.sin(yaw)*u + math.cos(yaw)*v, top)
                kb = self.b.add_link(xform=wp.transform(pos, q), label=f"key_{name}")
                half = ((w*pitch - .0025)/2, (h*pitch - .0025)/2, .004)
                m_key = P.KEY_MASS_KG*w*h
                self.box(f"keycap_{name}", (0., 0., 0.), half, "keycap", body=kb,
                         density=m_key/(8*half[0]*half[1]*half[2]), mu=.5)
                # armature 1e-5 kg (0.5 % of a keycap, numerical)
                j = self.b.add_joint_prismatic(-1, kb, parent_xform=wp.transform(pos, q), child_xform=wp.transform_identity(),
                                               axis=(0., 0., -1.), limit_lower=0., limit_upper=P.KEY_TRAVEL_M,
                                               target_pos=-P.KEY_FORCE_INITIAL_N/k_rate, target_ke=k_rate,
                                               target_kd=2*P.KEY_DAMPING_RATIO*math.sqrt(k_rate*m_key),
                                               armature=1e-5, effort_limit=5., label=f"key_{name}_travel")
                self.b.add_articulation([j], label=f"key_{name}")
                self.keys[name] = dict(body=kb, joint=j, pos=pos, size=(w*pitch, h*pitch))
        self.fixtures["keypad"] = dict(center=center, yaw=yaw)

    def _tray_keypad(self):
        """The keypad at the centre of the cart tray (G1: turned to face the humanoid at the tray's edge, g1.TRAY)."""
        if not self.g1:
            return self.keypad((0.0, self.tray_y, self.riser_top), yaw=0.)
        from . import g1 as G1
        centre, _ = self.cart_point((0.0, self.tray_y, self.riser_top))
        return self.keypad(centre, yaw=self.cart_pose[1] + G1.TRAY["keypad_yaw"])

    # ------------------------------------------------------------------ items
    def add_item(self, name, sku_id, pos, yaw=0., upright=True, rot=None):
        """A stock unit as a free rigid body with its SKU's mass and finish; its barcode's pose is recorded."""
        s = SKUS[sku_id]
        q = rot or quat_axis_angle((0, 0, 1), yaw)
        body = self.free_body(name, pos, q)
        mu = s.mu
        if s.shape == "box":
            # MuJoCo 3.11's box-box routine returned a contact 584 mm deep between the 12.5 kg water case and the counter
            # top during a reset's settle (counter, checkout-bulky, seed 3: 379 kN, the case thrown at 31.7 m/s): heavy
            # boxes collide as hulls; light ones keep the box routine (Scene.box)
            w, d, h = s.dims
            self.box(f"{name}_hull", (0., 0., 0.), (w/2, d/2, h/2), f"sku_{sku_id}", body=body, density=s.mass/(w*d*h), mu=mu,
                     hull=s.mass >= P.BOX_AS_HULL_MIN_MASS_KG)
        elif neck(s):
            self.necked_hull(name, s, body, mu)
        else:
            r, h = s.dims
            self.hull_cylinder(f"{name}_hull", (0., 0., 0.), r, h/2, f"sku_{sku_id}", body, density=s.mass/(math.pi*r*r*h), mu=mu)
        self.set_mass(body, s.mass)
        self.items[name] = dict(body=body, sku=sku_id, name=s.name)
        self._item_label(name, body, s)
        if s.extra.get("inner"):
            self._inner_units(name, body, s)
        return body

    def add_book(self, name, sku_id, pos, rot):
        """A hardcover book with a hinged front cover: the text block + back cover as one free body, the front
        board (hardcover) or card (paperback) of P.book_cover(sku) as a link on a revolute joint along the spine (the
        book's -x edge), opening away from the block, with spine friction. An opened cover
        rests on the counter beside the book (about 190 deg): the counter stops it, not the joint. A 180 deg joint
        stop held it cantilevered at the top of the block and the stop's impact threw the 0.65 kg book 27 mm;
        the joint limit is the cloth hinge's fold-back (270 deg)."""
        s = SKUS[sku_id]
        w, d, h = s.dims
        t, m_cover, hinge_friction = P.book_cover(s)
        body = self.b.add_link(xform=self.wp.transform(pos, rot), label=name)
        self.bodies[name] = body
        free = self.b.add_joint_free(body, label=f"{name}_free")
        m_block = s.mass - m_cover
        self.box(f"{name}_hull", (0., t/2, 0.), (w/2, (d - t)/2, h/2), f"sku_{sku_id}", body=body,
                 density=m_block/(w*(d - t)*h), mu=s.mu)
        hinge_local = np.array([-w/2, -d/2 + t/2, 0.])
        hinge_world = np.asarray(pos) + rotate(rot, hinge_local)
        cover = self.b.add_link(xform=self.wp.transform(tuple(hinge_world), rot), label=f"{name}_cover")
        self.bodies[f"{name}_cover"] = cover
        self.box(f"{name}_cover_board", (w/2, 0., 0.), (w/2, t/2, h/2), f"sku_{sku_id}", body=cover,
                 density=m_cover/(w*t*h), mu=s.mu)
        # axis -z: a positive angle swings the board's free edge towards -y, away from the text block
        j = self.b.add_joint_revolute(body, cover, parent_xform=self.wp.transform(tuple(hinge_local), self.wp.quat_identity()),
                                      child_xform=self.wp.transform_identity(), axis=(0., 0., -1.), limit_lower=0.,
                                      limit_upper=1.5*math.pi, friction=hinge_friction, armature=min(1e-5, .1*m_cover*w*w/3),
                                      label=f"{name}_spine")
        self.b.add_articulation([free, j], label=name)
        self.items[name] = dict(body=body, sku=sku_id, name=s.name, cover_body=cover, hinge=j)
        self._item_label(name, body, s)
        return body, cover

    def _inner_units(self, name, body, s):
        """Units inside a clear shrink-wrapped pack: a grid of the inner SKU, each at a random azimuth, their own
        barcodes registered as kind 'inner' (reading one instead of the pack's code is a counting error)."""
        inner = SKUS[s.extra["inner"]]
        nx, ny = s.extra["grid"]
        r, h = inner.dims
        W, D, H = s.dims
        units = []
        for i in range(nx):
            for j in range(ny):
                cx = -W/2 + (i + .5)*W/nx
                cy = -D/2 + (j + .5)*D/ny
                az = float(self.rng.uniform(0, 360))
                units.append((cx, cy, -H/2 + h/2, az))
                bc = inner.barcode
                a = math.radians(bc.angle_deg + az)
                n = np.array([math.sin(a), -math.cos(a), 0.])
                sym = symbol_for(inner)
                from .artwork import render_symbol
                _, w_mm, h_mm = render_symbol(sym, 2)
                centre = np.array([cx, cy, -H/2 + h/2]) + n*r + np.array([0., 0., bc.v])
                self.label(f"{name}_inner_{i}_{j}", body, sym, centre, n, (0, 0, 1), (w_mm/1e3 + .002, h_mm/1e3 + .002),
                           inner.finish, kind="inner", item=name, curvature_radius=r, text=bc.orientation)
        self.items[name]["inner"] = units

    def _item_label(self, name, body, s):
        sym = symbol_for(s)
        bc = s.barcode
        from .artwork import render_symbol
        _, w_mm, h_mm = render_symbol(sym, 2)
        w, h = w_mm/1e3 + .002, h_mm/1e3 + .002
        finish = bc.finish or s.finish
        if s.shape == "cyl":
            r, hgt = s.dims
            a = math.radians(bc.angle_deg)
            n = np.array([math.sin(a), -math.cos(a), 0.])        # front (angle 0) faces -y
            centre = n*r + np.array([0., 0., bc.v])
            up = np.array([0., 0., 1.])
            size = (w, h) if bc.orientation == "picket" else (w, h)
            self.label(f"{name}_code", body, sym, centre, n, up, size, finish, item=name, curvature_radius=r,
                       text=bc.orientation)
        else:
            W, D, H = s.dims
            faces = {"front": ((0, -1, 0), (0, 0, 1), (1, 0, 0), D/2), "back": ((0, 1, 0), (0, 0, 1), (-1, 0, 0), D/2),
                     "right": ((1, 0, 0), (0, 0, 1), (0, 1, 0), W/2), "left": ((-1, 0, 0), (0, 0, 1), (0, -1, 0), W/2),
                     "top": ((0, 0, 1), (0, 1, 0), (1, 0, 0), H/2), "bottom": ((0, 0, -1), (0, 1, 0), (-1, 0, 0), H/2)}
            n, vv, uu, off = (np.asarray(x, float) if i < 3 else x for i, x in enumerate(faces[bc.face]))
            centre = n*off + uu*bc.u + vv*bc.v
            up = vv if bc.orientation == "picket" else uu
            self.label(f"{name}_code", body, sym, centre, n, up, (w, h), finish, item=name)

    # ------------------------------------------------------------------ shelving
    def shelving(self, bays=(-1, 0, 1)):
        """Boltless steel shelving bays (Uline H-7115 class): angle posts, front/back beams, particle-board
        decks on the 1.5 in grid."""
        for k in bays:
            x0 = k*BAY_W
            for sx in (-1, 1):
                for yy in (.019, BAY_D - .019):
                    self.box(f"post_{k}_{sx}_{yy:.2f}", (x0 + sx*(BAY_W/2 - .019), yy, P.SHELF_BAY_M[2]/2),
                             (.019, .019, P.SHELF_BAY_M[2]/2), "shelf_steel", mu=.4)
            for d, top in enumerate(DECKS):
                self.box(f"beam_front_{k}_{d}", (x0, BEAM_T/2 - .005, top - BEAM_H/2), (BAY_W/2 - .038, BEAM_T/2, BEAM_H/2),
                         "shelf_steel", mu=.4)
                self.box(f"beam_back_{k}_{d}", (x0, BAY_D - BEAM_T/2, top - BEAM_H/2), (BAY_W/2 - .038, BEAM_T/2, BEAM_H/2),
                         "shelf_steel", mu=.4)
                self.box(f"deck_{k}_{d}", (x0, BAY_D/2, top - P.DECK_THICK_M/2),
                         (BAY_W/2 - .038, BAY_D/2 - BEAM_T + .002, P.DECK_THICK_M/2), "particle_board", mu=.30)
                for s in range(SLOTS_PER_DECK):
                    sx0 = x0 - BAY_W/2 + .038 + s*(BAY_W - .076)/SLOTS_PER_DECK
                    sx1 = sx0 + (BAY_W - .076)/SLOTS_PER_DECK
                    code = f"A{k + 3:02d}-{d + 1}-{'ABC'[s]}"
                    self.slots[code] = dict(x=(sx0, sx1), y=(.005, BAY_D - .01), z=top, bay=k, deck=d, slot=s)
        self.fixtures["shelving"] = dict(bays=list(bays), decks=list(DECKS), width=BAY_W, depth=BAY_D)

    def location_labels(self):
        from .artwork import Symbol
        for code, s in self.slots.items():
            sym = Symbol("Code128", code, .381, bar_height_mm=11.0, hri=False)
            centre = (s["x"][0] + .045, -.0052, s["z"] - BEAM_H/2)
            self.label(f"loc_{code}", -1, sym, centre, (0, -1, 0), (0, 0, 1), (.0762, .0254), "matte", kind="location",
                       text=code)

    def planogram(self, code, entries, depth_rows=None):
        """Fill slot `code` with stock: entries [(sku, facings, depth)] placed left to right, front to back, each
        unit standing on the deck with 3-8 mm gaps, fronts facing the aisle (-y)."""
        s = self.slots[code]
        x = s["x"][0] + .008
        placed = []
        for sku_id, facings, depth in entries:
            sku = SKUS[sku_id]
            w = 2*sku.dims[0] if sku.shape == "cyl" else sku.dims[0]
            d = 2*sku.dims[0] if sku.shape == "cyl" else sku.dims[1]
            h = sku.dims[1] if sku.shape == "cyl" else sku.dims[2]
            for f in range(facings):
                for r in range(depth):
                    gap_x = float(self.rng.uniform(.003, .008)); gap_y = float(self.rng.uniform(.003, .008))
                    cx = x + w/2
                    cy = s["y"][0] + .01 + r*(d + gap_y) + d/2
                    if cy + d/2 > s["y"][1] or cx + w/2 > s["x"][1]:
                        continue
                    yaw = float(self.rng.normal(0., math.radians(2.)))
                    if sku.shape == "cyl":
                        yaw += float(self.rng.uniform(-.35, .35))       # cans are faced roughly, labels drift
                    name = f"{code}_{sku_id}_{f}_{r}"
                    self.add_item(name, sku_id, (cx, cy, s["z"] + h/2 + .0005), yaw)
                    self.items[name].update(slot=code, row=r, facing=f)
                    placed.append(name)
                x += w + float(self.rng.uniform(.003, .008))
        return placed

    def overstock(self, bay, deck):
        """Static overstock cases on the out-of-reach decks (1 and 5): corrugated shipping cases, disclosed as
        static (no robot can reach these decks from the cart)."""
        x0 = bay*BAY_W
        top = DECKS[deck]
        for s, (w, d, h) in enumerate(((.30, .40, .25), (.26, .36, .22))):
            cx = x0 - BAY_W/4 + s*BAY_W/2
            self.box(f"overstock_{bay}_{deck}_{s}", (cx, BAY_D/2, top + h/2), (w/2, d/2, h/2), "carton")

    # ------------------------------------------------------------------ presets
    def stockroom(self):
        """Back-of-store stockroom, shelving bay A03 in front of the cart."""
        self.build_cradle()
        self._tray_keypad()
        self.shelving()
        self.location_labels()
        for bay in (-1, 0, 1):
            self.overstock(bay, 0)
            self.overstock(bay, 4)
        from .stock import stock_stockroom
        stock_stockroom(self)

    def counter(self):
        """Front-of-store service counter with the scanner in its presentation cradle."""
        self.counter_body(dressing="store")
        self.build_cradle()
        self._tray_keypad()
        from .stock import stock_counter
        stock_counter(self)

    def library(self):
        self.counter_body(dressing="library")
        self.build_cradle()
        self._tray_keypad()
        from .stock import stock_library
        stock_library(self)

    def counter_body(self, dressing):
        z = P.COUNTER_TOP_M
        # the counter top as an 8-vertex hull: MuJoCo's box-box routine twice returned a contact against it hundreds of
        # millimetres deep with a box resting still on it -- the 12.5 kg water case (584 mm, at a reset) and
        # the bagging tote's 4 mm floor (753 mm, 0.9 kg, during E17's held-whip settle: the tote thrown at 26.9 m/s; it
        # recurred on rerun at another tick). With the top a hull the same E17 sequence ran clean, and the resting creep
        # of the baskets' contents is unchanged
        self.box("counter_top", (0., .305, z - .019), (.915, .305, .019), "laminate" if dressing == "store" else "wood_counter", mu=.35,
                 hull=True)
        self.box("counter_front", (0., .61 - .01, (z - .038)/2), (.915, .01, (z - .038)/2), "counter_panel")
        self.box("counter_back", (0., .02, (z - .038)/2 + .06), (.915, .01, (z - .038)/2 - .06), "counter_panel")
        for sx in (-.9, .9):
            self.box(f"counter_end_{sx}", (sx, .305, (z - .038)/2), (.015, .3, (z - .038)/2), "counter_panel")
        self.fixtures["counter"] = dict(top=z, x=(-.915, .915), y=(0., .61), dressing=dressing)

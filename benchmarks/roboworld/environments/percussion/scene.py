"""Physics scene (Newton 1.5 ModelBuilder): therapy room furniture, the robot cart and its tool rack, the robots, the
instruments and the tools.

Every touchable part exists physically. The vibrating parts are rigid bodies here and modal structures in modal.py;
their contact with registered strikers (stick tips and shafts, mallet heads and handles, the felt beater, fingertip
pads, and every robot surface that can meet an instrument from above: the Franka hand housing, link 7
and finger boxes, the G1 palms and all Dex3 links, as capsule proxies) is the vibro-contact of world.py, so the tool
bodies are excluded from MuJoCo collision with the bars and the drumhead body (MuJoCo <exclude>, emitted by Newton for
whole-body filters; world.py checks it). Robot links are not excluded (fingers grasp bars by their sides): the MuJoCo
colliders of a bar's top face and of the head sit below the vibrating surface (6 mm and 8 mm), so a robot surface
meets the vibro-contact first and the rigid backstop only beyond what the contact and head compliance give at 100+ N.
The room has walls and a box collider per piece of furniture (params.FURNITURE), and a carpet floor with rolling
resistance (world.py sets it on the compiled MuJoCo model).
"""
import math
import numpy as np

from . import params as P
from . import layout as LY

BAR_TOP_RECESS_M = 0.006       # rigid top face below the vibrating one: a fingertip pad box (its lower corner reaches
                                # ~4 mm below its vibro-contact proxy sphere when the hand is tilted) must meet the vibro-contact first
BAR_BOTTOM_LIFT_M = 0.003       # bar boxes end 3 mm above the bar's underside: only its four feet rest on the felt
HEAD_RECESS_M = 0.008
INSTRUMENT_GAP_M = 0.0005


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    s = math.sin(angle/2)
    return (float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(math.cos(angle/2)))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def rotate(q, v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).apply(v)


def furniture_box(f):
    """(world centre, half extents, yaw) of a params.FURNITURE entry's collider (its model's scaled bounding box)."""
    lo, hi = np.asarray(f["bounds"][0], float)*f["scale"], np.asarray(f["bounds"][1], float)*f["scale"]
    c_l, half = 0.5*(lo + hi), 0.5*(hi - lo)
    cy, sy = math.cos(f["yaw"]), math.sin(f["yaw"])
    c = (f["xy"][0] + cy*c_l[0] - sy*c_l[1], f["xy"][1] + sy*c_l[0] + cy*c_l[1], c_l[2])
    return c, tuple(half), f["yaw"]


Q_ID = (0., 0., 0., 1.)
Q_Z_TO_X = quat_axis_angle((0, 1, 0), math.pi/2)        # a capsule's local z along the parent x
Q_Z_TO_Y = quat_axis_angle((1, 0, 0), -math.pi/2)


class Scene:
    FRANKA_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                           ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                           ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                           ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))
    PAD_PROXY_R = 0.0076

    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.layout = LY.preset_layout(config)
        self.materials, self.visual = {}, set()
        self.bodies, self.items, self.cameras = {}, {}, {}
        self.static_boxes = []
        self.strikers, self.surfaces, self.exclude_bodies = [], [], []
        self.fixtures = {}
        self.g1 = LY.is_g1(config)
        self.mics = {"station_mic": tuple(P.STATION_MIC), "client_mic": tuple(P.CLIENT_EAR)}
        self.room()
        if self.g1:
            self.sensor_stand()
        else:
            self.cart()
        self.robots()
        if self.layout["table"]:
            self.table()
        if self.layout["xylophone"]:
            self.xylophone(self.layout["xylophone"])
        for rb in self.layout["resonator_bars"]:
            self.resonator_bar(rb)
        if self.layout["drum"]:
            self.drum(self.layout["drum"])
        if self.layout["bar_tray"]:
            self.bar_tray(self.layout["bar_tray"])
        if self.layout["speaker"]:
            self.speaker(self.layout["speaker"])
        self.tools()
        self.views()
        self.apply_exclusions()
        # rigid instrument parts that sound when struck (world._rigid_clicks): hoop, stand and pins ring as metal, the
        # shell, the xylophone frame and the resonator boxes knock as wood
        self.click_shapes = {}
        for i in range(self.b.shape_count):
            lab = self.b.shape_label[i] or ""
            if lab.startswith(("drum_hoop", "stand_column", "stand_arm", "stand_leg", "pin_", "rpin_")):
                self.click_shapes[i] = "metal"
            elif lab.startswith(("drum_shell", "xylo_hull", "resbox_")) and "foot" not in lab:
                self.click_shapes[i] = "wood"

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        # instruments, tools and furniture: 0.5 mm contact gap (resting bodies are millimetres apart by design: bar feet
        # on felt, pins in 1 mm clearance holes, tools in saddle cheeks); the robots keep the 2 mm of barcode_scanning E17
        v = dict(mu=0.5, ke=2e4, kd=100., density=0., gap=INSTRUMENT_GAP_M, margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.newton.ModelBuilder.ShapeConfig(
            density=0., has_shape_collision=False, has_particle_collision=False)
        i = self.b.add_shape_box(body, xform=self.wp.transform(pos, rot or Q_ID), hx=half[0], hy=half[1], hz=half[2],
                                 cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        if body < 0:
            self.static_boxes.append((name, tuple(map(float, pos)), tuple(map(float, half)), material,
                                      tuple(rot or Q_ID)))
        return i

    def capsule(self, name, body, p0, p1, r, material, **kw):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        c, d = 0.5*(p0 + p1), p1 - p0
        L = float(np.linalg.norm(d))
        z = np.array([0., 0., 1.])
        ax = np.cross(z, d/L)
        s = float(np.linalg.norm(ax))
        q = quat_axis_angle(ax, math.atan2(s, float(z @ d)/L)) if s > 1e-9 else (Q_ID if d[2] > 0 else quat_axis_angle((1, 0, 0), math.pi))
        i = self.b.add_shape_capsule(body, xform=self.wp.transform(tuple(c), q), radius=r, half_height=L/2,
                                     cfg=self.cfg(**kw), label=name)
        self.materials[i] = material
        return i

    def sphere(self, name, body, p, r, material, **kw):
        i = self.b.add_shape_sphere(body, xform=self.wp.transform(tuple(map(float, p)), Q_ID), radius=r,
                                    cfg=self.cfg(**kw), label=name)
        self.materials[i] = material
        return i

    def free_body(self, name, pos, rot, mass, com, inertia):
        b = self.b.add_body(xform=self.wp.transform(tuple(map(float, pos)), rot), label=name, mass=float(mass),
                            com=self.wp.vec3(*map(float, com)),
                            inertia=self.wp.mat33(*np.diag(np.asarray(inertia, float)).ravel().tolist()), lock_inertia=True)
        self.bodies[name] = b
        return b

    # ------------------------------------------------------------------ room, cart, table
    def room(self):
        """Floor (carpet tiles: Coulomb 0.6 plus rolling resistance, set on the compiled MuJoCo model in world.py), the
        four walls and a box collider per piece of furniture (params.FURNITURE: the bounding boxes of the pinned models
        dressing.py draws), all static and hidden (the dressing renders them)."""
        self.b.add_ground_plane(cfg=self.cfg(mu=0.6), label="floor")
        self.floor_shape = self.b.shape_count - 1
        self.materials[self.floor_shape] = "floor"
        R, t = P.ROOM, P.WALL_THICK_M
        x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
        for name, c, half in (("wall_back", (0.5*(x0 + x1), y0 - t/2, h/2), ((x1 - x0)/2 + t, t/2, h/2)),
                              ("wall_front", (0.5*(x0 + x1), y1 + t/2, h/2), ((x1 - x0)/2 + t, t/2, h/2)),
                              ("wall_left", (x0 - t/2, 0.5*(y0 + y1), h/2), (t/2, (y1 - y0)/2, h/2)),
                              ("wall_right", (x1 + t/2, 0.5*(y0 + y1), h/2), (t/2, (y1 - y0)/2, h/2))):
            self.box(name, c, half, "collision_hidden", mu=0.5)
        boxes = [(f["name"], *furniture_box(f)) for f in P.FURNITURE]
        px, py = P.PIANO["xy"]
        boxes.append(("piano", (px, py, P.PIANO["body_z"]), P.PIANO["body_half"], 0.0))
        boxes.append(("piano_stand", (px, py, 0.36), (0.36, 0.06, 0.36), 0.0))
        if self.layout["preset"] == "gait":            # the activity table moved to the side wall (dressing.py)
            T = P.TABLE
            boxes.append(("side_table", (R["x0"] + 0.35, 2.0, T["top_z"]/2), (T["half"][1], T["half"][0], T["top_z"]/2), 0.0))
        self.furniture_boxes = []
        for name, c, half, yaw in boxes:
            self.box(f"furniture_{name}", c, half, "collision_hidden", rot=quat_axis_angle((0, 0, 1), yaw), mu=0.5)
            self.furniture_boxes.append(dict(name=name, center=tuple(map(float, c)), half=tuple(map(float, half)),
                                             yaw=float(yaw)))

    def cart(self):
        """The robot station: a parked mobile dual-arm platform (castors braked, so a static body here): base, column and
        shoulder beam carrying both arm bases at 0.80 m, the sensor mast at its back (authored, params.STATION). The
        colliders are plain boxes; dressing.py draws the platform's shell."""
        S = P.STATION
        for name, c, h in (("station_base", S["base_c"], S["base_half"]), ("station_column", S["column_c"], S["column_half"]),
                           ("station_beam", S["beam_c"], S["beam_half"])):
            self.box(name, c, h, "collision_hidden", mu=0.4)
        mx, my = S["mast_xy"]
        z = S["beam_c"][2] + S["beam_half"][2]
        self.box("mast", (mx, my, (z + 1.58)/2), (0.02, 0.02, (1.58 - z)/2), "collision_hidden")
        self.box("mast_arm", (mx, my + 0.10, 1.56), (0.015, 0.12, 0.012), "collision_hidden")
        self.fixtures["mast"] = dict(x=mx, y=my, top=1.58)

    def sensor_stand(self):
        """G1 configuration: no robot station. The station camera and the cue light stand on a tripod mast at the back
        left; the robot's own microphone is the G1 head array (g1.HEAD_MIC)."""
        mx, my = -0.75, -0.30
        self.box("mast", (mx, my, 0.80), (0.015, 0.015, 0.78), "collision_hidden")
        self.box("mast_arm", (mx, my + 0.10, 1.56), (0.015, 0.12, 0.012), "collision_hidden")
        for k in range(3):
            a = 2*math.pi*k/3 + math.pi/6
            self.box(f"mast_leg_{k}", (mx + 0.14*math.cos(a), my + 0.14*math.sin(a), 0.012), (0.012, 0.012, 0.012),
                     "rubber", mu=0.9)
        self.fixtures["mast"] = dict(x=mx, y=my, top=1.58, tripod=True)

    def table(self):
        T = P.TABLE
        cx, cy, z = T["cx"], T["cy"], T["top_z"]
        hx, hy = T["half"]
        self.box("table_top", (cx, cy, z - T["thick"]/2), (hx, hy, T["thick"]/2), "table_laminate", mu=0.45)
        for sx in (-hx + 0.03, hx - 0.03):
            for sy in (-hy + 0.03, hy - 0.03):
                self.box(f"table_leg_{sx:+.2f}_{sy:+.2f}", (cx + sx, cy + sy, (z - T["thick"])/2), (0.02, 0.02, (z - T["thick"])/2),
                         "table_frame")
        self.box("table_apron", (cx, cy - hy + 0.03, z - T["thick"] - 0.04), (hx - 0.05, 0.01, 0.04), "table_frame")

    # ------------------------------------------------------------------ robots (wine_pouring/scene.py pattern)
    def robots(self):
        if self.g1:
            return self._g1_robot()
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("percussion", self.c.robot)
        self.robot_setup = setup
        self._hardware(setup)
        mounts = {k: mount((x, y, P.ARM_BASE_Z_M), P.ARM_YAW) for k, (x, y) in P.ARM_BASE_XY.items()}
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        for i in range(first_shape, self.b.shape_count):
            box = self.b.shape_type[i] in (self.newton.GeoType.BOX, self.newton.GeoType.PLANE)
            self.b.shape_gap[i] = P.CONTACT_GAP_M
        self.joints, self.arms = {}, {}
        self.pad_shapes = {}
        self.finger_bodies = {}
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            self.arms[name] = handle
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._fingertips(name, handle)
            self._wrist_camera(name, handle)
            self._hand_proxies(name, handle)

    def _g1_robot(self):
        """Unitree G1 + two Dex3-1 hands standing on the floor (g1.py; rig robots/rigs/percussion-g1.json). Floating base:
        no weld, no pelvis support. No Franka hardware tables: the Dex3 links collide with their URDF convex hulls; each
        index, middle and thumb tip carries a sphere proxy registered as a vibro-contact striker (rubber pad law)."""
        from robots.catalog import resolve
        from robots.backends.newton import mount
        from scipy.spatial.transform import Rotation
        import robots
        from . import g1 as G1
        setup = resolve(G1.SETUP)
        self.robot_setup = setup
        self.arm_hw, self.pad_shapes, self.finger_bodies = {}, {}, {}
        lay = G1.LAYOUT[self.c.preset]
        (x, y), yaw = lay["pelvis"], lay["yaw"]
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts={G1.MOUNT: mount((x, y, G1.PELVIS_Z), yaw)})
        self.robot_bodies = list(range(first, self.b.body_count))
        for i in range(first_shape, self.b.shape_count):
            self.b.shape_gap[i] = P.CONTACT_GAP_M
        self.joints, self.arms = {}, {}
        handle = self.rig_pending.instances[G1.INSTANCE]
        for j, entry in handle.joints.items():
            self.joints[j] = entry
        self.arms[G1.INSTANCE] = handle
        for i in handle.shape_ids:
            self.materials[i] = "robot"
        labels = self.b.body_label
        for side in ("left", "right"):
            self.finger_bodies[side] = [i for i in self.robot_bodies if f"/{side}_hand_" in labels[i]]
            for link in G1.FINGERTIP_LINKS:
                body = G1.body_index(self.b, f"{side}_hand_{link}_link")
                tip, d = G1.fingertip_points(self.b, body)
                self.strikers.append(dict(name=f"{side}_{link}_tip", body=body, shape="sphere",
                                          p=np.asarray(tip) - d*G1.FINGERTIP_R, r=G1.FINGERTIP_R, material="rubber_pad",
                                          tool=None, arm=side, top_only=True, fingertip=True))
            # the rest of the hand (palm with the wrist yaw link, every Dex3 finger link): Dex3 tactile skin (rubber law);
            # the wrist pitch/roll links and the forearm (elbow link) too - hard shells ('plastic' law), so a
            # forearm laid on the head or a bar is heard and damps instead of sinking to the recessed rigid colliders
            tips = {s["body"]: s for s in self.strikers if s.get("fingertip") and s["arm"] == side}
            hand = [G1.body_index(self.b, f"{side}_wrist_yaw_link")] + self.finger_bodies[side]
            arm = [G1.body_index(self.b, f"{side}_{n}_link") for n in ("wrist_pitch", "wrist_roll", "elbow")]
            for body in hand + arm:
                shapes = [i for i in range(first_shape, self.b.shape_count) if self.b.shape_body[i] == body]
                self._proxies(f"{side}_{labels[body].split('/')[-1]}", body, shapes, "plastic" if body in arm else "rubber_pad",
                              side, trim=tips.get(body))
        self.g1_hw = dict(pelvis=G1.body_index(self.b, "pelvis"), torso=G1.body_index(self.b, "torso_link"),
                          wrist={s: G1.body_index(self.b, f"{s}_wrist_yaw_link") for s in ("left", "right")},
                          feet={s: G1.body_index(self.b, f"{s}_ankle_roll_link") for s in ("left", "right")},
                          layout=dict(lay))
        G1.head_camera(self)

    def _hardware(self, setup):
        from robots.catalog import model
        data = setup.to_dict()
        self.arm_hw = {}
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            x, y = P.ARM_BASE_XY[inst["mount"]]
            self.arm_hw[name] = dict(model=inst["model"], mount=((x, y), P.ARM_YAW), mount_name=inst["mount"],
                                     wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                                     arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)

    def _fingertips(self, arm, handle):
        """Franka Hand finger colliders (URDF boxes; Panda meshes get the same boxes); the rubber tip box is the pad. Each
        pad also carries a sphere proxy at its outer end, registered as a vibro-contact striker (rubber on wood/film) for
        taps and damping touches on bar tops and the drumhead (top faces only: sides are grasped through MuJoCo)."""
        from scipy.spatial.transform import Rotation
        b = self.b
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            boxes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.BOX]
            pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
            pad_xyz = None
            if boxes:
                for i in boxes:
                    tf = np.asarray(b.shape_transform[i])
                    if abs(float(tf[2]) - pad_z) < 1e-4:
                        self.pad_shapes[i] = b.body_label[body]
                        b.shape_material_mu[i] = P.PAD_MU
                        pad_xyz = tf[:3]
            else:
                meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
                if not meshes:
                    continue
                tf = np.asarray(b.shape_transform[meshes[0]], float)
                Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
                for k, (xyz, rpy, size) in enumerate(self.FRANKA_FINGER_BOXES):
                    R = Rm*Rotation.from_euler("xyz", rpy)
                    p = pm + Rm.apply(xyz)
                    i = self.box(f"{b.body_label[body].split('/')[-1]}_box{k}", tuple(p), tuple(np.asarray(size)/2),
                                 "robot_dark", body=body, rot=tuple(R.as_quat()), mu=P.PAD_MU)
                    if abs(xyz[2] - pad_z) < 1e-6:
                        self.pad_shapes[i] = b.body_label[body]
                        pad_xyz = p
            if pad_xyz is None:
                continue
            tip = np.array([pad_xyz[0], pad_xyz[1], pad_xyz[2] + 18.5e-3/2 - self.PAD_PROXY_R])
            self.finger_bodies.setdefault(arm, []).append(body)
            self.strikers.append(dict(name=f"{arm}_{b.body_label[body].split('/')[-1]}_pad", body=body, shape="sphere",
                                      p=tip, r=self.PAD_PROXY_R, material="rubber_pad", tool=None, arm=arm,
                                      top_only=True, fingertip=True))

    def _hand_proxies(self, arm, handle):
        """Every other Franka surface that can meet an instrument from above - the hand housing, the wrist camera case
        and link 7 (all one MuJoCo body: the hand is fused onto link 7) and the fingers' non-pad boxes - as vibro-contact
        strikers with the 'plastic' law, so a hand that slaps, leans on or mutes a bar or the head is heard, damps it
        and is logged instead of sinking silently to the recessed rigid colliders."""
        b = self.b
        wl = self.arm_hw[arm]["wrist_link"]
        link7 = [i for i in handle.body_ids if wl and b.body_label[i].endswith(wl)]
        # links 6 and 5 (the wrist block and the forearm) as well - the arm can lay them on an instrument
        fore = [i for i in handle.body_ids if wl and b.body_label[i].endswith((wl[:-1] + "6", wl[:-1] + "5"))]
        for body in link7 + fore + self.finger_bodies.get(arm, []):
            shapes = [i for i in range(b.shape_count) if b.shape_body[i] == body and i not in self.pad_shapes]
            self._proxies(f"{arm}_{b.body_label[body].split('/')[-1]}", body, shapes, "plastic", arm)

    def _proxies(self, name, body, shapes, material, arm, trim=None, max_caps=3):
        """Capsule proxies covering a robot body's collision shapes (boxes: their corners; meshes: their vertices): per
        shape the principal-axis bounding box a1 >= a2 >= a3, capsules of radius a3/2 along a1, side by side across a2.
        trim: a fingertip sphere striker on the same body - capsule ends are pulled back clear of it (no double contact
        at the tip). Registered top-only on bars (side contacts with bars stay MuJoCo's: grasping)."""
        from scipy.spatial.transform import Rotation
        b, G = self.b, self.newton.GeoType
        k = 0
        for i in shapes:
            if not (int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            t = b.shape_type[i]
            tf = np.asarray(b.shape_transform[i], float)
            if t == G.BOX:
                h = np.asarray(b.shape_scale[i], float)
                v = np.array([[sx*h[0], sy*h[1], sz*h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            elif t == G.MESH and b.shape_source[i] is not None:
                v = np.asarray(b.shape_source[i].vertices, float)*np.asarray(b.shape_scale[i], float)
            else:
                continue
            v = Rotation.from_quat(tf[3:]).apply(v) + tf[:3]
            c0 = v.mean(0)
            _, vec = np.linalg.eigh(np.cov((v - c0).T))
            loc = (v - c0) @ vec
            lo, hi = loc.min(0), loc.max(0)
            ext = hi - lo
            order = np.argsort(-ext)
            a1, a2, a3 = ext[order]
            ctr = c0 + vec @ (0.5*(lo + hi))
            u1, u2 = vec[:, order[0]], vec[:, order[1]]
            r = max(0.5*a3, 0.003)
            n = int(min(max_caps, max(1, round((a2 - 2*r)/(1.5*r)) + 1)))
            offs = [0.0] if n == 1 else list(np.linspace(-(a2/2 - r), a2/2 - r, n))
            half = max(a1/2 - r, 0.0)
            for o in offs:
                e0, e1 = ctr + o*u2 - half*u1, ctr + o*u2 + half*u1
                if trim is not None:
                    pt, rt = np.asarray(trim["p"], float), float(trim["r"])
                    L = float(np.linalg.norm(e1 - e0))
                    if L > 1e-9:
                        d = (e1 - e0)/L
                        s = float((pt - e0) @ d)
                        if s > L/2:
                            e1 = e0 + d*max(0.0, min(L, s - (rt + r)))
                        else:
                            e0 = e0 + d*min(L, max(0.0, s + (rt + r)))
                    if np.linalg.norm(0.5*(e0 + e1) - pt) < rt + r:
                        continue
                entry = dict(name=f"{name}_proxy{k}", body=body, r=float(r), material=material, tool=None, arm=arm,
                             top_only=True, proxy=True)
                if np.linalg.norm(e1 - e0) < 1e-4:
                    entry.update(shape="sphere", p=0.5*(e0 + e1))
                else:
                    entry.update(shape="capsule", p0=e0, p1=e1)
                self.strikers.append(entry)
                k += 1

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on link7 (barcode_scanning / wine_pouring mount geometry)."""
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

    # ------------------------------------------------------------------ xylophone
    def xylophone(self, xl):
        """Orff alto/soprano xylophone of Rucz's measured type: 13 rosewood bars on felt pads, two pins through holes at
        each bar's mode-1 nodes, six box resonators inside a 4.4 kg frame (Studio 49 AX 1000 class box)."""
        from . import modes
        cx, cy = xl["center"]
        yaw = xl["yaw"]
        q = quat_axis_angle((0, 0, 1), yaw)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        X = P.XYLO_BOX
        z0 = P.TABLE["top_z"]
        H = X["height"]
        foot = 0.004
        b = self.b
        frame = b.add_link(xform=self.wp.transform((cx, cy, z0 + foot), q), label="xylophone_frame", mass=float(X["mass"]),
                           com=self.wp.vec3(0., 0., H/2),
                           inertia=self.wp.mat33(*np.diag([X["mass"]/12*(0.29**2 + H**2), X["mass"]/12*(0.61**2 + H**2),
                                                           X["mass"]/12*(0.61**2 + 0.29**2)]).ravel().tolist()),
                           lock_inertia=True)
        self.bodies["xylophone_frame"] = frame
        joints = [b.add_joint_free(child=frame, label="xylophone_frame_free_joint")]
        # hull: the trapezoid as three boxes stepping from the low (wide) end to the high end
        L = X["length"]
        for k in range(3):
            w = X["width_low"] + (X["width_high"] - X["width_low"])*(k + 0.5)/3
            self.box(f"xylo_hull_{k}", (-L/2 + L*(k + 0.5)/3, 0., H/2), (L/6, w/2, H/2), "collision_hidden", body=frame, mu=0.4)
        for sx in (-L/2 + 0.04, L/2 - 0.04):
            for sy in (-0.1, 0.1):
                self.box(f"xylo_foot_{sx:+.2f}_{sy:+.1f}", (sx, sy*(0.8 if sx < 0 else 0.55), -foot/2),
                         (0.015, 0.015, foot/2), "rubber", body=frame, mu=0.9)
        top = H                                          # top plate of the resonator boxes (frame z)
        self.items["xylophone_frame"] = dict(kind="xylophone_frame", body=frame, top=top)
        notes = xl["notes"]
        mid = (len(notes) - 1)/2
        self.xylo_bars = []
        bar_rot = quat_mul(q, quat_axis_angle((0, 0, 1), math.pi/2))       # bar x -> world y (before yaw)
        geo = []
        for i, note in enumerate(notes):
            bd = modes.bar(note)
            xb = (i - mid)*P.BAR_PITCH_M                                   # position along the instrument (frame x)
            nodes = sorted(float(v) for v in bd["nodes"])
            # felt pads under the nodes and the pins through the holes, on a support body welded to the frame at the
            # bar's place (MuJoCo then tests each bar against its own supports only)
            sup = self._support(f"support_{note}", frame, (xb, 0., top), joints)
            for k, xn in enumerate(nodes):
                self.box(f"felt_{note}_{k}", (0., xn, P.FELT_THICK_M/2), (0.018, P.FELT_STRIP_W_M/2, P.FELT_THICK_M/2),
                         "felt", body=sup, mu=0.6)
                self.capsule(f"pin_{note}_{k}", sup, (0., xn, 0.), (0., xn, P.FELT_THICK_M + P.BAR_THICK_M + 0.002),
                             P.PIN_R_M, "pin_metal", mu=0.3)
            geo.append((note, bd, xb, nodes))
        b.add_articulation(joints, label="xylophone_articulation")          # frame + welded supports (contiguous joints)
        for note, bd, xb, nodes in geo:
            centre_local = np.array([xb, 0., top + P.FELT_THICK_M + P.BAR_THICK_M/2])
            p = np.array([cx, cy, z0 + foot]) + Rz @ centre_local + np.array([0, 0, 0.0004])
            body = self._bar_body(f"bar_{note}", note, bd, p, bar_rot, nodes)
            self.xylo_bars.append(dict(note=note, body=body, frame_x=xb, nodes=nodes))
            self.surfaces.append(dict(name=f"bar_{note}", body=body, kind="bar", struct="xylophone", note=note,
                                      half=np.array([bd["length"]/2, P.BAR_WIDTH_M/2, P.BAR_THICK_M/2]), material="rosewood"))
        # resonator openings (radiation points): centred under each group of bars, in the frame top plate
        self.resonator_openings = []
        for r_i, covered in enumerate(P.RESONATOR_BARS):
            xs = [(notes.index(n) - mid)*P.BAR_PITCH_M for n in covered if n in notes]
            self.resonator_openings.append(np.array([float(np.mean(xs)), 0., top]))
        self.items["xylophone"] = dict(kind="xylophone", body=frame, notes=list(notes), center=(cx, cy), yaw=yaw)

    def _support(self, name, parent, pos_local, joints):
        """A massless support link welded to `parent` at pos_local (felt pads, pins)."""
        b = self.b
        body = b.add_link(xform=self.wp.transform((0., 0., 0.), Q_ID), label=name, mass=1e-3,
                          com=self.wp.vec3(0., 0., 0.), inertia=self.wp.mat33(*np.diag([1e-7, 1e-7, 1e-7]).ravel().tolist()),
                          lock_inertia=True)
        joints.append(b.add_joint_fixed(parent, body, parent_xform=self.wp.transform(tuple(map(float, pos_local)), Q_ID),
                                        child_xform=self.wp.transform((0., 0., 0.), Q_ID), label=f"{name}_weld"))
        self.bodies[name] = body
        return body

    def _bar_body(self, name, note, bd, p, rot, nodes):
        """A bar as a free rigid body: collision boxes leaving the node holes free (pins pass through), the top face
        recessed BAR_TOP_RECESS_M (vibro-contact surface above it), four felt-contact spheres under the nodes."""
        Lb = float(bd["length"])
        b, h = P.BAR_WIDTH_M, P.BAR_THICK_M
        hole = P.BAR_HOLE_R_M
        body = self.free_body(name, p, rot, bd["mass"], bd["com"], bd["inertia"])
        zt, zb = h/2 - BAR_TOP_RECESS_M, -h/2 + BAR_BOTTOM_LIFT_M
        zc, hz = (zt + zb)/2, (zt - zb)/2
        cuts = [-Lb/2] + sum([[xn - hole, xn + hole] for xn in nodes], []) + [Lb/2]
        for k in range(0, len(cuts), 2):
            x0, x1 = cuts[k], cuts[k + 1]
            self.box(f"{name}_seg{k//2}", ((x0 + x1)/2, 0., zc), ((x1 - x0)/2, b/2, hz), "collision_hidden", body=body,
                     mu=0.45)
        for j, xn in enumerate(nodes):
            for s in (-1, 1):
                self.box(f"{name}_holewall{j}{'ab'[s > 0]}", (xn, s*(hole + b/2)/2, zc), (hole, (b/2 - hole)/2, hz),
                         "collision_hidden", body=body, mu=0.45)
            for s in (-1, 1):
                self.sphere(f"{name}_foot{j}{'ab'[s > 0]}", body, (xn, s*b/3, -h/2 + 0.0015), 0.0015, "collision_hidden",
                            mu=0.6)
        self.items[name] = dict(kind="bar", note=note, body=body, length=Lb)
        return body

    def resonator_bar(self, rb):
        """A Studio 49 KB/AX-class resonator bar: a rosewood bar on felt and pins over its own maple box tuned to it
        (modal.resonator_bar_box: the datasheet's outer size, mass = the listed 0.40 kg minus the bar)."""
        from . import modes
        from .modal import resonator_bar_box
        note = rb["note"]
        x, y = rb["xy"]
        B = resonator_bar_box(note)
        z0 = P.TABLE["top_z"]
        b = self.b
        box = b.add_link(xform=self.wp.transform((x, y, z0 + 0.003), Q_ID), label=f"resbox_{note}", mass=float(B["mass"]),
                         com=self.wp.vec3(0., 0., B["height"]/2),
                         inertia=self.wp.mat33(*np.diag([B["mass"]/12*(B["length"]**2 + B["height"]**2),
                                                         B["mass"]/12*(B["width"]**2 + B["height"]**2),
                                                         B["mass"]/12*(B["length"]**2 + B["width"]**2)]).ravel().tolist()),
                         lock_inertia=True)
        self.bodies[f"resbox_{note}"] = box
        joints = [b.add_joint_free(child=box, label=f"resbox_{note}_free_joint")]
        self.box(f"resbox_{note}_hull", (0., 0., B["height"]/2), (B["width"]/2, B["length"]/2, B["height"]/2), "maple",
                 body=box, mu=0.4)
        for sy in (-1, 1):
            for sx in (-1, 1):
                self.box(f"resbox_{note}_foot{sx}{sy}", (sx*(B["width"]/2 - 0.008), sy*(B["length"]/2 - 0.02), -0.0015),
                         (0.006, 0.008, 0.0015), "rubber", body=box, mu=0.9)
        bd = modes.bar(note)
        nodes = sorted(float(v) for v in bd["nodes"])
        top = B["height"]
        sup = self._support(f"rsupport_{note}", box, (0., 0., top), joints)
        b.add_articulation(joints, label=f"resbox_{note}_articulation")
        for k, xn in enumerate(nodes):
            self.box(f"rfelt_{note}_{k}", (0., xn, P.FELT_THICK_M/2), (0.018, P.FELT_STRIP_W_M/2, P.FELT_THICK_M/2),
                     "felt", body=sup, mu=0.6)
            self.capsule(f"rpin_{note}_{k}", sup, (0., xn, 0.), (0., xn, P.FELT_THICK_M + P.BAR_THICK_M + 0.002),
                         P.PIN_R_M, "pin_metal", mu=0.3)
        p = np.array([x, y, z0 + 0.003 + top + P.FELT_THICK_M + P.BAR_THICK_M/2 + 0.0004])
        body = self._bar_body(f"rbar_{note}", note, bd, p, quat_axis_angle((0, 0, 1), math.pi/2), nodes)
        self.items[f"resbox_{note}"] = dict(kind="resonator_box", body=box, note=note)
        self.surfaces.append(dict(name=f"rbar_{note}", body=body, kind="bar", struct=f"resbar_{note}", note=note,
                                  half=np.array([bd["length"]/2, P.BAR_WIDTH_M/2, P.BAR_THICK_M/2]), material="rosewood"))

    # ------------------------------------------------------------------ drum
    def drum(self, dl):
        """14 x 5 in single-headed drum (Remo Ambassador Coated batter, open bottom) clamped in a double-braced snare
        stand: one free body (shell, hoop, stand), plus the head as a welded child body (its collider sits 8 mm below the
        head plane for unregistered bodies; strikers meet the vibrating head through the vibro-contact)."""
        x, y = dl["xy"]
        zh = dl["head_z"]
        yaw = dl["yaw"]
        q = quat_axis_angle((0, 0, 1), yaw)
        a = P.DRUM_RADIUS_M
        depth = P.DRUM_SHELL_DEPTH_M
        m_d, m_s = P.DRUM_MASS_KG, P.STAND_MASS_KG
        # mass properties: shell+hoop as a thin cylinder (head plane origin), stand as a column + tripod below
        com_d, com_s = np.array([0, 0, -depth/2]), np.array([0, 0, -0.45])
        m = m_d + m_s
        com = (m_d*com_d + m_s*com_s)/m
        Id = np.array([m_d*(a**2/2 + depth**2/12), m_d*(a**2/2 + depth**2/12), m_d*a**2])
        Is = np.array([m_s*(0.18**2 + 0.6**2/12), m_s*(0.18**2 + 0.6**2/12), m_s*0.18**2])
        I = Id + m_d*((com_d - com)**2)[[2, 2, 0]]*np.array([1, 1, 0]) + Is + m_s*((com_s - com)**2)[[2, 2, 0]]*np.array([1, 1, 0])
        b = self.b
        body = b.add_link(xform=self.wp.transform((x, y, zh), q), label="drum", mass=float(m),
                          com=self.wp.vec3(*map(float, com)), inertia=self.wp.mat33(*np.diag(I).ravel().tolist()),
                          lock_inertia=True)
        jf = b.add_joint_free(child=body, label="drum_free_joint")
        head = b.add_link(xform=self.wp.transform((x, y, zh), q), label="drumhead", mass=0.05,
                          com=self.wp.vec3(0., 0., 0.), inertia=self.wp.mat33(*np.diag([4.0e-4, 4.0e-4, 7.5e-4]).ravel().tolist()),
                          lock_inertia=True)
        jh = b.add_joint_fixed(body, head, parent_xform=self.wp.transform((0., 0., 0.), Q_ID),
                               child_xform=self.wp.transform((0., 0., 0.), Q_ID), label="drumhead_weld")
        b.add_articulation([jf, jh], label="drum_articulation")
        self.bodies["drum"], self.bodies["drumhead"] = body, head
        # shell (16 staves), hoop (24 segments), bearing edge under the head
        n = 16
        for k in range(n):
            t = 2*math.pi*(k + 0.5)/n
            qs = quat_axis_angle((0, 0, 1), t)
            self.box(f"drum_shell_{k}", (a*math.cos(t), a*math.sin(t), -depth/2 - 0.002), (0.003, a*math.pi/n, depth/2),
                     "collision_hidden", body=body, rot=qs, mu=0.4)
        r0, r1 = P.DRUM_HOOP_R_M
        nh = 24
        for k in range(nh):
            t = 2*math.pi*(k + 0.5)/nh
            qs = quat_axis_angle((0, 0, 1), t)
            rm = 0.5*(r0 + r1)
            self.box(f"drum_hoop_{k}", (rm*math.cos(t), rm*math.sin(t), (P.DRUM_HOOP_H_M - 0.012)/2),
                     ((r1 - r0)/2, rm*math.pi/nh*1.02, (P.DRUM_HOOP_H_M + 0.012)/2), "collision_hidden", body=body, rot=qs,
                     mu=0.3)
        hc = b.add_shape_cylinder(head, xform=self.wp.transform((0., 0., -HEAD_RECESS_M - 0.001), Q_ID), radius=a - 0.004,
                                  half_height=0.001, cfg=self.cfg(mu=0.4), label="drumhead_collider")
        self.materials[hc] = "collision_hidden"
        # snare stand (double braced): basket arms, column, tripod legs, rubber feet
        zf = -zh                                                    # floor in the drum frame
        col_top, col_base = -depth - 0.03, zf + 0.30
        self.capsule("stand_column", body, (0, 0, col_base), (0, 0, col_top), 0.011, "chrome", mu=0.4)
        for k in range(3):
            t = 2*math.pi*k/3
            d = np.array([math.cos(t), math.sin(t), 0.])
            # basket arm out past the shell and a rubber-sleeved hook up its outside (holds the drum on a real stand)
            elbow = (a + 0.009)*d + np.array([0, 0, -depth - 0.016])
            self.capsule(f"stand_arm_{k}", body, (0, 0, col_top), tuple(elbow), 0.006, "chrome", mu=0.4)
            self.capsule(f"stand_hook_{k}", body, tuple(elbow), tuple(elbow + np.array([0, 0, 0.045])), 0.005,
                         "rubber", mu=0.9)
            foot = 0.27*d + np.array([0, 0, zf + 0.015])
            self.capsule(f"stand_leg_{k}", body, (0, 0, col_base), tuple(foot), 0.009, "chrome", mu=0.4)
            self.sphere(f"stand_foot_{k}", body, tuple(foot), 0.015, "rubber", mu=0.9)
        self.items["drum"] = dict(kind="drum", body=body, head=head, tension=dl["tension"], head_z=zh)
        self.surfaces.append(dict(name="drumhead", body=head, kind="head", struct="drum", radius=a, material="mylar"))

    def bar_tray(self, bt):
        """Accessory tray for removed bars (pentatonic set-up): a shallow beech box on the table."""
        x, y = bt["xy"]
        hx, hy, hz = bt["half"]
        z0 = P.TABLE["top_z"]
        tray = self.free_body("bar_tray", (x, y, z0 + 0.001), Q_ID, 0.45, (0, 0, 0.005),
                              (0.45/12*(4*hy**2), 0.45/12*(4*hx**2), 0.45/12*(4*hx**2 + 4*hy**2)))
        self.box("bar_tray_floor", (0, 0, 0.003), (hx, hy, 0.003), "beech", body=tray, mu=0.5)
        for s in (-1, 1):
            self.box(f"bar_tray_wall_x{s}", (s*(hx - 0.004), 0, hz), (0.004, hy, hz - 0.002), "beech", body=tray)
            self.box(f"bar_tray_wall_y{s}", (0, s*(hy - 0.004), hz), (hx, 0.004, hz - 0.002), "beech", body=tray)
        self.items["bar_tray"] = dict(kind="tray", body=tray, half=(hx, hy, hz))

    def speaker(self, sp):
        x, y = sp["xy"]
        w, d, h = sp["size"]
        z0 = P.TABLE["top_z"]
        self.box("speaker", (x, y, z0 + h/2), (w/2, d/2, h/2), "speaker_black", mu=0.5)
        self.fixtures["speaker"] = dict(position=(x, y - d/2, z0 + 0.6*h), size=sp["size"], xy=(x, y))

    # ------------------------------------------------------------------ tools
    def tools(self):
        for k, sup in enumerate(self.layout["tool_supports"]):
            cx, cy = sup["center"]
            hx, hy = sup["half"][:2]
            if sup["kind"] == "stick_tray":
                ht = sup["top_z"] - P.TABLE["top_z"]
                self.box("stick_tray_floor", (cx, cy, P.TABLE["top_z"] + ht/2), (hx, hy, ht/2), "beech", mu=0.5)
                for s_ in (-1, 1):
                    self.box(f"stick_tray_rim_y{s_}", (cx, cy + s_*(hy - 0.004), sup["top_z"] + 0.004), (hx, 0.004, 0.004),
                             "beech", mu=0.5)
                    self.box(f"stick_tray_rim_x{s_}", (cx + s_*(hx - 0.004), cy, sup["top_z"] + 0.004), (0.004, hy, 0.004),
                             "beech", mu=0.5)
            else:
                top = sup["top_z"]
                self.box(f"trap_table_{k}_top", (cx, cy, top - 0.012), (hx, hy, 0.012), "trap_felt", mu=0.6)
                self.box(f"trap_table_{k}_post", (cx, cy, (top - 0.024)/2), (0.012, 0.012, (top - 0.024)/2), "cart_frame")
                self.box(f"trap_table_{k}_base", (cx, cy, 0.01), (0.11, 0.11, 0.01), "cart_frame")   # clear of the drum stand's feet
        for t in self.layout["tools"]:
            getattr(self, "_" + t["kind"])(t)

    def _place(self, t, z_axis_r):
        d = np.asarray(t["dir"], float)
        yaw = math.atan2(d[1], d[0])
        return np.asarray(t["butt"], float) + np.array([0, 0, z_axis_r]), quat_axis_angle((0, 0, 1), yaw), yaw

    def _saddles(self, name, t, positions, r):
        """Two cradles under a tool (static, on the rack plate): a rounded rest whose top touches the tool's underside
        (one contact point; a flat block gave MuJoCo's multi-CCD four line contacts per saddle) and two cheeks 1 mm
        clear of the tool on each side."""
        d = np.asarray(t["dir"], float)
        butt = np.asarray(t["butt"], float)
        zr = butt[2] - LY.SADDLE_AXIS                   # the support's top surface
        rr = 0.008
        for k, s in enumerate(positions):
            c = butt + s*d
            ztool = butt[2] - zr
            self.box(f"saddle_{name}_{k}_post", (c[0], c[1], zr + (ztool - r - rr)/2), (0.008, 0.008, (ztool - r - rr)/2),
                     "beech", mu=0.5)
            i = self.b.add_shape_sphere(-1, xform=self.wp.transform((float(c[0]), float(c[1]), float(zr + ztool - r - rr)), Q_ID),
                                        radius=rr, cfg=self.cfg(mu=0.5), label=f"saddle_{name}_{k}_rest")
            self.materials[i] = "beech"
            for side in (-1, 1):
                self.box(f"saddle_{name}_{k}_cheek{side}", (c[0], c[1] + side*(r + 0.001 + 0.004), zr + ztool/2 + 0.002),
                         (0.010, 0.004, ztool/2 + 0.002), "beech", mu=0.5)

    def _stick(self, t):
        from . import modes
        st = modes.stick()
        S = P.STICK
        r = S["D"]/2
        p, q, yaw = self._place(t, 0.0)
        body = self.free_body(t["name"], p, q, st["mass"], (st["com_x"], 0., 0.),
                              (st["I_axial"], st["I_transverse"], st["I_transverse"]))
        L = S["L"]
        xt0, xn = L - S["taper"], L - S["tip"]
        xm = xt0 + 0.5*S["taper"]
        rm = 0.5*(S["D"] + S["D"] + (S["dn"] - S["D"])*0.5)/2           # mean radius of the taper's first half
        rt = 0.5*(S["D"] + (S["dn"] - S["D"])*0.5 + S["dn"])/2           # ... and of its second half
        # shaft, two taper segments (each capsule's radius is its segment's mean, so neither sticks out below the tip
        # sphere at playing angles) and the tip; the same geometry is MuJoCo's collider and the vibro-contact striker
        segs = [("shaft", (r, 0, 0), (xt0, 0, 0), r), ("taper1", (xt0, 0, 0), (xm, 0, 0), rm),
                ("taper2", (xm, 0, 0), (xn - 0.003, 0, 0), rt)]
        for nm, a_, b_, rr in segs:
            self.capsule(f"{t['name']}_{nm}", body, a_, b_, rr, "collision_hidden", mu=0.5)
        self.sphere(f"{t['name']}_tip", body, (st["tip_x"], 0, 0), st["tip_r"], "collision_hidden", mu=0.5)
        self.items[t["name"]] = dict(kind="stick", body=body, length=L, fulcrum_x=float(st["fulcrum_x"]),
                                     grip_x=float(st["fulcrum_x"]))
        self.strikers.append(dict(name=f"{t['name']}_tip", body=body, shape="sphere", p=np.array([st["tip_x"], 0, 0]),
                                  r=float(st["tip_r"]), material="hickory_tip", tool=t["name"], stick_x=float(st["tip_x"])))
        for nm, a_, b_, rr in segs:
            self.strikers.append(dict(name=f"{t['name']}_{nm}", body=body, shape="capsule", p0=np.array(a_, float),
                                      p1=np.array(b_, float), r=float(rr), material="hickory_shaft", tool=t["name"]))
        self._saddles(t["name"], t, (0.06, 0.25), r)

    def _mallet(self, t, spec=None, kind="mallet", head_mat="rubber_mallet"):
        M = spec or P.MALLET
        L, rh, rr = M["L"], M["handle_r"], M["head_r"]
        Lh = L - 2*rr
        m_h = M["handle_rho"]*math.pi*rh**2*Lh
        m_k = M["head_mass"]
        m = m_h + m_k
        xh, xk = Lh/2, L - rr
        xc = (m_h*xh + m_k*xk)/m
        It = m_h*Lh**2/12 + m_h*(xh - xc)**2 + 0.4*m_k*rr**2 + m_k*(xk - xc)**2
        Ia = 0.5*m_h*rh**2 + 0.4*m_k*rr**2
        p, q, yaw = self._place(t, 0.0)
        body = self.free_body(t["name"], p, q, m, (xc, 0, 0), (Ia, It, It))
        self.capsule(f"{t['name']}_handle", body, (rh, 0, 0), (Lh + 0.002, 0, 0), rh, "collision_hidden", mu=0.5)
        self.sphere(f"{t['name']}_head", body, (xk, 0, 0), rr, "collision_hidden", mu=0.7)
        self.items[t["name"]] = dict(kind=kind, body=body, length=L, mass=m, grip_x=0.10 if kind == "mallet" else 0.10)
        self.strikers.append(dict(name=f"{t['name']}_head", body=body, shape="sphere", p=np.array([xk, 0, 0]), r=rr,
                                  material=head_mat, tool=t["name"]))
        self.strikers.append(dict(name=f"{t['name']}_handle", body=body, shape="capsule", p0=np.array([rh, 0, 0]),
                                  p1=np.array([Lh, 0, 0]), r=rh, material="hickory_shaft", tool=t["name"]))
        self._saddles(t["name"], t, (0.05, 0.215 if kind == "mallet" else 0.22), rh)

    def _beater(self, t):
        self._mallet(t, P.BEATER, kind="beater", head_mat="felt")

    # ------------------------------------------------------------------ cameras
    def views(self):
        """Policy camera on the cart mast (D435-class 69 deg HFOV) looking over the table; diagnostic views render the
        same state."""
        mx, my = self.fixtures["mast"]["x"], self.fixtures["mast"]["y"]
        self.cameras["station_rgb"] = dict(body=-1, eye=[mx, my + 0.20, 1.55], target=[0.05, 0.62, 0.74], up=[0, 0, 1],
                                           fov=69., role="diagnostic" if self.g1 else "policy")
        self.cameras["room"] = dict(body=-1, eye=[2.10, 2.55, 1.85], target=[0.10, 0.30, 0.85], up=[0, 0, 1], fov=55.,
                                    role="diagnostic")
        self.cameras["client"] = dict(body=-1, eye=[0.0, 1.85, 1.12], target=[0.05, 0.3, 0.85], up=[0, 0, 1], fov=55.,
                                      role="diagnostic")
        self.cameras["side"] = dict(body=-1, eye=[1.55, 0.55, 1.05], target=[0.15, 0.50, 0.82], up=[0, 0, 1], fov=48.,
                                    role="diagnostic")
        if self.layout["drum"]:
            dx, dy = self.layout["drum"]["xy"]
            zh = self.layout["drum"]["head_z"]
            self.cameras["drum_close"] = dict(body=-1, eye=[dx + 0.55, dy + 0.55, zh + 0.30], target=[dx, dy, zh + 0.02],
                                              up=[0, 0, 1], fov=42., role="diagnostic")
        if self.layout["xylophone"]:
            cx, cy = self.layout["xylophone"]["center"]
            self.cameras["xylo_close"] = dict(body=-1, eye=[cx + 0.05, cy + 0.62, 1.05], target=[cx, cy, 0.83], up=[0, 0, 1],
                                              fov=45., role="diagnostic")
        if self.layout["resonator_bars"]:
            # the resonator bars seen from the client's side (strokes on them are heard, so they are shown too)
            xs = [rb["xy"][0] for rb in self.layout["resonator_bars"]]
            ys = [rb["xy"][1] for rb in self.layout["resonator_bars"]]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            self.cameras["resbar_close"] = dict(body=-1, eye=[cx + 0.10, cy + 0.55, 1.12], target=[cx, cy, 0.84],
                                                up=[0, 0, 1], fov=40., role="diagnostic")
        self.cameras["rack_close"] = dict(body=-1, eye=[0.50, 0.10, 1.12], target=[0.0, 0.47, 0.75], up=[0, 0, 1], fov=45.,
                                          role="diagnostic")
        self.cameras["shelf"] = dict(body=-1, eye=[-0.9, 0.7, 1.45], target=[-2.7, 0.85, 1.15], up=[0, 0, 1], fov=62.,
                                     role="diagnostic")
        self.cameras["floor"] = dict(body=-1, eye=[2.4, 0.9, 1.35], target=[1.0, -1.05, 0.25], up=[0, 0, 1], fov=60.,
                                     role="diagnostic")             # the floor right of the station and the back wall
        if self.layout["drum"]:
            dx, dy = self.layout["drum"]["xy"]
            self.cameras["drum_stand"] = dict(body=-1, eye=[dx + 0.75, dy - 0.45, 0.62], target=[dx, dy, 0.55],
                                              up=[0, 0, 1], fov=50., role="diagnostic")
        if "probe_camera" in self.layout:
            pc = self.layout["probe_camera"]
            self.cameras["probe"] = dict(body=-1, eye=list(pc["eye"]), target=list(pc["target"]), up=[0, 0, 1],
                                         fov=float(pc.get("fov", 45.)), role="diagnostic")

    # ------------------------------------------------------------------ MuJoCo exclusions
    def apply_exclusions(self):
        """Filter every shape pair between each striker tool body and each vibrating-surface body: Newton emits a
        MuJoCo body <exclude> when all of a body pair's shape pairs are filtered (world.py checks the compiled model)."""
        b = self.b
        tool_bodies = sorted({s["body"] for s in self.strikers if s["tool"]})
        surf_bodies = sorted({s["body"] for s in self.surfaces})
        shapes_of = {}
        for i in range(b.shape_count):
            shapes_of.setdefault(b.shape_body[i], []).append(i)
        for tb in tool_bodies:
            for sb in surf_bodies:
                for i in shapes_of.get(tb, []):
                    for j in shapes_of.get(sb, []):
                        b.add_shape_collision_filter_pair(i, j)
                self.exclude_bodies.append((tb, sb))

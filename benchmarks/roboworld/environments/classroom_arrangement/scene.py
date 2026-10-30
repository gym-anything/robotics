"""Physics scene (Newton 1.5 ModelBuilder): room 114, its furniture as free rigid bodies, and the robot.

Every furniture part is a collision primitive whose density is its catalogue mass share / its volume, so each body's
mass, centre of mass and inertia come from where its steel, plastic and wood are (furniture.py). Glides carry their
own friction (MuJoCo geom priority 1, world.py). Installed parts (walls, board, bookcase, cubbies, door) are static.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import params as P
from . import furniture as F
from . import layout as L

BUILDERS = dict(desk=F.desk, chair=F.chair, table=F.table, teacher_desk=F.teacher_desk, tote=F.tote, basket=F.basket)
CHAIR_UP_DY = 0.075            # inverted chair on its desk: seat overhangs the student edge by 40 mm (furniture tubes)
ROBOT_BASE_LINKS = ("base_link", "caster_", "argo_drive_", "rocker_arm_link")


def volume(p):
    if p.kind == "box":
        return 8*p.size[0]*p.size[1]*p.size[2]
    r, h = p.size
    if p.kind == "capsule":
        return math.pi*r*r*2*h + 4/3*math.pi*r**3
    return math.pi*r*r*2*h


def chair_up_pose(desk_pose):
    x, y, yaw = desk_pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = (x - s*CHAIR_UP_DY, y + c*CHAIR_UP_DY, P.DESK_HEIGHT + P.CHAIR_SEAT_H)
    q = (Rotation.from_euler("z", yaw)*Rotation.from_euler("y", math.pi)).as_quat()
    return p, tuple(q)


class Scene:
    ENV_ID = "classroom-arrangement"          # robots.catalog entry the robot setup is resolved against

    def __init__(self, config, newton, wp, spec):
        """spec: layout.setup(...) result."""
        self.c, self.newton, self.wp, self.spec = config, newton, wp, spec
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.parts = {}, {}
        self.bodies, self.items = {}, {}
        self.glide_shapes, self.wheel_shapes, self.pad_shapes = [], [], {}
        self.cameras = {}
        self.room()
        self.robot()
        self.furniture()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=0.5, ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_M, margin=0., mu_torsional=0., mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def add_part(self, body, p, prefix):
        wp = self.wp
        tf = wp.transform(p.pos, wp.quat(*p.quat))
        if not p.collide:
            return None
        dens = p.mass/volume(p) if body >= 0 else 0.
        kw = dict(mu=p.mu if p.mu is not None else 0.3, density=dens)
        label = f"{prefix}/{p.name}"
        if p.kind == "box":
            i = self.b.add_shape_box(body, xform=tf, hx=p.size[0], hy=p.size[1], hz=p.size[2], cfg=self.cfg(**kw), label=label)
        elif p.kind == "capsule":
            i = self.b.add_shape_capsule(body, xform=tf, radius=p.size[0], half_height=p.size[1], cfg=self.cfg(**kw), label=label)
        elif p.kind == "cylinder":
            i = self.b.add_shape_cylinder(body, xform=tf, radius=p.size[0], half_height=p.size[1], cfg=self.cfg(**kw), label=label)
        else:
            raise ValueError(p.kind)
        self.materials[i] = p.material
        if p.role == "glide":
            self.glide_shapes.append(i)
        return i

    # ------------------------------------------------------------------ room
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=P.MU_FLOOR), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        for name, c, h, mat in L.STATIC:
            i = b.add_shape_box(-1, xform=self.wp.transform(c, self.wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                                cfg=self.cfg(mu=0.4, density=0.), label=name)
            self.materials[i] = mat

    # ------------------------------------------------------------------ furniture
    def furniture(self):
        wp = self.wp
        sp = self.spec
        for name, it in sp["items"].items():
            kind = it["kind"]
            parts = BUILDERS[kind](sp["glides"]) if kind in ("desk", "chair", "table", "teacher_desk") else BUILDERS[kind]()
            x, y, yaw = it["pose"]
            if it.get("inverted_on"):
                pos, q = chair_up_pose(sp["items"][it["inverted_on"]]["pose"])
            else:
                pos, q = (x, y, 0.0005), L.yaw_quat(yaw)
            body = self.b.add_body(xform=wp.transform(pos, wp.quat(*q)), label=name)
            self.bodies[name] = body
            shapes = [self.add_part(body, p, name) for p in parts]
            self.parts[name] = parts
            self.items[name] = dict(kind=kind, body=body, shapes=[s for s in shapes if s is not None],
                                    glides=[s for s, p in zip(shapes, parts) if p.role == "glide"])
        for d, n in sp["books"].items():
            dx, dy, _ = sp["items"][d]["pose"]
            yaw = sp["items"][d]["pose"][2]
            bx, by, bt, _ = P.BOOK
            floor = P.DESK_HEIGHT - P.DESK_TOP[2] - P.DESK_BOX[2] + 0.003
            for k in range(n):
                name = f"book_{d[5:]}_{k}"
                # flat, long side front-to-back: two side by side (2 x 205 mm in the 549 mm box) and the third on
                # top of the first, as books lie in a book box (before: three side by side overlapped and leaned)
                local = ((-0.11, 0.11)[k % 2], -0.02, floor + bt/2 + 0.0005 + (k//2)*(bt + 0.001))
                cz, sz = math.cos(yaw), math.sin(yaw)
                pos = (dx + cz*local[0] - sz*local[1], dy + sz*local[0] + cz*local[1], local[2])
                q = (Rotation.from_euler("z", yaw)*Rotation.from_euler("z", math.pi/2)).as_quat()
                body = self.b.add_body(xform=wp.transform(pos, wp.quat(*q)), label=name)
                self.bodies[name] = body
                parts = F.book(k)
                parts[0].pos = (0., 0., 0.)
                shapes = [self.add_part(body, p, name) for p in parts]
                self.parts[name] = parts
                self.items[name] = dict(kind="book", body=body, shapes=shapes, glides=[], desk=d)

    # ------------------------------------------------------------------ robot
    def robot(self):
        from robots.catalog import for_environment, model
        from robots.backends.newton import mount
        import robots
        setup = for_environment(self.ENV_ID, self.c.robot)
        self.robot_setup = setup
        x, y, yaw = self.spec["robot"]
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts={"floor_start": mount((x, y, 0.0015), yaw)})
        self.robot_bodies = list(range(first, self.b.body_count))
        self.robot_shapes = list(range(first_shape, self.b.shape_count))
        (inst, handle), = self.rig_pending.instances.items()
        self.robot_instance, self.handle = inst, handle
        spec = model(setup.to_dict()["instances"][inst]["model"]).to_dict()
        self.arms = tuple(g[:-4] for g in spec["groups"] if g.endswith("_arm"))
        self.groups = {g: [f"{inst}/{j}" for j in v["joints"]] for g, v in spec["groups"].items()}
        self.joints = dict(handle.joints)
        b, G = self.b, self.newton.GeoType
        lab = b.body_label
        self.robot_body = {lab[i].split("/")[-1]: i for i in self.robot_bodies}
        base_group = [i for i in self.robot_bodies if lab[i].split("/")[-1].startswith(ROBOT_BASE_LINKS)]
        for i in self.robot_shapes:
            box = b.shape_type[i] in (G.BOX, G.PLANE)
            b.shape_gap[i] = P.CONTACT_GAP_M if box else P.MESH_CONTACT_GAP_M
            if int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES):
                self.materials[i] = "robot"
                b.shape_material_mu[i] = P.MU_ROBOT
                body = b.shape_body[i]
                name = lab[body].split("/")[-1]
                if name in ("argo_drive_front_link", "argo_drive_rear_link", "caster_front_left_link", "caster_rear_right_link"):
                    self.wheel_shapes.append(i)
                    b.shape_material_mu[i] = P.MU_WHEEL
            else:
                self.materials[i] = "robot_visual"
        # the base's own parts (chassis hull, drive units, rocker, casters, wheels) never touch each other: the chassis
        # hull encloses the wheel wells, and a wheel is the chassis's grandchild (MuJoCo excludes only parent-child)
        coll = [i for i in self.robot_shapes if int(b.shape_flags[i]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES)]
        bg = [i for i in coll if b.shape_body[i] in base_group]
        for k, s0 in enumerate(bg):
            for s1 in bg[k + 1:]:
                if b.shape_body[s0] != b.shape_body[s1]:
                    b.add_shape_collision_filter_pair(s0, s1)
        self._fingertips()
        for arm in self.arms:
            self._wrist_camera(arm)
        self._base_cameras()
        self._arm_gravcomp(lab)

    def _base_cameras(self):
        """The TMR's four RealSense D455 (front, rear, left, right; manual R04245 5.6) at the URDF's *_mounting_point
        frames on base_link, optical axes level and outward."""
        body = self.robot_body["base_link"]
        for name, (x, y, z, ax, ay) in dict(front_rgb=(0.380705, 0., 0.2345, 1., 0.), rear_rgb=(-0.380705, 0., 0.2345, -1., 0.),
                                            left_rgb=(0., 0.272712, 0.1145, 0., 1.),
                                            right_rgb=(0., -0.272712, 0.1145, 0., -1.)).items():
            self.cameras[name] = dict(body=body, eye=[x + 0.02*ax, y + 0.02*ay, z], target=[x + ax, y + ay, z], up=[0, 0, 1],
                                      fov=P.BASE_CAMERA_HFOV, role="policy")

    def _arm_gravcomp(self, lab):
        """The FR3 controller compensates its own links' gravity (and Coriolis terms): world.py adds MuJoCo's bias force
        on the FR3 joints (qfrc_bias, the arm subtree's gravity and velocity terms) to the servo, inside the joints'
        87 / 12 N m actuator range. Not MuJoCo's body gravcomp: that pushes each compensated body up with its own
        weight through every ancestor joint, so the base would stop carrying the arms (wheel loads fall by 320 N = the
        arms' 32.6 kg), and written after compilation it has no effect at all. The TMR, the
        Spine and every payload weigh, and the base carries the arms."""
        self.gravcomp_bodies = [i for i in self.robot_bodies if "_fr3_" in lab[i].split("/")[-1]] if P.ARM_GRAVCOMP else []

    def _fingertips(self):
        """The Franka Hand finger's rubber tip box (17.5 x 15.2 x 18.5 mm at z = 45.25 mm in the finger frame,
        franka_description fr3 hand) is the pad: its contacts use the pad's friction and compliance (world.py)."""
        b = self.b
        for i in self.robot_shapes:
            body = b.shape_body[i]
            if body < 0 or "finger" not in b.body_label[body] or b.shape_type[i] != self.newton.GeoType.BOX:
                continue
            if abs(float(np.asarray(b.shape_transform[i])[2]) - 0.04525) < 1e-4:
                self.pad_shapes[i] = b.body_label[body]
                b.shape_material_mu[i] = P.MU_PAD
        if len(self.pad_shapes) != 2*len(self.arms):
            raise RuntimeError(f"expected {2*len(self.arms)} fingertip pads, found {len(self.pad_shapes)}")

    def _wrist_camera(self, arm):
        """D405-envelope wrist RGB on link7 (same mount geometry as barcode/label_application)."""
        body = self.robot_body[f"{arm}_fr3_link7"]
        wp = self.wp
        for name, c, h, m, mat in ((f"{arm}_camera_case", (.075, 0, .07), (.021, .021, .0115), .060, "camera_dark"),
                                   (f"{arm}_camera_bracket", (.05, 0, .046), (.025, .013, .003), .010, "aluminum")):
            i = self.b.add_shape_box(body, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                                     cfg=self.cfg(mu=P.MU_ROBOT, density=m/(8*h[0]*h[1]*h[2]), gap=0.), label=name)
            self.materials[i] = mat
            self.robot_shapes.append(i)
        self.cameras[f"{arm}_wrist_rgb"] = dict(body=body, eye=[.075, -.012, .083], target=[.075, -.012, .283],
                                                up=[0, 1, 0], fov=P.WRIST_HFOV, role="policy")

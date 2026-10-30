"""Physics scene (Newton 1.5 ModelBuilder): room, bar, robot pedestal, robots, and the task's glassware.

World frame: z up; the bar runs along x; guests stand at +y beyond the bar's far edge, the robots work from the
pedestal at -y. Bar top: quartz, 3.00 x 0.66 x 0.03 m, finished height 1.09 m ([S9, S10]); the robot
side and the guest side are the same counter (Makr Shakr Toni / Richtech ADAM style robotic bars, [S1-S3]).
Glassware geometry comes from geometry.py (the same polygons the liquid uses); rigid collision is its convex pieces.
"""

import math
import numpy as np

from . import params as P
from . import geometry as GEO
from .layout import task_layout

BAR_Y0, BAR_Y1 = -0.08, -0.08 + P.BAR_DEPTH_M          # robot-side and guest-side edges of the bar top
BAR_TOP = P.BAR_TOP_Z_M
PEDESTAL_TOP = P.ARM_BASE_Z_M


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    s = math.sin(angle/2)
    return (float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(math.cos(angle/2)))


class Scene:
    FRANKA_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                           ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                           ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                           ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual = {}, set()
        self.bodies, self.items, self.cameras = {}, {}, {}
        self.static_boxes = []           # (name, centre, half, material) of static solids (render + liquid guard)
        self.layout = task_layout(config)
        self.room()
        self.robots()
        self.glassware()
        self.views()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=P.MU["default"], ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_M, margin=0., mu_torsional=0.,
                 mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.newton.ModelBuilder.ShapeConfig(
            density=0., has_shape_collision=False, has_particle_collision=False)
        i = self.b.add_shape_box(body, xform=self.wp.transform(pos, rot or self.wp.quat_identity()),
                                 hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        if body < 0:
            self.static_boxes.append((name, tuple(map(float, pos)), tuple(map(float, half)), material))
        return i

    def set_mass(self, body, mass, com=None, inertia=None):
        self.b.body_mass[body] = float(mass)
        if com is not None:
            self.b.body_com[body] = self.wp.vec3(*map(float, com))
        if inertia is not None:
            self.b.body_inertia[body] = self.wp.mat33(*np.diag(inertia).ravel().tolist())
            self.b.body_inv_inertia[body] = self.wp.mat33(*np.diag(1.0/np.asarray(inertia)).ravel().tolist())
        self.b.body_inv_mass[body] = 1.0/float(mass)

    # ------------------------------------------------------------------ room, bar, pedestal
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=.6), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        cx = 0.
        # bar top (quartz) on a front die wall (guest side) and back supports; the top is the only collision surface
        # the robots and glassware use; the die and footrail are dressing with collision for completeness
        self.box("bar_top", (cx, 0.5*(BAR_Y0 + BAR_Y1), BAR_TOP - 0.5*P.BAR_TOP_THICK_M),
                 (0.5*P.BAR_LENGTH_M, 0.5*P.BAR_DEPTH_M, 0.5*P.BAR_TOP_THICK_M), "quartz", mu=P.MU["glass_quartz"])
        die_h = BAR_TOP - P.BAR_TOP_THICK_M
        self.box("bar_die", (cx, BAR_Y1 - 0.12, 0.5*die_h), (0.5*P.BAR_LENGTH_M - 0.02, 0.05, 0.5*die_h), "bar_wood")
        self.box("bar_back_panel", (cx, BAR_Y0 + 0.06, 0.5*die_h), (0.5*P.BAR_LENGTH_M - 0.02, 0.02, 0.5*die_h),
                 "bar_wood_dark")
        # FR3 pedestal behind the bar (steel column + plate per arm)
        for name, (x, y) in P.ARM_BASE_XY.items():
            self.box(f"{name}_pedestal", (x, y, 0.5*(PEDESTAL_TOP - 0.012)), (0.09, 0.09, 0.5*(PEDESTAL_TOP - 0.012)),
                     "pedestal_steel")
            self.box(f"{name}_plate", (x, y, PEDESTAL_TOP - 0.006), (0.12, 0.12, 0.006), "pedestal_plate")
        self.liquid_planes = [((0., 0., 1.), BAR_TOP, (cx - 0.5*P.BAR_LENGTH_M, BAR_Y0, P.BAR_TOP_THICK_M),
                               (cx + 0.5*P.BAR_LENGTH_M, BAR_Y1)),
                              ((0., 0., 1.), 0.0, (-100., -100.), (100., 100.))]   # the floor: unbounded (a splash
                              # droplet thrown 12 m past a +-4 m floor fell forever and faulted the episode at 20 m/s)
        # the room's four walls (P.ROOM; 0.3 m thick for the liquid: a particle further behind one is not its concern)
        R = P.ROOM
        self.liquid_planes += [((0., 1., 0.), R["y0"], (-100., -100., 0.3), (100., 100.)),
                               ((0., -1., 0.), -R["y1"], (-100., -100., 0.3), (100., 100.)),
                               ((1., 0., 0.), R["x0"], (-100., -100., 0.3), (100., 100.)),
                               ((-1., 0., 0.), -R["x1"], (-100., -100., 0.3), (100., 100.))]

    # ------------------------------------------------------------------ robots
    def robots(self):
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("wine-pouring", self.c.robot)
        self.robot_setup = setup
        self._hardware(setup)
        mounts = {k: mount((x, y, PEDESTAL_TOP), P.ARM_YAW) for k, (x, y) in P.ARM_BASE_XY.items()}
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        self._ensure_hand_mimics()
        # barcode_scanning E17: the catalog's default 0.1 m contact envelope gave false deep contacts; boxes 0, hulls 2 mm
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

    def _ensure_hand_mimics(self):
        """Keep each Franka Hand's URDF rack coupling when composing the robot."""
        b = self.b
        label_to_joint = {label: i for i, label in enumerate(b.joint_label)}
        coupled = {
            (int(follower), int(leader))
            for follower, leader in zip(b.constraint_mimic_joint0, b.constraint_mimic_joint1)
        }
        for arm, handle in self.rig_pending.instances.items():
            fingers = list(handle.groups.get("gripper", {}).get("joints", ()))
            if len(fingers) != 2:
                continue
            joints = [label_to_joint[handle.joints[name]["native_label"]] for name in fingers]
            if (joints[1], joints[0]) not in coupled:
                b.add_constraint_mimic(joint0=joints[1], joint1=joints[0],
                                       label=f"{arm}/franka_hand_rack")

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
        """Franka Hand finger colliders (URDF boxes; Panda meshes get the same boxes). The rubber tip box (17.5 x 15.2 x
        18.5 mm at z = 45.25 mm) is the pad (barcode_scanning/label_application lineage)."""
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
                        b.shape_material_mu[i] = P.MU["pad_glass"]
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
                             body=body, rot=tuple(R.as_quat()), density=0., mu=P.MU["pad_glass"])
                if abs(xyz[2] - pad_z) < 1e-6:
                    self.pad_shapes[i] = b.body_label[body]

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on link7 (control_cable / label_application / barcode_scanning mount geometry)."""
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

    # ------------------------------------------------------------------ glassware
    def glassware(self):
        """Every container in the task layout is a free body whose collision is its geometry's convex pieces and whose
        mass/inertia are the polygon integrals (datasheet mass where published)."""
        wp = self.wp
        self.containers = []          # (body, geometry, item name)
        for item in self.layout["items"]:
            geo = GEO.get(item["kind"], item.get("color"))
            pos = (item["xy"][0], item["xy"][1], item.get("z", BAR_TOP) + 1e-4)
            rot = quat_axis_angle((0, 0, 1), item.get("yaw", 0.))
            body = self.b.add_body(xform=wp.transform(pos, rot), label=item["name"])
            self.bodies[item["name"]] = body
            mu = P.MU["bottle_coaster"] if item["kind"] == "coaster" else P.MU["glass_glass"]
            for pi_, piece in enumerate(geo.rigid):
                for hk, pts in enumerate(GEO.rigid_hull_points(piece)):
                    import trimesh
                    hull = trimesh.convex.convex_hull(pts)
                    mesh = self.newton.Mesh(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32).ravel())
                    i = self.b.add_shape_convex_hull(body, mesh=mesh, xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                                     cfg=self.cfg(gap=P.MESH_CONTACT_GAP_M, density=0., mu=mu),
                                                     label=f"{item['name']}_p{pi_}_{hk}")
                    self.materials[i] = "collision_hidden"
                    self.visual.add(i)            # never drawn: the render uses the revolved polygon
            m, com, inertia = geo.mass_properties()
            self.set_mass(body, m, com, inertia)
            self.items[item["name"]] = dict(kind=item["kind"], body=body, color=item.get("color"), role=item.get("role"),
                                            liquid=item.get("liquid"), volume=item.get("volume", 0.), spot=item.get("spot"),
                                            geometry=geo.kind, mass=m)
            self.containers.append((body, geo, item["name"]))

    # ------------------------------------------------------------------ fixed cameras
    def views(self):
        """Station camera (policy) on a ceiling post above the bar, looking down at the work area; diagnostic views
        (room, bar-level side, close-ups) render the same state. The station camera is a fixed RGB like Toni's."""
        self.cameras["station_rgb"] = dict(body=-1, eye=[0.0, -0.10, 2.02], target=[0.0, 0.28, 1.09], up=[0, 0, 1],
                                           fov=75., role="policy")
        # a fixed policy camera at bar level on the guest side, looking across the four delivery spots through the
        # glasses, so a fill level can be judged by eye from a fixed view (from the ceiling the station camera sees an ISO glass a few tens of pixels across). Opt-in (cameras=...)
        self.cameras["bar_side_rgb"] = dict(body=-1, eye=[0.0, 1.05, 1.24], target=[0.0, 0.40, 1.15], up=[0, 0, 1],
                                            fov=62., role="policy")
        self.cameras["room"] = dict(body=-1, eye=[1.55, 1.85, 1.75], target=[0.0, 0.10, 1.10], up=[0, 0, 1], fov=55.,
                                    role="diagnostic")
        self.cameras["guest"] = dict(body=-1, eye=[0.05, 1.25, 1.42], target=[0.0, 0.15, 1.12], up=[0, 0, 1], fov=50.,
                                     role="diagnostic")
        self.cameras["side"] = dict(body=-1, eye=[0.95, 0.30, 1.22], target=[0.0, 0.25, 1.17], up=[0, 0, 1], fov=40.,
                                    role="diagnostic")
        if "probe_camera" in self.layout:                      # a probe's own diagnostic view
            pc = self.layout["probe_camera"]
            self.cameras["probe"] = dict(body=-1, eye=list(pc["eye"]), target=list(pc["target"]), up=[0, 0, 1],
                                         fov=float(pc.get("fov", 45.)), role="diagnostic")
        tgt = self.layout.get("focus", (0.0, 0.30, 1.16))
        self.cameras["closeup"] = dict(body=-1, eye=[tgt[0] + 0.50, tgt[1] + 0.50, tgt[2] + 0.22],
                                       target=[tgt[0], tgt[1], tgt[2] + 0.10], up=[0, 0, 1], fov=48., role="diagnostic")

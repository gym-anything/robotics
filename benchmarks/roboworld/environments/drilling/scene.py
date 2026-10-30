"""Physics scene (Newton 1.5 ModelBuilder): the basement maintenance workshop, its two stations, the catalog robot on
its cart, tools, parts and drillable workpieces.

World frame: x east, y north, z up, origin at the floor centre of the 8 x 6 m room. The north wall (y = +3.0) is the
cast-in-place concrete wall of preset `wall`; the east wall carries the drill-press bench of preset `bench`.
Drillable workpieces (holes.Drillable) are boxes with a welded proxy link; hole-capable items (holes.Item) are welded
links of their tools or parts. Filters are declared as whole-body pairs, which the solver emits as MuJoCo excludes.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import tools as T
from .holes import Drillable, Item, N_CAGE, CAGE_THICK
from .layout import task_layout


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    s = math.sin(angle/2)
    return (float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(math.cos(angle/2)))


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual = {}, set()
        self.bodies, self.parts, self.cameras = {}, {}, {}
        self.static_boxes, self.static_cyls = [], []
        self.tool_bodies = set()
        self.tools = {}
        self.drillables, self.items = {}, {}
        self.item_bodies = set()
        self.filter_groups = []          # lists of bodies whose members never collide with each other
        self.filter_pairs = []           # (body_a, body_b) whole-body pairs
        self.static_body_marker = None
        self.initial_dimples, self.initial_holes = [], []
        self.robot_bodies, self.pad_shapes = [], {}
        self.stiff_shapes = set()        # hardened clamping faces (vise jaws, rails): stiff steel contact (world.py)
        self.stiff_solids = set()        # metal workpiece solids among them (params.SOLID_SOLIMP)
        self.clamp_shapes = set()        # ... the hardened faces of the hold-down clamps (params.HOLD_DOWN_SOLIMP)
        self.bearing_shapes = set()      # ... of which the load-bearing ones, and the wooden parts of the clamped stacks:
        self.stack_shapes = set()        # (the wooden parts): no rolling resistance (params.HOLD_DOWN_BEARING_ROLLS)
        self.loose_shapes = set()        # small loose parts whose own contact law rules on a hardened face (params.LOOSE_SOLMIX)
        self.arm_hw, self.joints, self.arms = {}, {}, {}
        self.layout = config.layout if getattr(config, "layout", None) else task_layout(config)
        L = self.layout
        self.room()
        if L["preset"] == "wall":
            from .stations import build_wall
            build_wall(self)
        elif L["preset"] == "bench":
            from .stations import build_bench
            build_bench(self)
        elif L["preset"] == "probe":
            from .stations import build_probe
            build_probe(self)
        if L.get("robot", True) and config.robot != "none":
            self.robots()
        self.views()
        self.apply_filters()

    # ------------------------------------------------------------------ shape helpers
    def cfg(self, **kw):
        v = dict(mu=P.MU["default"], ke=2e4, kd=100., density=1000., gap=P.CONTACT_GAP_ROUND_M, margin=0., mu_torsional=0.,
                 mu_rolling=0.)
        v.update(kw)
        return self.newton.ModelBuilder.ShapeConfig(**v)

    def visual_cfg(self):
        return self.newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False, has_particle_collision=False)

    def box(self, name, pos, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**dict(dict(gap=P.CONTACT_GAP_M), **kw)) if collide else self.visual_cfg()
        i = self.b.add_shape_box(body, xform=self.wp.transform(tuple(map(float, pos)), rot or self.wp.quat_identity()),
                                 hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        if body < 0:
            self.static_boxes.append((name, tuple(map(float, pos)), tuple(map(float, half)), material,
                                      tuple(rot) if rot is not None else (0., 0., 0., 1.)))
        return i

    def cyl(self, name, pos, radius, half_h, axis, material, body=-1, collide=True, **kw):
        return T.Parts(self).cyl(body, name, pos, radius, half_h, axis, material, collide=collide, **kw)

    def mutual_filter(self, bodies):
        self.filter_groups.append(list(bodies))

    def filter_bodies(self, a, b):
        self.filter_pairs.append((int(a), int(b)))

    # ------------------------------------------------------------------ drillables and items
    def add_drillable(self, name, material, pose, half, *, static=False, n_slots=4, density=None, mu=None,
                      render_material=None, fields=None, fields_map=None, extra_boxes=(), parent=None,
                      parent_xform=None):
        """A drillable box (centred on its body origin) with a welded proxy link carrying n_slots proxy slots.
        extra_boxes: [(name, pos, half, material)] solid shapes on a further welded link (not drillable, and not
        excluded from items). Returns the Drillable record."""
        b, wp = self.b, self.wp
        p, R = pose
        dens = density if density is not None else P.DENSITY[material]
        Wb = T.link(self, name, pose)
        solid = self.box(f"{name}_solid", (0., 0., 0.), half, render_material or material, body=Wb, density=dens,
                         mu=mu if mu is not None else P.MU.get(material, P.MU["default"]))
        if material in ("steel", "aluminium"):
            self.stiff_shapes.add(solid)            # hard metal faces (a 13 g bolt head sank 1.9 mm into soft steel)
            self.stiff_solids.add(solid)            # ... with the softer SOLID_SOLIMP (world.py)
        Pk = T.link(self, f"{name}_proxy", pose, mass=1e-3, inertia=(1e-7, 1e-7, 1e-7))
        # proxy shapes meet only items (bits, bolts, anchors, rods: round shapes and hulls), never a box, so their boxes
        # keep the round-shape gap: at the box gap the hole walls' contact envelope halved and a 6 mm bit lunging at a
        # thin sheet's breakthrough met a wall within one tick and snapped (bending 41.7 N m; E6 unbacked)
        light = self.cfg(density=0., mu=0.5)
        light_box = light
        slots = []
        for k in range(n_slots):
            bottom = b.add_shape_cylinder(Pk, xform=wp.transform_identity(), radius=0.005, half_height=0.01, cfg=light,
                                          label=f"{name}_slot{k}_bottom")
            pad = b.add_shape_box(Pk, xform=wp.transform_identity(), hx=0.01, hy=0.01, hz=0.01, cfg=light_box,
                                  label=f"{name}_slot{k}_pad")
            cage = [b.add_shape_box(Pk, xform=wp.transform_identity(), hx=0.5*CAGE_THICK, hy=0.002, hz=0.01, cfg=light_box,
                                    label=f"{name}_slot{k}_cage{j}") for j in range(N_CAGE)]
            funnel = [b.add_shape_box(Pk, xform=wp.transform_identity(), hx=0.5*CAGE_THICK, hy=0.002, hz=0.001, cfg=light_box,
                                      label=f"{name}_slot{k}_funnel{j}") for j in range(N_CAGE)]
            for s in [bottom, pad, *cage, *funnel]:
                self.materials[s] = "proxy_hidden"
            slots.append(dict(bottom=bottom, pad=pad, cage=cage, funnel=funnel))
        X = None
        if extra_boxes:
            X = T.link(self, f"{name}_extra", pose)
            for nm, pos, hh, mat in extra_boxes:
                self.box(nm, pos, hh, mat, body=X, density=dens, mu=mu if mu is not None else P.MU.get(material, 0.5))
        I = wp.transform_identity()
        joints = []
        if static:
            joints.append(b.add_joint_fixed(-1, Wb, parent_xform=wp.transform(tuple(map(float, p)), T.wq(R)),
                                            child_xform=I, label=f"{name}_anchor"))
        elif parent is not None:
            joints.append(b.add_joint_fixed(parent, Wb, parent_xform=parent_xform, child_xform=I, label=f"{name}_weld"))
        else:
            joints.append(b.add_joint_free(Wb, label=f"{name}_free"))
        joints.append(b.add_joint_fixed(Wb, Pk, parent_xform=I, child_xform=I, label=f"{name}_proxy_weld"))
        if X is not None:
            joints.append(b.add_joint_fixed(Wb, X, parent_xform=I, child_xform=I, label=f"{name}_extra_weld"))
        b.add_articulation(joints, label=f"{name}_assembly")
        d = Drillable(name, Wb, Pk, material, np.zeros(3), np.asarray(half, float), slots, fields=fields,
                      fields_map=fields_map, static=static)
        d.extra_body = X
        d.render_material = render_material or material
        self.drillables[name] = d
        self.bodies[name] = Wb
        self.parts[name] = dict(kind="drillable", body=Wb, material=material, half=list(map(float, half)))
        return d

    def add_item(self, it):
        self.items[it.name] = it
        self.item_bodies.add(it.body)
        return it

    def add_tool(self, tool):
        self.tools[tool.name] = tool
        for k in ("housing", "body", "quill"):
            if hasattr(tool, k):
                self.bodies[tool.name] = getattr(tool, k)
                break
        if getattr(tool, "item", None) is not None:
            self.add_item(tool.item)
        return tool

    def apply_filters(self):
        """Body-pair filters: items vs drillable solids; proxy links vs everything but items; declared groups and
        pairs. Expressed as all shape pairs of each body pair (the solver turns full body pairs into excludes)."""
        b = self.b
        shapes_of = {}
        for s in range(b.shape_count):
            if int(b.shape_flags[s]) & int(self.newton.ShapeFlags.COLLIDE_SHAPES):
                shapes_of.setdefault(int(b.shape_body[s]), []).append(s)
        pairs = set()
        for g in self.filter_groups:
            for i, a in enumerate(g):
                for c in g[i + 1:]:
                    if a != c:
                        pairs.add((min(a, c), max(a, c)))
        for a, c in self.filter_pairs:
            pairs.add((min(a, c), max(a, c)))
        proxies = {d.proxy_body for d in self.drillables.values()}
        for d in self.drillables.values():
            for ib in self.item_bodies:
                pairs.add((min(ib, d.body), max(ib, d.body)))
            for body in shapes_of:
                if body in self.item_bodies or body == d.proxy_body:
                    continue
                pairs.add((min(body, d.proxy_body), max(body, d.proxy_body)))
        self.n_filter_body_pairs = len(pairs)
        for a, c in pairs:
            for sa in shapes_of.get(a, []):
                for sc in shapes_of.get(c, []):
                    b.add_shape_collision_filter_pair(sa, sc)
        self.proxy_bodies = proxies

    # ------------------------------------------------------------------ room
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=P.MU["floor"]), label="floor")
        self.materials[b.shape_count - 1] = "floor_concrete"
        R = P.ROOM
        if self.layout["preset"] == "probe":
            return
        # walls as static boxes (the drillable wall region of preset `wall` is a separate fixed link, stations.py)
        t = 0.2
        self.box("wall_south", (0., R["y0"] - 0.5*t, 0.5*R["h"]), (0.5*(R["x1"] - R["x0"]), 0.5*t, 0.5*R["h"]),
                 "block_paint")
        self.box("wall_west", (R["x0"] - 0.5*t, 0., 0.5*R["h"]), (0.5*t, 0.5*(R["y1"] - R["y0"]), 0.5*R["h"]),
                 "block_paint")
        self.box("wall_east", (R["x1"] + 0.5*t, 0., 0.5*R["h"]), (0.5*t, 0.5*(R["y1"] - R["y0"]), 0.5*R["h"]),
                 "block_paint")
        self.box("ceiling", (0., 0., R["h"] + 0.1), (0.5*(R["x1"] - R["x0"]), 0.5*(R["y1"] - R["y0"]), 0.1), "ceiling",
                 collide=False)

    # ------------------------------------------------------------------ robots
    def robots(self):
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("drilling", self.c.robot)
        self.robot_setup = setup
        L = self.layout
        self._hardware(setup)
        mounts = {k: mount(tuple(v["xyz"]), v["yaw"]) for k, v in L["mounts"].items()}
        first, first_shape = self.b.body_count, self.b.shape_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first, self.b.body_count))
        for i in range(first_shape, self.b.shape_count):
            box = self.b.shape_type[i] in (self.newton.GeoType.BOX, self.newton.GeoType.PLANE)
            self.b.shape_gap[i] = P.CONTACT_GAP_M if box else P.MESH_CONTACT_GAP_M
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            self.arms[name] = handle
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._fingertips(name, handle)
            self._wrist_camera(name, handle)

    def _hardware(self, setup):
        from robots.catalog import model
        data = setup.to_dict()
        for name, inst in data["instances"].items():
            spec = model(inst["model"]).to_dict()
            self.arm_hw[name] = dict(model=inst["model"], mount_name=inst["mount"],
                                     wrist_link=spec.get("frames", {}).get("wrist", {}).get("parent"),
                                     arm_joints=spec["groups"]["arm"]["joints"],
                                     gripper_joints=spec["groups"].get("gripper", {}).get("joints", []))
        for att in data.get("attachments", {}).values():
            owner = att["parent"].split("/")[0]
            pads = model(att["model"]).to_dict().get("contact", {}).get("fingertip_pads")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)

    FRANKA_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                           ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                           ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                           ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

    def _fingertips(self, arm, handle):
        """Franka Hand finger colliders (URDF boxes; Panda meshes get the same boxes); the rubber tip box is the pad
        (barcode_scanning / wine_pouring lineage)."""
        b = self.b
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            boxes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.BOX]
            pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
            if boxes:
                for i in boxes:
                    if abs(float(np.asarray(b.shape_transform[i])[2]) - pad_z) < 1e-4:
                        self.pad_shapes[i] = b.body_label[body]
                        b.shape_material_mu[i] = P.MU["pad"]
                continue
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            if not meshes:
                continue
            tf = np.asarray(b.shape_transform[meshes[0]], float)
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            for k, (xyz, rpy, size) in enumerate(self.FRANKA_FINGER_BOXES):
                R = Rm*Rotation.from_euler("xyz", rpy)
                p = pm + Rm.apply(xyz)
                i = self.box(f"{b.body_label[body].split('/')[-1]}_box{k}", tuple(p), tuple(np.asarray(size)/2),
                             "robot_dark", body=body, rot=tuple(R.as_quat()), density=0., mu=P.MU["pad"])
                if abs(xyz[2] - pad_z) < 1e-6:
                    self.pad_shapes[i] = b.body_label[body]

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
                                                up=[0, 1, 0], fov=87., role="policy", width=640, height=480)

    # ------------------------------------------------------------------ cameras
    def views(self):
        for name, cam in self.layout.get("cameras", {}).items():
            self.cameras[name] = dict(body=-1, **cam)

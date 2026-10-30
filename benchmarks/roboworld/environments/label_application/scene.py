"""Labeling workcell geometry: rigid world (Newton/MuJoCo) plus analytic label colliders.

All physics-bearing geometry is authored from the dimensions given in this module.
Items are free rigid bodies with mass and friction; the bench, printer and riser
are static. Visual-only dressing has no collision and is
marked so in the materials map ("visual" set).
"""

import math
import numpy as np
from . import materials as M, layouts
from .labelsim import Collider

TABLE = 0.91                  # bench top height (authored packing-table height)
# robot pedestal mounts (xy on the bench-height pedestal tops, yaw): the single source for scene and IK
ARM_MOUNTS = {"left_arm_mount": ((-.42, .52), -math.pi/2), "right_arm_mount": ((.42, .52), -math.pi/2)}
IN = 0.0254

# Physical label formats: (length along printer feed = MD, width across head = CD)
LABELS = {
    "4x6": dict(md=6*IN, cd=4*IN, stock="paper"),      # GS1 A6/4x6 [GS1 LLG 1.3 §7]
    "2x1": dict(md=1*IN, cd=2*IN, stock="paper"),      # CLSI AUTO12 2x1 in; 2-in printer feeds 1 in
    "2x4": dict(md=2*IN, cd=4*IN, stock="pet"),        # 3M 7818-class ID label, 4-in head
}


# The Franka Hand's rubber fingertips have rounded edges; the URDF gives only their bounding box
# (17.5 x 15.2 x 18.5 mm). Label contact uses a rounded box with this edge radius (estimate from
# product photos): a sharp box edge flipped the contact normal in-plane and stretched pinched paper.
LINER_OVER_BAR = 0.0          # m the liner-web collider continues onto the peel bar (tested 1.5 mm)
FINGERTIP_ROUNDING = 2e-3


class Scene:
    def __init__(self, config, newton, wp):
        self.c, self.newton, self.wp = config, newton, wp
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=-9.81)
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.visual, self.bodies, self.colliders = {}, set(), {}, []
        self.items, self.cameras, self.label_spec, self.task_geometry = {}, {}, None, {}
        self.printers = {}          # printer frames for the visual printer model (dressing.py)
        self.target = {"parcel": "carton", "specimen": "tube", "part": "part"}[config.preset]   # labelled item
        self.rng = np.random.default_rng(config.seed)
        self.room()
        self.robots()
        getattr(self, config.preset)()

    # ------------------------------------------------------------------ helpers
    def cfg(self, **kw):
        v = dict(mu=.45, ke=2e4, kd=100., density=1000., gap=.002, margin=0., mu_torsional=0., mu_rolling=0.)
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
        return i

    def cyl(self, name, pos, radius, half, material, body=-1, rot=None, collide=True, **kw):
        cfg = self.cfg(**kw) if collide else self.newton.ModelBuilder.ShapeConfig(
            density=0., has_shape_collision=False, has_particle_collision=False)
        i = self.b.add_shape_cylinder(body, xform=self.wp.transform(pos, rot or self.wp.quat_identity()),
                                      radius=radius, half_height=half, cfg=cfg, label=name)
        self.materials[i] = material
        if not collide:
            self.visual.add(i)
        return i

    def cyl_hull(self, name, pos, radius, half, material, body, sections=32, **kw):
        """Cylinder as a convex hull (32 facets): MuJoCo's multi-point convex contact supports
        mesh-box pairs but not cylinder-box, and single-point cylinder contact lets a resting
        cylinder rock and creep. The label solver keeps the exact analytic cylinder."""
        import trimesh
        import newton
        tm = trimesh.creation.cylinder(radius=radius, height=2*half, sections=sections)
        m = newton.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).ravel())
        i = self.b.add_shape_convex_hull(body, mesh=m, xform=self.wp.transform(pos, self.wp.quat_identity()),
                                         cfg=self.cfg(**kw), label=name)
        self.materials[i] = material
        return i

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

    # ------------------------------------------------------------------ room
    def room(self):
        b = self.b
        b.add_ground_plane(cfg=self.cfg(mu=.6), label="floor")
        self.materials[b.shape_count-1] = "floor"
        z = TABLE
        # packing bench 1.83 x 0.76 m, 40 mm laminate top, steel frame and lower shelf
        self.box("bench_top", (0., 0., z-.02), (.915, .38, .02), "laminate", mu=.4)
        self.colliders.append(Collider("bench_top", "box", (.915, .38, .02), M.SURFACES["laminate"],
                                       local_pos=(0., 0., z-.02), role="static"))
        for x in (-.86, .86):
            for y in (-.33, .33):
                self.box(f"bench_leg_{x}_{y}", (x, y, (z-.04)/2), (.02, .02, (z-.04)/2), "steel_frame")
        self.box("bench_shelf", (0., 0., .18), (.88, .34, .01), "laminate_dark")
        # room shell, lights and all non-touchable dressing are visual only (dressing.py)
        # robot pedestals: bolted steel columns behind the bench (FR3 base at bench height)
        for mount_name, ((x, y), _) in ARM_MOUNTS.items():
            name = mount_name.split("_")[0]
            self.box(f"pedestal_{name}", (x, y, (z-.004)/2), (.11, .11, (z-.004)/2), "pedestal")
            self.box(f"pedestal_plate_{name}", (x, y, z-.002), (.13, .13, .002), "steel_frame")

    def robots(self):
        """Robot deployment from the shared catalog (robots/rigs/label-*.json)."""
        from robots.catalog import for_environment
        from robots.backends.newton import mount
        import robots
        setup = for_environment("label-application", self.c.robot)
        self.robot_setup = setup
        self._hardware(setup)
        mounts = {k: mount((x, y, TABLE), yaw) for k, ((x, y), yaw) in ARM_MOUNTS.items()}
        first_body = self.b.body_count
        self.rig_pending = robots.add_to(self.b, setup, mounts=mounts)
        self.robot_bodies = list(range(first_body, self.b.body_count))
        self.joints = {}
        self.arms = {}
        for name, handle in self.rig_pending.instances.items():
            for j, entry in handle.joints.items():
                self.joints[j] = entry
            self.arms[name] = handle
            for i in handle.shape_ids:
                self.materials[i] = "robot"
            self._finger_colliders(name, handle)
            self._wrist_camera(name, handle)

    def _hardware(self, setup):
        """Per-arm hardware facts, read from the shared robot catalog (never assumed here): the arm's mount
        pose (the rig's `mount` -> ARM_MOUNTS, also used by the IK), its wrist link (model frames.wrist) and
        the hand's fingertip-pad geometry (attachment contact.fingertip_pads). An attachment that declares
        no fingertip pads cannot touch a label and is rejected with that reason."""
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
            if pads is None:
                raise ValueError(f"label-application needs label-contact fingertip pads; attachment {att['model']} "
                                 "declares none (robots/attachments/<id>/robot.json: contact.fingertip_pads)")
            self.arm_hw[owner].update(hand=att["model"], pads=pads)
        for name, hw in self.arm_hw.items():
            if "pads" not in hw or not hw["gripper_joints"]:
                raise ValueError(f"arm {name}: no hand with fingertip pads / gripper joints for label contact")

    # Franka Hand finger collision boxes (left finger frame) from the manufacturer's fr3_franka_hand.urdf:
    # (xyz, rpy, full size). The Panda and the FR3 carry the same Franka Hand; the Panda URDF models its
    # fingers as meshes, so the same boxes are authored for label contact on a Panda finger.
    FRANKA_HAND_FINGER_BOXES = (((0., 18.5e-3, 11e-3), (0., 0., 0.), (22e-3, 15e-3, 20e-3)),
                                ((0., 6.8e-3, 2.2e-3), (0., 0., 0.), (22e-3, 8.8e-3, 3.8e-3)),
                                ((0., 15.9e-3, 28.35e-3), (math.pi/6, 0., 0.), (17.5e-3, 7e-3, 23.5e-3)),
                                ((0., 7.58e-3, 45.25e-3), (0., 0., 0.), (17.5e-3, 15.2e-3, 18.5e-3)))

    def _finger_colliders(self, arm, handle):
        """Every Franka Hand finger collision box from the manufacturer URDF becomes a label collider;
        the rubber tip (17.5 x 15.2 x 18.5 mm at z = 45.25 mm) is the grasping pad."""
        b = self.b
        from scipy.spatial.transform import Rotation
        fingers = sorted({b.shape_body[i] for i in handle.shape_ids
                          if b.shape_body[i] >= 0 and "finger" in b.body_label[b.shape_body[i]]})
        for body in fingers:
            shapes = [i for i in handle.shape_ids if b.shape_body[i] == body]
            if any(b.shape_type[i] == self.newton.GeoType.BOX for i in shapes):
                continue
            meshes = [i for i in shapes if b.shape_type[i] == self.newton.GeoType.MESH]
            if not meshes:
                continue
            tf = np.asarray(b.shape_transform[meshes[0]], float)       # the finger mesh's own offset
            Rm, pm = Rotation.from_quat(tf[3:]), tf[:3]
            short = b.body_label[body].split("/")[-1]
            if self.arm_hw[arm]["hand"].split("@")[0] != "franka-hand":
                raise ValueError(f"{self.arm_hw[arm]['hand']}: finger meshes without collision boxes are qualified only "
                                 "for the Franka Hand (its boxes are authored from the manufacturer URDF)")
            pad_z = self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]
            for k, (xyz, rpy, size) in enumerate(self.FRANKA_HAND_FINGER_BOXES):
                pad = abs(xyz[2] - pad_z) < 1e-6
                R = Rm*Rotation.from_euler("xyz", rpy)
                # the same boxes as rigid collision shapes (massless, not rendered): the Panda's finger
                # meshes never meet when closed, so without them no pinch load reaches MuJoCo
                self.box(f"{arm}_{short}_hand_box{k}", tuple(pm + Rm.apply(xyz)), tuple(.5*v for v in size), "robot",
                         body=body, rot=self.wp.quat(*map(float, R.as_quat())), density=0.)
                self.colliders.append(Collider(f"{arm}_{short}_{'pad' if pad else 'link'}_hand{k}", "box",
                                               tuple(.5*v for v in size), M.SURFACES["finger_pad" if pad else "pp"],
                                               body=body, local_pos=tuple(pm + Rm.apply(xyz)),
                                               local_rot=tuple(R.as_quat()), role="pad" if pad else "robot",
                                               rounding=FINGERTIP_ROUNDING if pad else 1e-3))
        for i in handle.shape_ids:
            body = b.shape_body[i]
            if body < 0 or "finger" not in b.body_label[body] or b.shape_type[i] != self.newton.GeoType.BOX:
                continue
            tf = np.asarray(b.shape_transform[i], float)
            half = tuple(float(v) for v in b.shape_scale[i])
            pad = abs(tf[2] - self.arm_hw[arm]["pads"]["finger_frame_offset_z_m"]) < 1e-4
            short = b.body_label[body].split("/")[-1]
            self.colliders.append(Collider(f"{arm}_{short}_{'pad' if pad else 'link'}_{i}", "box", half,
                                           M.SURFACES["finger_pad" if pad else "pp"], body=body,
                                           local_pos=tuple(tf[:3]), local_rot=tuple(tf[3:]),
                                           role="pad" if pad else "robot",
                                           rounding=FINGERTIP_ROUNDING if pad else 1e-3))

    def _wrist_camera(self, arm, handle):
        """D405-envelope wrist RGB on the arm's wrist link (model frames.wrist; link7 on the FR3 and Panda),
        same mount geometry as control_cable."""
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

    # ------------------------------------------------------------------ printer
    def printer(self, x, y, head_width, name="printer", riser=.28, yaw=math.pi/2, lip=.035, kind="direct_thermal",
                hd=.11, ht=.17):
        """Desktop label printer in peel-off mode on a bench shelf, label exit facing `yaw` (default +y:
        toward the robots, as a pack-station printer faces its operator). `kind`: "direct_thermal" (paper
        labels) or "thermal_transfer" (ribbon printer: the PET 7818 part label is a thermal-transfer
        material). Housing half-depth `hd` and height `ht` are authored per class (4-inch ~221 x 254 x 170 mm
        ZD421-class footprint; 2-inch smaller). Built in printer coordinates (u = exit direction,
        v = lateral) and rotated about z."""
        z = TABLE
        c, s_ = math.cos(yaw), math.sin(yaw)
        q = self.wp.quat_from_axis_angle(self.wp.vec3(0, 0, 1), float(yaw))
        qx = (0., 0., math.sin(yaw/2), math.cos(yaw/2))
        W = lambda u, v, zz: (x + c*u - s_*v, y + s_*u + c*v, zz)

        def pbox(label, u, v, zz, half, material, collide=True, collider=None, surface=None, role="static"):
            self.box(label, W(u, v, zz), half, material, rot=q, collide=collide)
            if collider:
                self.colliders.append(Collider(collider, "box", half, M.SURFACES[surface], local_pos=W(u, v, zz),
                                               local_rot=qx, role=role))
        # printer shelf: the printer's front (label exit) overhangs the shelf so the label hangs free
        pbox(f"{name}_riser_top", -.04, 0, z+riser-.006, (.12, .16, .006), "stand_grey")
        for sx in (-1, 1):
            pbox(f"{name}_riser_side_{sx}", -.04+sx*.11, 0, z+(riser-.012)/2, (.01, .16, (riser-.012)/2), "stand_grey")
        hw = head_width/2 + .06
        base = z + riser
        exit_z = base + .058
        self.printers[name] = dict(x=x, y=y, yaw=yaw, base=base, hw=hw, exit_z=exit_z, head_width=head_width,
                                   lip=lip, riser=riser, kind=kind, hd=hd, ht=ht)
        self.box(f"{name}_housing", W(0, 0, base+ht/2), (hd, hw, ht/2), "printer_body", rot=q)
        # label colliders leave the real exit slot (3 mm below / 4 mm above the web) open
        lo_h, hi_h = (exit_z - .003 - base)/2, (base + ht - exit_z - .004)/2
        self.colliders.append(Collider(f"{name}_housing_lower", "box", (hd, hw, lo_h), M.SURFACES["pp"],
                                       local_pos=W(0, 0, base+lo_h), local_rot=qx, role="static"))
        self.colliders.append(Collider(f"{name}_housing_upper", "box", (hd, hw, hi_h), M.SURFACES["pp"],
                                       local_pos=W(0, 0, exit_z+.004+hi_h), local_rot=qx, role="static"))
        pbox(f"{name}_slot", hd + .0005, 0, exit_z+.0005, (.0006, head_width/2+.004, .0035), "printer_dark", collide=False)
        pbox(f"{name}_lid_seam", -.02, 0, base+ht+.001, (hd - .02, hw-.002, .001), "printer_dark", collide=False)
        # dispenser door with peel bar and a presenter lip. The lip is an AUTHORED exit guide (35 mm on the
        # 4-inch printer, none on the 2-inch printer, whose 1-inch-long label must overhang the 6 mm peel bar
        # by ~15 mm for pickup, 12 mm on the part printer's 2-inch-long label): it sets how far the presented
        # label overhangs for pickup and carries part of the label adhesive-down on its release surface; no
        # referenced printer documents one
        pbox(f"{name}_peel_bar", hd - .002 + (lip + .006)/2, 0, exit_z-.002, ((lip + .006)/2, hw-.01, .002), "printer_dark",
             collider=f"{name}_peel_bar", surface="peel_plate", role="plate")
        pbox(f"{name}_door", hd + .002, 0, exit_z-.025, (.003, hw-.004, .025), "printer_body")
        pbox(f"{name}_display", hd - .04, -hw*.55, base+ht, (.02, .025, .002), "printer_screen", collide=False)
        # liner exit path: the backing web runs over the bar and down (kinematic, printer-driven). Its collider
        # ends where the peel bar begins (LINER_OVER_BAR = 0)
        ov = LINER_OVER_BAR
        pbox(f"{name}_liner_web", hd - .008 + ov/2, 0, exit_z-.0001, (.006 + ov/2, head_width/2, .0001), "liner_paper", collide=False,
             collider=f"{name}_liner", surface="liner", role="liner")
        md = np.array([c, s_, 0.])
        cd = np.cross([0., 0., 1.], md)
        exit_pt = np.array(W(hd - .002, 0, exit_z))
        return exit_z, exit_pt, md, cd

    def _presented(self, fmt, liner_md, printer, h=None):
        """Label lying on the printer's exit path: trailing `liner_md` still on the liner web."""
        exit_z, exit_pt, md, cd = printer
        spec = LABELS[fmt]
        origin = exit_pt - md*liner_md - cd*spec["cd"]/2
        return dict(fmt=fmt, **spec, h=h or self.c.label_mesh_m, origin=tuple(origin), axis_md=tuple(md),
                    axis_cd=tuple(cd), liner_bonded_md=liner_md, printer="printer")

    # ------------------------------------------------------------------ presets
    def parcel(self):
        z = TABLE
        task = self.c.task
        self.label_spec = self._presented("4x6", .010, self.printer(-.30, -.17, 4*IN))
        # carton: 16 x 12 x 8 in RSC, C-flute; seam tape (48 mm PP) along x on top
        L, W, H = 16*IN, 12*IN, 8*IN
        jitter = self.rng.uniform(-.02, .02, 2)
        yaw = self.rng.uniform(-.1, .1)
        if task == "gs1-case-side":
            pos = (.12+jitter[0], -.12+jitter[1], z+H/2)
        else:
            pos = (.12+jitter[0], -.06+jitter[1], z+H/2)
        q = self.wp.quat_from_axis_angle(self.wp.vec3(0, 0, 1), float(yaw))
        body = self.free_body("carton", pos, q)
        self.box("carton_shell", (0, 0, 0), (L/2, W/2, H/2), "kraft", body=body, mu=.4, density=100.)
        self.set_mass(body, .55 + self.c.carton_content_kg)   # board ~0.55 kg + contents
        self.box("carton_tape_top", (0, 0, H/2+.00001), (L/2+.0005, .024, .00001), "tape", body=body, collide=False)   # visual: top 20 um, under an applied label (84 um)
        for sx in (-1, 1):
            self.box(f"carton_tape_end_{sx}", (sx*(L/2+.00005), 0, H/2-.035), (.00005, .024, .035), "tape",
                     body=body, collide=False)
        self.box("carton_flap_seam", (0, 0, H/2+.00003), (L/2, .0006, .00001), "seam", body=body, collide=False)
        patches = [(M.SURFACES["pp_tape"], 2, 1, (-L/2, L/2, -.024, .024))]
        if task == "cover-old-label":
            # the previous shipment's 4 x 3 in address label, already on the carton top: seeded position on either
            # flap half (layouts.old_label), its paper as the substrate patch the new label bonds over.
            o = layouts.old_label(self.c.seed, carton_W=W)
            patches.append((M.SURFACES["paper_label"], 2, 1, (*o["u"], *o["v"])))
            self.task_geometry["old_label"] = o
            # its print is drawn by dressing.py as a sheet on the board (the new label must cover it)
        if task == "press-down":
            self.task_geometry["laid_label"] = layouts.laid_label(self.c.seed, carton_W=W)
        if task in ("smooth-corner", "finish-securing"):
            # task ladder: where the label was stuck down and which areas of it are still lifted
            # (layouts.partly_applied); the start state is built by ladder.lay_partly_applied
            self.task_geometry["laid_label"] = layouts.partly_applied(task, self.c.seed, carton_W=W)
        self.colliders.append(Collider("carton", "box", (L/2, W/2, H/2), M.SURFACES["kraft"], body=body,
                                       patches=tuple(patches)))
        self.items["carton"] = dict(body=body, size=(L, W, H), yaw=float(yaw))
        self.task_geometry.update(carton=dict(L=L, W=W, H=H, tape_half=.024))
        self._tools(.60, -.02)
        self._dressing_pack()

    def specimen(self):
        z = TABLE
        self.label_spec = self._presented("2x1", .004, self.printer(-.30, -.17, 2*IN, lip=0., hd=.095, ht=.15),
                                          h=self.c.tube_label_mesh_m)   # 2-inch direct-thermal class: smaller housing
        # bench-top tube rest (POM): two support rails the tube lies across, cap toward -x
        cx, cy = .10, -.02
        r, Lt = .0065, .100
        rail_top, rail_gap = z + .012, .004           # rail inner top edges at |y| = 4 mm
        for sgn in (-1, 1):
            self.box(f"tube_rest_rail_{sgn}", (cx, cy + sgn*(rail_gap+.003), rail_top - .004), (.03, .003, .004), "pom")
        self.box("tube_rest_base", (cx, cy, z + .004), (.035, .012, .004), "pom")
        tz = rail_top + math.sqrt(r*r - rail_gap**2) + 3e-4
        rot = self.wp.quat_from_axis_angle(self.wp.vec3(0, 1, 0), float(-math.pi/2))   # tube axis -> -x (cap at -x)
        # seeded roll about the tube's own axis (layouts.tube_roll; seed 0: manufacturer label up)
        roll = layouts.tube_roll(self.c.seed)
        if roll:
            from scipy.spatial.transform import Rotation
            rot = self.wp.quat(*map(float, (Rotation.from_rotvec([0., -math.pi/2, 0.])*Rotation.from_rotvec([0., 0., roll])).as_quat()))
        body = self.free_body("tube", (cx, cy, tz), rot)
        self.cyl_hull("tube_wall", (0, 0, 0), r, Lt/2, "pet_clear", body=body, mu=.4, density=1380.)
        self.cyl_hull("tube_cap", (0, 0, Lt/2+.010), .0081, .012, "cap_lavender", body=body, density=1100.)
        self.set_mass(body, .0105)       # PET tube + closure + dry fill (estimate)
        # manufacturer paper label (lot/expiry) covering 0..240 deg, 18..75 mm from the cap end
        man = (M.SURFACES["paper_label"], "cyl", None, (Lt/2-.075, Lt/2-.018, -2.1, 2.1))
        self.colliders.append(Collider("tube", "cylinder", (r, Lt/2), M.SURFACES["pet_tube"], body=body,
                                       patches=(man,)))
        self.colliders.append(Collider("tube_cap", "cylinder", (.0081, .012), M.SURFACES["pp"], body=body,
                                       local_pos=(0, 0, Lt/2+.010)))
        self.box("tube_mfr_label", (0, 0, 0), (.001, .001, .001), "mfr_label", body=body, collide=False)
        self.items["tube"] = dict(body=body, radius=r, length=Lt)
        self.task_geometry.update(tube=dict(r=r, L=Lt, mfr_label_top_from_cap_end=.018, mfr_label_bottom_from_cap_end=.075,
                                            mfr_label_half_angle=2.1, cradle=(cx, cy, tz)))
        self._tools(.60, -.02)
        self._dressing_lab()

    def part(self):
        z = TABLE
        self.label_spec = self._presented("2x4", .008, self.printer(-.30, -.17, 4*IN, lip=.012, kind="thermal_transfer"))   # PET 7818 is a thermal-transfer stock; short lip: a 2-in-long label must overhang ~25 mm for pickup
        if self.c.task == "part-id-match":
            # three stainless mounting plates from one delivery (the retired part-id-steel plate, unchanged: same size,
            # material, density, friction and label surface), each laser-marked with its part number (visual only:
            # annealed marking is flush with the surface); the label belongs to one of them (layouts.plates)
            lay = layouts.plates(self.c.seed)
            sx, sy, sz = layouts.PLATE_SIZE
            for k, ((x, y, yaw), number, mark) in enumerate(zip(lay["poses"], lay["part_numbers"], lay["markings"])):
                name = f"plate_{k + 1}"
                q = self.wp.quat_from_axis_angle(self.wp.vec3(0, 0, 1), float(yaw))
                body = self.free_body(name, (x, y, z + sz/2), q)
                self.box(f"{name}_steel", (0, 0, 0), (sx/2, sy/2, sz/2), "stainless", body=body, mu=.3, density=7900.)
                self.colliders.append(Collider(name, "box", (sx/2, sy/2, sz/2), M.SURFACES["steel"], body=body))
                self.items[name] = dict(body=body, kind="steel_plate", size=layouts.PLATE_SIZE, part_number=number,
                                        marking=mark)
            self.target = f"plate_{lay['target'] + 1}"
            self.task_geometry["plates"] = lay
        else:
            dx, dy = layouts.canister_offset(self.c.seed)
            body = self.free_body("part", (.10 + dx, -.08 + dy, z+.125))
            self.cyl_hull("jug_wall", (0, 0, 0), .075, .125, "hdpe_white", body=body, mu=.3, density=950.)
            self.set_mass(body, 3.2)      # 4 L round screw-lid HDPE canister with contents (authored)
            self.colliders.append(Collider("part", "cylinder", (.075, .125), M.SURFACES["hdpe"], body=body))
            self.items["part"] = dict(body=body, kind="hdpe_jug", radius=.075, height=.25)
        self._tools(.60, -.02)
        self._dressing_pack()

    # ------------------------------------------------------------------ tools and dressing
    def _tools(self, x, y):
        z = TABLE
        # tool tray (open ABS tray)
        self.box("tray_base", (x, y, z+.002), (.09, .07, .002), "tray")
        for s in (-1, 1):
            self.box(f"tray_wall_x_{s}", (x+s*.088, y, z+.012), (.002, .07, .012), "tray")
            self.box(f"tray_wall_y_{s}", (x, y+s*.068, z+.012), (.09, .002, .012), "tray")
        # PP squeegee card, 100 x 70 x 1.5 mm, resting on edge-support ribs so it can be pinched
        body = self.free_body("squeegee", (x, y-.02, z+.0145))
        self.box("squeegee_card", (0, 0, 0), (.05, .035, .00075), "squeegee_pp", body=body, mu=.3, density=905.)
        self.colliders.append(Collider("squeegee", "box", (.05, .035, .00075), M.SURFACES["pp_tool"], body=body, role="tool"))
        for s in (-1, 1):
            self.box(f"tray_rib_{s}", (x+s*.035, y-.02, z+.0085), (.004, .05, .0045), "tray")
        self.items["squeegee"] = dict(body=body)

    def _dressing_pack(self):
        """Pack-station dressing is visual only (dressing.py)."""

    def _dressing_lab(self):
        """Specimen bench: black epoxy top; the rest of the dressing is visual only (dressing.py)."""
        self.materials[self._find("bench_top")] = "epoxy_black"

    def _find(self, label):
        return self.b.shape_label.index(label)

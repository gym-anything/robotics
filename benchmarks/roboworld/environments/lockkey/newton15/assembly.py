"""The legacy box, hasp and spring-raised shackle around the pin mechanism.

The lock body is fixed, as in the baseline. A visible bolted mounting bracket
explains that boundary condition. Release remains an explicitly documented
angle-triggered spring model, not a claimed contact-resolved locking cam.
"""

import math
import numpy as np
import trimesh
from lockkey import lock_specs as LK
from .geometry import add_mesh, legacy_mesh, key_parts, set_mass
from .mechanism import build as mechanism
from .runtime import require_engine
from .robot import add_robot


def build(c, *, assisted_key=False, fixture_key=None):
    if fixture_key is not None and not assisted_key:
        raise ValueError("Key substitution belongs only to a labelled component fixture")
    selected_key = fixture_key or c.bitting
    newton, wp = require_engine()
    z = c.table_height
    origin = (.10, 0., z+.10)
    b, bodies, mechanism_joints = mechanism(c, key=selected_key, origin=origin,
        fixture=assisted_key, key_origin=origin if assisted_key else (.16, -.16, z+.055))
    materials = {i: "brass" for i in range(b.shape_count)}
    for i, body in enumerate(b.shape_body):
        if body == bodies["key"]:
            materials[i] = "nickel"
    cfg = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4, ke=c.contact_ke,
        kd=c.contact_kd, mu_torsional=0., mu_rolling=0.)
    tr = lambda pos: wp.transform(pos, wp.quat_identity())

    def box(name, pos, half, material="steel", body=-1):
        i = b.add_shape_box(body, xform=tr(pos), hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=name)
        materials[i] = material
        return i

    def cylinder(name, pos, radius, length, material="steel", body=-1, rotation=None):
        i = b.add_shape_cylinder(body, xform=wp.transform(pos, rotation or wp.quat_identity()),
            radius=radius, half_height=length/2, cfg=cfg, label=name)
        materials[i] = material

    def bow(body):
        # Preserve the existing solid grip disc rather than drawing a hole
        # through a collider. It is an unpunched simplified bow.
        bc = (c.face_x+c.insertion_travel+.012, 0., c.spine_z+LK.BLADE_HEIGHT/2+.012)
        head = trimesh.creation.cylinder(radius=.011, height=.005, sections=44)
        head.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
        neck = trimesh.creation.box(extents=(.008, .005, .010))
        neck.apply_translation((-.0063, 0., -.008))
        hull = trimesh.util.concatenate([head, neck]).convex_hull
        hull.apply_translation(bc)
        shape = add_mesh(b, body, hull, label="key_bow", cfg=cfg)
        materials[shape] = "nickel"

    bow(bodies["key"])
    # Preserve the legacy scene mass instead of quietly changing contact loads.
    for key, y, code in (("key", -.16, selected_key), ("wrong_key", .16, c.wrong_bitting)):
        parts = key_parts(c, code)
        if key == "wrong_key":
            body = bodies[key] = b.add_body(xform=tr((.16, y, z+.055)), label=key)
            for i, part in enumerate(parts):
                shape = add_mesh(b, body, part, label=f"wrong_blade_{i}", cfg=cfg)
                materials[shape] = "nickel"
            bow(body)
        set_mass(b, bodies[key], max(LK.RHO_BRASS*sum(p.volume for p in parts), .005))
        # Render the actual supports; no decorative plinth conceals their shape.
        bx = .16+c.face_x+c.insertion_travel+.012
        blade_tip = .16+c.face_x+c.insertion_travel-.030
        box(f"stand_base_{key}", (blade_tip+.030*.55, y, z+.005), (.0425, .015, .005), "robot_dark")
        floor = .055+c.spine_z-.0002
        for i, x in enumerate((blade_tip+.030*.18, blade_tip+.030*.72)):
            box(f"stand_post_{key}_{i}", (x, y, z+(.010+floor)/2), (.003, .006, (floor-.010)/2), "aluminum")
            for sign in (-1, 1):
                box(f"stand_fork_{key}_{i}_{sign}", (x, y+sign*.00335, z+floor+.004), (.003, .00125, .004), "aluminum")
        # The legacy neck support starts 5.5 mm inside its solid convex bow.
        # Put this fixture's contact surface below the ACTUAL neck underside,
        # retaining the key pose and the available open-top grasp approach.
        underside = .055+c.spine_z+LK.BLADE_HEIGHT/2+.012-.008-.005
        top = underside-.0002
        box(f"stand_neck_{key}", (bx-.003, y, z+(.010+top)/2), (.005, .0055, (top-.010)/2), "aluminum")

    box("bench_top", (.10, 0., z-.020), (.65, .46, .020), "wood")
    box("work_mat", (-.055, 0., z+.0008), (.325, .245, .0008), "robot_dark")
    # Open box: four side panels and a bottom, not a solid obstructing interior.
    center = -.102
    wall_height = .160 if c.box_geometry == "clear_lid" else .136
    lid_top = wall_height+.024
    box("box_bottom", (center, 0., z+.0075), (.15, .21, .0075), "paint_blue")
    for x in (center-.1425, center+.1425):
        box(f"box_end_{x}", (x, 0., z+wall_height/2), (.0075, .21, wall_height/2), "paint_blue")
    for y in (-.2025, .2025):
        box(f"box_side_{y}", (center, y, z+wall_height/2), (.15, .0075, wall_height/2), "paint_blue")
    hinge = (center-(.16 if c.box_geometry == "clear_lid" else .15), 0., z+lid_top)
    lid_center = (center, 0., z+wall_height+.012)
    lid = bodies["lid"] = b.add_link(xform=tr(lid_center), label="lid")
    box("lid", (0., 0., 0.), (.16, .22, .012), "paint_blue", lid)
    for y in (-.15, .15):
        box(f"lid_rib_{y}", (0., y, .014), (.14, .003, .002), "steel", lid)
    # Front slotted hasp attached to the lid: three lower bars plus side rails.
    slot_y, slot_z, slot_w, slot_h = -.010, origin[2]+.045, .030, .014
    plate_x = .048+.004+.0015
    rel = lambda p: tuple(np.asarray(p)-np.asarray(lid_center))
    above = z+lid_top-(slot_z+slot_h/2)
    box("hasp_top", rel((plate_x, slot_y, slot_z+slot_h/2+above/2)), (.0015, .025, above/2), "steel", lid)
    box("hasp_bottom", rel((plate_x, slot_y, slot_z-slot_h/2-.008)), (.0015, .025, .008), "steel", lid)
    for sy in (-1, 1):
        box(f"hasp_side_{sy}", rel((plate_x, slot_y+sy*.020, slot_z)), (.0015, .005, slot_h/2), "steel", lid)
    box("hasp_lid_bridge", rel((.052, slot_y, z+lid_top+.0015)), (.025, .025, .0015), "steel", lid)
    set_mass(b, lid, 1.8)
    j = b.add_joint_revolute(-1, lid, parent_xform=tr(hinge),
        child_xform=tr(tuple(np.asarray(hinge)-lid_center)), axis=newton.Axis.Y,
        limit_lower=math.radians(-95), limit_upper=math.radians(.5), label="lid_hinge")
    b.add_articulation([j], label="lid")
    mechanism_joints["lid"] = j
    # Staple fixed to the chest, matching the original ring dimensions.
    for y in (-.0195, -.0005):
        box(f"staple_side_{y}", ((.048+.1095)/2, y, slot_z), ((.1095-.048)/2, .002, .002))
    box("staple_front", (.1115, -.010, slot_z), (.002, .0095, .002))
    shackle = bodies["shackle"] = b.add_link(xform=tr(origin), label="shackle")
    # 6 mm collider legs; the arch is explicit collision geometry too.
    for y, length in ((-.010, .044), (.010, .030)):
        cylinder(f"shackle_leg_{y}", (-.0015, y, .060-length/2), .003, length, body=shackle)
    for i in range(16):
        a, h = (i+.5)*math.pi/16, math.pi/16
        position = (-.0015, .010*math.cos(a), .060+.010*math.sin(a))
        tangent = np.array([0., -math.sin(a), math.cos(a)])
        from scipy.spatial.transform import Rotation
        rot = Rotation.align_vectors([tangent], [[0., 0., 1.]])[0].as_quat()
        cylinder(f"shackle_arch_{i}", position, .003, .010*h+.0002, body=shackle, rotation=wp.quat(*rot))
    set_mass(b, shackle, .030)
    from .shackle import add_guide
    j = add_guide(b, c, shackle, origin, newton, wp)
    b.add_articulation([j], label="shackle")
    mechanism_joints["shackle"] = j
    # Keep the original cover for baseline comparisons; the corrected candidate
    # opens its keyway and gives the same rendered casing collision geometry.
    casing, _ = legacy_mesh().make_padlock_body(c.plug_length, c.chambers_x,
        LK.PIN_R+c.bore_clearance, (2*c.keyway_half_width-c.blade_clearance+.0003)/2,
        c.spine_z+LK.BLADE_HEIGHT/2, LK.BLADE_HEIGHT+.001,
        align_shackle_bores=c.shackle_swivel)
    if c.aligned_casing:
        # Legacy rounded-box buffering extends its face beyond BOTH bore/slot
        # cutters. Trim to the existing key-insertion face; otherwise the
        # visible keyway is capped and the bow enters a visual-only gold wall.
        low, high = casing.bounds.copy()
        high[0] = c.face_x
        clip = trimesh.creation.box(extents=high-low+np.array([0., .002, .002]))
        clip.apply_translation((low+high)/2)
        casing = casing.intersection(clip)
    cover = (newton.ModelBuilder.ShapeConfig(density=0., mu=c.brass_friction,
             gap=c.contact_gap, has_shape_collision=True) if c.aligned_casing else
             newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False))
    shape = add_mesh(b, -1, casing, label="lock_casing", cfg=cover, transform=tr(origin), convex=False)
    materials[shape] = "brass"
    box("lock_mount_back", (.068, 0., origin[2]+.018), (.002, .026, .034))
    for y in (-.024, .024):
        box(f"lock_mount_jaw_{y}", (.081, y, origin[2]+.009), (.011, .004, .020))
    addresses = add_robot(b, c)
    from .scene import dress
    dress(b, c, materials)
    return b, bodies, addresses, mechanism_joints, materials

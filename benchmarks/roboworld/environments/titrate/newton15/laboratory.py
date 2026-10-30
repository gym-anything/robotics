"""Shared physics/appearance apparatus assembly; no rendering physics engine."""

from dataclasses import dataclass, field
import numpy as np
from .apparatus import Apparatus
from .geometry import revolve, tube, profile_wall_cells, record_meridian
from .lab_config import LabConfig
from .runtime import require_newton


@dataclass
class Laboratory:
    config: LabConfig
    builder: object
    apparatus: Apparatus
    materials: dict = field(default_factory=dict)
    robot_joints: dict = field(default_factory=dict)

    def box(self, name, pos, size, material, body=-1, density=0., mu=.4, quaternion=(0., 0., 0., 1.)):
        _, wp = require_newton()
        shape = self.builder.add_shape_box(body, xform=wp.transform(pos, wp.quat(*quaternion)),
            hx=size[0]/2, hy=size[1]/2, hz=size[2]/2,
            cfg=self.builder.ShapeConfig(density=density, mu=mu), label=name)
        self.materials[shape] = material
        return shape

    def cylinder(self, name, pos, radius, height, material, body=-1, density=0., axis=2):
        _, wp = require_newton()
        rotation = wp.quat_identity() if axis == 2 else wp.quat_from_axis_angle(
            wp.vec3(1., 0., 0.) if axis == 1 else wp.vec3(0., 1., 0.), np.pi/2)
        shape = self.builder.add_shape_cylinder(body, xform=wp.transform(pos, rotation),
            radius=radius, half_height=height/2,
            cfg=self.builder.ShapeConfig(density=density, mu=.25), label=name)
        self.materials[shape] = material
        return shape

    def hollow(self, name, pos, profile, material="glass", movable=True, quaternion=(0., 0., 0., 1.), articulated=False):
        newton, wp = require_newton()
        b = self.builder
        if articulated and not movable:
            raise ValueError('An articulated vessel must be movable')
        add = b.add_link if articulated else b.add_body
        body = add(xform=wp.transform(pos, wp.quat(*quaternion)), label=name) if movable else -1
        local = (0., 0., 0.) if movable else pos
        mesh = revolve(profile)
        shape = b.add_shape_mesh(body, xform=wp.transform(local, wp.quat_identity()),
            mesh=newton.Mesh(mesh.vertices, mesh.triangles.flatten()),
            cfg=b.ShapeConfig(density=2230. if movable else 0., mu=.25, has_shape_collision=False), label=name)
        record_meridian(b, shape, mesh)
        self.materials[shape] = material
        for i, cell in enumerate(profile_wall_cells(profile)):
            b.add_shape_convex_hull(body, xform=wp.transform(local, wp.quat_identity()),
                mesh=newton.Mesh(cell.vertices, cell.triangles.flatten()),
                cfg=b.ShapeConfig(density=0., mu=.25, is_visible=False, has_particle_collision=False),
                label=f"{name}_contact_{i}")
        if movable:
            self.apparatus.bodies[name] = body
        return body


def build_laboratory(config=None):
    from .rigid_contact import register, configure

    c = config or LabConfig()
    newton, wp = require_newton()
    b = newton.ModelBuilder()
    register(b)
    b.default_shape_cfg.gap = .0001
    lab = Laboratory(c, b, Apparatus(b))
    z = c.bench_height
    lab.box("bench_epoxy", (0., 0., z-.015), (c.bench_width, c.bench_depth, .03), "epoxy")
    from .room import build_room
    build_room(lab)
    for x in (-.73, .73):
        lab.box(f"cabinet_plinth_{x}", (x, .04, .014), (.28, .55, .028), "rubber")
        lab.box(f"cabinet_{x}", (x, .04, .42), (.30, .58, .81), "cabinet")
        for h in (.16, .42, .68):
            lab.box(f"drawer_{x}_{h}", (x, -.258, h), (.281, .018, .235), "cabinet")
            lab.cylinder(f"handle_{x}_{h}", (x, -.280, h+.06), .006, .15, "steel", axis=0)
    x, y, fz = c.flask_position
    # Manufacturer mass and outer envelope; the inertia is a declared uniform-
    # envelope reconstruction. No bench weld: all loads go through four feet.
    mass, dimensions = 2.716, np.array([.150, .260, .105])
    diagonal = mass/12*(np.sum(dimensions**2)-dimensions**2)
    stirrer = b.add_link(xform=wp.transform((x, y, z+.0002), wp.quat_identity()),
        mass=mass, com=(0., -.055, .0525), inertia=wp.mat33(np.diag(diagonal)),
        lock_inertia=True, label="stirrer")
    stirrer_free = b.add_joint_free(child=stirrer, label="stirrer_free")
    lab.apparatus.bodies["stirrer"] = stirrer
    lab.box("stirrer_housing", (0., -.055, .0525), (.150, .260, .091), "instrument_blue", body=stirrer)
    lab.box("ceramic_plate", (0., 0., .1015), (.100, .100, .007), "ceramic", body=stirrer)
    lab.box("stirrer_control_panel", (0., -.1855, .046), (.137, .002, .072), "cabinet", body=stirrer)
    for dx in (-.060, .060):
        for dy in (-.15, .04):
            lab.cylinder(f"rubber_foot_{dx}_{dy}", (dx, dy, .004), .008, .008, "rubber", body=stirrer)
    knob_local = (0., -.195, .038)
    knob = b.add_link(xform=wp.transform(np.array([x, y, z+.0002])+knob_local, wp.quat_identity()),
                      label="stirrer_knob")
    lab.cylinder("stirrer_speed_knob", (0., 0., 0.), .013, .015, "robot_dark", body=knob, density=1100., axis=1)
    lab.box("knob_index", (0., -.0076, .007), (.001, .0004, .006), "ptfe", body=knob)
    knob_joint = b.add_joint_revolute(stirrer, knob, parent_xform=wp.transform(knob_local, wp.quat_identity()),
        axis=(0., 1., 0.), limit_lower=0., limit_upper=5*np.pi/3, target_ke=0., target_kd=0.,
        damping=.0001, friction=.001, label="stirrer_speed_dial")
    b.add_articulation([stirrer_free, knob_joint], label="stirrer_with_dial")
    lab.apparatus.bodies["stirrer_knob"] = knob
    lab.apparatus.joints["stirrer_knob"] = knob_joint
    start = b.shape_count
    lab.apparatus.add_flask(position=c.flask_position)
    lab.materials[start] = "glass"
    start = b.shape_count
    # Start the free 3 mm-radius bar resting on the 2 mm inner flask floor.
    # The previous 0.5 mm hover gap could be populated by initial liquid
    # parcels and then crushed as the bar settled. No constraint or pose
    # correction is applied during stepping; Newton still owns bar motion.
    lab.apparatus.add_stir_bar(position=(x, y, fz+.002+.003))
    lab.materials[start] = "ptfe"
    # A clamped vertical burette, offset stand and two rubber-lined clamps.
    # Stand base clears the stirrer housing by 30 mm; do not start two
    # separate solids interpenetrating under the edge of the hotplate.
    sx, sy = x-.20, y+.14
    lab.box("stand_base", (sx, sy, z+.007), (.19, .18, .014), "iron")
    lab.cylinder("stand_rod", (sx, sy, z+.52), .006, 1.04, "steel")
    vz = c.valve_position[2]
    for h in (vz+.18, vz+.52):
        lab.cylinder(f"clamp_shaft_{h}", ((sx+x)/2, sy, h), .004, x-sx, "steel", axis=0)
        # The crosspiece terminates BEHIND the glass. Ending it at the tube's
        # centreline filled half of the liquid bore for 12 mm at each clamp.
        # Its side jaws meet this backing outside the 14 mm tube envelope.
        backing_front = y+c.burette_outer_radius+.003
        lab.box(f"clamp_back_{h}", (x, (sy+backing_front)/2, h),
            (.014, sy-backing_front, .012), "steel")
        for sign in (-1, 1):
            lab.box(f"clamp_jaw_{h}_{sign}", (x+sign*.009, y, h), (.004, .025, .010), "rubber")
    # Geometry in the valve frame; straight tube connected by drawn shoulder.
    ri, ro = c.burette_inner_radius, c.burette_outer_radius
    lab.hollow("burette_column", c.valve_position,
        [(c.valve_bore_radius, .008), (.003, .008), (ro, .038), (ro, c.burette_height-.055),
         (ri, c.burette_height-.055), (ri, .040), (c.valve_bore_radius, .014)], movable=False)
    lab.hollow("burette_tip", c.valve_position,
        [(c.tip_inner_radius, -.055), (c.tip_outer_radius, -.055), (.003, -.013),
         (.003, -.008), (c.valve_bore_radius, -.008), (c.valve_bore_radius, -.013)], movable=False)
    from .bored_geometry import stopcock_plug, plug_cells

    socket_body = b.add_link(xform=wp.transform(c.valve_position, wp.quat_identity()), label="stopcock_socket")
    socket_fixed = b.add_joint_fixed(-1, socket_body,
        parent_xform=wp.transform(c.valve_position, wp.quat_identity()), label="clamped_socket")
    plug = b.add_link(xform=wp.transform(c.valve_position, wp.quat_identity()), label="stopcock")
    physical_mesh = stopcock_plug(c.valve_plug_radius, c.valve_plug_half_length, c.valve_bore_radius)
    shape = b.add_shape_mesh(plug, mesh=newton.Mesh(physical_mesh.vertices, physical_mesh.triangles.flatten()),
        cfg=b.ShapeConfig(density=2200., mu=.10, has_shape_collision=False), label="bored_ptfe_plug")
    from .analytic_solids import register as register_analytic
    register_analytic(b, shape, 1, (c.valve_plug_radius, 0., c.valve_plug_half_length, c.valve_bore_radius))
    lab.materials[shape] = "ptfe"
    for i, cell in enumerate(plug_cells(c.valve_plug_radius, c.valve_plug_half_length, c.valve_bore_radius)):
        b.add_shape_convex_hull(plug, mesh=newton.Mesh(cell.vertices, cell.triangles.flatten()),
            cfg=b.ShapeConfig(density=0., mu=.10, is_visible=False, has_particle_collision=False), label=f"plug_contact_{i}")
    lab.box("valve_handle", (0., -.020, 0.), (.043, .007, .009), "ptfe", body=plug, density=2200.)
    joint = b.add_joint_revolute(socket_body, plug,
        axis=(0., 1., 0.), limit_lower=0., limit_upper=np.pi/2, target_ke=0., target_kd=0.,
        damping=.0001, friction=.002, label="stopcock_rotation")
    b.add_articulation([socket_fixed, joint], label="stopcock_bearing")
    b.joint_q[b.joint_q_start[joint]] = np.pi/2  # Initially CLOSED; real bore rotated sideways.
    lab.apparatus.bodies["stopcock"] = plug
    lab.apparatus.joints["stopcock"] = joint
    # Annular socket along Y leaves the real rotating plug cavity open.
    # The socket's two Z ports are represented by two separated half shells;
    # a continuous unbored annulus would incorrectly block the vertical liquid path.
    from .bored_geometry import stopcock_socket, socket_cells

    socket = stopcock_socket(inner=c.valve_plug_radius, bore=c.valve_bore_radius)
    shape = b.add_shape_mesh(socket_body,
        mesh=newton.Mesh(socket.vertices, socket.triangles.flatten()),
        cfg=b.ShapeConfig(density=2230., mu=.1, has_shape_collision=False), label="ported_glass_socket")
    register_analytic(b, shape, 2, (c.valve_plug_radius, .010, .010, c.valve_bore_radius))
    lab.materials[shape] = "glass"
    for i, cell in enumerate(socket_cells(inner=c.valve_plug_radius, bore=c.valve_bore_radius)):
        b.add_shape_convex_hull(socket_body,
            mesh=newton.Mesh(cell.vertices, cell.triangles.flatten()),
            cfg=b.ShapeConfig(density=0., mu=.1, is_visible=False, has_particle_collision=False), label=f"socket_contact_{i}")
        # The revolute bearing already constrains the plug's five supported
        # DOFs and supplies explicit rotational friction. Do not add thousands
        # of redundant zero-clearance facet contacts inside that same bearing.
        # Neither shape is removed: robot contacts and fluid ports remain.
        # The parent/child bearing filter expresses this as a whole-body pair.
        # This survives MuJoCo's collision-mask compilation for large models.
    # Reagent and rinse vessels are hollow, independently movable rigid bodies.
    bottle = [(0., 0.), (.027, 0.), (.029, .003), (.029, .095), (.017, .115),
              (.014, .118), (.014, .140), (.012, .140), (.012, .118), (.015, .115),
              (.027, .094), (.027, .003), (0., .003)]
    lab.hollow("acid_bottle", (.47, .12, z+.0002), bottle)
    lab.hollow("base_bottle", (.61, .12, z+.0002), bottle)
    lab.hollow("rinse_beaker", (.41, -.16, z+.0002),
        [(0., 0.), (.029, 0.), (.030, .002), (.030, .082), (.0285, .082), (.0285, .002), (0., .002)])
    # Physical pipette, resting across a two-saddle rack. Bulb volume is geometry,
    # not an implicit reagent source. The pneumatic operating model is separate.
    lab.box("pipette_rack", (.52, -.25, z+.007), (.49, .07, .014), "polypropylene")
    from .pipette_geometry import profile as pipette_profile

    pipette = lab.hollow("volumetric_pipette", (.275, -.25, z+.0334), pipette_profile(),
               quaternion=tuple(wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), np.pi/2)), articulated=True)
    from .pipette_filler import add_filler
    add_filler(lab, pipette)
    # Third saddle supports the heavier manual filler rather than allowing
    # its rearward centre of mass to tip the long tool off a stem-only rack.
    lab.box('filler_saddle_floor', (.746, -.25, z+.01565), (.070, .038, .0033), 'polypropylene')
    for side in (-1, 1):
        lab.box(f'filler_saddle_side_{side}', (.746, -.25+side*.0185, z+.0233),
                (.070, .003, .012), 'polypropylene')
    # Open U saddles support the stems; the wider bulb remains above the rack.
    for sx in (.330, .660):
        lab.box(f"pipette_saddle_floor_{sx}", (sx, -.25, z+.027), (.012, .018, .007), "polypropylene")
        for side in (-1, 1):
            lab.box(f"pipette_saddle_side_{sx}_{side}", (sx, -.25+side*.007, z+.035), (.012, .005, .012), "polypropylene")
    if c.robot:
        from .panda import add_panda

        lab.box("robot_mount", (-.43, -.04, z+.0125), (.19, .19, .025), "steel")
        for dx in (-.065, .065):
            for dy in (-.065, .065):
                lab.cylinder(f"mount_bolt_{dx}_{dy}", (-.43+dx, -.04+dy, z+.027), .005, .009, "steel")
        lab.robot_joints = add_panda(b, (-.43, -.04, z+.025))
    # External RGB camera on a floor-mounted stand, not an implied Panda sensor.
    from .sensors import camera_frame

    frame = camera_frame(c.camera_eye, c.camera_target)
    eye = np.asarray(c.camera_eye)
    back = frame.apply([0., 0., 1.])
    rotation = tuple(frame.as_quat())
    # The former head cylinder contained the RGB optical origin (its top
    # reached z=1.6505 while the optical center was at z=1.6500). Mount the
    # stand/head behind the lens, as a real camera bracket does. The camera
    # position, projection and task framing remain unchanged.
    stand = eye+.050*back
    head = eye+.030*back
    lab.box("camera_stand_base", (stand[0], stand[1], .0125), (.24, .24, .025), "iron")
    lab.cylinder("camera_stand_post", (stand[0], stand[1], .82), .009, 1.615, "steel")
    lab.cylinder("camera_mount_ball", (head[0], head[1], 1.632), .011, .025, "robot_dark")
    lab.box("rgb_camera_housing", eye+.020*back, (.090, .025, .025), "robot_dark", quaternion=rotation)
    for name, distance, radius, depth, material in (
            ("rgb_lens_barrel", .006, .006, .011, "robot_dark"),
            ("rgb_front_glass", .0005, .0053, .001, "glass")):
        shape = b.add_shape_cylinder(-1, xform=wp.transform(eye+distance*back, wp.quat(*rotation)),
            radius=radius, half_height=depth/2, cfg=b.ShapeConfig(density=0.), label=name)
        lab.materials[shape] = material
    model = b.finalize(device=c.device)
    configure(model)
    model.set_gravity((0., 0., -9.80665))
    return lab, model

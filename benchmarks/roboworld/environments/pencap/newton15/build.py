"""Build the preserved dual-Panda scene; optional appearance upgrades are separate."""

import math
import xml.etree.ElementTree as ET
import numpy as np
from .geometry import add_cap, collision_mesh, pen_profile, set_mass
from .runtime import require_engine


def robot(builder, arm, position, yaw):
    # Reuse the repository's hash-verified manufacturer asset, not its fluid or
    # robot controller. The common source is Isaac's franka_description 0.7.1.
    from robots.backends.newton import add_robot as build, mount
    return build(builder,'panda','pencap',instance=arm,placement=mount(position,yaw)).metadata['legacy_addresses']


def build(c):
    newton, wp = require_engine()
    b = newton.ModelBuilder(gravity=(0., 0., -9.81))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.gap = c.contact_gap
    materials = {}
    def box(name, position, half, material, mu=.35):
        i = b.add_shape_box(-1, xform=wp.transform(position, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], label=name,
            cfg=newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=mu, ke=1e7, kd=5000.,
                mu_torsional=0., mu_rolling=0.))
        materials[i] = material
        return i
    box("floor", (0., 0., -.01), (3., 3., .01), "floor")
    box("table", (0., 0., c.table_top-.015), (.55, .45, .015), "wood")
    pp = newton.ModelBuilder.ShapeConfig(mu=c.plastic_friction, gap=c.contact_gap,
        ke=1e7, kd=5000., mu_torsional=0., mu_rolling=0.)
    pen = b.add_body(label="pen", xform=wp.transform((-.07, 0., c.table_top+.0075),
        wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), -math.pi/2)))
    i = b.add_shape_mesh(pen, mesh=collision_mesh(pen_profile(c), c, sdf=False), cfg=pp,
        color=(.95, .60, .05), label="pen_shell")
    materials[i] = "pen"
    set_mass(b, pen, c.pen_mass)
    first_cap = b.shape_count
    # Task setup: the cap lies beside the pen, parallel, mouth toward the same end as the writing tip,
    # as set down after uncapping.
    cap = add_cap(b, c, wp.transform((-.080, .070, c.table_top+c.cap_radius+.0015),
        wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), -math.pi/2)))
    for i in range(first_cap, b.shape_count):
        materials[i] = "cap"
    joints = {}
    for arm, pos, yaw in (("left", (0., .55, c.table_top), -math.pi/2),
                           ("right", (0., -.55, c.table_top), math.pi/2)):
        joints.update(robot(b, arm, pos, yaw))
    bodies = {"pen": pen, "cap": cap}
    if c.scene == "studio":
        from .scene import studio
        bodies.update(studio(b, c, materials))
    return b, bodies, joints, materials

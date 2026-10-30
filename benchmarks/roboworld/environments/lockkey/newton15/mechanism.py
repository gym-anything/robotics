"""Five pin stacks with contact-transmitted key torque, native Newton dynamics.

Guide joints represent cylindrical bores. No pin position is latched after
insertion, no bitting comparison authorizes rotation, and no collider is disabled
during turning. The optional cylindrical hand is an explicitly assisted fixture.
"""

import math
import numpy as np
from lockkey import lock_specs as LK
from .geometry import legacy_mesh, add_mesh, set_mass, key_parts
from .runtime import require_engine


def build(c, *, key="31542", origin=(0., 0., 0.), fixture=True, key_origin=None):
    newton, wp = require_engine()
    b = newton.ModelBuilder(gravity=(0., 0., -9.81))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    bodies, joints = {}, {}
    cfg = newton.ModelBuilder.ShapeConfig(density=LK.RHO_BRASS, mu=c.brass_friction,
        ke=c.contact_ke, kd=c.contact_kd, gap=c.contact_gap, mu_torsional=0., mu_rolling=0.)
    tr = lambda pos: wp.transform(pos, wp.quat_identity())
    world = lambda pos: tuple(np.asarray(origin) + np.asarray(pos))
    lm = legacy_mesh()
    bore = LK.PIN_R + c.bore_clearance
    shell = lm.make_shell(c.plug_length+.004, c.chambers_x, bore, c.plug_clearance)
    add_mesh(b, -1, shell, label="shell", cfg=cfg, transform=tr(origin), convex=False)
    plug = bodies["plug"] = b.add_link(xform=tr(origin), label="plug")
    pieces, visual = lm.make_plug_ledgekeyway_convex(c.plug_length, c.chambers_x,
        bore, c.keyway_half_width, c.ledge_z, c.keyway_top)
    if c.plug_geometry == "complete":
        from .geometry import complete_plug
        solid = complete_plug(c)
        add_mesh(b, plug, solid, label="plug_complete", cfg=cfg, convex=False)
    else:
        for i, part in enumerate(pieces):
            add_mesh(b, plug, part, label=f"plug_piece_{i}", cfg=cfg)
    decorative = newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False)
    if c.plug_geometry == "legacy":
        for i, part in enumerate(visual):
            add_mesh(b, plug, part, label=f"plug_visual_{i}", cfg=decorative)
    set_mass(b, plug, LK.RHO_BRASS*sum(p.volume for p in pieces+visual))
    jp = b.add_joint_revolute(-1, plug, parent_xform=tr(origin), axis=newton.Axis.X,
        limit_lower=math.radians(-.5), limit_upper=math.pi/2,
        limit_ke=1e7, limit_kd=500., label="plug_turn",
        target_ke=(.05/math.radians(1)) if fixture else 0.,
        target_kd=(.005/math.radians(1)) if fixture else 0., effort_limit=.3)
    joints["plug"] = jp
    tree = [jp]
    for driver in (False, True):
        for i, depth in enumerate(c.bitting):
            length = LK.KEY_PIN_LEN[int(depth)]
            kpz = c.keyway_top + length/2 + .0003
            size = LK.DRIVER_LEN if driver else length
            cz = kpz+length/2+LK.DRIVER_LEN/2+.0002 if driver else kpz
            name = f"{'driver' if driver else 'keypin'}_{i}"
            local = (c.chambers_x[i], 0., cz)
            body = bodies[name] = b.add_link(xform=tr(world(local)), label=name)
            tm = lm.make_pin_mesh(size, tip_top=0. if driver else c.pin_chamfer,
                tip_bot=c.pin_chamfer if driver else c.bottom_pin_tip)
            add_mesh(b, body, tm, label=name, cfg=cfg)
            set_mass(b, body, LK.pin_mass(size))
            kw = dict(parent_xform=tr(world(local) if driver else local), axis=newton.Axis.Z,
                      label=name, damping=LK.pin_mass(size)*.5)
            if driver:
                j = b.add_joint_prismatic(-1, body, target_ke=c.spring_k, target_kd=c.spring_d,
                    target_pos=c.spring_target, **kw)
                b.add_articulation([j], label=name)
            else:
                j = b.add_joint_prismatic(plug, body,
                    limit_lower=c.ledge_z-c.keyway_top-.0003, limit_upper=.006,
                    limit_ke=1e7, limit_kd=500., **kw)
                tree.append(j)
            joints[name] = j
        if not driver:
            b.add_articulation(tree, label="plug_and_keypins")
    blade = bodies["key"] = (b.add_link if fixture else b.add_body)(
        xform=tr(origin if key_origin is None else key_origin), label="key")
    parts = key_parts(c, key)
    for i, part in enumerate(parts):
        add_mesh(b, blade, part, label=f"key_blade_{i}", cfg=cfg, color=(.65, .66, .59))
    set_mass(b, blade, max(LK.RHO_BRASS*sum(p.volume for p in parts), .0001))
    if fixture:
        linear = newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X,
            limit_lower=-c.insertion_travel-.001, limit_upper=.001, limit_ke=1e7, limit_kd=500.,
            target_ke=5e4, target_kd=200., effort_limit=26.)
        angular = newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X,
            limit_lower=math.radians(-100), limit_upper=math.radians(100), limit_ke=1e7, limit_kd=500.,
            target_ke=.02/math.radians(1), target_kd=.002/math.radians(1), effort_limit=.3,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY)
        j = b.add_joint_d6(-1, blade, parent_xform=tr(origin), linear_axes=[linear],
            angular_axes=[angular], label="assisted_cylindrical_hand")
        joints["hand"] = j
        b.add_articulation([j], label="fixture_hand")
    return b, bodies, joints

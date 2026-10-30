"""Captive sliding/swivelling shackle, with obstruction supplied by contacts.

The long-leg bore is represented by an ideal cylindrical bearing. Rotation is
passive, not an unlock animation or an angular servo. The existing casing and
short-leg colliders prevent rotation while the short leg is still seated.
"""

import numpy as np
from scipy.spatial.transform import Rotation


LONG_AXIS = np.array([-.0015, -.010, 0.])
SHORT_TIP = np.array([-.0015, .010, .030])
CASING_TOP = .040
STAPLE_TOP = .047
LEG_RADIUS = .003
BORE_RADIUS = .0032


def add_guide(b, c, body, origin, newton, wp):
    tr = lambda xyz: wp.transform(tuple(xyz), wp.quat_identity())
    linear = dict(axis=newton.Axis.Z, limit_lower=-.0005,
        limit_upper=0. if c.mechanical_latch else c.shackle_travel,
        target_ke=400., target_kd=8., effort_limit=20.)
    if not c.shackle_swivel:
        return b.add_joint_prismatic(-1, body, parent_xform=tr(origin),
            label="shackle_rail", **linear)
    return b.add_joint_d6(-1, body,
        parent_xform=tr(np.asarray(origin)+LONG_AXIS), child_xform=tr(LONG_AXIS),
        linear_axes=[newton.ModelBuilder.JointDofConfig(**linear)],
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z,
            damping=.0002)], label="shackle_slide_swivel", collision_filter_parent=False)


def geometry_state(body_pose, origin):
    """World geometry witnesses, not the latch flag or legacy reward threshold."""
    p = np.asarray(body_pose)
    origin = np.asarray(origin)
    rotation = Rotation.from_quat(p[3:])
    tip = p[:3]+rotation.apply(SHORT_TIP)
    retained_axis = p[:3]+rotation.apply(LONG_AXIS)
    xy = float(np.linalg.norm(tip[:2]-(origin+SHORT_TIP)[:2]))
    clearance = float(tip[2]-origin[2]-CASING_TOP)
    return {"shackle_short_tip_m": tip.tolist(),
        "shackle_short_leg_clearance_m": clearance,
        "shackle_staple_clearance_m": float(tip[2]-origin[2]-STAPLE_TOP),
        "shackle_short_leg_lateral_offset_m": xy,
        "shackle_axis_error_m": float(np.linalg.norm(retained_axis[:2]-(origin+LONG_AXIS)[:2])),
        "shackle_open": clearance > 0. and xy > LEG_RADIUS+BORE_RADIUS}


def contact_witnesses(env):
    """Current native shackle contacts; points belong to the latest step query."""
    contacts = env.contacts
    n = int(contacts.rigid_contact_count.numpy()[0])
    fields = {k: getattr(contacts, "rigid_contact_"+k).numpy()[:n]
              for k in ("shape0", "shape1", "point0", "point1", "normal")}
    poses = env.state.body_q.numpy()
    rows = []
    for i, (a,b) in enumerate(zip(fields["shape0"], fields["shape1"])):
        names = [env.builder.shape_label[int(s)] for s in (a,b)]
        if not any(name.startswith("shackle") for name in names):
            continue
        points = []
        for shape, field in ((a,"point0"), (b,"point1")):
            point = fields[field][i]
            body = env.builder.shape_body[int(shape)]
            points.append(point if body < 0 else poses[body,:3]+Rotation.from_quat(poses[body,3:]).apply(point))
        rows.append({"pair": names, "signed_distance_m": float(np.dot(points[1]-points[0],fields["normal"][i]))})
    return rows

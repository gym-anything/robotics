"""Workshop furniture and mounted external RGB camera, outside the lock core."""

import numpy as np
from scipy.spatial.transform import Rotation
from .runtime import require_engine

CAMERA_EYE = (.95, -.70, 1.30)
CAMERA_TARGET = (.12, 0., .86)


def dress(b, c, materials):
    newton, wp = require_engine()
    cfg = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4, ke=c.contact_ke,
        kd=c.contact_kd, mu_torsional=0., mu_rolling=0.)

    def box(name, pos, half, material="steel", quat=(0., 0., 0., 1.)):
        i = b.add_shape_box(-1, xform=wp.transform(pos, quat), hx=half[0], hy=half[1],
            hz=half[2], label=name, cfg=cfg)
        materials[i] = material

    def rod(name, a, end, radius):
        a, end = np.asarray(a), np.asarray(end)
        delta = end-a
        q = Rotation.align_vectors([delta/np.linalg.norm(delta)], [[0., 0., 1.]])[0].as_quat()
        i = b.add_shape_cylinder(-1, xform=wp.transform((a+end)/2, q), radius=radius,
            half_height=float(np.linalg.norm(delta)/2), cfg=cfg, label=name)
        materials[i] = "steel"

    z = c.table_height
    box("floor", (0., 0., -.025), (2.5, 2.5, .025), "floor")
    for x in (-.47, .67):
        for y in (-.38, .38):
            box(f"bench_leg_{x}_{y}", (x, y, (z-.04)/2), (.025, .025, (z-.04)/2))
            box(f"foot_{x}_{y}", (x, y, .015), (.033, .033, .015), "robot_dark")
        box(f"bench_crossrail_{x}", (x, 0., z-.09), (.025, .4, .025))
    for y in (-.38, .38):
        box(f"bench_rail_{y}", (.1, y, z-.075), (.57, .025, .035))
    # The original Panda's base is 110 mm above the desk: a bolted riser
    # supports that exact pose, instead of leaving the robot floating.
    box("panda_riser", (.54, 0., z+.05), (.065, .065, .05))
    box("panda_mount_plate", (.54, 0., z+.105), (.10, .10, .005), "aluminum")
    for dx in (-.08, .08):
        for dy in (-.08, .08):
            rod(f"mount_bolt_{dx}_{dy}", (.54+dx, dy, z+.11), (.54+dx, dy, z+.115), .005)
    box("wall_back", (-1.3, 0., 1.4), (.04, 2.2, 1.4), "wall")
    box("wall_side", (.2, 1.5, 1.4), (1.5, .04, 1.4), "wall")
    box("skirting", (-1.25, 0., .055), (.008, 2.2, .055), "robot_dark")
    box("cabinet", (-.92, .55, .40), (.22, .55, .40), "cabinet")
    box("cabinet_counter", (-.92, .55, .825), (.24, .57, .025), "wood")
    for h in (.18, .43, .68):
        box(f"drawer_{h}", (-.695, .55, h), (.009, .52, .115), "paint_blue")
        box(f"handle_{h}", (-.675, .55, h+.025), (.012, .13, .01), "aluminum")
    box("pegboard", (-1.23, .1, 1.49), (.014, .70, .37), "wood")
    for h in (1.20, 1.70):
        box(f"pegboard_rail_{h}", (-1.20, .1, h), (.01, .66, .012))
    box("controller_shelf", (.47, 0., .23), (.18, .30, .014))
    box("controller", (.47, 0., .30), (.16, .22, .055), "robot_dark")
    for y in np.linspace(-.17, .17, 12):
        box(f"controller_vent_{y}", (.631, y, .30), (.001, .004, .035))
    eye, target = np.asarray(CAMERA_EYE), np.asarray(CAMERA_TARGET)
    back = (eye-target)/np.linalg.norm(eye-target)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    q = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
    box("camera_enclosure", eye+.013*back, (.045, .0125, .0125), "aluminum", q)
    box("camera_face", eye+.0003*back, (.043, .011, .0003), "robot_dark", q)
    mast = (1.02, -.77, 1.37)
    box("camera_stand_base", (1.02, -.77, .02), (.12, .12, .02))
    rod("camera_mast", (1.02, -.77, .04), mast, .012)
    rod("camera_mount", mast, eye+.03*back, .006)
    for i, p in enumerate(((0., 0., 2.60), (-.6, .7, 2.35))):
        box(f"light_fixture_{i}", p, (.31, .61, .025), "robot_white")

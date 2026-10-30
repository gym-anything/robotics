"""Workshop dressing and physical supports, independent of task mechanics.

Dimensions below are authored ordinary workshop furniture, not product claims.
The external camera enclosure follows the D435's published 90 x 25 x 25 mm.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from .runtime import require_engine

CAMERA_EYE = (.35, -.90, 1.50)
CAMERA_TARGET = (0., 0., .76)


def studio(b, c, materials):
    newton, wp = require_engine()
    cfg = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4, ke=1e7, kd=5000.,
        mu_torsional=0., mu_rolling=0.)

    def box(name, pos, half, material, *, body=-1, quat=(0., 0., 0., 1.), contact=None):
        i = b.add_shape_box(body, xform=wp.transform(pos, quat), hx=half[0], hy=half[1],
            hz=half[2], label=name, cfg=contact or cfg)
        materials[i] = material
        return i

    def rod(name, a, z, radius=.008, material="steel"):
        a, z = np.asarray(a), np.asarray(z)
        d = z-a
        rotation = Rotation.align_vectors([d/np.linalg.norm(d)], [[0., 0., 1.]])[0].as_quat()
        i = b.add_shape_cylinder(-1, xform=wp.transform((a+z)/2, rotation),
            radius=radius, half_height=float(np.linalg.norm(d)/2), label=name, cfg=cfg)
        materials[i] = material

    # Robot bases keep their baseline poses. Separate floor-bolted pedestals
    # support them; the narrow baseline desktop is not enlarged to ease the task.
    for side in (-1, 1):
        y = side*.55
        box(f"pedestal_{side}_foot", (0., y, .012), (.16, .15, .012), "steel")
        box(f"pedestal_{side}_column", (0., y, .37), (.055, .055, .346), "steel")
        box(f"pedestal_{side}_plate", (0., y, c.table_top-.012), (.13, .10, .012), "aluminum")
        for x in (-.095, .095):
            for dy in (-.075, .075):
                rod(f"mount_bolt_{side}_{x}_{dy}", (x, y+dy, c.table_top),
                    (x, y+dy, c.table_top+.005), .006)
    for x in (-.48, .48):
        for y in (-.38, .38):
            box(f"table_leg_{x}_{y}", (x, y, .36), (.025, .025, .36), "steel")
            box(f"table_foot_{x}_{y}", (x, y, .012), (.032, .032, .012), "robot_dark")
        box(f"table_rail_{x}", (x, 0., .69), (.022, .4, .03), "steel")
    for y in (-.38, .38):
        box(f"table_rail_y_{y}", (0., y, .69), (.48, .022, .03), "steel")
    box("back_wall", (-1.8, 0., 1.5), (.06, 2.3, 1.5), "wall")
    box("side_wall", (0., 2.3, 1.5), (1.8, .06, 1.5), "wall")
    box("front_wall", (0., -2.3, 1.5), (1.8, .06, 1.5), "wall")
    box("ceiling", (0., 0., 3.03), (1.8, 2.3, .03), "wall")
    box("wall_trim", (-1.72, 0., .05), (.012, 2.3, .05), "robot_white")
    box("cabinet", (-1.48, .3, .44), (.24, .72, .40), "robot_white")
    box("cabinet_top", (-1.48, .3, .855), (.27, .75, .015), "wood")
    for z in (.24, .52, .76):
        box(f"drawer_{z}", (-1.23, .3, z), (.012, .69, .105), "cabinet")
        box(f"drawer_handle_{z}", (-1.205, .3, z+.02), (.013, .12, .012), "steel")
    # Electronics are on a shelf below the task, not floating beside the arms.
    box("controller_shelf", (0., 0., .22), (.40, .30, .012), "steel")
    for y in (-.16, .16):
        box(f"robot_controller_{y}", (0., y, .28), (.22, .12, .047), "robot_dark")
        for x in np.linspace(-.16, .16, 9):
            box(f"controller_vent_{y}_{x}", (x, y-.121, .28), (.009, .001, .025), "steel")
    # All mount members exist in collision geometry as well as the renderer.
    eye, target = np.asarray(CAMERA_EYE), np.asarray(CAMERA_TARGET)
    back = (eye-target)/np.linalg.norm(eye-target)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    quat = Rotation.from_matrix(np.column_stack((right, up, back))).as_quat()
    box("bench_camera", eye+.013*back, (.045, .0125, .0125), "aluminum", quat=quat)
    box("bench_camera_front", eye+.0003*back, (.043, .011, .0003), "robot_dark", quat=quat)
    mast = np.array([.35, -1.00, 1.54])
    box("camera_stand_foot", (.35, -1., .02), (.12, .12, .02), "steel")
    rod("camera_stand", (.35, -1., .04), mast, .012)
    rod("camera_mount_arm", mast, eye+.030*back, .006)
    for i, pos in enumerate(((0., 0., 2.62), (-.9, .3, 2.32))):
        box(f"luminaire_{i}", pos, (.31, .61, .025), "robot_white")
    # A bound notebook is a movable rigid object, 220 g; no fixed prop is
    # falsely presented as a loose manipulable object.
    notebook = b.add_body(label="notebook", mass=.22,
        xform=wp.transform((.33, .24, c.table_top+.009), wp.quat_identity()))
    # FORCE_SPACE maps (ke,kd) through (1-dmax)/mass, not a unit-mass
    # spring. With m=.22 kg, dmax=.95 and ke=1e7, the old pair kd=5000
    # gave dampratio=.377 and unresolved alternating edge impacts at 960 Hz.
    # Equal-priority mixing with the table's kd=5000 gives kd_pair=15000,
    # dampratio=1.131. This is numerical contact damping, not measured paper
    # viscosity. Geometry, mass, stiffness, friction and pen/cap are unchanged.
    notebook_contact = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.4,
        ke=1e7, kd=25000., mu_torsional=0., mu_rolling=0.)
    box("notebook_pages", (0., 0., 0.), (.074, .105, .007), "paper", body=notebook, contact=notebook_contact)
    box("notebook_cover", (0., 0., -.0075), (.076, .107, .0005), "cap", body=notebook, contact=notebook_contact)
    box("notebook_spine", (-.074, 0., 0.), (.002, .107, .008), "cap", body=notebook, contact=notebook_contact)
    from .geometry import set_mass
    set_mass(b, notebook, .22)
    return {"notebook": notebook}

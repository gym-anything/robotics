"""Task-only poses and static nest, kept outside newton15, which is left unchanged.

No settling or physics is performed here. Existing shape/material/dynamics data
is untouched. Apply before finalize, so resets use the same authored start.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from .tasks import task_setup, validate

FIXTURE_XY = (-.20, 0.)
FIXTURE_HALF = .030
BORE_HALF = .007
FIXTURE_HEIGHT = .070
BORE_DEPTH = .050
NOTEBOOK_HALF = (.074, .105)
NOTEBOOK_TOP = .007  # scene.py pages, relative to body origin


def pose(position, rotation):
    return np.r_[position, rotation.as_quat()]


def default_poses(c):
    """The existing build.py/scene.py starts, also used by runtime-free tests."""
    lying = Rotation.from_euler("z", -90, degrees=True)
    return {"pen": pose((-.07, 0., c.table_top+.0075), lying),
            "cap": pose((-.080, .070, c.table_top+c.cap_radius+.0015), lying),
            "notebook": pose((.33, .24, c.table_top+.009), Rotation.identity())}


def pen_in_fixture(c):
    """The pen standing tail down in the capping fixture's bore (a fixture task's start, tasks/<task>/setup.py)."""
    # +Y points from tip to tail, hence downward. Tail clears the bore floor
    # by two contact gaps; nothing is pinned or welded in the nest.
    tip = (*FIXTURE_XY, c.table_top+FIXTURE_HEIGHT-BORE_DEPTH+c.pen_length+2*c.contact_gap)
    return pose(tip, Rotation.from_euler("x", -90, degrees=True))


def cap_on_pen(pen, depth):
    """The cap on the writing tip of the pen at pose `pen`, its mouth `depth` from the tip along the pen's axis."""
    rp = Rotation.from_quat(pen[3:])
    return pose(pen[:3]+rp.apply([0., depth, 0.]),
                rp*Rotation.from_euler("z", 180, degrees=True))


def start(task, c):
    """A task's start: the parts' poses (default_poses, moved by setup in its setup.py) and its changes to the default
    scene (setup's result: the parts it moved and its bench props; None for the scene as build.py authors it)."""
    validate(task)
    poses = default_poses(c)
    return poses, task_setup(task).setup(c, poses)


def start_poses(task, c):
    return start(task, c)[0]


def fixture_boxes(c):
    """Five nonoverlapping boxes: 60 mm square outside, 14 mm bore, 50 mm deep."""
    x, y = FIXTURE_XY
    h, r = FIXTURE_HALF, BORE_HALF
    floor = c.table_top+FIXTURE_HEIGHT-BORE_DEPTH
    wall_z = floor+BORE_DEPTH/2
    boxes = [("floor", (x, y, (c.table_top+floor)/2), (h, h, (floor-c.table_top)/2))]
    for sign in (-1, 1):
        boxes += [(f"x{sign}", (x+sign*(h+r)/2, y, wall_z), ((h-r)/2, h, BORE_DEPTH/2)),
                  (f"y{sign}", (x, y+sign*(h+r)/2, wall_z), (r, (h-r)/2, BORE_DEPTH/2))]
    return boxes


def apply_start(builder, bodies, materials, c, task, newton, wp):
    """Only authored object transforms and appended static task props may change (the task's start: start)."""
    poses, changes = start(task, c)
    if changes is None:  # the scene as build.py authors it (seat-cap)
        return
    if c.scene != "studio":
        raise ValueError("The seven new pencap tasks require the studio scene")
    for name in changes["moved"]:
        p = poses[name]
        builder.body_q[bodies[name]] = wp.transform(p[:3], p[3:])
    if "capping_fixture" in changes["props"]:
        contact = newton.ModelBuilder.ShapeConfig(gap=c.contact_gap, mu=.35, ke=1e7, kd=5000.,
                                                 mu_torsional=0., mu_rolling=0.)
        for name, position, half in fixture_boxes(c):
            i = builder.add_shape_box(-1, xform=wp.transform(position, wp.quat_identity()),
                                      hx=half[0], hy=half[1], hz=half[2],
                                      label="capping_fixture_"+name, cfg=contact)
            materials[i] = "aluminum"


def reach_report(c):
    """Conservative distance screen, NOT IK or a collision-free path certificate.

    It does not load the manufacturer URDF. Report base and shoulder
    distances for every authored interaction, including a central relay point.
    The 855 mm manufacturer reach quoted by the design is only a screen.
    """
    report = {}
    for task in ("pen-on-notebook", "unload-fixture", "reseat-cap", "load-fixture", "pass-pen", "uncap", "cap-in-fixture"):
        p, changes = start(task, c)
        points = {}
        for name, length in (("pen", c.pen_length), ("cap", c.cap_length)):
            axis = Rotation.from_quat(p[name][3:]).as_matrix()[:, 1]
            for label, u in (("origin", 0.), ("mid", length/2), ("end", length)):
                points[f"{name}_{label}"] = p[name][:3]+axis*u
        points["notebook_target"] = p["notebook"][:3]+[0., 0., NOTEBOOK_TOP+.0065]
        points["relay"] = np.array([0., 0., c.table_top+.10])
        if "capping_fixture" in changes["props"]:
            points["fixture_mouth"] = np.array([*FIXTURE_XY, c.table_top+FIXTURE_HEIGHT])
            points["upright_grasp"] = np.array([*FIXTURE_XY, c.table_top+.12])
        report[task] = {}
        for name, point in points.items():
            distances = {}
            for arm, y in (("left", .55), ("right", -.55)):
                base = np.array([0., y, c.table_top])
                distances[arm] = {"base_m": float(np.linalg.norm(point-base)),
                                  "shoulder_m": float(np.linalg.norm(point-base-[0., 0., .333]))}
            report[task][name] = {"position_m": point.tolist(), "arms": distances}
    return report

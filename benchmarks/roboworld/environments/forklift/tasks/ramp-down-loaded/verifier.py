"""Outcome of ramp-down-loaded: the pallet set down upright inside the drop zone and resting on the asphalt, the forks
backed out and lowered, the truck at rest, and never faster than 6 km/h on the ramp.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by) and progress(now, initial,
approach) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import math

import numpy as np

from forklift import params as P
from forklift.evaluator import C, _forks_low, blades, inside_rect, pallet_corners, shared
from forklift.task_progress import set_down, set_down_progress


DROP_ZONE = {"east": "drop_zone", "west": "drop_zone_w"}       # facility.yard zones


def verify(world, start):
    dz = world.scene.zones[DROP_ZONE[start.get("drop_zone", "east")]]

    def dest(n, xy):
        return n == "carried" and inside_rect(xy, dz["x"], dz["y"])
    crit, det = shared(world, start, dest_floor=dest)
    corners, o, R = pallet_corners(world, "carried")
    inside = all(inside_rect(c, dz["x"], dz["y"]) for c in corners)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    crit.append(C("in_drop_zone", inside, inside and tilt < 5.0, "-", "inside the drop zone, upright", "authored"))
    # set down, as in the floor bays: resting on the asphalt, the forks out of it and lowered, the truck at rest
    fz = [c[2] - world.scene.ground_height(*c[:2]) for c in corners]
    crit.append(C("resting_on_floor", fz, all(abs(z) <= 0.010 for z in fz), "m", "corners within 10 mm of the ground",
                  "authored (as the floor bays)"))
    bl, _ = blades(world)
    gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) - P.GMA_PALLET["L"] / 2
    crit.append(C("backed_out", gap, gap >= 0.5, "m", "fork tips >= 0.5 m from the pallet", "authored (as the floor bays)"))
    h, ok = _forks_low(world, 0.15)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips at or below travel height (0.15 m)", "authored (as the floor bays)"))
    crit.append(C("stationary", world.events.rest_s, world.events.rest_s >= 1.0, "s", ">= 1 s at rest", "authored"))
    v = world.events.ramp_max_speed * 3.6
    crit.append(C("slow_on_ramp", v, v <= 6.0, "km/h", "<= 6 km/h on the ramp", "OSHA 1910.178(n)(7): slowly (read); number authored"))
    return crit, det


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): the set-down measures in the drop zone, judged level."""
    return set_down(world, by, world.scene.zones[DROP_ZONE[start.get("drop_zone", "east")]], yard=True)


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): the set-down progress."""
    return set_down_progress(now, initial, approach)

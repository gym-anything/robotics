"""Outcome of start-and-park: the truck parked inside the parking bay's lines and at rest for 2 s, its forks lowered and
tilted forward onto the floor, the direction lever in neutral, the parking brake set, the engine off and the key
turned off.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by) and progress(now, initial,
approach) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import math

import numpy as np

from forklift.evaluator import C, _forks_low, blades, footprint, shared, yaw_of
from forklift.task_progress import clip, ramp


def verify(world, start):
    crit, det = shared(world, start)
    bay = world.scene.zones["parking"]
    p, R = world.body_pose("chassis")
    fp = footprint(float(p[0]), float(p[1]), yaw_of(R))
    worst = max(max(bay["x"][0] - x, x - bay["x"][1], bay["y"][0] - y, y - bay["y"][1]) for x, y in fp)
    crit.append(C("in_parking_bay", dict(outside_by_m=worst, footprint=[[round(x, 3), round(y, 3)] for x, y in fp]),
                  worst <= 0.0, "m", "the truck's footprint (counterweight to fork tips) inside the parking bay's lines",
                  "authored"))
    sig = world.signals
    rest = world.events.rest_s
    crit.append(C("stationary_2s", rest, rest >= 2.0, "s", ">= 2 s at rest", "authored"))
    crit.append(C("parking_brake_set", sig.get("park", 0.0), sig.get("park", 0.0) > 0.5, "-", "applied",
                  "OSHA 1910.178(m)(5)(i): brakes set (read)"))
    crit.append(C("direction_neutral", sig.get("direction"), sig.get("direction") == 0, "-", "neutral",
                  "OSHA 1910.178(m)(5)(i): controls neutral (read)"))
    crit.append(C("engine_off", world.drivetrain.running, not world.drivetrain.running, "-", "off",
                  "OSHA 1910.178(m)(5)(i): power shut off (read)"))
    crit.append(C("key_off", sig.get("key"), sig.get("key") == "off", "-", "key in the OFF position",
                  "Adaptalift 'How to park a forklift safely': turn the key or keyless switch to the off position (read)"))
    h, ok = _forks_low(world, 0.020)
    crit.append(C("forks_lowered", h, ok, "m", "fork tips <= 20 mm above the floor",
                  "OSHA 1910.178(m)(5)(i): load engaging means fully lowered (read)"))
    bl, axis = blades(world)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2])))))
    tips = [float(b["tip"][2] - world.scene.ground_height(*b["tip"][:2])) for b in bl]
    # at the carriage's lower stop the blades lie almost flat: tilting the mast forward puts the tips on the floor and
    # stalls there (blade pitch -0.07 deg with the tips touching, seed-0 demo); tilted back, the tips stand off it
    crit.append(C("forks_tilted_to_floor", dict(blade_pitch_deg=pitch, tip_heights_m=tips),
                  pitch <= 0.0 and max(tips) <= 0.005, "-", "both fork tips <= 5 mm above the floor (resting on it), the "
                  "blades sloping down to them (pitch <= 0)",
                  "Adaptalift: tilt the mast forward so the tyne tips are touching the ground; Dalhousie safe work "
                  "instructions (2018): forks lowered to the floor surface and tilted slightly forward (both read)"))
    return crit, det


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): how far the truck stands outside the bay, and the
    parking procedure's quality."""
    ev = world.events
    tilt = by["forks_tilted_to_floor"]
    quality = np.mean((clip(ev.rest_s / 2), clip(by["parking_brake_set"] / 0.5),
                       clip(1 - abs(by["direction_neutral"])), float(not by["engine_off"]),
                       float(by["key_off"] == "off"), ramp(by["forks_lowered"], 0.020, 0.30),
                       ramp(tilt["blade_pitch_deg"], 0, 5) * ramp(max(tilt["tip_heights_m"]), 0.005, 0.15)))
    return dict(distance=max(0.0, by["in_parking_bay"]["outside_by_m"]), quality=float(quality))


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): the approach to the bay, weighted by the procedure."""
    a = approach(now["distance"], initial["distance"])
    return a * (0.5 + 0.5 * now["quality"])

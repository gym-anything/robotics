"""Outcome of drive-to-pallet: the truck at rest in front of the pallet, fork tips 0.20-0.30 m short of it, blades level
at 20-60 mm above the floor and in line with its openings, centred on it and square to it.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by) and progress(now, initial,
approach) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import math

import numpy as np

from forklift import evaluator as E
from forklift import params as P
from forklift.evaluator import C, ang_diff, blades, pallet_corners, shared
from forklift.task_progress import band, ramp


FORK_HALF_W = 0.051                 # half the blade width (asset blades 102 mm, measured)


def verify(world, start):
    crit, det = shared(world, start)
    ev = world.events
    crit.append(C("stationary_2s", ev.rest_s, ev.rest_s >= 2.0, "s", ">= 2 s at rest", "authored"))
    corners, o, R = pallet_corners(world, "target")
    g = P.GMA_PALLET
    bl, axis = blades(world)
    # pallet face toward the truck: the pallet end nearest the fork tips
    px = R[:, 0]
    ends = [o + px * g["L"] / 2, o - px * g["L"] / 2]
    tipc = (bl[0]["tip"] + bl[1]["tip"]) / 2
    face = min(ends, key=lambda e: np.linalg.norm(e[:2] - tipc[:2]))
    dist = float(np.linalg.norm(face[:2] - tipc[:2]))
    crit.append(C("tips_short_of_pallet", dist, 0.20 <= dist <= 0.30, "m", "0.20-0.30 m short of the stringer ends",
                  "OSHA eTool: stop 8-12 in short (read)"))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, axis[2]))))
    crit.append(C("blades_level", pitch, abs(pitch) <= 1.0, "deg", "within +-1 deg", "derived (17 mm over the blade)"))
    zb = [b["tip"][2] - world.scene.ground_height(*b["tip"][:2]) for b in bl]
    crit.append(C("blade_height", zb, all(0.020 <= z <= 0.060 for z in zb), "m", "blade bottoms 20-60 mm above the floor",
                  "derived from the GMA opening (89 mm on a 16 mm deck)"))
    from forklift.loads import fork_openings
    # each blade inside its opening clear of the stringers (the asset's forks are fixed at +-293 mm, the GMA openings are
    # centred at +-245 mm and 451 mm wide, so a centred truck has each blade 48 mm outboard of its opening centre), and
    # the forks centred on the pallet (the load centred on the carriage)
    clear, lat = [], []
    for b in bl:
        rel = R.T @ (b["tip"] - o)
        lat.append(rel[1])
        clear.append(max(max(ow / 2 - abs(rel[1] - oc) for oc, ow, *_ in fork_openings()) - FORK_HALF_W, -1.0))
    crit.append(C("forks_in_openings", clear, all(v >= 0.040 for v in clear), "m",
                  ">= 40 mm from each blade to the stringers", "authored (blade 102 mm, opening 451 mm)"))
    centre = float(np.mean(lat))
    crit.append(C("centred_on_pallet", centre, abs(centre) <= 0.050, "m", "fork centre within +-50 mm of the pallet centre",
                  "authored"))
    yaw_err = abs(math.degrees(ang_diff(math.atan2(axis[1], axis[0]), math.atan2(px[1], px[0]))))
    yaw_err = min(yaw_err, 180.0 - yaw_err)
    crit.append(C("square_to_pallet", yaw_err, yaw_err <= 3.0, "deg", "fork axis within 3 deg of the pallet axis", "authored"))
    return crit, det


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): the fork tips' distance to the stop point in front of
    either end of the pallet, the quality of the stop and where the pallet is."""
    bl, axis = E.blades(world)
    _, o, R = E.pallet_corners(world, "target")
    tip = (bl[0]["tip"] + bl[1]["tip"]) / 2
    goals = [o + sign * R[:, 0] * (P.GMA_PALLET["L"] / 2 + 0.25) for sign in (-1, 1)]
    d = min(float(np.linalg.norm(tip[:2] - g[:2])) for g in goals)
    quality = np.mean((ramp(abs(by["tips_short_of_pallet"] - 0.25), 0.05, 0.6),
                       ramp(abs(by["blades_level"]), 1, 5),
                       min(band(z, 0.02, 0.06, 0, 0.2) for z in by["blade_height"]),
                       ramp(-min(by["forks_in_openings"]), -0.04, 0.05),
                       ramp(abs(by["centred_on_pallet"]), 0.05, 0.3), ramp(by["square_to_pallet"], 3, 15)))
    return dict(distance=d, quality=float(quality), pallet=o.tolist())


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): the approach weighted by the stop's quality, lost as the
    pallet is pushed."""
    a = approach(now["distance"], initial["distance"])
    moved = float(np.linalg.norm(np.array(now["pallet"]) - initial["pallet"]))
    return a * (0.4 + 0.6 * now["quality"]) * ramp(moved, 0.02, 0.3)

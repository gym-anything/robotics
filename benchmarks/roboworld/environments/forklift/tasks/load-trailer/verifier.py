"""Outcome of load-trailer: at a secured trailer, the pallet resting on the trailer floor behind the nose pallets on its
side, against the side wall, level, the truck back on the dock and the trailer not moved (evaluator._load_trailer, as
for load-trailer-chocked); at an unsecured one (signal red), the truck and its load never past the leveler's inner
edge, waiting at the door at rest with the parking brake set, in neutral, the pallet still on the forks.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by), progress(now, initial, approach)
and entered_on_red(world, start, history) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import math

import numpy as np

from forklift import evaluator as E
from forklift import params as P
from forklift.evaluator import C, _load_trailer, blades, pallet_corners, shared
from forklift.task_progress import _rect_quality, band, clip, gained, ramp


REACH_LOADED = 1.80     # m: a carried 48 in pallet's far edge ahead of the front axle (fork face 0.556 m + 1.219 m)


def _unsecured_trailer(world, start):
    """load-trailer with the trailer unsecured (restraint signal red): the job is not to board it. The truck and its
    load never reach the leveler (the pit's inner edge), and the truck is left stopped short of it, in neutral, the
    parking brake set, the load still on the forks."""
    crit, det = shared(world, start)
    d = world.scene.zones["doors"][start["target"]["door"]]
    edge = d["face_y"] - P.LEVELER["length"]
    traj = np.array(world.events.traj) if world.events.traj else np.zeros((0, 5))
    reach = float(np.max(traj[:, 2] + REACH_LOADED * np.sin(traj[:, 3]))) if len(traj) else -99.0
    crit.append(C("stayed_off_the_leveler", reach - edge, reach < edge, "m",
                  "the truck's loaded front never past the leveler's inner edge (OSHA 1910.178(m)(7): trucks boarded "
                  "only with brakes set and wheel chocks in place, read)", "read; reach derived"))
    # the job is done at the door: the load brought up to it, waiting for the signal, not left where the truck started.
    # The waiting place is between the staging lane's painted end and the leveler.
    lane = world.scene.zones["staging"]["lane" + start["target"]["door"][4:]]
    if len(traj):
        t_, x_, y_, yaw_, v_ = traj[-1]
        front = float(y_ + REACH_LOADED * math.sin(yaw_))
    else:
        front = -99.0
    crit.append(C("waiting_at_the_door", front - lane["y"][1], front >= lane["y"][1], "m",
                  "where the truck stopped, its loaded front past the end of the staging lane's lines (the load brought "
                  "up to the door to wait for the green signal)", "authored (the work order)"))
    sig = world.signals
    rest = world.events.rest_s
    ok = rest >= 2.0 and sig.get("park", 0.0) > 0.5 and sig.get("direction") == 0
    crit.append(C("stopped_short", dict(rest_s=rest, park=sig.get("park"), direction=sig.get("direction")), ok, "-",
                  ">= 2 s at rest, parking brake set, direction neutral", "authored"))
    corners, o, R = pallet_corners(world, "carried")
    bl, _ = blades(world)
    on_forks = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl) < P.GMA_PALLET["L"]
    crit.append(C("load_on_forks", on_forks, on_forks, "-", "the pallet still on the forks", "authored"))
    return crit, det


def verify(world, start):
    tr0 = start["trailers"][start["target"]["door"]]
    if not tr0.get("restraint") and not tr0.get("chocks"):
        return _unsecured_trailer(world, start)
    return _load_trailer(world, start)


def unsecured(start):
    tr = start["trailers"][start["target"]["door"]]
    return not tr.get("restraint") and not tr.get("chocks")


def _front(world):
    p, R = world.body_pose("chassis")
    return float(p[1] + REACH_LOADED * math.sin(E.yaw_of(R)))


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): at an unsecured trailer the distance still to go to the
    waiting place and the quality of the wait; otherwise the pallet's distance to its slot behind the nose pallets,
    the placement's quality and the truck's exit from the trailer."""
    ev, zones = world.events, world.scene.zones
    bl, axis = E.blades(world)
    corners, o, R = E.pallet_corners(world, "carried")
    if unsecured(start):
        lane = zones["staging"]["lane" + start["target"]["door"][4:]]
        gap = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl)
        quality = np.mean((clip(ev.rest_s / 2), clip(world.signals.get("park", 0) / 0.5),
                           clip(1 - abs(world.signals.get("direction", 0))), ramp(gap, P.GMA_PALLET["L"], 2.0)))
        return dict(distance=max(0.0, lane["y"][1] - _front(world)), quality=float(quality))
    tr, to, TR = E._trailer_frame(world, start["target"]["door"])
    local = (corners - to) @ TR
    ac, _, _ = E.pallet_corners(world, start["target"]["behind"])
    ahead_x = float(((ac - to) @ TR)[:, 0].min())
    sign = 1 if start["target"].get("side", "L") == "L" else -1
    goal = (ahead_x - P.GMA_PALLET["L"] / 2 - 0.025,
            sign * (tr["W"] / 2 - P.GMA_PALLET["W"] / 2 - E.WALL_GAP_MAX / 2))
    local_centre = TR.T @ (o - to)
    placement = (_rect_quality(local, dict(x=(0, tr["L"]), y=(-tr["W"] / 2, tr["W"] / 2))) *
                 ramp(max(abs(z) for z in local[:, 2]), 0.015, 0.25) *
                 band(by["gap_to_pallets_ahead"], -0.005, 0.05, -0.15, 0.4) *
                 band(by["gap_to_side_wall"], 0, E.WALL_GAP_MAX, -0.15, 0.5) * ramp(by["level"], 2, 6))
    p, Rc = world.body_pose("chassis")
    fp = E.footprint(float(p[0]), float(p[1]), E.yaw_of(Rc))
    ymin, ymax = min(v[1] for v in fp), max(v[1] for v in fp)
    face = zones["doors"][start["target"]["door"]]["face_y"]
    return dict(distance=float(np.linalg.norm(local_centre[:2] - goal)), placement=placement,
                exit=clip((face - ymin) / max(ymax - ymin, 1e-9)))


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): at an unsecured trailer the approach to the waiting place
    weighted by the wait's quality; otherwise approach, placement and exit."""
    a = approach(now["distance"], initial["distance"])
    if "quality" in now:
        return a * (0.5 + 0.5 * now["quality"])
    s = gained(now["placement"], initial["placement"])
    e = s * now["exit"]  # current placement, no required history or order of operations
    return 0.45 * a + 0.35 * s + 0.2 * e


def entered_on_red(world, start, *, history=True):
    """forklift-tasks-v1 and -v2 (TaskProgress.entered_on_red): at an unsecured trailer, whether the loaded front is
    past the leveler's inner edge now or (history) was at any sampled point of the trajectory."""
    if not unsecured(start):
        return False
    edge = world.scene.zones["doors"][start["target"]["door"]]["face_y"] - P.LEVELER["length"]
    return _front(world) >= edge or (history and any(y + REACH_LOADED * math.sin(yaw) >= edge
                                                     for _, _, y, yaw, _ in world.events.traj))

"""Outcome of put-away-rack: the pallet resting on the front and rear beams of the target location and on no other, at
the beam level, 51-101 mm past the front beam, at least 50 mm from the upright and the neighbouring pallet, level,
every stored load where it stood, and the forks withdrawn and lowered.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by) and progress(now, initial,
approach) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import math

import numpy as np

from forklift import evaluator as E
from forklift import params as P
from forklift.evaluator import C, _forks_low, blades, body_frame, pallet_corners, shared
from forklift.task_progress import band, clip, gained, ramp


def verify(world, start):
    crit, det = shared(world, start)
    loc = world.scene.zones["rack_locations"][start["rack_location"]]
    corners, o, R = pallet_corners(world, "carried")
    from forklift.facility import BEAM_W
    beam_x = (loc["beams"][0] + BEAM_W / 2, loc["beams"][1] - BEAM_W / 2)     # the location's front and rear beams
    m, d = world.solver.mj_model, world.solver.mj_data
    pal_geoms = {g for g in range(m.ngeom) if world.mj_geom_label(g).startswith("carried:")}
    beams, other_beams = set(), set()
    for i in range(d.ncon):
        c = d.contact[i]
        for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if ga in pal_geoms:
                lab = world.mj_geom_label(gb)
                if "_beam" in lab:
                    parts = lab.split(":")[0].split("_")          # <row><frame>_beam<bay-1>_<level>_<x>
                    try:
                        bay, lv, bx = int(parts[1][4:]) + 1, int(parts[2]), float(parts[-1])
                    except (IndexError, ValueError):
                        continue
                    mine = bay == loc["bay"] and lv == loc["level"] and min(abs(bx - x) for x in beam_x) < 0.01
                    (beams if mine else other_beams).add(parts[-1])
    crit.append(C("on_both_beams", dict(target_location=sorted(beams), elsewhere=sorted(other_beams)),
                  len(beams) >= 2 and not other_beams, "-",
                  f"contact on the front and the rear beam of bay {loc['bay']} at beam level {loc['level']} "
                  f"({loc['z']:.2f} m), on no other beam", "authored"))
    fz = [float(cz) for cz in corners[:, 2]]
    crit.append(C("at_beam_level", fz, all(abs(z - loc["z"]) <= 0.03 for z in fz), "m",
                  f"pallet bottom corners within 30 mm of the beam tops at {loc['z']:.2f} m", "authored"))
    from forklift.facility import RACK_BEAM, BAY_PITCH
    xs = corners[:, 0]
    overhang = float(loc["aisle_x"] - xs.min()) if loc["face"] == "w" else float(xs.max() - loc["aisle_x"])
    crit.append(C("front_overhang", overhang, 0.051 <= overhang <= 0.101, "m", "51-101 mm past the front beam",
                  "Steel King: 3 in nominal +-1 in (read)"))
    slot = int(start["rack_location"].split("-")[-1])
    yc = loc["y"] + (RACK_BEAM / 4 if slot == 0 else -RACK_BEAM / 4)
    ys = corners[:, 1]
    lo_face, hi_face = yc - RACK_BEAM / 2, yc + RACK_BEAM / 2
    other = start["rack_location"][:-1] + str(1 - slot)
    gaps = [float(ys.min() - lo_face) if slot == 0 else float(hi_face - ys.max())]
    if other in world.scene.layout.zones["rack_locations"] and other not in world.scene.layout.reserved:
        oy = world.scene.layout.zones["rack_locations"][other]["y"]
        gaps.append(float((oy - P.GMA_PALLET["W"] / 2) - ys.max()) if slot == 0 else float(ys.min() - (oy + P.GMA_PALLET["W"] / 2)))
    else:
        gaps.append(float(hi_face - ys.max()) if slot == 0 else float(ys.min() - lo_face))
    clear = min(gaps)
    crit.append(C("side_clearance", clear, clear >= 0.05, "m", ">= 50 mm to the upright and the neighbouring pallet",
                  "3-6 in between loads (read)"))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    crit.append(C("level", tilt, tilt <= 2.0, "deg", "within 2 deg", "authored"))
    # the stored loads that are bodies (at and beside the slot): where they were
    moved = {n: float(np.linalg.norm(body_frame(world, info["body"])[0] - np.asarray(world.scene.load_start_pose[n])))
             for n, info in world.scene.loads.items() if n != "carried" and not n.startswith("carried_")}
    worst = max(moved.values()) if moved else 0.0
    crit.append(C("stored_loads_undisturbed", dict(max_m=worst, moved={k: v for k, v in moved.items() if v > 0.005}),
                  worst <= 0.030, "m", "every stored load within 30 mm of where it stood", "authored (as unload-trailer)"))
    bl, _ = blades(world)
    out_ = min(float(np.linalg.norm(b["tip"][:2] - o[:2])) for b in bl)
    h, ok = _forks_low(world, 0.30)
    crit.append(C("forks_withdrawn_lowered", dict(tip_to_pallet_m=out_, tip_height_m=h), out_ > 1.0 and ok, "-",
                  "tips clear of the location and at travel height", "authored"))
    return crit, det


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): the pallet's distance to its place on the beams, its
    height error, the placement's quality and the forks' withdrawal."""
    zones = world.scene.zones
    corners, o, R = E.pallet_corners(world, "carried")
    loc = zones["rack_locations"][start["rack_location"]]
    sign = 1 if loc["face"] == "w" else -1
    goal = (loc["aisle_x"] + sign * (P.GMA_PALLET["L"] / 2 - 0.076), loc["y"])
    # Geometry, not a count of beam contacts: target-height and footprint proximity
    # grade placement continuously; the unchanged success rule still needs both beams.
    d = float(np.linalg.norm(o[:2] - goal))
    height_error = max(abs(z - loc["z"]) for z in by["at_beam_level"])
    placement = (ramp(d, 0.03, 0.4) * ramp(height_error, 0.03, 0.3) *
                 band(by["front_overhang"], 0.051, 0.101, -0.05, 0.25) *
                 ramp(-by["side_clearance"], -0.05, 0) * ramp(by["level"], 2, 6))
    withdraw = by["forks_withdrawn_lowered"]
    return dict(distance=d, height_error=abs(float(o[2]) - loc["z"]), placement=placement,
                exit=clip(withdraw["tip_to_pallet_m"]) * ramp(withdraw["tip_height_m"], 0.30, 1.0))


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): approach, lift to the beam level, placement and exit."""
    a = approach(now["distance"], initial["distance"])
    s = gained(now["placement"], initial["placement"])
    e = s * now["exit"]  # current placement, no required history or order of operations
    h = approach(now["height_error"], initial["height_error"])
    return 0.35 * a + 0.15 * h + 0.3 * s + 0.2 * e

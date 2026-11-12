"""Outcome of aisle-cruise: each straight section of the main aisle driven at 6.5-8.0 km/h for at least 80 % of its
length, both intersections crossed at 5 km/h or less with the horn sounded within 3 s before, the truck within 0.6 m
of the centreline, and stopped with its front wheels within 0.2 m of the far end's white line.

verify(world, start) gets the world at the evaluation (bodies, event latches, signals) and the resolved start, and
returns (criteria, details): the shared site rules (evaluator.shared) and the task's own criteria, each evaluator.C,
which evaluator.evaluate scores (the fraction met). measurements(world, start, by) and progress(now, initial,
approach) are the task's forklift-tasks-v1 and -v2 grading (task_progress.py)."""
import numpy as np

from forklift.evaluator import C, shared
from forklift.task_progress import clip, ramp


def front_axle_line(world):
    t = world.tyres.telemetry
    xs = [np.asarray(t[n]["contact"]) for n in ("wheel_fl", "wheel_fr") if t[n]["contact"] is not None]
    if xs:
        return np.mean(xs, axis=0)
    p, R = world.body_pose("chassis")
    return p


# the straight sections judged for the cruise speed, in the order driven: between the start and the first
# intersection's slow-down, between the intersections, and between the second intersection and the stop (the autopilot
# slows from 1.7 m before an intersection's entry edge and brakes to the far line at 0.6 m/s^2). Westward the start is
# 9 m from the first intersection (16.5 m eastward), so its first section is the 3 m in which a truck accelerating at
# 0.5 m/s^2 from the start is at speed
CRUISE_SECTIONS = {"east": ((6.0, 18.5), (25.0, 32.5), (39.0, 45.5)),
                   "west": ((40.0, 43.0), (26.0, 33.5), (10.0, 19.0))}


CRUISE_END = {"east": "W1_end", "west": "west_end"}       # facility.warehouse stop_lines


def verify(world, start):
    crit, det = shared(world, start)
    ev = world.events
    traj = np.array(ev.traj) if ev.traj else np.zeros((0, 5))
    way = start.get("cruise", "east")
    sections = CRUISE_SECTIONS[way]
    for k, (a, b) in enumerate(sections):
        m = (traj[:, 1] >= a) & (traj[:, 1] <= b) if len(traj) else np.zeros(0, bool)
        if m.sum() < 2:
            crit.append(C(f"cruise_section_{k + 1}", 0.0, False, "fraction", ">= 0.80 of the section at 6.5-8.0 km/h",
                          "authored"))
            continue
        x = traj[m, 1]; v = np.abs(traj[m, 4]) * 3.6
        dx = np.abs(np.diff(x, prepend=x[0]))
        inband = ((v >= 6.5) & (v <= 8.0)) * dx
        frac = float(inband.sum() / max(dx.sum(), 1e-6))
        crit.append(C(f"cruise_section_{k + 1}", frac, frac >= 0.8, "fraction", ">= 0.80 at 6.5-8.0 km/h", "authored"))
    for zn in ("I1", "I2"):
        cr = [c for c in ev.crossings if c["zone"] == zn]
        if not cr:
            crit.append(C(f"intersection_{zn}", None, False, "-", "crossed at <= 5 km/h, horn within 3 s before",
                          "OSHA 1910.178(n)(4): slow down and sound the horn (read); numbers authored"))
            continue
        c0 = cr[0]
        ok = c0["speed"] * 3.6 <= 5.0 and c0["since_horn"] is not None and c0["since_horn"] <= 3.0
        crit.append(C(f"intersection_{zn}", dict(speed_kmh=c0["speed"] * 3.6, since_horn_s=c0["since_horn"]), ok, "-",
                      "<= 5 km/h, horn within 3 s before",
                      "OSHA 1910.178(n)(4): slow down and sound the horn (read); numbers authored"))
    ycl = sum(world.scene.zones["main_aisle"]["y"]) / 2
    if len(traj):
        inside = traj[(traj[:, 1] > 4.0) & (traj[:, 1] < 46.0)]
        dev = float(np.max(np.abs(inside[:, 2] - ycl))) if len(inside) else 99.0
    else:
        dev = 99.0
    crit.append(C("aisle_centreline", dev, dev <= 0.6, "m", "truck centre within 0.6 m of the aisle centreline", "authored"))
    fa = front_axle_line(world)
    err = float(fa[0] - world.scene.zones["stop_lines"][CRUISE_END[way]]["x"])
    crit.append(C("at_far_line", err, abs(err) <= 0.2 and ev.rest_s >= 1.0, "m", "stopped, front wheels within +-0.2 m "
                  f"of the {way} end's white line", "authored"))
    return crit, det


def measurements(world, start, by):
    """forklift-tasks-v1 and -v2 (task_progress.measurements): each section's in-band share, each intersection's
    crossing, the centreline and the stop at the far line."""
    ev = world.events
    way = start.get("cruise", "east")
    traj = ev.traj
    sections = []
    for a, b in CRUISE_SECTIONS[way]:
        # In-band distance, capped by the unique span traversed: a tiny in-band move or
        # repeated shuttling inside one part of a section cannot earn the whole section.
        xs = [r for r in traj if a <= r[1] <= b]
        good = sum(abs(q[1] - p[1]) for p, q in zip(xs, xs[1:]) if 6.5 <= abs(q[4]) * 3.6 <= 8.0)
        span = max((r[1] for r in xs), default=a) - min((r[1] for r in xs), default=a)
        sections.append(min(clip(good / (0.8 * (b - a))), clip(span / (b - a))))
    crossings = []
    for name in ("intersection_I1", "intersection_I2"):
        c = by[name]
        crossings.append(0.0 if c is None or c["since_horn_s"] is None else
                         ramp(c["speed_kmh"], 5, 9) * ramp(c["since_horn_s"], 3, 8))
    error = abs(by["at_far_line"])
    return dict(distance=error, sections=sections, crossings=crossings,
                centre=ramp(by["aisle_centreline"], 0.6, 1.5),
                stop=ramp(error, 0.2, 2.0) * clip(ev.rest_s))


def progress(now, initial, approach):
    """forklift-tasks-v1 and -v2 (task_progress.progress): the mean of the sections, the crossings, the centreline
    held on the approach and the stop."""
    a = approach(now["distance"], initial["distance"])
    return float(np.mean([*now["sections"], *now["crossings"], a * now["centre"], now["stop"]]))

"""Outcome of rebar-relocate: the three marks clear of the bar each hold an anchor hole (within 3 mm of the mark and
3 degrees of the wall's normal, 108 to 125 mm deep, the reinforcement not cut); the hole over the bar was started at its
mark, reached the bar and stopped without cutting it; and its replacement on the same channel line, at least twice the
stopped hole's depth and at most 150 mm from it, meets the anchor-hole criteria.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: task_set_v2_evaluator)
and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under drilling-tasks-v2 the
continuous progress(ev) and its terms(ev) as well."""
import numpy as np

from drilling import params as P
from drilling.catalogue_evaluator import clip, mean
from drilling.layout import WALL_AXIS_TOL_DEG, WALL_MARK_TOL_M
from drilling.task import _angle_deg, _check, _ramp

# rebar-relocate: an anchor hole's depth window (task._anchor_hole) and the far mat's front face, where the depth
# credit reaches 0 (spec: 149 mm; concrete.ConcreteFields puts the far bars at WALL_THICK - FAR_MAT_COVER - r)
ANCHOR_DEPTH_M = (0.108, 0.125)
FAR_MAT_FACE_M = P.WALL_THICK_M - P.FAR_MAT_COVER_M - P.REBAR_D_M
REBAR_CUT_M = 0.0005                     # task._anchor_hole / checks: more than 0.5 mm into a bar cuts it
REPLACEMENT_FAR_M = 0.200                # spec: the distance credit outside the window is 0 at 200 mm


def verify(ev):
    return ev.evaluate_checks(checks, progress=progress, terms=terms)


def reached_bar(ev, h):
    """The stopped hole reached the reinforcement: the hole module recorded steel under the bit, or the hole's floor
    lies within the bit's radius of a bar's surface (wall fields)."""
    if any(e.get("event") == "rebar_contact" for e in h.events):
        return True
    d = ev.w.hm.drillables["wall"]
    if d.fields is None:
        return False
    floor = h.p0 + h.a*h.depth
    return d.fields.bar_distance(d.to_fields(floor))[0] <= 0.5e-3*h.d_mm


def checks(ev):
    """Every mark but the one over the bar meets the anchor-hole criteria; the hole over the bar was started, reached
    the bar and stopped without cutting it; a replacement on its channel line, at least ABORTED_HOLE_DISTANCE_FACTOR
    x the stopped hole's depth and at most RELOCATE_MAX_M from it (ETAG 001 Part 1 7.3: the aborted hole may stay
    unfilled at that distance), meets the anchor-hole criteria."""
    out = []
    kbar = ev.L["rebar_mark"]
    for k, mk in enumerate(ev.L["marks"]):
        if k != kbar:
            out += ev._anchor_hole(mk, f"hole {k}")[0]
    m = ev._wall_mark_local(ev.L["marks"][kbar])
    hit = ev.nearest("wall", m)
    aborted = hit[0] if hit is not None and hit[1] <= WALL_MARK_TOL_M else None
    out.append(_check(f"hole {kbar} (over the bar): started within {WALL_MARK_TOL_M*1e3:.0f} mm of its mark",
                      aborted is not None))
    if aborted is not None:
        out.append(_check(f"hole {kbar} (over the bar): reached the bar", reached_bar(ev, aborted),
                          round(aborted.depth*1e3, 1), "steel under the bit"))
        out.append(_check(f"hole {kbar} (over the bar): the bar was not cut (damage at most 0.5 mm)",
                          aborted.rebar_damage_m <= 0.0005, round(aborted.rebar_damage_m*1e3, 3), "<= 0.5 mm"))
        d_min = P.ABORTED_HOLE_DISTANCE_FACTOR*aborted.depth
        others = [ev._wall_mark_local(mk) for k, mk in enumerate(ev.L["marks"]) if k != kbar]
        cands, ok = [], []
        for h in ev.holes("wall"):
            if h is aborted:
                continue
            dz = abs(float(h.p0[2] - aborted.p0[2]))
            dx = abs(float(h.p0[0] - aborted.p0[0]))
            if dz <= WALL_MARK_TOL_M and d_min <= dx <= P.RELOCATE_MAX_M and \
                    all(np.linalg.norm((h.p0 - o)[[0, 2]]) > 0.02 for o in others):
                cands.append(h)
        for h in cands:
            mk2 = dict(x=float(ev.w.hm.world_point(ev.w.hm.drillables["wall"], h.p0)[0]),
                       z=float(ev.w.hm.world_point(ev.w.hm.drillables["wall"], h.p0)[2]))
            checks, _ = ev._anchor_hole(mk2, "replacement")
            ok.append(all(c["passed"] for c in checks))
        out.append(_check(f"a replacement on the channel line {d_min*1e3:.0f}-{P.RELOCATE_MAX_M*1e3:.0f} mm from the "
                          "stopped hole meeting the anchor-hole criteria", any(ok), len(cands),
                          f"{P.ABORTED_HOLE_DISTANCE_FACTOR:g} x {aborted.depth*1e3:.1f} mm to "
                          f"{P.RELOCATE_MAX_M*1e3:.0f} mm"))
    return out


def wall_bore(ev, h, e):
    """rebar-relocate anchor-hole credit (spec task 7): ramp(entry; 3, 6 mm) x ramp(axis; 3, 6 deg) x
    min(depth / 108 mm, 1) x ramp(depth; 125 mm, far-mat face). A cut bar or a broken bit's stub zeroes it."""
    if not ev._intact(h) or h.rebar_damage_m > REBAR_CUT_M:
        return 0.
    fd = h.full_depth()
    return (_ramp(e, WALL_MARK_TOL_M, 2*WALL_MARK_TOL_M)
            * _ramp(_angle_deg(h.a, -h.face_normal), WALL_AXIS_TOL_DEG, 2*WALL_AXIS_TOL_DEG)
            * clip(fd/ANCHOR_DEPTH_M[0])*_ramp(fd, ANCHOR_DEPTH_M[1], FAR_MAT_FACE_M))


def stop_share(ev, h):
    """S of the hole over the bar: 1 once it reached the bar uncut, else its depth / the depth at which it would
    reach the bar (read from the wall's bar geometry, as reached_bar does)."""
    if h.rebar_damage_m > REBAR_CUT_M or not ev._intact(h):
        return 0.
    if reached_bar(ev, h):
        return 1.
    d = ev.w.hm.drillables["wall"]
    if d.fields is not None:
        gap = d.fields.bar_distance(d.to_fields(h.p0 + h.a*h.depth))[0] - 0.5e-3*h.d_mm
    else:
        gap = P.REBAR_COVER_M + P.REBAR_D_M - h.depth   # the vertical bar's face: cover + the horizontal layer
    gap = max(gap, 0.)
    return clip(h.depth/(h.depth + gap)) if h.depth + gap > 0. else 0.


def rebar_state(ev):
    """Per-mark credit M_k (spec task 7), the number of holes that cut a bar, and the stopped hole."""
    L, kbar = ev.L, ev.L["rebar_mark"]
    marks = [ev._wall_mark_local(mk) for mk in L["marks"]]
    credits = []
    for k, m in enumerate(marks):
        if k == kbar:
            credits.append(None)
            continue
        hit = ev.nearest("wall", m)
        credits.append(0. if hit is None or hit[1] > 0.02 else wall_bore(ev, *hit))
    hit = ev.nearest("wall", marks[kbar])
    stopped = hit if hit is not None and hit[1] <= 2*WALL_MARK_TOL_M else None
    S = 0. if stopped is None else stop_share(ev, stopped[0])*_ramp(stopped[1], WALL_MARK_TOL_M,
                                                                        2*WALL_MARK_TOL_M)
    # the replacement: on the stopped hole's channel line, 2 x its depth to 150 mm from it; without a stopped
    # hole, from the mark with the layout's nominal minimum (twice the bar's face depth plus 10 mm)
    if stopped is not None:
        ref, d_min = stopped[0].p0, P.ABORTED_HOLE_DISTANCE_FACTOR*stopped[0].depth
    else:
        ref, d_min = marks[kbar], L["replacement"]["min_m"]
    others = [m for k, m in enumerate(marks) if k != kbar]
    R = 0.
    for h in ev.holes("wall"):
        if stopped is not None and h is stopped[0]:
            continue
        if not all(np.linalg.norm((h.p0 - o)[[0, 2]]) > 0.02 for o in others):
            continue
        dz = abs(float(h.p0[2] - ref[2]))
        dx = abs(float(h.p0[0] - ref[0]))
        near = clip((dx - 0.5*d_min)/(0.5*d_min)) if d_min > 0. else 1.
        window = min(near, _ramp(dx, P.RELOCATE_MAX_M, REPLACEMENT_FAR_M))
        R = max(R, wall_bore(ev, h, dz)*window)
    cut = sum(h.rebar_damage_m > REBAR_CUT_M for h in ev.holes("wall"))
    bar_cut = stopped is not None and stopped[0].rebar_damage_m > REBAR_CUT_M
    credits[kbar] = 0. if bar_cut else (S + R)/2.
    return credits, cut, S, R


def progress(ev):
    credits, cut, _, _ = rebar_state(ev)
    return clip(mean(credits) - 0.25*cut)


def terms(ev):
    """The drilling-tasks-v2 report's progress_terms."""
    credits, cut, S, R = rebar_state(ev)
    return dict(marks=[round(c, 4) for c in credits], stopped_hole=round(S, 4), replacement=round(R, 4),
                holes_cutting_a_bar=int(cut))

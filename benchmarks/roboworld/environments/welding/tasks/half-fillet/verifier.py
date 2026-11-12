"""Outcome of half-fillet: a 1/4 in fillet over the right half of the near-side joint (HALF_SPAN), sound and fused as
fillet-2f's span, every crater filled to size, the left half left unwelded up to the second tack (HALF_LEFT_FREE),
undercut within D1.1's limits over the whole joint and the weld solid.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle=True, the on-demand
evaluation, judges a weld still freezing with the arc off as it freezes hands-off) and returns the evaluation:
success, score, checks, progress, reward, judged and flags (judge_half_fillet below on Evaluator._final_rows). WELD:
the task is judged on the weld's sections (the evaluator reads the start state's, Evaluator.start_rows)."""
import numpy as np

from welding.task import (END_ALLOWANCE, FREE_ZONE_PENALTY, L_BRACKET, NOMINAL_LEG, START_OVERLAP_MAX, _in,
                          _joint_checks, _key, _ok, _section_defects, _undercut_beyond_allowance, _window_checks,
                          free_share, section_credit, window_credit)

WELD = True
# half-fillet: the right half of the joint, END_ALLOWANCE in from its end to its middle, judged as fillet-2f judges its
# span (size, throat, per-section soundness, fusion, the start zone's lap bound), every crater of the weld laid in the
# episode or touching the half filled to size, and the left half left unwelded up to the second tack (its metal starts
# at 125.75 mm in the tacked start). 70 mm of span hold the aim_tacks start's 5.5 mm of undersize (5.25-10.25 mm)
# inside D1.1's 10 %; a 2 in stitch's 45 mm do not (below)
HALF_SPAN = (END_ALLOWANCE, L_BRACKET/2)
HALF_FUSION_X = (6e-3, .25*L_BRACKET, 70e-3)
HALF_LEFT_FREE = (100e-3, 120e-3)


def verify(e, settle=False):
    return judge_half_fillet(e, *e._final_rows(settle))


def episode_metal(start_rows):
    """A predicate on sections: True where the section holds weld metal the episode laid or melted again, False where it
    is bare or holds the start state's metal as it froze then (the fixture's tacks, untouched)."""
    old = {_key(r): r.get("freeze_t") for r in (start_rows or []) if r["weld_area_m2"] > 0}

    def new(r):
        if r["weld_area_m2"] <= 0:
            return False
        k = _key(r)
        if k not in old:
            return True
        t, t0 = r.get("freeze_t"), old[k]
        return t is None or t0 is None or abs(t - t0) > 1e-9
    return new


def _half_reading(e, rows, new):
    """half-fillet's craters judged (every one touching the half, and every one of the metal the episode laid,
    wherever it lies: the second tack's, untouched, is the fixture's), and the raw progress: the half's fillet credit
    less, per length of the half, the judged craters' deficit beyond it and the start zone's rejected metal (as
    fillet_progress charges what success judges outside its span)."""
    lo, hi = HALF_SPAN
    beyond = _undercut_beyond_allowance(rows)
    craters = [c for c in e._craters(rows) if c["rows"] and any(_in(r, lo, hi) or new(r) for r in c["rows"])]
    in_crater = {_key(r) for c in craters for r in c["rows"]}
    L = sum(r["dx_m"] for r in rows if _in(r, lo, hi))
    charge = 0.
    for r in rows:
        if not 0. <= r["x_m"] <= L_BRACKET or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            charge += r["dx_m"]*(1. - section_credit(r, beyond)[0])
        elif r["x_m"] < lo and _section_defects(r, beyond, START_OVERLAP_MAX):
            charge += r["dx_m"]
    raw = window_credit(rows, lo, hi, beyond) - (charge/L if L > 0 else 0.)
    return craters, in_crater, beyond, raw


def judge_half_fillet(e, rows, judged=None, flags=None):
    """half-fillet (A9) on one state's sections. The half (HALF_SPAN) as fillet-2f judges its span; beyond it the
    judged craters' metal at one cell of overlap, the start zone's at START_OVERLAP_MAX, the rest of the metal the
    episode laid for convexity and porosity (fillet-2f judges both on all weld metal); the craters (_half_reading)
    filled to 6.35 mm; undercut on the whole joint; HALF_LEFT_FREE bare; solid. Progress: (raw - raw0) / (1 -
    raw0) less FREE_ZONE_PENALTY x the share of HALF_LEFT_FREE holding metal, clipped to [0, 1]; raw0 the tacked
    start's."""
    lo, hi = HALF_SPAN
    new = episode_metal(e.start_rows)
    craters, in_crater, beyond, raw = _half_reading(e, rows, new)
    checks = _window_checks(rows, lo, hi, HALF_FUSION_X, beyond)
    outside = []
    for r in rows:
        if not 0. <= r["x_m"] <= L_BRACKET or _in(r, lo, hi) or r["weld_area_m2"] <= 0:
            continue
        if _key(r) in in_crater:
            bad = _section_defects(r, beyond)
        elif r["x_m"] < lo:
            bad = _section_defects(r, beyond, START_OVERLAP_MAX)
        elif new(r):
            bad = [d for d in _section_defects(r, beyond) if d in ("convexity", "porosity")]
        else:
            bad = []
        if bad:
            outside.append((round(r["x_m"]*1e3, 2), bad))
    worst = min(craters, key=lambda c: c["min_size_mm"]) if craters else None
    checks["sound_beyond_half"] = dict(rejected_sections=outside[:40],
                                       start_zone_overlap_limit_mm=START_OVERLAP_MAX*1e3, ok=_ok(not outside))
    checks["crater_filled"] = dict(min_size_mm=worst["min_size_mm"] if worst else 0.,
                                   craters=[{k: v for k, v in c.items() if k != "rows"} for c in craters],
                                   limit_mm=NOMINAL_LEG*1e3, ok=_ok(craters and all(c["ok"] for c in craters)))
    checks.update(_joint_checks(rows, HALF_LEFT_FREE, "left_half_unwelded"))
    checks = e._named(checks)
    raw0 = _half_reading(e, e.start_rows, episode_metal(e.start_rows))[3] if e.start_rows else 0.
    share = free_share(rows, *HALF_LEFT_FREE)
    net = (raw - raw0)/(1. - raw0) if raw0 < 1. else float(raw >= 1.)
    reward = dict(progress=float(np.clip(net - FREE_ZONE_PENALTY*share, 0., 1.)), raw=raw, start_raw=raw0,
                  net_progress=float(net), left_half_metal_share=share)
    return e._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                     flags=flags or {})

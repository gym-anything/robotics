"""Outcome of medallions: 25 +- 5 mm medallions cut from the thick part of the pork tenderloin, the thickest and
thinnest within 5 mm, cuts square to its axis, no tag ends.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator: task.Evaluator
for the released checks (released_checks), or the task-set evaluator of filleting-tasks-v1 or -v2
(task_set_evaluator.py, task_set_v2_evaluator.py), whose cutting grading (_cut_task) reads PRODUCT, released_checks,
product and progress below."""
import numpy as np

from filleting.task import NOTHING_MADE, TIME_LIMIT_S, check
from filleting.task_set_progress import clip, ramp

PRODUCT = "slices"


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(released_checks)
    return e.evaluate_graded()


def released_checks(e):
    """Medallions 25 +- 5 mm thick from the thick part: pieces of 30 g or more with both ends cut; thickness along
    the tenderloin's axis (rest space) measured on their tets more than 6 mm from the piece's outline, thickest
    minus thinnest <= 5 mm, blade surfaces within 10 deg of perpendicular to the axis; no loose piece of 1 to 5 g.
    (The IMPS suspension check is not modelled.)"""
    env, w = e.env, e.env.world
    ts = w.tissue
    lab, frags, main = e._parts()
    cen = ts.hX[ts.htets[:len(w.cut.orig)]].mean(1)
    rows = []
    for c, f in frags.items():
        if f["mass"] < .03:
            continue
        p = cen[f["tets"]]
        r = np.hypot(p[:, 1] - p[:, 1].mean(), p[:, 2] - p[:, 2].mean())
        core = p[r <= max(r.max() - .006, 0.)]
        if len(core) < 4:
            continue
        ends = e._cut_faces_between(lab, c)
        both = len(ends) and (ends[:, 0].min() < p[:, 0].mean() < ends[:, 0].max())
        th = float(np.ptp(core[:, 0])) + w.config.mesh_h/2      # tet centroids sit ~h/4 inside each face
        rows.append(dict(fragment=c, mass_g=f["mass"]*1e3, thickness_mm=th*1e3, both_ends_cut=bool(both)))
    # a medallion is a slice: both ends cut and at most twice the nominal thickness
    meds = [r for r in rows if r["both_ends_cut"] and r["thickness_mm"] <= 50.]
    n, a = e._sweep_normals()
    tilt = float(np.degrees(np.arccos(np.clip(np.sum(np.abs(n[:, 0])*a)/max(a.sum(), 1e-15), 0, 1)))) if len(a) else None
    ths = [r["thickness_mm"] for r in meds]
    tags = [c for c, f in frags.items() if 1e-3 <= f["mass"] < 5e-3]
    checks = dict([check("medallions", len(meds), 1, "count", len(meds) >= 1),
                   check("thickness_25_5mm", float(max((abs(t - 25.) for t in ths), default=np.nan)), 5., "mm worst",
                         bool(ths) and all(abs(t - 25.) <= 5. for t in ths)),
                   check("thickness_spread_mm", float(np.ptp(ths)) if ths else None, 5., "mm", bool(ths) and np.ptp(ths) <= 5.),
                   check("cut_tilt_deg", tilt, 10., "deg from perpendicular",
                         bool(meds) and tilt is not None and tilt <= 10., "" if meds else NOTHING_MADE),
                   check("no_tag_ends", len(tags), 0, "pieces of 1-5 g", not tags),
                   check("time", env.sim_time, TIME_LIMIT_S["C3"], "s", env.sim_time <= TIME_LIMIT_S["C3"])])
    dm = e._common_cut(checks, frags, [c for c, f in frags.items() if f["mass"] >= .03])
    return checks, dict(**dm, pieces=rows)


def product(e, ref, reg):
    """The number of medallions the start allows."""
    # Nominal thick 100 mm of the 350 mm loin, scaled by this seed's length.
    return max(1, int(round(4*e.env.world.workpiece.extra["model"].fL)))


def progress(e, checks, metrics, frags, partial, tilt):
    """Raw progress: the medallions' thicknesses and spread, the best unfinished cut and the cut tilt."""
    meds = [r for r in metrics["pieces"] if r["both_ends_cut"] and r["thickness_mm"] <= 50.]
    quality = sum(ramp(abs(r["thickness_mm"] - 25.), 5., 12.) for r in meds)
    spread = checks["thickness_spread_mm"]["value"]
    raw = clip((quality + .5*partial)/e.target_count)*tilt*(ramp(spread, 5., 12.) if meds else 1.)
    return raw

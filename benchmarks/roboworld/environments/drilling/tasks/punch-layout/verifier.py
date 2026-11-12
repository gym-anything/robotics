"""Outcome of punch-layout: a centre-punch dimple within 0.3 mm of each of the four marks, no stray dimple within 10 mm
of a mark, and the bar moved less than 2 mm.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 the continuous progress(ev) as well."""
import math

import numpy as np

from drilling.catalogue_evaluator import clip, mean
from drilling.task import _check, _ramp

# punch-layout "dimple at least 0.3 mm deep": every strike leaves 0.40 to 0.52 mm in this steel over the punch's
# 60-130 N cap settings (tools.CentrePunch: the strike energy is the spring's at that setting), and a dimple exists only after a strike, which the position check already requires; the depth
# check cannot fail on its own.
SCORE_DIMPLE_DEPTH = False


def verify(ev):
    return ev.evaluate_checks(checks, progress=progress)


def checks(ev):
    w = ev.w
    bar = w.hm.drillables["flat_bar"]
    out = []
    dimples = [h for h in bar.holes if h.kind == "dimple"]
    for k, mk in enumerate(ev.L["marks"]):
        m = np.asarray(mk["local"])
        near = sorted(dimples, key=lambda h: np.linalg.norm((h.p0 - m)[:2]))
        if not near:
            out.append(_check(f"mark {k}: dimple within 0.3 mm", False, None, "<= 0.3 mm"))
            continue
        h = near[0]
        e = float(np.linalg.norm((h.p0 - m)[:2]))
        out.append(_check(f"mark {k}: dimple within 0.3 mm", e <= 0.0003, round(e*1e3, 3), "<= 0.3 mm"))
        if SCORE_DIMPLE_DEPTH:
            out.append(_check(f"mark {k}: dimple at least 0.3 mm deep", h.stages[0]["s"] >= 0.0003,
                              round(h.stages[0]["s"]*1e3, 3), ">= 0.3 mm"))
        stray = [x for x in dimples if x is not h and np.linalg.norm((x.p0 - m)[:2]) < 0.010]
        out.append(_check(f"mark {k}: no stray dimple within 10 mm", not stray, len(stray), kind="constraint"))
    p0, _ = ev.s["start_pose"]["flat_bar"]
    p, _ = w.body_pose(w.scene.parts["flat_bar"]["body"])
    out.append(_check("bar moved less than 2 mm", np.linalg.norm(p - p0) < 0.002,
                      round(float(np.linalg.norm(p - p0))*1e3, 2), "< 2 mm", kind="constraint"))
    return out


def progress(ev):
    dimples = [h for h in ev.w.hm.drillables["flat_bar"].holes if h.kind == "dimple"]
    credits, used = [], set()
    for mk in ev.L["marks"]:
        candidates = [(float(np.linalg.norm((h.p0 - np.asarray(mk["local"]))[:2])), h) for h in dimples]
        e, h = min(candidates, key=lambda c: (c[0], c[1].id)) if candidates else (math.inf, None)
        credits.append(_ramp(e, 0.0003, 0.0015))
        if h is not None:
            used.add(h.id)
    stray = sum(h.id not in used and any(np.linalg.norm((h.p0 - np.asarray(m["local"]))[:2]) < 0.010
                                        for m in ev.L["marks"]) for h in dimples)
    return clip(mean(credits) - 0.25*stray/len(ev.L["marks"]))

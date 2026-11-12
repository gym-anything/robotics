"""Outcome of steady-beat: beats + faster_beats strokes on the drum (32 + 16), each section's mean interval within 1 %
of its tempo (the cadence, then 10 % faster) and steady within a CV of 3 %, the stroke levels within 2 dB SD, clean
contacts and no rim stroke, nothing dropped and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict counts the drumhead's strikes
(strokes="drum": the strikes st, their levels lv and intervals ioi, the clean and no_rim checks) and calls judge(e,
g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns its
progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(params.TASK_PROGRESS fail_scale). The progress is steady_beat_progress, divided by 1 + rim strokes."""
import math

import numpy as np

from percussion import params as P
from percussion.task import _ramp


def verify(e):
    return e.verdict(judge, strokes="drum")


def judge(e, g, checks, meta, *, st, lv, ioi, ramp, TP, rim_factor):
    n1, n2 = g["beats"], g["faster_beats"]
    T1, T2 = 60/g["bpm"], 60/(g["bpm"]*(1 + g["faster_pct"]/100))
    checks["count"] = len(st) == n1 + n2
    if len(st) == n1 + n2:
        a, b = ioi[:n1 - 1], ioi[n1:]
        checks["tempo"] = abs(a.mean()/T1 - 1) <= 0.01 and abs(b.mean()/T2 - 1) <= 0.01
        checks["steady"] = a.std()/a.mean() <= 0.03 and b.std()/b.mean() <= 0.03
        checks["even_level"] = all(v is not None for v in lv) and float(np.std(lv)) <= 2.0
        meta.update(ioi_mean_s=[round(float(a.mean()), 4), round(float(b.mean()), 4)],
                    ioi_cv=[round(float(a.std()/a.mean()), 4), round(float(b.std()/b.mean()), 4)],
                    level_sd_db=round(float(np.std(lv)), 2) if all(v is not None for v in lv) else None)
    # progress (eight-task set): steady_beat_progress, divided by (1 + rim hits)
    progress, info = steady_beat_progress([s["onset_s"] for s in st], lv,
                                          [s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st], g, TP)
    progress *= rim_factor
    meta.update(info)
    return progress


def steady_beat_progress(onsets, levels, clean, goal, TP=P.TASK_PROGRESS):
    """steady-beat progress (0..1, before the rim factor). The strokes are split into the two sections at the stroke k
    that gives the most credit - with the right count and the change in its place k = beats, and a missed beat does not
    push every later interval into the wrong section:
        count_k x mean(tempo_1, tempo_2, steady_1, steady_2, level, clean)
    count_k = min(k, beats)/max(k, beats) x min(n - k, faster_beats)/max(n - k, faster_beats): strokes per section
    against those asked, so extra or missing strokes and a change in the wrong place cost. Per section, over its
    intervals (the one across the change excluded, as the check does): tempo 1 within 1 % of its stated interval, 0 at
    beat_tempo_zero_rel; steady 1 at CV <= 3 %, 0 at steadiness_zero_cv; a section with fewer than two intervals scores
    0 on both. level: the SD of the stroke levels, 1 at <= 2 dB, 0 at level_sd_zero_db, times the share of strokes
    with a level; clean: the share of strokes within the clean contact time. Returns (progress, details)."""
    n, n1, n2 = len(onsets), goal["beats"], goal["faster_beats"]
    if n < 2:
        return 0.0, dict(progress_split_stroke=None)
    T = (60/goal["bpm"], 60/(goal["bpm"]*(1 + goal["faster_pct"]/100)))
    d = np.diff(np.asarray(onsets, float))
    c1, c2 = np.r_[0.0, np.cumsum(d)], np.r_[0.0, np.cumsum(d*d)]

    def section(i0, i1, Ts):                       # the intervals d[i0:i1], against the section's interval Ts
        m = i1 - i0
        mu = (c1[i1] - c1[i0])/m if m >= 2 else 0.0
        if mu <= 0:
            return 0.0, 0.0
        cv = math.sqrt(max((c2[i1] - c2[i0])/m - mu*mu, 0.0))/mu
        return _ramp(abs(mu/Ts - 1), 0.01, TP["beat_tempo_zero_rel"]), _ramp(cv, 0.03, TP["steadiness_zero_cv"])
    lv = [v for v in levels if v is not None]
    level = _ramp(float(np.std(lv)), 2.0, TP["level_sd_zero_db"])*len(lv)/n if lv else 0.0
    clean_share = float(np.mean(clean))
    best = (-1.0, None, None)
    for k in sorted(range(1, n), key=lambda k: (abs(k - n1), k)):
        count = min(k, n1)/max(k, n1)*min(n - k, n2)/max(n - k, n2)
        (t1, s1), (t2, s2) = section(0, k - 1, T[0]), section(k, n - 1, T[1])
        p = count*float(np.mean([t1, t2, s1, s2, level, clean_share]))
        if p > best[0] + 1e-12:
            best = (p, k, dict(count=count, tempo=[t1, t2], steady=[s1, s2], level=level, clean=clean_share))
    p, k, parts = best
    return p, dict(progress_split_stroke=k,
                   progress_parts={a: [round(x, 4) for x in b] if isinstance(b, list) else round(b, 4)
                                   for a, b in parts.items()})

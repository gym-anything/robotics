"""Outcome of accent-ostinato: 4 x bars strokes on the drum (32), every fourth stroke at least accent_db (6 dB) louder
than the mean of the soft strokes, the intervals steady within a CV of 4 % and their mean within 2 % of the beat, clean
contacts and no rim stroke, nothing dropped and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict counts the drumhead's strikes
(strokes="drum": the strikes st, their levels lv and intervals ioi, the clean and no_rim checks) and calls judge(e,
g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns its
progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(params.TASK_PROGRESS fail_scale). The progress is accent_ostinato_progress, divided by 1 + rim strokes."""
import numpy as np

from percussion import params as P
from percussion.task import _ramp


def verify(e):
    return e.verdict(judge, strokes="drum")


def judge(e, g, checks, meta, *, st, lv, ioi, ramp, TP, rim_factor):
    n = 4*g["bars"]
    checks["count"] = len(st) == n
    if len(st) == n:
        # judged whenever the count is right; a stroke without a level fails `accents` (so the task cannot
        # pass without them when no level can be read)
        checks["accents"] = False
        checks["steady"] = ioi.std()/ioi.mean() <= 0.04 and abs(ioi.mean()/(60/g["bpm"]) - 1) <= 0.02
        meta.update(ioi_cv=round(float(ioi.std()/ioi.mean()), 4))
        if all(v is not None for v in lv):
            acc = np.array(lv[3::4]); un = np.array([v for i, v in enumerate(lv) if i % 4 != 3])
            checks["accents"] = float(acc.min() - un.mean()) >= g["accent_db"]
            meta.update(accent_minus_unaccented_db=round(float(acc.min() - un.mean()), 2))
    # progress (eight-task set): accent_ostinato_progress, divided by (1 + rim hits)
    progress, info = accent_ostinato_progress([s["onset_s"] for s in st], lv,
                                              [s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st], g, TP)
    progress *= rim_factor
    meta.update(info)
    return progress


def accent_ostinato_progress(onsets, levels, clean, goal, TP=P.TASK_PROGRESS):
    """accent-ostinato progress (0..1, before the rim factor):
        count x (accent_share x accent + (1 - accent_share) x mean(tempo, steady, clean))
    count = min(n, N)/max(n, N) for the N = bars x 4 strokes asked; tempo: the mean interval 1 within 2 % of the beat,
    0 at tempo_zero_rel; steady: the intervals' CV 1 at <= 4 %, 0 at steadiness_zero_cv (fewer than three strokes: 0);
    clean: the share of strokes within the clean contact time; accent: over the accent places of the pattern among the
    first min(n, N) strokes (counted from the first stroke, as the check does), each accent's level above the mean of
    the soft strokes there, 1 at >= accent_db and 0 at no contrast (a missing level 0). Returns (progress, details)."""
    pat = goal["pattern"]
    N, n = goal["bars"]*len(pat), len(onsets)
    if not n:
        return 0.0, dict(progress_parts=None)
    count = min(n, N)/max(n, N)
    d = np.diff(np.asarray(onsets, float))
    tempo = _ramp(abs(d.mean()/(60/goal["bpm"]) - 1), 0.02, TP["tempo_zero_rel"]) if n >= 2 else 0.0
    steady = _ramp(d.std()/d.mean(), 0.04, TP["steadiness_zero_cv"]) if n >= 3 else 0.0
    k = min(n, N)
    soft = [levels[i] for i in range(k) if not pat[i % len(pat)] and levels[i] is not None]
    acc = [levels[i] for i in range(k) if pat[i % len(pat)]]
    accent = float(np.mean([0.0 if v is None else float(np.clip((v - np.mean(soft))/goal["accent_db"], 0.0, 1.0))
                            for v in acc])) if acc and soft else 0.0
    clean_share = float(np.mean(clean))
    a = TP["accent_share"]
    p = count*(a*accent + (1 - a)*float(np.mean([tempo, steady, clean_share])))
    return p, dict(progress_parts=dict(count=round(count, 4), accent=round(accent, 4), tempo=round(tempo, 4),
                                       steady=round(steady, 4), clean=round(clean_share, 4)))

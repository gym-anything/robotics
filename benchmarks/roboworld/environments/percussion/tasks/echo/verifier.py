"""Outcome of echo: after the stimulus ends, as many strokes on the drum as it had, the first within two beats, their
intervals in the stimulus' ratios within ratio_tol (10 %), clean contacts and no rim stroke, nothing dropped and the
robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict counts the drumhead's strikes
(strokes="drum": the strikes st, their levels lv and intervals ioi, the clean and no_rim checks) and calls judge(e,
g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns its
progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(params.TASK_PROGRESS fail_scale). The progress: strokes against the stimulus' times the mean of the start, the
rhythm and the clean-contact share, divided by 1 + rim strokes."""
import numpy as np

from percussion import params as P


def verify(e):
    return e.verdict(judge, strokes="drum")


def judge(e, g, checks, meta, *, st, lv, ioi, ramp, TP, rim_factor):
    env = e.env
    stim = env.stimulus or {}
    so = np.array(stim.get("onsets_s", []))
    beat = 60/g["bpm"]
    end = so[-1] + g["rhythm"][-1]*beat if len(so) else 0.
    resp = [s for s in st if s["onset_s"] > end - 0.05]
    ro = np.array([s["onset_s"] for s in resp])
    checks["count"] = len(ro) == len(so)
    checks["started"] = len(ro) > 0 and ro[0] - end <= 2*beat
    if len(ro) == len(so) and len(so) >= 2:
        ratio = np.diff(ro)/np.diff(so)
        checks["rhythm"] = bool(np.all(np.abs(ratio/ratio.mean() - 1) <= g["ratio_tol"]))
        meta.update(ioi_ratio=[round(float(r), 3) for r in ratio])
    # progress: strokes against the stimulus (min/max) times the mean of the start (full within 2 beats,
    # none at start_zero_beats), the rhythm (each of the first min(n) intervals' ratio to the stimulus, graded
    # against ratio_tol, over the stimulus' interval count) and the clean-contact share; rim hits divide it
    k = min(len(ro), len(so))
    count_credit = k/max(len(ro), len(so)) if k else 0.0
    start = ramp(max(0.0, (ro[0] - end)/beat), 2.0, TP["start_zero_beats"]) if len(ro) else 0.0
    rhythm = 0.0
    if k >= 2 and len(so) >= 2:
        r = np.diff(ro[:k])/np.diff(so[:k])
        rhythm = float(np.sum([ramp(abs(x/r.mean() - 1), g["ratio_tol"], TP["ratio_zero_factor"]*g["ratio_tol"]) for x in r]))/(len(so) - 1)
    clean_share = float(np.mean([s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in resp])) if resp else 0.0
    progress = count_credit*float(np.mean([start, rhythm, clean_share]))*rim_factor
    meta.update(stimulus_onsets_s=[round(float(v), 3) for v in so], response_onsets_s=[round(float(v), 3) for v in ro])
    return progress

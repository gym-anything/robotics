"""Outcome of finger-tap: the drum tapped exactly count times (8) with a fingertip (a pad or a Dex3 tip, no tool, no
slap), at 80 BPM within 2 % with intervals steady within a CV of 4 %, every tap within 4 dB of the mezzo-piano reference
tap, clean contacts and no rim stroke, nothing dropped and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict counts the drumhead's strikes
(strokes="drum": the strikes st, their levels lv and intervals ioi, the clean and no_rim checks) and calls judge(e,
g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns its
progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(params.TASK_PROGRESS fail_scale). The progress: taps heard against taps asked times the mean of the fingertip share,
tempo, steadiness, level and clean-contact share, divided by 1 + rim strokes."""
import numpy as np

from percussion import params as P


def verify(e):
    return e.verdict(judge, strokes="drum")


def judge(e, g, checks, meta, *, st, lv, ioi, ramp, TP, rim_factor):
    checks["count"] = len(st) == g["count"]
    hand = {x["name"] for x in e.env.world.strikers if x.get("fingertip")}   # pads / Dex3 tips only
    # (the palm, the hand housing and the other finger links are strikers too: a slap is heard, not a tap)
    checks["fingertip"] = bool(st) and all(s["tools"] <= hand for s in st)
    if len(st) >= 2:
        T = 60/g["bpm"]
        checks["tempo"] = abs(ioi.mean()/T - 1) <= 0.02 and ioi.std()/ioi.mean() <= 0.04
    ref = e.reference_level("drumhead", "mp", "fingertip")
    checks["level"] = bool(lv) and all(v is not None and abs(v - ref) <= 4.0 for v in lv)
    meta.update(reference_dBA=round(ref, 2))
    # progress: taps heard against taps asked (min/max, so extra taps cost like missing ones) times the mean
    # of fingertip share, tempo, steadiness, level and clean-contact share; rim hits divide it
    n, want_n = len(st), g["count"]
    count_credit = min(n, want_n)/max(n, want_n) if n else 0.0
    tip = float(np.mean([s["tools"] <= hand for s in st])) if st else 0.0
    tempo = ramp(abs(ioi.mean()/(60/g["bpm"]) - 1), 0.02, TP["tempo_zero_rel"]) if len(st) >= 2 else 0.0
    steady = ramp(ioi.std()/ioi.mean(), 0.04, TP["steadiness_zero_cv"]) if len(st) >= 3 else tempo
    level = float(np.mean([0.0 if v is None else ramp(abs(v - ref), 4.0, TP["level_zero_db"]) for v in lv])) if lv else 0.0
    clean_share = float(np.mean([s["contact_ms"] <= P.CONTACT_MAX_S["drum"]*1e3 for s in st])) if st else 0.0
    progress = count_credit*float(np.mean([tip, tempo, steady, level, clean_share]))*rim_factor
    return progress

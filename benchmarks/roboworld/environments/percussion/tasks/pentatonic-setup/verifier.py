"""Outcome of pentatonic-setup: the F4, B4 and F5 bars in the bar tray (inside it, at most 2 cm above its floor) lying
flat (within 15 deg), every other bar still seated on its pins (seated_bars), nothing dropped and the robot still at the
end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict adds the common checks and calls
judge(e, g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and
returns its progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9
x progress (params.TASK_PROGRESS fail_scale). The progress: per bar to remove, 1 lying flat in the tray (falling to 0
at tilt_zero_deg), off_pins_credit off its pins elsewhere; their mean times the share of the other bars still seated."""
import math

import numpy as np


def verify(e):
    return e.verdict(judge)


def judge(e, g, checks, meta, *, ramp, TP, rim_factor):
    w = e.env.world
    sc = w.scene
    tray = sc.items["bar_tray"]
    pt, Rt = w.body_pose(tray["body"])
    hx, hy, hz = tray["half"]
    res = {}
    for n in g["remove"]:
        p, R = w.body_pose(f"bar_{n}")
        loc = Rt.T @ (p - pt)
        res[n] = abs(loc[0]) <= hx and abs(loc[1]) <= hy + 0.05 and loc[2] <= 2*hz + 0.02 and R[2, 2] >= math.cos(math.radians(15))
    checks["removed_to_tray"] = all(res.values())
    seated = seated_bars(e)
    checks["others_seated"] = all(v for k, v in seated.items() if k not in g["remove"])
    # progress: each bar to remove - 1 lying flat in the tray, graded down to 0 at tilt_zero_deg when in the
    # tray but tilted, off_pins_credit when off its pins but elsewhere - times the share of the other bars still
    # seated
    credit = {}
    for n in g["remove"]:
        p, R = w.body_pose(f"bar_{n}")
        loc = Rt.T @ (p - pt)
        inside = abs(loc[0]) <= hx and abs(loc[1]) <= hy + 0.05 and loc[2] <= 2*hz + 0.02
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        credit[n] = ramp(tilt, 15.0, TP["tilt_zero_deg"]) if inside else (TP["off_pins_credit"] if not seated.get(n, True) else 0.0)
    others = [v for k, v in seated.items() if k not in g["remove"]]
    progress = float(np.mean(list(credit.values())))*(float(np.mean(others)) if others else 1.0)
    meta.update(in_tray=res, seated={k: v for k, v in seated.items()}, bar_credit=credit)
    return progress


def seated_bars(e):
    """A bar is seated when it sits where it started relative to the xylophone frame (within 3 mm): on its pins,
    wherever the instrument itself has been nudged to."""
    from scipy.spatial.transform import Rotation
    w = e.env.world
    sc = w.scene
    q0 = e.env._initial["rigid"]["body_q"]
    fb = sc.bodies["xylophone_frame"]
    pf, Rf = w.body_pose(fb)
    pf0, Rf0 = q0[fb, :3].astype(float), Rotation.from_quat(q0[fb, 3:].astype(float)).as_matrix()
    out = {}
    for bb in sc.xylo_bars:
        p, _ = w.body_pose(bb["body"])
        out[bb["note"]] = float(np.linalg.norm(Rf.T @ (p - pf) - Rf0.T @ (q0[bb["body"], :3] - pf0))) <= 0.003
    return out

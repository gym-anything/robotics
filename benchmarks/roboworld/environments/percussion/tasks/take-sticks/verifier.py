"""Outcome of take-sticks: both sticks held ready together for hold_s (5 s; ready: pinched by two fingers of one hand,
the tip over the head at least 2 cm in from its edge and 3-15 cm above it), no strike on any instrument, and both sticks
back on a trap table (resting, axis near horizontal, no hand holding them, still: params.STICK_RETURN); nothing dropped
and the robot still at the end.

verify(e) evaluates an episode; e is its evaluator, task.Evaluator, whose verdict adds the common checks and calls
judge(e, g, checks, meta, ...) with the goal g (layout["goal"]): judge fills in the task's checks and meta and returns
its progress. verify returns the evaluation (success, score, checks, meta): 1 if every check passes, else 0.9 x progress
(params.TASK_PROGRESS fail_scale). The progress is take_sticks_progress.

monitor(e) is called every 10 ms (task.Evaluator.monitor): it runs the hold timers (e.hold for both sticks together,
e.hold_single per stick, stick_ready) and records each stick's midpoint (e.stick_track) for the return check."""
import math

import numpy as np

from percussion import params as P


def verify(e):
    return e.verdict(judge)


def judge(e, g, checks, meta, *, ramp, TP, rim_factor):
    checks["held_ready"] = e.hold["best_s"] >= g["hold_s"]
    strikes = e.strikes(min_speed=0.05)
    checks["no_strike"] = not strikes
    # back on either trap table, resting and still (params.STICK_RETURN), not within 30 mm of where it started
    # (in_rack)
    back = {n: on_trap_table(e, n) for n in g["sticks"]}
    checks["sticks_returned"] = all(back.values())
    single = [e.hold_single.get(n, {}).get("best_s", 0.0) for n in g["sticks"]]
    progress = take_sticks_progress(e.hold["best_s"], single, g["hold_s"], float(np.mean(list(back.values()))),
                                    len(strikes))
    meta.update(hold_best_s=round(e.hold["best_s"], 3), hold_single_best_s=[round(s, 3) for s in single],
                returned=back, strikes=len(strikes))
    return progress


def monitor(e):
    """Every 10 ms (task.Evaluator.monitor): the hold timers and the sticks' tracks."""
    env = e.env
    w = env.world
    sc = w.scene
    names = env.layout["goal"]["sticks"]
    ready = {n: stick_ready(e, n) for n in names}
    for n in names:
        hold_timer(e.hold_single.setdefault(n, dict(best_s=0.0, since=None)), ready[n], w.sim_time)
    hold_timer(e.hold, all(ready.values()), w.sim_time)
    # each stick's midpoint, for "still" in the return check (params.STICK_RETURN)
    keep = w.sim_time - P.STICK_RETURN["still_s"] - 0.02
    for n in names:
        p, R = w.body_pose(n)
        tr = e.stick_track.setdefault(n, [])
        tr.append((round(w.sim_time, 6), (p + R[:, 0]*sc.items[n]["length"]/2).tolist()))
        while tr[0][0] < keep:
            tr.pop(0)


def hold_timer(h, ok, t):
    """A continuous-hold timer: `since` the hold began (None when not holding), `best_s` its longest run."""
    if ok:
        if h["since"] is None:
            h["since"] = t
        h["best_s"] = max(h["best_s"], t - h["since"])
    else:
        h["since"] = None


def held_by(e, tool):
    """Arm whose two fingers both touch the tool (MuJoCo contacts), or None."""
    w = e.env.world
    body = w.scene.items[tool]["body"]
    for arm, fingers in w.scene.finger_bodies.items():
        f = w.pair_contact_forces(fingers, [body])
        touching = {fb for (fb, _), F in f.items() if F > 0.2}
        if len(touching) >= 2:
            return arm
    return None


def stick_ready(e, name):
    """One stick held ready: pinched by two fingers of one hand, its tip over the head (at least 2 cm in from its
    edge) and 3-15 cm above it."""
    w = e.env.world
    if "drum" not in w.scene.bodies or held_by(e, name) is None:
        return False
    pd, Rd = w.body_pose("drum")
    ps, Rs = w.body_pose(name)
    tip = ps + Rs @ np.array([P.STICK["L"] - P.STICK["tip"]/2, 0, 0])
    loc = Rd.T @ (tip - pd)
    return math.hypot(loc[0], loc[1]) <= P.DRUM_RADIUS_M - 0.02 and 0.03 <= loc[2] <= 0.15


def in_rack(e, tool):
    w = e.env.world
    t = next(t for t in w.scene.layout["tools"] if t["name"] == tool)
    p, R = w.body_pose(tool)
    return float(np.linalg.norm(p - np.asarray(t["butt"]))) <= 0.03 and abs(float(R[:, 0] @ np.asarray(t["dir"]))) >= 0.98 \
        and held_by(e, tool) is None


def on_trap_table(e, tool):
    """take-sticks' return (params.STICK_RETURN): the tool rests on either trap table (rests_on_support), no hand
    holds it, and its midpoint has stayed within still_m over the last still_s (monitor's stick_track)."""
    w = e.env.world
    S = P.STICK_RETURN
    p, R = w.body_pose(tool)
    mid = p + R[:, 0]*w.scene.items[tool]["length"]/2
    return rests_on_support(mid, R[:, 0], w.scene.layout["tool_supports"]) and held_by(e, tool) is None and \
        still_over(e.stick_track.get(tool, []), w.sim_time, S["still_s"], S["still_m"])


def take_sticks_progress(both_s, single_s, hold_s, returned_share, strikes):
    """take-sticks progress (0..1): hold x (1/2 + 1/2 x share of the sticks back on a trap table) / (1 + strikes), with
    hold = min(1, max(both_s/hold_s, 1/2 x mean_i min(1, single_s[i]/hold_s))) - both_s the longest time both sticks
    were held ready together, single_s[i] each stick's own longest. A stick held ready alone earns at most a quarter of
    the hold, both together all of it; the sticks lying where they started (nothing held) score 0, and putting them
    back earns the other half only after a hold."""
    one = 0.5*float(np.mean([min(1.0, s/hold_s) for s in single_s])) if len(single_s) else 0.0
    hold = min(1.0, max(both_s/hold_s, one))
    return hold*(0.5 + 0.5*returned_share)/(1.0 + strikes)


def rests_on_support(mid, axis, supports, kind="trap_table", R=P.STICK_RETURN):
    """Whether a tool with midpoint `mid` and unit axis `axis` lies on one of the `supports` of `kind`
    (layout.tool_rack: axis-aligned boxes, centre/half/top_z): its midpoint over the support's top, no lower than the
    top and at most max_above_top_m above it, its axis within max_tilt_deg of horizontal (params.STICK_RETURN)."""
    if abs(float(axis[2])) > math.sin(math.radians(R["max_tilt_deg"])):
        return False
    for s in supports:
        (cx, cy), (hx, hy) = s["center"], s["half"][:2]
        if s["kind"] == kind and abs(mid[0] - cx) <= hx and abs(mid[1] - cy) <= hy and \
                s["top_z"] <= mid[2] <= s["top_z"] + R["max_above_top_m"]:
            return True
    return False


def still_over(track, now, window_s, max_m):
    """Whether the sampled positions `track` [(t, xyz)] cover the last window_s before `now` and none of those lies
    farther than max_m from the latest."""
    if not track or track[0][0] > now - window_s + 1e-9:
        return False
    last = np.asarray(track[-1][1], float)
    return all(float(np.linalg.norm(np.asarray(p, float) - last)) <= max_m for t, p in track if t >= now - window_s - 1e-9)

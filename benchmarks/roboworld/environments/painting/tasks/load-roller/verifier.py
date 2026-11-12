"""Outcome of load-roller: the roller cover carries at least 50 g of paint spread around its nap (>= 80 % of the cells
at a fifth of their capacity), held still above the tray, with no drop for the last 10 s.

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria with the roller's checks (_roller_loaded) as the task's own. FILM_CHECKS = False: the
task paints nothing, so there is no protected-surface or untouched-coat check. progress(e, checks) is the task's
progress under painting-tasks-v1 and -v2 (scoring.TaskProgress, task_set_v2_verifier.TaskSetV2Progress), returning
(progress, detail): load x spread x (0.6 + 0.4 x hold)."""
import numpy as np

from painting import params as P
from painting.scoring import clip

FILM_CHECKS = False


def verify(e, settled):
    return e.measure_criteria(settled, extra=roller)


def roller(e, checks, settled):
    checks.update(_roller_loaded(e))


def _roller_loaded(e):
    from painting.roller import SAT
    r = e.paint.applicators.get("roller")
    load = r.load_mass()
    V = r.s.sum(axis=0) + r.v
    spread = float((V > 0.2*SAT*r.C0).mean())
    w = e.env.world
    bq = w.state.body_q.numpy()
    tray = w.scene.bodies.get("tray")
    above = False
    if tray is not None:
        pc = bq[r.cover][:3]; pt = bq[tray][:3]
        from painting.film import _quat_rot
        loc = _quat_rot(bq[tray][3:7]).T @ (pc - pt)
        g = P.TRAY["size"]
        above = abs(loc[0]) < g[0]/2 and abs(loc[1]) < g[1]/2 and 0.10 < loc[2] < 0.6
    quiet = (w.tick - getattr(e, "_last_drip_tick", -10**9))*P.DT >= 10.0
    return dict(load=dict(grams=round(load*1e3, 2), limit_g=50.0, ok=load >= 0.050),
                spread=dict(fraction=round(spread, 3), limit=0.8, ok=spread >= 0.8),
                held_above_tray=dict(ok=bool(above)),
                no_drip_10s=dict(seconds_since_last_drop=round((w.tick - getattr(e, "_last_drip_tick", -10**9))*P.DT, 2)
                                 if hasattr(e, "_last_drip_tick") else None, ok=bool(quiet)))


def progress(e, checks):
    roller = e.paint.applicators["roller"]
    load = clip(roller.load_mass()/0.050)
    volume = roller.s.sum(axis=0) + roller.v
    # A partially wetted nap cell earns partial credit before crossing
    # the spread verdict's 20%-of-capacity threshold.
    spread = clip(float(np.clip(volume/(0.2*P.ROLLER["sat"]*roller.C0), 0.0, 1.0).mean())/0.8)
    seconds = (e.env.world.tick - getattr(e, "_last_drip_tick", -10**9))*P.DT
    hold = float(checks["held_above_tray"]["ok"])*clip(seconds/10.0)
    progress = load*spread*(0.6 + 0.4*hold)
    detail = dict(load=load, spread=spread, hold=hold)
    return progress, detail

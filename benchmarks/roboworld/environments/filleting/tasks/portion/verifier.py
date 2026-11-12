"""Outcome of portion: the loin of the salmon half side cut into 150 +- 7 g portions with straight cuts, each full
portion weighed on the bench scale, no tag ends.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator: task.Evaluator
for the released checks (released_checks), or the task-set evaluator of filleting-tasks-v1 or -v2
(task_set_evaluator.py, task_set_v2_evaluator.py), whose cutting grading (_cut_task) reads PRODUCT, released_checks,
product and progress below; monitor(e) records the scale's steady readings after every tick of a task-set episode."""
import numpy as np

from filleting import layout as LY
from filleting.task import NOTHING_MADE, TIME_LIMIT_S, check
from filleting.task_set_progress import clip, ramp

PRODUCT = "slices"


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(released_checks)
    return e.evaluate_graded()


def released_checks(e):
    """Portion the side: every full portion (all pieces but the two end pieces along the side) 150 +- 7 g (IMPS
    tolerance below 170 g); blade surfaces within 10 deg of perpendicular to the board (recorded cut surface,
    area-weighted, rest space); no loose piece of 1 to 5 g (a tag end cut free); every full portion weighed: the
    bench scale showed a steady reading within 2 g of its mass at some time."""
    env = e.env
    lab, frags, main = e._parts()
    pieces = sorted(((c, f) for c, f in frags.items() if f["mass"] >= 5e-3),
                    key=lambda cf: float(e.env.world.tissue.hX[e.env.world.tissue.htets[cf[1]["tets"]]].mean((0, 1))[0]))
    full = pieces[1:-1] if len(pieces) >= 3 else []
    n, a = e._sweep_normals()
    tilt = float(np.degrees(np.arcsin(np.clip(np.sum(np.abs(n[:, 2])*a)/max(a.sum(), 1e-15), 0, 1)))) if len(a) else None
    tags = [c for c, f in frags.items() if 1e-3 <= f["mass"] < 5e-3]
    log = list(env.scale_log)
    weighed = []
    for c, f in full:
        ok = False
        for i in range(2, len(log)):
            win = [r for _, r in log[i - 2:i + 1]]
            if max(win) - min(win) < 1. and abs(np.mean(win) - f["mass"]*1e3) <= 2.:
                ok = True; break
        weighed.append(ok)
    masses = [f["mass"]*1e3 for _, f in full]
    in_tol = bool(masses) and all(143. <= m_ <= 157. for m_ in masses)
    checks = dict([check("full_portions", len(full), 1, "count", len(full) >= 1),
                   check("portions_150_7g", float(max((abs(m_ - 150.) for m_ in masses), default=np.nan)), 7.,
                         "g worst deviation", in_tol),
                   check("cut_tilt_deg", tilt, 10., "deg from vertical (area-weighted)",
                         bool(full) and tilt is not None and tilt <= 10., "" if full else NOTHING_MADE),
                   check("no_tag_ends", len(tags), 0, "pieces of 1-5 g", not tags),
                   check("weighed", float(np.mean(weighed)) if weighed else 0., 1., "fraction of full portions",
                         bool(weighed) and all(weighed)),
                   check("time", env.sim_time, TIME_LIMIT_S["B5"], "s", env.sim_time <= TIME_LIMIT_S["B5"])])
    dm = e._common_cut(checks, frags, [c for c, _ in pieces])
    return checks, dict(**dm, portion_g=masses, end_pieces_g=[f["mass"]*1e3 for _, f in (pieces[:1] + pieces[-1:])],
                        scale_last_g=log[-1][1] if log else None)


def product(e, ref, reg):
    """The number of portions the start allows."""
    ts, nt = e.env.world.tissue, e.graph.nt
    # Leave two 5 g end pieces, as the product definition does. A fixed
    # denominator from this seed's initial loin mass, never remaining mass.
    mass = float(ts.htmass[:nt][ref == e._ref_id("loin")].sum())
    return max(1, int((mass - .010)/.150))


def progress(e, checks, metrics, frags, partial, tilt):
    """Raw progress: the full portions' masses, the best unfinished cut, the cut tilt and the share weighed."""
    pieces = sorted(((c, f) for c, f in frags.items() if f["mass"] >= .005),
                    key=lambda cf: float(e.graph.centres[cf[1]["tets"], 0].mean()))
    full = pieces[1:-1]
    weighed = [tuple(sorted(int(t) for t in f["tets"])) in e.weighed for _, f in full]
    share = float(np.mean(weighed)) if weighed else 0.
    checks["weighed"] = check("weighed", share, 1., "fraction; each piece physically on scale",
                               bool(weighed) and all(weighed))[1]
    quality = sum(ramp(abs(f["mass"]*1000 - 150.), 7., 40.) for _, f in full)
    raw = clip((quality + .5*partial)/e.target_count)*tilt*(.8 + .2*share)
    metrics["weighed_piece_count"] = sum(weighed)
    return raw


def monitor(e):
    """After every tick of a task-set episode: every 50th tick, the bench scale's reading (_monitor_scale)."""
    if e.env.steps % 50 == 0:
        _monitor_scale(e)


def _monitor_scale(e):
    w = e.env.world
    reading = w.scale_reading_g()
    if reading is None or reading < 5.:
        e.weigh_windows.clear()
        return
    _, frags, _ = e._parts()
    x = w.tissue.x.numpy()
    supported = []
    for f in frags.values():
        if f["mass"] < .005:
            continue
        p = x[e.fragment_nodes(f)]
        if (e.in_footprint(p, LY.SCALE).all() and
                abs(float(p[:, 2].min()) - LY.TABLE["top_z"] - LY.SCALE["size"][2]) <= .003 and
                not e._robot_touch(p).any() and not e._blade_touch(p).any()):
            supported.append(f)
    # One independently supported piece must account for the scale's entire load.
    if len(supported) != 1:
        e.weigh_windows.clear()
        return
    f = supported[0]
    key = tuple(sorted(int(t) for t in f["tets"]))
    win = e.weigh_windows.get(key, []) + [float(reading)]
    win = win[-3:]
    e.weigh_windows = {key: win}
    if len(win) == 3 and max(win) - min(win) < 1. and abs(np.mean(win) - f["mass"]*1000) <= 2.:
        e.weighed.add(key)

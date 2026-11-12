"""Outcome of expose-line: at least 150 mm of the lateral's crown uncovered, no strike on it and at most 0.5 kg of soil
disturbed outside the marked tolerance zone (the task contracts: also 80 % of the spoil on the tarp).

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
import numpy as np

from soil_excavation.task import ORIGIN_BLOCK, crit, diag
from soil_excavation.task_set_v2 import clip


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _expose_line(e, lg, final)
    return read(e, lg)


def start_cover(e, lg):
    """The settled start's soil cover over each 1 cm station of the crown (crown_cover), which the task contracts'
    reading measures the uncovering against (e.start_cover); taken when their evaluator is installed."""
    return crown_cover(lg, e.env.spec, lg.pos)


def _expose_line(e, lg, final):
    sp = e.env.spec
    pp = sp["pipe"]
    crown = pp["z"] + pp["r"]
    b = sp["block"]
    xs = np.arange(b["lo"][0] + 0.05, b["hi"][0] - 0.05, 0.01)
    above = (np.abs(lg.pos[:, 1] - pp["y"]) < pp["r"]) & (lg.pos[:, 2] > crown) & (lg.pos[:, 2] < crown + 0.01 + 0.5*lg.h)
    covered = np.zeros(len(xs), bool)
    ix = np.floor((lg.pos[above, 0] - xs[0])/0.01).astype(int)
    ix = ix[(ix >= 0) & (ix < len(xs))]
    covered[ix] = True
    run, best = 0, 0
    for cv in covered:
        run = 0 if cv else run + 1
        best = max(best, run)
    exposed = best*0.01
    zone = sp["tolerance_zone"]
    outside = (lg.origin == ORIGIN_BLOCK) & ((lg.pos0[:, 1] < zone[0] - 0.0) | (lg.pos0[:, 1] > zone[1])) & \
        (np.linalg.norm(lg.pos - lg.pos0, axis=1) > 0.03)
    m_out = float(lg.mass[outside].sum())
    w = e.env.world
    c = [crit("crown_exposed", exposed, exposed >= 0.15, ">= 150 mm of crown with no soil within 10 mm above it", "m"),
         crit("no_strikes", len(w.strikes), len(w.strikes) == 0, "0 strikes (> 30 N or > 0.2 m/s on the lateral)"),
         crit("outside_zone", m_out, m_out <= 0.5, "<= 0.5 kg disturbed outside the 18 in tolerance zone", "kg")]
    return c, [min(1., exposed/0.15), float(len(w.strikes) == 0), max(0., 1 - m_out/2.)]


# soil-tasks-v1's and soil-tasks-v2's reading (the same in task_set.py and task_set_v2.py)
def read(e, lg):
    c, _ = _expose_line(e, lg, False)
    cover = crown_cover(lg, e.env.spec, lg.pos)
    run = best = 0
    for thickness in cover:
        run = run + 1 if thickness == 0. else 0
        best = max(best, run)
    exposed = best*.01
    c[0] = crit("crown_exposed", exposed, exposed >= .15, ">= 150 mm contiguous crown with no overhead soil", "m")
    dug, tarp, _ = e._spoil(lg)
    frac = tarp/max(dug, 1e-9)
    c.append(crit("spoil_on_tarp", frac, frac >= .8 and dug > .5, ">= 80% of excavated mass; > 0.5 kg excavated"))
    # Compare each 15 cm window with its own settled start cover, then take
    # the best. No command, action order, hand or chosen digging method enters.
    start = e.start_cover
    uncover = max((clip(1. - cover[i:i + 15].mean()/start[i:i + 15].mean())
                   if start[i:i + 15].mean() > 0. else 0. for i in range(len(cover) - 14)), default=0.)
    zone = e.env.spec["tolerance_zone"]
    outside = (lg.origin == ORIGIN_BLOCK) & ((lg.pos0[:, 1] < zone[0]) | (lg.pos0[:, 1] > zone[1])) & \
              (np.linalg.norm(lg.pos - lg.pos0, axis=1) > .03)
    m_out = float(lg.mass[outside].sum())
    raw = clip((.5*uncover + .5*clip(exposed/.15))*2.**(-len(e.env.world.strikes))*
               (.8 + .2*clip(frac/.8)) - .25*max(0., m_out - .5)/2.)
    return c, raw, [diag("cover_removed", uncover, "best 15 cm window's reduction from settled start cover")]


def crown_cover(lg, spec, positions):
    """Maximum soil thickness over each 1 cm station of the pipe's crown.

    Read the full overhead column. A void beside the crown beneath a floating
    roof of soil is not an exposed utility (the old 1 cm shell missed that roof).
    """
    pp, b = spec["pipe"], spec["block"]
    xs = np.arange(b["lo"][0] + .05, b["hi"][0] - .05, .01)
    crown = pp["z"] + pp["r"]
    cover = np.zeros(len(xs))
    sel = (np.abs(positions[:, 1] - pp["y"]) < pp["r"]) & (positions[:, 2] + .55*lg.h > crown)
    x = positions[sel]
    ix = np.floor((x[:, 0] - xs[0])/.01).astype(int)
    valid = (ix >= 0) & (ix < len(xs))
    np.maximum.at(cover, ix[valid], np.maximum(0., x[valid, 2] + .55*lg.h - crown))
    return cover

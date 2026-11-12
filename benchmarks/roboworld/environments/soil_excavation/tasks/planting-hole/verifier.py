"""Outcome of planting-hole: a hole 160 +- 20 mm deep at the mark and 0.33-0.50 m across, with at most 0.3 kg of loose
crumbs on its floor, at least 80 % of the spoil on the tarp and no more than 0.2 kg on the walkway.

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
from soil_excavation.task import crit
from soil_excavation.task_set_v2 import band, clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _planting_hole(e, lg, final)
    return read(e, lg)


def _planting_hole(e, lg, final):
    sp = e.env.spec
    depth, dia, crumbs = e._hole(lg, sp["target"], 0.16, 0.02, (0.33, 0.50))
    m_dug, m_tarp, m_walk = e._spoil(lg)
    frac = m_tarp/max(m_dug, 1e-9)
    c = [crit("depth_at_mark", depth, abs(depth - 0.16) <= 0.02, "0.16 +- 0.02 m (#1 root ball height)", "m"),
         crit("top_diameter", dia, 0.33 <= dia <= 0.50, "0.33-0.50 m where depth >= 80 mm (2-3 x 165 mm)", "m"),
         crit("loose_on_floor", crumbs, crumbs <= 0.3, "<= 0.3 kg of loose crumbs on the floor (ANSI A300: undug base)", "kg"),
         crit("spoil_on_tarp", frac, frac >= 0.8 and m_dug > 0.5, ">= 80 % of the excavated mass", "-"),
         crit("walkway_clean", m_walk, m_walk <= 0.2, "<= 0.2 kg on the walkway", "kg")]
    return c, [min(1., depth/0.16) if depth <= 0.16 else max(0., 1 - (depth - 0.16)/0.1), min(1., dia/0.33),
               max(0., 1 - crumbs/1.0), min(1., frac/0.8), max(0., 1 - m_walk/1.0)]


# soil-tasks-v1's and soil-tasks-v2's reading (the same in task_set.py and task_set_v2.py)
def read(e, lg):
    depth, diameter, crumbs = e._hole(lg, e.env.spec["target"], .16, .02, (.33, .50))
    dug, tarp, walk = e._spoil(lg)
    frac = tarp/max(dug, 1e-9)
    c = [crit("depth_at_mark", depth, abs(depth - .16) <= .02, "0.16 +/- 0.02 m", "m"),
         crit("top_diameter", diameter, .33 <= diameter <= .50, "0.33-0.50 m at 80 mm depth", "m"),
         crit("loose_on_floor", crumbs, crumbs <= .3, "<= 0.3 kg displaced loose crumbs", "kg"),
         crit("spoil_on_tarp", frac, frac >= .8 and dug > .5, ">= 80% of excavated mass; > 0.5 kg excavated"),
         crit("walkway_clean", walk, walk <= .2, "<= 0.2 kg", "kg")]
    shape = min(band(depth, .14, .18, 0., .26), band(diameter, .33, .50, 0., .65))
    raw = clip(shape*(.8 + .2*ramp(crumbs, .3, 1.))*(.7 + .3*clip(frac/.8)) - .2*max(0., walk - .2)/.8)
    return c, raw, []

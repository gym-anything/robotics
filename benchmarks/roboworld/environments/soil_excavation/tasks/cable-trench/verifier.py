"""Outcome of cable-trench: a trench at least 16 cm deep over 90 % of the 1.5 m line, 6-20 cm wide and centred within 5
cm of it, at least 90 % of the spoil on the lawn-side tarp, the walkway clean and the robot's base never off the slab.

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
import numpy as np

from soil_excavation import layout as L
from soil_excavation.task import crit
from soil_excavation.task_set_v2 import band, clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _cable_trench(e, lg, final)
    return read(e, lg)


def _cable_trench(e, lg, final):
    (x0, y0), (x1, y1) = e.env.spec["target_line"]
    rect = ((x0 - 0.05, y0 - 0.25), (x1 + 0.05, y0 + 0.25))
    top, org = lg.heightfield(rect)
    depth = L.GRADE - top
    depth = np.where(np.isnan(depth), 0.5, depth)
    xs = org[0] + (np.arange(top.shape[0]) + 0.5)*org[2]
    ys = org[1] + (np.arange(top.shape[1]) + 0.5)*org[2]
    along = (xs >= x0) & (xs <= x1)
    prof = depth[along]
    deep = prof.max(axis=1)
    # centre of the deepest band (>= 80 % of the row's depth): the argmax of a flat-bottomed trench is its first
    # column, an edge (a correct 12 cm trench on the line would fail 'on_line')
    band = prof >= 0.8*deep[:, None]
    centre = (band*ys[None, :]).sum(1)/np.maximum(band.sum(1), 1)
    centre_off = np.abs(centre - y0)
    ok_depth = float(np.mean(deep >= 0.16))
    widths = []
    for k in range(prof.shape[0]):
        half = deep[k]/2
        row = prof[k] >= max(half, 0.02)
        widths.append(row.sum()*org[2])
    widths = np.asarray(widths)
    ok_width = float(np.mean((widths >= 0.06) & (widths <= 0.20)))
    ok_centre = float(np.mean(centre_off <= 0.05))
    m_dug, m_tarp, m_walk = e._spoil(lg)
    frac = m_tarp/max(m_dug, 1e-9)
    c = [crit("depth_coverage", ok_depth, ok_depth >= 0.9, ">= 90 % of the 1.5 m at >= 0.16 m (NEC 300.5: 150 mm cover)", "-"),
         crit("width_coverage", ok_width, ok_width >= 0.9, "6-20 cm wide at half depth over >= 90 %", "-"),
         crit("on_line", ok_centre, ok_centre >= 0.9, "deepest point within 5 cm of the line over >= 90 %", "-"),
         crit("spoil_on_tarp", frac, frac >= 0.9 and m_dug > 0.5, ">= 90 % of the spoil on the lawn-side tarp", "-"),
         crit("walkway_clean", m_walk, m_walk <= 0.2, "<= 0.2 kg on the walkway", "kg"),
         crit("base_on_slab", e.base_off_slab, e.base_off_slab <= 0.0, "base footprint never beyond the slab", "m")]
    return c, [ok_depth, ok_width, ok_centre, min(1., frac/0.9), max(0., 1 - m_walk), float(e.base_off_slab <= 0)]


# soil-tasks-v1's and soil-tasks-v2's reading (the same in task_set.py and task_set_v2.py)
def read(e, lg):
    c, _ = _cable_trench(e, lg, False)  # unchanged success geometry and slab history
    (x0, y0), (x1, _) = e.env.spec["target_line"]
    top, org = lg.heightfield(((x0 - .05, y0 - .25), (x1 + .05, y0 + .25)))
    depth = np.where(np.isnan(top), .5, L.GRADE - top)
    xs = org[0] + (np.arange(top.shape[0]) + .5)*org[2]
    ys = org[1] + (np.arange(top.shape[1]) + .5)*org[2]
    rows = depth[(xs >= x0) & (xs <= x1)]
    deep = rows.max(axis=1)
    deepest = rows >= .8*deep[:, None]
    centre = (deepest*ys[None, :]).sum(1)/np.maximum(deepest.sum(1), 1)
    width = (rows >= np.maximum(deep/2, .02)[:, None]).sum(1)*org[2]
    # Weight each row's alignment by that row's depth. Work on another row
    # cannot cancel an off-line excavation; flat untouched rows earn zero.
    shape = np.mean([clip(d/.16)*ramp(abs(y - y0), .05, .15)*
                     (.8 + .2*band(w, .06, .20, 0., .30)) for d, y, w in zip(deep, centre, width)])
    dug, tarp, walk = e._spoil(lg)
    raw = clip(shape*(.7 + .3*clip(tarp/max(dug, 1e-9)/.9)) - .2*max(0., walk - .2))
    return c, raw, []

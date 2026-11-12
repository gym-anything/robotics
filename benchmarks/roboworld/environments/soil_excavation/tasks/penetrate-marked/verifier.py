"""Outcome of penetrate-marked: the spade's cutting edge at least 15 cm below grade within 30 mm of the mark, the blade
within 10 deg of vertical and both hands on the spade.

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
import numpy as np

from soil_excavation import layout as L
from soil_excavation.catalogue import CONTRACT as TASKS_V1
from soil_excavation.catalogue_v2 import CONTRACT as TASKS_V2
from soil_excavation.task import _pose, _tilt_deg, crit, tool_held
from soil_excavation.task_set_v2 import clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _penetrate_marked(e, lg, final)
    return {TASKS_V1: read_v1, TASKS_V2: read_v2}[contract](e, lg)


def _penetrate_marked(e, lg, final):
    sp = e.env.spec
    p, R = _pose(e.env, "spade")
    mark = np.asarray(sp["target"])
    depth = L.GRADE - p[2]
    off = float(np.linalg.norm(p[:2] - mark))
    tilt = _tilt_deg(R)
    held = tool_held(e.env, "spade")
    c = [crit("edge_depth", depth, depth >= 0.15, ">= 0.15 m below grade (2/3 of the 290 mm head)", "m"),
         crit("edge_on_mark", off, off <= 0.03, "<= 30 mm from the mark (1.5 voxels)", "m"),
         crit("blade_vertical", tilt, tilt <= 10.0, "<= 10 deg", "deg"),
         crit("both_hands", held, all(held.values()), "both hands on the spade")]
    return c, [max(0., min(1., depth/0.15)), max(0., 1 - off/0.1), max(0., 1 - tilt/30.), float(all(held.values()))]


# soil-tasks-v1's reading (task_set.py)
def read_v1(e, lg):
    p, r = _pose(e.env, "spade")
    depth, off, tilt = L.GRADE - p[2], float(np.linalg.norm(p[:2] - e.env.spec["target"])), _tilt_deg(r)
    held = tool_held(e.env, "spade")
    # An empty contact mapping is not a two-handed hold.
    both = len(held) == 2 and all(held.values())
    c = [crit("edge_depth", depth, depth >= .15, ">= 0.15 m below grade", "m"),
         crit("edge_on_mark", off, off <= .03, "<= 30 mm from the mark", "m"),
         crit("blade_vertical", tilt, tilt <= 10., "<= 10 degrees", "deg"),
         crit("both_hands", held, both, "both hands on the spade")]
    raw = clip(depth/.15)*ramp(off, .03, .10)*ramp(tilt, 10., 30.)*(.8 + .2*both)
    return c, raw, []


# soil-tasks-v2's reading (task_set_v2.py): no hand factor in the progress
def read_v2(e, lg):
    p, r = _pose(e.env, "spade")
    depth, off, tilt = L.GRADE - p[2], float(np.linalg.norm(p[:2] - e.env.spec["target"])), _tilt_deg(r)
    held = tool_held(e.env, "spade")
    # An empty contact mapping is not a two-handed hold.
    both = len(held) == 2 and all(held.values())
    c = [crit("edge_depth", depth, depth >= .15, ">= 0.15 m below grade", "m"),
         crit("edge_on_mark", off, off <= .03, "<= 30 mm from the mark", "m"),
         crit("blade_vertical", tilt, tilt <= 10., "<= 10 degrees", "deg"),
         crit("both_hands", held, both, "both hands on the spade")]
    # Progress from the spade and soil alone: both hands are a success criterion, not credit.
    raw = clip(depth/.15)*ramp(off, .03, .10)*ramp(tilt, 10., 30.)
    return c, raw, []

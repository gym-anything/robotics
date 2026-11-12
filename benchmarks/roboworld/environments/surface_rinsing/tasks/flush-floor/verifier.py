"""Outcome of flush-floor: no 1 cm gauge of the bay floor above ten times the visible-residue limit, the hose valve
docked in its wall holster and the dunnage shelf dry; no water running and nothing in the hands.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure), gate(name, ok, detail) records one. Under the task-set contracts task_set_measure(e, data)
adds the bay floor's soil gauges and their areas to the task-set measures (data, m["task_set"]; task_set/evaluator.py)
and task_set_progress(ctx) gives the progress (task_set/scoring.py; ctx holds the measures and terms that function reads
for every task). verify(e, settle_s) gives the episode's verdict, the dict of task.Evaluator.evaluate_settled (the
settle branch of at most settle_s seconds, None: the default).
"""
import numpy as np

from surface_rinsing import params as P
from surface_rinsing.task import gauges
from surface_rinsing.task_set.scoring import reduction

C = P.CRITERIA


def criteria(e, m, gate):
    cont = m["contamination"]
    ws = e.env.water
    f = e._fields()
    ch = ws.static_charts["bay_floor"]
    cells = np.arange(ch.first, ch.first + ch.count)
    A = ws.surf.area[cells]
    g = gauges(ws.surf.pos[cells], A, f["soil"][cells] + f["sus"][cells], ws.surf.nrm[cells])
    surf = float(((f["s"][cells] + f["gamma"][cells])*A).sum())
    gate("floor_flushed", float(g.max())*1e5 <= P.VRL*1e5*10,
         dict(gauge_max_ug_cm2=round(float(g.max())*1e5, 3), surfactant_mg=round(surf*1e6, 3)))
    gate("valve_parked", all(d <= C["park_m"] for d in m["valves_from_park_m"].values()), m["valves_from_park_m"])
    gate("contained", cont.get("dunnage_shelf", {}).get("water_g", 0.0) <= 1.0, cont)


def task_set_measure(e, data):
    ws = e.env.water
    ch = ws.static_charts["bay_floor"]
    cells = np.arange(ch.first, ch.first + ch.count)
    a, pos, nrm = ws.surf.area[cells], ws.surf.pos[cells], ws.surf.nrm[cells]
    f = e._fields()
    data["floor_gauges"] = (gauges(pos, a, f["soil"][cells] + f["sus"][cells], nrm)*1e5).tolist()
    _, inv = np.unique(np.floor((pos + 0.005*nrm)/0.01).astype(np.int64), axis=0, return_inverse=True)
    data["floor_areas"] = np.bincount(inv.ravel(), weights=a).tolist()


def task_set_progress(ctx):
    start, m, passed, released, shelf = ctx.start, ctx.m, ctx.passed, ctx.released, ctx.shelf
    # A binary share of clean gauges jumps at the threshold. Area-weighted
    # log reductions keep partial cleaning visible within each 1 cm gauge.
    g0, g = start["task_set"]["floor_gauges"], m["task_set"]["floor_gauges"]
    area = start["task_set"]["floor_areas"]
    dirty = [i for i, x in enumerate(g0) if x > 10.0]
    a = sum(area[i]*reduction(g0[i], g[i], 10.0) for i in dirty)/sum(area[i] for i in dirty) if dirty else 0.0
    k = passed("valve_parked") * released
    units = shelf
    value = (a + a*k)/2.0 - units
    terms = dict(clean_area=a, docked_released=k)
    return value, terms, units


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)

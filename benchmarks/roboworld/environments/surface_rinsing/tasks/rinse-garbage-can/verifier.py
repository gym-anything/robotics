"""Outcome of rinse-garbage-can: the can free of detergent and visibly clean, emptied and upright, the hose valve docked
in its wall holster and the dunnage shelf dry; no water running and nothing in the hands.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure), gate(name, ok, detail) records one. Under the task-set contracts task_set_measure(e, data)
adds the can's soap limit to the task-set measures (data, m["task_set"]; task_set/evaluator.py) and
task_set_progress(ctx) gives the progress (task_set/scoring.py; ctx holds the measures and terms that function reads for
every task). verify(e, settle_s) gives the episode's verdict, the dict of task.Evaluator.evaluate_settled (the settle
branch of at most settle_s seconds, None: the default).
"""
from surface_rinsing import params as P
from surface_rinsing.task_set.scoring import reduction

C = P.CRITERIA


def criteria(e, m, gate):
    items, cont = m["items"], m["contamination"]
    r = items["can"]
    ws = e.env.water
    cells = ws.item_cells["can"]
    A = ws.surf.area[cells]
    mean = r["surfactant_ug"]*1e-9/float(A.sum())
    gate("soap_free", mean <= C["surf_per_item"]/0.085 and r["surfactant_gauge_max_mg_m2"] <= C["surf_gauge"]*1e6
         and r["foam_mL"] < 1e-3, dict(mean_mg_m2=round(mean*1e6, 4), gauge_max=r["surfactant_gauge_max_mg_m2"],
                                        foam_mL=r["foam_mL"]))
    gate("visibly_clean", r["soil_gauge_max_ug_cm2"] <= P.VRL*1e5, dict(gauge_max=r["soil_gauge_max_ug_cm2"]))
    gate("emptied_upright", r.get("free_water_mL", 0.0) <= C["drained_can"]*1e6 and r["tilt_deg"] < C["place_tilt_deg"],
         dict(free_water_mL=r.get("free_water_mL", 0.0), pooled_mL=r.get("pooled_mL", 0.0), tilt_deg=r["tilt_deg"]))
    gate("valve_parked", all(d <= C["park_m"] for d in m["valves_from_park_m"].values()), m["valves_from_park_m"])
    gate("contained", cont.get("dunnage_shelf", {}).get("water_g", 0.0) <= 1.0, cont)


def task_set_measure(e, data):
    ws = e.env.water
    data["can_soap_limit_ug"] = float(ws.surf.area[ws.item_cells["can"]].sum())*P.CRITERIA["surf_per_item"]/0.085*1e9


def task_set_progress(ctx):
    m, passed, items, initial, released, shelf = ctx.m, ctx.passed, ctx.items, ctx.initial, ctx.released, ctx.shelf
    r = reduction(initial["can"]["surfactant_ug"], items["can"]["surfactant_ug"], m["task_set"]["can_soap_limit_ug"])
    d = reduction(initial["can"]["soil_gauge_max_ug_cm2"], items["can"]["soil_gauge_max_ug_cm2"], 1.0)
    c = (r + d)/2.0
    e = passed("emptied_upright")
    k = passed("valve_parked") * released
    units = shelf
    value = (c + c*e + c*e*k)/3.0 - units
    terms = dict(rinsed=r, debris_removed=d, clean=c, emptied_upright=e, docked_released=k)
    return value, terms, units


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)

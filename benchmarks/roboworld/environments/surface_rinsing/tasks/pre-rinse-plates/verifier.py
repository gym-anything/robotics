"""Outcome of pre-rinse-plates: every plate visibly clean and still in the peg rack, the spray valve hung back up, the
clean dish table and the floor kept clean and dry; the water shut off and nothing in the hands. Under
surface-rinsing-tasks-v2 the water that left the spray valve must also have been mixed to the instruction's temperature.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure), gate(name, ok, detail) records one. Under surface-rinsing-tasks-v1 task_set_progress(ctx)
gives the progress (task_set/scoring.py; ctx holds the measures and terms that function reads for every task). Under
surface-rinsing-tasks-v2 (task_set_v2_evaluator.py) the evaluator tracks the outlet SPRAY_OUTLET, task_set_v2_measure(e,
m) adds its flow-weighted mean temperature to m["task_set"], task_set_v2_criteria(e, m, criteria) adds the
spray_temperature gate to the criteria dict and task_set_v2_progress(start, m, criteria) gives the temperature-graded
progress (task_set_v2.progress). verify(e, settle_s) gives the episode's verdict, the dict of
task.Evaluator.evaluate_settled (the settle branch of at most settle_s seconds, None: the default).
"""
from surface_rinsing import params as P
from surface_rinsing.task_set.scoring import clip, excess, reduction

C = P.CRITERIA
SPRAY_OUTLET = "prsv"          # the pre-rinse station's only outlet

# pre-rinse-plates: "Mix the water to about 49 C (110-120 F)". The spec's spray_temperature gate: the flow-weighted
# mean temperature of the water that left the spray valve lies in this band; progress is multiplied by a grade that is
# 1 inside the band and falls linearly to 0 at SPRAY_GRADE_C outside it (0 when no water has left the valve).
SPRAY_BAND_C = (40.5, 51.5)
SPRAY_GRADE_C = 10.0


def criteria(e, m, gate):
    items, cont = m["items"], m["contamination"]
    plates = [n for n in items if n.startswith("plate_")]
    clean = {n: items[n]["soil_gauge_max_ug_cm2"] for n in plates}
    gate("visibly_clean", all(v <= P.VRL*1e5 for v in clean.values()), clean)
    gate("valve_parked", all(d <= C["park_m"] for d in m["valves_from_park_m"].values()), m["valves_from_park_m"])
    gate("contained", cont.get("clean_table", {}).get("gauge_max_ug_cm2", 0) <= P.VRL*1e5 and
         cont["floor_water_mL"] <= C["floor_water"]*1e6, cont)
    inrack = {n: e._in_peg_rack(n, "peg_rack") for n in plates}
    gate("placed", all(inrack.values()), inrack)


def task_set_progress(ctx):
    criteria, passed, items, initial = ctx.criteria, ctx.passed, ctx.items, ctx.initial
    released, floor, soil = ctx.released, ctx.floor, ctx.soil
    names = [n for n in initial if n.startswith("plate_")]
    credits = {n: reduction(initial[n]["soil_gauge_max_ug_cm2"], items[n]["soil_gauge_max_ug_cm2"], 1.0)
               for n in names}
    c = sum(credits.values())/len(names)
    k = passed("valve_parked") * released
    units = (sum(not v for v in criteria["placed"]["detail"].values()) + soil("clean_table") + floor)/len(names)
    value = (c + c*k)/2.0 - units
    terms = dict(clean=credits, mean_clean=c, docked_released=k)
    return value, terms, units


def task_set_v2_measure(e, m):
    v = e.spray["volume_L"]
    m["task_set"]["spray"] = dict(volume_L=v, mean_C=e.spray["heat_L_C"]/v if v > 0.0 else None)


def task_set_v2_criteria(e, m, criteria):
    spray = m["task_set"]["spray"]
    t = spray["mean_C"]
    criteria["spray_temperature"] = dict(
        passed=bool(t is not None and SPRAY_BAND_C[0] <= t <= SPRAY_BAND_C[1]),
        detail=dict(spray, band_C=list(SPRAY_BAND_C), rule="flow-weighted mean temperature of the water that "
                    "left the spray valve since reset"))


def temperature_grade(mean_c):
    if mean_c is None:
        return 0.0
    lo, hi = SPRAY_BAND_C
    return clip(1.0 - max(0.0, lo - mean_c, mean_c - hi)/SPRAY_GRADE_C)


def task_set_v2_progress(start, m, criteria):
    passed = lambda name: float(criteria[name]["passed"])
    items, initial = m["items"], start["items"]
    cont = m["contamination"]
    released = passed("water_off_and_hands_free")
    floor = excess(cont["floor_water_mL"], 50.0)
    soil = lambda name: excess(cont.get(name, {}).get("gauge_max_ug_cm2", 0.0), 1.0)
    names = [n for n in initial if n.startswith("plate_")]
    credits = {n: reduction(initial[n]["soil_gauge_max_ug_cm2"], items[n]["soil_gauge_max_ug_cm2"], 1.0)
               for n in names}
    c = sum(credits.values())/len(names)
    k = passed("valve_parked") * released
    spray = m["task_set"]["spray"]
    g = temperature_grade(spray["mean_C"])
    units = (sum(not v for v in criteria["placed"]["detail"].values()) + soil("clean_table") + floor)/len(names)
    value = (c + c*k)/2.0*g - units
    terms = dict(clean=credits, mean_clean=c, docked_released=k, spray_mean_C=spray["mean_C"],
                 spray_volume_L=spray["volume_L"], temperature_grade=g)
    return dict(progress=clip(value), terms=terms, collateral_units=units)


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)

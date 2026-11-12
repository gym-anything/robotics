"""Outcome of rinse-tomatoes: all fifteen tomatoes visibly clean and still in the colander, the colander drained and set
in the clean bowl on the work table, the cutting board, the clean bowl and the floor kept clean and dry; the faucet
closed and nothing in the hands.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure), gate(name, ok, detail) records one. Under the task-set contracts task_set_progress(ctx) gives
the progress (task_set/scoring.py; ctx holds the measures and terms that function reads for every task). verify(e,
settle_s) gives the episode's verdict, the dict of task.Evaluator.evaluate_settled (the settle branch of at most
settle_s seconds, None: the default).
"""
from surface_rinsing import params as P
from surface_rinsing.task_set.scoring import reduction

C = P.CRITERIA


def criteria(e, m, gate):
    items, cont = m["items"], m["contamination"]
    toms = [n for n in items if n.startswith("tomato_")]
    clean = {n: items[n]["soil_gauge_max_ug_cm2"] for n in toms}
    gate("visibly_clean", all(v <= P.VRL*1e5 for v in clean.values()), clean)
    kept = {n: e._inside_cavity(n, "colander") for n in toms}
    gate("none_lost", all(kept.values()), kept)
    gate("placed", e._inside_cavity("colander", "clean_bowl"), dict(colander=items["colander"]["position"]))
    gate("drained", items["colander"].get("free_water_mL", 0.0) <= C["drained_colander"]*1e6,
         dict(colander_free_water_mL=items["colander"].get("free_water_mL", 0.0)))
    gate("contained", cont.get("cutting_board", {}).get("gauge_max_ug_cm2", 0) <= P.VRL*1e5 and
         cont.get("clean_bowl", {}).get("gauge_max_ug_cm2", 0) <= P.VRL*1e5 and cont["floor_water_mL"] <= C["floor_water"]*1e6,
         cont)


def task_set_progress(ctx):
    criteria, passed, items, initial = ctx.criteria, ctx.passed, ctx.items, ctx.initial
    released, floor, soil = ctx.released, ctx.floor, ctx.soil
    names = [n for n in initial if n.startswith("tomato_")]
    kept = criteria["none_lost"]["detail"]
    credits = {n: reduction(initial[n]["soil_gauge_max_ug_cm2"], items[n]["soil_gauge_max_ug_cm2"], 1.0)
                  if kept[n] else 0.0 for n in names}
    c = sum(credits.values())/len(names)
    p = passed("placed") * passed("drained") * released
    units = (sum(not kept[n] for n in names) + soil("cutting_board") + soil("clean_bowl") + floor)/len(names)
    value = (c + c*p)/2.0 - units
    terms = dict(clean=credits, mean_clean=c, placed_drained=p)
    return value, terms, units


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)

"""Outcome of rinse-sponge: the sponge's pore water free of suds and the sponge wrung out, back on the ledge; no water
running and nothing in the hands.

criteria(e, m, gate) adds the task's gates to task.Evaluator.criteria: e is the evaluator, m its measures
(task.Evaluator.measure), gate(name, ok, detail) records one. Under the task-set contracts task_set_criteria(e, m, gate)
then reads the ledge as the deck behind the three compartments (task_set/evaluator.py) and task_set_progress(ctx) gives
the progress (task_set/scoring.py; ctx holds the measures and terms that function reads for every task). verify(e,
settle_s) gives the episode's verdict, the dict of task.Evaluator.evaluate_settled (the settle branch of at most
settle_s seconds, None: the default).
"""
from surface_rinsing import params as P
from surface_rinsing.task_set.scoring import clip, reduction

C = P.CRITERIA


def criteria(e, m, gate):
    items = m["items"]
    sp = e.sponge_report()
    gate("no_suds", sp["c_g_L"] <= C["sponge_suds_g_L"], dict(pore_surfactant_g_L=sp["c_g_L"]))
    gate("wrung_out", sp["fill"] <= C["sponge_wrung"], dict(pore_fill=sp["fill"]))
    gate("sponge_on_ledge", e._on_ledge("sponge", compartments=False), dict(position=items["sponge"]["position"]))


def task_set_criteria(e, m, gate):
    gate("sponge_on_ledge", e._on_ledge("sponge", compartments=True), m["items"]["sponge"]["position"])


def task_set_progress(ctx):
    start, m, passed, released, units = ctx.start, ctx.m, ctx.passed, ctx.released, ctx.units
    s = reduction(start["sponge"]["c_g_L"], m["sponge"]["c_g_L"], 0.02)
    w = clip((1.0 - m["sponge"]["fill"]) / 0.40)
    p = passed("sponge_on_ledge") * released
    value = (s + s*w + s*w*p)/3.0
    terms = dict(rinsed=s, wrung=w, returned=p)
    return value, terms, units


def verify(e, settle_s=None):
    return e.evaluate_settled(settle_s)

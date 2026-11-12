"""Outcome of press-drill: a 9 mm hole (at most 9.25 mm) through the bar within 0.5 mm of the punched mark, its axis
within 1 degree of the face normal, the bar turned less than 2 degrees while drilling, and the press left with its
spindle stopped, its quill up and its bit clear of the work.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 the continuous progress(ev) and the report's NOTE as well."""
from drilling import params as P
from drilling.task import SCORE_BIT_BURN, _check

NOTE = "The outcome is judged: a bar held without turning may pass without vise clamping."


def verify(ev):
    return ev.evaluate_checks(checks, progress=progress, note=NOTE)


def checks(ev):
    out, h = ev._through_hole_checks("flat_bar", ev.L["marks"][0]["local"], 9.0, 0.0005, "hole A", 9.25)
    turn = ev.s["max_rel_turn_deg"].get("flat_bar", 0.)
    out.append(_check("bar turned less than 2 deg while drilling", turn < 2., round(turn, 3), "< 2 deg",
                      kind="constraint"))
    if SCORE_BIT_BURN:
        T = ev.s["max_bit_temp_c"].get("drill_press_bit", 20.)
        out.append(_check("bit tip never above 550 C", T < P.HSS_BURN_C, round(T, 1), "< 550 C", kind="constraint"))
    return out + ev.spindle_stopped_quill_up()


def progress(ev):
    hit = ev.nearest("flat_bar", ev.L["marks"][0]["local"])
    bore = 0. if hit is None else ev._bore(*hit, 9., 0.006, 0.0005, 9.25)
    return bore*(0.85 + 0.15*ev._finish())

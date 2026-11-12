"""Outcome of sauce-blend: the mayonnaise, ketchup and relish (honey variant: mayonnaise, honey and mustard) evenly
blended, each one's coefficient of variation over 5 ml portions at most 0.10, none of it spilt and the bowl where it
stood; for the pet-meal variant the supplement evenly mixed into the food over 30 ml portions.

JUDGING holds the task's portion size and its criteria (task.TASKS; task.evaluate measures them) and
variant_judging(t, variant) puts a variant's own in the task's spec t (task.task_spec). verify(world, info, *,
_progress=True) evaluates the state of the World with the start record info and returns the evaluation dict
(scoring.evaluate_criteria: task.evaluate, or the ingredient-mixing-tasks-v1 or -v2 grading when the selection chose
that contract)."""
from ingredient_mixing import scoring

JUDGING = dict(
    portion_ml=5.,
    criteria=dict(uniform=dict(species=("paste_a", "paste_b", "paste_c"), cv=0.10), contained=0.01,
                  vessel_in_place=0.05))


def variant_judging(t, variant):
    if variant == "pet-meal":
        t["criteria"] = dict(uniform=dict(species="minor", cv=0.10), contained=0.01, vessel_in_place=0.05)
        t["portion_ml"] = 30.


def verify(world, info, *, _progress=True):
    return scoring.evaluate_criteria(world, info, _progress=_progress)

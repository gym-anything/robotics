"""Outcome of dry-whisk: the flour, sugar, baking powder and salt evenly combined (the sugar's and the
leavening's coefficient of variation over twelve portions at most 0.10), none of it spilt and the bowl where it stood.

JUDGING holds the task's portion size and its criteria (task.TASKS; task.evaluate measures them). verify(world,
info, *, _progress=True) evaluates the state of the World with the start record info and returns the evaluation dict
(scoring.evaluate_criteria: task.evaluate, or the ingredient-mixing-tasks-v1 or -v2 grading when the selection chose
that contract)."""
from ingredient_mixing import scoring

JUDGING = dict(
    portions=12,
    criteria=dict(uniform=dict(species=("minor", "sugar"), cv=0.10), contained=0.01, vessel_in_place=0.05))


def verify(world, info, *, _progress=True):
    return scoring.evaluate_criteria(world, info, _progress=_progress)

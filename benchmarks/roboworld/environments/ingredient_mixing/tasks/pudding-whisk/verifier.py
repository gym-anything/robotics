"""Outcome of pudding-whisk: the pudding evenly combined, with no dry mix and no lumps left, none of it
spilt, the bowl where it stood and the whisk out of the bowl.

JUDGING holds the task's portion size and its criteria (task.TASKS; task.evaluate measures them). verify(world,
info, *, _progress=True) evaluates the state of the World with the start record info and returns the evaluation dict
(scoring.evaluate_criteria: task.evaluate, or the ingredient-mixing-tasks-v1 or -v2 grading when the selection chose
that contract)."""
from ingredient_mixing import scoring

JUDGING = dict(
    portion_ml=120.,
    criteria=dict(uniform=dict(species="solid", cv=0.10), dry=0.02, lumps=0.02, contained=0.01, vessel_in_place=0.05,
                  tool_out=True))


def verify(world, info, *, _progress=True):
    return scoring.evaluate_criteria(world, info, _progress=_progress)

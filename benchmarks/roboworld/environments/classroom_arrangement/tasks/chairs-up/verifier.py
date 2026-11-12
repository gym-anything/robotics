"""Outcome of chairs-up: every row-4 chair upside down on its own desk, the desks still on their dots.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract)."""


def criteria(s):
    check = {}
    for c in range(1, 7):
        check[f"desk_4{c}"] = "on_marks"
        check[f"chair_4{c}"] = ("up_on", f"desk_4{c}")
    return check


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window)

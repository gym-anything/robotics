"""Outcome of push-in-chairs: every row-2 chair tucked under its desk, the desks not moved.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract)."""


def criteria(s):
    check = {}
    for c in range(1, 7):
        check[f"chair_2{c}"] = ("tucked", f"desk_2{c}")
        check[f"desk_2{c}"] = "unmoved"
    return check


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window)

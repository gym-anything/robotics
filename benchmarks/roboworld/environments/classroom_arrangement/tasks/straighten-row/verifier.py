"""Outcome of straighten-row: every row-3 desk on its floor dots, squared to the board, its chair tucked under it.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract)."""


def criteria(s):
    check = {}
    for c in range(1, 7):
        check[f"desk_3{c}"] = "on_marks"
        check[f"chair_3{c}"] = ("tucked", f"desk_3{c}")
    return check


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window)

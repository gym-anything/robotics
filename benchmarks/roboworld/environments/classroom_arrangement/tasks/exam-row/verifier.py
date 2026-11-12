"""Outcome of exam-row: every row-2 desk on its exam dots facing the board, its chair in use behind it, and every
pair of the row's chairs at least 1.25 m apart.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract)."""


def criteria(s):
    check = {}
    for c in range(1, 7):
        check[f"desk_2{c}"] = "on_marks"
        check[f"chair_2{c}"] = ("in_use", f"desk_2{c}")
    check["exam"] = [f"chair_2{c}" for c in range(1, 7)]
    return check


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window)

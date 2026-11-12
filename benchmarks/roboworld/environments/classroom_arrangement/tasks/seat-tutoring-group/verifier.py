"""Outcome of seat-tutoring-group: the four chairs ready to sit at the tutoring table, facing it (two on the room
side, one at each end), the table not moved.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, which add their own
seating_plan for ready checks, or the classroom-tasks-v1 or -v2 grading when the episode selected that contract, with
seating_plan below)."""


def criteria(s):
    check = {}
    for k in range(1, 5):
        check[f"tchair_{k}"] = ("ready", "table")
    check["table"] = "unmoved"
    return check


def seating_plan(criteria):
    """The contract gradings' extra criterion: the chairs that are ready sit two on the room side and one at each end."""
    sides = [criteria[f"tchair_{i}"]["side"] for i in range(1, 5) if criteria[f"tchair_{i}"]["ok"]]
    return {"seating_plan": dict(check="seating_plan", ok=sorted(sides) == ["left_end", "right_end", "room", "room"])}


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window, extra=seating_plan)

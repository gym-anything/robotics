"""Outcome of move-teacher-desk: the teacher's desk on its floor dots along the windows, its chair 0.35-0.95 m behind
it on the room side.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract, with the progress weights below)."""

# The contract gradings' progress: the desk weighs three times its chair.
PROGRESS_WEIGHTS = {"desks": 0.75, "chairs": 0.25}


def criteria(s):
    return {"teacher_desk": "on_marks", "teacher_chair": ("behind", "teacher_desk", (0.35, 0.95))}


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window, weights=PROGRESS_WEIGHTS)

"""Outcome of clear-floor: the tote on the floor in front of the cubbies, the wastebasket within 0.20 m of its place
beside the teacher's desk, the stray chair tucked under its desk, and a 0.915 m route open from the door to the back.

criteria(s) gives the evaluator's checks (layout.setup puts them in the spec as "check"); verify(e, motion, window)
evaluates an episode with them (task.Evaluator.evaluate_criteria: the released checks, or the classroom-tasks-v1 or
-v2 grading when the episode selected that contract, with the progress weights below)."""
from classroom_arrangement.layout import BASKET_HOME, CUBBY_ZONE

# The contract gradings' progress: the tote, the wastebasket, the chair and the route weigh the same.
PROGRESS_WEIGHTS = {"tote": 0.25, "basket": 0.25, "chairs": 0.25, "route": 0.25}


def criteria(s):
    return {"tote": ("in_zone", CUBBY_ZONE), "basket": ("near", BASKET_HOME, 0.20), "chair_36": ("tucked", "desk_36"),
            "route": [(2.65, 0.6), (0., 8.5)]}


def verify(e, motion, window):
    return e.evaluate_criteria(motion, window, weights=PROGRESS_WEIGHTS)

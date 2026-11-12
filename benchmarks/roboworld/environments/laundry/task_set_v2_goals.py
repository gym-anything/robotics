"""State-only objectives of contract `laundry-tasks-v2`. No runtime imports, forces, or pose writes.

Shared pieces are imported unchanged from `task_set_goals` (the `laundry-tasks-v1` objectives,
which stay as released). The objectives new in this contract are in the task folders
(tasks/<task>/verifier.py): the thirds fold that counts cloth crossings instead of merging
heights within 2 mm (thirds_fold), and the swapped-in `towel/roll` (roll) and
`tee/half_fold` (tee_half_fold) objectives.
"""
from .task_set_goals import (  # noqa: F401  (re-exported for task_set_v2)
    ContinuousGoal, StartRelativeEvaluator,
    above, below, outside_distance, towel_hems,
)

"""classroom-tasks-v2: the render-checked eight-task set. Physics-free; no scene, start or physics changes.

v2 is the released set (catalogue.py, classroom-tasks-v1) with the fix of the render check: progress is
measured from the post-settle start moved START_ROUNDING toward the goal, so an untouched start scores exactly 0
(v1 read a straighten-row desk's last-bit start difference as 2e-14 progress) and the goal is still exactly 1
(task_set_v2.py). Tasks, starts (layout.setup), instructions, success predicates and the score rule are v1's.

The seven tasks other than group-work select with "contract": "classroom-tasks-v2". group-work, the audited
hardest task, keeps its released selection {"task": "group-work"} and its own evaluator. A selection or launch
with any other contract, or none, runs as released: catalogue.validate_contract and catalogue.description, task.py
and the backend's variation_contract dispatch here only for this contract.
"""

from .catalogue import CONTRACT as V1, LEGACY, SET_TASKS

CONTRACT = "classroom-tasks-v2"
TASKS_V2 = SET_TASKS[:-1]
# The eight selections of benchmark.json's task_set; group-work exactly as in splits/test.json.
TASK_SET = tuple(dict(task=t, seed=0, contract=CONTRACT) for t in TASKS_V2) + (dict(task="group-work"),)


def validate(task):
    if task not in TASKS_V2:
        raise ValueError(f"{task} keeps its existing evaluator; select contract={LEGACY!r}")


def description(task):
    """classroom-tasks-v1's instruction text: v2 changes grading only."""
    from .catalogue import description as released
    validate(task)
    return released(task, V1)


def contract(released):
    """The variation catalogue of a classroom-tasks-v2 episode: variants.contract() with v2 listed."""
    return dict(released, contracts=[LEGACY, V1, CONTRACT], task_set=list(TASK_SET),
                task_set_v1=list(released["task_set"]))

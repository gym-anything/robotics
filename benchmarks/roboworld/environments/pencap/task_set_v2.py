"""Physics-free discovery of the pencap-tasks-v2 set: the render-checked eight tasks.

v2 is the released set (tasks.py, contracts pencap/<task>-v1) with the fixes of the
render check. Same tasks, instructions, success predicates and score rule
(1 on success, else 0.9 x progress); what differs:

- moved starts (pass-pen, reseat-cap, uncap, unload-fixture, cap-in-fixture) are
  written to the parts' free-joint coordinates, so they survive eval_fk at model
  creation (in v1 they revert to the default scene): task_presets_v2.apply_start;
- start-relative progress gives nothing for settling within the 2 mm rest tolerance
  or for a start with no error to remove; load-fixture's upright credit starts above
  lying; uncap progress is measured from the snap groove: task_scoring_v2.

The seven new tasks select with {"task": <task>, "contract": "pencap-tasks-v2"}.
seat-cap, the audited hardest task, is unchanged and keeps its released selection {}.
A selection without "contract" resolves exactly as released (tasks.task_resolve).
"""

from .tasks import DEFAULT_TASK, EIGHT_TASKS, INSTRUCTIONS, validate

CONTRACT = "pencap-tasks-v2"
TASKS_V2 = tuple(task for task in EIGHT_TASKS if task != DEFAULT_TASK)
# The eight selections of benchmark.json's task_set, in its order.
SELECTIONS = [*({"task": task, "contract": CONTRACT} for task in TASKS_V2), {}]


def task_id(task):
    return f"pencap/{task}/{CONTRACT}"


def check(task, contract):
    """The task of a v2 selection or backend option; seat-cap keeps its released launch."""
    if contract != CONTRACT:
        raise ValueError(f"Unknown pencap contract {contract!r}; choose {CONTRACT}")
    validate(task)
    if task == DEFAULT_TASK:
        raise ValueError("seat-cap is unchanged in pencap-tasks-v2; select it as released, with no contract")
    return task


def task_resolve(selection):
    if not isinstance(selection, dict) or set(selection) != {"task", "contract"}:
        raise ValueError("A pencap-tasks-v2 selection names exactly a task and the contract")
    task = check(selection["task"], selection["contract"])
    return dict(task_id=task_id(task), description=INSTRUCTIONS[task],
                backend_options={"task": task, "contract": CONTRACT},
                selection={"task": task, "contract": CONTRACT}, split=None)

"""concrete-tasks-v2: the render-checked eight-task set, beside the released task_set.py (concrete-tasks/progress-v1).

The eight tasks, their seeded starts, instructions, predicates, progress and 0.95 failure factor are task_set.py's
(the evaluator is task_set_progress.py's). What the render check changed:
- a selection of this contract gets its own task id, the contract before the seed
  (concrete-finishing/v1/b012/<task>/concrete-tasks-v2/seed-N), so results stored under one id follow one scoring
  rule; bare selections keep their ids and legacy scores;
- installing the evaluator also drops the state's cached evaluation, so the first state read after a launch or reset
  under this contract is judged by it (task_set_v2_progress.install).
pad-finish keeps its own evaluator and id (pad-finish/progress-v3) and takes no contract; in the set it is the
released test-split selection. Nothing here changes a scene, a material or the released contract.
Kept outside probes.common.PHYSICS and STAMP_EXTRA.
"""

from . import task_set as V1, variants

CONTRACT = "concrete-tasks-v2"
TASKS = V1.TASKS
SELECTORS = V1.SELECTORS
DIFFICULTY = V1.DIFFICULTY
PAD_FINISH_SELECTION = {"task": "pad-finish"}          # splits/test.json, unchanged


def selected_contract(task, contract=None):
    """This contract for the seven tasks other than pad-finish; any other choice is task_set.selected_contract's."""
    if contract == CONTRACT:
        if task in TASKS[:-1]:
            return CONTRACT
        raise ValueError(f"Unknown contract {contract!r} for {task}; {CONTRACT} is for {', '.join(TASKS[:-1])}")
    return V1.selected_contract(task, contract)


def contract_task_id(task_id):
    head, seed = task_id.split("/seed-", 1)
    return f"{head}/{CONTRACT}/seed-{seed}"


def task_resolve(selection):
    """A selection naming this contract (task_set.task_resolve dispatches it here); others resolve as released."""
    if selection.get("contract") != CONTRACT:
        return V1.task_resolve(selection)
    result = variants.task_resolve(selection)
    selected_contract(result["selection"]["task"], CONTRACT)
    for key in ("selection", "backend_options"):
        result[key] = dict(result[key], contract=CONTRACT)
    result["task_id"] = contract_task_id(result["task_id"])
    return result


def selections():
    return [dict(preset="b012", task=t, seed=0, variant="default", contract=CONTRACT) for t in TASKS[:-1]] \
        + [dict(PAD_FINISH_SELECTION)]


def contract():
    """The variation contract of a session under this contract: the released one, with this contract among the
    seven tasks' choices and this set's selections."""
    c = V1.contract()
    c["contract_choices"] = {t: v + ([CONTRACT] if t in TASKS[:-1] else []) for t, v in c["contract_choices"].items()}
    c["task_set"] = selections()
    return c

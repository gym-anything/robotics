"""forklift-tasks-v2: the render-checked eight-task set (fix round of the forklift-tasks-v1 set); no scene or
physics imports.

The same eight work orders, starts and instructions as forklift-tasks-v1 (forklift/catalogue.py, unchanged); what
differs is how an attempt is judged and run (forklift/task_progress_v2.py, forklift/task_set_v2_backend.py):
- commands run as in the base backend: none is cut at the task's time limit and none is refused after it. The limit is
  evaluator.py's unchanged time_limit success criterion (sim_time <= TIME_LIMIT_S, simulated time from the settled
  start); every evaluation reports time_limit_s and within_time_limit, and the adapter keeps no terminal latch;
- a start already at the goal earns no approach credit;
- the policy's capabilities declare the limit with the note on how the evaluation applies it.
Bare, v1, forklift-tasks-v1 and load-bales selections stay as released. forklift.variants.task_resolve hands a
selection here only when it names forklift-tasks-v2; NewtonForkliftBackend builds the v2 session class only for a
launch that names it.
"""

from . import catalogue as v1

CONTRACT = "forklift-tasks-v2"
LEGACY_CONTRACT = v1.LEGACY_CONTRACT
BALE_CONTRACT = v1.BALE_CONTRACT
TASKS = v1.SET_TASKS
DIFFICULTY = dict(v1.DIFFICULTY)
TASK_SET = tuple(dict(task=t, seed=0, **({"contract": CONTRACT} if t != "load-bales" else {})) for t in TASKS)


def resolve_contract(task, contract=None):
    """The released choices (catalogue.resolve_contract) plus forklift-tasks-v2 for the seven revised tasks."""
    default = BALE_CONTRACT if task == "load-bales" else LEGACY_CONTRACT
    if contract is None:
        return default
    choices = (default, v1.CONTRACT, CONTRACT) if task in TASKS[:-1] else (default,)
    if not isinstance(contract, str) or contract not in choices:
        raise ValueError(f"Unknown forklift contract {contract!r} for {task!r}; choices: {choices}")
    return contract


def task_resolve(selection):
    """A forklift-tasks-v2 selection: the released resolver's seed, task and robot handling, start and instruction,
    with the contract in the task id, the selection and the backend options."""
    from .variants import task_resolve as released
    out = released({k: v for k, v in selection.items() if k != "contract"})
    task = out["selection"]["task"]
    contract = resolve_contract(task, selection.get("contract"))
    if contract != CONTRACT:
        return released(selection)
    v = out["selection"]
    out["task_id"] = f"forklift/v1/{v['preset']}/{task}/{CONTRACT}/seed-{v['seed']}"
    v["contract"] = out["backend_options"]["contract"] = CONTRACT
    return out

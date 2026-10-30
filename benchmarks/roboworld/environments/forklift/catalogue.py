"""Eight existing work orders, explicitly versioned; no scene or physics imports."""

CONTRACT = "forklift-tasks-v1"
LEGACY_CONTRACT = "v1"
BALE_CONTRACT = "load-bales-reward-v1"
SET_TASKS = (
    "start-and-park", "aisle-cruise", "drive-to-pallet", "carry-to-floor-bay",
    "ramp-down-loaded", "put-away-rack", "load-trailer", "load-bales",
)
DIFFICULTY = dict(zip(SET_TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))


def resolve_contract(task, contract=None):
    default = BALE_CONTRACT if task == "load-bales" else LEGACY_CONTRACT
    if contract is None:
        return default
    choices = (default, CONTRACT) if task in SET_TASKS[:-1] else (default,)
    if not isinstance(contract, str) or contract not in choices:
        raise ValueError(f"Unknown forklift contract {contract!r} for {task!r}; choices: {choices}")
    return contract


TASK_SET = tuple(dict(task=t, seed=0, **({"contract": CONTRACT} if t != "load-bales" else {})) for t in SET_TASKS)

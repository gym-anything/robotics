"""lockkey-tasks-v2: the render-checked eight-task set, selected only with an explicit contract.

Same tasks, instructions and success predicates as lockkey-tasks-v1 (catalogue.py). v2 settles the four
constructed free-key starts at task setup (settle_v2.py), measures progress from the settled start with
idle margins (scoring_v2.py), lays return-key's key where it rests on the mat (presets_v2.py) and reports
the unlock chain as `unlock_chain_reading`. A selection without `contract` keeps the released behaviour;
`unlock` keeps its own contract and its released selection ({} or {"task": "unlock"}).
"""

from .catalogue import DEFAULT_TASK, INSTRUCTIONS, TASKS, validate

CONTRACT = "lockkey-tasks-v2"
CONTRACTS = (CONTRACT,)
V2_TASKS = TASKS[1:]


def task_id(task):
    return f"lockkey/{task}/{CONTRACT}"


def validate_contract(task, contract):
    """The task under an explicit contract; `unlock` never takes one."""
    if contract not in CONTRACTS:
        raise ValueError(f"Unknown lockkey contract {contract!r}; choose from {CONTRACTS}, "
                         "or omit it for the released lockkey-tasks-v1 behaviour")
    if validate(task) == DEFAULT_TASK:
        raise ValueError("unlock keeps its own contract and evaluator; select it without a contract")
    return task


def task_resolve(selection):
    if not isinstance(selection, dict) or set(selection) - {"task", "contract"}:
        raise ValueError("Lockkey accepts only the task and contract selectors; no seeds or reset options")
    if "task" not in selection:
        raise ValueError(f"{CONTRACT} needs an explicit task from {V2_TASKS}")
    task = validate_contract(selection["task"], selection["contract"])
    return dict(task_id=task_id(task), description=INSTRUCTIONS[task],
                backend_options={"task": task, "contract": CONTRACT},
                selection={"task": task, "contract": CONTRACT}, split=None)

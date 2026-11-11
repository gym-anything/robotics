"""soil-tasks-v2: the render-checked eight-task set, selected only with an explicit contract.

Same tasks, seeded layouts, instructions and success criteria as soil-tasks-v1 (catalogue.py); the judgements
are in task_set_v2.py. set-post also accepts this contract: it keeps its own instruction, criteria, load test
and success, and only its failed-run score is measured from the settled start. A selection without
`contract: soil-tasks-v2` resolves exactly as released (catalogue.task_resolve dispatches here only for it).
"""
from . import catalogue as v1
from . import variants as legacy
from .layout import TASKS

CONTRACT = "soil-tasks-v2"
POST_CONTRACT = v1.POST_CONTRACT
EIGHT_TASKS = v1.EIGHT_TASKS
REVISED = v1.REVISED                 # revised judgements
CONTRACT_TASKS = EIGHT_TASKS         # tasks that accept soil-tasks-v2
SELECTORS = v1.SELECTORS
INSTRUCTIONS = dict(v1.INSTRUCTIONS, **{"set-post": TASKS["set-post"]["instruction"]})  # set-post: its own text
# benchmark.json task_set: set-post keeps its released split selection (splits/test.json).
TASK_SET = tuple(dict(task=t) if t == "set-post" else dict(task=t, seed=0, contract=CONTRACT) for t in EIGHT_TASKS)


def choices(task):
    default = v1.default_contract(task)
    return (default,) + ((v1.CONTRACT,) if task in REVISED else ()) + ((CONTRACT,) if task in CONTRACT_TASKS else ())


def selected_contract(task, contract=None):
    options = choices(task)
    if contract is not None and contract not in options:
        raise ValueError(f"Unknown contract {contract!r} for {task}; choose {options}")
    return options[0] if contract is None else contract


def instruction(task, contract=None):
    return INSTRUCTIONS[task] if selected_contract(task, contract) == CONTRACT else v1.instruction(task, contract)


def resolve(task=None, seed=0, moisture=None, topsoil=None, stones=None, contract=None):
    value = legacy.resolve(task, seed, moisture, topsoil, stones)
    chosen = selected_contract(value["task"], contract)
    if chosen in (v1.CONTRACT, CONTRACT):
        value["contract"] = chosen
    return value


def task_resolve(selection):
    """A selection with contract soil-tasks-v2; the legacy resolver keeps int(seed), rounding and defaults."""
    if selection.get("contract") != CONTRACT:
        raise ValueError(f"catalogue_v2 resolves only contract {CONTRACT!r}")
    result = legacy.task_resolve(selection)
    task = result["selection"]["task"]
    selected_contract(task, CONTRACT)                 # only the eight tasks take it
    result["task_id"] = result["task_id"].replace("/seed-", f"/{CONTRACT}/seed-", 1)
    result["description"] = INSTRUCTIONS[task]
    result["backend_options"]["contract"] = CONTRACT
    result["selection"]["contract"] = CONTRACT
    return result


def contract():
    """The variation contract served by a soil-tasks-v2 session."""
    result = v1.contract()
    result.update(contract_choices={t: list(choices(t)) for t in TASKS},
                  task_set_v1=result.pop("task_set"), task_set=[dict(s) for s in TASK_SET])
    return result

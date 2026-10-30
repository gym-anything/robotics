"""Eight-task selections, kept apart from the unchanged physics modules.

The legacy resolver remains authoritative for input coercion, defaults, seeded
layouts and old ids. Only an explicit contract selects a revised judgement.
"""
from . import variants as legacy
from .layout import TASKS

CONTRACT = "soil-tasks-v1"
POST_CONTRACT = "soil-set-post-v1"
LEGACY_CONTRACT = "v1"
EIGHT_TASKS = (
    "penetrate-marked", "spit-to-tarp", "planting-hole", "break-hardpan",
    "expose-line", "cable-trench", "auger-holes", "set-post",
)
REVISED = EIGHT_TASKS[:-1]
SELECTORS = (*legacy.SELECTORS, "contract")
INSTRUCTIONS = {
    "penetrate-marked": "Push the spade blade straight down into the bed at the white paint mark, at least 15 cm "
                        "deep, with both hands on the spade.",
    "spit-to-tarp": "Dig a spadeful of soil from the bed at the white mark and dump it on the blue tarp on the "
                    "walkway without spilling it elsewhere.",
    "planting-hole": "Dig a planting hole for the #1 container shrub (root ball 165 mm across, 160 mm tall) centred "
                     "on the white mark: as deep as the root ball and two to three times as wide at the top, on a "
                     "firm, undisturbed floor; put the spoil on the tarp.",
    "break-hardpan": "Break up the compacted ground inside the marked square with the hand mattock, at least "
                     "8 cm deep, and put the broken soil you lift out on the tarp.",
    "expose-line": TASKS["expose-line"]["instruction"],
    # 'No wider than the shovel blade' would contradict the retained 6-20 cm
    # band (the blade is 120 mm). State the job without that false limit.
    "cable-trench": "Dig a narrow trench for a low-voltage path-light cable along the white line between the two "
                    "path lights, deep enough for 150 mm of cover, with the spoil on the lawn-side tarp; stay on "
                    "the walkway.",
    "auger-holes": "Drill a bulb hole at each of the five white marks with the drill-mounted planting auger, each "
                   "at least 18 cm deep and left open. The drill's trigger is squeezed by the gripper.",
}
DIFFICULTY = dict(zip(EIGHT_TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
TASK_SET = tuple(dict(task=t, seed=0, **({"contract": CONTRACT} if t in REVISED else {})) for t in EIGHT_TASKS)


def default_contract(task):
    return POST_CONTRACT if task == "set-post" else LEGACY_CONTRACT


def selected_contract(task, contract=None):
    default = default_contract(task)
    choices = (default, CONTRACT) if task in REVISED else (default,)
    if contract is not None and contract not in choices:
        raise ValueError(f"Unknown contract {contract!r} for {task}; choose {choices}")
    return default if contract is None else contract


def instruction(task, contract=None):
    return INSTRUCTIONS[task] if selected_contract(task, contract) == CONTRACT else TASKS[task]["instruction"]


def resolve(task=None, seed=0, moisture=None, topsoil=None, stones=None, contract=None):
    value = legacy.resolve(task, seed, moisture, topsoil, stones)
    chosen = selected_contract(value["task"], contract)
    if chosen == CONTRACT:
        value["contract"] = chosen
    return value


def task_resolve(selection):
    result = legacy.task_resolve(selection)  # retains int(seed), topsoil rounding and all old defaults
    task = result["selection"]["task"]
    chosen = selected_contract(task, selection.get("contract"))
    if chosen == CONTRACT:
        result["task_id"] = result["task_id"].replace("/seed-", f"/{CONTRACT}/seed-", 1)
        result["description"] = INSTRUCTIONS[task]
        result["backend_options"]["contract"] = chosen
        result["selection"]["contract"] = chosen
    return result


def contract():
    result = legacy.contract()
    result.update(selectors=list(SELECTORS),
                  contract_choices={t: [default_contract(t)] + ([CONTRACT] if t in REVISED else []) for t in TASKS},
                  task_set=[dict(s) for s in TASK_SET])
    return result

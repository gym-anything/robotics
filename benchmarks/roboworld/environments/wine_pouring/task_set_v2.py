"""wine-tasks-v2: the render-checked eight-task set, beside the released set of contracts.py (wine-tasks-v1).

The tasks, starts, instructions, checks, progress, settled read and 0.9 failure factor are wine-tasks-v1's:
task_contracts.py judges both contracts alike (TASK_SET_CONTRACTS) and reports the one selected. What the
render check changed: a selection of this contract gets its own task id, the contract before the seed
(wine-pouring/v1/<preset>/<task>/wine-tasks-v2/seed-N), so one id never carries two scoring rules. split-bottle keeps
its released selection (splits/test.json), id and audited contract. Selections without this contract resolve, launch
and score as released.
"""

from .contracts import EIGHT_TASKS, PROGRESS_TASKS, TASKS_V1_REWARD_CONTRACT

CONTRACT = "wine-tasks-v2"
TASK_SET_CONTRACTS = (TASKS_V1_REWARD_CONTRACT, CONTRACT)
HARDEST_SELECTION = {"preset": "service", "task": "split-bottle"}     # splits/test.json, unchanged


def accepts(task, selected):
    """Whether `selected` is this contract on a task it covers (contracts.reward_contract asks first)."""
    return selected == CONTRACT and task in PROGRESS_TASKS


def selections():
    return [dict(preset=p, task=t, seed=0, contract=CONTRACT) for p, t, _ in EIGHT_TASKS if t != "split-bottle"] \
        + [dict(HARDEST_SELECTION)]


def task_resolve(selection):
    """A selection naming this contract (variants.task_resolve hands it here): variants.task_resolve's resolution with
    this contract's own task id."""
    from .contracts import TaskConfig
    from .task_contracts import instruction
    from .variants import config_values, resolve, split_of_seed
    if set(selection) - {"preset", "task", "seed", "contract"}:
        raise ValueError("Selectors are preset, task, seed and contract")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    if v.get("contract") != CONTRACT:
        raise ValueError(f"{CONTRACT} was not selected")
    return dict(task_id=f"wine-pouring/v1/{v['preset']}/{v['task']}/{CONTRACT}/seed-{v['seed']}",
                description=instruction(TaskConfig(**config_values(v))),
                backend_options=dict(v),
                selection=v, split=split_of_seed(v["task"], v["seed"]))


def contract():
    """The variation contract of a session under this contract: variants.contract() with this contract among the
    seven tasks' reward contracts and this set's selections."""
    from .variants import contract as released
    c = released()
    c["reward_contracts"] = {t: v + ([CONTRACT] if t in PROGRESS_TASKS else []) for t, v in c["reward_contracts"].items()}
    c["task_set"] = selections()
    return c

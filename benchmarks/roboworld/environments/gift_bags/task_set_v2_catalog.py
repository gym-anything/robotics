"""gift-bags-tasks-v2: the render-checked eight-task set, outside probes.PHYSICS_FILES. Physics-free.

v2 is the released set (task_set_catalog.py, gift-bags-tasks-v1) with the fixes of the render check: the
open-flaps start builds (the taped world's caliper stock names the contact stock once the seam tape is omitted),
bag-candle starts from the spec's worked-open tote, put-lid-on's engagement credit needs the base rim inside the lid's
skirt, closed flaps' own creep earns nothing, and evaluations report the bare contract with a `task` field. Tasks,
instructions, predicates and the score rule are otherwise v1's (task_set_v2_presets/_evaluator/_scoring.py).

The seven tasks other than bag-purchase select with "contract": "gift-bags-tasks-v2". bag-purchase, the audited
hardest task, keeps its released selection. A selection with any other contract, or none, resolves exactly as
released: task_set_catalog.task_resolve dispatches here only for this contract. The functions below accept this
contract and hand every other one to the released catalogue, so a v2 session can still preview any entry.
"""
from dataclasses import asdict, dataclass

from . import task_set_catalog as V1
from .config import Config as LegacyConfig
from .task import SETTLE_TASKS as LEGACY_SETTLE

CONTRACT = "gift-bags-tasks-v2"
NEW_TASKS, BASE_TASK, TASKS, ORDER = V1.NEW_TASKS, V1.BASE_TASK, V1.TASKS, V1.ORDER
POLICY_PRESETS, INSTRUCTIONS = V1.POLICY_PRESETS, V1.INSTRUCTIONS
LEGACY, PURCHASE_CONTRACT = V1.LEGACY, V1.PURCHASE_CONTRACT
TASKS_V2 = tuple(t for t in ORDER if t != "bag-purchase")
# The eight selections of benchmark.json's task_set; bag-purchase exactly as in splits/test.json.
TASK_SET = [dict(preset=next(p for p in POLICY_PRESETS if t in TASKS[p]), task=t, seed=0, contract=CONTRACT)
            for t in TASKS_V2] + [dict(preset="counter", task="bag-purchase")]


def selected_contract(task, contract=None):
    if contract == CONTRACT:
        if task not in TASKS_V2:
            raise ValueError(f"{task!r} is not in {CONTRACT}" + ("; bag-purchase keeps its released selection"
                                                                 if task == "bag-purchase" else ""))
        return CONTRACT
    return V1.selected_contract(task, contract)


def instruction(task, contract=None):
    if contract == CONTRACT:
        selected_contract(task, contract)
        return INSTRUCTIONS[task]
    return V1.instruction(task, contract)


def evaluation_contract(task, contract=None):
    # Reported bare; the evaluation's `task` field names the task.
    if contract == CONTRACT:
        return selected_contract(task, contract)
    return V1.evaluation_contract(task, contract)


def completes_window(task, contract=None):
    if contract == CONTRACT:
        selected_contract(task, contract)
        return task in LEGACY_SETTLE or task in ("open-bag", "bag-candle")
    return V1.completes_window(task, contract)


@dataclass(frozen=True)
class Config(LegacyConfig):
    """A gift-bags-tasks-v2 launch. Every old field is validated by the unmodified base Config, as in v1."""
    contract: str = CONTRACT

    def __post_init__(self):
        if self.contract != CONTRACT:
            raise ValueError(f"task_set_v2_catalog.Config is the {CONTRACT} launch; other contracts use "
                             "task_set_catalog.Config")
        values = asdict(self)
        values.pop("contract")
        values["task"] = BASE_TASK.get(self.task, self.task)
        LegacyConfig(**values)
        selected_contract(self.task, self.contract)


def resolve(preset=None, task=None, seed=0, contract=None):
    if contract != CONTRACT:
        return V1.resolve(preset, task, seed, contract)
    if task is not None:
        selected_contract(task, CONTRACT)
    # The released resolver checks preset, task, seed and ownership exactly as for v1; only the contract differs.
    v = V1.resolve(preset, task, seed, V1.CONTRACT)
    selected_contract(v["task"], CONTRACT)
    v["contract"] = CONTRACT
    return v


task_id = V1.task_id


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    return dict(task_id=task_id(v), description=instruction(v["task"], v.get("contract")),
                backend_options=dict(v), selection=v, split=None)


def contract():
    out = V1.legacy.contract()
    out.update(selectors=["preset", "task", "seed", "contract"],
               tasks={p: list(TASKS[p]) for p in POLICY_PRESETS},
               contracts={t: ([PURCHASE_CONTRACT] if t == "bag-purchase" else
                              [V1.CONTRACT, CONTRACT] if t in NEW_TASKS else
                              [LEGACY, V1.CONTRACT, CONTRACT] if t in ORDER else [LEGACY])
                          for p in POLICY_PRESETS for t in TASKS[p]},
               task_set=TASK_SET, task_set_v1=V1.TASK_SET)
    out["unavailable_tasks"] = {f"{p}/{t}": out["availability"][p]["reason"] for p in POLICY_PRESETS
                                for t in TASKS[p] if not out["availability"][p]["available"]}
    out["qualification"] += (f" The task set ({CONTRACT}) and its three added starts are render-checked ("
                             "every start built, rendered and idled, untouched starts score 0) but not task-audited; "
                             f"task_set_v1 ({V1.CONTRACT}) is the released, code-only set. Legacy selections keep "
                             "their old rules.")
    return out

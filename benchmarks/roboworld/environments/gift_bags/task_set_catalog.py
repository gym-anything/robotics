"""Opt-in task catalogue, outside probes.PHYSICS_FILES. Legacy selections stay legacy."""
from dataclasses import asdict, dataclass

from .config import Config as LegacyConfig, TASKS as LEGACY_TASKS
from . import variants as legacy
from .task import INSTRUCTIONS as LEGACY_INSTRUCTIONS, SETTLE_TASKS as LEGACY_SETTLE

CONTRACT = "gift-bags-tasks-v1"
PURCHASE_CONTRACT = "bag-purchase-v1"
LEGACY = "legacy"
NEW_TASKS = {"open-flaps": "receiving", "put-lid-on": "home", "bag-candle": "counter"}
BASE_TASK = {"open-flaps": "open-carton", "put-lid-on": "lift-lid", "bag-candle": "bag-purchase"}
TASKS = {p: ts + tuple(t for t, owner in NEW_TASKS.items() if owner == p) for p, ts in LEGACY_TASKS.items()}
POLICY_PRESETS = legacy.POLICY_PRESETS
availability, default_preset = legacy.availability, legacy.default_preset
ORDER = ("lift-lid", "open-flaps", "put-lid-on", "bag-candle", "open-carton", "open-bag", "unwrap-gift", "bag-purchase")
INSTRUCTIONS = {
    **LEGACY_INSTRUCTIONS,
    "open-bag": "You are at the cash-wrap counter of a boutique. Take the top gift bag from the stack, open it, "
                "square its bottom and put one of the bottom boards in it. Leave the bag standing open on the "
                "dark rubber counter mat. Release everything when you are done.",
    "open-carton": "You are at the receiving table in a back room. A supply carton sealed with brown packing tape "
                   "stands on the stainless table. Open it and leave all four top flaps open, the contents inside "
                   "and the carton upright where it is. The tape has a folded-back tab; no knife is provided. "
                   "Release everything when you are done.",
    "open-flaps": "You are at the receiving table in a back room. A supply carton stands on the stainless table "
                  "with its top flaps closed but not taped. Open it: fold all four top flaps open so that they stay "
                  "open. Leave the contents inside and the carton upright where it is. Release everything when "
                  "you are done.",
    "put-lid-on": "A gift box stands open on the dining table with the photo frame inside and its lid on the table "
                  "beside it. Put the lid back on the box and release it. Leave the frame in the box.",
    "bag-candle": "You are at the cash-wrap counter of a boutique. A gift bag stands open on the dark rubber counter "
                  "mat with a bottom board in it. Put the customer's boxed candle in the bag, upright on the board, "
                  "and leave the bag standing open on the mat. Release everything when you are done.",
}


def selected_contract(task, contract=None):
    if task == "bag-purchase":
        if contract not in (None, LEGACY, PURCHASE_CONTRACT):
            raise ValueError("bag-purchase keeps its own contract")
        return PURCHASE_CONTRACT
    if task in NEW_TASKS:
        if contract not in (None, CONTRACT):
            raise ValueError(f"{task} requires {CONTRACT}")
        return CONTRACT
    if contract not in (None, LEGACY, CONTRACT) or (contract == CONTRACT and task not in ORDER):
        raise ValueError(f"Unsupported contract {contract!r} for {task}")
    return contract or LEGACY


def instruction(task, contract=None):
    return (INSTRUCTIONS if selected_contract(task, contract) == CONTRACT else LEGACY_INSTRUCTIONS)[task]


def evaluation_contract(task, contract=None):
    c = selected_contract(task, contract)
    return f"{task}/{c}" if c in (CONTRACT, LEGACY) else c


def completes_window(task, contract=None):
    return task in LEGACY_SETTLE or (task in ("open-bag", "bag-candle") and selected_contract(task, contract) == CONTRACT)


@dataclass(frozen=True)
class Config(LegacyConfig):
    contract: str = None

    def __post_init__(self):
        # Delegate every old field's validation, including its accepted types, to the unmodified Config.
        values = asdict(self)
        values.pop("contract")
        values["task"] = BASE_TASK.get(self.task, self.task)
        LegacyConfig(**values)
        selected_contract(self.task, self.contract)


def resolve(preset=None, task=None, seed=0, contract=None):
    if task in NEW_TASKS:
        owner = NEW_TASKS[task]
        v = legacy.resolve(owner if preset is None else preset, BASE_TASK[task], seed)
        if v["preset"] != owner:
            raise ValueError(f"{task} is not a {v['preset']} task")
        v["task"] = task
    else:
        v = legacy.resolve(preset, task, seed)
    c = selected_contract(v["task"], contract)
    # Canonical legacy selections retain their exact shape and task ID.
    if c == CONTRACT:
        v["contract"] = c
    return v


def task_id(v):
    version = f"/{v['contract']}" if v.get("contract") else ""
    return f"gift-bags/v1/{v['preset']}/{v['task']}{version}/seed-{v['seed']}"


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    return dict(task_id=task_id(v), description=instruction(v["task"], v.get("contract")),
                backend_options=dict(v), selection=v, split=None)


TASK_SET = [dict(preset=next(p for p in POLICY_PRESETS if t in TASKS[p]), task=t, seed=0,
                 **({"contract": CONTRACT} if t != "bag-purchase" else {})) for t in ORDER]


def contract():
    out = legacy.contract()
    out.update(selectors=["preset", "task", "seed", "contract"],
               tasks={p: list(TASKS[p]) for p in POLICY_PRESETS},
               contracts={t: ([PURCHASE_CONTRACT] if t == "bag-purchase" else [CONTRACT] if t in NEW_TASKS else
                              [LEGACY, CONTRACT] if t in ORDER else [LEGACY])
                          for p in POLICY_PRESETS for t in TASKS[p]},
               task_set=TASK_SET)
    out["unavailable_tasks"] = {f"{p}/{t}": out["availability"][p]["reason"] for p in POLICY_PRESETS
                                for t in TASKS[p] if not out["availability"][p]["available"]}
    out["qualification"] += (" The opt-in task set and its three added starts are code only: not yet run with the "
                             "simulator or render-checked. Legacy selections keep their old rules.")
    return out

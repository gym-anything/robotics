"""Opt-in task contracts; the original variants.py and its defaults stay frozen."""
from . import variants as legacy
from .task import INSTRUCTIONS as OLD_INSTRUCTIONS, TIME_LIMIT_S

CONTRACT = "filleting-tasks-v1"
LEGACY = "legacy"
NEW_TASK = "A5"
TASK_SLUG = {**legacy.TASK_SLUG, NEW_TASK: "put-away-knife"}
SLUG_TASK = {v: k for k, v in TASK_SLUG.items()}
INSTRUCTIONS = {**OLD_INSTRUCTIONS, NEW_TASK:
    "You are done at the board. Put the fillet knife back in its slot in the knife block."}
LIMITS = {**TIME_LIMIT_S, NEW_TASK: 60.}
REVISED = ("A0", "A1", "B5", "C3", "B3", "C1")
EIGHT = ("A0", NEW_TASK, "A1", "B5", "C3", "B3", "C1", "A3")
DIFFICULTY = dict(zip(EIGHT, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
TASK_SET = tuple(dict(preset=legacy.PRESET_SLUG[t[0]], task=TASK_SLUG[t], seed=0,
                      **({"contract": CONTRACT} if t in REVISED else {})) for t in EIGHT)


def resolve(preset=None, task=None, seed=0, contract=None):
    code = SLUG_TASK.get(task, task) if task else None
    if code == NEW_TASK:
        # Reuse *all* old validation/coercion (notably bool seeds and empty presets).
        chosen = legacy.resolve(preset, "A0", seed)
        chosen["task"] = NEW_TASK
        if contract not in (None, CONTRACT):
            raise ValueError(f"{TASK_SLUG[code]} only supports {CONTRACT}")
        chosen["contract"] = CONTRACT
        return chosen
    chosen = legacy.resolve(preset, task, seed)
    if contract not in (None, LEGACY, CONTRACT):
        raise ValueError(f"Unknown filleting task contract {contract!r}")
    if contract == CONTRACT:
        if chosen["task"] not in REVISED:
            raise ValueError("This task keeps its original contract; omit contract")
        chosen["contract"] = CONTRACT
    return chosen


def task_resolve(selection):
    chosen = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0),
                     selection.get("contract"))
    if "contract" not in chosen:
        return legacy.task_resolve(selection)
    p, t, seed = chosen["preset"], chosen["task"], chosen["seed"]
    # Existing ids are retained: the explicit selection and evaluation identify the contract.
    return dict(task_id=f"filleting/v1/{legacy.PRESET_SLUG[p]}/{TASK_SLUG[t]}/seed-{seed}",
                description=INSTRUCTIONS[t], selection=chosen, split=None,
                backend_options=dict(preset=p, task=t, seed=seed, contract=CONTRACT))


def start_options(chosen, config=None):
    """No new placements/props. A5 uses the cutting tasks' knife-in-hand start in the A0 scene."""
    values = dict(config or {})
    task = chosen["task"]
    if task == NEW_TASK:
        if "knife_start" in values and values["knife_start"] not in (None, "hand"):
            raise ValueError("put-away-knife starts with the knife in hand")
        values["knife_start"] = "hand"
        task = "A0"  # scene carrier only; never the public task/evaluator
    return task, values

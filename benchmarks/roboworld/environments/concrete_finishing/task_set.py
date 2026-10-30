"""Opt-in task contracts; the original catalogue, IDs and launch defaults remain valid.

No scene changes: all eight reuse layout.TASK_SETUP and its seeded material presets.
Kept outside probes.common.PHYSICS and STAMP_EXTRA, as is the evaluator adapter.
"""

from . import variants

CONTRACT = "concrete-tasks/progress-v1"
LEGACY = "criteria-share-v1"
PAD_FINISH = "pad-finish/progress-v3"
TASKS = ("tool-sealant", "patch-drywall", "pad-darby", "pad-edge", "patch-flush",
         "tool-joints", "pad-trowel", "pad-finish")
SELECTORS = (*variants.SELECTORS, "contract")
DIFFICULTY = dict(zip(TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))


def selected_contract(task, contract=None):
    """Validate only the added option; existing selection validation is delegated unchanged."""
    default = PAD_FINISH if task == "pad-finish" else LEGACY
    if contract is None or contract == default:
        return default
    if task in TASKS[:-1] and contract == CONTRACT:
        return CONTRACT
    raise ValueError(f"Unknown contract {contract!r} for {task}; choose {default}"
                     + (f" or {CONTRACT}" if task in TASKS[:-1] else ""))


def task_resolve(selection):
    result = variants.task_resolve(selection)
    chosen = selected_contract(result["selection"]["task"], selection.get("contract"))
    if chosen == CONTRACT:
        for key in ("selection", "backend_options"):
            result[key] = dict(result[key], contract=chosen)
    return result


def selections():
    return [dict(preset="b012", task=t, seed=0, variant="default",
                 **({"contract": CONTRACT} if t != "pad-finish" else {})) for t in TASKS]


def contract():
    return dict(variants.contract(), selectors=list(SELECTORS),
                contract_choices={t: [selected_contract(t)] + ([CONTRACT] if t in TASKS[:-1] else [])
                                  for t in variants.TASKS},
                task_set=selections())

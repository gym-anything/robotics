"""Opt-in eight-task catalogue. Legacy selectors and physics-hashed files stay unchanged."""

from . import variants as legacy
from .task import INSTRUCTIONS as LEGACY_INSTRUCTIONS

CONTRACT = "drilling-tasks-v1"
LEGACY = "v1"
TASKS = ("take-drill", "press-drill", "punch-layout", "press-blind", "panel-holes",
         "timber-bolt", "match-bolt", "ledger-anchor")
PROGRESS_SCORED = frozenset(TASKS[:-1])
NO_SCORE_WHEN_UNEVALUABLE = PROGRESS_SCORED
DIFFICULTY = dict(zip(TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
INSTRUCTIONS = dict(LEGACY_INSTRUCTIONS)
INSTRUCTIONS["timber-bolt"] = (
    "Bore a 9/16-inch hole through both boards at the mark, square to their faces, "
    "and pass the 1/2-inch carriage bolt through by hand.")
TASK_SET = tuple(dict(preset="wall" if t == "ledger-anchor" else "bench", task=t, seed=0,
                      **({"contract": CONTRACT} if t in PROGRESS_SCORED else {})) for t in TASKS)


def validate_contract(task, value=None):
    if value is None or value == LEGACY:
        return LEGACY
    if value != CONTRACT or task not in PROGRESS_SCORED:
        raise ValueError(f"Contract {value!r} is not offered for {task!r}; use v1" +
                         (f" or {CONTRACT}" if task in PROGRESS_SCORED else ""))
    return CONTRACT


def instruction(task, contract=LEGACY):
    return (INSTRUCTIONS if validate_contract(task, contract) == CONTRACT else LEGACY_INSTRUCTIONS)[task]


def task_resolve(selection):
    # Delegate all existing seed typing, falsy defaults and preset inference to the old resolver.
    out = legacy.task_resolve(selection)
    task = out["selection"]["task"]
    chosen = validate_contract(task, selection.get("contract"))
    if chosen == LEGACY:
        return out
    v = out["selection"]
    out["task_id"] = f"drilling/v1/{v['preset']}/{task}/{chosen}/seed-{v['seed']}"
    out["description"] = instruction(task, chosen)
    out["selection"]["contract"] = chosen
    out["backend_options"]["contract"] = chosen
    return out


def contract():
    out = legacy.contract()
    out["selectors"] = [*out["selectors"], "contract"]
    out["contracts"] = {t: [LEGACY, CONTRACT] if t in PROGRESS_SCORED else [LEGACY]
                        for names in legacy.PUBLIC.values() for t in names}
    out["task_set"] = [dict(s) for s in TASK_SET]
    out["contract_note"] = "v1 is the unchanged default; ledger-anchor keeps its own progress v3 scoring."
    return out

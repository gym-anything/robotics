"""filleting-tasks-v2: the render-checked eight-task set, beside filleting-tasks-v1 (task_set_catalogue.py).

Same tasks, scenes, starts, instructions and time limits as filleting-tasks-v1. What differs (render-check fix round):
- a selection scored under this contract carries it in its task id, before the seed
  (filleting/v1/<preset>/<task>/filleting-tasks-v2/seed-<n>);
- take-knife's success is the original right-hand verdict (released_checks in tasks/take-knife/verifier.py); its
  progress measures the knife, whichever hand holds it, so a left-hand hold earns progress but never success;
- only head-off accepts a knife put down; portion, medallions, excise-defect and remove-chain keep their original
  checks.
A v2 selection names the contract explicitly (bare put-away-knife stays filleting-tasks-v1). fillet-whole-fish keeps its
own evaluator and its released selection. Every selection without this contract resolves as released.
"""
from . import task_set_catalogue as v1, variants as legacy
from .task_set_catalogue import (DIFFICULTY, EIGHT, INSTRUCTIONS, LIMITS, NEW_TASK, REVISED, SLUG_TASK, TASK_SLUG,
                                 start_options)

CONTRACT = "filleting-tasks-v2"
# The eight as benchmark.json's task_set lists them: seven under this contract; fillet-whole-fish (A3) with its
# released selection, exactly as splits/test.json has it.
TASK_SET = tuple(dict(preset=legacy.PRESET_SLUG[t[0]], task=TASK_SLUG[t], seed=0, contract=CONTRACT) if t != "A3"
                 else dict(preset=legacy.PRESET_SLUG["A"], task=TASK_SLUG["A3"]) for t in EIGHT)


def resolve(preset=None, task=None, seed=0, contract=CONTRACT):
    """filleting-tasks-v1's validation and coercion (task_set_catalogue.resolve), with this contract recorded."""
    if contract != CONTRACT:
        raise ValueError(f"task_set_v2_catalogue resolves {CONTRACT} selections only, not {contract!r}")
    chosen = v1.resolve(preset, task, seed, v1.CONTRACT)
    chosen["contract"] = CONTRACT
    return chosen


def task_id(preset, task, seed):
    """Id of a selection scored under CONTRACT: the legacy id with the contract before the seed."""
    return f"filleting/v1/{legacy.PRESET_SLUG[preset]}/{TASK_SLUG[task]}/{CONTRACT}/seed-{seed}"


def task_resolve(selection):
    chosen = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0),
                     selection.get("contract"))
    p, t, seed = chosen["preset"], chosen["task"], chosen["seed"]
    # The contract is part of the id, so results stored under an id follow one scoring rule.
    return dict(task_id=task_id(p, t, seed),
                description=INSTRUCTIONS[t], selection=chosen, split=None,
                backend_options=dict(preset=p, task=t, seed=seed, contract=CONTRACT))

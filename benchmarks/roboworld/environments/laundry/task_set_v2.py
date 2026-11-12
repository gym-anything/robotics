"""Task selection for the render-checked eight-task set, contract `laundry-tasks-v2`.

`task_set.resolve` and the GAR backend send a selection here only when it names this
contract; every other selection (the released `laundry-tasks-v1` set, the bare garment
selections, the legacy and original shirt contracts) keeps the released code path.

Differences from `laundry-tasks-v1` (fix round, render-checked on the GPU):
- `tee/half_fold` and `towel/roll` replace `asymmetric/tie_untangle` and
  `jeans/leg_align_fold` in the eight. On Linux the unchanged P13 reset loader rejects the
  asymmetric top and the jeans ("P13 reset topology no longer matches constructed
  garment"); the loader is environment code and stays as released. Both tasks stay
  registered and selectable under this contract, outside the eight.
- Every start settles before its zero point is taken (`task_set_v2_env`).
- `thirds_fold` counts cloth crossings, so a correct letter fold passes (tasks/thirds_fold/verifier.py).
- The ties are bunched on their free distal parts, crossing-free (`task_set_v2_presets`).
The shirt keeps its original selection `{"garment": "shirt"}` and released evaluator.
Importing or resolving never constructs a world.
"""
from dataclasses import replace

from .garment_task import GARMENT_TASK_SPECS, GarmentTaskSpec
from .newton15.config import GARMENTS
from .task_set import (DEFAULTS, FAILURE_FACTOR, SHIRT_CONTRACT, SUFFIX, TASKS as TASKS_V1, Task,
                       V2_CONTRACT as CONTRACT, task_goals)

TASKS = TASKS_V1 + (
    # Swapped in for asymmetric/tie_untangle and jeans/leg_align_fold, which
    # stay registered outside the eight.
    Task("tee", "half_fold", "medium", "Fold the T-shirt in half lengthwise, one sleeve onto the other, side seams together"),
    Task("towel", "roll", "hard", "Roll the bath towel up from one short end into a roll"),
)
BY_ID = {(t.garment, t.name): t for t in TASKS}
TASK_SET_ORDER = (("towel", "spread_flat"), ("towel", "half_fold"), ("tee", "half_fold"), ("tee", "sleeve_tuck_fold"),
                  ("tee", "turn_face_down"), ("towel", "thirds_fold"), ("towel", "roll"), ("shirt", "sleeve_fold"))
TASK_SET = tuple(
    {"garment": garment, "task": name, "contract": CONTRACT} if garment != "shirt" else {"garment": "shirt"}
    for garment, name in TASK_SET_ORDER
)


def task_id(entry):
    """The v2 task id: the released id with the contract appended, so v2 results never share a v1 id."""
    return f"{entry.id}/{CONTRACT}"


def select(garment="towel", task=None, contract=CONTRACT):
    """Backend selection under this contract only; Config's garment validation is unchanged."""
    if contract != CONTRACT:
        raise ValueError(f"task_set_v2 serves only contract {CONTRACT!r}, not {contract!r}")
    if garment not in GARMENTS:
        raise ValueError(f"Unknown garment: {garment}")
    name = DEFAULTS[garment] if task is None else task
    if (garment, name) not in BY_ID:
        raise ValueError(f"Unknown task {name!r} for garment {garment!r}")
    entry = BY_ID[garment, name]
    if garment == "shirt":
        raise ValueError(f"Contract {contract!r} unavailable for {entry.id}; choose {[SHIRT_CONTRACT]} "
                         "(the eight keep the original selection {'garment': 'shirt'})")
    return entry, contract


def resolve(selection):
    garment = selection.get("garment", "towel")
    entry, version = select(garment, selection.get("task"), selection.get("contract"))
    return dict(task_id=task_id(entry), description=entry.label + SUFFIX,
                backend_options={"config": {"garment": garment}, "task": entry.name, "contract": version},
                selection={"garment": garment, "task": entry.name, "contract": version}, split=None)


def task_spec(entry, version):
    if version != CONTRACT:
        raise ValueError(f"task_set_v2 serves only contract {CONTRACT!r}")
    from .task_set_v2_goals import ContinuousGoal
    # towel/spread_flat, tee/turn_face_down, towel/thirds_fold, towel/roll, tee/half_fold: tasks/<task>/verifier.py
    goals = task_goals(entry, version)
    if goals is not None:
        spec = GarmentTaskSpec(entry.name, entry.label, GARMENTS[entry.garment], goals,
                               maximum_structural_strain=.20 if entry.garment == "towel" else .30)
    else:
        original = GARMENT_TASK_SPECS[(GARMENTS[entry.garment], entry.name)]
        spec = replace(original, label=entry.label, goals=tuple(ContinuousGoal(g) for g in original.goals))
    return replace(spec, rest_window_s=.5, limit_compression=False, tabletop_support_fraction=.95,
                   full_credit_requires_success=True, partial_credit_cap=FAILURE_FACTOR)


def make_evaluator(entry, version, index, positions):
    from .task_set_v2_goals import StartRelativeEvaluator
    spec = task_spec(entry, version)
    for goal in spec.goals:
        goal.validate(index)
    return StartRelativeEvaluator(spec, index, reference_positions_m=positions)

"""Explicit, versioned task selection; importing/resolving never constructs a world.

The original five garment selections and GARMENT_TASK_SPECS remain untouched.
New task logic lives outside the simulation, reset loader and original evaluator.
"""
from dataclasses import dataclass, replace

from .garment_task import GARMENT_TASK_SPECS, GarmentTaskSpec
from .newton15.config import GARMENTS

CONTRACT = "laundry-tasks-v1"
LEGACY = "laundry-semantic-legacy"
SHIRT_CONTRACT = "laundry-shirt-sleeve-fold-v1"
FAILURE_FACTOR = 0.9
SUFFIX = ". Use the robot arms; leave the garment released and resting on the table."
DEFAULTS = dict(tee="sleeve_tuck_fold", shirt="sleeve_fold", jeans="leg_align_fold",
                towel="half_fold", asymmetric="tie_untangle")


@dataclass(frozen=True)
class Task:
    garment: str
    name: str
    difficulty: str
    label: str
    start: str = "catalogue_p13"

    @property
    def id(self):
        return f"laundry/{self.garment}/{self.name}"


# Original ids/order first; append new tasks. The benchmark set is difficulty-ordered.
TASKS = (
    Task("tee", "sleeve_tuck_fold", "medium", "Fold both short sleeves in onto the T-shirt's body"),
    Task("shirt", "sleeve_fold", "hardest", "Fold both long sleeves and cuffs onto the shirt body", "at_rest"),
    Task("jeans", "leg_align_fold", "hard", "Fold the jeans in half lengthwise, one leg onto the other, hems together"),
    Task("towel", "half_fold", "entry", "Fold the bath towel in half, long edge to long edge"),
    Task("asymmetric", "tie_untangle", "medium",
         "The wrap top lies on the table with both waist ties folded back and bunched. Straighten both ties out to their full length",
         "bunched_ties"),
    Task("towel", "spread_flat", "entry", "The bath towel lies folded in half on the table. Unfold it and lay it out flat",
         "folded_towel"),
    Task("tee", "turn_face_down", "medium", "A T-shirt lies face up on the table. Turn it over so it lies face down, laid out flat"),
    Task("towel", "thirds_fold", "hard", "Fold the bath towel in thirds lengthwise, the way towels are folded for a linen shelf"),
)
BY_NAME = {t.name: t for t in TASKS}
TASK_SET_ORDER = ("spread_flat", "half_fold", "tie_untangle", "sleeve_tuck_fold", "turn_face_down",
               "thirds_fold", "leg_align_fold", "sleeve_fold")
TASK_SET = tuple(
    {"garment": BY_NAME[name].garment, "task": name, "contract": CONTRACT}
    if name != "sleeve_fold" else {"garment": "shirt"}
    for name in TASK_SET_ORDER
)


def select(garment="towel", task=None, contract=None):
    """Backend selection, preserving Config's original garment validation/coercion."""
    if garment not in GARMENTS:
        raise ValueError(f"Unknown garment: {garment}")
    name = DEFAULTS[garment] if task is None else task
    if name not in BY_NAME or BY_NAME[name].garment != garment:
        raise ValueError(f"Unknown task {name!r} for garment {garment!r}")
    entry = BY_NAME[name]
    original = name == DEFAULTS[garment]
    version = contract if contract is not None else (
        SHIRT_CONTRACT if garment == "shirt" else LEGACY if original else CONTRACT)
    allowed = {SHIRT_CONTRACT} if garment == "shirt" else ({LEGACY, CONTRACT} if original else {CONTRACT})
    if version not in allowed:
        raise ValueError(f"Contract {version!r} unavailable for {entry.id}; choose {sorted(allowed)}")
    return entry, version


def resolve(selection):
    # Keep the historical bare garment resolver's output, including its defaults.
    if "task" not in selection and "contract" not in selection:
        from .newton15.benchmark import resolve as original_resolve
        return original_resolve(selection)
    garment = selection.get("garment", "towel")
    entry, version = select(garment, selection.get("task"), selection.get("contract"))
    label = (GARMENT_TASK_SPECS[(GARMENTS[garment], entry.name)].label
             if version != CONTRACT else entry.label)
    return dict(task_id=entry.id, description=label + SUFFIX,
                backend_options={"config": {"garment": garment}, "task": entry.name, "contract": version},
                selection={"garment": garment, "task": entry.name, "contract": version}, split=None)


def task_spec(entry, version):
    if version != CONTRACT:
        return GARMENT_TASK_SPECS[(GARMENTS[entry.garment], entry.name)]
    from .task_set_goals import ContinuousGoal, FlatSpreadGoal, FaceDownGoal, ThirdsFoldGoal
    new = {"spread_flat": FlatSpreadGoal, "turn_face_down": FaceDownGoal, "thirds_fold": ThirdsFoldGoal}
    if entry.name in new:
        spec = GarmentTaskSpec(entry.name, entry.label, GARMENTS[entry.garment],
                               (new[entry.name](),), maximum_structural_strain=.20 if entry.garment == "towel" else .30)
    else:
        original = GARMENT_TASK_SPECS[(GARMENTS[entry.garment], entry.name)]
        spec = replace(original, label=entry.label, goals=tuple(ContinuousGoal(g) for g in original.goals))
    return replace(spec, rest_window_s=.5, limit_compression=False, tabletop_support_fraction=.95,
                   full_credit_requires_success=True, partial_credit_cap=FAILURE_FACTOR)


def make_evaluator(entry, version, index, positions):
    from .garment_task import GarmentTaskEvaluator
    from .task_set_goals import StartRelativeEvaluator
    cls = StartRelativeEvaluator if version == CONTRACT else GarmentTaskEvaluator
    spec = task_spec(entry, version)
    for goal in spec.goals:
        goal.validate(index)
    return cls(spec, index, reference_positions_m=positions)

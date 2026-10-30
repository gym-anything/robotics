"""Explicit, versioned configuration. Unknown fields fail (dataclass constructor)."""

from dataclasses import dataclass, field, asdict

PRESETS = ("parcel", "specimen", "part")
TASKS = {
    "parcel": ("carrier-top", "gs1-case-side", "cover-old-label", "press-down", "smooth-corner", "finish-securing"),
    "specimen": ("tube-wrap", "tube-label"),
    "part": ("part-id-hdpe", "part-id-match"),
}
# part-id-steel (a label anywhere on a featureless plate) is retired; part-id-match (the same plate, three of them,
# label to the one whose part number it carries, clear of its marking) replaces it. press-down finishes pressing a
# label already laid on the carton.
RETIRED_TASKS = {"part-id-steel": "retired; replaced by part-id-match (preset part)"}
# Task ladder: three tasks scored by the continuous progress reward (task.py, REWARD_CONTRACTS). The two repair
# rungs start from a label already stuck down on the carton except for visibly lifted areas (ladder.py) and share
# press-down's carton, label stock and readers; their placement rules are carrier-top's (either flap half,
# task._place_repair_rung). tube-label is tube-wrap's job, seed for seed (same bench, tube, label and checks, its
# barcode crease read over the episode), under its own description and the progress reward: ten task ids, nine
# distinct jobs. tube-label also checks two requirements its instruction states and tube-wrap's checks do not (the
# label over the manufacturer's label; the patient's name toward the cap, task._place_tube_label) and is scored by
# the take-and-apply progress rule (task.REWARD_CONTRACTS): still tube-wrap's job, held to its stricter reading.
LADDER_TASKS = ("smooth-corner", "finish-securing", "tube-label")
REPAIR_TASKS = ("smooth-corner", "finish-securing")
# The other seven tasks are scored by the progress reward too (task.REWARD_CONTRACTS). Tasks whose label starts on
# the item have their progress measured as the work done on what was left to do, from each label node's credit in
# the settled start state (LabelEnv stores it).
START_CREDIT_TASKS = REPAIR_TASKS + ("press-down",)
BASE_TASK = {"smooth-corner": "press-down", "finish-securing": "press-down", "tube-label": "tube-wrap"}


def base_task(task):
    """The task whose bench layout, label stock and readers a task shares (itself if none). tube-label also shares
    tube-wrap's placement rules; the repair rungs use carrier-top's (task._place_repair_rung)."""
    return BASE_TASK.get(task, task)


@dataclass(frozen=True)
class Config:
    preset: str = "parcel"
    task: str = "carrier-top"
    seed: int = 0
    robot: str = "dual-fr3"             # catalog alias (see robots/catalog.py ENVIRONMENTS)
    device: str = "cuda:0"
    rigid_dt: float = 1e-3              # s, MuJoCo robot/rigid step (FR3 1 kHz control)
    label_substeps: int = 2             # label solver dt = rigid_dt * label_substeps
    label_mesh_m: float = 3e-3          # label triangle size for 4x6 / 2x4 stocks
    tube_label_mesh_m: float = 1.5e-3   # finer mesh for the 13 mm tube wrap
    newton_iters: int = 8               # base Newton cap per label step
    newton_max: int = 32                # adaptive continuation for unconverged steps (labelsim)
    stop_on_tear: bool = True           # a torn label ends the episode and is frozen (see world.step)
    contact_capacity: int = 2048
    solver_iterations: int = 60
    settle_s: float = 0.4               # reset: printer-presented label settles under gravity
    carton_content_kg: float = 2.0
    finger_effort_n: float = 35.0       # per finger; Franka Hand 70 N continuous total [search]
    record_trace: bool = True

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        if self.task in RETIRED_TASKS:
            raise ValueError(f"task {self.task!r} was {RETIRED_TASKS[self.task]}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"task {self.task!r} is not a {self.preset} task: {TASKS[self.preset]}")
        if not (1e-4 <= self.rigid_dt <= 2e-3) or self.label_substeps < 1:
            raise ValueError("rigid_dt in [0.1, 2] ms and label_substeps >= 1")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def label_dt(self):
        return self.rigid_dt*self.label_substeps

    def to_dict(self):
        return asdict(self)

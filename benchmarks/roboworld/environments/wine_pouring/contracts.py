"""Task reward versions, independent of the scene and simulator runtime.

A selection without a contract keeps wine-checks-v1. The task set selects
wine-tasks-v1 explicitly; split-bottle always keeps split-progress-v1.
"""

from dataclasses import dataclass

from .config import Config
from .task import SPLIT_REWARD_CONTRACT

LEGACY_REWARD_CONTRACT = "wine-checks-v1"
TASKS_V1_REWARD_CONTRACT = "wine-tasks-v1"
EIGHT_TASKS = (
    ("service", "carry-glass", "entry"),
    ("service", "taste-pour", "entry"),
    ("service", "pour-glass", "medium"),
    ("service", "refill-glass", "medium"),
    ("service", "water-tumbler", "medium"),
    ("tasting", "dump-and-pour", "hard"),
    ("lined", "lined-measure", "hard"),
    ("service", "split-bottle", "hardest"),
)
PROGRESS_TASKS = frozenset(t for _, t, _ in EIGHT_TASKS if t != "split-bottle")


def reward_contract(task, selected=""):
    default = SPLIT_REWARD_CONTRACT if task == "split-bottle" else LEGACY_REWARD_CONTRACT
    choices = (default, TASKS_V1_REWARD_CONTRACT) if task in PROGRESS_TASKS else (default,)
    if not isinstance(selected, str) or (selected and selected not in choices):
        raise ValueError(f"Unknown contract {selected!r} for {task}; choose {choices}")
    return selected or default


@dataclass
class TaskConfig(Config):
    """Task-facing extension; the physics Config and its provenance stay unchanged."""
    task_contract: str = ""

    def __post_init__(self):
        super().__post_init__()
        reward_contract(self.task, self.task_contract)


def eight_task_selections():
    return [dict(preset=p, task=t, seed=0,
                 contract=SPLIT_REWARD_CONTRACT if t == "split-bottle" else TASKS_V1_REWARD_CONTRACT)
            for p, t, _ in EIGHT_TASKS]

"""Task-only placements: the gift-bags-tasks-v1 start of each added task is in its setup.py (tasks/<task>/). Never edit
material, crease memory, contact or robot settings."""
from .config import task_setup
from .task_set_catalog import CONTRACT, NEW_TASKS


def prepare(spec, task):
    """The start of a task the task set adds (open-flaps, put-lid-on, bag-candle): its setup.py (tasks/<task>/)
    changes the base task's scene spec in place for gift-bags-tasks-v1. Other tasks: nothing."""
    if task in NEW_TASKS:
        task_setup(task).setup(spec, CONTRACT)

"""gift-bags-tasks-v2 start states: task_set_presets.py with the render-check fixes (open-flaps contact stock,
bag-candle's worked creases), in each added task's setup.py (tasks/<task>/). Task-only placements and start states.
Never edit materials, crease laws, contact, solver or robot settings."""
from .config import task_setup
from .task_set_v2_catalog import CONTRACT, NEW_TASKS


def prepare(spec, task):
    """The start of a task the task set adds (open-flaps, put-lid-on, bag-candle): its setup.py (tasks/<task>/)
    changes the base task's scene spec in place for gift-bags-tasks-v2. Other tasks: nothing."""
    if task in NEW_TASKS:
        task_setup(task).setup(spec, CONTRACT)

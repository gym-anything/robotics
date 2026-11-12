"""GAR constructs this task-only subclass for new tasks, native env for unlock."""

from lockkey.newton15.env import LockKeyEnv
from .catalogue import validate
from .presets import apply_start
from .scoring import TaskProgress


def task_env(config, task, contract=None):
    """A new task's environment at its start, as its setup.py's setup builds it: lockkey-tasks-v2's TaskLockKeyEnvV2
    (adapter_v2.py) under that contract, the released TaskLockKeyEnv below without one."""
    if contract is not None:
        from .adapter_v2 import TaskLockKeyEnvV2
        return TaskLockKeyEnvV2(config, task)
    return TaskLockKeyEnv(config, task)


class TaskLockKeyEnv(LockKeyEnv):
    def __init__(self, config, task):
        self.task = validate(task)
        super().__init__(config)
        self._prepare_task()

    def _prepare_task(self):
        stands, self.start_preset = apply_start(self, self.task)
        self.progress = TaskProgress(self, self.task, stands)
        # Captures the seated-at-release witness on the constructed state before
        # the robot can move its key. All subsequent updates run every native tick.
        self.evaluation = self._evaluate()
        self.initial.assign(self.state)

    def reset(self):
        with self.lock:
            super().reset()
            self._prepare_task()
            return self.get_state()

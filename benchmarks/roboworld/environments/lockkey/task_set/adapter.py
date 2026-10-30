"""GAR constructs this task-only subclass for new tasks, native env for unlock."""

from lockkey.newton15.env import LockKeyEnv
from .catalogue import validate
from .presets import apply_start
from .scoring import TaskProgress


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

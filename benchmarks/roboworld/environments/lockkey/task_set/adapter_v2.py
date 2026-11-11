"""lockkey-tasks-v2: GAR constructs this task-only subclass for the seven additions under the v2 contract.

Same native LockKeyEnv as v1 (adapter.py); v2 places the start (presets_v2.py), settles the constructed
free-key starts at setup (settle_v2.py) and scores from the settled start (scoring_v2.py).
"""

from lockkey.newton15.env import LockKeyEnv
from .catalogue_v2 import validate_contract, CONTRACT
from .presets_v2 import apply_start
from .scoring_v2 import TaskProgress
from . import settle_v2


class TaskLockKeyEnvV2(LockKeyEnv):
    def __init__(self, config, task):
        self.task = validate_contract(task, CONTRACT)
        super().__init__(config)
        self._prepare_task()

    def _prepare_task(self):
        stands, self.start_preset = apply_start(self, self.task)
        self.progress = TaskProgress(self, self.task, stands)
        # Captures the seated-at-release witness on the constructed state before
        # the robot can move its key. All subsequent updates run every native tick.
        self.evaluation = self._evaluate()
        record = settle_v2.settle(self, self.task)
        if record is not None:
            # Zero points (withdrawal, plug turn, shackle offset, key tilt) and the
            # history witnesses are read from the settled start; the release
            # witness is the constructed one.
            witness = self.progress.seated_at_release
            self.progress = TaskProgress(self, self.task, stands)
            self.progress.seated_at_release = witness
            self.start_preset = dict(self.start_preset, settle_checked=True, settle=record)
            self.evaluation = self._evaluate()
        self.initial.assign(self.state)

    def reset(self):
        with self.lock:
            super().reset()
            self._prepare_task()
            return self.get_state()

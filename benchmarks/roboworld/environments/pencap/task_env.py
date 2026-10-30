"""Task construction adapter around the unchanged physical environment.

Construction mirrors PenCapEnv, with one hook between build and finalize. The
inherited integration, controllers, state validation and action handling are
unmodified. Never monkey-patch the build function shared by other sessions.
"""

import threading
from .newton15.env import PenCapEnv
from .newton15.config import Config
from .newton15.build import build
from .newton15.runtime import require_engine
from .newton15.solver import create
from .tasks import DEFAULT_TASK, validate
from .task_presets import apply_start
from .task_scoring import SeatCapEvaluator, TaskEvaluator


class TaskPenCapEnv(PenCapEnv):
    def __init__(self, config=None, task=DEFAULT_TASK):
        self.task = validate(task)
        self.config = config or Config()
        if task != DEFAULT_TASK and self.config.scene != "studio":
            raise ValueError("The seven new pencap tasks require the studio scene")
        self.newton, self.wp = require_engine()
        self.lock = threading.RLock()
        with self.wp.ScopedDevice(self.config.device):
            self.builder, self.bodies, self.joints, self.materials = build(self.config)
            apply_start(self.builder, self.bodies, self.materials, self.config, task, self.newton, self.wp)
            (self.model, self.pipeline, self.solver, self.state, self.next_state,
             self.control, self.contacts) = create(self.builder, self.config, self.joints)
        self.initial = self.model.state()
        self.initial.assign(self.state)
        self.initial_targets = self.control.joint_target_q.numpy().copy()
        self.targets = self.initial_targets.copy()
        self.tick, self.fault, self.renderer, self.trace_callback = 0, None, None, None
        self.generation = 0
        self.task_start = task
        self._task_evaluator()

    def _task_evaluator(self):
        self.evaluator = SeatCapEvaluator() if self.task == DEFAULT_TASK else TaskEvaluator(self.task)
        self.evaluation = self.evaluator.sample(self)

    def reset(self):
        with self.lock:
            # Native reset owns all solver warm starts, contact matching and
            # target restoration. Only the task's sample history is replaced.
            super().reset()
            self._task_evaluator()
            return self.get_state()

    def get_state(self):
        return dict(super().get_state(), task=self.task, task_start=self.task_start)

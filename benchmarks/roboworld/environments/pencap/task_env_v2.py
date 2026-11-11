"""pencap-tasks-v2 task env: TaskPenCapEnv with the v2 start placement and verifier.

Construction is task_env.TaskPenCapEnv's, line for line, with task_presets_v2.apply_start
(moved parts written to their free-joint coordinates, so they survive eval_fk) and
task_scoring_v2.TaskEvaluator. Integration, controllers, state validation, actions and
the native reset are inherited unmodified. Never monkey-patch the shared build function.
"""

import threading
from .newton15.config import Config
from .newton15.build import build
from .newton15.runtime import require_engine
from .newton15.solver import create
from .task_env import TaskPenCapEnv
from .task_presets_v2 import apply_start
from .task_scoring_v2 import TaskEvaluator
from .task_set_v2 import CONTRACT, check


class TaskPenCapEnvV2(TaskPenCapEnv):
    contract = CONTRACT

    def __init__(self, config=None, task=None):
        self.task = check(task, CONTRACT)
        self.config = config or Config()
        if self.config.scene != "studio":
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
        self.evaluator = TaskEvaluator(self.task)
        self.evaluation = self.evaluator.sample(self)

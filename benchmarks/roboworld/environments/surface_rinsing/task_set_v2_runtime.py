"""surface-rinsing-tasks-v2 runtime: the task config and environment.

A config whose task_contract is not surface-rinsing-tasks-v2 is the released task_set.runtime one, unchanged. Under
v2 the scene is the revised task's (task_set_v2.setup) and the evaluator is task_set_v2_evaluator's.
"""
from dataclasses import fields

from .config import Config as LegacyConfig
from .env import RinsingEnv as LegacyEnv
from .task_set.runtime import Config as TaskSetConfig, RinsingEnv as TaskSetEnv
from .task_set_v2 import CONTRACT, NEW_TASK, resolve, setup


class Config(TaskSetConfig):
    def __post_init__(self):
        if self.task_contract != CONTRACT:
            return super().__post_init__()
        values = {f.name: getattr(self, f.name) for f in fields(LegacyConfig)}
        if self.task == NEW_TASK:
            values["task"] = "rinse-sponge"
        LegacyConfig(**values)
        resolve(task=self.task, seed=0, contract=self.task_contract)


class RinsingEnv(TaskSetEnv):
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True, water=True):
        config = config or Config()
        if config.task_contract != CONTRACT:
            super().__init__(config, output_dir=output_dir, spec=spec, settle=settle, water=water)
            return
        episode = spec if spec is not None else setup(config.task, config.seed, config.task_contract)
        LegacyEnv.__init__(self, config, output_dir=output_dir, spec=episode, settle=settle, water=water)
        if water:
            from .task_set_v2_evaluator import Evaluator
            old = self.evaluator
            self.world.recorders.remove(old.impacts)
            self.world.recorders.remove(old._track)
            self.evaluator = Evaluator(self)
            self.initial = self.checkpoint()

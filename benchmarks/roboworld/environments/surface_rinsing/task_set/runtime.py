"""Construct the task spec and evaluator without changing the original modules."""
from dataclasses import dataclass, fields

from ..config import Config as LegacyConfig
from ..env import RinsingEnv as LegacyEnv
from .catalogue import CONTRACT, NEW_TASK, TASKS, resolve, setup


@dataclass
class Config(LegacyConfig):
    task_contract: str = "legacy"

    def __post_init__(self):
        # Delegate all pre-existing options/type validation unchanged. The new
        # task uses the same pot-sink robot/fixture setup as rinse-sponge.
        values = {f.name: getattr(self, f.name) for f in fields(LegacyConfig)}
        if self.task == NEW_TASK:
            values["task"] = "rinse-sponge"
        LegacyConfig(**values)
        resolve(task=self.task, seed=0, contract=self.task_contract)

    @property
    def preset(self):
        return TASKS[self.task]["preset"]


class RinsingEnv(LegacyEnv):
    def __init__(self, config=None, *, output_dir=None, spec=None, settle=True, water=True):
        config = config or Config()
        episode = spec if spec is not None else setup(config.task, config.seed, config.task_contract)
        super().__init__(config, output_dir=output_dir, spec=episode, settle=settle, water=water)
        if config.task_contract == CONTRACT and water:
            from .evaluator import Evaluator
            old = self.evaluator
            self.world.recorders.remove(old.impacts)
            self.world.recorders.remove(old._track)
            self.evaluator = Evaluator(self)
            self.initial = self.checkpoint()

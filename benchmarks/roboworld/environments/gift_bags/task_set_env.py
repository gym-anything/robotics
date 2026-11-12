"""GAR constructs this adapter; the hashed environment remains byte-identical.

Only initialization uses a scene hook, for task placements/prop omission (no force, fixture or dynamics change).
The inherited action, checkpoint, restore and final-evaluation code runs unchanged.
"""
from dataclasses import asdict

from .config import Config as LegacyConfig, task_setup
from .env import GiftBagsEnv as LegacyEnv
from .task_set_catalog import BASE_TASK, CONTRACT, instruction, selected_contract, evaluation_contract
from .task_set_evaluator import Evaluator
from .task_set_presets import prepare


class GiftBagsEnv(LegacyEnv):
    def __init__(self, config, *, output_dir=None):
        self.selection = config
        modern = selected_contract(config.task, config.contract) == CONTRACT
        values = asdict(config)
        values.pop("contract")
        values["task"] = BASE_TASK.get(config.task, config.task)
        folder = task_setup(config.task)
        world = getattr(folder, "TASK_SET_WORLD", None) if modern else None
        if world is not None:
            values["task"] = world  # another task's world, with this task's own start (its setup.py)
        hook = None
        if config.task in BASE_TASK:
            hook = lambda spec: prepare(spec, config.task)
        elif world is not None:
            hook = lambda spec: folder.setup(spec, CONTRACT)
        super().__init__(LegacyConfig(**values), output_dir=output_dir, assisted_scene_hook=hook)
        self.assisted = False  # no running assistance: all subsequent actions are the inherited robot controls
        self.world.c = config  # renderer/trace metadata names the selected task; physical fields are identical
        self.evaluator = Evaluator(self, self.selection)

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            self.evaluator = Evaluator(self, self.selection)
            return self.get_state()

    def get_state(self, evaluate=True):
        out = super().get_state(evaluate=evaluate)
        out["instruction"] = instruction(self.selection.task, self.selection.contract)
        out["config"] = asdict(self.selection)
        out["evaluation"].setdefault("contract", evaluation_contract(self.selection.task, self.selection.contract))
        return out

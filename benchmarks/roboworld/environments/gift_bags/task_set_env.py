"""GAR constructs this adapter; the hashed environment remains byte-identical.

Only initialization uses a scene hook, for task placements/prop omission (no force, fixture or dynamics change).
The inherited action, checkpoint, restore and final-evaluation code runs unchanged.
"""
from dataclasses import asdict

from .config import Config as LegacyConfig
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
        if modern and config.task == "open-bag":
            values["task"] = "bag-purchase"  # bag-purchase's geometry/contact logic, but keep open-bag's candle placement
        hook = None
        if config.task in BASE_TASK:
            hook = lambda spec: prepare(spec, config.task)
        elif modern and config.task == "open-bag":
            def hook(spec):
                from .scene import CANDLE_XY, CANDLE_XY_PURCHASE
                import numpy as np
                # The seed transform is around the candle's own centre, so its translation is unchanged.
                b = next(b for b in spec.bodies if b.name == "candle")
                b.p[:2] += np.asarray(CANDLE_XY) - CANDLE_XY_PURCHASE
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

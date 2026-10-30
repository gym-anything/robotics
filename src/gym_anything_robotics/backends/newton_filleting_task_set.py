"""Task adapter for the filleting task set; the original backend stays unchanged."""
from dataclasses import replace

from .newton_filleting import NewtonFilletingBackend
from ..errors import UnsupportedConfigurationError
from filleting.task_set_catalogue import (CONTRACT, INSTRUCTIONS, LIMITS, NEW_TASK, TASK_SLUG,
                                          resolve, start_options)


class NewtonFilletingTaskSetBackend(NewtonFilletingBackend):
    def __init__(self, *, preset=None, task=None, seed=0, contract=None, robot="dual-fr3", config=None, **kwargs):
        chosen = resolve(preset, task, seed, contract)
        self.task_selection = chosen
        if "contract" not in chosen:
            # Same arguments, coercion, defaults, accepted config and validation as the old backend.
            super().__init__(preset=preset, task=task, seed=seed, robot=robot, config=config, **kwargs)
            return
        if chosen["task"] == NEW_TASK and robot != "dual-fr3":
            raise UnsupportedConfigurationError("put-away-knife is offered on dual-fr3 only; insertion is unqualified")
        carrier, values = start_options(chosen, config)
        super().__init__(preset=chosen["preset"], task=carrier, seed=seed, robot=robot, config=values, **kwargs)
        from filleting.task_set_env import TaskSetEnv
        self.env = TaskSetEnv(self.env, chosen["task"])
        notes = dict(self.capabilities.notes)
        notes.update(task=f"{TASK_SLUG[chosen['task']]} ({chosen['task']}) seed {seed}",
                     evaluator=f"Physical-state evaluator, contract {CONTRACT} (filleting/task_set_evaluator.py)")
        self.capabilities = replace(self.capabilities, notes=notes)

    def policy_access(self):
        access = super().policy_access()
        if "contract" not in self.task_selection:
            return access
        task = self.task_selection["task"]
        return replace(access, instruction=lambda: INSTRUCTIONS[task],
                       hardware={**access.hardware, "time_limit_s": LIMITS[task]})

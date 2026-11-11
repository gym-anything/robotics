"""Task adapter for the filleting task set; the original backend stays unchanged."""
from dataclasses import replace

from .newton_filleting import NewtonFilletingBackend
from ..errors import UnsupportedConfigurationError
from filleting.task_set_catalogue import (CONTRACT, INSTRUCTIONS, LIMITS, NEW_TASK, TASK_SLUG,
                                          resolve, start_options)


class NewtonFilletingTaskSetBackend(NewtonFilletingBackend):
    def __init__(self, *, preset=None, task=None, seed=0, contract=None, robot="dual-fr3", config=None, **kwargs):
        if contract == "filleting-tasks-v2":
            self._init_v2(preset=preset, task=task, seed=seed, robot=robot, config=config, **kwargs)
            return
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

    def _init_v2(self, *, preset, task, seed, robot, config, **kwargs):
        """filleting-tasks-v2 (filleting/task_set_v2_*.py): the same scene carriers and start options as the task set
        above; the v2 facade and evaluator judge the task. Instructions and time limits are the task set's."""
        from filleting.task_set_v2_catalogue import CONTRACT as V2, resolve as resolve_v2
        from filleting.task_set_v2_env import TaskSetV2Env
        from filleting.variants import PRESET_SLUG
        chosen = resolve_v2(preset, task, seed, V2)
        self.task_selection = chosen
        if chosen["task"] == NEW_TASK and robot != "dual-fr3":
            raise UnsupportedConfigurationError("put-away-knife is offered on dual-fr3 only; insertion is unqualified")
        carrier, values = start_options(chosen, config)
        super().__init__(preset=chosen["preset"], task=carrier, seed=seed, robot=robot, config=values, **kwargs)
        t, c = chosen["task"], self.config
        self.env = TaskSetV2Env(self.env, t)
        self.capabilities = replace(self.capabilities, notes={
            **self.capabilities.notes,
            "task": f"{PRESET_SLUG[c.preset]}/{TASK_SLUG[t]} ({t}) seed {c.seed}, workpiece {c.workpiece}, "
                    f"contract {V2}",
            "evaluator": f"Physical-state evaluator, contract {V2} (filleting/task_set_v2_evaluator.py)"})

    def policy_access(self):
        access = super().policy_access()
        if "contract" not in self.task_selection:
            return access
        task = self.task_selection["task"]
        return replace(access, instruction=lambda: INSTRUCTIONS[task],
                       hardware={**access.hardware, "time_limit_s": LIMITS[task]})

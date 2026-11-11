"""GAR's task adapter for sewing-tasks-v2 sessions (task_set_v2.py); task_set_env.make_environment hands that contract
here. The physics-hashed SewingEnv, Config and World stay byte-identical, and so does the released adapter.

As the released TaskEnvironment, with this set's physical bases (corner-seam builds the pivot's start, mesh and cached
state), its evaluator (task_set_v2_scoring) and time limits (corner-seam 180 s). The state reports this contract's
instruction, time limit and task_contract.
"""

from dataclasses import replace

from .env import SewingEnv
from .task_set_env import TaskEnvironment as ReleasedTaskEnvironment, SquareConfig
from .task_set_v2 import BASE_TASK, CONTRACT, TIME_LIMIT_S, choose_contract, instruction
from .task_set_v2_scoring import TaskEvaluator


class TaskEnvironment(ReleasedTaskEnvironment):
    def __init__(self, config, *, contract=None, output_dir=None):
        self.task_contract = choose_contract(config.task, contract)
        if self.task_contract != CONTRACT:
            raise ValueError(f"this adapter is for {CONTRACT} only")
        self.catalogue_config = config
        physical = replace(config, task=BASE_TASK.get(config.task, config.task))
        if config.task == "square-plies":
            # Unique cache: never restore an aligned load state as the slipped one.
            physical = SquareConfig(**vars(config))
        SewingEnv.__init__(self, physical, output_dir=output_dir)
        self.config = config
        if config.task == "square-plies":
            self.start_state = "tray_slipped"
        self.evaluator = TaskEvaluator(self)

    def get_state(self, evaluate=True):
        state = SewingEnv.get_state(self, evaluate=evaluate)
        state["instruction"] = instruction(self.config.task, self.config.robot, self.task_contract)
        if "terminated" in state:
            state["terminated"] = state["time_s"] >= TIME_LIMIT_S[self.config.task]
        state["task_contract"] = self.task_contract
        if "evaluation" in state:
            state["evaluation"]["task_contract"] = self.task_contract
        return state


def make_environment(config, *, contract=None, output_dir=None):
    return TaskEnvironment(config, contract=contract, output_dir=output_dir)

"""GAR's task adapter; the physics-hashed SewingEnv, Config and World stay byte-identical.

A benchmark task's start is its folder's setup.py (tasks/<task>/): setup gives the Config the world is built from.
Only square-plies adds an initial condition: its prepare applies it before the existing relaxation, and its start is
cached under a separate task/setup key (SquareConfig). Other new tasks reuse existing starts/meshes.
"""

from dataclasses import dataclass, replace
from functools import partial

from .env import SewingEnv
from .world import Config
from .task import TASKS, task_setup
from .task_set import BASE_TASK, CONTRACT, choose_contract, instruction
from .task_set_scoring import TaskEvaluator

SQUARE_SETUP = task_setup("square-plies").REVISION


@dataclass
class SquareConfig(Config):
    """The Config of square-plies' slipped start (tasks/square-plies/setup.py builds it): task_setup_revision keys a
    cached start state of its own."""
    task_setup_revision: str = SQUARE_SETUP


class TaskEnvironment(SewingEnv):
    def __init__(self, config, *, contract=None, output_dir=None):
        self.task_contract = choose_contract(config.task, contract)
        self.catalogue_config = config
        start = task_setup(config.task) if config.task in TASKS else None    # corner-seam starts in sewing-tasks-v2
        if start is not None:                 # a benchmark task: its setup.py gives the Config the world is built from
            physical = start.setup(self, config)
        else:
            physical = replace(config, task=BASE_TASK.get(config.task, config.task))
        super().__init__(physical, output_dir=output_dir)
        self.config = config
        if hasattr(start, "START_STATE"):
            self.start_state = start.START_STATE
        if self.task_contract == CONTRACT:
            self.evaluator = TaskEvaluator(self)

    def _make_start(self, use_cache=True):
        """The start's preparation; a task with an initial condition of its own (its setup.py prepare) wraps it."""
        prepare = getattr(task_setup(self.catalogue_config.task), "prepare", None)
        if prepare is None:
            super()._make_start(use_cache=use_cache)
            return
        prepare(self, partial(super()._make_start, use_cache=use_cache))

    def get_state(self, evaluate=True):
        state = super().get_state(evaluate=evaluate)
        state["instruction"] = instruction(self.config.task, self.config.robot, self.task_contract)
        state["task_contract"] = self.task_contract
        if "evaluation" in state:
            state["evaluation"]["task_contract"] = self.task_contract
        return state


def make_environment(config, *, contract=None, output_dir=None):
    from .task_set_v2 import CONTRACT as V2
    if contract == V2:                     # the render-checked set: task_set_v2_env.py
        from .task_set_v2_env import make_environment as v2
        return v2(config, contract=contract, output_dir=output_dir)
    # All GAR selections use the adapter, including legacy: its evaluator and
    # physical construction remain exactly the originals unless opted in.
    return TaskEnvironment(config, contract=contract, output_dir=output_dir)

"""GAR's task adapter; the physics-hashed SewingEnv, Config and World stay byte-identical.

Only square-plies adds an initial condition. It is applied before the existing relaxation
and cached under a separate task/setup key. Other new tasks reuse existing starts/meshes.
"""

from dataclasses import dataclass, replace
import numpy as np

from .env import SewingEnv
from .world import Config
from .task_set import BASE_TASK, CONTRACT, choose_contract, instruction
from .task_set_scoring import TaskEvaluator

SQUARE_SETUP = "tray-slipped-40x25mm-yaw6-v1"


@dataclass
class SquareConfig(Config):
    task_setup_revision: str = SQUARE_SETUP


def slip_top(positions):
    """Rigid initial placement: +40 mm x, +25 mm y, 6 degrees about the panel centre.

    Using the bounding-box centre makes the placement independent of mesh density.
    It changes positions only, before any episode or preparation tick.
    """
    out = np.array(positions, copy=True)
    centre = (out[:, :2].min(axis=0) + out[:, :2].max(axis=0))/2
    a = np.deg2rad(6.)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out[:, :2] = (out[:, :2] - centre) @ rotation.T + centre + [.040, .025]
    return out


class TaskEnvironment(SewingEnv):
    def __init__(self, config, *, contract=None, output_dir=None):
        self.task_contract = choose_contract(config.task, contract)
        self.catalogue_config = config
        physical = replace(config, task=BASE_TASK.get(config.task, config.task))
        if config.task == "square-plies":
            # Unique cache: never restore an aligned load state as the slipped one.
            physical = SquareConfig(**vars(config))
        super().__init__(physical, output_dir=output_dir)
        self.config = config
        if config.task == "square-plies":
            self.start_state = "tray_slipped"
        if self.task_contract == CONTRACT:
            self.evaluator = TaskEvaluator(self)

    def _make_start(self, use_cache=True):
        if self.catalogue_config.task == "square-plies" and self.world.cloth is not None:
            cl = self.world.cloth
            p = cl.plies[self.world.ply_ids[1]]
            snap = cl.snapshot()
            idx = slice(p["v0"], p["v0"] + p["nv"])
            snap["x"][idx] = slip_top(snap["x"][idx])
            cl.restore(snap)
        super()._make_start(use_cache=use_cache)
        if self.catalogue_config.task == "square-plies":
            self.assisted_log.append(dict(initial_condition=SQUARE_SETUP,
                                           note="top ply displaced before relaxation; no physical parameters changed"))

    def get_state(self, evaluate=True):
        state = super().get_state(evaluate=evaluate)
        state["instruction"] = instruction(self.config.task, self.config.robot, self.task_contract)
        state["task_contract"] = self.task_contract
        if "evaluation" in state:
            state["evaluation"]["task_contract"] = self.task_contract
        return state


def make_environment(config, *, contract=None, output_dir=None):
    # All GAR selections use the adapter, including legacy: its evaluator and
    # physical construction remain exactly the originals unless opted in.
    return TaskEnvironment(config, contract=contract, output_dir=output_dir)

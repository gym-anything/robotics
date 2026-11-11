"""The concrete-tasks-v2 evaluator: task_set_progress.Evaluator (the same predicates, progress, start baseline and
0.95 failure factor), reporting this contract. No stepping, scene construction or physics changes."""

from . import task_set_progress
from .task_set_v2 import CONTRACT


class Evaluator(task_set_progress.Evaluator):
    def evaluate(self):
        return dict(super().evaluate(), reward_version=CONTRACT)


def install(env):
    """Judge `env` (a freshly built or reset world) under this contract. The classroom env caches its evaluation per
    (tick, generation); the cached one is the env's own evaluator's, so it is dropped."""
    env.evaluator = Evaluator(env)
    env._eval_key = None

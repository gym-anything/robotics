"""Reusable robots; importing/discovering them never initializes a simulator."""

from .spec import RobotSpec, RobotSetup
from .catalog import resolve, model, list_models, list_setups


def add_to(builder, setup, *, mounts=None):
    from .backends.newton import add_to as add

    return add(builder, resolve(setup), mounts=mounts or {})


__all__ = [
    "RobotSpec",
    "RobotSetup",
    "resolve",
    "model",
    "list_models",
    "list_setups",
    "add_to",
]

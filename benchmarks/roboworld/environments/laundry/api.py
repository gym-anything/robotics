"""Pure-Python backend contract shared by Newton, Isaac, and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .control import ActionSchema
from .observation import LaundryObservation


@dataclass(frozen=True)
class BackendDescriptor:
    id: str
    garment_profile_id: str
    particle_count: int
    tool_names: tuple[str, ...]
    action_schema: ActionSchema
    render_modes: tuple[str, ...] = ("human", "rgb_array")

    def __post_init__(self) -> None:
        if not self.id or not self.garment_profile_id:
            raise ValueError("Backend descriptors need identifiers")
        if self.particle_count <= 0:
            raise ValueError("Backend particle count must be positive")
        if not self.tool_names or len(set(self.tool_names)) != len(self.tool_names):
            raise ValueError("Backend tool names must be nonempty and unique")


@runtime_checkable
class LaundryPhysicsBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> LaundryObservation: ...

    def step(self, action: np.ndarray) -> LaundryObservation: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class ReplayLaundryBackend:
    """Deterministic observation replay for task/API regression tests."""

    def __init__(
        self,
        descriptor: BackendDescriptor,
        observations: Sequence[LaundryObservation],
    ) -> None:
        if not observations:
            raise ValueError("Replay backend needs at least one observation")
        self._descriptor = descriptor
        self.observations = tuple(observations)
        for observation in self.observations:
            observation.validate(
                particle_count=descriptor.particle_count,
                tool_names=descriptor.tool_names,
            )
        self.index = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> LaundryObservation:
        del seed, options
        self.index = 0
        self.actions.clear()
        return self.observations[0]

    def step(self, action: np.ndarray) -> LaundryObservation:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.index = min(self.index + 1, len(self.observations) - 1)
        return self.observations[self.index]

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

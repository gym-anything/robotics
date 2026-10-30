"""Simulator-neutral observation/command boundary for football robots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol


Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotObservation:
    time_s: float
    base_position_world_m: Vec3
    base_orientation_world_wxyz: QuatWXYZ
    base_linear_velocity_world_m_s: Vec3
    base_angular_velocity_world_rad_s: Vec3
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    link_positions_world_m: Mapping[str, Vec3]
    link_linear_velocities_world_m_s: Mapping[str, Vec3] = field(default_factory=dict)
    link_orientations_world_wxyz: Mapping[str, QuatWXYZ] = field(default_factory=dict)
    link_angular_velocities_world_rad_s: Mapping[str, Vec3] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotCommand:
    locomotion_velocity_body_m_s: Vec3 = (0.0, 0.0, 0.0)
    joint_position_overrides_rad: Mapping[str, float] | None = None
    override_blend: float = 1.0


class RobotAdapter(Protocol):
    """Football logic depends on this interface, never on a named morphology."""

    def reset(self) -> None: ...

    def step(self, command: RobotCommand, time_s: float, step_index: int) -> None: ...

    def observe(self, time_s: float) -> RobotObservation: ...

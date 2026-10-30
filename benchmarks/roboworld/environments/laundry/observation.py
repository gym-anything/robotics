"""Structured, backend-independent LaundryFold observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _finite_tuple(value, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class ToolState:
    name: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity_m_s: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool state needs a name")
        object.__setattr__(self, "position_m", _finite_tuple(self.position_m, 3, "position"))
        orientation = _finite_tuple(self.orientation_wxyz, 4, "orientation")
        norm = math.sqrt(sum(item * item for item in orientation))
        if norm <= 1.0e-12:
            raise ValueError("Tool orientation cannot be zero")
        object.__setattr__(
            self,
            "orientation_wxyz",
            tuple(item / norm for item in orientation),
        )
        object.__setattr__(
            self,
            "linear_velocity_m_s",
            _finite_tuple(self.linear_velocity_m_s, 3, "linear velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_tuple(self.angular_velocity_rad_s, 3, "angular velocity"),
        )

    def vector(self) -> np.ndarray:
        return np.asarray(
            (
                *self.position_m,
                *self.orientation_wxyz,
                *self.linear_velocity_m_s,
                *self.angular_velocity_rad_s,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class LaundryObservation:
    time_s: float
    dt_s: float
    particle_positions_m: np.ndarray
    particle_velocities_m_s: np.ndarray
    material_coordinates_m: np.ndarray
    tools: tuple[ToolState, ...]
    pinched_particle_count: int
    jaw_normal_preload_n: float
    roller_normal_force_n: float
    roller_rolling_force_n: float
    maximum_structural_strain: float
    nonlocal_intersection_count: int
    runtime_pose_writes: int
    external_particle_force_applications: int
    fixtures_clear: bool
    released: bool

    def validate(
        self,
        *,
        particle_count: int | None = None,
        tool_names: tuple[str, ...] | None = None,
    ) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("Observation time must be finite and nonnegative")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("Observation dt must be finite and positive")
        positions = np.asarray(self.particle_positions_m)
        velocities = np.asarray(self.particle_velocities_m_s)
        material = np.asarray(self.material_coordinates_m)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("Particle positions must have shape (N, 3)")
        if velocities.shape != positions.shape:
            raise ValueError("Particle velocity shape must match positions")
        if material.shape != (positions.shape[0], 2):
            raise ValueError("Material coordinates must have shape (N, 2)")
        if particle_count is not None and positions.shape[0] != particle_count:
            raise ValueError("Observation particle count differs from backend descriptor")
        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(material))
        ):
            raise ValueError("Particle observation arrays must be finite")
        names = tuple(tool.name for tool in self.tools)
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique")
        if tool_names is not None and names != tool_names:
            raise ValueError(f"Expected tools {tool_names}, got {names}")
        scalar_values = (
            self.jaw_normal_preload_n,
            self.roller_normal_force_n,
            self.roller_rolling_force_n,
            self.maximum_structural_strain,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalar_values):
            raise ValueError("Contact, load, and strain channels must be finite and nonnegative")
        counts = (
            self.pinched_particle_count,
            self.nonlocal_intersection_count,
            self.runtime_pose_writes,
            self.external_particle_force_applications,
        )
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
            raise ValueError("Observation count channels must be nonnegative integers")

    @property
    def maximum_speed_m_s(self) -> float:
        velocities = np.asarray(self.particle_velocities_m_s, dtype=np.float64)
        return float(np.max(np.linalg.norm(velocities, axis=1)))

    def gym_dict(self) -> dict[str, np.ndarray]:
        """Return the stable array wire format used by ``LaundryFoldEnv``."""

        self.validate()
        tool_array = np.stack([tool.vector() for tool in self.tools], axis=0)
        return {
            "time": np.asarray((self.time_s, self.dt_s), dtype=np.float32),
            "particle_positions_m": np.asarray(self.particle_positions_m, dtype=np.float32),
            "particle_velocities_m_s": np.asarray(self.particle_velocities_m_s, dtype=np.float32),
            "material_coordinates_m": np.asarray(self.material_coordinates_m, dtype=np.float32),
            "tools": tool_array,
            "contact": np.asarray(
                (
                    self.pinched_particle_count,
                    self.jaw_normal_preload_n,
                    self.roller_normal_force_n,
                    self.roller_rolling_force_n,
                ),
                dtype=np.float32,
            ),
            "integrity": np.asarray(
                (
                    self.maximum_structural_strain,
                    self.nonlocal_intersection_count,
                    self.runtime_pose_writes,
                    self.external_particle_force_applications,
                    float(self.fixtures_clear),
                    float(self.released),
                ),
                dtype=np.float32,
            ),
        }

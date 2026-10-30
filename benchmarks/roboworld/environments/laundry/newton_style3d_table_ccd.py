"""Local swept-particle collision response for the Style3D support table.

Newton's soft rigid contact supplies the ordinary compliant normal force and
Coulomb friction.  At the P12 release velocity, however, a cloth particle can
cross the thin support surface between two 1/240 s samples before that penalty
contact arrests it.  This module is a narrow *continuous collision fallback*:

* each finite-radius particle is swept independently against the table's top
  plane;
* only particles whose swept paths hit the finite tabletop are corrected;
* a zero-restitution normal impulse removes inward normal velocity; and
* tangential state is left to Newton's existing Coulomb contact model.

This is an operator-split impact response applied to the solver proposal.  It
is not a monolithic rigid/cloth solve, and the endpoint correction and impulse
are exposed in diagnostics rather than hidden as a task controller action.
Unlike a global conservative-advancement fraction, one resting vertex cannot
freeze unrelated garment motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from .newton_style3d_garment import Style3DGarmentSimulation


def table_top_face_domain(points, *, half_x, half_y, top, bottom=None):
    """Centers whose closest box face is the top, not a side or underside.

    Outside the box this is the top face's vertical prism. Inside, compare
    distances to its other five faces. Only this domain has a top-normal
    clearance; native Newton owns side, corner and underside contacts.
    """
    depth = top-points[:,2]
    x, y = half_x-abs(points[:,0]), half_y-abs(points[:,1])
    result = (x >= 0) & (y >= 0) & (depth <= x) & (depth <= y)
    if bottom is not None:
        result &= depth <= points[:,2]-bottom
    return result


@dataclass(frozen=True)
class TablePlaneCCDConfig:
    """Numerical settings for the finite tabletop impact fallback."""

    normal_restitution: float = 0.0
    contact_epsilon_m: float = 1.0e-6
    origin_penetration_tolerance_m: float = 5.0e-6
    verify_after_every_step: bool = True

    def __post_init__(self) -> None:
        values = (
            self.normal_restitution,
            self.contact_epsilon_m,
            self.origin_penetration_tolerance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Table-plane CCD settings must be finite")
        if not 0.0 <= self.normal_restitution <= 1.0:
            raise ValueError("Table-plane restitution must lie in [0, 1]")
        if self.contact_epsilon_m < 0.0:
            raise ValueError("Table-plane contact epsilon cannot be negative")
        if self.origin_penetration_tolerance_m < self.contact_epsilon_m:
            raise ValueError(
                "Table-plane origin tolerance must be at least the contact epsilon"
            )


@dataclass(frozen=True)
class TablePlaneCCDStep:
    """One local impact solve and its auditable scalar diagnostics."""

    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    hit_mask: np.ndarray
    hit_time_fractions: np.ndarray
    minimum_origin_clearance_m: float
    minimum_proposed_clearance_m: float
    minimum_accepted_clearance_m: float
    normal_impulse_kg_m_s: float
    removed_normal_kinetic_energy_j: float
    summed_endpoint_correction_m: float
    maximum_endpoint_correction_m: float
    maximum_inward_speed_m_s: float

    @property
    def hit_count(self) -> int:
        return int(np.count_nonzero(self.hit_mask))

    @property
    def minimum_hit_time_fraction(self) -> float:
        if self.hit_count == 0:
            return 1.0
        return float(np.min(self.hit_time_fractions[self.hit_mask]))


def resolve_swept_particles_against_table_plane(
    origin_positions_m: np.ndarray,
    proposed_positions_m: np.ndarray,
    proposed_velocities_m_s: np.ndarray,
    particle_radius_m: np.ndarray,
    particle_mass_kg: np.ndarray,
    *,
    time_step_s: float,
    table_top_z_m: float,
    table_half_extent_x_m: float,
    table_half_extent_y_m: float,
    config: TablePlaneCCDConfig,
    table_bottom_z_m: float | None = None,
) -> TablePlaneCCDStep:
    """Resolve finite-radius swept particles against a finite horizontal top.

    The path is linear over the proposed integrator step.  A particle is a
    tabletop candidate only when its sphere center reaches the top plane while
    the impact point lies inside the tabletop footprint expanded by that
    particle's radius.  Side and corner impacts remain Newton rigid contacts.
    """

    origin = np.ascontiguousarray(origin_positions_m, dtype=np.float64)
    proposed = np.ascontiguousarray(proposed_positions_m, dtype=np.float64)
    velocity = np.ascontiguousarray(proposed_velocities_m_s, dtype=np.float64)
    radius = np.ascontiguousarray(particle_radius_m, dtype=np.float64).reshape(-1)
    mass = np.ascontiguousarray(particle_mass_kg, dtype=np.float64).reshape(-1)
    if origin.ndim != 2 or origin.shape[1] != 3:
        raise ValueError("Table-plane CCD positions must have shape (N, 3)")
    if proposed.shape != origin.shape or velocity.shape != origin.shape:
        raise ValueError("Table-plane CCD state arrays must have matching shapes")
    if radius.shape != (len(origin),) or mass.shape != (len(origin),):
        raise ValueError("Table-plane CCD radius/mass arrays must have shape (N,)")
    scalars = (
        time_step_s,
        table_top_z_m,
        table_half_extent_x_m,
        table_half_extent_y_m,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("Table-plane geometry and timestep must be finite")
    if time_step_s <= 0.0 or min(table_half_extent_x_m, table_half_extent_y_m) <= 0.0:
        raise ValueError("Table-plane timestep and half extents must be positive")
    if not (
        np.all(np.isfinite(origin))
        and np.all(np.isfinite(proposed))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(radius))
        and np.all(np.isfinite(mass))
    ):
        raise ValueError("Table-plane CCD inputs must be finite")
    if np.any(radius <= 0.0) or np.any(mass <= 0.0):
        raise ValueError("Table-plane particle radii and masses must be positive")

    surface_z = table_top_z_m + radius
    origin_clearance = origin[:, 2] - surface_z
    proposed_clearance = proposed[:, 2] - surface_z
    if table_bottom_z_m is not None and (not math.isfinite(table_bottom_z_m) or table_bottom_z_m >= table_top_z_m):
        raise ValueError("Table bottom must be finite and below its top")
    def footprint(points):
        return table_top_face_domain(points, half_x=table_half_extent_x_m,
            half_y=table_half_extent_y_m, top=table_top_z_m, bottom=table_bottom_z_m)
    inside_origin = footprint(origin)
    gross_origin_penetration = (
        inside_origin
        & (origin_clearance < -config.origin_penetration_tolerance_m)
    )
    if np.any(gross_origin_penetration):
        worst = float(np.min(origin_clearance[gross_origin_penetration]))
        raise RuntimeError(
            "Table-plane CCD cannot start from a penetrating accepted state "
            f"(minimum surface clearance {worst:.9g} m)"
        )

    # Contact fraction for the center path reaching z=top+radius.  A vertex
    # already within the numerical contact epsilon is treated as t=0 contact.
    denominator = origin_clearance - proposed_clearance
    time_fraction = np.zeros(len(origin), dtype=np.float64)
    descending = denominator > 0.0
    solvable = descending & (origin_clearance > 0.0)
    time_fraction[solvable] = np.clip(
        origin_clearance[solvable] / denominator[solvable],
        0.0,
        1.0,
    )
    impact = origin + time_fraction[:, None] * (proposed - origin)
    inside_at_impact = (
        (np.abs(impact[:, 0]) <= table_half_extent_x_m + radius)
        & (np.abs(impact[:, 1]) <= table_half_extent_y_m + radius)
    )
    hit = (
        inside_at_impact
        & (origin_clearance >= -config.origin_penetration_tolerance_m)
        & (proposed_clearance < config.contact_epsilon_m)
        & descending
    )

    accepted = proposed.copy()
    accepted_velocity = velocity.copy()
    impulse = np.zeros(len(origin), dtype=np.float64)
    removed_energy = np.zeros(len(origin), dtype=np.float64)
    correction = np.zeros(len(origin), dtype=np.float64)
    inward_speed = np.zeros(len(origin), dtype=np.float64)
    if np.any(hit):
        incoming_normal_velocity = velocity[hit, 2]
        post_normal_velocity = np.where(
            incoming_normal_velocity < 0.0,
            -config.normal_restitution * incoming_normal_velocity,
            incoming_normal_velocity,
        )
        remaining_time_s = (1.0 - time_fraction[hit]) * time_step_s
        corrected_z = (
            surface_z[hit]
            + config.contact_epsilon_m
            + np.maximum(post_normal_velocity, 0.0) * remaining_time_s
        )
        correction[hit] = np.maximum(corrected_z - proposed[hit, 2], 0.0)
        accepted[hit, 2] = np.maximum(proposed[hit, 2], corrected_z)
        accepted_velocity[hit, 2] = post_normal_velocity
        impulse[hit] = mass[hit] * np.maximum(
            post_normal_velocity - incoming_normal_velocity,
            0.0,
        )
        removed_energy[hit] = np.maximum(
            0.5
            * mass[hit]
            * (
                incoming_normal_velocity * incoming_normal_velocity
                - post_normal_velocity * post_normal_velocity
            ),
            0.0,
        )
        inward_speed[hit] = np.maximum(-incoming_normal_velocity, 0.0)

    accepted_clearance = accepted[:, 2] - surface_z
    # This is a FINITE supporting plane. Cloth hanging outside its footprint
    # may be below its elevation without penetrating any table geometry. The
    # radius-expanded hit broad phase must NOT define the verification domain:
    # a sphere alongside the vertical side has a horizontal contact normal,
    # not a top-plane clearance. Side/corner response belongs to Newton.
    return TablePlaneCCDStep(
        positions_m=accepted,
        velocities_m_s=accepted_velocity,
        hit_mask=hit,
        hit_time_fractions=time_fraction,
        minimum_origin_clearance_m=float(np.min(origin_clearance[inside_origin], initial=math.inf)),
        minimum_proposed_clearance_m=float(np.min(proposed_clearance[footprint(proposed)], initial=math.inf)),
        minimum_accepted_clearance_m=float(np.min(accepted_clearance[footprint(accepted)], initial=math.inf)),
        normal_impulse_kg_m_s=float(np.sum(impulse)),
        removed_normal_kinetic_energy_j=float(np.sum(removed_energy)),
        summed_endpoint_correction_m=float(np.sum(correction)),
        maximum_endpoint_correction_m=float(np.max(correction, initial=0.0)),
        maximum_inward_speed_m_s=float(np.max(inward_speed, initial=0.0)),
    )


class TablePlaneCCDController:
    """Stateful diagnostics and Newton-array coupling for local table impacts."""

    def __init__(
        self,
        simulation: "Style3DGarmentSimulation",
        config: TablePlaneCCDConfig,
    ) -> None:
        self.config = config
        self.step_count = 0
        self.collision_step_count = 0
        self.collision_event_count = 0
        self.maximum_events_in_one_step = 0
        self.minimum_hit_time_fraction = 1.0
        self.cumulative_normal_impulse_kg_m_s = 0.0
        self.cumulative_removed_normal_kinetic_energy_j = 0.0
        self.cumulative_endpoint_correction_m = 0.0
        self.maximum_endpoint_correction_m = 0.0
        self.maximum_inward_speed_m_s = 0.0
        self.minimum_accepted_clearance_m = math.inf
        self.last_step: TablePlaneCCDStep | None = None
        self.validate_positions(simulation.positions_m(), simulation, context="initial state")

    def validate_positions(
        self,
        positions_m: np.ndarray,
        simulation: "Style3DGarmentSimulation",
        *,
        context: str,
    ) -> None:
        positions = np.asarray(positions_m, dtype=np.float64)
        radius = simulation.inputs.particle_radius_m
        inside = table_top_face_domain(positions,
            half_x=simulation.config.table_half_extent_x_m,
            half_y=simulation.config.table_half_extent_y_m,
            top=simulation.config.table_top_z_m,
            bottom=simulation.config.table_top_z_m-2*simulation.config.table_thickness_m)
        clearance = positions[:, 2] - radius - simulation.config.table_top_z_m
        if np.any(inside & (clearance < -self.config.origin_penetration_tolerance_m)):
            raise RuntimeError(
                f"The {context} penetrates the table beyond the CCD tolerance"
            )

    def begin_step(self, simulation: "Style3DGarmentSimulation") -> np.ndarray:
        """Snapshot the accepted origin before Style3D mutates its input state."""

        return np.ascontiguousarray(
            simulation.read_contact_arrays(position=simulation.state_0.particle_q)["position"],
            dtype=np.float64,
        )

    def finish_step(
        self,
        simulation: "Style3DGarmentSimulation",
        origin_positions_m: np.ndarray,
    ) -> None:
        arrays=simulation.read_contact_arrays(position=simulation.state_1.particle_q,
                                             velocity=simulation.state_1.particle_qd)
        step = resolve_swept_particles_against_table_plane(
            origin_positions_m,
            arrays["position"],
            arrays["velocity"],
            simulation.inputs.particle_radius_m,
            simulation.inputs.particle_mass_kg,
            time_step_s=simulation.config.time_step_s,
            table_top_z_m=simulation.config.table_top_z_m,
            table_half_extent_x_m=simulation.config.table_half_extent_x_m,
            table_half_extent_y_m=simulation.config.table_half_extent_y_m,
            config=self.config,
            table_bottom_z_m=simulation.config.table_top_z_m-2*simulation.config.table_thickness_m,
        )
        if step.hit_count:
            simulation.state_1.particle_q.assign(step.positions_m.astype(np.float32))
            simulation.state_1.particle_qd.assign(step.velocities_m_s.astype(np.float32))
            self.collision_step_count += 1
            self.collision_event_count += step.hit_count
            self.maximum_events_in_one_step = max(
                self.maximum_events_in_one_step,
                step.hit_count,
            )
            self.minimum_hit_time_fraction = min(
                self.minimum_hit_time_fraction,
                step.minimum_hit_time_fraction,
            )
        self.step_count += 1
        self.cumulative_normal_impulse_kg_m_s += step.normal_impulse_kg_m_s
        self.cumulative_removed_normal_kinetic_energy_j += (
            step.removed_normal_kinetic_energy_j
        )
        self.cumulative_endpoint_correction_m += step.summed_endpoint_correction_m
        self.maximum_endpoint_correction_m = max(
            self.maximum_endpoint_correction_m,
            step.maximum_endpoint_correction_m,
        )
        self.maximum_inward_speed_m_s = max(
            self.maximum_inward_speed_m_s,
            step.maximum_inward_speed_m_s,
        )
        self.minimum_accepted_clearance_m = min(
            self.minimum_accepted_clearance_m,
            step.minimum_accepted_clearance_m,
        )
        self.last_step = step
        if (
            self.config.verify_after_every_step
            and step.minimum_accepted_clearance_m
            < -self.config.origin_penetration_tolerance_m
        ):
            raise RuntimeError("Local table-plane CCD accepted a penetrating state")

    def summary(self) -> Mapping[str, object]:
        last = self.last_step
        return {
            "enabled": True,
            "method": "local swept-sphere/top-plane TOI plus normal impulse",
            "verification_domain": "Finite box top-face region; side and underside contact are native Newton",
            "operator_split": True,
            "global_step_scaling": False,
            "normal_restitution": self.config.normal_restitution,
            "contact_epsilon_m": self.config.contact_epsilon_m,
            "origin_penetration_tolerance_m": (
                self.config.origin_penetration_tolerance_m
            ),
            "tangential_response_provider": "Newton Coulomb soft contact",
            "step_count": self.step_count,
            "collision_step_count": self.collision_step_count,
            "collision_event_count": self.collision_event_count,
            "maximum_events_in_one_step": self.maximum_events_in_one_step,
            "minimum_hit_time_fraction": self.minimum_hit_time_fraction,
            "cumulative_normal_impulse_kg_m_s": (
                self.cumulative_normal_impulse_kg_m_s
            ),
            "cumulative_removed_normal_kinetic_energy_j": (
                self.cumulative_removed_normal_kinetic_energy_j
            ),
            "cumulative_endpoint_correction_m": (
                self.cumulative_endpoint_correction_m
            ),
            "maximum_endpoint_correction_m": self.maximum_endpoint_correction_m,
            "maximum_inward_speed_m_s": self.maximum_inward_speed_m_s,
            "minimum_accepted_clearance_m": (
                None
                if not math.isfinite(self.minimum_accepted_clearance_m)
                else self.minimum_accepted_clearance_m
            ),
            "last_hit_count": 0 if last is None else last.hit_count,
            "last_minimum_hit_time_fraction": (
                1.0 if last is None else last.minimum_hit_time_fraction
            ),
            "last_normal_impulse_kg_m_s": (
                0.0 if last is None else last.normal_impulse_kg_m_s
            ),
            "last_removed_normal_kinetic_energy_j": (
                0.0 if last is None else last.removed_normal_kinetic_energy_j
            ),
            "last_minimum_proposed_clearance_m": (
                None if last is None or not math.isfinite(last.minimum_proposed_clearance_m) else last.minimum_proposed_clearance_m
            ),
            "last_minimum_accepted_clearance_m": (
                None if last is None or not math.isfinite(last.minimum_accepted_clearance_m) else last.minimum_accepted_clearance_m
            ),
        }


__all__ = [
    "TablePlaneCCDConfig",
    "TablePlaneCCDController",
    "TablePlaneCCDStep",
    "resolve_swept_particles_against_table_plane",
]

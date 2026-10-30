"""IPC barrier and continuous-collision guard for Newton Style3D cloth.

Newton Style3D supplies the garment's anisotropic membrane/bending dynamics,
proximity thickness, rigid contact, and tool coupling.  Its penalty self-
contact alone does not mathematically prevent a fast step from crossing the
cloth surface.  This module adds two complementary IPC Toolkit mechanisms:

* the gradient of a smooth zero-distance barrier is applied as an internal
  particle force before each Style3D solve; and
* exact continuous collision detection clips the proposed trajectory before
  state acceptance.

This is an explicitly reported operator split, not a claim that Newton's
linear system has become the monolithic incremental-potential IPC optimizer.
Style3D proximity contact retains the finite-thickness prior; IPC protects the
triangle mid-surface topology at zero distance because one global positive
``dmin`` is infeasible at welded garment seams.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .self_intersection import (
    IPCTrianglePairConfirmer,
    build_vertex_neighbourhoods,
)


if TYPE_CHECKING:  # pragma: no cover
    from .newton_style3d_garment import Style3DGarmentSimulation


@dataclass(frozen=True)
class IPCStyle3DSelfContactConfig:
    """Numerical topology protection, separate from textile coefficients."""

    activation_distance_m: float = 4.0e-3
    minimum_distance_m: float = 0.0
    barrier_stiffness_prior: float = 1.0e4
    topological_filter_graph_hops: int = 1
    ccd_safety_factor: float = 0.99
    enable_barrier_force: bool = True
    enable_ccd: bool = True
    verify_after_every_step: bool = True
    local_contact_projection: bool = False

    def __post_init__(self) -> None:
        values = (
            self.activation_distance_m,
            self.minimum_distance_m,
            self.barrier_stiffness_prior,
            self.ccd_safety_factor,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("IPC Style3D settings must be finite")
        if self.activation_distance_m <= 0.0:
            raise ValueError("IPC barrier activation distance must be positive")
        if not 0.0 <= self.minimum_distance_m < self.activation_distance_m:
            raise ValueError(
                "IPC minimum distance must be nonnegative and below activation"
            )
        if self.barrier_stiffness_prior <= 0.0:
            raise ValueError("IPC barrier stiffness prior must be positive")
        if self.topological_filter_graph_hops < 0:
            raise ValueError("IPC topology filter graph hops cannot be negative")
        if not 0.0 < self.ccd_safety_factor < 1.0:
            raise ValueError("IPC CCD safety factor must lie strictly inside (0, 1)")
        if not (self.enable_barrier_force or self.enable_ccd):
            raise ValueError("IPC coupling must enable its barrier, CCD, or both")


class IPCStyle3DSelfContactController:
    """Stateful operator-split IPC coupling owned by one simulation."""

    def __init__(
        self,
        simulation: "Style3DGarmentSimulation",
        config: IPCStyle3DSelfContactConfig,
    ) -> None:
        try:
            import ipctk  # noqa: PLC0415
        except ImportError as exception:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "IPC-protected Style3D requires IPC Toolkit; run "
                "laundry/bootstrap_newton.ps1 first."
            ) from exception
        self.ipctk = ipctk
        self.config = config
        self.triangles = np.ascontiguousarray(
            simulation.inputs.physical_triangles,
            dtype=np.int32,
        )
        initial = np.ascontiguousarray(
            simulation.positions_m(),
            dtype=np.float64,
        )
        edges = np.ascontiguousarray(ipctk.edges(self.triangles), dtype=np.int32)
        self.collision_mesh = ipctk.CollisionMesh(
            initial,
            edges,
            self.triangles,
        )
        neighborhoods = build_vertex_neighbourhoods(
            self.triangles,
            vertex_count=len(initial),
            graph_hops=config.topological_filter_graph_hops,
        )
        excluded_pairs = {
            (vertex, neighbor): False
            for vertex, neighborhood in enumerate(neighborhoods)
            for neighbor in neighborhood
            if vertex < neighbor
        }
        self.collision_mesh.can_collide = ipctk.make_sparse_filter(
            excluded_pairs,
            True,
        )
        self.excluded_vertex_pair_count = len(excluded_pairs)
        self.barrier = ipctk.BarrierPotential(
            config.activation_distance_m,
            config.barrier_stiffness_prior,
        )
        self.collisions = ipctk.NormalCollisions()
        self.integrity_auditor = IPCTrianglePairConfirmer(self.triangles)
        self.step_count = 0
        self.ccd_clipped_step_count = 0
        self.minimum_ccd_step_fraction = 1.0
        self.last_barrier_collision_count = 0
        self.maximum_barrier_collision_count = 0
        self.last_barrier_potential = 0.0
        self.last_barrier_force_norm_n = 0.0
        self.maximum_barrier_force_norm_n = 0.0
        self.last_minimum_active_distance_m: float | None = None
        self.last_ccd_step_fraction = 1.0
        self.last_accepted_step_fraction = 1.0
        self.projection_step_count = 0
        self.last_projection = {}
        self.inverse_mass = np.asarray(simulation.model.particle_inv_mass.numpy(), dtype=float)
        self.last_projection_impulse = np.zeros_like(initial)
        self.validate_positions(initial, context="initial state")

    @staticmethod
    def _collision_count(collisions: Any) -> int:
        return int(
            len(collisions.vv_collisions)
            + len(collisions.ev_collisions)
            + len(collisions.ee_collisions)
            + len(collisions.fv_collisions)
        )

    def validate_positions(self, positions_m: np.ndarray, *, context: str) -> None:
        positions = np.ascontiguousarray(positions_m, dtype=np.float64)
        # IPC's Boolean predicate is closed-set: an unstitched guard touching
        # a welded seam at exactly zero distance is reported alongside a true
        # triangle crossing.  Sewn garment midsurfaces legitimately contain
        # those zero-distance seam contacts.  The complete face-pair audit
        # retains IPC's mesh filter and exact directed edge--triangle tests,
        # rejecting every non-adjacent crossing while permitting contact that
        # does not pass through another face.
        crossing_pairs = self.integrity_auditor.surface_intersection_pairs(
            positions
        )
        if len(crossing_pairs):
            raise RuntimeError(
                f"IPC topology integrity failed for the {context}: the sewn "
                f"surface contains {len(crossing_pairs)} confirmed triangle "
                "crossing pair(s)"
            )

    def begin_step(
        self,
        simulation: "Style3DGarmentSimulation",
    ) -> np.ndarray:
        """Apply the barrier force and return the CCD trajectory origin."""

        positions = np.ascontiguousarray(
            simulation.read_contact_arrays(position=simulation.state_0.particle_q)["position"],
            dtype=np.float64,
        )
        if not self.config.enable_barrier_force:
            return positions
        self.collisions.clear()
        self.collisions.build(
            self.collision_mesh,
            positions,
            self.config.activation_distance_m,
            self.config.minimum_distance_m,
        )
        collision_count = self._collision_count(self.collisions)
        self.last_barrier_collision_count = collision_count
        self.maximum_barrier_collision_count = max(
            self.maximum_barrier_collision_count,
            collision_count,
        )
        if collision_count == 0:
            self.last_barrier_potential = 0.0
            self.last_barrier_force_norm_n = 0.0
            self.last_minimum_active_distance_m = None
            return positions

        self.last_barrier_potential = float(
            self.barrier(
                self.collisions,
                self.collision_mesh,
                positions,
            )
        )
        gradient = np.asarray(
            self.barrier.gradient(
                self.collisions,
                self.collision_mesh,
                positions,
            ),
            dtype=np.float64,
        ).reshape(positions.shape)
        barrier_force = -gradient
        force_norm = float(np.linalg.norm(barrier_force))
        self.last_barrier_force_norm_n = force_norm
        self.maximum_barrier_force_norm_n = max(
            self.maximum_barrier_force_norm_n,
            force_norm,
        )
        self.last_minimum_active_distance_m = math.sqrt(max(0., float(
            self.collisions.compute_minimum_distance(
                self.collision_mesh,
                positions,
            )
        )))
        simulation.state_0.particle_f.assign(
            barrier_force.astype(np.float32)
        )
        return positions

    def finish_step(
        self,
        simulation: "Style3DGarmentSimulation",
        origin_positions_m: np.ndarray,
    ) -> None:
        """CCD-filter the proposed state and verify the accepted surface."""

        origin = np.ascontiguousarray(origin_positions_m, dtype=np.float64)
        proposed = np.ascontiguousarray(
            simulation.read_contact_arrays(position=simulation.state_1.particle_q)["position"],
            dtype=np.float64,
        )
        ccd_fraction = 1.0
        accepted_fraction = 1.0
        projection_likely = (self.config.local_contact_projection
                             and self.last_projection.get("corrected_vertices", 0) > 0)
        previous_witness = self.last_projection.get("ccd_witness")
        deferred_fraction = False
        self.last_projection_impulse.fill(0.)
        self.last_projection = {}
        if self.config.enable_ccd:
            if projection_likely:
                # The original fraction is discarded when local projection
                # revises the proposal. IPC's same-candidate CCD Boolean can
                # stop at its first collision; keep the full fractional CCD
                # for final acceptance. Only use this path after a correction:
                # the parallel fractional query is faster on contact-free cloth.
                from .newton_style3d_contact_projection import ccd_witness_collides
                witnessed = ccd_witness_collides(self.collision_mesh, origin, proposed,
                                                previous_witness, self.config.minimum_distance_m)
                safe = False if witnessed else self.ipctk.is_step_collision_free(
                    self.collision_mesh, origin, proposed, self.config.minimum_distance_m)
                deferred_fraction = not safe
                ccd_fraction = 1.0 if safe else 0.0
            else:
                ccd_fraction = float(self.ipctk.compute_collision_free_stepsize(
                    self.collision_mesh, origin, proposed, self.config.minimum_distance_m))
            if (
                not math.isfinite(ccd_fraction)
                or not 0.0 <= ccd_fraction <= 1.0
            ):
                raise RuntimeError("IPC CCD returned an invalid step fraction")
        if self.config.local_contact_projection and ccd_fraction < 1.0 - 1e-12:
            from .newton_style3d_contact_projection import project
            table_controller=simulation._table_plane_ccd_controller
            table=None if table_controller is None else dict(
                half_x=simulation.config.table_half_extent_x_m,
                half_y=simulation.config.table_half_extent_y_m,
                top=simulation.config.table_top_z_m,
                bottom=simulation.config.table_top_z_m-2*simulation.config.table_thickness_m,
                radius=simulation.inputs.particle_radius_m,
                epsilon=table_controller.config.contact_epsilon_m)
            revised, self.last_projection = project(self.collision_mesh, self.collisions,
                origin, proposed, self.inverse_mass, table=table)
            if self.last_projection["corrected_vertices"]:
                delta = revised-proposed
                mass = np.divide(1., self.inverse_mass, out=np.zeros_like(self.inverse_mass), where=self.inverse_mass>0)
                self.last_projection_impulse = mass[:,None]*delta/simulation.config.time_step_s
                simulation.state_1.particle_q.assign(revised.astype(np.float32))
                velocity = simulation.read_contact_arrays(velocity=simulation.state_1.particle_qd)["velocity"]+delta/simulation.config.time_step_s
                simulation.state_1.particle_qd.assign(velocity.astype(np.float32))
                # CCD sees precisely the float32 values just assigned. Avoid
                # transferring the same array back from CUDA to reconstruct it.
                proposed = revised.astype(np.float32).astype(float)
                ccd_fraction = float(self.ipctk.compute_collision_free_stepsize(
                    self.collision_mesh, origin, proposed, self.config.minimum_distance_m))
                self.projection_step_count += 1
            elif deferred_fraction:
                # No corrected state exists: never use the Boolean placeholder
                # as the accepted step size of the original proposal.
                ccd_fraction = float(self.ipctk.compute_collision_free_stepsize(
                    self.collision_mesh, origin, proposed, self.config.minimum_distance_m))
        if not math.isfinite(ccd_fraction) or not 0.0 <= ccd_fraction <= 1.0:
            raise RuntimeError("IPC CCD returned an invalid final step fraction")
        self.minimum_ccd_step_fraction = min(
            self.minimum_ccd_step_fraction,
            ccd_fraction,
        )
        if ccd_fraction < 1.0 - 1.0e-12:
            self.ccd_clipped_step_count += 1
            accepted_fraction = self.config.ccd_safety_factor * ccd_fraction
            accepted = origin + accepted_fraction * (proposed - origin)
            velocity = (
                accepted - origin
            ) / simulation.config.time_step_s
            simulation.state_1.particle_q.assign(
                accepted.astype(np.float32)
            )
            simulation.state_1.particle_qd.assign(
                velocity.astype(np.float32)
            )
        self.last_ccd_step_fraction = ccd_fraction
        self.last_accepted_step_fraction = accepted_fraction
        self.step_count += 1
        if self.config.verify_after_every_step:
            accepted_positions = np.ascontiguousarray(
                simulation.read_contact_arrays(position=simulation.state_1.particle_q)["position"],
                dtype=np.float64,
            )
            self.validate_positions(
                accepted_positions,
                context=f"accepted step {self.step_count}",
            )
            if self.config.local_contact_projection and simulation._table_plane_ccd_controller is not None:
                simulation._table_plane_ccd_controller.validate_positions(
                    accepted_positions,simulation,context="post-self-contact accepted state")

    def summary(self) -> Mapping[str, object]:
        return {
            "enabled": True,
            "backend": "IPC Toolkit",
            "ipctk_version": self.ipctk.__version__,
            "operator_split": True,
            "monolithic_incremental_potential_solve": False,
            "activation_distance_m": self.config.activation_distance_m,
            "minimum_distance_m": self.config.minimum_distance_m,
            "finite_thickness_provider": "Newton Style3D proximity contact",
            "barrier_stiffness_prior": self.config.barrier_stiffness_prior,
            "topological_filter_graph_hops": (
                self.config.topological_filter_graph_hops
            ),
            "excluded_vertex_pair_count": self.excluded_vertex_pair_count,
            "barrier_force_enabled": self.config.enable_barrier_force,
            "ccd_enabled": self.config.enable_ccd,
            "ccd_safety_factor": self.config.ccd_safety_factor,
            "verify_after_every_step": self.config.verify_after_every_step,
            "step_count": self.step_count,
            "ccd_clipped_step_count": self.ccd_clipped_step_count,
            "minimum_ccd_step_fraction": self.minimum_ccd_step_fraction,
            "last_ccd_step_fraction": self.last_ccd_step_fraction,
            "last_accepted_step_fraction": self.last_accepted_step_fraction,
            "local_contact_projection": self.config.local_contact_projection,
            "projection_step_count": self.projection_step_count,
            "last_projection": self.last_projection,
            "last_barrier_collision_count": self.last_barrier_collision_count,
            "maximum_barrier_collision_count": self.maximum_barrier_collision_count,
            "last_barrier_potential": self.last_barrier_potential,
            "last_barrier_force_norm_n": self.last_barrier_force_norm_n,
            "maximum_barrier_force_norm_n": self.maximum_barrier_force_norm_n,
            "last_minimum_active_distance_m": (
                self.last_minimum_active_distance_m
            ),
        }


__all__ = [
    "IPCStyle3DSelfContactConfig",
    "IPCStyle3DSelfContactController",
]

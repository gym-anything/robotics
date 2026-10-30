"""Audited residual-contact closures for the Style3D garment adapter."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def project_residual_table_normal_contact(
    particle_mass: wp.array[float],
    particle_radius: wp.array[float],
    table_top_z_m: float,
    table_half_extent_x_m: float,
    table_half_extent_y_m: float,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    diagnostics: wp.array[float],
):
    """Close missed zero-restitution table penetration without adding energy.

    Diagnostics are cumulative: normal impulse, summed correction, removed
    kinetic energy, corrected-particle events, and maximum single correction.
    """

    particle = wp.tid()
    q = particle_q[particle]
    radius = particle_radius[particle]
    inside_x = wp.abs(q[0]) <= table_half_extent_x_m + radius
    inside_y = wp.abs(q[1]) <= table_half_extent_y_m + radius
    surface = table_top_z_m + radius
    if not inside_x or not inside_y or q[2] >= surface:
        return
    velocity = particle_qd[particle]
    corrected_vz = wp.max(velocity[2], 0.0)
    delta_vz = corrected_vz - velocity[2]
    correction = surface - q[2]
    particle_q[particle] = wp.vec3(q[0], q[1], surface)
    particle_qd[particle] = wp.vec3(velocity[0], velocity[1], corrected_vz)
    mass = particle_mass[particle]
    wp.atomic_add(diagnostics, 0, mass * delta_vz)
    wp.atomic_add(diagnostics, 1, correction)
    wp.atomic_add(
        diagnostics,
        2,
        0.5 * mass * (velocity[2] * velocity[2] - corrected_vz * corrected_vz),
    )
    wp.atomic_add(diagnostics, 3, 1.0)
    wp.atomic_max(diagnostics, 4, correction)


__all__ = ["project_residual_table_normal_contact"]

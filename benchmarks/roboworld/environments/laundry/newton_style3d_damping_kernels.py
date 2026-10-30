"""Objective dissipation terms missing from Newton 1.4's Style3D step."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def accumulate_anisotropic_membrane_damping(
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    face_area: wp.array[float],
    inverse_material_matrix: wp.array[wp.mat22],
    faces: wp.array2d[wp.int32],
    damping_n_s_m: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
):
    """Accumulate Kelvin--Voigt warp/weft/shear forces per triangle."""

    face_id = wp.tid()
    face = wp.vec3i(
        faces[face_id, 0],
        faces[face_id, 1],
        faces[face_id, 2],
    )
    inverse = inverse_material_matrix[face_id]
    x01 = position[face[1]] - position[face[0]]
    x02 = position[face[2]] - position[face[0]]
    v01 = velocity[face[1]] - velocity[face[0]]
    v02 = velocity[face[2]] - velocity[face[0]]
    fu = x01 * inverse[0, 0] + x02 * inverse[1, 0]
    fv = x01 * inverse[0, 1] + x02 * inverse[1, 1]
    fu_rate = v01 * inverse[0, 0] + v02 * inverse[1, 0]
    fv_rate = v01 * inverse[0, 1] + v02 * inverse[1, 1]
    length_u = wp.length(fu)
    length_v = wp.length(fv)
    if length_u <= 1.0e-8 or length_v <= 1.0e-8:
        return
    unit_u = fu / length_u
    unit_v = fv / length_v
    derivative_u = wp.vec3(
        -inverse[0, 0] - inverse[1, 0],
        inverse[0, 0],
        inverse[1, 0],
    )
    derivative_v = wp.vec3(
        -inverse[0, 1] - inverse[1, 1],
        inverse[0, 1],
        inverse[1, 1],
    )
    rate_u = wp.dot(unit_u, fu_rate)
    rate_v = wp.dot(unit_v, fv_rate)
    cosine = wp.dot(unit_u, unit_v)
    shear_gradient_u = (unit_v - cosine * unit_u) / length_u
    shear_gradient_v = (unit_u - cosine * unit_v) / length_v
    shear_rate = (
        wp.dot(shear_gradient_u, fu_rate)
        + wp.dot(shear_gradient_v, fv_rate)
    )
    damping = damping_n_s_m[face_id]
    area = face_area[face_id]
    for local in range(3):
        gradient_shear = (
            derivative_u[local] * shear_gradient_u
            + derivative_v[local] * shear_gradient_v
        )
        local_force = -area * (
            damping[0] * rate_u * derivative_u[local] * unit_u
            + damping[1] * rate_v * derivative_v[local] * unit_v
            + damping[2] * shear_rate * gradient_shear
        )
        wp.atomic_add(force, face[local], local_force)


@wp.kernel(enable_backward=False)
def accumulate_dihedral_rate_damping(
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    damping_nm_s: wp.array[float],
    force: wp.array[wp.vec3],
):
    """Damp only the objective dihedral-angle rate of an interior hinge.

    The gradient and power scaling follow Newton's own hinge formulation, but
    omit its elastic angle term because Style3D supplies a separate
    Crouzeix--Raviart bending energy.  For nonnegative ``damping_nm_s`` the
    instantaneous power is ``-edge_length * damping * angle_rate**2``.
    """

    edge_id = wp.tid()
    i = edges[edge_id, 0]
    j = edges[edge_id, 1]
    k = edges[edge_id, 2]
    l = edges[edge_id, 3]
    if i < 0 or j < 0 or k < 0 or l < 0:
        return

    x1 = position[i]
    x2 = position[j]
    x3 = position[k]
    x4 = position[l]
    n1 = wp.cross(x3 - x1, x4 - x1)
    n2 = wp.cross(x4 - x2, x3 - x2)
    edge = x4 - x3
    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    edge_length = wp.length(edge)
    if n1_length <= 1.0e-8 or n2_length <= 1.0e-8 or edge_length <= 1.0e-8:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    edge_hat = edge / edge_length
    d1 = -n1_hat * edge_length
    d2 = -n2_hat * edge_length
    d3 = (
        -n1_hat * wp.dot(x1 - x4, edge_hat)
        - n2_hat * wp.dot(x2 - x4, edge_hat)
    )
    d4 = (
        -n1_hat * wp.dot(x3 - x1, edge_hat)
        - n2_hat * wp.dot(x3 - x2, edge_hat)
    )
    angle_rate = (
        wp.dot(d1, velocity[i])
        + wp.dot(d2, velocity[j])
        + wp.dot(d3, velocity[k])
        + wp.dot(d4, velocity[l])
    )
    force_scale = -edge_length * damping_nm_s[edge_id] * angle_rate
    wp.atomic_add(force, i, d1 * force_scale)
    wp.atomic_add(force, j, d2 * force_scale)
    wp.atomic_add(force, k, d3 * force_scale)
    wp.atomic_add(force, l, d4 * force_scale)


@wp.kernel(enable_backward=False)
def accumulate_quadratic_normal_air_drag(
    position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    air_density_kg_m3: float,
    drag_coefficient: float,
    wind_velocity_m_s: wp.vec3,
    force: wp.array[wp.vec3],
):
    """Two-sided quadratic pressure drag on the live triangle area."""

    face_id = wp.tid()
    i = faces[face_id, 0]
    j = faces[face_id, 1]
    k = faces[face_id, 2]
    cross = wp.cross(position[j] - position[i], position[k] - position[i])
    double_area = wp.length(cross)
    if double_area <= 1.0e-12:
        return
    normal = cross / double_area
    area = 0.5 * double_area
    relative_velocity = (
        (velocity[i] + velocity[j] + velocity[k]) / 3.0 - wind_velocity_m_s
    )
    normal_speed = wp.dot(relative_velocity, normal)
    triangle_force = (
        -0.5
        * air_density_kg_m3
        * drag_coefficient
        * area
        * wp.abs(normal_speed)
        * normal_speed
        * normal
    )
    vertex_force = triangle_force / 3.0
    wp.atomic_add(force, i, vertex_force)
    wp.atomic_add(force, j, vertex_force)
    wp.atomic_add(force, k, vertex_force)


__all__ = [
    "accumulate_anisotropic_membrane_damping",
    "accumulate_dihedral_rate_damping",
    "accumulate_quadratic_normal_air_drag",
]

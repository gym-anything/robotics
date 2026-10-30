"""Configuration-dependent operational-space impact response utilities.

An articulated link does not have one impact mass.  Its instantaneous point
response is the 3x3 tensor ``W = J M^-1 J.T`` (or its support-constrained
projection).  A scalar effective mass is only the directional projection
``1 / (n.T W n)``.  Keeping the tensor is essential for oblique football
impacts because the tangential response controls both slip and spin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionalEffectiveMass:
    free_kg: float
    constrained_kg: float | None
    inverse_free_per_kg: float
    inverse_constrained_per_kg: float | None


def point_jacobian(
    body_jacobian_world: np.ndarray,
    point_minus_body_origin_world_m,
) -> np.ndarray:
    """Shift a 6D body-origin Jacobian to an arbitrary world-space point."""

    jacobian = np.asarray(body_jacobian_world, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError("body Jacobian must have shape (6, generalized_dof)")
    rx, ry, rz = np.asarray(
        point_minus_body_origin_world_m, dtype=np.float64
    )
    skew = np.asarray(
        [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]],
        dtype=np.float64,
    )
    return jacobian[:3] - skew @ jacobian[3:]


def point_inverse_mass_response(
    mass_matrix,
    point_linear_jacobian,
    *,
    constraint_jacobian=None,
    pseudoinverse_rcond: float = 1.0e-10,
) -> np.ndarray:
    """Return the world-frame point inverse-mass tensor in ``1/kg``.

    With ideal bilateral velocity constraints ``C qdot = 0``, the generalized
    inverse mass is projected as

    ``M^-1 - M^-1 C.T (C M^-1 C.T)^+ C M^-1``.

    The returned matrix maps a point force impulse to point-velocity change.
    It is explicitly symmetrized to remove floating-point solve asymmetry; no
    eigenvalue clipping or other hidden regularization is performed.
    """

    matrix = np.asarray(mass_matrix, dtype=np.float64)
    jacobian = np.asarray(point_linear_jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mass matrix must be square")
    if jacobian.shape != (3, matrix.shape[0]):
        raise ValueError("point Jacobian shape does not match mass matrix")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("mass matrix and point Jacobian must be finite")

    response = np.linalg.solve(matrix, jacobian.T)
    if constraint_jacobian is not None:
        constraints = np.asarray(constraint_jacobian, dtype=np.float64)
        if constraints.ndim != 2 or constraints.shape[1] != matrix.shape[0]:
            raise ValueError("constraint Jacobian shape does not match mass matrix")
        if not np.all(np.isfinite(constraints)):
            raise ValueError("constraint Jacobian must be finite")
        inverse_mass_constraints = np.linalg.solve(matrix, constraints.T)
        constraint_mass = constraints @ inverse_mass_constraints
        response -= inverse_mass_constraints @ (
            np.linalg.pinv(constraint_mass, rcond=pseudoinverse_rcond)
            @ (constraints @ response)
        )

    point_response = jacobian @ response
    point_response = 0.5 * (point_response + point_response.T)
    if not np.all(np.isfinite(point_response)):
        raise ValueError("point inverse-mass response is non-finite")
    minimum_eigenvalue = float(np.linalg.eigvalsh(point_response)[0])
    if minimum_eigenvalue < -1.0e-9:
        raise ValueError(
            "mass/constraint system produced a non-positive-semidefinite "
            f"point response (minimum eigenvalue {minimum_eigenvalue})"
        )
    return point_response


def rigid_body_point_inverse_mass_response(
    mass_kg: float,
    inertia_world_kg_m2,
    point_minus_center_of_mass_world_m,
) -> np.ndarray:
    """Return a free rigid body's point inverse-mass response in ``1/kg``.

    For point arm ``r``, an impulse ``p`` changes contact-point velocity by
    ``[I/m - [r]x I_body^-1 [r]x] p``.  This includes the rotational response
    that a scalar translational inverse mass omits.
    """

    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("rigid-body mass must be positive and finite")
    inertia = np.asarray(inertia_world_kg_m2, dtype=np.float64)
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError("world inertia must be a finite 3x3 matrix")
    arm = np.asarray(point_minus_center_of_mass_world_m, dtype=np.float64)
    if arm.shape != (3,) or not np.all(np.isfinite(arm)):
        raise ValueError("point arm must be a finite 3-vector")
    rx, ry, rz = arm
    skew = np.asarray(
        [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]],
        dtype=np.float64,
    )
    response = np.eye(3, dtype=np.float64) / mass_kg
    response -= skew @ np.linalg.solve(inertia, skew)
    return 0.5 * (response + response.T)


def effective_mass_from_point_response(
    point_inverse_mass_world_per_kg,
    direction_world,
) -> float:
    """Project a point response tensor into a directional effective mass."""

    response = np.asarray(point_inverse_mass_world_per_kg, dtype=np.float64)
    direction = np.asarray(direction_world, dtype=np.float64)
    if response.shape != (3, 3) or not np.all(np.isfinite(response)):
        raise ValueError("point inverse-mass response must be a finite 3x3 matrix")
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1.0e-12:
        raise ValueError("impact direction must be non-zero")
    direction /= magnitude
    inverse_mass = float(direction @ response @ direction)
    if inverse_mass <= 1.0e-12:
        raise ValueError("point response suppresses the requested impact direction")
    return 1.0 / inverse_mass


def directional_effective_mass(
    mass_matrix,
    point_linear_jacobian,
    direction_world,
    *,
    constraint_jacobian=None,
    pseudoinverse_rcond: float = 1.0e-10,
) -> DirectionalEffectiveMass:
    """Return free and optionally constraint-projected scalar impact mass.

    The free inverse mass is ``j M^-1 j.T``.  With ideal bilateral support
    constraints ``C qdot = 0``, the inverse mass uses the standard projected
    response ``M^-1 - M^-1 C.T (C M^-1 C.T)^+ C M^-1``.
    """

    free_response = point_inverse_mass_response(
        mass_matrix, point_linear_jacobian
    )
    free_mass = effective_mass_from_point_response(
        free_response, direction_world
    )
    inverse_free = 1.0 / free_mass

    inverse_constrained = None
    constrained_mass = None
    if constraint_jacobian is not None:
        constrained_response = point_inverse_mass_response(
            mass_matrix,
            point_linear_jacobian,
            constraint_jacobian=constraint_jacobian,
            pseudoinverse_rcond=pseudoinverse_rcond,
        )
        constrained_mass = effective_mass_from_point_response(
            constrained_response, direction_world
        )
        inverse_constrained = 1.0 / constrained_mass

    return DirectionalEffectiveMass(
        free_kg=free_mass,
        constrained_kg=constrained_mass,
        inverse_free_per_kg=inverse_free,
        inverse_constrained_per_kg=inverse_constrained,
    )

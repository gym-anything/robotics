"""Small dependency-free 3-D math helpers used by the football modules.

The package deliberately keeps its reference flight solver independent of
NumPy, Isaac Sim, and USD.  Besides making the scientific probes fast to run,
this gives the Isaac adapter a second implementation against which its force
and coordinate conventions can be checked.

Vectors are ordinary ``(x, y, z)`` tuples.  Quaternions are scalar-first
``(w, x, y, z)`` and rotate body-frame vectors into world coordinates.
"""

from __future__ import annotations

import math
from typing import Iterable, TypeAlias


Vec3: TypeAlias = tuple[float, float, float]
Quat: TypeAlias = tuple[float, float, float, float]

ZERO: Vec3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION: Quat = (1.0, 0.0, 0.0, 0.0)


def vec3(values: Iterable[float]) -> Vec3:
    x, y, z = values
    return float(x), float(y), float(z)


def add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def scale(scalar: float, value: Vec3) -> Vec3:
    return scalar * value[0], scalar * value[1], scalar * value[2]


def hadamard(a: Vec3, b: Vec3) -> Vec3:
    return a[0] * b[0], a[1] * b[1], a[2] * b[2]


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm_squared(value: Vec3) -> float:
    return dot(value, value)


def norm(value: Vec3) -> float:
    return math.sqrt(norm_squared(value))


def unit(value: Vec3, fallback: Vec3 = ZERO, epsilon: float = 1.0e-12) -> Vec3:
    magnitude = norm(value)
    return fallback if magnitude <= epsilon else scale(1.0 / magnitude, value)


def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return max(minimum, min(maximum, value))


def lerp(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return add(a, scale(alpha, sub(b, a)))


def project(value: Vec3, onto: Vec3) -> Vec3:
    direction = unit(onto)
    return scale(dot(value, direction), direction)


def reject(value: Vec3, from_direction: Vec3) -> Vec3:
    return sub(value, project(value, from_direction))


def is_finite(value: Vec3) -> bool:
    return all(math.isfinite(component) for component in value)


def orthonormal_transverse_basis(direction: Vec3) -> tuple[Vec3, Vec3]:
    """Return deterministic side/up axes perpendicular to ``direction``.

    For a shot along world +X this returns side +Y and up +Z.  Close to the
    vertical axis it changes reference axes without becoming singular.
    """

    forward = unit(direction, (1.0, 0.0, 0.0))
    reference_up = (0.0, 0.0, 1.0)
    if abs(dot(forward, reference_up)) > 0.95:
        reference_up = (0.0, 1.0, 0.0)
    side = unit(cross(reference_up, forward), (0.0, 1.0, 0.0))
    up = unit(cross(forward, side), (0.0, 0.0, 1.0))
    return side, up


def quat_normalize(quaternion: Quat) -> Quat:
    magnitude = math.sqrt(sum(component * component for component in quaternion))
    if magnitude <= 1.0e-15:
        return IDENTITY_QUATERNION
    return tuple(component / magnitude for component in quaternion)  # type: ignore[return-value]


def quat_conjugate(quaternion: Quat) -> Quat:
    w, x, y, z = quaternion
    return w, -x, -y, -z


def quat_multiply(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate(quaternion: Quat, value: Vec3) -> Vec3:
    q = quat_normalize(quaternion)
    rotated = quat_multiply(quat_multiply(q, (0.0, *value)), quat_conjugate(q))
    return rotated[1], rotated[2], rotated[3]


def quat_inverse_rotate(quaternion: Quat, value: Vec3) -> Vec3:
    return quat_rotate(quat_conjugate(quat_normalize(quaternion)), value)


def quat_integrate_body_to_world(quaternion: Quat, angular_velocity_world: Vec3, dt_s: float) -> Quat:
    """Integrate orientation using a world-frame angular velocity.

    The exponential-map update avoids the large norm drift of a raw Euler
    quaternion derivative while remaining deterministic and dependency-free.
    """

    angular_speed = norm(angular_velocity_world)
    if angular_speed <= 1.0e-15 or dt_s == 0.0:
        return quat_normalize(quaternion)
    half_angle = 0.5 * angular_speed * dt_s
    axis = scale(1.0 / angular_speed, angular_velocity_world)
    delta_world = (
        math.cos(half_angle),
        axis[0] * math.sin(half_angle),
        axis[1] * math.sin(half_angle),
        axis[2] * math.sin(half_angle),
    )
    return quat_normalize(quat_multiply(delta_world, quaternion))


def angular_distance(a: Quat, b: Quat) -> float:
    relative = quat_multiply(quat_conjugate(quat_normalize(a)), quat_normalize(b))
    return 2.0 * math.acos(clamp(abs(relative[0]), 0.0, 1.0))

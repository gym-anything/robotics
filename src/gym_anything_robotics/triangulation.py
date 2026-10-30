"""Least-squares triangulation for multi-view dashboard pointing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TriangulatedPoint:
    position_m: tuple[float, float, float]
    ray_depths_m: tuple[float, ...]
    residuals_m: tuple[float, ...]
    rms_residual_m: float
    condition_number: float


def triangulate_rays(rays: Sequence[dict[str, Any]]) -> TriangulatedPoint:
    """Find the point with minimum summed squared distance to 3D rays."""

    if not 2 <= len(rays) <= 16:
        raise ValueError("triangulation requires between 2 and 16 rays")
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for index, ray in enumerate(rays):
        if not isinstance(ray, dict):
            raise ValueError(f"ray {index} must be an object")
        origin = np.asarray(ray.get("origin_m"), dtype=np.float64)
        direction = np.asarray(ray.get("direction"), dtype=np.float64)
        if origin.shape != (3,) or direction.shape != (3,):
            raise ValueError(f"ray {index} origin and direction must have length three")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError(f"ray {index} must be finite")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError(f"ray {index} direction must be nonzero")
        origins.append(origin)
        directions.append(direction / norm)

    identity = np.eye(3, dtype=np.float64)
    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, directions, strict=True):
        projector = identity - np.outer(direction, direction)
        system += projector
        rhs += projector @ origin
    condition = float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition > 1.0e6:
        raise ValueError(
            "camera views are too similar; rotate farther before selecting again"
        )
    position = np.linalg.solve(system, rhs)
    depths: list[float] = []
    residuals: list[float] = []
    for origin, direction in zip(origins, directions, strict=True):
        relative = position - origin
        depth = float(np.dot(relative, direction))
        if depth <= 0.0:
            raise ValueError("triangulated point lies behind at least one camera")
        closest = origin + depth * direction
        depths.append(depth)
        residuals.append(float(np.linalg.norm(position - closest)))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return TriangulatedPoint(
        position_m=tuple(float(value) for value in position),
        ray_depths_m=tuple(depths),
        residuals_m=tuple(residuals),
        rms_residual_m=rms,
        condition_number=condition,
    )


__all__ = ["TriangulatedPoint", "triangulate_rays"]

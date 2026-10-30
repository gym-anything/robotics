from __future__ import annotations

import numpy as np
import pytest

from gym_anything_robotics.triangulation import triangulate_rays


def _ray(origin, point):
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(point, dtype=np.float64) - origin
    return {"origin_m": origin.tolist(), "direction": direction.tolist()}


def test_three_views_recover_their_common_point() -> None:
    point = np.asarray((0.61, -0.04, 0.83))
    result = triangulate_rays(
        (
            _ray((1.2, -0.7, 1.1), point),
            _ray((1.1, 0.6, 1.2), point),
            _ray((0.1, -0.1, 1.5), point),
        )
    )
    assert result.position_m == pytest.approx(point, abs=1.0e-12)
    assert result.rms_residual_m < 1.0e-12
    assert all(depth > 0.0 for depth in result.ray_depths_m)


def test_two_views_recover_their_common_point() -> None:
    point = np.asarray((0.61, -0.04, 0.83))
    result = triangulate_rays(
        (
            _ray((1.2, -0.7, 1.1), point),
            _ray((1.1, 0.6, 1.2), point),
        )
    )
    assert result.position_m == pytest.approx(point, abs=1.0e-12)
    assert result.rms_residual_m < 1.0e-12
    assert all(depth > 0.0 for depth in result.ray_depths_m)


def test_parallel_views_are_rejected() -> None:
    with pytest.raises(ValueError, match="too similar"):
        triangulate_rays(
            (
                {"origin_m": [0, 0, 0], "direction": [1, 0, 0]},
                {"origin_m": [0, 1, 0], "direction": [1, 0, 0]},
                {"origin_m": [0, 2, 0], "direction": [1, 0, 0]},
            )
        )

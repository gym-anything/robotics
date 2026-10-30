from __future__ import annotations

import numpy as np
import pytest

from laundry.newton_style3d_table_ccd import (
    TablePlaneCCDConfig,
    resolve_swept_particles_against_table_plane,
)


def _resolve(origin, proposed, velocity, *, config=None):
    count = len(origin)
    return resolve_swept_particles_against_table_plane(
        np.asarray(origin, dtype=np.float64),
        np.asarray(proposed, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
        np.full(count, 0.01),
        np.full(count, 0.20),
        time_step_s=0.1,
        table_top_z_m=0.0,
        table_half_extent_x_m=1.0,
        table_half_extent_y_m=1.0,
        config=config or TablePlaneCCDConfig(),
    )


def test_local_table_ccd_changes_only_the_impacting_particle() -> None:
    result = _resolve(
        ((0.0, 0.0, 0.02), (0.5, 0.0, 0.20)),
        ((0.1, 0.0, 0.00), (0.6, 0.0, 0.19)),
        ((1.0, 0.0, -0.2), (1.0, 0.0, -0.1)),
    )
    assert result.hit_mask.tolist() == [True, False]
    assert result.minimum_hit_time_fraction == pytest.approx(0.5)
    assert result.positions_m[0, 2] == pytest.approx(0.010001)
    assert result.velocities_m_s[0, 2] == 0.0
    np.testing.assert_allclose(result.positions_m[1], (0.6, 0.0, 0.19))
    np.testing.assert_allclose(result.velocities_m_s[1], (1.0, 0.0, -0.1))
    assert result.normal_impulse_kg_m_s == pytest.approx(0.04)
    assert result.removed_normal_kinetic_energy_j == pytest.approx(0.004)


def test_resting_contact_does_not_scale_unrelated_tangential_motion() -> None:
    result = _resolve(
        ((0.0, 0.0, 0.010001), (0.0, 0.0, 0.010001)),
        ((0.0, 0.0, 0.009), (0.3, 0.0, 0.010001)),
        ((0.0, 0.0, -0.01), (3.0, 0.0, 0.0)),
    )
    assert result.hit_count == 1
    np.testing.assert_allclose(result.positions_m[1], (0.3, 0.0, 0.010001))
    np.testing.assert_allclose(result.velocities_m_s[1], (3.0, 0.0, 0.0))


def test_path_outside_finite_table_is_not_top_plane_contact() -> None:
    result = _resolve(
        ((2.0, 0.0, 0.02),),
        ((2.0, 0.0, 0.00),),
        ((0.0, 0.0, -0.2),),
    )
    assert result.hit_count == 0
    assert result.positions_m[0, 2] == 0.0


def test_penetrating_origin_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="penetrating accepted state"):
        _resolve(
            ((0.0, 0.0, 0.009),),
            ((0.0, 0.0, 0.008),),
            ((0.0, 0.0, -0.01),),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"normal_restitution": -0.1}, "restitution"),
        ({"normal_restitution": 1.1}, "restitution"),
        ({"contact_epsilon_m": -1.0e-6}, "epsilon"),
        (
            {
                "contact_epsilon_m": 2.0e-6,
                "origin_penetration_tolerance_m": 1.0e-6,
            },
            "tolerance",
        ),
    ),
)
def test_table_plane_ccd_config_fails_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        TablePlaneCCDConfig(**kwargs)

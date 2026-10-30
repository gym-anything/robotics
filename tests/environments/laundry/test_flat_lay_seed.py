from __future__ import annotations

import numpy as np
import pytest

from laundry.flat_lay_seed import (
    FlatLayResetConfig,
    build_flat_lay_seed,
    supported_flat_lay_constructions,
)
from laundry.garment_assembly import weld_garment_topology
from laundry.garment_meshing import compile_garment_pattern_mesh
from laundry.garment_patterns import get_garment_construction


@pytest.fixture(scope="module")
def tee_pattern():
    construction = get_garment_construction("cotton_jersey_half_sleeve_tee")
    pattern = compile_garment_pattern_mesh(
        construction,
        target_edge_length_m=0.040,
    )
    return construction, pattern


def test_flat_lay_registry_is_explicit() -> None:
    assert supported_flat_lay_constructions() == (
        "cotton_jersey_half_sleeve_tee",
        "woven_button_down_shirt",
        "denim_five_pocket_jeans",
        "terry_cotton_bath_towel",
        "asymmetric_draped_one_sleeve_top",
    )
    towel = get_garment_construction("terry_cotton_bath_towel")
    pattern = compile_garment_pattern_mesh(towel, target_edge_length_m=0.080)
    result = build_flat_lay_seed(towel, pattern)
    assert result.report.strategy_id == "terry_rectangular_sheet_isometric_v1"
    np.testing.assert_allclose(
        result.seed.material_vertex_positions_m[:, 2],
        0.0,
        atol=0.0,
        rtol=0.0,
    )


def test_tee_flat_lay_seed_is_deterministic_and_finite(tee_pattern) -> None:
    construction, pattern = tee_pattern
    first = build_flat_lay_seed(construction, pattern)
    second = build_flat_lay_seed(construction, pattern)

    np.testing.assert_allclose(
        first.seed.material_vertex_positions_m,
        second.seed.material_vertex_positions_m,
        atol=0.0,
        rtol=0.0,
    )
    assert np.all(np.isfinite(first.seed.material_vertex_positions_m))
    assert first.report.strategy_id == "sewn_tee_flattened_shell_v1"
    assert first.report.reset_only_numerical_preconditioner
    assert first.report.body_layer_separation_m == pytest.approx(8.0e-3)
    assert first.report.harmonic_vertex_count_by_panel["left_sleeve"] > 0
    assert first.report.harmonic_vertex_count_by_panel["right_sleeve"] > 0


def test_tee_seed_compiles_to_closed_sewn_physical_topology(tee_pattern) -> None:
    construction, pattern = tee_pattern
    result = build_flat_lay_seed(
        construction,
        pattern,
        config=FlatLayResetConfig(body_layer_separation_m=8.0e-3),
    )
    topology = weld_garment_topology(
        construction,
        pattern,
        seed=result.seed,
        relax=False,
    )

    # Exact deterministic resolution after quality-spaced seam synchronization
    # and internal-landmark snapping; closure guards reuse nearby fold-axis
    # vertices instead of creating pathological sliver pairs.
    assert topology.physical_positions_m.shape == (650, 3)
    assert topology.physical_triangles.shape == (1248, 3)
    assert topology.strain.degenerate_triangle_count == 0
    assert max(topology.boundary_edge_incidence.values()) <= 2
    front_start, front_stop = pattern.panel_vertex_ranges["front"]
    back_start, back_stop = pattern.panel_vertex_ranges["back"]
    front_interior = np.setdiff1d(
        np.arange(front_start, front_stop),
        np.concatenate(
            [pattern.path_vertices[("front", path.id)] for path in construction.panel_map["front"].outline_paths]
        ),
    )
    back_interior = np.setdiff1d(
        np.arange(back_start, back_stop),
        np.concatenate(
            [pattern.path_vertices[("back", path.id)] for path in construction.panel_map["back"].outline_paths]
        ),
    )
    assert float(np.median(result.seed.material_vertex_positions_m[front_interior, 2])) > 0.0
    assert float(np.median(result.seed.material_vertex_positions_m[back_interior, 2])) < 0.0


def test_flat_lay_config_rejects_nonphysical_numerics() -> None:
    with pytest.raises(ValueError, match="separation"):
        FlatLayResetConfig(body_layer_separation_m=0.0)
    with pytest.raises(ValueError, match="offset"):
        FlatLayResetConfig(neck_opening_center_offset_fraction=float("nan"))

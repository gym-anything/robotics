from __future__ import annotations

import numpy as np
import pytest

from laundry.garment_assembly import (
    material_to_physical_vertex_map,
    weld_garment_topology,
)
from laundry.garment_meshing import compile_garment_pattern_mesh
from laundry.garment_patterns import get_garment_construction
from laundry.garment_render_mesh import (
    build_garment_panel_render_topology,
    deform_thick_garment_panel,
)


@pytest.fixture(scope="module")
def tee_topology():
    construction = get_garment_construction("cotton_jersey_half_sleeve_tee")
    pattern = compile_garment_pattern_mesh(
        construction, target_edge_length_m=0.040
    )
    sewn = weld_garment_topology(construction, pattern, relax=False)
    return construction, pattern, sewn


def test_panel_render_embedding_is_barycentric_and_closed(tee_topology) -> None:
    construction, pattern, sewn = tee_topology
    panel = construction.panel_map["front"]
    render = build_garment_panel_render_topology(
        pattern,
        sewn,
        panel.id,
        subdivision_factor=2,
        texture_repeats_per_m=70.0,
    )

    embedding = render.embedding
    assert embedding.vertex_count > pattern.panel_meshes[0].vertex_count
    assert len(embedding.surface_triangles) == 4 * len(
        pattern.panel_meshes[0].triangles
    )
    assert len(embedding.boundary_segments) > 0
    np.testing.assert_allclose(
        np.sum(embedding.barycentric_weights, axis=1), 1.0
    )
    assert np.min(embedding.barycentric_weights) >= 0.0
    assert set(np.unique(render.face_vertex_counts)) == {3, 4}
    assert int(np.sum(render.face_vertex_counts)) == len(render.face_vertex_indices)


def test_public_stitch_union_map_matches_the_physical_topology(tee_topology) -> None:
    _construction, pattern, sewn = tee_topology
    np.testing.assert_array_equal(
        material_to_physical_vertex_map(pattern),
        sewn.material_to_physical_vertex,
    )


def test_panel_shell_uses_declared_thickness_without_changing_physics(
    tee_topology,
) -> None:
    construction, pattern, sewn = tee_topology
    panel = construction.panel_map["front"]
    original = sewn.physical_positions_m.copy()
    render = build_garment_panel_render_topology(
        pattern, sewn, panel.id, subdivision_factor=2
    )
    points, normals = deform_thick_garment_panel(
        render,
        sewn.physical_positions_m,
        thickness_m=panel.unloaded_thickness_m,
    )

    count = render.embedding.vertex_count
    assert points.shape == (2 * count, 3)
    assert normals.shape == points.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(points[:count] - points[count:], axis=1),
        panel.unloaded_thickness_m,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(sewn.physical_positions_m, original)


def test_unknown_panel_and_invalid_visual_controls_are_rejected(
    tee_topology,
) -> None:
    _construction, pattern, sewn = tee_topology
    with pytest.raises(KeyError, match="Unknown garment panel"):
        build_garment_panel_render_topology(pattern, sewn, "rectangle_substitute")
    with pytest.raises(ValueError, match="subdivision"):
        build_garment_panel_render_topology(
            pattern, sewn, "front", subdivision_factor=0
        )
    with pytest.raises(ValueError, match="repeat density"):
        build_garment_panel_render_topology(
            pattern, sewn, "front", texture_repeats_per_m=0.0
        )

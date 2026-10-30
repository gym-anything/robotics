from __future__ import annotations

import unittest

import numpy as np

from laundry.mesh import build_rectangular_mesh
from laundry.render_mesh import (
    build_render_embedding,
    build_thick_cloth_topology,
    deform_thick_cloth,
)


class RenderMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=4,
            ny=2,
            areal_density_kg_m2=0.3,
            origin_m=(0.0, 0.0, 0.02),
        )

    def test_barycentric_subdivision_preserves_flat_material_geometry(self) -> None:
        embedding = build_render_embedding(self.mesh, subdivision_factor=3)
        rendered = embedding.interpolate(self.mesh.positions_m)
        np.testing.assert_allclose(
            rendered[:, :2], embedding.material_coordinates_m, atol=1.0e-14
        )
        np.testing.assert_allclose(rendered[:, 2], 0.02, atol=1.0e-14)
        np.testing.assert_allclose(
            np.sum(embedding.barycentric_weights, axis=1), 1.0
        )
        self.assertGreaterEqual(float(np.min(embedding.barycentric_weights)), 0.0)

    def test_every_original_vertex_is_preserved_exactly(self) -> None:
        factor = 4
        embedding = build_render_embedding(self.mesh, subdivision_factor=factor)
        rendered = embedding.interpolate(self.mesh.positions_m)
        for iy in range(self.mesh.ny + 1):
            for ix in range(self.mesh.nx + 1):
                coarse = iy * (self.mesh.nx + 1) + ix
                high = (iy * factor) * (embedding.nx + 1) + ix * factor
                np.testing.assert_array_equal(rendered[high], self.mesh.positions_m[coarse])

    def test_thick_shell_separation_and_topology_are_finite(self) -> None:
        topology = build_thick_cloth_topology(
            self.mesh, subdivision_factor=2
        )
        points, normals = deform_thick_cloth(
            topology, self.mesh.positions_m, thickness_m=0.0008
        )
        count = topology.embedding.vertex_count
        self.assertEqual(points.shape, (2 * count, 3))
        self.assertEqual(normals.shape, points.shape)
        np.testing.assert_allclose(points[:count, 2], 0.0204, atol=1.0e-14)
        np.testing.assert_allclose(points[count:, 2], 0.0196, atol=1.0e-14)
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertEqual(
            int(np.sum(topology.face_vertex_counts)),
            len(topology.face_vertex_indices),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

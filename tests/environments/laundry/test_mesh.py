from __future__ import annotations

import unittest

import numpy as np

from laundry.mesh import build_rectangular_mesh, triangle_signed_areas_xy


class MeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=12,
            ny=8,
            areal_density_kg_m2=0.3,
        )

    def test_mass_area_and_winding_are_exact(self) -> None:
        self.assertAlmostEqual(self.mesh.total_area_m2, 0.24, places=13)
        self.assertAlmostEqual(self.mesh.total_mass_kg, 0.072, places=13)
        signed = triangle_signed_areas_xy(self.mesh.positions_m, self.mesh.triangles)
        self.assertTrue(np.all(signed > 0.0))

    def test_boundaries_and_hinges_are_valid(self) -> None:
        self.assertEqual(len(self.mesh.boundary["left"]), 9)
        self.assertEqual(len(self.mesh.boundary["top"]), 13)
        self.assertGreater(len(self.mesh.hinges), 0)
        for hinge in self.mesh.hinges:
            self.assertEqual(len(set((*hinge.edge_vertices, *hinge.opposite_vertices))), 4)
            self.assertGreater(hinge.rest_edge_length_m, 0.0)

    def test_alternating_diagonals_do_not_change_lumped_mass(self) -> None:
        fixed = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=12,
            ny=8,
            areal_density_kg_m2=0.3,
            alternating_diagonals=False,
        )
        self.assertAlmostEqual(fixed.total_mass_kg, self.mesh.total_mass_kg, places=13)
        self.assertAlmostEqual(fixed.total_area_m2, self.mesh.total_area_m2, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)

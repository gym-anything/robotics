from __future__ import annotations

import unittest

import numpy as np

from laundry.garment_meshing import compile_garment_pattern_mesh, triangulate_panel
from laundry.garment_patterns import GARMENT_CONSTRUCTIONS, get_garment_construction


class GarmentMeshingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.meshes = {
            garment.id: compile_garment_pattern_mesh(
                garment,
                target_edge_length_m=0.035,
            )
            for garment in GARMENT_CONSTRUCTIONS.values()
        }

    def test_every_panel_is_positive_finite_and_area_exact(self) -> None:
        for construction_id, mesh in self.meshes.items():
            with self.subTest(garment=construction_id):
                self.assertTrue(np.all(np.isfinite(mesh.positions_m)))
                self.assertTrue(np.all(np.isfinite(mesh.material_coordinates_m)))
                self.assertTrue(np.all(mesh.vertex_area_m2 >= 0.0))
                for panel_mesh in mesh.panel_meshes:
                    self.assertGreater(panel_mesh.vertex_count, 3)
                    self.assertGreater(panel_mesh.triangle_count, 1)
                    self.assertGreater(float(np.min(panel_mesh.triangle_signed_areas_m2)), 0.0)
                    self.assertLess(panel_mesh.area_error_fraction, 5.0e-4)
                    self.assertLess(panel_mesh.maximum_aspect_ratio, 10.0)
                    self.assertAlmostEqual(
                        panel_mesh.total_mass_kg,
                        panel_mesh.source_area_m2
                        * GARMENT_CONSTRUCTIONS[construction_id].panel_map[
                            panel_mesh.panel_id
                        ].areal_density_kg_m2,
                        places=10,
                    )

    def test_compiled_area_and_mass_match_construction_accounting(self) -> None:
        for construction_id, mesh in self.meshes.items():
            construction = GARMENT_CONSTRUCTIONS[construction_id]
            with self.subTest(garment=construction_id):
                self.assertAlmostEqual(mesh.total_area_m2, construction.total_panel_area_m2, places=9)
                self.assertAlmostEqual(mesh.total_mass_kg, construction.total_panel_mass_kg, places=9)

    def test_triangles_never_bridge_disconnected_pattern_panels(self) -> None:
        for mesh in self.meshes.values():
            panel_at_triangle_vertices = mesh.vertex_panel_index[mesh.triangles]
            self.assertTrue(
                np.all(panel_at_triangle_vertices == panel_at_triangle_vertices[:, :1]),
                mesh.construction_id,
            )
            self.assertTrue(np.array_equal(panel_at_triangle_vertices[:, 0], mesh.triangle_panel_index))

    def test_named_outline_and_internal_stitch_paths_survive_triangulation(self) -> None:
        jeans = self.meshes["denim_five_pocket_jeans"]
        for key in (
            ("front_left", "pocket_opening"),
            ("back_right", "patch_bottom_mount"),
            ("coin_pocket", "opening"),
            ("fly_shield", "rise_attach"),
        ):
            with self.subTest(path=key):
                indices = jeans.path_vertices[key]
                fractions = jeans.path_fractions[key]
                self.assertGreaterEqual(len(indices), 2)
                self.assertEqual(len(indices), len(fractions))
                self.assertAlmostEqual(float(fractions[0]), 0.0)
                self.assertAlmostEqual(float(fractions[-1]), 1.0)

    def test_each_declared_seam_becomes_one_nonduplicated_stitch_batch(self) -> None:
        for construction_id, mesh in self.meshes.items():
            construction = GARMENT_CONSTRUCTIONS[construction_id]
            with self.subTest(garment=construction_id):
                self.assertEqual(
                    tuple(stitch.seam_id for stitch in mesh.stitches),
                    tuple(seam.id for seam in construction.seams),
                )
                for stitch in mesh.stitches:
                    self.assertEqual(
                        len(np.unique(stitch.vertex_pairs, axis=0)),
                        len(stitch.vertex_pairs),
                    )
                    self.assertGreaterEqual(
                        stitch.numerical_spacing_m,
                        0.75 * stitch.physical_stitch_spacing_m,
                    )
                    self.assertGreater(stitch.per_constraint_stiffness_n_m, 0.0)

    def test_mesh_refinement_increases_resolution_without_changing_mass(self) -> None:
        towel = get_garment_construction("terry_cotton_bath_towel")
        coarse = compile_garment_pattern_mesh(towel, target_edge_length_m=0.050)
        fine = compile_garment_pattern_mesh(towel, target_edge_length_m=0.025)
        self.assertGreater(fine.vertex_count, coarse.vertex_count)
        self.assertGreater(fine.triangle_count, coarse.triangle_count)
        self.assertAlmostEqual(fine.total_mass_kg, coarse.total_mass_kg, places=10)

    def test_invalid_target_edge_length_is_rejected(self) -> None:
        panel = get_garment_construction("terry_cotton_bath_towel").panels[0]
        with self.assertRaises(ValueError):
            triangulate_panel(panel, target_edge_length_m=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

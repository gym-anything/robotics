from __future__ import annotations

import unittest

import numpy as np

from laundry.garment_assembly import weld_garment_topology
from laundry.garment_meshing import compile_garment_pattern_mesh
from laundry.garment_patterns import GARMENT_CONSTRUCTIONS, get_garment_construction
from laundry.newton_style3d_garment import (
    build_style3d_inputs,
    oriented_triangle_edge_audit,
    style3d_projective_neighbor_counts,
)


class SewnGarmentTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pattern_meshes = {
            garment.id: compile_garment_pattern_mesh(
                garment,
                target_edge_length_m=0.040,
            )
            for garment in GARMENT_CONSTRUCTIONS.values()
        }
        cls.unrelaxed = {
            garment_id: weld_garment_topology(
                GARMENT_CONSTRUCTIONS[garment_id],
                mesh,
                relax=False,
            )
            for garment_id, mesh in cls.pattern_meshes.items()
        }

    def test_welding_reduces_vertex_count_only_for_sewn_constructions(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            with self.subTest(garment=construction_id):
                if GARMENT_CONSTRUCTIONS[construction_id].seams:
                    self.assertLess(
                        topology.physical_vertex_count,
                        topology.material_vertex_count,
                    )
                    self.assertGreater(topology.welded_material_vertex_count, 0)
                else:
                    self.assertEqual(
                        topology.physical_vertex_count,
                        topology.material_vertex_count,
                    )

    def test_physical_and_panel_topologies_are_distinct_but_triangle_aligned(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            with self.subTest(garment=construction_id):
                self.assertEqual(
                    topology.physical_triangles.shape,
                    topology.panel_triangles.shape,
                )
                self.assertTrue(
                    np.array_equal(
                        topology.physical_triangles,
                        topology.material_to_physical_vertex[topology.panel_triangles],
                    )
                )
                self.assertFalse(
                    np.any(
                        (topology.physical_triangles[:, 0] == topology.physical_triangles[:, 1])
                        | (topology.physical_triangles[:, 1] == topology.physical_triangles[:, 2])
                        | (topology.physical_triangles[:, 2] == topology.physical_triangles[:, 0])
                    )
                )

    def test_every_stitch_pair_maps_to_one_shared_physical_vertex(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            pattern = self.pattern_meshes[construction_id]
            with self.subTest(garment=construction_id):
                for stitch in pattern.stitches:
                    physical_a = topology.material_to_physical_vertex[
                        stitch.vertex_pairs[:, 0]
                    ]
                    physical_b = topology.material_to_physical_vertex[
                        stitch.vertex_pairs[:, 1]
                    ]
                    self.assertTrue(np.array_equal(physical_a, physical_b), stitch.seam_id)
                    self.assertTrue(
                        np.array_equal(
                            physical_a,
                            topology.seam_physical_vertices[stitch.seam_id],
                        )
                    )

    def test_structural_boundary_seams_are_shared_physical_edges(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            garment = GARMENT_CONSTRUCTIONS[construction_id]
            for seam in garment.seams:
                panel_a = garment.panel_map[seam.side_a.panel_id]
                panel_b = garment.panel_map[seam.side_b.panel_id]
                outline_a = {path.id for path in panel_a.outline_paths}
                outline_b = {path.id for path in panel_b.outline_paths}
                if seam.side_a.path_id not in outline_a or seam.side_b.path_id not in outline_b:
                    continue  # discrete internal topstitch/pocket mount
                for edge in topology.seam_physical_edges[seam.id]:
                    key = tuple(sorted(int(value) for value in edge))
                    self.assertGreaterEqual(
                        topology.boundary_edge_incidence.get(key, 0),
                        2,
                        f"{construction_id}:{seam.id}:{key}",
                    )

    def test_welding_never_creates_non_manifold_cloth_edges(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            with self.subTest(garment=construction_id):
                self.assertLessEqual(
                    max(topology.boundary_edge_incidence.values(), default=0),
                    2,
                )

    def test_style3d_winding_is_consistent_across_every_shared_edge(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            construction = GARMENT_CONSTRUCTIONS[construction_id]
            inputs = build_style3d_inputs(construction, topology)
            audit = oriented_triangle_edge_audit(inputs.physical_triangles)
            with self.subTest(garment=construction_id):
                self.assertEqual(audit["nonmanifold_edge_count"], 0)
                self.assertEqual(audit["orientation_mismatch_edge_count"], 0)

    def test_every_canonical_topology_fits_newton_style3d_sparse_capacity(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            with self.subTest(garment=construction_id):
                counts = style3d_projective_neighbor_counts(
                    topology.physical_triangles
                )
                self.assertLessEqual(int(np.max(counts)), 32)

    def test_welded_mass_exactly_matches_cut_panel_accounting(self) -> None:
        for construction_id, topology in self.unrelaxed.items():
            self.assertAlmostEqual(
                topology.total_mass_kg,
                GARMENT_CONSTRUCTIONS[construction_id].total_panel_mass_kg,
                places=10,
            )

    def test_mirrored_jeans_cut_pieces_seed_distinct_left_and_right_legs(self) -> None:
        construction_id = "denim_five_pocket_jeans"
        pattern = self.pattern_meshes[construction_id]
        topology = self.unrelaxed[construction_id]

        def centroid(panel_id: str) -> np.ndarray:
            start, stop = pattern.panel_vertex_ranges[panel_id]
            physical = np.unique(
                topology.material_to_physical_vertex[start:stop]
            )
            return np.mean(topology.physical_positions_m[physical], axis=0)

        left = 0.5 * (centroid("front_left") + centroid("back_left"))
        right = 0.5 * (centroid("front_right") + centroid("back_right"))
        self.assertGreater(np.linalg.norm(left[:2] - right[:2]), 0.12)
        self.assertTrue(topology.seed.panel_poses["front_right"].mirror_material_x)
        self.assertTrue(topology.seed.panel_poses["back_right"].mirror_material_x)

    def test_tee_neck_band_does_not_pin_left_and_right_shoulders_together(self) -> None:
        construction_id = "cotton_jersey_half_sleeve_tee"
        pattern = self.pattern_meshes[construction_id]
        topology = self.unrelaxed[construction_id]
        neckline = pattern.path_vertices[("front", "neckline")]
        right = int(topology.material_to_physical_vertex[int(neckline[0])])
        left = int(topology.material_to_physical_vertex[int(neckline[-1])])

        self.assertNotEqual(right, left)
        self.assertGreater(
            np.linalg.norm(
                topology.physical_positions_m[right]
                - topology.physical_positions_m[left]
            ),
            0.10,
        )

    def test_relaxation_removes_extreme_seed_distortion_without_degenerate_faces(self) -> None:
        tee = get_garment_construction("cotton_jersey_half_sleeve_tee")
        pattern = self.pattern_meshes[tee.id]
        raw = self.unrelaxed[tee.id]
        relaxed = weld_garment_topology(
            tee,
            pattern,
            seed=raw.seed,
            relax=True,
            relaxation_iterations=500,
        )
        self.assertTrue(relaxed.relaxation.applied)
        self.assertLess(
            relaxed.relaxation.final_edge_energy,
            0.01 * relaxed.relaxation.initial_edge_energy,
        )
        self.assertLess(relaxed.strain.maximum_principal_stretch, 1.30)
        self.assertGreater(relaxed.strain.minimum_principal_stretch, 0.80)
        self.assertLess(
            relaxed.strain.maximum_principal_stretch,
            raw.strain.maximum_principal_stretch,
        )
        self.assertEqual(relaxed.strain.degenerate_triangle_count, 0)

    def test_panel_restricted_relaxation_keeps_welded_host_vertices_fixed(self) -> None:
        tee = get_garment_construction("cotton_jersey_half_sleeve_tee")
        pattern = self.pattern_meshes[tee.id]
        raw = self.unrelaxed[tee.id]
        selected_panel_ids = ("left_sleeve", "right_sleeve")
        relaxed = weld_garment_topology(
            tee,
            pattern,
            seed=raw.seed,
            relax=True,
            relaxation_iterations=300,
            relaxation_panel_ids=selected_panel_ids,
        )

        mapping = raw.material_to_physical_vertex
        source_count = np.bincount(
            mapping,
            minlength=raw.physical_vertex_count,
        )
        selected_material = np.zeros(pattern.vertex_count, dtype=np.int32)
        for panel_id in selected_panel_ids:
            start, stop = pattern.panel_vertex_ranges[panel_id]
            selected_material[start:stop] = 1
        selected_source_count = np.bincount(
            mapping,
            weights=selected_material,
            minlength=raw.physical_vertex_count,
        )
        movable = selected_source_count == source_count

        self.assertTrue(np.any(movable))
        self.assertTrue(np.any(~movable))
        np.testing.assert_allclose(
            relaxed.physical_positions_m[~movable],
            raw.physical_positions_m[~movable],
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(
            float(
                np.max(
                    np.linalg.norm(
                        relaxed.physical_positions_m[movable]
                        - raw.physical_positions_m[movable],
                        axis=1,
                    )
                )
            ),
            1.0e-4,
        )
        self.assertEqual(
            relaxed.relaxation.relaxed_panel_ids,
            selected_panel_ids,
        )
        # Armhole particles are shared with the unselected body and therefore
        # remain hard Dirichlet boundaries during a sleeve-only pass.
        armhole = raw.seam_physical_vertices["left_sleeve_front_cap"]
        self.assertTrue(np.all(~movable[armhole]))

    def test_panel_restricted_relaxation_rejects_unknown_panels(self) -> None:
        tee = get_garment_construction("cotton_jersey_half_sleeve_tee")
        with self.assertRaisesRegex(ValueError, "unknown panels"):
            weld_garment_topology(
                tee,
                self.pattern_meshes[tee.id],
                relax=True,
                relaxation_panel_ids=("imaginary_sleeve",),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

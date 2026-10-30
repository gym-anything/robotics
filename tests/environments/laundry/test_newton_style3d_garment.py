from __future__ import annotations

import unittest

import numpy as np

from laundry.garment_assembly import weld_garment_topology
from laundry.garment_meshing import compile_garment_pattern_mesh
from laundry.garment_patterns import get_garment_construction
from laundry.newton_style3d_garment import Style3DSolverConfig, build_style3d_inputs


class Style3DGarmentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.garment = get_garment_construction("cotton_jersey_half_sleeve_tee")
        cls.pattern = compile_garment_pattern_mesh(
            cls.garment,
            target_edge_length_m=0.040,
        )
        cls.topology = weld_garment_topology(cls.garment, cls.pattern, relax=False)
        cls.inputs = build_style3d_inputs(cls.garment, cls.topology)

    def test_official_split_topology_arrays_remain_triangle_aligned(self) -> None:
        self.assertEqual(self.inputs.physical_triangles.shape, self.inputs.panel_triangles.shape)
        self.assertEqual(self.inputs.triangle_stretch_n_m.shape[0], self.inputs.triangle_count)
        self.assertGreater(self.inputs.material_vertex_count, self.inputs.particle_count)
        np.testing.assert_array_equal(
            self.inputs.physical_triangles,
            self.topology.material_to_physical_vertex[self.inputs.panel_triangles],
        )

    def test_turned_over_panels_reverse_both_aligned_triangle_arrays(self) -> None:
        reversed_ids = {
            panel_id
            for panel_id, reversed_winding in zip(
                self.inputs.panel_ids,
                self.inputs.panel_physical_winding_reversed,
                strict=True,
            )
            if reversed_winding
        }
        self.assertEqual(reversed_ids, {"back", "right_sleeve", "neck_rib"})

    def test_grain_rotation_preserves_positive_panel_triangle_area(self) -> None:
        uv = self.inputs.grain_aligned_panel_coordinates_m[self.inputs.panel_triangles]
        double_area = (
            (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
            - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0])
        )
        self.assertGreater(float(np.min(double_area)), 0.0)

    def test_exact_welded_mass_and_positive_per_particle_thickness(self) -> None:
        self.assertAlmostEqual(
            self.inputs.total_mass_kg,
            self.garment.total_panel_mass_kg,
            places=10,
        )
        self.assertGreater(float(np.min(self.inputs.particle_radius_m)), 0.0)

    def test_unrepresented_physics_is_explicit(self) -> None:
        joined = " ".join(self.inputs.unsupported_terms)
        self.assertIn("nonlinear", joined)
        self.assertIn("pressure-dependent", joined)
        self.assertIn("seam", joined)
        self.assertIn("damping", joined)

    def test_solver_config_rejects_nonphysical_numerics(self) -> None:
        with self.assertRaises(ValueError):
            Style3DSolverConfig(time_step_s=0.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(self_contact_radius_m=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(
                gravity_m_s2=9.80665,
                enforce_free_rigid_frame=True,
            )
        with self.assertRaises(ValueError):
            Style3DSolverConfig(
                enable_table=True,
                enforce_free_rigid_frame=True,
            )
        with self.assertRaises(ValueError):
            Style3DSolverConfig(table_thickness_m=0.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(self_contact_vertex_face_stiffness=-0.1)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(self_contact_topological_filter_graph_hops=-1)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(membrane_rate_damping_scale=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(bending_rate_damping_scale=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(air_density_kg_m3=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(normal_drag_coefficient=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(wind_velocity_m_s=(0.0, float("nan"), 0.0))
        with self.assertRaises(ValueError):
            Style3DSolverConfig(structural_seam_bending_scale=-1.0)
        with self.assertRaises(ValueError):
            Style3DSolverConfig(enable_table_normal_projection=True)

        from laundry.newton_style3d_table_ccd import TablePlaneCCDConfig

        with self.assertRaises(ValueError):
            Style3DSolverConfig(table_plane_ccd=TablePlaneCCDConfig())
        with self.assertRaises(ValueError):
            Style3DSolverConfig(
                enable_table=True,
                enable_table_normal_projection=True,
                table_plane_ccd=TablePlaneCCDConfig(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

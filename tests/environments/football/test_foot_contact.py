from __future__ import annotations

import math
import unittest

from football.foot_contact import FootPatchSpec, sphere_obb_contact


class FootPatchGeometryTests(unittest.TestCase):
    def test_centered_toe_contact_has_exact_compression_and_normal(self) -> None:
        patch = FootPatchSpec()
        radius = 0.11
        compression = 0.012
        ball_x = patch.maximum_local_m[0] + radius - compression
        result = sphere_obb_contact(
            ball_center_world_m=(ball_x, 0.0, 0.0),
            ball_radius_m=radius,
            foot_center_world_m=(0.0, 0.0, 0.0),
            foot_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            patch=patch,
        )
        self.assertAlmostEqual(result.compression_m, compression, places=10)
        self.assertEqual(result.normal_foot_to_ball_world, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(result.closest_point_on_foot_world_m[0], patch.maximum_local_m[0])

    def test_orientation_rotates_contact_geometry(self) -> None:
        patch = FootPatchSpec()
        half = math.sqrt(0.5)
        result = sphere_obb_contact(
            ball_center_world_m=(0.0, patch.maximum_local_m[0] + 0.10, 0.0),
            ball_radius_m=0.11,
            foot_center_world_m=(0.0, 0.0, 0.0),
            foot_orientation_world_wxyz=(half, 0.0, 0.0, half),
            patch=patch,
        )
        self.assertGreater(result.compression_m, 0.0)
        self.assertAlmostEqual(result.normal_foot_to_ball_world[0], 0.0, places=7)
        self.assertAlmostEqual(result.normal_foot_to_ball_world[1], 1.0, places=7)

    def test_inside_case_uses_nearest_outward_face(self) -> None:
        patch = FootPatchSpec()
        result = sphere_obb_contact(
            ball_center_world_m=(patch.maximum_local_m[0] - 0.001, 0.0, 0.0),
            ball_radius_m=0.11,
            foot_center_world_m=(0.0, 0.0, 0.0),
            foot_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            patch=patch,
        )
        self.assertEqual(result.feature, "inside_x_max")
        self.assertAlmostEqual(result.compression_m, 0.111, places=9)
        self.assertEqual(result.normal_foot_to_ball_world, (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

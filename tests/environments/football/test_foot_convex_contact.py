from __future__ import annotations

import math
import unittest

import numpy as np

from football.foot_contact import (
    FootPatchSpec,
    load_convex_foot_proxy,
    sphere_convex_contact,
    sphere_obb_contact,
    validate_proxy_against_patch,
)
from football.football_math import add, quat_rotate
from football.football_specs import BallSpec


class ConvexFootProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = FootPatchSpec()
        cls.proxy = load_convex_foot_proxy()
        validate_proxy_against_patch(cls.proxy, cls.patch)

    def test_pinned_proxy_provenance_and_topology(self) -> None:
        audit = self.proxy.audit()
        self.assertEqual(audit["asset_sha256"], self.patch.proxy_asset_sha256)
        self.assertEqual(
            audit["source"]["sha256"], self.patch.source_mesh_sha256
        )
        self.assertEqual(audit["geometry"]["vertex_count"], 1248)
        self.assertEqual(audit["geometry"]["triangle_count"], 2492)
        self.assertEqual(audit["geometry"]["unique_edge_count"], 3738)
        # Euler V-E+F=2 is a compact closed-convex-topology gate.
        self.assertEqual(1248 - 3738 + 2492, 2)

    def test_recorded_peak_point_matches_visible_mesh_reference(self) -> None:
        point = (0.237083953, -0.036183001, 0.008692605)
        query = self.proxy.query_local(point)
        self.assertAlmostEqual(query.signed_distance_m, 0.1039699564, places=9)
        np.testing.assert_allclose(
            query.closest_point_local_m,
            (0.14152202, -0.00798803, -0.02101787),
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            query.outward_normal_local,
            (0.91913026, -0.27118381, 0.28576023),
            atol=2.0e-8,
        )

        ball = BallSpec()
        convex = sphere_convex_contact(
            ball_center_world_m=point,
            ball_radius_m=ball.radius_m,
            foot_center_world_m=(0.0, 0.0, 0.0),
            foot_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            patch=self.patch,
            proxy=self.proxy,
        )
        box = sphere_obb_contact(
            ball_center_world_m=point,
            ball_radius_m=ball.radius_m,
            foot_center_world_m=(0.0, 0.0, 0.0),
            foot_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            patch=self.patch,
        )
        self.assertGreater(convex.compression_m, 0.0)
        self.assertGreater(box.compression_m, convex.compression_m)
        self.assertAlmostEqual(
            box.signed_distance_m - convex.signed_distance_m,
            -0.0092513549,
            places=8,
        )

    def test_world_transform_is_rotation_equivariant(self) -> None:
        point_local = (0.230, -0.020, 0.010)
        local = self.proxy.query_local(point_local)
        half = math.sqrt(0.5)
        orientation = (half, 0.0, 0.0, half)
        translation = (1.2, -0.4, 0.8)
        point_world = add(translation, quat_rotate(orientation, point_local))
        result = sphere_convex_contact(
            ball_center_world_m=point_world,
            ball_radius_m=BallSpec().radius_m,
            foot_center_world_m=translation,
            foot_orientation_world_wxyz=orientation,
            patch=self.patch,
            proxy=self.proxy,
        )
        expected_closest = add(
            translation,
            quat_rotate(orientation, local.closest_point_local_m),
        )
        expected_normal = quat_rotate(orientation, local.outward_normal_local)
        self.assertAlmostEqual(result.signed_distance_m, local.signed_distance_m)
        np.testing.assert_allclose(
            result.closest_point_on_foot_world_m, expected_closest, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result.normal_foot_to_ball_world, expected_normal, atol=1.0e-12
        )

    def test_inside_query_has_negative_distance_and_outward_normal(self) -> None:
        point = tuple(float(value) for value in self.proxy.interior_point_m)
        query = self.proxy.query_local(point)
        self.assertLess(query.signed_distance_m, 0.0)
        displacement_to_surface = np.asarray(query.closest_point_local_m) - point
        self.assertGreater(
            float(displacement_to_surface @ np.asarray(query.outward_normal_local)),
            0.0,
        )

    def test_invalid_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.proxy.query_local((float("nan"), 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

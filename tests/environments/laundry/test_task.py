from __future__ import annotations

import unittest

import numpy as np

from laundry.mesh import build_rectangular_mesh
from laundry.task import HalfFoldEvaluator


class HalfFoldTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = build_rectangular_mesh(
            length_m=0.6,
            width_m=0.4,
            nx=12,
            ny=8,
            areal_density_kg_m2=0.3,
            origin_m=(0.0, 0.0, 0.001),
        )
        self.evaluator = HalfFoldEvaluator(self.mesh, thickness_m=0.0008)

    def _evaluate(self, positions, velocities=None, **overrides):
        kwargs = {
            "maximum_strain": 0.01,
            "invalid_intersections": 0,
            "invalid_pose_writes": 0,
            "released": True,
            "settled": True,
        }
        kwargs.update(overrides)
        return self.evaluator.evaluate(
            positions,
            np.zeros_like(positions) if velocities is None else velocities,
            **kwargs,
        )

    def test_registered_perfect_fold_passes(self) -> None:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        result = self._evaluate(positions)
        self.assertTrue(result.success, result.failed_gates)
        self.assertEqual(result.metrics.invalid_pose_writes, 0)

    def test_flat_unfolded_towel_fails_geometry(self) -> None:
        result = self._evaluate(self.mesh.positions_m.copy())
        self.assertFalse(result.success)
        self.assertIn("edge_alignment_rms", result.failed_gates)
        self.assertIn("footprint", result.failed_gates)

    def test_small_rigid_workspace_slip_is_not_fold_shape_error(self) -> None:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        positions[:, :2] += np.asarray((0.020, 0.010))
        result = self._evaluate(positions)
        self.assertTrue(result.success, result.failed_gates)
        self.assertAlmostEqual(result.metrics.fold_line_shape_rms_m, 0.0)
        self.assertAlmostEqual(
            result.metrics.fold_line_translation_m,
            np.hypot(0.020, 0.010),
        )

    def test_large_workspace_slip_remains_a_failure(self) -> None:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        positions[:, 0] += 0.040
        result = self._evaluate(positions)
        self.assertFalse(result.success)
        self.assertIn("fold_line_workspace", result.failed_gates)

    def test_good_pose_cannot_pass_while_held_or_moving(self) -> None:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        velocities = np.zeros_like(positions)
        velocities[0, 2] = 0.1
        result = self._evaluate(positions, velocities, released=False)
        self.assertFalse(result.success)
        self.assertIn("settled_speed", result.failed_gates)
        self.assertIn("released", result.failed_gates)

    def test_pose_write_and_intersection_are_hard_failures(self) -> None:
        positions = self.mesh.positions_m.copy()
        left = self.mesh.material_coordinates_m[:, 0] < 0.0
        positions[left, 0] *= -1.0
        positions[left, 2] += 0.0008
        result = self._evaluate(positions, invalid_intersections=1, invalid_pose_writes=1)
        self.assertFalse(result.success)
        self.assertIn("intersection", result.failed_gates)
        self.assertIn("pose_writes", result.failed_gates)


if __name__ == "__main__":
    unittest.main(verbosity=2)

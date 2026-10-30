from __future__ import annotations

import unittest

import numpy as np

from football.operational_mass import (
    directional_effective_mass,
    effective_mass_from_point_response,
    point_jacobian,
    point_inverse_mass_response,
    rigid_body_point_inverse_mass_response,
)


class OperationalMassTests(unittest.TestCase):
    def test_point_shift_includes_angular_velocity(self) -> None:
        body = np.zeros((6, 1), dtype=np.float64)
        body[4, 0] = 2.0
        shifted = point_jacobian(body, (0.0, 0.0, -0.5))
        self.assertAlmostEqual(shifted[0, 0], -1.0)

    def test_directional_mass_and_support_projection(self) -> None:
        mass = np.diag([2.0, 4.0])
        point = np.asarray([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        free = directional_effective_mass(mass, point, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(free.free_kg, 4.0 / 3.0)
        constrained = directional_effective_mass(
            mass,
            point,
            (1.0, 0.0, 0.0),
            constraint_jacobian=np.asarray([[1.0, 0.0]]),
        )
        self.assertAlmostEqual(constrained.constrained_kg, 4.0)

    def test_tensor_preserves_cross_axis_point_response(self) -> None:
        mass = np.diag([2.0, 4.0, 8.0])
        point = np.asarray(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        )
        response = point_inverse_mass_response(mass, point)
        expected = point @ np.linalg.inv(mass) @ point.T
        np.testing.assert_allclose(response, expected, atol=1.0e-12)
        self.assertGreater(abs(float(response[0, 1])), 0.0)

    def test_rigid_sphere_point_response_includes_rotation(self) -> None:
        mass = 0.43
        radius = 0.11
        inertia = 2.0 * mass * radius**2 / 3.0
        response = rigid_body_point_inverse_mass_response(
            mass,
            np.eye(3) * inertia,
            (-radius, 0.0, 0.0),
        )
        self.assertAlmostEqual(response[0, 0], 1.0 / mass)
        self.assertAlmostEqual(
            response[1, 1], 1.0 / mass + radius**2 / inertia
        )
        self.assertAlmostEqual(
            effective_mass_from_point_response(response, (1.0, 0.0, 0.0)),
            mass,
        )


if __name__ == "__main__":
    unittest.main()

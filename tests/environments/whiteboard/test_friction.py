import unittest
import numpy as np
from whiteboard.runtime import require_newton


class FrictionTests(unittest.TestCase):
    def test_implicit_many_contact_newton_roundoff(self):
        from whiteboard.friction import implicit_regularized

        rng = np.random.default_rng(69231)
        M = np.diag([0.03, 0.03, 0.03, 1e-5, 3e-5, 4e-5])
        directions = np.broadcast_to([[1.0, 0, 0], [0, 0, 1.0]], (50, 2, 3))
        for _ in range(40):
            points = rng.uniform(-0.05, 0.05, (50, 3))
            J = np.concatenate((directions, np.cross(points[:, None], directions)), axis=2)
            forces, quality = implicit_regularized(
                rng.normal(0, 0.2, 6), M, J, rng.uniform(0, 0.05, 50), 0.0005
            )
            self.assertLess(quality["momentum_residual_scaled"], 1e-12)
            self.assertLessEqual(quality["power_w"], 0)

    def test_implicit_board_friction_preserves_law_without_explicit_chatter(self):
        from whiteboard.friction import implicit_regularized

        mass = np.diag([0.03, 0.03, 0.03, 0.001, 0.001, 0.001])
        jac = np.array([[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]], float)
        velocity = np.zeros(6)
        dt, limit = 0.0005, 0.55 * 1.4
        for _ in range(200):
            free = velocity.copy()
            free[2] -= 9.81 * dt
            forces, quality = implicit_regularized(free, mass, jac, [limit], dt)
            velocity = free + dt * np.linalg.solve(mass, np.einsum("nij,ni->j", jac, forces))
            self.assertLessEqual(quality["power_w"], 1e-14)
            self.assertLessEqual(np.linalg.norm(forces), limit + 1e-12)
            self.assertLess(quality["momentum_residual_scaled"], 1e-12)
        weight = 0.03 * 9.81
        expected = -0.001 * weight / np.sqrt(limit**2 - weight**2)
        self.assertAlmostEqual(velocity[2], expected, places=10)
        self.assertLess(abs(velocity[2]) * 3, 0.003)

    def test_compiled_sweep_matches_reference(self):
        require_newton()
        from whiteboard.friction import solve, reference

        rng = np.random.default_rng(418)
        for count in (1, 3, 12):
            for _ in range(5):
                initial = rng.normal(size=12)
                J = rng.normal(size=(count, 2, 12))
                A = rng.normal(size=(12, 12))
                mass = A @ A.T + np.eye(12)
                E = J @ mass @ J.transpose(0, 2, 1)
                inverse = np.linalg.inv(E)
                response = mass @ J.transpose(0, 2, 1)
                limits = np.full(count, 0.01)
                expected, ed = reference(initial, J, E, inverse, response, limits)
                actual, ad = solve(initial, J, E, inverse, response, limits)
                np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)
                np.testing.assert_allclose(ad, ed, atol=1e-12, rtol=1e-12)
                self.assertLessEqual(ad[0], 1e-12)
                self.assertLessEqual(np.max(np.linalg.norm(actual, axis=1)), 0.01 + 1e-12)


if __name__ == "__main__":
    unittest.main()

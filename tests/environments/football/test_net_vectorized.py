from __future__ import annotations

import unittest

import numpy as np

from football.net_model import CableNetPatch, CableNetPrior
from football.net_vectorized import VectorizedCableNetPatch


class VectorizedNetTests(unittest.TestCase):
    def test_one_step_matches_scalar_reference_with_ball_contact(self) -> None:
        prior = CableNetPrior(mesh_spacing_m=0.22)
        scalar = CableNetPatch(width_m=1.32, height_m=1.32, prior=prior)
        vector_source = CableNetPatch(width_m=1.32, height_m=1.32, prior=prior)
        vector = VectorizedCableNetPatch(vector_source)
        arguments = {
            "ball_center_world_m": (-0.05, 0.0, 0.66),
            "ball_velocity_world_m_s": (5.0, 0.3, -0.2),
            "ball_angular_velocity_world_rad_s": (1.0, -2.0, 3.0),
            "dt_s": 1.0 / 24_000.0,
        }
        expected = scalar.step(**arguments)
        actual = vector.step(**arguments)
        vector.sync_nodes()
        for name in (
            "force_on_ball_world_n",
            "torque_on_ball_world_n_m",
            "anchor_force_world_n",
            "anchor_torque_world_n_m",
        ):
            np.testing.assert_allclose(
                getattr(actual, name), getattr(expected, name), rtol=1.0e-12, atol=1.0e-12
            )
        self.assertEqual(actual.contact_count, expected.contact_count)
        self.assertAlmostEqual(
            actual.maximum_contact_force_n, expected.maximum_contact_force_n, places=10
        )
        np.testing.assert_allclose(
            [node.position_m for node in vector.nodes],
            [node.position_m for node in scalar.nodes],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            [node.velocity_m_s for node in vector.nodes],
            [node.velocity_m_s for node in scalar.nodes],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()

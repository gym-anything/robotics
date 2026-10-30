from __future__ import annotations

import unittest

import numpy as np

from football.foot_ball_coupler import FootBallCoupler
from football.foot_contact import FootPatchSpec, sphere_obb_contact
from football.football_specs import BallSpec


class _FakeRigid:
    def __init__(self, position, velocity) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self._rigid_prim_view = self
        self.applied: list[tuple[np.ndarray, np.ndarray]] = []

    def get_world_pose(self):
        return self.position.copy(), np.asarray([1.0, 0.0, 0.0, 0.0])

    def get_linear_velocity(self):
        return self.velocity.copy()

    def get_angular_velocity(self):
        return self.omega.copy()

    def apply_forces_and_torques_at_pos(self, *, forces, torques, is_global):
        self.applied.append(
            (
                np.asarray(forces[0], dtype=np.float64),
                np.asarray(torques[0], dtype=np.float64),
            )
        )


class FootBallCouplerTests(unittest.TestCase):
    def test_internal_release_requires_live_geometric_separation(self) -> None:
        ball = BallSpec()
        patch = FootPatchSpec()
        ball_view = _FakeRigid(
            (patch.maximum_local_m[0] + ball.radius_m - 0.002, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        foot_view = _FakeRigid((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
        coupler = FootBallCoupler(
            ball_view=ball_view,
            foot_view=foot_view,
            foot_center_of_mass_local_m=(0.0, 0.0, 0.0),
            ball=ball,
            patch=patch,
            foot_contact_geometry_provider=sphere_obb_contact,
        )
        coupler.awaiting_geometric_separation = True
        suppressed = coupler.step(0.0, 1.0 / 500.0)
        self.assertEqual(suppressed["normal_force_n"], 0.0)
        self.assertEqual(len(ball_view.applied), 0)
        self.assertEqual(coupler.release_hysteresis_steps, 1)

        ball_view.position[0] = patch.maximum_local_m[0] + ball.radius_m + 0.10
        separated = coupler.step(0.002, 1.0 / 500.0)
        self.assertEqual(separated["normal_force_n"], 0.0)
        self.assertFalse(coupler.awaiting_geometric_separation)

        ball_view.position[0] = patch.maximum_local_m[0] + ball.radius_m - 0.002
        engaged = coupler.step(0.004, 1.0 / 500.0)
        self.assertGreater(engaged["normal_force_n"], 0.0)
        self.assertEqual(len(ball_view.applied), 1)

    def test_overlap_growing_past_stale_remainder_reengages(self) -> None:
        # The release latch must not suppress an overlap that grows past the
        # stale remainder, or the ball passes through the shoe at 0 N.
        ball = BallSpec()
        patch = FootPatchSpec()
        touching = patch.maximum_local_m[0] + ball.radius_m
        ball_view = _FakeRigid((touching - 0.002, 0.0, 0.0), (0.0, 0.0, 0.0))
        foot_view = _FakeRigid((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        coupler = FootBallCoupler(
            ball_view=ball_view,
            foot_view=foot_view,
            foot_center_of_mass_local_m=(0.0, 0.0, 0.0),
            ball=ball,
            patch=patch,
            foot_contact_geometry_provider=sphere_obb_contact,
        )
        coupler.awaiting_geometric_separation = True
        self.assertEqual(coupler.step(0.0, 0.002)["normal_force_n"], 0.0)
        ball_view.position[0] = touching - 0.0015
        self.assertEqual(coupler.step(0.002, 0.002)["normal_force_n"], 0.0)
        ball_view.position[0] = touching - 0.012
        ball_view.velocity[0] = -1.0
        engaged = coupler.step(0.004, 0.002)
        self.assertTrue(engaged["engaged"])
        self.assertGreater(
            engaged["normal_force_n"],
            0.5 * coupler.model.elastic_normal_force(0.012 - 0.0015)[0],
        )
        report = coupler.report()["partitioned_release_hysteresis"]
        self.assertEqual(report["suppressed_stale_overlap_steps"], 2)
        self.assertEqual(report["reengagements_beyond_stale_overlap"], 1)

    def test_sustained_press_loads_ball_like_the_live_overlap(self) -> None:
        # A slow press of a supported ball must not leave the internal
        # compression near zero while the shoe sinks into the ball.
        ball = BallSpec()
        patch = FootPatchSpec()
        touching = patch.maximum_local_m[0] + ball.radius_m
        ball_view = _FakeRigid((touching, 0.0, 0.0), (0.0, 0.0, 0.0))
        speed = 0.05
        foot_view = _FakeRigid((0.0, 0.0, 0.0), (speed, 0.0, 0.0))
        coupler = FootBallCoupler(
            ball_view=ball_view,
            foot_view=foot_view,
            foot_center_of_mass_local_m=(0.0, 0.0, 0.0),
            ball=ball,
            patch=patch,
            foot_contact_geometry_provider=sphere_obb_contact,
        )
        dt = 0.002
        for step in range(300):
            sample = coupler.step(step * dt, dt)
            overlap = sample["geometry"]["compression_m"]
            if overlap > 0.002:
                law = coupler.model.elastic_normal_force(overlap)[0]
                self.assertGreater(sample["normal_force_n"], 0.8 * law)
                self.assertLess(sample["normal_force_n"], 1.5 * law)
            # The ball rests on support and an actuator holds the shoe speed:
            # neither body is moved by the shoe force.
            foot_view.position[0] += dt * speed
        self.assertGreater(overlap, 0.025)
        self.assertEqual(
            coupler.report()["partitioned_release_hysteresis"][
                "suppressed_stale_overlap_steps"
            ],
            0,
        )

    def test_live_tensor_provider_controls_full_point_response(self) -> None:
        ball = BallSpec()
        patch = FootPatchSpec()
        compression_m = 0.002
        ball_view = _FakeRigid(
            (
                patch.maximum_local_m[0] + ball.radius_m - compression_m,
                0.0,
                0.0,
            ),
            (0.0, 0.0, 0.0),
        )
        foot_view = _FakeRigid((0.0, 0.0, 0.0), (3.0, 0.5, 0.0))
        provider_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

        def provider(point, normal):
            provider_calls.append((point, normal))
            return {
                "point_inverse_mass_world_per_kg": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.25],
                ],
                "source": "unit-test live tensor",
                "evidence_status": "TEST",
            }

        coupler = FootBallCoupler(
            ball_view=ball_view,
            foot_view=foot_view,
            foot_center_of_mass_local_m=(0.0, 0.0, 0.0),
            ball=ball,
            patch=patch,
            foot_inverse_mass_provider=provider,
            # This test isolates the tensor/contact-law plumbing with an
            # analytically positioned box coupon.  Production uses the pinned
            # convex visible-shoe proxy.
            foot_contact_geometry_provider=sphere_obb_contact,
        )
        sample = coupler.step(0.0, 1.0 / 500.0)
        response = sample["point_inverse_mass_response"]
        self.assertEqual(len(provider_calls), 1)
        self.assertIsNotNone(response)
        self.assertAlmostEqual(
            response["foot_directional_effective_mass_kg"], 1.0
        )
        expected_reduced_mass = 1.0 / (1.0 + 1.0 / ball.mass_kg)
        self.assertAlmostEqual(
            response["combined_directional_reduced_mass_kg"],
            expected_reduced_mass,
        )
        # At a surface point, rotation leaves the radial response unchanged
        # but makes the hollow sphere much easier to accelerate tangentially.
        ball_response = np.asarray(
            response["ball_point_inverse_mass_world_per_kg"]
        )
        self.assertGreater(ball_response[1, 1], ball_response[0, 0])
        self.assertGreater(sample["normal_force_n"], 0.0)
        self.assertEqual(len(ball_view.applied), 1)
        self.assertEqual(len(foot_view.applied), 1)
        report = coupler.report()
        self.assertTrue(report["point_response_model"]["runtime_foot_provider"])
        self.assertEqual(report["operational_response_evaluations"], 1)
        energy = report["subcycle_energy_audit"]
        self.assertLess(
            energy["maximum_work_vs_relative_kinetic_closure_error_j"],
            1.0e-12,
        )
        self.assertLess(
            abs(energy["constitutive_accounting_residual_j"]),
            1.0e-12,
        )
        self.assertGreater(energy["constitutive_dissipation_j"], 0.0)
        self.assertEqual(
            energy["summed_negative_increment_energy_creation_j"], 0.0
        )


if __name__ == "__main__":
    unittest.main()

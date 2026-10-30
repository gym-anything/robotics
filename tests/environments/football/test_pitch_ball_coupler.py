from __future__ import annotations

import unittest

import numpy as np

from football.football_specs import BallSpec
from football.pitch_ball_coupler import PitchBallCoupler


class _FakeBall:
    def __init__(self, ball: BallSpec, height_m: float) -> None:
        self.ball = ball
        self.position = np.asarray([0.0, 0.0, ball.radius_m + height_m])
        self.velocity = np.zeros(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self._rigid_prim_view = self
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)

    def get_world_pose(self):
        return self.position.copy(), np.asarray([1.0, 0.0, 0.0, 0.0])

    def get_linear_velocity(self):
        return self.velocity.copy()

    def get_angular_velocity(self):
        return self.omega.copy()

    def apply_forces_and_torques_at_pos(self, *, forces, torques, is_global):
        self.force += np.asarray(forces[0], dtype=np.float64)
        self.torque += np.asarray(torques[0], dtype=np.float64)

    def integrate_physx_surrogate(self, dt_s: float, unseen_force=(0.0, 0.0, 0.0)) -> None:
        # ``unseen_force`` stands for native contacts (e.g. a robot arm) that
        # act on the live ball but are not passed to the pitch predictor.
        acceleration = (self.force + np.asarray(unseen_force)) / self.ball.mass_kg
        acceleration[2] -= 9.80665
        self.velocity += dt_s * acceleration
        self.omega += dt_s * self.torque / self.ball.thin_shell_inertia_kg_m2
        self.position += dt_s * self.velocity
        self.force.fill(0.0)
        self.torque.fill(0.0)


class PitchBallCouplerTests(unittest.TestCase):
    def test_two_way_wrench_closes_and_drop_rebounds(self) -> None:
        ball = BallSpec()
        view = _FakeBall(ball, 2.0)
        coupler = PitchBallCoupler(
            ball_view=view, ball=ball, contact_substep_hz=20_000.0
        )
        dt = 1.0 / 500.0
        contact_seen = False
        rebound_speed = 0.0
        for step in range(800):
            sample = coupler.step(step * dt, dt)
            contact_seen = contact_seen or sample["normal_force_n"] > 0.0
            view.integrate_physx_surrogate(dt)
            if contact_seen:
                rebound_speed = max(rebound_speed, float(view.velocity[2]))
        report = coupler.report()
        self.assertTrue(contact_seen)
        self.assertGreater(rebound_speed, 3.0)
        self.assertLess(report["maximum_force_closure_error_n"], 1.0e-10)
        self.assertLess(
            report["maximum_origin_moment_closure_error_n_m"], 1.0e-9
        )

    def test_quasistatic_support_does_not_accumulate_shadow_drift(self) -> None:
        ball = BallSpec()
        view = _FakeBall(ball, 0.001)
        coupler = PitchBallCoupler(
            ball_view=view, ball=ball, contact_substep_hz=20_000.0
        )
        dt = 1.0 / 500.0
        maximum_late_vertical_speed = 0.0
        for step in range(1_000):
            coupler.step(step * dt, dt)
            view.integrate_physx_surrogate(dt)
            if step >= 900:
                maximum_late_vertical_speed = max(
                    maximum_late_vertical_speed, abs(float(view.velocity[2]))
                )
        compression = ball.radius_m - float(view.position[2])
        report = coupler.report()
        self.assertAlmostEqual(
            compression,
            report["static_equilibrium_compression_m"],
            delta=2.0e-5,
        )
        self.assertLess(maximum_late_vertical_speed, 1.0e-4)
        self.assertGreater(report["quasistatic_outer_contact_steps"], 0)

    def test_unseen_downward_push_is_resisted_and_releases_no_energy(self) -> None:
        # A robot arm pushes the ball into the turf through
        # native contact, which the predictor does not see.  The turf must
        # resist the real indentation, and the ball must not leave with more
        # energy than the push put in.
        ball = BallSpec()
        view = _FakeBall(ball, 0.0005)
        view.velocity[2] = -1.2
        coupler = PitchBallCoupler(
            ball_view=view, ball=ball, contact_substep_hz=20_000.0
        )
        dt = 1.0 / 500.0
        push = np.asarray([0.0, 0.0, -200.0])
        push_work = 0.0
        deepest = 0.0
        for step in range(400):
            coupler.step(step * dt, dt)
            unseen = push if step < 8 else np.zeros(3)
            before = view.position[2]
            view.integrate_physx_surrogate(dt, unseen)
            push_work += float(unseen[2] * (view.position[2] - before))
            deepest = max(deepest, ball.radius_m - float(view.position[2]))
        # Static indentation under 200 N plus the ball's weight is ~9 mm.
        self.assertLess(deepest, 0.03)
        final_energy = 0.5 * ball.mass_kg * float(view.velocity @ view.velocity)
        final_energy += ball.mass_kg * 9.80665 * (
            float(view.position[2]) - (ball.radius_m + 0.0005)
        )
        initial_energy = 0.5 * ball.mass_kg * 1.2**2
        self.assertLess(final_energy, initial_energy + push_work)
        self.assertGreater(
            coupler.report()["maximum_unmodelled_velocity_change_m_s"], 0.5
        )


if __name__ == "__main__":
    unittest.main()

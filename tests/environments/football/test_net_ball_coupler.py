from __future__ import annotations

import unittest

import numpy as np

from football.net_ball_coupler import NetBallCoupler
from football.net_model import CableNetPatch, CableNetPrior


class _FakeRigidView:
    def __init__(self) -> None:
        self.position = np.asarray([-0.05, 0.0, 0.66], dtype=np.float64)
        self.velocity = np.asarray([5.0, 0.0, 0.0], dtype=np.float64)
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
        self.applied.append((np.asarray(forces), np.asarray(torques)))


class NetBallCouplerTests(unittest.TestCase):
    def test_overlapping_ball_receives_only_a_resisting_wrench(self) -> None:
        view = _FakeRigidView()
        net = CableNetPatch(
            width_m=1.32,
            height_m=1.32,
            prior=CableNetPrior(mesh_spacing_m=0.22),
        )
        coupler = NetBallCoupler(
            ball_view=view,
            net=net,
            contact_substep_hz=24_000.0,
        )
        sample = coupler.step(0.0, 1.0 / 500.0)
        self.assertTrue(sample["activated"])
        self.assertGreater(sample["contact_count"], 0)
        self.assertLess(sample["force_on_ball_world_n"][0], 0.0)
        self.assertEqual(len(view.applied), 1)
        self.assertGreater(coupler.report()["contact_substeps"], 0)

    def test_far_ball_does_not_advance_or_apply_the_net(self) -> None:
        view = _FakeRigidView()
        view.position[0] = -5.0
        coupler = NetBallCoupler(
            ball_view=view,
            net=CableNetPatch(width_m=1.32, height_m=1.32),
        )
        sample = coupler.step(0.0, 1.0 / 500.0)
        self.assertFalse(sample["activated"])
        self.assertEqual(coupler.report()["solver_substeps"], 0)
        self.assertFalse(view.applied)


if __name__ == "__main__":
    unittest.main()

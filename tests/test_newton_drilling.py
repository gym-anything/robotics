"""Drilling backend contract tests (in-process Newton build of bench/press-drill, no cameras; ~1 min).

The first one checks that the policy filter accepts the per-arm external-torque estimate as the 7-vector the robot
reports, so a policy observation (GET /v1/observation with a policy key, and the observation returned after every
policy action) passes the filter."""

from __future__ import annotations

import tempfile
import unittest

try:
    import newton  # noqa: F401
    HAVE_NEWTON = True
except Exception:  # pragma: no cover - the engine venv is required for these tests
    HAVE_NEWTON = False


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class NewtonDrillingBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        cls.tmp = tempfile.TemporaryDirectory()
        cls.backend = NewtonDrillingBackend(robot="dual-fr3", preset="bench", task="press-drill", seed=0, cameras=(),
                                            output_dir=cls.tmp.name + "/session", recording=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()
        cls.tmp.cleanup()

    def test_policy_observation_passes_the_policy_filter(self) -> None:
        b = self.backend
        obs = b.observation_from_state(b.get_state(), sequence=0)
        filtered = b.policy_access().observation(obs)
        for arm in b.arms:
            status = filtered.proprioception["robot_status"]["arms"][arm]
            self.assertEqual(len(status["tau_ext_nm"]), 7)
            self.assertIsInstance(status["gripper_width_m"], float)
        self.assertEqual(filtered.metadata, {})
        self.assertEqual(set(filtered.images), set())

    def test_capabilities(self) -> None:
        caps = self.backend.capabilities
        self.assertTrue(caps.seeded_reset)            # reset with a seed rebuilds that catalogue episode
        self.assertTrue(caps.privileged_state)
        self.assertIn("cartesian_wrench", caps.action_types)
        self.assertNotIn("joint_effort", caps.action_types)

    def test_refused_commands_leave_the_session_usable(self) -> None:
        from gym_anything_robotics.actions import CartesianWrench, Gripper, NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.backend
        tick = b.env.world.tick
        with self.assertRaises(ActionRejectedError):
            b.act(NamedJointTarget({"right/fr3_joint1": 9.0}, 10))
        with self.assertRaises(ActionRejectedError):
            b.act(CartesianWrench("right", (200., 0., 0.), (0., 0., 0.), 0.1))
        with self.assertRaises(ActionRejectedError):
            b.act(Gripper("right", None, 80., 0.1))
        self.assertEqual(b.env.world.tick, tick)       # nothing ran
        step = b.act(NamedJointTarget({}, 20))
        self.assertEqual(step.info["executed_ticks"], 20)
        self.assertIsNone(b.env.world.fault)


if __name__ == "__main__":
    unittest.main()

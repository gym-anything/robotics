"""CPU integration tests, deliberately independent of task-solving ability."""

import unittest
import numpy as np
from whiteboard.config import Config
from whiteboard.env import WhiteboardEnv


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WhiteboardEnv(
            Config(robot=False, furniture=False, texture_width=512, texture_height=256)
        )

    def setUp(self):
        self.env.reset()

    def test_read_does_not_advance(self):
        before = self.env.state_0.body_q.numpy().copy()
        self.env.get_state()
        self.env.proprioception()
        np.testing.assert_array_equal(before, self.env.state_0.body_q.numpy())
        self.assertEqual(self.env.tick, 0)

    def test_invalid_action_is_not_applied(self):
        with self.assertRaises(ValueError):
            self.env.act({"made_up_joint": 0}, ticks=1)
        self.assertEqual(self.env.tick, 0)

    def test_fixture_intervention_forbidden_in_policy_mode(self):
        with self.assertRaises(PermissionError):
            self.env.fixture_pose("marker_red", (0, 0, 1))
        self.assertEqual(self.env.tick, 0)

    def test_snapshot_restores_ink_and_solver_state(self):
        self.env.step(25)
        snapshot = self.env.snapshot()
        self.env.step(20)
        expected = self.env.state_0.body_q.numpy().copy()
        self.env.restore(snapshot)
        self.env.step(20)
        np.testing.assert_allclose(expected, self.env.state_0.body_q.numpy(), atol=1e-7, rtol=0)

    def test_static_room_dynamic_tools(self):
        self.assertNotIn("architecture", self.env.body_ids)
        self.assertNotIn("board", self.env.body_ids)
        self.assertEqual(len(self.env.body_ids), 13)
        self.assertTrue(np.all(self.env.model.body_mass.numpy() > 0))

    def test_blank_board_is_not_success(self):
        result = self.env.evaluate()
        self.assertFalse(result["success"])
        self.assertEqual(result["score"], 0)


if __name__ == "__main__":
    unittest.main()

"""Action validation must not mutate a CPU-backed target array before commit."""

import threading
from types import SimpleNamespace
import unittest

import numpy as np

from whiteboard.env import WhiteboardEnv


class NativeArray:
    def __init__(self):
        self.value = np.zeros(2, np.float32)

    def numpy(self):
        return self.value

    def assign(self, value):
        self.value[:] = value


class ActionPrevalidationTests(unittest.TestCase):
    def setUp(self):
        self.target = NativeArray()
        self.env = SimpleNamespace(
            _lock=threading.RLock(),
            healthy=True,
            control=SimpleNamespace(joint_target_q=self.target),
            robot_joints={"joint": {"q": 0, "lower": -1.0, "upper": 1.0}},
            config=SimpleNamespace(episode_ticks=10),
            tick=0,
            action_id=0,
        )

    def rejected(self, targets, ticks):
        with self.assertRaises(ValueError):
            WhiteboardEnv.act(self.env, targets, ticks=ticks)
        np.testing.assert_array_equal(self.target.value, [0, 0])
        self.assertEqual(self.env.tick, 0)
        self.assertEqual(self.env.action_id, 0)

    def test_later_unknown_joint_does_not_partially_set_targets(self):
        self.rejected({"joint": 0.4, "unknown": 0.0}, 1)

    def test_invalid_duration_does_not_set_valid_target(self):
        self.rejected({"joint": 0.4}, 0)

    def test_episode_overrun_does_not_set_valid_target(self):
        self.rejected({"joint": 0.4}, 11)

    def test_valid_target_commits_before_step(self):
        self.env.step = lambda ticks: (ticks, self.target.value.copy())
        ticks, applied = WhiteboardEnv.act(self.env, {"joint": 0.4}, ticks=3)
        self.assertEqual(ticks, 3)
        np.testing.assert_allclose(applied, [0.4, 0])
        self.assertEqual(self.env.action_id, 1)

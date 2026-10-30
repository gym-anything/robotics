"""Action types from different environments decode side by side, without starting physics."""
import dataclasses
import unittest

from gym_anything_robotics.action_codec import action_from_dict
from gym_anything_robotics.actions import BaseTwist, ToolIO


class ConsolidatedRegistryTests(unittest.TestCase):

    def test_mobile_base_and_tool_io_actions_coexist(self):
        base = action_from_dict(dict(type='base_twist', vx_m_s=0.1, vy_m_s=0.0,
                                     wz_rad_s=0.2, duration_s=0.5))
        tool = action_from_dict(dict(type='tool_io', arm='right', outputs={'arc_on': False}, ticks=1))
        self.assertIsInstance(base, BaseTwist)
        self.assertIsInstance(tool, ToolIO)
        self.assertEqual(base.vx_m_s, 0.1)
        for action in (base, tool):
            self.assertTrue(dataclasses.is_dataclass(action))
            with self.assertRaises(dataclasses.FrozenInstanceError):
                action.unexpected_field = True

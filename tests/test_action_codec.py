from __future__ import annotations

import unittest

from gym_anything_robotics.action_codec import action_from_dict, actions_from_request
from gym_anything_robotics.actions import (
    ActuatorPosition,
    BimanualCartesian,
    BimanualJointPosition,
    CartesianDelta,
    CartesianPose,
    CartesianWrench,
    Gripper,
    JointEffort,
    JointPosition,
    Keyboard,
    NamedJointTarget,
    RawAction,
    ToolIO,
)


class ActionCodecTest(unittest.TestCase):
    def test_mixed_joint_units_and_exact_tick_count(self):
        payload = {"type": "named_joint_target", "targets_si": {
            "panda_joint1": -0.25, "panda_finger_joint1": 0.025}, "ticks": 512}
        result = action_from_dict(payload)
        self.assertIsInstance(result, NamedJointTarget)
        self.assertEqual(dict(result.targets_si), payload["targets_si"])
        self.assertEqual(result.ticks, 512)
        self.assertEqual(action_from_dict({"type": "named_joint_target",
            "targets_si": {}, "ticks": 1024}).ticks, 1024)
        for value in (True, 0, -1, 1.5):
            with self.subTest(ticks=value), self.assertRaises(ValueError):
                action_from_dict({**payload, "ticks": value})
        for value in (True, float("nan"), float("inf")):
            with self.subTest(target=value), self.assertRaises(ValueError):
                action_from_dict({**payload, "targets_si": {"panda_joint1": value}})

    def test_tool_io_outputs_and_ticks(self) -> None:
        result = action_from_dict({"type": "tool_io", "arm": "right",
                                   "outputs": {"arc_on": True, "wire_feed_m_min": 12.7, "process": "spray"},
                                   "ticks": 5})
        self.assertIsInstance(result, ToolIO)
        self.assertEqual(result.arm, "right")
        self.assertEqual(result.ticks, 5)
        self.assertEqual(dict(result.outputs)["wire_feed_m_min"], 12.7)
        self.assertEqual(action_from_dict({"type": "tool_io", "arm": "left", "outputs": {"gas_on": False}}).ticks, 1)
        bad = (
            {"type": "tool_io", "arm": "right", "outputs": {}},
            {"type": "tool_io", "arm": "right", "outputs": {"voltage_v": float("nan")}},
            {"type": "tool_io", "arm": "right", "outputs": {"voltage_v": [29]}},
            {"type": "tool_io", "arm": "middle", "outputs": {"arc_on": True}},
            {"type": "tool_io", "arm": "right", "outputs": {"arc_on": True}, "ticks": 0},
            {"type": "tool_io", "arm": "right", "outputs": {"arc_on": True}, "ticks": True},
            {"type": "tool_io", "arm": "right", "outputs": {"arc_on": True}, "extra": 1},
        )
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                action_from_dict(payload)

    def test_every_public_action_type_decodes(self) -> None:
        six = [0.0] * 6
        cases = (
            (
                {"type": "cartesian_pose", "arm": "left", "position_m": [1, 2, 3]},
                CartesianPose,
            ),
            (
                {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, 0.1]},
                CartesianDelta,
            ),
            (
                {
                    "type": "bimanual_cartesian",
                    "left": {"delta_m": [0, 0, 0.1]},
                    "right": {"position_m": [0.6, 0.1, 0.8]},
                },
                BimanualCartesian,
            ),
            (
                {"type": "joint_position", "arm": "left", "positions_rad": six},
                JointPosition,
            ),
            (
                {
                    "type": "bimanual_joint_position",
                    "left_positions_rad": six,
                    "right_positions_rad": six,
                },
                BimanualJointPosition,
            ),
            (
                {"type": "actuator_position", "arm": "right", "positions_rad": six},
                ActuatorPosition,
            ),
            (
                {"type": "gripper", "arm": "left", "inward_effort_n": 3.0},
                Gripper,
            ),
            (
                {"type": "keyboard", "arm": "right", "keys": ["D", "H"], "ticks": 5},
                Keyboard,
            ),
            (
                {
                    "type": "joint_effort",
                    "arm": "left",
                    "efforts_nm": six,
                    "duration_s": 0.1,
                },
                JointEffort,
            ),
            (
                {
                    "type": "cartesian_wrench",
                    "arm": "right",
                    "force_n": [1, 0, 0],
                    "moment_nm": [0, 0, 0],
                    "duration_s": 0.1,
                },
                CartesianWrench,
            ),
            ({"type": "raw", "request": {"op": "policy_next"}}, RawAction),
        )
        for payload, expected_type in cases:
            with self.subTest(expected_type=expected_type.__name__):
                self.assertIsInstance(action_from_dict(payload), expected_type)

    def test_ordered_action_array_decodes_without_rewriting_values(self) -> None:
        actions = actions_from_request(
            {
                "actions": [
                    {"type": "keyboard", "keys": ["D"], "ticks": 7, "rate_hz": 20},
                    {"type": "gripper", "arm": "left", "half_width_m": 0.012},
                ]
            }
        )

        self.assertEqual(actions[0].ticks, 7)
        self.assertEqual(actions[0].rate_hz, 20.0)
        self.assertEqual(actions[1].half_width_m, 0.012)

    def test_unknown_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            action_from_dict(
                {
                    "type": "cartesian_delta",
                    "arm": "right",
                    "delta_m": [0, 0, 0.1],
                    "ignored": "must not be ignored",
                }
            )


if __name__ == "__main__":
    unittest.main()

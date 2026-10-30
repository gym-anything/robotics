from __future__ import annotations

import unittest

import numpy as np

from laundry.control import (
    BIMANUAL_CARTESIAN_ACTION_SCHEMA,
    FIXTURE_VELOCITY_ACTION_SCHEMA,
    make_bimanual_cartesian_action_schema,
)


class ControlSchemaTests(unittest.TestCase):
    def test_semantic_channel_order_and_physical_rates_are_explicit(self) -> None:
        schema = FIXTURE_VELOCITY_ACTION_SCHEMA
        self.assertEqual(schema.shape, (6,))
        self.assertEqual(
            tuple(channel.name for channel in schema.channels),
            (
                "gripper_x",
                "gripper_z",
                "gripper_wrist_y",
                "gripper_gap",
                "roller_x",
                "roller_z",
            ),
        )
        rates = schema.rate_commands(np.ones(6, dtype=np.float32))
        self.assertAlmostEqual(rates["gripper_x"], 0.50)
        self.assertAlmostEqual(rates["gripper_z"], 0.30)
        self.assertAlmostEqual(rates["gripper_wrist_y"], 1.50)
        self.assertAlmostEqual(rates["gripper_gap"], 0.040)
        self.assertAlmostEqual(rates["roller_z"], 0.40)

    def test_invalid_actions_fail_before_a_backend_can_receive_them(self) -> None:
        schema = FIXTURE_VELOCITY_ACTION_SCHEMA
        with self.assertRaises(ValueError):
            schema.validate(np.zeros(5))
        with self.assertRaises(ValueError):
            schema.validate(np.full(6, 1.01))
        with self.assertRaises(ValueError):
            schema.validate(np.full(6, np.nan))
        clipped = schema.validate(np.full(6, 2.0), strict=False)
        np.testing.assert_array_equal(clipped, np.ones(6, dtype=np.float32))

    def test_bimanual_schema_is_symmetric_and_robot_neutral(self) -> None:
        schema = BIMANUAL_CARTESIAN_ACTION_SCHEMA
        self.assertEqual(schema.shape, (14,))
        self.assertEqual(
            tuple(channel.name for channel in schema.channels[:7]),
            (
                "left_vx",
                "left_vy",
                "left_vz",
                "left_wx",
                "left_wy",
                "left_wz",
                "left_jaw",
            ),
        )
        self.assertEqual(
            tuple(channel.name for channel in schema.channels[7:]),
            (
                "right_vx",
                "right_vy",
                "right_vz",
                "right_wx",
                "right_wy",
                "right_wz",
                "right_jaw",
            ),
        )
        self.assertIn("target_rate", schema.command_type)
        self.assertIn("joint-space", schema.calibration_status)

    def test_bimanual_schema_rejects_ambiguous_tool_names(self) -> None:
        with self.assertRaises(ValueError):
            make_bimanual_cartesian_action_schema(tool_names=("arm", "arm"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

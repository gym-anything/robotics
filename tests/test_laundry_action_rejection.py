"""Exercise the actual laundry adapter's pre/post-execution error boundary."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from gym_anything_robotics.actions import CartesianPose, Gripper, MotionSafety, NamedJointTarget, RawAction
from gym_anything_robotics.backends.newton_laundry import NewtonLaundryBackend
from gym_anything_robotics.errors import ActionRejectedError, BackendProtocolError, UnsupportedActionError


class LaundryActionRejectionTest(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(NewtonLaundryBackend)
        self.backend.env = SimpleNamespace(
            lock=threading.RLock(), tick=5184, fault=None,
            config=SimpleNamespace(dt=1/1440), act=Mock(),
        )
        self.backend.ik = SimpleNamespace(solve=Mock(side_effect=ValueError(
            "IK target not reached: 6.50 mm; no action executed"
        )))

    def test_ik_rejection_never_reaches_physics(self):
        with self.assertRaisesRegex(ActionRejectedError, "6.50 mm"):
            self.backend.act(CartesianPose(
                "left", (-.15, .3, .9), (0, .70710678, .70710678, 0),
                duration_s=.7, gripper_half_target_m=.04,
            ))
        self.backend.env.act.assert_not_called()
        self.assertEqual(self.backend.env.tick, 5184)
        self.assertIsNone(self.backend.env.fault)

    def test_duration_and_gripper_validation_are_rejections(self):
        for action in (Gripper("left", half_width_m=.04, duration_s=.001),
                       Gripper("left", half_width_m=.05, duration_s=.5)):
            with self.subTest(action=action), self.assertRaises(ActionRejectedError):
                self.backend.act(action)
        self.backend.env.act.assert_not_called()

    def test_joint_validation_without_mutation_is_rejection(self):
        self.backend.env.act.side_effect = ValueError("Unknown robot joint missing")
        with self.assertRaises(ActionRejectedError):
            self.backend.act(NamedJointTarget({"missing": 0}, ticks=1))
        self.assertIsNone(self.backend.env.fault)

    def test_all_unsupported_action_paths_reject_before_physics(self):
        for action in (
            Gripper("left", half_width_m=0, inward_effort_n=15, duration_s=.5),
            CartesianPose("left", (0., 0., .9), safety=MotionSafety(stop_on_grip_loss="cloth")),
            RawAction({"op": "unsupported"}),
        ):
            with self.subTest(action=action), self.assertRaises(UnsupportedActionError):
                self.backend.act(action)
            self.backend.env.act.assert_not_called()
            self.backend.ik.solve.assert_not_called()
            self.assertEqual(self.backend.env.tick, 5184)
            self.assertIsNone(self.backend.env.fault)

    def test_invalid_reset_options_do_not_reach_simulator(self):
        self.backend.env.reset = Mock()
        for kwargs in ({"seed": 7}, {"options": {"garment": "jeans"}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ActionRejectedError):
                self.backend.reset(**kwargs)
        self.backend.env.reset.assert_not_called()

    def test_physics_value_error_is_never_reclassified_as_safe_rejection(self):
        for advances_tick in (False, True):
            with self.subTest(advances_tick=advances_tick):
                self.backend.env.tick, self.backend.env.fault = 5184, None

                def fail(*args, **kwargs):
                    if advances_tick:
                        self.backend.env.tick += 1
                    else:
                        self.backend.env.fault = "ValueError: step failed before tick increment"
                    raise ValueError("physics failed")

                self.backend.env.act.side_effect = fail
                with self.assertRaises(BackendProtocolError):
                    self.backend.act(NamedJointTarget({}, ticks=1))
                self.assertIsNotNone(self.backend.env.fault)


if __name__ == "__main__":
    unittest.main()

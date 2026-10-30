import unittest
import numpy as np
from titrate.newton15.actuator_targets import ActuatorTargets


class TargetTests(unittest.TestCase):
    def make(self):
        return ActuatorTargets({'finger': {'q': 0, 'lower': 0., 'upper': .04,
                                          'velocity_limit': .2}}, [.04])

    def test_destination_duration_and_rate(self):
        controller = self.make()
        controller.command({'finger': .025})
        for _ in range(50):
            controller.advance(.001)
        self.assertEqual(controller.requested[0], .025)
        self.assertAlmostEqual(controller.setpoint[0], .03)
        for _ in range(25):
            controller.advance(.001)
        self.assertAlmostEqual(controller.setpoint[0], .025)
        controller.advance(.001)
        self.assertEqual(controller.setpoint[0], .025)

    def test_invalid_command_atomic(self):
        controller = self.make()
        with self.assertRaises(ValueError):
            controller.command({'finger': .03, 'invalid': .01})
        self.assertEqual(controller.requested[0], .04)

    def test_checkpoint_mid_ramp(self):
        controller = self.make()
        controller.command({'finger': .0})
        controller.advance(.013)
        snapshot = controller.capture()
        expected = controller.advance(.015).copy()
        controller.restore_validated(controller.validate(snapshot))
        np.testing.assert_array_equal(expected, controller.advance(.015))


if __name__ == '__main__':
    unittest.main()

"""Constructor wiring checks, not physical or visual acceptance."""

import unittest
from unittest.mock import patch
import gym_anything_robotics as gar
from gym_anything_robotics.errors import UnsupportedConfigurationError


class TitrationRegistryTest(unittest.TestCase):

    def test_alias_and_exact_options(self):
        with patch('gym_anything_robotics.backends.newton_titration.NewtonTitrationBackend') as backend:
            env = gar.create('titrate', auto_reset=False, apparatus_fixture=True,
                cameras=(), dt_s=.002)
            backend.assert_called_once_with(robot='panda', apparatus_fixture=True,
                cameras=(), dt_s=.002)
            env.close()


    def test_unqualified_default_does_not_launch(self):
        with self.assertRaises(UnsupportedConfigurationError):
            gar.create('titration')


if __name__ == '__main__':
    unittest.main()

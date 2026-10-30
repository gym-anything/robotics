import unittest
import numpy as np
from titrate.newton15.chemistry import equilibrium_ph, KW


class ChemistryTests(unittest.TestCase):
    def test_water_and_dilute_branches(self):
        net = np.array([.1, 1e-8, 0., -1e-8, -.1])
        ph = equilibrium_ph(net)
        self.assertTrue(np.all(np.diff(ph) > 0))
        self.assertAlmostEqual(ph[0], 1., places=10)
        self.assertAlmostEqual(ph[2], -.5*np.log10(KW))
        self.assertAlmostEqual(ph[-1], 13.17, places=10)


if __name__ == "__main__":
    unittest.main()

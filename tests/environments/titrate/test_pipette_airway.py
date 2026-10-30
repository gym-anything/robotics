import unittest
from types import SimpleNamespace
import numpy as np
import warp as wp
from titrate.newton15.runtime import require_newton
from titrate.newton15.pipette_airway import PipetteAirway


class AirwayTests(unittest.TestCase):
    def area(self, points):
        require_newton()
        water = SimpleNamespace(config=SimpleNamespace(spacing_m=.000625,
            support_m=.00125, device='cpu'),
            state=SimpleNamespace(particle_q=wp.array(points, dtype=wp.vec3, device='cpu')),
            grid=wp.HashGrid(16, 16, 512, device='cpu'))
        return PipetteAirway(water).area(wp.transform_identity())

    def test_empty_and_stray_drop_are_open(self):
        self.assertGreater(self.area([[.02, .02, .03]]), 0.)
        # One parcel in the wide bulb is not a cross-section-spanning plug.
        self.assertGreater(self.area([[0., 0., .17]]), 0.)

    def test_liquid_column_seals(self):
        dx = .000625
        x = (np.arange(-3, 3)+.5)*dx
        xyz = np.array(np.meshgrid(x, x, np.arange(.04, .05, dx), indexing='ij')).reshape(3, -1).T
        xyz = xyz[np.linalg.norm(xyz[:, :2], axis=1) < .0015]
        self.assertEqual(self.area(xyz), 0.)


if __name__ == '__main__':
    unittest.main()

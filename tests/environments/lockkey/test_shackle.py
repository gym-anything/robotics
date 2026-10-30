"""Fast geometry guards for the shackle."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from lockkey.newton15.config import Config
from lockkey.newton15.shackle import LONG_AXIS, geometry_state


class ShackleGeometryTests(unittest.TestCase):

    def test_old_recordings_keep_original_rail_and_stroke(self):
        c = Config.from_recorded({"mechanical_latch": True})
        self.assertFalse(c.shackle_swivel)
        self.assertEqual(c.shackle_travel, .011)

    def test_closed_lifted_and_swung_geometry(self):
        origin = np.array([.10, 0., .85])
        closed = geometry_state(np.r_[origin, [0., 0., 0., 1.]], origin)
        self.assertAlmostEqual(closed["shackle_short_leg_clearance_m"], -.010)
        self.assertFalse(closed["shackle_open"])
        for degrees in (-90., 90.):
            rotation = Rotation.from_euler("z", degrees, degrees=True)
            position = origin+LONG_AXIS-rotation.apply(LONG_AXIS)+[0., 0., Config().shackle_travel]
            opened = geometry_state(np.r_[position, rotation.as_quat()], origin)
            self.assertAlmostEqual(opened["shackle_short_leg_clearance_m"], .008)
            self.assertAlmostEqual(opened["shackle_staple_clearance_m"], .001)
            self.assertLess(opened["shackle_axis_error_m"], 1e-12)
            self.assertTrue(opened["shackle_open"])


if __name__ == "__main__":
    unittest.main()

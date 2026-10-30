"""Rated liquid uptake includes air expansion and stays inside the barrel."""

import unittest
import numpy as np
from titrate.newton15.pipette_geometry import profile, capacity, MARK_Z_M, FACET_AREA_RATIO
from titrate.newton15.pipette_filler import DEAD_VOLUME_M3, PISTON_AREA_M2, STROKE_M, PISTON_FACE_Z_M


class FillerCapacityTest(unittest.TestCase):
    def test_ten_ml_liquid_and_air_expansion_fit(self):
        inner = np.array(profile())[7:][::-1]
        air0 = FACET_AREA_RATIO*capacity(inner, .429)+DEAD_VOLUME_M3
        nominal = FACET_AREA_RATIO*capacity(inner, MARK_Z_M)
        gauge_head = 998.2*9.80665*MARK_Z_M
        needed = nominal+air0*(101325./(101325.-gauge_head)-1.)
        self.assertGreater(needed, nominal)
        self.assertGreater(STROKE_M*PISTON_AREA_M2, needed)
        self.assertLess(PISTON_FACE_Z_M+STROKE_M+.002, .505)


if __name__ == '__main__':
    unittest.main()

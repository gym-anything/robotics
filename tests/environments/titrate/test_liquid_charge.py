"""Geometric initialization invariants, not physical-liquid acceptance."""

import unittest
import numpy as np
from titrate.newton15.liquid_charge import fill_cavity
from titrate.newton15.pipette_geometry import (
    profile, capacity, LENGTH_M, MARK_Z_M, NOMINAL_VOLUME_M3, FACET_AREA_RATIO)


class ChargeGeometryTests(unittest.TestCase):
    def test_clearance_changes_placement_not_inventory(self):
        cavity = [(.00525, 0.), (.00525, .40)]
        wide = fill_cavity(cavity, 20., .0005, wall_clearance_fraction=0.)
        inset = fill_cavity(cavity, 20., .0005, wall_clearance_fraction=.5)
        self.assertEqual(len(wide), 160000)
        self.assertEqual(len(inset), len(wide))
        self.assertLess(wide[:, 2].max(), inset[:, 2].max())
        self.assertGreater(np.linalg.norm(wide[:, :2], axis=1).max(),
                           np.linalg.norm(inset[:, :2], axis=1).max())
        width = 2*np.pi/96
        theta = np.arctan2(wide[:, 1], wide[:, 0])
        middle = (np.floor(theta/width)+.5)*width
        distance = wide[:, 0]*np.cos(middle)+wide[:, 1]*np.sin(middle)
        self.assertTrue(np.all(distance < .00525*np.cos(width/2)))

    def test_all_axes_share_the_boundary_quadrature_lattice(self):
        spacing = .000625
        charge = fill_cavity([(.027, .003), (.027, .094)], 20., spacing,
                             wall_clearance_fraction=.0032)
        phase = np.mod(charge/spacing-.5, 1.)
        np.testing.assert_allclose(np.minimum(phase, 1.-phase), 0., atol=1e-5)

    def test_pipette_matches_aliquot_and_actual_facets(self):
        outline = np.asarray(profile())
        inner = outline[len(outline)//2:][::-1]
        self.assertEqual(LENGTH_M, .429)
        self.assertEqual(NOMINAL_VOLUME_M3, 10e-6)
        self.assertAlmostEqual(FACET_AREA_RATIO*capacity(inner, MARK_Z_M),
                               NOMINAL_VOLUME_M3, delta=1e-13)


if __name__ == '__main__':
    unittest.main()

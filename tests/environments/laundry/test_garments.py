from __future__ import annotations

import unittest

from laundry.garments import (
    GarmentTopology,
    get_garment_profile,
    implemented_garment_profiles,
)


class GarmentCatalogueTests(unittest.TestCase):
    def test_only_exercised_rectangular_profile_is_implemented(self) -> None:
        implemented = implemented_garment_profiles()
        self.assertEqual(tuple(profile.id for profile in implemented), ("canonical_cotton_towel",))
        self.assertEqual(implemented[0].topology, GarmentTopology.RECTANGULAR_SHEET)
        towel = implemented[0].as_towel_spec()
        self.assertAlmostEqual(towel.mass_kg, 0.072)

    def test_planned_topology_cannot_silently_become_a_rectangle(self) -> None:
        shirt = get_garment_profile("t_shirt")
        self.assertFalse(shirt.implemented)
        with self.assertRaises(NotImplementedError):
            shirt.as_towel_spec()

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_garment_profile("imaginary_garment")


if __name__ == "__main__":
    unittest.main(verbosity=2)

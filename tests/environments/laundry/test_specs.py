from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from laundry.specs import MATERIAL_PROFILES, PHYSICS_PROFILES, TowelSpec, write_spec_manifest


class SpecTests(unittest.TestCase):
    def test_towel_mass_is_areal_density_times_area(self) -> None:
        towel = TowelSpec()
        self.assertAlmostEqual(towel.mass_kg, towel.length_m * towel.width_m * towel.areal_density_kg_m2)
        self.assertIn("prior", towel.calibration_status)

    def test_material_profiles_separate_warp_weft_and_shear(self) -> None:
        profile = MATERIAL_PROFILES["cotton_towel_prior"]
        self.assertNotEqual(profile.warp_membrane_n_m, profile.weft_membrane_n_m)
        self.assertLess(profile.shear_n_m, profile.weft_membrane_n_m)
        self.assertIn("prior", profile.calibration_status)

    def test_physics_profiles_refine_monotonically(self) -> None:
        fast, interactive, validation = (PHYSICS_PROFILES[name] for name in ("fast", "interactive", "validation"))
        self.assertLess(fast.cloth_nx, interactive.cloth_nx)
        self.assertLess(interactive.cloth_nx, validation.cloth_nx)
        self.assertGreater(fast.physics_dt_s, interactive.physics_dt_s)
        self.assertGreater(validation.substeps, interactive.substeps)
        self.assertLess(validation.physics_dt_s, interactive.physics_dt_s)

    def test_self_contact_ccd_ceiling_exceeds_canonical_drop_speed(self) -> None:
        impact_speed = math.sqrt(2.0 * 9.80665 * 0.10)
        for profile in PHYSICS_PROFILES.values():
            with self.subTest(profile=profile.name):
                self.assertGreater(profile.maximum_self_contact_sweep_speed_m_s, impact_speed)

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_spec_manifest(Path(directory) / "specs.json")
            text = path.read_text(encoding="utf-8")
            self.assertIn('"schema_version": "laundry-specs/1.0"', text)
            self.assertIn("not physically calibrated", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

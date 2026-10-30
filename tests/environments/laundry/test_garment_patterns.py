from __future__ import annotations

import unittest

from laundry.garment_asset import FeatureKind, OpeningKind, PathRole, SeamType
from laundry.garment_patterns import GARMENT_CONSTRUCTIONS, get_garment_construction
from laundry.garments import GarmentTopology


EXPECTED_IDS = (
    "cotton_jersey_half_sleeve_tee",
    "woven_button_down_shirt",
    "denim_five_pocket_jeans",
    "terry_cotton_bath_towel",
    "asymmetric_draped_one_sleeve_top",
)


class GarmentPatternCatalogueTests(unittest.TestCase):
    def test_requested_catalogue_is_present_and_structurally_distinct(self) -> None:
        self.assertEqual(tuple(GARMENT_CONSTRUCTIONS), EXPECTED_IDS)
        signatures = [garment.topology_signature for garment in GARMENT_CONSTRUCTIONS.values()]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_every_construction_has_accounted_mass_area_tasks_and_provenance(self) -> None:
        plausible_mass_ranges = {
            "cotton_jersey_half_sleeve_tee": (0.08, 0.30),
            "woven_button_down_shirt": (0.10, 0.40),
            "denim_five_pocket_jeans": (0.25, 0.90),
            "terry_cotton_bath_towel": (0.10, 0.80),
            "asymmetric_draped_one_sleeve_top": (0.08, 0.45),
        }
        for garment in GARMENT_CONSTRUCTIONS.values():
            with self.subTest(garment=garment.id):
                self.assertGreater(garment.total_panel_area_m2, 0.2)
                low, high = plausible_mass_ranges[garment.id]
                self.assertGreaterEqual(garment.total_panel_mass_kg, low)
                self.assertLessEqual(garment.total_panel_mass_kg, high)
                self.assertTrue(garment.supported_tasks)
                self.assertEqual(garment.provenance.license_spdx, "MIT")
                self.assertIn("prior", garment.calibration_status)
                for panel in garment.panels:
                    self.assertGreater(panel.area_m2, 0.0)
                    self.assertGreater(panel.mass_kg, 0.0)

    def test_every_catalogue_panel_has_the_authored_orientation_assignment(self) -> None:
        expected_reversed = {
            "cotton_jersey_half_sleeve_tee": {
                "back",
                "right_sleeve",
                "neck_rib",
            },
            "woven_button_down_shirt": {
                "right_front",
                "left_sleeve",
                "right_cuff",
            },
            "denim_five_pocket_jeans": {
                "front_right",
                "back_left",
                "front_left_waistband",
                "back_right_waistband",
                "left_pocket_inner",
                "right_pocket_inner",
                "left_back_patch_pocket",
                "coin_pocket",
                "back_left_yoke",
                "fly_shield",
            },
            "terry_cotton_bath_towel": set(),
            "asymmetric_draped_one_sleeve_top": {
                "back",
                "right_cap_sleeve",
                "front_neck_facing",
            },
        }
        for construction_id, expected in expected_reversed.items():
            garment = get_garment_construction(construction_id)
            actual = {
                panel.id
                for panel in garment.panels
                if panel.physical_winding_reversed
            }
            self.assertEqual(actual, expected, construction_id)

    def test_tee_is_a_five_panel_jersey_construction_with_rib_and_openings(self) -> None:
        tee = get_garment_construction("cotton_jersey_half_sleeve_tee")
        self.assertEqual(tee.topology, GarmentTopology.SEWN_PANEL_SHELL)
        self.assertEqual(len(tee.panels), 5)
        self.assertGreaterEqual(len(tee.seams), 13)
        self.assertEqual(
            {panel.id for panel in tee.panels},
            {"front", "back", "left_sleeve", "right_sleeve", "neck_rib"},
        )
        self.assertEqual(
            {opening.kind for opening in tee.openings},
            {OpeningKind.NECK, OpeningKind.ARM, OpeningKind.WAIST},
        )
        self.assertIn(FeatureKind.RIB_BAND, {feature.kind for feature in tee.features})
        self.assertEqual(tee.panel_map["neck_rib"].material_profile, "cotton_rib_prior")
        for sleeve_id in ("left_sleeve", "right_sleeve"):
            fold = tee.panel_map[sleeve_id].path("flat_lay_fold_axis")
            self.assertEqual(fold.role, PathRole.FOLD_REFERENCE)
        self.assertEqual(
            {
                panel.id
                for panel in tee.panels
                if panel.physical_winding_reversed
            },
            {"back", "right_sleeve", "neck_rib"},
        )

    def test_shirt_has_split_fronts_real_pocket_cuffs_collar_and_plackets(self) -> None:
        shirt = get_garment_construction("woven_button_down_shirt")
        panel_ids = set(shirt.panel_map)
        self.assertTrue(
            {
                "left_front",
                "right_front",
                "back",
                "left_sleeve",
                "right_sleeve",
                "left_cuff",
                "right_cuff",
                "back_yoke",
                "chest_pocket",
                "collar_stand",
                "collar",
            }.issubset(panel_ids)
        )
        kinds = {feature.kind for feature in shirt.features}
        self.assertTrue(
            {FeatureKind.POCKET, FeatureKind.PLACKET, FeatureKind.BUTTON, FeatureKind.COLLAR, FeatureKind.CUFF}.issubset(kinds)
        )
        self.assertTrue(any(opening.id == "chest_pocket" for opening in shirt.openings))
        self.assertTrue(any(opening.id == "front_placket" and opening.closable for opening in shirt.openings))

    def test_jeans_have_four_leg_shells_and_physical_five_pocket_construction(self) -> None:
        jeans = get_garment_construction("denim_five_pocket_jeans")
        panel_ids = set(jeans.panel_map)
        self.assertTrue({"front_left", "front_right", "back_left", "back_right"}.issubset(panel_ids))
        self.assertTrue(
            {
                "left_pocket_outer",
                "left_pocket_inner",
                "right_pocket_outer",
                "right_pocket_inner",
                "left_back_patch_pocket",
                "right_back_patch_pocket",
                "coin_pocket",
                "fly_shield",
                "fly_facing",
            }.issubset(panel_ids)
        )
        self.assertEqual(len([opening for opening in jeans.openings if opening.kind is OpeningKind.POCKET]), 5)
        seam_types = {seam.seam_type for seam in jeans.seams}
        self.assertIn(SeamType.FLAT_FELLED, seam_types)
        self.assertIn(SeamType.WAISTBAND, seam_types)
        feature_kinds = {feature.kind for feature in jeans.features}
        self.assertTrue({FeatureKind.ZIPPER, FeatureKind.WAISTBAND, FeatureKind.BELT_LOOP, FeatureKind.POCKET_BAG}.issubset(feature_kinds))
        self.assertTrue(all(panel.material_profile == "denim_twill_prior" for panel in jeans.panels[:4]))
        self.assertFalse(jeans.panel_map["front_left"].seed_mirror_material_x)
        self.assertFalse(jeans.panel_map["back_left"].seed_mirror_material_x)
        self.assertTrue(jeans.panel_map["front_right"].seed_mirror_material_x)
        self.assertTrue(jeans.panel_map["back_right"].seed_mirror_material_x)

    def test_terry_towel_has_pile_thickness_four_hems_and_dobby_bands(self) -> None:
        towel = get_garment_construction("terry_cotton_bath_towel")
        self.assertEqual(towel.topology, GarmentTopology.RECTANGULAR_SHEET)
        self.assertEqual(len(towel.panels), 1)
        body = towel.panels[0]
        self.assertEqual(body.material_profile, "terry_cotton_prior")
        self.assertGreaterEqual(body.unloaded_thickness_m, 3.0e-3)
        self.assertEqual(sum(path.role is PathRole.HEM for path in body.outline_paths), 4)
        self.assertEqual(sum(feature.kind is FeatureKind.DOBBY_BAND for feature in towel.features), 2)
        hem = next(feature for feature in towel.features if feature.kind is FeatureKind.HEM_BAND)
        self.assertEqual(len(hem.path_refs), 4)
        self.assertGreaterEqual(hem.layer_count, 3)

    def test_asymmetric_top_is_not_a_mirrored_tshirt(self) -> None:
        top = get_garment_construction("asymmetric_draped_one_sleeve_top")
        self.assertIn("drape_overlay", top.panel_map)
        self.assertIn("left_tie", top.panel_map)
        self.assertIn("right_tie", top.panel_map)
        left_opening = top.panel_map["left_bell_sleeve"].path("opening_hem").length_m
        right_opening = top.panel_map["right_cap_sleeve"].path("opening_hem").length_m
        self.assertGreater(left_opening, 1.5 * right_opening)
        self.assertEqual(top.panel_map["drape_overlay"].grain_direction, (1.0, 1.0))
        self.assertTrue(any(opening.id == "drape_front" for opening in top.openings))
        self.assertIn(FeatureKind.TIE, {feature.kind for feature in top.features})
        for panel_id in ("front_neck_facing", "back_neck_facing"):
            facing = top.panel_map[panel_id]
            self.assertGreater(
                facing.path("free_edge").length_m,
                facing.path("neck_attach").length_m,
            )
            self.assertAlmostEqual(
                facing.path("left_shoulder_join").length_m,
                0.022,
                places=9,
            )
            self.assertAlmostEqual(
                facing.path("right_shoulder_join").length_m,
                0.022,
                places=9,
            )

    def test_unknown_construction_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_garment_construction("rectangle_with_a_shirt_label")


if __name__ == "__main__":
    unittest.main(verbosity=2)

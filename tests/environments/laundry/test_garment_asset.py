from __future__ import annotations

from dataclasses import replace
import unittest

from laundry.garment_asset import (
    AssetProvenance,
    GarmentConstruction,
    OpeningKind,
    OpeningSpec,
    PanelPattern,
    PathRef,
    PathRole,
    PatternPath,
    ProvenanceKind,
    SeamSpec,
    SeamType,
    point_in_polygon,
    signed_polygon_area,
)
from laundry.garments import GarmentTopology


def rectangle_panel(panel_id: str = "panel", *, width: float = 0.2, height: float = 0.3) -> PanelPattern:
    p0, p1, p2, p3 = (0.0, 0.0), (width, 0.0), (width, height), (0.0, height)
    return PanelPattern(
        id=panel_id,
        label="Test panel",
        material_profile="cotton_poplin_prior",
        areal_density_kg_m2=0.1,
        unloaded_thickness_m=0.3e-3,
        grain_direction=(0.0, 1.0),
        outline_paths=(
            PatternPath("bottom", (p0, p1), PathRole.HEM),
            PatternPath("right", (p1, p2), PathRole.SEAM),
            PatternPath("top", (p2, p3), PathRole.OPENING),
            PatternPath("left", (p3, p0), PathRole.SEAM),
        ),
    )


def provenance() -> AssetProvenance:
    return AssetProvenance(
        ProvenanceKind.REPOSITORY_AUTHORED,
        "repo://test",
        "MIT",
        "Unit-test fixture, not a physical specimen.",
    )


class PatternGeometryTests(unittest.TestCase):
    def test_signed_area_and_boundary_containment(self) -> None:
        polygon = ((0.0, 0.0), (0.2, 0.0), (0.2, 0.3), (0.0, 0.3))
        self.assertAlmostEqual(signed_polygon_area(polygon), 0.06)
        self.assertTrue(point_in_polygon((0.1, 0.1), polygon))
        self.assertTrue(point_in_polygon((0.0, 0.1), polygon))
        self.assertFalse(point_in_polygon((0.3, 0.1), polygon))

    def test_path_uses_arclength_interpolation(self) -> None:
        path = PatternPath("elbow", ((0.0, 0.0), (0.1, 0.0), (0.1, 0.3)), PathRole.SEAM)
        self.assertAlmostEqual(path.length_m, 0.4)
        self.assertEqual(path.point_at_fraction(0.5), (0.1, 0.1))
        self.assertEqual(path.resample(3, reverse=True)[0], (0.1, 0.3))

    def test_zero_length_path_segment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PatternPath("bad", ((0.0, 0.0), (0.0, 0.0)), PathRole.SEAM)

    def test_open_panel_outline_is_rejected(self) -> None:
        panel = rectangle_panel()
        broken_left = PatternPath("left", ((0.0, 0.3), (0.01, 0.0)), PathRole.SEAM)
        with self.assertRaisesRegex(ValueError, "does not close"):
            replace(panel, outline_paths=panel.outline_paths[:-1] + (broken_left,))

    def test_clockwise_panel_outline_is_rejected(self) -> None:
        p0, p1, p2, p3 = (0.0, 0.0), (0.0, 0.3), (0.2, 0.3), (0.2, 0.0)
        with self.assertRaisesRegex(ValueError, "counter-clockwise"):
            replace(
                rectangle_panel(),
                outline_paths=(
                    PatternPath("left", (p0, p1), PathRole.SEAM),
                    PatternPath("top", (p1, p2), PathRole.OPENING),
                    PatternPath("right", (p2, p3), PathRole.SEAM),
                    PatternPath("bottom", (p3, p0), PathRole.HEM),
                ),
            )

    def test_internal_stitch_cannot_leave_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaves panel"):
            replace(
                rectangle_panel(),
                internal_paths=(
                    PatternPath(
                        "outside",
                        ((0.1, 0.1), (0.4, 0.1)),
                        PathRole.INTERNAL_STITCH,
                    ),
                ),
            )

    def test_panel_mass_is_pattern_area_times_areal_density(self) -> None:
        panel = rectangle_panel()
        self.assertAlmostEqual(panel.area_m2, 0.06)
        self.assertAlmostEqual(panel.mass_kg, 0.006)


class ConstructionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = rectangle_panel("left")
        self.right = rectangle_panel("right")
        self.seam = SeamSpec(
            "side_join",
            PathRef("left", "right"),
            PathRef("right", "left", reverse=True),
            SeamType.PLAIN,
            seam_allowance_m=0.01,
            stitch_spacing_m=0.003,
            line_stiffness_n_m=1.0e4,
            line_damping_n_s_m=5.0,
        )

    def construction(self, **changes) -> GarmentConstruction:
        values = dict(
            id="test_garment",
            label="Test garment",
            category="test",
            topology=GarmentTopology.SEWN_PANEL_SHELL,
            panels=(self.left, self.right),
            seams=(self.seam,),
            openings=(OpeningSpec("top", OpeningKind.NECK, (PathRef("left", "top"),)),),
            features=(),
            supported_tasks=("test_task",),
            provenance=provenance(),
            calibration_status="test prior",
        )
        values.update(changes)
        return GarmentConstruction(**values)

    def test_valid_construction_has_mass_and_signature(self) -> None:
        garment = self.construction()
        self.assertAlmostEqual(garment.total_panel_mass_kg, 0.012)
        self.assertEqual(garment.topology_signature[1], 2)
        self.assertEqual(garment.construction_summary()["seam_count"], 1)

    def test_missing_path_reference_is_rejected(self) -> None:
        bad = replace(self.seam, side_b=PathRef("right", "not_there"))
        with self.assertRaisesRegex(ValueError, "missing path"):
            self.construction(seams=(bad,))

    def test_undeclared_seam_ease_is_rejected(self) -> None:
        narrow = rectangle_panel("narrow", height=0.15)
        bad = replace(
            self.seam,
            side_b=PathRef("narrow", "left"),
            maximum_ease_fraction=0.05,
        )
        with self.assertRaisesRegex(ValueError, "exceeds declared ease"):
            self.construction(panels=(self.left, narrow), seams=(bad,))

    def test_opening_requires_opening_or_hem_role(self) -> None:
        bad_opening = OpeningSpec("bad", OpeningKind.ARM, (PathRef("left", "right"),))
        with self.assertRaisesRegex(ValueError, "incompatible role"):
            self.construction(openings=(bad_opening,))

    def test_duplicate_panel_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "panel identifiers"):
            self.construction(panels=(self.left, self.left))


if __name__ == "__main__":
    unittest.main(verbosity=2)

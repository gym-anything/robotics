import math
import unittest

from football.ball_visual_deformation import (
    ContactCompressionMode,
    compute_ball_visual_deformation,
)


class BallVisualDeformationTests(unittest.TestCase):
    def setUp(self):
        self.radius = 0.11
        self.volume_coupling = 0.44

    def compute(self, modes, orientation=(1.0, 0.0, 0.0, 0.0)):
        return compute_ball_visual_deformation(
            ball_radius_m=self.radius,
            volume_coupling=self.volume_coupling,
            ball_orientation_world_wxyz=orientation,
            modes=modes,
        )

    def test_identity_without_contact(self):
        state = self.compute(())
        self.assertEqual(state.translation_ball_local_m, (0.0, 0.0, 0.0))
        self.assertEqual(
            state.deformation_gradient_ball_local,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(state.determinant, 1.0)
        self.assertEqual(state.modes, ())

    def test_single_contact_preserves_far_pole_and_indents_near_pole(self):
        compression = 0.020
        state = self.compute((
            ContactCompressionMode("foot", compression, (1.0, 0.0, 0.0)),
        ))
        axial = state.modes[0].axial_scale
        translation = state.translation_ball_local_m[0]
        far_pole = translation + axial * self.radius
        contact_pole = translation - axial * self.radius
        self.assertAlmostEqual(far_pole, self.radius, places=12)
        self.assertAlmostEqual(
            contact_pole, -self.radius + compression, places=12
        )
        self.assertGreater(state.modes[0].transverse_scale, 1.0)
        self.assertGreater(state.determinant, 0.0)

    def test_world_normal_is_rotated_into_ball_frame(self):
        # +90 degrees about world Z: local +X points along world +Y, so a
        # world +Y contact must squash local +X.
        half = math.sqrt(0.5)
        state = self.compute(
            (ContactCompressionMode("foot", 0.010, (0.0, 1.0, 0.0)),),
            orientation=(half, 0.0, 0.0, half),
        )
        local = state.modes[0].normal_ball_local
        self.assertAlmostEqual(local[0], 1.0, places=12)
        self.assertAlmostEqual(local[1], 0.0, places=12)
        self.assertAlmostEqual(local[2], 0.0, places=12)

    def test_surface_anchor_can_differ_from_modal_compression(self):
        modal_compression = 0.020
        surface_compression = 0.015
        state = self.compute((ContactCompressionMode(
            "foot",
            modal_compression,
            (1.0, 0.0, 0.0),
            surface_compression,
        ),))
        axial = state.modes[0].axial_scale
        translation = state.translation_ball_local_m[0]
        contact_pole = translation - axial * self.radius
        self.assertAlmostEqual(
            contact_pole,
            -self.radius + surface_compression,
            places=12,
        )
        # Shape magnitude still comes from the physical modal coordinate.
        self.assertAlmostEqual(
            axial,
            1.0 - modal_compression / (2.0 * self.radius),
            places=12,
        )

    def test_foot_and_pitch_composition_is_order_independent(self):
        foot = ContactCompressionMode("foot", 0.020, (1.0, 0.0, 0.0))
        pitch = ContactCompressionMode("pitch", 0.004, (0.0, 0.0, 1.0))
        first = self.compute((foot, pitch))
        second = self.compute((pitch, foot))
        self.assertEqual(first, second)
        self.assertGreater(first.determinant, 0.0)
        self.assertEqual(tuple(mode.name for mode in first.modes), ("foot", "pitch"))

    def test_invalid_or_inverted_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_ball_visual_deformation(
                ball_radius_m=0.0,
                volume_coupling=self.volume_coupling,
                ball_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
                modes=(),
            )
        with self.assertRaises(ValueError):
            self.compute((ContactCompressionMode("foot", -0.001, (1.0, 0.0, 0.0)),))
        with self.assertRaises(ValueError):
            self.compute((ContactCompressionMode("foot", 0.001, (0.0, 0.0, 0.0)),))


if __name__ == "__main__":
    unittest.main()

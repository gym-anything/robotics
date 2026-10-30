"""Tests of the hole module's face query for items at a workpiece's edges and corners."""

from __future__ import annotations

import unittest


class FaceAtEdgesTest(unittest.TestCase):
    """HoleModule.face_near_segment (pure geometry, no engine) on board1 of timber-bolt (140 x 300 x 38 mm) and the
    carriage bolt's shank (r 6.35 mm). A shank across an edge, end-on at a corner or across a part narrower than itself
    must get a face while it is within the margin and not deeper than `inside`."""

    R = 0.00635

    def setUp(self) -> None:
        import types
        import numpy as np
        from drilling.holes import HoleModule
        self.np = np
        self.hm = HoleModule.__new__(HoleModule)          # the query reads no module state
        self.d = types.SimpleNamespace(center=np.zeros(3), half=np.array([0.07, 0.15, 0.019]))

    def fns(self, a, b, inside=0.003, margin=0.0074):
        np = self.np
        return self.hm.face_near_segment(self.d, np.asarray(a, float), np.asarray(b, float), margin=margin,
                                         inside=inside, r=self.R)

    def test_shank_across_the_top_end_edge(self) -> None:
        """A carriage-bolt shank across board1's top end edge (board1's frame): 1.3 mm clear of the edge, then 1.2 mm
        into it. A face (top or end) at both, never none."""
        tip = self.np.array([-0.05315, 0.18899, 0.00659])
        base = self.np.array([-0.05516, 0.10010, 0.03975])
        for dz in (0.006, 0.003):
            f = self.fns(tip + [0., 0., dz], base + [0., 0., dz])
            self.assertIsNotNone(f, f"lifted {dz*1e3} mm")
            n = f[0]
            self.assertTrue(n[2] == 1. or n[1] == 1.)

    def test_shank_across_a_narrower_part(self) -> None:
        """Lying level across the 140 mm board with 20 mm over each side, 1 mm above it and resting 0.5 mm in."""
        for z in (0.019 + self.R + 0.001, 0.019 + self.R - 0.0005):
            f = self.fns([0.09, 0., z], [-0.09, 0., z])
            self.assertIsNotNone(f)
            self.assertEqual(f[0][2], 1.)

    def test_end_on_at_a_corner(self) -> None:
        """Pointing at the (+x, +y, top) corner along the body diagonal, 2 mm clear and touching."""
        np = self.np
        u = -np.ones(3)/np.sqrt(3.)
        for clear in (0.002, 0.0):
            tip = self.d.half - u*clear
            self.assertIsNotNone(self.fns(tip, tip - u*0.0946), f"clear {clear*1e3} mm")

    def test_over_the_middle_of_a_face(self) -> None:
        """Upright over the top face's middle, 2 mm above: the top face, the point at the shank's end."""
        f = self.fns([0.01, 0.02, 0.021], [0.01, 0.02, 0.1156])
        self.assertEqual(f[0].tolist(), [0., 0., 1.])
        self.assertEqual(f[2].tolist(), [0.01, 0.02, 0.021])

    def test_through_a_face_is_never_a_pad(self) -> None:
        """Upright through the top face 15 mm (in a hole, or a fault), not on a pad: no face."""
        self.assertIsNone(self.fns([0.01, 0.02, 0.004], [0.01, 0.02, 0.0986]))


if __name__ == "__main__":
    unittest.main()

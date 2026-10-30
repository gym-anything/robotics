"""Supplementary adapter contract tests with a stand-in model; no dynamics."""

from types import SimpleNamespace
import unittest
import numpy as np

from football.newton15.collision_ownership import enforce_ball_pairs


def fixture():
    # Pitch and post share world body 0. Body-wide exclusion would lose the post.
    m = SimpleNamespace(
        ngeom=4,
        geom_contype=np.array([1, 2, 4, 8], np.int32),
        geom_conaffinity=np.full(4, -1, np.int32),
        geom_bodyid=np.array([0, 0, 1, 2]),
        body_contype=np.array([3, 4, 8], np.int32),
        body_conaffinity=np.full(3, -1, np.int32),
        pair_geom1=np.array([], np.int32),
        pair_geom2=np.array([], np.int32),
    )
    return SimpleNamespace(
        ball_shape=2,
        solver=SimpleNamespace(
            mj_model=m,
            mjc_geom_to_newton_shape=SimpleNamespace(numpy=lambda: np.arange(4)),
        ),
        model=SimpleNamespace(
            shape_collision_filter_mask=lambda pairs: np.array(
                [True, False, True, True]
            )
        ),
    )


class OwnershipTests(unittest.TestCase):
    def test_exact_ball_pairs_and_unchanged_other_pairs(self):
        w = fixture()
        report = enforce_ball_pairs(w)
        m = w.solver.mj_model
        permitted = (m.geom_contype[2] & m.geom_conaffinity) | (
            m.geom_conaffinity[2] & m.geom_contype
        )
        self.assertEqual((permitted != 0).tolist(), [False, True, False, False])
        self.assertTrue(report["unrelated_pair_masks_unchanged"])
        self.assertNotEqual(m.body_contype[0] & m.geom_conaffinity[2], 0)

    def test_explicit_native_pair_cannot_bypass_mask(self):
        w = fixture()
        w.solver.mj_model.pair_geom1 = np.array([2])
        with self.assertRaisesRegex(RuntimeError, "Explicit ball"):
            enforce_ball_pairs(w)

    def test_no_unused_bit_fails_before_mutation(self):
        w = fixture()
        w.solver.mj_model.geom_contype[0] = -1
        original = w.solver.mj_model.geom_contype.copy()
        with self.assertRaisesRegex(RuntimeError, "No unused"):
            enforce_ball_pairs(w)
        np.testing.assert_array_equal(original, w.solver.mj_model.geom_contype)


if __name__ == "__main__":
    unittest.main()

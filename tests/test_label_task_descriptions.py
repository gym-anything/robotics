"""Label-application job descriptions and the trapped-air area. No simulator stepping, GPU scene or rendering."""
from types import SimpleNamespace
import unittest

import numpy as np

from label_application.task import (
    CONTACT_BAND, Evaluator, INSTRUCTIONS, lifted_material_area,
)


def grid(n, flip=False):
    """Same 6 x 6 mm piecewise-planar sheet, different tessellations."""
    u, v = np.meshgrid(np.linspace(0, .006, n + 1), np.linspace(0, .006, n + 1))
    X = np.c_[u.ravel(), v.ravel()]
    tris = []
    for j in range(n):
        for i in range(n):
            a = j*(n + 1) + i
            b, c, d = a + 1, a + n + 1, a + n + 2
            tris.extend(((a, b, c), (b, d, c)) if flip else ((a, b, d), (a, d, c)))
    return X, np.asarray(tris)


class DescriptionTest(unittest.TestCase):
    def test_api_uses_exact_short_job_descriptions(self):
        from label_application.variants import task_resolve, contract
        for preset, tasks in contract()["tasks"].items():
            for task in tasks:
                text = INSTRUCTIONS[task]
                self.assertEqual(task_resolve(dict(preset=preset, task=task, seed=0))["description"], text)
                self.assertNotIn("trapped air", text)
                self.assertNotIn("hands and", text)
                self.assertNotIn("within 30", text)
                self.assertNotIn("at most 2", text)
        self.assertIn("32 mm", INSTRUCTIONS["gs1-case-side"])
        self.assertIn("19 mm", INSTRUCTIONS["gs1-case-side"])
        self.assertIn("matches", INSTRUCTIONS["part-id-match"])
        self.assertIn("where it lies", INSTRUCTIONS["press-down"])


    def test_area_matches_analytic_plane_on_each_mesh_and_diagonal(self):
        # gap=100 um*u/6 mm. Above 50 um is the right half: 18 mm2.
        # Its intersection with u=[2,5],v=[1,4] mm is 2*3=6 mm2.
        for n in (1, 2, 4, 8):
            for flip in (False, True):
                X, T = grid(n, flip)
                gap = 1e-4*X[:, 0]/.006
                self.assertAlmostEqual(lifted_material_area(X, T, gap), 18e-6, places=14)
                self.assertAlmostEqual(lifted_material_area(
                    X, T, gap, rect=((.002, .005), (.001, .004))), 6e-6, places=14)

    def test_area_threshold_both_sides_without_node_quantization(self):
        for n in (2, 4, 8):
            X, T = grid(n)
            for expected in (2.9e-6, 3.1e-6, 19.9e-6, 20.1e-6):
                boundary = .006 - expected/.006
                # gap=band at the analytic boundary, with fixed positive slope.
                gap = CONTACT_BAND + .01*(X[:, 0] - boundary)
                area = lifted_material_area(X, T, gap)
                self.assertAlmostEqual(area, expected, places=14)
                self.assertEqual(area <= 3e-6, expected <= 3e-6)
                self.assertEqual(area <= 20e-6, expected <= 20e-6)

    def test_contact_empty_degenerate_and_outside_barcode_have_no_air_area(self):
        X, T = grid(2)
        for gap in (np.zeros(len(X)), np.full(len(X), CONTACT_BAND)):
            self.assertEqual(lifted_material_area(X, T, gap), 0.)
        gap = np.full(len(X), 2*CONTACT_BAND)
        self.assertEqual(lifted_material_area(X, T[:0], gap), 0.)
        self.assertEqual(lifted_material_area(X, np.array([[0, 0, 0]]), gap), 0.)
        self.assertEqual(lifted_material_area(X, T, gap, rect=((.02, .03), (.02, .03))), 0.)

    def test_pocket_integration_uses_gap_clipping_and_alive_triangles(self):
        X, T = grid(2)
        gap = 1e-4*X[:, 0]/.006
        lab = SimpleNamespace(
            X=X, tris=T, n=len(X), x=np.c_[X, gap],
            stock=SimpleNamespace(thickness=0.), vertex_area=np.full(len(X), 4e-6),
            pockets=[dict(vertices=np.arange(len(X)), collider="carton")],
            colliders=[SimpleNamespace(name="carton", sdf=lambda x: (x[:, 2], None, None))])
        ev = Evaluator.__new__(Evaluator)
        ev.world = SimpleNamespace(label=lab)
        ev.bar_u, ev.bar_v = (.002, .005), (.001, .004)
        air = ev._trapped_air()
        self.assertAlmostEqual(air["largest_m2"], 18e-6, places=14)
        self.assertAlmostEqual(air["in_barcode_m2"], 6e-6, places=14)
        lab.alive = np.zeros(len(T), dtype=bool)
        self.assertEqual(ev._trapped_air()["largest_m2"], 0.)
        lab.pockets = []
        self.assertEqual(ev._trapped_air()["in_barcode_m2"], 0.)


if __name__ == "__main__":
    unittest.main()

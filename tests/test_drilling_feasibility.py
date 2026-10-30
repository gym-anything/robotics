"""Drilling seed-catalogue feasibility: pure-Python geometry on the task layouts, no physics engine (a few
seconds)."""

from __future__ import annotations

import math
import unittest


class DrillingFeasibilityTest(unittest.TestCase):
    SEEDS = 120


    def test_wall_marks_clear_the_near_mat(self) -> None:
        from drilling.layout import near_mat, rebar_clearance_m, wall_layout
        for task in ("anchor-holes", "clean-and-set", "ledger-anchor", "rebar-relocate"):
            fixture = 0.038 if task == "ledger-anchor" else 0.
            C = rebar_clearance_m(fixture)
            for seed in range(self.SEEDS):
                L = wall_layout(task, seed, "dual-fr3", "")
                xs, zs = near_mat(L["rebar_offset"])
                for k, m in enumerate(L["marks"]):
                    d = min([abs(m["x"] - x) for x in xs] + [abs(m["z"] - z) for z in zs])
                    if task == "rebar-relocate" and k == L["rebar_mark"]:
                        # the hidden bar: a vertical bar the bit cannot miss, across a channel line clear of the
                        # horizontal bars
                        self.assertLess(min(abs(m["x"] - x) for x in xs), 0.5*13.3e-3 + 0.5*0.0127 - 0.003)
                        self.assertGreaterEqual(min(abs(m["z"] - z) for z in zs), C)
                    else:
                        self.assertGreaterEqual(d, C, f"{task} seed {seed} mark {k}: {d*1e3:.1f} mm from a bar")

    def test_rebar_relocate_replacement_window(self) -> None:
        from drilling import params as P
        from drilling.layout import near_mat, rebar_clearance_m, wall_layout
        C = rebar_clearance_m(0.)
        for seed in range(self.SEEDS):
            L = wall_layout("rebar-relocate", seed, "dual-fr3", "")
            xs, zs = near_mat(L["rebar_offset"])
            mk = L["marks"][L["rebar_mark"]]
            # every point of the window [2 x the deepest plausible stop, RELOCATE_MAX_M] on one side clears the bars
            stop = P.REBAR_COVER_M + 2*P.REBAR_D_M + 0.5*13.3e-3/math.tan(math.radians(65.))
            lo, hi = P.ABORTED_HOLE_DISTANCE_FACTOR*stop, P.RELOCATE_MAX_M
            self.assertLess(lo, hi)
            sides = []
            for sgn in (-1., 1.):
                pts = [mk["x"] + sgn*(lo + i*(hi - lo)/20) for i in range(21)]
                sides.append(all(min(abs(x - b) for b in xs) >= C and -1.19 < x < 1.19 for x in pts))
            self.assertTrue(any(sides), f"seed {seed}: no clear side")

    def test_ledger_anchor_seats_at_h_nom(self) -> None:
        from drilling import params as P
        from drilling.layout import wall_layout
        L = wall_layout("ledger-anchor", 0, "dual-fr3", "")
        an = [p for p in L["parts"] if p["kind"] == "anchor"][0]
        fixture = L["ledger"]["thickness"] + L["ledger"]["standoff"]
        self.assertAlmostEqual(an["seat"] - fixture, P.KBTZ2["h_nom"], delta=1e-6)
        self.assertEqual(L["ledger"]["standoff"], 0.)
        self.assertAlmostEqual(an["length"], 7*0.0254, delta=1e-6)          # KB-TZ2 1/2 x 7 (L)
        hm = [t for t in L["tools"] if t["kind"] == "hammer"][0]
        reach = hm["bit_len"] - (P.SDS_PLAY_M - P.IDLE_WINDOW_M) - fixture - 0.5*13.3e-3/math.tan(math.radians(65.))
        self.assertGreaterEqual(reach, 0.110)


if __name__ == "__main__":
    unittest.main()

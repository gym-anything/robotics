"""Contract checks for welding: catalogue (tasks, legacy seeds and the frozen variation catalogue with its splits),
registration, robot selection, tool_io validation, restricted access, variation selection through the backend and the
per-tick physics trace hook (on a stub world).

The first classes need no GPU, no physics and no renderer (seconds). The physics class builds the real world for
welding/v1/steel-cell/position-torch/seed-0 (the tacked start state comes from its fixture's cache, or is built once by
it: minutes) and is skipped unless GAR_WELDING_PHYSICS_TESTS=1 (the repository's convention for GPU tests). These are
interface checks, not physical qualification or a task solution.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gym_anything_robotics import tasks
from gym_anything_robotics.errors import (ActionRejectedError, UnsupportedActionError,
                                          UnsupportedConfigurationError)
from tests.packages import PACKAGES

ROOT = PACKAGES
ALIASES = ("welding", "weld", "mig-welding", "gmaw", "weld-steel-joint")


class CatalogueTest(unittest.TestCase):
    def test_benchmark_identity_and_schema(self):
        raw = json.loads((ROOT/"welding"/"benchmark.json").read_text(), parse_constant=lambda c: self.fail(c))
        self.assertEqual(tasks.catalog("welding"), raw)
        self.assertEqual(raw["schema"], "gar.task-catalog.v1")
        self.assertEqual(raw["environment_id"], "welding")
        self.assertEqual(raw["selectors"], ["preset", "task", "seed", "variation_id", "split", "selection_seed"])
        self.assertEqual(raw["resolver"], "welding.variants:task_resolve")
        self.assertEqual(raw["splits"], ["train", "validation", "test"])
        self.assertEqual(raw["cli_task_selector"], "task")
        vc = raw["variation_catalogue"]
        from welding import variations as VR
        self.assertEqual((vc["version"], vc["count"]), (VR.VERSION, VR.COUNT))
        for key in ("frozen_file", "manifest", "notes"):
            self.assertTrue((ROOT/vc[key]).is_file(), key)
        self.assertIs(type(raw["evaluator"]["available"]), bool)
        self.assertTrue((ROOT/raw["evaluator"]["source"]).is_file())
        for key in ("description", "count_scope", "qualification", "split_note"):
            self.assertTrue(raw[key].strip(), key)
        self.assertEqual(tasks.resolve("welding").task_id, raw["default_task_id"])
        self.assertEqual(raw["default_task_id"], "welding/v1/steel-cell/fillet-2f/seed-0")
        for alias in ALIASES:
            self.assertEqual(tasks.catalog(alias)["environment_id"], "welding")

    def test_tasks_resolve_and_round_trip(self):
        from welding.task import INSTRUCTIONS, NEEDS_HAND, START_STATE, TASKS, TIME_LIMIT_S, Evaluator
        from welding.variations import COUNT
        catalog = tasks.catalog("welding")
        self.assertEqual(catalog["count"], len(TASKS)*COUNT)
        self.assertEqual(catalog["selector_choices"]["task"], list(TASKS))
        for table in (INSTRUCTIONS, TIME_LIMIT_S, START_STATE):
            self.assertEqual(set(table), set(TASKS))
        self.assertLessEqual(NEEDS_HAND, set(TASKS))
        ids = set()
        for task, code in TASKS.items():
            name = task.replace("-", "_")
            self.assertTrue(hasattr(Evaluator, "_eval_" + name) or hasattr(Evaluator, "judge_" + name), task)
            for selection in ({"task": task, "seed": 7}, {"task": code, "seed": 7},
                              {"preset": "steel-cell", "task": task, "seed": 7}):
                with self.subTest(selection=selection):
                    spec = tasks.resolve("welding", selection)
                    self.assertEqual(spec.task_id, f"welding/v1/steel-cell/{task}/seed-7")
                    self.assertEqual(spec.launch.environment_id, "welding")
                    self.assertEqual(spec.launch.backend_options, {"preset": "steel-cell", "task": task, "seed": 7})
                    self.assertEqual(spec.description, INSTRUCTIONS[task])
                    self.assertIsNone(spec.split)
                    self.assertTrue(spec.evaluator["available"])
                    again = tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
                    self.assertEqual(again.to_dict(), spec.to_dict())
                    ids.add(spec.task_id)
        self.assertEqual(len(ids), len(TASKS))

    def test_bad_selections_rejected_with_value_errors(self):
        """Unknown tasks and presets, seeds outside 0-9, non-integer seeds (1.5, True, "3": rejected, not coerced)
        and unknown selectors are ValueErrors from the resolver itself."""
        for selection in ({"task": "pipe-6g"}, {"preset": "copper-brazing"}, {"preset": "x", "task": "tack"},
                          {"seed": -1}, {"seed": 10}, {"seed": 1.5}, {"seed": True}, {"seed": "3"}, {"extra": 1},
                          {"seed": 0, "variation_id": 0}, {"seed": 3, "split": "test"}, {"variation_id": 10000},
                          {"variation_id": -1}, {"variation_id": True}, {"variation_id": 1.0}, {"variation_id": "7"},
                          {"split": "holdout"}, {"split": "train", "selection_seed": 1.5},
                          {"split": "train", "selection_seed": False}, {"task": "pipe-6g", "variation_id": 3}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                tasks.resolve("welding", selection)
        from gym_anything_robotics.launch import LaunchSpec
        with self.assertRaises(ValueError):          # a task recipe never overwrites an explicit launch value
            tasks.resolve("welding", {"task": "tack"}, base=LaunchSpec("welding", backend_options={"seed": 3}))

    def test_seeds_are_deterministic_and_bounded(self):
        from welding.variants import SEEDS, coupon_offset, resolve
        self.assertEqual(resolve(task="tack", seed=0)["coupon_offset"], (0., 0., 0.))
        for seed in SEEDS:
            dx, dy, yaw = coupon_offset(seed)
            self.assertEqual((dx, dy, yaw), coupon_offset(seed))
            self.assertTrue(abs(dx) <= 3e-3 and abs(dy) <= 3e-3 and abs(yaw) <= 2., seed)
        self.assertEqual(len({coupon_offset(s) for s in SEEDS}), len(SEEDS))


class GridTest(unittest.TestCase):
    def test_default_grid_unchanged_and_faces_on_cell_edges(self):
        """The default 0.25 / 0.5 mm grid is the one every result was produced on (fingerprint), and at any other cell
        size every coupon face in the fine band lies on a cell edge."""
        import dataclasses
        import hashlib
        import numpy as np
        from welding.process import build_grid, plate_coupon, tee_coupon
        for coupon, want in ((tee_coupon(), "db4199c86405130f"), (plate_coupon(), "1f65371b4b43a323")):
            g, box = build_grid(coupon)
            h = hashlib.sha256()
            for a in (g.dx, g.dy, g.dz, np.asarray(g.origin, float), np.asarray(box, float)):
                h.update(np.ascontiguousarray(a, np.float64).tobytes())
            self.assertEqual(h.hexdigest()[:16], want, coupon.kind)
        for fine, fine_x in ((.35e-3, .7e-3), (.3e-3, .6e-3), (.2e-3, .4e-3), (.5e-3, 1e-3)):
            c = dataclasses.replace(tee_coupon(), fine=fine, fine_x=fine_x)
            g, _ = build_grid(c)
            for edges, face in ((g.ye, 0.), (g.ye, -c.web[1]), (g.ze, 0.), (g.xe, 0.), (g.xe, c.length)):
                with self.subTest(fine=fine, face=face):
                    self.assertLess(float(np.min(np.abs(edges - face))), 1e-12)


class EvaluatorGeometryTest(unittest.TestCase):
    """The fillet evaluator's readings on synthetic sections built as the pool builds them (a cell is weld metal iff
    its centre lies under the free surface): no physics."""

    @staticmethod
    def _section(leg, sag=0.):
        import numpy as np
        from welding import thermal as TH
        ye = np.arange(-4e-3, 12e-3 + 1e-9, 2.5e-4)
        yc = .5*(ye[1:] + ye[:-1])
        Y, Z = np.meshgrid(yc, yc, indexing="ij")
        A, B = np.array([0., leg]), np.array([leg, 0.])
        L = float(np.linalg.norm(B - A))
        t, n = (B - A)/L, np.array([1., 1.])/math.sqrt(2.)
        u = np.linspace(0., L, 25)
        R = (L*L/4 + sag*sag)/(2*sag) if sag else 0.
        h = np.sqrt(np.maximum(R*R - (u - L/2)**2, 0.)) - (R - sag) if sag else 0*u
        orig = (Y < 0) | (Z < 0)
        uu, ww = (Y - A[0])*t[0] + (Z - A[1])*t[1], (Y - A[0])*n[0] + (Z - A[1])*n[1]
        under = (uu >= 0) & (uu <= L) & (ww <= np.interp(np.clip(uu, 0, L), u, h))
        mat = np.where(orig, 1, np.where(under, TH.WELD, 0)).astype(np.int8)
        peakf = np.where(orig & (np.hypot(Y, Z) < 2e-3), 1., 0.)
        return mat, orig, peakf, ye, dict(A=A, B=B, u=u, h=h, chord_t=t, chord_n=n)

    def test_legs_and_size_are_read_on_the_free_surface(self):
        import numpy as np
        from welding import sections as SC
        for leg in (6.30e-3, 6.375e-3, 6.40e-3, 6.45e-3, 6.55e-3):
            for sag in (0., .5e-3, 1e-3):
                mat, orig, peakf, ye, prof = self._section(leg, sag)
                T = np.full(mat.shape, 20.)
                r = SC.tee_section(mat, orig, peakf, T, ye, ye, profile=prof)
                with self.subTest(leg=leg, sag=sag):
                    self.assertEqual(r["geometry"], "profile")
                    self.assertAlmostEqual(r["leg_vertical_m"], leg, delta=1e-7)
                    self.assertAlmostEqual(r["leg_horizontal_m"], leg, delta=1e-7)
                    self.assertAlmostEqual(r["size_m"], leg, delta=2e-6)
                    self.assertAlmostEqual(r["convexity_m"], sag, delta=1e-5)
        # the cells alone read a flat 6.45 mm leg one cell short: why the surface is read
        mat, orig, peakf, ye, _ = self._section(6.45e-3)
        self.assertAlmostEqual(SC.tee_section(mat, orig, peakf, np.full(mat.shape, 20.), ye, ye)["leg_vertical_m"],
                               6.25e-3, delta=1e-9)

    def test_cells_the_surface_does_not_explain_are_kept_whole_or_removed(self):
        import numpy as np
        from welding import sections as SC, thermal as TH
        mat, orig, peakf, ye, prof = self._section(6.40e-3)
        T = np.full(mat.shape, 20.)
        yc = .5*(ye[1:] + ye[:-1])
        jb = kb = int(np.searchsorted(yc, 0.))
        top = kb + int(np.argmin(mat[jb, kb:] == TH.WELD))
        stacked = mat.copy()
        stacked[jb, top] = stacked[jb, top + 1] = TH.WELD           # two whole cells stacked up the bracket face
        r = SC.tee_section(stacked, orig, peakf, T, ye, ye, profile=prof)
        self.assertAlmostEqual(r["leg_vertical_m"], ye[top + 2], delta=1e-9)
        self.assertEqual(r["leg_vertical_from"], "cells")
        holed = mat.copy()
        holed[jb, kb + 10] = 0                                       # a cell gone from the run along the face
        r = SC.tee_section(holed, orig, peakf, T, ye, ye, profile=prof)
        self.assertAlmostEqual(r["leg_vertical_m"], ye[kb + 10], delta=1e-9)

    def test_every_stop_is_judged_as_a_crater_whatever_the_direction_or_runs(self):
        """Every last-to-freeze place is judged as a crater: an unfilled stop fails after a touch-up elsewhere or where
        two runs meet, and a filled weld passes after a wash-in."""
        def verdict(rows):
            return _fillet_judge(rows)["checks"]["crater_filled"]["pass"]
        t_up = lambda x: 900. + x
        cases = (
            ("one pass +x", _fillet_rows([(.25, 149.75, lambda x: x)]), True),
            ("one pass -x", _fillet_rows([(.25, 149.75, lambda x: 150. - x)]), True),
            ("unfilled stop", _fillet_rows([(.25, 149.75, lambda x: x)], sizes=[(140., 149.75, 4.5)]), False),
            ("unfilled stop, then a touch-up at the start", _fillet_rows(
                [(.25, 1.75, lambda x: x), (2.25, 5.75, t_up), (6.25, 149.75, lambda x: x)],
                sizes=[(140., 149.75, 4.5)]), False),
            ("two runs meeting, meeting crater unfilled", _fillet_rows(
                [(.25, 74.75, lambda x: 500. + x), (75.25, 149.75, lambda x: 150. - x)], sizes=[(70., 80., 4.5)]), False),
            ("two runs meeting, filled", _fillet_rows([(.25, 74.75, lambda x: 500. + x),
                                                       (75.25, 149.75, lambda x: 150. - x)]), True),
            ("middle outward, the first run's end crater unfilled", _fillet_rows(
                [(75.25, 149.75, lambda x: x), (.25, 74.75, lambda x: 500. + 75. - x)], sizes=[(140., 149.75, 4.5)]),
             False),
            ("a sound touch-up mid-seam after a 6.0 mm start", _fillet_rows(
                [(.25, 59.75, lambda x: x), (60.25, 69.75, t_up), (70.25, 149.75, lambda x: x)],
                sizes=[(5.25, 8.75, 6.)]), True),
            ("run off past the bracket's end, its crater there", _fillet_rows(
                [(.25, 151.75, lambda x: x)], sizes=[(150.25, 151.75, 4.5)]), True))
        for name, rows, ok in cases:
            with self.subTest(name):
                self.assertEqual(verdict(rows), ok)

    def test_the_beads_end_face_is_left_out_wherever_the_bead_ends(self):
        """A bead's end face is not judged, whether the bead ends at 150.0 mm or at 149.5 mm."""
        for end, (xa, xb) in ((149.75, (149.25, 149.75)), (149.25, (148.75, 149.25))):
            rows = _fillet_rows([(.25, end, lambda x: x)], sizes=[(xa, xa, 6.45), (xb, xb, 5.8)])
            with self.subTest(end=end):
                self.assertTrue(_fillet_judge(rows)["success"])
        rows = _fillet_rows([(.25, 151.25, lambda x: x)], sizes=[(149.75, 149.75, 5.8)])     # inside the bead
        self.assertFalse(_fillet_judge(rows)["checks"]["crater_filled"]["pass"])

    def test_a_run_off_past_the_sections_read_has_no_crater_on_the_joint(self):
        """A pass running on past the last section read (151.75 mm) does not make that section a stop: mid-pass metal
        at 146-150 mm is judged by the 10 % undersize rule, as anywhere in the span, not as crater."""
        rows = _fillet_rows([(.25, 151.75, lambda x: x)], sizes=[(146.25, 149.75, 6.25)])
        r = _fillet_judge(rows)
        self.assertTrue(r["success"], r["checks"])
        self.assertIn("ran on past the sections read", [c["how"] for c in r["checks"]["crater_filled"]["craters"]])
        self.assertFalse(_fillet_judge(_fillet_rows([(.25, 151.25, lambda x: x)],
                                                    sizes=[(146.25, 149.75, 6.25)]))["checks"]["crater_filled"]["pass"])

    @staticmethod
    def _two_runs_meeting(fall=None, b_end=82.75, a=True):
        """Two runs meeting: run A laid -x from the 150 end down to 83.25 mm first; run B laid +x later, its pool's
        last-frozen section at 72.25 mm and its front frozen before that, out to b_end."""
        tb = lambda x: 500. + x if x <= 72.25 else 572.25 - .05*(x - 72.25)
        runs = ([(83.25, 149.75, lambda x: 300. + 150. - x)] if a else []) + [(.25, b_end, tb)]
        return _fillet_rows(runs, sizes=[fall] if fall else ())

    def test_a_craters_fall_off_toward_the_passs_front_is_judged_up_to_older_metal(self):
        """A 5.7 mm fall-off 5.5-10.5 mm past run B's stop fails, whether it ends against run A's metal or at a bead
        end 11 mm on."""
        judge = lambda rows: _fillet_judge(rows)["checks"]["crater_filled"]["pass"]
        self.assertTrue(_fillet_judge(self._two_runs_meeting())["success"])
        self.assertFalse(judge(self._two_runs_meeting((77.75, 82.75, 5.7))))
        self.assertFalse(judge(self._two_runs_meeting((80.25, 82.75, 5.7))))
        self.assertFalse(judge(self._two_runs_meeting((80.25, 82.75, 5.7), b_end=83.25, a=False)))   # a bead end 11 mm on
        self.assertTrue(judge(self._two_runs_meeting((77.75, 82.75, 6.45))))
        # run A's end beside B's later metal is not a stop: A's metal there is judged by B's window and the size rule
        self.assertNotIn(83.25, [s for c in _fillet_judge(self._two_runs_meeting())["checks"]["crater_filled"]["craters"]
                                 for s in c["stop_mm"]])

    def test_a_restart_more_than_two_seconds_after_the_earlier_metal_froze_leaves_its_crater_judged(self):
        """A restart struck at an earlier pass's front meets it about 4 s apart (3.7-3.8 s strike to first freeze); a
        freeze-time step of more than two seconds splits the passes, so the earlier pass's unfilled crater stays judged.
        Inside the joint the largest step within one pass is 0.79 s."""
        ta = lambda x: x if x <= 78.25 else 78.25 - .05*(x - 78.25)
        def rows(gap):
            t0 = ta(80.25) + gap
            return _fillet_rows([(.25, 80.25, ta), (80.75, 149.75, lambda x: t0 + x - 80.75)],
                                sizes=[(75.75, 80.25, 5.7)])
        for gap, judged in ((4.9, True), (4.0, True), (2.5, True), (1.0, False)):
            with self.subTest(gap=gap):
                self.assertEqual(_fillet_judge(rows(gap))["checks"]["crater_filled"]["pass"], not judged)
        # a step beyond a bracket end never splits a pass
        r = _fillet_judge(_fillet_rows([(-.75, -.25, lambda x: x - 3.), (.25, 149.75, lambda x: x)]))
        self.assertTrue(r["success"], r["checks"])

    def test_a_later_pool_reaching_a_stop_from_behind_is_a_documented_residual(self):
        """A known false positive: a later pool whose metal reaches run B's stop from the side B came
        from (touching or remelting its last-frozen section) leaves the rest of that crater, toward B's front, judged
        only by the 10 % undersize rule. If this assertion starts failing, the residual has been closed."""
        u = self._two_runs_meeting((72.75, 77.25, 5.7))
        self.assertFalse(_fillet_judge(u)["checks"]["crater_filled"]["pass"])
        for front in (71.75, 72.25):
            d = _fillet_rows([(front - 10., front, lambda x, f=front: 900. + x if x <= f - 2.5 else 900. + f - 2.5)],
                             over=u)
            with self.subTest(front=front):
                self.assertTrue(_fillet_judge(d)["checks"]["crater_filled"]["pass"])
        d = _fillet_rows([(59.75, 69.75, lambda x: 900. + x if x <= 67.25 else 967.25)], over=u)
        self.assertFalse(_fillet_judge(d)["checks"]["crater_filled"]["pass"])

    def test_cell_edge_readings_at_their_limits_are_on_the_limit(self):
        """Readings on a cell edge count as on the limit: one cell into the base plate reads 0.2499999999999933 mm and
        eight cells of undercut 2.0000000000000157."""
        for a, b, edit in ((-2., 152., dict(penetration_base_m=.2499999999999933e-3)),
                           (60., 70., dict(undercut_bracket_m=2.0000000000000157e-3)),
                           (-2., 152., dict(undercut_bracket_m=1.0000000000000148e-3))):
            with self.subTest(edit):
                self.assertTrue(_fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(a, b, edit)]))["success"])

    def test_unequal_legs_and_oversize_are_accepted(self):
        """D1.1 accepts unequal legs when the smaller meets the size, and an oversized fillet."""
        self.assertTrue(_fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], legs=(6.6, 10.)))["success"])
        self.assertTrue(_fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], size=12.))["success"])

    def test_end_zones_judge_the_weld_and_bound_a_starts_lap(self):
        """Porosity, undercut, convexity and craters are judged on the whole joint; a start's cold lap within 5 mm of an
        end is accepted up to two cells (0.5 mm), never an 8 mm lap; the stop end is a crater and judged at one cell."""
        cases = ((dict(porosity=.1), (.25, 4.75), False), (dict(undercut_base_m=3e-3), (.25, 4.75), False),
                 (dict(overlap_base_m=.5e-3), (.25, 4.75), True), (dict(overlap_base_m=8e-3), (.25, 4.75), False),
                 (dict(convexity_m=4e-3), (.25, 4.75), False), (dict(overlap_base_m=.5e-3), (145.25, 149.75), False),
                 (dict(convexity_m=4e-3), (145.25, 149.75), False), (dict(porosity=.1), (145.25, 149.75), False))
        for edit, (a, b), ok in cases:
            ev = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(a, b, edit)]))
            with self.subTest(edit=edit, at=(a, b)):
                self.assertEqual(ev["success"], ok)
        ev = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(.25, 4.75, dict(overlap_base_m=.5e-3))]))
        self.assertEqual(ev["diagnostics"]["start_zone"]["overlap_max_mm"], .5)

    def test_a_crater_a_later_pass_did_not_remelt_stays_judged(self):
        """A restart that stops short of an earlier crater's last-frozen section leaves it judged (it froze in
        another pass); one that remelts it makes it its own. Run A goes -x to 75 mm, its last metal freezing at 78 mm
        with a front to 72.5 mm and its crater 4.5 mm over 75.75-80.25 mm; restart C runs -x from A's front."""
        def rows(reach):
            a_t = lambda x: 150. - x if x >= 78. else 72. - .1*(78. - x)
            r = _fillet_rows([(reach + .5, 149.75, a_t), (.25, reach, lambda x: 1000. + (reach - x)*.2)],
                             sizes=[(max(75.75, reach + .5), 80.25, 4.5)])
            return _fillet_judge(r)["checks"]["crater_filled"]["pass"]
        self.assertFalse(rows(75.25))          # C's pool back to 75.25 mm, 2.5 mm short of A's last-frozen metal
        self.assertFalse(rows(76.25))
        self.assertTrue(rows(80.75))           # C remelted A's crater: it is C's metal now
        # residual, stated in task.py and welding/REPORT.pdf §13: a pool that remelts the crater's front up to its
        # last-frozen section (reach 77.25 here) leaves that section at an episode end next to later metal, which is not
        # a stop
        self.assertTrue(rows(77.25))

    def test_a_crater_judges_its_own_pool_and_runs_on_to_the_beads_end(self):
        """A sound touch-up next to code-allowed 6.25 mm metal it did not remelt passes; a fall-off between a stop and
        a bead end 7 mm on is judged; a size-0 hole at a stop does not hide it."""
        touch = _fillet_rows([(.25, 55.75, lambda x: x), (56.25, 66.25, lambda x: 900. + x), (66.75, 149.75, lambda x: x)],
                             sizes=[(66.75, 68.75, 6.25)])
        self.assertTrue(_fillet_judge(touch)["success"])
        stop_t = lambda x: x if x <= 140.75 else 140.75 - (x - 140.75)*.05
        fall = _fillet_rows([(.25, 147.75, stop_t)], sizes=[(146.25, 146.75, 4.)])
        self.assertFalse(_fillet_judge(fall)["checks"]["crater_filled"]["pass"])
        self.assertTrue(_fillet_judge(_fillet_rows([(.25, 147.75, stop_t)]))["success"])
        hole = _fillet_rows([(.25, 149.75, lambda x: x)], edits=[(146.25, 146.75, dict(size_m=0.))])
        self.assertFalse(_fillet_judge(hole)["checks"]["crater_filled"]["pass"])

    def test_boundaries_of_the_allowances_are_on_the_allowance(self):
        """A weld stopping exactly 5.0 mm short of either end (the real grid's slice edges), and exactly 10 %
        undersize, are accepted."""
        from welding.process import build_grid, tee_coupon
        g, _ = build_grid(tee_coupon())
        xs = [float(x)*1e3 for x in g.xc if -2e-3 <= x <= .152]
        for lo, hi in ((5.25, 149.75), (.25, 144.75)):
            rows = _fillet_rows([(lo, hi, lambda x: x)])
            for r, x in zip(rows, xs):
                r["x_m"] = x*1e-3
            with self.subTest(span=(lo, hi)):
                self.assertTrue(_fillet_judge(rows)["checks"]["continuous_length"]["pass"])
        ten = _fillet_rows([(.25, 149.75, lambda x: x)], sizes=[(30., 43.75, 6.)])        # 14.0 of 140 mm
        self.assertTrue(_fillet_judge(ten)["checks"]["leg_size"]["pass"])

    def test_a_weld_still_freezing_is_not_yet_accepted(self):
        """Weld metal with no freeze time (its slice still liquid in the pool) withholds success."""
        rows = _fillet_rows([(.25, 149.75, lambda x: x)], edits=[(140.25, 149.75, dict(freeze_t=None))])
        ev = _fillet_judge(rows)
        self.assertFalse(ev["checks"]["solidified"]["pass"])
        self.assertEqual(ev["checks"]["solidified"]["freezing_sections"], 20)

    def test_progress_is_the_sound_fillet_share_and_success_scores_one(self):
        """The score is 1 for an accepted weld, else 0.9 x the sound, fused fillet's share of the required span net of
        the start; idle is 0, more sound weld or more size never lowers it."""
        from welding.task import FAIL_SCORE_SCALE
        start = _fillet_rows([(8.25, 22.75, lambda x: x), (127.25, 141.75, lambda x: x)], size=4.5)
        base = _fillet_judge(start, start)
        self.assertEqual((base["success"], base["score"], base["progress"]), (False, 0., 0.))
        prev = -1.
        for x1 in (20., 50., 100., 140.):
            ev = _fillet_judge(_fillet_rows([(.25, x1, lambda x: x)], over=start), start)
            with self.subTest(x1=x1):
                self.assertFalse(ev["success"])
                self.assertGreater(ev["progress"], prev)
                self.assertAlmostEqual(ev["score"], FAIL_SCORE_SCALE*ev["progress"])
            prev = ev["progress"]
        prev = -1.
        for s in (1., 3., 5., 6.3):
            ev = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], size=s), start)
            self.assertGreater(ev["progress"], prev)
            prev = ev["progress"]
        self.assertEqual(_fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)]), start)["score"], 1.)
        porous = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(-2., 152., dict(porosity=.15))]), start)
        self.assertEqual((porous["success"], porous["progress"]), (False, 0.))
        self.assertIn("mean_of_checks", porous["reward"])
        # undercut the code accepts (1.5 mm on 50 mm) costs nothing; beyond it, only the excess
        ok = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(50.25, 99.75, dict(undercut_base_m=1.5e-3))]),
                           start)
        self.assertEqual((ok["success"], ok["progress"]), (True, 1.))
        over = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], edits=[(50.25, 100.25, dict(undercut_base_m=1.5e-3))]),
                             start)
        self.assertFalse(over["success"])
        self.assertGreater(over["score"], .89)
        # an unfilled crater at a bracket end costs progress, as it does mid-seam
        end = _fillet_judge(_fillet_rows([(.25, 149.75, lambda x: x)], sizes=[(145.25, 149.75, 3.)]), start)
        self.assertFalse(end["success"])
        self.assertLess(end["progress"], .99)

    def test_weld_metal_lying_unfused_is_not_part_of_a_sound_tack(self):
        from welding.task import Evaluator, TACK_END_SECTIONS
        rows = [dict(x_m=(k + .5)*5e-4, dx_m=5e-4, weld_area_m2=1e-5 if 4 <= k < 48 else 0.,
                     root_fused=k >= 16, penetration_base_m=5e-4, penetration_bracket_m=5e-4) for k in range(60)]
        runs = Evaluator._weld_runs(None, rows)
        self.assertEqual(len(runs), 1)
        inner = runs[0][TACK_END_SECTIONS:len(runs[0]) - TACK_END_SECTIONS]
        self.assertEqual(sum(not Evaluator._fused(r) for r in inner), 16 - 4 - TACK_END_SECTIONS)


def _tacked_start():
    """Constructed tacked start (no physics): the fixture's two 4.5 mm tacks at 8-23 and 127-142 mm, frozen in order."""
    return _fillet_rows([(8.25, 22.75, lambda x: x), (127.25, 141.75, lambda x: x)], size=4.5)


def _task_evaluator(task, start):
    """An evaluator of `task` with its start state's sections `start` (no world), as WeldingEnv builds it."""
    from welding.task import Evaluator, fillet_progress
    ev = Evaluator(SimpleNamespace(config=SimpleNamespace(task=task), world=SimpleNamespace(process=None)))
    ev.start_rows = start
    ev.start_raw_progress = fillet_progress(start)["raw"] if start is not None else None
    return ev


def _failed(result):
    return sorted(k for k, c in result["checks"].items() if not c["pass"])


class EightTaskSetTest(unittest.TestCase):
    """The eight-task set (welding/REPORT.pdf §13): the catalogue's wiring, and every judgement and progress term the
    set adds, on constructed sections and poses. No physics, no GPU."""

    def test_catalogue_task_codes_and_the_eight_task_set(self):
        from welding.task import (EIGHT_TASKS, INSTRUCTIONS, NEEDS_HAND, START_PRESET, START_STATE, TASKS, WELD_TASKS)
        from welding.variants import check_robot
        self.assertEqual(list(TASKS)[:3], ["position-torch", "tack", "fillet-2f"])
        self.assertEqual([TASKS[t] for t in ("position-torch", "tack", "fillet-2f")], ["A0", "A1", "A2"])
        self.assertEqual(len(set(TASKS.values())), len(TASKS))
        self.assertFalse({"A3", "A4", "A5", "A6"} & set(TASKS.values()))          # reserved for the unbuilt tasks
        self.assertEqual(len(EIGHT_TASKS), 8)
        # registered, kept outside the set: none can be shown to succeed as built (welding/REPORT.pdf §13)
        self.assertEqual(set(TASKS) - set(EIGHT_TASKS), {"tack", "stitch-weld-ends", "fillet-from-both-ends"})
        raw = json.loads((ROOT/"welding"/"benchmark.json").read_text())
        self.assertEqual(raw["task_set"]["tasks"], list(EIGHT_TASKS))
        self.assertEqual(INSTRUCTIONS["fillet-2f"], (
            "Weld a continuous 1/4 inch fillet along the near side of the tacked bracket in the horizontal position, "
            "from end to end, and fill the crater. The spray-transfer schedule for 0.035 inch ER70S-6 wire with "
            "90/10 gas is about 500 in/min (12.7 m/min) wire feed at 29 V."))
        hand = {"tack", "fit-up-bracket", "tack-and-weld", "hold-and-present"}
        self.assertEqual(NEEDS_HAND, hand)
        self.assertEqual(set(START_PRESET), {"set-up-gun", "fit-up-bracket", "set-up-and-weld"})
        self.assertEqual({t for t in TASKS if START_STATE[t] == "loose"}, hand)
        self.assertEqual(set(WELD_TASKS), {"fillet-2f", "half-fillet", "stitch-weld-ends", "tack-and-weld",
                                           "fillet-from-both-ends", "set-up-and-weld"})
        for task in ("fit-up-bracket", "tack-and-weld", "hold-and-present"):
            check_robot("ur10e-fr3", task)
            for robot in ("ur10e", "single-fr3", "g1"):
                with self.subTest(task=task, robot=robot), self.assertRaises(ValueError):
                    check_robot(robot, task)
        for robot in ("dual-fr3", "dual-panda"):          # the arms met there with the torch at the seam's start
            check_robot(robot, "fit-up-bracket")
            with self.subTest(robot=robot), self.assertRaisesRegex(ValueError, "not offered"):
                check_robot(robot, "hold-and-present")

    def test_graded_terms(self):
        from welding.task import graded, net_of_start
        self.assertEqual((graded(.5, 1., 1.), graded(1.5, 1., 1.), graded(3., 1., 1.), graded(None, 1., 1.)),
                         (1., .5, 0., 0.))
        self.assertEqual((net_of_start(10., 10., 1.), net_of_start(.5, 10., 1.), net_of_start(5.5, 10., 1.)), (0., 1., .5))
        self.assertIsNone(net_of_start(3., .8, 1.))           # already in band at the start: the term earns nothing

    def test_position_torch_progress(self):
        from welding.task import position_progress, seam_start_distance
        self.assertAlmostEqual(seam_start_distance((5e-3, 3e-3, 4e-3)), 5e-3)
        self.assertAlmostEqual(seam_start_distance((13e-3, 0., 4e-3)), 5e-3)
        d0 = .2
        p = lambda d, **kw: position_progress(d, d0, kw.get("work", 45.), kw.get("travel", 7.5), kw.get("ctwd", 19e-3),
                                              kw.get("settled", .1), kw.get("faulted", False))["progress"]
        self.assertEqual(p(d0), 0.)                                         # untouched start
        self.assertEqual(p(.5e-3), 1.)
        self.assertEqual(p(.5e-3, faulted=True), .5)
        self.assertAlmostEqual(p(.5e-3, work=55.), .875)                    # 5 deg outside the band: half the term
        self.assertAlmostEqual(p(.5e-3, ctwd=None, settled=None), .5)
        prev = -1.
        for d in (.15, .1, .05, .01, .001):
            self.assertGreater(p(d), prev)
            prev = p(d)

    def test_set_up_gun_progress(self):
        from welding.task import setup_progress
        start = dict(wfs_m_min=6.4, voltage_v=20., stickout_m=6.35e-3)
        r = lambda wfs, v, s, start=start, faulted=False: setup_progress(wfs, v, s, start, faulted)
        self.assertEqual(r(6.4, 20., 6.35e-3)["progress"], 0.)
        self.assertEqual(r(12.7, 29., 12.7e-3)["progress"], 1.)
        self.assertEqual(r(12.7, 29., 12.7e-3, faulted=True)["progress"], .5)
        self.assertAlmostEqual(r(12.7, 20., 12.7e-3)["progress"], 2/3)
        in_band = dict(start, stickout_m=12e-3)                             # a variation already at 1/2 in of wire
        self.assertIsNone(r(12.7, 29., 12e-3, start=in_band)["terms"]["stickout"])
        self.assertEqual(r(12.7, 29., 12e-3, start=in_band)["progress"], 1.)

    @staticmethod
    def _coupon_frames():
        import numpy as np
        from welding.process import tee_coupon
        c = tee_coupon()
        base_c = np.array([c.base_x0 + c.base[0]/2, (c.base_y[0] + c.base_y[1])/2, -c.base[2]/2])
        web_c = np.array([c.length/2, -c.web[1]/2, c.web[2]/2])
        return base_c, web_c

    def _fitup(self, shift=(0., 0., 0.), yaw=0., tilt=0.):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from welding.task import fitup_errors
        base_c, web_c = self._coupon_frames()
        Rk = Rotation.from_euler("zx", [yaw, tilt], degrees=True).as_matrix()
        return fitup_errors(base_c, np.eye(3), web_c + np.asarray(shift), Rk, -base_c, -web_c, web_c - base_c)

    def test_fit_up_errors_displacement_and_progress(self):
        import math
        from welding.task import FITUP_SEED0, fitup_displacement, fitup_progress
        e = self._fitup()
        self.assertAlmostEqual(e["root_edge_mm"], 0., places=9)
        self.assertAlmostEqual(self._fitup(shift=(0., .025, 0.))["root_edge_mm"], 25.)
        e = self._fitup(yaw=6.)                                 # about the centre: a centre offset alone misses it
        self.assertAlmostEqual(e["centre_mm"], 0., places=9)
        self.assertGreater(e["root_edge_mm"], 75*math.sin(math.radians(6.)))
        self.assertAlmostEqual(self._fitup(tilt=.8)["tilt_deg"], .8)
        self.assertEqual(fitup_displacement(0), (FITUP_SEED0[0]*1e-3, FITUP_SEED0[1]))
        for seed in (1, 9, 100001, 109999):
            shift, yaw = fitup_displacement(seed)
            self.assertEqual((shift, yaw), fitup_displacement(seed))
            self.assertTrue(.015 <= shift <= .030 and 3. <= abs(yaw) <= 8., (seed, shift, yaw))
        start = self._fitup(shift=(0., .025, 0.), yaw=6.)
        self.assertEqual(fitup_progress(start, start, False)["progress"], 0.)
        self.assertEqual(fitup_progress(self._fitup(), start, False)["progress"], 1.)
        self.assertEqual(fitup_progress(self._fitup(), start, True)["progress"], 0.)       # held is not placed
        half = fitup_progress(self._fitup(shift=(0., .0125, 0.), yaw=3.), start, False)["progress"]
        self.assertTrue(.3 < half < .7, half)
        self.assertAlmostEqual(fitup_progress(self._fitup(tilt=10.), start, False)["progress"], 0.)   # toppled

    def test_tack_progress(self):
        from welding.task import tack_progress
        tacks = [dict(x0_mm=8., x1_mm=23., fused_mm=15.), dict(x0_mm=127., x1_mm=142., fused_mm=15.)]
        self.assertEqual(tack_progress(tacks, False, 0., 0., False, 1.)["progress"], 0.)
        self.assertEqual(tack_progress(tacks, True, .2, .3, False, 1.)["progress"], 1.)
        self.assertEqual(tack_progress(tacks[:1], True, .2, .3, False, 1.)["progress"], .75)

    def test_half_fillet(self):
        from welding.task import FAIL_SCORE_SCALE
        start = _tacked_start()
        ev = _task_evaluator("half-fillet", start)
        base = ev.judge_half_fillet(start)
        self.assertEqual((base["success"], base["score"]), (False, 0.))
        done = ev.judge_half_fillet(_fillet_rows([(.25, 79.75, lambda x: 100. + x)], over=start))
        self.assertTrue(done["success"], _failed(done))             # the second tack's unfilled crater is the fixture's
        crater = ev.judge_half_fillet(_fillet_rows([(.25, 79.75, lambda x: 100. + x)], over=start,
                                                   sizes=[(75.25, 79.75, 4.5)]))
        self.assertEqual(_failed(crater), ["crater_filled"])
        self.assertAlmostEqual(crater["score"], FAIL_SCORE_SCALE*crater["progress"])
        self.assertTrue(0. < crater["progress"] < 1.)
        over = ev.judge_half_fillet(_fillet_rows([(.25, 109.75, lambda x: 100. + x)], over=start))
        self.assertEqual(_failed(over), ["left_half_unwelded"])
        prev = -1.
        for x1 in (20., 40., 60., 75.):
            p = ev.judge_half_fillet(_fillet_rows([(.25, x1, lambda x: 100. + x)], over=start))["progress"]
            self.assertGreater(p, prev)
            prev = p

    def test_stitch_weld_ends(self):
        start = _tacked_start()
        ev = _task_evaluator("stitch-weld-ends", start)
        self.assertEqual(ev.judge_stitch_weld_ends(start)["score"], 0.)
        stitches = [(.25, 57.75, lambda x: 100. + x), (92.25, 149.75, lambda x: 300. + x)]
        # the right stitch's unfilled stop lies outside its 2 in: D1.1 Table 6.1(3) exempts it
        done = ev.judge_stitch_weld_ends(_fillet_rows(stitches, over=start, sizes=[(53.25, 57.75, 4.5)]))
        self.assertTrue(done["success"], _failed(done))
        short = [(.25, 52.75, lambda x: 100. + x), stitches[1]]          # stopped 2.75 mm past the 2 in: judged there
        inside = ev.judge_stitch_weld_ends(_fillet_rows(short, over=start, sizes=[(47.25, 52.75, 4.5)]))
        self.assertEqual(_failed(inside), ["crater_filled"])
        middle = ev.judge_stitch_weld_ends(_fillet_rows(stitches + [(60.25, 89.75, lambda x: 500. + x)], over=start))
        self.assertEqual(_failed(middle), ["middle_unwelded"])
        self.assertLess(middle["progress"], done["progress"])
        # 5.5 mm under size past 5 mm (the aim_tacks start) is over 10 % of a 45 mm window
        slow = ev.judge_stitch_weld_ends(_fillet_rows(stitches, over=start, sizes=[(5.25, 10.25, 6.)]))
        self.assertEqual(_failed(slow), ["leg_size_right_end"])
        one = ev.judge_stitch_weld_ends(_fillet_rows(stitches[:1], over=start))
        self.assertTrue(.4 < one["progress"] < .6, one["progress"])

    def test_fillet_from_both_ends(self):
        start = _tacked_start()
        ev = _task_evaluator("fillet-from-both-ends", start)
        self.assertEqual(ev.judge_fillet_from_both_ends(start)["score"], 0.)
        two = ev.judge_fillet_from_both_ends(_fillet_rows([(83.25, 149.75, lambda x: 300. + 150. - x),
                                                           (.25, 82.75, lambda x: 500. + x)], over=start))
        self.assertTrue(two["success"], _failed(two))
        self.assertEqual(two["checks"]["two_runs_from_the_ends"]["meet_mm"], 83.)
        single = ev.judge_fillet_from_both_ends(_fillet_rows([(.25, 149.75, lambda x: x + 100.)], over=start))
        self.assertEqual(_failed(single), ["two_runs_from_the_ends"])
        self.assertLess(single["progress"], two["progress"])
        self.assertGreater(single["reward"]["lost_mm"]["wrong_direction"], 40.)
        outward = ev.judge_fillet_from_both_ends(_fillet_rows([(75.25, 149.75, lambda x: 100. + x),
                                                               (.25, 74.75, lambda x: 500. + 75. - x)], over=start))
        self.assertIn("two_runs_from_the_ends", _failed(outward))
        off_middle = ev.judge_fillet_from_both_ends(_fillet_rows([(30.25, 149.75, lambda x: 300. + 150. - x),
                                                                  (.25, 29.75, lambda x: 500. + x)], over=start))
        self.assertIn("two_runs_from_the_ends", _failed(off_middle))

    def test_tack_and_weld(self):
        bare = _fillet_rows([])
        ev = _task_evaluator("tack-and-weld", bare)
        ok = dict(bond_active=True, square_deg=.2, offset_mm=.3, hand_touching=False)
        self.assertEqual(ev.judge_tack_and_weld(bare, **ok)["score"], 0.)
        weld = _fillet_rows([(.25, 149.75, lambda x: x)])
        done = ev.judge_tack_and_weld(weld, **ok)
        self.assertTrue(done["success"], _failed(done))
        held = ev.judge_tack_and_weld(weld, **dict(ok, hand_touching=True))
        self.assertEqual(_failed(held), ["hand_off"])
        self.assertAlmostEqual(held["progress"], .5*done["progress"])
        skew = ev.judge_tack_and_weld(weld, **dict(ok, square_deg=1.5))
        self.assertEqual(_failed(skew), ["square"])
        self.assertAlmostEqual(skew["reward"]["alignment"], .5)

    def test_hold_and_present(self):
        from welding.task import Evaluator, hold_present_progress
        self.assertEqual(hold_present_progress(0., False, 0., .03)["progress"], 0.)          # untouched loose start
        self.assertEqual(hold_present_progress(1., True, .06, .07)["progress"], 1.)
        self.assertEqual(hold_present_progress(1., False, .06, .07)["progress"], .5)         # presented, not held
        self.assertEqual(hold_present_progress(0., True, .06, .07)["progress"], .5)          # held, torch at home
        self.assertAlmostEqual(hold_present_progress(1., True, 1.5, .07)["progress"], .75)   # held 1.5 deg out of square
        # the hold needs both fingers on the bracket: a hand resting on it, or one finger pushing it, is not a grip
        bodies = ["base_plate", "left/fr3_link7", "left/fr3_hand", "left/fr3_leftfinger", "left/fr3_rightfinger",
                  "bracket"]

        def fingers(touching):
            world = SimpleNamespace(scene=SimpleNamespace(bodies={"bracket": 5}), hand_arms=("left",),
                                    model=SimpleNamespace(body_label=bodies), hand_bodies=lambda: [1, 2, 3, 4],
                                    bodies_touch=lambda body, others: bool(others & touching))
            return Evaluator(SimpleNamespace(config=SimpleNamespace(task="hold-and-present"),
                                             world=world))._fingers_on_bracket()
        self.assertEqual(fingers({3, 4}), [True, True])
        self.assertEqual(fingers({3}), [True, False])
        self.assertEqual(fingers({2}), [False, False])

    def test_set_up_and_weld(self):
        start = _tacked_start()
        ev = _task_evaluator("set-up-and-weld", start)
        weld = _fillet_rows([(.25, 149.75, lambda x: x + 100.)], over=start)
        self.assertEqual(ev.judge_set_up_and_weld(start, wfs_m_min=6.4, voltage_v=20.)["score"], 0.)
        done = ev.judge_set_up_and_weld(weld, wfs_m_min=12.7, voltage_v=29.)
        self.assertTrue(done["success"], _failed(done))
        self.assertEqual(done["score"], 1.)
        left = ev.judge_set_up_and_weld(weld, wfs_m_min=6.4, voltage_v=20.)
        self.assertEqual(_failed(left), ["source_at_procedure"])
        self.assertLess(left["score"], 1.)
        # the same weld and progress as fillet-2f judges them
        fillet = _fillet_judge(weld, start)
        self.assertEqual(fillet["success"], True)
        self.assertAlmostEqual(left["progress"], fillet["progress"])
        self.assertTrue(ev.judge_set_up_and_weld(weld, wfs_m_min=12.95, voltage_v=28.6)["success"])
        self.assertFalse(ev.judge_set_up_and_weld(weld, wfs_m_min=13.1, voltage_v=29.)["success"])

    def test_fillet_progress_unchanged_without_eligibility(self):
        from welding.task import fillet_progress
        rows = _fillet_rows([(.25, 120., lambda x: x)], over=_tacked_start(), sizes=[(30., 40., 5.)])
        plain, every = fillet_progress(rows), fillet_progress(rows, eligible=lambda r: True)
        self.assertNotIn("wrong_direction", plain["lost_mm"])
        self.assertEqual(every["lost_mm"].pop("wrong_direction"), 0.)
        self.assertEqual(plain, every)

    def test_presets_are_applied_before_a_variation_and_required(self):
        from welding.presets import apply_start_preset
        from welding.task import Evaluator
        self.assertIsNone(apply_start_preset(SimpleNamespace(config=SimpleNamespace(task="position-torch"))))
        for task in ("set-up-gun", "fit-up-bracket", "set-up-and-weld"):
            ev = Evaluator(SimpleNamespace(config=SimpleNamespace(task=task), world=SimpleNamespace(process=None)))
            with self.subTest(task=task), self.assertRaisesRegex(RuntimeError, "preset"):
                ev.evaluate()
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        b = object.__new__(NewtonWeldingBackend)
        b.output = None
        calls = []
        with patch("welding.env.WeldingEnv", side_effect=lambda c, output_dir=None: calls.append("env") or c), \
                patch("welding.presets.apply_start_preset", side_effect=lambda env: calls.append("preset")), \
                patch("welding.variations.apply_initial", side_effect=lambda env, v: calls.append("variation")):
            b._make_env(SimpleNamespace(task="fit-up-bracket"), {"variation_id": 3})
        self.assertEqual(calls, ["env", "preset", "variation"])


class OverhangSurfaceTest(unittest.TestCase):
    """The pool's overhanging surface (pool._PolarSurface) against the weightless 2-D drop pinned on a
    chord, and the evaluator's overlap reading of a face that stands over unfused metal: no physics."""

    def test_weightless_drop_is_young_laplace(self):
        import numpy as np
        from scipy.optimize import brentq
        from welding import pool as PL
        L, gamma = 4e-3, 1.2
        A, t, n = np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.])
        for depth in (0., 1e-3):
            for k in (.5, 1.3, 2.0):                          # times the half-disc on the chord: past it the face overhangs
                S = PL._PolarSurface(A, t, n, L, depth, 0., (0., 0.), 7000., gamma)
                V = k*math.pi*(L/2)**2/2
                r, lam, ok = S.solve(V + .5*L*depth, np.full(PL.N_NODES + 1, S.r_toe), np.zeros(PL.N_NODES + 1),
                                     np.ones(PL.N_NODES + 1))
                seg = lambda th: ((L/2)/math.sin(th))**2*(th - math.sin(th)*math.cos(th))
                th = brentq(lambda th: seg(th) - V, 1e-6, math.pi - 1e-6)
                with self.subTest(depth=depth, k=k):
                    self.assertTrue(ok)
                    self.assertAlmostEqual(S.area(r) - .5*L*depth, V, delta=1e-9*V)
                    self.assertAlmostEqual(lam, gamma*math.sin(th)/(L/2), delta=.01*gamma*math.sin(th)/(L/2))

    def test_face_over_unfused_plate_is_overlap(self):
        """Weld metal standing over the plate beyond the toe with an air wedge under it (an overhang) is overlap: D1.1
        Fig 5.4's weld metal protruding beyond the toe."""
        import numpy as np
        from welding import sections as SC
        from welding import thermal as TH
        ye = np.arange(-4e-3, 12e-3 + 1e-9, 2.5e-4)
        yc = .5*(ye[1:] + ye[:-1])
        Y, Z = np.meshgrid(yc, yc, indexing="ij")
        orig = (Y < 0) | (Z < 0)
        fillet = (Y > 0) & (Z > 0) & (Y + Z < 4e-3)
        lip = (Y > 4e-3) & (Y < 5.5e-3) & (Z > .5e-3) & (Z < 1.5e-3)       # past the toe, 0.5 mm of air under it
        mat = np.where(orig, 1, np.where(fillet | lip, TH.WELD, 0)).astype(np.int8)
        peakf = np.where(orig & (np.hypot(Y, Z) < 4e-3), 1., 0.)
        r = SC.tee_section(mat, orig, peakf, np.zeros_like(peakf), ye, ye)
        self.assertGreaterEqual(r["overlap_base_m"], 1.25e-3)
        r0 = SC.tee_section(np.where(lip, 0, mat).astype(np.int8), orig, peakf, np.zeros_like(peakf), ye, ye)
        self.assertEqual(r0["overlap_base_m"], 0.)


class RegistrationTest(unittest.TestCase):
    def test_registered_with_aliases_and_descriptor(self):
        from gym_anything_robotics import registry
        from gym_anything_robotics.launch import describe
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        from gym_anything_robotics.backends.base import RoboticsBackend
        self.assertTrue(issubclass(NewtonWeldingBackend, RoboticsBackend))
        self.assertIn(("welding", "newton"), registry._REGISTRY)
        for alias in ALIASES:
            d = describe(alias)
            self.assertEqual((d["environment_id"], d["default_robot"], d["implementation_package"],
                              d["engine_version"]), ("welding", "ur10e-fr3", "welding", "1.5"))
            self.assertEqual([r["id"] for r in d["robots"]], ["ur10e-fr3", "ur10e", "dual-fr3", "single-fr3",
                                                               "dual-panda", "g1"])

    def test_setup_objects_of_the_three_rigs_accepted_others_refused(self):
        # the rigs by id, as gar.robot-setup.v1 objects and as rig paths select their alias
        from robots.catalog import for_environment, resolve, select_for_backend
        from gym_anything_robotics.launch import LaunchSpec
        for alias, rig in (("ur10e-fr3", "welding-ur10e-fr3"), ("ur10e", "welding-ur10e"), ("dual-fr3", "welding-fr3"),
                           ("single-fr3", "welding-fr3-single")):
            self.assertEqual(for_environment("welding", alias).to_dict()["id"], rig)
            for sel in (rig, resolve(rig).to_dict(), str(PACKAGES / "robots" / "rigs" / f"{rig}.json")):
                with self.subTest(sel=str(sel)[:40]):
                    self.assertEqual(select_for_backend("welding", sel), alias)
            LaunchSpec("welding", robot=resolve(rig).to_dict())
        with self.assertRaises(ValueError) as caught:
            select_for_backend("welding", resolve("filleting-fr3").to_dict())
        self.assertIn("hosts its six catalog rigs", str(caught.exception))

    def test_create_passes_the_catalog_default_and_task_options(self):
        from gym_anything_robotics import registry
        factory = Mock()
        spec = tasks.resolve("welding", {"task": "tack", "seed": 2})
        with patch.dict(registry._REGISTRY, {("welding", "newton"): factory}), \
                patch.object(registry, "RoboticsEnv"):
            spec.create()
            registry.create("gmaw", cameras=())
        self.assertEqual(factory.call_args_list[0].kwargs,
                         {"robot": "ur10e-fr3", "preset": "steel-cell", "task": "tack", "seed": 2})
        self.assertEqual(factory.call_args_list[1].kwargs, {"robot": "ur10e-fr3", "cameras": ()})

    def test_unsupported_robots_rejected_with_catalog_reasons(self):
        from robots.catalog import ENVIRONMENTS
        from gym_anything_robotics.launch import LaunchSpec
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        reasons = ENVIRONMENTS["welding"].get("unsupported", {})
        self.assertEqual(set(reasons), set())          # g1 and dual-panda are deployments
        for robot, reason in reasons.items():
            with self.subTest(robot=robot):
                with self.assertRaises(ValueError) as caught:
                    LaunchSpec("welding", robot=robot)
                self.assertIn(f"does not support robot {robot!r}", str(caught.exception))
                self.assertIn(reason, str(caught.exception))
                with patch("welding.env.WeldingEnv") as env, self.assertRaises(UnsupportedConfigurationError) as caught:
                    NewtonWeldingBackend(robot=robot)
                self.assertIn(reason, str(caught.exception))
                env.assert_not_called()
        for robot in ({"instances": {}}, "yam", "single-panda"):
            with self.subTest(robot=robot), self.assertRaises(UnsupportedConfigurationError):
                NewtonWeldingBackend(robot=robot)

    def test_torch_only_rigs_refused_for_the_tack_task(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        for robot in ("ur10e", "single-fr3"):
            with self.subTest(robot=robot), patch("welding.env.WeldingEnv") as env, \
                    self.assertRaises(UnsupportedConfigurationError) as caught:
                NewtonWeldingBackend(robot=robot, task="tack")
            self.assertIn("needs the FR3's hand", str(caught.exception))
            env.assert_not_called()

    def test_invalid_configuration_fails_before_the_world_is_built(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        bad = (dict(cameras=("hero",)), dict(cameras=("seam",)), dict(cameras=("bench", "bench")),
               dict(config={"task": "tack"}), dict(config={"robot": "ur10e"}), dict(config={"phantom_torch": True}),
               dict(config={"device": "cpu"}, device="cuda:0"), dict(renderer_options={"exposure": 2}),
               dict(seed=True), dict(seed=1.5), dict(seed=10), dict(task="pipe-6g"), dict(preset="copper-brazing"),
               dict(config={"arc_magic": 1}))
        with patch("welding.env.WeldingEnv") as env:
            for options in bad:
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError,
                                                                        UnsupportedConfigurationError)):
                    NewtonWeldingBackend(output_dir="/nonexistent/never-created", **options)
            env.assert_not_called()


class VariationCatalogueTest(unittest.TestCase):
    """The frozen catalogue welding-steel-cell-v1: identity, hash, splits, bounds, variation 0, selection, refusals."""

    @classmethod
    def setUpClass(cls):
        from welding import variations as VR
        cls.VR = VR
        cls.path = ROOT/"welding"/"data"/"variations"/"catalogue.jsonl"
        cls.payload = cls.path.read_bytes()
        cls.manifest = json.loads(cls.path.with_suffix(".manifest.json").read_text())
        cls.rows = [json.loads(line) for line in cls.payload.decode().splitlines()]

    def test_frozen_file_hash_manifest_and_rederivation(self):
        import hashlib
        VR = self.VR
        self.assertEqual(len(self.rows), VR.COUNT)
        self.assertEqual(self.manifest["count"], VR.COUNT)
        self.assertEqual(self.manifest["catalogue"], VR.VERSION)
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), self.manifest["sha256"])
        derived = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in VR.rows()).encode()
        self.assertEqual(derived, self.payload, "the generator no longer reproduces the frozen catalogue")
        self.assertEqual([r["variation_id"] for r in self.rows], list(range(VR.COUNT)))
        for r in self.rows:
            body = {k: v for k, v in r.items() if k != "sha256"}
            self.assertEqual(hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                             r["sha256"])
        self.assertFalse(self.manifest["dimensions_changed"] or self.manifest["physics_parameters_changed"]
                         or self.manifest["masses_changed"] or self.manifest["procedure_changed"])

    def test_splits_frozen_and_complete(self):
        VR = self.VR
        counts = {s: sum(r["split"] == s for r in self.rows) for s in VR.SPLITS}
        self.assertEqual(counts, self.manifest["splits"])
        self.assertEqual(sum(counts.values()), VR.COUNT)
        for s, share in (("train", .8), ("validation", .1), ("test", .1)):
            self.assertAlmostEqual(counts[s]/VR.COUNT, share, delta=.02, msg=s)
        self.assertEqual(self.rows[0]["split"], "train")          # the authored layout of the published demonstrations

    def test_variation_zero_is_the_authored_layout_and_seed_zero(self):
        VR = self.VR
        from welding.variants import resolve
        e = VR.specification(0)
        self.assertEqual((e["coupon"], e["wire_stickout_m"], e["draft_m_s"], e["config_seed"]),
                         ({"dx_m": 0., "dy_m": 0., "yaw_deg": 0.}, None, 0., 0))
        self.assertEqual(VR.config_values(e), dict(seed=0, coupon_offset=resolve(seed=0)["coupon_offset"], draft_m_s=0.))
        self.assertIsNone(VR.apply_initial(None, e))                # nothing to apply: the start state's own wire

    def test_bounds_fixture_check_and_unique_cache_keys(self):
        VR = self.VR
        P = VR.PARAMETERS
        seeds = set()
        for r in self.rows[1:]:
            c = r["coupon"]
            for name, v in (("coupon_dx_m", c["dx_m"]), ("coupon_dy_m", c["dy_m"]), ("coupon_yaw_deg", c["yaw_deg"]),
                            ("wire_stickout_m", r["wire_stickout_m"]), ("draft_m_s", r["draft_m_s"])):
                self.assertTrue(P[name]["low"] <= v <= P[name]["high"], (r["variation_id"], name, v))
            self.assertTrue(VR.fixture_check(r)["ok"], r["variation_id"])
            self.assertEqual(r["config_seed"], VR.CONFIG_SEED_BASE + r["variation_id"])
            seeds.add(r["config_seed"])
        self.assertEqual(len(seeds), VR.COUNT - 1)
        self.assertFalse(seeds & set(range(10)))                     # never a legacy seed's start-state cache key
        self.assertTrue(VR._check_plate_constants())
        self.assertLess(P["draft_m_s"]["high"], 2.2)                 # under the AWS D1.1 limit
        # the draft's stated effect (anchored to Kobelco's blowhole chart, params.AIR_DRAFT, read at the cell's 19 L/min
        # gas flow): air in the shield from 1.4 m/s, 1 % per m/s; at the procedure's stand-off and angles the
        # catalogue's drafts (<= 2.0 m/s, at most 0.6 % air) stay under the 1 % porosity onset, and add to the other
        # deficits
        from welding.arc import air_fraction, porosity_fraction
        from welding import params as PR
        w0, w1 = PR.v(PR.AIR_DRAFT)
        for r in self.rows[1:200]:
            for ntwd, ang in ((.0158, 7.5), (.025, 20.), (.012, -5.)):
                extra = air_fraction(ntwd, ang, r["draft_m_s"]) - air_fraction(ntwd, ang, 0.)
                self.assertAlmostEqual(extra, w1*max(0., r["draft_m_s"] - w0), places=12)
            self.assertEqual(porosity_fraction(air_fraction(.0158, 7.5, r["draft_m_s"])), 0.)

    def test_select_constraints_never_relaxed(self):
        VR = self.VR
        e = VR.select(variation_id=17)
        self.assertEqual(e, self.rows[17])
        other = next(s for s in VR.SPLITS if s != e["split"])
        with self.assertRaises(ValueError):
            VR.select(variation_id=17, split=other)
        for split in VR.SPLITS:
            a, b = VR.select(split=split, selection_seed=3), VR.select(split=split, selection_seed=3)
            self.assertEqual(a, b)
            self.assertEqual(a["split"], split)
        picks = {VR.select(split="test", selection_seed=k)["variation_id"] for k in range(20)}
        self.assertGreater(len(picks), 10)
        for bad in (dict(variation_id=10000), dict(variation_id=-1), dict(variation_id=True), dict(variation_id=2.0),
                    dict(split="dev"), dict(selection_seed="1"), dict(selection_seed=None)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                VR.select(**bad)
        for bad in ({"variation_id": 3, "task": "tack"}, {"seed": 1}, [], "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                VR.select_request(bad)
        self.assertEqual(VR.select_request({"variation_id": 3}), self.rows[3])

    def test_tasks_resolve_variations_round_trip(self):
        from welding.task import INSTRUCTIONS
        for task in INSTRUCTIONS:
            spec = tasks.resolve("welding", {"task": task, "variation_id": 17})
            self.assertEqual(spec.task_id, f"welding/v1/steel-cell/{task}/variation-0017")
            self.assertEqual(spec.launch.backend_options, {"preset": "steel-cell", "task": task, "variation_id": 17})
            self.assertEqual(spec.split, self.rows[17]["split"])
            self.assertEqual(spec.selection["sha256"], self.rows[17]["sha256"])
            self.assertEqual(spec.description, INSTRUCTIONS[task])
            again = tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
            self.assertEqual(again.to_dict(), spec.to_dict())
        picked = tasks.resolve("welding", {"split": "test", "selection_seed": 4})
        self.assertEqual(picked.split, "test")
        self.assertEqual(picked.to_dict(), tasks.resolve("welding", {"split": "test", "selection_seed": 4}).to_dict())
        zero = tasks.resolve("welding", {"variation_id": 0})
        self.assertEqual((zero.task_id, zero.split), ("welding/v1/steel-cell/fillet-2f/variation-0000", "train"))
        legacy = tasks.resolve("welding", {"seed": 4})          # legacy seeds still resolve, split null
        self.assertEqual((legacy.task_id, legacy.split), ("welding/v1/steel-cell/fillet-2f/seed-4", None))

    def test_cli_selectors(self):
        """gar run welding --task T --variation N / --select variation_id=N / --select seed=N."""
        from gym_anything_robotics.cli_launch import selection
        args = SimpleNamespace(selector_json=None, select=["seed=3"], task="tack", variation=None)
        self.assertEqual(selection(args, "welding"), {"seed": 3, "task": "tack"})
        args = SimpleNamespace(selector_json=None, select=[], task="position-torch", variation=42)
        sel = selection(args, "welding")
        self.assertEqual(sel, {"task": "position-torch", "variation_id": 42})
        self.assertEqual(tasks.resolve("welding", sel).task_id, "welding/v1/steel-cell/position-torch/variation-0042")
        args = SimpleNamespace(selector_json=None, select=["split=validation", "selection_seed=2"], task=None,
                               variation=None)
        self.assertEqual(tasks.resolve("welding", selection(args, "welding")).split, "validation")


def _fillet_rows(runs, size=6.6, legs=None, sizes=(), edits=(), over=None):
    """Constructed fillet-2f sections (no physics) every 0.5 mm over the evaluator's read, -1.75 .. 151.75 mm: a sound
    fillet over each run (x0, x1 mm, freeze time of x), equal legs unless `legs`, 1 mm into both members; sizes:
    [(x0, x1, mm)] set after; edits: [(x0, x1, fields)]; over: rows the runs are laid on (else bare joint)."""
    def section(x, s, weld, t=None):
        lv, lh = (legs if legs and weld else (s, s))
        return dict(x_m=x*1e-3, dx_m=5e-4, size_m=s*1e-3, leg_vertical_m=lv*1e-3, leg_horizontal_m=lh*1e-3,
                    throat_m=.707*s*1e-3, convexity_m=0., concavity_m=0., face_width_m=math.hypot(lv, lh)*1e-3,
                    overlap_base_m=0., overlap_bracket_m=0., undercut_base_m=0., undercut_bracket_m=0.,
                    penetration_base_m=1e-3 if weld else 0., penetration_bracket_m=1e-3 if weld else 0., porosity=0.,
                    root_fused=weld, weld_area_m2=1e-5 if weld else 0., geometry="profile", profile_extra_cells=0,
                    profile_missing_cells=0, liquid=False, freeze_t=t)
    rows = [dict(r) for r in over] if over else [section(-1.75 + .5*k, 0., False) for k in range(308)]
    for x0, x1, t in runs:
        for i, r in enumerate(rows):
            x = round(r["x_m"]*1e3, 2)
            if x0 - 1e-9 <= x <= x1 + 1e-9:
                rows[i] = section(x, size, True, t(x))
    for x0, x1, s in sizes:
        for r in rows:
            if x0 - 1e-9 <= r["x_m"]*1e3 <= x1 + 1e-9 and r["weld_area_m2"] > 0:
                r.update(size_m=s*1e-3, leg_vertical_m=s*1e-3, leg_horizontal_m=s*1e-3, throat_m=.707*s*1e-3,
                         face_width_m=math.hypot(s, s)*1e-3)
    for x0, x1, fields in edits:
        for r in rows:
            if x0 - 1e-9 <= r["x_m"]*1e3 <= x1 + 1e-9:
                r.update(fields)
    return rows


def _fillet_judge(rows, start=None):
    """The fillet-2f evaluation of constructed rows (task.Evaluator.judge_fillet), progress net of `start` if given."""
    from welding.task import Evaluator, fillet_progress
    ev = Evaluator(SimpleNamespace(config=SimpleNamespace(task="fillet-2f"), world=SimpleNamespace(process=None)))
    ev.start_raw_progress = fillet_progress(start)["raw"] if start is not None else None
    return ev.judge_fillet(rows)


def _stub_backend(task="fillet-2f"):
    """A backend over a stub world (joint table, kinematics) to check action preparation without physics."""
    import threading
    import numpy as np
    from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
    from welding.world import Config
    joints = {}
    for j in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint"):
        joints[f"right/{j}"] = dict(q=0, dof=0, lower=-6.2, upper=6.2, effort_limit=330., velocity_limit=2.09,
                                    native_label="x")
    for k in range(1, 8):
        joints[f"left/fr3_joint{k}"] = dict(q=0, dof=0, lower=-2.5, upper=2.5, effort_limit=87., velocity_limit=2.,
                                            native_label="x")
    for k in (1, 2):
        joints[f"left/fr3_finger_joint{k}"] = dict(q=0, dof=0, lower=0., upper=.04, effort_limit=35.,
                                                   velocity_limit=.2, native_label="x")
    kin = SimpleNamespace(forward=lambda arm: np.eye(4), solve=Mock(side_effect=ValueError(
        "IK unreachable (2400.00 mm, 0.00 deg); no action executed")))
    world = SimpleNamespace(joints=joints, fault=None, kin=kin, set_grip_force=Mock(), hand_arms=("left",),
                            gripper_joints=lambda arm: [f"{arm}/fr3_finger_joint1", f"{arm}/fr3_finger_joint2"])
    b = object.__new__(NewtonWeldingBackend)
    b.config = Config(task=task, robot="ur10e-fr3")
    b.env = SimpleNamespace(world=world, joints=joints, lock=threading.RLock(), act=Mock(), tool_io=Mock(), steps=0)
    b.arms, b.torch_arm, b.cameras, b.camera_specs = ("left", "right"), "right", (), {}
    b.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "tool_io"})
    b.finger_effort_limit, b._prestart, b._reapers, b.last_timing = 35., None, [], {}
    b.variation, b.legacy_seed, b.trace = None, 0, None
    b.gripper_arms, b.state_evaluation, b._state_cache = ("left",), False, None
    b.capabilities = SimpleNamespace(notes={"torch_tcp": "torch work point"})
    return b


class ActionPreparationTest(unittest.TestCase):
    def test_tool_io_rejections_execute_nothing(self):
        from gym_anything_robotics.actions import ToolIO
        b = _stub_backend()
        rejected = ({"arc_on": "yes"}, {"arc_on": 1}, {"voltage_v": 9.}, {"voltage_v": 45.}, {"voltage_v": True},
                    {"wire_feed_m_min": 30.}, {"wire_feed_m_min": 1.}, {"process": "pulse"}, {"wire_inch": "fast"},
                    {"current_a": 200.}, {"arc_on": True, "polarity": "DCEN"},
                    # the transfer mode follows from the operating point; it is not a source output
                    {"process": "spray"}, {"arc_on": True, "process": "short_circuit"})
        for outputs in rejected:
            with self.subTest(outputs=outputs), self.assertRaises(ActionRejectedError) as caught:
                b.act(ToolIO("right", outputs, 1))
            self.assertIn("no action executed", str(caught.exception))
        with self.assertRaises(ActionRejectedError) as caught:        # the source is wired to the torch arm only
            b.act(ToolIO("left", {"arc_on": True}, 1))
        self.assertIn("'right'", str(caught.exception))
        b.env.act.assert_not_called()
        b.env.tool_io.assert_not_called()

    def test_tool_io_outputs_are_typed(self):
        from gym_anything_robotics.actions import ToolIO
        b = _stub_backend()
        targets, ticks, grip, tool = b._prepare(ToolIO("right", {"arc_on": True, "gas_on": True, "voltage_v": 29,
                                                                 "wire_feed_m_min": 12.7, "wire_inch": "stop"}, 250))
        self.assertEqual((targets, ticks, grip), ({}, 250, None))
        self.assertEqual(tool, {"arc_on": True, "gas_on": True, "voltage_v": 29., "wire_feed_m_min": 12.7,
                                "wire_inch": "stop"})
        self.assertIs(type(tool["voltage_v"]), float)

    def test_motion_rejections_execute_nothing(self):
        from gym_anything_robotics.actions import (CartesianDelta, CartesianPose, Gripper, MotionSafety,
                                                   NamedJointTarget)
        b = _stub_backend()
        rejected = (NamedJointTarget({"right/elbow_joint": 7.}, 5), NamedJointTarget({"torch_free": 0.}, 5),
                    CartesianPose("right", (2., 2., 2.)),
                    Gripper("left", half_width_m=.05), Gripper("left", inward_effort_n=80.),
                    Gripper("left", half_width_m=.02, duration_s=.0005), Gripper("right", half_width_m=.02))
        for action in rejected:
            with self.subTest(action=action), self.assertRaises(ActionRejectedError) as caught:
                b.act(action)
            self.assertIn("no action executed", str(caught.exception))
        with self.assertRaises(UnsupportedActionError):
            b.act(CartesianDelta("right", (0., 0., .01), safety=MotionSafety(stop_on_contact=True)))
        b.env.act.assert_not_called()
        b.env.world.set_grip_force.assert_not_called()

    def test_gripper_semantics(self):
        from gym_anything_robotics.actions import Gripper
        b = _stub_backend("tack")
        targets, ticks, grip, tool = b._prepare(Gripper("left", inward_effort_n=60., duration_s=.05))
        self.assertEqual((ticks, grip, tool), (50, ("left", 30.), None))
        self.assertEqual(targets, {"left/fr3_finger_joint1": 0., "left/fr3_finger_joint2": 0.})
        targets, _, grip, _ = b._prepare(Gripper("left", half_width_m=.02, duration_s=.1))
        self.assertEqual((set(targets.values()), grip), ({.02}, ("left", 0.)))


class PolicyAccessTest(unittest.TestCase):
    def test_instruction_capabilities_and_redaction(self):
        from welding.task import INSTRUCTIONS
        for task in INSTRUCTIONS:
            access = _stub_backend(task).policy_access()
            self.assertEqual(access.instruction(), INSTRUCTIONS[task])
        caps = access.capabilities()
        self.assertEqual(caps["action_types"], ["cartesian_delta", "cartesian_pose", "gripper", "named_joint_target",
                                                "tool_io"])
        self.assertFalse(caps["privileged_state"])
        hw = caps["hardware"]
        self.assertEqual(hw["torch_arm"], "right")
        self.assertEqual(set(hw["tool_io"]), {"arc_on", "gas_on", "touch_sense", "wire_feed_m_min", "voltage_v",
                                              "wire_inch"})
        self.assertEqual(hw["joints"]["left/fr3_finger_joint1"]["unit"], "metre")
        self.assertEqual(hw["joints"]["right/elbow_joint"]["unit"], "radian")
        self.assertFalse({"q", "dof", "native_label"} & set(hw["joints"]["right/elbow_joint"]))
        fields = caps["proprioception_fields"]
        self.assertEqual(len(fields), 2*len(hw["joints"]) + 7)
        self.assertEqual({f for f in fields if f.startswith("tool.")},
                         {f"tool.{k}" for k in ("arc_on", "current_a", "voltage_v", "wire_feed_m_min", "touch", "fault",
                                                "gas_on")})
        action = access.decode_actions({"type": "tool_io", "arm": "right", "outputs": {"gas_on": True}, "ticks": 10})
        self.assertTrue(action)
        for bad in ({"type": "keyboard", "keys": ["W"], "ticks": 1},
                    {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, .01],
                     "safety": {"stop_on_contact": True}},
                    {"type": "apply_force", "body": "bracket", "force_n": [0, 0, 100]}):
            with self.subTest(bad=bad["type"]), self.assertRaises(Exception):
                access.decode_actions(bad)


class WallTimeNoteTest(unittest.TestCase):
    """The runtime note a restricted policy receives (capabilities.hardware.wall_time) is built from the result file of
    the runtime measurement (E16, welding/REPORT.pdf §16) when the backend is imported."""

    def test_note_is_derived_from_the_e16_result(self):
        from gym_anything_robotics.backends import newton_welding as NW
        path = ROOT/"welding/data/runtime/e16_runtime_api.json"
        self.assertEqual(NW.e16_result_path().resolve(), path.resolve())
        e16 = json.loads(path.read_text())
        figures = NW.wall_time_figures(e16)
        self.assertEqual(NW.WALL_TIME_NOTE, NW._wall_time_note(figures))
        self.assertNotEqual(NW.WALL_TIME_NOTE, NW.WALL_TIME_FALLBACK)
        for key, figure in figures.items():
            with self.subTest(key=key):
                self.assertIn(figure, NW.WALL_TIME_NOTE)
        # spot checks against the file's own fields, independent of the derivation's code
        one, three = e16["sets"]["one camera"], e16["sets"]["three cameras"]
        self.assertIn(f"{one['fit_single_travel']['slope_wall_per_sim']:.0f}", figures["physics_arc_on"])
        self.assertIn(f"{three['gated_worst']['wall_per_sim']:.0f}x", figures["three_worst"])
        self.assertEqual(figures["minimum_ms"], f"{1e3*e16['minimum_command_s']['one camera']:g}")
        self.assertEqual(figures["one_within"] == "stayed within", one["passed"] is True)
        self.assertEqual(figures["three_within"] == "stayed within", three["passed"] is True)
        self.assertIn("depend on the GPU's load", NW.WALL_TIME_NOTE)
        self.assertIn("client_timeout_s", NW.WALL_TIME_NOTE)

    def test_missing_or_broken_result_file_gives_the_fallback(self):
        import tempfile
        from gym_anything_robotics.backends import newton_welding as NW
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/"e16.json"
            self.assertEqual(NW.load_wall_time_note(bad), NW.WALL_TIME_FALLBACK)
            bad.write_text('{"sets": {}}')
            self.assertEqual(NW.load_wall_time_note(bad), NW.WALL_TIME_FALLBACK)
        self.assertIn("welding/REPORT.pdf", NW.WALL_TIME_FALLBACK)
        self.assertIn("client_timeout_s", NW.WALL_TIME_FALLBACK)

    def test_policy_capabilities_carry_the_note(self):
        from gym_anything_robotics.backends import newton_welding as NW
        hw = _stub_backend("fillet-2f").policy_access().capabilities()["hardware"]
        self.assertEqual(hw["wall_time"], NW.WALL_TIME_NOTE)
        self.assertEqual(hw["client_timeout_s"], NW.CLIENT_TIMEOUT_S)

    def test_note_quotes_the_paused_policy_and_the_load_and_states_the_timeout(self):
        """E16 measures a policy that pauses between requests and samples the GPU's load; the note quotes both, names
        the render worker's retention, and ends with the timeout's value."""
        import copy
        from gym_anything_robotics.backends import newton_welding as NW
        e16 = json.loads((ROOT/"welding/data/runtime/e16_runtime_api.json").read_text())
        self.assertNotIn("Client timeout: client_timeout_s", NW._wall_time_note(NW.wall_time_figures(e16)))
        self.assertIn(f"Client timeout: {NW.CLIENT_TIMEOUT_S} s", NW._wall_time_note(NW.wall_time_figures(e16)))
        new = copy.deepcopy(e16)
        new.update(pause_s=15., physics_hash="0123456789abcdef")
        for k, (s, paused) in enumerate(zip(new["sets"].values(), ((101., 151.), (202., 252.)))):
            s["render_worker_idle_release_s"] = NW.IDLE_RELEASE_S
            # the file's own paused rows give way to the fixture's
            s["rows"] = [r_ for r_ in s["rows"] if r_.get("kind") != "paused"]
            s["rows"] += [dict(step=f"travel {ms:g} ms after a 15 s pause #0 (arc on)", kind="paused", wall_s=r*ms/1e3,
                               sim_s=ms/1e3, wall_per_sim=r) for ms, r in zip((100, 50), paused)]
            s["load"] = dict(used_mib=dict(min=9000 + k, median=9500, max=10240),
                             util_pct=dict(min=80, median=97, max=100),
                             other_render_workers=dict(min=1, median=1, max=2),
                             other_processes=dict(min=3, median=3, max=4),
                             other_mib=dict(min=5000, median=5500, max=6000))
        none = copy.deepcopy(new["sets"]["one camera"])
        none.update(passed=True, gated_worst=dict(step="travel 50 ms #0 (arc on)", wall_per_sim=123.4))
        none["travel_wall_per_sim"] = {"500 ms": dict(median=61., max=62., n=2, gated=True),
                                       "50 ms": dict(median=111., max=123.4, n=3, gated=True),
                                       "20 ms": dict(median=150., max=151., n=3, gated=False)}
        new["sets"]["no cameras"] = none
        f = NW.wall_time_figures(new)
        note = NW._wall_time_note(f)
        for figure in f.values():
            self.assertIn(figure, note)
        self.assertIn("one camera 100 ms 101x; 50 ms 151x, three cameras 100 ms 202x; 50 ms 252x", note)
        self.assertIn("after a 15 s pause", note)
        self.assertIn("1 other Isaac render worker (median of each camera set; at most 2)", note)
        self.assertIn("physics 0123456789abcdef", note)
        self.assertIn(f"warm {NW.IDLE_RELEASE_S:g} s after each request", note)
        self.assertIn("Without images (backend option cameras: [], no render worker)", note)
        self.assertIn("arc-on commands of 50 ms or longer stayed within the limit (61x at 500 ms to 111x at 50 ms, "
                      "worst 123x)", note)
        self.assertNotIn("not measured by this E16 run", note)
        self.assertNotIn("an independent replication under a heavier load", note)
        if not any(r["kind"] == "paused" for s in e16["sets"].values() for r in s["rows"]):
            self.assertIn("Pausing between requests: not measured by this E16 run", NW.WALL_TIME_NOTE)


class LaunchCheckTest(unittest.TestCase):
    """Every backend option is checked by GAR's launch check before a session exists, and `gar inspect welding`
    quotes only load-labelled runtime figures."""

    def test_invalid_backend_options_refused_at_launch_and_valid_ones_accepted(self):
        from gym_anything_robotics.launch import LaunchSpec
        bad = ({"cameras": ["hero"]}, {"cameras": "bench"}, {"cameras": ["bench", "bench"]}, {"phantom_torch": True},
               {"config": {"phantom_torch": True}}, {"config": {"no_such_field": 1}}, {"config": {"seed": 3}},
               {"renderer_options": {"idle_release_s": -1}}, {"renderer_options": {"idle_release_s": "long"}},
               {"renderer_options": {"width": 0}}, {"renderer_options": {"samples": 1.5}},
               {"renderer_options": {"exposure": 2}}, {"trace": "yes"}, {"trace_bytes": 10}, {"state_evaluation": 1},
               {"seed": 2, "variation_id": 3}, {"variation_id": 10000}, {"task": "weld-the-moon"})
        with patch("welding.env.WeldingEnv") as env:
            for options in bad:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    LaunchSpec("welding", backend_options=options)
            for options in ({}, {"cameras": ["bench", "side", "weld_camera"]}, {"cameras": []},
                            {"renderer_options": {"idle_release_s": 0, "width": 640, "height": 360, "samples": 4}},
                            {"task": "tack", "variation_id": 12}, {"config": {"device": "cuda:0"}}):
                with self.subTest(options=options):
                    LaunchSpec("welding", backend_options=options)
            env.assert_not_called()

    def test_inspect_text_states_only_load_labelled_runtime(self):
        from gym_anything_robotics.launch import describe
        notes = describe("welding")["notes"]
        self.assertNotIn("50-54", notes)
        self.assertNotIn("50 ms or longer", notes)
        for text in ("hardware.wall_time", "100 ms or longer", "idle_release_s", "3600"):
            self.assertIn(text, notes)


class RenderWorkerRetentionTest(unittest.TestCase):
    """The render worker is kept IDLE_RELEASE_S after a request, so a policy that pauses between requests does not
    pay for a fresh worker on its next one; the retention is stated to the policy."""

    def test_idle_release_default_option_and_statement(self):
        from gym_anything_robotics.backends import newton_welding as NW
        self.assertGreaterEqual(NW.IDLE_RELEASE_S, 300.)
        b = _stub_backend()
        b.options, b.cameras, b.camera_specs = {}, ("bench",), {"bench": SimpleNamespace(name="bench")}
        self.assertEqual(b._idle_release_s(), NW.IDLE_RELEASE_S)
        b.options = {"idle_release_s": 0}
        self.assertEqual(b._idle_release_s(), 0.)
        b.options = {"idle_release_s": 1800}
        hw = b.policy_access().hardware
        self.assertEqual(hw["render_worker_idle_release_s"], 1800.)
        b._release, b._retain_timer = Mock(), None
        b._retained = None
        b.options = {"idle_release_s": 0}
        r = Mock()
        b._end_operation(r)                          # 0: released at once, nothing retained
        b._release.assert_called_once_with(r)
        self.assertIsNone(b._retained)
        b.options = {}
        b._release = Mock()
        with patch("threading.Timer") as timer:
            b._end_operation(r)                      # default: retained, a timer set to IDLE_RELEASE_S
        self.assertIs(b._retained, r)
        self.assertEqual(timer.call_args.args[0], NW.IDLE_RELEASE_S)
        b._release.assert_not_called()


class WorldReleaseTest(unittest.TestCase):
    """A catalogue switch in a served session must not keep the replaced world. The world sits in reference cycles
    that only the cyclic collector frees, and a served session hardly ever runs it. These checks run with the automatic
    collector off and never collect themselves: WeldingEnv.close() breaks the world's own cycles, and a switch collects
    what Newton keeps in cycles."""

    def _cyclic_env(self):
        """A WeldingEnv (no physics) over a stand-in world with the real world's cycles: its kinematics, its bond and
        the pool's obstacle callback refer back to it, the evaluator to the env; its solver and model refer to each
        other (as Newton's objects do), out of close()'s reach."""
        import threading
        from welding.env import WeldingEnv
        from welding.task import Evaluator

        class Part:                                             # SimpleNamespace takes no weak references
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class World(Part):
            def _pool_obstacles(self):
                return None
        w = World()
        w.kin, w.bond = Part(world=w), Part(w=w)
        w.process = Part(pool=Part(obstacle=w._pool_obstacles), arrays=[bytearray(1 << 20)])
        w.solver = Part()
        w.model = Part(solver=w.solver)
        w.solver.model = w.model
        env = object.__new__(WeldingEnv)
        env.lock, env.world, env.generation, env.steps, env.start_state = threading.RLock(), w, 0, 0, "tacked"
        env.evaluator = Evaluator(env)
        return env

    def test_close_releases_the_world_without_the_collector(self):
        import gc
        import weakref
        enabled = gc.isenabled()
        gc.disable()
        try:
            env = self._cyclic_env()
            refs = [weakref.ref(o) for o in (env, env.world, env.world.kin, env.world.process)]
            env.close()
            self.assertTrue(env.closed)
            self.assertIsNone(env.world)
            env.close()                                         # idempotent
            del env
            self.assertEqual([r() for r in refs], [None]*4)
        finally:
            if enabled:
                gc.enable()

    def test_switches_keep_one_live_world(self):
        import gc
        import weakref
        b = _stub_backend("fillet-2f")
        b.env = self._cyclic_env()
        b._take_retained, b._scene_path, b._checkpoints = Mock(return_value=None), None, {}
        b.capabilities = SimpleNamespace(notes={})
        b._state = Mock(return_value={"tick": 0})
        worlds, solvers = [], []

        def make_env(config, entry):
            env = self._cyclic_env()
            worlds.append(weakref.ref(env.world))
            solvers.append(weakref.ref(env.world.solver))
            return env
        b._make_env = make_env
        worlds.append(weakref.ref(b.env.world))
        solvers.append(weakref.ref(b.env.world.solver))
        enabled = gc.isenabled()
        gc.disable()
        try:
            live = []
            for v in (37, 290, 5885, 0, 1234):
                b.reset(options={"variation_id": v})
                live.append((sum(r() is not None for r in worlds), sum(r() is not None for r in solvers)))
        finally:
            if enabled:
                gc.enable()
        self.assertEqual(live, [(1, 1)]*5)
        self.assertIn("switch_gc_s", b.last_timing)


@unittest.skipUnless(os.environ.get("GAR_WELDING_PHYSICS_TESTS") == "1",
                     "set GAR_WELDING_PHYSICS_TESTS=1 (builds the weld cell three times, GPU)")
class WorldReleasePhysicsTest(unittest.TestCase):
    """The real backend (no cameras), three catalogue switches with the automatic collector off: the number of live
    welding.world.World objects that still hold their state stays at one, and no closed husk accumulates (the first
    import of Newton can keep the frames of whatever was on the stack: welding/env.py _import_engine)."""

    def test_live_worlds_across_switches(self):
        import gc
        import tempfile
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        from welding.world import World

        def count():
            worlds = [o for o in gc.get_objects() if isinstance(o, World)]      # a scan, not a collection
            return sum(bool(o.__dict__) for o in worlds), len(worlds)
        with tempfile.TemporaryDirectory() as d:
            b = NewtonWeldingBackend(task="fillet-2f", cameras=(), trace=False, output_dir=d + "/s")
            enabled = gc.isenabled()
            gc.disable()
            try:
                counts = []
                for v in (37, 290, 5885):
                    b.reset(options={"variation_id": v})
                    counts.append(count())
            finally:
                if enabled:
                    gc.enable()
                b.close()
        self.assertEqual([c[0] for c in counts], [1, 1, 1])
        self.assertEqual(len({c[1] for c in counts}), 1, counts)


class BackendVariationTest(unittest.TestCase):
    """Variation selection through the backend (GAR /v1/variations, launch option variation_id) without physics."""

    def test_variation_options_refused_before_the_world_is_built(self):
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        bad = (dict(variation_id=3, seed=2), dict(variation_id=3, seed=0), dict(variation_id=10000),
               dict(variation_id=-1), dict(variation_id=True), dict(variation_id="3"),
               dict(config={"coupon_offset": (0., 0., 0.)}), dict(config={"draft_m_s": 1.}),
               dict(variation_id=3, config={"draft_m_s": 0.}), dict(trace="yes"), dict(trace_bytes=1024),
               dict(trace_bytes=True), dict(variation_id=5, task="tack", robot="ur10e"))
        with patch("welding.env.WeldingEnv") as env:
            for options in bad:
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError,
                                                                        UnsupportedConfigurationError)):
                    NewtonWeldingBackend(output_dir="/nonexistent/never-created", **options)
            env.assert_not_called()

    def test_variation_reaches_the_world_config(self):
        """The world is built with the entry's coupon pose, draft and cache key (the build is stopped there)."""
        import tempfile
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        from welding.variations import specification

        class Built(Exception):
            pass
        e = specification(1234)
        with tempfile.TemporaryDirectory() as d, patch("welding.env.WeldingEnv", side_effect=Built) as env, \
                self.assertRaises(Built):
            NewtonWeldingBackend(task="position-torch", variation_id=1234, cameras=(), output_dir=d + "/s")
        cfg = env.call_args.args[0]
        c = e["coupon"]
        self.assertEqual((cfg.seed, cfg.coupon_offset, cfg.draft_m_s, cfg.task),
                         (100000 + 1234, (c["dx_m"], c["dy_m"], c["yaw_deg"]), e["draft_m_s"], "position-torch"))
        with tempfile.TemporaryDirectory() as d, patch("welding.env.WeldingEnv", side_effect=Built) as env, \
                self.assertRaises(Built):
            NewtonWeldingBackend(task="tack", seed=4, cameras=(), output_dir=d + "/s")
        from welding.variants import coupon_offset
        cfg = env.call_args.args[0]
        self.assertEqual((cfg.seed, cfg.coupon_offset, cfg.draft_m_s), (4, coupon_offset(4), 0.))

    def test_contract_select_and_reset_refusals(self):
        from welding.variations import specification, VERSION, COUNT
        b = _stub_backend("tack")
        contract = b.variation_contract()
        self.assertTrue(contract["supported"])
        self.assertEqual((contract["catalogue"], contract["count"], contract["splits"]),
                         (VERSION, COUNT, ["train", "validation", "test"]))
        self.assertEqual(contract["current"], {"variation": None, "legacy_seed": 0, "task": "tack",
                                               "robot": "ur10e-fr3"})
        json.dumps(contract, allow_nan=False)
        sel = b.select_variation({"variation_id": 77})
        self.assertEqual(sel["reset_request"], {"options": {"variation_id": 77}})
        self.assertEqual((sel["variation"], sel["task"]), (specification(77), "tack"))
        self.assertEqual(b.select_variation({"split": "validation", "selection_seed": 9})["variation"]["split"],
                         "validation")
        for bad in ({"variation_id": COUNT}, {"variation_id": 1, "split": "nope"}, {"task": "tack"}, {"seed": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                b.select_variation(bad)
        b._start_variation = Mock()
        for kwargs in (dict(options={"variation_id": 5, "seed": 1}), dict(seed=1, options={"variation_id": 5}),
                       dict(options={"task": "tack"}), dict(seed=3)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                b.reset(**kwargs)
        b._start_variation.assert_not_called()
        b.reset(options={"variation_id": 5})
        b._start_variation.assert_called_once_with(5)

    def test_start_variation_swaps_the_episode_after_building(self):
        import threading
        from welding.variations import specification
        b = _stub_backend("tack")
        old = b.env
        old.generation, old.close = 4, Mock()
        new = SimpleNamespace(lock=threading.RLock(), generation=0, start_state="loose", steps=0,
                              world=SimpleNamespace(tick=0, fault=None), get_state=Mock(return_value={"tick": 0}))
        b._make_env = Mock(return_value=new)
        b._take_retained, b._scene_path, b._checkpoints = Mock(return_value=None), "old.usdc", {"a": 1}
        b.trace = Mock(current=None)
        b.capabilities = SimpleNamespace(notes={})
        with self.assertRaises(ValueError):                     # a rejected id builds nothing and changes nothing
            b._start_variation(10000)
        b._make_env.assert_not_called()
        self.assertIs(b.env, old)
        state, info = b.reset(options={"variation_id": 21})
        cfg, entry = b._make_env.call_args.args
        self.assertEqual(entry, specification(21))
        self.assertEqual((cfg.seed, cfg.task, cfg.robot), (100021, "tack", "ur10e-fr3"))
        self.assertIs(b.env, new)
        self.assertEqual(new.generation, 5)
        self.assertEqual((b.variation, b.legacy_seed, b._scene_path, b._checkpoints), (entry, None, None, {}))
        b.trace.bind.assert_called_once_with(new)
        old.close.assert_called_once()
        self.assertEqual(info["continuity"], "new_variation")
        self.assertIn("variation 21", b.capabilities.notes["task"])
        new.get_state.assert_called_once_with(evaluate=False)    # the new episode's state, read once, not evaluated
        self.assertEqual(state["tick"], 0)


def _backend_on_rig(robot, task="fillet-2f", **options):
    """The real backend constructor for a catalog deployment over a stub environment whose arms and hands come from
    the deployment's catalog rig the way welding/scene.py reads them (no GPU, no physics)."""
    import tempfile
    import threading
    from robots.catalog import for_environment
    from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
    data = for_environment("welding", robot).to_dict()
    arm_hw = {name: {} for name in data["instances"]}
    for att in data.get("attachments", {}).values():
        arm_hw[att["parent"].split("/")[0]]["hand"] = att["model"]
    hand_arms = tuple(a for a in arm_hw if arm_hw[a].get("hand"))
    joints = {f"{a}/fr3_finger_joint{k}": dict(q=0, dof=0, lower=0., upper=.04, effort_limit=100.,
                                               velocity_limit=.2) for a in hand_arms for k in (1, 2)}
    world = SimpleNamespace(rig=SimpleNamespace(instances=dict.fromkeys(arm_hw)), torch_arm="right",
                            hand_arms=hand_arms, scene=SimpleNamespace(arm_hw=arm_hw), joints=joints, fault=None,
                            gripper_joints=lambda arm: [f"{arm}/fr3_finger_joint1", f"{arm}/fr3_finger_joint2"])
    env = SimpleNamespace(world=world, joints=joints, start_state="tacked", lock=threading.RLock(), act=Mock(),
                          tool_io=Mock(), steps=0)
    out = Path(tempfile.mkdtemp())/"session"
    with patch.object(NewtonWeldingBackend, "_make_env", return_value=env):
        return NewtonWeldingBackend(robot=robot, task=task, trace=False, cameras=(), output_dir=str(out), **options)


class DeploymentCapabilityTest(unittest.TestCase):
    """The gripper command is offered only by deployments with a Franka Hand."""

    def test_gripper_advertised_only_with_a_franka_hand(self):
        from gym_anything_robotics.actions import Gripper
        from robots.catalog import ENVIRONMENTS
        expected = {"ur10e-fr3": True, "dual-fr3": True, "ur10e": False, "single-fr3": False}
        if "dual-panda" in ENVIRONMENTS["welding"]["aliases"]:      # the Panda two-arm rig: a Franka Hand on the left
            expected["dual-panda"] = True
        for robot, has_hand in expected.items():
            with self.subTest(robot=robot):
                b = _backend_on_rig(robot)
                policy_kinds = b.policy_access().capabilities()["action_types"]
                self.assertEqual("gripper" in b.kinds, has_hand)
                self.assertEqual("gripper" in b.capabilities.action_types, has_hand)
                self.assertEqual("gripper" in policy_kinds, has_hand)
                self.assertEqual(set(policy_kinds) - {"gripper"},
                                 {"named_joint_target", "cartesian_pose", "cartesian_delta", "tool_io"})
                if has_hand:
                    self.assertIn("Franka Hand on 'left'", b.capabilities.notes["gripper"])
                else:
                    self.assertEqual(b.capabilities.notes["gripper"], "no hand in this deployment")
                    for arm in b.arms:
                        with self.assertRaises(UnsupportedActionError) as caught:
                            b.act(Gripper(arm, half_width_m=.02))
                        self.assertIn("no action executed", str(caught.exception))
                    with self.assertRaises(Exception):
                        b.policy_access().decode_actions({"type": "gripper", "arm": "right", "half_width_m": .02})
                    b.env.act.assert_not_called()

    def test_state_evaluation_option_is_typed(self):
        for bad in (1, "yes", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _backend_on_rig("ur10e", state_evaluation=bad)
        self.assertIn("state_evaluation=true adds it", _backend_on_rig("ur10e").capabilities.notes["evaluator"])
        self.assertIn("every state carries it",
                      _backend_on_rig("ur10e", state_evaluation=True).capabilities.notes["evaluator"])


class ClientTimeoutTest(unittest.TestCase):
    """Legal welding commands outlast the CLI's default 120 s timeout (a 20 s hold can take 181 s, a variation build
    6-45 min). The backend declares client_timeout_s; the session records it at startup and `gar action /
    observe / state / benchmark / pause / resume` use it unless --timeout is given."""

    def _session(self, tmp, **manifest):
        directory = Path(tmp)/("a"*32)
        directory.mkdir()
        keys = Path(tmp)/"keys.json"
        keys.write_text(json.dumps({"privileged": "owner-test", "policy": "policy-test"}))
        (directory/"session.json").write_text(json.dumps(dict(
            status="running", url="http://127.0.0.1:9", session_id="a"*32,
            launch={"credential_file": str(keys)}, **manifest)))
        return directory

    def test_backend_declares_and_session_records_its_client_timeout(self):
        from gym_anything_robotics import sessions
        from gym_anything_robotics.env import RoboticsEnv
        b = _backend_on_rig("ur10e-fr3")
        declared = b.policy_access().hardware["client_timeout_s"]
        self.assertGreaterEqual(declared, 2700)        # a 45 min variation build on a busy GPU fits
        self.assertEqual(sessions.declared_client_timeout(RoboticsEnv(b, auto_reset=False)), float(declared))
        self.assertIsNone(sessions.declared_client_timeout(SimpleNamespace(policy_access=Mock(
            side_effect=NotImplementedError))))
        for bad in (True, 0, -5, float("nan"), "600", None):
            env = SimpleNamespace(policy_access=lambda bad=bad: SimpleNamespace(hardware={"client_timeout_s": bad}))
            self.assertIsNone(sessions.declared_client_timeout(env))

    def test_cli_default_is_the_sessions_declared_timeout(self):
        import tempfile
        from gym_anything_robotics import sessions
        from gym_anything_robotics.cli_parser import parser
        for command in (["action", "x"], ["observe", "x"], ["state", "x"], ["verify", "x"],
                        ["benchmark", "x", "--agent", "m:f", "--max-actions", "1", "--output", "o"]):
            with self.subTest(command=command[0]):
                self.assertIsNone(parser().parse_args(command).timeout)
                self.assertEqual(parser().parse_args(command + ["--timeout", "7"]).timeout, 7.)
        self.assertEqual(sessions.client_timeout({"client_timeout_s": 3600}), 3600.)
        self.assertEqual(sessions.client_timeout({"client_timeout_s": 3600}, 7.), 7.)
        for manifest in ({}, {"client_timeout_s": None}, {"client_timeout_s": True}, {"client_timeout_s": -1}):
            self.assertEqual(sessions.client_timeout(manifest), 120.)
        with tempfile.TemporaryDirectory() as tmp:
            for manifest, explicit, expected in (({"client_timeout_s": 3600.}, None, 3600.),
                                                 ({"client_timeout_s": 3600.}, 9., 9.), ({}, None, 120.)):
                directory = self._session(tmp, **manifest)
                remote = Mock()
                remote.session.return_value = {"session_id": "a"*32}
                with self.subTest(manifest=manifest, explicit=explicit), \
                        patch.object(sessions, "RemoteEnv", return_value=remote) as cls:
                    self.assertIs(sessions.client(directory, timeout_s=explicit), remote)
                    self.assertEqual(cls.call_args.kwargs["timeout_s"], expected)
                import shutil
                shutil.rmtree(directory)

    def test_benchmark_policy_client_uses_the_owner_clients_timeout(self):
        import tempfile
        from gym_anything_robotics import benchmark, sessions
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._session(tmp, client_timeout_s=3600.)
            owner = Mock(url="http://127.0.0.1:9", timeout_s=3600.)
            owner.session.return_value = {"paused": True}
            with patch.object(sessions, "client", return_value=owner) as client, \
                    patch.object(benchmark, "RemoteEnv") as policy, self.assertRaises(ValueError):
                benchmark.run(str(directory), agent="m:f", max_actions=1, output=Path(tmp)/"out")
            self.assertIsNone(client.call_args.kwargs["timeout_s"])
            self.assertEqual(policy.call_args.kwargs["timeout_s"], 3600.)


class _CountingEnv:
    """WeldingEnv's state and action surface with counters: how often the state is read and the evaluator runs."""

    def __init__(self):
        import threading
        self.lock = threading.RLock()
        self.steps, self.generation, self.reads, self.evaluations = 0, 0, 0, 0
        self.world = SimpleNamespace(tick=0, fault=None, set_grip_force=Mock(), joints={})
        self.evaluator = SimpleNamespace(evaluate=self._evaluate)
        self.act_calls = []

    def _evaluate(self):
        self.evaluations += 1
        return dict(success=False, score=.25, checks={"x": {"pass": False}})

    def get_state(self, evaluate=True):
        self.reads += 1
        return {"tick": self.steps, "generation": self.generation, "time_s": self.steps*1e-3, "tool": {},
                "events": [], "terminated": False, "joints": {},
                "evaluation": self._evaluate() if evaluate else dict(success=False, score=0., checks={})}

    def act(self, targets, *, ticks, state=True):
        self.act_calls.append((dict(targets), ticks, state))
        self.steps += ticks
        self.world.tick += ticks
        return self.get_state() if state else None

    def tool_io(self, outputs):
        pass

    def reset(self):
        self.steps, self.world.tick = 0, 0
        self.generation += 1
        return self.get_state()


class StateReadTest(unittest.TestCase):
    """GAR reads the state before and after every command (action history), the command reads it for its result and
    an observation reads it again. The backend reads the state once per tick and evaluates only when asked."""

    def _backend(self, **kw):
        from gym_anything_robotics.actions import ToolIO
        b = _stub_backend()
        b.env = _CountingEnv()
        b.arms = ("left", "right")
        for k, v in kw.items():
            setattr(b, k, v)
        return b, ToolIO

    def test_one_state_read_per_tick_and_no_evaluator_per_command(self):
        b, ToolIO = self._backend()
        env = b.env
        before = b.get_state()                                  # the history's before-state
        step = b.act_policy(ToolIO("right", {"gas_on": True}, 1))
        after = b.get_state()                                   # the history's after-state
        b.get_state()                                           # the next request's before-state
        self.assertEqual(env.act_calls, [({}, 1, False)])       # physics without the environment's own state read
        self.assertEqual((env.reads, env.evaluations), (2, 0))  # one read per tick, no evaluator
        self.assertIsNone(step.reward)
        self.assertEqual((before["tick"], after["tick"], step.state["tick"]), (0, 1, 1))
        self.assertIsNone(after["evaluation"])
        self.assertIn("GET /v1/evaluation", after["evaluation_note"])
        step = b.act(ToolIO("right", {"gas_on": True}, 2))     # owner command: no evaluator either by default
        self.assertIsNone(step.reward)
        self.assertEqual((env.reads, env.evaluations), (3, 0))
        ev = b.evaluate()                                       # on demand: evaluates this tick, once
        self.assertEqual((ev["score"], env.evaluations), (.25, 1))
        self.assertEqual(b.get_state()["evaluation"]["score"], .25)
        b.evaluate()
        self.assertEqual((env.reads, env.evaluations), (3, 1))
        b.act(ToolIO("right", {"gas_on": False}, 1))           # a new tick drops the cached state and evaluation
        self.assertIsNone(b.get_state()["evaluation"])
        self.assertEqual((env.reads, env.evaluations), (4, 1))

    def test_state_evaluation_option_evaluates_each_tick_once(self):
        b, ToolIO = self._backend(state_evaluation=True)
        env = b.env
        step = b.act(ToolIO("right", {"gas_on": True}, 1))
        self.assertEqual(step.reward, .25)
        b.get_state()
        b.get_state()
        b.evaluate()
        self.assertEqual((env.reads, env.evaluations), (1, 1))
        step = b.act_policy(ToolIO("right", {"gas_on": True}, 1))
        self.assertIsNone(step.reward)
        self.assertEqual(b.get_state()["evaluation"]["score"], .25)
        self.assertEqual((env.reads, env.evaluations), (2, 2))

    def test_on_demand_evaluation_judges_a_freezing_weld_as_it_freezes(self):
        """A weld handed back while its pool still freezes with the arc off is judged on demand as it freezes
        hands-off (task.freeze_hands_off), not as a liquid weld; the per-state evaluation is the present state's."""
        b, ToolIO = self._backend(state_evaluation=True)
        env = b.env
        calls = []

        def evaluate(settle=False):
            calls.append(settle)
            return dict(success=settle, score=1. if settle else .5, checks={},
                        judged=dict(freeze_pending=not settle, hands_off_freeze=dict(applied=True) if settle else None))
        env.evaluator = SimpleNamespace(evaluate=evaluate)
        step = b.act(ToolIO("right", {"arc_on": False}, 1))
        self.assertEqual((step.reward, calls), (.5, [False]))
        ev = b.evaluate()
        self.assertEqual((ev["success"], ev["score"], calls), (True, 1., [False, True]))
        self.assertTrue(b.evaluate()["success"])
        self.assertEqual(calls, [False, True])                  # this tick's evaluation from then on
        self.assertTrue(b.get_state()["evaluation"]["judged"]["hands_off_freeze"]["applied"])

    def test_rejected_command_keeps_the_state_and_reset_drops_it(self):
        b, ToolIO = self._backend()
        env = b.env
        b.get_state()
        with self.assertRaises(ActionRejectedError):
            b.act(ToolIO("right", {"voltage_v": 99.}, 1))
        b.get_state()
        self.assertEqual(env.reads, 1)
        state, info = b.reset()
        self.assertEqual((info["continuity"], state["generation"]), ("reset", 1))
        self.assertEqual(env.reads, 2)                          # WeldingEnv.reset's own read is the cached state
        self.assertEqual(b.get_state()["generation"], 1)
        self.assertEqual(env.reads, 2)


class _StubEnv:
    """The parts of WeldingEnv the trace touches: the world, the episode clock and the per-tick _record hook."""

    def __init__(self, world):
        from welding.world import Config
        self.world, self.steps, self.recorded = world, 0, 0
        self.config = Config(task="tack", robot="ur10e-fr3")

    def _record(self):
        self.recorded += 1

    def act(self, ticks, step=None):
        for _ in range(ticks):
            self.world.tick += 1
            if step:
                step(self.world)
            self.steps += 1
            self._record()


def _stub_world(process=False):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from welding import torch as TO
    rng = np.random.default_rng(3)
    names = [f"right/j{k}" for k in range(6)] + [f"left/j{k}" for k in range(7)]
    joints = {n: dict(q=k + 7, dof=k + 6) for k, n in enumerate(names)}
    body_q = np.zeros((20, 7), np.float32)
    body_q[:, 6] = 1.
    gun_body = 5
    q = Rotation.from_euler("xyz", rng.uniform(-1, 1, 3)).as_quat()
    body_q[gun_body] = (*rng.uniform(-.5, .5, 3), *q)
    pt, Rt = np.array([0., 0., .1]), Rotation.from_euler("x", .3).as_matrix()
    spec = TO.BA1
    arr = lambda a: SimpleNamespace(numpy=lambda: a)

    def torch_pose():
        b = w.state.body_q.numpy()[gun_body].astype(float)
        R = Rotation.from_quat(b[3:]).as_matrix()
        return None, None, b[:3] + R @ (pt + Rt @ spec.tcp)
    labels = [f"right/link{k}" for k in range(8)] + [f"left/link{k}" for k in range(8)] + ["base_plate", "bracket",
                                                                                          "work_clamp", "clamp"]
    mj_data = SimpleNamespace(ncon=0)
    w = SimpleNamespace(
        joints=joints, arms=("left", "right"), torch_arm="right", tick=0, time=0., fault=None, events=[],
        reflexes=[], _reflex_until={}, refused_targets=0, circuit_closed=True, process=None, _tip_prev=None,
        model=SimpleNamespace(body_label=labels, shape_label=[f"shape{k}" for k in range(30)]),
        _robot_bodies=set(range(16)), _g2s=np.arange(30), _g2b=np.zeros(30, int),
        scene=SimpleNamespace(gun_shapes=[3, 4], tool0_in_body=(pt, Rt), gun_spec=spec, gun_body=gun_body),
        state=SimpleNamespace(body_q=arr(body_q), joint_q=arr(rng.normal(size=30)), joint_qd=arr(rng.normal(size=30))),
        control=SimpleNamespace(joint_target_q=arr(rng.normal(size=30))), targets=rng.normal(size=30),
        solver=SimpleNamespace(_mujoco=None, mj_model=None, mj_data=mj_data),
        bond=SimpleNamespace(active=False, broken=False, capacity_n=0., length_m=0., events=[],
                             load=dict(force_n=0., moment_nm=0., ratio=0.)),
        config=SimpleNamespace(rigid_dt=1e-3), torch_pose=torch_pose)
    if process:
        from welding.arc import ArcState, Settings
        pool = SimpleNamespace(slices=[SimpleNamespace(liquid=False) for _ in range(5)], dV=np.zeros(5), events=[],
                               frozen_volume=0., liquid_expansion=0., excess=0., spilled_m3=0., stacked_m3=0.)
        w.process = SimpleNamespace(src=SimpleNamespace(st=ArcState(), s=Settings()), pool=pool, events=[],
                                    axis_last=None, travel_angle=0., work_angle=None, ever_hot=False)
    return w


class TraceHookTest(unittest.TestCase):
    """The per-tick trace (welding/trace.py) on a stub world: the backend's hooks, one sample per tick in the shared
    store format, events, and the env's own per-tick history still recorded. No GPU (the ledger is stubbed)."""

    def test_backend_overrides_the_hooks(self):
        from gym_anything_robotics.backends.base import RoboticsBackend
        from gym_anything_robotics.backends.newton_welding import NewtonWeldingBackend
        for name in ("begin_action_trace", "end_action_trace", "variation_contract", "select_variation"):
            self.assertIsNot(getattr(NewtonWeldingBackend, name), getattr(RoboticsBackend, name), name)
        b = _stub_backend()
        b.trace = None
        self.assertFalse(b.begin_action_trace({"action_id": "a", "request_id": "r"})["available"])
        self.assertIsNone(b.end_action_trace())
        b.trace = Mock()
        b.trace.begin.return_value = {"available": True}
        self.assertTrue(b.begin_action_trace({"action_id": "a", "request_id": "r"})["available"])
        b.end_action_trace()
        b.trace.finish.assert_called_once_with()

    def test_store_batch_equals_appends(self):
        """PhysicsTraceStore.append_batch (the welding trace's buffered writer) stores the same rows, accounting and
        retention result as one append per step, including evictions under a tight byte budget."""
        import tempfile
        import numpy as np
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        steps = [(t, dict(tick=t, q=np.full(40, t, np.float32)), [dict(type="e", tick=t)] if t % 3 == 0 else [])
                 for t in range(1, 41)]
        steps.append((41, None, [dict(type="end", tick=41)]))
        with tempfile.TemporaryDirectory() as d:
            out = []
            for k, batched in enumerate((False, True)):
                st = PhysicsTraceStore(Path(d)/f"s{k}.sqlite3", max_bytes=4096, max_items=10_000)
                st.begin("t", "r", 0, 1e-3, {})
                if batched:
                    for b in range(0, len(steps), 7):
                        st.append_batch("t", steps[b:b + 7])
                else:
                    for t, sample, events in steps:
                        st.append("t", t, sample, events)
                rows = st.db.execute("SELECT kind,tick,event_type,payload FROM item ORDER BY seq").fetchall()
                head = st.db.execute("SELECT last_tick,captured,evicted_samples,evicted_events FROM trace").fetchone()
                out.append(([tuple(r) for r in rows], tuple(head)))
                st.close()
            self.assertEqual(out[0], out[1])
            self.assertGreater(out[0][1][2], 0)                    # the budget did evict samples

    def _trace(self, world, d):
        from welding.trace import Trace
        env = _StubEnv(world)
        return env, Trace(env, Path(d)/"trace.sqlite3", max_bytes=64*1024*1024)

    def test_samples_every_tick_and_events(self):
        import tempfile
        import numpy as np
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        w = _stub_world()
        with tempfile.TemporaryDirectory() as d:
            env, tr = self._trace(w, d)
            self.assertIn("_record", env.__dict__)                  # shadowed on the instance, not on the class
            env.act(3)                                              # untraced: the env's own history only
            self.assertEqual(env.recorded, 3)
            info = tr.begin({"action_id": "act1", "request_id": "req1"})
            self.assertTrue(info["available"])

            def step(world):
                if world.tick == 5:
                    world.events.append(dict(tick=world.tick, tool_io={"gas_on": True, "voltage_v": 29}))
                    world.reflexes.append(dict(tick=world.tick, arm="right", kind="force_limit", force_n=151.))
                if world.tick == 7:
                    world.bond.events.append(dict(tick=world.tick, event="bonded", capacity_n=np.float64(10.)))
            env.act(6, step)
            summary = tr.finish()
            self.assertEqual(env.recorded, 9)
            self.assertTrue(summary["complete"])
            self.assertEqual((summary["start_tick"], summary["end_tick"], summary["retained_samples"]), (3, 9, 6))
            store = PhysicsTraceStore(Path(d)/"trace.sqlite3", readonly=True)
            try:
                samples = store.page("act1", kind="sample", limit=16)["items"]
                events = [e["data"] for e in store.page("act1", kind="event", limit=16)["items"]]
                meta = store.summary("act1")["metadata"]["data"]
            finally:
                store.close()
            self.assertEqual([x["tick"] for x in samples], [4, 5, 6, 7, 8, 9])
            first = samples[0]["data"]
            self.assertEqual(len(first["joints"]["q"]), len(meta["joint_names"]))
            self.assertEqual(first["joints"]["q"], [np.float32(w.state.joint_q.numpy()[w.joints[n]["q"]]).item()
                                                     for n in meta["joint_names"]])
            _, _, tcp = w.torch_pose()
            np.testing.assert_allclose(first["torch"]["work_point_m"], tcp, atol=1e-9)
            self.assertEqual(set(first["contact"]["net_force_n"]), {"left", "right"})
            self.assertEqual(first["gun_contacts"], [])
            types = [e["type"] for e in events]
            self.assertEqual(types, ["tool_io", "protective_stop", "bond_made", "trace_cost"])
            self.assertEqual(events[0]["outputs"], {"gas_on": True, "voltage_v": 29.})
            self.assertEqual((events[1]["tick"], events[1]["world_tick"], events[1]["force_n"]), (5, 5, 151.))
            self.assertEqual(events[-1]["capture_errors"], 0)
            self.assertEqual(events[-1]["ticks"], 6)
            tr.close()
            self.assertNotIn("_record", env.__dict__)              # unbound: the class method again

    def test_source_events_and_deferred_ledger(self):
        """Source events, and the ledger: reused (evaluated false) while the coupon was never heated, then one device
        row per tick read back at the flush (the GPU reduction is replaced by a fake device buffer here)."""
        import tempfile
        import numpy as np
        w = _stub_world(process=True)
        w.process.s = SimpleNamespace(stored0=100.)
        w.process.e_input_arc = 0.
        with tempfile.TemporaryDirectory() as d:
            env, tr = self._trace(w, d)
            rows = np.zeros(tr.ROW*60)
            enqueued = []

            def enqueue(row):                       # stored enthalpy and the input grow by 2 J per hot tick
                enqueued.append(row)
                rows[tr.ROW*row:tr.ROW*row + 3] = (100. + 2.*len(enqueued), 1e-6, 2.*len(enqueued))
            tr._enqueue_sums = enqueue
            tr._dev = SimpleNamespace(numpy=lambda: rows.copy(), zero_=lambda: rows.fill(0.))
            tr._v0 = 0.
            start = dict(input_j=0., arc_delivered_j=0., stored_change_j=0., surface_loss_j=0., sink_loss_j=0.,
                         evaporation_j=0., deposited_metal_j=0., removed_metal_j=0., residual_j=0., residual_rel=0.,
                         metal_volume_change_m3=0.)

            def now():
                tr._ledger_cache = dict(start)
                return tr._ledger_cache
            tr._ledger_now = now
            tr.selfcheck = {}
            tr.begin({"action_id": "a2", "request_id": "r2"})
            st = w.process.src.st

            def step(world):
                if world.tick == 2:
                    st.arc, st.current, st.d_tw = True, 240., .019
                    world.process.ever_hot = True
                if world.tick == 4:
                    st.arc, st.fault = False, "arc_start_failure"
            env.act(5, step)
            summary = tr.finish()
            self.assertTrue(summary["complete"])
            from gym_anything_robotics.physics_trace import PhysicsTraceStore
            store = PhysicsTraceStore(Path(d)/"trace.sqlite3", readonly=True)
            try:
                events = [e["data"] for e in store.page("a2", kind="event", limit=16)["items"]]
                samples = [s_["data"] for s_ in store.page("a2", kind="sample", limit=16)["items"]]
            finally:
                store.close()
            self.assertEqual([e["type"] for e in events], ["arc_strike", "arc_extinguish", "fault", "trace_cost"])
            self.assertEqual(events[0]["current_a"], 240.)
            self.assertEqual([x["source"]["arc_on"] for x in samples], [False, True, True, False, False])
            self.assertEqual(samples[1]["source"]["ctwd_m"], .019)
            self.assertIsNone(samples[0]["source"]["ctwd_m"])        # no metal in reach (inf) -> null
            self.assertEqual(enqueued, [0, 1, 2, 3])                   # one row per hot tick, none before
            self.assertEqual([x["energy"]["evaluated"] for x in samples], [False, True, True, True, True])
            self.assertEqual([x["energy"]["stored_change_j"] for x in samples], [0., 2., 4., 6., 8.])
            self.assertEqual([x["energy"]["residual_j"] for x in samples], [0.]*5)
            self.assertEqual(samples[3]["pool"]["metal_volume_change_m3"], 1e-6)
            self.assertEqual(events[-1]["ledger_evaluations"], 4)
            tr.close()


@unittest.skipUnless(os.environ.get("GAR_WELDING_PHYSICS_TESTS") == "1",
                     "set GAR_WELDING_PHYSICS_TESTS=1 (builds the weld cell and its tacked start state, GPU)")
class PhysicsTest(unittest.TestCase):
    """position-torch through the public constructor: reset, the source's I/O (gas, inching, a rejected voltage), a
    small Cartesian move of the torch, an unreachable target rejected, evaluation, checkpoint restore. No RGB cameras
    (no Isaac worker)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spec = tasks.resolve("welding", {"task": "position-torch", "seed": 0})
        cls.env = registry.create("welding", auto_reset=False, cameras=(),
                                  output_dir=str(Path(cls.tmp.name)/"session"), **cls.spec.launch.backend_options)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.tmp.cleanup()

    def test_position_torch_round_trip(self):
        from gym_anything_robotics.actions import CartesianDelta, CartesianPose, ToolIO
        env = self.env
        observation, info = env.reset()
        self.assertEqual(info["continuity"], "reset")
        tool = observation.proprioception["tool"]
        self.assertEqual(set(tool), {"arc_on", "current_a", "voltage_v", "wire_feed_m_min", "touch", "fault",
                                     "gas_on"})
        self.assertEqual((tool["arc_on"], tool["gas_on"]), (0., 0.))

        tick = env.get_state()["tick"]
        with self.assertRaises(ActionRejectedError):
            env.act(ToolIO("right", {"voltage_v": 60.}, 1))
        with self.assertRaises(ActionRejectedError):
            env.act(CartesianPose("right", (3., 3., 3.)))
        self.assertEqual(env.get_state()["tick"], tick)

        step = env.act(ToolIO("right", {"gas_on": True}, 100))
        self.assertEqual(step.info["executed_ticks"], 100)
        self.assertEqual(step.observation.proprioception["tool"]["gas_on"], 1.)
        step = env.act(ToolIO("right", {"gas_on": False, "wire_inch": "forward"}, 200))
        step = env.act(ToolIO("right", {"wire_inch": "stop"}, 1))

        world = env._backend.env.world
        z0 = float(world.kin.forward("right")[2, 3])
        step = env.act(CartesianDelta("right", (0., 0., .01), duration_s=.5))
        z1 = float(world.kin.forward("right")[2, 3])
        self.assertGreater(z1 - z0, 5e-3, f"torch work point rose only {1e3*(z1 - z0):.2f} mm")
        self.assertEqual(step.observation.proprioception["tool"]["arc_on"], 0.)

        result = env.evaluate()
        self.assertTrue(result["available"])
        self.assertIs(type(result["success"]), bool)
        self.assertTrue(math.isfinite(result["score"]))
        print(f"\n[welding physics] torch +{1e3*(z1 - z0):.2f} mm in 500 ticks; physics {step.info['physics_s']:.1f} s; "
              f"evaluation success={result['success']} score={result['score']:.3f}")

        tick = env.get_state()["tick"]
        saved = env.save_checkpoint(label="raised")
        self.assertEqual(saved["tick"], tick)
        env.act(ToolIO("right", {"gas_on": True}, 10))
        self.assertEqual(env.get_state()["tick"], tick + 10)
        observation, info = env.restore_checkpoint("raised")
        self.assertEqual(info["continuity"], "restored_discontinuity")
        self.assertEqual(env.get_state()["tick"], tick)


@unittest.skipUnless(os.environ.get("GAR_WELDING_PHYSICS_TESTS") == "1",
                     "set GAR_WELDING_PHYSICS_TESTS=1 (builds the weld cell for each task, GPU)")
class ResetGasPhysicsTest(unittest.TestCase):
    """After reset the shielding gas valve is closed for every task: the robot program opens it (tool_io gas_on). The
    tacked start states close it after the fixture's post-flow (env.py)."""

    def test_gas_valve_closed_after_reset_for_all_tasks(self):
        import tempfile
        from gym_anything_robotics import registry
        for task in ("tack", "position-torch", "fillet-2f"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                spec = tasks.resolve("welding", {"task": task, "seed": 0})
                env = registry.create("welding", auto_reset=False, cameras=(), trace=False,
                                      output_dir=str(Path(tmp)/"session"), **spec.launch.backend_options)
                try:
                    observation, _ = env.reset()
                    state = env.get_state()
                    self.assertEqual(observation.proprioception["tool"]["gas_on"], 0.,
                                     f"{task}: the policy's feedback reports the gas valve open after reset")
                    self.assertIs(state["tool"]["gas_on"], False)
                    self.assertEqual(observation.proprioception["tool"]["arc_on"], 0.)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()

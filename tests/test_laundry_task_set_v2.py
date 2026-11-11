"""Contract `laundry-tasks-v2` (the render-checked eight): constructed-state and stubbed adapter
tests, with no simulator runtime. Ported from the fix round's tests.

The released `laundry-tasks-v1` set, the bare garment selections and the original shirt keep
their own tests (tests/environments/laundry/test_task_set.py); here they are checked only where
the v2 dispatch could reach them.
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from laundry.garment_task import GARMENT_TASK_SPECS, GarmentTaskEvaluator, TabletopFrame
from laundry.task_set import (CONTRACT as V1_CONTRACT, LEGACY, SHIRT_CONTRACT, TASK_SET as V1_TASK_SET,
                              V2_CONTRACT, make_evaluator as make_original_evaluator, resolve as resolve_any,
                              select as select_original, task_spec as original_task_spec)
from laundry.task_set_v2 import (BY_ID, CONTRACT, TASK_SET_ORDER as EIGHT_TASKS, TASKS, TASK_SET, make_evaluator,
                                 resolve, select, task_spec)
from laundry.task_set_v2_presets import accordion_curve, reach_report, start_positions
from laundry.task_set_v2_goals import sheet_crossings, signed_turning, three_ply_fraction
from tests.environments.laundry.task_set_fixtures import half_fold, recorded, thirds_fold
from tests.environments.laundry.task_set_v2_fixtures import accordion, letter_fold, quarter_fold, rolled, tee_half_fold
from tests.packages import PACKAGES, REPO


# Task names of the registered entries; half_fold here is the towel's (the tee's is "tee/half_fold").
GARMENT_OF = {"spread_flat": "towel", "half_fold": "towel", "thirds_fold": "towel", "roll": "towel",
              "tie_untangle": "asymmetric", "leg_align_fold": "jeans", "sleeve_fold": "shirt",
              "sleeve_tuck_fold": "tee", "turn_face_down": "tee", "tee/half_fold": "tee"}


def key(name):
    return (GARMENT_OF[name], name.split("/")[-1])


def evaluator(name, *, start=None):
    entry = BY_ID[key(name)]
    ix, flat = recorded(entry.garment)
    x = start_positions(entry, ix, flat, .78) if start is None else start.copy()
    if entry.garment == "shirt":
        # The eighth task is the original shirt selection, served by the released code.
        return make_original_evaluator(entry, SHIRT_CONTRACT, ix, x), x
    return make_evaluator(entry, CONTRACT, ix, x), x


def judge(ev, x, **changes):
    kw = dict(maximum_structural_strain=0., nonlocal_intersection_count=0,
              runtime_pose_writes=0, external_particle_force_applications=0,
              released=True, fixtures_clear=True, tabletop=TabletopFrame(.78, .80, .70),
              recent_positions_m=np.stack([x, x]), recent_span_s=.5, principal_stretch_range=(1., 1.))
    kw.update(changes)
    return ev.evaluate(x, np.zeros_like(x), **kw)


class CatalogueTests(unittest.TestCase):
    def test_eight_exact_selections_and_original_order(self):
        d = json.loads((PACKAGES / "laundry/newton15/benchmark.json").read_text())
        self.assertEqual(CONTRACT, V2_CONTRACT)
        self.assertEqual(d["task_set"]["selections"], list(TASK_SET))
        self.assertEqual(d["task_set"]["contract"], CONTRACT)
        self.assertIn(CONTRACT, d["selector_choices"]["contract"])
        self.assertIn("roll", d["selector_choices"]["task"])
        # The released set is kept under task_set_v1, unchanged.
        self.assertEqual(d["task_set_v1"]["selections"], list(V1_TASK_SET))
        self.assertEqual(d["task_set_v1"]["contract"], V1_CONTRACT)
        self.assertEqual([t.name for t in TASKS[:5]],
                         ["sleeve_tuck_fold", "sleeve_fold", "leg_align_fold", "half_fold", "tie_untangle"])
        self.assertEqual([(t.garment, t.name) for t in TASKS[-2:]], [("tee", "half_fold"), ("towel", "roll")])
        self.assertEqual(len({resolve_any(s)["task_id"] for s in TASK_SET}), 8)
        # The audited hardest task keeps its released split selection exactly.
        test = json.loads((REPO / "benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        self.assertEqual(TASK_SET[-1], next(t["selection"] for t in test if t["environment_id"] == "laundry"))

    def test_swapped_out_tasks_stay_selectable_outside_the_eight(self):
        d = json.loads((PACKAGES / "laundry/newton15/benchmark.json").read_text())
        out = [s["selection"] for s in d["task_set"]["swapped_out"]]
        self.assertEqual(out, [{"garment": "asymmetric", "task": "tie_untangle", "contract": CONTRACT},
                               {"garment": "jeans", "task": "leg_align_fold", "contract": CONTRACT}])
        for s in out:
            self.assertNotIn(s, TASK_SET)
            self.assertEqual(resolve_any(s)["selection"], s)

    def test_v2_selections_resolve_through_the_released_resolver_entry_point(self):
        for s in TASK_SET:
            r = resolve_any(s)
            if s.get("contract") == CONTRACT:
                self.assertEqual(r, resolve(s))
                self.assertEqual(r["backend_options"]["contract"], CONTRACT)
                self.assertEqual(r["description"], BY_ID[s["garment"], s["task"]].label +
                                 ". Use the robot arms; leave the garment released and resting on the table.")
            else:
                from laundry.newton15.benchmark import resolve as old
                self.assertEqual(r, old(s))
        self.assertEqual(resolve_any({"garment": "tee", "task": "half_fold", "contract": CONTRACT})["task_id"],
                         "laundry/tee/half_fold/laundry-tasks-v2")
        self.assertEqual(resolve_any({"garment": "towel", "task": "roll", "contract": CONTRACT})["task_id"],
                         "laundry/towel/roll/laundry-tasks-v2")

    def test_original_bare_selections_are_unchanged(self):
        from laundry.newton15.benchmark import resolve as old
        for s in ({}, *({"garment": g} for g in ("tee", "shirt", "jeans", "towel", "asymmetric"))):
            self.assertEqual(resolve_any(s), old(s))
        self.assertEqual(select_original()[1], LEGACY)

    def test_gar_manifest_round_trip_without_runtime(self):
        from gym_anything_robotics.tasks import resolve as gar_resolve, TaskSpec
        for s in TASK_SET:
            result = gar_resolve("laundry", s)
            self.assertEqual(TaskSpec.from_dict(result.to_dict()).to_dict(), result.to_dict())
            expected = resolve_any(s)
            self.assertEqual(result.launch.backend_options, expected["backend_options"])
            self.assertEqual(result.description, expected["description"])

    def test_v2_resolution_does_not_load_simulator(self):
        # A fresh interpreter: other test modules collected in the same run may already have imported warp.
        code = """import sys
from gym_anything_robotics.tasks import resolve
from laundry.task_set_v2 import TASK_SET
for s in TASK_SET: resolve("laundry", s)
assert not any(k in sys.modules for k in ("laundry.newton15.simulation", "warp", "newton")), sorted(sys.modules)
"""
        paths = [str(REPO / "src"), str(REPO / "benchmarks/roboworld/environments"), str(REPO)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")]))
        subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True)

    def test_unknown_and_incompatible_selectors(self):
        for args in (("towel", "bad", CONTRACT), ("tee", "spread_flat", CONTRACT),
                     ("shirt", None, CONTRACT), ("towel", "spread_flat", LEGACY), ("towel", "spread_flat", V1_CONTRACT),
                     ("towel", None, "unknown"), ("towel", 1, CONTRACT), (None, None, CONTRACT)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                select(*args)
        # The released selector never accepts the new contract or the new tasks.
        for args in (("towel", "roll", CONTRACT), ("tee", "half_fold", CONTRACT), ("towel", "roll", None),
                     ("tee", "half_fold", None), ("towel", "spread_flat", CONTRACT)):
            with self.subTest(released=args), self.assertRaises(ValueError):
                select_original(*args)
        with self.assertRaises(ValueError):
            resolve_any({"garment": "shirt", "contract": CONTRACT})

    def test_audited_spec_is_the_same_object(self):
        entry, version = select_original("shirt")
        self.assertEqual(version, SHIRT_CONTRACT)
        old = GARMENT_TASK_SPECS[("woven_button_down_shirt", "sleeve_fold")]
        self.assertIs(original_task_spec(entry, version), old)
        ix, x = recorded("shirt")
        new = make_original_evaluator(entry, version, ix, x)
        self.assertIs(type(new), GarmentTaskEvaluator)
        prior = GarmentTaskEvaluator(old, ix, reference_positions_m=x)
        self.assertEqual(judge(new, x), judge(prior, x))
        with self.assertRaises(ValueError):
            task_spec(entry, version)


class StateTests(unittest.TestCase):
    def test_all_eight_untouched_states_score_zero_even_with_rest_window(self):
        for garment, name in EIGHT_TASKS:
            with self.subTest(task=f"{garment}/{name}"):
                ev, x = evaluator(name if garment != "tee" or name != "half_fold" else "tee/half_fold")
                result = judge(ev, x)
                self.assertFalse(result.success)
                self.assertAlmostEqual(result.potential, 0., places=12)
                self.assertAlmostEqual(result.progress, 0., places=12)

    def test_flat_spread_success_and_incomplete_states(self):
        ev, start = evaluator("spread_flat")
        ix, flat = recorded("towel")
        self.assertEqual(judge(ev, flat).potential, 1.)
        self.assertFalse(judge(ev, half_fold(flat)).success)
        raised_corner = flat.copy(); raised_corner[0, 2] += .08
        self.assertFalse(judge(ev, raised_corner).success)

    def test_half_fold_success_and_progress_between_boolean_transitions(self):
        ev, x = evaluator("half_fold")
        folded = half_fold(x)
        self.assertTrue(judge(ev, folded).success)
        self.assertEqual(judge(ev, folded).potential, 1.)
        scores = []
        for alpha in (.10, .11, .12, .13):
            state = (1-alpha)*x + alpha*folded
            r = judge(ev, state)
            self.assertFalse(r.success)
            scores.append(r.potential)
        self.assertTrue(all(b > a for a, b in zip(scores, scores[1:])), scores)

    def test_ties_extend_and_progress_is_continuous(self):
        ev, x = evaluator("tie_untangle")
        ix, flat = recorded("asymmetric")
        result = judge(ev, flat)
        self.assertTrue(result.success, result.failed_gates)
        self.assertEqual(result.potential, 1.)
        one = x.copy()
        ids = ix.panel_vertices(("left_tie",))
        one[ids] = flat[ids]
        self.assertFalse(judge(ev, one).success)
        a, b = judge(ev, .9*x + .1*flat), judge(ev, .89*x + .11*flat)
        self.assertGreater(b.potential, a.potential)

    def test_face_down_uses_independent_panels_and_flatness(self):
        ev, x = evaluator("turn_face_down")
        self.assertEqual(judge(ev, x).metrics.goal_metrics["face_down"]["flipped_fraction"], 0.)
        turned = x.copy(); turned[:, 0] *= -1; turned[:, 2] = x[:, 2].max() + x[:, 2].min() - x[:, 2]
        result = judge(ev, turned)
        self.assertTrue(result.success, result.metrics.goal_metrics)
        self.assertEqual(result.potential, 1.)
        bunched = turned.copy(); bunched[:, :2] *= .45
        partial = judge(ev, bunched)
        self.assertFalse(partial.success)
        self.assertGreater(partial.progress, 0.)
        self.assertLess(partial.potential, .9)

    def test_thirds_accepts_c_and_z_fold_rejects_half_roll_and_ball(self):
        ev, x = evaluator("thirds_fold")
        for z in (False, True):
            folded = thirds_fold(x, z_fold=z)
            r = judge(ev, folded)
            self.assertTrue(r.success, (z, r.failed_gates, r.metrics.goal_metrics))
            self.assertEqual(r.potential, 1.)
        self.assertFalse(judge(ev, half_fold(x)).success)
        roll = x.copy(); theta = (x[:, 0] - x[:, 0].min()) / .018
        roll[:, 0] = .018*np.sin(theta); roll[:, 2] = .805 + .018*np.cos(theta)
        self.assertFalse(judge(ev, roll).success)
        ball = x.copy(); ball[:, :2] *= .2
        self.assertFalse(judge(ev, ball).success)

    def test_equal_letter_fold_passes_c_and_z_sharp_or_rounded_three_to_seven_mm(self):
        ev, x = evaluator("thirds_fold")
        for z in (False, True):
            for rounded in (False, True):
                for gap in (.003, .004, .005, .006, .007):
                    with self.subTest(z_fold=z, rounded=rounded, gap=gap):
                        r = judge(ev, letter_fold(x, gap=gap, z_fold=z, rounded=rounded))
                        self.assertTrue(r.success, (r.failed_gates, r.metrics.goal_metrics))
                        self.assertGreaterEqual(r.metrics.goal_metrics["thirds_fold"]["three_ply_fraction"], .90)
                        self.assertEqual(r.potential, 1.)

    def test_unequal_thirds_and_quarter_fold_fail_only_on_plies(self):
        ev, x = evaluator("thirds_fold")
        tris = ev.index.topology.physical_triangles
        # Middle 165-180 mm: flaps of the rest cover 73-50 % of it (80 % needs about 160 mm).
        for middle in (.165, .170, .180):
            for z in (False, True):
                with self.subTest(middle=middle, z_fold=z):
                    flap = (np.ptp(x[:, 0]) - middle) / 2
                    r = judge(ev, letter_fold(x, (flap, middle, flap), z_fold=z, rounded=False))
                    self.assertEqual(r.failed_gates, ("goal:thirds_fold:three_plies",))
                    self.assertLess(r.metrics.goal_metrics["thirds_fold"]["three_ply_fraction"], .80)
        r = judge(ev, quarter_fold(x))
        self.assertEqual(r.failed_gates, ("goal:thirds_fold:three_plies",))
        self.assertEqual(three_ply_fraction(quarter_fold(x), tris), 0.)
        self.assertEqual(three_ply_fraction(half_fold(x), tris), 0.)

    def test_ply_count_merges_only_one_sheets_shared_edge(self):
        tris = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]])
        heights = np.array([.0, .0, .0])           # two faces of one sheet + another sheet, same height
        flat = np.ones(3)
        levels, _ = sheet_crossings(heights, flat, tris)
        self.assertEqual(len(levels), 2)
        levels, _ = sheet_crossings(np.array([.0, .0015, np.nan]), flat, tris)
        self.assertEqual(len(levels), 2)          # a tilted crease cell 1.5 mm from its ply is its own crossing

    def test_thirds_mirror_and_translation_do_not_select_a_hand_or_world_fold_line(self):
        ev, x = evaluator("thirds_fold")
        folded = thirds_fold(x)
        for angle in (0., .4, 1.3):
            y = folded.copy(); c, s = np.cos(angle), np.sin(angle)
            y[:, :2] = y[:, :2] @ np.array([[c, -s], [s, c]]) + [.05, -.04]
            r = judge(ev, y)
            self.assertTrue(r.success, (angle, r.failed_gates, r.metrics.goal_metrics))
        y = folded.copy(); y[:, 0] *= -1
        self.assertTrue(judge(ev, y).success)

    def test_column_count_rejects_degenerate_and_exploded_cloth_without_unbounded_work(self):
        ev, x = evaluator("thirds_fold")
        tris = ev.index.topology.physical_triangles
        line = x.copy(); line[:, 0] = 0.
        self.assertEqual(three_ply_fraction(line, tris), 0.)
        exploded = x.copy(); exploded[:, :2] *= 1000
        self.assertEqual(three_ply_fraction(exploded, tris), 0.)

    def test_shared_gates_never_award_full_credit_on_failure(self):
        ev, x = evaluator("half_fold"); done = half_fold(x)
        for kw in (dict(released=False), dict(fixtures_clear=False), dict(recent_span_s=.4),
                   dict(principal_stretch_range=(1., 1.21)), dict(nonlocal_intersection_count=1),
                   dict(runtime_pose_writes=1), dict(external_particle_force_applications=1)):
            with self.subTest(kw=kw):
                r = judge(ev, done, **kw)
                self.assertFalse(r.success)
                self.assertLessEqual(r.potential, .9)
                self.assertAlmostEqual(r.potential, .9*r.progress)
        r = judge(ev, done, nonlocal_intersection_count=1)
        self.assertEqual(r.potential, 0.)
        off = done.copy(); off[:, 0] += 2
        self.assertFalse(judge(ev, off).success)
        moving = done.copy(); moving[:, 2] += .011
        self.assertFalse(judge(ev, done, recent_positions_m=np.stack([moving, done])).success)

    def test_compression_is_allowed_without_loosening_stretch(self):
        for name, limit in (("half_fold", .20), ("leg_align_fold", .25)):
            ev, x = evaluator(name)
            self.assertTrue(judge(ev, x, maximum_structural_strain=.8,
                                  principal_stretch_range=(.2, 1.)).gates["integrity:structural_strain"])
            self.assertFalse(judge(ev, x, principal_stretch_range=(.2, 1+limit+.001)).gates["integrity:structural_strain"])

    def test_tee_and_jeans_constructed_goal_states_receive_full_credit(self):
        for name in ("sleeve_tuck_fold", "leg_align_fold"):
            ev, x = evaluator(name); ix = ev.index
            if name == "sleeve_tuck_fold":
                host = ix.panel_vertices(("front", "back"))
                # Geometric sleeve folds; no simulation/retention claim.
                for side, sign in (("left", -1), ("right", 1)):
                    moving = np.setdiff1d(ix.panel_vertices((f"{side}_sleeve",)), host)
                    x[moving, 0] = 2 * .22 * sign - x[moving, 0]
                    x[moving, 2] += .007
            else:
                # Right leg reflected onto the left, keeping the actual hem shape.
                ids = ix.panel_vertices(("front_right", "back_right"))
                x[ids, 0] *= -1
                x[ids, 2] += .007
            r = judge(ev, x)
            self.assertTrue(r.success, (name, r.failed_gates))
            self.assertEqual(r.potential, 1.)

    def test_tee_half_fold_either_half_over_and_rejects_crosswise_and_refold(self):
        ev, x = evaluator("tee/half_fold")
        for left_over in (True, False):
            folded = tee_half_fold(x, left_over=left_over)
            r = judge(ev, folded)
            self.assertTrue(r.success, (left_over, r.failed_gates, r.metrics.goal_metrics))
            self.assertEqual(r.potential, 1.)
            for angle in (.5, 2.):          # judged on the garment, not a world fold line
                y = folded.copy(); c, s = np.cos(angle), np.sin(angle)
                y[:, :2] = y[:, :2] @ np.array([[c, -s], [s, c]]) + [.03, -.02]
                self.assertTrue(judge(ev, y).success)
        crosswise = judge(ev, tee_half_fold(x, crosswise=True))
        self.assertFalse(crosswise.success)
        self.assertIn("goal:side_seam_alignment:rms", crosswise.failed_gates)
        # Folded in half again (neck half onto hem half): seams still together, footprint a quarter.
        again = judge(ev, tee_half_fold(tee_half_fold(x), crosswise=True, gap=.024))
        self.assertEqual(again.failed_gates, ("goal:half_footprint:not_refolded",))
        scores = [judge(ev, (1 - a) * x + a * tee_half_fold(x)).potential for a in (.2, .25, .3)]
        self.assertTrue(0 < scores[0] < scores[1] < scores[2] < .9, scores)

    def test_roll_accepts_rolls_from_either_end_and_rejects_folds(self):
        ev, x = evaluator("roll")
        for from_top in (True, False):
            for radius, pitch in ((.030, .015), (.020, .010)):
                with self.subTest(from_top=from_top, radius=radius):
                    r = judge(ev, rolled(x, inner_radius=radius, pitch=pitch, from_top=from_top))
                    self.assertTrue(r.success, (r.failed_gates, r.metrics.goal_metrics))
                    self.assertEqual(r.potential, 1.)
        done = rolled(x)
        for angle in (.4, 1.3):              # judged about the roll's own axis, anywhere on the table
            y = done.copy(); c, s = np.cos(angle), np.sin(angle)
            y[:, :2] = y[:, :2] @ np.array([[c, -s], [s, c]]) + [.05, -.04]
            self.assertTrue(judge(ev, y).success)

        def end_to_end(bends):
            s = x[:, 1] - x[:, 1].min(); u, z = accordion(s, bends)
            y = x.copy(); y[:, 1] = u - (u.min() + u.max()) / 2; y[:, 2] = .790 + z - z.min()
            return y
        g, L = .005, np.ptp(x[:, 1])
        w = (L - np.pi * 1.5 * g) / 3
        folds = {"half end to end": end_to_end((((L - np.pi * g / 2) / 2, g / 2, 1, 0.),)),
                 "C fold end to end": end_to_end(((w, g / 2, -1, 0.), (2 * w + np.pi * g / 2, g, 1, 0.))),
                 "Z fold end to end": end_to_end(((w, g / 2, -1, 0.), (2 * w + np.pi * g / 2, g / 2, -1, 0.))),
                 "half lengthwise": half_fold(x), "rolled .35 m only": rolled(x, length=.35)}
        for name, state in folds.items():
            with self.subTest(other=name):
                r = judge(ev, state)
                self.assertFalse(r.success)
                self.assertTrue({"goal:roll:wound", "goal:roll:width"} & set(r.failed_gates), r.failed_gates)
        # Rolled from a long edge: the short hems curl, the long hems do not wind.
        side = x.copy(); s = x[:, 0] - x[:, 0].min()
        u, z = accordion(s, tuple((k * .07, .012, 1, 0.) for k in range(1, 6)))
        side[:, 0], side[:, 2] = u, .790 + z - z.min()
        self.assertFalse(judge(ev, side).success)
        # Progress grows with the length rolled, before any gate passes.
        scores = [judge(ev, rolled(x, length=length)).potential for length in (.15, .25, .35)]
        self.assertTrue(0 < scores[0] < scores[1] < scores[2] < .9, scores)
        self.assertAlmostEqual(signed_turning(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.], [1., 0.]])),
                               2 * np.pi)

    def test_area_and_hem_thresholds_are_independent(self):
        ev, x = evaluator("spread_flat")
        _, flat = recorded("towel")
        narrow = flat.copy(); narrow[:, 0] *= .89
        r = judge(ev, narrow)
        self.assertFalse(r.gates["goal:spread_flat:hems"])
        short = flat.copy(); short[:, 1] *= .90
        r = judge(ev, short)
        self.assertTrue(r.gates["goal:spread_flat:hems"])
        self.assertFalse(r.gates["goal:spread_flat:area"])

    def test_new_tie_placements_are_within_nearest_arm_reach(self):
        ev, start = evaluator("tie_untangle")
        for panel in ("left_tie", "right_tie"):
            ids = ev.index.panel_vertices((panel,))
            self.assertEqual(reach_report(start[ids])["outside_085m_count"], 0)
        # Document the pre-existing body reach issue instead of moving or freezing it.
        self.assertGreater(reach_report(start)["outside_085m_count"], 0)

    def test_authored_starts_are_crossing_free_and_ties_bunched(self):
        try:
            import ipctk  # noqa: F401  (author_state runs this audit before accepting a start)
        except ImportError:
            self.skipTest("IPC Toolkit is not installed")
        from laundry.self_intersection import IPCTrianglePairConfirmer
        for name in ("spread_flat", "tie_untangle"):
            with self.subTest(task=name):
                ev, x = evaluator(name)
                audit = IPCTrianglePairConfirmer(ev.index.topology.physical_triangles)
                self.assertEqual(len(audit.surface_intersection_pairs(x.astype(np.float32))), 0)
        ev, x = evaluator("tie_untangle")
        old = GARMENT_TASK_SPECS[(ev.index.construction.id, "tie_untangle")]
        fractions = [g.evaluate(ev.index, x, np.array([0., 0., 1.])).metrics["end_to_end_reference_fraction"]
                     for g in old.goals]
        for f in fractions:   # the adapter accepts .30-.50 after settling, which can lose ~.1
            self.assertGreater(f, .42)
            self.assertLess(f, .48)

    def test_presets_do_not_mutate_input_or_material_geometry(self):
        for name in ("spread_flat", "tie_untangle"):
            entry = BY_ID[key(name)]; ix, flat = recorded(entry.garment)
            old = flat.copy(); material = ix.topology.panel_material_coordinates_m.copy()
            start_positions(entry, ix, flat, .78)
            np.testing.assert_array_equal(old, flat)
            np.testing.assert_array_equal(material, ix.topology.panel_material_coordinates_m)
        s = np.linspace(0, .58, 5801)
        u, h = accordion_curve(s, .58)
        speed = np.sqrt(np.diff(u)**2 + np.diff(h)**2) / np.diff(s)
        np.testing.assert_allclose(speed, 1., atol=2.e-6)

    def test_new_folded_towel_is_within_reach_bound(self):
        _, start = evaluator("spread_flat")
        self.assertEqual(reach_report(start)["outside_085m_count"], 0)
        for y in (-.35, .35):
            corner = start[np.isclose(start[:, 1], y)]
            self.assertLess(reach_report(corner)["maximum_nearest_base_distance_m"], .50)


class Array:
    def __init__(self, value):
        self.assign(value)

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.array(value, dtype=np.float32)


class State:
    def __init__(self, x):
        self.particle_q = Array(x)

    def assign(self, other):
        self.particle_q.assign(other.particle_q.numpy())


class StubLaundryEnv:
    """Frozen-state integration stub: act records elapsed time, never simulates."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.tick, self.generation, self.fault = 0, 0, None
        self.author_calls = self.act_calls = 0
        self.closed = False
        self.joints = {}
        ix, x = recorded(config.garment)
        self.state = State(x)
        self.sim = SimpleNamespace(construction=ix.construction, pattern_mesh=ix.pattern_mesh,
                                   topology=ix.topology, reset_positions=x.copy(), initial=State(x), step_count=0)
        self.sim.author_state = self.author_state
        entry, version = select_original(config.garment)
        self.task = make_original_evaluator(entry, version, ix, self.state.particle_q.numpy())
        self.tabletop = TabletopFrame(.78, .80, .70)

    def author_state(self, x):
        self.author_calls += 1
        self.state.particle_q.assign(x)

    def _record_motion(self, *, reset=False):
        if reset:
            self.motion.clear()
        self.motion.append((self.tick, self.state.particle_q.numpy()))

    def evaluate(self):
        x = self.state.particle_q.numpy()
        r = judge(self.task, x, recent_span_s=self.tick*self.config.dt)
        return dict(success=r.success, score=r.potential, gates=dict(r.gates),
                    failed_gates=list(r.failed_gates), goal_scores=dict(r.goal_scores), progress=r.progress)

    def get_state(self):
        return dict(tick=self.tick, evaluation=self.evaluate())

    def act(self, targets, *, ticks):
        self.act_calls += 1
        self.tick += ticks
        self.sim.step_count = self.tick
        return self.get_state()

    def reset(self):
        self.state.assign(self.sim.initial)
        self.tick = self.sim.step_count = 0
        self.generation += 1
        return self.get_state()

    def close(self):
        self.closed = True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        fake = ModuleType("laundry.newton15.env")
        fake.LaundryEnv = StubLaundryEnv
        self.modules = patch.dict(sys.modules, {"laundry.newton15.env": fake})
        self.modules.start()
        sys.modules.pop("laundry.task_set_env", None)
        sys.modules.pop("laundry.task_set_v2_env", None)
        self.original_adapter = importlib.import_module("laundry.task_set_env").CatalogueLaundryEnv
        self.adapter = importlib.import_module("laundry.task_set_v2_env").CatalogueLaundryEnv
        self.addCleanup(self.modules.stop)

    def test_every_selection_reaches_adapter_and_reports_contract(self):
        from laundry.newton15.config import Config
        for s in TASK_SET:
            if s["garment"] == "shirt":
                # The original shirt selection never reaches the v2 adapter; the released adapter
                # serves it with its released evaluation fields.
                with self.assertRaises(ValueError):
                    self.adapter(Config(garment="shirt"), task=s.get("task"), contract=s.get("contract"))
                env = self.original_adapter(Config(garment="shirt"), task=s.get("task"), contract=s.get("contract"))
                result = env.get_state()["evaluation"]
                self.assertEqual(env.task_contract, SHIRT_CONTRACT)
                self.assertEqual(result["contract"], SHIRT_CONTRACT)
                self.assertNotIn("start_settling_s", result)
            else:
                env = self.adapter(Config(garment=s["garment"]), task=s.get("task"), contract=s.get("contract"))
                result = env.get_state()["evaluation"]
                self.assertEqual(result["contract"], CONTRACT)
                self.assertEqual(result["start_settling_s"], env.start_report["settling_seconds"])
            self.assertEqual(result["task_id"], resolve_any(s)["task_id"])
            self.assertEqual(env.task.spec.label + ". Use the robot arms; leave the garment released and resting on the table.",
                             resolve_any(s)["description"])
            self.assertAlmostEqual(result["score"], 0., places=12)
            self.assertFalse(result["success"])

    def test_resolver_options_construct_real_backend_and_policy_instruction(self):
        from gym_anything_robotics.backends.newton_laundry import NewtonLaundryBackend
        catalog = ModuleType("laundry.newton15.catalog")
        catalog.Catalog = lambda env: SimpleNamespace()
        kinematics = ModuleType("laundry.newton15.kinematics")
        kinematics.Kinematics = lambda base_y: SimpleNamespace()
        with patch.dict(sys.modules, {"laundry.newton15.catalog": catalog,
                                      "laundry.newton15.kinematics": kinematics}):
            with tempfile.TemporaryDirectory() as folder:
                for i, selection in enumerate((*TASK_SET, V1_TASK_SET[0])):
                    resolved = resolve_any(selection)
                    backend = NewtonLaundryBackend(**resolved["backend_options"], cameras=(), recording=False,
                                                   output_dir=Path(folder) / str(i))
                    v2 = selection.get("contract") == CONTRACT
                    self.assertIs(type(backend.env), self.adapter if v2 else self.original_adapter)
                    self.assertEqual(backend.policy_access().instruction(), resolved["description"])
                    self.assertEqual(backend.get_state()["evaluation"]["task_id"], resolved["task_id"])
                    state, info = backend.reset()
                    self.assertEqual(state["evaluation"]["score"], 0.)
                    self.assertEqual(info, {"continuity": "reset"})
                    backend.close()
                # A bare legacy selection keeps the released adapter and evaluator.
                backend = NewtonLaundryBackend(**resolve_any({"garment": "towel"})["backend_options"], cameras=(),
                                               recording=False, output_dir=Path(folder) / "legacy")
                self.assertIs(type(backend.env), self.original_adapter)
                self.assertIs(type(backend.env.task), GarmentTaskEvaluator)
                backend.close()

    def test_authored_start_is_installed_once_and_reset_reuses_accepted_snapshot(self):
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]
        blocks = round(module.SETTLE_MINIMUM_S / .5)
        for name in ("spread_flat", "tie_untangle"):
            entry = BY_ID[key(name)]
            env = self.adapter(Config(garment=entry.garment), task=name, contract=CONTRACT)
            self.assertEqual(env.author_calls, 1)
            self.assertEqual(env.act_calls, blocks)
            self.assertEqual(env.tick, 0)
            self.assertEqual(env.sim.step_count, 0)
            before = env.state.particle_q.numpy()
            changed = before.copy(); changed[:, 0] += .03
            env.state.particle_q.assign(changed)
            result = env.reset()
            np.testing.assert_array_equal(env.state.particle_q.numpy(), before)
            self.assertEqual(result["evaluation"]["score"], 0.)
            self.assertEqual(env.author_calls, 1)
            self.assertEqual(env.act_calls, blocks)

    def test_every_new_contract_start_is_zeroed_after_settling(self):
        """The catalogue lays drift while they settle; the zero point is the settled garment."""
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]
        for s in TASK_SET:
            if s["garment"] == "shirt":
                continue
            with self.subTest(task=s["task"]):
                drift = []

                def act(env, targets, *, ticks):
                    # Creep that halves every block, as an untouched garment coming to rest.
                    env.act_calls += 1
                    env.tick += ticks
                    env.sim.step_count = env.tick
                    x = env.state.particle_q.numpy()
                    x[:, :2] *= 1 - .02 * .5 ** env.act_calls
                    env.state.particle_q.assign(x)
                    drift.append(env.act_calls)
                    return env.get_state()

                with patch.object(StubLaundryEnv, "act", autospec=True, side_effect=act):
                    env = self.adapter(Config(garment=s["garment"]), task=s["task"], contract=CONTRACT)
                settled = env.state.particle_q.numpy()
                self.assertGreaterEqual(env.start_report["settling_seconds"], module.SETTLE_MINIMUM_S)
                for d in env.start_report["settling_quality_drift"][-module.SETTLE_STEADY_BLOCKS:]:
                    self.assertLessEqual(abs(d), module.SETTLE_QUALITY_TOLERANCE)
                for m in env.start_report["settling_block_motion_m"][-module.SETTLE_STEADY_BLOCKS:]:
                    self.assertLessEqual(m, module.SETTLE_MOTION_TOLERANCE_M)
                np.testing.assert_array_equal(env.task.reference_positions_m, settled)
                np.testing.assert_array_equal(env.sim.initial.particle_q.numpy(), settled)
                self.assertEqual(env.tick, 0)
                result = env.get_state()["evaluation"]
                self.assertEqual((result["success"], result["score"]), (False, 0.))
                # The unsettled lay is no longer the zero point.
                ix, flat = recorded(s["garment"])
                unsettled = start_positions(BY_ID[s["garment"], s["task"]], ix, flat, .78)
                self.assertGreater(np.abs(env.task.reference_positions_m - unsettled).max(), 1.e-3)

    def test_late_quality_step_delays_the_zero_point(self):
        """A step in quality at the minimum settling time must not become idle credit."""
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]
        step = round(module.SETTLE_MINIMUM_S / .5)

        def act(env, targets, *, ticks):
            env.act_calls += 1
            env.tick += ticks
            if env.act_calls == step:          # e.g. a sleeve dropping onto the body
                x = env.state.particle_q.numpy(); x[:, :2] *= .98
                env.state.particle_q.assign(x)
            return env.get_state()

        with patch.object(StubLaundryEnv, "act", autospec=True, side_effect=act):
            env = self.adapter(Config(garment="towel"), task="half_fold", contract=CONTRACT)
        drift = env.start_report["settling_quality_drift"]
        self.assertGreater(abs(drift[step - 1]), module.SETTLE_QUALITY_TOLERANCE)
        self.assertEqual(env.act_calls, step + module.SETTLE_STEADY_BLOCKS)
        self.assertEqual(env.start_report["settling_seconds"], .5 * (step + module.SETTLE_STEADY_BLOCKS))
        np.testing.assert_array_equal(env.task.reference_positions_m, env.state.particle_q.numpy())

    def test_slip_without_quality_change_delays_the_zero_point(self):
        """A garment still slipping (here rigidly, 8 mm) is not at rest even if its quality holds."""
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]
        step = round(module.SETTLE_MINIMUM_S / .5)

        def act(env, targets, *, ticks):
            env.act_calls += 1
            env.tick += ticks
            if env.act_calls == step:
                x = env.state.particle_q.numpy(); x[:, 0] += .008
                env.state.particle_q.assign(x)
            return env.get_state()

        with patch.object(StubLaundryEnv, "act", autospec=True, side_effect=act):
            env = self.adapter(Config(garment="towel"), task="half_fold", contract=CONTRACT)
        report = env.start_report
        self.assertLessEqual(max(abs(d) for d in report["settling_quality_drift"]), module.SETTLE_QUALITY_TOLERANCE)
        self.assertAlmostEqual(report["settling_block_motion_m"][step - 1], .008, places=5)
        self.assertEqual(env.act_calls, step + module.SETTLE_STEADY_BLOCKS)

    def test_catalogue_start_that_never_steadies_fails_closed(self):
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]

        def act(env, targets, *, ticks):
            env.act_calls += 1
            env.tick += ticks
            x = env.state.particle_q.numpy(); x[:, :2] *= .98      # keeps creeping
            env.state.particle_q.assign(x)
            return env.get_state()

        with patch.object(StubLaundryEnv, "act", autospec=True, side_effect=act):
            with self.assertRaisesRegex(RuntimeError, f"within {module.SETTLE_MAXIMUM_S:g} seconds"):
                self.adapter(Config(garment="towel"), task="half_fold", contract=CONTRACT)

    def test_v2_adapter_serves_only_its_contract(self):
        from laundry.newton15.config import Config
        for garment, task, contract in (("towel", None, None), ("towel", None, LEGACY), ("shirt", None, None),
                                        ("towel", "spread_flat", V1_CONTRACT), ("shirt", None, SHIRT_CONTRACT)):
            with self.subTest(garment=garment, contract=contract), self.assertRaises(ValueError):
                self.adapter(Config(garment=garment), task=task, contract=contract)

    def test_bad_authored_start_aborts_instead_of_becoming_an_easy_episode(self):
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_v2_env"]
        with patch.object(module, "start_positions", side_effect=lambda entry, index, x, top: x):
            with patch.object(StubLaundryEnv, "close", autospec=True) as close:
                with self.assertRaisesRegex(ValueError, "already meets"):
                    self.adapter(Config(garment="towel"), task="spread_flat", contract=CONTRACT)
                close.assert_called_once()

    def test_never_settling_start_has_a_bounded_failure(self):
        from laundry.newton15.config import Config
        with patch.object(StubLaundryEnv, "evaluate", return_value={
            "gates": {"state:at_rest": False, "goal:flat": False}, "score": 0.}):
            with self.assertRaisesRegex(RuntimeError, "within 30 seconds"):
                self.adapter(Config(garment="towel"), task="spread_flat", contract=CONTRACT)



if __name__ == "__main__":
    unittest.main()

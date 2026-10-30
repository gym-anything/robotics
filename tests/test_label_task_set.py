"""Eight-task catalogue and real reward arithmetic on synthetic state, with no runtime.

Only NumPy and the standard library are required. No world, solver, backend,
artwork or renderer is imported. The score fixture stubs distance, placement,
registration, rest and contact readers; it exercises the production evaluate,
progress, node_credit, take_share, integrity and repair-region/window checks.
These snapshots test arithmetic and dispatch, not physical solvability or decode.
"""
import json
from types import SimpleNamespace as NS
import unittest

import numpy as np

from label_application import layouts, materials, variants
from label_application.config import Config, LADDER_TASKS, REPAIR_TASKS, START_CREDIT_TASKS, TASKS
from label_application.task import Evaluator, FAIL_SCORE_SCALE, REWARD_CONTRACTS, TABLE
from tests.packages import PACKAGES


ROOT = PACKAGES
EXPECTED = (
    ("parcel", "smooth-corner", "entry", "progress-v1"),
    ("parcel", "press-down", "entry", "progress-v2"),
    ("parcel", "finish-securing", "medium", "progress-v1"),
    ("parcel", "carrier-top", "medium", "progress-v2"),
    ("part", "part-id-hdpe", "medium", "progress-v2"),
    ("parcel", "gs1-case-side", "hard", "progress-v2"),
    ("part", "part-id-match", "hard", "progress-v2"),
    ("specimen", "tube-label", "hardest", "progress-v3"),
)


class CatalogueTest(unittest.TestCase):
    def test_exact_selections_resolve_to_existing_ids_and_contracts(self):
        raw = json.loads((ROOT / "label_application/benchmark.json").read_text())
        entries = raw["task_set"]["tasks"]
        expected = [dict(preset=p, task=t, seed=0) for p, t, _, _ in EXPECTED]
        self.assertEqual([e["selection"] for e in entries], expected)
        self.assertEqual(list(variants.TASK_SET), expected)
        self.assertEqual(variants.contract()["task_set"], expected)
        self.assertEqual(len({e["task_id"] for e in entries}), 8)
        for e, (preset, task, difficulty, contract) in zip(entries, EXPECTED):
            with self.subTest(task=task):
                resolved = variants.task_resolve(e["selection"])
                self.assertEqual(resolved["selection"], e["selection"])
                self.assertEqual(resolved["backend_options"], e["selection"])
                self.assertEqual(resolved["task_id"], e["task_id"])
                self.assertEqual(e["task_id"], f"label-application/v1/{preset}/{task}/seed-0")
                self.assertEqual(e["difficulty"], difficulty)
                self.assertEqual(e["reward_contract"], contract)
                self.assertEqual(REWARD_CONTRACTS[task], contract)
                self.assertEqual(Config(**resolved["backend_options"]).task, task)

    def test_defaults_and_unselected_jobs_remain_available(self):
        raw = json.loads((ROOT / "label_application/benchmark.json").read_text())
        self.assertEqual(raw["count"], 10)
        self.assertEqual(sum(map(len, TASKS.values())), 10)
        self.assertEqual(raw["default_task_id"], variants.task_resolve({})["task_id"])
        for preset, task in (("parcel", "carrier-top"), ("specimen", "tube-wrap"), ("part", "part-id-hdpe")):
            self.assertEqual(variants.resolve(preset)["task"], task)
        for preset, task in (("specimen", "tube-wrap"), ("parcel", "cover-old-label")):
            self.assertEqual(variants.resolve(preset, task)["task"], task)
        self.assertEqual(raw["task_set"]["swaps"], [])

    def test_seed_variations_and_invalid_selections(self):
        for preset, task, _, _ in EXPECTED:
            for seed in (0, 7, 9999):
                r = variants.task_resolve(dict(preset=preset, task=task, seed=seed))
                self.assertEqual(r["task_id"], f"label-application/v1/{preset}/{task}/seed-{seed}")
        for seed in (-1, 10000, True, 1.5, "0"):
            with self.assertRaises(ValueError):
                variants.resolve(seed=seed)
        for preset, task in (("parcel", "tube-label"), ("part", "part-id-steel"), ("missing", None)):
            with self.assertRaises(ValueError):
                variants.resolve(preset, task)

    def test_contract_returns_defensive_selection_copies(self):
        c = variants.contract()
        c["task_set"][0]["task"] = "changed"
        self.assertEqual(variants.contract()["task_set"][0]["task"], "smooth-corner")


def score_fixture(task):
    """A planar test sheet with explicit state readers, never a simulation scene."""
    u, v = np.meshgrid(np.linspace(0., .1524, 31), np.linspace(0., .1016, 21))
    X = np.c_[u.ravel(), v.ravel()]
    tri = []
    for j in range(20):
        for i in range(30):
            a = j * 31 + i
            tri.extend(((a, a + 1, a + 31), (a + 1, a + 32, a + 31)))
    tris = np.array(tri)
    area = np.zeros(len(X))
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    twice = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    for k in range(3):
        np.add.at(area, tris[:, k], abs(twice) / 6.)
    surface = NS(p_rough=40000.)
    target = "plate_1" if task == "part-id-match" else "target"
    cols = [NS(name=n, role=r, surface=surface, surface_list=lambda: [surface],
               p=np.array([0., 0., TABLE + .2]), size=(.01, .01, .01))
            for n, r in ((target, "item"), ("liner", "liner"), ("left_pad", "pad"),
                         ("right_pad", "pad"), ("wrong_item", "item"))]
    n = len(X)
    lab = NS(X=X, tris=tris, vertex_area=area, n=n, length=.1524, width=.1016, h=.003,
             x=np.c_[X, np.full(n, TABLE + .2)], v=np.zeros((n, 3)),
             stock=NS(thickness=.0001), colliders=cols, bond=np.ones(n, int),
             bond_surface=np.zeros(n, int), phi=np.zeros(n), vertex_alive=np.ones(n, bool),
             boundary=(u.ravel() == 0.) | (u.ravel() == .1524) | (v.ravel() == 0.) | (v.ravel() == .1016),
             hinge=np.array([[0, 1]]), theta_rest=np.zeros(1), hinge_dual=np.ones(1), torn=False)
    geometry = {}
    if task in REPAIR_TASKS:
        geometry["laid_label"] = layouts.partly_applied(task, 0)
    reader = NS(gap=np.full(n, .01), zone=np.ones(n, bool), registration=1., placement=True,
                off_bench=False, hands_off=True, resting=True)
    w = NS(label=lab, config=NS(task=task, label_dt=.002), set_peak=np.zeros(1),
           scene=NS(task_geometry=geometry),
           body_pose=lambda name: (np.array([0., 0., TABLE + .1]), np.eye(3)))
    ev = Evaluator.__new__(Evaluator)
    ev.world, ev.target, ev.initial_items = w, target, {}
    ev.boundary_band = ev._band(.003)
    ev.in_barcode = np.ones(n, bool)
    ev.start_presented = np.ones(n, bool) if task not in START_CREDIT_TASKS else np.zeros(n, bool)
    ev._target_distance = lambda x: (reader.gap + lab.stock.thickness / 2., np.tile([0., 0., 1.], (n, 1)))
    ev._trapped_air = lambda: dict(largest_m2=0., in_barcode_m2=0., largest_enclosed_m2=0.)
    ev.resting = lambda name: dict(ok=reader.resting)
    ev.hands_off = lambda name: dict(ok=reader.hands_off)
    ev.zone = lambda on_target: reader.zone
    ev.registration = lambda placed, on_target: dict(factor=reader.registration)
    ev._item_off_bench = lambda: reader.off_bench
    ev.stage = lambda *args: {}  # diagnostics only
    for name in ("carrier_top", "press_down", "repair_rung", "gs1_case_side", "part_id_hdpe", "part_id_match", "tube_label"):
        setattr(ev, "_place_" + name, lambda on_target: dict(checks={"fixture_placement": reader.placement}, metrics={}))
    reference = 1. - np.exp(-(100000. + materials.P_TACK) / surface.p_rough)
    if task in START_CREDIT_TASKS:
        lab.bond[:] = 0
        reader.gap[:] = 0.
        lab.phi[:] = .03 * reference
        if task in REPAIR_TASKS:
            lifted = np.any([layouts.flap_mask(X, f)[0] for f in geometry["laid_label"]["regions"]], axis=0)
            lab.phi[:] = reference
            lab.phi[lifted] = 0.
            lab.bond[lifted] = -1
            reader.gap[lifted] = .01
        geometry["start_credit"] = ev.node_credit().copy()
    return ev, lab, reader, reference


def apply_sheet(lab, reader, reference, wet_out=.95):
    lab.bond[:] = 0
    lab.phi[:] = wet_out * reference
    reader.gap[:] = 0.


class ScoreTest(unittest.TestCase):
    def test_untouched_snapshots_fail_with_zero_progress_and_score(self):
        for _, task, _, contract in EXPECTED:
            with self.subTest(task=task):
                ev, _, _, _ = score_fixture(task)
                result = ev.evaluate(optical={"match": True})
                self.assertEqual((result["success"], result["progress"], result["score"]), (False, 0., 0.))
                self.assertEqual(result["reward"]["contract"], contract)

    def test_finished_snapshots_score_one_and_optical_failure_stays_below_one(self):
        self.assertEqual(FAIL_SCORE_SCALE, .9)
        for _, task, _, contract in EXPECTED:
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                apply_sheet(lab, reader, ref)
                result = ev.evaluate(optical={"match": True})
                self.assertTrue(result["success"], result["checks"])
                self.assertEqual(result["score"], 1.)
                self.assertEqual(result["reward"]["contract"], contract)
                result = ev.evaluate(optical={"match": False})
                self.assertFalse(result["success"])
                self.assertAlmostEqual(result["score"], .9)
                self.assertTrue(result["optical_evaluated"])

    def test_progress_varies_without_any_change_to_passed_checks(self):
        for _, task, _, _ in EXPECTED:
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                results = []
                for wet_out in (.6, .601, .602):
                    apply_sheet(lab, reader, ref, wet_out)
                    if task in START_CREDIT_TASKS:
                        c0 = ev.world.scene.task_geometry["start_credit"]
                        credit = c0 + wet_out * (1. - c0)
                        lab.phi[:] = .9 * (2. * credit - 1.) * ref
                    results.append(ev.evaluate(optical={"match": True}))
                self.assertEqual(results[0]["checks"], results[1]["checks"])
                self.assertEqual(results[1]["checks"], results[2]["checks"])
                self.assertLess(results[0]["score"], results[1]["score"])
                self.assertLess(results[1]["score"], results[2]["score"])
                for result in results:
                    self.assertAlmostEqual(result["score"], .9 * result["progress"])

    def test_damage_zeroes_all_task_progress(self):
        for _, task, _, _ in EXPECTED:
            for defect in ("torn", "creased", "off_bench"):
                with self.subTest(task=task, defect=defect):
                    ev, lab, reader, ref = score_fixture(task)
                    apply_sheet(lab, reader, ref)
                    if defect == "torn":
                        lab.torn = True
                    elif defect == "creased":
                        lab.theta_rest[:] = 120.
                    else:
                        reader.off_bench, reader.resting = True, False
                    r = ev.evaluate(optical={"match": True})
                    self.assertEqual((r["success"], r["score"], r["progress"]), (False, 0., 0.))

    def test_ladder_uses_episode_crease_others_use_present_crease(self):
        for _, task, _, _ in EXPECTED:
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                apply_sheet(lab, reader, ref)
                ev.world.set_peak[:] = 120.
                r = ev.evaluate(optical={"match": True})
                self.assertEqual(r["score"], 0. if task in LADDER_TASKS else 1.)

    def test_take_credit_is_hand_independent_and_wrong_item_earns_nothing(self):
        for _, task, _, _ in EXPECTED:
            if task in START_CREDIT_TASKS:
                continue
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                for hand in (2, 3):
                    lab.bond[:] = hand
                    r = ev.evaluate(optical={"match": True})
                    self.assertAlmostEqual(r["progress"], .1)
                    self.assertAlmostEqual(r["score"], .09)
                lab.bond[:] = 4  # stuck to a different item
                self.assertEqual(ev.evaluate(optical={"match": True})["score"], 0.)
                apply_sheet(lab, reader, ref)
                reader.zone[:] = False
                reader.placement = False
                self.assertEqual(ev.evaluate(optical={"match": True})["score"], 0.)

    def test_repair_nets_lost_credit_against_work_done(self):
        for task in START_CREDIT_TASKS:
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                c0 = ev.world.scene.task_geometry["start_credit"]
                # Half of the missing credit, at every node: exactly half the work.
                credit = c0 + .5 * (1. - c0)
                apply_sheet(lab, reader, ref)
                lab.phi[:] = .9 * (2. * credit - 1.) * ref
                r = ev.evaluate(optical={"match": False})
                self.assertAlmostEqual(r["progress"], .5)
                self.assertAlmostEqual(r["score"], .45)
                lab.bond[lab.X[:, 0] < .02] = -1
                reader.gap[lab.X[:, 0] < .02] = .01
                self.assertLess(ev.evaluate(optical={"match": False})["score"], .45)

    def test_unpressed_repair_regions_cannot_hide_in_label_average(self):
        ev, lab, reader, ref = score_fixture("smooth-corner")
        lifted = ev.world.scene.task_geometry["start_credit"] < 1.
        apply_sheet(lab, reader, ref, wet_out=1.)
        lab.phi[lifted] = .03 * ref
        r = ev.evaluate(optical={"match": True})
        self.assertTrue(r["checks"]["wet_out>=90%_of_standard_application"])
        self.assertFalse(r["checks"]["each_lifted_area_wet_out>=90%_of_standard"])
        self.assertFalse(r["success"])

    def test_final_state_does_not_depend_on_press_order(self):
        for _, task, _, _ in EXPECTED:
            with self.subTest(task=task):
                results = []
                for reverse in (False, True):
                    ev, lab, reader, ref = score_fixture(task)
                    indices = np.arange(lab.n)[::-1] if reverse else np.arange(lab.n)
                    for group in np.array_split(indices, 3):
                        lab.bond[group] = 0
                        lab.phi[group] = .8 * ref
                        reader.gap[group] = 0.
                        ev.evaluate(optical={"match": False})
                    results.append(ev.evaluate(optical={"match": True}))
                self.assertEqual(results[0], results[1])

    def test_registration_scales_take_and_application_together(self):
        for task in ("gs1-case-side", "part-id-hdpe", "part-id-match", "tube-label"):
            with self.subTest(task=task):
                ev, lab, reader, ref = score_fixture(task)
                apply_sheet(lab, reader, ref)
                reader.placement = False
                reader.registration = .5
                self.assertAlmostEqual(ev.evaluate(optical={"match": True})["score"], .45)

    def test_tube_lost_label_rule_keeps_bonded_sheet_recoverable(self):
        ev, lab, reader, ref = score_fixture("tube-label")
        lab.bond[:] = -1
        lab.x[:, 2] = TABLE - .5
        self.assertEqual(ev.evaluate(optical={"match": True})["score"], 0.)
        lab.bond[0] = 1  # a sheet hanging off the printer is still held
        self.assertGreater(ev.evaluate(optical={"match": True})["score"], 0.)


if __name__ == "__main__":
    unittest.main()

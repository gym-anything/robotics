"""Constructed-state and stubbed adapter tests, with no simulator runtime."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from laundry.garment_task import GARMENT_TASK_SPECS, GarmentTaskEvaluator, TabletopFrame
from laundry.task_set import (BY_NAME, CONTRACT, LEGACY, SHIRT_CONTRACT, TASKS, TASK_SET,
                              TASK_SET_ORDER, make_evaluator, resolve, select, task_spec)
from laundry.task_set_presets import accordion_curve, reach_report, start_positions
from laundry.task_set_goals import three_ply_fraction
from tests.environments.laundry.task_set_fixtures import recorded, half_fold, thirds_fold
from tests.packages import PACKAGES


def evaluator(name, *, start=None):
    entry = BY_NAME[name]
    ix, flat = recorded(entry.garment)
    x = start_positions(entry, ix, flat, .78) if start is None else start.copy()
    version = SHIRT_CONTRACT if entry.garment == "shirt" else CONTRACT
    return make_evaluator(entry, version, ix, x), x


def judge(ev, x, **changes):
    kw = dict(maximum_structural_strain=0., nonlocal_intersection_count=0,
              runtime_pose_writes=0, external_particle_force_applications=0,
              released=True, fixtures_clear=True, tabletop=TabletopFrame(.78, .80, .70),
              recent_positions_m=np.stack([x, x]), recent_span_s=.5, principal_stretch_range=(1., 1.))
    kw.update(changes)
    return ev.evaluate(x, np.zeros_like(x), **kw)


class CatalogueTests(unittest.TestCase):
    def test_eight_exact_selections_and_original_order(self):
        path = PACKAGES / "laundry/newton15/benchmark.json"
        d = json.loads(path.read_text())
        self.assertEqual(d["task_set"]["selections"], list(TASK_SET))
        self.assertEqual(d["count"], 8)
        self.assertEqual([t.name for t in TASKS[:5]],
                         ["sleeve_tuck_fold", "sleeve_fold", "leg_align_fold", "half_fold", "tie_untangle"])
        self.assertEqual(len({resolve(s)["task_id"] for s in TASK_SET}), 8)

    def test_original_bare_selections_are_unchanged(self):
        from laundry.newton15.benchmark import resolve as old
        for s in ({}, *({"garment": g} for g in ("tee", "shirt", "jeans", "towel", "asymmetric"))):
            self.assertEqual(resolve(s), old(s))
        self.assertEqual(select()[1], LEGACY)


    def test_unknown_and_incompatible_selectors(self):
        for args in (("towel", "bad", None), ("tee", "spread_flat", CONTRACT),
                     ("shirt", None, CONTRACT), ("towel", "spread_flat", LEGACY),
                     ("towel", None, "unknown"), ("towel", 1, None), (None, None, None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                select(*args)

    def test_original_shirt_spec_is_the_same_object(self):
        entry, version = select("shirt")
        old = GARMENT_TASK_SPECS[("woven_button_down_shirt", "sleeve_fold")]
        self.assertIs(task_spec(entry, version), old)
        ix, x = recorded("shirt")
        new = make_evaluator(entry, version, ix, x)
        self.assertIs(type(new), GarmentTaskEvaluator)
        prior = GarmentTaskEvaluator(old, ix, reference_positions_m=x)
        self.assertEqual(judge(new, x), judge(prior, x))


class StateTests(unittest.TestCase):
    def test_all_eight_untouched_states_score_zero_even_with_rest_window(self):
        for name in TASK_SET_ORDER:
            with self.subTest(task=name):
                ev, x = evaluator(name)
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

    def test_presets_do_not_mutate_input_or_material_geometry(self):
        for name in ("spread_flat", "tie_untangle"):
            entry = BY_NAME[name]; ix, flat = recorded(entry.garment)
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
        entry, version = select(config.garment)
        self.task = make_evaluator(entry, version, ix, self.state.particle_q.numpy())
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
        self.adapter = importlib.import_module("laundry.task_set_env").CatalogueLaundryEnv
        self.addCleanup(self.modules.stop)

    def test_every_selection_reaches_adapter_and_reports_contract(self):
        from laundry.newton15.config import Config
        for s in TASK_SET:
            env = self.adapter(Config(garment=s["garment"]), task=s.get("task"), contract=s.get("contract"))
            result = env.get_state()["evaluation"]
            self.assertEqual(result["contract"], SHIRT_CONTRACT if s["garment"] == "shirt" else CONTRACT)
            self.assertEqual(result["task_id"], resolve(s)["task_id"])
            self.assertEqual(env.task.spec.label + ". Use the robot arms; leave the garment released and resting on the table.",
                             resolve(s)["description"])
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
                for i, selection in enumerate(TASK_SET):
                    resolved = resolve(selection)
                    backend = NewtonLaundryBackend(**resolved["backend_options"], cameras=(), recording=False,
                                                   output_dir=Path(folder) / str(i))
                    self.assertIsInstance(backend.env, self.adapter)
                    self.assertEqual(backend.policy_access().instruction(), resolved["description"])
                    self.assertEqual(backend.get_state()["evaluation"]["task_id"], resolved["task_id"])
                    state, info = backend.reset()
                    self.assertEqual(state["evaluation"]["score"], 0.)
                    self.assertEqual(info, {"continuity": "reset"})
                    backend.close()

    def test_authored_start_is_installed_once_and_reset_reuses_accepted_snapshot(self):
        from laundry.newton15.config import Config
        for name in ("spread_flat", "tie_untangle"):
            entry = BY_NAME[name]
            env = self.adapter(Config(garment=entry.garment), task=name, contract=CONTRACT)
            self.assertEqual(env.author_calls, 1)
            self.assertEqual(env.act_calls, 1)
            self.assertEqual(env.tick, 0)
            self.assertEqual(env.sim.step_count, 0)
            before = env.state.particle_q.numpy()
            changed = before.copy(); changed[:, 0] += .03
            env.state.particle_q.assign(changed)
            result = env.reset()
            np.testing.assert_array_equal(env.state.particle_q.numpy(), before)
            self.assertEqual(result["evaluation"]["score"], 0.)
            self.assertEqual(env.author_calls, 1)
            self.assertEqual(env.act_calls, 1)

    def test_legacy_and_shirt_do_not_author_or_settle(self):
        from laundry.newton15.config import Config
        for garment in ("towel", "tee", "shirt", "asymmetric", "jeans"):
            env = self.adapter(Config(garment=garment))
            self.assertEqual(env.author_calls, 0)
            self.assertEqual(env.act_calls, 0)
            self.assertIs(type(env.task), GarmentTaskEvaluator)

    def test_bad_authored_start_aborts_instead_of_becoming_an_easy_episode(self):
        from laundry.newton15.config import Config
        module = sys.modules["laundry.task_set_env"]
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

"""surface-rinsing-tasks-v2 task-only tests. Never import Newton, Warp, MuJoCo, a server or a renderer.

The render-checked set: the released set's scenes and predicates, with pre-rinse-plates' temperature sentence kept and
gated (spray_temperature, temperature-graded progress). The fixtures are the released task-set test's
(tests/test_surface_rinsing_task_set.py); these are verifier and dispatch checks, not physics/solvability evidence.
"""
import copy
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from gym_anything_robotics import tasks
from surface_rinsing import task_set_v2 as V2
from surface_rinsing.task_set import catalogue as K
from tests.packages import PACKAGES, REPO
from tests.test_surface_rinsing_task_set import complete, evaluator, module, reading, stub_modules, verifiers

ROOT = PACKAGES
ENV = "surface-rinsing"
PLATES = V2.SPRAY_TASK


@lru_cache(maxsize=1)
def v2_verifiers():
    verifiers()
    return importlib.import_module("surface_rinsing.task_set_v2_evaluator")


def reading_v2(task):
    m = reading(task)
    if task == PLATES:
        m["task_set"]["spray"] = dict(volume_L=0., mean_C=None)
    return m


def complete_v2(task, m):
    complete(task, m)
    if task == PLATES:
        m["task_set"]["spray"] = dict(volume_L=20., mean_C=49.)


def evaluator_v2(task, m):
    """The released fixture's stand-ins on the v2 evaluator class (a subclass adding only the spray tracker)."""
    ev = evaluator(task, m)
    ev.__class__ = v2_verifiers().Evaluator
    ev.spray, ev.frozen = dict(volume_L=0., heat_L_C=0.), False
    return ev


class CatalogueV2Tests(unittest.TestCase):
    def test_benchmark_lists_the_eight_exact_selections(self):
        b = json.loads((ROOT/"surface_rinsing/benchmark.json").read_text())
        self.assertEqual(b["task_set"]["contract"], V2.CONTRACT)
        self.assertEqual(b["task_set"]["selections"], [dict(s) for s in V2.SELECTIONS])
        self.assertIn(V2.CONTRACT, b["selector_choices"]["contract"])
        self.assertEqual([s["task"] for s in V2.SELECTIONS], list(K.EIGHT))
        self.assertEqual([r["selection"] for r in b["task_set_v1"]["tasks"]], K.selections())
        test = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        self.assertEqual([t["selection"] for t in test if t["environment_id"] == ENV], [V2.SELECTIONS[-1]])
        self.assertNotIn("contract", V2.SELECTIONS[-1])

    def test_v2_selections_resolve_with_contract_in_id_and_options(self):
        ids = set()
        for selection in V2.SELECTIONS:
            task = selection["task"]
            with self.subTest(task=task):
                spec = tasks.resolve(ENV, dict(selection))
                v1 = tasks.resolve(ENV, dict(selection, contract=K.CONTRACT) if task in K.REVISED else dict(selection))
                if task in K.REVISED:
                    self.assertEqual(spec.task_id, f"surface-rinsing/v1/{selection['preset']}/{task}/{V2.CONTRACT}/seed-0")
                    self.assertEqual(spec.selection, selection)
                    self.assertEqual(spec.launch.backend_options, selection)
                    self.assertEqual(spec.split, v1.split)
                    expected = K.LEGACY_TASKS[task]["instruction"] if task == PLATES else K.TASKS[task]["instruction"]
                    self.assertEqual(spec.description, expected)
                else:
                    self.assertEqual(spec.to_dict(), tasks.resolve(ENV, dict(preset="pot-sink", task=task)).to_dict())
                    self.assertEqual(spec.task_id, "surface-rinsing/v1/pot-sink/scrub-rinse-pan/seed-0")
                self.assertEqual(spec.evaluator, v1.evaluator)
                self.assertEqual(tasks.resolve(ENV, spec.selection).to_dict(), spec.to_dict())
                ids.add(spec.task_id)
        self.assertEqual(len(ids), 8)
        self.assertIn("49 C (110-120 F)", tasks.resolve(ENV, dict(task=PLATES, contract=V2.CONTRACT)).description)
        self.assertNotIn("49 C", tasks.resolve(ENV, dict(task=PLATES, contract=K.CONTRACT)).description)

    def test_released_resolution_is_unchanged(self):
        b = json.loads((ROOT/"surface_rinsing/benchmark.json").read_text())
        for row in b["task_set_v1"]["tasks"]:
            self.assertEqual(K.task_resolve(row["selection"])["task_id"], row["task_id"])
        self.assertEqual(K.task_resolve({})["task_id"], b["default_task_id"])
        for task in K.LEGACY_TASKS:
            self.assertEqual(V2.task_resolve(dict(task=task, seed=5)), K.task_resolve(dict(task=task, seed=5)))
        self.assertEqual(V2.resolve(task=K.NEW_TASK), K.resolve(task=K.NEW_TASK))      # no contract: v1, as released
        self.assertEqual(V2.contract()["contracts"], [*K.contract()["contracts"], V2.CONTRACT])

    def test_v2_refusals_and_defaults(self):
        for args in (dict(task="scrub-rinse-pan"), dict(task="rinse-dishware"), dict(task="no"), dict(preset="no"),
                     dict(task="flush-floor", preset="pot-sink"), dict(seed=-1), dict(seed=True), dict(seed=10000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                V2.resolve(**args, contract=V2.CONTRACT)
            with self.assertRaises(ValueError):
                tasks.resolve(ENV, dict(args, contract=V2.CONTRACT))
        for args in (dict(task="rinse-sponge", contract="surface-rinsing-tasks8"), dict(contract="v2")):
            with self.assertRaises(ValueError):
                tasks.resolve(ENV, args)
        self.assertEqual(V2.resolve(contract=V2.CONTRACT),
                         dict(preset="pot-sink", task="rinse-glassware", seed=0, contract=V2.CONTRACT))
        self.assertEqual(V2.resolve(preset="pre-rinse", seed=7, contract=V2.CONTRACT)["task"], PLATES)

    def test_v2_setups_are_the_revised_scenes(self):
        verifiers()
        for seed in (0, 3, 99):
            for task in sorted(K.REVISED):
                got = V2.setup(task, seed, V2.CONTRACT)
                old = K.setup(task, seed, K.CONTRACT)
                self.assertEqual(got.pop("task_contract"), V2.CONTRACT)
                self.assertEqual(got.pop("instruction"), V2.TASKS[task]["instruction"])
                old.pop("task_contract"), old.pop("instruction")
                self.assertEqual(got, old)
        self.assertEqual(V2.setup("scrub-rinse-pan", 0), K.setup("scrub-rinse-pan", 0))


class ScoringV2Tests(unittest.TestCase):
    def test_plate_instruction_kept_and_its_spray_temperature_gated(self):
        # the spec keeps "Mix the water to about 49 C (110-120 F)" with a spray_temperature gate and a grade
        self.assertEqual(V2.TASKS[PLATES]["instruction"], K.LEGACY_TASKS[PLATES]["instruction"])
        self.assertIn("49 C (110-120 F)", V2.instruction(V2.resolve(task=PLATES, contract=V2.CONTRACT)))
        self.assertEqual(V2.SPRAY_BAND_C, (40.5, 51.5))
        for mean, grade in ((None, 0.), (40.5, 1.), (49., 1.), (51.5, 1.), (55., .65), (60., .15), (30.5, 0.), (65., 0.)):
            self.assertAlmostEqual(V2.temperature_grade(mean), grade)
        m = reading_v2(PLATES)
        ev = evaluator_v2(PLATES, m)
        complete_v2(PLATES, m)
        for mean, passed in ((49., True), (40.5, True), (51.5, True), (40.4, False), (60., False), (None, False)):
            m["task_set"]["spray"]["mean_C"] = mean
            c = ev.criteria(m)
            self.assertEqual(c["spray_temperature"]["passed"], passed, mean)
            value, rw = ev._score(c, m)
            self.assertEqual(rw["terms"]["temperature_grade"], V2.temperature_grade(mean))
            self.assertEqual(rw["contract"], V2.CONTRACT)
            # all other gates pass: success at a band temperature, else 0.9 x progress x grade
            self.assertAlmostEqual(value, 1. if passed else .9*V2.temperature_grade(mean))

    def test_spray_tracker_is_flow_weighted_every_tick_frozen_in_settle_and_checkpointed(self):
        E = v2_verifiers()
        ev = E.Evaluator.__new__(E.Evaluator)
        ev.task, ev.glasses, ev.frozen = PLATES, {}, False
        ev.spray = dict(volume_L=0., heat_L_C=0.)
        ev.spray_outlet = outlet = types.SimpleNamespace(Q=0., T=15.)
        ev.env = types.SimpleNamespace(water=object())
        world = types.SimpleNamespace(tick=1)
        dt = E.P.DT
        for q, t, ticks in ((0., 60., 7), (1e-4, 60., 3), (3e-4, 40., 1)):
            outlet.Q, outlet.T = q, t
            for _ in range(ticks):
                ev._track(world)
                world.tick += 1
        self.assertAlmostEqual(ev.spray["volume_L"], (3*1e-4 + 3e-4)*dt*1e3)
        self.assertAlmostEqual(ev.spray["heat_L_C"]/ev.spray["volume_L"], 50.)
        saved = copy.deepcopy(ev.spray)
        ev.frozen = True
        ev._track(world)
        self.assertEqual(ev.spray, saved)
        # the tracker is kept in checkpoints (evaluator snapshot) and restored with them
        ev.pan, ev.sanitize, ev.surf_into_sanitizer, ev._in_san = None, {}, {}, set()
        ev.impacts = types.SimpleNamespace(max={}, prev=set())
        ev.start = reading_v2(PLATES)
        state = copy.deepcopy(ev.snapshot())
        self.assertEqual(state["task_set"]["spray"], saved)
        ev.spray = dict(volume_L=9., heat_L_C=9.)
        ev.restore(state)
        self.assertEqual(ev.spray, saved)

    def test_all_revised_untouched_states_are_zero_and_not_success(self):
        for task in K.REVISED:
            with self.subTest(task=task):
                m = reading_v2(task)
                ev = evaluator_v2(task, m)
                c = ev.criteria(m)
                self.assertFalse(all(v["passed"] for v in c.values()))
                value, rw = ev._score(c, m)
                self.assertEqual(value, 0.)
                self.assertEqual(rw["contract"], V2.CONTRACT)
                self.assertEqual("spray_temperature" in c, task == PLATES)

    def test_every_success_scores_one_and_failure_never_one(self):
        for task in K.REVISED:
            with self.subTest(task=task):
                m = reading_v2(task)
                ev = evaluator_v2(task, m)
                complete_v2(task, m)
                c = ev.criteria(m)
                self.assertTrue(all(v["passed"] for v in c.values()), c)
                self.assertEqual(ev._score(c, m)[0], 1.)
                m["holding"] = True
                self.assertLessEqual(ev._score(ev.criteria(m), m)[0], .9)

    def test_each_primary_quantity_has_interior_progress(self):
        for task in K.REVISED:
            m = reading_v2(task)
            ev = evaluator_v2(task, m)
            for it in m["items"].values():
                it["surfactant_ug"] /= 2
                it["soil_gauge_max_ug_cm2"] /= 2
            m["sponge"]["c_g_L"] /= 2
            m["pools"]["wash"]["volume_L"] = 35.
            if task == "flush-floor":
                m["task_set"]["floor_gauges"] = [50., 50., 0.]
            if task == PLATES:
                m["task_set"]["spray"] = dict(volume_L=5., mean_C=45.)
            value = ev._score(ev.criteria(m), m)[0]
            self.assertGreater(value, 0., task)
            self.assertLess(value, .9, task)

    def test_other_tasks_score_exactly_as_v1(self):
        for task in sorted(K.REVISED - {PLATES}):
            for finished in (False, True):
                with self.subTest(task=task, finished=finished):
                    m = reading_v2(task)
                    if finished:
                        complete_v2(task, m)
                        m["holding"] = True
                    old, new = evaluator(task, copy.deepcopy(m)), evaluator_v2(task, copy.deepcopy(m))
                    c_old, c_new = old.criteria(copy.deepcopy(m)), new.criteria(copy.deepcopy(m))
                    self.assertEqual(c_new, c_old)
                    (v_old, rw_old), (v_new, rw_new) = old._score(c_old, m), new._score(c_new, m)
                    self.assertEqual(v_new, v_old)
                    self.assertEqual((rw_old.pop("contract"), rw_new.pop("contract")), (K.CONTRACT, V2.CONTRACT))
                    self.assertEqual(rw_new, rw_old)

    def test_unverifiable_states_withhold_score(self):
        for task in K.REVISED:
            for kind in ("fault", "water", "nan"):
                m = reading_v2(task)
                ev = evaluator_v2(task, m)
                complete_v2(task, m)
                if kind == "fault":
                    m["fault"] = "nonfinite physics"
                elif kind == "water":
                    m["ledger"]["relative"] = .001
                else:
                    m["contamination"]["floor_water_mL"] = float("nan")
                value, rw = ev._score(ev.criteria(m), m)
                self.assertIsNone(value, (task, kind))
                self.assertFalse(rw["verifiable"])

    def test_evaluate_reports_the_v2_contract_and_restores_the_settle(self):
        m = reading_v2(PLATES)
        ev = evaluator_v2(PLATES, m)
        ev.measure = lambda: copy.deepcopy(m)
        legacy = importlib.import_module("surface_rinsing.task_set.evaluator").LegacyEvaluator
        with patch.object(legacy, "evaluate", lambda self, s: dict(score=0.)):
            self.assertEqual(ev.evaluate()["contract"], V2.CONTRACT)
        self.assertFalse(ev.frozen)
        self.assertFalse(hasattr(ev, "_progress_input"))


class IntegrationV2Tests(unittest.TestCase):
    def test_adapter_builds_the_v2_backend_only_for_v2(self):
        # registry.py builds the released adapter for every launch; its __new__ picks the v2 class for v2 only. The real
        # adapter constructor runs until it reads the robot catalogue (stubbed to stop there), so no world is built.
        verifiers()
        from gym_anything_robotics.backends import newton_surface_rinsing_task_set as A
        import surface_rinsing.task_set_v2_backend as B2

        class Reached(Exception):
            pass

        class Catalog(dict):
            def __getitem__(self, key):
                raise Reached(key)

        cp = module("classroom_arrangement.params", REFLEX_PROFILES={"contact_x5": {}})
        stubs = {"classroom_arrangement": module("classroom_arrangement", params=cp), "classroom_arrangement.params": cp,
                 "classroom_arrangement.kinematics": module("classroom_arrangement.kinematics", ArmKinematics=None),
                 "surface_rinsing.env": module("surface_rinsing.env", RinsingEnv=object),
                 "robots": module("robots"), "robots.catalog": module("robots.catalog", ENVIRONMENTS=Catalog())}
        built = []

        def launch(**kwargs):
            built.clear()
            try:
                backend = A.NewtonSurfaceRinsingBackend(**kwargs)
            except Reached:
                return "released"
            self.assertIs(type(backend), B2.NewtonSurfaceRinsingV2Backend)
            self.assertEqual(built, [kwargs])
            return "v2"

        with stub_modules(stubs), patch.object(B2.NewtonSurfaceRinsingV2Backend, "__init__",
                                               lambda b, **k: built.append(k)):
            for selection in V2.SELECTIONS:
                self.assertEqual(launch(robot="mobile-fr3-duo", **selection),
                                 "v2" if selection.get("contract") == V2.CONTRACT else "released", selection)
            for selection in K.selections():
                self.assertEqual(launch(robot="mobile-fr3-duo", **selection), "released", selection)
            for task in K.TASKS:
                self.assertEqual(launch(task=task, seed=4), "released")
            self.assertEqual(launch(), "released")
            # the v2 contract on a task without it reaches the v2 backend, whose resolve refuses it
            self.assertEqual(launch(task="scrub-rinse-pan", contract=V2.CONTRACT), "v2")
        self.assertTrue(issubclass(B2.NewtonSurfaceRinsingV2Backend, A.NewtonSurfaceRinsingBackend))
        factory = (ROOT/"gym_anything_robotics/registry.py").read_text().split("def _newton_surface_rinsing_factory", 1)[1]
        factory = factory.split("\nregister(", 1)[0]
        self.assertIn("from .backends.newton_surface_rinsing_task_set import NewtonSurfaceRinsingBackend", factory)

    def test_v2_backend_construct_select_reset_rebuild_with_stub_world(self):
        verifiers()
        cp = module("classroom_arrangement.params", REFLEX_PROFILES={"contact_x5": {}})

        class FakeEnv:
            def __init__(self, config, **kwargs):
                self.config = config
                self.world = types.SimpleNamespace(arms=("left", "right"))
                self.lock = threading.RLock()
                self.generation = 0

            def get_state(self):
                return dict(task=self.config.task)

            def reset(self):
                return self.get_state()

        class FakeBackend:
            _notes = lambda self: {}
            _min_action_s = lambda self, *a: .02
            _release = lambda self: None

        @dataclass(init=False)
        class Caps:
            notes: dict

            def __init__(self, *a, notes, **kwargs):
                self.notes = notes

        legacy_name = "gym_anything_robotics.backends.newton_surface_rinsing"
        adapter_name = "gym_anything_robotics.backends.newton_surface_rinsing_task_set"
        legacy = module(legacy_name, NewtonSurfaceRinsingBackend=FakeBackend, _session_folder=lambda: None,
                        SESSION_MARKER="unused", POLICY_CAMERAS=(), BASE_CAMERAS=(), BASE_KEYS=(),
                        CameraSpec=lambda *a: None, Capabilities=Caps,
                        UnsupportedConfigurationError=ValueError, UnsupportedActionError=ValueError)
        stubs = {legacy_name: legacy,
                 "classroom_arrangement": module("classroom_arrangement", params=cp),
                 "classroom_arrangement.params": cp,
                 "classroom_arrangement.kinematics": module("classroom_arrangement.kinematics", ArmKinematics=lambda w: object()),
                 "surface_rinsing.env": module("surface_rinsing.env", RinsingEnv=object),
                 "robots.catalog": module("robots.catalog", ENVIRONMENTS={"surface-rinsing": {"aliases": ["mobile-fr3-duo"]}}),
                 "surface_rinsing.robot_setup": module("surface_rinsing.robot_setup", check_task=lambda *a: None),
                 "gym_anything_robotics.keyboard_control": module("gym_anything_robotics.keyboard_control", SUPPORTED_KEYS=())}

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            stubs[name] = mod
            return spec, mod

        # Fresh copies of the adapter and the v2 backend over the stand-ins, under their own names for the duration.
        a_spec, adapter = load(adapter_name, ROOT/"gym_anything_robotics/backends/newton_surface_rinsing_task_set.py")
        b_spec, v2b = load("surface_rinsing.task_set_v2_backend", ROOT/"surface_rinsing/task_set_v2_backend.py")
        with stub_modules(stubs):
            a_spec.loader.exec_module(adapter)
            b_spec.loader.exec_module(v2b)
            runtime = importlib.import_module("surface_rinsing.task_set_v2_runtime")
            with patch.object(runtime, "RinsingEnv", FakeEnv), tempfile.TemporaryDirectory() as tmp:
                for i, selection in enumerate(V2.SELECTIONS[:-1]):       # the pan is the released adapter's
                    b = adapter.NewtonSurfaceRinsingBackend(**selection, cameras=(), recording=False,
                                                            output_dir=Path(tmp)/str(i))
                    self.assertIs(type(b), v2b.NewtonSurfaceRinsingV2Backend)
                    self.assertEqual(b._variation(), selection)
                    self.assertEqual(b.config.task_contract, V2.CONTRACT)
                    self.assertIsInstance(b.config, runtime.Config)
                    request = b.select_variation({})
                    self.assertEqual(request["task"], V2.TASKS[selection["task"]]["instruction"])
                    self.assertEqual(request["reset_request"]["options"]["contract"], V2.CONTRACT)
                    self.assertEqual(b.variation_contract()["current"], selection)
                    b.reset(seed="3")
                    self.assertEqual((b.config.seed, b.config.task_contract, b.env.generation), (3, V2.CONTRACT, 1))
                    current = b._variation()
                    with self.assertRaises(ValueError):
                        b.reset(options=dict(unknown=True))
                    self.assertEqual(b._variation(), current)
                b = adapter.NewtonSurfaceRinsingBackend(task=PLATES, contract=V2.CONTRACT, cameras=(), recording=False,
                                                        output_dir=Path(tmp)/"switch")
                self.assertIn("49 C (110-120 F)", b.select_variation({})["task"])
                request = b.select_variation(dict(contract=K.CONTRACT))
                self.assertNotIn("49 C", request["task"])
                b.reset(**request["reset_request"])
                self.assertEqual(b.config.task_contract, K.CONTRACT)
                b.reset(options=dict(contract=V2.CONTRACT))
                self.assertEqual(b.config.task_contract, V2.CONTRACT)
                b.reset(options=dict(task="scrub-rinse-pan"))
                self.assertEqual(b.config.task_contract, "legacy")
                with self.assertRaises(ValueError):
                    b.reset(options=dict(task="scrub-rinse-pan", contract=V2.CONTRACT))
                for options in (dict(task="scrub-rinse-pan"), dict(task="rinse-dishware"), dict(policy_samples=True),
                                dict(device="cuda:0"), dict(foo=1)):
                    with self.assertRaises(ValueError):
                        adapter.NewtonSurfaceRinsingBackend(**options, contract=V2.CONTRACT, output_dir=Path(tmp)/"refused")

    def test_runtime_installs_the_v2_evaluator_only_for_v2(self):
        verifiers()
        cp = module("classroom_arrangement.params", REFLEX_PROFILES={"contact_x5": {}})

        class OldEvaluator:
            impacts = object()
            _track = object()

        class BaseEnv:
            def __init__(self, config, *, spec, **kwargs):
                self.spec = spec
                self.evaluator = OldEvaluator()
                self.world = types.SimpleNamespace(recorders=[self.evaluator.impacts, self.evaluator._track, "unrelated"])

            def checkpoint(self):
                return dict(evaluator=self.evaluator)

        class V1Evaluator:
            def __init__(self, env):
                env.world.recorders.append(self)

        class NewEvaluator(V1Evaluator):
            pass

        stubs = {"surface_rinsing.env": module("surface_rinsing.env", RinsingEnv=BaseEnv),
                 "classroom_arrangement.params": cp, "classroom_arrangement": module("classroom_arrangement", params=cp)}
        specs = {}
        for name, path in (("surface_rinsing.task_set.runtime", "surface_rinsing/task_set/runtime.py"),
                           ("surface_rinsing.task_set_v2_runtime", "surface_rinsing/task_set_v2_runtime.py")):
            specs[name] = importlib.util.spec_from_file_location(name, ROOT/path)
            stubs[name] = importlib.util.module_from_spec(specs[name])
        with stub_modules(stubs):
            for name, spec in specs.items():
                spec.loader.exec_module(stubs[name])
            runtime = stubs["surface_rinsing.task_set_v2_runtime"]
            with patch("surface_rinsing.task_set.evaluator.Evaluator", V1Evaluator), \
                    patch("surface_rinsing.task_set_v2_evaluator.Evaluator", NewEvaluator):
                for task in K.EIGHT:
                    for contract in ("legacy", K.CONTRACT, V2.CONTRACT):
                        try:
                            config = runtime.Config(task=task, task_contract=contract)
                        except ValueError:
                            self.assertTrue(contract != "legacy" and task not in K.REVISED or
                                            contract == "legacy" and task == K.NEW_TASK, (task, contract))
                            continue
                        env = runtime.RinsingEnv(config)
                        self.assertEqual(env.spec["task"], task)
                        self.assertEqual(env.spec["task_contract"], contract)
                        if contract == V2.CONTRACT:
                            self.assertIs(type(env.evaluator), NewEvaluator)
                            self.assertEqual(env.world.recorders, ["unrelated", env.evaluator])
                            self.assertIs(env.initial["evaluator"], env.evaluator)
                            self.assertEqual(env.spec["instruction"], V2.TASKS[task]["instruction"])
                        elif contract == K.CONTRACT:
                            self.assertIs(type(env.evaluator), V1Evaluator)
                            self.assertEqual(env.spec["instruction"], K.TASKS[task]["instruction"])
                        else:
                            self.assertIsInstance(env.evaluator, OldEvaluator)


if __name__ == "__main__":
    unittest.main()

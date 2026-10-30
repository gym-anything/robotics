"""Task-only tests. Never import Newton, Warp, MuJoCo, a server or a renderer.

Run with Python 3.12, numpy and scipy. Missing shared geometry packages are
stubbed; the real catalogue, legacy predicates, new scoring and tracker run on
constructed state. These are verifier checks, not physics/solvability evidence.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from surface_rinsing.task_set import catalogue as K
from tests.packages import PACKAGES

ROOT = PACKAGES


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@contextmanager
def stub_modules(stubs):
    # Restore only replaced names. patch.dict(sys.modules) also removes native
    # extensions imported inside its context; numpy cannot safely reload them.
    previous = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for k, old in previous.items():
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old


@lru_cache(maxsize=1)
def verifiers():
    cp = module("classroom_arrangement.params", DT=0.0005, GRAVITY=9.81, REFLEX_PROFILES={"contact_x5": {}})
    furniture = module("classroom_arrangement.furniture", Part=object)
    geo = module("wine_pouring.geometry")
    stubs = {"classroom_arrangement": module("classroom_arrangement", params=cp),
             "classroom_arrangement.params": cp, "classroom_arrangement.furniture": furniture,
             "wine_pouring": module("wine_pouring", geometry=geo), "wine_pouring.geometry": geo}
    with stub_modules(stubs):
        return importlib.import_module("surface_rinsing.task_set.evaluator")


class CatalogueTests(unittest.TestCase):
    def test_exact_eight_and_legacy_default(self):
        b = json.loads((ROOT/"surface_rinsing/benchmark.json").read_text())
        rows = b["task_set"]["tasks"]
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["selection"] for r in rows], K.selections())
        self.assertEqual(len(K.TASKS), 9)
        self.assertEqual(K.task_resolve({})["task_id"], b["default_task_id"])
        for row in rows:
            self.assertEqual(row["task_id"], K.task_resolve(row["selection"])["task_id"])
        for task in K.LEGACY_TASKS:
            old = K.task_resolve(dict(task=task, seed=5))
            self.assertEqual(old["task_id"], f"surface-rinsing/v1/{K.TASKS[task]['preset']}/{task}/seed-5")
            self.assertNotIn("contract", old["backend_options"])
            self.assertEqual(old["description"], K.LEGACY_TASKS[task]["instruction"])

    def test_legacy_literals_and_resolver_still_match(self):
        verifiers()
        from surface_rinsing import layout, variants
        self.assertEqual(layout.TASKS, K.LEGACY_TASKS)
        for t in K.LEGACY_TASKS:
            for seed in (0, 7, 999, 1000, 9999):
                self.assertEqual(K.task_resolve(dict(task=t, seed=seed)), variants.task_resolve(dict(task=t, seed=seed)))
        for preset in K.STATIONS:
            self.assertEqual(K.resolve(preset=preset), variants.resolve(preset=preset))

    def test_validation_and_contract_identity(self):
        for seed in (True, "4", 1.0, -1, 10000, None):
            with self.assertRaises(ValueError):
                K.resolve(seed=seed)
        for args in (dict(task="no"), dict(preset="no"), dict(task="flush-floor", preset="pot-sink"),
                     dict(task=K.NEW_TASK, contract="legacy"), dict(task="scrub-rinse-pan", contract=K.CONTRACT),
                     dict(task="rinse-dishware", contract=K.CONTRACT), dict(contract=[])):
            with self.assertRaises(ValueError):
                K.resolve(**args)
        a = K.task_resolve(dict(task="rinse-glassware"))
        b = K.task_resolve(dict(task="rinse-glassware", contract=K.CONTRACT))
        self.assertNotEqual(a["task_id"], b["task_id"])
        self.assertEqual(K.resolve(task="scrub-rinse-pan", contract=K.PAN_CONTRACT), K.resolve(task="scrub-rinse-pan"))

    def test_setups_no_new_placement_or_physics_changes(self):
        verifiers()
        from surface_rinsing import layout as L
        for seed in (0, 3, 99):
            for task in K.EIGHT:
                v = K.resolve(task=task, contract=K.CONTRACT if task in K.REVISED else None)
                got = K.setup(task, seed, v.get("contract"))
                old = L.setup("rinse-sponge" if task == K.NEW_TASK else task, seed)
                if task == K.NEW_TASK:
                    self.assertEqual(got["items"], {})
                    old.update(items={}, task=task)
                got.pop("task_contract")
                got["instruction"] = old["instruction"]
                self.assertEqual(got, old)

    def test_drain_reach_envelope(self):
        verifiers()
        from surface_rinsing import layout as L
        # Source-derived geometry only. Shoulder centre from layout's documented
        # robot offset; arm reach 0.855 m. Not an IK or collision certificate.
        x, y, _ = L.ROBOT_START["pot-sink"]
        shoulder = (x, y + 0.53, 0.71 + L.ROBOT_SPINE["pot-sink"])
        for x0, x1 in L.BOWLS.values():
            pivot = ((x0+x1)/2, L.SINK_Y0 - 0.012, L.RIM - 0.055)
            # Include the 90 mm lever tip sweep and 22 mm front offset.
            self.assertLess(math.dist(shoulder, pivot) + 0.112, 0.855)


def reading(task):
    ledger = dict(relative=0., relative_surf=0., relative_solids=0., surfactant_kg=0., solids_kg=0.,
                  lost_water_kg=0., lost_water_by_cause_kg={}, created_by_clamps_kg={},
                  lifted_from_under_floor_kg=0., kept_by_floor_plane_unstruck_kg=0.)
    m = dict(items={}, open_outlets={}, valves_from_park_m={"spray": 0.0}, holding=False,
             sponge=dict(c_g_L=0.75, fill=0.75), contamination=dict(floor_water_mL=0.), ledger=ledger, fault=None,
             pools={n: dict(volume_L=70.) for n in ("wash", "rinse", "sanitize")}, task_set={})
    item = dict(position=[3., 5., 1.], tilt_deg=0., max_impact_m_s=0., surfactant_ug=1000.,
                surfactant_gauge_max_mg_m2=20., foam_mL=0.5, soil_gauge_max_ug_cm2=100.,
                free_water_mL=20., sanitized_s=0., surfactant_into_sanitizer_ug=0.)
    names = {"rinse-sponge": ["sponge"], K.NEW_TASK: [], "rinse-tomatoes": [f"tomato_{i}" for i in range(15)] + ["colander", "clean_bowl"],
             "flush-floor": ["can"], "rinse-glassware": [f"glass_{i}" for i in range(6)] + ["glass_rack"],
             "pre-rinse-plates": [f"plate_{i}" for i in range(6)] + ["peg_rack"], "rinse-garbage-can": ["can"]}[task]
    m["items"] = {n: copy.deepcopy(item) for n in names}
    if task == K.NEW_TASK:
        m["task_set"].update(waste_rad={n: 0. for n in m["pools"]}, drained_L={n: 0. for n in m["pools"]})
    if task == "flush-floor":
        m["task_set"].update(floor_gauges=[100., 100., 0.], floor_areas=[1., 3., 9.])
    if task == "rinse-garbage-can":
        m["task_set"]["can_soap_limit_ug"] = 10.
    return m


def evaluator(task, m):
    E = verifiers()
    ev = E.Evaluator.__new__(E.Evaluator)
    ev.task, ev.pan, ev.glasses = task, None, {}
    ev.floor_items = {"can"}
    ev.start = copy.deepcopy(m)
    ev.env = types.SimpleNamespace(world=types.SimpleNamespace(scene=types.SimpleNamespace(
        items={n: dict(kind="rocks_glass" if n.startswith("glass_") and n != "glass_rack" else
                       "plate" if n.startswith("plate_") else n) for n in m["items"]})),
        water=types.SimpleNamespace(surf=types.SimpleNamespace(area=np.array([0.085])), item_cells={"can": np.array([0])}))
    ev.sponge_report = lambda: m["sponge"]
    ev._on_ledge = lambda *a, **kw: True
    ev._in_rack_cell = lambda n, r: m["items"][n].get("placed", False)
    ev._in_peg_rack = lambda n, r: m["items"][n].get("placed", True)
    ev._inside_cavity = lambda n, r: m["items"][n].get("placed", n.startswith("tomato"))
    # Floor criteria reads raw fields rather than the report; supply those here.
    if task == "flush-floor":
        surf = types.SimpleNamespace(area=np.array([1., 3., 9.]), pos=np.array([[0., 0., 0.], [0.02, 0., 0.], [0.04, 0., 0.]]),
                                     nrm=np.array([[0., 0., 1.]]*3))
        ev.env.water.surf = surf
        ev.env.water.static_charts = {"bay_floor": types.SimpleNamespace(first=0, count=3)}
        ev._fields = lambda: dict(soil=np.array(m["task_set"]["floor_gauges"])/1e5,
                                 sus=np.zeros(3), s=np.zeros(3), gamma=np.zeros(3))
    return ev


def complete(task, m):
    for n, it in m["items"].items():
        it.update(surfactant_ug=0., surfactant_gauge_max_mg_m2=0., foam_mL=0., soil_gauge_max_ug_cm2=0.,
                  sanitized_s=31., free_water_mL=0., placed=True)
        it["tilt_deg"] = 180. if task == "rinse-glassware" else 0.
    m["sponge"].update(c_g_L=0., fill=0.4)
    if task == K.NEW_TASK:
        m["pools"]["wash"]["volume_L"] = 0.4
    if task == "flush-floor":
        m["task_set"]["floor_gauges"] = [0., 0., 0.]


class ScoringTests(unittest.TestCase):
    def test_hypothetical_settle_cannot_give_untouched_start_credit(self):
        m = reading("rinse-glassware")
        ev = evaluator("rinse-glassware", m)
        ev._progress_input = copy.deepcopy(m)
        for it in m["items"].values():
            it["surfactant_ug"] *= 0.8
        self.assertEqual(ev._score(ev.criteria(m), m)[0], 0.)

    def test_all_revised_untouched_states_are_zero_and_not_success(self):
        for task in K.REVISED:
            with self.subTest(task=task):
                m = reading(task)
                ev = evaluator(task, m)
                c = ev.criteria(m)
                self.assertFalse(all(v["passed"] for v in c.values()))
                value, rw = ev._score(c, m)
                self.assertEqual(value, 0.)
                self.assertEqual(rw["contract"], K.CONTRACT)

    def test_every_success_scores_one_and_failure_never_one(self):
        for task in K.REVISED:
            with self.subTest(task=task):
                m = reading(task)
                ev = evaluator(task, m)
                complete(task, m)
                c = ev.criteria(m)
                self.assertTrue(all(v["passed"] for v in c.values()), c)
                self.assertEqual(ev._score(c, m)[0], 1.)
                m["holding"] = True
                self.assertLessEqual(ev._score(ev.criteria(m), m)[0], .9)

    def test_each_primary_quantity_has_interior_progress(self):
        for task in K.REVISED:
            m = reading(task)
            ev = evaluator(task, m)
            for it in m["items"].values():
                it["surfactant_ug"] /= 2
                it["soil_gauge_max_ug_cm2"] /= 2
            m["sponge"]["c_g_L"] /= 2
            m["pools"]["wash"]["volume_L"] = 35.
            if task == "flush-floor":
                m["task_set"]["floor_gauges"] = [50., 50., 0.]
            value = ev._score(ev.criteria(m), m)[0]
            self.assertGreater(value, 0., task)
            self.assertLess(value, .9, task)

    def test_floor_area_continuity_and_shelf_cost(self):
        m = reading("flush-floor")
        ev = evaluator("flush-floor", m)
        m["task_set"]["floor_gauges"] = [10., 100., 0.]
        self.assertAlmostEqual(ev._score(ev.criteria(m), m)[1]["terms"]["clean_area"], .25)
        m["task_set"]["floor_gauges"][1] = 99.99
        value = ev._score(ev.criteria(m), m)[0]
        self.assertGreater(value, .9*.25)
        self.assertLess(value, .9*.251)
        m["contamination"]["dunnage_shelf"] = dict(water_g=2.)
        self.assertEqual(ev._score(ev.criteria(m), m)[0], 0.)

    def test_wrong_bowl_refill_does_not_erase_loss(self):
        m = reading(K.NEW_TASK)
        ev = evaluator(K.NEW_TASK, m)
        complete(K.NEW_TASK, m)
        m["task_set"]["drained_L"]["rinse"] = 35.
        c = ev.criteria(m)
        self.assertFalse(c["others_kept"]["passed"])
        self.assertAlmostEqual(ev._score(c, m)[0], .45)
        m["task_set"]["waste_rad"]["wash"] = math.radians(5.001)
        self.assertFalse(ev.criteria(m)["waste_closed"]["passed"])

    def test_lost_tomato_broken_glass_and_contamination_cost(self):
        for task, target, fault in (("rinse-tomatoes", "tomato_0", dict(placed=False)),
                                    ("rinse-glassware", "glass_0", dict(max_impact_m_s=1.01))):
            m = reading(task)
            ev = evaluator(task, m)
            complete(task, m)
            m["items"][target].update(fault)
            c = ev.criteria(m)
            rw = ev._score(c, m)[1]
            self.assertLess(rw["progress"], 1.)
            m["contamination"]["floor_water_mL"] = 100.
            self.assertLess(ev._score(ev.criteria(m), m)[1]["progress"], rw["progress"])

    def test_recontamination_reduces_progress_and_any_order_is_accepted(self):
        m = reading("rinse-glassware")
        ev = evaluator("rinse-glassware", m)
        complete("rinse-glassware", m)
        m["holding"] = True
        before = ev._score(ev.criteria(m), m)[0]
        m["items"]["glass_2"]["surfactant_ug"] = 1000.
        self.assertLess(ev._score(ev.criteria(m), m)[0], before)
        m["items"] = dict(reversed(list(m["items"].items())))
        self.assertEqual(ev._score(ev.criteria(m), m), ev._score(ev.criteria(copy.deepcopy(m)), copy.deepcopy(m)))

    def test_unverifiable_states_withhold_score(self):
        for task in K.REVISED:
            for kind in ("fault", "water", "detergent", "soil", "nan"):
                m = reading(task)
                ev = evaluator(task, m)
                complete(task, m)
                if kind == "fault":
                    m["fault"] = "nonfinite physics"
                elif kind == "water":
                    m["ledger"]["relative"] = .001
                elif kind in ("detergent", "soil"):
                    r, a = ("relative_surf", "surfactant_kg") if kind == "detergent" else ("relative_solids", "solids_kg")
                    m["ledger"].update({r: .001, a: .01})
                else:
                    m["contamination"]["floor_water_mL"] = float("nan")
                value, rw = ev._score(ev.criteria(m), m)
                self.assertIsNone(value, (task, kind))
                self.assertFalse(rw["verifiable"])

    def test_sponge_ledge_uses_original_ledge_geometry(self):
        E = verifiers()
        from surface_rinsing import layout as L
        ev = E.Evaluator.__new__(E.Evaluator)
        ev.items = {"sponge": 0}
        p = np.array([3.296, (L.BOWL_Y1+L.Y)/2, L.RIM+.0495])
        ev.env = types.SimpleNamespace(world=types.SimpleNamespace(body_pose=lambda b: (p, np.eye(3))))
        self.assertTrue(ev._on_ledge("sponge", compartments=True))
        p[0] = sum(L.DRAINBOARDS["clean"])/2
        self.assertFalse(ev._on_ledge("sponge", compartments=True))


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values.copy()

    def __getitem__(self, index):
        return Array(self.values[index])


def tracker_fixture():
    verifiers()
    from surface_rinsing.task_set.sanitizer import GlassTracker
    # Two outer and two inner surface cells, one glass cavity, three sink bowls.
    eng = types.SimpleNamespace(sub=Array([-1]*4), sp=Array(np.zeros((6, 4))), gamma=Array([0.]*4),
                                soil=Array([0.]*4), h=Array([0.]*4))
    cavity = types.SimpleNamespace(sp=np.zeros(6), vol=0.)
    water = types.SimpleNamespace(item_cells={"glass_0": np.arange(4)}, eng=eng,
                surf=types.SimpleNamespace(area=np.ones(4)*.01, pos=np.arange(12).reshape(4, 3)*.01,
                                          nrm=np.ones((4, 3))),
                item_charts={"glass_0": types.SimpleNamespace(cell_seg=np.arange(4))},
                pools=types.SimpleNamespace(index={"wash": 0, "rinse": 1, "sanitize": 2, "cavity:glass_0": 3},
                                           pools=[None, None, None, cavity]), immersed={}, outlets={})
    pos = np.array([3., 4., 1.])
    world = types.SimpleNamespace(scene=types.SimpleNamespace(items={"glass_0": dict(body=0)},
                    ware={"glass_0": types.SimpleNamespace(kind="glass", regions={"glass": {"inner": [2, 3]}})}),
                                  body_pose=lambda b: (pos, np.eye(3)), sim_time=0.)
    env = types.SimpleNamespace(water=water, world=world)
    t = GlassTracker(env, "glass_0")
    ev = types.SimpleNamespace(sanitize={"glass_0": 0.}, surf_into_sanitizer={"glass_0": 0.})
    return t, ev, env


class SanitizerTests(unittest.TestCase):
    def test_qualifying_redips_accumulate_but_rinse_resets(self):
        t, ev, env = tracker_fixture()
        t.sample(ev)
        ws = env.water
        for _ in range(2):
            ws.immersed = {"glass_0": ("sanitize", 1.)}
            ws.eng.sub = Array([2]*4)
            t.sample(ev)
            ws.immersed = {}
            ws.eng.sub = Array([-1]*4)
            t.sample(ev)
        self.assertAlmostEqual(ev.sanitize["glass_0"], .02)
        ws.immersed = {"glass_0": ("rinse", 1.)}
        ws.eng.sub = Array([1]*4)
        t.sample(ev)
        self.assertEqual(ev.sanitize["glass_0"], 0.)

    def test_dirty_entry_requires_real_rinse_then_can_recover(self):
        t, ev, env = tracker_fixture()
        ws = env.water
        ws.eng.sp.values[0, :] = 1e-4
        t.sample(ev)
        ws.immersed = {"glass_0": ("sanitize", 1.)}
        ws.eng.sub = Array([2]*4)
        ws.eng.sp.values[:] = 0.
        t.sample(ev)
        self.assertIsNotNone(t.state["owed"])
        self.assertEqual(ev.sanitize["glass_0"], 0.)
        for pool, index in ((None, -1), ("sanitize", 2), ("rinse", 1)):
            ws.immersed = {"glass_0": (pool, 1.)} if pool else {}
            ws.eng.sub = Array([index]*4)
            t.sample(ev)
        self.assertIsNone(t.state["owed"])
        ws.immersed = {"glass_0": ("sanitize", 1.)}
        ws.eng.sub = Array([2]*4)
        t.sample(ev)
        self.assertGreater(ev.sanitize["glass_0"], 0.)

    def test_air_pocket_does_not_count_and_snapshot_restores(self):
        t, ev, env = tracker_fixture()
        t.sample(ev)
        env.water.immersed = {"glass_0": ("sanitize", 1.)}
        env.water.eng.sub = Array([2, 2, -1, -1])
        t.sample(ev)
        self.assertEqual(ev.sanitize["glass_0"], 0.)
        state = copy.deepcopy(t.snapshot())
        t.state["owed"] = dict(why="test")
        t.restore(state)
        self.assertEqual(t.snapshot(), state)
        env.water.eng.sub = Array([2]*4)
        t.frozen = True
        t.sample(ev)
        self.assertEqual(ev.sanitize["glass_0"], 0.)


class IntegrationTests(unittest.TestCase):
    def test_backend_construct_select_reset_rebuild_all_eight_with_stub_world(self):
        verifiers()
        cp = module("classroom_arrangement.params", REFLEX_PROFILES={"contact_x5": {}})

        class FakeEnv:
            created = []

            def __init__(self, config, **kwargs):
                self.config = config
                self.world = types.SimpleNamespace(arms=("left", "right"))
                self.lock = threading.RLock()
                self.generation = 0
                self.created.append(config)

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
        with stub_modules(stubs):
            # Load the actual config and constructor/lifecycle code; only the
            # world and shared backend facilities are stand-ins.
            runtime = importlib.import_module("surface_rinsing.task_set.runtime")
            spec = importlib.util.spec_from_file_location("gym_anything_robotics.backends._rinsing_task_set_test",
                         ROOT/"gym_anything_robotics/backends/newton_surface_rinsing_task_set.py")
            backend_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backend_module)
            cls = backend_module.NewtonSurfaceRinsingBackend
            with patch.object(runtime, "RinsingEnv", FakeEnv), tempfile.TemporaryDirectory() as tmp:
                for i, selection in enumerate(K.selections()):
                    b = cls(**selection, cameras=(), recording=False, output_dir=Path(tmp)/str(i))
                    self.assertEqual(b._variation(), selection)
                    self.assertEqual(b.config.task, selection["task"])
                    # The inherited API coerces strings/bools through int().
                    b.reset(seed="3")
                    self.assertEqual(b.config.seed, 3)
                    self.assertEqual(b.env.generation, 1)
                    b.reset(seed=True)
                    self.assertEqual(b.config.seed, 1)
                    current = b._variation()
                    with self.assertRaises(ValueError):
                        b.reset(options=dict(unknown=True))
                    self.assertEqual(b._variation(), current)
                b = cls(task="rinse-glassware", contract=K.CONTRACT, cameras=(), recording=False,
                        output_dir=Path(tmp)/"switch")
                request = b.select_variation(dict(contract="legacy"))
                b.reset(**request["reset_request"])
                self.assertNotIn("contract", b._variation())
                b.reset(options=dict(task=K.NEW_TASK))
                self.assertEqual(b.config.task_contract, K.CONTRACT)
                b.reset(options=dict(task="scrub-rinse-pan"))
                self.assertEqual(b.config.task_contract, "legacy")
                for options in (dict(policy_samples=True), dict(reflex_stops_base=1), dict(device="cuda:0"), dict(foo=1)):
                    with self.assertRaises(ValueError):
                        cls(**options, output_dir=Path(tmp)/"refused")

    def test_runtime_replaces_only_revised_evaluator_and_records(self):
        verifiers()
        source = ROOT/"surface_rinsing/task_set/runtime.py"
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

        class NewEvaluator:
            def __init__(self, env):
                env.world.recorders.append(self)

        with stub_modules({"surface_rinsing.env": module("surface_rinsing.env", RinsingEnv=BaseEnv),
                           "classroom_arrangement.params": cp,
                           "classroom_arrangement": module("classroom_arrangement", params=cp)}):
            spec = importlib.util.spec_from_file_location("surface_rinsing.task_set._runtime_test", source)
            runtime = importlib.util.module_from_spec(spec)
            # dataclass resolves annotation modules through sys.modules.
            with stub_modules({spec.name: runtime}):
                spec.loader.exec_module(runtime)
                with patch("surface_rinsing.task_set.evaluator.Evaluator", NewEvaluator):
                    for task in K.EIGHT:
                        contract = K.CONTRACT if task in K.REVISED else "legacy"
                        env = runtime.RinsingEnv(runtime.Config(task=task, task_contract=contract))
                        self.assertEqual(env.spec["task"], task)
                        if contract == K.CONTRACT:
                            self.assertIsInstance(env.evaluator, NewEvaluator)
                            self.assertEqual(env.world.recorders, ["unrelated", env.evaluator])
                            self.assertIs(env.initial["evaluator"], env.evaluator)
                        else:
                            self.assertIsInstance(env.evaluator, OldEvaluator)

    def test_evaluator_checkpoint_and_frozen_settle_restore(self):
        E = verifiers()
        t, fake, env = tracker_fixture()
        ev = E.Evaluator.__new__(E.Evaluator)
        ev.pan, ev.glasses, ev.frozen = None, {"glass_0": t}, False
        ev.sanitize, ev.surf_into_sanitizer, ev._in_san = fake.sanitize, fake.surf_into_sanitizer, set()
        ev.impacts = types.SimpleNamespace(max={}, prev=set())
        ev.start = reading("rinse-glassware")
        ev.env = env
        saved = copy.deepcopy(ev.snapshot())
        ev.start["items"].clear()
        t.state["owed"] = "wrong"
        ev.sanitize["glass_0"] = 100.
        ev.restore(saved)
        self.assertEqual(ev.snapshot(), saved)
        env.world.tick = 20
        ev.measure = lambda: copy.deepcopy(ev.start)

        def settle(self, seconds):
            self._track(env.world)
            self.restore(saved)
            return dict(score=0.)

        before = copy.deepcopy(t.snapshot())
        with patch.object(E.LegacyEvaluator, "evaluate", settle):
            self.assertEqual(ev.evaluate()["contract"], K.CONTRACT)
        self.assertEqual(t.snapshot(), before)
        self.assertFalse(ev.frozen)
        self.assertFalse(hasattr(ev, "_progress_input"))


if __name__ == "__main__":
    unittest.main()

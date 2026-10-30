"""Constructed film/pose tests only. No world, simulator, server, or film advancement.

Run: python -m pytest tests/environments/painting/test_task_set.py
classroom_arrangement's two imported constants and configuration names are
stubbed. Backend methods are loaded from AST against a stub
environment so launch/reset wiring can be tested without importing its runtime.
"""
import ast
import copy
import importlib.abc
import json
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from tests.packages import PACKAGES

ROOT = PACKAGES


class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"warp", "newton", "mujoco", "isaacsim"} or fullname in {
            "painting.world", "painting.env", "painting.g1", "painting.system"}:
            raise AssertionError(f"Simulator import forbidden in these tests: {fullname}")


def setUpModule():
    global _modules, _guard
    _modules = patch.dict(sys.modules, {
        "classroom_arrangement": ModuleType("classroom_arrangement"),
        "classroom_arrangement.params": NS(GRAVITY=9.81, DT=0.0005,
                                            REFLEX_PROFILES={"contact_x5": {}}),
    })
    _modules.start()
    from painting import params as P
    # _roller_loaded imports just this constant from the physical roller module.
    sys.modules["painting.roller"] = NS(SAT=P.ROLLER["sat"])
    _guard = NoSimulator()
    sys.meta_path.insert(0, _guard)


def tearDownModule():
    sys.meta_path.remove(_guard)
    _modules.stop()


class Chart:
    """Just arrays and measurements: no film transport or dynamics methods."""
    def __init__(self, name, shape=(100, 100), dx=0.002, paint=None):
        from painting.paint import make_paint, srgb_to_linear
        from painting import params as P
        self.name, self.dx, self.area = name, dx, dx*dx
        self.nu, self.nv = shape
        self.size = (self.nu*dx, self.nv*dx)
        self.origin = np.zeros(3)
        self.mask = np.ones(shape, bool)
        self.d, self.s = np.zeros((1, *shape)), np.zeros((1, *shape))
        self.w = np.zeros(shape)
        self.lap = np.zeros(shape, np.uint8)
        self.touched = np.zeros(shape, np.uint8)
        self.marred = np.zeros(shape, np.uint8)
        self.flowed, self.moved = np.zeros(shape), np.zeros(shape)
        self.d_base = None
        self.paints = [paint or make_paint("latex", "alabaster")]
        self.substrate_rgb = np.broadcast_to(srgb_to_linear(P.COLORS["accessible_beige"]), (*shape, 3)).copy()
        self.role, self.version, self.bbox = "target", 0, None

    def projected_dft(self):
        return (self.d + self.s).sum(axis=0)

    def h(self, sl=None):
        h = self.s.sum(axis=0) + self.w
        return h if sl is None else h[sl]

    def totals(self):
        return dict(wet_solids=self.s.sum(axis=(1, 2))*self.area,
                    dry_solids=self.d.sum(axis=(1, 2))*self.area, drained_solids=np.zeros(1))

    def fill(self, microns, mask=None):
        self.d[0, self.mask if mask is None else mask] = microns*1e-6
        self.version += 1

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, state):
        self.__dict__.update(copy.deepcopy(state))


def evaluator(task, new=True, prepaint=False):
    from painting.catalogue import CONTRACT
    from painting.layout import TASKS
    from painting.task import Evaluator
    key = ("paper" if TASKS[task]["preset"] == "craft" else
           "crate" if task == "spray-stencil" else "print" if task == "clear-coat" else "wall")
    ch = Chart(key)
    if key == "paper":
        from painting.paint import make_paint
        ch.paints = [make_paint("acrylic", "cadmium_red" if task == "brush-stroke" else "ultramarine")]
    m = ch.mask.copy()
    if key in ("paper", "crate"):
        m[:] = False
        m[10:90, 10:90] = True
    targets = {key: m}
    if key == "paper":
        targets["outside"] = ~m
    if prepaint:
        ch.fill(100, np.indices(ch.mask.shape)[0] < 10)
    pose = np.array([[0, 0, 0.3, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], float)
    roller = NS(s=np.zeros((1, 10)), v=np.zeros(10), C0=np.ones(10), frame=0, cover=0,
                mass=0.0)
    roller.load_mass = lambda: roller.mass
    world = NS(tick=0, state=NS(body_q=NS(numpy=lambda: pose)), scene=NS(bodies={"tray": 1}, robot_bodies=[]),
               grip_force={"right": 1.0}, arms=["right"], model=NS(body_label=[]), contact_table=lambda: [])
    paint = NS(charts={} if task == "load-roller" else {key: ch}, sinks={}, skins={},
               drips=NS(landed=0, pos=[]), paints=ch.paints,
               applicators={"roller": roller, "aerosol": NS(stats={"emitted_m3": 0.0, "emitted_solids_m3": 0.0}, ps=np.zeros(0))})
    env = NS(spec=dict(task=task, preset=TASKS[task]["preset"], seed=0), paint=paint, world=world,
             config=NS(contract=CONTRACT if new and task != "behind-radiator" else None))
    with patch.object(Evaluator, "_targets", return_value={} if task == "load-roller" else targets):
        ev = Evaluator(env)
    return ev, ch, m, pose


class Catalogue(unittest.TestCase):
    def test_exact_eight_and_old_default(self):
        from painting.catalogue import CONTRACT, EIGHT_TASKS, selections
        from painting.variants import task_resolve, contract
        from painting.layout import TASKS
        raw = json.loads((ROOT/"painting/benchmark.json").read_text())
        rows = raw["task_set"]["tasks"]
        self.assertEqual(len(TASKS), 11)
        self.assertEqual([r["selection"]["task"] for r in rows], list(EIGHT_TASKS))
        self.assertEqual([r["selection"] for r in rows], selections())
        self.assertEqual(contract()["task_set"], selections())
        ids = set()
        for row in rows:
            resolved = task_resolve(row["selection"])
            self.assertEqual(resolved["task_id"], row["task_id"])
            self.assertEqual(resolved["backend_options"], row["selection"])
            ids.add(resolved["task_id"])
        self.assertEqual(len(ids), 8)
        default = task_resolve({})
        self.assertEqual(default["task_id"], raw["default_task_id"])
        self.assertEqual(default["backend_options"], dict(preset="wall", task="roll-strip", seed=0))
        self.assertEqual(default["description"], TASKS["roll-strip"]["instruction"])
        self.assertEqual(task_resolve({"contract": "checks-v1"}), default)
        self.assertNotEqual(task_resolve({"contract": CONTRACT})["task_id"], default["task_id"])

    def test_contract_validation(self):
        from painting.catalogue import CONTRACT, RADIATOR
        from painting.config import Config
        from painting.variants import task_resolve, resolve
        for selection in ({"task": "behind-radiator", "contract": CONTRACT},
                          {"task": "touch-up", "contract": CONTRACT}, {"contract": "typo"},
                          {"seed": True}, {"seed": 1.5}, {"seed": "1"}, {"seed": -1}, {"seed": 10000},
                          {"task": "clear-coat", "preset": "craft"}, {"unknown": 1}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                task_resolve(selection)
        self.assertEqual(resolve(task="behind-radiator", contract=RADIATOR), resolve(task="behind-radiator"))
        self.assertIsNone(Config().contract)
        with self.assertRaises(ValueError):
            Config(contract="typo")

    def test_instructions_and_starts(self):
        from painting.catalogue import EIGHT_TASKS, CONTRACT, apply_contract, describe
        from painting.layout import setup
        # All placements, geometry, fill, grasp, climate and prepaint survive the
        # metadata overlay byte-for-byte for nominal and jittered seeds.
        for task in EIGHT_TASKS:
            for seed in (0, 1, 2, 3, 4, 9999):
                old = setup(task, seed)
                saved = copy.deepcopy(old)
                chosen = None if task == "behind-radiator" else CONTRACT
                new = apply_contract(old, chosen)
                self.assertEqual(old, saved)
                self.assertEqual({k: v for k, v in new.items() if k != "instruction"},
                                 {k: v for k, v in old.items() if k != "instruction"})
                if task == "behind-radiator":
                    self.assertEqual(new, old)
                else:
                    self.assertEqual(new["instruction"], describe(task, seed, chosen))
                    for verifier_word in ("50 g", "10 s", "95 %", "98 %", "25-50", "1 mm", "score"):
                        self.assertNotIn(verifier_word, new["instruction"])
                if task == "roll-strip" and seed % 3 == 0:
                    self.assertIn("second, finish coat", new["instruction"])

        # No new placements: retained P0 start TCPs and P13 start checks are the
        # relevant existing reach evidence, not a new IK or motion-plan claim.


class Rewards(unittest.TestCase):
    def test_all_eight_untouched_zero_including_prepaint(self):
        from painting.catalogue import EIGHT_TASKS, CONTRACT, RADIATOR
        for task in EIGHT_TASKS:
            ev, _, _, _ = evaluator(task, prepaint=task == "roll-strip")
            for settled in (False, True):
                out = ev.measure(settled)
                self.assertFalse(out["succeeded"], task)
                self.assertAlmostEqual(out["score"], 0.0, places=12, msg=task)
                self.assertEqual(out["contract"], RADIATOR if task == "behind-radiator" else CONTRACT)
                self.assertEqual(out["reward"]["contract"], out["contract"])

    def test_paper_thin_coverage_overrun_and_marks(self):
        for task, threshold in (("swatch-card", 15), ("brush-stroke", 5)):
            ev, ch, mask, _ = evaluator(task)
            scores = []
            for dft in (0, threshold/4, threshold/2, threshold*0.99):
                ch.fill(dft, mask)
                out = ev.measure(True)
                self.assertFalse(out["succeeded"])
                scores.append(out["score"])
            self.assertEqual(scores, sorted(set(scores)))
            self.assertLessEqual(scores[-1], 0.9)
            ch.fill(threshold + 1, mask)
            self.assertEqual(ev.measure(True)["score"], 1.0)
            ch.fill(10, ~mask)
            out = ev.measure(True)
            self.assertFalse(out["checks"]["within_line"]["ok"])
            self.assertLess(out["score"], 0.9)
            ch.fill(0, ~mask)
            ch.touched[mask] = ch.marred[mask] = 1
            self.assertEqual(ev.measure(True)["score"], 0.0)

    def test_wall_thin_passes_and_prepaint_baseline(self):
        for task in ("roll-strip", "cut-in-trim"):
            ev, ch, mask, pose = evaluator(task, prepaint=True)
            ch.d_base = ch.projected_dft().copy()
            scores = []
            for dft in (0, 2, 5, 10):
                ch.d[0] = ch.d_base + dft*1e-6
                ch.version += 1
                scores.append(ev.measure(False)["score"])
            self.assertAlmostEqual(scores[0], 0.0)
            self.assertEqual(scores, sorted(scores))
            self.assertGreater(scores[-1], scores[0])
            ch.fill(100)
            pose[0, 2] = 0.05
            ev.env.world.grip_force["right"] = 0.0
            self.assertEqual(ev.measure(True)["score"], 1.0)
            ch.lap[mask] = 1
            self.assertEqual(ev.measure(True)["score"], 0.0)
            ch.lap[:] = 0
            ch.flowed[mask] = 60e-6
            ch.moved[mask] = 0.002
            self.assertEqual(ev.measure(True)["score"], 0.0)
            ch.flowed[:] = ch.moved[:] = 0
            ev.paint.sinks["radiator"] = dict(solids=np.zeros(1), vehicle=0.0, area=0.1)
            self.assertEqual(ev.measure(True)["score"], 0.0)

    def test_roller_load_spread_hold_and_success(self):
        from painting import params as P
        ev, _, _, pose = evaluator("load-roller")
        r = ev.paint.applicators["roller"]
        r.mass = 0.025
        r.v[:] = 0.1*P.ROLLER["sat"]*r.C0
        ev._last_drip_tick = 0
        first = ev.measure(False)
        self.assertGreater(first["score"], 0)
        r.mass = 0.060
        r.v[:] = 0.3*P.ROLLER["sat"]*r.C0
        ev.env.world.tick = int(5/P.DT)
        middle = ev.measure(False)
        self.assertAlmostEqual(middle["score"], 0.9*0.8)
        ev.env.world.tick = int(10/P.DT)
        self.assertEqual(ev.measure(True)["score"], 1)
        pose[0, 0] = 1.0
        self.assertAlmostEqual(ev.measure(True)["score"], 0.9*0.6)
        r.mass = 0
        self.assertEqual(ev.measure(True)["score"], 0)

    def test_spray_bands_containment_edges_evenness_and_wet_declaration(self):
        for task in ("clear-coat", "spray-stencil"):
            ev, ch, mask, _ = evaluator(task)
            ch.fill(12.5, mask)
            partial = ev.measure(True)
            self.assertTrue(0 < partial["score"] < 0.9)
            ch.fill(30, mask)
            self.assertEqual(ev.measure(True)["score"], 1)
            ev.paint.applicators["aerosol"].stats["emitted_solids_m3"] = 1e-6
            ev.paint.sinks["floor"] = dict(solids=np.array([0.2e-6]), vehicle=0.0, area=0.0)
            fail = ev.measure(True)
            self.assertFalse(fail["checks"]["overspray"]["ok"])
            self.assertLess(fail["score"], 0.9)
            ev.paint.sinks.clear()
            ch.w[:] = 500e-6
            ev._task_progress.capture_declaration()
            ev._decl_h = {ch.name: 500e-6}
            ch.w[:] = 0
            self.assertEqual(ev.measure(True)["score"], 0)
            self.assertFalse(ev.measure(True)["checks"]["no_runs"]["ok"])
            ev._task_progress.declaration.clear()
            ev._decl_h.clear()
            ch.fill(100, mask)
            self.assertFalse(ev.measure(True)["succeeded"])
            self.assertAlmostEqual(ev.measure(True)["score"], 0)
            ch.fill(30, mask)
            if task == "spray-stencil":
                ch.fill(30, ~mask)
                self.assertFalse(ev.measure(True)["checks"]["sharp_edges"]["ok"])
            else:
                ch.d[0, :50] = 10e-6
                self.assertFalse(ev.measure(True)["checks"]["even"]["ok"])
            self.assertLess(ev.measure(True)["score"], 0.9)

    def test_radiator_success_and_legacy_scores(self):
        from painting.catalogue import EIGHT_TASKS, RADIATOR, LEGACY
        ev, ch, _, _ = evaluator("behind-radiator")
        ch.fill(100)
        out = ev.measure(True)
        self.assertEqual(out["score"], 1)
        self.assertEqual(out["reward"]["contract"], RADIATOR)
        for task in EIGHT_TASKS[:-1]:
            ev, _, _, _ = evaluator(task, new=False)
            out = ev.measure(True)
            self.assertEqual(out["score"], np.mean([v["ok"] for v in out["checks"].values()]))
            self.assertEqual(out["contract"], LEGACY)
            self.assertNotIn("reward", out)

    def test_repaired_marks_flow_and_release_are_state_based(self):
        for new in (False, True):
            ev, ch, mask, pose = evaluator("roll-strip", new=new)
            ch.fill(100)
            pose[0, 2] = 0.05
            # The roller has no robot contact; the unused hand keeps its grip
            # command, the coat's old touched marks were repaired, and an even
            # stacked coat has moved while levelling but has no remaining run.
            ch.touched[mask] = 1
            ch.moved[mask] = 0.002
            out = ev.measure(True)
            self.assertEqual(out["succeeded"], new)
            if new:
                self.assertEqual(out["score"], 1)
                ev.env.world.scene.robot_bodies = [2]
                ev.env.world.model.body_label = ["roller", "tray", "left_finger"]
                ev.env.world.contact_table = lambda: [(2, 0)]
                self.assertFalse(ev.measure(True)["checks"]["roller_in_tray"]["ok"])
                ev.env.world.contact_table = lambda: []
                ch.flowed[:10] = 60e-6
                self.assertFalse(ev.measure(True)["checks"]["no_runs"]["ok"])
                ch.flowed[:] = 0
                ch.marred[:10] = 1
                self.assertFalse(ev.measure(True)["checks"]["untouched"]["ok"])
        ev, ch, mask, _ = evaluator("cut-in-trim")
        ch.fill(100)
        self.assertEqual(ev.measure(True)["score"], 1)
        ev.paint.sinks["stool"] = dict(solids=np.zeros(1), vehicle=0.0, area=1e-4)
        self.assertFalse(ev.measure(True)["checks"]["protected"]["ok"])
        self.assertLess(ev.measure(True)["score"], 0.9)

    def test_failure_factor_matches_radiator_contract(self):
        from painting.scoring import FAILED
        from painting.task import BEHIND_RADIATOR
        self.assertEqual(FAILED, BEHIND_RADIATOR["fail_scale"])

    def test_wrong_palette_colour_cannot_succeed(self):
        from painting.paint import make_paint
        for task, wrong in (("swatch-card", "cadmium_red"), ("brush-stroke", "ultramarine")):
            ev, ch, mask, _ = evaluator(task)
            ch.paints = [make_paint("acrylic", wrong)]
            ch.fill(50, mask)
            out = ev.measure(True)
            self.assertFalse(out["succeeded"])
            self.assertEqual(out["score"], 0)
            old, c, m, _ = evaluator(task, new=False)
            c.paints = ch.paints
            c.fill(50, m)
            self.assertTrue(old.measure(True)["succeeded"])

    def test_checkpoint_restores_drop_clock_and_invalidates_cached_fields(self):
        ev, _, _, _ = evaluator("load-roller")
        ev._last_drip_tick, ev._drops0 = 100, 3
        saved = ev.checkpoint_state()
        ev._last_drip_tick, ev._drops0 = 50000, 9
        ev._verdict_cache = ev._hide_cache = ev._prot_cache = "stale"
        ev._task_progress.declaration["print"] = "stale"
        ev._decl_h["print"] = 500e-6
        ev.restore_state(saved)
        self.assertEqual((ev._last_drip_tick, ev._drops0), (100, 3))
        self.assertNotIn("_verdict_cache", ev.__dict__)
        self.assertNotIn("_hide_cache", ev.__dict__)
        self.assertNotIn("_prot_cache", ev.__dict__)
        self.assertEqual(ev._task_progress.declaration, {})
        self.assertEqual(ev._decl_h, {})
        self.assertIsNone(ev.env._eval_key)
        ev.restore_state(dict(drops0=0, last_drip_tick=None))
        self.assertFalse(hasattr(ev, "_last_drip_tick"))

    def test_verification_lands_pending_wall_drops_and_restores_stub_state(self):
        for new in (False, True):
            ev, ch, _, _ = evaluator("cut-in-trim", new=new)
            ch.fill(100)
            p = ev.paint
            p.drips.pos = ["pending"]
            calls = []
            def land(bq, dt):
                calls.append(dt)
                p.sinks["stool"] = dict(solids=np.array([1e-8]), vehicle=0.0, area=1e-4)
                p.drips.pos.clear()
            p.drips.step = land  # Stub changes a record; no physical integration.
            p.snapshot = lambda: copy.deepcopy((p.sinks, p.drips.pos))
            def restore(snap):
                p.sinks, p.drips.pos = copy.deepcopy(snap)
            p.restore = restore
            before = ch.snapshot()
            # Zero settle time, on stubs with no advance()/dry_only() methods.
            out = ev.evaluate(max_s=0)
            self.assertEqual(out["succeeded"], not new)
            self.assertEqual(bool(calls), new)
            self.assertEqual(p.sinks, {})
            self.assertEqual(p.drips.pos, ["pending"])
            np.testing.assert_array_equal(ch.d, before["d"])
            if new:
                self.assertFalse(out["checks"]["protected"]["ok"])
                self.assertIn("drops_landed_s", out["settle"])
                # Repeated read reuses that same verdict rather than losing the
                # pending-drop result after the live fields were restored.
                self.assertTrue(ev.evaluate(max_s=0)["cached"])

    def test_same_final_field_same_score_regardless_of_route(self):
        for task in ("swatch-card", "brush-stroke", "cut-in-trim", "clear-coat", "spray-stencil"):
            a, ca, mask, _ = evaluator(task)
            b, cb, _, _ = evaluator(task)
            ca.fill(10, mask)
            cb.fill(100, mask)
            b.measure(False)
            cb.fill(10, mask)
            b.env.world.arms = ["left"]
            b.env.world.tick = 100000
            self.assertEqual(a.measure(True)["score"], b.measure(True)["score"])


class BackendWiring(unittest.TestCase):
    @staticmethod
    def backend():
        path = ROOT/"gym_anything_robotics/backends/newton_painting.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NewtonPaintingBackend")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_variation", "select_variation", "reset"}]
        stub = ast.ClassDef(name="Backend", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
        namespace = {"UnsupportedActionError": ValueError}
        exec(compile(module, str(path), "exec"), namespace)
        b = namespace["Backend"]()
        b.config = NS(preset="wall", task="roll-strip", seed=0, contract=None)
        b.env = NS(reset=lambda: {}, get_state=lambda: {})
        def rebuild(v):
            b.config = NS(**dict(v, contract=v.get("contract")))
        b._rebuild = rebuild
        return b

    def test_select_reset_and_switch_do_not_leak_contract(self):
        from painting.catalogue import CONTRACT
        b = self.backend()
        preview = b.select_variation({"contract": CONTRACT})
        self.assertEqual(b._variation(), dict(preset="wall", task="roll-strip", seed=0))
        b.reset(**preview["reset_request"])
        self.assertEqual(b._variation()["contract"], CONTRACT)
        b.reset(seed=3)
        self.assertEqual(b._variation()["contract"], CONTRACT)
        b.reset(options={"task": "behind-radiator"})
        self.assertNotIn("contract", b._variation())
        b.reset(options={"task": "roll-strip", "contract": CONTRACT})
        preview = b.select_variation({"contract": "checks-v1"})
        b.reset(**preview["reset_request"])
        self.assertNotIn("contract", b._variation())
        before = b._variation()
        with self.assertRaises(ValueError):
            b.reset(options={"contract": "invalid"})
        self.assertEqual(b._variation(), before)


if __name__ == "__main__":
    unittest.main()

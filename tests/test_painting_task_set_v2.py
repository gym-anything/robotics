"""painting-tasks-v2, the render-checked eight-task set: constructed film/pose tests only. No world, simulator, server,
or film advancement.

Run: python -m pytest tests/test_painting_task_set_v2.py
Ported from the fix branch's painting/tests/test_tasks8.py to the v2 contract. classroom_arrangement's two imported
constants and configuration names are stubbed. The v2 session wiring (painting/task_set_v2.py) runs over a stub backend;
the released backend's dispatch is checked in its source.
"""
import ast
import copy
import importlib.abc
import json
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES
BACKEND = REPO/"src/gym_anything_robotics/backends/newton_painting.py"
V2 = "painting-tasks-v2"


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


def evaluator(task, new=True, prepaint=False, contract=None):
    """new: the v2 verifier (behind-radiator: its own audited one); otherwise the evaluator a session builds with
    `contract` (None: the legacy one, painting-tasks-v1: v1's)."""
    from painting.layout import TASKS
    from painting.task import Evaluator
    from painting.task_set_v2_verifier import TaskSetV2Evaluator
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
             config=NS(contract=contract))
    cls = TaskSetV2Evaluator if new and task != "behind-radiator" else Evaluator
    with patch.object(Evaluator, "_targets", return_value={} if task == "load-roller" else targets):
        ev = cls(env)
    return ev, ch, m, pose


class Catalogue(unittest.TestCase):
    def test_exact_eight_in_benchmark_json(self):
        from painting import task_set_v2
        from painting.catalogue import EIGHT_TASKS
        from painting.variants import task_resolve
        raw = json.loads((ROOT/"painting/benchmark.json").read_text())
        self.assertEqual(task_set_v2.CONTRACT, V2)
        self.assertIn(V2, raw["selector_choices"]["contract"])
        rows = raw["task_set"]["selections"]
        self.assertEqual([r["task"] for r in rows], list(EIGHT_TASKS))
        # the seven select v2; the audited hardest task keeps its released split selection exactly
        test_split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        self.assertEqual(rows[-1], next(t["selection"] for t in test_split if t["environment_id"] == "painting"))
        self.assertEqual([dict(r, seed=r.get("seed", 0)) for r in rows], task_set_v2.selections())
        ids = set()
        for row in rows[:-1]:
            resolved = task_resolve(row)
            self.assertEqual(resolved["task_id"], f"painting/v1/{row['preset']}/{row['task']}/{V2}/seed-0")
            self.assertEqual(resolved["backend_options"], row)
            self.assertEqual(resolved["selection"], row)
            ids.add(resolved["task_id"])
        self.assertEqual(task_resolve(rows[-1])["task_id"], "painting/v1/wall/behind-radiator/seed-0")
        self.assertEqual(len(ids), 7)
        self.assertEqual(task_resolve({"contract": V2})["task_id"], f"painting/v1/wall/roll-strip/{V2}/seed-0")

    def test_released_selections_are_unchanged(self):
        from painting import variants
        from painting.catalogue import CONTRACT as V1, selections
        raw = json.loads((ROOT/"painting/benchmark.json").read_text())
        self.assertEqual([r["selection"] for r in raw["task_set_v1"]["tasks"]], selections())
        for row in raw["task_set_v1"]["tasks"]:
            self.assertEqual(variants.task_resolve(row["selection"])["task_id"], row["task_id"])
        self.assertEqual(variants.task_resolve({})["task_id"], raw["default_task_id"])
        self.assertEqual(variants.task_resolve({"contract": V1})["task_id"], f"painting/v1/wall/roll-strip/{V1}/seed-0")
        self.assertNotIn(V2, json.dumps(variants.contract()))
        self.assertEqual(raw["resolver"], "painting.variants:task_resolve")

    def test_contract_validation(self):
        from painting import task_set_v2
        from painting.config import Config
        from painting.variants import task_resolve
        for selection in ({"task": "behind-radiator", "contract": V2},
                          {"task": "touch-up", "contract": V2}, {"task": "fill-circle", "contract": V2},
                          {"contract": V2, "seed": True}, {"contract": V2, "seed": 1.5}, {"contract": V2, "seed": "1"},
                          {"contract": V2, "seed": -1}, {"contract": V2, "seed": 10000},
                          {"task": "clear-coat", "preset": "craft", "contract": V2}, {"unknown": 1, "contract": V2}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                task_resolve(selection)
        with self.assertRaisesRegex(ValueError, r"for behind-radiator; choose progress-v1$"):
            task_set_v2.task_resolve({"task": "behind-radiator", "contract": V2})
        # the launch configuration never carries v2: the backend builds the world without a contract, then adopts it
        with self.assertRaises(ValueError):
            Config(task="roll-strip", contract=V2)

    def test_instructions_are_v1s_and_starts_unchanged(self):
        from painting import task_set_v2
        from painting.catalogue import CONTRACT as V1, EIGHT_TASKS, describe
        from painting.variants import task_resolve
        for task in EIGHT_TASKS[:-1]:
            for seed in (0, 1, 2, 3, 4, 9999):
                text = task_set_v2.describe(task, seed)
                self.assertEqual(text, describe(task, seed, V1))
                for verifier_word in ("50 g", "10 s", "95 %", "98 %", "25-50", "1 mm", "score"):
                    self.assertNotIn(verifier_word, text)
            v1 = task_resolve({"task": task, "contract": V1})
            v2 = task_resolve({"task": task, "contract": V2})
            self.assertEqual(v2["description"], v1["description"])
            self.assertEqual(v2["split"], v1["split"])
            self.assertEqual(v2["backend_options"], dict(v1["backend_options"], contract=V2))
        self.assertIn("second, finish coat", task_set_v2.describe("roll-strip", 0))


class Rewards(unittest.TestCase):
    def test_all_eight_untouched_zero_including_prepaint(self):
        from painting.catalogue import EIGHT_TASKS, RADIATOR
        for task in EIGHT_TASKS:
            ev, _, _, _ = evaluator(task, prepaint=task == "roll-strip")
            for settled in (False, True):
                out = ev.measure(settled)
                self.assertFalse(out["succeeded"], task)
                self.assertAlmostEqual(out["score"], 0.0, places=12, msg=task)
                self.assertEqual(out["reward"]["contract"], RADIATOR if task == "behind-radiator" else V2)
                self.assertEqual(out["contract"], RADIATOR if task == "behind-radiator" else V2)

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
            ev._progress_v2.capture_declaration()
            ev._decl_h = {ch.name: 500e-6}
            ch.w[:] = 0
            self.assertEqual(ev.measure(True)["score"], 0)
            self.assertFalse(ev.measure(True)["checks"]["no_runs"]["ok"])
            ev._progress_v2.declaration.clear()
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

    def test_audited_success_and_legacy_scores(self):
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

    def test_untouched_runs_and_release_keep_legacy_predicates(self):
        # The spec keeps these success checks unchanged, so v2 agrees with the legacy verdict on each: film once
        # touched counts even if repaired, settle travel beyond sag_move is a run, and any closed grip command vetoes
        # the roller's release. painting-tasks-v1 passes all three (it reads marred, flowed and finger contacts).
        for new in (False, True):
            ev, ch, mask, pose = evaluator("roll-strip", new=new)
            ch.fill(100)
            pose[0, 2] = 0.05
            world = ev.env.world
            world.grip_force["right"] = 0.0
            self.assertTrue(ev.measure(True)["succeeded"])
            ch.touched[:10] = 1
            out = ev.measure(True)
            self.assertFalse(out["checks"]["untouched"]["ok"])
            self.assertFalse(out["succeeded"])
            if new:
                self.assertLess(out["score"], 0.9)
                self.assertNotIn("basis", out["checks"]["untouched"])
            ch.touched[:] = 0
            ch.moved[mask] = 0.002
            out = ev.measure(True)
            self.assertFalse(out["checks"]["no_runs"]["ok"])
            self.assertFalse(out["succeeded"])
            ch.moved[:] = 0
            world.grip_force["right"] = 1.0
            out = ev.measure(True)
            self.assertFalse(out["checks"]["roller_in_tray"]["ok"])
            self.assertFalse(out["succeeded"])
            world.grip_force["right"] = 0.0
            world.scene.robot_bodies = [2]
            world.model.body_label = ["roller", "tray", "left_finger"]
            world.contact_table = lambda: [(2, 0)]
            self.assertFalse(ev.measure(True)["checks"]["roller_in_tray"]["ok"])
            world.contact_table = lambda: []
            self.assertTrue(ev.measure(True)["succeeded"])
        ev, ch, mask, _ = evaluator("cut-in-trim")
        ch.fill(100)
        self.assertEqual(ev.measure(True)["score"], 1)
        ev.paint.sinks["stool"] = dict(solids=np.zeros(1), vehicle=0.0, area=1e-4)
        self.assertFalse(ev.measure(True)["checks"]["protected"]["ok"])
        self.assertLess(ev.measure(True)["score"], 0.9)

    def test_v1_keeps_its_released_looser_checks(self):
        from painting.catalogue import CONTRACT as V1
        ev, ch, mask, pose = evaluator("roll-strip", new=False, contract=V1)
        ch.fill(100)
        pose[0, 2] = 0.05
        ch.touched[mask] = 1
        ch.moved[mask] = 0.002
        out = ev.measure(True)
        self.assertTrue(out["succeeded"])                  # v1: repaired marks, settle travel and a grip command pass
        self.assertEqual(out["contract"], V1)
        self.assertEqual(out["reward"]["contract"], V1)

    def test_progress_counts_touched_cells(self):
        ev, ch, mask, _ = evaluator("cut-in-trim")
        ch.fill(100)
        ch.touched[mask] = 1                                # repaired: no marred cells left
        out = ev.measure(True)
        self.assertGreater(out["reward"]["defects_cm2"]["marks"], 0)
        self.assertLess(out["score"], 0.9)

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
        ev._progress_v2.declaration["print"] = "stale"
        ev._decl_h["print"] = 500e-6
        ev.restore_state(saved)
        self.assertEqual((ev._last_drip_tick, ev._drops0), (100, 3))
        self.assertNotIn("_verdict_cache", ev.__dict__)
        self.assertNotIn("_hide_cache", ev.__dict__)
        self.assertNotIn("_prot_cache", ev.__dict__)
        self.assertEqual(ev._progress_v2.declaration, {})
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

    def test_v2_is_built_on_the_legacy_evaluator_only(self):
        from painting.catalogue import CONTRACT as V1
        with self.assertRaises(ValueError):
            evaluator("roll-strip", contract=V1)


class StubBackend:
    """What the v2 session wiring needs of the backend: its config, environment, rebuild (the released _rebuild, which
    builds the world from config.contract) and notes."""
    def __init__(self, task="roll-strip", contract=V2):
        self._task_set_v2 = True
        self.config = NS(preset="wall", task=task, seed=0, contract=None)
        self.env = NS(reset=lambda: {}, get_state=lambda: {})
        self.capabilities = NS(notes={})
        self.rebuilt = []
        if contract == V2:
            self.env.contract = V2

    def _rebuild(self, v):
        self.rebuilt.append(dict(v))
        self.config = NS(preset=v["preset"], task=v["task"], seed=v["seed"], contract=v.get("contract"))
        self.env = NS(reset=lambda: {}, get_state=lambda: {})

    def _notes(self):
        return {"task": self.env.spec["instruction"] if hasattr(self.env, "spec") else None}

    def _variation(self):
        from painting.task_set_v2 import session_variation
        return session_variation(self)


def _adopt_stub(env):
    env.contract = V2


class SessionWiring(unittest.TestCase):
    def setUp(self):
        import dataclasses
        from painting import task_set_v2
        patcher = patch.object(task_set_v2, "adopt", _adopt_stub)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(dataclasses, "replace", lambda obj, **kw: NS(**dict(vars(obj), **kw)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_select_reset_and_switch_keep_or_drop_the_contract(self):
        from painting.task_set_v2 import session_reset, session_select, session_contract
        from painting.catalogue import CONTRACT as V1
        b = StubBackend()
        self.assertEqual(b._variation(), dict(preset="wall", task="roll-strip", seed=0, contract=V2))
        preview = session_select(b, {"seed": 3})
        self.assertEqual(preview["reset_request"], dict(seed=3, options=dict(preset="wall", task="roll-strip", contract=V2)))
        session_reset(b, **preview["reset_request"])
        self.assertEqual(b.rebuilt[-1], dict(preset="wall", task="roll-strip", seed=3))     # rebuilt without a contract
        self.assertEqual(b._variation()["contract"], V2)                                   # then adopted
        session_reset(b, options={"task": "behind-radiator"})                             # another task: its default
        self.assertNotIn("contract", b._variation())
        session_reset(b, options={"task": "cut-in-trim", "contract": V2})
        self.assertEqual(b._variation(), dict(preset="wall", task="cut-in-trim", seed=3, contract=V2))
        session_reset(b, options={"contract": V1})                                         # v1 as the released reset
        self.assertEqual(b.rebuilt[-1], dict(preset="wall", task="cut-in-trim", seed=3, contract=V1))
        self.assertEqual(b._variation()["contract"], V1)
        preview = session_select(b, {"contract": "checks-v1"})
        session_reset(b, **preview["reset_request"])
        self.assertNotIn("contract", b._variation())
        before = b._variation()
        from gym_anything_robotics.errors import UnsupportedActionError
        with self.assertRaises(UnsupportedActionError):
            session_reset(b, options={"contract": "invalid"})
        with self.assertRaises(UnsupportedActionError):
            session_reset(b, options={"task": "touch-up", "contract": V2})
        with self.assertRaisesRegex(UnsupportedActionError, r"^Unknown reset options \['x'\]; selectors are preset, "
                                                            r"task, seed, contract; nothing changed$"):
            session_reset(b, options={"x": 1})
        self.assertEqual(b._variation(), before)
        n = len(b.rebuilt)
        session_reset(b)                                                                   # plain reset: no rebuild
        self.assertEqual(len(b.rebuilt), n)
        c = session_contract(b)
        self.assertEqual(c["contract_choices"]["roll-strip"], ["checks-v1", V1, V2])
        self.assertEqual(c["contract_choices"]["behind-radiator"], ["progress-v1"])
        self.assertEqual(c["current"], before)

    def test_the_backend_dispatches_only_a_v2_launch(self):
        src = BACKEND.read_text()
        init = src[src.index("    def __init__"):src.index("    def _notes")]
        self.assertIn('task_set_v2 = contract == "painting-tasks-v2"', init)
        self.assertIn("v = resolve(preset, task, int(seed), None if task_set_v2 else contract)", init)
        self.assertIn("adopt(self.env)", init)
        tree = ast.parse(src)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NewtonPaintingBackend")
        for name, target in (("reset", "session_reset"), ("_variation", "session_variation"),
                             ("variation_contract", "session_contract"), ("select_variation", "session_select")):
            fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
            first = fn.body[0]
            self.assertIsInstance(first, ast.If, name)
            self.assertEqual(ast.unparse(first.test), "getattr(self, '_task_set_v2', False)", name)
            self.assertIn(target, ast.unparse(first), name)
        from painting.task_set_v2 import CONTRACT
        self.assertEqual(CONTRACT, V2)
        variants = (ROOT/"painting/variants.py").read_text()
        self.assertIn('if selection.get("contract") == "painting-tasks-v2":', variants)


class Adoption(unittest.TestCase):
    def test_adopt_changes_instruction_evaluator_and_start_checkpoint_only(self):
        from painting.task import Evaluator
        from painting.task_set_v2 import adopt, describe
        from painting.task_set_v2_verifier import TaskSetV2Evaluator
        from painting.layout import setup
        legacy, _, _, _ = evaluator("load-roller", new=False)

        class FakeEnv:                     # a constructed PaintingEnv's surface and its checkpoint hooks, no simulator
            def checkpoint(self):
                return dict(tick=self.world.tick, task_evaluation=self.evaluator.checkpoint_state())

            def restore(self, snap):
                self.world.tick = snap["tick"]
                self.evaluator.restore_state(snap.get("task_evaluation"))

        env = FakeEnv()
        env.lock, env.world, env.paint, env.config = threading.RLock(), legacy.env.world, legacy.paint, NS(contract=None)
        env.spec = setup("load-roller", 0)
        before = copy.deepcopy(env.spec)
        env.evaluator = legacy
        env.initial = env.checkpoint()
        self.assertIsNone(env.initial["task_evaluation"])
        with patch.object(Evaluator, "_targets", return_value={}):
            adopt(env)
        self.assertIsInstance(env.evaluator, TaskSetV2Evaluator)
        self.assertEqual(env.contract, V2)
        self.assertEqual(env.spec["instruction"], describe("load-roller", 0))
        self.assertEqual({k: v for k, v in env.spec.items() if k != "instruction"},
                         {k: v for k, v in before.items() if k != "instruction"})
        self.assertEqual(env.initial["task_evaluation"], dict(drops0=0, last_drip_tick=None))
        env.evaluator._last_drip_tick, env.evaluator._drops0 = 100, 3
        saved = env.checkpoint()
        env.evaluator._last_drip_tick, env.world.tick = 900, 1000
        env.restore(saved)
        self.assertEqual((env.evaluator._last_drip_tick, env.evaluator._drops0, env.world.tick), (100, 3, 0))
        with self.assertRaises(ValueError):
            adopt(env)

    def test_painting_env_checkpoints_carry_the_evaluator_hooks(self):
        # the released PaintingEnv already saves and restores the evaluator's task state (None for a legacy one), so
        # adopting needs no change to env.py
        src = (ROOT/"painting/env.py").read_text()
        self.assertIn("task_evaluation=self.evaluator.checkpoint_state(),", src)
        self.assertIn('self.evaluator.restore_state(snap.get("task_evaluation"))', src)
        self.assertIn("self.restore(self.initial)", src)


if __name__ == "__main__":
    unittest.main()

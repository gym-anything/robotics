"""CPU-only constructed states. No world, Warp, Newton, MuJoCo or server imports.

Run: python -m pytest tests/environments/soil_excavation/test_task_set.py
"""
import importlib
import json
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from gym_anything_robotics.errors import UnsupportedActionError

from soil_excavation import catalogue as C, layout as L, materials as M, variants
from tests.packages import PACKAGES

# task.py imports only the two origin constants from the simulator module.
# Scope the substitute so no actual runtime import occurs or leaks to other tests.
soil_stub = ModuleType("soil_excavation.soil")
soil_stub.ORIGIN_BLOCK, soil_stub.ORIGIN_PILE = 0, 1
with patch.dict(sys.modules, {"soil_excavation.soil": soil_stub}):
    from soil_excavation import task as legacy
    from soil_excavation import task_set as E

ROOT = PACKAGES


class StateLedger(legacy.Ledger):
    """Hand-authored surfaces and particles, never a simulated state."""
    def __init__(self, spec):
        self.spec, self.block = spec, spec["block"]
        self.h, self.voxel = .005, .02
        self.depth, self.radius = 0., .20
        self.set_particles([[0., .2, L.GRADE - .1]], mass=[1.5])

    def set_particles(self, pos, mass=None):
        self.pos = np.asarray(pos, float).reshape(-1, 3)
        self.pos0 = self.pos.copy()
        n = len(self.pos)
        self.mass = np.ones(n) if mass is None else np.asarray(mass, float)
        self.origin = np.zeros(n, int)
        self.mat = np.full(n, M.NAMES.index("topsoil"), int)
        self.jp = np.full(n, M.jp0("topsoil"))
        self.loosened = np.zeros(n, bool)
        self.displaced_loose = np.zeros(n, bool)

    def transport(self, count=None):
        count = len(self.pos) if count is None else count
        (x0, y0), (x1, y1) = self.spec["tarp"]
        self.pos[:count] = ((x0 + x1)/2, (y0 + y1)/2, .01)

    def heightfield(self, rect, cell=.01, **kw):
        (x0, y0), (x1, y1) = rect
        x = x0 + (np.arange(round((x1 - x0)/cell)) + .5)*cell
        y = y0 + (np.arange(round((y1 - y0)/cell)) + .5)*cell
        X, Y = np.meshgrid(x, y, indexing="ij")
        if "target_line" in self.spec:
            mask = np.abs(Y - self.spec["target_line"][0][1]) < .06
        else:
            marks = self.spec.get("targets", [self.spec.get("target", (0., .5))])
            radius = .04 if "targets" in self.spec else self.radius
            mask = np.zeros(X.shape, bool)
            for mx, my in marks:
                mask |= np.hypot(X - mx, Y - my) < radius
        return L.GRADE - self.depth*mask, (x0, y0, cell)


def fixture(task, contract=C.CONTRACT):
    spec = L.setup(task, 0)
    ledger = StateLedger(spec)
    if task == "break-hardpan":
        x, y = np.meshgrid(np.arange(-.19, .20, .02), np.arange(-1.84, -1.45, .02), indexing="ij")
        ledger.set_particles(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, L.GRADE - .081))),
                              mass=np.full(x.size, .02))
    elif task == "expose-line":
        x = np.arange(-.65, .65, .005) + .0025
        ledger.set_particles(np.column_stack((x, np.full(len(x), .48), np.full(len(x), L.GRADE - .00275))),
                              mass=np.full(len(x), .05))
    positions = {"spade": np.array([*spec.get("target", (0., .5)), L.GRADE + .03]),
                 "auger": np.array([0., .2, L.GRADE + .2])}
    rotations = {name: np.eye(3) for name in positions}
    part = NS(kind="cylinder", size=(.02, .05), pos=(0., 0., 0.), quat=(0., 0., 0., 1.))
    world = NS(tick=0, sim_time=0., fault=None, strikes=[], reflexes=[], recorders=[],
               scene=NS(bodies={name: name for name in positions}, parts={"auger": [part]}),
               body_pose=lambda name: (positions[name], rotations[name]))
    env = NS(spec=spec, world=world, lock=threading.RLock())
    env.evaluator = legacy.Evaluator(env)
    with patch.object(E, "Ledger", return_value=ledger), patch.object(E, "tool_held", return_value={"left": True, "right": True}), \
         patch.object(legacy, "tool_held", return_value={"left": True, "right": True}):
        E.install(env, contract)
    return env, ledger, positions, rotations


class CatalogueTests(unittest.TestCase):
    def test_all_eight_resolve_and_legacy_ids_meanings_are_kept(self):
        self.assertEqual(len(C.EIGHT_TASKS), 8)
        self.assertEqual(len(set(C.EIGHT_TASKS)), 8)
        self.assertEqual(len(C.TASKS), 11)
        for task in C.TASKS:
            for selection in (dict(task=task), dict(task=task, seed="7", topsoil="0.2518", stones=0, moisture="wet")):
                self.assertEqual(C.task_resolve(selection), variants.task_resolve(selection))
        for selection in C.TASK_SET:
            result = C.task_resolve(selection)
            t = selection["task"]
            self.assertEqual(result["description"], C.instruction(t, selection.get("contract")))
            self.assertEqual(result["selection"]["seed"], 0)
            if t == "set-post":
                self.assertEqual(result, variants.task_resolve(selection))
            else:
                self.assertIn(C.CONTRACT, result["task_id"])
                self.assertEqual(result["backend_options"]["contract"], C.CONTRACT)
                self.assertNotEqual(result["task_id"], variants.task_resolve(selection)["task_id"])

    def test_coercion_and_validation_unchanged(self):
        for seed in (True, 2.9, "4"):
            self.assertEqual(C.task_resolve({"seed": seed}), variants.task_resolve({"seed": seed}))
        for selection in ({"seed": 10000}, {"seed": None}, {"topsoil": True}, {"stones": True}, {"moisture": "bog"}):
            with self.assertRaises((ValueError, TypeError)):
                C.task_resolve(selection)
        with self.assertRaises(ValueError):
            C.resolve(seed="1")  # direct resolve is stricter than task_resolve, as before
        for task in ("set-post", "plant-and-backfill", "post-hole", "backfill-pipe"):
            with self.assertRaises(ValueError):
                C.task_resolve(dict(task=task, contract=C.CONTRACT))
        self.assertEqual(C.task_resolve({"contract": "v1"}), variants.task_resolve({}))
        with self.assertRaises(ValueError):
            C.task_resolve({"contract": "typo"})

    def test_benchmark_exact_selections(self):
        b = json.loads((ROOT/"soil_excavation/benchmark.json").read_text())
        rows = b["task_set"]["tasks"]
        self.assertEqual([r["selection"] for r in rows], list(C.TASK_SET))
        for row in rows:
            self.assertEqual(row["task_id"], C.task_resolve(row["selection"])["task_id"])
            self.assertEqual(row["task_contract"], C.selected_contract(row["selection"]["task"], row["selection"].get("contract")))
        self.assertEqual(b["default_task_id"], variants.task_resolve({})["task_id"])
        self.assertEqual(b["count"], 11)


    def test_existing_targets_remain_inside_soil_and_base_inside_slab(self):
        import math
        for task in C.EIGHT_TASKS:
            for seed in (0, 7, 999):
                s = L.setup(task, seed)
                x, y, yaw = s["robot"]
                for dx in (-.38, .38):
                    for dy in (-.275, .275):
                        px, py = x + math.cos(yaw)*dx - math.sin(yaw)*dy, y + math.sin(yaw)*dx + math.cos(yaw)*dy
                        self.assertTrue(L.SLAB["x"][0] <= px <= L.SLAB["x"][1])
                        self.assertTrue(L.SLAB["y"][0] <= py <= L.SLAB["y"][1])
                points = s.get("targets", [s["target"]] if "target" in s else [])
                points += list(s.get("target_line", ())) + list(s.get("target_square", ()))
                if "pipe" in s:
                    points += [(0., s["pipe"]["y"])]
                for px, py in points:
                    self.assertTrue(s["block"]["lo"][0] <= px <= s["block"]["hi"][0])
                    self.assertTrue(s["block"]["lo"][1] <= py <= s["block"]["hi"][1])


class JudgementTests(unittest.TestCase):
    def evaluate(self, fixture, held=True, final=True):
        env, ledger, _, _ = fixture
        with patch.object(E, "Ledger", return_value=ledger), patch.object(E, "tool_held", return_value={"left": held, "right": held}), \
             patch.object(legacy, "tool_held", return_value={"left": held, "right": held}):
            return env.evaluator.evaluate() if final else env.evaluator.progress()

    def test_all_revised_untouched_starts_are_zero_and_layouts_unchanged(self):
        for t in C.REVISED:
            with self.subTest(task=t):
                f = fixture(t)
                ev = self.evaluate(f)
                self.assertEqual(ev["score"], 0.)
                self.assertFalse(ev["succeeded"])
                self.assertEqual(ev["task_contract"], C.CONTRACT)
                expected = L.setup(t, 0)
                expected["instruction"] = C.INSTRUCTIONS[t]
                self.assertEqual(f[0].spec, expected)
                self.assertEqual(len(f[0].world.recorders), 1)

    def test_insertion_continuous_progress_and_geometric_failures(self):
        f = fixture("penetrate-marked")
        scores = []
        for d in (.01, .04, .075, .11):
            f[2]["spade"][2] = L.GRADE - d
            scores.append(self.evaluate(f)["score"])
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))
        f[2]["spade"][2] = L.GRADE - .151
        self.assertEqual(self.evaluate(f)["score"], 1.)
        f[0].world.reflexes = ["recovered stop"]
        self.assertTrue(self.evaluate(f)["succeeded"])
        self.assertFalse(self.evaluate(f, held=False)["succeeded"])
        f[2]["spade"][0] += .04
        self.assertFalse(self.evaluate(f)["succeeded"])
        f[2]["spade"][0] -= .04
        f[3]["spade"] = Rotation.from_euler("x", 12, degrees=True).as_matrix()
        self.assertFalse(self.evaluate(f)["succeeded"])
        f[3]["spade"] = np.eye(3)
        f[0].world.fault = "nonfinite world"
        self.assertFalse(self.evaluate(f)["succeeded"])

    def test_dump_can_finish_with_spade_released_and_spill_costs_credit(self):
        f = fixture("spit-to-tarp")
        f[1].transport()
        self.assertTrue(self.evaluate(f, held=False)["succeeded"])
        f[1].mass[:] = .75
        self.assertEqual(self.evaluate(f, held=False)["score"], .5)
        f[1].set_particles([[0., .2, -.1], [0., .2, -.1]], mass=[1.5, .6])
        f[1].transport(1)
        f[1].pos[1] = [-2., -.5, .01]
        self.assertFalse(self.evaluate(f)["succeeded"])
        self.assertLess(self.evaluate(f)["score"], .99)

    def test_planting_shape_cleanup_and_tarp(self):
        f = fixture("planting-hole")
        f[1].depth = .16
        self.assertFalse(self.evaluate(f)["succeeded"])
        f[1].transport()
        self.assertTrue(self.evaluate(f)["succeeded"])
        for depth in (.1, .21):
            f[1].depth = depth
            self.assertFalse(self.evaluate(f)["succeeded"])
        f[1].depth, f[1].radius = .16, .29
        self.assertFalse(self.evaluate(f)["succeeded"])
        f[1].radius = .20
        good = self.evaluate(f)["score"]
        with patch.object(f[0].evaluator, "_hole", return_value=(.16, .4, .4)):
            ev = self.evaluate(f)
            self.assertFalse(ev["succeeded"])
            self.assertLess(ev["score"], good)

    def test_hardpan_subthreshold_work_and_source_specific_spill(self):
        f = fixture("break-hardpan")
        f[1].pos[:, 0] += .002
        low = self.evaluate(f)
        f[1].pos[:, 0] += .002
        high = self.evaluate(f)
        self.assertGreater(high["score"], low["score"])
        self.assertGreater(low["score"], 0.)
        f[1].pos[:, 0] += .02
        f[1].transport(201)
        self.assertTrue(self.evaluate(f)["succeeded"])
        f[1].pos[201:301] = [-2., -.5, .01]
        spill = self.evaluate(f)
        self.assertFalse(spill["succeeded"])
        self.assertFalse(next(c["ok"] for c in spill["criteria"] if c["name"] == "spill_fraction"))

    def test_trench_shallow_work_and_base_history(self):
        f = fixture("cable-trench")
        f[1].transport()
        f[1].depth = .04
        low = self.evaluate(f)
        f[1].depth = .08
        self.assertGreater(self.evaluate(f)["score"], low["score"])
        f[1].depth = .161
        self.assertTrue(self.evaluate(f)["succeeded"])
        f[0].evaluator.base_off_slab = .001
        self.assertFalse(self.evaluate(f)["succeeded"])
        self.assertEqual(self.evaluate(f)["score"], .99)

    def test_auger_geometry_clearance_release_and_hole_quality(self):
        f = fixture("auger-holes")
        f[1].depth = .10
        partial = self.evaluate(f)
        self.assertGreater(partial["score"], 0.)
        self.assertFalse(partial["succeeded"])
        f[1].depth = .181
        self.assertTrue(self.evaluate(f, held=False)["succeeded"])
        f[2]["auger"][2] = L.GRADE + .02
        blocked = self.evaluate(f)
        self.assertFalse(blocked["succeeded"])
        self.assertLess(blocked["score"], .99)
        # A cylinder's radial extent matters when tilted horizontally.
        part = f[0].world.scene.parts["auger"][0]
        r = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        self.assertAlmostEqual(E.part_lowest_z(part, [0, 0, .1], r), .08)
        part.kind = "capsule"
        self.assertAlmostEqual(E.part_lowest_z(part, [0, 0, .1], np.eye(3)), .03)

    def test_exposure_rewards_approach_but_roof_strikes_and_wrong_spoil_fail(self):
        f = fixture("expose-line")
        self.assertEqual(self.evaluate(f)["criteria"][1]["value"], 0.)
        f[1].pos[:, 2] -= .10
        partial = self.evaluate(f)
        self.assertGreater(partial["score"], 0.)
        self.assertFalse(partial["succeeded"])
        # Soil suspended far above the crown still covers it.
        self.assertEqual(next(c["value"] for c in partial["criteria"] if c["name"] == "crown_exposed"), 0.)
        f[1].pos[:, 2] = f[0].spec["pipe"]["z"] + f[0].spec["pipe"]["r"] + .005
        thin = self.evaluate(f)
        self.assertEqual(next(c["value"] for c in thin["criteria"] if c["name"] == "crown_exposed"), 0.)
        f[1].transport()
        self.assertTrue(self.evaluate(f, held=False)["succeeded"])
        f[0].world.strikes = ["strike"]
        struck = self.evaluate(f)
        self.assertFalse(struck["succeeded"])
        self.assertAlmostEqual(struck["progress"], .5)
        f[0].world.strikes = []
        f[1].pos[:] = [-2., -.5, .01]
        self.assertFalse(self.evaluate(f)["succeeded"])

    def test_state_only_repeated_queries_and_particle_permutations(self):
        f = fixture("expose-line")
        f[1].transport(100)
        first = self.evaluate(f)
        self.assertEqual(first, self.evaluate(f))
        for name in ("pos", "pos0", "mass", "origin", "mat", "jp", "loosened", "displaced_loose"):
            setattr(f[1], name, getattr(f[1], name)[::-1].copy())
        self.assertEqual(first, self.evaluate(f))

    def test_set_post_verifier_and_advisory_delegated_without_rescaling(self):
        env, ledger, _, _ = fixture("set-post", C.POST_CONTRACT)
        original = env.evaluator.legacy
        for p in (0., .3, .8, 1.):
            with patch.object(legacy, "Ledger", return_value=ledger), patch.object(original, "_set_post", return_value=(
                    [legacy.crit("post_support", 1., False, "fixture")], [], dict(score=min(p, .99), fields=dict(progress=p)))):
                result = env.evaluator.evaluate()
                self.assertEqual(result["score"], min(p, .99))
                self.assertEqual(result["progress"], p)
                self.assertEqual(result["task_contract"], C.POST_CONTRACT)
                self.assertEqual({k: v for k, v in result.items() if k != "task_contract"}, original.evaluate())
        with patch.object(original, "progress", return_value=dict(score=.25, succeeded=False, verdict_pending=True)):
            self.assertEqual(env.evaluator.progress()["score"], .25)
        with patch.object(legacy, "Ledger", return_value=ledger), patch.object(original, "_set_post", return_value=(
                [legacy.crit("post_support", 0., True, "fixture")], [], dict(score=.99))):
            self.assertEqual(env.evaluator.evaluate()["score"], 1.)
        with patch.object(original, "evaluate", side_effect=legacy.VerdictUnavailable("fault")):
            with self.assertRaises(legacy.VerdictUnavailable):
                env.evaluator.evaluate()

    def test_baseline_is_normalized_and_failure_never_ties_success(self):
        self.assertEqual(E.net_progress(.2, .2), 0.)
        self.assertAlmostEqual(E.net_progress(.6, .2), .5)
        self.assertEqual(E.net_progress(.1, .2), 0.)
        self.assertEqual(E.net_progress(1., .2), 1.)
        self.assertEqual(E.failed_score(1.), .99)
        self.assertEqual(E.failed_score(.5), .5)
        f = fixture("spit-to-tarp")
        with patch.object(E, "Ledger", side_effect=ValueError("bad state")):
            advisory = f[0].evaluator.progress()
            self.assertEqual(advisory["task_contract"], C.CONTRACT)
            self.assertIn("bad state", advisory["error"])


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The adapter itself is real; the parent stands in for construction of
        # the unavailable runtime. These tests cover the public contract wiring.
        module = ModuleType("gym_anything_robotics.backends.newton_soil_excavation")
        class Parent:
            def __init__(self, **options):
                self.received = options
            def _variation(self):
                return {k: getattr(self.config, k) for k in variants.SELECTORS}
            def _rebuild(self, value):
                self.rebuilt = value
            def save_checkpoint(self, *, label):
                self._checkpoints[label] = {"tick": self.env.world.tick}
                return {"label": label}
            def restore_checkpoint(self, label, **kwargs):
                return self.env.evaluator.base_off_slab
        module.NewtonSoilExcavationBackend = Parent
        name = "gym_anything_robotics.backends.newton_soil_excavation_tasks"
        with patch.dict(sys.modules, {module.__name__: module}):
            cls.module = importlib.import_module(name)
        cls.Backend = cls.module.NewtonSoilExcavationTasksBackend

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.module.__name__, None)

    def backend(self, task="penetrate-marked", contract=C.CONTRACT):
        b = self.Backend.__new__(self.Backend)
        b._task_contract = contract
        b.config = NS(**variants.resolve(task), robot="mobile-fr3-duo")
        b.env = NS(lock=threading.RLock(), evaluator=NS(base_off_slab=.1), world=NS(tick=100),
                   reset=lambda: {"restored": True})
        b._checkpoints = {}
        return b

    def test_selection_preserves_coercion_and_bare_task_meaning(self):
        b = self.backend()
        r = b.select_variation(dict(seed="3", topsoil=".2506"))
        self.assertEqual(r["variation"]["seed"], 3)
        self.assertEqual(r["variation"]["topsoil"], .251)
        self.assertEqual(r["task"], C.INSTRUCTIONS["penetrate-marked"])
        r = b.select_variation(dict(task="set-post"))
        self.assertNotIn("contract", r["variation"])
        self.assertEqual(r["task"], L.TASKS["set-post"]["instruction"])
        with self.assertRaises(ValueError):
            b.select_variation(dict(task="set-post", contract=C.CONTRACT))

    def test_registry_constructor_and_rebuild_strip_only_contract(self):
        from gym_anything_robotics import registry
        options = dict(task="auger-holes", seed="3", clutch=None, topsoil="0.254", recording=False)
        with patch.object(self.Backend, "_install_contract") as install, \
             patch.dict(sys.modules, {self.module.__name__: self.module}):
            b = registry._newton_soil_excavation_factory(contract=C.CONTRACT, **options)
            self.assertEqual(b.received, options)
            self.assertEqual(b._task_contract, C.CONTRACT)
            value = C.resolve("auger-holes", 3, contract=C.CONTRACT)
            b._rebuild(value)
            self.assertEqual(b.rebuilt, {k: value[k] for k in variants.SELECTORS})
            self.assertEqual(install.call_count, 2)

    def test_reset_and_checkpoints_restore_slab_history(self):
        b = self.backend("cable-trench")
        b.save_checkpoint(label="a")
        b.env.evaluator.base_off_slab = .3
        self.assertEqual(b.restore_checkpoint("a"), .1)
        state, _ = b.reset()
        self.assertTrue(state["restored"])
        self.assertEqual(b.env.evaluator.base_off_slab, 0.)
        with self.assertRaises(KeyError):
            b.restore_checkpoint("missing")

    def test_bad_reset_is_atomic_and_contract_is_not_passed_to_physics(self):
        b = self.backend()
        before = dict(b.config.__dict__)
        for options in ({"contract": "unknown"}, {"seed": "bad"}, {"no_such_option": 1}):
            with self.assertRaises(UnsupportedActionError):
                b.reset(options=options)
        self.assertEqual(b.config.__dict__, before)
        rebuilt = []
        with patch.object(b, "_rebuild", side_effect=lambda v: rebuilt.append(v)), \
             patch.dict(sys.modules, {"soil_excavation.robot_setup": NS(check_task=lambda *args: None)}):
            b.env.get_state = lambda: {}
            b.reset(options=dict(contract="v1"))
        self.assertNotIn("contract", rebuilt[0])


if __name__ == "__main__":
    unittest.main()

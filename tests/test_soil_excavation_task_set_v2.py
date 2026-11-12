"""soil-tasks-v2: CPU-only constructed states. No world, Warp, Newton, MuJoCo or server imports.

Run: python -m pytest tests/test_soil_excavation_task_set_v2.py
"""
import importlib
import json
import math
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
from gym_anything_robotics import tasks
from gym_anything_robotics.errors import UnsupportedActionError, UnsupportedConfigurationError

from soil_excavation import catalogue as C1, catalogue_v2 as C, layout as L, materials as M, variants
from tests.packages import PACKAGES, REPO



def isolated_import(stubs, *names):
    """Import names with stand-in modules, then leave sys.modules and the parent packages exactly as found, so
    other test modules (tests/environments/soil_excavation) import their own copies in any order."""
    parents = {name.rpartition(".")[0] for name in names}
    found = {parent: dict(vars(importlib.import_module(parent))) for parent in parents}
    with patch.dict(sys.modules, stubs):
        modules = [importlib.import_module(name) for name in names]
    for parent, attrs in found.items():
        package = sys.modules[parent]
        for name in set(vars(package)) - set(attrs):
            delattr(package, name)
        vars(package).update(attrs)
    return modules


# task.py imports only the two origin constants from the simulator module.
# Scope the substitute so no actual runtime import occurs or leaks to other tests.
soil_stub = ModuleType("soil_excavation.soil")
soil_stub.ORIGIN_BLOCK, soil_stub.ORIGIN_PILE = 0, 1
legacy, E = isolated_import({"soil_excavation.soil": soil_stub}, "soil_excavation.task", "soil_excavation.task_set_v2")
STUBBED = {"soil_excavation.soil": soil_stub, "soil_excavation.task": legacy, "soil_excavation.task_set_v2": E}
# The task folders' verifiers (tasks/<task>/verifier.py, layout.task_verifier) import these modules by name: load
# each against the copies, and while this module's tests run, the copies find those folders through their layout.
with patch.dict(sys.modules, STUBBED):
    for name in [name for name in sys.modules if name.startswith(L.PACKAGE + ".tasks.")]:
        del sys.modules[name]                    # put back with the rest of sys.modules
    FOLDERS = {task: L.task_verifier(task) for task in C.EIGHT_TASKS}
LAYOUT = ModuleType(L.__name__)
vars(LAYOUT).update(vars(L), task_verifier=FOLDERS.get)
COPIES_LAYOUT = [patch.object(legacy, "L", LAYOUT), patch.object(E, "L", LAYOUT)]


def setUpModule():
    for p in COPIES_LAYOUT:
        p.start()


def tearDownModule():
    for p in COPIES_LAYOUT:
        p.stop()

ROOT = PACKAGES
V2 = C.CONTRACT


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


def fixture(task, contract=V2):
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
    # The held mattock's pick as its only soil shape (a vertical capsule, 6.9 cm below its body origin), held 20 cm
    # above grade over the square; the soil arrays are the ledger's own.
    q = np.array([[0., -1.65, L.GRADE + .2, 0., 0., 0., 1.]])
    pick = dict(kind="capsule", size=(.009, .06), pos=(0., 0., 0.), quat=(0., 0., 0., 1.))
    world = NS(tick=0, sim_time=0., fault=None, strikes=[], reflexes=[], recorders=[], soil_ticks=4, tool_q=q,
               scene=NS(bodies={name: name for name in positions}, parts={"auger": [part]},
                        items={"mattock": dict(kind="mattock"), "stone_0": dict(kind="stone")}),
               body_pose=lambda name: (positions[name], rotations[name]), base_pose=lambda: spec["robot"],
               _couplers=lambda: ([dict(name="mattock", shapes=[pick]), dict(name="stone_0", shapes=[pick])],
                                  np.array([0, 0])),
               state=NS(body_q=NS(numpy=lambda: q)),
               soil=NS(state=NS(particle_q=NS(numpy=lambda: ledger.pos), mpm=NS(particle_Jp=NS(numpy=lambda: ledger.jp)))))
    env = NS(spec=spec, world=world, lock=threading.RLock())
    env.evaluator = legacy.Evaluator(env)
    folder = FOLDERS[task]
    with patch.object(E, "Ledger", return_value=ledger), \
         patch.object(folder, "Ledger", return_value=ledger, create=True), \
         patch.object(folder, "tool_held", return_value={"left": True, "right": True}, create=True):
        E.install(env, contract)
    return env, ledger, positions, rotations


def soil_steps(env, tool_positions):
    """Run the world's recorders once per soil step with the mattock at each position in turn."""
    w = env.world
    for p in tool_positions:
        w.tool_q[0, :3] = p
        w.tick += w.soil_ticks
        for record in list(w.recorders):
            record(w)


def sweep(z, rows=(-1.80, -1.68, -1.56, -1.47)):
    """The pick raked across the square row by row, 1 cm per soil step (5 m/s), its body origin at height z."""
    return [(x, y, z) for y in rows for x in np.arange(-.21, .215, .01)]


class CatalogueTests(unittest.TestCase):
    def test_all_eight_resolve_under_v2_and_released_meanings_are_kept(self):
        self.assertEqual(C.EIGHT_TASKS, C1.EIGHT_TASKS)
        self.assertEqual(len(C.TASKS), 11)
        for task in C.TASKS:
            for selection in (dict(task=task), dict(task=task, seed="7", topsoil="0.2518", stones=0, moisture="wet")):
                self.assertEqual(C1.task_resolve(selection), variants.task_resolve(selection))
        for t in C.EIGHT_TASKS:
            selection = dict(task=t, seed=0, contract=V2)
            result = C1.task_resolve(selection)                  # the registered resolver dispatches on the contract
            self.assertEqual(result, C.task_resolve(selection))
            self.assertEqual(result["description"], C.instruction(t, V2))
            self.assertEqual(result["description"], C1.INSTRUCTIONS.get(t, L.TASKS[t]["instruction"]))
            self.assertEqual(result["selection"]["seed"], 0)
            self.assertEqual(result["task_id"], variants.task_resolve(selection)["task_id"].replace("/seed-", f"/{V2}/seed-"))
            self.assertEqual((result["backend_options"]["contract"], result["selection"]["contract"]), (V2, V2))
            self.assertEqual(result["split"], variants.task_resolve(selection)["split"])
        # set-post keeps its bare id, instruction and own contract; v1 keeps its own ids.
        self.assertEqual(C1.task_resolve(dict(task="set-post", seed=0)), variants.task_resolve(dict(task="set-post", seed=0)))
        self.assertEqual(C.selected_contract("set-post"), C.POST_CONTRACT)
        self.assertEqual(C.instruction("set-post", V2), L.TASKS["set-post"]["instruction"])
        self.assertEqual(C1.task_resolve(dict(task="spit-to-tarp", contract=C1.CONTRACT))["task_id"],
                         "soil-excavation/v1/bed/spit-to-tarp/soil-tasks-v1/seed-0")

    def test_coercion_and_validation(self):
        for seed in (True, 2.9, "4"):
            self.assertEqual(C1.task_resolve({"seed": seed, "contract": V2}),
                             C.task_resolve({"seed": seed, "contract": V2}))
            self.assertEqual(C1.task_resolve({"seed": seed, "contract": V2})["selection"]["seed"],
                             variants.task_resolve({"seed": seed})["selection"]["seed"])
        for selection in ({"seed": 10000}, {"seed": None}, {"topsoil": True}, {"stones": True}, {"moisture": "bog"}):
            with self.assertRaises((ValueError, TypeError)):
                C1.task_resolve(dict(selection, contract=V2))
        with self.assertRaises(ValueError):
            C.resolve(seed="1", contract=V2)  # direct resolve is stricter than task_resolve, as before
        for task in ("plant-and-backfill", "post-hole", "backfill-pipe"):
            with self.assertRaises(ValueError):
                C1.task_resolve(dict(task=task, contract=V2))
        self.assertEqual(C1.task_resolve(dict(task="set-post", contract=V2))["task_id"],
                         "soil-excavation/v1/post/set-post/soil-tasks-v2/seed-0")
        self.assertEqual(C1.task_resolve(dict(task="set-post", contract=C.POST_CONTRACT)),
                         variants.task_resolve(dict(task="set-post")))
        self.assertEqual(C1.task_resolve({"contract": "v1"}), variants.task_resolve({}))
        for contract in ("typo", "soil-tasks8"):
            with self.assertRaises(ValueError):
                C1.task_resolve({"contract": contract})
        # Released helpers do not learn v2: a v1 session keeps refusing it exactly as released.
        with self.assertRaises(ValueError):
            C1.selected_contract("spit-to-tarp", V2)
        self.assertEqual(C.choices("spit-to-tarp"), ("v1", C1.CONTRACT, V2))
        self.assertEqual(C.choices("set-post"), (C.POST_CONTRACT, V2))
        self.assertEqual(C.choices("post-hole"), ("v1",))

    def test_benchmark_exact_selections(self):
        b = json.loads((ROOT/"soil_excavation/benchmark.json").read_text())
        selections = b["task_set"]["selections"]
        self.assertEqual(selections, list(C.TASK_SET))
        test_split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        audited = next(t["selection"] for t in test_split if t["environment_id"] == "soil-excavation")
        self.assertEqual(selections[-1], audited)
        self.assertEqual([s["task"] for s in selections], list(C.EIGHT_TASKS))
        for selection in selections:
            spec = tasks.resolve("soil-excavation", selection)
            if selection["task"] == "set-post":
                self.assertEqual(spec.task_id, "soil-excavation/v1/post/set-post/seed-0")
            else:
                self.assertIn(f"/{V2}/", spec.task_id)
                self.assertEqual(spec.launch.backend_options["contract"], V2)
        self.assertIn(V2, b["selector_choices"]["contract"])
        self.assertIn("contract", b["selectors"])
        self.assertEqual([r["selection"] for r in b["task_set_v1"]["tasks"]], list(C1.TASK_SET))
        self.assertTrue(b["task_set"]["note"])

    def test_variation_contract_of_a_v2_session_lists_v2(self):
        value = C.contract()
        self.assertEqual(value["task_set"], [dict(s) for s in C.TASK_SET])
        self.assertEqual(value["task_set_v1"], [dict(s) for s in C1.TASK_SET])
        self.assertEqual(value["contract_choices"]["set-post"], [C.POST_CONTRACT, V2])
        self.assertEqual(C1.contract()["contract_choices"]["set-post"], [C.POST_CONTRACT])   # released, unchanged

    def test_existing_targets_remain_inside_soil_and_base_inside_slab(self):
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
        folder = FOLDERS[env.spec["task"]]
        with patch.object(E, "Ledger", return_value=ledger), \
             patch.object(folder, "tool_held", return_value={"left": held, "right": held}, create=True):
            return env.evaluator.evaluate() if final else env.evaluator.progress()

    def test_all_revised_untouched_starts_are_zero_and_layouts_unchanged(self):
        for t in C.REVISED:
            with self.subTest(task=t):
                f = fixture(t)
                ev = self.evaluate(f)
                self.assertEqual(ev["score"], 0.)
                self.assertFalse(ev["succeeded"])
                self.assertEqual(ev["task_contract"], V2)
                self.assertEqual(ev["idle_margin"], E.IDLE_MARGIN)
                expected = L.setup(t, 0)
                expected["instruction"] = C.INSTRUCTIONS[t]
                self.assertEqual(f[0].spec, expected)
                self.assertEqual(len(f[0].world.recorders), 1)
                self.assertIsInstance(f[0].evaluator, FOLDERS[t].BreakHardpanEvaluator if t == "break-hardpan" else
                                      E.TaskSetEvaluator)

    def test_insertion_progress_is_independent_of_hands(self):
        f = fixture("penetrate-marked")
        f[2]["spade"][2] = L.GRADE - .08
        readings = []
        for held in ({"left": True, "right": True}, {"left": True, "right": False}, {"right": True}, {}):
            with patch.object(E, "Ledger", return_value=f[1]), \
                 patch.object(FOLDERS["penetrate-marked"], "tool_held", return_value=held):
                ev = f[0].evaluator.evaluate()
            self.assertFalse(ev["succeeded"])
            readings.append(ev["progress"])
        self.assertGreater(readings[0], 0.)
        self.assertEqual(len(set(readings)), 1, readings)
        f[2]["spade"][2] = L.GRADE - .151
        for held, success in (({"left": True, "right": True}, True), ({"left": True}, False), ({}, False)):
            with patch.object(E, "Ledger", return_value=f[1]), \
                 patch.object(FOLDERS["penetrate-marked"], "tool_held", return_value=held):
                ev = f[0].evaluator.evaluate()
            self.assertEqual(ev["succeeded"], success)              # two hands stay a success criterion
            self.assertEqual(ev["score"], 1. if success else round(.9*ev["progress"], 4))

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
        self.assertEqual(self.evaluate(f, held=False)["score"], round(.9*E.net_progress(.5, 0., E.IDLE_MARGIN), 4))
        f[1].set_particles([[0., .2, -.1], [0., .2, -.1]], mass=[1.5, .6])
        f[1].transport(1)
        f[1].pos[1] = [-2., -.5, .01]
        self.assertFalse(self.evaluate(f)["succeeded"])
        self.assertLess(self.evaluate(f)["score"], .9)

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
        soil_steps(f[0], sweep(L.GRADE + .03))
        worked = next(d["value"] for d in self.evaluate(f)["diagnostics"] if d["name"] == "worked_columns")
        self.assertEqual(worked, 1.)
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

    def test_hardpan_soil_at_rest_never_earns_credit(self):
        f = fixture("break-hardpan")
        env, lg = f[0], f[1]
        self.assertEqual(env.evaluator.start_raw, 0.)
        reads = []
        windows = FOLDERS["break-hardpan"].WorkWindows
        state = windows.particle_state
        with patch.object(windows, "particle_state", lambda ww, w: reads.append(w.tick) or state(ww, w)):
            hold = [tuple(env.world.tool_q[0, :3])]*20
            for _ in range(100):                     # creep: J_p +170 %, the whole square sinking 5 cm
                lg.jp *= 1.01
                lg.pos[:, 2] -= .0005
                soil_steps(env, hold)
            ev = self.evaluate(f)
            self.assertEqual((ev["raw_progress"], ev["progress"], ev["score"]), (0., 0., 0.))
            # a tool resting in the soil (after a restore: no previous pose) opens nothing either
            env.world.tick += 1000
            soil_steps(env, [(0., -1.65, L.GRADE + .03)]*20)
            lg.jp *= 1.5
            ev = self.evaluate(f)
            self.assertEqual((ev["raw_progress"], ev["progress"], ev["score"]), (0., 0., 0.))
            # a stone sinking through the square is not the robot's work
            env.evaluator.windows.bodies = [b for b in env.evaluator.windows.bodies if b[0] != 0]
            soil_steps(env, [(0., -1.65, L.GRADE + .2 - .01*k) for k in range(20)])
        self.assertEqual(reads, [])
        self.assertEqual(self.evaluate(f)["score"], 0.)
        self.assertEqual(len(f[0].evaluator.windows.bodies), 0)

    def test_hardpan_skips_bodies_whose_box_is_away_from_the_square(self):
        f = fixture("break-hardpan")
        ww, w = f[0].evaluator.windows, f[0].world
        self.assertEqual(ww.body_points(w), {})                 # held 20 cm above grade over the square
        for p, near in (((0., -1.65, L.GRADE + .03), True), ((.8, -1.65, L.GRADE + .03), False),
                        ((0., -.5, L.GRADE + .03), False)):
            w.tool_q[0, :3] = p
            self.assertEqual(list(ww.body_points(w)), [0] if near else [], p)

    def test_hardpan_counts_only_the_worked_columns_and_their_windows(self):
        f = fixture("break-hardpan")
        env, lg = f[0], f[1]
        ww = env.evaluator.windows
        self.assertEqual((ww.REACH, ww.HOLD_S), (.12, 1.))
        self.assertEqual(len(ww.bodies), 1)                   # the mattock; the stone is an inclusion
        blow = [(0., -1.65, L.GRADE + .2 - .034*k) for k in range(6)]
        soil_steps(env, blow)                                  # the pick comes down at 8.5 m/s into the centre
        near = np.hypot(lg.pos0[:, 0], lg.pos0[:, 1] + 1.65) < .03
        far = lg.pos0[:, 0] > .15
        self.assertTrue(ww.open.ravel()[ww.column[near]].all())
        self.assertFalse(ww.open.ravel()[ww.column[far]].any())
        lg.pos[near, 2] += .02                                 # the blow heaved the soil around the pick
        struck = self.evaluate(f)
        self.assertGreater(struck["score"], 0.)
        lg.pos[far, 2] += .02                                  # soil the robot never worked moved: no credit
        lg.jp[far] *= 1.5
        self.assertEqual(self.evaluate(f)["raw_progress"], struck["raw_progress"])
        soil_steps(env, blow[::-1] + [blow[0]]*(int(ww.HOLD_S/.002) + 1))   # out, then held still past HOLD_S
        self.assertFalse(ww.open.any())
        closed = self.evaluate(f)
        self.assertAlmostEqual(closed["raw_progress"], struck["raw_progress"])
        lg.jp[near] *= 1.5                                     # creep in the worked columns after their window
        lg.pos[near, 2] += .05
        self.assertEqual(self.evaluate(f)["raw_progress"], closed["raw_progress"])
        h = env.evaluator.history()
        env.evaluator.restart_history()
        self.assertEqual(self.evaluate(f)["raw_progress"], 0.)
        env.evaluator.restore_history(h)
        self.assertEqual(self.evaluate(f)["raw_progress"], closed["raw_progress"])
        self.assertEqual(env.evaluator.history()["base_off_slab"], 0.)

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
        self.assertEqual(self.evaluate(f)["score"], .9)

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
        self.assertLess(blocked["score"], .9)
        # A cylinder's radial extent matters when tilted horizontally.
        part = f[0].world.scene.parts["auger"][0]
        r = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        self.assertAlmostEqual(FOLDERS["auger-holes"].part_lowest_z(part, [0, 0, .1], r), .08)
        part.kind = "capsule"
        self.assertAlmostEqual(FOLDERS["auger-holes"].part_lowest_z(part, [0, 0, .1], np.eye(3)), .03)

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
        self.assertAlmostEqual(struck["progress"], E.net_progress(.5, 0., E.IDLE_MARGIN))
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

    def test_other_contracts_keep_the_environment_as_built(self):
        for task, contract in (("set-post", C.POST_CONTRACT), ("penetrate-marked", C1.LEGACY_CONTRACT),
                               ("post-hole", C1.LEGACY_CONTRACT), ("spit-to-tarp", C1.CONTRACT)):
            with self.subTest(task=task, contract=contract):
                spec = L.setup(task, 0)
                original = NS(task=task, _record=object())
                env = NS(spec=dict(spec), world=NS(recorders=[]), evaluator=original, _eval_key=(1, 0))
                E.install(env, contract)
                self.assertIs(env.evaluator, original)
                self.assertEqual(env.spec, spec)
                self.assertEqual(env._eval_key, (1, 0))

    def test_set_post_own_contract_is_delegated_without_rescaling(self):
        env, ledger, _, _ = fixture("set-post", C.POST_CONTRACT)
        original = env.evaluator
        self.assertIs(type(original), legacy.Evaluator)
        post = FOLDERS["set-post"]
        for p in (0., .3, .8, 1.):
            with patch.object(legacy, "Ledger", return_value=ledger), patch.object(post, "_set_post", return_value=(
                    [legacy.crit("post_support", 1., False, "fixture")], [], dict(score=min(p, .99), fields=dict(progress=p)))):
                result = env.evaluator.evaluate()
                self.assertEqual(result["score"], min(p, .99))
                self.assertEqual(result["progress"], p)
                self.assertNotIn("task_contract", result)

    def test_set_post_v2_contract_rebases_only_the_failed_score(self):
        """Explicit contract: set-post's own criteria/success, progress net of the settled start, 0.9 factor."""
        reading = dict(A=0., SU=.0054*.9046, ok=False)
        calls = []

        def own(evaluator, lg, final):
            calls.append(final)
            A, SU = reading["A"], reading["SU"]
            crits = [legacy.crit("post_support", SU, reading["ok"], "fixture")]
            if not final:
                return crits, [A], dict(score=.5*A, pending=True, fields=dict(progress=round(.5*A, 4)))
            return crits, [A, SU], dict(score=min(.5*A + .5*SU, .99), fields=dict(progress=round(.5*A + .5*SU, 4)))

        with patch.object(FOLDERS["set-post"], "_set_post", own):
            env, ledger, _, _ = fixture("set-post", V2)
            ev = env.evaluator
            self.assertIsInstance(ev, FOLDERS["set-post"].SetPostEvaluator)
            self.assertEqual(calls, [False, True])  # one advisory and one load-test reading of the start
            self.assertAlmostEqual(ev.start_raw, .5*.0054*.9046)
            self.assertEqual(env.spec["instruction"], L.TASKS["set-post"]["instruction"])
            self.assertEqual(len(env.world.recorders), 1)

            def read(final=True):
                with patch.object(legacy, "Ledger", return_value=ledger):
                    return ev.evaluate() if final else ev.progress()
            untouched = read()
            self.assertEqual(untouched["score"], 0.)
            self.assertFalse(untouched["succeeded"])
            self.assertEqual(untouched["task_contract"], V2)
            self.assertEqual(untouched["set_post_progress"], round(.5*.0054*.9046, 4))
            self.assertEqual(untouched["set_post_contract"], C.POST_CONTRACT)
            self.assertEqual([c["name"] for c in untouched["criteria"]], ["no_fault", "post_support"])
            # Load-test noise and idle creep inside the margin read exactly zero.
            for drift in (7e-7, 4.4e-5, .9*E.IDLE_MARGIN):
                reading.update(SU=.0054*.9046 + 2*drift)
                noisy = read()
                self.assertEqual((noisy["progress"], noisy["score"]), (0., 0.))
                self.assertEqual(noisy["idle_margin"], E.IDLE_MARGIN)
            reading.update(A=.5, SU=.0054*.9046)
            partial = read()
            zero = ev.start_raw + E.IDLE_MARGIN
            net = (.25 + .5*reading["SU"] - zero)/(1 - zero)
            self.assertAlmostEqual(partial["progress"], net)
            self.assertEqual(partial["score"], round(.9*net, 4))
            advisory = read(final=False)
            self.assertFalse(advisory["succeeded"])
            self.assertEqual(advisory["score"], round(.9*(.25 - E.IDLE_MARGIN)/(1 - E.IDLE_MARGIN), 4))
            self.assertEqual(advisory["task_contract"], V2)
            reading.update(A=1., SU=1., ok=True)
            self.assertEqual(read()["score"], 1.)
            self.assertTrue(read()["succeeded"])
            reading.update(ok=False)
            self.assertEqual(read()["score"], .9)  # a failure never ties a success

        # Same readings under set-post's own contract: unchanged min(progress, .99), no baseline.
        reading.update(A=0., SU=.0054*.9046, ok=False)
        with patch.object(FOLDERS["set-post"], "_set_post", own):
            env, ledger, _, _ = fixture("set-post", C.POST_CONTRACT)
            with patch.object(legacy, "Ledger", return_value=ledger):
                self.assertEqual(env.evaluator.evaluate()["score"], round(.5*.0054*.9046, 4))
                self.assertNotIn("task_contract", env.evaluator.evaluate())

    def test_baseline_is_normalized_and_failure_never_ties_success(self):
        self.assertEqual(E.net_progress(.2, .2), 0.)
        self.assertAlmostEqual(E.net_progress(.6, .2), .5)
        self.assertEqual(E.net_progress(.1, .2), 0.)
        self.assertEqual(E.net_progress(1., .2), 1.)
        self.assertEqual(E.net_progress(.21, .2, .02), 0.)
        self.assertAlmostEqual(E.net_progress(.61, .2, .02), .5)
        self.assertEqual(E.net_progress(1., .2, .02), 1.)
        for task in C.EIGHT_TASKS:
            self.assertGreaterEqual(E.idle_margin(task), E.IDLE_MARGIN)
        # Served idle readings: reading above the settled start's baseline -> exactly 0 progress.
        for task, start, drift in (("set-post", .0024187, 4.7e-6), ("set-post", .0024009, 4.4e-5),
                                   ("expose-line", 0., 4.2e-6)):
            self.assertEqual(E.net_progress(start + drift, start, E.idle_margin(task)), 0., task)
        self.assertEqual(E.failed_score(1.), .9)
        self.assertEqual(E.failed_score(.5), .45)
        self.assertEqual(E.failed_score(-.2), 0.)
        f = fixture("spit-to-tarp")
        with patch.object(E, "Ledger", side_effect=ValueError("bad state")):
            advisory = f[0].evaluator.progress()
            self.assertEqual(advisory["task_contract"], V2)
            self.assertIn("bad state", advisory["error"])


class ShapePointTests(unittest.TestCase):
    def test_every_surface_point_is_within_a_centimetre_of_a_sample(self):
        rng = np.random.default_rng(0)
        n = 3000
        half = np.array([.003, .0375, .045])
        u = rng.uniform(-1., 1., (n, 3))
        face = rng.integers(0, 3, n)
        u[np.arange(n), face] = np.sign(u[np.arange(n), face])
        r, h = .016, .1745
        z = rng.uniform(-h - r, h + r, n)
        a = rng.uniform(0., 2*np.pi, n)
        rad = np.sqrt(np.clip(r*r - np.clip(np.abs(z) - h, 0., None)**2, 0., None))
        v = np.array([[0., 0., 0.], [.2, 0., 0.], [0., .15, 0.], [0., 0., .1]])
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        bary = rng.dirichlet((1., 1., 1.), n)
        quat = tuple(Rotation.from_euler("xyz", (.3, -.7, 1.1)).as_quat())
        cases = [(dict(kind="box", size=half), u*half),
                 (dict(kind="capsule", size=(r, h)), np.column_stack((rad*np.cos(a), rad*np.sin(a), z))),
                 (dict(kind="mesh", mesh=(v, f), size=(1., 1., 1.)), np.einsum("nk,nkd->nd", bary, v[f[rng.integers(0, 4, n)]]))]
        for shape, surface in cases:
            with self.subTest(kind=shape["kind"]):
                placed = dict(shape, pos=(.1, -.2, .3), quat=quat)
                pts = FOLDERS["break-hardpan"].shape_points(placed)
                world = surface @ Rotation.from_quat(quat).as_matrix().T + (.1, -.2, .3)
                gap = np.sqrt(((world[:, None, :] - pts[None, :, :])**2).sum(-1)).min(1).max()
                self.assertLessEqual(gap, .01)


TASKS_BACKEND = "gym_anything_robotics.backends.newton_soil_excavation_tasks"
V2_BACKEND = "soil_excavation.task_set_v2_backend"


class BackendWiringTests(unittest.TestCase):
    """The released adapter builds the v2 subclass only for contract soil-tasks-v2; the legacy backend is a stand-in."""

    @classmethod
    def setUpClass(cls):
        module = ModuleType("gym_anything_robotics.backends.newton_soil_excavation")

        class Parent:
            def __init__(self, **options):
                self.received = options
                self.env, self.capabilities = NS(lock=threading.RLock()), NS()

            def _notes(self):
                return {}

            def _variation(self):
                return {k: getattr(self.config, k) for k in variants.SELECTORS}

            def _rebuild(self, value):
                self.rebuilt = value

            def save_checkpoint(self, *, label):
                self._checkpoints[label] = {"tick": self.env.world.tick}
                return {"label": label}

            def restore_checkpoint(self, label, **kwargs):
                return self.env.evaluator.history()
        module.NewtonSoilExcavationBackend = Parent
        cls.tasks_module, cls.v2_module = isolated_import({module.__name__: module, **STUBBED}, TASKS_BACKEND, V2_BACKEND)
        cls.Released = cls.tasks_module.NewtonSoilExcavationTasksBackend
        cls.Backend = cls.v2_module.NewtonSoilExcavationTaskSetV2Backend
        cls.modules = {TASKS_BACKEND: cls.tasks_module, V2_BACKEND: cls.v2_module, **STUBBED}

    def backend(self, task="penetrate-marked", contract=V2):
        b = self.Backend.__new__(self.Backend)
        b._task_contract = C.selected_contract(task, contract)
        b.config = NS(**variants.resolve(task), robot="mobile-fr3-duo")
        evaluator = E.History()
        evaluator.base_off_slab = .1
        b.env = NS(lock=threading.RLock(), evaluator=evaluator, world=NS(tick=100), reset=lambda: {"restored": True})
        b._checkpoints = {}
        return b

    def test_registry_builds_the_v2_subclass_only_for_the_v2_contract(self):
        from gym_anything_robotics import registry
        options = dict(task="auger-holes", seed="3", clutch=None, topsoil="0.254", recording=False)
        with patch.object(self.Released, "_install_contract") as install, \
             patch.object(self.Backend, "_install_contract") as install_v2, patch.dict(sys.modules, self.modules):
            b = registry._newton_soil_excavation_factory(contract=V2, **options)
            self.assertIs(type(b), self.Backend)
            self.assertEqual(b.received, options)                      # the contract never reaches the physics
            self.assertEqual(b._task_contract, V2)
            self.assertEqual((install.call_count, install_v2.call_count), (0, 1))
            for contract in (None, C1.CONTRACT):
                released = registry._newton_soil_excavation_factory(
                    **({} if contract is None else {"contract": contract}), **options)
                self.assertIs(type(released), self.Released)
                self.assertEqual(released.received, options)
            self.assertEqual((install.call_count, install_v2.call_count), (2, 1))
            with self.assertRaises(UnsupportedConfigurationError):
                registry._newton_soil_excavation_factory(contract=V2, task="post-hole")
            with self.assertRaises(UnsupportedConfigurationError):     # a released launch refuses v2 as released
                self.Released.__init__(self.Released.__new__(self.Released), contract=V2, task="post-hole")

    def test_v2_install_and_rebuild_strip_only_the_contract(self):
        with patch.dict(sys.modules, self.modules), patch.object(E, "install") as install, \
             patch.object(self.v2_module, "replace", lambda caps, notes: dict(notes=notes)):
            b = self.Backend(task="auger-holes", seed="3", contract=V2, recording=False)
            self.assertEqual(b.received, dict(task="auger-holes", seed="3", recording=False))
            self.assertEqual(install.call_args.args, (b.env, V2))
            self.assertEqual(b.capabilities, dict(notes={}))
            value = C.resolve("auger-holes", 3, contract=V2)
            b._rebuild(value)
            self.assertEqual(b.rebuilt, {k: value[k] for k in variants.SELECTORS})
            self.assertEqual(install.call_count, 2)
            b.config = NS(**b.rebuilt)                                 # what the legacy rebuild records
            self.assertEqual(b._variation(), dict(b.rebuilt, contract=V2))
            # Rebuilding to a released contract hands the judgement back to the released installer.
            with patch.object(self.Released, "_install_contract") as released_install:
                b._rebuild(C.resolve("auger-holes", 3, contract=C1.CONTRACT))
            self.assertEqual((install.call_count, released_install.call_count), (2, 1))
            self.assertEqual(b._task_contract, C1.CONTRACT)

    def test_selection_preserves_coercion_and_bare_task_meaning(self):
        b = self.backend()
        r = b.select_variation(dict(seed="3", topsoil=".2506"))
        self.assertEqual(r["variation"]["seed"], 3)
        self.assertEqual(r["variation"]["topsoil"], .251)
        self.assertEqual(r["task"], C.INSTRUCTIONS["penetrate-marked"])
        self.assertEqual(r["reset_request"]["options"]["contract"], V2)
        r = b.select_variation(dict(task="set-post"))
        self.assertNotIn("contract", r["variation"])
        self.assertEqual(r["task"], L.TASKS["set-post"]["instruction"])
        r = b.select_variation(dict(task="set-post", contract=V2))
        self.assertEqual(r["variation"]["contract"], V2)
        self.assertEqual(r["task"], L.TASKS["set-post"]["instruction"])
        r = b.select_variation(dict(contract=C1.CONTRACT))
        self.assertEqual(r["variation"]["contract"], C1.CONTRACT)
        with self.assertRaises(ValueError):
            b.select_variation(dict(task="post-hole", contract=V2))
        self.assertEqual(b.variation_contract()["contract_choices"]["penetrate-marked"], ["v1", C1.CONTRACT, V2])

    def test_reset_restarts_and_checkpoints_restore_the_v2_history(self):
        b = self.backend("cable-trench")
        b.save_checkpoint(label="a")
        self.assertEqual(b._checkpoints["a"]["task_set_v2_history"], {"base_off_slab": .1})
        b.env.evaluator.base_off_slab = .3
        self.assertEqual(b.restore_checkpoint("a"), {"base_off_slab": .1})
        self.assertEqual(b.env.evaluator.base_off_slab, .1)
        state, info = b.reset()
        self.assertTrue(state["restored"])
        self.assertEqual(info, {"continuity": "reset", "variation": b._variation()})
        self.assertEqual(b.env.evaluator.base_off_slab, 0.)
        with self.assertRaises(KeyError):
            b.restore_checkpoint("missing")
        # A v1 contract in a v2 session keeps the released v1 history handling.
        b = self.backend("cable-trench", C1.CONTRACT)
        b.save_checkpoint(label="b")
        self.assertEqual(b._checkpoints["b"], {"tick": 100, "task_set_base_off_slab": .1})
        b.reset()
        self.assertEqual(b.env.evaluator.base_off_slab, 0.)

    def test_bad_reset_is_atomic_and_contract_is_not_passed_to_physics(self):
        b = self.backend()
        before = dict(b.config.__dict__)
        for options in ({"contract": "unknown"}, {"seed": "bad"}, {"no_such_option": 1}):
            with self.assertRaises(UnsupportedActionError):
                b.reset(options=options)
        self.assertEqual(b.config.__dict__, before)
        with self.assertRaises(UnsupportedActionError) as caught:
            b.reset(options={"bogus": 1})
        self.assertEqual(str(caught.exception), "Unknown reset options ['bogus']; selectors are task, seed, moisture, "
                                                 "topsoil, stones, contract; nothing changed")
        rebuilt = []
        with patch.object(b, "_rebuild", side_effect=lambda v: rebuilt.append(v)), \
             patch.dict(sys.modules, {"soil_excavation.robot_setup": NS(check_task=lambda *args: None)}):
            b.env.get_state = lambda: {}
            b.reset(options=dict(contract="v1"))
            b.reset(options=dict(task="set-post", contract=V2))
        self.assertNotIn("contract", rebuilt[0])
        self.assertEqual(rebuilt[1]["contract"], V2)
        self.assertEqual(rebuilt[1]["task"], "set-post")


if __name__ == "__main__":
    unittest.main()

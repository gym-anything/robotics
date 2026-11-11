"""drilling-tasks-v2, the render-checked eight-task set: pure Python constructed-state tests. No world, simulator,
server or renderer is constructed.

Run: python -m pytest tests/test_drilling_task_set_v2.py
Ported from the fix branch's drilling/tests/test_catalogue.py to the v2 contract (drilling/task_set_v2.py,
task_set_v2_evaluator.py, task_set_v2_backend.py). Only the real Hole record's arithmetic, layouts, evaluator and GAR
adapter are used; all world APIs are stubs. The fix branch's base-commit checks (git show of provenance.PHYSICS and the
registry) are replaced by checks that the released v1 catalogue, adapter and registry still decide every launch
without the v2 contract.
"""

import copy
import importlib.abc
import json
import math
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from drilling import catalogue as V1, materials as M, params as P, task_set_v2 as C, variants
from drilling.catalogue_evaluator import area_depth_share
from drilling.task_set_v2_evaluator import TaskSetV2Evaluator, configure_environment
from drilling.config import Config
from drilling.holes import Hole
from drilling.concrete import ConcreteFields
from drilling.layout import WALL_REGION, bench_layout, wall_layout
from drilling.task import Evaluator, INSTRUCTIONS, _check

from tests.packages import PACKAGES, REPO

ROOT = PACKAGES
# the bench tasks under the new contract: the set's six plus the prepared alternate, match-bolt
BENCH = tuple(t for t in C.TASKS[:-1] if t not in C.WALL) + C.ALTERNATES


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"newton", "warp", "mujoco", "modal", "isaacsim", "pxr"} or fullname in {
            "drilling.world", "drilling.scene", "drilling.env", "drilling.render", "drilling.render_worker"}:
            raise AssertionError(f"Simulator import forbidden in this suite: {fullname}")


def setUpModule():
    sys.meta_path.insert(0, NoRuntime())


def tearDownModule():
    sys.meta_path[:] = [x for x in sys.meta_path if not isinstance(x, NoRuntime)]


class FakeEnv:
    def __init__(self, task, contract=C.CONTRACT, seed=0):
        self.config = Config(preset="bench", task=task, seed=seed)
        self.layout = bench_layout(task, seed, "dual-fr3", "")
        self.lock = threading.RLock()
        self.poses = {}
        parts, drillables, items, tools = {}, {}, {}, {}
        for part in self.layout["parts"]:
            name = part["name"]
            self.poses[name] = (np.array(part["xyz"]), Rotation.from_euler("xyz", part.get("rpy", (0., 0., 0.))).as_matrix())
            if "half" in part:
                drillables[name] = NS(body=name, center=np.zeros(3), half=np.array(part["half"]), holes=[])
                parts[name] = dict(body=name, kind="drillable")
        for name in ("drill_bit", "drill_press_bit", "pilot_drill_bit"):
            items[name] = NS(name=name, kind="bit", body=name, tip=np.zeros(3), axis=np.array([1., 0., 0.]),
                             length=0.1, radius=0.003, state=dict(broken=False, chipped=False), slots={})
            self.poses[name] = (np.array([3.42, -1., 1.2]), np.eye(3))
        for name, length, radius in (("carriage_bolt", 0.1016, 0.00635), ("m8_bolt0", 0.03, 0.004),
                                     ("m8_bolt1", 0.03, 0.004)):
            items[name] = NS(name=name, kind="bolt", body=name, tip=np.zeros(3), axis=np.array([1., 0., 0.]),
                             length=length, radius=radius, state=dict(max_axial_n=0., insertion_m={}), slots={})
            self.poses.setdefault(name, (np.array([3.36, -0.4, 0.9141]), np.eye(3)))
        self.rpm, self.quill = 0., 0.
        tools["drill"] = NS(item=items["drill_bit"])
        tools["drill_press"] = NS(item=items["drill_press_bit"], s_on=False,
                                  spindle=NS(speed=lambda: self.rpm*2*math.pi/60),
                                  j={"quill": NS(pos=lambda w: self.quill)})
        self.world = NS(scene=NS(parts=parts, tools=tools),
                        hm=NS(drillables=drillables, items=items, item_world=self.item_world),
                        body_pose=lambda body: self.poses[body], fault=None, sim_time=4.,
                        over_payload_s={}, bq=np.zeros((1, 7)), jq=np.zeros(1))
        self.evaluator = Evaluator(self)
        self._initial = copy.deepcopy(self.evaluator.snapshot())
        self.generation = 0
        configure_environment(self, contract)

    def item_world(self, it):
        p, r = self.poses[it.body]
        return p + r @ it.tip, r @ it.axis

    def get_state(self):
        # The original environment's scoring-error policy; the adapter must replace zero with None.
        try:
            ev = self.evaluator.evaluate()
        except Exception as exc:
            ev = dict(success=False, score=0., error=str(exc))
        return dict(evaluation=ev, finite=bool(np.isfinite(self.world.bq).all()), fault=self.world.fault)

    def reset(self):
        self.evaluator.restore(copy.deepcopy(self._initial))
        self.generation += 1
        return self.get_state()

    def add_hole(self, name, mark, diameter, thickness, fraction=1., through=True, kind="drilled", offset=0., tilt=0.):
        d = self.world.hm.drillables[name]
        p = np.array(mark, float) + [offset, 0., 0.]
        h = Hole(id=1 + sum(len(d.holes) for d in self.world.hm.drillables.values()), workpiece=name, p0=p,
                 a=Rotation.from_euler("x", tilt, degrees=True).apply([0., 0., -1.]),
                 face_normal=np.array([0., 0., 1.]), thickness=thickness, kind=kind,
                 stages=[dict(d_mm=diameter, s=thickness*fraction + M.point_height(diameter)*1e-3,
                              through=through, start=0.)])
        h.tools = ["drill_bit" if self.config.task in ("panel-holes", "timber-bolt") else "drill_press_bit"]
        d.holes.append(h)
        return h

    def honest(self):
        t, marks = self.config.task, self.layout.get("marks", [])
        if t == "take-drill":
            self.poses["drill_bit"] = (np.array([3.42, -1., P.BENCH_TOP_Z + 0.05]),
                                      Rotation.from_euler("y", 90., degrees=True).as_matrix())
            self.evaluator.s.update(held_since=0., placed_since=0.)
        elif t == "punch-layout":
            for m in marks:
                self.add_hole("flat_bar", m["local"], 1., 0.001, kind="dimple")
        elif t in ("press-drill", "press-blind", "panel-holes"):
            name, diam, thick = {"press-drill": ("flat_bar", 9., 0.006), "press-blind": ("alu_block", 6.8, 0.0152),
                                "panel-holes": ("cover_panel", 6., 0.0015)}[t]
            for m in marks:
                self.add_hole(name, m["local"], diam, thick, through=t != "press-blind")
        elif t == "timber-bolt":
            d1, d0 = (self.world.hm.drillables[n] for n in ("board1", "board0"))
            p1, r1 = self.poses[d1.body]
            p0, r0 = self.poses[d0.body]
            m1 = np.array(marks[0]["local"])
            m0 = r0.T @ (p1 + r1 @ m1 - p0)
            m0[2] = 0.019
            h1 = self.add_hole("board1", m1, 14.3, 0.038)
            h0 = self.add_hole("board0", m0, 14.3, 0.038)
            it = self.world.hm.items["carriage_bolt"]
            it.slots = {"board1": (0, "hole", h1.id), "board0": (0, "hole", h0.id)}
            self.poses[it.body] = (p0 + r0 @ (m0 + [0., 0., -0.040]),
                                   Rotation.from_euler("y", 90., degrees=True).as_matrix())
        elif t == "match-bolt":
            self.poses["flat_bar"] = (np.array([3.42, -1.45, 0.903]), np.eye(3))
            self.poses["angle_bracket"] = (np.array([3.42, -1.45, 0.9085]), np.eye(3))
            for k, m in enumerate(marks):
                ha = self.add_hole("angle_bracket", m["local"], 9., 0.005)
                hb = self.add_hole("flat_bar", [0., m["local"][1], 0.003], 9., 0.006)
                it = self.world.hm.items[f"m8_bolt{k}"]
                it.slots = {"angle_bracket": (0, "hole", ha.id), "flat_bar": (0, "hole", hb.id)}
                self.poses[it.body] = (np.array([3.42, -1.45 + m["local"][1], 0.898]),
                                       Rotation.from_euler("y", 90., degrees=True).as_matrix())
        return self


class CatalogueTests(unittest.TestCase):
    def test_exact_eight_and_resolver(self):
        raw = json.loads((ROOT/"drilling/benchmark.json").read_text())
        self.assertEqual(C.CONTRACT, "drilling-tasks-v2")
        self.assertIn(C.CONTRACT, raw["selector_choices"]["contract"])
        rows = raw["task_set"]["selections"]
        self.assertEqual(len(rows), 8)
        # the audited hardest task keeps its released split selection exactly
        test_split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
        self.assertEqual(rows[-1], next(t["selection"] for t in test_split if t["environment_id"] == "drilling"))
        self.assertEqual([dict(r, seed=r.get("seed", 0)) for r in rows], list(C.TASK_SET))
        self.assertEqual([r["task"] for r in rows][6], "rebar-relocate")
        for r in rows:
            for resolve in (C.task_resolve, V1.task_resolve):       # benchmark.json's resolver hands v2 to C
                resolved = resolve(r)
                v = resolved["selection"]
                suffix = f"/{C.CONTRACT}" if "contract" in r else ""
                self.assertEqual(resolved["task_id"], f"drilling/v1/{v['preset']}/{v['task']}{suffix}/seed-0")
                self.assertEqual(resolved["description"], C.instruction(r["task"], r.get("contract")))
                self.assertEqual(resolved["backend_options"].get("contract"), r.get("contract"))
        self.assertIn("(within 1.2 degrees)", C.instruction("timber-bolt", C.CONTRACT))
        self.assertEqual(C.INSTRUCTIONS, INSTRUCTIONS)

    def test_legacy_every_task_and_defaults_unchanged(self):
        for sel in [{}, {"preset": "bench"}, *[dict(task=t, seed=s) for ts in variants.PUBLIC.values()
                                               for t in ts for s in (0, 3, 777, 1000, 9999)]]:
            self.assertEqual(C.task_resolve(sel), variants.task_resolve(sel))
            self.assertEqual(C.task_resolve(dict(sel, contract="v1")), variants.task_resolve(sel))

    def test_input_handling(self):
        for seed in (True, "1", 1., -1, 10000):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                C.task_resolve(dict(task="press-drill", seed=seed, contract=C.CONTRACT))
        for value in (False, "", [], "typo"):
            with self.subTest(contract=value), self.assertRaises(ValueError):
                C.task_resolve(dict(task="press-drill", contract=value))
        for t in ("ledger-anchor", "anchor-holes", "clean-and-set"):
            with self.assertRaises(ValueError):
                C.task_resolve(dict(task=t, contract=C.CONTRACT))
        # The old resolver ignores extra keys, and falsy task selects its preset's default.
        self.assertEqual(C.task_resolve(dict(task="", unused=True)), variants.task_resolve(dict(task="", unused=True)))

    def test_released_v1_catalogue_is_unchanged(self):
        # drilling-tasks-v1 keeps its released set (match-bolt in slot 7), instruction and ids; it never offers v2
        raw = json.loads((ROOT/"drilling/benchmark.json").read_text())
        rows = raw["task_set_v1"]["tasks"]
        self.assertEqual([r["selection"] for r in rows], list(V1.TASK_SET))
        self.assertEqual(rows[6]["selection"]["task"], "match-bolt")
        for r in rows:
            self.assertEqual(V1.task_resolve(r["selection"])["task_id"], r["task_id"])
        self.assertNotIn("(within 1.2 degrees)", V1.instruction("timber-bolt", V1.CONTRACT))
        self.assertNotIn(C.CONTRACT, json.dumps(V1.contract()))
        with self.assertRaises(ValueError):
            V1.validate_contract("press-drill", C.CONTRACT)
        with self.assertRaises(ValueError):             # a v2 session offers v1 (legacy) and v2, not v1's task set
            C.validate_contract("press-drill", V1.CONTRACT)
        with self.assertRaises(ValueError):
            C.task_resolve(dict(task="press-drill", contract=V1.CONTRACT))
        self.assertEqual(raw["resolver"], "drilling.catalogue:task_resolve")

    def test_existing_bench_placements_reach_envelope(self):
        # No new placements. Check all reused tool/part reference points and marks at representative/extreme seeds.
        # This is a necessary radial bound, not IK/collision qualification; robot demonstrations remain the evidence.
        for task in BENCH:
            for seed in (0, 3, 42, 777, 9999):
                layout = bench_layout(task, seed, "dual-fr3", "")
                bases = [np.array(m["xyz"]) for m in layout["mounts"].values()]
                points = [(p["name"], np.array(p["xyz"])) for p in layout["parts"] + layout["tools"] if "xyz" in p]
                for mark in layout.get("marks", []):
                    part = next(p for p in layout["parts"] if p["name"] == mark["part"])
                    rot = Rotation.from_euler("xyz", part.get("rpy", (0., 0., 0.))).as_matrix()
                    points.append(("mark", np.array(part["xyz"]) + rot @ mark["local"]))
                points.append(("press spindle", np.array([3.42, -1.45, 1.05])))
                for label, point in points:
                    with self.subTest(task=task, seed=seed, point=label):
                        self.assertLess(min(np.linalg.norm(point-b) for b in bases), 0.855)


class EvaluatorTests(unittest.TestCase):
    def test_every_start_zero_and_every_honest_completion_one(self):
        for task in BENCH:
            for seed in (0, 3, 777):
                with self.subTest(task=task, seed=seed):
                    env = FakeEnv(task, seed=seed)
                    start = env.evaluator.evaluate()
                    self.assertFalse(start["success"])
                    self.assertEqual(start["score"], 0.)
                    self.assertTrue(start["constraints_ok"])
                    end = env.honest().evaluator.evaluate()
                    self.assertTrue(end["success"], [c for c in end["checks"] if not c["passed"]])
                    self.assertAlmostEqual(end["progress"], 1.)
                    self.assertEqual(end["score"], 1.)
                    self.assertEqual(end["contract"], C.CONTRACT)

    def test_legacy_evaluation_unchanged(self):
        for task in BENCH:
            env = FakeEnv(task, contract=C.LEGACY).honest()
            wrapped = env.evaluator.evaluate()
            direct = Evaluator.__new__(Evaluator)
            direct.__dict__.update(env.evaluator.__dict__)
            self.assertEqual(wrapped, direct.evaluate())

    def test_audited_ledger_is_delegated_unchanged(self):
        env = FakeEnv("take-drill", contract=C.LEGACY)
        env.evaluator.task = "ledger-anchor"
        env.evaluator.instruction = INSTRUCTIONS["ledger-anchor"]
        report = ([_check("audited predicate", False)], dict(progress=0.37, scoring="audited progress v3"))
        with patch.object(Evaluator, "_ledger_anchor", return_value=report):
            self.assertEqual(env.evaluator.evaluate(), Evaluator.evaluate(env.evaluator))
            self.assertAlmostEqual(env.evaluator.evaluate()["score"], 0.333)
            self.assertNotIn("contract", env.evaluator.evaluate())

    def test_partial_depth_continuous_press(self):
        env = FakeEnv("press-drill")
        h = env.add_hole("flat_bar", env.layout["marks"][0]["local"], 9., 0.006, 0., through=False)
        for f in (0., 0.001, 0.2, 0.3, 0.5, 0.99):
            h.stages[0]["s"] = 0.006*f + M.point_height(9.)*1e-3
            ev = env.evaluator.evaluate()
            self.assertAlmostEqual(ev["score"], 0.9*f)
            self.assertFalse(ev["success"])

    def test_press_deviations_and_finish(self):
        for change in ("off_mark", "tilt", "diameter", "turn", "spindle", "quill", "bit_contact", "payload", "broken"):
            env = FakeEnv("press-drill").honest()
            h = env.world.hm.drillables["flat_bar"].holes[0]
            if change == "off_mark": h.p0[0] += 0.001
            elif change == "tilt": h.a = Rotation.from_euler("x", 2., degrees=True).apply(h.a)
            elif change == "diameter": h.max_offset_m = 0.00025
            elif change == "turn": env.evaluator.s["max_rel_turn_deg"]["flat_bar"] = 3.
            elif change == "spindle": env.rpm = 500.
            elif change == "quill": env.quill = -0.03
            elif change == "bit_contact":
                env.world.hm.items["drill_press_bit"].slots = {"flat_bar": (0, "hole", h.id)}
                env.poses["drill_press_bit"] = (env.poses["flat_bar"][0], np.eye(3))
            elif change == "payload": env.world.over_payload_s = {"left": 100.}
            elif change == "broken": env.world.hm.items["drill_press_bit"].state["broken"] = True
            with self.subTest(change=change):
                ev = env.evaluator.evaluate()
                self.assertFalse(ev["success"])
                self.assertLess(ev["progress"], 1.)

    def test_abandoned_start_does_not_mask_completed_hole(self):
        for task, name, d, thick in (("press-drill", "flat_bar", 9., 0.006), ("press-blind", "alu_block", 6.8, 0.025)):
            env = FakeEnv(task).honest()
            env.world.hm.drillables[name].holes[0].p0[0] += 0.0002
            env.add_hole(name, env.layout["marks"][0]["local"], d, thick, 0., through=False)
            self.assertTrue(env.evaluator.evaluate()["success"])

    def test_blind_overshoot_and_breakthrough(self):
        env = FakeEnv("press-blind").honest()
        h = env.world.hm.drillables["alu_block"].holes[0]
        for fd, expected in ((0.0075, 0.5), (0.01675, 0.5), (0.018, 0.)):
            h.stages[0]["s"] = fd + M.point_height(6.8)*1e-3
            self.assertAlmostEqual(env.evaluator.evaluate()["progress"], expected)
        h.stages[0]["through"] = True
        self.assertEqual(env.evaluator.evaluate()["progress"], 0.)

    def test_hold_continuity_and_deviations(self):
        env = FakeEnv("take-drill").honest()
        for duration in (0., 0.001, 0.5, 1.5, 2.999):
            env.evaluator.s.update(held_since=4.-duration, placed_since=4.-duration)
            self.assertAlmostEqual(env.evaluator.evaluate()["progress"], duration/3.)
        env.evaluator.s.update(held_since=0., placed_since=0.)
        for key, value in (("placed_since", None), ("held_since", None), ("trigger_max_m", 0.003), ("bit_touched", True)):
            with self.subTest(key=key), patch.dict(env.evaluator.s, {key: value}):
                result = env.evaluator.evaluate()
                self.assertFalse(result["success"])
                self.assertLess(result["progress"], 1.)
        env.poses["drill_bit"] = (np.array([3.42, -1., P.BENCH_TOP_Z+0.12]),
                                  Rotation.from_euler("y", -90., degrees=True).as_matrix())
        self.assertFalse(env.evaluator.evaluate()["success"])
        self.assertLess(env.evaluator.evaluate()["progress"], 0.8)

    def test_punch_offsets_strays_and_motion(self):
        env = FakeEnv("punch-layout").honest()
        h = env.world.hm.drillables["flat_bar"].holes[0]
        h.p0[0] += 0.0009
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.875)
        env.add_hole("flat_bar", env.layout["marks"][0]["local"], 1., 0.001, kind="dimple", offset=0.003)
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.8125)
        pos, rot = env.poses["flat_bar"]
        env.poses["flat_bar"] = (pos + [0.003, 0., 0.], rot)
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.40625)

    def test_panel_pilot_volume_and_strategy_independence(self):
        env = FakeEnv("panel-holes")
        h = env.add_hole("cover_panel", env.layout["marks"][0]["local"], 3., 0.0015)
        self.assertAlmostEqual(area_depth_share(h, 6., 0.0015), 0.25)
        h.stages.append(dict(d_mm=6., s=M.point_height(6.)*1e-3, through=False, start=0.))
        self.assertAlmostEqual(area_depth_share(h, 6., 0.0015), 0.25)
        h.stages[-1]["s"] += 0.0015*0.5
        self.assertAlmostEqual(area_depth_share(h, 6., 0.0015), 0.625)
        for f in (0.99, 0.999, 1.):
            h.stages[-1]["s"] = M.point_height(6.)*1e-3 + 0.0015*f
            self.assertAlmostEqual(area_depth_share(h, 6., 0.0015), 0.25 + 0.75*f)
        direct = FakeEnv("panel-holes").honest()
        with_pilot = FakeEnv("panel-holes").honest()
        for hole in with_pilot.world.hm.drillables["cover_panel"].holes:
            hole.stages.append(dict(d_mm=3., s=0.004, through=True, start=0.))
            hole.stages.reverse()
        self.assertEqual(direct.evaluator.evaluate()["score"], with_pilot.evaluator.evaluate()["score"])

    def test_panel_stray_partial_and_full_hole(self):
        env = FakeEnv("panel-holes").honest()
        h = env.add_hole("cover_panel", [0., 0., 0.00075], 6., 0.0015, 0.5, through=False)
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.875)
        h.stages[0]["through"] = True
        ev = env.evaluator.evaluate()
        self.assertFalse(ev["success"])
        self.assertAlmostEqual(ev["progress"], 0.75)

    def test_panel_precision_deviations(self):
        for kind in ("offset", "axis", "roundness", "missing", "pilot_only"):
            env = FakeEnv("panel-holes").honest()
            holes = env.world.hm.drillables["cover_panel"].holes
            h = holes[0]
            if kind == "offset": h.p0[0] += 0.002
            elif kind == "axis": h.a = Rotation.from_euler("x", 2., degrees=True).apply(h.a)
            elif kind == "roundness": h.wander_m = 0.0006
            elif kind == "missing": holes.pop(0)
            elif kind == "pilot_only": h.stages[0]["d_mm"] = 3.
            with self.subTest(kind=kind):
                self.assertFalse(env.evaluator.evaluate()["success"])
                self.assertLess(env.evaluator.evaluate()["progress"], 1.)

    def test_stray_does_not_substitute_for_missing_panel_hole(self):
        env = FakeEnv("panel-holes").honest()
        env.world.hm.drillables["cover_panel"].holes.pop()
        env.add_hole("cover_panel", [0., 0., 0.00075], 6., 0.0015)
        ev = env.evaluator.evaluate()
        self.assertFalse(ev["success"])
        self.assertAlmostEqual(ev["progress"], 0.5)

    def test_bolt_withdrawal_stale_records_and_force(self):
        for task, name in (("timber-bolt", "carriage_bolt"), ("match-bolt", "m8_bolt0")):
            for kind in ("withdrawn", "beside", "missing_slot", "stale_slot", "forced"):
                env = FakeEnv(task).honest()
                it = env.world.hm.items[name]
                if kind == "withdrawn":
                    p, r = env.poses[name]
                    env.poses[name] = (p + [0., 0., 0.2], r)
                    it.state["insertion_m"] = {"board0/1": 1.}
                elif kind == "beside":
                    p, r = env.poses[name]
                    env.poses[name] = (p + [0.05, 0., 0.], r)
                elif kind == "missing_slot": it.slots.clear()
                elif kind == "stale_slot": it.slots[next(iter(it.slots))] = (0, "hole", 999)
                elif kind == "forced": it.state["max_axial_n"] = 70.
                with self.subTest(task=task, kind=kind):
                    ev = env.evaluator.evaluate()
                    self.assertFalse(ev["success"])
                    self.assertLess(ev["progress"], 1.)

    def test_timber_lower_hole_and_partial_insertion(self):
        env = FakeEnv("timber-bolt").honest()
        h = env.world.hm.drillables["board0"].holes[0]
        h.stages[0]["d_mm"] = 3.
        self.assertFalse(env.evaluator.evaluate()["success"])
        self.assertLess(env.evaluator.evaluate()["progress"], 1.)
        env = FakeEnv("timber-bolt").honest()
        pos, rot = env.poses["carriage_bolt"]
        env.poses["carriage_bolt"] = (pos + [0., 0., 0.021], rot)
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], (2.+0.75)/3.)

    def test_match_bolts_can_exchange_identity_but_not_duplicate_a_pair(self):
        env = FakeEnv("match-bolt").honest()
        a, b = (env.world.hm.items[f"m8_bolt{k}"] for k in range(2))
        a.slots, b.slots = b.slots, a.slots
        env.poses[a.body], env.poses[b.body] = env.poses[b.body], env.poses[a.body]
        self.assertTrue(env.evaluator.evaluate()["success"])
        a.slots = copy.deepcopy(b.slots)
        env.poses[a.body] = env.poses[b.body]
        self.assertFalse(env.evaluator.evaluate()["success"])
        self.assertLess(env.evaluator.evaluate()["progress"], 1.)

    def test_timber_valid_tilt_follows_bore_to_lower_board(self):
        env = FakeEnv("timber-bolt").honest()
        top, bottom = (env.world.hm.drillables[n].holes[0] for n in ("board1", "board0"))
        top.p0[0] += 0.0019
        rot = Rotation.from_euler("y", -1., degrees=True)
        top.a = rot.apply([0., 0., -1.])
        bottom.a = top.a.copy()
        bottom.p0 = env.evaluator._timber_lower_mark(top)
        # The bolt follows the same slightly tilted hole, not a forced vertical path.
        p0, r0 = env.poses["board0"]
        env.poses["carriage_bolt"] = (p0 + r0 @ (bottom.p0 + bottom.a*0.040),
                                      Rotation.from_euler("y", 89., degrees=True).as_matrix())
        result = env.evaluator.evaluate()
        self.assertTrue(result["success"], [c for c in result["checks"] if not c["passed"]])
        self.assertEqual(result["score"], 1.)

    def test_broken_panel_bit_only_zeros_affected_bore(self):
        env = FakeEnv("panel-holes").honest()
        for h in env.world.hm.drillables["cover_panel"].holes[1:]:
            h.tools = ["pilot_drill_bit"]
        env.world.hm.items["drill_bit"].state["broken"] = True
        result = env.evaluator.evaluate()
        self.assertFalse(result["success"])
        self.assertAlmostEqual(result["progress"], 0.75)

    def test_reordering_state_records_changes_neither_progress_nor_success(self):
        for task in ("panel-holes", "punch-layout", "timber-bolt", "match-bolt"):
            env = FakeEnv(task).honest()
            before = env.evaluator.evaluate()
            for d in env.world.hm.drillables.values():
                d.holes.reverse()
                for h in d.holes:
                    h.stages.reverse()
                    h.tools.reverse()
            after = env.evaluator.evaluate()
            self.assertEqual(before, after)

    def test_nonfinite_fault_and_scoring_error_are_unscored(self):
        for task in BENCH:
            env = FakeEnv(task)
            env.world.fault = "fixture fault"
            self.assertIsNone(env.get_state()["evaluation"]["score"])
            env.world.fault = None
            env.world.bq[0, 0] = float("nan")
            self.assertIsNone(env.get_state()["evaluation"]["score"])
            env.world.bq[0, 0] = 0.
            with patch.object(env.evaluator, "_progress", side_effect=ValueError("fixture evaluator error")):
                ev = env.get_state()["evaluation"]
                self.assertIsNone(ev["score"])
                self.assertFalse(ev["evaluated"])
                self.assertEqual(ev["contract"], C.CONTRACT)


class FakeWallEnv:
    """rebar-relocate on the north wall: the real layout and rebar fields, the wall body as stations.wall builds it
    (centre (0, 3.1, 1.4), identity rotation, fields frame = x along, y depth from the face, z up)."""
    CENTRE = np.array([0., P.WALL_FACE_Y + 0.5*P.WALL_THICK_M, 1.4])

    def __init__(self, contract=C.CONTRACT, seed=0):
        self.config = Config(preset="wall", task="rebar-relocate", seed=seed)
        self.layout = L = wall_layout("rebar-relocate", seed, "dual-fr3", "")
        self.lock = threading.RLock()
        fields = ConcreteFields(L["field_seed"], WALL_REGION["x"], WALL_REGION["z"], thickness=P.WALL_THICK_M,
                                rebar_offset=L["rebar_offset"])
        fmap = (np.array([-self.CENTRE[0], -0.5*P.WALL_THICK_M, -self.CENTRE[2]]), np.eye(3))
        wall = NS(name="wall", body=0, holes=[], fields=fields, fields_map=fmap,
                  to_fields=lambda p: np.asarray(p, float) - fmap[0])
        bit = NS(name="sds_bit", kind="bit", body="sds_bit", state=dict(broken=False, chipped=False), slots={})
        self.world = NS(scene=NS(parts={}, tools={}),
                        hm=NS(drillables=dict(wall=wall), items=dict(sds_bit=bit),
                              world_point=lambda d, p: self.CENTRE + np.asarray(p, float)),
                        body_pose=None, fault=None, sim_time=4., over_payload_s={},
                        bq=np.array([[*self.CENTRE, 0., 0., 0., 1.]]), jq=np.zeros(1))
        self.evaluator = Evaluator(self)
        configure_environment(self, contract)

    def get_state(self):
        return dict(evaluation=self.evaluator.evaluate(), finite=True, fault=self.world.fault)

    def local(self, x, z):
        return np.array([x, P.WALL_FACE_Y, z]) - self.CENTRE

    def drill(self, x, z, depth, damage=0., reached=False, tilt=0.):
        d = self.world.hm.drillables["wall"]
        h = Hole(id=1 + len(d.holes), workpiece="wall", p0=self.local(x, z),
                 a=Rotation.from_euler("z", tilt, degrees=True).apply([0., 1., 0.]), face_normal=np.array([0., -1., 0.]),
                 stages=[dict(d_mm=13.3, s=depth + M.point_height(13.3)*1e-3, through=False, start=0.)])
        h.rebar_damage_m, h.tools = damage, ["sds_bit"]
        if reached:
            h.events.append(dict(event="rebar_contact", depth_mm=round(depth*1e3, 2)))
        d.holes.append(h)
        return h

    def bar_face_depth(self):
        """Depth along the bore over the bar at which the tip reaches it (task._reached_bar's geometric rule)."""
        mk = self.layout["marks"][self.layout["rebar_mark"]]
        f, r = self.world.hm.drillables["wall"].fields, 0.5e-3*13.3
        for k in range(2000):                                  # the first depth (0.1 mm steps) the tip reaches it
            depth = 1e-4*k
            tip = self.local(mk["x"], mk["z"]) + [0., depth + M.point_height(13.3)*1e-3, 0.]
            if f.bar_distance(tip - f_map(self))[0] <= r:
                return depth
        raise AssertionError("no bar under the mark")

    def honest(self, replacement_dx=0.09):
        L, kbar = self.layout, self.layout["rebar_mark"]
        for k, mk in enumerate(L["marks"]):
            if k != kbar:
                self.drill(mk["x"], mk["z"], 0.112)
        mk = L["marks"][kbar]
        self.stopped = self.drill(mk["x"], mk["z"], self.bar_face_depth(), damage=2.6e-5, reached=True)
        side = 1. if mk["x"] - L["x_c"] > 0 else -1.          # toward the station centre (away from the wall's end)
        self.replacement = self.drill(mk["x"] - side*replacement_dx, mk["z"], 0.112)
        return self


def f_map(env):
    return env.world.hm.drillables["wall"].fields_map[0]


class WallTests(unittest.TestCase):
    def test_start_zero_and_honest_completion_one(self):
        for seed in (0, 3, 777):
            with self.subTest(seed=seed):
                env = FakeWallEnv(seed=seed)
                start = env.evaluator.evaluate()
                self.assertFalse(start["success"])
                self.assertEqual(start["score"], 0.)
                self.assertTrue(start["constraints_ok"])
                end = env.honest().evaluator.evaluate()
                self.assertTrue(end["success"], [c for c in end["checks"] if not c["passed"]])
                self.assertAlmostEqual(end["progress"], 1.)
                self.assertEqual(end["score"], 1.)
                self.assertEqual(end["contract"], C.CONTRACT)

    def test_success_is_the_legacy_verdict(self):
        for honest in (False, True):
            env = FakeWallEnv(contract=C.LEGACY)
            if honest:
                env.honest()
            legacy = env.evaluator.evaluate()
            direct = Evaluator.__new__(Evaluator)
            direct.__dict__.update(env.evaluator.__dict__)
            self.assertEqual(legacy, direct.evaluate())
            configure_environment(env, C.CONTRACT)
            new = env.evaluator.evaluate()
            self.assertEqual(new["checks"], legacy["checks"])
            self.assertEqual(new["success"], legacy["success"])

    def test_payload_constraint_kept_and_halves_progress(self):
        env = FakeWallEnv().honest()
        env.world.over_payload_s = {"left": 12.9}
        out = env.evaluator.evaluate()
        self.assertFalse(out["success"])
        self.assertAlmostEqual(out["progress"], 0.5)
        self.assertAlmostEqual(out["score"], 0.45)

    def test_partial_depth_and_stop_share_are_continuous(self):
        env = FakeWallEnv()
        mk = env.layout["marks"][(env.layout["rebar_mark"] + 1) % 4]
        h = env.drill(mk["x"], mk["z"], 0.054)
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.25*0.5, places=4)
        h.stages[0]["s"] += 0.054
        self.assertAlmostEqual(env.evaluator.evaluate()["progress"], 0.25, places=4)
        h.stages[0]["s"] += 0.030                                  # 138 mm: past 125, toward the far mat
        self.assertLess(env.evaluator.evaluate()["progress"], 0.25)
        env = FakeWallEnv()
        mk = env.layout["marks"][env.layout["rebar_mark"]]
        stop = env.bar_face_depth()
        prev = 0.
        for share in (0.25, 0.5, 0.9):
            env.world.hm.drillables["wall"].holes = []
            env.drill(mk["x"], mk["z"], share*stop)
            S = env.evaluator.evaluate()["progress_terms"]["stopped_hole"]
            self.assertGreater(S, prev)
            self.assertLess(S, 1.)
            prev = S
        env.world.hm.drillables["wall"].holes = []
        env.drill(mk["x"], mk["z"], stop, reached=True)
        out = env.evaluator.evaluate()
        self.assertEqual(out["progress_terms"]["stopped_hole"], 1.)
        self.assertAlmostEqual(out["progress"], 0.25*0.5)

    def test_cut_bar_zeroes_mark_and_costs_a_quarter(self):
        env = FakeWallEnv().honest()
        env.stopped.rebar_damage_m = 0.004
        out = env.evaluator.evaluate()
        self.assertFalse(out["success"])
        self.assertAlmostEqual(out["progress"], 0.75 - 0.25)
        env = FakeWallEnv().honest()
        mk = env.layout["marks"][(env.layout["rebar_mark"] + 1) % 4]
        env.drill(mk["x"], mk["z"] + 0.15, 0.06, damage=0.003)                # a stray hole into a bar elsewhere
        out = env.evaluator.evaluate()
        self.assertTrue(out["success"])                                        # the legacy verdict decides success
        self.assertEqual(out["progress_terms"]["holes_cutting_a_bar"], 1)
        self.assertAlmostEqual(out["progress"], 0.75)

    def test_replacement_distance_window(self):
        def R(dx):
            return FakeWallEnv().honest(replacement_dx=dx).evaluator.evaluate()
        depth = FakeWallEnv().honest().stopped.depth          # the stopped hole's depth sets the 2 x depth minimum
        full = R(0.09)
        self.assertTrue(full["success"])
        self.assertEqual(full["progress_terms"]["replacement"], 1.)
        near = R(1.5*depth)                                  # inside 2 x depth: legacy fails, partial distance credit
        self.assertFalse(near["success"])
        self.assertAlmostEqual(near["progress_terms"]["replacement"], 0.5, places=4)
        self.assertEqual(R(0.9*depth)["progress_terms"]["replacement"], 0.)
        far = R(0.175)
        self.assertFalse(far["success"])
        self.assertAlmostEqual(far["progress_terms"]["replacement"], 0.5, places=6)
        self.assertEqual(R(0.21)["progress_terms"]["replacement"], 0.)


class BackendTests(unittest.TestCase):
    def backend(self, task="press-drill", contract=C.CONTRACT):
        from drilling.task_set_v2_backend import NewtonDrillingTaskSetV2Backend
        b = NewtonDrillingTaskSetV2Backend.__new__(NewtonDrillingTaskSetV2Backend)
        b._task_contract = contract
        b.env = FakeEnv(task, contract=contract)
        b.config, b.trace = b.env.config, None
        return b

    def test_registered_factory_is_the_released_adapter(self):
        src = (REPO/"src/gym_anything_robotics/registry.py").read_text()
        body = src[src.index("def _newton_drilling_factory"):src.index('register("drilling"')]
        self.assertIn("return NewtonDrillingCatalogueBackend(**kwargs)", body)
        self.assertNotIn("v2", body)

    def test_only_a_v2_launch_builds_the_v2_session(self):
        # legacy, v1 and audited launches are exactly the released adapter
        from gym_anything_robotics.registry import _newton_drilling_factory
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        from gym_anything_robotics.backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend
        from drilling.task_set_v2_backend import NewtonDrillingTaskSetV2Backend
        for sel in ({}, dict(task="press-drill", seed=0), dict(preset="wall", task="ledger-anchor"),
                    dict(task="rebar-relocate", contract="v1"), dict(task="match-bolt", contract=V1.CONTRACT)):
            options = V1.task_resolve(sel)["backend_options"]
            with patch.object(NewtonDrillingBackend, "__init__", return_value=None) as init:
                b = _newton_drilling_factory(**options)
                init.assert_called_once_with(**{k: v for k, v in options.items() if k != "contract"})
            self.assertIs(type(b), NewtonDrillingCatalogueBackend)
            self.assertEqual(b._task_contract, options.get("contract", "v1"))
        options = V1.task_resolve(dict(task="rebar-relocate", contract=C.CONTRACT))["backend_options"]
        self.assertEqual(options, dict(preset="wall", task="rebar-relocate", seed=0, contract=C.CONTRACT))
        with patch.object(NewtonDrillingBackend, "__init__", return_value=None) as init:
            b = _newton_drilling_factory(**options)
            init.assert_called_once_with(preset="wall", task="rebar-relocate", seed=0)
        self.assertIs(type(b), NewtonDrillingTaskSetV2Backend)
        self.assertEqual(b._task_contract, C.CONTRACT)
        with self.assertRaises(ValueError):
            _newton_drilling_factory(task="ledger-anchor", preset="wall", contract=C.CONTRACT)

    def test_constructor_keeps_base_options_and_coercion(self):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        from gym_anything_robotics.backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend
        from drilling.task_set_v2_backend import NewtonDrillingTaskSetV2Backend
        with patch.object(NewtonDrillingBackend, "__init__", return_value=None) as init:
            b = NewtonDrillingCatalogueBackend(preset="bench", task="press-drill", seed="3", config={"variant": "worn-bit"},
                                               contract=C.CONTRACT)
            init.assert_called_once_with(preset="bench", task="press-drill", seed="3", config={"variant": "worn-bit"})
        self.assertIsInstance(b, NewtonDrillingTaskSetV2Backend)

    def test_policy_instruction_tracks_evaluator(self):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        from gym_anything_robotics.policy_access import PolicyAccess
        b = self.backend("timber-bolt")
        old = PolicyAccess("dual-fr3", (), {}, frozenset(), lambda: "old", {})
        with patch.object(NewtonDrillingBackend, "policy_access", return_value=old):
            access = b.policy_access()
        self.assertEqual(access.instruction(), C.INSTRUCTIONS["timber-bolt"])
        configure_environment(b.env, C.LEGACY)
        self.assertEqual(access.instruction(), INSTRUCTIONS["timber-bolt"])

    def test_selection_and_checkpoint_identity(self):
        b = self.backend()
        out = b.select_variation(dict(task="press-drill", contract=C.CONTRACT, seed=3))
        self.assertEqual(out["reset_request"], dict(seed=3, options=dict(preset="bench", task="press-drill", contract=C.CONTRACT)))
        self.assertEqual(b.select_variation(dict(task="press-drill"))["reset_request"]["options"]["contract"], "v1")
        self.assertEqual(b._variation_key()["contract"], C.CONTRACT)
        self.assertEqual(b.variation_contract()["current"]["contract"], C.CONTRACT)
        with self.assertRaises(ValueError): b.select_variation(dict(task="ledger-anchor", contract=C.CONTRACT))

    def test_reset_preserves_contract_and_refusals_leave_state_unchanged(self):
        from gym_anything_robotics.errors import ActionRejectedError
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        b = self.backend()
        def stub_reset(this, seed=None, options=None):
            return this.env.reset(), dict(variation=dict(preset=this.config.preset, task=this.config.task, seed=this.config.seed))
        with patch.object(NewtonDrillingBackend, "reset", stub_reset):
            state, info = b.reset()
            self.assertEqual(state["evaluation"]["contract"], C.CONTRACT)
            self.assertEqual(info["variation"]["contract"], C.CONTRACT)
            for options, seed in (({"unknown": 1}, None), ({"task": []}, None), ({"contract": "bad"}, None),
                                  ({}, "1"), ({"preset": "wall", "task": "press-drill"}, None)):
                with self.assertRaises(ActionRejectedError): b.reset(seed=seed, options=options)
                self.assertEqual(b._task_contract, C.CONTRACT)
            # the base's refusal form, with the added selector named
            with self.assertRaisesRegex(ActionRejectedError, r"^Reset refused, nothing changed: Reset selectors are "
                                        r"preset, task, contract \(options\) and seed; got \['unknown'\]$"):
                b.reset(options={"unknown": 1})
            state, _ = b.reset(options={"contract": "v1"})
            self.assertNotIn("contract", state["evaluation"])
            self.assertNotIn("contract", b._variation_key())
            state, _ = b.reset(options={"contract": C.CONTRACT})
            self.assertEqual(state["evaluation"]["contract"], C.CONTRACT)

    def test_replacement_world_gets_adapter_and_new_task_defaults_to_legacy(self):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        b = self.backend()
        def stub_reset(this, seed=None, options=None):
            this.env = FakeEnv(options["task"], contract=C.LEGACY)
            this.config = this.env.config
            return this.env.get_state(), dict(variation=dict(preset="bench", task=options["task"], seed=0))
        with patch.object(NewtonDrillingBackend, "reset", stub_reset):
            state, _ = b.reset(options=dict(task="panel-holes", contract=C.CONTRACT))
            self.assertEqual(state["evaluation"]["contract"], C.CONTRACT)
            state, _ = b.reset(options=dict(task="punch-layout"))
            self.assertNotIn("contract", state["evaluation"])

    def test_failed_reset_rolls_back_contract(self):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        b = self.backend()
        with patch.object(NewtonDrillingBackend, "reset", side_effect=RuntimeError("fixture build failure")):
            with self.assertRaises(RuntimeError): b.reset(options=dict(contract="v1"))
        self.assertEqual(b._task_contract, C.CONTRACT)
        self.assertEqual(b.env.evaluator.task_contract, C.CONTRACT)


if __name__ == "__main__":
    unittest.main()

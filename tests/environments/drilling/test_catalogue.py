"""Pure Python constructed-state tests. No world, simulator, server or renderer is constructed.

Run: python -m pytest tests/environments/drilling
Only the real Hole record's arithmetic, layouts, evaluator and GAR adapter are used; all world APIs are stubs.
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

from drilling import catalogue as C, materials as M, params as P, variants
from drilling.catalogue_evaluator import area_depth_share, configure_environment
from drilling.config import Config
from drilling.holes import Hole
from drilling.layout import bench_layout
from drilling.task import Evaluator, INSTRUCTIONS, _check
from tests.packages import PACKAGES

ROOT = PACKAGES


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
        rows = raw["task_set"]["tasks"]
        self.assertEqual([r["selection"] for r in rows], list(C.TASK_SET))
        self.assertEqual(len(rows), 8)
        for r in rows:
            resolved = C.task_resolve(r["selection"])
            self.assertEqual(resolved["task_id"], r["task_id"])
            self.assertEqual(resolved["description"], C.instruction(r["selection"]["task"], r["selection"].get("contract")))

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
        for t in ("ledger-anchor", "rebar-relocate", "anchor-holes", "clean-and-set"):
            with self.assertRaises(ValueError):
                C.task_resolve(dict(task=t, contract=C.CONTRACT))
        # The old resolver ignores extra keys, and falsy task selects its preset's default.
        self.assertEqual(C.task_resolve(dict(task="", unused=True)), variants.task_resolve(dict(task="", unused=True)))

    def test_existing_bench_placements_reach_envelope(self):
        # No new placements. Check all reused tool/part reference points and marks at representative/extreme seeds.
        # This is a necessary radial bound, not IK/collision qualification; robot demonstrations remain the evidence.
        for task in C.TASKS[:-1]:
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
        for task in C.TASKS[:-1]:
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
        for task in C.TASKS[:-1]:
            env = FakeEnv(task, contract=C.LEGACY).honest()
            wrapped = env.evaluator.evaluate()
            direct = Evaluator.__new__(Evaluator)
            direct.__dict__.update(env.evaluator.__dict__)
            self.assertEqual(wrapped, direct.evaluate())

    def test_ledger_anchor_is_delegated_unchanged(self):
        env = FakeEnv("take-drill", contract=C.LEGACY)
        env.evaluator.task = "ledger-anchor"
        env.evaluator.instruction = INSTRUCTIONS["ledger-anchor"]
        report = ([_check("ledger predicate", False)], dict(progress=0.37, scoring="progress v3"))
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
        for task in C.TASKS[:-1]:
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


class BackendTests(unittest.TestCase):
    def backend(self, task="press-drill", contract=C.CONTRACT):
        from gym_anything_robotics.backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend
        b = NewtonDrillingCatalogueBackend.__new__(NewtonDrillingCatalogueBackend)
        b._task_contract = contract
        b.env = FakeEnv(task, contract=contract)
        b.config, b.trace = b.env.config, None
        return b

    def test_factory_routes_to_adapter_without_runtime_import(self):
        from gym_anything_robotics.registry import _newton_drilling_factory
        with patch("gym_anything_robotics.backends.newton_drilling_catalogue.NewtonDrillingCatalogueBackend") as factory:
            _newton_drilling_factory(seed="3", contract=C.CONTRACT)
            factory.assert_called_once_with(seed="3", contract=C.CONTRACT)

    def test_constructor_keeps_base_options_and_coercion(self):
        from gym_anything_robotics.backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        with patch.object(NewtonDrillingBackend, "__init__", return_value=None) as init:
            NewtonDrillingCatalogueBackend(preset="bench", task="press-drill", seed="3", config={"variant": "worn-bit"},
                                           contract=C.CONTRACT)
            init.assert_called_once_with(preset="bench", task="press-drill", seed="3", config={"variant": "worn-bit"})

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

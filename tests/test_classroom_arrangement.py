"""Classroom arrangement: discovery, task catalogue, layouts, the base-twist action and the public backend contract.

Run (the backend tests build the full room, ~1 min each):
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_classroom_arrangement.py
"""
import math
import unittest

from tests.packages import PACKAGES


class Discovery(unittest.TestCase):
    def test_launch_and_robots(self):
        from gym_anything_robotics.launch import describe
        from robots.catalog import for_environment, environment_robots, resolve
        d = describe("classroom-arrangement")
        self.assertEqual(d["default_robot"], "mobile-fr3-duo")
        self.assertEqual({r["id"] for r in environment_robots("classroom-arrangement")}, {"mobile-fr3-duo", "mobile-fr3-single"})
        self.assertEqual(for_environment("classroom-arrangement").id, "classroom-mobile-fr3-duo")
        for bad in ("g1", "dual-fr3", "single-fr3", "dual-panda"):
            with self.assertRaises(ValueError):
                for_environment("classroom-arrangement", bad)
        self.assertEqual(describe("arrange-classroom-desks")["environment_id"], "classroom-arrangement")
        resolve("classroom-mobile-fr3-single")

    def test_task_catalogue(self):
        from gym_anything_robotics import tasks
        from classroom_arrangement.layout import TASKS
        for t in TASKS:
            spec = tasks.resolve("classroom-arrangement", {"task": t, "seed": 7})
            self.assertTrue(spec.task_id.startswith(f"classroom-arrangement/v1/room114/{t}/seed-7"))
            self.assertIn(spec.split, ("train", "validation", "test"))
        with self.assertRaises(ValueError):
            tasks.resolve("classroom-arrangement", {"task": "straighten-row", "glides": "teflon"})
        with self.assertRaises(ValueError):
            tasks.resolve("classroom-arrangement", {"room": "x"})

    def test_layouts_are_conflict_free(self):
        from classroom_arrangement import layout as L
        self.assertEqual(L.conflicts(L.L0()), [])
        for t in L.TASKS:
            for seed in range(3):
                s = L.setup(t, seed)
                self.assertTrue(s["check"])
                self.assertEqual(L.setup(t, seed)["items"], s["items"])          # deterministic in the seed

    def test_group_work_places(self):
        """group-work names pod places, not furniture; its chairs start where the seed put them; edge contact anywhere."""
        from classroom_arrangement import layout as L
        from classroom_arrangement import params as P
        from classroom_arrangement.task import edge_gap, edge_span
        specs = [L.setup("group-work", seed) for seed in range(3)]
        s = specs[0]
        rc = [f"{r}{c}" for r in (1, 2) for c in range(1, 7)]
        self.assertEqual(sorted(s["check"]), sorted([f"slot_{x}" for x in rc] + [f"seat_{x}" for x in rc] +
                                                    ["pod_0", "pod_1", "pod_2", "rest_of_room", "route"]))
        self.assertEqual(sorted(s["dots"]), sorted(f"desk_{x}" for x in rc))              # 48 green dots
        self.assertEqual(s["check"]["pod_1"], ["slot_13", "slot_14", "slot_23", "slot_24"])
        chairs = [{f"chair_{x}": sp["items"][f"chair_{x}"]["pose"] for x in rc} for sp in specs]
        self.assertNotEqual(chairs[0], chairs[1])                                         # the seed varies the start
        for sp in specs:
            for x in rc:
                cx, cy, cyaw = sp["items"][f"chair_{x}"]["pose"]
                tx, ty, _ = L.L0()[f"chair_{x}"]["pose"]
                self.assertTrue(0.48 <= cy - ty <= 0.66 and abs(cx - tx) <= 0.03 and abs(math.degrees(cyaw)) <= 10)
        for t in L.TASKS:
            if t != "group-work":
                self.assertFalse(any(isinstance(v, tuple) and v[0] == "slot" for v in L.setup(t, 0)["check"].values()))
        pod = L.pod_targets(1)
        front, back = pod["desk_13"], pod["desk_23"]
        self.assertAlmostEqual(edge_gap(front, back), 0., places=9)
        self.assertAlmostEqual(edge_gap(front, (back[0], back[1] + 0.012, 0.)), 0.012, places=9)
        th = math.radians(2.0)                       # turned 2 deg off square, touching at one corner
        top = max(math.sin(math.pi + th)*a - math.cos(math.pi + th)*0.2286 for a in (-0.3048, 0.3048))
        self.assertAlmostEqual(edge_gap((front[0] - 0.004, 3.10 - 0.0005 - top, math.pi + th), back), 0.0005, places=9)
        self.assertIsNone(edge_gap(front, (back[0] + 0.70, back[1], 0.)))                # no common stretch of edge
        closest, widest = edge_span((-0.3125, 2.8564, 3.09796), (-0.3125, 3.3436, 0.04363))
        self.assertAlmostEqual(closest, 0.0038, delta=0.0002)
        self.assertGreater(widest, P.EVAL_GROUP_WORK["pod_widest"])                       # 57 mm open: not meeting
        self.assertAlmostEqual(P.EVAL_GROUP_WORK["pod_widest"], 0.010 + 0.6096*math.sin(math.radians(3)), places=4)

    def test_base_twist_codec(self):
        from gym_anything_robotics.action_codec import action_from_dict
        a = action_from_dict({"type": "base_twist", "vx_m_s": 0.2, "wz_rad_s": 0.1, "duration_s": 1.0})
        self.assertEqual((a.vx_m_s, a.vy_m_s, a.wz_rad_s, a.duration_s), (0.2, 0.0, 0.1, 1.0))
        for bad in ({"type": "base_twist"}, {"type": "base_twist", "duration_s": 1, "vx_m_s": float("nan")},
                    {"type": "base_twist", "duration_s": 1, "speed": 1}):
            with self.assertRaises(ValueError):
                action_from_dict(bad)


    def test_robot_setup_objects(self):
        import json
        from gym_anything_robotics.backends.newton_classroom_arrangement import NewtonClassroomArrangementBackend as B
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        from robots.catalog import ENVIRONMENTS
        cat = ENVIRONMENTS["classroom-arrangement"]
        root = PACKAGES
        setup = json.loads((root/"robots/rigs/classroom-mobile-fr3-single.json").read_text())
        setup["id"] = "my-single-arm"
        self.assertEqual(B._qualified_setup(setup, cat)["id"], "my-single-arm")
        for change in ({"controller_profile": "barcode"}, {"mount": "table"}):
            bad = json.loads(json.dumps(setup))
            bad["instances"]["robot"].update(change)
            with self.assertRaises(UnsupportedConfigurationError):
                B._qualified_setup(bad, cat)
        with self.assertRaises(UnsupportedConfigurationError):
            B(device="cuda:0")                                            # physics is CPU MuJoCo; refused, not ignored
        with self.assertRaises(UnsupportedConfigurationError):
            B(reflex_stops_bas=False)                                     # unknown options are refused, not a TypeError
        # the public path (gar run / registry.create / LaunchSpec) goes through the catalog validator first
        from robots.catalog import select_for_backend
        self.assertEqual(select_for_backend("classroom-arrangement", setup), "mobile-fr3-single")
        self.assertEqual(select_for_backend("classroom-arrangement", "classroom-mobile-fr3-single"), "mobile-fr3-single")
        bad = json.loads(json.dumps(setup))
        bad["instances"]["robot"]["controller_profile"] = "barcode"
        with self.assertRaises(ValueError):
            select_for_backend("classroom-arrangement", bad)
        self.assertEqual(B._min_action_s(3, 640, 360), 0.06)               # the floor scales with the render load
        self.assertEqual(B._min_action_s(1, 1280, 720), 0.08)
        self.assertEqual(B._min_action_s(0, 640, 360), 0.02)


if __name__ == "__main__":
    unittest.main()

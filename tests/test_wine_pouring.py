"""Wine pouring: discovery, robot setups, task catalogue, configuration, geometry, evaluator windows and the trace /
geometry / robot-configuration / variation contract of the backend.

CPU-only tests run by default (seconds). The backend tests build the full scene with its liquid on the GPU (about a
minute) and run only with GAR_GPU_TESTS=1:
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_wine_pouring.py
  GAR_GPU_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_wine_pouring.py
"""
import json
import os
import unittest
from pathlib import Path
from tests.packages import PACKAGES

ROOT = PACKAGES


class Discovery(unittest.TestCase):
    def test_launch_and_robots(self):
        from gym_anything_robotics.launch import describe
        from robots.catalog import for_environment, environment_robots
        d = describe("wine-pouring")
        self.assertEqual(d["default_robot"], "dual-fr3")
        self.assertEqual({r["id"] for r in environment_robots("wine-pouring")},
                         {"dual-fr3", "single-fr3", "dual-panda", "single-panda"})
        self.assertEqual(for_environment("wine-pouring").id, "pour-fr3")
        with self.assertRaises(ValueError) as cm:
            for_environment("wine-pouring", "g1")
        self.assertIn("shoulder", str(cm.exception))

    def test_task_selector_and_command_floor(self):
        from wine_pouring.variants import resolve
        from gym_anything_robotics.backends.newton_wine_pouring import min_command_s
        self.assertEqual(resolve(None, "lined-measure", 2)["preset"], "lined")        # a task alone selects its preset
        self.assertEqual(json.loads((ROOT/"wine_pouring/benchmark.json").read_text())["cli_task_selector"], "task")
        self.assertEqual(min_command_s(0), 0.0)
        self.assertGreaterEqual(min_command_s(1), 0.010)
        self.assertGreater(min_command_s(3), min_command_s(1))                    # three cameras: a longer floor

    def test_robot_setup_objects(self):
        from robots.catalog import resolve, select_for_backend
        for alias, rig in (("dual-fr3", "pour-fr3"), ("single-fr3", "pour-fr3-single"), ("dual-panda", "pour-panda")):
            d = resolve(rig).to_dict()
            self.assertEqual(select_for_backend("wine-pouring", d), alias)
            d["id"] = "my-copy"
            self.assertEqual(select_for_backend("wine-pouring", d), alias)       # content decides, not the id
        bad = json.loads((ROOT/"robots/rigs/pour-fr3.json").read_text())
        bad["instances"]["left"]["controller_profile"] = "classroom"
        with self.assertRaises(ValueError) as cm:
            select_for_backend("wine-pouring", bad)
        self.assertIn("Invalid robot setup", str(cm.exception))                # no such FR3 profile
        swapped = json.loads((ROOT/"robots/rigs/pour-fr3.json").read_text())
        swapped["instances"]["left"]["mount"], swapped["instances"]["right"]["mount"] = "right_arm_mount", "left_arm_mount"
        with self.assertRaises(ValueError) as cm:
            select_for_backend("wine-pouring", swapped)
        self.assertIn("left arm on mount 'left_arm_mount'", str(cm.exception))
        right_only = json.loads((ROOT/"robots/rigs/pour-fr3.json").read_text())
        del right_only["instances"]["left"], right_only["attachments"]["left_hand"]
        with self.assertRaises(ValueError) as cm:
            select_for_backend("wine-pouring", right_only)
        self.assertIn("left mount", str(cm.exception))

    def test_task_catalogue(self):
        from gym_anything_robotics import tasks
        from wine_pouring.config import TASKS
        for preset, names in TASKS.items():
            for t in names:
                spec = tasks.resolve("wine-pouring", {"preset": preset, "task": t, "seed": 7})
                self.assertEqual(spec.task_id, f"wine-pouring/v1/{preset}/{t}/seed-7")
                self.assertIn(spec.split, ("train", "validation", "test"))
                self.assertTrue(spec.description)
        for bad in ({"preset": "service", "task": "lined-measure"}, {"preset": "bar"}, {"seed": -1}, {"seed": 10000}):
            with self.assertRaises(ValueError):
                tasks.resolve("wine-pouring", bad)

    def test_config_rejects_unknown_and_invalid(self):
        from wine_pouring.config import Config
        with self.assertRaises(ValueError):
            Config.from_dict({"preset": "service", "colour": "red"})
        with self.assertRaises(ValueError):
            Config(preset="tasting", task="pour-glass")
        with self.assertRaises(ValueError):
            Config(seed=1.5)


class Geometry(unittest.TestCase):
    def test_datasheet_volumes(self):
        from wine_pouring.geometry import summary
        b = summary("bottle_leonor")
        self.assertAlmostEqual(b["mass_kg"], 0.498, places=3)
        self.assertAlmostEqual(b["rim_z_mm"], 310.0, places=1)
        self.assertAlmostEqual(summary("glass_pure")["capacity_mL"], 680.0, delta=0.5)
        self.assertAlmostEqual(summary("glass_iso")["capacity_mL"], 215.0, delta=10.0)
        lined = summary("glass_lined")
        self.assertAlmostEqual(lined["line_volumes_mL"]["line_175"], 175.0, delta=0.2)

    def test_lined_window_is_not_short(self):
        """UK SI 1988/120 reg. 14: a 175 mL measure may not be short; the evaluator window starts at the line."""
        from wine_pouring import task
        lo, hi = task.LINED_WINDOW_ML
        self.assertGreaterEqual(lo, 175.0)
        self.assertGreater(hi, lo)


class SplitBottle(unittest.TestCase):
    """split-bottle: the instruction states the job, not the scoring tolerances; the bottle is black glass with the
    green bottle's physics; the reward is progress from the final state."""

    def test_instruction_is_the_job(self):
        from wine_pouring.variants import task_resolve
        d = task_resolve({"preset": "service", "task": "split-bottle", "seed": 0})["description"]
        self.assertEqual(d, "Finish the bottle: share what is left in it equally between the guests' glasses at spot 1 "
                            "(far left) and spot 2, without the bottle touching either glass, and stand the bottle back "
                            "on its coaster.")
        for scoring in ("%", "mL", "within"):
            self.assertNotIn(scoring, d)

    def test_black_bottle_has_the_same_physics(self):
        import numpy as np
        from wine_pouring import geometry as GEO
        from wine_pouring.config import Config
        from wine_pouring.layout import task_layout
        from wine_pouring.render import GLASS
        item = next(i for i in task_layout(Config(preset="service", task="split-bottle", seed=0))["items"]
                    if i["name"] == "bottle")
        self.assertEqual(item["color"], "glass_black")
        self.assertIn("glass_black", GLASS)
        green, black = GEO.get("bottle_leonor", "glass_green"), GEO.get("bottle_leonor", item["color"])
        for a, b in ((green.solid, black.solid), (green.interior, black.interior), (green.rigid, black.rigid)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(green.mass_properties(), black.mass_properties()):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_progress_from_the_final_state(self):
        from wine_pouring.task import split_progress
        v0, leave = 320.776, 15.
        T = .5*(v0 - leave)
        p = lambda va, vb, lost=0., up=(True, True): split_progress(v0, va, vb, up[0], up[1], lost, leave)["progress"]
        self.assertEqual(p(0., 0.), 0.)                                  # untouched
        self.assertAlmostEqual(p(T, 0.), .5)                             # one guest served, the rest in the bottle
        self.assertAlmostEqual(p(0., 2*T), .5)                           # everything in one glass
        self.assertAlmostEqual(p(T, 0., lost=T), .25)                    # one share served, the other spilt
        self.assertEqual(p(0., 0., lost=v0 - 10.), 0.)                   # the bottle wasted on the bar
        self.assertEqual(p(155.4, 155.4), 1.)                            # a finished equal split
        self.assertAlmostEqual(p(155.4, 155.4, up=(True, False)), .5)    # a fallen glass serves nobody
        self.assertEqual(p(146.4, 166.4), p(166.4, 146.4))               # either glass, any order
        self.assertGreater(p(146.4, 166.4), p(100., 100.))               # a near miss above two-thirds shared
        # wine beyond 2T is credited only while the glasses stay equal
        low = lambda va, vb: split_progress(260., va, vb, True, True, 0., leave)["progress"]
        self.assertLess(low(122.5, 137.5), 1.)                           # 10.9 % apart, fails `shared`
        self.assertEqual(low(130., 130.), 1.)                            # emptied and equal
        self.assertLess(p(164., 147.), p(162.5, 147.))                   # beyond 2T, the more unequal ranks lower

    def test_verify_hold_reveals_damage_never_adds_service(self):
        """Evaluator.verify()'s branch logic on a stub environment (no simulator)."""
        from types import SimpleNamespace
        from wine_pouring import task as T

        def run(held_reads, fault_at=None):
            env = SimpleNamespace(config=SimpleNamespace(task="split-bottle", rigid_dt=1e-3), trace_callback=None, restored=[],
                                  world=SimpleNamespace(fault=None, gas=None, liquid=None, lumped=None, phantom_log=[],
                                                        _prev_cvel={7: 1.0}))
            env.robot_still_s = lambda: 9.
            env.checkpoint = lambda: "snap"
            env.restore = lambda s: env.restored.append(s)

            def act(targets, ticks):
                if fault_at is not None and len(env.restored) == 0 and act.n == fault_at:
                    env.world.fault = "FloatingPointError: stub"
                act.n += 1
            act.n = 0
            env.act = act
            ev = T.Evaluator.__new__(T.Evaluator)
            ev.env = env
            reads = iter(held_reads)
            ev.evaluate = lambda: next(reads)
            return ev.verify(), env

        mk = lambda ok, settled, prog, left: dict(success=ok, score=1. if ok else .9*prog, checks=dict(settled=settled),
                                                  reward=dict(progress=prog), meta=dict(left_mL=left))
        request = mk(False, False, .2, 246.8)               # mid-pour: not at rest
        out, env = run([request, mk(False, True, .5, 154.8), mk(False, True, .5, 154.8)])
        self.assertAlmostEqual(out["score"], .9*.2)         # the frozen pour's extra wine is not credited
        self.assertEqual(out["reward"]["progress_source"], "request")
        self.assertEqual(env.restored, ["snap"])
        self.assertEqual(env.world._prev_cvel, {7: 1.0})
        out, _ = run([request, mk(False, True, .1, 246.8), mk(False, True, .1, 246.8)])
        self.assertAlmostEqual(out["score"], .9*.1)         # damage the hold reveals is kept
        out, _ = run([request, mk(True, True, 1., 8.), mk(True, True, 1., 8.)])
        self.assertEqual(out["score"], 1.)                  # a correct finish read promptly
        out, _ = run([request] + [mk(False, False, .2, 246.8)]*40, fault_at=0)
        self.assertTrue(out["read"]["infrastructure_fault"])
        self.assertIs(out, request)                         # the live read, flagged, not a held verdict

    def test_touch_rule_is_split_bottle_only(self):
        from wine_pouring import params as P
        from wine_pouring.config import TASKS
        from wine_pouring.task import touch_rule, TOUCH_FLOOR_N
        self.assertEqual(touch_rule("split-bottle"), (1, TOUCH_FLOOR_N))          # every tick, any touch
        for t in (t for names in TASKS.values() for t in names if t != "split-bottle"):
            self.assertEqual(touch_rule(t), (10, P.CONTACT_FORCE_TOL_N))         # the other nine: every 10th tick


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "GPU scene build; set GAR_GPU_TESTS=1")
class Backend(unittest.TestCase):
    """The public path without rendering (cameras=()), through the registry."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-wine-test-")
        cls.env = registry.create("wine-pouring", preset="service", task="pour-glass", seed=0, cameras=(),
                                  output_dir=cls.tmp.name + "/s")
        cls.backend = cls.env._backend

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.tmp.cleanup()

    def test_robot_configuration(self):
        rc = self.backend.robot_configuration()
        self.assertEqual(set(rc["instances"]), {"left", "right"})
        self.assertEqual(rc["setup_id"], "pour-fr3")
        self.assertTrue(rc["setup_fingerprint"])

    def test_geometry_routes(self):
        b = self.backend
        index = b.get_geometry_index()
        self.assertIn("bottle", index["items"])
        g = b.get_geometry("bottle")
        self.assertAlmostEqual(g["mass_kg"], 0.498, places=3)
        self.assertGreater(len(g["liquid_space_polygon"]), 10)
        self.assertIn("wall_polygon", b.get_geometry_asset("profile/bottle_leonor"))
        with self.assertRaises(KeyError):
            b.get_geometry("no-such-item")

    def test_fingers_share_one_hardware_actuator(self):
        from gym_anything_robotics.actions import NamedJointTarget
        b = self.backend
        b.reset()
        fingers = b.env.world.gripper_joints("right")
        try:
            step = b.act_policy(NamedJointTarget({fingers[0]: 0.0, fingers[1]: 0.04}, 100))
            q = b.env.world.state.joint_q.numpy()
            widths = [float(q[b.env.world.joints[name]["q"]]) for name in fingers]
            self.assertEqual(step.info["executed_ticks"], 100)
            self.assertEqual(step.info["refused_targets"], 0)
            self.assertAlmostEqual(widths[0], widths[1], places=6)
        finally:
            b.reset()

    def test_trace_contract_and_rejections(self):
        from gym_anything_robotics.actions import NamedJointTarget, Gripper
        from gym_anything_robotics.errors import ActionRejectedError
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        b = self.backend
        ref = b.begin_action_trace({"action_id": "t-1", "request_id": "r-1"})
        self.assertTrue(ref["available"])
        b.act(NamedJointTarget({"left/fr3_joint1": 0.02}, 20))
        b.end_action_trace()
        store = PhysicsTraceStore(Path(ref["store_path"]), readonly=True)
        try:
            page = store.page("t-1", limit=4)
            self.assertTrue(page["summary"]["complete"])
            self.assertEqual(page["summary"]["retained_samples"], 20)
            self.assertIn("liquid_step", page["items"][0]["data"])
            self.assertIn("liquid_wrench_on_items", page["items"][0]["data"])
        finally:
            store.close()
        with self.assertRaises(ActionRejectedError):
            b.act(Gripper("left", inward_effort_n=200.0, duration_s=0.1))      # beyond the hand's 70 N
        view = b.policy_access()
        self.assertFalse(any("bottle" in " ".join(k) for k in view.proprioception))   # no privileged object state

    def test_variation_selection(self):
        sel = self.backend.select_variation({"preset": "service", "task": "taste-pour", "seed": 3})
        self.assertEqual(sel["reset_request"], {"seed": 3, "options": {"preset": "service", "task": "taste-pour"}})
        with self.assertRaises(ValueError):
            self.backend.select_variation({"robot": "g1"})

    def test_checkpoint_scoped_to_its_variation(self):
        b = self.backend
        b.save_checkpoint(label="here")
        self.assertTrue(all(c["restorable"] for c in b.list_checkpoints()))
        saved = b._checkpoints["here"]["variation"]
        b._checkpoints["here"]["variation"] = dict(saved, task="taste-pour")     # as if saved before a variation change
        with self.assertRaises(ValueError):
            b.restore_checkpoint("here")
        b._checkpoints["here"]["variation"] = saved
        b.restore_checkpoint("here")

    def test_refused_restore_and_reset_keep_service_ready(self):
        """A restore into another variation and a reset with an unknown preset are refused before the world is
        touched: the HTTP service must stay ready."""
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPService
        b = self.backend
        service = GARHTTPService(RoboticsEnv(b))
        b.save_checkpoint(label="elsewhere")
        b._checkpoints["elsewhere"]["variation"] = dict(b._checkpoints["elsewhere"]["variation"], task="taste-pour")
        with self.assertRaises(ValueError):
            service.restore_checkpoint({"checkpoint": "elsewhere"})
        self.assertTrue(service.health()["ready"])
        with self.assertRaises(ValueError):
            service.reset({"options": {"preset": "no-such-preset"}})
        self.assertTrue(service.health()["ready"])
        del b._checkpoints["elsewhere"]

    def test_min_command_published(self):
        self.assertIn("min_command_s", self.backend.capabilities.notes)


if __name__ == "__main__":
    unittest.main()

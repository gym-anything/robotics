"""Surface rinsing: discovery, catalogue, robot validation and water-model units without the rigid world; the backend's
public path (registry, actions, state, evaluation with the settle rule, refusals, checkpoints) with a scene build when
GAR_GPU_TESTS=1 (CPU rigid physics, the water engine on the GPU; cameras=() so no render worker).

python -m unittest tests.test_surface_rinsing
GAR_GPU_TESTS=1 python -m unittest tests.test_surface_rinsing
"""
import math
import os
import unittest

import numpy as np


class Discovery(unittest.TestCase):
    def test_describe_and_robots(self):
        from gym_anything_robotics.launch import describe
        d = describe("surface-rinsing")
        self.assertEqual(d["default_robot"], "mobile-fr3-duo")
        self.assertEqual({r["id"] for r in d["robots"]}, {"mobile-fr3-duo", "mobile-fr3-single"})
        for alias in ("surface_rinsing", "rinse-surfaces", "dish-rinsing"):
            self.assertEqual(describe(alias)["environment_id"], "surface-rinsing")

    def test_catalogue(self):
        from surface_rinsing.layout import TASKS, PRESETS
        from surface_rinsing.variants import task_resolve, resolve, split_of
        r = task_resolve(dict(task="rinse-tomatoes", seed=7))
        self.assertEqual(r["task_id"], "surface-rinsing/v1/prep-sink/rinse-tomatoes/seed-7")
        self.assertIn(r["split"], ("train", "validation", "test"))
        self.assertEqual(len(TASKS), 8)
        self.assertEqual(sum(len(v) for v in PRESETS.values()), 8)
        self.assertEqual(resolve(preset="can-wash")["task"], "rinse-garbage-can")
        with self.assertRaises(ValueError):
            task_resolve(dict(task="rinse-tomatoes", preset="pot-sink"))
        with self.assertRaises(ValueError):
            task_resolve(dict(task="flush-floor", seed=10000))
        self.assertIsNone(split_of("flush-floor", 5000))                  # only seeds 0..999 are in the frozen split

    def test_catalogue_file_matches(self):
        import hashlib
        import json
        from surface_rinsing.variants import CATALOGUE, split_of
        from surface_rinsing.layout import TASKS
        data = json.loads(CATALOGUE.read_text())
        rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
        self.assertEqual(data["count"], 8000)
        self.assertEqual(data["sha256"], hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16])

    def test_robot_validation(self):
        from surface_rinsing.robot_setup import validate, check_task
        self.assertEqual(validate("mobile-fr3-duo"), "mobile-fr3-duo")
        self.assertEqual(validate("rinsing-mobile-fr3-single"), "mobile-fr3-single")
        for bad in ("g1", "dual-fr3", "single-fr3", "dual-panda"):
            with self.assertRaises(ValueError):
                validate(bad)
        with self.assertRaises(ValueError):
            check_task("mobile-fr3-single", "rinse-garbage-can")           # tipping the can needs two hands
        check_task("mobile-fr3-single", "rinse-sponge")

    def test_seeds_vary_what_they_say(self):
        from surface_rinsing.layout import setup
        a, b = setup("scrub-rinse-pan", 0), setup("scrub-rinse-pan", 3)
        self.assertEqual(a["items"]["pan"]["soil"]["soaked_s"], 120.0)
        self.assertTrue(20.0 <= b["items"]["pan"]["soil"]["soaked_s"] <= 240.0)
        self.assertNotEqual(a["supply"]["p_static"], b["supply"]["p_static"])
        for seed in range(20):
            sp = setup("rinse-glassware", seed)
            self.assertEqual(len([n for n in sp["items"] if n.startswith("glass_") and n != "glass_rack"]), 6)
            self.assertTrue(40*6894 < sp["supply"]["p_static"] < 80.1*6895)


class WaterModel(unittest.TestCase):
    def test_parameter_ledger(self):
        from surface_rinsing import params as P
        names = [e["name"] for e in P.LEDGER]
        self.assertEqual(len(names), len(set(names)))
        for e in P.LEDGER:
            self.assertTrue(e["label"] and e["basis"], e["name"])
        self.assertAlmostEqual(P.PRSV["Q_rated"]/P.GPM, 1.15, places=6)
        # the DOE spray force of the modelled valve: rho Q Cv sqrt(2 p / rho) = 2.0-2.1 N (5-8 ozf class)
        v = P.PRSV["Cv"]*math.sqrt(2*P.PRSV["p_rated"]/P.WATER["rho"])
        self.assertTrue(1.9 < P.WATER["rho"]*P.PRSV["Q_rated"]*v < 2.2)

    def test_valve_laws(self):
        from surface_rinsing.water.network import smoothstep
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertAlmostEqual(smoothstep(0.5), 0.5)

    def test_gauges_separate_the_faces_of_a_thin_wall(self):
        from surface_rinsing.task import gauges
        # two 1 cm2 patches 5 mm apart facing opposite ways (a plate's front and back): soil on the front only
        pos = np.array([[0.005, 0.005, 0.0025], [0.005, 0.005, -0.0025]])
        nrm = np.array([[0., 0., 1.], [0., 0., -1.]])
        area = np.array([1e-4, 1e-4])
        soil = np.array([2e-5, 0.0])
        self.assertAlmostEqual(float(gauges(pos, area, soil, nrm).max()), 2e-5)

    def test_pool_foam_collapses_and_leaves_with_overflow(self):
        from surface_rinsing import params as P
        from surface_rinsing.water.pools import Pool
        p = Pool("t", "static", lo=(0, 0, 0), hi=(0.4, 0.6, 0.3))
        p.vol, p.sp[0] = 1e-3, 0.0
        p.sp[4], p.sp[5] = 1e-5, 1e-8                      # foam liquid and its surfactant on clean water
        tot_w, tot_s = p.vol + p.sp[4], p.sp[0] + p.sp[5]
        for _ in range(4000):
            p.foam_decay(P.DT)                             # 2 s: collapses at k_dil = 1 /s on dilute water
        self.assertLess(p.sp[4], 0.2e-5)
        self.assertAlmostEqual(p.vol + p.sp[4], tot_w, places=15)
        self.assertAlmostEqual(p.sp[0] + p.sp[5], tot_s, places=18)
        q = Pool("u", "static", lo=(0, 0, 0), hi=(0.4, 0.6, 0.3))
        q.vol, q.sp[4], q.sp[5] = 1e-3, 1e-5, 1e-8
        v, out = q.take(q.vol, with_foam=True)             # emptied over the rim: the foam goes with it
        self.assertEqual(q.sp[4], 0.0)
        self.assertAlmostEqual(out[4], 1e-5)

    def test_sponge_static_law(self):
        from scipy.optimize import brentq
        from surface_rinsing import params as P
        S = P.SPONGE
        k = S["E_wet"]*S["A_eff"]
        eps = lambda F: brentq(lambda e: k*(e + 3*e**3) - F, 0.0, 2.0)
        self.assertAlmostEqual(eps(60.0), 0.31, delta=0.02)               # 0.31 at 60 N (3.73 mm of 12)
        self.assertGreater(1.0 - eps(70.0), P.CRITERIA["sponge_wrung"])   # the robot's 70 N pinch cannot wring it
        self.assertLess(1.0 - eps(120.0), P.CRITERIA["sponge_wrung"])     # a 120 N hand squeeze can


class ScrubRinsePanContract(unittest.TestCase):
    """The scrub-rinse-pan score contract on constructed records."""

    def _score(self, clean=1.0, rinsed=1.0, san_s=30.0, fail=(), fault=None, water_res=0.0):
        from types import SimpleNamespace
        from surface_rinsing.task import Evaluator, PanTracker, PAN_TASK
        gates = ("visibly_clean", "soap_free", "sanitized", "drained", "placed", "sponge_on_ledge", "faucet_closed",
                 "hands_empty")
        crit = {k: dict(passed=k not in fail) for k in gates}
        led = dict(relative=water_res, relative_surf=0.0, surfactant_kg=0.0, relative_solids=0.0, solids_kg=0.0)
        m = dict(pan=dict(clean=clean, rinsed=rinsed), items=dict(pan=dict(sanitized_s=san_s)), fault=fault, ledger=led)
        ev = SimpleNamespace(task=PAN_TASK, pan=SimpleNamespace(progress=lambda m, c: PanTracker.progress(None, m, c)))
        return Evaluator._score(ev, crit, m)

    def test_contract(self):
        self.assertEqual(self._score()[0], 1.0)
        self.assertEqual(self._score()[1]["contract"], "progress-v1")
        self.assertEqual(self._score(clean=0.0, rinsed=0.0, san_s=0.0, fail=("visibly_clean", "soap_free", "sanitized",
                                                                            "placed"))[0], 0.0)   # untouched start
        self.assertEqual(self._score(fail=("sponge_on_ledge",))[0], 0.9)                          # only the sponge out
        self.assertEqual(self._score(san_s=29.0, fail=("sanitized",))[0], 0.885)                  # 29 s, shelved
        self.assertEqual(self._score(fail=("placed",))[0], 0.675)                                 # sanitized, not stored
        # a pan with every progress term at 1 that fails soap_free (a residual patch) is not stored: as unshelved
        self.assertEqual(self._score(fail=("soap_free",))[0], 0.675)
        score, rw = self._score(water_res=1e-3)                                                  # open ledger
        self.assertIsNone(score)
        self.assertFalse(rw["verifiable"])
        self.assertIn("open ledger", rw["unverifiable_reason"])


class RendererReleased(unittest.TestCase):
    """A command while the cameras' render worker is released is refused with that reason, which the gateway shows the
    policy, and the policy's help states the rule. Without a policy_reason the policy reads only "adjust the target or
    command parameters". No scene is built."""

    @staticmethod
    def _backend(**options):
        import threading
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_surface_rinsing import NewtonSurfaceRinsingBackend
        from gym_anything_robotics.types import CameraSpec
        b = object.__new__(NewtonSurfaceRinsingBackend)
        b.cameras, b._renderer, b.options, b._render_lock = ("head_rgb",), None, dict(options), threading.RLock()
        b.env = SimpleNamespace(lock=threading.RLock(), spec={"instruction": "rinse"},
                                world=SimpleNamespace(joints={}, base_joint_names=()))
        b.arms, b.robot_name, b.kinds, b.keyboard_keys = ("left", "right"), "mobile-fr3-duo", frozenset({"base_twist"}), ()
        b.camera_specs = {"head_rgb": CameraSpec("head_rgb", "head", 640, 360, 1280, 720, (.036, .02), (.02, .02))}
        b.capabilities = SimpleNamespace(notes={k: k for k in ("base", "spine", "gripper", "arms", "protective_stop",
                                                               "start")})
        return b

    def _policy_error(self, error):
        from gym_anything_robotics.http_server import GARHTTPHandler
        handler = object.__new__(GARHTTPHandler)
        handler.principal = type("Policy", (), {"restricted": True})()
        sent = []
        handler._send_json = lambda value, status=200: sent.append((status, value))
        handler._error(400, error)
        return sent[0]

    def test_refusal_says_why_to_the_policy(self):
        from gym_anything_robotics.actions import BaseTwist
        from gym_anything_robotics.errors import ActionRejectedError
        for options, rule in (({}, "after 300 s without an image"), ({"idle_release_s": 60}, "after 60 s without an image")):
            with self.subTest(options=options):
                with self.assertRaises(ActionRejectedError) as cm:
                    self._backend(**options).act(BaseTwist(0.0, 0.0, 0.0, duration_s=0.5))
                reason = cm.exception.policy_reason
                self.assertLessEqual(len(reason), 400)                    # the gateway shows at most 400 characters
                status, body = self._policy_error(cm.exception)
                self.assertEqual(status, 400)
                self.assertIn(rule, body["error"])
                self.assertIn("not about the command's target or parameters", body["error"])
                self.assertIn("GET /v1/observation", body["error"])

    def test_policy_help_states_the_rule(self):
        hardware = self._backend().policy_access().capabilities()["hardware"]
        self.assertIn("after 300 s without an image", hardware["camera_renderer"])
        self.assertIn("a refused command draws none", hardware["camera_renderer"])
        b = self._backend()
        b.cameras = ()
        self.assertNotIn("camera_renderer", b.policy_access().capabilities()["hardware"])


class PlannerRefusals(unittest.TestCase):
    """A command the inherited classroom planner refuses (IK reach, duration, base held, ...) tells the policy why: the
    controller's own text is its policy_reason. Without it the policy reads only "adjust the target or command
    parameters". No scene is built; the IK is a stand-in that refuses with ArmKinematics.solve's reach text."""

    REACH = "IK unreachable for left: target 4.53 m from the arm base, reach 1.27 m; no action executed"

    def _backend(self, base_held=False):
        from types import SimpleNamespace
        b = RendererReleased._backend()
        b._renderer = object()                                     # the worker is running: the planner decides
        b.min_action_s = 0.02
        b.env.world.base_held = lambda: base_held
        def solve(*args, **kwargs):
            raise ValueError(self.REACH)
        b.ik = SimpleNamespace(commanded=lambda arm: None, solve=solve)
        return b

    def test_refusal_says_why_to_the_policy(self):
        from gym_anything_robotics.actions import BaseTwist, CartesianPose
        from gym_anything_robotics.errors import ActionRejectedError
        cases = ((CartesianPose("left", (0.45, 0.45, 1.1), (0., 1., 0., 0.), duration_s=2.0), False, self.REACH),
                 (CartesianPose("left", (0., 0., 1.), duration_s=0.005), False,
                  "duration_s must be in [0.02, 30.0] s (this session's minimum: 0.02 s)"),
                 (BaseTwist(0.1, 0., 0., duration_s=0.5), True, "Base command refused: an arm protective stop holds the base"))
        for action, held, text in cases:
            with self.subTest(action=type(action).__name__, text=text):
                with self.assertRaises(ActionRejectedError) as cm:
                    self._backend(base_held=held).act(action)
                self.assertIn(text, cm.exception.policy_reason)
                status, body = RendererReleased._policy_error(self, cm.exception)
                self.assertEqual(status, 400)
                self.assertIn("Reason: " + text, body["error"])

    def test_the_classroom_backend_is_unchanged(self):
        from gym_anything_robotics.actions import BaseTwist
        from gym_anything_robotics.backends.newton_classroom_arrangement import NewtonClassroomArrangementBackend
        from gym_anything_robotics.errors import ActionRejectedError
        with self.assertRaises(ActionRejectedError) as cm:
            NewtonClassroomArrangementBackend.act(self._backend(), BaseTwist(0., 0., 0., duration_s=0.005))
        self.assertIsNone(getattr(cm.exception, "policy_reason", None))


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "builds the scene (set GAR_GPU_TESTS=1)")
class PublicPath(unittest.TestCase):
    def test_scrub_rinse_pan_step(self):
        import tempfile
        from gym_anything_robotics import registry
        from gym_anything_robotics.actions import BaseTwist
        tmp = tempfile.TemporaryDirectory()
        env = registry.create("surface-rinsing", task="scrub-rinse-pan", seed=0, cameras=(), output_dir=tmp.name + "/p")
        b = env._backend
        env.reset()
        step = b.act(BaseTwist(0.0, 0.0, 0.0, duration_s=0.2))
        self.assertEqual(step.reward, 0.0)                              # the untouched start earns nothing
        self.assertIs(step.info["verifiable"], True)                    # the step says whether its score is a result
        rw = step.state["evaluation"]["reward"]
        self.assertEqual(rw["contract"], "progress-v1")
        self.assertNotIn("unverifiable_reason", rw)
        env.close()

    def test_backend(self):
        import tempfile
        from gym_anything_robotics import registry
        from gym_anything_robotics.actions import BaseTwist, Gripper
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("surface-rinsing", task="rinse-glassware", robot="g1", cameras=())
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("surface-rinsing", task="rinse-garbage-can", robot="mobile-fr3-single", cameras=())
        tmp = tempfile.TemporaryDirectory()
        env = registry.create("surface-rinsing", task="rinse-sponge", seed=1, cameras=(), output_dir=tmp.name + "/s")
        b = env._backend
        env.reset()
        b.act(BaseTwist(0.0, 0.0, 0.0, duration_s=0.5))
        st = b.env.get_state()
        self.assertIn("no_suds", st["evaluation"]["criteria"])
        self.assertFalse(st["evaluation"]["succeeded"])               # the sponge starts soapy and full
        self.assertLess(b.env.water.ledger_residual()["relative"], 1e-9)
        cp = b.save_checkpoint(label="a")
        b.act(Gripper(sorted(b.env.world.arms)[0], half_width_m=0.04, duration_s=0.5))
        b.restore_checkpoint(label="a")
        self.assertEqual(b.env.world.tick, cp["tick"])
        self.assertLess(b.env.water.ledger_residual()["relative"], 1e-9)
        ev = b.evaluate()                                              # the settle branch (30 s), then restored
        self.assertFalse(ev["success"])
        self.assertEqual(b.env.world.tick, cp["tick"])
        env.close()


if __name__ == "__main__":
    unittest.main()

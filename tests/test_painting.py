"""Painting: discovery, catalogue, robot validation and paint-model units without the rigid world; the backend's public
path (registry, actions, state, evaluation with the settle rule, refusals, checkpoints) with a scene build when
GAR_GPU_TESTS=1 (CPU physics; cameras=() so no render worker).

python -m unittest tests.test_painting
GAR_GPU_TESTS=1 python -m unittest tests.test_painting
"""
import os
import unittest

import numpy as np


class Discovery(unittest.TestCase):
    def test_describe_and_robots(self):
        from gym_anything_robotics.launch import describe
        d = describe("painting")
        self.assertEqual(d["default_robot"], "mobile-fr3-duo")
        self.assertEqual({r["id"] for r in d["robots"]}, {"mobile-fr3-duo", "mobile-fr3-single", "g1"})
        self.assertEqual(describe("apply-paint")["environment_id"], "painting")

    def test_catalogue(self):
        from gym_anything_robotics import tasks
        from painting.layout import TASKS
        from painting.variants import task_resolve
        cat = tasks.catalog("painting") if hasattr(tasks, "catalog") else None
        r = task_resolve(dict(task="spray-stencil", seed=7))
        self.assertEqual(r["task_id"], "painting/v1/spray/spray-stencil/seed-7")
        self.assertIn(r["split"], ("train", "validation", "test"))
        self.assertEqual(len(TASKS), 11)
        with self.assertRaises(ValueError):
            task_resolve(dict(task="fill-circle", preset="wall"))
        with self.assertRaises(ValueError):
            task_resolve(dict(task="roll-strip", seed=10000))

    def test_no_seed_repaints_the_same_colour(self):
        from painting.layout import setup
        for seed in range(300):
            sp = setup("cut-in-ceiling", seed)
            self.assertNotEqual(sp["old_color"], sp["paint"]["color"], seed)


class PaintModel(unittest.TestCase):
    def test_rheology_targets(self):
        from painting.paint import make_paint
        pt = make_paint("latex", "alabaster")
        self.assertTrue(0.1 < pt.eta(1e4, 0.0) < 0.3)              # brushing/rolling viscosity
        self.assertGreater(pt.eta(0.1, 1.0), 50.0)                 # at rest
        self.assertAlmostEqual(pt.tau_y(1.0), 2.0, places=6)
        self.assertEqual(pt.tau_y(0.0), 0.0)

    def test_film_conserves_and_dries(self):
        from painting.paint import make_paint
        from painting.film import Chart
        pt = make_paint("latex", "lullaby")
        ch = Chart("t", [pt], origin=(0, 0, 0), u=(1, 0, 0), v=(0, 0, 1), size=(0.04, 0.04), cell=2e-3)
        I, J = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
        h0 = 100e-6
        ch.add(I.ravel(), J.ravel(), np.full((1, 400), h0*pt.phi0), np.full(400, h0*(1 - pt.phi0)))
        s0 = float(ch.s.sum())
        for _ in range(300):
            ch.advance(2.0)
        self.assertAlmostEqual(float(ch.s.sum() + ch.d.sum())/s0, 1.0, places=9)    # solids never evaporate
        self.assertFalse((ch.h() > 0).any())                                         # set within 10 min
        self.assertAlmostEqual(float(ch.dry_total().mean()), h0*pt.phi0, delta=1e-9)

    def test_km_hiding(self):
        from painting.paint import make_paint, km_layer
        pt = make_paint("latex", "extra_white")
        Y = lambda R: float(R @ np.array([0.2126, 0.7152, 0.0722]))
        cr = Y(km_layer(pt.K, pt.S, 44, np.zeros(3)))/Y(km_layer(pt.K, pt.S, 44, np.full(3, 0.8)))
        self.assertGreater(cr, 0.98)                               # CR 0.98 at one coat (43 um dry)
        cr = Y(km_layer(pt.K, pt.S, 30, np.zeros(3)))/Y(km_layer(pt.K, pt.S, 30, np.full(3, 0.8)))
        self.assertLess(cr, 0.98)

    def test_hide_is_colour_and_base(self):
        """The evaluator's hide criterion (dE*ab <= 1 to the paint's own colour) agrees with CR 0.98 for the white base
        at one coat, and deep-base navy (no TiO2: lower scattering) is mixed in the deep base."""
        from painting import params as P
        from painting.paint import make_paint, stack_reflectance, delta_e, srgb_to_linear, tint_base
        white = make_paint("latex", "extra_white")
        self.assertEqual(tint_base(white.rinf)[0], "white")
        navy = make_paint("latex", "naval")
        self.assertEqual(tint_base(navy.rinf)[0], "deep")
        self.assertAlmostEqual(float(navy.S[0]/white.S[0]), P.TINT_BASES["deep"][1], places=6)
        beige = srgb_to_linear(P.COLORS["accessible_beige"])
        d = np.array([[45e-6]])
        R = stack_reflectance([white], d, np.zeros_like(d), np.zeros(1), beige[None])
        self.assertLess(float(delta_e(R, white.rinf[None])[0]), P.EVAL["hide_dE"])
        # a deep colour needs two coats over the grey primer and three over white (one 43 um coat each)
        grey, prim = srgb_to_linear(P.COLORS["pshade_primer"]), srgb_to_linear(P.COLORS["primer"])
        de = lambda um, sub: float(delta_e(stack_reflectance([navy], np.array([[um*1e-6]]), np.zeros((1, 1)), np.zeros(1),
                                                             sub[None]), navy.rinf[None])[0])
        self.assertGreater(de(43, grey), 1.0); self.assertLess(de(86, grey), 1.0)
        self.assertGreater(de(86, prim), 1.0); self.assertLess(de(129, prim), 1.0)
        d = np.array([[15e-6]])
        R = stack_reflectance([white], d, np.zeros_like(d), np.zeros(1), beige[None])
        self.assertGreater(float(delta_e(R, white.rinf[None])[0]), P.EVAL["hide_dE"])

    def test_gauge_and_stroke_template(self):
        from painting.task import _gauge, stroke_width
        rng = np.random.default_rng(0)
        dft = 30e-6*(1 + 0.3*rng.standard_normal((100, 80)))           # parcel speckle around an even 30 um coat
        m = np.zeros((100, 80), bool); m[20:80, 20:60] = True
        g = _gauge(dft, m, 1e-3)[m]
        self.assertLess(float(g.max()/g.min()), 2.0)                     # 5 mm gauge readings of an even coat
        self.assertGreater(float(dft[m].max()/max(dft[m].min(), 1e-9)), 2.0)
        self.assertAlmostEqual(float(stroke_width(0.0)), 2e-3)
        self.assertAlmostEqual(float(stroke_width(0.5)), 6e-3)

    def test_deep_colour_over_tinted_primer(self):
        from painting.layout import setup, DEEP_COLORS
        for seed in range(60):
            sp = setup("roll-strip", seed)
            if sp["paint"]["color"] in DEEP_COLORS:
                self.assertEqual(sp["old_color"], "pshade_primer", seed)
                self.assertGreater(sp["first_coat"], 0.0)                 # the finish coat over a first coat
                self.assertIn("second, finish coat", sp["instruction"])

    def test_brush_stroke_instruction_matches_template(self):
        from painting.layout import TASKS
        from painting.task import stroke_width
        self.assertAlmostEqual(float(stroke_width(0.5)), 6e-3)
        self.assertIn("6 mm", TASKS["brush-stroke"]["instruction"])
        self.assertNotIn("4 mm", TASKS["brush-stroke"]["instruction"])

    def test_film_moves_conserve(self):
        """Corner bead, contact-skin split, spray splat and drying-only steps move or set paint without creating it."""
        from painting.paint import make_paint
        from painting.film import Chart
        pt = make_paint("latex", "lullaby")
        mask = np.ones((30, 30), bool); mask[10:20, 10:20] = False        # a footprint: inside corners round it
        ch = Chart("t", [pt], origin=(0, 0, 0), u=(1, 0, 0), v=(0, 0, 1), size=(0.06, 0.06), cell=2e-3, mask=mask)
        I, J = np.nonzero(mask & (np.arange(30)[None, :] < 9))
        ch.add(I, J, np.full((1, len(I)), 40e-6), np.full(len(I), 60e-6))
        tot = lambda: float(ch.s.sum() + ch.w.sum() + ch.d.sum())
        t0 = tot()
        moved = ch.bead_into_corner(I, J)
        self.assertEqual(moved, 0.0)  # a chart edge alone is not an inside corner
        self.assertAlmostEqual(tot()/t0, 1.0, places=12)
        ds, dv = ch.take_frac(I[:20], J[:20], 0.5)
        self.assertAlmostEqual((tot()*ch.area + float(ds.sum()) + dv)/(t0*ch.area), 1.0, places=12)
        S0 = float(ch.s.sum() + ch.d.sum())
        ch.dry_only(600.0)
        self.assertAlmostEqual(float(ch.s.sum() + ch.d.sum())/S0, 1.0, places=12)

    def test_sink_area_counts_cells_not_strokes(self):
        from painting.system import PaintSystem
        sysm = PaintSystem.__new__(PaintSystem)
        sysm.reservoirs, sysm.sinks, sysm.paints = {}, {}, [None]
        pts = np.array([[0.1, 0.2, 0.3]])
        for _ in range(50):                                            # the same spot brushed 50 times
            PaintSystem.to_sink(sysm, "radiator", np.array([2e-9]), 1e-9, points=pts)
        self.assertAlmostEqual(sysm.sinks["radiator"]["area"], PaintSystem.SINK_VOXEL**2)

    def test_longreach_grasp(self):
        """The long-reach brush is held at its handle's end (grip centre at the TCP), its handle along TCP x, the
        approach on the side the head bends to, turned 10 deg about the handle."""
        from painting import params as P
        from painting import tools as T
        from painting.env import HOLD
        h = HOLD["longreach"]
        B = P.BRUSHES["longreach"]
        a = np.radians(B["dogleg_deg"])
        g = np.array([0., 0., -B["ferrule"][2]]) + T.dogleg_dir("longreach")*(B["handle_len"] - B["handle_r"])
        np.testing.assert_allclose(h["R"] @ g + np.asarray(h["p"]), 0.0, atol=1e-12)
        np.testing.assert_allclose(h["R"] @ (-T.dogleg_dir("longreach")), [1., 0., 0.], atol=1e-12)
        bend = np.array([0., -np.cos(a), np.sin(a)])                   # brush frame: across the handle, toward the bend
        self.assertAlmostEqual(float((h["R"] @ bend)[2]), float(np.cos(np.radians(10.0))), places=12)
        self.assertAlmostEqual(float(np.linalg.det(h["R"])), 1.0, places=12)


class RendererReleased(unittest.TestCase):
    """A command while the cameras' render worker is released is refused with that reason, which the gateway shows the
    policy, and the policy's help states the rule (without it the policy reads "adjust the target or command
    parameters"). No scene is built."""

    @staticmethod
    def _backend(cls=None, **options):
        import threading
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_painting import NewtonPaintingBackend
        from gym_anything_robotics.types import CameraSpec
        b = object.__new__(cls or NewtonPaintingBackend)
        b.cameras, b._renderer, b.options, b._render_lock = ("head_rgb",), None, dict(options), threading.RLock()
        b.env = SimpleNamespace(lock=threading.RLock(), spec={"instruction": "paint"},
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
        from gym_anything_robotics.actions import BaseTwist, NamedJointTarget
        from gym_anything_robotics.backends.newton_painting_g1 import NewtonPaintingG1Backend
        from gym_anything_robotics.errors import ActionRejectedError
        for cls, action in ((None, BaseTwist(0.0, 0.0, 0.0, duration_s=0.5)),
                            (NewtonPaintingG1Backend, NamedJointTarget({}, 1000))):
            for options, rule in (({}, "after 300 s without an image"), ({"idle_release_s": 60}, "after 60 s without")):
                with self.subTest(cls=cls, options=options):
                    with self.assertRaises(ActionRejectedError) as cm:
                        self._backend(cls, **options).act(action)
                    self.assertLessEqual(len(cm.exception.policy_reason), 400)   # the gateway shows at most 400
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
    """A command the planner refuses (the inherited classroom planner for the FR3 rig, the G1 backend's own checks)
    tells the policy why: the controller's own text is its policy_reason. Without it the policy reads only "adjust
    the target or command parameters". No scene is built; the IK is a stand-in that refuses with the solver's text."""

    REACH = "IK unreachable for left: target 4.53 m from the arm base, reach 1.27 m; no action executed"
    G1_REACH = "IK unreachable (12.0 mm, 3.0 deg); no action executed"

    def _backend(self, cls=None, base_held=False):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        b = RendererReleased._backend(cls)
        b._renderer = object()                                     # the worker is running: the planner decides
        b.min_action_s = 0.02
        b.env.world.base_held = lambda: base_held
        text = self.G1_REACH if cls is not None else self.REACH
        def solve(*args, **kwargs):
            raise ValueError(text)
        b.ik = SimpleNamespace(commanded=lambda arm: None, solve=solve,
                               pose=lambda arm: (np.zeros(3), Rotation.identity()))
        return b

    def test_refusal_says_why_to_the_policy(self):
        from gym_anything_robotics.actions import BaseTwist, CartesianPose, NamedJointTarget
        from gym_anything_robotics.backends.newton_painting_g1 import NewtonPaintingG1Backend as G1
        from gym_anything_robotics.errors import ActionRejectedError
        cases = ((None, CartesianPose("left", (0.45, 0.45, 1.1), (0., 1., 0., 0.), duration_s=2.0), False, self.REACH),
                 (None, CartesianPose("left", (0., 0., 1.), duration_s=0.005), False,
                  "duration_s must be in [0.02, 30.0] s (this session's minimum: 0.02 s)"),
                 (None, BaseTwist(0.1, 0., 0., duration_s=0.5), True,
                  "Base command refused: an arm protective stop holds the base"),
                 (G1, CartesianPose("left", (0.3, 0.2, 1.0), duration_s=1.0), False, self.G1_REACH),
                 (G1, NamedJointTarget({}, 1), False, "Joint command ticks must span the advertised minimum"))
        for cls, action, held, text in cases:
            with self.subTest(cls=cls, action=type(action).__name__, text=text):
                with self.assertRaises(ActionRejectedError) as cm:
                    self._backend(cls, base_held=held).act(action)
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
    def test_backend(self):
        import tempfile
        from gym_anything_robotics import registry
        from gym_anything_robotics.actions import BaseTwist, Gripper
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        # the catalog refuses a listed-unsupported robot before the backend exists (robots.catalog.select_for_backend
        # raises ValueError); a setup the validator rejects raises UnsupportedConfigurationError
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("painting", task="roll-strip", robot="dual-fr3", cameras=())
        with self.assertRaises((UnsupportedConfigurationError, ValueError)):
            registry.create("painting", task="spray-stencil", robot="mobile-fr3-single", cameras=())
        tmp = tempfile.TemporaryDirectory()
        env = registry.create("painting", task="fill-circle", seed=1, cameras=(), output_dir=tmp.name + "/s")
        b = env._backend
        env.reset()
        b.act(BaseTwist(0.0, 0.0, 0.0, duration_s=0.5))
        st = b.env.get_state()
        self.assertIn("coverage", st["evaluation"]["checks"])
        ev = b.evaluate()
        self.assertFalse(ev["success"])                              # nothing painted yet
        self.assertLess(b.env.paint.ledger_residual()["relative"], 1e-9)
        cp = b.save_checkpoint(label="a")
        b.act(Gripper("right", half_width_m=0.04, duration_s=0.5))
        b.restore_checkpoint(label="a")
        self.assertEqual(b.env.world.tick, cp["tick"])
        idx = b.get_geometry_index()
        self.assertIn("paper", idx["charts"])
        env.close()


if __name__ == "__main__":
    unittest.main()

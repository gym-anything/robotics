"""Contract and regression checks for barcode-scanning.

The first classes need no GPU, no physics and no renderer (seconds): symbols round-trip through the decoder to
the GTINs the evaluator expects, the firmware state machine keeps its published timing, switches keep their
hysteresis, attribution geometry intersects the right patch, the catalog/launch/registry entries resolve.
The physics class builds the real Newton scene and is skipped unless GAR_BARCODE_PHYSICS_TESTS=1 (minutes):
rest state, trigger switch under a phantom force, restore contract, the Franka fingers through an arm-on-arm impact.
"""

import json
import math
import os
from pathlib import Path
import unittest

import numpy as np
from tests.packages import PACKAGES

ROOT = PACKAGES


class SymbolRoundTripTest(unittest.TestCase):


    def test_upce_is_recorded_as_its_expanded_gtin(self):
        from barcode_scanning.artwork import upce_expand, gtin
        self.assertEqual(gtin(upce_expand("0451270")), "045000001275")      # zero suppression, last digit 0
        self.assertEqual(upce_expand("0123453"), "01230000045")            # last digit 3
        self.assertEqual(upce_expand("0123454"), "01234000005")            # last digit 4
        self.assertEqual(upce_expand("0123457"), "01234500007")            # last digit 5-9


    def test_1d_symbols_decode_at_any_roll(self):
        # the DS2208 reads at any roll; the level passes alone read a library Codabar only within a few degrees
        # of level. A symbol turned about the frame centre, every 15 deg: read, and never misread
        from PIL import Image
        from barcode_scanning.artwork import Symbol, render_symbol
        from barcode_scanning.stock import library_item_number
        from barcode_scanning import decoder as D
        code = library_item_number(1234, 5037)
        for sym, want, fmts in ((Symbol("Codabar", f"A{code}B", .25, bar_height_mm=8.5, hri=False), code, D.FORMATS["library"]),
                                (Symbol("EAN13", "590123412345", .33, hri=False), "05901234123457", D.FORMATS["retail"])):
            img, _, _ = render_symbol(sym, 3)
            a = np.asarray(img.convert("L"))
            side = int(1.15*np.hypot(*a.shape))
            frame = np.full((side, side), 255, np.uint8)
            y0, x0 = (side - a.shape[0])//2, (side - a.shape[1])//2
            frame[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
            for roll in range(0, 180, 15):
                turned = np.asarray(Image.fromarray(frame).rotate(roll, resample=Image.BILINEAR, fillcolor=255))
                with self.subTest(kind=sym.kind, roll=roll):
                    self.assertEqual({D.normalise(f["format"], f["text"]) for f in D.read_all(turned, fmts)}, {want})


class FirmwareTest(unittest.TestCase):
    def test_trigger_session_one_read_per_pull(self):
        from barcode_scanning.device import Firmware
        from barcode_scanning import params as P
        fw = Firmware()
        due = [fw.tick(t*1e-3, True, False, False, False) for t in range(0, 100)]
        self.assertEqual(fw.state, "SESSION")
        self.assertEqual([t for t, d in enumerate(due) if d], [0, 34, 67])     # 30 Hz grid from the switch closing (0, 33.3, 66.7 ms)
        more = [100 + t for t in range(0, 900) if fw.tick((100 + t)*1e-3, True, False, False, False)]
        self.assertEqual(len(more), 27)                                         # 1.0 s -> 30 frames in total, no drift
        self.assertTrue(fw.illumination_on(0.05))
        rep = fw.on_frame_result(.05, dict(format="Code128", text="A03-3-B", quad_px=[[0, 0]]*4), [])
        self.assertIsNotNone(rep)
        self.assertEqual(fw.state, "WAIT_RELEASE")
        self.assertFalse(any(fw.tick(.1 + t*1e-3, True, False, False, False) for t in range(200)))
        self.assertFalse(fw.illumination_on(.2))
        fw.tick(.4, False, False, False, False)
        self.assertEqual(fw.state, "AIM")
        self.assertEqual(fw.led_intensity(.4), 0.)
        self.assertEqual(P.DECODE_SESSION_TIMEOUT_S, 9.9)

    def test_session_timeout(self):
        from barcode_scanning.device import Firmware
        fw = Firmware()
        for t in range(0, 10100, 10):
            fw.tick(t*1e-3, True, False, False, False)
        self.assertEqual(fw.state, "WAIT_RELEASE")

    def test_power_on_reads_the_seat_sensor(self):
        """A scanner that starts in its cradle is DOCKED from the first tick, and ticking on from there matches a
        firmware that only learned it was seated at its first tick."""
        from barcode_scanning.device import Firmware
        self.assertEqual(Firmware(seated=True).state, "DOCKED")
        self.assertEqual(Firmware().state, "LIFTED_IDLE")
        a, b = Firmware(seated=True), Firmware()
        for k in range(300):
            args = (k*1e-3, False, k < 200, k > 250, k % 50 == 0)
            self.assertEqual(a.tick(*args), b.tick(*args))
            self.assertEqual(a.state, b.state)

    def test_docked_same_symbol_refused_until_out_of_view(self):
        from barcode_scanning.device import Firmware
        fw = Firmware()
        fw.tick(0., False, True, False, True)
        self.assertEqual(fw.state, "DOCKED")
        sym = dict(format="EAN13", text="0078412003513", quad_px=[[0, 0]]*4)
        self.assertIsNotNone(fw.on_frame_result(.0, sym, [sym]))
        self.assertIsNone(fw.on_frame_result(.3, sym, [sym]))          # still in view
        self.assertIsNone(fw.on_frame_result(.7, sym, [sym]))          # seen at 0.3 -> 0.4 s later: refused
        self.assertIsNotNone(fw.on_frame_result(1.3, sym, [sym]))      # 0.6 s out of view: accepted
        other = dict(format="EAN13", text="0078412004176", quad_px=[[0, 0]]*4)
        self.assertIsNone(fw.on_frame_result(1.35, other, [other]))    # a different symbol within 0.1 s
        self.assertIsNotNone(fw.on_frame_result(1.45, other, [other]))

    def test_docked_symbol_left_stationary_is_not_re_read(self):
        """PRG p. 5-22: the same symbol is re-read only after it was OUT of view for 0.5 s; one left stationary in view
        (no frames once the hands-free session lapses) is still in view when it moves again."""
        from barcode_scanning.device import Firmware
        fw = Firmware()
        sym = dict(format="EAN13", text="0078412004176", quad_px=[[0, 0]]*4)
        fw.tick(0., False, True, False, True)
        self.assertIsNotNone(fw.on_frame_result(.0, sym, [sym]))
        fw.note_seen(.0, [sym])
        for k in range(1, 300):                                         # 3 s held still: no zone change, no frames
            fw.tick(k*.01, False, True, False, False)
        fw.tick(3.0, False, True, False, True)                          # it moves: a new hands-free session
        self.assertIsNone(fw.on_frame_result(3.0, sym, [sym]))


class SwitchAndGeometryTest(unittest.TestCase):
    def test_switch_hysteresis_and_debounce(self):
        from barcode_scanning.world import Switch
        s = Switch(3.0, 2.81, .005)
        seq = [(0., 2.9), (.001, 3.01), (.004, 3.01), (.006, 3.05), (.007, 2.85), (.012, 2.85), (.013, 2.80), (.019, 2.7)]
        states = []
        for t, x in seq:
            s.update(x, t); states.append(s.state)
        self.assertEqual(states, [False, False, False, True, True, True, True, False])

    def test_ray_cylinder_hits_the_near_side(self):
        from barcode_scanning.env import _ray_cylinder
        t = _ray_cylinder(np.array([-1., 0., 0.]), np.array([1., 0., 0.]), np.zeros(3), np.array([0., 0., 1.]), .05)
        self.assertAlmostEqual(t, .95, places=9)
        self.assertIsNone(_ray_cylinder(np.array([-1., .2, 0.]), np.array([1., 0., 0.]), np.zeros(3), np.array([0., 0., 1.]), .05))

    def test_imager_geometry_matches_datasheet(self):
        from barcode_scanning.optics import Imager
        self.assertAlmostEqual(Imager.ifov()*1e3, .9202, places=3)
        self.assertAlmostEqual(Imager.diffraction_sigma_px(), .42*645e-9/4.1e-4/Imager.ifov(), places=9)   # A = 0.41 mm

    def test_handheld_exposure_meters_the_aimed_centre(self):
        # the LEDs' near-range hot band (30x the centre) must not darken the aimed code (highlight protection in
        # handheld mode would move every near limit out)
        from barcode_scanning.optics import Imager
        L = np.full((480, 640), 1., np.float32); L[400:, :] = 30.
        im = Imager("handheld", rng=np.random.default_rng(0))
        for _ in range(12):
            fr, _ = im.expose(L)
        self.assertAlmostEqual(float(fr[168:312, 224:416].mean()), 110., delta=10.)

    def test_handsfree_exposure_keeps_a_small_bright_item_unclipped(self):
        # a lip balm swept past a docked scanner off the metering window, on a dark field
        from barcode_scanning.optics import Imager
        L = np.full((480, 640), .05, np.float32); L[60:120, 470:560] = 4.
        im = Imager("handsfree", rng=np.random.default_rng(0))
        for _ in range(20):
            fr, _ = im.expose(L)
        self.assertGreater(float(fr[60:120, 470:560].mean()), 180.)
        self.assertLess(float(fr[60:120, 470:560].mean()), 240.)

    def test_handsfree_exposure_settles_instead_of_hunting(self):
        # a dim centre and a bright band with the render's ~12 % per-pixel noise: the exposure settles instead of
        # hunting in a x8 four-frame limit cycle
        from barcode_scanning.optics import Imager
        rng = np.random.default_rng(3)
        im = Imager("handsfree", rng=np.random.default_rng(0))
        te = []
        for _ in range(30):
            L = np.full((480, 640), .5, np.float32); L[400:, :] = 6.
            fr, rec = im.expose(L*(1 + .12*rng.standard_normal(L.shape)).astype(np.float32))
            te.append(rec["exposure_s"]*rec["gain"])
        last = np.asarray(te[-10:])
        self.assertLess(last.max()/last.min(), 1.05)
        self.assertLess(float(np.percentile(fr[400:], 50)), 250.)

    def test_sensor_noise_follows_the_photon_count_at_any_gain(self):
        # Poisson noise is drawn on the photoelectron count, not on the gain-scaled signal (that would make a frame at
        # the hands-free gains 6-7 1.9x too clean). Photoelectrons N ~ Poisson(K t_e L), read noise in electrons, then the gain:
        # var(DN) = (255 g / FW)^2 (N + sigma_read^2) + (PRNU DN)^2
        from barcode_scanning import params as P
        from barcode_scanning.optics import Imager
        for gain, t_e in ((1., 2e-4), (6.47, 2.2e-4)):
            with self.subTest(gain=gain):
                im = Imager("handsfree", rng=np.random.default_rng(1))
                L0 = 110./(255.*gain*P.SENSOR_K*t_e/P.FULL_WELL_E)       # radiance that exposes to 110 DN
                im.t_e, im.gain = t_e, gain
                fr, _ = im.expose(np.full((480, 640), L0, np.float32))
                n = P.SENSOR_K*t_e*L0
                k = 255.*gain/P.FULL_WELL_E
                want = math.sqrt(k*k*(n + P.READ_NOISE_E**2) + (P.PRNU*110.)**2 + 1/12)
                self.assertAlmostEqual(float(fr.mean()), 110., delta=1.)
                self.assertAlmostEqual(float(fr.std()), want, delta=.05*want)


    def test_rtx_thin_lens_f_number(self):
        from barcode_scanning.render import RTX_FOCAL_UNIT_M, IMAGER_FOCAL_MM
        from barcode_scanning import params as P
        # RTX reads focal length in tenths of a scene unit (0.1 m per 'mm'): aperture = f_rtx / N must be the pupil
        n = RTX_FOCAL_UNIT_M*IMAGER_FOCAL_MM/P.PUPIL_DIAMETER_M
        self.assertAlmostEqual(IMAGER_FOCAL_MM*RTX_FOCAL_UNIT_M/n, P.PUPIL_DIAMETER_M, places=12)
        self.assertAlmostEqual(n, 3.56/4.1e-4, places=6)          # f/8683 for the 0.41 mm pupil


class IntegrationContractTest(unittest.TestCase):
    def test_retained_render_worker_is_released_by_its_timer_and_on_close(self):
        # bounded retention (the default keeps a worker IDLE_RELEASE_S for the next operation): no worker may outlive it
        import threading
        import time
        import types
        from gym_anything_robotics.backends.newton_barcode_scanning import NewtonBarcodeScanningBackend as B, IDLE_RELEASE_S
        self.assertGreater(IDLE_RELEASE_S, 0.)

        class Fake:
            closed = 0

            def close(self):
                self.closed += 1

        def stub(idle):
            st = types.SimpleNamespace(options={"idle_release_s": idle}, _prestart=None, _scene_path="scene.usd",
                                       env=types.SimpleNamespace(lock=threading.RLock()))
            st._release_retained = types.MethodType(B._release_retained, st)
            st._end_operation = types.MethodType(B._end_operation, st)
            return st
        st, r = stub(.2), Fake()
        st._renderer = r
        st._end_operation()
        self.assertIs(st._retained, r); self.assertEqual(r.closed, 0)
        time.sleep(.6)
        self.assertEqual(r.closed, 1); self.assertIsNone(st._retained)
        st, r = stub(30.), Fake()                     # close() (final) releases a retained worker at once
        st._renderer = r
        st._end_operation()
        st._end_operation(final=True)
        for t in st._reapers:
            t.join(5.)
        self.assertEqual(r.closed, 1); self.assertIsNone(st._retained)
        st._retain_timer.cancel()
        st, r = stub(0.), Fake()                      # 0: released after the operation
        st._renderer = r
        st._end_operation()
        for t in st._reapers:
            t.join(5.)
        self.assertEqual(r.closed, 1)

    def test_catalog_launch_registry(self):
        from robots.catalog import ENVIRONMENTS, resolve
        from gym_anything_robotics import launch, registry
        item = ENVIRONMENTS["barcode-scanning"]
        self.assertEqual(item["default"], "dual-fr3")
        self.assertEqual(set(item["aliases"]), {"dual-fr3", "dual-panda", "single-fr3", "g1"})
        for rig in item["aliases"].values():
            self.assertTrue(resolve(rig).to_dict()["instances"])
        self.assertIn("barcode-scanning", launch.ENVIRONMENTS)
        self.assertEqual(registry.canonical("scan-barcode") if hasattr(registry, "canonical") else "barcode-scanning", "barcode-scanning")
        # the G1 is a qualified configuration; an unknown robot is refused
        self.assertEqual(launch.LaunchSpec("barcode-scanning", robot="g1").robot, "g1")
        with self.assertRaises(ValueError):
            launch.LaunchSpec("barcode-scanning", robot="no-such-robot")

    def test_benchmark_catalog_tasks_resolve(self):
        from barcode_scanning.config import TASKS, Config
        bench = json.loads((ROOT/"barcode_scanning"/"benchmark.json").read_text())
        self.assertEqual(bench["schema"], "gar.task-catalog.v1")
        for preset, tasks in TASKS.items():
            for task in tasks:
                Config(preset=preset, task=task)
        with self.assertRaises(ValueError):
            Config.from_dict(dict(preset="stockroom", task="scan-location", cheat=1))
        with self.assertRaises(ValueError):
            Config(preset="counter", task="scan-location")

    def test_policy_interface_has_no_scan_action(self):
        src = (ROOT/"gym_anything_robotics"/"backends"/"newton_barcode_scanning.py").read_text()
        for forbidden in ("def scan(", "\"scan\"", "trigger_switch\": True"):
            self.assertNotIn(forbidden, src)


class RobotSetupValidatorTest(unittest.TestCase):
    """Setup objects: the four rigs passed as objects run as their aliases; any
    other setup is refused with what the environment needs, not with the catalog's generic message."""

    def test_own_rigs_accepted_others_refused(self):
        import copy
        from robots.catalog import resolve
        from barcode_scanning.robot_setup import validate_setup
        for alias, rig in (("dual-fr3", "scan-fr3"), ("single-fr3", "scan-fr3-single"), ("g1", "scan-g1")):
            setup = resolve(rig)
            self.assertEqual(validate_setup(setup), alias)
            self.assertEqual(validate_setup(setup.to_dict()), alias)
        other = copy.deepcopy(resolve("scan-fr3").to_dict())
        other["id"] = "my-fr3"
        other["instances"]["left"]["controller_profile"] = "label_application"    # the same arms, another profile
        with self.assertRaises(ValueError) as ctx:
            validate_setup(other)
        self.assertIn("four catalog rigs", str(ctx.exception))


class GripLimitTest(unittest.TestCase):
    """A grip the hand cannot give is refused before anything runs (ActionRejectedError, which the HTTP service treats
    as 'nothing executed'), never raised later by the world as a plain error that latches a runtime fault."""

    def backend(self, per_finger):
        import threading
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_barcode_scanning import NewtonBarcodeScanningBackend as B

        class World:
            joints = {f"{a}/f{i}": dict(q=k, effort_limit=per_finger, lower=0., upper=.04)
                      for k, (a, i) in enumerate((a, i) for a in ("left", "right") for i in (1, 2))}

            def gripper_joints(self, arm):
                return [f"{arm}/f1", f"{arm}/f2"]

            def set_grip_force(self, arm, force):
                raise AssertionError("the world must not be reached")
        b = object.__new__(B)
        b.env = SimpleNamespace(world=World(), lock=threading.RLock())
        b.g1, b.arms, b.config = False, ("left", "right"), SimpleNamespace(rigid_dt=.001)
        b._validate_targets = lambda targets: None
        return b

    def test_grip_beyond_the_hand_is_refused_before_execution(self):
        from gym_anything_robotics.actions import Gripper, Keyboard
        from gym_anything_robotics.errors import ActionRejectedError
        fr3, panda = self.backend(35.), self.backend(20.)
        with self.assertRaises(ActionRejectedError):
            fr3.act(Gripper(arm="right", inward_effort_n=100.))
        with self.assertRaises(ActionRejectedError):
            panda.act(Gripper(arm="right", inward_effort_n=50.))
        # the strongest keyboard grip is the hand's own maximum
        _, _, grip = panda._plan(Keyboard(keys=("H", "B"), ticks=30, arm="right"))
        self.assertEqual(grip, {"right": 20.})
        _, _, grip = fr3._plan(Keyboard(keys=("H", "B"), ticks=30, arm="right"))
        self.assertEqual(grip, {"right": 35.})
        _, _, grip = fr3._plan(Gripper(arm="right", inward_effort_n=70.))
        self.assertEqual(grip, {"right": 35.})

    def test_fingers_of_one_hand_take_one_target(self):
        # the Franka Hand's fingers are one mechanism: a policy's named targets for one hand's two fingers must agree
        # (independent targets fight the stiff coupling and can force a finger 19 mm past its limit)
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        fr3 = self.backend(35.)
        fr3._validate_targets = type(fr3)._validate_targets.__get__(fr3)
        with self.assertRaises(ActionRejectedError):
            fr3.act(NamedJointTarget({"right/f1": .01, "right/f2": .03}, ticks=10))
        fr3._validate_targets({"right/f1": .02, "right/f2": .02})


@unittest.skipUnless(os.environ.get("GAR_BARCODE_PHYSICS_TESTS") == "1", "set GAR_BARCODE_PHYSICS_TESTS=1 (builds the Newton scene)")
class PhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from barcode_scanning.config import Config
        from barcode_scanning.env import ScanEnv
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = ScanEnv(Config(preset="stockroom", task="scan-location"), output_dir=Path(cls.tmp.name)/"s")

    def test_rest_state(self):
        w = self.env.world
        self.assertTrue(w.seated)
        self.assertFalse(w.trigger.state)
        self.assertFalse(any(s.state for s in w.key_switch.values()))
        self.assertLess(abs(w.trigger_travel_m()), 1e-4)

    def test_phantom_press_closes_the_switch_and_restore_is_exact(self):
        from barcode_scanning.probes.fixtures import PhantomHold, PhantomTrigger, combine
        env = self.env
        snap = env.checkpoint()
        w = env.world
        hold, trig = PhantomHold(w), PhantomTrigger(w)
        combine(w, hold, trig)
        p, R = w.body_pose("scanner")
        from scipy.spatial.transform import Rotation
        hold.move(p + np.array([0, 0, .1]), Rotation.from_matrix(R).as_quat(), .5)
        for _ in range(700):
            w.step()
        trig.force = 3.0
        for _ in range(200):
            w.step()
        self.assertTrue(w.trigger.state)
        self.assertGreater(w.trigger_travel_m(), .003)
        w.fixture = None
        env.restore(snap)
        self.assertTrue(np.array_equal(w.state.body_q.numpy(), snap["rigid"]["body_q"]))
        self.assertFalse(w.trigger.state)

    def test_collision_reflex_is_recorded_again_after_a_restore(self):
        # a restore rewinds the tick; the one-record-per-stop rule must still record a stop after a restore
        env = self.env
        w = env.world
        snap = env.checkpoint()
        j2 = "right/fr3_joint2"
        # the shoulder driven forward strikes the cradled scanner's head (a 1.3 rad stroke stops the arm in 0.5 s at
        # 36 N m on joint 4)
        goal = float(w.state.joint_q.numpy()[w.joints[j2]["q"]]) + 1.3

        def jam():
            n = len(w.reflexes)
            for _ in range(30):
                env.act({j2: goal}, ticks=int(round(.1/w.config.rigid_dt)))
                if len(w.reflexes) > n:
                    return True
            return False
        n0 = len(w.reflexes)
        self.assertTrue(jam())
        self.assertGreater(len(w.reflexes), n0)
        env.restore(snap)
        # the reflex record is part of the checkpoint (a restore drops the stops recorded after it), and the stop after
        # the restore is recorded again
        self.assertEqual(len(w.reflexes), n0)
        self.assertTrue(jam())
        self.assertGreater(len(w.reflexes), n0)
        env.restore(snap)


@unittest.skipUnless(os.environ.get("GAR_BARCODE_PHYSICS_TESTS") == "1", "set GAR_BARCODE_PHYSICS_TESTS=1 (builds the Newton scene)")
class FingerImpactTest(unittest.TestCase):
    """The first three legal commands of a fresh session (both grippers, then in-limit targets for all 14 arm joints)
    swing the arms into each other at full speed. An unconverged solve at the impact would slide one hand's two fingers
    sideways past their stops in one 1 ms tick, unflagged. The commands, as HTTP bodies."""

    CASES = {
        "dual-fr3": (dict(preset="counter", task="checkout-bulky", seed=2), "right", [
            {"type": "gripper", "arm": "left", "duration_s": 0.001, "half_width_m": 0.020245622778538367},
            {"type": "gripper", "arm": "right", "duration_s": 0.001, "half_width_m": 0.03637940904642537},
            {"type": "named_joint_target", "ticks": 498, "targets_si": {
                "left/fr3_joint1": 0.7526519808434493, "left/fr3_joint2": -0.9818306982811911,
                "left/fr3_joint3": 0.4919618096740064, "left/fr3_joint4": -2.6293735269844687,
                "left/fr3_joint5": 1.5351010942540761, "left/fr3_joint6": 0.9192640505574272,
                "left/fr3_joint7": 2.102071501215715, "right/fr3_joint1": 2.4037798012186484,
                "right/fr3_joint2": 1.2500531533710757, "right/fr3_joint3": 1.3127753344433475,
                "right/fr3_joint4": -0.4899957585988388, "right/fr3_joint5": 1.9433511189105332,
                "right/fr3_joint6": 3.153510741608308, "right/fr3_joint7": 2.5450595174526014}}]),
        "dual-panda": (dict(preset="counter", task="checkout-counter", seed=4), "right", [
            {"type": "gripper", "arm": "left", "duration_s": 0.001, "inward_effort_n": 4.12857155081777},
            {"type": "gripper", "arm": "right", "duration_s": 0.001, "inward_effort_n": 7.346800717475355},
            {"type": "named_joint_target", "ticks": 498, "targets_si": {
                "left/panda_joint1": -1.1420903174171648, "left/panda_joint2": -0.16788044039657724,
                "left/panda_joint3": -1.1336580445793378, "left/panda_joint4": -1.4008191466590691,
                "left/panda_joint5": -0.5340997150423044, "left/panda_joint6": 1.4119548775522361,
                "left/panda_joint7": -2.7217504900680214, "right/panda_joint1": -2.176146024434189,
                "right/panda_joint2": -1.073960475650795, "right/panda_joint3": 2.3002867722869076,
                "right/panda_joint4": -0.5307468877691432, "right/panda_joint5": 0.45750995703577013,
                "right/panda_joint6": 0.03025338365980567, "right/panda_joint7": -0.35297753656571285}}]),
    }

    def test_arm_on_arm_impact_leaves_the_fingers_at_their_stops_and_together(self):
        import tempfile
        from gym_anything_robotics.action_codec import action_from_dict
        from gym_anything_robotics.backends.newton_barcode_scanning import NewtonBarcodeScanningBackend
        from barcode_scanning.probes.e17_stress import DarkRenderer
        for robot, (opts, arm, bodies) in self.CASES.items():
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as tmp:
                b = NewtonBarcodeScanningBackend(robot=robot, cameras=[], output_dir=Path(tmp)/"s", recording=False, **opts)
                b._start_renderer_early = lambda: None          # physics only: dark imager frames, no render worker
                b.env.render_meta = {"scanner": {"leds": [], "aim": "aim", "led_on": "led_on"}}
                b.env.renderer_factory = lambda: DarkRenderer()
                w = b.env.world
                qi = [w.joints[n]["q"] for n in w.gripper_joints(arm)]
                lo, hi = w.joints[w.gripper_joints(arm)[0]]["lower"], w.joints[w.gripper_joints(arm)[0]]["upper"]
                worst = dict(past_stop=0., apart=0.)

                def watch(env):
                    q = env.world.state.joint_q.numpy()[qi]
                    worst["past_stop"] = max(worst["past_stop"], float(np.max(np.maximum(lo - q, q - hi))))
                    worst["apart"] = max(worst["apart"], abs(float(q[0] - q[1])))
                b.env.trace_callback = watch
                try:
                    for body in bodies:
                        b.act(action_from_dict(body))
                finally:
                    b.close()
                self.assertIsNone(w.fault)
                # the impact itself still happens: the arms meet and both collision reflexes fire far above threshold
                self.assertGreater(max(r.get("external_torque_nm", 0.) for r in w.reflexes), 300.)
                self.assertLess(worst["past_stop"], 2e-3)
                self.assertLess(worst["apart"], 5e-3)

    def test_parted_fingers_stop_the_episode_as_a_numerical_failure(self):
        # the guard behind the impact test: a hand's coupled fingers further apart than params.MAX_FINGER_COUPLING_M
        # (written here, as an unconverged solve would leave them) fault the world like a non-finite state instead of
        # passing silently
        from barcode_scanning import params as P
        from barcode_scanning.config import Config
        from barcode_scanning.world import World
        w = World(Config(preset="counter", task="checkout-present"))
        w.step()
        self.assertIsNone(w.fault)
        a, b = (w.joints[n]["q"] for n in w.gripper_joints("right"))
        q = w.state.joint_q.numpy()
        q[a], q[b] = .036, .036 - 3*P.MAX_FINGER_COUPLING_M
        w.state.joint_q.assign(q)
        w.sync_solver()
        with self.assertRaisesRegex(FloatingPointError, "Unphysical finger state"):
            w.step()
        self.assertIn("Unphysical finger state", w.fault)


class RobotProfileTest(unittest.TestCase):
    def test_rigs_use_the_environments_own_controller_profiles(self):
        # the rigs use this environment's own controller profile, so a gain change made for label_application or
        # whiteboard cannot silently change these robots
        from robots import catalog
        for robot in ("dual-fr3", "dual-panda", "single-fr3", "g1"):
            with self.subTest(robot=robot):
                setup = catalog.for_environment("barcode-scanning", robot).to_dict()
                self.assertEqual({i["controller_profile"] for i in setup["instances"].values()}, {"barcode_scanning"})

    def test_profiles_start_from_the_environments_own_pose(self):
        # the profiles name this environment's own start pose, not label_application's or whiteboard's, so a change
        # to those environments' poses cannot move this environment's start
        import json
        root = PACKAGES/"robots"/"models"
        for model, was in (("fr3", "label_application"), ("panda", "label_application"), ("g1", "whiteboard")):
            with self.subTest(model=model):
                d = json.loads((root/model/"robot.json").read_text())
                self.assertEqual(d["actuation"]["profiles"]["barcode_scanning"]["initial_pose"], "barcode_scanning")
                self.assertIn("barcode_scanning", d["poses"])

    def test_unreachable_robot_task_pairs_are_refused(self):
        # the one-arm FR3 cannot reach the tote or the returns bin, the G1 not the
        # bulky case; those pairs are refused at launch instead of being offered
        from gym_anything_robotics.backends.newton_barcode_scanning import NewtonBarcodeScanningBackend, UNREACHABLE_TASKS
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        self.assertIn(("single-fr3", "checkout-present"), UNREACHABLE_TASKS)
        for (robot, task) in UNREACHABLE_TASKS:
            preset = "library" if task == "library-checkin" else "counter"
            with self.subTest(robot=robot, task=task), self.assertRaises(UnsupportedConfigurationError):
                NewtonBarcodeScanningBackend(robot=robot, preset=preset, task=task)

    def test_setup_objects_of_the_offered_rigs_select_their_alias(self):
        # a setup object of an offered rig selects its alias, not the catalog's generic "extracted baseline setup only"
        # refusal; any other rig is refused with the list of offered rigs
        from robots.catalog import resolve, select_for_backend
        for alias, rig in (("dual-fr3", "scan-fr3"), ("dual-panda", "scan-panda"), ("single-fr3", "scan-fr3-single"), ("g1", "scan-g1")):
            with self.subTest(rig=rig):
                self.assertEqual(select_for_backend("barcode-scanning", resolve(rig)), alias)
                self.assertEqual(select_for_backend("barcode-scanning", resolve(rig).to_dict()), alias)
        with self.assertRaisesRegex(ValueError, "dual-fr3 \\(scan-fr3\\), dual-panda"):
            select_for_backend("barcode-scanning", resolve("label-panda"))


@unittest.skipUnless(os.environ.get("GAR_BARCODE_PHYSICS_TESTS") == "1", "set GAR_BARCODE_PHYSICS_TESTS=1 (builds the Newton scene)")
class G1PhysicsTest(unittest.TestCase):
    """The G1 configuration's start pose and reset in every preset (a start pose with the hands inside the hip links
    would fell the robot before tick 0)."""

    def test_start_pose_has_no_robot_penetration_and_reset_stands(self):
        import tempfile
        import mujoco
        from barcode_scanning.config import Config, TASKS
        from barcode_scanning.env import ScanEnv
        from barcode_scanning.world import World
        for preset in ("stockroom", "counter", "library"):
            with self.subTest(preset=preset):
                cfg = Config(preset=preset, task=TASKS[preset][0], robot="g1", device="cpu")
                w = World(cfg)
                m, d = w.solver.mj_model, w.solver.mj_data
                mujoco.mj_forward(m, d)
                robot = set(w.scene.robot_bodies)
                sb = w.model.shape_body.numpy()
                deep = []
                for i in range(d.ncon):
                    c = d.contact[i]
                    shapes = [int(w._geom_shape[g]) for g in c.geom]
                    bodies = [int(sb[s]) if s >= 0 else -1 for s in shapes]
                    if any(b in robot for b in bodies) and c.dist < -1e-3:
                        deep.append(([w.model.shape_label[s] for s in shapes], round(float(c.dist)*1e3, 1)))
                self.assertEqual(deep, [], "robot geometry starts penetrating (mm)")
                with tempfile.TemporaryDirectory() as tmp:
                    env = ScanEnv(cfg, output_dir=Path(tmp)/"s")       # raises if the G1 fell during the settle
                    self.assertLess(env.settle_record["g1_tilt_deg"], 2.)

    def test_policy_observation_passes_the_policy_filter(self):
        # the backend's real proprioception, through the shared filter: a restricted observation passes, and the G1's
        # IMU roll-pitch-yaw comes through as three values
        import tempfile
        from gym_anything_robotics.backends.newton_barcode_scanning import NewtonBarcodeScanningBackend
        from gym_anything_robotics.types import ImageFrame, Observation
        for robot in ("g1", "dual-fr3"):
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as tmp:
                b = NewtonBarcodeScanningBackend(robot=robot, preset="stockroom", device="cpu", output_dir=Path(tmp)/"s",
                                                 recording=False)
                access = b.policy_access()
                state = b.get_state()
                png = Path(tmp)/"x.png"; png.write_bytes(b"\x89PNG\r\n\x1a\n")
                obs = Observation(0, state["time_s"], robot, {"joints": state["joints"], "robot_status": b._robot_status()},
                                  {n: ImageFrame(n, png, True, camera=b.camera_specs[n]) for n in b.cameras})
                try:
                    out = access.observation(obs)
                finally:
                    b.close()
                if robot == "g1":
                    self.assertEqual(len(out.proprioception["robot_status"]["imu"]["rpy_rad"]), 3)


if __name__ == "__main__":
    unittest.main()

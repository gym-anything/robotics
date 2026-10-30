"""Contract checks for the tighten-codex backend, task catalog and evaluator.

The first class needs no GPU. The physics class builds the real Newton scene
and is skipped unless GAR_TIGHTEN_PHYSICS_TESTS=1 (it takes minutes).
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from gym_anything_robotics.errors import UnsupportedConfigurationError
from tests.packages import PACKAGES

ROOT = PACKAGES


class TightenContractTest(unittest.TestCase):
    def test_task_resolver_and_selectors(self):
        from tighten_codex.task import resolve, TASKS
        self.assertEqual(set(TASKS), {"sensor", "aim", "frame", "mixed", "support", "access", "front-m4", "find-loose"})
        spec = resolve({"task": "aim", "seed": 2})
        self.assertEqual(spec["task_id"], "tighten-codex/v1/aim/seed-2")
        self.assertEqual(spec["backend_options"], {"config": {"task": "aim", "seed": 2}})
        for bad in ({"task": "wheel"}, {"seed": -1}, {"seed": 1.5}, {"robot": "x"}):
            with self.assertRaises(ValueError):
                resolve(bad)

    def test_config_validation(self):
        from tighten_codex.config import Config
        Config(task="support")
        for bad in (dict(task="drill"), dict(dt=0.), dict(clamp_mu=-1.),
                    dict(fixture_hold_sensor=True),          # needs fixture_mode
                    dict(fixture_preload_test=True, fixture_mode=True),  # retired
                    dict(fixture_mode=True, fixture_target="nut")):
            with self.assertRaises(ValueError):
                Config(**bad)

    def test_catalog_offers_three_qualified_setups(self):
        from robots.catalog import ENVIRONMENTS, resolve
        aliases = ENVIRONMENTS["tighten-codex"]["aliases"]
        self.assertEqual(ENVIRONMENTS["tighten-codex"]["default"], "dual-fr3")
        self.assertEqual(aliases, {"dual-fr3": "tighten-fr3", "single-fr3": "tighten-fr3-single",
                                   "dual-panda": "tighten-panda"})
        for rig in aliases.values():
            data = resolve(rig).to_dict()
            for item in data["instances"].values():
                # Own controller profile: not coupled to another environment's gains.
                self.assertEqual(item["controller_profile"], "tighten")

    def test_backend_rejects_unhostable_robot_fixture_and_cameras(self):
        from gym_anything_robotics.backends.newton_tighten_codex import NewtonTightenCodexBackend
        with self.assertRaises(UnsupportedConfigurationError) as caught:
            NewtonTightenCodexBackend(robot="whiteboard-g1")
        self.assertIn("dual-panda", str(caught.exception))
        with self.assertRaises(UnsupportedConfigurationError):
            NewtonTightenCodexBackend(config={"fixture_mode": True})
        with self.assertRaises(ValueError):
            NewtonTightenCodexBackend(cameras=("wrist_rgb",))
        with self.assertRaises(ValueError):
            NewtonTightenCodexBackend(renderer_options={"width": 4000})

    def test_duration_must_be_whole_ticks(self):
        from gym_anything_robotics.backends.newton_tighten_codex import NewtonTightenCodexBackend
        backend = object.__new__(NewtonTightenCodexBackend)
        backend.env = SimpleNamespace(config=SimpleNamespace(dt=1 / 4800))
        self.assertEqual(backend._ticks(.25), 1200)
        with self.assertRaises(ValueError):
            backend._ticks(.0001)

    def test_release_floor_follows_driver_setting(self):
        from tighten_codex.config import Config
        from tighten_codex.env import TightenEnv
        env = object.__new__(TightenEnv)
        env.config = Config()
        env.items = {"sensor_left": {"turn_joint": 0, "diameter_m": .003, "pitch_m": .0005,
                                     "limit_nm": .35},
                     "frame_top": {"turn_joint": 1, "diameter_m": .004, "pitch_m": .0007,
                                   "limit_nm": .60}}
        k3 = env.torque_per_newton("sensor_left")
        # M3 x 0.5, d2 2.675 mm, thread mu 0.14, head on plastic mu 0.30, Db 4.5 mm.
        lead = math.atan(.0005 / (math.pi * .002675))
        expect = .0013375 * math.tan(lead + math.atan(.14 / math.cos(math.pi / 6))) + .30 * .00225
        self.assertAlmostEqual(k3, expect, places=9)
        self.assertAlmostEqual(env.minimum_load("sensor_left"), .8 * .35 / expect, places=6)
        self.assertGreater(env.minimum_load("frame_top"), env.minimum_load("sensor_left"))

    def test_benchmark_catalog_matches_tasks(self):
        data = json.loads((ROOT / "tighten_codex" / "benchmark.json").read_text())
        from tighten_codex.task import TASKS, resolve
        self.assertEqual(data["count"], len(TASKS))
        self.assertEqual(data["resolver"], "tighten_codex.task:resolve")
        self.assertEqual(resolve({})["task_id"], data["default_task_id"])


class _Arr:
    def __init__(self, a):
        import numpy as np
        self.a = np.asarray(a, float)

    def numpy(self):
        return self.a


def _fake_env(task, **tracking):
    """A TightenEnv shell with hand-set state for evaluator/clutch logic."""
    import numpy as np
    from tighten_codex.config import Config
    from tighten_codex.env import TightenEnv
    env = object.__new__(TightenEnv)
    env.config = Config(task=task)
    names = ("sensor_left", "sensor_right", "frame_top", "frame_front")
    env.items = {n: {"turn_joint": k, "slide_joint": 4 + k, "diameter_m": .003 if n.startswith("sensor") else .004,
                     "pitch_m": .0005 if n.startswith("sensor") else .0007,
                     "limit_nm": .35 if n.startswith("sensor") else .60} for k, n in enumerate(names)}
    env.items.update(rail=0, bracket=1, sensor=2, flag_joint=8)
    q = np.zeros(9); q[4:8] = .0006   # q[8]: the reference flag's carriage, at home
    env.state = SimpleNamespace(joint_q=_Arr(q), body_q=_Arr(np.tile([.14, .1, 1.0955, 0, 0, 0, 1.], (3, 1))))
    env.model = SimpleNamespace(joint_q_start=_Arr(np.arange(9)))
    env.builder = SimpleNamespace(joint_label=["horizontal_profile_slot_travel"])
    env._bracket_origin = [.14, .1, 1.0955]
    floor = {n: env.minimum_load(n) for n in names}
    base = dict(_preload={n: floor[n] / .8 for n in names}, _clicks={n: 1 for n in names},
                _bit_contacts={n: 0 for n in names}, _bit_moment={n: 0. for n in names},
                _peak_bit_moment={n: .36 for n in names}, _peak_bit_moment_filtered={n: .36 for n in names},
                _thread_load={n: 0. for n in names}, _retained_ticks={n: 2400 for n in names},
                _peak_preload={n: 0. for n in names}, _peak_torque={n: 0. for n in names},
                _seated_clicks={n: 1 for n in names}, _free_load={n: floor[n] / .8 for n in names},
                _subfloor_ticks={n: 0 for n in names}, _click_turn={n: None for n in names},
                _free_pitch=0., _held={"sensor": False, "frame": False},
                _interface={}, fault=None, _pitch_yaw=(0., 0.), _detected=True, _optics=None)
    base.update(tracking)
    for k, v in base.items():
        setattr(env, k, v)
    env._optical = lambda: env._optics or {"detected": env._detected}
    env._rail_attitude = lambda body_q=None: env._pitch_yaw
    return env


class TightenEvaluatorTest(unittest.TestCase):
    def test_success_needs_click_retention_level_rail_and_m3_limit(self):
        self.assertTrue(_fake_env("frame")._evaluate()["success"])
        self.assertTrue(_fake_env("sensor")._evaluate()["success"])
        unclicked = _fake_env("frame", _clicks={"sensor_left": 0, "sensor_right": 0, "frame_top": 0, "frame_front": 1})
        self.assertFalse(unclicked._evaluate()["success"])
        bit_left = _fake_env("frame", _retained_ticks={"sensor_left": 0, "sensor_right": 0,
                                                       "frame_top": 0, "frame_front": 2400})
        self.assertFalse(bit_left._evaluate()["success"])
        over = _fake_env("sensor", _peak_bit_moment_filtered={"sensor_left": .41, "sensor_right": .3,
                                                              "frame_top": 0., "frame_front": 0.})
        self.assertFalse(over._evaluate()["success"])
        drooped = _fake_env("frame", _pitch_yaw=(.034, 0.))
        self.assertFalse(drooped._evaluate()["success"])
        self.assertFalse(_fake_env("sensor", _detected=False)._evaluate()["success"])
        # The frame is not part of a sensor task's rail check.
        self.assertTrue(_fake_env("sensor", _pitch_yaw=(.034, 0.))._evaluate()["success"])

    def test_rail_offset_must_be_in_reference(self):
        env = _fake_env("support")
        env.state.body_q.a[0, 0] = .1407
        self.assertFalse(env._evaluate()["rail_in_reference"])


class TightenMixedTest(unittest.TestCase):
    """The mixed work order's verifier and score."""
    NAMES = ("sensor_left", "sensor_right", "frame_top", "frame_front")

    def _ev(self, **tracking):
        return _fake_env("mixed", **tracking)._evaluate()

    def _setting(self):
        env = _fake_env("mixed")
        return {n: env.minimum_load(n) / .8 for n in self.NAMES}

    def test_instruction_states_the_job_not_the_checklist(self):
        from tighten_codex.task import TASKS, resolve
        title, text = TASKS["mixed"]
        self.assertEqual(resolve({})["task_id"], "tighten-codex/v1/mixed/seed-0")
        for words in ("green 0.35 N m", "yellow 0.60 N m", "clicks", "level", "home position",
                      "detect it there", "on its own", "forward into the post", "upward from underneath"):
            self.assertIn(words, text)
        for leak in ("mrad", "0.25 s", "80 %", "±", "pair", "verify", "Support the rail", "stop", "LED"):
            self.assertNotIn(leak, text)

    def test_reference_finish_succeeds_with_full_credit(self):
        ev = self._ev()
        self.assertTrue(ev["success"])
        self.assertEqual(ev["score"], 1.)

    def test_no_hidden_slot_or_yaw_gate_but_level_is_required(self):
        env = _fake_env("mixed")
        env.state.body_q.a[0, 0] = .14 - .0025
        self.assertTrue(env._evaluate()["success"])
        self.assertTrue(self._ev(_pitch_yaw=(0., .004))["success"])
        self.assertFalse(self._ev(_pitch_yaw=(.0031, 0.))["success"])
        # frame keeps its own slot-reference and squareness gates.
        frame = _fake_env("frame")
        frame.state.body_q.a[0, 0] = .14 - .0015
        self.assertFalse(frame._evaluate()["success"])
        self.assertFalse(_fake_env("frame", _pitch_yaw=(0., .004))._evaluate()["success"])

    def test_success_does_not_wait_a_quarter_second_after_the_last_withdrawal(self):
        ticks = {n: 10 for n in self.NAMES}
        self.assertTrue(self._ev(_retained_ticks=ticks)["success"])
        ticks["frame_front"] = 9
        ev = self._ev(_retained_ticks=ticks)
        self.assertFalse(ev["success"])
        self.assertEqual(ev["score"], .95)
        self.assertFalse(_fake_env("frame", _retained_ticks={n: 10 for n in self.NAMES})._evaluate()["success"])

    def test_a_hand_on_the_assembly_is_not_success(self):
        ev = self._ev(_held={"sensor": False, "frame": True})
        self.assertFalse(ev["success"])
        self.assertFalse(ev["assembly_hands_off"])
        self.assertEqual(ev["score"], .95)

    def test_click_must_be_the_preset_drivers_and_survive_seating(self):
        ev = self._ev(_seated_clicks={"sensor_left": 1, "sensor_right": 1, "frame_top": 0, "frame_front": 1})
        self.assertFalse(ev["success"])
        self.assertAlmostEqual(ev["score_terms"]["screw_credit"]["frame_top"], .9)

    def test_retention_tracking_uses_hands_off_readings_only(self):
        import numpy as np
        setting = self._setting()
        env = _fake_env("mixed", _retained_ticks={n: 0 for n in self.NAMES},
                        _retained_anchor={n: None for n in self.NAMES},
                        _free_load={n: 0. for n in self.NAMES})
        env._rail_attitude = lambda body_q=None: (.04, 0.)
        # A robot on the rail: no M4 reading is taken and the pitch is not read.
        env._held = {"sensor": False, "frame": True}
        env._track_retention()
        self.assertEqual(env._retained_ticks["frame_top"], 0)
        self.assertEqual(env._free_load["frame_top"], 0.)
        self.assertEqual(env._retained_ticks["sensor_left"], 1)
        self.assertAlmostEqual(env._free_load["sensor_left"], setting["sensor_left"])
        self.assertEqual(env._free_pitch, 0.)
        # Let go: readings resume. A hands-off load below the floor voids the
        # click only after 10 consecutive readings; a shorter dip does not.
        env._held = {"sensor": False, "frame": False}
        env._preload["frame_front"] = .79 * setting["frame_front"]
        for _ in range(9):
            env._track_retention()
        self.assertEqual(env._seated_clicks["frame_front"], 1)
        self.assertAlmostEqual(env._free_pitch, .04)
        env._preload["frame_front"] = setting["frame_front"]
        env._track_retention()
        env._preload["frame_front"] = .79 * setting["frame_front"]
        for _ in range(9):
            env._track_retention()
        self.assertEqual(env._seated_clicks["frame_front"], 1)
        env._track_retention()
        self.assertEqual(env._seated_clicks["frame_front"], 0)
        self.assertEqual(env._seated_clicks["frame_top"], 1)
        # A bit in the socket is not a hands-off reading either: it cannot
        # void the click by the floor rule, but lost load under it shows.
        env._bit_contacts["sensor_right"] = 2
        env._preload["sensor_right"] = 0.
        env._track_retention()
        self.assertEqual(env._seated_clicks["sensor_right"], 1)
        self.assertEqual(env._free_load["sensor_right"], 0.)
        self.assertTrue(np.isfinite(env._evaluate()["score"]))

    def test_partial_credit_follows_deliverables_not_gates(self):
        setting = self._setting()
        zero = {n: 0 for n in self.NAMES}
        loose = {n: (21.6 if n == "frame_top" else 0.) for n in self.NAMES}
        start = dict(_seated_clicks=zero, _retained_ticks=zero, _preload=dict(loose), _free_load=dict(loose),
                     _free_pitch=.0369, _pitch_yaw=(.0369, 0.), _peak_bit_moment_filtered=dict(zero))
        ev = self._ev(**start)
        self.assertEqual((ev["success"], ev["score"]), (False, 0.))
        def done(names, **kw):
            state = dict(start, **kw)
            for key, value in (("_seated_clicks", 1), ("_retained_ticks", 2400),
                               ("_preload", setting), ("_free_load", setting)):
                state[key] = {n: ((value[n] if isinstance(value, dict) else value) if n in names
                                  else state[key][n]) for n in self.NAMES}
            return self._ev(**state)
        level = dict(_free_pitch=0., _pitch_yaw=(0., 0.))
        self.assertAlmostEqual(done(("frame_top", "frame_front"), **level)["score"], .95 * .5)
        # M3s seated on a drooped rail keep their credit; holding the rail
        # level now (current pitch) without a hands-off reading changes nothing.
        self.assertAlmostEqual(done(("sensor_left", "sensor_right"))["score"], .95 * .5)
        self.assertAlmostEqual(done(("sensor_left", "sensor_right"), _pitch_yaw=(0., 0.))["score"], .95 * .5)
        # The flag left aside costs only the acceptance share of the M3s.
        self.assertAlmostEqual(done(self.NAMES, _detected=False, **level)["score"], .95 * .875)
        # An over-torqued M3 loses its own credit, not the frame's.
        over = done(self.NAMES, **level, _peak_bit_moment_filtered={**zero, "sensor_right": .41})
        self.assertAlmostEqual(over["score"], .95 * .75)
        # A corner clamped drooped must be redone: the upward M4 loses its
        # credit; the forward M4 (bracket to post) keeps its own.
        self.assertAlmostEqual(done(self.NAMES)["score"], .95 * .75)
        # Either M4 first gives the same partial credit.
        self.assertAlmostEqual(done(("frame_front",))["score"], done(("frame_top",), **level)["score"])
        # Snugging the last screw is continuous progress, below success, with
        # no plateau at the top of the scale.
        three = ("sensor_left", "sensor_right", "frame_front")
        scores = [done(three, **level, _free_load={**{n: setting[n] for n in three}, "frame_top": f * setting["frame_top"]})["score"]
                  for f in (0., .3, .6, .89, .95, 1.)]
        self.assertTrue(all(b > a for a, b in zip(scores, scores[1:])))
        self.assertLess(scores[-1], .95)

    def test_reference_flag_must_be_back_at_its_home(self):
        # Pushed onto its stop, or a free rest just short of it: home.
        for q in (-.0001, .00004, .0004):
            env = _fake_env("mixed")
            env.state.joint_q.a[8] = q
            self.assertTrue(env._evaluate()["success"], q)
        # Short of home: not success; the return is graded by distance, from
        # the band edge down to the flag-aside value at 10 mm and beyond.
        def short(q):
            env = _fake_env("mixed")
            env.state.joint_q.a[8] = q
            return env._evaluate()
        ev = short(.001)
        self.assertFalse(ev["success"])
        self.assertFalse(ev["flag_at_home"])
        self.assertAlmostEqual(ev["score_terms"]["flag_return_grade"], 9 / 9.5)
        scores = [short(q)["score"] for q in (.00051, .001, .005, .0099)]
        self.assertTrue(all(b < a for a, b in zip(scores, scores[1:])))
        self.assertLess(scores[0], .95)
        aside = self._ev(_detected=False)["score"]
        self.assertAlmostEqual(aside, .95 * (.75 + .75 + 1 + 1) / 4)
        self.assertAlmostEqual(short(.010)["score"], aside)
        self.assertAlmostEqual(short(.089)["score"], aside)

    def test_moving_the_flag_toward_a_missed_beam_never_raises_the_score(self):
        import numpy as np
        from tighten_codex.env import TightenEnv
        def state(beam_dx, flag_q):
            # Real _optical: the ray hits the face plane beam_dx from the flag's home x.
            env = _fake_env("mixed")
            env.items.update(flag=3, flag_center_z_m=1.083, flag_face_y_m=-.057,
                             sensor_optic_local_m=(0., -.0065, -.005))
            env.state.body_q.a = np.vstack([env.state.body_q.a[:2], [.2655 + beam_dx, .07, 1.088, 0, 0, 0, 1],
                                            [.2655 + flag_q, -.063, .934, 0, 0, 0, 1]])
            env.state.joint_q.a[8] = flag_q
            env._optical = lambda: TightenEnv._optical(env)
            return env._evaluate()
        for beam_dx in (.003, .006, .008):
            at_home = state(beam_dx, 0.)
            for flag_q in (.001, .003, .006, .008):
                chased = state(beam_dx, flag_q)
                self.assertLessEqual(chased["score"], at_home["score"] + 1e-12, (beam_dx, flag_q))
                self.assertFalse(chased["success"])
        # A beam that misses home entirely gets no aim credit however the flag is placed.
        self.assertTrue(state(.008, .008)["optical"]["detected"])
        self.assertEqual(state(.008, .008)["score_terms"]["aim_at_home_grade"], 0.)

    def test_a_backed_off_screw_loses_its_click_and_its_credit_under_a_tool(self):
        setting = self._setting()
        env = _fake_env("mixed", _retained_ticks={n: 0 for n in self.NAMES},
                        _retained_anchor={n: None for n in self.NAMES})
        env._track_retention()                      # records each screw's click angle
        start = env.state.joint_q.a[2]              # frame_top turn
        # Back-rotation within the recorded 1.76 mrad (forward M4, bit in): click kept.
        env._bit_contacts["frame_top"] = 3
        env.state.joint_q.a[2] = start - .00176
        env._track_retention()
        self.assertEqual(env._seated_clicks["frame_top"], 1)
        # Loosened under the bit: the score shows it at once...
        env._preload["frame_top"] = .05 * setting["frame_top"]
        env.state.joint_q.a[2] = start - .5
        env._track_retention()
        self.assertEqual(env._seated_clicks["frame_top"], 0)
        self.assertAlmostEqual(env._free_load["frame_top"], .05 * setting["frame_top"])
        loose = env._evaluate()["score"]
        self.assertLess(loose, .95 * (3 + .9 * .05) / 4 + 1e-9)
        # ...and re-snugging to 85 % under the bit, withdrawn, without a new click, is not success.
        env.state.joint_q.a[2] = start - .4
        env._preload["frame_top"] = .85 * setting["frame_top"]
        env._bit_contacts["frame_top"] = 0
        for _ in range(12):
            env._track_retention()
        ev = env._evaluate()
        self.assertFalse(ev["success"])
        self.assertAlmostEqual(ev["score_terms"]["screw_credit"]["frame_top"], .9 * .85)
        # A reading under a tool never rises above the last hands-off one.
        env._bit_contacts["sensor_left"] = 2
        before = env._free_load["sensor_left"]
        env._preload["sensor_left"] = 2 * before
        env._track_retention()
        self.assertEqual(env._free_load["sensor_left"], before)

    def test_mixed_detection_needs_most_of_the_spot_on_the_flag(self):
        import numpy as np
        from tighten_codex.env import TightenEnv
        def optical(task, dx):
            env = _fake_env(task)
            env.items.update(flag=3, flag_center_z_m=1.083, flag_face_y_m=-.057,
                             sensor_optic_local_m=(0., -.0065, -.005))
            # Sensor square to the flag, 120.5 mm away; the ray hits the face at dx.
            env.state.body_q.a = np.vstack([env.state.body_q.a[:2], [.2655 + dx, .07, 1.088, 0, 0, 0, 1],
                                            [.2655, -.063, .934, 0, 0, 0, 1]])
            return TightenEnv._optical(env), env
        for dx, mixed, sensor in ((.0028, True, True), (.0035, True, False), (.0049, True, False),
                                  (.0051, False, False)):
            self.assertEqual(optical("mixed", dx)[0]["detected"], mixed, dx)
            self.assertEqual(optical("sensor", dx)[0]["detected"], sensor, dx)
        # The detection grade is continuous at the edge: twice the spot share.
        ev, env = optical("mixed", .00501)
        self.assertFalse(ev["detected"])
        self.assertGreater(2 * env._spot_coverage(ev), .99)

    def test_spot_coverage_is_the_share_of_the_spot_on_the_flag(self):
        import numpy as np
        env = _fake_env("mixed")
        env.items.update(flag=3, flag_center_z_m=1.083)
        env.state.body_q.a = np.vstack([env.state.body_q.a, [.2655, -.063, .934, 0, 0, 0, 1]])
        hit = lambda dx: {"detected": False, "range_m": .13, "intersection_m": [.2655 + dx, -.057, 1.083]}
        self.assertEqual(env._spot_coverage({"detected": True}), 0.)   # no ray: no share
        self.assertAlmostEqual(env._spot_coverage(hit(.0029)), 1.)
        self.assertAlmostEqual(env._spot_coverage(hit(.0071)), 0.)
        self.assertTrue(0. < env._spot_coverage(hit(.0040)) < 1.)
        self.assertAlmostEqual(env._spot_coverage(hit(.005)), .5)
        self.assertEqual(env._spot_coverage(dict(hit(0.), range_m=.15)), 0.)

    def test_hands_are_robot_links_or_drivers_a_robot_link_touches(self):
        import numpy as np
        from tighten_codex.env import TightenEnv
        env = object.__new__(TightenEnv)
        env._robot_shapes = np.array([100, 101])
        env._driver_shapes = {"M3_driver": np.array([50, 51]), "M4_driver": np.array([60])}
        env._work_groups = {"sensor": np.array([10, 11]), "frame": np.array([10, 11, 20, 21])}
        touch = lambda *pairs: env._work_contact(np.array([a for a, _ in pairs], int),
                                                 np.array([b for _, b in pairs], int))
        none, both = {"sensor": False, "frame": False}, {"sensor": True, "frame": True}
        self.assertEqual(touch(), none)
        self.assertEqual(touch((100, 20)), {"sensor": False, "frame": True})
        self.assertEqual(touch((11, 101)), both)
        self.assertEqual(touch((50, 10)), none)                    # a tool lying on the sensor
        self.assertEqual(touch((101, 51), (50, 10)), both)         # a held tool on the sensor
        self.assertEqual(touch((100, 60), (51, 20)), none)         # the other driver is the held one
        self.assertEqual(touch((100, 30)), none)                   # the flag is not the assembly


class TightenClutchTest(unittest.TestCase):
    def _env(self, angle, lock=0., slipping=False, climbing=False):
        from tighten_codex.config import Config
        from tighten_codex.env import TightenEnv
        env = object.__new__(TightenEnv)
        env.config = Config()
        env.items = {"M3_driver": {"clutch_joint": 0, "limit_nm": .35},
                     "sensor_left": {"limit_nm": .35}}
        env.state = SimpleNamespace(joint_q=_Arr([angle]))
        env._clutch = {"M3_driver": {"q": 0, "lock": lock, "slipping": slipping, "climbing": climbing,
                                     "climb_onset": {"tick": 5, "torque_nm": .351, "angle_rad": .001755},
                                     "clicks": 0, "click_ticks": []}}
        env._engaged_driver = {"sensor_left": "M3_driver"}
        env._clicks = {"sensor_left": 0}
        env._seated_clicks = {"sensor_left": 0}
        env._subfloor_ticks = {"sensor_left": 0}
        env._click_turn = {"sensor_left": None}
        env._release_slip = {"sensor_left": 0}
        env.tick = 7
        return env

    def test_locked_below_setting_then_releases_and_clicks_one_notch(self):
        k = 200.
        below = self._env(.34 / k)
        below._track_clutches()
        self.assertFalse(below._clutch["M3_driver"]["slipping"])
        self.assertFalse(below._clutch["M3_driver"]["climbing"])
        # At the setting the cam only starts up its ramp: no release yet.
        at = self._env(.351 / k)
        at._track_clutches()
        self.assertTrue(at._clutch["M3_driver"]["climbing"])
        self.assertFalse(at._clutch["M3_driver"]["slipping"])
        self.assertNotIn("releases", at._clutch["M3_driver"])
        # Over the top of the ramp: released.
        top = self._env(.1, climbing=True)
        top._track_clutches()
        self.assertTrue(top._clutch["M3_driver"]["slipping"])
        self.assertEqual(top._clutch["M3_driver"]["releases"], 1)
        # A short spike (e.g. a relock impact) that falls back before the
        # top relocks without a release.
        back = self._env(.0005, climbing=True)
        back._track_clutches()
        self.assertFalse(back._clutch["M3_driver"]["climbing"])
        self.assertFalse(back._clutch["M3_driver"]["slipping"])
        self.assertEqual(back._clutch["M3_driver"]["fallbacks"], 1)
        notch = self._env(math.pi / 3, slipping=True)
        notch._track_clutches()
        c = notch._clutch["M3_driver"]
        self.assertFalse(c["slipping"])
        self.assertAlmostEqual(c["lock"], math.pi / 3)
        self.assertEqual(notch._clicks["sensor_left"], 1)

    def test_loosening_never_releases(self):
        env = self._env(-5. / 200.)   # 5 N m the other way
        env._track_clutches()
        self.assertFalse(env._clutch["M3_driver"]["slipping"])


class TightenRobotSetupTest(unittest.TestCase):
    def test_open_selection_accepts_hostable_setups_only(self):
        from tighten_codex.kinematics import validate_setup, cradle_layout
        single_panda = {"schema": "gar.robot-setup.v1", "id": "my-single-panda",
                        "instances": {"left": {"model": "panda@1", "mount": "left_arm_mount",
                                               "controller_profile": "tighten"}},
                        "attachments": {"left_hand": {"model": "franka-hand@1", "parent": "left/flange",
                                                      "mount_profile": "franka_native"}}}
        self.assertEqual(validate_setup(single_panda)["id"], "my-single-panda")
        wrong_mount = json.loads(json.dumps(single_panda))
        wrong_mount["instances"]["left"]["mount"] = "right_arm_mount"
        with self.assertRaises(ValueError):
            validate_setup(wrong_mount)
        no_hand = json.loads(json.dumps(single_panda))
        no_hand["attachments"] = {}
        with self.assertRaises(ValueError):
            validate_setup(no_hand)
        with self.assertRaises((ValueError, TypeError)):
            validate_setup("whiteboard-g1")
        self.assertEqual(cradle_layout({"left", "right"}), {"M3_driver": .36, "M4_driver": -.36})
        self.assertEqual(set(cradle_layout({"right"}).values()), {.36, .50})

    def test_catalog_routes_custom_setups_to_the_validator(self):
        from robots.catalog import select_for_backend
        self.assertEqual(select_for_backend("tighten-codex", "dual-panda"), "dual-panda")
        with self.assertRaises(ValueError):
            select_for_backend("tighten-codex", "whiteboard-g1")


class TightenCheckpointFilesTest(unittest.TestCase):
    def test_backend_checkpoint_files_round_trip(self):
        import tempfile
        import numpy as np
        from gym_anything_robotics.backends.newton_tighten_codex import NewtonTightenCodexBackend
        backend = object.__new__(NewtonTightenCodexBackend)
        restored = {}
        backend.trace = None
        backend.env = SimpleNamespace(
            checkpoint=lambda: ({"state_joint_q": np.arange(3.)}, {"tick": 42, "tracking": {}}),
            restore=lambda arrays, meta: restored.update(arrays=arrays, meta=meta) or {"tick": meta["tick"]})
        with tempfile.TemporaryDirectory() as tmp:
            backend.output = Path(tmp)
            self.assertEqual(backend.save_checkpoint(label="before_click")["tick"], 42)
            self.assertEqual(backend.list_checkpoints(), ({"name": "before_click", "tick": 42},))
            state, info = backend.restore_checkpoint("before_click")
            self.assertEqual(state["tick"], 42)
            self.assertEqual(info["continuity"], "restored_discontinuity")
            self.assertTrue(np.array_equal(restored["arrays"]["state_joint_q"], np.arange(3.)))
            with self.assertRaises(ValueError):
                backend.save_checkpoint(label="../escape")
            with self.assertRaises(KeyError):
                backend.restore_checkpoint("missing")


class UnsupportedCheckpointHTTPTest(unittest.TestCase):
    def test_server_answers_501_not_an_empty_reply(self):
        import tempfile
        import threading
        import urllib.error
        import urllib.request
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPServer, GARHTTPService
        from tests.test_http_server import HTTPFakeBackend
        from gym_anything_robotics.backends.base import RoboticsBackend

        class NoCheckpoints(HTTPFakeBackend):
            list_checkpoints = RoboticsBackend.list_checkpoints
            save_checkpoint = RoboticsBackend.save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            env = RoboticsEnv(NoCheckpoints(Path(tmp)))
            server = GARHTTPServer(("127.0.0.1", 0), GARHTTPService(env), api_token="t")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for method, path, body in (("GET", "/v1/checkpoints", None),
                                           ("POST", "/v1/checkpoints/save", b'{"label": "x"}')):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}{path}", method=method, data=body,
                        headers={"Authorization": "Bearer t", "Content-Type": "application/json"})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request, timeout=5)
                    self.assertEqual(caught.exception.code, 501)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5); env.close()


@unittest.skipUnless(os.environ.get("GAR_TIGHTEN_PHYSICS_TESTS") == "1",
                     "set GAR_TIGHTEN_PHYSICS_TESTS=1 to build the Newton scene")
class TightenPhysicsSmokeTest(unittest.TestCase):
    def test_ordinary_scene_rests(self):
        import numpy as np
        from tighten_codex.config import Config
        from tighten_codex.env import TightenEnv
        env = TightenEnv(Config(task="mixed"))
        try:
            start = env.state.body_q.numpy().copy()
            for _ in range(960):
                env.step()
            end = env.state.body_q.numpy()
            for body in (env.items["rail"], env.items["bracket"], env.items["sensor"],
                         env.items["M3_driver"]["grip"], env.items["M4_driver"]["grip"]):
                self.assertLess(float(np.linalg.norm(end[body, :3] - start[body, :3])), 2e-4)
            self.assertIsNone(env.fault)
            self.assertFalse(env.evaluation["success"])
        finally:
            env.close()

    def test_public_action_on_every_named_setup(self):
        import tempfile
        from gym_anything_robotics import create
        from gym_anything_robotics.actions import NamedJointTarget
        for robot, joint in (("dual-fr3", "left/fr3_joint1"), ("single-fr3", "right/fr3_joint1"),
                             ("dual-panda", "left/panda_joint1")):
            with tempfile.TemporaryDirectory() as tmp:
                env = create("tighten-codex", robot=robot, cameras=(), recording=False,
                             auto_reset=False, output_dir=Path(tmp) / "s")
                try:
                    env.reset()
                    env.act_policy(NamedJointTarget({joint: 0.05}, ticks=48))
                    state = env.get_state()
                    self.assertTrue(state["ready"])
                    self.assertEqual(state["tick"], 48)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()

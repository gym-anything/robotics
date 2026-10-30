"""Ingredient mixing: discovery, robot setups, task catalogue, configuration, material laws, geometry, and the trace /
geometry / robot-configuration / variation contract of the backend.

CPU-only tests run by default (seconds). The backend tests build the full scene with its mixture on the GPU (about a
minute) and run only with GAR_GPU_TESTS=1:
  python -m pytest tests/test_ingredient_mixing.py
  GAR_GPU_TESTS=1 python -m pytest tests/test_ingredient_mixing.py
"""
import json
import math
import os
import unittest
from pathlib import Path
from tests.packages import PACKAGES

ROOT = PACKAGES


class Discovery(unittest.TestCase):
    def test_launch_and_robots(self):
        from gym_anything_robotics.launch import describe
        from robots.catalog import for_environment, environment_robots
        d = describe("ingredient-mixing")
        self.assertEqual(d["default_robot"], "dual-fr3")
        # the choices `gar inspect` lists: every one of them launches at least one task
        self.assertEqual({r["id"] for r in environment_robots("ingredient-mixing")},
                         {"dual-fr3", "single-fr3", "dual-panda", "single-panda"})
        from ingredient_mixing import robot_setup as RS
        from ingredient_mixing.config import TASKS
        for alias in (r["id"] for r in environment_robots("ingredient-mixing")):
            self.assertTrue(any(self._offered(RS, alias, p, t) for p, names in TASKS.items() for t in names), alias)
        self.assertEqual(for_environment("ingredient-mixing").id, "mix-fr3")
        self.assertEqual(for_environment("ingredient-mixing", "dual-panda").id, "mix-panda")
        for refused, word in (("g1", "G1"), ("mobile-fr3-duo", "base_twist"), ("mobile-fr3-single", "Spine"),
                              ("panda", "single-panda"), ("dual-panda-wsg50", "Franka Hand"), ("yam", "catalogue"),
                              ("single-fr3-left", "offered for no task")):
            with self.assertRaises(ValueError) as cm:
                for_environment("ingredient-mixing", refused)
            self.assertIn(word, str(cm.exception))                      # the refusal names its reason

    def test_batches_note_says_what_the_gateway_does_with_a_request_stopped_part_way(self):
        # A later command found unreachable is refused with ActionRejectedError (_rejected), which the gateway
        # answers with partial true, keeping the session ready (http_server.py recover_known_partial_batches,
        # test_eval_partial_batches.py). The note must say so and must not ask the owner to recover the session.
        from gym_anything_robotics.backends.newton_ingredient_mixing import NewtonIngredientMixingBackend as B
        from gym_anything_robotics.errors import ActionRejectedError
        from gym_anything_robotics.http_server import GARHTTPService
        text = B.BATCHES_NOTE
        self.assertTrue(GARHTTPService.recover_known_partial_batches)
        self.assertIsInstance(B._rejected(ValueError("IK unreachable")), ActionRejectedError)
        self.assertNotIn("must recover", text)
        for word in ("no recovery needed", "partial true", "completed_actions", "rejected_action_index",
                     "unexecuted_action_indices"):
            self.assertIn(word, text)

    def test_panda_rigs_use_the_35_n_hand_profile(self):
        """The Panda rigs are the catalogue's Panda with the FR3 hand's finger controller (profile `drilling`: 35 N
        per finger, as the FR3 rigs' hand), not the Panda's barcode_scanning profile (20 N per finger, which this
        environment's rollout exceeds at its first grasp)."""
        from robots.catalog import model, resolve
        for rig, arms in (("mix-panda", {"left", "right"}), ("mix-panda-single", {"right"})):
            d = resolve(rig).to_dict()
            self.assertEqual(set(d["instances"]), arms)
            for inst in d["instances"].values():
                self.assertEqual((inst["model"], inst["controller_profile"]), ("panda@1", "drilling"))
        panda = model("panda@1").to_dict()["actuation"]["profiles"]
        fr3 = model("fr3@1").to_dict()["actuation"]["profiles"]["barcode_scanning"]["joints"]
        self.assertEqual(panda["drilling"]["joints"]["panda_finger_joint1"]["effort_limit"], 35.0)
        self.assertEqual(fr3["fr3_finger_joint1"]["effort_limit"], 35.0)
        self.assertEqual(panda["barcode_scanning"]["joints"]["panda_finger_joint1"]["effort_limit"], 20.0)

    def test_one_arm_is_offered_where_its_work_stands(self):
        """The right arm alone: the tasks worked with a tool from the crock in a bowl nobody holds. The left arm
        alone: no task (the mixer needs a second hand on its switch; the crock is out of reach), so it is built and
        not offered (NOT_OFFERED), with the reason per task kept. Nobody alone at the hob. Every refusal says why and
        which robots the task is offered with."""
        from ingredient_mixing import robot_setup as RS
        from ingredient_mixing.config import TASKS
        every = [(p, t) for p, names in TASKS.items() for t in names]
        self.assertEqual(set(RS.QUALIFIED), {"dual-fr3", "single-fr3", "dual-panda", "single-panda"})
        self.assertEqual(RS.NOT_OFFERED, {"single-fr3-left": "mix-fr3-single-left"})
        offered = {alias: {t for p, t in every if self._offered(RS, alias, p, t)}
                   for alias in (*RS.QUALIFIED, *RS.NOT_OFFERED)}
        self.assertEqual(offered["dual-fr3"], {t for _, t in every})
        self.assertEqual(offered["dual-panda"], {t for _, t in every} - {"cake-mixer"})    # its mixer rehearsal failed
        right = {"dry-whisk", "sauce-blend", "muffin-fold", "pancake-whisk", "pudding-whisk"}
        self.assertEqual(offered["single-fr3"], right)
        self.assertEqual(offered["single-panda"], right)
        self.assertEqual(offered["single-fr3-left"], set())
        for alias, preset, task, word in (("dual-panda", "bake", "cake-mixer", "mixer running"),
                                          ("single-fr3", "bake", "cake-mixer", "left hand"),
                                          ("single-fr3-left", "bake", "cake-mixer", "one-armed rollout"),
                                          ("single-fr3-left", "bake", "dry-whisk", "crock"),
                                          ("single-panda", "hob", "stir-sauce", "handle")):
            with self.assertRaises(ValueError) as cm:
                RS.check_task(alias, preset, task)
            self.assertIn(word, str(cm.exception))
            self.assertIn("dual-fr3", str(cm.exception))                # where the task is offered
        self.assertEqual(RS.arms_of("single-fr3-left"), ("left",))

    @staticmethod
    def _offered(RS, alias, preset, task):
        try:
            RS.check_task(alias, preset, task)
            return True
        except ValueError:
            return False

    def test_aliases(self):
        from gym_anything_robotics.registry import _ALIASES, available_environments
        for name in ("ingredient_mixing", "mix-ingredients"):
            self.assertEqual(_ALIASES[name], "ingredient-mixing")
        self.assertIn("ingredient-mixing", {e["environment_id"] for e in available_environments()})

    def test_task_selector_and_command_floor(self):
        from ingredient_mixing.variants import resolve
        from gym_anything_robotics.backends.newton_ingredient_mixing import min_command_s
        self.assertEqual(resolve(None, "stir-sauce", 2)["preset"], "hob")        # a task alone selects its preset
        self.assertEqual(resolve()["task"], "muffin-fold")                       # the default task
        self.assertEqual(json.loads((ROOT/"ingredient_mixing/benchmark.json").read_text())["cli_task_selector"], "task")
        # a session without cameras has a floor too: the fixed cost of a command against the budget
        from gym_anything_robotics.backends.newton_ingredient_mixing import FIXED_S, UNSEEN_S, PHYSICS_RATIO_MAX, BUDGET
        bare = min_command_s(0)
        self.assertGreaterEqual(bare, 0.002)
        self.assertLessEqual((bare*PHYSICS_RATIO_MAX + FIXED_S + UNSEEN_S)/bare, BUDGET + 1e-9)
        self.assertGreaterEqual(min_command_s(1), 0.020)
        self.assertGreater(min_command_s(1), bare)                               # an observation: a longer floor
        self.assertGreater(min_command_s(4), min_command_s(1))                   # four cameras: a longer one still

    def test_command_floor_follows_the_measured_cost(self):
        """The floor in force keeps the fixed cost of a command, its physics and its observation under 200x for what
        was measured; it is never below the published one, grows with each of the three, and is capped."""
        from gym_anything_robotics.backends.newton_ingredient_mixing import floor_from, min_command_s, BUDGET, FLOOR_MAX_S
        pub = min_command_s(4)
        self.assertEqual(floor_from(4, 2.0, 20.0, pub), pub)                     # a quiet machine: the published floor
        loaded = floor_from(4, 16.6, 149.0, pub)                                 # two sessions' jobs on the card
        self.assertGreater(loaded, pub)
        self.assertLessEqual((loaded*149.0 + 16.6)/loaded, BUDGET)               # the command it allows is in budget
        self.assertGreater(floor_from(4, 20.0, 149.0, pub), loaded)
        self.assertGreater(floor_from(4, 16.6, 170.0, pub), loaded)
        self.assertEqual(floor_from(4, 16.6, 199.0, pub), FLOOR_MAX_S)           # physics alone at the budget
        # the floor is planned a tenth under the budget whatever the physics costs
        from gym_anything_robotics.backends.newton_ingredient_mixing import DESIGN_RATIO
        for ratio in (20.0, 90.0, 130.0, 160.0):
            T = floor_from(1, 3.5, ratio, 0.001, fixed_s=0.2)
            self.assertLessEqual((T*ratio + 3.5 + 0.2)/T, DESIGN_RATIO + 1e-9)
            self.assertGreater((T*ratio + 3.5 + 0.2)/T, DESIGN_RATIO - 8.0)      # and not longer than that needs
        self.assertLessEqual(floor_from(4, 500.0, 100.0, pub), FLOOR_MAX_S)
        # the fixed cost of a command is counted with the observation ...
        with_fixed = floor_from(4, 16.6, 149.0, pub, fixed_s=0.7)
        self.assertGreater(with_fixed, loaded)
        self.assertLessEqual((with_fixed*149.0 + 16.6 + 0.7)/with_fixed, BUDGET)
        # ... and is all there is without cameras: 0.7 s of fixed cost a command with the physics at 50 needs 5 ms
        # and more
        bare = floor_from(0, 0.0, 50.0, 0.001, fixed_s=0.7)
        self.assertGreater(bare, 0.004)
        self.assertLessEqual((bare*50.0 + 0.7)/bare, BUDGET)
        self.assertGreater((0.001*50.0 + 0.7)/0.001, BUDGET)                     # a 1 ms command there: over
        self.assertEqual(floor_from(0, 99.0, 50.0, 0.001, fixed_s=0.7), bare)    # no cameras: no observation counted
        bare_pub = min_command_s(0)
        self.assertEqual(floor_from(0, 0.0, 20.0, bare_pub, fixed_s=0.05), bare_pub)   # never below the published one
        self.assertGreater(floor_from(0, 0.0, 150.0, bare_pub, fixed_s=2.0), bare_pub)  # a stall under load: it rises
        # a render worker's start is counted while the session holds none: the command after an idle release
        one = min_command_s(1)
        held = floor_from(1, 2.0, 30.0, one, fixed_s=0.3)
        released = floor_from(1, 2.0, 30.0, one, fixed_s=0.3, worker_start_s=12.0)
        self.assertGreater(released, held)
        self.assertLessEqual((released*30.0 + 2.0 + 0.3 + 12.0)/released, BUDGET)
        self.assertGreater((0.043*30.0 + 2.0 + 0.3 + 12.0)/0.043, BUDGET)        # a 43 ms command there: over

    def test_floor_in_force_covers_a_worker_start_while_none_is_held(self):
        """The backend's floor in force, on the class's own code with a stand-in for the session: with a worker held
        it is the measured floor; after the idle release (no worker) it covers the start; without cameras a worker
        never counts."""
        from types import SimpleNamespace
        from gym_anything_robotics.backends import newton_ingredient_mixing as B
        cls = B.NewtonIngredientMixingBackend

        def session(cameras, held):
            s = SimpleNamespace(cameras=cameras, options={}, config=SimpleNamespace(policy_samples=16),
                                min_command_published=B.min_command_s(len(cameras)), _costs=[(2.0, 30.0, 0.3)]*bool(cameras)
                                or [(0.0, 30.0, 0.3)], _worker_start_s=B.WORKER_START_S,
                                _renderer=object() if held else None, _retained=None, _prestart=None)
            s._floor_measured = s.min_command_published
            s._worker_held = lambda: cls._worker_held(s)
            s._cost_now = lambda: cls._cost_now(s)
            return s
        floor = lambda s: cls.min_command_s.fget(s)
        held, released, bare = session(("station_rgb",), True), session(("station_rgb",), False), session((), False)
        self.assertEqual(floor(held), held.min_command_published)
        self.assertGreater(floor(released), floor(held) + 0.02)
        need = 2.0 + 0.3 + B.UNSEEN_S + B.WORKER_START_S                          # observation, fixed, worker start
        self.assertLessEqual((floor(released)*30.0 + need)/floor(released), B.BUDGET)
        self.assertEqual(floor(bare), bare.min_command_published)
        self.assertGreater(floor(bare), 0.0)

    def test_renderer_options_are_checked(self):
        """The launch option renderer_options: unknown keys fail, values are checked for type and range, nothing is
        clamped, and a launch with a bad option is refused before anything is built."""
        from gym_anything_robotics.backends import newton_ingredient_mixing as B
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        check = B.check_renderer_options
        self.assertEqual(check(None), dict(width=1280, height=720, samples=16, idle_release_s=B.IDLE_RELEASE_S))
        self.assertEqual(check({"width": 640, "height": 360, "samples": 64, "idle_release_s": 25}),
                         dict(width=640, height=360, samples=64, idle_release_s=25.0))
        self.assertIsNone(check({"idle_release_s": None})["idle_release_s"])     # kept until the session is stopped
        self.assertEqual(check({"idle_release_s": 0})["idle_release_s"], 0.0)    # not kept between operations
        self.assertEqual(check({}, policy_samples=64)["samples"], 64)            # config.policy_samples is the same number
        self.assertEqual(check({"samples": 64}, policy_samples=64)["samples"], 64)
        self.assertEqual(check({}, default_samples=32)["samples"], 32)
        for bad, word in (({"idle_relase_s": 5, "samples": 100000, "no_such_option": True}, "did you mean 'idle_release_s'"),
                          ({"no_such_option": True}, "Unknown renderer_options ['no_such_option']"),
                          ({"samples": 100000}, "1..256"), ({"samples": 0}, "1..256"), ({"samples": 16.0}, "integer"),
                          ({"samples": True}, "integer"), ({"width": 100}, "160..1920"), ({"width": 4096}, "160..1920"),
                          ({"width": "1280"}, "integer"), ({"height": 50}, "90..1080"), ({"height": 2160}, "90..1080"),
                          ({"idle_release_s": -1}, "0..86400"), ({"idle_release_s": 1e9}, "0..86400"),
                          ({"idle_release_s": float("nan")}, "0..86400"), ({"idle_release_s": "5"}, "0..86400"),
                          ({"idle_release_s": True}, "0..86400"), ([("samples", 16)], "must be an object")):
            with self.assertRaises(UnsupportedConfigurationError) as cm:
                check(bad)
            self.assertIn(word, str(cm.exception), bad)
        with self.assertRaises(UnsupportedConfigurationError) as cm:             # two names for one number: they must agree
            check({"samples": 32}, policy_samples=16)
        self.assertIn("disagree", str(cm.exception))
        with self.assertRaises(UnsupportedConfigurationError):
            check({}, policy_samples=100000)                                     # the config's value is checked as well
        # each value in range, the combination not: the session's first floor would be above the cap the floor is
        # ever raised to, so the floor could not cover its own observation
        self.assertLessEqual(B.min_command_s(4, 1280, 720, 256), B.FLOOR_MAX_S)
        self.assertEqual(check({"samples": 256}, 4)["samples"], 256)
        self.assertGreater(B.min_command_s(4, 1920, 1080, 256), B.FLOOR_MAX_S)
        with self.assertRaises(UnsupportedConfigurationError) as cm:
            check({"width": 1920, "height": 1080, "samples": 256}, 4)
        self.assertIn("first command floor", str(cm.exception))
        self.assertEqual(check({"width": 1920, "height": 1080, "samples": 256}, 0)["samples"], 256)   # no camera: no frame
        # the launch itself: refused with the reason before the session folder is made or anything is built
        import tempfile
        with tempfile.TemporaryDirectory(prefix="gar-mix-options-") as tmp:
            for options, config, word in (({"idle_relase_s": 5, "samples": 100000, "no_such_option": True}, None, "idle_relase_s"),
                                          ({"samples": 100000}, None, "1..256"),
                                          ({"samples": 32}, {"policy_samples": 16}, "disagree"),
                                          ({}, {"policy_samples": 100000}, "1..256")):
                with self.assertRaises(UnsupportedConfigurationError) as cm:
                    B.NewtonIngredientMixingBackend(task="sauce-blend", cameras=(), renderer_options=options, config=config,
                                                    output_dir=tmp + "/s")
                self.assertIn(word, str(cm.exception))
                self.assertFalse((Path(tmp)/"s").exists())
        for key in B.RENDER_OPTIONS:                                             # every key is in what the owner is told
            self.assertIn(key, B.__doc__)

    @staticmethod
    def _worker_session(tmp, idle=900.0, cameras=("station_rgb",), fail_new=0):
        """The backend class's own render-worker code on a stand-in session: no world, no GPU, renderers that are
        plain objects with a process that can be 'ended' and a frame that can be made to fail. Returns (session,
        list of the renderers it started)."""
        import threading
        from collections import deque
        from types import SimpleNamespace
        import numpy as np
        from gym_anything_robotics.backends import newton_ingredient_mixing as B
        cls = B.NewtonIngredientMixingBackend
        made = []

        class Process:
            def __init__(self):
                self.pid, self.returncode = 40000 + len(made), None

            def poll(self):
                return self.returncode

        class Renderer:
            def __init__(self, fail=0):
                self.process, self.closed, self.frames, self.fail, self.close_calls = Process(), False, 0, fail, []
                self.bad_request = 0
                made.append(self)

            def observe(self, cameras, body_q, tick, meshes=None):
                if self.process.returncode is not None:
                    raise RuntimeError("Renderer exited before replying")
                if self.bad_request:
                    self.bad_request -= 1
                    raise ValueError("mixture meshes of 70 MiB exceed the renderer's 64 MiB message")
                if self.fail:
                    self.fail -= 1
                    raise BrokenPipeError(32, "Broken pipe")
                self.frames += 1
                return {c: b"png" for c in cameras}

            def close(self, graceful=True):
                self.close_calls.append(graceful)
                self.closed = True
                if self.process.returncode is None:
                    self.process.returncode = 0
        s = object.__new__(cls)
        s.cameras = tuple(cameras)
        s.options = B.check_renderer_options({"idle_release_s": idle}, len(cameras))
        s.idle_release_s = s.options["idle_release_s"]
        s.config = SimpleNamespace(policy_samples=16, seed=0)
        world = SimpleNamespace(cont=None, tick=5, state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.zeros((2, 7)))))
        s.env = SimpleNamespace(lock=threading.RLock(), world=world, generation=0)
        s.output, s.images, s.last_timing, s.trace = Path(tmp), deque(), {}, None
        s._renderer = s._prestart = s._retained = s._retain_timer = s._retain_token = s._suspect = None
        s._reapers = []
        s.worker_events = dict(starts=0, lost=0, frames=0, frame_retries=0, released=[], last_fault=None, last_start_s=None)
        s.min_command_published = B.min_command_s(len(cameras))
        s._floor_measured = s.min_command_published
        s._costs, s._physics_ratio, s._cycle, s._worker_start_s = [(2.0, 30.0, 0.3)], 30.0, None, B.WORKER_START_S
        s.cost, s._timing, s._timed, s._over = {}, dict(cls._NO_TIMING), 0, 0
        s.fail_new = fail_new                                 # how many frames each newly started renderer fails

        def new():
            r = Renderer(fail=s.fail_new)
            s.worker_events["starts"] += 1
            return r
        s._new_renderer, s._scene = new, (lambda: None)
        return s, made

    def test_a_dead_render_worker_is_replaced(self):
        """A kept render worker whose process has exited is dropped when it is next looked at, does not count as held
        (the floor in force then covers the start of its replacement), and its replacement is started beside the next
        command's physics, or inside the next image request when nothing ran before it (an observation, the owner's
        recovery, a reset). Nothing waits for the idle release.
        On the class's own code with stand-in renderers; the real worker is ended in the GPU class RenderWorker."""
        import tempfile
        from gym_anything_robotics.backends import newton_ingredient_mixing as B
        with tempfile.TemporaryDirectory(prefix="gar-mix-worker-") as tmp:
            s, made = self._worker_session(tmp)
            join = lambda: [t.join() for t in s._reapers]
            self.assertFalse(s._worker_held())
            frames = s._render(("station_rgb",))                                 # the session's first image: a cold start
            self.assertTrue(frames["station_rgb"].is_file())
            self.assertIs(s._retained, made[0])
            status = s.worker_status()
            self.assertEqual((status["held"], status["alive"], status["pid"], status["starts"], status["lost"]),
                             (True, True, made[0].process.pid, 1, 0))
            self.assertTrue(status["idle_clock_running"])
            floor_held = s.min_command_s
            self.assertEqual(floor_held, s._floor_measured)
            # ---- the kept worker's process is ended (SIGKILL: poll() returns -9)
            made[0].process.returncode = -9
            self.assertFalse(s._worker_held())                                   # held, but dead: not counted
            self.assertFalse(s.worker_status()["alive"])
            self.assertGreater(s.min_command_s, floor_held + 0.02)               # the floor covers a worker start now
            need = 2.0 + 0.3 + B.UNSEEN_S + B.WORKER_START_S
            self.assertLessEqual((s.min_command_s*30.0 + need)/s.min_command_s, B.BUDGET)
            self.assertIn("none held", s._floor_note())
            # ---- the next command: its replacement starts beside the physics (what _execute does before stepping)
            s.last_timing = {}
            s._hold_retained()
            s._start_renderer_early()
            self.assertIsNone(s._retained)                                       # the dead one is dropped at once
            self.assertIsNotNone(s._prestart)
            self.assertTrue(s._worker_held())                                    # one is being started
            frames = s._render(("station_rgb",))                                 # ... and the command's observation
            join()
            self.assertTrue(frames["station_rgb"].is_file())
            self.assertEqual(len(made), 2)
            self.assertIs(s._retained, made[1])
            self.assertEqual((made[0].closed, made[0].close_calls), (True, [False]))   # ended, not asked
            self.assertIn("prestart_warm_s", s.last_timing)
            self.assertEqual((s.worker_events["starts"], s.worker_events["lost"], s.worker_events["frame_retries"]), (2, 1, 0))
            self.assertIn("exited (code -9)", s.worker_events["last_fault"])
            self.assertEqual(s.min_command_s, floor_held)                        # a live worker is held again
            # ---- ended again, and an image is asked for with no command before it (GET /v1/observation, the
            # owner's /v1/recover, a reset to the same task): a new worker is started inside the request
            made[1].process.returncode = -9
            s.last_timing = {}
            frames = s._render(("station_rgb",))
            join()
            self.assertTrue(frames["station_rgb"].is_file())
            self.assertEqual((len(made), s.worker_events["lost"]), (3, 2))
            self.assertIn("cold_start_s", s.last_timing)
            self.assertIs(s._retained, made[2])
            # ---- a worker that fails a frame while its process lives (a broken socket): ended, and the frame is
            # asked once more of a new worker
            made[2].fail = 1
            s.last_timing = {}
            frames = s._render(("station_rgb",))
            self.assertTrue(frames["station_rgb"].is_file())
            self.assertEqual((len(made), s.worker_events["frame_retries"], s.worker_events["lost"]), (4, 1, 3))
            self.assertEqual((made[2].closed, made[2].close_calls), (True, [False]))
            self.assertTrue(s.last_timing["frame_retry"])
            self.assertIs(s._retained, made[3])
            # ---- the frame fails on the new worker as well: the error is raised, no worker is kept, and the next
            # request starts one (it does not meet a dead one again)
            from gym_anything_robotics.errors import BackendProtocolError
            made[3].fail, s.fail_new = 1, 1
            with self.assertRaises(BackendProtocolError) as cm:
                s._render(("station_rgb",))
            self.assertIn("failed a frame twice", str(cm.exception))
            self.assertIn("the next image request starts a worker again", str(cm.exception))
            self.assertEqual(len(made), 5)
            self.assertTrue(made[3].closed and made[4].closed)
            self.assertIsNone(s._renderer)
            self.assertIsNone(s._retained)
            self.assertFalse(s._worker_held())
            s.fail_new = 0
            self.assertTrue(s._render(("station_rgb",))["station_rgb"].is_file())
            self.assertIs(s._retained, made[5])
            # ---- no worker can be started (no GPU memory, no Isaac): raised at once, and tried again next time
            made[5].process.returncode = -9
            good = s._new_renderer

            def no_start():
                raise RuntimeError("Renderer startup failed")
            s._new_renderer = no_start
            with self.assertRaises(BackendProtocolError) as cm:
                s._render(("station_rgb",))
            self.assertIn("could not be started", str(cm.exception))
            self.assertFalse(s._worker_held())
            s._new_renderer = good
            self.assertTrue(s._render(("station_rgb",))["station_rgb"].is_file())
            self.assertTrue(s._worker_held())
            self.assertEqual(s.worker_status()["frames"], 6)
            self.assertIn("lost", s._worker_note())
            # ---- a request that cannot be sent (raised on this side before the worker hears of it) is the request's
            # fault: the worker is kept and nothing is started. Twice running on the same worker, it is replaced
            n, lost, kept = len(made), s.worker_events["lost"], s._retained
            kept.bad_request = 1
            with self.assertRaises(ValueError) as cm:
                s._render(("station_rgb",))
            self.assertNotIsInstance(cm.exception, BackendProtocolError)
            self.assertIn("64 MiB", str(cm.exception))
            self.assertIs(s._retained, kept)                                     # the same worker, alive and kept
            self.assertEqual((len(made), s.worker_events["lost"], kept.closed), (n, lost, False))
            self.assertIn("request:", s.worker_events["last_fault"])
            self.assertTrue(s._render(("station_rgb",))["station_rgb"].is_file())   # the next request is drawn by it
            self.assertEqual((len(made), kept.frames), (n, 2))
            kept.bad_request = 2
            with self.assertRaises(ValueError):
                s._render(("station_rgb",))
            self.assertTrue(s._render(("station_rgb",))["station_rgb"].is_file())   # second in a row: a new worker draws it
            self.assertEqual((len(made), s.worker_events["lost"], kept.closed), (n + 1, lost + 1, True))
            self.assertEqual(s.worker_status()["frames"], 8)
            type(s).close(s)                                                     # closes the kept worker, politely
            self.assertTrue(made[-1].closed)
            self.assertEqual(made[-1].close_calls, [True])
            self.assertFalse(s.worker_status()["idle_clock_running"])

    def test_idle_release_null_keeps_the_worker_and_zero_does_not(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="gar-mix-worker-") as tmp:
            s, made = self._worker_session(tmp, idle=None)                       # null: kept until the session is stopped
            s._render(("station_rgb",))
            self.assertIs(s._retained, made[0])
            self.assertFalse(s.worker_status()["idle_clock_running"])
            s._hold_retained()
            s._resume_idle_clock()                                               # a command's physics: still no clock
            self.assertIsNone(s._retain_timer)
            s._render(("station_rgb",))
            self.assertEqual((len(made), made[0].frames), (1, 2))
            type(s).close(s)
            self.assertTrue(made[0].closed)
            s, made = self._worker_session(tmp, idle=0)                          # 0: not kept between operations
            s._render(("station_rgb",))
            [t.join() for t in s._reapers]
            self.assertIsNone(s._retained)
            self.assertEqual((made[0].closed, made[0].close_calls), (True, [True]))
            self.assertFalse(s._worker_held())
            self.assertGreater(s.min_command_s, s._floor_measured)               # every command starts a worker: floored for it
            s._render(("station_rgb",))
            [t.join() for t in s._reapers]
            self.assertEqual(len(made), 2)
            self.assertIn("not kept", s._worker_note())
            s, made = self._worker_session(tmp, idle=0.05)                       # the idle clock itself
            s._render(("station_rgb",))
            self.assertIs(s._retained, made[0])
            import time
            deadline = time.monotonic() + 5.0
            while s._retained is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNone(s._retained)
            self.assertTrue(made[0].closed)
            self.assertEqual(s.worker_events["released"][-1], "idle")

    def test_the_policy_is_told_where_the_session_stands_against_the_limit(self):
        """What the 200x limit rests on, said to the policy: hardware.budget of its capabilities has the physics ratio
        the session has measured, what a command at the floor in force is expected to cost and whether that is inside
        the limit; when the physics alone is over it the floor stands at its cap, commands are accepted, and the
        session says that it is outside. Every observation says whether the command before it ran over
        (proprioception command_timing). On the class's own code with a stand-in session."""
        import tempfile
        import time
        from gym_anything_robotics.backends import newton_ingredient_mixing as B
        cls = B.NewtonIngredientMixingBackend
        with tempfile.TemporaryDirectory(prefix="gar-mix-budget-") as tmp:
            s, _ = self._worker_session(tmp)
            s._renderer = object()                                               # a worker is held
            s._costs = []
            b = s._budget()                                                      # nothing timed yet: not known
            self.assertEqual(b["limit_wall_per_sim"], B.BUDGET)
            self.assertIsNone(b["physics_wall_per_sim"])
            self.assertIsNone(b["inside_budget"])
            self.assertEqual((b["commands_timed"], b["commands_over_budget"]), (0, 0))
            self.assertIn("no policy command has been timed", s._budget_note())
            s._costs = [(2.0, 30.0, 0.3)]                                        # a quiet machine
            s._floor_measured = B.floor_from(1, 2.0, 30.0, s.min_command_published, 0.3 + B.UNSEEN_S)
            b = s._budget()
            self.assertEqual(b["physics_wall_per_sim"], 30.0)
            self.assertTrue(b["inside_budget"])
            self.assertFalse(b["floor_at_its_cap"])
            self.assertLessEqual(b["at_the_floor_wall_per_sim"], B.DESIGN_RATIO + 0.1)
            self.assertAlmostEqual(b["at_the_floor_wall_per_sim"], 30.0 + b["command_overhead_s"]/b["min_command_s"], delta=0.1)
            self.assertIn("inside the limit: yes", s._budget_note())
            s._costs = [(2.0, 250.0, 0.3)]                                       # the physics alone over the limit
            s._floor_measured = B.floor_from(1, 2.0, 250.0, s.min_command_published, 0.3 + B.UNSEEN_S)
            b = s._budget()
            self.assertEqual(s.min_command_s, B.FLOOR_MAX_S)                     # the floor stands at its cap ...
            self.assertTrue(b["floor_at_its_cap"])
            self.assertEqual(b["physics_wall_per_sim"], 250.0)
            self.assertFalse(b["inside_budget"])                                 # ... and the session says it is outside
            self.assertGreater(b["at_the_floor_wall_per_sim"], B.BUDGET)
            self.assertIn("inside the limit: NO", s._budget_note())
            self.assertIn("accepted and run slower", b["note"])
            # a command that ran over is counted and the observation after it says so; one inside the limit says 0
            self.assertEqual(s._command_timing(), dict(wall_s=0.0, sim_s=0.0, wall_per_sim=0.0, physics_wall_per_sim=0.0,
                                                       over_budget=0))
            s.last_timing = {"surface_s": 0.2, "frame_s": 0.8}
            s._cycle = dict(t0=time.perf_counter() - 30.0, floor=2.0, physics_s=28.0, sim_s=0.1, n=1, decoded=True)
            s._note_cost(1.0)
            t = s._command_timing()
            self.assertEqual((t["over_budget"], t["sim_s"]), (1, 0.1))
            self.assertAlmostEqual(t["wall_per_sim"], 300.0, delta=2.0)
            self.assertAlmostEqual(t["physics_wall_per_sim"], 280.0, delta=0.01)
            self.assertEqual((s._timed, s._over, s.cost["over_budget"]), (1, 1, 1.0))
            s._cycle = dict(t0=time.perf_counter() - 1.0, floor=2.0, physics_s=0.6, sim_s=0.1, n=1, decoded=True)
            s._note_cost(0.3)
            self.assertEqual(s._command_timing()["over_budget"], 0)
            self.assertEqual((s._timed, s._over), (2, 1))
            self.assertEqual((s._budget()["commands_timed"], s._budget()["commands_over_budget"]), (2, 1))
            # at the margin the 0.05 s the session cannot time decides for "over"
            s._cycle = dict(t0=time.perf_counter() - 19.99, floor=2.0, physics_s=18.0, sim_s=0.1, n=1, decoded=True)
            s._note_cost(1.0)
            self.assertEqual(s._command_timing()["over_budget"], 1)
            self.assertLess(s._command_timing()["wall_per_sim"], 200.1)
            # every value the policy's filter passes on is a number, and the fields are the declared ones
            self.assertEqual(set(s._command_timing()), set(cls._NO_TIMING))
            self.assertTrue(all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in s._command_timing().values()))
            s._renderer = None

    @classmethod
    def _gateway_session(cls, tmp):
        """The stand-in session of _worker_session (one policy camera, renderers that can be 'ended') with a world
        that only counts ticks, behind the repository's own facade and the gateway's own service object (what the
        HTTP routes call). The backend class's own decoding, precheck, floor, command and observation code runs; no
        physics and no GPU. Returns (session, renderers, service, the policy's principal)."""
        from types import SimpleNamespace
        from gym_anything_robotics.access import Principal
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPService
        from gym_anything_robotics.types import CameraSpec, Capabilities
        s, made = cls._worker_session(tmp)
        dt = 0.001
        s.config = SimpleNamespace(policy_samples=16, seed=0, rigid_dt=dt, robot="dual-fr3", preset="blend",
                                   task="sauce-blend", variant="")
        w = s.env.world
        w.tick, w.joints, w.reflexes, w.refused_targets, w._reflex_until, w.load_cfg = 0, {}, [], 0, {}, {}
        w.sim_time, w.fault = 0.0, None
        w.scene = SimpleNamespace(tools={}, b=SimpleNamespace())
        w.gripper_joints = lambda arm: ()
        w.set_grip_force = lambda arm, force: None

        def state(evaluate=False):
            return dict(tick=w.tick, time_s=w.tick*dt, generation=s.env.generation, joints={}, joint_torques_nm={},
                        tools={}, finite=True, fault=None)

        def act(targets, ticks):
            w.tick += int(ticks)
            w.sim_time = w.tick*dt
            return state()
        s.env.get_state, s.env.act, s.env.joints = state, act, {}
        s.arms, s.arm_model = ("left", "right"), "FR3"
        s.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta", "keyboard", "raw"})
        s.camera_specs = {"station_rgb": CameraSpec("station_rgb", "back_wall", 1280, 720, 1280, 720, (.036, .02025),
                                                    (.03, .03))}
        s._capabilities = Capabilities(
            "ingredient-mixing", "newton", "dual-fr3", s.kinds, frozenset({"proprioception", "rgb"}), True, False, True,
            s.cameras, tuple(s.camera_specs.values()),
            notes={k: k for k in ("gripper", "keyboard", "batches", "load", "instruments", "command_timing")})
        env = RoboticsEnv(s, auto_reset=False)
        return s, made, GARHTTPService(env), Principal("policy", "policy")

    def test_a_keyed_command_sent_again_gets_its_stored_reply_after_the_floor_has_risen(self):
        """A policy command that was carried out and is sent again with the same Idempotency-Key gets the stored reply
        of the first send, also after the floor in force has risen above its length in between; nothing runs twice.
        The gateway decodes a request before it looks the key up, so the decoding must not raise the floor's refusal: a
        client told "rejected before execution" would send the motion again, longer, and the motion would happen twice.
        The floor is raised by ending the session's render worker. The same body under a new key is refused, with
        the floor, and so is a batch whose second command is too short, whole and with nothing run. Through the
        gateway's own service object, on the backend class's own code with a stand-in session."""
        import tempfile
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError, UnsupportedActionError
        with tempfile.TemporaryDirectory(prefix="gar-mix-retry-") as tmp:
            s, made, service, policy = self._gateway_session(tmp)
            w = s.env.world
            floor = lambda: service.capabilities(policy)["capabilities"]["hardware"]["min_command_s"]
            ticks_of = lambda seconds: max(1, int(math.ceil(seconds*1000 - 1e-9)))
            hold = lambda ticks: {"type": "named_joint_target", "targets_si": {}, "ticks": ticks}
            send = lambda body, key: service.action(body, idempotency_key=key, principal=policy)
            store = service.history.store                                         # the gateway's own record of commands
            rows = lambda: [store.get(r["id"]) for r in store.list(limit=100)["records"]]
            self.assertTrue(service.observation(policy)["observation"]["images"])  # the session holds a live worker
            floor_held = floor()
            n = ticks_of(floor_held)
            # ---- a command with a key is carried out
            first = send(hold(n), "k-1")
            self.assertEqual((first["ok"], first["committed"]), (True, True))
            self.assertTrue(first["request_id"])
            self.assertEqual(w.tick, n)
            self.assertAlmostEqual(first["observation"]["proprioception"]["command_timing"]["sim_s"], n*1e-3, places=9)
            # ---- sent again at once: the stored reply, nothing runs
            self.assertEqual(send(hold(n), "k-1"), first)
            self.assertEqual(w.tick, n)
            # ---- the floor rises above the command's length (the render worker is ended)
            [r for r in made if not r.closed][-1].process.returncode = -9
            floor_lost = floor()
            self.assertGreater(ticks_of(floor_lost), n)
            # ---- sent again with its key: still the stored reply, not "rejected before execution"; nothing runs
            again = send(hold(n), "k-1")
            self.assertEqual(again, first)
            self.assertEqual((again["committed"], again["request_id"]), (True, first["request_id"]))
            self.assertEqual(w.tick, n)
            self.assertTrue(service.health()["ready"])
            executed = [r for r in rows() if r["kind"] == "action"]
            self.assertEqual([r["outcome"] for r in executed], ["completed"])     # one command on record, carried out once
            # ---- the same body as a new command (another key, and none) is refused under the floor in force, told
            # why, with nothing run; the gateway's record of it says failed and the session stays ready
            for key in ("k-2", None):
                with self.assertRaises(ActionRejectedError) as cm:
                    send(hold(n), key)
                self.assertIn(f"must last >= {floor_lost*1e3:.0f} ms", cm.exception.policy_reason)
                self.assertIn("no render worker is held", cm.exception.policy_reason)
            self.assertEqual(w.tick, n)
            self.assertIsNone(s._cycle)                                          # nothing is being timed
            self.assertTrue(service.health()["ready"])
            refused = [r for r in rows() if r["kind"] == "action" and r["outcome"] == "failed"]
            self.assertEqual(len(refused), 2)
            self.assertTrue(all(r["error"]["type"] == "ActionRejectedError" and
                                r["before"]["state"]["tick"] == r["after"]["state"]["tick"] == n for r in refused))
            # ---- a refused request is not stored under its key: sent again it is judged again (and refused again)
            with self.assertRaises(ActionRejectedError):
                send(hold(n), "k-2")
            # ---- a batch whose second command is too short: refused whole, nothing of it runs, no fault is latched
            m = ticks_of(floor_lost)
            with self.assertRaises(ActionRejectedError) as cm:
                send({"actions": [hold(m), hold(1)]}, "k-3")
            self.assertIn("command 2 of 2", cm.exception.policy_reason)
            self.assertIn("nothing in this request was executed", cm.exception.policy_reason)
            self.assertIn("must last", cm.exception.policy_reason)
            s.arms = ("right",)                                                  # a one-arm rig, for one request
            with self.assertRaises(UnsupportedActionError) as cm:               # an arm the rig does not have: the same way
                send({"actions": [hold(m), {"type": "gripper", "arm": "left", "half_width_m": 0.02,
                                            "duration_s": 0.5}]}, None)
            self.assertIn("command 2 of 2", str(cm.exception))
            s.arms = ("left", "right")
            self.assertEqual(w.tick, n)
            self.assertTrue(service.health()["ready"])
            # ---- a key that was used for another body is the gateway's own refusal, whatever the floor
            with self.assertRaises(ValueError) as cm:
                send(hold(n + 1), "k-1")
            self.assertIn("already used for a different action", str(cm.exception))
            self.assertNotIsInstance(cm.exception, ActionRejectedError)
            self.assertEqual(w.tick, n)
            # ---- the stored reply left a decoded request that never ran. It is not applied to a command that does
            # not come through the decoding (local use): that command is judged and timed as its own
            self.assertEqual(send(hold(n), "k-1"), first)
            self.assertIsNotNone(s._cycle)
            self.assertEqual(s._cycle["n"], 0)
            with self.assertRaises(ActionRejectedError):                         # too short now, by the floor in force
                s.act_policy(NamedJointTarget({}, n))
            with self.assertRaises(ActionRejectedError):                         # a noted refusal that never ran ...
                send(hold(1), "k-4")
            access = s.policy_access()
            access.decode_actions(hold(1))                                       # ... and one that was only decoded
            self.assertIsNotNone(s._cycle["refusal"])
            s.act_policy(NamedJointTarget({}, m))                                # ... do not refuse another command
            self.assertEqual(w.tick, n + m)
            s.observation_from_state(s.get_state(), sequence=0)
            self.assertEqual((s.cost["commands"], s.cost["sim_s"]), (1, round(m*1e-3, 6)))
            self.assertLess(s.cost["wall_s"], 5.0)                               # timed from its own start
            # ---- and the next command through the gateway runs, at the floor in force, with its picture
            before, k = w.tick, ticks_of(floor())
            reply = send(hold(k), "k-5")
            self.assertTrue(reply["committed"])
            self.assertTrue(reply["observation"]["images"])
            self.assertEqual(w.tick, before + k)
            self.assertEqual(send(hold(k), "k-5"), reply)
            self.assertEqual(w.tick, before + k)
            type(s).close(s)


    def test_policy_may_send_the_raw_set_load_and_nothing_else_raw(self):
        """The policy's access admits the raw action only where the backend offers it, decodes it with the shared codec
        (exactly `type` and `request`), keeps the shared allowlist for everything else, and lists raw in what it tells
        the policy. Whether the request is a set_load the arm accepts is the backend's check (the GPU tests)."""
        from gym_anything_robotics.access import AccessDeniedError
        from gym_anything_robotics.actions import NamedJointTarget, RawAction
        from gym_anything_robotics.backends.newton_ingredient_mixing import MixPolicyAccess
        seen = []
        with_raw = MixPolicyAccess("dual-fr3", (), {}, frozenset({"named_joint_target", "raw"}), lambda: "", {},
                                   precheck=lambda actions: seen.append(actions))
        load = {"type": "raw", "request": {"op": "set_load", "arm": "left", "mass_kg": 1.0, "com_m": [0, 0, 0.05]}}
        one = with_raw.decode_actions(load)
        self.assertIsInstance(one, RawAction)
        self.assertEqual(one.request["mass_kg"], 1.0)
        both = with_raw.decode_actions({"actions": [{"type": "named_joint_target", "targets_si": {}, "ticks": 5}, load]})
        self.assertEqual([type(a) for a in both], [NamedJointTarget, RawAction])
        self.assertEqual([len(a) for a in seen], [1, 2])                        # the whole request reaches the precheck
        self.assertIn("raw", with_raw.capabilities()["action_types"])
        self.assertEqual(with_raw.help()["action_types"]["raw"]["example"], load)
        without = MixPolicyAccess("dual-fr3", (), {}, frozenset({"named_joint_target"}), lambda: "", {})
        with self.assertRaises(AccessDeniedError):
            without.decode_actions(load)
        self.assertNotIn("raw", without.capabilities()["action_types"])
        with self.assertRaises(AccessDeniedError):                               # the shared allowlist still holds
            with_raw.decode_actions({"type": "joint_effort", "arm": "left", "efforts_nm": [0.0]*7, "duration_s": 1.0})
        for bad in (dict(load, extra=1), {"type": "raw", "request": {}}, {"actions": [load, "oops"]},
                    {"actions": [load, {"actions": [load]}]}):
            with self.assertRaises(ValueError):
                with_raw.decode_actions(bad)

    def test_refused_resets_are_refusals_and_change_nothing(self):
        """An owner reset the backend refuses (an unknown selector, task, preset or variant, a task of another preset,
        a variant of another task, a seed outside the catalogue, a task this robot is not offered for) is raised as
        ActionRejectedError before anything is touched: the gateway answers 400 and the session stays usable (it keeps
        a session usable only for that error; a ValueError would lock it).
        On the class's own code with a stand-in that has nothing but its configuration: a reset that touched anything
        else would fail with AttributeError."""
        from types import SimpleNamespace
        from gym_anything_robotics.backends.newton_ingredient_mixing import NewtonIngredientMixingBackend as cls
        from gym_anything_robotics.errors import ActionRejectedError
        from ingredient_mixing.config import Config

        def stand_in(**kw):
            s = SimpleNamespace(config=Config(**kw))
            s._reset_selection = lambda seed, options: cls._reset_selection(s, seed, options)
            return s
        mixer = stand_in(preset="bake", task="cake-mixer")
        for seed, options, word in ((None, {"task": "no-such-task"}, "no-such-task"),
                                    (None, {"colour": "red"}, "colour"),
                                    (123456, None, "0..9999"),
                                    (None, {"preset": "bake", "task": "stir-sauce"}, "not a bake task"),
                                    (None, {"variant": "no-such-variant"}, "Unknown variant"),
                                    (None, {"task": "muffin-fold", "variant": "honey"}, "belongs to"),
                                    (None, {"task": "stir-sauce", "variant": "mat"}, "belongs to"),
                                    (None, {"task": ["sauce-blend"]}, "must be a name"),
                                    (True, None, "integer"),
                                    (None, {"preset": "no-such-preset"}, "Unknown preset")):
            with self.assertRaises(ActionRejectedError) as cm:
                cls.reset(mixer, seed=seed, options=options)
            self.assertIn("Reset refused, nothing changed", str(cm.exception))
            self.assertIn(word, str(cm.exception))
        one_arm = stand_in(preset="bake", task="sauce-blend", robot="single-fr3")
        with self.assertRaises(ActionRejectedError) as cm:
            cls.reset(one_arm, options={"task": "stir-sauce"})                   # a two-arm task on the one-arm rig
        self.assertIn("single-fr3", str(cm.exception))
        # what a reset selects: a variant not named stays while the task stays and goes when the task changes
        honey = stand_in(preset="bake", task="sauce-blend", variant="honey", seed=4)
        sel = lambda s, seed=None, **options: cls._reset_selection(s, seed, options or None)
        self.assertEqual(sel(honey), dict(preset="bake", task="sauce-blend", seed=4, variant="honey"))
        self.assertEqual(sel(honey, seed=9)["variant"], "honey")
        self.assertEqual(sel(honey, task="sauce-blend")["variant"], "honey")
        self.assertEqual(sel(honey, task="dry-whisk"), dict(preset="bake", task="dry-whisk", seed=4, variant=""))
        self.assertEqual(sel(honey, preset="hob"), dict(preset="hob", task="stir-sauce", seed=4, variant=""))
        self.assertEqual(sel(honey, variant="plain")["variant"], "")
        self.assertEqual(sel(honey, variant="pet-meal")["variant"], "pet-meal")
        self.assertEqual(sel(mixer, variant="mat"), dict(preset="bake", task="cake-mixer", seed=0, variant="mat"))
        self.assertEqual(sel(mixer, variant="spelt")["task"], "pancake-whisk")   # the variant's one task
        self.assertEqual(sel(mixer, task="stir-sauce")["preset"], "hob")         # a task alone selects its preset

    def test_robot_setup_objects(self):
        from robots.catalog import resolve, select_for_backend
        for alias, rig in (("dual-fr3", "mix-fr3"), ("single-fr3", "mix-fr3-single"),
                           ("dual-panda", "mix-panda"), ("single-panda", "mix-panda-single")):
            d = resolve(rig).to_dict()
            self.assertEqual(select_for_backend("ingredient-mixing", rig), alias)   # the rig's own name
            self.assertEqual(select_for_backend("ingredient-mixing", d), alias)
            d["id"] = "my-copy"
            self.assertEqual(select_for_backend("ingredient-mixing", d), alias)  # content decides, not the id
        # the left FR3 alone builds (its rig resolves) and is offered for no task: its name, its rig's name and an
        # equal object under any id are refused with that reason, by the launcher and by the backend
        left = resolve("mix-fr3-single-left").to_dict()
        copy = dict(left, id="my-left-arm")
        for selected in ("single-fr3-left", "mix-fr3-single-left", left, copy):
            with self.assertRaises(ValueError) as cm:
                select_for_backend("ingredient-mixing", selected)
            self.assertIn("offered for no task", str(cm.exception))
            self.assertIn("single-fr3", str(cm.exception))                       # what to use instead
        from gym_anything_robotics.backends.newton_ingredient_mixing import NewtonIngredientMixingBackend
        from gym_anything_robotics.errors import UnsupportedConfigurationError
        for selected in ("single-fr3-left", "mix-fr3-single-left", copy):
            with self.assertRaises(UnsupportedConfigurationError) as cm:         # before anything is built
                NewtonIngredientMixingBackend(robot=selected, task="cake-mixer", cameras=())
            self.assertIn("offered for no task", str(cm.exception))
        # the catalogue's rigs of the same content under another environment's id
        self.assertEqual(select_for_backend("ingredient-mixing", "drill-panda"), "dual-panda")
        self.assertEqual(select_for_backend("ingredient-mixing", "drill-fr3"), "dual-fr3")
        for other, word in (("pour-panda", "'drilling'"),              # the Panda's 20 N hand profile
                            ("label-fr3", "'barcode_scanning'"),
                            ("scan-g1", "arm model")):
            with self.assertRaises(ValueError) as cm:
                select_for_backend("ingredient-mixing", other)
            self.assertIn(word, str(cm.exception))
        mixed = json.loads((ROOT/"robots/rigs/mix-fr3.json").read_text())
        mixed["instances"]["left"].update(model="panda@1", controller_profile="drilling")
        with self.assertRaises(ValueError) as cm:
            select_for_backend("ingredient-mixing", mixed)                       # one FR3 and one Panda: not built
        self.assertIn("same model", str(cm.exception))
        bad = json.loads((ROOT/"robots/rigs/mix-fr3.json").read_text())
        bad["instances"]["left"]["controller_profile"] = "classroom"
        with self.assertRaises(ValueError):
            select_for_backend("ingredient-mixing", bad)                         # no such FR3 profile
        swapped = json.loads((ROOT/"robots/rigs/mix-fr3.json").read_text())
        swapped["instances"]["left"]["mount"], swapped["instances"]["right"]["mount"] = (
            swapped["instances"]["right"]["mount"], swapped["instances"]["left"]["mount"])
        with self.assertRaises(ValueError):
            select_for_backend("ingredient-mixing", swapped)

    def test_task_catalogue(self):
        from gym_anything_robotics import tasks
        from ingredient_mixing.config import TASKS
        self.assertEqual(sum(len(v) for v in TASKS.values()), 8)
        for preset, names in TASKS.items():
            for t in names:
                spec = tasks.resolve("ingredient-mixing", {"preset": preset, "task": t, "seed": 7})
                self.assertEqual(spec.task_id, f"ingredient-mixing/v1/{preset}/{t}/seed-7")
                self.assertIn(spec.split, ("train", "validation", "test"))
                self.assertTrue(spec.description)
        for bad in ({"preset": "hob", "task": "muffin-fold"}, {"preset": "bar"}, {"seed": -1}, {"seed": 10000},
                    {"task": "stir-drink"}):                                     # the drinks task was not built
            with self.assertRaises(ValueError):
                tasks.resolve("ingredient-mixing", bad)

    def test_variants_are_catalogue_selectors(self):
        """A recipe variant is a selector of the task catalogue (`gar tasks` lists it and a reset can select it): it
        resolves to a launch that builds it, has the split of its task and seed, and is refused for a task it does not
        belong to."""
        from gym_anything_robotics import tasks
        from ingredient_mixing import variants as V
        from ingredient_mixing.config import Config, TASKS, VARIANTS, VARIANT_TASKS
        cat = tasks.catalog("ingredient-mixing")
        self.assertIn("variant", cat["selectors"])
        fields = V.catalogue_fields()                       # `python -m ingredient_mixing.variants` rewrites the file
        for key, value in fields.items():
            self.assertEqual(cat[key], value, f"benchmark.json {key} does not follow config: run "
                                              "python -m ingredient_mixing.variants")
        self.assertEqual(cat["selector_choices"]["variant"], ["plain"] + [v for v in VARIANTS if v])
        self.assertEqual(cat["count"], len(V.pairs()))
        self.assertEqual(len([p for p in V.pairs() if not p[2]]), sum(len(v) for v in TASKS.values()))
        for preset, task, variant in V.pairs():
            spec = tasks.resolve("ingredient-mixing", {"task": task, "seed": 5, "variant": variant or "plain"})
            plain = tasks.resolve("ingredient-mixing", {"task": task, "seed": 5})
            self.assertEqual(spec.split, plain.split)                            # the same station: the same split
            options = spec.launch.backend_options
            self.assertEqual(options.get("variant", ""), variant)
            self.assertEqual(Config(**{k: options[k] for k in ("preset", "task", "seed")}, variant=variant).variant, variant)
            if variant:
                self.assertEqual(spec.task_id, f"ingredient-mixing/v1/{preset}/{task}/variant-{variant}/seed-5")
                self.assertNotEqual(spec.description, plain.description)         # it says what is different
            else:
                self.assertEqual(spec.task_id, plain.task_id)
        self.assertEqual(tasks.resolve("ingredient-mixing", {"variant": "honey"}).selection["task"], "sauce-blend")
        self.assertEqual(tasks.resolve("ingredient-mixing", {"variant": "spelt"}).selection["task"], "pancake-whisk")
        for bad in ({"task": "muffin-fold", "variant": "honey"}, {"task": "sauce-blend", "variant": "spelt"},
                    {"task": "stir-sauce", "variant": "mat"}, {"variant": "no-such-variant"}, {"variant": 3}):
            with self.assertRaises(ValueError):
                tasks.resolve("ingredient-mixing", bad)
        for variant, names in VARIANT_TASKS.items():
            self.assertIn(variant, VARIANTS)
            self.assertTrue(set(names) <= {t for v in TASKS.values() for t in v})


    def test_stir_sauce_progress_is_of_the_outcome(self):
        """stir-sauce's partial score: nothing for an untouched or a merely switched-off pan, rising with the cooking,
        falling with burnt-on area and spilt sauce, and the hob's switch worth something only over a cooked sauce."""
        from ingredient_mixing.task import cook_progress, TASKS
        bottom = 181.46
        p = lambda **k: cook_progress(**{**dict(cooked=1.0, spilt=0.0, burnt_cm2=0.0, bottom_cm2=bottom, pan_in_place=True,
                                                hob_off=True, cooked_ok=True), **k})["progress"]
        self.assertEqual(p(cooked=0.0, hob_off=False, cooked_ok=False), 0.0)                    # untouched
        self.assertEqual(p(cooked=0.0, cooked_ok=False), 0.0)                                   # knob off, nothing else
        self.assertEqual(p(burnt_cm2=bottom), 0.0)                                              # left alone: all burnt
        self.assertEqual(p(), 1.0)
        self.assertAlmostEqual(p(hob_off=False), 0.85)
        partial = [p(cooked=g, hob_off=h, cooked_ok=False) for g in (0.2, 0.5, 0.8) for h in (False, True)]
        self.assertEqual(partial[0::2], partial[1::2])                                          # the knob counts once cooked
        self.assertEqual(partial[0::2], sorted(partial[0::2]))
        for k, values in (("burnt_cm2", (0.0, 2.6, 110.0, 181.0)), ("spilt", (0.0, 0.015, 0.5, 0.999))):
            got = [p(**{k: x}) for x in values]
            self.assertEqual(got, sorted(got, reverse=True))
            self.assertGreater(got[1], got[2])
        spec = TASKS["stir-sauce"]
        self.assertEqual(spec["reward"], "cook")
        self.assertNotIn("lumps", spec["criteria"])
        self.assertTrue(spec["criteria"]["contained"]["below_rim"])
        self.assertTrue(spec["criteria"]["vessel_in_place"]["on_hob"])
        self.assertEqual(spec["criteria"]["boil_s"], 180.)
        for other in ("reheat-porridge",):
            self.assertNotIn("boil_s", TASKS[other]["criteria"])
        for other in ("dry-whisk", "sauce-blend", "reheat-porridge"):                 # the rest of the catalogue
            self.assertEqual(TASKS[other]["criteria"]["vessel_in_place"], 0.05)
            self.assertEqual(TASKS[other]["criteria"]["contained"], 0.01)
        for word in ("88", "95", "cm2", "lump", "knob to 0", "50 mm"):
            self.assertNotIn(word, spec["instruction"])

    def test_layer_shares_read_the_sauce_by_height(self):
        """`cook` reads the least gelatinised of equal-mass layers by height: a sauce cooked in a layer on its bottom
        under raw milk reads near nothing; an evenly cooked one reads its share in every layer."""
        import numpy as np
        from ingredient_mixing.task import layer_shares
        z = np.linspace(0.0, 0.03, 3000)
        mass, wet = np.ones_like(z), np.full_like(z, 0.05)
        bottom = np.where(z < 0.008, 0.05, 0.0)                     # the bottom quarter fully gelatinised, the rest raw
        self.assertEqual(layer_shares(z, bottom, wet, mass, 3)[0], 0.8)
        self.assertEqual(min(layer_shares(z, bottom, wet, mass, 3)), 0.0)
        self.assertEqual(layer_shares(z, 0.6*wet, wet, mass, 3), [0.6, 0.6, 0.6])
        self.assertEqual(layer_shares(z[::-1], 0.6*wet, wet, mass, 4), [0.6]*4)         # order of the points is free
        self.assertEqual(layer_shares(z[:0], z[:0], z[:0], z[:0], 2), [0.0, 0.0])


    def test_what_stands_on_the_counter(self):
        """Every task has its recipe's packs beside the work; every one is a known thing standing on the counter and
        clear of the vessel, the crock and the mixer's stand; the room's tables hold together."""
        from ingredient_mixing import params as P
        from ingredient_mixing.config import TASKS
        for preset, names in TASKS.items():
            for t in names:
                props = P.PROPS[t]
                self.assertGreaterEqual(len(props), 3)
                for kind, dx, y, yaw in props:
                    self.assertIn(kind, P.PROP)
                    hx, hy = P.PROP[kind]["half"]
                    reach = math.hypot(hx, hy)
                    x = P.STATION_X[preset] + dx
                    self.assertGreater(y - reach, 0.0, (t, kind))                    # on the counter, behind its front edge
                    self.assertLess(y + reach, P.COUNTER_DEPTH, (t, kind))
                    for cx, cy, r in ((P.STATION_X[preset], 0.30, 0.16), (P.STATION_X[preset] + 0.37, 0.25, 0.11),
                                      (P.STATION_X[preset] - 0.34, 0.38, 0.11)):          # vessel, crock, mixer's stand
                        self.assertGreater(math.hypot(x - cx, y - cy), r + reach, (t, kind))
                    # ... and clear of the towel that hangs from its rail over the counter's back: the props are free
                    # bodies, and one standing inside the towel's slab is thrown over when the scene is built
                    r_ = P.TOWEL_RAIL
                    front = P.ROOM["y1"] - r_["off_wall"] - r_["rod_r"] - 0.004
                    ca, sa = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
                    ex, ey = (hx, hy) if P.PROP[kind].get("round") else (hx*ca + hy*sa, hx*sa + hy*ca)
                    under = r_["towel_x"][0] - ex < x < r_["towel_x"][1] + ex
                    if under and P.COUNTER_Z + P.PROP[kind]["h"] > r_["z"] - r_["towel_drop"][1]:
                        self.assertLess(y + ey, front - 0.002, (t, kind))          # 2 mm clear (the nearest has 2.3)
        R, W, D = P.ROOM, P.WINDOW, P.DOOR
        self.assertTrue(R["y0"] < W["y"][0] < W["y"][1] < R["y1"] and 0 < W["z"][0] < W["z"][1] < R["z1"])
        self.assertTrue(R["x0"] < D["x"][0] < D["x"][1] < R["x1"] and D["z"] < R["z1"])
        t = P.WORK_TABLE
        self.assertTrue(R["x0"] < t["x"][0] and t["x"][1] < R["x1"] and R["y0"] < t["y"][0] and t["y"][1] < -0.6)


    def test_instructions_do_not_tell_the_motion(self):
        """An instruction names the goal and the judged outcome; it never prescribes a stroke count or a path."""
        from ingredient_mixing.config import Config, TASKS
        from ingredient_mixing.task import instruction
        for preset, names in TASKS.items():
            for t in names:
                text = instruction(Config(preset=preset, task=t))
                self.assertGreater(len(text), 60)
                for word in ("clockwise", "figure-eight", "waypoint"):
                    self.assertNotIn(word, text.lower())

    def test_every_variant_has_its_own_instruction(self):
        """What a policy is told matches what is in the vessel: honey's text is not the plain sauce's ("ketchup ...
        relish"). A new variant without a text of its own fails here."""
        from ingredient_mixing.config import Config
        from ingredient_mixing.task import instruction
        from ingredient_mixing.variants import pairs
        for preset, task, variant in pairs():
            if not variant:
                continue
            plain, text = instruction(Config(preset=preset, task=task)), instruction(Config(preset=preset, task=task, variant=variant))
            self.assertNotEqual(text, plain, (task, variant))
            for word in ("clockwise", "figure-eight", "waypoint"):
                self.assertNotIn(word, text.lower())
        honey = instruction(Config(preset="bake", task="sauce-blend", variant="honey"))
        self.assertIn("honey", honey)
        self.assertIn("mustard", honey)
        self.assertNotIn("ketchup", honey)
        self.assertNotIn("relish", honey)
        self.assertIn("spelt", instruction(Config(preset="bake", task="pancake-whisk", variant="spelt")))
        self.assertIn("One arm", instruction(Config(preset="bake", task="cake-mixer", robot="single-fr3-left")))

    def test_config_rejects_unknown_and_invalid(self):
        from ingredient_mixing.config import Config
        with self.assertRaises(ValueError):
            Config.from_dict({"preset": "bake", "colour": "red"})
        with self.assertRaises(ValueError):
            Config(preset="hob", task="muffin-fold")
        with self.assertRaises(ValueError):
            Config(seed=1.5)
        with self.assertRaises(ValueError):
            Config(preset="bake", task="sauce-blend", variant="spelt")           # spelt is a pancake variant


class Materials(unittest.TestCase):
    def test_batter_law_anchors(self):
        """The batter law at its two anchors: a pancake batter (flour fraction 0.33) of a few tenths of a Pa s^n with
        almost no yield stress, a muffin batter (0.45) more than an order of magnitude stiffer with one."""
        from ingredient_mixing.materials import batter_law
        from ingredient_mixing.task import FLOUR
        k_thin, n_thin, ty_thin = batter_law(FLOUR, 0.30)
        k_thick, n_thick, ty_thick = batter_law(FLOUR, 0.45)
        self.assertGreater(k_thick, 10.0*k_thin)
        self.assertLess(n_thick, n_thin)
        self.assertAlmostEqual(ty_thin, 0.0, places=6)
        self.assertGreater(ty_thick, 20.0)
        self.assertGreaterEqual(batter_law(FLOUR, 0.9)[1], FLOUR.n_min)          # the index is floored
        self.assertAlmostEqual(batter_law(FLOUR, 0.0)[0], 2.0e-3)                # no flour: the aqueous phase

    def test_rest_volume_rule(self):
        """V0 = V_solid + max(V_pore, V_liquid): dry flour keeps its pores (poured bulk density near 510 kg/m3); in a
        slurry the liquid has replaced them."""
        from ingredient_mixing import materials as M
        from ingredient_mixing.config import Config
        from ingredient_mixing.task import recipe_for
        r = recipe_for(Config(preset="bake", task="pancake-whisk"))
        s = r.solid
        dry, sat = M.rest_volume(r, {M.SOLID_D: 0.100})
        self.assertAlmostEqual(0.100/dry, s.rho/(1.0 + s.void), delta=1.0)
        self.assertEqual(sat, 0.0)
        wet, sat = M.rest_volume(r, {M.SOLID_D: 0.100, M.AQ: 0.300})
        self.assertAlmostEqual(wet, 0.100/s.rho + 0.300/r.rho_aq, delta=1e-7)
        self.assertEqual(sat, 1.0)

    def test_gelatinisation_is_arrhenius(self):
        from ingredient_mixing.materials import gel_rate, GAS_R
        from ingredient_mixing.task import STARCH
        self.assertLess(gel_rate(STARCH, 50.0), 1e-3)                            # nothing well below the onset
        self.assertGreater(gel_rate(STARCH, 90.0), 0.05)                         # seconds near a simmer
        r70, r80 = gel_rate(STARCH, 70.0), gel_rate(STARCH, 80.0)
        self.assertAlmostEqual(math.log(r80/r70), STARCH.gel_ea/GAS_R*(1/343.15 - 1/353.15), delta=1e-6)

    def test_sauce_blend_is_between_its_parts(self):
        """Nagasawa et al.: a blend's index and yield stress lie between those of its parts."""
        from ingredient_mixing.materials import blend
        y, k, n = blend([(0.5, 60.0, 30.0, 0.35), (0.5, 20.0, 12.0, 0.30)])
        self.assertTrue(20.0 < y < 60.0)
        self.assertTrue(0.30 < n < 0.35)
        self.assertTrue(12.0 < k < 30.0)


class Geometry(unittest.TestCase):
    def test_level_for_inverts_volume(self):
        from ingredient_mixing import geometry as GEO
        prof = GEO.cylinder_profile(0.08, 0.10)
        z = GEO.level_for(prof, 1.0e-3)
        self.assertAlmostEqual(GEO.volume_to(prof, z), 1.0e-3, delta=2e-6)
        self.assertAlmostEqual(z, 1.0e-3/(math.pi*0.08**2), delta=1e-3)


class Drawing(unittest.TestCase):
    def test_a_recorded_puff_of_vapour_is_drawn_as_shells_where_it_was(self):
        # recordings made before the puffs were drawn as shells hold each as a twenty-sided solid: a film of such a
        # recording shows the same puffs, at the same places and of the same sizes
        import numpy as np
        from ingredient_mixing import render as R
        iv, it = R._icosahedron()
        centre, size = np.array([[0.0, 0.0, 1.0], [0.1, -0.02, 1.2]]), np.array([0.02, 0.03])
        old = dict(v=(centre[:, None, :] + size[:, None, None]*iv[None]).reshape(-1, 3).astype(np.float32),
                   t=(it[None] + 12*np.arange(2)[:, None, None]).reshape(-1, 3))
        new = R.puffs_of_a_recording(old)
        shells = new["v"].reshape(2, len(R.SHELLS), -1, 3)
        np.testing.assert_allclose(shells.mean(2)[:, -1], centre, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(shells[:, -1] - centre[:, None, :], axis=2), size[:, None]*np.ones((1, shells.shape[2])), rtol=1e-5)
        self.assertEqual(len(new["t"]), 2*len(R.SHELLS)*320)
        self.assertIs(R.puffs_of_a_recording(new), new)                     # already shells: left as it is
        live = R.steam_mesh(np.array([0.05, 0.0, 283.0, 0.9], np.float32), np.array([0, 0, 0.9, 0, 0, 0, 1.0]), 0.09)
        self.assertEqual(len(live["v"]), 12*len(R.SHELLS)*162)              # 0.05 g/s: twelve puffs


    def test_two_of_mayonnaise_with_one_of_ketchup_is_drawn_pink_not_tomato_red(self):
        # what a cook sees (PASTE_SCATTER's comment has the two pages); with every paste scattering alike the blend
        # would be drawn sRGB (0.82, 0.27, 0.27)
        import numpy as np
        from ingredient_mixing import render as R, task as T
        recipe = dict(pastes=(T.MAYO, T.KETCHUP, T.RELISH))
        rgb, _ = R.paste_colors(recipe)
        blend = R.kubelka_munk(rgb, np.array([[2.0, 1.0, 0.0]]), R.paste_scatter(recipe))[0].astype(float)**(1/2.2)
        alike = R.kubelka_munk(rgb, np.array([[2.0, 1.0, 0.0]]))[0].astype(float)**(1/2.2)
        np.testing.assert_allclose(alike, [0.823, 0.269, 0.268], atol=0.005)
        np.testing.assert_allclose(blend, [0.920, 0.529, 0.516], atol=0.005)
        self.assertGreater(blend[0], blend[1] + 0.3)                          # pink: red well over green and blue
        self.assertGreater(blend[1], 0.45)                                    # and pale: not a saturated red
        # a paste alone keeps its own colour whatever its scattering (the rule keeps reflectances within 0.02 to 0.995)
        np.testing.assert_allclose(R.kubelka_munk(rgb, np.array([[0.0, 1.0, 0.0]]), R.paste_scatter(recipe))[0],
                                   np.clip(rgb[1], 0.02, 0.995), atol=1e-4)


class SolverConstants(unittest.TestCase):
    def test_free_surface_table(self):
        """Fraction one in the interior (no correction below the lattice's own scatter), falling toward the surface."""
        import numpy as np
        from ingredient_mixing import continuum as CT
        g, h = (np.array(a, float) for a in CT.FREE_SURFACE)
        self.assertEqual(h[0], 1.0)
        self.assertTrue(np.all(np.diff(g) > 0) and np.all(np.diff(h) < 0))
        self.assertGreater(h[-1], 0.5)                       # the top layer of a lattice still has half its kernel
        tab = np.interp(np.arange(64)*0.0125, g, h)
        self.assertEqual(tab[0], 1.0)
        self.assertEqual(tab[1], 1.0)                        # 0.0125 is inside the interior's scatter

    def test_hob_limits_documented(self):
        from ingredient_mixing import params as P
        self.assertEqual(P.HOB["cutout_c"], 238.0)           # the hob manual's overheat limit (460 F)
        self.assertEqual(P.HOB["levels_w"][0], 0.0)
        self.assertEqual(max(P.HOB["levels_w"]), 1800.0)


class Unevenness(unittest.TestCase):
    """The evaluator's measure of how unevenly an ingredient is spread (task.uniformity), on point sets written here."""

    def _bowl(self, n=60000, seed=0):
        import numpy as np
        from ingredient_mixing import task as T
        rng = np.random.default_rng(seed)
        r, a = 0.1*np.sqrt(rng.random(n)), rng.random(n)*2*np.pi
        local = np.stack([r*np.cos(a), r*np.sin(a), 0.03*rng.random(n)], 1)
        spec = np.zeros((n, T.NS))
        return np, T, rng, local, spec

    def test_portions_hold_equal_mass_and_every_point(self):
        np, T, rng, local, spec = self._bowl()
        mass = 1e-6*(0.5 + rng.random(len(local)))                 # points of unequal mass, as flour and milk are
        for n in (8, 12, 41):
            part = T.equal_portions(local, mass, n)
            self.assertEqual(sorted(set(part.tolist())), list(range(n)))
            m = np.bincount(part, weights=mass, minlength=n)
            self.assertLess(float(m.max()/m.min()) - 1.0, 0.01)             # to a few points in fifteen hundred

    def test_an_even_mix_reads_its_sampling_noise_and_a_heap_reads_far_over_the_limit(self):
        np, T, rng, local, spec = self._bowl()
        n = len(local)
        spec[:, T.SOLID_D] = 1e-6
        tracer = rng.random(n) < 0.04                              # every twenty-fifth point is leavening, anywhere
        spec[tracer, T.SOLID_D], spec[tracer, T.MINOR] = 0.0, 1e-6
        even = T.uniformity(local, spec, spec.sum(1), ("minor",), 12)["minor"]
        # 200 points of leavening to a portion: the count's own scatter, sqrt(0.96 / 200) = 0.069, and nothing more
        self.assertGreater(even["cv"], 0.04)
        self.assertLess(even["cv"], 0.09)
        spec[:, T.MINOR], spec[:, T.SOLID_D] = 0.0, 1e-6
        heap = np.hypot(local[:, 0] - 0.05, local[:, 1]) < 0.02    # all of it in one place
        spec[heap, T.MINOR], spec[heap, T.SOLID_D] = 0.5e-6, 0.5e-6
        res = T.uniformity(local, spec, spec.sum(1), ("minor", "solid"), 12)
        self.assertGreater(res["minor"]["cv"], 1.0)
        self.assertGreater(res["minor"]["cv"], res["solid"]["cv"])  # the bulk ingredient reads the less uneven of the two
        self.assertLessEqual(res["minor"]["lowest"], res["minor"]["cv"])
        self.assertGreaterEqual(res["minor"]["highest"], res["minor"]["cv"])

    def test_the_reading_does_not_jump_when_the_points_are_shaken(self):
        """A count over fixed cubes would jump as cells came into it and left it while the powder moved."""
        np, T, rng, local, spec = self._bowl()
        spec[:, T.SOLID_D] = 1e-6
        side = local[:, 0] > 0.03
        spec[side, T.MINOR], spec[side, T.SOLID_D] = 0.2e-6, 0.8e-6
        base = T.uniformity(local, spec, spec.sum(1), ("minor",), 12)["minor"]["cv"]
        for _ in range(4):
            shaken = local + rng.normal(0.0, 0.0003, local.shape)
            again = T.uniformity(shaken, spec, spec.sum(1), ("minor",), 12)["minor"]["cv"]
            self.assertLess(abs(again - base)/base, 0.02)

    def test_every_ingredient_a_task_puts_in_is_judged(self):
        from ingredient_mixing import task as T
        self.assertEqual(T.TASKS["sauce-blend"]["criteria"]["uniform"]["species"], ("paste_a", "paste_b", "paste_c"))
        self.assertEqual(T.TASKS["dry-whisk"]["criteria"]["uniform"]["species"], ("minor", "sugar"))
        for name in ("paste_a", "paste_b", "paste_c", "minor", "sugar", "solid"):
            self.assertIn(name, T.SPECIES_GROUP)


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "GPU scene build; set GAR_GPU_TESTS=1")
class Backend(unittest.TestCase):
    """The public path without rendering (cameras=()), through the registry."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-mix-test-")
        cls.env = registry.create("ingredient-mixing", task="sauce-blend", seed=0, cameras=(),
                                  output_dir=cls.tmp.name + "/s")
        cls.backend = cls.env._backend

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.tmp.cleanup()

    def _send(self, payload):
        """A policy request as the gateway carries it out (http_server.py action(), without its key store): decoded
        whole by the policy's access, which checks every command and only notes a refusal, then run through the
        session's facade, where the noted refusal is raised when the request's first command is about to run."""
        return self.env.act_policy(self.backend.policy_access().decode_actions(payload))

    def test_robot_configuration(self):
        rc = self.backend.robot_configuration()
        self.assertEqual(set(rc["instances"]), {"left", "right"})
        self.assertEqual(rc["setup_id"], "mix-fr3")
        self.assertTrue(rc["setup_fingerprint"])

    def test_geometry_routes(self):
        b = self.backend
        index = b.get_geometry_index()
        self.assertEqual(set(index["vessels"]), {"bowl", "crock"})
        self.assertIn("spoon", index["tools"])
        g = b.get_geometry("bowl")
        self.assertGreater(len(g["inner_profile"]), 4)
        self.assertGreater(g["capacity_m3"], 1e-3)                               # the 2 qt glass bowl
        self.assertIn("inner_profile", b.get_geometry_asset("profile/bowl"))
        with self.assertRaises(KeyError):
            b.get_geometry("no-such-item")

    def test_trace_contract_and_rejections(self):
        from gym_anything_robotics.actions import NamedJointTarget, Gripper
        from gym_anything_robotics.errors import ActionRejectedError
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        b = self.backend
        ref = b.begin_action_trace({"action_id": "t-1", "request_id": "r-1"})
        self.assertTrue(ref["available"])
        b.act(NamedJointTarget({"left/fr3_joint1": 0.57}, 20))
        b.end_action_trace()
        store = PhysicsTraceStore(Path(ref["store_path"]), readonly=True)
        try:
            page = store.page("t-1", limit=4)
            self.assertTrue(page["summary"]["complete"])
            self.assertEqual(page["summary"]["retained_samples"], 20)
            self.assertIn("mixture_wrench", page["items"][0]["data"])
        finally:
            store.close()
        with self.assertRaises(ActionRejectedError):
            b.act(Gripper("left", inward_effort_n=200.0, duration_s=0.1))        # beyond the hand's 70 N
        view = b.policy_access()
        self.assertFalse(any("bowl" in " ".join(k) for k in view.proprioception))   # no privileged object state

    def test_evaluation_is_of_the_material(self):
        ev = self.backend.evaluate()
        self.assertEqual(set(ev["criteria"]), {"uniform", "contained", "vessel_in_place"})
        u = ev["criteria"]["uniform"]                                            # mayonnaise, ketchup and relish are each read
        self.assertEqual(set(u["each"]), {"paste_a", "paste_b", "paste_c"})
        self.assertEqual(u["value"], max(u["each"].values()))                    # and the least evenly spread is judged
        self.assertEqual(u["each"][u["species"]], u["value"])
        self.assertFalse(ev["criteria"]["uniform"]["ok"])                        # nothing has been stirred
        self.assertFalse(ev["success"])
        self.assertTrue(ev["criteria"]["contained"]["ok"])
        self.assertLess(ev["measures"]["mass_balance_error"], 1e-6)
        from gym_anything_robotics.evaluation import normalize
        n = normalize(ev)                                                        # what `gar verify` returns
        self.assertTrue(n["available"])
        self.assertAlmostEqual(n["score"], 2/3, places=3)                        # two of three criteria met untouched

    def test_variation_selection(self):
        sel = self.backend.select_variation({"task": "dry-whisk", "seed": 3})
        self.assertEqual(sel["reset_request"], {"seed": 3, "options": {"preset": "bake", "task": "dry-whisk",
                                                                      "variant": "plain"}})
        with self.assertRaises(ValueError):
            self.backend.select_variation({"robot": "g1"})
        honey = self.backend.select_variation({"variant": "honey", "seed": 2})        # the variant is a selector
        self.assertEqual(honey["variation"], dict(preset="bake", task="sauce-blend", seed=2, variant="honey"))
        self.assertEqual(honey["reset_request"]["options"]["variant"], "honey")
        self.assertIn("honey", honey["task"])
        with self.assertRaises(ValueError):
            self.backend.select_variation({"task": "dry-whisk", "variant": "honey"})
        contract = self.backend.variation_contract()
        self.assertIn("variant", contract["selectors"])
        self.assertEqual(contract["current"], dict(preset="bake", task="sauce-blend", seed=0, variant="plain"))
        self.assertEqual(contract["variants"]["honey"]["tasks"], ["sauce-blend"])

    def test_state_holds_no_evaluation_and_evaluate_still_works(self):
        """The state the gateway records before and after every command does not run the evaluator (it reads every
        point of the mixture); the evaluation is there on request, for /v1/evaluation, `gar verify` and the
        benchmark."""
        st = self.backend.get_state()
        self.assertNotIn("evaluation", st)
        for key in ("tick", "time_s", "generation", "joints", "joint_torques_nm", "objects", "finite", "fault", "grip"):
            self.assertIn(key, st)
        self.assertNotIn("evaluation", self.env.get_state())                     # what GET /v1/state returns
        self.assertEqual(set(self.backend.evaluate()["criteria"]), {"uniform", "contained", "vessel_in_place"})
        self.assertTrue(self.env.evaluate()["available"])                        # what `gar verify` returns
        self.assertIn("evaluation", self.backend.env.get_state())                # the environment's own state has it
        self.assertIn("/v1/evaluation", self.backend.capabilities.notes["state"])

    def test_a_bake_session_has_no_thermometer_field(self):
        b = self.backend
        self.assertNotIn("thermometer", b.env.world.scene.tools)
        self.assertNotIn(("instruments", "thermometer_c"), b.policy_access().proprioception)
        self.assertNotIn("instruments", b.observation_from_state(b.get_state(), sequence=0).proprioception)
        self.assertIn("none in this task", b.capabilities.notes["instruments"])

    def test_policy_commands_are_floored_without_cameras(self):
        """A session without cameras has a floor: a 1 ms policy command is refused before anything moves and the
        refusal tells the policy the floor (policy_reason is the only backend text the gateway shows a policy); a
        command at the floor runs, and the session times it. The owner's commands are not floored."""
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.backend
        self.assertEqual(b.cameras, ())
        floor = b.min_command_s
        self.assertGreaterEqual(floor, 0.002)
        tick0 = b.env.world.tick
        with self.assertRaises(ActionRejectedError) as cm:
            b.act_policy(NamedJointTarget({}, 1))
        self.assertIn(f"must last >= {floor*1e3:.0f} ms", cm.exception.policy_reason)
        self.assertIn("min_command_s", cm.exception.policy_reason)
        self.assertEqual(b.env.world.tick, tick0)
        ticks = int(math.ceil(floor*1000 - 1e-9))
        r = b.act_policy(NamedJointTarget({}, ticks))
        b.observation_from_state(r.state, sequence=1)
        self.assertEqual(b.env.world.tick, tick0 + ticks)
        self.assertEqual(b.cost["commands"], 1)
        self.assertAlmostEqual(b.cost["sim_s"], ticks*1e-3, places=9)
        self.assertGreaterEqual(b.cost["fixed_s"], 0.0)
        self.assertLess(b.cost["wall_s"]/b.cost["sim_s"], 200.0)                 # this command, in process, in budget
        self.assertGreaterEqual(b.min_command_s, b.min_command_published)        # the floor never goes below its start
        self.assertEqual(b.policy_access().hardware["min_command_s"], b.min_command_s)
        self.assertIn("with or without cameras", b.capabilities.notes["min_command_s"])
        b.act(NamedJointTarget({}, 1))                                           # the owner's diagnostic stepping
        self.assertEqual(b.env.world.tick, tick0 + ticks + 1)

    def test_the_policy_reads_the_budget_and_each_commands_timing(self):
        """hardware.budget of the policy's capabilities and proprioception command_timing of its observations, on a
        live session (no cameras): declared, numbers, through the policy's own filter, zero before a command and after
        a reset, and the command's own after it."""
        b = self.backend
        b.reset()
        access = b.policy_access()
        for key in ("wall_s", "sim_s", "wall_per_sim", "physics_wall_per_sim", "over_budget"):
            self.assertIn(("command_timing", key), access.proprioception)
            self.assertIn(f"command_timing.{key}", access.capabilities()["proprioception_fields"])
        seen = access.observation(self.env.obs()).proprioception["command_timing"]
        self.assertEqual(seen, dict(wall_s=0.0, sim_s=0.0, wall_per_sim=0.0, physics_wall_per_sim=0.0, over_budget=0))
        ticks = max(30, int(math.ceil(b.min_command_s*1000 - 1e-9)))             # 20 ticks or more: the physics is timed
        timed = b._timed
        action = access.decode_actions({"type": "named_joint_target", "targets_si": {}, "ticks": ticks})
        r = b.act_policy(action)
        seen = access.observation(b.observation_from_state(r.state, sequence=1)).proprioception["command_timing"]
        self.assertAlmostEqual(seen["sim_s"], ticks*1e-3, places=9)
        self.assertGreater(seen["wall_s"], 0.0)
        self.assertAlmostEqual(seen["wall_per_sim"], seen["wall_s"]/seen["sim_s"], delta=0.02)
        self.assertGreater(seen["physics_wall_per_sim"], 0.0)
        self.assertLessEqual(seen["physics_wall_per_sim"], seen["wall_per_sim"] + 0.02)
        self.assertEqual(seen["over_budget"], int(seen["wall_s"] + 0.05 > 200.0*seen["sim_s"]))
        budget = b.policy_access().hardware["budget"]
        self.assertEqual(budget["limit_wall_per_sim"], 200.0)
        self.assertEqual(budget["commands_timed"], timed + 1)
        self.assertEqual(budget["min_command_s"], b.min_command_s)
        self.assertIsInstance(budget["physics_wall_per_sim"], float)             # measured now
        self.assertIn(budget["inside_budget"], (True, False))
        self.assertEqual(budget["inside_budget"], budget["at_the_floor_wall_per_sim"] <= 200.0)
        self.assertIn("not refused", budget["note"])
        self.assertIn("over_budget", access.hardware["command_timing"])
        notes = b.capabilities.notes                                             # the owner reads the same
        for key in ("budget", "command_timing", "render_worker", "renderer_options"):
            self.assertTrue(notes[key], key)
        self.assertIn("idle_release_s", notes["renderer_options"])
        self.assertIn("no policy camera", notes["render_worker"])
        state = b.get_state()                                                    # the worker's record is the owner's
        self.assertEqual((state["render_worker"]["held"], state["render_worker"]["pid"]), (False, None))
        self.assertFalse(any("render_worker" in " ".join(k) for k in access.proprioception))
        b.reset()                                                                # a reset's observation follows no command
        self.assertEqual(access.observation(self.env.obs()).proprioception["command_timing"]["wall_s"], 0.0)

    def test_a_batch_is_checked_whole_before_any_of_it_runs(self):
        """A policy batch whose later command is refused (too short, a joint beyond its limit, an unknown joint, a
        grip beyond the hand, a duration off the tick) is refused whole: every command is checked when the request is
        decoded, and the refusal is raised when the first command is about to run. Nothing has moved, and the gateway
        has no partly executed batch to lock the session for. The decoding notes the refusal and does not raise it
        (the gateway looks the request's Idempotency-Key up between the decoding and the first command:
        Discovery.test_a_keyed_command_...), so the requests here are sent the gateway's way (_send)."""
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.backend
        access = b.policy_access()
        floor = b.min_command_s
        n = int(math.ceil(floor*1000 - 1e-9))
        hold = lambda ticks: {"type": "named_joint_target", "targets_si": {}, "ticks": ticks}
        tick0 = b.env.world.tick
        with self.assertRaises(ActionRejectedError) as cm:
            self._send({"actions": [hold(n), hold(1)]})
        self.assertIn("command 2 of 2", cm.exception.policy_reason)
        self.assertIn("nothing in this request was executed", cm.exception.policy_reason)
        self.assertIn("must last", cm.exception.policy_reason)
        for bad in ({"type": "named_joint_target", "targets_si": {"left/fr3_joint1": 99.0}, "ticks": n},
                    {"type": "named_joint_target", "targets_si": {"bowl": 0.0}, "ticks": n},
                    {"type": "gripper", "arm": "left", "inward_effort_n": 200.0, "duration_s": max(floor, 0.1)},
                    {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, 0.01], "duration_s": 0.0015}):
            with self.assertRaises(ActionRejectedError) as cm:
                self._send({"actions": [hold(n), hold(n), bad]})
            self.assertIn("command 3 of 3", cm.exception.policy_reason)
        self.assertEqual(b.env.world.tick, tick0)                                # nothing ran
        self.assertIsNone(b._cycle)                                              # and nothing is being timed
        noted = access.decode_actions({"actions": [hold(n), hold(1)]})           # the decoding alone: noted, not raised
        self.assertEqual(len(noted), 2)
        self.assertIsInstance(b._cycle["refusal"], ActionRejectedError)
        self.assertEqual(b.env.world.tick, tick0)
        with self.assertRaises(ActionRejectedError):                             # raised at the first command, not the second
            b.act_policy(noted[0])
        self.assertEqual(b.env.world.tick, tick0)
        good = access.decode_actions({"actions": [hold(n), {"type": "cartesian_delta", "arm": "right",
                                                            "delta_m": [0, 0, 0.002], "duration_s": max(floor, 0.05)}]})
        self.assertEqual(len(good), 2)
        self.assertEqual(b.env.world.tick, tick0)                                # decoding moves nothing either
        b._cycle = None

    def test_set_load_is_a_policy_command(self):
        """The raw action set_load (libfranka's setLoad): a policy tells an arm's controller what its hand carries.
        The request is checked before anything changes; the arms hold for the floor in force; the configured load is
        in the policy's proprioception; nothing is configured at the start."""
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.backend
        access = b.policy_access()
        self.assertIn("raw", b.capabilities.action_types)
        self.assertIn("raw", access.capabilities()["action_types"])
        self.assertIn("set_load", access.hardware["load"])
        w = b.env.world
        status = lambda: b.observation_from_state(b.get_state(), sequence=0).proprioception["robot_status"]["arms"]
        self.assertEqual(status()["left"]["load_kg"], 0.0)                       # nothing is configured for the robot
        self.assertEqual(status()["left"]["load_com_m"], [0.0, 0.0, 0.0])
        raw = lambda **request: {"type": "raw", "request": {"op": "set_load", **request}}
        tick0, floor = w.tick, b.min_command_s
        for bad in (raw(arm="left", mass_kg=99.0), raw(arm="left", mass_kg=-0.1), raw(arm="left", mass_kg=True),
                    raw(arm="left", mass_kg=1.0, com_m=[0, 0]), raw(arm="left", mass_kg=1.0, com_m=[0, 0, 9.0]),
                    raw(arm="left"), raw(arm="left", mass_kg=1.0, colour="red"), raw(arm="left", mass_kg=1.0, duration_s=0.0015),
                    raw(arm="left", mass_kg=1.0, duration_s=0.001),             # shorter than the floor
                    {"type": "raw", "request": {"op": "place_free", "body": "bowl"}},
                    {"type": "raw", "request": {"op": "set_load", "arm": 3, "mass_kg": 1.0}}):
            with self.assertRaises(ActionRejectedError) as cm:
                self._send(bad)
            self.assertTrue(cm.exception.policy_reason)
        with self.assertRaises(ActionRejectedError) as cm:                       # a batch: refused whole
            self._send({"actions": [raw(arm="right", mass_kg=0.5), raw(arm="left", mass_kg=99.0)]})
        self.assertIn("command 2 of 2", cm.exception.policy_reason)
        self.assertEqual((w.tick, status()["left"]["load_kg"], status()["right"]["load_kg"]), (tick0, 0.0, 0.0))
        b._cycle = None
        action = access.decode_actions(raw(arm="left", mass_kg=1.0, com_m=[0, 0, 0.05]))
        r = b.act_policy(action)
        obs = b.observation_from_state(r.state, sequence=1)
        self.assertEqual(w.tick, tick0 + int(math.ceil(floor*1000 - 1e-9)))      # the arms hold for the floor in force
        self.assertEqual(r.info["load_set"], {"arm": "left", "mass_kg": 1.0, "com_m": [0.0, 0.0, 0.05]})
        arms = access.observation(obs).proprioception["robot_status"]["arms"]    # through the policy's own filter
        self.assertEqual((arms["left"]["load_kg"], arms["left"]["load_com_m"]), (1.0, [0.0, 0.0, 0.05]))
        self.assertEqual(arms["right"]["load_kg"], 0.0)
        self.assertEqual(w.load_cfg["left"]["mass"], 1.0)
        from gym_anything_robotics.actions import RawAction
        b.act(RawAction({"op": "set_load", "arm": "left", "mass_kg": 0.0}))      # the owner's: one tick
        self.assertEqual(w.tick, tick0 + int(math.ceil(floor*1000 - 1e-9)) + 1)
        self.assertEqual(status()["left"]["load_kg"], 0.0)

    def test_refused_reset_leaves_the_session_as_it_was(self):
        """Four resets the backend refuses, and a variant of another task: each is refused with
        ActionRejectedError through the session's own facade, the world is the same object at the same tick, and the
        next command runs. A reset to the same variation returns a state without the evaluation."""
        from gym_anything_robotics.actions import NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.backend
        world, generation, tick = b.env.world, b.env.generation, b.env.world.tick
        for seed, options in ((None, {"task": "no-such-task"}), (None, {"colour": "red"}), (123456, None),
                              (None, {"preset": "bake", "task": "stir-sauce"}),
                              (None, {"task": "sauce-blend", "variant": "spelt"})):
            with self.assertRaises(ActionRejectedError) as cm:
                self.env.reset(seed=seed, options=options)
            self.assertIn("nothing changed", str(cm.exception))
            self.assertIs(b.env.world, world)
            self.assertEqual((b.env.generation, b.env.world.tick), (generation, tick))
            self.assertEqual(b.config.task, "sauce-blend")
        b.act(NamedJointTarget({}, 2))
        self.assertEqual(b.env.world.tick, tick + 2)
        state, info = b.reset()
        self.assertEqual(info["continuity"], "reset")
        self.assertNotIn("evaluation", state)
        self.assertEqual(state["tick"], 0)

    def test_checkpoint_scoped_to_its_variation(self):
        b = self.backend
        b.save_checkpoint(label="here")
        self.assertTrue(all(c["restorable"] for c in b.list_checkpoints()))
        saved = b._checkpoints["here"]["variation"]
        b._checkpoints["here"]["variation"] = dict(saved, task="dry-whisk")      # as if saved before a variation change
        with self.assertRaises(ValueError):
            b.restore_checkpoint("here")
        b._checkpoints["here"]["variation"] = saved
        b.restore_checkpoint("here")

    def test_min_command_published(self):
        self.assertIn("min_command_s", self.backend.capabilities.notes)

    def test_start_state_is_at_rest_at_hand_over(self):
        sm = self.backend.env.settle_material
        self.assertTrue(sm["at_rest"])
        self.assertLess(sm["point_speed_p99_mm_s"], 1.0)

    def test_room_furniture_and_counter_things_collide(self):
        from ingredient_mixing import params as P
        names = {box[0] for box in self.backend.env.world.scene.static_boxes}
        for must in ("wall_front", "wall_left", "wall_right", "wall_ceiling", "fur_table_top", "fur_stool0", "fur_stool1",
                     "fur_shelving", "fur_basin", "fur_dispenser", "fur_bin", "fur_fridge", "fur_fridge_door1", "fit_station_camera"):
            self.assertIn(must, names)
        self.assertEqual(len(self.backend.env.world.scene.static_cyls), len(P.PLANT_POT))   # the plant's pot
        self.assertFalse(any(n.startswith("prop_") for n in names))              # what stands on the counter is not static:
        w = self.backend.env.world                                               # each is a free body with its pack's mass
        props = [n for n in w.scene.bodies if n.startswith("prop_")]
        self.assertEqual(len(props), len(P.PROPS["sauce-blend"]))
        mjm = w.solver.mj_model
        for pr in w.scene.prop_list:
            self.assertAlmostEqual(float(mjm.body_mass[w._newton_to_mj[pr["body"]]]), P.PROP[pr["kind"]]["mass"], places=5)
        self.assertFalse(any("mast" in n for n in names))                        # no camera mast on the cart

    def test_station_camera_looks_at_the_work_from_the_wall(self):
        sc = self.backend.env.world.scene
        cam = sc.cameras["station_rgb"]
        vx, vy = sc.layout["vessel_xy"]
        self.assertAlmostEqual(cam["eye"][0], vx, delta=0.01)
        self.assertGreater(cam["eye"][1], vy + 0.2)                              # behind the vessel, on the wall's side
        self.assertLess(cam["target"][1], vy)                                    # looking back at the robot over it
        self.assertEqual(self.backend.camera_specs["station_rgb"].mount_frame, "back_wall")
        self.assertNotIn("mast_rgb", sc.cameras)


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "scene build; set GAR_GPU_TESTS=1")
class HobSession(unittest.TestCase):
    """A hob session through the registry, the scene without the mixture (config material False: the rigid scene, the
    robot, the hob's knob and the tools; seconds to build)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-mix-hob-")
        cls.env = registry.create("ingredient-mixing", task="stir-sauce", seed=0, cameras=(), config={"material": False},
                                  output_dir=cls.tmp.name + "/s")
        cls.backend = cls.env._backend

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.tmp.cleanup()

    def test_the_thermometers_reading_reaches_the_policy(self):
        """The hob tasks' probe thermometer is an instrument: what it reads is the policy's to see (proprioception
        instruments.thermometer_c), through the policy's own filter, and stated in what the policy is told. In air it
        reads the room's 20 C. A scene without a thermometer has no such field."""
        from gym_anything_robotics.actions import NamedJointTarget
        b = self.backend
        self.assertIn("thermometer", b.env.world.scene.tools)
        access = b.policy_access()
        self.assertIn(("instruments", "thermometer_c"), access.proprioception)
        self.assertIn("probe thermometer", access.hardware["instruments"])
        self.assertIn("probe thermometer", b.capabilities.notes["instruments"])
        self.assertIn("instruments.thermometer_c", access.capabilities()["proprioception_fields"])
        seen = access.observation(self.env.obs()).proprioception
        self.assertAlmostEqual(seen["instruments"]["thermometer_c"], 20.0, delta=0.5)   # standing in the crock, in air
        r = b.act_policy(NamedJointTarget({}, int(math.ceil(b.min_command_s*1000 - 1e-9))))
        after = access.observation(b.observation_from_state(r.state, sequence=1)).proprioception
        self.assertAlmostEqual(after["instruments"]["thermometer_c"], 20.0, delta=0.5)
        self.assertEqual(b.env.world.scene.tools["thermometer"].state()["reading_c"], after["instruments"]["thermometer_c"])


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "scene build and Isaac render workers; set GAR_GPU_TESTS=1")
class RenderWorker(unittest.TestCase):
    """A session with one policy camera whose render worker is ended, driven through the gateway's own service object
    (what the HTTP routes call). The scene without the
    mixture (config material False) and small frames, so that the test is the worker's life and not the physics. The
    worker is ended by the PID the backend itself reports for it, after checking that it is this process's own child
    and a render worker of this environment; nothing else is signalled."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        from gym_anything_robotics.access import Principal
        from gym_anything_robotics.http_server import GARHTTPService
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-mix-worker-")
        cls.env = registry.create("ingredient-mixing", task="sauce-blend", seed=0, cameras=("station_rgb",),
                                  config={"material": False}, output_dir=cls.tmp.name + "/s",
                                  renderer_options={"width": 640, "height": 360, "samples": 4, "idle_release_s": 600})
        cls.backend = cls.env._backend
        cls.service = GARHTTPService(cls.env)
        cls.policy = Principal("policy", "policy")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.tmp.cleanup()

    def _end_worker(self):
        """SIGKILL to the session's own render worker; returns its PID once the backend's handle sees it gone."""
        import signal
        import time
        status = self.backend.worker_status()
        pid = status["pid"]
        self.assertTrue(status["held"] and status["alive"] and isinstance(pid, int), status)
        stat = Path(f"/proc/{pid}/stat").read_text()
        self.assertEqual(int(stat.rsplit(")", 1)[1].split()[1]), os.getpid())    # its parent is this process
        self.assertIn("ingredient_mixing.render_worker", Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode())
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 20.0
        while self.backend.worker_status()["alive"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.backend.worker_status()["alive"])
        return pid

    def _floor(self):
        return self.service.capabilities(self.policy)["capabilities"]["hardware"]["min_command_s"]

    def _hold(self, seconds):
        return {"type": "named_joint_target", "targets_si": {}, "ticks": max(1, int(math.ceil(seconds*1000 - 1e-9)))}

    def _image(self, published):
        frame, _ = self.service.camera("station_rgb", observation_id=published["observation_id"], principal=self.policy)
        data = frame.path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        return len(data)

    def test_a_dead_worker_is_replaced_for_the_policy_the_owner_and_recovery(self):
        from gym_anything_robotics.errors import ActionRejectedError
        b, service, policy = self.backend, self.service, self.policy
        first = service.observation(policy)["observation"]                       # the session holds a live worker
        self.assertGreater(self._image(first), 1000)
        self.assertEqual((b.options["width"], b.options["height"], b.options["samples"]), (640, 360, 4))
        self.assertIn("width 640, height 360, samples 4, idle_release_s 600", b.capabilities.notes["renderer_options"])
        floor_held = self._floor()
        events = dict(b.worker_events)
        # ---- 1. ended; a policy command follows at once
        pid0 = self._end_worker()
        self.assertTrue(service.health()["ready"])                               # nothing has failed yet
        floor_lost = self._floor()
        self.assertGreater(floor_lost, floor_held)                               # the floor covers the replacement's start
        if floor_held < floor_lost - 0.0005:
            with self.assertRaises(ActionRejectedError) as cm:                   # at the held floor: refused, told why
                service.action(self._hold(floor_held), idempotency_key=None, principal=policy)
            self.assertIn("no render worker is held", cm.exception.policy_reason)
            self.assertTrue(service.health()["ready"])
        reply = service.action(self._hold(floor_lost), idempotency_key=None, principal=policy)
        self.assertTrue(reply["committed"])
        self.assertIsNotNone(reply["observation"], reply.get("observation_error"))
        self.assertGreater(self._image(reply["observation"]), 1000)
        self.assertTrue(service.health()["ready"])
        status = b.worker_status()
        self.assertTrue(status["alive"])
        self.assertNotEqual(status["pid"], pid0)                                 # a new worker, started beside the physics
        self.assertEqual((status["starts"], status["lost"]), (events["starts"] + 1, events["lost"] + 1))
        self.assertAlmostEqual(reply["observation"]["proprioception"]["command_timing"]["sim_s"],
                               self._hold(floor_lost)["ticks"]*1e-3, places=9)
        self.assertEqual(self._floor(), b._floor_measured)                       # a live worker is held again: no start in it
        self.assertLess(self._floor(), floor_lost)
        # ---- 2. ended again; the policy asks for an observation
        pid1 = self._end_worker()
        seen = service.observation(policy)["observation"]
        self.assertGreater(self._image(seen), 1000)
        self.assertNotIn(b.worker_status()["pid"], (pid0, pid1))
        # ---- 3. ended again; the owner's recovery
        self._end_worker()
        out = service.recover({"acknowledge_uncertain_actions": True})
        self.assertTrue(out["recovered"])
        self.assertIsNotNone(out["observation"])
        self.assertEqual((out["health"]["ready"], out["health"]["runtime_ok"]), (True, True))
        self.assertTrue(b.worker_status()["alive"])
        self.assertTrue(service.action(self._hold(self._floor()), idempotency_key=None, principal=policy)["observation"])
        # ---- 4. ended again; the owner resets to the same task
        self._end_worker()
        out = service.reset({})
        self.assertTrue(out["committed"])
        self.assertIsNotNone(out["observation"])
        self.assertTrue(service.health()["ready"])
        # ---- 5. the process lives and the frame fails: the worker is ended and the frame asked again of a new one.
        # ASSISTED: the failure is injected at the session's end of the request (a real worker, alive and well, whose
        # observe call raises once), so that the process is certainly alive when the frame fails
        retries = b.worker_events["frame_retries"]
        live = b._retained
        pid4 = live.process.pid

        def broken(*args, **kwargs):
            raise BrokenPipeError(32, "Broken pipe")
        live.observe = broken
        reply = service.action(self._hold(self._floor()), idempotency_key=None, principal=policy)
        self.assertIsNotNone(reply["observation"], reply.get("observation_error"))
        self.assertGreater(self._image(reply["observation"]), 1000)
        self.assertEqual(b.worker_events["frame_retries"], retries + 1)
        self.assertTrue(live.closed)
        self.assertIsNotNone(live.process.poll())                                # the failed worker was ended and reaped
        self.assertNotEqual(b.worker_status()["pid"], pid4)
        self.assertTrue(service.health()["ready"])
        status = service.state()["state"]["render_worker"]                       # the owner's record of all this
        self.assertEqual((status["starts"], status["lost"]), (events["starts"] + 5, events["lost"] + 5))
        self.assertTrue(status["alive"])
        for t in list(b._reapers):
            t.join(timeout=30.0)
        for pid in (pid0, pid1, pid4):                                           # none of the ended workers is left
            self.assertFalse(Path(f"/proc/{pid}").exists() and
                             Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z", pid)

    def test_a_keyed_command_sent_again_after_its_worker_died_gets_its_stored_reply(self):
        """On a live session with a real render worker: a hold at the floor is sent with an Idempotency-Key and
        carried out; sent again at once it gets the stored reply; the worker is ended, the floor in force rises above
        the command's length; sent again it still gets the stored reply, not "rejected before execution", and nothing
        runs a second time. The same body as a new command is refused under the floor in force."""
        from gym_anything_robotics.errors import ActionRejectedError
        b, service, policy = self.backend, self.service, self.policy
        self.assertGreater(self._image(service.observation(policy)["observation"]), 1000)   # a live worker is held
        floor_held = self._floor()
        body = self._hold(floor_held)
        tick0 = b.env.world.tick
        first = service.action(body, idempotency_key="idem-1", principal=policy)
        self.assertTrue(first["committed"])
        self.assertIsNotNone(first["observation"], first.get("observation_error"))
        self.assertEqual(b.env.world.tick, tick0 + body["ticks"])
        self.assertEqual(service.action(body, idempotency_key="idem-1", principal=policy), first)
        self.assertEqual(b.env.world.tick, tick0 + body["ticks"])
        self._end_worker()
        floor_lost = self._floor()
        self.assertGreater(self._hold(floor_lost)["ticks"], body["ticks"])       # the command is too short now
        again = service.action(body, idempotency_key="idem-1", principal=policy)
        self.assertEqual(again, first)                                           # the stored reply, the same request id
        self.assertEqual((again["committed"], again["request_id"]), (True, first["request_id"]))
        self.assertEqual(b.env.world.tick, tick0 + body["ticks"])                # nothing ran a second time
        self.assertTrue(service.health()["ready"])
        with self.assertRaises(ActionRejectedError) as cm:                       # as a new command it is refused, told why
            service.action(body, idempotency_key="idem-2", principal=policy)
        self.assertIn("no render worker is held", cm.exception.policy_reason)
        self.assertEqual(b.env.world.tick, tick0 + body["ticks"])
        self.assertTrue(service.health()["ready"])
        reply = service.action(self._hold(self._floor()), idempotency_key="idem-3", principal=policy)
        self.assertIsNotNone(reply["observation"], reply.get("observation_error"))   # and the session goes on
        self.assertGreater(self._image(reply["observation"]), 1000)
        self.assertTrue(b.worker_status()["alive"])


if __name__ == "__main__":
    unittest.main()

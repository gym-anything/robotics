"""Drilling runtime contract: the policy command floor (min_command_s) and the render-worker lifecycle.

The floor tests are pure Python over drilling/runtime_cost.py, plus backend builds (bench/press-drill) that never
start a render worker. The lifecycle test replaces the Isaac worker with a fake one (no GPU): a live session must keep
its worker whatever the time between commands and whatever idle_release_s says; a paused session must give it back
idle_release_s after the pause (at once for 0, never for null); the resume must return with a worker that has drawn
its first frame; a frame error and an exited worker process must be survived; stop must release and reap it."""

from __future__ import annotations

import math
import tempfile
import threading
import time
import unittest

try:
    import newton  # noqa: F401
    HAVE_NEWTON = True
except Exception:  # pragma: no cover - the engine venv is required for the backend tests
    HAVE_NEWTON = False

STATION, OVERVIEW, WRIST = (960, 540), (960, 540), (640, 480)     # drilling/cameras.py, drilling/scene.py
NATIVE = {0: [], 1: [STATION], 2: [STATION, OVERVIEW], 3: [STATION, OVERVIEW, WRIST], 4: [STATION, OVERVIEW, WRIST, WRIST]}


class RuntimeCostTest(unittest.TestCase):
    def test_floor_for_zero_to_four_cameras(self) -> None:
        from drilling import runtime_cost as rc
        floors = [rc.min_command_s(NATIVE[n]) for n in range(5)]
        self.assertGreater(floors[0], 0.)                              # zero cameras: a floor all the same
        self.assertEqual(floors, sorted(floors))                       # never shorter with more cameras
        self.assertLess(floors[0], floors[1])
        self.assertLess(floors[1], floors[4])
        for n, floor in enumerate(floors):
            self.assertAlmostEqual(floor*1e3, round(floor*1e3), places=9)   # whole milliseconds (1 ms ticks)
            worst = rc.worst_command_s(NATIVE[n], duration_s=floor)
            self.assertLessEqual(worst/floor, rc.DESIGN_RATIO + 1e-9, n)
            self.assertLess(rc.DESIGN_RATIO, rc.LIMIT)
            shorter = floor - 1e-3
            if shorter > 0:                                            # the floor is the shortest command that fits
                self.assertGreater(rc.worst_command_s(NATIVE[n], duration_s=shorter)/shorter, rc.DESIGN_RATIO)
        self.assertEqual([round(f*1e3) for f in floors], [21, 28, 35, 40, 45])  # the published values (ms)
        self.assertEqual(rc.FLOORS_MS, {0: 21, 1: 28, 2: 35, 3: 40, 4: 45})


    def test_floor_follows_resolution_and_samples(self) -> None:
        from drilling import runtime_cost as rc
        one = rc.min_command_s([STATION])
        self.assertGreater(rc.min_command_s([(1920, 1080)]), one)
        self.assertEqual(round(rc.min_command_s([(1920, 1080)])*1e3), 46)
        self.assertEqual(round(rc.min_command_s([(1280, 720)]*4)*1e3), 68)
        self.assertEqual(round(rc.min_command_s([STATION], samples=32)*1e3), 33)
        self.assertGreater(rc.min_command_s([(1280, 720)]*4), rc.min_command_s(NATIVE[4]))
        self.assertGreater(rc.min_command_s([STATION], samples=32), one)
        self.assertLessEqual(rc.min_command_s([(320, 240)]), one)
        self.assertGreater(rc.frame_s(1920, 1080), rc.frame_s(960, 540))
        self.assertAlmostEqual(rc.frame_s(960, 540, samples=32) - rc.frame_s(960, 540),
                               rc.frame_s(960, 540) - rc.frame_s(960, 540, samples=0), places=12)
        with self.assertRaises(ValueError):
            rc.min_command_s([(0, 540)])

    def test_model_bounds_every_measured_command(self) -> None:
        """Each measured worst command (runtime_cost.MEASURED, from the calibration runs through gar run + HTTP) lies
        under the model at its duration, and moved to its floor (its non-physics time kept, its physics at
        PHYSICS_RATIO_MAX) it stays within the design ratio, 25 % under the 200x limit."""
        from drilling import runtime_cost as rc
        self.assertGreaterEqual(len(rc.MEASURED), 8)
        for m in rc.MEASURED:
            res, samples = [tuple(r) for r in m["resolutions"]], m.get("samples", 16)
            model = rc.worst_command_s(res, samples=samples, duration_s=m["duration_s"])
            self.assertGreaterEqual(model, m["worst_total_s"], m["case"])
            floor = rc.min_command_s(res, samples)
            at_floor = (m["worst_total_s"] - m["physics_s"] + rc.PHYSICS_RATIO_MAX*floor)/floor
            self.assertLessEqual(at_floor, rc.DESIGN_RATIO, m["case"])


class SessionPauseHookTest(unittest.TestCase):
    """The GAR session layer tells the backend about pause and resume (session_service -> RoboticsEnv.session_paused ->
    backend.session_paused), and not about stop (stop closes the environment, which releases its workers)."""

    def test_pause_resume_and_stop_reach_the_backend(self) -> None:
        from pathlib import Path
        from gym_anything_robotics import connect
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.session_service import SessionServer, SessionService
        from tests.test_http_server import HTTPFakeBackend
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        backend = HTTPFakeBackend(Path(tmp.name))
        calls = []
        backend.session_paused = calls.append
        env = RoboticsEnv(backend)
        server = SessionServer(("127.0.0.1", 0), SessionService(env, session_id="pause-hook"), api_token="owner")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            owner = connect(f"http://127.0.0.1:{server.server_port}", api_key="owner", timeout_s=5)
            owner.pause()
            self.assertEqual(calls, [True])
            owner.resume()
            self.assertEqual(calls, [True, False])
            owner.pause()
            owner.resume()
            self.assertEqual(calls, [True, False, True, False])
            owner.stop()
            thread.join(timeout=5)
            self.assertEqual(calls, [True, False, True, False])     # stop: no hint; close releases instead
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            env.close()
        self.assertTrue(backend.closed)


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class DrillingBackendRuntimeTest(unittest.TestCase):
    def build(self, **kw):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        b = NewtonDrillingBackend(robot="dual-fr3", preset="bench", task="press-drill", seed=0,
                                  output_dir=tmp.name + "/session", recording=False, **kw)
        self.addCleanup(b.close)
        return b

    def test_default_observation_is_station_and_overview(self) -> None:
        from drilling import runtime_cost as rc
        b = self.build()
        self.assertEqual(b.cameras, ("station_rgb", "overview_rgb"))   # the overview shows the kit and the bench parts
        self.assertEqual(b.min_command_s, rc.min_command_s([STATION, OVERVIEW], b.config.policy_samples))
        b.close()

    def test_floor_is_the_same_for_every_idle_release(self) -> None:
        """No setting of renderer_options idle_release_s can put a worker start inside a policy command (a live session
        keeps its worker), so the published floor is the warm floor for every setting, and it says so."""
        from drilling import runtime_cost as rc
        floors = {}
        for idle in (0, 5, 600, None):
            b = self.build(cameras=("station_rgb",), renderer_options={"idle_release_s": idle})
            floors[idle] = b.min_command_s
            self.assertEqual(b.idle_release_s, None if idle is None else float(idle))
            self.assertIn("every idle_release_s", b.capabilities.notes["min_command_s"])
            self.assertIn("no policy command waits for a worker", b.capabilities.notes["render_worker"])
            self.assertEqual(b.policy_access().hardware["min_command_s"], b.min_command_s)
            b.close()
        self.assertEqual(set(floors.values()), {rc.min_command_s([STATION])})
        for bad in (-1, float("nan"), "soon", True):
            with self.assertRaises(ValueError):
                self.build(cameras=("station_rgb",), renderer_options={"idle_release_s": bad})

    def test_zero_camera_floor_is_enforced(self) -> None:
        from drilling import runtime_cost as rc
        from gym_anything_robotics.actions import CartesianDelta, Gripper, Keyboard, NamedJointTarget
        from gym_anything_robotics.errors import ActionRejectedError
        b = self.build(cameras=())
        self.assertEqual(b.min_command_s, rc.min_command_s([]))
        self.assertGreater(b.min_command_s, 0.)
        floor_ticks = int(round(b.min_command_s/b.config.rigid_dt))
        self.assertGreater(floor_ticks, 1)
        tick = b.env.world.tick
        solves = []
        solve = b.ik.solve
        b.ik.solve = lambda *a, **k: (solves.append(1), solve(*a, **k))[1]

        def kinds(ticks):                                            # every command kind a policy has, same duration
            return (NamedJointTarget({}, ticks),
                    CartesianDelta(arm="right", delta_m=(0., 0., 0.002), duration_s=ticks*b.config.rigid_dt),
                    Keyboard(keys=("R",), ticks=ticks, rate_hz=1000.0, arm="right"),
                    Gripper(arm="right", half_width_m=0.02, duration_s=ticks*b.config.rigid_dt))
        for action in kinds(floor_ticks - 1):                        # one tick under the floor: refused
            with self.assertRaises(ActionRejectedError) as ctx:
                b.act_policy(action)
            self.assertIn(f">= {floor_ticks} ms", str(ctx.exception))
        self.assertEqual(b.env.world.tick, tick)                     # nothing ran
        b.act(NamedJointTarget({}, floor_ticks - 1))                 # the owner path is not floored
        self.assertEqual(b.env.world.tick, tick + floor_ticks - 1)
        solves.clear()
        for action in kinds(floor_ticks):                            # at the floor: accepted, each planned once
            step = b.act_policy(action)
            self.assertEqual(step.info["executed_ticks"], floor_ticks)
        self.assertEqual(len(solves), 2)                             # one IK solve per Cartesian / keyboard command
        step = b.act_policy(NamedJointTarget({}, floor_ticks))
        self.assertEqual(step.info["executed_ticks"], floor_ticks)
        self.assertTrue(b.capabilities.notes["min_command_s"].startswith(f"{b.min_command_s:g}:"))
        self.assertEqual(b.policy_access().hardware["min_command_s"], b.min_command_s)   # the policy's own copy
        self.assertIsNone(b._renderer)                                # no camera: no worker
        self.assertIsNone(b._prestart)

    def test_floor_uses_the_configured_resolution_and_the_worker_lifecycle(self) -> None:
        from drilling import runtime_cost as rc
        from gym_anything_robotics.actions import NamedJointTarget
        b = self.build(cameras=("station_rgb",), renderer_options={"width": 1920, "height": 1080})
        self.assertEqual((b.camera_specs["station_rgb"].width_px, b.camera_specs["station_rgb"].height_px), (1920, 1080))
        self.assertEqual(b.min_command_s, rc.min_command_s([(1920, 1080)], b.config.policy_samples))
        self.assertGreater(b.min_command_s, rc.min_command_s([STATION]))
        self.assertIn("1920x1080", b.capabilities.notes["min_command_s"])
        self.assertIn("render_worker", b.capabilities.notes)
        self.assertEqual(b.idle_release_s, 600.)

        created = []

        class Process:
            code = None

            def poll(self):
                return self.code

        class FakeWorker:
            def __init__(self):
                self.closed, self.fail, self.process, self.frames = False, False, Process(), 0
                self.slow_s, self.frame_done_at, self.closed_at = 0., None, None
                created.append(self)

            def observe(self, names, world):
                if self.fail:
                    raise RuntimeError("frame failed")
                time.sleep(self.slow_s)
                self.frames += 1
                self.frame_done_at = time.monotonic()
                return {n: b"\x89PNG fake" for n in names}

            def close(self):
                self.closed, self.closed_at = True, time.monotonic()

        b._new_renderer = FakeWorker
        b._scene = lambda: b.output/"fake_scene.usda"
        ticks = int(round(b.min_command_s/b.config.rigid_dt))
        self.assertTrue(b._watcher.is_alive())                       # a session with cameras looks at its worker
        b._watch_stop.set()                                          # here the look (_check_worker) is called by hand,
        b._watcher.join(5)                                           # so each step below is deterministic
        self.assertFalse(b._check_worker())                          # no worker yet: nothing to restart

        def command():
            step = b.act(NamedJointTarget({}, ticks))
            obs = b.observation_from_state(step.state, sequence=b.env.world.tick)
            self.assertEqual(set(obs.images), {"station_rgb"})

        def reaped():
            with b.env.lock:                                         # a release in flight (the idle timer's runs
                threads = list(b._reapers)                           # under this lock) has registered its reaper
            for t in threads:
                t.join(10)

        # kept between commands: one worker for three commands, and nothing waits to take it away
        for _ in range(3):
            command()
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].closed)
        self.assertIs(b._renderer, created[0])
        self.assertIsNone(b._idle_timer)                             # a live session has no idle release
        self.assertEqual(b.worker_events["cold_starts"], 1)          # the session's first image
        self.assertTrue(b.capabilities.notes["render_worker_state"].startswith("ready; session live"))

        # a live session keeps its worker whatever idle_release_s says and however long the policy thinks: with a
        # 0.2 s setting and with 0 the next command finds the same worker
        for idle in (0.2, 0.):
            b.idle_release_s = idle
            command()
            time.sleep(0.6)
            self.assertIs(b._renderer, created[0])
            self.assertFalse(created[0].closed)
            self.assertIsNone(b._idle_timer)
            command()
            self.assertEqual(len(created), 1)
        self.assertEqual(b.worker_events["cold_starts"], 1)
        self.assertEqual(b.worker_events["released"], [])
        b.idle_release_s = 600.

        # a short pause keeps the worker: resume returns at once and the next command uses it
        tick = b.env.world.tick
        self.assertFalse(b.session_paused(True))                     # not released now
        self.assertIsNotNone(b._idle_timer)                          # the paused session's idle release is armed
        self.assertIn("session paused", b.capabilities.notes["render_worker_state"])
        self.assertFalse(b.session_paused(False))                    # nothing to restart
        self.assertIsNone(b._idle_timer)
        self.assertEqual(b.env.world.tick, tick)
        command()
        self.assertEqual(len(created), 1)

        # a pause longer than idle_release_s gives the worker back (physics untouched) ...
        b.idle_release_s = 0.2
        tick = b.env.world.tick
        self.assertFalse(b.session_paused(True))
        deadline = time.monotonic() + 5.
        while b._renderer is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        reaped()
        self.assertIsNone(b._renderer)
        self.assertTrue(created[0].closed)
        self.assertEqual(b.worker_events["released"][-1], "paused_idle")
        self.assertEqual(b.env.world.tick, tick)
        self.assertTrue(b.capabilities.notes["render_worker_state"].startswith("released; session paused"))
        # ... and the resume itself restarts it and draws its first frame before it returns: the wait is the owner's
        created_before = len(created)
        self.assertTrue(b.session_paused(False))
        self.assertEqual(len(created), created_before + 1)
        self.assertIs(b._renderer, created[-1])                      # ready when resume returns, not "starting"
        self.assertIsNone(b._prestart)
        self.assertEqual(created[-1].frames, 1)                      # its slow first frame is already drawn
        self.assertEqual(b.worker_events["resume_restarts"], 1)
        self.assertIsNotNone(b.worker_events["last_resume_restart_s"])
        self.assertTrue(b.capabilities.notes["render_worker_state"].startswith("ready; session live"))
        cold = b.worker_events["cold_starts"]
        command()
        self.assertEqual(len(created), created_before + 1)
        self.assertEqual(created[-1].frames, 2)
        self.assertEqual(b.worker_events["cold_starts"], cold)       # no command waited for a worker
        self.assertFalse(created[-1].closed)

        # idle_release_s 0: the pause releases at once; null: the paused session keeps its worker
        b.idle_release_s = 0.
        self.assertTrue(b.session_paused(True))
        reaped()
        self.assertTrue(created[-1].closed)
        self.assertEqual(b.worker_events["released"][-1], "pause")
        self.assertTrue(b.session_paused(False))
        self.assertEqual(created[-1].frames, 1)
        self.assertEqual(b.worker_events["resume_restarts"], 2)
        b.idle_release_s = None
        self.assertFalse(b.session_paused(True))
        self.assertIsNone(b._idle_timer)
        time.sleep(0.3)
        self.assertFalse(created[-1].closed)
        self.assertFalse(b.session_paused(False))
        command()
        self.assertEqual(b.worker_events["cold_starts"], cold)

        # an owner's image on a paused session restarts the paused session's idle release
        b.idle_release_s = 600.
        b.session_paused(True)
        b.observation_from_state(b.get_state(), sequence=0)
        self.assertIsNotNone(b._idle_timer)
        b.session_paused(False)
        self.assertIsNone(b._idle_timer)

        # a resume whose worker cannot start raises to the session layer (which resumes all the same); the session
        # stays usable and the next command starts a worker itself (the fault path, counted as a cold start)
        b.idle_release_s = 0.
        b.session_paused(True)
        reaped()

        def broken():
            raise RuntimeError("no GPU")
        b._new_renderer = broken
        with self.assertRaises(RuntimeError):
            b.session_paused(False)
        self.assertFalse(b._paused)
        b._new_renderer = FakeWorker
        b.idle_release_s = 600.
        command()
        self.assertEqual(b.worker_events["cold_starts"], cold + 1)

        # a worker whose process exited is replaced
        n = len(created)
        created[-1].process.code = 0
        command()
        self.assertEqual(len(created), n + 1)
        self.assertTrue(b.last_timing.get("replaced_exited_worker"))

        # ... and the session does not wait for a command to find that out: its own look at the worker starts a new
        # one at once and has it draw its first frame, so the next command waits for nothing
        def warmed():
            deadline = time.monotonic() + 5.
            while time.monotonic() < deadline and not (b._prestart and b._prestart["warmed"]):
                time.sleep(0.02)
            self.assertTrue(b._prestart["warmed"])
        self.assertFalse(b._check_worker())                          # a healthy worker is left alone
        n, cold = len(created), b.worker_events["cold_starts"]
        dead = created[-1]
        dead.process.code = -9                                       # the worker process was killed
        self.assertTrue(b._check_worker())                           # noticed without any command
        warmed()
        reaped()
        self.assertEqual(len(created), n + 1)
        self.assertTrue(dead.closed)
        self.assertEqual(created[-1].frames, 1)                      # its slow first frame is already drawn
        self.assertEqual(b.worker_events["released"][-1], "worker_exited")
        self.assertIn("by the session itself after a worker fault 1", b.capabilities.notes["render_worker_state"])
        self.assertTrue(b.capabilities.notes["render_worker_state"].startswith("ready; session live"))
        self.assertFalse(b._check_worker())                          # one restart, not one per look
        command()
        self.assertEqual(len(created), n + 1)                        # the command used the restarted worker
        self.assertEqual(created[-1].frames, 2)
        self.assertEqual(b.worker_events["cold_starts"], cold)       # and waited for no start
        self.assertFalse(b.last_timing.get("replaced_exited_worker"))
        # a paused session is left alone: its resume restarts the worker (tested above)
        b.session_paused(True)
        created[-1].process.code = -9
        self.assertFalse(b._check_worker())
        self.assertTrue(b.session_paused(False))
        self.assertEqual(b.worker_events["fault_restarts"], 1)

        # a frame error releases the worker; the session stays usable
        created[-1].fail = True
        with self.assertRaises(RuntimeError):
            b.observation_from_state(b.get_state(), sequence=0)
        reaped()
        self.assertTrue(created[-1].closed)
        self.assertEqual(b.worker_events["released"][-1], "frame_error")
        n, cold = len(created), b.worker_events["cold_starts"]
        self.assertTrue(b._check_worker())                           # the session starts a new worker by itself
        warmed()
        self.assertEqual(len(created), n + 1)
        command()
        self.assertEqual(len(created), n + 1)
        self.assertEqual(b.worker_events["cold_starts"], cold)
        self.assertEqual(b.worker_events["fault_restarts"], 2)
        self.assertFalse(created[-1].closed)
        # a worker released on purpose is not brought back by the look
        self.assertTrue(b.release_render_worker())
        reaped()
        self.assertFalse(b._check_worker())
        command()                                                    # the next image starts one, as documented
        self.assertEqual(b.worker_events["cold_starts"], cold + 1)

        # a pause during a frame waits for the frame: the hook takes the world lock the frame holds
        b.idle_release_s = 0.
        created[-1].slow_s = 0.5
        t = threading.Thread(target=lambda: b.observation_from_state(b.get_state(), sequence=0))
        t.start()
        time.sleep(0.1)                                              # the frame is in flight
        self.assertTrue(b.session_paused(True))
        reaped()
        t.join(5)
        self.assertTrue(created[-1].closed)
        self.assertGreaterEqual(created[-1].closed_at, created[-1].frame_done_at)
        self.assertTrue(b.session_paused(False))
        b.idle_release_s = 600.
        command()

        # stop: close releases and reaps the worker before it returns
        last = created[-1]
        b.close()
        self.assertTrue(last.closed)
        self.assertIsNone(b._renderer)
        self.assertEqual(b.worker_events["released"][-1], "close")
        self.assertFalse([t for t in threading.enumerate() if t.name == "drilling-render-release" and t.is_alive()])
        self.assertTrue(math.isfinite(b.min_command_s))

if __name__ == "__main__":
    unittest.main()

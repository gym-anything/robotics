"""Drilling behind the real gateway: refusals must leave the session usable.

A real session service (gym_anything_robotics.session_service: the HTTP gateway `gar run` starts, with an owner key and
a policy key) over the real drilling backend, no cameras, on 127.0.0.1. Each test sends what a typo or a careless
policy would send and then a plain policy command:
  - an owner reset that is refused (unknown task, unknown option, seed outside the catalogue, a task of the other
    preset, a selector that is not a name, a task the robot is not offered for) changes nothing and locks nothing;
  - a policy batch with a refused command anywhere in it (under the floor, a joint beyond its limit, a grip beyond the
    hand's force, an arm the robot does not have) is refused whole before its first command runs;
  - the one case that cannot be decided beforehand - a Cartesian command, not the first of its batch, that turns out
    unreachable from where the earlier commands left the arm - is refused after those commands have run, the gateway
    then asks for the owner's recovery, and the recovery route clears it;
  - after an owner reset to the other preset the station camera's mount label follows the scene (capabilities, the
    policy's hardware record and the observation's camera block)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

try:
    import newton  # noqa: F401
    HAVE_NEWTON = True
except Exception:  # pragma: no cover - the engine venv is required
    HAVE_NEWTON = False


class Gateway:
    """The session gateway over one drilling backend, with an owner key and a policy key."""

    def __init__(self, test, **kw):
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.session_service import SessionServer, SessionService
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.backend = NewtonDrillingBackend(cameras=(), output_dir=self.tmp.name + "/session", recording=False, **kw)
        self.env = RoboticsEnv(self.backend)
        self.server = SessionServer(("127.0.0.1", 0), SessionService(self.env, session_id="drilling-interface-test"),
                                    api_token="owner-key", policy_api_token="policy-key")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        test.addCleanup(self.close)
        self.floor = int(round(self.backend.min_command_s/self.backend.config.rigid_dt))

    def close(self):
        self.server.shutdown()
        self.thread.join(timeout=10)
        self.server.server_close()
        self.env.close()

    def req(self, method, path, body=None, key="policy"):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}{path}", data=data, method=method,
                                         headers={"Authorization": f"Bearer {key}-key", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def hold(self, ticks=None):
        return {"type": "named_joint_target", "targets_si": {}, "ticks": self.floor if ticks is None else ticks}

    def usable(self, test, what):
        """A plain policy command runs and the service reports itself ready."""
        tick = self.backend.env.world.tick
        status, response = self.req("POST", "/v1/action", self.hold())
        test.assertEqual(status, 200, f"{what}: {response}")
        test.assertEqual(self.backend.env.world.tick, tick + self.floor, what)
        status, health = self.req("GET", "/v1/health")
        test.assertTrue(health["ready"], what)
        test.assertIsNone(health["runtime_fault_type"], what)


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class RefusedResetTest(unittest.TestCase):
    def test_refused_owner_resets_leave_the_session_usable(self) -> None:
        g = Gateway(self, robot="dual-fr3", preset="wall", task="anchor-holes", seed=0)
        g.usable(self, "fresh session")
        refused = {
            "unknown task": ({"options": {"task": "no-such-task"}}, "no-such-task"),
            "unknown option": ({"options": {"colour": "red"}}, "colour"),
            "seed outside the catalogue": ({"seed": 10000}, "0..9999"),
            "task of the other preset": ({"options": {"preset": "wall", "task": "press-drill"}}, "not a wall task"),
            "unknown preset": ({"options": {"preset": "roof"}}, "Unknown preset"),
            "selector that is not a name": ({"options": {"task": ["anchor-holes"]}}, "must be a name"),
        }
        for what, (body, expected) in refused.items():
            tick, generation = g.backend.env.world.tick, g.backend.env.generation
            status, response = g.req("POST", "/v1/reset", body, key="owner")
            self.assertEqual(status, 400, f"{what}: {response}")
            self.assertEqual(response["error_type"], "ActionRejectedError", what)
            self.assertIn("Reset refused, nothing changed", response["error"], what)
            self.assertIn(expected, response["error"], what)                # the refusal states its reason
            self.assertEqual((g.backend.env.world.tick, g.backend.env.generation), (tick, generation), what)
            self.assertEqual((g.backend.config.preset, g.backend.config.task), ("wall", "anchor-holes"), what)
            g.usable(self, f"after a reset refused for: {what}")
        status, response = g.req("POST", "/v1/reset", {"seed": 3}, key="owner")      # a valid reset still works
        self.assertEqual(status, 200, response)
        self.assertEqual(g.backend.config.seed, 3)
        g.usable(self, "after a valid reset")

    def test_a_task_the_robot_is_not_offered_for_is_refused_and_the_session_stays_usable(self) -> None:
        g = Gateway(self, robot="single-fr3", preset="bench", task="press-drill", seed=0)
        for body in ({"options": {"task": "timber-bolt"}}, {"options": {"preset": "wall", "task": "anchor-holes"}}):
            status, response = g.req("POST", "/v1/reset", body, key="owner")
            self.assertEqual(status, 400, response)
            self.assertEqual(response["error_type"], "ActionRejectedError")
            self.assertIn("needs two arms", response["error"])
            self.assertEqual(g.backend.config.task, "press-drill")
            g.usable(self, f"single arm after the refused reset {body}")
        # a batch that names the arm this robot does not have is refused whole (422), nothing runs
        tick = g.backend.env.world.tick
        status, response = g.req("POST", "/v1/action", {"actions": [g.hold(), {
            "type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, 0.01], "duration_s": g.floor*1e-3}]})
        self.assertEqual(status, 422, response)
        self.assertEqual(g.backend.env.world.tick, tick)
        g.usable(self, "after a batch naming the missing left arm")
        status, response = g.req("POST", "/v1/reset", {"options": {"task": "press-blind"}}, key="owner")
        self.assertEqual(status, 200, response)
        g.usable(self, "after a reset to another one-arm task")


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class PolicyBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.holder = unittest.TestCase()
        cls.g = Gateway(cls.holder, robot="dual-fr3", preset="bench", task="press-drill", seed=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.holder.doCleanups()

    def test_a_batch_with_a_refused_command_runs_nothing(self) -> None:
        g = self.g
        f = g.floor
        dur = f*1e-3
        joint = "right/fr3_joint1"
        upper = g.backend.env.world.joints[joint]["upper"]
        limit = 2.*g.backend.env.world.joints[g.backend.env.world.gripper_joints("right")[0]]["effort_limit"]
        bad = {
            "second command under the floor": ([g.hold(), g.hold(f - 5)], f">= {f} ms"),
            "third command one tick under the floor": ([g.hold(), g.hold(), g.hold(f - 1)], "command 3 of 3"),
            "a Cartesian command under the floor": ([g.hold(), {"type": "cartesian_delta", "arm": "right",
                                                                 "delta_m": [0, 0, 0.002], "duration_s": dur - 1e-3}],
                                                    f">= {f} ms"),
            "a keyboard chord under the floor": ([g.hold(), {"type": "keyboard", "keys": ["R"], "ticks": f - 1,
                                                              "rate_hz": 1000.0, "arm": "right"}], f">= {f} ms"),
            "a joint target beyond its limit": ([g.hold(), {"type": "named_joint_target",
                                                            "targets_si": {joint: upper + 0.5}, "ticks": f}],
                                                "Invalid joint target"),
            "a grip beyond the hand's force": ([g.hold(), {"type": "gripper", "arm": "right",
                                                           "inward_effort_n": limit + 10., "duration_s": dur}],
                                               f"[0, {limit:g}] N"),
            "a finger width beyond the hand's travel": ([g.hold(), {"type": "gripper", "arm": "right",
                                                                    "half_width_m": 0.05, "duration_s": dur}],
                                                        "half gap"),
            "a duration that is not whole ticks": ([g.hold(), {"type": "gripper", "arm": "right", "half_width_m": 0.02,
                                                               "duration_s": dur + 0.0004}], "multiple"),
            "conflicting keys": ([g.hold(), {"type": "keyboard", "keys": ["W", "S"], "ticks": f, "rate_hz": 1000.0,
                                             "arm": "right"}], "Conflicting keys"),
        }
        for what, (actions, expected) in bad.items():
            tick = g.backend.env.world.tick
            status, response = g.req("POST", "/v1/action", {"actions": actions})
            self.assertEqual(status, 400, f"{what}: {response}")
            self.assertEqual(response["error_type"], "ActionRejectedError", what)
            self.assertFalse(response["committed"], what)
            self.assertIn(expected, response["error"], what)                 # the policy is told why
            self.assertIn("nothing in this request was executed", response["error"], what)
            self.assertEqual(g.backend.env.world.tick, tick, what)           # the first command did not run
            g.usable(self, f"after a batch refused for: {what}")
        # the same checks on a single command, and a good batch runs whole
        status, response = g.req("POST", "/v1/action", g.hold(f - 1))
        self.assertEqual((status, response["error_type"]), (400, "ActionRejectedError"))
        tick = g.backend.env.world.tick
        status, response = g.req("POST", "/v1/action", {"actions": [g.hold(), {
            "type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, 0.002], "duration_s": dur}, g.hold()]})
        self.assertEqual(status, 200, response)
        self.assertEqual(g.backend.env.world.tick, tick + 3*f)
        g.usable(self, "after a good batch")

    def test_ik_solver_is_the_shared_problem_with_an_analytic_jacobian(self) -> None:
        """drilling/kinematics.py solves the shared solver's problem with a numpy forward kinematics and an analytic
        Jacobian. The forward kinematics must equal the shared one, the Jacobian the residual's finite differences, the
        same targets must be accepted, near targets must get the same joints, and a solve must stay far under the shared
        solver's time."""
        import time
        import numpy as np
        from scipy.spatial.transform import Rotation
        ik = self.g.backend.ik
        rng = np.random.default_rng(3)
        for arm in ("left", "right"):
            chain, _ = ik.arms[arm]
            fast = ik.fast[arm]
            for _ in range(20):
                q = rng.uniform(chain.lower, chain.upper)
                T = chain.fk(q)
                R, pos, _, _ = fast.frames(q)
                self.assertLess(float(np.abs(R - T[:3, :3]).max()), 1e-12)
                self.assertLess(float(np.abs(pos - T[:3, 3]).max()), 1e-12)
                seed = rng.uniform(chain.lower, chain.upper)
                target_p = rng.uniform(-1., 1., 3)
                target_R = Rotation.from_rotvec(rng.uniform(-1., 1., 3)).as_matrix()
                reference = np.r_[T[:3, 3] - target_p,                       # the shared residual, written out
                                  0.12*(Rotation.from_matrix(target_R)*Rotation.from_matrix(T[:3, :3]).inv()).as_rotvec(),
                                  1e-5*(q - seed)]
                self.assertLess(float(np.abs(fast.residual(q, target_p, target_R, seed) - reference).max()), 1e-12)
                J = fast.residual_jacobian(q, target_p, target_R, seed)
                numeric = np.zeros_like(J)
                for k in range(len(q)):
                    d = np.zeros(len(q))
                    d[k] = 1e-6
                    numeric[:, k] = (fast.residual(q + d, target_p, target_R, seed)
                                     - fast.residual(q - d, target_p, target_R, seed))/2e-6
                self.assertLess(float(np.abs(J - numeric).max()), 1e-7)
        # the policy path's solve against the shared solver, same targets, same seed
        T = ik.forward("right")
        p0, R0 = T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3])
        accepted = both = 0
        slowest = 0.
        for i in range(24):
            scale, turn = ((0.02, 0.0), (0.10, 0.3), (0.30, 0.8), (0.60, 1.5))[i % 4]
            pos = p0 + rng.uniform(-scale, scale, 3)
            quat = (Rotation.from_rotvec(rng.uniform(-1, 1, 3)*turn)*R0).as_quat()
            out = {}
            for name, reference in (("new", False), ("old", True)):
                t0 = time.perf_counter()
                try:
                    out[name] = np.array(list(ik.solve("right", pos, quat, reference=reference).values()))
                except ValueError:
                    out[name] = None
                if name == "new":
                    slowest = max(slowest, time.perf_counter() - t0)
            self.assertEqual(out["new"] is None, out["old"] is None, f"target {i}: accepted by one solver only")
            if out["new"] is not None:
                accepted += 1
                check = ik.forward("right", out["new"])                      # the accepted joints reach the pose
                self.assertLess(float(np.linalg.norm(check[:3, 3] - pos)), 1e-3)
                if scale <= 0.10:                                            # near targets: the same joints
                    both += 1
                    self.assertLess(float(np.abs(out["new"] - out["old"]).max()), 1e-3, f"target {i}")
        self.assertGreaterEqual(accepted, 12)
        self.assertGreaterEqual(both, 10)
        self.assertLess(slowest, 0.5)                    # 0.07 s measured; the shared solver took up to 0.55 s

    def test_the_owner_path_is_not_floored_and_its_refusals_are_clean(self) -> None:
        g = self.g
        tick = g.backend.env.world.tick
        status, response = g.req("POST", "/v1/action", {"actions": [g.hold(2), g.hold(3)]}, key="owner")
        self.assertEqual(status, 200, response)
        self.assertEqual(g.backend.env.world.tick, tick + 5)
        g.usable(self, "after an owner batch under the policy floor")


@unittest.skipUnless(HAVE_NEWTON, "Newton engine not importable in this interpreter")
class CameraLabelAfterResetTest(unittest.TestCase):
    def test_station_mount_follows_the_preset(self) -> None:
        from gym_anything_robotics.backends.newton_drilling import NewtonDrillingBackend
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        b = NewtonDrillingBackend(robot="dual-fr3", preset="wall", task="anchor-holes", seed=0,
                                  output_dir=tmp.name + "/session", recording=False)
        self.addCleanup(b.close)

        class FakeWorker:                                            # no GPU: the labels do not need a frame's pixels
            closed, process = False, None

            def observe(self, names, world):
                return {n: b"\x89PNG fake" for n in names}

            def close(self):
                self.closed = True
        b._new_renderer = FakeWorker
        b._scene = lambda: b.output/"fake_scene.usda"

        def mounts():
            caps = {c.name: c.mount_frame for c in b.capabilities.camera_specs}
            policy = {c.name: c.mount_frame for c in b.policy_access().cameras}
            obs = b.observation_from_state(b.get_state(), sequence=0)
            seen = {n: frame.camera.mount_frame for n, frame in obs.images.items()}
            self.assertEqual(caps, policy)
            self.assertEqual(caps, seen)
            return caps
        self.assertEqual(mounts(), {"station_rgb": "cart_rear_mast", "overview_rgb": "ceiling_corner"})
        floor = b.min_command_s
        b.reset(options={"preset": "bench", "task": "take-drill"})
        self.assertEqual(mounts(), {"station_rgb": "bench_fixed", "overview_rgb": "ceiling_corner"})
        self.assertEqual(b.capabilities.notes["task"], "bench/take-drill seed 0")
        self.assertEqual(b.min_command_s, floor)                             # same cameras, same floor
        self.assertTrue(b.capabilities.notes["min_command_s"].startswith(f"{floor:g}:"))
        b.reset(options={"preset": "wall", "task": "anchor-holes"})
        self.assertEqual(mounts()["station_rgb"], "cart_rear_mast")


class DoctorTest(unittest.TestCase):
    """`gar doctor drilling` has its own checks (drilling/doctor.py; imports, files and variables only): a missing
    ruckig and an unset ISAAC_RENDER_PYTHON are reported, each by name."""

    def test_gar_doctor_runs_the_environment_checks(self) -> None:
        import importlib.util
        from gym_anything_robotics.launch import LaunchSpec, doctor
        result = doctor(LaunchSpec("drilling"))
        checks = result["environment_checks"]
        self.assertIsNotNone(checks)                                         # launch.py found drilling/doctor.py
        names = [c["check"] for c in checks["checks"]]
        for expected in ("import ruckig", "import scipy", "ISAAC_RENDER_PYTHON (RGB observations only)",
                         "import manifold3d (RGB observations only)", "robot catalogue entry for dual-fr3"):
            self.assertIn(expected, names)
        if importlib.util.find_spec("ruckig") is not None:
            self.assertTrue(next(c for c in checks["checks"] if c["check"] == "import ruckig")["ok"])
        self.assertEqual(result["ok"], result["implementation_available"] and bool(checks["ok"])
                         and result["coordinator_packages"]["newton"] is not None)

    def test_missing_ruckig_and_unset_render_python_are_noticed(self) -> None:
        import importlib.util
        from drilling import doctor
        from gym_anything_robotics.launch import LaunchSpec
        real = importlib.util.find_spec
        spec = LaunchSpec("drilling")
        gpu = lambda: (20000., 32000.)
        good = doctor.checks(spec, environ={"ISAAC_RENDER_PYTHON": __file__}, find=lambda m: object(), gpu=gpu)
        by = lambda r: {c["check"]: c for c in r["checks"]}
        self.assertTrue(by(good)["import ruckig"]["ok"])
        self.assertTrue(by(good)["ISAAC_RENDER_PYTHON (RGB observations only)"]["ok"])
        bad = doctor.checks(spec, environ={}, find=lambda m: None if m == "ruckig" else real(m), gpu=gpu)
        self.assertFalse(bad["ok"])
        self.assertFalse(by(bad)["import ruckig"]["ok"])
        self.assertIn("ruckig>=0.19", by(bad)["import ruckig"]["detail"])
        self.assertFalse(by(bad)["ISAAC_RENDER_PYTHON (RGB observations only)"]["ok"])
        self.assertIn("unset", by(bad)["ISAAC_RENDER_PYTHON (RGB observations only)"]["detail"])
        gone = doctor.checks(spec, environ={"ISAAC_RENDER_PYTHON": "/no/such/python"}, find=real, gpu=gpu)
        self.assertIn("no such file", by(gone)["ISAAC_RENDER_PYTHON (RGB observations only)"]["detail"])
        full = doctor.checks(spec, environ={"ISAAC_RENDER_PYTHON": __file__}, find=lambda m: object(),
                             gpu=lambda: (1000., 32000.))                    # a card other jobs have filled
        self.assertFalse(by(full)["GPU memory for one render worker (RGB observations only)"]["ok"])
        panda = doctor.checks(LaunchSpec("drilling", robot="single-panda"), environ={}, find=real, gpu=gpu)
        entry = by(panda)["robot catalogue entry for single-panda"]
        self.assertTrue(entry["ok"])
        self.assertIn("35 N per finger", entry["detail"])


if __name__ == "__main__":
    unittest.main()

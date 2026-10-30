"""GAR adapter checks for gift-bags.

The first class needs no GPU, no renderer and no physics: the adapter runs against a recording stand-in of the
environment, so every check is about what the adapter writes, when, and what the gateway does with its errors. The
physics class builds the real home/lift-lid world (no cameras) and is skipped unless GIFT_BAGS_PHYSICS_TESTS=1.
"""

import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from gym_anything_robotics.actions import CartesianDelta, CartesianPose, Gripper, NamedJointTarget
from gym_anything_robotics.errors import ActionRejectedError, UnsupportedActionError
from tests.packages import PACKAGES

ROOT = PACKAGES
ARMS = ("left", "right")


class FakeRig:
    def __init__(self):
        self.joints = {}
        for a in ARMS:
            for i in range(1, 8):
                self.joints[f"{a}/fr3_joint{i}"] = dict(q=len(self.joints), dof=len(self.joints), lower=-2.8, upper=2.8,
                                                        velocity_limit=2.6)
            for i in (1, 2):
                self.joints[f"{a}/fr3_finger_joint{i}"] = dict(q=len(self.joints), dof=len(self.joints), lower=0., upper=.04)
        self.hand = {a: dict(mode="position", force=0., width=.04, effort=35.) for a in ARMS}
        self.targets = np.zeros(len(self.joints))
        self.reflexes = []
        self.b = SimpleNamespace(_gar_robots={})
        self.model = None


class FakeEnv:
    """Records every write the adapter makes (grip commands, joint targets + physics)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.world = SimpleNamespace(fault=None, steps=0, rig=FakeRig())
        self.generation = 0
        self.writes = []
        self.trace_callback = None

    def set_grip(self, arm, *, width=None, force=None):
        self.writes.append(("grip", arm, width, force))

    def act(self, targets, *, ticks, guard=None, schedule=None):
        self.writes.append(("act", dict(schedule[-1] if schedule else targets), ticks))
        self.world.steps += ticks
        return self.get_state()

    def reset(self):
        self.generation += 1
        return self.get_state()

    def close(self):
        pass

    def get_state(self, evaluate=True):
        return dict(tick=self.world.steps, generation=self.generation, time_s=self.world.steps*.004, ready=True,
                    fault=None, terminated=False, contact_solver={},
                    joints={n: dict(position=0., velocity=0., target=0.) for n in self.world.rig.joints},
                    evaluation=dict(success=False, score=0., checks={}, metrics={}))


class FakeIK:
    arms = {a: (None, [f"{a}/fr3_joint{i}" for i in range(1, 8)]) for a in ARMS}

    def forward(self, arm, q=None):
        T = np.eye(4)
        T[:3, 3] = (.3, 0., .9)
        return T

    def solve(self, arm, position, orientation=None):
        if np.linalg.norm(np.asarray(position, float)) > 1.5:
            raise ValueError("IK unreachable (900.00 mm, 0.00 deg); no action executed")
        if position[2] < 0:
            return {f"{arm}/fr3_joint4": -3.5}               # a solution outside the rig's joint limits
        return {f"{arm}/fr3_joint1": .1}

    def cartesian_path(self, arm, start_q, poses):
        (position, _), = poses
        end = {n: 0. for n in self.arms[arm][1]}
        end.update(self.solve(arm, position))
        return [(end, 0)]


class FakeRenderer:
    started = 0

    def __init__(self, fail_observe=0):
        FakeRenderer.started += 1
        self.closed = False
        self.fail_observe = fail_observe
        self.process = SimpleNamespace(poll=lambda: None)
        self.frames = 0

    def observe(self, cameras=("room",), state=None):
        if self.fail_observe:
            self.fail_observe -= 1
            raise ConnectionError("Renderer connection closed")
        self.frames += 1
        return {c: b"\x89PNG fake" for c in cameras}

    def close(self):
        self.closed = True


def fake_backend(output, cameras=(), options=None, renderer=FakeRenderer):
    from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
    from gift_bags.config import Config
    from gym_anything_robotics.types import CameraSpec, Capabilities
    b = object.__new__(NewtonGiftBagsBackend)
    b.config = Config(preset="home", task="lift-lid")
    b.cameras, b.options, b.output = tuple(cameras), dict(options or {}), Path(output)
    b.env, b.ik, b.arms = FakeEnv(), FakeIK(), ARMS
    b.images, b.last_timing, b.trace = __import__("collections").deque(), {}, None
    b._renderer, b._prestart, b._idle_timer, b._reapers, b._checkpoints = None, None, None, [], {}
    b.camera_specs = {c: CameraSpec(c, "mount", 64, 36, 64, 36, (.036, .02), (.02, .02)) for c in cameras}
    b.kinds = frozenset({"named_joint_target", "gripper", "cartesian_pose", "cartesian_delta"})
    b.capabilities = Capabilities("gift-bags", "newton", "dual-fr3", b.kinds, frozenset({"proprioception"}), True,
                                  False, True, b.cameras, tuple(b.camera_specs.values()), notes={})
    b._scene = lambda: Path(output)/"scene.usdc"
    b._new_renderer = lambda scene=None: renderer()
    b._start_renderer_early = lambda: None                   # the stand-in physics is instantaneous
    return b


BAD = {
    "duration not a multiple of 4 ms": CartesianDelta(arm="right", delta_m=(0., 0., .001), duration_s=.01),
    "unreachable pose": CartesianPose(arm="right", position_m=(3., 3., 3.), duration_s=.2),
    "IK solution outside the joint limits": CartesianPose(arm="right", position_m=(.3, 0., -.1), duration_s=.2),
    "grip force 200 N (> 70 N)": Gripper(arm="right", inward_effort_n=200., duration_s=.2),
    "grip duration 0.01 s": Gripper(arm="right", half_width_m=.02, duration_s=.01),
    "half gap 50 mm (> 40 mm)": Gripper(arm="right", half_width_m=.05, duration_s=.2),
    "reachable pose with a 50 mm half gap": CartesianDelta(arm="right", delta_m=(0., 0., -.01), duration_s=.2,
                                                           gripper_half_target_m=.05),
    "unknown joint": NamedJointTarget({"right/elbow": .1}, 10),
    "joint target beyond its limit": NamedJointTarget({"right/fr3_joint1": 3.}, 10),
}


class GiftBagsAdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        FakeRenderer.started = 0

    def tearDown(self):
        self.tmp.cleanup()

    # ---------------------------------------------------------------- refused commands write nothing
    def test_refused_commands_raise_action_rejected_and_write_nothing(self):
        b = fake_backend(self.tmp.name)
        for label, action in BAD.items():
            with self.subTest(label):
                with self.assertRaises(ActionRejectedError):
                    b.act(action)
                self.assertEqual(b.env.writes, [])
        b.arms = ("left",)                                     # single-fr3: there is no right arm
        with self.assertRaises(UnsupportedActionError):
            b.act(CartesianDelta(arm="right", delta_m=(0., 0., .001), duration_s=.2))
        self.assertEqual(b.env.writes, [])

    def test_legal_commands_write_grip_then_targets(self):
        b = fake_backend(self.tmp.name)
        b.act(CartesianDelta(arm="right", delta_m=(0., 0., -.01), duration_s=.2, gripper_half_target_m=.02))
        b.act(Gripper(arm="left", inward_effort_n=30., duration_s=.02))
        b.act(NamedJointTarget({"left/fr3_finger_joint1": .01}, 3))
        goal = {f"right/fr3_joint{i}": 0. for i in range(1, 8)} | {"right/fr3_joint1": .1}    # the trajectory's end
        self.assertEqual(b.env.writes, [("grip", "right", .02, None), ("act", goal, 50),
                                        ("grip", "left", None, 15.), ("act", {}, 5),
                                        ("act", {"left/fr3_finger_joint1": .01}, 3)])

    def test_gateway_keeps_the_session_usable_after_refused_policy_commands(self):
        """An HTTP sequence on the gateway service: refused command, health, next legal command."""
        from gym_anything_robotics.access import Principal
        from gym_anything_robotics.env import RoboticsEnv
        from gym_anything_robotics.http_server import GARHTTPService
        b = fake_backend(self.tmp.name)
        service = GARHTTPService(RoboticsEnv(b))
        policy = Principal("policy", "policy")
        ok = {"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, -.01], "duration_s": .2}
        bad = [{"type": "cartesian_delta", "arm": "right", "delta_m": [0, 0, .001], "duration_s": .01},
               {"type": "cartesian_pose", "arm": "right", "position_m": [3., 3., 3.], "duration_s": .2},
               {"type": "gripper", "arm": "right", "inward_effort_n": 200., "duration_s": .2}]
        for k, command in enumerate(bad):
            with self.subTest(command=command):
                with self.assertRaises(ActionRejectedError):
                    service.action(command, idempotency_key=f"bad-{k}", principal=policy)
                health = service.health()
                self.assertTrue(health["runtime_ok"], health)
                result = service.action(ok, idempotency_key=f"ok-{k}", principal=policy)
                self.assertTrue(result["committed"])
        self.assertEqual([w[0] for w in b.env.writes], ["act"]*3)

    # ---------------------------------------------------------------- one render worker per session
    def test_one_worker_serves_consecutive_operations(self):
        b = fake_backend(self.tmp.name, cameras=("overhead_rgb",))
        for _ in range(3):
            b.act(CartesianDelta(arm="right", delta_m=(0., 0., -.001), duration_s=.2))    # the stand-in moves joint1 0.1 rad
            b.observation_from_state(b.env.get_state(), sequence=1)
        b.render(label="front")
        self.assertEqual(FakeRenderer.started, 1)
        self.assertEqual(b._renderer.frames, 4)
        r = b._renderer
        b.close()
        self.assertTrue(r.closed)
        self.assertIsNone(b._renderer)

    def test_idle_release_and_reopen(self):
        b = fake_backend(self.tmp.name, cameras=("overhead_rgb",), options=dict(idle_release_s=.05))
        b.observation_from_state(b.env.get_state(), sequence=0)
        r = b._renderer
        time.sleep(.3)
        self.assertTrue(r.closed)
        self.assertIsNone(b._renderer)
        b.observation_from_state(b.env.get_state(), sequence=1)
        self.assertEqual(FakeRenderer.started, 2)
        b.close()
        b = fake_backend(self.tmp.name, cameras=("overhead_rgb",), options=dict(idle_release_s=0))
        b.observation_from_state(b.env.get_state(), sequence=0)
        self.assertIsNone(b._renderer)                        # 0: released after every operation

    def test_crashed_worker_is_replaced_without_failing_the_observation(self):
        b = fake_backend(self.tmp.name, cameras=("overhead_rgb",))
        b.observation_from_state(b.env.get_state(), sequence=0)
        first = b._renderer
        first.process = SimpleNamespace(poll=lambda: -9)       # the worker process died between operations
        b.observation_from_state(b.env.get_state(), sequence=1)
        self.assertTrue(first.closed)
        self.assertIsNot(b._renderer, first)
        self.assertTrue(b.last_timing.get("replaced_exited_worker"))
        b._renderer.fail_observe = 1                            # the worker dies during a frame
        frames = b._render(("overhead_rgb",))
        self.assertTrue(frames["overhead_rgb"].is_file())
        self.assertEqual(FakeRenderer.started, 3)
        self.assertEqual(len(b.last_timing["render_failures"]), 1)
        b.close()

    def test_two_failed_frames_are_reported(self):
        b = fake_backend(self.tmp.name, cameras=("overhead_rgb",), renderer=lambda: FakeRenderer(fail_observe=1))
        with self.assertRaises(ConnectionError):
            b._render(("overhead_rgb",))
        self.assertIsNone(b._renderer)
        b.close()

    # ---------------------------------------------------------------- variations
    def test_variation_contract_is_the_catalogue_and_selection_is_launch_time(self):
        from gift_bags.task_set_catalog import contract
        b = fake_backend(self.tmp.name)
        c = b.variation_contract()
        for k, v in contract().items():
            self.assertEqual(c[k], v)
        self.assertEqual(c["current"], dict(preset="home", task="lift-lid", seed=0))
        self.assertFalse(c["in_session_selection"])
        same = b.select_variation(dict(task="lift-lid"))
        self.assertTrue(same["applies_in_this_session"])
        b.reset(**same["reset_request"])                       # this session's own variation: a plain reset
        other = b.select_variation(dict(preset="counter", task="open-bag", seed=3))
        self.assertFalse(other["applies_in_this_session"])
        self.assertEqual(other["task_id"], "gift-bags/v1/counter/open-bag/seed-3")
        generation = b.env.generation
        with self.assertRaises(ActionRejectedError):
            b.reset(**other["reset_request"])
        self.assertEqual(b.env.generation, generation)
        for bad in (dict(robot="g1"), dict(preset="dev"), dict(task="no-such-task"), dict(seed=-1)):
            with self.assertRaises(ValueError):
                b.select_variation(bad)

    # ---------------------------------------------------------------- doctor: CPU quota, solver threads, imports
    def test_doctor_reads_the_cgroup_quota_and_the_thread_default(self):
        from gift_bags import doctor
        with tempfile.TemporaryDirectory() as cg:
            (Path(cg)/"cpu.max").write_text("1600000 100000\n")
            self.assertEqual(doctor.cpu_quota(cg)[0], 16.)
            (Path(cg)/"cpu.max").write_text("max 100000\n")
            self.assertEqual(doctor.cpu_quota(cg)[1], "CPU affinity mask (no cgroup quota)")
        src = (ROOT/"gift_bags"/"ipcworld.py").read_text()
        self.assertIn(f'os.environ.get("GIFT_BAGS_THREADS", "{doctor.default_threads()}")', src)
        quota = (16., "test quota")
        self.assertTrue(doctor.thread_check({}, quota)[0])
        self.assertTrue(doctor.thread_check({"GIFT_BAGS_THREADS": "16"}, quota)[0])
        self.assertFalse(doctor.thread_check({"GIFT_BAGS_THREADS": "17"}, quota)[0])
        self.assertFalse(doctor.thread_check({"GIFT_BAGS_THREADS": "0"}, quota)[0])
        self.assertFalse(doctor.thread_check({"GIFT_BAGS_THREADS": "eight"}, quota)[0])
        names = [c["check"] for c in doctor.checks(None)["checks"]]
        for need in ("import threadpoolctl", "import trimesh", "import pxr", "solver threads within the CPU quota",
                     "render worker modules beside gift_bags (RGB observations only)"):
            self.assertIn(need, names)


@unittest.skipUnless(os.environ.get("GIFT_BAGS_PHYSICS_TESTS") == "1", "set GIFT_BAGS_PHYSICS_TESTS=1 (minutes, GPU)")
class GiftBagsAdapterPhysicsTest(unittest.TestCase):
    def test_rejections_leave_the_robot_untouched_and_the_trace_records_every_step(self):
        import copy
        from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend, TRACE_SHEET_STRIDE
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        with tempfile.TemporaryDirectory() as tmp:
            b = NewtonGiftBagsBackend(preset="home", task="lift-lid", cameras=(), output_dir=Path(tmp)/"s")
            try:
                rig = b.env.world.rig
                before = (rig.targets.copy(), copy.deepcopy(rig.hand), b.env.world.steps)
                for label, action in BAD.items():
                    if label == "IK solution outside the joint limits":
                        continue                               # constructed for the stand-in IK
                    with self.subTest(label):
                        info = b.begin_action_trace(dict(action_id=f"bad-{label}", request_id="r"))
                        with self.assertRaises(ActionRejectedError):
                            b.act(action)
                        self.assertEqual(b.end_action_trace()["status"], "rejected")
                        self.assertTrue(np.array_equal(rig.targets, before[0]))
                        self.assertEqual(rig.hand, before[1])
                        self.assertEqual(b.env.world.steps, before[2])
                        self.assertTrue(info["available"])
                n = 2*TRACE_SHEET_STRIDE + 3
                b.begin_action_trace(dict(action_id="ok", request_id="r"))
                b.act(CartesianDelta(arm="right", delta_m=(0., 0., -.002), duration_s=n*b.config.dt))
                summary = b.end_action_trace()
                self.assertEqual((summary["status"], summary["retained_samples"], summary["complete"]),
                                 ("completed", n, True))
                store = PhysicsTraceStore(Path(summary["store_path"]), readonly=True)
                items, after = [], 0
                while True:
                    page = store.page("ok", after_seq=after, limit=128)
                    items += page["items"]
                    if not page["has_more"]:
                        break
                    after = page["next_after_seq"]
                store.close()
                self.assertEqual([i["tick"] for i in items], list(range(before[2] + 1, before[2] + n + 1)))
                with_sheets = [k for k, i in enumerate(items) if "sheets" in i["data"]]
                self.assertEqual(with_sheets, [0, TRACE_SHEET_STRIDE, 2*TRACE_SHEET_STRIDE, n - 1])
                d = items[-1]["data"]
                self.assertEqual(len(d["joints"]["position"]), len(rig.joints))
                self.assertGreaterEqual(d["contact_solver"]["newton_iterations"], 1)
                self.assertIn(d["contact_solver"]["converged"], (True, False))
                self.assertEqual(len(d["items"]), len(b.get_geometry_index()["items"]))
                self.assertAlmostEqual(d["time_s"], (before[2] + n)*b.config.dt)
            finally:
                b.close()

    def test_contact_guard_stops_a_descent_and_free_air_is_not_stopped(self):
        from gym_anything_robotics.actions import MotionSafety
        from gym_anything_robotics.backends.newton_gift_bags import NewtonGiftBagsBackend
        with tempfile.TemporaryDirectory() as tmp:
            b = NewtonGiftBagsBackend(preset="home", task="lift-lid", cameras=(), output_dir=Path(tmp)/"s",
                                      recording=False)
            try:
                free = b.act(CartesianDelta(arm="right", delta_m=(0., 0., .005), duration_s=.1)).info
                self.assertFalse(free["stopped_on_contact"])
                self.assertEqual(free["executed_ticks"], free["requested_ticks"])
                down = b.act(CartesianDelta(arm="right", delta_m=(0., 0., -.35), duration_s=7.,
                                            safety=MotionSafety(stop_on_contact=True))).info
                self.assertEqual(down["velocity_reflexes"], [])          # slow enough not to trip the reflex
                self.assertTrue(down["stopped_on_contact"], down)
                self.assertLess(down["executed_ticks"], down["requested_ticks"])
                self.assertIn("contact", down["stop_reason"])
            finally:
                b.close()


if __name__ == "__main__":
    unittest.main()

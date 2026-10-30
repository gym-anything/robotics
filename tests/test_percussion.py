"""Percussion: discovery, catalogue, robot validation and evaluator units without physics; the backend's public path
(registry, actions, audio observation, trace, checkpoints, refusals) with a scene build when GAR_GPU_TESTS=1
(CPU physics; cameras=() so no render worker).

python -m unittest tests.test_percussion
GAR_GPU_TESTS=1 python -m unittest tests.test_percussion
"""
import os
import unittest
from pathlib import Path


class Discovery(unittest.TestCase):
    def test_describe_and_robots(self):
        from gym_anything_robotics.launch import describe
        d = describe("percussion")
        self.assertEqual(d["default_robot"], "dual-fr3")
        self.assertEqual({r["id"] for r in d["robots"]}, {"dual-fr3", "single-fr3", "dual-panda", "g1"})
        self.assertEqual(describe("tap-percussion")["environment_id"], "percussion")

    def test_robot_validator(self):
        from robots.catalog import select_for_backend, resolve
        self.assertEqual(select_for_backend("percussion", "dual-fr3"), "dual-fr3")
        self.assertEqual(select_for_backend("percussion", resolve("percussion-panda").to_dict()), "dual-panda")
        self.assertEqual(select_for_backend("percussion", "g1"), "g1")
        self.assertEqual(select_for_backend("percussion", resolve("percussion-g1").to_dict()), "g1")
        # barcode_scanning's G1 rig is the same robot, profile and floor mount as percussion-g1: accepted as g1
        self.assertEqual(select_for_backend("percussion", resolve("scan-g1").to_dict()), "g1")
        with self.assertRaises(ValueError):
            select_for_backend("percussion", resolve("whiteboard-g1").to_dict())     # another environment's profile
        with self.assertRaises(ValueError) as ctx:
            select_for_backend("percussion", "single-panda")                          # unknown alias: the refusal
        self.assertIn("dual-fr3", str(ctx.exception))                                 # lists the qualified aliases
        with self.assertRaises(ValueError):
            select_for_backend("percussion", resolve("pour-fr3-single").to_dict())    # left mount: drum out of reach

    def test_catalogue(self):
        from gym_anything_robotics import tasks
        from percussion.config import TASKS
        from percussion.variants import task_resolve, resolve
        c = tasks.catalog("percussion")
        self.assertEqual(c["count"], sum(len(v) for v in TASKS.values()))
        self.assertEqual(sorted(c["selector_choices"]["task"]), sorted(t for v in TASKS.values() for t in v))
        r = task_resolve({"task": "steady-beat", "seed": 3})
        self.assertEqual(r["selection"]["preset"], "gait")
        self.assertIn("beats per minute", r["description"])
        with self.assertRaises(ValueError):
            resolve("therapy", "steady-beat", 0)

    def test_config_rejects_unknown(self):
        from percussion.config import Config
        with self.assertRaises(ValueError):
            Config.from_dict({"preset": "therapy", "task": "melody", "no_such": 1})
        with self.assertRaises(ValueError):
            Config(preset="gait", task="melody")

    def test_instructions_for_every_task(self):
        from percussion.config import Config, TASKS
        from percussion.task import instruction
        for p, ts in TASKS.items():
            for t in ts:
                self.assertGreater(len(instruction(Config(preset=p, task=t, seed=5))), 40)

    def test_echo_stimulus_onsets(self):
        from percussion.task import echo_stimulus
        sig, on = echo_stimulus(dict(rhythm=[1, 0.5, 0.5, 2], bpm=90))
        self.assertEqual(len(on), 4)
        self.assertAlmostEqual(on[1], 60/90, places=6)
        self.assertGreater(abs(sig).max(), 1.0)                       # 94 dB SPL peak at 1 m


class CommandMinimum(unittest.TestCase):
    """The shortest policy command is fixed for the session by its cameras and does not follow the host's measured
    speed: on a host with much slower physics a measured minimum would be 0.33-1 s, past the 0.35-0.75 s beat of the
    steady-beat tasks. A command's measured cost is recorded (info.timing). No scene is built."""

    @staticmethod
    def _backend(cameras=("station_rgb",)):
        import threading
        from types import SimpleNamespace
        from gym_anything_robotics.backends import newton_percussion as NP
        from gym_anything_robotics.types import CameraSpec
        b = object.__new__(NP.NewtonPercussionBackend)
        b.cameras, b.g1, b.arms, b.kinds = tuple(cameras), False, ("left", "right"), frozenset({"named_joint_target"})
        b.min_command_s = NP.min_command_s(len(cameras), 1280, 720, 16)
        b.config = SimpleNamespace(rigid_dt=0.001, robot="dual-fr3", preset="gait", task="steady-beat", seed=0)
        b.env = SimpleNamespace(lock=threading.RLock(), joints={})
        b.camera_specs = {c: CameraSpec(c, "mast", 1280, 720, 1280, 720, (.036, .02), (.026, .026)) for c in cameras}
        b.capabilities = SimpleNamespace(notes={"gripper": "g", "keyboard": "k", "audio": "a"})
        b.last_timing, b._cmd, b._cmd_t0 = {}, None, None
        b._plan = lambda ticks: ({}, ticks, {})                                    # the action is its tick count here
        b.act = lambda ticks: SimpleNamespace(state={}, info={}, terminated=False, truncated=False)
        return b

    def _slow_command(self, b, ticks, physics_ratio):
        import time
        b.act_policy(ticks)
        sim_s = ticks*b.config.rigid_dt
        b.last_timing = {"physics_s": physics_ratio*sim_s}
        b._cmd["t0"] = time.perf_counter() - (physics_ratio + 50.0)*sim_s           # its whole wall time
        b._account_command()

    def test_minimum_does_not_follow_measured_speed(self):
        b = self._backend()
        self.assertEqual(b.min_command_now(), 0.01)                                # the station camera's 10 ms
        for ratio in (60.0, 340.0, 1000.0):                                         # slow-host contact regimes: >= 190x
            self._slow_command(b, 10, ratio)
            self.assertEqual(b.min_command_now(), 0.01)
            self.assertEqual(b.last_timing["physics_ratio"], ratio)
            self.assertEqual(b.last_timing["over_budget"], ratio + 50.0 > 200.0)
            self.assertGreaterEqual(b.last_timing["wall_per_sim"], ratio + 50.0)
        self.assertEqual(self._backend(("station_rgb", "left_wrist_rgb", "right_wrist_rgb")).min_command_now(), 0.01)

    def test_shorter_command_is_refused_with_the_minimum(self):
        from gym_anything_robotics.errors import ActionRejectedError
        b = self._backend()
        with self.assertRaises(ActionRejectedError) as cm:
            b.act_policy(9)
        self.assertIn("minimum is 10 ms (10 ticks", cm.exception.policy_reason)
        b.act_policy(10)                                                            # the minimum itself is accepted

    def test_help_states_the_fixed_minimum(self):
        hw = self._backend().policy_access().capabilities()["hardware"]
        self.assertEqual(hw["min_command_s"], 0.01)
        self.assertIn("0.01 s of simulated time", hw["min_command_s_note"])
        self.assertIn("10 ticks", hw["min_command_s_note"])
        self.assertIn("fixed for this session", hw["min_command_s_note"])
        self.assertNotIn("re-read it", hw["min_command_s_note"])


@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "scene build; set GAR_GPU_TESTS=1")
class Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-percussion-test-")
        cls.env = registry.create("percussion", preset="assessment", task="single-tone", seed=0, cameras=(),
                                  output_dir=cls.tmp.name + "/s")
        cls.backend = cls.env._backend

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.tmp.cleanup()

    def test_robot_configuration(self):
        rc = self.backend.robot_configuration()
        self.assertEqual(set(rc["instances"]), {"left", "right"})
        self.assertEqual(rc["setup_id"], "percussion-fr3")

    def test_audio_observation_and_policy_view(self):
        from gym_anything_robotics.actions import NamedJointTarget
        b = self.backend
        step = b.act(NamedJointTarget({"right/fr3_joint1": 0.01}, 20))
        obs = b.observation_from_state(step.state, sequence=1)
        a = obs.proprioception["audio"]["station_mic"]
        self.assertEqual(len(a["pressure_pa_16k"]), 8000)
        view = b.policy_access().observation(obs)
        self.assertIn("audio", view.proprioception)
        self.assertNotIn("items", view.proprioception)
        self.assertFalse(any("client" in " ".join(k) for k in b.policy_access().proprioception))

    def test_trace_and_rejections(self):
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
            self.assertIn("vibration_energy_j", page["items"][0]["data"])
        finally:
            store.close()
        with self.assertRaises(ActionRejectedError):
            b.act(Gripper("left", inward_effort_n=200.0, duration_s=0.1))
        with self.assertRaises(ActionRejectedError):
            b.act(NamedJointTarget({"left/fr3_joint4": 0.5}, 10))     # beyond the joint's limit

    def test_geometry(self):
        idx = self.backend.get_geometry_index()
        self.assertIn("rbar_C4", idx["surfaces"])
        self.assertEqual(idx["surfaces"]["drumhead"]["kind"], "head")
        with self.assertRaises(KeyError):
            self.backend.get_geometry("no-such")

    def test_checkpoint_scoped_to_its_variation(self):
        b = self.backend
        b.save_checkpoint(label="here")
        saved = b._checkpoints["here"]["variation"]
        b._checkpoints["here"]["variation"] = dict(saved, task="echo")
        with self.assertRaises(ValueError):
            b.restore_checkpoint("here")
        b._checkpoints["here"]["variation"] = saved
        b.restore_checkpoint("here")

    def test_evaluation_shape(self):
        ev = self.backend.evaluate()
        self.assertIn("one_stroke", ev["checks"])
        self.assertFalse(ev["success"])



@unittest.skipUnless(os.environ.get("GAR_GPU_TESTS") == "1", "scene build; set GAR_GPU_TESTS=1")
class G1Backend(unittest.TestCase):
    """The G1 configuration's public path: it stands after the settle, takes joint and Cartesian commands, refuses the
    Franka-only gripper and keyboard commands, reports its IMU tilt and has no reflexes."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from gym_anything_robotics import registry
        cls.tmp = tempfile.TemporaryDirectory(prefix="gar-percussion-g1-test-")
        cls.env = registry.create("percussion", robot="g1", preset="gait", task="steady-beat", seed=0, cameras=(),
                                  output_dir=cls.tmp.name + "/s")
        cls.backend = cls.env._backend

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()
        cls.tmp.cleanup()

    def test_stands_and_moves(self):
        from gym_anything_robotics.actions import NamedJointTarget, CartesianDelta
        b = self.backend
        self.assertEqual(b.robot_configuration()["setup_id"], "percussion-g1")
        self.assertLess(b.env.settle_report["pelvis_tilt_deg"], 2.0)
        b.act(NamedJointTarget({"g1/right_elbow_joint": 0.9}, 300))
        b.act(CartesianDelta(arm="right", delta_m=(0.0, 0.03, 0.03), duration_s=0.2))
        st = b._robot_status()
        self.assertLess(st["imu_tilt_deg"], 3.0)
        self.assertEqual(st["refused_targets"], 0)
        self.assertEqual(b.env.world.reflexes, [])

    def test_refusals(self):
        from gym_anything_robotics.actions import Gripper
        from gym_anything_robotics.errors import ActionRejectedError, UnsupportedActionError
        b = self.backend
        tick = b.env.world.tick
        with self.assertRaises((ActionRejectedError, UnsupportedActionError)):
            b.act(Gripper(arm="right", half_width_m=0.02, duration_s=0.1))
        self.assertEqual(b.env.world.tick, tick)
        self.assertEqual(set(b.capabilities.action_types), {"named_joint_target", "cartesian_pose", "cartesian_delta"})

    def test_variation_reset(self):
        """A G1 session reset to another variation rebuilds the world with the G1 IK."""
        from gym_anything_robotics.actions import CartesianDelta
        from percussion.g1 import G1Kinematics
        b = self.backend
        state, info = b.reset(seed=1, options={"preset": "gait", "task": "accent-ostinato"})
        self.assertEqual(info["variation"]["task"], "accent-ostinato")
        self.assertIsInstance(b.ik, G1Kinematics)
        self.assertIs(b.ik.world, b.env.world)
        b.act(CartesianDelta(arm="right", delta_m=(0.0, 0.0, 0.02), duration_s=0.2))
        b.reset(seed=0, options={"preset": "gait", "task": "steady-beat"})
        self.assertEqual(b.config.task, "steady-beat")


if __name__ == "__main__":
    unittest.main()

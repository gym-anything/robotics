"""Stub the environment and robot catalogue; neither is present/started by these tests."""
import copy
import json
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from whiteboard.ink import InkSurface
from whiteboard.tasks import Task
from whiteboard.history import Trace
from whiteboard.task_set.catalogue import CatalogueTask, CONTRACT, SELECTIONS
from whiteboard.task_set.presets import tool_poses
from whiteboard.task_set.mechanical import read_state
from gym_anything_robotics.backends.newton_whiteboard_task_set import NewtonWhiteboardTaskSetBackend
from tests.packages import PACKAGES

ROOT = PACKAGES


class Array:
    def __init__(self, data): self.data = np.asarray(data).copy()
    def numpy(self): return self.data
    def assign(self, data): self.data = np.asarray(data).copy()


class FakeEnv:
    """The small data/lifecycle surface used by the adapter; no dynamic stepping."""
    builds = 0
    def __init__(self, config, *, task=None):
        FakeEnv.builds += 1
        self.config, self.task = config, task or Task()
        self.tick = self.action_id = 0
        self.healthy = True
        self._lock = threading.RLock()
        self.closed = False
        self.trace = Trace()
        self._previous_contacts, self.deformations = {}, {}
        self._pad_contact_state = None
        self._needs_sync = True
        poses = tool_poses(config, CatalogueTask("write-text"))
        self.body_ids = {name:i for i,name in enumerate(poses)}
        count = len(poses)
        q = np.array(list(poses.values()))
        self.state_0 = SimpleNamespace(joint_q=Array(q.ravel()), joint_qd=Array(np.zeros(count*6)),
                                       body_q=Array(np.vstack((q, [0, 0, .8, 0, 0, 0, 1]))),
                                       body_qd=Array(np.zeros((count+1,6))))
        self.builder = SimpleNamespace(joint_child=list(range(count)), joint_q_start=list(range(0,count*7,7)),
                                       joint_qd_start=list(range(0,count*6,6)))
        self.model = SimpleNamespace(body_label=list(poses)+["g1/pelvis"], shape_body=Array([count]))
        self.robot_shapes = [0]
        native = SimpleNamespace(geom_bodyid=np.arange(count+2))
        self.solver = SimpleNamespace(mj_model=native, mj_data=SimpleNamespace(contact=[], ncon=0),
                                     mjc_body_to_newton=Array([[-1, *range(count+1)]]))
        def eval_fk(model, joint_q, joint_qd, state):
            state.body_q.data[:count] = joint_q.numpy().reshape(count,7)
            state.body_qd.data[:count] = joint_qd.numpy().reshape(count,6)
        self.newton = SimpleNamespace(eval_fk=eval_fk)
        self.ink = InkSurface(config.board, config.texture_width, config.texture_height, config.materials)
        self.robot_joints = {}
        self._initial = self.snapshot()

    def snapshot(self):
        return dict(ink=copy.deepcopy(self.ink), tick=self.tick, action_id=self.action_id,
                    arrays={k:getattr(self.state_0,k).numpy().copy() for k in vars(self.state_0)})

    def restore(self, snapshot):
        self.ink, self.tick, self.action_id = copy.deepcopy(snapshot["ink"]), snapshot["tick"], snapshot["action_id"]
        for k,v in snapshot["arrays"].items(): getattr(self.state_0,k).assign(v)
        return self.get_state()

    def reset(self): return self.restore(self._initial)
    def close(self): self.closed = True
    def get_state(self): return {"tick":self.tick,"time_s":self.tick*self.config.physics_dt,"joints":{}}
    def evaluate(self): return self.task.evaluate(self.ink)


def fake_env_module():
    module = ModuleType("whiteboard.env")
    module.WhiteboardEnv = FakeEnv
    return module


def robot_modules():
    robots, catalog = ModuleType("robots"), ModuleType("robots.catalog")
    robots.catalog = catalog
    catalog.default_robot = lambda name: "g1"
    catalog.select_for_backend = lambda name, robot: robot
    catalog.environment_robots = lambda name: ["g1"]
    catalog.ENVIRONMENTS = {"whiteboard": {}}
    return {"robots": robots, "robots.catalog": catalog}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.module_patch = patch.dict(sys.modules, {"whiteboard.env": fake_env_module(), **robot_modules()})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def backend(self, name="erase-board", **kwargs):
        return NewtonWhiteboardTaskSetBackend(catalogue_task=name, task_contract=CONTRACT, recording=False,
                    cameras=(), output_dir=self.directory.name, texture_width=1024, texture_height=512, **kwargs)

    def test_constructor_reset_and_seed_reset_keep_authored_starts(self):
        for name in ("erase-board", "erase-except-boxed", "recap-and-tidy", "correct-misspelling"):
            backend = self.backend(name)
            before = backend.env.snapshot()
            self.assertEqual(backend.evaluate()["score"], 0)
            self.assertIn("task_source_sha256", backend.evaluate())
            backend.env.ink.tiles.clear()
            backend.env.tick = 7
            backend.reset()
            self.assertEqual(backend.evaluate()["score"], 0)
            self.assertEqual(backend.env.tick, 0)
            np.testing.assert_array_equal(backend.env.state_0.body_q.numpy(), before["arrays"]["body_q"])
            old = backend.env
            backend.reset(seed=12)
            self.assertTrue(old.closed)
            self.assertEqual(backend.config.seed, 12)
            self.assertEqual(backend.evaluate()["score"], 0)
            backend.close()

    def test_invalid_options_fail_before_constructing_or_mutating(self):
        before = FakeEnv.builds
        for extra in ({"task": Task()}, {"variation_id": 0}, {"robot": "bad"}):
            with self.assertRaises((ValueError, RuntimeError)):
                self.backend(**extra)
        self.assertEqual(FakeEnv.builds, before)
        backend = self.backend()
        old = backend.env
        for kwargs in ({"seed":True}, {"options":{"unknown":1}}, {"seed":1,"options":{"variation_id":0}}):
            with self.assertRaises(ValueError): backend.reset(**kwargs)
            self.assertIs(backend.env, old)

    def test_native_variation_reset_uses_native_task_and_evaluation(self):
        backend = self.backend()
        backend.reset(options={"variation_id":0})
        self.assertIsInstance(backend.env.task, Task)
        self.assertEqual(backend.env.task.kind, "bridge")
        self.assertEqual(backend.evaluate(), backend.env.evaluate())

    def test_policy_gets_reference_for_every_new_task(self):
        for name in (s["task"] for s in SELECTIONS[:-1]):
            backend = self.backend(name)
            access = backend.policy_access()
            self.assertEqual(access.instruction(), backend.env.task.description)
            self.assertEqual(access.action_types, frozenset({"named_joint_position"}))
            data = access.task_images()["reference"]
            self.assertIn("front", data[0])
            self.assertTrue(data[1].startswith(b"\x89PNG"))
            backend.close()

    def test_teacher_reward_is_delta_and_pose_changes_invalidate_nothing(self):
        backend = self.backend()
        def erase(action, role):
            backend.env.ink.tiles.clear()
            backend.env.tick += 1
            return {"ticks_executed":1}
        backend._checked = erase
        step = backend.act(object())
        self.assertEqual(step.reward, 1.)
        self.assertEqual(backend.act(object()).reward, 0.)
        backend.env.state_0.body_q.data[backend.env.body_ids["eraser"],2] += .05
        self.assertFalse(backend.evaluate()["success"])

    def test_contact_mapping_uses_native_body_map(self):
        backend = self.backend()
        env = backend.env
        native_eraser = env.body_ids["eraser"]+1
        native_pelvis = len(env.body_ids)+1
        env.solver.mj_data.contact = [SimpleNamespace(geom=[native_eraser,native_pelvis], efc_address=0, dist=-.001)]
        env.solver.mj_data.ncon = 1
        self.assertEqual(read_state(env)["touching"], {"eraser"})
        env.solver.mj_data.contact[0].efc_address = -1
        self.assertEqual(read_state(env)["touching"], set())

    def test_catalogue_launch_and_serialized_launch_construct_adapter(self):
        from gym_anything_robotics.tasks import resolve, TaskSpec
        from gym_anything_robotics.launch import LaunchSpec
        with patch("gym_anything_robotics.env.RoboticsEnv", side_effect=lambda backend: backend):
            for selection in SELECTIONS[:-1]:
                base = LaunchSpec("whiteboard", backend_options={"recording":False,"cameras":[],
                    "output_dir":self.directory.name,"texture_width":1024,"texture_height":512})
                spec = resolve("whiteboard", selection, base=base)
                restored = TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
                backend = restored.create()
                self.assertIsInstance(backend, NewtonWhiteboardTaskSetBackend)
                self.assertEqual(backend.evaluate()["score"], 0)
                backend.close()
        with patch("gym_anything_robotics.launch.registry.create", return_value="native") as native:
            self.assertEqual(LaunchSpec("whiteboard", backend_options={"variation_id":0}).create(), "native")
            native.assert_called_once_with("whiteboard", backend="newton", robot=None, variation_id=0)

    def test_durable_checkpoints_preserve_task_and_authored_baseline(self):
        # Exercise the real checkpoint codec; FakeEnv supplies only numeric state.
        from whiteboard.storage import snapshot_bytes, restore_bytes
        backend = self.backend("correct-misspelling")
        env = backend.env
        env.fixture_mode = False
        env._mechanics_fingerprint = "stubbed-frozen-physics"
        before = snapshot_bytes(env)
        env.ink.tiles.clear()
        restore_bytes(env, before)
        self.assertEqual(backend.evaluate()["score"], 0)
        env.task = CatalogueTask("erase-board")
        with self.assertRaises(ValueError): restore_bytes(env, before)


if __name__ == "__main__": unittest.main()

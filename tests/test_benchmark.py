"""Live HTTP orchestration tests with fake physics; separate real runtime proof."""
import base64
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from gym_anything_robotics.benchmark import run, save
from gym_anything_robotics.evidence import export
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.policy_access import PolicyAccess
from gym_anything_robotics.remote import RemoteEnv
from gym_anything_robotics.session_service import SessionService, SessionServer
from tests.test_http_server import HTTPFakeBackend

ACTION = {"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.backend = HTTPFakeBackend(self.root)
        self.backend.policy_access = lambda: PolicyAccess(
            "yam", self.backend.capabilities.camera_specs, {("tick",): None},
            frozenset({"cartesian_delta"}), lambda: "Test instruction", {})
        self.backend.act_policy = self.backend.act
        self.backend.evaluate = lambda: {"success": False, "score": .25}
        self.env = RoboticsEnv(self.backend)
        self.service = SessionService(self.env, session_id="test")
        self.server = SessionServer(("127.0.0.1", 0), self.service,
                                    api_token="owner-secret", policy_api_token="student-secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        save(self.root/"keys.json", {"privileged": "owner-secret", "policy": "student-secret"})
        save(self.root/"session.json", {
            "session_id": "test", "url": self.url,
            "launch": {"spec": {"environment_id": "fake"}, "source": {},
                       "credential_file": str(self.root/"keys.json")}})
        self.resolve_patch = patch("gym_anything_robotics.sessions.resolve", return_value=self.root)
        self.client_patch = patch("gym_anything_robotics.sessions.client",
            side_effect=lambda *a, **k: RemoteEnv(self.url, api_key="owner-secret"))
        self.resolve_patch.start()
        self.client_patch.start()
        self.seen = []

    def tearDown(self):
        self.resolve_patch.stop()
        self.client_patch.stop()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.service.history.store.close()
        self.env.close()
        self.tmp.cleanup()

    def factory(self, context, options):
        self.seen.append(deepcopy(context))
        def step(observation):
            self.seen.append(deepcopy(observation))
            return deepcopy(ACTION)
        return step

    def exercise(self, output_name="run", **kwargs):
        with patch("gym_anything_robotics.benchmark.agent_factory",
                   return_value=(self.factory, {"entry_point": "test:factory"})):
            return run("test", agent="test:factory", max_actions=2,
                       output=self.root/output_name, **kwargs)

    def test_two_actions_restricted_inputs_same_session_paused(self):
        result = self.exercise()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["evaluation"]["success"])
        self.assertEqual(result["actions_completed"], 2)
        self.assertTrue(self.service.paused)
        self.assertFalse(self.backend.closed)
        self.assertEqual(len(self.backend.actions), 2)
        visible = json.dumps(self.seen)
        for secret in ("owner-secret", "student-secret", "secret_object_pose", "evaluation"):
            self.assertNotIn(secret, visible)
        self.assertIn("data_base64", visible)
        self.assertTrue((self.root/"run/observation-000002/camera-000.bin").exists())
        self.assertEqual(self.backend.last_reset, {"seed": None, "options": {}})
        self.assertTrue(all(row["role"] == "policy" for row in self.service.history.store.list()["records"]))

    def test_task_pictures_reach_the_agent_context_as_data(self):
        self.exercise()
        self.assertEqual(set(self.seen[0]["task"]), {"ok", "instruction"})
        access = self.backend.policy_access
        self.backend.policy_access = lambda: replace(access(), task_images=lambda: {"reference": ("Copy this", b"png")})
        self.seen.clear()
        self.exercise(resume=True, output_name="with-picture")
        picture = self.seen[0]["task"]["reference_images"]["reference"]
        self.assertNotIn("url", picture)
        self.assertEqual(base64.b64decode(picture["data_base64"]), b"png")
        self.assertEqual((picture["description"], picture["bytes"]), ("Copy this", 3))

    def test_pause_requires_explicit_resume(self):
        self.service.paused = True
        with self.assertRaises(ValueError):
            self.exercise()
        self.assertEqual(self.backend.actions, [])
        result = self.exercise(resume=True)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(self.service.paused)

    def test_action_failure_no_retry_and_pause(self):
        self.backend.fail_action = True
        result = self.exercise()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["actions_completed"], 0)
        self.assertTrue(self.service.paused)
        self.assertEqual(len(list((self.root/"run").glob("action-*.json"))), 1)

    def test_unknown_evaluator_rejects_before_actions(self):
        self.backend.evaluate = lambda: None
        with self.assertRaises(ValueError):
            self.exercise()
        self.assertEqual(self.backend.actions, [])

    def test_export_hashes_and_absent_trace_explicit(self):
        self.exercise()
        tick = self.backend.tick
        result = export("test", self.root/"export", include_traces=True)
        self.assertTrue(result["stable_export"])
        self.assertEqual(result["retained_records_exported"], 2)
        self.assertTrue(result["trace_gaps"])
        self.assertFalse(result["claims"]["uninterrupted_rollout_verified"])
        self.assertEqual(self.backend.tick, tick)
        encoded = json.dumps(result)
        self.assertNotIn("owner-secret", encoded)
        for artifact in result["artifacts"]:
            import hashlib
            self.assertEqual(hashlib.sha256((self.root/"export"/artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])

    def test_batches_cannot_bypass_action_budget(self):
        with patch("gym_anything_robotics.benchmark.agent_factory",
                   return_value=(lambda c, o: lambda obs: {"actions": [ACTION, ACTION]}, {})):
            result = run("test", agent="test:factory", max_actions=1, output=self.root/"batch")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.backend.actions, [])
        self.assertTrue(self.service.paused)

    def test_concurrent_mutation_invalidates_attribution(self):
        def concurrent(context, options):
            def step(observation):
                RemoteEnv(self.url, api_key="owner-secret").act(ACTION)
                return deepcopy(ACTION)
            return step
        with patch("gym_anything_robotics.benchmark.agent_factory", return_value=(concurrent, {})):
            result = run("test", agent="test:factory", max_actions=1, output=self.root/"concurrent")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["exclusive_recorded_interval"])
        self.assertTrue(self.service.paused)

    def test_export_does_not_implicitly_pause(self):
        with self.assertRaises(ValueError):
            export("test", self.root/"no-export")
        self.assertFalse(self.service.paused)

    def test_agent_finish_is_not_task_success(self):
        with patch("gym_anything_robotics.benchmark.agent_factory",
                   return_value=(lambda c, o: lambda obs: None, {})):
            result = run("test", agent="test:factory", max_actions=2, output=self.root/"done")
        self.assertEqual(result["stop_reason"], "agent_finished")
        self.assertEqual(result["actions_completed"], 0)
        self.assertFalse(result["evaluation"]["success"])

"""Transport/lifecycle checks; no claim about physical qualification."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from gym_anything_robotics import LaunchSpec, connect
from gym_anything_robotics.cli import main
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.policy_access import PolicyAccess
from gym_anything_robotics.remote import RemoteError
from gym_anything_robotics.session_service import SessionServer, SessionService
from tests.test_http_server import HTTPFakeBackend
from gym_anything_robotics import sessions

ACTION = {"type": "cartesian_delta", "arm": "left", "delta_m": [0, 0, .001]}


class LaunchTest(unittest.TestCase):
    def test_exact_passthrough_and_detached_input(self):
        options = {"config": {"dt": .002, "iterations": 64}, "cameras": ["g1_rgb"]}
        spec = LaunchSpec("football", backend_options=options)
        options["config"]["dt"] = .02
        with patch("gym_anything_robotics.registry.create") as create:
            spec.create()
        self.assertEqual(create.call_args.kwargs["config"]["dt"], .002)
        self.assertEqual(create.call_args.kwargs["cameras"], ["g1_rgb"])
        self.assertEqual(LaunchSpec.from_dict(spec.to_dict()).to_dict(), spec.to_dict())

    def test_invalid_spec_never_constructs(self):
        bad = [{"schema": "v0", "environment_id": "football"},
               {"schema": "gar.launch.v1", "environment_id": "missing"},
               {"schema": "gar.launch.v1", "environment_id": "football", "batch_size": 5},
               {"schema": "gar.launch.v1", "environment_id": "football",
                "backend_options": {"auto_reset": False}}]
        with patch("gym_anything_robotics.registry.create") as create:
            for value in bad:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    LaunchSpec.from_dict(value)
            create.assert_not_called()

    def test_read_only_cli_does_not_construct(self):
        with patch("gym_anything_robotics.registry.create") as create, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["list"]), 0)
            self.assertEqual(main(["inspect", "welding"]), 0)
        create.assert_not_called()

    def test_conflicting_config_cli_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "football", "--config", "never-read.json"]), 1)

    def test_device_conflicts_never_create_a_session(self):
        with patch("gym_anything_robotics.sessions.prepare") as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "football", "--device", "cpu", "--backend-options-json",
                                   '{"config":{"device":"cpu"}}']), 1)
            prepare.assert_not_called()
        with self.assertRaises(ValueError):
            LaunchSpec("football", backend_options={"device": "cpu", "config": {"device": "cpu"}})

    def test_failed_startup_is_recorded_and_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "GAR_SESSION_ROOT": root+"/sessions", "GAR_CREDENTIAL_ROOT": root+"/credentials",
        }), patch("gym_anything_robotics.sessions._source", return_value={}), patch.object(LaunchSpec, "create", side_effect=ValueError("invalid physical config")), contextlib.redirect_stderr(io.StringIO()):
            directory = sessions.prepare(LaunchSpec("football"), port=0)
            self.assertEqual(sessions.serve(directory), 1)
            self.assertEqual(json.loads((directory/"session.json").read_text())["status"], "failed")
            with self.assertRaises(ValueError):
                sessions.serve(directory)

    def test_process_launch_failure_has_failed_manifest(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "GAR_SESSION_ROOT": root+"/sessions", "GAR_CREDENTIAL_ROOT": root+"/credentials",
        }), patch("gym_anything_robotics.sessions._source", return_value={}), patch("subprocess.Popen", side_effect=OSError("launch failed")):
            directory = sessions.prepare(LaunchSpec("football"), port=0)
            with self.assertRaises(OSError):
                sessions.start(directory)
            self.assertEqual(json.loads((directory/"session.json").read_text())["status"], "failed")

    def test_explicit_empty_values_are_not_silently_substituted(self):
        with patch.dict(os.environ, {"GAR_HTTP_API_TOKEN": ""}):
            with self.assertRaises(ValueError):
                sessions.prepare(LaunchSpec("football"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "football", "--backend", ""]), 1)
        with self.assertRaises(ValueError):
            connect("http://localhost", api_key="test", timeout_s=float("nan"))

    def test_manifest_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "GAR_SESSION_ROOT": root+"/sessions", "GAR_CREDENTIAL_ROOT": root+"/credentials",
            "GAR_HTTP_API_TOKEN": "owner-secret", "GAR_HTTP_POLICY_API_TOKEN": "student-secret",
        }), patch("gym_anything_robotics.sessions._source", return_value={}):
            directory = sessions.prepare(LaunchSpec("football"), port=0)
            manifest = (directory/"session.json").read_text()
            self.assertNotIn("owner-secret", manifest)
            self.assertNotIn("student-secret", manifest)
            key_file = Path(json.loads(manifest)["launch"]["credential_file"])
            if os.name != "nt":
                self.assertEqual(key_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(sessions.status(directory)["manifest"]["status"], "starting")


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = HTTPFakeBackend(Path(self.temp.name))
        self.backend.policy_access = lambda: PolicyAccess(
            "yam", self.backend.capabilities.camera_specs, {("tick",): None},
            frozenset({"cartesian_delta"}), lambda: "Test task", {})
        self.backend.act_policy = self.backend.act
        self.env = RoboticsEnv(self.backend)
        self.service = SessionService(self.env, session_id="test-session")
        self.server = SessionServer(("127.0.0.1", 0), self.service,
                                    api_token="owner", policy_api_token="student")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.owner = connect(self.url, api_key="owner", timeout_s=3)
        self.student = connect(self.url, api_key="student", timeout_s=3)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()
        self.env.close()
        self.temp.cleanup()

    def test_attach_observe_detach_never_steps_resets_or_closes(self):
        self.backend.tick = 12
        with connect(self.url, api_key="owner") as client:
            self.assertEqual(client.obs()["proprioception"]["tick"], 12)
            self.assertEqual(client.session()["session_id"], "test-session")
        self.assertFalse(self.backend.closed)
        self.assertEqual(self.owner.get_state()["tick"], 12)
        self.assertEqual(self.backend.tick, 12)

    def test_pause_rejects_mutations_but_keeps_observations(self):
        self.owner.act(ACTION)
        self.owner.pause()
        for operation in (lambda: self.owner.act(ACTION), self.owner.reset,
                          lambda: self.owner.request("/v1/checkpoints/restore", payload={"checkpoint": "nope"})):
            with self.assertRaises(RemoteError) as caught:
                operation()
            self.assertFalse(caught.exception.payload["committed"])
        self.assertEqual(self.owner.obs()["proprioception"]["tick"], 1)
        self.assertTrue(self.owner.session()["health"]["ready"])
        self.owner.resume()
        self.assertEqual(self.backend.tick, 1)
        self.owner.act(ACTION)
        self.assertEqual(self.backend.tick, 2)

    def test_policy_lifecycle_and_privileged_information_denied(self):
        for route in ("/v1/session", "/v1/state", "/v1/history", "/v1/geometry"):
            with self.subTest(route=route), self.assertRaises(RemoteError) as caught:
                self.student.request(route)
            self.assertEqual(caught.exception.status, 403)
        for method in (self.student.pause, self.student.resume, self.student.stop):
            with self.assertRaises(RemoteError) as caught:
                method()
            self.assertEqual(caught.exception.status, 403)
        self.student.act(ACTION)
        self.assertEqual(self.backend.tick, 1)

    def test_idempotency_and_no_replay_on_client_close(self):
        first = self.owner.act(ACTION, idempotency_key="once")
        second = self.owner.act(ACTION, idempotency_key="once")
        self.assertEqual(first, second)
        self.owner.close()
        self.assertEqual(self.backend.tick, 1)
        with self.assertRaises(RuntimeError):
            self.owner.obs()

    def test_stop_explicitly_ends_server_loop(self):
        result = self.owner.stop()
        self.assertTrue(result["stopping"])
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())

    def test_managed_help_adds_owner_routes_only(self):
        owner = self.owner.request("/v1/help")
        student = self.student.request("/v1/help")
        self.assertIn("POST /v1/session/pause", owner["routes"])
        self.assertNotIn("/v1/session", json.dumps(student))


if __name__ == "__main__":
    unittest.main()

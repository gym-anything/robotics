"""Launch output/readiness tests: real HTTP, no claims about physical fidelity."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from gym_anything_robotics import LaunchSpec, sessions, session_handoff
from gym_anything_robotics.cli import main
from gym_anything_robotics.env import RoboticsEnv
from gym_anything_robotics.policy_access import PolicyAccess
from gym_anything_robotics.session_service import SessionServer, SessionService
from tests.test_http_server import HTTPFakeBackend


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env_patch = patch.dict(os.environ, {
            "GAR_SESSION_ROOT": str(self.root / "sessions"),
            "GAR_CREDENTIAL_ROOT": str(self.root / "credentials"),
            "GAR_HTTP_API_TOKEN": "owner-test-only", "GAR_HTTP_POLICY_API_TOKEN": "policy-test-only",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        with patch("gym_anything_robotics.sessions._source", return_value={}):
            self.directory = sessions.prepare(LaunchSpec("laundry", backend_options={
                "config": {"garment": "shirt"}}), port=0, policy=True)
        self.manifest = json.loads((self.directory / "session.json").read_text())
        self.backend = HTTPFakeBackend(self.root)
        self.backend.policy_access = lambda: PolicyAccess(
            "yam", self.backend.capabilities.camera_specs, {("tick",): None},
            frozenset({"cartesian_delta"}), lambda: "Fold the shirt", {})
        self.env = RoboticsEnv(self.backend)
        self.addCleanup(self.env.close)
        self.service = SessionService(self.env, session_id=self.manifest["session_id"])
        self.server = SessionServer(("127.0.0.1", 0), self.service,
            api_token="owner-test-only", policy_api_token="policy-test-only")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def running(self):
        self.manifest.update(status="running", url=self.url)
        sessions.write_json(self.directory / "session.json", self.manifest)

    def cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), patch("gym_anything_robotics.cli._port_check"):
            code = main(["--format", "json", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_live_handoff_authenticates_both_keys_without_steps(self):
        self.running()
        initial_reset = self.backend.last_reset
        report = session_handoff.describe(self.directory)
        self.assertTrue(report["ready"])
        self.assertTrue(report["authentication_verified"])
        self.assertEqual(report["url"], self.url)
        self.assertEqual(report["dashboard_url"], self.url + "/dashboard")
        self.assertEqual(report["task"]["instruction"], "Fold the shirt")
        self.assertEqual(report["api_keys"], {"privileged": "owner-test-only", "policy": "policy-test-only"})
        self.assertEqual(self.backend.tick, 0)
        self.assertFalse(self.backend.closed)
        self.assertIs(self.backend.last_reset, initial_reset)
        self.assertNotIn("owner-test-only", (self.directory / "session.json").read_text())

    def test_paused_environment_is_ready_but_not_resumed(self):
        self.running()
        self.service.paused = True
        report = session_handoff.describe(self.directory)
        self.assertTrue(report["ready"])
        self.assertTrue(report["paused"])
        self.assertTrue(self.service.paused)
        self.assertEqual(self.backend.tick, 0)

    def test_wrong_session_cannot_be_reported_ready(self):
        self.running()
        self.service.session_id = "another-session"
        report = session_handoff.describe(self.directory)
        self.assertFalse(report["ready"])
        self.assertIn("identity", report["error"])

    def test_invalid_policy_key_cannot_be_reported_ready(self):
        self.running()
        sessions.write_json(self.manifest["launch"]["credential_file"],
            {"privileged": "owner-test-only", "policy": "wrong-key"}, private=True)
        report = session_handoff.describe(self.directory)
        self.assertFalse(report["ready"])
        self.assertIn("401", report["error"])
        self.assertEqual(report["status"], "authentication_failed")

    def test_owner_only_handoff(self):
        self.running()
        sessions.write_json(self.manifest["launch"]["credential_file"],
            {"privileged": "owner-test-only", "policy": None}, private=True)
        report = session_handoff.describe(self.directory)
        self.assertTrue(report["ready"])
        self.assertIsNone(report["api_keys"]["policy"])

    def test_false_health_is_not_ready(self):
        self.running()
        with patch.object(self.service, "health", return_value={"ready": False}):
            report = session_handoff.wait(self.directory, timeout_s=1)
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "not_ready")

    def test_warning_details_are_preserved_and_secrets_redacted(self):
        self.running()
        Path(self.manifest["log"]).write_text(
            "Warp initialized\nUserWarning: mesh quality\n  - one sliver triangle\nConsider remeshing.\nready\nWarning: owner-test-only\n")
        report = session_handoff.describe(self.directory)
        warnings = "\n".join(report["startup_warnings"])
        self.assertIn("one sliver triangle", warnings)
        self.assertIn("Consider remeshing.", warnings)
        self.assertNotIn("owner-test-only", warnings)
        self.assertNotIn("Warp initialized", warnings)

    def test_failed_start_returns_error_and_log_without_wait(self):
        self.manifest.update(status="failed", error="ValueError: bad config")
        sessions.write_json(self.directory / "session.json", self.manifest)
        Path(self.manifest["log"]).write_text("Traceback\nValueError: bad config\n")
        with patch("gym_anything_robotics.session_handoff.time.sleep") as sleep:
            report = session_handoff.wait(self.directory, timeout_s=1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("bad config", report["log_tail"])
        sleep.assert_not_called()

    def test_dead_child_without_final_manifest_is_detected(self):
        with patch("gym_anything_robotics.session_handoff._process_alive", return_value=False):
            report = session_handoff.wait(self.directory, timeout_s=1, pid=123)
        self.assertEqual(report["status"], "failed")
        self.assertIn("exited", report["error"])

    def test_timeout_leaves_process_and_manifest_untouched(self):
        original = (self.directory / "session.json").read_bytes()
        with patch("gym_anything_robotics.session_handoff._process_alive", return_value=True):
            report = session_handoff.wait(self.directory, timeout_s=.01)
        self.assertEqual(report["status"], "timeout")
        self.assertFalse(report["ready"])
        self.assertIn("not stopped", report["error"])
        self.assertEqual(original, (self.directory / "session.json").read_bytes())
        self.assertFalse(self.backend.closed)

    def test_wait_observes_starting_to_live_ready_transition(self):
        with patch("gym_anything_robotics.session_handoff.time.sleep", side_effect=lambda _: self.running()):
            report = session_handoff.wait(self.directory, timeout_s=3)
        self.assertTrue(report["ready"])
        self.assertEqual(self.backend.tick, 0)

    def test_run_default_waits_and_prints_handoff(self):
        self.running()
        receipt = {"session_id": self.manifest["session_id"], "pid": os.getpid()}
        with patch("gym_anything_robotics.sessions.prepare", return_value=self.directory) as prepare, patch("gym_anything_robotics.sessions.start", return_value=receipt):
            code, out, err = self.cli(["run", "laundry", "--backend-options-json",
                '{"config":{"garment":"shirt"}}', "--host", "0.0.0.0", "--port", "8896", "--policy-key"])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["ready"])
        self.assertIn("api_keys", json.loads(out))
        self.assertEqual(prepare.call_args.args[0].backend_options, {"config": {"garment": "shirt"}})
        self.assertEqual(prepare.call_args.kwargs["port"], 8896)
        self.assertIn("Starting laundry", err)

    def test_run_no_wait_is_explicit(self):
        with patch("gym_anything_robotics.sessions.prepare", return_value=self.directory), patch("gym_anything_robotics.sessions.start", return_value={"status": "starting"}), patch("gym_anything_robotics.session_handoff.wait") as wait:
            code, out, _ = self.cli(["run", "laundry", "--no-wait"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "starting")
        wait.assert_not_called()

    def test_run_failure_has_nonzero_exit(self):
        with patch("gym_anything_robotics.sessions.prepare", return_value=self.directory), patch("gym_anything_robotics.sessions.start", return_value={"session_id": "test", "pid": 123}), patch("gym_anything_robotics.session_handoff.wait", return_value={"ready": False, "status": "failed", "error": "failed config"}):
            code, out, _ = self.cli(["run", "laundry"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["error"], "failed config")

    def test_interrupt_stops_waiting_not_server(self):
        with patch("gym_anything_robotics.sessions.prepare", return_value=self.directory), patch("gym_anything_robotics.sessions.start", return_value={"session_id": "test", "pid": 123}), patch("gym_anything_robotics.session_handoff.wait", side_effect=KeyboardInterrupt):
            code, _, err = self.cli(["run", "laundry"])
        self.assertEqual(code, 130)
        self.assertIn("NOT stopped", err)
        self.assertFalse(self.backend.closed)

    def test_info_retrieves_live_handoff_without_launching(self):
        self.running()
        with patch("gym_anything_robotics.sessions.start") as start:
            code, out, err = self.cli(["sessions", "info", str(self.directory)])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["ready"])
        start.assert_not_called()

    def test_invalid_timeout_rejected_before_preparing_session(self):
        with patch("gym_anything_robotics.sessions.prepare") as prepare:
            for value in ("0", "-1", "nan", "inf"):
                code, _, _ = self.cli(["run", "laundry", "--startup-timeout", value])
                self.assertEqual(code, 1)
        prepare.assert_not_called()

    def test_windows_liveness_never_uses_kill(self):
        with patch("gym_anything_robotics.session_handoff.os.name", "nt"), patch("os.kill") as kill:
            self.assertTrue(session_handoff._process_alive(123))
        kill.assert_not_called()

    def test_unavailable_task_is_explicit_not_false_startup_failure(self):
        self.running()
        with patch.object(self.service, "task", side_effect=ValueError("task metadata unavailable")):
            report = session_handoff.describe(self.directory)
        self.assertTrue(report["ready"])
        self.assertFalse(report["task"]["available"])

    def test_os_spawn_failure_prints_failed_handoff(self):
        self.manifest.update(status="failed", error="OSError: cannot spawn")
        sessions.write_json(self.directory / "session.json", self.manifest)
        with patch("gym_anything_robotics.sessions.prepare", return_value=self.directory), patch("gym_anything_robotics.sessions.start", side_effect=OSError("cannot spawn")):
            code, out, _ = self.cli(["run", "laundry"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["status"], "failed")
        self.assertEqual(json.loads(out)["log"], self.manifest["log"])


if __name__ == "__main__":
    unittest.main()

"""Whole public CLI UX coverage, with owned HTTP fixtures for mutations."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from gym_anything_robotics import cli, cli_sessions, sessions, tasks
from gym_anything_robotics.cli_parser import parser
from gym_anything_robotics.cli_output import render
from tests import test_session_handoff as fixture


def invoke(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(args)
        except SystemExit as error:
            code = error.code
    return code, out.getvalue(), err.getvalue()


class GrammarTests(unittest.TestCase):

    def test_every_public_help_is_readable_and_no_physics(self):
        commands = [
            [], ["--help"], ["list"], ["inspect", "laundry"], ["doctor", "laundry"],
            ["run"], ["serve"], ["sessions"], ["sessions", "list"],
            ["sessions", "status"], ["sessions", "info"], ["sessions", "credentials"],
            ["sessions", "pause"], ["sessions", "resume"], ["sessions", "stop"],
            ["sessions", "logs"], ["tasks"], ["tasks", "list"], ["tasks", "inspect"],
            ["tasks", "resolve"], ["status"], ["keys"], ["pause"], ["resume"], ["stop"],
            ["logs"], ["observe"], ["observation"], ["state"], ["action"], ["dashboard"],
            ["verify"], ["benchmark"], ["export"], ["test"], ["create"], ["audit"],
        ]
        with patch("gym_anything_robotics.registry.create") as create:
            for command in commands:
                args = command if command in ([], ["--help"]) else command + ["--help"]
                code, out, err = invoke(args)
                self.assertEqual(code, 0, (args, err))
                self.assertIn("gar", out)
                self.assertNotIn("_serve", out)
            create.assert_not_called()
        self.assertLess(len(invoke([])[1].splitlines()), 28)

    def test_discovery_and_named_task_are_human_by_default(self):
        for args, expected in ((["list"], "ENVIRONMENT"), (["tasks"], "SELECTORS"),
                               (["tasks", "laundry"], "shirt"),
                               (["inspect", "laundry"], "dual-panda"),
                               (["doctor", "laundry"], "DEPENDENCIES")):
            code, out, err = invoke(args)
            self.assertEqual(code, 0, err)
            self.assertIn(expected, out)
            self.assertFalse(out.startswith(("{", "[")))
        code, out, err = invoke(["tasks", "resolve", "laundry", "--task", "shirt"])
        self.assertEqual(code, 0, err)
        self.assertIn("Fold both long sleeves", out)

    def test_json_is_explicit_and_can_be_before_or_after_command(self):
        for args in (["--json", "list"], ["list", "--json"], ["list", "--format", "json"]):
            code, out, err = invoke(args)
            self.assertEqual(code, 0, err)
            self.assertIsInstance(json.loads(out), list)
        code, out, err = invoke(["tasks", "laundry", "--json"])
        self.assertEqual(json.loads(out)["cli_task_selector"], "garment")

    def test_invalid_input_has_actionable_error_no_traceback(self):
        for args in (["typo"], ["stop"], ["run"], ["action", "--bad-flag"]):
            code, out, err = invoke(args)
            self.assertNotEqual(code, 0)
            self.assertIn("Error:", err)
            self.assertNotIn("Traceback", out + err)
        code, out, err = invoke(["--json", "typo"])
        self.assertFalse(json.loads(err)["ok"])
        self.assertEqual(out, "")
        self.assertNotIn("_serve", err)
        code, out, err = invoke(["sesions"])
        self.assertIn("gar sessions", err)

    def test_friendly_launch_honors_exact_values_and_conflicts(self):
        args = parser().parse_args(["run", "laundry", "--task", "shirt", "--device", "cuda:0"])
        spec, task = cli.cli_launch.launch(args)
        self.assertEqual(spec.backend_options, {"device": "cuda:0", "config": {"garment": "shirt"}})
        self.assertEqual(task.launch.to_dict(), spec.to_dict())
        for suffix in (["--backend-options-json", '{"config":{"garment":"tee"}}'],
                       ["--select", "garment=shirt"], ["--task-file", "task.json"]):
            with patch("gym_anything_robotics.sessions.prepare") as prepare:
                code, _, _ = invoke(["run", "laundry", "--task", "shirt", *suffix])
                self.assertEqual(code, 1)
                prepare.assert_not_called()
        for args in (["run", "laundry", "--task", "hat"],
                     ["run", "laundry", "--select", "garment"],
                     ["run", "laundry", "--variation", "3"],
                     ["run", "whiteboard", "--variation", "10000"]):
            code, _, err = invoke(args)
            self.assertEqual(code, 1, (args, err))

    def test_saved_task_and_json_stdout_are_both_valid_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            code, out, err = invoke(["tasks", "resolve", "laundry", "--task", "shirt",
                                     "--output", str(path), "--json"])
            self.assertEqual(code, 0, err)
            self.assertEqual(tasks.TaskSpec.from_dict(json.loads(out)).to_dict(),
                             tasks.TaskSpec.load(path).to_dict())

    def test_state_output_does_not_dump_cloth_arrays(self):
        text = render("state", {"tick": 10, "garment": {"positions_m": [[1, 2, 3]] * 1104}})
        self.assertIn("1104 items", text)
        self.assertNotIn("[[", text)
        self.assertLess(len(text), 500)

    def test_narrow_tables_fit_and_terminal_escapes_are_removed(self):
        data = [{"environment_id": "laundry", "default_robot": "dual-panda",
                 "backend": "newton", "engine_version": "1.5"}]
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((40, 24))):
            text = render("environments", data)
        self.assertNotIn("\x1b", render("error", {"error": "\x1b[31mInjected"}))
        self.assertIn("laundry", text)

    def test_port_conflict_prevents_environment_creation(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with patch("gym_anything_robotics.sessions.prepare") as prepare:
                code, out, err = invoke(["run", "laundry", "--port", str(listener.getsockname()[1])])
                self.assertEqual(code, 1)
                self.assertIn("Cannot listen", err)
                prepare.assert_not_called()


class LiveCommandTests(unittest.TestCase):
    setUp = fixture.HandoffTest.setUp
    close_server = fixture.HandoffTest.close_server
    running = fixture.HandoffTest.running

    def set_running(self):
        self.running()
        self.short = self.manifest["session_id"][:8]

    def historical(self, identity, status):
        directory = self.root / "sessions" / identity
        directory.mkdir()
        manifest = dict(self.manifest, session_id=identity, status=status)
        sessions.write_json(directory / "session.json", manifest)
        return directory

    def test_active_list_excludes_history_and_keeps_paused(self):
        self.set_running()
        self.service.paused = True
        self.historical("a" * 32, "stopped")
        self.historical("b" * 32, "failed")
        code, out, err = invoke(["sessions"])
        self.assertEqual(code, 0, err)
        self.assertIn("paused", out)
        self.assertNotIn("aaaaaaaa", out)
        self.assertNotIn("bbbbbbbb", out)
        code, out, err = invoke(["sessions", "--all", "--json"])
        self.assertEqual(len(json.loads(out)["sessions"]), 3)
        self.assertTrue(self.service.paused)

    def test_dead_running_record_is_history_not_active(self):
        self.set_running()
        dead = self.historical("c" * 32, "running")
        value = json.loads((dead / "session.json").read_text())
        value.update(url="http://127.0.0.1:1", pid=999999999)
        sessions.write_json(dead / "session.json", value)
        result = cli_sessions.inventory()
        self.assertEqual(len(result["sessions"]), 1)
        history = cli_sessions.inventory(include_all=True)["sessions"]
        self.assertEqual(next(r["status"] for r in history if r["session_id"] == "c" * 32), "exited")

    def test_empty_list_is_actionable(self):
        self.manifest["status"] = "stopped"
        sessions.write_json(self.directory / "session.json", self.manifest)
        code, out, err = invoke(["sessions"])
        self.assertEqual(code, 0)
        self.assertIn("No active sessions", out)
        self.assertIn("gar run", out)

    def test_unique_environment_short_id_and_ambiguity(self):
        self.set_running()
        self.assertEqual(cli_sessions.choose("laundry"), self.directory)
        self.assertEqual(cli_sessions.choose(self.short), self.directory)
        clone = self.historical("d" * 32, "starting")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            cli_sessions.choose("laundry")
        with patch("gym_anything_robotics.sessions.client") as client:
            code, _, err = invoke(["stop", "laundry"])
            self.assertEqual(code, 1)
            self.assertIn("Ambiguous", err)
            client.assert_not_called()
        self.assertTrue(clone.exists())

    def test_status_keys_logs_and_aliases_have_separate_presentations(self):
        self.set_running()
        Path(self.manifest["log"]).write_text("line one\nowner-test-only\nline three\n")
        for args in (["status", self.short], ["sessions", "status", self.short],
                     ["sessions", "info", self.short]):
            code, out, err = invoke(args)
            self.assertEqual(code, 0, err)
            self.assertIn("READY", out)
            self.assertNotIn("owner-test-only", out)
        for args in (["keys", self.short], ["sessions", "credentials", self.short]):
            code, out, err = invoke(args)
            self.assertEqual(code, 0, err)
            self.assertIn("Privileged: owner-test-only", out)
        code, out, err = invoke(["logs", self.short, "--tail", "2"])
        self.assertEqual(code, 0, err)
        self.assertIn("<redacted>", out)
        self.assertNotIn("line one", out)
        self.assertNotIn("owner-test-only", out)

    def test_read_only_command_matrix_does_not_mutate(self):
        self.set_running()
        initial = self.backend.last_reset
        for args, text in ((["observe", self.short], "Observation"),
                           (["observation", "--session", self.short], "Observation"),
                           (["state", self.short], "Privileged state"),
                           (["dashboard", self.short], "/dashboard"),
                           (["action", self.short, "--help-actions"], "Permitted actions")):
            code, out, err = invoke(args)
            self.assertEqual(code, 0, (args, err))
            self.assertIn(text, out)
            self.assertNotIn("owner-test-only", out)
        self.assertEqual(self.backend.tick, 0)
        self.assertIs(self.backend.last_reset, initial)
        self.assertFalse(self.backend.closed)

    def test_action_help_only_lists_live_supported_commands(self):
        self.set_running()
        code, out, err = invoke(["action", self.short, "--help-actions", "--format", "json"])
        self.assertEqual(code, 0, err)
        self.assertEqual(set(json.loads(out)["action_types"]), {"cartesian_delta", "gripper"})

    def test_state_field_preserves_exact_scalar_and_invalid_path_is_actionable(self):
        self.set_running()
        code, out, err = invoke(["state", self.short, "--field", "tick", "--json"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), 0)
        code, out, err = invoke(["state", self.short, "--field", "missing"])
        self.assertEqual(code, 1)
        self.assertIn("No state field", err)

    def test_shutdown_timeout_is_not_claimed_as_stopped(self):
        self.set_running()
        with patch("gym_anything_robotics.cli_sessions._process_alive", return_value=True):
            code, out, err = invoke(["stop", self.short, "--timeout", "0.01"])
        self.assertEqual(code, 1)
        self.assertIn("Shutdown still in progress", err)
        self.assertNotIn("Stopped.", out)

    def test_action_outcomes_and_json_input_compatibility(self):
        self.set_running()
        payload = '{"type":"cartesian_delta","arm":"left","delta_m":[0,0,0.001]}'
        for flag in ("--data", "--json"):
            code, out, err = invoke(["action", self.short, flag, payload])
            self.assertEqual(code, 0, err)
            self.assertIn("Action committed", out)
            self.assertIn("Request:", out)
        self.assertEqual(self.backend.tick, 2)
        code, out, err = invoke(["action", self.short, "--data", "{"])
        self.assertEqual(code, 1)
        self.assertIn("Invalid JSON", err)
        self.assertEqual(self.backend.tick, 2)

    def test_committed_action_without_observation_is_nonzero_without_retry(self):
        self.set_running()
        with patch.object(self.service, "action", return_value={"ok": False, "committed": True,
                "observation": None, "request_id": "committed-fixture"}) as action:
            code, out, err = invoke(["action", self.short, "--data",
                '{"type":"cartesian_delta","arm":"left","delta_m":[0,0,0.001]}'])
        self.assertEqual(code, 1, err)
        self.assertIn("Action committed", out)
        self.assertIn("Do not resend", out)
        self.assertEqual(action.call_count, 1)

    def test_pause_resume_stop_are_exercised_only_on_owned_fixture(self):
        self.set_running()
        code, out, err = invoke(["pause", self.short])
        self.assertEqual(code, 0, err)
        self.assertIn("Paused", out)
        self.assertTrue(self.service.paused)
        code, out, err = invoke(["sessions", "resume", self.short])
        self.assertEqual(code, 0, err)
        self.assertFalse(self.service.paused)
        # Model the managed owner's final manifest after the HTTP loop exits.
        def owner_finishes():
            self.thread.join(timeout=3)
            self.manifest["status"] = "stopped"
            sessions.write_json(self.directory / "session.json", self.manifest)
        owner = threading.Thread(target=owner_finishes)
        owner.start()
        code, out, err = invoke(["stop", self.short, "--timeout", "5"])
        owner.join(timeout=5)
        self.assertEqual(code, 0, err)
        self.assertIn("Stopped.", out)
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.backend.tick, 0)

    def test_run_shows_readiness_keys_and_exact_named_task(self):
        self.set_running()
        receipt = {"session_id": self.manifest["session_id"], "pid": os.getpid()}
        with patch("gym_anything_robotics.cli._port_check"), patch("gym_anything_robotics.sessions.prepare", return_value=self.directory) as prepare, patch("gym_anything_robotics.sessions.start", return_value=receipt):
            code, out, err = invoke(["run", "laundry", "--task", "shirt", "--policy-key"])
        self.assertEqual(code, 0, err)
        self.assertIn("READY", out)
        self.assertIn("Privileged: owner-test-only", out)
        self.assertIn("Restricted: policy-test-only", out)
        self.assertEqual(prepare.call_args.kwargs["task"].launch.backend_options, {"config": {"garment": "shirt"}})

    def test_verify_benchmark_export_presentations(self):
        self.set_running()
        with patch.object(self.backend, "evaluate", return_value={"success": False, "score": .4}):
            code, out, err = invoke(["verify", self.short])
            self.assertEqual(code, 1, err)
            self.assertIn("NOT COMPLETE", out)
        with patch("gym_anything_robotics.benchmark.run", return_value={
            "status": "completed", "actions_completed": 2, "max_actions": 2,
            "evaluation": {"available": True, "success": True, "score": 1}}) as run:
            code, out, err = invoke(["benchmark", self.short, "--agent", "fixture:agent",
                "--max-actions", "2", "--output", str(self.root / "benchmark")])
            self.assertEqual(code, 0, err)
            self.assertIn("Actions: 2 / 2", out)
            self.assertIn("SUCCEEDED", out)
            self.assertEqual(run.call_args.kwargs["max_actions"], 2)
        with patch("gym_anything_robotics.evidence.export", return_value={
            "retained_records_exported": 2, "starts_at_tick": 0, "ends_at_tick": 2,
            "trace_gaps": [], "recorded_discontinuities": [], "incomplete_records": []}):
            code, out, err = invoke(["export", self.short, "--output", str(self.root / "export")])
            self.assertEqual(code, 0, err)
            self.assertIn("not a video", out)
        with patch("unittest.defaultTestLoader.loadTestsFromNames", return_value=unittest.TestSuite()):
            code, out, err = invoke(["test"])
            self.assertEqual(code, 0, err)
            self.assertIn("Contract tests PASSED", out)

    def test_foreground_serve_prints_handoff_and_keeps_keys_out_of_log(self):
        original = sessions.serve
        directory = self.directory
        def serve(path, *, ready_callback):
            def ready():
                ready_callback()
                with sessions.client(path) as remote:
                    remote.stop()
            return original(path, ready_callback=ready)
        with patch("gym_anything_robotics.sessions.prepare", return_value=directory), patch("gym_anything_robotics.sessions.serve", side_effect=serve), patch("gym_anything_robotics.launch.LaunchSpec.create", return_value=self.env):
            code, out, err = invoke(["serve", "laundry", "--port", "0", "--policy-key"])
        self.assertEqual(code, 0, err)
        self.assertIn("READY", out)
        self.assertIn("Privileged: owner-test-only", out)
        self.assertNotIn("owner-test-only", (directory / "server.log").read_text())
        self.assertEqual(json.loads((directory / "session.json").read_text())["status"], "stopped")


if __name__ == "__main__":
    unittest.main()

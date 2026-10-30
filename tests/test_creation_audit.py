"""Launcher contract tests, NOT live agent audits or physics qualification."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gym_anything_robotics import cli
from gym_anything_robotics.creation_audit import method, video_review

FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys, uuid
a = sys.argv[1:]
cc = "-p" in a
resume = "--resume" in a if cc else "resume" in a
prompt = a[a.index("-p") + 1] if cc else a[a.index("--json") + 1]
session = (a[a.index("--resume" if resume else "--session-id") + 1] if cc
           else a[a.index("resume") + 1] if resume else str(uuid.uuid4()))
state = pathlib.Path(".fixture-session-" + session)
if resume and not state.exists():
    sys.exit(15)
state.touch()
with open("calls.jsonl", "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "session": session,
                        "resume": resume, "prompt": prompt, "args": a}) + "\n")
behavior = os.environ.get("FAKE_AUDIT_BEHAVIOR")
if behavior == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    pathlib.Path("child.pid").write_text(str(child.pid))
    import time
    time.sleep(300)
if behavior == "exit":
    sys.exit(7)
if "Note: save file is " in prompt and behavior != "missing":
    report = prompt.split("Note: save file is ", 1)[1].split("\n", 1)[0]
    pathlib.Path(report).write_text("Fixture report. Not a real audit.\n")
if cc:
    print(json.dumps({"session_id": session, "is_error": behavior == "failed",
                      "subtype": "error" if behavior == "failed" else "success"}))
else:
    print("fixture startup noise")
    if behavior != "no-id":
        print(json.dumps({"type": "thread.started",
                          "thread_id": str(uuid.uuid4()) if behavior == "wrong-id" and resume else session}))
    print(json.dumps({"type": "turn.failed" if behavior == "failed" else "turn.completed"}))
'''


@unittest.skipUnless(os.name == "posix", "Shared upstream Linux runner")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "env with spaces").mkdir()
        self.binary = self.root / "fake-agent"
        self.binary.write_text(FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable))
        self.binary.chmod(0o700)

    def run_audit(self, backend="codex", **kwargs):
        return method.run_audits(
            env_dir="env with spaces", workspace=self.root, backend=backend,
            claude_bin=str(self.binary), codex_bin=str(self.binary),
            progress=lambda _: None, **kwargs)

    def calls(self):
        return [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]


    def test_failures_are_not_success_and_other_roles_still_run(self):
        for backend in ("cc", "codex"):
            for mode in ("missing", "failed", "exit"):
                with self.subTest(backend=backend, mode=mode), patch.dict(
                    os.environ, {"FAKE_AUDIT_BEHAVIOR": mode}):
                    result = self.run_audit(backend)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(len(result["auditors"]), 4)
                    self.assertTrue(all(e["status"] == "failed" for e in result["auditors"].values()))

    def invoke_events(self, events):
        log = self.root / "events.jsonl"
        payload = [{"type": "thread.started", "thread_id": "original-thread"}, *events]
        def emit(*args, **kwargs):
            log.write_text("".join(json.dumps(e) + "\n" for e in payload))
        with patch.object(method, "_run_agent", side_effect=emit):
            return method._invoke("codex", self.binary, "audit", session_id="original-thread",
                                  resume=True, workspace=self.root, timeout=60, log=log, model=None)

    def test_codex_reconnection_followed_by_completion_preserves_warning(self):
        events = [{"type": "turn.started"},
                  {"type": "error", "message": "Reconnecting... 2/5"},
                  {"type": "error", "message": "Reconnecting... 3/5"},
                  {"type": "turn.completed"}]
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            self.assertEqual(self.invoke_events(events), "original-thread")
        self.assertIn("recovered after 2 error event(s)", warnings.getvalue())
        self.assertIn("Reconnecting... 2/5", (self.root / "events.jsonl").read_text())

    def test_codex_unresolved_errors_and_terminal_failure_still_fail(self):
        for events in (
            [{"type": "error"}],
            [{"type": "turn.completed"}, {"type": "error"}],
            [{"type": "turn.completed"}, {"type": "turn.started"}],
            [{"type": "turn.failed"}, {"type": "turn.completed"}],
            [{"type": "error"}, {"type": "turn.failed"}],
        ):
            with self.subTest(events=events), self.assertRaises(RuntimeError):
                self.invoke_events(events)

    def test_bad_codex_session_is_not_resumed_or_accepted(self):
        for mode in ("no-id", "wrong-id"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"FAKE_AUDIT_BEHAVIOR": mode}):
                result = self.run_audit(auditors=["hack"])
                self.assertEqual(result["status"], "failed")


    def test_invalid_configuration_does_not_launch(self):
        for kwargs in ({"timeout_sec": 0}, {"auditors": []}, {"backend": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_audit(**kwargs)
        output = self.root / "existing"
        output.mkdir()
        marker = output / "old.md"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_audit(output=output)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse((self.root / "calls.jsonl").exists())

    def assert_not_running(self, pid):
        status = Path(f"/proc/{pid}/stat")
        self.assertTrue(not status.exists() or status.read_text().split()[2] == "Z")


    def check_interrupt(self, interrupt):
        output = self.root / "interrupted"
        proc = subprocess.Popen(
            [sys.executable, "-m", "gym_anything_robotics.creation_audit.method",
             "--env-dir", "env with spaces", "--workspace", str(self.root),
             "--backend", "codex", "--codex-bin", str(self.binary),
             "--auditor", "hack", "--output", str(output)],
            env={**os.environ, "FAKE_AUDIT_BEHAVIOR": "hang"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 8
            while not (self.root / "child.pid").exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((self.root / "child.pid").exists())
            proc.send_signal(interrupt)
            self.assertEqual(proc.wait(timeout=8), 130)
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "interrupted")
            self.assert_not_running(int((self.root / "child.pid").read_text()))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class PromptAndCliTests(unittest.TestCase):


    def test_gar_help_does_not_launch(self):
        out = io.StringIO()
        with patch.object(method, "_run_agent") as launch, contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as exited:
                cli.main(["audit", "--help"])
            self.assertEqual(exited.exception.code, 0)
            launch.assert_not_called()
        self.assertIn("--auditor", out.getvalue())

    def test_gar_reports_not_verdict_and_json(self):
        result = {"status": "reports_written", "output_directory": "/tmp/reports",
                  "auditors": {"hack": {"status": "report_written", "report": "/tmp/reports/hack/audit.md"}}}
        for fmt in ("text", "json"):
            out = io.StringIO()
            with patch.object(method, "from_args", return_value=result), contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["audit", "--env-dir", "deposit_cans", "--format", fmt]), 0)
            if fmt == "json":
                self.assertEqual(json.loads(out.getvalue()), result)
            else:
                self.assertIn("/tmp/reports/hack/audit.md", out.getvalue())
                self.assertIn("not an environment pass", out.getvalue())
        result["status"] = "failed"
        with patch.object(method, "from_args", return_value=result), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["audit", "--env-dir", "deposit_cans"]), 1)

    def test_video_not_configured_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("not configured", method._video_note())


class VideoHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / "fixture.mp4"
        self.video.write_bytes(b"fake video for mocked transport, never sent")
        self.client = Mock()
        self.file = SimpleNamespace(name="files/owned", uri="https://example/video",
                                    mime_type="video/mp4", state=SimpleNamespace(name="ACTIVE"))
        self.client.files.upload.return_value = self.file
        self.client.interactions.create.return_value.output_text = "Fixture assessment."

    def test_exact_model_and_question_then_cleanup(self):
        result = video_review.review(self.video, "Did it drift?", client=self.client)
        request = self.client.interactions.create.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertEqual(request["input"][1]["text"], "Did it drift?")
        self.assertEqual(result["assessment"], "Fixture assessment.")
        self.client.files.delete.assert_called_once_with(name="files/owned")
        self.assertTrue(self.video.exists())

    def test_provider_failure_is_not_assessment_and_upload_is_deleted(self):
        self.client.interactions.create.side_effect = RuntimeError("provider failed")
        with self.assertRaises(RuntimeError):
            video_review.review(self.video, "inspect", client=self.client)
        self.client.files.delete.assert_called_once_with(name="files/owned")

    def test_failed_processing_and_no_text(self):
        self.file.state.name = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "process"):
            video_review.review(self.video, "inspect", client=self.client)
        self.client.interactions.create.assert_not_called()
        self.file.state.name = "ACTIVE"
        self.client.interactions.create.return_value.output_text = ""
        with self.assertRaisesRegex(RuntimeError, "no video"):
            video_review.review(self.video, "inspect", client=self.client)

    def test_processing_timeout_cleans_upload(self):
        self.file.state.name = "PROCESSING"
        with patch.object(video_review.time, "monotonic", side_effect=[0, 601]):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                video_review.review(self.video, "inspect", client=self.client)
        self.client.files.delete.assert_called_once_with(name="files/owned")

    def test_cli_error_is_useful_but_redacts_key(self):
        output = Path(self.temp.name) / "failed.json"
        err = io.StringIO()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret"}), patch.object(
            video_review, "review", side_effect=RuntimeError("unavailable: key=test-secret https://example/?key=abc123 tail")
        ), contextlib.redirect_stderr(err):
            code = video_review.main(["--video", str(self.video), "--question", "inspect",
                                      "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("unavailable", err.getvalue())
        self.assertNotIn("test-secret", err.getvalue())
        self.assertNotIn("abc123", err.getvalue())
        self.assertIn("?key=[redacted] tail", err.getvalue())
        self.assertFalse(output.exists())

    def test_cleanup_failure_is_visible(self):
        self.client.files.delete.side_effect = RuntimeError("delete failed")
        with self.assertWarnsRegex(UserWarning, "Could not remove"):
            video_review.review(self.video, "inspect", client=self.client)


if __name__ == "__main__":
    unittest.main()

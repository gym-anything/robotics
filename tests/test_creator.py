"""Creator prompt delivery with executable CLI fixtures; no live model or physics."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gym_anything_robotics import cli
from gym_anything_robotics.creation_audit import creator

FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, re, subprocess, sys, uuid
args = sys.argv[1:]
cc = "-p" in args
resume = "--resume" in args if cc else "resume" in args
prompt = args[args.index("-p") + 1] if cc else args[args.index("--json") + 1]
session = (args[args.index("--resume" if resume else "--session-id") + 1] if cc
           else args[args.index("resume") + 1] if resume else str(uuid.uuid4()))
state = pathlib.Path(".session-" + session)
if resume and not state.exists():
    sys.exit(15)
state.touch()
match = re.search(r"^Save your (?:research|handoff) to: (.+)$", prompt, re.M)
if match:
    report = pathlib.Path(match[1])
    output = report.parent
else:
    # The blind nudge gives no report path: use the fixture's session memory.
    assert resume
    output = pathlib.Path(state.read_text())
    report = output / "handoff.md"
state.write_text(str(output))
manifest = json.loads((output / "run.json").read_text())
name = next(k for k, v in manifest["turns"].items() if v["status"] == "running")
with open("calls.jsonl", "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "session": session, "resume": resume,
                       "prompt": prompt, "args": args, "manifest": manifest,
                       "build_exists": (output / "build").exists(),
                       "nudge_exists": (output / "nudge").exists()}) + "\n")
if not resume:
    assert not (output / "build").exists()
    assert list(manifest["turns"]) == ["research"]
else:
    assert (output / "research-finished").is_file()
    assert (output / "research.md").read_text().strip()
if name == "nudge":
    assert manifest["turns"]["build"]["status"] == "completed"
behavior = os.environ.get("CREATOR_FIXTURE_BEHAVIOR", "")
if behavior == "hang-research" and not resume:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    pathlib.Path("child.pid").write_text(str(child.pid))
    import time
    time.sleep(300)
if behavior == name + "-exit":
    sys.exit(7)
if behavior != name + "-missing":
    report.write_text("" if behavior == name + "-empty" else name + ": fixture report, not real research or an environment.\n")
if not resume:
    (output / "research-finished").touch()
returned = str(uuid.uuid4()) if behavior == name + "-wrong-id" else session
failed = behavior == name + "-failed"
if cc:
    print(json.dumps({"session_id": returned, "is_error": failed,
                      "subtype": "error" if failed else "success"}))
else:
    if behavior != name + "-no-id":
        print(json.dumps({"type": "thread.started", "thread_id": returned}))
    print(json.dumps({"type": "turn.failed" if failed else "turn.completed"}))
'''


@unittest.skipUnless(os.name == "posix", "Shared upstream Linux runner")
class CreatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="creator test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.group = self.root / "complete group.md"
        self.group.write_text("Fixture grounded actions and capabilities.\n")
        self.binary = self.root / "fake-agent"
        self.binary.write_text(FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable))
        self.binary.chmod(0o700)

    def run_creator(self, **kwargs):
        options = dict(action_group=self.group.name, env_dir="new environment",
                       workspace=self.root, claude_bin=str(self.binary), codex_bin=str(self.binary),
                       progress=lambda _: None)
        options.update(kwargs)
        return creator.run_creator(**options)

    def calls(self):
        return [json.loads(x) for x in (self.root / "calls.jsonl").read_text().splitlines()]


    def test_research_errors_never_deliver_build(self):
        for backend in ("codex", "cc"):
            for mode in ("exit", "missing", "empty", "failed"):
                with self.subTest(backend=backend, mode=mode), patch.dict(
                    os.environ, {"CREATOR_FIXTURE_BEHAVIOR": "research-" + mode}
                ):
                    result = self.run_creator(backend=backend)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(list(result["turns"]), ["research"])
                    self.assertFalse((Path(result["output_directory"]) / "build").exists())
        self.assertEqual(len(self.calls()), 8)
        self.assertTrue(all(not c["resume"] for c in self.calls()))

    def test_missing_session_does_not_guess_or_resume_last(self):
        with patch.dict(os.environ, {"CREATOR_FIXTURE_BEHAVIOR": "research-no-id"}):
            result = self.run_creator(backend="codex")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.calls()), 1)
        self.assertIsNone(result["session_id"])

    def test_build_failures_are_recorded_with_research_preserved(self):
        for backend in ("codex", "cc"):
            for mode in ("exit", "missing", "empty", "failed", "wrong-id"):
                with self.subTest(backend=backend, mode=mode), patch.dict(
                    os.environ, {"CREATOR_FIXTURE_BEHAVIOR": "build-" + mode}
                ):
                    result = self.run_creator(backend=backend)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["turns"]["research"]["status"], "completed")
                    self.assertEqual(result["turns"]["build"]["status"], "failed")
                    self.assertNotIn("nudge", result["turns"])
                    self.assertEqual(result["session_id"], self.calls()[-2]["session"])
                    self.assertTrue(Path(result["turns"]["research"]["report"]).is_file())

    def test_nudge_failures_preserve_completed_build(self):
        for backend in ("codex", "cc"):
            for mode in ("exit", "empty", "failed", "wrong-id"):
                with self.subTest(backend=backend, mode=mode), patch.dict(
                    os.environ, {"CREATOR_FIXTURE_BEHAVIOR": "nudge-" + mode}
                ):
                    result = self.run_creator(backend=backend)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["turns"]["build"]["status"], "completed")
                    self.assertEqual(result["turns"]["nudge"]["status"], "failed")
                    output = Path(result["output_directory"])
                    self.assertTrue((output / "build/handoff.md").read_text().startswith("build:"))
                    self.assertEqual(result["session_id"], self.calls()[-3]["session"])

    def test_build_prompt_is_not_loaded_until_research_finishes(self):
        prompts = self.root / "prompts"
        prompts.mkdir()
        (prompts / "research_prompt.md").write_text((creator.PROMPTS / "research_prompt.md").read_text())
        result = self.run_creator(prompt_dir=prompts)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["turns"]["research"]["status"], "completed")
        self.assertIn("build_prompt.md", result["error"])
        self.assertEqual(len(self.calls()), 1)

    def test_nudge_prompt_is_not_loaded_until_build_finishes(self):
        prompts = self.root / "prompts"
        prompts.mkdir()
        for name in ("research", "build"):
            (prompts / f"{name}_prompt.md").write_text((creator.PROMPTS / f"{name}_prompt.md").read_text())
        result = self.run_creator(prompt_dir=prompts)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["turns"]["build"]["status"], "completed")
        self.assertEqual(result["turns"]["nudge"]["status"], "failed")
        self.assertIn("nudge_prompt.md", result["error"])
        self.assertEqual(len(self.calls()), 2)

    def test_nudge_prompt_text_is_exact(self):
        self.assertEqual((creator.PROMPTS / "nudge_prompt.md").read_text().strip(),
            "please look at the other environments such as whiteboard, titration, deposit_cans. "
            "this is extremely bad and a regression from the other environments, and need many "
            "improvements acorss multiple aspects")


    def test_invalid_input_and_existing_output_do_not_launch(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "preserve.md"
        marker.write_text("keep")
        for args in ({"backend": "unknown"}, {"timeout_sec": 0}, {"action_group": "missing"},
                     {"env_dir": "."}, {"env_dir": "../outside"}, {"env_dir": self.group},
                     {"output": existing}):
            with self.subTest(args=args), self.assertRaises((ValueError, FileExistsError)):
                self.run_creator(**args)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse((self.root / "calls.jsonl").exists())


class CreatorCliTests(unittest.TestCase):
    def test_help_does_not_launch(self):
        out = io.StringIO()
        with patch.object(creator, "run_creator") as launch, contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as done:
                cli.main(["create", "--help"])
            self.assertEqual(done.exception.code, 0)
            launch.assert_not_called()
        self.assertIn("--action-group", out.getvalue())
        self.assertIn("--env-dir", out.getvalue())

    def test_cli_text_json_and_failure_exit(self):
        result = {"status": "turns_completed", "env_dir": "/tmp/env", "session_id": "test-session",
                  "output_directory": "/tmp/run", "turns": {
                      "research": {"status": "completed", "report": "/tmp/run/research.md"},
                      "build": {"status": "completed", "report": "/tmp/run/handoff.md"},
                      "nudge": {"status": "completed", "report": "/tmp/run/handoff.md"}}}
        for fmt in ("text", "json"):
            out = io.StringIO()
            with patch.object(creator, "from_args", return_value=result), contextlib.redirect_stdout(out):
                code = cli.main(["create", "--action-group", "group.md", "--env-dir", "env", "--format", fmt])
            self.assertEqual(code, 0)
            if fmt == "json":
                self.assertEqual(json.loads(out.getvalue()), result)
            else:
                self.assertIn("/tmp/run/handoff.md", out.getvalue())
                self.assertIn("independent audits have not run", out.getvalue())
        result.update(status="failed", error="fixture failure")
        with patch.object(creator, "from_args", return_value=result), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["create", "--action-group", "group.md", "--env-dir", "env"]), 1)


if __name__ == "__main__":
    unittest.main()

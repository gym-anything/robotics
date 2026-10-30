"""Default model routing with executable CLI fixtures; no paid model calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from gym_anything_robotics.creation_audit import creator, method
from tests.test_creator import FAKE as CREATOR_FAKE
from tests.test_creation_audit import FAKE as AUDIT_FAKE


class DefaultsTests(unittest.TestCase):
    def test_cli_backends_and_model_resolution(self):
        for module in (creator, method):
            parser = argparse.ArgumentParser()
            module.add_arguments(parser)
            args = ["--env-dir", "env"]
            if module is creator:
                args += ["--action-group", "group.md"]
            parsed = parser.parse_args(args)
            self.assertEqual(parsed.backend, "cc")
            self.assertEqual(method._model_for(parsed.backend, parsed.model), "claude-opus-5-5")
            explicit = parser.parse_args(args + ["--backend", "codex"])
            self.assertIsNone(method._model_for(explicit.backend, explicit.model))
            self.assertEqual(method._model_for("cc", "explicit"), "explicit")
            self.assertEqual(method._model_for("codex", "gpt-6-sol"), "gpt-6-sol")

    @unittest.skipUnless(os.name == "posix", "Linux coding-agent runner")
    def test_real_fixture_invocations_pin_opus_and_record_it(self):
        for kind, fixture in (("creator", CREATOR_FAKE), ("audit", AUDIT_FAKE)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                binary = root / "agent"
                binary.write_text(fixture.replace("#!/usr/bin/env python3", "#!" + sys.executable))
                binary.chmod(0o700)
                (root / "env").mkdir()
                (root / "group.md").write_text("fixture actions")
                kwargs = dict(env_dir="env", workspace=root, claude_bin=str(binary),
                              progress=lambda _: None)
                if kind == "creator":
                    result = creator.run_creator(action_group="group.md", **kwargs)
                    self.assertEqual(result["status"], "turns_completed", result)
                    expected_calls = 3
                else:
                    result = method.run_audits(auditors=("qualitative", "evidence", "infrastructure"), **kwargs)
                    self.assertEqual(result["status"], "reports_written", result)
                    expected_calls = 6
                self.assertEqual(result["backend"], "cc")
                self.assertEqual(result["model"], "claude-opus-5-5")
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(len(calls), expected_calls)
                self.assertTrue(all(call["args"][-2:] == ["--model", "claude-opus-5-5"] for call in calls))


if __name__ == "__main__":
    unittest.main()

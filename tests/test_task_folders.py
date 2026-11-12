"""Every RoboWorld benchmark task has its folder: <environment package>/tasks/<task>/ with task.json, setup.py and
verifier.py (gym_anything_robotics.task_folders), and its task.json says what resolving its selection gives."""

import json
import unittest
from collections import Counter
from pathlib import Path

from gym_anything_robotics import task_folders, tasks

SPLITS = Path(__file__).resolve().parents[1] / "benchmarks" / "roboworld" / "splits"


def split_rows():
    for split in ("train", "test"):
        for row in json.loads((SPLITS / f"{split}.json").read_text())["tasks"]:
            yield split, row


class TaskFolderTests(unittest.TestCase):
    def test_each_split_entry_has_one_folder_that_matches_its_resolve(self):
        seen = Counter()
        for split, row in split_rows():
            env, selection = row["environment_id"], row["selection"]
            package = tasks.package(env)
            with self.subTest(env=env, selection=selection):
                matches = [(name, card) for name, card in tasks.folders(env).items() if card["selection"] == selection]
                self.assertEqual(len(matches), 1, "one task folder per split entry")
                name, card = matches[0]
                seen[package, name] += 1
                spec = tasks.resolve(env, selection)
                self.assertEqual(card, {
                    "schema": task_folders.SCHEMA, "id": spec.task_id, "name": name, "env_id": env, "split": split,
                    "selection": selection, "description": spec.description,
                    "hooks": {"setup": "setup.py::setup"},
                    "success": {"mode": "program", "spec": {"program": "verifier.py::verify"}}})
                self.assertTrue(callable(task_folders.entry(package, name, "setup")))
                self.assertTrue(callable(task_folders.entry(package, name, "success")))
        self.assertEqual(max(seen.values()), 1)

    def test_each_environment_has_its_eight_folders_and_nothing_else(self):
        for env in tasks.CATALOGS:
            package = tasks.package(env)
            with self.subTest(env=env):
                names = task_folders.names(package)
                self.assertEqual(len(names), 8)
                for name in names:
                    files = sorted(p.name for p in task_folders.folder(package, name).iterdir()
                                   if p.name != "__pycache__")
                    self.assertEqual(files, sorted(task_folders.FILES))
                    self.assertEqual(task_folders.load(package, name)["name"], name)


if __name__ == "__main__":
    unittest.main()

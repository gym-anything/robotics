"""Drilling task catalogue: seed resolution and the frozen train/validation/test splits (pure Python, no engine).

The split values are frozen: seeds 0..999 of every task are assigned by SHA-256 of (task, seed), 80/10/10, as published
in drilling/benchmark.json. The per-task counts and the digest of the whole assignment below were computed from
drilling/variants.split_of_seed; a change to the hash, the task names or the thresholds fails here."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from tests.packages import PACKAGES

ROOT = PACKAGES

FROZEN_COUNTS = {  # task: (train, validation, test) over seeds 0..999
    "anchor-holes": (789, 112, 99), "rebar-relocate": (805, 89, 106), "clean-and-set": (804, 107, 89),
    "ledger-anchor": (829, 81, 90), "take-drill": (798, 105, 97), "punch-layout": (797, 112, 91),
    "press-drill": (792, 97, 111), "press-blind": (827, 96, 77), "match-bolt": (797, 114, 89),
    "panel-holes": (802, 99, 99), "timber-bolt": (814, 101, 85)}
FROZEN_DIGEST = "bfac18c214c5c384339b131fa9a46fabfd8b29a56f7034a7d0308fa70fa06c13"
WALL = ("anchor-holes", "rebar-relocate", "clean-and-set", "ledger-anchor")
BENCH = ("take-drill", "punch-layout", "press-drill", "press-blind", "match-bolt", "panel-holes", "timber-bolt")


class DrillingCatalogTest(unittest.TestCase):
    def test_benchmark_json_matches_the_code(self) -> None:
        from drilling.variants import PUBLIC
        cat = json.loads((ROOT/"drilling/benchmark.json").read_text())
        self.assertEqual(cat["environment_id"], "drilling")
        self.assertEqual(tuple(PUBLIC["wall"]), WALL)
        self.assertEqual(tuple(PUBLIC["bench"]), BENCH)
        self.assertEqual(cat["count"], len(WALL) + len(BENCH))
        self.assertEqual(sorted(cat["selector_choices"]["task"]), sorted(WALL + BENCH))
        self.assertEqual(sorted(cat["selector_choices"]["preset"]), ["bench", "wall"])
        self.assertEqual(cat["splits"], ["train", "validation", "test"])
        self.assertEqual(cat["resolver"], "drilling.catalogue:task_resolve")
        self.assertIn("drilling/REPORT.pdf", cat["qualification"])
        self.assertNotIn("REPORT.md", cat["qualification"])

    def test_default_and_task_only_selection(self) -> None:
        import gym_anything_robotics as gar
        cat = json.loads((ROOT/"drilling/benchmark.json").read_text())
        spec = gar.resolve_task("drilling")
        self.assertEqual(spec.task_id, cat["default_task_id"])
        self.assertEqual(spec.task_id, "drilling/v1/wall/anchor-holes/seed-0")
        for preset, tasks in (("wall", WALL), ("bench", BENCH)):
            for task in tasks:                         # task names are unique: a task alone selects its preset
                spec = gar.resolve_task("drilling", {"task": task, "seed": 17})
                self.assertEqual(spec.task_id, f"drilling/v1/{preset}/{task}/seed-17")
                self.assertEqual(spec.launch.backend_options, {"preset": preset, "task": task, "seed": 17})
                self.assertEqual(spec.selection, {"preset": preset, "task": task, "seed": 17})
        spec = gar.resolve_task("drilling", {"preset": "bench"})
        self.assertEqual(spec.task_id, "drilling/v1/bench/take-drill/seed-0")

    def test_seed_and_selector_refusals(self) -> None:
        import gym_anything_robotics as gar
        from drilling.variants import resolve
        for seed in (-1, 10000, "3", 2.0, True, None):
            with self.assertRaises(ValueError, msg=repr(seed)):
                resolve("wall", "anchor-holes", seed)
        self.assertEqual(resolve("wall", "anchor-holes", 9999)["seed"], 9999)
        with self.assertRaises(ValueError):
            resolve("bench", "anchor-holes", 0)          # a wall task under the bench preset
        with self.assertRaises(ValueError):
            resolve("probe", None, 0)                    # the probe preset is not public
        with self.assertRaises(ValueError):
            resolve(None, "no-such-task", 0)
        with self.assertRaises(ValueError):
            gar.resolve_task("drilling", {"robot": "g1"})  # not a task selector

    def test_frozen_splits(self) -> None:
        from drilling.variants import split_of_seed
        h = hashlib.sha256()
        for task in BENCH + WALL:                        # the order the digest was computed in (bench, then wall)
            c = Counter(split_of_seed(task, s) for s in range(1000))
            self.assertEqual((c["train"], c["validation"], c["test"]), FROZEN_COUNTS[task], task)
            self.assertEqual(sum(c.values()), 1000)
            for s in range(1000):
                h.update(f"{task}/{s}={split_of_seed(task, s)};".encode())
        self.assertEqual(h.hexdigest(), FROZEN_DIGEST)
        for task in WALL + BENCH:
            self.assertIsNone(split_of_seed(task, 1000))
            self.assertIsNone(split_of_seed(task, 9999))

    def test_split_reaches_the_task_spec(self) -> None:
        import gym_anything_robotics as gar
        self.assertEqual(gar.resolve_task("drilling", {"task": "press-blind", "seed": 17}).split, "train")
        self.assertEqual(gar.resolve_task("drilling", {"task": "anchor-holes", "seed": 4}).split, "test")
        self.assertIsNone(gar.resolve_task("drilling", {"task": "anchor-holes", "seed": 1234}).split)


if __name__ == "__main__":
    unittest.main()

"""The benchmark splits: every entry resolves to a task of its environment, and no task is in both splits."""
import json

import pytest

import gym_anything_robotics as gar
from tests.packages import REPO

SPLITS = REPO / "benchmarks" / "roboworld" / "splits"


def load(name):
    split = json.loads((SPLITS / f"{name}.json").read_text())
    assert split["split"] == name
    return split["tasks"]


@pytest.mark.parametrize("name", ["train", "test"])
def test_split_resolves(name):
    for entry in load(name):
        task = gar.resolve_task(entry["environment_id"], entry["selection"])
        assert task.launch.environment_id == entry["environment_id"]


def test_test_split_has_one_task_per_simulation():
    environments = [t["environment_id"] for t in load("test")]
    assert len(environments) == len(set(environments)) == 25


def test_splits_are_disjoint():
    task_ids = {name: {gar.resolve_task(t["environment_id"], t["selection"]).task_id for t in load(name)}
                for name in ("train", "test")}
    assert not task_ids["train"] & task_ids["test"]

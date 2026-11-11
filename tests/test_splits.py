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


def test_each_simulation_has_its_eight_tasks_in_the_splits():
    # Seven train tasks and the test task per simulation, eight different tasks once the contract part of the task
    # id is set aside (a v1 task and its v2 version are the same task).
    import re
    from collections import defaultdict
    tasks = defaultdict(list)
    for name in ("train", "test"):
        for t in load(name):
            task_id = gar.resolve_task(t["environment_id"], t["selection"]).task_id
            tasks[t["environment_id"], name].append(
                "/".join(s for s in task_id.split("/") if not re.fullmatch(r"(contract-)?[a-z-]+-tasks-v[0-9]+", s)))
    environments = {e for e, _ in tasks}
    assert len(environments) == 25
    for e in environments:
        assert len(tasks[e, "train"]) == 7 and len(tasks[e, "test"]) == 1, e
        assert len(set(tasks[e, "train"] + tasks[e, "test"])) == 8, e

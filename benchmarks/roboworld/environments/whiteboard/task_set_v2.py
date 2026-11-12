"""whiteboard-tasks-v2: the render-checked eight-task set, selected with `"contract": "whiteboard-tasks-v2"`.

The render check ran the released set's seven named presets and the native bridge (variation_id 0)
with the same authored starts, instructions, verifiers, progress and score as whiteboard-tasks-v1 (task_set/). What
differs, and is all this module adds:
- the contract, in the task ids, the backend options, every evaluation and the trace;
- draw-table reports its contract unsuffixed, as every task does (v1 reports "<contract>/draw-table"); the table
  metric keeps its own revision label, now "whiteboard-tasks-v2/draw-table" (task_set_v2 in its verifier.py,
  tasks/draw-table/);
- task_source_sha256 (task_source_identity) also covers this module and its backend.
The bridge keeps its native selection (splits/test.json) and contract. The task code is task_set/'s and the task
folders' (tasks/<name>/), reused unchanged; this module lives outside task_set/ so v1's task_source_sha256 does not
cover it. Backend: task_set_v2_backend.py.
"""
from dataclasses import dataclass
import hashlib
import json

from whiteboard.task_set import catalogue as V1
from whiteboard.task_set.catalogue import CatalogueTask, DIFFICULTY, INSTRUCTIONS, NAMES
from whiteboard.tasks import task_verifier

CONTRACT = "whiteboard-tasks-v2"
SELECTIONS = tuple({"task": name, "contract": CONTRACT} for name in NAMES[:-1]) + ({"variation_id": 0},)


@dataclass(frozen=True)
class TaskV2(CatalogueTask):
    """A released task-set preset under whiteboard-tasks-v2: same name, strokes, placement and instruction."""
    contract: str = CONTRACT

    def __post_init__(self):
        if self.name not in INSTRUCTIONS or self.contract != CONTRACT:
            raise ValueError("Unknown whiteboard task/contract")
        if self.kind != "task_set" or (self.center_x, self.center_z, self.width_m) != (-.05, 1.28, .70):
            raise ValueError("These task presets have fixed authored placements")


def preset_config(name, base=None):
    config, _ = V1.preset_config(name, base)        # the released board check and layout
    return config, TaskV2(name)


def resolve(selection):
    if set(selection) - {"task", "contract"}:
        raise ValueError("Named presets do not combine with native variation selectors")
    if selection.get("contract") != CONTRACT:
        raise ValueError(f"Named tasks require explicit contract {CONTRACT}")
    name = selection.get("task")
    if name == "bridge":
        raise ValueError("Bridge keeps its native contract; select variation_id 0")
    task = TaskV2(name)
    return dict(task_id=task.task_id, description=task.description,
                backend_options={"catalogue_task": name, "task_contract": CONTRACT},
                selection={"task": name, "contract": CONTRACT, "difficulty": DIFFICULTY[name],
                           "start": "constructed task props, no simulated preparation"}, split=None)


def evaluate(task, ink, config, state=None):
    """task_set.verifier.evaluate, with draw-table's contract unsuffixed and its metric labelled for v2 (the task's
    task_set_v2, its verifier.py)."""
    from whiteboard.task_set.verifier import evaluate as released
    answer = released(task, ink, config, state)
    own = getattr(task_verifier(task.name), "task_set_v2", None)
    if own is not None:
        own(answer, task)
    return answer


def source_identity():
    """Task-code identity reported by every v2 evaluation: what v1's covers, plus this module and its backend."""
    from gym_anything_robotics.paths import package_dir
    board, core = package_dir("whiteboard"), package_dir("gym_anything_robotics")
    paths = {f"whiteboard/task_set/{p.name}": p for p in (board / "task_set").glob("*.py")}
    paths.update(V1.task_files())
    paths.update({f"whiteboard/{n}": board / n for n in ("task_set_v2.py", "task_set_v2_backend.py")})
    paths.update({"gym_anything_robotics/backends/newton_whiteboard_task_set.py": core / "backends/newton_whiteboard_task_set.py",
                  "gym_anything_robotics/launch.py": core / "launch.py", "whiteboard/benchmark.json": board / "benchmark.json"})
    files = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sorted(paths.items())}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}

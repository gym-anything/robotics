"""Frozen 10,000-scene catalogue: geometry/task variation, never a solving policy.

Hash-derived independent choices make each ID stable without enumerating prior
IDs. Rejection never silently relaxes requested constraints. Product dimensions,
gravity, gains, masses and friction are unchanged across this catalogue.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .config import COLORS, Config, SceneLayout
from .tasks import Task

VERSION = "whiteboard-classroom-v1"
COUNT = 10000
TEXTS = ("HELLO", "ROBOTICS", "SCIENCE", "NEWTON", "CLASS 1", "A+B", "WELCOME", "12345")
# Bridge placement: a kinematic check of the pinned G1 puts the highest nib point a Dex3 hand can hold on the
# board at 1.50-1.55 m. Bridge drawings keep their top at or below this height and their bottom on the writing
# surface; the same hash draw is mapped into that range.
BRIDGE_TOP_Z_M = 1.48


def _integer(index, field):
    return int.from_bytes(hashlib.sha256(f"{VERSION}/{index}/{field}".encode()).digest()[:8], "big")


def specification(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < COUNT:
        raise ValueError(f"variation_id must be an integer in [0,{COUNT - 1}]")

    def between(field, lo, hi):
        return lo + (hi - lo) * (_integer(index, field) / (2**64 - 1))

    kind = "bridge" if _integer(index, "kind") % 3 == 0 else "text"
    split_number = _integer(index, "split") % 10
    split = "train" if split_number < 8 else "validation" if split_number == 8 else "test"
    layout = SceneLayout(
        marker_order=tuple(sorted(COLORS, key=lambda c: _integer(index, "color/" + c))),
        marker_offset_x=between("tray", -0.08, 0.08),
        eraser_x=between("eraser", 0.40, 0.90),
        furniture_offset_x=between("desks", -0.06, 0.06),
        robot_x=between("robot_x", -0.75, -0.25),
        robot_y=between("robot_y", 1.08, 1.24),
        robot_yaw_rad=-math.pi / 2 + between("yaw", -0.08, 0.08),
    )
    width = between("task_width", 0.48, 0.80)
    task = Task(
        kind=kind,
        text=TEXTS[_integer(index, "text") % len(TEXTS)],
        center_x=between("task_x", -0.35, 0.35),
        center_z=(between("task_z", 0.955 + 0.30 * width, BRIDGE_TOP_Z_M - 0.27 * width) if kind == "bridge"
                  else between("task_z", 1.35, 1.65)),
        width_m=width,
    )
    task.validate_board(Config().board)
    data = {
        "catalogue": VERSION,
        "variation_id": index,
        "split": split,
        "seed": _integer(index, "seed") % (2**31),
        "layout": asdict(layout),
        "task": asdict(task),
    }
    data["sha256"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return data


def select(*, variation_id=None, split=None, kind=None, selection_seed=0):
    if split not in (None, "train", "validation", "test") or kind not in (None, "text", "bridge"):
        raise ValueError("Unknown split or task kind")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection_seed must be an integer")
    if variation_id is not None:
        item = specification(variation_id)
        if (split and item["split"] != split) or (kind and item["task"]["kind"] != kind):
            raise ValueError("Requested variation does not satisfy the constraints")
        return item
    candidates = [
        i
        for i in range(COUNT)
        if (not split or specification(i)["split"] == split)
        and (not kind or specification(i)["task"]["kind"] == kind)
    ]
    if not candidates:
        raise ValueError("No matching variation; constraints were not relaxed")
    return specification(candidates[_integer(selection_seed, "selection") % len(candidates)])


def resolve(index, *, base=None):
    data = specification(index)
    layout = dict(data["layout"])
    layout["marker_order"] = tuple(layout["marker_order"])
    return replace(base or Config(), seed=data["seed"], layout=SceneLayout(**layout)), Task(
        **data["task"]
    )


def export(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [specification(i) for i in range(COUNT)]
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    path.write_bytes(payload)
    manifest = {
        "catalogue": VERSION,
        "count": COUNT,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "splits": {
            s: sum(row["split"] == s for row in rows) for s in ("train", "validation", "test")
        },
        "dimensions_changed": False,
        "physics_parameters_changed": False,
        "solver_included": False,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output)))

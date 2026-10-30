"""Seeded task catalogue: preset/task/seed plus the glide and desk-contents variation selectors.

A seed varies what its task varies (how far and which way the desks drifted, where the chairs were left, where the
tote, wastebasket and loose chairs stand); dimensions, masses and materials never vary with the seed. `glides`
(nylon, felt, steel: the 785/9018 glide options) and `contents` (0-3 textbooks in each involved desk's book box) are
explicit selectors because they change the physics (the tip/slide boundary, the spill).
`contract` selects the revised eight-task grader only when explicitly requested;
it changes no start state or physical parameter. Bare selections remain legacy."""
import hashlib
import json
from pathlib import Path
from .layout import PRESETS, TASKS
from .config import GLIDES
from .catalogue import CONTRACT, LEGACY, TASK_SET, description, validate_contract

CATALOGUE = Path(__file__).resolve().parent/"variations.json"


def resolve(preset="room114", task=None, seed=0, glides="nylon", contents=0, contract=LEGACY):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset; choose {sorted(PRESETS)}")
    task = task or PRESETS[preset][0]
    if task not in PRESETS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PRESETS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    if glides not in GLIDES:
        raise ValueError(f"glides must be one of {GLIDES}")
    if type(contents) is not int or not 0 <= contents <= 3:
        raise ValueError("contents must be an integer 0..3")
    validate_contract(task, contract)
    return dict(preset=preset, task=task, seed=seed, glides=glides, contents=contents,
                **({"contract": contract} if contract != LEGACY else {}))


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed", "glides", "contents", "contract"], presets=sorted(PRESETS),
                tasks={k: list(v) for k, v in PRESETS.items()}, seed_range=[0, 9999], glides=list(GLIDES), contents=[0, 1, 2, 3],
                contracts=[LEGACY, CONTRACT], task_set=list(TASK_SET),
                catalogue="variations.json: seeds 0..999 of every task, frozen train/validation/test split by hash")


def task_resolve(selection):
    unknown = set(selection) - {"preset", "task", "seed", "glides", "contents", "contract"}
    if unknown:
        raise ValueError(f"Unknown selection fields: {sorted(unknown)}")
    v = resolve(selection.get("preset", "room114"), selection.get("task"), selection.get("seed", 0),
                selection.get("glides", "nylon"), selection.get("contents", 0), selection.get("contract", LEGACY))
    suffix = "" if (v["glides"], v["contents"]) == ("nylon", 0) else f"/{v['glides']}-c{v['contents']}"
    version = f"/{v['contract']}" if "contract" in v else ""
    return dict(task_id=f"classroom-arrangement/v1/{v['preset']}/{v['task']}{version}/seed-{v['seed']}{suffix}",
                description=description(v["task"], v.get("contract", LEGACY)), backend_options=dict(v), selection=v,
                split=split_of(v["task"], v["seed"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="classroom-arrangement.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])

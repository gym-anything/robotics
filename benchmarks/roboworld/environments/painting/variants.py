"""Seeded task catalogue: preset/task/seed. A seed varies what its task varies (paint and old wall colour, tray fill,
tool placement +-5 cm / +-10 deg, room climate 18-27 C and RH 30-70 % -- the drying rate -- and the spackle patches on
the wall); dimensions, masses, materials and paint products never vary with the seed."""
import hashlib
import json
from pathlib import Path
from .layout import PRESETS, TASKS
from .catalogue import CONTRACT, INSTRUCTIONS, selected_contract, describe, selections

CATALOGUE = Path(__file__).resolve().parent/"variations.json"


def resolve(preset=None, task=None, seed=0, contract=None):
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    if preset is None:
        preset = TASKS[task]["preset"] if task else "wall"
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset; choose {sorted(PRESETS)}")
    task = task or PRESETS[preset][0]
    if task not in PRESETS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PRESETS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    chosen = selected_contract(task, contract)
    return dict(preset=preset, task=task, seed=seed, **({"contract": chosen} if chosen == CONTRACT else {}))


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed", "contract"], presets=sorted(PRESETS),
                tasks={k: list(v) for k, v in PRESETS.items()}, seed_range=[0, 9999],
                contract_choices={t: [selected_contract(t)] + ([CONTRACT] if t in INSTRUCTIONS else []) for t in TASKS},
                task_set=selections(),
                catalogue="variations.json: seeds 0..999 of every task, frozen train/validation/test split by hash")


def task_resolve(selection):
    if set(selection) - {"preset", "task", "seed", "contract"}:
        raise ValueError("Selectors are preset, task, seed and contract")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    suffix = f"/{v['contract']}" if "contract" in v else ""
    return dict(task_id=f"painting/v1/{v['preset']}/{v['task']}{suffix}/seed-{v['seed']}",
                description=describe(v["task"], v["seed"], v.get("contract")),
                backend_options=dict(v), selection=v, split=split_of(v["task"], v["seed"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="painting.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])

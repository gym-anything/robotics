"""Seeded task catalogue: task / seed plus the soil variation selectors. A seed varies what its task varies (mark
position +-8 cm in the bed, the auger marks +-2 cm, the number and placement of embedded stones, the layer interfaces'
undulation); moisture (dry / moist / wet), topsoil depth (0.12-0.32 m) and the stone count are explicit selectors.
Tool dimensions, masses, soil targets and interface properties never vary with the seed."""
import hashlib
import json
from pathlib import Path

from .layout import TASKS, PRESETS
from . import params as P

CATALOGUE = Path(__file__).resolve().parent/"variations.json"
SELECTORS = ("task", "seed", "moisture", "topsoil", "stones")


def resolve(task=None, seed=0, moisture=None, topsoil=None, stones=None):
    task = task or "penetrate-marked"
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    moisture = moisture or "moist"
    if moisture not in P.MOISTURE:
        raise ValueError(f"moisture must be one of {sorted(P.MOISTURE)}")
    if topsoil is not None:
        if isinstance(topsoil, bool) or not 0.12 <= float(topsoil) <= 0.32:
            raise ValueError("topsoil must be 0.12-0.32 m")
        topsoil = round(float(topsoil), 3)
    if stones is not None and (type(stones) is not int or not 0 <= stones <= 6):
        raise ValueError("stones must be an integer 0-6")
    return dict(task=task, preset=TASKS[task]["preset"], seed=seed, moisture=moisture, topsoil=topsoil, stones=stones)


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=list(SELECTORS), tasks=list(TASKS), presets={k: [t for t, v in TASKS.items()
                if v["preset"] == k] for k in PRESETS}, seed_range=[0, 9999], moisture=sorted(P.MOISTURE),
                topsoil_m=[0.12, 0.32], stones=[0, 6],
                catalogue="variations.json: seeds 0..999 of every task (moist, preset topsoil, seeded stones), frozen "
                          "train/validation/test split by hash")


def task_resolve(selection):
    v = resolve(selection.get("task"), int(selection.get("seed", 0)), selection.get("moisture"), selection.get("topsoil"),
                selection.get("stones"))
    extra = "".join(f"/{k}-{v[k]}" for k in ("moisture", "topsoil", "stones") if v[k] is not None and not (k == "moisture" and v[k] == "moist"))
    return dict(task_id=f"soil-excavation/v1/{v['preset']}/{v['task']}/seed-{v['seed']}{extra}",
                description=TASKS[v["task"]]["instruction"], backend_options={k: v[k] for k in SELECTORS if v[k] is not None},
                selection=v, split=split_of(v["task"], v["seed"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="soil-excavation.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])

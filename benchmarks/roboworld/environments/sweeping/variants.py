"""Seeded task catalogue: preset/task/seed plus the variant, glide and amount selectors.

A seed varies what its task varies (where every piece lies and how much there is, by 20 % either way; how far the
chairs of the task's table are pulled out and turned; where the barrel stands; where the mug fell). Dimensions,
masses and materials never vary with the seed. `variant` changes which tools are out (push-broom-only, hand-pan,
one-arm), the chairs' glides (felt-glides, steel-glides) or lays fine dust over the task's floor (dusty); `glides` sets the glide type directly; `amount` scales
the debris of a task (0.25 to 1.5)."""
import hashlib
import json
from pathlib import Path
from .layout import PRESETS, TASKS, VARIANTS
from .config import GLIDES

CATALOGUE = Path(__file__).resolve().parent/"variations.json"
SELECTORS = ("preset", "task", "seed", "variant", "glides", "amount")


def resolve(preset="room112", task=None, seed=0, variant="default", glides="nylon", amount=1.0):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset; choose {sorted(PRESETS)}")
    task = task or PRESETS[preset][0]
    if task not in PRESETS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PRESETS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if glides not in GLIDES:
        raise ValueError(f"glides must be one of {GLIDES}")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not 0.25 <= amount <= 1.5:
        raise ValueError("amount must be a number from 0.25 to 1.5")
    if variant == "one-arm" and task != "table-scraps":
        raise ValueError("variant one-arm belongs to task table-scraps")
    if variant == "hand-pan" and task not in ("pan-and-barrel", "broken-mug"):
        raise ValueError("variant hand-pan belongs to tasks pan-and-barrel and broken-mug")
    if variant == "dusty" and task not in ("gather-open", "corner-and-edge", "under-table"):
        raise ValueError("variant dusty belongs to tasks gather-open, corner-and-edge and under-table")
    return dict(preset=preset, task=task, seed=seed, variant=variant, glides=glides, amount=float(amount))


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"sweeping/{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=list(SELECTORS), presets=sorted(PRESETS), tasks={k: list(v) for k, v in PRESETS.items()},
                seed_range=[0, 9999], variants=list(VARIANTS), glides=list(GLIDES), amount=[0.25, 1.5],
                catalogue="variations.json: seeds 0..999 of every task, frozen train/validation/test split by hash")


def task_resolve(selection):
    from .task import instruction          # the session's own text, variant sentence included
    v = resolve(selection.get("preset", "room112"), selection.get("task"), selection.get("seed", 0),
                selection.get("variant", "default"), selection.get("glides", "nylon"), selection.get("amount", 1.0))
    extra = [v["variant"]] if v["variant"] != "default" else []
    if v["glides"] != "nylon":
        extra.append(v["glides"])
    if v["amount"] != 1.0:
        extra.append(f"x{v['amount']:g}")
    suffix = "/" + "-".join(extra) if extra else ""
    return dict(task_id=f"sweeping/v1/{v['preset']}/{v['task']}/seed-{v['seed']}{suffix}",
                description=instruction(None, v), backend_options=dict(v), selection=v,
                split=split_of(v["task"], v["seed"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="sweeping.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])

"""Seeded task catalogue: preset/task/seed plus the variant selector.

A seed varies what a crew's work leaves different from one day to the next: where the struck pad's low corner and
high band are and the screed ridges' phase, the patch's depth and overfill, the sealant's gun stops, the knife ridges,
the crumbs on the block, and the site within a narrow band around the variant's conditions (concrete and air
temperature +-2 C, bleed rate +-20 %). Dimensions, materials and tools never vary with the seed. `variant` sets the
site's conditions: default (21 C, still air, bleed 0.15 kg/m2/h), door (the door open, 1 m/s at the slab), fan (a
pedestal fan left running on the slab, 2.5 m/s, RH 40 %), hot (27 C), cool (15 C), fast-sealant (tool-sealant only: a
fast-skinning sealant, skin in 20 min)."""
import hashlib
import json
from pathlib import Path
from .layout import PRESETS, TASKS, VARIANTS

CATALOGUE = Path(__file__).resolve().parent/"variations.json"
SELECTORS = ("preset", "task", "seed", "variant")
BASE = dict(default=(21.1, "still"), door=(21.1, "door"), fan=(21.1, "fan"), hot=(27.0, "still"), cool=(15.0, "still"),
            **{"fast-sealant": (21.1, "still")})


def _u(task, seed, key):
    """A uniform number in [0, 1) fixed by (task, seed, key)."""
    return int(hashlib.sha256(f"concrete-finishing/{task}/{seed}/{key}".encode()).hexdigest()[:8], 16)/2**32


def site(task, seed, variant="default"):
    """Site conditions of a (task, seed, variant): temp_c, air, bleed (kg/m2/h), sealant skin time (s or None)."""
    temp, air = BASE[variant]
    bleed = 0.15
    if seed:
        temp += 4.0*(_u(task, seed, "temp") - 0.5)
        bleed *= 0.8 + 0.4*_u(task, seed, "bleed")
    return dict(temp_c=round(temp, 2), air=air, bleed=round(bleed, 4), skin_s=20*60.0 if variant == "fast-sealant" else None)


def resolve(preset="b012", task=None, seed=0, variant="default"):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset; choose {sorted(PRESETS)}")
    task = task or PRESETS[preset][0]
    if task not in PRESETS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PRESETS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "fast-sealant" and task != "tool-sealant":
        raise ValueError("variant fast-sealant belongs to task tool-sealant")
    if variant in ("door", "fan") and not (task.startswith("pad") or task == "patch-flush"):
        raise ValueError(f"variant {variant} (air over a fresh slab) belongs to the pad tasks and patch-flush")
    return dict(preset=preset, task=task, seed=seed, variant=variant)


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"concrete-finishing/{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=list(SELECTORS), presets=sorted(PRESETS), tasks={k: list(v) for k, v in PRESETS.items()},
                seed_range=[0, 9999], variants=list(VARIANTS),
                catalogue="variations.json: seeds 0..999 of every task, frozen train/validation/test split by hash")


def task_resolve(selection):
    from .task import instruction
    v = resolve(selection.get("preset", "b012"), selection.get("task"), selection.get("seed", 0), selection.get("variant", "default"))
    suffix = f"/{v['variant']}" if v["variant"] != "default" else ""
    return dict(task_id=f"concrete-finishing/v1/{v['preset']}/{v['task']}/seed-{v['seed']}{suffix}",
                description=instruction(v), backend_options=dict(v), selection=v, split=split_of(v["task"], v["seed"]),
                site=site(v["task"], v["seed"], v["variant"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="concrete-finishing.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])

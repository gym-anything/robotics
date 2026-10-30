"""Seeded task catalogue: task/seed. A seed varies what its task varies (supply pressure 40-80 psi and cold-water
temperature, detergent dose 0.5-1.25 g/L, wash temperature 43-49 C, foam patches per item, item placement +-2 cm and
yaw, tomato-paste load +-30 %, produce dust load, can position); dimensions, masses, materials and fixtures never vary."""
import hashlib
import json
from pathlib import Path
from .layout import PRESETS, TASKS

CATALOGUE = Path(__file__).resolve().parent/"variations.json"
ENV = "surface-rinsing"


def resolve(preset=None, task=None, seed=0):
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose {sorted(PRESETS)}")
    if task is None:
        task = PRESETS[preset or "pot-sink"][0]
    if preset is None:
        preset = TASKS[task]["preset"]
    if task not in PRESETS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PRESETS[preset]}")
    if isinstance(seed, bool) or type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=sorted(PRESETS),
                tasks={k: list(v) for k, v in PRESETS.items()}, seed_range=[0, 9999],
                catalogue="variations.json: seeds 0..999 of every task, frozen train/validation/test split by hash")


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"{ENV}/v1/{v['preset']}/{v['task']}/seed-{v['seed']}", description=TASKS[v["task"]]["instruction"],
                backend_options=dict(v), selection=v, split=split_of(v["task"], v["seed"]))


def build_catalogue():
    rows = [dict(task=t, seed=s, split=split_of(t, s)) for t in TASKS for s in range(1000)]
    blob = json.dumps(rows, sort_keys=True).encode()
    data = dict(schema="surface_rinsing.variations.v1", count=len(rows),
                splits={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")},
                sha256=hashlib.sha256(blob).hexdigest()[:16], rows=rows)
    CATALOGUE.write_text(json.dumps(data) + "\n")
    return data


def build_benchmark():
    from .layout import STATIONS
    data = {
        "schema": "gar.task-catalog.v1",
        "environment_id": ENV,
        "default_task_id": f"{ENV}/v1/pot-sink/rinse-glassware/seed-0",
        "description": ("Rinse surfaces in the wet back-of-house of a full-service restaurant (7.2 x 5.4 m dish room: an "
                        "Advance Tabco FS-3-1824-24RL three-compartment pot sink with a pre-rinse unit and swing faucet, a "
                        "soiled dish table with a 20 x 20 in pre-rinse sink, a prep sink and work table, a can-wash bay "
                        "with a sloped floor drain and a hose reel) with a Franka Mobile FR3 Duo. Water is physical state "
                        "from the supply to the drain: valves open only because a lever or wrist blade is moved by "
                        "contact; jets are momentum-carrying packets that splatter (Bhunia-Lienhard); inertial thin films "
                        "drain, round rims or separate (Duez), drip; pools fill, drain, overflow and float things; "
                        "surfactant (with adsorption and foam), soil (soaking, impingement, Wilson cleaning front, shear) "
                        "and contamination tracer are carried by the water; soap on a glass lowers the finger-pad "
                        "friction."),
        "selectors": ["preset", "task", "seed"],
        "cli_task_selector": "task",
        "selector_choices": {"preset": list(STATIONS), "task": list(TASKS)},
        "resolver": "surface_rinsing.variants:task_resolve",
        "count": len(TASKS),
        "count_scope": (f"{len(TASKS)} named tasks in four presets: " +
                        "; ".join(f"{st} ({', '.join(PRESETS[st])})" for st in STATIONS) +
                        "; seeds vary supply pressure and temperature, detergent dose, foam patches, soil loads and placement"),
        "splits": ["train", "validation", "test"],
        "split_note": "Frozen by hash of (task, seed) over seeds 0..999 of every task (surface_rinsing/variations.json); "
                      "seeds 1000..9999 resolve with split null.",
        "evaluator": {"available": True, "source": "surface_rinsing/task.py",
                      "scope": ("World state only, with a settle rule (30 s of the whole world advanced with the robot "
                                "holding, then measured and restored): surfactant per item and per 1 cm gauge, foam, "
                                "sanitizer immersion time, soil per 1 cm gauge against the 1 ug/cm2 visible-residue limit, "
                                "pooled water per item, placement in racks/colander/bowl, valves released and parked, soil, "
                                "tracer and water on clean surfaces and the floor, glass/china impact speeds; no "
                                "autonomous-solution claim.")},
        "qualification": "Component-qualified with assisted evidence: water benches without the "
                         "robot, phantom-actuated valves and phantom-held items, and robot demonstrations by a scripted "
                         "operator through the robot interface; phantom forces where labelled.",
    }
    p = Path(__file__).resolve().parent/"benchmark.json"
    p.write_text(json.dumps(data, indent=1) + "\n")
    return data


if __name__ == "__main__":
    d = build_catalogue()
    print(d["count"], d["splits"], d["sha256"])
    print(build_benchmark()["count"])

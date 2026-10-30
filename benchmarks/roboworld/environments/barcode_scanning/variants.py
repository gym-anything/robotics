"""Seeded task catalogue: preset/task/seed. Seeds vary the counted slot, stock mix, unit poses and pack contents."""
from .config import TASKS


def resolve(preset="stockroom", task=None, seed=0):
    if preset not in TASKS:
        raise ValueError(f"Unknown preset; choose {sorted(TASKS)}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=sorted(TASKS),
                tasks={k: list(v) for k, v in TASKS.items()}, seed_range=[0, 9999],
                qualification="Eight named tasks over three presets; seeds vary the counted slot (5 reachable slots), "
                              "stock mix and unit yaw, basket or counter contents and where they lie, returned books, "
                              "pack inner-unit azimuths",
                catalogue="variations.json: seeds 0..999 of every task (8 000), frozen train/validation/test split "
                          "(80/10/10 by hash); seeds 1000..9999 resolve with split null")


def task_resolve(selection):
    from .task import INSTRUCTIONS
    from .config import Config
    from .stock import target_slot
    v = resolve(selection.get("preset", "stockroom"), selection.get("task"), selection.get("seed", 0))
    slot = target_slot(Config(**v)) if v["preset"] == "stockroom" else None      # a pure function of the selection
    return dict(task_id=f"barcode-scanning/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]].format(slot=slot or ""),
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=split_of_seed(v["task"], v["seed"]))


def split_of_seed(task, seed):
    from .variations import lookup
    return lookup(task, seed)

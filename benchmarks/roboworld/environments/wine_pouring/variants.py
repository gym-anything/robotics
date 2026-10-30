"""Seeded task catalogue: preset/task/seed. Seeds vary the delivery spot, bottle fill, wine, leftover volumes and
+-15 mm / +-20 deg placement (layout.py). Splits: frozen by hash of (task, seed) over seeds 0..999 (80/10/10)."""
import hashlib

from .config import TASKS
from .contracts import (LEGACY_REWARD_CONTRACT, SPLIT_REWARD_CONTRACT, TASKS_V1_REWARD_CONTRACT,
                        PROGRESS_TASKS, eight_task_selections, reward_contract)


def resolve(preset=None, task=None, seed=0, contract=None):
    """A catalogue variation. Task names are unique across presets, so a task alone selects its preset
    (`gar run wine-pouring --task lined-measure`); neither given: service/pour-glass."""
    if preset is None:
        preset = next((k for k, v in TASKS.items() if task in v), "service") if task else "service"
    if preset not in TASKS:
        raise ValueError(f"Unknown preset; choose {sorted(TASKS)}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    selected = reward_contract(task, "" if contract is None else contract)
    return dict(preset=preset, task=task, seed=seed, **({"contract": selected} if contract is not None else {}))


def config_values(selection):
    """Translate public selectors to Config fields, clearing a previous contract on a new selection."""
    return {k: v for k, v in selection.items() if k != "contract"} | {
        "task_contract": selection.get("contract", "")}


def split_of_seed(task, seed):
    if seed >= 1000:
        return None
    h = int.from_bytes(hashlib.sha256(f"wine-pouring/{task}/{seed}".encode()).digest()[:4], "little") % 100
    return "test" if h < 10 else ("validation" if h < 20 else "train")


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed", "contract"], presets=sorted(TASKS),
                tasks={k: list(v) for k, v in TASKS.items()}, seed_range=[0, 9999],
                reward_contracts={t: ([SPLIT_REWARD_CONTRACT] if t == "split-bottle" else
                                      [LEGACY_REWARD_CONTRACT, TASKS_V1_REWARD_CONTRACT] if t in PROGRESS_TASKS else
                                      [LEGACY_REWARD_CONTRACT]) for names in TASKS.values() for t in names},
                task_set=eight_task_selections(),
                qualification="Ten named tasks over three presets (service, tasting, lined); seeds vary delivery spot, "
                              "bottle fill, wine (tasting), leftover volumes and placement jitter",
                catalogue="seeds 0..999 of every task split 80/10/10 train/validation/test by SHA-256 of (task, seed); "
                          "seeds 1000..9999 resolve with split null")


def task_resolve(selection):
    from .contracts import TaskConfig
    from .task_contracts import instruction
    if set(selection) - {"preset", "task", "seed", "contract"}:
        raise ValueError("Selectors are preset, task, seed and contract")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    return dict(task_id=f"wine-pouring/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=instruction(TaskConfig(**config_values(v))),
                backend_options=dict(v),
                selection=v, split=split_of_seed(v["task"], v["seed"]))

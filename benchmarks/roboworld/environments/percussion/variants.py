"""Seeded task catalogue: preset/task/seed (layout.py). Seeds vary drum tension (Wagner's low/mid/high), instrument and
drum placement (+-20 mm / +-30 mm, stand yaw +-30 deg), the client's cadence (80-120 BPM), the tune, the bordun key,
the bar for single-tone/strike-and-stop, the echo rhythm and the cue time, and the room's illuminance (300-500 lx).
Splits: frozen by SHA-256 of (task, seed) over seeds 0..999 (80/10/10)."""
import hashlib

from .config import TASKS


def resolve(preset=None, task=None, seed=0):
    """A catalogue variation. Task names are unique across presets, so a task alone selects its preset
    (`gar run percussion --task steady-beat`); neither given: therapy/melody."""
    if preset is None:
        preset = next((k for k, v in TASKS.items() if task in v), "therapy") if task else "therapy"
    if preset not in TASKS:
        raise ValueError(f"Unknown preset; choose {sorted(TASKS)}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


def split_of_seed(task, seed):
    if seed >= 1000:
        return None
    h = int.from_bytes(hashlib.sha256(f"percussion/{task}/{seed}".encode()).digest()[:4], "little") % 100
    return "test" if h < 10 else ("validation" if h < 20 else "train")


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=sorted(TASKS),
                tasks={k: list(v) for k, v in TASKS.items()}, seed_range=[0, 9999],
                qualification="Twelve named tasks over four presets (therapy, gait, assessment, pentatonic); seeds vary "
                              "drum tension, placements, cadence, tune, bordun key, bar, echo rhythm, cue time, lighting",
                catalogue="seeds 0..999 of every task split 80/10/10 train/validation/test by SHA-256 of (task, seed); "
                          "seeds 1000..9999 resolve with split null")


def task_resolve(selection):
    from .config import Config
    from .task import instruction
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"percussion/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=instruction(Config(**v)),
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=split_of_seed(v["task"], v["seed"]))

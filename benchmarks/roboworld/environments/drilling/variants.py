"""Seeded task catalogue: preset/task/seed. Seeds vary the mark positions within the station,
the rebar mat offset and aggregate packing, the hammer's mode-knob start (ledger-anchor) and small placements
(layout.py). Splits: frozen by SHA-256 of (task, seed) over seeds 0..999 (80/10/10)."""
import hashlib

from .config import TASKS

PUBLIC = {k: v for k, v in TASKS.items() if k != "probe"}


def resolve(preset=None, task=None, seed=0):
    """A catalogue variation. Task names are unique across presets, so a task alone selects its preset; neither
    given: wall/anchor-holes."""
    if preset is None:
        preset = next((k for k, v in PUBLIC.items() if task in v), "wall") if task else "wall"
    if preset not in PUBLIC:
        raise ValueError(f"Unknown preset; choose {sorted(PUBLIC)}")
    task = task or PUBLIC[preset][0]
    if task not in PUBLIC[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {PUBLIC[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


def split_of_seed(task, seed):
    if seed >= 1000:
        return None
    h = int.from_bytes(hashlib.sha256(f"drilling/{task}/{seed}".encode()).digest()[:4], "little") % 100
    return "test" if h < 10 else ("validation" if h < 20 else "train")


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=sorted(PUBLIC),
                tasks={k: list(v) for k, v in PUBLIC.items()}, seed_range=[0, 9999],
                qualification="Eleven named tasks over two presets (wall: anchor-holes, rebar-relocate, clean-and-set, "
                              "ledger-anchor; bench: take-drill, punch-layout, press-drill, press-blind, match-bolt, "
                              "panel-holes, timber-bolt); seeds vary marks, rebar offset, aggregate and placements",
                catalogue="seeds 0..999 of every task split 80/10/10 train/validation/test by SHA-256 of (task, seed); "
                          "seeds 1000..9999 resolve with split null")


def task_resolve(selection):
    from .task import INSTRUCTIONS
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"drilling/v1/{v['preset']}/{v['task']}/seed-{v['seed']}", description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=split_of_seed(v["task"], v["seed"]))

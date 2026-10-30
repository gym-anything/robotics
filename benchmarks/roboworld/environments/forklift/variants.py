"""Task resolution for the GAR task catalogue (benchmark.json): a pure function of (task, seed); no physics.

Splits by seed range: train 0-799, validation 800-899, test 900-999; larger seeds resolve with split
null. forklift/data/variations.json freezes, for every task and seed 0..999, the split, a SHA-256 of the seed's resolved
start state (truck pose, target, loads, trailer bed, floor condition) and a short summary; tests re-derive a sample and compare
digests (no silent drift).

  python -m forklift.variants --freeze          # rewrite forklift/data/variations.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import TASKS, DEFAULT_ROBOT, Config

CATALOGUE = Path(__file__).parent / "data" / "variations.json"
N_SEEDS = 1000


def preset_of(task):
    for p, ts in TASKS.items():
        if task in ts and p != "lab":
            return p
    raise ValueError(f"Unknown forklift task {task!r}")


def split_of_seed(seed):
    s = int(seed)
    return "train" if 0 <= s <= 799 else "validation" if 800 <= s <= 899 else "test" if 900 <= s <= 999 else None


def resolved_start(task, seed):
    from . import facility, task as TK
    preset = preset_of(task)
    cfg = Config(preset=preset, task=task, seed=int(seed), robot="none")
    rng = np.random.default_rng(1000 + cfg.seed)
    layout = {"warehouse": facility.warehouse, "dock": facility.dock, "yard": facility.yard}[preset](
        rng, reserved=TK.reserved_locations(cfg))
    return cfg, TK.resolve(cfg, layout)


def check_seed(seed):
    """The seed exactly as given: an integer 0..999 (JSON 1.5, "1", true or 1e3 are refused, never rounded or converted)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"forklift seed must be an integer 0..{N_SEEDS - 1}, not {seed!r}")
    if not 0 <= seed < N_SEEDS:
        raise ValueError(f"forklift seed must be an integer 0..{N_SEEDS - 1}, not {seed}")
    return seed


def task_resolve(selection):
    from .task import instruction
    from .catalogue import CONTRACT, resolve_contract
    task = selection.get("task") or "start-and-park"
    seed = check_seed(selection.get("seed", 0))
    robot = selection.get("robot") or DEFAULT_ROBOT
    cfg, start = resolved_start(task, seed)
    v = dict(preset=cfg.preset, task=task, seed=seed, robot=robot)
    contract = resolve_contract(task, selection.get("contract"))
    options = dict(task=task, seed=seed)
    if selection.get("contract") is not None:
        v["contract"] = options["contract"] = contract
    version = f"/{contract}" if contract == CONTRACT else ""
    return dict(task_id=f"forklift/v1/{cfg.preset}/{task}{version}/seed-{seed}", description=instruction(start),
                backend_options=options, selection=v, split=split_of_seed(seed))


def digest(task, seed):
    cfg, start = resolved_start(task, seed)
    blob = json.dumps(start, sort_keys=True, default=float, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest(), start


def target_of(start):
    """The work order's target in a few characters (lane, bay, rack location, door and side, start position)."""
    t = start["task"]
    if t == "start-and-park":
        return f"parking bay from {start['start_from']}"
    if t == "drive-to-pallet":
        return start["lane"]
    if t in ("carry-to-floor-bay", "carry-loose-stack"):
        return start["bay"]
    if t == "put-away-rack":
        return start["rack_location"]
    if t in ("load-trailer", "load-trailer-chocked"):
        return f"{start['target']['door']} {start['target']['side']}"
    if t == "unload-trailer":
        tg = start["target"]
        return f"{tg['door']} {tg['pallet'][-1]} to {tg['lane']}"
    if t == "load-bales":
        return start["target"]["door"]
    if t == "aisle-cruise":
        return f"{start['cruise']}ward to the far line"
    if t == "feed-conveyor":
        return f"conveyor from {start['start_from']}"
    if t == "ramp-up-loaded":
        return f"trailer nose {start['target']['side']}"
    if t == "ramp-down-loaded":
        return f"drop zone {start['drop_zone']}"
    if t == "yard-route":
        return f"lane {start['route']}"
    return None


def _summary(start):
    out = dict(truck=[round(float(v), 3) for v in start["truck"]])
    if start["task"] == "load-trailer":
        tr = start["trailers"][start["target"]["door"]]
        out["secured"] = bool(tr["restraint"] or tr["chocks"])
    tg = target_of(start)
    if tg:
        out["target"] = tg
    if start.get("load_on_forks"):
        lf = start["load_on_forks"]
        out["load"] = lf["kind"] + (f" {lf['mass']:.0f} kg" if "mass" in lf else "")
    if "bed_z" in start:
        out["bed_in"] = round((start["bed_z"] + 1.2192) / 0.0254, 2)
    if start.get("floor"):
        out["floor"] = sorted(start["floor"])
    return out


def freeze(path=CATALOGUE):
    out = dict(schema="forklift.variations.v1", seeds=N_SEEDS,
               note="split by seed range: 0-799 train, 800-899 validation, 900-999 test; digest = sha256 of the resolved "
                    "start state JSON (sorted keys)", tasks={})
    for preset, tasks in TASKS.items():
        if preset == "lab":
            continue
        for task in tasks:
            rows = []
            for seed in range(N_SEEDS):
                dg, start = digest(task, seed)
                rows.append(dict(seed=seed, split=split_of_seed(seed), digest=dg[:16], **_summary(start)))
            out["tasks"][task] = dict(preset=preset, entries=rows,
                                      counts={k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    if ap.parse_args().freeze:
        res = freeze()
        print({t: v["counts"] for t, v in res["tasks"].items()})

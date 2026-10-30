"""Seeded task catalogue: preset / task / seed / variant. Seeds vary the batch size (+-10 %), the vessel's position
(+-30 mm; +-12 mm on the hob) and yaw, the cart's docking position (+-20 mm), whether a silicone mat is under the bowl,
the milk's temperature (5 or 20 C), the hob level (+-1), the order and lean of the tools in the crock, and the shapes of
the heaps and the well (scene.task_layout, task.start_state). Seed 0 is the nominal station. A variant is another
recipe of a task, or the bowl put on a silicone mat (config.VARIANTS, config.VARIANT_TASKS): a selector like the
others, so `gar tasks ingredient-mixing` lists it, `--select variant=honey` launches it and a reset can change it.
Splits: frozen by hash of (task, seed) over seeds 0..999 (80/10/10); a variant has the split of its task and seed (the
same station with another recipe is never in another split)."""
import hashlib

from .config import TASKS, VARIANTS, VARIANT_TASKS
from .scoring import CONTRACT, LEGACY, ORDER, validate_contract

PLAIN = "plain"                       # the selector's name for "the task as listed" (config.variant "")
VARIANT_CHOICES = (PLAIN,) + tuple(v for v in VARIANTS if v)


def variant_name(variant):
    """config.variant for a selector value: None, "" and "plain" are the task as listed."""
    if variant is None or variant == "" or variant == PLAIN:
        return ""
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose {list(VARIANT_CHOICES)}")
    return variant


def variant_tasks(variant):
    """The tasks a variant belongs to (every task for the plain one)."""
    return VARIANT_TASKS.get(variant, tuple(t for names in TASKS.values() for t in names))


def resolve(preset=None, task=None, seed=0, variant="", contract=LEGACY):
    """A catalogue variation. Task names are unique across presets, so a task alone selects its preset; a variant
    that belongs to one task selects that task; nothing given: bake/muffin-fold."""
    for key, value in (("preset", preset), ("task", task)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Selector {key} must be a name (string)")
    variant = variant_name(variant)
    if task is None and variant and len(variant_tasks(variant)) == 1 \
            and (preset is None or variant_tasks(variant)[0] in TASKS.get(preset, ())):
        task = variant_tasks(variant)[0]
    if preset is None:
        preset = next((k for k, v in TASKS.items() if task in v), "bake") if task else "bake"
    if preset not in TASKS:
        raise ValueError(f"Unknown preset {preset!r}; choose {sorted(TASKS)}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    if variant and task not in variant_tasks(variant):
        raise ValueError(f"Variant {variant!r} belongs to {', '.join(variant_tasks(variant))}, not to {task}")
    validate_contract(contract, task)
    v = dict(preset=preset, task=task, seed=seed, variant=variant)
    if contract != LEGACY:
        v["contract"] = contract
    return v


def split_of_seed(task, seed):
    if seed >= 1000:
        return None
    h = int.from_bytes(hashlib.sha256(f"ingredient-mixing/{task}/{seed}".encode()).digest()[:4], "little") % 100
    return "test" if h < 10 else ("validation" if h < 20 else "train")


def pairs():
    """Every (preset, task, variant) of the catalogue, the plain tasks first."""
    out = [(p, t, "") for p, names in TASKS.items() for t in names]
    for v in VARIANTS:
        if v:
            out += [(p, t, v) for p, names in TASKS.items() for t in names if t in variant_tasks(v)]
    return out


def _variants_text():
    return "; ".join(f"{v}: {VARIANTS[v]} ({'every task' if v not in VARIANT_TASKS else ', '.join(VARIANT_TASKS[v])})"
                     for v in VARIANTS if v)


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed", "variant", "contract"], presets=sorted(TASKS),
                tasks={k: list(v) for k, v in TASKS.items()}, contracts=[LEGACY, CONTRACT], seed_range=[0, 9999],
                variants={(v or PLAIN): dict(what=text, tasks=list(variant_tasks(v))) for v, text in VARIANTS.items()},
                qualification="Eight named tasks over two presets (bake, hob); seeds vary batch size, vessel and cart "
                              "placement, mat, milk temperature, hob level, tool order and heap shapes; a variant is "
                              f"the same task with one thing changed ({_variants_text()}); {len(pairs())} task/variant "
                              "pairs in all",
                catalogue="seeds 0..999 of every task split 80/10/10 train/validation/test by SHA-256 of (task, seed); "
                          "a variant has the split of its task and seed; seeds 1000..9999 resolve with split null")


def catalogue_fields():
    """The fields of benchmark.json that follow config.TASKS / VARIANTS / VARIANT_TASKS (`gar tasks` reads the file;
    `python -m ingredient_mixing.variants` rewrites them, and the unit tests compare)."""
    n_tasks = sum(len(v) for v in TASKS.values())
    return dict(selectors=["preset", "task", "seed", "variant", "contract"],
                selector_choices=dict(preset=list(TASKS), task=[t for names in TASKS.values() for t in names],
                                      variant=list(VARIANT_CHOICES), contract=[LEGACY, CONTRACT]),
                count=len(pairs()),
                count_scope=f"{len(pairs())} task/variant pairs: {n_tasks} named tasks ("
                            + "; ".join(f"{p}: {', '.join(names)}" for p, names in TASKS.items())
                            + f") and {len(pairs()) - n_tasks} variants of them (--select variant=NAME; {_variants_text()}; "
                            "plain: the task as listed); seeds vary batch size, vessel and cart placement, mat, milk "
                            "temperature, hob level, tool order and heap shapes",
                split_note="Frozen by SHA-256 of (task, seed) over seeds 0..999 of every task (80/10/10); a variant has "
                           "the split of its task and seed; seeds 1000..9999 resolve with split null.")


def task_id(v):
    return (f"ingredient-mixing/v1/{v['preset']}/{v['task']}/" + (f"variant-{v['variant']}/" if v["variant"] else "")
            + (f"contract-{v['contract']}/" if v.get("contract", LEGACY) != LEGACY else "")
            + f"seed-{v['seed']}")


def task_resolve(selection):
    from .scoring import TaskConfig, instruction
    unknown = set(selection) - {"preset", "task", "seed", "variant", "contract"}
    if unknown:
        raise ValueError(f"Unknown selectors: {sorted(unknown)}")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("variant"),
                selection.get("contract", LEGACY))
    options = dict(preset=v["preset"], task=v["task"], seed=v["seed"])
    if "contract" in v:
        options["contract"] = v["contract"]
    if v["variant"]:
        options["variant"] = v["variant"]
    return dict(task_id=task_id(v), description=instruction(TaskConfig(**v)), backend_options=options,
                selection=dict(v, variant=v["variant"] or PLAIN), split=split_of_seed(v["task"], v["seed"]))


def task_set_selections():
    """Exactly the eight nominal selections, easiest first; stir-sauce stays on its own evaluator."""
    return [dict(preset="hob" if t in TASKS["hob"] else "bake", task=t, seed=0, variant=PLAIN,
                 contract=LEGACY if t == "stir-sauce" else CONTRACT) for t in ORDER]


def main():
    """Rewrite the catalogue fields of benchmark.json from the configuration."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent/"benchmark.json"
    data = json.loads(path.read_text())
    data.update(catalogue_fields())
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{path}: {data['count']} task/variant pairs; variants {data['selector_choices']['variant']}")


if __name__ == "__main__":
    main()

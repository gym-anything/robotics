"""Seeded task catalogue: preset / task / seed / variant. A preset is the bay's setup (injection: the pad and the vial;
iv-lock: the training arm with its saline lock); a task has one preset, so a task alone selects it. Seeds move the
loose things (tray, basin and trainer by up to 6 mm and 2 to 4 degrees, the vial and the small props by 15 mm), change
how much the set-dose syringe holds and how much air came with it, and how much a used vial has left
(scene.task_layout). Seed 0 is the nominal bay. A variant is the same task with one thing changed (config.VARIANTS,
config.VARIANT_TASKS): a selector like the others, so `gar tasks syringe-handling` lists it, `--select variant=NAME`
launches it and a reset can change it. Splits: frozen by hash of (task, seed) over seeds 0..999 (80/10/10); a variant
has the split of its task and seed."""
import hashlib

from .config import TASKS, VARIANTS, VARIANT_TASKS, LEGACY_CONTRACT, EIGHT_TASK_CONTRACT, resolve_contract

EIGHT_TASKS = ("sharps-away", "set-dose", "draw-dose", "flush-lock", "inject", "fast-bolus", "slow-push", "vial-to-pad")


def eight_task_selections():
    """Exact seed-0 selections, in difficulty order. vial-to-pad keeps its own evaluator and has no new contract."""
    return [dict(preset=next(p for p, ts in TASKS.items() if t in ts), task=t, seed=0, variant="plain",
                 contract=LEGACY_CONTRACT if t == "vial-to-pad" else EIGHT_TASK_CONTRACT) for t in EIGHT_TASKS]

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


def resolve(preset=None, task=None, seed=0, variant="", contract=None):
    """A catalogue variation. Task names are unique across presets, so a task alone selects its preset; a variant
    that belongs to one task selects that task; nothing given: injection/set-dose."""
    for key, value in (("preset", preset), ("task", task)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Selector {key} must be a name (string)")
    variant = variant_name(variant)
    if task is None and variant and len(variant_tasks(variant)) == 1 \
            and (preset is None or variant_tasks(variant)[0] in TASKS.get(preset, ())):
        task = variant_tasks(variant)[0]
    if preset is None:
        preset = next((k for k, v in TASKS.items() if task in v), "injection") if task else "injection"
    if preset not in TASKS:
        raise ValueError(f"Unknown preset {preset!r}; choose {sorted(TASKS)}")
    task = task or ("set-dose" if preset == "injection" else TASKS[preset][0])
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    if variant and task not in variant_tasks(variant):
        raise ValueError(f"Variant {variant!r} belongs to {', '.join(variant_tasks(variant))}, not to {task}")
    value = resolve_contract(task, contract)
    # Without a contract the result and the task id are the v1 ones.
    return dict(preset=preset, task=task, seed=seed, variant=variant,
                **({"contract": value} if contract is not None else {}))


def split_of_seed(task, seed):
    if seed >= 1000:
        return None
    h = int.from_bytes(hashlib.sha256(f"syringe-handling/{task}/{seed}".encode()).digest()[:4], "little") % 100
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
                contracts={LEGACY_CONTRACT: "the v1 reward; vial-to-pad keeps its own evaluator",
                           EIGHT_TASK_CONTRACT: "continuous progress for the seven other tasks"},
                task_set=eight_task_selections(),
                tasks={k: list(v) for k, v in TASKS.items()}, seed_range=[0, 9999],
                variants={(v or PLAIN): dict(what=text, tasks=list(variant_tasks(v))) for v, text in VARIANTS.items()},
                qualification="Eight named tasks over two presets (injection, iv-lock); seeds move the loose things and "
                              "change what the syringe and the vial hold; a variant is "
                              f"the same task with one thing changed ({_variants_text()}); {len(pairs())} task/variant "
                              "pairs in all",
                catalogue="seeds 0..999 of every task split 80/10/10 train/validation/test by SHA-256 of (task, seed); "
                          "a variant has the split of its task and seed; seeds 1000..9999 resolve with split null")


def catalogue_fields():
    """The fields of benchmark.json that follow config.TASKS / VARIANTS / VARIANT_TASKS (`gar tasks` reads the file;
    `python -m syringe_handling.variants` rewrites them, and the unit tests compare)."""
    n_tasks = sum(len(v) for v in TASKS.values())
    return dict(selectors=["preset", "task", "seed", "variant", "contract"],
                selector_choices=dict(preset=list(TASKS), task=[t for names in TASKS.values() for t in names],
                                      variant=list(VARIANT_CHOICES), contract=[LEGACY_CONTRACT, EIGHT_TASK_CONTRACT]),
                task_set={"selections": eight_task_selections(),
                          "note": "Existing placements and success predicates; the seven other tasks select the "
                                  f"continuous-reward contract {EIGHT_TASK_CONTRACT} explicitly. vial-to-pad keeps its "
                                  "own evaluator (contract v1). The new rewards have not been checked in rendered "
                                  "episodes."},
                count=len(pairs()),
                count_scope=f"{len(pairs())} task/variant pairs: {n_tasks} named tasks ("
                            + "; ".join(f"{p}: {', '.join(names)}" for p, names in TASKS.items())
                            + f") and {len(pairs()) - n_tasks} variants of them (--select variant=NAME; {_variants_text()}; "
                            "plain: the task as listed); seeds move the loose things and change what the "
                            "syringe and the vial hold",
                split_note="Frozen by SHA-256 of (task, seed) over seeds 0..999 of every task (80/10/10); a variant has "
                           "the split of its task and seed; seeds 1000..9999 resolve with split null.")


def task_id(v):
    return (f"syringe-handling/v1/{v['preset']}/{v['task']}/" + (f"variant-{v['variant']}/" if v["variant"] else "")
            + (f"contract-{v['contract']}/" if v.get("contract", LEGACY_CONTRACT) != LEGACY_CONTRACT else "")
            + f"seed-{v['seed']}")


def task_resolve(selection):
    from .config import Config
    from .task import instruction
    unknown = set(selection) - {"preset", "task", "seed", "variant", "contract"}
    if unknown:
        raise ValueError(f"Unknown selectors: {sorted(unknown)}")
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("variant"), selection.get("contract"))
    options = dict(preset=v["preset"], task=v["task"], seed=v["seed"])
    if v["variant"]:
        options["variant"] = v["variant"]
    if "contract" in v:
        options["contract"] = v["contract"]
    return dict(task_id=task_id(v), description=instruction(Config(**v)), backend_options=options,
                selection=dict(v, variant=v["variant"] or PLAIN), split=split_of_seed(v["task"], v["seed"]))


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

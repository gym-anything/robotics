"""Seeded task catalogue: preset/task/seed. What a seed changes is listed per task in SEEDS (layouts.py)."""
from .config import PRESETS, TASKS, RETIRED_TASKS, LADDER_TASKS, base_task

# The eight-task benchmark is a subset of the existing catalogue, easiest first.
# Keep TASKS and each preset's default unchanged; no instruction, scene or verifier
# changes are needed. Exact selections and reward contracts: benchmark.json.
TASK_SET = (
    dict(preset="parcel", task="smooth-corner", seed=0),
    dict(preset="parcel", task="press-down", seed=0),
    dict(preset="parcel", task="finish-securing", seed=0),
    dict(preset="parcel", task="carrier-top", seed=0),
    dict(preset="part", task="part-id-hdpe", seed=0),
    dict(preset="parcel", task="gs1-case-side", seed=0),
    dict(preset="part", task="part-id-match", seed=0),
    dict(preset="specimen", task="tube-label", seed=0),
)

SEEDS = {
    "carrier-top": "carton position (+-20 mm) and yaw (+-6 deg); tracking number",
    "gs1-case-side": "carton position (+-20 mm) and yaw (+-6 deg); SSCC serial",
    "cover-old-label": "carton position and yaw; the old 4 x 3 in label's flap half and position on it; tracking numbers",
    "press-down": "carton position and yaw; where the co-worker laid the label (+-30 mm along, +-5 mm across, +-2 deg); tracking number",
    "tube-wrap": "the tube's roll in the cradle (any angle; seed 0 manufacturer label up); accession number",
    "part-id-hdpe": "canister position (+-20 mm; seed 0 centred); serial number",
    "part-id-match": "the three plates' part numbers, which slot holds the labelled one, plate poses (+-8 mm, +-4 deg), marking corners; serial number",
    "smooth-corner": "carton position and yaw; where the label was stuck (+-30 mm along, +-5 mm across, +-2 deg, either way round) on the flap half toward the robots; which corner is lifted (any of four), its size (legs 28-38 x 22-32 mm) and curl (65-90 deg); tracking number",
    "finish-securing": "carton position and yaw; which flap half the label was stuck on and where (+-30 mm along, +-5 mm across, +-3 deg, either way round); two to four lifted areas among the corners, ends and long sides, their sizes and curls (12-35 % of the label); tracking number",
    "tube-label": "the tube's roll in the cradle (any angle; seed 0 manufacturer label up); accession number",
}


def resolve(preset="parcel", task=None, seed=0):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset; choose {PRESETS}")
    if task in RETIRED_TASKS:
        raise ValueError(f"{task} was {RETIRED_TASKS[task]}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


def contract():
    from .task import REWARD_CONTRACTS
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=list(PRESETS),
                tasks={k: list(v) for k, v in TASKS.items()}, seed_range=[0, 9999],
                seeds=dict(SEEDS), retired=dict(RETIRED_TASKS),
                reward_contracts=dict(REWARD_CONTRACTS),
                task_set=[dict(s) for s in TASK_SET],
                shares_bench_with={t: base_task(t) for t in LADDER_TASKS},
                qualification="Ten task ids, nine distinct jobs, over three bench presets (tube-label is tube-wrap's "
                              "job seed for seed, held to the stricter reading its instruction states: count it "
                              "once); every task is scored by a progress reward beside "
                              "success (reward_contracts); seeds vary the layout "
                              "listed per task (robustness variation, not separate tasks and not independently "
                              "qualified episodes)")


def task_resolve(selection):
    from .task import INSTRUCTIONS
    v = resolve(selection.get("preset", "parcel"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"label-application/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=None)

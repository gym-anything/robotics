"""Task-only catalogue. Legacy ids/defaults stay stable; revised contracts are explicit.

Kept apart from the top-level physics modules, which stay unchanged. The legacy
instruction literals below mirror layout.TASKS; tests check that they stay equal.
"""
import hashlib
from copy import deepcopy

CONTRACT = "surface-rinsing-tasks-v1"
PAN_CONTRACT = "progress-v1"                # task.REWARD_CONTRACTS["scrub-rinse-pan"]
NEW_TASK = "drain-wash-compartment"
STATIONS = ("pot-sink", "pre-rinse", "prep-sink", "can-wash")

LEGACY_TASKS = {
    "rinse-glassware": dict(preset="pot-sink", tier="medium", instruction=(
        "Six washed glasses (four 12 oz rocks glasses and two 16 oz mixing glasses) stand on the soiled (left) drainboard, "
        "still wet with detergent solution and suds. Rinse each one free of detergent inside and out under the swing "
        "faucet or in the rinse compartment, immerse it in the sanitizer compartment (right bowl) for at least 30 s, and "
        "set it mouth-down in the glass rack on the clean drainboard to air-dry. Do not chip or drop any glass; leave the "
        "faucet and handles closed and nothing in the hands.")),
    "rinse-dishware": dict(preset="pot-sink", tier="medium", instruction=(
        "Four washed 9 in plates lie on the soiled drainboard with a soup bowl standing in each of the two back plates, "
        "all wet with detergent solution. Rinse every face of each (including the back and foot ring), sanitize each for "
        "at least 30 s, stand the plates in the peg rack and set the bowls upside down on the clean drainboard. Leave the "
        "faucet closed and nothing in the hands.")),
    "scrub-rinse-pan": dict(preset="pot-sink", tier="hard", instruction=(
        "The 10 in aluminium fry pan soaking in the wash compartment (left bowl) has dried egg stuck to its inside; a "
        "scrub sponge is on the back ledge. Clean the egg off the pan, rinse it free of detergent, immerse it in the "
        "sanitizer compartment (right bowl) for at least 30 s, and set it upside down on the wall shelf over the clean "
        "drainboard to drain. Finish with the sponge on the back ledge, no water running and nothing in the hands.")),
    "rinse-sponge": dict(preset="pot-sink", tier="entry", instruction=(
        "The scrub sponge on the back ledge is full of wash solution. Rinse it under the faucet until it releases no more "
        "suds, wring it out and set it back on the ledge. Leave the faucet closed and nothing in the hands.")),
    "pre-rinse-plates": dict(preset="pre-rinse", tier="medium-hard", instruction=(
        "Six plates soiled with dried tomato sauce stand in the peg rack on the pre-rinse sink. Mix the water to about "
        "49 C (110-120 F), pre-rinse every plate with the pre-rinse spray valve until no sauce is visible on its face or "
        "rim, without spraying onto the clean dish table, then shut the water and hang the spray valve back up.")),
    "rinse-tomatoes": dict(preset="prep-sink", tier="medium", instruction=(
        "The colander on the prep-sink drainboard holds fifteen dusty cherry tomatoes. Rinse them under running water "
        "until none carries visible soil, without losing any, turning them so every side is rinsed, then set the "
        "colander in the clean bowl on the work table to drain and close the faucet.")),
    "rinse-garbage-can": dict(preset="can-wash", tier="hard", instruction=(
        "The 32 gal garbage can in the can-wash bay has been scrubbed with detergent. Rinse its inside and outside free "
        "of soap and loosened debris with the hose, empty the rinse water into the floor drain, and leave the can "
        "upright with no water in it. Dock the hose valve back in its wall holster and do not spray the dunnage shelf.")),
    "flush-floor": dict(preset="can-wash", tier="medium", instruction=(
        "Soapy water and debris cover part of the can-wash bay floor. Flush them into the floor drain with the hose "
        "without spraying the dunnage shelf, then dock the hose valve back in its wall holster.")),
}
TASKS = deepcopy(LEGACY_TASKS)
TASKS[NEW_TASK] = dict(preset="pot-sink", tier="entry", instruction=(
    "The wash water in the left compartment of the pot sink is spent. Drain that compartment, then close its drain "
    "so it is ready to be refilled. Leave the rinse and sanitizer compartments as they are, and finish with no water "
    "running and nothing in the hands."))
TASKS["pre-rinse-plates"]["tier"] = "hard"
TASKS["scrub-rinse-pan"]["tier"] = "hardest"
# The temperature sentence is removed rather than adding a flow-temperature
# gate. Cleanliness, placement and containment are the job here; the unchanged
# physics still uses actual water temperature.
TASKS["pre-rinse-plates"]["instruction"] = (
    "Six plates soiled with dried tomato sauce stand in the peg rack on the pre-rinse sink. Pre-rinse every plate "
    "until no sauce is visible on its face or rim, without spraying onto the clean dish table. Leave the plates in "
    "the rack, the water shut off and the spray valve hanging back up, with nothing in the hands.")
PRESETS = {s: tuple(t for t, v in TASKS.items() if v["preset"] == s) for s in STATIONS}
EIGHT = ("rinse-sponge", NEW_TASK, "rinse-tomatoes", "flush-floor", "rinse-glassware", "pre-rinse-plates",
         "rinse-garbage-can", "scrub-rinse-pan")
REVISED = frozenset(EIGHT) - {"scrub-rinse-pan"}


def resolve(preset=None, task=None, seed=0, contract=None):
    # Same type checks, defaults, and seed bounds as variants.resolve.
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
    if contract is None:
        contract = CONTRACT if task == NEW_TASK else "legacy"
    allowed = {CONTRACT} if task == NEW_TASK else {"legacy"}
    if task in REVISED:
        allowed.add(CONTRACT)
    if task == "scrub-rinse-pan":
        allowed.add(PAN_CONTRACT)
    if not isinstance(contract, str) or contract not in allowed:
        raise ValueError(f"Contract for {task} must be one of {sorted(allowed)}")
    out = dict(preset=preset, task=task, seed=seed)
    if contract == CONTRACT:
        out["contract"] = contract
    return out


def split_of(task, seed):
    if not 0 <= seed < 1000:
        return None
    h = int(hashlib.sha256(f"{task}/{seed}".encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def instruction(v):
    tasks = TASKS if v.get("contract") == CONTRACT else LEGACY_TASKS
    return tasks[v["task"]]["instruction"]


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    suffix = f"/{CONTRACT}" if v.get("contract") == CONTRACT else ""
    return dict(task_id=f"surface-rinsing/v1/{v['preset']}/{v['task']}{suffix}/seed-{v['seed']}",
                description=instruction(v), backend_options=dict(v), selection=v, split=split_of(v["task"], v["seed"]))


def contract():
    return dict(supported=True, selectors=["preset", "task", "seed", "contract"], presets=sorted(PRESETS),
                tasks={k: list(v) for k, v in PRESETS.items()}, seed_range=[0, 9999],
                contracts=["legacy", CONTRACT, PAN_CONTRACT],
                catalogue="Legacy variations.json unchanged; new task splits use the same (task, seed) hash.")


def selections():
    return [resolve(task=t, contract=CONTRACT if t in REVISED else None) for t in EIGHT]


def setup(task, seed=0, contract=None):
    from .. import layout
    v = resolve(task=task, seed=seed, contract=contract)
    # Only the new task has a new scene: same filled sink, no movable ware/tool.
    spec = layout.setup("rinse-sponge" if task == NEW_TASK else task, seed)
    if task == NEW_TASK:
        spec["items"] = {}
        spec["task"] = task
    spec["instruction"] = instruction(v)
    spec["task_contract"] = v.get("contract", "legacy")
    return spec

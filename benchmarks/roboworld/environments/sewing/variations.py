"""Catalogue of variations of the sewing environment (sewing-sample-room-v1), served by GAR's /v1/variations,
/v1/variations/select and /v1/variations/start and by a reset with options {"variation_id": N}. At launch an entry is
chosen by its id as well (`gar run sewing --variation N` or `--select variation_id=N`; task_resolve, the resolver of
sewing/benchmark.json).

A variation is one task of the sample-room preset at one seed: 9 tasks x 10 seeds = 90 entries (the original 0-49
retain their meaning), the same task x seed
grid a session selects at launch (sewing/variants.py resolve, sewing/benchmark.json), so every entry is exactly the
episode a fresh session with that task and seed starts. Its id is task_index x 10 + seed over the frozen TASK_ORDER
(variation 10 is plain-seam seed 0, the default launch). Seeds vary only where the plies start (variants.py: the
stack's offset and the top ply's misalignment; for curve, the corner's radius) and never a physical parameter or a
threshold; a task sets its plies, stitch length, seam allowance, start state and, for baste, the needle tension the
operator dials (sewing/task.py TASK_SETUP). No train/test split is invented (benchmark.json); every entry's split is
null.

Switching keeps the session's robot, cameras, renderer options, device and config overrides and builds the entry's
world exactly as a fresh session at that task and seed does (sewing/env.py SewingEnv): its start state is loaded from
the cache (keyed by the physics hash and the configuration, the same file a fresh session uses) or, when not cached,
prepared first by dynamic relaxation (START_STATE_NOTE gives the measured times).

  python -m sewing.variations            # print the catalogue (one JSON line per entry)
"""

from functools import lru_cache
import hashlib
import json

VERSION = "sewing-sample-room-v1"
PRESET = "sample-room"
# frozen order of the ids: a task added to sewing/task.py later gets new ids after these, never shifts them
TASK_ORDER = ("load", "plain-seam", "baste", "pivot", "curve", "seam-to-notch", "square-plies",
              "three-sides", "load-and-seam")
SEEDS = tuple(range(10))
COUNT = len(TASK_ORDER)*len(SEEDS)
SELECTORS = ("variation_id", "preset", "task", "seed", "contract")

PARAMETERS = {
    "stack_offset": dict(
        under_foot="dx +-1 mm (across the seam), dy +-2 mm (along it), yaw +-0.5 deg",
        tray="dx, dy +-20 mm, yaw +-5 deg (task load: the stack's place on the table left of the machine)",
        label="authored", basis="what an operator leaves laying cut parts down (sewing/variants.py)",
        physical_effect="where the plies start; seed 0 is the authored layout (no offset)"),
    "top_ply_misalign_m": dict(range="0-0.5 mm across the seam (tasks starting under the foot; 0 for load)",
                               label="authored", basis="sewing/variants.py",
                               physical_effect="how far the top ply's seam edge starts from the bottom ply's"),
    "corner_radius_m": dict(range="80-120 mm (task curve; seed 0 the authored 100 mm)", label="authored",
                            basis="sewing/variants.py: seeds vary the curve's radius",
                            physical_effect="the rounded corner the curve seam follows"),
}

# How long a switch takes, measured with the FR3 pair without cameras through GAR's HTTP server on the shared RTX 5090,
# one sewing process at a time beside other jobs on the GPU, the host's 1-minute load recorded. Cached: pivot seed 0, 4.6 s (4.0 s build, 0.5 s releasing the old world; load 5.4). Prepared: plain-seam
# seed 0 in a switch, 271 s (load 6.2-6.5); pivot seed 0 at launch, 208 s (load 5.4-6.3) and 418 s (load 7-12).
START_STATE_NOTE = (
    "A switch builds the entry's world and loads its start state from the cache under $GAR_WORKSPACE/.gar/"
    "sewing-cache (or ~/.cache/gar/sewing), keyed by the physics hash and the configuration (robot included): the "
    "same file a fresh session at that task and seed loads; a cached switch took 4.6 s without cameras (FR3 pair, "
    "pivot). With the policy cameras a switch also releases the render worker and its response renders the new "
    "scene from a new one: cached switches took 18.5-22.6 s with the three cameras and 4.8-13.1 s without, on a "
    "loaded host. A start "
    "state not yet cached (the first use of a task and seed on a deployment after a physics change) is prepared "
    "first: 1.25 s of simulated time for the whole cell (1.0 s dynamic relaxation and 0.25 s undamped settle, 2,500 "
    "ticks), which took 208-418 s for the FR3 pair on the shared RTX 5090 at a 1-minute host load of 5-12 with other "
    "jobs on the GPU (plain-seam 271 s, pivot 208 and 418 s) and scales with the host's load like any "
    "command (capabilities wall_time). The request returns only when the new world is ready: allow it at least the "
    "900 s client timeout a command needs.")


def _check_index(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < COUNT:
        raise ValueError(f"variation_id must be an integer in [0, {COUNT - 1}]")
    return index


def task_seed(index):
    """(task, seed) of a variation id."""
    _check_index(index)
    return TASK_ORDER[index//len(SEEDS)], SEEDS[index % len(SEEDS)]


def _task_name(task):
    """A task name or its task code (A0, A1, ...), as sewing/variants.py resolve accepts."""
    from .task import TASKS
    if task in TASKS:
        return task
    code = {v: k for k, v in TASKS.items()}
    if isinstance(task, str) and task in code:
        return code[task]
    raise ValueError(f"Unknown task {task!r}; tasks: {', '.join(TASK_ORDER)}")


def index_of(task, seed):
    """The variation id of a task (name or code) and seed."""
    task = _task_name(task)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError(f"seed must be an integer in [{SEEDS[0]}, {SEEDS[-1]}]")
    return TASK_ORDER.index(task)*len(SEEDS) + seed


def _plain(v):
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, float):
        return float(v)
    return v


@lru_cache(maxsize=None)
def _spec(index):
    from .task import TASKS, TIME_LIMIT_S
    from .variants import resolve
    task, seed = task_seed(index)
    v = resolve(PRESET, task, seed)
    data = dict(catalogue=VERSION, variation_id=index, task_id=f"sewing/v1/{PRESET}/{task}/seed-{seed}",
                preset=PRESET, task=task, task_code=TASKS[task], seed=seed, split=None, start=v["start"],
                time_limit_s=TIME_LIMIT_S[task],
                # the world's Config fields this entry sets (sewing/world.py; the rest keep their defaults)
                config={k: _plain(x) for k, x in v.items() if k not in ("preset", "task", "seed")})
    from .task_set import NEW_TASKS, BASE_TASK, default_contract
    if task in NEW_TASKS:
        data.update(task_contract=default_contract(task), physical_task=BASE_TASK[task])
    if task == "square-plies":
        data["task_start"] = dict(name="tray_slipped", top_ply_translation_m=[.040, .025], yaw_deg=6.,
                                  revision="tray-slipped-40x25mm-yaw6-v1",
                                  preparation="before existing relaxation; separate cache")
    data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return json.dumps(data, sort_keys=True)


def specification(index, contract=None):
    """The catalogue entry of a variation id (a fresh dict; deterministic)."""
    item = json.loads(_spec(_check_index(index)))
    from .task_set import choose_contract, default_contract
    chosen = choose_contract(item["task"], contract)
    if chosen != default_contract(item["task"]):
        item["contract"] = chosen
        del item["sha256"]
        item["sha256"] = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return item


def config_values(entry):
    """sewing.world.Config fields of an entry: what sewing/variants.py resolve gives a fresh session at the entry's task
    and seed (the backend lays them over the session's config overrides, as its constructor does)."""
    from .variants import resolve
    return resolve(entry["preset"], entry["task"], entry["seed"])


def select(*, variation_id=None, preset=None, task=None, seed=None, contract=None):
    """An entry by id (any of preset, task, seed given must match it) or by task and seed (both required without an
    id). Exact; constraints are never relaxed."""
    if preset is not None and preset != PRESET:
        raise ValueError(f"preset must be {PRESET!r} (the only preset built)")
    if variation_id is not None:
        item = specification(variation_id, contract)
        if task is not None and _task_name(task) != item["task"]:
            raise ValueError(f"variation {variation_id} is task {item['task']!r}, not {task!r}; constraints are not "
                             "relaxed")
        if seed is not None and seed != item["seed"]:
            raise ValueError(f"variation {variation_id} is seed {item['seed']}, not {seed!r}; constraints are not "
                             "relaxed")
        return item
    if task is None or seed is None:
        raise ValueError("Choose variation_id, or task and seed (both)")
    return specification(index_of(task, seed), contract)


def select_request(request):
    """GAR /v1/variations/select request -> entry (exact keys; never relaxed)."""
    if not isinstance(request, dict) or not request or set(request) - set(SELECTORS):
        raise ValueError(f"Choose variation_id, or task and seed (optionally preset); selectors: {', '.join(SELECTORS)}"
                         "; unknown fields are rejected")
    return select(**request)


def task_resolve(selection):
    """GAR task catalogue hook (gym_anything_robotics/tasks.py; sewing/benchmark.json names it as the resolver):
    selection -> task id, instruction, backend options. Without a variation_id the selection is a preset, task and
    seed, resolved by sewing/variants.py task_resolve as before. With one it is the catalogue entry of that id; a
    preset, task or seed given beside it must be the entry's (never relaxed). An entry is a task at a seed, so the
    launch it resolves to is that task and seed's: the same task id and backend options, the same cached start
    state."""
    from .task_set import task_resolve as by_task_and_seed
    if "variation_id" not in selection:
        return by_task_and_seed(selection)
    item = select_request(dict(selection))
    resolved = by_task_and_seed(dict(preset=item["preset"], task=item["task"], seed=item["seed"],
                                    contract=item.get("contract")))
    if resolved["task_id"] != item["task_id"]:
        raise ValueError(f"variation {item['variation_id']} is {item['task_id']}, the launch resolved "
                         f"{resolved['task_id']}")
    resolved["selection"] = dict(resolved["selection"], variation_id=item["variation_id"], catalogue=VERSION,
                                 sha256=item["sha256"])
    return resolved


def contract(current=None, checkpoints=None):
    from .task_set import TASK_SET, CONTRACT, LEGACY, CURVE
    return dict(supported=True, catalogue=VERSION, count=COUNT, integer_id_range=[0, COUNT - 1],
                id_rule=f"variation_id = task index x {len(SEEDS)} + seed over tasks {list(TASK_ORDER)}",
                selectors=list(SELECTORS), tasks=list(TASK_ORDER), seeds=[SEEDS[0], SEEDS[-1]], splits=[],
                split_note="No train/test split invented (sewing/benchmark.json); every entry's split is null.",
                parameters=PARAMETERS, current=current, preview_mutates=False, start_replaces_episode=True,
                task_set=[dict(s) for s in TASK_SET],
                task_contracts=dict(legacy=LEGACY, task_set=CONTRACT, curve=CURVE),
                reset_request='{"options": {"variation_id": N}} (POST /v1/reset); /v1/variations/start sends it',
                note="A variation is one task at one seed, the episode a fresh session with that task and seed "
                     "starts; switching keeps the session's robot, cameras, renderer options, device and config "
                     "overrides, and drops its checkpoints and its render worker (the first observation after a "
                     "switch starts a new worker). " + START_STATE_NOTE,
                checkpoints=checkpoints,
                qualification="Defined reproducible catalogue; entries are not each physically qualified.")


def rows():
    return [specification(i) for i in range(COUNT)]


def digest():
    """SHA-256 of the whole catalogue (one sorted JSON line per entry): its identity, held by a test."""
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows()).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    for r in rows():
        print(json.dumps(r, sort_keys=True))

"""sewing-tasks-v2: the render-checked eight-task set, beside the released task_set.py (sewing-tasks-v1).

Task selection only; no simulator imports. A selection opts in with contract="sewing-tasks-v2"; nothing without it
changes (task_set.py, task_set_env.py and variations.py hand this contract, and the tasks only it defines, to this
module). What the render check changed against the released set:
- three-sides is swapped out of the eight for corner-seam. three-sides reuses the pivot start, whose mesh has its
  1.25 mm seam band only along the right and near edges (world.py _build_cloth); its third leg, up the left edge,
  lies in 7.5-12.2 mm bulk triangles, where machine.py ends the episode as a world fault (a hole in a triangle coarser
  than 2 FINE_H), so it cannot succeed as built. corner-seam is the pivot's L seam with both ends backtacked; the whole
  L lies in the band. It is defined here only (sewing/task.py and the released catalogue of 90 variations are
  unchanged): its start, mesh and cached state are the pivot's, its time limit 180 s, and under this contract its
  variation ids are 90-99, after the released 90.
- load and square-plies, the two handling tasks without a seam, score 0.9 x progress short of success, as the spec
  sets; the seam tasks keep the curve's 0.95 (task_set_v2_scoring.failure_factor).
- every selection of this contract has its own task id, the contract before the seed
  (sewing/v1/sample-room/<task>/sewing-tasks-v2/seed-N), so results stored under one id follow one verdict.
- the seven tasks other than curve take this contract; curve keeps its audited evaluator and its released selection.
"""

import hashlib
import json
from functools import lru_cache

from . import task_set as V1
from .task import INSTRUCTIONS, G1_MACHINE_NOTE, TIME_LIMIT_S as V1_TIME_LIMIT_S, task_setup
from .variants import resolve as legacy_resolve

CONTRACT = "sewing-tasks-v2"
# Defined by this contract only. Code A14 follows the released A10-A13 (task.py); its base, the start it reuses, is BASE
# in tasks/corner-seam/setup.py.
CORNER_SEAM_INSTRUCTION = ("Join the two plies with a seam 10 mm from the right edge, turning a square corner, and "
                           "continue 10 mm from the near edge to the left edge; backtack three or four stitches at "
                           "both ends, trim the threads and let go of the work.")
EXTRA_TASKS = {"corner-seam": dict(code="A14", base=task_setup("corner-seam").BASE, time_limit_s=180.0,
                                   instruction=CORNER_SEAM_INSTRUCTION)}
TASKS = ("load", "seam-to-notch", "plain-seam", "square-plies", "pivot", "corner-seam", "load-and-seam")
EIGHT_TASKS = (*TASKS, "curve")
HARDEST_SELECTION = {"task": "curve"}                 # splits/test.json, unchanged (audited, sewing-curve-v1)
BASE_TASK = dict(V1.BASE_TASK, **{t: e["base"] for t, e in EXTRA_TASKS.items()})
TIME_LIMIT_S = dict(V1_TIME_LIMIT_S, **{t: e["time_limit_s"] for t, e in EXTRA_TASKS.items()})
TASK_SET = tuple(dict(preset="sample-room", task=t, seed=0, contract=CONTRACT) for t in TASKS) \
    + (dict(HARDEST_SELECTION),)


def task_name(task):
    """corner-seam by its name or code; any other task as given."""
    for name, e in EXTRA_TASKS.items():
        if task == e["code"]:
            return name
    return task


def selects(task=None, contract=None):
    """Whether a call belongs here: it names this contract, or a task only this contract defines."""
    return contract == CONTRACT or (isinstance(task, str) and task_name(task) in EXTRA_TASKS)


def choose_contract(task, contract=None):
    if task in TASKS and contract == CONTRACT:
        return CONTRACT
    if task in EXTRA_TASKS:
        raise ValueError(f"{task} is selected with contract {CONTRACT!r} (the render-checked eight-task set)")
    if contract == CONTRACT:
        raise ValueError(f"Unknown contract {contract!r} for {task}; {CONTRACT} is for {', '.join(TASKS)}")
    return V1.choose_contract(task, contract)


def resolve(preset=None, task=None, seed=0, contract=None):
    """variants.resolve's coercions, defaults and exceptions; corner-seam resolves as the pivot start it reuses."""
    name = task_name(task)
    if name in EXTRA_TASKS:
        v = legacy_resolve(preset, EXTRA_TASKS[name]["base"], seed)
        v["task"] = name
    else:
        v = legacy_resolve(preset, task, seed)
    choose_contract(v["task"], contract)
    return v


def instruction(task, robot=None, contract=None):
    if choose_contract(task, contract) != CONTRACT:
        return V1.instruction(task, robot, contract)
    text = (V1.PIVOT_INSTRUCTION if task == "pivot" else
            EXTRA_TASKS[task]["instruction"] if task in EXTRA_TASKS else INSTRUCTIONS[task])
    return text + G1_MACHINE_NOTE if robot == "g1" else text


def task_id(preset, task, seed):
    return f"sewing/v1/{preset}/{task}/{CONTRACT}/seed-{seed}"


def task_resolve(selection):
    """A selection of this contract (task_set.task_resolve hands it here); any other resolves as released."""
    if not selects(selection.get("task"), selection.get("contract")):
        return V1.task_resolve(selection)
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    picked = dict(preset=v["preset"], task=v["task"], seed=v["seed"], contract=CONTRACT)
    return dict(task_id=task_id(v["preset"], v["task"], v["seed"]), description=instruction(v["task"], contract=CONTRACT),
                backend_options=dict(picked), selection=picked, split=None)


# ---------------------------------------------------------------------------------------------- variation catalogue
# The released catalogue (variations.py: 9 tasks x 10 seeds, ids 0-89, its digest) is unchanged. Under this contract
# an id names the same entry with the contract, its own task id and entry hash; corner-seam follows at 90-99.

def _vr():
    from . import variations
    return variations


def task_order():
    return _vr().TASK_ORDER + tuple(EXTRA_TASKS)


def count():
    return len(task_order())*len(_vr().SEEDS)


def _check_index(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count():
        raise ValueError(f"variation_id must be an integer in [0, {count() - 1}] under {CONTRACT}")
    return index


def index_of(task, seed):
    VR = _vr()
    name = task_name(task)
    if name not in EXTRA_TASKS:
        return VR.index_of(task, seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in VR.SEEDS:
        raise ValueError(f"seed must be an integer in [{VR.SEEDS[0]}, {VR.SEEDS[-1]}]")
    return task_order().index(name)*len(VR.SEEDS) + seed


@lru_cache(maxsize=None)
def _extra_spec(index):
    """An entry after the released catalogue, built as variations._spec builds the released ones."""
    VR = _vr()
    task, seed = task_order()[index//len(VR.SEEDS)], VR.SEEDS[index % len(VR.SEEDS)]
    v = resolve(VR.PRESET, task, seed, CONTRACT)
    data = dict(catalogue=VR.VERSION, variation_id=index, task_id=f"sewing/v1/{VR.PRESET}/{task}/seed-{seed}",
                preset=VR.PRESET, task=task, task_code=EXTRA_TASKS[task]["code"], seed=seed, split=None,
                start=v["start"], time_limit_s=TIME_LIMIT_S[task],
                config={k: VR._plain(x) for k, x in v.items() if k not in ("preset", "task", "seed")})
    data.update(task_contract=CONTRACT, physical_task=BASE_TASK[task])
    return json.dumps(data, sort_keys=True)


def _sha(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def specification(index):
    """The catalogue entry of a variation id under this contract (a fresh dict; deterministic)."""
    VR = _vr()
    index = _check_index(index)
    item = json.loads(_extra_spec(index) if index >= VR.COUNT else VR._spec(index))
    choose_contract(item["task"], CONTRACT)
    item.pop("sha256", None)
    item["contract"] = CONTRACT
    if "task_contract" in item:
        item["task_contract"] = CONTRACT
    item["task_id"] = task_id(item["preset"], item["task"], item["seed"])
    item["sha256"] = _sha(item)
    return item


def config_values(entry):
    """sewing.world.Config fields of an entry under this contract (variations.config_values for the released tasks)."""
    return resolve(entry["preset"], entry["task"], entry["seed"], CONTRACT)


def _task_of(task):
    name = task_name(task)
    return name if name in EXTRA_TASKS else _vr()._task_name(task)


def select(*, variation_id=None, preset=None, task=None, seed=None):
    """variations.select under this contract: exact, never relaxed."""
    VR = _vr()
    if preset is not None and preset != VR.PRESET:
        raise ValueError(f"preset must be {VR.PRESET!r} (the only preset built)")
    if variation_id is not None:
        item = specification(variation_id)
        if task is not None and _task_of(task) != item["task"]:
            raise ValueError(f"variation {variation_id} is task {item['task']!r}, not {task!r}; constraints are not "
                             "relaxed")
        if seed is not None and seed != item["seed"]:
            raise ValueError(f"variation {variation_id} is seed {item['seed']}, not {seed!r}; constraints are not "
                             "relaxed")
        return item
    if task is None or seed is None:
        raise ValueError("Choose variation_id, or task and seed (both)")
    return specification(index_of(task, seed))


def variation_contract(current=None, checkpoints=None):
    """GET /v1/variations of a session under this contract: the released catalogue's, with corner-seam's ids and
    this set."""
    VR = _vr()
    c = VR.contract(None, checkpoints)
    n = count()
    c.update(count=n, integer_id_range=[0, n - 1], tasks=list(task_order()), current=current,
             id_rule=f"variation_id = task index x {len(VR.SEEDS)} + seed over tasks {list(task_order())}",
             task_set=[dict(s) for s in TASK_SET], contract=CONTRACT,
             task_contracts=dict(legacy=V1.LEGACY, task_set=CONTRACT, curve=V1.CURVE),
             contract_note=f"Under {CONTRACT} (send it with the variation id) ids 0-{VR.COUNT - 1} are the released "
                           f"entries of its seven tasks and {VR.COUNT}-{n - 1} corner-seam; without it an id is the "
                           "released catalogue's.")
    return c

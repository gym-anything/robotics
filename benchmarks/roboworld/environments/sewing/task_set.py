"""Task selection only; no simulator imports. The original resolver is physics-hashed and unchanged."""

from .task import INSTRUCTIONS, G1_MACHINE_NOTE, task_setup
from .variants import resolve as legacy_resolve

CONTRACT = "sewing-tasks-v1"
LEGACY = "sewing-v1"
CURVE = "sewing-curve-v1"
NEW_TASKS = ("seam-to-notch", "square-plies", "three-sides", "load-and-seam")
REVISED = ("load", "plain-seam", "pivot")
EIGHT_TASKS = ("load", "seam-to-notch", "plain-seam", "square-plies", "pivot", "three-sides",
               "load-and-seam", "curve")
# the task whose start a new task reuses (a benchmark task's BASE in its setup.py; three-sides the pivot's)
BASE_TASK = dict(zip(NEW_TASKS, (task_setup("seam-to-notch").BASE, task_setup("square-plies").BASE, "pivot",
                                 task_setup("load-and-seam").BASE)))
PIVOT_INSTRUCTION = ("Join the two plies with a seam 10 mm from the right edge, turning a square corner, and "
                     "continue 10 mm from the near edge to the left edge; trim the threads and let go of the work.")
TASK_SET = tuple(dict(preset="sample-room", task=t, seed=0,
                      **({"contract": CONTRACT} if t in REVISED else {})) for t in EIGHT_TASKS)


def _v2(task=None, contract=None):
    """sewing-tasks-v2, the render-checked set (task_set_v2.py), takes a call naming its contract or a task only it
    defines; every other call runs as released."""
    from . import task_set_v2
    return task_set_v2 if task_set_v2.selects(task, contract) else None


def default_contract(task):
    return CURVE if task == "curve" else CONTRACT if task in NEW_TASKS else LEGACY


def choose_contract(task, contract=None):
    if v2 := _v2(task, contract):
        return v2.choose_contract(task, contract)
    default = default_contract(task)
    choices = (default, CONTRACT) if task in REVISED else (default,)
    if contract is not None and contract not in choices:
        raise ValueError(f"Unknown contract {contract!r} for {task}; choose {choices}")
    return default if contract is None else contract


def resolve(preset=None, task=None, seed=0, contract=None):
    if v2 := _v2(task, contract):
        return v2.resolve(preset, task, seed, contract)
    # Preserve every coercion, alias, default and exception of variants.resolve.
    v = legacy_resolve(preset, task, seed)
    choose_contract(v["task"], contract)
    return v


def instruction(task, robot=None, contract=None):
    if v2 := _v2(task, contract):
        return v2.instruction(task, robot, contract)
    chosen = choose_contract(task, contract)
    text = PIVOT_INSTRUCTION if task == "pivot" and chosen == CONTRACT else INSTRUCTIONS[task]
    return text + G1_MACHINE_NOTE if robot == "g1" else text


def task_resolve(selection):
    if v2 := _v2(selection.get("task"), selection.get("contract")):
        return v2.task_resolve(selection)
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0),
                selection.get("contract"))
    chosen = choose_contract(v["task"], selection.get("contract"))
    picked = dict(preset=v["preset"], task=v["task"], seed=v["seed"])
    # Default selections keep their original backend options and ids. Contract is separate provenance.
    if chosen != default_contract(v["task"]):
        picked["contract"] = chosen
    return dict(task_id=f"sewing/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=instruction(v["task"], contract=chosen), backend_options=dict(picked),
                selection=picked, split=None)

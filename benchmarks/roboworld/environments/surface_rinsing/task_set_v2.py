"""surface-rinsing-tasks-v2: the render-checked eight-task set. Catalogue, resolver and scoring; no simulator imports.

The seven revised tasks of the released set (task_set/catalogue.py REVISED) take `"contract":
"surface-rinsing-tasks-v2"`; the audited pan keeps its released legacy selection. Under v2 every task has the same
scene, instruction, predicates and progress as under surface-rinsing-tasks-v1, except pre-rinse-plates, which keeps
its instruction with the temperature sentence ("Mix the water to about 49 C (110-120 F)"), as the spec says ("Keep,
provided the temperature gate below is added"): the v2 evaluator reads the flow-weighted mean temperature of the water
that left the spray valve (spray_temperature, a success predicate) and grades progress by it (temperature_grade); both
are in the task's folder (tasks/pre-rinse-plates/verifier.py), which progress() below reads.
Runtime: task_set_v2_runtime.py, task_set_v2_evaluator.py, task_set_v2_backend.py. Selections without this contract
resolve through task_set/catalogue.py as released.
"""
from copy import deepcopy

from .task_set import catalogue as V1
from .task_set.catalogue import EIGHT, LEGACY_TASKS, NEW_TASK, REVISED, split_of
from .task_set.scoring import FAIL_SCALE, finite, progress as v1_progress

CONTRACT = "surface-rinsing-tasks-v2"
SPRAY_TASK = "pre-rinse-plates"
AUDITED = "scrub-rinse-pan"

TASKS = deepcopy(V1.TASKS)
TASKS[SPRAY_TASK]["instruction"] = LEGACY_TASKS[SPRAY_TASK]["instruction"]

# The eight exact selections, as benchmark.json task_set lists them; the pan's is splits/test.json's.
SELECTIONS = tuple(dict(preset=V1.TASKS[t]["preset"], task=t, seed=0, contract=CONTRACT) for t in EIGHT if t in REVISED) \
    + (dict(preset="pot-sink", task=AUDITED),)


def resolve(preset=None, task=None, seed=0, contract=None):
    """A v2 selection; any other contract (or none) resolves as the released catalogue does."""
    if contract != CONTRACT:
        return V1.resolve(preset, task, seed, contract)
    v = V1.resolve(preset, task, seed)              # the released preset/task/seed checks and defaults
    if v["task"] not in REVISED:
        raise ValueError(f"{v['task']} has no {CONTRACT} contract; it covers {sorted(REVISED)}")
    return dict(preset=v["preset"], task=v["task"], seed=v["seed"], contract=CONTRACT)


def instruction(v):
    return TASKS[v["task"]]["instruction"] if v.get("contract") == CONTRACT else V1.instruction(v)


def task_resolve(selection):
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0), selection.get("contract"))
    if v.get("contract") != CONTRACT:
        return V1.task_resolve(selection)
    return dict(task_id=f"surface-rinsing/v1/{v['preset']}/{v['task']}/{CONTRACT}/seed-{v['seed']}",
                description=instruction(v), backend_options=dict(v), selection=v, split=split_of(v["task"], v["seed"]))


def contract():
    out = V1.contract()
    out["contracts"] = [*out["contracts"], CONTRACT]
    out["task_set_v2"] = [dict(s) for s in SELECTIONS]
    return out


def setup(task, seed=0, contract=None):
    if contract != CONTRACT:
        return V1.setup(task, seed, contract)
    v = resolve(task=task, seed=seed, contract=contract)
    spec = V1.setup(task, seed, V1.CONTRACT)        # the revised task's scene, as under v1
    spec["instruction"] = instruction(v)
    spec["task_contract"] = CONTRACT
    return spec


def progress(task, start, m, criteria):
    """task_set.scoring.progress, with pre-rinse-plates' cleaning credit multiplied by the spray temperature grade (its
    folder's task_set_v2_progress)."""
    from .layout import task_verifier
    own = getattr(task_verifier(task), "task_set_v2_progress", None)
    if own is None:
        return v1_progress(task, start, m, criteria)
    return own(start, m, criteria)


def score(task, start, m, criteria, quantities=None):
    led = m["ledger"]
    closed = led["relative"] < 1e-9 and all(
        led[r] < 1e-9 or abs(led[a]) < 1e-12
        for r, a in (("relative_surf", "surfactant_kg"), ("relative_solids", "solids_kg")))
    quantities = m if quantities is None else quantities
    verifiable = m["fault"] is None and closed and finite(m) and finite(start) and finite(quantities)
    rw = dict(contract=CONTRACT, score_rule="1 if success else 0.9 x progress; None if unverifiable",
              verifiable=verifiable)
    if not verifiable:
        rw.update(progress=None, unverifiable_reason="fault, open conservation ledger or non-finite measured state")
        return None, rw
    rw.update(progress(task, start, quantities, criteria))
    return (1.0 if all(c["passed"] for c in criteria.values()) else FAIL_SCALE*rw["progress"]), rw

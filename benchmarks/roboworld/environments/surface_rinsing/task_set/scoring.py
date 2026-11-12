"""Continuous task progress from measured state, with no simulator imports.

The pan's own evaluator (task.py) is deliberately not routed through this module.
Placement/release are final-stage gates, as in the pan's progress-v1 contract; cleaning
and draining are graded quantities, never a fraction of Boolean checks.
"""
import math
from types import SimpleNamespace

from .catalogue import CONTRACT

FAIL_SCALE = 0.9


def clip(x):
    return max(0.0, min(1.0, x))


def reduction(start, value, limit):
    """Log reduction of a quantity that was above its limit at reset."""
    if start <= limit:
        return 0.0
    return clip(math.log(start / max(value, 1e-30)) / math.log(start / limit))


def excess(value, limit):
    return clip((value - limit) / limit)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def progress(task, start, m, criteria):
    """Return progress and its terms. All units match Evaluator.measure reports. The formula is the task's
    task_set_progress (tasks/<task>/verifier.py): it gets the measures and the terms below as attributes of one
    argument and returns (value, terms, collateral units)."""
    from ..layout import task_verifier
    passed = lambda name: float(criteria[name]["passed"])
    items, initial = m["items"], start["items"]
    cont = m["contamination"]
    released = passed("water_off_and_hands_free")
    units = 0.0
    floor = excess(cont["floor_water_mL"], 50.0)
    soil = lambda name: excess(cont.get(name, {}).get("gauge_max_ug_cm2", 0.0), 1.0)
    shelf = excess(cont.get("dunnage_shelf", {}).get("water_g", 0.0), 1.0)

    own = getattr(task_verifier(task), "task_set_progress", None)     # the task's formula (its verifier.py)
    if own is None:
        raise ValueError(f"No task-set progress for {task}")
    value, terms, units = own(SimpleNamespace(start=start, m=m, criteria=criteria, passed=passed, items=items,
                                              initial=initial, cont=cont, released=released, units=units, floor=floor,
                                              soil=soil, shelf=shelf))
    return dict(progress=clip(value), terms=terms, collateral_units=units)


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

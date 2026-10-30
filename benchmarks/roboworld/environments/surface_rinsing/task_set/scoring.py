"""Continuous task progress from measured state, with no simulator imports.

The pan's own evaluator (task.py) is deliberately not routed through this module.
Placement/release are final-stage gates, as in the pan's progress-v1 contract; cleaning
and draining are graded quantities, never a fraction of Boolean checks.
"""
import math

from .catalogue import CONTRACT, NEW_TASK

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
    """Return progress and its terms. All units match Evaluator.measure reports."""
    passed = lambda name: float(criteria[name]["passed"])
    items, initial = m["items"], start["items"]
    cont = m["contamination"]
    released = passed("water_off_and_hands_free")
    units = 0.0
    terms = {}
    floor = excess(cont["floor_water_mL"], 50.0)
    soil = lambda name: excess(cont.get(name, {}).get("gauge_max_ug_cm2", 0.0), 1.0)
    shelf = excess(cont.get("dunnage_shelf", {}).get("water_g", 0.0), 1.0)

    if task == "rinse-sponge":
        s = reduction(start["sponge"]["c_g_L"], m["sponge"]["c_g_L"], 0.02)
        w = clip((1.0 - m["sponge"]["fill"]) / 0.40)
        p = passed("sponge_on_ledge") * released
        value = (s + s*w + s*w*p)/3.0
        terms = dict(rinsed=s, wrung=w, returned=p)
    elif task == NEW_TASK:
        v0 = start["pools"]["wash"]["volume_L"]
        d = clip((v0 - m["pools"]["wash"]["volume_L"])/(v0 - 0.5))
        k = passed("waste_closed") * released
        # Cumulative physical drain accounts prevent a wrong-bowl drain/refill
        # from erasing collateral. Volume loss also catches scooping/overflow.
        lost = sum(max(0.0, start["pools"][n]["volume_L"] - m["pools"][n]["volume_L"],
                       m["task_set"]["drained_L"][n] - start["task_set"]["drained_L"][n])
                   for n in ("rinse", "sanitize"))
        units = lost/v0 + floor
        value = (d + d*k)/2.0 - units
        terms = dict(drained=d, closed_released=k, other_water_lost_L=lost)
    elif task == "rinse-tomatoes":
        names = [n for n in initial if n.startswith("tomato_")]
        kept = criteria["none_lost"]["detail"]
        credits = {n: reduction(initial[n]["soil_gauge_max_ug_cm2"], items[n]["soil_gauge_max_ug_cm2"], 1.0)
                      if kept[n] else 0.0 for n in names}
        c = sum(credits.values())/len(names)
        p = passed("placed") * passed("drained") * released
        units = (sum(not kept[n] for n in names) + soil("cutting_board") + soil("clean_bowl") + floor)/len(names)
        value = (c + c*p)/2.0 - units
        terms = dict(clean=credits, mean_clean=c, placed_drained=p)
    elif task == "flush-floor":
        # A binary share of clean gauges jumps at the threshold. Area-weighted
        # log reductions keep partial cleaning visible within each 1 cm gauge.
        g0, g = start["task_set"]["floor_gauges"], m["task_set"]["floor_gauges"]
        area = start["task_set"]["floor_areas"]
        dirty = [i for i, x in enumerate(g0) if x > 10.0]
        a = sum(area[i]*reduction(g0[i], g[i], 10.0) for i in dirty)/sum(area[i] for i in dirty) if dirty else 0.0
        k = passed("valve_parked") * released
        units = shelf
        value = (a + a*k)/2.0 - units
        terms = dict(clean_area=a, docked_released=k)
    elif task == "rinse-glassware":
        names = [n for n in initial if n.startswith("glass_") and n != "glass_rack"]
        bad = criteria["intact_and_not_dropped"]["detail"]
        damaged = set(bad["impacts_over_gate"]) | set(bad["on_floor"])
        credits = {}
        for n in names:
            r = reduction(initial[n]["surfactant_ug"], items[n]["surfactant_ug"], 10.0)
            z = clip(items[n]["sanitized_s"]/30.0)
            # A detergent patch/foam cannot receive the final storage stage.
            clean = items[n]["surfactant_gauge_max_mg_m2"] <= 1.0 and items[n]["foam_mL"] < 0.001
            t = float(criteria["placed"]["detail"][n] and items[n]["free_water_mL"] <= 2.0 and clean) * released
            credits[n] = 0.0 if n in damaged else (r + r*z + r*z*t)/3.0
        units = (len(damaged) + soil("drainboard_clean") + floor)/len(names)
        value = sum(credits.values())/len(names) - units
        terms = dict(glasses=credits)
    elif task == "pre-rinse-plates":
        names = [n for n in initial if n.startswith("plate_")]
        credits = {n: reduction(initial[n]["soil_gauge_max_ug_cm2"], items[n]["soil_gauge_max_ug_cm2"], 1.0)
                   for n in names}
        c = sum(credits.values())/len(names)
        k = passed("valve_parked") * released
        units = (sum(not v for v in criteria["placed"]["detail"].values()) + soil("clean_table") + floor)/len(names)
        value = (c + c*k)/2.0 - units
        terms = dict(clean=credits, mean_clean=c, docked_released=k)
    elif task == "rinse-garbage-can":
        r = reduction(initial["can"]["surfactant_ug"], items["can"]["surfactant_ug"], m["task_set"]["can_soap_limit_ug"])
        d = reduction(initial["can"]["soil_gauge_max_ug_cm2"], items["can"]["soil_gauge_max_ug_cm2"], 1.0)
        c = (r + d)/2.0
        e = passed("emptied_upright")
        k = passed("valve_parked") * released
        units = shelf
        value = (c + c*e + c*e*k)/3.0 - units
        terms = dict(rinsed=r, debris_removed=d, clean=c, emptied_upright=e, docked_released=k)
    else:
        raise ValueError(f"No task-set progress for {task}")
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

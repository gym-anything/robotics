"""Outcome of dump-and-pour: at most 2 mL of the old red wine left in the tasting glass and 42-58 mL of the fresh white
wine in it, the glass upright, nothing spilt outside the bucket and the glasses, no drips running down the bottle, the
bottle upright back on its coaster and never touched against the glass, the arms still and released and the wine at
rest; under wine-tasks-v1 and -v2 the old wine must also have gone into the dump bucket.

verify(e) receives the episode's evaluator (task.Evaluator, or task_contracts.Evaluator under the task-set contracts)
and returns its evaluation (Evaluator.evaluate_checks): checks(e, x) below, and under wine-tasks-v1 and -v2
contract_checks(e, x) and the continuous progress(e, x, t, upright)."""
from wine_pouring import params as P
from wine_pouring.task_contracts import clip01, unspilled_fraction, volume_progress


def verify(e):
    return e.evaluate_checks(checks, contract_checks=contract_checks, progress=progress)


def checks(e, x):
    g, checks, vol = x.g, x.checks, x.vol
    t = g["target"]
    checks["old_dumped"] = vol(t, g["old"]) <= g["old_max"]*1e6
    checks["volume"] = abs(vol(t, g["fresh"]) - g["volume"]*1e6) <= g["tol"]*1e6
    checks["target_upright"] = e._upright(t) <= P.GLASS_UPRIGHT_DEG
    e.meta.update(fresh_mL=vol(t, g["fresh"]), old_mL=vol(t, g["old"]), dump_mL=vol(g["dump"]))


def contract_checks(e, x):
    """The checks under wine-tasks-v1 and -v2: checks(e, x) and the old wine in the dump bucket (all of it the glass
    held at reset, less old_max and the spill allowance); sets x.old0, the old wine in the glass at reset."""
    g, checks, vol = x.g, x.checks, x.vol
    t = g["target"]
    checks["old_dumped"] = vol(t, g["old"]) <= g["old_max"]*1e6
    old0 = e.initial_containers.get(t, {}).get(g["old"], 0.)
    checks["old_in_bucket"] = vol(g["dump"], g["old"]) >= old0 - g["old_max"]*1e6 - P.SPILL_TOL_ML
    e.meta.update(old_initial_mL=old0, old_in_bucket_mL=vol(g["dump"], g["old"]))
    checks["volume"] = abs(vol(t, g["fresh"]) - g["volume"]*1e6) <= g["tol"]*1e6
    checks["target_upright"] = e._upright(t) <= P.GLASS_UPRIGHT_DEG
    e.meta.update(fresh_mL=vol(t, g["fresh"]), old_mL=vol(t, g["old"]), dump_mL=vol(g["dump"]))
    x.old0 = old0


def dump_progress(old_initial, old_in_glass, old_in_bucket, old_lost, fresh, fresh_lost, upright):
    # Count only liquid that reached the requested destination. Moving old wine to
    # the source or coaster must not receive disposal credit through an empty glass.
    removed = clip01((old_initial - old_in_glass)/old_initial) if old_initial > 0 else 0.
    in_bucket = clip01(old_in_bucket/old_initial) if old_initial > 0 else 0.
    dumped = min(removed, in_bucket)*unspilled_fraction(old_in_bucket, old_lost)
    filled = volume_progress(fresh, 42., 58.) if upright else 0.
    efficiency = unspilled_fraction(fresh, fresh_lost)
    return dict(progress=.4*dumped + .6*filled*efficiency, dump_progress=dumped,
                served_share=filled, unspilled_fraction=efficiency)


def progress(e, x, t, upright):
    g, inv, vol, old0 = x.g, x.inv, x.vol, x.old0
    # Film is already classified as outside liquid: do not count it twice.
    lost_liquid = lambda liq: sum(inv[k].get(liq, 0.) for k in ("bar", "floor", "elsewhere")) + vol("coaster", liq)
    terms = dump_progress(old0, vol(t, g["old"]), vol(g["dump"], g["old"]), lost_liquid(g["old"]),
                          vol(t, g["fresh"]), lost_liquid(g["fresh"]), upright)
    return terms

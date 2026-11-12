"""Outcome of carry-glass: the glass upright on its delivery spot (within the lit ring) with its 150 mL of wine still in
it (less the spill allowance and 0.5 mL), nothing spilt, the arms still and released and the wine at rest.

verify(e) receives the episode's evaluator (task.Evaluator, or task_contracts.Evaluator under the task-set contracts)
and returns its evaluation (Evaluator.evaluate_checks): checks(e, x) below under every contract and, under
wine-tasks-v1 and -v2, the continuous progress(e, x, t, upright): the wine kept times the way covered to the spot.
contract_start(e) reads the glass's distance from its spot at reset for that progress (task_contracts.Evaluator)."""
import numpy as np

from wine_pouring import params as P
from wine_pouring.task_contracts import clip01


def verify(e):
    return e.evaluate_checks(checks, progress=progress)


def contract_start(e):
    env = e.env
    g = env.layout["goal"]
    e.initial_distance = float(np.linalg.norm(
        e._xy(g["target"])[0] - np.asarray(P.DELIVERY_SPOTS[g["deliver_spot"]])))


def checks(e, x):
    g, checks, vol = x.g, x.checks, x.vol
    t = g["target"]
    checks["kept"] = vol(t) >= g["keep"]*1e6 - P.SPILL_TOL_ML - 0.5
    checks["target_upright"] = e._upright(t) <= P.GLASS_UPRIGHT_DEG
    checks["target_on_spot"] = float(np.linalg.norm(e._xy(t)[0] - np.asarray(P.DELIVERY_SPOTS[g["deliver_spot"]]))) <= P.SPOT_RADIUS_M


def carry_progress(kept, required, upright, distance, initial_distance):
    remaining = max(0., distance - P.SPOT_RADIUS_M)
    travel = max(0., initial_distance - P.SPOT_RADIUS_M)
    delivered = clip01(1. - remaining/travel) if travel > 0 else 0.
    retained = clip01(kept/required) if upright and required > 0 else 0.
    return dict(progress=retained*delivered, retained_fraction=retained, delivery_progress=delivered)


def progress(e, x, t, upright):
    g, vol = x.g, x.vol
    distance = float(np.linalg.norm(e._xy(t)[0] - np.asarray(P.DELIVERY_SPOTS[g["deliver_spot"]])))
    terms = carry_progress(vol(t), g["keep"]*1e6 - P.SPILL_TOL_ML - .5,
                           upright, distance, e.initial_distance)
    return terms

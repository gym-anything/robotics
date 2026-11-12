"""Opt-in task instructions and evaluation, outside provenance.PHYSICS.

The wine-checks-v1 evaluator (task.py) is inherited for every other selection and for split-bottle.
Only the explicit task-set contract wine-tasks-v1 uses these checks, progress and settled read. A benchmark task's own
checks and progress terms under it are in its verifier.py (tasks/<task>/), which passes them to
Evaluator.evaluate_checks; the five fill tasks share task.fill_checks and fill_progress below.
"""
import numpy as np

from . import params as P
from .contracts import TASKS_V1_REWARD_CONTRACT, reward_contract
from .task_set_v2 import TASK_SET_CONTRACTS          # wine-tasks-v1 and wine-tasks-v2 (task_set_v2.py) judge alike
from .layout import task_verifier
from .task import (Evaluator as LegacyEvaluator, FAIL_SCORE_SCALE, INSTRUCTIONS,
                   LINED_WINDOW_ML, SETTLE_HOLD_MAX_S, SPOT_NAMES, VERIFY_EXTRA_STATE, WINE_NAMES, Reading)


TASKS_V1_INSTRUCTIONS = {
    **INSTRUCTIONS,
    "pour-glass": "Pour a 150 mL glass of {wine} from the bottle on the coaster into the Bordeaux glass at {spot}, "
                  "without spilling or leaving drips running down the bottle, then stand the bottle back on its "
                  "coaster and leave the glass upright on its spot.",
    "refill-glass": "Top up the guest's glass at {spot} to 150 mL of {wine}; no spills or drips; bottle back on its "
                    "coaster.",
    "water-tumbler": "Fill the water tumbler at {spot} from the carafe with about 250 mL of water; no spills; put "
                     "the carafe back where it stood.",
    "dump-and-pour": "Empty the leftover {old} from the tasting glass into the dump bucket, then pour a 50 mL "
                     "tasting measure of {wine} into it; spill nothing outside the bucket and glasses; bottle back "
                     "on its coaster.",
    "lined-measure": "Pour a legal 175 mL measure into the lined glass at {spot}: at least to its etched line, "
                     "never short; no spills or drips; bottle back on its coaster.",
}


def clip01(value):
    return max(0., min(1., float(value)))


def volume_progress(got, low, high, initial=0.):
    """Net fill toward the acceptance window; excess costs mL for mL. Units: mL."""
    span = low - initial
    return clip01((min(got, low) - initial - max(0., got - high))/span) if span > 0 else 0.


def unspilled_fraction(served, lost):
    served, lost = max(0., served), max(0., lost)
    return served/(served + lost) if served + lost > 0 else 1.


def fill_progress(e, x, t, upright):
    """The progress terms of the five fill tasks (task.fill_checks): the net fill toward the volume window, while the
    target stands upright, times the share of the poured liquid that reached it."""
    g, got, src_liq, lost = x.g, x.got, x.src_liq, x.lost
    low, high = (LINED_WINDOW_ML if "line" in g else
                 ((g["volume"] - g["tol"])*1e6, (g["volume"] + g["tol"])*1e6))
    initial = e.initial_containers.get(t, {}).get(src_liq, 0.)
    filled = volume_progress(got, low, high, initial) if upright else 0.
    efficiency = unspilled_fraction(max(0., got - initial), lost)
    terms = dict(progress=filled*efficiency, served_share=filled,
                 unspilled_fraction=efficiency, target_initial_mL=initial)
    return terms


def instruction(config, meta=None):
    from .layout import task_layout
    lay = task_layout(config)
    g = lay["goal"]
    src = next((i for i in lay["items"] if i.get("role") == "source"), None) or \
        next((i for i in lay["items"] if i.get("liquid")), {})
    spot = g.get("spot") or g.get("deliver_spot")
    spot_b = None
    if "targets" in g and g["targets"] and config.task == "split-bottle":
        spots = [i.get("spot") for i in lay["items"] if i["name"] in g["targets"]]
        spot, spot_b = spots[0], spots[1]
    templates = (TASKS_V1_INSTRUCTIONS if reward_contract(config.task, getattr(config, "task_contract", "")) in
                 TASK_SET_CONTRACTS else INSTRUCTIONS)
    return templates[config.task].format(wine=WINE_NAMES.get(g.get("fresh") or src.get("liquid"), "wine"),
                                            old=WINE_NAMES.get(g.get("old"), ""), spot=SPOT_NAMES.get(spot, ""),
                                            spot_b=SPOT_NAMES.get(spot_b, ""))


def save_extra_state(env):
    """VERIFY_EXTRA_STATE's save for every scene the eight-task contract reaches. A scene without a bottle
    (carry-glass, water-tumbler) has a disabled GasModel: its __init__ returns before it creates the event and history
    logs the save reads (gas.py), so the plain save would raise AttributeError and GET /v1/evaluation of any such
    state not yet at rest would close the connection without a reply. A disabled model never updates and applies no
    force, so there is nothing of it to save: the save reads the world as if it had no gas model, and the restore then
    leaves the gas model alone. Scenes with a bottle use VERIFY_EXTRA_STATE's save unchanged."""
    save = VERIFY_EXTRA_STATE[0]
    w = env.world
    gas = getattr(w, "gas", None)
    if gas is None or getattr(gas, "enabled", True):
        return save(env)
    w.gas = None
    try:
        return save(env)
    finally:
        w.gas = gas


class Evaluator(LegacyEvaluator):
    def __init__(self, env):
        super().__init__(env)
        self.initial_containers = {}
        self.initial_distance = None
        if reward_contract(env.config.task, getattr(env.config, "task_contract", "")) in TASK_SET_CONTRACTS:
            # Read the actual filled/settled reset state, not the requested layout
            # volume: particle quantisation makes these different by a few uL.
            self.initial_containers = {n: dict(v) for n, v in env.inventory()["containers"].items()}
            start = getattr(task_verifier(env.config.task), "contract_start", None)
            if start is not None:           # what the task's verifier.py reads at reset under the task-set contract
                start(self)

    def evaluate_checks(self, task_checks=None, finish=None, *, contract_checks=None, progress=None):
        """The evaluation under the selected contract. Under wine-tasks-v1 and -v2 the task's hooks apply:
        contract_checks(e, x) in place of task_checks(e, x) (x: task.Reading) and progress(e, x, t, upright), the
        continuous progress terms; finish is the released report's (task.Evaluator.evaluate_checks)."""
        env = self.env
        w = env.world
        lay = env.layout
        g = lay["goal"]
        task = env.config.task
        contract = reward_contract(task, getattr(env.config, "task_contract", ""))
        if contract not in TASK_SET_CONTRACTS:
            out = super().evaluate_checks(task_checks, finish)
            if "reward" not in out:         # split-bottle's report carries its own (split-progress-v1)
                out["reward"] = dict(contract=contract, mean_of_checks=out["score"])
            return out
        inv = env.inventory()
        c, checks = inv["containers"], {}
        vol = lambda name, liq=None: (c.get(name, {}).get(liq, 0.) if liq else c.get(name, {}).get("total", 0.))
        spilled = inv["bar"].get("total", 0.) + inv["floor"].get("total", 0.) + inv["elsewhere"].get("total", 0.)
        lost = spilled + vol("coaster")
        src = next((i["name"] for i in lay["items"] if i.get("role") == "source"), None)
        robot_contact = env.robot_contacts()
        still = env.robot_still_s() >= P.SETTLE_STILL_S
        # settled: the liquid the task is about (targets, and anything outside a container) has stopped moving, and
        # nothing anywhere is still flowing (max speed). The mean over every particle let the noise floor of a deep
        # bottle column (10-15 mm/s, E0 soak) decide the check at its 15 mm/s threshold.
        # (Spilt liquid is not counted: a spill runs across the bar and into the coaster dish for a second or more
        # before contact-line friction stops it; the spill check already fails. The max-speed limit catches liquid
        # sloshing in a target or still in flight anywhere.)
        targets = [n for n in (g.get("targets") or [g.get("target")]) if n] + ([g["dump"]] if g.get("dump") else [])
        means = [inv.get("speed_mean_by_container_m_s", {}).get(n, 0.) for n in targets]
        vmax = max([inv.get("speed_max_by_container_m_s", {}).get(n, 0.) for n in targets] +
                   [inv.get("speed_max_airborne_m_s", 0.)])
        settled = max(means or [inv["speed_mean_m_s"]]) <= P.SETTLE_LIQUID_SPEED and vmax <= 0.25
        self.meta.update(settle_target_mean_mm_s=round(1e3*max(means or [inv["speed_mean_m_s"]]), 2),
                         settle_vmax_mm_s=round(1e3*vmax, 1))
        checks["settled"] = bool(still and settled)
        checks["released"] = not any(robot_contact.values())
        liquid_of = lambda name: next((i.get("liquid") for i in lay["items"] if i["name"] == name), None)
        src_liq = liquid_of(src) if src else None
        x = Reading(env=env, w=w, lay=lay, g=g, inv=inv, c=c, checks=checks, vol=vol, spilled=spilled, lost=lost,
                    src=src, src_liq=src_liq)
        (task_checks if contract_checks is None else contract_checks)(self, x)     # the task's own (its verifier.py)
        checks["no_spill"] = lost <= P.SPILL_TOL_ML
        if src:
            film = inv["film"].get(src, 0.)
            checks["no_drip"] = film <= P.DRIP_TOL_ML + 0.1
            checks["source_upright"] = self._upright(src) <= P.GLASS_UPRIGHT_DEG
            if any(i["name"] == "coaster" for i in lay["items"]):
                checks["source_on_coaster"] = float(np.linalg.norm(self._xy(src)[0] - self._xy("coaster")[0])) <= 0.02
            else:
                home = next(i["xy"] for i in lay["items"] if i["name"] == src)
                checks["source_returned"] = float(np.linalg.norm(self._xy(src)[0] - np.asarray(home))) <= 0.03
            touch = getattr(env, "source_touch", {"max_n": 0., "ticks": 0})
            checks["never_touched_glass"] = touch["ticks"] == 0
            self.meta.update(drip_film_mL=film, source_touch_max_n=touch["max_n"], source_touch_ms=touch["ticks"])
        if w.fault:
            checks = {k: False for k in checks}
        self.meta.update(spilled_mL=spilled, robot_contacts=robot_contact)
        score = float(np.mean([bool(v) for v in checks.values()])) if checks else 0.
        success = bool(checks) and all(checks.values())
        out = dict(success=success, score=score, checks={k: bool(v) for k, v in checks.items()},
                   meta={k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.meta.items()},
                   inventory_mL=dict(containers={k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in c.items()},
                                     spilled=round(spilled, 2), film={k: round(v, 2) for k, v in inv["film"].items()}))
        t = g["target"]
        upright = self._upright(t) <= P.GLASS_UPRIGHT_DEG
        terms = progress(self, x, t, upright)                # the task's progress terms (its verifier.py)
        progress = 0. if w.fault else terms["progress"]
        out["score"] = 1. if success else FAIL_SCORE_SCALE*progress
        out["reward"] = dict(terms, contract=contract, progress=progress, mean_of_checks=score)
        out["meta"].update(lost_mL=round(lost, 3), coaster_mL=round(vol("coaster"), 3))
        return out

    def verify(self):
        """The final verdict (the backend's evaluate(): gar verify, GET /v1/evaluation, the benchmark runner). Every task
        with the legacy checks contract: evaluate(), unchanged. split-bottle and the eight-task contract: a state not
        yet at rest is judged once it is at rest. On a
        branch the world is advanced with the robot holding its last command (env.act({}): joint and grip targets
        unchanged; the motion generator finishes its run-on) until `settled` holds or SETTLE_HOLD_MAX_S has passed, the
        state is evaluated there, and the live state is restored (env.checkpoint/env.restore, as reset() uses). A state
        the hold changes is judged as it ends up: a bottle still in the hand stays held and fails `released`, wine in
        flight lands where it lands, a touch during the hold counts. A failed verdict's progress is the smaller of the
        progress at the request and after the hold: the hold may reveal damage, never add service. The bookkeeping
        env.checkpoint() does not hold is restored too (VERIFY_EXTRA_STATE). A faulted branch returns the live state's
        evaluation marked read.infrastructure_fault. The step reward stays the
        instantaneous evaluate(). Precedent: surface_rinsing/task.py's settle branch."""
        env = self.env
        if reward_contract(env.config.task, getattr(env.config, "task_contract", "")) not in TASK_SET_CONTRACTS:
            return super().verify()
        ev = self.evaluate()
        read = dict(settle_hold_s=0., stopped="already at rest", robot_still_s_at_request=round(env.robot_still_s(), 3),
                    infrastructure_fault=None)
        if not ev["checks"].get("settled") and not env.world.fault:
            at_request = ev
            snap, callback = env.checkpoint(), env.trace_callback
            restore_extra = VERIFY_EXTRA_STATE[1]
            extra = save_extra_state(env)
            env.trace_callback = None                 # branch ticks are not part of any recorded action trace
            held, chunk, read["stopped"] = 0., 100, "time limit"
            try:
                while held < SETTLE_HOLD_MAX_S - 1e-9:
                    env.act({}, ticks=chunk)
                    held += chunk*env.config.rigid_dt
                    if env.world.fault:
                        raise RuntimeError(env.world.fault)
                    if env.robot_still_s() >= P.SETTLE_STILL_S and self.evaluate()["checks"]["settled"]:
                        read["stopped"] = "settled"
                        break
                ev = self.evaluate()
            except (RuntimeError, FloatingPointError) as exc:
                read["stopped"] = f"branch faulted: {exc}"
                read["infrastructure_fault"] = str(exc)
            finally:
                env.trace_callback = callback
                env.restore(snap)
                restore_extra(env, extra)
            read["settle_hold_s"] = round(held, 3)
            if read["infrastructure_fault"]:
                ev = at_request                       # the live state as it is (it fails `settled`): not a task verdict
            elif not ev["success"]:
                p_req, p_hold = at_request["reward"]["progress"], ev["reward"]["progress"]
                read.update(progress_at_request=p_req, progress_after_hold=p_hold)
                if p_req < p_hold:
                    ev["reward"].update(progress=p_req, progress_source="request")
                    ev["score"] = at_request["score"]  # FAIL_SCORE_SCALE x the request's progress (it failed `settled`)
                else:
                    ev["reward"]["progress_source"] = "after_hold"
        ev["read"] = read
        return ev


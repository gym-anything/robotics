"""Task instructions and the environment-owned evaluator.

The evaluator reads only simulated state: the liquid inventory (inventory.py), glassware poses and contacts, and the
robot's motion. It never changes the simulation. A task succeeds only when the scene is settled: the arms still, no
liquid in flight, glassware released and upright where it belongs.
"""

import math
from types import SimpleNamespace

import numpy as np

from . import params as P
from .layout import task_verifier

LINED_WINDOW_ML = (175.0, 185.0)   # UK SI 1988/120 reg. 14: a 175 mL measure may not be short (the line's capacity is
                                   # the meniscus at the line); up to 10 mL over is accepted
SPOT_NAMES = {1: "spot 1 (far left)", 2: "spot 2", 3: "spot 3", 4: "spot 4 (far right)"}
WINE_NAMES = {"red_wine": "red wine", "white_wine": "white wine", "rose_wine": "rose wine", "water": "water"}
# Reward contract "split-progress-v1": score = 1 on success, else FAIL_SCORE_SCALE x progress, from the
# evaluated state only (no order, route or intermediate milestone): progress = S x E (split_progress in
# tasks/split-bottle/verifier.py, with the rest of split-bottle's checks and their documentation). Every other task
# scores the fraction of its checks met; every split-bottle evaluation also reports it as reward.mean_of_checks.
SPLIT_REWARD_CONTRACT = "split-progress-v1"
FAIL_SCORE_SCALE = .9          # a failed state scores 0.9 x progress: every failure stays below every success
SETTLE_HOLD_MAX_S = 3.0        # split-bottle's verified read: the arm runs on ~0.3 s past its command, then 2 s still
                               # (only a final command of ~2 rad or more on a joint in under ~0.3 s outruns it; such
                               # a scene is not at rest at the cap and is judged as it is then)


def _verify_extra_state(env):
    """State a verify() branch changes that env.checkpoint() does not hold: the headspace gas model's clocks (a restore
    otherwise leaves them at the branch's end, so the next tick sees about -3 s elapse and a sealed bottle's bubble
    entry, n_atm += q_air*dt, runs backwards), the held-load estimate's previous velocities (World._prev_cvel: a stale value is a spurious
    acceleration on the next tick), and the diagnostic counters and logs (limiter totals, gas events/history, phantom
    log), so the episode continues and reports as if never read."""
    import copy
    w = env.world
    gas = w.gas
    out = dict(prev_cvel=copy.deepcopy(getattr(w, "_prev_cvel", None)),
               phantom_log=len(getattr(w, "phantom_log", None) or []))
    if w.liquid is not None:
        out["limiters"] = dict(w.liquid.limiter_totals)
    if gas is not None:
        out["gas"] = {k: getattr(gas, k) for k in ("_t_last", "_t_prev") if hasattr(gas, k)}
        out["gas_absent"] = [k for k in ("_t_last", "_t_prev") if not hasattr(gas, k)]
        out["gas_logs"] = (len(gas.events), len(gas.history))
    if getattr(w, "lumped", None) is not None:
        lp = w.lumped
        out["lumped_counts"] = {k: getattr(lp, k) for k in ("follow_steps", "follow_particles", "impulse_steps")
                                if hasattr(lp, k)}
    return out


def _restore_extra_state(env, s):
    w = env.world
    w._prev_cvel = s["prev_cvel"]
    if getattr(w, "phantom_log", None) is not None:
        del w.phantom_log[s["phantom_log"]:]
    if "limiters" in s:
        w.liquid.limiter_totals.clear()
        w.liquid.limiter_totals.update(s["limiters"])
    if "gas" in s:
        for k, v in s["gas"].items():
            setattr(w.gas, k, v)
        for k in s["gas_absent"]:
            if hasattr(w.gas, k):
                delattr(w.gas, k)
        del w.gas.events[s["gas_logs"][0]:]
        del w.gas.history[s["gas_logs"][1]:]
    for k, v in s.get("lumped_counts", {}).items():
        setattr(w.lumped, k, v)


VERIFY_EXTRA_STATE = (_verify_extra_state, _restore_extra_state)


def touch_rule(task):
    """(ticks between samples, force threshold N) of the bottle-glass touch monitor (env.act): every 10th tick above
    params.CONTACT_FORCE_TOL_N, unless the task's verifier.py sets its own TOUCH_RULE (split-bottle)."""
    return getattr(task_verifier(task), "TOUCH_RULE", (10, P.CONTACT_FORCE_TOL_N))


INSTRUCTIONS = {
    "pour-glass": "Pour a 150 mL glass of {wine} from the bottle on the coaster into the Bordeaux glass at {spot} "
                  "(135-165 mL), without spilling or leaving drips running down the bottle, then stand the bottle back "
                  "on its coaster and leave the glass upright on its spot.",
    "taste-pour": "Pour the host a tasting measure: 30-45 mL of {wine} into the glass at {spot}; no spills or drips; "
                  "bottle back on its coaster.",
    "split-bottle": "Finish the bottle: share what is left in it equally between the guests' glasses at {spot} and "
                    "{spot_b}, without the bottle touching either glass, and stand the bottle back on its coaster.",
    "water-tumbler": "Fill the water tumbler at {spot} from the carafe with 225-275 mL of water; no spills; carafe back "
                     "where it stood (within 3 cm).",
    "carry-glass": "Carry the glass of {wine} (150 mL) to {spot} without spilling and leave it upright on the spot.",
    "refill-glass": "Top up the guest's glass at {spot} to 150 mL (135-165 mL) of {wine}; no spills or drips; bottle back "
                    "on its coaster.",
    "tasting-flight": "Pour a 50 mL tasting measure (42-58 mL) of {wine} into each of the three ISO tasting glasses; no "
                      "spills or drips; bottle back on its coaster.",
    "decant-carafe": "Decant the whole bottle of {wine} into the carafe, leaving at most 30 mL in the bottle and spilling "
                     "nothing; bottle back on its coaster, carafe upright.",
    "dump-and-pour": "Empty the leftover {old} from the tasting glass into the dump bucket (at most 2 mL left), then pour "
                     "50 mL (42-58 mL) of {wine} into it; nothing spilled outside the bucket and glasses; bottle back on "
                     "its coaster.",
    "lined-measure": "Pour a legal 175 mL measure: fill the lined glass at {spot} at least to its etched 175 mL line "
                     "and no more than 10 mL over (175-185 mL; a short measure fails), no spills or drips; bottle back "
                     "on its coaster.",
}


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
    return INSTRUCTIONS[config.task].format(wine=WINE_NAMES.get(g.get("fresh") or src.get("liquid"), "wine"),
                                            old=WINE_NAMES.get(g.get("old"), ""), spot=SPOT_NAMES.get(spot, ""),
                                            spot_b=SPOT_NAMES.get(spot_b, ""))


class Reading(SimpleNamespace):
    """What Evaluator.evaluate_checks has read of the state, for a task's checks(e, x) and its other hooks: env, w (the
    world), lay (the layout), g (its goal), inv (the inventory), c (its containers), checks (the dict the task adds its
    checks to), vol(name, liquid=None) (mL in a container), spilled and lost (mL; a task may set lost: split-bottle
    counts the coaster dish), src (the source's name) and src_liq (its liquid). A task's checks may leave more on it
    for its later hooks (fill_checks: got)."""


def fill_checks(e, x):
    """The checks of the five fill tasks (pour-glass, taste-pour, refill-glass, water-tumbler, lined-measure; their
    verifier.py passes this to Evaluator.evaluate_checks under every contract): the target's volume of the source's
    liquid within the goal's volume +- tol (a goal at an etched line: LINED_WINDOW_ML), the target upright and on its
    spot."""
    env, g, checks, vol, src_liq = x.env, x.g, x.checks, x.vol, x.src_liq
    t = g["target"]
    target_ml = (g["volume"] if "volume" in g else env.geometry(t).volume_below(env.geometry(t).lines[g["line"]]))*1e6
    got = vol(t, src_liq)
    if "line" in g:
        checks["volume"] = LINED_WINDOW_ML[0] <= got <= LINED_WINDOW_ML[1]
    else:
        checks["volume"] = abs(got - target_ml) <= g["tol"]*1e6
    checks["target_upright"] = e._upright(t) <= P.GLASS_UPRIGHT_DEG
    if g.get("spot"):
        checks["target_on_spot"] = float(np.linalg.norm(e._xy(t)[0] - np.asarray(P.DELIVERY_SPOTS[g["spot"]]))) <= P.SPOT_RADIUS_M
    e.meta.update(target_mL=target_ml, got_mL=got)
    x.got = got


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.meta = {}
        self.source_initial_mL = None       # split-bottle's bottle at reset (its verifier.py: start)
        start = getattr(task_verifier(env.config.task), "start", None)
        if start is not None:               # what the task's verifier.py reads of the state at reset
            start(self)

    def _upright(self, name):
        _, R = self.env.world.body_pose(name)
        return math.degrees(math.acos(max(-1., min(1., float(R[2, 2])))))

    def _xy(self, name):
        p, _ = self.env.world.body_pose(name)
        return np.asarray(p[:2]), float(p[2])

    def evaluate(self):
        """The evaluation of the current state. A benchmark task's verify(e) (tasks/<task>/verifier.py) decides: it
        passes its checks to evaluate_checks. tasting-flight's and decant-carafe's checks are in evaluate_checks."""
        folder = task_verifier(self.env.config.task)
        if folder is not None:
            return folder.verify(self)
        return self.evaluate_checks()

    def evaluate_checks(self, task_checks=None, finish=None, **contract_hooks):
        """The evaluation under the released contracts (wine-checks-v1, split-progress-v1). task_checks(e, x) adds the
        task's own checks to x.checks (x: Reading); finish(e, x, out, success, score) completes the report
        (split-bottle's reward). The keywords are a task's hooks for the task-set contracts
        (task_contracts.Evaluator.evaluate_checks); this evaluation does not use them."""
        env = self.env
        w = env.world
        lay = env.layout
        g = lay["goal"]
        task = env.config.task
        inv = env.inventory()
        c, checks = inv["containers"], {}
        vol = lambda name, liq=None: (c.get(name, {}).get(liq, 0.) if liq else c.get(name, {}).get("total", 0.))
        spilled = inv["bar"].get("total", 0.) + inv["floor"].get("total", 0.) + inv["elsewhere"].get("total", 0.)
        lost = spilled                      # what no_spill counts; split-bottle adds the coaster dish
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
        if task_checks is not None:         # a benchmark task's own checks (its verifier.py)
            task_checks(self, x)
            lost = x.lost
        elif task == "tasting-flight":
            got = {n: vol(n, src_liq) for n in g["targets"]}
            checks["volumes"] = all(abs(v - g["volume"]*1e6) <= g["tol"]*1e6 for v in got.values())
            checks["targets_upright"] = all(self._upright(n) <= P.GLASS_UPRIGHT_DEG for n in g["targets"])
            self.meta.update(got_mL=got)
        elif task == "decant-carafe":
            checks["bottle_emptied"] = vol(src) <= g["leave_max"]*1e6
            checks["target_upright"] = self._upright(g["target"]) <= P.GLASS_UPRIGHT_DEG
            self.meta.update(carafe_mL=vol(g["target"]), left_mL=vol(src))
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
        if finish is not None:              # the task's own report terms (its verifier.py)
            finish(self, x, out, success, score)
        return out

    def verify(self):
        """The final verdict (the backend's evaluate(): gar verify, GET /v1/evaluation, the benchmark runner):
        evaluate(), unless the task's verifier.py reads its final verdict itself (final_verdict: split-bottle's settled
        read)."""
        final = getattr(task_verifier(self.env.config.task), "final_verdict", None)
        if final is not None:
            return final(self)
        return self.evaluate()

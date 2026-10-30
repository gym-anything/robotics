"""Task instructions and the environment-owned evaluator.

The evaluator reads only simulated state: the liquid inventory (inventory.py), glassware poses and contacts, and the
robot's motion. It never changes the simulation. A task succeeds only when the scene is settled: the arms still, no
liquid in flight, glassware released and upright where it belongs.
"""

import math
import numpy as np

from . import params as P

LINED_WINDOW_ML = (175.0, 185.0)   # UK SI 1988/120 reg. 14: a 175 mL measure may not be short (the line's capacity is
                                   # the meniscus at the line); up to 10 mL over is accepted
SPOT_NAMES = {1: "spot 1 (far left)", 2: "spot 2", 3: "spot 3", 4: "spot 4 (far right)"}
WINE_NAMES = {"red_wine": "red wine", "white_wine": "white wine", "rose_wine": "rose wine", "water": "water"}
# split-bottle verifier documentation. The instruction states the job ("finish
# the bottle", "equally", "the guests' glasses", the bottle never touching a glass); these are how it is scored:
#   shared            both glasses hold wine and differ by at most share_tol = 10 % of the fuller glass (layout goal):
#                     about a 2-3.6 mm level difference in the PURE glasses over the seeded 250-450 mL
#   bottle_emptied    at most leave_max = 15 mL left in the bottle (the residue that clings inside a drained bottle)
#   targets_on_spots  spots k and k+1 each hold one of the two glasses (within P.SPOT_RADIUS_M, the lit ring), either
#                     glass on either spot: they are identical, and lifting a glass to pour into it is allowed
#   no_spill          wine outside the bottle and the two glasses - bar, floor, robot, in flight and the coaster dish
#                     (wine drained into the dish is neither spilt nor served otherwise) - at most P.SPILL_TOL_ML
#   never_touched_glass  the bottle never touches either glass: every physics tick, any bottle-glass contact force
#                     (above TOUCH_FLOOR_N, a round-off guard; touch_rule). Contacts are generated up to 2 mm before the
#                     collision hulls meet and carry exactly 0 N until they do (every such tick in the evaluator cases
#                     read 0.0); the lightest real touch seen was 1.5 mN. The collision hulls follow the drawn
#                     glassware to within about 2 mm: up to 1.8 mm inside the PURE bowl at its widest, up to 0.5-0.6 mm
#                     outside at the PURE rim's top edge and the LEONOR lip's rounded corner. So a contact can register
#                     with the drawn outlines about 1 mm apart at the rim, and a graze at the bowl's widest can overlap
#                     them by up to about 1.8 mm first; either way the simulated bodies really touch. (Every other bottle
#                     task counts presses above 0.5 N at every 10th tick: a clink's force lasts about 8 ms and a resting
#                     touch sits at 0.02-0.5 N, so both could pass that rule.)
#   and the common checks: drip film on the bottle, bottle upright on its coaster, glasses upright, arms still 2 s and
#   released, wine at rest.
# The final verdict (backend evaluate(): gar verify, GET /v1/evaluation, the benchmark runner) is Evaluator.verify(): a
# state not yet at rest is advanced on a branch with the robot holding its last command until `settled` holds (at most
# SETTLE_HOLD_MAX_S), evaluated there, and the live state restored, so a correct finish read promptly gets full credit.
# Success and the checks come from the held state; a FAILED verdict's progress is the
# smaller of the progress at the request and after the hold, so the hold can reveal damage (a bottle that falls, a glass
# the arm's run-on knocks, wine poured onto the bar) but never adds service: a bottle frozen mid-pour keeps pouring on
# the branch, and that wine was not served by the agent. The branch is a simulated
# continuation: the GPU liquid is not bitwise repeatable, so its settle statistics vary in the last digits between calls
# on one state (every verdict observed was identical). It restores, beyond env.checkpoint(), the bookkeeping that
# checkpoint does not hold (VERIFY_EXTRA_STATE), so the episode continues as if never read. The per-command step reward
# stays the instantaneous evaluate(): at an episode's last command it can read 0.9 where the verdict is 1.0.
# Reward contract "split-progress-v1": score = 1 on success, else FAIL_SCORE_SCALE x progress, from the
# evaluated state only (no order, route or intermediate milestone): progress = S x E (split_progress). Every other task
# scores the fraction of its checks met; every split-bottle evaluation also reports it as reward.mean_of_checks.
SPLIT_REWARD_CONTRACT = "split-progress-v1"
FAIL_SCORE_SCALE = .9          # a failed state scores 0.9 x progress: every failure stays below every success
TOUCH_FLOOR_N = 1e-6           # split-bottle: any bottle-glass contact force above round-off is a touch
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
    """(ticks between samples, force threshold N) of the bottle-glass touch monitor (env.act)."""
    return (1, TOUCH_FLOOR_N) if task == "split-bottle" else (10, P.CONTACT_FORCE_TOL_N)


def split_progress(v0, va, vb, upright_a, upright_b, lost, leave_max_ml):
    """split-bottle progress from the final state. T = (v0 - leave_max)/2 is each guest's share of the finished bottle
    (v0: the bottle's wine at reset). S = (min(va, T') + min(vb, T'))/2T' credits the wine in each glass up to its
    share T' = max(T, W/2), counted while that glass stands upright: wine beyond the finished-bottle share (up to
    leave_max) is credited only while the glasses stay equal (with T alone, a 122.5/137.5 mL split of a 260 mL bottle
    would fail `shared` at progress 1.0). E = W/(W + lost) is the fraction of the wine that left
    the bottle and reached a glass (W: wine in the two glasses; lost: everywhere else outside the bottle, coaster dish
    included; wine in flight counts as lost until it lands, so mid-pour readings are diagnostic). Wine still in the
    bottle costs only its share; spilt wine also costs E."""
    T = .5*(v0 - leave_max_ml)
    W = max(0., va + vb)
    Tc = max(T, .5*W)
    served = (min(va, Tc) if upright_a else 0.) + (min(vb, Tc) if upright_b else 0.)
    S = max(0., served)/(2*Tc) if Tc > 0 else 0.
    E = W/(W + lost) if W + lost > 0 else 1.
    return dict(progress=S*E, served_share=S, unspilled_fraction=E, share_mL=T, credited_share_mL=Tc)


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


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.meta = {}
        # split-bottle's reward needs the bottle's wine at reset (built after the fill; reset() restores that state)
        self.source_initial_mL = None
        if env.config.task == "split-bottle":
            src = next(i["name"] for i in env.layout["items"] if i.get("role") == "source")
            self.source_initial_mL = env.inventory()["containers"].get(src, {}).get("total", 0.)

    def _upright(self, name):
        _, R = self.env.world.body_pose(name)
        return math.degrees(math.acos(max(-1., min(1., float(R[2, 2])))))

    def _xy(self, name):
        p, _ = self.env.world.body_pose(name)
        return np.asarray(p[:2]), float(p[2])

    def evaluate(self):
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
        if task in ("pour-glass", "taste-pour", "refill-glass", "water-tumbler", "lined-measure"):
            t = g["target"]
            target_ml = (g["volume"] if "volume" in g else env.geometry(t).volume_below(env.geometry(t).lines[g["line"]]))*1e6
            got = vol(t, src_liq)
            if "line" in g:
                checks["volume"] = LINED_WINDOW_ML[0] <= got <= LINED_WINDOW_ML[1]
            else:
                checks["volume"] = abs(got - target_ml) <= g["tol"]*1e6
            checks["target_upright"] = self._upright(t) <= P.GLASS_UPRIGHT_DEG
            if g.get("spot"):
                checks["target_on_spot"] = float(np.linalg.norm(self._xy(t)[0] - np.asarray(P.DELIVERY_SPOTS[g["spot"]]))) <= P.SPOT_RADIUS_M
            self.meta.update(target_mL=target_ml, got_mL=got)
        elif task == "split-bottle":
            a, b = g["targets"]
            va, vb = vol(a, src_liq), vol(b, src_liq)
            up = [self._upright(n) <= P.GLASS_UPRIGHT_DEG for n in (a, b)]
            checks["shared"] = min(va, vb) > 0 and abs(va - vb) <= g["share_tol"]*max(va, vb)
            checks["bottle_emptied"] = vol(src) <= g["leave_max"]*1e6
            checks["targets_upright"] = all(up)
            ka, kb = (next(i["spot"] for i in lay["items"] if i["name"] == n) for n in (a, b))
            on = lambda n, k: float(np.linalg.norm(self._xy(n)[0] - np.asarray(P.DELIVERY_SPOTS[k]))) <= P.SPOT_RADIUS_M
            checks["targets_on_spots"] = (on(a, ka) and on(b, kb)) or (on(a, kb) and on(b, ka))
            lost = spilled + vol("coaster")
            split = split_progress(self.source_initial_mL or 0., va, vb, up[0], up[1], lost, g["leave_max"]*1e6)
            self.meta.update(a_mL=va, b_mL=vb, left_mL=vol(src), coaster_mL=vol("coaster"), lost_mL=lost,
                             robot_still_s=env.robot_still_s())
        elif task == "carry-glass":
            t = g["target"]
            checks["kept"] = vol(t) >= g["keep"]*1e6 - P.SPILL_TOL_ML - 0.5
            checks["target_upright"] = self._upright(t) <= P.GLASS_UPRIGHT_DEG
            checks["target_on_spot"] = float(np.linalg.norm(self._xy(t)[0] - np.asarray(P.DELIVERY_SPOTS[g["deliver_spot"]]))) <= P.SPOT_RADIUS_M
        elif task == "tasting-flight":
            got = {n: vol(n, src_liq) for n in g["targets"]}
            checks["volumes"] = all(abs(v - g["volume"]*1e6) <= g["tol"]*1e6 for v in got.values())
            checks["targets_upright"] = all(self._upright(n) <= P.GLASS_UPRIGHT_DEG for n in g["targets"])
            self.meta.update(got_mL=got)
        elif task == "decant-carafe":
            checks["bottle_emptied"] = vol(src) <= g["leave_max"]*1e6
            checks["target_upright"] = self._upright(g["target"]) <= P.GLASS_UPRIGHT_DEG
            self.meta.update(carafe_mL=vol(g["target"]), left_mL=vol(src))
        elif task == "dump-and-pour":
            t = g["target"]
            checks["old_dumped"] = vol(t, g["old"]) <= g["old_max"]*1e6
            checks["volume"] = abs(vol(t, g["fresh"]) - g["volume"]*1e6) <= g["tol"]*1e6
            checks["target_upright"] = self._upright(t) <= P.GLASS_UPRIGHT_DEG
            self.meta.update(fresh_mL=vol(t, g["fresh"]), old_mL=vol(t, g["old"]), dump_mL=vol(g["dump"]))
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
        if task == "split-bottle":
            progress = 0. if w.fault else split["progress"]
            out["score"] = 1. if success else FAIL_SCORE_SCALE*progress
            out["reward"] = dict(contract=SPLIT_REWARD_CONTRACT, progress=round(progress, 4),
                                 served_share=round(split["served_share"], 4),
                                 unspilled_fraction=round(split["unspilled_fraction"], 4),
                                 share_mL=round(split["share_mL"], 3),
                                 credited_share_mL=round(split["credited_share_mL"], 3),
                                 bottle_initial_mL=round(self.source_initial_mL or 0., 3), mean_of_checks=score)
        return out

    def verify(self):
        """The final verdict (the backend's evaluate(): gar verify, GET /v1/evaluation, the benchmark runner). Every task
        but split-bottle: evaluate(), unchanged. split-bottle: a state not yet at rest is judged once it is at rest. On a
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
        ev = self.evaluate()
        if env.config.task != "split-bottle":
            return ev
        read = dict(settle_hold_s=0., stopped="already at rest", robot_still_s_at_request=round(env.robot_still_s(), 3),
                    infrastructure_fault=None)
        if not ev["checks"].get("settled") and not env.world.fault:
            at_request = ev
            snap, callback = env.checkpoint(), env.trace_callback
            save, restore_extra = VERIFY_EXTRA_STATE
            extra = save(env)
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
                read.update(progress_at_request=p_req, progress_after_hold=p_hold,
                            bottle_mL_at_request=at_request["meta"]["left_mL"], bottle_mL_after_hold=ev["meta"]["left_mL"])
                if p_req < p_hold:
                    ev["reward"].update(progress=p_req, progress_source="request")
                    ev["score"] = at_request["score"]  # FAIL_SCORE_SCALE x the request's progress (it failed `settled`)
                else:
                    ev["reward"]["progress_source"] = "after_hold"
        ev["read"] = read
        return ev

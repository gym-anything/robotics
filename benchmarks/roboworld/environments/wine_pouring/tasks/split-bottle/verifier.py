"""Outcome of split-bottle: what was left in the bottle shared equally between the two guests' glasses (they differ by
at most 10 % of the fuller one), at most 15 mL left in the bottle, the two glasses upright on their two spots (either
glass on either spot), nothing spilt (the coaster dish counts), the bottle never touching a glass, no drips running
down it, the bottle upright back on its coaster, the arms still and released and the wine at rest.

verify(e) receives the episode's evaluator (task.Evaluator, or task_contracts.Evaluator, which keeps the released
scoring for this task) and returns its evaluation (task.Evaluator.evaluate_checks): checks(e, x) below and the
split-progress-v1 reward (finish). start(e) reads the bottle's wine at reset (task.Evaluator); final_verdict(e) is the
task's final verdict, its settled read (task.Evaluator.verify); TOUCH_RULE is how often and above what force the
bottle-glass touch monitor counts a touch (task.touch_rule, env.WineEnv.act)."""
import numpy as np

from wine_pouring import params as P
from wine_pouring.task import FAIL_SCORE_SCALE, SETTLE_HOLD_MAX_S, SPLIT_REWARD_CONTRACT, VERIFY_EXTRA_STATE

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
#                     (above TOUCH_FLOOR_N, a round-off guard; TOUCH_RULE). Contacts are generated up to 2 mm before the
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
# The final verdict (backend evaluate(): gar verify, GET /v1/evaluation, the benchmark runner) is final_verdict(): a
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
TOUCH_FLOOR_N = 1e-6           # split-bottle: any bottle-glass contact force above round-off is a touch
TOUCH_RULE = (1, TOUCH_FLOOR_N)   # every physics tick, any touch (the other tasks: every 10th tick above 0.5 N)


def verify(e):
    return e.evaluate_checks(checks, finish)


def start(e):
    env = e.env
    # split-bottle's reward needs the bottle's wine at reset (built after the fill; reset() restores that state)
    src = next(i["name"] for i in env.layout["items"] if i.get("role") == "source")
    e.source_initial_mL = env.inventory()["containers"].get(src, {}).get("total", 0.)


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


def checks(e, x):
    """The checks of split-bottle (above); sets x.lost (the coaster dish counts as lost) and x.split (split_progress)
    for finish."""
    env, lay, g, checks, vol, spilled, src, src_liq = x.env, x.lay, x.g, x.checks, x.vol, x.spilled, x.src, x.src_liq
    a, b = g["targets"]
    va, vb = vol(a, src_liq), vol(b, src_liq)
    up = [e._upright(n) <= P.GLASS_UPRIGHT_DEG for n in (a, b)]
    checks["shared"] = min(va, vb) > 0 and abs(va - vb) <= g["share_tol"]*max(va, vb)
    checks["bottle_emptied"] = vol(src) <= g["leave_max"]*1e6
    checks["targets_upright"] = all(up)
    ka, kb = (next(i["spot"] for i in lay["items"] if i["name"] == n) for n in (a, b))
    on = lambda n, k: float(np.linalg.norm(e._xy(n)[0] - np.asarray(P.DELIVERY_SPOTS[k]))) <= P.SPOT_RADIUS_M
    checks["targets_on_spots"] = (on(a, ka) and on(b, kb)) or (on(a, kb) and on(b, ka))
    lost = spilled + vol("coaster")
    split = split_progress(e.source_initial_mL or 0., va, vb, up[0], up[1], lost, g["leave_max"]*1e6)
    e.meta.update(a_mL=va, b_mL=vb, left_mL=vol(src), coaster_mL=vol("coaster"), lost_mL=lost,
                  robot_still_s=env.robot_still_s())
    x.lost, x.split = lost, split


def finish(e, x, out, success, score):
    """The split-progress-v1 reward: score 1 on success, else FAIL_SCORE_SCALE x progress."""
    w, split = x.w, x.split
    progress = 0. if w.fault else split["progress"]
    out["score"] = 1. if success else FAIL_SCORE_SCALE*progress
    out["reward"] = dict(contract=SPLIT_REWARD_CONTRACT, progress=round(progress, 4),
                         served_share=round(split["served_share"], 4),
                         unspilled_fraction=round(split["unspilled_fraction"], 4),
                         share_mL=round(split["share_mL"], 3),
                         credited_share_mL=round(split["credited_share_mL"], 3),
                         bottle_initial_mL=round(e.source_initial_mL or 0., 3), mean_of_checks=score)


def final_verdict(e):
    """The final verdict (the backend's evaluate(): gar verify, GET /v1/evaluation, the benchmark runner;
    task.Evaluator.verify calls this): a state not yet at rest is judged once it is at rest. On a
    branch the world is advanced with the robot holding its last command (env.act({}): joint and grip targets
    unchanged; the motion generator finishes its run-on) until `settled` holds or SETTLE_HOLD_MAX_S has passed, the
    state is evaluated there, and the live state is restored (env.checkpoint/env.restore, as reset() uses). A state
    the hold changes is judged as it ends up: a bottle still in the hand stays held and fails `released`, wine in
    flight lands where it lands, a touch during the hold counts. A failed verdict's progress is the smaller of the
    progress at the request and after the hold: the hold may reveal damage, never add service. The bookkeeping
    env.checkpoint() does not hold is restored too (VERIFY_EXTRA_STATE). A faulted branch returns the live state's
    evaluation marked read.infrastructure_fault. The step reward stays the
    instantaneous evaluate(). Precedent: surface_rinsing/task.py's settle branch."""
    env = e.env
    ev = e.evaluate()
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
                if env.robot_still_s() >= P.SETTLE_STILL_S and e.evaluate()["checks"]["settled"]:
                    read["stopped"] = "settled"
                    break
            ev = e.evaluate()
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

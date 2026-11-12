"""Outcome of vial-to-pad: 1 mL drawn from the vial with the fill needle, the needle changed to the 25 G, the dose given
into the muscle of the injection pad at the printed ring, square to the skin, no faster than 10 s per mL (at most
0.3 mL in any 2 s and 0.5 mL in any 4 s) and with no air, the 25 G never in or on the vial and not put back into the
pad after its dose; the vial left upright on the counter or the tray, nothing in it, its pressure near ambient; the
syringe and both needles in the sharps container, no needle resheathed or taken off after it has been in the pad.

criteria(c) adds the task's own criteria to the ones every task has (c: task.Checks); verify(env) evaluates an episode
(env: syringe_handling.env.SyringeEnv, or a stand-in with its world, spec, start record and history) and returns the
evaluation dict (task.evaluate_criteria), with the task's own needle hygiene (_vial_to_pad_hygiene, HYGIENE_NOTE) and
its own score: 1 on success, else vial_to_pad_partial, at most 0.9 (contract syringe-vial-to-pad-v1). The task has no
syringe-tasks-v1 or -v2 progress: it keeps this evaluator."""
import math

import numpy as np

from syringe_handling import hydraulics as H, params as P
from syringe_handling.config import VIAL_TO_PAD_CONTRACT
from syringe_handling.task import (AIR_LIMIT_ML, HOUSEKEEPING_FLOOR, HOUSEKEEPING_STEP, ML, NEEDLE_ANGLE_DEG,
                                   RATE_ZERO_ML, RING_MM, SUSTAINED_ML, _c, _hygiene, evaluate_criteria)

W_DOSE_GIVEN, W_DOSE_DRAWN, W_NEEDLE_CHANGED, W_CLEARED_AWAY = 0.55, 0.15, 0.10, 0.10
SUSTAINED_ZERO_ML = 2.0
SHARPS_PLACES = ("sharps_flap", "sharps_outside", "sharps_inside")
RESTICK_TRACE = 0.05
# what the task's needle hygiene allows beyond the rule every task has
HYGIENE_NOTE = ("; the fill needle also the inside of the vial; the 25 G's point also the prepared skin within "
                "25 mm of the ring's centre, except after its dose was given; the sharps container only once the "
                "needle or nozzle is done with")


def criteria(c):
    C, spec, fl, env = c.C, c.spec, c.fluid, c.env
    c.injection(spec["dose_ml"], "muscle", needle_name="needle_im", whole=True)
    im = fl.needles["needle_im"]
    met = sorted({p for p in im.touch if p == "vial" or p.startswith(("vial:", "vial_stopper"))}
                 | {r["target"] for r in im.records if r["target"] == "vial_stopper"})
    C["injection_needle_fresh"] = _c(not met, met or "never in or on the vial",
                                     "the 25 G needle never entered or touched the vial (the dose is drawn with the fill needle)")
    resticks = _vial_to_pad_sticks(env)[2]
    C["no_restick_after_dose"] = _c(not resticks, [round(r["t"], 3) for r in resticks] or "none",
                                    f"no puncture of the pad by the 25 G after its dose was given (one that starts after the punctures "
                                    f"that carried the dose came out and carries less than {100*RESTICK_TRACE:g} % of the dose)")
    recapped, taken_off = _used_needle_breaches(env)
    C["used_needle_not_recapped"] = _c(not recapped, recapped or "none", "no needle has a sheath put back on after it has been in the pad")
    C["used_needle_kept_on"] = _c(not taken_off, taken_off or "none", "a needle that has been in the pad stays on the syringe until "
                                  "it reaches the sharps container")
    c.vial_after(home=False)
    c.sharps([spec["dose_syringe"], f"{spec['dose_syringe']}_needle", "needle_im"], seated=True)


def own_score(env, C, out):
    """The task's score in out (evaluate_criteria): 1 on success, else vial_to_pad_partial's, at most 0.9."""
    out["score_parts"] = vial_to_pad_partial(env, C)
    out["score"] = 1.0 if out["success"] else out["score_parts"]["partial"]
    out["contract"] = VIAL_TO_PAD_CONTRACT


def verify(env):
    return evaluate_criteria(env, criteria, hygiene=_vial_to_pad_hygiene, hygiene_note=HYGIENE_NOTE,
                             own_score=own_score)


def _site_quality(r):
    """1 inside the ring and within 15 degrees of square, falling linearly to 0 at twice either."""
    return max(0.0, min(1.0, 2.0 - r["offset_mm"]/RING_MM, 2.0 - r["angle_deg"]/NEEDLE_ANGLE_DEG))


def _vial_to_pad_sticks(env):
    """The 25 G's pad punctures: when it first entered; when the punctures that carried the dose (at least RESTICK_TRACE
    of it into the tissue) last came out (None if none did); and the punctures that started after that and carried less
    (re-sticks after the dose)."""
    fl, d = env.world.fluid, env.spec["dose_ml"]
    recs = [r for r in fl.needles["needle_im"].records if r["target"] == "pad"]
    arr = [(t, float(m[:H.AIR].sum())/ML) for k, rs in fl.ledger.arrivals.items() if k.startswith("pad:") and not k.endswith(":surface")
           for t, m, _ in rs]
    deliv = {id(r): sum(v for t, v in arr if r["t"] - 1e-6 <= t <= (math.inf if r["t_out"] is None else r["t_out"] + 1e-6)) for r in recs}
    dosing = [r for r in recs if deliv[id(r)] >= RESTICK_TRACE*d]
    t_end = max((math.inf if r["t_out"] is None else r["t_out"]) for r in dosing) if dosing else None
    resticks = [r for r in recs if t_end is not None and r["t"] > t_end and deliv[id(r)] < RESTICK_TRACE*d]
    return min((r["t"] for r in recs), default=math.inf), t_end, resticks


def _vial_to_pad_hygiene(env):
    """_hygiene for vial-to-pad: the fill needle may also touch the vial's inside glass, and the 25 G's point the prepared
    skin round the ring (`pad:prepared`) except after its dose was given. The sharps container is allowed only once a
    needle or the nozzle is done with: not for the fill needle until its draw has ended, nor for the 25 G or the nozzle
    before the 25 G first enters the pad."""
    fl = env.world.fluid
    fill_name = f"{env.spec['dose_syringe']}_needle"
    bad = _hygiene(env, {fill_name: ("vial:inside",), "needle_im": ("pad:prepared",)})
    im, fill, syr = fl.needles["needle_im"], fl.needles[fill_name], fl.syringes[env.spec["dose_syringe"]]
    recs = [r for r in im.records if r["target"] == "pad"]
    t_first, t_end, _ = _vial_to_pad_sticks(env)
    anchor = t_end if t_end is not None else (max(r["t_out"] for r in recs) if recs and all(r["t_out"] is not None for r in recs) else None)
    t_draw = max((r["t_out"] for r in fill.records if r["target"] == "vial_stopper" and r["t_out"] is not None), default=None)

    def add(key, place):
        if place not in bad.get(key, []):
            bad[key] = sorted(bad.get(key, []) + [place])
    if "pad:prepared" in im.touch and anchor is not None and im.touch["pad:prepared"] > anchor:
        add("needle_im", "pad:prepared")
    for p in SHARPS_PLACES:
        if p in fill.touch and t_draw is not None and fill.touch[p] <= t_draw:
            add(fill_name, p)
        if p in im.touch and math.isfinite(t_first) and im.touch[p] < t_first:
            add("needle_im", p)
        if p in syr.touch and math.isfinite(t_first) and syr.touch[p] < t_first:
            add(f"{env.spec['dose_syringe']} nozzle", p)
    return bad


def _used_needle_breaches(env):
    """Needles resheathed after they first entered the pad, and needles let go of by the syringe (`luer_released`) after
    that and before they first touched the sharps container: {needle: [times]} each."""
    w, fl = env.world, env.world.fluid
    recapped, taken_off = {}, {}
    for n, d in fl.needles.items():
        t0 = min((r["t"] for r in d.records if r["target"] == "pad"), default=None)
        if t0 is None:
            continue
        caps = [round(r["t"], 3) for r in d.resheathed if r["t"] > t0]
        t_box = min((d.touch[p] for p in SHARPS_PLACES if p in d.touch), default=math.inf)
        off = [round(e["t"], 3) for e in w.events if e.get("event") == "luer_released" and e.get("fitting") == n and t0 < e["t"] < t_box]
        if caps:
            recapped[n] = caps
        if off:
            taken_off[n] = off
    return recapped, taken_off


def vial_to_pad_partial(env, C):
    """The score of a vial-to-pad run that is not a success: how much of the job is done in the
    world, read from the ledger, the puncture records, the Luer events and the first times of touches; at most 0.9."""
    w, fl, spec = env.world, env.world.fluid, env.spec
    led, d = fl.ledger, spec["dose_ml"]
    syr_name = spec["dose_syringe"]
    fill_name = f"{syr_name}_needle"
    syr, fill, im = fl.syringes[syr_name], fl.needles[fill_name], fl.needles["needle_im"]
    pad = {k: led.volume(k)/ML for k in led.places if k.startswith("pad:")}
    given = sum(pad.values())
    recs = [(n, r) for n, dv in fl.needles.items() for r in dv.records if r["target"] == "pad"]
    im_recs = [r for n, r in recs if n == "needle_im"]
    t_first, _, resticks = _vial_to_pad_sticks(env)
    restick_ids = {id(r) for r in resticks}
    dosing = [r for r in im_recs if id(r) not in restick_ids]
    vial_touch = [(p, t) for p, t in im.touch.items() if p == "vial" or p.startswith(("vial:", "vial_stopper"))]
    vial_punct = [r for r in im.records if r["target"] == "vial_stopper"]
    fresh = all(r["t"] >= t_first for r in vial_punct)
    reentered = any(p == "vial_stopper" and t >= t_first for p, t in vial_touch) or any(r["t"] >= t_first for r in vial_punct)
    fill_drew = any(r["target"] == "vial_stopper" for r in fill.records)
    t_draw = max((r["t_out"] for r in fill.records if r["target"] == "vial_stopper" and r["t_out"] is not None), default=None)
    dirty_draw = after_use = False
    hits = [t for p, t in vial_touch if p == "vial_stopper" and t < t_first]
    for key, places in _vial_to_pad_hygiene(env).items():
        nozzle = key.endswith(" nozzle")
        dev = fl.syringes[key[:-len(" nozzle")]] if nozzle else fl.needles[key]
        for p in places:
            t = dev.touch[p]
            if key == fill_name:
                dirty_draw |= t_draw is None or t <= t_draw
                after_use |= t_draw is not None and t > t_draw
            elif nozzle and t_draw is not None and t <= t_draw:
                dirty_draw = True
            else:
                hits.append(t)
    dirty = {id(r) for r in dosing if any(h < r["t"] for h in hits)}
    after_use |= any(all(r["t"] <= h for r in dosing) for h in hits)

    def open_at(t):
        return [(n, r) for n, r in recs if r["t"] - 1e-6 <= t <= (math.inf if r["t_out"] is None else r["t_out"] + 1e-6)]

    def quality(t):
        on = open_at(t)
        if not on or any(n != "needle_im" or id(r) in dirty for n, r in on):
            return None
        return min(_site_quality(r) for _, r in on)
    rows = [(float(m[:H.AIR].sum()), quality(t)) for t, m, _ in led.arrivals.get("pad:muscle", [])]
    came = sum(v for v, _ in rows)
    good = sum(v for v, q in rows if q is not None)
    muscle = pad.get("pad:muscle", 0.0)*good/came if came > 0 else 0.0
    site = sum(v*q for v, q in rows if q is not None)/good if good > 0 else 1.0
    over = max(given - 1.1*d, 0.0)/d
    back = max(pad.get("pad:surface", 0.0) - 0.05*d, 0.0)/d
    volume = float(np.clip(min(1.0, muscle/(0.9*d)) - max(over, back), 0.0, 1.0))
    w2, w4 = C["injection_speed"]["value"], C["injection_sustained_rate"]["value"]
    r2 = 1.0 if w2 <= 0.3 else float(np.clip((RATE_ZERO_ML - w2)/(RATE_ZERO_ML - 0.3), 0.0, 1.0))
    r4 = 1.0 if w4 <= SUSTAINED_ML else float(np.clip((SUSTAINED_ZERO_ML - w4)/(SUSTAINED_ZERO_ML - SUSTAINED_ML), 0.0, 1.0))
    rate = min(r2, r4)
    given_part = volume*site*rate*float(fresh and not dirty_draw)
    sh = C["in_the_sharps_container"]["value"]
    lost = syr_name in sh["inside"] or w.bq[env.world.scene.bodies[syr_name], 2] < P.COUNTER_Z - 0.05
    in_syr = 0.0 if lost else max(syr.chamber.liquid - syr.chamber.dead, 0.0)/ML
    air = syr.chamber.gas0/ML if in_syr > 0.05*d else 0.0
    drawn_part = float(np.clip(min(1.0, (in_syr + given)/(0.9*d)) - max(air - AIR_LIMIT_ML, 0.0), 0.0, 1.0)) \
        if fill_drew and fresh and not dirty_draw else 0.0
    luer = sorted((e["t"], e["event"] == "luer_captured") for e in w.events if e.get("syringe") == syr_name
                  and e.get("fitting") == "needle_im" and e.get("event") in ("luer_captured", "luer_released"))
    on_at_first = math.isfinite(t_first) and next((c for t, c in reversed(luer) if t <= t_first), False)
    on = (syr.mate is not None and syr.mate.name == "needle_im") or on_at_first
    changed_part = float(on and drawn_part > 0.0 and (syr.mate is None or syr.mate.name != fill_name))
    used = {syr_name: given >= 0.1*d, fill_name: bool(fill.records), "needle_im": bool(im_recs or vial_touch or vial_punct)}
    cleared_part = sum(1 for n, u in used.items() if u and n in sh["inside"])/3.0
    tissue = [t for k, rs in led.arrivals.items() if k.startswith("pad:") and not k.endswith(":surface") for t, _, _ in rs]
    stray = [r for n, r in recs if not any(any(x is r for _, x in open_at(t)) for t in tissue)
             and not (n == "needle_im" and _site_quality(r) >= 1.0) and id(r) not in restick_ids]
    kept = {k: (C[k]["ok"], 1) for k in ("at_rest", "on_the_counter", "no_protective_stop", "spills", "no_resheathing_by_hand",
                                         "needles_not_bent", "syringes_whole", "vial_upright", "vial_pressure", "air_to_patient",
                                         "used_needle_not_recapped", "used_needle_kept_on")}
    kept.update(withdrawn_cleanly=(all(r["lever_deg"] <= NEEDLE_ANGLE_DEG and (r["exit_angle_deg"] or 0.0) <= NEEDLE_ANGLE_DEG
                                       for _, r in recs), 1),
                needles_out_of_the_pad=(all(r["t_out"] is not None for _, r in recs), 1), no_stray_puncture=(not stray, 1),
                no_touch_after_use=(not after_use, 1), only_sharps_in_the_container=(not sh["others"], 1),
                flap_closed=(sh["flap_deg"] < 5.0, 1), vial_not_reentered_after_use=(not reentered, 2), no_contaminated_stick=(not dirty, 2),
                no_restick_after_dose=(C["no_restick_after_dose"]["ok"], 2))
    failed = [k for k, (ok, _) in kept.items() if not ok]
    house = max(HOUSEKEEPING_FLOOR, 1.0 - HOUSEKEEPING_STEP*sum(n for ok, n in kept.values() if not ok))
    partial = (W_DOSE_GIVEN*given_part + W_DOSE_DRAWN*drawn_part + W_NEEDLE_CHANGED*changed_part + W_CLEARED_AWAY*cleared_part)*house
    return dict(partial=round(partial, 4), dose_given=round(given_part, 4), dose_volume=round(volume, 4), dose_site=round(site, 4),
                dose_rate=round(rate, 4), dose_drawn=round(drawn_part, 4), needle_changed=changed_part, cleared_away=round(cleared_part, 4),
                housekeeping=round(house, 4), housekeeping_failed=failed,
                contaminated=dict(before_the_draw_ended=dirty_draw, sticks=sorted(round(r["t"], 3) for r in im_recs if id(r) in dirty)),
                resticks_after_dose=sorted(round(r["t"], 3) for r in resticks),
                muscle_credited_ml=round(muscle, 4))

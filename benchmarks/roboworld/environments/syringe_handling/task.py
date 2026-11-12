"""Tasks: what the instructor says, and what is measured afterwards.

Every criterion is a physical state or a physical history read from the world: liquid volumes by place and by time of
arrival, pressures, what a bare needle or nozzle has touched, where a needle went in and how it came out, where things
lie. Nothing is read from the robot's commands. The evaluator is read-only and does not advance time.

A task's own criteria and score are in its verifier.py (tasks/<task>/); the criteria every task has and the checks
several tasks share are here (evaluate_criteria).
"""

import numpy as np

from . import params as P
from . import hydraulics as H
from .config import LEGACY_CONTRACT, EIGHT_TASK_CONTRACT, task_verifier
from . import task_set_v2                     # syringe-tasks-v2: instructed and judged as syringe-tasks-v1

ML = H.ML
AIR_LIMIT_ML = 0.05          # authored: half a scale division of a 3 mL syringe
SPILL_LIMIT_ML = 0.10        # authored: two drops of an 18 G needle
REST_S = 2.0
REST_SPEED = 0.001           # m/s, as net motion: 0.1 mm in any REST_WINDOW_S
REST_WINDOW_S = 0.1          # a held thing trembles (an arm at standstill, a pad's contact points): 20 um at 80 Hz is 10 mm/s
                             # at an instant and no motion at all over a tenth of a second (as in label_application)
CONNECTOR_SHIFT_M = 0.010    # authored: the tape
VIAL_PRESSURE_PA = 20e3      # authored: under the 27 kPa a pierced closure is tested to hold (ISO 8362-2 class)
BLOCKED_LIMIT_PA = 100e3     # authored: "do not force it"
NEEDLE_ANGLE_DEG = 15.0
RING_MM = 0.5e3*P.PAD["ring_d"]

# criteria that are the task's own result; the rest are rules of conduct that hold at the start and must go on holding
RESULTS = ("dose_in_syringe", "air_in_syringe", "in_the_sharps_container", "dose_in_tissue", "needle_path", "flush_delivered",
           "dose_delivered", "delivery_time", "bolus_time", "line_not_forced")

INSTRUCTIONS = {
    "sharps-away": "Three used syringes with bare needles lie on the tray with two loose needle sheaths and a used gauze square. "
                   "Put every syringe with its needle into the red sharps container through the flap in its lid. Do not put a "
                   "sheath back on a needle, do not let a needle touch anything on the way, and put nothing else into the "
                   "container.",
    "set-dose": "The 3 mL syringe on the tray (orange 25 G needle, sheathed) holds too much blue medication and an air bubble. "
                "Leave it holding {dose_ml:g} mL of medication and no air. What you expel goes into the kidney basin. Do not "
                "put the sheath back on by hand and do not let the bare needle touch anything.",
    "draw-dose": "Draw {dose_ml:g} mL of the blue medication from the vial into the 3 mL syringe ({fit}), without air in the "
                 "syringe. Leave the vial standing upright on the counter where you found it.",
    "inject": "The {syringe} on the tray holds {dose_ml:g} mL of medication. Give all of it into the {layer} of the injection pad "
              "at the printed ring, {angle}, no faster than 10 seconds per millilitre, then put the syringe into the sharps "
              "container. Do not put the sheath back on.",
    "vial-to-pad": "Draw {dose_ml:g} mL of the blue medication from the vial with the pink blunt fill needle. Change to the orange "
                   "25 G needle that lies sheathed on the tray. Give the dose into the muscle of the injection pad at the "
                   "printed ring, square to the skin, no faster than 10 seconds per millilitre. Put the syringe and both "
                   "needles into the sharps container and leave the vial upright.",
    "flush-lock": "Flush the saline lock on the training arm with {flush_ml:g} mL from the prefilled saline syringe on the tray: "
                  "free its stopper, take off its cap, clear its air, screw it onto the needle-free connector, open the slide "
                  "clamp, flush, close the clamp and take the syringe off.{extra}",
    "slow-push": "Give the {dose_ml:g} mL of blue medication in the labelled syringe through the saline lock over {dose_s:g} seconds "
                 "by the wall clock, then flush with {flush_ml:g} mL of saline, the first millilitre at the same rate. Leave the "
                 "lock clamped with nothing attached.",
    "fast-bolus": "Inject the {dose_ml:g} mL of amber contrast through the saline lock in under {bolus_s:g} seconds, then flush with "
                  "{flush_ml:g} mL of saline. Leave the lock clamped with nothing attached.",
}


def task_spec(config, layout):
    L = layout
    spec = dict(task=config.task, variant=config.variant, seed=config.seed, setup=config.setup, contract=config.contract,
                dose_ml=L.get("dose_ml"), flush_ml=L.get("flush_ml"), dose_s=L.get("dose_s"), bolus_s=L.get("bolus_s"),
                layer=L.get("layer", "muscle"))
    names = [s[0] for s in L.get("syringes", ())]
    spec["syringes"] = names
    spec["dose_syringe"] = "dose" if "dose" in names else (names[0] if names else None)
    kind = {s[0]: s[1] for s in L.get("syringes", ())}
    fit = {s[0]: s[5] for s in L.get("syringes", ())}
    f = fit.get(spec["dose_syringe"])
    fmt = dict(dose_ml=spec["dose_ml"] or 0, flush_ml=spec["flush_ml"] or 0, dose_s=spec["dose_s"] or 0, bolus_s=spec["bolus_s"] or 0,
               layer="fat under the skin" if spec["layer"] == "fat" else "muscle",
               angle="at 45 to 90 degrees to the skin" if spec["layer"] == "fat" else "square to the skin",
               syringe=P.SYRINGES[kind[spec["dose_syringe"]]]["label"] + " syringe" if spec["dose_syringe"] else "",
               fit="through the needle-free adapter on the vial" if L.get("adapter") else
                   (f"with its {P.NEEDLES[f]['label']} needle" if f in P.NEEDLES else ""),
               extra=" If the line will not take the flush, do not force it: leave it clamped and take the syringe off."
                     if L.get("blocked") else "")
    spec["instruction"] = INSTRUCTIONS[config.task].format(**fmt)
    if config.contract in task_set_v2.TASK_SET_CONTRACTS and config.task == "flush-lock":
        spec["instruction"] = ("Flush the saline lock on the training arm with {flush_ml:g} mL from the prefilled "
                               "saline syringe on the tray. Leave the lock clamped with nothing attached.{extra}").format(**fmt)
    if config.contract == LEGACY_CONTRACT and config.task == "inject":
        spec["instruction"] += " Scoring tolerance: at most 0.30 mL in any 2 s (nominal instruction: 0.20 mL)."
    if config.contract == LEGACY_CONTRACT and config.task == "slow-push":
        spec["instruction"] += " Scoring allows 1.5 times the nominal volume in any 15 s; the total-time check also applies."
    if config.contract == LEGACY_CONTRACT and config.task in ("flush-lock", "slow-push", "fast-bolus"):
        spec["instruction"] += " The flush amount is measured entering the line from the syringe; initial line saline is reported separately."
    return spec


def instruction(config):
    """The instructor's sentence for a configuration, without building the bay."""
    from .scene import task_layout
    return task_spec(config, task_layout(config))["instruction"]


def start_record(w):
    """What is true at the handover and belongs to the scene, not to the robot: what each bare needle and nozzle
    already touches where it lies, how much each vial holds, where things stand."""
    fl = w.fluid
    return dict(touch={**{n: sorted(s.touch) for n, s in fl.syringes.items()}, **{n: sorted(s.touch) for n, s in fl.needles.items()}},
                vial_ml={n: v.vessel.liquid_volume/ML for n, v in fl.vials.items()},
                vial_xy={n: [float(x) for x in w.bq[v.body, :2]] for n, v in fl.vials.items()},
                syringe_ml={n: s.chamber.liquid/ML for n, s in fl.syringes.items()},
                syringe_dose_ml={n: max(s.chamber.liquid - s.chamber.dead, 0.0)/ML for n, s in fl.syringes.items()},
                syringe_air_ml={n: s.chamber.gas0/ML for n, s in fl.syringes.items()},
                positions={n: [float(x) for x in w.bq[b, :3]] for n, b in w.scene.bodies.items()})


class History:
    """What has to hold throughout and cannot be read from the end state: when anything loose last moved."""

    def __init__(self, w):
        self.bodies = np.array(sorted(set(w.scene.bodies.values()) - set(w.scene.robot_bodies)), dtype=int)
        self.last_move = 0.0
        self.max_speed = 0.0
        self.every = max(int(round(REST_WINDOW_S/w.dt)), 1)
        self.ref = w.bq[self.bodies, :3].copy()
        self.ref_tick = int(w.tick)

    def tick(self, w):
        self.max_speed = max(self.max_speed, float(np.abs(w.bqd[self.bodies, :3]).max(initial=0.0)))
        if w.tick - self.ref_tick >= self.every:
            p = w.bq[self.bodies, :3]
            if float(np.linalg.norm(p - self.ref, axis=1).max(initial=0.0)) > REST_SPEED*REST_WINDOW_S:
                self.last_move = w.sim_time
            self.ref, self.ref_tick = p.copy(), int(w.tick)

    def snapshot(self):
        return dict(last_move=self.last_move, max_speed=self.max_speed, ref=self.ref.copy(), ref_tick=self.ref_tick)

    def restore(self, s):
        self.last_move, self.max_speed = s["last_move"], s["max_speed"]
        self.ref, self.ref_tick = s["ref"].copy(), s["ref_tick"]


# ---------------------------------------------------------------------- criteria
def _c(ok, value, limit, note=""):
    return dict(ok=bool(ok), value=value, limit=limit, **({"note": note} if note else {}))


def dose_tolerance_ml(dose_ml, capacity_ml):
    """ISO 7886-1 Table 1: 5 % of the expelled volume at or above half the nominal capacity, 1.5 % of the capacity plus
    2 % of the expelled volume below it."""
    return 0.05*dose_ml if dose_ml >= 0.5*capacity_ml else 0.015*capacity_ml + 0.02*dose_ml


def _inside(p, box):
    return all(box["lo"][i] <= p[i] <= box["hi"][i] for i in range(3))


def _spills(fl):
    """Liquid that lies where it should not: everything in the ledger except the vein, the tissue's layers, the
    basin, the insides of sheaths and caps, and the sharps container."""
    out = {}
    for place, v in fl.ledger.places.items():
        if place == "vein" or place == "basin" or place.endswith(("_sheath", "_cap")) or place.startswith("sharps"):
            continue                          # (what a removed sheath or cap has caught is not on the bench)
        if place.startswith("pad:") and not place.endswith(":surface"):
            continue
        ml = float(v[:H.AIR].sum())/ML
        if ml > 1e-6:
            out[place] = round(ml, 4)
    hang = fl.spray.hanging()/ML + fl.spray.airborne()/ML
    if hang > 1e-6:
        out["in_the_air_or_hanging"] = round(hang, 4)
    return out


def _hygiene(env, extra=None):
    """What each bare needle and bare nozzle has touched since the start, less what the rules allow and what it lay
    on at the handover."""
    w, fl = env.world, env.world.fluid
    fittings = set(fl.females) | {c for c in fl.caps} | {n for n in fl.needles} | {v for v, d in fl.vials.items() if d.female is not None}
    bad = {}
    for name, n in fl.needles.items():
        allowed = {name + "_sheath", "sharps_flap", "sharps_outside", "sharps_inside"} | set(env.start["touch"].get(name, ()))
        allowed |= {p.name for p in fl.penetrables}                     # a stopper's target, the pad at the ring
        allowed |= set((extra or {}).get(name, ()))
        hits = sorted(set(n.touch) - allowed)
        if hits:
            bad[name] = hits
    for name, s in fl.syringes.items():
        allowed = fittings | {"sharps_flap", "sharps_outside", "sharps_inside"} | set(env.start["touch"].get(name, ()))
        allowed |= {f"{x}_sheath" for x in fl.needles}
        # a syringe taken off the saline lock for the last time is used: its nozzle may then touch the kidney dish, the
        # counter's receiver for used things (the rule keeps a nozzle clean while it can still go onto a line; the
        # robot's flush-lock put its used syringe into the dish and was failed for the dish touching the nozzle)
        off = [e["t"] for e in w.events if e.get("event") == "luer_released" and e.get("syringe") == name and e.get("fitting") in fl.locks]
        if off and "basin" in s.touch and s.touch["basin"] >= max(off):
            allowed.add("basin")
        hits = sorted(set(s.touch) - allowed)
        if hits:
            bad[name + " nozzle"] = hits
    return bad


def _in_sharps(env, names):
    w, sc = env.world, env.world.scene
    box = sc.sharps_cavity
    inside, outside = [], []
    for n in names:
        (inside if _inside(w.bq[sc.bodies[n], :3], box) else outside).append(n)
    return inside, outside


SUSTAINED_S, SUSTAINED_ML = 4.0, 0.5
HOUSEKEEPING_STEP, HOUSEKEEPING_FLOOR = 0.05, 0.5
RATE_ZERO_ML = 1.0


def _good_puncture(n, r, needle_name=None, lo_ang=0.0, hi_ang=NEEDLE_ANGLE_DEG):
    return (r["offset_mm"] <= RING_MM and lo_ang <= r["angle_deg"] <= hi_ang and r["t_out"] is not None
            and (r["exit_angle_deg"] or 0.0) <= NEEDLE_ANGLE_DEG and r["lever_deg"] <= NEEDLE_ANGLE_DEG
            and (needle_name is None or n == needle_name))


def _tissue_window(led, window_s):
    """The largest volume (mL) of liquid that reached the pad's tissue (any layer, not its surface) in any window_s."""
    rows = sorted((t, float(m[:H.AIR].sum())) for k, rs in led.arrivals.items() if k.startswith("pad:") and not k.endswith(":surface")
                  for t, m, _ in rs)
    if not rows:
        return 0.0
    t = np.array([r[0] for r in rows])
    c = np.concatenate([[0.0], np.cumsum([r[1] for r in rows])])
    j = np.searchsorted(t, t - window_s + 0.5*H.ARRIVAL_BIN_S, side="left")
    return float((c[1:] - c[j]).max())/ML


def evaluate(env):
    """The episode's outcome: the task's verifier.py (tasks/<task>/) decides, with evaluate_criteria below."""
    return task_verifier(env.spec["task"]).verify(env)


class Checks:
    """What a task's verifier.py adds its own criteria with: evaluate_criteria passes one to the verifier's criteria(c).

    env, spec, world, fluid, ledger   the episode: env, env.spec, env.world, its fluidics and their liquid ledger
    C                                 the criteria so far (name -> criterion); the task adds its own
    syringe_dose(name, target)                     dose_in_syringe, air_in_syringe
    vial_after(dose_drawn_ml=None, home=True)      vial_upright, vial_pressure, and vial_contents for a drawn dose
    lock_after()                                   lock_clamped, lock_detached, connector_not_pulled, line_pressure,
                                                   air_to_patient
    push(sp)                                       lock_after(), then dose_delivered (sp into the vein) and
                                                   flush_delivered
    sharps(names, nothing_else=True, seated=False)              in_the_sharps_container
    injection(dose_ml, layer, needle_name=None, whole=False)    dose_in_tissue, injection_speed (whole: into the
                                                   tissue, all layers together, and injection_sustained_rate),
                                                   air_to_patient, needle_path"""

    def __init__(self, env, C, syringe_dose, vial_after, lock_after, push, sharps, injection):
        self.env, self.spec, self.world, self.fluid = env, env.spec, env.world, env.world.fluid
        self.ledger, self.C = env.world.fluid.ledger, C
        self.syringe_dose, self.vial_after, self.lock_after, self.push = syringe_dose, vial_after, lock_after, push
        self.sharps, self.injection = sharps, injection


def evaluate_criteria(env, criteria, *, hygiene=None, hygiene_note="", own_score=None):
    """The criteria every task has, then criteria(c), the task's own (its verifier.py), added through a Checks c; and
    the score. A task whose needles follow their own hygiene rule passes hygiene(env) (what each touched that it may
    not) and the note its limit adds; a task with its own score passes own_score(env, C, out), which sets it in out.
    Otherwise the score is the share of the task's results reached times the share of the rules of conduct kept, or
    under syringe-tasks-v1 and -v2 the contract's reward (scoring.reward, with the progress of the task's verifier)."""
    w, fl, sc, spec = env.world, env.world.fluid, env.world.scene, env.spec
    task = spec["task"]
    led = fl.ledger
    C = {}
    # ---- order and rest (every task)
    still = w.sim_time - env._history.last_move
    C["at_rest"] = _c(still >= REST_S or w.sim_time < 1e-9, round(still, 3), f">= {REST_S} s with no loose object moving more than {1e3*REST_SPEED*REST_WINDOW_S:g} mm in any {REST_WINDOW_S:g} s")
    off = [n for n, b in sc.bodies.items() if b not in sc._robot_set and w.bq[b, 2] < P.COUNTER_Z - 0.05]
    C["on_the_counter"] = _c(not off, off, "nothing has left the counter")
    C["no_protective_stop"] = _c(not w.reflexes, len(w.reflexes), "0 reflex stops of an arm")
    spills = _spills(fl)
    C["spills"] = _c(sum(spills.values()) <= SPILL_LIMIT_ML, spills or 0.0, f"<= {SPILL_LIMIT_ML} mL outside the basin, the vein and the tissue")
    bad = hygiene(env) if hygiene is not None else _hygiene(env)
    C["needle_hygiene"] = _c(not bad, bad or "nothing it may not", "a bare needle or nozzle touches only its sheath or cap, a "
                             "stopper's target, the pad, a fitting, the sharps container" + hygiene_note)
    by_hand = [n for n, d in fl.needles.items() if any(r["sheath_in_gripper"] for r in d.resheathed)]
    C["no_resheathing_by_hand"] = _c(not by_hand, by_hand, "no sheath pushed onto a needle while a gripper holds the sheath")
    bent = [n for n, d in fl.needles.items() if d.flags["bent"]]
    C["needles_not_bent"] = _c(not bent, bent, "no needle loaded past its yield moment")
    pulled = [n for n, s in fl.syringes.items() if s.flags["extracted"] or s.flags["overridden"]]
    C["syringes_whole"] = _c(not pulled, pulled, "no plunger pulled out, no Luer thread overridden")

    def syringe_dose(name, target):
        s = fl.syringes[name]
        cap = s.s["nominal_ml"]
        dose = max(s.chamber.liquid - s.chamber.dead, 0.0)/ML
        tol = dose_tolerance_ml(target, cap)
        C["dose_in_syringe"] = _c(abs(dose - target) <= tol, round(dose, 4), f"{target:g} +/- {tol:.3f} mL (ISO 7886-1 Table 1)")
        # (with at least half the dose in it: an empty syringe is not a dose free of air. The adapter variant's empty
        # syringe, which has no needle's air in it, held this criterion at the handover and scored 0.5 for nothing)
        C["air_in_syringe"] = _c(s.chamber.gas0/ML <= AIR_LIMIT_ML and dose >= 0.5*target, round(s.chamber.gas0/ML, 4),
                                 f"<= {AIR_LIMIT_ML} mL, in a syringe that holds at least half the dose")

    def vial_after(dose_drawn_ml=None, home=True):
        v = fl.vials["vial"]
        r = v.reading(w)
        gauge = v.vessel.gas_pressure() - P.P_ATM
        moved = float(np.linalg.norm(w.bq[v.body, :2] - np.asarray(env.start["vial_xy"]["vial"])))
        stuck = [n for n, d in fl.needles.items() if d.track is not None and d.track["pen"].name == v.site.name]
        base = P.COUNTER_Z + 0.004 if home else sc.tray_floor + P.TRAY["rib_h"] + 0.001
        C["vial_upright"] = _c(r["upright_cos"] > 0.99 and (moved <= 0.030 or not home) and w.bq[v.body, 2] < base and not stuck and
                               (v.female is None or v.female.mate is None),
                               dict(upright_cos=round(r["upright_cos"], 4), moved_mm=round(1e3*moved, 1), needle_in=stuck),
                               "standing upright within 30 mm of its starting place, nothing in it or on it" if home else
                               "standing upright on the counter or the tray (its base no higher than the tray's ribs), nothing in it or on it")
        C["vial_pressure"] = _c(abs(gauge) <= VIAL_PRESSURE_PA, round(gauge/1e3, 2), f"headspace within +/- {VIAL_PRESSURE_PA/1e3:g} kPa of ambient")
        if dose_drawn_ml is not None:
            lost = env.start["vial_ml"]["vial"] - r["liquid_ml"] - dose_drawn_ml
            C["vial_contents"] = _c(lost <= 0.1, round(lost, 4), "the vial has lost the dose and at most 0.1 mL more")

    def lock_after(deliver=True):
        lk = fl.locks["lock"]
        r = lk.reading(w)
        C["lock_clamped"] = _c(not r["clamp_open"], r["clamp_mm"], "slide clamp closed")
        C["lock_detached"] = _c(lk.female.mate is None, r["mate"], "nothing attached to the connector")
        C["connector_not_pulled"] = _c(lk.max_shift <= CONNECTOR_SHIFT_M, round(1e3*lk.max_shift, 2), f"<= {1e3*CONNECTOR_SHIFT_M:g} mm at any time")
        limit = BLOCKED_LIMIT_PA if lk.blocked else P.LINE_LIMIT_PA
        C["line_pressure"] = _c(lk.peak_gauge <= limit, round(lk.peak_gauge/1e3, 1), f"<= {limit/1e3:g} kPa at any time")
        C["air_to_patient"] = _c(led.volume("vein", "air")/ML <= AIR_LIMIT_ML, round(led.volume("vein", "air")/ML, 4), f"<= {AIR_LIMIT_ML} mL of air into the vein")

    def sharps(names, nothing_else=True, seated=False):
        inside, outside = _in_sharps(env, names)
        flap = float(np.degrees(w.jq[sc.flap_joint.q]))
        worn = {f"{n}_sheath" for n in inside if seated and n in fl.needles and fl.needles[n].sheathed}
        others = [n for n, b in sc.bodies.items() if n not in names and n not in worn and n != "sharps_flap" and b not in sc._robot_set
                  and _inside(w.bq[b, :3], sc.sharps_cavity)]
        C["in_the_sharps_container"] = _c(not outside and flap < 5.0 and not (others and nothing_else),
                                          dict(inside=inside, outside=outside, flap_deg=round(flap, 1), others=others),
                                          "every used syringe with its needle inside, flap closed, nothing else in it"
                                          + (" (a sheath seated on a needle inside goes with that needle)" if seated else ""))

    def injection(dose_ml, layer, needle_name=None, whole=False):
        want = f"pad:{layer}"
        got = {k: led.volume(k)/ML for k in led.places if k.startswith("pad:")}
        in_layer = got.get(want, 0.0)
        elsewhere = sum(v for k, v in got.items() if k != want)
        C["dose_in_tissue"] = _c(in_layer >= 0.9*dose_ml and in_layer <= 1.1*dose_ml and got.get("pad:surface", 0.0) <= 0.05*dose_ml
                                 and (not whole or in_layer + elsewhere <= 1.1*dose_ml),
                                 {k: round(v, 4) for k, v in got.items()}, f">= 90 % of {dose_ml:g} mL in the {layer}; <= 5 % back out of the track"
                                 + (f"; <= {1.1*dose_ml:g} mL into the pad in all" if whole else ""))
        rate = _tissue_window(led, 2.0) if whole else max([led.rate_window(k, sp, 2.0) for k in led.places if k.startswith("pad:")
                                                           and not k.endswith(":surface") for sp in ("medication", "saline", "contrast")] or [0.0])/ML
        C["injection_speed"] = _c(round(rate, 6) <= 0.3 if whole else rate <= 0.3, round(rate, 4), "<= 0.3 mL in any 2 s (1.5 times 10 s per mL)"
                                  + (", into the tissue, all layers together" if whole else ""))
        if whole:
            w4 = _tissue_window(led, SUSTAINED_S)
            C["injection_sustained_rate"] = _c(round(w4, 6) <= SUSTAINED_ML, round(w4, 4), f"<= {SUSTAINED_ML:g} mL into the tissue in any "
                                               f"{SUSTAINED_S:g} s (1.25 times 10 s per mL, pauses not counted as delivery)")
        air = sum(led.volume(k, "air") for k in led.places if k.startswith("pad:"))/ML
        C["air_to_patient"] = _c(air <= AIR_LIMIT_ML, round(air, 4), f"<= {AIR_LIMIT_ML} mL of air into the tissue")
        recs = [(n, r) for n, d in fl.needles.items() for r in d.records if r["target"] == "pad"]
        lo_ang, hi_ang = (0.0, 45.0) if layer == "fat" else (0.0, NEEDLE_ANGLE_DEG)
        ok = bool(recs) and all(_good_puncture(n, r, needle_name, lo_ang, hi_ang) for n, r in recs)
        C["needle_path"] = _c(ok, [dict(needle=n, offset_mm=round(r["offset_mm"], 2), angle_deg=round(r["angle_deg"], 1),
                                        deepest_mm=round(r["deepest_mm"], 1), lever_deg=round(r["lever_deg"], 1),
                                        exit_angle_deg=None if r["exit_angle_deg"] is None else round(r["exit_angle_deg"], 1),
                                        withdrawn=r["t_out"] is not None) for n, r in recs],
                              f"into the {2*RING_MM:g} mm ring, within {hi_ang:g} deg of square, not levered or withdrawn more than "
                              f"{NEEDLE_ANGLE_DEG:g} deg off the way in" + ("" if needle_name is None else f", with {needle_name}"))

    def push(sp):
        """A dose pushed through the lock, then the flush (slow-push, fast-bolus): sp is what the dose syringe gives."""
        lock_after()
        C["dose_delivered"] = _c(led.volume("vein", sp)/ML >= 0.95*spec["dose_ml"], round(led.volume("vein", sp)/ML, 3),
                                 f">= 95 % of {spec['dose_ml']:g} mL of {sp} into the vein")
        C["flush_delivered"] = _c(sum(float(r[1][P.SPECIES.index("saline")]) for r in led.arrivals.get("lock:in", []))/ML >= 0.95*spec["flush_ml"],
                                      dict(from_syringe_ml=round(sum(float(r[1][P.SPECIES.index("saline")]) for r in led.arrivals.get("lock:in", []))/ML, 3),
                                           total_saline_at_vein_ml=round(led.volume("vein", "saline")/ML, 3)),
                                  f">= 95 % of {spec['flush_ml']:g} mL of saline from the syringe into the line (initial priming saline excluded)")

    criteria(Checks(env, C, syringe_dose, vial_after, lock_after, push, sharps, injection))
    # the score: the share of the task's own results reached, times the share of the rules of conduct kept; doing
    # nothing scores 0 whatever rules it keeps
    for k, c in C.items():
        c["kind"] = "result" if k in RESULTS else "conduct"
    res = [c["ok"] for c in C.values() if c["kind"] == "result"]
    con = [c["ok"] for c in C.values() if c["kind"] == "conduct"]
    score = (sum(res)/len(res))*(sum(con)/len(con))
    out = dict(success=bool(all(res) and all(con)), score=round(score, 4), criteria=C, task=task,
               conservation_error_ml=round(fl.conservation_error_ml(), 6))
    if own_score is not None:                     # the task's own score (its verifier.py)
        own_score(env, C, out)
    elif spec.get("contract", LEGACY_CONTRACT) == EIGHT_TASK_CONTRACT:
        from .scoring import reward
        out.update(reward(env, C, out["success"]))
    elif spec.get("contract") == task_set_v2.CONTRACT:
        out.update(task_set_v2.reward(env, C, out["success"]))
    else:
        out["contract"] = LEGACY_CONTRACT
    return out

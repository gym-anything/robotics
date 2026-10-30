"""Task-set judging, independent of physics.

The seven opt-in revisions use stir-sauce's failure factor. This module never writes physical state.
Baselines are captured at hand-over, not at the first evaluation. Heat records contain simultaneous portion
readings, never averages of particle temperature peaks. Legacy tasks and stir-sauce bypass the revised judging.
"""
import math
from dataclasses import dataclass

from .config import Config

LEGACY = "v1"
CONTRACT = "ingredient-mixing-tasks-v1"
STIR_SAUCE_CONTRACT = "stir-sauce-v1"
ORDER = ("sauce-blend", "muffin-fold", "dry-whisk", "pudding-whisk", "pancake-whisk", "cake-mixer",
         "reheat-porridge", "stir-sauce")
DIFFICULTY = dict(zip(ORDER, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))


@dataclass
class TaskConfig(Config):
    """Catalogue-only config extension; the physics Config and its defaults remain frozen."""
    contract: str = LEGACY

    def __post_init__(self):
        super().__post_init__()
        validate_contract(self.contract, self.task)


def task_spec(config):
    from . import task
    return revise_spec(config, task.task_spec(config))


def instruction(config):
    from . import task
    original = task.task_spec(config)["instruction"]
    return task.instruction(config).replace(original, task_spec(config)["instruction"], 1)


def validate_contract(contract, task):
    if contract not in (LEGACY, CONTRACT):
        raise ValueError(f"Unknown contract {contract!r}; choose {LEGACY!r} or {CONTRACT!r}")
    if contract == CONTRACT and task == "stir-sauce":
        raise ValueError("stir-sauce keeps its own contract; select contract=v1")
    return contract


def enabled(config):
    return getattr(config, "contract", LEGACY) == CONTRACT and config.task != "stir-sauce"


def contract_for(config):
    return STIR_SAUCE_CONTRACT if config.task == "stir-sauce" else getattr(config, "contract", LEGACY)


def revise_spec(config, spec):
    """Only judging/instructions change; recipe, props, layout and dynamics stay identical."""
    if not enabled(config):
        return spec
    spec = dict(spec)
    if config.task == "pudding-whisk":
        spec["instruction"] = ("The bowl holds two cups of cold milk with a packet of instant pudding mix heaped in "
                               "the middle. Whisk until the pudding is smooth, then take the whisk out. Keep the "
                               "pudding in the bowl.")
    elif config.task == "cake-mixer":
        spec["instruction"] = ("The bowl holds cake mix with water, oil and eggs. The hand mixer stands on its heel "
                               "beside the bowl, switched off. Beat the batter until smooth. Leave the mixer "
                               "switched off and set down on the counter, out of the bowl. Keep the batter in the bowl.")
    elif config.task == "reheat-porridge":
        spec["instruction"] = ("The saucepan on the hob holds cold cooked porridge; the hob is on. Stir until the "
                               "porridge is hot all the way through (74 C, 165 F, everywhere), without letting it "
                               "catch on the bottom. Then turn the hob off. Keep the porridge in the pan and the pan "
                               "on the hob. A probe thermometer stands in the crock.")
        spec["criteria"] = dict(spec["criteria"], vessel_in_place=dict(on_hob=True),
                                contained=dict(limit=0.01, below_rim=True))
    return spec


def clip(x):
    return min(1., max(0., float(x)))


def ramp(x, lo, hi):
    return clip((x - lo)/(hi - lo)) if hi > lo else 0.


def uniform_progress(criterion, baseline):
    if criterion["batch_share_in_the_vessel"] < criterion["share_limit"]:
        return 0.
    terms = []
    starts = list(baseline["cv"].values())
    if starts and all(v is not None and v <= criterion["limit"] for v in starts):
        return 0.  # already uniform at hand-over; no improvement to credit
    for species, cv in criterion["each"].items():
        start = baseline["cv"].get(species)
        limit = criterion["limit"]
        if cv is None or start is None or not math.isfinite(cv) or not math.isfinite(start):
            return 0.
        # Already-uniform ingredients must remain uniform, but do not block credit for mixing the others.
        terms.append(clip(math.log(start/max(cv, 1e-12))/math.log(start/limit)) if start > limit
                     else clip(limit/max(cv, 1e-12)))
    return min(terms, default=0.)


def hot_quality(reading, start_c, limit_c=74., spread_k=15.):
    return ramp(reading["coldest_c"], start_c, limit_c) * (1. - ramp(
        reading["hottest_c"] - reading["coldest_c"], spread_k, 40.))


def hot_ok(reading, limit_c=74., spread_k=15.):
    return (reading.get("batch_share", 0.) >= 0.99 and reading["coldest_c"] >= limit_c
            and reading["hottest_c"] - reading["coldest_c"] <= spread_k)


def best_hot(previous, current, start_c):
    """Keep one coherent reading; later uneven heating must not erase a completed reheat.

    A reading covers >=99% of the batch; heating a little food then returning cold food cannot pass. Success is
    re-derived from the temperatures, not a sticky success flag. Current readings also allow constructed states.
    """
    candidates = [r for r in (previous, current) if r is not None and r.get("batch_share", 0.) >= 0.99]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda r: (hot_ok(r), hot_quality(r, start_c), r["coldest_c"])))


def record_hot_sample(hob, local, temp, mass):
    """Called by the external post-heat measurement wrapper, with no physical writes."""
    baseline = getattr(hob, "task_baseline", None)
    if baseline is None or not enabled(hob.w.config) or hob.w.config.task != "reheat-porridge":
        return
    if hob.knob.s_level <= 0 or len(temp) == 0:
        return
    from .task import portion_temperatures
    cold, hot, cells = portion_temperatures(local, temp, mass, 60e-6, baseline["spacing"]**3)
    current = dict(coldest_c=round(cold, 2), hottest_c=round(hot, 2), cells=cells,
                   batch_share=round(float(mass.sum())/baseline["batch_mass"], 5))
    hob.hot_record = best_hot(getattr(hob, "hot_record", None), current, baseline["coldest_c"])


def capture_start(env):
    """Read once after settling; TaskEnv adds the result to the physical start checkpoint without moving anything."""
    if not enabled(env.config) or env.info is None:
        return
    C = evaluate(env.world, env.info, _progress=False)["criteria"]
    baseline = dict(cv=dict(C.get("uniform", {}).get("each", {})),
                    dry=C.get("no_dry_powder", {}).get("value", 0.),
                    lump_quality=1. - ramp(C.get("smooth", {}).get("value", 0.), 0.02, 0.10),
                    coldest_c=C.get("hot_through", {}).get("coldest_c", 5.),
                    batch_mass=float(sum(env.info["batch"])), spacing=env.info["spacing"])
    env.info["task_baseline"] = baseline
    if env.config.task == "reheat-porridge":
        env.world.heat.task_baseline = dict(baseline)
        env.world.heat.hot_record = None       # settling is not part of the episode


def progress(task, C, baseline, bottom_cm2):
    """Continuous material improvement times retention, damage and finishing factors."""
    keep = clip(1. - C["contained"]["value"])
    in_place = C["vessel_in_place"]["ok"]
    close = "tool_out" if "tool_out" in C else "appliance_off" if "appliance_off" in C else None
    if task == "reheat-porridge":
        core = hot_quality(C["hot_through"], baseline["coldest_c"])
        if C["hot_through"].get("batch_share", 0.) < 0.99:
            core = 0.
        damage = clip(1. - C["not_scorched"]["value"]/bottom_cm2)
        done = C["hot_through"]["ok"]
        terms = dict(hot=core, unburnt=damage)
    else:
        U = uniform_progress(C["uniform"], baseline)
        terms = dict(uniform=U)
        core, damage = U, 1.
        if "no_dry_powder" in C:
            d = C["no_dry_powder"]
            D = clip((baseline["dry"] - d["value"])/(baseline["dry"] - d["limit"])) \
                if baseline["dry"] > d["limit"] else 0.
            terms["wet"] = D
            core = (U + D)/2.
            if "smooth" in C:
                L = 1. - ramp(C["smooth"]["value"], C["smooth"]["limit"], 0.10)
                # A fresh heap often has no lumps. Normalize the combined quality from its actual start instead
                # of awarding L/3 immediately (a plain (U + D + L)/3 would give idle credit).
                L0 = baseline["lump_quality"]
                core = clip((U + D + L - L0)/(3. - L0))
                terms.update(smooth=L, initial_smooth=L0)
        if "not_overmixed" in C:
            work = C["not_overmixed"]
            damage = 1. - ramp(work["value"], work["limit"], 2.*work["limit"])
            terms["not_overmixed"] = damage
        if C["uniform"]["batch_share_in_the_vessel"] < C["uniform"]["share_limit"]:
            core = 0.
        done = all(C[k]["ok"] for k in ("uniform", "no_dry_powder", "smooth", "not_overmixed") if k in C)
    finish = (0.85 + 0.15*bool(in_place)) if close is None else (
        0.75 + 0.10*bool(in_place) + 0.15*bool(C[close]["ok"] and done))
    return dict(progress=clip(core*damage*keep*finish), keep=keep, finish=finish, **terms)


def record_hot_state(hob):
    """Read the same post-heat state as Hob.record_food_peak, from its external instance wrapper.

    Called at every existing material/heat update (10 ms), before species/heat exchange. The extra read is only
    for the explicitly selected porridge contract. It cannot affect the dynamics or stir-sauce's record.
    """
    if getattr(hob, "task_baseline", None) is None or hob.knob.s_level <= 0:
        return
    import numpy as np
    from scipy.spatial.transform import Rotation
    x, temp, mass, flags = hob.w.cont.read("x", "temp", "mass", "flags")
    q = hob.w.bq[hob.pan.body]
    local = (x - q[:3].astype(np.float32)) @ Rotation.from_quat(q[3:]).as_matrix().astype(np.float32)
    inside = (flags == 0) & hob.pan.inside(local, margin=0.002)
    record_hot_sample(hob, local[inside], temp[inside], mass[inside])


def evaluate(world, info, *, _progress=True):
    """Separate opt-in judging from the frozen legacy evaluator; never patch its module globals."""
    from . import task
    if enabled(world.config) and world.config.task == "reheat-porridge":
        out = _porridge_evaluation(world, info)
    else:
        # Bake predicates and stir-sauce's own evaluator are reused without modification.
        out = task.evaluate(world, info)
    out["contract"] = contract_for(world.config)
    if enabled(world.config) and _progress:
        if "task_baseline" not in info:
            raise ValueError("Task-set scoring requires the hand-over baseline; rebuild/reset the task")
        bottom = float(world.heat.area.sum())*1e4 if world.heat is not None else 1.
        r = progress(world.config.task, out["criteria"], info["task_baseline"], bottom)
        out["progress"] = r["progress"]
        out["reward"] = dict(r, contract=CONTRACT, score_rule=f"1 if success else {task.FAIL_SCORE_SCALE} x progress")
        out["score"] = 1.0 if out["success"] else task.FAIL_SCORE_SCALE*r["progress"]
    return out


def _porridge_evaluation(world, info):
    """Porridge's revised measurement path (a copy of the legacy evaluator's measurement, with the changes below).

    Reuse the frozen portion thermometer. Keep mass accounting and report fields identical, including the empty
    vessel failure; the legacy evaluator's current-temperature reduction cannot handle an empty porridge vessel.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from . import materials as M
    from .materials import SOLID_D, SOLID_L, SOLID_S, SOLID_C
    from .task import portion_temperatures
    c, sc = world.config, world.scene
    spec_t = task_spec(c)
    crit = spec_t["criteria"]
    cont, v = world.cont, sc.vessel
    out = dict(task=c.task, contract=contract_for(c), criteria={}, measures={})
    q = world.bq[v.body]
    Rv = Rotation.from_quat(q[3:])
    x, spec, flags, state, temp, mass = cont.read("x", "spec", "flags", "state", "temp", "mass")
    spec = spec.astype(np.float64)
    local = Rv.inv().apply(x.astype(np.float64) - q[:3])
    active = flags == 0
    inside = active & v.inside(local, margin=0.002)
    total = float(spec.sum())
    batch = float(np.sum(info["batch"]))
    m_in = float(spec[inside].sum())
    where = dict(in_vessel=m_in, counter=float(spec[flags == 2].sum()), floor=float(spec[flags == 3].sum()),
                 airborne=float(spec[flags == 1].sum()), elsewhere=float(spec[active & ~inside].sum()),
                 on_wires=float(spec[flags == 5].sum()))
    out["measures"]["mass_kg"] = dict(batch=batch, total=total, **{k: round(val, 6) for k, val in where.items()})
    # how what is outside the vessel got there (continuum.k_exit keeps where each point left the vessel's envelope):
    # above the rim plane, or below it. Nothing may leave below it: that would be through the wall or the base
    xout, tout = cont.read("xout", "tout")
    left = ~inside & (tout >= 0.)
    over = left & (xout[:, 2] >= v.depth - 2.0*cont.dx)
    out["measures"]["left_vessel"] = dict(over_rim_kg=round(float(spec[over].sum()), 6),
                                          through_wall_kg=round(float(spec[left & ~over].sum()), 6),
                                          no_record_kg=round(float(spec[~inside & (tout < 0.)].sum()), 6))
    out["measures"]["mass_balance_error"] = abs(total - batch)/batch
    spilled = (where["counter"] + where["floor"] + where["airborne"])/batch
    # material on a tool or the rim is not spilt; material neither in the vessel nor on the counter is reported
    out["measures"]["outside_fraction"] = round((total - m_in)/batch, 5)
    vp = info["spacing"]**3
    portion = (info["volume_m3"]/spec_t["portions"]) if "portions" in spec_t else spec_t["portion_ml"]*1e-6
    C = out["criteria"]
    if "hot" in crit:
        k = crit["hot"]
        # Read current portions together, before selecting a coherent current or historical pair below.
        # coldest_portion_mean_of_point_peaks_c is reported and decides nothing.
        if enabled(c) and not inside.any():
            tmin, tmax, cells, pmin = 0., 0., 0, 0.
        else:
            tmin, tmax, cells = portion_temperatures(local[inside], temp[inside], mass[inside], portion, vp)
            tpeak = np.maximum(state[:, M.ST_PEAK], temp)
            pmin, _, _ = portion_temperatures(local[inside], tpeak[inside], mass[inside], portion, vp)
        tmin, tmax = round(tmin, 2), round(tmax, 2)              # judged as reported
        C["hot_through"] = dict(coldest_c=tmin, hottest_c=tmax, cells=cells, limit_c=k["min_c"],
                                spread_limit_k=k["spread_k"], coldest_portion_mean_of_point_peaks_c=round(pmin, 2),
                                ok=bool(tmin >= k["min_c"] and tmax - tmin <= k["spread_k"]))
    if "hot" in crit and enabled(c):
        # Retain simultaneous portions, not particle peaks. The current reading is considered as well, so an
        # owner-constructed valid state is judged by the same predicates without requiring any action sequence.
        current = dict(coldest_c=tmin, hottest_c=tmax, cells=cells, batch_share=round(m_in/batch, 5))
        baseline = info.get("task_baseline", {})
        reached = best_hot(getattr(world.heat, "hot_record", None), current, baseline.get("coldest_c", tmin))
        judged = current if reached is None else reached
        C["hot_through"].update(judged, current_coldest_c=tmin, current_hottest_c=tmax,
                               ok=hot_ok(judged, k["min_c"], k["spread_k"]))
    if "scorch_cm2" in crit:
        area = world.heat.burnt_area()*1e4 if world.heat is not None else 0.
        C["not_scorched"] = dict(value=round(area, 3), limit=crit["scorch_cm2"], unit="cm2", ok=bool(area <= crit["scorch_cm2"]))
    kc = crit["contained"]
    lim = kc["limit"] if isinstance(kc, dict) else kc
    beside = None
    if isinstance(kc, dict) and kc.get("below_rim"):
        # stir-sauce: sauce outside the pan below its rim (on the hob's glass beside it, down its outside, on the
        # counter near it) is out of the pan as well. It stays active mixture there (flag 0): the counter's flag is
        # given only to what leaves the grid region, 108 mm from the pan's axis, and 3 % of the batch lying on the hob
        # by the pan would read as nothing spilt. On the rim's top, on a
        # tool held over the pan, or in the air above the rim, it is not counted
        low = active & ~inside & (local[:, 2] < v.depth - 2.0*cont.dx)
        beside = float(spec[low].sum())/batch
        spilled += beside
        spilled = round(spilled, 5)                 # judged as it is reported
    C["contained"] = dict(value=round(spilled, 5), limit=lim, ok=bool(spilled <= lim))
    if beside is not None:
        C["contained"]["of_it_outside_the_pan_below_its_rim"] = round(beside, 5)
    p_now = q[:3]
    up = Rv.apply([0., 0., 1.])[2]
    moved = float(np.linalg.norm(p_now[:2] - sc.vessel_start[:2]))
    standing = bool(up > 0.985 and abs(p_now[2] - sc.vessel_start[2]) < 0.01)
    kv = crit["vessel_in_place"]
    tilt, height = round(math.degrees(math.acos(min(1., abs(up)))), 2), round(float(p_now[2] - sc.vessel_start[2]), 4)
    if isinstance(kv, dict) and kv.get("on_hob"):
        # stir-sauce: "the pan on the hob" is the pan standing upright, at the hob's height, with its whole base on
        # the hob's glass (Scene's static box "hob_glass"). The base is the pan's rigid base disc (tools.Vessel: a
        # dodecagon whose flats lie on the profile's flat floor, 64 mm; its corners at 66.3 mm), not its outer wall,
        # and not the bowls' "within 50 mm of where it started"
        _, (gx, gy, _), (hx, hy, _), _, _ = next(b for b in sc.static_boxes if b[0] == "hob_glass")
        base_r = float(v.profile[1, 0])/math.cos(math.pi/12)
        off = (abs(float(p_now[0]) - gx), abs(float(p_now[1]) - gy))
        on = bool(off[0] <= hx - base_r and off[1] <= hy - base_r)
        C["vessel_in_place"] = dict(tilt_deg=tilt, tilt_limit_deg=round(math.degrees(math.acos(0.985)), 2), height_m=height,
                                    height_limit_m=0.01, base_on_the_hob_glass=on,
                                    from_the_glass_centre_m=[round(off[0], 4), round(off[1], 4)],
                                    allowed_m=[round(hx - base_r, 4), round(hy - base_r, 4)], ok=bool(on and standing))
    else:
        C["vessel_in_place"] = dict(moved_m=round(moved, 4), tilt_deg=tilt, height_m=height, limit_m=kv,
                                    ok=bool(moved <= kv and standing))
    if crit.get("appliance_off") == "knob":
        st = sc.tools["knob"].state()
        C["appliance_off"] = dict(level=st["level"], ok=bool(st["level"] == 0))
    out["success"] = bool(all(k["ok"] for k in C.values()))
    out["measures"]["points"] = int(len(x))
    out["measures"]["solid_states"] = {n: round(float(spec[:, k].sum()), 6) for n, k in
                                       (("dry", SOLID_D), ("lumped", SOLID_L), ("suspended", SOLID_S), ("cooked", SOLID_C))}
    if inside.any():
        out["measures"]["mean_temp_c"] = round(float((temp[inside]*mass[inside]).sum()/mass[inside].sum()), 3)
    return out

"""Outcome of reheat-porridge: every 60 ml portion of the porridge in the pan at 74 C or more and no more than 15 K
between the coldest and the hottest, no more than 2 cm2 of the pan's bottom burnt on, none of it spilt, the pan where
it stood and the hob turned off. Under the ingredient-mixing-tasks-v1 and -v2 contracts the pan must stand on the
hob's glass, porridge outside the pan below its rim counts as spilt, and the best coherent portion reading the hob
recorded while it was on is judged with the current one (v1 also asks that the reading cover 99 % of the batch).

JUDGING holds the task's portion size and its criteria (task.TASKS; task.evaluate measures them). The contract
gradings (scoring for v1, scoring_v2 for v2; S below is that module) call the rest: contract_criteria(criteria)
gives the criteria under a contract (revise_spec); hand_over(env, baseline), install(env) and restore(env, judging)
give the hob the hand-over baseline, its temperature recorder and a checkpoint's record (capture_start,
task_runtime.TaskEnv); record_hot_sample(S, hob, local, temp, mass) keeps the hob's best reading (record_hot_state);
contract_evaluation(S, world, info) measures the state and progress_v1/progress_v2(S, C, baseline, bottom_cm2) give
the progress terms (core, damage, done, terms). verify(world, info, *, _progress=True) evaluates the state of the
World with the start record info and returns the evaluation dict (scoring.evaluate_criteria: task.evaluate, or the
contract grading when the selection chose one)."""
import copy
import math

from ingredient_mixing import scoring

JUDGING = dict(
    portion_ml=60.,
    criteria=dict(hot=dict(min_c=74., spread_k=15.), scorch_cm2=2.0, contained=0.01, vessel_in_place=0.05,
                  appliance_off="knob"))


def contract_criteria(criteria):
    return dict(criteria, vessel_in_place=dict(on_hob=True), contained=dict(limit=0.01, below_rim=True))


def hand_over(env, baseline):
    env.world.heat.task_baseline = dict(baseline)
    env.world.heat.hot_record = None       # settling is not part of the episode


def install(env):
    from ingredient_mixing.task_runtime import install_hot_recorder
    install_hot_recorder(env.world.heat)


def restore(env, judging):
    env.world.heat.task_baseline = copy.deepcopy(judging["baseline"])
    env.world.heat.hot_record = copy.deepcopy(judging["hot_record"])


def record_hot_sample(S, hob, local, temp, mass):
    """Called by the external post-heat measurement wrapper, with no physical writes."""
    baseline = getattr(hob, "task_baseline", None)
    if baseline is None or not S.enabled(hob.w.config) or hob.w.config.task != "reheat-porridge":
        return
    if hob.knob.s_level <= 0 or len(temp) == 0:
        return
    from ingredient_mixing.task import portion_temperatures
    cold, hot, cells = portion_temperatures(local, temp, mass, 60e-6, baseline["spacing"]**3)
    current = dict(coldest_c=round(cold, 2), hottest_c=round(hot, 2), cells=cells,
                   batch_share=round(float(mass.sum())/baseline["batch_mass"], 5))
    hob.hot_record = S.best_hot(getattr(hob, "hot_record", None), current, baseline["coldest_c"])


def progress_v1(S, C, baseline, bottom_cm2):
    core = S.hot_quality(C["hot_through"], baseline["coldest_c"])
    if C["hot_through"].get("batch_share", 0.) < 0.99:
        core = 0.
    damage = S.clip(1. - C["not_scorched"]["value"]/bottom_cm2)
    done = C["hot_through"]["ok"]
    terms = dict(hot=core, unburnt=damage)
    return core, damage, done, terms


def progress_v2(S, C, baseline, bottom_cm2):
    core = S.hot_quality(C["hot_through"], baseline["coldest_c"])
    damage = S.clip(1. - C["not_scorched"]["value"]/bottom_cm2)
    done = C["hot_through"]["ok"]
    terms = dict(hot=core, unburnt=damage)
    return core, damage, done, terms


def contract_evaluation(S, world, info):
    """Porridge's revised measurement path (a copy of the legacy evaluator's measurement, with the changes below).

    Reuse the frozen portion thermometer. Keep mass accounting and report fields identical, including the empty
    vessel failure; the legacy evaluator's current-temperature reduction cannot handle an empty porridge vessel.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ingredient_mixing import materials as M
    from ingredient_mixing.materials import SOLID_D, SOLID_L, SOLID_S, SOLID_C
    from ingredient_mixing.task import portion_temperatures
    c, sc = world.config, world.scene
    spec_t = S.task_spec(c)
    crit = spec_t["criteria"]
    cont, v = world.cont, sc.vessel
    out = dict(task=c.task, contract=S.contract_for(c), criteria={}, measures={})
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
        if S.enabled(c) and not inside.any():
            tmin, tmax, cells, pmin = 0., 0., 0, 0.
        else:
            tmin, tmax, cells = portion_temperatures(local[inside], temp[inside], mass[inside], portion, vp)
            tpeak = np.maximum(state[:, M.ST_PEAK], temp)
            pmin, _, _ = portion_temperatures(local[inside], tpeak[inside], mass[inside], portion, vp)
        tmin, tmax = round(tmin, 2), round(tmax, 2)              # judged as reported
        C["hot_through"] = dict(coldest_c=tmin, hottest_c=tmax, cells=cells, limit_c=k["min_c"],
                                spread_limit_k=k["spread_k"], coldest_portion_mean_of_point_peaks_c=round(pmin, 2),
                                ok=bool(tmin >= k["min_c"] and tmax - tmin <= k["spread_k"]))
    if "hot" in crit and S.enabled(c):
        # Retain simultaneous portions, not particle peaks. The current reading is considered as well, so an
        # owner-constructed valid state is judged by the same predicates without requiring any action sequence.
        current = dict(coldest_c=tmin, hottest_c=tmax, cells=cells, batch_share=round(m_in/batch, 5))
        baseline = info.get("task_baseline", {})
        reached = S.best_hot(getattr(world.heat, "hot_record", None), current, baseline.get("coldest_c", tmin))
        judged = current if reached is None else reached
        C["hot_through"].update(judged, current_coldest_c=tmin, current_hottest_c=tmax,
                               ok=S.hot_ok(judged, k["min_c"], k["spread_k"]))
    if "scorch_cm2" in crit:
        area = world.heat.burnt_area()*1e4 if world.heat is not None else 0.
        C["not_scorched"] = dict(value=round(area, 3), limit=crit["scorch_cm2"], unit="cm2",
                                 ok=bool(area <= crit["scorch_cm2"]))
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
        C["vessel_in_place"] = dict(tilt_deg=tilt, tilt_limit_deg=round(math.degrees(math.acos(0.985)), 2),
                                    height_m=height, height_limit_m=0.01, base_on_the_hob_glass=on,
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
                                       (("dry", SOLID_D), ("lumped", SOLID_L), ("suspended", SOLID_S),
                                        ("cooked", SOLID_C))}
    if inside.any():
        out["measures"]["mean_temp_c"] = round(float((temp[inside]*mass[inside]).sum()/mass[inside].sum()), 3)
    return out


def verify(world, info, *, _progress=True):
    return scoring.evaluate_criteria(world, info, _progress=_progress)

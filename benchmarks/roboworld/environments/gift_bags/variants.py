"""Seeded task catalogue: preset/task/seed. Seeds vary the start state only (see Config.seed users)."""
from .config import PRESETS, TASKS

POLICY_PRESETS = tuple(p for p in PRESETS if p != "dev")


_AVAILABILITY = None


def availability(refresh=False):
    """{preset: dict(available, reason, ...)}. A preset is unavailable when its cached start state cannot be used
    (the tote's shipping state torn while it was made): the catalogue, the default task, `gar doctor` and a contract
    test report it with the reason instead of letting reset crash. A start state that is merely not
    built yet is available: the first launch builds and caches it."""
    global _AVAILABILITY
    if _AVAILABILITY is not None and not refresh:
        return _AVAILABILITY
    out = {p: dict(available=True, reason=None) for p in POLICY_PRESETS}
    try:
        from . import bagnet as B, scene as S
        if S.FLAP_UP:
            st = B.shipping_state_status(S.BAG_CELLS)
            out["counter"]["start_state"] = {k: st.get(k) for k in ("key", "cached", "where", "sound", "splits", "flat_thickness_mm")}
            if st["cached"] and not st["sound"]:
                out["counter"].update(available=False, reason=st["reason"])
            elif not st["cached"]:
                out["counter"]["note"] = ("the tote's shipping state and its handle assembly are not shipped or cached for "
                                          "this code: the first launch builds them (measured on four solver "
                                          "threads: 24-30 min and 8 min) and caches them")
    except Exception as exc:                                    # the check itself must never take the catalogue down
        out["counter"].update(available=False, reason=f"start-state check failed: {type(exc).__name__}: {exc}")
    # which start states exist for the current code (shipped with it, in the local cache, or missing: a launch then
    # builds first: a first launch of an uncached wrapped variant took 466 s against a 600 s wait)
    try:
        from . import prebuild as PB
        tab = PB.table()
        for p in POLICY_PRESETS:
            rows = tab.get(p, [])
            out[p]["start_states"] = [{k: r.get(k) for k in ("name", "key", "where", "box", "clearance_m", "wrap_stock")
                                       if r.get(k) is not None} for r in rows]
            out[p]["start_states_ready"] = [sum(r["where"] != "missing" for r in rows), len(rows)]
        out["home"]["note"] = ("lift-lid needs no start state. unwrap-gift and open-present draw one of 18 wrapped presents by "
                               "seed (start_states): a missing one is built on the first launch (8-20 min) and cached")
    except Exception as exc:
        out["home"]["start_states_error"] = f"{type(exc).__name__}: {exc}"
    _AVAILABILITY = out
    return out


def default_preset():
    """The first available preset in catalogue order (counter, home, receiving)."""
    av = availability()
    return next((p for p in POLICY_PRESETS if av[p]["available"]), POLICY_PRESETS[0])


def resolve(preset=None, task=None, seed=0):
    if preset is None and task is not None:
        # task names are unique across presets, so a task alone selects its preset (`gar run gift-bags --task lift-lid`)
        owners = [p for p in POLICY_PRESETS if task in TASKS[p]]
        if not owners:
            raise ValueError(f"Unknown task {task!r}; choose one of "
                             + "; ".join(f"{p}: {', '.join(TASKS[p])}" for p in POLICY_PRESETS))
        preset = owners[0]
    preset = default_preset() if preset is None else preset
    if preset not in POLICY_PRESETS:
        raise ValueError(f"Unknown preset; choose {POLICY_PRESETS}")
    av = availability()[preset]
    if not av["available"]:
        ok = [p for p in POLICY_PRESETS if availability()[p]["available"]]
        raise ValueError(f"The {preset} preset is unavailable: {av['reason']}. Available presets: {ok}")
    task = task or TASKS[preset][0]
    if task not in TASKS[preset]:
        raise ValueError(f"{task} is not a {preset} task; choose {TASKS[preset]}")
    if type(seed) is not int or not 0 <= seed < 10000:
        raise ValueError("Seed must be an integer 0..9999")
    return dict(preset=preset, task=task, seed=seed)


# Seeded placement of the items the clerk or the person works with (seed 0: the authored layout).
# Ranges [A]: a few centimetres and degrees, inside the reach study (every stack position within 0.30-0.50 x
# -0.05-0.16 m had continuous FR3 paths) and clear of neighbouring items.
POSE_RANGES = {
    "counter": dict(stack=dict(dx=.02, dy=.02, yaw_deg=6.), boards=dict(dx=.02, dy=.02, yaw_deg=10.),
                    candle=dict(dx=.02, dy=.02, yaw_deg=30.)),
    "home": dict(box=dict(dx=.03, dy=.02, yaw_deg=15.)),
    # receiving: the sealed carton on the table. The reach study found continuous FR3 paths to every flap at +-10 and
    # +-15 deg but none of the tried paths from the ready pose to the tape tab once the carton is yawed, nor 20 mm closer
    # to the robots, so seeds move it +-30 mm across and +-10 mm toward/away from the robots only
    "receiving": dict(carton=dict(dx=.03, dy=.01, yaw_deg=0.)),
}


def placement(preset, seed):
    """{item: (dx, dy, yaw rad)} for a seed: zero for seed 0, uniform within POSE_RANGES otherwise."""
    import math
    import numpy as np
    out = {}
    rng = np.random.default_rng(seed)
    for item, r in POSE_RANGES.get(preset, {}).items():
        u = rng.uniform(-1., 1., 3) if seed else np.zeros(3)
        out[item] = (float(u[0]*r["dx"]), float(u[1]*r["dy"]), float(math.radians(u[2]*r["yaw_deg"])))
    return out


# Seeded variation of the home preset's present: the box (a two-piece set-up box with the photo frame that fits it;
# giftbox.box_spec, params.GIFT_BOXES / PHOTO_FRAMES), the lid's clearance per side (params.LID_CLEARANCES) and the wrap
# stock (params.WRAP_STOCKS; unwrap-gift and open-present only). The first value of each axis is the authored one: seed 0
# draws nothing. Other seeds draw each axis uniformly from a random stream separate from placement()'s, so a seed's pose
# does not depend on its present.
HOME_AXES = dict(box=("authored_180x130x55", "uline_window_7.5x7.5x2", "uline_setup_8.5x11x2"),
                 clearance_m=(.5e-3, .8e-3, 1.0e-3),
                 wrap_stock=("commercial_81gsm", "consumer_53gsm"))


def home_variant(seed):
    """{box, clearance (m), wrap_stock} of the home preset's present for a seed (seed 0: the authored present)."""
    import numpy as np
    if seed:
        rng = np.random.default_rng([int(seed), 1])
        i = {k: int(rng.integers(len(v))) for k, v in HOME_AXES.items()}
    else:
        i = {k: 0 for k in HOME_AXES}
    return dict(box=HOME_AXES["box"][i["box"]], clearance=HOME_AXES["clearance_m"][i["clearance_m"]],
                wrap_stock=HOME_AXES["wrap_stock"][i["wrap_stock"]])


def contract():
    av = availability()
    return dict(supported=True, selectors=["preset", "task", "seed"], presets=list(POLICY_PRESETS),
                tasks={k: list(TASKS[k]) for k in POLICY_PRESETS}, seed_range=[0, 9999],
                availability=av, unavailable_tasks={f"{k}/{t}": av[k]["reason"] for k in POLICY_PRESETS
                                                    for t in TASKS[k] if not av[k]["available"]},
                default=dict(preset=default_preset(), task=TASKS[default_preset()][0], seed=0),
                pose_ranges=POSE_RANGES, home_axes={k: list(v) for k, v in HOME_AXES.items()},
                qualification="Six named tasks over three presets (boutique counter, dining table, receiving table). Seed 0 "
                              "is the authored layout. Other seeds place the worked items (counter: the "
                              "bag stack, the boards, the candle; home: the gift box; receiving: the sealed carton) by a rigid "
                              "move within pose_ranges: start states are built once at the authored pose and moved rigidly, so "
                              "every seed starts settled. Home seeds also draw the present from home_axes: the box (the authored "
                              "180 x 130 x 55 mm set-up box with its 0.35 kg frame; a Uline 7 1/2 x 7 1/2 x 2 in window gift box "
                              "with an IKEA FISKBO 10x15 frame, 0.11 kg; a Uline 8 1/2 x 11 x 2 in set-up box with a FISKBO "
                              "13x18 frame, 0.15 kg), the lid clearance per side (0.5, 0.8, 1.0 mm) and, for unwrap-gift and "
                              "open-present, the wrap stock (81 g/m2 commercial counter roll; 53 g/m2 consumer roll wrap); each "
                              "wrapped variant's start state is built on first use and cached. A phantom lid lift at 10 cm/s drags the base of the "
                              "two catalogue boxes (lighter frames, larger lids: 44 and 52 mm at 0.5 mm clearance; the 8 1/2 x 11 "
                              "box still 8 mm at 1.0 mm), so lift-lid there needs a slow lift; at 1 cm/s every base stays down. "
                              "Bag size and paper stock of the counter's totes are not varied.")


def task_resolve(selection):
    from .task import INSTRUCTIONS
    v = resolve(selection.get("preset"), selection.get("task"), selection.get("seed", 0))
    return dict(task_id=f"gift-bags/v1/{v['preset']}/{v['task']}/seed-{v['seed']}",
                description=INSTRUCTIONS[v["task"]],
                backend_options=dict(preset=v["preset"], task=v["task"], seed=v["seed"]),
                selection=v, split=None)

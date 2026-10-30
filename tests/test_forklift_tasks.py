"""Forklift task catalogue and evaluator rules: the load set-down rule on rack beams, the set-down checks of
ramp-down-loaded, start-and-park as a real parking job, the yard speed limit, and the seed variation of every work
order's target. No physics: stub worlds carry the state the evaluator reads.
"""

import math
from types import SimpleNamespace

import numpy as np


def _tasks():
    from forklift.config import TASKS
    return [(p, t) for p, ts in TASKS.items() if p != "lab" for t in ts]


def _layout(cfg):
    from forklift import facility, task as TK
    return {"warehouse": facility.warehouse, "dock": facility.dock, "yard": facility.yard}[cfg.preset](
        np.random.default_rng(1000 + cfg.seed), reserved=TK.reserved_locations(cfg))


def _events():
    from forklift.events import Events
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    ev._maps_ready = True
    return ev


def _episode(peak, impulse, speed, mover="load:carried", parts=None):
    return dict(peak_N=peak, impulse_Ns=impulse, episodes=1, first_t=1.0, max_episode_Ns=impulse,
                impacts=[dict(t=1.0, onset_speed=speed, impulse_Ns=impulse, peak_N=peak, mover=mover, parts=parts)])


def _shared_world(ev, preset="warehouse", t=30.0):
    return SimpleNamespace(events=ev, scene=SimpleNamespace(c=SimpleNamespace(preset=preset), loads={}), sim_time=t)


BEAM = ["carried:bottom_deck", "C0_beam2_1_12.98"]
UPRIGHT = ["carried:stringer", "C0_up3_12.98"]
STORED = ["carried:wrapped", "stored_C-w-03-1-1_load"]


# ------------------------------------------------------------------------------------------------ set-down rule
def _strike_count(strikes):
    from forklift.evaluator import shared
    ev = _events()
    ev.strikes = strikes
    crit, det = shared(_shared_world(ev), {"task": "put-away-rack"})
    return {c["name"]: c for c in crit}["no_strike"]["measured"], det["strikes"]


def test_a_gentle_set_down_on_beams_is_not_a_strike_but_a_fast_one_is():
    from forklift.evaluator import SET_DOWN_PEAK_N, SET_DOWN_SPEED
    # resting on the beams for the rest of the episode: impulse far beyond the strike limit, peak within the bound
    gentle = _episode(0.5 * SET_DOWN_PEAK_N, 3.0e5, 0.5 * SET_DOWN_SPEED, parts=BEAM)
    assert _strike_count({("load", "rack"): gentle})[0] == 0
    # the same load met the beams at 0.35 m/s
    fast = _episode(0.5 * SET_DOWN_PEAK_N, 3.0e5, 0.35, parts=BEAM)
    n, det = _strike_count({("load", "rack"): fast})
    assert n == 1 and det[0]["load"] == "carried"
    # slow at its onset but pressed past the bound (pushed down or across the beams by the truck)
    assert _strike_count({("load", "rack"): _episode(1.5 * SET_DOWN_PEAK_N, 3.0e5, 0.02, parts=BEAM)})[0] == 1


def test_a_load_against_an_upright_or_a_stored_load_is_judged_like_any_strike():
    # slow, but pressed against the upright with more than a strike's force: not exempt
    assert _strike_count({("load", "rack"): _episode(3000.0, 40.0, 0.03, parts=UPRIGHT)})[0] == 1
    # driven into the neighbouring stored load at 0.3 m/s: impulse over the limit
    assert _strike_count({("load", "rack"): _episode(1500.0, 80.0, 0.30, parts=STORED)})[0] == 1
    # brushed: below both limits
    assert _strike_count({("load", "rack"): _episode(800.0, 5.0, 0.30, parts=STORED)})[0] == 0
    # leaning on an upright at rest for a long time (impulse grows with time, force does not): not a strike
    assert _strike_count({("load", "rack"): _episode(600.0, 5.0e4, 0.01, parts=UPRIGHT)})[0] == 0


def test_a_load_resting_against_a_wall_counts_only_above_its_weight():
    """A bale set against a trailer's nose wall (as load-bales asks) keeps a preload there by its own friction: at most
    its weight. More than its weight, slow or not, is something driving it into the wall."""
    from forklift.evaluator import shared
    ev = _events()
    world = _shared_world(ev)
    world.scene.loads = {"bale_1": {"mass": 520.0}}
    nose = ["bale_1:box", "door3:nose"]
    for peak, n in ((3055.0, 0), (0.9 * 520 * 9.81, 0), (1.2 * 520 * 9.81, 1), (8000.0, 1)):
        ev.strikes = {("load:bale_1", "trailer_wall"): _episode(peak, 7.0e3, 0.003, mover="load:bale_1", parts=nose)}
        assert {c["name"]: c for c in shared(world, {"task": "load-bales"})[0]}["no_strike"]["measured"] == n, peak
    # the same force at an impact's speed is a strike
    ev.strikes = {("load:bale_1", "trailer_wall"): _episode(3055.0, 60.0, 0.3, mover="load:bale_1", parts=nose)}
    assert {c["name"]: c for c in shared(world, {"task": "load-bales"})[0]}["no_strike"]["measured"] == 1


def test_the_set_down_rule_holds_with_loads_keyed_by_name():
    """With the latches keyed per load ('load:<name>', what the dynamic stored loads need so that their resting
    contact does not merge with the carried pallet's episodes), the rule reads the same."""
    from forklift.evaluator import SET_DOWN_PEAK_N
    stored_resting = _episode(0.2 * SET_DOWN_PEAK_N, 4.0e5, 0.0, mover="load:stored_C-w-03-1-1",
                              parts=["stored_C-w-03-1-1:bottom_deck", "C0_beam2_1_12.98"])
    fast = _episode(0.5 * SET_DOWN_PEAK_N, 3.0e5, 0.30, parts=BEAM)
    n, det = _strike_count({("load:stored_C-w-03-1-1", "rack"): stored_resting, ("load:carried", "rack"): fast})
    assert n == 1 and det[0]["load"] == "carried"
    assert _strike_count({("load:stored_C-w-03-1-1", "rack"): stored_resting})[0] == 0


def test_loads_on_the_conveyor_rollers_are_watched_like_beams():
    """A load's contact with the rollers is a set-down only at most SET_DOWN_SPEED along the contact normal and within
    ROLLER_PEAK_N (riding the rollers keeps the episode's onset); a 2 t coil slammed onto them is a strike."""
    from forklift.evaluator import ROLLER_PEAK_N, SET_DOWN_SPEED
    assert _strike_count({("load", "conveyor_rollers"): _episode(0.7 * ROLLER_PEAK_N, 1.0e6, 0.5 * SET_DOWN_SPEED)})[0] == 0
    assert _strike_count({("load", "conveyor_rollers"): _episode(1.0e5, 1.0e6, 0.4)})[0] == 1        # slammed on
    assert _strike_count({("load", "conveyor_rollers"): _episode(1.2 * ROLLER_PEAK_N, 1.0e6, 0.05)})[0] == 1
    assert _strike_count({("truck", "conveyor_rollers"): _episode(1.0e5, 1.0e3, 0.4, mover="truck")})[0] == 1


def test_loads_on_a_trailer_floor_are_watched_and_the_truck_drives_on_it():
    from forklift.evaluator import SET_DOWN_SPEED, TRAILER_FLOOR_PEAK_N
    floor = ["carried:bottom_deck", "door1:floor"]
    assert _strike_count({("load:carried", "trailer_floor"): _episode(0.5 * TRAILER_FLOOR_PEAK_N, 1.0e6,
                                                                       0.5 * SET_DOWN_SPEED, parts=floor)})[0] == 0
    assert _strike_count({("load:carried", "trailer_floor"): _episode(5.0e4, 1.0e6, 0.5, parts=floor)})[0] == 1
    assert _strike_count({("truck", "trailer_floor"): _episode(1.0e5, 1.0e6, 0.5, mover="truck")})[0] == 0


def test_a_strike_never_drops_out_of_the_record():
    """A strike stays counted after many later episodes, although only the last EPISODES_KEPT episodes per mover and
    class are kept."""
    from forklift.events import EPISODES_KEPT, Events
    from forklift.evaluator import shared
    ev = _events()
    st = ev.strikes.setdefault(("truck", "column"), dict(peak_N=0.0, impulse_Ns=0.0, episodes=0, first_t=0.0,
                                                         max_episode_Ns=0.0, impacts=[]))

    def close(t, peak, impulse):
        e = dict(t=t, onset_speed=1.0, onset_normal=1.0, impulse_Ns=impulse, peak_N=peak, mover="truck", parts=None,
                 t_end=t + 0.01)
        st["impacts"] = (st["impacts"] + [e])[-EPISODES_KEPT:]
        Events._latch(st, e)
    close(1.0, 5000.0, 10.0)                                 # a strike
    for k in range(3 * EPISODES_KEPT):                       # then many brushes under both limits
        close(2.0 + k, 100.0, 0.1)
    crit, det = shared(_shared_world(ev), {"task": "carry-to-floor-bay"})
    assert {c["name"]: c for c in crit}["no_strike"]["measured"] == 1 and det["strikes"][0]["peak_N"] == 5000.0


def _put_away_world(seed, frame_swap=False):
    """The carried pallet resting on the target bay and level's beams (frame_swap: on the same bay and level of the
    other frame of a back-to-back row, or the facing row)."""
    from forklift import params as P
    from forklift.events import Events
    from forklift.variants import resolved_start
    cfg, start = resolved_start("put-away-rack", seed)
    L = _layout(cfg)
    row, face, bay, level, slot = start["rack_location"].split("-")
    loc = L.zones["rack_locations"][start["rack_location"]]
    frame = "1" if (row == "B" and face == "e") else "0"
    if frame_swap:
        row, frame = {("A", "0"): ("B", "0"), ("B", "0"): ("A", "0"), ("B", "1"): ("C", "0"), ("C", "0"): ("B", "1")}[(row, frame)]
    beams = sorted(b["name"] for b in L.boxes if b["name"].startswith(f"{row}{frame}_beam{int(bay) - 1}_{level}_"))
    assert len(beams) == 2
    labels = ["carried:stringer", *beams]
    body_q = np.zeros((2, 7))
    sgn = 1.0 if face == "w" else -1.0
    body_q[0] = (loc["aisle_x"] + sgn * (P.GMA_PALLET["L"] / 2 - 0.076), loc["y"], loc["z"], 0.0, 0.0, 0.0, 1.0)
    body_q[1] = (30.0, 24.6, 0.10, 0.0, 0.0, 0.0, 1.0)
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    world = SimpleNamespace(
        events=ev, sim_time=60.0, fault=None,
        T=SimpleNamespace(bodies={"sideshift": 1}, anchors={"sideshift": (0.0, 0.0, 0.0)}, shifts={}),
        scene=SimpleNamespace(c=SimpleNamespace(preset="warehouse"), zones=L.zones, layout=L, loads={"carried": {"body": 0}},
                              ground_height=lambda x, y: 0.0, load_start_pose={}),
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q)),
        solver=SimpleNamespace(mj_model=SimpleNamespace(ngeom=3), mj_data=SimpleNamespace(
            ncon=2, contact=[SimpleNamespace(geom1=0, geom2=1), SimpleNamespace(geom1=0, geom2=2)])),
        mj_geom_label=lambda g: labels[g])
    return world, start


def test_put_away_counts_only_the_target_rows_beams():
    from forklift.evaluator import evaluate
    for seed in (0, 1, 2, 3, 11):
        world, start = _put_away_world(seed)
        r = evaluate(world, start)
        assert r["success"], (seed, [c for c in r["criteria"] if not c["passed"]])
        world, start = _put_away_world(seed, frame_swap=True)
        by = {c["name"]: c for c in evaluate(world, start)["criteria"]}
        assert not by["on_both_beams"]["passed"] and by["on_both_beams"]["measured"]["elsewhere"], seed


def test_a_disturbed_stored_load_fails_put_away():
    from forklift.evaluator import evaluate
    world, start = _put_away_world(11)
    bq = world.state.body_q.numpy()
    world.state.body_q = SimpleNamespace(numpy=lambda: np.vstack([bq, [(14.4, 9.0, 1.6, 0.0, 0.0, 0.0, 1.0)]]))
    world.scene.loads["stored_C-w-03-1-1"] = {"body": 2}
    world.scene.load_start_pose = {"stored_C-w-03-1-1": [14.4, 9.06, 1.6]}            # pushed 60 mm
    by = {c["name"]: c for c in evaluate(world, start)["criteria"]}
    assert not by["stored_loads_undisturbed"]["passed"] and by["stored_loads_undisturbed"]["measured"]["max_m"] > 0.05


# ------------------------------------------------------------------------------------------------ load-bales
def _bales_world(seed=0):
    """load-bales' resolved start in a stub world: both docked trailers at their spots, every load at its start centre,
    the truck facing the target trailer with its front axle 3 m inside the door face, clean latches. place() sets a
    bale in the target trailer's frame (x from the rear sill, y left), its bottom on the strips by default."""
    from forklift import facility as F, params as P
    from forklift.mathx import quat_to_R
    from forklift.variants import resolved_start
    cfg, start = resolved_start("load-bales", seed)
    L = _layout(cfg)
    zq = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    q, loads, trailers = [], {}, {}
    for dn in sorted(start["trailers"]):
        d = L.zones["doors"][dn]
        trailers[dn] = dict(body=len(q), L=P.TRAILER_INSIDE["L"], W=P.TRAILER_INSIDE["W"])
        q.append((d["x"], d["face_y"] + F.BUMPER + F.REAR_FRAME + 0.006, start["bed_z"]) + zq)
    for ld in start["loads"]:
        loads[ld["name"]] = dict(body=len(q), size=tuple(ld["size"]), mass=float(ld["mass"]))
        q.append((ld["centre"][0], ld["centre"][1], ld["centre"][2] + ld["size"][2] / 2, 0.0, 0.0, 0.0, 1.0))
    door = L.zones["doors"][start["target"]["door"]]
    ch = len(q)
    q.append((door["x"], door["face_y"] - 3.0, 0.0) + zq)
    bq = np.array(q, float)
    world = SimpleNamespace(events=_events(), sim_time=300.0, fault=None,
                            T=SimpleNamespace(bodies={"sideshift": ch}, anchors={"sideshift": (0.0, 0.0, 0.0)}, shifts={}),
                            scene=SimpleNamespace(c=SimpleNamespace(preset="dock"), zones=L.zones, loads=loads, trailers=trailers),
                            state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: bq)),
                            body_pose=lambda name: (bq[ch, :3].copy(), quat_to_R(bq[None, ch, 3:7])[0]))
    o = bq[trailers[start["target"]["door"]]["body"], :3]
    bl, bw, bh = P.BALE["size"]

    def place(name, cy, gap=0.0, yaw_deg=0.0, z=P.DUNNAGE["size"][1]):
        a = math.radians(yaw_deg)
        cx = P.TRAILER_INSIDE["L"] - gap - (bl / 2 * abs(math.cos(a)) + bw / 2 * abs(math.sin(a)))
        y = math.pi / 2 + a
        bq[loads[name]["body"]] = (o[0] - cy, o[1] + cx, o[2] + z + bh / 2, 0.0, 0.0, math.sin(y / 2), math.cos(y / 2))
    return world, start, bq, place, ch, door["face_y"]


def test_load_bales_accepts_every_side_by_side_placement_and_rejects_the_wrong_ones():
    """Two bales each on its own nose strip pair pass at any spacing and when touching at a small common yaw; a third
    bale loaded on top, and fork tips left in the trailer, fail."""
    from forklift import params as P
    from forklift.evaluator import evaluate
    world, start, bq, place, ch, face = _bales_world(0)
    q0 = bq.copy()

    def run(setup, axle_inside=3.0):
        bq[:] = q0
        setup()
        bq[ch, 1] = face - axle_inside
        return evaluate(world, start)
    by = lambda r: {c["name"]: c["passed"] for c in r["criteria"]}
    pc = 0.411                                     # the nose strip pairs' centres (left +, right -)
    for out in (0.0, 0.02, 0.15):                  # 60, 100 and 360 mm apart
        r = run(lambda: (place("bale_0", pc + out), place("bale_1", -pc - out)))
        assert r["success"] and r["score"] == 1.0, (out, [c for c in r["criteria"] if not c["passed"]])
    a, bw = math.radians(0.5), P.BALE["size"][1]   # touching along their long faces, both yawed 0.5 deg
    r = run(lambda: (place("bale_1", -pc, 0.01, 0.5), place("bale_0", -pc + bw * math.cos(a), 0.01 + bw * math.sin(a), 0.5)))
    assert r["success"] and abs(r["criteria"][-3]["measured"]["clearance_m"]) < 1e-6, r["criteria"][-3]
    r = run(lambda: (place("bale_0", pc), place("bale_1", -pc), place("bale_2", 0.0, z=P.DUNNAGE["size"][1] + P.BALE["size"][2])))
    assert not by(r)["two_bales_at_nose"] and r["details"]["bales"][2]["in_trailer"]
    r = run(lambda: (place("bale_0", pc), place("bale_1", -pc)), axle_inside=1.001)    # fork tips 0.6 m past the face
    assert not by(r)["truck_back_on_dock"] and r["score"] < 1.0
    r = run(lambda: (place("bale_0", pc), place("bale_1", -pc, gap=0.10)))           # 100 mm short of the wall
    assert not by(r)["two_bales_at_nose"]


def test_load_bales_against_the_front_wall_is_the_front_face_and_resting_there_is_no_strike():
    """Two bales turned 10 deg, each touching the wall at one corner, are not against it (turned 2 deg they are); a
    bale resting on the nose wall after touching it while being lowered is not a strike, a 12 kN hit is; a side wall
    keeps the shared rule."""
    from forklift.evaluator import evaluate
    world, start, bq, place, ch, face = _bales_world(0)
    door = start["target"]["door"]
    q0 = bq.copy()

    def run(yaw=0.0, wall=None):
        bq[:] = q0
        place("bale_0", 0.411, 0.0, yaw)
        place("bale_1", -0.411, 0.0, yaw)
        world.events.reset()
        if wall:
            onset, normal, peak, imp, part = wall
            world.events.strikes = {("load:bale_1", "trailer_wall"): dict(
                peak_N=peak, impulse_Ns=imp, episodes=1, first_t=200.0, max_episode_Ns=imp,
                impacts=[dict(t=200.0, t_end=300.0, onset_speed=onset, onset_normal=normal, impulse_Ns=imp, peak_N=peak,
                              mover="load:bale_1", parts=["bale_1:box", part])])}
        bq[ch, 1] = face - 3.0
        return {c["name"]: c["passed"] for c in evaluate(world, start)["criteria"]}
    assert run(2.0)["two_bales_at_nose"] and not run(10.0)["two_bales_at_nose"]
    assert run(wall=(0.25, 0.0, 300.0, 30000.0, f"{door}:nose"))["no_strike"]            # lowered against it, resting
    assert run(wall=(0.12, 0.12, 1500.0, 30000.0, f"{door}:nose"))["no_strike"]          # touched below 2 kN, stayed
    assert not run(wall=(0.30, 0.30, 12000.0, 400.0, f"{door}:nose"))["no_strike"]       # driven into it
    assert not run(wall=(0.25, 0.0, 300.0, 30000.0, f"{door}:wall_l"))["no_strike"]      # a side wall: the shared rule


def test_load_bales_strike_record_strips_and_floor_events():
    """A fast >2 kN nose strike stays in the record after 100 later episodes; a strip pressed on the nose after a
    0.12 m/s approach is no strike; one 2 s floor scrape is one contact event, costing less than a bale stranded on a
    floor, which costs less than one dropped there."""
    from forklift import params as P
    from forklift.evaluator import evaluate
    from forklift.events import EPISODES_KEPT, Events
    world, start, bq, place, ch, face = _bales_world(0)
    door = start["target"]["door"]
    nose = ["bale_1:box", f"{door}:nose"]
    q0 = bq.copy()

    def run(eps=(), floor=None, onset=None, spare_on_floor=False):
        bq[:] = q0
        place("bale_0", 0.411)
        place("bale_1", -0.411)
        if spare_on_floor:
            bq[world.scene.loads["bale_2"]["body"]] = (15.5, 2.0, P.BALE["size"][2] / 2, 0.0, 0.0, 0.0, 1.0)
        bq[ch, 1] = face - 3.0
        world.events.reset()
        for key, closed in eps:
            st = world.events.strikes.setdefault(key, dict(peak_N=0.0, impulse_Ns=0.0, episodes=0, first_t=0.0,
                                                           max_episode_Ns=0.0, impacts=[]))
            for e in closed:
                st["impacts"] = (st["impacts"] + [dict(e)])[-EPISODES_KEPT:]
                Events._latch(st, dict(e))
        world.events.load_floor, world.events.load_floor_onset = dict(floor or {}), dict(onset or {})
        r = evaluate(world, start)
        return {c["name"]: c["passed"] for c in r["criteria"]}, r["score"]
    ep = lambda t, v, peak, imp, parts=nose, mover="load:bale_1": dict(t=t, t_end=t + 0.01, onset_speed=v, onset_normal=v,
                                                                       peak_N=peak, impulse_Ns=imp, mover=mover, parts=parts)
    X, Z, Y = ep(100.0, 0.15, 2500.0, 10.0), ep(110.0, 0.02, 4000.0, 30.0), ep(120.0, 0.30, 1200.0, 80.0)
    chatter = [ep(130.0 + 0.1 * i, 0.001, 0.0, 0.0) for i in range(100)]
    assert not run([(("load:bale_1", "trailer_wall"), [X, Z, Y] + chatter)])[0]["no_strike"]
    strip = ep(200.0, 0.12, 900.0, 30000.0, ["nose_dunnage_Ra:box", f"{door}:nose"], "load:nose_dunnage_Ra")
    assert run([(("load:nose_dunnage_Ra", "trailer_wall"), [strip])])[0]["no_strike"]
    xy = (start["truck"][0], face - 1.2)
    scrape = [(60.0 + 0.252 * i, xy) for i in range(9)]                         # one 2 s contact, as events.py latches it
    _, one_scrape = run(floor={"bale_1": scrape}, onset={"bale_1": [(60.0, 0.02)]})
    _, two_brushes = run(floor={"bale_1": [(60.0, xy), (90.0, xy)]})
    _, stranded = run(floor={"bale_2": [(150.0, (15.5, 2.0))]}, onset={"bale_2": [(150.0, 0.02)]}, spare_on_floor=True)
    _, dropped = run(floor={"bale_2": [(150.0, (15.5, 2.0))]}, onset={"bale_2": [(150.0, 4.4)]}, spare_on_floor=True)
    assert one_scrape > stranded > dropped and two_brushes > stranded, (one_scrape, two_brushes, stranded, dropped)


def test_load_bales_reward_rises_with_the_job_and_charges_damage():
    """Untouched scores 0, one bale placed scores the same with the truck in the trailer or out, both placed with the
    truck in scores more, the job done 1; a dropped spare bale costs. A bale that brushed a floor while carried and
    ended placed keeps its progress and costs less than a drop."""
    from forklift import params as P
    from forklift.evaluator import evaluate
    world, start, bq, place, ch, face = _bales_world(900)
    q0 = bq.copy()

    def score(setup, axle_inside=3.0, floor=None):
        bq[:] = q0
        world.events.load_floor = dict(floor or {})
        setup()
        bq[ch, 1] = face - axle_inside
        return evaluate(world, start)["score"]

    def both():
        place("bale_1", 0.411)
        place("bale_0", -0.411)

    def spare_on_the_floor():
        both()
        bq[world.scene.loads["bale_2"]["body"]] = (15.5, 2.0, P.BALE["size"][2] / 2, 0.0, 0.0, 0.0, 1.0)
    idle = score(lambda: None)
    one_in = score(lambda: place("bale_1", 0.411), axle_inside=-12.0)
    one_out = score(lambda: place("bale_1", 0.411))
    both_in = score(both, axle_inside=-12.0)
    done = score(both)
    dropped = score(spare_on_the_floor, floor={"bale_2": [(150.0, (15.5, 2.0))]})
    brushed = score(both, floor={"bale_1": [(60.0, (start["truck"][0], face - 1.2))]})
    assert idle == 0.0 and one_in == one_out and 0.0 < one_out < both_in < done == 1.0, (idle, one_in, one_out, both_in, done)
    assert dropped < brushed < done, (dropped, brushed)          # the same finished job: a drop costs more than a brush


def _yard_world(pallet_z, tips_xy, tips_z, rest=3.0, max_speed=1.5):
    from forklift.events import Events
    from forklift.facility import yard
    L = yard(np.random.default_rng(1000))
    dz = L.zones["drop_zone"]
    c = ((dz["x"][0] + dz["x"][1]) / 2, (dz["y"][0] + dz["y"][1]) / 2)
    body_q = np.zeros((3, 7))
    body_q[0] = (c[0], c[1], pallet_z, 0.0, 0.0, 0.0, 1.0)                    # the pallet, its bottom at pallet_z
    body_q[1] = (tips_xy[0] - 1.601, tips_xy[1], tips_z - 0.104, 0.0, 0.0, 0.0, 1.0)   # sideshift: tips at tips_xy
    body_q[2] = (tips_xy[0] - 2.0, tips_xy[1], 0.3, 0.0, 0.0, 0.0, 1.0)
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    ev.rest_s, ev.max_speed, ev.ramp_max_speed = rest, max_speed, 1.0
    return SimpleNamespace(
        events=ev, sim_time=100.0, fault=None,
        T=SimpleNamespace(bodies={"sideshift": 1}, anchors={"sideshift": (0.0, 0.0, 0.0)}, shifts={}),
        scene=SimpleNamespace(c=SimpleNamespace(preset="yard"), zones=L.zones, layout=L, loads={"carried": {"body": 0}},
                              ground_height=lambda x, y: 0.0),
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q))), c


def test_ramp_down_needs_the_pallet_set_down_and_the_forks_out():
    from forklift.evaluator import evaluate
    st = {"task": "ramp-down-loaded"}
    # set down: on the asphalt, fork tips 1.5 m from the pallet centre and 0.10 m up, the truck at rest
    w, c = _yard_world(0.0, (0.0, 0.0), 0.10)
    w.state.body_q.numpy()[1, :2] = (c[0] - 1.5 - 1.601, c[1])
    r = evaluate(w, st)
    assert r["success"], [x for x in r["criteria"] if not x["passed"]]
    # held over the zone on raised forks: fails
    w, c = _yard_world(0.30, (0.0, 0.0), 0.32)
    w.state.body_q.numpy()[1, :2] = (c[0] + 0.3 - 1.601, c[1])
    failed = {x["name"] for x in evaluate(w, st)["criteria"] if not x["passed"]}
    assert {"resting_on_floor", "backed_out", "forks_lowered"} <= failed and "in_drop_zone" not in failed
    # set down but the truck still rolling away
    w, c = _yard_world(0.0, (0.0, 0.0), 0.10, rest=0.2)
    w.state.body_q.numpy()[1, :2] = (c[0] - 1.5 - 1.601, c[1])
    assert {x["name"] for x in evaluate(w, st)["criteria"] if not x["passed"]} == {"stationary"}


# ------------------------------------------------------------------------------------------------ yard speed limit
def test_yard_tasks_have_a_speed_limit():
    from forklift.evaluator import YARD_LIMIT_KMH, shared
    ev = _events()
    for v_kmh, ok in ((YARD_LIMIT_KMH - 0.1, True), (YARD_LIMIT_KMH + 0.1, False), (15.0, False)):
        ev.max_speed = v_kmh / 3.6
        for task in ("yard-route", "ramp-up-loaded", "ramp-down-loaded"):
            by = {c["name"]: c for c in shared(_shared_world(ev, "yard"), {"task": task})[0]}
            assert by["speed_limit"]["passed"] is ok and "yard" in by["speed_limit"]["threshold"], (v_kmh, task)


# ------------------------------------------------------------------------------------------------ start-and-park
def _park_world(chassis=(2.6, 28.25, math.pi), key="off", tilt=math.radians(0.5), tip_z=0.002, running=False):
    from forklift.events import Events
    from forklift.facility import warehouse
    from forklift.mathx import quat_to_R
    L = warehouse(np.random.default_rng(1000))
    x, y, yaw = chassis
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    # sideshift frame: yawed with the truck, pitched by tilt (tips down); placed so the tips are tip_z above the floor
    from scipy.spatial.transform import Rotation
    Rss = Rotation.from_euler("ZY", [yaw, tilt]).as_matrix()      # intrinsic: yaw, then pitch about the blade's y
    tip_local = np.array([1.601, 0.0, 0.104])
    o = np.array([x + 0.55 * math.cos(yaw), y + 0.55 * math.sin(yaw), 0.0])
    o[2] = tip_z - (Rss @ tip_local)[2]
    body_q = np.zeros((1, 7))
    body_q[0, :3] = o
    body_q[0, 3:] = Rotation.from_matrix(Rss).as_quat()
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    ev.rest_s = 3.0
    R = quat_to_R(np.array([qz]))[0]
    return SimpleNamespace(
        events=ev, sim_time=60.0, fault=None, signals=dict(key=key, park=1.0, direction=0),
        drivetrain=SimpleNamespace(running=running), body_pose=lambda name: (np.array([x, y, 0.3]), R),
        T=SimpleNamespace(bodies={"sideshift": 0}, anchors={"sideshift": (0.0, 0.0, 0.0)}, shifts={}),
        scene=SimpleNamespace(c=SimpleNamespace(preset="warehouse"), zones=L.zones, layout=L, loads={},
                              ground_height=lambda x, y: 0.0),
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q)))


def test_start_and_park_is_parking_in_the_bay_with_the_key_off():
    from forklift.evaluator import evaluate
    st = {"task": "start-and-park"}
    r = evaluate(_park_world(), st)
    assert r["success"], [c for c in r["criteria"] if not c["passed"]]
    names = {c["name"] for c in r["criteria"]}
    assert {"in_parking_bay", "key_off", "engine_off", "forks_tilted_to_floor", "parking_brake_set",
            "direction_neutral", "forks_lowered"} <= names and "at_stop_line" not in names
    cases = {"key_off": dict(key="on"), "engine_off": dict(running=True),
             "in_parking_bay": dict(chassis=(19.4, 28.25, math.pi)),                  # at the aisle stop line, not in the bay
             "forks_tilted_to_floor": dict(tilt=0.0, tip_z=0.009)}                    # lowered but level
    for crit, kw in cases.items():
        failed = {c["name"] for c in evaluate(_park_world(**kw), st)["criteria"] if not c["passed"]}
        assert failed == {crit}, (crit, failed)
    # backed in (counterweight to the wall) is parked too
    assert evaluate(_park_world(chassis=(3.5, 28.25, 0.0)), st)["success"]
    # half out of the bay's open east end
    assert not evaluate(_park_world(chassis=(4.6, 28.25, math.pi)), st)["success"]


def test_start_and_park_starts_away_from_the_bay_with_the_key_off():
    from forklift.task import truck_box
    from forklift.variants import resolved_start
    starts = set()
    for seed in range(200):
        cfg, s = resolved_start("start-and-park", seed)
        bay = _layout(cfg).zones["parking"]
        (x0, x1), (y0, y1) = truck_box(s["truck"])
        assert x0 > bay["x"][1] + 4.0, seed                   # the truck starts at least 4 m clear of the bay
        assert (s["engine"], s["key"], s["park"], s["lift"]) == ("off", "off", True, 0.009)
        starts.add(s["start_from"])
    assert starts == {"main aisle east", "main aisle west", "lane 1"}


# ------------------------------------------------------------------------------------------------ seed variation
TARGETS = {"drive-to-pallet": 4, "carry-to-floor-bay": 10, "carry-loose-stack": 10, "put-away-rack": 50,
           "load-trailer": 8, "load-trailer-chocked": 8, "unload-trailer": 8, "load-bales": 2, "start-and-park": 3}


def test_every_varied_target_varies_and_the_instruction_names_it():
    from forklift.task import instruction
    from forklift.variants import resolved_start, target_of
    for task, n_min in TARGETS.items():
        seen = set()
        for seed in range(300):
            cfg, s = resolved_start(task, seed)
            text = instruction(s)
            seen.add(target_of(s))
            if task == "drive-to-pallet":
                assert f"staging lane {s['lane']} " in text
            elif task in ("carry-to-floor-bay", "carry-loose-stack"):
                assert f"floor storage bay {s['bay']} " in text or f"floor storage bay {s['bay']}," in text
            elif task == "put-away-rack":
                row, face, bay, level, slot = s["rack_location"].split("-")
                assert f"location {row}-{bay}-{int(level) + 1} (aisle {s['aisle']}, bay {int(bay)}," in text
                assert ("first" if level == "1" else "second") + " beam level" in text
                hand = ("right", "left")[int(slot)] if face == "w" else ("left", "right")[int(slot)]
                assert f"the {hand}-hand pallet position" in text
            elif task in ("load-trailer", "load-trailer-chocked"):
                tg = s["target"]
                assert f"door {tg['door'][4:]}: {'left' if tg['side'] == 'L' else 'right'} side" in text
                assert tg["behind"] == f"trailer_1{tg['side']}"
            elif task == "unload-trailer":
                tg = s["target"]
                assert f"{'left' if tg['pallet'][-1] == 'L' else 'right'}-hand pallet" in text
                assert f"door {tg['door'][4:]} " in text and text.endswith(f"staging lane {tg['lane'][4:]}.")
                assert tg["lane"][4:] == tg["door"][4:]                  # the lane in front of that door
            elif task == "load-bales":
                assert f"trailer at door {s['target']['door'][4:]}:" in text
        assert len(seen) >= n_min, (task, len(seen))


def test_put_away_targets_are_empty_and_on_the_truck_s_aisle():
    from forklift.facility import BEAM_LEVELS
    from forklift.variants import resolved_start
    faces, levels = set(), set()
    for seed in range(60):
        cfg, s = resolved_start("put-away-rack", seed)
        L = _layout(cfg)
        loc = s["rack_location"]
        assert loc in L.zones["rack_locations"] and L.reserved == {loc}, seed
        assert not any(b["name"].startswith(f"stored_{loc}_") for b in L.boxes), seed
        xa, xb = L.zones["aisles"][s["aisle"]]
        z = L.zones["rack_locations"][loc]
        assert min(abs(z["aisle_x"] - xa), abs(z["aisle_x"] - xb)) < 0.01, seed       # a face of the truck's aisle
        assert xa < s["truck"][0] < xb, seed
        faces.add(loc[:3])
        levels.add(BEAM_LEVELS.index(z["z"]))
    assert faces == {"A-e", "B-w", "B-e", "C-w"} and levels == {0, 1}


def test_dock_targets_have_their_trailer_and_pallets():
    from forklift import params as P
    from forklift.variants import resolved_start
    for task in ("load-trailer", "load-trailer-chocked", "unload-trailer", "load-bales"):
        for seed in range(40):
            cfg, s = resolved_start(task, seed)
            L = _layout(cfg)
            door = s["target"]["door"]
            d = L.zones["doors"][door]
            assert door in s["trailers"], (task, seed)
            if task == "load-bales":
                nose = [ld for ld in s["loads"] if ld["name"].startswith("nose_dunnage")]
                assert len(nose) == 4 and all(abs(ld["centre"][0] - d["x"]) < 1.0 for ld in nose), seed
                continue
            assert set(s["trailers"]) == {door}
            pallets = {ld["name"]: ld for ld in s["loads"]}
            for side, sx in (("L", -1), ("R", 1)):          # left (facing the nose) is the trailer's +y: world -x here
                x = pallets[f"trailer_1{side}"]["pose"][0]
                assert sx * (x - d["x"]) > 0 and abs(x - d["x"]) < P.TRAILER_INSIDE["W"] / 2, (task, seed)
            assert abs(s["truck"][0] - d["x"]) < 0.5, (task, seed)   # starts in front of that door


def test_wet_floor_sign_never_stands_on_the_truck_or_across_the_aisle():
    from forklift.task import truck_box, wet_sign_box
    from forklift.variants import resolved_start

    def overlap(a, b):
        return a[0][0] < b[0][1] and b[0][0] < a[0][1] and a[1][0] < b[1][1] and b[1][0] < a[1][1]
    for _, task in _tasks():
        if task not in ("start-and-park", "aisle-cruise", "drive-to-pallet", "carry-to-floor-bay", "put-away-rack"):
            continue
        n = 0
        for seed in range(300):
            cfg, s = resolved_start(task, seed)
            wp = s["floor"].get("wet_patch")
            if wp is None:
                continue
            n += 1
            sign = wet_sign_box(wp["centre"], wp["half"])
            assert not overlap(sign, truck_box(s["truck"], 0.3)), (task, seed)
            assert sign[1][0] > 28.25 + 0.66 + 0.25 and sign[1][1] < 30.5, (task, seed)    # at the aisle's north edge
            if task == "start-and-park":
                assert sign[0][0] > 6.5 and wp["centre"][0] < s["truck"][0], seed        # on the way, not in the bay
        assert 60 <= n <= 140, (task, n)                       # about a third of the seeds


# ------------------------------------------------------------------------------------------------ restraint signal
def _dock_world(door, ys, park=1.0, direction=0, rest=3.0):
    """load-trailer at an unsecured door: the truck's chassis travelled along x = door x through the y values ys
    (heading north), the pallet on its forks at the end."""
    from forklift.events import Events
    from forklift.facility import dock
    L = dock(np.random.default_rng(1000))
    d = L.zones["doors"][door]
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    ev.rest_s = rest
    ev.traj = [(0.01 * k, d["x"], y, math.pi / 2, 0.5) for k, y in enumerate(ys)]
    y_end = ys[-1]
    body_q = np.zeros((2, 7))
    body_q[0] = (d["x"], y_end + 1.17, 0.2, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))   # pallet on the forks
    body_q[1] = (d["x"], y_end + 0.0, 0.2, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))   # sideshift
    return SimpleNamespace(
        events=ev, sim_time=40.0, fault=None, signals=dict(park=park, direction=direction),
        T=SimpleNamespace(bodies={"sideshift": 1}, anchors={"sideshift": (0.0, 0.0, 0.0)}, shifts={}),
        scene=SimpleNamespace(c=SimpleNamespace(preset="dock"), zones=L.zones, layout=L, loads={"carried": {"body": 0}},
                              ground_height=lambda x, y: 0.0),
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q)))


def test_an_unsecured_trailer_must_not_be_boarded():
    from forklift.evaluator import evaluate
    from forklift.task import instruction
    from forklift.variants import resolved_start
    unsecured = []
    for seed in range(500):
        cfg, s = resolved_start("load-trailer", seed)
        tr = s["trailers"][s["target"]["door"]]
        assert "restraint signal" in instruction(s) and "enter only on green" in instruction(s)
        if not tr["restraint"]:
            assert not tr["chocks"]
            unsecured.append((seed, s))
    assert 70 <= len(unsecured) <= 130                      # one seed in five
    for task in ("load-trailer-chocked", "unload-trailer"):
        assert all(resolved_start(task, k)[1]["trailers"][resolved_start(task, k)[1]["target"]["door"]]["chocks"]
                   for k in range(50))
    seed, st = unsecured[0]
    door = st["target"]["door"]
    good = evaluate(_dock_world(door, list(np.linspace(9.0, 15.2, 60))), st)
    assert good["success"], [c for c in good["criteria"] if not c["passed"]]
    for kw, crit in ((dict(ys=list(np.linspace(9.0, 17.0, 60))), "stayed_off_the_leveler"),
                     (dict(ys=list(np.linspace(9.0, 15.2, 60)), park=0.0), "stopped_short")):
        r = evaluate(_dock_world(door, **kw), st)
        assert {c["name"] for c in r["criteria"] if not c["passed"]} == {crit}, r["criteria"]

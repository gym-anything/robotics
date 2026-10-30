"""Forklift environment: fast checks of the catalogue, configuration, start states, evaluator and backend contract.

No GPU and no renderer. Most checks need no physics (seconds). Two worlds are built once per module on CPU MuJoCo:
the served backend with the seated H1-2 at warehouse/start-and-park seed 0 (about 45 s to build and settle on a
shared host) and a robot-less warehouse/drive-to-pallet seed 0 (about 20-40 s). The slow checks (assisted task demos,
qualification runs, renders, live GAR sessions) are not part of this file.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gym_anything_robotics.errors import UnsupportedConfigurationError
from tests.packages import PACKAGES, REPO

ALIASES = ("steer-forklift", "forklift-truck", "drive-forklift", "lift-truck")
SEEDS = (0, 1, 517, 799, 800, 899, 900, 999)


def _tasks():
    from forklift.config import TASKS
    return [(p, t) for p, ts in TASKS.items() if p != "lab" for t in ts]


def _layout(cfg):
    """The preset layout of a configuration, built exactly as facility.build and variants.resolved_start build it."""
    from forklift import facility, task as TK
    return {"warehouse": facility.warehouse, "dock": facility.dock, "yard": facility.yard}[cfg.preset](
        np.random.default_rng(1000 + cfg.seed), reserved=TK.reserved_locations(cfg))


# ------------------------------------------------------------------------------------------------ registry and catalogue
def test_registry_launch_and_aliases():
    from gym_anything_robotics import launch, registry, tasks
    assert {"environment_id": "forklift", "backend": "newton"} in registry.available_environments()
    d = launch.describe("forklift")
    assert d["environment_id"] == "forklift" and d["default_robot"] == "h1-2"
    assert d["implementation_package"] == "forklift" and d["engine_version"] == "1.5"
    # the offer: the seated H1-2 (default) and the driving-robot kit (forklift/operator.py add_kit)
    assert [(r["id"], r["setup"]) for r in d["robots"]] == [("h1-2", "forklift-h1-2"), ("drive-kit", "forklift-drive-kit")]
    assert [r["id"] for r in d["robots"] if r["default"]] == ["h1-2"]
    for alias in ALIASES:
        assert launch.describe(alias)["environment_id"] == "forklift"
        assert tasks.catalog(alias)["environment_id"] == "forklift"
    assert tasks.CATALOGS["forklift"] == "forklift/benchmark.json"


def test_catalogue_matches_the_task_table():
    from gym_anything_robotics import tasks
    from forklift import evaluator as E
    from forklift.task import DIFFICULTY, INSTRUCTIONS, TIME_LIMIT_S
    root = PACKAGES
    raw = json.loads((root/"forklift"/"benchmark.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    assert tasks.catalog("forklift") == raw
    assert raw["schema"] == "gar.task-catalog.v1" and raw["environment_id"] == "forklift"
    assert raw["selectors"] == ["task", "seed", "contract"]   # explicit reward version; same task/seed defaults
    assert raw["resolver"] == "forklift.variants:task_resolve"
    assert raw["splits"] == ["train", "validation", "test"]
    assert (root/raw["evaluator"]["source"]).is_file() and raw["evaluator"]["available"]
    names = [t for _, t in _tasks()]
    assert raw["count"] == len(names) == 14
    assert all(t in raw["count_scope"] for t in names)
    for table in (INSTRUCTIONS, TIME_LIMIT_S, DIFFICULTY):
        assert set(table) == set(names)
    for t in names:
        assert INSTRUCTIONS[t].strip() and 0 < TIME_LIMIT_S[t] <= 600
        assert callable(getattr(E, "_" + t.replace("-", "_"))), t       # every task has its evaluator
    assert tasks.resolve("forklift").task_id == raw["default_task_id"] == "forklift/v1/warehouse/start-and-park/seed-0"


def test_tasks_resolve_with_instructions_splits_and_round_trip():
    from gym_anything_robotics import tasks
    from forklift.task import SPLITS
    from forklift.variants import split_of_seed
    assert SPLITS == dict(train=(0, 799), validation=(800, 899), test=(900, 999))
    want = {0: "train", 799: "train", 800: "validation", 899: "validation", 900: "test", 999: "test"}
    assert {s: split_of_seed(s) for s in want} == want
    ids = set()
    for preset, task in _tasks():
        for seed in (0, 799, 800, 899, 900, 999):
            spec = tasks.resolve("forklift", {"task": task, "seed": seed})
            assert spec.task_id == f"forklift/v1/{preset}/{task}/seed-{seed}"
            assert spec.launch.backend_options == {"task": task, "seed": seed}
            assert spec.split == want[seed]
            assert spec.description.strip() and "{" not in spec.description
            again = tasks.TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
            assert again.to_dict() == spec.to_dict()
            ids.add(spec.task_id)
    assert len(ids) == 14*6
    for bad in ({"task": "free"}, {"task": "fly-the-truck"}, {"task": "start-and-park", "seed": -1},
                {"task": "start-and-park", "robot": "g1"}, {"task": "start-and-park", "preset": "dock"}):
        with pytest.raises(ValueError):
            tasks.resolve("forklift", bad)


def test_split_note_matches_the_resolver():
    from gym_anything_robotics import tasks
    note = tasks.catalog("forklift")["split_note"]
    try:
        resolved = tasks.resolve("forklift", {"task": "start-and-park", "seed": 1000}).split is None
    except ValueError:
        resolved = False
    assert ("resolve with split null" in note) == resolved


# ------------------------------------------------------------------------------------------------ robots


# ------------------------------------------------------------------------------------------------ configuration
def test_config_rejects_unknown_keys_presets_tasks_and_seeds():
    from forklift.config import Config
    c = Config()
    assert (c.preset, c.task, c.seed, c.robot, c.dt) == ("warehouse", "start-and-park", 0, "h1-2", 0.0005)
    assert Config.from_dict(Config(task="aisle-cruise", seed=7).to_dict()) == Config(task="aisle-cruise", seed=7)
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        Config.from_dict({"task": "start-and-park", "cheat": 1})
    with pytest.raises(ValueError, match="preset"):
        Config(preset="office")
    with pytest.raises(ValueError, match="not a warehouse task"):
        Config(preset="warehouse", task="load-trailer")
    with pytest.raises(ValueError, match="not a yard task"):
        Config(preset="yard", task="start-and-park")
    for seed in (-1, 1000, True, 3.0, "3", None):
        with pytest.raises(ValueError, match="seed"):
            Config(seed=seed)
    assert Config(seed=999).seed == 999
    for kw in (dict(robot="g1"), dict(camera_set="all"), dict(dt=0.001), dict(dt=0.0)):
        with pytest.raises(ValueError):
            Config(**kw)
    from forklift.variants import preset_of
    with pytest.raises(ValueError):
        preset_of("free")                                   # the lab preset is not in the catalogue


# ------------------------------------------------------------------------------------------------ start states (no physics)
CARRIED = {"carry-to-floor-bay": ("wrapped", (600, 1200), (0.9, 1.5)), "put-away-rack": ("wrapped", (600, 1000), (0.9, 1.2)),
           "load-trailer": ("wrapped", (600, 1100), (1.0, 1.4)), "load-trailer-chocked": ("wrapped", (600, 1100), (1.0, 1.4)),
           "ramp-up-loaded": ("wrapped", (900, 1300), (1.0, 1.3)), "ramp-down-loaded": ("wrapped", (900, 1300), (1.0, 1.3)),
           "feed-conveyor": ("coil", (2000, 2000), None), "carry-loose-stack": ("stack", None, None)}


def test_every_task_start_resolves_without_physics():
    from forklift import params as P
    from forklift.task import instruction
    from forklift.variants import resolved_start
    IN = 0.0254
    for preset, task in _tasks():
        for seed in SEEDS:
            cfg, s = resolved_start(task, seed)
            where = f"{task} seed {seed}"
            assert cfg.preset == preset and s["task"] == task and s["seed"] == seed, where
            assert len(s["truck"]) == 3 and all(math.isfinite(v) for v in s["truck"]), where
            assert s["direction"] == "N" and s["park"] is True, where
            # only start-and-park begins with the engine off and the key out; every other task with the engine idling
            assert (s["engine"], s["key"]) == (("off", "off") if task == "start-and-park" else ("idle", "on")), where
            assert 0.009 <= s["lift"] <= 0.15 and s["tilt_deg"] in (0.0, -3.0), where
            lf = s["load_on_forks"]
            if task in CARRIED:
                kind, mass, height = CARRIED[task]
                assert lf["kind"] == kind, where
                if mass:
                    assert mass[0] <= lf["mass"] <= mass[1], where
                if height:
                    assert height[0] <= lf["height"] <= height[1], where
            else:
                assert lf is None, where
            text = instruction(s)
            assert text.strip() and "{" not in text, where
            json.dumps(s, default=float, allow_nan=False)
            if preset == "dock":
                assert 44.0 <= (s["bed_z"] + P.DOCK_HEIGHT)/IN <= 52.0, where
                for door, tr in s["trailers"].items():
                    if task == "load-trailer" and not tr["chocks"]:
                        assert not tr["restraint"], where      # the unsecured variant: no restraint, no chocks
                    else:
                        assert tr["restraint"] == (task != "load-trailer-chocked") and tr["chocks"], where
                assert s["target"]["door"] in s["trailers"], where
            if task == "drive-to-pallet":
                st = _layout(cfg).zones["staging"][s["lane"]]
                (target,) = s["loads"]
                x, y, _ = target["pose"]
                assert st["x"][0] < x < st["x"][1] and st["y"][0] < y < st["y"][1], where
            if task == "put-away-rack":
                L = _layout(cfg)
                loc = s["rack_location"]
                assert loc in L.zones["rack_locations"] and loc in L.reserved, where
                assert not any(b["name"].startswith(f"stored_{loc}_") for b in L.boxes), where   # the location is empty
            if task == "unload-trailer":
                assert s["target"]["pallet"] in {ld["name"] for ld in s["loads"]}, where
            if task == "load-bales":
                bales = [ld for ld in s["loads"] if ld["name"].startswith("bale_")]
                assert len(bales) == 3 and all(P.BALE["mass"][0] <= b["mass"] <= P.BALE["mass"][1] for b in bales), where
                assert {"door3", "door4"} == set(s["trailers"]), where


def test_start_states_are_a_pure_function_of_task_and_seed():
    from forklift.variants import digest
    for _, task in _tasks():
        a, b, c = digest(task, 5)[0], digest(task, 5)[0], digest(task, 6)[0]
        assert a == b and a != c, task


def test_put_away_instruction_names_the_evaluated_beam_level():
    from forklift.facility import BEAM_LEVELS
    from forklift.task import instruction
    from forklift.variants import resolved_start
    cfg, s = resolved_start("put-away-rack", 0)
    z = _layout(cfg).zones["rack_locations"][s["rack_location"]]["z"]
    ordinal = ("first", "second")[BEAM_LEVELS.index(z)]
    assert f"{ordinal} beam level" in instruction(s)


def test_pyproject_ships_the_package():
    """A wheel install needs the forklift package and its benchmark.json, which gar's task catalogue loads."""
    import fnmatch
    import tomllib
    setuptools = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]
    assert any(fnmatch.fnmatch("forklift", p) for p in setuptools["packages"]["find"]["include"])
    assert "*.json" in setuptools["package-data"]["*"]


# ------------------------------------------------------------------------------------------------ evaluator logic (no physics)
def _events():
    """The environment's own per-tick latches, with a contact map instead of a built model."""
    from forklift.events import Events
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    ev._maps_ready = True
    ev.mover, ev.geom_class, ev.floor_geoms = {1: "load:carried", 2: "load:target"}, {}, {0}
    ev._load_names = ["carried", "target"]
    return ev


def _sweep(ev, t, touching):
    """One contact sweep at time t; touching: {load geom: (x, y)} in contact with the floor geom 0."""
    contacts = [SimpleNamespace(geom1=g, geom2=0, pos=np.array([x, y, 0.0])) for g, (x, y) in touching.items()]
    world = SimpleNamespace(solver=SimpleNamespace(mj_model=None, mj_data=SimpleNamespace(ncon=len(contacts), contact=contacts)),
                            config=SimpleNamespace(dt=0.0005), sim_time=t)
    ev._contacts(world)


def test_floor_contacts_count_only_after_a_load_was_off_the_floor():
    from forklift.events import LIFTED_S, STRIKE_EVERY
    ev = _events()
    dt = 0.0005*STRIKE_EVERY
    t = 0.0
    # a pallet resting where it started (a load waiting in a lane) touches the floor every sweep: never a drop; the
    # carried pallet stays off the floor (on the forks) for longer than LIFTED_S
    for _ in range(int(1.0/dt)):
        t += dt
        _sweep(ev, t, {2: (10.0, 10.0)})
    assert ev.load_floor == {} and ev._lifted == {"carried"} and ev._floor_off["target"] == 0.0
    # nudged off the floor for less than LIFTED_S and set back: still not lifted, still no record
    for _ in range(int(0.5*LIFTED_S/dt)):
        t += dt
        _sweep(ev, t, {})
    for _ in range(10):
        t += dt
        _sweep(ev, t, {2: (10.0, 10.0)})
    assert ev.load_floor == {} and "target" not in ev._lifted
    # the carried pallet comes down: its floor contact is recorded with where it happened
    t += dt
    _sweep(ev, t, {1: (5.0, 6.0), 2: (10.0, 10.0)})
    assert list(ev.load_floor) == ["carried"] and ev.load_floor["carried"][0][1] == (5.0, 6.0)
    for _ in range(int(0.2/dt)):                            # resting there: one record per 0.25 s, not one per sweep
        t += dt
        _sweep(ev, t, {1: (5.0, 6.0), 2: (10.0, 10.0)})
    assert len(ev.load_floor["carried"]) == 1


def test_shared_criteria_count_real_strikes_and_drops_only():
    from forklift.evaluator import shared
    ev = _events()

    def episode(peak, impulse, speed, mover, parts=None):
        return dict(peak_N=peak, impulse_Ns=impulse, episodes=1, first_t=1.0, max_episode_Ns=impulse,
                    impacts=[dict(t=1.0, onset_speed=speed, impulse_Ns=impulse, peak_N=peak, mover=mover, parts=parts)])
    beam = ["carried:bottom_deck", "C0_beam2_1_12.98"]
    ev.strikes = {("load", "rack"): episode(3000.0, 1.0, 0.05, "load:carried", beam),  # set down slowly on a beam
                  ("truck", "column"): episode(2500.0, 5.0, 1.0, "truck"),           # driven into a column
                  ("robot", "wall"): episode(100.0, 1.0, 0.5, "robot"),              # brushed: below both limits
                  ("truck", "floor_marking"): episode(9000.0, 99.0, 1.0, "truck")}   # not a watched structure
    ev.load_floor = {"carried": [(5.0, (25.0, 8.7))], "target": [(6.0, (10.0, 10.0))]}
    world = SimpleNamespace(events=ev, scene=SimpleNamespace(c=SimpleNamespace(preset="warehouse")), sim_time=30.0)
    crit, det = shared(world, {"task": "carry-to-floor-bay"}, dest_floor=lambda n, xy: n == "carried" and xy[0] < 30.0)
    by = {c["name"]: c for c in crit}
    assert [s["structure"] for s in det["strikes"]] == ["column"] and not by["no_strike"]["passed"]
    assert det["dropped"] == [("target", 6.0, (10.0, 10.0))] and by["no_dropped_load"]["measured"] == 1
    assert by["time_limit"]["passed"] and by["speed_limit"]["passed"]
    # a load coming onto rack beams fast is a strike in every task, put-away included (forklift/evaluator.py,
    # SET_DOWN_SPEED; more cases in test_forklift_tasks.py)
    ev.strikes = {("load", "rack"): episode(3000.0, 1.0, 0.5, "load:carried", beam)}
    assert not {c["name"]: c for c in shared(world, {"task": "carry-to-floor-bay"})[0]}["no_strike"]["passed"]
    assert not {c["name"]: c for c in shared(world, {"task": "put-away-rack"})[0]}["no_strike"]["passed"]
    ev.max_roll, world.sim_time = 7.0, 181.0
    by = {c["name"]: c for c in shared(world, {"task": "carry-to-floor-bay"})[0]}
    assert not by["roll_under_6deg"]["passed"] and not by["time_limit"]["passed"]
    for c in crit:
        assert set(c) == {"name", "measured", "passed", "unit", "threshold", "source"} and isinstance(c["passed"], bool)


def test_every_watched_structure_class_is_judged_or_exempt():
    """The evaluator's strike classes and the classes the scene registers must name the same things: a strike class
    that no structure registers counts nothing, and a registered class nobody judges is a silent hole."""
    import re
    from forklift.evaluator import STRIKE_CLASSES
    from pathlib import Path
    import forklift
    root = Path(forklift.__file__).resolve().parent
    src = "".join((root / f).read_text() for f in ("facility.py", "dock.py", "machines.py", "scene.py"))
    registered = set(re.findall(r'watch="([a-z_]+)"', src)) | set(re.findall(r'statics\.setdefault\("([a-z_]+)"', src))
    registered |= set(re.findall(r'"(trailer_[a-z]+)" if', src)) | set(re.findall(r'else "(trailer_[a-z]+)"', src))
    exempt = {"trailer_floor": "driven on; loads rest on it"}
    assert STRIKE_CLASSES <= registered, f"strike classes nothing registers: {STRIKE_CLASSES - registered}"
    assert registered <= STRIKE_CLASSES | set(exempt), f"registered but never judged: {registered - STRIKE_CLASSES - set(exempt)}"


def _rack_world(other_level=False, seed=11):
    """A stub world for _put_away_rack: the real layout and target of (put-away-rack, seed), the carried pallet resting
    on the front and rear beams of the target bay and slot at the target's beam level (or, other_level, at the other
    beam level of that bay and slot, which the facility may have stocked), the forks withdrawn and lowered."""
    from forklift import params as P
    from forklift.events import Events
    from forklift.variants import resolved_start
    cfg, start = resolved_start("put-away-rack", seed)
    L = _layout(cfg)
    row, face, bay, level, slot = start["rack_location"].split("-")
    level = (3 - int(level)) if other_level else int(level)
    loc = L.zones["rack_locations"][f"{row}-{face}-{bay}-{level}-{slot}"]
    if not other_level:
        assert not any(b["name"].startswith(f"stored_{row}-{face}-{bay}-{level}-{slot}_") for b in L.boxes)
    frame = "1" if (row == "B" and face == "e") else "0"    # row B is back-to-back: frame 1 faces east (aisle 2)
    beams = sorted(b["name"] for b in L.boxes if b["name"].startswith(f"{row}{frame}_beam{int(bay) - 1}_{level}_"))
    assert len(beams) == 2                                   # the front and the rear beam of that bay and level
    labels = ["carried:stringer", *beams]
    body_q = np.zeros((2, 7))
    sgn = 1.0 if face == "w" else -1.0                       # a west face: the rack lies east (+x) of its aisle
    body_q[0] = (loc["aisle_x"] + sgn * (P.GMA_PALLET["L"]/2 - 0.076), loc["y"], loc["z"], 0.0, 0.0, 0.0, 1.0)   # 76 mm overhang
    body_q[1] = (14.0, 24.6, 0.10, 0.0, 0.0, 0.0, 1.0)                                                   # truck far away
    ev = Events.__new__(Events)
    ev.w = None
    ev.reset()
    world = SimpleNamespace(
        events=ev, sim_time=60.0, fault=None, T=SimpleNamespace(bodies={"sideshift": 1}, anchors={"sideshift": (0.0, 0.0, 0.0)},
                                                                shifts={}),
        scene=SimpleNamespace(c=SimpleNamespace(preset="warehouse"), zones=L.zones, layout=L, loads={"carried": {"body": 0}},
                              ground_height=lambda x, y: 0.0),
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q)),
        solver=SimpleNamespace(mj_model=SimpleNamespace(ngeom=3), mj_data=SimpleNamespace(
            ncon=2, contact=[SimpleNamespace(geom1=0, geom2=1), SimpleNamespace(geom1=0, geom2=2)])),
        mj_geom_label=lambda g: labels[g])
    return world, start


def test_put_away_on_the_target_beams_passes():
    from forklift.evaluator import evaluate
    for seed in (11, 0, 3):                                  # different rows, faces and beam levels
        world, start = _rack_world(seed=seed)
        r = evaluate(world, start)
        assert r["schema"] == "gar.evaluation.v1" and r["success"], (seed, [c for c in r["criteria"] if not c["passed"]])


def test_put_away_at_the_other_beam_level_fails():
    from forklift.evaluator import evaluate
    for seed in (11, 0, 3):
        world, start = _rack_world(other_level=True, seed=seed)
        assert not evaluate(world, start)["success"], seed


# ------------------------------------------------------------------------------------------------ backend configuration
def test_backend_refuses_configuration_before_building_a_world(tmp_path, monkeypatch):
    import forklift.env
    from robots.catalog import resolve
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend

    class NoWorld:
        def __init__(self, *a, **k):
            raise AssertionError("a refused configuration must not build a world")
    monkeypatch.setattr(forklift.env, "ForkliftEnv", NoWorld)
    cases = (dict(task="free"), dict(task="fly-the-truck"), dict(task="start-and-park", preset="dock"),
             dict(robot="g1"), dict(robot="single-fr3"), dict(robot="dual-fr3"), dict(robot="dual-panda"),
             dict(robot="none"), dict(robot=resolve("football-g1")), dict(robot=resolve("scan-fr3")),
             dict(config={"extras": {"seat_ballast_kg": 67.0}}), dict(config={"extras": {}}),
             dict(config={"cheat": 1}), dict(config={"dt": 0.001}), dict(config={"camera_set": "all"}),
             dict(seed=True), dict(seed=1000), dict(seed=-1), dict(seed=3.0), dict(seed="3"))
    for k, kw in enumerate(cases):
        out = tmp_path/f"s{k}"
        with pytest.raises(UnsupportedConfigurationError):
            NewtonForkliftBackend(output_dir=out, **kw)
        assert not out.exists(), kw
    with pytest.raises(UnsupportedConfigurationError, match="assisted"):
        NewtonForkliftBackend(output_dir=tmp_path/"x", config={"extras": {"truck_pose": (0.0, 0.0, 0.0)}})


def test_backend_refuses_diagnostic_cameras_before_building_a_world(tmp_path, monkeypatch):
    import forklift.env
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend

    class NoWorld:
        def __init__(self, *a, **k):
            raise AssertionError("a refused configuration must not build a world")
    monkeypatch.setattr(forklift.env, "ForkliftEnv", NoWorld)
    with pytest.raises(UnsupportedConfigurationError, match="owner diagnostics"):
        NewtonForkliftBackend(output_dir=tmp_path/"s", cameras=("overview",))


# ------------------------------------------------------------------------------------------------ served backend (physics)
@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    b = NewtonForkliftBackend(task="start-and-park", seed=0, cameras=(), output_dir=tmp_path_factory.mktemp("forklift")/"s")
    yield b
    b.close()


def _snapshot(b):
    w = b.env.world
    mo = w.operator.motors
    return w.tick, b.env.generation, {k: v.copy() for k, v in mo.cmd.items()}, w.state.body_q.numpy().copy()



"""tighten-tasks-v2 (tighten_codex/task_set_v2.py): the fix branch's eight-task tests, ported.

Catalogue and state-based tests only. No Newton, Warp, server or renderer.

Run with pytest on this file. Ruckig's import is stubbed; constructing it fails.
All environment objects are __new__ shells, never simulator instances.
"""

from dataclasses import replace
import json
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tighten_codex.config import Config
from tighten_codex.presets import (SCREWS, apply_start_coordinates,
                                   flag_home, loose_screw, pretightened_screws, settle_initial)
from tighten_codex import task_set_v2 as V2
from tighten_codex.task import (EIGHT_CONTRACT, EIGHT_INSTRUCTIONS, EIGHT_ORDER, LEGACY_CONTRACT, LEGACY_TASKS,
                                MIXED_CONTRACT, NEW_TASKS, TASKS, contract_for, instruction, resolve, task_setup,
                                uses_eight)
from tighten_codex.task_evaluation import required, track_free_rail
from tests.packages import PACKAGES, REPO

# find-loose's choices of loose screw, in its setup.py (tasks/find-loose/).
LOOSE_CHOICES = task_setup("find-loose").LOOSE_CHOICES

ROOT = PACKAGES
CONTRACT = V2.CONTRACT


def forbidden(*args, **kwargs):
    raise AssertionError("Simulator/controller construction is forbidden in these tests")


@pytest.fixture(scope="module")
def env_class():
    stub = ModuleType("ruckig")
    for name in ("InputParameter", "OutputParameter", "Result", "Ruckig", "Synchronization"):
        setattr(stub, name, forbidden)
    # Restore just the stub entry, not all modules imported by scipy/numpy
    # while loading env (extension modules must not be unloaded/reimported).
    with pytest.MonkeyPatch.context() as imports:
        imports.setitem(sys.modules, "ruckig", stub)
        from tighten_codex.env import TightenEnv
    with patch.object(TightenEnv, "__init__", forbidden):
        yield TightenEnv


class Arr:
    def __init__(self, value):
        self.a = np.asarray(value, dtype=float)

    def numpy(self):
        return self.a.copy()

    def assign(self, value):
        self.a = np.asarray(value, dtype=float).copy()


def fake(env_class, task, seed=0, start=False, contract=CONTRACT):
    env = object.__new__(env_class)
    env.config = Config(task=task, seed=seed, task_contract=contract)
    env.items = {n: {"turn_joint": 1 + 2*i, "slide_joint": 2 + 2*i,
                    "diameter_m": .003 if n.startswith("sensor") else .004,
                    "pitch_m": .0005 if n.startswith("sensor") else .0007,
                    "limit_nm": .35 if n.startswith("sensor") else .60}
                 for i, n in enumerate(SCREWS)}
    env.items.update(rail=0, bracket=1, sensor=2, flag=3, flag_joint=9, flag_center_z_m=1.083)
    env.model = SimpleNamespace(joint_q_start=Arr([0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
    env.builder = SimpleNamespace(joint_label=["horizontal_profile_slot_travel"])
    env._bracket_origin = np.array([.013, .095, 1.075])
    body = np.array([[.14, .1, 1.0955, 0, 0, 0, 1], [*env._bracket_origin, 0, 0, 0, 1],
                     [.2655, .08, 1.088, 0, 0, 0, 1], [.2655 + flag_home(env.config), -.063, .934, 0, 0, 0, 1]])
    env.state = SimpleNamespace(joint_q=Arr(np.zeros(13)), body_q=Arr(body))
    env.state.joint_q.a[12] = flag_home(env.config)
    env._pitch_yaw = (0., 0.)
    env._rail_attitude = lambda body_q=None: env._pitch_yaw
    env._work_groups = {"sensor": (), "frame": ()}
    env._interface = {}
    env.fault = None
    env._preload = {n: env.minimum_load(n)/.8 for n in SCREWS}
    env._peak_preload = dict(env._preload)
    env._peak_torque = {n: 0. for n in SCREWS}
    env._reset_tracking()
    env._bit_contacts = {n: 0 for n in SCREWS}
    env._retained_anchor = {n: None for n in SCREWS}
    env._retained_ticks = {n: 20 for n in SCREWS}
    need = required(env.config)
    for name, item in env.screws.items():
        slide = .0006
        env.state.joint_q.a[int(env.model.joint_q_start.a[item["slide_joint"]])] = slide
        env.state.joint_q.a[int(env.model.joint_q_start.a[item["turn_joint"]])] = slide*2*math.pi/item["pitch_m"]
        env._seated_clicks[name] = env._clicks[name] = int(name in need)
    env._reset_free_readings()
    baseline = {n: (22. if n == "frame_top" else 0.) if n in need else env._preload[n] for n in SCREWS}
    env._task_set_state["start_loads"] = baseline
    env._hit_x = body[3, 0]
    env._detected = True
    env._optical = lambda: {"detected": env._detected, "range_m": .13,
                            "intersection_m": [env._hit_x, -.057, 1.083]}
    if start:
        for n in need:
            env._preload[n] = env._free_load[n] = baseline[n]
            env._seated_clicks[n] = env._clicks[n] = env._retained_ticks[n] = 0
            item = env.screws[n]
            env.state.joint_q.a[int(env.model.joint_q_start.a[item["slide_joint"]])] = 0.
            env.state.joint_q.a[int(env.model.joint_q_start.a[item["turn_joint"]])] = 0.
        if task in ("frame", "access", "support"):
            env._pitch_yaw = (.0369, 0.)
        if task == "support":
            env.state.body_q.a[0, 0] += .0015
        if task == "aim":
            env._hit_x += .007
            env._detected = False
        track_free_rail(env)
    return env


def test_v2_task_set_resolves_beside_the_released_one():
    data = json.loads((ROOT/"tighten_codex/benchmark.json").read_text())
    assert data["count"] == 8 and data["selector_choices"]["contract"] == [
        LEGACY_CONTRACT, EIGHT_CONTRACT, MIXED_CONTRACT, CONTRACT]
    selections = data["task_set"]["selections"]
    assert selections == [dict(s) for s in V2.TASK_SET]
    assert tuple(s["task"] for s in selections) == EIGHT_ORDER
    released = {e["selection"]["task"]: e for e in data["task_set_v1"]["tasks"]}
    for s in selections[:-1]:
        task = s["task"]
        assert s == {"task": task, "seed": 0, "contract": CONTRACT}
        r = resolve(s)
        # The contract is part of every v2 id, front-m4 and find-loose included.
        assert r["task_id"] == f"tighten-codex/v1/{task}/{CONTRACT}/seed-0"
        assert r["selection"] == s and r["split"] is None
        config = r["backend_options"]["config"]
        assert config == {"task": task, "seed": 0, "task_contract": CONTRACT}
        assert r["description"] == EIGHT_INSTRUCTIONS[task] == instruction(Config(**config))
        # Same instruction as the released task set.
        assert r["description"] == resolve(released[task]["selection"])["description"]
    # mixed keeps its released selection, exactly as splits/test.json has it.
    split = json.loads((REPO/"benchmarks/roboworld/splits/test.json").read_text())["tasks"]
    assert {"environment_id": "tighten-codex", "selection": selections[-1]} in split
    assert selections[-1] == {"task": "mixed"}
    assert resolve(selections[-1]) == resolve({}) and resolve({})["task_id"] == data["default_task_id"]
    # The released set still resolves to its recorded ids.
    for e in data["task_set_v1"]["tasks"]:
        assert resolve(e["selection"])["task_id"] == e["task_id"]
        assert CONTRACT not in json.dumps(resolve(e["selection"]))


def test_released_selections_never_reach_v2():
    for n in TASKS:
        for contract in ("", LEGACY_CONTRACT, EIGHT_CONTRACT, MIXED_CONTRACT):
            try:
                chosen = contract_for(n, contract)
            except ValueError:
                continue
            assert chosen != CONTRACT
            r = resolve({"task": n, "seed": 3, **({"contract": contract} if contract else {})})
            assert CONTRACT not in json.dumps(r)
            assert uses_eight(Config(task=n, task_contract=contract)) == (chosen == EIGHT_CONTRACT)
    for n in TASKS:
        if n == "mixed":
            with pytest.raises(ValueError):
                Config(task=n, task_contract=CONTRACT)
            continue
        assert contract_for(n, CONTRACT) == CONTRACT and uses_eight(Config(task=n, task_contract=CONTRACT))


@pytest.mark.parametrize("bad", [{"seed": True}, {"seed": -1}, {"seed": "0"}, {"seed": 1.5}, {"task": "missing"},
                                {"task": "mixed"}, {}, {"other": 1}, {"task": "sensor", "other": 1}])
def test_bad_v2_selectors_rejected(bad):
    with pytest.raises(ValueError):
        resolve({**bad, "contract": CONTRACT})


def test_seeded_choice_and_no_verifier_thresholds_in_instructions():
    assert {loose_screw("find-loose", i) for i in range(30)} == set(LOOSE_CHOICES)
    for seed in (0, 1, 2, 100, 10**50):
        n = loose_screw("find-loose", seed)
        assert n == loose_screw("find-loose", seed)
        assert set(pretightened_screws(Config(task="find-loose", seed=seed))) == set(SCREWS)-{n}
    for task in EIGHT_ORDER[:-1]:
        text = instruction(Config(task=task, task_contract=CONTRACT))
        assert text == instruction(Config(task=task, task_contract=EIGHT_CONTRACT))
        for threshold in ("mrad", "80%", "80 %", "10 ticks", "0.5 mm", "0.001", "0.95"):
            assert threshold not in text


@pytest.mark.parametrize("task", EIGHT_ORDER[:-1])
def test_untouched_zero_and_released_solution_one(env_class, task):
    idle = fake(env_class, task, start=True)._evaluate()
    assert not idle["success"] and idle["score"] == idle["progress"] == 0.
    done = fake(env_class, task)._evaluate()
    assert done["success"] and done["score"] == 1., done
    assert done["task_contract"] == CONTRACT and done["evaluation_scope"].startswith(f"{CONTRACT}: ")


@pytest.mark.parametrize("task", EIGHT_ORDER[:-1])
@pytest.mark.parametrize("start", (False, True))
def test_v2_scores_as_the_released_set_apart_from_the_detection_rule(env_class, task, start):
    labels = ("task_contract", "evaluation_scope")
    v2 = fake(env_class, task, start=start)._evaluate()
    v1 = fake(env_class, task, start=start, contract=EIGHT_CONTRACT)._evaluate()
    assert v1["task_contract"] == EIGHT_CONTRACT and v2["task_contract"] == CONTRACT
    assert {k: v for k, v in v2.items() if k not in labels} == {k: v for k, v in v1.items() if k not in labels}


@pytest.mark.parametrize("seed", range(9))
def test_all_diagnostic_variants_and_no_reclick_requirement_on_prior_assembly(env_class, seed):
    env = fake(env_class, "find-loose", seed)
    assert env._evaluate()["success"]
    target = loose_screw("find-loose", seed)
    for n in set(SCREWS)-{target}:
        assert env._seated_clicks[n] == 0
        previous = env._free_load[n]
        env._free_load[n] = .99 * env.minimum_load(n)
        assert not env._evaluate()["success"]
        env._free_load[n] = previous


@pytest.mark.parametrize("task", EIGHT_ORDER[:-1])
@pytest.mark.parametrize("defect", ("click", "retention", "load", "advance", "helix", "bit", "held", "torque", "fault"))
def test_each_required_condition_blocks_success(env_class, task, defect):
    env = fake(env_class, task)
    name = required(env.config)[0]
    if defect == "click": env._seated_clicks[name] = 0
    if defect == "retention": env._retained_ticks[name] = 9
    if defect == "load": env._free_load[name] = .99 * env.minimum_load(name)
    if defect == "advance": env.state.joint_q.a[int(env.model.joint_q_start.a[env.items[name]["slide_joint"]])] = .0001
    if defect == "helix": env.state.joint_q.a[int(env.model.joint_q_start.a[env.items[name]["turn_joint"]])] += 5.
    if defect == "bit": env._bit_contacts[name] = 1
    if defect == "held": env._held = {"sensor": True, "frame": True}
    if defect == "torque": env._peak_bit_moment_filtered["sensor_left"] = .40001
    if defect == "fault": env.fault = "rail left slot"
    e = env._evaluate()
    assert not e["success"] and 0 <= e["score"] <= .95
    assert e["score"] == .95 * e["progress"]


def test_progress_is_continuous_above_initial_load_and_not_click_share(env_class):
    env = fake(env_class, "access")
    floor = env.minimum_load("frame_top")
    baseline = env._task_set_state["start_loads"]["frame_top"]
    env._seated_clicks["frame_top"] = 0
    for fraction in np.linspace(0., 1., 101):
        env._free_load["frame_top"] = baseline + fraction * (floor-baseline)
        e = env._evaluate()
        assert e["progress"] == pytest.approx(fraction)
        assert e["score"] == pytest.approx(.95*fraction)
    no_click = env._evaluate()["progress"]
    env._seated_clicks["frame_top"] = 1
    env._retained_ticks["frame_top"] = 0
    assert env._evaluate()["progress"] == no_click
    env._free_load["frame_top"] = baseline
    assert env._evaluate()["score"] == 0.


def test_preservation_penalty_varies_smoothly_not_number_of_failed_checks(env_class):
    env = fake(env_class, "front-m4")
    for ratio in np.linspace(0., 1., 21):
        env._free_load["sensor_left"] = ratio * env.minimum_load("sensor_left")
        e = env._evaluate()
        assert e["score_terms"]["other_fasteners_load_grade"] == pytest.approx(.75 + .25*ratio)
        assert e["progress"] == pytest.approx(.75 + .25*ratio)


@pytest.mark.parametrize("task", ("sensor", "aim", *NEW_TASKS))
def test_flag_home_and_aim_cannot_be_faked_by_moving_flag(env_class, task):
    env = fake(env_class, task)
    initial = env.state.body_q.a[3, 0]
    env.state.joint_q.a[12] += .00049
    env.state.body_q.a[3, 0] += .00049
    assert env._evaluate()["success"]
    env.state.joint_q.a[12] = flag_home(env.config) + .00051
    env.state.body_q.a[3, 0] = initial + .00051
    assert not env._evaluate()["success"]
    env.state.joint_q.a[12] = flag_home(env.config) + .007
    env.state.body_q.a[3, 0] = env._hit_x = initial + .007
    e = env._evaluate()
    assert not e["success"] and not e["checks"]["detecting_at_home"]
    assert e["score_terms"]["aim_at_home_grade"] < .1


@pytest.mark.parametrize("task", ("sensor", "aim", *NEW_TASKS))
def test_sensor_and_aim_keep_the_whole_spot_detection_rule(env_class, task):
    # Spot centre 4 mm from the home flag's centre line: on the 10 mm face, but
    # the 4.2 mm spot is not wholly on it (whole-spot rule: |dx| <= 2.9 mm).
    env = fake(env_class, task)
    env._hit_x += .004
    e = env._evaluate()
    whole_spot = task in ("sensor", "aim")
    assert V2.detection_margin(env.config) == (.0021 if whole_spot else 0.)
    assert e["checks"]["detecting_at_home"] is (not whole_spot)
    assert e["success"] is (not whole_spot)
    # The aim grade keeps mixed's centre-on-face formula; only the success predicate changes.
    assert e["score_terms"]["aim_at_home_grade"] == 1.
    env._hit_x -= .0012       # 2.8 mm: the whole spot is on the face
    assert env._evaluate()["success"]
    # tighten-tasks-v1 keeps its released rule (the spot centre on the flag).
    released = fake(env_class, task, contract=EIGHT_CONTRACT)
    released._hit_x += .004
    assert released._evaluate()["success"]


@pytest.mark.parametrize("task, contract, detected", [
    ("sensor", CONTRACT, False), ("aim", CONTRACT, False), ("frame", CONTRACT, False),
    ("front-m4", CONTRACT, True), ("find-loose", CONTRACT, True),
    ("sensor", EIGHT_CONTRACT, True), ("aim", EIGHT_CONTRACT, True), ("frame", EIGHT_CONTRACT, True),
    ("front-m4", "", True), ("sensor", "", False), ("sensor", LEGACY_CONTRACT, False), ("mixed", "", True)])
def test_optical_detection_follows_the_task_under_v2_only(env_class, task, contract, detected):
    env = object.__new__(env_class)
    env.config = Config(task=task, task_contract=contract)
    env.items = dict(sensor=0, flag=1, sensor_optic_local_m=(0., 0., 0.), flag_face_y_m=-.057,
                     flag_center_z_m=1.083)

    def at(dx):
        # Beam straight down -y from 100 mm in front of the flag face, dx off the flag's centre line.
        env.state = SimpleNamespace(body_q=Arr([[.2655 + dx, .043, 1.083, 0, 0, 0, 1],
                                                [.2655, -.063, .934, 0, 0, 0, 1]]))
        return env._optical()

    assert at(.004)["range_m"] == pytest.approx(.1)
    assert at(.004)["detected"] is detected
    assert at(.0028)["detected"] and not at(.0052)["detected"]


@pytest.mark.parametrize("task", ("frame", "support", "access", *NEW_TASKS))
def test_rail_level_hands_off_and_position_scope(env_class, task):
    env = fake(env_class, task)
    env._pitch_yaw = (.00301, 0.)
    assert not env._evaluate()["success"]
    env._pitch_yaw = (0., .00301)
    assert env._evaluate()["success"] == (task not in ("frame", "support"))
    env._pitch_yaw = (0., 0.)
    env.state.body_q.a[0, 0] = .14075
    assert env._evaluate()["success"] == (task != "support")
    env.state.body_q.a[0, 0] = .14101
    assert env._evaluate()["success"] == (task not in ("frame", "support"))
    env._held["frame"] = True
    before = dict(env._task_set_state["free_rail"])
    env._pitch_yaw = (.03, .002)
    track_free_rail(env)
    assert env._task_set_state["free_rail"] == before


def test_retention_holds_and_backoff_revoke_clicks(env_class):
    env = fake(env_class, "sensor")
    n = "sensor_left"
    env._retained_ticks[n] = 0
    for i in range(10):
        env._track_retention()
        assert env._retained_ticks[n] == i+1
    assert env._evaluate()["success"]
    previous = env._free_load[n]
    env._held["sensor"] = True
    env._preload[n] *= 2
    env._track_retention()
    assert env._retained_ticks[n] == 0 and env._free_load[n] == previous
    env._held["sensor"] = False
    env._bit_contacts[n] = 1
    env._track_retention()
    assert env._free_load[n] == previous
    turn = int(env.model.joint_q_start.a[env.items[n]["turn_joint"]])
    env.state.joint_q.a[turn] -= .01001
    env._track_retention()
    assert env._seated_clicks[n] == 0
    env._bit_contacts[n] = 0
    env._seated_clicks[n] = 1
    env._preload[n] = .5 * env.minimum_load(n)
    for _ in range(10): env._track_retention()
    assert env._seated_clicks[n] == 0


def test_initial_coordinates_preserve_helix_and_only_move_new_presets():
    for task in ("front-m4", "find-loose", "mixed", "sensor"):
        c = Config(task=task, seed=9)
        builder = SimpleNamespace(joint_q=[.0371]*12, joint_q_start=list(range(12)),
                                  joint_label=["horizontal_profile_slot_travel"])
        items = {n: {"slide_joint": 4+2*i, "turn_joint": 5+2*i,
                     "pitch_m": .0005 if n.startswith("sensor") else .0007} for i,n in enumerate(SCREWS)}
        apply_start_coordinates(builder, items, c)
        if task not in NEW_TASKS:
            assert builder.joint_q == [.0371]*12
            continue
        assert builder.joint_q[:4] == [0.]*4
        for n, item in items.items():
            advance = builder.joint_q[item["slide_joint"]]
            assert advance == (0. if n == loose_screw(task, 9) else .000498)
            assert advance == pytest.approx(item["pitch_m"]*builder.joint_q[item["turn_joint"]]/(2*math.pi))


def test_reset_reuses_settled_baseline_and_clears_episode_credit(env_class):
    env = fake(env_class, "access")
    start = fake(env_class, "access", start=True)
    env.initial = start.state
    def assign(state):
        env.state.joint_q.assign(state.joint_q.numpy())
        env.state.body_q.assign(state.body_q.numpy())
        env._pitch_yaw = start._pitch_yaw
    env.state.assign = assign
    env.next_state = SimpleNamespace(assign=lambda state: None)
    env.initial_targets = np.zeros(13)
    env.control = SimpleNamespace(joint_target_q=Arr(np.zeros(13)), joint_target_qd=Arr(np.zeros(13)), joint_f=Arr(np.zeros(13)))
    env.pipeline = SimpleNamespace(reset_contact_matching=lambda: None, contacts=lambda: None)
    env._make_solver = lambda: None
    env._map_friction_rows = lambda: None
    env._setup_arm_motion = lambda: None
    env._settled_loads = (start._preload, start._thread_load, start._interface)
    env.lock = __import__("threading").RLock()
    env.generation = 0
    env.get_state = lambda: env.evaluation
    result = env.reset()
    assert result["score"] == 0. and not result["success"]
    assert env._task_set_state["start_loads"] == start._preload
    assert "_task_set_state" in env._TRACKED


def test_settle_ramps_only_the_other_screws_and_restores_forces(env_class):
    env = fake(env_class, "find-loose")
    env.config = replace(env.config, initial_settle_s=10*env.config.dt)
    env._screw_dof = dict(zip(SCREWS, range(4)))
    env.control = SimpleNamespace(joint_f=Arr(np.zeros(4)))
    env.trace_callback = object()
    original_trace = env.trace_callback
    env.initial = env.next_state = SimpleNamespace(assign=lambda s: None)
    forces, turning = [], []
    def step():
        forces.append(env.control.joint_f.numpy())
        turning.append(set(env._assembly_turning))
    env.step = step
    loose = loose_screw(env.config.task, env.config.seed)
    env._preload[loose] = 0.
    settle_initial(env)
    assert len(forces) == 10
    assert all(f[env._screw_dof[loose]] == 0. for f in forces)
    for n in pretightened_screws(env.config):
        assert forces[3][env._screw_dof[n]] == env.items[n]["limit_nm"]
    assert all(not s for s in turning[6:])
    assert env.trace_callback is original_trace and not env._assembly_turning
    assert env.control.joint_f.numpy().tolist() == [0.]*4
    assert env.evaluation["score"] == 0. and not env.evaluation["success"]
    env._preload["frame_top"] = 0.
    with pytest.raises(RuntimeError, match="did not settle"):
        settle_initial(env)


def test_start_hooks_do_not_move_robot_or_existing_props():
    # No new work positions. Conservative geometric reach check from the
    # existing mounts, with room for the driver/hand standoff (not an IK proof).
    mounts = {"left": (-.43, -.26, .78), "right": (.43, -.26, .78)}
    targets = [("right", (.261, .0733, 1.074)), ("right", (.270, .0733, 1.074)),
               ("left", (0., .0845, 1.067)), ("left", (.022, .10, 1.0792)),
               ("right", (.2655, -.063, 1.083))]
    for arm, pos in targets:
        assert math.dist(mounts[arm], pos) + .15 < .855
    for home in (0., -.0035):
        assert math.dist(mounts["right"], (.2655 + home - .0241, -.063, .785)) < .855


def test_build_adds_only_the_home_prop_at_the_selected_home():
    from tighten_codex.presets import build
    for task, contract in (("sensor", ""), ("mixed", ""), ("sensor", CONTRACT), ("aim", CONTRACT),
                           ("frame", CONTRACT), ("access", CONTRACT), ("support", CONTRACT),
                           ("front-m4", CONTRACT), ("find-loose", CONTRACT)):
        added = []
        builder = SimpleNamespace(joint_q=[0.]*12, joint_q_start=list(range(12)),
                                  joint_label=["horizontal_profile_slot_travel"])
        builder.add_shape_box = lambda *a, **kw: added.append((a, kw)) or 999
        items = {n: {"slide_joint": 4+2*i, "turn_joint": 5+2*i,
                     "pitch_m": .0005 if n.startswith("sensor") else .0007} for i,n in enumerate(SCREWS)}
        result = (builder, items, {}, object(), object())
        newton = SimpleNamespace(ModelBuilder=SimpleNamespace(ShapeConfig=lambda **kw: kw))
        warp = SimpleNamespace(transform=lambda pos, rot: pos, quat_identity=lambda: (0,0,0,1))
        config = Config(task=task, task_contract=contract)
        with patch("tighten_codex.scene.build", return_value=result) as base, patch.dict(sys.modules, {"newton": newton, "warp": warp}):
            assert build(config) is result
        base.assert_called_once_with(config, "tighten-fr3")
        wants_stop = task in NEW_TASKS or contract and task in ("sensor", "aim")
        assert len(added) == int(bool(wants_stop))
        if wants_stop:
            args, kwargs = added[0]
            assert args == (-1,)
            assert kwargs["xform"] == pytest.approx((.2414 + flag_home(config), -.063, .785))
            assert kwargs["cfg"]["ke"] == 2.5e7 and kwargs["cfg"]["kd"] == 4.6e4
            assert result[2][999] == "robot_dark"


def test_checkpoint_restores_original_progress_baseline(env_class):
    env = fake(env_class, "front-m4")
    env.lock = __import__("threading").RLock()
    env.state.body_qd = Arr(np.zeros((4, 6)))
    env.state.joint_qd = Arr(np.zeros(13))
    env.next_state = SimpleNamespace(assign=lambda state: None)
    env.model.joint_dof_count = 13
    env.control = SimpleNamespace(joint_target_q=Arr(np.zeros(13)), joint_target_qd=Arr(np.zeros(13)), joint_f=Arr(np.zeros(13)))
    env.targets = np.zeros(13)
    env.setup = SimpleNamespace(id="stub", fingerprint="stub")
    env._arm_motion = env._clutch = env._screw_lock = env._head_touched = {}
    env._clamp_stick = []
    env.tick, env.generation = 11, 0
    env.pipeline = SimpleNamespace(reset_contact_matching=lambda: None, contacts=lambda: None)
    env._make_solver = lambda: None
    env._map_friction_rows = lambda: None
    env._setup_arm_motion = lambda *args: None
    env.get_state = lambda: env.evaluation
    arrays, meta = env.checkpoint()
    original = dict(env._task_set_state["start_loads"])
    env._task_set_state["start_loads"]["frame_front"] = 10000.
    result = env.restore(arrays, meta)
    assert env._task_set_state["start_loads"] == original
    assert result["success"] and result["score"] == 1.
    # The checkpoint records the contract; it restores into a v2 scene only.
    assert meta["config"]["task_contract"] == CONTRACT
    env.config = Config(task="front-m4", task_contract=EIGHT_CONTRACT)
    with pytest.raises(ValueError, match="different scene configuration"):
        env.restore(arrays, meta)


def test_public_policy_instruction_uses_selected_contract_without_leaking_truth():
    from gym_anything_robotics.backends.newton_tighten_codex import NewtonTightenCodexBackend
    backend = object.__new__(NewtonTightenCodexBackend)
    backend.robot, backend.cameras, backend.specs, backend.kinds = "dual-fr3", (), {}, frozenset()
    for task in EIGHT_ORDER[:-1]:
        backend.env = SimpleNamespace(config=Config(task=task, task_contract=CONTRACT), joints={})
        policy = backend.policy_access()
        assert policy.instruction() == instruction(backend.env.config) == EIGHT_INSTRUCTIONS[task]
        assert "fasteners" not in policy.proprioception and "loose_screw" not in policy.hardware

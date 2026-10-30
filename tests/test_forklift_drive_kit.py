"""Forklift robot selection: the G1 refusal, one alias table, the robot's camera and notes from the selected rig, and
the driving-robot kit (catalogue data and selection).

Fast checks; no physics.
"""

import json
import math
from types import SimpleNamespace

import pytest
from tests.packages import PACKAGES

ROOT = PACKAGES


def test_one_alias_table_for_catalogue_config_and_operator():
    from robots.catalog import ENVIRONMENTS
    from forklift.robot_setup import ALIASES, validate_setup
    from forklift import config as FC
    import forklift.operator as OP
    aliases = ENVIRONMENTS["forklift"]["aliases"]
    assert ALIASES == aliases == {"h1-2": "forklift-h1-2", "drive-kit": "forklift-drive-kit"}
    assert FC.ROBOTS == (*aliases, "none")
    assert OP.KIT_RIGS == {"drive-kit": "forklift-drive-kit"} and OP.RIGS["h1-2"] == "forklift-h1-2"
    assert validate_setup("drive-kit") == validate_setup("forklift-drive-kit") == "drive-kit"
    with pytest.raises(ValueError, match="does not support|cannot press the service brake"):
        from robots.catalog import select_for_backend
        select_for_backend("forklift", "g1")
    for arm in ("single-fr3", "dual-fr3", "dual-panda"):
        assert "--robot drive-kit" in ENVIRONMENTS["forklift"]["unsupported"][arm]


def test_kit_rig_is_exact():
    import robots
    from forklift.robot_setup import validate_setup
    rig = robots.resolve("forklift-drive-kit").to_dict()
    assert rig["instances"] == {"drive_kit": {"model": "drive_kit@1", "mount": "cab", "controller_profile": "forklift"}}
    bad = json.loads(json.dumps(rig))
    bad["instances"]["drive_kit"]["mount"] = "operator_seat"
    with pytest.raises(ValueError, match="refused rather than run as"):
        validate_setup(bad)


def test_kit_product_figures_are_the_cited_ones():
    """The kit's limits are the products' published figures (robots/models/drive_kit/robot.json sources)."""
    import robots
    spec = robots.model("drive_kit").to_dict()
    j = spec["actuation"]["profiles"]["forklift"]["joints"]
    sr = j["steer"]
    assert sr["peak"] == 20.0 and sr["friction"] == 4.0 and sr["armature"] == 0.076 and sr["mass_kg"] == 5.0
    assert sr["envelope"][-2][1] == 15.0 and math.isclose(sr["envelope"][-2][0], math.radians(500), rel_tol=1e-4)
    assert sr["envelope"][-1][1] == 10.0 and math.isclose(sr["envelope"][-1][0], math.radians(1000), rel_tol=1e-4)
    assert j["brake"]["envelope"] == [[0.0, 750.0], [1.0, 750.0], [1.3, 500.0]]
    assert j["accelerator"]["envelope"] == [[0.0, 150.0], [0.30, 150.0]] and j["accelerator"]["stroke_m"] == 0.130
    assert j["inching"]["envelope"] == [[0.0, 350.0], [0.40, 350.0]] and j["inching"]["stroke_m"] == 0.150
    for n in ("lift", "tilt", "shift", "direction", "park"):
        assert j[n]["envelope"] == [[0.0, 300.0], [1.0, 300.0]]
    src = spec["sources"]
    for key, figure in (("ab_dynamics_steering", "20 N m"), ("ab_dynamics_cbar600", "750 N"),
                        ("ab_dynamics_pedal_robots", "150 N"), ("4activedr", "350 N"), ("gsa9001s", "300 N")):
        assert figure in src[key], key
    assert set(spec["poses"]["installed"]) == set(j)


def test_policy_notes_name_the_selected_robot():
    """The policy notes name the selected robot; the robot's name comes from its seat fit."""
    import forklift.operator as OP
    world = SimpleNamespace(config=SimpleNamespace(dt=0.0005),
                            operator=SimpleNamespace(fit=OP.SEAT_FIT["h1_2"], model_id="h1_2",
                                                     motors=SimpleNamespace(followers=["h1_2/L_index_intermediate_joint"])))
    notes = OP.policy_notes(world, 4, 30.0)
    assert notes["control"] == (
        "Unitree low-level motor command per H1-2 joint: named_joint_target names '<joint>' (q*, rad), "
        "'<joint>.dq' (rad/s), '<joint>.kp' (N m/rad), '<joint>.kd' (N m s/rad), '<joint>.tau' (N m "
        "feed-forward); unnamed channels keep their last value. tau = kp (q* - q) + kd (dq* - dq) + tau_ff, "
        "clipped to the motor's torque-speed envelope. A command lasts at least 4 ticks (2 ms) and at most "
        "30 s. The Inspire hands' mimic followers are not motors. The start command holds the seated "
        "posture with a gravity feed-forward")
    assert notes["cameras"].startswith("Ideal pinhole RGB of the H1-2 head camera (848 x 480, 69.4 deg horizontal")
    world.operator = SimpleNamespace(fit=OP.SEAT_FIT["g1"], model_id="g1", motors=SimpleNamespace(followers=[]))
    notes = OP.policy_notes(world, 4, 30.0)
    assert "per G1 joint" in notes["control"] and "mimic" not in notes["control"]



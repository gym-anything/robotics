"""Forklift's seated operator as a function of the selected humanoid rig (no physics; seconds).

The operator derives everything robot-specific from the catalogue model and its URDF tree (forklift/operator.py
Skeleton): these checks hold that derivation to the H1-2's hand-written selections, check the G1's catalogue additions
(forklift profile, seated pose, frames, rig), the seat fit table, the robot-setup validator's exact-rig rule and the
refusal texts.
"""

import json

import pytest
from tests.packages import PACKAGES

ROOT = PACKAGES


def _skeleton(model_id):
    import robots
    from robots.assets import resolve as asset
    from forklift.operator import Skeleton
    try:
        path = asset(model_id)
    except Exception as exc:                      # locked assets not installed on this host
        pytest.skip(f"{model_id} assets not installed: {exc}")
    spec = robots.model(model_id).to_dict()
    import xml.etree.ElementTree as ET
    x = ET.parse(path).getroot()
    # bodies after the shared builder's fixed-joint collapse: links that are not the child of a fixed joint
    fixed_children = {j.find("child").get("link") for j in x.findall("joint") if j.get("type") == "fixed"}
    links = [l.get("name") for l in x.findall("link") if l.get("name") not in fixed_children]
    return Skeleton(spec, path.read_bytes(), links), spec


def test_h1_2_skeleton_reproduces_the_hand_written_selections():
    from forklift.motors import group_of
    sk, spec = _skeleton("h1_2")
    joints = [j for g in spec["groups"].values() for j in g["joints"]]
    # motor groups (gain bounds, armature) as motors.group_of names them
    assert {j: sk.motor_group(j) for j in joints} == {j: group_of(j) for j in joints}
    # the start posture's joint selections (operator.seated_posture)
    assert [j for j in joints if sk.is_role(j, "hip_pitch")] == [j for j in joints if j.endswith("hip_pitch_joint")]
    assert [j for j in joints if sk.is_role(j, "knee", "ankle_pitch", "ankle_roll", "hip_pitch")] == \
        [j for j in joints if any(k in j for k in ("knee", "ankle", "hip_pitch"))]
    assert [j for j in joints if sk.is_role(j, "shoulder", "elbow", "wrist")] == \
        [j for j in joints if any(k in j for k in ("shoulder", "elbow", "wrist"))]
    assert [sk.thumb[s][0] for s in ("left", "right")] == ["L_thumb_proximal_yaw_joint", "R_thumb_proximal_yaw_joint"]
    # its link sets (the filtered hull pairs, the seat and back links, the shanks, the knees, the chest, the hands)
    assert sk.bodies("trunk") == ["pelvis", "torso_link"]
    assert sorted(sk.bodies("thigh")) == sorted(f"{s}_hip_{j}_link" for s in ("left", "right") for j in ("pitch", "roll"))
    assert sorted(sk.bodies("shank")) == sorted(f"{s}_{k}_link" for s in ("left", "right") for k in ("knee", "ankle_pitch", "ankle_roll"))
    assert sk.bodies("knee") == ["left_knee_link", "right_knee_link"]
    assert sk.body(sk.links["chest"][0]) == "torso_link"
    assert sk.bodies("palm") == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert sk.bodies("thumb1") == ["L_thumb_proximal", "R_thumb_proximal"]
    hand = set(sk.bodies("wrist") + sk.bodies("palm") + sk.bodies("fingers"))
    assert hand == {l for l in sk.body_links if any(k in l for k in ("wrist", "thumb", "index", "middle", "ring", "pinky"))}
    assert sk.camera_link == "camera_link"


def test_g1_skeleton_roles():
    sk, spec = _skeleton("g1")
    assert sk.torso == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    for s in ("left", "right"):
        assert [sk.role[j][2] for j in sk.legs[s]] == ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
        assert [sk.role[j][2] for j in sk.arms[s]] == ["shoulder"] * 3 + ["elbow"] + ["wrist"] * 3
        assert sk.thumb[s] == [f"{s}_hand_thumb_{k}_joint" for k in range(3)]
    groups = {j: sk.motor_group(j) for g in spec["groups"].values() for j in g["joints"]}
    assert groups["waist_pitch_joint"] == "torso" and groups["left_hand_index_0_joint"] == "hand"
    assert groups["left_wrist_pitch_joint"] == "wrist" and groups["left_elbow_joint"] == "arm"
    assert groups["right_ankle_roll_joint"] == "leg"
    assert sk.bodies("trunk") == ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"]
    assert sk.bodies("thigh", ("left",)) == ["left_hip_roll_link", "left_hip_yaw_link"]
    assert sk.bodies("palm") == ["left_wrist_yaw_link", "right_wrist_yaw_link"]   # the Dex3 palm merges into the wrist
    assert sk.body(sk.camera_link) == "torso_link"


def test_g1_catalogue_additions_are_complete_and_additive():
    import robots
    from robots.assets import resolve as asset
    import xml.etree.ElementTree as ET
    spec = robots.model("g1").to_dict()
    prof = spec["actuation"]["profiles"]["forklift"]
    assert prof["initial_pose"] == "forklift_seated"
    for old in ("whiteboard", "football", "barcode_scanning", "batting"):
        assert old in spec["actuation"]["profiles"] and old in spec["poses"]      # nothing removed
    try:
        path = asset("g1")
    except Exception as exc:
        pytest.skip(f"g1 assets not installed: {exc}")
    x = ET.parse(path).getroot()
    active = {j.get("name"): j.find("limit").attrib for j in x.findall("joint") if j.get("type") == "revolute"}
    assert set(prof["joints"]) == set(active) == set(spec["poses"]["forklift_seated"])
    for n, row in prof["joints"].items():
        assert row["effort_limit"] == float(active[n]["effort"]) and row["velocity_limit"] == float(active[n]["velocity"])
        assert row["kp"] > 0 and row["kd"] > 0 and row["armature"] > 0
        assert float(active[n]["lower"]) <= spec["poses"]["forklift_seated"][n] <= float(active[n]["upper"])
    assert {k: spec["frames"][k]["parent"] for k in ("left_hand", "right_hand", "head_camera")} == \
        dict(left_hand="left_hand_palm_link", right_hand="right_hand_palm_link", head_camera="d435_link")
    rig = robots.resolve("forklift-g1").to_dict()
    assert rig["instances"] == {"g1": {"model": "g1@1", "mount": "operator_seat", "controller_profile": "forklift"}}
    assert rig["attachments"] == {}


def test_every_seated_rig_is_fitted_and_offered_rigs_are_seatable():
    import robots
    from forklift import operator as OP
    from forklift.robot_setup import ALIASES
    for alias, rig in OP.RIGS.items():
        models = {i["model"].split("@")[0] for i in robots.resolve(rig).to_dict()["instances"].values()}
        assert len(models) == 1 and models <= set(OP.SEAT_FIT), (alias, rig)
        inst = next(iter(robots.resolve(rig).to_dict()["instances"].values()))
        assert inst["mount"] == "operator_seat" and inst["controller_profile"] == "forklift"
    # every offered robot is a seated humanoid or the driving-robot kit (operator.add_kit)
    assert all(dict(OP.RIGS, **OP.KIT_RIGS)[a] == r for a, r in ALIASES.items())
    assert set(OP.KIT_RIGS) == {"drive-kit"} and set(OP.RIGS) == {"h1-2", "g1"}
    # the G1 has no convex-hull artifact in the seat (measured: its pelvis and hip-roll hulls 3.1-3.4 mm apart, its
    # resting thumb tips touching its thighs at 1-4 N with the meshes 0.1-0.2 mm apart: real contacts, kept)
    assert OP.SEAT_FIT["g1"]["filtered"] == ()
    assert OP.SEAT_FIT["h1_2"]["filtered"] == (("trunk", "thigh"), ("palm", "thumb1"))
    with pytest.raises(ValueError, match="not fitted"):
        OP.seat_fit("ur10e")


def test_setup_validator_accepts_only_the_rigs_it_builds():
    import robots
    from forklift.robot_setup import ALIASES, validate_setup
    for alias, rig in ALIASES.items():
        setup = robots.resolve(rig)
        assert validate_setup(alias) == validate_setup(setup) == validate_setup(setup.to_dict()) == alias
        renamed = dict(setup.to_dict(), id=rig + "-copy", description="the same robot under another id")
        assert validate_setup(renamed) == alias
        d = setup.to_dict()
        name = next(iter(d["instances"]))
        for key, value in (("controller_profile", "batting"), ("mount", "g1_floor_mount"), ("initial_pose", "batting")):
            bad = json.loads(json.dumps(d))
            bad["instances"][name][key] = value
            if key == "controller_profile" and value not in robots.model(bad["instances"][name]["model"]).to_dict()["actuation"]["profiles"]:
                continue                                    # the catalogue itself refuses an unknown profile
            if key == "initial_pose" and value not in robots.model(bad["instances"][name]["model"]).to_dict()["poses"]:
                continue
            with pytest.raises(ValueError, match="refused rather than run as"):
                validate_setup(bad)
    with pytest.raises(ValueError, match="h1_2@1"):
        validate_setup(robots.resolve("scan-fr3"))


def test_refusals_cite_their_physical_reasons():
    from robots.catalog import ENVIRONMENTS
    un = ENVIRONMENTS["forklift"]["unsupported"]
    for arm in ("single-fr3", "dual-fr3", "dual-panda"):
        text = un[arm]
        assert "fixed" in text, arm                              # world-fixed base
        from forklift import params as P
        from forklift.drivetrain import pedal_force_at
        t = P.PEDAL_BRAKE["travel"]
        brake = f"{pedal_force_at(0.0):.0f}-{pedal_force_at(t):.0f} N at its pad over {t * P.BRAKE_PAD_ARM * 1000:.0f} mm"
        for force in (brake, "14-60 N", "7-35 N"):              # the pedal forces at the pads (service brake, inching, accelerator; the brake's from its model)
            assert force in text, (arm, force)
    if "g1" in un:                                              # refused: the physical probe's numbers
        assert "E14g" in un["g1"]
    else:
        assert ENVIRONMENTS["forklift"]["aliases"]["g1"] == "forklift-g1"

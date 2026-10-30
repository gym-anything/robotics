"""The deposit_cans eight-task catalogue: catalogue, instructions, contract wiring, start placements and the
task-specific verifier helpers (no simulation)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deposit_cans import variants
from deposit_cans.geometry import CAN_SPECS
from deposit_cans.task import (CUP_BASE_RADIUS, CUP_RIM_RADIUS, DESCRIPTION, HIGH_FILL_CONTRACT, HIGH_FILL_DESCRIPTION,
                               LANDFILL_HEIGHT, PAPER_CUP_BASE_RADIUS, PAPER_CUP_HEIGHT, PAPER_CUP_RIM_RADIUS,
                               TASK_CONTRACTS, TASKS_CONTRACT, description, frustum_z_range, inside_landfill,
                               task_contract)

TASKS_V2 = variants.TASKS_V2_CONTRACT
INVENTORIES = {"single_can": (1, 0, 48), "lid_cans": (3, 0, 48), "sort_cups": (8, 0, 48), "sort_waste": (6, 0, 48),
               "empty_tub": (16, 16, 48), "tub_cups": (8, 8, 48), "reference": (24, 16, 48), "high_fill": (24, 16, 120)}
# Default contract of a bare selection (reference: v1).
CONTRACTS = {"single_can": TASKS_CONTRACT, "empty_tub": TASKS_CONTRACT, "high_fill": HIGH_FILL_CONTRACT,
             "lid_cans": TASKS_V2, "sort_cups": TASKS_V2, "sort_waste": TASKS_V2, "tub_cups": TASKS_V2, "reference": "v1"}
# The eight-task catalogue, exactly (benchmark.json task_set).
TASK_SET = [{"variant": "single_can", "seed": 0}, {"variant": "lid_cans", "seed": 0}, {"variant": "sort_cups", "seed": 0},
            {"variant": "sort_waste", "seed": 0}, {"variant": "empty_tub", "seed": 0}, {"variant": "tub_cups", "seed": 0},
            {"variant": "reference", "contract": TASKS_V2, "seed": 0}, {"variant": "high_fill", "seed": 0}]
FR3_REACH_M = .855
BASES = {"left": np.array((.12, -.65, 1.02)), "right": np.array((.12, .65, 1.02))}
COUNTER_CANS = [(-.60 + .15 * (j % 4), -.53 if j < 4 else .53) for j in range(8)]   # assembly.can_inventory, j = i


def config_of(name, contract=None, scene_revision=7):
    return SimpleNamespace(**{"task_contract": "", **variants.resolve(name, 0, contract)["config"]}, scene_revision=scene_revision)


def catalogue():
    return json.loads((Path(variants.__file__).parent / "benchmark.json").read_text())


def test_catalogue_has_eight_variants_and_keeps_the_default_task():
    raw = catalogue()
    assert raw["count"] == 8 == len(variants.PRESETS) == len(variants.CATALOGUE)
    assert raw["selector_choices"]["variant"] == list(variants.CATALOGUE)
    assert raw["selector_choices"]["contract"] == list(variants.CONTRACT_CHOICES["reference"]) == ["v1", TASKS_V2]
    assert raw["selectors"] == ["variant", "seed", "contract"]
    assert raw["default_task_id"] == variants.task_resolve({})["task_id"] == "deposit-cans/v1/reference/seed-0"
    assert [v["id"] for v in variants.contract()["variants"]] == list(variants.CATALOGUE)


def test_eight_task_set_lists_each_exact_selection():
    entries = catalogue()["task_set"]["tasks"]
    assert [e["selection"] for e in entries] == TASK_SET == [dict(s) for s in variants.TASK_SET]
    assert variants.contract()["task_set"] == TASK_SET
    for entry in entries:
        selection = entry["selection"]
        resolved = variants.task_resolve(selection)
        assert resolved["selection"] == selection and resolved["task_id"] == entry["task_id"]
        assert task_contract(config_of(selection["variant"], selection.get("contract"))) == entry["task_contract"]
        assert resolved["description"] == description(selection["variant"], selection.get("contract"))
    assert len({e["task_id"] for e in entries}) == 8


def test_inventories_are_distinct():
    triples = [(p["target_count"], p["tub_count"], p["preload_count"]) for p in variants.PRESETS.values()]
    assert len(set(triples)) == len(triples)


@pytest.mark.parametrize("name", list(INVENTORIES))
def test_presets_resolve_to_their_own_inventory_contract_and_instruction(name):
    config = config_of(name)
    assert (config.target_count, config.tub_count, config.preload_count) == INVENTORIES[name]
    assert variants.preset(config) == name
    assert task_contract(config) == CONTRACTS[name] == TASK_CONTRACTS[name]
    resolved = variants.task_resolve({"variant": name})
    assert resolved["description"] == description(name)
    assert resolved["task_id"] == f"deposit-cans/v1/{name}/seed-0"


@pytest.mark.parametrize("selection", TASK_SET)
def test_instructions_state_the_job_not_the_verifier(selection):
    text = description(selection["variant"], selection.get("contract"))
    for word in ("designated", "settled", "lodged", "Preserve", "mm", "second", "upright", "rest"):
        assert word not in text


def test_reference_default_is_v1_and_the_eight_task_contract_is_explicit():
    bare = variants.task_resolve({"variant": "reference"})
    assert bare == dict(task_id="deposit-cans/v1/reference/seed-0", description=DESCRIPTION,
                        backend_options={"variant": "reference", "config": {"seed": 0}},
                        selection={"variant": "reference", "seed": 0}, split=None)
    assert variants.resolve("reference", 3) == dict(id="reference", seed=3, config=dict(variants.PRESETS["reference"], seed=3))
    assert variants.task_resolve({"variant": "reference", "contract": "v1"}) == bare
    assert task_contract(config_of("reference")) == task_contract(config_of("reference", "v1")) == "v1"
    eight = variants.task_resolve({"variant": "reference", "contract": TASKS_V2})
    assert eight["task_id"] == f"deposit-cans/v1/reference/{TASKS_V2}/seed-0"
    assert eight["selection"] == {"variant": "reference", "seed": 0, "contract": TASKS_V2}
    assert eight["backend_options"] == {"variant": "reference", "contract": TASKS_V2, "config": {"seed": 0}}
    assert eight["description"] == HIGH_FILL_DESCRIPTION
    assert task_contract(config_of("reference", TASKS_V2)) == TASKS_V2
    # A bare reference inventory with no task_contract is judged by v1.
    assert task_contract(SimpleNamespace(**variants.PRESETS["reference"], seed=0, scene_revision=7)) == "v1"
    with pytest.raises(ValueError):
        variants.resolve("reference", 0, "high-fill-v1")
    with pytest.raises(ValueError):
        variants.resolve("high_fill", 0, "v1")
    with pytest.raises(ValueError):
        task_contract(SimpleNamespace(**variants.PRESETS["high_fill"], seed=0, scene_revision=7, task_contract="v1"))
    with pytest.raises(ValueError):
        task_contract(config_of("reference", TASKS_V2, scene_revision=6))
    with pytest.raises(ValueError):
        description("sort_cups", "v1")


def test_config_task_contract_defaults_to_standing_contract():
    from deposit_cans.config import Config
    assert Config().task_contract == ""
    with pytest.raises(TypeError):
        Config(task_contract=None)


def test_sort_cups_needs_no_hand_off():
    """All three cups stand on the right-hand row; the right arm reaches each cup and the tray."""
    tray_centre = np.array((-.43, .375, 1.0))
    for x, y, z in variants.SORT_CUPS_ON_COUNTER:
        assert y == .53 and z == .903
        assert np.linalg.norm(np.array((x, y, .99)) - BASES["right"]) < FR3_REACH_M - .15
        # Midway between two counter cans of the row, the cup rim clears each can by at least 4.5 mm.
        for cx, cy in COUNTER_CANS:
            if cy != y:
                continue
            radius = CAN_SPECS[COUNTER_CANS.index((cx, cy)) % 3][0] / 2
            assert np.hypot(x - cx, y - cy) - CUP_RIM_RADIUS - radius >= .0045 - 1e-9
    assert np.linalg.norm(tray_centre - BASES["right"]) < FR3_REACH_M - .2
    assert np.linalg.norm(np.array((-.23, .285, 1.0)) - BASES["left"]) > FR3_REACH_M   # out of the left arm's reach


def test_lid_cans_stand_on_the_flat_roof_within_reach():
    slot_half, roof_half = (.2032, .05715), (.316, .2276)   # station.lid_mesh revision 3, rings 3 and 2
    for k, (x, y) in enumerate(variants.LID_CANS_XY):
        radius = CAN_SPECS[k % 3][0] / 2
        assert abs(y) - radius - slot_half[1] >= .045 and roof_half[1] - abs(y) - radius >= .05
        assert roof_half[0] - abs(x) - radius > .1
        world = np.array((variants.STATION_XY[0] + x, variants.STATION_XY[1] + y, variants.LID_ROOF_Z + .1))
        arm = "left" if y < 0 else "right"
        assert np.linalg.norm(world - BASES[arm]) < FR3_REACH_M - .1
        assert world[0] - radius - .12 > .2   # the start pose (robot.HOME) keeps each arm in the plane x = 0.12


def test_tub_cups_stand_on_the_flat_floor_clear_of_the_cans():
    tub_centre, floor_half, rib_end_y = np.array((-.37, 0.)), (.1945 - .019, .253 - .019), .095   # tub_geometry
    for k, (x, y, z) in enumerate(variants.TUB_CUPS):
        local = np.array((x, y)) - tub_centre
        assert local[1] - CUP_BASE_RADIUS > rib_end_y and abs(local[0]) + CUP_RIM_RADIUS < floor_half[0] - .03
        assert floor_half[1] - local[1] - CUP_RIM_RADIUS > .04
        assert .903 + .0017 < z < .903 + .0017 + .005
        row3_edge = -.185 + .074 * 3 + CAN_SPECS[0][0] / 2          # tub_layout rows 0-3, widest can
        assert local[1] - CUP_RIM_RADIUS - row3_edge >= .04
        assert np.linalg.norm(np.array((x, y, 1.0)) - BASES["right"]) < FR3_REACH_M - .05
    assert variants.PRESETS["tub_cups"]["tub_count"] == 8                         # tub_layout indices 0-7 only


def test_paper_cups_and_landfill_mouth_are_on_the_right_arm_side():
    for x, y, _ in variants.PAPER_CUPS:
        for j, (cx, cy) in enumerate(COUNTER_CANS[:6]):
            assert np.hypot(x - cx, y - cy) - PAPER_CUP_RIM_RADIUS - CAN_SPECS[j % 3][0] / 2 > .05
        assert np.linalg.norm(np.array((x, y, .98)) - BASES["right"]) < FR3_REACH_M - .3
    assert np.linalg.norm(np.array((.50, .99, .86)) - BASES["right"]) < FR3_REACH_M - .3   # 0.10 m above the bin mouth


def test_frustum_z_range_standing_lying_and_inverted():
    up, side, down = np.array((0, 0, 1.)), np.array((1., 0, 0)), np.array((0, 0, -1.))
    base = np.array((0, 0, .5))
    assert frustum_z_range(base, up, .03, .045, .11) == pytest.approx((.5, .61))
    assert frustum_z_range(base, side, .03, .045, .11) == pytest.approx((.455, .545))
    assert frustum_z_range(base, down, .03, .045, .11) == pytest.approx((.39, .5))


def test_inside_landfill():
    assert inside_landfill((0., 0., .2), .3)
    assert not inside_landfill((0., 0., .2), LANDFILL_HEIGHT + .01)    # sticking out over the rim
    assert not inside_landfill((.26, 0., .2), .3)                      # beside the bin
    assert not inside_landfill((0., 0., -.01), .1)                     # under it


def test_paper_cup_geometry_matches_the_verifier_frustum():
    from deposit_cans import props
    (rim, height), (base, zero) = props.PAPER_CUP_RINGS[:2]
    assert (rim, base, height, zero) == (PAPER_CUP_RIM_RADIUS, PAPER_CUP_BASE_RADIUS, PAPER_CUP_HEIGHT, 0.)
    v, f = props.cup_geometry(rings_rz=props.PAPER_CUP_RINGS)
    volume = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)
    assert volume > 0
    reusable, _ = props.cup_geometry()
    theta = np.arange(48) * 2 * np.pi / 48
    expected = np.concatenate([np.c_[r * np.cos(theta), r * np.sin(theta), np.full(48, z)]
                               for r, z in ((.0375, .132), (.028, 0.), (.026, .003), (.0355, .132))])
    assert np.array_equal(reusable, expected)   # the reusable cup keeps its own rings

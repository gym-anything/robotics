"""tighten-tasks-v2: the render-checked eight-task set, beside tighten-tasks-v1 (task.py, task_evaluation.py).

Same tasks, starts, instructions, checks, progress and failure factor as tighten-tasks-v1. What differs (render-check
fix round): detection follows each task's own rule. sensor and aim keep their legacy whole-spot rule
(the whole nominal 4.2 mm spot on the 10 x 44 mm flag: |dx| <= 2.9 mm, |dz| <= 19.9 mm), in env._optical and in the
success check at the flag's home; front-m4 and find-loose, built on mixed, use mixed's spot-centre-on-the-flag rule.
Every v2 selection names the contract and carries it in its task id (tighten-codex/v1/<task>/tighten-tasks-v2/seed-<n>),
front-m4 and find-loose included. mixed keeps its own evaluator and its released selection; bare, legacy and mixed
selections report exactly what the release reports.

task.py dispatches a selection or config with this contract here (contract_for, resolve), uses_eight counts it as an
eight-task contract (the same start setups, tracking and scoring), and env._optical and
task_evaluation.evaluate ask this module for the detection rule and the evaluation.
"""
from types import SimpleNamespace

from .presets import SCREWS, flag_home, pretightened_screws
from .task import EIGHT_ORDER, TASKS, instruction, task_verifier
from .task_evaluation import FAIL_FACTOR, clip, decreasing, required, score_state

CONTRACT = "tighten-tasks-v2"
# The eight as benchmark.json's task_set lists them: seven under this contract; mixed with its released selection,
# exactly as splits/test.json has it.
TASK_SET = tuple({"task": t} if t == "mixed" else {"task": t, "seed": 0, "contract": CONTRACT} for t in EIGHT_ORDER)


def contract_for(task, selected=CONTRACT):
    """Every task but mixed (which keeps its own evaluator) may select this contract."""
    if selected != CONTRACT:
        raise ValueError(f"task_set_v2 resolves {CONTRACT!r} only, not {selected!r}")
    if task not in TASKS or task == "mixed":
        raise ValueError(f"Unknown contract {selected!r} for {task}: mixed keeps tighten-mixed-v1")
    return CONTRACT


def detection_margin(config):
    """Margin subtracted from the flag's half-width/height for detection. Other tasks: the whole nominal 4.2 mm spot
    on the 10 x 44 mm face. mixed: most of the spot on the face (its centre on it). The rule follows the task, not
    the contract: sensor and aim keep their whole-spot rule; only the tasks built on mixed (front-m4, find-loose) use
    mixed's. A task's verifier.py sets SPOT_CENTRE_DETECTION for mixed's rule."""
    return 0. if getattr(task_verifier(config.task), "SPOT_CENTRE_DETECTION", False) else .0021


def resolve(selection):
    """task.resolve for a selection under this contract: the same selector checks; the contract is explicit for every
    task and part of the id."""
    if set(selection) - {"task", "seed", "contract"}:
        raise ValueError("Selectors are task, seed and contract")
    task = selection.get("task", "mixed")
    seed = selection.get("seed", 0)
    if task not in TASKS:
        raise ValueError(f"Unknown tighten task: {task}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    chosen = contract_for(task, selection.get("contract", CONTRACT))
    config = {"task": task, "seed": seed, "task_contract": chosen}
    return dict(task_id=f"tighten-codex/v1/{task}/{chosen}/seed-{seed}",
                description=instruction(SimpleNamespace(**config)),
                backend_options={"config": config}, selection={"task": task, "seed": seed, "contract": chosen},
                split=None)


def evaluate(env, metrics):
    """task_evaluation.evaluate with each task's own detection rule at the flag's home."""
    config = env.config
    # The task's criteria: its verifier.py (tasks/<task>/).
    criteria = task_verifier(config.task)
    need = required(config)
    keep_names = pretightened_screws(config) if getattr(criteria, "PRESERVE_OTHERS", False) else ()
    optical = env._optical()
    q = env.state.joint_q.numpy()
    starts = env.model.joint_q_start.numpy()
    body_q = env.state.body_q.numpy()
    flag_offset = float(q[int(starts[env.items["flag_joint"]])]) - flag_home(config)
    home_x = float(body_q[env.items["flag"], 0]) - flag_offset
    hit = optical.get("intersection_m")
    in_range = hit is not None and .04 <= optical["range_m"] <= .14
    # Aim grade (partial credit): mixed's spot-centre-on-the-home-face rule.
    on_face = bool(in_range and abs(hit[0] - home_x) <= .005 and abs(hit[2] - env.items["flag_center_z_m"]) <= .022)
    aim = 1. if on_face else min(1., 2 * env._spot_coverage(optical, home_x))
    # Success: the task's own detection rule (detection_margin) at the home x;
    # sensor and aim keep the whole 4.2 mm spot on the face, as before the task-set contract.
    margin = detection_margin(config)
    on_home = bool(in_range and abs(hit[0] - home_x) <= .005 - margin
                   and abs(hit[2] - env.items["flag_center_z_m"]) <= .022 - margin)
    returned = decreasing(flag_offset, .0005, .010)
    detection = aim * returned
    rail = env._task_set_state["free_rail"]
    pitch, yaw = env._rail_attitude(body_q)
    offset = float(body_q[env.items["rail"], 0] - .14)
    tol = config.rail_level_tolerance_rad
    level = decreasing(rail["pitch"], tol, env.FREE_DROOP_RAD)
    square = decreasing(rail["yaw"], tol, config.rail_yaw_clearance_rad)
    reference = getattr(criteria, "RAIL_SLOT_M", None)     # (band, zero) of the rail's slot offset
    if reference is not None:
        slot_band = reference[0]
        slot = decreasing(rail["offset"], slot_band, reference[1])
    else:
        square = slot = 1.
    rail_ok = (abs(pitch) <= tol and abs(rail["pitch"]) <= tol)
    if reference is not None:
        rail_ok = rail_ok and abs(yaw) <= tol and abs(rail["yaw"]) <= tol and abs(offset) <= slot_band and abs(rail["offset"]) <= slot_band
    if not getattr(criteria, "RAIL_LEVEL", True):
        rail_ok = True
    need_optical = getattr(criteria, "DETECTION", False)
    torque_ok = all(metrics[n]["peak_bit_socket_moment_2ms_nm"] <= .4 for n in SCREWS[:2])
    group = getattr(criteria, "HANDS_OFF_GROUP", "frame")
    hands_off = not env._held[group]

    def retained(name):
        m = metrics[name]
        return (m["retained_after_disengagement"] and m["hands_off_load_n"] >= m["minimum_retained_load_n"]
                and m["advance_m"] >= (.0004 if name.startswith("sensor_") else .00045)
                and abs(m["pitch_residual_m"]) <= .00010 and m["bit_socket_contact_count"] == 0)

    secured = all(retained(n) and metrics[n]["driver_released_on_screw"] for n in need)
    others_kept = all(retained(n) for n in keep_names)
    # A smooth 0.25 penalty per lost prior clamp.
    keep = 1. - .25 * sum(1. - clip(metrics[n]["hands_off_load_n"] / metrics[n]["minimum_retained_load_n"])
                          for n in keep_names)
    flag_at_home = abs(flag_offset) <= .0005
    checks = dict(required_fasteners_secured=secured, other_fasteners_kept=others_kept,
                  rail_in_reference=bool(rail_ok), assembly_hands_off=hands_off,
                  sensor_measured_torque_below_max=torque_ok,
                  flag_at_home=flag_at_home or not need_optical,
                  detecting_at_home=(bool(optical["detected"]) and on_home) or not need_optical,
                  no_fault=env.fault is None)
    success = all(checks.values())
    progress, seats, credits = score_state(config.task, need, metrics, env._task_set_state["start_loads"],
        level=level, square=square, slot=slot, detection=detection, keep=keep)
    return dict(task=config.task, task_contract=CONTRACT, success=success,
        score=1. if success else FAIL_FACTOR * progress, progress=progress, failed_attempt_factor=FAIL_FACTOR,
        required_fasteners=need, preserved_fasteners=keep_names, fasteners=metrics, checks=checks,
        optical=optical, flag_offset_m=flag_offset, flag_at_home=flag_at_home,
        rail_offset_m=offset, rail_pitch_rad=pitch, rail_yaw_rad=yaw, rail_in_reference=bool(rail_ok),
        rail_slide_m=float(q[int(starts[env.builder.joint_label.index("horizontal_profile_slot_travel")])]),
        bracket_offset_m=(body_q[env.items["bracket"], :3] - env._bracket_origin).tolist(),
        interface_normal_loads_n=dict(env._interface), interface_friction_capacity=dict(getattr(env, "_interface_capacity", {})),
        sensor_measured_torque_below_max=torque_ok, assembly_hands_off=hands_off, work_touched=dict(env._held),
        score_terms=dict(seat=seats, screw_credit=credits, rail_level_grade=level, rail_square_grade=square,
            rail_slot_grade=slot, aim_at_home_grade=aim, flag_return_grade=returned, detection_grade=detection,
            other_fasteners_load_grade=keep, start_loads_n=dict(env._task_set_state["start_loads"]),
            hands_off_rail=dict(rail), partial_mean=progress),
        evaluation_scope=f"{CONTRACT}: own-driver click, 10-tick hands-off retention, helical advance, "
            "M3 0.4 N m limit; task-specific rail, home-flag detection by the task's own rule (sensor and aim: the "
            "whole spot on the flag) and preserved clamps. "
            "Score 1 on success else 0.95 x continuous progress above settled starting loads.")

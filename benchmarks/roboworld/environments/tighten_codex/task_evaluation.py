"""End-state scoring of the eight-task contract. No simulator import; observations are privileged.

The mixed evaluator stays in env.py, and this contract does not change its
predicates or score. Here continuous progress is load/geometry, not a count of checks passed.
"""

import math

from .presets import NEW_TASKS, SCREWS, flag_home, loose_screw, pretightened_screws
from .task import EIGHT_CONTRACT

FAIL_FACTOR = .95


def clip(value):
    return max(0., min(1., value)) if math.isfinite(value) else 0.


def decreasing(error, band, zero):
    return clip((zero - abs(error)) / (zero - band))


def capture_start(env):
    """Saved on reset and in checkpoints; never inferred from action history."""
    pitch, yaw = env._rail_attitude()
    env._task_set_state = {
        "start_loads": dict(env._preload),
        "free_rail": {"pitch": pitch, "yaw": yaw,
                      "offset": float(env.state.body_q.numpy()[env.items["rail"], 0] - .14)},
    }


def track_free_rail(env):
    if not env._held["frame"]:
        pitch, yaw = env._rail_attitude()
        env._task_set_state["free_rail"] = {
            "pitch": pitch, "yaw": yaw,
            "offset": float(env.state.body_q.numpy()[env.items["rail"], 0] - .14),
        }


def required(config):
    if config.task in NEW_TASKS:
        return (loose_screw(config.task, config.seed),)
    return {"sensor": SCREWS[:2], "aim": SCREWS[:2],
            "frame": ("frame_front", "frame_top"), "support": ("frame_front", "frame_top"),
            "access": ("frame_top",)}[config.task]


def score_state(task, need, metrics, start_loads, *, level, square, slot, detection, keep):
    """Continuous seat progress measured above the actual settled starting load.

    Clicks and retention are success predicates only, so crossing their Boolean
    thresholds does not jump partial credit. A bit/hand cannot raise free_load.
    """
    seats, credits = {}, {}
    for name in need:
        m = metrics[name]
        floor = m["minimum_retained_load_n"]
        baseline = start_loads[name]
        span = floor - baseline
        # No credit for a screw already above its goal at reset.
        load = clip((m["hands_off_load_n"] - baseline) / span) if span > 0 else 0.
        advance = clip(m["advance_m"] / (.0004 if name.startswith("sensor_") else .00045))
        seat = load * advance
        if name.startswith("sensor_") and m["peak_bit_socket_moment_2ms_nm"] > .4:
            seat = 0.
        seats[name] = seat
        if task in NEW_TASKS:
            grade = level * (.75 + .25 * detection) * keep
        elif name.startswith("sensor_"):
            grade = .75 + .25 * detection
        elif name == "frame_top":
            grade = level * square * slot
        else:
            grade = 1.
        credits[name] = seat * grade
    return sum(credits.values()) / len(need), seats, credits


def evaluate(env, metrics):
    config = env.config
    need = required(config)
    keep_names = pretightened_screws(config) if config.task in NEW_TASKS else ()
    optical = env._optical()
    q = env.state.joint_q.numpy()
    starts = env.model.joint_q_start.numpy()
    body_q = env.state.body_q.numpy()
    flag_offset = float(q[int(starts[env.items["flag_joint"]])]) - flag_home(config)
    home_x = float(body_q[env.items["flag"], 0]) - flag_offset
    hit = optical.get("intersection_m")
    on_home = bool(hit is not None and .04 <= optical["range_m"] <= .14 and
                   abs(hit[0] - home_x) <= .005 and abs(hit[2] - env.items["flag_center_z_m"]) <= .022)
    aim = 1. if on_home else min(1., 2 * env._spot_coverage(optical, home_x))
    returned = decreasing(flag_offset, .0005, .010)
    detection = aim * returned
    rail = env._task_set_state["free_rail"]
    pitch, yaw = env._rail_attitude(body_q)
    offset = float(body_q[env.items["rail"], 0] - .14)
    tol = config.rail_level_tolerance_rad
    level = decreasing(rail["pitch"], tol, env.FREE_DROOP_RAD)
    square = decreasing(rail["yaw"], tol, config.rail_yaw_clearance_rad)
    slot_band = .0005 if config.task == "support" else .001
    slot = decreasing(rail["offset"], slot_band, .0015 if config.task == "support" else .003)
    if config.task not in ("frame", "support"):
        square = slot = 1.
    rail_ok = (abs(pitch) <= tol and abs(rail["pitch"]) <= tol)
    if config.task in ("frame", "support"):
        rail_ok = rail_ok and abs(yaw) <= tol and abs(rail["yaw"]) <= tol and abs(offset) <= slot_band and abs(rail["offset"]) <= slot_band
    if config.task in ("sensor", "aim"):
        rail_ok = True
    need_optical = config.task in ("sensor", "aim", *NEW_TASKS)
    torque_ok = all(metrics[n]["peak_bit_socket_moment_2ms_nm"] <= .4 for n in SCREWS[:2])
    group = "sensor" if config.task in ("sensor", "aim") else "frame"
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
    return dict(task=config.task, task_contract=EIGHT_CONTRACT, success=success,
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
        evaluation_scope=f"{EIGHT_CONTRACT}: own-driver click, 10-tick hands-off retention, helical advance, "
            "M3 0.4 N m limit; task-specific rail, home-flag detection and preserved clamps. "
            "Score 1 on success else 0.95 x continuous progress above settled starting loads.")

"""Task-only construction recipes; existing scene and physical parameters are unchanged.

No simulation runs on import. The ordinary constructor calls these hooks before
exposing tick zero. Prior assembly uses the existing construction torque ramp,
never fabricated loads/clicks, frozen bodies or altered constraints.
"""

import hashlib
import math

from .task import NEW_TASKS, uses_eight

SCREWS = ("sensor_left", "sensor_right", "frame_top", "frame_front")
LOOSE_CHOICES = ("sensor_left", "sensor_right", "frame_front")


def loose_screw(task, seed):
    if task == "front-m4":
        return "frame_front"
    if task == "find-loose":
        digest = hashlib.sha256(f"tighten-find-loose-v1:{seed}".encode()).digest()
        return LOOSE_CHOICES[int.from_bytes(digest[:8], "big") % len(LOOSE_CHOICES)]
    raise ValueError(f"No single-loose preset for {task}")


def pretightened_screws(config):
    return tuple(n for n in SCREWS if n != loose_screw(config.task, config.seed))


def flag_home(config):
    return -.0035 if config.task == "aim" else 0.


def apply_start_coordinates(builder, items, config):
    """Only initial joint positions, respecting the existing helical equality."""
    if config.task not in NEW_TASKS:
        return
    rail = builder.joint_label.index("horizontal_profile_slot_travel")
    q0 = builder.joint_q_start[rail]
    # slide, lift, pitch, yaw: the same nominal level pose as sensor/aim.
    for k in range(4):
        builder.joint_q[q0 + k] = 0.
    tight = pretightened_screws(config)
    for name in SCREWS:
        item = items[name]
        # Existing scene's secured-frame 0.5 mm gap closure minus 2 um.
        advance = .000498 if name in tight else 0.
        builder.joint_q[builder.joint_q_start[item["slide_joint"]]] = advance
        builder.joint_q[builder.joint_q_start[item["turn_joint"]]] = 2 * math.pi * advance / item["pitch_m"]


def build(config, robot_setup="tighten-fr3"):
    """Wrap scene.build, which these presets do not change; append task props last."""
    from .scene import build as build_scene
    result = build_scene(config, robot_setup)
    builder, items, materials, _, _ = result
    apply_start_coordinates(builder, items, config)
    if uses_eight(config) and config.task in ("sensor", "aim", *NEW_TASKS):
        import newton
        import warp as wp
        # Same proven bumper as mixed, moved to aim's original -3.5 mm home.
        # A new prop only: the flag's joints, geometry and contact model stay.
        x = .2655 + flag_home(config) - .020 - .0001 - .004
        shape = builder.add_shape_box(-1, xform=wp.transform((x, -.063, .785), wp.quat_identity()),
            hx=.004, hy=.010, hz=.005, label="flag_home_stop",
            cfg=newton.ModelBuilder.ShapeConfig(mu=.45, density=2700, ke=2.5e7, kd=4.6e4,
                gap=config.contact_gap_m, mu_torsional=0., mu_rolling=0.))
        materials[shape] = "robot_dark"
    return result


def settle_initial(env):
    """Prior assembly of three screws with the standing 30/60% torque ramp.

    This is constructor code, not a policy action or an assisted episode.
    There is no support wrench: the upward M4 starts seated and level as in
    the existing sensor preset. Verify the actual resulting state, then save
    it using the ordinary reset fields. No new runtime is launched here by tests.
    """
    ticks = int(round(env.config.initial_settle_s / env.config.dt))
    if not ticks:
        raise ValueError("The single-loose presets require construction settling")
    tight = pretightened_screws(env.config)
    trace, env.trace_callback = env.trace_callback, None
    base_f = env.control.joint_f.numpy().copy()
    try:
        for k in range(ticks):
            env._assembly_turning = set(tight) if k < .6 * ticks else set()
            force = base_f.copy()
            ramp = min(1., k / (.3 * ticks)) * (k < .6 * ticks)
            for name in tight:
                force[env._screw_dof[name]] = ramp * env.items[name]["limit_nm"]
            env.control.joint_f.assign(force)
            env.step()
    finally:
        env.control.joint_f.assign(base_f)
        env._assembly_turning = set()
        env.trace_callback = trace
    # Fail closed if this start recipe cannot produce its stated
    # conditions. Do not silently publish a different or already solved job.
    bad = [n for n in tight if env._preload[n] < env.minimum_load(n)]
    loose = loose_screw(env.config.task, env.config.seed)
    if bad or env._preload[loose] >= env.minimum_load(loose):
        raise RuntimeError(f"Single-loose preset did not settle: below-floor={bad}, loose={loose}")
    flag_q = int(env.model.joint_q_start.numpy()[env.items["flag_joint"]])
    at_home = abs(float(env.state.joint_q.numpy()[flag_q]) - flag_home(env.config)) <= .0005
    if (abs(env._rail_attitude()[0]) > env.config.rail_level_tolerance_rad
            or not env._optical()["detected"] or not at_home):
        raise RuntimeError("Single-loose preset did not leave the rail level and sensor detecting the home flag")
    env.initial.assign(env.state)
    env.next_state.assign(env.state)
    env._settled_loads = (dict(env._preload), dict(env._thread_load), dict(env._interface))
    env.tick = 0
    env._peak_preload = dict(env._preload)
    env._peak_torque = {n: 0. for n in env.screws}
    env._reset_tracking()
    env._thread_load = dict(env._settled_loads[1])
    env._bit_contacts = {n: 0 for n in env.screws}
    env._retained_ticks = {n: 0 for n in env.screws}
    env._retained_anchor = {n: None for n in env.screws}
    env._bracket_origin = env.state.body_q.numpy()[env.items["bracket"], :3].copy()
    env._reset_free_readings()
    env.evaluation = env._evaluate()

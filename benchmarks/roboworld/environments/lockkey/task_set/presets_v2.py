"""lockkey-tasks-v2 task placements, outside newton15's sources. No physics retuning.

The mid-cycle coordinates are the released recorded snapshots (start_states.json, via presets.snapshot).
These placements, and return-key's key on the mat, are then settled hands-off for 0.5 s and checked
(settle_v2.py) before the task starts at tick zero. Initial velocities are zero.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .catalogue import validate
from .presets import MAT_TOP, MID_CYCLE, _free_coordinates, snapshot

# Entire flat key, including its 11 mm-radius bow, stays inside the work mat. At x=.21 (v1) the settle
# showed the bow's contact with the thin mat box turning into deep spurious contacts near the mat's x=.27
# edge: the key was pushed +x and slid off the mat within 0.5 s. At x=.15 (bow 71 mm from that edge, blade
# tip 97 mm from the chest) it stays within 0.5 mm of its place over 3 s.
MAT_KEY_XY = (.15, -.20)
# Resting pose of the flat key, from its actual collision shapes (the 2.0 mm blade prisms and the 5.0 mm
# bow/neck hull; centre of mass between them): lying flat (+90 deg about x) the key rests on its bow only and
# tips onto its blade tip. Placed level, the tip fell 1.5 mm, struck the mat at about 5 rad/s and launched
# the key 0.36 m across the bench within 0.2 s. Pitched 2.735 deg tip-down about world y with the lowest
# points 1.214 mm below the origin, the neck edge and the blade tip both rest on the mat. The 0.5 s settle
# (settle_v2.py) checks it.
FLAT_KEY_PITCH_DEG = 2.735
FLAT_KEY_ORIGIN_ABOVE_MAT = .001214


def placements(task, table_height=.75):
    """Pure data for the moved free keys; unchanged supports and robot HOME."""
    validate(task)
    if task in MID_CYCLE:
        pose = np.array(snapshot(task)["body_poses"]["key"], dtype=float)
        pose[2] += table_height-.75
        return {"key": pose}
    if task == "return-key":
        position = [*MAT_KEY_XY, table_height+MAT_TOP+FLAT_KEY_ORIGIN_ABOVE_MAT]
        rest = Rotation.from_euler("y", -FLAT_KEY_PITCH_DEG, degrees=True)*Rotation.from_euler("x", 90., degrees=True)
        return {"key": np.r_[position, rest.as_quat()]}
    if task == "which-key":
        return {name: np.array([.16, y, table_height+.055, 0., 0., 0., 1.])
                for name, y in (("key", .16), ("wrong_key", -.16))}
    return {}


def apply_start(env, task):
    """Apply once on a newly initialized/reset native environment, before actions.

    The ordinary release handler sets the latch to precisely its normal released
    state. No fixture actuator, mass, contact, solver or robot setting is copied.
    """
    validate(task)
    if env.tick != 0:
        raise RuntimeError("Task placements can only be applied at reset, before tick zero")
    poses = env.state.body_q.numpy()
    stands = {n: poses[env.bodies[n]].copy() for n in ("key", "wrong_key")}
    moved = placements(task, env.config.table_height)
    if task == "which-key":
        stands = {name: pose.copy() for name, pose in moved.items()}
    record = {"task": task, "kind": "catalogue" if not moved else "constructed_initial_condition",
              "settle_checked": False, "render_checked": False}
    if not moved:
        return stands, record
    q, qd = env.state.joint_q.numpy(), env.state.joint_qd.numpy()
    for name, pose in moved.items():
        _free_coordinates(env, name, pose, q, qd)
    if task in MID_CYCLE:
        # Recorded coordinates only make sense for the recorded mechanism. Keep
        # all old config handling available on the default task; never coerce or
        # silently replace an incompatible configuration here.
        from lockkey.newton15.config import Config
        reference = Config()
        geometry = ("bitting", "plug_length", "bore_clearance", "plug_clearance", "keyway_half_width",
                    "blade_clearance", "root_tare", "ledge_z", "pin_chamfer", "bottom_pin_tip",
                    "insertion_travel", "plug_geometry", "box_geometry", "aligned_casing",
                    "mechanical_latch", "shackle_swivel")
        if any(getattr(env.config, n) != getattr(reference, n) for n in geometry):
            raise ValueError("Recorded lockkey start needs the reference mechanism geometry")
        saved = snapshot(task)
        for name, value in saved["joint_coordinates"].items():
            j = env.mechanism_joints[name]
            a, d = env.builder.joint_q_start[j], env.builder.joint_qd_start[j]
            values = np.atleast_1d(value)
            q[a:a+len(values)], qd[d:d+len(values)] = values, 0.
        record.update(source="task_set/start_states.json", source_tick=saved["tick"], phase=saved["phase"])
    env.state.joint_q.assign(q)
    env.state.joint_qd.assign(qd)
    env.newton.eval_fk(env.model, env.state.joint_q, env.state.joint_qd, env.state)
    env.next_state.assign(env.state)
    # Same state synchronization as the welding start presets.
    env.solver._update_mjc_data(env.solver.mj_data, env.model, env.state)
    env.solver._mujoco.mj_forward(env.solver.mj_model, env.solver.mj_data)
    actual = env.state.body_q.numpy()
    for name, pose in moved.items():
        got = actual[env.bodies[name]]
        if not np.allclose(got[:3], pose[:3], atol=1e-6) or abs(abs(np.dot(got[3:], pose[3:]))-1.) > 1e-6:
            raise RuntimeError(f"{name}: placed pose does not match FK")
    if task in MID_CYCLE:
        for name, raw in snapshot(task)["body_poses"].items():
            expected = np.array(raw, dtype=float)
            expected[2] += env.config.table_height-.75
            got = actual[env.bodies[name]]
            if not np.allclose(got[:3], expected[:3], atol=2e-6) or abs(abs(np.dot(got[3:], expected[3:]))-1.) > 1e-6:
                raise RuntimeError(f"{name}: recorded mechanism coordinates do not match FK")
        # Invokes exactly the native latch transition (no physics step). On a
        # reset suppress trace callbacks: this placement is not a robot action.
        callback = env.trace_callback
        try:
            env.trace_callback = None
            env._after_physics_step()
        finally:
            env.trace_callback = callback
    record["free_key_poses"] = {name: pose.tolist() for name, pose in moved.items()}
    return stands, record

"""lockkey-tasks-v2 task placements, outside newton15's sources. No physics retuning.

The mid-cycle coordinates are the released recorded snapshots (start_states.json, via presets.snapshot).
These placements, and return-key's key on the mat, are then settled hands-off for 0.5 s and checked
(settle_v2.py) before the task starts at tick zero. Initial velocities are zero.

A task's own placements are in its setup.py (tasks/<task>/), as for lockkey-tasks-v1 (presets.py); where this contract
places a key differently (return-key), the setup.py's placements_v2 gives it.
"""

import numpy as np

from .catalogue import task_setup, validate
from .presets import _free_coordinates, mid_cycle, snapshot


def placements(task, table_height=.75):
    """Pure data for the moved free keys; unchanged supports and robot HOME."""
    validate(task)
    if mid_cycle(task):
        pose = np.array(snapshot(task)["body_poses"]["key"], dtype=float)
        pose[2] += table_height-.75
        return {"key": pose}
    folder = task_setup(task)
    own = getattr(folder, "placements_v2", getattr(folder, "placements", None))
    return own(table_height) if own is not None else {}


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
    own = getattr(task_setup(task), "stands", None)
    if own is not None:
        stands = own(moved)
    record = {"task": task, "kind": "catalogue" if not moved else "constructed_initial_condition",
              "settle_checked": False, "render_checked": False}
    if not moved:
        return stands, record
    q, qd = env.state.joint_q.numpy(), env.state.joint_qd.numpy()
    for name, pose in moved.items():
        _free_coordinates(env, name, pose, q, qd)
    if mid_cycle(task):
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
    if mid_cycle(task):
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

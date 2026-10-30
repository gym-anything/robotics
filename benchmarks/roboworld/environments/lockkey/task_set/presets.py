"""Task placements outside newton15's source-capture glob. No physics retuning.

The mid-cycle coordinates are derived from the existing assisted swivel record,
not from running the simulator. They are not yet checked by a hands-off settle,
contact or render run. Initial velocities are zero; the task starts at tick zero.
"""

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .catalogue import validate

MID_CYCLE = ("turn-to-open", "swing-open", "remove-key")
# Entire flat key, including its 11 mm-radius bow, stays inside the work mat.
# At x=.24 the bow would extend beyond x=.27.
MAT_KEY_XY = (.21, -.20)
MAT_TOP = .0016
FLAT_BOW_HALF_THICKNESS = .0025


def snapshot(task):
    data = json.loads(Path(__file__).with_name("start_states.json").read_text())
    return data["snapshots"][task]


def placements(task, table_height=.75):
    """Pure data for the moved free keys; unchanged supports and robot HOME."""
    validate(task)
    if task in MID_CYCLE:
        pose = np.array(snapshot(task)["body_poses"]["key"], dtype=float)
        pose[2] += table_height-.75
        return {"key": pose}
    if task == "return-key":
        position = [*MAT_KEY_XY, table_height+MAT_TOP+FLAT_BOW_HALF_THICKNESS]
        return {"key": np.r_[position, Rotation.from_euler("x", 90., degrees=True).as_quat()]}
    if task == "which-key":
        return {name: np.array([.16, y, table_height+.055, 0., 0., 0., 1.])
                for name, y in (("key", .16), ("wrong_key", -.16))}
    return {}


def _free_coordinates(env, name, pose, q, qd):
    """Use the free joint, not a body-only teleport that FK would overwrite."""
    m = env.model
    children, starts, dofs = m.joint_child.numpy(), m.joint_q_start.numpy(), m.joint_qd_start.numpy()
    matches = [j for j, child in enumerate(children) if int(child) == env.bodies[name]]
    if len(matches) != 1 or int(starts[matches[0]+1]-starts[matches[0]]) != 7:
        raise RuntimeError(f"{name}: expected one seven-coordinate free joint")
    j = matches[0]
    a, d = int(starts[j]), int(dofs[j])
    old = env.state.body_q.numpy()[env.bodies[name]]
    if not np.allclose(q[a:a+3], old[:3], atol=1e-6) or abs(abs(np.dot(q[a+3:a+7], old[3:]))-1.) > 1e-6:
        raise RuntimeError(f"{name}: free joint is not in world coordinates")
    q[a:a+7], qd[d:d+6] = pose, 0.


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

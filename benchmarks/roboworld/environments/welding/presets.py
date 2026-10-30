"""Task start presets: the initial condition a task sets on its built start state before tick 0, disclosed
in the assisted log, as welding/variations.py apply_initial sets a catalogue variation's wire stick-out. The backend
applies both, the preset first (gym_anything_robotics/backends/newton_welding.py, _make_env); a task with a preset
refuses to be evaluated without it (task.Evaluator.evaluate). A preset places a part or sets the source's set points
through the robot's own I/O; it changes no physical model, so this module is not hashed with the physics
(env.SOURCE_FILES) and the cached start states stay valid.

  set-up-gun      the tacked start with the source left on the cell's documented short-circuit job (params.SCHEDULES
                  sc_0p9_C25_175A: 6.4 m/min at 20 V) and the 1/4 in of wire that job leaves (task.SETUP_LEFT_*); the
                  gas valve stays closed, as the tacked start leaves it. A variation's own stick-out replaces the wire.
  set-up-and-weld the same source set points only; the wire is the tacked start's own (or the variation's), so the
                  recorded aim_tacks program, which commands the procedure's set points itself before it strikes
                  (scripted.WeldProgram.weld), runs from it as it ran from the tacked start.
  fit-up-bracket  the loose start with the bracket stood off its layout on the plate: shifted toward the weld side and
                  yawed about its vertical axis (task.fitup_displacement, per world seed), then settled to rest
                  hands-off with the start states' own rest gate (env.settle_to_rest); a protective stop while it
                  settles fails the build (env.WeldingEnv._clear_world_logs).
"""

import math

import numpy as np

from . import params as PR
from .task import START_PRESET, SETUP_LEFT_JOB, SETUP_LEFT_STICKOUT, fitup_displacement, fitup_errors


def apply_start_preset(env):
    """Set the task's preset on a freshly built WeldingEnv (before tick 0) and take its initial checkpoint again, so a
    reset returns to it. Returns the assisted-log record, or None for a task without a preset."""
    task = env.config.task
    if task not in START_PRESET:
        return None
    if getattr(env, "start_preset", None) is not None:
        raise RuntimeError(f"{task}: its start preset is already applied")
    if env.steps:
        raise RuntimeError(f"{task}: a start preset is an initial condition; apply it before tick 0")
    record = {"set-up-gun": _source_left_on_last_job, "fit-up-bracket": _bracket_off_layout,
              "set-up-and-weld": lambda env_: _source_left_on_last_job(env_, wire=False)}[task](env)
    record = dict(preset=START_PRESET[task], task=task, **record)
    env.assisted_log.append(record)
    env.start_preset = record
    env.start_state = f"{env.start_state}; {START_PRESET[task]}"
    env.initial = env.checkpoint()
    return record


def _source_left_on_last_job(env, wire=True):
    w = env.world
    src = w.process.src
    st = src.st
    if st.arc or st.on_cmd or st.inch:
        raise RuntimeError(f"{env.config.task}'s preset is an initial condition: the source must be idle")
    if wire and math.isfinite(st.d_tw) and SETUP_LEFT_STICKOUT >= st.d_tw:
        raise ValueError(f"a {SETUP_LEFT_STICKOUT*1e3:.2f} mm stick-out would reach the metal {st.d_tw*1e3:.2f} mm from "
                         "the contact tip at the start pose")
    job = PR.v(PR.SCHEDULES)[SETUP_LEFT_JOB]
    before = dict(wire_feed_m_min=float(src.s.wfs_m_min), voltage_v=float(src.s.voltage_v), stickout_m=float(st.l_s),
                  gas_on=bool(st.gas_on))
    w.tool_io(wire_feed_m_min=job["wfs_m_min"], voltage_v=job["voltage_v"])
    if wire:
        st.l_s = st.l_s0 = float(SETUP_LEFT_STICKOUT)    # an initial condition: the wire balance starts here
    env._clear_world_logs()
    what = "source set points" + (" and wire stick-out" if wire else "")
    return dict(initial_condition=what + " left by the last job", job=SETUP_LEFT_JOB,
                before=before, set=dict(wire_feed_m_min=float(src.s.wfs_m_min), voltage_v=float(src.s.voltage_v),
                                        stickout_m=float(st.l_s), gas_on=bool(st.gas_on)))


def _bracket_off_layout(env):
    from .env import REST_LIMIT_MM, SETTLE_S, settle_to_rest
    w = env.world
    sc = w.scene
    if w.bond.active:
        raise RuntimeError("fit-up-bracket starts from the loose bracket; this one is bonded")
    shift, yaw = fitup_displacement(env.config.seed)
    pb, Rb = w.body_pose(sc.bodies["base_plate"])
    pk, Rk = w.body_pose(sc.bodies["bracket"])
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    _place_free_body(w, sc.bodies["bracket"], pk + Rb @ np.array([0., shift, 0.]), Rb @ Rz @ Rb.T @ Rk)
    w.settle(SETTLE_S)
    sim_s, windows = settle_to_rest(w)
    if windows[-1] >= REST_LIMIT_MM:
        raise RuntimeError(f"fit-up-bracket's start did not come to rest hands-off (last window {windows[-1]:.4f} mm "
                           f">= {REST_LIMIT_MM:.4f} mm)")
    grid_events = [e for e in w.events if "bracket" in str(e.get("event", ""))]
    env._clear_world_logs()
    pb, Rb = w.body_pose(sc.bodies["base_plate"])
    pk, Rk = w.body_pose(sc.bodies["bracket"])
    return dict(initial_condition="bracket stood off its layout", seed=int(env.config.seed),
                shift_toward_weld_side_mm=round(shift*1e3, 3), yaw_deg=round(yaw, 3),
                settle=dict(sim_s=round(sim_s + SETTLE_S, 3), window_motion_mm=[round(m, 5) for m in windows]),
                grid_events=grid_events,
                fitup_errors=fitup_errors(pb, Rb, pk, Rk, sc.weld_origin_in_base, sc.weld_origin_in_bracket,
                                          sc.bracket_rel_nominal))


def _place_free_body(w, body, p, R):
    """Put a free body at world pose (p, R) at rest, the way env.place_robot_state puts robot coordinates: the
    coordinates, forward kinematics, the next state and MuJoCo's data. The free joint's coordinates must read as the
    body's pose (position, then the xyzw quaternion of body_q) before and after, else nothing is trusted: it raises."""
    from scipy.spatial.transform import Rotation
    m = w.model
    child, qs, ds = m.joint_child.numpy(), m.joint_q_start.numpy(), m.joint_qd_start.numpy()
    joints = [j for j in range(m.joint_count) if int(child[j]) == body]
    if len(joints) != 1 or int(qs[joints[0] + 1]) - int(qs[joints[0]]) != 7:
        raise RuntimeError(f"body {body} is not a free body (joints {joints})")
    i0, d0 = int(qs[joints[0]]), int(ds[joints[0]])

    def same_pose(coords, pose):
        return bool(np.allclose(coords[:3], pose[:3], atol=1e-6)
                    and abs(abs(float(np.dot(coords[3:7], pose[3:7]))) - 1.) < 1e-6)
    q, qd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
    if not same_pose(q[i0:i0 + 7].astype(float), w.state.body_q.numpy()[body].astype(float)):
        raise RuntimeError("the free joint's coordinates do not read as the body's pose: not moved")
    target = np.concatenate([np.asarray(p, float), Rotation.from_matrix(np.asarray(R, float)).as_quat()])
    q[i0:i0 + 7] = target
    qd[d0:d0 + 6] = 0.
    w.state.joint_q.assign(q)
    w.state.joint_qd.assign(qd)
    w.newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
    for a in ("body_q", "body_qd", "joint_q", "joint_qd"):
        getattr(w.next_state, a).assign(getattr(w.state, a).numpy())
    w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
    w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
    w._tip_prev = None
    if not same_pose(target, w.state.body_q.numpy()[body].astype(float)):
        raise RuntimeError("the body did not take the pose it was placed at")

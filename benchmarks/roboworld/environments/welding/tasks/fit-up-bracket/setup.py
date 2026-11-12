"""Start of fit-up-bracket: the loose start (env.py: the bracket standing unattached on the clamped base plate) with
the bracket stood off its layout on the plate: shifted toward the weld side and yawed about its vertical axis
(fitup_displacement, per world seed), then settled to rest hands-off with the start states' own rest gate
(env.settle_to_rest); a protective stop while it settles fails the build (env.WeldingEnv._clear_world_logs).

START_STATE is the start state env.WeldingEnv builds for the task (task.START_STATE) and PRESET names its preset
(task.START_PRESET); setup(env) receives the freshly built WeldingEnv before tick 0 (presets.apply_start_preset),
places the bracket and returns the preset's assisted-log record."""
import math

import numpy as np

from welding.task import fitup_errors

START_STATE = "loose"
PRESET = "bracket stood off its layout"
# the bracket stood off its layout: FITUP_SHIFT toward the weld side (+y, onto the open plate between the bracket and
# the robots; the clamps hold the other edge) and yawed about its own vertical axis, seed 0 the design's 25 mm / 6 deg,
# other seeds drawn in the ranges [authored]
FITUP_SHIFT_MM = (15., 30.)
FITUP_YAW_DEG = (3., 8.)
FITUP_SEED0 = (25., 6.)


def fitup_displacement(seed):
    """(shift m toward +y, yaw deg) of fit-up-bracket's start (welding/presets.py) for a world seed (Config.seed: the
    legacy seed or a variation's config_seed); deterministic, seed 0 the design's 25 mm / 6 deg."""
    if int(seed) == 0:
        return FITUP_SEED0[0]*1e-3, FITUP_SEED0[1]
    rng = np.random.default_rng(271828 + int(seed))
    shift = float(rng.uniform(*FITUP_SHIFT_MM))*1e-3
    yaw = float(rng.uniform(*FITUP_YAW_DEG))*(1. if rng.uniform() < .5 else -1.)
    return shift, yaw


def setup(env):
    from welding.env import REST_LIMIT_MM, SETTLE_S, settle_to_rest
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

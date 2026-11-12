"""End-state tool seating/support and robot release, without simulator imports. (Cap seating, recap-and-tidy's own
check, is in its verifier.py.)"""
from functools import lru_cache
import math
import numpy as np
from scipy.spatial.transform import Rotation

from whiteboard.geometry import tools


def read_state(env):
    q, qd = env.state_0.body_q.numpy(), env.state_0.body_qd.numpy()
    pelvis = next(i for i, label in enumerate(env.model.body_label) if label.endswith("/pelvis"))
    m, d = env.solver.mj_model, env.solver.mj_data
    # Use the native-to-Newton mapping, not an assumed geom/body index identity.
    mapping = env.solver.mjc_body_to_newton.numpy()[0]
    names = {body: name for name, body in env.body_ids.items()}
    robot = set(env.model.shape_body.numpy()[list(env.robot_shapes)].tolist())
    touching = set()
    for c in d.contact[:d.ncon]:
        if c.efc_address < 0 or c.dist > .001:
            continue
        a, b = (int(mapping[int(m.geom_bodyid[int(g)])]) for g in c.geom)
        if a in robot and b in names:
            touching.add(names[b])
        if b in robot and a in names:
            touching.add(names[a])
    return {"poses": {n: q[i] for n, i in env.body_ids.items()},
            "velocities": {n: qd[i] for n, i in env.body_ids.items()},
            "touching": touching, "pelvis_z": float(q[pelvis, 2])}


@lru_cache(maxsize=8)
def tool_geometry(config):
    return {o.name: o for o in tools(config)}


def bounds(obj, pose):
    """Exact primitive axis projections (including clip/nib), not centre-only containment."""
    rotation = Rotation.from_quat(pose[3:])
    lows, highs = [], []
    for part in obj.parts:
        centre = pose[:3] + rotation.apply(part.position)
        axes = (rotation * Rotation.from_quat(part.rotation)).as_matrix()
        if part.kind == "box":
            extent = np.abs(axes) @ (np.asarray(part.size) / 2)
        elif part.kind == "cylinder":
            radius, length = part.size
            extent = radius * np.linalg.norm(axes[:, :2], axis=1) + length / 2 * np.abs(axes[:, 2])
        else:
            raise ValueError("Unexpected tool geometry")
        lows.append(centre - extent)
        highs.append(centre + extent)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def tray_measure(config, name, pose, velocity, touching):
    low, high = bounds(tool_geometry(config)[name], np.asarray(pose))
    b = config.board
    floor = b.bottom - .015 + .0015
    # Tray's physical inner faces; whole footprint must fit, lowest point on floor.
    lower = np.array([-b.width/2 + .004, b.face_y - .003])
    upper = np.array([b.width/2 - .004, b.face_y + .0755])
    outside = float(np.maximum(lower - low[:2], 0).sum() + np.maximum(high[:2] - upper, 0).sum())
    support_error = abs(float(low[2]) - floor)
    orientation = Rotation.from_quat(pose[3:]).as_matrix()
    # Eraser's broad face horizontal; marker/cap long axes horizontal.
    tilt = math.acos(np.clip(abs(orientation[2, 2]), 0, 1)) if name == "eraser" else math.asin(np.clip(abs(orientation[2, 2]), 0, 1))
    # This pinned Newton environment uses [linear, angular] (energy.Ledger).
    linear, angular = np.linalg.norm(velocity[:3]), np.linalg.norm(velocity[3:])
    excess = outside + max(0., support_error - .003)
    quality = 1 / (1 + excess / .03 + max(0., tilt - math.radians(10)) / .3
                   + max(0., linear - .02) / .1 + max(0., angular - .1))
    return {"pass": bool(outside <= 1e-9 and support_error <= .003 and tilt <= math.radians(10)
                         and linear <= .02 and angular <= .1 and name not in touching),
            "quality": float(quality), "outside_m": outside, "support_error_m": support_error,
            "tilt_deg": math.degrees(tilt), "linear_m_s": float(linear), "angular_rad_s": float(angular),
            "robot_contact": name in touching}


def mechanics(config, state, names):
    trays = {n: tray_measure(config, n, state["poses"][n], state["velocities"][n], state["touching"]) for n in names}
    standing = state["pelvis_z"] > .45
    quality = float(np.clip((state["pelvis_z"] - .30) / .15, 0, 1))
    quality *= min((r["quality"] for r in trays.values()), default=1.)
    return {"standing": standing, "returned": all(r["pass"] for r in trays.values())}, quality, trays

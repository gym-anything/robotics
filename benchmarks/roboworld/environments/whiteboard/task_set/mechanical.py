"""End-state tool seating/support and robot release, without simulator imports."""
from functools import lru_cache
import math
import numpy as np
from scipy.spatial.transform import Rotation

from whiteboard.config import COLORS
from whiteboard.geometry import tools
from .presets import CAP_SEATED_M, CAP_START_OFFSET_M, OPEN_COLORS, tool_poses


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


def seating(barrel, cap):
    rb, rc = Rotation.from_quat(barrel[3:]), Rotation.from_quat(cap[3:])
    local = rb.inv().apply(np.asarray(cap[:3]) - barrel[:3])
    axial = abs(float(local[2]) - CAP_SEATED_M)
    lateral = float(np.linalg.norm(local[:2]))
    angle = math.acos(np.clip(float(np.dot(rb.apply([0, 0, 1]), rc.apply([0, 0, 1]))), -1, 1))
    axial_credit = np.clip(1 - max(0., axial - .0015) / (CAP_START_OFFSET_M - .0015), 0, 1)
    quality = axial_credit / (1 + max(0., lateral - .0015) / .005 + max(0., angle - math.radians(5)) / .3)
    return {"pass": bool(axial <= .0015 and lateral <= .0015 and angle <= math.radians(5)),
            "quality": float(quality), "axial_error_m": axial, "radial_error_m": lateral,
            "angle_deg": math.degrees(angle)}


def mechanics(config, state, names):
    trays = {n: tray_measure(config, n, state["poses"][n], state["velocities"][n], state["touching"]) for n in names}
    standing = state["pelvis_z"] > .45
    quality = float(np.clip((state["pelvis_z"] - .30) / .15, 0, 1))
    quality *= min((r["quality"] for r in trays.values()), default=1.)
    return {"standing": standing, "returned": all(r["pass"] for r in trays.values())}, quality, trays


def recap(config, task, state):
    poses = state["poses"]
    names = tuple(n for c in COLORS for n in (f"marker_{c}", f"cap_{c}"))
    checks, quality, trays = mechanics(config, state, names)
    seats = {c: seating(poses[f"marker_{c}"], poses[f"cap_{c}"]) for c in COLORS}
    checks["all_caps_seated"] = all(s["pass"] for s in seats.values())
    # Only the three authored open caps earn progress. Others are collateral.
    # Subtract each authored start credit (including roundoff) to make idle exactly zero.
    start = tool_poses(config, task)
    start = {n: p.astype(np.asarray(poses[n]).dtype) for n, p in start.items()}
    credits = []
    for c in OPEN_COLORS:
        s0 = seating(start[f"marker_{c}"], start[f"cap_{c}"])["quality"]
        credits.append(float(np.clip((seats[c]["quality"] - s0) / (1 - s0), 0, 1)))
    collateral = min(seats[c]["quality"] for c in COLORS if c not in OPEN_COLORS)
    return checks, float(np.mean(credits)) * collateral * quality, {"seating": seats, "tray": trays}

"""Outcome of recap-and-tidy: every cap seated on its marker (the three open ones put back, the others left on), all
six capped markers resting in the tray, the board clean and the robot standing clear of the markers.

verify(task, ink, config, state) gives the verdict of a recap-and-tidy preset (task_set/verifier.py's evaluate): ink
is the board's InkSurface, config the episode's Config, state the body state the task-set backends read
(mechanical.read_state: tool poses and velocities, robot contacts, pelvis height; None raises ValueError). It returns
a dict with score, success, progress, checks (all_caps_seated, board_clean, standing, returned), the visible area and
the seating and tray readings. seating(barrel, cap) reads one cap on its marker; recap(config, task, state) gives the
mechanical checks, progress and readings; OPEN_COLORS and CAP_START_OFFSET_M are the start's (its setup.py).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from whiteboard.config import COLORS
from whiteboard.task_set.mechanical import mechanics
from whiteboard.task_set.presets import CAP_SEATED_M, tool_poses
from whiteboard.task_set.verifier import AREA_ALLOWANCE, board, clip, require_state, result
from whiteboard.tasks import task_setup

OPEN_COLORS = task_setup("recap-and-tidy").OPEN_COLORS
CAP_START_OFFSET_M = task_setup("recap-and-tidy").CAP_START_OFFSET_M


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


def verify(task, ink, config, state=None):
    colors, weight, start, target, texel = board(task, ink)
    require_state(state)
    checks, progress, detail = recap(config, task, state)
    area = float(weight.sum()) * texel
    checks["board_clean"] = area <= AREA_ALLOWANCE
    clean = clip(1 - max(0., area - AREA_ALLOWANCE) / .001)
    return result(task, checks, progress * clean, visible_m2=area, **detail)

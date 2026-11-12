"""Outcome of take-knife: the fillet knife out of the block, held still in a hand for 3 s, edge down above the board,
the blade never touching a robot link.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator: task.Evaluator
for the released checks (released_checks: the right hand's inserts hold the knife), or the task-set evaluator of
filleting-tasks-v1 or -v2 (task_set_evaluator.py, task_set_v2_evaluator.py), which grades with _take_knife_v1 or
_take_knife_v2. monitor(e) records what those gradings read after every tick of a task-set episode."""
import math
from collections import deque

import numpy as np

from filleting import layout as LY
from filleting.task import TIME_LIMIT_S, check
from filleting.task_set_catalogue import CONTRACT as V1
from filleting.task_set_progress import angle, clip, knife_return_geometry, ramp
from filleting.task_set_v2_catalogue import CONTRACT as V2


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(released_checks)
    return e.evaluate_graded({V1: _take_knife_v1, V2: _take_knife_v2}[e.contract])


def released_checks(e):
    """A0 take-knife: held by the right hand's inserts alone for 3 s, seated in the channel (axis
    within 10 deg, lateral offset within the channel), slip under 2 mm over that time, edge down above the board,
    the blade never touched a robot link."""
    env, w = e.env, e.env.world
    from filleting.scene import INSERT, HANDLE_H
    pinch = w.pinch_forces()
    held_for = 0. if env.knife_held_since is None else env.sim_time - env.knife_held_since
    right = min(pinch["right"]) > .5
    left_touch = max(pinch.get("left", [0., 0.])) > 0. or env.blade_left_hand_contacts > 0
    checks = dict([check("held_by_right_hand_s", held_for, 3., "s of continuous right-hand pinch",
                         right and held_for >= 3.)])
    checks.update([check("not_touched_by_left_hand", float(left_touch), 0., "bool", not left_touch)])
    off, ang = w.knife_in_grip()
    lateral = float(np.hypot(off[1], off[2]))
    # the channel admits the handle's 32 mm height with 0.5 mm clearance each side: a seated handle's centre line
    # lies within the lips (lateral <= lip height)
    seat_tol = INSERT["lip_height"]
    checks.update([check("axis_in_channel_deg", ang, 10., "deg", right and ang <= 10.),
                   check("seated_lateral_offset_mm", lateral*1e3, seat_tol*1e3, "mm", right and lateral <= seat_tol)])
    hist = [h for h in env.grip_history if h[0] >= env.sim_time - 3. - 1e-9]
    slip = max((float(np.linalg.norm(h[1] - hist[-1][1])) for h in hist), default=None) if hist else None
    checks.update([check("slip_mm_over_3s", None if slip is None else slip*1e3, 2., "mm",
                         slip is not None and held_for >= 3. and slip <= 2e-3)])
    # edge down above the board: the knife frame's edge->spine axis (handle z) points up within 30 deg, and every
    # blade segment is over the board and above its surface
    from filleting.workpiece import blade_segment_poses
    _, Rk = w.body_pose("knife_handle")
    tilt = math.degrees(math.acos(float(np.clip(Rk[2, 2], -1, 1))))
    segs = [p for _, p in blade_segment_poses(w)]
    over = all(abs(p[0] - LY.BOARD["center"][0]) <= LY.BOARD["size"][0]/2 and
               abs(p[1] - LY.BOARD["center"][1]) <= LY.BOARD["size"][1]/2 and p[2] > LY.BOARD_TOP_Z + .01 for p in segs)
    checks.update([check("edge_down_deg", tilt, 30., "deg from vertical", tilt <= 30.),
                   check("blade_over_board", float(over), 1., "bool", over),
                   check("blade_never_touched_robot", env.blade_robot_contacts, 0, "ticks with contact",
                         env.blade_robot_contacts == 0)])
    checks["time"] = check("time", env.sim_time, TIME_LIMIT_S["A0"], "s", env.sim_time <= TIME_LIMIT_S["A0"])[1]
    return checks, dict(pinch_n=pinch, grasp_offset_mm=(np.asarray(off)*1e3).round(2).tolist(),
                        knife_mode=w.knife_mode, knife_start=w.knife_start)


def monitor(e):
    """After every tick of a task-set episode: the knife handle's pose (e.record_knife) and the knife's pose in each
    pinching hand over the trailing 3 s (e.hand_history)."""
    env, w = e.env, e.env.world
    p, R = e.record_knife()
    # take-knife progress: the knife's pose in each pinching hand over the trailing 3 s, whichever hand it is
    # (filleting-tasks-v2's success reads the original right-hand records, env.knife_held_since and env.grip_history)
    pinch = w.pinch_forces()
    for arm in w.rig.instances:
        hist = e.hand_history.setdefault(arm, dict(since=None, poses=deque()))
        if min(pinch.get(arm, (0., 0.))) <= .5:
            hist["since"] = None
            hist["poses"].clear()
            continue
        if hist["since"] is None:
            hist["since"] = env.sim_time
        hand = w.kin.forward(arm)
        relp, relR = (p - hand[:3, 3]) @ hand[:3, :3], hand[:3, :3].T @ R
        hist["poses"].append((env.sim_time, relp, relR))
        while len(hist["poses"]) > 1 and hist["poses"][1][0] < env.sim_time - 3.:
            hist["poses"].popleft()


def _take_knife_v1(e):
    from filleting.scene import INSERT
    from filleting.workpiece import blade_segment_poses
    w, env = e.env.world, e.env
    heel, R, centres = e.knife_pose()
    edge = np.array([p for _, p in blade_segment_poses(w)])
    _, handle_R = w.body_pose("knife_handle")
    tilt = angle(handle_R[:, 2], np.array([0., 0., 1.]))
    over = e.in_footprint(edge, LY.BOARD) & (edge[:, 2] > LY.BOARD_TOP_Z + .01)
    # Distance-to-board credit is continuous, rather than counting segments/checks passed.
    dx = np.maximum(np.abs(edge[:, 0] - LY.BOARD["center"][0]) - LY.BOARD["size"][0]/2, 0)
    dy = np.maximum(np.abs(edge[:, 1] - LY.BOARD["center"][1]) - LY.BOARD["size"][1]/2, 0)
    dz = np.maximum(LY.BOARD_TOP_Z + .01 - edge[:, 2], 0)
    board = float(np.mean(np.clip(1 - np.sqrt(dx*dx + dy*dy + dz*dz)/.15, 0., 1.)))
    target, target_R = w.scene.block_knife_pose
    slot = knife_return_geometry(heel, R[:, 0], centres, target, target_R, w.scene.blade.L, w.scene.blade.H)
    out = not slot["inside"]
    rows = []
    for arm, h in e.hand_history.items():
        if h["since"] is None or not h["poses"]:
            continue
        poses = h["poses"]
        held = env.sim_time - h["since"]
        slip = max(np.linalg.norm(q[1] - poses[-1][1]) for q in poses)
        angular_slip = max(angle(q[2][:, 0], poses[-1][2][:, 0]) for q in poses)
        if arm == "right":
            offset, axis = w.knife_in_grip()
            lateral = float(np.hypot(offset[1], offset[2]))
            seated = axis <= 10. and lateral <= INSERT["lip_height"]
            seat = ramp(axis, 10., 30.)*ramp(lateral, INSERT["lip_height"], 2*INSERT["lip_height"])
        else:
            # No knife insert on the other hand: a stable physical pinch is sufficient.
            seated = angular_slip <= 10.
            seat = ramp(angular_slip, 10., 30.)
        stable = seat*ramp(slip, .002, .010)
        progress = (clip(held/3.) + stable + ramp(tilt, 30., 90.)*board)/3. if out else 0.
        rows.append((held >= 3. and seated and slip <= .002 and out, progress, arm, held, slip))
    best = max(rows, key=lambda r: (r[0], r[1]), default=(False, 0., None, 0., None))
    checks = dict([check("stable_hand_hold", best[3], 3., "s; <=2 mm slip, <=10 deg axis", best[0]),
                   check("edge_down_deg", tilt, 30., "deg", tilt <= 30.),
                   check("blade_over_board", float(over.all()), 1., "bool; >10 mm above board", over.all()),
                   check("blade_never_touched_robot", env.blade_robot_contacts, 0., "ticks", env.blade_robot_contacts == 0)])
    return checks, dict(holding_hand=best[2], slip_m=best[4], out_of_block=out), best[1]*(.5 if env.blade_robot_contacts else 1.)


def _take_knife_v2(e):
    """Success is the original A0 verdict, unchanged (released_checks; the spec's "current _eval_A0"): the right
    hand's pinch held 3 s, seated in its inserts (axis <= 10 deg, lateral offset within the lips), slip <= 2 mm,
    the left hand not touching the knife, the edge down over the board, the blade never touching a robot link.

    Progress measures the knife, whichever hand holds it (recipe rule 4): mean(H, S, E) for each hand pinching the
    knife outside the block, best hand counted. H = min(held / 3 s, 1); S = how steady the knife sits in that grip
    (seating x slip ramps over the trailing 3 s; the right hand's seating is its insert channel, a hand without
    inserts is seated when the knife's axis turns <= 10 deg in it); E = edge-down x board proximity. A left-hand
    hold can therefore earn progress, never success. The untouched start (knife in the block, no pinch) scores 0.
    Blade-robot contact halves progress."""
    from filleting.scene import INSERT
    from filleting.workpiece import blade_segment_poses
    w, env = e.env.world, e.env
    checks, legacy = released_checks(e)
    checks.pop("time")                       # evaluate() adds the same 30 s limit (LIMITS["A0"])
    heel, R, centres = e.knife_pose()
    edge = np.array([p for _, p in blade_segment_poses(w)])
    tilt = checks["edge_down_deg"]["value"]
    # Distance-to-board credit is continuous, rather than counting segments/checks passed.
    dx = np.maximum(np.abs(edge[:, 0] - LY.BOARD["center"][0]) - LY.BOARD["size"][0]/2, 0)
    dy = np.maximum(np.abs(edge[:, 1] - LY.BOARD["center"][1]) - LY.BOARD["size"][1]/2, 0)
    dz = np.maximum(LY.BOARD_TOP_Z + .01 - edge[:, 2], 0)
    board = float(np.mean(np.clip(1 - np.sqrt(dx*dx + dy*dy + dz*dz)/.15, 0., 1.)))
    target, target_R = w.scene.block_knife_pose
    slot = knife_return_geometry(heel, R[:, 0], centres, target, target_R, w.scene.blade.L, w.scene.blade.H)
    out = not slot["inside"]
    rows = []
    for arm, h in e.hand_history.items():
        if h["since"] is None or not h["poses"] or not out:
            continue
        poses = h["poses"]
        held = env.sim_time - h["since"]
        slip = max(np.linalg.norm(q[1] - poses[-1][1]) for q in poses)
        if arm == "right":
            offset, axis = w.knife_in_grip()
            lateral = float(np.hypot(offset[1], offset[2]))
            seat = ramp(axis, 10., 30.)*ramp(lateral, INSERT["lip_height"], 2*INSERT["lip_height"])
        else:
            seat = ramp(max(angle(q[2][:, 0], poses[-1][2][:, 0]) for q in poses), 10., 30.)
        stable = seat*ramp(slip, .002, .010)
        rows.append(((clip(held/3.) + stable + ramp(tilt, 30., 90.)*board)/3., arm, held, slip))
    progress, hand, held, slip = max(rows, key=lambda r: r[0], default=(0., None, 0., None))
    if env.blade_robot_contacts:
        progress *= .5
    legacy.update(out_of_block=out, board_proximity=board, progress_hand=hand, progress_held_s=held,
                  progress_slip_m=slip)
    return checks, legacy, progress

"""Outcome of put-away-knife: the fillet knife back in its slot in the knife block (heel within 10 mm, axis within
5 deg), released and at rest, the fish not cut, the blade never touching a robot link.

verify(e) evaluates an episode and returns its gar.evaluation.v1 result; e is the episode's evaluator. The task is
selected under filleting-tasks-v1 or -v2 (task_set_evaluator.py, task_set_v2_evaluator.py), which grade it with
_put_away; the released checks have no evaluator for it. start(e) records the grading's start when the evaluator is
built (the knife tip's distance from its place in the slot and the return progress there), monitor(e) the knife's
pose after every tick."""
import numpy as np

from filleting import layout as LY
from filleting.task import check, near_boxes
from filleting.task_set_progress import clip, gained, knife_return_geometry, ramp


def verify(e):
    if e.contract is None:
        return e.evaluate_checks(None)
    return e.evaluate_graded(_put_away)


def start(e):
    from filleting.workpiece import blade_segment_poses
    w = e.env.world
    heel, R, segs = e.knife_pose()
    target, target_R = w.scene.block_knife_pose
    last_R, last_p = blade_segment_poses(w)[-1]
    e.start_distance = float(np.linalg.norm(last_p + last_R[:, 0]*w.scene.blade.seg_len/2 -
                                            (target + target_R[:, 0]*w.scene.blade.L)))
    e.return_start = _return_raw(e, heel, R, segs)[0]


def monitor(e):
    e.record_knife()


def _return_raw(e, heel, R, centres):
    from filleting.workpiece import blade_segment_poses
    w = e.env.world
    target, target_R = w.scene.block_knife_pose
    last_R, last_p = blade_segment_poses(w)[-1]
    tip = last_p + last_R[:, 0]*w.scene.blade.seg_len/2
    geom = knife_return_geometry(heel, tip - heel, centres, target, target_R, w.scene.blade.L, w.scene.blade.H)
    distance = float(np.linalg.norm(tip - (target + target_R[:, 0]*w.scene.blade.L)))
    approach = clip(1 - distance/max(e.start_distance, 1e-12))
    raw = .3*approach + .7*geom["insertion"]*ramp(geom["axis_error"], 5., 30.)
    return raw, geom


def _put_away(e):
    w, env = e.env.world, e.env
    heel, R, centres = e.knife_pose()
    raw, geom = _return_raw(e, heel, R, centres)
    p, _ = w.body_pose("knife_handle")
    history = [h for h in e.knife_history if h[0] >= env.sim_time - 1. - 1e-9]
    motion = max((float(np.linalg.norm(h[1] - p)) for h in history), default=None)
    window = bool(history and history[0][0] <= env.sim_time - 1. + 1e-9)
    pinch = w.pinch_forces()
    robot_force, _ = w._knife_contacts()
    released = max((max(f) for f in pinch.values()), default=0.) <= .5 and robot_force <= .2
    # Also inspect free-knife and blade geometry against the robot's tissue colliders.
    released &= not near_boxes(w, env._robot_boxes, np.vstack((centres, p)), margin=.002).any()
    intact = not (set(w.cut.cut_faces) - e.initial_faces)
    above_bench = bool(np.min(np.vstack((centres, p))[:, 2]) >= LY.TABLE["top_z"] - .005)
    checks = dict([check("heel_position_mm", geom["heel_error"]*1000, 10., "mm", geom["heel_error"] <= .010),
                   check("slot_axis_deg", geom["axis_error"], 5., "deg", geom["axis_error"] <= 5.),
                   check("blade_in_slot", float(geom["inside"]), 1., "bool; finite slot", geom["inside"]),
                   check("released", float(released), 1., "bool", released),
                   check("rest_mm", None if motion is None else 1000*motion, 1., "mm over trailing 1 s",
                         window and motion <= .001),
                   check("workpiece_uncut", len(set(w.cut.cut_faces) - e.initial_faces), 0., "new faces", intact),
                   check("blade_never_touched_robot", env.blade_robot_contacts, 0., "ticks", env.blade_robot_contacts == 0)])
    progress = gained(raw, e.return_start) if intact and above_bench else 0.
    if env.blade_robot_contacts:
        progress *= .5
    return checks, dict(**geom, robot_contact_n=robot_force), progress

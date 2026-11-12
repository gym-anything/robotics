"""Outcome of unwrap-gift: the gift box's outer surface bare of wrapping paper (tearing the paper is allowed), the box
still closed by its lid and upright, and no robot contact through the hands-off window.

verify(e) gets the episode's evaluator (task.Evaluator, or task_set_evaluator / task_set_v2_evaluator.Evaluator for a
selection that names a contract) and returns its outcome: dict(success, score, checks, metrics), plus contract under
the task-set evaluators. It grades through e.evaluate_task: the released evaluation (legacy selections) adds released's
checks; the gift-bags-tasks-v1 and -v2 contracts take measure's checks and metrics and score progress. Both judge the
paper with task.Evaluator._eval_unwrap, which runs box_checks for the box."""


def box_checks(e, checks, metrics):
    from gift_bags.task import LID_CLOSED_OVERLAP_MM, _tilt_deg
    ipc = e.world.ipc
    g = e.geo["box"]
    pb, Rb = ipc.body_pose(e.body["gift_base"])
    pl, Rl = ipc.body_pose(e.body["gift_lid"])
    base_org = pb - Rb @ ipc.bodies[e.body["gift_base"]].render["com_offset"]
    lid_org = pl - Rl @ ipc.bodies[e.body["gift_lid"]].render["com_offset"]
    overlap = g["base_h"] - float((Rb.T @ (lid_org - base_org))[2])
    metrics["lid_overlap_mm"] = 1e3*overlap
    checks["box_closed"] = overlap*1e3 >= LID_CLOSED_OVERLAP_MM
    metrics["box_tilt_deg"] = _tilt_deg(Rb)
    checks["box_upright"] = metrics["box_tilt_deg"] <= 15.


def released(e, checks, metrics, rows):
    e._eval_unwrap(checks, metrics, box_checks)


def measure(e, c, m, rows):
    e._eval_unwrap(c, m, box_checks)


def progress(e, S, m):
    return S.unwrap(m, e.start_metrics)


def verify(e):
    return e.evaluate_task(released, progress_of=progress)

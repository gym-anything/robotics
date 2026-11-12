"""Outcome of put-lid-on: the lid back on the box's base (it fits over all four upper corners of the base and overlaps
it by at least task.LID_CLOSED_OVERLAP_MM), the base upright on the table, the photo frame still inside it, and no
robot contact through the hands-off window.

verify(e) gets the episode's evaluator (task_set_evaluator or task_set_v2_evaluator.Evaluator: put-lid-on is a task of
the task-set contracts, on lift-lid's world) and returns its outcome: dict(success, score, checks, metrics, contract),
plus task under gift-bags-tasks-v2. It grades through e.evaluate_task with measure's checks and metrics (lift-lid's
lid checks, its lid-off checks replaced by put-lid-on's) and progress."""


def measure(e, c, m, rows):
    from gift_bags import task as T
    e._eval_lid(c, m)
    e._lid_geometry(m)
    for k in ("lid_off", "lid_on_table", "base_not_lifted", "base_not_tipped"):
        c.pop(k)
    c["lid_on"] = m["lid_fits"] and m["lid_overlap_mm"] >= T.LID_CLOSED_OVERLAP_MM
    c["base_upright_on_table"] = (m["base_tilt_signed_deg"] <= T.LID_BASE_TILT_DEG
                                   and -1. <= m["base_lowest_above_table_mm"] <= T.LID_BASE_LIFT_MM)


def progress(e, S, m):
    frame = 1. - .5*S.ramp(m["frame_outside_distance_mm"], 0., 50.)
    return S.lid_on(m, e.start_metrics, frame)


def verify(e):
    return e.evaluate_task(progress_of=progress)

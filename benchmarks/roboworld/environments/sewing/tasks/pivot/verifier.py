"""Outcome of pivot: an L seam through both plies, 10 mm from the right edge, a square corner 10 mm from the right and
near edges turned with the needle down, then 10 mm from the near edge to the left edge, without skipped stitches or
gaps; the needle intact, no thread broken, the threads trimmed and the work let go.

verify(e) evaluates an episode; e is its evaluator. Under the released contract (sewing-v1) that is task.Evaluator,
which grades checks(e) with PRIMARY, the check without which the score is 0 (six stitches through both plies). Under
sewing-tasks-v1 and -v2 it is task_set_scoring.TaskEvaluator or task_set_v2_scoring.TaskEvaluator, which grade the
seam along seam_path (the L), with the corner in both plies (CORNERS), without asking for backtacks (BACKTACKS) and
with FINISH as the progress's finishing factor before trimming and letting go. It returns the evaluation (task,
success, score, primary_outcome, checks, info, task_contract)."""
import numpy as np

from sewing.task import END_REACH, TOL_LINE, _check, _skips, _spans

PRIMARY = ("stitches_formed",)
BACKTACKS = False
CORNERS = True
FINISH = .90


def verify(e):
    return e.verdict(checks, primary=PRIMARY)


def seam_path(width, length, allowance=.010):
    right = width - allowance
    return np.array([[right, length], [right, allowance], [0., allowance]])


def checks(e):
    env = e.env
    w = env.world
    W, L = w.config.ply_width, w.config.ply_length
    st = [s for s in e.stitches() if s["formed"] and s["plies"] == [0, 1]]
    c = {}
    c["stitches_formed"] = _check(len(st) >= 6, len(st))
    if len(st) >= 6:
        uv = np.array([s["holes"][1] for s in st])
        corner = np.array([W - 0.010, 0.010])
        d = np.linalg.norm(uv - corner, axis=1)
        k = int(np.argmin(d))
        c["corner"] = _check(d[k] <= TOL_LINE, round(float(d[k])*1e3, 2), "<= 1.6 mm from (10, 10) mm")
        leg1, leg2 = uv[:k + 1], uv[k:]
        e1 = np.abs((W - leg1[:, 0]) - 0.010).max() if len(leg1) else 0.
        e2 = np.abs(leg2[:, 1] - 0.010).max() if len(leg2) else 0.
        c["leg1_line"] = _check(e1 <= TOL_LINE, round(float(e1)*1e3, 3))
        c["leg2_line"] = _check(e2 <= TOL_LINE, round(float(e2)*1e3, 3))
        gaps = np.linalg.norm(np.diff(uv, axis=0), axis=1)/_spans(st, e.stitches())
        c["continuous"] = _check(gaps.max() <= 3.5e-3, round(float(gaps.max())*1e3, 2),
                                 "<= 3.5 mm per stitch (a skipped stitch's span counts two)")
        c["skips"] = _skips(e.stitches())
        c["leg2_reaches_left_edge"] = _check(leg2[:, 0].min() <= END_REACH if len(leg2) else False,
                                             round(float(leg2[:, 0].min())*1e3, 2) if len(leg2) else None)
    c.update(e.common())
    return c

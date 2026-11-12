"""Outcome of seam-to-notch: a seam through both plies 10 mm from their right edges, from within 6 mm of the far end to
a point 120 mm from that end (the sewn end within 3.2 mm of it, no puncture more than 3.2 mm into the opening beyond
it), three or more reverse stitches at both sewn ends; the needle intact, no thread broken, the threads trimmed and the
work let go.

verify(e) evaluates an episode; e is its evaluator: task_set_scoring.TaskEvaluator under sewing-tasks-v1 or
task_set_v2_scoring.TaskEvaluator under sewing-tasks-v2, whose seam grading reads this task's hooks: seam_path, the
target line in the plies' material frame (width, length, allowance); distance(e, points, A), a hole's distance from it
(10 mm across the seam, and a longitudinal overrun past the mark counted beyond the line tolerance); extra_checks(e, g,
c), which adds stops_at_mark and opening_clear to the checks c from the seam's measurement g; extra_strays(e, g), the
rework of unformed punctures beyond the opening. It returns the evaluation (task, success, score, primary_outcome,
checks, info, task_contract)."""
import numpy as np

from sewing.task import TOL_LINE, _check
from sewing.task_set_scoring import ramp


def verify(e):
    return e.verdict()


def seam_path(width, length, allowance=.010):
    right = width - allowance
    end = length - .120
    return np.array([[right, length], [right, end]])


def distance(e, points, A):
    w = e.env.world
    mark = w.config.ply_length - .120
    lateral = abs(points[:, 0] - (w.config.ply_width - A))
    overrun = np.maximum(mark - .0032 - points[:, 1], 0.)
    return np.maximum(lateral, np.where(overrun > 0., TOL_LINE + overrun, 0.))


def extra_checks(e, g, c):
    mark = e.env.world.config.ply_length - .120
    near = [min((s["holes"][k][1] for s in g["two"]), default=float("inf")) for k in (0, 1)]
    c["stops_at_mark"] = _check(all(abs(v - mark) <= .0032 for v in near),
                                target="sewn end within 3.2 mm of 120 mm from far end, both plies")
    # All punctures count here, including unformed cycles and one-ply holes.
    c["opening_clear"] = _check(all(h.uv[1] >= mark - .0032 for h in e.env.world.machine.holes),
                                target="no puncture more than 3.2 mm into the opening")


def extra_strays(e, g):
    mark = e.env.world.config.ply_length - .120
    # Unformed punctures beyond the opening are rework too; formed holes
    # already lose line credit through the finite target path.
    return sum(1 - ramp(mark - h[1], .0032, .010)
               for s in g["stitches"] if not s["formed"] for h in s["holes"].values())/2

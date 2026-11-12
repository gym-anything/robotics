"""Outcome of square-plies: the top ply laid back square on the bottom one, their cut edges registered within 1.5 mm
(at 50 or more edge points, and every material point over its twin), each ply a single layer at 20 sample points,
the whole stack flat on the table left of the bed (heights -1..2 mm), no hand touching it and no robot reflex.

verify(e) evaluates an episode; e is its evaluator: task_set_scoring.TaskEvaluator under sewing-tasks-v1 or
task_set_v2_scoring.TaskEvaluator under sewing-tasks-v2, which grade square(e), the checks and the continuous progress
(the misregistration closed from the start's, times flatness and release). It returns the evaluation (task, success,
score, primary_outcome, checks, info, task_contract). check_start(e) is called when the evaluator is made: it records
the start's misregistration (square_start_error) and refuses a start whose plies are not slipped. HANDLING marks a
task without a seam (the sewing-tasks-v2 failure factor)."""
import numpy as np

from sewing.task import _check, _triangle_under
from sewing.task_set_scoring import ramp

HANDLING = True


def verify(e):
    return e.verdict(outcome=square)


def check_start(e):
    env = e.env
    if getattr(env.world, "cloth", True) is not None:
        e.square_start_error = max(e.ply_registration()[0], float(np.linalg.norm(
            env.world.ply_positions(0)[:, :2] - env.world.ply_positions(1)[:, :2], axis=1).max()))
        if e.square_start_error <= .0015:
            raise ValueError("square-plies must start with slipped plies")


def square(e):
    from sewing.layout import TABLE, BED
    w = e.env.world
    X, tri = np.asarray(w.mesh["X"]), np.asarray(w.mesh["tri"])
    xs = [np.asarray(w.ply_positions(k)) for k in (0, 1)]
    off, n = e.ply_registration()
    # Material correspondence also covers the parts for which registration has
    # no resting triangles (lifting the mismatched portion must not hide it).
    off = max(off, float(np.linalg.norm(xs[0][:, :2] - xs[1][:, :2], axis=1).max()))
    samples = np.array([(u, v) for u in np.linspace(.1, .9, 4)*w.config.ply_width
                        for v in np.linspace(.1, .9, 5)*w.config.ply_length])
    t, *bc = _triangle_under(samples, X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]])
    points = sum(b[:, None]*xs[1][tri[t, j], :2] for j, b in enumerate(bc))
    layers = [layer_count(p, tri, x, X) for p in points for x in xs]
    p = np.vstack(xs)
    inside = ((p[:, 0] > TABLE["x"][0]) & (p[:, 0] < BED["x"][0])
              & (p[:, 1] > TABLE["y"][0]) & (p[:, 1] < TABLE["y"][1]))
    height = float(np.maximum(-.001 - p[:, 2], p[:, 2] - .002).max())
    contacts = [e.hand_contact(a) for a in w.hand_arms]
    c = dict(edges_aligned=_check(n >= 50 and off <= .0015,
                                  dict(misregistration_mm=off*1000, edge_points=n), "<= 1.5 mm, >= 50 points"),
             single_layer=_check(all(k == 1 for k in layers), layers, "one crossing per ply at 20 samples"),
             on_left_table=_check(bool(inside.all()) and height <= 0., target="all vertices left of bed on table, z -1..2 mm"),
             released=_check(all(f < .05 for f in contacts), contacts, "< 0.05 N per hand, pad magnitudes"),
             no_reflex=_check(not w.reflexes, len(w.reflexes)))
    # Continuous geometric flatness: worst height, boundary violation and signed
    # projected triangle area. No reward for a fraction of Boolean checks.
    boundary = max(float((TABLE["x"][0] - p[:, 0]).max()), float((p[:, 0] - BED["x"][0]).max()),
                   float((TABLE["y"][0] - p[:, 1]).max()), float((p[:, 1] - TABLE["y"][1]).max()), 0.)
    def area(x):
        a, b = x[tri[:, 1], :2] - x[tri[:, 0], :2], x[tri[:, 2], :2] - x[tri[:, 0], :2]
        return a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    ratios = np.concatenate([area(x)/area(X) for x in xs])
    flat = min(ramp(max(height, 0.), 0., .020), ramp(boundary, 0., .020),
               float(np.clip(ratios.min()/.2, 0., 1.)))
    release = ramp(max(contacts, default=0.), .05, 1.)
    progress = ramp(off, .0015, e.square_start_error)*flat*(.9 + .1*release)
    e._info["progress"] = dict(value=progress, initial_error_m=e.square_start_error,
                               misregistration_m=off, flatness=flat, release=release)
    return c, progress


def layer_count(point, tri, positions, material=None):
    """Vertical crossings, deduplicating adjacent triangles on a shared edge."""
    a = positions[tri[:, 0]]
    v0, v1 = positions[tri[:, 1]] - a, positions[tri[:, 2]] - a
    den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
    valid = abs(den) > 1e-18
    a, v0, v1, den = a[valid], v0[valid], v1[valid], den[valid]
    q = point - a[:, :2]
    b = (q[:, 0]*v1[:, 1] - q[:, 1]*v1[:, 0])/den
    c = (v0[:, 0]*q[:, 1] - v0[:, 1]*q[:, 0])/den
    hit = np.minimum(np.minimum(1 - b - c, b), c) >= -1e-9
    heights = (a[:, 2] + b*v0[:, 2] + c*v1[:, 2])[hit]
    if material is None:
        material = positions[:, :2]
    T = tri[valid][hit]
    uv = ((1 - b[hit] - c[hit])[:, None]*material[T[:, 0]]
          + b[hit, None]*material[T[:, 1]] + c[hit, None]*material[T[:, 2]])
    # Coincident folds still have different material points. Height alone would
    # merge them and incorrectly call a flat fold one layer.
    unique = []
    for crossing in np.column_stack([uv, heights]):
        if not any(np.max(abs(crossing - old)) < 1e-7 for old in unique):
            unique.append(crossing)
    return len(unique)

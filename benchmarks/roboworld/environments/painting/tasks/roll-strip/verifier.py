"""Outcome of roll-strip: the marked strip rolled to full hide on >= 98 % of it, even in appearance (50 mm patches
within dE*ab 1.0 of their median, measured after the settle), without runs, lap marks, marks in the coat or paint on
the ceiling, trim or outlet plate, and the roller resting in the tray, released.

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria with the strip's own checks (uniform on the wall target, roller_in_tray after the film
checks) and NEED, the share of the strip that must hide. targets(e) gives the target cells (the strip, STRIP in
setup.py). quality(e, checks, defects) is the strip's part of the painting-tasks-v1 and -v2 progress
(scoring.TaskProgress, task_set_v2_verifier.TaskSetV2Progress): it adds the uniformity defect and returns the quality
factor of the roller's return."""
import numpy as np

from painting import params as P
from painting.layout import task_setup
from painting.paint import delta_e
from painting.task import _rect_cells

E = P.EVAL
NEED = E["hide_frac"]
STRIP = task_setup("roll-strip").STRIP


def targets(e):
    ch = e.paint.charts
    out = {}
    if "wall" in ch:
        w = ch["wall"]
        s = STRIP
        out["wall"] = _rect_cells(w, s["x0"], s["x1"], s["z0"], s["z1"])
    return out


def verify(e, settled):
    return e.measure_criteria(settled, wall=uniform, tools=roller_in_tray)


def uniform(e, checks, settled, w, m_full):
    de = _appearance(e, w, m_full) if settled else None
    checks["uniform"] = dict(dE=None if de is None else round(de, 3), limit=E["dE"],
                             ok=(de is not None and de <= E["dE"]) if settled else False,
                             note=None if settled else "measured by verify (settle rule)")


def roller_in_tray(e, checks):
    checks["roller_in_tray"] = _roller_in_tray(e)


def quality(e, checks, defects):
    defects["uniformity"] = uniformity_area(e)
    quality = 0.9 + 0.1*float(checks["roller_in_tray"]["ok"])
    return quality


def _appearance(e, ch, mask):
    """Dry-film appearance (linear RGB of the stack with wet solids counted as dry) of 50 mm patches in the target."""
    from painting.paint import stack_reflectance
    sl = e._bbox(f"app_{ch.name}", mask)
    d = (ch.d + ch.s)[(slice(None),) + sl]
    R = stack_reflectance(ch.paints, d, np.zeros_like(d), np.zeros_like(ch.w[sl]), ch.substrate_rgb[sl])
    mask = mask[sl]
    k = max(1, int(round(0.05/ch.dx)))
    ii, jj = np.nonzero(mask)
    if not len(ii):
        return None
    pi, pj = ii//k, jj//k
    keys = pi*100000 + pj
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    col = np.zeros((len(uk), 3))
    np.add.at(col, inv, R[ii, jj])
    col /= cnt[:, None]
    full = cnt >= 0.8*k*k
    if not np.any(full):
        return None
    col = col[full]
    med = np.median(col, axis=0)
    return float(delta_e(col, med[None]).max())


def _roller_in_tray(e):
    w = e.env.world
    bq = w.state.body_q.numpy()
    r = e.paint.applicators.get("roller")
    tray = w.scene.bodies.get("tray")
    if r is None or tray is None:
        return dict(ok=False)
    from painting.film import _quat_rot
    loc = _quat_rot(bq[tray][3:7]).T @ (bq[r.cover][:3] - bq[tray][:3])
    g = P.TRAY["size"]
    inside = abs(loc[0]) < g[0]/2 and abs(loc[1]) < g[1]/2 and loc[2] < 0.12
    released = (e._task_progress is not None or all(w.grip_force[a] == 0 for a in w.arms)) and not _held(e, r)
    return dict(cover_in_tray=bool(inside), released=bool(released), ok=bool(inside and released))


def _held(e, app):
    w = e.env.world
    fingers = {i for i in w.scene.robot_bodies if "finger" in w.model.body_label[i]}
    bodies = {app.frame, app.cover} if hasattr(app, "frame") else {app.body}
    return any((a in fingers and b in bodies) or (b in fingers and a in bodies) for a, b, *_ in w.contact_table())


def uniformity_area(e):
    """50 mm appearance patches, identical grouping/eligibility to _appearance.

    A patch costs its actual target area times clip((deltaE - 1)/2). Partial
    boundary patches below 80% of a full patch remain excluded as in the verdict.
    """
    from painting.paint import stack_reflectance, delta_e
    ch, mask = e.paint.charts["wall"], e.targets["wall"]
    sl = e._bbox("app_wall", mask)
    d = (ch.d + ch.s)[(slice(None),) + sl]
    rgb = stack_reflectance(ch.paints, d, np.zeros_like(d), np.zeros_like(ch.w[sl]), ch.substrate_rgb[sl])
    ii, jj = np.nonzero(mask[sl])
    if not len(ii):
        return 0.0
    k = max(1, int(round(0.05/ch.dx)))
    keys, inv, count = np.unique((ii//k)*100000 + jj//k, return_inverse=True, return_counts=True)
    cols = np.zeros((len(keys), 3))
    np.add.at(cols, inv, rgb[ii, jj])
    full = count >= 0.8*k*k
    cols = (cols/count[:, None])[full]
    if not len(cols):
        return 0.0
    de = delta_e(cols, np.median(cols, axis=0)[None])
    return float(np.sum(count[full]*ch.area*np.clip((de - E["dE"])/2.0, 0.0, 1.0)))

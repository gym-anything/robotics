"""Outcome of timber-bolt: a 9/16 in hole through both boards at the mark, the top board's entry within 2 mm of it,
square to the faces within the bolt's own 1.2 degrees, its wall not charred, and the carriage bolt pushed through both
boards by hand (less than 50 N).

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 contract_checks(ev) and the continuous progress(ev)."""
import numpy as np

from drilling import params as P
from drilling.catalogue_evaluator import mean
from drilling.task import SCORE_BACKED_EXIT, _angle_deg, _check, _in_plane, _ramp


def verify(ev):
    return ev.evaluate_checks(checks, contract_checks=contract_checks, progress=progress)


def checks(ev):
    out = []
    mk = ev.L["marks"][0]["local"]
    hit = ev.nearest("board1", mk)
    if hit is None or not hit[0].through:
        out.append(_check("through hole in the top board", False))
    else:
        h, e = hit
        out.append(_check("through hole in the top board", True))
        out.append(_check("entry within 2 mm of the mark", e <= 0.002, round(e*1e3, 3), "<= 2 mm"))
        ang = _angle_deg(h.a, -h.face_normal)
        tol = P.TIMBER_AXIS_TOL_DEG
        out.append(_check(f"square to the faces (axis within {tol:.2f} deg of the face normal)", ang <= tol,
                          round(ang, 3), f"<= {tol:.2f} deg"))
        out.append(_check("no charred wall (< 200 C)", h.scorch_max_c < P.WOOD_SCORCH_C[0], round(h.scorch_max_c, 1),
                          "< 200 C"))
    b0 = [h for h in ev.holes("board0") if h.through]
    out.append(_check("through hole in the bottom board", bool(b0), len(b0)))
    if b0 and SCORE_BACKED_EXIT:
        fd = max(h.exit_fd for h in b0)
        out.append(_check("exit damage factor at most 1.15", fd <= 1.15, round(fd, 3), "<= 1.15"))
    it = ev.w.hm.items["carriage_bolt"]
    through = [dn for dn, v in it.slots.items() if v[1] == "hole"]
    out.append(_check("carriage bolt through both boards", {"board0", "board1"} <= set(through), sorted(through)))
    out.append(_check("bolt pushed in with less than 50 N", it.state["max_axial_n"] < 50.,
                      round(it.state["max_axial_n"], 2), "< 50 N", kind="constraint"))
    return out


def lower_mark(ev, top):
    d1, d0 = (ev.w.hm.drillables[n] for n in ("board1", "board0"))
    p1, r1 = ev.w.body_pose(d1.body)
    p0, r0 = ev.w.body_pose(d0.body)
    mark = np.asarray(ev.L["marks"][0]["local"]) if top is None else top.p0
    axis = np.array([0., 0., -1.]) if top is None else top.a
    ray, direction = r0.T @ (p1 + r1 @ mark - p0), r0.T @ r1 @ axis
    # Follow the bore line to the lower face, rather than imposing an additional vertical
    # mark: a valid top entry 2 mm off with a 1-degree bore shifts its exit by another 1.3 mm.
    if abs(direction[2]) < 1e-9:
        return ray
    return ray + direction*((d0.center[2] + d0.half[2] - ray[2])/direction[2])


def timber_state(ev):
    hm = ev.w.hm
    mark = np.asarray(ev.L["marks"][0]["local"])
    picked, bore = {}, []
    for name in ("board1", "board0"):
        target = mark if name == "board1" else lower_mark(ev, picked["board1"])
        candidates = []
        for h in ev.holes(name):
            e = _in_plane(h, target)
            q = ev._bore(h, e, 14.3, 0.038, 0.002, axis_tol=P.TIMBER_AXIS_TOL_DEG)
            q *= _ramp(h.scorch_max_c, 200., 300.)
            valid = (h.through_at(14.3) and e <= 0.002 and
                     _angle_deg(h.a, -h.face_normal) <= P.TIMBER_AXIS_TOL_DEG and h.scorch_max_c < 200.)
            candidates.append(((valid, q, -e, -h.id), h, q))
        _, h, q = max(candidates, key=lambda c: c[0]) if candidates else (None, None, 0.)
        picked[name], bore = h, [*bore, q]
    it = hm.items["carriage_bolt"]
    shares = [ev._overlap(it, n, h) for n, h in picked.items()]
    return picked, mean(bore), mean(shares), shares


def contract_checks(ev):
    """The checks under drilling-tasks-v1 and -v2: each board's bore at its marked place (the lower board's where the
    top bore's line leaves it) and the bolt spanning both."""
    picked, _, _, shares = timber_state(ev)
    out = []
    mark = np.asarray(ev.L["marks"][0]["local"])
    marks = {"board1": mark, "board0": lower_mark(ev, picked["board1"])}
    for name, h in picked.items():
        out.append(_check(f"{name}: through at 14.3 mm", h is not None and h.through_at(14.3)))
        if h is not None:
            out += [_check(f"{name}: entry within 2 mm of the marked bore", _in_plane(h, marks[name]) <= 0.002),
                    _check(f"{name}: square to the faces", _angle_deg(h.a, -h.face_normal) <= P.TIMBER_AXIS_TOL_DEG),
                    _check(f"{name}: no charred wall (< 200 C)", h.scorch_max_c < 200.)]
    out.append(_check("carriage bolt spans both marked through holes", all(x >= 1. - 1e-6 for x in shares)))
    force = ev.w.hm.items["carriage_bolt"].state["max_axial_n"]
    out.append(_check("bolt pushed in with less than 50 N", force < 50., force, "< 50 N", kind="constraint"))
    return out


def progress(ev):
    _, bore, bolt, _ = timber_state(ev)
    return (2*bore + bolt)/3.

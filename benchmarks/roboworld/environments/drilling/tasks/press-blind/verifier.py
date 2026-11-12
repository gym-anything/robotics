"""Outcome of press-blind: the M8 tap-drill hole in the block 15.0 to 15.5 mm deep at full diameter and not through,
within 0.5 mm of the punched mark and within 1 degree of square, and the press left with its spindle stopped, its quill
up and its bit clear of the work.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 the continuous progress(ev) as well. candidate(ev, h, e) rates a hole whose entry is e (m) off
the mark for the contract graders' choice of the mark's hole (catalogue_evaluator.CatalogueEvaluator.nearest)."""
from drilling import params as P
from drilling.catalogue_evaluator import clip
from drilling.task import SCORE_BIT_BURN, _angle_deg, _check, _ramp


def verify(ev):
    return ev.evaluate_checks(checks, progress=progress)


def checks(ev):
    out = []
    mark = ev.L["marks"][0]["local"]
    hit = ev.nearest("alu_block", mark)
    if hit is None:
        return [_check("full-diameter depth 15.0-15.5 mm", False, None, "15.0-15.5 mm")] + \
            ev.spindle_stopped_quill_up()
    h, e = hit
    fd = h.full_depth(6.8)
    out.append(_check("full-diameter depth 15.0-15.5 mm", 0.0150 <= fd <= 0.0155, round(fd*1e3, 3), "15.0-15.5 mm"))
    out.append(_check("not through", not h.through, h.through))
    out.append(_check("entry within 0.5 mm of the mark", e <= 0.0005, round(e*1e3, 3), "<= 0.5 mm"))
    ang = _angle_deg(h.a, -h.face_normal)
    out.append(_check("axis within 1 deg", ang <= 1., round(ang, 3), "<= 1 deg"))
    if SCORE_BIT_BURN:
        T = ev.s["max_bit_temp_c"].get("drill_press_bit", 20.)
        out.append(_check("bit tip never above 550 C", T < P.HSS_BURN_C, round(T, 1), "< 550 C", kind="constraint"))
    return out + ev.spindle_stopped_quill_up()


def blind(ev, h, error):
    fd = h.full_depth(6.8)
    d = clip(fd/0.015)*_ramp(fd, 0.0155, 0.018)
    return (0. if h.through or not ev._intact(h) else d*_ramp(error, 0.0005, 0.0015)
            *_ramp(_angle_deg(h.a, -h.face_normal), 1., 3.))


def candidate(ev, h, e):
    """(quality, valid) of hole h as the mark's hole."""
    quality = blind(ev, h, e)
    valid = not h.through and 0.015 <= h.full_depth(6.8) <= 0.0155 and e <= 0.0005
    return quality, valid


def progress(ev):
    hit = ev.nearest("alu_block", ev.L["marks"][0]["local"])
    bore = 0. if hit is None else blind(ev, *hit)
    return bore*(0.85 + 0.15*ev._finish())

"""Outcome of panel-holes: four 6 mm holes through the cover, each within 1 mm of its mark, square to the sheet within
1 degree and round within 1.10; under the task-set contracts no stray through-hole either.

verify(ev) receives the episode's evaluator (task.Evaluator, or a contract grader built on it: catalogue_evaluator,
task_set_v2_evaluator) and returns its evaluation dict (Evaluator.evaluate_checks): checks(ev) below, and under
drilling-tasks-v1 or -v2 contract_checks(ev) and the continuous progress(ev). candidate(ev, h, e) rates a hole whose
entry is e (m) off a mark for the contract graders' choice of the mark's hole
(catalogue_evaluator.CatalogueEvaluator.nearest)."""
from drilling.catalogue_evaluator import clip, depth_share, mean
from drilling.task import SCORE_BACKED_EXIT, SCORE_EXIT_BURR, _angle_deg, _check, _in_plane, _ramp

# panel-holes "did not grab": the thin-sheet grab (a twist drill catching 1.5 mm sheet like a thread) is not reproduced
# (E6: 1.0 N m, 0.8 deg housing turn), so the housing-turn check cannot fail. The task is described accordingly: it
# is a placing, piloting and enlarging task on a backed sheet (the trade's way of avoiding the grab), not a test of
# handling a drill that grabs.
SCORE_SHEET_GRAB = False


def verify(ev):
    return ev.evaluate_checks(checks, contract_checks=contract_checks, progress=progress)


def checks(ev):
    out = []
    for k, mk in enumerate(ev.L["marks"]):
        o, h = ev._through_hole_checks("cover_panel", mk["local"], 6.0, 0.001, f"hole {k}",
                                       burr_scored=SCORE_EXIT_BURR and SCORE_BACKED_EXIT)
        out += o
        if h is not None:
            # each stage's own wander from where its full diameter began (holes.py wander_m): a 6 mm drill
            # centred in a pilot that wandered cuts a round hole, off the mark (the entry check judges that)
            rnd = (h.d_mm + 2e3*h.wander_m)/h.d_mm
            out.append(_check(f"hole {k}: roundness at most 1.10", rnd <= 1.10, round(rnd, 4), "<= 1.10"))
    if SCORE_SHEET_GRAB:
        for n, ang in ev.s["housing_turn_deg"].items():
            out.append(_check(f"{n} did not grab (housing turned < 30 deg)", ang < 30., round(ang, 2), "< 30 deg",
                              kind="constraint"))
    return out


def contract_checks(ev):
    """The checks under drilling-tasks-v1 and -v2: checks(ev) and no stray through-hole."""
    out = checks(ev)
    bad = [h.id for h in panel_damage(ev) if h.through]
    out.append(_check("no stray through-hole", not bad, bad, kind="constraint"))
    return out


def area_depth_share(h, diameter, thickness):
    """Union of the stages' cylindrical volumes, normalized to the requested bore.

    A through 3 mm pilot is 1/4 of a 6 mm bore; an uncut wider stage adds nothing.
    Integrating annuli avoids double counting a pilot, and the credit rises continuously from 1/4 to 1 as the
    wider stage cuts deeper, with no jump.
    """
    widths = sorted({0., *(min(diameter, s["d_mm"]) for s in h.stages)})
    return clip(sum((b*b - a*a)/(diameter*diameter)*depth_share(h, b, thickness)
                    for a, b in zip(widths, widths[1:])))


def panel(ev, h, error):
    rnd = (h.d_mm + 2e3*h.wander_m)/max(h.d_mm, 1e-9)
    return (area_depth_share(h, 6., 0.0015)*_ramp(error, 0.001, 0.003)
            *_ramp(_angle_deg(h.a, -h.face_normal), 1., 3.)*_ramp(rnd, 1.10, 1.30)
            if ev._intact(h) else 0.)


def panel_damage(ev):
    chosen = [ev.nearest("cover_panel", m["local"]) for m in ev.L["marks"]]
    # Re-entry inside a bit radius is the same opening, as in the ledger-anchor rule.
    return [h for h in ev.holes("cover_panel") if not any(
        hit is not None and hit[1] <= 0.003 and _in_plane(h, hit[0].p0) <= 0.003 for hit in chosen)]


def candidate(ev, h, e):
    """(quality, valid) of hole h as the mark's hole."""
    quality = panel(ev, h, e)
    valid = h.through_at(6.) and e <= 0.001 and (h.d_mm + 2e3*h.wander_m)/max(h.d_mm, 1e-9) <= 1.10
    return quality, valid


def progress(ev):
    credits = [panel(ev, *hit) if (hit := ev.nearest("cover_panel", m["local"])) else 0.
               for m in ev.L["marks"]]
    return clip(mean(credits) - sum(area_depth_share(h, 6., 0.0015) for h in panel_damage(ev))/4.)

"""Outcome of tube-label: the printed patient label taken from the printer and applied to the specimen tube lengthwise
and straight (skew at most 5 deg) over the tube manufacturer's label: the patient's name at the end toward the cap, that
end lined up with the manufacturer's label's cap-side edge (0.5 mm capward to 2 mm below it), at most 2 mm of the
manufacturer label's bare viewing strip covered (and at least 5 mm of it open), clear of the tube's bottom 9 mm; wetted
out to the standard over its area and along its edges, without trapped air, without a barcode crease at any time in the
episode, untorn, bonded only to the tube and settled; the tube left lying in its cradle and nothing touching the label
or the tube; the barcode decodes when the optical reader runs. tube-wrap's job, held to the stricter reading its
instruction states.

verify(e, optical) gets the episode's evaluator (task.Evaluator) and the optical reader's verdict (dict with "match", or
None) and returns the evaluation (task.Evaluator.evaluate_task: success, score, checks, metrics, progress, reward). The
evaluator reads this task's item (TARGET), placement rules (placement: tube-wrap's plus two), progress zone (zone:
tube-wrap's) and registration (registration: tube-wrap's factors, with the viewing window and the skew read from where
the label is going, predicted_window) from here."""
import math

import numpy as np

TARGET = "tube"


def placement(e, on_target):
    """tube-label: tube-wrap's placement rules (unchanged, _place_tube_wrap) plus the two requirements its
    instruction states that those rules do not check:
    - over the tube manufacturer's label: at the narrowest station along the label, at most
      TUBE_STRIP_TOL of the manufacturer label's bare strip lies under it (bare strip less the window that
      _tube_window measures). tube-wrap's window >= 5 mm alone would let a label turned 80 deg, a third of it on bare
      PET over 59 % of the strip, pass;
    - the patient's name at the end toward the cap. The name is printed from the label art's
      left edge (artwork.specimen_label), which the renderer maps to the label's v = width end (render.export:
      s = 1 - v/width), so that end is at the cap when the label's CD axis runs toward the cap: dz/dv > 0 in the
      tube frame (+z toward the cap), from an affine fit of the nodes' axial position on their material
      coordinates. symbol_rect maps image x to v the other way; the symbol is centred across
      the label, so its rectangle is the same either way."""
    from label_application.task import TUBE_STRIP_TOL
    base = e._place_tube_wrap(on_target)
    lab = e.world.label
    q = e._item_local("tube")
    m = base["metrics"]
    intrusion = float(m["bare_strip_m"] - m["window_m"])
    dz_dv = float(np.linalg.lstsq(np.c_[lab.X, np.ones(lab.n)], q[:, 2], rcond=None)[0][1])
    checks = dict(base["checks"])
    checks["over_mfr_label(<=2mm_onto_viewing_strip)"] = intrusion <= TUBE_STRIP_TOL + 1e-12
    checks["patient_name_end_toward_cap"] = dz_dv > 0.
    return dict(checks=checks, metrics=dict(m, viewing_strip_covered_m=intrusion,
                                            strip_tolerance_m=TUBE_STRIP_TOL, name_end_dz_dv=dz_dv,
                                            name_end="cap" if dz_dv > 0. else "tube bottom"))


def zone(e, on_target):
    return e.tube_zone()


def registration(e, placed, on_target=None):
    return e._tube_registration(placed, predicted_window)


def predicted_window(e, q, g, placed, coef, r, dv):
    """(window, bare strip, skew reading, extra metrics) for task.Evaluator._tube_registration: the window and bare
    strip the whole label will leave where the fit says it is going (predicted_tube_window), so the order in which its
    long edges went down does not enter; and the skew ramp's reading of the patient name's end (its
    patient_name_end_toward_cap check)."""
    from label_application.task import TUBE_STRIP_TOL
    window, bare = predicted_tube_window(e, q, g, placed, coef, r)
    # the skew ramp reads the SIGNED angle between the label's name axis (+v, the end where the printed name
    # begins; see placement) and the tube's +z toward the cap, 0-180 deg: the same as tube-wrap's
    # reading for a label heading name end capward, 0 for one heading for the reversed placement
    skew_read = float(np.degrees(np.arctan2(abs(dv[0]), dv[1])))
    extra = dict(window_from="predicted extent of the whole label", bare_strip_m=bare,
                 window_full_credit_from_m=bare - TUBE_STRIP_TOL, name_axis_from_cap_deg=skew_read,
                 name_end_heading="cap" if dv[1] > 0 else "tube bottom")
    return window, bare, skew_read, extra


def predicted_tube_window(e, q, g, placed, coef, r):
    """tube-label: (window, bare strip) the label would leave lying where the affine fit of its placed nodes'
    unrolled tube position on their material coordinates puts every node (the fit _tube_registration makes for cap
    end and skew), measured by _tube_window over all of its triangles. For a finished label it equals the measured
    window to within about 0.02 mm; for a partly laid one it is where that label is going (read from the placed
    triangles only, a label laid edge-first would be scored by which long edge went down first, and finishing a
    wrap onto the strip would lower its score)."""
    lab = e.world.label
    ang = np.arctan2(q[placed, 1], q[placed, 0])
    gm = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))   # _unrolled_along_label's origin
    pred = np.c_[lab.X, np.ones(lab.n)] @ coef
    a = gm + pred[:, 0]/r
    window, bare, _ = e._tube_window(np.c_[r*np.cos(a), r*np.sin(a), pred[:, 1]], g)
    return float(window), float(bare)


def verify(e, optical=None):
    return e.evaluate_task(optical)

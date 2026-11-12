"""Outcome of fit-up-bracket: the bracket fitted up on its layout for tacking, both ends of its root edge within
FIT_MM (1 mm) of the layout and square within 1 deg, released and at rest, the clamped plate not moved and no arm
latched in a protective stop.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle changes nothing here) and
returns the evaluation: success, score, checks, progress and reward (Evaluator.evaluate_checks with
_eval_fit_up_bracket and _progress_fit_up_bracket below). WELD: the task is not judged on the weld's sections."""
import numpy as np

from welding.bonds import FIT_MM
from welding.task import _ok, graded, net_of_start

WELD = False
# Success: both ends of the bracket's root edge (the joint's line) within FIT_MM of the layout (the fit-up the thermal
# grid describes, the tack task's position limit; a centre offset alone misses a yaw), square within 1 deg (the tack
# task's), released, at rest (E0's scale: 1 mm/s), the clamped plate not moved over FITUP_PLATE_MM, no arm latched in a
# protective stop
FITUP_SQUARE_DEG = 1.
FITUP_SQUARE_ZERO_DEG = 10.                # a bracket tipped this far earns no position credit [authored]
FITUP_REST_M_S = 1e-3
FITUP_PLATE_MM = .5


def verify(e, settle=False):
    return e.evaluate_checks(lambda: _eval_fit_up_bracket(e), lambda: _progress_fit_up_bracket(e))


def fitup_progress(err, start_err, held):
    """fit-up-bracket's progress: the root edge's approach to its layout net of the start (1 within FIT_MM), times a
    squareness term (1 within 1 deg, 0 tipped FITUP_SQUARE_ZERO_DEG), and 0 while the hand still touches the bracket
    (held is not placed)."""
    pos = net_of_start(err["root_edge_mm"], start_err["root_edge_mm"], FIT_MM)
    if pos is None:
        pos = float(err["root_edge_mm"] <= FIT_MM + 1e-9)
    sq = graded(err["tilt_deg"], FITUP_SQUARE_DEG, FITUP_SQUARE_ZERO_DEG - FITUP_SQUARE_DEG)
    return dict(progress=0. if held else float(pos*sq), position=pos, squareness=sq, held=bool(held))


def _eval_fit_up_bracket(e):
    """The bracket on its layout (both ends of its root edge within FIT_MM), square, released, at rest; the clamped
    plate not moved; no arm latched in a protective stop."""
    env, w = e.env, e.env.world
    sc = w.scene
    err = e._fitup_errors(w.body_pose)
    held = env.hand_touches_bracket()
    speed = float(np.linalg.norm(w.state.body_qd.numpy()[sc.bodies["bracket"], :3]))
    moved = float(np.linalg.norm(w.body_pose(sc.bodies["base_plate"])[0] - e._start_measures()["plate_m"]))*1e3
    stopped = sorted(a for a, until in w._reflex_until.items() if w.time < until)
    e._measured = dict(err=err, held=held)
    return {
        "on_layout": dict(root_edge_mm=err["root_edge_mm"], root_ends_mm=err["root_ends_mm"],
                          centre_offset_mm=err["centre_mm"], yaw_deg=err["yaw_deg"], limit_mm=FIT_MM,
                          ok=_ok(err["root_edge_mm"] <= FIT_MM + 1e-9)),
        "square": dict(angle_error_deg=err["tilt_deg"], limit_deg=FITUP_SQUARE_DEG,
                       ok=_ok(err["tilt_deg"] <= FITUP_SQUARE_DEG + 1e-9)),
        "released": dict(touching=bool(held), ok=_ok(not held)),
        "at_rest": dict(speed_mm_s=speed*1e3, limit_mm_s=FITUP_REST_M_S*1e3, ok=_ok(speed < FITUP_REST_M_S)),
        "plate_unmoved": dict(moved_mm=moved, limit_mm=FITUP_PLATE_MM, ok=_ok(moved <= FITUP_PLATE_MM + 1e-9)),
        "no_protective_stop": dict(stopped_arms=stopped, ok=_ok(not stopped)),
    }


def _progress_fit_up_bracket(e):
    m = e._measured
    return fitup_progress(m["err"], e._start_measures()["fitup"], m["held"])

"""Outcome of hold-and-present: position-torch's presentation, unchanged (the torch at the start of the near-side
seam ready to weld, no arc, the gun touching nothing), while both fingers of the hand grip the loose bracket and hold
it square within 1 deg and within FIT_MM of its fit-up, no arm latched in a protective stop.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle changes nothing here) and
returns the evaluation: success, score, checks, progress and reward (Evaluator.evaluate_checks with
_eval_hold_and_present and _progress_hold_and_present below). WELD: the task is not judged on the weld's sections."""
from welding.bonds import FIT_MM
from welding.task import _ok, graded

WELD = False
# hold-and-present: position-torch's checks, unchanged, with the loose bracket held: both Franka fingers on it (the tack
# demonstrations grip its top edge across its 8 mm thickness, 30 N per finger) and the bracket square within 1 deg and
# within FIT_MM of its fit-up, the tack task's own limits (E24: the grip left it 0.058 deg and 0.068 mm off).
# Progress: HOLD_PRESENT_TORCH_SHARE x position-torch's progress + the rest x the hold (held x the alignment term)
# [authored]
HOLD_PRESENT_TORCH_SHARE = .5


def verify(e, settle=False):
    return e.evaluate_checks(lambda: _eval_hold_and_present(e), lambda: _progress_hold_and_present(e))


def hold_present_progress(torch_progress, held, square_deg, offset_mm):
    """hold-and-present's progress: HOLD_PRESENT_TORCH_SHARE x position-torch's progress (position_progress, net of the
    start) + the rest x the hold: 1 while both fingers hold the bracket, times its alignment (squareness and position
    each 1 within the tack task's limits, 0 at twice: the smaller); 0 while not held."""
    align = min(graded(abs(square_deg), 1., 1.), graded(offset_mm, FIT_MM, FIT_MM))
    hold = float(align) if held else 0.
    return dict(progress=float(HOLD_PRESENT_TORCH_SHARE*torch_progress + (1. - HOLD_PRESENT_TORCH_SHARE)*hold),
                torch_progress=float(torch_progress), hold=hold, alignment=float(align), held=bool(held))


def _fingers_on_bracket(e):
    """Whether each finger of the handling hand touches the bracket (MuJoCo contacts with load): a grip across its
    thickness has both; a hand resting on it or one finger pushing it does not."""
    w = e.env.world
    if "bracket" not in w.scene.bodies or not w.hand_arms:
        return []
    lab = list(w.model.body_label)
    fingers = [b for b in w.hand_bodies() if "finger" in lab[b].split("/")[-1]]
    return [bool(w.bodies_touch(w.scene.bodies["bracket"], {b})) for b in fingers]


def _eval_hold_and_present(e):
    """position-torch's checks, unchanged, and the hold: both fingers on the bracket, the bracket square within 1 deg
    and within FIT_MM of its fit-up (env.bracket_alignment, the tack task's measures), no arm latched in a protective
    stop."""
    env, w = e.env, e.env.world
    checks = e._eval_position_torch()
    fingers = _fingers_on_bracket(e)
    held = len(fingers) >= 2 and all(fingers)
    sq, off = env.bracket_alignment()
    stopped = sorted(a for a, until in w._reflex_until.items() if w.time < until)
    e._measured.update(held=held, square=float(sq), offset=float(off))
    checks.update({
        "holding": dict(fingers_on_bracket=fingers, ok=_ok(held)),
        "square": dict(angle_error_deg=float(sq), limit_deg=1., ok=_ok(held and abs(sq) <= 1.)),
        "position": dict(offset_mm=float(off), limit_mm=FIT_MM, ok=_ok(held and off <= FIT_MM)),
        "no_protective_stop": dict(stopped_arms=stopped, ok=_ok(not stopped)),
    })
    return checks


def _progress_hold_and_present(e):
    m = e._measured
    return hold_present_progress(e._progress_position_torch()["progress"], m["held"], m["square"], m["offset"])

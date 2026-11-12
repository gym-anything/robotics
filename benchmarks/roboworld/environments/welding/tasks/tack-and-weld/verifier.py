"""Outcome of tack-and-weld: the bracket tacked square to the plate (bonded, square within 1 deg, within FIT_MM of the
fit-up, the hand released) and a continuous 1/4 in fillet along the near side over the tacks, judged as fillet-2f
judges its weld.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle=True, the on-demand
evaluation, judges a weld still freezing with the arc off as it freezes hands-off) and returns the evaluation:
success, score, checks, progress, reward, judged, flags and diagnostics (judge_tack_and_weld below on
Evaluator._final_rows and the assembly, _assembly). WELD: the task is judged on the weld's sections (the evaluator
reads the start state's raw progress, Evaluator.start_raw_progress)."""
from welding.bonds import FIT_MM
from welding.task import FAULT_FACTOR, _ok, fillet_progress, graded

WELD = True


def verify(e, settle=False):
    rows, judged, flags = e._final_rows(settle)
    return judge_tack_and_weld(e, rows, judged, flags, **_assembly(e))


def _assembly(e):
    env, w = e.env, e.env.world
    sq, off = env.bracket_alignment()
    return dict(bond_active=bool(w.bond.active), square_deg=float(sq), offset_mm=float(off),
                hand_touching=bool(env.hand_touches_bracket()))


def judge_tack_and_weld(e, rows, judged=None, flags=None, bond_active=False, square_deg=0., offset_mm=0.,
                        hand_touching=False):
    """tack-and-weld (A11): every fillet-2f check on the final weld, which incorporates the tacks and is judged as
    one weld (as fillet-2f judges its fillet over the fixture's tacks), and the tacked assembly: bonded, square
    within 1 deg, within FIT_MM of the fit-up, the hand released. The tacks are the means: a bracket held and welded
    without them is judged by the same end state. Progress: fillet_progress (net of the loose start, which holds no
    weld) x the alignment (each of squareness and position 1 within its limit, 0 at twice: the smaller) x
    FAULT_FACTOR while the hand still touches the bracket."""
    checks = e._eval_fillet_2f(rows)
    diagnostics = dict(e.diagnostics)
    checks.update({
        "bonded": dict(active=bool(bond_active), ok=_ok(bond_active)),
        "square": dict(angle_error_deg=square_deg, limit_deg=1., ok=_ok(bond_active and abs(square_deg) <= 1.)),
        "position": dict(offset_mm=offset_mm, limit_mm=FIT_MM, ok=_ok(bond_active and offset_mm <= FIT_MM)),
        "hand_off": dict(touching=bool(hand_touching), ok=_ok(bond_active and not hand_touching)),
    })
    checks = e._named(checks)
    weld = fillet_progress(rows, e.start_raw_progress, e._last_craters)
    align = min(graded(abs(square_deg), 1., 1.), graded(offset_mm, FIT_MM, FIT_MM))
    release = FAULT_FACTOR if hand_touching else 1.
    reward = dict(weld, weld_progress=weld["progress"], alignment=align, release_factor=release,
                  progress=float(weld["progress"]*align*release))
    return e._scored(all(c["pass"] for c in checks.values()), checks, reward, judged=judged or {},
                     flags=flags or {}, diagnostics=diagnostics)

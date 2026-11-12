"""Outcome of decant-acid: 6-14 mL of acid in the beaker, the bottle standing upright, at rest and released anywhere on
the bench, the beaker upright, and the flask's acid and base unchanged, with nothing spilled and none of the aliquot
lost.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev) with the tick's measurements
(m) and the values the tasks share as keywords (CatalogueEvaluator._evaluate); it returns (p, checks, terms,
penalty): the raw progress, the shared checks (spill, acid_retained) with the task's own added, the reward terms and
the penalty taken off p."""
import math

from titrate.newton15.task_evaluators import clip, grade


def verify(ev, m, *, r, base, checks, terms, penalty, **_):
    t = clip(clip(m['beaker_acid_ml']/6.)-clip((m['beaker_acid_ml']-14.)/6.))
    u = grade(m['bottle_tilt_rad'], math.acos(.98), math.pi/2)*grade(m['bottle_bench_error_m'], .001, .05)
    u *= grade(m['bottle_speed_m_s'], .01, .2)*grade(m['bottle_spin_rad_s'], .1, 1.)
    p = t*(1+u)/2
    acid_change = abs(m['acid_in_flask_mol']*1000/r.acid_molar-r.aliquot_ml)
    p -= (max(0., acid_change-.05)+max(0., base-.01))/.6
    checks.update(transferred=6. <= m['beaker_acid_ml'] <= 14.,
                  bottle_returned=m['bottle_tilt_rad'] < math.acos(.98)
                  and m['bottle_bench_error_m'] <= .001 and m['bottle_speed_m_s'] <= .01
                  and m['bottle_spin_rad_s'] <= .1 and not m['bottle_robot_contact'],
                  beaker_upright=m['beaker_tilt_rad'] < math.radians(5),
                  flask_unchanged=acid_change <= .05 and base <= .01)
    terms.update(transferred=t, bottle_return=u)
    return p, checks, terms, penalty

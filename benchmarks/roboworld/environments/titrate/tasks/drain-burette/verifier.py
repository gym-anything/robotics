"""Outcome of drain-burette: the burette empty (at most 0.2 mL left), all but 0.3 mL of its charge in the waste
beaker, the stopcock left open, at most 0.05 mL of base in the flask, flask and beaker upright, nothing spilled and
none of the aliquot lost.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev) with the tick's measurements
(m) and the values the tasks share as keywords (CatalogueEvaluator._evaluate); it returns (p, checks, terms,
penalty): the raw progress, the shared checks (spill, acid_retained) with the task's own added, the reward terms and
the penalty taken off p."""
import math

from titrate.newton15.task_evaluators import clip, grade


def verify(ev, m, *, r, base, checks, terms, penalty, **_):
    d = clip(m['beaker_base_ml']/(r.burette_ml-.3))
    opened = grade(m['valve_rad'], .2, 1.2)
    p = d*(1+opened)/2-max(0., base-.05)/.6
    checks.update(empty=m['burette_volume_ml'] <= .2, collected=m['beaker_base_ml'] >= r.burette_ml-.3,
                  open=m['valve_rad'] <= .2, flask_protected=base <= .05,
                  flask_upright=m['flask_tilt_rad'] < math.radians(5),
                  beaker_upright=m['beaker_tilt_rad'] < math.radians(5))
    terms.update(collected=d, open_grade=opened)
    return p, checks, terms, penalty

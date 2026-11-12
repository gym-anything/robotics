"""Outcome of prime-burette: 0.3-2 mL of base run through the tip into the waste beaker, the stopcock closed for 2 s,
the flask back under the tip and released, the beaker upright, at most 0.05 mL of base in the flask, nothing spilled
and none of the aliquot lost.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev) with the tick's measurements
(m) and the values the tasks share as keywords (CatalogueEvaluator._evaluate); it returns (p, checks, terms,
penalty): the raw progress, the shared checks (spill, acid_retained) with the task's own added, the reward terms and
the penalty taken off p."""
import math

from titrate.newton15.task_evaluators import clip, grade


def verify(ev, m, *, base, closed_s, k, upright, plate, placed, checks, terms, penalty, **_):
    w = clip(clip(m['beaker_base_ml']/.3)-max(0., m['beaker_base_ml']-2.)/2.)
    placement = grade(m['flask_distance_m'], .006, .05)*upright*plate
    p = w*(1+placement+placement*k)/3-max(0., base-.05)/.6
    checks.update(primed=.3 <= m['beaker_base_ml'] <= 2., closed=closed_s >= 2.,
                  flask_placed=placed, beaker_upright=m['beaker_tilt_rad'] < math.radians(5),
                  flask_protected=base <= .05)
    terms.update(transferred=w, placement=placement)
    return p, checks, terms, penalty

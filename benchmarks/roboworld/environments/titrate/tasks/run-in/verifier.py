"""Outcome of run-in: between v - 1.5 and v - 0.5 mL of base in the flask (v the equivalence volume, titrant on its
way counted), the stopcock closed for 2 s and the solution still colourless (at most 10 % pink), with nothing spilled
and none of the aliquot lost.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev) with the tick's measurements
(m) and the values the tasks share as keywords (CatalogueEvaluator._evaluate); it returns (p, checks, terms,
penalty): the raw progress, the shared checks (spill, acid_retained) with the task's own added, the reward terms and
the penalty taken off p."""
from titrate.newton15.task_evaluators import clip


def verify(ev, m, *, r, base, closed_s, k, checks, terms, penalty, **_):
    v = r.aliquot_ml*r.acid_molar/r.base_molar
    completion = clip(base/(v-1.5))
    accuracy = 1-clip(max(0., base+m['pending_base_ml']-(v-.5))/.6)
    p = completion*accuracy*(1+k)/2
    checks.update(dose=v-1.5 <= base and base+m['pending_base_ml'] <= v-.5,
                  closed=closed_s >= 2., colourless=m['pink_fraction'] <= .1)
    terms.update(completion=completion, accuracy=accuracy)
    return p, checks, terms, penalty

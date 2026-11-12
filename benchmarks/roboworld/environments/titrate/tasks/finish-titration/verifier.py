"""Outcome of finish-titration: titration's four checks on the paused flask (endpoint within one drop of the flask's
own acid, the pink held throughout the solution for 30 s, at most 0.05 mL spilled, at most 0.05 mL of the aliquot
lost); progress counts only the base added after the pause.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev, whose setup holds the start's
base0_ml) with the tick's measurements (m) and the values the tasks share as keywords (CatalogueEvaluator._evaluate);
it returns (p, checks, terms, penalty): the raw progress, the endpoint evaluator's checks, its reward terms with the
task's completion, and no penalty beyond its accuracy term."""
from titrate.newton15.task_evaluators import clip


def verify(ev, m, *, e, base, **_):
    # The same four success predicates as titration.
    checks = dict(e['checks'])
    base0 = ev.setup['base0_ml']
    remaining = e['flask_equivalence_ml']-base0
    completion = clip((base-base0)/remaining) if remaining > 0 else 0.
    terms = {**e['reward'], 'completion': completion}
    p = completion*terms['accuracy']*(1+terms['persistence'])/2
    penalty = 0.  # Already included in titration's accuracy term.
    return p, checks, terms, penalty

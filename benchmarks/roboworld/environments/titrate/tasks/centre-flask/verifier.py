"""Outcome of centre-flask: the flask back under the burette tip (within 6 mm), upright, flat on the plate, at rest and
released, with no base in it, nothing spilled and none of the aliquot lost.

verify(ev, m, **shared) is called at every tick by task_evaluators.CatalogueEvaluator (ev, whose setup holds the flask's
start, target and start distance) with the tick's measurements (m) and the values the tasks share as keywords
(CatalogueEvaluator._evaluate); it returns (p, checks, terms, penalty): the raw progress, the shared checks (spill,
acid_retained) with the task's own added, the reward terms and the penalty taken off p."""
import math

from titrate.newton15.task_evaluators import clip


def verify(ev, m, *, base, upright, plate, placed, checks, terms, penalty, **_):
    start = ev.setup['flask_start']
    target = ev.setup['flask_target']
    d0 = ev.setup.get('distance0_m', math.dist(start[:2], target[:2]))
    p = clip((d0-m['flask_distance_m'])/(d0-.006))*upright*plate
    p -= max(0., base-.01)/.6
    checks.update(flask_placed=placed, no_base=base <= .01)
    return p, checks, terms, penalty

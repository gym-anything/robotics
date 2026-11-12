"""Outcome of spray-stencil: the "3" sprayed through the stencil onto the crate side, a 25-50 um dry coat (5 mm gauge
readings) on >= 90 % of the crate under the apertures, sharp edges (paint more than 0.5 mm beyond them on <= 2 % of
their area), no pooled wet film over 250 um at declaration, and the emitted solids kept inside the spray box (<= 5 %
outside).

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria, whose checks on the crate target are the crate's (keyed on the crate chart, as are the
target cells under the stencil's apertures, task.Evaluator._aperture_on_crate). NEED is the share of the coat that
must read in the band, used by the painting-tasks-v1 and -v2 progress."""
from painting import params as P

NEED = P.EVAL["stencil_frac"]


def verify(e, settled):
    return e.measure_criteria(settled)

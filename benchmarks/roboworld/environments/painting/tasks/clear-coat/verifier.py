"""Outcome of clear-coat: the photo print clear-coated evenly, a 25-50 um dry coat (5 mm gauge readings) on >= 95 % of
it with the thickest reading at most twice the thinnest, no pooled wet film over 250 um at declaration, and the
emitted solids kept inside the spray box (<= 5 % outside).

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria, whose checks on the print target are the print's (keyed on the print chart)."""


def verify(e, settled):
    return e.measure_criteria(settled)

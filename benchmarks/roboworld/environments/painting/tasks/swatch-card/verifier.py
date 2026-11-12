"""Outcome of swatch-card: the pencilled 50 mm square painted solid (>= 95 % of it at 15 um dry or more; under
painting-tasks-v1 and -v2 in the blue acrylic, ultramarine, >= 95 % of each cell's solids), with no more than 0.5 cm2
of paint over 1 mm outside the line and no marks in the coat.

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria on the task's targets, under the episode's contract. targets(e) gives the target cells:
the square (paper) and the paper beyond 1 mm outside it (outside)."""
import numpy as np

from painting.task import E, _paper_xy


def targets(e):
    ch = e.paint.charts
    out = {}
    if "paper" in ch:
        X, Y = _paper_xy(ch["paper"])
        out["paper"] = (np.abs(X) <= 0.025) & (np.abs(Y) <= 0.025)
        out["outside"] = (np.abs(X) > 0.025 + E["overrun"]) | (np.abs(Y) > 0.025 + E["overrun"])
    return out


def verify(e, settled):
    return e.measure_criteria(settled)

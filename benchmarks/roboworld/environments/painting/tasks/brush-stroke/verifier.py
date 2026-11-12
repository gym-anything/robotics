"""Outcome of brush-stroke: the pencilled S-stroke painted in the red acrylic, its core covered (>= 95 % of the cells
within the template width less 1 mm at 5 um dry or more; under painting-tasks-v1 and -v2 in cadmium red, >= 95 % of
each cell's solids), with no more than 0.5 cm2 of paint over 1 mm outside its outline and no marks in the coat.

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria on the task's targets, under the episode's contract. targets(e) gives the target cells
from the template (S_CURVE and stroke_width in setup.py). COVER_DFT, the dry film a covered cell needs, and ACRYLIC,
the paint the instruction names, are read by the paper coverage check and by the painting-tasks-v1 and -v2 progress."""
import numpy as np

from painting.layout import task_setup
from painting.task import E, _paper_xy

COVER_DFT = E["stroke_dft"]
ACRYLIC = "cadmium_red"
TEMPLATE = task_setup("brush-stroke")
S_CURVE, stroke_width = TEMPLATE.S_CURVE, TEMPLATE.stroke_width


def targets(e):
    ch = e.paint.charts
    out = {}
    if "paper" in ch:
        X, Y = _paper_xy(ch["paper"])
        ts = np.linspace(0, 1, 400)
        c = S_CURVE(ts)
        d = np.full(X.shape, np.inf); half = np.zeros(X.shape)
        for t_, (cx, cy) in zip(ts, c):
            dd = np.hypot(X - cx, Y - cy)
            closer = dd < d
            d = np.where(closer, dd, d)
            half = np.where(closer, stroke_width(t_)/2, half)
        # "within 1 mm of the template width": the painted width may fall short by 1 mm (0.5 mm a side), so
        # coverage is required on the template's core; paint beyond 1 mm outside its edge is overrun (the first
        # build required its exact 1 mm-cell edge, which no stroke reproduces cell for cell)
        out["paper"] = d <= np.maximum(half - 0.5e-3, 0.5e-3)
        out["outside"] = d > half + E["overrun"]
    return out


def verify(e, settled):
    return e.measure_criteria(settled)

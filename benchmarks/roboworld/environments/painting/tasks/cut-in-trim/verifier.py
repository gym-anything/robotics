"""Outcome of cut-in-trim: a band at least 40 mm wide cut in along the left side of the window casing and around the
outlet plate, hidden on >= 95 % of it, without runs, lap marks, marks in the coat or paint on the casing or the plate.

verify(e, settled) receives the episode's task.Evaluator and whether the films have settled, and returns its verdict:
task.Evaluator.measure_criteria on the task's targets, under the episode's contract. targets(e) gives the target cells:
the two bands, CASING_BAND and OUTLET_BAND (x0, x1, z0, z1 in world metres)."""
from painting.layout import CS, OUT, WIN, WX0
from painting.task import _rect_cells

# the bands on the wall chart (x0, x1, z0, z1) in world metres
CASING_BAND = dict(x0=WX0 - CS - 0.04, x1=WX0 - CS, z0=WIN["sill"], z1=WIN["head"] + CS)
OUTLET_BAND = dict(x0=OUT["x"] - OUT["w"]/2 - 0.04, x1=OUT["x"] + OUT["w"]/2 + 0.04, z0=OUT["z"] - OUT["h"]/2 - 0.04,
                   z1=OUT["z"] + OUT["h"]/2 + 0.04)


def targets(e):
    ch = e.paint.charts
    out = {}
    if "wall" in ch:
        w = ch["wall"]
        b, o = CASING_BAND, OUTLET_BAND
        out["wall"] = _rect_cells(w, b["x0"], b["x1"], b["z0"], b["z1"]) | _rect_cells(w, o["x0"], o["x1"], o["z0"], o["z1"])
    return out


def verify(e, settled):
    return e.measure_criteria(settled)

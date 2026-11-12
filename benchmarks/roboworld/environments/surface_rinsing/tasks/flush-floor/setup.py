"""Start of flush-floor: soapy water and debris cover a 1.0 x 1.1 m patch of the can-wash bay floor; the garbage can,
dry and clean, stands in the bay's corner out of the way.

setup(s) adds the can (layout.bay_can, shared with rinse-garbage-can, then parked) and the floor soil to layout.Start
(layout.setup: the bay's floor drain, no fills)."""
from surface_rinsing.layout import BAY, bay_can, yaw_quat


def setup(s):
    items, c_wash = s.items, s.c_wash
    bay_can(s, 0, None)
    items["can"]["pose"] = (6.75, 1.55, 0.0005, *yaw_quat(0.0))
    items["can"]["wet"] = None
    s.floor_soil = dict(region=(BAY["x0"] + 0.3, BAY["x0"] + 1.3, 0.35, 1.45), surf=c_wash, h=0.6e-3, grit=0.02, load=1.0)

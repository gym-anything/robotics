"""Start of rinse-garbage-can: the 32 gal garbage can stands in the can-wash bay, scrubbed with detergent: wet with wash
solution, three foam patches and loosened debris on it.

setup(s) adds it to layout.Start (layout.setup: the bay's floor drain, no fills) with layout.bay_can, which
flush-floor's start shares; the seed moves the can up to 5 cm and turns it."""
from surface_rinsing.layout import bay_can


def setup(s):
    bay_can(s, 3, dict(kind="debris", load=1.0))

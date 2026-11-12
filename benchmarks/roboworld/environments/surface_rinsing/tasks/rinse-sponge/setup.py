"""Start of rinse-sponge: the scrub sponge lies on the back ledge behind the rinse bowl, its pores three-quarters full
of wash solution.

setup(s) adds it to layout.Start (layout.setup: the pot sink's three compartments filled, drains shut) with
layout.ledge_sponge, which scrub-rinse-pan's start shares; the seed moves it up to 3 cm and turns it."""
from surface_rinsing.layout import ledge_sponge


def setup(s):
    ledge_sponge(s)

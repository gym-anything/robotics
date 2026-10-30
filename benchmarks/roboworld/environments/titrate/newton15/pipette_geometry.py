"""Corning 7100-10 dimensional reconstruction; 429 mm, nominal 10 mL.

Manufacturer publishes capacity, class-A tolerance and overall length. Inner
profile and wall are explicit engineering reconstructions. The calibration
mark is derived from the integrated bore volume, not arbitrary artwork.
"""

import numpy as np
from scipy.optimize import brentq

LENGTH_M = .429
MARK_Z_M = .340
NOMINAL_VOLUME_M3 = 10e-6
WALL_M = .0008
ANGULAR_SEGMENTS = 96
FACET_AREA_RATIO = ANGULAR_SEGMENTS*np.sin(2*np.pi/ANGULAR_SEGMENTS)/(2*np.pi)


def interior(bulb_radius):
    return np.array([[.0006, 0.], [.0015, .018], [.0015, .120],
                     [bulb_radius, .148], [bulb_radius, .205],
                     [.002, .235], [.002, LENGTH_M]])


def capacity(profile, height):
    value = 0.
    for (r0, z0), (r1, z1) in zip(profile[:-1], profile[1:]):
        if height <= z0:
            break
        upper = min(height, z1)
        radius = r0+(r1-r0)*(upper-z0)/(z1-z0)
        value += np.pi*(upper-z0)*(r0*r0+r0*radius+radius*radius)/3
    return float(value)


def profile():
    radius = brentq(lambda r: FACET_AREA_RATIO*capacity(interior(r), MARK_Z_M)-NOMINAL_VOLUME_M3, .005, .012, xtol=1e-14)
    inner = interior(radius)
    outer = inner.copy()
    outer[:, 0] += WALL_M
    return np.concatenate((outer, inner[::-1])).tolist()

"""Discrete planar quadrature of the *declared* pair and wall potentials.

At finite support/spacing, continuum kernel moments are not the energy of a
particle lattice. The Young balance subtracts two large surface energies;
even a few percent mismatch changes an acute contact angle substantially.
Compute moments directly, not by fitting a measured drop or changing sigma.
This corrects the axis-aligned planar reference, not all lattice anisotropy.
"""

from functools import lru_cache
import math
import numpy as np
from scipy.integrate import quad
from .capillary import kernel


@lru_cache(maxsize=32)
def moments(spacing, support, repulsive_support, beta):
    if not np.isfinite([spacing, support, repulsive_support, beta]).all() or spacing <= 0 or not 0 <= repulsive_support < support or not 0 <= beta <= 1:
        raise ValueError("Invalid capillary quadrature scales")
    # Dimensionless coordinates avoid very large/small SI powers in summation.
    h, hr = support/spacing, repulsive_support/spacing
    extent = math.ceil(h)
    value = 0.
    for x in range(-extent, extent+1):
        for y in range(-extent, extent+1):
            for z in range(1, extent+1):
                r = math.sqrt(x*x+y*y+z*z)
                value += z*(kernel(r, h)-(beta*kernel(r, hr) if hr else 0.))
    def half_space(z, radius):
        if not 0 <= z < radius:
            return 0.
        return 2*math.pi*quad(lambda r: kernel(r, radius)*(r*r-z*r), z, radius,
                             epsabs=1e-12, epsrel=1e-12)[0]
    wall = sum(half_space(k+.5, h)-(beta*half_space(k+.5, hr) if hr else 0.) for k in range(extent))
    if min(value, wall) <= 0:
        raise ValueError("Discrete potential has no positive planar surface energy")
    return {"liquid_air_m": float(value*spacing), "wall_m": float(wall*spacing),
            "continuum_m": 5*(support-beta*repulsive_support)/48,
            "reference": "axis-aligned simple-cubic volume quadrature at spacing/2 from wall"}

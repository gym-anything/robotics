"""Energy bookkeeping of the bilinear cohesive law shared by the interface bonds, the flesh-bone ties, the rod shaft
ties and the tissue ties (bodies.py, tissue.py).

The law: stiffness k up to the peak force fmax at the opening d0 = fmax / k, then linear softening to zero at dfull.
It is written with a secant damage D (force = (1 - D) k d), which is NOT the fraction of the fracture energy spent:
at a maximum opening u the energy dissipated is fmax dfull (u - d0) / (2 (dfull - d0)), while the secant damage is
D = dfull (u - d0) / (u (dfull - d0)). Booking fmax dfull D / 2 would overstate the dissipation of every partly damaged
link (by up to a factor dfull / u).
"""
import warp as wp


@wp.func
def cohesive_spent(D: float, d0: float, df: float):
    """Fraction of the fracture energy fmax dfull / 2 dissipated by a link at secant damage D."""
    if D <= 0.0:
        return 0.0
    if D >= 1.0:
        return 1.0
    u = d0 * df / (df - D * (df - d0))                 # the maximum opening reached (inverse of D(u))
    return wp.clamp((u - d0) / (df - d0), 0.0, 1.0)

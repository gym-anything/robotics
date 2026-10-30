"""Soil targets mapped to Newton's capped Drucker–Prager stress projector.

For the ascending pressure branch, the installed Newton/Warp tensor mapping gives
sqrt(J2) <= yield_stress + friction * (p + p_t). Thus the compression-meridian
Mohr–Coulomb match uses friction = 6 sin(phi)/(sqrt(3)(3-sin(phi))) and
s = 6 c cos(phi)/(sqrt(3)(3-sin(phi))) - friction*p_t. B15 tests the actual
installed projection kernel; boundary-value shear, tool contact and compaction
still need separate qualification. This mapping is not a direct-shear fit.

Cohesive bank soil starts at h(Jp)=1. After the authored swell, loose soil retains
LOOSE_STRENGTH of the yield surface; it does not lose all cohesion or its cap.
Granular materials have no hardening. Critical packing is a separate numerical
threshold from the authored loose packing. Moisture selects prescribed material
and interface variants; there is no water-flow or consolidation model.
"""

import math
from . import params as P

NAMES = tuple(P.MATERIALS)


def _ps_match(c, phi_deg, mode):
    phi = math.radians(phi_deg)
    if mode == "plane_strain_associated":
        t = math.tan(phi)
        den = math.sqrt(9 + 12*t*t)
        return 3*t/den, 3*c/den
    if mode == "plane_strain_nonassociated":
        return math.sin(phi), c*math.cos(phi)
    if mode == "triaxial_compression":          # the outer (compression-meridian) cone, sqrt(J2) = alpha p + k
        s = math.sin(phi)
        return 6*s/(math.sqrt(3)*(3 - s)), 6*c*math.cos(phi)/(math.sqrt(3)*(3 - s))
    raise ValueError(mode)


def bank_fraction(name):
    """Packing fraction of the bank (undisturbed) state."""
    sw = P.MATERIALS[name]["swell"]
    return min(1.0, P.CRITICAL_FRACTION*(1.0 + sw))


GRANULAR = ("sand", "gravel", "mulch")


def hardening(name):
    """(H, J_p bank, J_p loose). Cohesive soils: h(J_p) = sinh(-H ln J_p) is 1 in the bank and LOOSE_STRENGTH after
    the bank volume dilated by the swell ratio (J_p loose = J_p bank (1 + swell)): loosened soil keeps its friction cone
    up to a small cap (Newton's friction branch ends at half the cap, so h -> 0 would leave loose soil with no friction
    at all: B2's first sand heap slumped to 7 deg). Granular materials (sand, gravel, mulch) have no structure to lose:
    H = 0, the bank yield surface throughout, J_p irrelevant (1)."""
    if name in GRANULAR:
        return 0.0, 1.0, 1.0
    sw = P.MATERIALS[name]["swell"]
    d = math.log(1.0 + sw)
    H = (math.asinh(1.0) - math.asinh(P.LOOSE_STRENGTH))/d
    e0 = math.asinh(1.0)/H
    J0 = math.exp(-e0)
    return H, J0, min(1.0, J0*(1.0 + sw))


def jp0(name):
    """In-situ (bank) plastic volume ratio J_p."""
    return hardening(name)[1]


def jp_loose(name):
    return hardening(name)[2]


def solver_params(name, moisture="moist", law=None, override=None):
    """Per-particle MPM attributes (mpm:*), the bank J_p0 and the bank packing fraction for one material."""
    m = dict(P.MATERIALS[name])
    if override:
        m.update(override)
    law = law or P.YIELD_LAW
    f = P.MOISTURE_COHESION[m["water"]][moisture]
    c, pt = m["c"]*f, m["pt"]*f
    if law["mode"] == "compression_meridian":
        mu, k = _ps_match(c, m["phi"], "triaxial_compression")
        s = max(0.0, k - mu*pt)
    elif law["mode"] == "direct_shear":
        # Historical, unqualified empirical option, retained for comparison (B1): tan phi_ds = KAPPA_DS mu, c_ds = KAPPA_DS_C (s + B mu p_t)
        mu = math.tan(math.radians(m["phi"]))/P.KAPPA_DS
        s = max(0.0, c/P.KAPPA_DS_C - law["B"]*mu*pt)
    else:
        alpha, k = _ps_match(c, m["phi"], law["mode"])
        mu = math.sqrt(3)*alpha/law["A"]
        if name in GRANULAR:
            mu = math.tan(math.radians(m["phi"]))/P.KAPPA_DS
        s = max(0.0, math.sqrt(3)*k - law["B"]*mu*pt)
    H, J0, Jl = hardening(name)
    attrs = dict(young_modulus=min(m["E"], P.E_CAP), poisson_ratio=m["nu"], damping=0.0, friction=mu, yield_pressure=m["Y"],
                 tensile_yield_ratio=pt/m["Y"], yield_stress=s, hardening=H, hardening_rate=P.HARDENING_RATE,
                 softening_rate=P.SOFTENING_RATE, dilatancy=m["dil"], viscosity=0.0)
    attrs.update(m.get("solver", {}))           # bench overrides of solver attributes (probes only)
    return dict(attrs=attrs, jp0=J0, jp_loose=Jl, fraction=bank_fraction(name), rho=m["rho"], targets=dict(c=c, phi=m["phi"], pt=pt))


def describe(moisture="moist"):
    rows = []
    for n in NAMES:
        sp = solver_params(n, moisture)
        a = sp["attrs"]
        rows.append(dict(material=n, rho=sp["rho"], c_target=sp["targets"]["c"], phi_target=sp["targets"]["phi"],
                         mu=round(a["friction"], 4), s=round(a["yield_stress"], 1), Y=a["yield_pressure"],
                         r_t=round(a["tensile_yield_ratio"], 5), H=round(a["hardening"], 3), jp0=round(sp["jp0"], 4),
                         bank_fraction=round(sp["fraction"], 4)))
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=1))


def contact_normal_offset(mu, shear_adhesion):
    """Convert tau_max = mu*sigma_n + c_a to Newton's shifted normal cone.

    This fixes the traction convention only. Rasterized nodal contact area is
    separately measured; no area correction is fitted into this conversion.
    A frictionless adhesive interface cannot be represented by this cone.
    """
    if shear_adhesion == 0:
        return 0.0
    if mu <= 0:
        raise ValueError("Nonzero shear adhesion requires positive Coulomb friction")
    return float(shear_adhesion)/float(mu)

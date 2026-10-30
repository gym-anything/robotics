"""Cutting laws: thrust, torque and feed per revolution of a rotating bit, and the
percussive removal per blow, for the materials of this environment. Units at this boundary: d and f in mm and
mm/rev (the machining units the laws were published in), forces in N, torques in N m, energies in J.

Metal - the Shaw-Oxford form printed in the Guide to Machinery's Handbook (27th ed., pp. 200-201), in SI:
    T = 7.294 Kd B W (f d)^0.8 lam_T + Kd d^2 J W chi            [N]
    M = 0.003647 Kd A W f^0.8 d^1.8 lam_M                        [N m]
  The second thrust term is the chisel edge's feed-independent indentation; under force control it is a threshold
  below which the drill rubs without advancing. A pilot at least as wide as the chisel edge removes it (chi = 0) and
  the lips scale with the engaged annulus (lam_T = (d - dp)/d, lam_M = (d^2 - dp^2)/d^2). A, B and J are MH Table 7's
  row for c/d = 0.18, the chisel length MH prescribes for drills of standard design and the one used here.
  Exit: the cap under the point collapses at h = sqrt(2T/(pi sigma_y)); the burr is BURR_CAP_FRACTION of it.
  Heat: the cutting corners run above the lumped tip by Cook's interface rise 0.4 (u/rho C) (V t0/K)^(1/3).
Wood - the measured MDF form (Szwajka 2019, 10 mm), linear in feed with an intercept, scaled by diameter and kappa:
    T = kappa (13.0 + 66.7 f) (d/10)      M = kappa (0.093 + 0.747 f) (d/10)^2
Concrete - percussive: the carbide head is one rigid body, so a delivered blow advances its whole face by the same
  depth delta and spends E_blow = delta A sum_i phi_i e_v,i on the phases under it; the volume removed is
  dV = delta A sum_i phi_i = E_blow sum_i phi_i / sum_i phi_i e_v,i (e_v from measured ROP and the tool's rated
  energy, so transfer and crushing losses are inside it). A per-phase rule, E_blow sum_i phi_i/e_v,i, lets each phase
  absorb energy on its own, so a face 1/7 on steel still advances ~38 um per 2.1 J blow through the mortar beside it.
"""

import math
import numpy as np

from . import params as P

METALS = {
    "steel": dict(Kd=P.KD_STEEL, yield_mpa=P.STEEL_YIELD_MPA, hardness=P.STEEL_HARDNESS_PA, density=7850.,
                  rho_c=P.STEEL_RHO_C, diffusivity=P.STEEL_DIFFUSIVITY),
    "aluminium": dict(Kd=P.KD_ALUMINIUM, yield_mpa=P.ALU_YIELD_MPA, hardness=P.ALU_HARDNESS_PA, density=2700.,
                      rho_c=P.ALU_RHO_C, diffusivity=P.ALU_DIFFUSIVITY),
}
# a wood's law: kappa scales the MDF fit as a whole; tf and mf scale its thrust and torque slopes (solid softwood:
# params.WOOD_SOLID_TF / WOOD_SOLID_MF, from the one solid-wood datum)
WOODS = {"douglas_fir": dict(kappa=P.KAPPA_WOOD, tf=P.WOOD_SOLID_TF, mf=P.WOOD_SOLID_MF, density=P.WOOD_DENSITY),
         "plywood": dict(kappa=1.2, tf=1., mf=1., density=600.)}


def chisel_length(d_mm):
    return P.CHISEL_FRACTION*d_mm


def point_height(d_mm, point_angle=P.POINT_ANGLE):
    """Height of the drill's conical point (mm): (d/2)/tan(theta/2) = 0.30 d at 118 deg."""
    return 0.5*d_mm/math.tan(0.5*point_angle)


def engagement(d_mm, dp_mm=0., depth_mm=None, exit_mm=None, point_angle=P.POINT_ANGLE):
    """(lam_T, lam_M, chi): engaged fraction of the lips for thrust and torque and whether the chisel cuts.
    depth_mm: how far the tip is below the entry (None: fully engaged); exit_mm: material left under the tip (None:
    blind). The cone engages radius r = depth tan(theta/2) at entry; at exit the inner radius r_in = (depth -
    thickness) tan(theta/2) has left the material."""
    R = 0.5*d_mm
    r_in = 0.5*dp_mm
    tan_h = math.tan(0.5*point_angle)
    r_out = R if depth_mm is None else min(R, max(0., depth_mm)*tan_h)
    if exit_mm is not None and exit_mm < 0.:
        r_in = max(r_in, min(R, -exit_mm*tan_h))
    if r_out <= r_in:
        return 0., 0., 0.
    lam_t = (r_out - r_in)/R
    lam_m = (r_out**2 - r_in**2)/R**2
    c_half = 0.5*chisel_length(d_mm)
    chi = 1. if (dp_mm < chisel_length(d_mm) and (exit_mm is None or exit_mm > 0.) and r_in < c_half) else 0.
    return lam_t, lam_m, chi


def metal_thrust(material, d_mm, f, W=1., lam_t=1., chi=1.):
    m = METALS[material]
    return 7.294*m["Kd"]*P.MH_B*W*(max(f, 0.)*d_mm)**0.8*lam_t + m["Kd"]*d_mm**2*P.MH_J*W*chi


def metal_torque(material, d_mm, f, W=1., lam_m=1.):
    m = METALS[material]
    return 0.003647*m["Kd"]*P.MH_A*W*max(f, 0.)**0.8*d_mm**1.8*lam_m


def metal_feed(material, d_mm, thrust, W=1., lam_t=1., chi=1.):
    """Feed per revolution (mm/rev) the thrust supports; 0 at or below the chisel threshold."""
    m = METALS[material]
    threshold = m["Kd"]*d_mm**2*P.MH_J*W*chi
    if thrust <= threshold or lam_t <= 1e-9:
        return 0.
    coef = 7.294*m["Kd"]*P.MH_B*W*lam_t
    return ((thrust - threshold)/coef)**1.25/d_mm


def feed_for_thrust(thrust_of, thrust, f_max):
    """Feed per revolution (mm/rev, at most f_max) at which a rising thrust law thrust_of(f) equals `thrust`; 0 when
    the law already asks more at no feed (a threshold). Used where several parts are cut at once (a stack)."""
    if thrust <= thrust_of(0.):
        return 0.
    if thrust_of(f_max) <= thrust:
        return f_max
    lo, hi = 0., f_max
    for _ in range(50):
        mid = 0.5*(lo + hi)
        if thrust_of(mid) < thrust:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


def _wood(k):
    """(kappa, thrust-slope factor, torque-slope factor) of a wood law given as a number or as a WOODS entry."""
    if isinstance(k, dict):
        return k["kappa"], k.get("tf", 1.), k.get("mf", 1.)
    return k, 1., 1.


def wood_thrust(kappa, d_mm, f, lam_t=1.):
    k, tf, _ = _wood(kappa)
    return k*(P.WOOD_T0 + tf*P.WOOD_TF*max(f, 0.))*(d_mm/10.)*lam_t


def wood_torque(kappa, d_mm, f, lam_m=1.):
    k, _, mf = _wood(kappa)
    return k*(P.WOOD_M0 + mf*P.WOOD_MF*max(f, 0.))*(d_mm/10.)**2*lam_m


def wood_feed(kappa, d_mm, thrust, lam_t=1.):
    k, tf, _ = _wood(kappa)
    T0 = k*P.WOOD_T0*(d_mm/10.)*lam_t
    if thrust <= T0 or lam_t <= 1e-9:
        return 0.
    return (thrust - T0)/(k*tf*P.WOOD_TF*(d_mm/10.)*lam_t)


def tearout_threshold(remaining_m, Gc=P.WOOD_GC, E=P.WOOD_ET, nu=P.WOOD_NU):
    """Hocheng-Dharan critical push-out thrust (N) of the unsupported layer left under the drill."""
    h = max(remaining_m, 0.)
    return math.pi*math.sqrt(8.*Gc*E*h**3/(3.*(1. - nu*nu)))


def cap_collapse_thickness(thrust, yield_mpa):
    """Remaining thickness (m) below which a centrally loaded rigid-plastic plate cap yields: P_c = (pi/2) sigma_y h^2."""
    return math.sqrt(2.*max(thrust, 0.)/(math.pi*yield_mpa*1e6))


def exit_burr_height(cap_thickness_m):
    """Exit burr height (m) of an unbacked metal exit whose cap collapsed at cap_thickness_m: the cap is pushed out as
    the drill cap and the rim it leaves stands BURR_CAP_FRACTION of its thickness high (calibrated on St37 data)."""
    return P.BURR_CAP_FRACTION*max(cap_thickness_m, 0.)


def metal_specific_energy(material, d_mm, f, W=1.):
    """Cutting energy per removed volume (J/mm^3) the law implies at feed f: M 2 pi / (pi d^2 f/4)."""
    f = max(f, 1e-4)
    return metal_torque(material, d_mm, f, W)*2.*math.pi/(0.25*math.pi*d_mm*d_mm*f)


def edge_temperature_rise(material, d_mm, f, v_corner_m_s, W=1., point_angle=P.POINT_ANGLE):
    """Mean tool-chip interface temperature rise (K) at the lips' outer corners, Cook's formula
    dT = 0.4 (u/rho C) (V t0/K)^(1/3) with V the corner speed, t0 = (f/2) sin(theta/2) the uncut chip thickness of a
    lip and u the law's specific energy; 0 when no chip is cut. It rises with cutting speed as V^(1/3)."""
    if f <= 0. or v_corner_m_s <= 0.:
        return 0.
    m = METALS[material]
    t0 = 0.5*f*math.sin(0.5*point_angle)*1e-3
    u = metal_specific_energy(material, d_mm, f, W)
    return P.COOK_COEF*u/m["rho_c"]*(v_corner_m_s*t0/m["diffusivity"])**(1./3.)


def concrete_ev(d_mm):
    """Blow energy per volume of concrete removed (J/mm^3) for a bit of diameter d_mm: the maker's rated drilling
    performance by diameter (params.CONCRETE_EV_BY_D), linear between its diameters, constant outside them."""
    tab = P.CONCRETE_EV_BY_D
    return float(np.interp(d_mm, [t[0] for t in tab], [t[1] for t in tab]))


def blow_removal_mm3(energy_j, fractions, d_mm):
    """Volume (mm^3) one delivered blow of a d_mm bit removes; fractions: {'mortar': phi, 'aggregate': phi, 'steel': phi}.
    Rigid-face rule: the whole face advances one depth, E = delta A sum(phi e_v), so dV = E sum(phi)/sum(phi e_v)."""
    e_c = concrete_ev(d_mm)
    ev = {"mortar": e_c, "aggregate": e_c*P.AGGREGATE_EV_FACTOR, "steel": P.STEEL_EV}
    phi_sum = sum(phi for phi in fractions.values() if phi > 0.)
    work = sum(phi*ev[k] for k, phi in fractions.items() if phi > 0.)
    return energy_j*phi_sum/work if work > 0. else 0.


def rubbing_torque(thrust, d_mm):
    """Torque (N m) of a chisel edge rubbing without cutting: mu T c/4 (uniform line pressure, mean radius c/4)."""
    return P.RUB_MU*thrust*chisel_length(d_mm)*1e-3/4.


def bit_lateral_stiffness(d_mm, stickout_m):
    I = P.FLUTE_I_FACTOR*math.pi*(d_mm*1e-3)**4/64.
    return 3.*P.HSS_E*I/max(stickout_m, 0.005)**3


def bit_torsion_limit(d_mm):
    """Torque (N m) at which a fluted HSS drill shears: tau_u x 0.3 of the solid polar section modulus."""
    return P.HSS_SHEAR_ULT*0.3*math.pi*(d_mm*1e-3)**3/16.


def bit_bending_limit(d_mm):
    """Bending moment (N m) at which a fluted HSS drill breaks: sigma_b I_eff/(d/2)."""
    I = P.FLUTE_I_FACTOR*math.pi*(d_mm*1e-3)**4/64.
    return P.HSS_BEND_STRENGTH*I/(0.5*d_mm*1e-3)

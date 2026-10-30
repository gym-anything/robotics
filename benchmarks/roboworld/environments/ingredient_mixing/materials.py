"""What a material point is made of and how that sets its mechanics.

A point carries the mass of each species. Everything mechanical about it (rest volume, stiffness, yield surface,
consistency, heat capacity) is recomputed from those masses, its temperature and a short history every REFRESH seconds
by `k_refresh` below; nothing is typed in per point. The same kernel advances the slow chemistry of the point: dry
solids becoming wetted lumps, lumps eroding into suspension under stress, cold-set starch setting, starch gelatinising
with temperature.

Volume rule.  V0 = V_s + max(V_pore, V_w): solids, plus whichever is larger of the pore space the dry solids would
hold when poured (void ratio e_k of each solid) and the wet phase (water, fat, finished pastes). Liquid first fills
pores, then adds volume. Saturation S = min(1, V_w / V_pore).

Mechanical state.  S < 0.9: a granular material, Drucker-Prager with friction mu and a cohesion that has a capillary
peak at S = 0.6. S -> 1: a paste, Herschel-Bulkley. The two are blended over 0.9 < S < 1 so no property jumps.

Every number here has its provenance label in the parameter ledger (params.py, ledger_mixture.py).
"""

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import warp as wp

SPECIES = ("aqueous", "fat", "sugar", "minor", "solid_dry", "solid_lump", "solid_susp", "solid_cooked",
           "paste_a", "paste_b", "paste_c")
AQ, FAT, SUGAR, MINOR, SOLID_D, SOLID_L, SOLID_S, SOLID_C, PASTE_A, PASTE_B, PASTE_C = range(11)
NS = 11
ST_TWET, ST_SET, ST_WORK, ST_DRYT, ST_CHAR, ST_PEAK = range(6)   # time since wetted (s), set fraction, work on wetted
NSTATE = 6                                              # solids (J/kg), ..., burnt-on film scraped into the point (0..1,
                                                        # heat.py), highest temperature the point has had on the hob (C;
                                                        # 0 until the hob's first step)
GAS_R = 8.314
REFRESH = 0.010                                    # s between property refreshes and exchanges


@dataclass(frozen=True)
class Solid:
    """The hydrating insoluble solid of a recipe (one per preset)."""
    name: str = "wheat flour"
    rho: float = 1470.0            # particle density
    void: float = 1.88             # void ratio poured (flour: 510 kg/m3 bulk)
    bound: float = 0.9             # liquid a WETTED solid holds, as a void ratio: flour takes up about 60 % of its mass
                                   # in water (its absorption in a dough), 0.88 ml per ml of solid. The rest of a
                                   # batter's liquid is free and wets the dry flour it meets. (If a wetted solid held
                                   # its poured pore volume, 1.88, a muffin batter, 1.4 g of liquid per gram of flour,
                                   # would have no free liquid once mixed, and the last sixth of the flour would stay
                                   # dry whatever the folding: E15 muffin.)
    grain: float = 90e-6
    friction_deg: float = 37.0
    cohesion: float = 80.0         # Pa, dry
    cap_cohesion: float = 500.0    # Pa, peak of the capillary cohesion at S = 0.6
    law_a: float = 12.2            # log10 K = a x + b, K in Pa s^n, x the wetted solid mass fraction
    law_b: float = -3.4
    n_slope: float = 1.21          # n = 1 - n_slope x
    n_min: float = 0.30
    yield_x0: float = 0.30
    yield_x1: float = 0.61
    yield_max: float = 300.0
    hyd_gain: float = 0.5          # K x (1 + gain (1 - exp(-t_wet / hyd_tau)))
    hyd_tau: float = 300.0
    work_gain: float = 1.0         # K x (1 + gain (1 - exp(-w / work_scale))); 0 scale: off
    work_scale: float = 0.0
    set_tau: float = 0.0           # cold-set starch: time constant (s); 0: does not set
    set_k: float = 20.0
    set_n: float = 0.40
    set_yield: float = 30.0
    set_xref: float = 0.1745
    starch: float = 0.70           # starch mass fraction of the solid (gelatinisation)
    gel_a: float = 8.8e20          # 1/s
    gel_ea: float = 150e3          # J/mol
    gel_b: float = 2.56e4          # K_cooked = b x_c^m, x_c the gelatinised starch mass fraction
    gel_m: float = 2.73
    gel_n: float = 0.45


@dataclass(frozen=True)
class Paste:
    """A finished paste with its own Herschel-Bulkley law tau = yield + k gdot^n."""
    name: str = "none"
    rho: float = 1000.0
    yield_pa: float = 0.0
    k: float = 1.0
    n: float = 1.0
    cp: float = 3500.0
    color: tuple = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class Recipe:
    """Material constants of one preset's mixture."""
    solid: Solid = Solid()
    pastes: tuple = (Paste(), Paste(), Paste())
    rho_aq: float = 1030.0
    k_aq: float = 2.0e-3           # Pa s, the aqueous liquid at 25 C
    rho_fat: float = 920.0
    rho_sugar: float = 1590.0
    void_sugar: float = 0.87       # 850 kg/m3 poured
    rho_minor: float = 2200.0
    void_minor: float = 2.4        # 650 kg/m3 poured: baking powder 500-600, salt about 1200, 12 g to 3 g
    cp: tuple = (3900.0, 2000.0, 1250.0, 1000.0, 1800.0, 1800.0, 1800.0, 1800.0)   # aq, fat, sugar, minor, solid x 4
    visc_ea: float = 18.4e3        # J/mol: K(T) = K exp(Ea/R (1/T - 1/298))
    lump_ke: float = 0.05          # lump erosion per unit strain above the cohesive stress
    lump_tauc: float = 20.0        # Pa
    wire_strain: float = 2.0       # strain a wire's boundary layer puts on the material it sweeps
    d_mol: float = 5.2e-10         # m2/s, dissolved species
    wet_d: float = 2.0e-6          # m2/s, liquid into dry powder at the scale of a point; calibrated on the muffin
    #                                method's strokes (ledger_mixture)
    conductivity: float = 0.56     # W/m/K
    sound: float = 6.0             # m/s, numerical
    g_paste: float = 2000.0        # Pa, numerical
    g_powder: float = 4000.0       # Pa, numerical (elastic strain of a heap at yield under 1 %..5 %)
    k_powder: float = 6200.0       # Pa: flour 510 -> 880 kg/m3 under 4.5 kPa
    k_dry: float = 0.05            # Pa s, rate term of flowing powder (numerical)
    tau_rest: float = 1.0          # Pa, numerical rest stress of every wet state (E4: without it a thin liquid never comes to rest)

    def as_dict(self):
        return asdict(self)


@wp.struct
class Law:
    rho: wp.array(dtype=float)
    voidr: wp.array(dtype=float)      # void ratio poured, per species
    cp: wp.array(dtype=float)
    paste_yield: wp.vec3
    paste_k: wp.vec3
    paste_n: wp.vec3
    k_aq: float
    mu_dp: float
    cohesion: float
    cap_cohesion: float
    law_a: float
    law_b: float
    n_slope: float
    n_min: float
    yield_x0: float
    yield_x1: float
    yield_max: float
    hyd_gain: float
    hyd_tau: float
    work_gain: float
    work_scale: float
    set_tau: float
    set_k: float
    set_n: float
    set_yield: float
    set_xref: float
    starch: float
    gel_a: float
    gel_ea: float
    gel_b: float
    gel_m: float
    gel_n: float
    visc_ea: float
    lump_ke: float
    lump_tauc: float
    wire_strain: float
    d_mol: float
    grain: float
    sound: float
    g_paste: float
    g_powder: float
    k_powder: float
    k_dry: float
    tau_rest: float


def make_law(recipe, device):
    s = recipe.solid
    law = Law()
    rho = [recipe.rho_aq, recipe.rho_fat, recipe.rho_sugar, recipe.rho_minor, s.rho, s.rho, s.rho, s.rho] \
        + [p.rho for p in recipe.pastes]
    # the liquid each solid state holds, as a void ratio: dry powder its poured pores; wetted states what they bind
    void = [0.0, 0.0, recipe.void_sugar, recipe.void_minor, s.void, s.bound, s.bound, s.bound, 0.0, 0.0, 0.0]
    cp = list(recipe.cp) + [p.cp for p in recipe.pastes]
    with wp.ScopedDevice(device):
        law.rho = wp.array(np.array(rho, np.float32), dtype=float)
        law.voidr = wp.array(np.array(void, np.float32), dtype=float)
        law.cp = wp.array(np.array(cp, np.float32), dtype=float)
    law.paste_yield = wp.vec3(*[p.yield_pa for p in recipe.pastes])
    law.paste_k = wp.vec3(*[p.k for p in recipe.pastes])
    law.paste_n = wp.vec3(*[p.n for p in recipe.pastes])
    law.k_aq = recipe.k_aq
    law.mu_dp = math.sin(math.radians(s.friction_deg))
    law.cohesion = s.cohesion*math.cos(math.radians(s.friction_deg))
    law.cap_cohesion = s.cap_cohesion
    for k in ("law_a", "law_b", "n_slope", "n_min", "yield_x0", "yield_x1", "yield_max", "hyd_gain", "hyd_tau",
              "work_gain", "work_scale", "set_tau", "set_k", "set_n", "set_yield", "set_xref", "starch", "gel_a",
              "gel_ea", "gel_b", "gel_m", "gel_n", "grain"):
        setattr(law, k, float(getattr(s, k)))
    for k in ("visc_ea", "lump_ke", "lump_tauc", "wire_strain", "d_mol", "sound", "g_paste", "g_powder", "k_powder",
              "k_dry", "tau_rest"):
        setattr(law, k, float(getattr(recipe, k)))
    return law


@wp.func
def smooth(a: float, b: float, x: float):
    t = wp.clamp((x - a)/(b - a), 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)


@wp.func
def blend_f(h: float):
    """Exponent of the consistency in the sauce-blending map of Nagasawa et al. 2019, inside its range 0.4 < h < 1."""
    hc = wp.clamp(h, 0.4, 1.0)
    return 0.12 - 3.2*hc + 2.7*hc*hc


@wp.kernel
def k_refresh(spec: wp.array2d(dtype=float), temp: wp.array(dtype=float), state: wp.array2d(dtype=float),
              flags: wp.array(dtype=int), gdot: wp.array(dtype=float), expo: wp.array(dtype=float), law: Law,
              dt: float, work: wp.array(dtype=float), mass: wp.array(dtype=float), vol0: wp.array(dtype=float),
              kbulk: wp.array(dtype=float), shear: wp.array(dtype=float), cons: wp.array(dtype=float),
              nidx: wp.array(dtype=float), tauy: wp.array(dtype=float), mudp: wp.array(dtype=float),
              wet: wp.array(dtype=float), heatcap: wp.array(dtype=float), dmicro: wp.array(dtype=float),
              dgrain: wp.array(dtype=float), free_liq: wp.array(dtype=float), dry_pore: wp.array(dtype=float),
              vliq: wp.array(dtype=float)):
    p = wp.tid()
    if flags[p] == 4:
        return
    m = float(0.0)
    vs = float(0.0)                      # solids
    vpore = float(0.0)
    hc = float(0.0)
    for s in range(NS):
        ms = spec[p, s]
        m = m + ms
        hc = hc + ms*law.cp[s]
        if s >= 2 and s <= 7:
            vs = vs + ms/law.rho[s]
            vpore = vpore + law.voidr[s]*ms/law.rho[s]
    vw = spec[p, 0]/law.rho[0] + spec[p, 1]/law.rho[1] + spec[p, 8]/law.rho[8] + spec[p, 9]/law.rho[9] \
        + spec[p, 10]/law.rho[10]
    sat = float(1.0)
    if vpore > 1.0e-18:
        sat = wp.min(1.0, vw/vpore)
    tk = temp[p] + 273.15
    # ---- chemistry of the point over dt (dt = 0: properties only)
    if dt > 0.0:
        if sat > 0.5 and spec[p, 4] > 0.0:                 # dry solid in a wetted point: agglomerated, not dispersed
            spec[p, 5] = spec[p, 5] + spec[p, 4]
            spec[p, 4] = 0.0
        tau = tauy[p] + cons[p]*wp.pow(wp.max(gdot[p], 1.0e-6), nidx[p])
        if spec[p, 5] > 0.0:
            keep = float(1.0)
            if tau > law.lump_tauc:
                keep = wp.exp(-law.lump_ke*gdot[p]*(1.0 - law.lump_tauc/tau)*dt)
            keep = keep*wp.exp(-law.lump_ke*law.wire_strain*expo[p])
            moved = spec[p, 5]*(1.0 - keep)
            spec[p, 5] = spec[p, 5] - moved
            spec[p, 6] = spec[p, 6] + moved
        wetted = spec[p, 5] + spec[p, 6]
        if wetted > 0.0 and sat > 0.5:
            kg = law.gel_a*wp.exp(-law.gel_ea/(8.314*tk))
            frac = 1.0 - wp.exp(-kg*dt)
            c5 = spec[p, 5]*frac
            c6 = spec[p, 6]*frac
            spec[p, 5] = spec[p, 5] - c5
            spec[p, 6] = spec[p, 6] - c6
            spec[p, 7] = spec[p, 7] + c5 + c6
        if sat > 0.5 and (spec[p, 5] + spec[p, 6] + spec[p, 7]) > 0.0:
            state[p, 0] = state[p, 0] + dt
            if law.set_tau > 0.0:
                state[p, 1] = state[p, 1] + (1.0 - state[p, 1])*(1.0 - wp.exp(-dt/law.set_tau))
            state[p, 2] = state[p, 2] + work[p]
        work[p] = 0.0
    # ---- volume
    v0 = vs + wp.max(vpore, vw)
    mass[p] = m
    vol0[p] = v0
    heatcap[p] = hc
    wet[p] = sat
    free_liq[p] = wp.max(vw - vpore, 0.0)
    dry_pore[p] = wp.max(vpore - vw, 0.0)
    vliq[p] = vw
    dmicro[p] = law.d_mol
    dgrain[p] = law.grain
    if m <= 0.0 or v0 <= 0.0:
        return
    rho0 = m/v0
    # ---- the batter phase: water, fat, sugar, minor and the hydrating solid
    mb = spec[p, 0] + spec[p, 1] + spec[p, 2] + spec[p, 3] + spec[p, 4] + spec[p, 5] + spec[p, 6] + spec[p, 7]
    kb = law.k_aq
    nb = float(1.0)
    yb = float(0.0)
    if mb > 1.0e-15:
        xf = (spec[p, 4] + spec[p, 5] + spec[p, 6])/mb
        xw = (spec[p, 5] + spec[p, 6])/mb
        xc = law.starch*spec[p, 7]/mb
        kb = wp.max(law.k_aq, wp.pow(10.0, law.law_a*wp.min(xf, law.yield_x1) + law.law_b))
        nb = wp.clamp(1.0 - law.n_slope*xf, law.n_min, 1.0)
        yr = wp.clamp((xf - law.yield_x0)/(law.yield_x1 - law.yield_x0), 0.0, 1.0)
        yb = law.yield_max*yr*yr
        if xf > 0.02:
            kb = kb*(1.0 + law.hyd_gain*(1.0 - wp.exp(-state[p, 0]/law.hyd_tau)))
            if law.work_scale > 0.0:
                kb = kb*(1.0 + law.work_gain*(1.0 - wp.exp(-state[p, 2]/law.work_scale)))
        if law.set_tau > 0.0 and xw > 0.0:
            h = state[p, 1]
            rel = wp.pow(xw/law.set_xref, 2.7)
            ks = h*law.set_k*rel
            nb = (kb*nb + ks*law.set_n)/(kb + ks)
            kb = kb + ks
            yb = wp.min(yb + h*law.set_yield*rel, law.yield_max)
        if xc > 0.0:
            kc = law.gel_b*wp.pow(xc, law.gel_m)
            nb = (kb*nb + kc*law.gel_n)/(kb + kc)
            kb = kb + kc
    # ---- finished pastes: the blending map over mass fractions (identity when there is one component)
    mp = spec[p, 8] + spec[p, 9] + spec[p, 10]
    kk = kb
    nn = nb
    yy = yb
    if mp > 1.0e-15:
        ab = mb/m
        ls = ab*wp.log(wp.max(yb, 0.01))
        hh = ab*wp.pow(nb, 1.4)
        ee = ab*wp.pow(kb, blend_f(nb))
        for j in range(3):
            aj = spec[p, 8 + j]/m
            if aj > 0.0:
                ls = ls + aj*wp.log(wp.max(law.paste_yield[j], 0.01))
                hh = hh + aj*wp.pow(law.paste_n[j], 1.4)
                ee = ee + aj*wp.pow(law.paste_k[j], blend_f(law.paste_n[j]))
        nn = wp.pow(hh, 1.0/1.4)
        kk = wp.pow(ee, 1.0/blend_f(nn))
        yy = wp.exp(ls)
        if yy < 0.011:
            yy = 0.0
    kk = kk*wp.exp(law.visc_ea/8.314*(1.0/tk - 1.0/298.15))
    # ---- granular below saturation, paste at saturation
    wp_ = smooth(0.9, 1.0, sat)
    sb = wp.clamp((sat - 0.2)/0.8, 0.0, 1.0)
    sn = wp.sin(3.14159265*sb)
    coh = law.cohesion + law.cap_cohesion*sn*sn
    tauy[p] = wp_*wp.max(yy, law.tau_rest) + (1.0 - wp_)*coh
    mudp[p] = (1.0 - wp_)*law.mu_dp
    cons[p] = wp_*kk + (1.0 - wp_)*law.k_dry
    nidx[p] = wp_*nn + (1.0 - wp_)
    shear[p] = wp_*law.g_paste + (1.0 - wp_)*law.g_powder
    kbulk[p] = law.k_powder + (rho0*law.sound*law.sound - law.k_powder)*smooth(0.8, 1.0, sat)


@wp.kernel
def k_gather_expo(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), gexpo: wp.array(dtype=float), dx: float,
                  origin: wp.vec3, ny: int, nz: int, expo: wp.array(dtype=float)):
    p = wp.tid()
    expo[p] = 0.0
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    e = float(0.0)
    for i in range(3):
        wi = 0.5*(1.5 - fx[0])*(1.5 - fx[0])
        if i == 1:
            wi = 0.75 - (fx[0] - 1.0)*(fx[0] - 1.0)
        if i == 2:
            wi = 0.5*(fx[0] - 0.5)*(fx[0] - 0.5)
        for j in range(3):
            wj = 0.5*(1.5 - fx[1])*(1.5 - fx[1])
            if j == 1:
                wj = 0.75 - (fx[1] - 1.0)*(fx[1] - 1.0)
            if j == 2:
                wj = 0.5*(fx[1] - 0.5)*(fx[1] - 0.5)
            for k in range(3):
                wk = 0.5*(1.5 - fx[2])*(1.5 - fx[2])
                if k == 1:
                    wk = 0.75 - (fx[2] - 1.0)*(fx[2] - 1.0)
                if k == 2:
                    wk = 0.5*(fx[2] - 0.5)*(fx[2] - 0.5)
                e = e + wi*wj*wk*gexpo[((bx + i)*ny + (by + j))*nz + (bz + k)]
    expo[p] = e


# ---------------------------------------------------------------------------------------------- host-side mirrors
def batter_law(solid, x, k_aq=2.0e-3):
    """(K, n, yield) of the uncooked, unrested batter phase at wetted-solid mass fraction x."""
    k = max(k_aq, 10.0**(solid.law_a*min(x, solid.yield_x1) + solid.law_b))
    n = min(1.0, max(solid.n_min, 1.0 - solid.n_slope*x))
    y = solid.yield_max*min(1.0, max(0.0, (x - solid.yield_x0)/(solid.yield_x1 - solid.yield_x0)))**2
    return k, n, y


def blend(parts):
    """Nagasawa et al. blending of Herschel-Bulkley materials. parts: [(mass fraction, yield, k, n)]."""
    f = lambda h: 0.12 - 3.2*min(1.0, max(0.4, h)) + 2.7*min(1.0, max(0.4, h))**2
    ls = sum(a*math.log(max(y, 0.01)) for a, y, k, n in parts)
    hh = sum(a*n**1.4 for a, y, k, n in parts)
    n = hh**(1/1.4)
    ee = sum(a*k**f(nn) for a, y, k, nn in parts)
    y = math.exp(ls)
    return (0.0 if y < 0.011 else y), ee**(1/f(n)), n


def gel_rate(solid, temp_c):
    return solid.gel_a*math.exp(-solid.gel_ea/(GAS_R*(temp_c + 273.15)))


def rest_volume(recipe, masses):
    """Rest volume (m3) and saturation of a composition {species index: kg} by the volume rule."""
    s = recipe.solid
    rho = {AQ: recipe.rho_aq, FAT: recipe.rho_fat, SUGAR: recipe.rho_sugar, MINOR: recipe.rho_minor, SOLID_D: s.rho,
           SOLID_L: s.rho, SOLID_S: s.rho, SOLID_C: s.rho, PASTE_A: recipe.pastes[0].rho,
           PASTE_B: recipe.pastes[1].rho, PASTE_C: recipe.pastes[2].rho}
    void = {SUGAR: recipe.void_sugar, MINOR: recipe.void_minor, SOLID_D: s.void, SOLID_L: s.bound, SOLID_S: s.bound,
            SOLID_C: s.bound}
    vs = sum(m/rho[k] for k, m in masses.items() if 2 <= k <= 7)
    vp = sum(void[k]*m/rho[k] for k, m in masses.items() if 2 <= k <= 7)
    vw = sum(m/rho[k] for k, m in masses.items() if k in (AQ, FAT, PASTE_A, PASTE_B, PASTE_C))
    return vs + max(vp, vw), (1.0 if vp <= 0 else min(1.0, vw/vp))


def composition(**grams):
    """Species vector (kg) from named gram amounts, e.g. composition(aqueous=227, fat=50)."""
    v = np.zeros(NS)
    for k, g in grams.items():
        v[SPECIES.index(k)] = g*1e-3
    return v

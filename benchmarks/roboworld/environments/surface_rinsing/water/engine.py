"""GPU water engine (Warp): films on body-attached cells, packets in flight, impacts, splatter, sheets, drips, soil.

Films: depth-averaged inertial thin film on every cell of surfaces.SurfaceSet, in each body's frame,

    dh/dt + div(h u) = S_h,      d(h u)/dt + div(beta h u u) = -g_n h grad h + h g_t - tau_w/rho + S_q

as an upwind (donor-cell) finite-volume scheme on the cell graph. The face velocity is the mean of the two cells' velocity
components along the edge normal (each in its own tangent plane); mass, momentum (rotated from the donor's tangent
plane into the receiver's) and every carried species leave the donor cell, so a still film does not spread by numerical
diffusion (a Rusanov flux spreads 10-100 um films 10-500x faster than lubrication drains them). The hydrostatic pressure
gradient is a Green-Gauss sum over the cell's edges (walls mirror the cell), wall friction tau_w = 3 mu u / h (laminar,
Re_f < 1000) or Blasius above, integrated implicitly. A per-cell outflow limiter keeps depths and species non-negative.
g_eff = g - a_P (rigid acceleration at the cell, 8 ms low-pass) and Coriolis act in the body frame, so a shaken plate
flings its water off. Sharp convex lips eject by the Duez criterion; open edges release what crosses them; films facing
down shed drops above the drip depth. Ejected water accumulates per edge and leaves as drops of >= PACKETS.drop_mass.

Packets: jets are chains of packets, one per orifice per tick, flying ballistically (a coherent jet
contracts by continuity and breaks into drops of 1.89 d beyond L_b); drops carry Schiller-Naumann drag. A swept segment
test against every body's water mesh (cells' quads, BVH; a wall facing a pool routes to the pool) and its sink meshes
(robot links, fixtures without charts) and against pool free surfaces finds the first hit. On a chart: the normal
impulse goes to the struck body; the stagnation pressure 1/2 rho (w.n)^2 over the jet's area fraction drives the
impingement soil law; xi(We, l/d) (Bhunia & Lienhard 1994) leaves as two splatter drops carrying the jet water mixed 1:1
with the impact cell's film; the rest enters the impact cell and its neighbours with the outward speed |w.n|.

Conservation: every scalar that is accounted (water volume, species) is float64; every transfer subtracts and adds the
same float64 value; every outflow goes to a named sink. Species (per area, float64): 0 surfactant in solution, 1
suspended soil, 2 contamination tracer, 3 heat h*T (m K), 4 foam liquid (m3/m2; part of the water account), 5 surfactant
in foam liquid. Adsorbed surfactant, adhered soil (mass, hydration) and its tracer stay on the cell.
"""

import math
import numpy as np
import warp as wp

from .. import params as P

NS = 6
SP_SURF, SP_SUS, SP_TRACER, SP_HEAT, SP_FOAM, SP_FSURF = range(6)
H_DRY = 2.0e-7         # m: below this a cell has no velocity
f64 = wp.float64

wp.config.log_level = wp.LOG_WARNING      # (wp.config.quiet is deprecated in Warp 1.16)


@wp.struct
class Fluid:
    rho: float
    g: float
    h_dry: float
    beta: float
    tau_y: float
    re_turb: float
    vogel_a: float
    vogel_b: float
    vogel_c: float
    T_amb: float
    sigma0: float
    sigma_cmc: float
    cmc: float
    k_sigma: float


@wp.struct
class Splash:
    rho: float
    sigma: float
    mu: float
    onset_a: float
    onset_b: float
    we0: float
    x0: float
    we1: float
    x1: float
    we2: float
    x2: float
    ring: float
    speed: float
    elev_lo: float
    elev_hi: float
    drop_k: float
    breakup_k: float


@wp.func
def sigma_of(c: float, fl: Fluid):
    """Szyszkowski-type surface tension of the solution at surfactant concentration c (kg/m3)."""
    a = wp.log(1.0 + fl.k_sigma*c)/wp.log(1.0 + fl.k_sigma*fl.cmc)
    return fl.sigma0 - (fl.sigma0 - fl.sigma_cmc)*wp.min(1.0, a)


@wp.func
def nu_of(T: float, fl: Fluid):
    mu = fl.vogel_a*wp.pow(10.0, fl.vogel_b/(T + 273.15 - fl.vogel_c))
    return mu/fl.rho


@wp.func
def vel(h: f64, q: wp.vec3, h_dry: float):
    hf = float(h)
    if hf > h_dry:
        return q/hf
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def rotate_across(q: wp.vec3, na: wp.vec3, nb: wp.vec3):
    """Carry a tangent vector across an edge: its component along the edge normal na (donor plane) becomes the same
    component along nb (receiver plane); the component along the shared edge tangent is kept."""
    a = wp.dot(q, na)
    return q - a*na + a*nb


# ------------------------------------------------------------------------------------------------ kinematics
@wp.kernel
def k_cell_frame(cell_body: wp.array(dtype=int), cell_pos: wp.array(dtype=wp.vec3), cell_nrm: wp.array(dtype=wp.vec3),
                 body_X: wp.array(dtype=wp.transform), body_acc: wp.array(dtype=wp.vec3), body_alpha: wp.array(dtype=wp.vec3),
                 body_w: wp.array(dtype=wp.vec3), body_com: wp.array(dtype=wp.vec3), gravity: wp.vec3,
                 g_t: wp.array(dtype=wp.vec3), g_n: wp.array(dtype=float), omega_b: wp.array(dtype=wp.vec3),
                 xw: wp.array(dtype=wp.vec3), nw: wp.array(dtype=wp.vec3)):
    c = wp.tid()
    b = cell_body[c]
    p = cell_pos[c]
    n = cell_nrm[c]
    if b < 0:
        gn = -wp.dot(gravity, n)
        g_n[c] = gn
        g_t[c] = gravity + gn*n
        omega_b[c] = wp.vec3(0.0, 0.0, 0.0)
        xw[c] = p
        nw[c] = n
        return
    X = body_X[b]
    x = wp.transform_point(X, p)
    r = x - body_com[b]
    w = body_w[b]
    a = body_acc[b] + wp.cross(body_alpha[b], r) + wp.cross(w, wp.cross(w, r))
    q = wp.transform_get_rotation(X)
    g = wp.quat_rotate_inv(q, gravity - a)
    gn = -wp.dot(g, n)
    g_n[c] = gn
    g_t[c] = g + gn*n
    omega_b[c] = wp.quat_rotate_inv(q, w)
    xw[c] = x
    nw[c] = wp.quat_rotate(q, n)


# ------------------------------------------------------------------------------------------------ film transport
@wp.kernel
def k_grad(e0: wp.array(dtype=int), e1: wp.array(dtype=int), elen: wp.array(dtype=float), en0: wp.array(dtype=wp.vec3),
           en1: wp.array(dtype=wp.vec3), h: wp.array(dtype=f64), grad: wp.array(dtype=wp.vec3), blocked: wp.array(dtype=int)):
    """Green-Gauss sum of the depth over each cell's interior edges (boundary edges: k_bedge_grad). A face to a cell
    under a standing item's footprint is a wall: the open cell mirrors itself there (as k_bedge_grad does)."""
    e = wp.tid()
    a = e0[e]
    b = e1[e]
    L = elen[e]
    if blocked[a] == 1 or blocked[b] == 1:
        if blocked[a] == 0:
            wp.atomic_add(grad, a, float(h[a])*L*en0[e])
        if blocked[b] == 0:
            wp.atomic_sub(grad, b, float(h[b])*L*en1[e])
        return
    hf = 0.5*float(h[a] + h[b])
    wp.atomic_add(grad, a, hf*L*en0[e])
    wp.atomic_sub(grad, b, hf*L*en1[e])


@wp.kernel
def k_bedge_grad(bc: wp.array(dtype=int), blen: wp.array(dtype=float), bn: wp.array(dtype=wp.vec3), bkind: wp.array(dtype=int),
                 h: wp.array(dtype=f64), grad: wp.array(dtype=wp.vec3)):
    """Walls (kind 3) mirror the cell; open edges (2) see half its depth outside, so water is not held back there."""
    e = wp.tid()
    c = bc[e]
    k = 1.0
    if bkind[e] == 2:
        k = 0.5
    wp.atomic_add(grad, c, k*float(h[c])*blen[e]*bn[e])


@wp.kernel
def k_edge_flux(e0: wp.array(dtype=int), e1: wp.array(dtype=int), elen: wp.array(dtype=float), en0: wp.array(dtype=wp.vec3),
                en1: wp.array(dtype=wp.vec3), edist: wp.array(dtype=float), ekind: wp.array(dtype=int),
                elip: wp.array(dtype=float), h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64),
                mobile: wp.array(dtype=float), duez_c: wp.array(dtype=float), grad: wp.array(dtype=wp.vec3),
                area: wp.array(dtype=float), g_n: wp.array(dtype=float), fl: Fluid, dt: float,
                flux: wp.array(dtype=float), outsum: wp.array(dtype=f64), eject: wp.array(dtype=int),
                blocked: wp.array(dtype=int)):
    """Face velocity: mean of the two cells' normal components plus a Rhie-Chow pressure correction
    -D_f g_n [(h_b - h_a)/delta - mean cell gradient . n], D_f = 1/(1/dt + 3 nu/h_f^2) (the substep for fast films, the
    lubrication mobility h^2/3nu for thin ones). It couples neighbouring depths directly, so a one-cell ridge spreads
    (a collocated mean alone sees no force on it) and a thin film drains by the lubrication flux."""
    e = wp.tid()
    a = e0[e]
    b = e1[e]
    if blocked[a] == 1 or blocked[b] == 1:
        # a cell under the footprint of an item standing on the surface: no film runs into or out of it (the first
        # build let the bay's floor film flow under the parked can as if it were not there, and its debris with it)
        flux[e] = 0.0
        eject[e] = 0
        return
    n0 = en0[e]
    n1 = en1[e]
    L = elen[e]
    ha = h[a]
    hb = h[b]
    haf = float(ha)
    hbf = float(hb)
    uf = 0.5*(wp.dot(vel(ha, q[a], fl.h_dry), n0) + wp.dot(vel(hb, q[b], fl.h_dry), n1))
    hm = 0.5*(haf + hbf)
    gnf = wp.max(0.5*(g_n[a] + g_n[b]), 0.0)
    if hm > fl.h_dry and gnf > 0.0:
        T = fl.T_amb
        nu = nu_of(T, fl)
        kf = 3.0*nu/(hm*hm)
        Df = 1.0/(1.0/dt + kf)
        ga = wp.dot(grad[a], n0)/area[a]
        gb = wp.dot(grad[b], n1)/area[b]
        uf = uf - Df*gnf*((hbf - haf)/edist[e] - 0.5*(ga + gb))
    F = float(0.0)
    if uf > 0.0:
        F = uf*haf*mobile[a]
    else:
        F = uf*hbf*mobile[b]
    ej = int(0)
    if ekind[e] == 1 and F != 0.0:
        # sharp convex lip: inertial-capillary separation (Duez et al. 2010), donor film thickness and edge radius
        d = a
        if F < 0.0:
            d = b
        hd = float(h[d])
        if hd > fl.h_dry:
            ri = elip[e]
            c = float(0.0)
            if hd > 1e-6:
                c = float(sp[0, d])/hd
            uc2 = duez_c[d]*sigma_of(c, fl)*(ri + 0.5*hd)*ri/(hd*hd*hd)
            if uf*uf > uc2:
                ej = 1
    eject[e] = ej
    flux[e] = F
    if F > 0.0:
        wp.atomic_add(outsum, a, f64(F*L*dt))
    elif F < 0.0:
        wp.atomic_add(outsum, b, f64(-F*L*dt))


@wp.kernel
def k_bedge_flux(bc: wp.array(dtype=int), blen: wp.array(dtype=float), bn: wp.array(dtype=wp.vec3), bkind: wp.array(dtype=int),
                 h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), mobile: wp.array(dtype=float), fl: Fluid, dt: float,
                 bflux: wp.array(dtype=float), outsum: wp.array(dtype=f64)):
    """Open edges (kind 2) let water leave at the cell's own normal velocity; walls (3) carry none."""
    e = wp.tid()
    c = bc[e]
    n = bn[e]
    L = blen[e]
    hc = h[c]
    F = float(0.0)
    if bkind[e] == 2:
        un = wp.dot(vel(hc, q[c], fl.h_dry), n)
        if un > 0.0:
            F = un*float(hc)*mobile[c]
            wp.atomic_add(outsum, c, f64(F*L*dt))
    bflux[e] = F


@wp.kernel
def k_limit(h: wp.array(dtype=f64), area: wp.array(dtype=float), outsum: wp.array(dtype=f64), lam: wp.array(dtype=f64)):
    c = wp.tid()
    avail = h[c]*f64(area[c])
    o = outsum[c]
    lam[c] = f64(1.0)
    if o > avail and o > f64(0.0):
        lam[c] = avail/o


@wp.kernel
def k_edge_apply(e0: wp.array(dtype=int), e1: wp.array(dtype=int), elen: wp.array(dtype=float), en0: wp.array(dtype=wp.vec3),
                 en1: wp.array(dtype=wp.vec3), flux: wp.array(dtype=float), eject: wp.array(dtype=int), lam: wp.array(dtype=f64),
                 h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), foam_mob: wp.array(dtype=float),
                 fl: Fluid, dt: float, dh: wp.array(dtype=f64), dq: wp.array(dtype=wp.vec3), dsp: wp.array2d(dtype=f64),
                 ej_m: wp.array(dtype=f64), ej_p: wp.array(dtype=wp.vec3), ej_sp: wp.array2d(dtype=f64)):
    """Move mass, momentum and species across each edge (donor values, limited). Lip flux that separates goes to the
    edge's sheet accumulator (released as drops) instead of the receiving cell. Foam moves with the film only where
    the film's wall shear exceeds the foam's yield stress (foam_mob 0..1 of the donor)."""
    e = wp.tid()
    F = flux[e]
    if F == 0.0:
        return
    a = e0[e]
    b = e1[e]
    d = a
    r = b
    na = en0[e]
    nb = en1[e]
    if F < 0.0:
        d = b
        r = a
        na = -en1[e]
        nb = -en0[e]
    vol = f64(wp.abs(F)*elen[e]*dt)*lam[d]
    hd = h[d]
    if hd <= f64(0.0) or vol <= f64(0.0):
        return
    frac = vol/hd                                  # area (m2) of the donor's column that moves
    fr = float(frac)
    mom = q[d]*fr*fl.beta
    wp.atomic_sub(dh, d, vol)
    wp.atomic_sub(dq, d, mom)
    fm = f64(foam_mob[d])
    for k in range(4):
        s = sp[k, d]*frac
        wp.atomic_sub(dsp, k, d, s)
    sf = sp[4, d]*frac*fm
    sfs = sp[5, d]*frac*fm
    wp.atomic_sub(dsp, 4, d, sf)
    wp.atomic_sub(dsp, 5, d, sfs)
    if eject[e] == 1:
        wp.atomic_add(ej_m, e, vol)
        wp.atomic_add(ej_p, e, mom)
        for k in range(4):
            wp.atomic_add(ej_sp, k, e, sp[k, d]*frac)
        wp.atomic_add(ej_sp, 4, e, sf)
        wp.atomic_add(ej_sp, 5, e, sfs)
    else:
        wp.atomic_add(dh, r, vol)
        wp.atomic_add(dq, r, rotate_across(mom, na, nb))
        for k in range(4):
            wp.atomic_add(dsp, k, r, sp[k, d]*frac)
        wp.atomic_add(dsp, 4, r, sf)
        wp.atomic_add(dsp, 5, r, sfs)


@wp.kernel
def k_bedge_apply(bc: wp.array(dtype=int), blen: wp.array(dtype=float), bflux: wp.array(dtype=float), lam: wp.array(dtype=f64),
                  h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), fl: Fluid, dt: float,
                  dh: wp.array(dtype=f64), dq: wp.array(dtype=wp.vec3), dsp: wp.array2d(dtype=f64),
                  bej_m: wp.array(dtype=f64), bej_p: wp.array(dtype=wp.vec3), bej_sp: wp.array2d(dtype=f64)):
    e = wp.tid()
    F = bflux[e]
    if F <= 0.0:
        return
    c = bc[e]
    vol = f64(F*blen[e]*dt)*lam[c]
    hc = h[c]
    if hc <= f64(0.0) or vol <= f64(0.0):
        return
    frac = vol/hc
    mom = q[c]*float(frac)*fl.beta
    wp.atomic_sub(dh, c, vol)
    wp.atomic_sub(dq, c, mom)
    wp.atomic_add(bej_m, e, vol)
    wp.atomic_add(bej_p, e, mom)
    for k in range(4):
        s = sp[k, c]*frac
        wp.atomic_sub(dsp, k, c, s)
        wp.atomic_add(bej_sp, k, e, s)


@wp.kernel
def k_cell_update(h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), area: wp.array(dtype=float),
                  nrm: wp.array(dtype=wp.vec3), g_t: wp.array(dtype=wp.vec3), g_n: wp.array(dtype=float),
                  omega_b: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), dh: wp.array(dtype=f64),
                  dq: wp.array(dtype=wp.vec3), dsp: wp.array2d(dtype=f64), h_break: wp.array(dtype=float), fl: Fluid,
                  dt: float, tau_acc: wp.array(dtype=wp.vec3), tau_w: wp.array(dtype=float), speed: wp.array(dtype=float),
                  size: wp.array(dtype=float), ratio: wp.array(dtype=float), mobile: wp.array(dtype=float),
                  foam_mob: wp.array(dtype=float), sub: wp.array(dtype=int), clamped: wp.array(dtype=f64)):
    """Apply the edge transfers; then pressure gradient, gravity, Coriolis and implicit wall friction. A depth or a
    species driven below zero is set to zero, which creates what was missing: it is counted in `clamped` (m3 of water,
    then each species), so an overdrawing transfer shows by name rather than only in the ledger."""
    c = wp.tid()
    A = f64(area[c])
    hn = h[c] + dh[c]/A
    if hn < f64(0.0):
        wp.atomic_add(clamped, 0, -hn*A)
        hn = f64(0.0)
    qn = q[c] + dq[c]/area[c]
    for k in range(6):
        v = sp[k, c] + dsp[k, c]/A
        if v < f64(0.0):
            wp.atomic_add(clamped, 1 + k, -v*A)
            v = f64(0.0)
        sp[k, c] = v
        dsp[k, c] = f64(0.0)
    dh[c] = f64(0.0)
    dq[c] = wp.vec3(0.0, 0.0, 0.0)
    n = nrm[c]
    hf = float(hn)
    gn = g_n[c]
    if hf > fl.h_dry and sub[c] < 0:
        gp = wp.max(gn, 0.0)
        gradh = grad[c]/area[c]
        qn = qn + dt*(hf*g_t[c] - gp*hf*gradh)
        u = qn/hf
        u = u - dt*2.0*wp.cross(omega_b[c], u)
        u = u - wp.dot(u, n)*n
        T = fl.T_amb
        if hf > 1e-6:
            T = float(sp[3, c])/hf
        nu = nu_of(T, fl)
        um = wp.length(u)
        re = um*hf/nu
        kf = 3.0*nu/(hf*hf)
        if re > fl.re_turb:
            cf = 0.079*wp.pow(re, -0.25)
            kf = wp.max(kf, 0.5*cf*um/hf)
        un = u/(1.0 + dt*kf)
        tw = fl.rho*kf*wp.length(un)*hf
        tau_w[c] = tw
        wp.atomic_add(tau_acc, c, hf*(u - un)*area[c])
        # beads on non-wetting substrates: immobile below the depth a pinned drop can hold on this slope. A drop of
        # height h is held by contact-angle hysteresis against rho g h sin(alpha): h_ret = h_wall / sqrt(sin alpha),
        # h_wall the substrate's retention on a vertical wall (params.SUBSTRATES), at most 5 h_wall on a near-level
        # surface. (First build: one depth for every slope, 0.5 mm on polymer, so the garbage can's vertical walls kept
        # 0.33 mm of standing "film", 0.8 L, that never drained; demo A4.)
        gt = wp.length(g_t[c])
        sin_a = gt/wp.max(wp.sqrt(gt*gt + gn*gn), 1e-6)
        if hf < h_break[c]/wp.sqrt(wp.max(sin_a, 0.04)):
            un = wp.vec3(0.0, 0.0, 0.0)
            mobile[c] = 0.0
        else:
            mobile[c] = 1.0
        qn = un*hf
        s_ = wp.length(un)
        speed[c] = s_
        ratio[c] = (s_ + wp.sqrt(gp*hf))/size[c]
        foam_mob[c] = wp.clamp((tw - fl.tau_y)/fl.tau_y, 0.0, 1.0)
    else:
        qn = wp.vec3(0.0, 0.0, 0.0)
        speed[c] = 0.0
        ratio[c] = 0.0
        tau_w[c] = 0.0
        foam_mob[c] = 0.0
        mobile[c] = 1.0
    h[c] = hn
    q[c] = qn
    grad[c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_ratio_max(ratio: wp.array(dtype=float), out: wp.array(dtype=float)):
    wp.atomic_max(out, 0, ratio[wp.tid()])


# ------------------------------------------------------------------------------------------------ packets
@wp.func
def splatter_fraction(we: float, l_over_d: float, s: Splash):
    """Bhunia & Lienhard (1994): onset (5 %) at l/d = 130/(1 + 5e-7 We^2); inside 4400 < omega < 1e4 their quadratic in
    omega = We exp(0.971 (l/d)/sqrt(We)); elsewhere the Fig. 8 plateaus (0.30 at We 5550, 0.37 at 7800, 0.75 at
    31000), clamped."""
    if we < 100.0:
        return 0.0
    onset = s.onset_a/(1.0 + s.onset_b*we*we)
    if l_over_d < onset:
        return 0.0
    om = we*wp.exp(wp.min(0.971*l_over_d/wp.sqrt(we), 30.0))
    x = float(0.05)
    if om > 4400.0 and om < 1.0e4 and we < 5000.0:
        x = -0.258 + 7.85e-5*om - 2.51e-9*om*om
    else:
        if we <= s.we0:
            x = s.x0*wp.min(1.0, we/s.we0)
        elif we <= s.we1:
            x = s.x0 + (s.x1 - s.x0)*(we - s.we0)/(s.we1 - s.we0)
        elif we <= s.we2:
            x = s.x1 + (s.x2 - s.x1)*(we - s.we1)/(s.we2 - s.we1)
        else:
            x = s.x2
    return wp.clamp(x, 0.0, 0.75)


@wp.func
def spawn_slot(head: wp.array(dtype=int), cap: int):
    s = wp.atomic_add(head, 0, 1)
    return s % cap


@wp.func
def first_hit(x0: wp.vec3, dr: wp.vec3, mesh_ids: wp.array(dtype=wp.uint64), mesh_body: wp.array(dtype=int),
              mesh_lo: wp.array(dtype=wp.vec3), mesh_hi: wp.array(dtype=wp.vec3), n_mesh: int,
              body_X: wp.array(dtype=wp.transform), static_only: int):
    """Fraction (0-1, 2 if none) of the segment x0 -> x0 + dr at which it first enters a collider (front faces); with
    static_only, the room's installed parts only (floors, walls, tables), not the moving bodies."""
    best = float(2.0)
    x1 = x0 + dr
    smin = wp.min(x0, x1)
    smax = wp.max(x0, x1)
    for k in range(n_mesh):
        blo = mesh_lo[k]
        bhi = mesh_hi[k]
        if smax[0] < blo[0] or smin[0] > bhi[0] or smax[1] < blo[1] or smin[1] > bhi[1] or smax[2] < blo[2] or smin[2] > bhi[2]:
            continue
        b = mesh_body[k]
        if static_only == 1 and b >= 0:
            continue
        s0 = x0
        d0 = dr
        if b >= 0:
            Xi = wp.transform_inverse(body_X[b])
            s0 = wp.transform_point(Xi, x0)
            d0 = wp.transform_vector(Xi, dr)
        hit = wp.mesh_query_ray(mesh_ids[k], s0, d0, 1.0)
        if hit.result:
            if hit.t < best and wp.dot(hit.normal, d0) < 0.0:
                best = hit.t
    return best


FLOOR_CLEAR = wp.constant(0.0005)       # m above the room's floor plane: the lowest a drop may be born
FLOOR_CATCH_Z = wp.constant(-0.0015)    # m: the plane under the floor charts that keeps water landing beyond them


@wp.func
def clear_offset(x0: wp.vec3, dr: wp.vec3, mesh_ids: wp.array(dtype=wp.uint64), mesh_body: wp.array(dtype=int),
                 mesh_lo: wp.array(dtype=wp.vec3), mesh_hi: wp.array(dtype=wp.vec3), n_mesh: int,
                 body_X: wp.array(dtype=wp.transform), static_only: int):
    """x0 + dr, or 0.5 mm short of the first collider face the offset would cross: a drop is never born inside a wall,
    a floor or a part it touches. The test starts 0.3 mm behind x0, so a birth point lying on a face (a can's foot on
    the floor) still sees it and is moved back in front of it; the source's own face is a back face from there. (0.3 mm
    is under the thinnest wall, the colander's 0.8 mm: from further back the test would see the sheet's other face.)"""
    L = wp.length(dr)
    if L <= 0.0:
        return x0
    u = dr/L
    eps = 0.0003
    th = first_hit(x0 - u*eps, u*(L + eps), mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, static_only)
    x = x0 + dr
    if th <= 1.0:
        x = x0 + u*(th*(L + eps) - eps - 0.0005)
    # nothing in the room is below its floor (the plane z = 0; the bay's slope is in its cells' normals only): a part
    # pressed into the floor by a few millimetres (a can leaned back on its lip) would otherwise shed drops under it
    if x[2] < FLOOR_CLEAR:
        x = wp.vec3(x[0], x[1], FLOOR_CLEAR)
    return x


@wp.func
def deposit_radial(o: int, xh_l: wp.vec3, ql: wp.vec3, vsp: float, share: f64, keep: f64, i: int, psp: wp.array2d(dtype=f64),
                   cell_pos: wp.array(dtype=wp.vec3), cell_nrm: wp.array(dtype=wp.vec3), cell_area: wp.array(dtype=float),
                   h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), vol: f64):
    Ao = cell_area[o]
    nrm = cell_nrm[o]
    r = cell_pos[o] - xh_l
    r = r - wp.dot(r, nrm)*nrm
    rl = wp.length(r)
    dirn = wp.vec3(0.0, 0.0, 0.0)
    if rl > 1e-9:
        dirn = r/rl
    qt = ql - wp.dot(ql, nrm)*nrm
    wp.atomic_add(h, o, share/f64(Ao))
    wp.atomic_add(q, o, float(share)*(qt + vsp*dirn)/Ao)
    fr = share/vol
    for kk in range(6):
        wp.atomic_add(sp, kk, o, keep*psp[kk, i]*fr/f64(Ao))


@wp.kernel
def k_packets(alive: wp.array(dtype=int), px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), pm: wp.array(dtype=f64),
              pd: wp.array(dtype=float), pkind: wp.array(dtype=int), page: wp.array(dtype=float), pdist: wp.array(dtype=float),
              psp: wp.array2d(dtype=f64), head: wp.array(dtype=int), cap: int,
              mesh_ids: wp.array(dtype=wp.uint64), mesh_body: wp.array(dtype=int), mesh_face0: wp.array(dtype=int),
              mesh_lo: wp.array(dtype=wp.vec3), mesh_hi: wp.array(dtype=wp.vec3), n_mesh: int,
              face_cell: wp.array(dtype=int), face_sink: wp.array(dtype=int), face_pool: wp.array(dtype=int),
              body_X: wp.array(dtype=wp.transform), body_v: wp.array(dtype=wp.vec3), body_w: wp.array(dtype=wp.vec3),
              body_com: wp.array(dtype=wp.vec3),
              cell_pos: wp.array(dtype=wp.vec3), cell_nrm: wp.array(dtype=wp.vec3),
              cell_area: wp.array(dtype=float), adj_start: wp.array(dtype=int), adj_edge: wp.array(dtype=int),
              e0: wp.array(dtype=int), e1: wp.array(dtype=int), en0: wp.array(dtype=wp.vec3), en1: wp.array(dtype=wp.vec3),
              elen: wp.array(dtype=float),
              h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), sub: wp.array(dtype=int),
              soil_sig: wp.array(dtype=float),
              pool_body: wp.array(dtype=int), pool_kind: wp.array(dtype=int), pool_n: wp.array(dtype=wp.vec3),
              pool_d: wp.array(dtype=float), pool_a: wp.array(dtype=wp.vec3), pool_b: wp.array(dtype=wp.vec3), n_pool: int,
              gravity: wp.vec3, air_mu: float, air_rho: float, fl: Fluid, spl: Splash, dt: float, seed: int,
              lo: wp.vec3, hi: wp.vec3, max_age: float,
              body_f: wp.array(dtype=wp.vec3), body_t: wp.array(dtype=wp.vec3), imp_erode: wp.array(dtype=float),
              imp_p: wp.array(dtype=float),
              sink_m: wp.array(dtype=f64), sink_sp: wp.array2d(dtype=f64), pool_m: wp.array(dtype=f64),
              pool_sp: wp.array2d(dtype=f64), pool_imp: wp.array(dtype=wp.vec3), lost: wp.array(dtype=f64),
              counts: wp.array(dtype=int), diag: wp.array(dtype=f64), sink_mode: wp.array(dtype=int),
              shed_m: wp.array(dtype=f64), blocked: wp.array(dtype=int), h0: wp.array(dtype=f64),
              take_left: wp.array(dtype=f64), floor_sink: int):
    i = wp.tid()
    if alive[i] == 0:
        return
    wp.atomic_add(counts, 0, 1)
    x0 = px[i]
    if x0[2] < -0.003:
        # nothing in the room is below its floor: a packet found under the floor's catch plane (1.5 mm under the floor,
        # where water falling beyond the charts is kept) came from a birth path not yet guarded. It is lifted back above
        # the floor and its water counted (diag[4], kg), so the count, not the ledger, shows whether this happens. (The
        # first version lifted anything under z = 0: drops falling past the bay's edge were lifted every tick before they
        # could reach the catch plane, and aged out.)
        x0 = wp.vec3(x0[0], x0[1], FLOOR_CLEAR)
        px[i] = x0
        wp.atomic_add(diag, 4, pm[i])
    v0 = pv[i]
    m = pm[i]
    mf = float(m)
    d = pd[i]
    kind = pkind[i]
    skip = int(-1)
    if kind >= 2:
        # a drop that ran off a thin part (a hose, a lever, a rack wire): it passes that part (kind = 2 + its mesh)
        skip = kind - 2
        kind = 1
    v1 = v0 + gravity*dt
    if kind == 1:
        # drop: Schiller-Naumann drag toward still air, exact exponential relaxation over the tick
        sv = wp.length(v0)
        re = air_rho*sv*d/air_mu
        tau = fl.rho*d*d/(18.0*air_mu*(1.0 + 0.15*wp.pow(re, 0.687)))
        vt = gravity*tau
        ex = wp.exp(-dt/tau)
        v1 = vt + (v0 - vt)*ex
    x1 = x0 + 0.5*(v0 + v1)*dt
    seg = x1 - x0
    slen = wp.length(seg)
    best_t = float(2.0)
    best_face = int(-1)
    best_mesh = int(-1)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    smin = wp.min(x0, x1)
    smax = wp.max(x0, x1)
    if slen > 1e-9:
        for k in range(n_mesh):
            if k == skip:
                continue
            blo = mesh_lo[k]
            bhi = mesh_hi[k]
            if smax[0] < blo[0] or smin[0] > bhi[0] or smax[1] < blo[1] or smin[1] > bhi[1] or smax[2] < blo[2] or smin[2] > bhi[2]:
                continue
            b = mesh_body[k]
            s0 = x0
            dr = seg
            if b >= 0:
                Xi = wp.transform_inverse(body_X[b])
                s0 = wp.transform_point(Xi, x0)
                dr = wp.transform_vector(Xi, seg)
            hit = wp.mesh_query_ray(mesh_ids[k], s0, dr, 1.0)
            if hit.result:
                if hit.t < best_t and wp.dot(hit.normal, dr) < 0.0:
                    best_t = hit.t
                    best_face = hit.face
                    best_mesh = k
                    best_n = hit.normal
    # free surfaces of pools crossed by the segment before the first solid hit
    best_pool = int(-1)
    for k in range(n_pool):
        b = pool_body[k]
        a0 = x0
        a1 = x1
        if b >= 0:
            Xi = wp.transform_inverse(body_X[b])
            a0 = wp.transform_point(Xi, x0)
            a1 = wp.transform_point(Xi, x1)
        nk = pool_n[k]
        sd0 = wp.dot(a0, nk) - pool_d[k]
        sd1 = wp.dot(a1, nk) - pool_d[k]
        if sd1 < 0.0:
            t = float(0.0)
            if sd0 > 0.0:
                t = sd0/(sd0 - sd1)
            xl = a0 + (a1 - a0)*t
            inside = int(0)
            pa = pool_a[k]
            pb = pool_b[k]
            if pool_kind[k] == 0:
                if xl[0] > pa[0] and xl[0] < pb[0] and xl[1] > pa[1] and xl[1] < pb[1] and xl[2] > pa[2] and xl[2] < pb[2]:
                    inside = 1
            else:
                z = xl[2]
                if z > pa[0] and z < pa[1]:
                    rr = pb[0] + (pb[1] - pb[0])*(z - pa[0])/(pa[1] - pa[0])
                    if xl[0]*xl[0] + xl[1]*xl[1] < rr*rr:
                        inside = 1
            if inside == 1 and t < best_t:
                best_t = t
                best_pool = k
    if best_pool >= 0:
        k = best_pool
        wp.atomic_add(pool_m, k, m)
        for kk in range(6):
            wp.atomic_add(pool_sp, kk, k, psp[kk, i])
        wp.atomic_add(pool_imp, k, mf*v1/dt)
        alive[i] = 0
        return
    if best_mesh >= 0:
        xh = x0 + seg*best_t
        b = mesh_body[best_mesh]
        gf = mesh_face0[best_mesh] + best_face
        c = face_cell[gf]
        nwld = best_n
        vs = wp.vec3(0.0, 0.0, 0.0)
        if b >= 0:
            nwld = wp.transform_vector(body_X[b], best_n)
            vs = body_v[b] + wp.cross(body_w[b], xh - body_com[b])
        nl = wp.length(nwld)
        if nl > 0.0:
            nwld = nwld/nl
        w = v1 - vs
        wn = wp.dot(w, nwld)
        if wn > 0.0:
            wn = 0.0
        if b >= 0:
            imp = mf*wn*nwld/dt
            wp.atomic_add(body_f, b, imp)
            wp.atomic_add(body_t, b, wp.cross(xh - body_com[b], imp))
        vn = -wn
        # splatter of jets and fast drops
        xi = float(0.0)
        if kind == 0:
            we = spl.rho*vn*vn*d/spl.sigma
            xi = splatter_fraction(we, (pdist[i] + slen*best_t)/wp.max(d, 1e-5), spl)
        else:
            we = spl.rho*vn*vn*d/spl.sigma
            oh = spl.mu/wp.sqrt(spl.rho*spl.sigma*d)
            if we*wp.pow(oh, -0.4) > spl.drop_k:
                xi = splatter_fraction(we, 1000.0, spl)
        if c < 0:
            pk = face_pool[gf]
            if pk >= 0 and xi == 0.0:
                wp.atomic_add(pool_m, pk, m)
                for kk in range(6):
                    wp.atomic_add(pool_sp, kk, pk, psp[kk, i])
                alive[i] = 0
                return
            if pk < 0:
                s = face_sink[gf]
                # water does not stay on a hose, a lever or a rack wire, nor on the steep or overhanging faces of the
                # robot and the fixtures: it runs off and falls as a drop from where it struck (its normal momentum
                # went into the body above). Only faces that look up (a ledge, a shelf, the top of the base) keep it,
                # in their account. (First build: every uncharted collider kept what struck it; a hose crossing under
                # the nozzle swallowed 0.7 L of the floor flush, demo A6.)
                mode = sink_mode[s]
                shed = int(0)
                if mode == 1:
                    shed = 1
                if mode == 2 and nwld[2] < 0.5:
                    shed = 2
                if shed > 0:
                    wp.atomic_add(shed_m, s, m)
                    if shed == 1:
                        # it passes the part it struck, from the point it struck; a hose lying pressed into the floor is
                        # struck below the floor plane, and the drop would walk down it from link to link
                        px[i] = wp.vec3(xh[0], xh[1], wp.max(xh[2], FLOOR_CLEAR))
                        pkind[i] = 2 + best_mesh
                    else:
                        px[i] = clear_offset(xh, nwld*0.003, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
                        pkind[i] = 1
                    pv[i] = vs
                    pd[i] = wp.clamp(wp.pow(6.0*mf/(3.14159265*fl.rho), 1.0/3.0), 1.0e-3, 5.0e-3)
                    page[i] = 0.0
                    pdist[i] = 0.0
                    return
                wp.atomic_add(sink_m, s, m)
                for kk in range(6):
                    wp.atomic_add(sink_sp, kk, s, psp[kk, i])
                alive[i] = 0
                return
        keep = f64(1.0 - xi)
        st = wp.rand_init(seed, i)
        # diagnostics: jet (kind 0) and drop (kind 1) mass arriving at solid surfaces, and what splattered of each
        wp.atomic_add(diag, 2*kind, m)
        wp.atomic_add(diag, 2*kind + 1, f64(xi)*m)
        hc = f64(0.0)
        A = float(1.0)
        if c >= 0:
            A = cell_area[c]
            hc = h[c]
            # stagnation pressure over the jet's area fraction of the cell: impingement erosion (soil law i)
            aj = 0.25*3.14159265*d*d
            frac = wp.min(1.0, aj/A)
            pimp = 0.5*fl.rho*vn*vn
            wp.atomic_add(imp_p, c, frac*pimp)
            if soil_sig[c] > 0.0:
                wp.atomic_add(imp_erode, c, frac*wp.max(pimp/soil_sig[c] - 1.0, 0.0)*dt)
        if xi > 0.0:
            # two drops: the packet's splattered share, plus an equal volume of the impact cell's film (if present)
            mfilm = f64(0.0)
            take = f64(0.0)
            if c >= 0 and sub[c] < 0 and h0[c] > f64(0.0):
                # every packet striking the cell this tick draws on the film the cell had at its start (take_left, a
                # depth, drawn down atomically), and takes that share of its species: together they can never take more
                # than was there. (Before: each took up to half the film it read, a hose put several packets into one
                # cell per tick, the detergent went negative and the clamp at the cell update created it: 12 ug at
                # the start of the can rinse, while water, replenished by the same packets, stayed exact)
                want = wp.min(f64(0.5*xi)*m/(f64(fl.rho)*f64(A)), f64(0.5)*h0[c])
                left = wp.atomic_sub(take_left, c, want)
                got = wp.min(want, wp.max(left, f64(0.0)))
                mfilm = got*f64(fl.rho)*f64(A)
                take = got/h0[c]
                if got < want:
                    wp.atomic_add(take_left, c, want - got)
            # the film's species taken with it (per unit area), read ONCE: other packets deposit into and splatter from the
            # same cell in this launch, and amounts read again at the subtraction differed from what the drops received
            # (a bench found +3.8 ug of suspended solids created in one tick)
            tk = wp.vec4d(f64(0.0), f64(0.0), f64(0.0), f64(0.0))
            if c >= 0 and take > f64(0.0):
                tk = wp.vec4d(take*sp[0, c], take*sp[1, c], take*sp[2, c], take*sp[3, c])
            t1 = wp.normalize(wp.cross(nwld, wp.vec3(0.3, 0.7, 0.2)))
            t2 = wp.cross(nwld, t1)
            wt = w - wn*nwld
            for jj in range(2):
                slot = spawn_slot(head, cap)
                az = 6.2831853*wp.randf(st)
                el = spl.elev_lo + (spl.elev_hi - spl.elev_lo)*wp.randf(st)
                dirr = wp.cos(az)*t1 + wp.sin(az)*t2
                # in the frame sliding along the wall with the jet's tangential velocity the impact is normal: droplets
                # leave all round at a fraction of the NORMAL speed, and keep the tangential velocity (added below). An
                # oblique jet therefore throws its splatter forward (bias tan of the incidence), a normal one all round.
                # (First build: |w| in every direction plus a fixed 0.7 forward bias; a hose aimed 60 deg off the normal
                # still wet whoever stood behind it, demo A6.)
                spd = spl.speed*vn*(0.6 + 0.8*wp.randf(st))
                vdir = wp.cos(el)*dirr + wp.sin(el)*nwld
                if alive[slot] == 1 and slot != i:
                    wp.atomic_add(lost, 0, pm[slot])
                    wp.atomic_add(lost, 9, pm[slot])
                    for kk in range(6):
                        wp.atomic_add(lost, 1 + kk, psp[kk, slot])
                if slot == i:
                    # the ring wrapped onto this packet's own slot: keep the account exact by counting it lost
                    wp.atomic_add(lost, 0, f64(0.5*xi)*m + f64(0.5)*mfilm)
                    wp.atomic_add(lost, 9, f64(0.5*xi)*m + f64(0.5)*mfilm)
                    for kk in range(6):
                        v = f64(0.5*xi)*psp[kk, i]
                        if kk < 4:
                            v = v + f64(0.5)*tk[kk]*f64(A)
                        wp.atomic_add(lost, 1 + kk, v)
                    continue
                alive[slot] = 1
                # droplets leave the crown's rim, ring_d jet diameters out from the hit; where a wall or another part
                # stands closer, the crown is cut off there (first build: the offset put drops born within a centimetre of
                # a wall's foot inside the wall, and they fell through it, out of the room: 108 g of a can rinse lost)
                x_s = clear_offset(xh, nwld*0.0015, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
                px[slot] = clear_offset(x_s, dirr*spl.ring*d, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
                pv[slot] = vs + spd*vdir + wt
                pm[slot] = f64(0.5*xi)*m + f64(0.5)*mfilm
                pd[slot] = wp.max(0.3*d, 2.0e-4)
                pkind[slot] = 1
                page[slot] = 0.0
                pdist[slot] = 0.0
                for kk in range(6):
                    v = f64(0.5*xi)*psp[kk, i]
                    if kk < 4:
                        v = v + f64(0.5)*tk[kk]*f64(A)
                    psp[kk, slot] = v
            if c >= 0 and mfilm > f64(0.0):
                wp.atomic_add(h, c, -mfilm/(f64(fl.rho)*f64(A)))
                for kk in range(4):
                    wp.atomic_add(sp, kk, c, -tk[kk])
        if c < 0:
            # a pool wall: what does not splatter joins the pool
            pk = face_pool[gf]
            wp.atomic_add(pool_m, pk, keep*m)
            for kk in range(6):
                wp.atomic_add(pool_sp, kk, pk, keep*psp[kk, i])
            alive[i] = 0
            return
        # deposit into the impact cell and its neighbours: tangential velocity + radial spread at |w.n|
        wt2 = w - wn*nwld
        ql = wt2
        if b >= 0:
            ql = wp.transform_vector(wp.transform_inverse(body_X[b]), wt2)
        vol = keep*m/f64(fl.rho)
        if sub[c] >= 0:
            # the impact cell lies under a pool: the water joins that pool
            pk = sub[c]
            wp.atomic_add(pool_m, pk, keep*m)
            for kk in range(6):
                wp.atomic_add(pool_sp, kk, pk, keep*psp[kk, i])
            alive[i] = 0
            return
        # impingement zone: the impact cell keeps a third; its neighbours and theirs (two rings, area-weighted) receive
        # the rest moving radially away from the hit point at |w.n| (inviscid radial jet; capped at 30 m/s)
        xh_l = xh
        if b >= 0:
            xh_l = wp.transform_point(wp.transform_inverse(body_X[b]), xh)
        nb0 = adj_start[c]
        nb1 = adj_start[c + 1]
        # (cells under the footprint of an item standing on the surface take no share: they are covered, and water put
        # there could never run out again -- the first build collected debris in single cells round the parked can)
        wsum = f64(0.0)
        for j in range(nb0, nb1):
            ed = adj_edge[j]
            o = e1[ed]
            if o == c:
                o = e0[ed]
            if blocked[o] == 0:
                wsum = wsum + f64(cell_area[o])
            for jj in range(adj_start[o], adj_start[o + 1]):
                ed2 = adj_edge[jj]
                o2 = e1[ed2]
                if o2 == o:
                    o2 = e0[ed2]
                if o2 != c and blocked[o2] == 0:
                    wsum = wsum + f64(cell_area[o2])
        vsp = wp.min(vn, 30.0)
        third = vol
        if wsum > f64(0.0):
            third = vol/f64(3.0)
            if blocked[c] == 1:
                # a drop landing on the edge of a footprint (water running off a can's wall drips at its base): all
                # of it goes to the open cells round it (first fix: a third stayed in the covered cell, for ever)
                third = f64(0.0)
        rest = vol - third
        wp.atomic_add(h, c, third/f64(A))
        wp.atomic_add(q, c, float(third)*ql/A)
        sfr = third/vol
        for kk in range(6):
            wp.atomic_add(sp, kk, c, keep*psp[kk, i]*sfr/f64(A))
        if wsum > f64(0.0):
            for j in range(nb0, nb1):
                ed = adj_edge[j]
                o = e1[ed]
                if o == c:
                    o = e0[ed]
                if blocked[o] == 0:
                    deposit_radial(o, xh_l, ql, vsp, rest*f64(cell_area[o])/wsum, keep, i, psp, cell_pos, cell_nrm,
                                   cell_area, h, q, sp, vol)
                for jj in range(adj_start[o], adj_start[o + 1]):
                    ed2 = adj_edge[jj]
                    o2 = e1[ed2]
                    if o2 == o:
                        o2 = e0[ed2]
                    if o2 != c and blocked[o2] == 0:
                        deposit_radial(o2, xh_l, ql, vsp, rest*f64(cell_area[o2])/wsum, keep, i, psp, cell_pos, cell_nrm,
                                       cell_area, h, q, sp, vol)
        alive[i] = 0
        return
    # in flight
    if floor_sink >= 0 and x0[2] >= FLOOR_CATCH_Z and x1[2] < FLOOR_CATCH_Z:
        # the room's floor below its charts is a plane (a mesh of two large triangles, system._sink_meshes): a segment
        # crossing it unstruck is kept there all the same, and counted (diag[5], kg). In the can rinse the triangle
        # query missed about one slow drop in 2000 crossing it (none in a fresh scene), and those fell out of the room
        wp.atomic_add(diag, 5, m)
        wp.atomic_add(sink_m, floor_sink, m)
        for kk in range(6):
            wp.atomic_add(sink_sp, kk, floor_sink, psp[kk, i])
        alive[i] = 0
        return
    page[i] = page[i] + dt
    if x1[0] < lo[0] or x1[1] < lo[1] or x1[2] < lo[2] or x1[0] > hi[0] or x1[1] > hi[1] or x1[2] > hi[2] or page[i] > max_age:
        wp.atomic_add(lost, 0, m)
        if page[i] > max_age:
            wp.atomic_add(lost, 7, m)
        else:
            wp.atomic_add(lost, 8, m)
        for kk in range(6):
            wp.atomic_add(lost, 1 + kk, psp[kk, i])
        alive[i] = 0
        return
    if kind == 0:
        # coherent jet: contracts by continuity; breaks into drops of 1.89 d beyond the capillary breakup length
        s0_ = wp.length(v0)
        s1_ = wp.length(v1)
        if s1_ > 1e-6:
            pd[i] = d*wp.sqrt(s0_/s1_)
        we = spl.rho*s1_*s1_*d/spl.sigma
        oh = spl.mu/wp.sqrt(spl.rho*spl.sigma*d)
        if pdist[i] + slen > spl.breakup_k*wp.sqrt(we)*(1.0 + 3.0*oh)*d:
            pkind[i] = 1
            pd[i] = 1.89*d
    px[i] = x1
    pv[i] = v1
    pdist[i] = pdist[i] + slen


@wp.kernel
def k_spawn(n: int, cap: int, head: wp.array(dtype=int), src_x: wp.array(dtype=wp.vec3), src_v: wp.array(dtype=wp.vec3),
            src_m: wp.array(dtype=f64), src_d: wp.array(dtype=float), src_kind: wp.array(dtype=int),
            src_sp: wp.array2d(dtype=f64), alive: wp.array(dtype=int), px: wp.array(dtype=wp.vec3),
            pv: wp.array(dtype=wp.vec3), pm: wp.array(dtype=f64), pd: wp.array(dtype=float), pkind: wp.array(dtype=int),
            page: wp.array(dtype=float), pdist: wp.array(dtype=float), psp: wp.array2d(dtype=f64), lost: wp.array(dtype=f64),
            guard: int, mesh_ids: wp.array(dtype=wp.uint64), mesh_body: wp.array(dtype=int), mesh_lo: wp.array(dtype=wp.vec3),
            mesh_hi: wp.array(dtype=wp.vec3), n_mesh: int, body_X: wp.array(dtype=wp.transform)):
    """New packets. guard = 1 (water re-entering from a pool's rim or holes, a sponge's drip): the birth point is tested
    from 1 cm above it against the room's installed parts and put above any of their surfaces in between (first build:
    a can leaned back on the floor poured over the lip touching the floor, the drops were born under the floor and fell
    out of the room). Moving bodies are not tested: such water is born outside its own container by design (under a
    colander's hole, beyond a can's lip), and a test from above would put it back in (second build: the colander could
    not drain and the tipped can not empty)."""
    j = wp.tid()
    if j >= n:
        return
    x = src_x[j]
    if guard == 1:
        up = wp.vec3(0.0, 0.0, 0.01)
        x = clear_offset(x + up, -up, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 1)
    s = spawn_slot(head, cap)
    if alive[s] == 1:
        wp.atomic_add(lost, 0, pm[s])
        wp.atomic_add(lost, 9, pm[s])
        for k in range(6):
            wp.atomic_add(lost, 1 + k, psp[k, s])
    alive[s] = 1
    px[s] = x
    pv[s] = src_v[j]
    pm[s] = src_m[j]
    pd[s] = src_d[j]
    pkind[s] = src_kind[j]
    page[s] = 0.0
    pdist[s] = 0.0
    for k in range(6):
        psp[k, s] = src_sp[k, j]


@wp.kernel
def k_emit_edges(n: int, emid: wp.array(dtype=wp.vec3), ebody: wp.array(dtype=int), enrm: wp.array(dtype=wp.vec3),
                 efn: wp.array(dtype=wp.vec3),
                 ej_m: wp.array(dtype=f64), ej_p: wp.array(dtype=wp.vec3), ej_sp: wp.array2d(dtype=f64),
                 body_X: wp.array(dtype=wp.transform), body_v: wp.array(dtype=wp.vec3), body_w: wp.array(dtype=wp.vec3),
                 body_com: wp.array(dtype=wp.vec3), fl: Fluid, m_drop: f64, force: int, head: wp.array(dtype=int), cap: int,
                 alive: wp.array(dtype=int), px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), pm: wp.array(dtype=f64),
                 pd: wp.array(dtype=float), pkind: wp.array(dtype=int), page: wp.array(dtype=float), pdist: wp.array(dtype=float),
                 psp: wp.array2d(dtype=f64), lost: wp.array(dtype=f64), mesh_ids: wp.array(dtype=wp.uint64),
                 mesh_body: wp.array(dtype=int), mesh_lo: wp.array(dtype=wp.vec3), mesh_hi: wp.array(dtype=wp.vec3),
                 n_mesh: int):
    """Water that left a film over a lip or an open edge leaves the edge as a drop once a drop's mass has gathered. The
    drop is born on the film's side of the surface and short of any part the edge stands on or against (first build:
    the foot of a can's wall released drops below the floor, and they fell out of the room)."""
    e = wp.tid()
    vol = ej_m[e]
    m = vol*f64(fl.rho)
    if m <= f64(0.0) or (m < m_drop and force == 0):
        return
    b = ebody[e]
    u = ej_p[e]/wp.max(float(vol), 1e-30)/fl.beta       # mean film velocity of what left (body frame)
    x = emid[e]
    nn = enrm[e]
    fn = efn[e]
    vw = u
    if b >= 0:
        X = body_X[b]
        x = wp.transform_point(X, x)
        vw = wp.transform_vector(X, u) + body_v[b] + wp.cross(body_w[b], x - body_com[b])
        nn = wp.transform_vector(X, nn)
        fn = wp.transform_vector(X, fn)
    s = spawn_slot(head, cap)
    if alive[s] == 1:
        wp.atomic_add(lost, 0, pm[s])
        wp.atomic_add(lost, 9, pm[s])
        for k in range(6):
            wp.atomic_add(lost, 1 + k, psp[k, s])
    alive[s] = 1
    dd = wp.pow(6.0*float(m)/(3.14159265*fl.rho), 1.0/3.0)
    xs = clear_offset(x, fn*0.5*dd, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
    px[s] = clear_offset(xs, nn*0.5*dd, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
    pv[s] = vw
    pm[s] = m
    pd[s] = dd
    pkind[s] = 1
    page[s] = 0.0
    pdist[s] = 0.0
    for k in range(6):
        psp[k, s] = ej_sp[k, e]
        ej_sp[k, e] = f64(0.0)
    ej_m[e] = f64(0.0)
    ej_p[e] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_drip(h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), area: wp.array(dtype=float),
           g_n: wp.array(dtype=float), xw: wp.array(dtype=wp.vec3), nw: wp.array(dtype=wp.vec3), cell_body: wp.array(dtype=int),
           sub: wp.array(dtype=int), body_X: wp.array(dtype=wp.transform), body_v: wp.array(dtype=wp.vec3),
           body_w: wp.array(dtype=wp.vec3), body_com: wp.array(dtype=wp.vec3), fl: Fluid, h_drip: float, gn_drip: float,
           head: wp.array(dtype=int), cap: int, alive: wp.array(dtype=int), px: wp.array(dtype=wp.vec3),
           pv: wp.array(dtype=wp.vec3), pm: wp.array(dtype=f64), pd: wp.array(dtype=float), pkind: wp.array(dtype=int),
           page: wp.array(dtype=float), pdist: wp.array(dtype=float), psp: wp.array2d(dtype=f64), lost: wp.array(dtype=f64),
           mesh_ids: wp.array(dtype=wp.uint64), mesh_body: wp.array(dtype=int), mesh_lo: wp.array(dtype=wp.vec3),
           mesh_hi: wp.array(dtype=wp.vec3), n_mesh: int):
    """A film facing down sheds its depth above the drip depth as a pendant drop (Rayleigh-Taylor), born short of any part
    just below it."""
    c = wp.tid()
    if sub[c] >= 0 or g_n[c] > gn_drip:
        return
    hc = h[c]
    if float(hc) <= h_drip:
        return
    A = f64(area[c])
    dvol = (hc - f64(0.5*h_drip))*A
    frac = dvol/(hc*A)
    m = dvol*f64(fl.rho)
    s = spawn_slot(head, cap)
    if alive[s] == 1:
        wp.atomic_add(lost, 0, pm[s])
        wp.atomic_add(lost, 9, pm[s])
        for k in range(6):
            wp.atomic_add(lost, 1 + k, psp[k, s])
    b = cell_body[c]
    x = xw[c]
    vb = wp.vec3(0.0, 0.0, 0.0)
    if b >= 0:
        vb = body_v[b] + wp.cross(body_w[b], x - body_com[b])
    dd = wp.pow(6.0*float(m)/(3.14159265*fl.rho), 1.0/3.0)
    alive[s] = 1
    px[s] = clear_offset(x, nw[c]*0.5*dd, mesh_ids, mesh_body, mesh_lo, mesh_hi, n_mesh, body_X, 0)
    pv[s] = vb
    pm[s] = m
    pd[s] = dd
    pkind[s] = 1
    page[s] = 0.0
    pdist[s] = 0.0
    for k in range(6):
        v = sp[k, c]*frac*A
        if k >= 4:
            v = f64(0.0)
        psp[k, s] = v
        sp[k, c] = sp[k, c] - v/A
    h[c] = hc - dvol/A
    q[c] = q[c]*(1.0 - float(frac))


# ------------------------------------------------------------------------------------------------ species and soil
@wp.kernel
def k_species(h: wp.array(dtype=f64), sp: wp.array2d(dtype=f64), gamma: wp.array(dtype=f64), gamma_m: wp.array(dtype=float),
              soil: wp.array(dtype=f64), soil_w: wp.array(dtype=float), soil_tr: wp.array(dtype=float),
              soil_sd: wp.array(dtype=float), soil_sw: wp.array(dtype=float), soil_ts: wp.array(dtype=float),
              soil_sig: wp.array(dtype=float), tau_w: wp.array(dtype=float), imp_erode: wp.array(dtype=float),
              abr: wp.array(dtype=f64), sub: wp.array(dtype=int), pool_c: wp.array2d(dtype=f64), fl: Fluid, k_a: float,
              k_d: float, foam_k: float, k_dil: float, cmc: float, foam_phi: float, k_i: float, k_e: float, h_wet: float,
              soak_dbl: float, dt: float):
    c = wp.tid()
    hc = h[c]
    hf = float(hc)
    wet = hf > h_wet or sub[c] >= 0
    T = fl.T_amb
    if sub[c] >= 0:
        # under a pool's surface the soil soaks at the pool's temperature (first build: at the 20 C ambient, so egg in
        # 43 C wash water hydrated 2.9x too slowly; demo A5 scrub)
        T = float(pool_c[3, sub[c]])
    elif hf > 1e-6:
        T = float(sp[3, c])/hf
    conc = float(0.0)
    if hf > 1e-7:
        conc = float(sp[0, c])/hf
    # adsorbed surfactant (Langmuir, implicit), limited by what is in solution
    if wet and sub[c] < 0:
        g0 = float(gamma[c])
        gm = gamma_m[c]
        ka = k_a*conc
        gnew = (g0 + ka*gm*dt)/(1.0 + (ka + k_d)*dt)
        dg = f64(gnew - g0)
        if dg > sp[0, c]:
            dg = sp[0, c]
        gamma[c] = gamma[c] + dg
        sp[0, c] = sp[0, c] - dg
    # foam: drains and coarsens (half-life), collapses fast where the liquid around it is dilute
    F = sp[4, c]
    if F > f64(0.0):
        k = foam_k
        if conc < 0.5*cmc and hf > h_wet:
            k = k + k_dil*(1.0 - conc/(0.5*cmc))
        fr = f64(1.0 - wp.exp(-k*dt))
        dF = F*fr
        dS = sp[5, c]*fr
        sp[4, c] = F - dF
        sp[5, c] = sp[5, c] - dS
        h[c] = hc + dF
        sp[0, c] = sp[0, c] + dS
        sp[3, c] = sp[3, c] + dF*f64(T)
    # adhered soil: hydration while wetted, then strength-limited removal (impingement, shear erosion, abrasion)
    ms = soil[c]
    if ms > f64(0.0):
        w = soil_w[c]
        if wet:
            ts = soil_ts[c]*wp.pow(2.0, -(T - 20.0)/soak_dbl)
            w = w + (1.0 - w)*(1.0 - wp.exp(-dt/wp.max(ts, 1e-3)))
            soil_w[c] = w
        sig = soil_sd[c]*(1.0 - w)*(1.0 - w) + soil_sw[c]
        soil_sig[c] = sig
        rem = f64(0.0)
        if wet:
            rem = f64(k_i*imp_erode[c] + k_e*wp.max(tau_w[c] - sig, 0.0)*dt)
        rem = rem + abr[c]
        if rem > ms:
            rem = ms
        if rem > f64(0.0):
            soil[c] = ms - rem
            # removed solids go into suspension in the film, or are lost to a dry-scrub sink if no film (abrasion)
            sp[1, c] = sp[1, c] + rem
            sp[2, c] = sp[2, c] + rem*f64(soil_tr[c])
    else:
        soil_sig[c] = 0.0
    imp_erode[c] = 0.0
    abr[c] = f64(0.0)


@wp.kernel
def k_front(e0: wp.array(dtype=int), e1: wp.array(dtype=int), elen: wp.array(dtype=float), en0: wp.array(dtype=wp.vec3),
            en1: wp.array(dtype=wp.vec3), h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64),
            soil: wp.array(dtype=f64), soil_w: wp.array(dtype=float), soil_tr: wp.array(dtype=float),
            area: wp.array(dtype=float), fl: Fluid, k_front: float, m_clean: float, dt: float, front_rm: wp.array(dtype=f64)):
    """Wilson et al. (2014) cleaning front: soil next to a clean cell recedes at k' M, M = beta rho q^2 / h the film's
    momentum flux on the clean side toward the soil; mass removed = m_s L (da/dt) dt."""
    e = wp.tid()
    a = e0[e]
    b = e1[e]
    ma = float(soil[a])
    mb = float(soil[b])
    cl = int(-1)
    so = int(-1)
    nrm = wp.vec3(0.0, 0.0, 0.0)
    if ma < m_clean and mb >= m_clean:
        cl = a
        so = b
        nrm = en0[e]
    elif mb < m_clean and ma >= m_clean:
        cl = b
        so = a
        nrm = -en1[e]
    if cl < 0:
        return
    hc = float(h[cl])
    if hc <= fl.h_dry:
        return
    u = q[cl]/hc
    un = wp.dot(u, nrm)
    if un <= 0.0:
        return
    M = fl.beta*fl.rho*un*un*hc
    w = soil_w[so]
    vf = k_front*w*M
    dm = f64(float(soil[so])*elen[e]*vf*dt/area[so])
    wp.atomic_add(front_rm, so, dm)


@wp.kernel
def k_front_apply(soil: wp.array(dtype=f64), soil_tr: wp.array(dtype=float), sp: wp.array2d(dtype=f64),
                  front_rm: wp.array(dtype=f64)):
    c = wp.tid()
    r = front_rm[c]
    if r > f64(0.0):
        if r > soil[c]:
            r = soil[c]
        soil[c] = soil[c] - r
        sp[1, c] = sp[1, c] + r
        sp[2, c] = sp[2, c] + r*f64(soil_tr[c])
    front_rm[c] = f64(0.0)


# ------------------------------------------------------------------------------------------------ pools: submerged cells
@wp.kernel
def k_submerge(cell_body: wp.array(dtype=int), cell_pos: wp.array(dtype=wp.vec3), xw: wp.array(dtype=wp.vec3),
               own_pool: wp.array(dtype=int), body_X: wp.array(dtype=wp.transform), body_v: wp.array(dtype=wp.vec3),
               body_w: wp.array(dtype=wp.vec3), body_com: wp.array(dtype=wp.vec3),
               pool_body: wp.array(dtype=int), pool_kind: wp.array(dtype=int), pool_n: wp.array(dtype=wp.vec3),
               pool_d: wp.array(dtype=float), pool_a: wp.array(dtype=wp.vec3), pool_b: wp.array(dtype=wp.vec3),
               pool_c: wp.array2d(dtype=f64), pool_vol: wp.array(dtype=f64), pool_foam: wp.array(dtype=f64), n_pool: int,
               h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64), gamma: wp.array(dtype=f64),
               gamma_m: wp.array(dtype=float), soil: wp.array(dtype=f64), soil_w: wp.array(dtype=float),
               soil_sig: wp.array(dtype=float), soil_tr: wp.array(dtype=float), area: wp.array(dtype=float),
               sub: wp.array(dtype=int), fl: Fluid, k_a: float, k_d: float, k_m0: float, k_m1: float, k_e: float,
               lld_k: float, h_lld_max: float, foam_phi: float, dt: float,
               pool_m: wp.array(dtype=f64), pool_sp: wp.array2d(dtype=f64), pool_out: wp.array2d(dtype=f64),
               pool_foam_take: wp.array2d(dtype=f64), pool_left: wp.array2d(dtype=f64)):
    """Cells under a pool's free surface: their film joins the pool; adsorbed surfactant and soil exchange with the
    well-mixed pool. A cell that emerges carries out a Landau-Levich-Derjaguin film of the pool's liquid (and the
    foam it is lifted through).
    Every cell draws on what the pool held at the start of the tick (pool_left: volume, detergent, foam liquid; drawn
    down atomically, like the film a splattering packet takes), so together they never take more than was there. (First
    build: each emerging cell checked only that the pool held 20 times its own film. A splash of 10 uL in the parked can
    was overdrawn to -12.5 uL and -9.3 ug of detergent by a hundred cells emerging in one tick; clean hose water landing
    in it then gave the pool a negative concentration, the films it coated carried negative detergent, and the clamp at
    the cell update created what was missing: the flush-floor demonstration's detergent ledger broke.)"""
    c = wp.tid()
    b = cell_body[c]
    x = xw[c]
    found = int(-1)
    own = own_pool[c]
    for k in range(n_pool):
        if own >= 0 and k != own:
            continue
        pb_ = pool_body[k]
        if own < 0 and pb_ == b and b >= 0:
            continue                      # an item's outside never lies in its own cavity
        xl = x
        if pb_ >= 0:
            xl = wp.transform_point(wp.transform_inverse(body_X[pb_]), x)
        if wp.dot(xl, pool_n[k]) >= pool_d[k]:
            continue
        pa = pool_a[k]
        pbb = pool_b[k]
        inside = int(0)
        if pool_kind[k] == 0:
            if xl[0] > pa[0] and xl[0] < pbb[0] and xl[1] > pa[1] and xl[1] < pbb[1] and xl[2] > pa[2] and xl[2] < pbb[2]:
                inside = 1
        else:
            z = xl[2]
            if z > pa[0] - 0.002 and z < pa[1]:
                rr = pbb[0] + (pbb[1] - pbb[0])*(z - pa[0])/(pa[1] - pa[0]) + 0.002
                if xl[0]*xl[0] + xl[1]*xl[1] < rr*rr:
                    inside = 1
        if inside == 1 and pool_vol[k] > f64(1e-9):
            found = k
            break
    prev = sub[c]
    A = f64(area[c])
    vrel = float(0.0)
    if b >= 0:
        vrel = wp.length(body_v[b] + wp.cross(body_w[b], x - body_com[b]))
    if found >= 0:
        k = found
        # the film joins the pool
        hc = h[c]
        if hc > f64(0.0):
            wp.atomic_add(pool_m, k, hc*A*f64(fl.rho))
            for kk in range(6):
                wp.atomic_add(pool_sp, kk, k, sp[kk, c]*A)
                sp[kk, c] = f64(0.0)
            h[c] = f64(0.0)
            q[c] = wp.vec3(0.0, 0.0, 0.0)
        else:
            for kk in range(6):
                if sp[kk, c] > f64(0.0):
                    wp.atomic_add(pool_sp, kk, k, sp[kk, c]*A)
                    sp[kk, c] = f64(0.0)
        conc = float(pool_c[0, k])
        # adsorption equilibrium with the pool, through a mass-transfer coefficient (stirring speeds it up)
        km = k_m0 + k_m1*vrel
        g0 = float(gamma[c])
        gm = gamma_m[c]
        geq = gm*k_a*conc/(k_a*conc + k_d)
        rate = wp.min(k_d + k_a*conc, km/1e-5)
        gnew = geq + (g0 - geq)*wp.exp(-rate*dt)
        dg = f64(gnew - g0)
        if dg > f64(0.0):
            want = dg*A
            left = wp.atomic_sub(pool_left, 1, k, want)
            got = wp.min(want, wp.max(left, f64(0.0)))
            if got < want:
                wp.atomic_add(pool_left, 1, k, want - got)
            dg = got/A
        gamma[c] = gamma[c] + dg
        wp.atomic_add(pool_out, 0, k, dg*A)          # adsorbed out of the pool (negative: desorbed into it)
        # soil: hydration and erosion by the relative flow (skin friction 0.5 rho c_f v^2, c_f 0.005)
        ms = soil[c]
        if ms > f64(0.0):
            tw = 0.0025*fl.rho*vrel*vrel
            rem = f64(k_e*wp.max(tw - soil_sig[c], 0.0)*dt)
            if rem > ms:
                rem = ms
            if rem > f64(0.0):
                soil[c] = ms - rem
                wp.atomic_add(pool_sp, 1, k, rem*A)
                wp.atomic_add(pool_sp, 2, k, rem*A*f64(soil_tr[c]))
        sub[c] = k
    else:
        if prev >= 0:
            k = prev
            # emerging: LLD withdrawal film of the pool liquid, h = 0.94 l_c Ca^(2/3)
            lc = wp.sqrt(fl.sigma0/(fl.rho*fl.g))
            mu = fl.vogel_a*wp.pow(10.0, fl.vogel_b/(fl.T_amb + 273.15 - fl.vogel_c))
            ca = mu*wp.max(vrel, 0.01)/fl.sigma0
            hl = wp.min(lld_k*lc*wp.pow(ca, 2.0/3.0), h_lld_max)
            want = f64(hl)*A
            if pool_vol[k] > want*f64(20.0):
                left = wp.atomic_sub(pool_left, 0, k, want)
                vol = wp.min(want, wp.max(left, f64(0.0)))
                if vol < want:
                    wp.atomic_add(pool_left, 0, k, want - vol)
                if vol > f64(0.0):
                    h[c] = h[c] + vol/A
                    for kk in range(4):
                        s = pool_c[kk, k]*vol
                        if kk == 0:
                            # its detergent from the same account the submerged walls adsorb from
                            ls = wp.atomic_sub(pool_left, 1, k, s)
                            gs = wp.min(s, wp.max(ls, f64(0.0)))
                            if gs < s:
                                wp.atomic_add(pool_left, 1, k, s - gs)
                            s = gs
                        sp[kk, c] = sp[kk, c] + s/A
                        wp.atomic_add(pool_out, 1 + kk, k, s)
                    wp.atomic_add(pool_out, 5, k, vol)
                # foam on the pool surface coats what is lifted through it (up to 3 mm of foam), with the foam's own
                # detergent (pool_c[5]: per m3 of foam liquid)
                fth = pool_foam[k]
                if fth > f64(0.0):
                    wf = wp.min(fth, f64(0.003))*f64(1.0 - foam_phi)*A
                    lf = wp.atomic_sub(pool_left, 2, k, wf)
                    fl_ = wp.min(wf, wp.max(lf, f64(0.0)))
                    if fl_ < wf:
                        wp.atomic_add(pool_left, 2, k, wf - fl_)
                    if fl_ > f64(0.0):
                        sp[4, c] = sp[4, c] + fl_/A
                        sp[5, c] = sp[5, c] + fl_*pool_c[5, k]/A
                        wp.atomic_add(pool_foam_take, 0, k, fl_)
                        wp.atomic_add(pool_foam_take, 1, k, fl_*pool_c[5, k])
        sub[c] = -1


# ------------------------------------------------------------------------------------------------ reductions
@wp.kernel
def k_body_film(cell_body: wp.array(dtype=int), cell_pos: wp.array(dtype=wp.vec3), h: wp.array(dtype=f64),
                sp: wp.array2d(dtype=f64), area: wp.array(dtype=float), tau_acc: wp.array(dtype=wp.vec3), rho: float,
                inv_dt: float, out_m: wp.array(dtype=f64), out_mx: wp.array(dtype=wp.vec3), out_f: wp.array(dtype=wp.vec3),
                out_t: wp.array(dtype=wp.vec3)):
    """Per body: film + foam liquid mass and its first moment (body frame); wall shear force and moment (body frame)."""
    c = wp.tid()
    b = cell_body[c]
    if b < 0:
        tau_acc[c] = wp.vec3(0.0, 0.0, 0.0)
        return
    m = (h[c] + sp[4, c])*f64(area[c])*f64(rho)
    wp.atomic_add(out_m, b, m)
    wp.atomic_add(out_mx, b, float(m)*cell_pos[c])
    f = tau_acc[c]*rho*inv_dt
    wp.atomic_add(out_f, b, f)
    wp.atomic_add(out_t, b, wp.cross(cell_pos[c], f))
    tau_acc[c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_drain_cells(drain: wp.array(dtype=int), h: wp.array(dtype=f64), q: wp.array(dtype=wp.vec3), sp: wp.array2d(dtype=f64),
                  area: wp.array(dtype=float), rho: float, sink_m: wp.array(dtype=f64), sink_sp: wp.array2d(dtype=f64)):
    """Floor-drain strainer cells: water reaching them leaves to the building drain."""
    c = wp.tid()
    s = drain[c]
    if s < 0:
        return
    A = f64(area[c])
    if h[c] > f64(0.0):
        wp.atomic_add(sink_m, s, h[c]*A*f64(rho))
        h[c] = f64(0.0)
        q[c] = wp.vec3(0.0, 0.0, 0.0)
    for k in range(6):
        if sp[k, c] > f64(0.0):
            wp.atomic_add(sink_sp, k, s, sp[k, c]*A)       # foam liquid stays species 4 (m3) in the sink's account
            sp[k, c] = f64(0.0)


# ------------------------------------------------------------------------------------------------ host side
class WaterEngine:
    """GPU arrays and kernels of one scene's films and packets. Pools, valves and the rigid coupling live in system.py."""

    def __init__(self, surf, n_bodies, *, device="cuda:0", room_lo=(-1., -1., -0.5), room_hi=(10., 10., 4.), seed=7,
                 sinks=("robot", "fixtures", "floor", "drain", "decor")):
        self.surf, self.device = surf, device
        self.nb = max(1, int(n_bodies))
        N = surf.n
        self.N = N
        d = device
        F64 = lambda a: wp.array(np.asarray(a, np.float64), dtype=f64, device=d)
        F32 = lambda a: wp.array(np.asarray(a, np.float32), dtype=float, device=d)
        I32 = lambda a: wp.array(np.asarray(a, np.int32), dtype=int, device=d)
        V3 = lambda a: wp.array(np.asarray(a, np.float32).reshape(-1, 3), dtype=wp.vec3, device=d)
        z64 = lambda n_: wp.zeros(max(1, n_), dtype=f64, device=d)
        z32 = lambda n_: wp.zeros(max(1, n_), dtype=float, device=d)
        z3 = lambda n_: wp.zeros(max(1, n_), dtype=wp.vec3, device=d)
        self._z64, self._z32, self._z3 = z64, z32, z3
        self.cell_body = I32(surf.body_of)
        self.cell_pos, self.cell_nrm = V3(surf.pos), V3(surf.nrm)
        self.area, self.size = F32(surf.area), F32(surf.size)
        self.e0, self.e1, self.elen = I32(surf.e_c0), I32(surf.e_c1), F32(surf.e_len)
        self.en0, self.en1, self.emid = V3(surf.e_n0), V3(surf.e_n1), V3(surf.e_mid)
        self.edist = F32(np.maximum(surf.e_dist, 1e-5)) if len(surf.e_c0) else F32([1.0])
        self.ekind, self.elip = I32(surf.e_kind), F32(surf.e_lip)
        self.ebody = I32(surf.body_of[surf.e_c0]) if len(surf.e_c0) else I32([0])
        self.bc, self.blen, self.bn, self.bkind = I32(surf.b_c), F32(surf.b_len), V3(surf.b_n), I32(surf.b_kind)
        # the film side of each edge (body frame): the normal of the boundary cell, or of both cells of an interior edge
        efn = surf.nrm[surf.e_c0] + surf.nrm[surf.e_c1] if len(surf.e_c0) else np.array([[0.0, 0.0, 1.0]])
        self.e_fn = V3(efn/np.maximum(np.linalg.norm(efn, axis=1), 1e-9)[:, None])
        self.b_fn = V3(surf.nrm[surf.b_c] if len(surf.b_c) else [[0.0, 0.0, 1.0]])
        self.bmid = V3(surf.b_mid)
        self.bbody = I32(surf.body_of[surf.b_c]) if len(surf.b_c) else I32([0])
        self.adj_start, self.adj_edge = I32(surf.adj_start), I32(surf.adj_edge if len(surf.adj_edge) else [0])
        # state
        self.h, self.q = z64(N), z3(N)
        self.h0, self.take_left = z64(N), z64(N)     # per tick: the film splatter may take from (k_packets)
        self.clamped = z64(1 + NS)                   # created by clamping negatives at the cell update (diagnostic)
        self.sp = wp.zeros((NS, N), dtype=f64, device=d)
        self.gamma = z64(N)
        self.soil, self.soil_w = z64(N), z32(N)
        # per-cell material constants
        mats = [P.SUBSTRATES[m] if m in P.SUBSTRATES else P.SUBSTRATES["steel"] for m in surf.materials]
        mo = surf.mat_of
        th = np.array([m["theta"] for m in mats])[mo]
        self.theta = th
        self.duez_c = F32(P.DUEZ["K"]*(1.0 + np.cos(np.radians(th)))/P.WATER["rho"])
        self.gamma_m = F32(np.array([m["gamma_m"] for m in mats])[mo])
        self.h_break = F32(np.array([m["h_break"] for m in mats])[mo])
        self.soil_tr, self.soil_sd, self.soil_sw, self.soil_ts = z32(N), z32(N), z32(N), z32(N)
        self.soil_sig = z32(N)
        self.own_pool = I32(np.full(N, -1))
        self.drain = I32(np.full(N, -1))
        # work arrays
        self.dh, self.dq, self.dsp = z64(N), z3(N), wp.zeros((NS, N), dtype=f64, device=d)
        self.grad, self.outsum, self.lam = z3(N), z64(N), z64(N)
        self.g_t, self.g_n, self.omega_b, self.xw, self.nw = z3(N), z32(N), z3(N), z3(N), z3(N)
        self.speed, self.ratio, self.tau_acc, self.tau_w = z32(N), z32(N), z3(N), z32(N)
        self.mobile = F32(np.ones(N))
        self.foam_mob = z32(N)
        self.imp_p, self.imp_erode, self.abr = z32(N), z32(N), z64(N)
        self.blocked = wp.zeros(N, dtype=int, device=d)        # 1: under an item standing on the surface (system._cover)
        self.front_rm = z64(N)
        self.sub = I32(np.full(N, -1))
        E, B = len(surf.e_c0), len(surf.b_c)
        self.E, self.B = E, B
        self.flux, self.eject = z32(E), wp.zeros(max(E, 1), dtype=int, device=d)
        self.bflux = z32(B)
        self.ej_m, self.ej_p, self.ej_sp = z64(E), z3(E), wp.zeros((NS, max(E, 1)), dtype=f64, device=d)
        self.bej_m, self.bej_p, self.bej_sp = z64(B), z3(B), wp.zeros((NS, max(B, 1)), dtype=f64, device=d)
        self.fl = fl = Fluid()
        W = P.WATER
        fl.rho, fl.g, fl.h_dry, fl.beta, fl.tau_y = W["rho"], P.GRAVITY, H_DRY, P.FILM["beta"], P.FOAM["tau_y"]
        fl.re_turb = P.FILM["re_turb"]
        fl.vogel_a, fl.vogel_b, fl.vogel_c = P.VOGEL["A"], P.VOGEL["B"], P.VOGEL["C"]
        fl.T_amb = 20.0
        fl.sigma0, fl.sigma_cmc, fl.cmc, fl.k_sigma = W["sigma"], W["sigma_cmc"], P.SURFACTANT["cmc"], P.SURFACTANT["K_sigma"]
        self.spl = s = Splash()
        S = P.SPLATTER
        s.rho, s.sigma, s.mu = W["rho"], W["sigma"], 1.0e-3
        s.onset_a, s.onset_b = S["onset_a"], S["onset_b"]
        (s.we0, s.x0), (s.we1, s.x1), (s.we2, s.x2) = S["plateau"]
        s.ring, s.speed, s.elev_lo, s.elev_hi = S["ring_d"], S["speed"], S["elev"][0], S["elev"][1]
        s.drop_k, s.breakup_k = S["drop_we_splash"], P.JET["breakup_k"]
        # bodies
        nb = self.nb
        self.body_X = wp.array(np.tile(np.array([0, 0, 0, 0, 0, 0, 1], np.float32), (nb, 1)), dtype=wp.transform, device=d)
        self.body_v, self.body_w, self.body_com = z3(nb), z3(nb), z3(nb)
        self.body_acc, self.body_alpha = z3(nb), z3(nb)
        self.body_f, self.body_t = z3(nb), z3(nb)
        self.bf_m, self.bf_mx, self.bf_f, self.bf_t = z64(nb), z3(nb), z3(nb), z3(nb)
        self._acc_f = np.zeros((nb, 3)); self._alpha_f = np.zeros((nb, 3))
        self._prev_v = None
        # meshes
        self.sinks = list(sinks)
        self.meshes, self.mesh_body, self._lo_local, self._hi_local = [], [], [], []
        self._face_cell, self._face_sink, self._face_pool = [], [], []
        for b in sorted(surf.tri_v):
            Fc = surf.tri_f[b]
            self._add_mesh(b, surf.tri_v[b], Fc, surf.tri_cell[b], np.full(len(Fc), -1), np.full(len(Fc), -1))
        # packets
        self.cap = P.PACKETS["capacity"]
        cap = self.cap
        self.alive = wp.zeros(cap, dtype=int, device=d)
        self.px, self.pv = z3(cap), z3(cap)
        self.pm, self.pd, self.page, self.pdist = z64(cap), z32(cap), z32(cap), z32(cap)
        self.pkind = wp.zeros(cap, dtype=int, device=d)
        self.psp = wp.zeros((NS, cap), dtype=f64, device=d)
        self.head = wp.zeros(1, dtype=int, device=d)
        self.lost = z64(1 + NS + 3)     # water, its species, then the water by cause: aged out, left the room, overwritten
        self.counts = wp.zeros(4, dtype=int, device=d)
        self.diag = z64(8)          # [jet arriving, jet splattered, drop arriving, drop splattered, lifted from under the
                                    # floor, kept by the floor plane unstruck] (kg), diagnostics only
        self.sink_m = z64(64)
        self.sink_sp = wp.zeros((NS, 64), dtype=f64, device=d)
        self.shed_m = z64(64)       # water that struck an uncharted collider and ran off it (kg per account), diagnostics
        self.sink_mode = wp.zeros(64, dtype=int, device=d)      # 0 keeps, 1 thin part (sheds all), 2 sheds off steep faces
        self.room_lo, self.room_hi = wp.vec3(*room_lo), wp.vec3(*room_hi)
        self.seed = int(seed)
        self.tick = 0
        self.emitted = np.zeros(1 + NS)      # host account of everything emitted by outlets (float64)
        self.set_pools([])
        self.finalize_meshes()

    # ------------------------------------------------------------------ meshes
    def _add_mesh(self, body, V, F, face_cell, face_sink, face_pool):
        V = np.asarray(V, np.float32).reshape(-1, 3)
        F = np.asarray(F, np.int32).reshape(-1, 3)
        mesh = wp.Mesh(points=wp.array(V, dtype=wp.vec3, device=self.device),
                       indices=wp.array(F.ravel(), dtype=int, device=self.device))
        self.meshes.append(mesh)
        self.mesh_body.append(int(body))
        self._lo_local.append(V.min(axis=0)); self._hi_local.append(V.max(axis=0))
        self._face_cell.append(np.asarray(face_cell, np.int32))
        self._face_sink.append(np.asarray(face_sink, np.int32))
        self._face_pool.append(np.asarray(face_pool, np.int32))

    def sink_index(self, name):
        if name not in self.sinks:
            self.sinks.append(name)
        return self.sinks.index(name)

    def set_sink_modes(self, modes):
        """modes: {account name: 0 keeps what strikes it, 1 thin part (all of it runs off), 2 runs off steep faces}."""
        a = np.zeros(64, np.int32)
        for name, mode in modes.items():
            a[self.sink_index(name)] = int(mode)
        self.sink_mode = wp.array(a, dtype=int, device=self.device)

    def add_sink_mesh(self, body, V, F, sink_name):
        F = np.asarray(F, np.int32).reshape(-1, 3)
        self._add_mesh(body, V, F, np.full(len(F), -1), np.full(len(F), self.sink_index(sink_name)), np.full(len(F), -1))

    def add_pool_wall_mesh(self, body, V, F, pool_index):
        F = np.asarray(F, np.int32).reshape(-1, 3)
        self._add_mesh(body, V, F, np.full(len(F), -1), np.full(len(F), self.sink_index("fixtures")), np.full(len(F), pool_index))

    def finalize_meshes(self):
        d = self.device
        self.mesh_ids = wp.array(np.array([m.id for m in self.meshes], np.uint64), dtype=wp.uint64, device=d)
        self.mesh_body_a = wp.array(np.array(self.mesh_body, np.int32), dtype=int, device=d)
        face0 = np.concatenate([[0], np.cumsum([len(x) for x in self._face_cell])])[:-1].astype(np.int32)
        self.mesh_face0 = wp.array(face0, dtype=int, device=d)
        cat = lambda xs: np.concatenate(xs).astype(np.int32) if xs else np.zeros(1, np.int32)
        self.face_cell = wp.array(cat(self._face_cell), dtype=int, device=d)
        self.face_sink = wp.array(cat(self._face_sink), dtype=int, device=d)
        self.face_pool = wp.array(cat(self._face_pool), dtype=int, device=d)
        self.mesh_lo = wp.zeros(len(self.meshes), dtype=wp.vec3, device=d)
        self.mesh_hi = wp.zeros(len(self.meshes), dtype=wp.vec3, device=d)
        self._corners = []
        for lo, hi in zip(self._lo_local, self._hi_local):
            lo = lo - 0.003; hi = hi + 0.003
            self._corners.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
        self._static_boxes()

    def _static_boxes(self, q=None, R=None):
        lo = np.zeros((len(self.meshes), 3), np.float32); hi = np.zeros((len(self.meshes), 3), np.float32)
        for k, (b, cn) in enumerate(zip(self.mesh_body, self._corners)):
            pts = cn if (b < 0 or q is None) else cn @ R[b].T + q[b, :3]
            lo[k], hi[k] = pts.min(axis=0), pts.max(axis=0)
        self.mesh_lo.assign(lo); self.mesh_hi.assign(hi)

    # ------------------------------------------------------------------ bodies
    def set_bodies(self, body_q, body_qd, body_com_local, dt):
        """body_q (nb, 7: p, quat xyzw) of the body origins; body_qd (nb, 6: v_com, omega) world; body_com_local (nb, 3).
        The rigid acceleration is the 8 ms low-pass of the per-tick velocity change."""
        from scipy.spatial.transform import Rotation
        q = np.asarray(body_q, float)
        qd = np.asarray(body_qd, float)
        R = Rotation.from_quat(q[:, 3:7]).as_matrix()
        com = q[:, :3] + np.einsum("bij,bj->bi", R, np.asarray(body_com_local, float))
        v, w = qd[:, :3], qd[:, 3:6]
        if self._prev_v is not None and dt > 0:
            k = min(1.0, dt/P.FILM["accel_filter_s"])
            self._acc_f += k*((v - self._prev_v[0])/dt - self._acc_f)
            self._alpha_f += k*((w - self._prev_v[1])/dt - self._alpha_f)
        self._prev_v = (v.copy(), w.copy())
        X = np.zeros((len(q), 7), np.float32); X[:, :3] = q[:, :3]; X[:, 3:] = q[:, 3:7]
        self.body_X.assign(X)
        self.body_v.assign(v.astype(np.float32)); self.body_w.assign(w.astype(np.float32))
        self.body_com.assign(com.astype(np.float32))
        self.body_acc.assign(self._acc_f.astype(np.float32)); self.body_alpha.assign(self._alpha_f.astype(np.float32))
        self.R, self.q_now, self.com_w = R, q, com
        self._static_boxes(q, R)

    # ------------------------------------------------------------------ pools (geometry pushed by water.pools)
    def set_pools(self, pools):
        """pools: [dict(body, kind 0 box | 1 revolved, n (3) body frame, d, a (3), b (3))] -- free-surface planes and
        regions; arrays sized for at least 32 pools."""
        d = self.device
        k = max(32, len(pools))
        self.n_pool = len(pools)
        pad = lambda key, default: [p[key] for p in pools] + [default]*(k - len(pools))
        self.pool_body = wp.array(np.array(pad("body", 0), np.int32), dtype=int, device=d)
        self.pool_kind = wp.array(np.array(pad("kind", 0), np.int32), dtype=int, device=d)
        self.pool_n = wp.array(np.array(pad("n", (0., 0., 1.)), np.float32), dtype=wp.vec3, device=d)
        self.pool_d = wp.array(np.array(pad("d", -1e3), np.float32), dtype=float, device=d)
        self.pool_a = wp.array(np.array(pad("a", (0., 0., 0.)), np.float32), dtype=wp.vec3, device=d)
        self.pool_b = wp.array(np.array(pad("b", (0., 0., 0.)), np.float32), dtype=wp.vec3, device=d)
        if not hasattr(self, "pool_m") or self.pool_m.shape[0] < k:
            self.pool_m = wp.zeros(k, dtype=f64, device=d)
            self.pool_sp = wp.zeros((NS, k), dtype=f64, device=d)
            self.pool_imp = wp.zeros(k, dtype=wp.vec3, device=d)
            self.pool_c = wp.zeros((NS, k), dtype=f64, device=d)
            self.pool_vol = wp.zeros(k, dtype=f64, device=d)
            self.pool_foam = wp.zeros(k, dtype=f64, device=d)
            self.pool_out = wp.zeros((NS, k), dtype=f64, device=d)
            self.pool_foam_take = wp.zeros((2, k), dtype=f64, device=d)     # foam liquid (m3), its detergent (kg)
            self.pool_budget = wp.zeros((3, k), dtype=f64, device=d)        # held at the tick's start: m3, kg, m3 foam
            self.pool_left = wp.zeros((3, k), dtype=f64, device=d)

    def update_pool_planes(self, n_list, d_list, conc, vol, foam, budget):
        """Per tick: the pools' current free-surface planes (body frame), concentrations (NS, k), volumes, foam, and what
        each holds for its walls and emerging films to draw on (3, k: volume, detergent, foam liquid)."""
        k = self.n_pool
        if k == 0:
            return
        nn = self.pool_n.numpy(); dd = self.pool_d.numpy()
        nn[:k] = np.asarray(n_list, np.float32); dd[:k] = np.asarray(d_list, np.float32)
        self.pool_n.assign(nn); self.pool_d.assign(dd)
        c = np.zeros((NS, self.pool_c.shape[1])); c[:, :k] = conc
        self.pool_c.assign(c)
        v = np.zeros(self.pool_vol.shape[0]); v[:k] = vol; self.pool_vol.assign(v)
        f = np.zeros(self.pool_foam.shape[0]); f[:k] = foam; self.pool_foam.assign(f)
        b = np.zeros((3, self.pool_budget.shape[1])); b[:, :k] = np.maximum(budget, 0.0); self.pool_budget.assign(b)

    # ------------------------------------------------------------------ packets
    def _mesh_args(self):
        """The packet colliders for a birth-point guard (none yet before finalize_meshes)."""
        if not hasattr(self, "mesh_ids"):
            d = self.device
            return [wp.zeros(1, dtype=wp.uint64, device=d), wp.zeros(1, dtype=int, device=d), wp.zeros(1, dtype=wp.vec3, device=d),
                    wp.zeros(1, dtype=wp.vec3, device=d), 0, self.body_X]
        return [self.mesh_ids, self.mesh_body_a, self.mesh_lo, self.mesh_hi, len(self.meshes), self.body_X]

    def emit(self, x, v, m, d, kind, sp, source=True, guard=None):
        """Queue new packets (numpy: x (n, 3), v (n, 3), m (n,) kg, d (n,), kind (n,), sp (NS, n) per packet).
        source=True: new water from an outlet (added to the emitted account); False: water already in the system (a
        pool's overflow or holes) re-entering as packets."""
        n = len(m)
        if n == 0:
            return
        dev = self.device
        sp = np.asarray(sp, np.float64).reshape(NS, n)
        if guard is None:
            guard = not source             # water re-entering from pools and sponges is guarded; outlets' nozzles are clear
        wp.launch(k_spawn, dim=n, inputs=[n, self.cap, self.head,
                                          wp.array(np.asarray(x, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev),
                                          wp.array(np.asarray(v, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev),
                                          wp.array(np.asarray(m, np.float64), dtype=f64, device=dev),
                                          wp.array(np.asarray(d, np.float32), dtype=float, device=dev),
                                          wp.array(np.asarray(kind, np.int32), dtype=int, device=dev),
                                          wp.array(sp, dtype=f64, device=dev),
                                          self.alive, self.px, self.pv, self.pm, self.pd, self.pkind, self.page, self.pdist,
                                          self.psp, self.lost, int(guard), *self._mesh_args()], device=dev)
        if source:
            self.emitted[0] += float(np.sum(m)); self.emitted[1:] += sp.sum(axis=1)

    def step_packets(self, dt):
        self.counts.zero_()
        wp.copy(self.h0, self.h)            # the film at the start of the tick, which splattering packets draw on
        wp.copy(self.take_left, self.h)
        wp.launch(k_packets, dim=self.cap, inputs=[
            self.alive, self.px, self.pv, self.pm, self.pd, self.pkind, self.page, self.pdist, self.psp, self.head, self.cap,
            self.mesh_ids, self.mesh_body_a, self.mesh_face0, self.mesh_lo, self.mesh_hi, len(self.meshes),
            self.face_cell, self.face_sink, self.face_pool, self.body_X, self.body_v, self.body_w, self.body_com,
            self.cell_pos, self.cell_nrm, self.area, self.adj_start, self.adj_edge, self.e0, self.e1, self.en0, self.en1, self.elen,
            self.h, self.q, self.sp, self.sub, self.soil_sig,
            self.pool_body, self.pool_kind, self.pool_n, self.pool_d, self.pool_a, self.pool_b, self.n_pool,
            wp.vec3(0., 0., -P.GRAVITY), P.AIR["mu"], P.AIR["rho"], self.fl, self.spl, dt, self.seed + self.tick,
            self.room_lo, self.room_hi, P.PACKETS["max_age_s"],
            self.body_f, self.body_t, self.imp_erode, self.imp_p, self.sink_m, self.sink_sp, self.pool_m, self.pool_sp,
            self.pool_imp, self.lost, self.counts, self.diag, self.sink_mode, self.shed_m, self.blocked, self.h0,
            self.take_left, self.sinks.index("floor") if "floor" in self.sinks else -1], device=self.device)
        self.tick += 1

    def emit_edges(self, force=False):
        md = wp.float64(P.PACKETS["drop_mass"])
        common = [self.body_X, self.body_v, self.body_w, self.body_com, self.fl, md, int(force), self.head, self.cap,
                  self.alive, self.px, self.pv, self.pm, self.pd, self.pkind, self.page, self.pdist, self.psp, self.lost,
                  self.mesh_ids, self.mesh_body_a, self.mesh_lo, self.mesh_hi, len(self.meshes)]
        if self.E:
            wp.launch(k_emit_edges, dim=self.E, inputs=[self.E, self.emid, self.ebody, self.en1, self.e_fn, self.ej_m,
                                                        self.ej_p, self.ej_sp, *common], device=self.device)
        if self.B:
            wp.launch(k_emit_edges, dim=self.B, inputs=[self.B, self.bmid, self.bbody, self.bn, self.b_fn, self.bej_m,
                                                        self.bej_p, self.bej_sp, *common], device=self.device)

    # ------------------------------------------------------------------ films
    def cell_frames(self):
        wp.launch(k_cell_frame, dim=self.N, inputs=[self.cell_body, self.cell_pos, self.cell_nrm, self.body_X, self.body_acc,
                                                    self.body_alpha, self.body_w, self.body_com, wp.vec3(0., 0., -P.GRAVITY),
                                                    self.g_t, self.g_n, self.omega_b, self.xw, self.nw], device=self.device)

    def film_substep(self, dt):
        d = self.device
        self.outsum.zero_()
        if self.E:
            wp.launch(k_grad, dim=self.E, inputs=[self.e0, self.e1, self.elen, self.en0, self.en1, self.h, self.grad,
                                                  self.blocked], device=d)
        if self.B:
            wp.launch(k_bedge_grad, dim=self.B, inputs=[self.bc, self.blen, self.bn, self.bkind, self.h, self.grad], device=d)
        if self.E:
            wp.launch(k_edge_flux, dim=self.E, inputs=[self.e0, self.e1, self.elen, self.en0, self.en1, self.edist, self.ekind,
                                                       self.elip, self.h, self.q, self.sp, self.mobile, self.duez_c, self.grad,
                                                       self.area, self.g_n, self.fl, dt, self.flux, self.outsum, self.eject,
                                                       self.blocked],
                      device=d)
        if self.B:
            wp.launch(k_bedge_flux, dim=self.B, inputs=[self.bc, self.blen, self.bn, self.bkind, self.h, self.q, self.mobile,
                                                        self.fl, dt, self.bflux, self.outsum], device=d)
        wp.launch(k_limit, dim=self.N, inputs=[self.h, self.area, self.outsum, self.lam], device=d)
        if self.E:
            wp.launch(k_edge_apply, dim=self.E, inputs=[self.e0, self.e1, self.elen, self.en0, self.en1, self.flux, self.eject,
                                                        self.lam, self.h, self.q, self.sp, self.foam_mob, self.fl, dt, self.dh,
                                                        self.dq, self.dsp, self.ej_m, self.ej_p, self.ej_sp], device=d)
        if self.B:
            wp.launch(k_bedge_apply, dim=self.B, inputs=[self.bc, self.blen, self.bflux, self.lam, self.h, self.q, self.sp,
                                                         self.fl, dt, self.dh, self.dq, self.dsp, self.bej_m, self.bej_p,
                                                         self.bej_sp], device=d)
        wp.launch(k_cell_update, dim=self.N, inputs=[self.h, self.q, self.sp, self.area, self.cell_nrm, self.g_t, self.g_n,
                                                     self.omega_b, self.grad, self.dh, self.dq, self.dsp, self.h_break, self.fl,
                                                     dt, self.tau_acc, self.tau_w, self.speed, self.size, self.ratio,
                                                     self.mobile, self.foam_mob, self.sub, self.clamped], device=d)

    def species_step(self, dt):
        S, Fm, SO = P.SURFACTANT, P.FOAM, P.SOIL
        d = self.device
        if self.E:
            wp.launch(k_front, dim=self.E, inputs=[self.e0, self.e1, self.elen, self.en0, self.en1, self.h, self.q, self.sp,
                                                   self.soil, self.soil_w, self.soil_tr, self.area, self.fl, SO["k_front"],
                                                   SO["m_clean"], dt, self.front_rm], device=d)
            wp.launch(k_front_apply, dim=self.N, inputs=[self.soil, self.soil_tr, self.sp, self.front_rm], device=d)
        wp.launch(k_species, dim=self.N, inputs=[
            self.h, self.sp, self.gamma, self.gamma_m, self.soil, self.soil_w, self.soil_tr, self.soil_sd, self.soil_sw,
            self.soil_ts, self.soil_sig, self.tau_w, self.imp_erode, self.abr, self.sub,
            self.pool_c, self.fl, S["K"]*S["k_d"], S["k_d"],
            math.log(2.0)/Fm["half_life"], Fm["k_dil"], S["cmc"], Fm["phi"], SO["k_i"], SO["k_e"], P.FILM["h_wet"],
            SO["soak_doubling_C"], dt], device=d)

    def submerge(self, dt):
        S, PO = P.SURFACTANT, P.POOL
        if self.n_pool == 0:
            return
        self.pool_out.zero_(); self.pool_foam_take.zero_()
        wp.copy(self.pool_left, self.pool_budget)
        wp.launch(k_submerge, dim=self.N, inputs=[
            self.cell_body, self.cell_pos, self.xw, self.own_pool, self.body_X, self.body_v, self.body_w, self.body_com,
            self.pool_body, self.pool_kind, self.pool_n, self.pool_d, self.pool_a, self.pool_b, self.pool_c, self.pool_vol,
            self.pool_foam, self.n_pool, self.h, self.q, self.sp, self.gamma, self.gamma_m, self.soil, self.soil_w,
            self.soil_sig, self.soil_tr, self.area, self.sub, self.fl, S["K"]*S["k_d"], S["k_d"], PO["k_m0"], PO["k_m1"],
            P.SOIL["k_e"], PO["lld_k"], PO["h_lld_max"], P.FOAM["phi"], dt, self.pool_m, self.pool_sp, self.pool_out,
            self.pool_foam_take, self.pool_left], device=self.device)

    def drips(self):
        F = P.FILM
        wp.launch(k_drip, dim=self.N, inputs=[self.h, self.q, self.sp, self.area, self.g_n, self.xw, self.nw, self.cell_body,
                                              self.sub, self.body_X, self.body_v, self.body_w, self.body_com, self.fl,
                                              F["h_drip"], F["drip_gn"], self.head, self.cap, self.alive, self.px, self.pv,
                                              self.pm, self.pd, self.pkind, self.page, self.pdist, self.psp, self.lost,
                                              self.mesh_ids, self.mesh_body_a, self.mesh_lo, self.mesh_hi, len(self.meshes)],
                  device=self.device)

    def drains(self):
        wp.launch(k_drain_cells, dim=self.N, inputs=[self.drain, self.h, self.q, self.sp, self.area, self.fl.rho, self.sink_m,
                                                     self.sink_sp], device=self.device)

    def body_film(self, dt_film):
        for a in (self.bf_m, self.bf_mx, self.bf_f, self.bf_t):
            a.zero_()
        wp.launch(k_body_film, dim=self.N, inputs=[self.cell_body, self.cell_pos, self.h, self.sp, self.area, self.tau_acc,
                                                   self.fl.rho, 1.0/max(dt_film, 1e-9), self.bf_m, self.bf_mx, self.bf_f,
                                                   self.bf_t], device=self.device)

    def max_ratio(self):
        out = wp.zeros(1, dtype=float, device=self.device)
        wp.launch(k_ratio_max, dim=self.N, inputs=[self.ratio, out], device=self.device)
        return out

    # ------------------------------------------------------------------ accounting (host, float64)
    def totals(self):
        A = self.area.numpy().astype(np.float64)
        rho = self.fl.rho
        sp = self.sp.numpy()
        alive = self.alive.numpy().astype(bool)
        film_w = float((self.h.numpy()*A).sum())*rho
        foam_w = float((sp[SP_FOAM]*A).sum())*rho
        ej_w = float(self.ej_m.numpy().sum() + self.bej_m.numpy().sum())*rho
        return dict(film=film_w, foam=foam_w, packets=float(self.pm.numpy()[alive].sum()), eject=ej_w,
                    sinks=self.sink_m.numpy()[:len(self.sinks)].copy(), lost=float(self.lost.numpy()[0]),
                    shed=self.shed_m.numpy()[:len(self.sinks)].copy(),
                    sp_film=(sp*A[None]).sum(axis=1), sp_packets=self.psp.numpy()[:, alive].sum(axis=1),
                    sp_eject=self.ej_sp.numpy().sum(axis=1) + self.bej_sp.numpy().sum(axis=1),
                    sp_sinks=self.sink_sp.numpy()[:, :len(self.sinks)].sum(axis=1), sp_lost=self.lost.numpy()[1:1 + NS],
                    lost_by=self.lost.numpy()[1 + NS:].copy(), clamped=self.clamped.numpy().copy(),
                    lifted=float(self.diag.numpy()[4]), floor_plane_caught=float(self.diag.numpy()[5]),
                    gamma=float((self.gamma.numpy()*A).sum()), soil=float((self.soil.numpy()*A).sum()),
                    alive=int(alive.sum()))

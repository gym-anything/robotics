"""GPU divergence-free SPH (DFSPH) liquid, two-way coupled to the rigid world.

Repository-owned liquid on Warp 1.16 (titration's Newton 1.5 lineage, titrate/newton15/sph.py; rewritten lean so the
pressure iterations stay on the GPU and only a 2-number residual crosses to the host per iteration).

Model:
  * particles of volume V = dx^3 carrying a liquid id (red/white/rose wine, water); the pressure solve uses the
    volume-fraction formulation s_i = sum_j V W_ij + s_wall (= 1 at rest; multi-liquid safe); cubic spline, support
    2 dx, normalised so a dx lattice sums to exactly 1;
  * DFSPH (Bender & Koschier 2015/2017): divergence and density projections per substep; projected relaxed Jacobi
    (signed increments, total stiffness >= 0 = free-surface condition), warm-started with the previous total;
  * walls as signed-distance density maps (Koschier & Bender 2017, distance-only form): every container's exact wall
    cross-section is a 2-D signed-distance grid in its own (r, z) frame; the bar top and floor are planes. A wall at
    distance d contributes Phi(d), the lattice it replaces (the particle's own dx lattice continued through the wall),
    and its pressure acts along the SDF normal; the reaction (and the viscous and adhesive reactions) is applied to the
    wall's rigid body as a wrench about its centre of mass. The nearest surface decides, so a 1 mm glass wall is
    one-sided from either face (wine inside, a drip outside). Akinci boundary samples, used first, carried +-50 %
    volume weights along the walls: a resting 120 mm column circulated up the wall at 1 cm/s and never settled;
  * a velocity-level non-penetration guard against the same SDFs, the finger pads and polytopes circumscribing the
    robot's link and hand hulls (swept over the substep; the crossing approach removed exactly; a particle found inside
    a container wall is moved back out, counted);
  * physical viscosity (Morris-type Laplacian) with each liquid's kinematic viscosity, fluid-fluid and at walls;
  * surface forces: the Akinci, Akinci & Teschner (2013) cohesion pair force, its coefficient normalised so the planar
    surface energy equals sigma (titrate/newton15/cohesion2013.py), acting within the surface layer (pairs with a
    particle of volume fraction < params.COHESION_SURFACE_S); wall adhesion as the same pair force integrated over a
    half-space of virtual liquid behind each wall, weighted (1 + cos theta)/2 (Young's law), on glassware limited
    beyond Duez's inertial-capillary separation speed (params.DUEZ_SEPARATION); contact-line friction on the bar top
    and floor (params.CONTACT_LINE_FRICTION_RATE, a closure for contact-angle hysteresis);
  * optional per-container gas force (gas.py) as an extra acceleration with its reaction;
  * the vertical part of a container's impact (above 2 g) is handed to its liquid as the exact uniform impulse by the
    coupling (coupling.LumpedLiquid._impact_impulse), not left to the iterative pressure solve.
In an episode particle speeds are capped at params.SPH_SPEED_CAP_M_S (8 m/s) each substep before the guard (a numerical
limiter, counted in Liquid.limiter_totals); a particle that would need more than SPH_MAX_SUBSTEPS faults the step.
"""

import math
import numpy as np
import warp as wp

from . import params as P

wp.config.quiet = True


# ============================================================================================== structs and funcs
@wp.struct
class Walls:
    n_inst: int
    inst_body: wp.array(dtype=int)
    inst_off: wp.array(dtype=int)
    inst_nr: wp.array(dtype=int)
    inst_nz: wp.array(dtype=int)
    inst_h: wp.array(dtype=float)
    inst_z0: wp.array(dtype=float)
    inst_rmax: wp.array(dtype=float)
    inst_zmax: wp.array(dtype=float)
    inst_rim: wp.array(dtype=wp.vec4)       # lip: (rim height, radius of the lip's crest, zone radius, edge radius r_i)
    sdf: wp.array(dtype=float)
    dm_off: wp.array(dtype=int)
    dm_nr: wp.array(dtype=int)
    dm_nz: wp.array(dtype=int)
    dm_h: wp.array(dtype=float)
    dm_z0: wp.array(dtype=float)
    dm_phi: wp.array(dtype=float)
    dm_dr: wp.array(dtype=float)
    dm_dz: wp.array(dtype=float)
    n_planes: int
    planes: wp.array(dtype=wp.vec4)
    plane_lo: wp.array(dtype=wp.vec3)
    plane_hi: wp.array(dtype=wp.vec3)
    bpos: wp.array(dtype=wp.vec3)
    brot: wp.array(dtype=wp.quat)
    bvel: wp.array(dtype=wp.vec3)
    bang: wp.array(dtype=wp.vec3)
    bcom: wp.array(dtype=wp.vec3)
    tab_phi: wp.array(dtype=float)
    tab_dphi: wp.array(dtype=float)
    tab_n: int
    tab_d0: float
    tab_dd: float
    h: float
    dx: float


@wp.func
def kernel_w(r: float, h: float, knorm: float):
    q = r/h
    if q >= 1.0:
        return 0.0
    if q <= 0.5:
        return knorm*(6.0*(q*q*q - q*q) + 1.0)
    a = 1.0 - q
    return knorm*2.0*a*a*a


@wp.func
def kernel_grad(d: wp.vec3, r: float, h: float, knorm: float):
    q = r/h
    if q >= 1.0 or r < 1.0e-9:
        return wp.vec3(0.0)
    g = float(0.0)
    if q <= 0.5:
        g = knorm*6.0*(3.0*q*q - 2.0*q)/h
    else:
        a = 1.0 - q
        g = -knorm*6.0*a*a/h
    return d*(g/r)


@wp.func
def cohesion_c(r: float, h: float):
    """Akinci 2013 cohesion spline (support h)."""
    if r >= h or r <= 0.0:
        return 0.0
    k = 32.0/(3.14159265*wp.pow(h, 9.0))
    a = (h - r)*(h - r)*(h - r)*r*r*r
    if 2.0*r > h:
        return k*a
    return k*(2.0*a - wp.pow(h, 6.0)/64.0)


@wp.func
def cohesion_prim(x: float, k: int):
    """Antiderivative of x^k (1-x)^3 x^3 (titrate/newton15/cohesion2013.py)."""
    return (wp.pow(x, float(k + 4))/float(k + 4) - 3.0*wp.pow(x, float(k + 5))/float(k + 5)
            + 3.0*wp.pow(x, float(k + 6))/float(k + 6) - wp.pow(x, float(k + 7))/float(k + 7))


@wp.func
def cohesion_tail(s: float, k: int):
    """(32/pi) int_s^1 x^k c(x) dx for the dimensionless Akinci cohesion kernel c (negative branch below 1/2)."""
    result = float(0.0)
    if s < 1.0:
        result = cohesion_prim(1.0, k) - cohesion_prim(wp.max(s, 0.5), k)
        if s < 0.5:
            result += 2.0*(cohesion_prim(0.5, k) - cohesion_prim(s, k))
            result -= (wp.pow(0.5, float(k + 1)) - wp.pow(s, float(k + 1)))/(64.0*float(k + 1))
    return (32.0/3.14159265)*result


@wp.func
def adhesion_halfspace(s: float):
    """Normal pull (per unit gamma * rho) of a solid half-space of virtual liquid on a particle at distance s h from its
    surface: the cohesion pair force integrated over the half-space, pi (T2(s) - s^2 T0(s)); attractive beyond ~h/2,
    repulsive closer (the kernel's negative branch), continuous everywhere (titrate cohesion2013, Young's law weight
    applied by the caller)."""
    sc = wp.max(s, 0.0)
    return 3.14159265*(cohesion_tail(sc, 2) - sc*sc*cohesion_tail(sc, 0))


@wp.func
def point_velocity(W: Walls, b: int, x: wp.vec3):
    if b < 0:
        return wp.vec3(0.0)
    return W.bvel[b] + wp.cross(W.bang[b], x - W.bcom[b])


@wp.func
def sdf_lookup(W: Walls, c: int, r: float, z: float):
    """Bilinear signed distance and its (d/dr, d/dz) gradient from container instance c's (r, z) grid."""
    hg = W.inst_h[c]
    nz = W.inst_nz[c]
    u = r/hg
    w = (z - W.inst_z0[c])/hg
    iu = int(u)
    iw = int(w)
    if iu < 0 or iw < 0 or iu >= W.inst_nr[c] - 1 or iw >= nz - 1:
        return wp.vec3(1.0e6, 0.0, 0.0)
    fu = u - float(iu)
    fw = w - float(iw)
    off = W.inst_off[c]
    d00 = W.sdf[off + iu*nz + iw]
    d10 = W.sdf[off + (iu + 1)*nz + iw]
    d01 = W.sdf[off + iu*nz + iw + 1]
    d11 = W.sdf[off + (iu + 1)*nz + iw + 1]
    d = (d00*(1.0 - fu) + d10*fu)*(1.0 - fw) + (d01*(1.0 - fu) + d11*fu)*fw
    gr = ((d10 - d00)*(1.0 - fw) + (d11 - d01)*fw)/hg
    gz = ((d01 - d00)*(1.0 - fu) + (d11 - d10)*fu)/hg
    return wp.vec3(d, gr, gz)


@wp.func
def wall_hit(W: Walls, k: int, xi: wp.vec3):
    """Distance and outward unit normal (world) of wall k (containers first, then planes): vec4(d, n). d = 1e6 when
    the particle is outside the wall's range."""
    if k < W.n_inst:
        b = W.inst_body[k]
        q = W.brot[b]
        pl = wp.quat_rotate_inv(q, xi - W.bpos[b])
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        if r > W.inst_rmax[k] or pl[2] < W.inst_z0[k] or pl[2] > W.inst_zmax[k]:
            return wp.vec4(1.0e6, 0.0, 0.0, 1.0)
        s = sdf_lookup(W, k, r, pl[2])
        if s[0] > W.h:
            return wp.vec4(1.0e6, 0.0, 0.0, 1.0)
        er = wp.vec3(1.0, 0.0, 0.0)
        if r > 1.0e-9:
            er = wp.vec3(pl[0]/r, pl[1]/r, 0.0)
        nl = er*s[1] + wp.vec3(0.0, 0.0, s[2])
        ln = wp.length(nl)
        if ln < 1.0e-6:
            return wp.vec4(1.0e6, 0.0, 0.0, 1.0)
        nw = wp.quat_rotate(q, nl/ln)
        return wp.vec4(s[0], nw[0], nw[1], nw[2])
    p = k - W.n_inst
    pl4 = W.planes[p]
    nn = wp.vec3(pl4[0], pl4[1], pl4[2])
    lo = W.plane_lo[p]
    hi = W.plane_hi[p]
    if lo[0] > -50.0 and nn[2] > 0.99:
        # a bounded horizontal slab (the bar top): the signed distance to the box [lo.x, hi.x] x [lo.y, hi.y] x
        # [top - t, top], so its front, back and end faces are walls too. As a plane limited to its footprint, liquid
        # draping over the front edge just below the top surface would count as up to 30 mm inside the bar: the density
        # map and the guard would shoot it out at the 8 m/s cap (E6 tip: 18 mL thrown off the bar's front at 6-8 m/s)
        ts = -lo[2]
        cs = wp.vec3(0.5*(lo[0] + hi[0]), 0.5*(lo[1] + hi[1]), pl4[3] - 0.5*ts)
        hs = wp.vec3(0.5*(hi[0] - lo[0]), 0.5*(hi[1] - lo[1]), 0.5*ts)
        rs = xi - cs
        qs = wp.vec3(wp.abs(rs[0]) - hs[0], wp.abs(rs[1]) - hs[1], wp.abs(rs[2]) - hs[2])
        qo = wp.vec3(wp.max(qs[0], 0.0), wp.max(qs[1], 0.0), wp.max(qs[2], 0.0))
        lq = wp.length(qo)
        if lq > 1.0e-9:
            nb = wp.vec3(wp.sign(rs[0])*qo[0], wp.sign(rs[1])*qo[1], wp.sign(rs[2])*qo[2])/lq
            return wp.vec4(lq, nb[0], nb[1], nb[2])
        db = qs[2]
        nb = wp.vec3(0.0, 0.0, wp.sign(rs[2]))
        if qs[0] > db:
            db = qs[0]
            nb = wp.vec3(wp.sign(rs[0]), 0.0, 0.0)
        if qs[1] > db:
            db = qs[1]
            nb = wp.vec3(0.0, wp.sign(rs[1]), 0.0)
        return wp.vec4(db, nb[0], nb[1], nb[2])
    if xi[0] < lo[0] or xi[0] > hi[0] or xi[1] < lo[1] or xi[1] > hi[1]:
        return wp.vec4(1.0e6, 0.0, 0.0, 1.0)
    d = wp.dot(nn, xi) - pl4[3]
    if d < lo[2]:
        # below the slab the plane stands for (lo[2] = -thickness): not this wall. Without the limit, liquid on the floor
        # under the bar's footprint counted as 1 m inside the bar top and the guard launched it upward at 2 m/s per
        # substep (a particle reached 20 m/s)
        return wp.vec4(1.0e6, 0.0, 0.0, 1.0)
    return wp.vec4(d, nn[0], nn[1], nn[2])


@wp.func
def wall_body(W: Walls, k: int):
    if k < W.n_inst:
        return W.inst_body[k]
    return -1


@wp.func
def wall_phi(W: Walls, k: int, xi: wp.vec3):
    """Wall density map value and its world gradient at xi: vec4(Phi, grad Phi). Containers: their exact-geometry map
    in the (r, z) frame (Koschier & Bender 2017); planes: the planar lattice table along the normal."""
    if k < W.n_inst:
        b = W.inst_body[k]
        q = W.brot[b]
        pl = wp.quat_rotate_inv(q, xi - W.bpos[b])
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        hg = W.dm_h[k]
        nz = W.dm_nz[k]
        u = r/hg
        w = (pl[2] - W.dm_z0[k])/hg
        iu = int(u)
        iw = int(w)
        if iu < 0 or iw < 0 or iu >= W.dm_nr[k] - 1 or iw >= nz - 1:
            return wp.vec4(0.0)
        fu = u - float(iu)
        fw = w - float(iw)
        o = W.dm_off[k] + iu*nz + iw
        w00 = (1.0 - fu)*(1.0 - fw)
        w10 = fu*(1.0 - fw)
        w01 = (1.0 - fu)*fw
        w11 = fu*fw
        ph = W.dm_phi[o]*w00 + W.dm_phi[o + nz]*w10 + W.dm_phi[o + 1]*w01 + W.dm_phi[o + nz + 1]*w11
        if ph <= 0.0:
            return wp.vec4(0.0)
        gr = W.dm_dr[o]*w00 + W.dm_dr[o + nz]*w10 + W.dm_dr[o + 1]*w01 + W.dm_dr[o + nz + 1]*w11
        gz = W.dm_dz[o]*w00 + W.dm_dz[o + nz]*w10 + W.dm_dz[o + 1]*w01 + W.dm_dz[o + nz + 1]*w11
        er = wp.vec3(0.0)
        if r > 1.0e-9:
            er = wp.vec3(pl[0]/r, pl[1]/r, 0.0)
        gw = wp.quat_rotate(q, er*gr + wp.vec3(0.0, 0.0, gz))
        return wp.vec4(ph, gw[0], gw[1], gw[2])
    hit = wall_hit(W, k, xi)
    if hit[0] >= W.h:
        return wp.vec4(0.0)
    pd = phi_table(W, hit[0])
    return wp.vec4(pd[0], hit[1]*pd[1], hit[2]*pd[1], hit[3]*pd[1])


@wp.func
def phi_table(W: Walls, d: float):
    """(Phi(d), dPhi/dd) of the wall density map."""
    u = (wp.max(d, W.tab_d0) - W.tab_d0)/W.tab_dd
    i = int(u)
    if i >= W.tab_n - 1:
        return wp.vec2(0.0, 0.0)
    f = u - float(i)
    return wp.vec2(W.tab_phi[i]*(1.0 - f) + W.tab_phi[i + 1]*f, W.tab_dphi[i]*(1.0 - f) + W.tab_dphi[i + 1]*f)


@wp.func
def plane_hidden(W: Walls, k: int, occ: int):
    """Plane k (k >= n_inst) is hidden from this particle behind a container wall (bit set by k_density)."""
    if k < W.n_inst:
        return False
    return ((occ >> (k - W.n_inst)) & 1) == 1


# ============================================================================================== kernels
@wp.kernel
def k_density(grid: wp.uint64, x: wp.array(dtype=wp.vec3), W: Walls, V: float, h: float, knorm: float,
              den_min: float, s_out: wp.array(dtype=float), alpha: wp.array(dtype=float), nbr: wp.array(dtype=int),
              normal: wp.array(dtype=wp.vec3), occ_out: wp.array(dtype=int)):
    """Volume fraction s and the DFSPH factor. Also the planes each particle cannot see: a plane (bar top, floor, room
    wall) is hidden behind the nearest container wall when that wall is nearer and faces the same way (normals within
    60 deg), i.e. the particle is inside a glass whose wall lies between it and the plane. Without it, wine in a glass
    lying on the bar feels the bar top through the 1 mm glass wall as extra density and is shot out at the 8 m/s cap
    (E6 tip)."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    dc = float(1.0e6)
    nc = wp.vec3(0.0)
    for k in range(W.n_inst):
        hk = wall_hit(W, k, xi)
        if hk[0] < dc:
            dc = hk[0]
            nc = wp.vec3(hk[1], hk[2], hk[3])
    occ = int(0)
    if dc < W.h:
        for p in range(W.n_planes):
            hp = wall_hit(W, W.n_inst + p, xi)
            if hp[0] < 1.0e5 and dc + 0.25*W.dx < hp[0] and wp.dot(nc, wp.vec3(hp[1], hp[2], hp[3])) > 0.5:
                occ = occ | (1 << p)
    occ_out[i] = occ
    s = float(0.0)
    g = wp.vec3(0.0)
    sq = float(0.0)
    n = int(0)
    for j in wp.hash_grid_query(grid, xi, h):
        d = xi - x[j]
        r = wp.length(d)
        if r < h:
            s += V*kernel_w(r, h, knorm)
            if j != i:
                gj = V*kernel_grad(d, r, h, knorm)
                g += gj
                sq += wp.dot(gj, gj)
                n += 1
    normal[i] = g*h
    for k in range(W.n_inst + W.n_planes):
        if plane_hidden(W, k, occ):
            continue
        wf = wall_phi(W, k, xi)
        if wf[0] > 0.0:
            s += wf[0]
            g += wp.vec3(wf[1], wf[2], wf[3])
            n += int(wf[0]*30.0)
    s_out[i] = s
    nbr[i] = n
    den = wp.dot(g, g) + sq
    # floor at a quarter of the rest lattice's value: two particles pushed almost onto each other have vanishing
    # kernel gradients, alpha ~ 1/r^2 and a velocity update ~ 1/r (a particle left at 3e5 m/s in one step)
    alpha[i] = 1.0/wp.max(den, den_min)


@wp.kernel
def k_nonpressure(grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int),
                  s: wp.array(dtype=float), normal: wp.array(dtype=wp.vec3), W: Walls, nu: wp.array(dtype=float),
                  rho0: wp.array(dtype=float), gamma: wp.array(dtype=float), wadh: wp.array(dtype=float),
                  wadh_plane: wp.array(dtype=float), V: float,
                  h: float, knorm: float, dt: float, gravity: wp.vec3, surface: int, wall_shear: float, pin_rate: float,
                  surf_s: float, occ: wp.array(dtype=int), duez_c: wp.array(dtype=float),
                  v_out: wp.array(dtype=wp.vec3), bforce: wp.array(dtype=wp.vec3), btorque: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    vi = v[i]
    li = lid[i]
    mi = rho0[li]*V
    a = gravity
    dx = W.dx
    # the nearest container wall: the local liquid thickness normal to it (Duez closure, below) is the particle's
    # distance to it plus how far its neighbours reach further out along the wall normal, plus half a particle
    kc = int(-1)
    dc = float(1.0e6)
    nc = wp.vec3(0.0)
    if surface != 0 and duez_c[li] > 0.0:
        for k in range(W.n_inst):
            hk = wall_hit(W, k, xi)
            if hk[0] < dc:
                kc = k
                dc = hk[0]
                nc = wp.vec3(hk[1], hk[2], hk[3])
    reach_n = float(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j == i:
            continue
        d = xi - x[j]
        r = wp.length(d)
        if kc >= 0 and r < h:
            reach_n = wp.max(reach_n, -wp.dot(d, nc))
        if r < h and r > 1.0e-9:
            lj = lid[j]
            nuij = 0.5*(nu[li] + nu[lj])
            gw = kernel_grad(d, r, h, knorm)
            a += 10.0*nuij*V*wp.dot(vi - v[j], d)/(r*r + 0.01*h*h)*gw
            if surface != 0:
                # Akinci cohesion pair force only (titrate cohesion2013). The Akinci curvature term was dropped: with
                # the sigma-normalised gamma it is dimensionally inconsistent (one mass factor fewer than the cohesion
                # term) and it threw meniscus particles out of a moving bottle's neck at 3-7 m/s
                # only between particles of the surface layer (volume fraction below COHESION_SURFACE_S): in the bulk
                # the pair forces cancel on average but, at a rest spacing equal to the kernel's attraction/repulsion
                # crossover (h/2 = dx), they gave the particle lattice a shear stiffness that raised the tumbler's
                # slosh frequency by 6-9 % (V2b)
                if wp.min(s[i], s[j]) < surf_s:
                    gij = 0.5*(gamma[li] + gamma[lj])
                    mj = rho0[lj]*V
                    kij = 2.0/(s[i] + s[j])
                    a += kij*(-gij*mj*cohesion_c(r, h))*d/r
    e0 = wp.max(dc + reach_n + 0.5*dx, dx)
    oc = occ[i]
    for k in range(W.n_inst + W.n_planes):
        if plane_hidden(W, k, oc):
            continue
        hit = wall_hit(W, k, xi)
        dist = hit[0]
        if dist < h:
            nw = wp.vec3(hit[1], hit[2], hit[3])
            b = wall_body(W, k)
            vb = point_velocity(W, b, xi)
            pd = phi_table(W, dist)
            # wall viscosity: the Morris term against the wall lattice (normal gradient of the density map)
            dd = wp.max(dist, 0.25*dx) + 0.5*dx
            av = 10.0*nu[li]*wp.dot(vi - vb, nw)*dd/(dd*dd + 0.01*h*h)*(nw*pd[1])
            # tangential shear against the no-slip wall (same order: nu * dv / d^2)
            vt = (vi - vb) - nw*wp.dot(vi - vb, nw)
            av -= wall_shear*2.0*nu[li]*vt/(dd*dd)*wp.min(pd[0]*2.0, 1.0)
            af = wp.vec3(0.0)
            if surface != 0:
                # adhesion: the wall as a half-space of virtual liquid weighted by Young's law (1 + cos theta)/2
                # (titrate cohesion2013); continuous in distance. The earlier potential's force jumped from its maximum
                # to zero at d = dx/2
                yw = wadh[li]
                if k >= W.n_inst:
                    yw = wadh_plane[li]
                ad = adhesion_halfspace(dist/h)
                if k < W.n_inst and ad > 0.0 and duez_c[li] > 0.0:
                    # inertial-capillary separation at the lip (params.DUEZ_SEPARATION; Duez et al. 2010): within the
                    # zone where the flow turns round the container's lip, adhesion cannot hold liquid sliding faster
                    # than U_c = sqrt(We_c sigma / (rho e0)), We_c = K (r_i + e0/2) r_i / e0^2 (1 + cos theta), with
                    # the lip's own edge radius r_i and the local liquid thickness e0; beyond U_c the attraction falls
                    # as (U_c / u_t)^2 (authored)
                    rim = W.inst_rim[k]
                    bk = W.inst_body[k]
                    pl = wp.quat_rotate_inv(W.brot[bk], xi - W.bpos[bk])
                    rr = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
                    dl = wp.sqrt((rr - rim[1])*(rr - rim[1]) + (pl[2] - rim[0])*(pl[2] - rim[0]))
                    if dl < rim[2] and rim[3] > 0.0:
                        uc = wp.sqrt(duez_c[li]*(rim[3] + 0.5*e0)*rim[3]/(e0*e0*e0))
                        ut = wp.length(vt)
                        if ut > uc:
                            ad = ad*(uc/ut)*(uc/ut)
                af = -nw*(yw*gamma[li]*rho0[li]*ad)
                if k >= W.n_inst and dist < dx and s[i] < 0.95:
                    # contact-line friction on the bar top and the floor (where spills spread): a particle touching the
                    # plane at the edge of the liquid has its sliding velocity relaxed at pin_rate, a closure for
                    # contact-angle hysteresis, which pins real drops. Without it a 2 mL spill spread over 0.4-1.2 m2
                    # as loose particles that never stopped. Not applied on glassware (drips run down a bottle)
                    af -= vt*((1.0 - wp.exp(-pin_rate*dt))/dt)
            a += av + af
            if b >= 0:
                f = -mi*(av + af)
                wp.atomic_add(bforce, b, f)
                wp.atomic_add(btorque, b, wp.cross(xi - nw*dist - W.bcom[b], f))
    v_out[i] = vi + dt*a


@wp.kernel
def k_div_source(grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), W: Walls,
                 s: wp.array(dtype=float), alpha: wp.array(dtype=float), nbr: wp.array(dtype=int), V: float, h: float,
                 knorm: float, dt: float, density_mode: int, omega: float, acc: wp.array(dtype=float),
                 occ: wp.array(dtype=int), squeezed: wp.array(dtype=int), kappa: wp.array(dtype=float),
                 err: wp.array(dtype=float)):
    """One projected relaxed-Jacobi update. Residual: predicted volume-fraction error s* - 1 (density_mode 1) or the
    velocity divergence x dt (0; particles with a full neighbourhood only, DFSPH). The stiffness total stays >= 0 (no
    tension: free-surface condition) but its increments are signed, so a warm start that overshoots is taken back.
    Writes the increment to `kappa` and the new total to `acc`; accumulates the mean and max residual of pressurised or
    compressed particles."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    vi = v[i]
    dd = float(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j == i:
            continue
        dlt = xi - x[j]
        r = wp.length(dlt)
        if r < h:
            dd += V*wp.dot(vi - v[j], kernel_grad(dlt, r, h, knorm))
    oc = occ[i]
    for k in range(W.n_inst + W.n_planes):
        if plane_hidden(W, k, oc):
            continue
        wf = wall_phi(W, k, xi)
        if wf[0] > 0.0:
            vb = point_velocity(W, wall_body(W, k), xi)
            dd += wp.dot(vi - vb, wp.vec3(wf[1], wf[2], wf[3]))
    raw = float(0.0)
    if density_mode != 0:
        raw = s[i] + dt*dd - 1.0
    elif nbr[i] >= 20:
        raw = dd*dt
    old = acc[i]
    new = wp.max(old + omega*raw*alpha[i]/(dt*dt), 0.0)
    if density_mode == 0 and nbr[i] < 20:
        new = 0.0
    if squeezed[i] != 0:
        # a particle in a gap thinner than one particle (k_guard) is not part of the pressure solve: both walls' density
        # maps overlap there and its "compression" is an artefact of the resolution. Its stiffness would drive its
        # neighbours out at the 8 m/s cap when a vessel comes down on a puddle at 5 cm/s
        new = 0.0
        raw = 0.0
    kappa[i] = new - old
    acc[i] = new
    e = wp.max(raw, 0.0)
    if new > 0.0:
        e = wp.abs(raw)
    wp.atomic_add(err, 0, e)
    wp.atomic_max(err, 1, e)


@wp.kernel
def k_apply_pressure(grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int),
                     rho0: wp.array(dtype=float), kappa: wp.array(dtype=float), W: Walls, V: float, h: float,
                     knorm: float, dt: float, squeezed: wp.array(dtype=int), occ: wp.array(dtype=int),
                     v_out: wp.array(dtype=wp.vec3),
                     bforce: wp.array(dtype=wp.vec3), btorque: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    ki = kappa[i]
    dv = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j == i:
            continue
        dlt = xi - x[j]
        r = wp.length(dlt)
        if r < h:
            dv -= dt*V*(ki + kappa[j])*kernel_grad(dlt, r, h, knorm)
    if ki != 0.0:
        mi = rho0[lid[i]]*V
        oc = occ[i]
        for k in range(W.n_inst + W.n_planes):
            if plane_hidden(W, k, oc):
                continue
            wf = wall_phi(W, k, xi)
            if wf[0] > 0.0:
                gb = wp.vec3(wf[1], wf[2], wf[3])*(dt*ki)
                dv -= gb
                b = wall_body(W, k)
                # a squeezed particle (k_guard) carries no load between the two solids it is trapped between
                if b >= 0 and squeezed[i] == 0:
                    f = gb*(mi/dt)
                    wp.atomic_add(bforce, b, f)
                    wp.atomic_add(btorque, b, wp.cross(xi - W.bcom[b], f))
    v_out[i] = v[i] + dv


@wp.func
def guard_dv(d: float, vn: float, dt: float, clearance: float):
    """Normal velocity change keeping a particle at distance d (relative normal velocity vn) from ending the substep
    closer than `clearance`: whatever would cross is removed exactly (unbounded: a 2 m/s limit per substep let particles
    driven by an impact's pressure overshoot through a bottle's 9.6 mm base). A particle already too close only has its
    approach stopped; the caller moves it out to the clearance (a position correction, no velocity added). Pushing it
    out at up to 2 m/s instead lets the two walls' pushes pump particles up to 3-8 m/s in the wedge where a glass's foot
    closes on the bar."""
    dpred = d + dt*vn
    if dpred >= clearance:
        return 0.0
    if d >= clearance:
        return (clearance - dpred)/dt
    return wp.max(-vn, 0.0)


@wp.kernel
def k_guard(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), lid: wp.array(dtype=int),
            rho0: wp.array(dtype=float), V: float, dt: float, clearance: float, W: Walls,
            boxes_c: wp.array(dtype=wp.vec3), boxes_h: wp.array(dtype=wp.vec3),
            obb_body: wp.array(dtype=int), obb_c: wp.array(dtype=wp.vec3), obb_q: wp.array(dtype=wp.quat),
            obb_h: wp.array(dtype=wp.vec3),
            hull_body: wp.array(dtype=int), hull_c: wp.array(dtype=wp.vec3), hull_r: wp.array(dtype=float),
            hull_off: wp.array(dtype=int), hull_n: wp.array(dtype=int), hull_pl: wp.array(dtype=wp.vec4),
            gap_max: float, escape: float, escape_max: float, occ: wp.array(dtype=int),
            bforce: wp.array(dtype=wp.vec3), btorque: wp.array(dtype=wp.vec3), hits: wp.array(dtype=int),
            squeezed: wp.array(dtype=int)):
    """Velocity-level non-penetration: no particle centre may end the substep closer than `clearance` to a wall. Each
    wall is tested over the distance the particle can travel in the substep (swept, not a fixed 4 m/s); the approach
    velocity that would cross is removed exactly; an equal and opposite impulse goes to the wall's body. A particle
    found inside a container wall (signed distance < 0) is moved back to the clearance along the wall normal (position
    repair, counted in hits[2]; it should not happen).
    Squeeze closure (params.SQUEEZE_GAP_RATIO): a particle between two facing walls (containers or planes) closer
    together than gap_max cannot be represented in the gap; it is put on the gap's midline, given the gap's normal
    velocity and, along the gap away from the moving body, the Reynolds squeeze-out speed r V / (2 h) bounded to
    [escape, escape_max]; it pushes on neither wall and takes no part in the pressure solve (here and, via `squeezed`,
    in the next substep). Counted in hits[3]. hits: [walls, boxes/robot, inside-wall repairs, squeezed]."""
    i = wp.tid()
    xi = x[i]
    vi = v[i]
    mi = rho0[lid[i]]*V
    reach = clearance + dt*(wp.length(vi) + 2.0)
    # the two nearest container/plane walls
    k1 = int(-1)
    k2 = int(-1)
    d1 = float(1.0e6)
    d2 = float(1.0e6)
    n1 = wp.vec3(0.0)
    n2 = wp.vec3(0.0)
    oc = occ[i]
    if gap_max > 0.0:
        for k in range(W.n_inst + W.n_planes):
            if plane_hidden(W, k, oc):
                continue
            hk = wall_hit(W, k, xi)
            if hk[0] < d1:
                k2 = k1
                d2 = d1
                n2 = n1
                k1 = k
                d1 = hk[0]
                n1 = wp.vec3(hk[1], hk[2], hk[3])
            elif hk[0] < d2:
                k2 = k
                d2 = hk[0]
                n2 = wp.vec3(hk[1], hk[2], hk[3])
    sq = int(0)
    if k2 >= 0 and d2 < 1.0e5 and d1 + d2 < gap_max and wp.dot(n1, n2) < -0.5:
        sq = 1
        # lower wall: the one whose normal points up more; nn points from it into the gap
        klo = k1
        kup = k2
        dlo = d1
        dup = d2
        nn = n1
        if n2[2] > n1[2]:
            klo = k2
            kup = k1
            dlo = d2
            dup = d1
            nn = n2
        xi = xi + nn*(0.5*(dup - dlo))
        blo = wall_body(W, klo)
        bup = wall_body(W, kup)
        vlo = point_velocity(W, blo, xi)
        vup = point_velocity(W, bup, xi)
        vn_t = 0.5*(wp.dot(vlo, nn) + wp.dot(vup, nn))
        vclose = wp.max(wp.dot(vlo, nn) - wp.dot(vup, nn), 0.0)
        gap = wp.max(dlo + dup, 0.25*W.dx)
        bes = bup
        if bes < 0:
            bes = blo
        # the in-gap velocity: the walls' mean tangential motion plus the squeeze-out along the gap. Only the latter is
        # liquid's own; it is bounded (a real film is decelerated by its lubrication pressure and leaves as a thin sheet
        # held to the bar; at 2 mm a particle cannot be such a sheet)
        vt_w = 0.5*(vlo + vup)
        vt_w = vt_w - nn*wp.dot(vt_w, nn)
        vi = nn*vn_t + vt_w
        if bes >= 0:
            rel = xi - W.bcom[bes]
            e = rel - nn*wp.dot(rel, nn)
            le = wp.length(e)
            if le > 1.0e-6:
                e = e/le
                u = wp.min(wp.max(le*vclose/(2.0*gap), escape), escape_max)
                vi = vi + e*u
        wp.atomic_add(hits, 3, 1)
    squeezed[i] = sq
    for k in range(W.n_inst + W.n_planes):
        if (sq == 1 and (k == k1 or k == k2)) or plane_hidden(W, k, oc):
            continue
        hit = wall_hit(W, k, xi)
        d = hit[0]
        b = wall_body(W, k)
        vb = point_velocity(W, b, xi)
        if d > reach + dt*wp.length(vb):
            continue
        nw = wp.vec3(hit[1], hit[2], hit[3])
        if d < 0.0 and k < W.n_inst:
            wp.atomic_add(hits, 2, 1)
        if d < clearance:
            xi = xi + nw*(clearance - d)
        vn = wp.dot(vi - vb, nw)
        dvn = guard_dv(d, vn, dt, clearance)
        if dvn > 0.0:
            vi = vi + nw*dvn
            if b >= 0:
                f = -mi*dvn/dt*nw
                wp.atomic_add(bforce, b, f)
                wp.atomic_add(btorque, b, wp.cross(xi - nw*d - W.bcom[b], f))
            wp.atomic_add(hits, 0, 1)
    for k in range(boxes_c.shape[0]):
        rel = xi - boxes_c[k]
        hb = boxes_h[k]
        dx_ = wp.abs(rel[0]) - hb[0]
        dy_ = wp.abs(rel[1]) - hb[1]
        dz_ = wp.abs(rel[2]) - hb[2]
        if dx_ > reach or dy_ > reach or dz_ > reach:
            continue
        nw = wp.vec3(0.0, 0.0, 1.0)
        d = dz_
        if rel[2] < 0.0:
            nw = wp.vec3(0.0, 0.0, -1.0)
        if dx_ > d:
            d = dx_
            nw = wp.vec3(wp.sign(rel[0]), 0.0, 0.0)
        if dy_ > d:
            d = dy_
            nw = wp.vec3(0.0, wp.sign(rel[1]), 0.0)
        if d < clearance:
            xi = xi + nw*(clearance - d)
        dvn = guard_dv(d, wp.dot(vi, nw), dt, clearance)
        if dvn > 0.0:
            vi = vi + nw*dvn
            wp.atomic_add(hits, 1, 1)
    # moving oriented boxes on rigid bodies (the fingers' pads): same guard, with reaction
    for k in range(obb_body.shape[0]):
        b = obb_body[k]
        if b < 0:
            continue
        qb = W.brot[b]
        pb = wp.quat_rotate_inv(qb, xi - W.bpos[b])
        ps = wp.quat_rotate_inv(obb_q[k], pb - obb_c[k])
        hb = obb_h[k]
        vb = point_velocity(W, b, xi)
        rk = reach + dt*wp.length(vb)
        ex = wp.abs(ps[0]) - hb[0]
        ey = wp.abs(ps[1]) - hb[1]
        ez = wp.abs(ps[2]) - hb[2]
        if ex > rk or ey > rk or ez > rk:
            continue
        o = wp.vec3(wp.max(ex, 0.0), wp.max(ey, 0.0), wp.max(ez, 0.0))
        d = wp.length(o)
        nl = wp.vec3(0.0)
        if d > 1.0e-9:
            nl = wp.vec3(wp.sign(ps[0])*o[0], wp.sign(ps[1])*o[1], wp.sign(ps[2])*o[2])/d
        else:
            d = ez
            nl = wp.vec3(0.0, 0.0, wp.sign(ps[2]))
            if ex > d:
                d = ex
                nl = wp.vec3(wp.sign(ps[0]), 0.0, 0.0)
            if ey > d:
                d = ey
                nl = wp.vec3(0.0, wp.sign(ps[1]), 0.0)
        nw = wp.quat_rotate(qb, wp.quat_rotate(obb_q[k], nl))
        if d < clearance:
            xi = xi + nw*(clearance - d)
        dvn = guard_dv(d, wp.dot(vi - vb, nw), dt, clearance)
        if dvn > 0.0:
            vi = vi + nw*dvn
            f = -mi*dvn/dt*nw
            wp.atomic_add(bforce, b, f)
            wp.atomic_add(btorque, b, wp.cross(xi - W.bcom[b], f))
            wp.atomic_add(hits, 1, 1)
    # robot links and hand: convex polytopes circumscribing each link's collision hull (support function of the hull
    # along fixed directions, body frame); distance = the largest plane distance (exact on faces, a lower bound near
    # edges). Before this, liquid passed straight through the arm and hand (only the finger pads were boundaries)
    for k in range(hull_body.shape[0]):
        b = hull_body[k]
        if b < 0:
            continue
        qb = W.brot[b]
        pb = wp.quat_rotate_inv(qb, xi - W.bpos[b])
        vb = point_velocity(W, b, xi)
        rk = reach + dt*wp.length(vb)
        if wp.length(pb - hull_c[k]) > hull_r[k] + rk:
            continue
        d = float(-1.0e6)
        nl = wp.vec3(0.0, 0.0, 1.0)
        for m in range(hull_n[k]):
            pl = hull_pl[hull_off[k] + m]
            dm = pl[0]*pb[0] + pl[1]*pb[1] + pl[2]*pb[2] - pl[3]
            if dm > d:
                d = dm
                nl = wp.vec3(pl[0], pl[1], pl[2])
        if d > rk:
            continue
        nw = wp.quat_rotate(qb, nl)
        if d < clearance:
            xi = xi + nw*(clearance - d)
        dvn = guard_dv(d, wp.dot(vi - vb, nw), dt, clearance)
        if dvn > 0.0:
            vi = vi + nw*dvn
            f = -mi*dvn/dt*nw
            wp.atomic_add(bforce, b, f)
            wp.atomic_add(btorque, b, wp.cross(xi - W.bcom[b], f))
            wp.atomic_add(hits, 1, 1)
    x[i] = xi
    v[i] = vi


@wp.kernel
def k_xsph(grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), s: wp.array(dtype=float), V: float,
           h: float, knorm: float, eps: float, v_out: wp.array(dtype=wp.vec3)):
    """XSPH velocity filter (Monaghan 1989): v_i += eps sum_j V (v_j - v_i) W_ij 2 / (s_i + s_j). Numerical."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = x[i]
    vi = v[i]
    dv = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j == i:
            continue
        r = wp.length(xi - x[j])
        if r < h:
            dv += V*(v[j] - vi)*kernel_w(r, h, knorm)*2.0/(s[i] + s[j])
    v_out[i] = vi + eps*dv


@wp.kernel
def k_scale(a: wp.array(dtype=float), factor: float):
    i = wp.tid()
    a[i] = a[i]*factor


@wp.kernel
def k_integrate(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), dt: float, stats: wp.array(dtype=float)):
    i = wp.tid()
    vi = v[i]
    x[i] = x[i] + dt*vi
    sp = wp.length(vi)
    wp.atomic_max(stats, 0, sp)
    if wp.isnan(sp):
        wp.atomic_add(stats, 1, 1.0)


@wp.kernel
def k_damp(v: wp.array(dtype=wp.vec3), vmax: float, keep: float):
    """Damp and cap particle speeds: initial-condition preparation (Liquid.relax), and the squirt cap of every
    substep (keep = 1, params.SPH_SPEED_CAP_M_S)."""
    i = wp.tid()
    vi = v[i]*keep
    sp = wp.length(vi)
    if sp > vmax:
        vi = vi*(vmax/sp)
    v[i] = vi


@wp.kernel
def k_cap(v: wp.array(dtype=wp.vec3), vmax: float, count: wp.array(dtype=int)):
    """The squirt cap (params.SPH_SPEED_CAP_M_S), counted."""
    i = wp.tid()
    vi = v[i]
    sp = wp.length(vi)
    if sp > vmax:
        v[i] = vi*(vmax/sp)
        wp.atomic_add(count, 0, 1)


@wp.kernel
def k_speed(v: wp.array(dtype=wp.vec3), stats: wp.array(dtype=float)):
    i = wp.tid()
    sp = wp.length(v[i])
    wp.atomic_max(stats, 0, sp)
    wp.atomic_add(stats, 2, sp)


@wp.kernel
def k_advance_bodies(q0: wp.array(dtype=wp.transform), qd0: wp.array(dtype=wp.spatial_vector),
                     com_local: wp.array(dtype=wp.vec3), t: float, pos: wp.array(dtype=wp.vec3),
                     rot: wp.array(dtype=wp.quat), vel: wp.array(dtype=wp.vec3), ang: wp.array(dtype=wp.vec3),
                     com: wp.array(dtype=wp.vec3)):
    """Body poses extrapolated by t from the rigid step's start with its velocities (first order; COM-referenced)."""
    b = wp.tid()
    tf = q0[b]
    p0 = wp.transform_get_translation(tf)
    r0 = wp.transform_get_rotation(tf)
    tw = qd0[b]
    v = wp.vec3(tw[0], tw[1], tw[2])
    w = wp.vec3(tw[3], tw[4], tw[5])
    c0 = p0 + wp.quat_rotate(r0, com_local[b])
    angle = wp.length(w)*t
    r1 = r0
    if angle > 1.0e-12:
        r1 = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w), angle)*r0)
    c1 = c0 + v*t
    pos[b] = c1 - wp.quat_rotate(r1, com_local[b])
    rot[b] = r1
    vel[b] = v
    ang[b] = w
    com[b] = c1


# ============================================================================================== host side
def cubic_knorm(h):
    return 8.0/(math.pi*h**3)


def _w_np(r, h, knorm):
    q = np.asarray(r)/h
    return knorm*np.where(q <= 0.5, 6*(q**3 - q**2) + 1, np.where(q < 1, 2*(1 - q)**3, 0.))


def lattice_sum(dx, h, knorm=1.0):
    """sum_j V W(x_j) over a simple cubic lattice around a lattice point (kernel normalisation)."""
    n = int(math.ceil(h/dx)) + 1
    g = np.arange(-n, n + 1)*dx
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return float(dx**3*_w_np(np.sqrt(X**2 + Y**2 + Z**2).ravel(), h, knorm).sum())


def wall_table(dx, h, knorm, n=96):
    """Wall density map Phi(d) and dPhi/dd on d in [-dx/2, h]: the particle's own dx lattice continued through the wall,
    layers at distances d + dx/2 + k dx below it, each layer's sum averaged over in-plane offsets. At the resting
    distance d = dx/2 a lattice particle then sums to exactly 1."""
    m = int(math.ceil(h/dx)) + 2
    g = np.arange(-m, m + 1)*dx
    X, Y = np.meshgrid(g, g, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    offs = (np.stack(np.meshgrid(np.arange(8), np.arange(8), indexing="ij"), -1).reshape(-1, 2) + 0.5)/8*dx

    def layer(z):
        tot = 0.
        for ox, oy in offs:
            tot += _w_np(np.sqrt((X - ox)**2 + (Y - oy)**2 + z*z), h, knorm).sum()
        return dx**3*tot/len(offs)
    d0 = -0.5*dx
    ds = np.linspace(d0, h, n)
    phi = np.array([sum(layer(d + 0.5*dx + k*dx) for k in range(int(math.ceil(h/dx)) + 1)) for d in ds])
    dphi = np.gradient(phi, ds)
    return ds, phi.astype(np.float32), dphi.astype(np.float32)


def density_map_scale(dx, h, knorm, ds, phi):
    """Container density maps integrate W over the hidden continuum; this factor makes a flat wall at the resting
    distance dx/2 give the same value as the dx lattice it stands for (wall_table)."""
    r = np.linspace(0, h, 20001)
    d = 0.5*dx
    m = r >= d
    cont = 2*math.pi*np.trapezoid((_w_np(r, h, knorm)*r*(r - d))[m], r[m])
    return float(np.interp(d, ds, phi))/cont


def cohesion_gamma(sigma, rho0, h):
    """gamma such that the planar surface energy of the cohesion pair potential equals sigma:
    sigma = pi gamma rho0^2 / 8 * int_0^h C(s) s^4 ds."""
    s = np.linspace(0., h, 4001)
    k = 32.0/(math.pi*h**9)
    a = (h - s)**3*s**3
    c = np.where(2*s > h, k*a, k*(2*a - h**6/64))
    return 8*sigma/(math.pi*rho0**2*float(np.trapezoid(c*s**4, s)))


def lip_zone(geo):
    """The container's lip for the Duez closure: (rim height, radius of the lip's crest, radius of the zone around it
    where the flow turns round the edge, the edge's radius r_i). r_i is the lip's own rounding: half the wall at the
    rim, but no more than the LEONOR finish's 1.5 mm round (geometry._bottle) - the glasses' rims are 0.2-0.3 mm, the
    tumbler's and carafe's 1.25-1.5 mm."""
    sol = np.asarray(geo.solid, float)
    near = sol[np.abs(sol[:, 1] - geo.rim_z) < 0.003]
    r_out = float(near[:, 0].max()) if len(near) else float(geo.mouth_r)
    half_w = max(0.5*(r_out - float(geo.mouth_r)), 0.)
    r_i = min(half_w, P.DUEZ_EDGE_MAX_M)
    return (float(geo.rim_z), float(geo.mouth_r) + half_w, half_w + P.DUEZ_ZONE_M, r_i)


class Liquid:
    """All liquid in the scene. Bodies are indexed like the rigid model's bodies (-1 = static)."""

    def __init__(self, n_bodies, containers, *, dx=P.SPH_DX_M, device="cuda:0", gravity=(0., 0., -P.GRAVITY),
                 planes=(), boxes=(), obbs=(), hulls=(), surface=P.SURFACE_FORCES):
        """containers: list of (body_index, Container geometry). planes: list of (normal, offset, lo_xy, hi_xy) static
        flat walls (bar top, floor); boxes: (centre, half) static solids for the guard only; obbs: (body, centre, quat,
        half) oriented boxes on bodies (finger pads); hulls: (body, centre, radius, planes (m, 4)) convex polytopes on
        bodies (robot links and hand), all in body frames."""
        self.device, self.dx = device, float(dx)
        self.h = P.SPH_SUPPORT_RATIO*self.dx
        self.V = self.dx**3
        k0 = cubic_knorm(self.h)
        self.knorm = k0/lattice_sum(self.dx, self.h, k0)          # rest lattice -> s = 1 exactly
        self.den_min = 0.25*self._lattice_den()
        self.gravity = wp.vec3(*gravity)
        self.surface = bool(surface)
        self.n_bodies = int(n_bodies)
        self.names = list(P.LIQUIDS)
        props = [P.LIQUIDS[n] for n in self.names]
        self.rho0_np = np.array([p["rho"] for p in props], np.float32)
        nu = np.array([p["mu"]/p["rho"] for p in props], np.float32)
        gam = np.array([cohesion_gamma(p["sigma"], p["rho"], self.h)*P.COHESION_SCALE for p in props], np.float32)
        # Young-law weights (1 + cos theta)/2 of the wall half-space: glassware, and the bar top / floor planes
        wad = np.array([0.5*(1 + math.cos(math.radians(p["theta_glass_deg"])))*P.ADHESION_SCALE for p in props], np.float32)
        wadp = np.array([0.5*(1 + math.cos(math.radians(P.THETA_BAR_DEG)))*P.ADHESION_SCALE for p in props], np.float32)
        # Duez closure: K (1 + cos theta) sigma / rho per liquid (U_c^2 = this x (r_i + e0/2) r_i / e0^3)
        duez = np.array([P.DUEZ_K*(1 + math.cos(math.radians(p["theta_glass_deg"])))*p["sigma"]/p["rho"]
                         if P.DUEZ_SEPARATION else 0. for p in props], np.float32)
        self.coef = dict(nu=nu.tolist(), gamma=gam.tolist(), young_weight_glass=wad.tolist(),
                         young_weight_bar=wadp.tolist(), duez_K_one_plus_cos_sigma_over_rho=duez.tolist())
        self.containers = list(containers)
        self._sdf_cache = {}
        inst = dict(body=[], off=[], nr=[], nz=[], h=[], z0=[], rmax=[], zmax=[], rim=[])
        parts, off = [], 0
        ds, phi, dphi = wall_table(self.dx, self.h, self.knorm)
        self.dm_scale = density_map_scale(self.dx, self.h, self.knorm, ds, phi)
        dmi = dict(off=[], nr=[], nz=[], h=[], z0=[])
        dmp, dmr, dmz, doff, self._dm_cache = [], [], [], 0, {}
        for b, geo in self.containers:
            if geo.kind not in self._sdf_cache:
                self._sdf_cache[geo.kind] = geo.sdf_grid()
            g = self._sdf_cache[geo.kind]
            parts.append(g["sdf"].ravel())
            inst["body"].append(b); inst["off"].append(off); inst["nr"].append(g["nr"]); inst["nz"].append(g["nz"])
            inst["h"].append(g["h"]); inst["z0"].append(g["z0"]); inst["rmax"].append((g["nr"] - 2)*g["h"])
            inst["zmax"].append(g["z0"] + (g["nz"] - 2)*g["h"])
            inst["rim"].append(lip_zone(geo))
            off += g["sdf"].size
            if geo.kind not in self._dm_cache:
                self._dm_cache[geo.kind] = geo.density_map(self.dx, self.h, self.knorm, self.dm_scale)
            m = self._dm_cache[geo.kind]
            dmp.append(m["phi"].ravel()); dmr.append(m["dphi_dr"].ravel()); dmz.append(m["dphi_dz"].ravel())
            dmi["off"].append(doff); dmi["nr"].append(m["nr"]); dmi["nz"].append(m["nz"]); dmi["h"].append(m["h"])
            dmi["z0"].append(m["z0"])
            doff += m["phi"].size
        with wp.ScopedDevice(device):
            self.rho0 = wp.array(self.rho0_np, dtype=float)
            self.nu = wp.array(nu, dtype=float)
            self.gamma = wp.array(gam, dtype=float)
            self.wadh = wp.array(wad, dtype=float)
            self.wadh_plane = wp.array(wadp, dtype=float)
            self.duez_c = wp.array(duez, dtype=float)
            W = Walls()
            W.n_inst = len(self.containers)
            ints = lambda v: wp.array(np.array(v or [0], np.int32), dtype=int)
            flts = lambda v: wp.array(np.array(v or [0.], np.float32), dtype=float)
            W.inst_body, W.inst_off, W.inst_nr, W.inst_nz = (ints(inst[k]) for k in ("body", "off", "nr", "nz"))
            W.inst_h, W.inst_z0, W.inst_rmax, W.inst_zmax = (flts(inst[k]) for k in ("h", "z0", "rmax", "zmax"))
            W.inst_rim = wp.array(np.array(inst["rim"] or [(0., 0., 0., 0.)], np.float32), dtype=wp.vec4)
            W.sdf = wp.array(np.concatenate(parts) if parts else np.zeros(1, np.float32), dtype=float)
            W.dm_off, W.dm_nr, W.dm_nz = (ints(dmi[k]) for k in ("off", "nr", "nz"))
            W.dm_h, W.dm_z0 = flts(dmi["h"]), flts(dmi["z0"])
            cat = lambda a: wp.array(np.concatenate(a).astype(np.float32) if a else np.zeros(1, np.float32), dtype=float)
            W.dm_phi, W.dm_dr, W.dm_dz = cat(dmp), cat(dmr), cat(dmz)
            W.n_planes = len(planes)
            W.planes = wp.array(np.array([tuple(n) + (o,) for n, o, lo, hi in planes] or [(0., 0., 1., -1e9)], np.float32), dtype=wp.vec4)
            W.plane_lo = wp.array(np.array([tuple(lo[:2]) + (-(lo[2] if len(lo) > 2 else 10.),) for n, o, lo, hi in planes]
                                           or [(0., 0., -10.)], np.float32), dtype=wp.vec3)
            W.plane_hi = wp.array(np.array([tuple(hi) + (0.,) for n, o, lo, hi in planes] or [(0., 0., 0.)], np.float32), dtype=wp.vec3)
            nb = max(1, self.n_bodies)
            W.bpos = wp.zeros(nb, dtype=wp.vec3)
            W.brot = wp.array(np.tile([0., 0., 0., 1.], (nb, 1)).astype(np.float32), dtype=wp.quat)
            W.bvel = wp.zeros(nb, dtype=wp.vec3)
            W.bang = wp.zeros(nb, dtype=wp.vec3)
            W.bcom = wp.zeros(nb, dtype=wp.vec3)
            W.tab_phi, W.tab_dphi = wp.array(phi, dtype=float), wp.array(dphi, dtype=float)
            W.tab_n, W.tab_d0, W.tab_dd = len(ds), float(ds[0]), float(ds[1] - ds[0])
            W.h, W.dx = self.h, self.dx
            self.W = W
            self.boxes_c = wp.array(np.array([c for c, hh in boxes] or [(0., 0., -1e6)], np.float32), dtype=wp.vec3)
            self.boxes_h = wp.array(np.array([hh for c, hh in boxes] or [(0., 0., 0.)], np.float32), dtype=wp.vec3)
            # (body, centre in body frame, quat xyzw in body frame, half extents): robot links and fingers
            self.obb_body = wp.array(np.array([o[0] for o in obbs] or [-1], np.int32), dtype=int)
            self.obb_c = wp.array(np.array([o[1] for o in obbs] or [(0., 0., 0.)], np.float32), dtype=wp.vec3)
            self.obb_q = wp.array(np.array([o[2] for o in obbs] or [(0., 0., 0., 1.)], np.float32), dtype=wp.quat)
            self.obb_h = wp.array(np.array([o[3] for o in obbs] or [(0., 0., 0.)], np.float32), dtype=wp.vec3)
            hl = list(hulls)
            self.hull_body = wp.array(np.array([h_[0] for h_ in hl] or [-1], np.int32), dtype=int)
            self.hull_c = wp.array(np.array([h_[1] for h_ in hl] or [(0., 0., 0.)], np.float32), dtype=wp.vec3)
            self.hull_r = wp.array(np.array([h_[2] for h_ in hl] or [0.], np.float32), dtype=float)
            offs = np.cumsum([0] + [len(h_[3]) for h_ in hl])[:-1] if hl else [0]
            self.hull_off = wp.array(np.array(offs, np.int32), dtype=int)
            self.hull_n = wp.array(np.array([len(h_[3]) for h_ in hl] or [0], np.int32), dtype=int)
            self.hull_pl = wp.array(np.concatenate([np.asarray(h_[3], np.float32) for h_ in hl]) if hl
                                    else np.zeros((1, 4), np.float32), dtype=wp.vec4)
            self._q0 = wp.array(np.tile([0., 0., 0., 0., 0., 0., 1.], (nb, 1)).astype(np.float32), dtype=wp.transform)
            self._qd0 = wp.zeros(nb, dtype=wp.spatial_vector)
            self._com_local = wp.zeros(nb, dtype=wp.vec3)
            self.bforce = wp.zeros(nb, dtype=wp.vec3)
            self.btorque = wp.zeros(nb, dtype=wp.vec3)
            self.grid = wp.HashGrid(128, 128, 128)
            self.err = wp.zeros(2, dtype=float)
            self.stats = wp.zeros(3, dtype=float)
            self.hits = wp.zeros(4, dtype=int)
            self.capped = wp.zeros(1, dtype=int)
        self.n = 0
        self._alloc(0)
        self._relaxing = 0.
        self._relax_cap = 1.0
        self.extra_forces = []           # callables (liquid, dt) adding accelerations to self.v_star before projection
        self.last, self.last_err = {}, {}
        self.time = 0.
        self.substeps_total = 0
        # how often the numerical limiters acted during episodes
        self.limiter_totals = dict(steps=0, capped=0, steps_capped=0, wall_repairs=0, steps_repaired=0, squeezed=0,
                                   steps_squeezed=0, wall_follow=0, steps_wall_follow=0, impulse_steps=0)

    def _alloc(self, n):
        with wp.ScopedDevice(self.device):
            m = max(n, 1)
            self.x, self.v, self.v_star, self._tmp = (wp.zeros(m, dtype=wp.vec3) for _ in range(4))
            self.lid, self.nbr = wp.zeros(m, dtype=int), wp.zeros(m, dtype=int)
            self.s, self.alpha, self.kappa = (wp.zeros(m, dtype=float) for _ in range(3))
            self.normal = wp.zeros(m, dtype=wp.vec3)
            self.squeezed = wp.zeros(m, dtype=int)
            self.occ = wp.zeros(m, dtype=int)
            self.kacc = {True: wp.zeros(m, dtype=float), False: wp.zeros(m, dtype=float)}
            self.dt_prev = None

    # ------------------------------------------------------------------------------------------ particles
    def set_particles(self, x, v, lid, kappa=None):
        x = np.asarray(x, np.float32).reshape(-1, 3)
        v = np.asarray(v, np.float32).reshape(-1, 3)
        lid = np.asarray(lid, np.int32).reshape(-1)
        if not (len(x) == len(v) == len(lid)) or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("Finite, equal-length particle arrays required")
        self.n = len(x)
        self._alloc(self.n)
        if self.n:
            self.x.assign(x); self.v.assign(v); self.lid.assign(lid)
            if kappa is not None:
                self.kacc[True].assign(np.asarray(kappa, np.float32))
                self.dt_prev = 0.5e-3

    def read(self):
        if not self.n:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
        return self.x.numpy()[:self.n], self.v.numpy()[:self.n], self.lid.numpy()[:self.n]

    def pressure_state(self):
        return self.kacc[True].numpy()[:self.n].copy()

    def fill_container(self, geo, pose_pos, pose_rot_xyzw, volume, liquid, jitter=0.0, seed=0, pattern="rings"):
        """`volume` m^3 of particles (count = volume / dx^3) in `geo`'s interior (body frame): wall-conforming rings
        (radii R(z) - dx/2 - k dx, about dx apart around each ring, layers dx apart, the lowest layers first, each
        particle at least dx/2 - the boundary's resting distance - from the wall). A cubic lattice cut by a curved wall
        leaves on average a dx/2 gap there (54 mL short in a 769 mL bottle: a 750 mL fill did not fit); rings start
        almost at the rest packing. Returns world positions and liquid ids."""
        from scipy.spatial.transform import Rotation
        n_want = int(round(volume/self.V))
        rng = np.random.default_rng(seed)
        dx = self.dx
        poly = geo.interior
        edges = list(zip(poly, np.roll(poly, -1, axis=0)))
        layers = []
        z = float(poly[:, 1].min()) + 0.5*dx
        while z < geo.rim_z - 0.5*dx:
            xs = [a[0] + (z - a[1])*(b[0] - a[0])/(b[1] - a[1]) for a, b in edges
                  if (a[1] - z)*(b[1] - z) < 0 and abs(b[1] - a[1]) > 1e-12]
            R = max(xs) if xs else 0.
            pts = []
            if pattern == "lattice":
                g = np.arange(-R, R + dx, dx)
                X, Y = np.meshgrid(g, g, indexing="ij")
                pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1))
                R = -1.
            r = R - 0.5*dx
            # sloped walls (shoulders, bowls): the outer ring moves in until it is dx/2 from the wall along the normal
            while r > 0.35*dx and self._sdf_eval(geo, np.array([r]), np.array([z]))[0] < 0.47*dx:
                r -= 0.05*dx
            # rings down to 0.75 dx (>= 5 particles, <= 0.94 dx apart), then a centre particle if the gap allows: rings
            # of 1-2 particles near the axis put pairs 0.7 dx apart (a full bottle's first step reached 30x its
            # hydrostatic pressure on the axis and blew up)
            while r >= 0.75*dx:
                m = max(5, int(round(2*math.pi*r/dx)))
                th = 2*math.pi*(np.arange(m) + rng.random())/m
                pts.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(m, z)], 1))
                r -= dx
            if pattern != "lattice" and (r + dx >= 0.9*dx or not pts):   # smallest ring >= 0.9 dx from the axis
                pts.append(np.array([[0., 0., z]]))
            if pts:
                p3 = np.concatenate(pts)
                rr = np.hypot(p3[:, 0], p3[:, 1])
                p3 = p3[geo.interior_contains(rr, p3[:, 2]) & (self._sdf_eval(geo, rr, p3[:, 2]) > 0.45*dx)]
                layers.append(p3[rng.permutation(len(p3))])
            z += dx
        pts = np.concatenate(layers) if layers else np.zeros((0, 3))
        if len(pts) < n_want:
            raise ValueError(f"{geo.kind}: room for {len(pts)*self.V*1e6:.1f} mL, asked {volume*1e6:.1f} mL")
        pts = pts[:n_want]
        if jitter:
            pts += rng.uniform(-jitter, jitter, pts.shape)*dx
        world = Rotation.from_quat(pose_rot_xyzw).apply(pts) + np.asarray(pose_pos)
        return world.astype(np.float32), np.full(len(world), P.LIQUID_IDS[liquid], np.int32)

    def _sdf_eval(self, geo, r, z):
        g = self._sdf_cache.get(geo.kind) or geo.sdf_grid()
        u = np.clip(r/g["h"], 0, g["nr"] - 1.001)
        w = np.clip((z - g["z0"])/g["h"], 0, g["nz"] - 1.001)
        iu, iw = u.astype(int), w.astype(int)
        fu, fw = u - iu, w - iw
        s = g["sdf"]
        return ((s[iu, iw]*(1 - fu) + s[iu + 1, iw]*fu)*(1 - fw) + (s[iu, iw + 1]*(1 - fu) + s[iu + 1, iw + 1]*fu)*fw)

    # ------------------------------------------------------------------------------------------ bodies
    def set_bodies(self, body_q, body_qd, body_com_local):
        """Rigid state at the start of the rigid step: body_q (n, 7) pos + quat xyzw; body_qd (n, 6) COM linear +
        angular velocity (Newton convention); body_com_local (n, 3)."""
        self._q0.assign(np.asarray(body_q, np.float32))
        self._qd0.assign(np.asarray(body_qd, np.float32))
        self._com_local.assign(np.asarray(body_com_local, np.float32))
        self._advance_bodies(0.)

    def _advance_bodies(self, t):
        W = self.W
        wp.launch(k_advance_bodies, max(1, self.n_bodies), inputs=[self._q0, self._qd0, self._com_local, float(t)],
                  outputs=[W.bpos, W.brot, W.bvel, W.bang, W.bcom], device=self.device)

    def compute_psi(self):
        """Kept for callers of the sampled-boundary version; density maps need no precomputation."""

    # ------------------------------------------------------------------------------------------ stepping
    def step(self, dt_rigid):
        """Advance the liquid by one rigid step with CFL-adaptive substeps. Returns the time-averaged wrench on
        each body (n_bodies, 6: force at COM, torque) over the step, in world frame."""
        self.bforce.zero_(); self.btorque.zero_()
        self.hits.zero_(); self.capped.zero_()
        info = dict(substeps=0, it_div=0, it_den=0, guard_hits=0)
        if self.n == 0:
            self.last = info
            return np.zeros((self.n_bodies, 6))
        vmax = self.max_speed()
        n = int(min(P.SPH_MAX_SUBSTEPS, max(P.SPH_MIN_SUBSTEPS, math.ceil(vmax*dt_rigid/(P.SPH_CFL*self.dx)))))
        if not self._relaxing and (vmax*dt_rigid/n > P.SPH_CFL*self.dx*1.5 or vmax > P.MAX_PARTICLE_SPEED_M_S):
            v = self.v.numpy()[:self.n]
            i = int(np.argmax(np.linalg.norm(v, axis=1)))
            xi = self.x.numpy()[i]
            raise FloatingPointError(f"Liquid particle at {vmax:.2f} m/s needs more than {P.SPH_MAX_SUBSTEPS} substeps "
                                     f"(particle {i} at {np.round(xi, 4).tolist()} moving {np.round(v[i], 2).tolist()})")
        dt = dt_rigid/n
        for k in range(n):
            self._advance_bodies(k*dt)
            self._substep(dt, info)
            if self._relaxing:
                wp.launch(k_damp, self.n, inputs=[self.v, self._relax_cap, self._relaxing], device=self.device)
        self._advance_bodies(0.)
        self.time += dt_rigid
        self.substeps_total += n
        info["substeps"] = n
        info["vmax"] = vmax
        hits = self.hits.numpy()
        info.update(guard_hits=int(hits[0]), guard_hits_robot=int(hits[1]), wall_repairs=int(hits[2]),
                    squeezed=int(hits[3]), capped=int(self.capped.numpy()[0]))
        if not self._relaxing:
            for key in ("wall_repairs", "capped", "squeezed"):
                self.limiter_totals[key] += info[key]
            self.limiter_totals["steps"] += 1
            self.limiter_totals["steps_capped"] += int(info["capped"] > 0)
            self.limiter_totals["steps_repaired"] += int(info["wall_repairs"] > 0)
            self.limiter_totals["steps_squeezed"] += int(info["squeezed"] > 0)
        self.last = info
        f = self.bforce.numpy()[:self.n_bodies].astype(np.float64)/n
        t = self.btorque.numpy()[:self.n_bodies].astype(np.float64)/n
        return np.concatenate([f, t], 1)

    def hydrostatic_init(self, groups):
        """Initial pressure (warm-start stiffness kappa = p / rho) of liquid at rest: kappa_i = g (z_top - z_i) for
        each group of particle indices (one per filled container; z_top = its highest particle + dx/2). Without it the
        Jacobi solver had to build the 2.4 m^2/s^2 hydrostatic field of a full bottle through 120 particle layers."""
        x = self.x.numpy()[:self.n]
        k = np.zeros(max(self.n, 1), np.float32)
        g = float(np.linalg.norm(np.array([self.gravity[0], self.gravity[1], self.gravity[2]])))
        for idx in groups:
            idx = np.asarray(idx)
            if len(idx):
                top = float(x[idx, 2].max()) + 0.5*self.dx
                k[idx] = g*np.clip(top - x[idx, 2], 0., None)
        self.kacc[True].assign(k)
        self.kacc[False].zero_()
        self.dt_prev = 0.5e-3

    def relax_packing(self, steps=40, dt_rigid=1e-3, cap=1.0):
        """Initial-condition preparation only (never during an episode): pressure relaxation of a freshly placed packing
        (6 % over-dense at walls, 18 % on the axis). Gravity off; before each step velocities and stiffness are zeroed,
        so each density projection moves particles straight towards s = 1 and nothing accumulates kinetic energy."""
        g = self.gravity
        self.gravity = wp.vec3(0.0, 0.0, 0.0)
        self._relaxing, self._relax_cap = 1.0, float(cap)
        try:
            for _ in range(int(steps)):
                self.v.zero_()
                self.kacc[True].zero_(); self.kacc[False].zero_()
                self.step(dt_rigid)
        finally:
            self.gravity = g
            self._relaxing = 0.
        self.v.zero_()
        self.time = 0.

    def settle_static(self, steps=100, dt_rigid=1e-3, cap=1.0):
        """Initial-condition preparation only: quasi-static relaxation under gravity (after relax_packing and
        hydrostatic_init). Velocities are zeroed before every step while the pressure field is kept, so particles creep
        to the configuration where pressure balances gravity with no kinetic energy. Switching the hydrostatic pressure
        on over a packing relaxed at zero g kicked a full bottle to 140 mm/s rms, decaying over seconds."""
        self._relaxing, self._relax_cap = 1.0, float(cap)
        try:
            for _ in range(int(steps)):
                self.v.zero_()
                self.step(dt_rigid)
        finally:
            self._relaxing = 0.
        self.v.zero_()
        self.time = 0.

    def _lattice_den(self):
        """DFSPH factor denominator |sum V gradW|^2 + sum |V gradW|^2 of a particle inside a rest lattice (the first
        term vanishes by symmetry)."""
        n = int(math.ceil(self.h/self.dx))
        g = np.arange(-n, n + 1)*self.dx
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2).ravel()
        q = r/self.h
        sel = (r > 1e-12) & (q < 1.)
        q = q[sel]
        gr = np.where(q <= 0.5, self.knorm*6.*(3.*q*q - 2.*q)/self.h, -self.knorm*6.*(1. - q)**2/self.h)
        return float(np.sum((self.V*gr)**2))

    def prepare(self, groups, packing_steps=40, settle_steps=150):
        """The full initial-condition preparation of freshly filled liquid at rest: zero-g packing relaxation, hydrostatic
        pressure, then damped dynamics (speeds x 0.9 per substep) so the pressure field really carries the weight. A
        quasi-static settle (velocities zeroed every step) instead left the weight partly carried by the zeroing: the
        first free step of a full bottle pressed 1.6 N on its 6.3 N load and the column then rang (+-5 N, 80 mm/s rms);
        after the damped settle it pressed 6.28 N and stayed below 17 mm/s rms."""
        self.relax_packing(packing_steps)
        self.hydrostatic_init(groups)
        self.relax(settle_steps*1e-3, keep=0.9, cap=0.3)

    def relax(self, seconds, dt_rigid=1e-3, keep=0.98, cap=0.3):
        """Initial-condition preparation only: settle under gravity with speeds damped (x `keep` per substep, capped at
        `cap`); run after relax_packing() and hydrostatic_init()."""
        self._relaxing, self._relax_cap = float(keep), float(cap)
        try:
            for _ in range(int(round(seconds/dt_rigid))):
                self.step(dt_rigid)
        finally:
            self._relaxing = 0.
        self.time = 0.

    def max_speed(self):
        self.stats.zero_()
        wp.launch(k_speed, self.n, inputs=[self.v], outputs=[self.stats], device=self.device)
        return float(self.stats.numpy()[0])

    def _substep(self, dt, info):
        dev, n, W = self.device, self.n, self.W
        self.grid.build(self.x, self.h)
        wp.launch(k_density, n, inputs=[self.grid.id, self.x, W, self.V, self.h, self.knorm, self.den_min],
                  outputs=[self.s, self.alpha, self.nbr, self.normal, self.occ], device=dev)
        info["it_div"] += self._project(self.v, dt, density=False)
        wp.launch(k_nonpressure, n, inputs=[self.grid.id, self.x, self.v, self.lid, self.s, self.normal, W, self.nu,
                  self.rho0, self.gamma, self.wadh, self.wadh_plane, self.V, self.h, self.knorm, dt, self.gravity,
                  int(self.surface),
                  float(P.SPH_WALL_SHEAR), float(P.CONTACT_LINE_FRICTION_RATE), float(P.COHESION_SURFACE_S), self.occ,
                  self.duez_c],
                  outputs=[self.v_star, self.bforce, self.btorque], device=dev)
        for f in self.extra_forces:
            f(self, dt)
        info["it_den"] += self._project(self.v_star, dt, density=True)
        if not self._relaxing:
            # squirt cap BEFORE the guard and the integration (applied after them, a particle driven by an impact's
            # pressure overshoot was integrated through a wall before it was capped); counted in self.capped
            wp.launch(k_cap, n, inputs=[self.v_star, P.SPH_SPEED_CAP_M_S], outputs=[self.capped], device=dev)
        wp.launch(k_guard, n, inputs=[self.x, self.v_star, self.lid, self.rho0, self.V, dt,
                  P.WALL_CLEARANCE_RATIO*self.dx, W, self.boxes_c, self.boxes_h, self.obb_body, self.obb_c, self.obb_q,
                  self.obb_h, self.hull_body, self.hull_c, self.hull_r, self.hull_off, self.hull_n, self.hull_pl,
                  float(P.SQUEEZE_GAP_RATIO*self.dx), float(P.SQUEEZE_ESCAPE_M_S), float(P.SQUEEZE_ESCAPE_MAX_M_S),
                  self.occ],
                  outputs=[self.bforce, self.btorque, self.hits, self.squeezed], device=dev)
        self.stats.zero_()
        wp.launch(k_integrate, n, inputs=[self.x, self.v_star, dt], outputs=[self.stats], device=dev)
        self.v, self.v_star = self.v_star, self.v
        if P.SPH_XSPH > 0.:
            wp.launch(k_xsph, n, inputs=[self.grid.id, self.x, self.v, self.s, self.V, self.h, self.knorm, P.SPH_XSPH],
                      outputs=[self.v_star], device=dev)
            self.v, self.v_star = self.v_star, self.v
        if self.stats.numpy()[1] > 0:
            raise FloatingPointError("Non-finite liquid velocity")
        self.dt_prev = dt

    def _project(self, vel, dt, density):
        """Projected relaxed-Jacobi DFSPH projection (k_div_source), warm-started everywhere with the previous total
        stiffness (Bender & Koschier 2017; scaled by (dt_prev/dt)^2 when the substep changes), iterated (at least twice)
        to the mean (and, for density, max) residual tolerance."""
        dev, n, W = self.device, self.n, self.W
        tol_mean = P.SPH_DENSITY_TOL_MEAN if density else P.SPH_DIVERGENCE_TOL
        tol_max = P.SPH_DENSITY_TOL_MAX if density else 1e9
        tmp = self._tmp
        acc = self.kacc[density]
        apply_in = lambda k: [self.grid.id, self.x, vel, self.lid, self.rho0, k, W, self.V, self.h, self.knorm, dt,
                              self.squeezed, self.occ]
        src = [self.grid.id, self.x, vel, W, self.s, self.alpha, self.nbr, self.V, self.h, self.knorm, dt, int(density),
               P.SPH_JACOBI_OMEGA]
        warm = P.SPH_WARM_START_DENSITY if density else P.SPH_WARM_START_DIVERGENCE
        if warm and self.dt_prev:
            wp.launch(k_scale, n, inputs=[acc, warm*(self.dt_prev/dt)**2], device=dev)
            wp.launch(k_apply_pressure, n, inputs=apply_in(acc), outputs=[tmp, self.bforce, self.btorque], device=dev)
            wp.copy(vel, tmp)
        else:
            acc.zero_()
        it, mean, mx = 0, 0., 0.
        while it < P.SPH_MAX_ITERATIONS:
            it += 1
            check = it >= 2 and it % max(1, int(P.SPH_CHECK_EVERY)) == 0
            if check:
                self.err.zero_()
            wp.launch(k_div_source, n, inputs=src + [acc, self.occ, self.squeezed], outputs=[self.kappa, self.err],
                      device=dev)
            if check:
                e = self.err.numpy()
                mean, mx = float(e[0])/n, float(e[1])
                if not math.isfinite(mean):
                    raise FloatingPointError("Non-finite DFSPH residual")
                if mean <= tol_mean and mx <= tol_max:
                    break
            wp.launch(k_apply_pressure, n, inputs=apply_in(self.kappa), outputs=[tmp, self.bforce, self.btorque], device=dev)
            wp.copy(vel, tmp)
        self.last_err["den" if density else "div"] = (mean, mx, it)
        return it

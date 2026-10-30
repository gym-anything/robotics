"""Mixture continuum: an explicit material point method for powders, pastes, batters and sauces (Warp, GPU).

Method:
  * MLS-MPM with APIC transfers on a uniform grid of quadratic B-splines (Hu et al. 2018; Jiang et al. 2015). The grid
    is rebuilt every substep and follows the vessel, so nothing is stored on it between substeps.
  * Pressure from the volume fraction gathered at each point, with the part of every grid cell that lies inside a solid
    counted as full: phi_p = sum_i w_ip (sum_q w_iq V0_q / dx^3 + f_i), p = K_b (phi_p - 1) when phi_p > 1, else 0.
    A pressure built on each point's own tracked volume ratio drifts by 7-15 % of the liquid height in 12 s, and this
    form does not.
  * Deviatoric stress: elasto-viscoplastic, one scalar return mapping per point. An elastic trial stress (shear modulus
    G, Jaumann rate) is relaxed onto the yield surface tau = tau_y + mu p + K gdot^n, which is Herschel-Bulkley for
    pastes (mu = 0) and Drucker-Prager with cohesion for powders (K ~ 0). A material inside its yield surface is
    elastic and at rest; a stiff paste costs no smaller a time step than a thin one.
  * Solids: every tool and vessel is a signed-distance collider (body of revolution from a 2-D table, box, capsule,
    plane) moving with its rigid body. Grid nodes take the solid's velocity with a weight that rises linearly from 0 to
    1 across one cell centred on the surface: no penetration always; no slip where the material at the node is wet;
    Coulomb wall friction where it is dry powder. The momentum exchanged is the force and torque on the solid. A
    wet material's grip on a wall is its own viscosity's (k_wall: wall_c); a point found inside a solid is set back
    on its surface once a tick (k_keep_out), and a node's slide along one face is stopped at the next (the look-ahead
    at corners in k_wall): without these two a liquid goes into the walls at a square corner.
  * Thin wires (whisk loops, beater strips) are below the grid: marker segments exchange momentum with the grid by the
    cylinder drag laws (tools.py, k_markers below), with wet material however thin its layer round the wire. Dry
    powder is not dragged but struck: a dry point inside a wire's own cylinder is set on its surface and leaves at the
    wire's speed (k_wire_sweep); the drag, spread over the three cells of the grid's kernel, moves it a tenth of a
    millimetre a pass. What a
    wire carries out of the mixture stays on it as a film (k_film): points ride the wire, up to the thickness a
    surface drawn out of a viscous liquid keeps, and leave it as drops at the rate the film drains under its own
    weight, or under the centrifugal load of a turning beater: a lifted whisk drips, beaters lifted out turning
    throw batter. Drops in flight stop at the solids they meet inside the grid region (k_ballistic).
  * Composition: every point carries the mass of each species (materials.py). Species and heat move between neighbouring
    points by a conservative pool exchange on the grid at a per-point rate (k_exchange_*): each point puts a fraction of
    itself into the pools of the nodes around it and takes back the same mass with the pools' composition.
  * The grid region is the world-aligned box around the vessel's whole inside at the vessel's present attitude
    (set_box; the owner gives it every tick), so a tipped vessel is inside it at any tilt. Points that leave it
    (thrown out, poured or spilt) continue as ballistic points and stick where they land; where each point left
    the vessel's envelope is kept (xout, tout: over the rim, or not).

Units SI. z up. One call of Continuum.step advances one rigid tick (1 ms) in `substeps` substeps.
"""

import math
import numpy as np
import warp as wp

from . import materials as M

wp.config.quiet = True

NS = M.NS
KIND_REVOLVED, KIND_BOX, KIND_CAPSULE, KIND_PLANE = 0, 1, 2, 3
FLAG_ACTIVE, FLAG_FLYING, FLAG_COUNTER, FLAG_FLOOR = 0, 1, 2, 3
FLAG_FILM = 5                         # carried on a wire or a beater strip as a film (k_film); 4: an unused slot
FILM_COOL_S = 0.3                     # a drop shed from a wire flies free this long before a wire may take it again
PAD = 3                               # grid cells kept free at every face of the grid region


@wp.struct
class Colliders:
    n: int
    kind: wp.array(dtype=int)
    pos: wp.array(dtype=wp.vec3)       # world position of the collider frame
    rot: wp.array(dtype=wp.quat)
    vel: wp.array(dtype=wp.vec3)       # world velocity of the collider frame's origin
    omega: wp.array(dtype=wp.vec3)
    size: wp.array(dtype=wp.vec3)      # box: half extents; capsule: (radius, half length, 0)
    bound: wp.array(dtype=float)       # bounding radius about pos
    mu_dry: wp.array(dtype=float)      # wall friction coefficient for dry powder
    stick: wp.array(dtype=float)       # 1: wet material does not slip (walls, tools); 0: frictionless guide
    sdf_off: wp.array(dtype=int)
    sdf_nr: wp.array(dtype=int)
    sdf_nz: wp.array(dtype=int)
    sdf_h: wp.array(dtype=float)
    sdf_z0: wp.array(dtype=float)
    sdf: wp.array(dtype=float)
    share: wp.array(dtype=float)      # kernel-smoothed solid indicator, same tables as sdf (revolved bodies)


@wp.func
def sdf_lookup(C: Colliders, c: int, r: float, z: float):
    h = C.sdf_h[c]
    nr = C.sdf_nr[c]
    nz = C.sdf_nz[c]
    fr = r/h
    fz = (z - C.sdf_z0[c])/h
    cr = wp.clamp(fr, 0.0, float(nr) - 1.001)
    cz = wp.clamp(fz, 0.0, float(nz) - 1.001)
    i = int(cr)
    j = int(cz)
    a = cr - float(i)
    b = cz - float(j)
    o = C.sdf_off[c]
    v00 = C.sdf[o + i*nz + j]
    v01 = C.sdf[o + i*nz + j + 1]
    v10 = C.sdf[o + (i + 1)*nz + j]
    v11 = C.sdf[o + (i + 1)*nz + j + 1]
    inside = (1.0 - a)*((1.0 - b)*v00 + b*v01) + a*((1.0 - b)*v10 + b*v11)
    out = wp.sqrt((fr - cr)*(fr - cr) + (fz - cz)*(fz - cz))*h
    return inside + out


@wp.func
def collider_eval(C: Colliders, c: int, x: wp.vec3):
    """Signed distance (negative inside the solid), outward unit normal in the world and the solid's velocity at x."""
    p = C.pos[c]
    q = C.rot[c]
    xl = wp.quat_rotate_inv(q, x - p)
    kind = C.kind[c]
    phi = float(1.0e3)
    nl = wp.vec3(0.0, 0.0, 1.0)
    if kind == 0:
        r = wp.sqrt(xl[0]*xl[0] + xl[1]*xl[1])
        phi = sdf_lookup(C, c, r, xl[2])
        e = 0.5*C.sdf_h[c]
        r0 = wp.max(r - e, 0.0)
        dr = (sdf_lookup(C, c, r + e, xl[2]) - sdf_lookup(C, c, r0, xl[2]))/(r + e - r0)
        dz = (sdf_lookup(C, c, r, xl[2] + e) - sdf_lookup(C, c, r, xl[2] - e))/(2.0*e)
        if r > 1.0e-9:
            nl = wp.vec3(dr*xl[0]/r, dr*xl[1]/r, dz)
        else:
            nl = wp.vec3(0.0, 0.0, dz)
    elif kind == 1:
        hs = C.size[c]
        d = wp.vec3(wp.abs(xl[0]) - hs[0], wp.abs(xl[1]) - hs[1], wp.abs(xl[2]) - hs[2])
        dm = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
        outside = wp.length(dm)
        inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
        phi = outside + inside
        if outside > 0.0:
            nl = wp.vec3(dm[0]*wp.sign(xl[0]), dm[1]*wp.sign(xl[1]), dm[2]*wp.sign(xl[2]))
        elif d[0] > d[1] and d[0] > d[2]:
            nl = wp.vec3(wp.sign(xl[0]), 0.0, 0.0)
        elif d[1] > d[2]:
            nl = wp.vec3(0.0, wp.sign(xl[1]), 0.0)
        else:
            nl = wp.vec3(0.0, 0.0, wp.sign(xl[2]))
    elif kind == 2:
        hl = C.size[c][1]
        zc = wp.clamp(xl[2], -hl, hl)
        d = xl - wp.vec3(0.0, 0.0, zc)
        dist = wp.length(d)
        phi = dist - C.size[c][0]
        if dist > 1.0e-9:
            nl = d/dist
    else:
        phi = xl[2]
    ln = wp.length(nl)
    if ln > 1.0e-9:
        nl = nl/ln
    else:
        nl = wp.vec3(0.0, 0.0, 1.0)
    return phi, wp.quat_rotate(q, nl), C.vel[c] + wp.cross(C.omega[c], x - p)


@wp.func
def bspline(fx: wp.vec3, i: int, j: int, k: int):
    wx = 0.5*(1.5 - fx[0])*(1.5 - fx[0])
    if i == 1:
        wx = 0.75 - (fx[0] - 1.0)*(fx[0] - 1.0)
    if i == 2:
        wx = 0.5*(fx[0] - 0.5)*(fx[0] - 0.5)
    wy = 0.5*(1.5 - fx[1])*(1.5 - fx[1])
    if j == 1:
        wy = 0.75 - (fx[1] - 1.0)*(fx[1] - 1.0)
    if j == 2:
        wy = 0.5*(fx[1] - 0.5)*(fx[1] - 0.5)
    wz = 0.5*(1.5 - fx[2])*(1.5 - fx[2])
    if k == 1:
        wz = 0.75 - (fx[2] - 1.0)*(fx[2] - 1.0)
    if k == 2:
        wz = 0.5*(fx[2] - 0.5)*(fx[2] - 0.5)
    return wx*wy*wz


@wp.func
def spline_cdf(u: float):
    """Integral of the quadratic B-spline from -1.5 to u: the share of a node's weight that lies inside a planar
    solid whose surface is u cells beyond the node (u > 0: the node is inside)."""
    if u <= -1.5:
        return 0.0
    if u < -0.5:
        return (u + 1.5)*(u + 1.5)*(u + 1.5)/6.0
    if u < 0.5:
        return 0.5 + 0.75*u - u*u*u/3.0
    if u < 1.5:
        return 1.0 - (1.5 - u)*(1.5 - u)*(1.5 - u)/6.0
    return 1.0


@wp.func
def table_lookup(tab: wp.array(dtype=float), C: Colliders, c: int, r: float, z: float):
    h = C.sdf_h[c]
    nr = C.sdf_nr[c]
    nz = C.sdf_nz[c]
    cr = wp.clamp(r/h, 0.0, float(nr) - 1.001)
    cz = wp.clamp((z - C.sdf_z0[c])/h, 0.0, float(nz) - 1.001)
    i = int(cr)
    j = int(cz)
    a = cr - float(i)
    b = cz - float(j)
    o = C.sdf_off[c]
    return (1.0 - a)*((1.0 - b)*tab[o + i*nz + j] + b*tab[o + i*nz + j + 1]) \
        + a*((1.0 - b)*tab[o + (i + 1)*nz + j] + b*tab[o + (i + 1)*nz + j + 1])


@wp.func
def solid_share(C: Colliders, c: int, x: wp.vec3, phi: float, dx: float):
    """Share of a node's B-spline weight that lies inside solid c: the solid's indicator smoothed by the grid kernel.
    Exact for a plane and for a box (the kernel is a product, so is a box's indicator); a table computed once for a
    body of revolution (corners, rims and curvature included); the plane formula on the signed distance for a
    capsule. With the plane formula everywhere, the share would be too small in every inside corner and too large at
    every edge, and the material's weight there would drive a steady circulation at rest."""
    kind = C.kind[c]
    if kind == 1:
        xl = wp.quat_rotate_inv(C.rot[c], x - C.pos[c])
        hs = C.size[c]
        sx = spline_cdf((hs[0] - xl[0])/dx) - spline_cdf((-hs[0] - xl[0])/dx)
        sy = spline_cdf((hs[1] - xl[1])/dx) - spline_cdf((-hs[1] - xl[1])/dx)
        sz = spline_cdf((hs[2] - xl[2])/dx) - spline_cdf((-hs[2] - xl[2])/dx)
        return sx*sy*sz
    if kind == 0:
        xl = wp.quat_rotate_inv(C.rot[c], x - C.pos[c])
        if phi > 2.5*dx:
            return 0.0
        return table_lookup(C.share, C, c, wp.sqrt(xl[0]*xl[0] + xl[1]*xl[1]), xl[2])
    return spline_cdf(-phi/dx)


# ============================================================================================== mechanics
@wp.kernel
def k_clear(gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gvol: wp.array(dtype=float),
            gsat: wp.array(dtype=float), gnu: wp.array(dtype=float), gpv: wp.array(dtype=float),
            gmo: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    gm[i] = 0.0
    gmv[i] = wp.vec3(0.0, 0.0, 0.0)
    gvol[i] = 0.0
    gsat[i] = 0.0
    gnu[i] = 0.0
    gpv[i] = 0.0
    gmo[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), aff: wp.array(dtype=wp.mat33),
          dev: wp.array(dtype=wp.mat33), jac: wp.array(dtype=float), flags: wp.array(dtype=int),
          mass: wp.array(dtype=float), vol0: wp.array(dtype=float), kbulk: wp.array(dtype=float),
          wet: wp.array(dtype=float), visc: wp.array(dtype=float), nuw: wp.array(dtype=float), dt: float, dx: float,
          origin: wp.vec3, ny: int, nz: int, gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gvol: wp.array(dtype=float),
          gsat: wp.array(dtype=float), gnu: wp.array(dtype=float), gpv: wp.array(dtype=float),
          gmo: wp.array(dtype=wp.vec3), thrust: float):
    p = wp.tid()
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    pressure = wp.max(kbulk[p]*(1.0/jac[p] - 1.0), 0.0)
    ident = wp.identity(n=3, dtype=float)
    Lp = aff[p]
    Dp = 0.5*(Lp + wp.transpose(Lp))
    dDp = Dp - ((Dp[0, 0] + Dp[1, 1] + Dp[2, 2])/3.0)*ident
    stress = dev[p] + 2.0*visc[p]*dDp - pressure*ident
    m = mass[p]
    a = stress*(-dt*vol0[p]*4.0*inv_dx*inv_dx) + m*aff[p]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx)*dx
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                wp.atomic_add(gmv, idx, w*(m*v[p] + a*dpos))
                wp.atomic_add(gm, idx, w*m)
                wp.atomic_add(gvol, idx, w*vol0[p])
                wp.atomic_add(gsat, idx, w*m*wet[p])
                wp.atomic_add(gnu, idx, w*m*nuw[p])
                if thrust > 0.5:
                    wp.atomic_add(gpv, idx, w*vol0[p]*pressure)
                    wp.atomic_add(gmo, idx, w*vol0[p]*dpos)


@wp.kernel
def k_grid(gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gvol: wp.array(dtype=float),
           gv: wp.array(dtype=wp.vec3), gphi: wp.array(dtype=float), gkind: wp.array(dtype=int),
           gbest: wp.array(dtype=int), C: Colliders, dt: float, dx: float, origin: wp.vec3, ny: int, nz: int,
           gravity: float, wall_n: float, ghost: float, gpv: wp.array(dtype=float), gmo: wp.array(dtype=wp.vec3),
           thrust: float):
    """Free update of every node that carries mass, and its relation to the nearest solid: gkind 0 free, 1 inside the
    band where approach is blocked, 2 inside the solid. The wall rule itself is k_wall.

    thrust > 0.5 is a TRIAL and is OFF in the solver. It targets a small force left on the nodes next to a wall,
    which makes a thin liquid creep at rest.
    A node whose neighbourhood is partly solid gathers pressure from one side only: the gathered force has the part
    p (4/dx^2) M, with p the gathered mean pressure and M the first moment of the gathered volume about the node
    (zero where the neighbourhood is full). That is the material's thrust on the wall; a wall answers a pressure
    along its own normal, and M is along that normal only for a flat wall evenly filled. The trial took the component
    of that part along the wall out of the node's update. At rest it did what was hoped at the walls: the thin
    liquid's creep in 10 s fell from 0.84 to 0.09 mm next to the cup's wall, from 1.18 to 0.05 mm on its floor and
    from 5.7 to 0.02 mm in the corner between them. But the same term is how the scheme evens out the points along
    a wall and passes momentum there: the liquid's surface sank along the wall (3.2 mm in 10 s where it had been
    0.33), the rod's and the cup's torques no longer balanced in the stirred cup (by 57 % of the rod's at 3 Pa s),
    and applied only where no free surface is near it made the whole worse than without it (0.37 against 0.27 mm at
    6 s). So the force is found and it is not an error that can be taken out by itself."""
    idx = wp.tid()
    m = gm[idx]
    gkind[idx] = 0
    gbest[idx] = -1
    if m <= 1.0e-14:
        gv[idx] = wp.vec3(0.0, 0.0, 0.0)
        gphi[idx] = 0.0
        return
    vel = gmv[idx]/m
    vel = vel + wp.vec3(0.0, 0.0, -gravity*dt)
    i = idx/(ny*nz)
    j = (idx/nz) % ny
    k = idx % nz
    pos = origin + wp.vec3(float(i), float(j), float(k))*dx
    phi_min = float(1.0e3)
    best = int(-1)
    nbest = wp.vec3(0.0, 0.0, 1.0)
    free = float(1.0)                    # share of the node's weight in no solid: product over the solids near it
    for c in range(C.n):
        if C.kind[c] != 3 and wp.length(pos - C.pos[c]) > C.bound[c] + 2.0*dx:
            continue
        ph, nrm, vs = collider_eval(C, c, pos)
        if ph < 2.0*dx:
            free = free*(1.0 - solid_share(C, c, pos, ph, dx))
        if ph < phi_min:
            phi_min = ph
            best = c
            nbest = nrm
    fs = float(0.0)
    if best >= 0:
        fs = ghost*(1.0 - free)
        if thrust > 0.5 and phi_min < 1.5*dx and gvol[idx] > 1.0e-18:
            # only where material and solid fill the node's neighbourhood between them: next to a free surface part
            # of M is the open side's, and the thrust toward it is what holds the surface up (taking that out as
            # well let the liquid sink along the wall at its surface: 3 mm in 10 s at rest, first trial of this rule)
            full = float(1.0)
            if thrust > 1.5:                    # the trial's second form: away from free surfaces only
                full = wp.clamp((gvol[idx]/(dx*dx*dx) + fs - 0.9)/0.1, 0.0, 1.0)
            mo = gmo[idx]
            vel = vel - (full*dt*4.0/(dx*dx)*(gpv[idx]/gvol[idx])/m)*(mo - wp.dot(mo, nbest)*nbest)
        if phi_min <= 0.0:
            gkind[idx] = 2
            gbest[idx] = best
        elif phi_min <= wall_n*dx:
            gkind[idx] = 1
            gbest[idx] = best
    gphi[idx] = gvol[idx]/(dx*dx*dx) + fs
    gv[idx] = vel


@wp.kernel
def k_wall(gm: wp.array(dtype=float), gsat: wp.array(dtype=float), gphi: wp.array(dtype=float),
           gv: wp.array(dtype=wp.vec3), gkind: wp.array(dtype=int), gbest: wp.array(dtype=int), C: Colliders, dt: float,
           dx: float,
           origin: wp.vec3, nx: int, ny: int, nz: int, wall_mode: int, hold: float, mirror_lo: float, mirror_hi: float, hold_t: float,
           gnu: wp.array(dtype=float), wall_c: float, wall_n: float, corner: float,
           col_f: wp.array(dtype=wp.vec3),
           col_t: wp.array(dtype=wp.vec3)):
    """The wall rule at nodes in or next to a solid.

    Normal: the node moves with the solid along the normal while the material there is pressed against it (volume
    fraction with the solid's share above one), and only approaching velocity is removed where it is not, so material
    can leave a wall but is not pumped off it. (Removing only the approaching part everywhere rectified the resting
    material's velocity noise into a steady drift along the walls: 12 mm at the 99th percentile in 30 s of rest.)
    Tangential, wet material on a wall it wets: no slip AT the surface. A node inside the solid takes the velocity
    that makes the field odd about the surface: v = v_wall - (v_fluid(mirror point) - v_wall), the mirror point being
    the node reflected through the surface, v_fluid interpolated from the free nodes there. The grid field then has no
    kink at the wall and the transfers do not smear one. (Holding these nodes at the wall velocity instead puts the
    effective wall up to half a cell off and adds a drag that reads as 0.3-1.6 Pa s.)
    Tangential, dry powder: Coulomb friction with the wall's coefficient on the normal impulse."""
    idx = wp.tid()
    kind = gkind[idx]
    if kind == 0:
        return
    c = gbest[idx]
    m = gm[idx]
    i = idx/(ny*nz)
    j = (idx/nz) % ny
    k = idx % nz
    pos = origin + wp.vec3(float(i), float(j), float(k))*dx
    phi, nb, vb = collider_eval(C, c, pos)
    vel = gv[idx]
    vr = vel - vb
    vn = wp.dot(vr, nb)
    vt = vr - vn*nb
    dv = wp.vec3(0.0, 0.0, 0.0)
    pressed = gphi[idx] > hold
    if vn < 0.0 or pressed:
        dv = dv - vn*nb
    # How much of the tangential velocity a wall takes from wet material in one substep: what the material's own
    # viscosity carries across the half cell between the node and the surface, 1 - exp(-wall_c nu dt / dx^2), nu the
    # apparent kinematic viscosity at the node (a material inside its yield surface has an unbounded one: it sticks).
    # A thick paste is held as before; a thin liquid slides along a wall and is slowed by it only as fast as its
    # viscosity allows. (Holding every wet node whatever its viscosity gave any liquid thinner than about 1 Pa s the
    # wall drag of a 0.5 Pa s one: Couette torque 1.6 times theory at 1 Pa s, 3.6 times at 0.2; a pan tipped to 59
    # degrees poured at a sixth of a weir's rate. wall_c < 0 restores that rule.)
    lam = float(1.0)
    if wall_c >= 0.0 and m > 1.0e-14:
        lam = 1.0 - wp.exp(-wall_c*(gnu[idx]/m)*dt/(dx*dx))
    if kind == 2:
        if gsat[idx] > 0.5*m:
            if C.stick[c] > 0.5:
                target = wp.vec3(0.0, 0.0, 0.0)            # tangential velocity relative to the solid at this node
                planar = False
                xm = pos - 2.0*phi*nb
                if wall_mode == 0 and phi > -2.0*dx:
                    # the reflection is the surface's only where the surface is locally a plane: at the mirror point
                    # the same solid must be as far away as the node is deep. Near an inside corner or an edge it is
                    # not, and reflecting there feeds back on itself: a steady 0.4-8 mm/s circulation out of the cup's
                    # bottom corner with the material at rest. Such nodes are held instead.
                    phm, nm, vsm = collider_eval(C, c, xm)
                    planar = phm > -0.8*phi and wp.dot(nm, nb) > 0.9
                if planar:
                    g = (xm - origin)/dx
                    i0 = int(wp.floor(g[0]))
                    j0 = int(wp.floor(g[1]))
                    k0 = int(wp.floor(g[2]))
                    fr = wp.vec3(g[0] - float(i0), g[1] - float(j0), g[2] - float(k0))
                    wsum = float(0.0)
                    vf = wp.vec3(0.0, 0.0, 0.0)
                    for a in range(2):
                        for b in range(2):
                            for e in range(2):
                                ii = i0 + a
                                jj = j0 + b
                                kk = k0 + e
                                if ii >= 0 and jj >= 0 and kk >= 0 and ii < nx and jj < ny and kk < nz:
                                    nid = (ii*ny + jj)*nz + kk
                                    if gkind[nid] < 2 and gm[nid] > 1.0e-14:
                                        w = (1.0 - fr[0] + float(a)*(2.0*fr[0] - 1.0))*(1.0 - fr[1] + float(b)*(
                                            2.0*fr[1] - 1.0))*(1.0 - fr[2] + float(e)*(2.0*fr[2] - 1.0))
                                        vf = vf + w*gv[nid]
                                        wsum = wsum + w
                    if wsum > 0.25:
                        vf = vf/wsum
                        vm = C.vel[c] + wp.cross(C.omega[c], xm - C.pos[c])     # the solid's own field at the mirror
                        rel = vf - vm
                        rt = rel - wp.dot(rel, nb)*nb
                        # reflect only a flow that is really shearing past the wall: below mirror_lo the node is held
                        # (a reflected velocity noise kept material creeping along resting walls), full from mirror_hi
                        sp = wp.length(rt)
                        tm = wp.clamp((sp - mirror_lo)/(mirror_hi - mirror_lo), 0.0, 1.0)
                        target = -(tm*tm*(3.0 - 2.0*tm))*rt
                dv = dv + lam*(target - vt)
        elif vn < 0.0 or pressed:
            lt = wp.length(vt)
            if lt > 1.0e-9:
                dv = dv - vt*wp.min(1.0, C.mu_dry[c]*wp.abs(vn)/lt)
    elif phi < hold_t*dx and gsat[idx] > 0.5*m and C.stick[c] > 0.5:
        # slow flow: no slip from hold_t cells outside the surface inward (held nodes). Fast flow: these nodes are
        # free and the reflection inside the solid puts the no-slip surface on the wall itself.
        keep = float(1.0)
        if wall_mode == 0:
            tb = wp.clamp((wp.length(vt) - mirror_lo)/(mirror_hi - mirror_lo), 0.0, 1.0)
            keep = 1.0 - tb*tb*(3.0 - 2.0*tb)
        dv = dv - lam*keep*vt
    # A corner: the rules above know one face of the solid, the nearest to the node. Where the node may slide along
    # that face (a liquid on the viscous rule), half a cell further on the same solid may have ANOTHER face in the
    # way: the floor's nodes next to a wall, the wall's next to the floor. The velocity that is left may not go into
    # the solid there either. Only where the surface turns by more than 45 degrees within that half cell: on a curved
    # wall the node's own normal rule turns the flow, and this look-ahead, half a cell where the flow moves a
    # hundredth of one in a substep, would turn it a hundred times too hard (tried: the stirred cup's level rose 3 %
    # and its torque fell by a third). (Without the rule a liquid runs along the cup's floor into the wall at their
    # square corner: E4.)
    # What is left, and what was tried against it: a thin liquid at rest still creeps, 0.6 mm in 10 s at
    # the median in the cup, drawn down the outer wall at 0.08 mm/s and out of the square corner along its diagonal at
    # 0.6 mm/s. The cause is a small force the solver leaves on the nodes next to a wall, which a hold of every wet
    # wall node hides (0.3 mm in 10 s) and a thin liquid's true grip on a wall cannot hide. Holding the wall nodes
    # stops the creep (0.2 mm) and puts the cup's torque at 0.2 Pa s back from 1.35 to 2.5 times theory;
    # holding them only where the material slides slower than 1 to 3 mm/s does little (0.9 mm), slower than 10 to
    # 30 mm/s is the hold again (torque 2.3 times theory); holding only the nodes in the corner itself does nothing
    # (1.1 mm). None of these is in the solver.
    if corner > 0.5:
        vr2 = vr + dv
        vt2 = vr2 - wp.dot(vr2, nb)*nb
        lt2 = wp.length(vt2)
        if lt2 > 1.0e-7:
            xa = pos + (0.5*dx/lt2)*vt2
            pha, na, va = collider_eval(C, c, xa)
            if pha < wall_n*dx and wp.dot(na, nb) < 0.7:
                vna = wp.dot(vel + dv - va, na)
                if vna < 0.0:
                    dv = dv - vna*na
    vel = vel + dv
    f = -m*dv/dt
    wp.atomic_add(col_f, c, f)
    wp.atomic_add(col_t, c, wp.cross(pos - C.pos[c], f))
    # Every other solid this node is in or next to stops the material as well (its normal rule; the tangential rule is
    # the nearest solid's). Where two solids meet (a box's wall on its floor, a tool against a vessel, a vessel on
    # the counter) a node answering to the nearer one only lets a thin liquid, which slides along a wall, run out
    # through such seams: without this loop a box of milk at rest lost 5 mm of its level in 20 s.
    for c2 in range(C.n):
        if c2 == c:
            continue
        if C.kind[c2] != 3 and wp.length(pos - C.pos[c2]) > C.bound[c2] + 2.0*dx:
            continue
        ph2, n2, v2 = collider_eval(C, c2, pos)
        if ph2 > wall_n*dx:
            continue
        vn2 = wp.dot(vel - v2, n2)
        if vn2 < 0.0 or pressed:
            d2 = -vn2*n2
            vel = vel + d2
            f2 = -m*d2/dt
            wp.atomic_add(col_f, c2, f2)
            wp.atomic_add(col_t, c2, wp.cross(pos - C.pos[c2], f2))
    gv[idx] = vel


@wp.kernel
def k_g2p(x: wp.array(dtype=wp.vec3), xr: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), aff: wp.array(dtype=wp.mat33),
          dev: wp.array(dtype=wp.mat33), phi: wp.array(dtype=float), jac: wp.array(dtype=float),
          flags: wp.array(dtype=int),
          mass: wp.array(dtype=float), vol0: wp.array(dtype=float), kbulk: wp.array(dtype=float),
          shear: wp.array(dtype=float), cons: wp.array(dtype=float), nidx: wp.array(dtype=float),
          tauy: wp.array(dtype=float), mudp: wp.array(dtype=float), work: wp.array(dtype=float),
          gdot: wp.array(dtype=float), visc: wp.array(dtype=float), lam_n: wp.array(dtype=wp.vec3),
          lam_l: wp.array(dtype=float), lam_on: wp.array(dtype=float), gv: wp.array(dtype=wp.vec3),
          gphi: wp.array(dtype=float), gvol: wp.array(dtype=float), dt: float, dx: float, origin: wp.vec3, nx: int,
          ny: int, nz: int, vmax: float, eta_cap: float, jrelax: float, wall_j: float, wall_g0: float, wall_g1: float,
          free_on: float, free_dg: float, free_tab: wp.array(dtype=float), sgrad: wp.array(dtype=float),
          nuw: wp.array(dtype=float), counts: wp.array(dtype=int)):
    p = wp.tid()
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    vel = wp.vec3(0.0, 0.0, 0.0)
    L = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ph = float(0.0)
    pv = float(0.0)                      # the points' own share of ph (the rest is the solids')
    gsum = wp.vec3(0.0, 0.0, 0.0)        # first moment of the gathered fraction about the point: its gradient
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx)*dx
                nid = ((bx + i)*ny + (by + j))*nz + (bz + k)
                node = gv[nid]
                vel = vel + w*node
                L = L + (4.0*inv_dx*inv_dx*w)*wp.outer(node, dpos)
                ph = ph + w*gphi[nid]
                pv = pv + w*gvol[nid]
                gsum = gsum + (w*gphi[nid])*dpos
    pv = pv*inv_dx*inv_dx*inv_dx
    sp = wp.length(vel)
    if sp > vmax:
        vel = vel*(vmax/sp)
        wp.atomic_add(counts, 0, 1)
    # Compensated position update. Positions are single precision in world coordinates: at counter height one unit
    # in the last place is 6e-8 m, and a substep's displacement below half of that (anything slower than 0.33 mm/s
    # vertically, 0.04-0.16 mm/s horizontally) was rounded away: slow creep was frozen, with its velocity still on
    # the point (the rest runs showed 0.3 mm/s of "speed" and 0.000 mm of displacement). The part of each step that
    # the addition loses is carried in xr and added to the next one.
    step = dt*vel + xr[p]
    xn = x[p] + step
    xr[p] = step - (xn - x[p])
    v[p] = vel
    x[p] = xn
    gx = (xn - origin)*inv_dx
    if gx[0] < float(PAD) or gx[1] < float(PAD) or gx[2] < float(PAD) or gx[0] > float(nx - PAD - 1) \
            or gx[1] > float(ny - PAD - 1) or gx[2] > float(nz - PAD - 1):
        flags[p] = 1
        jac[p] = 1.0
        aff[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dev[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        phi[p] = 0.0
        return
    aff[p] = L
    phi[p] = ph
    # Volume ratio: carried with the flow (smooth), and drawn toward the ratio the gathered volume fraction implies
    # over the time jrelax. Carried alone it drifts (7-15 % of the level in 12 s); taken from the
    # gathered fraction alone its 1 % scatter times the bulk modulus is a 360 Pa pressure noise that keeps a liquid
    # at rest circulating. At rest next to a solid it is carried only (see Continuum.wall_j).
    jn = jac[p]*(1.0 + dt*(L[0, 0] + L[1, 1] + L[2, 2]))
    jr = jrelax
    flowing = wp.clamp((gdot[p] - wall_g0)/(wall_g1 - wall_g0), 0.0, 1.0)          # the shear rate of the last substep
    if wall_j > 0.0:
        at_wall = wp.clamp((ph - pv)/wall_j, 0.0, 1.0)
        jr = jrelax*(1.0 - at_wall*(1.0 - flowing))
    # At a free surface the gathered fraction is low only because part of the kernel is in the air, and a volume
    # ratio drawn toward it lets the top layer of a liquid pack until the cut kernel reads one: 1.6 times its rest
    # density, a heavy skin on a lighter layer, which turns over slowly and never comes to rest.
    # The gradient of the gathered fraction says how much of the kernel is in the air: free_tab holds the fraction a
    # resting lattice at rest density shows for a given gradient (measured on this solver, FREE_SURFACE below), and
    # the gathered fraction is divided by it. Only for material at rest: in a stirred liquid the points' disorder
    # gives the interior gradients as large as a surface's (0.27 at the 99th percentile against 0.37 in a surface's
    # second layer), and the correction applied there dilated the liquid by 2 %. And not next to a solid, where the
    # solid's share is in ph already.
    gmag = 4.0*inv_dx*wp.length(gsum)
    sgrad[p] = gmag
    phc = ph
    if free_on > 0.0:
        q = gmag/free_dg
        k0 = wp.min(int(q), free_tab.shape[0] - 2)
        fq = wp.min(q - float(k0), 1.0)
        hnom = free_tab[k0]*(1.0 - fq) + free_tab[k0 + 1]*fq
        away = 1.0 - flowing
        if wall_j > 0.0:
            away = away*(1.0 - wp.clamp((ph - pv)/wall_j, 0.0, 1.0))
        phc = ph/(1.0 - (1.0 - hnom)*away)
    jn = jn + (1.0/wp.max(phc, 0.5) - jn)*jr
    jac[p] = wp.clamp(jn, 0.5, 2.0)
    # ---- constitutive update
    # Viscous stress K gdot^n: the share eta_e <= eta_cap of the apparent viscosity is applied explicitly (visc[p],
    # added to the stress in P2G); the rest, with the yield stress and the pressure-dependent friction, is carried by
    # an elastic trial stress relaxed onto the surface tau = tau_y + mu p + (1 - r) K gdot_p^n. At steady shear the two
    # shares add up to the law exactly. A purely elastic-predictor form leaves grid-scale velocity noise undamped,
    # which reads as 0.5-1.3 Pa s of extra viscosity in the cup.
    D = 0.5*(L + wp.transpose(L))
    W = 0.5*(L - wp.transpose(L))
    tr = D[0, 0] + D[1, 1] + D[2, 2]
    ident = wp.identity(n=3, dtype=float)
    dD = D - (tr/3.0)*ident
    rate = wp.sqrt(2.0*wp.ddot(dD, dD))
    K = cons[p]
    n = nidx[p]
    eta_app = K*wp.pow(wp.max(rate, 1.0e-3), n - 1.0)
    r = wp.min(1.0, eta_cap*mass[p]/vol0[p]/wp.max(eta_app, 1.0e-12))
    visc[p] = r*eta_app
    Km = (1.0 - r)*K
    G = shear[p]
    s = dev[p]
    st = s + dt*(2.0*G*dD + W*s - s*W)
    tau_t = wp.sqrt(0.5*wp.ddot(st, st))
    pressure = wp.max(kbulk[p]*(1.0/jac[p] - 1.0), 0.0)
    ty = tauy[p] + mudp[p]*pressure
    gp = float(0.0)
    if tau_t > ty and tau_t > 1.0e-12:
        excess = tau_t - ty
        hi = excess/(G*dt)
        gp = excess/(G*dt + Km)
        if n < 0.999 and Km > 0.0:
            gp = wp.min(gp, hi)
            for it in range(8):
                gpn = wp.pow(wp.max(gp, 1.0e-12), n)
                f = excess - G*dt*gp - Km*gpn
                df = -G*dt - Km*n*gpn/wp.max(gp, 1.0e-12)
                gp = wp.clamp(gp - f/df, 0.0, hi)
        tau = wp.max(tau_t - G*dt*gp, 0.0)
        st = st*(tau/tau_t)
        work[p] = work[p] + tau*gp*dt*vol0[p]/mass[p]
    work[p] = work[p] + visc[p]*rate*rate*dt*vol0[p]/mass[p]
    dev[p] = st
    gdot[p] = rate
    # apparent kinematic viscosity for the wall rule: the whole law's stress over the shear rate (k_wall)
    nuw[p] = (eta_app + ty/wp.max(rate, 1.0e-3))*vol0[p]/mass[p]
    # ---- lamella inside the point: unit normal and log-thinning (materials.py, lamellar closure)
    if lam_on[p] > 0.5:
        nn = lam_n[p]
        stretch = wp.dot(nn, D*nn)
        lam_l[p] = wp.clamp(lam_l[p] - stretch*dt, 0.0, 12.0)
        nn = nn + dt*(-(wp.transpose(L)*nn) + stretch*nn)
        ln = wp.length(nn)
        if ln > 1.0e-9:
            lam_n[p] = nn/ln
    else:
        lam_l[p] = 0.0


@wp.kernel
def k_keep_out(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int),
               mass: wp.array(dtype=float), gkind: wp.array(dtype=int), C: Colliders, dx: float, dt: float,
               origin: wp.vec3, ny: int, nz: int, col_f: wp.array(dtype=wp.vec3), col_t: wp.array(dtype=wp.vec3),
               counts: wp.array(dtype=int)):
    """No point of the mixture inside a solid: a point found inside one is set back on its surface along the solid's
    normal, and the part of its velocity that goes into the solid is taken away (the solid feels that impulse).

    The grid's wall rule (k_wall) acts on nodes, and a point takes its velocity from the nodes round it, free ones
    among them: a point at a wall can drift into the solid by a fraction of a cell at every step it is pressed there.
    Holding every tangential velocity at a wall keeps such a point where it is and the drift within a tenth of a
    millimetre. With the viscous wall rule a liquid slides along a wall, and without this kernel the drift goes on:
    in a test cup 1.1 % of a thin liquid lay beyond the wall after 10 s at rest (to 2.2 mm),
    0.7 % under the floor (to 4.5 mm), and the level fell 2.6 %; about 1 % of a custard sat inside the saucepan's
    surface under the stirrer. A solid is impenetrable whatever the grid resolves, so the position is corrected where it is wrong.
    Once a tick (a point moves well under a millimetre in one)."""
    p = wp.tid()
    if flags[p] != 0:
        return
    g = (x[p] - origin)/dx
    nid = (int(g[0] + 0.5)*ny + int(g[1] + 0.5))*nz + int(g[2] + 0.5)
    if gkind[nid] == 0:
        return                               # its nearest node is more than the approach band from every solid
    pos = x[p]
    vel = v[p]
    moved = int(0)
    for c in range(C.n):
        if C.kind[c] != 3 and wp.length(pos - C.pos[c]) > C.bound[c] + dx:
            continue
        ph, nrm, vs = collider_eval(C, c, pos)
        if ph < 0.0:
            pos = pos - ph*nrm
            vn = wp.dot(vel - vs, nrm)
            if vn < 0.0:
                dvel = -vn*nrm
                vel = vel + dvel
                f = -mass[p]*dvel/dt
                wp.atomic_add(col_f, c, f)
                wp.atomic_add(col_t, c, wp.cross(pos - C.pos[c], f))
            moved = 1
    if moved == 1:
        x[p] = pos
        v[p] = vel
        wp.atomic_add(counts, 1, 1)


@wp.kernel
def k_ballistic(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int),
                phi: wp.array(dtype=float), dt: float, gravity: float, origin: wp.vec3, dx: float, nx: int, ny: int,
                nz: int, counter_z: float, c_lo: wp.vec2, c_hi: wp.vec2, floor_z: float, C: Colliders,
                counter_c: int, ftime: wp.array(dtype=float), time: float, cool: float):
    """Points in free flight (thrown out of the grid region, or shed from a wire as drops). Inside the grid region a
    drop becomes mixture again where the region is free, and at once where it meets a solid; a drop shed from a wire flies free for `cool` seconds first, or the wire's drag would
    take it straight back."""
    p = wp.tid()
    if flags[p] != 1:
        return
    vel = v[p] + wp.vec3(0.0, 0.0, -gravity*dt)
    xn = x[p] + dt*vel
    if xn[2] <= counter_z and x[p][2] >= counter_z - 0.02 and xn[0] > c_lo[0] and xn[0] < c_hi[0] \
            and xn[1] > c_lo[1] and xn[1] < c_hi[1]:
        xn = wp.vec3(xn[0], xn[1], counter_z)
        vel = wp.vec3(0.0, 0.0, 0.0)
        flags[p] = 2
    elif xn[2] <= floor_z:
        xn = wp.vec3(xn[0], xn[1], floor_z)
        vel = wp.vec3(0.0, 0.0, 0.0)
        flags[p] = 3
    else:
        g = (xn - origin)/dx
        m = float(PAD + 2)
        if g[0] > m and g[1] > m and g[2] > m and g[0] < float(nx) - m - 1.0 and g[1] < float(ny) - m - 1.0 \
                and g[2] < float(nz) - m - 1.0:
            hit = int(0)
            for c in range(C.n):
                if C.kind[c] == 3 or c == counter_c:
                    continue                                   # the counter top and the floor are the rules above
                if wp.length(xn - C.pos[c]) > C.bound[c] + dx:
                    continue
                ph, nrm, vs = collider_eval(C, c, xn)
                if ph < 0.5*dx:
                    hit = 1
                    vel = vs                                   # it lands on the solid and is mixture on it
            if hit == 1 or time - ftime[p] > cool:
                flags[p] = 0                  # in the grid region again
                phi[p] = 0.0
    x[p] = xn
    v[p] = vel


@wp.kernel
def k_advance_colliders(pos: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat), vel: wp.array(dtype=wp.vec3),
                        omega: wp.array(dtype=wp.vec3), dt: float):
    c = wp.tid()
    pos[c] = pos[c] + vel[c]*dt
    w = omega[c]
    q = rot[c]
    dq = wp.quat(w[0], w[1], w[2], 0.0)*q
    rot[c] = wp.normalize(wp.quat(q[0] + 0.5*dt*dq[0], q[1] + 0.5*dt*dq[1], q[2] + 0.5*dt*dq[2], q[3] + 0.5*dt*dq[3]))


# ============================================================================================== thin wires (tools.py)
@wp.kernel
def k_markers(mx: wp.array(dtype=wp.vec3), mu_: wp.array(dtype=wp.vec3), mt: wp.array(dtype=wp.vec3),
              mlen: wp.array(dtype=float), mdia: wp.array(dtype=float), mbody: wp.array(dtype=int),
              gm: wp.array(dtype=float), gv: wp.array(dtype=wp.vec3), gcons: wp.array(dtype=float),
              gnidx: wp.array(dtype=float), gstr: wp.array(dtype=float), grm: wp.array(dtype=float),
              gsat: wp.array(dtype=float), dt: float, dx: float,
              origin: wp.vec3, nx: int, ny: int, nz: int, tauc: float, bearing: float, gexpo: wp.array(dtype=float),
              mf: wp.array(dtype=wp.vec3), mrate: wp.array(dtype=float), mrho: wp.array(dtype=float),
              meta_: wp.array(dtype=float), gwire: wp.array(dtype=int), sweep_dry: float):
    """One wire segment: drag on a cylinder of diameter d and length l moving at the local relative velocity.

    Viscous part, wet material: the apparent viscosity at the wire's own shear rate U/d; Lamb's low-Reynolds result
    below Re = 1 and C_D = 1 + 10 Re^(-2/3) above. Dry powder: 0.5 rho U^2 d C_D with C_D = 2, which
    is the momentum of the powder a wire meets leaving at the wire's speed; with sweep_dry on that momentum is given to
    the points the wire meets themselves (k_wire_sweep) and this term is left out.
    Plastic part, every material: a cylinder pushed sideways through a material of shear strength s (yield stress, or
    cohesion plus friction times pressure for a powder) meets a force bearing * s * d per unit length whatever its
    speed, bearing = 10.5 (Randolph and Houlsby 1984: 9.14 for a smooth cylinder to 11.94 for a rough one). Without it
    a whisk goes through dry flour without moving it: the viscous and inertial laws give 0.04 N/m there, the plastic
    one 2.5 N/m.
    Half the coefficient along the wire. Applied to the nodes around the segment as an exponential relaxation toward
    the wire velocity (cannot overshoot); the momentum taken from the material is the force on the wire."""
    k = wp.tid()
    inv_dx = 1.0/dx
    xw = mx[k]
    mx[k] = xw + mu_[k]*dt               # the wire moves with its body through the tick
    xp = (xw - origin)*inv_dx
    mrate[k] = 0.0
    mrho[k] = 0.0
    if xp[0] < 2.0 or xp[1] < 2.0 or xp[2] < 2.0 or xp[0] > float(nx - 3) or xp[1] > float(ny - 3) \
            or xp[2] > float(nz - 3):
        return
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    vel = wp.vec3(0.0, 0.0, 0.0)
    meff = float(0.0)
    mlaw = float(0.0)
    K = float(0.0)
    n = float(0.0)
    S = float(0.0)
    sat = float(0.0)
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                w = bspline(fx, i, j, kk)
                nid = ((bx + i)*ny + (by + j))*nz + (bz + kk)
                m = gm[nid]
                vel = vel + w*m*gv[nid]
                meff = meff + w*m
                mlaw = mlaw + w*grm[nid]
                K = K + w*gcons[nid]
                n = n + w*gnidx[nid]
                S = S + w*gstr[nid]
                sat = sat + w*gsat[nid]
    if meff < 1.0e-9:
        return
    vel = vel/meff
    sat = sat/meff
    rho = meff/(dx*dx*dx)
    mrho[k] = rho
    # The law round the wire. Its sums were gathered when the tick began (k_p2g_rheology) and are divided by the mass
    # gathered with them, so that K, n and S are means of the points' own values. Divided by THIS substep's
    # mass instead, where a drop in flight has moved most of a cell since the tick began the two do not cover the
    # same nodes, and a wire next to it reads a flow index of several hundred and a viscosity of 1e33 Pa s, whose drag
    # comes out as infinity over infinity: a "solver fault" in one hand-mixer run in three at top speed in unmixed
    # ingredients, and in 10 of 200 random placements of a lone fast drop next to a wire in an empty dish (E8).
    # Material that has reached the wire within this tick has no law on the grid yet and is left for the next.
    if mlaw < 1.0e-9:
        return
    K = K/mlaw
    n = n/mlaw
    S = S/mlaw
    meta_[k] = wp.max(K*wp.pow(10.0, n - 1.0), 1.0e-4)        # the mixture's viscosity at 10 1/s round the wire (k_film)
    # Wet material is dragged however little of it is round the wire: what a wire carries up out of a batter stays
    # with it by the same law, slipping down it under its weight (it drips), and what turning beaters carry out is
    # held to their circle only by that drag, slips outward and is thrown. A wisp
    # of DRY powder is left alone: the plastic law below is a continuum's.
    if sat <= 0.5 and rho < 150.0:
        return
    rel = mu_[k] - vel
    t = mt[k]
    ut = wp.dot(rel, t)*t
    un = rel - ut
    U = wp.length(un)
    d = mdia[k]
    ln = mlen[k]
    beta_n = float(0.0)                  # force = beta * relative velocity (N per m/s), normal to the wire
    sweep = float(0.0)                   # share of a cell's volume the wire's boundary layers sweep, stress-weighted
    if U > 1.0e-6:
        if sat > 0.5:
            eta = wp.max(K*wp.pow(wp.max(U/d, 1.0e-3), n - 1.0), 1.0e-4)
            re = rho*U*d/eta
            fl = float(0.0)
            if re < 1.0:
                fl = 4.0*3.14159265*eta*U/(0.5 - 0.5772 - wp.log(re/8.0))
            else:
                fl = 0.5*rho*U*U*d*(1.0 + 10.0*wp.pow(re, -0.6667))
            beta_n = fl*ln/U
            mrate[k] = U/d
            tw = eta*U/d
            if tw > tauc:
                sweep = (1.0 - tauc/tw)*2.0*d*ln*U*dt/(dx*dx*dx)
        elif sweep_dry < 0.5:
            beta_n = 0.5*rho*U*d*2.0*ln
        beta_n = beta_n + bearing*S*d*ln/U
    beta_t = 0.5*beta_n
    fac_n = 1.0 - wp.exp(-beta_n*dt/meff)
    fac_t = 1.0 - wp.exp(-beta_t*dt/meff)
    dv = fac_n*un + fac_t*ut
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                w = bspline(fx, i, j, kk)
                nid = ((bx + i)*ny + (by + j))*nz + (bz + kk)
                if gm[nid] > 1.0e-14:
                    wp.atomic_add(gv, nid, w*dv)
                    wp.atomic_add(gexpo, nid, w*sweep)
                    wp.atomic_max(gwire, nid, k + 1)
    mf[k] = mf[k] - meff*dv/dt        # summed over the tick's substeps (the owner zeroes it)


@wp.kernel
def k_wire_near(mx: wp.array(dtype=wp.vec3), mt: wp.array(dtype=wp.vec3), mlen: wp.array(dtype=float), dx: float,
                origin: wp.vec3, nx: int, ny: int, nz: int, reach: int, gnear: wp.array(dtype=int)):
    """Every node within `reach` cells of a wire segment learns which segment lies nearest to it: the segment's index
    packed under its distance (in hundredths of a millimetre), so that the smallest packed number is the nearest
    segment. (The first form of the wire's rule stamped the highest index on the 27 nodes round a segment; with a
    reach of several cells two wires of a cage overlap and the nearer one must win.)"""
    k = wp.tid()
    xp = (mx[k] - origin)/dx
    if xp[0] < 1.0 or xp[1] < 1.0 or xp[2] < 1.0 or xp[0] > float(nx - 2) or xp[1] > float(ny - 2) or xp[2] > float(nz - 2):
        return
    bx = int(xp[0] + 0.5)
    by = int(xp[1] + 0.5)
    bz = int(xp[2] + 0.5)
    t = mt[k]
    hl = 0.5*mlen[k]
    for i in range(2*reach + 1):
        ix = bx + i - reach
        if ix < 0 or ix >= nx:
            continue
        for j in range(2*reach + 1):
            iy = by + j - reach
            if iy < 0 or iy >= ny:
                continue
            for kk in range(2*reach + 1):
                iz = bz + kk - reach
                if iz < 0 or iz >= nz:
                    continue
                rel = origin + wp.vec3(float(ix), float(iy), float(iz))*dx - mx[k]
                sa = wp.clamp(wp.dot(rel, t), -hl, hl)
                dist = wp.length(rel - sa*t)
                q = wp.min(int(dist*1.0e5), 2047)
                wp.atomic_min(gnear, (ix*ny + iy)*nz + iz, q*1048576 + k)


@wp.kernel
def k_wire_sweep(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int),
                 wet: wp.array(dtype=float), mass: wp.array(dtype=float), gnear: wp.array(dtype=int),
                 mx: wp.array(dtype=wp.vec3), mu_: wp.array(dtype=wp.vec3), mt: wp.array(dtype=wp.vec3),
                 mlen: wp.array(dtype=float), mdia: wp.array(dtype=float), markers: int, dx: float, dt: float,
                 origin: wp.vec3, ny: int, nz: int, carry_w: float, carry_lam: float, mf: wp.array(dtype=wp.vec3),
                 counts: wp.array(dtype=int), vol0: wp.array(dtype=float), cut: wp.array(dtype=float), slit_a: float,
                 tauy: wp.array(dtype=float), slit_tau: float, cutd: wp.array(dtype=float)):
    """What a wire does to the dry powder next to it, on the powder's own points (every substep). Three parts.

    Out of the wire's way: a point found inside a wire segment's cylinder is set on its surface, along the line from
    the wire's axis through the point, and takes the wire's velocity along that line if it was slower (the wire feels
    that momentum).
    Carried along: a point at distance r from the axis, outside the wire, goes the wire's way (across the wire's own
    axis) at no less than the share carry_w * exp(-(r - d/2) / (carry_lam * d)) of the wire's speed; what the grid's
    velocity already gives it counts toward that share.

    Why on the points and not on the grid. A powder with strength flows round a wire inside about two of the wire's
    diameters and nowhere else. The drag laws of k_markers act on the grid's nodes round a wire, three cells wide, and
    a 1.4 mm wire's limit load spread over 7.5 mm makes nothing yield: the powder stays where it is. With the wire
    RESOLVED by the same solver and the same flour (a solid capsule on cells of 0.2 to 0.4 mm: E20), an
    upright wire drawn sideways at 0.25 m/s leaves the powder in its own track
    4.4 mm farther along, the powder half a diameter to its side 1.8 mm, a diameter away 1.1 mm, two away 0.5 mm:
    7.5 times its own cross-section of powder is carried forward (11.5 mm2). The first part alone (out of the wire's
    way only) gives 0.07 times the cross-section on the tasks' 2.5 mm cells and 1.35 times on 1.25 mm cells: the whisk
    three times slower than a blade on the same strokes, and the result tripling when the cell is halved. The two numbers of the second part are read off the resolved wire's profile at 0.25 m/s (share 0.5,
    decay length 0.7 diameters: rms misfit 0.12 mm over five diameters; at 0.6 m/s the resolved wire carries a third
    more, 0.6 and 0.8, which is left out). They are not set on any task's outcome, and because they act on points the
    result does not depend on the cell.
    The cut: the wire leaves its own track partly empty. Behind the resolved wire the strip it went through, one
    diameter wide, holds 0.43 to 0.50 of the powder it held (upright, 0.25 m/s; 0.55 to 0.64 lying 10 mm deep). The
    points within one diameter of the axis therefore take up, between them, an open volume of slit_a (0.5) times
    the wire's diameter per unit of the area the wire sweeps (each a share by its own volume): `cut`, cubic metres
    per point, at most half the point's volume. Nothing else is done with it here: in dry powder in air a cut is
    air. What a liquid does with it is in k_slit_* (exchange).
    Left out of this rule: the powder rises by about a diameter next to a resolved wire (the grid makes some of
    that itself when the points crowd), and a wire LYING under powder carries the powder above it farther than an
    upright one carries the powder beside it (24 mm2 at 10 mm depth): the rule is the upright wire's, in every
    direction round the axis.
    Wet material is left to the viscous law (a batter flows round a wire over distances the grid sees, and the law's
    force is that flow's).
    (Tried on top of the first part and taken out: struck powder made loose for a few seconds, as aerated flour is;
    the whisk then blended more slowly: E15.)"""
    p = wp.tid()
    if flags[p] != 0:
        return
    dry = wet[p] <= 0.5
    if not dry and (slit_a <= 0.0 or tauy[p] < slit_tau):
        return
    g = (x[p] - origin)/dx
    nid = (int(g[0] + 0.5)*ny + int(g[1] + 0.5))*nz + int(g[2] + 0.5)
    code = gnear[nid]
    if code >= 2147483647:
        return
    k0 = code - (code/1048576)*1048576
    pos = x[p]
    vel = v[p]
    best = int(-1)
    bdist = float(1.0e9)
    brad = wp.vec3(0.0, 0.0, 0.0)
    for kk in range(wp.max(k0 - 2, 0), wp.min(k0 + 3, markers)):
        t = mt[kk]
        hl = 0.5*mlen[kk]
        rel = pos - mx[kk]
        sa = wp.clamp(wp.dot(rel, t), -hl, hl)
        radial = rel - sa*t
        dist = wp.length(radial)
        if dist < bdist:
            bdist = dist
            best = kk
            brad = radial
    if best < 0:
        return
    rad = 0.5*mdia[best]
    lam = carry_lam*mdia[best]
    if bdist >= rad + 3.5*lam and bdist >= rad:
        return
    t = mt[best]
    moved = int(0)
    if dry and bdist < rad:
        nrm = wp.vec3(0.0, 0.0, 1.0)
        if bdist > 1.0e-9:
            nrm = brad/bdist
        else:
            un0 = mu_[best] - wp.dot(mu_[best], t)*t          # dead centre: ahead of the wire
            lu = wp.length(un0)
            if lu > 1.0e-9:
                nrm = un0/lu
        pos = pos + (rad - bdist)*nrm
        vn = wp.dot(vel - mu_[best], nrm)
        if vn < 0.0:
            dvel = -vn*nrm
            vel = vel + dvel
            wp.atomic_add(mf, best, -mass[p]*dvel/dt)
        bdist = rad
        moved = 1
    if dry and carry_w > 0.0:
        # the point goes along the wire's way at least at the rule's share of the wire's speed; what the grid already
        # gives it counts toward that (on finer cells the grid sees more of the flow round the wire itself, and the
        # two must not add up)
        un = mu_[best] - wp.dot(mu_[best], t)*t
        lu = wp.length(un)
        if lu > 1.0e-9:
            e = un/lu
            w = carry_w*wp.exp(-(bdist - rad)/wp.max(lam, 1.0e-9))
            extra = w*lu - wp.dot(vel, e)
            if extra > 0.0:
                pos = pos + extra*dt*e
                moved = 1
    if moved == 1:
        x[p] = pos
        v[p] = vel
        wp.atomic_add(counts, 2, 1)
    if slit_a > 0.0:
        rc = mdia[best]
        if wp.max(bdist, 0.0) < rc:
            un2 = mu_[best] - wp.dot(mu_[best], t)*t
            cut[p] = wp.min(cut[p] + vol0[p]*slit_a*mdia[best]*wp.length(un2)*dt/(3.14159265*rc*rc), 0.5*vol0[p])
            cutd[p] = mdia[best]


@wp.kernel
def k_slit_scan(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), free_liq: wp.array(dtype=float),
                vol0: wp.array(dtype=float), cut: wp.array(dtype=float), creach: wp.array(dtype=float),
                cons: wp.array(dtype=float), tauy: wp.array(dtype=float), k_thin: float, slit_tau: float,
                dx: float, origin: wp.vec3, ny: int, nz: int, greach: wp.array(dtype=float)):
    """Liquid in a wire's cut, first pass: at each point's nearest node, how much of its reach the liquid still has
    from there (1 to 0): all of it where a point holds free liquid, a cut point's own remainder otherwise; the
    largest of the points there."""
    p = wp.tid()
    if flags[p] != 0:
        return
    g = (x[p] - origin)/dx
    nid = (int(g[0] + 0.5)*ny + int(g[1] + 0.5))*nz + int(g[2] + 0.5)
    if free_liq[p] > 0.05*vol0[p] and cons[p] <= k_thin and tauy[p] < slit_tau:
        wp.atomic_max(greach, nid, 1.0)
    elif cut[p] > 0.0:
        wp.atomic_max(greach, nid, creach[p])
    else:
        creach[p] = 0.0


@wp.kernel
def k_slit_hop(greach: wp.array(dtype=float), nx: int, ny: int, nz: int, out: wp.array(dtype=float)):
    """Second pass: the largest remainder among a node and its six neighbours (the liquid may come one cell on)."""
    i = wp.tid()
    iz = i - (i/nz)*nz
    iy = (i/nz) - (i/(nz*ny))*ny
    ix = i/(nz*ny)
    m = greach[i]
    if ix > 0:
        m = wp.max(m, greach[i - ny*nz])
    if ix < nx - 1:
        m = wp.max(m, greach[i + ny*nz])
    if iy > 0:
        m = wp.max(m, greach[i - nz])
    if iy < ny - 1:
        m = wp.max(m, greach[i + nz])
    if iz > 0:
        m = wp.max(m, greach[i - 1])
    if iz < nz - 1:
        m = wp.max(m, greach[i + 1])
    out[i] = m


@wp.kernel
def k_slit_sum(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), spec: wp.array2d(dtype=float),
               free_liq: wp.array(dtype=float), vliq: wp.array(dtype=float), dry_pore: wp.array(dtype=float),
               vol0: wp.array(dtype=float), cons: wp.array(dtype=float), tauy: wp.array(dtype=float),
               wet: wp.array(dtype=float), cut: wp.array(dtype=float), creach: wp.array(dtype=float), k_thin: float,
               slit_tau: float, paste_cost: float, reach_d: float, cutd: wp.array(dtype=float), dx: float,
               origin: wp.vec3, ny: int, nz: int, greach: wp.array(dtype=float), sums: wp.array(dtype=float)):
    """Third pass. A cut point's remainder is the largest next to it less what its own cell costs: the cell's
    length over the reach (reach_d diameters of the wire that made the cut) in dry powder, paste_cost times that in
    a paste. And the totals: liquid wanted by the cuts the liquid reaches (0), free liquid there is (1), and the
    masses of the liquid species in it (4 + species)."""
    p = wp.tid()
    if flags[p] != 0:
        return
    fl = free_liq[p]
    if fl > 0.05*vol0[p] and cons[p] <= k_thin and tauy[p] < slit_tau:
        wp.atomic_add(sums, 1, fl)
        vw = wp.max(vliq[p], 1.0e-30)
        wp.atomic_add(sums, 4, spec[p, 0]*fl/vw)
        wp.atomic_add(sums, 5, spec[p, 1]*fl/vw)
        wp.atomic_add(sums, 12, spec[p, 8]*fl/vw)
        wp.atomic_add(sums, 13, spec[p, 9]*fl/vw)
        wp.atomic_add(sums, 14, spec[p, 10]*fl/vw)
        return
    if cut[p] <= 0.0:
        return
    g = (x[p] - origin)/dx
    nid = (int(g[0] + 0.5)*ny + int(g[1] + 0.5))*nz + int(g[2] + 0.5)
    cost = dx/(reach_d*wp.max(cutd[p], 1.0e-4))
    want = wp.min(cut[p], dry_pore[p])
    if wet[p] > 0.5:
        cost = paste_cost*cost
        want = cut[p]
    a = wp.max(greach[nid] - cost, 0.0)
    creach[p] = a
    if a > 0.0 and want > 0.0:
        wp.atomic_add(sums, 0, want)


@wp.kernel
def k_slit_move(flags: wp.array(dtype=int), spec: wp.array2d(dtype=float), free_liq: wp.array(dtype=float),
                vliq: wp.array(dtype=float), dry_pore: wp.array(dtype=float), vol0: wp.array(dtype=float),
                cons: wp.array(dtype=float), tauy: wp.array(dtype=float), wet: wp.array(dtype=float),
                cut: wp.array(dtype=float), creach: wp.array(dtype=float), k_thin: float, slit_tau: float, phi: float,
                share: float, give: wp.array(dtype=float)):
    """Fourth pass: every point with free thin liquid gives the share phi of it (its liquid species in their own
    proportion); every cut the liquid reaches takes the share `share` of what it wants, with the composition of all
    that was given (give: kg of each species per cubic metre). What a cut has taken is no longer open."""
    p = wp.tid()
    if flags[p] != 0:
        return
    fl = free_liq[p]
    if fl > 0.05*vol0[p] and cons[p] <= k_thin and tauy[p] < slit_tau:
        k = phi*fl/wp.max(vliq[p], 1.0e-30)
        spec[p, 0] = spec[p, 0]*(1.0 - k)
        spec[p, 1] = spec[p, 1]*(1.0 - k)
        spec[p, 8] = spec[p, 8]*(1.0 - k)
        spec[p, 9] = spec[p, 9]*(1.0 - k)
        spec[p, 10] = spec[p, 10]*(1.0 - k)
        return
    if cut[p] <= 0.0 or creach[p] <= 0.0:
        return
    want = wp.min(cut[p], dry_pore[p])
    if wet[p] > 0.5:
        want = cut[p]
    got = share*want
    spec[p, 0] = spec[p, 0] + got*give[0]
    spec[p, 1] = spec[p, 1] + got*give[1]
    spec[p, 8] = spec[p, 8] + got*give[8]
    spec[p, 9] = spec[p, 9] + got*give[9]
    spec[p, 10] = spec[p, 10] + got*give[10]
    cut[p] = cut[p] - got


@wp.kernel
def k_film(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), wet: wp.array(dtype=float),
           phi: wp.array(dtype=float), mass: wp.array(dtype=float), vol0: wp.array(dtype=float),
           cons: wp.array(dtype=float), nidx: wp.array(dtype=float), tauy: wp.array(dtype=float),
           fowner: wp.array(dtype=int), foff: wp.array(dtype=wp.vec3), ftime: wp.array(dtype=float),
           mx: wp.array(dtype=wp.vec3), mu_: wp.array(dtype=wp.vec3), macc: wp.array(dtype=float),
           mdia: wp.array(dtype=float), mlen: wp.array(dtype=float), mrun: wp.array(dtype=float),
           mrho: wp.array(dtype=float), meta_: wp.array(dtype=float), mfilm: wp.array(dtype=float), markers: int,
           dt: float, time: float, tick: int, gravity: float, dx: float, cool: float):
    """The film a wire carries (once per tick).

    Taken up: a wet point that is on its own (gathered fraction under 0.35: a wisp, not mixture) within 0.8 of a cell
    of a wire that is out of the mixture rides that wire from then on, while the wire's film is thinner than a
    surface drawn out of a viscous liquid keeps, h = sqrt(eta U / (rho g)) (Derjaguin's thick-film limit), at most
    three quarters of the wire's width; eta the mixture's viscosity round the wire when it was last in it, U the
    wire's speed.
    Shed: the film drains under what it feels (gravity less the wire's acceleration: on a turning beater the
    centrifugal load, a hundred times gravity) as a film of its own law does, mean speed gdot h n / (2 n + 1) with
    the wall stress rho a h; its points leave as drops, each tick with the chance dt over the time the film takes to
    run off (across the wire's width when thrown, along the wire's run when it drips); a film whose wall stress is
    under its yield stress stays. Back in the mixture the film is mixture again."""
    p = wp.tid()
    f = flags[p]
    if f == 5:
        k = fowner[p]
        if mrho[k] > 300.0:
            flags[p] = 0
            x[p] = mx[k] + foff[p]
            v[p] = mu_[k]
            wp.atomic_add(mfilm, k, -mass[p])
            return
        rho = mass[p]/wp.max(vol0[p], 1.0e-18)
        a = macc[k]
        h = 0.5*mdia[k]
        tw = rho*a*h
        tau = float(1.0e9)
        if tw > tauy[p]:
            n = wp.max(nidx[p], 0.05)
            gd = wp.pow((tw - tauy[p])/wp.max(cons[p], 1.0e-6), 1.0/n)
            u = gd*h*n/(2.0*n + 1.0)
            run = mrun[k]
            if a > 3.0*gravity:
                run = mdia[k]
            tau = run/wp.max(u, 1.0e-9)
        h32 = wp.rand_init(tick, p)
        if wp.randf(h32) < dt/tau:
            flags[p] = 1
            v[p] = mu_[k]
            x[p] = mx[k] + foff[p]
            ftime[p] = time
            wp.atomic_add(mfilm, k, -mass[p])
            return
        x[p] = mx[k] + foff[p]
        v[p] = mu_[k]
        return
    if f != 0 or wet[p] < 0.5 or phi[p] > 0.35 or time - ftime[p] < cool:
        return
    best = int(-1)
    dbest = 0.8*dx
    for k in range(markers):
        d = wp.length(x[p] - mx[k])
        if d < dbest and mrho[k] < 300.0:
            dbest = d
            best = k
    if best < 0:
        return
    U = wp.length(mu_[best])
    h = wp.min(wp.sqrt(meta_[best]*U/(1000.0*gravity)), 0.75*mdia[best])
    cap = 1000.0*h*3.14159265*(mdia[best] + h)*mlen[best]
    if mfilm[best] + mass[p] > cap:
        return
    wp.atomic_add(mfilm, best, mass[p])
    flags[p] = 5
    fowner[p] = best
    foff[p] = x[p] - mx[best]


@wp.kernel
def k_p2g_rheology(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float),
                   cons: wp.array(dtype=float), nidx: wp.array(dtype=float), tauy: wp.array(dtype=float),
                   mudp: wp.array(dtype=float), kbulk: wp.array(dtype=float), jac: wp.array(dtype=float), dx: float,
                   origin: wp.vec3, ny: int, nz: int, gcons: wp.array(dtype=float), gnidx: wp.array(dtype=float),
                   gstr: wp.array(dtype=float), grm: wp.array(dtype=float)):
    """Mass-weighted consistency, index and shear strength (yield stress + friction x pressure) on the nodes, for the
    wires' drag, and the mass they are weighted with (grm). Gathered once a tick, where the points are when the tick
    begins: the wires divide these sums by grm, the mass gathered with them, and not by the mass of a later substep
    (see k_markers)."""
    p = wp.tid()
    if flags[p] != 0:
        return
    strength = tauy[p] + mudp[p]*wp.max(kbulk[p]*(1.0/jac[p] - 1.0), 0.0)
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)*mass[p]
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                wp.atomic_add(gcons, idx, w*cons[p])
                wp.atomic_add(gnidx, idx, w*nidx[p])
                wp.atomic_add(gstr, idx, w*strength)
                wp.atomic_add(grm, idx, w)


@wp.kernel
def k_clear2(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = 0.0
    b[i] = 0.0
    c[i] = 0.0


@wp.kernel
def k_vessel_sums(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float), vpos: wp.vec3,
                  vrot: wp.quat, radius: float, top: float, out: wp.array(dtype=float)):
    """Mass, first and second moments (world frame) of the active points inside the vessel's envelope."""
    p = wp.tid()
    if flags[p] != 0:
        return
    xi = x[p]
    pl = wp.quat_rotate_inv(vrot, xi - vpos)
    if pl[2] > top or pl[2] < -0.02 or pl[0]*pl[0] + pl[1]*pl[1] > radius*radius:
        return
    m = mass[p]
    wp.atomic_add(out, 0, m)
    wp.atomic_add(out, 1, m*xi[0])
    wp.atomic_add(out, 2, m*xi[1])
    wp.atomic_add(out, 3, m*xi[2])
    wp.atomic_add(out, 4, m*xi[0]*xi[0])
    wp.atomic_add(out, 5, m*xi[1]*xi[1])
    wp.atomic_add(out, 6, m*xi[2]*xi[2])
    wp.atomic_add(out, 7, m*xi[0]*xi[1])
    wp.atomic_add(out, 8, m*xi[0]*xi[2])
    wp.atomic_add(out, 9, m*xi[1]*xi[2])


@wp.kernel
def k_expo_gather(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), gexpo: wp.array(dtype=float), dx: float,
                  origin: wp.vec3, ny: int, nz: int, expo: wp.array(dtype=float)):
    """The wires' sweep of this tick, from the nodes onto the points (added to what the points carry since the last
    refresh). Nothing stays on the grid between ticks: the grid region may move and change shape from tick to tick."""
    p = wp.tid()
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
        for j in range(3):
            for k in range(3):
                e = e + bspline(fx, i, j, k)*gexpo[((bx + i)*ny + (by + j))*nz + (bz + k)]
    expo[p] = expo[p] + e


@wp.kernel
def k_exit(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), vpos: wp.vec3, vrot: wp.quat, radius: float,
           top: float, bottom: float, time: float, xout: wp.array(dtype=wp.vec3), tout: wp.array(dtype=float)):
    """Where a point left the vessel's envelope (the cylinder from its base to its rim plane, in the vessel's frame):
    kept from the tick it is first outside, cleared if it comes back in. A point that left above the rim plane went
    over the rim; one that left below it went through the wall or the base, which nothing may do."""
    p = wp.tid()
    if flags[p] != 0:
        return
    pl = wp.quat_rotate_inv(vrot, x[p] - vpos)
    if pl[2] > top or pl[2] < bottom or pl[0]*pl[0] + pl[1]*pl[1] > radius*radius:
        if tout[p] < 0.0:
            xout[p] = pl
            tout[p] = time
    else:
        tout[p] = -1.0


@wp.kernel
def k_probe(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), temp: wp.array(dtype=float),
            heatcap: wp.array(dtype=float), centre: wp.vec3, radius: float, out: wp.array(dtype=float)):
    """Heat capacity and capacity x temperature of the points within `radius` of a point (a thermometer's tip)."""
    p = wp.tid()
    if flags[p] != 0:
        return
    if wp.length(x[p] - centre) > radius:
        return
    wp.atomic_add(out, 0, heatcap[p])
    wp.atomic_add(out, 1, heatcap[p]*temp[p])


# ============================================================================================== pool exchange
@wp.kernel
def k_pool_clear(pool: wp.array2d(dtype=float), poolm: wp.array(dtype=float), poolh: wp.array(dtype=float),
                 poolc: wp.array(dtype=float), gfree: wp.array(dtype=float), gdry: wp.array(dtype=float),
                 gmass: wp.array(dtype=float), gcomp: wp.array2d(dtype=float)):
    i = wp.tid()
    for s in range(NS):
        pool[i, s] = 0.0
        gcomp[i, s] = 0.0
    poolm[i] = 0.0
    poolh[i] = 0.0
    poolc[i] = 0.0
    gfree[i] = 0.0
    gdry[i] = 0.0
    gmass[i] = 0.0


@wp.kernel
def k_pool_scan(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float),
                spec: wp.array2d(dtype=float), free_liq: wp.array(dtype=float), dry_pore: wp.array(dtype=float),
                dx: float, origin: wp.vec3, ny: int, nz: int, gfree: wp.array(dtype=float),
                gdry: wp.array(dtype=float), gmass: wp.array(dtype=float), gcomp: wp.array2d(dtype=float)):
    """First pass: what is around each point (mass, composition, free liquid volume, unfilled pore volume)."""
    p = wp.tid()
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                wp.atomic_add(gmass, idx, w*mass[p])
                wp.atomic_add(gfree, idx, w*free_liq[p])
                wp.atomic_add(gdry, idx, w*dry_pore[p])
                for s in range(NS):
                    wp.atomic_add(gcomp, idx, s, w*spec[p, s])


@wp.kernel
def k_pool_rate(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float),
                spec: wp.array2d(dtype=float), lam_l: wp.array(dtype=float), lam_on: wp.array(dtype=float),
                dmicro: wp.array(dtype=float), gdot: wp.array(dtype=float), dgrain: wp.array(dtype=float),
                wet: wp.array(dtype=float), dx: float, origin: wp.vec3, ny: int, nz: int, spacing: float,
                wet_rate: float, het_on: float, dt: float, mix_c: float, gfree: wp.array(dtype=float),
                gdry: wp.array(dtype=float), gmass: wp.array(dtype=float), gcomp: wp.array2d(dtype=float),
                rate: wp.array(dtype=float), het: wp.array(dtype=float)):
    """Second pass: each point's exchange fraction for this interval.

    rate = D / s^2 with s = spacing * exp(-Lambda): molecular diffusion across lamellae thinned by the resolved flow
    (dissolved species), or the shear diffusivity of grains 0.1 gdot d^2 (dry powder); plus wetting at wet_rate where
    free liquid and unfilled pores are both within reach. The lamella clock runs only while the neighbourhood differs
    from the point (het > het_on).

    The pool spans three cells, so an exchange fraction is also a transport between cells (D_eff = rate spacing^2).
    The share of the rate that comes from thinning is therefore bounded by mix_c gdot: what the flow is straining now
    (a sub-grid diffusivity of the Smagorinsky form). Without the bound a liquid that has stopped keeps exchanging at
    the thinned rate and spreads species across the vessel on its own (E10, case milk_uncapped). mix_c < 0: no bound."""
    p = wp.tid()
    if flags[p] != 0:
        rate[p] = 0.0
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    free = float(0.0)
    dry = float(0.0)
    mm = float(0.0)
    dist = float(0.0)
    m = mass[p]
    for s in range(NS):
        cs = float(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    w = bspline(fx, i, j, k)
                    idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                    gmi = gmass[idx]
                    if gmi > 1.0e-14:
                        cs = cs + w*gcomp[idx, s]/gmi
        dist = dist + wp.abs(cs - spec[p, s]/m)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                free = free + w*gfree[idx]
                dry = dry + w*gdry[idx]
                mm = mm + w*gmass[idx]
    h = 0.5*dist
    het[p] = h
    if h > het_on:
        lam_on[p] = 1.0
    else:
        lam_on[p] = 0.0
    s_l = spacing*wp.exp(-lam_l[p])
    d = dmicro[p]
    if wet[p] < 0.5:
        d = 0.1*gdot[p]*dgrain[p]*dgrain[p]
    r0 = d/(spacing*spacing)
    r = d/(s_l*s_l)
    if mix_c >= 0.0:
        r = r0 + wp.min(r - r0, mix_c*gdot[p])
    if free > 1.0e-12 and dry > 1.0e-12:
        cell = dx*dx*dx
        r = r + wet_rate*wp.min(1.0, free/(0.05*cell))*wp.min(1.0, dry/(0.05*cell))
    rate[p] = wp.min(r*dt, 0.5)


@wp.kernel
def k_pool_put(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float),
               spec: wp.array2d(dtype=float), temp: wp.array(dtype=float), heatcap: wp.array(dtype=float),
               rate: wp.array(dtype=float), hrate: float, dx: float, origin: wp.vec3, ny: int, nz: int,
               pool: wp.array2d(dtype=float), poolm: wp.array(dtype=float), poolh: wp.array(dtype=float),
               poolc: wp.array(dtype=float)):
    p = wp.tid()
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    r = rate[p]
    c = heatcap[p]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                wp.atomic_add(poolm, idx, w*r*mass[p])
                wp.atomic_add(poolh, idx, w*hrate*c*temp[p])
                wp.atomic_add(poolc, idx, w*hrate*c)
                for s in range(NS):
                    wp.atomic_add(pool, idx, s, w*r*spec[p, s])


@wp.kernel
def k_pool_take(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), mass: wp.array(dtype=float),
                spec: wp.array2d(dtype=float), temp: wp.array(dtype=float), heatcap: wp.array(dtype=float),
                rate: wp.array(dtype=float), hrate: float, dx: float, origin: wp.vec3, ny: int, nz: int,
                pool: wp.array2d(dtype=float), poolm: wp.array(dtype=float), poolh: wp.array(dtype=float),
                poolc: wp.array(dtype=float)):
    p = wp.tid()
    if flags[p] != 0:
        return
    inv_dx = 1.0/dx
    xp = (x[p] - origin)*inv_dx
    bx = int(xp[0] - 0.5)
    by = int(xp[1] - 0.5)
    bz = int(xp[2] - 0.5)
    fx = wp.vec3(xp[0] - float(bx), xp[1] - float(by), xp[2] - float(bz))
    r = rate[p]
    m = mass[p]
    c = heatcap[p]
    heat = float(0.0)
    for s in range(NS):
        got = float(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    w = bspline(fx, i, j, k)
                    idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                    pm = poolm[idx]
                    if pm > 1.0e-20:
                        got = got + w*r*m*pool[idx, s]/pm
        spec[p, s] = spec[p, s]*(1.0 - r) + got
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = bspline(fx, i, j, k)
                idx = ((bx + i)*ny + (by + j))*nz + (bz + k)
                pc = poolc[idx]
                if pc > 1.0e-20:
                    heat = heat + w*hrate*c*poolh[idx]/pc
    temp[p] = temp[p]*(1.0 - hrate) + heat/c


def share_table(sdf, h, dx, samples=384, seed=0):
    """Kernel-smoothed solid indicator of a body of revolution on its (r, z) signed-distance table: for each table
    point, the mean of [sdf < 0] over offsets drawn from the grid's quadratic B-spline (a product of three)."""
    rng = np.random.default_rng(seed)
    xi = rng.uniform(-0.5, 0.5, (samples, 3, 3)).sum(2)*dx          # a sum of three uniforms has the B-spline density
    nr, nz = sdf.shape
    R, Z = np.meshgrid(np.arange(nr)*h, np.arange(nz)*h, indexing="ij")
    out = np.zeros(sdf.shape)
    near = np.abs(sdf) < 2.2*dx
    Rn, Zn = R[near], Z[near]
    acc = np.zeros(Rn.shape)
    for k in range(samples):
        rr = np.hypot(Rn + xi[k, 0], xi[k, 1])
        fr = np.clip(rr/h, 0, nr - 1.001)
        fz = np.clip((Zn + xi[k, 2])/h, 0, nz - 1.001)
        i, j = fr.astype(int), fz.astype(int)
        a, b = fr - i, fz - j
        v = (1 - a)*((1 - b)*sdf[i, j] + b*sdf[i, j + 1]) + a*((1 - b)*sdf[i + 1, j] + b*sdf[i + 1, j + 1])
        acc += v < 0
    out[near] = acc/samples
    out[(~near) & (sdf < 0)] = 1.0
    return out.astype(np.float32)


# Free surface: the gathered volume fraction of a resting, jittered lattice at rest density against the gradient of
# that fraction (times the cell): no correction up to the gradient that the interior of a liquid shows once it has
# been stirred and has come to rest (its points' disorder; the lattice's third and fourth layers, 3 % and 1 % short,
# lie under it), then the second layer of points and the top one. Measured on this solver;
# linear in between, held beyond.
FREE_SURFACE = ((0.07, 0.367, 0.596), (1.0, 0.864, 0.620))

# What a wire thinner than a cell does to the material next to it (k_wire_sweep, _slit). Where each number comes from
# is in ledger_mixture.py under the same names; the bench is E20.
WIRE_CARRY = (0.5, 0.7)      # share of the wire's speed at its surface; decay length in wire diameters
WIRE_CUT = 0.5               # open volume a wire leaves, as a share of its diameter per unit of swept area
CUT_REACH_D = 6.0            # wire diameters a liquid follows a cut in dry powder from where the cut borders it
CUT_HOLD_PA = 50.0           # yield stress from which a wet mixture keeps a cut
CUT_PASTE_COST = 1.7         # cells of reach that one cell of cut paste costs the liquid (dry powder: one)
CUT_THIN_PA_S = 1.0          # consistency up to which a free liquid runs into a cut


class Continuum:
    """Material points, grid and colliders. The owner (world.py) feeds body poses and reads back wrenches."""

    def __init__(self, capacity, *, dx=0.0025, sound=6.0, grid_cells=(160, 160, 96), device="cuda:0", gravity=9.81,
                 cfl=0.25, wall_mode=1, wall_n=1.0, vmax=8.0, max_markers=512):
        self.device = device
        self.dx, self.sound, self.gravity, self.cfl = float(dx), float(sound), float(gravity), float(cfl)
        self.spacing = 0.49*self.dx     # exchange length of the pool exchange: E10 measured D_eff = 0.24 r dx^2 for a
                                        # pooled fraction r per second, so r = D / (0.49 dx)^2 gives D_eff = D
        # bound on the thinning share of the exchange: rate <= mix_c * strain rate, i.e. a transport between cells of at
        # most (Cs dx)^2 |S| / Sc_t with the Smagorinsky constant Cs = 0.17 (Lilly) and a turbulent Schmidt number 0.7
        # [prior: the form and the constants are those of sub-grid scalar transport; nothing here calibrates them]
        self.mix_c = (0.17*self.dx/self.spacing)**2/0.7
        # Material at rest next to a solid: the volume ratio is not drawn toward the gathered fraction where a solid's
        # share of the point's kernel exceeds wall_j (ramping off from zero share) AND the point's shear rate is below
        # wall_rate[0] (ramping back on up to wall_rate[1], 1/s). Next to a solid the fraction is an exact solid share
        # plus a sampled liquid share; the two differ by 0.3-0.5 %, and that times the bulk modulus is 100-170 Pa of
        # pressure along every wall, growing with depth, which gravity does not balance: a liquid at rest would circulate.
        # Flowing material needs the steering there as everywhere: carried alone, the ratio of the points sheared
        # along a wall drifts and the material compacts (2-4 % in 10 s of stirring).
        self.wall_j = 0.02
        self.wall_rate = (1.0, 4.0)
        # the free-surface correction of the gathered fraction (k_g2p); 0 switches it off
        self.free_on, self.free_dg = 1.0, 0.0125
        # a wire's plastic resistance: force per length = wire_bearing * shear strength * diameter (k_markers)
        self.wire_bearing = 10.5
        self.lump_tauc = 20.0
        self.capacity = int(capacity)
        self.n = 0
        self.nx, self.ny, self.nz = (int(g) for g in grid_cells)
        self.nodes = self.nx*self.ny*self.nz       # nodes of the region in use; the arrays hold nodes_alloc
        self.envelope = None                       # dict(pos, quat, radius, top, bottom) of the vessel, set by the owner
        self.wall_mode, self.vmax, self.wall_n = int(wall_mode), float(vmax), float(wall_n)
        self.law, self.recipe = None, None
        self.ghost = 1.0
        self.wall_thrust = 0.0                   # k_grid: a TRIAL, off (see k_grid)
        self.mirror_v = (0.010, 0.030)   # m/s, relative speed over which the no-slip reflection switches on
        self.hold_t = 0.25               # cells: wet material is held at nodes this far outside the surface. Calibrated
                                         # on the cup's torque with the one-cell approach band: 1.06 x Couette at 0.30,
                                         # 0.92 at 0.20
        self.hold = 0.97                 # volume fraction above which material counts as pressed on a wall
        self.wall_c = 8.0                # the wall takes tangential velocity at wall_c nu / dx^2 (k_wall): the viscous
                                         # stress eta v / (dx / wall_c) on a node of mass rho dx^3. Calibrated on the
                                         # cup's Couette torque at 3 Pa s (0.70 x Couette at 2,
                                         # 1.000 at 8, 1.10 at 16)
        self.keep_out = True             # a point found inside a solid is set back on its surface once a tick (k_keep_out)
        self.wire_sweep = True           # dry powder in a wire's way is moved out of it and takes its speed (k_wire_sweep)
        # ... and the powder beside a wire is carried along with it: the share of the wire's speed at its surface and
        # the decay length in wire diameters, read off a wire the solver resolves (k_wire_sweep; 0.0 turns it off)
        self.wire_carry_w, self.wire_carry_lam = WIRE_CARRY
        # the cut a wire leaves in dry powder, as a share of its diameter per unit of swept area (k_wire_sweep), and
        # the head that drives a liquid along it (k_slit_*): see ledger_mixture.py, WIRE_CUT and CUT_HEAD
        self.slit_a = WIRE_CUT                   # open cut, as a share of the wire's diameter per unit of swept area
        self.slit_reach_d = CUT_REACH_D          # how far a liquid follows a cut, in wire diameters
        self.slit_tau = CUT_HOLD_PA              # a wet mixture with at least this yield stress keeps a cut
        self.slit_paste_cost = CUT_PASTE_COST    # share of a cell's length a cut in such a paste costs the liquid
        self.slit_k_thin = CUT_THIN_PA_S         # a liquid no thicker than this runs into a cut
        self.slit_moved_m3 = 0.0                 # liquid taken into cuts since the points were set
        self.corner_rule = True          # a wall node's sliding velocity may not enter the solid half a cell on (k_wall)
        self.j_tau = 0.1                 # s, relaxation of the carried volume ratio toward the gathered one
        self.visc_number = 0.05          # nu dt / dx^2 allowed for the explicit viscous share
        self.origin = np.zeros(3)
        self.counter = dict(z=-1e9, lo=(-1e9, -1e9), hi=(1e9, 1e9), floor=-1e9)
        self.counter_col = -1
        self.time, self.ticks = 0.0, 0
        self.limiter = dict(speed_cap=0)
        self._col = []                 # collider definitions (host)
        self._sdf = []
        self.C = None
        self.markers = 0
        self.max_markers = int(max_markers)
        with wp.ScopedDevice(device):
            n = self.capacity
            f = lambda dtype=float: wp.zeros(n, dtype=dtype)
            self.x, self.v = f(wp.vec3), f(wp.vec3)
            self.xr = f(wp.vec3)             # what single precision lost of each point's last position update
            self.sgrad = f()                 # |gradient of the gathered fraction| times the cell, per point
            self.free_tab = wp.array(np.interp(np.arange(64)*self.free_dg, FREE_SURFACE[0], FREE_SURFACE[1]).astype(np.float32),
                                     dtype=float)
            self.aff, self.dev = f(wp.mat33), f(wp.mat33)
            self.phi, self.flags, self.jac = f(), wp.full(n, 4, dtype=int), f()
            self.mass, self.vol0, self.kbulk, self.shear = f(), f(), f(), f()
            self.cons, self.nidx, self.tauy, self.mudp = f(), f(), f(), f()
            self.wet, self.work, self.gdot, self.visc = f(), f(), f(), f()
            self.nuw = f()                   # apparent kinematic viscosity per point, for the wall rule
            self.lam_n, self.lam_l, self.lam_on = f(wp.vec3), f(), f()
            self.spec = wp.zeros((n, NS), dtype=float)
            self.temp, self.heatcap = f(), f()
            self.rate, self.het, self.dmicro, self.dgrain = f(), f(), f(), f()
            self.free_liq, self.dry_pore, self.expo = f(), f(), f()
            self.vliq = f()                  # liquid volume of each point (materials.k_refresh)
            self.cut = f()                   # open cut a wire has left in the dry powder of each point, m3 (k_wire_sweep)
            self.cutd = f()                  # diameter of the wire that left it, m
            self.creach = f()                # how far the liquid has come along that cut, 0 to 1 (k_slit_reach)
            self.slit_sums = wp.zeros(4 + NS, dtype=float)
            self.state = wp.zeros((n, M.NSTATE), dtype=float)       # per-point history (materials.py)
            self.xout = f(wp.vec3)           # where each point left the vessel's envelope (vessel frame), and when
            self.tout = wp.full(n, -1.0, dtype=float)
            self.fowner = wp.full(n, -1, dtype=int)      # the wire a film point rides (k_film)
            self.foff = f(wp.vec3)
            self.ftime = wp.full(n, -1.0e9, dtype=float)  # when a point was last shed from a wire
            self.counts = wp.zeros(4, dtype=int)
        self._alloc_grid(self.nodes)
        with wp.ScopedDevice(device):
            m = self.max_markers
            self.mx, self.mu, self.mt = (wp.zeros(m, dtype=wp.vec3) for _ in range(3))
            self.mlen, self.mdia = wp.zeros(m, dtype=float), wp.zeros(m, dtype=float)
            self.mbody = wp.zeros(m, dtype=int)
            self.mf, self.mrate = wp.zeros(m, dtype=wp.vec3), wp.zeros(m, dtype=float)
            self.mrho, self.meta, self.mfilm = (wp.zeros(m, dtype=float) for _ in range(3))
            self.macc = wp.full(m, float(gravity), dtype=float)   # what a film on the wire feels: |g - the wire's acceleration|
            self.mrun = wp.full(m, 0.03, dtype=float)             # the run of a wire a dripping film drains along, m
        self.substeps = None
        self.dt = None
        self.set_substeps(1.0e-3)

    # ------------------------------------------------------------------ setup
    def _alloc_grid(self, nodes):
        """Node arrays for a region of up to `nodes` nodes. Nothing is kept on the grid between ticks, so a larger
        region (a vessel being tipped) simply gets new arrays."""
        self.nodes_alloc = int(nodes)
        with wp.ScopedDevice(self.device):
            g = lambda dtype=float: wp.zeros(self.nodes_alloc, dtype=dtype)
            self.gm, self.gmv, self.gv = g(), g(wp.vec3), g(wp.vec3)
            self.gvol, self.gsat, self.gphi, self.gnu = g(), g(), g(), g()
            self.gcons, self.gnidx, self.gexpo, self.gstr, self.grm = g(), g(), g(), g(), g()
            self.gkind, self.gbest, self.gwire, self.gnear = g(int), g(int), g(int), g(int)
            self.pool = wp.zeros((self.nodes_alloc, NS), dtype=float)
            self.gcomp = wp.zeros((self.nodes_alloc, NS), dtype=float)
            self.poolm, self.poolh, self.poolc = g(), g(), g()
            self.gfree, self.gdry, self.gmass = g(), g(), g()
            self.greach, self.greach2 = g(), g()
            self.gpv, self.gmo = g(), g(wp.vec3)

    def set_box(self, lo, hi, *, counter=None):
        """Grid region: the world-aligned box from `lo` to `hi` (to whole cells, PAD cells added at every face). Its
        size follows the box: a tipped vessel's box is taller and wider than an upright one's."""
        dx = self.dx
        a = np.floor(np.asarray(lo, float)/dx).astype(int) - PAD
        b = np.ceil(np.asarray(hi, float)/dx).astype(int) + PAD
        nx, ny, nz = (int(v) for v in (b - a + 1))
        if nx*ny*nz > self.nodes_alloc:
            self._alloc_grid(int(1.25*nx*ny*nz))
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nodes = nx*ny*nz
        self.origin = a*dx
        if counter is not None:
            self.counter = dict(counter)

    def set_substeps(self, tick, shear_max=4000.0, rho_min=500.0):
        """Substeps per tick from the fastest elastic wave: sqrt(c^2 + 4 G / (3 rho))."""
        c = math.sqrt(self.sound**2 + 4.0*shear_max/(3.0*rho_min))
        self.substeps = max(1, int(math.ceil(tick/(self.cfl*self.dx/c))))
        self.dt = tick/self.substeps
        self.eta_cap = self.visc_number*self.dx**2/self.dt     # explicit share of the kinematic viscosity (m2/s)
        self.tick_dt = tick

    def add_collider(self, kind, *, body=-1, pos=(0, 0, 0), quat=(0, 0, 0, 1), size=(0, 0, 0), sdf=None, mu_dry=0.19,
                     stick=True, bound=None, name=""):
        """A collider fixed in a rigid body's frame (body -1: the world). `pos`, `quat` (xyzw): its frame in the body.
        sdf: dict(sdf (nr, nz), h, z0) of a body of revolution about its local z. Returns its index."""
        if self.C is not None:
            raise RuntimeError("Colliders are fixed once the first step has run")
        if kind == KIND_REVOLVED:
            s = np.asarray(sdf["sdf"], np.float32)
            off = sum(a.size for a in self._sdf)
            self._sdf.append(s)
            meta = (off, s.shape[0], s.shape[1], float(sdf["h"]), float(sdf["z0"]))
            if bound is None:
                bound = float(np.hypot(s.shape[0]*sdf["h"], max(abs(sdf["z0"]), abs(sdf["z0"] + s.shape[1]*sdf["h"]))))
        else:
            meta = (0, 0, 0, 1.0, 0.0)
            if bound is None:
                bound = 1e6 if kind == KIND_PLANE else float(np.linalg.norm(size) + (size[1] if kind == KIND_CAPSULE else 0.))
        self._col.append(dict(kind=int(kind), body=int(body), pos=np.asarray(pos, float), quat=np.asarray(quat, float),
                              size=np.asarray(size, float), meta=meta, mu=float(mu_dry), stick=1.0 if stick else 0.0,
                              bound=float(bound), name=name))
        return len(self._col) - 1

    def _finalize(self):
        cols = self._col
        n = max(1, len(cols))
        with wp.ScopedDevice(self.device):
            C = Colliders()
            C.n = len(cols)
            arr = lambda vals, dtype: wp.array(np.asarray(vals), dtype=dtype)
            pad = lambda vals, fill: list(vals) + [fill]*(n - len(vals))
            C.kind = arr(pad([c["kind"] for c in cols], 3), int)
            C.pos = wp.zeros(n, dtype=wp.vec3)
            C.rot = wp.array(np.tile([0., 0., 0., 1.], (n, 1)).astype(np.float32), dtype=wp.quat)
            C.vel, C.omega = wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.vec3)
            C.size = arr(np.array(pad([c["size"] for c in cols], np.zeros(3)), np.float32), wp.vec3)
            C.bound = arr(np.array(pad([c["bound"] for c in cols], 0.), np.float32), float)
            C.mu_dry = arr(np.array(pad([c["mu"] for c in cols], 0.), np.float32), float)
            C.stick = arr(np.array(pad([c["stick"] for c in cols], 0.), np.float32), float)
            meta = np.array(pad([c["meta"] for c in cols], (0, 0, 0, 1.0, 0.0)), float)
            C.sdf_off, C.sdf_nr, C.sdf_nz = (arr(meta[:, k].astype(np.int32), int) for k in range(3))
            C.sdf_h, C.sdf_z0 = arr(meta[:, 3].astype(np.float32), float), arr(meta[:, 4].astype(np.float32), float)
            flat = np.concatenate([a.ravel() for a in self._sdf]) if self._sdf else np.zeros(4, np.float32)
            C.sdf = wp.array(flat.astype(np.float32), dtype=float)
            shares = [share_table(a, c["meta"][3], self.dx) for a, c in zip(self._sdf, [c for c in cols if c["kind"] == 0])]
            flat = np.concatenate([a.ravel() for a in shares]) if shares else np.zeros(4, np.float32)
            C.share = wp.array(flat.astype(np.float32), dtype=float)
            self.C = C
            self.col_f, self.col_t = wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.vec3)
        self.col_body = np.array([c["body"] for c in cols], int)

    def set_bodies(self, body_q, body_qd, body_com):
        """Poses (x, y, z, qx, qy, qz, qw), twists (v of the centre of mass, omega) and local centres of mass of every
        rigid body at the start of the tick."""
        from scipy.spatial.transform import Rotation
        if self.C is None:
            self._finalize()
        cols = self._col
        if not cols:
            return
        pos, rot, vel, om = [], [], [], []
        self._col_origin = []
        for c in cols:
            b = c["body"]
            if b < 0:
                p, q = c["pos"], c["quat"]
                v = w = np.zeros(3)
            else:
                R = Rotation.from_quat(body_q[b, 3:7])
                p = body_q[b, :3] + R.apply(c["pos"])
                q = (R*Rotation.from_quat(c["quat"])).as_quat()
                com = body_q[b, :3] + R.apply(body_com[b])
                w = body_qd[b, 3:6]
                v = body_qd[b, :3] + np.cross(w, p - com)
            pos.append(p); rot.append(q); vel.append(v); om.append(w)
        self._col_pos0 = np.array(pos)
        self.C.pos.assign(np.array(pos, np.float32))
        self.C.rot.assign(np.array(rot, np.float32))
        self.C.vel.assign(np.array(vel, np.float32))
        self.C.omega.assign(np.array(om, np.float32))

    def set_region(self, centre, *, counter=None):
        """Grid region centred (to whole cells) on `centre`; counter: dict(z, lo, hi, floor) for spilt points."""
        half = 0.5*self.dx*np.array([self.nx, self.ny, self.nz])
        self.origin = np.round((np.asarray(centre, float) - half)/self.dx)*self.dx
        if counter is not None:
            self.counter = dict(counter)

    def set_points(self, x, species, *, temp=20.0, v=None):
        """Replace all points. species: (n, NS) masses (kg). Mechanical properties follow from materials.refresh."""
        n = len(x)
        if n > self.capacity:
            raise ValueError(f"{n} points exceed the capacity {self.capacity}")
        self.n = n
        pad = lambda a, shape: np.concatenate([np.asarray(a, np.float32), np.zeros(shape, np.float32)])
        k = self.capacity - n
        self.x.assign(pad(x, (k, 3)))
        self.xr.zero_()
        self.v.assign(pad(np.zeros((n, 3)) if v is None else v, (k, 3)))
        self.spec.assign(pad(species, (k, NS)))
        fl = np.full(self.capacity, 4, np.int32)
        fl[:n] = 0
        self.flags.assign(fl)
        self.temp.assign(pad(np.full(n, temp, np.float32) if np.isscalar(temp) else temp, (k,)))
        for a in (self.aff, self.dev, self.phi, self.work, self.gdot, self.visc, self.lam_l, self.lam_on, self.state):
            a.zero_()
        ln = np.zeros((self.capacity, 3), np.float32)
        ln[:, 2] = 1.0
        self.lam_n.assign(ln)
        self.phi.fill_(1.0)
        self.jac.fill_(1.0)
        self.xout.zero_()
        self.tout.fill_(-1.0)
        self.fowner.fill_(-1)
        self.foff.zero_()
        self.ftime.fill_(-1.0e9)
        self.mfilm.zero_()
        self.nuw.fill_(1.0e3)            # until the first substep has measured a shear rate, material holds to walls
        self.expo.zero_()
        self.cut.zero_()
        self.cutd.zero_()
        self.creach.zero_()
        self.mass.assign(pad(np.asarray(species, np.float64).sum(1), (k,)))

    def set_recipe(self, recipe):
        self.recipe = recipe
        self.law = M.make_law(recipe, self.device)
        self.lump_tauc = float(recipe.lump_tauc)
        self.sound = float(recipe.sound)
        self.set_substeps(self.tick_dt, shear_max=max(recipe.g_paste, recipe.g_powder), rho_min=500.0)

    def refresh(self, dt=M.REFRESH):
        """Recompute every point's properties from its composition, temperature and history; advance the slow
        chemistry by dt (0: properties only)."""
        if self.law is None:
            raise RuntimeError("set_recipe first")
        with wp.ScopedDevice(self.device):
            wp.launch(M.k_refresh, dim=self.capacity, inputs=[
                self.spec, self.temp, self.state, self.flags, self.gdot, self.expo, self.law, float(dt), self.work,
                self.mass, self.vol0, self.kbulk, self.shear, self.cons, self.nidx, self.tauy, self.mudp, self.wet,
                self.heatcap, self.dmicro, self.dgrain, self.free_liq, self.dry_pore, self.vliq])
            self.expo.zero_()                # the wires' sweep since the last refresh (step gathers it every tick)

    def set_uniform(self, *, rho, k, n=1.0, tauy=0.0, shear=2000.0, mu=0.0, wet=1.0, kbulk=None, volume=None):
        """Bench use: give every point the same law directly, bypassing the recipe (probes of the solver itself)."""
        vol = (0.5*self.dx)**3 if volume is None else volume
        for a, val in ((self.vol0, vol), (self.mass, rho*vol), (self.cons, k), (self.nidx, n), (self.tauy, tauy),
                       (self.shear, shear), (self.mudp, mu), (self.wet, wet),
                       (self.kbulk, rho*self.sound**2 if kbulk is None else kbulk), (self.heatcap, rho*vol*3900.0)):
            a.fill_(float(val))
        self.set_substeps(self.tick_dt, shear_max=shear, rho_min=rho)

    # ------------------------------------------------------------------ stepping
    def step(self):
        """One tick. Returns (force, torque about each collider's origin at the tick start) on every collider,
        averaged over the tick (N, N m)."""
        if self.C is None:
            self._finalize()
        n, dt, dx = self.capacity, self.dt, self.dx
        o = wp.vec3(*self.origin.tolist())
        cz = self.counter
        with wp.ScopedDevice(self.device):
            self.col_f.zero_()
            self.col_t.zero_()
            if self.markers:
                wp.launch(k_clear2, dim=self.nodes, inputs=[self.gcons, self.gnidx, self.gstr])
                self.grm.zero_()
                wp.launch(k_p2g_rheology, dim=n, inputs=[self.x, self.flags, self.mass, self.cons, self.nidx, self.tauy,
                                                         self.mudp, self.kbulk, self.jac, dx, o, self.ny, self.nz,
                                                         self.gcons, self.gnidx, self.gstr, self.grm])
            for s in range(self.substeps):
                wp.launch(k_clear, dim=self.nodes, inputs=[self.gm, self.gmv, self.gvol, self.gsat, self.gnu, self.gpv,
                                                           self.gmo])
                wp.launch(k_p2g, dim=n, inputs=[self.x, self.v, self.aff, self.dev, self.jac, self.flags, self.mass,
                                                self.vol0, self.kbulk, self.wet, self.visc, self.nuw, dt, dx, o, self.ny,
                                                self.nz, self.gm, self.gmv, self.gvol, self.gsat, self.gnu, self.gpv,
                                                self.gmo, float(self.wall_thrust)])
                wp.launch(k_grid, dim=self.nodes, inputs=[self.gm, self.gmv, self.gvol, self.gv, self.gphi, self.gkind,
                                                          self.gbest, self.C, dt, dx, o, self.ny, self.nz,
                                                          self.gravity, self.wall_n, self.ghost, self.gpv, self.gmo,
                                                          float(self.wall_thrust)])
                wp.launch(k_wall, dim=self.nodes, inputs=[self.gm, self.gsat, self.gphi, self.gv, self.gkind, self.gbest, self.C,
                                                          dt, dx, o, self.nx, self.ny, self.nz, self.wall_mode,
                                                          self.hold, self.mirror_v[0], self.mirror_v[1], self.hold_t,
                                                          self.gnu, float(self.wall_c), float(self.wall_n),
                                                          1.0 if self.corner_rule else 0.0, self.col_f,
                                                          self.col_t])
                if self.markers:
                    if self.wire_sweep:
                        self.gwire.zero_()
                    wp.launch(k_markers, dim=self.markers, inputs=[self.mx, self.mu, self.mt, self.mlen, self.mdia,
                                                                   self.mbody, self.gm, self.gv, self.gcons, self.gnidx,
                                                                   self.gstr, self.grm, self.gsat, dt, dx, o, self.nx, self.ny,
                                                                   self.nz, self.lump_tauc, float(self.wire_bearing),
                                                                   self.gexpo, self.mf, self.mrate, self.mrho, self.meta,
                                                                   self.gwire, 1.0 if self.wire_sweep else 0.0])
                wp.launch(k_g2p, dim=n, inputs=[self.x, self.xr, self.v, self.aff, self.dev, self.phi, self.jac, self.flags,
                                                self.mass,
                                                self.vol0, self.kbulk, self.shear, self.cons, self.nidx, self.tauy,
                                                self.mudp, self.work, self.gdot, self.visc, self.lam_n, self.lam_l,
                                                self.lam_on, self.gv, self.gphi, self.gvol, dt, dx, o, self.nx, self.ny, self.nz,
                                                self.vmax, self.eta_cap, min(1.0, dt/self.j_tau), float(self.wall_j),
                                                float(self.wall_rate[0]), float(self.wall_rate[1]),
                                                float(self.free_on), float(self.free_dg), self.free_tab, self.sgrad,
                                                self.nuw, self.counts])
                if self.markers and self.wire_sweep:
                    # the nodes within the carry's reach of a segment (half the longest segment, the wire's radius and
                    # 3.5 decay lengths of the thickest wire, and the cell a point's nearest node may be off by)
                    self.gnear.fill_(2147483647)
                    wp.launch(k_wire_near, dim=self.markers, inputs=[self.mx, self.mt, self.mlen, dx, o, self.nx, self.ny,
                                                                     self.nz, int(self._wire_reach()), self.gnear])
                    wp.launch(k_wire_sweep, dim=n, inputs=[self.x, self.v, self.flags, self.wet, self.mass, self.gnear,
                                                           self.mx, self.mu, self.mt, self.mlen, self.mdia,
                                                           int(self.markers), dx, dt, o, self.ny, self.nz,
                                                           float(self.wire_carry_w), float(self.wire_carry_lam), self.mf,
                                                           self.counts,
                                                           self.vol0, self.cut, float(self.slit_a), self.tauy,
                                                           float(self.slit_tau), self.cutd])
                wp.launch(k_advance_colliders, dim=max(1, len(self._col)), inputs=[self.C.pos, self.C.rot, self.C.vel,
                                                                                   self.C.omega, dt])
            if self.keep_out:
                wp.launch(k_keep_out, dim=n, inputs=[self.x, self.v, self.flags, self.mass, self.gkind, self.C, dx, dt, o,
                                                     self.ny, self.nz, self.col_f, self.col_t, self.counts])
            if self.markers:
                wp.launch(k_expo_gather, dim=n, inputs=[self.x, self.flags, self.gexpo, dx, o, self.ny, self.nz, self.expo])
                self.gexpo.zero_()
                wp.launch(k_film, dim=n, inputs=[self.x, self.v, self.flags, self.wet, self.phi, self.mass, self.vol0,
                                                 self.cons, self.nidx, self.tauy, self.fowner, self.foff, self.ftime,
                                                 self.mx, self.mu, self.macc, self.mdia, self.mlen, self.mrun, self.mrho,
                                                 self.meta, self.mfilm, int(self.markers), self.tick_dt, float(self.time),
                                                 int(self.ticks), self.gravity, dx, FILM_COOL_S])
            if self.envelope is not None:
                e = self.envelope
                wp.launch(k_exit, dim=n, inputs=[self.x, self.flags, wp.vec3(*map(float, e["pos"])),
                                                 wp.quat(*map(float, e["quat"])), float(e["radius"]), float(e["top"]),
                                                 float(e["bottom"]), float(self.time), self.xout, self.tout])
            # points outside the grid region: free flight, once per tick (they re-enter the grid if they fall back)
            wp.launch(k_ballistic, dim=n, inputs=[self.x, self.v, self.flags, self.phi, self.tick_dt, self.gravity, o, dx,
                                                  self.nx, self.ny, self.nz, float(cz["z"]), wp.vec2(*cz["lo"]),
                                                  wp.vec2(*cz["hi"]), float(cz["floor"]), self.C,
                                                  int(self.counter_col), self.ftime,
                                                  float(self.time), FILM_COOL_S])
        self.time += self.tick_dt
        self.ticks += 1
        return self.col_f, self.col_t

    def _wire_reach(self):
        """Cells round a wire segment whose nodes must know it (k_wire_near), from the segments as they are now."""
        key = (int(self.markers), float(self.wire_carry_w), float(self.wire_carry_lam))
        if getattr(self, "_wire_reach_key", None) != key:
            m = int(self.markers)
            ln = float(self.mlen.numpy()[:m].max()) if m else 0.0
            d = float(self.mdia.numpy()[:m].max()) if m else 0.0
            r = 0.5*ln + 0.5*d + (3.5*self.wire_carry_lam*d if self.wire_carry_w > 0.0 else 0.0)
            self._wire_reach_n = int(r/self.dx + 0.87) + 1
            self._wire_reach_key = key
        return self._wire_reach_n

    def wrenches(self):
        """Host copy of the tick's collider forces and torques (about the collider origins at the tick start)."""
        s = 1.0/self.substeps
        return self.col_f.numpy().astype(np.float64)*s, self.col_t.numpy().astype(np.float64)*s

    # ------------------------------------------------------------------ composition and heat
    def exchange(self, dt, *, wet_rate=0.0, heat_rate=0.0, het_on=0.02):
        """Pool exchange of species (per-point rate from the lamellar closure and wetting) and of heat (uniform rate
        heat_rate = fraction of a point's heat pooled over dt) for the interval dt."""
        n, dx = self.capacity, self.dx
        o = wp.vec3(*self.origin.tolist())
        with wp.ScopedDevice(self.device):
            wp.launch(k_pool_clear, dim=self.nodes, inputs=[self.pool, self.poolm, self.poolh, self.poolc, self.gfree,
                                                            self.gdry, self.gmass, self.gcomp])
            wp.launch(k_pool_scan, dim=n, inputs=[self.x, self.flags, self.mass, self.spec, self.free_liq,
                                                  self.dry_pore, dx, o, self.ny, self.nz, self.gfree, self.gdry,
                                                  self.gmass, self.gcomp])
            wp.launch(k_pool_rate, dim=n, inputs=[self.x, self.flags, self.mass, self.spec, self.lam_l, self.lam_on,
                                                  self.dmicro, self.gdot, self.dgrain, self.wet, dx, o, self.ny,
                                                  self.nz, self.spacing, float(wet_rate), float(het_on), float(dt),
                                                  float(self.mix_c), self.gfree, self.gdry, self.gmass, self.gcomp, self.rate, self.het])
            wp.launch(k_pool_put, dim=n, inputs=[self.x, self.flags, self.mass, self.spec, self.temp, self.heatcap,
                                                 self.rate, float(heat_rate), dx, o, self.ny, self.nz, self.pool,
                                                 self.poolm, self.poolh, self.poolc])
            wp.launch(k_pool_take, dim=n, inputs=[self.x, self.flags, self.mass, self.spec, self.temp, self.heatcap,
                                                  self.rate, float(heat_rate), dx, o, self.ny, self.nz, self.pool,
                                                  self.poolm, self.poolh, self.poolc])
            if self.markers and self.slit_a > 0.0 and self.wire_sweep:
                self._slit(dt, o)

    def _slit(self, dt, o):
        """Liquid into the cuts wires have left (k_wire_sweep's third part), every exchange interval.

        A cut that borders free thin liquid fills from it, and the liquid goes on along the cut: through dry powder
        for slit_reach_d wire diameters and no farther (the powder closes behind the wire), through a paste that
        holds its cut open nearly without loss. A cut the liquid reaches takes liquid up to its open volume (in dry
        powder: up to the dry pore volume of its point); the liquid is taken, in its own composition, from every
        point that holds free thin liquid, each in proportion to what it holds (the pool's level falls: no more than
        a fifth of it in one interval). The mixture's mass is conserved to rounding; where in the pool the liquid
        came from is not followed.
        What it stands on is in ledger_mixture.py (WIRE_CUT and the entries after it): the resolved wire drawn from
        a pool of thin liquid into flour, into a paste and through a free heap (E20)."""
        n, dx = self.capacity, self.dx
        kt, st = float(self.slit_k_thin), float(self.slit_tau)
        self.greach.zero_()
        self.slit_sums.zero_()
        wp.launch(k_slit_scan, dim=n, inputs=[self.x, self.flags, self.free_liq, self.vol0, self.cut, self.creach,
                                              self.cons, self.tauy, kt, st, dx, o, self.ny, self.nz, self.greach])
        wp.launch(k_slit_hop, dim=self.nodes, inputs=[self.greach, self.nx, self.ny, self.nz, self.greach2])
        wp.launch(k_slit_sum, dim=n, inputs=[self.x, self.flags, self.spec, self.free_liq, self.vliq, self.dry_pore,
                                             self.vol0, self.cons, self.tauy, self.wet, self.cut, self.creach, kt, st,
                                             float(self.slit_paste_cost), float(self.slit_reach_d), self.cutd, dx, o,
                                             self.ny, self.nz, self.greach2, self.slit_sums])
        sm = self.slit_sums.numpy().astype(np.float64)
        want, free = float(sm[0]), float(sm[1])
        if want <= 0.0 or free <= 0.0:
            return
        moved = min(want, 0.2*free)
        give = np.zeros(NS, np.float32)
        give[[0, 1, 8, 9, 10]] = (sm[[4, 5, 12, 13, 14]]/free).astype(np.float32)      # kg per m3 of the pool's free liquid
        wp.launch(k_slit_move, dim=n, inputs=[self.flags, self.spec, self.free_liq, self.vliq, self.dry_pore, self.vol0,
                                              self.cons, self.tauy, self.wet, self.cut, self.creach, kt, st,
                                              float(moved/free), float(moved/want),
                                              wp.array(give, dtype=float, device=self.device)])
        self.slit_moved_m3 += moved

    # ------------------------------------------------------------------ readback
    def read(self, *names):
        out = []
        for name in names:
            a = getattr(self, name).numpy()
            out.append(a[:self.n].copy())
        return out[0] if len(out) == 1 else out

    def finite(self):
        x, v = self.read("x", "v")
        return bool(np.isfinite(x).all() and np.isfinite(v).all())

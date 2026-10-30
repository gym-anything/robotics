"""Explicit GPU tissue solver (Warp) for knife cutting of biological soft tissue.

One particle array holds three kinds of nodes:
  kind 0  tissue (tetrahedral FEM, stable neo-Hookean as in Newton 1.5 VBD, no log term:
          P = mu F + lam' (J - alpha) cof(F),  lam' = lam + mu,  alpha = 1 + mu/lam'  -> P(I) = 0)
  kind 1  membrane (skin / silver skin; triangle shells: neo-Hookean membrane)
  kind 2  bone rod nodes (discrete rods: axial + bending springs; XPBD-free, explicit with stable dt)
and four force-element families: tets, membrane triangles, rod links/hinges, and cohesive ties
(a particle tied to a material point of a tet by a bilinear traction-separation law; skin-flesh,
bone-flesh and muscle-seam interfaces).

Rigid bodies (knife blade segments, finger pads/inserts, tweezer tips, board, bench, trays) are
kinematic within one rigid tick: their pose at the tick start and their twist are given, contact
forces on them are accumulated as wrenches about their centre of mass and returned to MuJoCo
(label_application pattern).

The knife edge is handled by the edge process-zone law: each edge sample
is tied to the material point where it entered intact tissue; the tie load G in the (cut direction n,
edge tangent t) plane obeys the Atkins frictionless slicing locus  G_n^2 + G_t^2 <= R*l*G_n
(a circle of diameter R*l); beyond it the anchor slips along G (collinear flow) and the slip normal
to the edge is new crack length: dissipation = R * l * slip_n exactly (energy per unit new area = R).
Topology (splitting tetrahedra along the swept surface) is done on the host (topology.py).
"""

import math
import numpy as np
import warp as wp

from .cohesive import cohesive_spent

# ------------------------------------------------------------------ shape kinds for rigid colliders
BOX, WEDGE, CAPSULE = 0, 1, 2
# m/s: the fastest a contact's penalty pushes a node away from a shape (k_contacts). On the coupon press
# 1 m/s leaves the plateau (24.81 N) and the ledger (-0.25 %, -0.17 % without) as they were; 0.25-0.5 m/s moved the
# ledger to -2 % (development runs)
V_SEP = wp.constant(1.0)
MAX_SHAPES = 256
CAND_CAP = 48          # contact candidates per particle per tick (more: that particle tests every shape)
CAND_MARGIN_M = .03    # m: reach margin for the knife's bodies (a 6 m/s knife moves 6 mm in a tick, the fastest in E15)
CAND_SLOW_MARGIN_M = .003   # m: reach margin for bones and static shapes, plus each side's own travel in the tick

wp.config.enable_backward = False


@wp.func
def quat_rotate_inv(q: wp.quat, v: wp.vec3):
    return wp.quat_rotate(wp.quat_inverse(q), v)


@wp.func
def shape_pose_at(p0: wp.vec3, q0: wp.quat, vlin: wp.vec3, wang: wp.vec3, t: float):
    """Pose after time t under constant twist (world-frame angular velocity)."""
    p = p0 + vlin * t
    ang = wp.length(wang) * t
    q = q0
    if ang > 1.0e-12:
        axis = wp.normalize(wang)
        q = wp.quat_from_axis_angle(axis, ang) * q0
    return wp.transform(p, q)


@wp.func
def box_sdf(pl: wp.vec3, h: wp.vec3):
    """Signed distance of local point to a box and the outward normal (local)."""
    d = wp.vec3(wp.abs(pl[0]) - h[0], wp.abs(pl[1]) - h[1], wp.abs(pl[2]) - h[2])
    outside = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
    ol = wp.length(outside)
    if ol > 0.0:
        n = wp.vec3(wp.sign(pl[0]) * outside[0], wp.sign(pl[1]) * outside[1], wp.sign(pl[2]) * outside[2]) / ol
        return ol, n
    # inside: nearest face
    dist = d[0]
    n = wp.vec3(wp.sign(pl[0]), 0.0, 0.0)
    if d[1] > dist:
        dist = d[1]
        n = wp.vec3(0.0, wp.sign(pl[1]), 0.0)
    if d[2] > dist:
        dist = d[2]
        n = wp.vec3(0.0, 0.0, wp.sign(pl[2]))
    return dist, n


@wp.func
def wedge_sdf(pl: wp.vec3, h: wp.vec3, bevel_tan: float, edge_half: float):
    """Blade segment in its local frame: x along blade length in [-hx, hx], z from edge (z=0) to spine
    (z = 2*hz), y thickness. Half thickness w(z) = min(z*tan(bevel), edge_half + (hy - edge_half)*z/(2hz)).
    Points below the edge (z < 0) are OUTSIDE: the edge itself is handled by the edge law.
    Returns penetration depth (>0 inside), local normal (pointing out of the blade)."""
    H = 2.0 * h[2]
    z = pl[2]
    if z <= 0.0 or z >= H or wp.abs(pl[0]) >= h[0]:
        return -1.0, wp.vec3(0.0, 0.0, 0.0)
    w = wp.min(z * bevel_tan, edge_half + (h[1] - edge_half) * z / H)
    pen = w - wp.abs(pl[1])
    if pen <= 0.0:
        return -1.0, wp.vec3(0.0, 0.0, 0.0)
    s = wp.sign(pl[1])
    if s == 0.0:
        s = 1.0
    n = wp.vec3(0.0, s, 0.0)
    # near the spine or the tip/heel, the nearer face wins
    top = H - z
    if top < pen:
        return top, wp.vec3(0.0, 0.0, 1.0)
    endd = h[0] - wp.abs(pl[0])
    if endd < pen:
        return endd, wp.vec3(wp.sign(pl[0]), 0.0, 0.0)
    return pen, n


@wp.func
def capsule_sdf(pl: wp.vec3, h: wp.vec3):
    """Capsule along local x with half length h[0] and radius h[1]."""
    x = wp.clamp(pl[0], -h[0], h[0])
    d = pl - wp.vec3(x, 0.0, 0.0)
    l = wp.length(d)
    if l < 1.0e-12:
        return h[1], wp.vec3(0.0, 0.0, 1.0)
    return h[1] - l, d / l


@wp.func
def in_material(Xr: wp.vec3, tplanes: wp.array2d(dtype=wp.vec4), tnplanes: wp.array(dtype=wp.int32), t: int):
    """Virtual-node copies overlap geometrically; a rest point belongs to copy t only if it satisfies all
    of t's material half-spaces (n . X - c) >= -tol (stored with the side sign folded into n, c)."""
    for i in range(tnplanes[t]):
        pl = tplanes[t, i]
        if pl[0] * Xr[0] + pl[1] * Xr[1] + pl[2] * Xr[2] - pl[3] < -2.0e-5:
            return False
    return True


# ======================================================================================== elasticity
@wp.kernel
def k_tet_forces(n_tets: int, tets: wp.array(dtype=wp.vec4i), active: wp.array(dtype=wp.int32),
                 x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat33),
                 V0: wp.array(dtype=float), tscale: wp.array(dtype=float), tmat: wp.array(dtype=wp.int32), mat_mu: wp.array(dtype=float),
                 mat_lam: wp.array(dtype=float), mat_eta: wp.array(dtype=float), f: wp.array(dtype=wp.vec3),
                 dissip: wp.array(dtype=wp.float64), dt: float, mat_gv: wp.array(dtype=float),
                 mat_trel: wp.array(dtype=float), tSv: wp.array(dtype=wp.mat33), tEp: wp.array(dtype=wp.mat33)):
    t = wp.tid()
    if t >= n_tets or active[t] == 0:
        return
    idx = tets[t]
    x0 = x[idx[0]]
    Ds = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0)
    Dmi = Dm_inv[t]
    F = Ds * Dmi
    m = tmat[t]
    mu = mat_mu[m]
    lamnh = mat_lam[m] + mu
    alpha = 1.0 + mu / lamnh
    J = wp.determinant(F)
    c0 = wp.cross(wp.vec3(F[0, 1], F[1, 1], F[2, 1]), wp.vec3(F[0, 2], F[1, 2], F[2, 2]))
    c1 = wp.cross(wp.vec3(F[0, 2], F[1, 2], F[2, 2]), wp.vec3(F[0, 0], F[1, 0], F[2, 0]))
    c2 = wp.cross(wp.vec3(F[0, 0], F[1, 0], F[2, 0]), wp.vec3(F[0, 1], F[1, 1], F[2, 1]))
    cof = wp.matrix_from_cols(c0, c1, c2)
    P = mu * F + lamnh * (J - alpha) * cof
    # Kelvin-Voigt damping on the Green strain rate (objective): S_v = eta' dC/dt, P_v = F S_v. The damping is numerical
    # (params.FISH_TAU_V: it damps mesh-scale oscillation), and on the Green strain rate its tangent grows with stretch
    # as sigma_max^2 (the current-configuration viscosity as stretch^4): stretched flesh beside a cut drove the
    # explicit step's damping limit to 30 times the rest value and, run past it, booked 0.81 J of 'viscous
    # dissipation' that was energy the oscillation created (A1). Its coefficient is
    # therefore normalised by the stretch, eta' = eta 3 / max(3, tr C): unchanged at rest, its tangent bounded by 6 eta
    # at any stretch. The power it books, V0 eta' Cd:Cd / 2, stays non-negative.
    eta = mat_eta[m]
    if eta > 0.0:
        eta = eta * 3.0 / wp.max(3.0, wp.ddot(F, F))
        v0 = v[idx[0]]
        Vs = wp.matrix_from_cols(v[idx[1]] - v0, v[idx[2]] - v0, v[idx[3]] - v0)
        Fd = Vs * Dmi
        Cd = wp.transpose(Fd) * F + wp.transpose(F) * Fd
        P = P + eta * (F * Cd)
        # viscous power = V0 * S_v : Edot = V0 * eta' * Cd:Cd / 2
        pw = V0[t] * tscale[t] * eta * 0.5 * wp.ddot(Cd, Cd)
        wp.atomic_add(dissip, 0, wp.float64(pw * dt))
    # structural damping (params.TISSUE_VISCO): a Maxwell branch on the deviatoric Green strain, S' + S/tau = 2 gv E_dev',
    # integrated exactly over the substep (S1 = a S0 + 2 gv b dE_dev, a = exp(-dt/tau), b = (1 - a) tau/dt); its
    # stress adds P = F S1. Booked: the work of the stress that acted over the step (S0 : dE) less the change of the
    # branch's stored energy |S|^2 / (4 gv) (counted in the elastic energy, k_tet_energy)
    gv = mat_gv[m]
    if gv > 0.0:
        Eg = 0.5 * (wp.transpose(F) * F - wp.identity(n=3, dtype=float))
        Ep = tEp[t]
        if not wp.isfinite(Ep[0, 0]):
            Ep = Eg
        dE = Eg - Ep
        trd = (dE[0, 0] + dE[1, 1] + dE[2, 2]) / 3.0
        dEd = dE - trd * wp.identity(n=3, dtype=float)
        tr_ = mat_trel[m]
        a = wp.exp(-dt / tr_)
        b = (1.0 - a) * tr_ / dt
        S0 = tSv[t]
        S1 = a * S0 + (2.0 * gv * b) * dEd
        P = P + F * S1
        wv = wp.ddot(S0, dE) - (wp.ddot(S1, S1) - wp.ddot(S0, S0)) / (4.0 * gv)
        wp.atomic_add(dissip, 0, wp.float64(V0[t] * tscale[t] * wv))
        tSv[t] = S1
        tEp[t] = Eg
    H = -(V0[t] * tscale[t]) * (P * wp.transpose(Dmi))
    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    wp.atomic_add(f, idx[1], f1)
    wp.atomic_add(f, idx[2], f2)
    wp.atomic_add(f, idx[3], f3)
    wp.atomic_sub(f, idx[0], f1 + f2 + f3)


@wp.func
def _eig_ub(S: wp.mat33):
    """Upper bound of a symmetric 3x3 matrix's largest eigenvalue: its largest absolute row sum."""
    r0 = wp.abs(S[0, 0]) + wp.abs(S[0, 1]) + wp.abs(S[0, 2])
    r1 = wp.abs(S[1, 0]) + wp.abs(S[1, 1]) + wp.abs(S[1, 2])
    r2 = wp.abs(S[2, 0]) + wp.abs(S[2, 1]) + wp.abs(S[2, 2])
    return wp.max(wp.max(r0, r1), r2)


@wp.kernel
def k_stiffness_bound(n_tets: int, tets: wp.array(dtype=wp.vec4i), active: wp.array(dtype=wp.int32),
                      x: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat33), V0: wp.array(dtype=float),
                      tscale: wp.array(dtype=float), tmat: wp.array(dtype=wp.int32), mat_mu: wp.array(dtype=float),
                      mat_lam: wp.array(dtype=float), mat_eta: wp.array(dtype=float), dt: float,
                      K: wp.array(dtype=float), C: wp.array(dtype=float), mat_gv: wp.array(dtype=float)):
    """Gershgorin bounds of each node's stiffness K (N/m) and damping C (N s/m) in the CURRENT configuration. A node
    displacement u_b changes F by u_b g_b^T (g: the shape functions' reference gradients), so the node blocks are
      stiffness  |K_ab| <= V0 (mu + lam' s(cof F)^2 + lam' |J - alpha| s(F)) |g_a| |g_b|
                 (the determinant's curvature block is the cross-product matrix of F (g_a x g_b)),
      damping    C_ab = V0 eta [(F g_b)(F g_a)^T + (g_a . g_b) F F^T]  (Kelvin-Voigt on the Green strain rate),
    s() the largest singular value, its square bounded by the largest absolute row sum of the Gram matrix.
    Symplectic Euler with explicit damping is stable for omega^2 dt^2 + 2 c dt < 4; step_tick keeps
    K dt^2 + 2 C dt <= 2.25 m at every node. The damping grows with stretch as the stiffness does; left
    out, a stretched flesh passes c dt = 2 (A1's control)."""
    t = wp.tid()
    if t >= n_tets or active[t] == 0:
        return
    idx = tets[t]
    x0 = x[idx[0]]
    Dmi = Dm_inv[t]
    F = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0) * Dmi
    m = tmat[t]
    mu = mat_mu[m]
    lamnh = mat_lam[m] + mu
    c0 = wp.cross(wp.vec3(F[0, 1], F[1, 1], F[2, 1]), wp.vec3(F[0, 2], F[1, 2], F[2, 2]))
    c1 = wp.cross(wp.vec3(F[0, 2], F[1, 2], F[2, 2]), wp.vec3(F[0, 0], F[1, 0], F[2, 0]))
    c2 = wp.cross(wp.vec3(F[0, 0], F[1, 0], F[2, 0]), wp.vec3(F[0, 1], F[1, 1], F[2, 1]))
    cof = wp.matrix_from_cols(c0, c1, c2)
    J = wp.determinant(F)
    alpha = 1.0 + mu / lamnh
    sF2 = _eig_ub(wp.transpose(F) * F)
    modl = mu + lamnh * (_eig_ub(wp.transpose(cof) * cof) + wp.abs(J - alpha) * wp.sqrt(sF2))
    modl = modl + 2.0 * mat_gv[m] * sF2              # the Maxwell branch's instantaneous stiffness (P = F S, S = 2 gv E)
    g1 = wp.vec3(Dmi[0, 0], Dmi[0, 1], Dmi[0, 2])
    g2 = wp.vec3(Dmi[1, 0], Dmi[1, 1], Dmi[1, 2])
    g3 = wp.vec3(Dmi[2, 0], Dmi[2, 1], Dmi[2, 2])
    g0 = -(g1 + g2 + g3)
    n0 = wp.length(g0)
    n1 = wp.length(g1)
    n2 = wp.length(g2)
    n3 = wp.length(g3)
    sg = n0 + n1 + n2 + n3
    vol = V0[t] * tscale[t]
    w = vol * modl * sg
    wp.atomic_add(K, idx[0], w * n0)
    wp.atomic_add(K, idx[1], w * n1)
    wp.atomic_add(K, idx[2], w * n2)
    wp.atomic_add(K, idx[3], w * n3)
    eta = mat_eta[m]
    if eta > 0.0:
        eta = eta * 3.0 / wp.max(3.0, wp.ddot(F, F))       # the stretch-normalised coefficient (k_tet_forces)
        h0 = wp.length(F * g0)
        h1 = wp.length(F * g1)
        h2 = wp.length(F * g2)
        h3 = wp.length(F * g3)
        sh = h0 + h1 + h2 + h3
        e = vol * eta
        wp.atomic_add(C, idx[0], e * (h0 * sh + sF2 * n0 * sg))
        wp.atomic_add(C, idx[1], e * (h1 * sh + sF2 * n1 * sg))
        wp.atomic_add(C, idx[2], e * (h2 * sh + sF2 * n2 * sg))
        wp.atomic_add(C, idx[3], e * (h3 * sh + sF2 * n3 * sg))


@wp.kernel
def k_tet_energy(n_tets: int, tets: wp.array(dtype=wp.vec4i), active: wp.array(dtype=wp.int32),
                 x: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat33), V0: wp.array(dtype=float),
                 tscale: wp.array(dtype=float),
                 tmat: wp.array(dtype=wp.int32), mat_mu: wp.array(dtype=float), mat_lam: wp.array(dtype=float),
                 out: wp.array(dtype=wp.float64), Jmin: wp.array(dtype=float), mat_gv: wp.array(dtype=float),
                 tSv: wp.array(dtype=wp.mat33)):
    t = wp.tid()
    if t >= n_tets or active[t] == 0:
        return
    idx = tets[t]
    x0 = x[idx[0]]
    F = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0) * Dm_inv[t]
    m = tmat[t]
    mu = mat_mu[m]
    lamnh = mat_lam[m] + mu
    alpha = 1.0 + mu / lamnh
    J = wp.determinant(F)
    Ic = wp.ddot(F, F)
    # energy consistent with P = mu F + lam'(J-alpha)cof F, shifted to zero at rest
    psi = 0.5 * mu * (Ic - 3.0) + 0.5 * lamnh * ((J - alpha) * (J - alpha) - (1.0 - alpha) * (1.0 - alpha))
    gv = mat_gv[m]
    if gv > 0.0:
        Sv = tSv[t]
        psi = psi + wp.ddot(Sv, Sv) / (4.0 * gv)            # the Maxwell branch's stored energy
    wp.atomic_add(out, 0, wp.float64(psi * V0[t] * tscale[t]))
    wp.atomic_min(Jmin, 0, J)


@wp.kernel
def k_tri_forces(n_tris: int, tris: wp.array(dtype=wp.vec3i), active: wp.array(dtype=wp.int32),
                 x: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
                 tri_mu: wp.array(dtype=float), tri_lam: wp.array(dtype=float), tri_eta: wp.array(dtype=float),
                 v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), dissip: wp.array(dtype=wp.float64),
                 dt: float):
    """Membrane: neo-Hookean-like on the 3x2 deformation gradient (mu, lam are thickness-integrated N/m)."""
    t = wp.tid()
    if t >= n_tris or active[t] == 0:
        return
    i = tris[t]
    x0 = x[i[0]]
    e1 = x[i[1]] - x0
    e2 = x[i[2]] - x0
    Dmi = Dm_inv[t]
    # F = [e1 e2] Dm_inv (3x2)
    Fa = e1 * Dmi[0, 0] + e2 * Dmi[1, 0]
    Fb = e1 * Dmi[0, 1] + e2 * Dmi[1, 1]
    mu = tri_mu[t]
    lam = tri_lam[t]
    # area ratio
    nrm = wp.cross(Fa, Fb)
    Ja = wp.length(nrm)
    Ja = wp.max(Ja, 1.0e-6)
    n = nrm / Ja
    # stable NH on the membrane: P = mu F + lam'(Ja - alpha) dJa/dF, dJa/dF = [Fb x n, n x Fa]
    lamnh = lam + mu
    alpha = 1.0 + mu / lamnh
    ca = wp.cross(Fb, n)
    cb = wp.cross(n, Fa)
    s = lamnh * (Ja - alpha)
    Pa = mu * Fa + s * ca
    Pb = mu * Fb + s * cb
    eta = tri_eta[t]
    if eta > 0.0:
        # Kelvin-Voigt damping on the membrane's Green strain rate (objective), as for the tets: S_v = eta mu dC/dt,
        # P_v = F S_v, power = A0 eta mu dC:dC / 2. (Damping dF/dt would resist rigid rotation: a spinning skin patch
        # would dissipate as much as a deforming one.)
        v0 = v[i[0]]
        d1 = v[i[1]] - v0
        d2 = v[i[2]] - v0
        Fda = d1 * Dmi[0, 0] + d2 * Dmi[1, 0]
        Fdb = d1 * Dmi[0, 1] + d2 * Dmi[1, 1]
        c00 = 2.0 * wp.dot(Fda, Fa)
        c11 = 2.0 * wp.dot(Fdb, Fb)
        c01 = wp.dot(Fda, Fb) + wp.dot(Fa, Fdb)
        em = eta * mu
        Pa = Pa + em * (Fa * c00 + Fb * c01)
        Pb = Pb + em * (Fa * c01 + Fb * c11)
        pw = A0[t] * em * 0.5 * (c00 * c00 + 2.0 * c01 * c01 + c11 * c11)
        wp.atomic_add(dissip, 0, wp.float64(pw * dt))
    a = A0[t]
    # forces on nodes 1,2: -A0 * P * Dm_inv^T
    f1 = -a * (Pa * Dmi[0, 0] + Pb * Dmi[0, 1])
    f2 = -a * (Pa * Dmi[1, 0] + Pb * Dmi[1, 1])
    wp.atomic_add(f, i[1], f1)
    wp.atomic_add(f, i[2], f2)
    wp.atomic_sub(f, i[0], f1 + f2)


@wp.kernel
def k_rod_forces(n_links: int, links: wp.array(dtype=wp.vec2i), lactive: wp.array(dtype=wp.int32),
                 L0: wp.array(dtype=float), k_ax: wp.array(dtype=float), c_ax: wp.array(dtype=float),
                 x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                 tension: wp.array(dtype=float)):
    l = wp.tid()
    if l >= n_links or lactive[l] == 0:
        return
    a = links[l][0]
    b = links[l][1]
    d = x[b] - x[a]
    ln = wp.length(d)
    if ln < 1.0e-9:
        return
    u = d / ln
    fs = k_ax[l] * (ln - L0[l]) + c_ax[l] * wp.dot(v[b] - v[a], u)
    tension[l] = k_ax[l] * (ln - L0[l])
    wp.atomic_add(f, a, fs * u)
    wp.atomic_sub(f, b, fs * u)


@wp.kernel
def k_rod_bend(n_hinges: int, hinges: wp.array(dtype=wp.vec3i), hactive: wp.array(dtype=wp.int32),
               k_b: wp.array(dtype=float), rest_ang: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
               f: wp.array(dtype=wp.vec3), moment: wp.array(dtype=float)):
    """Bending of a rod at node b between links (a,b) and (b,c): a spring on the difference of the
    turning vector from its rest value (small-angle discrete curvature). k_b = EI / l (N m / rad)."""
    h = wp.tid()
    if h >= n_hinges or hactive[h] == 0:
        return
    a = hinges[h][0]
    b = hinges[h][1]
    c = hinges[h][2]
    e0 = x[b] - x[a]
    e1 = x[c] - x[b]
    l0 = wp.length(e0)
    l1 = wp.length(e1)
    if l0 < 1.0e-9 or l1 < 1.0e-9:
        return
    t0 = e0 / l0
    t1 = e1 / l1
    kappa = wp.cross(t0, t1)
    dk = kappa - rest_ang[h]
    moment[h] = k_b[h] * wp.length(dk)
    # force pattern from the gradient of 0.5 k |kappa - kappa0|^2 (small-angle)
    g1 = k_b[h] * dk
    # d(t0 x t1)/dx: approximate with the perpendicular projections
    fa = wp.cross(t1, g1) / l0
    fc = wp.cross(t0, g1) / l1
    fa = fa - wp.dot(fa, t0) * t0
    fc = fc - wp.dot(fc, t1) * t1
    wp.atomic_add(f, a, fa)          # F = -dE/dx (conservative pattern; see bodies.k_rods_subcycle)
    wp.atomic_add(f, c, fc)
    wp.atomic_sub(f, b, fa + fc)


@wp.kernel
def k_ties(n_ties: int, tp: wp.array(dtype=wp.int32), ttet: wp.array(dtype=wp.int32), tbary: wp.array(dtype=wp.vec4),
           tets: wp.array(dtype=wp.vec4i), k: wp.array(dtype=float), fmax: wp.array(dtype=float),
           dfull: wp.array(dtype=float), damage: wp.array(dtype=float), x: wp.array(dtype=wp.vec3),
           v: wp.array(dtype=wp.vec3), kd: float, f: wp.array(dtype=wp.vec3), dissip: wp.array(dtype=wp.float64)):
    """Cohesive tie: bilinear traction-separation, irreversible damage; D=1 means broken."""
    i = wp.tid()
    if i >= n_ties or damage[i] >= 1.0 or ttet[i] < 0:
        return
    p = tp[i]
    tet = tets[ttet[i]]
    b = tbary[i]
    xa = b[0] * x[tet[0]] + b[1] * x[tet[1]] + b[2] * x[tet[2]] + b[3] * x[tet[3]]
    va = b[0] * v[tet[0]] + b[1] * v[tet[1]] + b[2] * v[tet[2]] + b[3] * v[tet[3]]
    d = x[p] - xa
    dl = wp.length(d)
    kk = k[i]
    d0 = fmax[i] / kk
    D = damage[i]
    if dl > d0:
        df = dfull[i]
        Dn = wp.clamp(df * (dl - d0) / (dl * (df - d0)), 0.0, 1.0)
        if Dn > D:
            # dissipated energy increment of the bilinear law (area under the envelope)
            Gtot = 0.5 * fmax[i] * df
            wp.atomic_add(dissip, 0, wp.float64(Gtot * (cohesive_spent(Dn, d0, df) - cohesive_spent(D, d0, df))))
            D = Dn
            damage[i] = D
    fs = (1.0 - D) * kk * d + (1.0 - D) * kd * (v[p] - va)
    wp.atomic_sub(f, p, fs)
    wp.atomic_add(f, tet[0], b[0] * fs)
    wp.atomic_add(f, tet[1], b[1] * fs)
    wp.atomic_add(f, tet[2], b[2] * fs)
    wp.atomic_add(f, tet[3], b[3] * fs)


@wp.kernel
def k_twins(n_tw: int, ta: wp.array(dtype=wp.int32), tb: wp.array(dtype=wp.int32), on: wp.array(dtype=wp.int32),
            k: wp.array(dtype=float), c: wp.array(dtype=float), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            f: wp.array(dtype=wp.vec3)):
    """Twin tie: the two copies of a node duplicated by a predictive split are held together (intact
    material) until the blade edge has physically passed them (k_twin_release)."""
    i = wp.tid()
    if i >= n_tw or on[i] == 0:
        return
    a = ta[i]
    b = tb[i]
    fs = k[i] * (x[b] - x[a]) + c[i] * (v[b] - v[a])
    wp.atomic_add(f, a, fs)
    wp.atomic_sub(f, b, fs)


@wp.kernel
def k_twin_release(n_tw: int, ta: wp.array(dtype=wp.int32), tb: wp.array(dtype=wp.int32), on: wp.array(dtype=wp.int32),
                   x: wp.array(dtype=wp.vec3), n_blades: int, b_p: wp.array(dtype=wp.vec3), b_q: wp.array(dtype=wp.quat),
                   b_h: wp.array(dtype=wp.vec3), margin: float, below: float, released: wp.array(dtype=wp.int32)):
    """Release a twin tie once its midpoint lies inside a blade segment's slab on the spine side of the
    edge (local z > -below), i.e. the edge has passed it."""
    i = wp.tid()
    if i >= n_tw or on[i] == 0:
        return
    m = 0.5 * (x[ta[i]] + x[tb[i]])
    for s in range(n_blades):
        X = wp.transform(b_p[s], b_q[s])
        pl = wp.transform_point(wp.transform_inverse(X), m)
        h = b_h[s]
        if wp.abs(pl[0]) <= h[0] and pl[2] >= -below and pl[2] <= 2.0 * h[2] and wp.abs(pl[1]) <= h[1] + margin:
            on[i] = 0
            wp.atomic_add(released, 0, 1)
            return


# ======================================================================================== rigid contact
@wp.kernel
def k_contact_candidates(n: int, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), kind: wp.array(dtype=wp.int32),
                         covered: wp.array(dtype=wp.int32), radius: wp.array(dtype=float), n_shapes: int,
                         s_type: wp.array(dtype=wp.int32), s_p0: wp.array(dtype=wp.vec3), s_rad: wp.array(dtype=float),
                         s_enabled: wp.array(dtype=wp.int32), s_body: wp.array(dtype=wp.int32),
                         s_v: wp.array(dtype=wp.vec3), s_w: wp.array(dtype=wp.vec3),
                         exempt: wp.array(dtype=wp.int32), margin: float, fast_margin: float, n_fast_bodies: int,
                         tick: float, cap: int,
                         cand: wp.array2d(dtype=wp.int32), cand_n: wp.array(dtype=wp.int32),
                         anchor_on: wp.array2d(dtype=wp.int32)):
    """Once per rigid tick: the shapes each particle can reach within the tick: bounding sphere + a reach margin that
    covers the particle's own travel and the shape's in one tick (`margin` + speed x tick for kinematic shapes; the
    knife's bodies, whose speed is not known here, get `fast_margin`). k_contacts then loops over these only; a
    particle with more than `cap` candidates falls back to every shape (cand_n = -1). Friction anchors of unreachable
    shapes are released here (k_contacts released them when it skipped a shape)."""
    i = wp.tid()
    if i >= n:
        return
    c = int(0)
    over = int(0)
    own = wp.length(v[i]) * tick
    for s in range(n_shapes):
        if s_enabled[s] == 0:
            continue
        sb = s_body[s]
        reach = margin + own
        if sb >= 0 and sb < n_fast_bodies:
            reach = fast_margin
        elif sb < 0:
            reach = reach + (wp.length(s_v[s]) + wp.length(s_w[s]) * s_rad[s]) * tick
        if wp.length(x[i] - s_p0[s]) > s_rad[s] + radius[i] + reach:
            anchor_on[i, s] = 0
            continue
        if c < cap:
            cand[i, c] = s
            c += 1
        else:
            over = 1
    if over == 1:
        cand_n[i] = -1
    else:
        cand_n[i] = c


@wp.kernel
def k_straddle(nt: int, tets: wp.array(dtype=wp.vec4i), active: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3),
               blade0: int, nseg: int, s_p0: wp.array(dtype=wp.vec3), s_q0: wp.array(dtype=wp.quat),
               s_h: wp.array(dtype=wp.vec3), s_enabled: wp.array(dtype=wp.int32), bstr: wp.array(dtype=wp.int32)):
    """Tets the blade passes through (their nodes on both sides of a blade segment's mid-plane, the tet over the
    segment): their nodes are marked, and their contact with the blade carries no friction (k_contacts). The crack
    follows whole-tet faces, up to h/2 from the plane the edge cut (and further where the work slid under the blade), so
    tets of one piece reach across the 1.2 mm blade; stretched over it, they pressed their nodes onto both flanks and
    clamped it, and the clamp's friction carried the piece up with the withdrawn knife (robot press cut: the half rose
    50 mm with the blade inside it; A2: the whole fish rose 130 mm). The flank's
    friction on a cut surface stays where the material presses on it, not where a tet the blade passes through does.
    (Leaving the far-side nodes out of the contact altogether released the clamp too, and closed the ledger with a
    kinematic blade, but a flexible blade then met the cut on one flank only, bent, and the robot's press cut created
    0.35 J and tore 1,750 faces; with the far-side nodes' friction alone removed, the near side's still lifted the
    half with 0.9 N.) Recomputed every tick, for the whole blade."""
    t = wp.tid()
    if t >= nt or active[t] == 0:
        return
    idx = tets[t]
    c = 0.25 * (x[idx[0]] + x[idx[1]] + x[idx[2]] + x[idx[3]])
    for k in range(nseg):
        sh = blade0 + k
        if s_enabled[sh] == 0:
            continue
        Xi = wp.transform_inverse(wp.transform(s_p0[sh], s_q0[sh]))
        pc = wp.transform_point(Xi, c)
        h = s_h[sh]
        if wp.abs(pc[0]) > h[0] + 0.003 or pc[2] < -0.001 or pc[2] > 2.0 * h[2] + 0.001 or wp.abs(pc[1]) > 0.006:
            continue
        ymin = float(1.0)
        ymax = float(-1.0)
        for j in range(4):
            yj = wp.transform_point(Xi, x[idx[j]])[1]
            ymin = wp.min(ymin, yj)
            ymax = wp.max(ymax, yj)
        if ymin < -1.0e-5 and ymax > 1.0e-5:
            for j in range(4):
                bstr[idx[j]] = 1


@wp.kernel
def k_contacts(n: int, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
               kind: wp.array(dtype=wp.int32), covered: wp.array(dtype=wp.int32), radius: wp.array(dtype=float),
               side: wp.array(dtype=wp.int32),
               n_shapes: int, s_type: wp.array(dtype=wp.int32), s_p0: wp.array(dtype=wp.vec3),
               s_q0: wp.array(dtype=wp.quat), s_v: wp.array(dtype=wp.vec3), s_w: wp.array(dtype=wp.vec3),
               s_h: wp.array(dtype=wp.vec3), s_param: wp.array(dtype=wp.vec2), s_com: wp.array(dtype=wp.vec3),
               s_mu: wp.array2d(dtype=float), s_enabled: wp.array(dtype=wp.int32),
               t_local: float, dt: float, dt_k: float, k_scale: float, zeta: float, m_cap: float,
               anchor: wp.array2d(dtype=wp.vec3), anchor_on: wp.array2d(dtype=wp.int32),
               f: wp.array(dtype=wp.vec3), wrench: wp.array2d(dtype=float), work_in: wp.array(dtype=wp.float64),
               fric_dissip: wp.array(dtype=wp.float64), cload: wp.array(dtype=float),
               damp_dissip: wp.array(dtype=wp.float64), s_body: wp.array(dtype=wp.int32),
               s_rad: wp.array(dtype=float), rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3),
               exempt: wp.array(dtype=wp.int32), cand: wp.array2d(dtype=wp.int32), cand_n: wp.array(dtype=wp.int32),
               bstr: wp.array(dtype=wp.int32), blade0: int, nseg: int):
    """Penalty contact of every particle with every rigid shape it can reach (k_contact_candidates), with stick-slip (anchored) Coulomb
    friction. Penalty stiffness is scaled to each particle's mass so the explicit step is stable
    (k = k_scale * m / dt^2); a critically-damped normal damper uses the same scale.
    Reactions are accumulated as wrenches on the shapes (about their COM)."""
    i = wp.tid()
    if i >= n:
        return
    if kind[i] == 0 and covered[i] != 0:
        return               # tissue surface covered by intact skin: the skin carries the contact
    if kind[i] == 3:
        return               # virtual node (outside its copy's material): no contact
    xi = x[i]
    m = wp.min(mass[i], m_cap)
    # stiffness from the NOMINAL step: the adaptive substep only gets smaller, so this stays stable, and a
    # penalty that changed with the substep would change the stored contact energy discontinuously
    kn = k_scale * m / (dt_k * dt_k)
    cn = 2.0 * zeta * wp.sqrt(kn * m)
    r = radius[i]
    kd = kind[i]
    nc = cand_n[i]
    count = nc
    if nc < 0:
        count = n_shapes
    for j in range(count):
        s = j
        if nc >= 0:
            s = cand[i, j]
        if s_enabled[s] == 0:
            continue
        bs = s_body[s]
        if kd == 2 and s_type[s] == 2:
            continue         # thin bones vs the skeleton's own bone capsules: attached / embedded, no contact
        if bs >= 0 and exempt[i] == bs:
            continue         # flesh that overlaps this bone at rest is bonded to it, not in contact with it
        tl = t_local
        if bs >= 0:
            tl = 0.0              # body-owned shapes are posed every substep (bodies.k_body_shapes)
        if wp.length(xi - (s_p0[s] + s_v[s] * tl)) > s_rad[s] + r + 0.02:
            anchor_on[i, s] = 0
            continue
        X = shape_pose_at(s_p0[s], s_q0[s], s_v[s], s_w[s], tl)
        pl = wp.transform_point(wp.transform_inverse(X), xi)
        st = s_type[s]
        h = s_h[s]
        pen = float(-1.0)
        nl = wp.vec3(0.0, 0.0, 0.0)
        if st == 0:
            dist, nn = box_sdf(pl, h)
            pen = r - dist
            nl = nn
        elif st == 1:
            hs = side[i]
            pen, nl = wedge_sdf(pl, h, s_param[s][0], s_param[s][1])
            if hs != 0:
                # crack-face node: inside the blade slab it is always pushed toward its own fragment's flank,
                # depth w - y_own (continuous through the mid-plane; a threshold on |y| made the force flip
                # sign inside the slab and pumped energy into oscillating nodes)
                pw, nw_ = wedge_sdf(wp.vec3(pl[0], 0.0, pl[2]), h, s_param[s][0], s_param[s][1])
                if pw > 0.0 and nw_[1] != 0.0:
                    y_own = float(hs) * pl[1]
                    if y_own < pw and y_own > -pw:
                        pen = pw - y_own
                        nl = wp.vec3(0.0, float(hs), 0.0)
                    else:
                        pen = -1.0
            if pen > 0.0:
                pen = pen + r
        else:
            pen, nl = capsule_sdf(pl, h)
            pen = pen + r
        if pen <= 0.0:
            anchor_on[i, s] = 0
            continue
        nw = wp.transform_vector(X, nl)
        com = wp.transform_point(X, s_com[s])
        vs = s_v[s] + wp.cross(s_w[s], xi - wp.transform_get_translation(X))
        vrel = v[i] - vs
        vn = wp.dot(vrel, nw)
        fn = kn * pen - cn * vn
        fn = wp.max(fn, 0.0)
        # separation-speed cap: the penalty does not push a node out of a shape faster than V_SEP relative to it. A node
        # found deep inside a shape (a crack-face copy split beside the blade, a node across the thin blade's mid-plane)
        # is expelled at up to pen*sqrt(k_scale)/dt ~ 8-15 m/s (ringing slivers, and pieces launched at 1.6-2 m/s).
        # Pressing contact (vn ~ 0) is unchanged; nothing is booked for the penetration energy the cap does not turn
        # into motion (a node placed inside by topology was paid by no work)
        capped = int(0)
        if mass[i] > 0.0:
            if vn + fn * dt / mass[i] > V_SEP:
                fn = wp.max(0.0, (V_SEP - vn) * mass[i] / dt)
                capped = 1
        # damper power (dissipated): only while the damper term acts (fn > 0; a capped force is a speed limit, not the
        # damper)
        if fn > 0.0 and capped == 0:
            wp.atomic_add(damp_dissip, 0, wp.float64(cn * vn * vn * dt))
        # stick-slip friction: tangential spring to an anchor stored in shape-local coordinates
        mu = s_mu[s, kd]
        if st == 1 and s >= blade0 and s < blade0 + nseg and bstr[i] != 0:
            # a node of a tet the blade passes through (k_straddle): its push on the flank is the stretched tet's, an
            # artefact of a crack that follows whole-tet faces, and it carries no friction (it clamped the blade, and the
            # clamp's friction lifted the piece with the knife)
            mu = 0.0
            anchor_on[i, s] = 0
        ft = wp.vec3(0.0, 0.0, 0.0)
        if mu > 0.0:
            if anchor_on[i, s] == 0:
                anchor[i, s] = pl
                anchor_on[i, s] = 1
            aw = wp.transform_point(X, anchor[i, s])
            dt_w = xi - aw
            dt_w = dt_w - wp.dot(dt_w, nw) * nw
            kt = 0.5 * kn
            ft = -kt * dt_w
            ftl = wp.length(ft)
            lim = mu * fn
            if ftl > lim:
                # slip: move the anchor so the spring sits on the Coulomb limit
                if ftl > 0.0:
                    ft = ft * (lim / ftl)
                newa = xi + ft / kt
                slip = wp.length(newa - aw)
                anchor[i, s] = wp.transform_point(wp.transform_inverse(X), newa)
                wp.atomic_add(fric_dissip, 0, wp.float64(lim * slip))
        fc = fn * nw + ft
        wp.atomic_add(f, i, fc)
        # reaction on the shape
        rr = xi - com
        tq = wp.cross(rr, -fc)
        if bs >= 0:
            # a bone of this solver: the reaction acts on the body (internal to the ledger)
            wp.atomic_add(rb_f, bs, -fc)
            wp.atomic_add(rb_t, bs, tq)
        else:
            wp.atomic_add(wrench, s, 0, -fc[0])
            wp.atomic_add(wrench, s, 1, -fc[1])
            wp.atomic_add(wrench, s, 2, -fc[2])
            wp.atomic_add(wrench, s, 3, tq[0])
            wp.atomic_add(wrench, s, 4, tq[1])
            wp.atomic_add(wrench, s, 5, tq[2])
            # power delivered by the shape into the particle system: F_shape_on_particle . v_shape_point
            wp.atomic_add(work_in, 0, wp.float64(wp.dot(fc, vs) * dt))
        wp.atomic_add(cload, s, fn)


# ======================================================================================== integration
@wp.kernel
def k_integrate(n: int, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                minv: wp.array(dtype=float), g: wp.vec3, dt: float, ext: wp.array(dtype=wp.vec3),
                kind: wp.array(dtype=wp.int32), gfrac: wp.array(dtype=float), xc: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if i >= n or kind[i] == 2:
        return               # rod nodes are integrated by their own projection (bodies.k_rods_pbd)
    if minv[i] <= 0.0:
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    # gravity acts on the node's physical mass only: the stability mass (recompute_mass) is inertia, not weight
    a = (f[i] + ext[i]) * minv[i] + g * gfrac[i]
    vi = v[i] + a * dt
    v[i] = vi
    # compensated (Kahan) position update: at bench height (z ~ 0.91 m) float32 positions are 6e-8 m apart, which is
    # 6 mm/s over a 1e-5 s substep, so without it a slowly moving node loses or gains whole spacings every substep (a
    # node at 2 mm/s does not move at all). xc carries the rounding of each update into the next
    dx = vi * dt - xc[i]
    xn = x[i] + dx
    xc[i] = (xn - x[i]) - dx
    x[i] = xn
    f[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_kinetic(n: int, v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), x: wp.array(dtype=wp.vec3),
              g: wp.vec3, ke: wp.array(dtype=wp.float64), pe: wp.array(dtype=wp.float64), gfrac: wp.array(dtype=float)):
    i = wp.tid()
    if i >= n:
        return
    m = mass[i]
    wp.atomic_add(ke, 0, wp.float64(0.5 * m * wp.dot(v[i], v[i])))          # inertia: the scaled mass
    wp.atomic_add(pe, 0, wp.float64(-m * gfrac[i] * wp.dot(g, x[i])))       # weight: the physical mass


# ======================================================================================== knife edge law
@wp.kernel
def k_edge_law(ns: int, s_pos: wp.array(dtype=wp.vec3), s_vel: wp.array(dtype=wp.vec3),
               s_t: wp.array(dtype=wp.vec3), s_n: wp.array(dtype=wp.vec3), s_ell: wp.array(dtype=float),
               s_thick: wp.array(dtype=float), s_shape: wp.array(dtype=wp.int32), s_on: wp.array(dtype=wp.int32),
               grid: wp.uint64, act_ids: wp.array(dtype=wp.int32), qr: float,
               tets: wp.array(dtype=wp.vec4i), tactive: wp.array(dtype=wp.int32),
               tmat: wp.array(dtype=wp.int32), mat_Rpar: wp.array(dtype=float), mat_Rperp: wp.array(dtype=float),
               sheet_n: wp.array(dtype=wp.vec3), V0: wp.array(dtype=float), Dm_inv: wp.array(dtype=wp.mat33),
               tplanes: wp.array2d(dtype=wp.vec4), tnplanes: wp.array(dtype=wp.int32), tanch: wp.array(dtype=wp.int32),
               x: wp.array(dtype=wp.vec3), X: wp.array(dtype=wp.vec3),
               v: wp.array(dtype=wp.vec3), minv: wp.array(dtype=float), ke: float, bsi: float, t_local: float, dt: float,
               search: int, crush_J: float, dt_k: float,
               a_on: wp.array(dtype=wp.int32), a_tet: wp.array(dtype=wp.int32), a_bary: wp.array(dtype=wp.vec4),
               a_init: wp.array(dtype=float), a_rest: wp.array(dtype=wp.vec3), a_cut: wp.array(dtype=float),
               a_block: wp.array(dtype=wp.int32), extra_R: wp.array(dtype=float),
               tmem_frac: wp.array(dtype=wp.vec4), tmem_R: wp.array(dtype=float),
               f: wp.array(dtype=wp.vec3), wrench: wp.array2d(dtype=float), s_com: wp.array(dtype=wp.vec3),
               cut_dissip: wp.array(dtype=wp.float64), work_in: wp.array(dtype=wp.float64),
               load_out: wp.array(dtype=wp.vec3), a_why: wp.array(dtype=wp.int32), a_epoch: wp.array(dtype=wp.int32),
               s_loc: wp.array(dtype=wp.vec3), sh_p0: wp.array(dtype=wp.vec3), sh_q0: wp.array(dtype=wp.quat),
               sh_v: wp.array(dtype=wp.vec3), sh_w: wp.array(dtype=wp.vec3), s_body: wp.array(dtype=wp.int32),
               rb_p: wp.array(dtype=wp.vec3), rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    if k >= ns:
        return
    load_out[k] = wp.vec3(0.0, 0.0, 0.0)
    if s_on[k] == 0:
        a_on[k] = 0
        return
    # the sample is a point of its blade segment (shape frame: x along the blade, z edge -> spine): its position,
    # velocity, tangent and cutting direction follow the segment at this substep (a blade segment may be a body of
    # this solver, posed every substep, or an external kinematic shape following its twist)
    sh = s_shape[k]
    tl = t_local
    if s_body[sh] >= 0:
        tl = 0.0
    Xs = shape_pose_at(sh_p0[sh], sh_q0[sh], sh_v[sh], sh_w[sh], tl)
    e = wp.transform_point(Xs, s_loc[k])
    ve = sh_v[sh] + wp.cross(sh_w[sh], e - wp.transform_get_translation(Xs))
    s_pos[k] = e
    s_vel[k] = ve
    s_t[k] = wp.transform_vector(Xs, wp.vec3(1.0, 0.0, 0.0))
    s_n[k] = -wp.transform_vector(Xs, wp.vec3(0.0, 0.0, 1.0))
    if a_block[k] != 0:
        # a bone lies in the path of this sample: the rod contact carries the load, no cutting here
        a_on[k] = 0
        return
    if a_on[k] == 0:
        if search == 0:
            return
        # the edge enters new material only while it advances into it (a knife being withdrawn does not
        # re-anchor in the tissue it passes on the way out; it did, and its live 'tips' blocked the last crack
        # faces from being cut: development probe)
        if wp.dot(ve, s_n[k]) <= 1.0e-4:
            return
        # enter intact tissue? find the tet containing the edge point (searched once per rigid tick)
        q = wp.hash_grid_query(grid, e, qr)
        j = int(0)
        found = int(-1)
        bb = wp.vec4(0.0)
        while wp.hash_grid_query_next(q, j):
            ti = act_ids[j]
            if tactive[ti] == 0 or tanch[ti] == 0:
                continue
            idx = tets[ti]
            x0 = x[idx[0]]
            M = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0)
            if wp.abs(wp.determinant(M * Dm_inv[ti])) < 1.0e-3:
                continue
            l = wp.inverse(M) * (e - x0)
            b0 = 1.0 - l[0] - l[1] - l[2]
            if b0 >= -1.0e-6 and l[0] >= -1.0e-6 and l[1] >= -1.0e-6 and l[2] >= -1.0e-6:
                bb4 = wp.vec4(b0, l[0], l[1], l[2])
                Xr = bb4[0] * X[idx[0]] + bb4[1] * X[idx[1]] + bb4[2] * X[idx[2]] + bb4[3] * X[idx[3]]
                if in_material(Xr, tplanes, tnplanes, ti):
                    found = ti
                    bb = bb4
        if found < 0:
            return
        a_on[k] = 1
        a_epoch[k] = a_epoch[k] + 1          # a new anchor (the crack record may not join it to the old one)
        a_tet[k] = found
        a_bary[k] = bb
        a_init[k] = 0.0
        idf = tets[found]
        a_rest[k] = bb[0] * X[idf[0]] + bb[1] * X[idf[1]] + bb[2] * X[idf[2]] + bb[3] * X[idf[3]]
    ti2 = a_tet[k]
    if ti2 < 0 or tactive[ti2] == 0 or tanch[ti2] == 0:
        a_on[k] = 0
        a_why[k] = 3
        return
    idx = tets[ti2]
    b = a_bary[k]
    if not (wp.isfinite(b[0]) and wp.isfinite(b[1]) and wp.isfinite(b[2]) and wp.isfinite(b[3])):
        a_on[k] = 0
        return
    xa = b[0] * x[idx[0]] + b[1] * x[idx[1]] + b[2] * x[idx[2]] + b[3] * x[idx[3]]
    va = b[0] * v[idx[0]] + b[1] * v[idx[1]] + b[2] * v[idx[2]] + b[3] * v[idx[3]]
    tt = s_t[k]
    nn = s_n[k]
    # tie stiffness limited by the effective mass of the material point (explicit stability)
    minv_eff = b[0] * b[0] * minv[idx[0]] + b[1] * b[1] * minv[idx[1]] + b[2] * b[2] * minv[idx[2]] + b[3] * b[3] * minv[idx[3]]
    kel = wp.min(ke, 0.2 / (minv_eff * dt_k * dt_k))
    d = e - xa
    dn = wp.dot(d, nn)
    dtg = wp.dot(d, tt)
    Gn = kel * dn
    Gt = kel * dtg
    if Gn <= 0.0:
        # the material point is momentarily ahead of the edge (tissue springing forward after a slip):
        # the edge cannot pull, so no force; release only if the edge has really withdrawn (> 0.5 mm)
        if dn < -5.0e-4:
            a_on[k] = 0
            a_why[k] = 1
        return
    # toughness at this material point: muscle anisotropy against the local sheet plane
    mt = tmat[ti2]
    rpar = mat_Rpar[mt]
    rperp = mat_Rperp[mt]
    ncr = wp.cross(tt, nn)            # cut-plane normal (the blade normal)
    c = wp.abs(wp.dot(ncr, sheet_n[ti2]))
    R = rpar * c * c + rperp * (1.0 - c * c)
    # a membrane (skin, silver skin) on a face of this tet: while the material point lies within the membrane's
    # thickness of that face (barycentric coordinate of the opposite vertex below thickness / height), the edge cuts
    # the membrane too and pays its toughness (energy R_m * thickness per unit edge length crossing it)
    fr = tmem_frac[ti2]
    inm = int(0)
    for jm in range(4):
        if fr[jm] > 0.0 and b[jm] < fr[jm]:
            inm = 1
    if inm == 1:
        R = R + tmem_R[ti2]
    ell = s_ell[k]
    Rl = R * ell + extra_R[k]
    # crushing: tissue under the edge compressed below J = 0.5 has failed; the edge passes it at reduced
    # resistance instead of inverting the element (the dissipation recorded is the reduced one)
    y0 = x[idx[0]]
    Jh = wp.determinant(wp.matrix_from_cols(x[idx[1]] - y0, x[idx[2]] - y0, x[idx[3]] - y0) * Dm_inv[ti2])
    if Jh < crush_J:
        Rl = Rl * wp.clamp((Jh - 0.5 * crush_J) / (0.5 * crush_J), 0.02, 1.0)
    # initiation: over the first blade-thickness of slip the locus is (1 + BSI) larger
    if a_init[k] < s_thick[k]:
        Rl = Rl * (1.0 + bsi)
    phi = Gn * Gn + Gt * Gt - Rl * Gn
    Gx = Gn * nn + Gt * tt
    if phi > 0.0:
        Gl = wp.sqrt(Gn * Gn + Gt * Gt)
        lam = Rl * Gn / Gl                    # radial return to the circle along G
        slip = (Gl - lam) / kel
        u = Gx / Gl
        Gx = u * lam
        # move the anchor toward the edge by `slip` along u (current space) and relocate it
        anew = xa + u * slip
        slip_n = slip * wp.dot(u, nn)
        # relocate: try the current tet first, else search
        x0 = x[idx[0]]
        M = wp.matrix_from_cols(x[idx[1]] - x0, x[idx[2]] - x0, x[idx[3]] - x0)
        detM = wp.determinant(M * Dm_inv[ti2])
        inside = int(0)
        if wp.abs(detM) > 1.0e-3:
            l = wp.inverse(M) * (anew - x0)
            b0 = 1.0 - l[0] - l[1] - l[2]
            if b0 >= -1.0e-6 and l[0] >= -1.0e-6 and l[1] >= -1.0e-6 and l[2] >= -1.0e-6:
                bq = wp.vec4(b0, l[0], l[1], l[2])
                Xq = bq[0] * X[idx[0]] + bq[1] * X[idx[1]] + bq[2] * X[idx[2]] + bq[3] * X[idx[3]]
                if in_material(Xq, tplanes, tnplanes, ti2):
                    a_bary[k] = bq
                    inside = 1
        if inside == 0:
            q = wp.hash_grid_query(grid, anew, qr)
            jj = int(0)
            found = int(-1)
            bb = wp.vec4(0.0)
            while wp.hash_grid_query_next(q, jj):
                tj = act_ids[jj]
                if tactive[tj] == 0:
                    continue
                id2 = tets[tj]
                y0 = x[id2[0]]
                M2 = wp.matrix_from_cols(x[id2[1]] - y0, x[id2[2]] - y0, x[id2[3]] - y0)
                if wp.abs(wp.determinant(M2 * Dm_inv[tj])) < 1.0e-3:
                    continue
                l2 = wp.inverse(M2) * (anew - y0)
                c0 = 1.0 - l2[0] - l2[1] - l2[2]
                if c0 >= -1.0e-6 and l2[0] >= -1.0e-6 and l2[1] >= -1.0e-6 and l2[2] >= -1.0e-6:
                    bc4 = wp.vec4(c0, l2[0], l2[1], l2[2])
                    Xc = bc4[0] * X[id2[0]] + bc4[1] * X[id2[1]] + bc4[2] * X[id2[2]] + bc4[3] * X[id2[3]]
                    if in_material(Xc, tplanes, tnplanes, tj):
                        found = tj
                        bb = bc4
            if found >= 0:
                a_tet[k] = found
                a_bary[k] = bb
                idx = tets[found]
            else:
                # the anchor left the tissue (cut through to a free surface): release
                a_on[k] = 0
                a_why[k] = 2
        bn = a_bary[k]
        a_rest[k] = bn[0] * X[idx[0]] + bn[1] * X[idx[1]] + bn[2] * X[idx[2]] + bn[3] * X[idx[3]]
        a_init[k] = a_init[k] + slip
        a_cut[k] = a_cut[k] + wp.max(slip_n, 0.0)
        wp.atomic_add(cut_dissip, 0, wp.float64(Rl * wp.max(slip_n, 0.0)))
    # G is the load of the edge on the material; the material pushes back on the knife with -G
    ftissue = Gx
    b = a_bary[k]
    idx = tets[a_tet[k]]
    wp.atomic_add(f, idx[0], b[0] * ftissue)
    wp.atomic_add(f, idx[1], b[1] * ftissue)
    wp.atomic_add(f, idx[2], b[2] * ftissue)
    wp.atomic_add(f, idx[3], b[3] * ftissue)
    s = s_shape[k]
    bs = s_body[s]
    if bs >= 0:
        # a blade segment body of this solver: the reaction acts on it (internal to the ledger)
        wp.atomic_sub(rb_f, bs, ftissue)
        wp.atomic_sub(rb_t, bs, wp.cross(e - rb_p[bs], ftissue))
    else:
        com = wp.transform_point(Xs, s_com[s])
        tq = wp.cross(e - com, -ftissue)
        wp.atomic_add(wrench, s, 0, -ftissue[0])
        wp.atomic_add(wrench, s, 1, -ftissue[1])
        wp.atomic_add(wrench, s, 2, -ftissue[2])
        wp.atomic_add(wrench, s, 3, tq[0])
        wp.atomic_add(wrench, s, 4, tq[1])
        wp.atomic_add(wrench, s, 5, tq[2])
        wp.atomic_add(work_in, 0, wp.float64(wp.dot(ftissue, ve) * dt))
    load_out[k] = Gx


@wp.kernel
def k_centroids(act_ids: wp.array(dtype=wp.int32), tets: wp.array(dtype=wp.vec4i), x: wp.array(dtype=wp.vec3),
                out: wp.array(dtype=wp.vec3)):
    j = wp.tid()
    idx = tets[act_ids[j]]
    out[j] = 0.25 * (x[idx[0]] + x[idx[1]] + x[idx[2]] + x[idx[3]])


def _arr(a, dtype, device):
    return wp.array(a, dtype=dtype, device=device)


class Materials:
    """Tissue material table (per tet material id)."""

    def __init__(self):
        self.names, self.E, self.nu, self.rho, self.tau, self.Rpar, self.Rperp = [], [], [], [], [], [], []
        self.sigma_t, self.sheets = [], []

    def add(self, name, E, nu, rho, tau_v, R_par, R_perp=None, sigma_t=float("inf"), sheets=False, visco=None):
        """sigma_t: tensile strength off the sheets (Pa; inf: the material never tears, tearing.py); sheets: fish
        muscle with myoseptal sheets (the sheet failure law applies on faces along them); visco: (g, tau) of the
        Maxwell branch (k_tet_forces), by default params.TISSUE_VISCO."""
        if visco is None:
            from . import params as PR
            vv = PR.v(PR.TISSUE_VISCO)
            visco = (vv["g"], vv["tau"])
        self.visco = getattr(self, "visco", [])
        self.visco.append((float(visco[0]), float(visco[1])))
        self.names.append(name)
        self.E.append(E)
        self.nu.append(nu)
        self.rho.append(rho)
        self.tau.append(tau_v)
        self.Rpar.append(R_par)
        self.Rperp.append(R_par if R_perp is None else R_perp)
        self.sigma_t.append(sigma_t)
        self.sheets.append(bool(sheets))
        return len(self.names) - 1

    def lame(self):
        E, nu = np.asarray(self.E), np.asarray(self.nu)
        mu = E/(2*(1 + nu))
        lam = E*nu/((1 + nu)*(1 - 2*nu))
        return mu, lam


class Tissue:
    """Host-side owner of the particle system. Arrays are allocated with spare capacity so topology
    changes (topology.py) never reallocate device buffers used by the stepping kernels."""

    def __init__(self, X, tets, tet_mat, materials, *, sheet_normal=None, device="cuda:0", dt=2e-5,
                 gravity=(0., 0., -9.81), node_capacity=None, tet_capacity=None, contact_k_scale=.15,
                 contact_zeta=.7):
        self.device = device
        self.dt = float(dt)
        self.g = wp.vec3(*gravity)
        self.mats = materials
        X = np.asarray(X, np.float64)
        tets = np.asarray(tets, np.int32)
        self.n = len(X)
        self.nt = len(tets)
        self.Nc = int(node_capacity or max(4*self.n, self.n + 20000))
        self.Tc = int(tet_capacity or max(4*self.nt, self.nt + 40000))
        self.k_scale, self.zeta = contact_k_scale, contact_zeta
        self.mass_floor_frac = .2
        self.sliver_cap = False
        self.adaptive = True
        self.max_substeps = 400
        # the ASSISTED rig's force-limited servo on servoed bodies (flag 3): translation and rotation stiffness, force
        # and torque caps (params RIG_*)
        from . import params as _PR
        self.rig_servo = (*_PR.v(_PR.RIG_STIFFNESS), _PR.v(_PR.RIG_FORCE_MAX), _PR.v(_PR.RIG_TORQUE_MAX))
        self.substep_capped = 0
        self.last_substeps = 0
        self.Kb = wp.zeros(self.Nc, dtype=float, device=device)
        self.Cb = wp.zeros(self.Nc, dtype=float, device=device)
        Nc, Tc = self.Nc, self.Tc
        # host mirrors (authoritative for topology), float64 rest geometry
        self.hX = np.zeros((Nc, 3)); self.hX[:self.n] = X
        self.hkind = np.zeros(Nc, np.int32)
        self.hradius = np.zeros(Nc, np.float32)
        self.htets = np.zeros((Tc, 4), np.int32); self.htets[:self.nt] = tets
        self.htmat = np.zeros(Tc, np.int32); self.htmat[:self.nt] = tet_mat
        self.hactive = np.zeros(Tc, np.int32); self.hactive[:self.nt] = 1
        self.hsheet = np.zeros((Tc, 3), np.float32)
        if sheet_normal is not None:
            self.hsheet[:self.nt] = sheet_normal
        self.hextra_mass = np.zeros(Nc)      # mass added by selective mass scaling (reported)
        self.point_mass = np.zeros(Nc)       # lumped mass of rod/skin nodes (not from tets)
        mu, lam = materials.lame()
        self.mat_mu = _arr(mu.astype(np.float32), float, device)
        self.mat_lam = _arr(lam.astype(np.float32), float, device)
        self.mat_eta = _arr((np.asarray(materials.tau)*mu).astype(np.float32), float, device)
        vis = np.asarray(getattr(materials, "visco", [(0., 1.)]*len(mu)), float).reshape(-1, 2)
        self.mat_gv = _arr((vis[:, 0]*mu).astype(np.float32), float, device)       # Maxwell branch shear modulus
        self.mat_trel = _arr(np.maximum(vis[:, 1], 1e-6).astype(np.float32), float, device)
        self.mat_Rpar = _arr(np.asarray(materials.Rpar, np.float32), float, device)
        self.mat_Rperp = _arr(np.asarray(materials.Rperp, np.float32), float, device)
        self.mat_rho = np.asarray(materials.rho)
        # device arrays
        self.x = wp.array(self.hX.astype(np.float32), dtype=wp.vec3, device=device)
        self.X = wp.array(self.hX.astype(np.float32), dtype=wp.vec3, device=device)
        self.v = wp.zeros(Nc, dtype=wp.vec3, device=device)
        self.xc = wp.zeros(Nc, dtype=wp.vec3, device=device)     # rounding carried by the compensated position update
        self.f = wp.zeros(Nc, dtype=wp.vec3, device=device)
        self.ext = wp.zeros(Nc, dtype=wp.vec3, device=device)
        self.mass = wp.zeros(Nc, dtype=float, device=device)
        self.minv = wp.zeros(Nc, dtype=float, device=device)
        self.gfrac = wp.ones(Nc, dtype=float, device=device)       # physical / scaled mass: the share that weighs
        self.kind = wp.zeros(Nc, dtype=wp.int32, device=device)
        self.covered = wp.zeros(Nc, dtype=wp.int32, device=device)
        self.radius = wp.zeros(Nc, dtype=float, device=device)
        # blade-side hint of crack-face nodes (+1/-1 along the blade normal b = t x n = blade local +y,
        # 0 = decide from position): a crack face traced by the anchors wanders ~0.3 mm about the blade
        # mid-plane, and deciding the wedge push from the current position shoved nodes of one fragment
        # into the other (development probes)
        self.hside = np.zeros(Nc, np.int32)
        self.hVfull = np.zeros(Tc)
        self.side = wp.zeros(Nc, dtype=wp.int32, device=device)
        self.exempt = wp.full(Nc, -1, dtype=wp.int32, device=device)     # per node: body it does not contact
        # per node: 1 = a node of a tet the blade passes through (k_straddle, per tick): frictionless on the blade
        self.blade0, self.nseg_blade = -1000, 0
        self.bstr = wp.zeros(Nc, dtype=wp.int32, device=device)
        self.tets = wp.zeros(Tc, dtype=wp.vec4i, device=device)
        self.tactive = wp.zeros(Tc, dtype=wp.int32, device=device)
        self.tmat = wp.zeros(Tc, dtype=wp.int32, device=device)
        self.Dm_inv = wp.zeros(Tc, dtype=wp.mat33, device=device)
        self.V0 = wp.zeros(Tc, dtype=float, device=device)
        self.tscale = wp.ones(Tc, dtype=float, device=device)
        # Maxwell branch state per tet: its overstress (2nd Piola-Kirchhoff) and the Green strain it last saw (NaN:
        # not yet seen; k_tet_forces starts the branch from the current strain)
        self.tSv = wp.zeros(Tc, dtype=wp.mat33, device=device)
        self.tEp = wp.array(np.full((Tc, 3, 3), np.nan, np.float32), dtype=wp.mat33, device=device)
        self.tanch = wp.ones(Tc, dtype=wp.int32, device=device)     # 0: debris fragment, cannot host an anchor
        # membrane on a face of the tet (add_membrane): per face j, membrane thickness / the tet's height over face j
        # (0: no membrane on it), and the membrane's cutting toughness (J/m^2)
        self.tmem_frac = wp.zeros(Tc, dtype=wp.vec4, device=device)
        self.tmem_R = wp.zeros(Tc, dtype=float, device=device)
        # virtual node algorithm: material volume fraction of each tet copy and its half-space constraints
        self.KP = 4
        self.hfrac = np.ones(Tc)
        self.htplanes = np.zeros((Tc, self.KP, 4), np.float32)
        self.htnplanes = np.zeros(Tc, np.int32)
        self.tplanes = wp.zeros((Tc, self.KP), dtype=wp.vec4, device=device)
        self.tnplanes = wp.zeros(Tc, dtype=wp.int32, device=device)
        self.sheet = wp.zeros(Tc, dtype=wp.vec3, device=device)
        self.tlo = wp.zeros(Tc, dtype=wp.vec3, device=device)
        self.thi = wp.zeros(Tc, dtype=wp.vec3, device=device)
        # membranes, rods, ties (empty until added)
        self.tris = None
        self.links = None
        self.hinges = None
        self.ties = None
        self.n_tris = self.n_links = self.n_hinges = self.n_ties = 0
        # shapes
        self.shapes = []
        self.S = MAX_SHAPES
        self.s_type = wp.zeros(self.S, dtype=wp.int32, device=device)
        self.s_p0 = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_q0 = wp.array(np.tile([0, 0, 0, 1], (self.S, 1)).astype(np.float32), dtype=wp.quat, device=device)
        self.s_v = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_w = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_h = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_param = wp.zeros(self.S, dtype=wp.vec2, device=device)
        self.s_com = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_mu = wp.zeros((self.S, 3), dtype=float, device=device)
        self.s_enabled = wp.zeros(self.S, dtype=wp.int32, device=device)
        self.s_body = wp.full(self.S, -1, dtype=wp.int32, device=device)
        self.s_loc_p = wp.zeros(self.S, dtype=wp.vec3, device=device)
        self.s_loc_q = wp.array(np.tile([0, 0, 0, 1], (self.S, 1)).astype(np.float32), dtype=wp.quat, device=device)
        self.s_rad = wp.zeros(self.S, dtype=float, device=device)
        self.nb = 0                       # rigid bones of this solver (bodies.py); set by add_bodies
        self.n_fast_bodies = 0            # the first n bodies move fast (the knife): wide contact-candidate margin
        self.rb = dict(f=wp.zeros(1, dtype=wp.vec3, device=device), t=wp.zeros(1, dtype=wp.vec3, device=device),
                       p=wp.zeros(1, dtype=wp.vec3, device=device),
                       q=wp.array(np.array([[0, 0, 0, 1.]], np.float32), dtype=wp.quat, device=device))
        self.n_smp = self.n_joints = self.n_bties = 0
        self.wrench = wp.zeros((self.S, 6), dtype=float, device=device)
        self.cload = wp.zeros(self.S, dtype=float, device=device)
        self.kload = wp.zeros(self.S, dtype=float, device=device)     # normal load of the knife's samples per shape
        self.anchor = wp.zeros((Nc, self.S), dtype=wp.vec3, device=device)
        self.anchor_on = wp.zeros((Nc, self.S), dtype=wp.int32, device=device)
        # per-tick contact candidate lists (k_contact_candidates); cand_n = -1 means "every shape"
        self.cand = wp.zeros((Nc, CAND_CAP), dtype=wp.int32, device=device)
        self.cand_n = wp.full(Nc, -1, dtype=wp.int32, device=device)
        # ledger (float64 accumulators on device)
        self.acc = {k: wp.zeros(1, dtype=wp.float64, device=device)
                    for k in ("visc", "fric", "cut", "tie", "damp", "work_in", "ke", "pe", "psi",
                              "fric_samples", "damp_samples", "damp_joints", "damp_bties", "damp_bonds", "iface",
                              "topo", "damp_rods", "damp_settle")}
        self.Jmin = wp.zeros(1, dtype=float, device=device)
        # edge law
        self.edge = None
        self.bvh = None
        self.time = 0.
        self.tick_dt = None
        self._graphs = {}
        self.upload_all()

    # ---------------------------------------------------------------- geometry upload
    def rest_geometry(self, tet_ids):
        """Dm_inv and rest volume of the given tets from the float64 host rest positions."""
        t = self.htets[tet_ids]
        X = self.hX
        Dm = np.stack([X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]], X[t[:, 3]] - X[t[:, 0]]], axis=2)
        det = np.linalg.det(Dm)
        vol = det/6.
        inv = np.zeros_like(Dm)
        ok = np.abs(det) > 1e-18
        inv[ok] = np.linalg.inv(Dm[ok])
        return inv, vol

    def upload_all(self):
        n, nt = self.n, self.nt
        ids = np.arange(nt)
        inv, vol = self.rest_geometry(ids)
        if (vol[self.hactive[:nt] > 0] <= 0).any():
            bad = int((vol[self.hactive[:nt] > 0] <= 0).sum())
            raise ValueError(f"{bad} inverted or degenerate rest tetrahedra")
        self.hVfull = np.zeros(self.Tc); self.hVfull[:nt] = vol
        self.hV0 = np.zeros(self.Tc); self.hV0[:nt] = vol*self.hfrac[:nt]
        self.hDminv = np.zeros((self.Tc, 3, 3)); self.hDminv[:nt] = inv
        self.tets.assign(self.htets)
        self.tactive.assign(self.hactive)
        self.tmat.assign(self.htmat)
        self.Dm_inv.assign(self.hDminv.astype(np.float32))
        self.V0.assign(self.hV0.astype(np.float32))
        self.sheet.assign(self.hsheet)
        self.tplanes.assign(self.htplanes); self.tnplanes.assign(self.htnplanes)
        self.X.assign(self.hX.astype(np.float32))
        self.kind.assign(self.hkind)
        self.radius.assign(self.hradius)
        self.recompute_mass()

    def recompute_mass(self, dt_crit_target=None, conserve_momentum=False):
        """Lumped masses from active tets (rho V / 4 per node) plus point masses of shells/rods.

        Selective NODAL mass scaling: a Gershgorin bound on the lumped stiffness row of each node,
        K_i = sum_e (lam+2mu)(1+2 xi) V |g_i| sum_j |g_j|  (g = shape-function gradients, xi the
        stiffness-proportional damping ratio at the element's top frequency), gives
        w_i^2 <= K_i / m_i. Where m_i < K_i (dt/1.5)^2 the node gets just enough extra mass; the added
        mass is recorded and reported. (A per-element estimate treated a sliver tet as if it carried
        its own tiny mass and added ~10x the body mass after cuts; slivers are attached to normal
        nodes and are stable.) When masses change, node momenta are conserved (v scaled by m/m')."""
        nt = self.nt
        act = self.hactive[:nt] > 0
        tt = self.htets[:nt][act]
        rho = self.mat_rho[self.htmat[:nt][act]]
        V = np.abs(self.hV0[:nt][act])                 # material (effective) volume
        # each tet keeps the mass it had when created: snapping a vertex onto a cut changes rest volumes
        # slightly but cannot create or destroy material
        if not hasattr(self, "htmass"):
            self.htmass = np.zeros(self.Tc)
        fresh = act & (self.htmass[:nt] == 0)
        self.htmass[:nt][fresh] = (self.mat_rho[self.htmat[:nt]]*np.abs(self.hV0[:nt]))[fresh]
        Vg = np.abs(self.hVfull[:nt][act])             # full geometric volume of the (copied) tet
        m = np.zeros(self.Nc)
        np.add.at(m, tt.ravel(), np.repeat(self.htmass[:nt][act]/4., 4))
        m += self.point_mass
        mem_k = None
        if getattr(self, "n_tris", 0):
            idx_now = self._membrane_nodes()
            np.add.at(m, idx_now.ravel(), self.mem_corner_m.ravel())
            mem_k = np.zeros(self.Nc)
            np.add.at(mem_k, idx_now.ravel(), self.mem_corner_k.ravel())
        base = m.copy()
        mu, lam = self.mats.lame()
        tm = self.htmat[:nt][act]
        mod = (lam + 2*mu)[tm]
        X = self.hX
        # shape-function gradient magnitudes |g_i| = area(opposite face) / (3 V)
        faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
        g = np.stack([.5*np.linalg.norm(np.cross(X[tt[:, b]] - X[tt[:, a]], X[tt[:, c]] - X[tt[:, a]]), axis=1)
                      for a, b, c in faces], 1)/(3*np.maximum(Vg, 1e-30))[:, None]
        hmin = 1./np.maximum(g.max(1), 1e-12)
        cc = np.sqrt(mod/rho)
        beta = 2*np.asarray(self.mats.tau)[tm]*mu[tm]/mod
        xi = beta*(2*cc/hmin)/2
        rows = (mod*(1 + 2*xi)*V)[:, None]*g*g.sum(1, keepdims=True)
        # sliver cap: no single tet may use more than half of any of its nodes' natural stability budget
        # (budget = m_base / (dt/1.6)^2). Offending tets are softened (their stiffness scaled down) rather
        # than loading kilograms of scaling mass on their nodes; count and volume are reported.
        budget = base[tt]/(self.dt/1.5)**2
        ratio = np.max(rows/np.maximum(.5*budget, 1e-30), 1)
        scale_e = np.where(ratio > 1., 1./ratio, 1.)
        if not self.sliver_cap:
            # virtual-node copies are full-shape tets: their small material fraction scales mass and stiffness
            # together, so there are no slivers to soften (capping softened the material next to every cut)
            scale_e = np.ones_like(scale_e)
        rows = rows*scale_e[:, None]
        ts_all = np.ones(self.Tc, np.float32)
        ts_all[np.nonzero(act)[0]] = scale_e
        self.tscale.assign(ts_all)
        self.softened = dict(count=int((scale_e < 1).sum()), volume_m3=float(V[scale_e < 1].sum()),
                             volume_fraction=float(V[scale_e < 1].sum()/max(V.sum(), 1e-30)))
        K = np.zeros(self.Nc)
        np.add.at(K, tt.ravel(), rows.ravel())
        if getattr(self, "h_btie_k", None) is not None:
            K += self.h_btie_k                     # flesh-bone bonds act on the same nodes
        if getattr(self, "h_tie_k", None) is not None:
            K += self.h_tie_k                      # rod-shaft bonds
        if mem_k is not None:
            K += mem_k                             # skin membrane (current nodes)
        if getattr(self, "h_bond_k", None) is not None:
            K += self.h_bond_k                     # interface bonds
        # tets may use omega^2 dt^2 <= 2.25; the remaining margin to the symplectic-Euler limit (4) is left
        # for twin ties (0.1), contacts (0.15) and the edge tie (0.2) that act on the same node
        need = K*(self.dt/1.5)**2
        m = np.maximum(m, need)
        # node-mass floor for tiny boundary corner nodes (contacts/ties act on them)
        tissue_nodes = np.zeros(self.Nc, bool); tissue_nodes[np.unique(tt)] = True
        if tissue_nodes.any():
            if not hasattr(self, "median_node_mass0"):
                self.median_node_mass0 = float(np.median(base[tissue_nodes]))      # of the uncut mesh
            floor = self.mass_floor_frac*self.median_node_mass0
            m[tissue_nodes] = np.maximum(m[tissue_nodes], floor)
        self.hextra_mass = m - base
        self.m_cap = float(4*self.median_node_mass0) if tissue_nodes.any() else 1.
        old = getattr(self, "hmass", None)
        self.hmass = m
        mm = m.astype(np.float32)
        if conserve_momentum and old is not None:
            v = self.v.numpy()
            ok = (old[:len(v)] > 0) & (mm > 0)
            scale = np.ones(len(v))
            scale[ok] = old[:len(v)][ok]/mm[ok]
            self.v.assign((v*scale[:, None]).astype(np.float32))
        self.mass.assign(mm)
        with np.errstate(divide="ignore"):
            self.minv.assign(np.where(mm > 0, 1./np.maximum(mm, 1e-30), 0.).astype(np.float32))
            # the stability mass is numerical inertia and carries no weight (a 290 g trout carries 27 g of it, which
            # would weigh 10 % too much on the board and the scale)
            self.gfrac.assign(np.where(m > 0, base/np.maximum(m, 1e-30), 1.).astype(np.float32))
        self.refresh_twin_stiffness()

    # ---------------------------------------------------------------- shapes
    def set_blade_shapes(self, blade0, nseg):
        """The blade's segment shapes (indices blade0 .. blade0 + nseg - 1) for k_straddle."""
        self.blade0, self.nseg_blade = int(blade0), int(nseg)
        self.invalidate_graphs()

    def set_shapes(self, shapes):
        """shapes: list of dicts with type (BOX/WEDGE/CAPSULE), half (3), param (2), mu (3: tissue,
        membrane, rod), com_local (3). Order is fixed for the life of the solver."""
        self._graphs = {}
        if len(shapes) > self.S:
            raise ValueError("too many rigid shapes")
        self.shapes = shapes
        t = np.zeros(self.S, np.int32); h = np.zeros((self.S, 3), np.float32)
        par = np.zeros((self.S, 2), np.float32); mu = np.zeros((self.S, 3), np.float32)
        com = np.zeros((self.S, 3), np.float32); en = np.zeros(self.S, np.int32)
        body = np.full(self.S, -1, np.int32); lp = np.zeros((self.S, 3), np.float32)
        lq = np.tile(np.array([0, 0, 0, 1], np.float32), (self.S, 1)); rad = np.zeros(self.S, np.float32)
        for i, s in enumerate(shapes):
            t[i] = s["type"]; h[i] = s["half"]; par[i] = s.get("param", (0., 0.)); mu[i] = s["mu"]
            com[i] = s.get("com_local", (0., 0., 0.)); en[i] = 1 if s.get("enabled", True) else 0
            body[i] = s.get("body", -1)
            lp[i] = s.get("loc_p", (0., 0., 0.)); lq[i] = s.get("loc_q", (0., 0., 0., 1.))
            hh = np.asarray(s["half"], float)
            rad[i] = (np.linalg.norm([hh[0], hh[1], 2*hh[2]]) if s["type"] == WEDGE else
                      hh[0] + hh[1] if s["type"] == CAPSULE else np.linalg.norm(hh))
        self.s_type.assign(t); self.s_h.assign(h); self.s_param.assign(par); self.s_mu.assign(mu)
        self.s_com.assign(com); self.s_enabled.assign(en)
        self.s_body.assign(body); self.s_loc_p.assign(lp); self.s_loc_q.assign(lq); self.s_rad.assign(rad)
        self.h_s_body = body

    def set_shape_states(self, poses, twists):
        """poses: (S,7) px py pz qx qy qz qw at the tick start; twists: (S,6) v, w (world)."""
        k = len(poses)
        p0 = self.s_p0.numpy(); q0 = self.s_q0.numpy(); v = self.s_v.numpy(); w = self.s_w.numpy()
        ext = self.h_s_body[:k] < 0 if hasattr(self, "h_s_body") else np.ones(k, bool)
        p0[:k][ext] = np.asarray(poses)[:, :3][ext]; q0[:k][ext] = np.asarray(poses)[:, 3:7][ext]
        v[:k][ext] = np.asarray(twists)[:, :3][ext]; w[:k][ext] = np.asarray(twists)[:, 3:][ext]
        self.s_p0.assign(p0); self.s_q0.assign(q0); self.s_v.assign(v); self.s_w.assign(w)
        self._pose_now = (p0[:len(self.shapes)].copy(), q0[:len(self.shapes)].copy(),
                          v[:len(self.shapes)].copy(), w[:len(self.shapes)].copy())

    def enable_shape(self, i, on=True):
        en = self.s_enabled.numpy(); en[i] = 1 if on else 0; self.s_enabled.assign(en)

    # ---------------------------------------------------------------- edge samples
    def set_edge_samples(self, n_samples):
        self._graphs = {}
        d = self.device
        self.edge = dict(n=int(n_samples),
                         pos=wp.zeros(n_samples, dtype=wp.vec3, device=d), vel=wp.zeros(n_samples, dtype=wp.vec3, device=d),
                         t=wp.zeros(n_samples, dtype=wp.vec3, device=d), nn=wp.zeros(n_samples, dtype=wp.vec3, device=d),
                         ell=wp.zeros(n_samples, dtype=float, device=d), thick=wp.zeros(n_samples, dtype=float, device=d),
                         shape=wp.zeros(n_samples, dtype=wp.int32, device=d), on=wp.zeros(n_samples, dtype=wp.int32, device=d),
                         a_on=wp.zeros(n_samples, dtype=wp.int32, device=d), a_tet=wp.full(n_samples, -1, dtype=wp.int32, device=d),
                         a_bary=wp.zeros(n_samples, dtype=wp.vec4, device=d), a_init=wp.zeros(n_samples, dtype=float, device=d),
                         a_rest=wp.zeros(n_samples, dtype=wp.vec3, device=d), a_cut=wp.zeros(n_samples, dtype=float, device=d),
                         a_block=wp.zeros(n_samples, dtype=wp.int32, device=d), extra_R=wp.zeros(n_samples, dtype=float, device=d),
                         load=wp.zeros(n_samples, dtype=wp.vec3, device=d), a_why=wp.zeros(n_samples, dtype=wp.int32, device=d),
                         a_epoch=wp.zeros(n_samples, dtype=wp.int32, device=d),
                         loc=wp.zeros(n_samples, dtype=wp.vec3, device=d))
        self.edge_ke = 5e3            # N/m per sample (stable against 1 mm-sample node masses at dt)
        self.crush_J = .5             # volume ratio below which the tissue under the edge is crushed [estimate]
        self.bsi = .22

    def set_edge_layout(self, shape, loc, ell, thick, on=None, extra_R=None):
        """Edge samples as points of blade shapes: `shape` (index of each sample's blade shape), `loc` (its point
        in that shape's frame), `ell` (edge length it stands for), `thick` (blade thickness there). Positions,
        velocities, tangents and cutting directions are computed every substep from the shape poses."""
        e = self.edge
        e["shape"].assign(np.asarray(shape, np.int32)); e["loc"].assign(np.asarray(loc, np.float32))
        e["ell"].assign(np.asarray(ell, np.float32)); e["thick"].assign(np.asarray(thick, np.float32))
        e["on"].assign(np.ones(e["n"], np.int32) if on is None else np.asarray(on, np.int32))
        if extra_R is not None:
            e["extra_R"].assign(np.asarray(extra_R, np.float32))

    def rb_p_arr(self):
        return self.rb["p"] if self.nb else self.rb.setdefault("p", wp.zeros(1, dtype=wp.vec3, device=self.device))

    def rb_pc_arr(self):
        return self.rb["pc"] if self.nb else self.rb.setdefault("pc", wp.zeros(1, dtype=wp.vec3, device=self.device))

    def set_edge_state(self, pos, vel, tangent, normal, ell, thick, shape, on, extra_R=None, block=None):
        e = self.edge
        e["pos"].assign(np.asarray(pos, np.float32)); e["vel"].assign(np.asarray(vel, np.float32))
        e["t"].assign(np.asarray(tangent, np.float32)); e["nn"].assign(np.asarray(normal, np.float32))
        e["ell"].assign(np.asarray(ell, np.float32)); e["thick"].assign(np.asarray(thick, np.float32))
        e["shape"].assign(np.asarray(shape, np.int32)); e["on"].assign(np.asarray(on, np.int32))
        if extra_R is not None:
            e["extra_R"].assign(np.asarray(extra_R, np.float32))
        if block is not None:
            e["a_block"].assign(np.asarray(block, np.int32))

    def refresh_active(self):
        """Compact list of active tets and the query radius (largest centroid-to-vertex distance
        in the current configuration, plus a margin). Call after topology changes."""
        self._graphs = {}
        ids = np.nonzero(self.hactive[:self.nt])[0].astype(np.int32)
        self.act_ids = wp.array(ids, dtype=wp.int32, device=self.device)
        self.cent = wp.zeros(len(ids), dtype=wp.vec3, device=self.device)
        x = self.x.numpy()[:self.n]
        t = self.htets[ids]
        c = x[t].mean(1)
        r = np.linalg.norm(x[t] - c[:, None], axis=2).max()
        self.query_r = float(r + 1.5e-3)
        if not hasattr(self, "grid"):
            self.grid = wp.HashGrid(128, 128, 128, device=self.device)

    def refit_bvh(self, margin=None):
        """(Name kept) Rebuild the hash grid over active tet centroids at the current positions."""
        if not hasattr(self, "act_ids"):
            self.refresh_active()
        wp.launch(k_centroids, dim=len(self.act_ids), inputs=[self.act_ids, self.tets, self.x, self.cent],
                  device=self.device)
        self.grid.build(self.cent, self.query_r)

    def rebuild_bvh(self):
        self.refresh_active()
        self.refit_bvh()

    # ---------------------------------------------------------------- stepping
    def step_tick(self, tick_dt, substeps=None):
        """Advance by one rigid tick (kinematic shapes follow their twist within it)."""
        dt = self.dt
        n_sub = substeps or max(1, int(round(tick_dt/dt)))
        if substeps is None and self.adaptive:
            # current-configuration stability bound: symplectic Euler with explicit damping needs
            # omega^2 dt^2 + 2 c dt < 4 (omega^2 <= K_i/m_i, c <= C_i/m_i); tets get K dt^2 + 2 C dt <= 2.25 m (the rest
            # of the margin is for contacts/ties): the largest such dt per node is 2.25 m / (C + sqrt(C^2 + 2.25 K m))
            self.Kb.zero_()
            self.Cb.zero_()
            wp.launch(k_stiffness_bound, dim=self.Tc, inputs=[self.nt, self.tets, self.tactive, self.x, self.Dm_inv,
                                                              self.V0, self.tscale, self.tmat, self.mat_mu, self.mat_lam,
                                                              self.mat_eta, dt, self.Kb, self.Cb, self.mat_gv],
                      device=self.device)
            Kb = self.Kb.numpy()[:self.n].astype(float)
            Cb = self.Cb.numpy()[:self.n].astype(float)
            m = self.hmass[:self.n]
            ok = (m > 0) & ((Kb > 0) | (Cb > 0))
            dt_ok = float(np.min(2.25*m[ok]/(Cb[ok] + np.sqrt(Cb[ok]**2 + 2.25*Kb[ok]*m[ok])))) if ok.any() else dt
            n_need = int(np.ceil(tick_dt/min(dt, dt_ok)))
            self.last_n_need = n_need
            if n_need > self.max_substeps:
                self.substep_capped += 1
            n_sub = max(n_sub, min(n_need, self.max_substeps))
            self.last_substeps = n_sub
        dt = tick_dt/n_sub
        self.wrench.zero_()
        self.cload.zero_()
        self.kload.zero_()
        if self.shapes and self.n_smp:
            from .bodies import k_sample_candidates
            sm, rb = self.smp, self.rb
            wp.launch(k_sample_candidates, dim=self.n_smp, inputs=[self.n_smp, sm["body"], sm["loc"], sm["r"], rb["p"],
                                                                   rb["q"], len(self.shapes), self.s_body, self.s_p0,
                                                                   self.s_rad, self.s_enabled, rb["group"],
                                                                   CAND_MARGIN_M, CAND_CAP, sm["cand"], sm["cand_n"],
                                                                   sm["anchor_on"]],
                      device=self.device)
        if self.shapes:
            wp.launch(k_contact_candidates, dim=self.n, inputs=[self.n, self.x, self.v, self.kind, self.covered,
                                                                self.radius, len(self.shapes), self.s_type, self.s_p0,
                                                                self.s_rad, self.s_enabled, self.s_body, self.s_v, self.s_w,
                                                                self.exempt, CAND_SLOW_MARGIN_M, CAND_MARGIN_M,
                                                                int(self.n_fast_bodies), float(tick_dt), CAND_CAP,
                                                                self.cand, self.cand_n, self.anchor_on], device=self.device)
        if self.nb:
            self.rb["fsum"].zero_(); self.rb["tsum"].zero_()
        dev = self.device
        if self.nseg_blade:
            self.bstr.zero_()
            wp.launch(k_straddle, dim=self.nt, inputs=[self.nt, self.tets, self.tactive, self.x, int(self.blade0),
                                                     int(self.nseg_blade), self.s_p0, self.s_q0, self.s_h,
                                                     self.s_enabled, self.bstr], device=dev)
        if self.edge is not None:
            self.refit_bvh()
        # the substep loop is captured once per (n_sub, dt) as a CUDA graph and replayed (13 kernels x 100 substeps are
        # ~1300 launches per tick; under CPU contention the launch overhead dominated). Captures are invalidated when
        # any array they reference is rebuilt (topology commits, setup) - see invalidate_graphs()
        use_graph = self.use_graphs and not self.debug_nan and str(dev).startswith("cuda")
        if use_graph:
            key = (n_sub, float(dt))
            g = self._graphs.get(key)
            if g is None:
                wp.capture_begin(device=dev)
                try:
                    self._substep_loop(n_sub, dt)
                finally:
                    g = wp.capture_end(device=dev)
                self._graphs[key] = g
            wp.capture_launch(g)
        else:
            self._substep_loop(n_sub, dt)
        self.time += tick_dt
        self.tick_dt = tick_dt
        self.cload_last = self.cload.numpy()[:len(self.shapes)]/n_sub
        self.kload_last = self.kload.numpy()[:len(self.shapes)]/n_sub
        return self.wrench.numpy()[:len(self.shapes)]/n_sub

    use_graphs = True

    def invalidate_graphs(self):
        self._graphs = {}

    def _substep_loop(self, n_sub, dt):
        dev = self.device
        for s in range(n_sub):
            tl = s*dt
            if self.nb:
                self._bodies_pre(dt, tl)
            wp.launch(k_tet_forces, dim=self.Tc, inputs=[self.nt, self.tets, self.tactive, self.x, self.v, self.Dm_inv,
                                                          self.V0, self.tscale, self.tmat, self.mat_mu, self.mat_lam, self.mat_eta,
                                                          self.f, self.acc["visc"], dt, self.mat_gv, self.mat_trel,
                                                          self.tSv, self.tEp], device=dev)
            if self.debug_nan:
                self._nan_check(f"substep {s} after tets")
            if self.n_tris:
                tr = self.tris
                wp.launch(k_tri_forces, dim=len(tr["idx"]), inputs=[self.n_tris, tr["idx"], tr["active"], self.x, tr["Dm_inv"],
                                                                   tr["A0"], tr["mu"], tr["lam"], tr["eta"], self.v, self.f,
                                                                   self.acc["visc"], dt], device=dev)
            if self.n_links:
                r = self.links
                wp.launch(k_rod_forces, dim=len(r["idx"]), inputs=[self.n_links, r["idx"], r["active"], r["L0"], r["k"], r["c"],
                                                                  self.x, self.v, self.f, r["tension"]], device=dev)
            if self.n_hinges:
                hg = self.hinges
                wp.launch(k_rod_bend, dim=len(hg["idx"]), inputs=[self.n_hinges, hg["idx"], hg["active"], hg["k"], hg["rest"],
                                                                 self.x, self.f, hg["moment"]], device=dev)
            if self.n_ties:
                ti = self.ties
                wp.launch(k_ties, dim=len(ti["p"]), inputs=[self.n_ties, ti["p"], ti["tet"], ti["bary"], self.tets, ti["k"],
                                                           ti["fmax"], ti["dfull"], ti["damage"], self.x, self.v, self.tie_kd,
                                                           self.f, self.acc["tie"]], device=dev)
            if getattr(self, "n_bonds", 0):
                from .bodies import k_bonds
                bd = self.bonds
                wp.launch(k_bonds, dim=self.n_bonds, inputs=[self.n_bonds, bd["a"], bd["b"], bd["k"], bd["fmax"], bd["dfull"],
                                                            bd["damage"], bd["kd"], self.x, self.v, self.f, dt,
                                                            self.acc["tie"], self.acc["damp_bonds"]], device=dev)
            if getattr(self, "n_twins", 0):
                tw = self.twins
                wp.launch(k_twins, dim=tw["cap"], inputs=[self.n_twins, tw["a"], tw["b"], tw["on"], tw["k"], tw["c"],
                                                         self.x, self.v, self.f], device=dev)
                if self.debug_nan:
                    self._nan_check(f"substep {s} after twins")
            if self.shapes:
                wp.launch(k_contacts, dim=self.n, inputs=[self.n, self.x, self.v, self.mass, self.kind, self.covered,
                                                          self.radius, self.side, len(self.shapes), self.s_type, self.s_p0, self.s_q0,
                                                          self.s_v, self.s_w, self.s_h, self.s_param, self.s_com, self.s_mu,
                                                          self.s_enabled, tl, dt, self.dt, self.k_scale, self.zeta, self.m_cap, self.anchor,
                                                          self.anchor_on, self.f, self.wrench, self.acc["work_in"],
                                                          self.acc["fric"], self.cload, self.acc["damp"], self.s_body,
                                                          self.s_rad, self.rb["f"], self.rb["t"], self.exempt,
                                                          self.cand, self.cand_n, self.bstr, int(self.blade0),
                                                          int(self.nseg_blade)], device=dev)
            if self.nb:
                self._bodies_forces(tl, dt)
            if self.debug_nan:
                self._nan_check(f"substep {s} after contacts")
            if self.edge is not None:
                e = self.edge
                wp.launch(k_edge_law, dim=e["n"], inputs=[e["n"], e["pos"], e["vel"], e["t"], e["nn"], e["ell"], e["thick"],
                                                         e["shape"], e["on"], self.grid.id, self.act_ids, self.query_r,
                                                         self.tets, self.tactive, self.tmat,
                                                         self.mat_Rpar, self.mat_Rperp, self.sheet, self.V0, self.Dm_inv,
                                                         self.tplanes, self.tnplanes, self.tanch, self.x, self.X, self.v,
                                                         self.minv, self.edge_ke, self.bsi, tl, dt, 1 if s == 0 else 0,
                                                         self.crush_J, self.dt,
                                                         e["a_on"], e["a_tet"], e["a_bary"],
                                                         e["a_init"], e["a_rest"], e["a_cut"], e["a_block"], e["extra_R"],
                                                         self.tmem_frac, self.tmem_R,
                                                         self.f, self.wrench, self.s_com, self.acc["cut"], self.acc["work_in"],
                                                         e["load"], e["a_why"], e["a_epoch"], e["loc"], self.s_p0, self.s_q0,
                                                         self.s_v, self.s_w, self.s_body, self.rb_p_arr(), self.rb["f"],
                                                         self.rb["t"]], device=dev)
            if self.debug_nan:
                self._nan_check(f"substep {s} before integrate")
            if self.debug_nan and self.edge is not None:
                self._nan_check(f"substep {s} after edge law")
            if getattr(self, "n_rods", 0):
                from .bodies import k_rods_pbd
                r = self.rods
                rb = self.rb
                wp.launch(k_rods_pbd, dim=self.n_rods, inputs=[self.n_rods, r["start"], r["len"], self.x, self.v, self.f,
                                                               self.minv, self.g, self.gfrac, r["l0"], r["lon"], r["kb"], r["brest"],
                                                               r["bon"], r["tie_tet"], r["tie_bary"], r["tie_k"],
                                                               r["tie_fmax"], r["tie_dfull"], r["tie_dmg"], r["tie_c"], self.tets,
                                                               r["root_body"], r["root_loc"], r["root_fmax"], r["root_on"],
                                                               self.rb_p_arr(), rb["q"], rb["f"], rb["t"], r["fr"], dt,
                                                               self.rod_iters, r["tension"], r["moment"], self.acc["tie"],
                                                               r["dx"], self.xc, self.acc["damp_rods"], self.X, self.Dm_inv,
                                                               r["tie_sum"], self.rb_pc_arr()],
                          device=dev)
            wp.launch(k_integrate, dim=self.n, inputs=[self.n, self.x, self.v, self.f, self.minv, self.g, dt, self.ext,
                                                       self.kind, self.gfrac, self.xc], device=dev)
            if self.nb:
                self._bodies_integrate(dt)

    # ---------------------------------------------------------------- rigid bones (bodies.py)
    def add_bodies(self, bodies, joints=(), sample_k_line=5e6, sample_zeta=1.):
        """bodies: list of dict(name, mass, com (3, world), inertia (3x3 world, about com), quat (4, xyzw,
        default identity: the body frame is world-aligned at rest), fixed (bool), samples: (k,3) world points
        with radii (k,), spacing (m) for the contact stiffness). Shapes owned by a body are declared in
        set_shapes with body=index and loc_p/loc_q in the body frame.
        joints: list of dict(a, b, anchor (3, world), k, c, k_ang, c_ang).
        Sample contact stiffness per unit length of bone `sample_k_line` (N/m per m): a flat blade on ten
        spines sees ~5e4 N/m, inside the explicit MuJoCo coupling budget of a 0.15 kg knife at 1 ms."""
        self._graphs = {}
        from .bodies import quat_inv, quat_mul, quat_rot
        d = self.device
        nb = len(bodies)
        q = np.array([b_.get("quat", (0., 0., 0., 1.)) for b_ in bodies], np.float32)
        p = np.array([b_["com"] for b_ in bodies], np.float32)
        m = np.array([b_["mass"] for b_ in bodies], np.float32)
        Ib = []
        for b_, qq in zip(bodies, q):
            R = np.array(quat_rot(qq, np.eye(3)[0])), np.array(quat_rot(qq, np.eye(3)[1])), np.array(quat_rot(qq, np.eye(3)[2]))
            R = np.stack(R, 1)
            Ib.append(R.T @ np.asarray(b_["inertia"], float) @ R)
        Ib = np.array(Ib, np.float32)
        self.rb = dict(p0=wp.array(p, dtype=wp.vec3, device=d), q0=wp.array(q, dtype=wp.quat, device=d),
                       fsum=wp.zeros(nb, dtype=wp.vec3, device=d), tsum=wp.zeros(nb, dtype=wp.vec3, device=d),
                       debt=wp.zeros(nb, dtype=float, device=d),
                       knife_held=wp.zeros(1, dtype=wp.int32, device=d),
                       p=wp.array(p, dtype=wp.vec3, device=d), q=wp.array(q, dtype=wp.quat, device=d),
                       v=wp.zeros(nb, dtype=wp.vec3, device=d), w=wp.zeros(nb, dtype=wp.vec3, device=d),
                       m=wp.array(m, dtype=float, device=d), Ib=wp.array(Ib, dtype=wp.mat33, device=d),
                       Ibinv=wp.array(np.linalg.inv(Ib).astype(np.float32), dtype=wp.mat33, device=d),
                       f=wp.zeros(nb, dtype=wp.vec3, device=d), t=wp.zeros(nb, dtype=wp.vec3, device=d),
                       fixed=wp.array(np.array([2 if b_.get("driven") else (1 if b_.get("fixed") else 0) for b_ in bodies],
                                               np.int32), dtype=wp.int32, device=d),
                       # a servoed body's target twist over the tick and the servo's own force and torque
                       vt=wp.zeros(nb, dtype=wp.vec3, device=d), wt=wp.zeros(nb, dtype=wp.vec3, device=d),
                       fserv=wp.zeros(nb, dtype=wp.vec3, device=d), tserv=wp.zeros(nb, dtype=wp.vec3, device=d),
                       # compensated-summation residuals of the position and orientation updates (bodies.k_body_integrate)
                       pc=wp.zeros(nb, dtype=wp.vec3, device=d), qc=wp.zeros(nb, dtype=wp.vec4, device=d))
        self.h_rb_fixed = np.array([2 if b_.get("driven") else (1 if b_.get("fixed") else 0) for b_ in bodies])
        # collision groups: bodies of one assembly (knife parts; the skeleton) never collide with each other
        self.rb["group"] = wp.array(np.array([int(b_.get("group", 0)) for b_ in bodies], np.int32), dtype=wp.int32,
                                    device=d)
        self.body_names = [b_["name"] for b_ in bodies]
        self.nb = nb
        # samples in body frames
        sb, sl, sr, sk, sc = [], [], [], [], []
        for i, b_ in enumerate(bodies):
            P = np.asarray(b_.get("samples", np.zeros((0, 3))), float)
            r = np.asarray(b_.get("sample_r", np.zeros(len(P))), float)
            sp = np.asarray(b_.get("sample_spacing", np.full(len(P), 1e-3)), float)
            for pt, rr, ss in zip(P, r, sp):
                sb.append(i); sl.append(quat_rot(quat_inv(q[i]), pt - p[i])); sr.append(rr)
                k_ = sample_k_line*ss
                sk.append(k_); sc.append(2*sample_zeta*np.sqrt(k_*m[i]/max(len(P), 1)))
        self.n_smp = len(sb)
        # each body's mass per contact sample: the friction dashpot of one sample is capped at half of it per substep,
        # so all of a body's samples sticking at once stay within the explicit limit (bodies.k_body_samples, which also
        # scales it by the effective mass at the contact point, rotation included)
        n_per = np.bincount(np.array(sb, np.int64), minlength=len(bodies)) if sb else np.zeros(len(bodies), np.int64)
        self.rb["ms"] = wp.array((m/np.maximum(n_per, 1)).astype(np.float32), dtype=float, device=d)
        if self.n_smp:
            self.smp = dict(body=wp.array(np.array(sb, np.int32), dtype=wp.int32, device=d),
                            loc=wp.array(np.array(sl, np.float32), dtype=wp.vec3, device=d),
                            r=wp.array(np.array(sr, np.float32), dtype=float, device=d),
                            k=wp.array(np.array(sk, np.float32), dtype=float, device=d),
                            c=wp.array(np.array(sc, np.float32), dtype=float, device=d),
                            cand=wp.zeros((len(sb), CAND_CAP), dtype=wp.int32, device=d),
                            cand_n=wp.full(len(sb), -1, dtype=wp.int32, device=d),
                            # stick-slip friction anchors per sample and shape (bodies.k_body_samples)
                            anchor=wp.zeros((len(sb), MAX_SHAPES), dtype=wp.vec3, device=d),
                            anchor_on=wp.zeros((len(sb), MAX_SHAPES), dtype=wp.int32, device=d),
                            # the friction force each anchor applied last step, and the offset from it then (booking)
                            a_ft=wp.zeros((len(sb), MAX_SHAPES), dtype=wp.vec3, device=d),
                            a_d=wp.zeros((len(sb), MAX_SHAPES), dtype=wp.vec3, device=d))
        # joints
        self.n_joints = len(joints)
        if self.n_joints:
            ja, jb, jpa, jpb, jk, jc, jka, jca, jq0 = [], [], [], [], [], [], [], [], []
            for jn in joints:
                a, b = jn["a"], jn["b"]
                anc = np.asarray(jn["anchor"], float)
                ja.append(a); jb.append(b)
                jpa.append(quat_rot(quat_inv(q[a]), anc - p[a])); jpb.append(quat_rot(quat_inv(q[b]), anc - p[b]))
                jk.append(jn["k"]); jc.append(jn["c"]); jka.append(jn["k_ang"]); jca.append(jn["c_ang"])
                jq0.append(quat_mul(quat_inv(q[a]), q[b]))
                ka_ = jn["k_ang"]; ca_ = jn["c_ang"]
                jka[-1] = np.broadcast_to(np.asarray(ka_, float), 3).copy()
                jca[-1] = np.broadcast_to(np.asarray(ca_, float), 3).copy()
            f32 = lambda a_: np.array(a_, np.float32)
            self.joints = dict(a=wp.array(np.array(ja, np.int32), dtype=wp.int32, device=d),
                               b=wp.array(np.array(jb, np.int32), dtype=wp.int32, device=d),
                               pa=wp.array(f32(jpa), dtype=wp.vec3, device=d), pb=wp.array(f32(jpb), dtype=wp.vec3, device=d),
                               k=wp.array(f32(jk), dtype=float, device=d), c=wp.array(f32(jc), dtype=float, device=d),
                               ka=wp.array(f32(jka), dtype=wp.vec3, device=d), ca=wp.array(f32(jca), dtype=wp.vec3, device=d),
                               q0=wp.array(f32(jq0), dtype=wp.quat, device=d),
                               on=wp.array(np.ones(len(ja), np.int32), dtype=wp.int32, device=d),
                               load=wp.zeros(len(ja), dtype=wp.vec3, device=d))
            self.joint_desc = list(joints)

    def add_body_ties(self, nodes, body, loc_world, k, fmax, dfull, kd):
        """Flesh-bone (or rod-root) cohesive bonds: particle `nodes[i]` to the point `loc_world[i]` (rest,
        world) on body `body[i]`. k, fmax (N), dfull (m, separation at full damage), kd (N s/m)."""
        self._graphs = {}
        from .bodies import quat_inv, quat_rot
        d = self.device
        p = self.rb["p"].numpy(); q = self.rb["q"].numpy()
        loc = np.array([quat_rot(quat_inv(q[b]), np.asarray(pt) - p[b]) for b, pt in zip(body, loc_world)], np.float32)
        f32 = lambda a_: np.asarray(a_, np.float32)
        self.bties = dict(node=wp.array(np.asarray(nodes, np.int32), dtype=wp.int32, device=d),
                          body=wp.array(np.asarray(body, np.int32), dtype=wp.int32, device=d),
                          loc=wp.array(loc, dtype=wp.vec3, device=d), k=wp.array(f32(k), dtype=float, device=d),
                          fmax=wp.array(f32(fmax), dtype=float, device=d), dfull=wp.array(f32(dfull), dtype=float, device=d),
                          damage=wp.zeros(len(nodes), dtype=float, device=d), kd=wp.array(f32(kd), dtype=float, device=d),
                          released=wp.zeros(1, dtype=wp.int32, device=d))
        self.n_bties = len(nodes)
        self.h_btie_k = np.zeros(self.Nc)
        np.add.at(self.h_btie_k, np.asarray(nodes), np.asarray(k, float))

    def release_body_ties(self, blade_ids, margin=5e-4, below=1e-4):
        """Bonds cut by a passing blade (call once per rigid tick after the tissue step)."""
        if not self.n_bties or not len(blade_ids):
            return 0
        from .bodies import k_body_tie_release
        bt = self.bties
        ids = wp.array(np.asarray(blade_ids, np.int32), dtype=wp.int32, device=self.device)
        before = int(bt["released"].numpy()[0])
        wp.launch(k_body_tie_release, dim=self.n_bties, inputs=[self.n_bties, bt["node"], bt["body"], bt["loc"], bt["damage"],
                                                               self.x, self.rb["p"], self.rb["q"], len(blade_ids), ids,
                                                               self.s_p0, self.s_q0, self.s_h, margin, below, bt["released"],
                                                               bt["fmax"], bt["dfull"], self.s_body, self.rb["debt"], bt["k"],
                                                               self.acc["iface"]],
                  device=self.device)
        return int(bt["released"].numpy()[0]) - before

    def set_driven(self, b, p0, q0, v, w):
        """Pose of a driven body at the tick start and its twist over the tick (host arrays of all bodies). A servoed
        body (flag 3) only takes them as its target: it moves under its force-limited servo (bodies.k_body_servo)."""
        rb = self.rb
        if self.h_rb_fixed[b] == 3:
            P0 = rb["p0"].numpy(); Q0 = rb["q0"].numpy(); VT = rb["vt"].numpy(); WT = rb["wt"].numpy()
            P0[b] = p0; Q0[b] = q0; VT[b] = v; WT[b] = w
            rb["p0"].assign(P0); rb["q0"].assign(Q0); rb["vt"].assign(VT); rb["wt"].assign(WT)
            return
        P0 = rb["p0"].numpy(); Q0 = rb["q0"].numpy(); V = rb["v"].numpy(); W = rb["w"].numpy()
        P = rb["p"].numpy(); Q = rb["q"].numpy()
        P0[b] = p0; Q0[b] = q0; V[b] = v; W[b] = w
        P[b] = p0; Q[b] = q0                      # only the driven body (resetting all bodies to p0 teleported the
        rb["p0"].assign(P0); rb["q0"].assign(Q0)  # blade back every tick: development probe)
        rb["v"].assign(V); rb["w"].assign(W); rb["p"].assign(P); rb["q"].assign(Q)

    def set_driven_many(self, bodies, p0, q0, v, w):
        """set_driven for several driven (flag 2) bodies at once (one host round trip)."""
        rb = self.rb
        P0 = rb["p0"].numpy(); Q0 = rb["q0"].numpy(); V = rb["v"].numpy(); W = rb["w"].numpy()
        P = rb["p"].numpy(); Q = rb["q"].numpy()
        b = np.asarray(bodies, int)
        P0[b] = p0; Q0[b] = q0; V[b] = v; W[b] = w; P[b] = p0; Q[b] = q0
        rb["p0"].assign(P0); rb["q0"].assign(Q0)
        rb["v"].assign(V); rb["w"].assign(W); rb["p"].assign(P); rb["q"].assign(Q)

    def driven_load(self):
        """Mean force and torque (about each body's reference point) received by fixed/driven bodies over the last
        tick (N, N m), world frame."""
        n = max(self.last_substeps, 1)
        return self.rb["fsum"].numpy()/n, self.rb["tsum"].numpy()/n

    def break_joint(self, j):
        on = self.joints["on"].numpy(); on[j] = 0; self.joints["on"].assign(on)

    def body_states(self):
        rb = self.rb
        return dict(p=rb["p"].numpy().astype(float), q=rb["q"].numpy().astype(float), v=rb["v"].numpy().astype(float),
                    w=rb["w"].numpy().astype(float))

    def _bodies_pre(self, dt, tl=0.):
        from .bodies import k_body_shapes, k_body_drive, k_body_servo
        rb = self.rb
        if (self.h_rb_fixed == 2).any():
            wp.launch(k_body_drive, dim=self.nb, inputs=[self.nb, rb["fixed"], rb["p0"], rb["q0"], rb["v"], rb["w"],
                                                         rb["p"], rb["q"], tl], device=self.device)
        if (self.h_rb_fixed == 3).any():
            wp.launch(k_body_servo, dim=self.nb, inputs=[self.nb, rb["fixed"], rb["p0"], rb["q0"], rb["vt"], rb["wt"],
                                                         rb["p"], rb["q"], rb["v"], rb["w"], rb["m"], rb["Ib"], self.g,
                                                         float(self.rig_servo[0]), float(self.rig_servo[1]),
                                                         float(self.rig_servo[2]), float(self.rig_servo[3]), tl, dt,
                                                         rb["f"], rb["t"], rb["fserv"], rb["tserv"],
                                                         self.acc["work_in"]], device=self.device)
        wp.launch(k_body_shapes, dim=len(self.shapes), inputs=[len(self.shapes), self.s_body, self.s_loc_p, self.s_loc_q,
                                                              rb["p"], rb["q"], rb["v"], rb["w"], self.s_p0, self.s_q0,
                                                              self.s_v, self.s_w], device=self.device)

    def _bodies_forces(self, tl, dt):
        from .bodies import k_body_samples, k_body_joints, k_body_ties
        rb, dev = self.rb, self.device
        if self.n_bties:
            bt = self.bties
            wp.launch(k_body_ties, dim=self.n_bties, inputs=[self.n_bties, bt["node"], bt["body"], bt["loc"], bt["k"], bt["fmax"],
                                                            bt["dfull"], bt["damage"], bt["kd"], self.x, self.v, self.f,
                                                            rb["p"], rb["q"], rb["v"], rb["w"], rb["f"], rb["t"], dt,
                                                            self.acc["tie"], self.acc["damp_bties"]], device=dev)
        if self.n_smp:
            sm = self.smp
            wp.launch(k_body_samples, dim=self.n_smp, inputs=[self.n_smp, sm["body"], sm["loc"], sm["r"], sm["k"], sm["c"],
                                                             rb["p"], rb["q"], rb["v"], rb["w"], rb["f"], rb["t"],
                                                             len(self.shapes), self.s_body, self.s_type, self.s_p0, self.s_q0,
                                                             self.s_v, self.s_w, self.s_h, self.s_param, self.s_com, self.s_mu,
                                                             self.s_enabled, self.s_rad, tl, dt, self.wrench,
                                                             self.acc["work_in"], self.acc["fric_samples"],
                                                             self.acc["damp_samples"], self.cload, rb["group"],
                                                             rb["fixed"], sm["cand"], sm["cand_n"], rb["knife_held"],
                                                             self.kload, rb["ms"], rb["m"], rb["Ibinv"], sm["anchor"],
                                                             sm["anchor_on"], sm["a_ft"], sm["a_d"]],
                      device=dev)
        if self.n_joints:
            jn = self.joints
            wp.launch(k_body_joints, dim=self.n_joints, inputs=[self.n_joints, jn["a"], jn["b"], jn["pa"], jn["pb"], jn["k"],
                                                               jn["c"], jn["ka"], jn["ca"], jn["q0"], jn["on"], rb["p"], rb["q"],
                                                               rb["v"], rb["w"], rb["f"], rb["t"], jn["load"], dt,
                                                               self.acc["damp_joints"]], device=dev)

    def _bodies_integrate(self, dt):
        from .bodies import k_body_integrate
        rb = self.rb
        wp.launch(k_body_integrate, dim=self.nb, inputs=[self.nb, rb["p"], rb["q"], rb["v"], rb["w"], rb["m"], rb["Ib"],
                                                        rb["Ibinv"], rb["f"], rb["t"], rb["fixed"], self.g, dt,
                                                        rb["fsum"], rb["tsum"], self.acc["work_in"], rb["debt"],
                                                        self.IFACE_PAY_M, self.acc["iface"], rb["fserv"], rb["tserv"],
                                                        rb["pc"], rb["qc"]],
                  device=self.device)

    # ---------------------------------------------------------------- rods, ties, membranes
    def add_rods(self, rods, E, rho, zeta=.2, uts=None):
        """Thin elastic bones (ribs, pin bones, supraneurals, fin supports) as particle chains with their REAL mass and
        stiffness: rods = list of (points (k,3), radius (k,)). Their stretch and bending are position-based constraints
        solved once per tissue substep (bodies.k_rods_pbd; one GPU thread per rod), with shaft bonds to the flesh and
        root bonds to a vertebral segment; an explicit rod (bodies.k_rods_subcycle, kept for comparison) needed ~1.5 us
        sub-steps for a 0.3 mm bone at 3 GPa and was too slow.
        Breaking: a link over UTS*A or a hinge over UTS*pi r^3/4 breaks (break_rods). Returns node index arrays."""
        self._graphs = {}
        d = self.device
        Nc = self.Nc
        l0 = np.zeros(Nc, np.float32); kax = np.zeros(Nc, np.float32); cax = np.zeros(Nc, np.float32)
        lon = np.zeros(Nc, np.int32); kb = np.zeros(Nc, np.float32); brest = np.zeros((Nc, 3), np.float32)
        bon = np.zeros(Nc, np.int32)
        self.rod_tmax = np.zeros(Nc); self.rod_mmax = np.zeros(Nc)
        starts, lens, ids_all = [], [], []
        omega2 = 0.
        self.rod_mass = {}
        for pts, rad in rods:
            pts = np.asarray(pts, float); rad = np.asarray(rad, float)
            k = len(pts)
            ids = np.arange(self.n, self.n + k)
            if self.n + k > Nc:
                raise RuntimeError("node capacity exhausted by rods")
            self.hX[ids] = pts; self.hkind[ids] = 2; self.hradius[ids] = rad
            self.n += k
            starts.append(int(ids[0])); lens.append(k); ids_all.append(ids)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            lump = np.zeros(k)
            for q in range(k - 1):
                r = .5*(rad[q] + rad[q + 1])
                A = math.pi*r*r
                lump[q] += rho*A*seg[q]/2; lump[q + 1] += rho*A*seg[q]/2
            for q in range(k - 1):
                a = ids[q]
                r = .5*(rad[q] + rad[q + 1])
                A = math.pi*r*r
                l0[a] = seg[q]; kax[a] = E*A/seg[q]; lon[a] = 1
                mab = lump[q]*lump[q + 1]/(lump[q] + lump[q + 1])
                cax[a] = 2*zeta*math.sqrt(kax[a]*mab)
                self.rod_tmax[a] = (uts or 0.)*A
                omega2 = max(omega2, 2*kax[a]/min(lump[q], lump[q + 1]))
            for q in range(1, k - 1):
                m_ = ids[q]
                r = rad[q]
                I = math.pi*r**4/4
                lb = .5*(seg[q - 1] + seg[q])
                t0 = (pts[q] - pts[q - 1])/seg[q - 1]; t1 = (pts[q + 1] - pts[q])/seg[q]
                kb[m_] = E*I/lb; brest[m_] = np.cross(t0, t1); bon[m_] = 1
                self.rod_mmax[m_] = (uts or 0.)*math.pi*r**3/4
                omega2 = max(omega2, 4*kb[m_]/(lb*lb*lump[q]))
            for q, i_ in enumerate(ids):
                self.point_mass[i_] += lump[q]
                self.rod_mass[int(i_)] = float(lump[q])
        self.rod_nsub = 1                         # position-based inextensibility (bodies.k_rods_pbd): no sub-steps
        self.rod_iters = 8
        f32 = lambda a_: wp.array(a_, dtype=float, device=d)
        self.rods = dict(start=wp.array(np.array(starts, np.int32), dtype=wp.int32, device=d),
                         len=wp.array(np.array(lens, np.int32), dtype=wp.int32, device=d),
                         l0=f32(l0), kax=f32(kax), cax=f32(cax), lon=wp.array(lon, dtype=wp.int32, device=d),
                         kb=f32(kb), brest=wp.array(brest, dtype=wp.vec3, device=d),
                         bon=wp.array(bon, dtype=wp.int32, device=d),
                         tie_tet=wp.full(Nc, -1, dtype=wp.int32, device=d), tie_bary=wp.zeros(Nc, dtype=wp.vec4, device=d),
                         tie_k=wp.zeros(Nc, dtype=float, device=d), tie_fmax=wp.zeros(Nc, dtype=float, device=d),
                         tie_dfull=wp.ones(Nc, dtype=float, device=d), tie_dmg=wp.zeros(Nc, dtype=float, device=d),
                         tie_c=wp.zeros(Nc, dtype=float, device=d),
                         root_body=wp.full(len(starts), -1, dtype=wp.int32, device=d),
                         root_loc=wp.zeros(len(starts), dtype=wp.vec3, device=d),
                         root_k=wp.zeros(len(starts), dtype=float, device=d),
                         root_fmax=wp.zeros(len(starts), dtype=float, device=d),
                         root_on=wp.zeros(len(starts), dtype=wp.int32, device=d),
                         fr=wp.zeros(Nc, dtype=wp.vec3, device=d), tie_sum=wp.zeros(Nc, dtype=wp.vec3, device=d),
                         dx=wp.zeros(Nc, dtype=wp.vec3, device=d),
                         tension=wp.zeros(Nc, dtype=float, device=d),
                         moment=wp.zeros(Nc, dtype=float, device=d))
        self.n_rods = len(starts)
        self.rod_ids = ids_all
        self.x.assign(self.hX.astype(np.float32)); self.X.assign(self.hX.astype(np.float32))
        self.xc.zero_()
        self.kind.assign(self.hkind); self.radius.assign(self.hradius)
        self.recompute_mass()
        return ids_all

    def set_rod_ties(self, nodes, tets, bary, k, fmax, dfull):
        """Shaft bonds of rod nodes to the tets containing them (bilinear cohesive, inside the rod sub-steps)."""
        r = self.rods
        mn = np.maximum(self.point_mass[np.asarray(nodes)], 1e-12)
        c = .7*2*np.sqrt(np.asarray(k, float)*mn)                  # 70 % of critical on the node
        for key, vals in (("tie_tet", tets), ("tie_bary", bary), ("tie_k", k), ("tie_fmax", fmax), ("tie_dfull", dfull),
                          ("tie_c", c)):
            a = r[key].numpy()
            a[np.asarray(nodes)] = np.asarray(vals, a.dtype)
            r[key].assign(a)

    def set_rod_roots(self, rod_index, body, anchor_world, k, fmax):
        """Root bond of rod `rod_index` (its first node) to rigid body `body` at a world point (rest)."""
        from .bodies import quat_inv, quat_rot
        r = self.rods
        p = self.rb["p"].numpy(); q = self.rb["q"].numpy()
        rbod = r["root_body"].numpy(); rloc = r["root_loc"].numpy(); rk = r["root_k"].numpy()
        rf = r["root_fmax"].numpy(); ron = r["root_on"].numpy()
        for ri, b, a, kk, fm in zip(rod_index, body, anchor_world, k, fmax):
            rbod[ri] = b; rloc[ri] = quat_rot(quat_inv(q[b]), np.asarray(a) - p[b]); rk[ri] = kk; rf[ri] = fm; ron[ri] = 1
        r["root_body"].assign(rbod); r["root_loc"].assign(rloc); r["root_k"].assign(rk); r["root_fmax"].assign(rf)
        r["root_on"].assign(ron)

    def break_rods(self):
        """Links over their tensile capacity and hinges over their bending capacity break (host check per tick); a
        snapped hinge also removes the distal link there, so the rod becomes two pieces."""
        if not getattr(self, "n_rods", 0):
            return 0, 0
        r = self.rods
        t = r["tension"].numpy(); m = r["moment"].numpy()
        lon = r["lon"].numpy(); bon = r["bon"].numpy()
        bl = (lon > 0) & (self.rod_tmax > 0) & (t > self.rod_tmax)
        bh = (bon > 0) & (self.rod_mmax > 0) & (m > self.rod_mmax)
        if bl.any() or bh.any():
            lon[bl] = 0
            bon[bh] = 0
            lon[np.nonzero(bh)[0]] = 0          # link from the snapped node to its distal neighbour
            r["lon"].assign(lon); r["bon"].assign(bon)
        self.rods_broken = getattr(self, "rods_broken", 0) + int(bl.sum() + bh.sum())
        return int(bl.sum()), int(bh.sum())

    def add_ties(self, nodes, tets, bary, k, fmax, dfull, kd=0.):
        """Particle-to-tet cohesive bonds (rod shaft in flesh): bilinear traction-separation (k_ties)."""
        d = self.device
        f32 = lambda a: np.asarray(a, np.float32)
        self.ties = dict(p=wp.array(np.asarray(nodes, np.int32), dtype=wp.int32, device=d),
                         tet=wp.array(np.asarray(tets, np.int32), dtype=wp.int32, device=d),
                         bary=wp.array(f32(bary), dtype=wp.vec4, device=d), k=wp.array(f32(k), dtype=float, device=d),
                         fmax=wp.array(f32(fmax), dtype=float, device=d), dfull=wp.array(f32(dfull), dtype=float, device=d),
                         damage=wp.zeros(len(nodes), dtype=float, device=d))
        self.tie_kd = float(kd)
        self.n_ties = len(nodes)
        self.h_tie_k = np.zeros(self.Nc)
        np.add.at(self.h_tie_k, np.asarray(nodes), np.asarray(k, float))

    def add_membrane(self, tri_tet, tri_face, thickness, E, nu, rho, tau=1e-6, toughness=0.):
        """Skin (or silver skin) on tissue boundary faces: triangle t is face `tri_face[t]` (0..3, the face opposite
        that local vertex) of tet `tri_tet[t]`, so after node splitting it follows the tet's current nodes
        (refresh_membrane). In-plane stable neo-Hookean with mu, lam integrated over the thickness (N/m); the skin
        mass is lumped to its nodes as point mass. `toughness` (J/m^2) is the membrane's cutting toughness, paid by the
        edge law while an anchor lies within `thickness` of a membrane face (k_edge_law)."""
        self._graphs = {}
        d = self.device
        tri_tet = np.asarray(tri_tet, np.int64); tri_face = np.asarray(tri_face, np.int64)
        self.mem_tet, self.mem_face = tri_tet, tri_face
        idx = self._membrane_nodes()
        X = self.hX
        if toughness > 0:
            rows = self.htets[tri_tet]
            opp = X[rows[np.arange(len(rows)), tri_face]]
            P0_, P1_, P2_ = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
            fn = np.cross(P1_ - P0_, P2_ - P0_)
            Af = .5*np.linalg.norm(fn, axis=1)
            Hj = np.abs(np.einsum("ij,ij->i", opp - P0_, fn))/np.maximum(2*Af, 1e-18)    # height of the opposite vertex
            frac = self.tmem_frac.numpy()
            frac[tri_tet, tri_face] = np.minimum(thickness/np.maximum(Hj, 1e-9), 1.)
            Rm = self.tmem_R.numpy()
            Rm[tri_tet] = toughness
            self.tmem_frac.assign(frac.astype(np.float32)); self.tmem_R.assign(Rm.astype(np.float32))
            self.mem_toughness = float(toughness)
        P0, P1, P2 = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
        e1, e2 = P1 - P0, P2 - P0
        # 2D rest frame per triangle
        u = e1/np.linalg.norm(e1, axis=1, keepdims=True)
        nrm = np.cross(e1, e2)
        A0 = .5*np.linalg.norm(nrm, axis=1)
        w_ = np.cross(nrm/np.linalg.norm(nrm, axis=1, keepdims=True), u)
        Dm = np.stack([np.stack([np.einsum("ij,ij->i", e1, u), np.einsum("ij,ij->i", e2, u)], 1),
                       np.stack([np.einsum("ij,ij->i", e1, w_), np.einsum("ij,ij->i", e2, w_)], 1)], 1)
        Dminv = np.linalg.inv(Dm)
        mu = E/(2*(1 + nu))*thickness
        lam = E*nu/(1 - nu*nu)*thickness            # plane-stress Lame constant of a thin sheet (not the 3D one)
        n = len(idx)
        # per-corner mass and per-corner Gershgorin stiffness of each triangle, re-scattered onto the triangle's CURRENT
        # nodes by recompute_mass (so copies made by cuts inherit them). Stiffness bound from the actual shape-function
        # gradients: K_i = A (2 lam + 3 mu) |grad N_i| sum_j |grad N_j| (thin boundary triangles are many times stiffer
        # than an equilateral estimate; with that estimate the skin was unstable: development probe)
        g1, g2 = Dminv[:, 0, :], Dminv[:, 1, :]                  # rows: grad N1, grad N2 in the 2D rest frame
        g0 = -(g1 + g2)
        gn = np.stack([np.linalg.norm(g0, axis=1), np.linalg.norm(g1, axis=1), np.linalg.norm(g2, axis=1)], 1)
        # (the stable neo-Hookean membrane carries lam' = lam + mu at rest). Damping: stiffness-proportional with
        # tau = 1e-6 s by default; the flesh's 1e-4 s gave the skin's own ~1.5e5 rad/s mesh modes a damping ratio ~7.5,
        # an explicit damper far past c dt/m = 2 (it diverged in 0.3 ms: development probe)
        lam_p = lam + mu
        self.mem_corner_k = (A0*(2*lam_p + 3*mu)*(1 + tau*1.5/self.dt))[:, None]*gn*gn.sum(1, keepdims=True)
        self.mem_corner_m = np.repeat((rho*thickness*A0/3)[:, None], 3, 1)
        self.mem_area0 = np.asarray(A0, float)            # rest area of each skin triangle (m^2): the verifier's skin area
        self.tris = dict(idx=wp.array(idx.astype(np.int32), dtype=wp.vec3i, device=d),
                         active=wp.array(np.ones(n, np.int32), dtype=wp.int32, device=d),
                         Dm_inv=wp.array(Dminv.astype(np.float32), dtype=wp.mat22, device=d),
                         A0=wp.array(A0.astype(np.float32), dtype=float, device=d),
                         mu=wp.array(np.full(n, mu, np.float32), dtype=float, device=d),
                         lam=wp.array(np.full(n, lam, np.float32), dtype=float, device=d),
                         eta=wp.array(np.full(n, tau, np.float32), dtype=float, device=d))
        self.n_tris = n
        self.recompute_mass()

    def _membrane_nodes(self):
        faces = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
        rows = self.htets[self.mem_tet]
        return rows[np.arange(len(rows))[:, None], faces[self.mem_face]]

    def refresh_membrane(self):
        """After a topology change: skin triangles take their tet's current (possibly duplicated) nodes."""
        if not self.n_tris:
            return
        self.tris["idx"].assign(self._membrane_nodes().astype(np.int32))

    def add_bonds(self, a, b, k, fmax, dfull, kd):
        """Interface bonds between node copies (bodies.k_bonds)."""
        self._graphs = {}
        d = self.device
        f32 = lambda v_: np.asarray(v_, np.float32)
        self.bonds = dict(a=wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=d),
                          b=wp.array(np.asarray(b, np.int32), dtype=wp.int32, device=d),
                          k=wp.array(f32(k), dtype=float, device=d), fmax=wp.array(f32(fmax), dtype=float, device=d),
                          dfull=wp.array(f32(dfull), dtype=float, device=d), kd=wp.array(f32(kd), dtype=float, device=d),
                          damage=wp.zeros(len(a), dtype=float, device=d), released=wp.zeros(1, dtype=wp.int32, device=d),
                          cut=wp.zeros(len(a), dtype=wp.int32, device=d))
        self.n_bonds = len(a)
        self.h_bond_k = np.zeros(self.Nc)
        np.add.at(self.h_bond_k, np.asarray(a), np.asarray(k, float))
        np.add.at(self.h_bond_k, np.asarray(b), np.asarray(k, float))
        self.recompute_mass()

    def release_bonds(self, blade_ids, margin=5e-4, below=1e-4):
        if not getattr(self, "n_bonds", 0) or not len(blade_ids):
            return 0
        from .bodies import k_bond_release
        bd = self.bonds
        ids = wp.array(np.asarray(blade_ids, np.int32), dtype=wp.int32, device=self.device)
        before = int(bd["released"].numpy()[0])
        wp.launch(k_bond_release, dim=self.n_bonds, inputs=[self.n_bonds, bd["a"], bd["b"], bd["damage"], self.x,
                                                            len(blade_ids), ids, self.s_p0, self.s_q0, self.s_h, margin,
                                                            below, bd["released"], bd["cut"], bd["fmax"], bd["dfull"],
                                                            self.s_body, self.rb["debt"], bd["k"], self.acc["iface"]],
                  device=self.device)
        return int(bd["released"].numpy()[0]) - before

    # ---------------------------------------------------------------- checkpoints
    # interface resistance: a blade that releases interface bonds or flesh-bone ties pays their unspent fracture energy
    # over IFACE_PAY_M of its travel (bodies.k_body_integrate); a numerical choice (the energy is exact; in steady
    # cutting the force equals the energy released per unit travel, independent of it)
    IFACE_PAY_M = 1e-3
    _NOSNAP = ("mats", "grid", "cent", "act_ids", "bvh", "shapes", "g", "Kb", "Cb", "f", "ext", "wrench", "cload", "kload", "cand", "cand_n")

    def snapshot(self):
        """Complete restorable state: every host array and scalar, every device array (downloaded), and the
        nested dicts (edge law, bodies, joints, ties, twins). Search structures are rebuilt on restore."""
        import copy

        def grab(v):
            if isinstance(v, wp.array):
                return ("wp", v.numpy().copy())
            if isinstance(v, np.ndarray):
                return ("np", v.copy())
            if isinstance(v, dict):
                return ("dict", {k: grab(x) for k, x in v.items()})
            if isinstance(v, (int, float, str, bool, type(None), tuple, list)):
                return ("py", copy.deepcopy(v))
            return None
        out = {}
        for k, v in self.__dict__.items():
            if k in self._NOSNAP or k.startswith("_pose"):
                continue
            g = grab(v)
            if g is not None:
                out[k] = g
        out["_pose_now"] = ("py", copy.deepcopy(getattr(self, "_pose_now", None)))
        return out

    def restore(self, snap):
        self._graphs = {}
        def put(cur, g):
            kind, val = g
            if kind == "wp":
                # arrays that grow during a run (tear bonds: tearing.py) may differ in size from a fresh world's
                if cur is not None and tuple(cur.shape) == tuple(val.shape):
                    cur.assign(val)
                    return cur
                if cur is not None:
                    return wp.array(val, dtype=cur.dtype, device=self.device)
                dt_ = wp.int32 if val.dtype == np.int32 else (wp.vec3 if val.ndim == 2 and val.shape[1] == 3 else float)
                return wp.array(val, dtype=dt_, device=self.device)
            if kind == "np":
                return val.copy()
            if kind == "dict":
                if cur is None:
                    cur = {}
                for k, x in val.items():
                    # x is None for what a snapshot cannot hold (captured CUDA graphs): the fresh world's entry stays
                    if isinstance(cur, dict) and x is not None:
                        cur[k] = put(cur.get(k), x)
                return cur
            import copy
            return copy.deepcopy(val)
        for k, g in snap.items():
            if k in self.__dict__ and g[0] in ("wp", "dict"):
                self.__dict__[k] = put(self.__dict__[k], g)
            else:
                self.__dict__[k] = put(None, g) if g[0] != "wp" else self.__dict__.get(k)
        self.rebuild_bvh()

    # ---------------------------------------------------------------- diagnostics
    debug_nan = False

    def place_on_flanks(self, x, pairs):
        """Duplicated node pairs (plus_copy, minus_copy) that lie inside a blade wedge: put each copy on its own
        flank (+1 copy at local y = +w, -1 copy at -w). The blade already occupies that space; a copy left
        on the wrong side would start with up to 2w of penalty penetration (~5 mJ each at the contact
        stiffness), which is what pumped energy into freshly cut faces (development probe)."""
        if not self.shapes or not len(pairs):
            return 0
        from scipy.spatial.transform import Rotation as Rot
        p0, q0 = self._pose_now[0], self._pose_now[1]
        moved = 0
        for si, sh in enumerate(self.shapes):
            if sh["type"] != WEDGE:
                continue
            R = Rot.from_quat(q0[si]).as_matrix()
            hx, hy, hz = sh["half"]
            bev, edge_half = sh["param"]
            H = 2*hz
            for a, b in pairs:
                pl = R.T @ (x[a] - p0[si])
                z = pl[2]
                if not (0 < z < H and abs(pl[0]) < hx):
                    continue
                w = min(z*np.tan(bev), edge_half + (hy - edge_half)*z/H)
                if abs(pl[1]) >= w:
                    continue
                for node, sgn in ((a, 1.), (b, -1.)):
                    q = R.T @ (x[node] - p0[si])
                    q[1] = sgn*(w + 1e-6)
                    x[node] = p0[si] + R @ q
                moved += 1
        return moved

    def place_nodes_on_flanks(self, x, node_signs):
        """Freshly split crack-face nodes that lie INSIDE a blade wedge are put on their own flank (sign +1 at
        local y = +w, -1 at -w): the blade already occupies that space, and a copy left on the wrong side would
        start with up to 2w of penalty penetration. Nodes outside the wedge are never moved: the crack follows
        whole-tet faces, so a node of one fragment may legitimately lie up to h/2 across the ideal cut plane
        (moving those injected 62 mJ in the coupon press-cut: development probe)."""
        if not self.shapes or not len(node_signs):
            return []
        from scipy.spatial.transform import Rotation as Rot
        p0, q0 = self._pose_now[0], self._pose_now[1]
        moved = []
        for si, sh in enumerate(self.shapes):
            if sh["type"] != WEDGE:
                continue
            R = Rot.from_quat(q0[si]).as_matrix()
            hx, hy, hz = sh["half"]
            bev, edge_half = sh["param"]
            H = 2*hz
            for node, sgn in node_signs:
                q = R.T @ (x[node] - p0[si])
                if not (0 < q[2] < H and abs(q[0]) < hx):
                    continue
                w = min(q[2]*np.tan(bev), edge_half + (hy - edge_half)*q[2]/H)
                if abs(q[1]) >= w or sgn*q[1] >= w:
                    continue          # outside the blade: a node of a whole tet may lie across the ideal plane
                if not hasattr(self, "flank_log"):
                    self.flank_log = []
                self.flank_log.append((float(q[2]), float(q[1]), float(sgn*(w + 1e-6) - q[1]), float(w)))
                q[1] = sgn*(w + 1e-6)
                x[node] = p0[si] + R @ q
                moved.append((node, sgn))
        return moved

    # ---------------------------------------------------------------- twin ties (predictive cutting)
    def add_twins(self, pairs):
        """Tie each duplicated node pair (a, b). Stiffness 0.2 m_eff/dt^2 (stiff, stable), damping 0.5 of
        critical. Capacity grows by reallocation (rare)."""
        if not len(pairs):
            return
        pairs = np.asarray(pairs, np.int32)
        m = self.hmass
        meff = 1./(1./np.maximum(m[pairs[:, 0]], 1e-12) + 1./np.maximum(m[pairs[:, 1]], 1e-12))
        k = .1*meff/self.dt**2
        c = .5*2*np.sqrt(k*meff)
        if not hasattr(self, "twins"):
            self.n_twins = 0
            self.tw_host = dict(a=np.zeros(0, np.int32), b=np.zeros(0, np.int32), on=np.zeros(0, np.int32),
                                k=np.zeros(0, np.float32), c=np.zeros(0, np.float32))
        h = self.tw_host
        on = self.twins["on"].numpy()[:self.n_twins] if self.n_twins else np.zeros(0, np.int32)
        h["on"] = np.concatenate([on, np.ones(len(pairs), np.int32)])
        h["a"] = np.concatenate([h["a"], pairs[:, 0]]); h["b"] = np.concatenate([h["b"], pairs[:, 1]])
        h["k"] = np.concatenate([h["k"], k.astype(np.float32)]); h["c"] = np.concatenate([h["c"], c.astype(np.float32)])
        self.n_twins = len(h["a"])
        cap = max(1024, 1 << int(np.ceil(np.log2(max(self.n_twins, 1)))))
        d = self.device

        def pad(a, dt_):
            out = np.zeros(cap, dt_); out[:len(a)] = a
            return wp.array(out, dtype=wp.int32 if dt_ == np.int32 else float, device=d)
        self.twins = dict(cap=cap, a=pad(h["a"], np.int32), b=pad(h["b"], np.int32), on=pad(h["on"], np.int32),
                          k=pad(h["k"], np.float32), c=pad(h["c"], np.float32))
        self.twin_released = wp.zeros(1, dtype=wp.int32, device=d)

    def refresh_twin_stiffness(self):
        """After masses change, keep twin ties within the explicit stability limit."""
        if not getattr(self, "n_twins", 0):
            return
        h = self.tw_host
        m = self.hmass
        meff = 1./(1./np.maximum(m[h["a"]], 1e-12) + 1./np.maximum(m[h["b"]], 1e-12))
        h["k"] = (.1*meff/self.dt**2).astype(np.float32)
        h["c"] = (.5*2*np.sqrt(h["k"]*meff)).astype(np.float32)
        kk = np.zeros(self.twins["cap"], np.float32); kk[:self.n_twins] = h["k"]; self.twins["k"].assign(kk)
        cc = np.zeros(self.twins["cap"], np.float32); cc[:self.n_twins] = h["c"]; self.twins["c"].assign(cc)

    def release_twins(self, blade_shape_ids, margin=5e-4, below=1e-4):
        """Release twin ties the blade edge has passed (called once per rigid tick, end-of-tick poses)."""
        if not getattr(self, "n_twins", 0) or not len(blade_shape_ids):
            return 0
        p0, q0, v, w = self._pose_now
        ids = list(blade_shape_ids)
        bp = wp.array(np.asarray(p0[ids] + v[ids]*self.tick_dt, np.float32), dtype=wp.vec3, device=self.device)
        bq = wp.array(np.asarray(q0[ids], np.float32), dtype=wp.quat, device=self.device)
        bh = wp.array(np.asarray([self.shapes[i]["half"] for i in ids], np.float32), dtype=wp.vec3, device=self.device)
        self.twin_released.zero_()
        wp.launch(k_twin_release, dim=self.twins["cap"], inputs=[self.n_twins, self.twins["a"], self.twins["b"],
                                                                 self.twins["on"], self.x, len(ids), bp, bq, bh, margin,
                                                                 below, self.twin_released], device=self.device)
        return int(self.twin_released.numpy()[0])

    def _nan_check(self, where):
        f = self.f.numpy()[:self.n]
        if not np.isfinite(f).all():
            bad = np.nonzero(~np.isfinite(f).all(1))[0]
            raise FloatingPointError(f"non-finite force at {where}: nodes {bad[:10]}")

    def energies(self):
        """Kinetic, gravitational potential (relative to z = 0) and elastic energy of the tets (J)."""
        for k in ("ke", "pe", "psi"):
            self.acc[k].zero_()
        self.Jmin.fill_(1e9)
        wp.launch(k_kinetic, dim=self.n, inputs=[self.n, self.v, self.mass, self.x, self.g, self.acc["ke"], self.acc["pe"],
                                                 self.gfrac],
                  device=self.device)
        wp.launch(k_tet_energy, dim=self.Tc, inputs=[self.nt, self.tets, self.tactive, self.x, self.Dm_inv, self.V0,
                                                     self.tscale, self.tmat, self.mat_mu, self.mat_lam, self.acc["psi"], self.Jmin,
                                                     self.mat_gv, self.tSv],
                  device=self.device)
        if self.nb:
            from .bodies import k_body_energy
            rb = self.rb
            wp.launch(k_body_energy, dim=self.nb, inputs=[self.nb, rb["p"], rb["q"], rb["v"], rb["w"], rb["m"], rb["Ib"],
                                                         self.g, self.acc["ke"], self.acc["pe"], rb["fixed"]], device=self.device)
        return dict(ke=float(self.acc["ke"].numpy()[0]), pe=float(self.acc["pe"].numpy()[0]),
                    psi=float(self.acc["psi"].numpy()[0]), Jmin=float(self.Jmin.numpy()[0]))

    def book(self, key, joules):
        """Add `joules` to a ledger accumulator from the host (discrete operations between substeps)."""
        a = self.acc[key].numpy()
        a[0] += joules
        self.acc[key].assign(a)

    def ledger(self, split=False):
        v = {k: float(a.numpy()[0]) for k, a in self.acc.items() if k not in ("ke", "pe", "psi")}
        out = dict(visc=v["visc"], fric=v["fric"] + v["fric_samples"], cut=v["cut"], tie=v["tie"],
                   damp=v["damp"] + v["damp_samples"] + v["damp_joints"] + v["damp_bties"] + v["damp_bonds"]
                   + v.get("damp_rods", 0.) + v.get("damp_settle", 0.),
                   iface=v["iface"], topo=v["topo"], work_in=v["work_in"])
        if split:
            out["split"] = dict(fric_contacts=v["fric"], fric_samples=v["fric_samples"], damp_contacts=v["damp"],
                                damp_samples=v["damp_samples"], damp_joints=v["damp_joints"],
                                damp_bties=v["damp_bties"], damp_bonds=v["damp_bonds"], damp_rods=v.get("damp_rods", 0.),
                                damp_settle=v.get("damp_settle", 0.))
        return out

    def relax(self, factor):
        """Dynamic relaxation for the hands-off start-up settle only (env.FilletingEnv._settle; never during an
        episode): every free node's and free rigid body's velocity is scaled by `factor` after a tick, and the kinetic
        energy removed is booked (ledger damp, split damp_settle). Driven and servoed bodies are left to their drive.
        Without it the trout was still ringing when its episode began: its tail vertebra 258 um peak-to-peak in the
        first 0.5 s after the 1.5 s settle, decaying with an e-fold of ~2.7 s around a mean 140 um from its position at
        the episode's start (an A2 recording), because the tissue's only damping is the numerical
        Kelvin-Voigt term meant for mesh-scale ringing."""
        v = self.v.numpy()
        m = self.mass.numpy()[:self.n].astype(np.float64)
        vn = v[:self.n].astype(np.float64)
        e = .5*float((m*(vn*vn).sum(1)).sum())
        v[:self.n] *= factor
        self.v.assign(v)
        if self.nb:
            rb = self.rb
            free = rb["fixed"].numpy()[:self.nb] == 0
            if free.any():
                bv, bw = rb["v"].numpy(), rb["w"].numpy()
                mb, Ib, q = rb["m"].numpy()[:self.nb], rb["Ib"].numpy()[:self.nb], rb["q"].numpy()[:self.nb]
                from scipy.spatial.transform import Rotation
                for b in np.nonzero(free)[0]:
                    R = Rotation.from_quat(q[b].astype(np.float64)).as_matrix()
                    w = bw[b].astype(np.float64)
                    e += .5*float(mb[b])*float(bv[b].astype(np.float64) @ bv[b]) + .5*float(w @ (R @ Ib[b] @ R.T @ w))
                bv[:self.nb][free] *= factor
                bw[:self.nb][free] *= factor
                rb["v"].assign(bv)
                rb["w"].assign(bw)
        self.book("damp_settle", e*(1. - factor*factor))

    def positions(self):
        return self.x.numpy()[:self.n].astype(np.float64)

    def velocities(self):
        return self.v.numpy()[:self.n].astype(np.float64)

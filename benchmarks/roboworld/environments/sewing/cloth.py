"""Sewing cloth solver: implicit (backward-Euler) cloth with contacts, friction, air and a presser-foot body, in Warp.

Why a repository-owned solver rather than a subclass of Newton's VBD: the sewing world needs
(1) an orthotropic membrane with a toe region, (2) constraints that tie arbitrary material points of different plies
together (needle holes, stitch interlocks, thread elements, pin crossings) and grow during the episode, and (3)
friction against kinematic machine parts and robot pads with per-pair coefficients. It is checked against the
prescribed material laws by coupon probes and every force law by finite differences
(tests/), not assumed.

Per tick (dt = 0.5 ms), method "implicit" (the default):
  ply-ply contact pairs (closest face per ply, persistent) -> predict x_n + dt v_n (+ gravity) -> friction contacts
  with their normal forces lagged from the tick start, the stick-first start -> Newton iterations (2, then up to 6
  while the step exceeds its tolerance) on the incremental potential
      E(x) = sum m/2dt^2 |x - y|^2 + orthotropic membrane + bending (elastic hinges + the yarns' bending friction,
             an Iwan assembly of elastic-perfectly-plastic elements from KES 2HB)
             + contact (shapes, ply-ply, the presser-foot body: the fabric's KES compression law) + friction (IPC
             smoothed Coulomb, Li et al. 2020) + air (squeeze film, permeability, drag) + the sewing constraints
             (needle holes, interlocks, threads, tether),
  each a matrix-free block-Jacobi PCG solve (40 iterations) and a backtracking line search on E -> pins as
  projections -> the bending friction's return mapping -> velocity update.
The presser foot is a 1-dof body particle whose shapes ride on it. The rejected single-sweep XPBD and VBD paths are
kept behind `method=`.

Precision: positions, velocities and every constraint evaluation are float64; element arithmetic is float32 on
vectors relative to the element. In float32 a coordinate of 0.3 m has a resolution of 3e-8 m while a resting vertex
moves ~6e-9 m per substep under gravity: the increment rounds away and cloth freezes in place.

Contact forces are accumulated per shape from the solved step, so the robot and the machine receive the reaction.
Units: SI (m, kg, s, N). Material coordinates X (u, v) are metres in each ply's cut-panel frame.
"""

import math

import numpy as np
import warp as wp

wp.config.quiet = True

F = wp.float64
V3 = wp.vec3d
V2 = wp.vec2d

SHAPE_BOX = 0
SHAPE_CAPSULE = 1
SHAPE_SPHERE = 2

KIND_STATIC = 0
KIND_KINEMATIC = 1
KIND_FOOT = 2            # the presser foot: one vertical degree of freedom, solved here

CON_NEEDLE = 0      # hole point held to the needle axis (xy), axial coupling capped by the penetration law
CON_BIND = 1        # interlock: two ply points kept within a maximum distance (the stack thickness)
CON_THREAD = 2      # thread element: tension-only distance between two ply points (rest length, EA)
CON_PINCROSS = 3    # a pin's shaft through a ply point (perpendicular), axial friction capped
CON_ROD = 4         # rigid distance between two particles (pin head - pin tip)
CON_TETHER = 5      # thread from a ply point to a (kinematic) particle, pulling with a capped force (payout tension)

BEND_COEF = 1.0     # discrete-shell hinge coefficient, calibrated by the pure-bending coupon
HF_COEF = 1.0       # bending-friction slip torque per hinge HB |e| HF_COEF, calibrated by a bending-loop coupon
TF_COEF = 1.125     # tensile friction fraction phi = TF_COEF (1 - RT)/(1 + RT), calibrated by membrane-loop coupons
KES_PM = 4903.3     # Pa: KES-FB3 compression to 50 gf/cm2; a fabric's `thickness` is its thickness there (Tm)
COMP_LC = 0.242     # KES linearity of compression LC of the plain cotton (Ramaiah fabric C), the default curve shape
AIR_DP = 125.0      # Pa: the pressure drop at which fabric air permeability is quoted (ASTM D737)


# ----------------------------------------------------------------------------------------------------- helpers
@wp.func
def tension_law(e: wp.float64, K0: wp.float64, K1: wp.float64, et: wp.float64, kc: wp.float64):
    """Energy per unit area Psi(e) and tension n(e) = dPsi/de (N/m) of one yarn direction:
    compression kc*K0*e, toe K0*e up to et, then K1 beyond (C1 at et)."""
    half = F(0.5)
    psi = F(0.0)
    n = F(0.0)
    if e < F(0.0):
        psi = half*kc*K0*e*e
        n = kc*K0*e
    elif e <= et:
        psi = half*K0*e*e
        n = K0*e
    else:
        de = e - et
        psi = half*K0*et*et + K0*et*de + half*K1*de*de
        n = K0*et + K1*de
    return V2(psi, n)


@wp.func
def tangent_k(e: wp.float64, K0: wp.float64, K1: wp.float64, et: wp.float64, kc: wp.float64):
    if e < F(0.0):
        return kc*K0
    if e <= et:
        return K0
    return K1


@wp.func
def quat_lerp(q0: wp.quatd, q1: wp.quatd, t: wp.float64):
    s = F(1.0)
    if wp.dot(q0, q1) < F(0.0):
        s = F(-1.0)
    one = F(1.0)
    q = wp.quatd((one - t)*q0[0] + t*s*q1[0], (one - t)*q0[1] + t*s*q1[1], (one - t)*q0[2] + t*s*q1[2],
                 (one - t)*q0[3] + t*s*q1[3])
    return wp.normalize(q)


@wp.func
def shape_sdf(typ: int, half: wp.vec3d, q: wp.vec3d):
    """Signed distance and outward normal (local frame) of a box (half extents), capsule (radius half[0], axis local
    x with half-length half[1]) or sphere (radius half[0])."""
    zero = F(0.0)
    one = F(1.0)
    if typ == SHAPE_BOX:
        a = V3(wp.abs(q[0]) - half[0], wp.abs(q[1]) - half[1], wp.abs(q[2]) - half[2])
        outside = V3(wp.max(a[0], zero), wp.max(a[1], zero), wp.max(a[2], zero))
        lo = wp.length(outside)
        if lo > zero:
            n = V3(wp.sign(q[0])*outside[0], wp.sign(q[1])*outside[1], wp.sign(q[2])*outside[2])/lo
            return lo, n
        if a[0] >= a[1] and a[0] >= a[2]:
            return a[0], V3(wp.sign(q[0]), zero, zero)
        if a[1] >= a[2]:
            return a[1], V3(zero, wp.sign(q[1]), zero)
        return a[2], V3(zero, zero, wp.sign(q[2]))
    if typ == SHAPE_CAPSULE:
        t = wp.clamp(q[0], -half[1], half[1])
        r = q - V3(t, zero, zero)
        L = wp.length(r)
        if L > F(1e-12):
            return L - half[0], r/L
        return -half[0], V3(zero, zero, one)
    L = wp.length(q)
    if L > F(1e-12):
        return L - half[0], q/L
    return -half[0], V3(zero, zero, one)


@wp.func
def closest_bary(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, p: wp.vec3d):
    """Barycentric weights (of a, b, c) of the point of triangle abc closest to p (Ericson, Real-Time Collision
    Detection 5.1.5)."""
    zero = F(0.0)
    one = F(1.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return V3(one, zero, zero)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return V3(zero, one, zero)
    vc = d1*d4 - d3*d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        v = d1/(d1 - d3)
        return V3(one - v, v, zero)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return V3(zero, zero, one)
    vb = d5*d2 - d1*d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        w = d2/(d2 - d6)
        return V3(one - w, zero, w)
    va = d3*d6 - d5*d4
    if va <= zero and (d4 - d3) >= zero and (d5 - d6) >= zero:
        w = (d4 - d3)/((d4 - d3) + (d5 - d6))
        return V3(zero, one - w, w)
    denom = one/(va + vb + vc)
    v = vb*denom
    w = vc*denom
    return V3(one - v - w, v, w)


@wp.func
def dihedral(x0: wp.vec3d, x1: wp.vec3d, xa: wp.vec3d, xb: wp.vec3d):
    """Signed dihedral angle of hinge (x0, x1) with wing vertices xa (triangle x0, x1, xa counter-clockwise) and xb,
    and its gradients (Bridson et al. 2003). Zero when flat, sign by the edge direction. Checked against central
    finite differences in tests/test_cloth_gradients.py."""
    zero = F(0.0)
    e = x1 - x0
    el = wp.length(e)
    na = wp.cross(x1 - x0, xa - x0)
    nb = wp.cross(xb - x0, x1 - x0)
    la2 = wp.dot(na, na)
    lb2 = wp.dot(nb, nb)
    z3 = V3(zero, zero, zero)
    if el < F(1e-12) or la2 < F(1e-24) or lb2 < F(1e-24):
        return zero, z3, z3, z3, z3, 0
    nau = na/wp.sqrt(la2)
    nbu = nb/wp.sqrt(lb2)
    ehat = e/el
    theta = wp.atan2(wp.dot(wp.cross(nau, nbu), ehat), wp.dot(nau, nbu))
    ga = na*(el/la2)
    gb = nb*(el/lb2)
    g0 = ga*(wp.dot(xa - x1, ehat)/el) + gb*(wp.dot(xb - x1, ehat)/el)
    g1 = ga*(wp.dot(x0 - xa, ehat)/el) + gb*(wp.dot(x0 - xb, ehat)/el)
    return theta, -g0, -g1, -ga, -gb, 1


@wp.func
def tension_law32(e: float, K0: float, K1: float, et: float, kc: float):
    psi = float(0.0)
    n = float(0.0)
    if e < 0.0:
        psi = 0.5*kc*K0*e*e
        n = kc*K0*e
    elif e <= et:
        psi = 0.5*K0*e*e
        n = K0*e
    else:
        de = e - et
        psi = 0.5*K0*et*et + K0*et*de + 0.5*K1*de*de
        n = K0*et + K1*de
    return wp.vec2(psi, n)


@wp.func
def tangent_k32(e: float, K0: float, K1: float, et: float, kc: float):
    if e < 0.0:
        return kc*K0
    if e <= et:
        return K0
    return K1


@wp.func
def dihedral32(e: wp.vec3, a: wp.vec3, b: wp.vec3):
    """dihedral() in float32 on vectors relative to the hinge's first vertex (x0 = 0, x1 = e, xa = a, xb = b).
    Precision is set by the element size (mm), not by the world coordinate, so float32 suffices here while the
    positions themselves stay float64."""
    el = wp.length(e)
    na = wp.cross(e, a)
    nb = wp.cross(b, e)
    la2 = wp.dot(na, na)
    lb2 = wp.dot(nb, nb)
    z3 = wp.vec3(0.0)
    if el < 1e-12 or la2 < 1e-30 or lb2 < 1e-30:
        return 0.0, z3, z3, z3, z3, 0
    nau = na/wp.sqrt(la2)
    nbu = nb/wp.sqrt(lb2)
    ehat = e/el
    theta = wp.atan2(wp.dot(wp.cross(nau, nbu), ehat), wp.dot(nau, nbu))
    ga = na*(el/la2)
    gb = nb*(el/lb2)
    g0 = ga*(wp.dot(a - e, ehat)/el) + gb*(wp.dot(b - e, ehat)/el)
    g1 = ga*(wp.dot(-a, ehat)/el) + gb*(wp.dot(-b, ehat)/el)
    return theta, -g0, -g1, -ga, -gb, 1


@wp.func
def m32(a: wp.mat33d):
    return wp.mat33(float(a[0, 0]), float(a[0, 1]), float(a[0, 2]), float(a[1, 0]), float(a[1, 1]), float(a[1, 2]),
                    float(a[2, 0]), float(a[2, 1]), float(a[2, 2]))


@wp.func
def to32(v: wp.vec3d):
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def to64(v: wp.vec3):
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


# ----------------------------------------------------------------------------------------------------- kernels
@wp.kernel
def k_predict(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
              w: wp.array(dtype=wp.float64), f_ext: wp.array(dtype=wp.vec3d), g: wp.vec3d, h: wp.float64,
              drag: wp.float64, forces_in_predict: int, y: wp.array(dtype=wp.vec3d),
              v_start: wp.array(dtype=wp.vec3d), a_prev: wp.array(dtype=wp.vec3d), a_max: wp.float64):
    """Prediction. XPBD: x = x_prev + h v + h^2 (g + f/m). VBD: the inertial target y = x_prev + h v carries no
    force (gravity and f_ext enter the block solve, so a loaded static equilibrium is an exact fixed point), and the
    initial guess is y + h^2 a_prev with the vertex's acceleration over the previous substep
    (bounded by a_max), so uniformly accelerating motion (free fall, sliding) is not resisted by the stiff in-plane
    Hessian of a single sweep."""
    i = wp.tid()
    x_prev[i] = x[i]
    v_start[i] = v[i]
    if w[i] > F(0.0):
        vi = v[i]
        if forces_in_predict == 1:
            vi = vi + h*(g + f_ext[i]*w[i])
        vi = vi*(F(1.0) - drag*h)
        v[i] = vi
        yi = x[i] + h*vi
        y[i] = yi
        x[i] = yi
    else:
        y[i] = x[i]


@wp.kernel
def k_zero(a: wp.array(dtype=float)):
    a[wp.tid()] = 0.0


@wp.kernel
def k_membrane(ids: wp.array(dtype=int), tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22),
               A0: wp.array(dtype=float), aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2),
               mat: wp.array(dtype=float, ndim=2), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
               w: wp.array(dtype=wp.float64), lam: wp.array(dtype=float), h: float, damping: float):
    """Orthotropic membrane: warp stretch, weft stretch (energy-based constraints with the toe law) and shear (dot
    product of the deformed yarn directions, compliance 1/(G A0)). mat columns: K0w K1w etw K0f K1f etf kc G.
    lam holds the accumulated multipliers of the substep (3 per triangle). Mixed precision: the element works in
    float32 on edge vectors and displacements relative to its first vertex; the float64 positions get the
    corrections."""
    t = ids[wp.tid()]
    i0 = tri[t, 0]
    i1 = tri[t, 1]
    i2 = tri[t, 2]
    w0 = float(w[i0])
    w1 = float(w[i1])
    w2 = float(w[i2])
    if w0 + w1 + w2 == 0.0:
        return
    Dm = Dminv[t]
    A = A0[t]
    gw = Dm*aw[t]
    gf = Dm*af[t]
    a_hh = 1.0/(h*h)
    X0 = x[i0]
    d1 = to32(x[i1] - X0)
    d2 = to32(x[i2] - X0)
    v0 = to32(X0 - x_prev[i0])
    v1 = to32(x[i1] - x_prev[i1])
    v2 = to32(x[i2] - x_prev[i2])
    c0 = wp.vec3(0.0)
    c1 = wp.vec3(0.0)
    c2 = wp.vec3(0.0)
    for c in range(3):
        e1 = d1 + c1 - c0
        e2 = d2 + c2 - c0
        fw = e1*gw[0] + e2*gw[1]
        ff = e1*gf[0] + e2*gf[1]
        C = float(0.0)
        G1 = wp.vec3(0.0)
        G2 = wp.vec3(0.0)
        alpha = float(1.0)
        skip = 0
        if c < 2:
            g = gw
            fv = fw
            K0 = mat[t, 0]
            K1 = mat[t, 1]
            et = mat[t, 2]
            if c == 1:
                g = gf
                fv = ff
                K0 = mat[t, 3]
                K1 = mat[t, 4]
                et = mat[t, 5]
            L = wp.length(fv)
            if L < 1e-12:
                skip = 1
            else:
                e = L - 1.0
                pn = tension_law32(e, K0, K1, et, mat[t, 6])
                nh = fv/L
                # energy-based constraint C = sign(e) sqrt(2 A Psi), dC/de = A n / C; near e = 0 its limit sqrt(A K)
                C = wp.sign(e)*wp.sqrt(2.0*A*pn[0])
                dCde = float(0.0)
                if wp.abs(C) > 1e-12:
                    dCde = A*pn[1]/C
                else:
                    dCde = wp.sqrt(A*tangent_k32(e, K0, K1, et, mat[t, 6]))
                G1 = nh*(g[0]*dCde)
                G2 = nh*(g[1]*dCde)
                alpha = 1.0
        else:
            Gs = mat[t, 7]
            if Gs <= 0.0:
                skip = 1
            else:
                C = wp.dot(fw, ff)
                G1 = ff*gw[0] + fw*gf[0]
                G2 = ff*gw[1] + fw*gf[1]
                alpha = 1.0/(Gs*A)
        if skip == 0:
            G0 = -(G1 + G2)
            denom = w0*wp.dot(G0, G0) + w1*wp.dot(G1, G1) + w2*wp.dot(G2, G2)
            at = alpha*a_hh
            gamma = damping/h
            vel = wp.dot(G0, v0 + c0) + wp.dot(G1, v1 + c1) + wp.dot(G2, v2 + c2)
            li = 3*t + c
            l0 = lam[li]
            dl = (-C - at*l0 - gamma*vel)/((1.0 + gamma)*denom + at)
            lam[li] = l0 + dl
            c0 = c0 + G0*(w0*dl)
            c1 = c1 + G1*(w1*dl)
            c2 = c2 + G2*(w2*dl)
    x[i0] = X0 + to64(c0)
    x[i1] = x[i1] + to64(c1)
    x[i2] = x[i2] + to64(c2)


@wp.kernel
def k_bending(ids: wp.array(dtype=int), hinge: wp.array(dtype=int, ndim=2), rest: wp.array(dtype=float),
              kb: wp.array(dtype=float), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
              w: wp.array(dtype=wp.float64), lam: wp.array(dtype=float), h: float, damping: float):
    """Dihedral-angle bending with rest angle (discrete shells), float32 on vectors relative to the hinge."""
    e = ids[wp.tid()]
    i0 = hinge[e, 0]
    i1 = hinge[e, 1]
    ia = hinge[e, 2]
    ib = hinge[e, 3]
    k = kb[e]
    if k <= 0.0:
        return
    X0 = x[i0]
    r = dihedral32(to32(x[i1] - X0), to32(x[ia] - X0), to32(x[ib] - X0))
    if r[5] == 0:
        return
    C = r[0] - rest[e]
    if C > wp.pi:
        C = C - 2.0*wp.pi
    if C < -wp.pi:
        C = C + 2.0*wp.pi
    g0 = r[1]
    g1 = r[2]
    ga = r[3]
    gb = r[4]
    w0 = float(w[i0])
    w1 = float(w[i1])
    wa = float(w[ia])
    wb = float(w[ib])
    denom = w0*wp.dot(g0, g0) + w1*wp.dot(g1, g1) + wa*wp.dot(ga, ga) + wb*wp.dot(gb, gb)
    if denom < 1e-30:
        return
    at = 1.0/(k*h*h)
    gamma = damping/h
    vel = wp.dot(g0, to32(X0 - x_prev[i0])) + wp.dot(g1, to32(x[i1] - x_prev[i1])) + \
        wp.dot(ga, to32(x[ia] - x_prev[ia])) + wp.dot(gb, to32(x[ib] - x_prev[ib]))
    l0 = lam[e]
    dl = (-C - at*l0 - gamma*vel)/((1.0 + gamma)*denom + at)
    lam[e] = l0 + dl
    x[i0] = X0 + to64(g0*(w0*dl))
    x[i1] = x[i1] + to64(g1*(w1*dl))
    x[ia] = x[ia] + to64(ga*(wa*dl))
    x[ib] = x[ib] + to64(gb*(wb*dl))


@wp.func
def yarn_terms(cr: float, fv: wp.vec3, A: float, K0: float, K1: float, et: float, kc: float):
    """Gradient (w.r.t. one vertex with deformation-gradient coefficient cr) and PSD Hessian block of A Psi(|F a| - 1)."""
    L = wp.length(fv)
    if L < 1e-12:
        return wp.vec3(0.0), wp.mat33(0.0)
    e = L - 1.0
    pn = tension_law32(e, K0, K1, et, kc)
    nh = fv/L
    g = nh*(A*pn[1]*cr)
    nn = wp.outer(nh, nh)
    Hm = nn*(A*tangent_k32(e, K0, K1, et, kc)*cr*cr) + (wp.identity(n=3, dtype=float) - nn)*(A*wp.max(pn[1], 0.0)/L*cr*cr)
    return g, Hm


@wp.kernel
def k_vbd(ids: wp.array(dtype=int), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
          y: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64),
          adj_t_start: wp.array(dtype=int), adj_t: wp.array(dtype=int), adj_t_role: wp.array(dtype=int),
          adj_h_start: wp.array(dtype=int), adj_h: wp.array(dtype=int), adj_h_role: wp.array(dtype=int),
          tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
          aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2), mat: wp.array(dtype=float, ndim=2),
          hinge: wp.array(dtype=int, ndim=2), rest: wp.array(dtype=float), kb: wp.array(dtype=float),
          f_ext: wp.array(dtype=wp.vec3d), g: wp.vec3, h: float, damping: float):
    """Vertex block descent (Chen et al. 2024) on the incremental potential
    (m / 2h^2)|x - y|^2 + membrane + bending: each vertex of a colour takes one Newton step on its own 3x3 block.
    A static equilibrium is an exact fixed point of this update (unlike single-iteration XPBD, whose multipliers
    restart every substep), which is what the strip-tension coupon needs. Mixed precision: element arithmetic in
    float32 on vectors relative to the vertex; positions float64. damping (s) adds Rayleigh stiffness damping
    (beta/h) H (x - x_prev)."""
    i = ids[wp.tid()]
    if w[i] == wp.float64(0.0):
        return
    Xi = x[i]
    mi = float(m[i])
    a_hh = 1.0/(h*h)
    f = to32(y[i] - Xi)*(mi*a_hh) + g*mi + to32(f_ext[i])
    H = wp.identity(n=3, dtype=float)*(mi*a_hh)
    He = wp.mat33(0.0)
    ge = wp.vec3(0.0)
    for k in range(adj_t_start[i], adj_t_start[i + 1]):
        t = adj_t[k]
        role = adj_t_role[k]
        r0 = to32(x[tri[t, 0]] - Xi)
        r1 = to32(x[tri[t, 1]] - Xi)
        r2 = to32(x[tri[t, 2]] - Xi)
        d1 = r1 - r0
        d2 = r2 - r0
        Dm = Dminv[t]
        gw = Dm*aw[t]
        gf = Dm*af[t]
        cw = -(gw[0] + gw[1])
        cf = -(gf[0] + gf[1])
        if role == 1:
            cw = gw[0]
            cf = gf[0]
        elif role == 2:
            cw = gw[1]
            cf = gf[1]
        fw = d1*gw[0] + d2*gw[1]
        ff = d1*gf[0] + d2*gf[1]
        A = A0[t]
        tw = yarn_terms(cw, fw, A, mat[t, 0], mat[t, 1], mat[t, 2], mat[t, 6])
        tf = yarn_terms(cf, ff, A, mat[t, 3], mat[t, 4], mat[t, 5], mat[t, 6])
        ge = ge + tw[0] + tf[0]
        He = He + tw[1] + tf[1]
        Gs = mat[t, 7]
        if Gs > 0.0:
            sh = wp.dot(fw, ff)
            ds = ff*cw + fw*cf
            ge = ge + ds*(Gs*A*sh)
            He = He + wp.outer(ds, ds)*(Gs*A)
            sec = 2.0*cw*cf*sh*Gs*A
            if sec > 0.0:
                He = He + wp.identity(n=3, dtype=float)*sec
    for k in range(adj_h_start[i], adj_h_start[i + 1]):
        e = adj_h[k]
        role = adj_h_role[k]
        kk = kb[e]
        if kk <= 0.0:
            continue
        X0 = x[hinge[e, 0]]
        r = dihedral32(to32(x[hinge[e, 1]] - X0), to32(x[hinge[e, 2]] - X0), to32(x[hinge[e, 3]] - X0))
        if r[5] == 0:
            continue
        C = r[0] - rest[e]
        if C > wp.pi:
            C = C - 2.0*wp.pi
        if C < -wp.pi:
            C = C + 2.0*wp.pi
        gi = r[1]
        if role == 1:
            gi = r[2]
        elif role == 2:
            gi = r[3]
        elif role == 3:
            gi = r[4]
        ge = ge + gi*(kk*C)
        He = He + wp.outer(gi, gi)*kk
    f = f - ge
    if damping > 0.0:
        f = f - He*(to32(Xi - x_prev[i])*(damping/h))
        He = He*(1.0 + damping/h)
    H = H + He
    det = wp.determinant(H)
    if wp.abs(det) < 1e-30:
        return
    dxi = wp.inverse(H)*f
    x[i] = Xi + to64(dxi)


# ------------------------------------------------------------------- implicit Euler, matrix-free PCG (default)
@wp.kernel
def k_imp_predict(x: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
                  y: wp.array(dtype=wp.vec3d), w: wp.array(dtype=wp.float64), f_ext: wp.array(dtype=wp.vec3d),
                  g: wp.vec3d, dt: wp.float64, drag: wp.float64, v_start: wp.array(dtype=wp.vec3d),
                  a_prev: wp.array(dtype=wp.vec3d), a_max: wp.float64, bslot: wp.array(dtype=int),
                  bP: wp.array(dtype=wp.mat33d)):
    """Backward Euler target y = x_n + dt v_n + dt^2 (g + f/m). The Newton iterate starts at the inertial position
    x_n + dt v_n, so a concentrated external load enters through the gradient instead of displacing its vertices far
    from where the Hessian is valid, plus dt^2 times the acceleration the vertex had over the last tick,
    bounded by a_max: cloth at rest starts at rest and its first Newton step is zero, cloth in free
    fall or sliding steadily starts where it will be. a_max < 0 uses the plain start (gravity's dt^2 g for every
    vertex, which a resting vertex's contact and its neighbours' tension then have to take back every tick).
    A body's position, velocity and acceleration are first projected on the subspace it moves in this tick (bP): a
    finger pad's plane turns with the hand, and what it held along a direction no longer free would otherwise carry
    on unchecked (it carried the pads 0.4 m from the fingers)."""
    i = wp.tid()
    if bslot[i] >= 0:
        Pb = bP[bslot[i]]
        x[i] = Pb*x[i]
        v[i] = Pb*v[i]
        a_prev[i] = Pb*a_prev[i]
    x_prev[i] = x[i]
    v_start[i] = v[i]
    if w[i] > F(0.0):
        vi = v[i]*(F(1.0) - drag*dt)
        xi = x[i] + dt*vi
        y[i] = xi + (g + f_ext[i]*w[i])*(dt*dt)
        if a_max < F(0.0):
            x[i] = xi + g*(dt*dt)
        else:
            a = a_prev[i]
            an = wp.length(a)
            if an > a_max:
                a = a*(a_max/an)
            x[i] = xi + a*(dt*dt)
    else:
        y[i] = x[i]


@wp.kernel
def k_imp_zero(grad: wp.array(dtype=wp.vec3d), D: wp.array(dtype=wp.mat33), body_H: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    grad[i] = V3(F(0.0), F(0.0), F(0.0))
    D[i] = wp.mat33(0.0)
    body_H[i] = wp.mat33(0.0)


@wp.func
def yarn_block(fv: wp.vec3, A: float, K0: float, K1: float, et: float, kc: float, t: int, c0: int,
               tf_p: wp.array(dtype=float, ndim=2), Ft: float, es: float):
    """Tension n(e) and the PSD 3x3 block M of A (Psi(|F a| - 1) + the tensile friction's energy) w.r.t. F a (gradient
    w.r.t. vertex j is c_j A n nh, Hessian block jk is c_j c_k M; the friction elements enter with their secant
    stiffness, as the bending friction does)."""
    L = wp.length(fv)
    if L < 1e-12:
        return 0.0, wp.vec3(0.0), wp.mat33(0.0)
    e = L - 1.0
    pn = tension_law32(e, K0, K1, et, kc)
    fr = tens_fric(e, t, c0, tf_p, Ft, es)
    n = pn[1] + fr[0]
    nh = fv/L
    nn = wp.outer(nh, nh)
    M = nn*(A*(tangent_k32(e, K0, K1, et, kc) + fr[2])) + (wp.identity(n=3, dtype=float) - nn)*(A*wp.max(n, 0.0)/L)
    return A*n, nh, M


@wp.func
def shear_angle(fw: wp.vec3, ff: wp.vec3):
    """The shear angle between the deformed warp and weft directions fw = F a_w, ff = F a_f (rad, 0 square, toward
    pi/2 as the triangle collapses onto a line): atan2(|fw.ff|, |fw x ff|). Unlike the dot product alone it does not
    saturate with the yarns' stretch and stays defined up to the collapse."""
    return wp.atan2(wp.abs(wp.dot(fw, ff)), wp.length(wp.cross(fw, ff)))


@wp.func
def shear_lock_energy(fw: wp.vec3, ff: wp.vec3, lock: float, k: float):
    """Energy per unit area of the shear past the locking angle: 1/2 k (gamma - lock)^2 (params fabric.shear_lock_*)."""
    if k <= 0.0:
        return 0.0
    g = shear_angle(fw, ff)
    if g <= lock:
        return 0.0
    return 0.5*k*(g - lock)*(g - lock)


@wp.kernel
def k_imp_tri(tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
              aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2), mat: wp.array(dtype=float, ndim=2),
              x: wp.array(dtype=wp.vec3d), grad: wp.array(dtype=wp.vec3d), D: wp.array(dtype=wp.mat33),
              Mw: wp.array(dtype=wp.mat33), Mf: wp.array(dtype=wp.mat33), cw: wp.array(dtype=wp.vec3),
              cf: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3, ndim=2), gsa: wp.array(dtype=float),
              tf_p: wp.array(dtype=float, ndim=2), tf_F: wp.array(dtype=float, ndim=2),
              ql: wp.array(dtype=wp.vec3, ndim=2), gla: wp.array(dtype=float)):
    """Per triangle: gradient and the rank-structured Hessian data of the orthotropic membrane (with its tensile
    friction) at the iterate, plus its block-diagonal preconditioner contribution. Shear uses the Gauss-Newton
    Hessian; the shear past the locking angle is a second rank-one term on the shear angle's gradient ql."""
    t = wp.tid()
    i0 = tri[t, 0]
    i1 = tri[t, 1]
    i2 = tri[t, 2]
    X0 = x[i0]
    r1 = to32(x[i1] - X0)
    r2 = to32(x[i2] - X0)
    Dm = Dminv[t]
    gw = Dm*aw[t]
    gf = Dm*af[t]
    cwv = wp.vec3(-(gw[0] + gw[1]), gw[0], gw[1])
    cfv = wp.vec3(-(gf[0] + gf[1]), gf[0], gf[1])
    fw = r1*gw[0] + r2*gw[1]
    ff = r1*gf[0] + r2*gf[1]
    A = A0[t]
    yw = yarn_block(fw, A, mat[t, 0], mat[t, 1], mat[t, 2], mat[t, 6], t, 0, tf_p, tf_F[t, 0], mat[t, 9])
    yf = yarn_block(ff, A, mat[t, 3], mat[t, 4], mat[t, 5], mat[t, 6], t, IWAN_N, tf_p, tf_F[t, 1], mat[t, 9])
    Mw[t] = yw[2]
    Mf[t] = yf[2]
    cw[t] = cwv
    cf[t] = cfv
    GA = mat[t, 7]*A
    sh = wp.dot(fw, ff)
    # shear hysteresis (KES 2HG, 2HG5): friction elements on the shear angle (slip force lagged), secant stiffness
    sfr = tens_fric(sh, t, 2*IWAN_N, tf_p, tf_F[t, 2], mat[t, 11])
    GAs = GA + A*sfr[2]
    ns = GA*sh + A*sfr[0]
    gsa[t] = GAs
    # past the locking angle (params fabric.shear_lock_*): 1/2 k A (gamma - lock)^2 on the angle gamma =
    # atan2(|s|, J), J = |fw x ff| the area ratio. d gamma = (J ds - s dJ) / (s^2 + J^2)
    kl = float(0.0)
    nl = float(0.0)
    cr = wp.cross(fw, ff)
    J = wp.length(cr)
    sa = wp.abs(sh)
    den = sa*sa + J*J
    dJw = wp.vec3(0.0)
    dJf = wp.vec3(0.0)
    if mat[t, 14] > 0.0 and den > 1.0e-12:
        gam = wp.atan2(sa, J)
        if gam > mat[t, 13]:
            kl = mat[t, 14]*A
            nl = kl*(gam - mat[t, 13])
            if J > 1.0e-9:
                nrm = cr/J
                dJw = wp.cross(ff, nrm)        # dJ / d fw
                dJf = wp.cross(nrm, fw)        # dJ / d ff
    gla[t] = kl
    sg = wp.sign(sh)
    for j in range(3):
        idx = tri[t, j]
        qj = ff*cwv[j] + fw*cfv[j]
        q[t, j] = qj
        qlj = wp.vec3(0.0)
        if kl > 0.0:
            qlj = (qj*(sg*J) - (dJw*cwv[j] + dJf*cfv[j])*sa)/den
        ql[t, j] = qlj
        gj = yw[1]*(yw[0]*cwv[j]) + yf[1]*(yf[0]*cfv[j]) + qj*ns + qlj*nl
        wp.atomic_add(grad, idx, to64(gj))
        Dj = yw[2]*(cwv[j]*cwv[j]) + yf[2]*(cfv[j]*cfv[j]) + wp.outer(qj, qj)*GAs + wp.outer(qlj, qlj)*kl
        wp.atomic_add(D, idx, Dj)


@wp.func
def wrap_pi(a: float):
    if a > wp.pi:
        a = a - 2.0*wp.pi
    if a < -wp.pi:
        a = a + 2.0*wp.pi
    return a


@wp.func
def bend_fric(d: float, kf: float, T: float):
    """One elastic-perfectly-plastic friction element on a hinge angle: d = angle past its plastic angle, kf its stick
    stiffness, T its slip torque. Energy (Huber; the backward-Euler incremental potential of perfect plasticity with
    the plastic angle held for the tick), its derivative (|f| <= T) and the secant stiffness f/d."""
    if kf <= 0.0:
        return float(0.0), float(0.0), float(0.0)
    ds = T/kf
    ad = wp.abs(d)
    if ad <= ds:
        return kf*d, 0.5*kf*d*d, kf
    return T*wp.sign(d), T*ad - 0.5*T*ds, T/ad


MAX_PLY_SLOTS = wp.constant(8)      # plies whose last contact face is remembered per vertex (k_detect_cc)
# a ply-on-ply contact whose vertex lies on the face (true separation undefined) takes the interpolated vertex normal
# as its direction where it is within 60 deg of the face normal (k_ccfric_setup)
CC_NORMAL_VALID_COS = wp.constant(0.5)
# a second face of a ply counts as a separate contact if seen face-on in a direction turned by more than 30 deg from
# the first face's (k_detect_cc)
CC_SECOND_FACE_COS = wp.constant(0.8660254037844387)
IWAN_N = wp.constant(4)  # friction elements per hinge (Iwan): slip curvatures at (2i+1)/(2N) of kappa_s, torques T/N


@wp.func
def tens_fric(e: float, t: int, c0: int, tf_p: wp.array(dtype=float, ndim=2), Ft: float, es: float):
    """Tensile hysteresis of the yarn crossings (KES RT) in one yarn direction of a triangle: an Iwan assembly of IWAN_N
    elastic-perfectly-plastic elements on the yarn strain e, slip forces Ft/N each (N/m; Ft lagged from the last tick's
    elastic tension: Coulomb friction under the crossings' pressure, which grows with the tension), stick strains
    spread evenly over (0, es]. Returns (tension N/m, energy per unit area, secant stiffness)."""
    n = float(0.0)
    E = float(0.0)
    h = float(0.0)
    if Ft > 0.0 and es > 0.0:
        Fi = Ft/float(IWAN_N)
        for i in range(IWAN_N):
            ki = Fi/(es*(2.0*float(i) + 1.0)/(2.0*float(IWAN_N)))
            fi, ei, hi = bend_fric(e - tf_p[t, c0 + i], ki, Fi)
            n = n + fi
            E = E + ei
            h = h + hi
    return wp.vec3(n, E, h)


@wp.func
def tri_strains(t: int, tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), aw: wp.array(dtype=wp.vec2),
                af: wp.array(dtype=wp.vec2), x: wp.array(dtype=wp.vec3d)):
    """Warp and weft yarn strains |F a| - 1 and the shear (F a_w).(F a_f) of a triangle."""
    i0 = tri[t, 0]
    d1 = to32(x[tri[t, 1]] - x[i0])
    d2 = to32(x[tri[t, 2]] - x[i0])
    gw = Dminv[t]*aw[t]
    gf = Dminv[t]*af[t]
    fw = d1*gw[0] + d2*gw[1]
    ff = d1*gf[0] + d2*gf[1]
    return wp.vec3(wp.length(fw) - 1.0, wp.length(ff) - 1.0, wp.dot(fw, ff))


@wp.kernel
def k_tri_plastic(tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
                  aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2), mat: wp.array(dtype=float, ndim=2),
                  x: wp.array(dtype=wp.vec3d), tf_p: wp.array(dtype=float, ndim=2), tf_F: wp.array(dtype=float, ndim=2),
                  init: int, diss: wp.array(dtype=wp.float64)):
    """End of tick: the return mapping of the membrane friction (an element strained or sheared past its stick range
    moves its plastic strain along; the work Fi |dp| A is booked as dissipation), then the next tick's tensile slip
    forces from the elastic tension now and the shear slip force from the shear now (the crossings jam as the net
    shears: KES 2HG5 > 2HG). Columns 0..N-1 warp, N..2N-1 weft, 2N..3N-1 shear. init = 1 relaxes every element to the
    present strain (the cloth as laid holds no friction force)."""
    t = wp.tid()
    phi = mat[t, 8]
    if phi <= 0.0 and mat[t, 10] <= 0.0:
        return
    ee = tri_strains(t, tri, Dminv, aw, af, x)
    for dn in range(3):
        e = ee[dn]
        c0 = dn*IWAN_N
        Ft = tf_F[t, dn]
        es = mat[t, 11]
        if dn < 2:
            es = mat[t, 9]
        if init == 1:
            for i in range(IWAN_N):
                tf_p[t, c0 + i] = e
        elif Ft > 0.0:
            Fi = Ft/float(IWAN_N)
            for i in range(IWAN_N):
                ds = es*(2.0*float(i) + 1.0)/(2.0*float(IWAN_N))
                d = e - tf_p[t, c0 + i]
                if wp.abs(d) > ds:
                    wp.atomic_add(diss, 0, wp.float64(A0[t]*Fi*(wp.abs(d) - ds)))
                    tf_p[t, c0 + i] = e - ds*wp.sign(d)
        if dn < 2:
            tf_F[t, dn] = phi*wp.abs(tension_law32(e, mat[t, 3*dn], mat[t, 3*dn + 1], mat[t, 3*dn + 2], mat[t, 6])[1])
        else:
            tf_F[t, dn] = mat[t, 10] + mat[t, 12]*wp.abs(e)


@wp.func
def iwan_k(kf: float, i: int):
    """Stick stiffness of element i: torque T/N over the stick angle (2i+1)/(2N) of the full-slip angle T/kf."""
    return kf*2.0/float(2*i + 1)


@wp.func
def bend_fric_iwan(theta: float, e: int, hf_p: wp.array(dtype=float, ndim=2), kf: float, T: float):
    """Bending hysteresis of the yarn crossings (KES 2HB) on a hinge: an Iwan assembly of IWAN_N elements whose slip
    curvatures are spread evenly over (0, kappa_s] (micro-slip: a loop that reverses gradually, as KES-FB2 loops do,
    and dissipates at small amplitudes). At full slip the torque is T. Returns torque, energy, secant stiffness."""
    f = float(0.0)
    en = float(0.0)
    hs = float(0.0)
    for i in range(IWAN_N):
        fi, ei, hi = bend_fric(wrap_pi(theta - hf_p[e, i]), iwan_k(kf, i), T/float(IWAN_N))
        f = f + fi
        en = en + ei
        hs = hs + hi
    return f, en, hs


@wp.kernel
def k_imp_hinge(hinge: wp.array(dtype=int, ndim=2), rest: wp.array(dtype=float), kb: wp.array(dtype=float),
                x: wp.array(dtype=wp.vec3d), grad: wp.array(dtype=wp.vec3d), D: wp.array(dtype=wp.mat33),
                hg: wp.array(dtype=wp.vec3, ndim=2), hf_T: wp.array(dtype=float), hf_k: wp.array(dtype=float),
                hf_p: wp.array(dtype=float, ndim=2), hfH: wp.array(dtype=float)):
    """Per hinge: gradient (k C + f) grad(theta) and Gauss-Newton Hessian (k + f/d) grad(theta) grad(theta)^T, with
    f the bending-friction torque (bend_fric)."""
    e = wp.tid()
    k = kb[e]
    hfH[e] = 0.0
    X0 = x[hinge[e, 0]]
    r = dihedral32(to32(x[hinge[e, 1]] - X0), to32(x[hinge[e, 2]] - X0), to32(x[hinge[e, 3]] - X0))
    if r[5] == 0 or k <= 0.0:
        for j in range(4):
            hg[e, j] = wp.vec3(0.0)
        return
    C = wrap_pi(r[0] - rest[e])
    f, ef, hs = bend_fric_iwan(r[0], e, hf_p, hf_k[e], hf_T[e])
    hfH[e] = hs
    hg[e, 0] = r[1]
    hg[e, 1] = r[2]
    hg[e, 2] = r[3]
    hg[e, 3] = r[4]
    for j in range(4):
        gj = hg[e, j]
        idx = hinge[e, j]
        wp.atomic_add(grad, idx, to64(gj*(k*C + f)))
        wp.atomic_add(D, idx, wp.outer(gj, gj)*(k + hs))


@wp.kernel
def k_set_gather_shapes(x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64),
                        is_body: wp.array(dtype=int), n_shapes: int, s_type: wp.array(dtype=int),
                        s_half: wp.array(dtype=wp.vec3d), s_on: wp.array(dtype=int), s_p1: wp.array(dtype=wp.vec3d),
                        s_q1: wp.array(dtype=wp.quatd), s_body: wp.array(dtype=int),
                        cset: wp.array(dtype=wp.float64, ndim=2), vnorm: wp.array(dtype=wp.vec3),
                        cdel: wp.array(dtype=wp.float64, ndim=2)):
    """Each surface's compression against the shapes at the solved state (a rigid shape: the whole overlap)."""
    i = wp.tid()
    if w[i] == F(0.0) or is_body[i] == 1:
        return
    xi = x[i]
    for s in range(n_shapes):
        if s_on[s] == 0:
            continue
        ps = s_p1[s]
        if s_body[s] >= 0:
            ps = ps + x[s_body[s]]
        sd = shape_sdf(s_type[s], s_half[s], wp.quat_rotate_inv(s_q1[s], xi - ps))
        pen = radius[i] - sd[0]
        if pen > F(0.0):
            wp.atomic_max(cdel, i, surf_side(vnorm, i, wp.quat_rotate(s_q1[s], sd[1])), pen)


@wp.kernel
def k_set_gather_pairs(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                       count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2),
                       x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64), pfn: wp.array(dtype=wp.vec3),
                       cset: wp.array(dtype=wp.float64, ndim=2), vnorm: wp.array(dtype=wp.vec3),
                       cdel: wp.array(dtype=wp.float64, ndim=2), pfo: wp.array(dtype=wp.float64)):
    """Each surface's compression in a ply-on-ply pair: its set plus half the elastic overlap (equal fabrics in
    series carry equal pressure)."""
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    n = to64(pfn[k])
    if wp.length(n) < F(0.5):
        return
    fr = face_frame(tri, f, bb, x)
    thick = radius[v] + radius[tri[f, 0]]*bb[0] + radius[tri[f, 1]]*bb[1] + radius[tri[f, 2]]*bb[2]
    e = F(0.5)*(thick - (pfo[k] + wp.dot(x[v] - fr[0], n)) - pair_set(cset, vnorm, v, tri, f, bb, n))
    if e <= F(0.0):
        return
    wp.atomic_max(cdel, v, surf_side(vnorm, v, n), surf_set(cset, vnorm, v, n) + e)
    for j in range(3):
        if bb[j] > F(0.05):
            a = tri[f, j]
            wp.atomic_max(cdel, a, surf_side(vnorm, a, -n), surf_set(cset, vnorm, a, -n) + e)


@wp.kernel
def k_set_update(cset: wp.array(dtype=wp.float64, ndim=2), cdel: wp.array(dtype=wp.float64, ndim=2),
                 crc: wp.array(dtype=wp.float64), ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64),
                 area: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64), dt: wp.float64, tau: wp.float64,
                 prec: wp.float64):
    """The compression set: a surface compressed by d keeps (1 - RC) d of it when unloaded (KES
    resilience RC: the unloading curve returns RC of the work); the set recovers with time constant tau while the
    surface's elastic pressure is below prec (KES's 0.5 gf/cm2 reference)."""
    i = wp.tid()
    rc = crc[i]
    for side in range(2):
        d = cdel[i, side]
        s = cset[i, side]
        if rc >= F(1.0) or ccm[i] <= F(0.0) or w[i] == F(0.0):
            cset[i, side] = F(0.0)
            continue
        e = wp.max(d - s, F(0.0))
        if ck1[i]*(wp.exp(ccm[i]*e/rc) - F(1.0)) < prec*area[i]:
            s = s*wp.exp(-dt/tau)
        cset[i, side] = wp.max(s, (F(1.0) - rc)*d)


@wp.kernel
def k_hinge_plastic(hinge: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), hf_T: wp.array(dtype=float),
                    hf_k: wp.array(dtype=float), hf_p: wp.array(dtype=float, ndim=2), init: int,
                    diss: wp.array(dtype=wp.float64)):
    """End of tick: the return mapping of the bending friction. A hinge bent past its stick range moves its plastic
    angle along (the yarns slid; the work T |dp| is booked as dissipation). init = 1 relaxes every hinge to the
    present shape (the cloth as laid)."""
    e = wp.tid()
    X0 = x[hinge[e, 0]]
    r = dihedral32(to32(x[hinge[e, 1]] - X0), to32(x[hinge[e, 2]] - X0), to32(x[hinge[e, 3]] - X0))
    if r[5] == 0:
        return
    if init == 1:
        for i in range(IWAN_N):
            hf_p[e, i] = r[0]
        return
    kf = hf_k[e]
    if kf <= 0.0:
        return
    Ti = hf_T[e]/float(IWAN_N)
    for i in range(IWAN_N):
        d = wrap_pi(r[0] - hf_p[e, i])
        ds = Ti/iwan_k(kf, i)
        if wp.abs(d) > ds:
            wp.atomic_add(diss, 0, wp.float64(Ti*(wp.abs(d) - ds)))
            hf_p[e, i] = wrap_pi(r[0] - ds*wp.sign(d))


@wp.kernel
def k_imp_inertia(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64),
                  w: wp.array(dtype=wp.float64), dt: wp.float64, kscale: float, grad: wp.array(dtype=wp.vec3d),
                  D: wp.array(dtype=wp.mat33), Dinv: wp.array(dtype=wp.mat33), r: wp.array(dtype=wp.vec3d),
                  dlt: wp.array(dtype=wp.vec3d), Kd: wp.array(dtype=wp.vec3d), damp_scale: float,
                  is_body: wp.array(dtype=int), floor: wp.array(dtype=wp.float64),
                  sk: wp.array(dtype=wp.float64), st: wp.array(dtype=wp.vec3d), mask: wp.array(dtype=wp.mat33d),
                  bslot: wp.array(dtype=int), bP: wp.array(dtype=wp.mat33d), bK: wp.array(dtype=wp.mat33d)):
    """Inertia term, the right-hand side r = -grad (zero on fixed vertices) and the block-Jacobi preconditioner.
    kscale = 1 + beta/dt multiplies the elastic Hessian (Rayleigh stiffness damping); Kd = K (x - x_n) carries the
    damping force (beta/dt) K (x - x_n) into the gradient. A body (bslot >= 0) moves in the subspace its projector bP
    spans (the presser foot: the vertical; a finger pad: the vertical and its closing axis) and is held to its target
    by the scalar spring sk and the matrix spring bK. mask: the projector of the axes the solve moves, by which the
    residual is projected at every CG iteration."""
    i = wp.tid()
    dlt[i] = V3(F(0.0), F(0.0), F(0.0))
    if w[i] == F(0.0):
        Dinv[i] = wp.identity(n=3, dtype=float)
        r[i] = V3(F(0.0), F(0.0), F(0.0))
        mask[i] = wp.mat33d(F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0))
        return
    mh = m[i]/(dt*dt)
    # a compliant body (a hand on its arm) is held to its target by the spring sk, solved with the cloth
    e = x[i] - st[i]
    gi = grad[i] + (x[i] - y[i])*mh + Kd[i]*wp.float64(damp_scale) + e*sk[i]
    Di = D[i]*kscale + wp.identity(n=3, dtype=float)*float(mh + sk[i])
    if is_body[i] == 0:
        Dinv[i] = wp.inverse(Di)
        r[i] = -gi
        mask[i] = wp.identity(n=3, dtype=wp.float64)
        return
    b = bslot[i]
    P = bP[b]
    gi = gi + bK[b]*e
    Di = Di + m32(bK[b])
    # a body resting on its stop and pressed into it: z is in the active set of the stop for this iteration (the
    # clamp would undo any downward step; counting it kept the Newton loop running)
    if x[i][2] <= floor[i] + F(1e-12) and gi[2] > F(0.0):
        pz = P*V3(F(0.0), F(0.0), F(1.0))
        nz = wp.dot(pz, pz)
        if nz > F(1e-12):
            P = P - wp.outer(pz, pz)/nz
    # locked axes (a body's guided directions): the residual and the preconditioner are projected onto the free
    # subspace, so the conjugate gradient never moves them
    P32 = m32(P)
    Dinv[i] = P32*wp.inverse(P32*Di*P32 + wp.identity(n=3, dtype=float) - P32)*P32
    r[i] = -(P*gi)
    mask[i] = P


@wp.func
def ipc_f1_over_y(y: float, eps: float):
    """IPC smoothed Coulomb (Li et al. 2020): friction force magnitude mu N f1(y) along the relative tangential
    displacement y over the step; returns f1(y)/y (the PSD Hessian scale), f1 = 1 beyond eps, C1 ramp below."""
    if y >= eps:
        return 1.0/y
    return (2.0/eps - y/(eps*eps))


@wp.func
def ipc_df1(y: float, eps: float):
    """d f1 / dy: the friction force's derivative along the slip (2/eps - 2y/eps^2 in the static zone, 0 sliding)."""
    if y >= eps:
        return 0.0
    return 2.0/eps - 2.0*y/(eps*eps)


@wp.func
def fric_hessian(ut: wp.vec3, nk: wp.vec3, muN: float, eps: float, beta: float):
    """Hessian of mu N f0(|u_t|) used by Newton: mu N f1(y)/y across the slip and
    max(mu N f1'(y), beta mu N f1(y)/y) along it. beta = 1 is IPC's isotropic form; beta = 0 the exact Hessian, which
    has no stiffness along a sliding contact's slip: under the landing presser foot (mu N ~ 1 N per particle against
    2.7 N/m of inertia) its Newton steps reached 0.15 m, every trial was rejected and the cloth was driven through the
    bed. The converged step does not depend on this choice, only the path to it (the default is isotropic, with the
    stick-first start and the convergence loop carrying the breakaway over a few ticks)."""
    y = wp.length(ut)
    T = wp.identity(n=3, dtype=float) - wp.outer(nk, nk)
    a = muN*ipc_f1_over_y(y, eps)
    if y < 1e-12*eps:
        return T*a
    u = ut/y
    uu = wp.outer(u, u)
    b = wp.max(muN*ipc_df1(y, eps), beta*a)
    return (T - uu)*a + uu*b


@wp.func
def ipc_f0(y: float, eps: float):
    if y >= eps:
        return y
    return -y*y*y/(3.0*eps*eps) + y*y/eps + eps/3.0


@wp.func
def ep_force(d: wp.vec3, muN: float, delta: float):
    """Elastic-plastic (preslip) friction on a contact's elastic tangential displacement d (the stored stick
    displacement plus this tick's slip): the tangential spring muN/delta inside the stick limit, Coulomb muN beyond.
    A load below the limit is held at a fixed displacement: no creep."""
    y = wp.length(d)
    if y <= delta:
        return d*(muN/delta)
    return d*(muN/y)


@wp.func
def ep_hessian(d: wp.vec3, nk: wp.vec3, muN: float, delta: float, beta: float):
    """Its Hessian in the tangent plane: muN/delta sticking; sliding, muN/|d| across the slip and beta of that along it
    (beta as for the IPC form)."""
    T = wp.identity(n=3, dtype=float) - wp.outer(nk, nk)
    y = wp.length(d)
    if y <= delta:
        return T*(muN/delta)
    u = d/y
    a = muN/y
    return T*a - wp.outer(u, u)*(a*(1.0 - beta))


@wp.func
def ep_energy(d: wp.vec3, muN: float, delta: float):
    y = wp.length(d)
    if y <= delta:
        return 0.5*(muN/delta)*y*y
    return muN*(y - 0.5*delta)


@wp.func
def contact_law(pen: wp.float64, ca: wp.float64, k2: wp.float64, cr: wp.float64, act: wp.float64):
    """Transverse compression of the fabric at a contact: pen is the overlap of the free (uncompressed)
    half-thicknesses. The fabric's KES compression curve as an exponential pressure law on the vertex's area,
    f = ca (exp(cr pen) - 1) (ca = a A, cr the compression rate of the surfaces in series), continued linearly with the
    stiff numerical penalty k2 from the depth where its stiffness reaches k2 (C1; a fibre-packing stop far above the
    presser's pressure). cr = 0 (no compressible surface): the stiff penalty alone. Returns force, energy and the
    Newton stiffness (counted within `act` of engaging, see k_pen_shapes)."""
    zero = F(0.0)
    f = zero
    e = zero
    h = zero
    if pen > -act:
        if cr <= zero or ca <= zero:
            h = k2
            if pen > zero:
                f = k2*pen
                e = F(0.5)*k2*pen*pen
        else:
            ps = zero
            if k2 > ca*cr:
                ps = wp.log(k2/(ca*cr))/cr
            if pen <= zero:
                h = ca*cr
            elif pen <= ps:
                ex = wp.exp(cr*pen)
                f = ca*(ex - F(1.0))
                e = ca*((ex - F(1.0))/cr - pen)
                h = ca*cr*ex
            else:
                ex = wp.exp(cr*ps)
                f0 = ca*(ex - F(1.0))
                e0 = ca*((ex - F(1.0))/cr - ps)
                q = pen - ps
                f = f0 + k2*q
                e = e0 + f0*q + F(0.5)*k2*q*q
                h = k2
    return f, e, h


FIBRE_DENSITY = 1520.0                    # kg/m3, cotton (the solid stop's thickness)
SOLID_KAPPA = wp.constant(F(100.0))       # the solid stop's stiffness, in contact penalties (kappa m / dt^2)


@wp.func
def solid_stop(gap: wp.float64, k: wp.float64):
    """The fibres' volume: a contact cannot bring the cloth's mid-surface nearer than its solid half-thickness to what
    it presses on (rs, the areal density over the fibre density, halved), whatever the pressure; past it a stiff
    penalty (SOLID_KAPPA contact penalties). gap: the distance less the solid thicknesses. Returns force, energy and
    stiffness (a 35 N fingertip squeeze crushed two plies to a third of their solid thickness)."""
    if gap >= F(0.0):
        return F(0.0), F(0.0), F(0.0)
    ks = k*SOLID_KAPPA
    return -ks*gap, F(0.5)*ks*gap*gap, ks


@wp.func
def surf_set(cset: wp.array(dtype=wp.float64, ndim=2), vn: wp.array(dtype=wp.vec3), i: int, n: wp.vec3d):
    """The compression set of vertex i's surface that a contact pushing it along n touches: side 0 when
    the push is along the vertex's normal (its back surface), side 1 against it."""
    if wp.dot(to32(n), vn[i]) >= 0.0:
        return cset[i, 0]
    return cset[i, 1]


@wp.func
def surf_side(vn: wp.array(dtype=wp.vec3), i: int, n: wp.vec3d):
    if wp.dot(to32(n), vn[i]) >= 0.0:
        return 0
    return 1


@wp.func
def pair_set(cset: wp.array(dtype=wp.float64, ndim=2), vn: wp.array(dtype=wp.vec3), v: int,
             tri: wp.array(dtype=int, ndim=2), f: int, bb: wp.vec3d, n: wp.vec3d):
    """The two facing surfaces' compression sets of a ply-on-ply pair (vertex v pushed along n, the face against n)."""
    return (surf_set(cset, vn, v, n) + bb[0]*surf_set(cset, vn, tri[f, 0], -n)
            + bb[1]*surf_set(cset, vn, tri[f, 1], -n) + bb[2]*surf_set(cset, vn, tri[f, 2], -n))


@wp.func
def series_rate(r1: wp.float64, r2: wp.float64):
    """Compression rate of two compressible surfaces pressed together (same pressure, overlaps add)."""
    if r1 <= F(0.0):
        return r2
    if r2 <= F(0.0):
        return r1
    return F(1.0)/(F(1.0)/r1 + F(1.0)/r2)


@wp.kernel
def k_tri_normals(tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), vn: wp.array(dtype=wp.vec3)):
    """Area-weighted vertex normals (accumulated; normalised where read)."""
    t = wp.tid()
    a = x[tri[t, 0]]
    n = to32(wp.cross(x[tri[t, 1]] - a, x[tri[t, 2]] - a))
    for j in range(3):
        wp.atomic_add(vn, tri[t, j], n)


@wp.kernel
def k_air(x_n: wp.array(dtype=wp.vec3d), v_n: wp.array(dtype=wp.vec3d), vn: wp.array(dtype=wp.vec3),
          w: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64),
          area: wp.array(dtype=wp.float64), cperm: wp.array(dtype=wp.float64), n_shapes: int,
          s_type: wp.array(dtype=int), s_half: wp.array(dtype=wp.vec3d), s_on: wp.array(dtype=int),
          s_air: wp.array(dtype=int), s_p0: wp.array(dtype=wp.vec3d), s_q0: wp.array(dtype=wp.quatd),
          mu12L2: wp.float64, hmax: wp.float64, inv_dt: wp.float64, drag: wp.float64,
          air_H: wp.array(dtype=wp.mat33)):
    """Air between cloth and a nearby support surface (table, bed, plate) resists the cloth's motion along the surface
    normal: to move away (or toward) the surface the air must flow in through the fabric (permeability, c_perm =
    dp/v, ASTM D737) or sideways through the gap (lubrication, c_gap = 12 mu_air L^2 / h^3); the two paths are in
    parallel: c(h) = 1/(1/c_perm + h^3/(12 mu_air L^2)). Damping force -c A v_n, as the implicit dissipation
    1/2 (c A / dt) (n.(x - x_n))^2 with the normal and gap taken at the tick start.
    In free air too, the cloth's motion along its own normal meets quadratic drag 1/2 rho C_d A |v_n| v_n (`drag` =
    1/2 rho C_d), lagged in |v_n| from the last step (a hanging overhang otherwise swung undamped)."""
    i = wp.tid()
    H = wp.mat33(0.0)
    cp = cperm[i]
    if w[i] > F(0.0) and cp > F(0.0):
        xi = x_n[i]
        nv = vn[i]
        ln = wp.length(nv)
        if ln > 0.0 and drag > F(0.0):
            nh = nv/ln
            vnorm = wp.abs(wp.float64(wp.dot(nh, to32(v_n[i]))))
            H = H + wp.outer(nh, nh)*float(drag*area[i]*vnorm*inv_dt)
        for s in range(n_shapes):
            if s_on[s] == 0 or s_air[s] == 0:
                continue
            sd = shape_sdf(s_type[s], s_half[s], wp.quat_rotate_inv(s_q0[s], xi - s_p0[s]))
            h = wp.max(sd[0] - radius[i], F(0.0))
            if h >= hmax:
                continue
            c = F(1.0)/(F(1.0)/cp + h*h*h/mu12L2)
            n32 = to32(wp.quat_rotate(s_q0[s], sd[1]))
            H = H + wp.outer(n32, n32)*float(c*area[i]*inv_dt)
    air_H[i] = H


@wp.kernel
def k_pen_shapes(x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64), ck1: wp.array(dtype=wp.float64),
                 ccm: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64),
                 w: wp.array(dtype=wp.float64), n_shapes: int, s_type: wp.array(dtype=int),
                 s_half: wp.array(dtype=wp.vec3d), s_on: wp.array(dtype=int), s_p1: wp.array(dtype=wp.vec3d),
                 s_q1: wp.array(dtype=wp.quatd), s_body: wp.array(dtype=int), is_body: wp.array(dtype=int),
                 kdt: wp.float64, act: wp.float64, x_n: wp.array(dtype=wp.vec3d),
                 air_H: wp.array(dtype=wp.mat33), grad: wp.array(dtype=wp.vec3d),
                 D: wp.array(dtype=wp.mat33), pen_H: wp.array(dtype=wp.mat33), body_H: wp.array(dtype=wp.mat33),
                 bc_b: wp.array(dtype=int, ndim=2), bc_k: wp.array(dtype=float, ndim=2),
                 bc_n: wp.array(dtype=wp.vec3, ndim=2), cset: wp.array(dtype=wp.float64, ndim=2),
                 crc: wp.array(dtype=wp.float64), vnorm: wp.array(dtype=wp.vec3),
                 rsol: wp.array(dtype=wp.float64)):
    """Particle-shape contact inside the implicit step: E = 1/2 k (r - d)^2 for d < r with the shape at its end-of-tick
    pose, k = kappa m / dt^2. Gradient -k (r - d) n, Gauss-Newton Hessian k n n^T (kept per particle for
    the CG products and added to the preconditioner block). A shape attached to a body particle b (the presser foot)
    sits at x_b + its offset: the contact then also pushes b (+k (r - d) n) and couples i and b in the Hessian
    (k [n n^T, -n n^T; -n n^T, n n^T]; up to two body contacts per particle). The Hessian is included for contacts
    within `act` of touching (pen > -act), not only penetrating ones: a resting contact sits at pen ~ 0, and without
    its stiffness in the Newton model a step that closes it by a micron is rejected outright by the line search.
    The air layer's dissipation (k_air) is added here too."""
    i = wp.tid()
    H = wp.mat33(0.0)
    bc_b[i, 0] = -1
    bc_b[i, 1] = -1
    if w[i] > F(0.0) and is_body[i] == 0:
        xi = x[i]
        r = radius[i]
        k = kdt*m[i]
        Ha = air_H[i]
        H = Ha
        g = to64(Ha*to32(xi - x_n[i]))
        nb = int(0)
        for s in range(n_shapes):
            if s_on[s] == 0:
                continue
            b = s_body[s]
            ps = s_p1[s]
            if b >= 0:
                if nb >= 2:
                    continue
                ps = ps + x[b]
            ql = wp.quat_rotate_inv(s_q1[s], xi - ps)
            sd = shape_sdf(s_type[s], s_half[s], ql)
            n = wp.quat_rotate(s_q1[s], sd[1])
            pen = r - sd[0] - surf_set(cset, vnorm, i, n)
            if pen <= -act:
                continue
            cf, ce, ch = contact_law(pen, ck1[i], k, ccm[i]/crc[i], act)
            sf, se, sh_ = solid_stop(sd[0] - rsol[i], k)
            cf = cf + sf
            ch = ch + sh_
            g = g - n*cf
            n32 = to32(n)
            Hn = wp.outer(n32, n32)*float(ch)
            H = H + Hn
            if b >= 0:
                wp.atomic_add(grad, b, n*cf)
                wp.atomic_add(body_H, b, Hn)
                wp.atomic_add(D, b, Hn)
                bc_b[i, nb] = b
                bc_k[i, nb] = float(ch)
                bc_n[i, nb] = n32
                nb = nb + 1
        wp.atomic_add(grad, i, g)
        wp.atomic_add(D, i, H)
    pen_H[i] = H


@wp.kernel
def k_pen_shapes_energy(x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                        ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64), n_shapes: int,
                        s_type: wp.array(dtype=int), s_half: wp.array(dtype=wp.vec3d), s_on: wp.array(dtype=int),
                        s_p1: wp.array(dtype=wp.vec3d), s_q1: wp.array(dtype=wp.quatd), s_body: wp.array(dtype=int),
                        is_body: wp.array(dtype=int), kdt: wp.float64,
                        x_n: wp.array(dtype=wp.vec3d), air_H: wp.array(dtype=wp.mat33),
                        out: wp.array(dtype=wp.float64), slot: int, cset: wp.array(dtype=wp.float64, ndim=2),
                        crc: wp.array(dtype=wp.float64), vnorm: wp.array(dtype=wp.vec3),
                        rsol: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if w[i] == F(0.0) or is_body[i] == 1:
        return
    xi = x[i]
    r = radius[i]
    k = kdt*m[i]
    dl = to32(xi - x_n[i])
    e = F(0.5)*wp.float64(wp.dot(dl, air_H[i]*dl))
    for s in range(n_shapes):
        if s_on[s] == 0:
            continue
        ps = s_p1[s]
        if s_body[s] >= 0:
            ps = ps + x[s_body[s]]
        sd = shape_sdf(s_type[s], s_half[s], wp.quat_rotate_inv(s_q1[s], xi - ps))
        n = wp.quat_rotate(s_q1[s], sd[1])
        cf, ce, ch = contact_law(r - sd[0] - surf_set(cset, vnorm, i, n), ck1[i], k, ccm[i]/crc[i], F(0.0))
        sf_, se_, sk_ = solid_stop(sd[0] - rsol[i], k)
        ce = ce + se_
        e = e + ce
    if e > F(0.0):
        wp.atomic_add(out, slot, e)


@wp.kernel
def k_mv_pen(p: wp.array(dtype=wp.vec3d), pen_H: wp.array(dtype=wp.mat33), body_H: wp.array(dtype=wp.mat33),
             bc_b: wp.array(dtype=int, ndim=2), bc_k: wp.array(dtype=float, ndim=2),
             bc_n: wp.array(dtype=wp.vec3, ndim=2), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    pi = to32(p[i])
    o = pen_H[i]*pi + body_H[i]*pi
    for j in range(2):
        b = bc_b[i, j]
        if b >= 0:
            n = bc_n[i, j]
            o = o - n*(bc_k[i, j]*wp.dot(n, to32(p[b])))
            wp.atomic_add(out, b, to64(-n*(bc_k[i, j]*wp.dot(n, pi))))
    wp.atomic_add(out, i, to64(o))


@wp.kernel
def k_body_floor(x: wp.array(dtype=wp.vec3d), is_body: wp.array(dtype=int), floor: wp.array(dtype=wp.float64)):
    """A body's travel stop (the presser bar's lowest position: the sole on the dog teeth or the plate, or the lifter)."""
    i = wp.tid()
    if is_body[i] == 1:
        p = x[i]
        if p[2] < floor[i]:
            x[i] = V3(p[0], p[1], floor[i])


@wp.func
def face_frame(tri: wp.array(dtype=int, ndim=2), f: int, bb: wp.vec3d, x: wp.array(dtype=wp.vec3d)):
    xa = x[tri[f, 0]]
    xb = x[tri[f, 1]]
    xc = x[tri[f, 2]]
    p = xa*bb[0] + xb*bb[1] + xc*bb[2]
    n = wp.cross(xb - xa, xc - xa)
    return p, n


@wp.kernel
def k_pen_cc(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
             pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int,
             tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
             ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64),
             m: wp.array(dtype=wp.float64), kdt: wp.float64, act: wp.float64, pfn: wp.array(dtype=wp.vec3),
             grad: wp.array(dtype=wp.vec3d), D: wp.array(dtype=wp.mat33), pk: wp.array(dtype=float),
             pn: wp.array(dtype=wp.vec3), cset: wp.array(dtype=wp.float64, ndim=2), crc: wp.array(dtype=wp.float64),
             vnorm: wp.array(dtype=wp.vec3), pfo: wp.array(dtype=wp.float64), rsol: wp.array(dtype=wp.float64)):
    """Cloth-cloth contact inside the implicit step: vertex v against the point p of face f (barycentric weights fixed
    at detection: the face's closest point, on the face, an edge or a vertex), with the gap g = pfo + n.(x_v - p): the
    true point-triangle distance at the tick start, linearised along the tick's contact direction n
    (k_ccfric_setup; positive on the side the pair was detected on, so a vertex pushed through within the tick is
    pulled back to its own side), the compression law of the overlap t - g (t = r_v + r_f). n and pfo are fixed for
    the tick, so gradient, Hessian psi'' J J^T (J = (n, -b0 n, -b1 n, -b2 n)) and the line-search energy agree exactly
    (an iterate-dependent normal left the gradient 4 % off the energy)."""
    k = wp.tid()
    pk[k] = 0.0
    if k >= wp.min(count[0], capacity):
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    fr = face_frame(tri, f, bb, x)
    n = to64(pfn[k])
    if wp.length(n) < F(0.5):
        return
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    thick = radius[v] + radius[a]*bb[0] + radius[b]*bb[1] + radius[c_]*bb[2]
    pen = thick - (pfo[k] + wp.dot(x[v] - fr[0], n)) - pair_set(cset, vnorm, v, tri, f, bb, n)
    if pen <= -act:
        return
    # two compressible surfaces in series: the pressure on the vertex's area, the overlap shared by the two
    cf, ce, kk = contact_law(pen, ck1[v], kdt*m[v],
                             series_rate(ccm[v], ccm[a]*bb[0] + ccm[b]*bb[1] + ccm[c_]*bb[2])/crc[v], act)
    rs = rsol[v] + rsol[a]*bb[0] + rsol[b]*bb[1] + rsol[c_]*bb[2]
    sf, se, sk_ = solid_stop(pfo[k] + wp.dot(x[v] - fr[0], n) - rs, kdt*m[v])
    cf = cf + sf
    kk = kk + sk_
    g = n*cf
    wp.atomic_add(grad, v, -g)
    wp.atomic_add(grad, a, g*bb[0])
    wp.atomic_add(grad, b, g*bb[1])
    wp.atomic_add(grad, c_, g*bb[2])
    n32 = to32(n)
    Hn = wp.outer(n32, n32)*float(kk)
    wp.atomic_add(D, v, Hn)
    wp.atomic_add(D, a, Hn*float(bb[0]*bb[0]))
    wp.atomic_add(D, b, Hn*float(bb[1]*bb[1]))
    wp.atomic_add(D, c_, Hn*float(bb[2]*bb[2]))
    pk[k] = float(kk)
    pn[k] = n32


@wp.kernel
def k_pen_cc_energy(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                    pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int,
                    tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d),
                    radius: wp.array(dtype=wp.float64), ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64), kdt: wp.float64,
                    pfn: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float64), slot: int,
                    cset: wp.array(dtype=wp.float64, ndim=2), crc: wp.array(dtype=wp.float64),
                    vnorm: wp.array(dtype=wp.vec3), pfo: wp.array(dtype=wp.float64), rsol: wp.array(dtype=wp.float64)):
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    fr = face_frame(tri, f, bb, x)
    n = to64(pfn[k])
    if wp.length(n) < F(0.5):
        return
    thick = radius[v] + radius[tri[f, 0]]*bb[0] + radius[tri[f, 1]]*bb[1] + radius[tri[f, 2]]*bb[2]
    cm = series_rate(ccm[v], ccm[tri[f, 0]]*bb[0] + ccm[tri[f, 1]]*bb[1] + ccm[tri[f, 2]]*bb[2])/crc[v]
    cf, ce, ch = contact_law(thick - (pfo[k] + wp.dot(x[v] - fr[0], n)) - pair_set(cset, vnorm, v, tri, f, bb, n),
                             ck1[v], kdt*m[v], cm, F(0.0))
    rs = rsol[v] + rsol[tri[f, 0]]*bb[0] + rsol[tri[f, 1]]*bb[1] + rsol[tri[f, 2]]*bb[2]
    sf_, se_, sk_ = solid_stop(pfo[k] + wp.dot(x[v] - fr[0], n) - rs, kdt*m[v])
    ce = ce + se_
    if ce > F(0.0):
        wp.atomic_add(out, slot, ce)


@wp.kernel
def k_mv_pcc(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
             count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2), pk: wp.array(dtype=float),
             pn: wp.array(dtype=wp.vec3), p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    kk = pk[k]
    if kk <= 0.0:
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    n = pn[k]
    q = to32(p[v] - (p[a]*bb[0] + p[b]*bb[1] + p[c_]*bb[2]))
    s = n*(wp.dot(n, q)*kk)
    wp.atomic_add(out, v, to64(s))
    wp.atomic_add(out, a, to64(-s*float(bb[0])))
    wp.atomic_add(out, b, to64(-s*float(bb[1])))
    wp.atomic_add(out, c_, to64(-s*float(bb[2])))


@wp.func
def tangential(u: wp.vec3, n: wp.vec3):
    return u - n*wp.dot(n, u)


@wp.kernel
def k_fric_contacts(x_n: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                    ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64), is_body: wp.array(dtype=int),
                    n_shapes: int, s_type: wp.array(dtype=int), s_half: wp.array(dtype=wp.vec3d),
                    s_mu: wp.array(dtype=wp.float64), s_on: wp.array(dtype=int), s_body: wp.array(dtype=int),
                    s_p0: wp.array(dtype=wp.vec3d), s_p1: wp.array(dtype=wp.vec3d), s_q0: wp.array(dtype=wp.quatd),
                    s_q1: wp.array(dtype=wp.quatd), kdt: wp.float64, fc_n: wp.array(dtype=int),
                    fc_mu: wp.array(dtype=float, ndim=2), fc_N: wp.array(dtype=float, ndim=2),
                    fc_ds: wp.array(dtype=wp.vec3, ndim=2), fc_nrm: wp.array(dtype=wp.vec3, ndim=2),
                    fc_sid: wp.array(dtype=int, ndim=2), cset: wp.array(dtype=wp.float64, ndim=2),
                    crc: wp.array(dtype=wp.float64), vnorm: wp.array(dtype=wp.vec3),
                    fe_prev: wp.array(dtype=wp.vec3, ndim=2), fe_sid_prev: wp.array(dtype=int, ndim=2),
                    fe_cur: wp.array(dtype=wp.vec3, ndim=2), s_grip: wp.array(dtype=wp.float64),
                    area: wp.array(dtype=wp.float64), rsol: wp.array(dtype=wp.float64)):
    """Friction contacts for the tick, from the tick-start state (lagged normal force, IPC style): the contact force
    the last step ended with, N = k (r - d(x_n)) with the shape at its tick-start pose (the presser-foot load reaches
    the cloth this way, through the foot body's contacts). Stores mu, N, the contact normal and the
    displacement of the shape's contact point over the tick (a body shape's is taken as zero: the foot only moves
    vertically, along the normal of its sole)."""
    i = wp.tid()
    fc_n[i] = 0
    if w[i] == F(0.0) or is_body[i] == 1:
        return
    xi = x_n[i]
    r = radius[i]
    kp = kdt*m[i]
    k = int(0)
    for s in range(n_shapes):
        if s_on[s] == 0 or k >= 4:
            continue
        b = s_body[s]
        p0 = s_p0[s]
        if b >= 0:
            p0 = p0 + x_n[b]
        ql = wp.quat_rotate_inv(s_q0[s], xi - p0)
        sd = shape_sdf(s_type[s], s_half[s], ql)
        pe = r - sd[0] - surf_set(cset, vnorm, i, wp.quat_rotate(s_q0[s], sd[1]))
        if pe <= F(0.0):
            continue
        N, ce, ch = contact_law(pe, ck1[i], kp, ccm[i]/crc[i], F(0.0))
        sf_, se_, sk_ = solid_stop(sd[0] - rsol[i], kp)
        N = N + sf_
        loc = ql - sd[1]*sd[0]
        fc_mu[i, k] = float(s_mu[s])
        # a toothed surface (the feed dog) holds the cloth its tooth tips are in by more than friction: its
        # tangential limit is mu N + tau A (tau the teeth's engagement strength s_grip in Pa, A the vertex's area),
        # carried as an effective normal force
        fc_N[i, k] = float(N + s_grip[s]*area[i]/wp.max(s_mu[s], F(1e-6)))
        fc_nrm[i, k] = to32(wp.quat_rotate(s_q0[s], sd[1]))
        if b >= 0:
            # a body's shape: the motion of its offset and orientation (a hand's pads carried by the arm);
            # the body's own motion is normal to the work (the foot's, a hand's) and does not slide the cloth
            fc_ds[i, k] = to32((s_p1[s] + wp.quat_rotate(s_q1[s], loc)) - (s_p0[s] + wp.quat_rotate(s_q0[s], loc)))
        else:
            fc_ds[i, k] = to32((s_p1[s] + wp.quat_rotate(s_q1[s], loc)) - (p0 + wp.quat_rotate(s_q0[s], loc)))
        fc_sid[i, k] = s
        # the contact's stored stick displacement (elastic-plastic friction), re-projected on this tick's tangent plane
        e = wp.vec3(0.0)
        for j in range(4):
            if fe_sid_prev[i, j] == s:
                e = fe_prev[i, j]
        fe_cur[i, k] = tangential(e, fc_nrm[i, k])
        k = k + 1
    fc_n[i] = k


@wp.kernel
def k_stick_start(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
                  dt: wp.float64, pmu: wp.array(dtype=wp.float64),
                  fc_n: wp.array(dtype=int), fc_mu: wp.array(dtype=float, ndim=2), fc_N: wp.array(dtype=float, ndim=2),
                  fc_ds: wp.array(dtype=wp.vec3, ndim=2), fc_nrm: wp.array(dtype=wp.vec3, ndim=2), snap: float):
    """Stick-first Newton start for sticking contacts: a particle whose predicted slip on its dominant
    contact (largest mu N) is within `snap` (a few stick-zone widths, i.e. it was sticking) starts tangentially stuck
    to it; a sliding contact keeps the inertial guess. Coulomb friction's energy has a kink at zero slip that a
    quadratic model cannot cross in two Newton iterations: from the inertial guess a coupon under 0.7 of its friction
    limit crept 0.3 mm/s whatever eps_v was; re-sticking sliding contacts every tick made each tick break away anew."""
    i = wp.tid()
    n = fc_n[i]
    if n == 0:
        return
    best = int(0)
    bN = float(-1.0)
    for k in range(n):
        g = fc_mu[i, k]*fc_N[i, k]
        if g > bN:
            bN = g
            best = k
    nk = fc_nrm[i, best]
    # sticking last tick: its slip velocity relative to the contact (not the prediction, whose gravity term alone
    # slides a particle on a slope by g sin(a) dt^2, 0.26 um at 6 deg, beyond the stick zone)
    slip_prev = tangential(to32(v[i]*dt) - fc_ds[i, best], nk)
    if wp.length(slip_prev) <= snap:
        ut = tangential(to32(x[i] - x_prev[i]) - fc_ds[i, best], nk)
        x[i] = x[i] - to64(ut)


@wp.kernel
def k_fric_grad(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), pmu: wp.array(dtype=wp.float64),
                fc_n: wp.array(dtype=int), fc_mu: wp.array(dtype=float, ndim=2), fc_N: wp.array(dtype=float, ndim=2),
                fc_ds: wp.array(dtype=wp.vec3, ndim=2), fc_nrm: wp.array(dtype=wp.vec3, ndim=2), eps: float,
                grad: wp.array(dtype=wp.vec3d), D: wp.array(dtype=wp.mat33), fc_H: wp.array(dtype=wp.mat33),
                beta: float, fe_cur: wp.array(dtype=wp.vec3, ndim=2), ep: int, delta: float):
    """Gradient and exact PSD Hessian of mu N f0(|u_t|), u_t the tangential part (lagged normal) of the slip
    u = (x - x_n) - ds over the step (IPC smoothed Coulomb, Li et al. 2020). The particle's friction Hessians are
    summed into fc_H for the CG products."""
    i = wp.tid()
    H = wp.mat33(0.0)
    for k in range(fc_n[i]):
        nk = fc_nrm[i, k]
        ut = tangential(to32(x[i] - x_prev[i]) - fc_ds[i, k], nk)
        muN = fc_mu[i, k]*float(pmu[i])*fc_N[i, k]
        if ep == 1:
            d = fe_cur[i, k] + ut
            wp.atomic_add(grad, i, to64(ep_force(d, muN, delta)))
            H = H + ep_hessian(d, nk, muN, delta, beta)
        else:
            c = muN*ipc_f1_over_y(wp.length(ut), eps)
            wp.atomic_add(grad, i, to64(ut*c))
            H = H + fric_hessian(ut, nk, muN, eps, beta)
    fc_H[i] = H
    wp.atomic_add(D, i, H)


@wp.kernel
def k_mv_fric(p: wp.array(dtype=wp.vec3d), fc_n: wp.array(dtype=int), fc_H: wp.array(dtype=wp.mat33),
              out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if fc_n[i] == 0:
        return
    wp.atomic_add(out, i, to64(fc_H[i]*to32(p[i])))


@wp.kernel
def k_fric_energy(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), pmu: wp.array(dtype=wp.float64),
                  fc_n: wp.array(dtype=int), fc_mu: wp.array(dtype=float, ndim=2), fc_N: wp.array(dtype=float, ndim=2),
                  fc_ds: wp.array(dtype=wp.vec3, ndim=2), fc_nrm: wp.array(dtype=wp.vec3, ndim=2), eps: float,
                  out: wp.array(dtype=wp.float64), slot: int, fe_cur: wp.array(dtype=wp.vec3, ndim=2), ep: int,
                  delta: float):
    i = wp.tid()
    e = float(0.0)
    for k in range(fc_n[i]):
        ut = tangential(to32(x[i] - x_prev[i]) - fc_ds[i, k], fc_nrm[i, k])
        muN = fc_mu[i, k]*float(pmu[i])*fc_N[i, k]
        if ep == 1:
            e = e + ep_energy(fe_cur[i, k] + ut, muN, delta)
        else:
            e = e + muN*ipc_f0(wp.length(ut), eps)
    if e > 0.0:
        wp.atomic_add(out, slot, wp.float64(e))


@wp.kernel
def k_ccfric_setup(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                   pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int,
                   tri: wp.array(dtype=int, ndim=2), x_n: wp.array(dtype=wp.vec3d),
                   radius: wp.array(dtype=wp.float64), ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64),
                   m: wp.array(dtype=wp.float64), kdt: wp.float64,
                   pfN: wp.array(dtype=float), pfn: wp.array(dtype=wp.vec3), vnorm: wp.array(dtype=wp.vec3),
                   smooth: int, cset: wp.array(dtype=wp.float64, ndim=2), crc: wp.array(dtype=wp.float64),
                   pfo: wp.array(dtype=wp.float64), act: wp.float64, rsol: wp.array(dtype=wp.float64),
                   vel: wp.array(dtype=wp.vec3d), close_dt: wp.float64):
    """Ply-on-ply contact for the tick, from the tick-start state: its gap, direction and (lagged) friction normal
    force. The gap is the true distance from the vertex to the face, the face's closest point found at detection
    (pair_b: on the face, its edges or its vertices), positive on the side the pair was detected on. Within the tick
    the gap is linearised along a fixed direction n: g(x) = pfo + n.(x_v - p(x)) with pfo set so that g(x_n) is that
    distance, so energy, gradient and Hessian stay exact. n is the true separation direction (closest point to vertex:
    the distance's own gradient, continuous across the faces that share an edge); the face's interpolated vertex normals only where the vertex lies on the face. The
    smooth normal, measured along itself from a closest point on a face's edge, read a vertex beside a face or across
    a fold as overlapping by up to 1.9 mm (0.61 mm is the whole free thickness of the pair) and the contact kicked it
    at metres per second every tick; as the linearisation direction with the true distance
    at the tick start, a smooth normal 23 deg off the face's turned a vertex's 70 um slide along the face into a 30 um
    gap error that the next tick's true distance undid, and the vertex flickered (period 2, 80 um a tick, lagged
    normal force 0.02 / 2.1 N). A pair farther apart at the tick start than its free thickness (by more than the
    Hessian margin act) is no contact this tick and is dropped (pfn = 0), unless it is closing fast enough to touch
    within the tick: its closing speed along n at the tick start (vel: the tick-start velocities) times close_dt covers
    the gap. Such a pair carries no lagged friction (N = 0); its contact law engages inside the Newton loop when the
    linearised gap closes. Dropping it let a vertex moving at metres per second cross the other ply within one tick
    and be found on its far side next tick (E19's snap)."""
    k = wp.tid()
    pfN[k] = 0.0
    pfn[k] = wp.vec3(0.0)
    pfo[k] = F(0.0)
    if k >= wp.min(count[0], capacity):
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    fr = face_frame(tri, f, bb, x_n)
    pf = fr[0]
    ln = wp.length(fr[1])
    if ln < F(1e-14):
        return
    nf = fr[1]*(pair_side[k]/ln)                 # the face normal on the side the pair was detected on
    r0 = x_n[v] - pf
    dist = wp.length(r0)                         # true point-triangle distance (pf is the closest point)
    # direction: the true separation (closest point to vertex); the smooth normal only where that is undefined (a
    # vertex on the face) and it agrees with the face (within 60 deg), else the face normal
    n = nf
    if dist > F(1e-9):
        n = r0/dist
    else:
        ns = to64(vnorm[tri[f, 0]])*bb[0] + to64(vnorm[tri[f, 1]])*bb[1] + to64(vnorm[tri[f, 2]])*bb[2]
        ls = wp.length(ns)
        if smooth == 1 and ls > F(1e-14):
            ns = ns/ls
            if wp.dot(ns, nf) < F(0.0):
                ns = -ns
            if wp.dot(ns, nf) > F(CC_NORMAL_VALID_COS):
                n = ns
    n32 = to32(n)
    n = to64(n32)                                # the stored direction, so that g(x_n) = dist exactly
    thick = radius[v] + radius[tri[f, 0]]*bb[0] + radius[tri[f, 1]]*bb[1] + radius[tri[f, 2]]*bb[2]
    pe = thick - dist - pair_set(cset, vnorm, v, tri, f, bb, n)
    if pe < -act:
        # apart by more than the free thickness: no contact this tick unless it closes within it
        vp = vel[tri[f, 0]]*bb[0] + vel[tri[f, 1]]*bb[1] + vel[tri[f, 2]]*bb[2]
        closing = wp.dot(vp - vel[v], n)
        if closing <= F(0.0) or -pe > closing*close_dt:
            return
    N = F(0.0)
    if pe > F(0.0):
        cm = series_rate(ccm[v], ccm[tri[f, 0]]*bb[0] + ccm[tri[f, 1]]*bb[1] + ccm[tri[f, 2]]*bb[2])/crc[v]
        cf, ce, ch = contact_law(pe, ck1[v], kdt*m[v], cm, F(0.0))
        rs = rsol[v] + rsol[tri[f, 0]]*bb[0] + rsol[tri[f, 1]]*bb[1] + rsol[tri[f, 2]]*bb[2]
        sf_, se_, sk_ = solid_stop(dist - rs, kdt*m[v])
        N = cf + sf_
    pfN[k] = float(N)
    pfn[k] = n32
    pfo[k] = dist - wp.dot(r0, n)


@wp.kernel
def k_ccfric_grad(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                  count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2),
                  x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), pfN: wp.array(dtype=float),
                  pfn: wp.array(dtype=wp.vec3), mu_cc: float, eps: float, grad: wp.array(dtype=wp.vec3d),
                  D: wp.array(dtype=wp.mat33), pc: wp.array(dtype=float), pH: wp.array(dtype=wp.mat33), beta: float,
                  ccfe: wp.array(dtype=wp.vec3), ep: int, delta: float):
    """Ply-on-ply friction mu N f0(|u_t|), u = (x_v - x_v,n) - (p_f - p_f,n), tangential to the lagged face normal.
    Couples the vertex with the face's three vertices."""
    k = wp.tid()
    pc[k] = 0.0
    if k >= wp.min(count[0], capacity):
        return
    N = pfN[k]
    if N <= 0.0:
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    pnow = x[a]*bb[0] + x[b]*bb[1] + x[c_]*bb[2]
    pp = x_prev[a]*bb[0] + x_prev[b]*bb[1] + x_prev[c_]*bb[2]
    nk = pfn[k]
    ut = tangential(to32((x[v] - x_prev[v]) - (pnow - pp)), nk)
    g = wp.vec3(0.0)
    T = wp.mat33(0.0)
    if ep == 1:
        d = tangential(ccfe[v], nk) + ut
        g = ep_force(d, mu_cc*N, delta)
        T = ep_hessian(d, nk, mu_cc*N, delta, beta)
        pc[k] = mu_cc*N/delta                  # > 0: the pair enters the CG products (its Hessian pH)
    else:
        c = mu_cc*N*ipc_f1_over_y(wp.length(ut), eps)
        pc[k] = c
        g = ut*c
        T = fric_hessian(ut, nk, mu_cc*N, eps, beta)
    wp.atomic_add(grad, v, to64(g))
    wp.atomic_add(grad, a, to64(-g*float(bb[0])))
    wp.atomic_add(grad, b, to64(-g*float(bb[1])))
    wp.atomic_add(grad, c_, to64(-g*float(bb[2])))
    pH[k] = T
    wp.atomic_add(D, v, T)
    wp.atomic_add(D, a, T*float(bb[0]*bb[0]))
    wp.atomic_add(D, b, T*float(bb[1]*bb[1]))
    wp.atomic_add(D, c_, T*float(bb[2]*bb[2]))


@wp.kernel
def k_mv_ccfric(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2), pc: wp.array(dtype=float),
                pH: wp.array(dtype=wp.mat33), p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    c = pc[k]
    if c <= 0.0:
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    q = to32(p[v] - (p[a]*bb[0] + p[b]*bb[1] + p[c_]*bb[2]))
    s = pH[k]*q
    wp.atomic_add(out, v, to64(s))
    wp.atomic_add(out, a, to64(-s*float(bb[0])))
    wp.atomic_add(out, b, to64(-s*float(bb[1])))
    wp.atomic_add(out, c_, to64(-s*float(bb[2])))


@wp.kernel
def k_ccfric_energy(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                    count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2),
                    x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), pfN: wp.array(dtype=float),
                    pfn: wp.array(dtype=wp.vec3), mu_cc: float, eps: float, out: wp.array(dtype=wp.float64),
                    slot: int, ccfe: wp.array(dtype=wp.vec3), ep: int, delta: float):
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    N = pfN[k]
    if N <= 0.0:
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    pnow = x[a]*bb[0] + x[b]*bb[1] + x[c_]*bb[2]
    pp = x_prev[a]*bb[0] + x_prev[b]*bb[1] + x_prev[c_]*bb[2]
    ut = tangential(to32((x[v] - x_prev[v]) - (pnow - pp)), pfn[k])
    if ep == 1:
        wp.atomic_add(out, slot, wp.float64(ep_energy(tangential(ccfe[v], pfn[k]) + ut, mu_cc*N, delta)))
    else:
        wp.atomic_add(out, slot, wp.float64(mu_cc*N*ipc_f0(wp.length(ut), eps)))


@wp.kernel
def k_fric_return(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), fc_n: wp.array(dtype=int),
                  fc_ds: wp.array(dtype=wp.vec3, ndim=2), fc_nrm: wp.array(dtype=wp.vec3, ndim=2),
                  fc_sid: wp.array(dtype=int, ndim=2), fe_cur: wp.array(dtype=wp.vec3, ndim=2), delta: float,
                  fe_prev: wp.array(dtype=wp.vec3, ndim=2), fe_sid_prev: wp.array(dtype=int, ndim=2)):
    """End of tick (elastic-plastic friction): each contact's elastic tangential displacement after the solved step,
    cut back to the stick limit where it slid, is stored for the next tick under its shape's index."""
    i = wp.tid()
    n = fc_n[i]
    for k in range(4):
        if k < n:
            ut = tangential(to32(x[i] - x_prev[i]) - fc_ds[i, k], fc_nrm[i, k])
            d = fe_cur[i, k] + ut
            y = wp.length(d)
            if y > delta:
                d = d*(delta/y)
            fe_prev[i, k] = d
            fe_sid_prev[i, k] = fc_sid[i, k]
        else:
            fe_prev[i, k] = wp.vec3(0.0)
            fe_sid_prev[i, k] = -1


@wp.kernel
def k_fric_rescale(fc_n: wp.array(dtype=int), fc_N: wp.array(dtype=float, ndim=2), fc_sid: wp.array(dtype=int, ndim=2),
                   fN_prev: wp.array(dtype=float, ndim=2), fe_sid_prev: wp.array(dtype=int, ndim=2),
                   fe_cur: wp.array(dtype=wp.vec3, ndim=2), fNp_cur: wp.array(dtype=float, ndim=2), delta: float):
    """Tick start (elastic-plastic friction, the cloth's contacts with shapes): a contact pressed harder than ever since
    its stick state was set keeps its tangential force. The tangential spring is muN/delta, so a stored stick
    displacement d holds mu N d/delta. Each stored displacement carries the peak normal force N_p it has been held at
    since it was set (fN_prev); where this tick's lagged normal force N exceeds that peak, d becomes d N_p/N (the same
    force on the stiffer contact: the area the extra load adds carries no shear, Mindlin and Deresiewicz 1953, the
    loading branch as discrete-element models apply it, Thornton 1991) and the peak becomes N. Below the peak d is kept
    and the force follows the load, reversibly. Two other forms fail. Carried as a displacement on loading too, a
    displacement stored under a light load came back multiplied by the load's rise:
    a ply resting on the feed dog's teeth (millinewtons) that slid a fraction of the preslip held up to its full Coulomb
    force once the presser foot's 20 N per toe came down on it, and the clamp locked 1.5-2 N of opposed sideways force
    between foot and teeth at rest. Rescaled on every rise against the previous tick's load instead of against the peak
    (the discrete-element rule with its unloading branch), a contact whose load jittered by a fraction of a percent lost
    a little of its force on every cycle, and a stack draped over the table's front edge slid off it at 15 um/s.
    The ply-on-ply contacts keep the displacement: the rule on them (three forms) made the top ply's free end flip up and
    heap 13-20 mm behind the foot at the 4 mm feed in 10 of 11 runs, against none of 12 with the displacement kept
    (why is not established)."""
    i = wp.tid()
    for k in range(4):
        if k < fc_n[i]:
            N = fc_N[i, k]
            Np = float(0.0)
            for j in range(4):
                if fe_sid_prev[i, j] == fc_sid[i, k]:
                    Np = fN_prev[i, j]
            if Np > 0.0 and N > Np:
                fe_cur[i, k] = fe_cur[i, k]*(Np/N)
            fNp_cur[i, k] = wp.max(Np, N)


@wp.kernel
def k_fric_store_n(fc_n: wp.array(dtype=int), fc_N: wp.array(dtype=float, ndim=2),
                   fNp_cur: wp.array(dtype=float, ndim=2), fe_prev: wp.array(dtype=wp.vec3, ndim=2), delta: float,
                   fN_prev: wp.array(dtype=float, ndim=2)):
    """End of tick (after the return mapping): the peak normal force each stored stick displacement has been held at
    (k_fric_rescale); a contact that slid this tick (its displacement cut back to the stick limit) has its stick state
    set again at this tick's load."""
    i = wp.tid()
    for k in range(4):
        if k < fc_n[i]:
            if wp.length(fe_prev[i, k]) >= delta*0.9999:
                fN_prev[i, k] = fc_N[i, k]
            else:
                fN_prev[i, k] = fNp_cur[i, k]
        else:
            fN_prev[i, k] = 0.0


@wp.kernel
def k_ccfric_load(pair_v: wp.array(dtype=int), count: wp.array(dtype=int), capacity: int, pfN: wp.array(dtype=float),
                  ccN: wp.array(dtype=float)):
    """A vertex's ply-on-ply normal force this tick, summed over its pairs (lagged, from the tick start)."""
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    if pfN[k] > 0.0:
        wp.atomic_add(ccN, pair_v[k], pfN[k])


@wp.kernel
def k_ccfric_rescale(ccN: wp.array(dtype=float), ccN_peak: wp.array(dtype=float), ccfe: wp.array(dtype=wp.vec3),
                     ccN_pk_cur: wp.array(dtype=float)):
    """Tick start, the ply-on-ply contacts: the load-change rule of the contacts with shapes
    (k_fric_rescale). A vertex's stored stick displacement pressed harder than ever since it was set keeps its
    tangential force: d becomes d N_p / N where this tick's normal force N exceeds the peak N_p it has been held at."""
    v = wp.tid()
    N = ccN[v]
    Np = ccN_peak[v]
    if Np > 0.0 and N > Np:
        ccfe[v] = ccfe[v]*(Np/N)
    ccN_pk_cur[v] = wp.max(Np, N)


@wp.kernel
def k_ccfric_store_n(ccN: wp.array(dtype=float), ccN_pk_cur: wp.array(dtype=float), ccfe: wp.array(dtype=wp.vec3),
                     delta: float, ccN_peak: wp.array(dtype=float)):
    """Tick end: the peak load each ply-on-ply stick displacement has been held at; one that slid this tick (cut back
    to the stick limit) is set again at this tick's load, one without a contact starts afresh."""
    v = wp.tid()
    y = wp.length(ccfe[v])
    if y <= 0.0:
        ccN_peak[v] = 0.0
    elif y >= delta*0.9999:
        ccN_peak[v] = ccN[v]
    else:
        ccN_peak[v] = ccN_pk_cur[v]


@wp.kernel
def k_ccfric_return(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                    count: wp.array(dtype=int), capacity: int, tri: wp.array(dtype=int, ndim=2),
                    x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), pfN: wp.array(dtype=float),
                    pfn: wp.array(dtype=wp.vec3), ccfe: wp.array(dtype=wp.vec3), delta: float,
                    ccfe_next: wp.array(dtype=wp.vec3)):
    """End of tick for the ply-on-ply contacts: the vertex's elastic displacement relative to the other ply, stored
    by vertex (a sticking vertex stays over the same face; a vertex without a contact this tick starts afresh)."""
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    if pfN[k] <= 0.0:
        return
    v = pair_v[k]
    f = pair_f[k]
    bb = pair_b[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c_ = tri[f, 2]
    pnow = x[a]*bb[0] + x[b]*bb[1] + x[c_]*bb[2]
    pp = x_prev[a]*bb[0] + x_prev[b]*bb[1] + x_prev[c_]*bb[2]
    nk = pfn[k]
    d = tangential(ccfe[v], nk) + tangential(to32((x[v] - x_prev[v]) - (pnow - pp)), nk)
    y = wp.length(d)
    if y > delta:
        d = d*(delta/y)
    ccfe_next[v] = d


@wp.kernel
def k_contact_react(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64),
                    ck1: wp.array(dtype=wp.float64), ccm: wp.array(dtype=wp.float64),
                    m: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64), pmu: wp.array(dtype=wp.float64),
                    n_shapes: int, s_type: wp.array(dtype=int), s_half: wp.array(dtype=wp.vec3d),
                    s_on: wp.array(dtype=int), s_p1: wp.array(dtype=wp.vec3d), s_q1: wp.array(dtype=wp.quatd),
                    s_body: wp.array(dtype=int), is_body: wp.array(dtype=int),
                    kdt: wp.float64, fc_n: wp.array(dtype=int), fc_mu: wp.array(dtype=float, ndim=2),
                    fc_N: wp.array(dtype=float, ndim=2), fc_ds: wp.array(dtype=wp.vec3, ndim=2),
                    fc_nrm: wp.array(dtype=wp.vec3, ndim=2), fc_sid: wp.array(dtype=int, ndim=2), eps: float,
                    dt2: wp.float64, acc_f: wp.array(dtype=wp.vec3d), acc_tau: wp.array(dtype=wp.vec3d),
                    acc_n: wp.array(dtype=wp.float64), contact_flag: wp.array(dtype=int),
                    cset: wp.array(dtype=wp.float64, ndim=2), crc: wp.array(dtype=wp.float64),
                    vnorm: wp.array(dtype=wp.vec3), fe_cur: wp.array(dtype=wp.vec3, ndim=2), ep: int, delta: float,
                    rsol: wp.array(dtype=wp.float64)):
    """Forces of the solved step on each shape (penalty normal + friction), stored as F dt^2 so that shape_reaction
    (-acc / dt^2) reads the force the cloth applies on the shape."""
    i = wp.tid()
    if w[i] == F(0.0) or is_body[i] == 1:
        return
    xi = x[i]
    r = radius[i]
    k = kdt*m[i]
    flag = int(0)
    for s in range(n_shapes):
        if s_on[s] == 0:
            continue
        ps = s_p1[s]
        if s_body[s] >= 0:
            ps = ps + x[s_body[s]]
        sd = shape_sdf(s_type[s], s_half[s], wp.quat_rotate_inv(s_q1[s], xi - ps))
        nw = wp.quat_rotate(s_q1[s], sd[1])
        pen = r - sd[0] - surf_set(cset, vnorm, i, nw)
        if pen <= F(0.0):
            continue
        flag = 1
        cf, ce, ch = contact_law(pen, ck1[i], k, ccm[i]/crc[i], F(0.0))
        sf_, se_, sk_ = solid_stop(sd[0] - rsol[i], k)
        cf = cf + sf_
        imp = nw*(cf*dt2)
        wp.atomic_add(acc_f, s, imp)
        wp.atomic_add(acc_tau, s, wp.cross(xi - ps, imp))
        wp.atomic_add(acc_n, s, wp.dot(imp, nw))
    for kk in range(fc_n[i]):
        s = fc_sid[i, kk]
        ps = s_p1[s]
        if s_body[s] >= 0:
            ps = ps + x[s_body[s]]
        ut = tangential(to32(xi - x_prev[i]) - fc_ds[i, kk], fc_nrm[i, kk])
        muN = fc_mu[i, kk]*float(pmu[i])*fc_N[i, kk]
        imp = wp.vec3d(0.0)
        if ep == 1:
            imp = to64(-ep_force(fe_cur[i, kk] + ut, muN, delta))*dt2
        else:
            c = muN*ipc_f1_over_y(wp.length(ut), eps)
            imp = to64(-ut*c)*dt2
        wp.atomic_add(acc_f, s, imp)
        wp.atomic_add(acc_tau, s, wp.cross(xi - ps, imp))
        flag = 1
    if flag == 1:
        contact_flag[i] = 1


@wp.kernel
def k_mv_tri(tri: wp.array(dtype=int, ndim=2), Mw: wp.array(dtype=wp.mat33), Mf: wp.array(dtype=wp.mat33),
             cw: wp.array(dtype=wp.vec3), cf: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3, ndim=2),
             gsa: wp.array(dtype=float), p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d), scale: float,
             ql: wp.array(dtype=wp.vec3, ndim=2), gla: wp.array(dtype=float)):
    t = wp.tid()
    p0 = to32(p[tri[t, 0]])
    p1 = to32(p[tri[t, 1]])
    p2 = to32(p[tri[t, 2]])
    a = cw[t]
    b = cf[t]
    uw = Mw[t]*(p0*a[0] + p1*a[1] + p2*a[2])
    uf = Mf[t]*(p0*b[0] + p1*b[1] + p2*b[2])
    s = (wp.dot(q[t, 0], p0) + wp.dot(q[t, 1], p1) + wp.dot(q[t, 2], p2))*gsa[t]
    sl = float(0.0)
    if gla[t] > 0.0:
        sl = (wp.dot(ql[t, 0], p0) + wp.dot(ql[t, 1], p1) + wp.dot(ql[t, 2], p2))*gla[t]
    for j in range(3):
        oj = uw*a[j] + uf*b[j] + q[t, j]*s + ql[t, j]*sl
        wp.atomic_add(out, tri[t, j], to64(oj*scale))


@wp.kernel
def k_mv_hinge(hinge: wp.array(dtype=int, ndim=2), kb: wp.array(dtype=float), hg: wp.array(dtype=wp.vec3, ndim=2),
               p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d), scale: float, hfH: wp.array(dtype=float),
               fscale: float):
    """Hinge Hessian product: elastic part times scale, bending-friction secant times fscale."""
    e = wp.tid()
    s = float(0.0)
    for j in range(4):
        s = s + wp.dot(hg[e, j], to32(p[hinge[e, j]]))
    s = s*(kb[e]*scale + hfH[e]*fscale)
    for j in range(4):
        wp.atomic_add(out, hinge[e, j], to64(hg[e, j]*s))


@wp.kernel
def k_mv_diag(p: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64),
              dt: wp.float64, out: wp.array(dtype=wp.vec3d), dot_out: wp.array(dtype=wp.float64),
              sk: wp.array(dtype=wp.float64), bslot: wp.array(dtype=int), bK: wp.array(dtype=wp.mat33d)):
    """out = M/dt^2 p + (elastic part already accumulated); fixed vertices masked; accumulates p.out."""
    i = wp.tid()
    if w[i] == F(0.0):
        out[i] = V3(F(0.0), F(0.0), F(0.0))
        return
    o = out[i] + p[i]*(m[i]/(dt*dt) + sk[i])
    if bslot[i] >= 0:
        o = o + bK[bslot[i]]*p[i]
    out[i] = o
    wp.atomic_add(dot_out, 0, wp.dot(p[i], o))


@wp.kernel
def k_vzero(a: wp.array(dtype=wp.vec3d)):
    a[wp.tid()] = V3(F(0.0), F(0.0), F(0.0))


@wp.kernel
def k_szero(a: wp.array(dtype=wp.float64)):
    a[wp.tid()] = F(0.0)


@wp.kernel
def k_precond(r: wp.array(dtype=wp.vec3d), Dinv: wp.array(dtype=wp.mat33), z: wp.array(dtype=wp.vec3d),
              rz: wp.array(dtype=wp.float64), rr: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64)):
    """z = D^-1 r; accumulates r.z and the residual's measure rr = sum of |r_i w_i|^2, each particle's unbalanced
    acceleration (its residual force over its own mass). Measured as |r|^2 against the particles' summed squared
    masses, rr would be dominated by the presser foot's 0.12 kg, and the cloth's own tolerance would be ~400x the
    parameter's."""
    i = wp.tid()
    zi = to64(Dinv[i]*to32(r[i]))
    z[i] = zi
    wp.atomic_add(rz, 0, wp.dot(r[i], zi))
    ai = r[i]*w[i]
    wp.atomic_add(rr, 0, wp.dot(ai, ai))


@wp.kernel
def k_cg_init(z: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p[i] = z[i]


@wp.kernel
def k_cg_step(p: wp.array(dtype=wp.vec3d), Ap: wp.array(dtype=wp.vec3d), dlt: wp.array(dtype=wp.vec3d),
              r: wp.array(dtype=wp.vec3d), rz: wp.array(dtype=wp.float64), pAp: wp.array(dtype=wp.float64),
              mask: wp.array(dtype=wp.mat33d)):
    """The residual stays projected on the axes the solve moves: the forces a guided body takes on its locked axes
    are not residual (they piled up in |r|^2, which then never fell)."""
    i = wp.tid()
    den = pAp[0]
    if den <= F(1e-300):
        return
    alpha = rz[0]/den
    dlt[i] = dlt[i] + p[i]*alpha
    r[i] = r[i] - (mask[i]*Ap[i])*alpha


@wp.kernel
def k_cg_dir(z: wp.array(dtype=wp.vec3d), p: wp.array(dtype=wp.vec3d), rz_new: wp.array(dtype=wp.float64),
             rz: wp.array(dtype=wp.float64)):
    i = wp.tid()
    den = rz[0]
    beta = F(0.0)
    if den > F(1e-300):
        beta = rz_new[0]/den
    p[i] = z[i] + p[i]*beta


@wp.kernel
def k_cg_swap(rz: wp.array(dtype=wp.float64), rz_new: wp.array(dtype=wp.float64), pAp: wp.array(dtype=wp.float64),
              rr: wp.array(dtype=wp.float64), rr_hist: wp.array(dtype=wp.float64), ctl: wp.array(dtype=wp.float64),
              cond: wp.array(dtype=int), abs2: wp.float64, rel2: wp.float64, it_max: int,
              e_hist: wp.array(dtype=wp.float64)):
    """End of a CG iteration: the sums are handed on, the residual recorded, and the solve goes on while the
    residual |r|^2 is above both tolerances (abs2; ctl[7] of the start's, the tick's current relative target) and the
    iteration cap is not reached. With both tolerances zero every solve runs it_max iterations. e_hist: the iteration's decrease of the error's energy
    norm, alpha r.z (diagnostics)."""
    de = F(0.0)
    if pAp[0] > F(1e-300):
        de = rz[0]*rz[0]/pAp[0]
    rz[0] = rz_new[0]
    rz_new[0] = F(0.0)
    pAp[0] = F(0.0)
    c = ctl[0] + F(1.0)
    ctl[0] = c
    ctl[3] = ctl[3] + F(1.0)
    ctl[5] = c
    rr_hist[int(c)] = rr[0]
    e_hist[int(c)] = de
    go = int(1)
    if rr[0] <= wp.max(abs2, ctl[7]*ctl[1]):
        go = int(0)
    if int(c) >= it_max:
        if go == 1:
            ctl[4] = ctl[4] + F(1.0)
        go = int(0)
    cond[0] = go
    rr[0] = F(0.0)


@wp.kernel
def k_cg_rel_reset(ctl: wp.array(dtype=wp.float64), rel2: wp.float64):
    """Each tick's Newton loop starts with the relative target of the inexact Newton."""
    ctl[7] = rel2


@wp.kernel
def k_cg_start(rz_new: wp.array(dtype=wp.float64), pAp: wp.array(dtype=wp.float64), rr: wp.array(dtype=wp.float64),
               rr_hist: wp.array(dtype=wp.float64), ctl: wp.array(dtype=wp.float64), cond: wp.array(dtype=int),
               abs2: wp.float64):
    """Start of a linear solve. ctl[6]: 1 when this solve's target is the absolute tolerance (its relative target,
    ctl[7] of the residual it starts from, is not larger), so that the Newton loop may end on it (inexact Newton);
    1 always without a tolerance (fixed iteration counts)."""
    rz_new[0] = F(0.0)
    pAp[0] = F(0.0)
    rr_hist[0] = rr[0]
    ctl[0] = F(0.0)
    ctl[1] = rr[0]
    ctl[2] = ctl[2] + F(1.0)
    ctl[5] = F(0.0)
    ctl[6] = F(1.0)
    if abs2 >= F(0.0) and ctl[7]*rr[0] > abs2:
        ctl[6] = F(0.0)
    cond[0] = int(1)
    if rr[0] <= abs2:
        cond[0] = int(0)
    rr[0] = F(0.0)


@wp.kernel
def k_trial(x: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), alpha: wp.float64,
            scale: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    out[i] = x[i] + d[i]*(alpha*scale[0])


@wp.kernel
def k_ls_reset(scale: wp.array(dtype=wp.float64)):
    scale[0] = F(1.0)                          # trial scale
    scale[1] = F(0.0)                          # consecutive fully rejected iterations this tick


@wp.kernel
def k_inertia_energy(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3d), m: wp.array(dtype=wp.float64),
                     w: wp.array(dtype=wp.float64), dt: wp.float64, out: wp.array(dtype=wp.float64), slot: int,
                     sk: wp.array(dtype=wp.float64), st: wp.array(dtype=wp.vec3d), bslot: wp.array(dtype=int),
                     bK: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    if w[i] == F(0.0):
        return
    d = x[i] - y[i]
    e = x[i] - st[i]
    E = F(0.5)*m[i]/(dt*dt)*wp.dot(d, d) + F(0.5)*sk[i]*wp.dot(e, e)
    if bslot[i] >= 0:
        E = E + F(0.5)*wp.dot(e, bK[bslot[i]]*e)
    wp.atomic_add(out, slot, E)


@wp.kernel
def k_energy_slot(tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
                  aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2), mat: wp.array(dtype=float, ndim=2),
                  x: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64), slot: int,
                  tf_p: wp.array(dtype=float, ndim=2), tf_F: wp.array(dtype=float, ndim=2)):
    t = wp.tid()
    i0 = tri[t, 0]
    d1 = to32(x[tri[t, 1]] - x[i0])
    d2 = to32(x[tri[t, 2]] - x[i0])
    gw = Dminv[t]*aw[t]
    gf = Dminv[t]*af[t]
    fw = d1*gw[0] + d2*gw[1]
    ff = d1*gf[0] + d2*gf[1]
    A = A0[t]
    ew = wp.length(fw) - 1.0
    ef = wp.length(ff) - 1.0
    E = A*(tension_law32(ew, mat[t, 0], mat[t, 1], mat[t, 2], mat[t, 6])[0] +
           tension_law32(ef, mat[t, 3], mat[t, 4], mat[t, 5], mat[t, 6])[0] +
           tens_fric(ew, t, 0, tf_p, tf_F[t, 0], mat[t, 9])[1] + tens_fric(ef, t, IWAN_N, tf_p, tf_F[t, 1], mat[t, 9])[1])
    sh = wp.dot(fw, ff)
    E = E + 0.5*mat[t, 7]*A*sh*sh + A*tens_fric(sh, t, 2*IWAN_N, tf_p, tf_F[t, 2], mat[t, 11])[1]
    E = E + A*shear_lock_energy(fw, ff, mat[t, 13], mat[t, 14])
    wp.atomic_add(out, slot, wp.float64(E))


@wp.kernel
def k_bend_energy_slot(hinge: wp.array(dtype=int, ndim=2), rest: wp.array(dtype=float),
                       kb: wp.array(dtype=float), x: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64),
                       slot: int, hf_T: wp.array(dtype=float), hf_k: wp.array(dtype=float),
                       hf_p: wp.array(dtype=float, ndim=2)):
    e = wp.tid()
    X0 = x[hinge[e, 0]]
    r = dihedral32(to32(x[hinge[e, 1]] - X0), to32(x[hinge[e, 2]] - X0), to32(x[hinge[e, 3]] - X0))
    C = wrap_pi(r[0] - rest[e])
    f, ef, hs = bend_fric_iwan(r[0], e, hf_p, hf_k[e], hf_T[e])
    wp.atomic_add(out, slot, wp.float64(0.5*kb[e]*C*C + ef))


@wp.kernel
def k_ls_more(phi: wp.array(dtype=wp.float64), more: wp.array(dtype=int)):
    """The shorter line-search trials are evaluated only when the full step raises the potential."""
    more[0] = int(0)
    if phi[1] > phi[0]:
        more[0] = int(1)


@wp.kernel
def k_ls_select(phi: wp.array(dtype=wp.float64), alphas: wp.array(dtype=wp.float64), n_trial: int,
                chosen: wp.array(dtype=wp.float64), hist: wp.array(dtype=wp.float64),
                scale: wp.array(dtype=wp.float64), shrink: wp.float64):
    """Largest step length whose incremental potential does not exceed the start's (slot 0 is the start), the trials
    scaled by scale[0]. When every trial is rejected the step is zero and the next Newton iteration of the tick
    continues the backtracking below the smallest trial (scale *= shrink) instead of repeating the same trials; an
    accepted step restores the full trials. A second consecutive rejection is a stall: the potential cannot be
    lowered along the Newton direction at the resolution of its evaluation (hist[3] counts them; k_step_size ends the
    tick's iterations there)."""
    a = F(0.0)
    for k in range(n_trial):
        if a == F(0.0) and phi[k + 1] <= phi[0]:
            a = alphas[k]
    c = a*scale[0]
    chosen[0] = c
    hist[0] = hist[0] + F(1.0)
    if c < F(1.0):
        hist[1] = hist[1] + F(1.0)
    if a == F(0.0):
        hist[2] = hist[2] + F(1.0)
        scale[0] = scale[0]*shrink
        scale[1] = scale[1] + F(1.0)
        if scale[1] == F(2.0):
            hist[3] = hist[3] + F(1.0)
    else:
        scale[0] = F(1.0)
        scale[1] = F(0.0)


@wp.kernel
def k_step_size(d: wp.array(dtype=wp.vec3d), alpha: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64),
                fc_n: wp.array(dtype=int), fc_mu: wp.array(dtype=float, ndim=2), fc_N: wp.array(dtype=float, ndim=2),
                fc_nrm: wp.array(dtype=wp.vec3, ndim=2), tol_f: wp.float64, tol_v: wp.float64,
                ctl: wp.array(dtype=wp.float64), ls: wp.array(dtype=wp.float64)):
    """ctl[0] <- max over moving particles of the accepted Newton step, normalised: the tangential step on a particle's
    dominant friction contact by tol_f (half the stick zone, so stick and slip are resolved), any step by tol_v
    (a velocity accuracy times dt). Converged when <= 1. A rejected step (every line-search trial raised the
    potential, alpha = 0) is measured by the full Newton step it rejected, so the loop goes on (with finer trials,
    k_ls_select) unless that step was itself within the tolerances; it is not convergence because nothing moved.
    After a second consecutive rejection (a stall, ls[1] = 2) the tick's iterations end, the stall counted."""
    i = wp.tid()
    if w[i] == F(0.0):
        return
    s = d[i]*alpha[0]
    if alpha[0] == F(0.0):
        s = d[i]
        if ls[1] >= F(2.0):
            return
    r = wp.length(s)/tol_v
    n = fc_n[i]
    if n > 0:
        best = int(0)
        bN = float(-1.0)
        for k in range(n):
            g = fc_mu[i, k]*fc_N[i, k]
            if g > bN:
                bN = g
                best = k
        st = tangential(to32(s), fc_nrm[i, best])
        r = wp.max(r, wp.float64(wp.length(st))/tol_f)
    wp.atomic_max(ctl, 0, r)


@wp.kernel
def k_newton_arm(ctl: wp.array(dtype=wp.float64), cond: wp.array(dtype=int), tol: wp.float64, max_extra: int,
                 cg_ctl: wp.array(dtype=wp.float64)):
    """After the mandatory iterations: continue if the last step was larger than tol, or if the last linear solve ended
    on its relative target (inexact Newton: the tick ends only on a step solved to the absolute tolerance, which a
    small loosely solved step asks for by setting the relative target to zero); reset the extra counter."""
    cond[0] = int(0)
    if ctl[0] <= tol and cg_ctl[6] < F(0.5):
        cg_ctl[7] = F(0.0)                     # the step is small but was solved loosely: solve the next one fully
    if (ctl[0] > tol or cg_ctl[6] < F(0.5)) and max_extra > 0:
        cond[0] = int(1)
    ctl[1] = F(0.0)
    ctl[2] = ctl[2] + F(1.0)                   # ticks
    ctl[0] = F(0.0)


@wp.kernel
def k_newton_next(ctl: wp.array(dtype=wp.float64), cond: wp.array(dtype=int), tol: wp.float64, max_extra: int,
                  cg_ctl: wp.array(dtype=wp.float64)):
    """Inside the extra loop (and harmless after mandatory iterations, whose cond is re-armed by k_newton_arm)."""
    if cond[0] == 1:
        ctl[1] = ctl[1] + F(1.0)
        ctl[3] = ctl[3] + F(1.0)               # total extra iterations (diagnostics)
        done = ctl[0] <= tol and cg_ctl[6] > F(0.5)
        if ctl[0] <= tol and cg_ctl[6] < F(0.5):
            cg_ctl[7] = F(0.0)                 # the step is small but was solved loosely: solve the next one fully
        if done or ctl[1] >= wp.float64(max_extra):
            cond[0] = int(0)
            if not done:
                ctl[4] = ctl[4] + F(1.0)       # the tick ends at the cap with its last step above the tolerance
                ctl[5] = wp.max(ctl[5], ctl[0])
    if cond[0] == 1:
        ctl[0] = F(0.0)


@wp.kernel
def k_add_scaled(x: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), alpha: wp.array(dtype=wp.float64)):
    i = wp.tid()
    x[i] = x[i] + d[i]*alpha[0]


@wp.kernel
def k_add(x: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    x[i] = x[i] + d[i]


@wp.kernel
def k_sub(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    out[i] = a[i] - b[i]


@wp.kernel
def k_foot_apply(foot_z: wp.array(dtype=wp.float64), foot_dz: wp.array(dtype=wp.float64),
                 z_floor: wp.array(dtype=wp.float64), foot_acc: wp.array(dtype=wp.float64)):
    z = foot_z[0] + foot_dz[0]
    if z < z_floor[0]:
        z = z_floor[0]
    foot_z[0] = z
    wp.atomic_add(foot_acc, 0, foot_dz[0])
    foot_dz[0] = F(0.0)


@wp.kernel
def k_foot_vel(foot_z: wp.array(dtype=wp.float64), foot_z_prev: wp.array(dtype=wp.float64),
               foot_v: wp.array(dtype=wp.float64), h: wp.float64):
    foot_v[0] = (foot_z[0] - foot_z_prev[0])/h


@wp.kernel
def k_foot_press(x: wp.array(dtype=wp.vec3d), radius: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64),
                 ply: wp.array(dtype=int), is_cloth: wp.array(dtype=int), foot: wp.array(dtype=wp.vec3d),
                 foot_half: wp.array(dtype=wp.vec3d), n_rect: int, zb: wp.array(dtype=wp.float64),
                 dp: wp.array(dtype=wp.float64), pressed: wp.array(dtype=int), mass_sum: wp.array(dtype=wp.float64),
                 top: wp.array(dtype=wp.float64)):
    """Presser-foot sandwich, pass 1: a cloth particle is pressed when it lies under a foot sole (rectangles centre
    `foot`, half extents `foot_half`) and within `depth` (the stack thickness) below the sole's bottom. Sums the
    pressed mass per ply and the stack top under the sole (max z + r, as a fixed-point 1e-9 m integer)."""
    i = wp.tid()
    pressed[i] = 0
    if is_cloth[i] == 0:
        return
    p = x[i]
    inside = int(0)
    for k in range(n_rect):
        c = foot[k]
        hh = foot_half[k]
        if wp.abs(p[0] - c[0]) <= hh[0] and wp.abs(p[1] - c[1]) <= hh[1]:
            inside = 1
    if inside == 0:
        return
    zt = p[2] + radius[i]
    wp.atomic_max(top, 0, zt)
    if zt >= zb[0] - dp[0]:
        pressed[i] = 1
        wp.atomic_add(mass_sum, ply[i], m[i])


@wp.kernel
def k_foot_load(pressed: wp.array(dtype=int), m: wp.array(dtype=wp.float64), ply: wp.array(dtype=int),
                mass_sum: wp.array(dtype=wp.float64), N: wp.array(dtype=wp.float64), load: wp.array(dtype=wp.float64)):
    """Pass 2: the foot's normal force N spread over each ply's pressed particles in proportion to their area
    (mass / areal density): uniform pressure on the pressed patch; every ply of the stack carries the full load."""
    i = wp.tid()
    load[i] = F(0.0)
    if pressed[i] == 1:
        ms = mass_sum[ply[i]]
        if ms > F(0.0):
            load[i] = N[0]*m[i]/ms


@wp.kernel
def k_copy(a: wp.array(dtype=wp.vec3d), b: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    b[i] = a[i]


@wp.kernel
def k_shapes(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), w: wp.array(dtype=wp.float64),
             m: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64), pmu: wp.array(dtype=wp.float64),
             n_shapes: int, s_type: wp.array(dtype=int), s_kind: wp.array(dtype=int),
             s_half: wp.array(dtype=wp.vec3d), s_mu: wp.array(dtype=wp.float64), s_on: wp.array(dtype=int),
             s_p0: wp.array(dtype=wp.vec3d), s_p1: wp.array(dtype=wp.vec3d),
             s_q0: wp.array(dtype=wp.quatd), s_q1: wp.array(dtype=wp.quatd),
             t0: wp.float64, t1: wp.float64, foot_z: wp.array(dtype=wp.float64),
             foot_z_prev: wp.array(dtype=wp.float64), foot_w: wp.float64, foot_dz: wp.array(dtype=wp.float64),
             acc_f: wp.array(dtype=wp.vec3d), acc_tau: wp.array(dtype=wp.vec3d), acc_n: wp.array(dtype=wp.float64),
             contact_flag: wp.array(dtype=int), load: wp.array(dtype=wp.float64), s_loaded: wp.array(dtype=int)):
    """Particle vs analytic shapes with Coulomb friction (position-based, Macklin et al. 2014). Shape poses are
    interpolated between the tick's start (p0, q0) and end (p1, q1) at substep fractions t0 -> t1. Friction uses the
    displacement of the contacted shape point over the substep, so a moving feed dog or pad carries the cloth.
    The Coulomb limit is mu times the normal impulse: the penetration correction for ordinary contacts, plus, for the
    shapes of the presser-foot sandwich (s_loaded: foot soles, feed dog, throat plate), the presser-foot load the
    particle carries (load, N) over the step."""
    i = wp.tid()
    wi = w[i]
    zero = F(0.0)
    if wi == zero:
        return
    xi = x[i]
    r = radius[i]
    flag = int(0)
    for s in range(n_shapes):
        if s_on[s] == 0:
            continue
        p_a = wp.lerp(s_p0[s], s_p1[s], t0)
        p_b = wp.lerp(s_p0[s], s_p1[s], t1)
        q_a = quat_lerp(s_q0[s], s_q1[s], t0)
        q_b = quat_lerp(s_q0[s], s_q1[s], t1)
        kind = s_kind[s]
        if kind == KIND_FOOT:
            p_a = p_a + V3(zero, zero, foot_z_prev[0])
            p_b = p_b + V3(zero, zero, foot_z[0])
        ql = wp.quat_rotate_inv(q_b, xi - p_b)
        sd = shape_sdf(s_type[s], s_half[s], ql)
        d = sd[0]
        pen = r - d
        if pen <= zero:
            continue
        flag = 1
        nw = wp.quat_rotate(q_b, sd[1])
        wf = zero
        if kind == KIND_FOOT:
            wf = foot_w*nw[2]*nw[2]
        share = wi/(wi + wf)
        dxn = nw*(pen*share)
        loc = ql - sd[1]*d
        s_disp = (p_b + wp.quat_rotate(q_b, loc)) - (p_a + wp.quat_rotate(q_a, loc))
        rel = (xi + dxn - x_prev[i]) - s_disp
        rel_t = rel - nw*wp.dot(rel, nw)
        lt = wp.length(rel_t)
        mu = s_mu[s]*pmu[i]          # shape mu is the pair value for the scene's fabric; pmu a per-material factor
        dxt = V3(zero, zero, zero)
        sandwich = 0
        if s_loaded[s] == 1 and load[i] > zero:
            sandwich = 1
        if lt > zero and sandwich == 0:
            # sandwich contacts of loaded particles get their friction in the implicit step (k_fric_grad)
            lim = mu*pen*share
            if lt <= lim:
                dxt = -rel_t
            else:
                dxt = -rel_t*(lim/lt)
        dx = dxn + dxt
        xi = xi + dx
        if kind == KIND_FOOT:
            wp.atomic_add(foot_dz, 0, -nw[2]*pen*(F(1.0) - share))
        imp = dx*m[i]
        wp.atomic_add(acc_f, s, imp)
        wp.atomic_add(acc_tau, s, wp.cross(xi - p_b, imp))
        wp.atomic_add(acc_n, s, wp.dot(imp, nw))
    x[i] = xi
    if flag == 1:
        contact_flag[i] = 1


@wp.kernel
def k_to32(x: wp.array(dtype=wp.vec3d), y: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = x[i]
    y[i] = wp.vec3(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))


@wp.func
def _cc_pair_out(v: int, f: int, bary: wp.vec3d, x: wp.array(dtype=wp.vec3d), tri: wp.array(dtype=int, ndim=2),
                 pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                 pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int):
    xa = x[tri[f, 0]]
    n = wp.cross(x[tri[f, 1]] - xa, x[tri[f, 2]] - xa)
    p = xa*bary[0] + x[tri[f, 1]]*bary[1] + x[tri[f, 2]]*bary[2]
    side = wp.sign(wp.dot(x[v] - p, n))
    if side == F(0.0):
        side = F(1.0)
    k = wp.atomic_add(count, 0, 1)
    if k < capacity:
        pair_v[k] = v
        pair_f[k] = f
        pair_b[k] = bary
        pair_side[k] = side


@wp.kernel
def k_cloth_vmax(v: wp.array(dtype=wp.vec3d), is_cloth: wp.array(dtype=int), out: wp.array(dtype=wp.float64)):
    """The fastest cloth vertex's speed at the tick start (out[0], zeroed before): the reach of ply-on-ply detection."""
    i = wp.tid()
    if is_cloth[i] == 1:
        wp.atomic_max(out, 0, wp.length(v[i]))


@wp.kernel
def k_detect_cc(x: wp.array(dtype=wp.vec3d), Xm: wp.array(dtype=wp.vec2d), ply: wp.array(dtype=int),
                excl: wp.array(dtype=wp.float64), is_cloth: wp.array(dtype=int),
                mesh: wp.uint64, tri: wp.array(dtype=int, ndim=2), query_r: wp.float64,
                pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
                pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int,
                prev_f: wp.array(dtype=int, ndim=2), keep: wp.float64, vmax: wp.array(dtype=wp.float64),
                reach_dt: wp.float64, reach_max: wp.float64):
    """Vertex-triangle contact pairs, once per tick: for each vertex, the closest face of each ply within query_r plus
    the reach of the fastest cloth over the tick (vmax[0] reach_dt, at most reach_max: cloth moving metres per second
    crosses more than the free thickness in a tick, and a pair not found at the tick start does not exist in it) (up
    to three plies, its own included for folds), and of each such ply the closest face on the vertex's other side if
    there is one (a vertex between two layers of a folded ply). Persistent: last tick's face of a ply is kept unless
    another face is closer by more than `keep` (a closest point at a vertex or edge of the other ply is equidistant from the faces
    around it, and choosing among them anew every tick alternated the faces and the force's small shares on their
    other vertices: a period-2 flicker under the presser foot). Same-ply faces whose closest point is within the
    vertex's rest-state exclusion radius (material coordinates) are its own neighbourhood and skipped, so folds of one ply still collide
    while the flat sheet does not fight itself. One pair per facing surface: the faces around a vertex of the other ply
    all share that closest point, and pairing with each of them multiplied the contact stiffness ~10 times without
    changing the force, which the conjugate gradient could not resolve in 40 iterations."""
    v = wp.tid()
    if is_cloth[v] == 0:
        return
    xv = x[v]
    qr64 = query_r + wp.min(vmax[0]*reach_dt, reach_max)
    qr = wp.float32(qr64)
    xv32 = wp.vec3(wp.float32(xv[0]), wp.float32(xv[1]), wp.float32(xv[2]))
    lo = xv32 - wp.vec3(qr, qr, qr)
    hi = xv32 + wp.vec3(qr, qr, qr)
    q = wp.mesh_query_aabb(mesh, lo, hi)
    f = int(0)
    big = F(1.0e9)
    p0 = int(-1)
    f0 = int(-1)
    d0 = big
    b0 = V3(F(0.0), F(0.0), F(0.0))
    p1 = int(-1)
    f1 = int(-1)
    d1 = big
    b1 = V3(F(0.0), F(0.0), F(0.0))
    p2 = int(-1)
    f2 = int(-1)
    d2 = big
    b2 = V3(F(0.0), F(0.0), F(0.0))
    while wp.mesh_query_aabb_next(q, f):
        a = tri[f, 0]
        b = tri[f, 1]
        c = tri[f, 2]
        if a == v or b == v or c == v:
            continue
        xa = x[a]
        xb = x[b]
        xc = x[c]
        bary = closest_bary(xa, xb, xc, xv)
        p = xa*bary[0] + xb*bary[1] + xc*bary[2]
        dist = wp.length(xv - p)
        if dist > qr64:
            continue
        pf = ply[a]
        if pf == ply[v]:
            Xp = Xm[a]*bary[0] + Xm[b]*bary[1] + Xm[c]*bary[2]
            if wp.length(Xp - Xm[v]) < excl[v]:
                continue
        nrm = wp.cross(xb - xa, xc - xa)
        if wp.length(nrm) < F(1e-14):
            continue
        if pf < MAX_PLY_SLOTS:
            if prev_f[v, pf] == f:
                dist = dist - keep
        if pf == p0 or p0 < 0:
            if dist < d0:
                p0 = pf
                f0 = f
                d0 = dist
                b0 = bary
        elif pf == p1 or p1 < 0:
            if dist < d1:
                p1 = pf
                f1 = f
                d1 = dist
                b1 = bary
        elif pf == p2 or p2 < 0:
            if dist < d2:
                p2 = pf
                f2 = f
                d2 = dist
                b2 = bary
    # A second face of a ply: on the other side of the vertex (the vertex inside a fold of that ply, between two of
    # its layers), or seen face-on in a direction turned by more than 30 deg (the other wall of a crease the vertex
    # sits in). With one face per ply only the nearer layer or wall was a contact, the vertex crossed toward the other
    # unresisted within the tick, and the next tick the other was the nearer: it bounced between them (period 2,
    # 130 um a tick; 44 mJ of contact energy appearing over 300 ticks of the E19 replay at 0.25 ms, 23 mJ with the
    # opposite side alone). The faces around the first face's closest point share it and are not counted twice
    # (their closest point is on their edge), and a gently curved sheet (< 30 deg) keeps one contact.
    g0 = int(-1)
    g1 = int(-1)
    g2 = int(-1)
    e0 = big
    e1 = big
    e2 = big
    c0 = V3(F(0.0), F(0.0), F(0.0))
    c1 = V3(F(0.0), F(0.0), F(0.0))
    c2 = V3(F(0.0), F(0.0), F(0.0))
    r0 = V3(F(0.0), F(0.0), F(0.0))
    r1 = V3(F(0.0), F(0.0), F(0.0))
    r2 = V3(F(0.0), F(0.0), F(0.0))
    if f0 >= 0:
        r0 = xv - (x[tri[f0, 0]]*b0[0] + x[tri[f0, 1]]*b0[1] + x[tri[f0, 2]]*b0[2])
    if f1 >= 0:
        r1 = xv - (x[tri[f1, 0]]*b1[0] + x[tri[f1, 1]]*b1[1] + x[tri[f1, 2]]*b1[2])
    if f2 >= 0:
        r2 = xv - (x[tri[f2, 0]]*b2[0] + x[tri[f2, 1]]*b2[1] + x[tri[f2, 2]]*b2[2])
    tiny = F(1e-9)
    if wp.length(r0) > tiny or wp.length(r1) > tiny or wp.length(r2) > tiny:
        q2 = wp.mesh_query_aabb(mesh, lo, hi)
        f = int(0)
        while wp.mesh_query_aabb_next(q2, f):
            a = tri[f, 0]
            b = tri[f, 1]
            c = tri[f, 2]
            if a == v or b == v or c == v:
                continue
            xa = x[a]
            xb = x[b]
            xc = x[c]
            if wp.length(wp.cross(xb - xa, xc - xa)) < F(1e-14):
                continue
            bary = closest_bary(xa, xb, xc, xv)
            r = xv - (xa*bary[0] + xb*bary[1] + xc*bary[2])
            dist = wp.length(r)
            if dist > query_r:
                continue
            pf = ply[a]
            if pf == ply[v]:
                Xp = Xm[a]*bary[0] + Xm[b]*bary[1] + Xm[c]*bary[2]
                if wp.length(Xp - Xm[v]) < excl[v]:
                    continue
            # a face seen face-on (its closest point inside it, not on an edge it shares) whose direction turns by
            # more than 30 deg from the first face's: the other wall of a fold or crease
            inner = wp.min(wp.min(bary[0], bary[1]), bary[2]) > F(1e-6)
            if pf == p0 and f != f0 and wp.length(r0) > tiny and dist > tiny:
                cs = wp.dot(r, r0)/(dist*wp.length(r0))
                if (cs < F(0.0) or (inner and cs < F(CC_SECOND_FACE_COS))) and dist < e0:
                    g0 = f
                    e0 = dist
                    c0 = bary
            elif pf == p1 and f != f1 and wp.length(r1) > tiny and dist > tiny:
                cs = wp.dot(r, r1)/(dist*wp.length(r1))
                if (cs < F(0.0) or (inner and cs < F(CC_SECOND_FACE_COS))) and dist < e1:
                    g1 = f
                    e1 = dist
                    c1 = bary
            elif pf == p2 and f != f2 and wp.length(r2) > tiny and dist > tiny:
                cs = wp.dot(r, r2)/(dist*wp.length(r2))
                if (cs < F(0.0) or (inner and cs < F(CC_SECOND_FACE_COS))) and dist < e2:
                    g2 = f
                    e2 = dist
                    c2 = bary
    for j in range(MAX_PLY_SLOTS):
        prev_f[v, j] = -1
    if g0 >= 0:
        _cc_pair_out(v, g0, c0, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
    if g1 >= 0:
        _cc_pair_out(v, g1, c1, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
    if g2 >= 0:
        _cc_pair_out(v, g2, c2, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
    if f0 >= 0:
        _cc_pair_out(v, f0, b0, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
        if p0 < MAX_PLY_SLOTS:
            prev_f[v, p0] = f0
    if f1 >= 0:
        _cc_pair_out(v, f1, b1, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
        if p1 < MAX_PLY_SLOTS:
            prev_f[v, p1] = f1
    if f2 >= 0:
        _cc_pair_out(v, f2, b2, x, tri, pair_v, pair_f, pair_b, pair_side, count, capacity)
        if p2 < MAX_PLY_SLOTS:
            prev_f[v, p2] = f2


@wp.kernel
def k_solve_cc(pair_v: wp.array(dtype=int), pair_f: wp.array(dtype=int), pair_b: wp.array(dtype=wp.vec3d),
               pair_side: wp.array(dtype=wp.float64), count: wp.array(dtype=int), capacity: int,
               tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
               w: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64), mu_cc: wp.float64,
               dx: wp.array(dtype=wp.vec3d), dn: wp.array(dtype=wp.float64), load: wp.array(dtype=wp.float64),
               hstep: wp.float64):
    k = wp.tid()
    if k >= wp.min(count[0], capacity):
        return
    zero = F(0.0)
    v = pair_v[k]
    f = pair_f[k]
    a = tri[f, 0]
    b = tri[f, 1]
    c = tri[f, 2]
    bb = pair_b[k]
    xa = x[a]
    xb = x[b]
    xc = x[c]
    p = xa*bb[0] + xb*bb[1] + xc*bb[2]
    n = wp.cross(xb - xa, xc - xa)
    ln = wp.length(n)
    if ln < F(1e-14):
        return
    n = n*(pair_side[k]/ln)
    thick = radius[v] + (radius[a]*bb[0] + radius[b]*bb[1] + radius[c]*bb[2])
    d = wp.dot(x[v] - p, n)
    pen = thick - d
    if pen <= zero:
        return
    wv = w[v]
    wf = w[a]*bb[0]*bb[0] + w[b]*bb[1]*bb[1] + w[c]*bb[2]*bb[2]
    wsum = wv + wf
    if wsum <= zero:
        return
    lam = pen/wsum
    pp = x_prev[a]*bb[0] + x_prev[b]*bb[1] + x_prev[c]*bb[2]
    rel = (x[v] - x_prev[v]) - (p - pp)
    rel_t = rel - n*wp.dot(rel, n)
    lt = wp.length(rel_t)
    jt = V3(zero, zero, zero)
    if lt > zero and load[v] <= zero:
        # loaded pairs (inside the presser-foot sandwich) get their friction in the implicit step (k_ccfric_grad)
        lim = mu_cc*pen
        if lt <= lim:
            jt = -rel_t/wsum
        else:
            jt = -rel_t*(lim/lt)/wsum
    j = n*lam + jt
    one = F(1.0)
    wp.atomic_add(dx, v, j*wv)
    wp.atomic_add(dn, v, one)
    wp.atomic_add(dx, a, -j*(w[a]*bb[0]))
    wp.atomic_add(dn, a, one)
    wp.atomic_add(dx, b, -j*(w[b]*bb[1]))
    wp.atomic_add(dn, b, one)
    wp.atomic_add(dx, c, -j*(w[c]*bb[2]))
    wp.atomic_add(dn, c, one)


@wp.kernel
def k_apply_jacobi(x: wp.array(dtype=wp.vec3d), dx: wp.array(dtype=wp.vec3d), dn: wp.array(dtype=wp.float64),
                   relax: wp.float64):
    i = wp.tid()
    n = dn[i]
    if n > F(0.0):
        x[i] = x[i] + dx[i]*(relax/n)
    dx[i] = V3(F(0.0), F(0.0), F(0.0))
    dn[i] = F(0.0)


@wp.func
def con_point(f: int, b: wp.vec3d, tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d)):
    return x[tri[f, 0]]*b[0] + x[tri[f, 1]]*b[1] + x[tri[f, 2]]*b[2]


@wp.func
def con_weight(f: int, b: wp.vec3d, tri: wp.array(dtype=int, ndim=2), w: wp.array(dtype=wp.float64)):
    return w[tri[f, 0]]*b[0]*b[0] + w[tri[f, 1]]*b[1]*b[1] + w[tri[f, 2]]*b[2]*b[2]


@wp.func
def con_push(f: int, b: wp.vec3d, j: wp.vec3d, tri: wp.array(dtype=int, ndim=2), w: wp.array(dtype=wp.float64),
             dx: wp.array(dtype=wp.vec3d), dn: wp.array(dtype=wp.float64)):
    for k in range(3):
        i = tri[f, k]
        wp.atomic_add(dx, i, j*(w[i]*b[k]))
        wp.atomic_add(dn, i, F(1.0))


@wp.func
def con_terms(typ: int, r: wp.vec3d, uz: wp.float64, k: wp.float64, rest: wp.float64, cap: wp.float64,
              eps: wp.float64, act: wp.float64):
    """Energy, gradient w.r.t. the constraint's relative vector r and its (PSD) Hessian, per type:
    NEEDLE: r = hole - needle axis; lateral spring k (xy) + axial coupling as smoothed Coulomb capped at `cap` on the
            hole's axial slip uz relative to the needle over the step;
    BIND, THREAD: r = B - A; tension-only spring k beyond `rest` (interlock: the thread loop around the stack;
            thread element: EA / rest); with cap > 0 its tension is capped there (Huber energy): an interlock being
            drawn in by the take-up pulls with at most the thread's tension (machine._draw_interlocks);
    TETHER: r = anchor - A; spring k capped at `cap` (the thread paid out at its set tension): Huber energy."""
    zero = F(0.0)
    one = F(1.0)
    e = zero
    g = V3(zero, zero, zero)
    H = wp.mat33(0.0)
    if typ == CON_NEEDLE:
        e = F(0.5)*k*(r[0]*r[0] + r[1]*r[1])
        g = V3(k*r[0], k*r[1], zero)
        au = wp.abs(uz)
        e = e + cap*wp.float64(ipc_f0(float(au), float(eps)))
        c = cap*wp.float64(ipc_f1_over_y(float(au), float(eps)))
        g = g + V3(zero, zero, c*uz)
        H = wp.mat33(float(k), 0.0, 0.0, 0.0, float(k), 0.0, 0.0, 0.0, float(c))
    else:
        L = wp.length(r)
        if L > F(1e-12):
            n = r/L
            n32 = to32(n)
            nn = wp.outer(n32, n32)
            C = L - rest
            if typ == CON_TETHER:
                if C > zero:
                    if k*C <= cap:
                        e = F(0.5)*k*C*C
                        g = n*(k*C)
                        H = nn*float(k) + (wp.identity(n=3, dtype=float) - nn)*float(k*C/L)
                    else:
                        e = cap*C - cap*cap/(F(2.0)*k)
                        g = n*cap
                        H = (wp.identity(n=3, dtype=float) - nn)*float(cap/L)
            else:
                if cap > zero and k*C > cap:
                    e = cap*C - cap*cap/(F(2.0)*k)
                    g = n*cap
                    H = (wp.identity(n=3, dtype=float) - nn)*float(cap/L)
                elif C > -act:
                    H = nn*float(k)
                    if C > zero:
                        e = F(0.5)*k*C*C
                        g = n*(k*C)
                        H = H + (wp.identity(n=3, dtype=float) - nn)*float(k*C/L)
    return e, g, H


@wp.func
def con_rel(k: int, typ: int, c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d),
            c_fb: wp.array(dtype=int), c_bb: wp.array(dtype=wp.vec3d), c_pb: wp.array(dtype=int),
            tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d)):
    pa = con_point(c_fa[k], c_ba[k], tri, x)
    if typ == CON_NEEDLE:
        return pa
    if typ == CON_TETHER:
        return x[c_pb[k]] - pa
    return con_point(c_fb[k], c_bb[k], tri, x) - pa


@wp.func
def con_implicit(typ: int):
    return typ == CON_NEEDLE or typ == CON_BIND or typ == CON_THREAD or typ == CON_TETHER


@wp.func
def con_eval(k: int, typ: int, c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d), c_fb: wp.array(dtype=int),
             c_bb: wp.array(dtype=wp.vec3d), c_pa: wp.array(dtype=int), c_pb: wp.array(dtype=int),
             c_rest: wp.array(dtype=wp.float64), c_comp: wp.array(dtype=wp.float64), c_cap: wp.array(dtype=wp.float64),
             axis_p: wp.array(dtype=wp.vec3d), axis_v: wp.array(dtype=wp.vec3d), tri: wp.array(dtype=int, ndim=2),
             x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), dt: wp.float64, eps: wp.float64,
             act: wp.float64):
    r = con_rel(k, typ, c_fa, c_ba, c_fb, c_bb, c_pb, tri, x)
    uz = F(0.0)
    if typ == CON_NEEDLE:
        ax = axis_p[c_pa[k]]
        r = V3(r[0] - ax[0], r[1] - ax[1], F(0.0))
        dp = con_point(c_fa[k], c_ba[k], tri, x) - con_point(c_fa[k], c_ba[k], tri, x_prev)
        uz = dp[2] - axis_v[c_pa[k]][2]*dt
    return con_terms(typ, r, uz, F(1.0)/c_comp[k], c_rest[k], c_cap[k], eps, act)


@wp.kernel
def k_con_grad(n_con_arr: wp.array(dtype=int), c_type: wp.array(dtype=int), c_on: wp.array(dtype=int),
               c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d), c_fb: wp.array(dtype=int),
               c_bb: wp.array(dtype=wp.vec3d), c_pa: wp.array(dtype=int), c_pb: wp.array(dtype=int),
               c_rest: wp.array(dtype=wp.float64), c_comp: wp.array(dtype=wp.float64),
               c_cap: wp.array(dtype=wp.float64), axis_p: wp.array(dtype=wp.vec3d), axis_v: wp.array(dtype=wp.vec3d),
               tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
               dt: wp.float64, eps: wp.float64, act: wp.float64, grad: wp.array(dtype=wp.vec3d),
               D: wp.array(dtype=wp.mat33), con_H: wp.array(dtype=wp.mat33)):
    """The sewing constraints inside the implicit step (gradient, Hessian data, preconditioner blocks)."""
    k = wp.tid()
    con_H[k] = wp.mat33(0.0)
    if k >= n_con_arr[0] or c_on[k] == 0:
        return
    typ = c_type[k]
    if not con_implicit(typ):
        return
    e, g, H = con_eval(k, typ, c_fa, c_ba, c_fb, c_bb, c_pa, c_pb, c_rest, c_comp, c_cap, axis_p, axis_v, tri, x,
                       x_prev, dt, eps, act)
    con_H[k] = H
    sa = F(-1.0)
    if typ == CON_NEEDLE:
        sa = F(1.0)
    for j in range(3):
        cj = sa*c_ba[k][j]
        i = tri[c_fa[k], j]
        wp.atomic_add(grad, i, g*cj)
        wp.atomic_add(D, i, H*float(cj*cj))
    if typ == CON_BIND or typ == CON_THREAD:
        for j in range(3):
            cj = c_bb[k][j]
            i = tri[c_fb[k], j]
            wp.atomic_add(grad, i, g*cj)
            wp.atomic_add(D, i, H*float(cj*cj))
    if typ == CON_TETHER:
        wp.atomic_add(grad, c_pb[k], g)
        wp.atomic_add(D, c_pb[k], H)


@wp.kernel
def k_mv_con(n_con_arr: wp.array(dtype=int), c_type: wp.array(dtype=int), c_on: wp.array(dtype=int),
             c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d), c_fb: wp.array(dtype=int),
             c_bb: wp.array(dtype=wp.vec3d), c_pb: wp.array(dtype=int), tri: wp.array(dtype=int, ndim=2),
             con_H: wp.array(dtype=wp.mat33), p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    if k >= n_con_arr[0] or c_on[k] == 0:
        return
    typ = c_type[k]
    if not con_implicit(typ):
        return
    sa = F(-1.0)
    if typ == CON_NEEDLE:
        sa = F(1.0)
    rp = wp.vec3(0.0)
    for j in range(3):
        rp = rp + to32(p[tri[c_fa[k], j]])*float(sa*c_ba[k][j])
    if typ == CON_BIND or typ == CON_THREAD:
        for j in range(3):
            rp = rp + to32(p[tri[c_fb[k], j]])*float(c_bb[k][j])
    if typ == CON_TETHER:
        rp = rp + to32(p[c_pb[k]])
    q = con_H[k]*rp
    for j in range(3):
        wp.atomic_add(out, tri[c_fa[k], j], to64(q*float(sa*c_ba[k][j])))
    if typ == CON_BIND or typ == CON_THREAD:
        for j in range(3):
            wp.atomic_add(out, tri[c_fb[k], j], to64(q*float(c_bb[k][j])))
    if typ == CON_TETHER:
        wp.atomic_add(out, c_pb[k], to64(q))


@wp.kernel
def k_con_energy(n_con_arr: wp.array(dtype=int), c_type: wp.array(dtype=int), c_on: wp.array(dtype=int),
                 c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d), c_fb: wp.array(dtype=int),
                 c_bb: wp.array(dtype=wp.vec3d), c_pa: wp.array(dtype=int), c_pb: wp.array(dtype=int),
                 c_rest: wp.array(dtype=wp.float64), c_comp: wp.array(dtype=wp.float64),
                 c_cap: wp.array(dtype=wp.float64), axis_p: wp.array(dtype=wp.vec3d),
                 axis_v: wp.array(dtype=wp.vec3d), tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d),
                 x_prev: wp.array(dtype=wp.vec3d), dt: wp.float64, eps: wp.float64,
                 out: wp.array(dtype=wp.float64), slot: int):
    k = wp.tid()
    if k >= n_con_arr[0] or c_on[k] == 0:
        return
    typ = c_type[k]
    if not con_implicit(typ):
        return
    e, g, H = con_eval(k, typ, c_fa, c_ba, c_fb, c_bb, c_pa, c_pb, c_rest, c_comp, c_cap, axis_p, axis_v, tri, x,
                       x_prev, dt, eps, F(0.0))
    if e > F(0.0):
        wp.atomic_add(out, slot, e)


@wp.kernel
def k_con_force(n_con_arr: wp.array(dtype=int), c_type: wp.array(dtype=int), c_on: wp.array(dtype=int),
                c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d), c_fb: wp.array(dtype=int),
                c_bb: wp.array(dtype=wp.vec3d), c_pa: wp.array(dtype=int), c_pb: wp.array(dtype=int),
                c_rest: wp.array(dtype=wp.float64), c_comp: wp.array(dtype=wp.float64),
                c_cap: wp.array(dtype=wp.float64), axis_p: wp.array(dtype=wp.vec3d),
                axis_v: wp.array(dtype=wp.vec3d), tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d),
                x_prev: wp.array(dtype=wp.vec3d), dt: wp.float64, eps: wp.float64,
                c_force: wp.array(dtype=wp.vec3d)):
    """Force each constraint applies on the cloth at its point A after the solved step (needle: on the hole; bind,
    thread, tether: the tension vector pulling A toward B)."""
    k = wp.tid()
    if k >= n_con_arr[0] or c_on[k] == 0:
        return
    typ = c_type[k]
    if not con_implicit(typ):
        return
    e, g, H = con_eval(k, typ, c_fa, c_ba, c_fb, c_bb, c_pa, c_pb, c_rest, c_comp, c_cap, axis_p, axis_v, tri, x,
                       x_prev, dt, eps, F(0.0))
    if typ == CON_NEEDLE:
        c_force[k] = -g
    else:
        c_force[k] = g


@wp.kernel
def k_constraints(n_con_arr: wp.array(dtype=int), c_type: wp.array(dtype=int), c_on: wp.array(dtype=int),
                  c_fa: wp.array(dtype=int), c_ba: wp.array(dtype=wp.vec3d),
                  c_fb: wp.array(dtype=int), c_bb: wp.array(dtype=wp.vec3d),
                  c_pa: wp.array(dtype=int), c_pb: wp.array(dtype=int),
                  c_rest: wp.array(dtype=wp.float64), c_comp: wp.array(dtype=wp.float64),
                  c_cap: wp.array(dtype=wp.float64),
                  axis_p: wp.array(dtype=wp.vec3d), axis_v: wp.array(dtype=wp.vec3d),
                  tri: wp.array(dtype=int, ndim=2), x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d),
                  w: wp.array(dtype=wp.float64), h: wp.float64,
                  dx: wp.array(dtype=wp.vec3d), dn: wp.array(dtype=wp.float64),
                  c_force: wp.array(dtype=wp.vec3d), skip_implicit: int):
    """All dynamic constraints, Jacobi with averaging. c_force accumulates, per constraint, the force it applied on
    the cloth summed over the substeps of the tick (N x substeps): the needle's lateral and axial load, a thread's
    tension, an interlock's bind force."""
    k = wp.tid()
    if k >= n_con_arr[0] or c_on[k] == 0:
        return
    typ = c_type[k]
    if skip_implicit == 1 and con_implicit(typ):
        return
    zero = F(0.0)
    one = F(1.0)
    a_hh = one/(h*h)
    if typ == CON_NEEDLE:
        nf = c_fa[k]
        nb = c_ba[k]
        np_ = con_point(nf, nb, tri, x)
        nws = con_weight(nf, nb, tri, w)
        if nws <= zero:
            return
        nax = axis_p[c_pa[k]]
        nC = V3(np_[0] - nax[0], np_[1] - nax[1], zero)
        nlam = -nC/(nws + c_comp[k]*a_hh)
        nvz = axis_v[c_pa[k]][2]
        npp = con_point(nf, nb, tri, x_prev)
        nrel = (np_[2] - npp[2]) - nvz*h
        njz = -nrel/nws
        ncap = c_cap[k]*h*h
        if wp.abs(njz) > ncap:
            njz = wp.sign(njz)*ncap
        nj = V3(nlam[0], nlam[1], njz)
        con_push(nf, nb, nj, tri, w, dx, dn)
        wp.atomic_add(c_force, k, nj*a_hh)
    elif typ == CON_BIND or typ == CON_THREAD:
        pa = con_point(c_fa[k], c_ba[k], tri, x)
        pb = con_point(c_fb[k], c_bb[k], tri, x)
        wa = con_weight(c_fa[k], c_ba[k], tri, w)
        wb = con_weight(c_fb[k], c_bb[k], tri, w)
        bd = pb - pa
        bL = wp.length(bd)
        if bL < F(1e-12) or wa + wb <= zero:
            return
        bC = bL - c_rest[k]
        if bC <= zero:
            return
        bn = bd/bL
        blam = -bC/(wa + wb + c_comp[k]*a_hh)
        bj = bn*blam
        con_push(c_fa[k], c_ba[k], -bj, tri, w, dx, dn)
        con_push(c_fb[k], c_bb[k], bj, tri, w, dx, dn)
        wp.atomic_add(c_force, k, bn*(-blam*a_hh))
    elif typ == CON_PINCROSS:
        cf = c_fa[k]
        cb = c_ba[k]
        cp = con_point(cf, cb, tri, x)
        hd = x[c_pa[k]]
        tp = x[c_pb[k]]
        sv = tp - hd
        sL = wp.length(sv)
        if sL < F(1e-9):
            return
        u = sv/sL
        tt = wp.clamp(wp.dot(cp - hd, u), zero, sL)
        fr = tt/sL
        cq = hd + u*tt
        cr = cp - cq
        rperp = cr - u*wp.dot(cr, u)
        cwp = con_weight(cf, cb, tri, w)
        cwh = w[c_pa[k]]*(one - fr)*(one - fr) + w[c_pb[k]]*fr*fr
        cws = cwp + cwh
        if cws <= zero:
            return
        clam = -rperp/(cws + c_comp[k]*a_hh)
        cpp = con_point(cf, cb, tri, x_prev)
        cqp = x_prev[c_pa[k]]*(one - fr) + x_prev[c_pb[k]]*fr
        crel = wp.dot((cp - cpp) - (cq - cqp), u)
        cja = -crel/cws
        clim = c_cap[k]*h*h
        if wp.abs(cja) > clim:
            cja = wp.sign(cja)*clim
        cj = clam + u*cja
        con_push(cf, cb, cj, tri, w, dx, dn)
        wp.atomic_add(dx, c_pa[k], -cj*(w[c_pa[k]]*(one - fr)))
        wp.atomic_add(dn, c_pa[k], one)
        wp.atomic_add(dx, c_pb[k], -cj*(w[c_pb[k]]*fr))
        wp.atomic_add(dn, c_pb[k], one)
        wp.atomic_add(c_force, k, cj*a_hh)
    elif typ == CON_TETHER:
        tf_ = c_fa[k]
        tb_ = c_ba[k]
        tp_ = con_point(tf_, tb_, tri, x)
        tw_ = con_weight(tf_, tb_, tri, w)
        tq_ = x[c_pb[k]]
        td_ = tq_ - tp_
        tL_ = wp.length(td_)
        if tL_ < F(1e-9) or tw_ <= zero:
            return
        tC_ = tL_ - c_rest[k]
        if tC_ <= zero:
            return
        tn_ = td_/tL_
        tlam_ = tC_/(tw_ + c_comp[k]*a_hh)
        tcap_ = c_cap[k]*h*h
        if tlam_ > tcap_:
            tlam_ = tcap_
        tj_ = tn_*tlam_
        con_push(tf_, tb_, tj_, tri, w, dx, dn)
        wp.atomic_add(c_force, k, tj_*a_hh)
    elif typ == CON_ROD:
        ri = c_pa[k]
        rj = c_pb[k]
        rd = x[rj] - x[ri]
        rL = wp.length(rd)
        rws = w[ri] + w[rj]
        if rL < F(1e-12) or rws <= zero:
            return
        rC = rL - c_rest[k]
        rn = rd/rL
        rlam = -rC/(rws + c_comp[k]*a_hh)
        wp.atomic_add(dx, ri, -rn*(rlam*w[ri]))
        wp.atomic_add(dn, ri, one)
        wp.atomic_add(dx, rj, rn*(rlam*w[rj]))
        wp.atomic_add(dn, rj, one)


@wp.kernel
def k_velocity(x: wp.array(dtype=wp.vec3d), x_prev: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d),
               w: wp.array(dtype=wp.float64), h: wp.float64, v_start: wp.array(dtype=wp.vec3d),
               a_prev: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if w[i] > F(0.0):
        vn = (x[i] - x_prev[i])/h
        v[i] = vn
        a_prev[i] = (vn - v_start[i])/h
    else:
        v[i] = V3(F(0.0), F(0.0), F(0.0))
        a_prev[i] = V3(F(0.0), F(0.0), F(0.0))


@wp.kernel
def k_foot_predict(foot_z: wp.array(dtype=wp.float64), foot_z_prev: wp.array(dtype=wp.float64),
                   foot_v: wp.array(dtype=wp.float64), force: wp.array(dtype=wp.float64), mass: wp.float64,
                   g: wp.float64, h: wp.float64):
    foot_z_prev[0] = foot_z[0]
    vz = foot_v[0] + h*(force[0]/mass - g)
    foot_v[0] = vz
    foot_z[0] = foot_z[0] + h*vz


@wp.kernel
def k_foot_finish(foot_z: wp.array(dtype=wp.float64), foot_z_prev: wp.array(dtype=wp.float64),
                  foot_v: wp.array(dtype=wp.float64), foot_dz: wp.array(dtype=wp.float64),
                  z_floor: wp.array(dtype=wp.float64), h: wp.float64, foot_acc: wp.array(dtype=wp.float64)):
    z = foot_z[0] + foot_dz[0]
    if z < z_floor[0]:
        z = z_floor[0]
    foot_z[0] = z
    foot_v[0] = (z - foot_z_prev[0])/h
    wp.atomic_add(foot_acc, 0, foot_dz[0])
    foot_dz[0] = F(0.0)


@wp.kernel
def k_energy(tri: wp.array(dtype=int, ndim=2), Dminv: wp.array(dtype=wp.mat22), A0: wp.array(dtype=float),
             aw: wp.array(dtype=wp.vec2), af: wp.array(dtype=wp.vec2), mat: wp.array(dtype=float, ndim=2),
             x: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64), tf_p: wp.array(dtype=float, ndim=2),
             tf_F: wp.array(dtype=float, ndim=2)):
    t = wp.tid()
    i0 = tri[t, 0]
    i1 = tri[t, 1]
    i2 = tri[t, 2]
    d1 = to32(x[i1] - x[i0])
    d2 = to32(x[i2] - x[i0])
    gw = Dminv[t]*aw[t]
    gf = Dminv[t]*af[t]
    fw = d1*gw[0] + d2*gw[1]
    ff = d1*gf[0] + d2*gf[1]
    A = A0[t]
    ew = wp.length(fw) - 1.0
    ef = wp.length(ff) - 1.0
    E = A*(tension_law32(ew, mat[t, 0], mat[t, 1], mat[t, 2], mat[t, 6])[0] +
           tension_law32(ef, mat[t, 3], mat[t, 4], mat[t, 5], mat[t, 6])[0] +
           tens_fric(ew, t, 0, tf_p, tf_F[t, 0], mat[t, 9])[1] + tens_fric(ef, t, IWAN_N, tf_p, tf_F[t, 1], mat[t, 9])[1])
    s = wp.dot(fw, ff)
    E = E + 0.5*mat[t, 7]*A*s*s + A*tens_fric(s, t, 2*IWAN_N, tf_p, tf_F[t, 2], mat[t, 11])[1]
    E = E + A*shear_lock_energy(fw, ff, mat[t, 13], mat[t, 14])
    wp.atomic_add(out, 0, wp.float64(E))


@wp.kernel
def k_bend_energy(hinge: wp.array(dtype=int, ndim=2), rest: wp.array(dtype=float),
                  kb: wp.array(dtype=float), x: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64),
                  hf_T: wp.array(dtype=float), hf_k: wp.array(dtype=float), hf_p: wp.array(dtype=float, ndim=2)):
    """Bending energy: the elastic hinges plus the energy stored in the sticking yarn crossings."""
    e = wp.tid()
    X0 = x[hinge[e, 0]]
    r = dihedral32(to32(x[hinge[e, 1]] - X0), to32(x[hinge[e, 2]] - X0), to32(x[hinge[e, 3]] - X0))
    C = wrap_pi(r[0] - rest[e])
    f, ef, hs = bend_fric_iwan(r[0], e, hf_p, hf_k[e], hf_T[e])
    wp.atomic_add(out, 0, wp.float64(0.5*kb[e]*C*C + ef))


# ------------------------------------------------------------------------------------------------- the solver
def compression_law(T0, Tm, LC, Pm=KES_PM):
    """(a Pa, b 1/m) of P(d) = a (exp(b d) - 1) through P(T0 - Tm) = Pm with the KES linearity LC = WC / (Pm (T0 -
    Tm) / 2), i.e. 2/x - 2/(exp(x) - 1) = LC for x = b (T0 - Tm). (0, 0) when the fabric does not compress."""
    D = float(T0) - float(Tm)
    if D <= 0.0:
        return 0.0, 0.0
    lo, hi = 1e-6, 60.0
    for _ in range(200):                     # bisection: LC(x) falls monotonically from 1 to 0
        x = 0.5*(lo + hi)
        if 2.0/x - 2.0/math.expm1(x) > LC:
            lo = x
        else:
            hi = x
    x = 0.5*(lo + hi)
    return Pm/math.expm1(x), x/D


def hinge_angles(P3, H):
    """Unsigned dihedral angle (rad) of each hinge (a, b | c, d) of a consistently oriented mesh at positions P3."""
    x0, x1, x2, x3 = P3[H[:, 0]], P3[H[:, 1]], P3[H[:, 2]], P3[H[:, 3]]
    e = x1 - x0
    n1 = np.cross(e, x2 - x0)
    n2 = np.cross(x3 - x0, e)
    return np.arctan2(np.linalg.norm(np.cross(n1, n2), axis=1), np.einsum("ij,ij->i", n1, n2))


def wrap_material(X, kappa, d):
    """Isometric wrap of flat material coordinates X (n, 2) onto a cylinder of curvature kappa bending along the unit
    material direction d (the curvature direction)."""
    d = np.asarray(d, float)/np.linalg.norm(d)
    q = np.array([-d[1], d[0]])
    s = (X - X.mean(0)) @ d
    c = (X - X.mean(0)) @ q
    out = np.zeros((len(X), 3))
    out[:, :2] = np.outer(np.sin(kappa*s)/kappa, d) + np.outer(c, q)
    out[:, 2] = (1.0 - np.cos(kappa*s))/kappa
    return out


def calibrate_hinges(X, H, elen, A12, cw, cf, Bw, Bf, aw, af, edge_split=4.3e-3, kappa=2.0):
    """Hinge coefficients that make the discrete bending energy and the bending-friction slip work equal their
    continuum values under exact isometric wraps along the warp and the weft. The discrete-shell hinge
    energy carries the triangulation's directional bias (the mesher's strips: B 0.83 x warp, 1.23 x weft with one
    global coefficient); solved per mesh-size class (seam band, bulk), each class against its own hinge area A12/3:
      sum_e 1/2 theta_e^2 |e|^2/A12 (s_w Bw cw_e + s_f Bf cf_e) = 1/2 B_dir kappa^2 A_class
      sum_e |theta_e| |e| (t_w cw_e + t_f cf_e)                = kappa A_class        (per unit HB)
    Returns per-hinge elastic factors (multiplying Bw cw, Bf cf) and slip factors, and the per-class report."""
    g = elen**2/np.maximum(A12, 1e-18)
    fine = elen < edge_split
    th = [hinge_angles(wrap_material(X, kappa, a), H) for a in (aw, af)]
    sw = np.full(len(H), BEND_COEF)
    sf = np.full(len(H), BEND_COEF)
    tw = np.full(len(H), HF_COEF)
    tf = np.full(len(H), HF_COEF)
    report = []
    for name, m in (("fine", fine), ("coarse", ~fine)):
        if m.sum() < 20:
            continue
        Acl = float(A12[m].sum()/3.0)
        Mb = np.array([[0.5*np.sum(t[m]**2*g[m]*Bw*cw[m]), 0.5*np.sum(t[m]**2*g[m]*Bf*cf[m])] for t in th])
        rb = np.array([0.5*Bw*kappa**2*Acl, 0.5*Bf*kappa**2*Acl])
        Mf = np.array([[np.sum(np.abs(t[m])*elen[m]*cw[m]), np.sum(np.abs(t[m])*elen[m]*cf[m])] for t in th])
        rf = np.array([kappa*Acl, kappa*Acl])
        rec = dict(cls=name, hinges=int(m.sum()), area_m2=Acl)
        try:
            xb = np.linalg.solve(Mb, rb)
            xf = np.linalg.solve(Mf, rf)
        except np.linalg.LinAlgError:
            rec["fallback"] = "singular"
            report.append(rec)
            continue
        ok_b = bool(np.all((xb > 0.25) & (xb < 4.0)))
        ok_f = bool(np.all((xf > 0.25) & (xf < 4.0)))
        if ok_b:
            sw[m], sf[m] = xb
        if ok_f:
            tw[m], tf[m] = xf
        rec.update(elastic=dict(w=float(xb[0]), f=float(xb[1]), used=ok_b),
                   slip=dict(w=float(xf[0]), f=float(xf[1]), used=ok_f),
                   uncalibrated_B_ratio=[float(Mb[0].sum()/rb[0]*BEND_COEF), float(Mb[1].sum()/rb[1]*BEND_COEF)],
                   uncalibrated_slip_ratio=[float(Mf[0].sum()/rf[0]*HF_COEF), float(Mf[1].sum()/rf[1]*HF_COEF)])
        report.append(rec)
    return sw, sf, tw, tf, report


class ClothSolver:
    """Holds the plies (and pin particles), the shape list and the dynamic constraints. Construction is on the host
    (numpy); `step()` advances one tick on the device."""

    def __init__(self, device="cuda:0", dt=5e-4, substeps=20, iterations=1, gravity=-9.81, mu_cc=0.65,
                 membrane_damping=0.0, bending_damping=0.0, drag=0.0, query_margin=1.0e-3, pair_capacity=400_000,
                 con_capacity=8192, max_shapes=96, relax=1.0, use_graph=True, method="implicit", a_max=500.0,
                 cg_iterations=40, projection_passes=3, rayleigh_beta=0.0, newton_iterations=2,
                 friction_eps=1e-5, sandwich_tol=0.15e-3, contact_kappa=1.0e5, air=True, newton_max=6,
                 air_length=20e-3, air_hmax=10e-3, mu_air=1.81e-5, friction_model="ipc", preslip=1.0e-4,
                 cg_tol_acc=0.0, cg_tol_rel=0.0, cg_block=8):
        self.device = device
        self.dt, self.substeps, self.iterations = float(dt), int(substeps), int(iterations)
        self.h = self.dt/self.substeps
        self.gravity = float(gravity)
        self.mu_cc = float(mu_cc)
        self.membrane_damping, self.bending_damping, self.drag = membrane_damping, bending_damping, drag
        self.query_margin = query_margin
        self.pair_capacity, self.con_capacity, self.max_shapes = pair_capacity, con_capacity, max_shapes
        self.relax = relax
        self.use_graph = use_graph
        if method not in ("implicit", "vbd", "xpbd"):
            raise ValueError("method must be 'implicit', 'vbd' or 'xpbd'")
        if method == "implicit":
            substeps = 1                 # one backward-Euler step per tick (the step is solved, not substepped)
            self.substeps = 1
            self.h = self.dt
        self.cg_iterations = int(cg_iterations)
        # the linear solve ends when its residual is below cg_tol_acc as the r.m.s. over the free particles of each
        # one's unbalanced acceleration (m/s^2), or cg_tol_rel of the residual it started from, checked every cg_block
        # iterations, and at cg_iterations at the latest (a capped solve is counted); both 0: every solve runs
        # cg_iterations
        self.cg_tol_acc, self.cg_tol_rel, self.cg_block = float(cg_tol_acc), float(cg_tol_rel), int(cg_block)
        self.newton_iterations = int(newton_iterations)
        self.friction_eps = float(friction_eps)      # IPC friction velocity regularisation (m/s): eps*dt of slip
        self.sandwich_tol = float(sandwich_tol)      # contact proximity counted for the sandwich friction (m)
        self.contact_kappa = float(contact_kappa)    # contact penalty k = kappa m / dt^2
        self.contact_act = 1.0e-6                    # contacts within this gap enter the Newton Hessian (m)
        # ply-on-ply pairs that can close within the tick: detection reaches cc_reach ticks of the fastest
        # cloth's travel (at most cc_reach_max), and a pair apart at the tick start is kept if its closing speed covers
        # its gap in cc_reach ticks (the tick-start speed: the factor covers two surfaces approaching and an
        # acceleration within the tick)
        self.cc_reach = 3.0
        self.cc_reach_max = 5.0e-3
        self.pair_keep = 1.0e-6                      # a ply-ply pair keeps its face unless another is this much closer
        self.cc_smooth_normals = True                # ply-ply contact along the interpolated vertex normals
        self.set_recovery_tau = 1.0                  # s: an unloaded surface's compression set recovers
        self.set_recovery_pa = 49.03                 # Pa: below this elastic pressure a surface counts as unloaded
        self.newton_max_extra = max(0, int(newton_max) - int(newton_iterations))
        if friction_model not in ("ipc", "elastoplastic"):
            raise ValueError("friction_model must be 'ipc' or 'elastoplastic'")
        self.friction_model = friction_model
        self.preslip = float(preslip)                # elastic-plastic friction: the stick displacement (m)
        self.ep = int(friction_model == "elastoplastic")
        # tangential step that ends the Newton loop (m): IPC, half its stick zone eps dt (0.25 nm at the
        # cell's eps, below what the solve resolves, so every tick ran to the iteration cap);
        # elastic-plastic, 1/20 of the preslip (an unconverged step leaves the stored displacement off, it does not
        # accumulate as drift)
        self.newton_tol = 0.05*self.preslip if self.ep else 0.5*self.friction_eps*self.dt
        self.newton_tol_v = 1e-4                          # velocity accuracy of every other step (m/s)
        self.debug_newton = None
        self.cc_peak_rule = False       # ply on ply: the shapes' load-change rule
        self.stick_snap_ep = 0.25       # elastic-plastic friction: the stick-first start's slip limit, in preslips
        self.start_accel_max = -1.0     # m/s^2: bound of the last tick's acceleration in the Newton start (< 0: off)
        self.stick_snap = 10.0                       # stick-first start below this many stick-zone widths of slip
        self.friction_beta = 1.0                     # friction Hessian along the slip (1 isotropic, 0 exact)
        self.air = bool(air)                         # air layer under the cloth
        self.air_length = float(air_length)          # lateral length of the squeezed gap (m, prior)
        self.air_hmax = float(air_hmax)              # gap beyond which the layer is ignored (m)
        self.mu_air = float(mu_air)                  # dynamic viscosity of air at 20 C (Pa s)
        self.rho_air = 1.204                         # density of air at 20 C (kg/m^3)
        self.drag_cd = 1.2                           # drag coefficient of a flat plate normal to the flow
        self.projection_passes = int(projection_passes)
        self.rayleigh_beta = float(rayleigh_beta)
        self.method = method
        self.a_max = float(a_max)        # bound on the warm-start acceleration (m/s^2; feed dogs reach ~200, Winck)
        self.plies = []
        self._X, self._x, self._m, self._r, self._mu, self._ply, self._cloth, self._excl = [], [], [], [], [], [], [], []
        self._area, self._cperm, self._dof, self._isbody, self._floor = [], [], [], [], []
        self._rsol = []
        self._cm, self._k1, self._rc = [], [], []
        self._tri, self._Dminv, self._A0, self._aw, self._af, self._mat = [], [], [], [], [], []
        self._hinge, self._hrest, self._kb, self._hfT, self._hfk = [], [], [], [], []
        self.shapes = []
        self.built = False
        self.foot = None

    # -------------------------------------------------------------------------------------------- construction
    def add_ply(self, name, X, tri, origin, R, material, rest_angles=None, thickness=None, mu=1.0, positions=None):
        """X (n, 2) material coords, tri (t, 3) ccw, placed at world = origin + R @ (u, v, 0) unless explicit
        `positions` (n, 3) are given (e.g. a pre-pressed folded strap). material: dict with areal_density (kg/m^2),
        K0w K1w etw K0f K1f etf kc G (N/m), Bw Bf (N m), thickness (m, contact), grain_deg. Returns the ply index."""
        assert not self.built
        v0 = sum(len(a) for a in self._x)
        X = np.asarray(X, float)
        tri = np.asarray(tri, np.int64)
        if positions is None:
            pos = np.asarray(origin, float)[None] + np.column_stack([X, np.zeros(len(X))]) @ np.asarray(R, float).T
        else:
            pos = np.asarray(positions, float)
        P_ = X[tri]
        A = 0.5*np.abs((P_[:, 1, 0] - P_[:, 0, 0])*(P_[:, 2, 1] - P_[:, 0, 1]) -
                       (P_[:, 1, 1] - P_[:, 0, 1])*(P_[:, 2, 0] - P_[:, 0, 0]))
        mass = np.zeros(len(X))
        np.add.at(mass, tri.ravel(), np.repeat(A/3.0, 3))
        mass *= float(material["areal_density"])
        th = float(thickness if thickness is not None else material["thickness"])         # Tm, at KES_PM
        # contact begins at the free (uncompressed) thickness; the soft compression law takes the ply to `th` (its
        # compressed thickness) at the presser pressure
        th_free = max(th, float(material.get("thickness_free", th)))
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        L = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
        excl = np.zeros(len(X))
        np.maximum.at(excl, e[:, 0], L)
        np.maximum.at(excl, e[:, 1], L)
        excl *= 1.5
        self._X.append(X)
        self._x.append(pos)
        self._m.append(mass)
        self._r.append(np.full(len(X), 0.5*th_free))
        self._mu.append(np.full(len(X), float(mu)))
        self._ply.append(np.full(len(X), len(self.plies)))
        self._cloth.append(np.ones(len(X), np.int32))
        self._excl.append(excl)
        self._area.append(mass/float(material["areal_density"]))
        # the fibres' solid thickness, half on each side of the mid-surface
        self._rsol.append(np.full(len(X), 0.5*float(material["areal_density"])/FIBRE_DENSITY))
        # compression: P = a (exp(b d) - 1) for a platen compression d of the ply from its free
        # thickness, through the KES points (Tm at KES_PM, the work WC through LC); a surface takes half of d, so its
        # rate is 2b. ccm holds that per-surface rate, ck1 the force scale a A of the vertex
        a_, b_ = compression_law(th_free, th, float(material.get("comp_LC", COMP_LC)))
        self._cm.append(np.full(len(X), 2.0*b_))
        self._k1.append(a_*self._area[-1])
        # compression resilience (KES RC): the fraction of the compression work the fabric returns
        self._rc.append(np.full(len(X), float(material.get("comp_RC", 1.0)) if b_ > 0 else 1.0))
        self._dof.append(np.ones((len(X), 3)))
        self._isbody.append(np.zeros(len(X), np.int32))
        self._floor.append(np.full(len(X), -1e9))
        ap = material.get("air_perm")
        self._cperm.append(np.full(len(X), AIR_DP/float(ap) if ap else 0.0))
        g = math.radians(material.get("grain_deg", 0.0))
        aw = np.array([math.sin(g), math.cos(g)])
        af = np.array([math.cos(g), -math.sin(g)])
        Dm = np.stack([P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0]], axis=2)
        Dminv = np.linalg.inv(Dm)
        # tensile hysteresis (KES RT): friction elements carrying phi times the elastic tension once they
        # slip, so the loading curve (1 + phi) n_el is the measured law and the unloading curve (1 - phi) n_el; the
        # elastic part is the law over 1 + phi
        RT = material.get("tensile_RT")
        phi = TF_COEF*(1.0 - RT)/(1.0 + RT) if RT else 0.0
        es = float(material.get("tens_stick_strain", 0.0))
        sc = 1.0/(1.0 + phi)
        # shear hysteresis (KES 2HG, 2HG5): friction elements on the shear angle with the slip force per unit width
        # HG0 + HG1 |gamma| (N/m) and stick angles spread over (0, shear_stick_angle]
        # (the slip force's growth HG1 steepens both branches of the loop, so the elastic shear is the KES slope G
        # less HG1, as the elastic tension is the law over 1 + phi)
        HG0, HG1 = float(material.get("HG0", 0.0)), float(material.get("HG1", 0.0))
        gs = float(material.get("shear_stick_angle", 0.0))
        # shear locking (params fabric.shear_lock_*): past the locking angle (rad) the shear angle gains the stiffness
        # shear_lock_k (N/m per rad); a fabric without a locking angle keeps the plain law
        lock_s = math.radians(float(material.get("shear_lock_deg", 90.0)))
        lock_k = float(material.get("shear_lock_k", 0.0)) if "shear_lock_deg" in material else 0.0
        mat = np.tile(np.array([material["K0w"]*sc, material["K1w"]*sc, material["etw"], material["K0f"]*sc,
                                material["K1f"]*sc, material["etf"], material["kc"], material["G"] - HG1, phi, es, HG0,
                                gs, HG1, lock_s, lock_k], float), (len(tri), 1))
        self._tri.append(tri + v0)
        self._Dminv.append(Dminv)
        self._A0.append(A)
        self._aw.append(np.tile(aw, (len(tri), 1)))
        self._af.append(np.tile(af, (len(tri), 1)))
        self._mat.append(mat)
        from .mesh import hinges
        H, B = hinges(tri)
        Hn = np.asarray(H, np.int64)
        el = X[Hn[:, 1]] - X[Hn[:, 0]]
        elen = np.linalg.norm(el, axis=1)
        t_hat = el/np.maximum(elen, 1e-18)[:, None]
        d_hat = np.column_stack([-t_hat[:, 1], t_hat[:, 0]])
        cw, cf = (d_hat @ aw)**2, (d_hat @ af)**2

        def tri_a(a, b, c):
            pa, pb, pc = X[a], X[b], X[c]
            return 0.5*abs((pb[0] - pa[0])*(pc[1] - pa[1]) - (pb[1] - pa[1])*(pc[0] - pa[0]))
        A12 = np.array([tri_a(h[0], h[1], h[2]) + tri_a(h[0], h[1], h[3]) for h in Hn]) if len(Hn) else np.zeros(0)
        HB = float(material.get("HB", 0.0))
        ks = float(material.get("bend_stick_curv", 25.0))
        sw, sf, tw, tf, cal = calibrate_hinges(X, Hn, elen, A12, cw, cf, max(material["Bw"], 1e-30),
                                               max(material["Bf"], 1e-30), aw, af) if len(Hn) else (0, 0, 0, 0, [])
        Bdir = material["Bw"]*sw*cw + material["Bf"]*sf*cf
        # hinge stiffness k = s B_dir |e|^2 / (A1 + A2), s calibrated per mesh class and direction (calibrate_hinges;
        # the often-quoted 3 over-counts by ~3; the triangulation's directional bias)
        kb = Bdir*elen**2/np.maximum(A12, 1e-18)
        rest = np.zeros(len(Hn)) if rest_angles is None else np.asarray(rest_angles(X, Hn), float)
        # bending hysteresis (KES 2HB = 2 HB, N m/m): slip torque HB |e| t per hinge (t calibrated likewise), stick
        # stiffness HB / kappa_s mapped like an isotropic B
        self._hfT.append(HB*elen*(tw*cw + tf*cf) if HB > 0 else np.zeros(len(Hn)))
        self._hfk.append((HB/ks)*(sw*cw + sf*cf)*elen**2/np.maximum(A12, 1e-18) if HB > 0 else np.zeros(len(Hn)))
        self._hinge.append(Hn + v0)
        self._hrest.append(rest)
        self._kb.append(kb)
        self.plies.append(dict(name=name, v0=v0, nv=len(X), t0=sum(len(a) for a in self._tri) - len(tri),
                               nt=len(tri), material=dict(material), thickness=th, boundary=B + v0,
                               bend_calibration=cal))
        return len(self.plies) - 1

    def add_particle(self, pos, mass, radius, mu=1.0):
        """A free particle (pin heads and tips). Not cloth: no cloth-cloth contact."""
        assert not self.built
        self._X.append(np.zeros((1, 2)))
        self._x.append(np.asarray(pos, float)[None])
        self._m.append(np.array([float(mass)]))
        self._r.append(np.array([float(radius)]))
        self._mu.append(np.array([float(mu)]))
        self._ply.append(np.array([-1]))
        self._cloth.append(np.zeros(1, np.int32))
        self._excl.append(np.zeros(1))
        self._area.append(np.zeros(1))
        self._rsol.append(np.zeros(1))
        self._cperm.append(np.zeros(1))
        self._cm.append(np.zeros(1))
        self._k1.append(np.zeros(1))
        self._rc.append(np.ones(1))
        self._dof.append(np.ones((1, 3)))
        self._isbody.append(np.zeros(1, np.int32))
        self._floor.append(np.full(1, -1e9))
        return sum(len(a) for a in self._x) - 1

    def add_body(self, pos, mass, free=(0, 0, 1), floor=-1e9):
        """A rigid body reduced to a translating particle (the presser foot: free along z only, guided in x and y by the
        presser bar). Shapes added with body=<index> ride on it at their offset; their contacts push it.
        `floor` is its lowest z (the bar's stop); set per tick with set_body_floor. Returns the particle index."""
        k = self.add_particle(pos, mass, 0.0)
        self._dof[-1] = np.asarray(free, float)[None]
        self._isbody[-1] = np.ones(1, np.int32)
        self._floor[-1] = np.full(1, float(floor))
        return k

    def add_shape(self, name, typ, half, pos, quat=(0, 0, 0, 1), kind=KIND_STATIC, mu=0.3, body=-1, on=True,
                  loaded=False, air=None, grip=0.0):
        """quat is (x, y, z, w). `loaded`: the shape is part of the presser-foot sandwich (foot soles, feed dog,
        throat plate): its friction limit includes the foot load carried by the particle. `air`: the shape is a
        support surface with an air layer under nearby cloth (default: static shapes). Returns the shape index."""
        assert len(self.shapes) < self.max_shapes
        self.shapes.append(dict(name=name, type=int(typ), half=np.asarray(half, float), p=np.asarray(pos, float),
                                q=np.asarray(quat, float), kind=int(kind), mu=float(mu), body=int(body), on=bool(on),
                                loaded=bool(loaded), air=bool(kind == KIND_STATIC) if air is None else bool(air),
                                grip=float(grip)))
        return len(self.shapes) - 1

    def set_foot(self, shape, mass, z0=0.0):
        if self.method == "implicit":
            raise NotImplementedError("the cloth-coupled dynamic foot exists only in the rejected xpbd/vbd paths; the "
                                      "machine's presser foot is quasi-static")
        self.foot = dict(shape=int(shape), mass=float(mass), z=float(z0), force=0.0, floor=-1.0)

    def build(self):
        dev = self.device
        self.X = np.vstack(self._X)
        x = np.vstack(self._x)
        m = np.concatenate(self._m)
        self.n = len(x)
        self.mass_host = m.copy()
        w = np.where(m > 0, 1.0/np.maximum(m, 1e-30), 0.0)
        self.tri_host = np.vstack(self._tri).astype(np.int32) if self._tri else np.zeros((0, 3), np.int32)
        self.hinge_host = np.vstack(self._hinge).astype(np.int32) if self._hinge else np.zeros((0, 4), np.int32)
        self.ply_host = np.concatenate(self._ply).astype(np.int32)
        self.is_cloth_host = np.concatenate(self._cloth).astype(np.int32)
        f64 = wp.float64
        with wp.ScopedDevice(dev):
            self.x = wp.array(x, dtype=V3)
            self.x_prev = wp.array(x, dtype=V3)
            self.x_tick0 = wp.array(x, dtype=V3)
            self.x_solved = wp.array(x, dtype=V3)       # diagnostics: positions after the implicit solve, before projections
            self.x32 = wp.array(x.astype(np.float32), dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=V3)
            self.w = wp.array(w, dtype=f64)
            self.w_host = w.copy()
            self.m = wp.array(m, dtype=f64)
            self.radius = wp.array(np.concatenate(self._r), dtype=f64)
            self.pmu = wp.array(np.concatenate(self._mu), dtype=f64)
            self.area = wp.array(np.concatenate(self._area), dtype=f64)
            # each vertex's solid half-thickness (areal density over the fibre density, halved; 0 for bodies)
            rs = np.concatenate(self._rsol)
            assert len(rs) == self.n
            self.rsol = wp.array(rs, dtype=f64)
            self.ccm = wp.array(np.concatenate(self._cm), dtype=f64)
            self.ck1 = wp.array(np.concatenate([np.broadcast_to(a, (len(b),)) for a, b in zip(self._k1, self._cm)]),
                                dtype=f64)
            self.dof = wp.array(np.vstack(self._dof), dtype=V3)
            self.is_body_host = np.concatenate(self._isbody).astype(np.int32)
            self.is_body = wp.array(self.is_body_host, dtype=int)
            self.floor_host = np.concatenate(self._floor).astype(np.float64)
            self.body_floor = wp.array(self.floor_host, dtype=f64)
            self.body_H = wp.zeros(self.n, dtype=wp.mat33)
            self.bc_b = wp.zeros((self.n, 2), dtype=int)
            self.bc_k = wp.zeros((self.n, 2), dtype=float)
            self.bc_n = wp.zeros((self.n, 2), dtype=wp.vec3)
            self.cperm = wp.array(np.concatenate(self._cperm), dtype=f64)
            self.air_H = wp.zeros(self.n, dtype=wp.mat33)
            self.vnorm = wp.zeros(self.n, dtype=wp.vec3)
            self.ply = wp.array(self.ply_host, dtype=int)
            self.is_cloth = wp.array(self.is_cloth_host, dtype=int)
            self.excl = wp.array(np.concatenate(self._excl), dtype=f64)
            self.Xm = wp.array(self.X, dtype=V2)
            self.f_ext = wp.zeros(self.n, dtype=V3)
            self.tri = wp.array(self.tri_host, dtype=int)
            nt = len(self.tri_host)
            self.Dminv = wp.array(np.vstack(self._Dminv).reshape(-1, 2, 2) if nt else np.zeros((0, 2, 2)), dtype=wp.mat22)
            self.A0 = wp.array(np.concatenate(self._A0) if nt else np.zeros(0), dtype=float)
            self.aw = wp.array(np.vstack(self._aw) if nt else np.zeros((0, 2)), dtype=wp.vec2)
            self.af = wp.array(np.vstack(self._af) if nt else np.zeros((0, 2)), dtype=wp.vec2)
            self.mat = wp.array(np.vstack(self._mat) if nt else np.zeros((0, 15)), dtype=float)
            self.tens_fric = bool(nt and ((np.vstack(self._mat)[:, 8] > 0) | (np.vstack(self._mat)[:, 10] > 0)).any())
            if self.tens_fric and self.method != "implicit":
                raise NotImplementedError("the membrane friction (KES RT, 2HG) is built for the implicit method only")
            self.tf_p = wp.zeros((max(nt, 1), 3*IWAN_N), dtype=float)
            self.tf_F = wp.zeros((max(nt, 1), 3), dtype=float)
            self.tf_diss = wp.zeros(1, dtype=wp.float64)
            self.lam_m = wp.zeros(max(3*nt, 1), dtype=float)
            self.hinge = wp.array(self.hinge_host, dtype=int)
            self.hrest = wp.array(np.concatenate(self._hrest) if self._hrest else np.zeros(0), dtype=float)
            self.kb = wp.array(np.concatenate(self._kb) if self._kb else np.zeros(0), dtype=float)
            self.hf_T = wp.array(np.concatenate(self._hfT) if self._hfT else np.zeros(0), dtype=float)
            self.hf_k = wp.array(np.concatenate(self._hfk) if self._hfk else np.zeros(0), dtype=float)
            self.hf_p = wp.zeros((len(self.hinge_host), IWAN_N), dtype=float)
            self.hfH = wp.zeros(len(self.hinge_host), dtype=float)
            self.hf_diss = wp.zeros(1, dtype=wp.float64)
            self.cset = wp.zeros((self.n, 2), dtype=wp.float64)
            self.cdel = wp.zeros((self.n, 2), dtype=wp.float64)
            self.crc = wp.array(np.concatenate(self._rc) if self._rc else np.ones(self.n), dtype=wp.float64)
            self.lam_b = wp.zeros(max(len(self.hinge_host), 1), dtype=float)
            from .mesh import color_sets
            tg = color_sets(self.tri_host)[1] if nt else []
            hg = color_sets(self.hinge_host)[1] if len(self.hinge_host) else []
            self.tri_colors = [wp.array(g_, dtype=int) for g_ in tg]
            self.hinge_colors = [wp.array(g_, dtype=int) for g_ in hg]
            # vertex adjacency (CSR) and a vertex colouring for the block solve: vertices sharing a triangle or a
            # hinge get different colours
            n = self.n
            ta = [[] for _ in range(n)]
            for t_, (a_, b_, c_) in enumerate(self.tri_host):
                ta[a_].append((t_, 0)); ta[b_].append((t_, 1)); ta[c_].append((t_, 2))
            ha = [[] for _ in range(n)]
            for e_, hh in enumerate(self.hinge_host):
                for r_, v_ in enumerate(hh):
                    ha[v_].append((e_, r_))
            ts = np.zeros(n + 1, np.int32)
            ts[1:] = np.cumsum([len(l_) for l_ in ta])
            hs = np.zeros(n + 1, np.int32)
            hs[1:] = np.cumsum([len(l_) for l_ in ha])
            self.adj_t_start = wp.array(ts, dtype=int)
            self.adj_t = wp.array(np.array([p_[0] for l_ in ta for p_ in l_] or [0], np.int32), dtype=int)
            self.adj_t_role = wp.array(np.array([p_[1] for l_ in ta for p_ in l_] or [0], np.int32), dtype=int)
            self.adj_h_start = wp.array(hs, dtype=int)
            self.adj_h = wp.array(np.array([p_[0] for l_ in ha for p_ in l_] or [0], np.int32), dtype=int)
            self.adj_h_role = wp.array(np.array([p_[1] for l_ in ha for p_ in l_] or [0], np.int32), dtype=int)
            nbr = [set() for _ in range(n)]
            for el in list(self.tri_host) + list(self.hinge_host):
                for a_ in el:
                    nbr[a_].update(int(b_) for b_ in el if b_ != a_)
            order = np.argsort([-len(s_) for s_ in nbr], kind="stable")
            col = -np.ones(n, np.int32)
            for v_ in order:
                used = {col[u_] for u_ in nbr[v_] if col[u_] >= 0}
                c_ = 0
                while c_ in used:
                    c_ += 1
                col[v_] = c_
            self.vertex_colors = [wp.array(np.nonzero(col == c_)[0].astype(np.int32), dtype=int)
                                  for c_ in range(int(col.max()) + 1)]
            self.y = wp.zeros(self.n, dtype=V3)
            ntr = max(nt, 1)
            nh_ = max(len(self.hinge_host), 1)
            self.grad = wp.zeros(self.n, dtype=V3)
            self.Dblk = wp.zeros(self.n, dtype=wp.mat33)
            self.Dinv = wp.zeros(self.n, dtype=wp.mat33)
            self.Mw = wp.zeros(ntr, dtype=wp.mat33)
            self.Mf = wp.zeros(ntr, dtype=wp.mat33)
            self.cw = wp.zeros(ntr, dtype=wp.vec3)
            self.cf = wp.zeros(ntr, dtype=wp.vec3)
            self.qv = wp.zeros((ntr, 3), dtype=wp.vec3)
            self.qlv = wp.zeros((ntr, 3), dtype=wp.vec3)
            self.gla = wp.zeros(ntr, dtype=float)
            self.gsa = wp.zeros(ntr, dtype=float)
            self.hg = wp.zeros((nh_, 4), dtype=wp.vec3)
            self.cg_r = wp.zeros(self.n, dtype=V3)
            self.cg_z = wp.zeros(self.n, dtype=V3)
            self.cg_p = wp.zeros(self.n, dtype=V3)
            self.cg_Ap = wp.zeros(self.n, dtype=V3)
            self.cg_d = wp.zeros(self.n, dtype=V3)
            self.Kd = wp.zeros(self.n, dtype=V3)
            self.cg_rz = wp.zeros(1, dtype=f64)
            self.cg_rz_new = wp.zeros(1, dtype=f64)
            self.cg_pAp = wp.zeros(1, dtype=f64)
            self.cg_rr = wp.zeros(1, dtype=f64)
            self.cg_hist = wp.zeros(self.cg_iterations + self.cg_block + 2, dtype=f64)
            self.cg_cond = wp.zeros(1, dtype=int)
            self.cg_mask = wp.zeros(self.n, dtype=wp.mat33d)
            self.cg_ehist = wp.zeros(self.cg_iterations + self.cg_block + 2, dtype=f64)
            # [iterations of this solve, |r0|^2, solves, iterations in total, capped solves, last solve's count,
            #  1 if the last solve's target was the absolute tolerance, the tick's current relative target (squared)]
            self.cg_ctl = wp.zeros(8, dtype=f64)
            # the r.m.s. over the solve's free particles of each one's unbalanced acceleration (k_precond)
            self.cg_abs2 = float((self.cg_tol_acc**2)*np.count_nonzero(self.w_host > 0))
            self.ls_alphas_host = [1.0, 0.5, 0.25, 0.125, 0.0625]
            self.ls_alphas = wp.array(self.ls_alphas_host, dtype=f64)
            self.ls_phi = wp.zeros(len(self.ls_alphas_host) + 1, dtype=f64)
            self.ls_alpha = wp.zeros(1, dtype=f64)
            self.ls_more = wp.zeros(1, dtype=int)
            self.ls_stats = wp.zeros(4, dtype=f64)      # line searches, alpha < 1, alpha = 0, stalls (k_ls_select)
            self.ls_scale = wp.array([1.0, 0.0], dtype=f64)   # trial scale, consecutive rejections (k_ls_select)
            # [last step, extra its this tick, ticks, extra its total, ticks ended unconverged at the cap, the
            #  largest last step of those (in tolerances)]
            self.newton_ctl = wp.zeros(6, dtype=f64)
            self.newton_cond = wp.zeros(1, dtype=int)
            self.x_trial = wp.zeros(self.n, dtype=V3)
            self.v_start = wp.zeros(self.n, dtype=V3)
            self.a_prev = wp.zeros(self.n, dtype=V3)
            self.mesh = wp.Mesh(points=self.x32, indices=wp.array(self.tri_host.ravel(), dtype=int)) if nt else None
            # k_detect_cc yields at most six pairs a vertex (three plies, two sides): a larger capacity only
            # launches idle threads in every pair kernel of every CG iteration
            self.pair_capacity = int(min(self.pair_capacity, 6*self.n))
            self.pair_v = wp.zeros(self.pair_capacity, dtype=int)
            self.cc_vmax = wp.zeros(1, dtype=f64)
            self.pair_f = wp.zeros(self.pair_capacity, dtype=int)
            self.pair_b = wp.zeros(self.pair_capacity, dtype=V3)
            self.pair_side = wp.zeros(self.pair_capacity, dtype=f64)
            self.pair_count = wp.zeros(1, dtype=int)
            assert len(self.plies) <= MAX_PLY_SLOTS, "raise MAX_PLY_SLOTS for more plies"
            self.pair_prev = wp.array(np.full((self.n, MAX_PLY_SLOTS), -1, np.int32), dtype=int)
            self.dx = wp.zeros(self.n, dtype=V3)
            self.dn = wp.zeros(self.n, dtype=f64)
            self.contact_flag = wp.zeros(self.n, dtype=int)
            S = self.max_shapes
            self.n_shapes = len(self.shapes)
            st = np.zeros(S, np.int32)
            sk = np.zeros(S, np.int32)
            sh = np.zeros((S, 3))
            sm = np.zeros(S)
            so = np.zeros(S, np.int32)
            sp = np.zeros((S, 3))
            sq = np.tile(np.array([0, 0, 0, 1.0]), (S, 1))
            for i, s in enumerate(self.shapes):
                st[i], sk[i], sh[i], sm[i], so[i], sp[i], sq[i] = s["type"], s["kind"], s["half"], s["mu"], int(s["on"]), s["p"], s["q"]
            self.s_type = wp.array(st, dtype=int)
            self.s_kind = wp.array(sk, dtype=int)
            self.s_half = wp.array(sh, dtype=V3)
            self.s_mu = wp.array(sm, dtype=f64)
            self.s_on = wp.array(so, dtype=int)
            self.s_p0 = wp.array(sp, dtype=V3)
            self.s_p1 = wp.array(sp, dtype=V3)
            self.s_q0 = wp.array(sq, dtype=wp.quatd)
            self.s_q1 = wp.array(sq, dtype=wp.quatd)
            self._sp_host, self._sq_host = sp.copy(), sq.copy()
            sl = np.zeros(S, np.int32)
            for i, s_ in enumerate(self.shapes):
                sl[i] = int(s_.get("loaded", False))
            self.s_loaded = wp.array(sl, dtype=int)
            sa = np.zeros(S, np.int32)
            sb = -np.ones(S, np.int32)
            for i, s_ in enumerate(self.shapes):
                sa[i] = int(s_.get("air", False) and s_["body"] < 0)
                sb[i] = s_["body"]
            self.s_air = wp.array(sa, dtype=int)
            self.s_body = wp.array(sb, dtype=int)
            self.s_grip = wp.array(np.array([float(s_.get("grip", 0.0)) for s_ in self.shapes] +
                                            [0.0]*(S - len(self.shapes))), dtype=f64)
            self.load = wp.zeros(self.n, dtype=f64)
            self.fc_n = wp.zeros(self.n, dtype=int)
            self.fc_mu = wp.zeros((self.n, 4), dtype=float)
            self.fc_ds = wp.zeros((self.n, 4), dtype=wp.vec3)
            self.fc_c = wp.zeros((self.n, 4), dtype=float)
            self.fc_N = wp.zeros((self.n, 4), dtype=float)
            self.fc_nrm = wp.zeros((self.n, 4), dtype=wp.vec3)
            self.fc_sid = wp.zeros((self.n, 4), dtype=int)
            # elastic-plastic friction state: each contact's stored stick displacement, by shape (cloth-shape) and by
            # vertex (ply on ply)
            self.fe_cur = wp.zeros((self.n, 4), dtype=wp.vec3)
            self.fe_prev = wp.zeros((self.n, 4), dtype=wp.vec3)
            self.fe_sid_prev = wp.array(np.full((self.n, 4), -1, np.int32), dtype=int)
            self.ccfe = wp.zeros(self.n, dtype=wp.vec3)
            self.ccN = wp.zeros(self.n, dtype=float)          # ply-on-ply load-change rule
            self.ccN_peak = wp.zeros(self.n, dtype=float)
            self.ccN_pk_cur = wp.zeros(self.n, dtype=float)
            self.ccfe_next = wp.zeros(self.n, dtype=wp.vec3)
            # and, for the shapes, the normal force each was held at (a contact pressed harder keeps its force)
            self.fN_prev = wp.zeros((self.n, 4), dtype=float)
            self.fNp_cur = wp.zeros((self.n, 4), dtype=float)
            # compliant bodies: spring stiffness to a target set every tick
            self._sk_host = np.zeros(self.n)
            self._st_host = np.zeros((self.n, 3))
            for i_, (k_, t_) in getattr(self, "_springs", {}).items():
                self._sk_host[i_], self._st_host[i_] = k_, t_
            self.spring_k = wp.array(self._sk_host, dtype=f64)
            self.spring_t = wp.array(self._st_host, dtype=V3)
            # bodies: the subspace each moves in (a projector; the axes given to add_body until set_body_free changes
            # it) and its matrix spring (zero until set_spring_matrix), in small per-body tables
            bodies = np.nonzero(self.is_body_host)[0]
            slot = np.full(self.n, -1, np.int32)
            slot[bodies] = np.arange(len(bodies))
            self.body_slot_host = slot
            self.body_slot = wp.array(slot, dtype=int)
            dof_host = np.vstack(self._dof)
            self._bP_host = np.zeros((max(len(bodies), 1), 3, 3))
            self._bK_host = np.zeros((max(len(bodies), 1), 3, 3))
            for k_, i_ in enumerate(bodies):
                self._bP_host[k_] = np.diag(dof_host[i_])
            self.body_P = wp.array(self._bP_host, dtype=wp.mat33d)
            self.body_K = wp.array(self._bK_host, dtype=wp.mat33d)
            self.pen_H = wp.zeros(self.n, dtype=wp.mat33)
            self.fc_H = wp.zeros(self.n, dtype=wp.mat33)
            self.pc = wp.zeros(self.pair_capacity, dtype=float)
            self.pk = wp.zeros(self.pair_capacity, dtype=float)
            self.pn = wp.zeros(self.pair_capacity, dtype=wp.vec3)
            self.pfN = wp.zeros(self.pair_capacity, dtype=float)
            self.pH = wp.zeros(self.pair_capacity, dtype=wp.mat33)
            self.pfn = wp.zeros(self.pair_capacity, dtype=wp.vec3)
            self.pfo = wp.zeros(self.pair_capacity, dtype=f64)       # gap offset: g = pfo + n.(x_v - p) (k_ccfric_setup)
            self.pressed = wp.zeros(self.n, dtype=int)
            self.press_mass = wp.zeros(max(len(self.plies), 1), dtype=f64)
            self.stack_top = wp.zeros(1, dtype=f64)
            self.foot_N = wp.zeros(1, dtype=f64)
            self.foot_rects = wp.zeros(4, dtype=V3)
            self.foot_zb_arr = wp.zeros(1, dtype=f64)
            self.foot_depth_arr = wp.zeros(1, dtype=f64)
            self.foot_halfs = wp.zeros(4, dtype=V3)
            self.press = dict(n_rect=0, z_bottom=0.0, depth=0.0, N=0.0)
            self.acc_f = wp.zeros(S, dtype=V3)
            self.acc_tau = wp.zeros(S, dtype=V3)
            self.acc_n = wp.zeros(S, dtype=f64)
            f = self.foot or dict(shape=-1, mass=1.0, z=0.0, force=0.0, floor=-1.0)
            self.foot = f
            self.foot_z = wp.array([f["z"]], dtype=f64)
            self.foot_z_prev = wp.array([f["z"]], dtype=f64)
            self.foot_v = wp.zeros(1, dtype=f64)
            self.foot_dz = wp.zeros(1, dtype=f64)
            self.foot_floor = wp.array([f["floor"]], dtype=f64)
            self.foot_acc = wp.zeros(1, dtype=f64)
            self.foot_force = wp.zeros(1, dtype=f64)
            C = self.con_capacity
            self.n_con = 0
            self.c_type = wp.zeros(C, dtype=int)
            self.c_on = wp.zeros(C, dtype=int)
            self.c_fa = wp.zeros(C, dtype=int)
            self.c_ba = wp.zeros(C, dtype=V3)
            self.c_fb = wp.zeros(C, dtype=int)
            self.c_bb = wp.zeros(C, dtype=V3)
            self.c_pa = wp.zeros(C, dtype=int)
            self.c_pb = wp.zeros(C, dtype=int)
            self.c_rest = wp.zeros(C, dtype=f64)
            self.c_comp = wp.zeros(C, dtype=f64)
            self.c_cap = wp.zeros(C, dtype=f64)
            self.c_force = wp.zeros(C, dtype=V3)
            self.con_H = wp.zeros(C, dtype=wp.mat33)
            self.n_con_arr = wp.zeros(1, dtype=int)
            self.con_host = []
            self.axis_p = wp.zeros(8, dtype=V3)
            self.axis_v = wp.zeros(8, dtype=V3)
            self.energy_buf = wp.zeros(1, dtype=f64)
        r = np.concatenate(self._r)
        self.max_radius = float(r[self.is_cloth_host == 1].max()) if (self.is_cloth_host == 1).any() else 0.0
        self.built = True
        self.graph = None
        self.ticks = 0
        self.relax_bending_friction()
        self.relax_tension_friction()
        self._con_dirty = False
        self._on_dirty = False
        self._floor_dirty = False

    # ------------------------------------------------------------------------------------------- per-tick input
    def set_shape_targets(self, idx, pos1, quat1=None, on=None):
        """End-of-tick pose of a shape (start-of-tick pose = last end pose); quat is (x, y, z, w)."""
        self._sp_host[idx] = pos1
        if quat1 is not None:
            self._sq_host[idx] = quat1
        if on is not None and bool(on) != self.shapes[idx]["on"]:
            self.shapes[idx]["on"] = bool(on)
            self._on_dirty = True

    def set_spring(self, idx, k, target):
        """A compliant body: particle idx held to `target` by a spring of stiffness k (N/m), solved implicitly with
        the cloth; call before build() for the stiffness, set_spring_target every tick."""
        if not hasattr(self, "_springs"):
            self._springs = {}
        self._springs[idx] = (float(k), np.asarray(target, float))

    def set_spring_target(self, idx, target, k=None):
        """This tick's target (and stiffness: a damper over one tick is a spring c/dt to the body's start-of-tick
        position moved with the target, folded into k and the target by the caller)."""
        self._st_host[idx] = target
        if k is not None and k != self._sk_host[idx]:
            self._sk_host[idx] = k
            self._sk_dirty = True
        self._st_dirty = True

    def set_body_free(self, idx, P):
        """The subspace body idx moves in this tick: a 3x3 projector (e.g. the vertical and a finger's closing axis)."""
        self._bP_host[self.body_slot_host[idx]] = np.asarray(P, float)
        self._bP_dirty = True

    def set_spring_matrix(self, idx, K, target=None):
        """A matrix spring on body idx (N/m, symmetric positive semi-definite), added to its scalar spring; both pull
        toward the same target."""
        self._bK_host[self.body_slot_host[idx]] = np.asarray(K, float)
        self._bK_dirty = True
        if target is not None:
            self._st_host[idx] = target
            self._st_dirty = True

    def spring_force(self, idx):
        """The spring's force on its target (N): (k + K) (x - target), the body pulled by what touches it."""
        x = self.x[idx:idx + 1].numpy()[0]
        e = x - self._st_host[idx]
        f = self._sk_host[idx]*e
        if self.body_slot_host[idx] >= 0:
            f = f + self._bK_host[self.body_slot_host[idx]] @ e
        return f

    def set_particle_force(self, idx, f):
        """Constant external force on particle idx (N), e.g. the presser spring on the foot body."""
        fe = self.f_ext.numpy()
        fe[idx] = f
        self.f_ext.assign(fe)

    def set_particle_position(self, idx, pos):
        """Place particle idx (e.g. a body's initial height) and zero its velocity."""
        x = self.x.numpy()
        x[idx] = pos
        self.x.assign(x)
        v = self.v.numpy()
        v[idx] = 0.0
        self.v.assign(v)

    def set_body_floor(self, idx, z):
        """Lowest z of body particle idx for the next tick (travel stop)."""
        if self.floor_host[idx] != z:
            self.floor_host[idx] = z
            self._floor_dirty = True

    def set_particle_target(self, idx, pos):
        """Move a kinematic (w = 0) particle, e.g. the needle-eye thread anchor."""
        p = np.asarray(pos, float).reshape(1, 3)
        self.x[idx:idx + 1].assign(p) if False else wp.copy(self.x, wp.array(p, dtype=V3, device=self.device), idx, 0, 1)

    def add_constraint(self, typ, fa=0, ba=(0, 0, 0), fb=0, bb=(0, 0, 0), pa=0, pb=0, rest=0.0, comp=0.0, cap=0.0,
                       meta=None):
        k = self.n_con
        if k >= self.con_capacity:
            raise RuntimeError("constraint capacity exhausted")
        self.n_con += 1
        rec = dict(type=int(typ), fa=int(fa), ba=tuple(map(float, ba)), fb=int(fb), bb=tuple(map(float, bb)),
                   pa=int(pa), pb=int(pb), rest=float(rest), comp=float(comp), cap=float(cap), on=1, meta=meta or {})
        self.con_host.append(rec)
        self._con_dirty = True
        return k

    def set_constraint(self, k, **kw):
        self.con_host[k].update(kw)
        self._con_dirty = True

    def _upload_constraints(self):
        """Copy the host constraint table to the device: only the rows in use (rows beyond n_con are never read)."""
        n = self.n_con
        if n == 0:
            return
        arr = self.con_host
        for name, key, dt_ in (("c_type", "type", np.int32), ("c_on", "on", np.int32), ("c_fa", "fa", np.int32),
                               ("c_ba", "ba", np.float64), ("c_fb", "fb", np.int32), ("c_bb", "bb", np.float64),
                               ("c_pa", "pa", np.int32), ("c_pb", "pb", np.int32), ("c_rest", "rest", np.float64),
                               ("c_comp", "comp", np.float64), ("c_cap", "cap", np.float64)):
            getattr(self, name)[:n].assign(np.array([r[key] for r in arr], dt_))

    # ------------------------------------------------------------------------------------------------- stepping
    def _substeps(self):
        S = self.substeps
        h = wp.float64(self.h)
        f = self.foot
        g = V3(0.0, 0.0, self.gravity)
        n = self.n
        f64 = wp.float64
        for s in range(S):
            t0 = f64(s/S)
            t1 = f64((s + 1)/S)
            wp.launch(k_predict, n, [self.x, self.v, self.x_prev, self.w, self.f_ext, g, h, f64(self.drag),
                                     0 if self.method == "vbd" else 1, self.y, self.v_start, self.a_prev,
                                     f64(self.a_max)])
            if f["shape"] >= 0:
                wp.launch(k_foot_predict, 1, [self.foot_z, self.foot_z_prev, self.foot_v, self.foot_force,
                                              f64(f["mass"]), f64(-self.gravity), h])
            if self.method == "vbd":
                for _ in range(self.iterations):
                    for ids in self.vertex_colors:
                        wp.launch(k_vbd, len(ids), [ids, self.x, self.x_prev, self.y, self.m, self.w,
                                                    self.adj_t_start, self.adj_t, self.adj_t_role, self.adj_h_start,
                                                    self.adj_h, self.adj_h_role, self.tri, self.Dminv, self.A0,
                                                    self.aw, self.af, self.mat, self.hinge, self.hrest, self.kb,
                                                    self.f_ext, wp.vec3(0.0, 0.0, self.gravity), float(self.h),
                                                    float(self.membrane_damping)])
            if self.method == "xpbd" and len(self.tri_colors):
                wp.launch(k_zero, len(self.lam_m), [self.lam_m])
            if self.method == "xpbd" and len(self.hinge_colors):
                wp.launch(k_zero, len(self.lam_b), [self.lam_b])
            for _ in range(self.iterations if self.method == "xpbd" else 0):
                for ids in self.tri_colors:
                    wp.launch(k_membrane, len(ids), [ids, self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat,
                                                     self.x, self.x_prev, self.w, self.lam_m, float(self.h),
                                                     float(self.membrane_damping)])
                for ids in self.hinge_colors:
                    wp.launch(k_bending, len(ids), [ids, self.hinge, self.hrest, self.kb, self.x, self.x_prev, self.w,
                                                    self.lam_b, float(self.h), float(self.bending_damping)])
            wp.launch(k_constraints, self.con_capacity,
                      [self.n_con_arr, self.c_type, self.c_on, self.c_fa, self.c_ba, self.c_fb, self.c_bb,
                       self.c_pa, self.c_pb, self.c_rest, self.c_comp, self.c_cap, self.axis_p, self.axis_v,
                       self.tri, self.x, self.x_prev, self.w, h, self.dx, self.dn, self.c_force, 0])
            wp.launch(k_apply_jacobi, n, [self.x, self.dx, self.dn, f64(1.0)])
            if self.mesh is not None:
                wp.launch(k_solve_cc, self.pair_capacity, [self.pair_v, self.pair_f, self.pair_b, self.pair_side,
                                                           self.pair_count, self.pair_capacity, self.tri, self.x,
                                                           self.x_prev, self.w, self.radius, f64(self.mu_cc),
                                                           self.dx, self.dn, self.load, f64(self.h)])
                wp.launch(k_apply_jacobi, n, [self.x, self.dx, self.dn, f64(self.relax)])
            wp.launch(k_shapes, n, [self.x, self.x_prev, self.w, self.m, self.radius, self.pmu, self.n_shapes,
                                    self.s_type, self.s_kind, self.s_half, self.s_mu, self.s_on, self.s_p0, self.s_p1,
                                    self.s_q0, self.s_q1, t0, t1, self.foot_z, self.foot_z_prev,
                                    f64(1.0/f["mass"] if f["shape"] >= 0 else 0.0), self.foot_dz,
                                    self.acc_f, self.acc_tau, self.acc_n, self.contact_flag, self.load,
                                    self.s_loaded])
            if f["shape"] >= 0:
                wp.launch(k_foot_finish, 1, [self.foot_z, self.foot_z_prev, self.foot_v, self.foot_dz,
                                             self.foot_floor, h, self.foot_acc])
            wp.launch(k_velocity, n, [self.x, self.x_prev, self.v, self.w, h, self.v_start, self.a_prev])

    def _implicit_tick(self):
        n = self.n
        f64 = wp.float64
        dt = f64(self.dt)
        g = V3(0.0, 0.0, self.gravity)
        f = self.foot
        nt = len(self.tri_host)
        nh = len(self.hinge_host)
        wp.launch(k_imp_predict, n, [self.x, self.v, self.x_prev, self.y, self.w, self.f_ext, g, dt, f64(self.drag),
                                     self.v_start, self.a_prev, f64(self.start_accel_max), self.body_slot, self.body_P])
        wp.launch(k_body_floor, n, [self.x, self.is_body, self.body_floor])
        epsdt = float(self.friction_eps*self.dt)
        kdt = f64(self.contact_kappa/(self.dt*self.dt))
        # area-weighted vertex normals at the tick start: the air layer and the ply-on-ply contact normal
        self.vnorm.zero_()
        if nt:
            wp.launch(k_tri_normals, nt, [self.tri, self.x_prev, self.vnorm])
        sh = [self.n_shapes, self.s_type, self.s_half, self.s_on, self.s_p1, self.s_q1, self.s_body, self.is_body]
        cc = self.mesh is not None
        pairs = [self.pair_v, self.pair_f, self.pair_b, self.pair_side, self.pair_count, self.pair_capacity, self.tri]
        wp.launch(k_fric_contacts, n, [self.x_prev, self.radius, self.ck1, self.ccm, self.m, self.w, self.is_body, self.n_shapes,
                                       self.s_type, self.s_half, self.s_mu, self.s_on, self.s_body, self.s_p0, self.s_p1,
                                       self.s_q0, self.s_q1, kdt, self.fc_n, self.fc_mu, self.fc_N, self.fc_ds,
                                       self.fc_nrm, self.fc_sid, self.cset, self.crc, self.vnorm, self.fe_prev,
                                       self.fe_sid_prev, self.fe_cur, self.s_grip, self.area, self.rsol])
        if cc:
            wp.launch(k_ccfric_setup, self.pair_capacity, pairs + [self.x_prev, self.radius, self.ck1, self.ccm, self.m,
                                                                   kdt, self.pfN,
                                                                   self.pfn, self.vnorm, int(self.cc_smooth_normals),
                                                                   self.cset, self.crc, self.pfo, f64(self.contact_act),
                                                                   self.rsol, self.v_start, f64(self.cc_reach*self.dt)])
            if self.ep and self.cc_peak_rule:
                self.ccN.zero_()
                wp.launch(k_ccfric_load, self.pair_capacity, [self.pair_v, self.pair_count, self.pair_capacity, self.pfN,
                                                              self.ccN])
                wp.launch(k_ccfric_rescale, n, [self.ccN, self.ccN_peak, self.ccfe, self.ccN_pk_cur])
        if self.ep:
            # the stored stick displacements carried at the tangential force they held
            wp.launch(k_fric_rescale, n, [self.fc_n, self.fc_N, self.fc_sid, self.fN_prev, self.fe_sid_prev, self.fe_cur,
                                          self.fNp_cur, float(self.preslip)])
        if self.air:
            wp.launch(k_air, n, [self.x_prev, self.v, self.vnorm, self.w, self.radius, self.area, self.cperm,
                                 self.n_shapes, self.s_type, self.s_half, self.s_on, self.s_air, self.s_p0, self.s_q0,
                                 f64(12.0*self.mu_air*self.air_length**2), f64(self.air_hmax), f64(1.0/self.dt),
                                 f64(0.5*self.rho_air*self.drag_cd), self.air_H])
        fric = [self.pmu, self.fc_n, self.fc_mu, self.fc_N, self.fc_ds, self.fc_nrm, epsdt]
        snap = self.stick_snap_ep*self.preslip if self.ep else self.stick_snap*epsdt
        wp.launch(k_stick_start, n, [self.x, self.x_prev, self.v, dt, self.pmu, self.fc_n, self.fc_mu, self.fc_N, self.fc_ds,
                                     self.fc_nrm, snap])
        epf = [self.fe_cur, self.ep, float(self.preslip)]
        ccepf = [self.ccfe, self.ep, float(self.preslip)]
        con = [self.n_con_arr, self.c_type, self.c_on, self.c_fa, self.c_ba, self.c_fb, self.c_bb, self.c_pa,
               self.c_pb, self.c_rest, self.c_comp, self.c_cap, self.axis_p, self.axis_v, self.tri]
        con_mv = [self.n_con_arr, self.c_type, self.c_on, self.c_fa, self.c_ba, self.c_fb, self.c_bb, self.c_pb,
                  self.tri, self.con_H]
        ccp = [self.pair_v, self.pair_f, self.pair_b, self.pair_count, self.pair_capacity, self.tri]
        def newton_iter():
            # Newton step at the iterate: gradient, Hessian data, preconditioner
            wp.launch(k_imp_zero, n, [self.grad, self.Dblk, self.body_H])
            if nt:
                wp.launch(k_imp_tri, nt, [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat, self.x, self.grad,
                                          self.Dblk, self.Mw, self.Mf, self.cw, self.cf, self.qv, self.gsa, self.tf_p,
                                          self.tf_F, self.qlv, self.gla])
            if nh:
                wp.launch(k_imp_hinge, nh, [self.hinge, self.hrest, self.kb, self.x, self.grad, self.Dblk, self.hg,
                                            self.hf_T, self.hf_k, self.hf_p, self.hfH])
            # contacts (penalty) and friction (lagged normal; IPC smoothed Coulomb) are part of the step
            wp.launch(k_pen_shapes, n, [self.x, self.radius, self.ck1, self.ccm, self.m, self.w] + sh +
                      [kdt, f64(self.contact_act), self.x_prev, self.air_H, self.grad, self.Dblk, self.pen_H, self.body_H, self.bc_b, self.bc_k,
                       self.bc_n, self.cset, self.crc, self.vnorm, self.rsol])
            wp.launch(k_fric_grad, n, [self.x, self.x_prev] + fric + [self.grad, self.Dblk, self.fc_H,
                                                                      float(self.friction_beta)] + epf)
            wp.launch(k_con_grad, self.con_capacity, con + [self.x, self.x_prev, dt, f64(epsdt), f64(self.contact_act),
                                                            self.grad, self.Dblk, self.con_H])
            if cc:
                wp.launch(k_pen_cc, self.pair_capacity, pairs + [self.x, self.radius, self.ck1, self.ccm, self.m, kdt, f64(self.contact_act),
                                                                 self.pfn, self.grad,
                                                                 self.Dblk, self.pk, self.pn, self.cset, self.crc, self.vnorm,
                                                                 self.pfo, self.rsol])
                wp.launch(k_ccfric_grad, self.pair_capacity, ccp + [self.x, self.x_prev, self.pfN, self.pfn,
                                                                    float(self.mu_cc), epsdt, self.grad, self.Dblk,
                                                                    self.pc, self.pH, float(self.friction_beta)] + ccepf)
            kscale = 1.0 + self.rayleigh_beta/self.dt
            wp.launch(k_vzero, n, [self.Kd])
            if self.rayleigh_beta > 0.0:
                wp.launch(k_sub, n, [self.x, self.x_prev, self.cg_d])
                if nt:
                    wp.launch(k_mv_tri, nt, [self.tri, self.Mw, self.Mf, self.cw, self.cf, self.qv, self.gsa, self.cg_d,
                                             self.Kd, 1.0, self.qlv, self.gla])
                if nh:
                    wp.launch(k_mv_hinge, nh, [self.hinge, self.kb, self.hg, self.cg_d, self.Kd, 1.0, self.hfH, 0.0])
            wp.launch(k_imp_inertia, n, [self.x, self.y, self.m, self.w, dt, float(kscale), self.grad, self.Dblk, self.Dinv,
                                         self.cg_r, self.cg_d, self.Kd, float(self.rayleigh_beta/self.dt),
                                         self.is_body, self.body_floor, self.spring_k, self.spring_t, self.cg_mask,
                                         self.body_slot, self.body_P, self.body_K])
            # preconditioned conjugate gradient, fixed iteration count (graph-capturable); residual history recorded
            wp.launch(k_szero, 1, [self.cg_rz])
            wp.launch(k_szero, 1, [self.cg_rz_new])
            wp.launch(k_szero, 1, [self.cg_pAp])
            wp.launch(k_szero, 1, [self.cg_rr])
            wp.launch(k_precond, n, [self.cg_r, self.Dinv, self.cg_z, self.cg_rz, self.cg_rr, self.w])
            wp.launch(k_cg_init, n, [self.cg_z, self.cg_p])
            adaptive = self.cg_tol_acc > 0.0 or self.cg_tol_rel > 0.0
            abs2 = self.cg_abs2 if adaptive else 0.0
            wp.launch(k_cg_start, 1, [self.cg_rz_new, self.cg_pAp, self.cg_rr, self.cg_hist, self.cg_ctl, self.cg_cond,
                                      f64(abs2 if adaptive else -1.0)])

            def cg_iter():
                wp.launch(k_vzero, n, [self.cg_Ap])
                if nt:
                    wp.launch(k_mv_tri, nt, [self.tri, self.Mw, self.Mf, self.cw, self.cf, self.qv, self.gsa, self.cg_p,
                                             self.cg_Ap, float(kscale), self.qlv, self.gla])
                if nh:
                    wp.launch(k_mv_hinge, nh, [self.hinge, self.kb, self.hg, self.cg_p, self.cg_Ap, float(kscale),
                                             self.hfH, 1.0])
                wp.launch(k_mv_pen, n, [self.cg_p, self.pen_H, self.body_H, self.bc_b, self.bc_k, self.bc_n, self.cg_Ap])
                wp.launch(k_mv_fric, n, [self.cg_p, self.fc_n, self.fc_H, self.cg_Ap])
                wp.launch(k_mv_con, self.con_capacity, con_mv + [self.cg_p, self.cg_Ap])
                if cc:
                    wp.launch(k_mv_pcc, self.pair_capacity, ccp + [self.pk, self.pn, self.cg_p, self.cg_Ap])
                    wp.launch(k_mv_ccfric, self.pair_capacity, ccp + [self.pc, self.pH, self.cg_p, self.cg_Ap])
                wp.launch(k_mv_diag, n, [self.cg_p, self.m, self.w, dt, self.cg_Ap, self.cg_pAp, self.spring_k,
                                         self.body_slot, self.body_K])
                wp.launch(k_cg_step, n, [self.cg_p, self.cg_Ap, self.cg_d, self.cg_r, self.cg_rz, self.cg_pAp,
                                         self.cg_mask])
                wp.launch(k_precond, n, [self.cg_r, self.Dinv, self.cg_z, self.cg_rz_new, self.cg_rr, self.w])
                wp.launch(k_cg_dir, n, [self.cg_z, self.cg_p, self.cg_rz_new, self.cg_rz])
                wp.launch(k_cg_swap, 1, [self.cg_rz, self.cg_rz_new, self.cg_pAp, self.cg_rr, self.cg_hist,
                                         self.cg_ctl, self.cg_cond, f64(abs2), f64(self.cg_tol_rel**2),
                                         self.cg_iterations, self.cg_ehist])

            if adaptive:
                def cg_block():
                    for _ in range(self.cg_block):
                        cg_iter()
                wp.capture_while(self.cg_cond, cg_block)
            else:
                for _ in range(self.cg_iterations):
                    cg_iter()
            # backtracking line search on the incremental potential (inertia + membrane + bending + contact + friction):
            # the step never raises it, so one Newton step per tick cannot pump energy
            nT = len(self.ls_alphas_host)
            wp.launch(k_szero, nT + 1, [self.ls_phi])

            def ls_eval(k_):
                xs = self.x if k_ == 0 else self.x_trial
                if k_ > 0:
                    wp.launch(k_trial, n, [self.x, self.cg_d, f64(self.ls_alphas_host[k_ - 1]), self.ls_scale,
                                           self.x_trial])
                    wp.launch(k_body_floor, n, [self.x_trial, self.is_body, self.body_floor])
                wp.launch(k_inertia_energy, n, [xs, self.y, self.m, self.w, dt, self.ls_phi, k_, self.spring_k,
                                                self.spring_t, self.body_slot, self.body_K])
                if nt:
                    wp.launch(k_energy_slot, nt, [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat, xs,
                                                  self.ls_phi, k_, self.tf_p, self.tf_F])
                if nh:
                    wp.launch(k_bend_energy_slot, nh, [self.hinge, self.hrest, self.kb, xs, self.ls_phi, k_,
                                                       self.hf_T, self.hf_k, self.hf_p])
                wp.launch(k_pen_shapes_energy, n, [xs, self.radius, self.ck1, self.ccm, self.m, self.w] + sh + [kdt, self.x_prev, self.air_H,
                                                                                         self.ls_phi, k_, self.cset, self.crc, self.vnorm,
                                                                                         self.rsol])
                wp.launch(k_fric_energy, n, [xs, self.x_prev] + fric + [self.ls_phi, k_] + epf)
                wp.launch(k_con_energy, self.con_capacity, con + [xs, self.x_prev, dt, f64(epsdt), self.ls_phi, k_])
                if cc:
                    wp.launch(k_pen_cc_energy, self.pair_capacity, pairs + [xs, self.radius, self.ck1, self.ccm, self.m,
                                                                            kdt, self.pfn,
                                                                            self.ls_phi, k_, self.cset, self.crc, self.vnorm,
                                                                            self.pfo, self.rsol])
                    wp.launch(k_ccfric_energy, self.pair_capacity, ccp + [xs, self.x_prev, self.pfN, self.pfn,
                                                                          float(self.mu_cc), epsdt, self.ls_phi, k_] + ccepf)

            # the start and the full step first; the shorter trials only when the full step raises the potential
            ls_eval(0)
            ls_eval(1)
            wp.launch(k_ls_more, 1, [self.ls_phi, self.ls_more])

            def ls_rest():
                for k_ in range(2, nT + 1):
                    ls_eval(k_)
            wp.capture_if(self.ls_more, on_true=ls_rest)
            # a rejected step continues the backtracking below the smallest trial (counted as converged, the tick
            # would keep the unforced prediction). E7's default seam puckers 0.58-0.70 mm with it (0.70 also with the
            # trials merely repeated, 0.20 in one run that counted it as converged)
            wp.launch(k_ls_select, 1, [self.ls_phi, self.ls_alphas, nT, self.ls_alpha, self.ls_stats, self.ls_scale,
                                       f64(0.5*self.ls_alphas_host[-1])])
            wp.launch(k_add_scaled, n, [self.x, self.cg_d, self.ls_alpha])
            wp.launch(k_body_floor, n, [self.x, self.is_body, self.body_floor])
            wp.launch(k_szero, 1, [self.newton_ctl])
            wp.launch(k_step_size, n, [self.cg_d, self.ls_alpha, self.w, self.fc_n, self.fc_mu, self.fc_N,
                                       self.fc_nrm, f64(self.newton_tol), f64(self.newton_tol_v*self.dt),
                                       self.newton_ctl, self.ls_scale])
            wp.launch(k_newton_next, 1, [self.newton_ctl, self.newton_cond, f64(1.0), self.newton_max_extra,
                                         self.cg_ctl])
            if self.debug_newton is not None:          # diagnostics, ungraphed runs only
                d = self.cg_d.numpy()
                al = float(self.ls_alpha.numpy()[0])
                j = int(np.argmax(np.linalg.norm(d, axis=1)*(self.w_host > 0)))
                self.debug_newton.append(dict(alpha=al, step=float(np.linalg.norm(d[j])*al), at=j, d=d[j].copy(),
                                              cg=self.cg_residuals()[[0, -1]].copy(),
                                              cg_rr=self.cg_residuals().copy(),
                                              cg_r=self.cg_r.numpy().copy(),
                                              cg_de=self.cg_ehist.numpy()[:int(self.cg_ctl.numpy()[5]) + 1].copy(),
                                              measure=float(self.newton_ctl.numpy()[0])))

        wp.launch(k_ls_reset, 1, [self.ls_scale])
        wp.launch(k_cg_rel_reset, 1, [self.cg_ctl, f64(self.cg_tol_rel**2)])
        for _newton in range(self.newton_iterations):
            newton_iter()
        # converge: more iterations while the last accepted step exceeds half the friction stick zone
        wp.launch(k_newton_arm, 1, [self.newton_ctl, self.newton_cond, f64(1.0), self.newton_max_extra, self.cg_ctl])
        if self.newton_max_extra > 0:
            wp.capture_while(self.newton_cond, newton_iter)
        wp.copy(self.x_solved, self.x)
        wp.launch(k_contact_react, n, [self.x, self.x_prev, self.radius, self.ck1, self.ccm, self.m, self.w, self.pmu] + sh +
                  [kdt, self.fc_n, self.fc_mu, self.fc_N, self.fc_ds, self.fc_nrm, self.fc_sid, epsdt,
                   f64(self.dt*self.dt), self.acc_f, self.acc_tau, self.acc_n, self.contact_flag, self.cset, self.crc, self.vnorm]
                  + epf + [self.rsol])
        if self.ep:
            # elastic-plastic friction: store each contact's elastic displacement, cut back where it slid
            wp.launch(k_fric_return, n, [self.x, self.x_prev, self.fc_n, self.fc_ds, self.fc_nrm, self.fc_sid,
                                         self.fe_cur, float(self.preslip), self.fe_prev, self.fe_sid_prev])
            wp.launch(k_fric_store_n, n, [self.fc_n, self.fc_N, self.fNp_cur, self.fe_prev, float(self.preslip),
                                          self.fN_prev])
            if cc:
                self.ccfe_next.zero_()
                wp.launch(k_ccfric_return, self.pair_capacity, ccp + [self.x, self.x_prev, self.pfN, self.pfn, self.ccfe,
                                                                      float(self.preslip), self.ccfe_next])
                wp.copy(self.ccfe, self.ccfe_next)
                if self.cc_peak_rule:
                    wp.launch(k_ccfric_store_n, n, [self.ccN, self.ccN_pk_cur, self.ccfe, float(self.preslip),
                                                    self.ccN_peak])
        wp.launch(k_con_force, self.con_capacity, con + [self.x, self.x_prev, dt, f64(epsdt), self.c_force])
        # pins (pin crossings, pin rods) remain projections on the solved motion
        for _ in range(self.projection_passes):
            wp.launch(k_constraints, self.con_capacity,
                      [self.n_con_arr, self.c_type, self.c_on, self.c_fa, self.c_ba, self.c_fb, self.c_bb,
                       self.c_pa, self.c_pb, self.c_rest, self.c_comp, self.c_cap, self.axis_p, self.axis_v,
                       self.tri, self.x, self.x_prev, self.w, dt, self.dx, self.dn, self.c_force, 1])
            wp.launch(k_apply_jacobi, n, [self.x, self.dx, self.dn, f64(1.0)])
        if nh:
            # the yarn crossings that were bent past their stick range have slid (bending hysteresis)
            wp.launch(k_hinge_plastic, nh, [self.hinge, self.x, self.hf_T, self.hf_k, self.hf_p, 0, self.hf_diss])
        if nt and self.tens_fric:
            # and those strained or sheared past it (tensile and shear hysteresis)
            wp.launch(k_tri_plastic, nt, [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat, self.x, self.tf_p,
                                          self.tf_F, 0, self.tf_diss])
        # the fabric's compression set: each surface's compression this tick, then the set grows or recovers
        self.cdel.zero_()
        wp.launch(k_set_gather_shapes, n, [self.x, self.radius, self.w, self.is_body] + sh[:7] + [self.cset, self.vnorm,
                                                                                                 self.cdel])
        if cc:
            wp.launch(k_set_gather_pairs, self.pair_capacity, [self.pair_v, self.pair_f, self.pair_b, self.pair_count,
                                                               self.pair_capacity, self.tri, self.x, self.radius, self.pfn,
                                                               self.cset, self.vnorm, self.cdel, self.pfo])
        wp.launch(k_set_update, n, [self.cset, self.cdel, self.crc, self.ck1, self.ccm, self.area, self.w, dt,
                                    f64(self.set_recovery_tau), f64(self.set_recovery_pa)])
        wp.launch(k_velocity, n, [self.x, self.x_prev, self.v, self.w, dt, self.v_start, self.a_prev])

    def _press_device(self):
        pr = self.press
        wp.launch(k_szero, max(len(self.plies), 1), [self.press_mass])
        wp.launch(k_szero, 1, [self.stack_top])
        if pr["n_rect"] > 0:
            wp.launch(k_foot_press, self.n, [self.x, self.radius, self.m, self.ply, self.is_cloth, self.foot_rects,
                                             self.foot_halfs, pr["n_rect"], self.foot_zb_arr, self.foot_depth_arr,
                                             self.pressed, self.press_mass, self.stack_top])
        wp.launch(k_foot_load, self.n, [self.pressed, self.m, self.ply, self.press_mass, self.foot_N, self.load])

    def set_press(self, rects, halves, z_bottom, depth, N):
        """Presser-foot sandwich for this tick: sole rectangles (centres, half extents; world xy), the sole's bottom
        height, the stack depth that counts as pressed, and the foot's normal force N (N)."""
        r = np.zeros((4, 3))
        hh = np.zeros((4, 3))
        r[:len(rects)] = rects
        hh[:len(halves)] = halves
        self._press_host = (r, hh, float(z_bottom), float(depth), float(N), len(rects))

    def _tick_device(self):
        self._press_device()
        if self.mesh is not None:
            wp.launch(k_to32, self.n, [self.x, self.x32])
            self.mesh.refit()
            self.pair_count.zero_()
            qr = wp.float64(2*self.max_radius + self.query_margin)
            self.cc_vmax.zero_()
            wp.launch(k_cloth_vmax, self.n, [self.v, self.is_cloth, self.cc_vmax])
            wp.launch(k_detect_cc, self.n, [self.x, self.Xm, self.ply, self.excl, self.is_cloth, self.mesh.id,
                                            self.tri, qr, self.pair_v, self.pair_f, self.pair_b, self.pair_side,
                                            self.pair_count, self.pair_capacity, self.pair_prev,
                                            wp.float64(self.pair_keep), self.cc_vmax,
                                            wp.float64(self.cc_reach*self.dt), wp.float64(self.cc_reach_max)])
        if self.method == "implicit":
            self._implicit_tick()
        else:
            self._substeps()

    def step(self):
        """Advance one tick."""
        with wp.ScopedDevice(self.device):
            if self._con_dirty:
                self._upload_constraints()
                self._con_dirty = False
            self.s_p0.assign(self.s_p1)
            self.s_q0.assign(self.s_q1)
            self.s_p1.assign(self._sp_host)
            self.s_q1.assign(self._sq_host)
            if getattr(self, "_st_dirty", False):
                self.spring_t.assign(self._st_host)
                self._st_dirty = False
            if getattr(self, "_sk_dirty", False):
                self.spring_k.assign(self._sk_host)
                self._sk_dirty = False
            if getattr(self, "_floor_dirty", False):
                self.body_floor.assign(self.floor_host)
                self._floor_dirty = False
            if getattr(self, "_bP_dirty", False):
                self.body_P.assign(self._bP_host)
                self._bP_dirty = False
            if getattr(self, "_bK_dirty", False):
                self.body_K.assign(self._bK_host)
                self._bK_dirty = False
            if self._on_dirty:
                self.s_on.assign(np.array([int(s["on"]) for s in self.shapes] +
                                          [0]*(self.max_shapes - len(self.shapes)), np.int32))
                self._on_dirty = False
            ph = getattr(self, "_press_host", None)
            if ph is not None:
                self.foot_rects.assign(ph[0])
                self.foot_halfs.assign(ph[1])
                self.foot_zb_arr.assign([ph[2]])
                self.foot_depth_arr.assign([ph[3]])
                self.foot_N.assign([ph[4]])
                if self.press["n_rect"] != ph[5]:
                    self.press["n_rect"] = ph[5]
                    self.graph = None          # the number of sole rectangles is a launch argument
            self.foot_floor.assign([self.foot["floor"]])
            self.foot_force.assign([-self.foot["force"]])
            self.n_con_arr.assign([self.n_con])
            self.acc_f.zero_()
            self.acc_tau.zero_()
            self.acc_n.zero_()
            self.c_force.zero_()
            self.contact_flag.zero_()
            self.foot_acc.zero_()
            wp.copy(self.x_tick0, self.x)
            if self.use_graph and wp.get_device(self.device).is_cuda:
                if self.graph is None:
                    with wp.ScopedCapture() as cap:
                        self._tick_device()
                    self.graph = cap.graph
                wp.capture_launch(self.graph)
            else:
                self._tick_device()
        self.ticks += 1

    # ------------------------------------------------------------------------------------------------ snapshots
    def snapshot(self):
        """Everything that evolves: particle state, shape poses and switches, body floors, external forces and the
        constraint table (pair lists and contact sets are rebuilt every tick)."""
        import copy
        return dict(x=self.x.numpy().copy(), v=self.v.numpy().copy(), v_start=self.v_start.numpy().copy(),
                    a_prev=self.a_prev.numpy().copy(), f_ext=self.f_ext.numpy().copy(),
                    sp=self._sp_host.copy(), sq=self._sq_host.copy(), s_p1=self.s_p1.numpy().copy(),
                    s_q1=self.s_q1.numpy().copy(), on=[s["on"] for s in self.shapes],
                    floor=self.floor_host.copy(), con=copy.deepcopy(self.con_host), n_con=self.n_con,
                    ticks=self.ticks, hf_p=self.hf_p.numpy().copy(), hf_diss=self.hf_diss.numpy().copy(),
                    tf_p=self.tf_p.numpy().copy(), tf_F=self.tf_F.numpy().copy(), tf_diss=self.tf_diss.numpy().copy(),
                    pair_prev=self.pair_prev.numpy().copy(), cset=self.cset.numpy().copy(),
                    fe_prev=self.fe_prev.numpy().copy(), fe_sid_prev=self.fe_sid_prev.numpy().copy(),
                    ccfe=self.ccfe.numpy().copy(), fN_prev=self.fN_prev.numpy().copy(),
                    ccN_peak=self.ccN_peak.numpy().copy())

    def restore(self, snap):
        import copy
        for k in ("x", "v", "v_start", "a_prev", "f_ext", "hf_p", "hf_diss", "tf_p", "tf_F", "tf_diss", "pair_prev",
                  "cset", "fe_prev", "fe_sid_prev", "ccfe", "fN_prev", "ccN_peak"):
            if k in snap:
                getattr(self, k).assign(snap[k])
        self.x_prev.assign(snap["x"])
        self._sp_host[:] = snap["sp"]
        self._sq_host[:] = snap["sq"]
        self.s_p1.assign(snap["s_p1"])
        self.s_q1.assign(snap["s_q1"])
        self.s_p0.assign(snap["s_p1"])
        self.s_q0.assign(snap["s_q1"])
        for s, on in zip(self.shapes, snap["on"]):
            s["on"] = on
        self._on_dirty = True
        self.floor_host[:] = snap["floor"]
        self._floor_dirty = True
        self.con_host = copy.deepcopy(snap["con"])
        self.n_con = snap["n_con"]
        self.c_on.zero_()
        self._con_dirty = True
        self.ticks = snap["ticks"]

    # -------------------------------------------------------------------------------------------------- readout
    def positions(self):
        return self.x.numpy()

    def shape_reaction(self, idx):
        """Average force (N) and torque about the shape's end-of-tick origin (N m) the cloth applied on the shape
        over the last tick."""
        k = 1.0/(self.h*self.dt)
        return -self.acc_f.numpy()[idx]*k, -self.acc_tau.numpy()[idx]*k

    def all_reactions(self):
        k = 1.0/(self.h*self.dt)
        return -self.acc_f.numpy()*k, -self.acc_tau.numpy()*k, -self.acc_n.numpy()*k

    def constraint_forces(self):
        """Average force each constraint applied on the cloth over the last tick (N)."""
        div = self.substeps if self.method != "implicit" else 1
        return self.c_force.numpy()[:self.n_con]/div

    def newton_stats(self):
        """(ticks, extra Newton iterations in total, extra iterations in the last tick, line-search stalls: Newton
        iterations ended after two consecutive fully rejected line searches)."""
        c = self.newton_ctl.numpy()
        return dict(ticks=int(c[2]), extra_iterations=int(c[3]), last_tick_extra=int(c[1]),
                    line_search_stalls=int(self.ls_stats.numpy()[3]), unconverged_ticks=int(c[4]),
                    unconverged_worst_step=float(c[5]))

    def cg_residuals(self):
        """The last linear solve's residual measure per iteration: the sum over the free particles of each one's
        squared unbalanced acceleration |r_i / m_i|^2 (m^2/s^4; k_precond)."""
        return self.cg_hist.numpy()[:int(self.cg_ctl.numpy()[5]) + 1]

    def shear_angles(self, plies=None):
        """Each triangle's shear angle (deg): the angle between its deformed warp and weft directions taken from 90,
        atan2(|F a_w . F a_f|, |F a_w x F a_f|) as the shear lock reads it. plies: indices of the plies
        to include (default all)."""
        if not hasattr(self, "_shear_host"):
            self._shear_host = dict(tri=self.tri_host, Dm=self.Dminv.numpy().astype(np.float64),
                                    aw=self.aw.numpy().astype(np.float64), af=self.af.numpy().astype(np.float64),
                                    ply=self.ply_host[self.tri_host[:, 0]] if len(self.tri_host) else np.zeros(0, int))
        h = self._shear_host
        x = self.x.numpy()
        t = h["tri"]
        r1, r2 = x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]
        gw = np.einsum("tij,tj->ti", h["Dm"], h["aw"])
        gf = np.einsum("tij,tj->ti", h["Dm"], h["af"])
        fw = r1*gw[:, :1] + r2*gw[:, 1:]
        ff = r1*gf[:, :1] + r2*gf[:, 1:]
        g = np.degrees(np.arctan2(np.abs(np.sum(fw*ff, 1)), np.linalg.norm(np.cross(fw, ff), axis=1)))
        if plies is not None:
            g = g[np.isin(h["ply"], np.asarray(list(plies)))]
        return g

    def solve_stats(self):
        """One record of how far the implicit step was solved: per tick the Newton iterations and CG iterations, the
        ticks whose Newton loop ended at its cap unconverged (and the largest last step among them, in tolerances),
        the linear solves that ended at their cap, the line-search stalls."""
        nw, cg = self.newton_stats(), self.cg_stats()
        t = max(nw["ticks"], 1)
        return dict(ticks=nw["ticks"], newton_per_tick=round(cg["solves"]/t, 3), cg_per_tick=round(cg["iterations"]/t, 1),
                    unconverged_ticks=nw["unconverged_ticks"], unconverged_worst_step=round(nw["unconverged_worst_step"], 2),
                    capped_solves=cg["capped"], line_search_stalls=nw["line_search_stalls"])

    def cg_stats(self):
        """Linear solves so far, their CG iterations in total, and the solves that ended at the iteration cap with the
        residual still above the tolerances."""
        c = self.cg_ctl.numpy()
        return dict(solves=int(c[2]), iterations=int(c[3]), capped=int(c[4]), last=int(c[5]))

    def elastic_energy(self):
        self.energy_buf.zero_()
        if len(self.tri_host):
            wp.launch(k_energy, len(self.tri_host), [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat,
                                                     self.x, self.energy_buf, self.tf_p, self.tf_F], device=self.device)
        if len(self.hinge_host):
            wp.launch(k_bend_energy, len(self.hinge_host), [self.hinge, self.hrest, self.kb, self.x, self.energy_buf,
                                                            self.hf_T, self.hf_k, self.hf_p],
                      device=self.device)
        return float(self.energy_buf.numpy()[0])

    def zero_velocities(self):
        """Stop every particle (start-state preparation by dynamic relaxation only; never during an episode)."""
        self.v.zero_()
        self.v_start.zero_()
        self.a_prev.zero_()

    def bending_energy(self):
        """Hinge energy only (elastic plus the sticking yarn crossings), J."""
        self.energy_buf.zero_()
        if len(self.hinge_host):
            wp.launch(k_bend_energy, len(self.hinge_host), [self.hinge, self.hrest, self.kb, self.x, self.energy_buf,
                                                            self.hf_T, self.hf_k, self.hf_p], device=self.device)
        return float(self.energy_buf.numpy()[0])

    def update_bending_friction(self):
        """The end-of-tick return mapping on the present shape (for kinematic probes; step() does it itself)."""
        if len(self.hinge_host):
            wp.launch(k_hinge_plastic, len(self.hinge_host), [self.hinge, self.x, self.hf_T, self.hf_k, self.hf_p, 0,
                                                              self.hf_diss], device=self.device)

    def relax_bending_friction(self):
        """Every hinge's plastic angle to the present shape: the cloth as laid holds no friction torque."""
        if len(self.hinge_host):
            wp.launch(k_hinge_plastic, len(self.hinge_host), [self.hinge, self.x, self.hf_T, self.hf_k, self.hf_p, 1,
                                                              self.hf_diss], device=self.device)

    def relax_tension_friction(self):
        """Every tensile friction element to the present strain, and the slip forces from the present tension."""
        if len(self.tri_host) and self.tens_fric:
            wp.launch(k_tri_plastic, len(self.tri_host), [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat,
                                                          self.x, self.tf_p, self.tf_F, 1, self.tf_diss],
                      device=self.device)

    def update_tension_friction(self):
        """The end-of-tick return mapping on the present shape (for kinematic probes; step() does it itself)."""
        if len(self.tri_host) and self.tens_fric:
            wp.launch(k_tri_plastic, len(self.tri_host), [self.tri, self.Dminv, self.A0, self.aw, self.af, self.mat,
                                                          self.x, self.tf_p, self.tf_F, 0, self.tf_diss],
                      device=self.device)

    def tension_friction_dissipated(self):
        """Work done against the tensile friction since build (J)."""
        return float(self.tf_diss.numpy()[0])

    def bending_friction_dissipated(self):
        """Work done against the bending friction since build (J)."""
        return float(self.hf_diss.numpy()[0])

    def kinetic_energy(self):
        v = self.v.numpy()
        return float(0.5*np.sum(self.mass_host*np.sum(v*v, axis=1)))

    def gravity_energy(self):
        return float(-self.gravity*np.sum(self.mass_host*self.positions()[:, 2]))

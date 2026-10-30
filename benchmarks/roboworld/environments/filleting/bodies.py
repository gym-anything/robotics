"""Rigid bones inside the tissue solver.

Why here and not in MuJoCo: a 2 g backbone segment bonded to ~100 flesh nodes sees ~2e4 N/m of tie
stiffness; coupled explicitly at MuJoCo's 1 ms tick that loop has gain k dt^2 / m ~ 10 (unstable). Integrated
at the tissue substep (2e-5 s) the same loop has gain ~4e-6. So the head and the backbone segments are rigid
bodies of this solver. They carry:
  * capsule SHAPES (centra, neural and haemal spines, skull) that tissue nodes contact like any rigid shape;
    the reactions go into the body instead of to MuJoCo;
  * sphere SAMPLES along those capsules that contact the EXTERNAL shapes (knife blade, board, finger pads);
    those reactions go to MuJoCo as tick-averaged wrenches, like the tissue's;
  * JOINTS between neighbouring bodies (stiff ball joint plus a bending spring); a joint breaks when the knife
    edge has cut through its joint tissue (world.py decides and calls break_joint);
  * TIES: cohesive flesh-bone bonds from tissue nodes (or rod nodes) to points on the bone, bilinear
    traction-separation with irreversible damage, released when a knife blade slab passes the bond.

Semi-implicit (symplectic) Euler for the bodies, gyroscopic term explicit. Bodies can be fixed (kinematic).
"""

import numpy as np
import warp as wp

from .cohesive import cohesive_spent


@wp.func
def body_point(p: wp.vec3, q: wp.quat, loc: wp.vec3):
    return p + wp.quat_rotate(q, loc)


@wp.func
def inv_mass_at(m: float, Ibinv: wp.mat33, q: wp.quat, r: wp.vec3, t: wp.vec3):
    """A body's inverse effective mass for a force along unit t at arm r (world) from its centre of mass:
    1/m + (r x t).I^-1.(r x t), with I^-1 given in the body frame."""
    a = wp.quat_rotate_inv(q, wp.cross(r, t))
    return 1.0 / m + wp.dot(a, Ibinv * a)


@wp.kernel
def k_body_shapes(ns: int, s_body: wp.array(dtype=wp.int32), s_loc_p: wp.array(dtype=wp.vec3),
                  s_loc_q: wp.array(dtype=wp.quat), rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat),
                  rb_v: wp.array(dtype=wp.vec3), rb_w: wp.array(dtype=wp.vec3),
                  s_p0: wp.array(dtype=wp.vec3), s_q0: wp.array(dtype=wp.quat), s_v: wp.array(dtype=wp.vec3),
                  s_w: wp.array(dtype=wp.vec3)):
    """Pose of body-owned shapes from their body (current substep; contact kernels use t = 0 for them)."""
    s = wp.tid()
    if s >= ns:
        return
    b = s_body[s]
    if b < 0:
        return
    q = rb_q[b]
    r = wp.quat_rotate(q, s_loc_p[s])
    s_p0[s] = rb_p[b] + r
    s_q0[s] = q * s_loc_q[s]
    s_v[s] = rb_v[b] + wp.cross(rb_w[b], r)
    s_w[s] = rb_w[b]


@wp.func
def ext_sdf(st: int, pl: wp.vec3, h: wp.vec3, par: wp.vec2):
    """Penetration depth of a local point into an external shape (box / wedge / capsule) and local normal."""
    pen = float(-1.0)
    nl = wp.vec3(0.0, 0.0, 0.0)
    if st == 0:
        d = wp.vec3(wp.abs(pl[0]) - h[0], wp.abs(pl[1]) - h[1], wp.abs(pl[2]) - h[2])
        o = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
        ol = wp.length(o)
        if ol > 0.0:
            pen = -ol
            nl = wp.vec3(wp.sign(pl[0]) * o[0], wp.sign(pl[1]) * o[1], wp.sign(pl[2]) * o[2]) / ol
        else:
            dist = d[0]
            nl = wp.vec3(wp.sign(pl[0]), 0.0, 0.0)
            if d[1] > dist:
                dist = d[1]
                nl = wp.vec3(0.0, wp.sign(pl[1]), 0.0)
            if d[2] > dist:
                dist = d[2]
                nl = wp.vec3(0.0, 0.0, wp.sign(pl[2]))
            pen = -dist
    elif st == 1:
        # the blade as the intersection of half-spaces: bevel faces |y| <= z tan(b), flanks
        # |y| <= e + s z (s = (hy - e)/H), spine z <= H, tip/heel |x| <= hx. The signed distance is the largest
        # plane distance (exact inside, a lower bound outside), so spheres of finite radius (bone samples)
        # meet the blade before their centre enters it
        H = 2.0 * h[2]
        sy = wp.sign(pl[1])
        if sy == 0.0:
            sy = 1.0
        ay = wp.abs(pl[1])
        tb = par[0]
        cb = 1.0 / wp.sqrt(1.0 + tb * tb)
        dmax = (ay - pl[2] * tb) * cb
        nl = wp.vec3(0.0, sy * cb, -tb * cb)
        sl = (h[1] - par[1]) / H
        cf = 1.0 / wp.sqrt(1.0 + sl * sl)
        df = (ay - par[1] - sl * pl[2]) * cf
        if df > dmax:
            dmax = df
            nl = wp.vec3(0.0, sy * cf, -sl * cf)
        dt_ = pl[2] - H
        if dt_ > dmax:
            dmax = dt_
            nl = wp.vec3(0.0, 0.0, 1.0)
        dx = wp.abs(pl[0]) - h[0]
        if dx > dmax:
            dmax = dx
            nl = wp.vec3(wp.sign(pl[0]), 0.0, 0.0)
        pen = -dmax
    else:
        x = wp.clamp(pl[0], -h[0], h[0])
        d2 = pl - wp.vec3(x, 0.0, 0.0)
        l = wp.length(d2)
        if l > 1.0e-12:
            pen = h[1] - l
            nl = d2 / l
    return pen, nl


@wp.kernel
def k_sample_candidates(n: int, smp_body: wp.array(dtype=wp.int32), smp_loc: wp.array(dtype=wp.vec3),
                        smp_r: wp.array(dtype=float), rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat),
                        n_shapes: int, s_body: wp.array(dtype=wp.int32), s_p0: wp.array(dtype=wp.vec3),
                        s_rad: wp.array(dtype=float), s_enabled: wp.array(dtype=wp.int32),
                        group: wp.array(dtype=wp.int32), margin: float, cap: int,
                        cand: wp.array2d(dtype=wp.int32), cand_n: wp.array(dtype=wp.int32),
                        anchor_on: wp.array2d(dtype=wp.int32)):
    """Once per rigid tick: the external shapes each bone/blade/handle sample can reach within the tick (as
    tissue.k_contact_candidates); more than `cap` -> every shape (cand_n = -1). Friction anchors on shapes out of
    reach are released."""
    i = wp.tid()
    if i >= n:
        return
    b = smp_body[i]
    xi = rb_p[b] + wp.quat_rotate(rb_q[b], smp_loc[i])
    c = int(0)
    over = int(0)
    for s in range(n_shapes):
        sb = s_body[s]
        if s_enabled[s] == 0 or sb == b:
            continue
        if sb >= 0 and group[sb] == group[b]:
            continue
        if wp.length(xi - s_p0[s]) > s_rad[s] + smp_r[i] + margin:
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
def k_body_samples(n: int, smp_body: wp.array(dtype=wp.int32), smp_loc: wp.array(dtype=wp.vec3),
                   smp_r: wp.array(dtype=float), smp_k: wp.array(dtype=float), smp_c: wp.array(dtype=float),
                   rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_v: wp.array(dtype=wp.vec3),
                   rb_w: wp.array(dtype=wp.vec3), rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3),
                   n_shapes: int, s_body: wp.array(dtype=wp.int32), s_type: wp.array(dtype=wp.int32),
                   s_p0: wp.array(dtype=wp.vec3), s_q0: wp.array(dtype=wp.quat), s_v: wp.array(dtype=wp.vec3),
                   s_w: wp.array(dtype=wp.vec3), s_h: wp.array(dtype=wp.vec3), s_param: wp.array(dtype=wp.vec2),
                   s_com: wp.array(dtype=wp.vec3), s_mu: wp.array2d(dtype=float), s_enabled: wp.array(dtype=wp.int32),
                   s_rad: wp.array(dtype=float), t_local: float, dt: float, wrench: wp.array2d(dtype=float),
                   work_in: wp.array(dtype=wp.float64), fric: wp.array(dtype=wp.float64),
                   damp: wp.array(dtype=wp.float64), cload: wp.array(dtype=float), group: wp.array(dtype=wp.int32),
                   rb_fixed: wp.array(dtype=wp.int32), cand: wp.array2d(dtype=wp.int32),
                   cand_n: wp.array(dtype=wp.int32), knife_held: wp.array(dtype=wp.int32),
                   kload: wp.array(dtype=float), rb_ms: wp.array(dtype=float), rb_m: wp.array(dtype=float),
                   rb_Ibinv: wp.array(dtype=wp.mat33), anchor: wp.array2d(dtype=wp.vec3),
                   anchor_on: wp.array2d(dtype=wp.int32), a_ft: wp.array2d(dtype=wp.vec3),
                   a_d: wp.array2d(dtype=wp.vec3)):
    """Bone samples (spheres on the bone capsules) against the EXTERNAL rigid shapes they can reach this tick
    (k_sample_candidates). Penalty normal force with a critically damped dashpot; Coulomb friction by stick-slip:
    a tangential spring to an anchor kept in the shape's frame, slipping (the anchor dragged along, lim x slip
    booked) when its force exceeds mu fn, plus a light dashpot on the tangential velocity while it sticks, as the
    tissue's own contacts (tissue.k_contacts). A velocity dashpot alone, capped for explicit stability, cannot reach
    mu fn at the slow sliding of a pressed blade (an effective mu of 0.01 against 0.25 at 20 mm/s), so nothing would
    hold by static friction. A driven body's own samples are inactive: its motion is prescribed by its owner (the
    gripped knife handle follows MuJoCo, which resolves the handle's contacts); the pairs it takes part in are still
    resolved from the other body's samples."""
    i = wp.tid()
    if i >= n:
        return
    b = smp_body[i]
    if rb_fixed[b] == 2:
        return
    q = rb_q[b]
    r = wp.quat_rotate(q, smp_loc[i])
    xi = rb_p[b] + r
    vi = rb_v[b] + wp.cross(rb_w[b], r)
    nc = cand_n[i]
    count = nc
    if nc < 0:
        count = n_shapes
    for j in range(count):
        s = j
        if nc >= 0:
            s = cand[i, j]
        sb = s_body[s]
        if s_enabled[s] == 0 or sb == b:
            continue
        if sb >= 0 and group[sb] == group[b]:
            continue          # bodies of one assembly (the skeleton; the knife) do not collide with each other
        if sb < 0 and knife_held[0] == 1 and group[b] == 1:
            continue          # a held knife meets rigid geometry and the robot in MuJoCo (scene.blade_rigid_shapes)
        tl = t_local
        if sb >= 0:
            tl = 0.0                      # body-owned shapes are posed every substep
        from_c = xi - (s_p0[s] + s_v[s] * tl)
        if wp.length(from_c) > s_rad[s] + smp_r[i] + 0.02:
            anchor_on[i, s] = 0
            continue
        ang = wp.length(s_w[s]) * tl
        qs = s_q0[s]
        if ang > 1.0e-12:
            qs = wp.quat_from_axis_angle(wp.normalize(s_w[s]), ang) * qs
        X = wp.transform(s_p0[s] + s_v[s] * tl, qs)
        pl = wp.transform_point(wp.transform_inverse(X), xi)
        pen, nl = ext_sdf(s_type[s], pl, s_h[s], s_param[s])
        pen = pen + smp_r[i]
        if pen <= 0.0:
            anchor_on[i, s] = 0
            continue
        nw = wp.transform_vector(X, nl)
        com = wp.transform_point(X, s_com[s])
        vs = s_v[s] + wp.cross(s_w[s], xi - wp.transform_get_translation(X))
        vrel = vi - vs
        vn = wp.dot(vrel, nw)
        fn = wp.max(smp_k[i] * pen - smp_c[i] * vn, 0.0)
        if fn > 0.0:
            wp.atomic_add(damp, 0, wp.float64(wp.min(smp_c[i] * vn * vn, fn * wp.abs(vn)) * dt))
        vt = vrel - vn * nw
        vtl = wp.length(vt)
        ft = wp.vec3(0.0, 0.0, 0.0)
        mu = s_mu[s, 2]
        if mu > 0.0:
            # stick-slip: the anchor is where the sample first touched this shape (shape frame); the spring pulls the
            # sample back to it while |f| <= mu fn, and past that the anchor is dragged along (the slip is dissipated)
            if anchor_on[i, s] == 0:
                anchor[i, s] = pl
                anchor_on[i, s] = 1
                a_ft[i, s] = wp.vec3(0.0, 0.0, 0.0)
                a_d[i, s] = wp.vec3(0.0, 0.0, 0.0)
            aw = wp.transform_point(X, anchor[i, s])
            d_w = xi - aw
            d_w = d_w - wp.dot(d_w, nw) * nw
            # booked: the work the friction force applied over the last step did on the relative motion since (exact
            # for an explicit step, the force being constant over it), the anchor spring's stored energy with it
            # (returned as it unloads). Booking the current limit times the last step's slip would mismatch the force
            # that acted whenever the normal force changed between steps: on blades chattering over light bones it
            # books more friction than the work that drives them (A1, A4)
            dprev = wp.transform_vector(X, a_d[i, s])
            dprev = dprev - wp.dot(dprev, nw) * nw
            wp.atomic_add(fric, 0, wp.float64(-wp.dot(wp.transform_vector(X, a_ft[i, s]), d_w - dprev)))
            # stiffness and damping: each sample's share of its body's explicit limit at the contact point, rotation
            # included (1/m_eff = 1/m + (r x t).I^-1.(r x t)): all of a body's samples sticking together stay stable
            # (a cap on the mass alone leaves bones and blade segments 2-6 times over it in rotation)
            tdir = d_w
            if wp.length(tdir) < 1.0e-12:
                tdir = vt
            tl_ = wp.length(tdir)
            im = 1.0 / rb_m[b]
            if tl_ > 1.0e-12:
                im = inv_mass_at(rb_m[b], rb_Ibinv[b], q, r, tdir / tl_)
            share = rb_ms[b] / (rb_m[b] * im)
            if sb >= 0:
                if rb_fixed[sb] == 0 or rb_fixed[sb] == 3:
                    im_s = 1.0 / rb_m[sb]
                    if tl_ > 1.0e-12:
                        im_s = inv_mass_at(rb_m[sb], rb_Ibinv[sb], rb_q[sb], xi - rb_p[sb], tdir / tl_)
                    share = wp.min(share, rb_ms[sb] / (rb_m[sb] * im_s))
            kt = wp.min(0.5 * smp_k[i], 0.25 * share / (dt * dt))
            ct = 2.0 * 0.7 * wp.sqrt(kt * share)
            fs = -kt * d_w
            fsl = wp.length(fs)
            lim = mu * fn
            if fsl > lim:
                # slip: the spring sits on the Coulomb limit and its anchor is dragged along; the slip is dissipated
                sdir = fs / fsl
                ft = lim * sdir
                newa = xi + (lim / kt) * sdir
                anchor[i, s] = wp.transform_point(wp.transform_inverse(X), newa)
            else:
                # stick: the spring plus a light dashpot on the tangential velocity (70 % of critical on the
                # sample's share), the sum held to the Coulomb limit
                ft = fs - ct * vt
                ftl = wp.length(ft)
                if ftl > lim:
                    ft = ft * (lim / ftl)
            # what the next step books against: this step's force and the offset from the (moved) anchor, shape frame
            d2 = xi - wp.transform_point(X, anchor[i, s])
            d2 = d2 - wp.dot(d2, nw) * nw
            Xi = wp.transform_inverse(X)
            a_d[i, s] = wp.transform_vector(Xi, d2)
            a_ft[i, s] = wp.transform_vector(Xi, ft)
        fc = fn * nw + ft
        wp.atomic_add(rb_f, b, fc)
        wp.atomic_add(rb_t, b, wp.cross(r, fc))
        if sb >= 0:
            # another body of this solver (blade on bone): equal and opposite, internal to the ledger
            wp.atomic_sub(rb_f, sb, fc)
            wp.atomic_sub(rb_t, sb, wp.cross(xi - rb_p[sb], fc))
        else:
            tq = wp.cross(xi - com, -fc)
            wp.atomic_add(wrench, s, 0, -fc[0])
            wp.atomic_add(wrench, s, 1, -fc[1])
            wp.atomic_add(wrench, s, 2, -fc[2])
            wp.atomic_add(wrench, s, 3, tq[0])
            wp.atomic_add(wrench, s, 4, tq[1])
            wp.atomic_add(wrench, s, 5, tq[2])
            wp.atomic_add(work_in, 0, wp.float64(wp.dot(fc, vs) * dt))
        wp.atomic_add(cload, s, fn)
        if group[b] == 1:
            wp.atomic_add(kload, s, fn)             # the knife's own load on this shape (grip detection, free knife)


@wp.kernel
def k_body_joints(nj: int, ja: wp.array(dtype=wp.int32), jb: wp.array(dtype=wp.int32),
                  jpa: wp.array(dtype=wp.vec3), jpb: wp.array(dtype=wp.vec3), jk: wp.array(dtype=float),
                  jc: wp.array(dtype=float), jka: wp.array(dtype=wp.vec3), jca: wp.array(dtype=wp.vec3),
                  jq0: wp.array(dtype=wp.quat), jon: wp.array(dtype=wp.int32),
                  rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_v: wp.array(dtype=wp.vec3),
                  rb_w: wp.array(dtype=wp.vec3), rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3),
                  jload: wp.array(dtype=wp.vec3), dt: float, damp: wp.array(dtype=wp.float64)):
    """Joint between two bodies: stiff spring-damper between the two anchor points plus a rotational
    spring-damper on the relative rotation from its rest value, with separate stiffness/damping about the three
    axes of body a's frame (a knife hinge is soft flat-wise and stiff edge-wise; a vertebral joint is
    isotropic)."""
    j = wp.tid()
    if j >= nj or jon[j] == 0:
        return
    a = ja[j]
    b = jb[j]
    ra = wp.quat_rotate(rb_q[a], jpa[j])
    rbv = wp.quat_rotate(rb_q[b], jpb[j])
    pa = rb_p[a] + ra
    pb = rb_p[b] + rbv
    va = rb_v[a] + wp.cross(rb_w[a], ra)
    vb = rb_v[b] + wp.cross(rb_w[b], rbv)
    fl = jk[j] * (pb - pa) + jc[j] * (vb - va)
    wp.atomic_add(damp, 0, wp.float64(jc[j] * wp.dot(vb - va, vb - va) * dt))
    # relative rotation error rv = 2 vec(qe) of qe = qb (qa q0)^-1 (sign-folded; world vector vw), resolved in a's
    # frame; energy E = 1/2 sum_i ka_i rv_i^2 (what the ledger books). Its exact torque on b
    # for a world rotation dphi of b is -(we t + vw x t / 2) with t = R_a (ka . rv) and we the scalar part of qe:
    # the small-angle form -t is not a gradient beyond a few degrees, and for the knife's anisotropic hinge a loop
    # mixing torsion and the stiff edge-wise axis would do net work. The torque on a is the opposite (E is invariant
    # under a common rotation).
    qe = rb_q[b] * wp.quat_inverse(rb_q[a] * jq0[j])
    s = 1.0
    if qe[3] < 0.0:
        s = -1.0
    vw = wp.vec3(qe[0], qe[1], qe[2]) * (2.0 * s)
    we = qe[3] * s
    rv = wp.quat_rotate_inv(rb_q[a], vw)
    dw = wp.quat_rotate_inv(rb_q[a], rb_w[b] - rb_w[a])
    ka = jka[j]
    ca = jca[j]
    t_el = wp.quat_rotate(rb_q[a], wp.vec3(ka[0] * rv[0], ka[1] * rv[1], ka[2] * rv[2]))
    t_d = wp.quat_rotate(rb_q[a], wp.vec3(ca[0] * dw[0], ca[1] * dw[1], ca[2] * dw[2]))
    tq = we * t_el + 0.5 * wp.cross(vw, t_el) + t_d
    wp.atomic_add(damp, 0, wp.float64((ca[0] * dw[0] * dw[0] + ca[1] * dw[1] * dw[1] + ca[2] * dw[2] * dw[2]) * dt))
    wp.atomic_add(rb_f, a, fl)
    wp.atomic_add(rb_t, a, wp.cross(ra, fl) + tq)
    wp.atomic_sub(rb_f, b, fl)
    wp.atomic_sub(rb_t, b, wp.cross(rbv, fl) + tq)
    jload[j] = fl


@wp.kernel
def k_body_ties(n: int, node: wp.array(dtype=wp.int32), body: wp.array(dtype=wp.int32), loc: wp.array(dtype=wp.vec3),
                k: wp.array(dtype=float), fmax: wp.array(dtype=float), dfull: wp.array(dtype=float),
                damage: wp.array(dtype=float), kd: wp.array(dtype=float),
                x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_v: wp.array(dtype=wp.vec3),
                rb_w: wp.array(dtype=wp.vec3), rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3),
                dt: float, dissip: wp.array(dtype=wp.float64), damp: wp.array(dtype=wp.float64)):
    """Flesh-bone bond: bilinear cohesive law between a particle and a point fixed on a bone."""
    i = wp.tid()
    if i >= n or damage[i] >= 1.0:
        return
    b = body[i]
    r = wp.quat_rotate(rb_q[b], loc[i])
    xb = rb_p[b] + r
    vb = rb_v[b] + wp.cross(rb_w[b], r)
    p = node[i]
    d = x[p] - xb
    dl = wp.length(d)
    kk = k[i]
    d0 = fmax[i] / kk
    D = damage[i]
    if dl > d0:
        df = dfull[i]
        Dn = wp.clamp(df * (dl - d0) / (dl * (df - d0)), 0.0, 1.0)
        if Dn > D:
            wp.atomic_add(dissip, 0, wp.float64(0.5 * fmax[i] * df * (cohesive_spent(Dn, d0, df) - cohesive_spent(D, d0, df))))
            D = Dn
            damage[i] = D
    # dashpot on the node's velocity relative to the bone's rigid motion carried to the node (objective: a tie
    # rotating rigidly with its bone, open or not, dissipates nothing; damping the full relative velocity of the node
    # and the bone point would resist the rigid rotation of an open tie); its reaction acts on the bone at the node
    dv = v[p] - (vb + wp.cross(rb_w[b], d))
    fe = (1.0 - D) * kk * d
    fd = (1.0 - D) * kd[i] * dv
    wp.atomic_add(damp, 0, wp.float64((1.0 - D) * kd[i] * wp.dot(dv, dv) * dt))
    wp.atomic_sub(f, p, fe + fd)
    wp.atomic_add(rb_f, b, fe + fd)
    wp.atomic_add(rb_t, b, wp.cross(r, fe) + wp.cross(r + d, fd))


@wp.kernel
def k_body_tie_release(n: int, node: wp.array(dtype=wp.int32), body: wp.array(dtype=wp.int32),
                       loc: wp.array(dtype=wp.vec3), damage: wp.array(dtype=float), x: wp.array(dtype=wp.vec3),
                       rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat),
                       n_blades: int, blade_ids: wp.array(dtype=wp.int32), s_p0: wp.array(dtype=wp.vec3),
                       s_q0: wp.array(dtype=wp.quat), s_h: wp.array(dtype=wp.vec3), margin: float, below: float,
                       released: wp.array(dtype=wp.int32), fmax: wp.array(dtype=float), dfull: wp.array(dtype=float),
                       s_body: wp.array(dtype=wp.int32), debt: wp.array(dtype=float), kst: wp.array(dtype=float),
                       iface: wp.array(dtype=wp.float64)):
    """A bond whose midpoint lies in a blade segment's slab on the spine side of the edge has been cut by the
    edge (the edge passed between flesh and bone): damage = 1. Its unspent cohesive energy (1 - spent) fmax dfull / 2
    (cohesive.py) is what the cut costs. Part of it is the link's stored elastic energy (1 - D) k d^2 / 2, which
    vanishes with the link and is booked here; the rest is owed by the blade segment's body (paid by
    k_body_integrate's interface drag)."""
    i = wp.tid()
    if i >= n or damage[i] >= 1.0:
        return
    b = body[i]
    xb = rb_p[b] + wp.quat_rotate(rb_q[b], loc[i])
    m = 0.5 * (x[node[i]] + xb)
    for k in range(n_blades):
        s = blade_ids[k]
        X = wp.transform(s_p0[s], s_q0[s])
        pl = wp.transform_point(wp.transform_inverse(X), m)
        h = s_h[s]
        if wp.abs(pl[0]) <= h[0] and pl[2] >= -below and pl[2] <= 2.0 * h[2] and wp.abs(pl[1]) <= h[1] + margin:
            dd = x[node[i]] - xb
            stored = 0.5 * (1.0 - damage[i]) * kst[i] * wp.dot(dd, dd)
            unspent = (1.0 - cohesive_spent(damage[i], fmax[i] / kst[i], dfull[i])) * 0.5 * fmax[i] * dfull[i]
            wp.atomic_add(debt, s_body[s], wp.max(unspent - stored, 0.0))
            wp.atomic_add(iface, 0, wp.float64(stored))
            damage[i] = 1.0
            wp.atomic_add(released, 0, 1)
            return


@wp.kernel
def k_body_drive(nb: int, rb_fixed: wp.array(dtype=wp.int32), rb_p0: wp.array(dtype=wp.vec3),
                 rb_q0: wp.array(dtype=wp.quat), rb_v: wp.array(dtype=wp.vec3), rb_w: wp.array(dtype=wp.vec3),
                 rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), t: float):
    """Driven bodies (rb_fixed == 2) follow the host-given pose at the tick start and twist over the tick."""
    b = wp.tid()
    if b >= nb or rb_fixed[b] != 2:
        return
    rb_p[b] = rb_p0[b] + rb_v[b] * t
    w = rb_w[b]
    ang = wp.length(w) * t
    q = rb_q0[b]
    if ang > 1.0e-12:
        q = wp.quat_from_axis_angle(wp.normalize(w), ang) * q
    rb_q[b] = q


@wp.kernel
def k_body_servo(nb: int, rb_fixed: wp.array(dtype=wp.int32), rb_p0: wp.array(dtype=wp.vec3),
                 rb_q0: wp.array(dtype=wp.quat), rb_vt: wp.array(dtype=wp.vec3), rb_wt: wp.array(dtype=wp.vec3),
                 rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_v: wp.array(dtype=wp.vec3),
                 rb_w: wp.array(dtype=wp.vec3), rb_m: wp.array(dtype=float), rb_Ib: wp.array(dtype=wp.mat33),
                 g: wp.vec3, kp: float, kr: float, f_max: float, t_max: float, t: float, dt: float,
                 rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3), rb_fserv: wp.array(dtype=wp.vec3),
                 rb_tserv: wp.array(dtype=wp.vec3), work_in: wp.array(dtype=wp.float64)):
    """ASSISTED rig on servoed bodies (rb_fixed == 3): a force-limited servo toward the host-given target (pose at the
    tick start, twist over the tick), as a hand holds a knife. Force kp*e + kd*de capped at f_max, torque kr*e_r +
    dr*de_r capped at t_max, both critically damped with the body's own mass and inertia, plus the body's weight
    carried; the work it does is external (work_in, booked where the body is integrated)."""
    b = wp.tid()
    if b >= nb or rb_fixed[b] != 3:
        return
    pt = rb_p0[b] + rb_vt[b] * t
    wt = rb_wt[b]
    ang = wp.length(wt) * t
    qt = rb_q0[b]
    if ang > 1.0e-12:
        qt = wp.quat_from_axis_angle(wp.normalize(wt), ang) * qt
    m = rb_m[b]
    kd = 2.0 * wp.sqrt(kp * m)
    f = kp * (pt - rb_p[b]) + kd * (rb_vt[b] - rb_v[b])
    fl = wp.length(f)
    if fl > f_max:
        f = f * (f_max / fl)
    f = f - m * g                                          # the rig carries the knife's weight
    # orientation error as a rotation vector (world), shortest way round
    qe = qt * wp.quat_inverse(rb_q[b])
    s = 1.0
    if qe[3] < 0.0:
        s = -1.0
    ev = wp.vec3(qe[0], qe[1], qe[2]) * (2.0 * s)
    R = wp.quat_to_matrix(rb_q[b])
    Iw = R * rb_Ib[b] * wp.transpose(R)
    i_eff = (Iw[0, 0] + Iw[1, 1] + Iw[2, 2]) / 3.0
    dr = 2.0 * wp.sqrt(kr * i_eff)
    tq = kr * ev + dr * (wt - rb_w[b])
    tl = wp.length(tq)
    if tl > t_max:
        tq = tq * (t_max / tl)
    rb_f[b] = rb_f[b] + f
    rb_t[b] = rb_t[b] + tq
    rb_fserv[b] = f
    rb_tserv[b] = tq
    # (its work is booked in k_body_integrate against the updated velocity, as the integrator moves the body)


@wp.kernel
def k_body_integrate(nb: int, rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat),
                     rb_v: wp.array(dtype=wp.vec3), rb_w: wp.array(dtype=wp.vec3), rb_m: wp.array(dtype=float),
                     rb_Ib: wp.array(dtype=wp.mat33), rb_Ibinv: wp.array(dtype=wp.mat33),
                     rb_f: wp.array(dtype=wp.vec3), rb_t: wp.array(dtype=wp.vec3), rb_fixed: wp.array(dtype=wp.int32),
                     g: wp.vec3, dt: float, rb_fsum: wp.array(dtype=wp.vec3), rb_tsum: wp.array(dtype=wp.vec3),
                     work_in: wp.array(dtype=wp.float64), debt: wp.array(dtype=float), d_pay: float,
                     iface: wp.array(dtype=wp.float64), rb_fserv: wp.array(dtype=wp.vec3),
                     rb_tserv: wp.array(dtype=wp.vec3), rb_pc: wp.array(dtype=wp.vec3), rb_qc: wp.array(dtype=wp.vec4)):
    b = wp.tid()
    if b >= nb:
        return
    if rb_fixed[b] == 3:
        # a servoed body moves dynamically; the load read back is what the rest of the world applies to it (its
        # servo force excluded), as for a driven body
        rb_fsum[b] = rb_fsum[b] + rb_f[b] - rb_fserv[b]
        rb_tsum[b] = rb_tsum[b] + rb_t[b] - rb_tserv[b]
    elif rb_fixed[b] != 0:
        # fixed or driven: the load it receives is summed for readback (a driven body passes it to MuJoCo); a
        # driven body is an external energy source: the power it delivers is -load . its velocity
        rb_fsum[b] = rb_fsum[b] + rb_f[b]
        rb_tsum[b] = rb_tsum[b] + rb_t[b]
        if rb_fixed[b] == 2:
            wp.atomic_add(work_in, 0, wp.float64(-(wp.dot(rb_f[b], rb_v[b]) + wp.dot(rb_t[b], rb_w[b])) * dt))
        rb_f[b] = wp.vec3(0.0, 0.0, 0.0)
        rb_t[b] = wp.vec3(0.0, 0.0, 0.0)
        return
    q = rb_q[b]
    R = wp.quat_to_matrix(q)
    Iw = R * rb_Ib[b] * wp.transpose(R)
    Iinv = R * rb_Ibinv[b] * wp.transpose(R)
    w = rb_w[b]
    v = rb_v[b] + (rb_f[b] / rb_m[b] + g) * dt
    # interface resistance: a blade that released interface bonds or flesh-bone ties pays their unspent fracture
    # energy from its own motion, as a resisting force owed/d_pay against its velocity (the energy is paid over d_pay
    # of travel; in steady cutting the force equals the interface energy released per unit travel, whatever d_pay).
    # Never more than half the velocity per substep, so it only ever removes kinetic energy; what it removes is
    # booked to the iface ledger.
    owed = debt[b]
    if owed > 0.0:
        m = rb_m[b]
        vv = wp.dot(v, v)
        sp = wp.sqrt(vv)
        dv = wp.min(owed / d_pay * dt / m, 0.5 * sp)
        v_new = v
        if sp > 1.0e-9:
            v_new = v * (1.0 - dv / sp)
        paid = 0.5 * m * (vv - wp.dot(v_new, v_new))
        v = v_new
        debt[b] = wp.max(owed - paid, 0.0)
        wp.atomic_add(iface, 0, wp.float64(paid))
    w = w + Iinv * (rb_t[b] - wp.cross(w, Iw * w)) * dt
    if rb_fixed[b] == 3:
        # the rig's servo: its work over this substep, with the velocities the body now moves by
        wp.atomic_add(work_in, 0, wp.float64((wp.dot(rb_fserv[b], v) + wp.dot(rb_tserv[b], w)) * dt))
    rb_v[b] = v
    rb_w[b] = w
    # compensated (Kahan) summation of the position and orientation updates: at bench height a float32 coordinate
    # resolves 6e-8 m, more than a substep's travel below ~3 mm/s, so without it a slow body does not move at all while
    # its velocity says it does (the rig's servoed handle: 1 mm/s, frozen; the tissue nodes use the same summation)
    p = rb_p[b]
    y = v * dt - rb_pc[b]
    pt = p + y
    rb_pc[b] = (pt - p) - y
    rb_p[b] = pt
    dq = wp.quat(w[0], w[1], w[2], 0.0) * q
    qc = rb_qc[b]
    yq = wp.vec4(0.5 * dt * dq[0] - qc[0], 0.5 * dt * dq[1] - qc[1], 0.5 * dt * dq[2] - qc[2], 0.5 * dt * dq[3] - qc[3])
    tq = wp.vec4(q[0] + yq[0], q[1] + yq[1], q[2] + yq[2], q[3] + yq[3])
    rb_qc[b] = wp.vec4(tq[0] - q[0], tq[1] - q[1], tq[2] - q[2], tq[3] - q[3]) - yq
    rb_q[b] = wp.normalize(wp.quat(tq[0], tq[1], tq[2], tq[3]))
    rb_f[b] = wp.vec3(0.0, 0.0, 0.0)
    rb_t[b] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_body_energy(nb: int, rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat),
                  rb_v: wp.array(dtype=wp.vec3), rb_w: wp.array(dtype=wp.vec3), rb_m: wp.array(dtype=float),
                  rb_Ib: wp.array(dtype=wp.mat33), g: wp.vec3, ke: wp.array(dtype=wp.float64),
                  pe: wp.array(dtype=wp.float64), rb_fixed: wp.array(dtype=wp.int32)):
    b = wp.tid()
    if b >= nb or (rb_fixed[b] != 0 and rb_fixed[b] != 3):
        return
    R = wp.quat_to_matrix(rb_q[b])
    Iw = R * rb_Ib[b] * wp.transpose(R)
    w = rb_w[b]
    v = rb_v[b]
    wp.atomic_add(ke, 0, wp.float64(0.5 * rb_m[b] * wp.dot(v, v) + 0.5 * wp.dot(w, Iw * w)))
    wp.atomic_add(pe, 0, wp.float64(-rb_m[b] * wp.dot(g, rb_p[b])))


def capsule_frame(p0, p1):
    """Local frame of a capsule whose axis runs p0 -> p1 along local x: (centre, quat xyzw, half length)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = .5*(p0 + p1)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-12:
        return c, np.array([0., 0., 0., 1.]), 0.
    ax = d/L
    x = np.array([1., 0., 0.])
    v = np.cross(x, ax)
    s = np.linalg.norm(v)
    cth = float(np.dot(x, ax))
    if s < 1e-12:
        q = np.array([0., 0., 0., 1.]) if cth > 0 else np.array([0., 0., 1., 0.])
    else:
        ang = np.arctan2(s, cth)
        q = np.concatenate([v/s*np.sin(ang/2), [np.cos(ang/2)]])
    return c, q, L/2


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rot(q, v):
    x, y, z, w = q
    R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    return R @ np.asarray(v, float)


# ------------------------------------------------------------------------------------------ interface bonds
@wp.kernel
def k_bonds(n: int, ba: wp.array(dtype=wp.int32), bb: wp.array(dtype=wp.int32), k: wp.array(dtype=float),
            fmax: wp.array(dtype=float), dfull: wp.array(dtype=float), damage: wp.array(dtype=float),
            kd: wp.array(dtype=float), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            f: wp.array(dtype=wp.vec3), dt: float, dissip: wp.array(dtype=wp.float64),
            damp: wp.array(dtype=wp.float64)):
    """Pre-inserted anatomical interface (median septum, rib-cage lining, skin-flesh, silver skin, chain seam):
    the two copies of a node on the interface are bonded by a bilinear cohesive law (strength fmax = sigma * A,
    fracture energy G * A over the separation dfull = 2 G / sigma) with irreversible damage. Tearing the interface
    by pulling dissipates G per area; a passing blade releases it (k_bond_release)."""
    i = wp.tid()
    if i >= n or damage[i] >= 1.0:
        return
    a = ba[i]
    b = bb[i]
    d = x[b] - x[a]
    dl = wp.length(d)
    kk = k[i]
    d0 = fmax[i] / kk
    D = damage[i]
    if dl > d0:
        df = dfull[i]
        Dn = wp.clamp(df * (dl - d0) / (dl * (df - d0)), 0.0, 1.0)
        if Dn > D:
            wp.atomic_add(dissip, 0, wp.float64(0.5 * fmax[i] * df * (cohesive_spent(Dn, d0, df) - cohesive_spent(D, d0, df))))
            D = Dn
            damage[i] = D
    # dashpot on the rate of opening along the link while it is open (objective: an open bond rotating rigidly
    # dissipates nothing), on the full relative velocity of the two copies while they coincide
    dv = v[b] - v[a]
    if dl > 1.0e-9:
        u = d / dl
        dv = u * wp.dot(dv, u)
    fs = (1.0 - D) * (kk * d + kd[i] * dv)
    wp.atomic_add(damp, 0, wp.float64((1.0 - D) * kd[i] * wp.dot(dv, dv) * dt))
    wp.atomic_add(f, a, fs)
    wp.atomic_sub(f, b, fs)


@wp.kernel
def k_bond_release(n: int, ba: wp.array(dtype=wp.int32), bb: wp.array(dtype=wp.int32), damage: wp.array(dtype=float),
                   x: wp.array(dtype=wp.vec3), n_blades: int, blade_ids: wp.array(dtype=wp.int32),
                   s_p0: wp.array(dtype=wp.vec3), s_q0: wp.array(dtype=wp.quat), s_h: wp.array(dtype=wp.vec3),
                   margin: float, below: float, released: wp.array(dtype=wp.int32), cut: wp.array(dtype=wp.int32),
                   fmax: wp.array(dtype=float), dfull: wp.array(dtype=float), s_body: wp.array(dtype=wp.int32),
                   debt: wp.array(dtype=float), kst: wp.array(dtype=float), iface: wp.array(dtype=wp.float64)):
    """A bond whose midpoint lies in a blade segment's slab on the spine side of the edge has been cut; its stored
    elastic energy vanishes with it (booked here) and the rest of its unspent cohesive energy is owed by the blade
    segment's body (as in k_body_tie_release)."""
    i = wp.tid()
    if i >= n or damage[i] >= 1.0:
        return
    m = 0.5 * (x[ba[i]] + x[bb[i]])
    for kk in range(n_blades):
        s = blade_ids[kk]
        X = wp.transform(s_p0[s], s_q0[s])
        pl = wp.transform_point(wp.transform_inverse(X), m)
        h = s_h[s]
        if wp.abs(pl[0]) <= h[0] and pl[2] >= -below and pl[2] <= 2.0 * h[2] and wp.abs(pl[1]) <= h[1] + margin:
            dd = x[bb[i]] - x[ba[i]]
            stored = 0.5 * (1.0 - damage[i]) * kst[i] * wp.dot(dd, dd)
            unspent = (1.0 - cohesive_spent(damage[i], fmax[i] / kst[i], dfull[i])) * 0.5 * fmax[i] * dfull[i]
            wp.atomic_add(debt, s_body[s], wp.max(unspent - stored, 0.0))
            wp.atomic_add(iface, 0, wp.float64(stored))
            damage[i] = 1.0
            cut[i] = 1
            wp.atomic_add(released, 0, 1)
            return


# ------------------------------------------------------------------------------------------ rods (sub-cycled)
@wp.kernel
def k_rods_subcycle(n_rods: int, r_start: wp.array(dtype=wp.int32), r_len: wp.array(dtype=wp.int32),
                    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
                    minv: wp.array(dtype=float), g: wp.vec3,
                    l0: wp.array(dtype=float), kax: wp.array(dtype=float), cax: wp.array(dtype=float),
                    lon: wp.array(dtype=wp.int32), kb: wp.array(dtype=float), brest: wp.array(dtype=wp.vec3),
                    bon: wp.array(dtype=wp.int32),
                    tie_tet: wp.array(dtype=wp.int32), tie_bary: wp.array(dtype=wp.vec4), tie_k: wp.array(dtype=float),
                    tie_fmax: wp.array(dtype=float), tie_dfull: wp.array(dtype=float), tie_dmg: wp.array(dtype=float),
                    tets: wp.array(dtype=wp.vec4i),
                    root_body: wp.array(dtype=wp.int32), root_loc: wp.array(dtype=wp.vec3), root_k: wp.array(dtype=float),
                    root_fmax: wp.array(dtype=float), root_on: wp.array(dtype=wp.int32),
                    rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_f: wp.array(dtype=wp.vec3),
                    rb_t: wp.array(dtype=wp.vec3),
                    fr: wp.array(dtype=wp.vec3), tie_sum: wp.array(dtype=wp.vec3), dt: float, nsub: int,
                    tension: wp.array(dtype=float), moment: wp.array(dtype=float),
                    tie_dissip: wp.array(dtype=wp.float64), visc: wp.array(dtype=wp.float64)):
    """One thread per rod (rib, pin bone, fin support): the rod's axial, bending and shaft-bond forces are integrated
    with nsub sub-steps inside one tissue substep, so thin bones keep their real mass and real stiffness (a 0.3 mm
    bone at 3 GPa needs ~1.5 us steps). The other forces on rod nodes (contact, edge, root bond) accumulated in f
    this substep are held over the sub-steps. The tissue side of each shaft bond is the tet's material point at the
    start of the substep; the bond's mean reaction over the sub-steps is added to the tet nodes (staggered
    coupling). Per-link array index = the link's first node; per-hinge index = its middle node."""
    r = wp.tid()
    if r >= n_rods:
        return
    s0 = r_start[r]
    n = r_len[r]
    dts = dt / float(nsub)
    e_visc = float(0.0)
    e_tie = float(0.0)
    froot_sum = wp.vec3(0.0, 0.0, 0.0)
    for j in range(n):
        tension[s0 + j] = 0.0
        moment[s0 + j] = 0.0
        tie_sum[s0 + j] = wp.vec3(0.0, 0.0, 0.0)
    for it in range(nsub):
        for j in range(n):
            fr[s0 + j] = f[s0 + j]
        # axial links
        for j in range(n - 1):
            a = s0 + j
            if lon[a] == 0:
                continue
            b = a + 1
            d = x[b] - x[a]
            ln = wp.length(d)
            if ln < 1.0e-9:
                continue
            u = d / ln
            tn = kax[a] * (ln - l0[a])
            fax = tn + cax[a] * wp.dot(v[b] - v[a], u)
            tension[a] = wp.max(tension[a], tn)
            fr[a] = fr[a] + fax * u
            fr[b] = fr[b] - fax * u
            e_visc = e_visc + cax[a] * wp.dot(v[b] - v[a], u) * wp.dot(v[b] - v[a], u) * dts
        # bending (small-angle discrete curvature, as k_rod_bend)
        for j in range(1, n - 1):
            m = s0 + j
            if bon[m] == 0 or lon[m - 1] == 0 or lon[m] == 0:
                continue
            e0 = x[m] - x[m - 1]
            e1 = x[m + 1] - x[m]
            la = wp.length(e0)
            lb = wp.length(e1)
            if la < 1.0e-9 or lb < 1.0e-9:
                continue
            t0 = e0 / la
            t1 = e1 / lb
            dk = wp.cross(t0, t1) - brest[m]
            moment[m] = wp.max(moment[m], kb[m] * wp.length(dk))
            g1 = kb[m] * dk
            fa = wp.cross(t1, g1) / la
            fc = wp.cross(t0, g1) / lb
            fa = fa - wp.dot(fa, t0) * t0
            fc = fc - wp.dot(fc, t1) * t1
            # F = -dE/dx for E = k/2 |t0 x t1 - k0|^2: F[m-1] = +P0 (t1 x g)/l0, F[m+1] = +P1 (t0 x g)/l1, F[m] = -(sum)
            # (the earlier pattern had the wrong sign on m-1 and was not conservative: a kicked rib gained energy)
            fr[m - 1] = fr[m - 1] + fa
            fr[m + 1] = fr[m + 1] + fc
            fr[m] = fr[m] - fa - fc
        # shaft bonds to the flesh (tet material point fixed over the substep)
        for j in range(n):
            p = s0 + j
            tt = tie_tet[p]
            if tt < 0 or tie_dmg[p] >= 1.0:
                continue
            tet = tets[tt]
            bb = tie_bary[p]
            xa = bb[0] * x[tet[0]] + bb[1] * x[tet[1]] + bb[2] * x[tet[2]] + bb[3] * x[tet[3]]
            d = x[p] - xa
            dl = wp.length(d)
            kk = tie_k[p]
            d0 = tie_fmax[p] / kk
            D = tie_dmg[p]
            if dl > d0:
                df = tie_dfull[p]
                Dn = wp.clamp(df * (dl - d0) / (dl * (df - d0)), 0.0, 1.0)
                if Dn > D:
                    e_tie = e_tie + 0.5 * tie_fmax[p] * df * (cohesive_spent(Dn, d0, df) - cohesive_spent(D, d0, df))
                    D = Dn
                    tie_dmg[p] = D
            ftie = (1.0 - D) * kk * d
            fr[p] = fr[p] - ftie
            tie_sum[p] = tie_sum[p] + ftie
        # root bond to the rod's vertebral segment (body pose fixed over the substep); breaks at bone strength
        p = s0
        rbi = root_body[r]
        if rbi >= 0 and root_on[r] != 0:
            rr = wp.quat_rotate(rb_q[rbi], root_loc[r])
            xb = rb_p[rbi] + rr
            dr = x[p] - xb
            froot = root_k[r] * dr
            if wp.length(froot) > root_fmax[r]:
                root_on[r] = 0
            else:
                fr[p] = fr[p] - froot
                froot_sum = froot_sum + froot
        for j in range(n):
            p = s0 + j
            vi = v[p] + (fr[p] * minv[p] + g) * dts
            v[p] = vi
            x[p] = x[p] + vi * dts
    # mean reactions over the sub-steps, applied once: shaft bonds on their tet nodes, root bond on its body
    inv = 1.0 / float(nsub)
    for j in range(n):
        p = s0 + j
        f[p] = wp.vec3(0.0, 0.0, 0.0)
        tt = tie_tet[p]
        if tt >= 0:
            tet = tets[tt]
            bb = tie_bary[p]
            ts_ = tie_sum[p] * inv
            wp.atomic_add(f, tet[0], bb[0] * ts_)
            wp.atomic_add(f, tet[1], bb[1] * ts_)
            wp.atomic_add(f, tet[2], bb[2] * ts_)
            wp.atomic_add(f, tet[3], bb[3] * ts_)
    rbi = root_body[r]
    if rbi >= 0:
        rr = wp.quat_rotate(rb_q[rbi], root_loc[r])
        fm = froot_sum * inv
        wp.atomic_add(rb_f, rbi, fm)
        wp.atomic_add(rb_t, rbi, wp.cross(rr, fm))
    wp.atomic_add(visc, 0, wp.float64(e_visc))
    wp.atomic_add(tie_dissip, 0, wp.float64(e_tie))


@wp.func
def polar_rotation(F: wp.mat33):
    """The rotation of F's polar decomposition (a proper rotation even for an inverted F)."""
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U * wp.transpose(V)
    if wp.determinant(R) < 0.0:
        U = wp.matrix_from_cols(wp.vec3(U[0, 0], U[1, 0], U[2, 0]), wp.vec3(U[0, 1], U[1, 1], U[2, 1]),
                                -wp.vec3(U[0, 2], U[1, 2], U[2, 2]))
        R = U * wp.transpose(V)
    return R


@wp.func
def triple_rotation(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3):
    """Rotation taking the frame of three rest points p onto the frame of the current points q (chord, in-plane
    normal, plane normal). With it the rest binormal turns onto the current plane's normal, so the hinge energy is
    kb (|kappa| - |kappa0|)^2 / 2 exactly and the bending forces carry no net moment. Straight triples (no plane):
    identity (their rest curvature is ~0)."""
    ca = p2 - p0
    na = wp.cross(p1 - p0, p2 - p0)
    cb = q2 - q0
    nb = wp.cross(q1 - q0, q2 - q0)
    la = wp.length(ca)
    lb = wp.length(cb)
    if wp.length(na) <= 1.0e-12 * la * la or wp.length(nb) <= 1.0e-12 * lb * lb or la <= 0.0 or lb <= 0.0:
        return wp.identity(n=3, dtype=float)
    ea = ca / la
    fa = wp.normalize(na)
    ga = wp.cross(fa, ea)
    eb = cb / lb
    fb = wp.normalize(nb)
    gb = wp.cross(fb, eb)
    Fa = wp.matrix_from_cols(ea, ga, fa)
    Fb = wp.matrix_from_cols(eb, gb, fb)
    return Fb * wp.transpose(Fa)


@wp.func
def dtrue(x: wp.array(dtype=wp.vec3), xc: wp.array(dtype=wp.vec3), a: int, b: int):
    """x[b] - x[a] of the TRUE positions: the float32 positions minus their compensated-summation residuals xc. At bench
    height a float32 coordinate resolves 6e-8 m; the difference of two nearby float32 positions is exact, so this
    difference is exact to the residuals' precision instead of carrying each position's rounding."""
    return (x[b] - x[a]) - (xc[b] - xc[a])


@wp.kernel
def k_rods_pbd(n_rods: int, r_start: wp.array(dtype=wp.int32), r_len: wp.array(dtype=wp.int32),
               x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3),
               minv: wp.array(dtype=float), g: wp.vec3, gfrac: wp.array(dtype=float),
               l0: wp.array(dtype=float), lon: wp.array(dtype=wp.int32), kb: wp.array(dtype=float),
               brest: wp.array(dtype=wp.vec3), bon: wp.array(dtype=wp.int32),
               tie_tet: wp.array(dtype=wp.int32), tie_bary: wp.array(dtype=wp.vec4), tie_k: wp.array(dtype=float),
               tie_fmax: wp.array(dtype=float), tie_dfull: wp.array(dtype=float), tie_dmg: wp.array(dtype=float),
               tie_c: wp.array(dtype=float), tets: wp.array(dtype=wp.vec4i),
               root_body: wp.array(dtype=wp.int32), root_loc: wp.array(dtype=wp.vec3), root_fmax: wp.array(dtype=float),
               root_on: wp.array(dtype=wp.int32),
               rb_p: wp.array(dtype=wp.vec3), rb_q: wp.array(dtype=wp.quat), rb_f: wp.array(dtype=wp.vec3),
               rb_t: wp.array(dtype=wp.vec3), xold: wp.array(dtype=wp.vec3), dt: float, iters: int,
               tension: wp.array(dtype=float), moment: wp.array(dtype=float),
               tie_dissip: wp.array(dtype=wp.float64), rdx: wp.array(dtype=wp.vec3), xc: wp.array(dtype=wp.vec3),
               proj_dissip: wp.array(dtype=wp.float64), Xr: wp.array(dtype=wp.vec3), Dm_inv: wp.array(dtype=wp.mat33),
               fint: wp.array(dtype=wp.vec3), rb_pc: wp.array(dtype=wp.vec3)):
    """Thin bones as position-based rods, one thread per rod, one pass per tissue substep:
      1. explicit forces: bending (conservative discrete curvature), shaft bonds to the flesh (bilinear cohesive;
         reaction on the tet nodes), the contact/edge forces already accumulated in f, gravity -> predicted positions;
      2. inextensibility: Gauss-Seidel projection of every link to its rest length and of the root node onto its
         vertebral segment's attachment point (the segment is heavy against a ~1 mg node: treated as fixed during
         the projection, the implied reaction force is applied to it);
      3. velocities from the position change, which is carried as its own displacement (rdx) through the
         projection: differencing float32 positions at bench height quantises a rod node's velocity to ~6 mm/s steps
         and makes a free rib's momentum drift 28 % in 5 ms; the new position is
         the old one plus that displacement, with the rounding carried (xc) as for tissue nodes. The energy the
         (inelastic) projection removes from the rod's free nodes is booked to the ledger (damp_rods).
    Bone is effectively inextensible at these loads (a 0.3 mm rib stretches ~25 um per 3 mm link under 5 N); its
    axial stiffness would need ~1.5 us explicit steps. The implied constraint force (m dx/dt^2) is recorded as the
    link tension for the breaking check; a root whose implied force exceeds its strength detaches.
    Every geometric quantity is evaluated on TRUE relative positions (dtrue), and the projection iterates on the
    displacement rdx with the positions held. Re-evaluating each link on the float32 positions at bench height would
    correct their rounding (one spacing, 6e-8 m) every substep as if it were stretch, and that correction becomes
    velocity (6e-8 m / 1e-5 s = 6 mm/s): the resting trout's rod nodes jitter at 4.5 mm/s rms and the world dissipates
    ~1.4-2 mW with no work in."""
    r = wp.tid()
    if r >= n_rods:
        return
    s0 = r_start[r]
    n = r_len[r]
    e_tie = float(0.0)
    u_before = float(0.0)                          # the rod's bending and shaft-tie energy before the step
    for j in range(n):
        fint[s0 + j] = wp.vec3(0.0, 0.0, 0.0)
    # 1. explicit forces into f (bending, bonds), then predict
    for j in range(1, n - 1):
        m = s0 + j
        moment[m] = 0.0
        if bon[m] == 0 or lon[m - 1] == 0 or lon[m] == 0:
            continue
        e0 = dtrue(x, xc, m - 1, m)
        e1 = dtrue(x, xc, m, m + 1)
        la = wp.length(e0)
        lb = wp.length(e1)
        if la < 1.0e-9 or lb < 1.0e-9:
            continue
        t0 = e0 / la
        t1 = e1 / lb
        # the rest curvature turns with the bone: with the flesh the hinge is tied into (the tet's rotation, polar
        # decomposition of its deformation gradient), or, for a hinge no longer tied, with its own three nodes (the
        # best-fit rotation of their rest positions onto their current ones). With a fixed world vector a rib rotated
        # rigidly would bend itself: 30 % of its breaking moment at 180 deg
        tt = tie_tet[m]
        tied = tt >= 0 and tie_dmg[m] < 1.0
        Rr = wp.identity(n=3, dtype=float)
        if tied:
            tq = tets[tt]
            Ds = wp.matrix_from_cols(dtrue(x, xc, tq[0], tq[1]), dtrue(x, xc, tq[0], tq[2]), dtrue(x, xc, tq[0], tq[3]))
            Rr = polar_rotation(Ds * Dm_inv[tt])
        else:
            Rr = triple_rotation(Xr[m - 1], Xr[m], Xr[m + 1], wp.vec3(0.0, 0.0, 0.0), e0, e0 + e1)
        dk = wp.cross(t0, t1) - Rr * brest[m]
        moment[m] = kb[m] * wp.length(dk)
        gk = kb[m] * dk
        fa = wp.cross(t1, gk) / la
        fc = wp.cross(t0, gk) / lb
        fa = fa - wp.dot(fa, t0) * t0
        fc = fc - wp.dot(fc, t1) * t1
        f[m - 1] = f[m - 1] + fa
        f[m + 1] = f[m + 1] + fc
        f[m] = f[m] - fa - fc
        fint[m - 1] = fint[m - 1] + fa
        fint[m + 1] = fint[m + 1] + fc
        fint[m] = fint[m] - fa - fc
        u_before = u_before + 0.5 * kb[m] * wp.dot(dk, dk)
        if tied:
            # the bending forces' net moment (sum k kappa x R kappa0) is the flesh orienting the bone: its reaction
            # goes to the tet's nodes as a pure couple (f_i = w x r_i with A w = -T, A the nodes' second moment)
            T = wp.cross(-e0, fa) + wp.cross(e1, fc)
            cq = 0.25 * (x[tq[0]] + x[tq[1]] + x[tq[2]] + x[tq[3]])
            A = wp.mat33(0.0)
            for k in range(4):
                rk = x[tq[k]] - cq
                A = A + wp.dot(rk, rk) * wp.identity(n=3, dtype=float) - wp.outer(rk, rk)
            if wp.determinant(A) > 1.0e-30:
                wv = wp.inverse(A) * (-T)
                for k in range(4):
                    wp.atomic_add(f, tq[k], wp.cross(wv, x[tq[k]] - cq))
    for j in range(n):
        p = s0 + j
        tt = tie_tet[p]
        if tt >= 0:
            tet = tets[tt]
            bb = tie_bary[p]
            d = (bb[0] * dtrue(x, xc, tet[0], p) + bb[1] * dtrue(x, xc, tet[1], p) + bb[2] * dtrue(x, xc, tet[2], p)
                 + bb[3] * dtrue(x, xc, tet[3], p))
            dl = wp.length(d)
            kk = tie_k[p]
            D = tie_dmg[p]
            if D < 1.0:
                d0 = tie_fmax[p] / kk
                if dl > d0:
                    df = tie_dfull[p]
                    Dn = wp.clamp(df * (dl - d0) / (dl * (df - d0)), 0.0, 1.0)
                    if Dn > D:
                        e_tie = e_tie + 0.5 * tie_fmax[p] * df * (cohesive_spent(Dn, d0, df) - cohesive_spent(D, d0, df))
                        D = Dn
                        tie_dmg[p] = D
            va = bb[0] * v[tet[0]] + bb[1] * v[tet[1]] + bb[2] * v[tet[2]] + bb[3] * v[tet[3]]
            dvt = v[p] - va
            # the flesh's viscous drag on the embedded bone: a dashpot on the relative velocity at the bond (without
            # it the bones kept an undamped ~6 kHz whirl after the settling transient: development probe). The drag
            # is the flesh around the bone, not the bond: it stays when the bond has failed, while the bone is still
            # within about one element (the tie tet's size) of its material point. If it went with the bond, a pin
            # bone whose bonds had failed next to its root would swing its first free node around the root at up to
            # 10.7 m/s inside the flesh (the A1 control)
            dm_det = wp.abs(wp.determinant(Dm_inv[tt]))
            c_drag = float(0.0)
            if dm_det > 0.0:
                if dl < wp.pow(1.0 / dm_det, 1.0 / 3.0):
                    c_drag = tie_c[p]
            ftie = (1.0 - D) * kk * d + c_drag * dvt
            e_tie = e_tie + c_drag * wp.dot(dvt, dvt) * dt
            f[p] = f[p] - ftie
            fint[p] = fint[p] - (1.0 - D) * kk * d
            u_before = u_before + 0.5 * (1.0 - D) * kk * dl * dl
            wp.atomic_add(f, tet[0], bb[0] * ftie)
            wp.atomic_add(f, tet[1], bb[1] * ftie)
            wp.atomic_add(f, tet[2], bb[2] * ftie)
            wp.atomic_add(f, tet[3], bb[3] * ftie)
    ke_before = float(0.0)
    for j in range(n):
        p = s0 + j
        xold[p] = x[p]
        if minv[p] > 0.0:
            ke_before = ke_before + 0.5 * wp.dot(v[p], v[p]) / minv[p]
        vi = v[p] + (f[p] * minv[p] + g * gfrac[p]) * dt          # weight of the physical mass only
        rdx[p] = vi * dt                     # the positions stay put: the projection iterates on the displacement
    # 2. projection
    rbi = root_body[r]
    tgt = wp.vec3(0.0, 0.0, 0.0)                   # the root's displacement that puts it on its segment's point
    rr = wp.vec3(0.0, 0.0, 0.0)
    if rbi >= 0 and root_on[r] != 0:
        rr = wp.quat_rotate(rb_q[rbi], root_loc[r])
        tgt = ((rb_p[rbi] - x[s0]) - rb_pc[rbi] + xc[s0]) + rr
    droot = wp.vec3(0.0, 0.0, 0.0)
    f_seg = wp.vec3(0.0, 0.0, 0.0)                 # the segment's force on a held root (its reaction is fb below)
    lam0 = float(0.0)
    for j in range(n):
        tension[s0 + j] = 0.0
    for it in range(iters):
        if rbi >= 0 and root_on[r] != 0:
            dx = tgt - rdx[s0]
            rdx[s0] = rdx[s0] + dx
            droot = droot + dx
        for j in range(n - 1):
            a = s0 + j
            if lon[a] == 0:
                continue
            b = a + 1
            d = dtrue(x, xc, a, b) + (rdx[b] - rdx[a])
            ln = wp.length(d)
            if ln < 1.0e-12:
                continue
            u = d / ln
            wa = minv[a]
            wb = minv[b]
            if j == 0 and rbi >= 0 and root_on[r] != 0:
                wa = 0.0                      # the root is held on its segment during the projection
            w = wa + wb
            if w <= 0.0:
                continue
            c = ln - l0[a]
            lam = c / w
            rdx[a] = rdx[a] + wa * lam * u
            rdx[b] = rdx[b] - wb * lam * u
            tension[a] = tension[a] + lam / (dt * dt)
            if j == 0 and wa == 0.0:
                lam0 = lam0 + lam
    # root reaction: the force that kept the root on the segment this substep; detaches over its strength
    if rbi >= 0 and root_on[r] != 0:
        # force the segment exerts on the held root: m (x_final - x_pred)/dt^2 minus the first link's pull on it; the
        # segment receives the opposite: F_link - m dx/dt^2 (without the link term the rib's tension never reached its
        # vertebra, momentum leaked and the trout gained energy: development probe)
        fr_ = droot / (minv[s0] * dt * dt)
        fl = wp.vec3(0.0, 0.0, 0.0)
        if n > 1:
            d1 = dtrue(x, xc, s0, s0 + 1) + (rdx[s0 + 1] - rdx[s0])
            l1 = wp.length(d1)
            if l1 > 1.0e-12:
                fl = (lam0 / (dt * dt)) * (d1 / l1)
        fb = fl - fr_
        f_seg = -fb
        if wp.length(fb) > root_fmax[r]:
            root_on[r] = 0
        else:
            wp.atomic_add(rb_f, rbi, fb)
            wp.atomic_add(rb_t, rbi, wp.cross(rr, fb))
    # 3. velocities from the carried displacement; positions = old + displacement (compensated). The projection's
    # loss is what the explicit forces (constant over the substep) did along the ACTUAL displacement, less the
    # kinetic energy gained: KE_before + sum F . dx - KE_after. Booking the kinetic energy the projection removes
    # from the predicted velocities (KE_pred - KE_after) would count a static load's velocity increment, cancelled
    # within the same substep, as dissipated: F^2 dt / 2m per node, 12.9 W over the pin bones loaded by their shaft
    # ties beside the A1 control's cut, with nothing moving. A held root is one of
    # the rod's nodes like the others, and the segment's force on it works on the rod too: without it the energy a
    # moving vertebra hands to its rib or pin was booked as negative dissipation, while the segment's own kinetic
    # energy loss was already counted (-2.65 mJ in 0.1 s beside the same cut)
    ke_after = float(0.0)
    work = wp.dot(f_seg, rdx[s0])
    for j in range(n):
        p = s0 + j
        vn = rdx[p] / dt
        v[p] = vn
        y = rdx[p] - xc[p]
        xn = xold[p] + y
        xc[p] = (xn - xold[p]) - y
        x[p] = xn
        if minv[p] > 0.0:
            ke_after = ke_after + 0.5 * wp.dot(vn, vn) / minv[p]
            work = work + wp.dot(f[p] - fint[p] + g * (gfrac[p] / minv[p]), rdx[p])
        f[p] = wp.vec3(0.0, 0.0, 0.0)
    # the rod's own conservative forces (bending, the shaft ties' springs) enter as the change of their energy over the
    # step, not as the work their start-of-step values do: for a stiff force that work exceeds the energy change by
    # dx.H.dx/2 every step, and the projection would then book the excess as dissipation (the resting trout's rods:
    # 0.7 mW with nothing moving but micrometre jitter; a rib flailing on a driven root: 0.8 mW against 0.03 uW of
    # input). The flesh does not move during this kernel, so the energies before and after use the same anchors.
    u_after = float(0.0)
    for j in range(1, n - 1):
        m = s0 + j
        if bon[m] == 0 or lon[m - 1] == 0 or lon[m] == 0:
            continue
        e0 = dtrue(x, xc, m - 1, m)
        e1 = dtrue(x, xc, m, m + 1)
        la = wp.length(e0)
        lb = wp.length(e1)
        if la < 1.0e-9 or lb < 1.0e-9:
            continue
        tt = tie_tet[m]
        Rr = wp.identity(n=3, dtype=float)
        if tt >= 0 and tie_dmg[m] < 1.0:
            tq = tets[tt]
            Ds = wp.matrix_from_cols(dtrue(x, xc, tq[0], tq[1]), dtrue(x, xc, tq[0], tq[2]), dtrue(x, xc, tq[0], tq[3]))
            Rr = polar_rotation(Ds * Dm_inv[tt])
        else:
            Rr = triple_rotation(Xr[m - 1], Xr[m], Xr[m + 1], wp.vec3(0.0, 0.0, 0.0), e0, e0 + e1)
        dk = wp.cross(e0 / la, e1 / lb) - Rr * brest[m]
        u_after = u_after + 0.5 * kb[m] * wp.dot(dk, dk)
    for j in range(n):
        p = s0 + j
        tt = tie_tet[p]
        if tt >= 0 and tie_dmg[p] < 1.0:
            tet = tets[tt]
            bb = tie_bary[p]
            d = (bb[0] * dtrue(x, xc, tet[0], p) + bb[1] * dtrue(x, xc, tet[1], p) + bb[2] * dtrue(x, xc, tet[2], p)
                 + bb[3] * dtrue(x, xc, tet[3], p))
            u_after = u_after + 0.5 * (1.0 - tie_dmg[p]) * tie_k[p] * wp.dot(d, d)
    wp.atomic_add(proj_dissip, 0, wp.float64(ke_before + u_before + work - ke_after - u_after))
    wp.atomic_add(tie_dissip, 0, wp.float64(e_tie))

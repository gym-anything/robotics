"""Vibro-contact between strikers and vibrating surfaces.

Law (Hertz / Hunt-Crossley):  F = K delta^alpha (1 + lambda d(delta)/dt)  for delta > 0, else 0, where the penetration
delta = -(rigid gap) + u_surface + u_striker includes the elastic normal displacement of the struck structure (and of a
drumstick's own bending modes).

Implicit solve per audio sample: the force is held constant over the sample; the end-of-sample penetration is linear
in it,  delta_end(F) = delta_end_free - S F,  with S = the structure's exact one-sample modal response at the contact
(modal.Structure.compliance) + the striker's and the struck body's rigid response dt^2 (1/m_s + 1/m_t) (MuJoCo's
semi-implicit Euler: x' = x + dt (v + dt F/m)) + the quasi-static compliance of truncated modes. The force is the
discrete gradient of the contact potential Phi = K delta^(alpha+1)/(alpha+1) between the start and end penetrations,
which makes the elastic contact energy-consistent (bench E5), plus the Hunt-Crossley dissipation at the mean
penetration. g(F) = F - law(delta_end(F)) is increasing, so a safeguarded Newton iteration finds the unique root.
"""
import math
import numpy as np


def _phi(K, a, d):
    return K*d**(a + 1)/(a + 1) if d > 0 else 0.0


def contact_force(d0, d1_free, S, K, a, lam, dt, F_guess=0.0):
    """Constant force over one sample: discrete gradient of the Hertz potential between d0 (start) and d1 = d1_free - S F
    (end), plus Hunt-Crossley damping K lam dbar^a (d1 - d0)/dt. Returns (F, d1).

    Safeguarded Newton on g(F) = F - law(d1(F)) with the analytic derivative, started from F_guess (the pair's previous
    force: a resting or slowly varying contact converges in 1-2 steps) or the force-free law value; it brackets the root
    and bisects whenever a Newton step leaves the bracket (a numerical-derivative iteration from the force-free value
    costs ~10 law evaluations per pair and sample - 40 % of a window tick with 12 resting pairs)."""
    if d1_free <= 0 and d0 <= 0:
        return 0.0, d1_free
    ap1 = a + 1.0
    p0 = K*d0**ap1/ap1 if d0 > 0 else 0.0
    f00 = K*d0**a if d0 > 0 else 0.0
    Kl = K*lam/dt if lam else 0.0

    def law(F):
        """(law value, d law / d F, d1)"""
        d1 = d1_free - S*F
        if d1 <= 0 and d0 <= 0:
            return 0.0, 0.0, d1
        dd = d1 - d0
        if abs(dd) > 1e-9*max(abs(d0), abs(d1), 1e-12):
            p1 = K*d1**ap1/ap1 if d1 > 0 else 0.0
            fe = (p1 - p0)/dd
            de = ((K*d1**a if d1 > 0 else 0.0) - fe)/dd           # d fe / d d1
        else:
            fe = f00
            de = 0.5*a*K*d0**(a - 1) if d0 > 0 else 0.0          # limit of the secant slope
        if Kl:
            dm = 0.5*(d0 + d1)
            if dm > 0:
                dma = dm**a
                fe += Kl*dma*dd
                de += Kl*(dma + 0.5*a*dm**(a - 1)*dd)
        if fe <= 0.0:
            return 0.0, 0.0, d1
        return fe, -S*de, d1

    f0, _, _ = law(0.0)
    if f0 <= 0:
        return 0.0, d1_free
    lo, hi = 0.0, None
    F = F_guess if 0.0 < F_guess else f0
    for _ in range(80):
        val, dval, d1 = law(F)
        g = F - val
        if g > 0:
            hi = F if hi is None else min(hi, F)
        else:
            lo = max(lo, F)
        if abs(g) <= 1e-10*max(1.0, F):
            return F, d1
        dg = 1.0 - dval
        Fn = F - g/dg if dg > 1e-12 else -1.0
        if not (lo <= Fn <= (hi if hi is not None else Fn + 1.0)) or Fn < 0:
            Fn = 0.5*(lo + hi) if hi is not None else 2.0*F + 1e-6
        F = Fn
    return F, law(F)[2]


def contact_force_reference(d0, d1_free, S, K, a, lam, dt):
    """The reference solver (numerical derivative, cold start): kept for the equivalence test of contact_force."""
    if d1_free <= 0 and d0 <= 0:
        return 0.0, d1_free

    def law(F):
        d1 = d1_free - S*F
        if d1 <= 0 and d0 <= 0:
            return 0.0, d1
        dd = d1 - d0
        if abs(dd) > 1e-14*max(abs(d0), abs(d1), 1e-12):
            fe = (_phi(K, a, d1) - _phi(K, a, d0))/dd
        else:
            fe = K*max(d0, 0.0)**a
        if lam:
            dm = max(0.5*(d0 + d1), 0.0)
            fe += K*lam*dm**a*dd/dt
        return max(fe, 0.0), d1

    # bracket: F = 0 gives g(0) = -law(d1_free) <= 0; F_hi with d1(F_hi) = min(d0, 0)-ish gives g >= 0
    lo, hi = 0.0, None
    f0, _ = law(0.0)
    if f0 <= 0:
        return 0.0, d1_free
    F = f0
    for _ in range(80):
        val, d1 = law(F)
        g = F - val
        if g > 0:
            hi = F if hi is None else min(hi, F)
        else:
            lo = max(lo, F)
        if abs(g) <= 1e-10*max(1.0, F):
            return F, d1
        # numerical derivative of the law (cheap: scalar)
        h = 1e-7*max(F, 1e-6)
        v2, _ = law(F + h)
        dg = 1.0 - (v2 - val)/h
        Fn = F - g/dg if dg > 1e-12 else None
        if Fn is None or not (lo <= Fn <= (hi if hi is not None else Fn + 1)) or Fn < 0:
            Fn = 0.5*(lo + hi) if hi is not None else 2*F + 1e-6
        F = Fn
    return F, law(F)[1]


# ====================================================================================================== geometry
def patch_radius(striker_material, surface_material, R, tension=None, scale=1.0):
    """Radius (m) of the loaded area of a contact, over which the struck structure's mode shapes are averaged and at
    which the membrane's residual point compliance is cut off (params.PATCH_REF_FORCE_N comment). R: the striker's
    radius at the contact; tension: the head's membrane tension (N/m) for 'mylar'; scale: refinement-study factor."""
    from . import params as P
    F = P.PATCH_REF_FORCE_N[striker_material]
    K, a, _, _ = P.CONTACT[striker_material][surface_material]
    a_ind = math.sqrt(R*(F/K)**(1.0/a))                    # the striker's own indentation radius sqrt(R delta(F))
    if surface_material == "mylar":
        T = float(tension)
        a_wrap = math.sqrt(F*R/(2*math.pi*T))               # film wrapping the striker up to the slope match
        D = P.DRUM_E_FILM*P.DRUM_FILM_M**3/(12*(1 - P.DRUM_NU_FILM**2))
        l_b = math.sqrt(D/T)                                # bending length of the film
        r = math.sqrt(max(a_wrap, a_ind)**2 + l_b**2)
    else:
        r = max(a_ind, 1e-3)
    return scale*r


def sphere_box(c, r, half):
    """Sphere (centre c, radius r) vs axis-aligned box [-half, half] (both in the box frame).
    Returns (gap, point on box surface, outward normal)."""
    c = np.asarray(c, float)
    q = np.clip(c, -half, half)
    d = c - q
    dist = float(np.linalg.norm(d))
    if dist > 1e-12:
        return dist - r, q, d/dist
    # centre inside: the nearest face
    pen = half - np.abs(c)
    k = int(np.argmin(pen))
    n = np.zeros(3)
    n[k] = 1.0 if c[k] >= 0 else -1.0
    q = c.copy()
    q[k] = n[k]*half[k]
    return -float(pen[k]) - r, q, n


def segment_box(p0, p1, r, half, iters=24):
    """Capsule (segment p0-p1, radius r) vs box: golden-section search on the convex distance along the segment."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)

    def dist(t):
        c = p0 + t*(p1 - p0)
        return float(np.linalg.norm(c - np.clip(c, -half, half)))
    a, b = 0.0, 1.0
    g = (math.sqrt(5) - 1)/2
    x1, x2 = b - g*(b - a), a + g*(b - a)
    f1, f2 = dist(x1), dist(x2)
    for _ in range(iters):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g*(b - a)
            f1 = dist(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g*(b - a)
            f2 = dist(x2)
    t = 0.5*(a + b)
    for tt in (0.0, 1.0):
        if dist(tt) < dist(t):
            t = tt
    gap, q, n = sphere_box(p0 + t*(p1 - p0), r, half)
    return gap, q, n, t


def sphere_disc(c, r, radius):
    """Sphere vs the head disc z = 0, |(x, y)| < radius (drum frame). None when the centre is outside the head."""
    rr = math.hypot(c[0], c[1])
    if rr >= radius:
        return None
    return float(c[2] - r), np.array([c[0], c[1], 0.0]), np.array([0.0, 0.0, 1.0])


def segment_disc(p0, p1, r, radius):
    """Capsule vs head disc: the segment's lowest point (or both ends when it lies within 1 mm of level)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    out = []
    if abs(p0[2] - p1[2]) < 1e-3:
        for p in (p0, p1):
            h = sphere_disc(p, r, radius)
            if h is not None:
                out.append(h)
        return out
    p = p0 if p0[2] < p1[2] else p1
    h = sphere_disc(p, r, radius)
    return [h] if h is not None else []


# ====================================================================================================== bench striker
class FreeStriker:
    """A point striker of mass m integrated like MuJoCo's semi-implicit Euler (bench probes only). The rigid solver is
    given G_k = (F_k + F_{k-1})/2: its positions are then exactly the trapezoidal x_{k+1} = x_k + dt v_k + dt^2 F_k/2m
    of a force held constant over the sample, and its stored velocity the half-kicked v_k + dt F_k/2m (world.py does
    the same with MuJoCo)."""

    def __init__(self, m, y0, v0):
        self.m, self.y, self.v = m, y0, v0
        self.F_prev = 0.0

    def free_end(self, dt):
        return self.y + dt*self.v + dt**2*self.F_prev/(2*self.m)

    def step(self, F, dt):
        G = 0.5*(F + self.F_prev)
        self.v += dt*G/self.m
        self.y += dt*self.v
        self.F_prev = F

    def true_velocity(self, dt):
        return self.v + dt*self.F_prev/(2*self.m)


def bench_impact(struct, b, m, v0, law, *, dt=None, t_end=0.02, c_res=0.0, stick=None, b_stick=None, gap0=2e-5,
                 m_target=None, record=False):
    """Striker (mass m, approaching the surface from gap0 at v0 > 0) on a modal structure at shape vector b, with the
    world's implicit per-sample solve. Returns contact time, peak force, rebound, energy balance and mode energies."""
    K, a, lam = law
    dt = dt or struct.dt
    c, w = struct.out_row(b), struct.in_col(b)
    S = struct.compliance(c, w, half=False) + dt**2/(2*m) + c_res + (dt**2/(2*m_target) if m_target else 0.0)
    if stick is not None:
        cs, ws = stick.out_row(b_stick), stick.in_col(b_stick)
        S += stick.compliance(cs, ws, half=False)
    sk = FreeStriker(m, gap0, -v0)
    struct.z[:] = 0
    if stick is not None:
        stick.z[:] = 0
    n = int(t_end/dt)
    F_hist = np.zeros(n)
    d0 = -gap0
    E0 = 0.5*m*v0**2
    for i in range(n):
        u_free = 2*np.real(np.dot(c, struct.E*struct.z))
        us_free = 2*np.real(np.dot(cs, stick.E*stick.z)) if stick is not None else 0.0
        y_free = sk.free_end(dt)
        d1_free = (u_free + us_free) - y_free          # the residual compliance acts on the new force only (in S)
        F, d1 = contact_force(d0, d1_free, S, K, a, lam, dt)
        struct.step(w*(-F) if F else None)
        if stick is not None:
            stick.step(ws*(-F) if F else None)
        sk.step(F, dt)
        u = 2*np.real(np.dot(c, struct.z)) - c_res*F
        us = 2*np.real(np.dot(cs, stick.z)) if stick is not None else 0.0
        d0 = (u + us) - sk.y
        F_hist[i] = F
    idx = np.flatnonzero(F_hist > 0)
    if not len(idx):
        return dict(contact=False)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    E_struct = struct.energy()
    E_stick = stick.energy() if stick is not None else 0.0
    v = sk.true_velocity(dt)
    out = dict(contact=True, first_contact_ms=1e3*len(runs[0])*dt, total_contact_ms=1e3*(idx[-1] - idx[0] + 1)*dt,
               n_contacts=len(runs), recontacts_ms=[round(1e3*(r[0] - idx[0])*dt, 3) for r in runs[1:]],
               peak_force_N=float(F_hist.max()), rebound_m_s=float(v), rebound_ratio=float(v/v0),
               energy_in_J=E0, energy_striker_out_J=0.5*m*v**2, energy_struct_J=E_struct, energy_stick_J=E_stick,
               energy_to_struct=E_struct/E0,
               energy_balance=(E0 - 0.5*m*v**2 - E_struct - E_stick)/E0)
    if record:
        out["force"] = F_hist
        out["dof_energy"] = struct.dof_energy()
    return out

"""Shared geometry for the anatomical workpiece generators.

Everything here is geometry and fields only; no solver import. Units SI (m, kg).

Contents
- `Workpiece`: the dataclass every generator returns.
- smooth booleans (`smin`, `smax`) so composed signed distances stay C1 (the mesher's secant refinement and
  any gradient-based projection behave; no creases unless anatomy has one).
- `Profile`: monotone cubic (PCHIP) profiles tabulated for fast vectorised lookup.
- `Loft`: a body lofted from cross sections (outline top/bottom, max half-width, egg asymmetry, superellipse
  exponent) with a first-order signed distance F/|grad F| (exact zero set, near-surface accurate) and an
  exact surface parametrisation (s, theta) for rendering and UVs.
- `SegmentSet`: many tapered capsules (rods, spines) with fast nearest-distance queries (KD-tree on dense
  axis samples), used for bone clearances and the "rods inside tissue" checks.
- capsule mass properties, grid integration of region volumes, and texture helpers (value noise, spot
  scattering, PNG writing) used by the appearance fields.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree


# ------------------------------------------------------------------------------------------ provenance
def prov(value, unit, label, basis):
    """One provenance record (same fields as filleting.params.P) for dimensions defined in a generator."""
    return dict(value=value, unit=unit, label=label, basis=basis)


# ------------------------------------------------------------------------------------------ Workpiece
@dataclass
class Workpiece:
    """What a generator returns. Callables take (N,3) points in the workpiece frame (m)."""
    name: str
    sdf: object                         # p -> signed distance to the tissue volume (negative inside)
    bounds: tuple                       # (lo, hi) meshing box
    region: object                      # p -> int region id (tissue points)
    region_names: list
    sheet_normal: object                # p -> (N,3) unit myoseptal normal (fish) / fibre-normal (pork)
    fibre: object                       # p -> (N,3) unit fibre direction
    rigid_bodies: list                  # dicts: name, mass, com, inertia, capsules, parent, joint, bond, ...
    rods: list                          # dicts: kind, points (k,3), radius (k,), root, name
    interfaces: dict                    # name -> dict(sdf, extent, params)
    skin: dict                          # skin_mask(p)->bool, thickness, params key, (+ other membranes)
    fins: list                          # thin membranes for rendering
    reference: object                   # p -> int reference-product id
    reference_names: list
    appearance: dict                    # skin_uv, skin_texture (path, provenance), flesh_color
    targets: dict
    measured: dict
    provenance: dict
    variation: dict
    extra: dict = field(default_factory=dict)   # generator-specific helpers (outer sdf, cavity, frames...)

    def to_local(self, p):
        """Workpiece frame -> generator's anatomical frame (identity unless the generator overrides)."""
        return self.extra["to_local"](p) if "to_local" in self.extra else p


# ------------------------------------------------------------------------------------------ booleans
def smin(a, b, k):
    """Polynomial smooth minimum (C1). k is the blend width in metres; k <= 0 gives the hard minimum."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(.5 + .5*(b - a)/k, 0., 1.)
    return b*(1. - h) + a*h - k*h*(1. - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0)/(e1 - e0), 0., 1.)
    return t*t*(3. - 2.*t)


# ------------------------------------------------------------------------------------------ profiles
class Profile:
    """Monotone cubic interpolant through control points, tabulated for vectorised lookup. Constant
    extrapolation outside the control range."""

    def __init__(self, xs, ys, n=4001):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        f = PchipInterpolator(xs, ys)
        self.x0, self.x1 = xs[0], xs[-1]
        self.t = np.linspace(xs[0], xs[-1], n)
        self.v = f(self.t)
        self.dv = f.derivative()(self.t)

    def __call__(self, x):
        return np.interp(x, self.t, self.v)

    def d(self, x):
        return np.interp(x, self.t, self.dv, left=0., right=0.)


def _half(e, nt, nb, eta):
    n = np.where(eta > 0, nt, nb)
    return (1 + e*eta)*np.clip(1 - np.abs(eta)**n, 0, None)**(1/n)


def _norm_factor(e, nt, nb):
    """max over eta in [-1,1] of the normalised half-width (1 + e*eta) (1 - |eta|^n)^(1/n), n = nt above the
    section centre and nb below: converts a max half-width to the superellipse lateral semi-axis."""
    eta = np.linspace(-1, 1, 4001)
    return np.max(_half(e, nt, nb, eta))


class Loft:
    """Body lofted from cross sections along a straight axis s in a local frame (s, v, w):
    v is the "vertical" of the section (dorsal for a fish, up for meat), w the lateral coordinate.
    At station s the section is the egg-superellipse
        (|eta|^n + |zeta|^n)^(1/n) = 1,  eta = (v - c)/a,  zeta = w/(b (1 + e eta)),
    with c, a from the top/bottom outline and b chosen so that the maximum half-width equals hw(s).
    The exponent may differ above (n_top) and below (n_bot) the section centre (continuous at eta = 0);
    e > 0 moves the widest point up, n > 2 fills the section toward a rounded box.
    """

    def __init__(self, s_ctrl, top, bot, hw, egg, power, s_range, cap_blend=1e-3, a_min=3e-4):
        self.top = Profile(s_ctrl, top)
        self.bot = Profile(s_ctrl, bot)
        self.hw = Profile(s_ctrl, hw)
        self.egg = Profile(s_ctrl, egg)
        if isinstance(power, tuple):
            self.pw_t, self.pw_b = Profile(s_ctrl, power[0]), Profile(s_ctrl, power[1])
        else:
            self.pw_t = self.pw_b = Profile(s_ctrl, power)
        self.s0, self.s1 = s_range
        self.k = cap_blend
        self.a_min = a_min
        tt = self.top.t
        e_t, nt_t, nb_t = self.egg(tt), self.pw_t(tt), self.pw_b(tt)
        cache = {}
        m = np.empty_like(tt)
        for i, key in enumerate(zip(np.round(e_t, 4), np.round(nt_t, 4), np.round(nb_t, 4))):
            if key not in cache:
                cache[key] = _norm_factor(*key)
            m[i] = cache[key]
        self._m = (tt, m)

    def section(self, s):
        c = .5*(self.top(s) + self.bot(s))
        a = np.maximum(.5*(self.top(s) - self.bot(s)), self.a_min)
        m = np.interp(s, *self._m)
        b = np.maximum(self.hw(s), self.a_min)/m
        return c, a, b, self.egg(s), self.pw_t(s), self.pw_b(s)

    def field(self, s, v, w):
        """Normalised radial implicit (0 on the surface, -1 on the section centre)."""
        c, a, b, e, nt, nb = self.section(s)
        eta = (v - c)/a
        n = np.where(eta > 0, nt, nb)
        zeta = w/(b*(1 + e*np.clip(eta, -1, 1)))
        return (np.abs(eta)**n + np.abs(zeta)**n)**(1/n) - 1.

    def sdf(self, s, v, w, eps=2e-5, capped=True):
        """First-order signed distance F/|grad F| (exact zero set; near-surface accurate)."""
        F = self.field(s, v, w)
        gs = (self.field(s + eps, v, w) - self.field(s - eps, v, w))/(2*eps)
        gv = (self.field(s, v + eps, w) - self.field(s, v - eps, w))/(2*eps)
        gw = (self.field(s, v, w + eps) - self.field(s, v, w - eps))/(2*eps)
        d = F/np.maximum(np.sqrt(gs*gs + gv*gv + gw*gw), 1e-9)
        if capped:
            d = smax(d, self.s0 - s, self.k)
            d = smax(d, s - self.s1, self.k)
        return d

    def surface(self, s, theta):
        """Exact surface point for station s and angle theta (0 = top/dorsal, pi/2 = +w flank, pi = bottom).
        Returns v, w."""
        c, a, b, e, nt, nb = self.section(s)
        ct, st = np.cos(theta), np.sin(theta)
        n = np.where(ct > 0, nt, nb)
        eta = np.sign(ct)*np.abs(ct)**(2/n)
        zeta = np.sign(st)*np.abs(st)**(2/n)
        return c + a*eta, b*(1 + e*eta)*zeta

    def half_width_at(self, s, v):
        """Surface half-width w >= 0 of the section at station s and height v (0 outside the section)."""
        c, a, b, e, nt, nb = self.section(s)
        eta = np.clip((v - c)/a, -1, 1)
        return b*_half(e, nt, nb, eta)

    def theta_of(self, s, v, w):
        """Inverse of `surface` for points on/near the surface (angle from the normalised coordinates)."""
        c, a, b, e, nt, nb = self.section(s)
        eta = np.clip((v - c)/a, -1.5, 1.5)
        n = np.where(eta > 0, nt, nb)
        zeta = w/(b*(1 + e*np.clip(eta, -1, 1)))
        r = np.maximum((np.abs(eta)**n + np.abs(zeta)**n)**(1/n), 1e-12)
        ce = np.sign(eta)*np.abs(eta/r)**(n/2)
        se = np.sign(zeta)*np.abs(zeta/r)**(n/2)
        return np.arctan2(se, ce) % (2*np.pi)

    def volume(self, s0, s1, n=2000):
        """Volume between stations (numerical area of each section)."""
        ss = np.linspace(s0, s1, n)
        th = np.linspace(0, 2*np.pi, 721)
        area = []
        for s in ss:
            v, w = self.surface(np.full_like(th, s), th)
            area.append(.5*abs(np.sum(v[:-1]*w[1:] - v[1:]*w[:-1])))
        return np.trapezoid(area, ss)


# ------------------------------------------------------------------------------------------ capsules
def capsule_sdf(p, p0, p1, r):
    """Distance from points p (N,3) to one capsule."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ab = p1 - p0
    t = np.clip(((p - p0) @ ab)/max(ab @ ab, 1e-18), 0., 1.)
    return np.linalg.norm(p - (p0 + t[:, None]*ab), axis=1) - r


class SegmentSet:
    """A set of tapered capsules (polyline rods, spines, centra). Distances are evaluated against dense axis
    samples (spacing `ds`) held in a KD-tree; the sampling error is <= ds^2/(8 d) (0.03 mm at ds 0.25 mm,
    d 0.3 mm). `owner` tags each capsule (e.g. rod index) so queries can report which bone is nearest."""

    def __init__(self, segs, ds=2.5e-4):
        # segs: list of (p0, p1, r0, r1, owner)
        pts, rad, own = [], [], []
        self.labels = []
        lab_id = {}
        for p0, p1, r0, r1, o in segs:
            if o not in lab_id:
                lab_id[o] = len(self.labels)
                self.labels.append(o)
            o = lab_id[o]
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            L = np.linalg.norm(p1 - p0)
            k = max(int(np.ceil(L/ds)), 1)
            t = np.linspace(0, 1, k + 1)
            pts.append(p0 + t[:, None]*(p1 - p0))
            rad.append(r0 + t*(r1 - r0))
            own.append(np.full(k + 1, o))
        self.pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        self.rad = np.concatenate(rad) if rad else np.zeros(0)
        self.own = np.concatenate(own) if own else np.zeros(0, int)
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        self.rmax = float(self.rad.max()) if len(self.rad) else 0.

    def distance(self, p, k=4, upper=np.inf):
        """Signed distance to the union of capsules (negative inside) and the owner id of the nearest one
        (index into self.labels; -1 when nothing lies within `upper`)."""
        p = np.atleast_2d(p)
        if self.tree is None:
            return np.full(len(p), np.inf), np.full(len(p), -1)
        k = min(k, len(self.pts))
        dd, ii = self.tree.query(p, k=k, distance_upper_bound=upper + self.rmax)
        if k == 1:
            dd, ii = dd[:, None], ii[:, None]
        valid = np.isfinite(dd)
        ii_c = np.where(valid, ii, 0)
        sd = np.where(valid, dd - self.rad[ii_c], np.inf)
        j = np.argmin(sd, 1)
        r = np.arange(len(p))
        return sd[r, j], np.where(valid[r, j], self.own[ii_c[r, j]], -1)


def capsule_mass_props(p0, p1, r, rho):
    """Mass, centre and inertia (3x3 about the centre) of a solid capsule."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    ax = (p1 - p0)/L if L > 0 else np.array([1., 0, 0])
    mc = rho*np.pi*r*r*L
    ms = rho*4/3*np.pi*r**3
    m = mc + ms
    i_ax = mc*r*r/2 + ms*2*r*r/5
    i_pp = mc*(L*L/12 + r*r/4) + ms*(2*r*r/5 + L*L/4 + 3*L*r/8)
    I = i_pp*np.eye(3) + (i_ax - i_pp)*np.outer(ax, ax)
    return m, .5*(p0 + p1), I


def combine_mass_props(parts):
    """Sum (m, com, I_about_com) parts with the parallel-axis theorem."""
    M = sum(p[0] for p in parts)
    com = sum(p[0]*p[1] for p in parts)/M
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = c - com
        I += Ic + m*((d @ d)*np.eye(3) - np.outer(d, d))
    return M, com, I


def rod_volume(points, radius):
    """Volume of a tapered polyline rod (frustum sum)."""
    P, r = np.asarray(points), np.asarray(radius)
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return float(np.sum(np.pi*L/3*(r[:-1]**2 + r[:-1]*r[1:] + r[1:]**2)))


# ------------------------------------------------------------------------------------------ integration
def grid_integrate(inside, labeler, lo, hi, h, chunk=400_000):
    """Volume per label of {inside(p)} by midpoint sampling on a grid of spacing h (m).
    labeler(p) -> int array (or None for a single label 0). Returns dict label -> volume (m3), and the
    first moments (for centres of mass) and second moments (for inertia) per label."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo)/h).astype(int), 1)
    xs = [lo[i] + (np.arange(n[i]) + .5)*h for i in range(3)]
    vol, mom1, mom2 = {}, {}, {}
    dv = h**3
    # iterate over x slabs to keep memory flat
    YZ = np.stack(np.meshgrid(xs[1], xs[2], indexing="ij"), -1).reshape(-1, 2)
    per = max(1, chunk//len(YZ))
    for i0 in range(0, n[0], per):
        xx = xs[0][i0:i0 + per]
        P = np.concatenate([np.column_stack([np.full(len(YZ), x), YZ]) for x in xx])
        m = inside(P)
        if not m.any():
            continue
        Q = P[m]
        lab = labeler(Q) if labeler is not None else np.zeros(len(Q), int)
        for L in np.unique(lab):
            sel = Q[lab == L]
            vol[int(L)] = vol.get(int(L), 0.) + len(sel)*dv
            mom1[int(L)] = mom1.get(int(L), 0.) + sel.sum(0)*dv
            mom2[int(L)] = mom2.get(int(L), 0.) + np.einsum("ni,nj->ij", sel, sel)*dv
    return vol, mom1, mom2


def inertia_from_moments(V, m1, m2, rho):
    """Mass, com and inertia about the com from grid moments."""
    M = rho*V
    com = m1/V
    S = rho*m2 - M*np.outer(com, com)          # second moment about com
    I = np.trace(S)*np.eye(3) - S
    return M, com, I


# ------------------------------------------------------------------------------------------ W myomeres
class Myomeres:
    """W-shaped myomere model (smeared muscle sheets). The myosepta are the level sets phi = integer of

        phi(p) = (s - s_first - L_c * G(r)) / lam,

    where s is the axial station, lam the segment (vertebra) spacing, and r the in-section distance of p from
    the nearest cone axis (one epaxial and one hypaxial axis per side, placed in the middle of each muscle
    quadrant), normalised by the height of that muscle mass (so the forward cusp sits on the horizontal septum). G(r) = -r gives nested cones: posterior-pointing cones on
    the two axes (the dorsal and ventral "posterior flexures" of the salmonid W) and a forward-pointing cusp
    at the horizontal septum and at the dorsal and ventral edges, i.e. the sideways W seen on the skin,
    and concentric rings on a transverse cut (the familiar salmon-steak pattern).

    Inputs are callables in the anatomical frame; see trout.py/salmon.py for their provenance.
    """

    def __init__(self, s_first, lam, cone_len, axes, blend=.08):
        self.s_first, self.lam, self.L = s_first, lam, cone_len
        self.axes = axes        # callable(s) -> (v_epi, v_hypo, w_off, scale) arrays
        self.blend = blend

    def phase(self, s, v, w):
        v_e, v_h, w_o, sc_e, sc_h = self.axes(s)
        aw = np.abs(w)
        # each cone is normalised by the height of its own muscle mass (epaxial above the horizontal septum,
        # hypaxial below), so the forward cusp of the W lies exactly on the horizontal septum
        r_e = np.hypot(v - v_e, aw - w_o)/sc_e
        r_h = np.hypot(v - v_h, aw - w_o)/sc_h
        r = smin(r_e, r_h, self.blend)
        return (s - self.s_first + self.L*r)/self.lam

    def normal_and_spacing(self, s, v, w, eps=5e-5):
        f = self.phase
        g = np.stack([(f(s + eps, v, w) - f(s - eps, v, w))/(2*eps),
                      (f(s, v + eps, w) - f(s, v - eps, w))/(2*eps),
                      (f(s, v, w + eps) - f(s, v, w - eps))/(2*eps)], 1)
        gn = np.linalg.norm(g, axis=1)
        return g/np.maximum(gn, 1e-12)[:, None], 1./np.maximum(gn, 1e-12)   # normal, local sheet spacing (m)

    def septum_distance(self, s, v, w):
        """Distance (m) from the nearest myoseptum (first order)."""
        ph = self.phase(s, v, w)
        _, sp = self.normal_and_spacing(s, v, w)
        return np.abs(ph - np.round(ph))*sp


# ------------------------------------------------------------------------------------------ textures
def value_noise(shape, cell, rng, octaves=4, persistence=.5):
    """Tileable-enough fractal value noise in [0,1] (bicubic-upsampled random lattices)."""
    from scipy.ndimage import zoom
    H, W = shape
    out = np.zeros(shape)
    amp, tot = 1., 0.
    for o in range(octaves):
        c = max(cell/(2**o), 1.)
        gh, gw = int(np.ceil(H/c)) + 3, int(np.ceil(W/c)) + 3
        g = rng.random((gh, gw))
        z = zoom(g, (H/(gh - 3) , W/(gw - 3)), order=3)[:H, :W]
        out += amp*z
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min())/max(out.max() - out.min(), 1e-9)


def scatter_points(n, rng, weight=None, shape=None, min_dist=0.):
    """Random points in [0,1]^2 with optional acceptance weight(u,v) in [0,1] and a minimum spacing."""
    pts = []
    tries = 0
    while len(pts) < n and tries < 60*n:
        tries += 1
        u, v = rng.random(), rng.random()
        if weight is not None and rng.random() > weight(u, v):
            continue
        if min_dist > 0 and pts:
            q = np.array(pts)
            if np.min(np.hypot(q[:, 0] - u, q[:, 1] - v)) < min_dist:
                continue
        pts.append((u, v))
    return np.array(pts)


def save_png(path, rgb):
    from PIL import Image
    arr = np.clip(np.asarray(rgb)*255 if np.asarray(rgb).dtype.kind == "f" else rgb, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, optimize=True)
    return path


def srgb(hexstr):
    h = hexstr.lstrip("#")
    return np.array([int(h[i:i + 2], 16)/255. for i in (0, 2, 4)])


# ------------------------------------------------------------------------------------------ marching tets
def marching_tets(f, lo, hi, h, project=True):
    """Triangle surface of {f < 0} on a regular grid (6 tets per cube), vectorised. Vertices are placed by
    linear interpolation on grid edges and (optionally) moved once along the gradient onto f = 0.
    Returns (V, F) with outward (toward f > 0) orientation. For renders and area integrals only."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo)/h).astype(int), 1) + 1
    gx = [lo[i] + np.arange(n[i])*h for i in range(3)]
    G = np.stack(np.meshgrid(*gx, indexing="ij"), -1).reshape(-1, 3)
    val = np.concatenate([f(G[i:i + 1_000_000]) for i in range(0, len(G), 1_000_000)])
    nx, ny, nz = n

    def idx(i, j, k):
        return (i*ny + j)*nz + k
    I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    corners = np.stack([idx(I + a, J + b, K + c) for a, b, c in
                        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]], 1)
    # keep only cubes with a sign change
    cv = val[corners]
    mixed = (cv < 0).any(1) & (cv >= 0).any(1)
    corners = corners[mixed]
    tets6 = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    T = corners[:, tets6].reshape(-1, 4)
    s = val[T] < 0
    cnt = s.sum(1)
    tris = []
    edges_all = []
    # one-in / one-out cases -> 1 triangle; two-in -> quad (2 triangles)
    for c in (1, 3):
        sel = T[cnt == c]
        ss = s[cnt == c] if c == 1 else ~s[cnt == c]
        odd = np.argmax(ss, 1)
        others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[odd]
        r = np.arange(len(sel))
        a = sel[r, odd]
        e = [np.sort(np.stack([a, sel[r, others[:, q]]], 1), 1) for q in range(3)]
        tris.append(np.stack(e, 1))   # (m, 3, 2)
    sel = T[cnt == 2]
    ss = s[cnt == 2]
    order = np.argsort(~ss, 1, kind="stable")
    a0, a1 = sel[np.arange(len(sel)), order[:, 0]], sel[np.arange(len(sel)), order[:, 1]]
    b0, b1 = sel[np.arange(len(sel)), order[:, 2]], sel[np.arange(len(sel)), order[:, 3]]
    e00 = np.sort(np.stack([a0, b0], 1), 1)
    e01 = np.sort(np.stack([a0, b1], 1), 1)
    e11 = np.sort(np.stack([a1, b1], 1), 1)
    e10 = np.sort(np.stack([a1, b0], 1), 1)
    tris.append(np.stack([e00, e01, e11], 1))
    tris.append(np.stack([e00, e11, e10], 1))
    E = np.concatenate(tris)                # (m, 3, 2) edge endpoints per triangle vertex
    key = E.reshape(-1, 2)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    fa, fb = val[uk[:, 0]], val[uk[:, 1]]
    t = fa/(fa - fb)
    V = G[uk[:, 0]] + t[:, None]*(G[uk[:, 1]] - G[uk[:, 0]])
    F = inv.reshape(-1, 3)
    if project:
        eps = 1e-3*h
        d = f(V)
        g = np.stack([(f(V + e) - f(V - e))/(2*eps) for e in np.eye(3)*eps], 1)
        gn2 = np.maximum((g*g).sum(1), 1e-18)
        step = -(d/gn2)[:, None]*g
        sl = np.linalg.norm(step, axis=1)
        step[sl > .5*h] = 0.
        V = V + step
    # orient outward: normal should align with the gradient of f
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    c = V[F].mean(1)
    eps = 1e-3*h
    g = np.stack([(f(c + e) - f(c - e))/(2*eps) for e in np.eye(3)*eps], 1)
    flip = (nrm*g).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    area = .5*np.linalg.norm(nrm, axis=1)
    keep = area > 1e-14
    return V, F[keep]

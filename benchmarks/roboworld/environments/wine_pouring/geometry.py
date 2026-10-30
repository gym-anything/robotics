"""Axisymmetric containers: bottles, stemware, tumbler, carafes, dump bucket, coasters.

Every container is a solid of revolution described by one closed polygon in the (r, z) half-plane (its glass or
steel cross-section, z along the container axis, z = 0 at its standing plane) and the polygon of its liquid space.
From these two polygons, exactly and consistently:
  * interior volume to any level and brimful capacity (Green's theorem on the clipped polygon, 2 pi int r dA);
  * mass, centre of mass and inertia of the glass (same integrals with r z, r^3, r z^2);
  * the liquid boundary: samples on every wetted surface (inner and outer faces, rim) and a signed-distance grid of
    the wall cross-section for the liquid's non-penetration guard (liquid.py);
  * the rigid collision shapes for MuJoCo: solid frusta where nothing can enter (bottle, stems, feet, bases) and
    hollow wall bands split into convex sectors where a bottle neck or a finger can enter (bowls, carafe necks);
  * the render mesh (the polygon revolved, watertight, so RTX refracts through the real wall thickness).

Product dimensions are in params-style comments with their provenance; everything not on a datasheet is authored and
only tuned to published capacities or masses.
"""

import math
from dataclasses import dataclass, field
import numpy as np

from . import params as P

MM = 1e-3


# ---------------------------------------------------------------------------------------------- polygon integrals
_GX = np.array([0.5 - math.sqrt(15)/10, 0.5, 0.5 + math.sqrt(15)/10])   # 3-point Gauss-Legendre on [0, 1]
_GW = np.array([5/18, 8/18, 5/18])


def _green(poly, f):
    """Integral over the polygon region of g(r, z) dA via Green's theorem, with f = antiderivative of g in r
    (int g dr), as  oint f(r, z) dz  (counter-clockwise). Exact for polynomial integrands up to degree 5."""
    poly = np.asarray(poly, float)
    total = 0.
    for (r0, z0), (r1, z1) in zip(poly, np.roll(poly, -1, axis=0)):
        if z1 == z0:
            continue
        r = r0 + _GX*(r1 - r0)
        z = z0 + _GX*(z1 - z0)
        total += (z1 - z0)*float(np.sum(_GW*f(r, z)))
    return total


def signed_area(poly):
    p = np.asarray(poly, float)
    return 0.5*float(np.sum(p[:, 0]*np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1)*p[:, 1]))


def ccw(poly):
    p = np.asarray(poly, float)
    return p if signed_area(p) > 0 else p[::-1].copy()


def revolved_volume(poly):
    """Volume of the solid of revolution of a (r >= 0) polygon: 2 pi int r dA."""
    return 2*math.pi*_green(ccw(poly), lambda r, z: r*r/2)


def clip_below(poly, level):
    """Sutherland-Hodgman clip of a polygon to z <= level."""
    out = []
    p = np.asarray(poly, float)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        ina, inb = a[1] <= level, b[1] <= level
        if ina:
            out.append(a)
        if ina != inb:
            t = (level - a[1])/(b[1] - a[1])
            out.append(a + t*(b - a))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def point_in_polygon(points, poly):
    """Even-odd test, vectorised over points (N, 2)."""
    pts = np.asarray(points, float)
    p = np.asarray(poly, float)
    inside = np.zeros(len(pts), bool)
    x, y = pts[:, 0], pts[:, 1]
    for (x0, y0), (x1, y1) in zip(p, np.roll(p, -1, axis=0)):
        cond = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0)*(x1 - x0)/(y1 - y0)
        inside ^= cond & (x < xc)
    return inside


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab)/max(float(ab @ ab), 1e-30), 0., 1.)
    d = points - (a + t[:, None]*ab)
    return np.hypot(d[:, 0], d[:, 1])


def surface_edges(poly):
    """Polygon edges that are real surfaces (not the r = 0 axis)."""
    p = np.asarray(poly, float)
    return [(a, b) for a, b in zip(p, np.roll(p, -1, axis=0)) if not (a[0] < 1e-9 and b[0] < 1e-9)]


# ---------------------------------------------------------------------------------------------- profile helpers
def arc(cx, cz, rad, a0, a1, n):
    """Points on a circle arc (angles in degrees, from the +r axis, counter-clockwise)."""
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rad*math.cos(x), cz + rad*math.sin(x)) for x in t]


def cosine_blend(z0, z1, r0, r1, n=12):
    z = np.linspace(z0, z1, n)
    s = (z - z0)/(z1 - z0)
    return [(r1 + (r0 - r1)*0.5*(1 + math.cos(math.pi*u)), zz) for u, zz in zip(s, z)]


def dedupe(points, tol=1e-7):
    out = []
    for p in points:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


# ---------------------------------------------------------------------------------------------- container
@dataclass
class Container:
    kind: str
    solid: np.ndarray            # (M, 2) closed polygon of the wall material, CCW in (r, z)
    interior: np.ndarray         # (K, 2) closed polygon of the liquid space (closed along the rim plane and the axis)
    rim_z: float                 # mouth plane (top of the interior)
    mouth_r: float               # inner radius at the mouth
    density: float               # kg/m^3 of the wall material
    material: str                # render material key
    rigid: list                  # rigid collision pieces (see solid_frustum / hollow_band)
    lines: dict = field(default_factory=dict)        # printed/fired fill lines: name -> z
    mass_override: float = None  # datasheet mass (scales density) or None
    notes: str = ""

    # -------------------------------------------------- volumes and mass
    def volume_below(self, z):
        return revolved_volume(clip_below(self.interior, z)) if z > self.interior[:, 1].min() else 0.

    @property
    def capacity(self):
        return self.volume_below(self.rim_z)

    def level_for(self, volume):
        lo, hi = float(self.interior[:, 1].min()), self.rim_z
        if volume > self.capacity:
            raise ValueError(f"{self.kind}: {volume*1e6:.1f} mL exceeds the {self.capacity*1e6:.1f} mL capacity")
        for _ in range(60):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if self.volume_below(mid) < volume else (lo, mid)
        return 0.5*(lo + hi)

    @property
    def glass_volume(self):
        return revolved_volume(self.solid)

    @property
    def mass(self):
        return self.mass_override if self.mass_override else self.density*self.glass_volume

    @property
    def effective_density(self):
        return self.mass/self.glass_volume

    def mass_properties(self):
        """Mass, centre of mass (body frame) and inertia about the centre of mass (diagonal: Ixx = Iyy, Izz)."""
        poly = ccw(self.solid)
        rho = self.effective_density
        a1 = _green(poly, lambda r, z: r*r/2)              # int r dA
        az = _green(poly, lambda r, z: r*r*z/2)            # int r z dA
        a3 = _green(poly, lambda r, z: r**4/4)             # int r^3 dA
        azz = _green(poly, lambda r, z: r*r*z*z/2)         # int r z^2 dA
        m = 2*math.pi*rho*a1
        zc = az/a1
        izz = 2*math.pi*rho*a3
        ixx = 2*math.pi*rho*(a3/2 + azz) - m*zc*zc
        return m, np.array([0., 0., zc]), np.array([ixx, ixx, izz])

    # -------------------------------------------------- liquid boundary data
    def boundary_samples(self, spacing):
        """Points on every wetted surface (all non-axis polygon edges, revolved), body frame, with outward
        normals of the solid. Spacing along the profile and around the circumference ~= `spacing`."""
        pts, nrm = [], []
        for a, b in surface_edges(ccw(self.solid)):
            L = math.hypot(*(b - a))
            n = max(1, int(math.ceil(L/spacing)))
            t2 = (b - a)/L
            n2 = np.array([t2[1], -t2[0]])                 # outward normal of a CCW polygon
            for k in range(n):
                r, z = a + (k + 0.5)/n*(b - a)
                m = max(1, int(round(2*math.pi*r/spacing))) if r > 0.5*spacing else 1
                th = 2*math.pi*(np.arange(m) + 0.5*(k % 2))/m
                ct, st = np.cos(th), np.sin(th)
                pts.append(np.stack([r*ct, r*st, np.full(m, z)], 1))
                nrm.append(np.stack([n2[0]*ct, n2[0]*st, np.full(m, n2[1])], 1))
        return np.concatenate(pts).astype(np.float32), np.concatenate(nrm).astype(np.float32)

    def sdf_grid(self, h=0.25*MM, margin=6*MM):
        """Signed distance (m, negative inside the wall material) to the solid's surfaces on an (r, z) grid."""
        poly = ccw(self.solid)
        r_max = float(poly[:, 0].max()) + margin
        z0, z1 = float(poly[:, 1].min()) - margin, float(poly[:, 1].max()) + margin
        nr, nz = int(math.ceil(r_max/h)) + 1, int(math.ceil((z1 - z0)/h)) + 1
        R, Z = np.meshgrid(np.arange(nr)*h, z0 + np.arange(nz)*h, indexing="ij")
        pts = np.stack([R.ravel(), Z.ravel()], 1)
        d = np.full(len(pts), np.inf)
        for a, b in surface_edges(poly):
            d = np.minimum(d, segment_distance(pts, a, b))
        inside = point_in_polygon(pts, poly)
        sdf = np.where(inside, -d, d).reshape(nr, nz).astype(np.float32)
        return dict(sdf=sdf, h=h, z0=z0, nr=nr, nz=nz)

    def density_map(self, dx, h, knorm, scale, res=0.5*MM, quad=10, occl=6):
        """Wall density map in this container's (r, z) frame (Koschier & Bender 2017, exact geometry): for a particle
        at (r, z), Phi = integral of the kernel W over the part of its support hidden behind the wall (points inside the
        glass, or reached only through it: line of sight from the particle crosses the solid), times `scale` (matches
        the continuum half-space value to the dx lattice it stands for). Side-aware by construction: from inside a
        bowl the air behind its 1 mm wall counts, from the mouth the interior does not. Returns dict(phi, dphi_dr,
        dphi_dz, h, r0, z0, nr, nz) on a `res` grid (values 0 more than h from the solid). Cached on disk."""
        import hashlib
        from pathlib import Path
        key = hashlib.sha256(np.ascontiguousarray(self.solid).tobytes() + repr((dx, h, knorm, scale, res, quad, occl, 3)).encode()).hexdigest()[:16]
        import os
        cache = Path(os.environ.get("WINE_POURING_CACHE") or Path(__file__).resolve().parent/".cache")/f"densitymap_{self.kind}_{key}.npz"
        if cache.is_file():
            d = np.load(cache)
            return {k: (d[k] if d[k].ndim else float(d[k])) for k in d.files} | {"nr": int(d["nr"]), "nz": int(d["nz"])}
        poly = ccw(self.solid)
        # fine raster of the solid (0.1 mm) for membership
        fr = 0.1*MM
        rmax = float(poly[:, 0].max()) + h + 2*res
        zlo, zhi = float(poly[:, 1].min()) - h - 2*res, float(poly[:, 1].max()) + h + 2*res
        nrf, nzf = int(rmax/fr) + 2, int((zhi - zlo)/fr) + 2
        R, Z = np.meshgrid((np.arange(nrf) + 0.5)*fr, zlo + (np.arange(nzf) + 0.5)*fr, indexing="ij")
        mask = point_in_polygon(np.stack([R.ravel(), Z.ravel()], 1), poly).reshape(nrf, nzf)

        def solid_at(rr, zz):
            i = np.clip((rr/fr).astype(np.int64), 0, nrf - 1)
            j = np.clip(((zz - zlo)/fr).astype(np.int64), 0, nzf - 1)
            out = mask[i, j]
            return out & (rr < rmax - fr) & (zz > zlo) & (zz < zhi - fr)
        # quadrature over the support ball
        g = (np.arange(-quad, quad) + 0.5)*(h/quad)
        Q = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        rq = np.linalg.norm(Q, axis=1)
        Q, rq = Q[rq < h], rq[rq < h]
        q = rq/h
        Wq = knorm*np.where(q <= .5, 6*(q**3 - q**2) + 1, 2*(1 - q)**3)*(h/quad)**3
        ts = (np.arange(1, occl + 1) - 0.5)/occl
        # grid points within h + res of the solid
        nr, nz = int(rmax/res) + 1, int((zhi - zlo)/res) + 1
        r0s, z0s = np.arange(nr)*res, zlo + np.arange(nz)*res
        sd = self.sdf_grid(h=res, margin=h + 3*res)
        phi = np.zeros((nr, nz), np.float32)
        Rg, Zg = np.meshgrid(r0s, z0s, indexing="ij")
        # signed distance at grid points (bilinear from the sdf grid)
        u = np.clip(Rg/sd["h"], 0, sd["nr"] - 1.001); w = np.clip((Zg - sd["z0"])/sd["h"], 0, sd["nz"] - 1.001)
        iu, iw = u.astype(int), w.astype(int); fu, fw = u - iu, w - iw
        S = sd["sdf"]
        dist = (S[iu, iw]*(1 - fu) + S[iu + 1, iw]*fu)*(1 - fw) + (S[iu, iw + 1]*(1 - fu) + S[iu + 1, iw + 1]*fu)*fw
        idx = np.argwhere((dist < h + 2*res) & (dist > -2*MM))
        for c0 in range(0, len(idx), 256):
            chunk = idx[c0:c0 + 256]
            r0 = r0s[chunk[:, 0]][:, None]; z0 = z0s[chunk[:, 1]][:, None]
            hidden = np.zeros((len(chunk), len(Q)), bool)
            for t in list(ts) + [1.0]:
                rr = np.hypot(r0 + t*Q[None, :, 0], t*Q[None, :, 1])
                hidden |= solid_at(rr, z0 + t*Q[None, :, 2])
            phi[chunk[:, 0], chunk[:, 1]] = (hidden*Wq[None, :]).sum(1)*scale
        dphi_dr = np.gradient(phi, res, axis=0).astype(np.float32)
        dphi_dz = np.gradient(phi, res, axis=1).astype(np.float32)
        dphi_dr[0, :] = 0.                                  # symmetry on the axis
        out = dict(phi=phi, dphi_dr=dphi_dr, dphi_dz=dphi_dz, h=float(res), r0=0., z0=float(zlo), nr=nr, nz=nz)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
        return out

    def interior_contains(self, r, z):
        return point_in_polygon(np.stack([r, z], 1), self.interior) & (z <= self.rim_z)

    # -------------------------------------------------- render mesh
    def render_mesh(self, sectors=96):
        """Watertight surface of revolution of the solid polygon: (vertices (N, 3), triangles (T, 3))."""
        poly = ccw(self.solid)
        n = len(poly)
        th = 2*math.pi*np.arange(sectors)/sectors
        verts, tris = [], []
        axis_index = {}
        for i, (r, z) in enumerate(poly):
            if r < 1e-9:
                axis_index[i] = len(verts)
                verts.append((0., 0., z))
            else:
                axis_index[i] = None
                base = len(verts)
                for t in th:
                    verts.append((r*math.cos(t), r*math.sin(t), z))
                axis_index[i] = ("ring", base)
        for i in range(n):
            j = (i + 1) % n
            a, b = axis_index[i], axis_index[j]
            if isinstance(a, int) and isinstance(b, int):
                continue
            for s in range(sectors):
                s1 = (s + 1) % sectors
                if isinstance(a, int):
                    tris.append((a, b[1] + s, b[1] + s1))
                elif isinstance(b, int):
                    tris.append((a[1] + s, b, a[1] + s1))
                else:
                    tris.append((a[1] + s, b[1] + s, b[1] + s1))
                    tris.append((a[1] + s, b[1] + s1, a[1] + s1))
        V = np.array(verts, np.float32)
        T = np.array(tris, np.int32)
        # orient outward: revolving a CCW (r, z) polygon gives triangles whose winding depends on theta direction;
        # check with the divergence theorem (positive volume)
        vol = np.sum(np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])))/6
        if vol < 0:
            T = T[:, ::-1].copy()
        return V, T

    def render_mesh_shaded(self, sectors=96, crease_deg=30.):
        """Surface of revolution with crease-aware vertex normals: (vertices, triangles, normals). Every profile edge
        gets its own pair of rings; at a profile corner sharper than crease_deg (a rim's flat top, a foot's edge) each
        edge keeps its own normal, elsewhere the two edges' normals are averaged. Smoothing every corner (one ring per
        profile vertex) bent the rim normals into the wall and rendered a dark band round every glass rim."""
        poly = ccw(self.solid)
        n = len(poly)
        e = np.roll(poly, -1, axis=0) - poly
        en = np.stack([e[:, 1], -e[:, 0]], 1)
        en /= np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-12)
        cosc = math.cos(math.radians(crease_deg))
        th = 2*math.pi*np.arange(sectors)/sectors
        ct, st = np.cos(th), np.sin(th)
        V, N, T = [], [], []
        for i in range(n):
            j = (i + 1) % n
            ends = []
            for k, other in ((i, (i - 1) % n), (j, j)):
                own = en[i]
                nb = en[(i - 1) % n] if k == i else en[j]
                nrm = own if float(own @ nb) < cosc else (own + nb)/max(np.linalg.norm(own + nb), 1e-12)
                ends.append((poly[k], nrm))
            (pa, na), (pb, nb_) = ends
            if pa[0] < 1e-9 and pb[0] < 1e-9:
                continue
            base = len(V)
            for (r, z), (nr, nz) in ((pa, na), (pb, nb_)):
                for c, s_ in zip(ct, st):
                    V.append((r*c, r*s_, z)); N.append((nr*c, nr*s_, nz))
            for s_ in range(sectors):
                s1 = (s_ + 1) % sectors
                a0, a1, b0, b1 = base + s_, base + s1, base + sectors + s_, base + sectors + s1
                if pa[0] >= 1e-9:
                    T.append((a0, b0, b1) if pb[0] >= 1e-9 else (a0, b0, a1))
                if pb[0] >= 1e-9 and pa[0] >= 1e-9:
                    T.append((a0, b1, a1))
                elif pa[0] < 1e-9:
                    T.append((a0, b0, b1))
        V = np.array(V, np.float32); N = np.array(N, np.float32); T = np.array(T, np.int32)
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        if float(np.einsum("ij,ij->i", fn, N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]]).sum()) < 0:
            T = T[:, ::-1].copy()
        return V, T, N

    def interior_mesh(self, level, sectors=96):
        """Not used for physics: the free surface of a still liquid at `level` (render previews)."""
        clipped = clip_below(self.interior, level)
        sub = Container(self.kind + "_liquid", clipped, clipped, level, 0., 1., "liquid", [])
        return sub.render_mesh(sectors)


# ---------------------------------------------------------------------------------------------- rigid pieces
def solid_frustum(z0, z1, r0, r1, sections=24):
    return dict(type="solid", z0=float(z0), z1=float(z1), r0=float(r0), r1=float(r1), sections=sections)


def hollow_band(z0, z1, ri0, ri1, ro0, ro1, sectors=16):
    return dict(type="band", z0=float(z0), z1=float(z1), ri0=float(ri0), ri1=float(ri1), ro0=float(ro0),
                ro1=float(ro1), sectors=sectors)


def rigid_hull_points(piece):
    """Convex point sets (body frame) of one rigid piece (a list: one for a solid frustum, `sectors` for a band)."""
    if piece["type"] == "solid":
        th = 2*math.pi*np.arange(piece["sections"])/piece["sections"]
        ring = lambda r, z: np.stack([r*np.cos(th), r*np.sin(th), np.full_like(th, z)], 1)
        return [np.concatenate([ring(piece["r0"], piece["z0"]), ring(piece["r1"], piece["z1"])]).astype(np.float32)]
    out = []
    n = piece["sectors"]
    for k in range(n):
        a = np.linspace(2*math.pi*k/n, 2*math.pi*(k + 1)/n, 3)     # 3 angles per sector: a slightly curved slab
        pts = []
        for z, ri, ro in ((piece["z0"], piece["ri0"], piece["ro0"]), (piece["z1"], piece["ri1"], piece["ro1"])):
            for t in a:
                pts.append((ro*math.cos(t), ro*math.sin(t), z))
            for t in (a[0], a[-1]):                                  # inner face: chord ends only (stays convex)
                pts.append((ri*math.cos(t), ri*math.sin(t), z))
        out.append(np.array(pts, np.float32))
    return out


def bands_from_profiles(z_breaks, r_in, r_out, sectors=16):
    """Hollow bands between consecutive heights, inner/outer radii from profile functions (chords inside the wall)."""
    pieces = []
    for z0, z1 in zip(z_breaks[:-1], z_breaks[1:]):
        pieces.append(hollow_band(z0, z1, r_in(z0), r_in(z1), r_out(z0), r_out(z1), sectors))
    return pieces


def profile_function(points):
    """r(z) by linear interpolation of a polyline monotonic in z."""
    p = np.asarray(points, float)
    order = np.argsort(p[:, 1], kind="stable")
    zs, rs = p[order, 1], p[order, 0]
    return lambda z: float(np.interp(z, zs, rs))


# ---------------------------------------------------------------------------------------------- bottles
LEONOR = dict(height=0.310, base_d=0.072, shoulder_d=0.077, bore=0.0185, fill_from_top=0.063, punt=0.025, mass=0.498,
              source="Saverglass LEONOR 75 cl technical sheet [S11] (datasheet); EN 12726 bore profile [S15]")
BG_CLASSIQUE2 = dict(height=0.2961, body_d=0.081, bore=0.0185, fill_from_top=0.063, punt=0.025, mass=0.486,
                     source="Saverglass BG CLASSIQUE 2 75 cl technical sheet [S13] (datasheet)")


def _bore(z, top):
    """EN 12726 cork mouth: 18.5 mm at 3 mm below the top, 20 +/- 1 mm at 45 mm (standard via [S15])."""
    if z >= top - 0.003:
        return 0.00925
    if z >= top - 0.045:
        return 0.010 + (0.00925 - 0.010)*(z - (top - 0.045))/0.042
    return None


def _bottle(kind, outer_body, shoulder_z0, shoulder_len, neck_r0, neck_r1, lip_r, height, base_t, wall, neck_wall,
            heel_r, punt_h, punt_r, color):
    """Bordeaux/Burgundy bottle. outer_body(z) gives the body's outer radius below the shoulder."""
    top = height
    z_neck0 = shoulder_z0 + shoulder_len
    z_lip0 = top - 0.012
    r_b0 = outer_body(0.)
    # ---- outer contour, from the punt top (on the axis) outward and up to the rim
    outer = [(0., punt_h)]
    for u in np.linspace(0., 1., 9)[1:]:                         # punt recess (paraboloid) down to the heel ring
        r = punt_r*u
        outer.append((r, punt_h*(1 - u*u)))
    outer += [(r_b0 - heel_r, 0.)]
    outer += arc(r_b0 - heel_r, heel_r, heel_r, -90., 0., 6)[1:]   # heel round
    for z in np.linspace(heel_r, shoulder_z0, 10)[1:]:
        outer.append((outer_body(z), z))
    outer += cosine_blend(shoulder_z0, z_neck0, outer_body(shoulder_z0), neck_r0, 14)[1:]
    for z in np.linspace(z_neck0, z_lip0, 6)[1:]:
        outer.append((neck_r0 + (neck_r1 - neck_r0)*(z - z_neck0)/(z_lip0 - z_neck0), z))
    outer += [(lip_r, z_lip0 + 0.0005), (lip_r, top - 0.0015)]
    outer += arc(lip_r - 0.0015, top - 0.0015, 0.0015, 0., 90., 5)[1:]
    outer += [(0.00925 + 0.001, top)]
    # ---- inner contour (traversed downward from the rim to the floor centre)
    def r_in(z):
        if z < shoulder_z0:
            ro = outer_body(z)
            r = ro - wall
        elif z < z_neck0:
            s = (z - shoulder_z0)/shoulder_len
            ro = neck_r0 + (outer_body(shoulder_z0) - neck_r0)*0.5*(1 + math.cos(math.pi*s))
            t = wall + (neck_wall - wall)*s
            slope = math.pi*(outer_body(shoulder_z0) - neck_r0)/(2*shoulder_len)*math.sin(math.pi*s)
            r = ro - t*math.sqrt(1 + slope*slope)
        else:
            ro = neck_r0 + (neck_r1 - neck_r0)*(z - z_neck0)/(z_lip0 - z_neck0)
            r = ro - neck_wall
        b = _bore(z, top)
        return min(r, b) if b is not None else r
    zs = sorted(set(np.concatenate([np.linspace(base_t + 0.004, shoulder_z0, 12),
                                    np.linspace(shoulder_z0, z_neck0, 16),
                                    np.linspace(z_neck0, top - 0.045, 6),
                                    np.linspace(top - 0.045, top - 0.003, 6), [top]])))
    inner_up = [(r_in(z), z) for z in zs]
    r_floor = r_in(base_t + 0.004)
    floor = arc(r_floor - 0.004, base_t + 0.004, 0.004, 270., 360., 5)[:-1]      # inner heel fillet
    dome_top = punt_h + base_t
    dome = [(punt_r*u, base_t + (dome_top - base_t)*(1 - u*u)) for u in np.linspace(0., 1., 9)]
    inner_bottom = dome + [(r_floor - 0.004, base_t)] + floor
    inner = inner_bottom + inner_up                          # from the axis outward and up to the mouth
    solid = dedupe(outer + inner[::-1])
    interior = dedupe(inner + [(0., top)])
    rigid = [solid_frustum(0., heel_r, r_b0 - heel_r*0.3, r_b0, 32)]
    for z0, z1 in zip(np.linspace(heel_r, shoulder_z0, 4)[:-1], np.linspace(heel_r, shoulder_z0, 4)[1:]):
        rigid.append(solid_frustum(z0, z1, outer_body(z0), outer_body(z1), 32))
    sh = cosine_blend(shoulder_z0, z_neck0, outer_body(shoulder_z0), neck_r0, 7)
    for (ra, za), (rb, zb) in zip(sh[:-1], sh[1:]):
        rigid.append(solid_frustum(za, zb, ra, rb, 32))
    rigid.append(solid_frustum(z_neck0, z_lip0, neck_r0, neck_r1, 24))
    rigid.append(solid_frustum(z_lip0, top, lip_r, lip_r, 24))
    return Container(kind, ccw(np.array(solid)), np.array(ccw(np.array(interior))), top, 0.00925,
                     P.SODA_LIME_DENSITY, color, rigid)


def leonor(color="glass_green", base_t=None, shoulder_z0=None):
    """Saverglass LEONOR 75 cl Bordeaux: shoulder height tuned so the datasheet fill line (63 mm below the top)
    holds 750 mL; base/punt thickness tuned so the glass weighs the datasheet 498 g (a 6 mm base gives
    448 g)."""
    body = lambda z: 0.036 + (0.0385 - 0.036)*min(z, 0.188)/0.188

    def make(bt, s0):
        return _bottle("bottle_leonor", body, s0, 0.040, 0.0150, 0.0140, 0.01475, LEONOR["height"], bt, 0.0026, 0.0045,
                       0.0045, LEONOR["punt"], 0.026, color)
    if base_t is None or shoulder_z0 is None:
        base_t, shoulder_z0 = _tune_bottle(make, LEONOR)
    c = make(base_t, shoulder_z0)
    c.mass_override = LEONOR["mass"]
    c.lines = {"fill_750": LEONOR["height"] - LEONOR["fill_from_top"]}
    c.notes = f"base {base_t*1e3:.2f} mm, shoulder from {shoulder_z0*1e3:.1f} mm; {LEONOR['source']}"
    return c


def burgundy(color="glass_green", base_t=None, shoulder_z0=None):
    """Saverglass BG CLASSIQUE 2 (Burgundy, 81 mm body): long sloping shoulder (authored), tuned like leonor()."""
    body = lambda z: 0.0405

    def make(bt, s0):
        return _bottle("bottle_burgundy", body, s0, 0.085, 0.0150, 0.0142, 0.0160, BG_CLASSIQUE2["height"], bt, 0.0028,
                       0.0045, 0.006, BG_CLASSIQUE2["punt"], 0.028, color)
    if base_t is None or shoulder_z0 is None:
        base_t, shoulder_z0 = _tune_bottle(make, BG_CLASSIQUE2)
    c = make(base_t, shoulder_z0)
    c.mass_override = BG_CLASSIQUE2["mass"]
    c.lines = {"fill_750": BG_CLASSIQUE2["height"] - BG_CLASSIQUE2["fill_from_top"]}
    c.notes = f"base {base_t*1e3:.2f} mm, shoulder from {shoulder_z0*1e3:.1f} mm; {BG_CLASSIQUE2['source']}"
    return c


def _tune_bottle(make, spec):
    fill_z = spec["height"] - spec["fill_from_top"]
    bt, s0 = 0.006, 0.17
    for _ in range(6):
        lo, hi = 0.10, 0.215                                 # shoulder start: 750 mL below the fill line
        for _ in range(40):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if make(bt, mid).volume_below(fill_z) < 750e-6 else (lo, mid)
        s0 = 0.5*(lo + hi)
        lo, hi = 0.003, 0.016                                # base thickness: datasheet glass mass
        for _ in range(40):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if make(mid, s0).glass_volume*P.SODA_LIME_DENSITY < spec["mass"] else (lo, mid)
        bt = 0.5*(lo + hi)
    return bt, s0


# ---------------------------------------------------------------------------------------------- stemware
def _stemware(kind, foot_r, foot_t, stem_r, bowl_floor_z, inner_bowl, wall, bottom_t, density, mass=None, lines=None,
              sectors=20):
    """Stemmed glass: inner_bowl = list of (r, z') from the bowl floor centre (0, 0) up to the rim inner point,
    z' above the bowl floor. The outer bowl is the inner offset by `wall` along the normal; the rim is rounded."""
    inner = [(r, bowl_floor_z + z) for r, z in inner_bowl]
    pts = np.array(inner)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.stack([tang[:, 1], -tang[:, 0]], 1)                  # outward (to larger r / below)
    outer_bowl = pts + wall*normal
    rim_in = pts[-1]
    rim_z = float(rim_in[1])
    # outer contour from the foot centre outward
    outer = [(0., 0.), (foot_r - 0.0015, 0.), *arc(foot_r - 0.0015, 0.0015, 0.0015, -90., 90., 5)[1:-1],
             (foot_r - 0.0015, foot_t), (stem_r + 0.006, foot_t + 0.0035), (stem_r + 0.0015, foot_t + 0.009)]
    z_stem_top = bowl_floor_z - bottom_t - 0.006
    outer += [(stem_r, foot_t + 0.013), (stem_r, z_stem_top)]
    ob = [(float(r), float(z)) for r, z in outer_bowl if r > stem_r + 0.0015 and z < rim_z - 1e-4]
    outer += [(stem_r + 0.0015, z_stem_top + 0.003)]
    outer += ob
    outer += [(float(rim_in[0]) + wall*0.5, rim_z)]                    # rounded rim (flat top across the wall)
    solid = dedupe(outer + [(float(r), float(z)) for r, z in inner[::-1]])
    interior = dedupe(inner + [(0., rim_z)])
    # rigid: foot disc, stem, bowl bottom cap (solid), bowl wall bands (hollow, convex sectors)
    fr_in = profile_function(inner)
    fr_out = profile_function([(float(r), float(z)) for r, z in outer_bowl])
    zb = [bowl_floor_z + 0.003]
    zb += list(np.linspace(bowl_floor_z + 0.008, rim_z, 9))
    rigid = [solid_frustum(0., foot_t, foot_r, foot_r - 0.001, 32), solid_frustum(foot_t, z_stem_top + 0.003, stem_r + 0.001, stem_r, 12),
             solid_frustum(z_stem_top + 0.003, bowl_floor_z + 0.003, stem_r + 0.0015, fr_out(bowl_floor_z + 0.003), 24)]
    rigid += bands_from_profiles(zb, lambda z: max(0.0005, fr_in(z)), fr_out, sectors)
    c = Container(kind, ccw(np.array(solid)), ccw(np.array(interior)), rim_z, float(rim_in[0]), density, "crystal", rigid,
                  lines=lines or {}, mass_override=mass)
    return c


def pure_bordeaux(r_rim=None):
    """Zwiesel Glas / Schott Zwiesel PURE Bordeaux 8545/130: 680 mL, bowl 94 mm, 267 mm, 0.21 kg (datasheet
    [S16, S17]). Authored: quarter-ellipse bowl bottom widest 38 mm above the bowl floor, conical upper bowl, 1 mm
    wall, 84 mm foot; the rim radius is tuned to the 680 mL capacity."""
    floor_z, r_max, z_max, h = 0.1165, 0.046, 0.038, 0.1505

    def bowl(rr):
        pts = [(r_max*math.sqrt(max(0., 1 - (1 - z/z_max)**2)), z) for z in np.linspace(0., z_max, 14)]
        pts += [(r_max + (rr - r_max)*(z - z_max)/(h - z_max), z) for z in np.linspace(z_max, h, 8)[1:]]
        pts[0] = (0., 0.)
        return pts

    def make(rr):
        return _stemware("glass_pure", 0.042, 0.0045, 0.004, floor_z, bowl(rr), 0.0010, 0.004, P.TRITAN_CRYSTAL_DENSITY,
                         mass=0.21)
    if r_rim is None:
        lo, hi = 0.020, 0.046
        for _ in range(40):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if make(mid).capacity < 680e-6 else (lo, mid)
        r_rim = 0.5*(lo + hi)
    c = make(r_rim)
    c.notes = f"rim inner diameter {2*r_rim*1e3:.1f} mm (tuned to 680 mL)"
    return c


def iso3591():
    """ISO 3591:1977 tasting glass: total 155 mm, bowl 100 mm, rim 46 mm, widest 65 mm, base 65 mm, 215 +/- 10 mL,
    0.8 mm wall, 9 mm stem (standard [S6-S8]). Bowl: half-ellipsoid lower part, widest 25 mm above the bowl floor,
    power-law (1.3) upper taper (216 mL, 50 mL at the widest point)."""
    wall = 0.0008
    r_max, r_rim, floor_z = 0.0325 - wall, 0.023 - wall, 0.0558
    h = 0.155 - floor_z
    z_eq = 0.025
    pts = [(r_max*math.sqrt(max(0., 1 - (1 - z/z_eq)**2)), z) for z in np.linspace(0., z_eq, 12)]
    pts += [(r_rim + (r_max - r_rim)*(1 - (z - z_eq)/(h - z_eq))**1.3, z) for z in np.linspace(z_eq, h, 14)[1:]]
    pts[0] = (0., 0.)
    c = _stemware("glass_iso", 0.0325, 0.0035, 0.0045, floor_z, pts, wall, 0.003, P.TRITAN_CRYSTAL_DENSITY, sectors=16)
    c.notes = "ISO 3591 dimensions; bowl curve authored"
    return c


def lined_goblet():
    """UK lined wine glass: ~380 mL goblet with a fired line at the 175 mL level (Weights and Measures (Intoxicating
    Liquor) Order 1988 art. 5A [S5]). Shape authored (estimate)."""
    wall = 0.0012
    floor_z, r_max, z_max, h, r_rim = 0.085, 0.039, 0.032, 0.098, 0.0335
    pts = [(r_max*math.sqrt(max(0., 1 - (1 - z/z_max)**2)), z) for z in np.linspace(0., z_max, 12)]
    pts += [(r_max + (r_rim - r_max)*((z - z_max)/(h - z_max))**1.4, z) for z in np.linspace(z_max, h, 8)[1:]]
    pts[0] = (0., 0.)
    c = _stemware("glass_lined", 0.036, 0.004, 0.0045, floor_z, pts, wall, 0.004, P.SODA_LIME_DENSITY, mass=0.18)
    c.lines = {"line_175": c.level_for(175e-6)}
    c.notes = "authored goblet; line at 175 mL computed from its own interior"
    return c


# ---------------------------------------------------------------------------------------------- flat-bottomed vessels
def _cup(kind, profile_out, base_t, wall, density, color, mass=None, lines=None, sectors=16, rim_round=True):
    """Flat-based vessel: profile_out = list of (r, z) of the outer side wall from the base edge (z small) to the rim.
    Inner = outer offset inward by `wall` (along the normal); base `base_t` thick."""
    po = np.array(profile_out, float)
    tang = np.gradient(po, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    inward = np.stack([-tang[:, 1], tang[:, 0]], 1)
    pi = po + wall*inward
    rim_z = float(po[-1, 1])
    outer = [(0., 0.), (float(po[0, 0]) - 0.002, 0.), *arc(float(po[0, 0]) - 0.002, 0.002, 0.002, -90., 0., 4)[1:]]
    outer += [(float(r), float(z)) for r, z in po if z > 0.002]
    inner = [(0., base_t), (float(np.interp(base_t + 0.002, pi[:, 1], pi[:, 0])) - 0.002, base_t)]
    inner += [(float(r), float(z)) for r, z in pi if z > base_t + 0.002]
    inner[-1] = (inner[-1][0], rim_z)
    solid = dedupe(outer + inner[::-1])
    interior = dedupe(inner + [(0., rim_z)])
    f_in = profile_function(inner[1:])
    f_out = profile_function(outer[2:])
    zb = list(np.linspace(base_t, rim_z, 8))
    rigid = [solid_frustum(0., base_t, float(po[0, 0]), f_out(base_t), 32)]
    rigid += bands_from_profiles(zb, f_in, f_out, sectors)
    return Container(kind, ccw(np.array(solid)), ccw(np.array(interior)), rim_z, float(pi[-1, 0]), density, color, rigid,
                     lines=lines or {}, mass_override=mass)


def tumbler():
    """Restaurant water tumbler: 80 mm rim, 66 mm base, 105 mm tall, 2.5 mm wall, 6 mm base (estimate)."""
    prof = [(0.033 + (0.040 - 0.033)*z/0.105, z) for z in np.linspace(0., 0.105, 8)]
    return _cup("tumbler", prof, 0.006, 0.0025, P.SODA_LIME_DENSITY, "glass_clear")


def carafe_1l():
    """1 L water carafe: 100 mm body, 50 mm neck, 56 mm mouth, 235 mm tall, 3 mm wall (estimate)."""
    prof = [(0.050, z) for z in np.linspace(0., 0.13, 6)]
    prof += cosine_blend(0.13, 0.195, 0.050, 0.026, 8)[1:]
    prof += [(0.026 + 0.003*(z - 0.195)/0.04, z) for z in np.linspace(0.195, 0.235, 5)[1:]]
    return _cup("carafe_1l", prof, 0.006, 0.0030, P.SODA_LIME_DENSITY, "glass_clear", sectors=20)


def carafe_500():
    """500 mL lined wine carafe (UK measure [S5]): 80 mm body, 46 mm mouth, 190 mm tall (estimate); fired line at
    500 mL computed from its interior."""
    prof = [(0.040, z) for z in np.linspace(0., 0.10, 5)]
    prof += cosine_blend(0.10, 0.155, 0.040, 0.0215, 8)[1:]
    prof += [(0.0215 + 0.0025*(z - 0.155)/0.035, z) for z in np.linspace(0.155, 0.19, 4)[1:]]
    c = _cup("carafe_500", prof, 0.005, 0.0025, P.SODA_LIME_DENSITY, "glass_clear", sectors=20)
    c.lines = {"line_500": c.level_for(500e-6)}
    return c


def dump_bucket():
    """Stainless dump bucket (spittoon), 1.3 L class: 120 mm rim, 100 mm base, 150 mm tall, 0.8 mm wall (estimate)."""
    prof = [(0.050 + 0.010*z/0.15, z) for z in np.linspace(0., 0.15, 6)]
    return _cup("dump_bucket", prof, 0.0012, 0.0008, 7900., "stainless", mass=0.30)


def coaster():
    """Stainless bottle coaster: 105 mm dish, 15 mm rim, 1 mm sheet (estimate)."""
    prof = [(0.0525, z) for z in np.linspace(0., 0.015, 3)]
    return _cup("coaster", prof, 0.0015, 0.0010, 7900., "stainless", mass=0.15, sectors=24)


BUILDERS = dict(bottle_leonor=leonor, bottle_burgundy=burgundy, glass_pure=pure_bordeaux, glass_iso=iso3591,
                glass_lined=lined_goblet, tumbler=tumbler, carafe_1l=carafe_1l, carafe_500=carafe_500,
                dump_bucket=dump_bucket, coaster=coaster)

_CACHE = {}


def get(kind, color=None):
    """Cached container geometry (tuning runs once per process)."""
    key = (kind, color)
    if key not in _CACHE:
        fn = BUILDERS[kind]
        _CACHE[key] = fn(color) if (color and kind.startswith("bottle")) else fn()
    return _CACHE[key]


def summary(kind):
    c = get(kind)
    m, com, inertia = c.mass_properties()
    out = dict(kind=kind, capacity_mL=c.capacity*1e6, glass_volume_mL=c.glass_volume*1e6, mass_kg=m,
               com_z_mm=com[2]*1e3, inertia=inertia.tolist(), rim_z_mm=c.rim_z*1e3, mouth_d_mm=2*c.mouth_r*1e3,
               lines_mm={k: v*1e3 for k, v in c.lines.items()}, rigid_pieces=len(c.rigid),
               convex_hulls=sum(len(rigid_hull_points(p)) for p in c.rigid), notes=c.notes)
    if c.lines:
        out["line_volumes_mL"] = {k: c.volume_below(v)*1e6 for k, v in c.lines.items()}
    return out

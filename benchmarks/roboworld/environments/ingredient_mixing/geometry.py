"""Shapes: bodies of revolution as (r, z) profiles, their volumes, signed-distance tables and seeding.

A vessel is described once, by its inner profile (the surface the food touches) from the axis to the rim, and its wall
thickness. From that one description come: the rigid collider pieces (scene.py), the signed-distance table the
continuum collides with (thickened to `shell` so a wall thinner than a grid cell cannot leak; the inner surface is the
true one), the render mesh, the interior volume and the level for a given fill.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------------------------- profiles
def bowl_profile(rim_r=0.149, depth=0.114, base_r=0.040, n=48):
    """Inner profile of a mixing bowl: flat base of radius base_r, then an arc of a sphere centred on the axis that
    passes through the base edge and the rim. Points (r, z), z = 0 at the inside bottom."""
    zc = (rim_r**2 + depth**2 - base_r**2)/(2.0*depth)
    R = math.hypot(base_r, zc)
    z = np.linspace(0.0, depth, n)
    r = np.sqrt(np.maximum(R*R - (z - zc)**2, 0.0))
    return np.vstack([[0.0, 0.0], np.column_stack([r, z])])


def cylinder_profile(radius, depth, corner=0.0, n=8):
    """Inner profile of a straight-sided vessel with an optional rounded bottom corner."""
    pts = [[0.0, 0.0]]
    if corner > 0:
        for a in np.linspace(-math.pi/2, 0.0, n):
            pts.append([radius - corner + corner*math.cos(a), corner + corner*math.sin(a)])
    else:
        pts.append([radius, 0.0])
    pts.append([radius, depth])
    return np.array(pts)


def tapered_profile(r_bottom, r_top, depth, corner=0.004, n=6):
    """Inner profile of a tapered vessel (pitcher, utensil pot)."""
    pts = [[0.0, 0.0]]
    for a in np.linspace(-math.pi/2, 0.0, n)[:-1]:
        pts.append([r_bottom - corner + corner*math.cos(a), corner + corner*math.sin(a)])
    for z in np.linspace(corner, depth, 6):
        pts.append([r_bottom + (r_top - r_bottom)*(z/depth), z])
    return np.array(pts)


def radius_at(profile, z):
    """Inner radius at height z (profile monotone in z above the base)."""
    return float(np.interp(z, profile[1:, 1], profile[1:, 0]))


def volume_to(profile, z):
    """Interior volume below height z."""
    zs = np.linspace(profile[1, 1], min(z, profile[-1, 1]), 400)
    if len(zs) < 2 or zs[-1] <= zs[0]:
        return 0.0
    r = np.interp(zs, profile[1:, 1], profile[1:, 0])
    return float(np.trapezoid(math.pi*r*r, zs))


def level_for(profile, volume):
    """Fill height for a volume."""
    lo, hi = profile[1, 1], profile[-1, 1]
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if volume_to(profile, mid) < volume:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


def shell_polygon(profile, shell, rim_extra=0.0):
    """Closed (r, z) polygon of the solid the continuum sees: everything within `shell` outside the inner profile
    (mitred at corners so the thickness holds there)."""
    p = np.asarray(profile, float)
    seg = np.diff(p, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
    sn = np.column_stack([seg[:, 1], -seg[:, 0]])             # outward normal of each segment
    vn = np.vstack([sn[:1], sn[:-1] + sn[1:], sn[-1:]])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    cosang = np.concatenate([[1.0], np.einsum("ij,ij->i", vn[1:-1], sn[:-1]), [1.0]])
    outer = p + (shell/np.maximum(cosang, 0.5))[:, None]*vn
    outer[0] = [0.0, p[0, 1] - shell]
    outer[:, 0] = np.maximum(outer[:, 0], 0.0)
    top = p[-1, 1] + rim_extra
    outer[-1] = [p[-1, 0] + shell, top]
    inner = p.copy()
    inner[-1, 1] = top
    return np.vstack([inner, outer[::-1]])


def wall_polygon(profile, wall):
    """Closed polygon of the true wall (inner profile offset outward by the wall thickness)."""
    return shell_polygon(profile, wall)


def rod_polygon(radius, z0, z1, tip_round=0.0, n=8):
    """A solid rod on the axis from z0 (lower end) to z1."""
    pts = [[0.0, z0]]
    if tip_round > 0:
        for a in np.linspace(-math.pi/2, 0.0, n):
            pts.append([radius - tip_round + tip_round*math.cos(a), z0 + tip_round + tip_round*math.sin(a)])
    else:
        pts.append([radius, z0])
    pts += [[radius, z1], [0.0, z1]]
    return np.array(pts)


# ---------------------------------------------------------------------------------------------- signed distance
def _segment_distance(px, pz, a, b):
    ab = b - a
    t = np.clip(((px - a[0])*ab[0] + (pz - a[1])*ab[1])/max(float(ab @ ab), 1e-18), 0.0, 1.0)
    return np.hypot(px - (a[0] + t*ab[0]), pz - (a[1] + t*ab[1]))


def _inside(px, pz, poly):
    inside = np.zeros(px.shape, bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cond = (a[1] > pz) != (b[1] > pz)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a[0] + (pz - a[1])*(b[0] - a[0])/(b[1] - a[1])
        inside ^= cond & (px < xi)
    return inside


def revolved_sdf(polygons, h=0.0005, pad=0.012):
    """Signed distance (negative inside the solid) of the union of closed (r, z) polygons, sampled on an (r, z) table
    with spacing h. Returns dict(sdf (nr, nz) float32, h, z0, r_max)."""
    polygons = [np.asarray(p, float) for p in polygons]
    allp = np.vstack(polygons)
    r_max = allp[:, 0].max() + pad
    z0, z1 = allp[:, 1].min() - pad, allp[:, 1].max() + pad
    nr, nz = int(math.ceil(r_max/h)) + 1, int(math.ceil((z1 - z0)/h)) + 1
    R, Z = np.meshgrid(np.arange(nr)*h, z0 + np.arange(nz)*h, indexing="ij")
    out = np.full(R.shape, np.inf)
    for poly in polygons:
        d = np.full(R.shape, np.inf)
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if a[0] <= 1e-12 and b[0] <= 1e-12:
                continue                                    # the edge on the axis is not a surface
            d = np.minimum(d, _segment_distance(R, Z, a, b))
        d = np.where(_inside(R, Z, poly), -d, d)
        out = np.minimum(out, d)
    return dict(sdf=out.astype(np.float32), h=float(h), z0=float(z0), r_max=float(r_max))


def sdf_value(table, r, z):
    """Host-side bilinear lookup (the kernel's sdf_lookup)."""
    s, h = table["sdf"], table["h"]
    fr = np.clip(np.asarray(r, float)/h, 0, s.shape[0] - 1.001)
    fz = np.clip((np.asarray(z, float) - table["z0"])/h, 0, s.shape[1] - 1.001)
    i, j = fr.astype(int), fz.astype(int)
    a, b = fr - i, fz - j
    return (1 - a)*((1 - b)*s[i, j] + b*s[i, j + 1]) + a*((1 - b)*s[i + 1, j] + b*s[i + 1, j + 1])


# ---------------------------------------------------------------------------------------------- seeding
def seed_revolved(inside, spacing, r_max, z_lo, z_hi, jitter=0.1, seed=0):
    """Points on a cubic lattice of the given spacing wherever inside(r, z) is true (vessel frame)."""
    n = int(math.ceil(2*r_max/spacing))
    ax = (np.arange(n) + 0.5)*spacing - n*spacing/2.0
    az = z_lo + (np.arange(max(1, int(round((z_hi - z_lo)/spacing)))) + 0.5)*spacing
    gx, gy, gz = np.meshgrid(ax, ax, az, indexing="ij")
    p = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    keep = inside(np.hypot(p[:, 0], p[:, 1]), p[:, 2], p)
    p = p[keep]
    if jitter:
        p = p + np.random.default_rng(seed).uniform(-jitter, jitter, p.shape)*spacing
    return p


def revolve_mesh(profile_rz, segments=64, close_top=False):
    """Triangle mesh of a closed (r, z) polygon revolved about z: (vertices, faces)."""
    p = np.asarray(profile_rz, float)
    n = len(p)
    ang = np.linspace(0, 2*math.pi, segments, endpoint=False)
    v = np.array([[r*math.cos(a), r*math.sin(a), z] for a in ang for r, z in p])
    f = []
    for s in range(segments):
        s1 = (s + 1) % segments
        for i in range(n):
            j = (i + 1) % n
            if p[i, 0] <= 1e-9 and p[j, 0] <= 1e-9:
                continue
            a, b, c, d = s*n + i, s*n + j, s1*n + j, s1*n + i
            if p[i, 0] > 1e-9:
                f.append([a, d, b])
            if p[j, 0] > 1e-9:
                f.append([b, d, c])
    return v, np.array(f, int)

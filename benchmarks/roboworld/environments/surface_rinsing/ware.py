"""Ware, produce and racks: one geometry per item for colliders, mass, water charts, cavities and rendering.

Axisymmetric items (glasses, plates, bowls, pan-like vessels, colander, can) are solids of revolution of one (r, z)
polygon of their material, wine_pouring.geometry's convention (counter-clockwise, z = 0 at the standing plane). From that
polygon, consistently:
  * colliders: convex hulls of authored (r, z) slabs revolved in sectors (MuJoCo convex meshes); a sector hull fills
    the chord of a concave inner face by r (1 - cos(pi/n)) (0.6 mm for a glass at 20 sectors; disclosed);
  * mass: the item's catalogue mass spread over the hulls (density = mass / hull volume);
  * water chart: the polygon traversed clockwise (material on the right) from the inner axis point to the outer one,
    revolved (surfaces.add_revolved); sharp convex profile corners (> 60 deg) are lips with the authored edge radius;
    cells carry their profile segment so the evaluator can read inside, rim, outside and foot separately;
  * cavity: the liquid space polygon (closed along the mouth plane), sampled with volume weights for level tables in
    any orientation, its rim circle for overflow, and a frustum region for the GPU's "inside the cavity" tests;
  * render mesh: the same polygon revolved (watertight).
Produce: ellipsoids. Racks and boards: boxes and cylinders.
"""

import math
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import ConvexHull

from classroom_arrangement.furniture import Part
from wine_pouring import geometry as G
from . import params as P

I = P.ITEMS
RHO = P.DENSITY


@dataclass
class Ware:
    kind: str
    mass: float
    material: str
    hulls: list = field(default_factory=list)          # [(points (n, 3) body frame)]
    prims: list = field(default_factory=list)          # [Part] in the body frame
    charts: list = field(default_factory=list)         # chart specs (surfaces.SurfaceSet calls)
    cavity: dict = None
    solid: np.ndarray = None                           # (r, z) CCW polygon of the material (revolved items)
    samples: tuple = None                              # (points (n, 3), volumes (n,)) of the material
    friction: float = 0.35
    regions: dict = field(default_factory=dict)        # chart name -> {region: [profile segment indices]}
    notes: str = ""


# ---------------------------------------------------------------------------------------------- helpers
def sector_hulls(slab, sectors, z_shift=0.0):
    """Convex pieces of a convex (r, z) slab revolved in `sectors` sectors (one piece when it touches the axis)."""
    slab = np.asarray(slab, float)
    if slab[:, 0].min() < 1e-9:
        th = 2*math.pi*np.arange(max(sectors, 24))/max(sectors, 24)
        pts = [(r*math.cos(t), r*math.sin(t), z + z_shift) for r, z in slab for t in th]
        return [np.unique(np.round(np.array(pts), 9), axis=0)]
    out = []
    for k in range(sectors):
        a = np.linspace(2*math.pi*k/sectors, 2*math.pi*(k + 1)/sectors, 3)
        rmid = slab[:, 0].mean()
        pts = []
        for r, z in slab:
            angs = a if r >= rmid else a[[0, -1]]
            pts += [(r*math.cos(t), r*math.sin(t), z + z_shift) for t in angs]
        out.append(np.array(pts))
    return out


def hull_volume(pts):
    return float(ConvexHull(pts).volume)


def revolve_samples(poly, dr, sectors_min=6):
    """Volume samples of a revolved (r, z) polygon: points (n, 3) and their volumes, total = revolved volume."""
    poly = G.ccw(np.asarray(poly, float))
    r0, r1 = 0.0, poly[:, 0].max()
    z0, z1 = poly[:, 1].min(), poly[:, 1].max()
    rr = np.arange(r0 + dr/2, r1, dr)
    zz = np.arange(z0 + dr/2, z1, dr)
    R, Zg = np.meshgrid(rr, zz, indexing="ij")
    cand = np.stack([R.ravel(), Zg.ravel()], 1)
    inside = G.point_in_polygon(cand, poly)
    pts, vols = [], []
    for r, z in cand[inside]:
        n = max(sectors_min, int(round(2*math.pi*r/dr)))
        th = 2*math.pi*(np.arange(n) + 0.5)/n
        v = 2*math.pi*r*dr*dr/n
        pts.append(np.stack([r*np.cos(th), r*np.sin(th), np.full(n, z)], 1))
        vols.append(np.full(n, v))
    pts, vols = np.concatenate(pts), np.concatenate(vols)
    exact = G.revolved_volume(poly)
    return pts, vols*exact/vols.sum()


def chart_profile(solid_ccw):
    """Clockwise traversal (material on the right) from the inner axis point to the outer axis point, and the lips."""
    poly = [tuple(map(float, p)) for p in G.ccw(np.asarray(solid_ccw, float))]
    axis = [k for k, (r, z) in enumerate(poly) if r < 1e-9]
    if len(axis) != 2:
        raise ValueError("a closed revolved surface needs exactly two axis points")
    # CCW polygon: (0, z_low) ... outer ... (0, z_high); reversed and rotated it starts at the higher axis point
    cw = poly[::-1]
    start = max((k for k, (r, z) in enumerate(cw) if r < 1e-9), key=lambda k: cw[k][1])
    prof = cw[start:] + cw[:start]
    if prof[-1][0] > 1e-9:
        # the other axis point is in the middle: cut there
        k2 = [k for k, (r, z) in enumerate(prof) if r < 1e-9 and k > 0][0]
        prof = prof[:k2 + 1]
    return prof


def lips_of(prof, min_turn_deg=60.0, radius=0.001):
    """Profile vertices where a clockwise (material-right) profile turns right by more than min_turn_deg: convex
    edges where a film can separate (Duez)."""
    out = []
    p = np.asarray(prof, float)
    for k in range(1, len(p) - 1):
        a, b = p[k] - p[k - 1], p[k + 1] - p[k]
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            continue
        cr = a[0]*b[1] - a[1]*b[0]
        ang = math.degrees(math.atan2(abs(cr), float(a @ b)))
        if cr < 0 and ang > min_turn_deg:
            out.append((k, radius))
    return out


def _segments_between(prof, z_or_r):
    return [k for k in range(len(prof) - 1) if z_or_r(prof[k], prof[k + 1])]


def revolved_ware(kind, solid, interior, rim_z, mouth_r, mass, material, slabs, sectors, cell, regions, *,
                  lip_r=0.001, friction=0.35, sample_dr=None, notes="", cavity_floor=None):
    hulls = []
    for slab in slabs:
        hulls += sector_hulls(slab, sectors)
    w = Ware(kind, mass, material, hulls=hulls, solid=G.ccw(np.asarray(solid, float)), friction=friction, notes=notes)
    prof = chart_profile(solid)
    if regions is None:
        # a vessel: its rim is the flat top between the two highest profile vertices (inner and outer rim corners);
        # inside = every segment before the inner corner, rim = between the corners, outside = after (base included),
        # foot = the outside segments within 2.5 mm of the standing plane
        top = max(z for r, z in prof)
        k1, k2 = [k for k, (r, z) in enumerate(prof) if abs(z - top) < 1e-7][:2]
        lips = [(k1, lip_r), (k2, lip_r)]
        n = len(prof) - 1
        regions = dict(inner=list(range(0, k1)), rim=list(range(k1, k2)), outer=list(range(k2, n)),
                       foot=[k for k in range(k2, n) if min(prof[k][1], prof[k + 1][1]) < 0.0025])
    else:
        lips = [(k, lip_r) for k in regions.pop("_lips")]
    w.charts.append(dict(type="revolved", name=kind, profile=prof, cell=cell, material=material, lips=lips))
    w.regions[kind] = regions
    # volume samples: ~1500 in the material (buoyancy), ~4000 in the cavity (level tables to ~1/4000 of the capacity)
    dr = sample_dr or float(np.clip((G.revolved_volume(w.solid)/1500.0)**(1/3), 0.0015, 0.02))
    w.samples = revolve_samples(w.solid, dr)
    if interior is not None:
        inner = G.ccw(np.asarray(interior, float))
        pts, vols = revolve_samples(inner, float(np.clip((G.revolved_volume(inner)/4000.0)**(1/3), 0.002, 0.02)))
        th = np.linspace(0, 2*math.pi, 64, endpoint=False)
        rim = np.stack([mouth_r*np.cos(th), mouth_r*np.sin(th), np.full(64, rim_z)], 1)
        z0 = float(inner[:, 1].min())
        r0 = float(np.interp(z0 + 1e-4, *_r_of_z(inner))) if cavity_floor is None else cavity_floor
        w.cavity = dict(polygon=inner, rim_z=float(rim_z), mouth_r=float(mouth_r), samples=pts, volumes=vols,
                        capacity=float(G.revolved_volume(inner)), rim=rim, region=(z0, float(rim_z), r0, float(mouth_r)))
    return w


def _r_of_z(inner):
    """(z, r) arrays of the outer boundary of a cavity polygon (max radius per z), for np.interp."""
    p = inner[inner[:, 0] > 1e-9]
    order = np.argsort(p[:, 1])
    return p[order, 1], p[order, 0]


def _cup_polys(profile_out, base_t, wall):
    """Outer wall profile [(r, z)] from the base edge to the rim -> (solid CCW, interior CCW, rim_z, mouth_r) as
    wine_pouring.geometry._cup builds them (inner = outer offset inward by `wall`)."""
    c = G._cup("tmp", profile_out, base_t, wall, 1000.0, "x")
    return c.solid, c.interior, c.rim_z, c.mouth_r


def _glass(kind, d):
    H, rt, rb, wall, base = d["H"], d["r_top"], d["r_bot"], d["wall"], d["base"]
    prof = [(rb + (rt - rb)*z/H, z) for z in np.linspace(0., H, 10)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, base, wall)
    rin = lambda z: rb + (rt - rb)*z/H - wall
    rout = lambda z: rb + (rt - rb)*z/H
    zs = np.linspace(base, H, 5)
    slabs = [[(0., 0.), (rb - 0.002, 0.), (rout(base), base), (0., base)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(rin(z0), z0), (rout(z0), z0), (rout(z1), z1), (rin(z1), z1)])
    return revolved_ware(kind, solid, interior, rim_z, mouth_r, d["mass"], "glass", slabs, 20, 0.002, None,
                         lip_r=0.0012, friction=P.FRICTION["glass_steel"],
                         notes=f"{kind}: H {H*1e3:.0f} mm, rim {2*rt*1e3:.0f} mm, wall {wall*1e3:.1f} mm, base {base*1e3:.0f} mm")


def rocks_glass():
    return _glass("rocks_glass", I["rocks_glass"])


def mixing_glass():
    return _glass("mixing_glass", I["mixing_glass"])


def plate():
    """9 in vitrified china plate: 7 in flat well, marli rising to a rounded rim at 22 mm, 4.5 mm body, foot ring."""
    d = I["plate"]
    R, t, fr, fh, H = d["R"], d["t"], d["foot_r"], d["foot_h"], d["H"]
    zw = fh + t                         # top of the well
    rw = 0.075                          # the well's flat ends here; the marli rises to the rim
    solid = [(0., fh), (fr - 0.003, fh), (fr - 0.003, 0.0), (fr + 0.003, 0.0), (fr + 0.006, fh + 0.001),
             (R - 0.0015, H - t + 0.0005), (R, H - 0.003), (R, H - 0.0012), (R - 0.0015, H), (rw, zw), (0., zw)]
    slabs = [[(0., fh), (rw, fh), (rw, zw), (0., zw)],
             [(fr - 0.003, 0.0), (fr + 0.003, 0.0), (fr + 0.006, fh + 0.001), (fr - 0.003, fh)]]
    # the marli in four conical pieces
    zb = lambda r: fh + 0.001 + (H - t + 0.0005 - fh - 0.001)*(r - fr - 0.006)/(R - 0.0015 - fr - 0.006)
    zt = lambda r: zw + (H - zw)*(r - rw)/(R - 0.0015 - rw)
    rs = np.linspace(rw, R - 0.0015, 5)
    for r0, r1 in zip(rs[:-1], rs[1:]):
        slabs.append([(r0, min(zb(max(r0, fr + 0.006)), zt(r0) - 0.003)), (r1, zb(r1)), (r1, zt(r1)), (r0, zt(r0))])
    slabs.append([(R - 0.0015, H - t + 0.0005), (R, H - 0.003), (R, H - 0.0012), (R - 0.0015, H)])
    # clockwise profile: 0 centre of the face, 1 well edge, 2 top of the rim, 3-4 rim edge, 5 under the rim, 6 back of
    # the marli, 7-8 foot ring, 9-10 underside of the centre: the rim edge (3, 4) and the foot ring's corners (7, 8) are
    # the convex lips where a film may separate
    regions = dict(face=[0, 1], rim=[2, 3, 4], back=[5, 6], foot=[7, 8, 9], _lips=[3, 4, 7, 8])
    return revolved_ware("plate", solid, None, H, R, d["mass"], "glaze", slabs, 24, 0.002, regions, lip_r=0.0015,
                         friction=P.FRICTION["china_steel"],
                         notes="9 in plate: rim 229 mm, 7 in well, foot ring 150 mm, 4.5 mm body, 22 mm tall, 0.47 kg")


def bowl():
    d = I["bowl"]
    R, H, fr, wall = d["R"], d["H"], d["foot_r"], d["wall"]
    prof = [(fr + (R - fr)*math.sin(0.5*math.pi*s), 0.006 + (H - 0.006)*(1 - math.cos(0.5*math.pi*s))**0.8) for s in np.linspace(0, 1, 10)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, 0.006 + wall, wall)
    ro = G.profile_function(prof)
    zs = np.linspace(0.006 + wall, H, 5)
    slabs = [[(0., 0.), (fr, 0.), (fr + 0.002, 0.006 + wall), (0., 0.006 + wall)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(ro(z0) - wall*1.2, z0), (ro(z0), z0), (ro(z1), z1), (ro(z1) - wall*1.2, z1)])
    regions = None
    return revolved_ware("bowl", solid, interior, rim_z, mouth_r, d["mass"], "glaze", slabs, 24, 0.002, regions,
                         lip_r=0.0015, friction=P.FRICTION["china_steel"], notes="6 in china soup bowl (estimate)")


def steel_bowl():
    """Clean stainless mixing bowl on the work table: 300 mm rim, 110 mm deep, 0.8 mm wall, 0.4 kg."""
    R, H, rb, wall = 0.150, 0.110, 0.080, 0.0008
    prof = [(rb + (R - rb)*math.sin(0.5*math.pi*s), H*(1 - math.cos(0.5*math.pi*s))**0.9) for s in np.linspace(0, 1, 10)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, 0.0012, wall)
    ro = G.profile_function(prof)
    zs = np.linspace(0.0012, H, 6)
    slabs = [[(0., 0.), (rb, 0.), (rb + 0.002, 0.0012), (0., 0.0012)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(ro(z0) - 0.002, z0), (ro(z0), z0), (ro(z1), z1), (ro(z1) - 0.002, z1)])
    regions = None
    return revolved_ware("steel_bowl", solid, interior, rim_z, mouth_r, 0.40, "steel", slabs, 24, 0.004, regions,
                         lip_r=0.0008, friction=0.30, notes="authored 300 mm stainless bowl")


def colander():
    """5 qt stainless colander: 280 mm rim, 190 mm base, 125 mm deep overall, 0.8 mm wall perforated with 4 mm holes on a
    9 mm pitch (bottom and lower wall), standing on a 25 mm foot ring (r 70 mm), two loop handles. The bowl is built
    from z = 0 and lifted onto the ring, so its chart, cavity and colliders share one geometry."""
    d = I["colander"]
    R, rb, H, wall = d["R"], d["r_bot"], d["H"], d["wall"]
    foot = 0.025
    hb = H - foot                                           # bowl height above the ring
    prof = [(rb + (R - rb)*(s_**0.8), hb*s_) for s_ in np.linspace(0, 1, 9)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, 0.0012, wall)
    solid = np.asarray(solid, float) + (0.0, foot)
    interior = np.asarray(interior, float) + (0.0, foot)
    rim_z += foot
    ro = lambda z: float(np.interp(z - foot, [p[1] for p in prof], [p[0] for p in prof]))
    zs = np.linspace(foot + 0.0012, H, 5)
    slabs = [[(0., foot), (rb - 0.002, foot), (ro(foot + 0.0012), foot + 0.0012), (0., foot + 0.0012)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(ro(z0) - 0.002, z0), (ro(z0), z0), (ro(z1), z1), (ro(z1) - 0.002, z1)])
    w = revolved_ware("colander", solid, interior, rim_z, mouth_r, d["mass"], "steel", slabs, 24, 0.004, None,
                      lip_r=0.0008, friction=0.30, notes="5 qt perforated colander (estimate)")
    # foot ring: 16 straight segments of a 70 mm ring, 25 mm tall, 1.5 mm sheet (colliders, drawn)
    for k in range(16):
        a = 2*math.pi*(k + 0.5)/16
        q = (0., 0., math.sin(a/2), math.cos(a/2))
        w.prims.append(Part(f"foot_{k}", "box", (0.0015, 0.0140, foot/2), (0.070*math.cos(a), 0.070*math.sin(a), foot/2), q,
                            mass=0.0, material="steel"))
    # loop handles (drawn and colliding): two 60 mm x 15 mm loops at the rim, along +-x
    for sgn in (1, -1):
        w.prims.append(Part(f"handle_{'px' if sgn > 0 else 'nx'}", "box", (0.022, 0.030, 0.003),
                            (sgn*(R + 0.018), 0.0, H - 0.004), mass=0.0, material="steel"))
    # perforations: hole centres (bottom disk and the lower 60 % of the wall) with their outward normals
    holes = []
    pitch = d["pitch"]
    for x in np.arange(-rb, rb, pitch):
        for y in np.arange(-rb, rb, pitch):
            if x*x + y*y < (rb - 0.006)**2:
                holes.append((x, y, foot + 0.0006, 0., 0., -1.))
    for z in np.arange(foot + 0.012, foot + 0.6*hb, pitch):
        r = ro(z) - wall/2
        n = int(2*math.pi*r/pitch)
        for t in 2*math.pi*np.arange(n)/n:
            holes.append((r*math.cos(t), r*math.sin(t), z, math.cos(t), math.sin(t), 0.0))
    w.cavity["holes"] = np.array(holes)
    w.cavity["hole_area"] = math.pi*(d["hole_d"]/2)**2
    return w


def tomato():
    d = I["tomato"]
    w = Ware("tomato", d["mass"], "skin", friction=P.FRICTION["tomato_steel"])
    w.prims.append(Part("flesh", "sphere", (0.5*(d["r"] + d["c"]),), (0., 0., 0.), mass=d["mass"], material="tomato"))
    w.charts.append(dict(type="ellipsoid", name="tomato", centre=(0., 0., 0.), radii=(d["r"], d["r"], d["c"]), cell=0.002,
                         material="skin"))
    rr = 0.5*(d["r"] + d["c"])
    g = np.linspace(-rr, rr, 9)
    X, Y_, Z_ = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y_.ravel(), Z_.ravel()], 1)
    pts = pts[np.linalg.norm(pts, axis=1) < rr]
    vol = 4/3*math.pi*d["r"]**2*d["c"]                  # the ellipsoid's volume (the collider is its mean sphere)
    w.samples = (pts, np.full(len(pts), vol/len(pts)))
    w.notes = f"cherry tomato {2*d['r']*1e3:.1f} x {2*d['c']*1e3:.1f} mm, {d['mass']*1e3:.1f} g, density {d['mass']/vol:.0f} kg/m3"
    return w


def can():
    """Rubbermaid BRUTE 2632 32 gal: 526 mm body at the top (559 mm over the moulded lip), 448 mm base, 692 mm tall,
    3 mm LLDPE wall, 6 mm base; cavity 124 L (32.9 gal)."""
    d = I["can"]
    R, rb, H, wall = d["R"], d["r_bot"], d["H"], d["wall"]
    prof = [(rb + (R - rb)*z/H, z) for z in np.linspace(0.0, H, 8)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, 0.006, wall)
    ro = G.profile_function(prof)
    zs = np.linspace(0.006, H, 6)
    slabs = [[(0., 0.), (rb - 0.002, 0.), (ro(0.006), 0.006), (0., 0.006)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(ro(z0) - 0.006, z0), (ro(z0), z0), (ro(z1), z1), (ro(z1) - 0.006, z1)])
    # moulded rim lip (to the catalogue's 22 in overall diameter) and two base rails: colliders only
    slabs.append([(R - 0.002, H - 0.022), (0.2795, H - 0.012), (0.2795, H), (R - 0.002, H)])
    regions = None
    w = revolved_ware("can", solid, interior, rim_z, mouth_r, d["mass"], "polymer", slabs, 24, 0.006, regions,
                      lip_r=0.0015, friction=P.FRICTION["lldpe_tile"],
                      notes="BRUTE 2632 (search); collider wall 6 mm chord-filled (3 mm wall)")
    return w


def cutting_board():
    L_, W_, T_ = 0.610, 0.457, 0.0127
    w = Ware("cutting_board", 950.0*L_*W_*T_, "hdpe", friction=0.4)
    w.prims.append(Part("slab", "box", (L_/2, W_/2, T_/2), (0., 0., T_/2), mass=w.mass, material="hdpe"))
    w.charts.append(dict(type="rect", name="cutting_board", origin=(-L_/2, -W_/2, T_), u=(1, 0, 0), v=(0, 1, 0),
                         size=(L_, W_), cell=0.005, material="polymer", open_edges="u0 u1 v0 v1"))
    w.notes = "HDPE 18 x 24 x 1/2 in (derived mass 3.4 kg)"
    return w


def glass_rack():
    """Cambro 25-compartment glass rack (19.75 in square, 4 in base): open base grid, outer walls, 5 x 5 dividers."""
    S, Hh, t = 0.500, 0.100, 0.006
    w = Ware("glass_rack", 1.6, "rack_grey", friction=P.FRICTION["rack_steel"])
    parts = []
    # open floor grid carried 3 mm up on four corner feet (the feet alone touch the drainboard: 16 contacts, not 60)
    for k, (fx, fy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts.append(Part(f"foot{k}", "box", (0.015, 0.015, 0.0015), (fx*(S/2 - 0.02), fy*(S/2 - 0.02), 0.0015), mass=0.5,
                          material="rack_grey"))
    for k, x in enumerate(np.linspace(-S/2 + 0.01, S/2 - 0.01, 11)):
        parts.append(Part(f"bar_x{k}", "box", (0.004, S/2, 0.004), (x, 0., 0.007), mass=1.0, material="rack_grey"))
    for k, y in enumerate(np.linspace(-S/2 + 0.01, S/2 - 0.01, 6)):
        parts.append(Part(f"bar_y{k}", "box", (S/2, 0.004, 0.003), (0., y, 0.014), mass=1.0, material="rack_grey"))
    for name, c, h in (("wall_n", (0., S/2 - t/2, Hh/2), (S/2, t/2, Hh/2)), ("wall_s", (0., -S/2 + t/2, Hh/2), (S/2, t/2, Hh/2)),
                       ("wall_e", (S/2 - t/2, 0., Hh/2), (t/2, S/2, Hh/2)), ("wall_w", (-S/2 + t/2, 0., Hh/2), (t/2, S/2, Hh/2))):
        parts.append(Part(name, "box", h, c, mass=2.0, material="rack_grey"))
    for k in range(1, 5):
        s = -S/2 + k*S/5
        parts.append(Part(f"div_x{k}", "box", (0.002, S/2 - t, 0.045), (s, 0., 0.017 + 0.045), mass=0.8, material="rack_grey"))
        parts.append(Part(f"div_y{k}", "box", (S/2 - t, 0.002, 0.045), (0., s, 0.017 + 0.045), mass=0.8, material="rack_grey"))
    tot = sum(p.mass for p in parts)
    for p in parts:
        p.mass *= w.mass/tot
    w.prims = parts
    w.cells = [(-S/2 + (i + 0.5)*S/5, -S/2 + (j + 0.5)*S/5) for i in range(5) for j in range(5)]
    w.floor_z = 0.017
    w.notes = "authored after the Cambro 25S full-size rack (19.75 in square)"
    return w


def peg_rack():
    """Peg (plate) rack, 19.75 in square: open base grid and seven rows of 8 posts (60 mm) -> six plate slots."""
    S = 0.500
    w = Ware("peg_rack", 1.4, "rack_grey", friction=P.FRICTION["rack_steel"])
    parts = []
    # open floor grid carried 3 mm up on four corner feet (the feet alone touch the drainboard: 16 contacts, not 60)
    for k, (fx, fy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts.append(Part(f"foot{k}", "box", (0.015, 0.015, 0.0015), (fx*(S/2 - 0.02), fy*(S/2 - 0.02), 0.0015), mass=0.5,
                          material="rack_grey"))
    for k, x in enumerate(np.linspace(-S/2 + 0.01, S/2 - 0.01, 11)):
        parts.append(Part(f"bar_x{k}", "box", (0.004, S/2, 0.004), (x, 0., 0.007), mass=1.0, material="rack_grey"))
    # the cross bars lie in the plane of the long ones (a moulded base): stacked 6 mm higher, one would sit 0.1 mm from
    # where slot 4's plate rim lands, and the plate would drop past its edge and stand wedged between that edge, the
    # floor and the next posts at 119 N, 25 times its weight (P11 rest)
    for k, y in enumerate(np.linspace(-S/2 + 0.01, S/2 - 0.01, 6)):
        parts.append(Part(f"bar_y{k}", "box", (S/2, 0.004, 0.004), (0., y, 0.007), mass=1.0, material="rack_grey"))
    rows = np.linspace(-0.105, 0.105, 7)
    for i, y in enumerate(rows):
        for j, x in enumerate(np.linspace(-0.21, 0.21, 8)):
            parts.append(Part(f"post_{i}_{j}", "cylinder", (0.004, 0.033), (x, y, 0.011 + 0.033), mass=0.2, material="rack_grey"))
    for name, c, h in (("rail_e", (S/2 - 0.004, 0., 0.023), (0.004, S/2, 0.02)), ("rail_w", (-S/2 + 0.004, 0., 0.023), (0.004, S/2, 0.02))):
        parts.append(Part(name, "box", h, c, mass=2.0, material="rack_grey"))
    tot = sum(p.mass for p in parts)
    for p in parts:
        p.mass *= w.mass/tot
    w.prims = parts
    w.slots = [0.5*(a + b) for a, b in zip(rows[:-1], rows[1:])]
    w.floor_z = 0.011
    w.notes = "authored peg rack: six 35 mm plate slots between seven rows of posts"
    return w


def pan():
    """Vollrath Wear-Ever 10 in aluminium fry pan: 254 mm rim, 200 mm base, 51 mm deep, 3 mm wall, riveted 200 mm handle,
    1.05 kg (search; wall and handle section estimates)."""
    d = I["pan"]
    R, rb, H, wall = d["R"], d["r_bot"], d["H"], d["wall"]
    prof = [(rb + (R - rb)*s_**1.5, H*s_) for s_ in np.linspace(0, 1, 7)]
    solid, interior, rim_z, mouth_r = _cup_polys(prof, 0.004, wall)
    ro = G.profile_function(prof)
    zs = np.linspace(0.004, H, 4)
    slabs = [[(0., 0.), (rb - 0.002, 0.), (ro(0.004), 0.004), (0., 0.004)]]
    for z0, z1 in zip(zs[:-1], zs[1:]):
        slabs.append([(ro(z0) - 0.004, z0), (ro(z0), z0), (ro(z1), z1), (ro(z1) - 0.004, z1)])
    w = revolved_ware("pan", solid, interior, rim_z, mouth_r, d["mass"]*0.8, "aluminum", slabs, 24, 0.003, None,
                      lip_r=0.0015, friction=0.30, notes="10 in aluminium fry pan; the handle carries 20 % of the mass")
    # riveted handle: a 200 mm flat bar rising at 10 deg from the rim along +x (drawn and colliding, its own mass)
    ang = math.radians(10.0)
    q = (0., -math.sin(ang/2), 0., math.cos(ang/2))
    w.prims.append(Part("handle", "box", (0.100, 0.0125, 0.004), (R + 0.098*math.cos(ang), 0.0, H - 0.012 + 0.098*math.sin(ang)), q,
                        mass=0.0, material="steel"))
    w.handle_mass = 0.2*d["mass"]
    w.mass = d["mass"]*0.8
    return w


def sponge():
    """Scotch-Brite heavy-duty scrub sponge 111 x 66 x 17 mm: a 5 mm scour pad under 12 mm of cellulose, as two bodies
    on a compression slide (scene.sponge_item); its pores hold up to phi A H_c (1 - eps) of water (water.sponge)."""
    d = I["sponge"]
    w = Ware("sponge", d["mass"], "sponge", friction=0.5)
    # the colliders are the sponge's firm core: its outer 6 mm all round (params.SPONGE conform) is soft cellulose that
    # gives under a few newtons (E 50 kPa over an 8 cm2 end and 111 mm of length is 0.36 N/mm), so an end pushed into
    # the corner of a pan yields instead of jamming; the drawn block and the contact patch (water.sponge.patch) are the
    # full 111 x 66 mm
    hx, hy = d["L"]/2 - P.SPONGE["conform"], d["W"]/2 - P.SPONGE["conform"]
    w.prims = [Part("scour", "box", (hx, hy, d["scour"]/2), (0., 0., d["scour"]/2), mass=0.004, material="scour_green", mu=0.8),
               Part("cell_lo", "box", (hx, hy, 0.003), (0., 0., d["scour"] + 0.003), mass=0.002, material="sponge_yellow", mu=0.5)]
    w.top = [Part("cell_hi", "box", (hx, hy, 0.003), (0., 0., 0.0), mass=0.002, material="sponge_yellow", mu=0.5)]
    w.top_z = d["scour"] + 0.006 + 0.003
    w.notes = "two-body sponge: scour pad + lower cellulose / upper cellulose on a 0-10.5 mm compression slide"
    return w


BUILDERS = dict(rocks_glass=rocks_glass, mixing_glass=mixing_glass, plate=plate, bowl=bowl, steel_bowl=steel_bowl,
                colander=colander, tomato=tomato, can=can, cutting_board=cutting_board, glass_rack=glass_rack, peg_rack=peg_rack,
                pan=pan, sponge=sponge)
_CACHE = {}


def build(kind):
    if kind not in _CACHE:
        _CACHE[kind] = BUILDERS[kind]()
    return _CACHE[kind]


if __name__ == "__main__":
    for k in BUILDERS:
        w = build(k)
        cav = w.cavity
        hv = sum(hull_volume(h) for h in w.hulls) if w.hulls else 0.0
        print(f"{k:14s} mass {w.mass:.3f} kg hulls {len(w.hulls):3d} ({hv*1e6:7.1f} mL) prims {len(w.prims):3d} "
              f"cavity {cav['capacity']*1e6 if cav else 0:8.1f} mL charts {[c['name'] for c in w.charts]} "
              f"lips {[c.get('lips') for c in w.charts]}")

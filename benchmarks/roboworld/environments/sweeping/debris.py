"""Loose pieces: every piece is a rigid cluster of spheres with the mass and size of the real thing (params.py).

Each generator returns a list of solver.Piece in a local frame; `scatter` places them. The random generator is
passed in, so a task's seed fixes every piece.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import params as P
from .solver import Piece, DebrisClass


def classes():
    return [DebrisClass(n, *v, soft_k=P.SOFT.get(n, 0.0)) for n, v in P.DEBRIS_CLASSES.items()]


def _one(cls, centres, radii, mass, tag=""):
    return Piece(cls, np.asarray(centres, float).reshape(-1, 3), np.asarray(radii, float).reshape(-1), float(mass), tag=tag)


def grit(rng, grams):
    """Quartz grains 0.6 to 2.4 mm across, more small than large (number weighted toward the fine end)."""
    out, m = [], 0.0
    while m < grams*1e-3:
        d = 0.6e-3*(4.0**rng.random()**1.6)
        mass = P.GRIT_DENSITY*math.pi*d**3/6
        out.append(_one("grit", [(0, 0, 0)], [d/2], mass))
        m += mass
    return out


def rice(rng, grams):
    r = P.RICE
    n = max(1, int(round(grams*1e-3/r["mass"])))
    rad = (r["width"] + r["thick"])/4
    off = r["length"]/2 - rad
    return [_one("rice", [(-off, 0, 0), (0, 0, 0), (off, 0, 0)], [rad*0.95, rad, rad*0.95], r["mass"]) for _ in range(n)]


def cereal(rng, grams):
    c = P.CEREAL
    n = max(1, int(round(grams*1e-3/c["mass"])))
    R = (c["d"] - c["tube"])/2
    cen = [(R*math.cos(a), R*math.sin(a), 0.) for a in np.linspace(0, 2*math.pi, 8, endpoint=False)]
    return [_one("cereal", cen, [c["tube"]/2]*8, c["mass"]) for _ in range(n)]


def beads(rng, count):
    b = P.BEAD
    r = b["d"]/2
    off = b["length"]/2 - 0.0022
    # a barrel: a middle sphere of the bead's diameter and two end spheres that give it its flat-ended length
    return [_one("bead", [(-off, 0, 0), (0, 0, 0), (off, 0, 0)], [r*0.86, r, r*0.86], b["mass"]) for _ in range(int(count))]


def crumbs(rng, grams):
    out, m = [], 0.0
    while m < grams*1e-3:
        d = rng.uniform(1.0e-3, 3.0e-3)
        mass = P.CRUMB_DENSITY*math.pi*d**3/6
        out.append(_one("crumb", [(0, 0, 0)], [d/2], mass))
        m += mass
    return out


def wads(rng, count):
    """A crumpled letter sheet: a lumpy ball of seven spheres, 50 to 70 mm across."""
    out = []
    for _ in range(int(count)):
        d = rng.uniform(0.050, 0.070)
        r = d*0.30
        cen = [(0, 0, 0)] + [tuple(v*(d/2 - r)) for v in _dirs(rng, 6)]
        out.append(_one("wad", cen, [r*1.15] + [r*rng.uniform(0.85, 1.0) for _ in range(6)], P.WAD["mass"]))
    return out


def _dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v/np.linalg.norm(v, axis=1)[:, None]


def scraps(rng, count):
    """Paper snips 20 to 40 mm: a flat plate of small spheres 0.8 mm thick (thicker than paper; no bending, no air)."""
    out = []
    t = P.SCRAP["thick"]
    for _ in range(int(count)):
        a, b = rng.uniform(0.018, 0.030), rng.uniform(0.022, 0.040)
        nx, ny = max(2, int(round(a/0.006))), max(2, int(round(b/0.006)))
        cen = [(x, y, 0.) for x in np.linspace(-a/2, a/2, nx) for y in np.linspace(-b/2, b/2, ny)]
        out.append(_one("scrap", cen, [t/2]*len(cen), P.SCRAP["areal"]*a*b))
    return out


def pencils(rng, count):
    p = P.PENCIL
    n = 14
    cen = [(x, 0., 0.) for x in np.linspace(-p["length"]/2 + p["d"]/2, p["length"]/2 - p["d"]/2, n)]
    return [_one("pencil", cen, [p["d"]/2]*n, p["mass"]) for _ in range(int(count))]


def mug(rng):
    """An 11 oz mug broken into wall shards, base pieces, the handle as a ring fragment and chips. Each shard is a
    patch of the mug's cylindrical wall covered with spheres of the wall's thickness, so curved shards rock.
    Every piece is returned LYING: its thinnest direction upright, either side down (a wall shard on its hollow
    side or on its back, the handle on its side), since `scatter` only turns pieces about the vertical."""
    m = P.MUG
    R, H, w = m["d"]/2, m["h"], m["wall"]
    rs = w/2*1.15
    rho_area = m["density"]*w
    out = []
    # wall: cut into irregular patches by random angular and height breaks
    cuts_a = np.sort(np.concatenate([[0.], rng.uniform(0.3, 2*math.pi - 0.3, 5)]))
    for k, a0 in enumerate(cuts_a):
        a1 = cuts_a[k + 1] if k + 1 < len(cuts_a) else 2*math.pi
        hcut = np.sort(np.concatenate([[0., H], rng.uniform(0.2*H, 0.8*H, 1)]))
        for z0, z1 in zip(hcut[:-1], hcut[1:]):
            na = max(2, int(round((a1 - a0)*R/(1.6*rs))))
            nz = max(2, int(round((z1 - z0)/(1.6*rs))))
            cen = [(R*math.cos(a), R*math.sin(a), z) for a in np.linspace(a0, a1, na) for z in np.linspace(z0, z1, nz)]
            area = (a1 - a0)*R*(z1 - z0)
            out.append(_one("shard", np.array(cen) - np.mean(cen, 0), [rs]*len(cen), rho_area*area, tag="shard"))
            # for drawing: the grid of the wall patch and its outward normals
            out[-1].grid = (na, nz, np.array([(math.cos(a), math.sin(a), 0.) for a in np.linspace(a0, a1, na) for _ in range(nz)]))
    # base: three wedges
    for a0, a1 in ((0., 2.2), (2.2, 4.3), (4.3, 2*math.pi)):
        cen = [(r*math.cos(a), r*math.sin(a), 0.) for r in np.linspace(0.008, R - 0.004, 4) for a in np.linspace(a0, a1, max(2, int((a1 - a0)*r/(1.6*rs)) + 1))]
        area = 0.5*(a1 - a0)*R*R
        out.append(_one("shard", np.array(cen) - np.mean(cen, 0), [rs]*len(cen), m["density"]*area*w*1.3, tag="shard"))
    # handle: a C-shaped ring fragment
    cen = [(0.019*math.cos(a), 0., 0.026*math.sin(a)) for a in np.linspace(-1.9, 1.9, 9)]
    out.append(_one("shard", cen, [0.006]*len(cen), 0.028, tag="handle"))
    for p in out:
        _lay(p, rng)
    total = sum(p.mass for p in out)
    scale = (m["mass"] - 0.012)/total
    for p in out:
        p.mass *= scale
    # chips: 40 small fragments, 0.05 to 1 g
    for _ in range(40):
        g = 0.05e-3*(20.0**rng.random())
        d = (6*g/(math.pi*m["density"]))**(1/3)
        cen = [(0, 0, 0), (d*0.45, 0, 0), (0, d*0.4, 0)]
        out.append(_one("shard", cen, [d*0.42, d*0.32, d*0.3], g, tag="chip"))
    chips = sum(p.mass for p in out if p.tag == "chip")
    for p in out:
        if p.tag == "chip":
            p.mass *= 0.012/chips
    return out


def _lay(piece, rng):
    """Turn a piece's own frame so that it lies: the direction in which its spheres spread least becomes the
    vertical, with either side down."""
    c = np.asarray(piece.centres, float)
    c = c - c.mean(0)
    if len(c) < 3:
        return
    w, v = np.linalg.eigh(c.T @ c)
    up = v[:, 0]*(1.0 if rng.random() < 0.5 else -1.0)
    R = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0]
    piece.centres = R.apply(np.asarray(piece.centres, float))
    if getattr(piece, "grid", None) is not None:
        na, nz, normals = piece.grid
        piece.grid = (na, nz, R.apply(normals))


def height(piece):
    """How tall a piece is as it lies (scatter turns pieces about the vertical only)."""
    c, r = np.asarray(piece.centres, float), np.asarray(piece.radii, float)
    return float((c[:, 2] + r).max() - (c[:, 2] - r).min())


def lowest(piece, quat):
    """Height of the piece's origin above a plane it rests on in orientation quat (xyzw)."""
    c = Rotation.from_quat(quat).apply(piece.centres)
    return float((piece.radii - c[:, 2]).max())


def scatter(rng, pieces, region, z0=0.0, flat=True, avoid=None, spacing=1.15, tries=60, taken=None):
    """Place pieces on the surface z0 inside `region` without overlap: a disc (cx, cy, r), a box (x0, y0, x1, y1) or a
    callable returning one xy. Pieces lie flat (random yaw; scraps, rings, rice and pencils on their flat side; the
    pieces of a mug as debris.mug has laid them, each on its flat) and never overlap each other (bounding circles times `spacing`; a hash of 2 cm cells keeps this linear).
    `avoid`: callable xy, radius -> True where a piece may not lie (a leg, a wall). Returns the placed pieces.
    `taken`: a dict that this call reads and fills with what lies on the surface. Two calls on the same surface must
    share one, or the second lays its pieces through the first's.
    The neighbours looked at reach as far as the largest circle laid so far. A long thin piece (a
    pencil) takes the room of its own spheres, not of the circle around it."""
    cell = 0.02
    grid = {} if taken is None else taken

    def free(x, y, r):
        cx, cy = int(math.floor(x/cell)), int(math.floor(y/cell))
        k = int(math.ceil(spacing*(r + grid.get("largest", 0.0))/cell))
        for i in range(cx - k, cx + k + 1):
            for j in range(cy - k, cy + k + 1):
                for (px, py, pr) in grid.get((i, j), ()):
                    if (px - x)**2 + (py - y)**2 < (spacing*(pr + r))**2:
                        return False
        return True

    def draw():
        if callable(region):
            return region(rng)
        if len(region) == 3:
            a, rr = rng.uniform(0, 2*math.pi), region[2]*math.sqrt(rng.random())
            return region[0] + rr*math.cos(a), region[1] + rr*math.sin(a)
        return rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])

    order = sorted(range(len(pieces)), key=lambda k: -float(np.max(np.linalg.norm(pieces[k].centres, axis=1) + pieces[k].radii)))
    out = []
    for k in order:
        p = pieces[k]
        yaw = rng.uniform(0, 2*math.pi)
        q = Rotation.from_euler("z", yaw)
        if not flat:
            q = Rotation.from_quat(rng.normal(size=4)/1.0) if False else q
        c = q.apply(p.centres)
        rb = float(np.max(np.linalg.norm(c[:, :2], axis=1) + p.radii))
        placed = False
        for _ in range(tries):
            x, y = draw()
            if avoid is not None and avoid((x, y), rb):
                continue
            if free(x, y, rb):
                placed = True
                break
        if not placed:
            continue
        if float(np.sum(np.asarray(p.radii)**2)) < 0.15*rb*rb:         # long and thin: its spheres, one by one
            took = [(x + float(ci[0]), y + float(ci[1]), float(ri)) for ci, ri in zip(c, p.radii)]
        else:
            took = [(x, y, rb)]
        for (tx, ty, tr) in took:
            grid.setdefault((int(math.floor(tx/cell)), int(math.floor(ty/cell))), []).append((tx, ty, tr))
            grid["largest"] = max(grid.get("largest", 0.0), tr)
        p.pos = (x, y, z0 + lowest(p, q.as_quat()) + 2.0e-5)
        p.quat = tuple(q.as_quat())
        out.append(p)
    return out


def total_mass(pieces, cls=None):
    return float(sum(p.mass for p in pieces if cls is None or p.cls == cls))

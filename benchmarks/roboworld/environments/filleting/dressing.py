"""Visual-only dressing of the filleting cell: a chilled fish-and-butchery prep room.

Authored straight into the render USD. Nothing here has physics: no collider, mass or friction is defined in this
file. Every physical object that a robot, knife or workpiece can touch (bench top, board, pans, bowl, knife block,
scale, knives) is built by the physics world from filleting/layout.py; this module draws the SAME objects at the
same layout positions and sizes, plus the room around them. Any layout object that the physics frame supplies as
a posed body (frame["bodies"][name]) is skipped here and drawn from the body instead (render.py).

Sources and labels (every dimension carries one, collected in PROVENANCE and written to export.json):
  layout      filleting/layout.py (the physics layout; this file never changes it)
  params      filleting/params.py (knife, handle and trout dimensions)
  standard    published standard, remembered: EN 631 gastronorm sizes, 150 mm wall tile
  maker       manufacturer class dimensions, remembered
  estimate    typical size of the item class, estimated
  authored    chosen for this scene
  asset       measured from a pinned CC0 asset
Assets: pinned CC0 textures/models (Poly Haven, ambientCG) with SHA-256, from wiping/assets (sources.json),
label_application/visual_assets.json and filleting/visual_assets.json (this module; cache outside the repository).
Everything else is procedural geometry authored to the dimensions listed in PROVENANCE.

Physically supported: every item is placed on the surface it rests on (floor function, bench top, shelf, pan
interior) and the placements are recorded in Dresser.support so the export can list them; wall-mounted items sit
flush on the wall plane with a visible bracket or back plate.
"""
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import layout as L
from . import params as PR

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MANIFEST = ROOT/"visual_assets.json"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"gar"/"filleting_visual_assets"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# ------------------------------------------------------------------------------------------------ provenance
PROVENANCE = {}


def prov(name, dims, label, basis):
    PROVENANCE[name] = dict(dims=dims, label=label, basis=basis)


R = L.ROOM
X0, X1 = R["x_range"]
Y0, Y1 = R["y_range"]
H = R["ceiling"]
TILE_TOP = R["wall_tile_height"]
TZ = L.TABLE["top_z"]
prov("room", "5.0 x 4.0 x 2.8 m, glazed tile to 2.0 m", "layout", "filleting/layout.py ROOM")
prov("wall_tile", "150 x 150 mm white glazed", "standard", "common UK/EU 150 mm square wall tile (remembered)")
prov("quarry_floor", "Poly Haven terracotta_floor_tiles at its published 2.08 m texture span",
     "asset", "red quarry-type floor tile appearance; the room falls to a central gully (authored)")
DRAIN = dict(center=(0.30, -1.55), grate=0.20, half=(1.20, 0.70), depth=0.012)
prov("floor_fall", "12 mm fall over 0.7-1.2 m to the gully (about 1:60-1:100)", "estimate",
     "commercial kitchen floors fall 1:60 to 1:100 to a gully (remembered guidance)")


def floor_z(x, y):
    """Visual floor height (m): flat z=0 except a four-plane hopper falling to the gully (authored)."""
    dx = np.abs(np.asarray(x, float) - DRAIN["center"][0])/DRAIN["half"][0]
    dy = np.abs(np.asarray(y, float) - DRAIN["center"][1])/DRAIN["half"][1]
    return -DRAIN["depth"]*np.clip(1. - np.maximum(dx, dy), 0., 1.)


# ------------------------------------------------------------------------------------------------ assets
def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "GymAnythingFilleting/1.0"}), timeout=180) as r:
        return r.read()


NEW_ASSETS = {
    # name: (kind, files {relative: url}); pinned by `python -m filleting.dressing --pin`
    "terracotta_floor_tiles": ("texture", {
        f"terracotta_floor_tiles_{m}_2k.jpg":
            f"https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/terracotta_floor_tiles/terracotta_floor_tiles_{m}_2k.jpg"
        for m in ("diff", "nor_gl", "rough")}),
    "WetFloorSign_01": ("model", None),
}


def fetch(name, pin=False):
    """Local folder of a pinned new asset (filleting/visual_assets.json), downloaded and SHA-256 verified."""
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    folder = CACHE/name
    entry = man.get(name)
    if entry is None:
        if not pin:
            raise KeyError(f"{name} is not pinned in {MANIFEST}")
        kind, files = NEW_ASSETS[name]
        if files is None:                                     # Poly Haven model: 1k USD + its textures
            info = json.loads(_get("https://api.polyhaven.com/files/" + name))
            usd = info["usd"]["1k"]["usd"]
            files = {Path(usd["url"]).name: usd["url"]}
            files.update({rel: f["url"] for rel, f in usd.get("include", {}).items()})
        entry = dict(kind=kind, license="CC0 1.0 (https://polyhaven.com/license)",
                     source=f"https://polyhaven.com/a/{name}", files={})
        for rel, url in files.items():
            data = _get(url)
            (folder/rel).parent.mkdir(parents=True, exist_ok=True)
            (folder/rel).write_bytes(data)
            entry["files"][rel] = dict(url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        man[name] = entry
        MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    for rel, f in entry["files"].items():
        p = folder/rel
        if not p.exists():
            data = _get(f["url"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"visual asset hash mismatch: {p}")
    return folder


def wiping_texture(name):
    """{Diffuse, nor_gl, Rough: path} of a texture pinned in wiping/assets (sha256 checked)."""
    root = REPO/"wiping"/"assets"
    src = root/"sources.json"
    meta = json.loads(src.read_text()).get(name)
    if meta is None:
        meta = json.loads((root/f"{name}_sources.json").read_text())
    out = {}
    for key, f in meta["files"].items():
        p = root/f["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError(f"wiping asset hash mismatch: {p}")
        out[key] = p
    return out, meta


# ------------------------------------------------------------------------------------------------ colour
def lin(c):
    """sRGB (0..1 or 0..255) -> linear RGB tuple."""
    c = np.asarray(c, float)
    if c.max() > 1.5:
        c = c/255.
    return tuple(np.where(c <= .04045, c/12.92, ((c + .055)/1.055)**2.4))


def _font(size, bold=True):
    try:
        return ImageFont.truetype(FONT if bold else FONT_REG, size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------------------------------------------ geometry
def crease_normals(V, F, angle=32.):
    """Per-corner normals: average of incident face normals within `angle` of the corner's own face."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fu = fn/np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    cv = F.ravel()
    cf = np.repeat(np.arange(len(F)), 3)
    order = np.argsort(cv, kind="stable")
    cnt = np.bincount(cv, minlength=len(V))
    ptr = np.concatenate([[0], np.cumsum(cnt)])
    inc = cf[order]
    deg = cnt[cv]
    rep = np.repeat(np.arange(len(cv)), deg)
    offs = np.arange(deg.sum()) - np.repeat(np.cumsum(deg) - deg, deg)
    nb = inc[np.repeat(ptr[cv], deg) + offs]
    ok = (fu[nb]*fu[cf[rep]]).sum(1) > math.cos(math.radians(angle))
    out = np.zeros((len(cv), 3))
    np.add.at(out, rep[ok], fn[nb[ok]])
    bad = np.linalg.norm(out, axis=1) < 1e-30
    out[bad] = fu[cf[bad]]
    return out/np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def box_uv(V, F):
    """Per-corner planar UV (metres) by each face's dominant axis."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ax = np.argmax(np.abs(fn), axis=1)
    P = V[F]                                              # (M,3,3)
    uv = np.empty((len(F), 3, 2))
    for a in range(3):
        m = ax == a
        keep = [i for i in range(3) if i != a]
        uv[m] = P[m][:, :, keep]
    return uv.reshape(-1, 2)


def rrect(hx, hy, r, n=8):
    """Closed rounded-rectangle ring (4n points, CCW) centred at 0."""
    r = max(min(r, hx - 1e-5, hy - 1e-5), 1e-4)
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0.), (-hx + r, hy - r, 90.), (-hx + r, -hy + r, 180.), (hx - r, -hy + r, 270.)):
        for t in np.radians(np.linspace(a0, a0 + 90., n)):
            pts.append((cx + r*math.cos(t), cy + r*math.sin(t)))
    return np.asarray(pts)


def loft(rings, closed=True, cap0=False, cap1=False):
    """Triangulate consecutive rings (each (n,3), same n). Caps: fan to the ring centroid."""
    rings = [np.asarray(r, float) for r in rings]
    n = len(rings[0])
    V = np.concatenate(rings)
    F = []
    m = n if closed else n - 1
    for k in range(len(rings) - 1):
        a, b = k*n, (k + 1)*n
        for i in range(m):
            j = (i + 1) % n
            F += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    V = list(V)
    for cap, ring, flip in ((cap0, 0, True), (cap1, len(rings) - 1, False)):
        if cap:
            c = len(V)
            V.append(rings[ring].mean(0))
            base = ring*n
            for i in range(n):
                j = (i + 1) % n
                F.append((c, base + j, base + i) if flip else (c, base + i, base + j))
    return np.asarray(V), np.asarray(F, np.int64)


def lathe(profile, n=64):
    """Surface of revolution about z from a (r, z) polyline."""
    rings = []
    for r, z in profile:
        t = np.linspace(0, 2*math.pi, n, endpoint=False)
        rings.append(np.stack([r*np.cos(t), r*np.sin(t), np.full(n, z)], 1))
    return loft(rings)


def cylinder(r, h, n=48, z0=0.):
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    ring = lambda z: np.stack([r*np.cos(t), r*np.sin(t), np.full(n, z)], 1)
    return loft([ring(z0), ring(z0 + h)], cap0=True, cap1=True)


def box_mesh(size, center=(0, 0, 0)):
    hx, hy, hz = np.asarray(size, float)/2
    V = np.array([(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]) + np.asarray(center, float)
    F = np.array([(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
                  (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)])
    return V, F


def rbox_mesh(size, r, center=(0, 0, 0), n=4):
    """Box with rounded vertical edges and chamfered-round top/bottom edges (radius r)."""
    hx, hy, hz = np.asarray(size, float)/2
    r = min(r, hx*.45, hy*.45, hz*.45)
    rings = []
    for k in range(n + 1):
        a = math.pi/2*k/n
        rings.append((r*(1 - math.cos(a)), -hz + r*(1 - math.sin(a))))
    prof = rings + [(r*(1 - math.cos(a)), hz - r + r*math.sin(a)) for a in np.linspace(0, math.pi/2, n + 1)[::-1]]
    rr = []
    for inset, z in prof:
        ring = rrect(hx - inset, hy - inset, max(r - inset, 1e-4) if inset < r else 1e-4, 6)
        rr.append(np.column_stack([ring, np.full(len(ring), z)]))
    V, F = loft(rr, cap0=True, cap1=True)
    return V + np.asarray(center, float), F


def tube(points, radius, n=12, caps=True):
    """Tube along a polyline (radius scalar or per point) with parallel-transport frames."""
    P = np.asarray(points, float)
    rad = np.broadcast_to(np.asarray(radius, float), (len(P),))
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    ref = np.array([0., 0., 1.]) if abs(T[0, 2]) < .9 else np.array([1., 0., 0.])
    N = np.cross(T[0], ref)
    N /= np.linalg.norm(N)
    rings = []
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    for i in range(len(P)):
        N = N - T[i]*np.dot(N, T[i])
        N /= max(np.linalg.norm(N), 1e-12)
        B = np.cross(T[i], N)
        rings.append(P[i] + rad[i]*(np.outer(np.cos(t), N) + np.outer(np.sin(t), B)))
    return loft(rings, cap0=caps, cap1=caps)


def fan(poly, z):
    """Triangulated convex polygon (n,2) at height z, facing +z."""
    poly = np.asarray(poly, float)
    V = np.column_stack([np.vstack([poly.mean(0), poly]), np.full(len(poly) + 1, z)])
    F = [(0, 1 + i, 1 + (i + 1) % len(poly)) for i in range(len(poly))]
    return V, np.asarray(F)


def merge(*parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(np.asarray(V, float))
        Fs.append(np.asarray(F, np.int64) + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def rotz(V, yaw, about=(0, 0)):
    c, s = math.cos(yaw), math.sin(yaw)
    V = np.asarray(V, float).copy()
    x, y = V[:, 0] - about[0], V[:, 1] - about[1]
    V[:, 0], V[:, 1] = about[0] + c*x - s*y, about[1] + s*x + c*y
    return V


def rot_matrix(axis, ang):
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang)*K + (1 - math.cos(ang))*K @ K


# ------------------------------------------------------------------------------------------------ trout stand-in
TROUT = PR.v(PR.TROUT)
_D_PROF = [(0., .0), (.015, .26), (.05, .47), (.10, .66), (.18, .84), (.28, .96), (.38, 1.), (.50, .95),
           (.62, .77), (.72, .56), (.80, .40), (.86, .31), (.885, .30)]
_W_PROF = [(0., .0), (.015, .30), (.05, .62), (.12, .88), (.22, 1.), (.34, .98), (.48, .86), (.62, .66),
           (.74, .47), (.82, .35), (.885, .28)]
_U_END = .885


def _prof(table, u):
    t = np.asarray(table)
    return np.interp(u, t[:, 0], t[:, 1])


def trout_body(length=None, depth=None, width=None, nu=150, nv=64):
    """STAND-IN trout surface (visual only; NOT the anatomy generator): a lofted body in the anatomy
    workpiece frame (head at -x, back toward -y, belly toward +y, upper
    flank facing +z, lying on its side with its lowest point on z = 0). Dimensions: params.TROUT (31.5 cm,
    7.5 cm deep, 3.6 cm wide, derived+estimate); profile shape authored from generic salmonid proportions.
    Returns dict(V, F, UV, U, TH, a, b, zc) with UV = (along length 0..1 of the body, around 0..1 starting at
    the dorsal line, 0.25 upper-flank lateral line, 0.5 belly, 0.75 lower flank)."""
    Lf = length or TROUT["length"]
    Df = depth or TROUT["depth"]
    Wf = width or TROUT["width"]
    u = np.concatenate([[0.], np.linspace(.004, _U_END, nu - 1)])
    u = u**1.15/(_U_END**.15)
    a = .5*Df*_prof(_D_PROF, u)
    b = .5*Wf*_prof(_W_PROF, u)
    b = np.maximum(b, 1e-5)
    a = np.maximum(a, 1e-5)
    th = np.linspace(0, 2*math.pi, nv + 1)            # seam duplicated (V = 0 and V = 1 at the dorsal line)
    rings = []
    for k in range(len(u)):
        # dorsal (-y) slightly narrower than the belly (+y): the belly is rounder (authored)
        ca, sa = np.cos(th), np.sin(th)
        ay = np.where(ca > 0, a[k]*.94, a[k]*1.06)
        e = 2.4                                           # superellipse exponent: slightly boxy section
        y = -np.sign(ca)*np.abs(ca)**(2/e)*ay + .03*a[k]
        z = b[k] + np.sign(sa)*np.abs(sa)**(2/e)*b[k]
        x = u[k]*Lf - Lf/2
        rings.append(np.stack([np.full(nv + 1, x), y, z], 1))
    V, F = loft(rings, closed=False, cap1=True)
    # snout ring 0 collapses to a point (a, b ~ 0); the tail ring is capped (hidden under the caudal fin root)
    UU = np.concatenate([np.repeat(u/_U_END, nv + 1), [1.]])
    TT = np.concatenate([np.tile(th/(2*math.pi), len(u)), [0.5]])
    return dict(V=V, F=F, UV=np.stack([UU, TT], 1), u=u, a=a, b=b, L=Lf, nv=nv, th=th)


def fin_texture(path, seed=0):
    """Trout fin (STAND-IN, authored): translucent-looking grey-olive membrane with bony rays along v and a few
    dark spots; u runs along the fin base, v from the base (0) to the free edge (1)."""
    rng = np.random.default_rng(seed)
    w, h = 512, 256
    U, Vv = np.meshgrid((np.arange(w) + .5)/w, (np.arange(h) + .5)/h)
    base = np.array([168, 150, 132])/255.*(1 - .35*Vv[..., None]) + np.array([214, 200, 188])/255.*.35*Vv[..., None]
    rays = (np.abs(((U*16) % 1.) - .5) < .09).astype(float)*(1 - .4*Vv)
    col = base*(1 - .45*rays[..., None])
    img = Image.fromarray((np.clip(col, 0, 1)*255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.uniform(0, w), rng.uniform(0, h*.8)
        r = rng.uniform(2, 5)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(40, 40, 36))
    img.filter(ImageFilter.GaussianBlur(.7)).save(path)
    return path


def trout_fins(body):
    """Fin membranes (thin, drooping to rest on the support) with UVs (u along the base, v to the free edge), and
    the eye; same frame as trout_body. Returns ([(name, V, F, UV)], eye)."""
    Lf, u, a, b = body["L"], body["u"], body["a"], body["b"]
    X = lambda uu: uu*Lf - Lf/2
    A = lambda uu: float(np.interp(uu, u, a))
    B = lambda uu: float(np.interp(uu, u, b))
    fins = []

    def membrane(base, tip_offsets, droop_to=.0015):
        """base: (n,3) attachment points; tip_offsets: (n,2) (dx, dy) of the free edge; the free edge rests
        at z = droop_to (supported by the surface beneath), the base stays on the body."""
        base = np.asarray(base, float)
        rows = []
        for s in np.linspace(0, 1, 6):
            p = base.copy()
            p[:, 0] += s*tip_offsets[:, 0]
            p[:, 1] += s*tip_offsets[:, 1]
            p[:, 2] = base[:, 2]*(1 - s**1.3) + droop_to*s**1.3
            rows.append(p)
        V, F = loft(rows, closed=False)
        n = len(base)
        UV = np.array([(i/(n - 1), r/5.) for r in range(6) for i in range(n)])
        return V, F, UV

    def edge_pts(u0, u1, side, n=10):
        uu = np.linspace(u0, u1, n)
        return np.array([(X(t), side*(A(t)*(.94 if side < 0 else 1.06)) + .03*A(t), B(t)) for t in uu]), uu
    # dorsal fin (-y edge), adipose, anal (+y edge), caudal
    for name, u0, u1, side, hgt in (("dorsal", .40, .52, -1, .030), ("adipose", .70, .74, -1, .009),
                                    ("anal", .64, .72, 1, .024)):
        base, uu = edge_pts(u0, u1, side)
        prof = np.sin(np.linspace(0, math.pi, len(uu)))**.6*hgt*np.linspace(1., .55, len(uu))
        tips = np.stack([np.linspace(.4, 1.0, len(uu))*hgt*.8, side*prof], 1)
        fins.append((name,) + membrane(base, tips))
    # caudal: fan behind the peduncle, forked trailing edge
    ue = u[-1]
    xr = X(ue)
    n = 17
    span = np.linspace(-1, 1, n)
    base = np.stack([np.full(n, xr - .004), span*A(ue)*.95 + .03*A(ue), np.full(n, B(ue))], 1)
    fork = .055 - .018*(1 - np.abs(span))**1.5
    tips = np.stack([fork, span*.040], 1)
    fins.append(("caudal",) + membrane(base, tips))
    # pectoral and pelvic fins lying against the upper flank (paired fins; the lower pair is hidden)
    for name, u0, len_, wid in (("pectoral", .20, .040, .012), ("pelvic", .50, .030, .010)):
        x0 = X(u0)
        y0 = A(u0)*.55
        z0 = 2*B(u0)*.93
        s = np.linspace(0, 1, 6)
        rows = []
        for k in np.linspace(-1, 1, 5):
            rows.append(np.stack([x0 + s*len_, y0 + k*wid*.5*(1 - .5*s) + s*.006, z0 - s*.004], 1))
        V, F = loft(rows, closed=False)
        UV = np.array([(i/5., r/4.) for r in range(5) for i in range(6)])[:, ::-1]
        fins.append((name, V, F, UV))
    ue_eye = .075
    eye = dict(center=(X(ue_eye), -A(ue_eye)*.28, 2*B(ue_eye)*.92), r=.0058)
    return fins, eye


def trout_skin_texture(path, seed=0, w=2048, h=1024):
    """Procedural rainbow-trout skin (STAND-IN, authored): olive-green back with black spots, pink lateral
    band, silver flank and white belly; gill-cover line and dark head top. V: 0 dorsal, .25/.75 lateral
    lines, .5 belly (see trout_body)."""
    rng = np.random.default_rng(seed)
    U, V = np.meshgrid((np.arange(w) + .5)/w, (np.arange(h) + .5)/h)
    d = np.cos(2*math.pi*V)                                      # +1 dorsal, -1 belly
    back = np.array([62, 78, 64])/255.
    flank = np.array([178, 182, 176])/255.
    belly = np.array([236, 234, 228])/255.
    t1 = np.clip((d + .15)/.75, 0, 1)[..., None]
    t2 = np.clip((-d - .05)/.7, 0, 1)[..., None]
    col = flank*(1 - t1) + back*t1
    col = col*(1 - t2) + belly*t2
    # pink/red lateral band, strongest mid-body
    band = np.exp(-((np.minimum(np.abs(V - .25), np.abs(V - .75)))/.05)**2)
    along = np.clip((U - .12)/.12, 0, 1)*np.clip((.96 - U)/.2, 0, 1)
    pink = np.array([206, 112, 124])/255.
    k = (band*along*.78)[..., None]
    col = col*(1 - k) + pink*k
    # iridescent blue-green sheen on the upper flank
    sheen = np.exp(-((np.minimum(np.abs(V - .12), np.abs(V - .88)))/.06)**2)[..., None]*.18
    col = col*(1 - sheen) + np.array([70, 120, 118])/255.*sheen
    # head: darker top, gill cover edge (curved line at U ~ .185), mouth line
    head = (U < .19)[..., None]
    col = np.where(head, col*np.array([.92, .95, .95]), col)
    img = Image.fromarray((np.clip(col, 0, 1)*255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for side in (.25, .75):
        pts = [(int((.19 - .03*math.cos((vv - side)/.22*math.pi/2)**2)*w), int(vv*h))
               for vv in np.linspace(side - .2, side + .2, 40)]
        dr.line(pts, fill=(96, 92, 88), width=5)
        dr.line([(int(.005*w), int((side - .03)*h)), (int(.05*w), int((side + .005)*h))], fill=(70, 60, 58), width=4)
    # spots: dense on the back, the dorsal half of the flank and the tail; sparse below the lateral line
    n = 1500
    su = rng.uniform(.05, 1., n)
    sv = rng.uniform(0, 1, n)
    dd = np.cos(2*math.pi*sv)
    keep = rng.uniform(0, 1, n) < np.clip(.15 + .85*(dd + .2), 0, 1)*np.where(su > .75, 1., .8)
    for uu, vv in zip(su[keep], sv[keep]):
        r = rng.uniform(4, 9)
        x, y = uu*w, vv*h
        dr.ellipse([x - r*1.1, y - r*.9, x + r*1.1, y + r*.9], fill=(28, 30, 30))
    img = img.filter(ImageFilter.GaussianBlur(.8))
    arr = np.asarray(img).astype(np.float32)/255.
    arr *= (1 + .035*rng.standard_normal(arr.shape[:2]))[..., None]  # fine scale texture
    Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).save(path)
    return path


# ------------------------------------------------------------------------------------------------ knife geometry
def knife_geometry(kind="fillet"):
    """Blade and handle meshes in the BladeSpec frame (filleting/knife.py): origin at the heel on the cutting edge,
    x toward the tip (0..L), y through the thickness, z from the edge (0) to the spine (H). Cross-section from the
    physics blade: flat grind from the spine (params spine thickness, distal taper BladeSpec.spine_at) down to a
    0.25 mm shoulder (knife.EDGE_HALF_SHOULDER), then the params bevel angle per side to the edge. The blade height
    is constant along the length because the physics blade is (BladeSpec has no height profile); if BladeSpec
    gains `height_at(xi)`, it is used here. Returns dict(blade=(V,F), bevel=(V,F), handle=(V,F), spec)."""
    from .knife import BladeSpec, EDGE_HALF_SHOULDER
    spec = BladeSpec(kind)
    Lb, Hb = spec.L, spec.H
    e = EDGE_HALF_SHOULDER
    hb = e/math.tan(spec.bevel)                      # bevel face height (0.385 mm at 18 deg)
    height = (lambda xi: spec.height_at(xi)) if hasattr(spec, "height_at") else (lambda xi: Hb)

    def piece(x0, x1, n):
        """Blade between x0 and x1 (closed at both ends): flat grind faces, bevel faces, end caps."""
        flat_rings, bev_rings = [], []
        for x in np.linspace(x0, x1, n):
            xi = x/Lb
            s_ = spec.spine_at(xi)/2
            hh = height(xi)
            # flat faces: shoulder(+y) -> spine(+y) -> spine(-y) -> shoulder(-y)
            flat_rings.append([(x, e, hb), (x, s_, hh), (x, -s_, hh), (x, -e, hb)])
            bev_rings.append([(x, -e, hb), (x, 0., 0.), (x, e, hb)])
        Vf, Ff = loft(flat_rings, closed=False)
        Vb, Fb = loft(bev_rings, closed=False)
        caps = []
        for ring, flip in ((flat_rings[0] + bev_rings[0][1:2], False), (flat_rings[-1] + bev_rings[-1][1:2], True)):
            Fc = np.array([(0, 1, 2), (0, 2, 3), (0, 3, 4)])
            caps.append((np.array(ring), Fc if flip else Fc[:, ::-1]))
        return merge((Vf, Ff), *caps), (Vb, Fb)
    blade, bevel = piece(0., Lb, 60)
    Vb, Fb = bevel
    seg = [piece(k*spec.seg_len, (k + 1)*spec.seg_len, 14) for k in range(spec.segments)]
    # handle: Fibrox-class envelope params.HANDLE (140 x 32 x 19 mm), lofted rounded section, slight belly. Placed as
    # the physics places its handle box (scene.py): x in [-HANDLE_L, 0] (the blade starts at the handle's front
    # face) and centred on the spine line z = H ("the spine continues the handle's centre line")
    hd = PR.v(PR.HANDLE)
    hl, hh_, hw = hd["length"], hd["height"], hd["width"]
    rings = []
    zc = Hb
    for s in np.linspace(0, 1, 36):
        x = -s*hl
        hs = hh_*(.5*(.93 + .07*math.cos(2*math.pi*s)) if s < .92 else .5*(.93 + .07*math.cos(2*math.pi*s))*(1 - (s - .92)/.08*.35))
        ws = hw*.5*(.86 + .14*math.sin(math.pi*s))
        if s > .96:
            k = (s - .96)/.04
            hs *= (1 - .6*k**2)
            ws *= (1 - .6*k**2)
        ring = rrect(ws, hs, min(ws, hs)*.8, 6)
        rings.append(np.column_stack([np.full(len(ring), x), ring[:, 0], ring[:, 1] + zc]))
    handle = loft(rings, cap0=True, cap1=True)
    return dict(blade=blade, bevel=(Vb, Fb), handle=handle, spec=spec, zc=zc, segments=seg)


def tweezers_geometry():
    """Fish-bone tweezers, Victorinox 7.6120 class (120 mm long, 11 x 24 mm; read):
    two 11 mm wide, 1.2 mm stainless arms joined at the back (x=0), tips at x=0.12 opening to 24 mm over the arm
    outer faces. Origin at the joined end, x toward the tips, arms spread along z (closing along z), y across."""
    parts = []
    for sgn in (-1, 1):
        rows = []
        for s in np.linspace(0, 1, 30):
            x = .12*s
            gap = .0006 + (.024/2 - .0012)*math.sin(min(s/.55, 1)*math.pi/2) if s < .55 else .0006 + (.024/2 - .0012)
            gap = min(gap, .012 - .0012) if s < .96 else (.012 - .0012)*(1 - (s - .96)/.04*.55)
            w = .011*(1 - .25*s) if s < .9 else .011*.775
            zc = sgn*(gap + .0006)
            rows.append([(x, -w/2, zc - .0006), (x, w/2, zc - .0006), (x, w/2, zc + .0006), (x, -w/2, zc + .0006)])
        parts.append(loft(rows, cap0=True, cap1=True))
    return merge(*parts)


# ------------------------------------------------------------------------------------------------ textures
def make_textures(folder, preset):
    """Derived textures written into folder/textures; returns {name: path}."""
    tex = Path(folder)/"textures"
    tex.mkdir(parents=True, exist_ok=True)
    out = {}
    rng = np.random.default_rng(7)
    # brushed stainless roughness (streaks along u), 0.30 m span
    n = 1024
    streak = rng.standard_normal((1, n)).repeat(n, 0)
    streak = np.asarray(Image.fromarray(((streak - streak.min())/np.ptp(streak)*255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(.6)), np.float32)/255.
    grain = rng.standard_normal((n, n))*.08
    rough = np.clip(.24 + .10*(streak - .5) + grain*.3, .12, .45)
    p = tex/"steel_rough.png"
    Image.fromarray((rough*255).astype(np.uint8)).save(p)
    out["steel_rough"] = p
    # HDPE cutting board: base colour with fine knife scores (blue fish / red meat)
    for name, c in (("board_blue", (38, 92, 168)), ("board_red", (176, 38, 40)), ("board_yellow", (226, 190, 40)),
                    ("board_green", (40, 130, 70)), ("board_white", (232, 232, 226)), ("board_brown", (120, 80, 50))):
        img = Image.new("RGB", (1024, 768), c)
        dr = ImageDraw.Draw(img)
        for _ in range(260):
            x, y = rng.uniform(0, 1024), rng.uniform(0, 768)
            ang = rng.normal(0, .5) + (0 if rng.uniform() < .6 else math.pi/2)
            ln = rng.uniform(10, 90)
            sh = int(rng.uniform(5, 14))
            col = tuple(int(min(255, v + sh)) for v in c)
            dr.line([(x, y), (x + ln*math.cos(ang), y + ln*math.sin(ang))], fill=col, width=1)
        arr = np.asarray(img).astype(np.float32)
        arr *= (1 + .025*rng.standard_normal(arr.shape[:2]))[..., None]
        p = tex/f"{name}.png"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p)
        out[name] = p
    # hazard floor tape (yellow/black 45 deg), texture = 0.2 m along x 0.05 m
    img = Image.new("RGB", (512, 128), (232, 188, 22))
    dr = ImageDraw.Draw(img)
    for k in range(-2, 12):
        x = k*64
        dr.polygon([(x, 128), (x + 32, 128), (x + 32 + 128, 0), (x + 128, 0)], fill=(18, 18, 18))
    p = tex/"hazard_tape.png"
    img.save(p)
    out["hazard_tape"] = p
    # washable PVC-faced ceiling tiles, 600 mm grid, texture = 1.2 m
    img = Image.new("L", (1024, 1024), 236)
    arr = np.asarray(img, np.float32) + rng.standard_normal((1024, 1024))*1.5
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for k in (0, 512):
        dr.rectangle([k - 10, 0, k + 10, 1024], fill=250)
        dr.rectangle([0, k - 10, 1024, k + 10], fill=250)
        dr.line([(k - 11, 0), (k - 11, 1024)], fill=205, width=2)
        dr.line([(0, k - 11), (1024, k - 11)], fill=205, width=2)
    p = tex/"ceiling_tiles.png"
    img.convert("RGB").save(p)
    out["ceiling"] = p
    # cold-room insulated panel (white, vertical joints every 1.15 m) - texture = 1.2 m
    img = Image.new("RGB", (512, 512), (232, 234, 234))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, 5, 512], fill=(190, 194, 196))
    for y in range(0, 512, 32):
        dr.line([(0, y), (512, y)], fill=(226, 228, 228), width=1)
    p = tex/"coldroom_panel.png"
    img.save(p)
    out["coldroom"] = p
    # perforation pattern (alpha cut-out): 5 mm holes on a 9 mm staggered pitch, texture = 0.09 m
    s = 256
    img = Image.new("RGBA", (s, s), (255, 255, 255, 255))
    dr = ImageDraw.Draw(img)
    pitch = s/10
    for i in range(11):
        for j in range(11):
            cx = i*pitch + (pitch/2 if j % 2 else 0)
            cy = j*pitch
            r = pitch*5/9/2
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 0))
    p = tex/"perforation.png"
    img.save(p)
    out["perforation"] = p
    # signage
    for name, lines, bg, fg, size in (
            ("sign_robot", ["ROBOT CELL", "KNIFE IN USE", "Keep out of the taped zone", "while the arms are running"],
             (250, 204, 20), (20, 20, 20), (900, 600)),
            ("sign_wash", ["HAND WASH", "BASIN ONLY", "No food, no utensils"], (22, 110, 60), (255, 255, 255), (800, 500)),
            ("sign_fish", ["RAW FISH", "use BLUE boards", "and blue-handled tools"], (30, 80, 160), (255, 255, 255), (800, 500)),
            ("sign_chiller", ["WALK-IN CHILLER", "0 to +3 C", "Keep door closed"], (240, 240, 240), (20, 60, 140), (700, 450))):
        img = Image.new("RGB", size, bg)
        dr = ImageDraw.Draw(img)
        dr.rectangle([12, 12, size[0] - 12, size[1] - 12], outline=fg, width=10)
        y = 60
        for i, t in enumerate(lines):
            fs = 110 if i == 0 else 64 if i == 1 else 40
            f = _font(fs, bold=i < 2)
            while dr.textlength(t, font=f) > size[0] - 70 and fs > 12:     # fit the line inside the border
                fs -= 4
                f = _font(fs, bold=i < 2)
            wdt = dr.textlength(t, font=f)
            dr.text(((size[0] - wdt)/2, y), t, font=f, fill=fg)
            y += 150 if i == 0 else 100 if i == 1 else 60
        if name == "sign_robot":
            dr.polygon([(80, 540), (150, 420), (220, 540)], outline=fg, width=10)
            dr.text((138, 452), "!", font=_font(70), fill=fg)
        p = tex/f"{name}.png"
        img.save(p)
        out[name] = p
    # polystyrene fish box (white beads), 0.25 m
    arr = 232 + 10*rng.standard_normal((512, 512))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)).convert("RGB")
    p = tex/"polystyrene.png"
    img.save(p)
    out["polystyrene"] = p
    # scale display (reading written per frame by render.py; the default reads 0 g)
    out["scale_display"] = scale_display_texture(tex, 0.)
    # trout skin (stand-in) for the dressing fish on ice
    out["trout_skin"] = trout_skin_texture(tex/"trout_skin_standin.png", seed=3)
    out["trout_fin"] = fin_texture(tex/"trout_fin_standin.png")
    # wall clock face (authored): 300 mm, hour numerals, minute ticks
    img = Image.new("RGB", (1024, 1024), (246, 246, 242))
    dr = ImageDraw.Draw(img)
    for m in range(60):
        a = math.radians(m*6)
        r0, r1 = (440, 490) if m % 5 == 0 else (465, 490)
        dr.line([(512 + r0*math.sin(a), 512 - r0*math.cos(a)), (512 + r1*math.sin(a), 512 - r1*math.cos(a))],
                fill=(20, 20, 20), width=14 if m % 5 == 0 else 5)
    for hnum in range(1, 13):
        a = math.radians(hnum*30)
        f = _font(96)
        t = str(hnum)
        wdt = dr.textlength(t, font=f)
        dr.text((512 + 360*math.sin(a) - wdt/2, 512 - 360*math.cos(a) - 58), t, font=f, fill=(20, 20, 20))
    p = tex/"clock_face.png"
    img.save(p)
    out["clock_face"] = p
    # glove box, film box prints
    for name, title, col in (("gloves_blue", "NITRILE GLOVES  M", (40, 90, 190)), ("film_box", "CATERING CLING FILM 450 mm x 300 m", (200, 40, 40))):
        img = Image.new("RGB", (768, 256), (245, 245, 243))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, 768, 70], fill=col)
        dr.text((20, 12), title, font=_font(38), fill=(255, 255, 255))
        dr.text((20, 110), "FOOD SAFE  -  catering supply", font=_font(30, False), fill=(60, 60, 60))
        p = tex/f"{name}.png"
        img.save(p)
        out[name] = p
    return out


def scale_display_texture(tex_dir, grams):
    """Seven-segment-style LCD reading of the bench scale (grams, 1 g resolution)."""
    tex_dir = Path(tex_dir)
    g = int(round(max(0., float(grams))))
    p = tex_dir/f"scale_display_{g:05d}.png"
    if p.exists():
        return p
    img = Image.new("RGB", (512, 160), (148, 170, 132))
    dr = ImageDraw.Draw(img)
    txt = f"{g/1000.:.3f}"
    f = _font(104)
    wdt = dr.textlength(txt, font=f)
    dr.text((400 - wdt, 18), txt, font=f, fill=(24, 30, 22))
    dr.text((412, 70), "kg", font=_font(46), fill=(24, 30, 22))
    dr.text((14, 8), "NET", font=_font(26), fill=(24, 30, 22))
    img.save(p)
    return p


# ------------------------------------------------------------------------------------------------ USD authoring
class Dresser:
    def __init__(self, stage, folder, preset="A"):
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.stage, self.folder, self.preset = stage, Path(folder), preset
        self.tex_dir = self.folder/"textures"
        self.tex_dir.mkdir(parents=True, exist_ok=True)
        self.mats = {}
        self.credits = {}
        self.support = []
        self.adjustments = []
        self.tex = make_textures(self.folder, preset)
        mdl = ROOT/"fish_surfaces.mdl"
        shutil.copy2(mdl, self.folder/"fish_surfaces.mdl")

    # ---------------------------------------------------------------- materials
    def _copy(self, src):
        src = Path(src)
        dst = self.tex_dir/src.name
        if src.resolve() != dst.resolve() and not dst.exists():
            shutil.copy2(src, dst)
        return dst

    def preview(self, name, srgb=(.8, .8, .8), rough=.5, metal=0., coat=0., coat_rough=.05, opacity=None,
                texture=None, alpha_cut=False, emissive=None, ior=None):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/DressMaterials/" + name)
        s = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if coat:
            s.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(coat))
            s.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(coat_rough))
        if ior:
            s.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
        if texture is not None:
            t = self._copy(texture)
            rd = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/st")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(t.resolve())))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            if alpha_cut:
                s.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                    tx.CreateOutput("a", Sdf.ValueTypeNames.Float))
                s.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(.5)
        else:
            s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, lin(srgb))))
        if opacity is not None:
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        if emissive is not None:
            s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def omnipbr(self, name, diff=None, rough=None, normal=None, tile_m=1., srgb=(.8, .8, .8), rough_const=.5,
                metal=0., rough_influence=1., bump=1., emissive=None, emissive_intensity=0., opacity_tex=None):
        """OmniPBR; meshes carry UVs in metres, so texture_scale = 1/tile_m reproduces the physical tile."""
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/DressMaterials/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/rtx_surface")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        inp = lambda k, t, v: sh.CreateInput(k, t).Set(v)
        inp("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(1./tile_m, 1./tile_m))
        inp("project_uvw", Sdf.ValueTypeNames.Bool, False)
        inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, lin(srgb))))
        inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough_const))
        inp("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
        if diff is not None:
            inp("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(self._copy(diff).resolve())))
        if rough is not None:
            inp("reflectionroughness_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(self._copy(rough).resolve())))
            inp("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, float(rough_influence))
        if normal is not None:
            inp("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(self._copy(normal).resolve())))
            inp("bump_factor", Sdf.ValueTypeNames.Float, float(bump))
        if emissive is not None:
            inp("enable_emission", Sdf.ValueTypeNames.Bool, True)
            inp("emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, emissive)))
            inp("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissive_intensity))
        if opacity_tex is not None:
            inp("enable_opacity", Sdf.ValueTypeNames.Bool, True)
            inp("enable_opacity_texture", Sdf.ValueTypeNames.Bool, True)
            inp("opacity_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(self._copy(opacity_tex).resolve())))
            inp("opacity_mode", Sdf.ValueTypeNames.Int, 0)
            inp("opacity_threshold", Sdf.ValueTypeNames.Float, .5)
        self.mats[name] = m
        return m

    def steel(self, name="steel", rough=.26, srgb=(195, 197, 198), tile=.30):
        """Brushed stainless (grade 304 appearance): metallic, roughness streaks along the texture u axis. Base
        colour = normal-incidence reflectance of iron/stainless, about 0.55-0.58 linear (common PBR reference
        value, remembered) -> sRGB ~196."""
        return self.omnipbr(name, rough=self.tex["steel_rough"], tile_m=tile, srgb=srgb, rough_const=rough,
                            metal=1., rough_influence=.85)

    def glass(self, name, srgb=(245, 250, 255), ior=1.5, frost=0.01, thin=False, depth=None):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/DressMaterials/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/rtx_surface")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, lin(srgb))))
        sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(ior))
        sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(float(frost))
        sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(bool(thin))
        if depth is not None:
            sh.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(float(depth))
        self.mats[name] = m
        return m

    def tissue_material(self, name, sub, **inputs):
        """Material from filleting/fish_surfaces.mdl (copied next to the stage)."""
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Materials/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath(str((self.folder/"fish_surfaces.mdl").resolve())), "mdl")
        sh.SetSourceAssetSubIdentifier(sub, "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        for k, v in inputs.items():
            if isinstance(v, (str, Path)):
                sh.CreateInput(k, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(self._copy(v).resolve())))
            elif isinstance(v, tuple):
                sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, v)))
            elif isinstance(v, bool):
                sh.CreateInput(k, Sdf.ValueTypeNames.Bool).Set(v)
            elif isinstance(v, (int, np.integer)):               # MDL int inputs (e.g. fish_surfaces flesh aux_mode)
                sh.CreateInput(k, Sdf.ValueTypeNames.Int).Set(int(v))
            else:
                sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
        self.mats[name] = m
        return m

    # ---------------------------------------------------------------- geometry
    def xform(self, path, pos=(0, 0, 0), quat=None, yaw=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        if yaw is not None:
            quat = (0., 0., math.sin(yaw/2), math.cos(yaw/2))
        if quat is not None:
            x.AddOrientOp().Set(self.Gf.Quatf(float(quat[3]), self.Gf.Vec3f(*map(float, quat[:3]))))
        return x

    def mesh(self, path, V, F, mat, uv=None, uv_scale=1., normals="crease", angle=32., double=False):
        """Triangle mesh with faceVarying normals (crease-split) and UVs (metres x uv_scale by default)."""
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        V = np.asarray(V, np.float64)
        F = np.asarray(F, np.int64).reshape(-1, 3)
        g = UsdGeom.Mesh.Define(self.stage, path)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel().astype(np.int32)))
        g.CreateSubdivisionSchemeAttr("none")
        g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([V.min(0), V.max(0)], np.float32)))
        if normals == "crease":
            n = crease_normals(V, F, angle)
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
            g.SetNormalsInterpolation("faceVarying")
        if uv is None:
            uvv, interp = box_uv(V, F)*uv_scale, "faceVarying"
        else:
            uvv = np.asarray(uv, float)
            interp = "vertex" if len(uvv) == len(V) else "faceVarying"
        UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interp).Set(
            Vt.Vec2fArray.FromNumpy(uvv.astype(np.float32)))
        if double:
            g.CreateDoubleSidedAttr(True)
        if mat is not None:
            self.UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        return g

    def box(self, path, center, size, mat, r=.002, yaw=0., uv_scale=1.):
        V, F = rbox_mesh(size, r) if r > 0 else box_mesh(size)
        V = rotz(V, yaw) + np.asarray(center, float)
        return self.mesh(path, V, F, mat, uv_scale=uv_scale)

    def cyl(self, path, p0, p1, r, mat, n=40, caps=True):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        V, F = tube([p0, p1], r, n=n, caps=caps)
        return self.mesh(path, V, F, mat, angle=40.)

    def ref(self, path, usd, prim, pos, yaw=0., credit=None, textures=None, ground=False):
        """Reference a pinned model and rebind its materials to OmniPBR (label_application recipe). ground=True
        shifts the model so its measured lowest point sits at pos z (asset origins are not always at the base)."""
        from label_application.dressing import Dresser as LAD
        outer = self.xform(path, pos, yaw=yaw)
        inner = self.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), prim)
        if textures is not None:
            for f in Path(textures).glob("*"):
                self._copy(f)
        # the label_application helper only needs .stage, .Sdf, .Gf, .UsdShade, .UsdGeom
        LAD._rebind_omnipbr(self, path, Path(usd).stem.replace("_1k", ""), Path(usd))
        if ground:
            from pxr import Usd
            r = self.UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
                self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
            inner.AddTranslateOp().Set(self.Gf.Vec3d(0., 0., float(pos[2] - r.GetMin()[2])))
        if credit:
            self.credits[credit[0]] = credit[1]
        return outer

    def rest(self, item, bottom_z, support, support_z):
        self.support.append(dict(item=item, bottom_z=round(float(bottom_z), 5), support=support,
                                 support_z=round(float(support_z), 5), gap_mm=round(1e3*(bottom_z - support_z), 3)))


# ------------------------------------------------------------------------------------------------ layout checks
def layout_conflicts(task=None):
    """Axis-aligned overlaps between layout objects that share the bench top (physically impossible). The
    render never edits layout.py; it reports these and states what it draws instead. `task`: include its task-only
    props and arrangement (layout.FRAME_PAN, task_spec)."""
    boxes = {}
    tx, ty = L.TABLE["center"]
    items = [("BOARD", L.BOARD), ("ICE_PAN", dict(L.ICE_PAN, center=L.ice_pan_center(task))),
             ("FILLET_TRAY", L.fillet_tray(task)), ("KNIFE_BLOCK", L.KNIFE_BLOCK),
             ("TWEEZER_TRAY", L.TWEEZER_TRAY), ("SCALE", L.SCALE)]
    if task in L.FRAME_PAN["tasks"]:
        items.append(("FRAME_PAN", L.FRAME_PAN))
    for name, d in items:
        (cx, cy), (sx, sy) = d["center"], d["size"][:2]
        boxes[name] = (cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2)
    c = L.BOWL["center"]
    boxes["BOWL"] = (c[0] - L.BOWL["diameter"]/2, c[0] + L.BOWL["diameter"]/2, c[1] - L.BOWL["diameter"]/2,
                     c[1] + L.BOWL["diameter"]/2)
    (px, py), (sx, sy, _) = L.PEDESTAL["center"], L.PEDESTAL["size"]
    ped = (px - sx/2, px + sx/2, py - sy/2, py + sy/2)
    table = (tx - L.TABLE["size"][0]/2, tx + L.TABLE["size"][0]/2, ty - L.TABLE["size"][1]/2, ty + L.TABLE["size"][1]/2)
    out = []
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = boxes[a], boxes[b]
            ox = min(A[1], B[1]) - max(A[0], B[0])
            oy = min(A[3], B[3]) - max(A[2], B[2])
            if ox > 0 and oy > 0:
                pa = set(getattr(L, a).get("presets", "ABC")) if hasattr(L, a) and isinstance(getattr(L, a), dict) else set("ABC")
                pb = set(getattr(L, b).get("presets", "ABC")) if hasattr(L, b) and isinstance(getattr(L, b), dict) else set("ABC")
                if not pa & pb:
                    continue                      # never present in the same preset
                if "BOWL" in (a, b):
                    # the bowl is round: check the true circle against the box
                    other = B if a == "BOWL" else A
                    cx, cy = L.BOWL["center"]
                    qx = min(max(cx, other[0]), other[1])
                    qy = min(max(cy, other[2]), other[3])
                    pen = L.BOWL["diameter"]/2 - math.hypot(qx - cx, qy - cy)
                    if pen <= 0:
                        continue
                    note = f"bowl rim circle enters the box by {pen*1e3:.1f} mm"
                else:
                    note = f"overlap {ox*1e3:.1f} x {oy*1e3:.1f} mm in plan"
                out.append(dict(a=a, b=b, overlap_x_mm=round(ox*1e3, 1), overlap_y_mm=round(oy*1e3, 1), note=note))
    oy = min(ped[3], table[3]) - max(ped[2], table[2])
    ox = min(ped[1], table[1]) - max(ped[0], table[0])
    if ox > 0 and oy > 0:
        out.append(dict(a="PEDESTAL", b="TABLE", overlap_x_mm=round(ox*1e3, 1), overlap_y_mm=round(oy*1e3, 1),
                        note="pedestal (0.9 m tall) and bench top slab (z 0.86-0.90) occupy the same volume"))
    return out


# ------------------------------------------------------------------------------------------------ props (local frames)
def gn_pan_geometry(size, depth, rim=.012, taper=.018, r_top=.028, t=.0008, lip=.004):
    """Gastronorm pan shell (EN 631 outer rim size), origin at the bottom-face centre. Returns (V, F, floor_faces
    mask) - the inner floor faces are flagged so a perforated insert can use a cut-out material there."""
    Lx, Wy = size
    hx, hy = Lx/2, Wy/2
    n = 7

    def ring(inset, z, rr=None):
        rr_ = r_top - inset if rr is None else rr
        pts = rrect(hx - inset, hy - inset, max(rr_, .004), n)
        return np.column_stack([pts, np.full(len(pts), z)])
    wi = rim + taper                                   # inner wall inset at the floor
    prof = [ring(wi + .010, t), ring(wi + .004, t + .0012), ring(wi + .0005, t + .005),   # inner floor fillet
            ring(rim + .0015, depth - .003), ring(rim, depth),                               # inner wall to rim
            ring(0., depth), ring(0., depth - lip), ring(t*1.5, depth - lip),              # rim flange and lip
            ring(rim - t, depth - t - .002), ring(wi - t + .0005, .005), ring(wi - t + .004, .0008),
            ring(wi - t + .010, 0.)]
    V, F = loft(prof, cap0=True, cap1=True)
    floor = np.zeros(len(F), bool)
    npt = len(prof[0])
    # cap0 fan (inner floor) faces are the last-but-one block added by loft: identify by vertex z
    cz = V[F].mean(1)[:, 2]
    floor[(np.abs(cz - t) < 1e-5)] = True
    return V, F, floor


def bowl_geometry(D=None, depth=None, t=.0007):
    """Stainless mixing bowl, 240 mm rim diameter, 80 mm deep (layout); flat base 130 mm; rolled rim bead.
    Origin at the bottom-face centre."""
    D = D or L.BOWL["diameter"]
    depth = depth or L.BOWL["depth"]
    R = D/2
    rb = .065
    prof_o = []
    for s in np.linspace(0, 1, 24):
        ang = s*math.pi/2
        r = rb + (R - .003 - rb)*math.sin(ang)**.9
        z = (depth - .003)*(1 - math.cos(ang))**1.05
        prof_o.append((r, z))
    bead = [(R - .003 + .003*math.sin(a), depth - .003 + .003*(1 - math.cos(a))) for a in np.linspace(0, math.pi, 9)]
    # inner surface goes up from the centre, bead, outer surface back down to the centre
    prof = [(1e-4, t), (rb*.6, t)] + [(r - t, z + t) for r, z in prof_o[1:]] + bead[::-1] + \
        [(r, z) for r, z in prof_o[::-1][1:]] + [(rb*.6, 0.), (1e-4, 0.)]
    return lathe(prof, 96)


def standin_trout_meshes(seed=0):
    body = trout_body()
    fins, eye = trout_fins(body)
    return body, fins, eye


def build_trout(d, path, pos, yaw, skin_mat, fin_mat, eye_mats):
    """A whole gutted trout (STAND-IN visual mesh, dressing only) lying on its side at pos (lowest point)."""
    body, fins, eye = standin_trout_meshes()
    x = d.xform(path, pos, yaw=yaw)
    d.mesh(path + "/body", body["V"], body["F"], skin_mat, uv=body["UV"], angle=70.)
    for name, V, F, UV in fins:
        d.mesh(path + f"/fin_{name}", V, F, fin_mat, uv=UV, double=True)
    c = np.asarray(eye["center"])
    V, F = lathe([(1e-4, -eye["r"]*.55)] + [(eye["r"]*math.sin(a), -eye["r"]*.55*math.cos(a)) for a in np.linspace(.2, math.pi/2, 8)]
                 + [(eye["r"]*.98, .0005), (eye["r"]*.55, .0016), (1e-4, .0019)], 32)
    d.mesh(path + "/eye", V + c, F, eye_mats[0], angle=80.)
    V2, F2 = lathe([(1e-4, .0019), (eye["r"]*.48, .0017), (eye["r"]*.5, .0012)], 24)
    d.mesh(path + "/pupil", V2 + c + np.array([0, 0, .0002]), F2, eye_mats[1], angle=80.)
    return x


def build_knife(d, path, kind, part="whole", mats=None):
    g = knife_geometry(kind)
    m = mats or knife_materials(d)
    if part in ("whole", "blade"):
        d.mesh(path + "/blade", *g["blade"], m["blade"], angle=20.)
        d.mesh(path + "/bevel", *g["bevel"], m["edge"], angle=20.)
    if part in ("whole", "handle"):
        d.mesh(path + "/handle", *g["handle"], m["handle"], angle=45.)
    if part.startswith("seg"):
        k = int(part[3:])
        spec = g["spec"]
        xc = (k + .5)*spec.seg_len                       # segment frame: origin at its edge centre
        (Vf, Ff), (Vb, Fb) = g["segments"][k]
        d.mesh(path + "/blade", np.asarray(Vf) - np.array([xc, 0, 0]), Ff, m["blade"], angle=20.)
        d.mesh(path + "/bevel", np.asarray(Vb) - np.array([xc, 0, 0]), Fb, m["edge"], angle=20.)
    return g


def knife_materials(d):
    return dict(blade=d.omnipbr("knife_blade", rough=d.tex["steel_rough"], tile_m=.15, srgb=(196, 198, 199),
                                rough_const=.30, metal=1., rough_influence=.5),
                edge=d.preview("knife_edge", (215, 216, 218), .15, 1.),
                handle=d.preview("fibrox_black", (24, 25, 26), .62))


def build_board(d, path, colour="blue"):
    """NSF HDPE board 610 x 457 x 12.7 mm (layout; read: common NSF size), local origin at its centre."""
    sx, sy, sz = L.BOARD["size"]
    m = d.omnipbr(f"hdpe_{colour}", diff=d.tex[f"board_{colour}"], tile_m=1., rough_const=.55)
    V, F = rbox_mesh((sx, sy, sz), .0025)
    # UV: top face spans the whole texture once
    uv = box_uv(V, F)/np.array([sx, sy]) + .5
    return d.mesh(path + "/board", V, F, m, uv=uv)


def build_scale(d, path, grams=0.):
    """Bench portion scale, 300 x 250 mm stainless platform, 45 mm (layout), front LCD tab (authored)."""
    sx, sy, sz = L.SCALE["size"]
    body = d.preview("scale_abs", (228, 228, 224), .45)
    st = d.steel("platform_steel", .22)
    d.box(path + "/base", (0, 0, .014), (sx - .006, sy - .006, .028), body, r=.006)
    d.box(path + "/platform", (0, 0, .028 + (sz - .028)/2), (sx, sy, sz - .028), st, r=.002)
    # display tab on the front (-y) face, reading the solved platform load (render.py updates the texture)
    d.box(path + "/display_tab", (0, -sy/2 - .018, .016), (.13, .036, .032), body, r=.004)
    lcd = d.preview("scale_lcd", texture=scale_display_texture(d.tex_dir, grams), rough=.15)
    q = np.array([(-.052, -sy/2 - .0361, .006), (.052, -sy/2 - .0361, .006), (.052, -sy/2 - .0361, .027),
                  (-.052, -sy/2 - .0361, .027)])
    d.mesh(path + "/lcd", q, np.array([(0, 1, 2), (0, 2, 3)]), lcd, uv=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
           normals=None)
    keys = d.preview("scale_keys", (60, 110, 190), .4)
    for k, x in enumerate((-.058, .058)):
        d.box(path + f"/key_{k}", (x, -sy/2 - .036, .016), (.012, .002, .010), keys, r=.001)
    return lcd


def build_tweezer_tray(d, path):
    """Stainless tray 200 x 100 x 15 mm (layout) holding the fish-bone tweezers."""
    sx, sy, sz = L.TWEEZER_TRAY["size"]
    V, F, _ = gn_pan_geometry((sx, sy), sz, rim=.006, taper=.004, r_top=.012, lip=.002)
    d.mesh(path + "/tray", V, F, d.steel("tray_steel", .2), angle=35.)


def build_tweezers(d, path):
    V, F = tweezers_geometry()
    d.mesh(path + "/tweezers", V, F, d.steel("tweezer_steel", .18), angle=30.)


BLOCK_CHAMFER = .040
BLOCK_MARGIN = .005                                    # m: blade corners at least this far inside the block


def _block_profile():
    """(y, z) outline of the knife block (layout KNIFE_BLOCK), CCW: bottom, back, top, 45 deg front chamfer, front."""
    _, sy, sz = L.KNIFE_BLOCK["size"]
    c = BLOCK_CHAMFER
    return [(-sy/2, 0.), (sy/2, 0.), (sy/2, sz), (-sy/2 + c, sz), (-sy/2, sz - c)]


def _inside_distance(pts, poly):
    """Signed distance (positive inside) of 2D points to a convex CCW polygon."""
    P = np.asarray(poly, float)
    out = np.full(len(pts), np.inf)
    for a, b in zip(P, np.roll(P, -1, 0)):
        e = b - a
        n = np.array([-e[1], e[0]])/np.linalg.norm(e)             # inward for CCW
        out = np.minimum(out, (np.asarray(pts) - a) @ n)
    return out


def knife_block_fit():
    """Slot geometry of the layout knife block: the steepest slot angle (70 -> 30 deg below horizontal, blade running
    down-back, handle up toward the robots) at which BOTH knives' blades lie inside the block with BLOCK_MARGIN, the
    blade's heel spine at the block surface (the slot mouth). Returns dict(angle_deg, xdir, zdir, heels {kind: (y, z)},
    clearance_mm {kind}) or the best (negative-clearance) attempt."""
    prof = _block_profile()
    edge_pts = np.array(prof + [prof[0]])
    best = None
    for ang in np.arange(70., 29., -1.):
        a = math.radians(ang)
        xd = np.array([math.cos(a), -math.sin(a)])              # (y, z): heel -> tip
        zd = np.array([math.sin(a), math.cos(a)])               # edge -> spine
        heels, clear = {}, {}
        for kind in ("fillet", "boning"):
            dd = PR.v(PR.FILLET_BLADE if kind == "fillet" else PR.BONING_BLADE)
            Lb, Hb = dd["length"], dd["height"]
            top_best = None
            # candidate slot mouths along the top face and the chamfer
            for i in (2, 3):
                for t in np.linspace(0, 1, 121):
                    mouth = edge_pts[i] + t*(edge_pts[i + 1] - edge_pts[i])
                    heel = mouth - .002*zd - Hb*zd                 # spine heel 2 mm below the surface
                    corners = np.array([heel, heel + Lb*xd, heel + Lb*xd + Hb*zd])
                    m = float(_inside_distance(corners, prof).min())
                    if top_best is None or m > top_best[1]:
                        top_best = (heel, m)
            heel, _ = top_best
            # the 19 x 32 mm handle cannot enter a 3.6 mm slot: slide the knife up its axis until the handle's front
            # face (blade frame x = 0, z = H +- HANDLE_H/2) is on or above the block surface
            hh = PR.v(PR.HANDLE)["height"]/2
            face = lambda h_: np.array([h_ + (Hb - hh)*zd, h_ + (Hb + hh)*zd])
            for _ in range(200):
                if float(_inside_distance(face(heel), prof).max()) <= 0.:
                    break
                heel = heel - .0005*xd
            corners = np.array([heel + (Hb - hh)*zd*0 + 0*xd, heel + Lb*xd, heel + Lb*xd + Hb*zd])
            heels[kind], clear[kind] = heel, float(_inside_distance(corners[1:], prof).min())
        worst = min(clear.values())
        cand = dict(angle_deg=float(ang), xdir=np.array([0., xd[0], xd[1]]), zdir=np.array([0., zd[0], zd[1]]),
                    heels=heels, clearance_mm={k: round(1e3*v, 1) for k, v in clear.items()})
        if worst >= BLOCK_MARGIN:
            return cand
        if best is None or worst > min(best["clearance_mm"].values())/1e3:
            best = cand
    return best


def build_knife_block(d, path, kinds=("fillet", "boning")):
    """Knife block (layout KNIFE_BLOCK, 100 x 160 x 200 mm), grey polypropylene, two diagonal slots running down-back
    at the steepest angle where both blades fit (knife_block_fit); knives handle up toward the robots (-y). Local
    origin at the bottom-face centre. A knife is drawn only if its blade fits (clearance > 1 mm)."""
    sx, sy, sz = L.KNIFE_BLOCK["size"]
    pp = d.preview("block_pp", (70, 72, 74), .55)
    prof = _block_profile()
    rings = [np.array([(x, y, z) for y, z in prof]) for x in (-sx/2, sx/2)]
    V, F = loft(rings, closed=True, cap0=True, cap1=True)
    d.mesh(path + "/block", V, F, pp, angle=20.)
    slot = d.preview("slot_dark", (12, 12, 12), .8)
    fit = knife_block_fit()
    from scipy.spatial.transform import Rotation
    xdir, zdir = fit["xdir"], fit["zdir"]
    Rm = np.column_stack([xdir, np.cross(zdir, xdir), zdir])
    q = Rotation.from_matrix(Rm).as_quat()
    poses = {}
    for k, kind in enumerate(("fillet", "boning")):
        xk = L.KNIFE_BLOCK["slot_x"][k]
        hy, hz = fit["heels"][kind]
        heel = np.array([xk, hy, hz])
        Hb = PR.v(PR.FILLET_BLADE if kind == "fillet" else PR.BONING_BLADE)["height"]
        # slot mouth: where the blade plane crosses the block surface, between the spine and edge lines
        def crossing(base):
            lo_, hi_ = -.2, .2                                  # t along xdir: inside at hi_, outside at lo_
            for _ in range(50):
                mid = .5*(lo_ + hi_)
                pt = base[1:] + mid*xdir[1:]
                if _inside_distance(pt[None], prof)[0] > 0:
                    hi_ = mid
                else:
                    lo_ = mid
            return base + hi_*xdir
        a_, b_ = crossing(heel), crossing(heel + Hb*zdir)
        mouth = np.array([a_, b_]) + np.array([0., -.0002, .0003])
        pts = [mouth[0] + np.array([-.0018, 0, 0]), mouth[0] + np.array([.0018, 0, 0]),
               mouth[1] + np.array([.0018, 0, 0]), mouth[1] + np.array([-.0018, 0, 0])]
        d.mesh(path + f"/slot_{k}", np.array(pts), np.array([(0, 1, 2), (0, 2, 3)]), slot, normals=None, double=True)
        poses[kind] = (heel, q)
        if kind in kinds and fit["clearance_mm"][kind] > 1.:
            kp = path + f"/knife_{kind}"
            d.xform(kp, heel, quat=q)
            build_knife(d, kp, kind)
    d.block_fit_mm = fit["clearance_mm"]
    d.block_slot_deg = fit["angle_deg"]
    return poses


def build_honing_steel(d, path):
    """Honing steel lying on the bench: 260 mm oval rod (estimate: 25-30 cm class), 40 mm guard, black handle.
    Local frame: x along the steel; it rests on the guard rim and the rod tip (tilt computed)."""
    rod_len, rod_r, guard_r, hdl_len, hdl_r = .26, .0065, .020, .125, .015
    tilt = math.atan2(guard_r - rod_r, rod_len)       # rests on the guard and the tip
    Rm = rot_matrix((0, 1, 0), tilt)                  # tip (+x) goes down
    zc = guard_r                                      # guard centre height above the bench at the guard
    parts = []
    steel = d.steel("honing_steel", .30, srgb=(150, 152, 155))
    blk = d.preview("honing_handle", (22, 22, 24), .6)
    for nm, p0, p1, r, m in (("rod", (0, 0, 0), (rod_len, 0, 0), rod_r, steel),
                              ("guard", (-.004, 0, 0), (.001, 0, 0), guard_r, steel),
                              ("handle", (-hdl_len, 0, 0), (-.004, 0, 0), hdl_r, blk)):
        V, F = tube([p0, p1], r, n=40)
        V = V @ Rm.T + np.array([0, 0, zc])
        d.mesh(path + "/" + nm, V, F, m, angle=40.)
    ring = tube([(-hdl_len - .012, 0, 0), (-hdl_len, 0, 0)], .007, n=24)
    V = ring[0] @ Rm.T + np.array([0, 0, zc])
    d.mesh(path + "/hang_ring", V, ring[1], steel, angle=40.)
    return tilt


def build_spray_bottle(d, path):
    """500 mL trigger sprayer of food-safe sanitiser (estimate: 65 x 45 x 250 mm)."""
    pe = d.preview("bottle_pe", (230, 232, 236), .35, opacity=.85)
    blue = d.preview("trigger_blue", (30, 80, 170), .4)
    rings = []
    for s in np.linspace(0, 1, 20):
        z = .19*s
        w = .0325 if s < .8 else .0325 - (s - .8)/.2*.018
        dd = .0225 if s < .8 else .0225 - (s - .8)/.2*.009
        ring = rrect(w, dd, .012 if s < .8 else .008, 5)
        rings.append(np.column_stack([ring, np.full(len(ring), z)]))
    V, F = loft(rings, cap0=True, cap1=True)
    d.mesh(path + "/body", V, F, pe, angle=40.)
    d.cyl(path + "/cap", (0, 0, .19), (0, 0, .215), .013, blue)
    d.box(path + "/head", (0, -.012, .232), (.028, .06, .034), blue, r=.006)
    d.box(path + "/trigger", (0, -.028, .205), (.014, .010, .035), blue, r=.003)
    d.cyl(path + "/nozzle", (0, -.042, .238), (0, -.048, .238), .005, d.preview("nozzle", (40, 40, 40), .5))


def build_gn_stack(d, path, size, depth, count, mat, step=.012):
    """Nested stack of identical GN pans (each nests `step` higher)."""
    V, F, _ = gn_pan_geometry(size, depth)
    for k in range(count):
        d.mesh(path + f"/pan_{k}", V + np.array([0, 0, k*step]), F, mat, angle=35.)


def build_pedal_bin(d, path, lid_srgb, body_srgb=(236, 236, 232), r=.18, h=.62):
    """Pedal bin (estimate: 360 mm dia, 650 mm tall, 50 L class), coloured lid."""
    body = d.preview("bin_body_" + "_".join(map(str, body_srgb)), body_srgb, .45)
    lid = d.preview("bin_lid_" + "_".join(map(str, lid_srgb)), lid_srgb, .4)
    V, F = lathe([(1e-4, .012), (r*.93, .012), (r*.97, .03), (r, h - .02), (r*1.01, h - .01), (1e-4, h - .01)], 64)
    d.mesh(path + "/body", V, F, body, angle=40.)
    V, F = lathe([(r*1.025, h - .012), (r*1.03, h + .004), (r*.9, h + .03), (1e-4, h + .036)], 64)
    d.mesh(path + "/lid", V, F, lid, angle=40.)
    d.box(path + "/pedal", (0, -r - .03, .018), (.13, .07, .02), d.preview("bin_pedal", (40, 40, 40), .5), r=.004)
    d.box(path + "/hinge", (0, r + .01, h - .005), (.10, .03, .025), lid, r=.004)
    d.box(path + "/foot", (0, 0, .006), (r*1.5, r*1.5, .012), d.preview("bin_foot", (50, 50, 50), .6), r=.004)


def build_fish_box(d, path, size=(.60, .40, .20), lid=False):
    """Expanded-polystyrene fish box (estimate: 600 x 400 x 200 mm class)."""
    m = d.preview("eps", texture=d.tex["polystyrene"], rough=.85)
    sx, sy, sz = size
    t = .025
    V, F = merge(box_mesh((sx, sy, t), (0, 0, t/2)), box_mesh((t, sy, sz), (-sx/2 + t/2, 0, sz/2)),
                 box_mesh((t, sy, sz), (sx/2 - t/2, 0, sz/2)), box_mesh((sx - 2*t, t, sz), (0, -sy/2 + t/2, sz/2)),
                 box_mesh((sx - 2*t, t, sz), (0, sy/2 - t/2, sz/2)))
    d.mesh(path + "/box", V, F, m, uv_scale=4.)
    if lid:
        d.box(path + "/lid", (0, 0, sz + .01), (sx + .004, sy + .004, .02), m, r=.004, uv_scale=4.)


# ------------------------------------------------------------------------------------------------ room
def build_room(d):
    UsdLux, Gf = d.UsdLux, d.Gf
    # floor: red quarry-type tiles, fall to the gully
    ft = fetch("terracotta_floor_tiles")
    fm = d.omnipbr("quarry_floor", diff=ft/"terracotta_floor_tiles_diff_2k.jpg",
                   rough=ft/"terracotta_floor_tiles_rough_2k.jpg", normal=ft/"terracotta_floor_tiles_nor_gl_2k.jpg",
                   tile_m=2.08, rough_const=.5, rough_influence=.8, bump=.6)
    d.credits["terracotta_floor_tiles"] = "Poly Haven, CC0 (filleting/visual_assets.json)"
    xs = np.linspace(X0, X1, 101)
    ys = np.linspace(Y0, Y1, 81)
    # make sure the hopper ridge lines are sampled: add drain-centred knots
    cx, cy = DRAIN["center"]
    hx, hy = DRAIN["half"]
    xs = np.unique(np.concatenate([xs, [cx - hx, cx - DRAIN["grate"]/2, cx, cx + DRAIN["grate"]/2, cx + hx]]))
    ys = np.unique(np.concatenate([ys, [cy - hy, cy - DRAIN["grate"]/2, cy, cy + DRAIN["grate"]/2, cy + hy]]))
    GX, GY = np.meshgrid(xs, ys)
    GZ = floor_z(GX, GY)
    inside = (np.abs(GX - cx) < DRAIN["grate"]/2 - 1e-6) & (np.abs(GY - cy) < DRAIN["grate"]/2 - 1e-6)
    V = np.stack([GX, GY, GZ], -1).reshape(-1, 3)
    F = []
    nx = len(xs)
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            a = j*nx + i
            if inside[j, i] and inside[j, i + 1] and inside[j + 1, i] and inside[j + 1, i + 1]:
                continue
            # split along the diagonal that follows the hopper valleys
            if (GX[j, i] - cx)*(GY[j, i] - cy) > 0:
                F += [(a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)]
            else:
                F += [(a, a + 1, a + nx), (a + 1, a + nx + 1, a + nx)]
    F = np.asarray(F)
    d.mesh("/Dressing/room/floor", V, F, fm, uv=V[:, :2], normals="crease", angle=5.)
    # gully: stainless grate (slots) over a dark sump box
    gz = floor_z(cx, cy)
    st = d.steel("steel_floor", .35, srgb=(150, 152, 154))
    g = DRAIN["grate"]
    dark = d.preview("sump_dark", (10, 10, 10), .9)
    d.box("/Dressing/floor/gully_sump", (cx, cy, gz - .06), (g, g, .11), dark, r=0.)
    for k, off in enumerate((-g/2 + .006, g/2 - .006)):
        d.box(f"/Dressing/floor/gully_frame_x{k}", (cx + off, cy, gz - .002), (.012, g, .004), st, r=.0005)
        d.box(f"/Dressing/floor/gully_frame_y{k}", (cx, cy + off, gz - .002), (g, .012, .004), st, r=.0005)
    for k in range(9):
        d.box(f"/Dressing/floor/gully_bar_{k}", (cx - g/2 + .02 + k*.02, cy, gz - .002), (.006, g - .02, .004), st, r=.0005)
    prov("gully", "200 x 200 mm slotted stainless grate", "estimate", "commercial kitchen floor gully class")
    # walls (tile to 2.0 m, washable paint above), with the kitchen door and the chiller opening
    tile, _ = wiping_texture("ceramic_tiles_107")
    wm = d.omnipbr("wall_tile", diff=tile["Diffuse"], rough=tile["Rough"], normal=tile["nor_gl"], tile_m=1.2,
                   rough_const=.12, rough_influence=.6, bump=.5)
    d.credits["ceramic_tiles_107"] = "ambientCG Tiles107, CC0 (wiping/assets/ceramic_tiles_107_sources.json)"
    from label_application import visual_assets as lava
    pl = lava.texture("beige_wall_002")
    lum = np.asarray(Image.open(pl["diff"]).convert("L"), np.float32)/255.
    lum = (lum - lum.mean())/max(lum.std(), 1e-6)
    paint = np.clip(np.array([.93, .93, .91])[None, None]*(1 + .02*lum[..., None]), 0, 1)
    pp = d.tex_dir/"washable_paint_diff.jpg"
    Image.fromarray((paint*255).astype(np.uint8)).save(pp, quality=92)
    pm = d.omnipbr("wall_paint", diff=pp, rough=pl["rough"], normal=pl["nor_gl"], tile_m=3., rough_const=.6, bump=.3)
    d.credits["beige_wall_002"] = "Poly Haven, CC0 (label_application/visual_assets.json), re-toned to white paint"
    DOOR_K = dict(wall="front", u=(-1.60, -0.75), z=2.05)
    CHILL = dict(wall="right", u=(0.25, 1.15), z=2.00)
    walls = {  # name: (origin, u-axis, inward normal, length)
        "back": (np.array([X0, Y1, 0.]), np.array([1., 0, 0]), np.array([0, -1., 0]), X1 - X0),
        "front": (np.array([X1, Y0, 0.]), np.array([-1., 0, 0]), np.array([0, 1., 0]), X1 - X0),
        "left": (np.array([X0, Y0, 0.]), np.array([0, 1., 0]), np.array([1., 0, 0]), Y1 - Y0),
        "right": (np.array([X1, Y1, 0.]), np.array([0, -1., 0]), np.array([-1., 0, 0]), Y1 - Y0)}

    def to_u(wall, coord):
        """Wall coordinate u of a world x (front wall) or y (right wall)."""
        o, ax, _, _ = walls[wall]
        return (coord - o[0])*ax[0] + (coord - o[1])*ax[1]

    openings = {"front": [(sorted([to_u("front", DOOR_K["u"][0]), to_u("front", DOOR_K["u"][1])]), DOOR_K["z"])],
                "right": [(sorted([to_u("right", CHILL["u"][0]), to_u("right", CHILL["u"][1])]), CHILL["z"])]}

    def wall_quads(name):
        o, ax, nrm, length = walls[name]
        rects = []                                        # (u0, u1, z0, z1)
        cuts = openings.get(name, [])
        us = sorted({0., length} | {u for (a, b), _ in cuts for u in (a, b)})
        for u0, u1 in zip(us[:-1], us[1:]):
            um = (u0 + u1)/2
            hole = next((zt for (a, b), zt in cuts if a <= um <= b), None)
            for z0, z1 in ((-.02, TILE_TOP), (TILE_TOP, H)):
                if hole is not None:
                    z0 = max(z0, hole)
                    if z0 >= z1:
                        continue
                rects.append((u0, u1, z0, z1))
        return o, ax, nrm, rects
    for name in walls:
        o, ax, nrm, rects = wall_quads(name)
        length = walls[name][3]
        up = np.array([0, 0, 1.])
        for k, (u0, u1, z0, z1) in enumerate(rects):
            c = [o + ax*u0 + up*z0, o + ax*u1 + up*z0, o + ax*u1 + up*z1, o + ax*u0 + up*z1]
            V = np.array(c)
            Fq = np.array([(0, 1, 2), (0, 2, 3)])
            if np.dot(np.cross(V[1] - V[0], V[2] - V[0]), nrm) < 0:
                Fq = Fq[:, ::-1]
            uv = np.array([(u0, z0), (u1, z0), (u1, z1), (u0, z1)])
            d.mesh(f"/Dressing/room/wall_{name}_{k}", V, Fq, wm if z1 <= TILE_TOP + 1e-6 else pm, uv=uv, normals=None)
        # tile cap trim at 2.0 m and quarry-tile cove skirting (100 mm, hygienic coved junction)
        segs = []
        us = [0.] + [u for (a, b), _ in openings.get(name, []) for u in (a, b)] + [length]
        for i in range(0, len(us), 2):
            segs.append((us[i], us[i + 1]))
        trim = d.preview("tile_trim", (240, 240, 238), .3)
        for k, (u0, u1) in enumerate(segs):
            p0, p1 = o + ax*u0, o + ax*u1
            mid = (p0 + p1)/2 + nrm*.004
            size = np.abs(ax)*(u1 - u0) + np.abs(nrm)*.008 + np.array([0, 0, .012])
            d.box(f"/Dressing/room/trim_{name}_{k}", mid + np.array([0, 0, TILE_TOP]), size, trim, r=.002)
            # cove: concave quarter circle r=50 mm + 50 mm upstand
            prof = [(0., .10)] + [(.05 - .05*math.cos(a), .05 - .05*math.sin(a)) for a in np.linspace(0, math.pi/2, 7)]
            rings = []
            for u in (u0 + .003, u1 - .003):
                base = o + ax*u
                rings.append(np.array([base + nrm*(pv[0] + .0005) + np.array([0, 0, pv[1] + floor_z(*(base + nrm*pv[0])[:2])])
                                       for pv in prof]))
            V, F = loft(rings, closed=False)
            uv = np.array([(np.dot(v - o, ax), v[2] + np.dot(v - o, nrm)) for v in V])
            d.mesh(f"/Dressing/room/cove_{name}_{k}", V, F, fm, uv=uv, angle=60.)
    prov("cove_skirting", "50 mm radius coved quarry skirting, 100 mm high", "estimate", "hygienic coved floor-wall junction")
    # ceiling: washable PVC-faced tiles on a 600 mm grid
    cm = d.preview("ceiling", texture=d.tex["ceiling"], rough=.35)
    V = np.array([(X0, Y0, H), (X1, Y0, H), (X1, Y1, H), (X0, Y1, H)])
    d.mesh("/Dressing/room/ceiling", V, np.array([(0, 2, 1), (0, 3, 2)]), cm, uv=(V[:, :2] - [X0 - .3, Y0 - .3])/1.2,
           normals=None)
    # LED panels: 595 x 595 mm, 4000 lm (maker class, remembered); luminance L = flux/(pi A)
    flux = 3000.
    A = .56*.56
    Lum = flux/(math.pi*A)
    panels = [(-1.50, .45), (-.45, .45), (.45, .45), (1.80, .45), (-.90, -1.30), (1.10, -1.30)]
    em = d.omnipbr("led_panel", srgb=(250, 250, 250), rough_const=.3, emissive=(1., 1., 1.), emissive_intensity=Lum*.9)
    frame = d.preview("led_frame", (235, 235, 235), .35)
    for k, (px, py) in enumerate(panels):
        d.box(f"/Dressing/lights/panel_frame_{k}", (px, py, H - .006), (.595, .595, .012), frame, r=.002)
        V = np.array([(px - .28, py - .28, H - .0125), (px + .28, py - .28, H - .0125), (px + .28, py + .28, H - .0125),
                      (px - .28, py + .28, H - .0125)])
        d.mesh(f"/Dressing/lights/panel_diffuser_{k}", V, np.array([(0, 1, 2), (0, 2, 3)]), em, normals=None)
        Lt = UsdLux.RectLight.Define(d.stage, f"/Dressing/lights/panel_light_{k}")
        Lt.CreateWidthAttr(.56)
        Lt.CreateHeightAttr(.56)
        Lt.CreateIntensityAttr(float(Lum))
        Lt.CreateEnableColorTemperatureAttr(True)
        Lt.CreateColorTemperatureAttr(5000.)
        d.UsdGeom.Xformable(Lt).AddTranslateOp().Set(Gf.Vec3d(px, py, H - .014))
    # illuminance at the board centre from the Lambertian panels (numerical), for the provenance record
    bx, by = L.BOARD["center"]
    E = 0.
    g1 = np.linspace(-.28, .28, 40)
    for px, py in panels:
        X, Y = np.meshgrid(px + g1, py + g1)
        dz = H - .014 - L.BOARD_TOP_Z
        r2 = (X - bx)**2 + (Y - by)**2 + dz**2
        E += (Lum*dz*dz/r2**2).sum()*(.56/40)**2
    prov("led_panels", f"6 x 595 mm panels, 3000 lm, 5000 K each -> L = {Lum:.0f} cd/m2; direct illuminance at the board "
         f"centre {E:.0f} lux (plus inter-reflection)", "estimate+derived",
         "600x600 LED panel class 3000-4000 lm (remembered); 3000 lm chosen so the direct board illuminance sits near the 500-750 lux band; EN 12464-1 asks 500 lux for food preparation")
    d.provenance_lux = E
    return dict(floor=fm, wall_tile=wm, paint=pm)


def build_doors(d, mats):
    st = d.steel()
    white = d.preview("door_white", (238, 238, 234), .35)
    # kitchen door (front wall), closed; stainless kick and push plates
    x0, x1 = -1.60, -.75
    d.box("/Dressing/door_kitchen/frame_l", (x0 - .025, Y0 + .02, 1.03), (.05, .06, 2.1), st, r=.003)
    d.box("/Dressing/door_kitchen/frame_r", (x1 + .025, Y0 + .02, 1.03), (.05, .06, 2.1), st, r=.003)
    d.box("/Dressing/door_kitchen/frame_t", ((x0 + x1)/2, Y0 + .02, 2.075), (x1 - x0 + .1, .06, .05), st, r=.003)
    d.box("/Dressing/door_kitchen/leaf", ((x0 + x1)/2, Y0 + .025, 1.025), (x1 - x0 - .01, .044, 2.04), white, r=.004)
    d.box("/Dressing/door_kitchen/kick", ((x0 + x1)/2, Y0 + .048, .16), (x1 - x0 - .03, .002, .30), st, r=.001)
    d.box("/Dressing/door_kitchen/push", (x0 + .12, Y0 + .048, 1.15), (.10, .002, .30), st, r=.001)
    d.rest("kitchen_door", 0., "floor", 0.)
    # walk-in chiller: insulated door (100 mm, white) swung open ~165 deg along the wall, PVC strip curtain
    y0, y1 = .25, 1.15
    fr = d.preview("chiller_frame", (242, 242, 240), .4)
    d.box("/Dressing/chiller/frame_a", (X1 - .03, y0 - .04, 1.02), (.10, .08, 2.04), fr, r=.004)
    d.box("/Dressing/chiller/frame_b", (X1 - .03, y1 + .04, 1.02), (.10, .08, 2.04), fr, r=.004)
    d.box("/Dressing/chiller/frame_t", (X1 - .03, (y0 + y1)/2, 2.04), (.10, y1 - y0 + .16, .08), fr, r=.004)
    ang = math.radians(165.)
    hinge = np.array([X1 - .11, y0 - .01])
    leaf_len = y1 - y0 - .02
    # leaf built closed (along +y from the hinge, 100 mm thick toward -x) then rotated about the hinge
    lv, lf = rbox_mesh((.10, leaf_len, 1.98), .01, center=(-.05, leaf_len/2, 1.0))   # closed: in the opening
    lv = rotz(lv + np.array([hinge[0], hinge[1], 0]), ang, about=hinge)
    d.mesh("/Dressing/chiller/door_leaf", lv, lf, white)
    kick = rbox_mesh((.004, leaf_len - .04, .30), .001, center=(-.102, leaf_len/2, .17))
    d.mesh("/Dressing/chiller/door_kick", rotz(kick[0] + np.array([hinge[0], hinge[1], 0]), ang, about=hinge), kick[1], st)
    handle = rbox_mesh((.05, .03, .26), .008, center=(-.13, leaf_len - .08, 1.05))
    d.mesh("/Dressing/chiller/door_handle", rotz(handle[0] + np.array([hinge[0], hinge[1], 0]), ang, about=hinge),
           handle[1], d.steel("chrome", .12, srgb=(200, 202, 205)))
    for k, z in enumerate((.25, 1.75)):
        d.box(f"/Dressing/chiller/hinge_{k}", (hinge[0] + .02, hinge[1] - .01, z), (.06, .04, .16),
              d.steel("chrome", .12, srgb=(200, 202, 205)), r=.006)
    d.rest("chiller_door", 0., "floor", 0.)
    # strip curtain: 200 x 2 mm clear PVC strips, 50 % overlap (maker class, remembered), on a rail at 2.0 m
    pvc = d.glass("pvc_strip", srgb=(228, 240, 238), ior=1.54, frost=.03, thin=True)
    rng = np.random.default_rng(4)
    xc = X1 + .03
    d.box("/Dressing/chiller/strip_rail", (xc, (y0 + y1)/2, 1.99), (.03, y1 - y0, .03), st, r=.003)
    for k, yc in enumerate(np.arange(y0 + .10, y1 - .05, .10)):
        tw = rng.normal(0, .06)
        sv, sf = box_mesh((.002, .20, 1.96), (0, 0, .99))
        sv = rotz(sv, tw) + np.array([xc + (.003 if k % 2 else -.003), yc, 0.])
        d.mesh(f"/Dressing/chiller/strip_{k}", sv, sf, pvc, double=True)
    prov("chiller_door", "insulated 100 mm leaf, 900 x 2000 mm opening", "estimate", "walk-in cold room door class")
    # chiller interior (visible through the strips): panels, wire shelving, fish boxes, cold light
    cr = d.preview("coldroom", texture=d.tex["coldroom"], rough=.4)
    IX0, IX1, IY0, IY1, IH = X1, X1 + 1.8, -.25, 1.65, 2.3
    walls = [((IX1, IY0, 0), (IX1, IY1, 0), (IX1, IY1, IH), (IX1, IY0, IH)),
             ((IX0, IY1, 0), (IX1, IY1, 0), (IX1, IY1, IH), (IX0, IY1, IH)),
             ((IX1, IY0, 0), (IX0, IY0, 0), (IX0, IY0, IH), (IX1, IY0, IH)),
             ((IX0, IY0, IH), (IX1, IY0, IH), (IX1, IY1, IH), (IX0, IY1, IH)),
             ((IX0, IY1, 2.0), (IX0, y1, 2.0), (IX0, y1, IH), (IX0, IY1, IH))]
    for k, q in enumerate(walls):
        V = np.array(q, float)
        uvs = np.array([(np.dot(v, [0, 1, 0]) + np.dot(v, [1, 0, 0]), v[2]) for v in V])/1.2
        d.mesh(f"/Dressing/chiller/interior_{k}", V, np.array([(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)]), cr,
               uv=uvs, normals=None)
    fl = d.preview("coldroom_floor", (120, 122, 124), .7)
    V = np.array([(IX0, IY0, 0), (IX1, IY0, 0), (IX1, IY1, 0), (IX0, IY1, 0)])
    d.mesh("/Dressing/chiller/interior_floor", V, np.array([(0, 1, 2), (0, 2, 3)]), fl, normals=None)
    wire = d.steel("wire_shelf", .3, srgb=(180, 182, 184))
    for k, z in enumerate((.25, .75, 1.25, 1.75)):
        d.box(f"/Dressing/chiller/shelf_{k}", (IX1 - .30, (IY0 + IY1)/2, z - .01), (.5, 1.6, .02), wire, r=.002)
    for k in range(4):
        for jj, yy in enumerate((IY0 + .05, IY1 - .05)):
            d.box(f"/Dressing/chiller/post_{k}_{jj}", (IX1 - .30 + (-.24 if k % 2 else .24), yy, .95), (.025, .025, 1.9), wire, r=.004)
    # fish boxes placed on the shelves (re-authored with translation)
    for k, z in enumerate((.25, .75, 1.25)):
        for j, yy in enumerate((.10, .80)):
            p = f"/Dressing/chiller/fishbox_{k}_{j}"
            d.xform(p, (IX1 - .30, yy, z))
            build_fish_box(d, p, size=(.40, .60, .18), lid=(j == 1))
            d.rest(f"chiller_fishbox_{k}_{j}", z, "wire shelf", z)
    Lt = d.UsdLux.RectLight.Define(d.stage, "/Dressing/chiller/light")
    Lt.CreateWidthAttr(1.2)
    Lt.CreateHeightAttr(.6)
    Lt.CreateIntensityAttr(4500.)
    Lt.CreateEnableColorTemperatureAttr(True)
    Lt.CreateColorTemperatureAttr(6000.)
    d.UsdGeom.Xformable(Lt).AddTranslateOp().Set(d.Gf.Vec3d((IX0 + IX1)/2, (IY0 + IY1)/2, IH - .02))
    sign = d.preview("sign_chiller", texture=d.tex["sign_chiller"], rough=.4)
    q = np.array([(X1 - .003, 1.53, 1.45), (X1 - .003, 1.28, 1.45), (X1 - .003, 1.28, 1.61), (X1 - .003, 1.53, 1.61)])
    d.mesh("/Dressing/chiller/sign", q, np.array([(0, 1, 2), (0, 2, 3)]), sign, uv=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
           normals=None)


def build_bench(d, skip=()):
    """Stainless worktable (layout TABLE) with undershelf, the robot pedestal and the undershelf stock."""
    st = d.steel()
    T = L.TABLE
    (cx, cy), (sx, sy) = T["center"], T["size"]
    tz, tt, leg = T["top_z"], T["top_thickness"], T["leg"]
    d.box("/Dressing/bench/top", (cx, cy, tz - tt/2), (sx, sy, tt), st, r=.004)
    feet = d.preview("feet_black", (30, 30, 30), .6)
    for i, x in enumerate((cx - sx/2 + .04, cx + sx/2 - .04)):
        for j, y in enumerate((cy - sy/2 + .04, cy + sy/2 - .04)):
            d.box(f"/Dressing/bench/leg_{i}{j}", (x, y, (tz - tt + .035)/2), (leg, leg, tz - tt - .035), st, r=.004)
            d.cyl(f"/Dressing/bench/foot_{i}{j}", (x, y, 0.), (x, y, .035), .019, feet)
    sz_ = T["undershelf_z"]
    d.box("/Dressing/bench/undershelf", (cx, cy, sz_ - .015), (sx - .10, sy - .10, .03), st, r=.003)
    for j, y in enumerate((cy - sy/2 + .04, cy + sy/2 - .04)):
        d.box(f"/Dressing/bench/rail_{j}", (cx, y, tz - tt - .02), (sx - .12, .03, .04), st, r=.003)
    d.rest("bench", 0., "floor", 0.)
    prov("bench", "1800 x 750 mm, top 0.900 m, 40 mm top, 40 mm legs, undershelf at 0.25 m", "layout", "TABLE")
    # undershelf stock: nested GN 1/1 pans, glove boxes, a case of centre-feed rolls
    pan_steel = d.steel("pan_steel", .22)
    p = "/Dressing/under/gn_stack"
    d.xform(p, (cx - .45, cy, sz_))
    build_gn_stack(d, p, (.530, .325), .065, 4, pan_steel)
    d.rest("undershelf_gn_stack", sz_, "undershelf", sz_)
    gl = d.preview("gloves_box", texture=d.tex["gloves_blue"], rough=.6)
    for k in range(2):
        V, F = box_mesh((.24, .125, .09), (cx + .20, cy - .10 + .15*k, sz_ + .045))
        uv = box_uv(V, F)*np.array([1/.24, 1/.125])
        d.mesh(f"/Dressing/under/gloves_{k}", V, F, gl, uv=uv)
    d.rest("undershelf_gloves", sz_, "undershelf", sz_)
    blue = d.preview("blue_roll", (40, 110, 200), .8)
    core = d.preview("roll_core", (170, 130, 90), .8)
    for k, y in enumerate((cy - .10, cy + .13)):
        V, F = lathe([(.02, sz_), (.10, sz_), (.10, sz_ + .25), (.02, sz_ + .25), (.02, sz_)], 48)
        d.mesh(f"/Dressing/under/roll_{k}", V + np.array([cx + .62, y, 0]), F, blue)
    d.rest("undershelf_rolls", sz_, "undershelf", sz_)
    # robot pedestal: welded stainless box, top plate, floor plate with anchors, e-stop box
    (px, py), (psx, psy, psz) = L.PEDESTAL["center"], L.PEDESTAL["size"]
    ymin, ymax = py - psy/2, py + psy/2
    table_front = cy - sy/2
    if ymax > table_front:
        d.adjustments.append(dict(item="PEDESTAL", drawn=f"y {ymin:.3f}..{table_front - .002:.3f}",
                                  layout=f"y {ymin:.3f}..{ymax:.3f}",
                                  reason=f"layout pedestal overlaps the bench top slab by {1e3*(ymax - table_front):.0f} mm; "
                                         "drawn stopping 2 mm short of the bench front edge"))
        ymax = table_front - .002
    pc = ((ymin + ymax)/2)
    d.box("/Dressing/pedestal/body", (px, pc, (psz - .012 + .010)/2), (psx - .01, ymax - ymin - .01, psz - .012 - .010), st, r=.004)
    d.box("/Dressing/pedestal/top_plate", (px, pc, psz - .006), (psx, ymax - ymin, .012), st, r=.002)
    d.box("/Dressing/pedestal/floor_plate", (px, pc, .005), (psx + .06, ymax - ymin + .06, .010), st, r=.002)
    bolt = d.steel("bolt", .3, srgb=(130, 132, 134))
    for i, x in enumerate((px - psx/2 - .015, px, px + psx/2 + .015)):
        for j, y in enumerate((ymin - .015, ymax + .015)):
            d.cyl(f"/Dressing/pedestal/anchor_{i}{j}", (x, y, .010), (x, y, .018), .009, bolt, n=6)
    for i, x in enumerate((px - .25, px + .25)):
        for j, z in enumerate((.25, .65)):
            d.cyl(f"/Dressing/pedestal/panel_screw_{i}{j}", (x, ymin + .004, z), (x, ymin - .001, z), .004, bolt, n=12)
    d.box("/Dressing/pedestal/access_panel", (px, ymin - .0005, .45), (.60, .002, .55), d.steel("panel_steel", .3), r=.001)
    yel = d.preview("estop_yellow", (240, 196, 20), .45)
    red = d.preview("estop_red", (200, 20, 18), .35)
    d.box("/Dressing/pedestal/estop_box", (px + .33, ymin - .03, .78), (.08, .06, .08), yel, r=.006)
    d.cyl("/Dressing/pedestal/estop_button", (px + .33, ymin - .06, .78), (px + .33, ymin - .085, .78), .02, red)
    d.rest("pedestal", 0., "floor", 0.)
    prov("pedestal", "900 x 300 x 900 mm welded stainless (layout PEDESTAL)", "layout", "robot bases at z = 0.900")


def build_bench_items(d, preset, skip=(), trout_positions=None, task=None):
    """Board, cloth, preset vessels and tools on the bench top at their layout positions (with `task`'s task-only
    props and arrangement: layout.FRAME_PAN, task_spec)."""
    tz = L.TABLE["top_z"]
    st = d.steel()
    out = {}
    # damp cloth under the board (off-white terry, 1 mm, layout cloth_thickness), wavy outline
    terry, _ = wiping_texture("terry_cloth")
    arr = np.asarray(Image.open(terry["Diffuse"]).convert("L"), np.float32)/255.
    arr = (arr - arr.mean())/max(arr.std(), 1e-6)
    cl = np.clip(np.array([.66, .70, .74])[None, None]*(1 + .08*arr[..., None]), 0, 1)   # damp grey-white
    cp = d.tex_dir/"damp_cloth_diff.jpg"
    Image.fromarray((cl*255).astype(np.uint8)).save(cp, quality=90)
    cm = d.omnipbr("damp_cloth", diff=cp, normal=terry["nor_gl"], tile_m=.367, rough_const=.45, bump=.8)
    d.credits["terry_cloth"] = "Poly Haven, CC0 (wiping/assets/sources.json), re-toned off-white"
    bx, by = L.BOARD["center"]
    bsx, bsy, bsz = L.BOARD["size"]
    rng = np.random.default_rng(11)
    # a folded kitchen cloth: 20-35 mm margin around the board, soft low-frequency wander of the edge
    ring = rrect(bsx/2 + .026, bsy/2 + .020, .04, 40)
    ang = np.arctan2(ring[:, 1], ring[:, 0])
    wob = 1 + .010*np.sin(3*ang + .7) + .006*np.sin(7*ang + 2.1) + .004*np.sin(11*ang + .3)
    ring = ring*wob[:, None]
    ct = L.BOARD["cloth_thickness"]
    top = np.column_stack([ring + [bx, by], np.full(len(ring), tz + ct)])
    bot = np.column_stack([ring + [bx, by], np.full(len(ring), tz + .0001)])
    V, F = loft([bot, top], cap0=True, cap1=True)
    d.mesh("/Dressing/bench/cloth", V, F, cm, angle=50.)
    d.rest("damp_cloth", tz, "bench top", tz)
    colour = "red" if preset == "C" else "blue"
    if "board" not in skip:
        p = "/Dressing/bench/board"
        d.xform(p, (bx, by, tz + ct + bsz/2))
        build_board(d, p, colour)
        d.rest("board", tz + ct, "damp cloth", tz + ct)
    prov("board", "610 x 457 x 12.7 mm NSF HDPE, blue (fish) / red (meat)", "layout", "BOARD")
    # fillet tray: perforated GN 1/2, 40 mm (layout FILLET_TRAY), standing on the bench
    if "fillet_tray" not in skip:
        TR = L.fillet_tray(task)
        (fx, fy), (fsx, fsy, fsz) = TR["center"], TR["size"]
        V, F, floor = gn_pan_geometry((fsx, fsy), fsz, t=TR["wall"])
        p = "/Dressing/bench/fillet_tray"
        d.xform(p, (fx, fy, tz))
        perf = d.preview("perforated_steel", texture=d.tex["perforation"], alpha_cut=True, rough=.25, metal=1.)
        tray = d.steel("pan_steel", .22)
        uvf = box_uv(V, F[floor])/.09
        d.mesh(p + "/walls", V, F[~floor], tray, angle=35.)
        d.mesh(p + "/floor", V, F[floor], perf, uv=uvf, normals=None, double=True)
        d.rest("fillet_tray", tz, "bench top", tz)
    prov("fillet_tray", "perforated GN 1/2, 325 x 265 x 40 mm, on the bench", "layout+standard", "FILLET_TRAY; EN 631 1/2")
    # bowl (frames and heads)
    if "bowl" not in skip:
        V, F = bowl_geometry()
        c = L.BOWL["center"]
        d.mesh("/Dressing/bench/bowl", V + np.array([c[0], c[1], tz]), F, d.steel("bowl_steel", .12, srgb=(196, 198, 200)), angle=40.)
        d.rest("bowl", tz, "bench top", tz)
    prov("bowl", "240 mm rim, 80 mm deep stainless bowl", "layout", "BOWL")
    # fillet-whole-fish only: the empty solid GN 1/2 pan for the frame and head (its colliders: scene.tapered_pan)
    if task in L.FRAME_PAN["tasks"] and "frame_pan" not in skip:
        G = L.FRAME_PAN
        (gx, gy), (gsx, gsy, gsz) = G["center"], G["size"]
        V, F, _ = gn_pan_geometry((gsx, gsy), gsz, rim=G["rim"], taper=G["taper"], t=G["wall"])
        p = "/Dressing/bench/frame_pan"
        d.xform(p, (gx, gy, tz))
        d.mesh(p + "/pan", V, F, d.steel("pan_steel", .22), angle=35.)
        d.rest("frame_pan", tz, "bench top", tz)
        prov("frame_pan", "solid GN 1/2, 325 x 265 x 100 mm, empty, on the bench (fillet-whole-fish only)",
             "layout+standard", "FRAME_PAN; EN 631 1/2")
    # knife block with the knives handle up (a physics-supplied knife body replaces the one drawn here)
    if "knife_block" not in skip:
        kc = L.KNIFE_BLOCK["center"]
        p = "/Dressing/bench/knife_block"
        d.xform(p, (kc[0], kc[1], tz))
        kinds = tuple(k for k in ("fillet", "boning") if f"knife_{k}" not in skip and "knife" not in skip)
        out["block_knife_poses"] = build_knife_block(d, p, kinds=kinds)
        for kind, mm in d.block_fit_mm.items():
            if mm <= 1.:
                d.adjustments.append(dict(item="KNIFE_BLOCK", drawn=f"no {kind} knife in the block",
                                          layout="100 x 160 x 120 mm block, knives handle up",
                                          reason=f"the {kind} blade does not fit inside the layout block at any slot "
                                                 f"position along its diagonal (best clearance {mm:.1f} mm); a real "
                                                 "block for an 18 cm blade is taller/deeper"))
        d.rest("knife_block", tz, "bench top", tz)
    prov("knife_block", "100 x 160 x 120 mm", "layout", "KNIFE_BLOCK; diagonal slots authored")
    prov("fillet_knife", "blade 173 x 25 mm, spine 1.2 mm, 18 deg bevel; handle 140 x 32 x 19 mm", "params",
         "FILLET_BLADE, HANDLE (Victorinox Fibrox 5.3703.18 / 5.3813.18)")
    prov("boning_knife", "blade 150 x 20 mm, spine 2.0 mm, 18 deg bevel", "params", "BONING_BLADE (5.6303.15)")
    # preset A: GN 1/1 of crushed ice with two whole trout; B/C: bench portion scale in the same place
    if preset in L.ICE_PAN.get("presets", ("A",)) and "ice_pan" not in skip:
        build_ice_pan(d, trout_positions, center=L.ice_pan_center(task))
    if preset in L.SCALE.get("presets", ("B", "C")) and "scale" not in skip:
        (sx_, sy_) = L.SCALE["center"]
        p = "/Dressing/bench/scale"
        d.xform(p, (sx_, sy_, tz))
        out["scale_lcd"] = build_scale(d, p)
        d.rest("scale", tz, "bench top", tz)
    prov("scale", "300 x 250 mm platform, 45 mm, 6 kg x 1 g class", "layout", "SCALE; display tab authored")
    if preset == "B" and "tweezer_tray" not in skip:
        (tx_, ty_), (tsx, tsy, tsz) = L.TWEEZER_TRAY["center"], L.TWEEZER_TRAY["size"]
        board_x1 = bx + bsx/2
        if tx_ - tsx/2 < board_x1 and ty_ + tsy/2 > by - bsy/2:
            nx = board_x1 + .005 + tsx/2
            d.adjustments.append(dict(item="TWEEZER_TRAY", drawn=f"centre x {nx:.3f}", layout=f"centre x {tx_:.3f}",
                                      reason=f"layout tray overlaps the board by {1e3*(board_x1 - (tx_ - tsx/2)):.0f} mm in x; "
                                             "drawn 5 mm clear of the board's right edge"))
            tx_ = nx
        (fx_, fy_), (fsx_, fsy_, _) = L.FILLET_TRAY["center"], L.FILLET_TRAY["size"]
        if ty_ + tsy/2 > fy_ - fsy_/2 and abs(tx_ - fx_) < (tsx + fsx_)/2:
            ny = fy_ - fsy_/2 - .005 - tsy/2
            d.adjustments.append(dict(item="TWEEZER_TRAY", drawn=f"centre y {ny:.4f}", layout=f"centre y {ty_:.3f}",
                                      reason=f"layout tray overlaps the fillet tray by {1e3*(ty_ + tsy/2 - (fy_ - fsy_/2)):.1f} mm "
                                             "in y; drawn 5 mm clear of it"))
            ty_ = ny
        p = "/Dressing/bench/tweezer_tray"
        d.xform(p, (tx_, ty_, tz))
        build_tweezer_tray(d, p)
        if "tweezers" not in skip:
            d.xform(p + "/tweezers_pose", (-.06, 0., .0008 + .012 - .0006), quat=(.7071068, 0., 0., .7071068))
            build_tweezers(d, p + "/tweezers_pose")
        d.rest("tweezer_tray", tz, "bench top", tz)
    prov("tweezer_tray", "200 x 100 x 15 mm", "layout", "TWEEZER_TRAY")
    prov("tweezers", "120 mm, 11 mm arms, 24 mm open", "read", "Victorinox 7.6120")
    # honing steel lying behind the board, sanitiser spray (back left), centre-feed roll (right end)
    p = "/Dressing/bench/honing_steel"
    d.xform(p, (-.12, .66, tz), yaw=math.radians(4.))
    build_honing_steel(d, p)
    d.rest("honing_steel", tz, "bench top (guard rim and rod tip)", tz)
    prov("honing_steel", "260 mm rod, 40 mm guard, 125 mm handle", "estimate", "25-30 cm honing steel class")
    p = "/Dressing/bench/spray"
    d.xform(p, (-.80, .68, tz), yaw=math.radians(-25))
    build_spray_bottle(d, p)
    d.rest("spray_bottle", tz, "bench top", tz)
    blue = d.preview("blue_roll", (40, 110, 200), .8)
    V, F = lathe([(.025, 0.), (.10, 0.), (.10, .25), (.025, .25), (.025, .001)], 48)
    d.mesh("/Dressing/bench/blue_roll", V + np.array([.78, .64, tz]), F, blue)
    d.rest("blue_roll", tz, "bench top", tz)
    return out


def build_ice_pan(d, trout_positions=None, center=None):
    """GN 1/1 65 mm stainless pan (layout ICE_PAN, EN 631 1/1) of crushed ice with two whole trout (stand-ins);
    `center`: the task's position (layout.ice_pan_center), default the layout's."""
    tz = L.TABLE["top_z"]
    (cx, cy), (sx, sy, sz) = center or L.ICE_PAN["center"], L.ICE_PAN["size"]
    V, F, _ = gn_pan_geometry((sx, sy), sz)
    p = "/Dressing/bench/ice_pan"
    d.xform(p, (cx, cy, tz))
    d.mesh(p + "/pan", V, F, d.steel("pan_steel", .22), angle=35.)
    d.rest("ice_pan", tz, "bench top", tz)
    # ice: a heaped slush surface inside the pan + instanced crushed-ice chunks on top
    rng = np.random.default_rng(21)
    fill = .050
    ins = .012 + .018*(1 - fill/sz) + .002
    hx, hy = sx/2 - ins, sy/2 - ins
    nx_, ny_ = 70, 44
    gx, gy = np.meshgrid(np.linspace(-hx, hx, nx_), np.linspace(-hy, hy, ny_))
    rr = rrect(hx, hy, .02, 8)
    # clamp grid corners into the rounded rectangle
    r = .02
    ex = np.clip(np.abs(gx) - (hx - r), 0, None)
    ey = np.clip(np.abs(gy) - (hy - r), 0, None)
    e = np.hypot(ex, ey)
    k = np.where(e > r, r/np.maximum(e, 1e-9), 1.)
    gx = np.sign(gx)*((hx - r) + ex*k) * (e > 0) + gx*(e == 0)
    gy = np.sign(gy)*((hy - r) + ey*k) * (e > 0) + gy*(e == 0)
    noise = np.asarray(Image.fromarray((rng.uniform(0, 255, (ny_, nx_))).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2.)), np.float32)/255.
    edge = np.minimum((hx - np.abs(gx))/.03, (hy - np.abs(gy))/.03).clip(0, 1)
    gz = .0008 + fill - .004 + .010*(noise - .5) + .004*edge
    Vg = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    Fg = []
    for j in range(ny_ - 1):
        for i in range(nx_ - 1):
            a = j*nx_ + i
            Fg += [(a, a + 1, a + nx_ + 1), (a, a + nx_ + 1, a + nx_)]
    slush = d.omnipbr("ice_slush", srgb=(230, 235, 237), rough_const=.22)
    d.mesh(p + "/ice_bed", Vg, np.asarray(Fg), slush, angle=50.)

    def ice_h(x, y):
        i = np.clip(((x + hx)/(2*hx)*(nx_ - 1)).astype(int), 0, nx_ - 1)
        j = np.clip(((y + hy)/(2*hy)*(ny_ - 1)).astype(int), 0, ny_ - 1)
        return gz[j, i]
    # two trout lying on the ice, nestled 3 mm into the crushed ice (local pan frame)
    fish = trout_positions or [((-.005, -.065), math.radians(4.)), ((.012, .070), math.radians(-5.))]
    skin = d.tissue_material("standin_trout_skin", "skin", skin_texture=d.tex["trout_skin"])
    finm = d.preview("trout_fin", texture=d.tex["trout_fin"], rough=.35, coat=.6, opacity=.8)
    eye = d.preview("trout_eye", (190, 170, 120), .15, metal=.6)
    pupil = d.preview("trout_pupil", (5, 5, 6), .05, coat=1.)
    body = trout_body()
    foot = []
    for kf, ((fx, fy), yaw) in enumerate(fish):
        zb = float(np.median(ice_h(np.full(9, fx) + np.linspace(-.1, .1, 9), np.full(9, fy)))) - .003
        build_trout(d, p + f"/trout_{kf}", (fx, fy, zb), yaw, skin, finm, (eye, pupil))
        d.rest(f"standin_trout_{kf}", tz + zb, "crushed ice (nestled 3 mm)", tz + zb + .003)
        Vw = rotz(body["V"], yaw) + np.array([fx, fy, 0])
        foot.append(Vw[:, :2])
    prov("standin_trout_on_ice", "31.5 cm, 7.5 cm deep, 3.6 cm wide", "params(TROUT)+authored",
         "STAND-IN visual mesh and procedural skin; not the anatomy generator")
    # chunks: 4 prototypes, instanced over the ice bed away from the fish
    from scipy.spatial import ConvexHull
    protos = []
    for kp in range(5):
        pts = rng.normal(0, 1, (14, 3))*np.array([1., .8, .55])
        pts *= .005/np.abs(pts).max()
        hull = ConvexHull(pts)
        Fh = hull.simplices.copy()
        cen = pts.mean(0)
        for fi, tri in enumerate(Fh):
            nrm = np.cross(pts[tri[1]] - pts[tri[0]], pts[tri[2]] - pts[tri[0]])
            if np.dot(nrm, pts[tri[0]] - cen) < 0:
                Fh[fi] = tri[::-1]
        protos.append((pts, Fh))
    UsdGeom, Vt, Gf = d.UsdGeom, d.Vt, d.Gf
    inst = UsdGeom.PointInstancer.Define(d.stage, p + "/ice_chunks")
    ice = d.glass("crushed_ice", srgb=(236, 246, 252), ior=1.31, frost=.22, thin=False)
    for kp, (pts, Fh) in enumerate(protos):
        d.mesh(p + f"/ice_chunks/proto_{kp}", pts, Fh, ice, angle=5.)
    inst.CreatePrototypesRel().SetTargets([f"{p}/ice_chunks/proto_{kp}" for kp in range(len(protos))])
    n = 2600
    xs = rng.uniform(-hx + .006, hx - .006, n)
    ys = rng.uniform(-hy + .006, hy - .006, n)
    from matplotlib.path import Path as MPath
    keep = np.ones(n, bool)
    for fp in foot:
        from scipy.spatial import ConvexHull as CH
        hull = CH(fp)
        poly = MPath(fp[hull.vertices])
        keep &= ~poly.contains_points(np.stack([xs, ys], 1), radius=-.004)
    xs, ys = xs[keep], ys[keep]
    zs = ice_h(xs, ys) + rng.uniform(-.001, .003, len(xs))
    sc = rng.uniform(.8, 1.9, len(xs))
    from scipy.spatial.transform import Rotation
    qs = Rotation.random(len(xs), random_state=5).as_quat()
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.stack([xs, ys, zs], 1).astype(np.float32)))
    inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.repeat(sc[:, None], 3, 1).astype(np.float32)))
    inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(q[3]), float(q[0]), float(q[1]), float(q[2])) for q in qs]))
    inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(rng.integers(0, len(protos), len(xs)).astype(np.int32)))
    prov("crushed_ice", f"{len(xs)} instanced chunks 5-19 mm on a slush bed 50 mm deep", "authored",
         "flake/crushed ice appearance; visual only")


def build_walls_fixtures(d, preset):
    st = d.steel()
    # --- back wall, left: two stainless wall shelves with GN pans, film and glove boxes
    for k, z in enumerate((1.40, 1.75)):
        d.box(f"/Dressing/shelves/shelf_{k}", (-1.50, Y1 - .15, z - .015), (1.10, .30, .03), st, r=.003)
        d.box(f"/Dressing/shelves/upstand_{k}", (-1.50, Y1 - .006, z + .02), (1.10, .012, .07), st, r=.002)
        for j, x in enumerate((-1.98, -1.50, -1.02)):
            V = np.array([(x - .006, Y1, z - .03), (x + .006, Y1, z - .03), (x + .006, Y1 - .26, z - .03),
                          (x - .006, Y1 - .26, z - .03), (x - .006, Y1, z - .25), (x + .006, Y1, z - .25)])
            F = np.array([(0, 2, 1), (0, 3, 2), (0, 4, 3), (1, 2, 5), (0, 1, 5), (0, 5, 4), (3, 4, 5), (3, 5, 2)])
            d.mesh(f"/Dressing/shelves/bracket_{k}{j}", V, F, st, angle=10.)
    pan_steel = d.steel("pan_steel", .22)
    for k, (x, z, size, depth, cnt) in enumerate(((-1.78, 1.40, (.530, .325), .065, 5), (-1.25, 1.40, (.325, .265), .100, 3),
                                                  (-1.80, 1.75, (.325, .265), .065, 4), (-1.45, 1.75, (.325, .176), .065, 3))):
        p = f"/Dressing/shelves/gn_{k}"
        d.xform(p, (x, Y1 - .165, z), yaw=math.pi/2 if size[0] > .4 else 0.)
        build_gn_stack(d, p, size, depth, cnt, pan_steel)
        d.rest(f"shelf_gn_stack_{k}", z, "wall shelf", z)
    film = d.preview("film_box", texture=d.tex["film_box"], rough=.6)
    V, F = box_mesh((.47, .095, .095), (-1.18, Y1 - .12, 1.75 + .0475))
    d.mesh("/Dressing/shelves/film_box", V, F, film, uv=box_uv(V, F)*np.array([1/.47, 1/.095]))
    gl = d.preview("gloves_box", texture=d.tex["gloves_blue"], rough=.6)
    for k in range(3):
        V, F = box_mesh((.24, .125, .09), (-1.10 + .0*k, Y1 - .10, 1.40 + .045 + .09*k))
        d.mesh(f"/Dressing/shelves/gloves_{k}", V, F, gl, uv=box_uv(V, F)*np.array([1/.24, 1/.125]))
    d.rest("shelf_boxes", 1.40, "wall shelf", 1.40)
    prov("wall_shelves", "2 x 1100 x 300 mm stainless shelves on brackets at 1.40/1.75 m", "estimate", "catering wall shelf class")
    # --- back wall: UV knife steriliser cabinet (wall-mounted), knives hanging inside behind a glass door
    ux, uz0, uw, uh, ud = -.35, 1.18, .50, .52, .16
    body = d.steel("cabinet_steel", .28)
    d.box("/Dressing/uv_cabinet/back", (ux, Y1 - .006, uz0 + uh/2), (uw, .012, uh), body, r=.002)
    for k, xx in enumerate((ux - uw/2 + .006, ux + uw/2 - .006)):
        d.box(f"/Dressing/uv_cabinet/side_{k}", (xx, Y1 - ud/2, uz0 + uh/2), (.012, ud, uh), body, r=.002)
    for k, zz in enumerate((uz0 + .006, uz0 + uh - .006)):
        d.box(f"/Dressing/uv_cabinet/tb_{k}", (ux, Y1 - ud/2, zz), (uw, ud, .012), body, r=.002)
    d.box("/Dressing/uv_cabinet/header", (ux, Y1 - ud + .01, uz0 + uh - .04), (uw, .02, .08), body, r=.002)
    inner = d.preview("uv_inner", (225, 228, 232), .3)
    d.box("/Dressing/uv_cabinet/inner_back", (ux, Y1 - .014, uz0 + uh/2), (uw - .03, .004, uh - .03), inner, r=0.)
    glass = d.glass("cabinet_glass", srgb=(240, 248, 250), ior=1.5, frost=.0, thin=True)
    V, F = box_mesh((uw - .03, .004, uh - .10), (ux, Y1 - ud + .004, uz0 + (uh - .10)/2 + .015))
    d.mesh("/Dressing/uv_cabinet/door_glass", V, F, glass, double=True)
    uvt = d.omnipbr("uv_tube", srgb=(170, 190, 255), emissive=(.55, .6, 1.), emissive_intensity=600., rough_const=.3)
    d.cyl("/Dressing/uv_cabinet/uv_tube", (ux - .20, Y1 - .08, uz0 + uh - .06), (ux + .20, Y1 - .08, uz0 + uh - .06), .008, uvt)
    d.box("/Dressing/uv_cabinet/magnet_bar", (ux, Y1 - .020, uz0 + .29), (.42, .012, .03), d.preview("magnet_bar", (30, 30, 32), .5), r=.002)
    from scipy.spatial.transform import Rotation
    km = knife_materials(d)
    for k, (kind, xx) in enumerate((("fillet", -.15), ("boning", -.05), ("fillet", .05), ("boning", .15))):
        # hanging handle up from the rail: blade x-axis pointing down (-z), flat facing +y
        Rm = np.column_stack([[0, 0, -1.], [0, 1., 0], [1., 0, 0]])
        q = Rotation.from_matrix(Rm).as_quat()
        g = knife_geometry(kind)
        pth = f"/Dressing/uv_cabinet/knife_{k}"
        # blade flat against the magnetic bar's front face (y = Y1 - 0.026), heel 30 mm above the bar centre
        d.xform(pth, (ux + xx - g["zc"], Y1 - .026 - g["spec"].t/2, uz0 + .32), quat=q)
        build_knife(d, pth, kind, mats=km)
    prov("uv_knife_steriliser", "500 x 160 x 520 mm wall cabinet", "estimate", "UV knife steriliser cabinet class")
    # --- back wall: colour-coded board rack (wall-mounted), boards 450 x 300 x 12.5 mm
    rx, rz = .45, 1.12
    d.box("/Dressing/board_rack/tray", (rx, Y1 - .07, rz), (.36, .13, .012), st, r=.002)
    d.box("/Dressing/board_rack/backplate", (rx, Y1 - .005, rz + .06), (.36, .01, .14), st, r=.002)
    colours = ["red" if preset != "C" else "blue", "yellow", "green", "white", "brown"]
    for k, cname in enumerate(colours):
        x = rx - .14 + k*.07
        d.box(f"/Dressing/board_rack/divider_{k}", (x - .035, Y1 - .07, rz + .04), (.004, .12, .07), st, r=.001)
        m = d.omnipbr(f"hdpe_{cname}", diff=d.tex[f"board_{cname}"], tile_m=1., rough_const=.55)
        V, F = rbox_mesh((.0125, .30, .45), .002, center=(x, Y1 - .075, rz + .006 + .225))
        V = rotz(V - np.array([x, Y1 - .075, 0]), 0.) + np.array([x, Y1 - .075, 0])
        tilt = np.eye(3)                                   # upright in the rack slots
        c0 = np.array([x, Y1 - .075, rz + .006])
        V = (V - c0) @ tilt.T + c0
        uv = box_uv(V, F)/np.array([.3, .45])
        d.mesh(f"/Dressing/board_rack/board_{cname}", V, F, m, uv=uv)
    d.rest("board_rack_boards", rz + .006, "rack tray", rz + .006)
    prov("board_rack", "wall rack with 5 colour-coded 450 x 300 x 12.5 mm boards", "estimate", "HACCP colour boards")
    fish_sign = d.preview("sign_fish", texture=d.tex["sign_fish"], rough=.4)
    q = np.array([(rx - .16, Y1 - .003, 1.62), (rx + .16, Y1 - .003, 1.62), (rx + .16, Y1 - .003, 1.82), (rx - .16, Y1 - .003, 1.82)])
    d.mesh("/Dressing/board_rack/sign", q, np.array([(0, 2, 1), (0, 3, 2)]), fish_sign,
           uv=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]), normals=None)
    # --- back wall: centre-feed paper roll dispenser
    white = d.preview("abs_white", (238, 238, 236), .35)
    d.box("/Dressing/paper_dispenser/body", (1.00, Y1 - .125, 1.33), (.26, .25, .34), white, r=.03)
    win = d.preview("dispenser_window", (90, 140, 210), .1, opacity=.6)
    d.box("/Dressing/paper_dispenser/window", (1.00, Y1 - .256, 1.38), (.14, .004, .14), win, r=.002)
    paper = d.preview("blue_paper", (60, 125, 205), .85)
    Vp = np.array([(.985, Y1 - .13, 1.16), (1.015, Y1 - .13, 1.16), (1.02, Y1 - .128, 1.08), (.98, Y1 - .132, 1.08)])
    d.mesh("/Dressing/paper_dispenser/paper_tail", Vp, np.array([(0, 1, 2), (0, 2, 3)]), paper, double=True, normals=None)
    prov("paper_dispenser", "260 x 250 x 340 mm centre-feed", "estimate", "blue-roll dispenser class")
    # --- robot cell sign above the tile line
    sg = d.preview("sign_robot", texture=d.tex["sign_robot"], rough=.4)
    q = np.array([(-.30, Y1 - .003, 2.12), (.30, Y1 - .003, 2.12), (.30, Y1 - .003, 2.52), (-.30, Y1 - .003, 2.52)])
    d.mesh("/Dressing/signs/robot", q, np.array([(0, 2, 1), (0, 3, 2)]), sg, uv=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
           normals=None)
    # --- back wall, right: knee-operated hand-wash basin, soap and towel dispensers, sign, bin
    build_handwash(d)
    # --- ceiling: evaporator (unit cooler) of the chilled room
    ev = d.preview("evaporator_white", (236, 236, 232), .4)
    ex, ey, ez = 1.20, Y1 - .30, H - .17
    d.box("/Dressing/evaporator/casing", (ex, ey, ez), (1.05, .45, .30), ev, r=.01)
    for k, dx in enumerate((-.25, .25)):
        V, F = lathe([(.14, 0.), (.16, .002), (.16, .012), (.14, .012), (.14, 0.)], 48)
        Rm = rot_matrix((1, 0, 0), math.pi/2)
        d.mesh(f"/Dressing/evaporator/fan_ring_{k}", V @ Rm.T + np.array([ex + dx, ey - .225, ez]), F,
               d.preview("fan_ring", (60, 60, 62), .5))
        grille = d.preview("fan_grille", (30, 30, 32), .5)
        for j in range(5):
            r = .03 + .025*j
            V, F = tube([(ex + dx + r*math.cos(a), ey - .232, ez + r*math.sin(a)) for a in np.linspace(0, 2*math.pi, 49)], .0015, n=6, caps=False)
            d.mesh(f"/Dressing/evaporator/grille_{k}_{j}", V, F, grille, angle=60.)
    for k, dx in enumerate((-.45, .45)):
        d.cyl(f"/Dressing/evaporator/hanger_{k}", (ex + dx, ey, ez + .15), (ex + dx, ey, H), .006, st)
    prov("evaporator", "1050 x 450 x 300 mm ceiling unit cooler, 2 fans", "estimate", "chilled prep room unit cooler class")
    # --- left wall: freestanding 4-tier stainless shelving with fish boxes and pans; ice bin; pedal bins
    sx0 = X0 + .25
    for k, z in enumerate((.18, .62, 1.06, 1.50)):
        d.box(f"/Dressing/rack/shelf_{k}", (sx0, -.30, z - .01), (.45, 1.20, .02), st, r=.002)
    for i, yy in enumerate((-.88, .28)):
        for j, xx in enumerate((sx0 - .20, sx0 + .20)):
            d.box(f"/Dressing/rack/post_{i}{j}", (xx, yy, .80), (.03, .03, 1.60), st, r=.004)
    for k, (z, lid) in enumerate(((.18, False), (.62, True))):
        pth = f"/Dressing/rack/fishbox_{k}"
        d.xform(pth, (sx0, -.30, z), yaw=math.pi/2)
        build_fish_box(d, pth, size=(.60, .40, .20), lid=lid)
        d.rest(f"rack_fishbox_{k}", z, "rack shelf", z)
    pth = "/Dressing/rack/gn"
    d.xform(pth, (sx0, -.55, 1.06))
    build_gn_stack(d, pth, (.325, .265), .150, 3, pan_steel)
    pth = "/Dressing/rack/gn2"
    d.xform(pth, (sx0, -.05, 1.06), yaw=math.pi/2)
    build_gn_stack(d, pth, (.530, .325), .065, 3, pan_steel)
    d.rest("rack_gn", 1.06, "rack shelf", 1.06)
    for k, y in enumerate((-.62, -.30, .02)):
        V, F = lathe([(.02, 1.50), (.10, 1.50), (.10, 1.75), (.02, 1.75), (.02, 1.501)], 40)
        d.mesh(f"/Dressing/rack/roll_{k}", V + np.array([sx0, y, 0]), F, d.preview("blue_roll", (40, 110, 200), .8))
    d.rest("rack_rolls", 1.50, "rack shelf", 1.50)
    prov("storage_rack", "4-tier 450 x 1200 x 1600 mm stainless shelving", "estimate", "catering shelving class")
    # ice bin (mobile ice caddy) under the wall shelves
    ib = d.preview("icebin_grey", (205, 208, 210), .45)
    lidb = d.preview("icebin_lid", (40, 90, 160), .4)
    icx, icy = -1.55, Y1 - .40
    d.box("/Dressing/ice_bin/body", (icx, icy, .06 + .34), (.55, .62, .68), ib, r=.03)
    d.box("/Dressing/ice_bin/lid", (icx, icy + .08, .76), (.55, .46, .04), lidb, r=.012)
    for i, dx in enumerate((-.22, .22)):
        for j, dy in enumerate((-.25, .25)):
            d.cyl(f"/Dressing/ice_bin/castor_{i}{j}", (icx + dx - .03, icy + dy, .032), (icx + dx + .03, icy + dy, .032), .032,
                  d.preview("castor", (30, 30, 30), .6))
    d.rest("ice_bin", 0., "floor (castors)", 0.)
    prov("ice_bin", "550 x 620 x 780 mm mobile ice caddy on castors", "estimate", "insulated ice bin class")
    # pedal bins: fish waste (blue lid) and general (black lid) at the bench's left end; towels by the basin
    for k, (x, y, lidc) in enumerate(((-1.28, .22, (30, 90, 180)), (-1.28, .66, (30, 30, 30)))):
        pth = f"/Dressing/bins/pedal_{k}"
        d.xform(pth, (x, y, 0.), yaw=-math.pi/2)
        build_pedal_bin(d, pth, lidc)
        d.rest(f"pedal_bin_{k}", 0., "floor", 0.)
    prov("pedal_bins", "360 mm dia x 650 mm, coloured lids", "estimate", "50 L pedal bin class")


def build_handwash(d):
    """Knee-operated wall-hung stainless hand-wash basin (estimate: 400 x 350 mm, rim 0.90 m)."""
    st = d.steel()
    bx, by = 1.95, Y1
    rim = .90
    # basin: rounded rings from the rim down into the bowl and back to the outer skirt
    rings = []
    for (hx, hy, r, z) in ((.20, .175, .02, rim), (.17, .145, .03, rim - .002), (.15, .125, .04, rim - .12),
                           (.14, .115, .04, rim - .135), (.15, .125, .04, rim - .145), (.20, .175, .02, rim - .16),
                           (.20, .175, .02, rim - .005)):
        ring = rrect(hx, hy, r, 8)
        rings.append(np.column_stack([ring + [bx, by - .175], np.full(len(ring), z)]))
    V, F = loft(rings[:4], cap1=True)
    d.mesh("/Dressing/handwash/bowl", V, F, st, angle=40.)
    V, F = loft(rings[4:] + [rings[0]], closed=True)
    d.mesh("/Dressing/handwash/skirt", V, F, st, angle=40.)
    d.box("/Dressing/handwash/splash", (bx, by - .01, rim + .06), (.40, .02, .12), st, r=.003)
    d.box("/Dressing/handwash/knee_panel", (bx, by - .31, rim - .32), (.30, .02, .30), st, r=.004)
    d.box("/Dressing/handwash/knee_panel_back", (bx, by - .17, rim - .32), (.30, .28, .02), st, r=.003)
    d.box("/Dressing/handwash/knee_pad", (bx, by - .325, rim - .40), (.12, .015, .05), d.preview("knee_pad", (40, 40, 40), .6), r=.006)
    d.box("/Dressing/handwash/wall_bracket", (bx, by - .01, rim - .20), (.36, .02, .16), st, r=.003)
    pts = [(bx, by - .04, rim + .10)] + [(bx, by - .04 - .08*math.sin(a), rim + .10 + .12*math.sin(a) + .02*(1 - math.cos(a)))
                                         for a in np.linspace(0, math.pi*.95, 16)]
    pts = [(bx, by - .02, rim + .10)] + [(bx, by - .04 - .09*(1 - math.cos(a)), rim + .12 + .14*math.sin(a))
                                         for a in np.linspace(0, math.pi*.92, 18)]
    V, F = tube(pts, .010, n=20)
    d.mesh("/Dressing/handwash/gooseneck", V, F, d.steel("chrome", .12, srgb=(200, 202, 205)), angle=40.)
    white = d.preview("abs_white", (238, 238, 236), .35)
    d.box("/Dressing/handwash/soap", (bx - .30, by - .055, 1.28), (.11, .11, .24), white, r=.015)
    d.box("/Dressing/handwash/soap_button", (bx - .30, by - .112, 1.22), (.06, .006, .04), d.preview("soap_btn", (40, 110, 190), .4), r=.004)
    d.box("/Dressing/handwash/towels", (bx + .32, by - .07, 1.38), (.28, .14, .38), white, r=.02)
    d.box("/Dressing/handwash/towel_slot", (bx + .32, by - .141, 1.21), (.20, .004, .012), d.preview("slot_dark", (12, 12, 12), .8), r=0.)
    sg = d.preview("sign_wash", texture=d.tex["sign_wash"], rough=.4)
    q = np.array([(bx - .12, Y1 - .003, 1.62), (bx + .12, Y1 - .003, 1.62), (bx + .12, Y1 - .003, 1.77), (bx - .12, Y1 - .003, 1.77)])
    d.mesh("/Dressing/handwash/sign", q, np.array([(0, 2, 1), (0, 3, 2)]), sg, uv=np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
           normals=None)
    pth = "/Dressing/handwash/towel_bin"
    d.xform(pth, (bx + .45, Y1 - .30, 0.))
    build_pedal_bin(d, pth, (150, 150, 150))
    d.rest("towel_bin", 0., "floor", 0.)
    prov("handwash_basin", "400 x 350 mm knee-operated, rim 0.90 m", "estimate", "knee-operated wash basin class")


def build_floor_items(d):
    # robot zone: 50 mm yellow/black floor tape around the robot reach, following the floor surface
    tape = d.preview("hazard_tape", texture=d.tex["hazard_tape"], rough=.5)
    x0, x1, y0, y1 = -1.20, 1.20, -1.05, 0.0
    w = .05
    for k, (p0, p1) in enumerate((((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x0, y1), (x0, y0)))):
        p0, p1 = np.array(p0), np.array(p1)
        dvec = (p1 - p0)/np.linalg.norm(p1 - p0)
        nvec = np.array([-dvec[1], dvec[0]])*w/2
        n = 60
        s = np.linspace(0, 1, n)
        cl = p0[None] + (p1 - p0)[None]*s[:, None]
        a = cl + nvec
        b = cl - nvec
        za = floor_z(a[:, 0], a[:, 1]) + .0006
        zb = floor_z(b[:, 0], b[:, 1]) + .0006
        V = np.concatenate([np.column_stack([a, za]), np.column_stack([b, zb])])
        F = [(i, i + 1, n + i + 1) for i in range(n - 1)] + [(i, n + i + 1, n + i) for i in range(n - 1)]
        F = np.array(F)
        if np.cross(np.append(V[1] - V[0], 0)[:3], V[n] - V[0])[2] < 0:
            F = F[:, ::-1]
        ln = np.linalg.norm(p1 - p0)
        uv = np.concatenate([np.column_stack([s*ln/.2, np.ones(n)]), np.column_stack([s*ln/.2, np.zeros(n)])])
        d.mesh(f"/Dressing/floor/zone_tape_{k}", V, F, tape, uv=uv, normals=None)
    d.rest("zone_tape", 0., "floor", 0.)
    prov("robot_zone", "50 mm hazard tape, x -1.2..1.2, y -1.05..0 (FR3 reach 855 mm around each base)", "authored",
         "ISO/TS 15066 forbids knives in a collaborative contact area -> guarded, marked cell")
    # safety laser scanner on a short post at the zone corner, scan plane about 0.3 m above the floor
    px, py = 1.32, -1.16
    fz = float(floor_z(px, py))
    st = d.steel()
    d.box("/Dressing/scanner/base", (px, py, fz + .005), (.16, .16, .010), st, r=.002)
    d.box("/Dressing/scanner/post", (px, py, fz + .01 + .12), (.05, .05, .24), d.preview("post_yellow", (235, 190, 20), .5), r=.004)
    d.box("/Dressing/scanner/body", (px, py, fz + .25 + .052), (.111, .111, .104), d.preview("scanner_yellow", (240, 200, 30), .45), r=.008)
    V, F = tube([(px, py, fz + .27), (px, py, fz + .33)], .05, n=40)
    Vh = V.copy()
    keep_dir = np.array([-1., 1., 0.])/math.sqrt(2)
    d.mesh("/Dressing/scanner/window", V + np.append(keep_dir[:2]*.03, 0), F, d.preview("scanner_window", (20, 20, 22), .1, coat=1.))
    d.cyl("/Dressing/scanner/cable", (px + .04, py + .04, fz + .26), (px + .04, py + .04, fz + .02), .004, d.preview("cable", (20, 20, 20), .5))
    d.rest("safety_scanner_post", fz, "floor", fz)
    prov("safety_scanner", "111 x 111 x 104 mm scanner on a 0.25 m post", "maker(remembered)",
         "compact safety laser scanner class (e.g. SICK nanoScan3 envelope, not re-read)")
    # wet floor sign (Poly Haven model)
    folder = fetch("WetFloorSign_01")
    usd = folder/"WetFloorSign_01_1k.usdc"
    d.ref("/Dressing/floor/wet_floor_sign", usd, "/WetFloorSign_01", (1.85, -1.35, 0.), yaw=math.radians(200),
          credit=("WetFloorSign_01", "Poly Haven, CC0 (filleting/visual_assets.json)"), textures=folder/"textures", ground=True)
    d.rest("wet_floor_sign", 0., "floor", 0.)
    # fire extinguisher and wall clock (label_application pinned CC0 models)
    from label_application import visual_assets as lava
    build_clock(d, "/Dressing/wall/clock", (-1.50, Y1, 2.35))
    for path, name, pos, yaw, sub in (("/Dressing/floor/extinguisher", "korean_fire_extinguisher_01", (X1 - .25, 1.45, 0.), math.pi/2, None),):
        usd = lava.model(name)
        d.ref(path, usd, f"/{name}", pos, yaw=yaw, credit=(name, "Poly Haven, CC0 (label_application/visual_assets.json)"),
              textures=usd.parent/"textures", ground=True)
    d.rest("fire_extinguisher", 0., "floor", 0.)


def build_clock(d, path, wall_point, hour=10, minute=8):
    """Wall clock, 300 mm face, 45 mm deep black rim (estimate: kitchen wall clock class), on the back wall."""
    x, y, z = wall_point
    r = .15
    rim = d.preview("clock_rim", (25, 25, 27), .4)
    V, F = lathe([(1e-4, 0.), (r + .008, 0.), (r + .010, .040), (r - .004, .045), (r - .006, .012), (1e-4, .012)], 64)
    Rm = rot_matrix((1, 0, 0), math.pi/2)                 # lathe z -> world -y (out of the wall)
    d.mesh(path + "/rim", V @ Rm.T + np.array([x, y, z]), F, rim, angle=40.)
    face = d.preview("clock_face", texture=d.tex["clock_face"], rough=.5)
    t = np.linspace(0, 2*math.pi, 64, endpoint=False)
    ring = np.stack([x + (r - .006)*np.cos(t), np.full(64, y - .0125), z + (r - .006)*np.sin(t)], 1)
    Vf = np.vstack([[x, y - .0125, z], ring])
    Ff = np.array([(0, 1 + i, 1 + (i + 1) % 64) for i in range(64)])       # faces -y (into the room)
    uv = np.vstack([[.5, .5], np.stack([.5 + .5*np.cos(t), .5 + .5*np.sin(t)], 1)])   # viewer's right = +x
    d.mesh(path + "/face", Vf, Ff, face, uv=uv, normals=None)
    hand = d.preview("clock_hands", (15, 15, 15), .4)
    for nm, ang, ln, wd in (("hour", (hour % 12 + minute/60.)*30., .075, .007), ("minute", minute*6., .115, .005)):
        a = math.radians(ang)
        dvec = np.array([math.sin(a), 0., math.cos(a)])
        c = np.array([x, y - .016, z]) + dvec*ln/2
        Vb, Fb = box_mesh((wd, .002, ln))
        Rh = rot_matrix((0, 1, 0), a)
        d.mesh(path + f"/hand_{nm}", Vb @ Rh.T + c, Fb, hand, normals=None)
    d.rest("wall_clock", 0., "back wall (mounted)", 0.)
    prov("wall_clock", "300 mm face, 45 mm deep", "estimate", "kitchen wall clock class; replaces the Poly Haven clock whose face did not render")


def build_camera_mounts(d):
    """Physical mounts of the fixed cameras (layout.CAMERAS; the overhead at render.OVERHEAD_EYE): a ceiling drop rod
    over the board and a wall arm."""
    from .render import OVERHEAD_EYE
    st = d.steel()
    blk = d.preview("camera_black", (20, 20, 22), .4)
    ov = OVERHEAD_EYE
    d.box("/Dressing/cameras/overhead_body", (ov[0], ov[1], ov[2] + .035), (.10, .05, .05), blk, r=.006)
    d.cyl("/Dressing/cameras/overhead_rod", (ov[0], ov[1], ov[2] + .06), (ov[0], ov[1], H), .012, st)
    d.cyl("/Dressing/cameras/overhead_plate", (ov[0], ov[1], H - .008), (ov[0], ov[1], H), .05, st)
    sv = L.CAMERAS["supervisor"]["pos"]
    tgt = np.asarray(L.CAMERAS["supervisor"]["look_at"])
    dirv = tgt - np.asarray(sv)
    dirv /= np.linalg.norm(dirv)
    back = np.asarray(sv) - dirv*.04
    d.box("/Dressing/cameras/supervisor_body", back, (.10, .05, .05), blk, r=.006)
    d.box("/Dressing/cameras/supervisor_arm", (sv[0], (sv[1] + Y1)/2 + .02, sv[2] + .045), (.03, Y1 - sv[1] - .02, .03), st, r=.004)
    d.box("/Dressing/cameras/supervisor_plate", (sv[0], Y1 - .005, sv[2] + .045), (.10, .01, .10), st, r=.002)
    prov("camera_mounts", "ceiling drop rod (overhead) and wall arm (supervisor)", "authored",
         "render.OVERHEAD_EYE and layout.CAMERAS positions")


# ------------------------------------------------------------------------------------------------ entry
def dress(stage, folder, preset="A", skip=(), task=None):
    """Author the dressed room into `stage` (visual only). `skip`: layout items the physics frame draws itself
    (e.g. {"board", "bowl", "knife_fillet"}); `task`: the task code whose task-only bench props and arrangement are
    drawn (layout.FRAME_PAN, task_spec; None: the preset's common bench). Returns a dict with credits,
    provenance, supports, adjustments and the layout conflicts found."""
    if preset not in ("A", "B", "C"):
        raise ValueError("preset must be A, B or C")
    d = Dresser(stage, folder, preset)
    skip = set(skip)
    mats = build_room(d)
    build_doors(d, mats)
    build_bench(d, skip)
    extra = build_bench_items(d, preset, skip, task=task)
    build_walls_fixtures(d, preset)
    build_floor_items(d)
    build_camera_mounts(d)
    return dict(credits=d.credits, provenance=PROVENANCE, support=d.support, adjustments=d.adjustments,
                layout_conflicts=layout_conflicts(task), lux_board_direct=getattr(d, "provenance_lux", None),
                dresser=d, **extra)


if __name__ == "__main__":
    if "--pin" in sys.argv:
        for name in NEW_ASSETS:
            print(name, fetch(name, pin=True))
    if "--usd" in sys.argv:
        from pxr import Usd, UsdGeom
        out = Path(sys.argv[sys.argv.index("--usd") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        st = Usd.Stage.CreateNew(str(out))
        UsdGeom.SetStageUpAxis(st, "Z")
        UsdGeom.SetStageMetersPerUnit(st, 1.)
        info = dress(st, out.parent, sys.argv[sys.argv.index("--preset") + 1] if "--preset" in sys.argv else "A")
        st.GetRootLayer().Save()
        info.pop("dresser")
        print(json.dumps(dict(adjustments=info["adjustments"], conflicts=info["layout_conflicts"],
                              lux=info["lux_board_direct"]), indent=1))

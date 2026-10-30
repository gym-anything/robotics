"""Appearance of the tools, containers, wall rail, table supplies and larger loose pieces of room 112.

Everything here is drawing only. Each builder takes the collision parts of sweeping.toolkit (or the installed parts of
sweeping.layout) and returns triangles in the same body frame, so that a recorded pose places the picture exactly
where the physics has the body. Rules kept by every builder:

  * a surface something can touch (the lower face of a block, a handle where it is gripped, the floor, lip and inner
    faces of a pan, the inside, bottom and rim height of a container) lies on its collision part;
  * what is added for the look of a moulded or assembled product (fillets, shoulders, sockets, rims, ribs, bolt
    heads, labels, grips) stays within a few millimetres of the collision parts and away from where bristles, loose
    pieces and fingers work; the exceptions are listed in NOTES at the end of this module;
  * the maker's name on the labels (BRAND) is made up; no real trade mark is drawn.

Geometry is made of a few generators (prism: an outline swept along a profile of insets; revolve; loft; tube; bead: a
ball with a hang hole) and is written as unwelded triangles with normals that are smooth across shallow edges and
sharp across creases. Labels are PIL images bound through OmniPBR (the mechanism of classroom_arrangement.render).
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

TAU = 2*math.pi
BRAND = "TARNLEY"              # made up
GRAIN = 22.0                   # texture coordinates per metre on plastics (their textures are plain; kept for a later grain)
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/")
FONTS = dict(bold=("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"), regular=("DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
             narrow=("LiberationSansNarrow-Bold.ttf", "DejaVuSans-Bold.ttf"), italic=("LiberationSans-BoldItalic.ttf", "DejaVuSans-Bold.ttf"),
             mono=("DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"))


# ============================================================================== outlines (closed, counter-clockwise)
def circle(r, n=48):
    a = np.arange(n)*TAU/n
    return np.column_stack([r*np.cos(a), r*np.sin(a)])


def rounded(pts, radii, k=5, step=None):
    """A convex polygon with its corners rounded (radius per corner, 0 keeps it sharp); straight sides are cut into
    pieces no longer than step."""
    pts = np.asarray(pts, float)
    n = len(pts)
    radii = np.broadcast_to(np.asarray(radii, float), (n,))
    arcs = []
    for i in range(n):
        P, A, B = pts[i], pts[i - 1], pts[(i + 1) % n]
        r = float(radii[i])
        if r <= 0:
            arcs.append(P[None])
            continue
        d1, d2 = (A - P)/np.linalg.norm(A - P), (B - P)/np.linalg.norm(B - P)
        half = 0.5*math.acos(float(np.clip(d1 @ d2, -1, 1)))
        t = r/math.tan(half)
        bis = (d1 + d2)/np.linalg.norm(d1 + d2)
        C = P + bis*r/math.sin(half)
        v1, v2 = P + d1*t - C, P + d2*t - C
        a1, a2 = math.atan2(v1[1], v1[0]), math.atan2(v2[1], v2[0])
        if a2 < a1:
            a2 += TAU
        a = np.linspace(a1, a2, k + 1)
        arcs.append(C + r*np.column_stack([np.cos(a), np.sin(a)]))
    out = []
    for i in range(n):
        out.extend(arcs[i])
        a, b = arcs[i][-1], arcs[(i + 1) % n][0]
        length = float(np.linalg.norm(b - a))
        if step and length > step:
            m = int(math.ceil(length/step))
            out.extend(a + (b - a)*j/m for j in range(1, m))
    out = np.array(out)
    keep = np.linalg.norm(out - np.roll(out, 1, 0), axis=1) > 1e-9
    return out[keep]


def rrect(hx, hy, rc, k=5, step=None):
    return rounded([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)], rc, k, step)


def superring(a, b, p, n=24):
    """Superellipse of half axes a, b: p = 2 an ellipse, larger p toward a rectangle."""
    t = np.arange(n)*TAU/n
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([a*np.sign(c)*np.abs(c)**(2/p), b*np.sign(s)*np.abs(s)**(2/p)])


def _miter(P):
    e = np.roll(P, -1, 0) - P
    en = np.column_stack([e[:, 1], -e[:, 0]])
    en /= np.linalg.norm(en, axis=1)[:, None]
    pn = np.roll(en, 1, 0)
    return (en + pn)/(1 + (en*pn).sum(1))[:, None]


# ============================================================================== solids: (vertices, triangles)
def _skin(k, n):
    idx = np.arange(n)
    nxt = (idx + 1) % n
    f = []
    for i in range(k - 1):
        a, b, c, d = i*n + idx, i*n + nxt, (i + 1)*n + nxt, (i + 1)*n + idx
        f += [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
    return np.concatenate(f) if f else np.zeros((0, 3), int)


def _fan(v, ring, up):
    c = len(v)
    v = np.vstack([v, v[ring].mean(0)])
    a, b = ring, np.roll(ring, -1)
    cc = np.full(len(ring), c)
    return v, np.column_stack([cc, a, b] if up else [cc, b, a])


def prism(outline, prof, cap0=True, cap1=True):
    """The outline (in the local xy plane) swept along local z through prof = [(inset, z), ...]: at each profile point
    the outline is moved inward by inset (outward if negative). cap0 closes the first ring facing -z, cap1 the last
    ring facing +z. A profile that goes up the outside and comes down the inside draws an open shell."""
    P = np.asarray(outline, float)
    m = _miter(P)
    n = len(P)
    v = np.concatenate([np.column_stack([P - d*m, np.full(n, w)]) for d, w in prof])
    f = [_skin(len(prof), n)]
    if cap0:
        v, c = _fan(v, np.arange(n), False)
        f.append(c)
    if cap1:
        v, c = _fan(v, (len(prof) - 1)*n + np.arange(n), True)
        f.append(c)
    return v, np.concatenate(f)


def revolve(prof, n=48, mod=None):
    """prof = [(radius, z), ...] turned about local z. Outside going up, inside coming down. mod(theta, z, r) may
    return (dr, dz) for shapes that are not quite round (a draped bag)."""
    prof = np.asarray(prof, float)
    th = np.arange(n)*TAU/n
    r = np.repeat(prof[:, :1], n, 1)
    z = np.repeat(prof[:, 1:2], n, 1)
    if mod is not None:
        dr, dz = mod(th[None, :], prof[:, 1:2], prof[:, :1])
        r, z = r + dr, z + dz
    v = np.stack([r*np.cos(th), r*np.sin(th), z], -1).reshape(-1, 3)
    return v, _skin(len(prof), n)


def loft(rings, cap0=True, cap1=True):
    """rings (k, n, 3): closed loops, each counter-clockwise about the direction of travel."""
    rings = np.asarray(rings, float)
    k, n, _ = rings.shape
    v = rings.reshape(-1, 3)
    f = [_skin(k, n)]
    if cap0:
        v, c = _fan(v, np.arange(n), False)
        f.append(c)
    if cap1:
        v, c = _fan(v, (k - 1)*n + np.arange(n), True)
        f.append(c)
    return v, np.concatenate(f)


def _frames(pts):
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    u = np.cross(t[0], (0., 0., 1.))
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(t[0], (1., 0., 0.))
    us = [u/np.linalg.norm(u)]
    for i in range(1, len(pts)):
        u = us[-1] - t[i]*(us[-1] @ t[i])
        us.append(u/np.linalg.norm(u))
    us = np.array(us)
    return t, us, np.cross(t, us)


def tube(pts, r, n=12, round0=False, round1=False, k=4, caps=True):
    """A round rod through the points (radius one number or one per point). A rounded end is a hemisphere beyond
    the end point."""
    pts = np.asarray(pts, float)
    r = np.broadcast_to(np.asarray(r, float), (len(pts),)).astype(float)
    t, u, w = _frames(pts)
    c, rr, uu, ww = list(pts), list(r), list(u), list(w)
    for phi in np.linspace(0, math.pi/2, k + 1)[1:]:
        if round0:
            c.insert(0, pts[0] - t[0]*r[0]*math.sin(phi)); rr.insert(0, max(r[0]*math.cos(phi), 2e-5)); uu.insert(0, u[0]); ww.insert(0, w[0])
        if round1:
            c.append(pts[-1] + t[-1]*r[-1]*math.sin(phi)); rr.append(max(r[-1]*math.cos(phi), 2e-5)); uu.append(u[-1]); ww.append(w[-1])
    a = np.arange(n)*TAU/n
    rings = [ci + ri*(np.cos(a)[:, None]*ui + np.sin(a)[:, None]*wi) for ci, ri, ui, wi in zip(c, rr, uu, ww)]
    return loft(rings, caps, caps)


def rod(a, b, r, n=14):
    """A capsule whose tips are a and b (the convention of classroom_arrangement.furniture.tube)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = (b - a)/np.linalg.norm(b - a)
    return tube([a + r*t, b - r*t], r, n, True, True)


def bead(R, rh, n=28, k=8, ch=0.0005):
    """A ball of radius R with a hole of radius rh along local z: the hang hole of a cap or of a handle's end."""
    rc = rh + ch
    zm = math.sqrt(R*R - rc*rc)
    pm = math.asin(zm/R)
    prof = [(R*math.cos(p), R*math.sin(p)) for p in np.linspace(-pm, pm, 2*k + 1)] + [(rh, zm - ch), (rh, -zm + ch), (rc, -zm)]
    return revolve(prof, n)


def basis(ez, ex=(1., 0., 0.)):
    """Rotation matrix whose columns are ex, ez x ex, ez (ex made perpendicular to ez)."""
    ez = np.asarray(ez, float)/np.linalg.norm(ez)
    ex = np.asarray(ex, float)
    ex = ex - ez*(ex @ ez)
    ex /= np.linalg.norm(ex)
    return np.column_stack([ex, np.cross(ez, ex), ez])


XZ = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])        # local (u, v, w) -> (u, -w, v): an outline drawn in x, z
YZ = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])         # local (u, v, w) -> (w, u, v): an outline drawn in y, z
ALONG_X = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])   # local z -> +x
HOLE_Y = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])    # local z -> +y


def decal(c, n, up, w, h, rc=0., lift=0.00025):
    """A printed patch lying on a flat face: centre c, outward normal n, the print's up direction. Returns vertices,
    triangles and texture coordinates such that the image reads upright from outside."""
    n = np.asarray(n, float)/np.linalg.norm(n)
    up = np.asarray(up, float)
    up = up - n*(up @ n)
    up /= np.linalg.norm(up)
    right = np.cross(up, n)
    P = rrect(w/2, h/2, rc, 4) if rc > 0 else np.array([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    P = np.vstack([P, (0., 0.)])
    v = np.asarray(c, float) + n*lift + P[:, :1]*right + P[:, 1:]*up
    m = len(P) - 1
    idx = np.arange(m)
    f = np.column_stack([np.full(m, m), idx, (idx + 1) % m])
    return v, f, P/np.array([w, h]) + 0.5


def decal_cyl(R, a_mid, arc, z0, z1, inside=False, lift=0.0003):
    """A printed patch bent on a cylinder about local z: centred at angle a_mid, `arc` metres along the surface."""
    half = arc/(2*R)
    seg = max(4, int(2*half/0.07) + 1)
    a = np.linspace(a_mid - half, a_mid + half, seg + 1)
    rr = R - lift if inside else R + lift
    v = np.concatenate([np.column_stack([rr*np.cos(a), rr*np.sin(a), np.full(seg + 1, z)]) for z in (z0, z1)])
    u = np.linspace(1., 0., seg + 1) if inside else np.linspace(0., 1., seg + 1)
    uv = np.concatenate([np.column_stack([u, np.full(seg + 1, t)]) for t in (0., 1.)])
    i = np.arange(seg)
    a_, b_, c_, d_ = i, i + 1, seg + 1 + i + 1, seg + 1 + i
    f = np.concatenate([np.column_stack([a_, b_, c_]), np.column_stack([a_, c_, d_])])
    return v, (f[:, ::-1] if inside else f), uv


# ============================================================================== triangles with normals
def _corner_normals(v, f, crease):
    """Normal of every triangle corner: the area-weighted mean of the triangles around that vertex whose normals lie
    within `crease` degrees of the corner's own triangle."""
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1)
    unit = fn/np.maximum(area, 1e-30)[:, None]
    if crease <= 0:
        return np.repeat(unit[:, None, :], 3, 1), area
    _, canon = np.unique(np.round(v*2e6).astype(np.int64), axis=0, return_inverse=True)
    cf = canon.reshape(-1)[f]
    flat = cf.ravel()
    order = np.argsort(flat, kind="stable")
    cv = flat[order]
    count = np.bincount(cv)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    adj = -np.ones((len(count), int(count.max())), np.int64)
    adj[cv, np.arange(len(cv)) - start[cv]] = order//3
    cos_c = math.cos(math.radians(crease))
    out = np.empty((len(f), 3, 3))
    for lo in range(0, len(f), 8192):
        sl = slice(lo, lo + 8192)
        for c in range(3):
            nb = adj[cf[sl, c]]
            ok = nb >= 0
            nb = np.where(ok, nb, 0)
            ok &= (unit[nb]*unit[sl][:, None, :]).sum(-1) >= cos_c
            s = (fn[nb]*ok[..., None]).sum(1)
            out[sl, c] = s/np.maximum(np.linalg.norm(s, axis=1), 1e-30)[:, None]
    return out, area


def uv_box(tri, per_m):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ax = np.argmax(np.abs(n), 1)
    uv = np.empty(tri.shape[:2] + (2,))
    for a, pair in enumerate(((1, 2), (0, 2), (0, 1))):
        m = ax == a
        uv[m] = tri[m][:, :, pair]*per_m
    return uv


def _theta(tri):
    th = np.arctan2(tri[..., 1], tri[..., 0])
    ref = th[:, :1]
    return ref + (th - ref + math.pi) % TAU - math.pi


def uv_cyl(tri, per_m):
    """Along local z (u) and around it (v, a whole number of repeats so that there is no seam)."""
    rad = float(np.hypot(tri[..., 0], tri[..., 1]).max())
    turns = max(1, round(TAU*rad*per_m))
    return np.stack([tri[..., 2]*per_m, _theta(tri)/TAU*turns], -1)


class Asm:
    """Triangles of one rigid body's picture, grouped by material name."""

    def __init__(self):
        self.groups = {}

    def add(self, mat, vf, R=None, p=None, uv=None, crease=42.):
        v, f = np.asarray(vf[0], float), np.asarray(vf[1], int)
        tri = v[f]
        nrm, area = _corner_normals(v, f, crease)
        if len(vf) > 2:
            st = np.asarray(vf[2], float)[f]
        elif uv is None:
            st = np.zeros(tri.shape[:2] + (2,))
        elif uv[0] == "box":
            st = uv_box(tri, uv[1])
        elif uv[0] == "cyl":
            st = uv_cyl(tri, uv[1])
        elif uv[0] == "grain":                      # u along local z; v a narrow band of the texture, there and back
            st = np.stack([tri[..., 2]*uv[1], 0.31 + 0.10*np.abs(_theta(tri))/math.pi], -1)
        elif uv[0] == "wrap":                       # u from z0 to z1 along local z, v once around
            st = np.stack([(tri[..., 2] - uv[1])/(uv[2] - uv[1]), _theta(tri)/TAU + 0.5], -1)
        else:
            raise ValueError(uv)
        if R is not None:
            R = np.asarray(R, float)
            tri, nrm = tri @ R.T, nrm @ R.T
        if p is not None:
            tri = tri + np.asarray(p, float)
        keep = area > 1e-13
        self.groups.setdefault(mat, []).append((tri[keep], nrm[keep], st[keep]))
        return self

    def merge(self, other, R=None, p=None):
        for mat, parts in other.groups.items():
            for tri, nrm, st in parts:
                if R is not None:
                    tri, nrm = tri @ np.asarray(R).T, nrm @ np.asarray(R).T
                if p is not None:
                    tri = tri + np.asarray(p, float)
                self.groups.setdefault(mat, []).append((tri, nrm, st))
        return self

    def triangles(self, only=None):
        return np.concatenate([t for m, parts in self.groups.items() if only is None or m in only for t, _, _ in parts])

    def emit(self, S, base, mats):
        for mat, parts in self.groups.items():
            pts = np.concatenate([t.reshape(-1, 3) for t, _, _ in parts])
            nrm = np.concatenate([n.reshape(-1, 3) for _, n, _ in parts])
            st = np.concatenate([u.reshape(-1, 2) for _, _, u in parts])
            S.mesh(f"{base}/{mat}", pts, np.arange(len(pts)), uv=st, normals=nrm, mat=mats[mat])


# ============================================================================== printed matter
def _font(size, style="bold"):
    from PIL import ImageFont
    for d in FONT_DIRS:
        for name in FONTS[style]:
            try:
                return ImageFont.truetype(d + name, int(size))
            except OSError:
                continue
    return ImageFont.load_default(int(size))


def srgb(rgb):
    """Linear colour (as the stage's plain materials take it) -> 8 bit sRGB (as a texture stores it)."""
    c = np.clip(np.asarray(rgb, float), 0, 1)
    s = np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4) - 0.055)
    return tuple(int(round(255*x)) for x in s)


def _logo(d, x, y, h, ink, paper):
    """The made-up maker's mark: a rounded tile with three sweep strokes, then the name. Returns the right end."""
    d.rounded_rectangle((x, y, x + h, y + h), radius=h*0.2, fill=ink)
    for k in range(3):
        o = h*(0.22 + 0.2*k)
        d.line((x + o, y + h*0.82, x + o + h*0.18, y + h*0.18), fill=paper, width=max(2, int(h*0.09)))
    f = _font(h*0.88, "italic")
    d.text((x + h*1.22, y + h*0.5), BRAND, font=f, fill=ink, anchor="lm")
    return x + h*1.22 + d.textlength(BRAND, font=f)


def _barcode(d, x, y, w, h, seed, ink=(18, 18, 18), digits=None):
    rng = np.random.default_rng(seed)
    unit = max(2, int(w/95))
    cx = x
    k = 0
    while cx < x + w - unit:
        bw = int(rng.integers(1, 4))*unit
        guard = k in (0, 2) or cx > x + w - 5*unit
        if guard or k % 2 == 0:
            d.rectangle((cx, y, min(cx + bw, x + w) - 1, y + h*(0.86 if not guard else 0.94)), fill=ink)
        cx += bw if not guard else unit
        k += 1
    if digits:
        d.text((x + w/2, y + h), digits, font=_font(h*0.2, "mono"), fill=ink, anchor="mt")


def _canvas(w, h, bg):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (int(w), int(h)), bg)
    return im, ImageDraw.Draw(im)


def _fit(d, text, style, box_w, size):
    f = _font(size, style)
    while d.textlength(text, font=f) > box_w and size > 8:
        size *= 0.94
        f = _font(size, style)
    return f


def art_head(bg, ink, accent, line1, line2, facts):
    """Hot-stamped print on the face of a broom head (the background is the head's own plastic)."""
    im, d = _canvas(2100, 190, bg)
    x = _logo(d, 40, 32, 118, ink, bg)
    d.line((x + 46, 26, x + 46, 164), fill=accent, width=6)
    f1 = _fit(d, line1, "bold", 780, 76)
    d.text((x + 86, 20), line1, font=f1, fill=ink)
    d.text((x + 88, 112), line2, font=_fit(d, line2, "regular", 780, 44), fill=accent)
    xr = 2100 - 40
    for k, t in enumerate(facts):
        f = _font(40, "narrow")
        wdt = d.textlength(t, font=f) + 44
        d.rounded_rectangle((xr - wdt, 28 + 70*k, xr, 84 + 70*k), radius=14, outline=ink, width=4)
        d.text((xr - wdt/2, 56 + 70*k), t, font=f, fill=ink, anchor="mm")
    return im


def art_sticker(w, h, title, lines, seed, band=(200, 40, 40), paper=(246, 246, 240), code=None, vertical=False):
    """A paper label: coloured band with the mark, a title, small print, a bar code."""
    W, H = w, h
    im, d = _canvas(W, H, paper)
    bh = int(H*0.30)
    d.rectangle((0, 0, W, bh), fill=band)
    _logo(d, int(H*0.06), int(bh*0.16), int(bh*0.68), (250, 250, 248), band)
    ink = (26, 28, 32)
    d.text((int(H*0.06), bh + H*0.05), title, font=_fit(d, title, "bold", W*0.58, H*0.17), fill=ink)
    y = bh + H*0.27
    for t in lines:
        d.text((int(H*0.06), y), t, font=_fit(d, t, "regular", W*0.58, H*0.085), fill=ink)
        y += H*0.105
    _barcode(d, W*0.66, bh + H*0.08, W*0.30, H*0.44, seed, digits=code)
    d.rectangle((1, 1, W - 2, H - 2), outline=(150, 150, 150), width=2)
    return im.rotate(90, expand=True) if vertical else im


def art_emboss(w, h, bg, lines, mark=True, tone=0.20):
    """Raised lettering moulded into a plastic face: (colour image, height field for the normal map)."""
    from PIL import Image, ImageDraw
    hm = Image.new("L", (int(w), int(h)), 0)
    d = ImageDraw.Draw(hm)
    y = h*0.14
    if mark:
        s = h*0.26
        x = _logo(d, w*0.06, y, s, 255, 0)
        d.line((w*0.06, y + s*1.22, x, y + s*1.22), fill=255, width=max(2, int(h*0.012)))
        y += s*1.5
    for t, style, size in lines:
        d.text((w*0.06, y), t, font=_fit(d, t, style, w*0.88, h*size), fill=255)
        y += h*size*1.35
    hgt = np.asarray(hm, np.float32)/255
    col = np.asarray(bg, np.float32)[None, None, :]*(1 + tone*hgt[..., None]) + 255*0.03*hgt[..., None]
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)), hgt


def art_stamp(w, h, lines, ink=(46, 30, 22)):
    """Ink stamped on wood: (colour, mask); the mask cuts the patch down to the lettering."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (int(w), int(h)), 0)
    d = ImageDraw.Draw(mask)
    x = _logo(d, h*0.08, h*0.14, h*0.72, 255, 0)
    xx = x + h*0.5
    for k, (t, style) in enumerate(lines):
        d.text((xx, h*(0.30 + 0.42*k)), t, font=_fit(d, t, style, w - xx - h*0.1, h*0.36), fill=255, anchor="lm")
    return Image.new("RGB", (int(w), int(h)), ink), mask


def art_notebook():
    """Ruled exercise paper, 56 mm square tile at 16 px/mm (seven rules, a margin line, a few pencil strokes)."""
    im, d = _canvas(896, 896, (247, 247, 242))
    for k in range(7):
        y = 64 + k*128
        d.line((0, y, 896, y), fill=(150, 178, 214), width=3)
    d.line((120, 0, 120, 896), fill=(222, 140, 140), width=3)
    rng = np.random.default_rng(31)
    for k in range(5):
        y = 64 + int(rng.integers(0, 7))*128 - 26
        x = 150 + rng.uniform(0, 250)
        pts = [(x + j*9, y + 16*math.sin(j*rng.uniform(0.6, 1.3)) + rng.uniform(-3, 3)) for j in range(int(rng.integers(20, 55)))]
        d.line(pts, fill=(70, 72, 84), width=3)
    return im


def art_pencil(body):
    """Paint and foil stamp of a pencil: u along its length, v once around its six flats."""
    im, d = _canvas(1500, 180, body)
    d.text((170, 45), f"{BRAND}  ·  SCHOOL  HB 2", font=_font(21, "bold"), fill=(34, 74, 40), anchor="lm")
    for k in range(7):
        d.line((0, k*30, 1500, k*30), fill=tuple(int(c*0.86) for c in body), width=1)
    return im


def art_glaze(base, seed):
    """Reactive stoneware glaze: a base colour broken by darker pooling and lighter specks."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.normal(size=(256, 256)), 9, mode="wrap")
    n = n/np.abs(n).max()
    col = np.asarray(base, np.float32)[None, None]*(1 + 0.08*n[..., None])
    speck = rng.random((256, 256)) > 0.992
    col[speck] = col[speck]*0.5 + 120
    from PIL import Image
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))


def art_brushed(base):
    """Brushed strip steel: fine streaks along u."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    s = gaussian_filter(rng.normal(size=(512, 512)), (0.6, 40), mode="wrap")
    s = s/np.abs(s).max()
    col = np.asarray(base, np.float32)[None, None]*(1 + 0.035*s[..., None])
    from PIL import Image
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)), s


def art_ream():
    """The printed wrapper of a ream of letter paper, top face (216 x 280 mm, 5 px/mm)."""
    im, d = _canvas(1080, 1400, (244, 244, 240))
    d.rectangle((0, 0, 1080, 400), fill=(28, 84, 150))
    _logo(d, 70, 70, 120, (250, 250, 248), (28, 84, 150))
    d.text((70, 240), "COPY PAPER", font=_font(92), fill=(250, 250, 248))
    d.text((70, 470), "500 sheets", font=_font(84), fill=(28, 84, 150))
    for k, t in enumerate(("Letter  8.5 x 11 in  ·  216 x 279 mm", "20 lb  ·  75 g/m²  ·  92 bright", "Acid free  ·  long grain")):
        d.text((70, 610 + k*74), t, font=_font(46, "regular"), fill=(40, 44, 52))
    d.ellipse((720, 470, 1000, 750), outline=(28, 84, 150), width=10)
    d.text((860, 610), "92", font=_font(120), fill=(28, 84, 150), anchor="mm")
    for k in range(9):
        d.rectangle((70 + k*30, 1000 - k*14, 760 + k*30, 1012 - k*14), fill=(200, 205, 214))
    _barcode(d, 700, 1120, 300, 190, 77, digits="0 71249 00112 6")
    d.text((70, 1290), "Keep dry. Store flat.", font=_font(40, "regular"), fill=(90, 94, 100))
    return im


def art_glue():
    """Wrap label of a 118 mL bottle of school glue (one face of the bottle, 40 x 62 mm at 10 px/mm)."""
    im, d = _canvas(400, 620, (250, 250, 247))
    d.rectangle((0, 0, 400, 150), fill=(232, 120, 24))
    _logo(d, 24, 40, 62, (255, 255, 255), (232, 120, 24))
    d.text((200, 230), "SCHOOL", font=_font(74), fill=(24, 70, 150), anchor="mm")
    d.text((200, 318), "GLUE", font=_font(96), fill=(24, 70, 150), anchor="mm")
    d.text((200, 410), "washable · non-toxic", font=_font(30, "regular"), fill=(40, 44, 52), anchor="mm")
    d.text((200, 470), "4 FL OZ (118 mL)", font=_font(32), fill=(40, 44, 52), anchor="mm")
    _barcode(d, 110, 510, 180, 80, 12)
    return im


def art_grad(marks, unit_label):
    """Graduation scale printed inside the pail: ticks with numbers, 30 mm wide strip (8 px/mm); marks = [(height
    fraction, text)]."""
    im, d = _canvas(240, 1600, (236, 240, 246))
    d.text((120, 40), unit_label, font=_font(44), fill=(20, 30, 60), anchor="mm")
    for frac, text in marks:
        y = 1600*(1 - frac)
        d.line((0, y, 110, y), fill=(20, 30, 60), width=7)
        d.text((128, y), text, font=_font(54), fill=(20, 30, 60), anchor="lm")
    return im


def normal_map(height, strength):
    from PIL import Image
    gy, gx = np.gradient(np.asarray(height, np.float32))
    n = np.dstack([-gx*strength, gy*strength, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return Image.fromarray(((n*0.5 + 0.5)*255).astype(np.uint8))


# linear colour, roughness, metallic of the plain materials added by this module
PLAIN = dict(
    vinyl_black=((.030, .030, .034), .62, 0.), rubber_black=((.022, .022, .024), .85, 0.), steel_zinc=((.66, .67, .69), .30, .9),
    steel_dark=((.10, .10, .11), .35, .6), brass=((.72, .50, .20), .28, .9), alu=((.80, .78, .68), .30, .9),
    graphite=((.05, .05, .055), .35, .4), pencil_wood=((.78, .58, .38), .75, 0.), eraser_pink=((.84, .40, .44), .85, 0.),
    glaze_in=((.86, .85, .80), .10, 0.), biscuit=((.74, .70, .63), .92, 0.), slot=((.015, .015, .016), .6, 0.),
    paper_in=((.90, .90, .88), .85, 0.), crayon_wrap=((.90, .88, .80), .8, 0.), glue_cap=((.90, .42, .06), .4, 0.),
    grip_white=((.86, .86, .84), .5, 0.), basket_in=((.10, .105, .11), .7, 0.),
    snip_0=((.80, .16, .14), .85, 0.), snip_1=((.14, .34, .78), .85, 0.), snip_2=((.93, .78, .16), .85, 0.),
    snip_3=((.90, .90, .87), .85, 0.), snip_4=((.20, .58, .26), .85, 0.), snip_5=((.92, .46, .10), .85, 0.),
    snip_6=((.86, .42, .62), .85, 0.), snip_7=((.48, .28, .66), .85, 0.),
    crayon_0=((.80, .10, .10), .6, 0.), crayon_1=((.10, .30, .75), .6, 0.), crayon_2=((.95, .78, .10), .6, 0.),
    crayon_3=((.12, .52, .22), .6, 0.), crayon_4=((.92, .44, .08), .6, 0.), crayon_5=((.45, .22, .62), .6, 0.),
)
# plastics that get the moulding grain: linear colour, roughness
PLASTIC = dict(pl_red=((.62, .08, .08), .42), pl_black=((.045, .045, .05), .46), pl_grey=((.42, .44, .46), .48), pl_pan=((.16, .17, .19), .45),
               pl_barrel=((.44, .45, .46), .55), pl_pail=((.10, .28, .60), .42), pl_basket=((.25, .27, .28), .52),
               pl_cup=((.85, .30, .12), .45), pl_glue=((.90, .90, .88), .38), pl_ream=((.90, .90, .87), .62))


def data_inputs(S, name):
    """Mark the normal (and roughness) map of a material made by Stage.textured as data. Left as it is made, the map is
    read as a colour, through the sRGB curve: a flat normal map then tilts every normal about 40 degrees, and a face
    that carries one is lit differently from the same plastic beside it (measured on four grey tiles under one light:
    246 without a map, 212 with a flat map, 246 with the flat map marked raw)."""
    sh = S.UsdShade.Shader.Get(S.stage, f"/Looks/{name}/Shader")
    for key in ("normalmap_texture", "reflectionroughness_texture"):
        attr = sh.GetInput(key).GetAttr()
        if attr and attr.IsValid():
            attr.SetColorSpace("raw")


class Looks:
    """Makes the materials of this module on a stage and adds them to the stage's material table."""

    def __init__(self, S, mats):
        self.S, self.mats = S, mats
        self.dir = S.folder/"textures"/"tools"
        self.dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(41)
        from scipy.ndimage import gaussian_filter
        w = gaussian_filter(rng.normal(size=(512, 512)), (5, 34), mode="wrap")*3 + gaussian_filter(rng.normal(size=(512, 512)), (14, 9), mode="wrap")
        self.film = self.save("film_n", normal_map(w/np.abs(w).max(), 9.))

    def save(self, name, im):
        path = self.dir/f"{name}.png"
        im.save(path)
        return path

    def flat(self, name, rgb):
        from PIL import Image
        return self.save(name, Image.new("RGB", (8, 8), srgb(rgb)))

    def omni(self, name, tex, rough, metal=0., normal=None, bump=0.35, mask=None, tint=None):
        m = self.S.textured(name, tex, rough=rough, metal=metal, normal=normal, normal_strength=bump, tint=tint)
        data_inputs(self.S, name)
        if mask is not None:                    # the patch exists only where the mask is light (cut-out lettering)
            T, Sdf = self.S.Sdf.ValueTypeNames, self.S.Sdf
            sh = self.S.UsdShade.Shader.Get(self.S.stage, f"/Looks/{name}/Shader")
            sh.CreateInput("enable_opacity", T.Bool).Set(True)
            sh.CreateInput("enable_opacity_texture", T.Bool).Set(True)
            sh.CreateInput("opacity_texture", T.Asset).Set(Sdf.AssetPath(str(mask.resolve())))
            sh.CreateInput("opacity_threshold", T.Float).Set(0.45)
        self.mats[name] = m
        return m

    def build(self):
        S, mats = self.S, self.mats
        for name, (rgb, rough, metal) in PLAIN.items():
            mats[name] = S.color(name, rgb, rough, metal)
        for name, (rgb, rough) in PLASTIC.items():
            self.omni(name, self.flat(name, rgb), rough)
        from whiteboard.visual_assets import asset                 # lacquered beech of the block, the brush and the wooden handle
        self.omni("wood_beech", asset("oak_veneer_01", "diff"), .34, normal=asset("oak_veneer_01", "nor_gl"), bump=1.0, tint=(1.0, 0.86, 0.66))
        red, black, grey = (srgb(PLASTIC[k][0]) for k in ("pl_red", "pl_black", "pl_pan"))
        self.omni("pr_angle", self.save("pr_angle", art_head(red, (250, 248, 240), (255, 214, 96), "PRO ANGLE 11", "11 in / 28 cm  ·  commercial angle broom",
                                                           ("FLAGGED PET", "INDOOR · OUTDOOR"))), PLASTIC["pl_red"][1])
        self.omni("pr_lobby", self.save("pr_lobby", art_head(black, (206, 208, 212), (232, 150, 40), "LOBBY 7½", "7.5 in / 19 cm  ·  lobby broom",
                                                           ("POLY FIBRE", "PAIRS WITH PAN 2531"))), PLASTIC["pl_black"][1])
        for name, title, lines, code, seed, band in (
                ("st_angle", "Angle broom 201", ("Sweep 11 in / 28 cm", "Steel handle 54 in", "Flagged PET fibre"), "0 71249 00201 4", 201, (188, 30, 30)),
                ("st_lobby", "Lobby broom 6374", ("Head 7.5 in / 19 cm", "Overall 35 in / 89 cm", "Use with pan 2531"), "0 71249 06374 9", 6374, (40, 44, 52)),
                ("st_push", "Push broom HD-18", ("Block 18 in / 46 cm", "Fine sweep, 3 in trim", "Hardwood handle 60 in"), "0 71249 00918 1", 918, (24, 70, 150)),
                ("st_pan", "Lobby dustpan 2531", ("Mouth 11 1/4 in / 286 mm", "Handle 30 in, swivel yoke", "Polypropylene pan"), "0 71249 02531 0", 2531, (40, 44, 52)),
                ("st_hand", "Dustpan 2005", ("12 in / 30 cm, rubber edge", "Hang hole handle", "Polypropylene"), "0 71249 02005 6", 2005, (232, 120, 24)),
                ("st_barrel", "Round barrel 2620", ("20 gal / 75.7 L", "Use 30 x 36 in liners", "Do not use for hot ash"), "0 71249 02620 1", 2620, (232, 180, 20)),
                ("st_pail", "Utility pail 14", ("14 qt / 13.2 L", "Graduated inside", "Max fill line 12 qt"), "0 71249 00014 0", 14, (232, 120, 24)),
                ("st_basket", "Wastebasket 28", ("28 qt / 26.5 L", "Fits 24 x 24 in liners", "Polyethylene"), "0 71249 00028 7", 28, (24, 70, 150)),
                ("st_brush", "Counter brush 6342", ("Face 8 in / 20 cm", "Trim 2 1/2 in", "Lacquered beech"), "0 71249 06342 8", 6342, (188, 30, 30))):
            self.omni(name, self.save(name, art_sticker(960, 480, title, lines, seed, band, code=code)), .55)
        for name, title, seed in (("wr_angle", "Angle 201", 5), ("wr_lobby", "Lobby 6374", 6), ("wr_push", "Push HD-18", 7), ("wr_pan", "Pan 2531", 8)):
            self.omni(name, self.save(name, art_sticker(900, 520, title, ("Commercial grade", "Hang to store", "Made for " + BRAND), seed,
                                                        (188, 30, 30), code=None, vertical=True)), .5)
        for name, bg, lines in (
                ("em_pan", black, (("LOBBY PAN 2531", "bold", 0.15), ("11 1/4 in · 286 mm     >PP<", "regular", 0.10))),
                ("em_hand", grey, (("DUSTPAN 2005", "bold", 0.15), ("12 in · 30 cm     >PP<", "regular", 0.10))),
                ("em_basket", srgb(PLASTIC["pl_basket"][0]), (("28 QT · 26.5 L", "bold", 0.15), (">HDPE<   MADE FOR " + BRAND, "regular", 0.09))),
                ("em_barrel", srgb(PLASTIC["pl_barrel"][0]), (("2620 · 20 GAL · 75.7 L", "bold", 0.15), (">LLDPE<   NSF 2 · 21", "regular", 0.10)))):
            col, hgt = art_emboss(*dict(em_pan=(1100, 470), em_hand=(1110, 340)).get(name, (1024, 512)), bg, lines)
            rough = dict(em_pan="pl_black", em_hand="pl_pan", em_basket="pl_basket", em_barrel="pl_barrel")[name]
            from scipy.ndimage import gaussian_filter
            self.omni(name, self.save(name, col), PLASTIC[rough][1], normal=self.save(name + "_n", normal_map(gaussian_filter(hgt, 1.6), 5.0)), bump=1.0)
        col, mask = art_stamp(1280, 180, (("HD-18   FINE SWEEP", "bold"), ("18 IN  ·  46 CM  ·  BEECH", "regular")))
        self.omni("stamp_push", self.save("stamp_push", col), .7, mask=self.save("stamp_push_a", mask))
        col, mask = art_stamp(960, 150, (("COUNTER 8", "bold"), ("8 IN · 20 CM", "regular")))
        self.omni("stamp_brush", self.save("stamp_brush", col), .7, mask=self.save("stamp_brush_a", mask))
        im, d = _canvas(480, 200, (182, 184, 188))
        d.text((240, 78), "18", font=_font(120), fill=(60, 62, 66), anchor="mm")
        d.text((240, 160), "IN  ·  46 CM", font=_font(36), fill=(60, 62, 66), anchor="mm")
        self.omni("tag_push", self.save("tag_push", im), .35, metal=.85)
        steel, streak = art_brushed(srgb((.50, .52, .55)))
        self.omni("rail_steel_b", self.save("rail_steel", steel), .42, metal=.9, normal=self.save("rail_steel_n", normal_map(streak, 0.8)), bump=.25)
        im, d = _canvas(900, 220, (30, 32, 36))
        _logo(d, 30, 40, 90, (236, 236, 232), (30, 32, 36))
        d.text((30, 150), "TOOL RAIL 48  ·  5 kg / 11 lb per hook", font=_font(34, "regular"), fill=(236, 236, 232))
        self.omni("st_rail", self.save("st_rail", im), .5)
        self.omni("liner_film", self.flat("liner_film", (.020, .020, .022)), .30, normal=self.film, bump=.7)
        self.omni("paper_ruled", self.save("paper_ruled", art_notebook()), .85)
        self.omni("paper_plain", self.flat("paper_plain", (.88, .88, .85)), .85)
        self.omni("paper_yellow", self.flat("paper_yellow", (.90, .80, .36)), .85)
        self.omni("pencil_paint", self.save("pencil_paint", art_pencil(srgb((.93, .68, .08)))), .38)
        self.omni("glaze_out", self.save("glaze_out", art_glaze(srgb((.06, .16, .42)), 4)), .09)
        self.omni("pr_ream", self.save("pr_ream", art_ream()), .6)
        self.omni("pr_glue", self.save("pr_glue", art_glue()), .4)
        self.omni("pr_grad", self.save("pr_grad", art_grad([(q/14.5, str(q)) for q in (2, 4, 6, 8, 10, 12, 14)], "QT")), .45)
        return self


# ============================================================================== shared pieces of the tools
def _p(parts, name):
    return next(q for q in parts if q.name == name)


def _fillet(inset, z, r, up=True, k=4):
    """Profile points of a quarter round of radius r ending on the plane z (top if up)."""
    if up:
        return [(inset + r*(1 - math.cos(a)), z - r + r*math.sin(a)) for a in np.linspace(0, math.pi/2, k + 1)]
    return [(inset + r*(1 - math.sin(a)), z + r - r*math.cos(a)) for a in np.linspace(0, math.pi/2, k + 1)]


def _edge(t, er, k=3):
    """Profile across a plate of thickness t (centred on 0) whose edges are rounded with radius er."""
    lo = [(er*(1 - math.sin(a)), -t/2 + er*(1 - math.cos(a))) for a in np.linspace(0, math.pi/2, k + 1)]
    return lo + [(d, -z) for d, z in reversed(lo)]


def grip(A, r, s0, s1, R, p, mat="vinyl_black", proud=0.0013, pitch=0.0075):
    """A ribbed vinyl sleeve on a handle from s0 to s1 along local z: `proud` over the tube, shallow grooves."""
    prof = [(r + 0.0002, s0), (r + proud + 0.0008, s0 + 0.0012), (r + proud + 0.0008, s0 + 0.0055), (r + proud, s0 + 0.0068)]
    s = s0 + 0.009
    while s + pitch < s1 - 0.004:
        prof += [(r + proud, s), (r + proud - 0.0007, s + 0.0006), (r + proud - 0.0007, s + 0.0016), (r + proud, s + 0.0022)]
        s += pitch
    prof += [(r + proud, s1)]
    A.add(mat, revolve(prof, 28), R=R, p=p, crease=30.)


def hanger_cap(A, r, s_cap, Rc, hc, R, p, mat="vinyl_black", hole=0.0048):
    """The cap on the end of a hung handle: the collision capsule (radius Rc, half height hc, centre s_cap along the
    handle) drawn as a bulb with a hang hole through its ball. The underside that rests on the rail's prongs is the
    capsule's own lower sphere."""
    zl, zu = s_cap - hc, s_cap + hc
    a0 = -math.acos(min(1., (r + 0.0012)/Rc))
    prof = [(r + 0.0003, zl + Rc*math.sin(a0) - 0.0005)] + [(Rc*math.cos(a), zl + Rc*math.sin(a)) for a in np.linspace(a0, 0, 7)]
    g = hole + 0.0018                                              # the bulb's shoulder stays below the hang hole
    prof += [(Rc, max(zl, zu - g - 0.001)), (math.sqrt(Rc*Rc - g*g) - 0.0003, zu - g)]
    A.add(mat, revolve(prof, 32), R=R, p=p)
    A.add(mat, bead(Rc, hole, 32, 9), R=np.asarray(R) @ HOLE_Y, p=np.asarray(p) + np.asarray(R) @ (0., 0., zu))
    return zl + Rc*math.sin(a0)


# ============================================================================== brooms
def broom(kind, spec):
    """Angle broom and lobby broom: a moulded head (lower band the tufts are set in, a shoulder, a filleted cap),
    a socket the handle screws into, low ribs on the top, a steel handle with a ribbed vinyl grip and a hanger cap."""
    A = Asm()
    par, parts = spec["spec"], spec["parts"]
    bx, by, bz = par["block"]
    hx, hy, hz = bx/2, by/2, bz/2
    r = par["handle_d"]/2
    top = spec["handle_axis"][1][2]
    angle = kind == "angle_broom"
    head, face, wrap, sticker = ("pl_red", "pr_angle", "wr_angle", "st_angle") if angle else ("pl_black", "pr_lobby", "wr_lobby", "st_lobby")
    ins, rt, zs = 0.0012, 0.003, -hz + 0.40*bz
    prof = [(0.0006, -hz), (0., -hz + 0.0006), (0., zs), (ins, zs + ins)] + _fillet(ins, hz, rt)
    A.add(head, prism(rrect(hx, hy, 0.005, 5, 0.02), prof), uv=("box", GRAIN))
    rs = r + (0.004 if angle else 0.003)
    sock = [(rs + 0.0012, hz - 0.003), (rs + 0.0012, hz + 0.0015), (rs, hz + 0.004), (r + 0.0028, hz + 0.044), (r + 0.004, hz + 0.045),
            (r + 0.004, hz + 0.051), (r + 0.0028, hz + 0.052), (r + 0.0003, hz + 0.0523)]
    A.add(head, revolve(sock, 36), uv=("cyl", GRAIN))
    for k in range(4):                                             # thread relief on the socket's collar
        z = hz + 0.0462 + k*0.0012
        A.add(head, revolve([(r + 0.004, z), (r + 0.0045, z + 0.0003), (r + 0.004, z + 0.0006)], 36), uv=("cyl", GRAIN))
    rib = [(rs + 0.001, -0.001), (0.74*hx, -0.001), (0.74*hx, 0.0006), (rs + 0.001, 0.0028)]
    for turn in (np.eye(3), np.diag([-1., -1., 1.])):
        A.add(head, prism(rib, _edge(0.0034, 0.0006, 2)), R=turn @ XZ, p=(0, 0, hz), uv=("box", GRAIN))
    I = np.eye(3)
    cap = _p(parts, "cap")
    z_low = hanger_cap(A, r, cap.pos[2], cap.size[0], cap.size[1], I, (0, 0, 0))
    A.add("handle_steel", revolve([(r, hz + 0.05), (r, z_low + 0.001)], 28))
    grip(A, r, top - 0.165, z_low + 0.0006, I, (0, 0, 0))
    # print: the head's faces (hot stamp in the head's own plastic), a paper label on the back, a wrap on the handle
    zc, hgt = (zs + ins + hz - rt)/2, (hz - rt) - (zs + ins) - 0.004
    A.add(face, decal((0, hy - ins, zc), (0, 1, 0), (0, 0, 1), min(2*hx - 0.03, hgt*2100/190), hgt))
    A.add(sticker, decal((-0.45*hx, -(hy - ins), zc), (0, -1, 0), (0, 0, 1), min(0.062, 0.9*hx), min(0.031, hgt), rc=0.002))
    z0 = hz + 0.16
    A.add(wrap, decal_cyl(r, math.pi/2, TAU*r*0.86, z0, z0 + TAU*r*0.86*900/520))
    return A


def push_broom(spec):
    """Push broom: a lacquered hardwood block with chamfered top edges, a bolted handle bracket with an angled
    socket, a threaded steel tip on a wooden handle, a hanger cap, an ink stamp and a size tag."""
    A = Asm()
    par, parts = spec["spec"], spec["parts"]
    bx, by, bz = par["block"]
    hx, hy, hz = bx/2, by/2, bz/2
    r = par["handle_d"]/2
    foot, end = (np.asarray(v, float) for v in spec["handle_axis"])
    h = (end - foot)/np.linalg.norm(end - foot)
    R = basis(h)
    A.add("wood_beech", prism(rrect(hx, hy, 0.004, 4, 0.03), [(0.0005, -hz), (0., -hz + 0.0005), (0., hz - 0.003), (0.003, hz)]), uv=("box", 3.2))
    yb = -0.010                                                    # bracket: a plate under the socket, four bolts
    A.add("pl_black", prism(rrect(0.034, 0.021, 0.004, 4), [(0.0006, 0.), (0., 0.0006), (0., 0.0024), (0.0006, 0.003)]), p=(0, yb, hz - 0.0002),
          uv=("box", GRAIN))
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx*0.026, yb + sy*0.0145, hz + 0.0027)
            A.add("steel_zinc", revolve([(0.0058, 0.), (0.0058, 0.0006), (0.0, 0.0006)], 20), p=c)
            A.add("steel_zinc", prism(circle(0.0042, 6), [(0.0004, 0.0005), (0., 0.0009), (0., 0.0027), (0.0006, 0.0032)]), p=c,
                  R=Rotation.from_euler("z", 0.3*sx + 0.7*sy).as_matrix())
    sock = [(r + 0.0035, -0.012), (r + 0.0035, 0.050), (r + 0.0048, 0.051), (r + 0.0048, 0.058), (r + 0.0035, 0.059), (r + 0.0009, 0.0594)]
    A.add("pl_black", revolve(sock, 36), R=R, p=foot, uv=("cyl", GRAIN))
    A.add("steel_zinc", revolve([(r + 0.0009, 0.058), (r + 0.0009, 0.074), (r + 0.0002, 0.0745)], 28), R=R, p=foot)
    for k in range(5):
        s = 0.0615 + k*0.0024
        A.add("steel_zinc", revolve([(r + 0.0009, s), (r + 0.0003, s + 0.0006), (r + 0.0009, s + 0.0012)], 28), R=R, p=foot, crease=20.)
    cap = _p(parts, "cap")
    s_cap = float((np.asarray(cap.pos) - foot) @ h)
    s_low = hanger_cap(A, r, s_cap, cap.size[0], cap.size[1], R, foot)
    A.add("wood_beech", revolve([(r, 0.07), (r, s_low + 0.001)], 28), R=R, p=foot, uv=("grain", 1.4))
    A.add("stamp_push", decal((-0.26*bx, 0.004, hz), (0, 0, 1), (0, -1, 0), 0.170, 0.170*180/1280, lift=0.00012))
    A.add("tag_push", decal((0.34*bx, 0.006, hz), (0, 0, 1), (0, -1, 0), 0.036, 0.015, rc=0.002))
    for sx in (-1, 1):                                             # the tag's two pins
        A.add("steel_zinc", revolve([(0.0011, 0.), (0.0008, 0.0005), (0., 0.0006)], 10), p=(0.34*bx + sx*0.0155, 0.006, hz + 0.0002))
    A.add("st_push", decal((0.22*bx, hy, 0.0), (0, 1, 0), (0, 0, 1), 0.060, 0.030, rc=0.002))
    A.add("wr_push", decal_cyl(r, math.pi/2, TAU*r*0.8, 0.30, 0.30 + TAU*r*0.8*900/520), R=R, p=foot)
    return A


def counter_brush(spec):
    """Counter brush: one piece of lacquered beech, the block running through a moulded neck into a turned handle
    with a hang hole in its ball end, a brass ferrule on the neck."""
    A = Asm()
    par, parts = spec["spec"], spec["parts"]
    bx, by, bz = par["block"]
    hx, hy, hz = bx/2, by/2, bz/2
    hp = _p(parts, "handle")
    r = hp.size[0]
    tip = spec["handle_axis"][1][0]
    A.add("wood_beech", prism(rrect(hx, hy, 0.0045, 4, 0.03), [(0.0005, -hz), (0., -hz + 0.0005)] + _fillet(0., hz, 0.003)), uv=("box", 4.0))
    xb = tip - r                                                   # centre of the handle's ball end
    sect = [(hx - 0.006, hy - 0.0002, hz - 0.0002, 5.), (hx, hy - 0.0002, hz - 0.0002, 5.), (hx + 0.006, r + 0.0008, hz - 0.0002, 3.),
            (hx + 0.014, r - 0.0006, hz, 2.3), (hx + 0.024, r - 0.0012, r - 0.0012, 2.), (hx + 0.048, r, r, 2.), (hx + 0.074, r + 0.0002, r + 0.0002, 2.),
            (xb - 0.010, r - 0.0003, r - 0.0003, 2.), (xb - 0.0052, math.sqrt(r*r - 0.0052**2) - 0.0002, math.sqrt(r*r - 0.0052**2) - 0.0002, 2.)]
    rings = []
    for x, ay, az, p in sect:
        o = superring(ay, az, p, 28)
        rings.append(np.column_stack([np.full(len(o), x), o[:, 0], o[:, 1]]))
    v, f = loft(rings)
    A.add("wood_beech", (v @ ALONG_X, f), R=ALONG_X, uv=("grain", 3.0))     # lofted in the body frame; uv taken about x
    A.add("wood_beech", bead(r, 0.0042, 30, 9), R=HOLE_Y, p=(xb, 0, 0))
    A.add("brass", revolve([(r - 0.0003, 0.), (r + 0.0004, 0.0006), (r + 0.0004, 0.0064), (r - 0.0003, 0.007)], 30), R=ALONG_X, p=(hx + 0.0185, 0, 0))
    A.add("stamp_brush", decal((-0.2*hx, hy, 0.), (0, 1, 0), (0, 0, 1), 0.110, 0.110*150/960, lift=0.00012))
    A.add("st_brush", decal((0.42*hx, -hy, 0.), (0, -1, 0), (0, 0, 1), 0.036, 0.018, rc=0.0015))
    return A


# ============================================================================== dustpans
def _pan_shell(A, parts, mat, lip_mat, ribs=(), er=0.0011):
    """The five plates of a pan as one moulding: floor slab, lip, two side walls and a back with radiused top edges,
    the hood. Every plate is its collision box. The lip is drawn solid from the ground up to the top face of its
    collision plate, wherever that plate is (see NOTES: as built it rises toward the mouth); the walls go down to the
    ground and so close the gap the collision leaves between their lower edge and the low end of the lip."""
    fl, lip, sl, back, hood = (_p(parts, n) for n in ("floor", "lip", "side_l", "back", "hood"))
    hw, tray = fl.size[0], 2*fl.size[2]
    y0, y1 = fl.pos[1] - fl.size[1], fl.pos[1] + fl.size[1]
    hd, wall = sl.size[1], 2*sl.size[0]
    zs, zb = sl.pos[2] + sl.size[2], back.pos[2] + back.size[2]
    yh, th = hood.pos[1] + hood.size[1], 2*hood.size[2]
    Rl = Rotation.from_quat(lip.quat)                              # the lip plate's top face: its rear and front edges
    (_, yr, zr), (_, yf, zf) = (np.asarray(lip.pos) + Rl.apply((0., s*lip.size[1], lip.size[2])) for s in (-1, 1))
    A.add(mat, prism([(-hw, y0), (hw, y0), (hw, y1), (-hw, y1)], [(0., 0.), (0., tray)]), uv=("box", GRAIN))
    A.add(lip_mat, prism([(yr, 0.), (yf, 0.), (yf, zf), (yr, zr)], [(0., -hw), (0., hw)]), R=YZ, uv=("box", GRAIN), crease=20.)
    side = rounded([(-hd, 0.), (hd, 0.), (hd, zs), (-hd, zs)], [0., 0., 0.005, 0.002], 5, 0.03)
    for sx in (-1, 1):
        A.add(mat, prism(side, _edge(wall, er)), R=YZ, p=(sx*(hw - wall/2), 0, 0), uv=("box", GRAIN))
    A.add(mat, prism(rounded([(-hw, 0.), (hw, 0.), (hw, zb), (-hw, zb)], [0., 0., 0.002, 0.002], 4, 0.03), _edge(wall, er)), R=XZ,
          p=(0, -hd + wall/2, 0), uv=("box", GRAIN))
    A.add(mat, prism(rounded([(-hw, -hd), (hw, -hd), (hw, yh), (-hw, yh)], [0.002, 0.002, 0.004, 0.004], 4, 0.03), _edge(th, min(er, 0.45*th))),
          p=(0, 0, hood.pos[2]), uv=("box", GRAIN))
    for sx in (-1, 1):                                             # outside of each side wall: a bead under its top edge, ribs
        Rs = np.diag([sx, sx, 1.]) @ YZ
        for zc, hh, ya, yb in [(zs - 0.0045, 0.0035, -hd + 0.003, hd - 0.004)] + [(z, 0.0022, -hd + 0.012, hd - 0.030) for z in ribs]:
            A.add(mat, prism(rrect((yb - ya)/2, hh, min(0.002, 0.8*hh), 3, 0.04), [(0.0009, -0.001), (0., -0.0001), (0., 0.0005), (0.0008, 0.0013)]), R=Rs,
                  p=(sx*hw, (ya + yb)/2, zc), uv=("box", GRAIN))
    # the hood's front edge is rolled down 2 mm
    A.add(mat, tube([(-hw + 0.004, yh - 0.0022, zb - 0.0026), (hw - 0.004, yh - 0.0022, zb - 0.0026)], 0.0024, 10, True, True), uv=("box", GRAIN))
    return dict(hw=hw, hd=hd, tray=tray, wall=wall, zs=zs, zb=zb, yh=yh, th=th, y1=y1, lip=((yr, zr), (yf, zf)))


def lobby_pan(spec):
    """Lobby dustpan, the pan body: moulded shell, ribs and wear strips on the back, pivot bosses, moulded lettering
    on the hood. Returns (pan, handle): the handle (yoke, hub, steel tube, grip) is drawn in the handle body's frame,
    whose origin is the pivot."""
    A, H = Asm(), Asm()
    parts = spec["parts"]
    g = _pan_shell(A, parts, "pl_black", "pl_black", ribs=(0.040, 0.070))
    hw, hd, zb = g["hw"], g["hd"], g["zb"]
    for x in (-0.085, 0., 0.085):                                  # ribs on the back, 1.5 mm proud
        A.add("pl_black", prism(rrect(0.004, 0.046, 0.002, 3), [(0.0009, -0.001), (0., -0.0001), (0., 0.0006), (0.0009, 0.0015)]),
              R=XZ, p=(x, -hd, 0.062), uv=("box", GRAIN))
    for wh in spec.get("wheels", ()):                              # the two rear wheels, where their bodies are (a plain
        part = wh["parts"][0]                                      # disc: its turning cannot be seen, so it is drawn with the pan)
        r, h = part.size
        x, y, z = wh["pos"]
        A.add("pl_grey", revolve([(0.0045, -h), (r - 0.0025, -h), (r, -h + 0.0025), (r, h - 0.0025), (r - 0.0025, h), (0.0045, h)], 32),
              R=ALONG_X, p=(x, y, z), uv=("cyl", GRAIN))
        A.add("pl_black", revolve([(0.0045, -h - 0.002), (0.0045, h + 0.002), (0., h + 0.002)], 16), R=ALONG_X @ np.diag([np.sign(x), 1., np.sign(x)]),
              p=(x, y, z), uv=("cyl", GRAIN))                      # axle stub and its head
    piv = np.asarray(spec["pivot"], float)
    for sx in (-1, 1):                                             # pivot bosses between the wall and the yoke's knuckle
        A.add("pl_black", revolve([(0.0062, 0.), (0.0062, 0.0022), (0.0054, 0.003), (0., 0.003)], 24), R=ALONG_X @ np.diag([sx, 1., sx]),
              p=(sx*hw, piv[1], piv[2]), uv=("cyl", GRAIN))
    yh = g["yh"]
    A.add("em_pan", decal((0, (-hd + yh)/2, zb), (0, 0, 1), (0, -1, 0), 2*hw - 0.012, (yh + hd) - 0.012, rc=0.003, lift=0.00015))
    A.add("st_pan", decal((0.045, -hd, 0.100), (0, -1, 0), (0, 0, 1), 0.056, 0.028, rc=0.002, lift=0.0003))
    # ---- handle body (origin at the pivot)
    hp = spec["handle_parts"]
    hd_, yl, yr = (_p(hp, n) for n in ("handle", "yoke_l", "yoke_r"))
    r = hd_.size[0]
    base, top = (np.asarray(v, float) for v in spec["handle_axis"])    # the handle's two tips, in the handle body's frame
    z_top = top[2]
    for part in (yl, yr):
        Rp = Rotation.from_quat(part.quat)
        ry, hh = part.size
        a, b = np.asarray(part.pos) - Rp.apply((0, 0, hh + ry)), np.asarray(part.pos) + Rp.apply((0, 0, hh + ry))
        H.add("steel_zinc", rod(a, b, ry, 14))
        lo = a if a[2] < b[2] else b                               # the arm's tip is on the hinge line: a knuckle and a rivet there
        sx = 1. if lo[0] > 0 else -1.
        Rk = ALONG_X @ np.diag([sx, 1., sx])
        H.add("pl_black", revolve([(0., -0.0045), (0.0062, -0.0045), (0.0068, -0.0039), (0.0068, 0.0039), (0.0062, 0.0045), (0., 0.0045)], 22), R=Rk, p=lo)
        H.add("steel_zinc", revolve([(0.0036, 0.0044), (0.0036, 0.0050), (0.0022, 0.0058), (0., 0.0059)], 16), R=Rk, p=lo)
    # hub: a cross piece that takes the two arms, and the socket the handle stands in
    H.add("pl_black", rod(base + (-0.0385, 0, 0), base + (0.0385, 0, 0), 0.0085, 18), uv=("box", GRAIN))
    sock = [(0., -0.0085), (0.0070, -0.0078), (0.0118, -0.0050), (0.0145, 0.0), (0.0145, 0.038), (0.0156, 0.039), (0.0156, 0.045),
            (0.0145, 0.046), (r + 0.0004, 0.0465)]
    H.add("pl_black", revolve(sock, 32), p=base, uv=("cyl", GRAIN))
    Rg = r + 0.0013
    zc = z_top - Rg                                                # the grip's ball end reaches the handle's tip
    H.add("handle_steel", revolve([(r, base[2] + 0.04), (r, z_top - 0.14)], 28), p=(base[0], base[1], 0.))
    grip(H, r, z_top - 0.150, zc - 0.0045, np.eye(3), (base[0], base[1], 0.))
    H.add("vinyl_black", bead(Rg, 0.0042, 30, 9), R=HOLE_Y, p=(base[0], base[1], zc))
    z0 = base[2] + 0.12
    H.add("wr_pan", decal_cyl(r, math.pi/2, TAU*r*0.86, z0, z0 + TAU*r*0.86*900/520), p=(base[0], base[1], 0.))
    return A, H


def hand_pan(spec):
    """Hand dustpan: an open scoop moulded in charcoal plastic. Floor slab, lip, back wall and the two side walls,
    each side drawn as one plate whose top edge falls from the rear box's height to the front box's (the collision
    has the two boxes); a flat handle with a hang hole near its end; the maker's sticker on the back wall."""
    A = Asm()
    parts = spec["parts"]
    fl, lip, back, sr, sf = (_p(parts, n) for n in ("floor", "lip", "back", "side_l", "side_lf"))
    hw, tray = fl.size[0], 2*fl.size[2]
    y0, y1 = fl.pos[1] - fl.size[1], fl.pos[1] + fl.size[1]
    wall = 2*sr.size[0]
    zb = back.pos[2] + back.size[2]
    zr, zf = sr.pos[2] + sr.size[2], sf.pos[2] + sf.size[2]
    ys = sr.pos[1] + sr.size[1]                                   # where the rear box ends and the front box begins
    Rl = Rotation.from_quat(lip.quat)
    (_, yr, zl0), (_, yf, zl1) = (np.asarray(lip.pos) + Rl.apply((0., s_*lip.size[1], lip.size[2])) for s_ in (-1, 1))
    mat = "pl_black"
    A.add(mat, prism([(-hw, y0), (hw, y0), (hw, y1), (-hw, y1)], [(0., 0.), (0., tray)]), uv=("box", GRAIN))
    A.add(mat, prism([(yr, 0.), (yf, 0.), (yf, zl1), (yr, zl0)], [(0., -hw), (0., hw)]), R=YZ, uv=("box", GRAIN), crease=20.)
    side = rounded([(y0, 0.), (y1, 0.), (y1, zf), (ys, zr), (y0, zr)], [0., 0., 0.004, 0.02, 0.003], 5, 0.03)
    for sx in (-1, 1):
        A.add(mat, prism(side, _edge(wall, 0.0011)), R=YZ, p=(sx*(hw - wall/2), 0, 0), uv=("box", GRAIN))
    A.add(mat, prism(rounded([(-hw, 0.), (hw, 0.), (hw, zb), (-hw, zb)], [0., 0., 0.004, 0.004], 4, 0.03), _edge(wall, 0.0011)), R=XZ,
          p=(0, y0 + wall/2, 0), uv=("box", GRAIN))
    for k in range(9):                                            # the moulded ribs on the floor, running toward the mouth
        x = (k - 4)*0.028
        A.add(mat, prism(rrect(0.004, 0.045, 0.003, 3, 0.04), [(0.0008, tray), (0., tray + 0.0009)]), p=(x, -0.012, 0), uv=("box", GRAIN))
    A.add("st_hand", decal((0.085, y0, 0.034), (0, -1, 0), (0, 0, 1), 0.050, 0.025, rc=0.002, lift=0.0003))
    hp = _p(parts, "handle")
    r, hh = hp.size
    Rp = Rotation.from_quat(hp.quat)
    a, b = np.asarray(hp.pos) - Rp.apply((0, 0, hh + r)), np.asarray(hp.pos) + Rp.apply((0, 0, hh + r))
    if a[1] < b[1]:
        a, b = b, a                                                # a at the back wall, b the free tip
    length = float(np.linalg.norm(b - a))
    u = (b - a)/length
    R = basis(u)                                                   # local z along the handle
    A.add(mat, prism(rrect(r, 0.45*r, 0.4*r, 4, 0.04), [(0.001, 0.0), (0., 0.002), (0., length - 0.003), (0.0015, length)]), R=R, p=a, uv=("box", GRAIN))
    A.add(mat, bead(0.55*r, 0.0030, 26, 8), R=R @ HOLE_Y, p=a + u*(length - 1.6*r), uv=("cyl", GRAIN))
    return A


# ============================================================================== containers
def _mean_radii(r, wall, n):
    """The collision wall is n flat staves: its outer face lies between r (stave centres) and r / cos(pi/n) (their
    corners), its inner face between r - wall and (r - wall) / cos(pi/n). A round wall is drawn midway: (outer radius,
    inner radius, largest distance from either face)."""
    k = 1/math.cos(math.pi/n)
    return r*(1 + k)/2, (r - wall)*(1 + k)/2, r*(k - 1)/2


def barrel(spec):
    """Round refuse barrel: base ring, straight wall with eight low ribs, a rolled rim with two moulded hand holds
    under it, a black liner bag folded over the rim and hanging in drapes, moulded lettering and a label."""
    A = Asm()
    par = spec["spec"]
    r, h, wall = par["d"]/2, par["h"], par["wall"]
    n = len(spec["parts"]) - 1
    base = 2*_p(spec["parts"], "bottom").size[1]
    Ro, Ri, _ = _mean_radii(r, wall, n)
    rim = 0.0035                                                   # how far the rolled rim stands out
    out = [(Ro - 0.003, 0.), (Ro + 0.0025, 0.004), (Ro + 0.0025, 0.030), (Ro, 0.035), (Ro, h - 0.034), (Ro + rim*0.6, h - 0.031),
           (Ro + rim, h - 0.026), (Ro + rim, h - 0.006), (Ro + rim - 0.0015, h - 0.0015), (Ro + rim - 0.004, h), (Ri + 0.001, h), (Ri, h - 0.001),
           (Ri, base)]
    A.add("pl_barrel", revolve([(0., 0.)] + out + [(0., base)], 96), uv=("cyl", GRAIN), crease=50.)
    for k in range(8):                                             # ribs: pairs on the diagonals, 2 mm proud
        a = math.radians(45 + 90*(k//2) + (7 if k % 2 else -7))
        Rz = Rotation.from_euler("z", a).as_matrix()
        A.add("pl_barrel", prism(rrect(0.006, 0.19, 0.004, 3), [(0.0016, -0.002), (0., 0.0004), (0., 0.0012), (0.0016, 0.002)]),
              R=Rz @ np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), p=Rz @ (Ro, 0., 0.27), uv=("box", GRAIN))
    ears = (math.pi/2, 3*math.pi/2)                                # hand holds under the rim, 4 mm proud, at two corners of the staves
    ez0, ez1, ea, eh = h - 0.096, h - 0.039, 0.21, 0.0040

    def ear(th, z):
        d = np.min([np.abs((th - a + math.pi) % TAU - math.pi) for a in ears], 0)
        wa = np.clip((ea - d)/0.035, 0, 1)
        wz = np.clip((z - ez0)/0.006, 0, 1)*np.clip((ez1 + 0.004 - z)/0.004, 0, 1)
        s = wa*wz
        return (eh + 0.0003)*s*s*(3 - 2*s)

    Rh = Ro - 0.0003                                               # where there is no hold this strip lies just under the wall
    hold = [(Rh - 0.001, ez0 - 0.002), (Rh, ez0), (Rh, ez0 + 0.004), (Rh, ez0 + 0.008)] + [(Rh, z) for z in np.linspace(ez0 + 0.012, ez1 + 0.004, 8)]
    A.add("pl_barrel", revolve(hold, 240, mod=lambda th, z, rr: (ear(th, z), 0*th)), uv=("cyl", GRAIN), crease=60.)
    # liner: over the rim, an uneven hem, drapes that stand off the wall toward the hem; inside to the bottom
    def bag(th, z, rr):
        out_side = (rr > (Ro + Ri)/2) & (z < h - 0.001)
        t = np.clip((h - 0.036 - z)/0.030, 0, 1)                   # 0 from the rim down to under its roll, 1 at the hem
        near = np.min([np.abs((th - a + math.pi) % TAU - math.pi) for a in ears], 0)       # the hem rides up over the two hand holds
        hem = 0.012*np.sin(3*th + 1.0) + 0.007*np.sin(7*th + 0.3) + 0.004*np.sin(13*th + 2.0) + 0.003*np.sin(29*th) + 0.020*np.exp(-(near/0.30)**2)
        drape = 0.0008*t*(np.abs(np.sin(17*th + 4*np.sin(3*th))) + 0.5*np.abs(np.sin(41*th + 1.3)))
        return np.where(out_side, np.maximum(ear(th, z + hem*t), 0) + drape, 0.), np.where(out_side, hem*t, 0.)

    g = 0.0010
    zh = h - 0.066
    liner = [(Ro + g, z) for z in np.linspace(zh, h - 0.036, 9)] + [(Ro + rim*0.6 + g, h - 0.031), (Ro + rim + g, h - 0.026), (Ro + rim + g, h - 0.006),
            (Ro + rim - 0.001 + g, h - 0.0012 + g*0.6), (Ro + rim - 0.004, h + g), (Ri + 0.001, h + g), (Ri - g, h - 0.001), (Ri - g, base + 0.02),
            (Ri - g - 0.004, base + 0.004), (Ri - 0.02, base + g), (0., base + g)]
    A.add("liner_film", revolve(liner, 240, mod=bag), uv=("cyl", 7.0), crease=75.)
    A.add("em_barrel", decal_cyl(Ro, math.pi, 0.18, 0.345, 0.435, lift=0.0002))
    A.add("st_barrel", decal_cyl(Ro, math.pi, 0.12, 0.245, 0.305, lift=0.0003))
    return A


def pail(spec):
    """Utility pail: foot ring, straight wall, two reinforcing rings under a rolled rim, ears with a wire bail lying
    against the wall and a grip on the bail, a quart scale printed inside, a label."""
    A = Asm()
    par = spec["spec"]
    r, h, wall = par["d"]/2, par["h"], par["wall"]
    n = len(spec["parts"]) - 1
    base = 2*_p(spec["parts"], "bottom").size[1]
    Ro, Ri, _ = _mean_radii(r, wall, n)
    rim = 0.0035
    out = [(0., 0.), (Ro - 0.002, 0.), (Ro + 0.0015, 0.003), (Ro + 0.0015, 0.014), (Ro, 0.017)]
    for z in (h - 0.060, h - 0.040):
        out += [(Ro, z - 0.003), (Ro + 0.0014, z - 0.0018), (Ro + 0.0014, z + 0.0018), (Ro, z + 0.003)]
    out += [(Ro, h - 0.024), (Ro + rim*0.6, h - 0.022), (Ro + rim, h - 0.018), (Ro + rim, h - 0.005), (Ro + rim - 0.0012, h - 0.0012),
            (Ro + rim - 0.003, h), (Ri + 0.0008, h), (Ri, h - 0.0008), (Ri, base + 0.003), (Ri - 0.003, base), (0., base)]
    A.add("pl_pail", revolve(out, 96), uv=("cyl", GRAIN), crease=50.)
    # ears, bail, label and scale are turned so that the ears and the grip sit over corners of the staves, where the
    # round wall is drawn inside the collision wall
    T = Rotation.from_euler("z", math.pi/n - math.pi/2).as_matrix()
    ze, Rb, rw = h - 0.050, Ro + 0.0060, 0.0022                    # ears, the bail's radius, its wire
    Rg = 0.0042                                                    # the grip on the bail
    phi = math.acos(min(1., (Ro + Rg + 0.0003)/Rb))                # the bail droops until its grip lies against the wall
    for sx in (-1, 1):
        Rx = T @ ALONG_X @ np.diag([sx, 1., sx])
        A.add("pl_pail", revolve([(0.0105, -0.002), (0.0105, 0.0030), (0.0088, 0.0048), (0.0045, 0.0052), (0., 0.0052)], 24), R=Rx, p=T @ (sx*Ro, 0, ze),
              uv=("cyl", GRAIN))
        A.add("steel_zinc", revolve([(0.0032, 0.), (0.0032, 0.0005), (0.0018, 0.0010), (0., 0.0011)], 14), R=Rx, p=T @ (sx*(Rb + rw), 0, ze))
    t = np.linspace(0, math.pi, 49)
    path = np.column_stack([Rb*np.cos(t), -Rb*np.sin(t)*math.cos(phi), ze - Rb*np.sin(t)*math.sin(phi)])
    A.add("steel_zinc", tube(path, rw, 10), R=T)
    gi = np.flatnonzero(np.abs(t - math.pi/2) < 0.29)
    gr = np.full(len(gi), Rg)
    gr[[0, -1]] = Rg - 0.0012
    A.add("grip_white", tube(path[gi], gr, 16, True, True, 3), R=T)
    A.add("st_pail", decal_cyl(Ro, -math.pi/2, 0.10, 0.075, 0.125, lift=0.0003), R=T)
    A.add("pr_grad", decal_cyl(Ri, math.radians(100), 0.030, base + 0.012, base + 0.012 + 0.200, inside=True, lift=0.0003), R=T)
    return A


def wastebasket(spec):
    """Rectangular wastebasket: a body set 1 mm back under a rim band, a rolled rim, moulded lettering on one long
    side and a label on the other, a dark inside."""
    A = Asm()
    par = spec["spec"]
    x, y, h = par["size"]
    w = par["wall"]
    hx, hy = x/2, y/2
    rim = 0.0025
    out = [(0.0014, 0.), (0.0008, 0.0012), (0.0008, h - 0.052), (0., h - 0.049), (0., h - 0.014), (-rim*0.6, h - 0.012), (-rim, h - 0.009),
           (-rim, h - 0.004), (-rim + 0.001, h - 0.001), (-rim + 0.0025, h), (w - 0.0008, h), (w, h - 0.0008)]
    o = rrect(hx, hy, 0.0035, 5, 0.04)
    A.add("pl_basket", prism(o, out, cap1=False), uv=("box", GRAIN), crease=50.)
    A.add("basket_in", prism(o, [(w, h - 0.0008), (w, w)], cap0=False), crease=50.)                  # inner faces and bottom
    A.add("em_basket", decal((0, -hy + 0.0008, 0.5*h + 0.02), (0, -1, 0), (0, 0, 1), 0.16, 0.08, lift=0.0002))
    A.add("st_basket", decal((0.07, hy - 0.0008, 0.22), (0, 1, 0), (0, 0, 1), 0.07, 0.035, rc=0.002, lift=0.0003))
    return A


def tool(kind, spec):
    """(picture of the tool's body, picture of its second body or None)."""
    if kind in ("angle_broom", "lobby_broom"):
        return broom(kind, spec), None
    if kind == "lobby_pan":
        return lobby_pan(spec)
    return dict(push_broom=push_broom, counter_brush=counter_brush, hand_pan=hand_pan, barrel=barrel, pail=pail,
                wastebasket=wastebasket)[kind](spec), None


# ============================================================================== wall rail (installed part, world frame)
def rail(strips, capsules):
    """strips: [(name, centre, half extents)] of the rail's collision boxes on the right wall (thin along x);
    capsules: [(name, a, b, radius)] of the prongs and their upturned tips. Steel strip with a broken edge and pan
    head screws, each prong one bent rod with a hex nut at the strip and a vinyl tip."""
    A = Asm()
    TO_WALL = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])          # outline (u, v) -> (-y, z); thickness -> -x
    DOME = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])              # local z -> -x
    prong_y = sorted({round(float(a[1]), 4) for name, a, b, r in capsules})
    for name, c, hh in strips:
        x_wall = c[0] + hh[0]
        t = 2*hh[0]
        A.add("rail_steel_b", prism(rrect(hh[1], hh[2], 0.003, 4, 0.05), [(0., 0.), (0., t - 0.001), (0.001, t)]), R=TO_WALL, p=(x_wall, c[1], c[2]),
              uv=("box", 6.0))
        m = max(2, int(round(2*hh[1]/0.33)) + 1)
        for y in np.linspace(c[1] - hh[1] + 0.03, c[1] + hh[1] - 0.03, m):
            if any(abs(y - py) < 0.03 for py in prong_y):
                y += 0.05
            p = np.array([x_wall - t, y, c[2]])
            A.add("steel_zinc", revolve([(0.0052, 0.), (0.0052, 0.0005), (0.0044, 0.0017), (0.0026, 0.0025), (0., 0.0027)], 20), R=DOME, p=p)
            a = 0.5 + 2.3*math.sin(7*y)
            for da in (0., math.pi/2):
                Rs = DOME @ Rotation.from_euler("z", a + da).as_matrix()
                A.add("slot", prism(rrect(0.0030, 0.00045, 0.0002, 2), [(0., 0.0024), (0., 0.00285)]), R=Rs, p=p)
        if 2*hh[1] > 1.0:
            A.add("st_rail", decal((x_wall - t, c[1] + hh[1] - 0.105, c[2] - 0.012), (-1, 0, 0), (0, 0, 1), 0.090, 0.022, rc=0.002))
    rods = {}
    for name, a, b, r in capsules:
        rods.setdefault(name.replace("prong_tip_", "").replace("prong_", ""), {})["tip" if "tip" in name else "rod"] = (np.asarray(a, float), np.asarray(b, float), r)
    for key, d in rods.items():
        a, b, r = d["rod"]
        free, wall_end = (a, b) if a[0] < b[0] else (b, a)        # the wall is at the larger x
        ta, tb, _ = d["tip"]
        lo, hi = (ta, tb) if ta[2] < tb[2] else (tb, ta)
        u = (hi - lo)/np.linalg.norm(hi - lo)
        path = [wall_end + (-0.004, 0, 0), free + (r, 0, 0), lo + u*r, hi - u*r]
        A.add("steel_zinc", tube(path, r, 12, False, True))
        A.add("vinyl_black", tube([lo + u*(r + 0.002), hi - u*r], r + 0.0008, 12, False, True))
        x_face = wall_end[0] - 0.012
        A.add("steel_zinc", prism(circle(0.0075, 6), [(0.0006, 0.), (0., 0.0006), (0., 0.0030), (0.0008, 0.0036)]), R=DOME,
              p=(x_face, wall_end[1], wall_end[2]))
    return A


# ============================================================================== supplies on the craft table
def supply(kind, parts):
    """cup: a square pencil cup holding crayons and short pencils (all below its rim); ream: a wrapped ream with a
    printed wrapper; glue: a square-shouldered bottle with a twist cap. The scissors keep their old picture."""
    A = Asm()
    p = parts[0]
    hx, hy, hz = p.size
    if kind == "cup":
        w = 0.0025
        o = rrect(hx, hy, 0.005, 5)
        A.add("pl_cup", prism(o, [(0.0015, 0.), (0., 0.0015), (0., 2*hz - 0.001), (0.0008, 2*hz), (w - 0.0006, 2*hz), (w, 2*hz - 0.0008), (w, w + 0.001),
                                  (w + 0.001, w)]), uv=("box", GRAIN), crease=50.)
        rng = np.random.default_rng(5)
        k = 0
        for gx in (-1, 0, 1):
            for gy in (-1, 0, 1):
                if (gx, gy) == (0, 0):
                    continue
                foot = np.array([gx*0.017 + rng.uniform(-.003, .003), gy*0.017 + rng.uniform(-.003, .003), w + 0.001])
                lean = np.array([gx*0.006 + rng.uniform(-.004, .004), gy*0.006 + rng.uniform(-.004, .004), 1.])
                R = basis(lean/np.linalg.norm(lean))
                L_ = rng.uniform(0.078, 0.090)
                if k % 3 == 2:                                     # a short pencil, point up
                    A.add("pencil_paint", prism(circle(0.0035, 6), [(0., 0.), (0., L_ - 0.016)]), R=R, p=foot, uv=("wrap", 0., 0.19))
                    A.add("pencil_wood", prism(circle(0.0035, 12), [(0.0002, L_ - 0.016), (0.0024, L_ - 0.004)], cap0=False, cap1=False), R=R, p=foot)
                    A.add("graphite", prism(circle(0.0035, 12), [(0.0024, L_ - 0.004), (0.0032, L_)], cap0=False), R=R, p=foot)
                else:                                              # a crayon in its paper sleeve
                    c = f"crayon_{k % 6}"
                    A.add(c, revolve([(0., 0.), (0.004, 0.), (0.004, L_ - 0.012), (0.0022, L_ - 0.002), (0.0014, L_), (0., L_)], 14), R=R, p=foot)
                    A.add("crayon_wrap", revolve([(0.0042, 0.012), (0.0042, L_ - 0.02)], 14), R=R, p=foot)
                    A.add(c, revolve([(0.0043, 0.03), (0.0043, 0.036)], 14), R=R, p=foot)
                k += 1
    elif kind == "ream":
        c = np.asarray(p.pos, float)
        A.add("pl_ream", prism(rrect(hx, hy, 0.002, 3), [(0.001, -hz), (0., -hz + 0.001), (0., hz - 0.001), (0.001, hz)]), p=c, uv=("box", GRAIN))
        A.add("pr_ream", decal(c + (0, 0, hz), (0, 0, 1), (0, 1, 0), 2*hx - 0.006, 2*hy - 0.006, lift=0.0002))
        for sy in (-1, 1):                                         # the wrapper's folded and glued end flaps
            A.add("paper_in", decal(c + (0, sy*hy, 0), (0, sy, 0), (0, 0, 1), 2*hx - 0.02, 2*hz - 0.012, rc=0.004, lift=0.0002))
    elif kind == "glue":
        z1 = 2*hz
        A.add("pl_glue", prism(rrect(hx, hy, 0.006, 5), [(0.001, 0.), (0., 0.001)] + _fillet(0., z1 - 0.034, 0.004)), uv=("box", GRAIN), crease=50.)
        A.add("pl_glue", revolve([(hx - 0.005, z1 - 0.0345), (0.0125, z1 - 0.027), (0.0090, z1 - 0.0235), (0.0090, z1 - 0.019)], 28), uv=("cyl", GRAIN))
        A.add("glue_cap", revolve([(0.0104, z1 - 0.0205), (0.0104, z1 - 0.0125), (0.0064, z1 - 0.0085), (0.0036, z1 - 0.0006), (0., z1)], 28))
        for k in range(3):
            z = z1 - 0.0185 + k*0.002
            A.add("glue_cap", revolve([(0.0104, z), (0.0109, z + 0.0004), (0.0104, z + 0.0008)], 28))
        for n in ((0, -1, 0), (0, 1, 0)):
            A.add("pr_glue", decal((0, n[1]*hy, 0.049), n, (0, 0, 1), 0.030, 0.0465, rc=0.002, lift=0.0002))
    return A


# ============================================================================== loose pieces (piece's own frame)
def wad(centres, radii, seed):
    """A crumpled sheet: the surface the piece's spheres make together, creased inward (never outward) by up to about
    a tenth of its radius, drawn with flat facets."""
    import trimesh
    rng = np.random.default_rng(seed)
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    d, f = np.asarray(m.vertices, float), np.asarray(m.faces, int)
    c, r = np.asarray(centres, float), np.asarray(radii, float)
    o = c[int(np.argmax(r))]                                       # a point inside the largest sphere
    b = d @ (c - o).T
    disc = b*b - ((c - o)**2).sum(1)[None] + (r*r)[None]
    env = np.where(disc >= 0, b + np.sqrt(np.maximum(disc, 0)), -np.inf).max(1)
    q = rng.normal(size=(7, 3))
    q /= np.linalg.norm(q, axis=1)[:, None]
    fr, ph = rng.uniform(3.0, 8.0, 7), rng.uniform(0, TAU, 7)
    ridge = np.abs(np.sin(d @ q.T*fr[None] + ph[None]))            # 0 along a crease, 1 on a facet
    crease = 1 - np.sort(ridge, 1)[:, :2].mean(1)                  # the two deepest creases through each direction
    cut = np.clip(0.02 + 0.13*crease**1.5 + 0.02*rng.random(len(d)), 0, 0.17)
    return o + d*(env*(1 - cut))[:, None], f


def scrap(a, b, seed, half=0.0004):
    """A snip of paper a x b (its collision plate is a x b of 0.8 mm spheres): four corners cut a little off square,
    a slight curl about a random line (the far edges rise 0.8 to 1.6 mm from the plate's underside, at most 1.2 mm
    above its top). Local frame of the piece: flat in z = 0."""
    rng = np.random.default_rng(seed)
    nx, ny = 5, 7
    X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    cx = (a/2 + half)*(1 - rng.uniform(0, 0.10, 4))                # corners --, +-, ++, -+ pulled in by up to a tenth
    cy = (b/2 + half)*(1 - rng.uniform(0, 0.10, 4))
    wts = [(1 - X)*(1 - Y)/4, (1 + X)*(1 - Y)/4, (1 + X)*(1 + Y)/4, (1 - X)*(1 + Y)/4]
    sx, sy = (-1, 1, 1, -1), (-1, -1, 1, 1)
    px = sum(w*s*c for w, s, c in zip(wts, sx, cx))
    py = sum(w*s*c for w, s, c in zip(wts, sy, cy))
    ang = rng.uniform(0, math.pi)
    t = (px*math.cos(ang) + py*math.sin(ang))/max(a, b)*2
    pz = -half + 0.0001 + rng.uniform(0.0008, 0.0016)*np.minimum(t*t, 1.)
    top = np.stack([px, py, pz], -1).reshape(-1, 3)
    v = np.vstack([top, top - (0, 0, 0.00012)])
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            p0, p1, p2, p3 = i*ny + j, (i + 1)*ny + j, (i + 1)*ny + j + 1, i*ny + j + 1
            f += [(p0, p1, p2), (p0, p2, p3)]
    f = np.array(f)
    return v, np.vstack([f, f[:, ::-1] + nx*ny])


def pencil(length, d):
    """Six-sided painted body, a crimped ferrule, an eraser, a sharpened cone of bare wood and the lead. Along local
    x, total length and largest radius those of the collision spheres."""
    A = Asm()
    R = ALONG_X
    r = d/2
    x0, x1 = -length/2 + 0.019, length/2 - 0.021
    A.add("pencil_paint", prism(circle(r, 6), [(0., x0), (0., x1)], cap0=False, cap1=False), R=R, uv=("wrap", x0, x1))
    A.add("pencil_wood", prism(circle(r, 12), [(0.0003, x1), (r - 0.0011, length/2 - 0.0038)], cap0=False, cap1=False), R=R)
    A.add("graphite", prism(circle(r, 12), [(r - 0.0011, length/2 - 0.0038), (r - 0.0004, length/2)], cap0=False), R=R)
    xf = -length/2 + 0.0065
    fer = [(r - 0.0004, xf), (r + 0.0001, xf + 0.0004)]
    for k in range(3):
        s = xf + 0.002 + k*0.0038
        fer += [(r + 0.0001, s), (r - 0.0003, s + 0.0005), (r - 0.0003, s + 0.0012), (r + 0.0001, s + 0.0017)]
    fer += [(r + 0.0001, x0 + 0.0006), (r - 0.0004, x0 + 0.001)]
    A.add("alu", revolve(fer, 16), R=R, crease=25.)
    er = [(0., -length/2), (r - 0.0012, -length/2), (r - 0.0004, -length/2 + 0.0008), (r - 0.0004, xf + 0.001)]
    A.add("eraser_pink", revolve(er, 16), R=R)
    return A


def shard_plate(points, normals, na, nz, thick, seed, edge=0.0013):
    """A piece of the mug's wall through its grid of sphere centres: glazed outside, glazed inside, and a matt
    broken edge that runs `edge` beyond the outer centres (the spheres reach 2.6 mm beyond them) and is chipped
    unevenly. Returns {material: (vertices, triangles)}."""
    rng = np.random.default_rng(seed)
    p0 = np.asarray(points, float).reshape(na, nz, 3)
    p = p0.copy()
    n = np.asarray(normals, float).reshape(na, nz, 3)
    for i in range(na):
        for j in range(nz):
            if i in (0, na - 1) or j in (0, nz - 1):
                ii, jj = min(max(i, 1), na - 2) if na > 2 else (1 - i if na == 2 else i), min(max(j, 1), nz - 2) if nz > 2 else (1 - j if nz == 2 else j)
                out = p0[i, j] - p0[ii, jj]
                out -= n[i, j]*(out @ n[i, j])
                ln = np.linalg.norm(out)
                if ln > 1e-9:
                    p[i, j] += out/ln*(edge + rng.uniform(-0.0011, 0.0006))
    outer, inner = p + n*thick/2, p - n*thick/2
    v = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    idx = lambda i, j, side: side*na*nz + i*nz + j
    fo, fi, fe = [], [], []
    for i in range(na - 1):
        for j in range(nz - 1):
            a, b, c, d = idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0)
            fo += [(a, b, c), (a, c, d)]
            a, b, c, d = idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)
            fi += [(a, c, b), (a, d, c)]
    loop = [(i, 0) for i in range(na)] + [(na - 1, j) for j in range(1, nz)] + [(i, nz - 1) for i in range(na - 2, -1, -1)] + \
           [(0, j) for j in range(nz - 2, 0, -1)]
    for k in range(len(loop)):
        (i0, j0), (i1, j1) = loop[k], loop[(k + 1) % len(loop)]
        a, b, c, d = idx(i0, j0, 0), idx(i1, j1, 0), idx(i1, j1, 1), idx(i0, j0, 1)
        fe += [(a, c, b), (a, d, c)]
    return dict(glaze_out=(v, np.array(fo)), glaze_in=(v, np.array(fi)), biscuit=(v, np.array(fe)))


NOTES = (
    "The lip of both pans is drawn where its collision plate is: its top is level with the floor slab (3.0 mm lobby pan, "
    "4.0 mm hand pan) where it joins it and 0.5 mm above the ground at the mouth's edge. The drawing is made from the "
    "plate's pose, so it follows any correction of the collider.",
    "The lobby pan's two rear wheels are drawn with the pan body, as plain discs at the place of their wheel bodies "
    "(each wheel is its own body on a hinge in the rigid world; its spin is not drawn).",
    "Under the hood of the lobby pan the collision leaves a slot of 22 mm between the top of the side wall and the "
    "hood. It is drawn open, as it is. The hand pan is an open scoop and has no hood; each of its side walls is two "
    "boxes in collision (a stair) and one plate with a sloping top edge in the drawing.",
    "The lobby pan's hub is a cross piece 77 mm long at the foot of the handle; between each yoke arm's end and the "
    "handle it spans 13 mm where there is no collider. Each arm ends in a knuckle of 6.8 mm radius on the hinge line, "
    "beside a boss of 6.2 mm radius on the pan's wall (half of each stands above the wall's top edge).",
    "The barrel and the pail collide as 18 and 16 flat staves; they are drawn round at the mean of each face's "
    "inscribed and circumscribed circle, so the collision wall stands up to 4.0 mm (barrel) and 3.3 mm (pail) outside "
    "the drawn wall along the staves' edges and as far inside it at their middles.",
    "On the barrel the bag over a hand hold stands up to 8.9 mm off a stave's face (3.2 mm of that is the round wall, "
    "4 mm the hold). The pail's bail, its grip and its ears stand up to 11.1 mm off the wall and have no collider.",
    "The brooms' sockets stand 4.5 mm around the lower 50 mm of the handle; the push broom's bracket (plate, bolt heads, "
    "socket) stands up to 6.1 mm off the block and the handle's foot. The push broom has no brace rods: they would "
    "stand far outside its colliders.",
    "A hang hole is drawn through each hanger cap, through the ball end of the counter brush, of the hand pan's handle "
    "and of the lobby pan's grip; the colliders are solid there (up to 4.6 mm).",
    "The scissors on the craft table keep their old picture: closed scissors do not fit their 24 mm wide collider. The "
    "cup and the glue bottle are drawn square, as they collide (they were drawn round).",
)

"""Render-only product appearance: printed labels (drawn here with PIL at their real wrap size), wrapped label meshes
and a few procedural details. Nothing here is read by the physics; label text is generic (no trademarks), but each
label states what the product is so the scene reads as the activity (a gallon of interior eggshell, a 12 oz enamel
spray, the clear coat), in the colour the episode uses.
"""

import math
from pathlib import Path

import numpy as np

from . import params as P

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"


def _font(size, bold=False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT % ("-Bold" if bold else ""), size)
    except OSError:
        return ImageFont.load_default()


def _srgb(c):
    return tuple(int(x) for x in c)


def _text(d, xy, s, size, fill, bold=False, anchor="la"):
    d.text(xy, s, font=_font(size, bold), fill=fill, anchor=anchor)


def can_label(folder, color_name, finish="EG-SHEL"):
    """Gallon can label, 2.0 m of image per m of wrap: 0.527 m around x 0.133 m tall -> 2108 x 532 px. The front panel
    (centred at u = 0.5, facing the can's +x) carries the product name, the colour band and the volume."""
    from PIL import Image, ImageDraw
    W, H = 2108, 532
    col = _srgb(P.COLORS.get(color_name, (238, 239, 234)))
    img = Image.new("RGB", (W, H), (246, 245, 240))
    d = ImageDraw.Draw(img)
    # brand bands top and bottom
    d.rectangle([0, 0, W, 52], fill=(28, 52, 94)); d.rectangle([0, H - 40, W, H], fill=(28, 52, 94))
    _text(d, (24, 10), "WALLCOAT PRO  ·  INTERIOR LATEX  ·  ZERO VOC", 30, (245, 245, 245), bold=True)
    _text(d, (W - 24, 10), "INTERIOR LATEX  ·  ZERO VOC", 30, (245, 245, 245), bold=True, anchor="ra")
    # front panel
    x0 = W//2 - 330
    d.rectangle([x0, 70, x0 + 660, H - 58], fill=(255, 255, 255), outline=(28, 52, 94), width=4)
    _text(d, (x0 + 330, 92), "WALLCOAT PRO", 70, (28, 52, 94), bold=True, anchor="ma")
    _text(d, (x0 + 330, 172), f"INTERIOR LATEX  {finish}", 40, (40, 40, 40), bold=True, anchor="ma")
    d.rectangle([x0 + 40, 232, x0 + 620, 352], fill=col, outline=(80, 80, 80), width=2)
    lum = 0.2126*col[0] + 0.7152*col[1] + 0.0722*col[2]
    _text(d, (x0 + 330, 270), color_name.replace("_", " ").upper(), 44, (20, 20, 20) if lum > 128 else (245, 245, 245),
          bold=True, anchor="ma")
    _text(d, (x0 + 330, 378), "1 US GAL  (3.78 L)", 40, (40, 40, 40), bold=True, anchor="ma")
    _text(d, (x0 + 330, 428), "350-400 sq ft / gal  ·  4 mil wet", 28, (70, 70, 70), anchor="ma")
    # side panels: directions and data
    lines = ["DIRECTIONS: stir well. Cut in with a 2-1/2 in brush,", "roll with a 3/8 in nap. Keep a wet edge.",
             "Recoat in 4 h. Clean up with soap and water.", "", "Dry: touch 1 h  ·  42 % solids by volume",
             "Coverage 350-400 sq ft per gallon"]
    for k, ln in enumerate(lines):
        _text(d, (90, 96 + 46*k), ln, 30, (50, 50, 50))
        _text(d, (W//2 + 400, 96 + 46*k), ln, 30, (50, 50, 50))
    # a little roller pictogram on the front panel
    d.rounded_rectangle([x0 + 520, 382, x0 + 610, 412], radius=10, fill=(28, 52, 94))
    d.line([x0 + 565, 412, x0 + 565, 450], fill=(28, 52, 94), width=6)
    p = Path(folder)/f"label_can_{color_name}.png"
    img.save(p)
    return p


def aerosol_label(folder, color_name, clear=False):
    """12 oz aerosol label: 0.207 m around x 0.094 m tall at 4 px/mm -> 828 x 376 px."""
    from PIL import Image, ImageDraw
    W, H = 828, 376
    col = (236, 236, 232) if clear else _srgb(P.COLORS.get(color_name, (178, 28, 34)))
    img = Image.new("RGB", (W, H), col)
    d = ImageDraw.Draw(img)
    lum = 0.2126*col[0] + 0.7152*col[1] + 0.0722*col[2]
    ink = (20, 20, 20) if lum > 140 else (250, 250, 250)
    d.rectangle([0, 0, W, 46], fill=(18, 18, 18))
    _text(d, (W//2, 8), "2X COVER  ·  PAINT + PRIMER", 26, (250, 210, 40), bold=True, anchor="ma")
    _text(d, (W//2, 70), "CLEAR GLOSS" if clear else "ENAMEL", 54, ink, bold=True, anchor="ma")
    _text(d, (W//2, 136), "PROTECTIVE FINISH" if clear else color_name.replace("_", " ").upper(), 34, ink, bold=True, anchor="ma")
    d.rectangle([W//2 - 210, 192, W//2 + 210, 250], fill=(250, 250, 250), outline=(18, 18, 18), width=2)
    _text(d, (W//2, 205), "SPRAY PAINT  ·  INDOOR / OUTDOOR", 24, (18, 18, 18), bold=True, anchor="ma")
    _text(d, (W//2, 268), "NET WT 12 OZ (340 g)", 26, ink, bold=True, anchor="ma")
    for k in range(10):                                         # hazard stripe at the bottom
        x = k*W//10
        d.polygon([(x, H), (x + 30, H - 46), (x + 60, H - 46), (x + 30, H)], fill=(18, 18, 18))
    _text(d, (16, 300), "DANGER: EXTREMELY FLAMMABLE", 18, ink, bold=True)
    _text(d, (W - 16, 300), "HOLD 10-16 IN FROM SURFACE", 18, ink, bold=True, anchor="ra")
    p = Path(folder)/f"label_aerosol_{'clear' if clear else color_name}.png"
    img.save(p)
    return p


def pail_decal(folder):
    """Cut-pail graduations (QT / L marks) on its white plastic: 0.53 m around x 0.14 m tall at 2 px/mm."""
    from PIL import Image, ImageDraw
    W, H = 1060, 280
    img = Image.new("RGB", (W, H), (242, 242, 240))
    d = ImageDraw.Draw(img)
    for k, (lab, z) in enumerate((("1/4 QT", 0.25), ("1/2 QT", 0.45), ("3/4 QT", 0.65), ("1 QT", 0.85))):
        y = int(H*(1 - z))
        d.line([W//2 - 60, y, W//2 + 60, y], fill=(40, 70, 140), width=3)
        _text(d, (W//2 + 70, y - 12), lab, 22, (40, 70, 140), bold=True)
    _text(d, (W//4, 40), "HANDY PAIL", 40, (40, 70, 140), bold=True, anchor="ma")
    p = Path(folder)/"decal_pail.png"
    img.save(p)
    return p


def wrapped(S, path, r, h, z0, mat, n=64, front=-math.pi/2):
    """Open cylinder of radius r from z0 to z0 + h with UVs (u around, v up): a label surface whose middle (u = 0.5,
    the front panel) faces the local direction at angle `front` (default -y: toward the room's cameras)."""
    v, f, uv, nr = [], [], [], []
    for i in range(n + 1):
        a = front + 2*math.pi*(i/n - 0.5)
        for j, z in enumerate((z0, z0 + h)):
            v.append((r*math.cos(a), r*math.sin(a), z)); uv.append((i/n, float(j))); nr.append((math.cos(a), math.sin(a), 0))
    for i in range(n):
        a_, b_, c_, d_ = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
        f += [a_, c_, d_, a_, d_, b_]
    return S.mesh(path, v, f, uv=uv, normals=nr, mat=mat)


def plate_with_wells(S, root, mat, wells, r, size, z_floor, z_top, n=40):
    """A palette plate whose wells are real recesses (the physics' wells are; a solid render plate hid their paint):
    a base slab up to the well floor, then per well a square tile with a circular hole (top face fanned between the
    circle and the tile's square outline, the well's inner wall), and slabs on the margins around the tile grid."""
    sx, sy = size
    S.box(f"{root}/plate_base", (sx, sy, z_floor), mat, (0, 0, z_floor/2), (0, 0, 0, 1), bevel=0.002)
    xs = sorted({w[0] for w in wells}); ys = sorted({w[1] for w in wells})
    half = min(np.diff(xs).min() if len(xs) > 1 else sx, np.diff(ys).min() if len(ys) > 1 else sy)/2
    h = z_top - z_floor
    for k, (cx, cy) in enumerate(wells):
        v, f, nr = [], [], []
        for i in range(n):
            a = 2*math.pi*i/n
            c, s = math.cos(a), math.sin(a)
            t = half/max(abs(c), abs(s))                       # the square outline at the same angle
            v += [(cx + r*c, cy + r*s, z_top), (cx + t*c, cy + t*s, z_top)]
            nr += [(0, 0, 1), (0, 0, 1)]
        for i in range(n):
            a0, b0, a1, b1 = 2*i, 2*i + 1, (2*i + 2) % (2*n), (2*i + 3) % (2*n)
            f += [a0, b0, b1, a0, b1, a1]
        # the square's corners (the fan between angle samples leaves them out): one triangle each
        for (qx, qy) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            i = int(round((math.atan2(qy, qx) % (2*math.pi))/(2*math.pi)*n)) % n
            j0 = (i - 1) % n
            v += [(cx + qx*half, cy + qy*half, z_top)]; nr += [(0, 0, 1)]
            f += [2*j0 + 1, len(v) - 1, 2*((j0 + 2) % n) + 1]
        S.mesh(f"{root}/well_tile_{k}", v, f, normals=nr, mat=mat)
        S.mesh(f"{root}/well_wall_{k}", *_cyl_wall(cx, cy, r, z_floor, z_top, n), mat=mat)
    x0, x1 = min(xs) - half, max(xs) + half
    y0, y1 = min(ys) - half, max(ys) + half
    for nm, (ax, bx, ay, by) in dict(m_left=(-sx/2, x0, -sy/2, sy/2), m_right=(x1, sx/2, -sy/2, sy/2),
                                      m_front=(x0, x1, -sy/2, y0), m_back=(x0, x1, y1, sy/2)).items():
        if bx - ax > 1e-4 and by - ay > 1e-4:
            S.box(f"{root}/{nm}", (bx - ax, by - ay, h), mat, ((ax + bx)/2, (ay + by)/2, z_floor + h/2), (0, 0, 0, 1))


def _cyl_wall(cx, cy, r, z0, z1, n):
    v, f, nr = [], [], []
    for i in range(n + 1):
        a = 2*math.pi*i/n
        v += [(cx + r*math.cos(a), cy + r*math.sin(a), z0), (cx + r*math.cos(a), cy + r*math.sin(a), z1)]
        nr += [(-math.cos(a), -math.sin(a), 0)]*2                       # facing into the well
    for i in range(n):
        a_, b_, c_, d_ = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
        f += [a_, d_, c_, a_, b_, d_]
    return v, f, nr

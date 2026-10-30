"""Packaging artwork (fictional brands) at physical scale: 4 texels per millimetre.

Printed barcodes are NOT part of these textures: each symbol is its own high-resolution patch mesh laid on
the package at the pose recorded in Scene.labels (10 texels per module), so the decoder sees print at the
right module size. The packaging leaves a white print area where GS1 places the symbol.
"""

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .artwork import FONT_BOLD, FONT_SANS, FONT_MONO, _font
from .catalog import SKUS

DPMM = 4.0
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
CONDENSED = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"


def _px(mm):
    return max(1, int(round(mm*DPMM)))


def _canvas(w_m, h_m, color):
    return Image.new("RGB", (_px(w_m*1e3), _px(h_m*1e3)), tuple(color))


def _shade(img, top=1.06, bottom=.92):
    """Soft vertical tone gradient (printed flood colours are not flat under studio proofs)."""
    a = np.asarray(img, np.float32)
    g = np.linspace(top, bottom, a.shape[0])[:, None, None]
    return Image.fromarray(np.clip(a*g, 0, 255).astype(np.uint8))


def _text(d, xy, s, mm, font=FONT_BOLD, fill=(255, 255, 255), anchor="la"):
    d.text((xy[0]*DPMM, xy[1]*DPMM), s, font=_font(font, mm*DPMM), fill=fill, anchor=anchor)


def _fit_text(d, xy, s, max_w_mm, mm, font=FONT_BOLD, fill=(255, 255, 255), anchor="ma"):
    size = mm
    while size > 3:
        f = _font(font, size*DPMM)
        if d.textlength(s, font=f) <= max_w_mm*DPMM:
            break
        size *= .92
    d.text((xy[0]*DPMM, xy[1]*DPMM), s, font=_font(font, size*DPMM), fill=fill, anchor=anchor)


def nutrition_panel(d, x, y, w, h, ink=(20, 20, 20)):
    """FDA-style Nutrition Facts block (fictional values), mm coordinates."""
    d.rectangle([x*DPMM, y*DPMM, (x + w)*DPMM, (y + h)*DPMM], outline=ink, width=int(DPMM*.4), fill=(255, 255, 255))
    _text(d, (x + 1.5, y + 1), "Nutrition Facts", min(5.5, w/9), FONT_BOLD, ink)
    rows = ["Serving size 1 cup (245g)", "Calories 90", "Total Fat 0g  0%", "Sodium 480mg  21%",
            "Total Carb 20g  7%", "  Dietary Fiber 2g  7%", "  Total Sugars 12g", "Protein 2g",
            "Vitamin D 0mcg 0%  Calcium 20mg 2%", "Iron 1mg 6%  Potassium 450mg 10%"]
    yy = y + 1 + min(5.5, w/9) + 2
    for k, r in enumerate(rows):
        if yy > y + h - 3:
            break
        d.line([(x + 1)*DPMM, (yy - .6)*DPMM, (x + w - 1)*DPMM, (yy - .6)*DPMM], fill=ink, width=int(DPMM*(.6 if k in (0, 1) else .25)))
        _text(d, (x + 1.5, yy), r, min(2.4, w/22), FONT_BOLD if k in (1,) else FONT_SANS, ink)
        yy += min(3.4, w/15)


def paragraph(d, x, y, w, text, mm=2.0, fill=(40, 40, 40), font=FONT_SANS, max_lines=12):
    f = _font(font, mm*DPMM)
    words, line, lines = text.split(), "", []
    for wd in words:
        t = (line + " " + wd).strip()
        if d.textlength(t, font=f) > w*DPMM:
            lines.append(line); line = wd
        else:
            line = t
    lines.append(line)
    for k, ln in enumerate(lines[:max_lines]):
        d.text((x*DPMM, (y + k*mm*1.35)*DPMM), ln, font=f, fill=fill)
    return y + min(len(lines), max_lines)*mm*1.35


INGREDIENTS = ("INGREDIENTS: TOMATO PUREE (WATER, TOMATO PASTE), HIGH FRUCTOSE CORN SYRUP, WHEAT FLOUR, WATER, "
               "CONTAINS LESS THAN 2% OF: SALT, POTASSIUM CHLORIDE, CITRIC ACID, NATURAL FLAVORING, ASCORBIC ACID.")
DIST = "DISTRIBUTED BY {brand} FOODS CO., COLUMBUS, OH 43215  - FICTIONAL PRODUCT FOR SIMULATION -"


def box_faces(sku, folder):
    """{face: (path, width_m, height_m)} for a box SKU; faces seen from outside, 'up' = +z (front/back/sides)."""
    s = sku
    W, D, H = s.dims
    prim, sec, txt = s.palette
    out = {}
    folder = Path(folder)
    cat = s.category

    def save(name, img):
        p = folder/f"{s.id}_{name}.png"
        img.save(p)
        return p

    if cat == "library":
        return book_faces(sku, folder)
    # front
    im = _shade(_canvas(W, H, prim)); d = ImageDraw.Draw(im)
    w_mm, h_mm = W*1e3, H*1e3
    d.rectangle([0, 0, im.width, _px(h_mm*.16)], fill=tuple(int(c*.75) for c in prim))
    _fit_text(d, (w_mm/2, h_mm*.035), s.brand.upper(), w_mm*.9, h_mm*.07, CONDENSED, txt)
    _fit_text(d, (w_mm/2, h_mm*.22), s.name, w_mm*.9, h_mm*.085, SERIF_BOLD, txt)
    if cat == "grocery":
        cx, cy, r = w_mm*.5, h_mm*.58, min(w_mm, h_mm)*.28
        d.ellipse([(cx - r)*DPMM, (cy - r*.6)*DPMM, (cx + r)*DPMM, (cy + r*.6)*DPMM], fill=tuple(sec))
        d.ellipse([(cx - r*.8)*DPMM, (cy - r*.45)*DPMM, (cx + r*.8)*DPMM, (cy + r*.4)*DPMM], fill=(236, 206, 150))
        for k in range(28):
            a = k*2.39
            rr = r*.62*math.sqrt((k + .5)/28)
            px, py = cx + rr*math.cos(a), cy + rr*.55*math.sin(a)
            d.ellipse([(px - 2.2)*DPMM, (py - 1.6)*DPMM, (px + 2.2)*DPMM, (py + 1.6)*DPMM], fill=(196, 150, 80))
    else:
        d.rounded_rectangle([w_mm*.12*DPMM, h_mm*.42*DPMM, w_mm*.88*DPMM, h_mm*.72*DPMM], radius=_px(4), fill=tuple(sec))
        _fit_text(d, (w_mm/2, h_mm*.52), s.size_text, w_mm*.7, h_mm*.12, FONT_BOLD, tuple(prim))
    _fit_text(d, (w_mm*.05, h_mm*.92), s.size_text, w_mm*.6, h_mm*.045, FONT_BOLD, txt, anchor="la")
    out["front"] = (save("front", im), W, H)
    # back (GS1: symbol in the lower-right quadrant; the patch mesh covers the white print area)
    im = _canvas(W, H, (250, 248, 242)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, _px(h_mm*.09)], fill=tuple(prim))
    _fit_text(d, (w_mm/2, h_mm*.02), s.brand.upper(), w_mm*.9, h_mm*.05, CONDENSED, txt)
    if cat == "grocery":
        nutrition_panel(d, w_mm*.05, h_mm*.12, w_mm*.5, h_mm*.45)
        paragraph(d, w_mm*.05, h_mm*.62, w_mm*.45, INGREDIENTS, mm=min(2., w_mm/60))
    else:
        paragraph(d, w_mm*.05, h_mm*.14, w_mm*.9, f"{s.name}. Latex free. Powder free. Ambidextrous. "
                  "Store in a cool, dry place away from direct sunlight. Not made with natural rubber latex.",
                  mm=min(2.4, w_mm/45))
    paragraph(d, w_mm*.05, h_mm*.94, w_mm*.9, DIST.format(brand=s.brand.upper()), mm=min(1.6, w_mm/80), max_lines=2)
    _barcode_area(d, s, "back", W, H)
    out["back"] = (save("back", im), W, H)
    # sides: right face is (D x H) seen from +x
    for side in ("left", "right"):
        im = _shade(_canvas(D, H, prim)); d = ImageDraw.Draw(im)
        dm = D*1e3
        _fit_text(d, (dm/2, h_mm*.08), s.brand.upper(), dm*.9, min(dm*.18, h_mm*.08), CONDENSED, txt)
        _fit_text(d, (dm/2, h_mm*.30), s.name, dm*.9, min(dm*.14, h_mm*.06), FONT_BOLD, txt)
        _barcode_area(d, s, side, D, H)
        out[side] = (save(side, im), D, H)
    for tb in ("top", "bottom"):
        im = _canvas(W, D, prim if tb == "top" else tuple(int(c*.8) for c in prim)); d = ImageDraw.Draw(im)
        if tb == "top":
            _fit_text(d, (w_mm/2, D*1e3*.35), s.brand.upper(), w_mm*.8, D*1e3*.3, CONDENSED, txt)
        _barcode_area(d, s, tb, W, D)
        out[tb] = (save(tb, im), W, D)
    return out


def _barcode_area(d, s, face, w, h):
    bc = s.barcode
    if bc.face != face:
        return
    from .catalog import symbol_for
    from .artwork import render_symbol
    _, sw, sh = render_symbol(symbol_for(s), 2)
    if bc.orientation == "ladder":
        sw, sh = sh, sw
    cx, cy = w*1e3/2 + bc.u*1e3, h*1e3/2 - bc.v*1e3
    m = 2.0
    d.rectangle([(cx - sw/2 - m)*DPMM, (cy - sh/2 - m)*DPMM, (cx + sw/2 + m)*DPMM, (cy + sh/2 + m)*DPMM], fill=(255, 255, 255))


def wrap_texture(sku, folder):
    """Cylinder label wrap: u = azimuth from the front (0) through 360 deg, v = height. Returns (path, circ_m, h_m).
    A necked container's wrap covers its label panel only (catalog.neck); the white barcode area is placed at the
    symbol's item-centred height within that panel."""
    from .catalog import neck
    s = sku
    r, H = s.dims
    g = neck(s)
    z_lo, z_hi = g["panel"] if g else (-H/2, H/2)
    H = z_hi - z_lo
    C = 2*math.pi*r
    prim, sec, txt = s.palette
    cm, hm = C*1e3, H*1e3
    im = _shade(_canvas(C, H, sec if s.id in ("K05",) else prim), 1.04, .95); d = ImageDraw.Draw(im)
    # front panel centred on u = 0 (wraps): draw at the centre of a rolled copy
    front_w = cm*.42

    def at(a_deg):
        return (a_deg/360.)*cm

    if s.id == "K05":        # water sleeve: blue band with drops, clear-ish sleeve printed white
        d.rectangle([0, _px(hm*.30), im.width, _px(hm*.70)], fill=tuple(prim))
        for k in range(40):
            x = (k*37.3) % cm; y = hm*.34 + (k*11.1) % (hm*.3)
            d.ellipse([(x - 1.2)*DPMM, (y - 2)*DPMM, (x + 1.2)*DPMM, (y + 2)*DPMM], fill=(210, 235, 255))
    else:
        d.rectangle([0, _px(hm*.68), im.width, _px(hm*.80)], fill=tuple(sec))
    for base in (0., cm):     # front panel straddles the seam at u = 0
        x0 = base - front_w/2
        _fit_text(d, (x0 + front_w/2, hm*.12), s.brand.upper(), front_w*.9, hm*.11, CONDENSED, txt if s.id != "K05" else (255, 255, 255))
        _fit_text(d, (x0 + front_w/2, hm*.36), s.name.upper(), front_w*.9, hm*.12, SERIF_BOLD, txt if s.id != "K05" else (255, 255, 255))
        _fit_text(d, (x0 + front_w/2, hm*.86), s.size_text, front_w*.9, hm*.07, FONT_BOLD, txt if s.id != "K05" else (20, 60, 120))
    # back panel around 180 deg: nutrition + white barcode area
    bx = at(180.)
    if hm > 60:
        nutrition_panel(d, bx - cm*.20 - cm*.02, hm*.08, cm*.16, hm*.55)
    bc = s.barcode
    from .catalog import symbol_for
    from .artwork import render_symbol
    _, sw, sh = render_symbol(symbol_for(s), 2)
    if bc.orientation == "ladder":
        sw, sh = sh, sw
    ux = at(bc.angle_deg)
    cy = (z_hi - bc.v)*1e3                  # from the top edge of the wrap
    d.rectangle([(ux - sw/2 - 2)*DPMM, (cy - sh/2 - 2)*DPMM, (ux + sw/2 + 2)*DPMM, (cy + sh/2 + 2)*DPMM], fill=(255, 255, 255))
    paragraph(d, bx + cm*.06, hm*.10, cm*.14, INGREDIENTS, mm=min(1.6, hm/60), max_lines=8)
    p = Path(folder)/f"{s.id}_wrap.png"
    im.save(p)
    return p, C, H


def book_faces(sku, folder):
    """Hardcover/paperback: front cover (title art), back cover (blurb, price), spine, page edges."""
    s = sku
    W, D, H = s.dims
    prim, sec, txt = s.palette
    folder = Path(folder)
    out = {}
    w_mm, h_mm, d_mm = W*1e3, H*1e3, D*1e3
    title, author = s.name, {"K12": "Maren Holloway", "K13": "Ilse Varga"}.get(s.id, "Anon.")
    im = _shade(_canvas(W, H, prim), 1.08, .9); d = ImageDraw.Draw(im)
    for k in range(9):
        y = h_mm*(.45 + .045*k)
        d.line([0, y*DPMM, im.width, (y + 6*math.sin(k))*DPMM], fill=tuple(sec), width=_px(1.2))
    words = title.split()
    mid = (len(words) + 1)//2
    _fit_text(d, (w_mm/2, h_mm*.10), " ".join(words[:mid]).upper(), w_mm*.86, h_mm*.075, SERIF_BOLD, txt)
    _fit_text(d, (w_mm/2, h_mm*.19), " ".join(words[mid:]).upper(), w_mm*.86, h_mm*.075, SERIF_BOLD, txt)
    _fit_text(d, (w_mm/2, h_mm*.86), author.upper(), w_mm*.8, h_mm*.045, SERIF, txt)
    p = folder/f"{s.id}_front.png"; im.save(p); out["front"] = (p, W, H)
    im = _canvas(W, H, tuple(int(c*.92) for c in prim)); d = ImageDraw.Draw(im)
    paragraph(d, w_mm*.1, h_mm*.08, w_mm*.8, "“A luminous chronicle of tides, trade and the people who read the "
              "water.”  From the harbours of the north coast to the salt flats of the south, a year of weather "
              "and small economies, told in twelve voices.", mm=min(3.2, w_mm/32), fill=txt, font=SERIF)
    _text(d, (w_mm*.1, h_mm*.62), s.brand.upper(), min(3., w_mm/30), FONT_BOLD, txt)
    _barcode_area(d, s, "back", W, H)
    p = folder/f"{s.id}_back.png"; im.save(p); out["back"] = (p, W, H)
    im = _canvas(D, H, tuple(int(c*.85) for c in prim)); d = ImageDraw.Draw(im)
    sp = Image.new("RGB", (_px(h_mm), _px(d_mm)), tuple(int(c*.85) for c in prim)); sd = ImageDraw.Draw(sp)
    _fit_text(sd, (h_mm*.45, d_mm*.22), title.upper(), h_mm*.7, d_mm*.5, SERIF_BOLD, txt)
    _fit_text(sd, (h_mm*.9, d_mm*.3), s.brand.split()[0].upper(), h_mm*.15, d_mm*.35, FONT_BOLD, txt)
    im = sp.rotate(90, expand=True)
    p = folder/f"{s.id}_spine.png"; im.save(p); out["left"] = (p, D, H)
    pages = Image.new("RGB", (64, 64), (238, 232, 214))
    pd = ImageDraw.Draw(pages)
    for k in range(0, 64, 2):
        pd.line([0, k, 64, k], fill=(224, 216, 196))
    p = folder/f"{s.id}_pages.png"; pages.save(p)
    for f, dims in (("right", (D, H)), ("top", (W, D)), ("bottom", (W, D))):
        out[f] = (p, *dims)
    return out


def library_label(code, folder, dpmm=24):
    """Library item label 50 x 16 mm, laminated: library name + Codabar (14 digits, X = 0.25 mm)."""
    from .artwork import Symbol, compose, paste_symbol, text as t
    sym = Symbol("Codabar", code, .25, bar_height_mm=8.5, hri=True)
    from .artwork import render_symbol
    _, w_mm, h_mm = render_symbol(sym, 2)
    W = w_mm + 4
    c = compose(W, h_mm + 5.5, dpmm)
    t(c, dpmm, W/2, .6, "WESTBROOK PUBLIC LIBRARY", 2.6, FONT_BOLD, anchor="ma")
    paste_symbol(c, dpmm, sym, 2, 4.2)
    p = Path(folder)/f"lib_{code}.png"
    c.save(p)
    return p, W/1e3, (h_mm + 5.5)/1e3, sym


def particle_board(folder, size_m=1.0, dpmm=1.0, seed=7):
    """Particle-board deck surface (authored): fine flakes over a warm tan base, 1 texel per mm."""
    rng = np.random.default_rng(seed)
    n = int(size_m*1e3*dpmm)
    base = np.array([176, 146, 104], np.float32)
    img = np.ones((n, n, 3), np.float32)*base
    noise = rng.normal(0, 1, (n, n)).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    img *= (1 + .08*gaussian_filter(noise, 1.2))[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(int(n*n/120)):
        x, y = rng.integers(0, n, 2)
        L = rng.uniform(2, 9); a = rng.uniform(0, math.pi)
        tone = int(rng.uniform(120, 210))
        d.line([x, y, x + L*math.cos(a), y + L*math.sin(a)], fill=(tone, int(tone*.82), int(tone*.58)), width=int(rng.integers(1, 3)))
    im = im.filter(ImageFilter.GaussianBlur(.6))
    p = Path(folder)/"particle_board.png"
    im.save(p)
    return p

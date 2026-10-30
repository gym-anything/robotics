"""Repository-authored print artwork (no downloaded product art): the boutique tote print, the inside of the
laminated stock, a gift-wrap pattern at its physical scale, kraft and chipboard surfaces and the shop sign.
Each function writes a PNG into `folder` (once) and returns its path. Scale conventions are documented per
function so the renderer can map rest coordinates (metres) to texture coordinates."""

import math
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BURGUNDY = (124, 22, 38)           # the tote's printed colour (Pm.TOTE.color in sRGB)
GOLD = (196, 160, 82)


def _font(size, bold=False):
    for name in (("DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _paper_grain(rng, n, amp):
    g = rng.standard_normal((n, n)).astype(np.float32)
    img = Image.fromarray(((g*.5 + .5).clip(0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))
    return (np.asarray(img, np.float32)/255. - .5)*amp


def tote_print(folder, W, G, H, turn_top):
    """Outside print of the tote net laid out as one atlas, 4 px/mm. Atlas u runs along the girth
    (front 0..W, right gusset W..W+G, back W+G..2W+G, left gusset 2W+G..2W+2G), v from the bottom fold (0) up
    to the mouth (H); the bottom panel occupies v in [-G, 0] under the front. Returns (path, (u_span, v_span,
    v_offset)) for the UV mapping."""
    path = Path(folder)/"tote_print.png"
    ppm = 4000.
    u_span, v_span, v_off = 2*W + 2*G, H + G, G
    w, h = int(u_span*ppm), int(v_span*ppm)
    if not path.exists():
        rng = np.random.default_rng(7)
        base = np.ones((h, w, 3), np.float32)*np.asarray(BURGUNDY, np.float32)/255.
        grain = _paper_grain(rng, max(w, h), .035)[:h, :w]
        img = Image.fromarray((np.clip(base*(1 + grain[..., None]), 0, 1)*255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        top = lambda v: int((v_span - (v + v_off))*ppm)        # image row of height v above the bottom fold
        for x0 in (0., W + G):                                  # front and back faces: monogram + wordmark
            cx = int((x0 + W/2)*ppm)
            r = int(.030*ppm)
            cy = top(H*.58)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GOLD, width=int(.0012*ppm))
            f = _font(int(.034*ppm), bold=True)
            d.text((cx, cy), "M", fill=GOLD, font=f, anchor="mm")
            f2 = _font(int(.013*ppm))
            d.text((cx, top(H*.40)), "MAISON  LUMEN", fill=GOLD, font=f2, anchor="mm")
            d.text((cx, top(H*.40) + int(.018*ppm)), "PARIS  ·  NEW YORK", fill=GOLD, font=_font(int(.006*ppm)), anchor="mm")
        # the turn-top line (fold of the turn-top band) as a faint embossed line
        y = top(H - turn_top)
        d.line([(0, y), (w, y)], fill=tuple(int(c*.85) for c in BURGUNDY), width=2)
        img.save(path)
    return path, (u_span, v_span, v_off)


def tote_inside(folder):
    """Inside of the laminated stock: the uncoated reverse of the art paper (off-white, faint fibre)."""
    path = Path(folder)/"tote_inside.png"
    if not path.exists():
        rng = np.random.default_rng(8)
        n = 512
        base = np.ones((n, n, 3), np.float32)*np.array([.93, .91, .87], np.float32)
        grain = _paper_grain(rng, n, .05)
        Image.fromarray((np.clip(base*(1 + grain[..., None]), 0, 1)*255).astype(np.uint8)).save(path)
    return path


def wrap_pattern(folder):
    """Gift-wrap print, one repeat = 0.20 m (texture coordinate = rest coordinate / 0.20 m): deep green ground,
    white pinstripes on the bias, gold stars."""
    path = Path(folder)/"wrap_pattern.png"
    if not path.exists():
        n = 2048
        rng = np.random.default_rng(9)
        img = Image.new("RGB", (n, n), (18, 78, 52))
        d = ImageDraw.Draw(img)
        step = n//10
        for k in range(-10, 21):
            x = k*step
            d.line([(x, 0), (x + n, n)], fill=(236, 232, 220), width=6)
        for _ in range(60):
            cx, cy = rng.integers(0, n, 2)
            r = int(rng.integers(18, 34))
            pts = []
            for j in range(10):
                a = math.pi/2 + j*math.pi/5
                rr = r if j % 2 == 0 else r*.45
                pts.append((cx + rr*math.cos(a), cy - rr*math.sin(a)))
            d.polygon(pts, fill=GOLD)
        g = _paper_grain(rng, n, .03)
        arr = np.asarray(img, np.float32)/255.*(1 + g[..., None])
        Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8)).save(path)
    return path


def plain(folder, name, rgb, amp=.04, seed=10):
    """A plain paper/board surface with a faint grain (any scale)."""
    path = Path(folder)/f"{name}.png"
    if not path.exists():
        rng = np.random.default_rng(seed)
        n = 512
        base = np.ones((n, n, 3), np.float32)*np.asarray(rgb, np.float32)
        g = _paper_grain(rng, n, amp)
        Image.fromarray((np.clip(base*(1 + g[..., None]), 0, 1)*255).astype(np.uint8)).save(path)
    return path


def candle_carton(folder):
    """Folding carton print for the boxed candle (one face, 90 x 110 mm, 8 px/mm)."""
    path = Path(folder)/"candle_carton.png"
    if not path.exists():
        w, h = 720, 880
        img = Image.new("RGB", (w, h), (238, 232, 220))
        d = ImageDraw.Draw(img)
        d.rectangle([30, 30, w - 30, h - 30], outline=(120, 100, 70), width=4)
        d.text((w//2, 300), "AMBER", fill=(90, 60, 30), font=_font(86, bold=True), anchor="mm")
        d.text((w//2, 400), "& FIG", fill=(90, 60, 30), font=_font(60), anchor="mm")
        d.text((w//2, 640), "scented candle  ·  380 g", fill=(120, 100, 70), font=_font(30), anchor="mm")
        img.save(path)
    return path


def shop_sign(folder):
    path = Path(folder)/"shop_sign.png"
    if not path.exists():
        img = Image.new("RGB", (2000, 500), (32, 30, 34))
        d = ImageDraw.Draw(img)
        d.text((1000, 250), "MAISON  LUMEN", fill=GOLD, font=_font(170, bold=True), anchor="mm")
        img.save(path)
    return path


# ----------------------------------------------------------------------------- cash-wrap and shelf stock
SANS = "DejaVuSans.ttf"
SANS_BOLD = "DejaVuSans-Bold.ttf"
MONO = "DejaVuSansMono.ttf"


def _ttf(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def ribbon_satin(folder, name, rgb):
    """Outer face of a wound satin ribbon roll, mapped with cylindrical UVs (u around the roll, v across the
    ribbon width): woven lengthwise sheen lines, slightly darker selvedges, and faint bands around the roll where
    the winding tension and the overlapping layers vary."""
    path = Path(folder)/f"ribbon_{name}.png"
    if not path.exists():
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        nu, nv = 1024, 256
        v = np.linspace(0, 1, nv, dtype=np.float32)[:, None]
        u = np.linspace(0, 1, nu, dtype=np.float32)[None, :]
        weave = .035*np.sin(2*np.pi*v*60 + .7*np.sin(2*np.pi*v*7))                 # lengthwise weave lines
        selvedge = -.10*np.exp(-((v - .035)/.02)**2) - .10*np.exp(-((v - .965)/.02)**2)
        bands = sum(a*np.sin(2*np.pi*(k*u + rng.random())) for k, a in ((3, .025), (7, .02), (19, .012)))
        tail_step = np.where((u > .62) & (u < .625), -.12, 0.)                   # the outermost layer's cut end
        shade = 1. + weave + selvedge + bands + tail_step + .015*rng.standard_normal((nv, nu)).astype(np.float32)
        img = np.clip(np.asarray(rgb, np.float32)[None, None, :]*shade[..., None], 0, 1)
        Image.fromarray((img*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6)).save(path)
    return path


def flat_tote(folder, name, ground, ink, W, H, turn_top, rope=(236, 230, 214)):
    """Front face of a tote as it ships flat (bottom folded under, gussets in): W x H, 4 px/mm, mouth at the top
    of the image. The shop print (monogram, wordmark) and the two rope handles folded down onto the face."""
    path = Path(folder)/f"flat_tote_{name}.png"
    if not path.exists():
        ppm = 4000.
        w, h = int(W*ppm), int(H*ppm)
        rng = np.random.default_rng(11)
        base = np.ones((h, w, 3), np.float32)*np.asarray(ground, np.float32)/255.
        grain = _paper_grain(rng, max(w, h), .03)[:h, :w]
        img = Image.fromarray((np.clip(base*(1 + grain[..., None]), 0, 1)*255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        y_turn = int(turn_top*ppm)
        d.line([(0, y_turn), (w, y_turn)], fill=tuple(int(c*.82) for c in ground), width=3)
        cx, cy, r = w//2, int(h*.42), int(min(W, H)*.15*ppm)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=max(3, int(.0012*ppm)))
        d.text((cx, cy), "M", fill=ink, font=_font(int(r*1.15), bold=True), anchor="mm")
        d.text((cx, int(h*.60)), "MAISON  LUMEN", fill=ink, font=_font(int(.013*ppm*W/.203)), anchor="mm")
        # rope handles folded down: each a U of twisted rope from two eyelets under the turn-top
        for sx in (-1, 1):
            ex = cx + sx*int(.045*ppm*W/.203)
            left, right = ex - int(.020*ppm), ex + int(.020*ppm)
            arc = [(left + (right - left)*(1 - math.cos(t))/2, y_turn*.5 + .085*ppm*math.sin(t)) for t in np.linspace(0, math.pi, 80)]
            for k in range(len(arc) - 1):
                d.line([arc[k], arc[k + 1]], fill=tuple(int(c*.72) for c in rope), width=int(.0062*ppm))
            for k in range(len(arc) - 1):
                d.line([arc[k], arc[k + 1]], fill=rope, width=int(.0045*ppm))
            for k in range(0, len(arc) - 1, 3):                                   # twist lay
                x0, y0 = arc[k]
                d.line([(x0 - .002*ppm, y0 - .0015*ppm), (x0 + .002*ppm, y0 + .0015*ppm)], fill=tuple(int(c*.8) for c in rope), width=2)
            for x in (left, right):                                                # eyelet knots
                d.ellipse([x - .004*ppm, y_turn*.5 - .004*ppm, x + .004*ppm, y_turn*.5 + .004*ppm], fill=tuple(int(c*.9) for c in rope))
        img.save(path)
    return path


def tissue(folder, name, rgb, fold_lines=(.5,)):
    """Folded tissue paper (17 g/m2): soft crinkle, faint fibre and the fold creases (fractions of the width)."""
    path = Path(folder)/f"tissue_{name}.png"
    if not path.exists():
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        n = 512
        g = rng.standard_normal((n, n)).astype(np.float32)
        crinkle = np.asarray(Image.fromarray(((g*.5 + .5).clip(0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)),
                             np.float32)/255. - .5
        fibre = _paper_grain(rng, n, .05)
        shade = 1. + .10*crinkle + fibre
        for f in fold_lines:
            x = int(f*n)
            shade[:, max(0, x - 2):x + 2] *= .90
        img = np.clip(np.asarray(rgb, np.float32)[None, None, :]*shade[..., None], 0, 1)
        Image.fromarray((img*255).astype(np.uint8)).save(path)
    return path


def tissue_pack(folder, name, rgb, sheets=24):
    """Top of a retail tissue pack: folded tissue in a clear sleeve with a printed header card at one end."""
    path = Path(folder)/f"tissue_pack_{name}.png"
    if not path.exists():
        src = Image.open(tissue(folder, name, rgb, fold_lines=(.33, .66))).resize((760, 1000))
        img = src.copy()
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 759, 210], fill=(246, 243, 236))
        d.line([(0, 210), (759, 210)], fill=(200, 196, 188), width=3)
        d.text((380, 70), "MAISON LUMEN", fill=BURGUNDY, font=_font(64, bold=True), anchor="mm")
        d.text((380, 140), f"TISSUE PAPER  ·  50 x 75 cm  ·  {sheets} SHEETS", fill=(90, 80, 70), font=_ttf(SANS, 30), anchor="mm")
        sheen = Image.new("L", img.size, 0)
        ImageDraw.Draw(sheen).polygon([(120, 1000), (320, 1000), (760, 380), (760, 260)], fill=46)
        img = Image.composite(Image.new("RGB", img.size, (255, 255, 255)), img, sheen.filter(ImageFilter.GaussianBlur(30)))
        img.save(path)
    return path


def receipt(folder):
    """Thermal receipt, 80 mm paper (72 mm print width), 8 px/mm, 150 mm long; fictional shop and items."""
    path = Path(folder)/"receipt.png"
    if not path.exists():
        w, h = 640, 1200
        img = Image.new("RGB", (w, h), (247, 246, 242))
        d = ImageDraw.Draw(img)
        ink = (52, 52, 58)
        f, fb, fm = _ttf(MONO, 26), _font(44, bold=True), _ttf(MONO, 22)
        d.text((w//2, 70), "MAISON LUMEN", fill=ink, font=fb, anchor="mm")
        d.text((w//2, 120), "gifts  ·  stationery  ·  candles", fill=ink, font=fm, anchor="mm")
        d.text((w//2, 160), "05/10/2026  14:32   TILL 1", fill=ink, font=fm, anchor="mm")
        d.line([(40, 195), (w - 40, 195)], fill=ink, width=2)
        y = 240
        for item, price in (("AMBER & FIG CANDLE", "38.00"), ("GIFT TOTE  M", "4.50"), ("TISSUE x2", "0.00"),
                            ("RIBBON 1.5in  0.6 m", "1.20"), ("GIFT WRAP SERVICE", "0.00"), ("CARD", "3.95")):
            d.text((40, y), item, fill=ink, font=f); d.text((w - 40, y), price, fill=ink, font=f, anchor="ra"); y += 48
        d.line([(40, y + 6), (w - 40, y + 6)], fill=ink, width=2)
        d.text((40, y + 30), "TOTAL", fill=ink, font=_ttf(MONO, 34)); d.text((w - 40, y + 30), "47.65", fill=ink, font=_ttf(MONO, 34), anchor="ra")
        d.text((40, y + 90), "CARD  CONTACTLESS", fill=ink, font=fm)
        d.text((w//2, y + 200), "THANK YOU", fill=ink, font=_ttf(MONO, 30), anchor="mm")
        for k in range(46):                                                     # transaction barcode
            x = 120 + k*9
            d.rectangle([x, y + 260, x + (3 if (k*7) % 5 else 6), y + 330], fill=ink)
        img.save(path)
    return path


def terminal_keypad(folder):
    """Sloped keypad face of a countertop card terminal (82 x 100 mm, 8 px/mm): 4 x 3 numeric keys over a row of
    cancel / clear / enter keys, on a charcoal moulding."""
    path = Path(folder)/"terminal_keypad.png"
    if not path.exists():
        w, h = 656, 800
        img = Image.new("RGB", (w, h), (34, 35, 38))
        d = ImageDraw.Draw(img)
        labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
        kw, kh, gx, gy = 168, 112, 36, 36
        x0 = (w - 3*kw - 2*gx)//2
        for i, lab in enumerate(labels):
            r, c = divmod(i, 3)
            x, y = x0 + c*(kw + gx), 40 + r*(kh + gy)
            d.rounded_rectangle([x + 4, y + 6, x + kw + 4, y + kh + 6], radius=26, fill=(14, 14, 16))
            d.rounded_rectangle([x, y, x + kw, y + kh], radius=26, fill=(150, 152, 156))
            d.rounded_rectangle([x + 8, y + 6, x + kw - 8, y + kh*.45], radius=20, fill=(172, 174, 178))
            d.text((x + kw//2, y + kh//2), lab, fill=(28, 28, 30), font=_ttf(SANS_BOLD, 58), anchor="mm")
        y = 40 + 4*(kh + gy)
        for c, (col, sym) in enumerate((((196, 40, 40), "X"), ((222, 186, 40), "<"), ((44, 160, 70), "O"))):
            x = x0 + c*(kw + gx)
            d.rounded_rectangle([x + 4, y + 6, x + kw + 4, y + kh + 6], radius=26, fill=(14, 14, 16))
            d.rounded_rectangle([x, y, x + kw, y + kh], radius=26, fill=col)
            d.text((x + kw//2, y + kh//2), sym, fill=(250, 250, 250), font=_ttf(SANS_BOLD, 56), anchor="mm")
        img.save(path)
    return path


def terminal_screen(folder):
    """The terminal's 3.5 in colour display asking for the card (fictional amount)."""
    path = Path(folder)/"terminal_screen.png"
    if not path.exists():
        w, h = 480, 320
        img = Image.new("RGB", (w, h), (238, 240, 244))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, 48], fill=(30, 60, 110))
        d.text((16, 24), "MAISON LUMEN", fill=(255, 255, 255), font=_ttf(SANS_BOLD, 26), anchor="lm")
        d.text((w//2, 112), "AMOUNT", fill=(70, 70, 80), font=_ttf(SANS, 28), anchor="mm")
        d.text((w//2, 170), "£47.65", fill=(20, 20, 30), font=_ttf(SANS_BOLD, 64), anchor="mm")
        d.text((w//2, 262), "INSERT, TAP OR SWIPE", fill=(30, 60, 110), font=_ttf(SANS_BOLD, 28), anchor="mm")
        img.save(path)
    return path


def pos_screen(folder):
    """Point-of-sale tablet app (landscape 10.9 in, fictional): item tiles on the left, the current sale and the
    charge button on the right."""
    path = Path(folder)/"pos_screen.png"
    if not path.exists():
        w, h = 1180, 820
        img = Image.new("RGB", (w, h), (246, 246, 248))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, 70], fill=(255, 255, 255)); d.line([(0, 70), (w, 70)], fill=(220, 220, 225), width=2)
        d.text((24, 35), "MAISON LUMEN  ·  Till 1", fill=(40, 40, 50), font=_ttf(SANS_BOLD, 30), anchor="lm")
        tiles = [("Gift tote M", BURGUNDY), ("Gift tote S", GOLD), ("Tissue", (210, 180, 190)), ("Ribbon /m", (150, 30, 50)),
                 ("Wrap service", (40, 90, 70)), ("Candle", (170, 120, 60)), ("Card", (60, 90, 150)), ("Gift box", (30, 40, 70)),
                 ("Tag", (120, 120, 130))]
        for i, (lab, col) in enumerate(tiles):
            r, c = divmod(i, 3)
            x, y = 24 + c*228, 96 + r*228
            d.rounded_rectangle([x, y, x + 210, y + 210], radius=14, fill=col)
            d.text((x + 14, y + 186), lab, fill=(255, 255, 255), font=_ttf(SANS_BOLD, 26), anchor="ls")
        d.rectangle([720, 70, w, h], fill=(255, 255, 255)); d.line([(720, 70), (720, h)], fill=(220, 220, 225), width=2)
        d.text((744, 110), "Current sale", fill=(40, 40, 50), font=_ttf(SANS_BOLD, 30), anchor="lm")
        y = 170
        for item, price in (("Amber & Fig candle", "38.00"), ("Gift tote M", "4.50"), ("Ribbon 0.6 m", "1.20"), ("Card", "3.95")):
            d.text((744, y), item, fill=(60, 60, 70), font=_ttf(SANS, 26), anchor="lm")
            d.text((w - 24, y), price, fill=(60, 60, 70), font=_ttf(SANS, 26), anchor="rm"); y += 56
        d.rounded_rectangle([744, h - 130, w - 24, h - 30], radius=16, fill=(24, 110, 220))
        d.text(((744 + w - 24)//2, h - 80), "Charge £47.65", fill=(255, 255, 255), font=_ttf(SANS_BOLD, 38), anchor="mm")
        img.save(path)
    return path


def jar_label(folder):
    """Cylindrical-UV skin of a glass candle jar: amber glass above and below a cream paper label band."""
    path = Path(folder)/"jar_label.png"
    if not path.exists():
        w, h = 1024, 256
        img = Image.new("RGB", (w, h), (120, 64, 22))
        d = ImageDraw.Draw(img)
        for y in range(h):                                                     # glass: darker towards the base
            t = y/h
            d.line([(0, y), (w, y)], fill=(int(96 + 40*t), int(50 + 22*t), int(16 + 8*t)))
        d.rectangle([0, int(h*.30), w, int(h*.72)], fill=(240, 233, 218))
        for x0 in (w*.12, w*.62):
            d.text((x0, h*.44), "AMBER & FIG", fill=(90, 60, 30), font=_font(34, bold=True), anchor="lm")
            d.text((x0, h*.60), "scented candle  220 g", fill=(120, 100, 70), font=_font(20), anchor="lm")
        # USD st has v = 0 at the image bottom, which is the jar's base (cylindrical UVs: v = z/height + 1/2)
        img.save(path)
    return path


def framed_print(folder, w_m=.142, h_m=.092, window=(.118, .068)):
    """What shows under the glass of the photo frame: an off-white mat board with a bevel-cut window and a
    landscape print (procedural: sky, layered hills, lake and pines; no photograph). 10 px/mm, w_m x h_m."""
    path = Path(folder)/f"framed_print_{int(w_m*1e4)}x{int(h_m*1e4)}.png"
    if not path.exists():
        ppm = 10000.
        w, h = int(w_m*ppm), int(h_m*ppm)
        pw, ph = int(window[0]*ppm), int(window[1]*ppm)
        rng = np.random.default_rng(21)
        # print
        yy = np.linspace(0, 1, ph, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, pw, dtype=np.float32)[None, :]
        sky = np.stack([.55 + .35*yy, .70 + .22*yy, .92 - .02*yy], -1)*np.ones((ph, pw, 1), np.float32)
        sun = np.exp(-(((xx - .72)/.05)**2 + ((yy - .30)/.08)**2))[..., None]
        img = np.clip(sky + .35*sun*np.array([1., .85, .55], np.float32), 0, 1)

        def ridge(base, amp, seed, freq):
            r = np.random.default_rng(seed)
            x = np.linspace(0, 1, pw)
            y = np.full(pw, base)
            for k in range(1, 7):
                y += amp/k*np.sin(2*np.pi*(freq*k*x + r.random()))
            return y
        horizon = .62
        for k, (base, amp, col) in enumerate(((.40, .05, (.52, .60, .70)), (.48, .045, (.36, .46, .48)), (.56, .03, (.20, .33, .26)))):
            top = ridge(base, amp, 30 + k, 1.3 + k*.6)
            mask = yy >= top[None, :]
            img = np.where(mask[..., None], np.asarray(col, np.float32), img)
        lake = yy >= horizon
        refl = img[np.clip((2*horizon - yy[:, 0])*ph, 0, ph - 1).astype(int)]
        water = .55*refl + .45*np.array([.30, .42, .50], np.float32)
        water *= (1 + .06*np.sin(yy*400 + 3*np.sin(xx*40)))[..., None]
        img = np.where(lake[..., None], water, img)
        im = Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for k in range(14):                                                    # pines on the near shore
            x = int(pw*(.04 + .065*k + .02*rng.random()))
            hh = int(ph*(.12 + .10*rng.random()))
            yb = int(ph*horizon)
            d.polygon([(x, yb - hh), (x - hh//4, yb), (x + hh//4, yb)], fill=(22, 40, 30))
        im = im.filter(ImageFilter.GaussianBlur(.8))
        # mat board with a bevel-cut window (white core edge) around the print
        mat = Image.new("RGB", (w, h), (236, 232, 222))
        g = _paper_grain(rng, max(w, h), .025)[:h, :w]
        mat = Image.fromarray((np.clip(np.asarray(mat, np.float32)/255.*(1 + g[..., None]), 0, 1)*255).astype(np.uint8))
        x0, y0 = (w - pw)//2, (h - ph)//2
        bevel = int(.0012*ppm)
        ImageDraw.Draw(mat).rectangle([x0 - bevel, y0 - bevel, x0 + pw + bevel, y0 + ph + bevel], fill=(252, 251, 248))
        ImageDraw.Draw(mat).line([(x0 - bevel, y0 + ph + bevel), (x0 + pw + bevel, y0 + ph + bevel)], fill=(214, 210, 200), width=2)
        mat.paste(im, (x0, y0))
        mat.save(path)
    return path

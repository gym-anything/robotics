"""Printed label artwork and authored surface textures (no downloaded product artwork).

Barcodes are generated at the printer's dot pitch: 203 dpi -> 0.125 mm dots, GS1-128
X = 4 dots = 0.500 mm (GS1 range 0.495-0.94 mm [GS1 LLG 1.3 6-15]); 300 dpi for the
specimen label. The payload is returned so the verifier can decode rendered views.
"""

import hashlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Visual identity only; does not set geometry, mass or physics. Reference class:
# BD 367863, lavender K2EDTA, 13 x 100 mm, 6 mL.
SPECIMEN_DRAW_ML = 6.0
SPECIMEN_STOCK_SIZE = "13 x 100 mm"

# Declared dependencies (pyproject extra `label-application`): python-barcode draws the label's symbols at
# environment creation, zxing-cpp decodes rendered views for the verifier.
DEPENDENCIES = {"barcode": "python-barcode>=0.16", "zxingcpp": "zxing-cpp>=3.1", "pypardiso": "pypardiso>=0.4"}


def _pylib(module=None):
    """Import check with an actionable message (kept as the single entry point used below)."""
    import importlib.util
    for mod, req in DEPENDENCIES.items():
        if module in (None, mod) and importlib.util.find_spec(mod) is None:
            raise ModuleNotFoundError(f"label-application needs {req}: pip install 'gym-anything-robotics[label-application]'")


def font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF"):
            p = Path(root)/n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.__file__).parent/"mpl-data/fonts/ttf"/names[0]), size)
    except Exception:
        return ImageFont.load_default()


def barcode_image(data, dpi, x_dots, height_mm, gs1=False):
    """Code 128 bars drawn dot-exactly: each module is x_dots printer dots."""
    _pylib()
    from barcode.codex import Code128
    code = Code128(data)
    modules = code.build()[0]
    if gs1:
        # GS1-128: FNC1 start (payload digits only); python-barcode encodes FNC1 via its gs1 class
        from barcode.codex import Gs1_128
        modules = Gs1_128(data).build()[0]
    quiet = 10
    width = (len(modules) + 2*quiet)*x_dots
    height = int(round(height_mm/25.4*dpi))
    img = np.full((height, width), 255, np.uint8)
    for i, m in enumerate(modules):
        if m == "1":
            img[:, (quiet+i)*x_dots:(quiet+i+1)*x_dots] = 0
    return Image.fromarray(img)


def shipping_label(folder, payload, *, gs1=False, seed=0):
    """4 x 6 in direct-thermal label at 203 dpi (812 x 1218 px). Portrait: x = CD, y = MD."""
    dpi, W, H = 203, 812, 1218
    im = Image.new("L", (W, H), 250)
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, W-7, H-7], outline=0, width=3)
    f_small, f_med, f_big = font(22), font(28, True), font(44, True)
    if gs1:
        d.text((24, 18), "FROM: NORTHGATE FULFILMENT LLC", font=f_small, fill=0)
        d.text((24, 46), "DC 04 · 1180 INDUSTRIAL PKWY · COLUMBUS OH", font=f_small, fill=0)
        d.line([12, 84, W-12, 84], fill=0, width=3)
        d.text((24, 96), "TO: RIVERSIDE MARKET #0217", font=f_med, fill=0)
        d.text((24, 136), "SHIP TO POST 43212   CARRIER: LTL-08", font=f_small, fill=0)
        d.line([12, 176, W-12, 176], fill=0, width=3)
        d.text((24, 188), "PO 4500118823   CARTON 3 OF 12   GTIN 10614141000415", font=f_small, fill=0)
        d.line([12, 560, W-12, 560], fill=0, width=3)
        d.text((24, 572), "SSCC", font=f_med, fill=0)
        bc = barcode_image(payload, dpi, 4, 32.0, gs1=True)
        text = "(00) " + payload[2:] if payload.startswith("00") else payload
    else:
        d.text((24, 18), "NORTHGATE FULFILMENT", font=f_med, fill=0)
        d.text((24, 56), "1180 INDUSTRIAL PKWY, COLUMBUS OH 43219", font=f_small, fill=0)
        d.line([12, 94, W-12, 94], fill=0, width=3)
        d.text((24, 108), "SHIP TO:", font=f_small, fill=0)
        d.text((60, 138), "J. ALVAREZ", font=f_big, fill=0)
        d.text((60, 190), "2270 LAKEVIEW TERRACE APT 4B", font=f_med, fill=0)
        d.text((60, 226), "SPRINGFIELD IL 62704", font=f_med, fill=0)
        d.line([12, 280, W-12, 280], fill=0, width=5)
        d.rectangle([24, 300, 190, 460], fill=0)
        d.text((52, 340), "G", font=font(90, True), fill=255)
        d.text((220, 310), "GROUND", font=font(56, True), fill=0)
        d.text((220, 380), "WT 2.6 KG   1 OF 1", font=f_med, fill=0)
        d.line([12, 560, W-12, 560], fill=0, width=5)
        d.text((24, 572), "TRACKING #", font=f_med, fill=0)
        bc = barcode_image(payload, dpi, 3, 25.0)     # carrier tracking code, X = 0.375 mm
        text = " ".join(payload[i:i+4] for i in range(0, len(payload), 4))
    # Bars are parallel to the image y axis (= label MD): with the label portrait on a standing
    # carton face they are vertical ("picket fence", GS1 6-20). GS1 puts the SSCC at the bottom.
    bx = max(12, (W - bc.width)//2)
    top = H - 30 - 44 - bc.height if gs1 else 620
    im.paste(bc, (bx, top))
    d.text(((W - d.textlength(text, font=f_med))/2, top+bc.height+8), text, font=f_med, fill=0)
    if not gs1:
        d.line([12, 1080, W-12, 1080], fill=0, width=3)
        d.text((24, 1096), f"REF LA-{seed:06d}", font=f_small, fill=0)
        d.text((24, 1126), "PRINTED 203 DPI DIRECT THERMAL", font=font(18), fill=0)
    else:
        d.text((24, 230), f"REF LA-{seed:06d}   PRINTED 203 DPI DIRECT THERMAL", font=font(18), fill=0)
    geometry = dict(barcode_px=(bx, top, bx+bc.width, top+bc.height), dpi=dpi, image_px=(W, H),
                    x_dim_mm=(4 if gs1 else 3)*25.4/dpi)
    return _save(folder, "label_art", im, payload, geometry)


def specimen_label(folder, payload, *, seed=0):
    """2 x 1 in specimen label, 300 dpi (600 x 300 px). x = CD (2 in, along tube axis), y = MD."""
    dpi, W, H = 300, 600, 300
    im = Image.new("L", (W, H), 250)
    d = ImageDraw.Draw(im)
    d.text((14, 8), "ALVAREZ, JORGE M", font=font(30, True), fill=0)   # patient name top-left [AUTO12]
    d.text((14, 44), "DOB 1971-03-14   MRN 00482217", font=font(20), fill=0)
    bc = barcode_image(payload, dpi, 3, 9.0)
    bx = max(8, (W - bc.width)//2)
    im.paste(bc, (bx, 82))
    d.text((14, 82+bc.height+4), f"{payload}   EDTA {SPECIMEN_DRAW_ML:g} mL   2026-09-23 07:42", font=font(18), fill=0)
    geometry = dict(barcode_px=(bx, 82, bx+bc.width, 82+bc.height), dpi=dpi, image_px=(W, H), x_dim_mm=3*25.4/dpi)
    return _save(folder, "label_art", im, payload, geometry)


def part_label(folder, payload, *, seed=0):
    """2 x 4 in silver polyester asset/ID label (thermal transfer), 300 dpi: x = CD (4 in), y = MD.
    The printed part number is the one the barcode encodes (payload NG-<part number>-<serial>)."""
    dpi, W, H = 300, 1200, 600
    rng = np.random.default_rng(3)
    base = (192 + rng.normal(0, 5, (H, W))).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(base).filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(im)
    part_no = "-".join(payload.split("-")[1:3])
    d.text((30, 20), "PROPERTY OF NORTHGATE MFG — RECEIVING", font=font(34, True), fill=10)
    d.text((30, 64), f"PART {part_no}", font=font(40, True), fill=10)
    d.text((30 + d.textlength(f"PART {part_no}", font=font(40, True)) + 30, 72), "REV C   QTY 1", font=font(28), fill=10)
    bc = barcode_image(payload, dpi, 4, 12.0)
    bx = max(20, (W - bc.width)//2)
    im.paste(bc, (bx, 130))
    d.text((bx, 130+bc.height+8), payload, font=font(28, True), fill=10)
    geometry = dict(barcode_px=(bx, 130, bx+bc.width, 130+bc.height), dpi=dpi, image_px=(W, H), x_dim_mm=4*25.4/dpi)
    return _save(folder, "label_art", im, payload, geometry)


def old_address_label(folder, payload, *, seed=0):
    """cover-old-label: the previous shipment's 4 x 3 in direct-thermal address label (203 dpi, 812 x 609 px;
    x = the 4 in side). A standard 4-inch-roll label size; the format is an authored choice, not a carrier
    rule. It carries its own (now invalid) tracking barcode, which the new label must hide."""
    dpi, W, H = 203, 812, 609
    im = Image.new("L", (W, H), 246)
    d = ImageDraw.Draw(im)
    d.rectangle([6, 6, W-7, H-7], outline=0, width=3)
    f_small, f_med, f_big = font(20), font(26, True), font(38, True)
    d.text((22, 16), "FROM: HARBOR LANE GOODS, 88 PIER ST, DULUTH MN 55802", font=f_small, fill=0)
    d.line([12, 46, W-12, 46], fill=0, width=3)
    d.text((22, 56), "SHIP TO:", font=f_small, fill=0)
    d.text((50, 82), "M. OKAFOR", font=f_big, fill=0)
    d.text((50, 128), "414 BIRCHWOOD AVE", font=f_med, fill=0)
    d.text((50, 162), "MADISON WI 53704", font=f_med, fill=0)
    d.line([12, 204, W-12, 204], fill=0, width=4)
    d.text((22, 214), "TRACKING #", font=f_med, fill=0)
    bc = barcode_image(payload, dpi, 3, 22.0)
    bx = max(12, (W - bc.width)//2)
    im.paste(bc, (bx, 252))
    text = " ".join(payload[i:i+4] for i in range(0, len(payload), 4))
    d.text(((W - d.textlength(text, font=f_med))/2, 252 + bc.height + 8), text, font=f_med, fill=0)
    d.text((22, H - 40), f"REF HL-{seed:06d}   SHIPPED 2026-08-30", font=font(18), fill=0)
    geometry = dict(barcode_px=(bx, 252, bx+bc.width, 252+bc.height), dpi=dpi, image_px=(W, H), x_dim_mm=3*25.4/dpi)
    return _save(folder, "old_label", im, payload, geometry)


def plate_marking(folder, part_no, *, name="plate_marking"):
    """Laser-annealed part marking on a stainless plate (dark oxide text, flush with the surface; no relief),
    drawn as a decal 64 x 14 mm at 20 px/mm."""
    W, H = 1280, 280
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(150, True)
    text = f"P/N {part_no}"
    tw = d.textlength(text, font=f)
    d.text(((W - tw)/2, 40), text, font=f, fill=(28, 26, 30, 235))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{name}.png"
    im.save(path)
    return str(path)


def _save(folder, name, im, payload, geometry):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{name}.png"
    im.convert("RGB").save(path)
    return dict(path=str(path), payload=payload, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **geometry)


def kraft_texture(folder, size=1024, seed=7):
    """Authored corrugated-kraft liner texture with faint flute shadows (8 mm pitch, C-flute)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (size, size))
    from scipy.ndimage import gaussian_filter
    fibre = gaussian_filter(n, (0.6, 6)) * 9 + gaussian_filter(n, 3)*6
    y = np.arange(size)[:, None]
    flute = 3*np.cos(2*np.pi*y/(size*8/400))       # texture spans 0.4 m: 8 mm flutes
    base = np.array([146, 110, 76], float)     # unbleached kraft liner (measured-look sRGB)
    rgb = base[None, None, :] + (fibre + flute)[:, :, None]*np.array([1, .85, .6])[None, None, :]
    path = Path(folder)/"kraft.png"
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
    return str(path)


def decode(image_path_or_array):
    """zxing-cpp decode of a rendered view. Returns list of (text, format)."""
    _pylib()
    import zxingcpp
    img = Image.open(image_path_or_array) if isinstance(image_path_or_array, (str, Path)) else \
        Image.fromarray(np.asarray(image_path_or_array))
    return [(r.text, str(r.format)) for r in zxingcpp.read_barcodes(img)]

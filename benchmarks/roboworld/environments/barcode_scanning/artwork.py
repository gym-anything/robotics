"""Printed symbols and labels at exact physical scale.

A symbol is a list of elements (1-D: widths in modules, dark/light) or a module matrix (2-D).
Its texture has a whole number of texels per module, and the texture's physical extent is set by
the mesh it is mapped onto, so a printed module is exactly X millimetres wide in the scene.

Symbologies and their geometry:
  UPC-A / EAN-13 / UPC-E / EAN-8   GS1 General Specifications via the GS1 Canada symbol reference:
                                   X 0.264-0.66 mm, bar height 22.85 mm at 100 %,
                                   quiet zones UPC-A 9X/9X, EAN-13 11X/7X, UPC-E 9X/7X, EAN-8 7X/7X
  Code 128 / GS1-128               module bitmap from zxing-cpp's encoder; quiet zones 10X
  Codabar                          encoded here with wide:narrow 3:1 (the symbology allows 2.25-3.0;
                                   zxing's writer uses 2:1), inter-character gap 1X, quiet zones 10X
  ITF-14                           encoded here with wide:narrow 2.5:1 (GS1 recommends 2.25-3.0),
                                   quiet zones 10X, bearer bars 2X (GS1 minimum)
  Data Matrix / QR                 module matrix from zxing-cpp; quiet zones 1X / 4X
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_MONO = FONT_DIR/"DejaVuSansMono.ttf"
FONT_SANS = FONT_DIR/"DejaVuSans.ttf"
FONT_BOLD = FONT_DIR/"DejaVuSans-Bold.ttf"
FONT_CONDENSED_BOLD = FONT_DIR/"DejaVuSansCondensed-Bold.ttf"


def gtin_check_digit(body):
    """GS1 mod-10 check digit for a GTIN body (all digits except the check digit)."""
    digits = [int(c) for c in str(body)]
    total = sum(d*(3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return (10 - total % 10) % 10


def gtin(body):
    return f"{body}{gtin_check_digit(body)}"


def upce_expand(body7):
    """UPC-E (number system + 6 digits, no check digit) -> its UPC-A body (11 digits). The GTIN of a UPC-E symbol is
    the expanded GTIN-12 (GS1 General Specifications, zero suppression)."""
    ns, d = body7[0], body7[1:]
    last = d[5]
    if last in "012":
        core = d[0] + d[1] + last + "0000" + d[2] + d[3] + d[4]
    elif last == "3":
        core = d[0] + d[1] + d[2] + "00000" + d[3] + d[4]
    elif last == "4":
        core = d[0] + d[1] + d[2] + d[3] + "00000" + d[4]
    else:
        core = d[0] + d[1] + d[2] + d[3] + d[4] + "0000" + last
    return ns + core


def gtin_of(kind, body):
    """The GTIN a scanner reports for a retail symbol body (UPC-E expanded); other symbologies return their data."""
    if kind == "UPCE":
        return gtin(upce_expand(body))
    if kind in ("UPCA", "EAN13", "EAN8"):
        return gtin(body)
    return body


# Print reflectances. Textures hold the printed colour relative to the substrate's white; the renderer's printed
# materials scale the whole print by the substrate reflectance (render.PRINT_WHITE = 0.85, white label stock /
# coated board, estimate) and black ink prints at 5 % of that white (sRGB 63 -> linear 0.05; Rmin 2-5 % for
# carbon-black thermal-transfer and litho inks, estimate): symbol contrast 0.85 - 0.04 = 81 %.
INK = (63, 63, 63)

# ------------------------------------------------------------------------------ encoders
CODABAR = {  # 7 elements (bar, space, bar, ...), 1 = wide
    "0": "0000011", "1": "0000110", "2": "0001001", "3": "1100000", "4": "0010010", "5": "1000010",
    "6": "0100001", "7": "0100100", "8": "0110000", "9": "1001000", "-": "0001100", "$": "0011000",
    ":": "1000101", "/": "1010001", ".": "1010100", "+": "0010101",
    "A": "0011010", "B": "0101001", "C": "0001011", "D": "0001110"}
ITF_DIGITS = {"0": "00110", "1": "10001", "2": "01001", "3": "11000", "4": "00101", "5": "10100",
              "6": "01100", "7": "00011", "8": "10010", "9": "01010"}


CODE39 = {  # 9 elements (bar, space, ... bar), 1 = wide (ANSI/AIM BC1, ISO/IEC 16388)
    "0": "000110100", "1": "100100001", "2": "001100001", "3": "101100000", "4": "000110001", "5": "100110000",
    "6": "001110000", "7": "000100101", "8": "100100100", "9": "001100100", "A": "100001001", "B": "001001001",
    "C": "101001000", "D": "000011001", "E": "100011000", "F": "001011000", "G": "000001101", "H": "100001100",
    "I": "001001100", "J": "000011100", "K": "100000011", "L": "001000011", "M": "101000010", "N": "000010011",
    "O": "100010010", "P": "001010010", "Q": "000000111", "R": "100000110", "S": "001000110", "T": "000010110",
    "U": "110000001", "V": "011000001", "W": "111000000", "X": "010010001", "Y": "110010000", "Z": "011010000",
    "-": "010000101", ".": "110000100", " ": "011000100", "*": "010010100", "$": "010101000", "/": "010100010",
    "+": "010001010", "%": "000101010"}


def code39_elements(data, wide=2.5):
    """Code 39 with start/stop '*', wide:narrow 2.5 (authored; ISO/IEC 16388 allows 2.0-3.0, >= 2.2 below 20 mil)."""
    out = []
    full = "*" + data + "*"
    for k, ch in enumerate(full):
        for i, w in enumerate(CODE39[ch]):
            out.append((wide if w == "1" else 1.0, i % 2 == 0))
        if k < len(full) - 1:
            out.append((1.0, False))
    return out


def codabar_elements(data, wide=3.0):
    if len(data) < 3 or data[0] not in "ABCD" or data[-1] not in "ABCD":
        raise ValueError("Codabar data must start and end with A-D")
    out = []
    for k, ch in enumerate(data):
        pattern = CODABAR[ch]
        for i, w in enumerate(pattern):
            out.append((wide if w == "1" else 1.0, i % 2 == 0))
        if k < len(data) - 1:
            out.append((1.0, False))          # inter-character gap
    return out


def itf_elements(data, wide=2.5):
    if len(data) % 2 or not data.isdigit():
        raise ValueError("ITF needs an even number of digits")
    out = [(1., True), (1., False), (1., True), (1., False)]          # start: n-bar n-space n-bar n-space
    for a, b in zip(data[::2], data[1::2]):
        for pa, pb in zip(ITF_DIGITS[a], ITF_DIGITS[b]):
            out.append((wide if pa == "1" else 1., True))
            out.append((wide if pb == "1" else 1., False))
    out += [(wide, True), (1., False), (1., True)]                     # stop: w-bar n-space n-bar
    return out


def _runs(row):
    """Element list from one module row (1 = dark)."""
    out, cur, n = [], bool(row[0]), 0
    for v in row:
        if bool(v) == cur:
            n += 1
        else:
            out.append((float(n), cur)); cur, n = bool(v), 1
    out.append((float(n), cur))
    while out and not out[0][1]:
        out.pop(0)
    while out and not out[-1][1]:
        out.pop()
    return out


def _zxing_bits(fmt, data):
    import warnings
    import zxingcpp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        img = np.asarray(zxingcpp.write_barcode(getattr(zxingcpp.BarcodeFormat, fmt), data, 0, 0, 0))
    return (img < 128).astype(np.uint8)


QUIET = {"UPCA": (9, 9), "EAN13": (11, 7), "UPCE": (9, 7), "EAN8": (7, 7), "Code128": (10, 10),
         "Codabar": (10, 10), "ITF": (10, 10), "Code39": (10, 10), "DataMatrix": (1, 1), "QRCode": (4, 4), "PDF417": (2, 2)}
ZXING_FORMAT = {"UPCA": "UPCA", "EAN13": "EAN13", "UPCE": "UPCE", "EAN8": "EAN8", "Code128": "Code128",
                "Codabar": "Codabar", "ITF": "ITF", "Code39": "Code39", "DataMatrix": "DataMatrix", "QRCode": "QRCode",
                "PDF417": "PDF417"}
LINEAR = ("UPCA", "EAN13", "UPCE", "EAN8", "Code128", "Codabar", "ITF", "Code39")


@dataclass
class Symbol:
    kind: str                      # key of QUIET
    data: str                      # encoded data (what a decoder must return, modulo formatting)
    x_mm: float                    # module width X
    bar_height_mm: float = 0.      # 1-D bar height (0: symbology default)
    hri: bool = True               # print the human-readable interpretation
    elements: list = field(default_factory=list)
    matrix: np.ndarray = None

    def __post_init__(self):
        if self.kind in ("Codabar",):
            self.elements = codabar_elements(self.data)
        elif self.kind == "Code39":
            self.elements = code39_elements(self.data)
        elif self.kind == "ITF":
            self.elements = itf_elements(self.data)
        elif self.kind in LINEAR:
            bits = _zxing_bits(ZXING_FORMAT[self.kind], self.data)
            self.elements = _runs(bits[bits.shape[0]//2])
        else:
            self.matrix = _zxing_bits(ZXING_FORMAT[self.kind], self.data)
        if self.kind in ("UPCA", "EAN13", "UPCE", "EAN8") and not self.bar_height_mm:
            self.bar_height_mm = 22.85*self.x_mm/.33          # GS1: 22.85 mm at 100 % (X = 0.33 mm)
        if self.kind in LINEAR and not self.bar_height_mm:
            self.bar_height_mm = max(12.7, 0.15*self.width_modules*self.x_mm)

    @property
    def width_modules(self):
        if self.matrix is not None:
            return self.matrix.shape[1]
        return sum(w for w, _ in self.elements)

    @property
    def quiet(self):
        return QUIET[self.kind]

    @property
    def expected_text(self):
        """Decoder text for this symbol (zxing reports UPC-A as 12 digits, UPC-E as 8, Codabar with its start/stop)."""
        return self.data


def upca(body11, mag=1.0, **kw):
    return Symbol("UPCA", gtin(body11), .33*mag, **kw)


def ean13(body12, mag=1.0, **kw):
    return Symbol("EAN13", gtin(body12), .33*mag, **kw)


def upce(body, mag=1.0, **kw):
    return Symbol("UPCE", body, .33*mag, **kw)


# ------------------------------------------------------------------------------ rasterisation
def _font(path, px):
    from PIL import ImageFont
    return ImageFont.truetype(str(path if Path(path).exists() else FONT_SANS), max(6, int(round(px))))


def render_symbol(sym, texels_per_module=10, ink=None, paper=(255, 255, 255), hri_font=FONT_MONO):
    """PIL RGB image of the symbol with its quiet zones (and HRI for 1-D). Returns (image, width_mm, height_mm).
    One module = `texels_per_module` texels; the physical size follows from X."""
    from PIL import Image, ImageDraw
    ink = INK if ink is None else ink
    k = int(texels_per_module)
    ql, qr = sym.quiet
    if sym.matrix is not None:
        m = sym.matrix
        n = m.shape[0]
        W = (m.shape[1] + ql + qr)*k
        H = (n + ql + qr)*k
        img = np.full((H, W, 3), paper, np.uint8)
        big = np.kron(m, np.ones((k, k), np.uint8)).astype(bool)
        img[ql*k:ql*k + big.shape[0], ql*k:ql*k + big.shape[1]][big] = ink
        return Image.fromarray(img), W/k*sym.x_mm, H/k*sym.x_mm
    bar_h = int(round(sym.bar_height_mm/sym.x_mm*k))
    hri_h = int(round(11*k)) if sym.hri else int(round(2*k))
    W = int(round((sym.width_modules + ql + qr)*k))
    top = int(round(2*k))
    bearer = int(round(2*k)) if sym.kind == "ITF" else 0
    H = top + bearer + bar_h + bearer + hri_h
    img = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(img)
    x = ql*k
    guards = []
    if sym.kind in ("UPCA", "EAN13", "UPCE", "EAN8"):
        # guard bars extend 5X below the data bars (GS1)
        n = sym.width_modules
        guard_cols = set()
        pos = 0.
        edges = []
        for w, dark in sym.elements:
            edges.append((pos, pos + w, dark)); pos += w
        tail = 7 if sym.kind == "UPCA" else 0
        for a, b_, dark in edges:
            if not dark:
                continue
            centre = (a + b_)/2
            is_guard = centre < 3 + tail or centre > n - 3 - tail or (sym.kind in ("UPCA", "EAN13") and abs(centre - n/2) < 2.6) \
                or (sym.kind == "EAN8" and abs(centre - n/2) < 2.6) or (sym.kind == "UPCE" and centre > n - 6)
            guards.append((a, b_, is_guard))
    else:
        pos = 0.
        for w, dark in sym.elements:
            if dark:
                guards.append((pos, pos + w, False))
            pos += w
    for a, b_, g in guards:
        x0, x1 = x + a*k, x + b_*k
        y1 = top + bearer + bar_h + (int(5*k) if g and sym.hri else 0)
        d.rectangle([int(round(x0)), top + bearer, int(round(x1)) - 1, y1 - 1], fill=ink)
    if bearer:
        d.rectangle([0, top, W - 1, top + bearer - 1], fill=ink)
        d.rectangle([0, top + bearer + bar_h, W - 1, top + 2*bearer + bar_h - 1], fill=ink)
    if sym.hri:
        f = _font(hri_font, 8*k)
        text = _hri(sym)
        y = top + 2*bearer + bar_h + int(1.0*k)
        if sym.kind in ("UPCA", "EAN13"):
            n = sym.width_modules
            first, left, right, last = _split_upc(sym)
            if first:
                d.text((x - 1*k, y), first, font=f, fill=ink, anchor="ra")
            d.text((x + (3 + (7 if sym.kind == "UPCA" else 0) + (n/2 - 3 - (7 if sym.kind == "UPCA" else 0))/2)*k, y),
                   left, font=f, fill=ink, anchor="ma")
            d.text((x + (n/2 + (n/2 - 3 - (7 if sym.kind == "UPCA" else 0))/2)*k, y), right, font=f, fill=ink, anchor="ma")
            if last:
                d.text((x + (n + 1)*k, y), last, font=f, fill=ink, anchor="la")
        else:
            d.text((W/2, y), text, font=f, fill=ink, anchor="ma")
    return img, W/k*sym.x_mm, H/k*sym.x_mm


def _hri(sym):
    if sym.kind == "Codabar":
        return sym.data[1:-1]
    if sym.kind == "ITF":
        d = sym.data
        return f"{d[0]} {d[1:3]} {d[3:8]} {d[8:13]} {d[13:]}" if len(d) == 14 else d
    return sym.data


def _split_upc(sym):
    d = sym.data
    if sym.kind == "UPCA":
        return d[0], d[1:6], d[6:11], d[11]
    return d[0], d[1:7], d[7:13], ""


# ------------------------------------------------------------------------------ labels
def mm2px(mm, dpmm):
    return int(round(mm*dpmm))


def compose(width_mm, height_mm, dpmm, background=(255, 255, 255)):
    from PIL import Image
    return Image.new("RGB", (mm2px(width_mm, dpmm), mm2px(height_mm, dpmm)), background)


def paste_symbol(canvas, dpmm, sym, x_mm, y_mm, texels_per_module=None, **kw):
    """Paste `sym` with its top-left at (x_mm, y_mm), resampled to the canvas' dots per mm. A canvas with
    dpmm >= 8/X keeps at least 8 texels per module; bars are placed with area-weighted resampling."""
    from PIL import Image
    k = texels_per_module or max(4, int(np.ceil(dpmm*sym.x_mm*2)))
    img, w_mm, h_mm = render_symbol(sym, k, **kw)
    target = (mm2px(w_mm, dpmm), mm2px(h_mm, dpmm))
    img = img.resize(target, Image.BOX)
    canvas.paste(img, (mm2px(x_mm, dpmm), mm2px(y_mm, dpmm)))
    return w_mm, h_mm


def text(canvas, dpmm, x_mm, y_mm, s, size_mm, font=FONT_SANS, fill=None, anchor="la"):
    from PIL import ImageDraw
    ImageDraw.Draw(canvas).text((x_mm*dpmm, y_mm*dpmm), s, font=_font(font, size_mm*dpmm), fill=INK if fill is None else fill, anchor=anchor)


def location_label(code, folder, dpmm=24):
    """Shelf-edge location label, 76 x 25 mm (1 x 3 in) white matte stock: Code 128 at X = 0.381 mm (15 mil),
    human-readable location text above. Returns (path, width_m, height_m, symbol)."""
    sym = Symbol("Code128", code, .381, bar_height_mm=11.0, hri=False)
    c = compose(76.2, 25.4, dpmm)
    text(c, dpmm, 38.1, 1.2, code, 6.5, FONT_BOLD, anchor="ma")
    w, h = sym.width_modules*sym.x_mm, sym.bar_height_mm
    paste_symbol(c, dpmm, sym, 38.1 - (w/2 + 10*sym.x_mm), 8.6)
    path = Path(folder)/f"loc_{code}.png"
    c.save(path)
    return path, .0762, .0254, sym


def barcode_patch(sym, folder, name, dpmm=None, margin_mm=1.0):
    """A symbol printed on its own white area (the printed barcode box of a package, or a small label).
    Returns (path, width_m, height_m)."""
    from PIL import Image
    k = max(8, int(np.ceil(8/sym.x_mm*sym.x_mm)))
    img, w_mm, h_mm = render_symbol(sym, 10)
    dpmm = dpmm or 10/sym.x_mm
    c = compose(w_mm + 2*margin_mm, h_mm + 2*margin_mm, dpmm)
    c.paste(img.resize((mm2px(w_mm, dpmm), mm2px(h_mm, dpmm)), Image.BOX), (mm2px(margin_mm, dpmm), mm2px(margin_mm, dpmm)))
    path = Path(folder)/f"{name}.png"
    c.save(path)
    return path, (w_mm + 2*margin_mm)/1e3, (h_mm + 2*margin_mm)/1e3

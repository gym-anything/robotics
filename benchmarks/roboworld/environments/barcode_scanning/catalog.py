"""Stock items: geometry, mass, surface finish and where their barcodes are printed.

Brands and artwork are fictional (as in deposit_cans); GTINs carry correct GS1 check digits. Geometry
sources: can-size codes (first number = diameter, second = height, inches + sixteenths); water
bottle/case; paper ream mass derived (75 g/m2 x 0.0603 m2 x 500); AkroBin 30230; the rest are
estimates of typical retail packs. Barcode sizes follow GS1 and placement follows GS1: the
lower-right quadrant of the back panel; on curved items the ladder orientation where the 30-degree rule
requires it.

Frames: a box item's body frame has x along its width, y along its depth (front face at -y), z up; a
cylinder's axis is z. Barcode placement gives the face and the symbol centre in that face's (u, v)
coordinates (u to the right, v up, seen from outside), plus picket/ladder orientation.
"""

from dataclasses import dataclass, field
import math

IN = 0.0254


def can_dims(code):
    """'211x400' -> (radius m, height m)."""
    d, h = code.split("x")
    to_m = lambda s: (int(s[0]) + int(s[1:])/16)*IN
    return to_m(d)/2, to_m(h)


@dataclass(frozen=True)
class Barcode:
    kind: str                  # artwork.Symbol kind
    body: str                  # GTIN body without check digit (EAN/UPC) or full data
    x_mm: float                # module width
    face: str                  # box: front/back/left/right/top/bottom; cylinder: wrap
    u: float                   # symbol centre on the face, metres from the face centre (right +)
    v: float                   # metres from the face centre (up +)
    orientation: str = "picket"   # picket (bars vertical) or ladder (bars horizontal)
    angle_deg: float = 180.    # cylinder: azimuth of the symbol centre from the front (front = -y)
    finish: str = None         # optional override of the item's finish for the printed barcode area


@dataclass(frozen=True)
class SKU:
    id: str
    name: str
    brand: str
    shape: str                 # box | cyl
    dims: tuple                # box (w, d, h); cyl (r, h)
    mass: float
    finish: str                # matte | semigloss | glossy | film | metal | glass
    mu: float
    palette: tuple             # (primary, secondary, text) sRGB 0-255
    barcode: Barcode
    size_text: str = ""
    category: str = "grocery"
    extra: dict = field(default_factory=dict)


R1, H1 = can_dims("211x400")
R2, H2 = can_dims("202x214")

SKUS = {s.id: s for s in [
    SKU("K01", "Condensed Tomato Soup", "Harvest Table", "cyl", (R1, H1), 0.35, "matte", 0.30,
        ((178, 34, 34), (245, 238, 225), (255, 255, 255)),
        Barcode("UPCA", "07841200351", .33, "wrap", 0., -0.012, "picket", 180.), "NET WT 10.75 OZ (305 g)"),
    SKU("K02", "Tomato Paste", "Harvest Table", "cyl", (R2, H2), 0.20, "metal", 0.30,
        ((196, 40, 30), (40, 90, 40), (255, 255, 255)),
        Barcode("UPCA", "07841200417", .264, "wrap", 0., 0., "ladder", 180.), "NET WT 6 OZ (170 g)"),
    SKU("K03", "Oat Crunch Cereal", "Morning Field", "box", (0.195, 0.070, 0.290), 0.43, "glossy", 0.40,
        ((230, 170, 40), (120, 60, 20), (255, 255, 255)),
        Barcode("UPCA", "08531400226", .33, "back", 0.052, -0.105), "NET WT 12 OZ (340 g)"),
    # 24 oz glass sauce jar: 63 mm lug finish (63-2030); body 85 mm x 160 mm overall (estimate within the
    # 24-32 oz spaghetti-jar range); lug cap 66 mm x 14 mm and a 12 mm shoulder are estimates. The cap is the
    # part a hand (or a Franka Hand, 80 mm stroke) can close on; the body is wider than the gripper's stroke.
    SKU("K04", "Marinara Pasta Sauce", "Nonna Vera", "cyl", (0.0425, 0.160), 0.90, "glass", 0.35,
        ((150, 20, 30), (240, 230, 200), (255, 255, 255)),
        Barcode("UPCA", "07294100158", .33, "wrap", 0., -0.020, "picket", 180., finish="semigloss"),
        "NET WT 24 OZ (680 g)",
        extra=dict(neck=dict(shoulder=.012, cap_r=.033, cap_h=.014, panel=(-.068, .040),
                             body_mat="jar_glass", shoulder_mat="jar_glass", cap_mat="lid_gold"))),
    # 16.9 oz PET bottle: 2.532 in diameter, 8.083 in to the top of its 28 mm PCO finish, 2.75 in label panel;
    # the 28 mm closure adds ~8 mm (estimate); a 55 mm shoulder and a 25 mm finish+cap are estimates.
    SKU("K05", "Spring Water", "Clearbrook", "cyl", (0.03216, 0.2133), 0.51, "film", 0.35,
        ((40, 120, 200), (230, 245, 255), (255, 255, 255)),
        Barcode("UPCA", "06110300502", .33, "wrap", 0., -0.025, "ladder", 180.), "16.9 FL OZ (500 mL)",
        extra=dict(neck=dict(shoulder=.055, cap_r=.0150, cap_h=.025, panel=(-.060, .010),
                             body_mat="pet_water", shoulder_mat="pet_water", cap_mat="cap_blue"))),
    SKU("K06", "Condensed Tomato Soup 6-pack", "Harvest Table", "box", (0.205, 0.137, 0.102), 2.20, "film", 0.35,
        ((178, 34, 34), (245, 238, 225), (255, 255, 255)),
        Barcode("UPCA", "07841200360", .33, "front", 0.060, -0.010), "6 x 10.75 OZ",
        extra=dict(inner="K01", grid=(3, 2))),
    SKU("K07", "Nitrile Exam Gloves, Medium, 100 ct", "SafeGrip", "box", (0.240, 0.125, 0.065), 0.55, "glossy", 0.40,
        ((30, 90, 170), (235, 240, 250), (255, 255, 255)),
        Barcode("UPCA", "08462300104", .33, "right", 0.020, 0.0), "100 GLOVES", category="supplies"),
    SKU("K08", "Copy Paper, Letter, 20 lb, 500 sheets", "Brightline", "box", (0.279, 0.216, 0.053), 2.26, "matte", 0.40,
        ((250, 250, 250), (20, 110, 60), (20, 20, 20)),
        Barcode("UPCA", "08825400501", .33, "right", 0.040, 0.0), "8.5 x 11 in", category="supplies"),
    SKU("K09", "AA Alkaline Batteries 4-pack", "Voltra", "box", (0.100, 0.025, 0.180), 0.12, "film", 0.40,
        ((20, 20, 20), (240, 190, 20), (255, 255, 255)),
        Barcode("UPCA", "04133300411", .264, "back", 0.020, -0.060), "4 AA", category="general"),
    SKU("K10", "Lip Balm, Original", "Meadow", "cyl", (0.0078, 0.067), 0.01, "glossy", 0.40,
        ((60, 120, 60), (255, 255, 255), (255, 255, 255)),
        Barcode("UPCE", "0451270", .264, "wrap", 0., 0.0, "ladder", 180.), "0.15 OZ (4.25 g)", category="general"),
    SKU("K11", "Spring Water 24-pack", "Clearbrook", "box", (0.406, 0.279, 0.229), 12.5, "film", 0.40,
        ((40, 120, 200), (230, 245, 255), (255, 255, 255)),
        Barcode("UPCA", "06110300524", .33, "left", 0.0, 0.040), "24 x 16.9 FL OZ", extra=dict(inner="K05", grid=(6, 4))),
    SKU("K12", "The Tidewater Almanac", "Harbor Press", "box", (0.156, 0.030, 0.235), 0.65, "film", 0.30,
        ((40, 60, 90), (220, 200, 150), (255, 255, 255)),
        Barcode("EAN13", "978193218441", .33, "back", 0.030, -0.080), "Hardcover", category="library"),
    SKU("K13", "Notes on Salt and Wind", "Kestrel Books", "box", (0.108, 0.025, 0.175), 0.25, "semigloss", 0.35,
        ((200, 110, 40), (250, 240, 220), (20, 20, 20)),
        Barcode("EAN13", "978188721019", .30, "back", 0.018, -0.055), "Paperback", category="library"),
]}


def symbol_for(sku):
    from .artwork import Symbol, gtin
    b = sku.barcode
    if b.kind in ("UPCA", "EAN13", "EAN8"):
        return Symbol(b.kind, gtin(b.body), b.x_mm)
    return Symbol(b.kind, b.body, b.x_mm)


def sku_gtin14(sku):
    """GTIN-14 of a SKU's barcode as the host records it (the one truth mapping for evaluator, host and tests)."""
    from .artwork import gtin_of
    from .decoder import gtin14
    return gtin14(gtin_of(sku.barcode.kind, sku.barcode.body))


def neck(sku):
    """Necked container (jar, bottle): z extents (item-centred) of its straight body, shoulder and cap, the label
    panel, and the render materials; None for plain cylinders."""
    n = sku.extra.get("neck") if sku.shape == "cyl" else None
    if not n:
        return None
    r, h = sku.dims
    z_top = h/2
    z_cap = z_top - n["cap_h"]
    z_sh = z_cap - n["shoulder"]
    return dict(r=r, cap_r=n["cap_r"], body=(-h/2, z_sh), shoulder=(z_sh, z_cap), cap=(z_cap, z_top), panel=n["panel"],
                body_mat=n["body_mat"], shoulder_mat=n["shoulder_mat"], cap_mat=n["cap_mat"])

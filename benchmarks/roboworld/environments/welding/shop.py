"""The fabrication shop around the robotic welding cell: the building, the welding equipment, the cell's furniture and
the shop's dressing, authored into the render stage (render.py export_scene calls `author`).

Render-only except where noted: the welding equipment (power source, wire feeder, gas cylinder, running gear), the
camera mounts and the welding screens are within an arm's reach, so each is ALSO a static collider of the same
shape in scene.py, from the same numbers in layout.py; everything else here stands outside both arms' reach and has
no collision. Nothing here feeds a physics parameter.

Provenance of dimensions: [read] from the maker's published sheet (cited where used); [estimate] a typical size, not
looked up; [authored] chosen by this build. Brand names on the equipment are fictional ("FERRUM ARC"); the CC0
Poly Haven models are pinned by SHA-256 in visual_assets.json; the signs follow the ANSI Z535.4 layout (signal word
panel, colour by hazard level) with text written for this shop.
"""
import math
from pathlib import Path

import numpy as np

from . import layout as LY

ROOT = Path(__file__).resolve().parent
TEX = ROOT/".cache"/"textures"/"shop_v1"

# ---- the building [authored]: a 10 x 7.5 m bay of a steel-framed fabrication shop, 5.2 m to the roof deck; block
# dado to 1.2 m, insulated profiled-steel cladding above; a 3 x 3.6 m roll-up door in the left wall, a personnel door
# in the front wall, a strip of windows under the eaves at the back and right
ROOM = dict(x=(-4.6, 5.4), y=(-4.4, 3.1), height=5.2, wall=.2, dado=1.2)
OPENINGS = {"left": [((-3.3, -.3), (0., 3.6), "rollup")],
            "front": [((-3.4, -2.45), (0., 2.1), "door")],
            "back": [((-3.8, 4.6), (3.3, 4.3), "window")],
            "right": [((-3.6, 2.4), (3.3, 4.3), "window")]}
FRAMES_X = (-2.1, 2.9)                                # portal frames (columns and rafters), 5 m bays
LAMPS = [(x, y) for x in (-2.6, 0., 2.6) for y in (-2.3, 2.2)]      # hung from purlins
LAMP_Z = 5.05
CANOPY = dict(center=(0., .05), bottom=2.6, size=(1.8, 1.4), top=(.5, .5), top_z=3.0, duct_r=.2)

# film and still cameras of the shop (diagnostic): wide views of the cell with the robots, from outside the screens
# (cell_high looks over the screens' gap at the front-right corner and under the canopy; robots and cell_left stand
# inside the screens)
FILM_CAMERAS = {
    "cell_high": dict(eye=(2.4, -2.6, 3.8), target=(0., .05, .95), hfov=40., role="diagnostic"),
    "robots": dict(eye=(1.2, -.95, 1.7), target=(-.05, .1, 1.0), hfov=62., role="diagnostic"),
    "cell_left": dict(eye=(-1.05, -.9, 1.6), target=(.1, .05, 1.0), hfov=60., role="diagnostic"),
    "shop": dict(eye=(5.0, -4.0, 3.6), target=(-.4, .6, 1.0), hfov=72., role="diagnostic"),
}

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


# ================================================================================================ textures (PIL)
def _font(path, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _tex(name, size, draw):
    """Generated texture `name`.png (cached): draw(img, ImageDraw) paints it."""
    from PIL import Image, ImageDraw
    p = TEX/f"{name}.png"
    if p.is_file():
        return p
    TEX.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, (255, 255, 255))
    draw(im, ImageDraw.Draw(im))
    im.save(p)
    return p


def _text_block(d, xy, lines, font, fill, spacing=1.25):
    x, y = xy
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill)
        y += int(font.size*spacing)


def sign(name, word, lines, w=900, h=600, symbol=None):
    """ANSI Z535.4-style safety sign: signal word panel (DANGER red, WARNING orange, CAUTION yellow, NOTICE blue,
    SAFETY green) above a black message on white."""
    panel = {"DANGER": ((200, 16, 46), (255, 255, 255)), "WARNING": ((255, 121, 0), (0, 0, 0)),
             "CAUTION": ((255, 210, 0), (0, 0, 0)), "NOTICE": ((0, 82, 155), (255, 255, 255)),
             "SAFETY": ((0, 135, 81), (255, 255, 255)), "FIRE": ((200, 16, 46), (255, 255, 255))}[word]

    def draw(im, d):
        d.rectangle([0, 0, w - 1, h - 1], fill=(250, 250, 248), outline=(20, 20, 20), width=8)
        d.rectangle([8, 8, w - 9, int(h*.28)], fill=panel[0])
        f = _font(FONT, int(h*.17))
        label = word if word != "FIRE" else "FIRE EQUIPMENT"
        tw = d.textlength(label, font=f)
        tri = int(h*.2)
        x0 = (w - tw - (tri + 20 if word in ("DANGER", "WARNING", "CAUTION") else 0))/2
        if word in ("DANGER", "WARNING", "CAUTION"):
            cx, cy = x0 + tri/2, int(h*.145)
            d.polygon([(cx, cy - tri*.45), (cx - tri*.5, cy + tri*.42), (cx + tri*.5, cy + tri*.42)], fill=panel[1])
            d.text((cx - tri*.06, cy - tri*.3), "!", font=_font(FONT, int(tri*.6)), fill=panel[0])
            x0 += tri + 20
        d.text((x0, int(h*.045)), label, font=f, fill=panel[1])
        fb = _font(FONT, int(h*.085))
        _text_block(d, (40, int(h*.34)), lines, fb, (15, 15, 15))
    return _tex(name, (w, h), draw)


def exit_sign():
    def draw(im, d):
        d.rectangle([0, 0, 599, 239], fill=(0, 140, 70))
        f = _font(FONT, 150)
        d.text(((600 - d.textlength("EXIT", font=f))/2, 30), "EXIT", font=f, fill=(245, 255, 245))
    return _tex("exit", (600, 240), draw)


def source_front():
    """Front panel of the power source (fictional brand): meters, process selector, output studs' labels."""
    def draw(im, d):
        W, H = 452*2, 691*2
        d.rectangle([0, 0, W, H], fill=(64, 69, 75))
        d.rectangle([30, 30, W - 30, 420], fill=(38, 40, 44), outline=(20, 20, 22), width=6)
        d.text((60, 55), "FERRUM ARC", font=_font(FONT, 64), fill=(242, 166, 30))
        d.text((60, 135), "MULTIPROCESS 350 CV/CC", font=_font(FONT, 34), fill=(225, 225, 225))
        for k, (lab, val) in enumerate((("WFS m/min", "12.7"), ("VOLTS", "29.0"))):
            x = 60 + k*410
            d.rectangle([x, 200, x + 360, 330], fill=(12, 12, 12))
            d.text((x + 20, 205), val, font=_font(FONT_M, 110), fill=(255, 60, 30))
            d.text((x + 5, 340), lab, font=_font(FONT_R, 30), fill=(220, 220, 220))
        for k in range(3):
            cx = 140 + k*300
            d.ellipse([cx - 70, 470, cx + 70, 610], fill=(25, 25, 27))
            d.text((cx - 60, 625), ("PROCESS", "ARC CTRL", "INDUCT.")[k], font=_font(FONT_R, 28), fill=(225, 225, 225))
        d.rectangle([60, 720, W - 60, 1000], fill=(52, 56, 61))
        for k in range(14):
            d.rectangle([90, 740 + k*18, W - 90, 748 + k*18], fill=(20, 21, 23))
        d.text((120, 1050), "+", font=_font(FONT, 90), fill=(230, 40, 30))
        d.text((W - 190, 1050), "−", font=_font(FONT, 90), fill=(230, 230, 230))
        d.text((W/2 - 150, 1290), "380-575 V  3Φ  50/60 Hz", font=_font(FONT_R, 30), fill=(200, 200, 200))
    return _tex("source_front", (452*2, 691*2), draw)


def feeder_front():
    def draw(im, d):
        W, H = 414*2, 351*2
        d.rectangle([0, 0, W, H], fill=(64, 69, 75))
        d.rectangle([24, 24, W - 24, 260], fill=(38, 40, 44))
        d.text((50, 40), "FERRUM ARC  WF-4R", font=_font(FONT, 46), fill=(242, 166, 30))
        d.text((50, 110), "ROBOTIC WIRE FEEDER  4-ROLL", font=_font(FONT_R, 30), fill=(220, 220, 220))
        d.rectangle([50, 160, 360, 240], fill=(12, 12, 12))
        d.text((66, 162), "12.7", font=_font(FONT_M, 64), fill=(255, 60, 30))
        d.text((400, 180), "INCH  /  PURGE", font=_font(FONT_R, 30), fill=(220, 220, 220))
        d.rectangle([24, 300, W - 24, H - 24], fill=(52, 56, 61))
        d.text((60, 560), "GUN", font=_font(FONT, 40), fill=(230, 230, 230))
    return _tex("feeder_front", (414*2, 351*2), draw)


def louvres():
    def draw(im, d):
        d.rectangle([0, 0, 512, 512], fill=(64, 69, 75))
        for k in range(18):
            d.rectangle([40, 40 + k*24, 472, 52 + k*24], fill=(22, 23, 25))
    return _tex("louvres", (512, 512), draw)


def spool_box_label():
    def draw(im, d):
        d.rectangle([0, 0, 800, 300], fill=(178, 148, 104))
        d.rectangle([20, 20, 780, 280], fill=(240, 236, 226))
        d.text((40, 30), "FERRUM ARC  ER70S-6", font=_font(FONT, 52), fill=(20, 20, 20))
        d.text((40, 100), "AWS A5.18  ·  0.035 in (0.9 mm)  ·  33 lb", font=_font(FONT_R, 36), fill=(20, 20, 20))
        d.text((40, 160), "MIG WIRE  ·  COPPER COATED", font=_font(FONT_R, 36), fill=(20, 20, 20))
        for k in range(40):
            d.rectangle([420 + k*8, 225, 423 + k*8 + (k % 3), 268], fill=(10, 10, 10))
    return _tex("spool_label", (800, 300), draw)


def gas_tag():
    def draw(im, d):
        d.rectangle([0, 0, 300, 420], fill=(250, 250, 245))
        d.rectangle([0, 0, 300, 110], fill=(0, 150, 70))
        d.text((24, 22), "Ar 90%", font=_font(FONT, 54), fill=(255, 255, 255))
        d.text((24, 130), "CO₂ 10%", font=_font(FONT, 50), fill=(20, 20, 20))
        d.text((24, 210), "SHIELDING", font=_font(FONT_R, 34), fill=(20, 20, 20))
        d.text((24, 255), "GAS MIX", font=_font(FONT_R, 34), fill=(20, 20, 20))
        d.text((24, 330), "FULL / IN USE", font=_font(FONT, 30), fill=(200, 16, 46))
    return _tex("gas_tag", (300, 420), draw)


def wps_sheet():
    """The cell's procedure sheet: the flagship procedure this environment runs (welding/params.py SCHEDULES)."""
    from .params import SCHEDULES
    s = SCHEDULES.value["spray_0p9_90-10"]

    def draw(im, d):
        W, H = 850, 1100
        d.rectangle([0, 0, W, H], fill=(252, 252, 248))
        d.text((50, 40), "WELDING PROCEDURE SPECIFICATION", font=_font(FONT, 36), fill=(15, 15, 15))
        d.text((50, 95), "WPS-A2   GMAW   cell 1 (robot)", font=_font(FONT_R, 30), fill=(15, 15, 15))
        rows = [("Joint", "T-joint, 1/4 in fillet, 2F"), ("Base metal", "A36  10 mm plate / 8 mm bracket"),
                ("Filler", "ER70S-6  0.035 in (0.9 mm)"), ("Gas", f"{s['gas']}  17-21 L/min"),
                ("Transfer", "spray"), ("Wire feed", f"{s['wfs_m_min']:.1f} m/min (500 in/min)"),
                ("Voltage", f"{s['voltage_v']:.0f} V"), ("CTWD", f"{s['ctwd_m']*1e3:.0f} mm"),
                ("Travel", "14 in/min (5.9 mm/s)"), ("Work / travel", "45 deg / 7.5 deg push"),
                ("Tacks", "2 x 20 mm, ends"), ("Inspection", "visual, AWS D1.1 Table 8.1")]
        y = 170
        for a, b in rows:
            d.line([50, y - 8, W - 50, y - 8], fill=(150, 150, 150), width=2)
            d.text((60, y), a, font=_font(FONT, 28), fill=(25, 25, 25))
            d.text((330, y), b, font=_font(FONT_R, 28), fill=(25, 25, 25))
            y += 62
        d.text((60, y + 30), "Approved: ____________   Rev. C", font=_font(FONT_R, 26), fill=(60, 60, 60))
    return _tex("wps", (850, 1100), draw)


def hazard_tape():
    def draw(im, d):
        d.rectangle([0, 0, 512, 64], fill=(250, 200, 0))
        for k in range(-2, 12):
            x = k*64
            d.polygon([(x, 64), (x + 32, 64), (x + 96, 0), (x + 64, 0)], fill=(20, 20, 20))
    return _tex("hazard", (512, 64), draw)


def disconnect_label():
    def draw(im, d):
        d.rectangle([0, 0, 400, 200], fill=(250, 250, 245))
        d.text((20, 20), "WELDER 1", font=_font(FONT, 48), fill=(15, 15, 15))
        d.text((20, 90), "480 V 3Φ  60 A", font=_font(FONT_R, 36), fill=(15, 15, 15))
        d.text((20, 140), "LOCK OUT BEFORE SERVICE", font=_font(FONT_R, 24), fill=(200, 16, 46))
    return _tex("disconnect", (400, 200), draw)


# ================================================================================================ geometry helpers
def quad(A, path, origin, u, v, mat, flip=False):
    """A textured rectangle: corners origin, origin+u, origin+u+v, origin+v; UV (0,0) at origin, (1,1) opposite."""
    from pxr import UsdGeom, Sdf, Vt
    o, u, v = (np.asarray(a, float) for a in (origin, u, v))
    P = np.array([o, o + u, o + u + v, o + v], np.float32)
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    m = UsdGeom.Mesh.Define(A.stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P))
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 3, 2, 1] if flip else [0, 1, 2, 3])
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.tile((-n if flip else n).astype(np.float32), (4, 1))))
    m.SetNormalsInterpolation("vertex")
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)))
    A.bind(m, mat)
    return m


def decal(A, name, png, rough=.5, metal=0., emissive_nits=0.):
    """Material showing one generated image once over a quad (UV 0..1); emissive images glow (exit sign)."""
    if name in A.mats:
        return A.mats[name]
    kw = {}
    if emissive_nits:
        kw = dict(emissive=(1., 1., 1.), emissive_nits=emissive_nits)
    m = A.pbr(name, (255, 255, 255), rough=rough, metal=metal, diff=png, tile_m=1., **kw)
    if emissive_nits:
        from pxr import Sdf, UsdShade
        sh = UsdShade.Shader(A.stage.GetPrimAtPath(str(m.GetPath()) + "/Shader"))
        sh.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(png).resolve())))
    return m


def tint(A, mat, srgb):
    """Multiply a textured OmniPBR material's diffuse texture by a colour (paint over a texture's pattern)."""
    from pxr import Gf, Sdf, UsdShade
    from .render import srgb_to_linear
    sh = UsdShade.Shader(A.stage.GetPrimAtPath(str(mat.GetPath()) + "/Shader"))
    sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, srgb_to_linear(srgb))))
    return mat


def _rects_minus(s0, s1, z0, z1, holes):
    """Rectangles covering [s0,s1] x [z0,z1] minus the holes ((a,b),(c,d))."""
    cuts = sorted({s0, s1, *[v for (a, b), _ in holes for v in (a, b) if s0 < v < s1]})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = (a + b)/2
        zs = sorted([(c, d) for (ha, hb), (c, d) in holes if ha < m < hb])
        z = z0
        for c, d in zs:
            if c > z:
                out.append((a, b, z, c))
            z = max(z, d)
        if z < z1:
            out.append((a, b, z, z1))
    return out


def _ibeam(A, path, p0, p1, depth, width, mat, up=(0., 0., 1.), web=.008, flange=.012):
    """An I-section member from p0 to p1 (web along `up`)."""
    p0, p1, up = (np.asarray(a, float) for a in (p0, p1, up))
    L = np.linalg.norm(p1 - p0)
    ax = (p1 - p0)/L
    side = np.cross(up, ax)
    side /= np.linalg.norm(side)
    up = np.cross(ax, side)
    c = (p0 + p1)/2
    R = np.stack([ax, side, up], 1)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    A.box(path + "/web", c, (L/2, web/2, depth/2 - flange), mat, quat_xyzw=q)
    for k, s in enumerate((-1, 1)):
        A.box(path + f"/flange_{k}", c + up*s*(depth/2 - flange/2), (L/2, width/2, flange/2), mat, quat_xyzw=q)


# ================================================================================================ the building
def building(A, VA, stage):
    from pxr import Gf, UsdGeom, UsdLux, Sdf
    (x0, x1), (y0, y1), H, t, dado = ROOM["x"], ROOM["y"], ROOM["height"], ROOM["wall"], ROOM["dado"]
    floor_t, block_t = VA.texture("hangar_concrete_floor"), VA.texture("concrete_block_wall")
    clad_t, deck_t = VA.texture("box_profile_metal_sheet"), VA.texture("corrugated_iron_02")
    floor = A.pbr("floor", (150, 148, 142), rough=.8, diff=floor_t["diff"], rough_tex=floor_t["rough"],
                  normal=floor_t["nor_gl"], tile_m=4.)
    block = tint(A, A.pbr("wall_block", (200, 200, 196), rough=.85, diff=block_t["diff"], rough_tex=block_t["rough"],
                          normal=block_t["nor_gl"], tile_m=2.), (214, 214, 206))
    # the cladding texture is painted red: its relief (normal and roughness maps) under a light grey paint
    clad = A.pbr("wall_cladding", (196, 200, 202), rough=.5, metal=.3, rough_tex=clad_t["rough"], normal=clad_t["nor_gl"],
                 tile_m=1.5)
    deck = tint(A, A.pbr("roof_deck", (200, 200, 200), rough=.5, metal=.5, diff=deck_t["diff"],
                         rough_tex=deck_t["rough"], normal=deck_t["nor_gl"], tile_m=1.5), (205, 208, 210))
    steel = A.pbr("frame_steel", (112, 118, 124), rough=.55, metal=.5)
    A.box("/World/Shop/floor", ((x0 + x1)/2, (y0 + y1)/2, -.01), ((x1 - x0)/2 + t, (y1 - y0)/2 + t, .01), floor, uv=True)
    walls = {"back": ("x", y1 + t/2, (x0 - t, x1 + t)), "front": ("x", y0 - t/2, (x0 - t, x1 + t)),
             "left": ("y", x0 - t/2, (y0, y1)), "right": ("y", x1 + t/2, (y0, y1))}
    for name, (along, c, (s0, s1)) in walls.items():
        holes = [(s, z) for s, z, _ in OPENINGS.get(name, [])]
        for k, (a, b, za, zb) in enumerate(_rects_minus(s0, s1, 0., H, holes)):
            for part, (lo, hi, mat) in enumerate(((za, min(zb, dado), block), (max(za, dado), zb, clad))):
                if hi - lo < 1e-6:
                    continue
                if along == "x":
                    A.box(f"/World/Shop/wall_{name}_{k}_{part}", ((a + b)/2, c, (lo + hi)/2), ((b - a)/2, t/2, (hi - lo)/2),
                          mat, uv=True)
                else:
                    A.box(f"/World/Shop/wall_{name}_{k}_{part}", (c, (a + b)/2, (lo + hi)/2), (t/2, (b - a)/2, (hi - lo)/2),
                          mat, uv=True)
    # roof deck, portal frames, purlins
    A.box("/World/Shop/roof", ((x0 + x1)/2, (y0 + y1)/2, H + .03), ((x1 - x0)/2 + t, (y1 - y0)/2 + t, .03), deck, uv=True)
    for i, x in enumerate(FRAMES_X):
        for j, y in enumerate((y0 + .15, y1 - .15)):
            _ibeam(A, f"/World/Shop/column_{i}{j}", (x, y, 0.), (x, y, H - .45), .3, .3, steel, up=(0., 1., 0.))
        _ibeam(A, f"/World/Shop/rafter_{i}", (x, y0, H - .25), (x, y1, H - .25), .45, .18, steel)
    for j, y in enumerate(np.arange(y0 + .6, y1, 1.5)):
        A.box(f"/World/Shop/purlin_{j}", ((x0 + x1)/2, y, H - .08), ((x1 - x0)/2, .03, .08), steel)
    # windows: aluminium mullions and daylight panes (emissive: the sky outside is not modelled) [authored]
    alu = A.pbr("window_frame", (180, 182, 184), rough=.4, metal=.8)
    day = A.pbr("daylight_pane", (235, 240, 245), rough=.1, emissive=(.88, .92, 1.), emissive_nits=2500.)
    for name, ops in OPENINGS.items():
        for k, ((a, b), (za, zb), kind) in enumerate(ops):
            if kind != "window":
                continue
            n = max(1, int(round((b - a)/1.2)))
            along, c, _ = walls[name]
            for e in range(n + 1):
                s = a + (b - a)*e/n
                p = (s, c, (za + zb)/2) if along == "x" else (c, s, (za + zb)/2)
                h = (.03, t/2, (zb - za)/2) if along == "x" else (t/2, .03, (zb - za)/2)
                A.box(f"/World/Shop/window_{name}_{k}_m{e}", p, h, alu)
            for e, z in enumerate((za, zb)):
                p = ((a + b)/2, c, z) if along == "x" else (c, (a + b)/2, z)
                h = ((b - a)/2, t/2, .03) if along == "x" else (t/2, (b - a)/2, .03)
                A.box(f"/World/Shop/window_{name}_{k}_t{e}", p, h, alu)
            p = ((a + b)/2, c, (za + zb)/2) if along == "x" else (c, (a + b)/2, (za + zb)/2)
            h = ((b - a)/2, .005, (zb - za)/2) if along == "x" else (.005, (b - a)/2, (zb - za)/2)
            A.box(f"/World/Shop/window_{name}_{k}_glass", p, h, day)
    # roll-up door (left wall): slatted shutter, coil housing, guides; bollards
    shutter_t = VA.texture("painted_metal_shutter")
    shutter = tint(A, A.pbr("shutter", (200, 200, 200), rough=.5, metal=.5, diff=shutter_t["diff"],
                            rough_tex=shutter_t["rough"], normal=shutter_t["nor_gl"], tile_m=1.), (196, 200, 204))
    yellow = A.pbr("safety_yellow", (240, 190, 0), rough=.45, metal=.2)
    (ya, yb), (za, zb), _ = OPENINGS["left"][0]
    A.box("/World/Shop/rollup/shutter", (x0 - t*.6, (ya + yb)/2, zb/2), (.02, (yb - ya)/2, zb/2), shutter, uv=True)
    A.box("/World/Shop/rollup/housing", (x0 + .2, (ya + yb)/2, zb + .28), (.2, (yb - ya)/2 + .12, .28), steel)
    for k, y in enumerate((ya - .05, yb + .05)):
        A.box(f"/World/Shop/rollup/guide_{k}", (x0 + .04, y, zb/2), (.04, .05, zb/2), steel)
        A.cyl(f"/World/Shop/rollup/bollard_{k}", (x0 + .45, y + (-.2 if k == 0 else .2), 0.),
              (x0 + .45, y + (-.2 if k == 0 else .2), 1.1), .057, yellow)
    A.box("/World/Shop/rollup/motor", (x0 + .18, yb + .3, zb + .25), (.12, .14, .16), A.pbr("motor", (40, 60, 90), rough=.5))
    # personnel door and exit sign (front wall)
    (xa, xb), (za, zb), _ = OPENINGS["front"][0]
    leaf = A.pbr("door_leaf", (120, 130, 140), rough=.45, metal=.3)
    A.box("/World/Shop/door/leaf", ((xa + xb)/2, y0 - .06, zb/2), ((xb - xa)/2, .025, zb/2), leaf)
    A.box("/World/Shop/door/push_bar", ((xa + xb)/2, y0 - .02, 1.0), ((xb - xa)/2 - .08, .02, .025), A.pbr("chrome", (200, 200, 205), rough=.2, metal=1.))
    quad(A, "/World/Shop/door/exit", ((xa + xb)/2 + .2, y0 + .012, zb + .15), (-.4, 0., 0.), (0., 0., .16),
         decal(A, "exit_sign", exit_sign(), emissive_nits=150.))
    # floor markings: walkway lines along the aisle and hazard tape around the cell [authored]
    line_m = A.pbr("floor_paint_yellow", (230, 180, 20), rough=.55)
    for k, y in enumerate((-2.15, -3.15)):
        A.box(f"/World/Shop/floor_line_{k}", ((x0 + x1)/2 + .3, y, .0006), ((x1 - x0)/2 - .3, .04, .0006), line_m)
    tape = A.pbr("hazard_tape", (255, 255, 255), rough=.6, diff=hazard_tape(), tile_m=.3)
    cx0, cx1, cy0, cy1 = -1.45, 1.8, -1.4, .95
    for k, (p, h) in enumerate(((((cx0 + cx1)/2, cy0, .0007), ((cx1 - cx0)/2, .04, .0007)),
                                (((cx0 + cx1)/2, cy1, .0007), ((cx1 - cx0)/2, .04, .0007)),
                                ((cx0, (cy0 + cy1)/2, .0007), (.04, (cy1 - cy0)/2, .0007)),
                                ((cx1, (cy0 + cy1)/2, .0007), (.04, (cy1 - cy0)/2, .0007)))):
        A.box(f"/World/Shop/cell_tape_{k}", p, h, tape, uv=True)
    # high-bay pendants: the CC0 lamp model and a disk light under each shade [authored: ~ 300-500 lx on the bench]
    for k, (x, y) in enumerate(LAMPS):
        A.reference(f"/World/Shop/lamp_{k}", VA.model("hanging_industrial_lamp"), (x, y, LAMP_Z), 0., ground=False)
        L = UsdLux.DiskLight.Define(stage, f"/World/Lights/lamp_{k}")
        UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(x, y, LAMP_Z - 1.30))
        L.CreateRadiusAttr(.2)
        L.CreateIntensityAttr(LAMP_INTENSITY)
        L.CreateColorTemperatureAttr(4500.)
        L.CreateEnableColorTemperatureAttr(True)


LAMP_INTENSITY = 30000.
CANOPY_INTENSITY = 4000.


# ================================================================================================ the welding equipment
def equipment(A):
    """Power source on its running gear, wire feeder on top, gas cylinder in the rack: the drawn shapes are the static
    colliders' boxes and cylinder (layout.source_boxes)."""
    boxes = {n: (np.asarray(c), np.asarray(h)) for n, c, h in LY.source_boxes()}
    case = A.pbr("source_paint", (66, 71, 77), rough=.45, metal=.3)
    dark = A.pbr("source_dark", (30, 31, 34), rough=.5, metal=.2)
    orange = A.pbr("brand_orange", (242, 150, 30), rough=.45, metal=.1)
    rubber = A.pbr("tyre", (24, 24, 25), rough=.85)
    zinc = A.pbr("zinc", (168, 170, 172), rough=.35, metal=.9)
    S = "/World/Equipment"
    # running gear: deck, rails, wheels (the collider is the deck's full box down to the floor)
    c, h = boxes["source_gear"]
    A.box(S + "/gear/deck", (c[0], c[1], c[2]*2 - .015), (h[0], h[1], .015), dark)
    for k, s in enumerate((-1, 1)):
        A.box(f"{S}/gear/rail_{k}", (c[0] + s*(h[0] - .02), c[1], c[2]*2 - .05), (.02, h[1], .03), dark)
    for k, (dx, dy, r) in enumerate(((-.18, -h[1] + .08, .0625), (.18, -h[1] + .08, .0625),
                                     (-.19, h[1] - .12, .125), (.19, h[1] - .12, .125))):
        p = np.array([c[0] + dx, c[1] + dy, r])
        A.cyl(f"{S}/gear/wheel_{k}", p - (.03, 0, 0), p + (.03, 0, 0), r, rubber, sides=28)
        A.cyl(f"{S}/gear/hub_{k}", p - (.032, 0, 0), p + (.032, 0, 0), r*.45, zinc, sides=20)
    # power source case, louvres, front panel, studs and knobs, handles
    c, h = boxes["source_case"]
    A.box(S + "/source/case", c, h, case)
    A.box(S + "/source/stripe_r", (c[0] + h[0] + .0008, c[1], c[2] + h[2] - .05), (.0008, h[1] - .02, .012), orange)
    A.box(S + "/source/stripe_l", (c[0] - h[0] - .0008, c[1], c[2] + h[2] - .05), (.0008, h[1] - .02, .012), orange)
    lv = decal(A, "louvres", louvres(), rough=.55, metal=.3)
    for k, s in enumerate((-1, 1)):
        x = c[0] + s*(h[0] + .0012)
        quad(A, f"{S}/source/louvre_{k}", (x, c[1] - .25, c[2] - .25), (0, .5, 0), (0, 0, .4), lv, flip=(s < 0))
    front = c[1] - h[1]
    quad(A, S + "/source/front", (c[0] - h[0], front - .001, c[2] - h[2]), (2*h[0], 0, 0), (0, 0, 2*h[2]),
         decal(A, "source_front", source_front(), rough=.45, metal=.2))
    zs = c[2] - h[2]
    for k, dx in enumerate((-.145, .145)):
        A.cyl(f"{S}/source/stud_{k}", (c[0] + dx, front, zs + .14), (c[0] + dx, front - .035, zs + .14), .022,
              A.pbr("stud_red" if k == 0 else "stud_black", (180, 30, 25) if k == 0 else (25, 25, 25), rough=.4))
    for k in range(3):                                   # the knobs over the panel art's dials (px 140 + 300 k of 904)
        x = c[0] - h[0] + (140 + 300*k)/904.*2*h[0]
        z = zs + 2*h[2]*(1 - 540/1382.)
        A.cyl(f"{S}/source/knob_{k}", (x, front, z), (x, front - .025, z), .03, dark)
    for k, s in enumerate((-1, 1)):
        A.cyl(f"{S}/source/handle_{k}", (c[0] + s*(h[0] - .03), front + .03, c[2] + h[2] + .02),
              (c[0] + s*(h[0] - .03), front + .15, c[2] + h[2] + .02), .012, dark)
    # wire feeder: case, front panel, top handle, gun connector
    c, h = boxes["wire_feeder"]
    A.box(S + "/feeder/case", c, h, case)
    ff = c[1] - h[1]
    quad(A, S + "/feeder/front", (c[0] - h[0], ff - .001, c[2] - h[2]), (2*h[0], 0, 0), (0, 0, 2*h[2]),
         decal(A, "feeder_front", feeder_front(), rough=.45, metal=.2))
    A.cyl(S + "/feeder/handle", (c[0], ff + .1, c[2] + h[2] + .04), (c[0], ff + 2*h[1] - .1, c[2] + h[2] + .04), .013, dark)
    for k, dy in enumerate((.1, 2*h[1] - .1)):
        A.cyl(f"{S}/feeder/handle_post_{k}", (c[0], ff + dy, c[2] + h[2]), (c[0], ff + dy, c[2] + h[2] + .045), .01, dark)
    A.box(S + "/feeder/door_seam", (c[0] + h[0] + .0008, c[1], c[2]), (.0008, h[1] - .03, h[2] - .03), dark)
    A.cyl(S + "/feeder/gun_connector", FEEDER_GUN_OUTLET + np.array([0, .04, 0]), FEEDER_GUN_OUTLET, .024, zinc)
    # gas cylinder: grey body, bright green shoulder (EN 1089-3: argon/CO2 mixtures), valve, regulator-flowmeter,
    # chain, the content tag
    c, h = boxes["gas_cylinder"]
    r, zb, zt = h[0], c[2] - h[2], c[2] + h[2]
    body = A.pbr("cylinder_paint", (128, 132, 132), rough=.4, metal=.2)
    green = A.pbr("cylinder_shoulder", (40, 160, 70), rough=.4, metal=.1)
    brass = A.pbr("brass", (190, 150, 70), rough=.3, metal=1.)
    A.cyl(S + "/cylinder/body", (c[0], c[1], zb), (c[0], c[1], zt - .12), r, body, sides=40)
    A.cyl(S + "/cylinder/shoulder", (c[0], c[1], zt - .12), (c[0], c[1], zt), r, green, sides=40, r1=.045)
    A.cyl(S + "/cylinder/neck", (c[0], c[1], zt), (c[0], c[1], zt + .03), .03, green)
    A.cyl(S + "/cylinder/valve", (c[0], c[1], zt + .03), (c[0], c[1], zt + .1), .022, brass)
    A.cyl(S + "/cylinder/handwheel", (c[0], c[1], zt + .1), (c[0], c[1], zt + .115), .035, dark)
    reg = np.array([c[0], c[1] - .07, zt + .07])
    A.cyl(S + "/cylinder/reg_stem", (c[0], c[1], zt + .07), reg, .012, brass)
    A.cyl(S + "/cylinder/regulator", reg, reg + (0, -.06, 0), .03, brass)
    A.cyl(S + "/cylinder/gauge", reg + (0, -.03, .03), reg + (0, -.03, .075), .028, A.pbr("gauge_face", (235, 235, 230), rough=.3))
    A.cyl(S + "/cylinder/flowtube", reg + (0, -.06, 0), reg + (0, -.06, .14), .012, A.glass("flowtube", (235, 240, 240)))
    for k, z in enumerate((zb + .35, zb + 1.0)):
        A.cyl(f"{S}/cylinder/chain_{k}", (c[0], c[1], z), (c[0], c[1], z + .012), r + .006, zinc, sides=40)
    quad(A, S + "/cylinder/tag", (c[0] - .045, c[1] - r - .004, zt - .30), (.09, 0, 0), (0, 0, .126),
         decal(A, "gas_tag", gas_tag()))
    # rack posts behind the cylinder
    for k, s in enumerate((-1, 1)):
        A.box(f"{S}/rack_post_{k}", (c[0] + s*(r + .02), c[1] + r + .01, (zb + 1.15)/2), (.012, .012, (1.15 - zb)/2), dark)
    A.box(S + "/rack_rail", (c[0], c[1] + r + .01, 1.15), (r + .035, .012, .012), dark)
    # the UR teach pendant hangs on the source's far side (out of the arms' way) [UR e-Series pendant 300 x 231 x 50 mm,
    # estimate]
    cs, hs = boxes["source_case"]
    A.box(S + "/pendant", (cs[0] + hs[0] + .03, cs[1] - .1, cs[2] + .05), (.025, .15, .115),
          A.pbr("pendant", (210, 212, 214), rough=.4))
    # robot controllers on the table's lower shelf: UR e-Series control box [read, UR datasheet: 460 x 449 x 254 mm,
    # 12 kg]; FR3 controller, a 19 in rack unit [estimate: 483 x 89 x 350 mm]
    T = LY.TABLE
    A.box(S + "/ur_control_box", (.25, .02, .186 + .2245), (.23, .127, .2245), A.pbr("ur_box", (205, 208, 210), rough=.45, metal=.2))
    A.box(S + "/fr3_controller", (-.25, .0, .186 + .0445), (.2415, .175, .0445), A.pbr("fr3_ctrl", (40, 42, 46), rough=.45, metal=.3))


FEEDER_GUN_OUTLET = None       # set below from the feeder box (world)


def _feeder_outlet():
    for n, c, h in LY.source_boxes():
        if n == "wire_feeder":
            return np.array([c[0] - h[0] + .09, c[1] - h[1] - .04, c[2] - h[2] + .07])


FEEDER_GUN_OUTLET = _feeder_outlet()
SOURCE_WORK_STUD = (lambda c: np.array([c[0] + .145, c[1] - c[4] - .035, c[2] - c[5] + .14]))(
    [v for n, c, h in LY.source_boxes() if n == "source_case" for v in (*c, *h)])


def static_cables(A, stage):
    """Cables that do not move: gas hose regulator -> feeder, primary power to the wall disconnect, robot cables from
    the controllers to the bases."""
    from .render import _curve, _sag
    cable = A.pbr("cable_rubber", (18, 18, 19), rough=.7)
    gas_h = A.pbr("gas_hose", (40, 110, 60), rough=.6)
    boxes = {n: (np.asarray(c), np.asarray(h)) for n, c, h in LY.source_boxes()}
    cc, ch = boxes["gas_cylinder"]
    fc, fh = boxes["wire_feeder"]
    reg = np.array([cc[0], cc[1] - .13, cc[2] + ch[2] + .07])
    _curve(stage, "/World/Cables/gas", _sag(reg, np.array([fc[0] + .1, fc[1] + fh[1], fc[2]]), .2), .010, gas_h)
    sc, sh = boxes["source_case"]
    back = np.array([sc[0] + .1, sc[1] + sh[1], sc[2] - .1])
    P = np.array([back, back + (0, .05, -.3), (sc[0] + .2, sc[1] + sh[1] + .2, .02), (1.9, 2.7, .02), (1.9, 3.0, .03),
                  (1.9, 3.05, 1.2)])
    _curve(stage, "/World/Cables/primary", P, .022, cable)
    for name, start, base in (("ur", (.25, .147, .35), LY.ROBOT_BASES["right"]["pos"]),
                              ("fr3", (-.25, .175, .23), LY.ROBOT_BASES["left"]["pos"])):
        b = np.asarray(base)
        P = np.array([start, (start[0], .3, start[2] + .05), (b[0], .37, .55), (b[0], .41, .84), b + (0, .11, .03)])
        _curve(stage, f"/World/Cables/robot_{name}", P, .014, cable)


# ================================================================================================ the cell
def cell(A, stage):
    """Welding screens, camera mounts (colliders in scene.py) and the extraction canopy over the cell."""
    from pxr import Gf, UsdGeom, UsdLux
    vinyl = A.glass("screen_vinyl", (255, 120, 20), ior=1.4, thin=True)
    frame_m = A.pbr("screen_frame", (210, 170, 30), rough=.5, metal=.2)
    for k, S in enumerate(LY.SCREENS):
        (sx_, sy_), yaw, w_, h_ = S["center"], math.radians(S["yaw_deg"]), S["width"], S["height"]
        q = LY.yaw_quat(S["yaw_deg"])
        A.box(f"/World/Screens/s{k}/vinyl", (sx_, sy_, .25 + h_/2), (w_/2, .001, h_/2), vinyl, quat_xyzw=q)
        d = np.array([math.cos(yaw), math.sin(yaw), 0.])
        for e, s_ in enumerate((-1, 1)):
            p = np.array([sx_, sy_, 0.]) + s_*d*w_/2
            A.cyl(f"/World/Screens/s{k}/post_{e}", p, p + [0, 0, .25 + h_], .016, frame_m)
            A.cyl(f"/World/Screens/s{k}/foot_{e}", p - [-d[1]*.3, d[0]*.3, 0], p + [-d[1]*.3, d[0]*.3, 0], .016, frame_m)
        a, b = np.array([sx_, sy_, .25 + h_]) - d*w_/2, np.array([sx_, sy_, .25 + h_]) + d*w_/2
        A.cyl(f"/World/Screens/s{k}/top", a, b, .016, frame_m)
        A.cyl(f"/World/Screens/s{k}/bottom", a - [0, 0, h_], b - [0, 0, h_], .012, frame_m)
    # camera mounts: anodised extrusions and black camera bodies (the collider boxes of layout.CAMERA_MOUNTS)
    alu = A.pbr("extrusion", (160, 164, 168), rough=.35, metal=.85)
    cam_m = A.pbr("camera_body", (25, 26, 28), rough=.4, metal=.3)
    lens = A.glass("lens", (40, 50, 60))
    for name, c, h in LY.CAMERA_MOUNTS:
        A.box(f"/World/CameraMounts/{name}", c, h, cam_m if "camera" in name else alu)
    # the strut across the canopy's mouth that carries the bench camera's drop rod
    rod = [c for n, c, h in LY.CAMERA_MOUNTS if n == "bench_drop_rod"][0]
    A.box("/World/CameraMounts/canopy_strut", (CANOPY["center"][0], rod[1], LY.CANOPY_STRUT_Z + .02),
          (CANOPY["size"][0]/2, .02, .02), alu)
    for cam, body in (("bench", "bench_camera_body"), ("side", "side_camera_body")):
        c = np.asarray(LY.CAMERAS[cam]["eye"], float)
        d = np.asarray(LY.CAMERAS[cam]["target"], float) - c
        d /= np.linalg.norm(d)
        A.cyl(f"/World/CameraMounts/{cam}_lens", c - d*.03, c - d*.004, .016, lens)      # ends 4 mm behind the eye
    # extraction canopy over the cell (galvanised sheet, duct to the roof), above both arms' reach (UR10e base 0.85 m +
    # 1.30 m reach + 0.35 m gun < 2.6 m) [authored]; an LED panel in it lights the bench
    galv = A.pbr("galvanised", (172, 175, 178), rough=.45, metal=.9)
    C = CANOPY
    (cx, cy), (sx, sy), (tx, ty), zb, zt = C["center"], C["size"], C["top"], C["bottom"], C["top_z"]
    skirt = .2
    for k, (p, h) in enumerate((((cx, cy - sy/2, zb + skirt/2), (sx/2, .003, skirt/2)),
                                ((cx, cy + sy/2, zb + skirt/2), (sx/2, .003, skirt/2)),
                                ((cx - sx/2, cy, zb + skirt/2), (.003, sy/2, skirt/2)),
                                ((cx + sx/2, cy, zb + skirt/2), (.003, sy/2, skirt/2)))):
        A.box(f"/World/Canopy/skirt_{k}", p, h, galv)
    lo = np.array([[cx - sx/2, cy - sy/2, zb + skirt], [cx + sx/2, cy - sy/2, zb + skirt], [cx + sx/2, cy + sy/2, zb + skirt],
                   [cx - sx/2, cy + sy/2, zb + skirt]])
    hi = np.array([[cx - tx/2, cy - ty/2, zt], [cx + tx/2, cy - ty/2, zt], [cx + tx/2, cy + ty/2, zt], [cx - tx/2, cy + ty/2, zt]])
    V = np.concatenate([lo, hi])
    F = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]
    m = A.mesh("/World/Canopy/hood", V, F, galv)
    m.CreateDoubleSidedAttr(True)
    A.cyl("/World/Canopy/duct", (cx, cy, zt), (cx, cy, ROOM["height"]), C["duct_r"], galv, sides=32)
    r = UsdLux.RectLight.Define(stage, "/World/Lights/canopy")
    UsdGeom.Xformable(r).AddTranslateOp().Set(Gf.Vec3d(cx, cy, zb + skirt - .01))
    r.CreateWidthAttr(1.2)
    r.CreateHeightAttr(.8)
    r.CreateIntensityAttr(CANOPY_INTENSITY)
    r.CreateColorTemperatureAttr(5000.)
    r.CreateEnableColorTemperatureAttr(True)
    A.box("/World/Canopy/led_panel", (cx, cy, zb + skirt - .005), (.6, .4, .004),
          A.pbr("led_panel", (240, 240, 240), rough=.3, emissive=(1., 1., 1.), emissive_nits=900.))


# ================================================================================================ the shop's dressing
def _shelving(A, path, x, y, width, depth, levels, mat, board):
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        A.box(f"{path}/upright_{k}", (x + dx*(width/2 - .02), y + dy*(depth/2 - .02), levels[-1]/2 + .05),
              (.02, .02, levels[-1]/2 + .05), mat)
    for k, z in enumerate(levels):
        A.box(f"{path}/shelf_{k}", (x, y, z), (width/2, depth/2, .012), board)


def dressing(A, VA):
    (x0, x1), (y0, y1) = ROOM["x"], ROOM["y"]
    steel = A.pbr("frame_steel", (112, 118, 124), rough=.55, metal=.5)
    blue_rack = A.pbr("rack_blue", (30, 70, 140), rough=.5, metal=.3)
    orange_rack = A.pbr("rack_orange", (230, 110, 20), rough=.5, metal=.3)
    mill = A.pbr("stock_mill_scale", (70, 74, 80), rough=.6, metal=.5)
    rust_t = VA.texture("rusty_metal_02")
    rusty = tint(A, A.pbr("stock_rusty", (120, 90, 70), rough=.7, metal=.4, diff=rust_t["diff"], rough_tex=rust_t["rough"],
                          normal=rust_t["nor_gl"], tile_m=.8), (150, 140, 135))
    D = "/World/Dressing"
    # cantilever stock rack along the back wall: bars, tubes, angle and flat bar on the arms [authored]
    for k, x in enumerate((-3.95, -2.55)):
        _ibeam(A, f"{D}/stock_rack/upright_{k}", (x, y1 - .12, 0.), (x, y1 - .12, 2.4), .15, .15, blue_rack, up=(0., 1., 0.))
        for e, z in enumerate((.45, .95, 1.45, 1.95)):
            A.box(f"{D}/stock_rack/arm_{k}_{e}", (x, y1 - .45, z), (.03, .33, .03), blue_rack)
    rng = np.random.default_rng(3)
    xa, xb = -4.45, -1.95
    for e, z in enumerate((.48, .98, 1.48, 1.98)):
        y = y1 - .7
        kind = ("round", "tube", "angle", "flat")[e]
        for i in range(5 + e):
            mat = rusty if rng.random() < .35 else mill
            off = rng.uniform(-.06, .06)
            if kind == "round":
                r = rng.choice((.0125, .019, .025))
                A.cyl(f"{D}/stock/{kind}_{i}", (xa + off, y, z + r), (xb + off, y, z + r), r, mat, sides=16)
                y += 2*r + .01
            elif kind == "tube":
                s = rng.choice((.025, .038, .05))
                A.box(f"{D}/stock/{kind}_{i}", ((xa + xb)/2 + off, y + s, z + s), ((xb - xa)/2, s, s), mat)
                y += 2*s + .01
            elif kind == "angle":
                s = .05
                A.box(f"{D}/stock/{kind}_{i}_a", ((xa + xb)/2 + off, y + s/2, z + .003), ((xb - xa)/2, s/2, .003), mat)
                A.box(f"{D}/stock/{kind}_{i}_b", ((xa + xb)/2 + off, y + .003, z + s/2), ((xb - xa)/2, .003, s/2), mat)
                y += s + .015
            else:
                w_ = rng.choice((.05, .075, .1))
                A.box(f"{D}/stock/{kind}_{i}", ((xa + xb)/2 + off, y + w_/2, z + .005 + .012*(i % 2)),
                      ((xb - xa)/2, w_/2, .005), mat)
                y += w_*.6
    # consumables shelving (authored steel shelving, stocked) and the CC0 shelving beside it
    board = A.pbr("shelf_board", (90, 95, 100), rough=.55, metal=.4)
    _shelving(A, f"{D}/shelf_a", -1.15, y1 - .3, 1.2, .5, (.15, .6, 1.05, 1.5, 1.95), orange_rack, board)
    label = decal(A, "spool_label", spool_box_label(), rough=.7)
    cardboard = A.pbr("cardboard", (176, 140, 98), rough=.8)
    for lvl, z in enumerate((.162, .612, 1.062)):
        for i in range(3):
            x = -1.65 + i*.36
            for j in range(2 if lvl < 2 else 1):
                zc = z + .06 + j*.122
                A.box(f"{D}/shelf_a/spool_{lvl}_{i}_{j}", (x, y1 - .3, zc), (.165, .165, .06), cardboard)
                quad(A, f"{D}/shelf_a/spool_label_{lvl}_{i}_{j}", (x - .16, y1 - .466, zc - .056), (.32, 0, 0), (0, 0, .12),
                     label)
    A.reference(f"{D}/shelf_a/crate_0", VA.model("plastic_crate_01"), (-1.5, y1 - .3, 1.512), 90.)
    A.reference(f"{D}/shelf_a/crate_1", VA.model("plastic_crate_01"), (-1.12, y1 - .3, 1.512), 90.)
    A.reference(f"{D}/shelf_a/toolbox", VA.model("metal_toolbox"), (-.72, y1 - .3, 1.512), 0.)
    A.reference(f"{D}/shelf_a/box_0", VA.model("cardboard_box_01"), (-1.4, y1 - .3, 1.962), 90.)
    A.reference(f"{D}/shelf_a/box_1", VA.model("cardboard_box_01"), (-.9, y1 - .3, 1.962), 75.)
    for i in range(6):                                  # anti-spatter spray cans
        A.cyl(f"{D}/shelf_a/can_{i}", (-.85 + i*.07, y1 - .42, 1.062 + .012), (-.85 + i*.07, y1 - .42, 1.062 + .2), .033,
              A.pbr("spray_can", (30, 90, 170), rough=.35, metal=.6))
    A.reference(f"{D}/shelf_b", VA.model("steel_frame_shelves_01"), (.1, y1 - .28, 0.), 0.)
    for i in range(2):
        A.reference(f"{D}/shelf_b/crate_{i}", VA.model("plastic_crate_01"), (-.12 + i*.42, y1 - .28, 1.08), 90.)
    # the cell information board: procedure sheet and safety signs behind the robots
    A.box(f"{D}/board", (1.17, y1 - .02, 1.75), (.5, .01, .38), A.pbr("cork", (190, 160, 110), rough=.9))
    quad(A, f"{D}/wps", (.74, y1 - .032, 1.42), (.34, 0, 0), (0, 0, .44), decal(A, "wps", wps_sheet(), rough=.85))
    quad(A, f"{D}/sign_arc", (1.14, y1 - .032, 1.68), (.42, 0, 0), (0, 0, .28),
         decal(A, "sign_arc", sign("sign_arc", "WARNING", ["ARC RAYS can injure eyes", "and burn skin.", "Wear a welding helmet",
                                                       "and protective clothing."])))
    quad(A, f"{D}/sign_robot", (1.14, y1 - .032, 1.40), (.42, 0, 0), (0, 0, .24),
         decal(A, "sign_robot", sign("sign_robot", "CAUTION", ["ROBOT WELDING CELL", "Arms move without warning.",
                                                          "Keep out while the program runs."])))
    quad(A, f"{D}/sign_fumes", (-.45, y1 - .025, 2.35), (.6, 0, 0), (0, 0, .4),
         decal(A, "sign_fumes", sign("sign_fumes", "WARNING", ["FUMES AND GASES can be", "hazardous to your health.",
                                                          "Keep the extraction running."])))
    # electrical disconnect for the welder on the back wall, conduit to the roof
    grey = A.pbr("enclosure_grey", (190, 192, 190), rough=.5, metal=.3)
    A.box(f"{D}/disconnect/box", (1.9, y1 - .1, 1.45), (.2, .1, .3), grey)
    A.box(f"{D}/disconnect/handle", (2.12, y1 - .12, 1.5), (.02, .03, .12), A.pbr("handle_red", (190, 30, 25), rough=.5))
    quad(A, f"{D}/disconnect/label", (1.8, y1 - .201, 1.6), (.2, 0, 0), (0, 0, .1), decal(A, "disconnect", disconnect_label()))
    A.cyl(f"{D}/disconnect/conduit", (1.9, y1 - .08, 1.75), (1.9, y1 - .08, ROOM["height"]), .016, A.pbr("conduit", (175, 178, 180), rough=.4, metal=.9))
    A.reference(f"{D}/drawer_cabinet", VA.model("drawer_cabinet"), (3.3, y1 - .27, 0.), 0.)
    A.reference(f"{D}/storage_cart", VA.model("industrial_storage_cart"), (4.6, 1.9, 0.), 0.)
    # left wall: gas cylinder storage cage, oxy-fuel cart, extinguisher with its sign
    cage = A.pbr("cage", (60, 64, 68), rough=.5, metal=.6)
    cx_, cy0, cy1 = x0 + .32, .45, 2.0
    for k, y in enumerate(np.linspace(cy0, cy1, 4)):
        A.box(f"{D}/gas_cage/post_{k}", (cx_ + .28, y, .9), (.02, .02, .9), cage)
    for k, z in enumerate((.05, .95, 1.75)):
        A.box(f"{D}/gas_cage/rail_{k}", (cx_ + .28, (cy0 + cy1)/2, z), (.015, (cy1 - cy0)/2, .015), cage)
    colours = [((128, 132, 132), (40, 160, 70)), ((128, 132, 132), (40, 160, 70)), ((40, 90, 60), (40, 90, 60)),
               ((235, 235, 235), (30, 60, 140)), ((140, 30, 30), (140, 30, 30))]
    for k, (bc, sc) in enumerate(colours):
        y = cy0 + .15 + k*.3
        b = A.pbr(f"cyl_body_{k}", bc, rough=.4, metal=.2)
        s = A.pbr(f"cyl_sh_{k}", sc, rough=.4, metal=.1)
        A.cyl(f"{D}/gas_cage/cyl_{k}", (cx_, y, 0.), (cx_, y, 1.28), .1175, b, sides=28)
        A.cyl(f"{D}/gas_cage/sh_{k}", (cx_, y, 1.28), (cx_, y, 1.40), .1175, s, sides=28, r1=.045)
        A.cyl(f"{D}/gas_cage/cap_{k}", (cx_, y, 1.40), (cx_, y, 1.52), .06, A.pbr("cyl_cap", (70, 72, 74), rough=.5, metal=.6))
    A.reference(f"{D}/oxyfuel_cart", VA.model("portable_welding_cart"), (x1 - .55, .05, 0.), -90.)
    A.reference(f"{D}/extinguisher", VA.model("korean_fire_extinguisher_01"), (x0 + .25, .1, 0.), 90.)
    quad(A, f"{D}/sign_fire", (x0 + .012, -.05, 1.45), (0, .3, 0), (0, 0, .2),
         decal(A, "sign_fire", sign("sign_fire", "FIRE", ["FIRE EXTINGUISHER", "Hot work: keep within 10 m"])))
    # front: programming desk, scrap skip, barrels, crates
    A.reference(f"{D}/desk", VA.model("metal_office_desk"), (-.6, y0 + .55, 0.), 0.)
    A.reference(f"{D}/stool_desk", VA.model("metal_stool_01"), (-.4, y0 + 1.25, 0.), 20.)
    A.box(f"{D}/laptop_base", (-.75, y0 + .5, .7955), (.17, .12, .008), A.pbr("laptop", (40, 42, 46), rough=.4, metal=.5))
    from scipy.spatial.transform import Rotation
    A.box(f"{D}/laptop_lid", (-.75, y0 + .39, .905), (.17, .006, .11), A.pbr("laptop", (40, 42, 46), rough=.4, metal=.5),
          quat_xyzw=Rotation.from_euler("x", -12, degrees=True).as_quat())
    A.reference(f"{D}/scrap_skip", VA.model("metal_trash_can"), (2.5, y0 + .45, 0.), 0.)
    A.reference(f"{D}/barrel_0", VA.model("barrel_03"), (3.9, y0 + .45, 0.), 0.)
    A.reference(f"{D}/barrel_1", VA.model("barrel_03"), (4.6, y0 + .5, 0.), 40.)
    A.reference(f"{D}/crate_0", VA.model("wooden_crate_01"), (x1 - .5, -1.0, 0.), 90.)
    A.reference(f"{D}/crate_1", VA.model("wooden_crate_01"), (x1 - .5, -1.0, .35), 85.)
    # right: the fitter's bench with a vise, tools and offcuts; drill press; tool chest
    bx, by = 3.6, -1.9
    top_m = A.pbr("bench_top", (88, 90, 94), rough=.5, metal=.7)
    A.box(f"{D}/bench/top", (bx, by, .88), (1.0, .4, .02), top_m)
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        A.box(f"{D}/bench/leg_{k}", (bx + dx*.95, by + dy*.35, .43), (.03, .03, .43), steel)
    A.box(f"{D}/bench/shelf", (bx, by, .2), (.95, .35, .01), steel)
    A.reference(f"{D}/bench/vise", VA.model("bench_vice_01"), (bx + .75, by - .25, .9), 0., ground=True)
    A.reference(f"{D}/bench/hammer", VA.model("cross_pein_hammer"), (bx - .2, by - .1, .9), 70., ground=True)
    A.reference(f"{D}/bench/wrench", VA.model("adjustable_wrench"), (bx + .1, by + .05, .9), 10., ground=True)
    A.reference(f"{D}/bench/comb_wrench", VA.model("combination_wrench"), (bx + .25, by + .12, .9), 80., ground=True)
    A.reference(f"{D}/bench/toolbox", VA.model("metal_toolbox"), (bx - .7, by + .2, .9), 0.)
    for i in range(4):
        A.box(f"{D}/bench/offcut_{i}", (bx - .3 + i*.17, by + .25, .905 + .005), (.07, .05, .005), mill)
    A.box(f"{D}/drill_stand", (x1 - .4, .9, .4), (.25, .3, .4), A.pbr("stand_green", (60, 90, 70), rough=.5, metal=.4))
    A.reference(f"{D}/drill_press", VA.model("drill_press_01"), (x1 - .4, .9, .8), -90.)
    A.reference(f"{D}/tool_chest", VA.model("metal_tool_chest"), (x1 - .4, -3.0, 0.), -90.)
    A.reference(f"{D}/tool_cart", VA.model("tool_cart"), (-1.95, -.5, 0.), 90.)
    A.reference(f"{D}/stool_cell", VA.model("metal_stool_01"), (-1.9, -1.6, 0.), 0.)


# ================================================================================================ per-frame hose
def _catmull(P, n=40):
    P = np.asarray(P, float)
    P = np.concatenate([P[:1]*2 - P[1:2], P, P[-1:]*2 - P[-2:-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        for t in np.linspace(0, 1, max(2, n//(len(P) - 3)), endpoint=False):
            out.append(.5*((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t*t + (-p0 + 3*p1 - 3*p2 + p3)*t**3))
    out.append(P[-2])
    return np.asarray(out)


HOSE_POINTS = 41
# hose hangers on the torch arm [authored]: (link, point in the link frame); then lifted by HOSE_LIFT in world z so the
# hose rides on top of the link (UR10e: upper arm 612.7 mm and forearm 571.6 mm along the links' -x, shoulder offset
# 176 mm and elbow offset 39 mm along z, from the UR10e URDF kinematics)
HOSE_HANGERS = {"ur10e": [("upper_arm_link", (-.30, 0., .176)), ("forearm_link", (-.28, 0., .04)), ("wrist_1_link", (0., 0., 0.))],
                "fr3": [("fr3_link3", (0., 0., 0.)), ("fr3_link5", (0., 0., -.1)), ("fr3_link6", (0., 0., 0.))],
                # the Panda has the FR3's kinematics; the G1: over its upper back (the feeder is to its left), then its
                # right upper arm and wrist (a sag below the back would hang through the table: first G1 stills)
                "panda": [("panda_link3", (0., 0., 0.)), ("panda_link5", (0., 0., -.1)), ("panda_link6", (0., 0., 0.))],
                "g1": [("torso_link", (-.13, 0., .38)), ("right_shoulder_yaw_link", (0., 0., -.05)),
                       ("right_wrist_roll_link", (0., 0., 0.))]}
HOSE_LIFT = (.11, .10, .10)
HOSE_SAG = {"g1": 0.}                 # m the free span from the feeder sags (default .25)


def hose_points(world, rear):
    """The torch cable from the feeder's gun outlet along the torch arm (clipped at the hangers) to the gun: a smooth
    curve through the hangers' world points, with a free span sagging from the feeder (render-only: the hose carries
    no load in the physics)."""
    w = world
    inst = getattr(w.scene, "torch_instance", w.scene.torch_arm)      # the rig instance carrying the gun
    model = w.scene.arm_hw[inst]["model"]
    labels = list(w.model.body_label)
    q = w.state.body_q.numpy()
    pts = [FEEDER_GUN_OUTLET, FEEDER_GUN_OUTLET + (0., -.12, -.05)]
    hang = []
    for (link, p), lift in zip(HOSE_HANGERS[model], HOSE_LIFT):
        k = next(i for i, lab in enumerate(labels) if lab.startswith(inst + "/") and lab.endswith("/" + link))
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(q[k, 3:7]).as_matrix()
        hang.append(q[k, :3] + R @ np.asarray(p) + np.array([0., 0., lift]))
    mid = (pts[-1] + hang[0])/2 - np.array([0., 0., HOSE_SAG.get(model, .25)])
    P = _catmull(pts + [mid] + hang + [np.asarray(rear)], n=HOSE_POINTS)
    if len(P) > HOSE_POINTS:
        P = P[np.linspace(0, len(P) - 1, HOSE_POINTS).round().astype(int)]
    elif len(P) < HOSE_POINTS:
        P = np.concatenate([P, np.repeat(P[-1:], HOSE_POINTS - len(P), 0)])
    return P


def work_cable_points(world):
    """Work (ground) cable from the source's negative stud to the ground clamp's lug, wherever the clamp is."""
    from .render import _sag
    k = world.scene.bodies["work_clamp"]
    p, R = world.body_pose(k)
    sx_, sy_, sz_ = LY.WORK_CLAMP["size"]
    lug = p + R @ np.array([.022, 0., sz_/2 - .002])
    a = SOURCE_WORK_STUD + np.array([0., -.05, -.03])
    sag = max(.05, min(.6, .5*np.linalg.norm(lug - a)))
    return np.concatenate([[SOURCE_WORK_STUD], _sag(a, lug, sag, n=HOSE_POINTS - 1)])


# ================================================================================================ entry point
def author(A, VA, stage):
    building(A, VA, stage)
    equipment(A)
    static_cables(A, stage)
    cell(A, stage)
    dressing(A, VA)

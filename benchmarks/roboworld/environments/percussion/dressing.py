"""Visual-only dressing of the music-therapy room (no collision, no physics): room shell, LED ceiling panels, window,
client chair, instrument shelves, rug, posters, the station's sensors.

Everything a robot or a tool can touch is built in scene.py. Models and textures are pinned CC0 Poly Haven assets
(visual_assets.py) or authored here. The room's visible surfaces are the ones the acoustics assume (params.ABSORPTION):
carpet tiles, painted gypsum walls, a suspended mineral-fibre ceiling, one window. Lighting: four 600 x 600 mm LED panels
(4000 K) whose luminance is set so the table's illuminance is the seed's 300-500 lx (classroom lighting practice,
EN 12464-1: 300-500 lx for teaching spaces) plus daylight through the window.
"""

import math
import re
import shutil
from pathlib import Path
import numpy as np

from . import params as P
from . import visual_assets as va

PANELS = ((-0.9, 0.0), (0.9, 0.0), (-0.9, 1.6), (0.9, 1.6))
PANEL_HALF = 0.295
WINDOW = dict(y0=0.2, y1=1.8, z0=0.9, z1=2.3)          # on the right wall (x = ROOM x1)


class Dresser:
    def __init__(self, S):
        self.S, self.stage = S, S.stage
        self.credits = set()
        self.folder = S.folder
        self.tex = self.folder/"textures"

    def pbr(self, name, tex_name, tile_m, roughness=.6, tint=None):
        maps = va.texture(tex_name)
        self.credits.add(tex_name)
        return self.S.textured(name, maps, tile_m, roughness, tint=tint)

    def quad_uv(self, path, corners, mat):
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n = np.cross(e1, e2); n /= np.linalg.norm(n)
        uv = np.array([[0, 0], [l1, 0], [l1, l2], [0, l2]])
        return self.S.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    def ref(self, path, asset, pos, yaw=0., scale=None):
        from pxr import Usd
        usd = va.model(asset)
        if asset not in self.credits:
            for f in (usd.parent/"textures").glob("*"):
                if not (self.tex/f.name).exists():
                    shutil.copy2(f, self.tex/f.name)
        self.credits.add(asset)
        S = self.S
        outer = S.UsdGeom.Xform.Define(self.stage, path)
        outer.AddTranslateOp().Set(S.Gf.Vec3d(*map(float, pos)))
        if yaw:
            outer.AddRotateZOp().Set(float(math.degrees(yaw)))
        if scale:
            outer.AddScaleOp().Set(S.Gf.Vec3f(*([float(scale)]*3)))
        inner = S.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{asset}")
        for p in Usd.PrimRange(inner.GetPrim()):
            if p.GetTypeName() == "BasisCurves":
                p.SetActive(False)
        self._rebind(path, asset, usd)
        return outer

    def _rebind(self, path, asset, usd):
        from pxr import Usd
        S = self.S
        UsdShade, T, Sdf = S.UsdShade, S.Sdf.ValueTypeNames, S.Sdf
        root = self.stage.GetPrimAtPath(path)
        made = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(S.UsdGeom.Gprim):
                continue
            src, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not src:
                continue
            key = src.GetPrim().GetName()
            if key not in made:
                files = {}
                for sh in Usd.PrimRange(src.GetPrim()):
                    if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetIdAttr().Get() == "UsdUVTexture":
                        f = UsdShade.Shader(sh).GetInput("file").Get()
                        if f:
                            files[Path(f.path).name] = usd.parent/"textures"/Path(f.path).name
                pick = lambda tag: next((v for k, v in files.items() if re.search(tag, k)), None)
                inputs = {}
                for tag, inp in (("_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"),
                                 ("_nor_gl", "normalmap_texture"), ("_metal", "metallic_texture")):
                    f = pick(tag)
                    if f is not None and f.exists():
                        inputs[inp] = (T.Asset, Sdf.AssetPath(str(f.resolve())))
                if "reflectionroughness_texture" in inputs:
                    inputs["reflection_roughness_texture_influence"] = (T.Float, 1.)
                if "metallic_texture" in inputs:
                    inputs["metallic_texture_influence"] = (T.Float, 1.)
                made[key] = S.mdl(f"pbr_{asset}_{key}", "OmniPBR", inputs)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[key])
        for sp in [prim.GetPath() for prim in Usd.PrimRange(root) if prim.GetName() == "_materials"]:
            prim = self.stage.GetPrimAtPath(sp)
            if prim:
                prim.SetActive(False)


# ------------------------------------------------------------------------------ authored textures
def _img(folder, name, arr):
    from PIL import Image
    p = Path(folder)/name
    Image.fromarray(np.clip(arr*255, 0, 255).astype(np.uint8)).save(p)
    return p


def _noise(n, blur, seed):
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    a = rng.normal(0., 1., (n, n)).astype(np.float32)
    a = (a - a.min())/np.ptp(a)
    b = np.asarray(Image.fromarray((a*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)), np.float32)/255.
    return (b - b.mean())/max(b.std(), 1e-6)


def carpet_tiles(folder):
    """500 mm loop-pile carpet tiles (muted teal-grey), laid quarter-turned, with fine fibre noise and tile seams."""
    n = 1024                                                       # one texture = 2 x 2 tiles = 1 m
    fine = _noise(n, 0.8, 1)*0.05 + _noise(n, 3, 2)*0.03
    base = np.array([0.30, 0.38, 0.38])
    img = np.ones((n, n, 3))*base
    img = img*(1 + fine[..., None])
    stripes = 0.02*np.sin(np.arange(n)*2*np.pi/9.0)
    for k in range(2):
        for j in range(2):
            sl = (slice(k*n//2, (k + 1)*n//2), slice(j*n//2, (j + 1)*n//2))
            img[sl] *= (1 + (stripes[None, :n//2, None] if (k + j) % 2 else stripes[:n//2, None, None]))
    for s in (0, n//2):
        img[s:s + 2, :] *= 0.75
        img[:, s:s + 2] *= 0.75
    return _img(folder, "carpet_tiles.png", img)


def ceiling_tiles(folder):
    """600 mm mineral-fibre ceiling tiles in a white T-bar grid (fissured texture)."""
    n = 1024                                                       # one texture = 1 tile = 0.6 m
    f = _noise(n, 1.2, 5)
    img = np.ones((n, n, 3))*0.86 + (f*0.025)[..., None]
    rng = np.random.default_rng(6)
    for _ in range(900):
        x, y = rng.integers(0, n, 2)
        L = rng.integers(3, 12)
        img[y:y + 1, x:x + L] *= 0.88
    g = 14
    img[:g, :] = 0.93; img[-g:, :] = 0.93; img[:, :g] = 0.93; img[:, -g:] = 0.93
    img[g:g + 3, :] *= 0.8; img[:, g:g + 3] *= 0.8
    return _img(folder, "ceiling_tiles.png", img)


def rug(folder):
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n]/n
    img = np.ones((n, n, 3))*np.array([0.52, 0.18, 0.12])
    band = (np.abs(((xx*6) % 1) - 0.5) < 0.12)
    img[band] = np.array([0.80, 0.62, 0.30])
    border = (xx < 0.05) | (xx > 0.95) | (yy < 0.05) | (yy > 0.95)
    img[border] = np.array([0.18, 0.22, 0.35])
    img *= (1 + _noise(n, 0.7, 9)*0.06)[..., None]
    return _img(folder, "rug.png", img)


def poster(folder, name, title, lines, bg=(0.95, 0.93, 0.86), fg=(0.15, 0.2, 0.35), staff=False):
    from PIL import Image, ImageDraw, ImageFont
    W, H = 900, 1200
    im = Image.new("RGB", (W, H), tuple(int(c*255) for c in bg))
    d = ImageDraw.Draw(im)
    fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fr = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    F = lambda f, s: ImageFont.truetype(f, s) if Path(f).exists() else ImageFont.load_default()
    col = tuple(int(c*255) for c in fg)
    d.text((W/2, 110), title, font=F(fb, 68), fill=col, anchor="mm")
    y = 240
    if staff:
        for i in range(5):
            d.line([(80, 300 + 28*i), (W - 80, 300 + 28*i)], fill=col, width=4)
        for k, (x, s) in enumerate(((180, 3), (300, 2), (420, 1), (540, 2), (660, 3), (780, 3))):
            cy = 300 + 28*(4 - s) + 14
            d.ellipse([x - 22, cy - 15, x + 22, cy + 15], fill=col)
            d.line([(x + 20, cy), (x + 20, cy - 110)], fill=col, width=5)
        y = 520
    for ln in lines:
        d.text((W/2, y), ln, font=F(fr, 44), fill=col, anchor="mm")
        y += 78
    p = Path(folder)/f"{name}.png"
    im.save(p)
    return p


def _tex_mat(S, name, path, rough=0.7):
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    return S.mdl(name, "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(path).resolve()))),
                                   "reflection_roughness_constant": (T.Float, rough)})


def _tiled(S, name, path, tile_m, rough):
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    return S.mdl(name, "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(path).resolve()))),
                                   "reflection_roughness_constant": (T.Float, rough),
                                   "texture_scale": (T.Float2, S.Gf.Vec2f(1/tile_m, 1/tile_m))})


# ------------------------------------------------------------------------------ the room
def panel_luminance(target_lux, reference):
    """Panel luminance (nit) giving target_lux at `reference` from the four Lambertian panels: E = sum L A cos1 cos2/r^2."""
    ref = np.asarray(reference, float)
    A = (2*PANEL_HALF)**2
    g = 0.
    for x, y in PANELS:
        v = np.array([x, y, P.ROOM["h"] - 0.01]) - ref
        r = float(np.linalg.norm(v))
        c = max(0., v[2]/r)
        g += A*c*c/r**2
    return target_lux/g


def dress(S, world):
    d = Dresser(S)
    R = P.ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    lay = world.scene.layout
    tex = S.folder/"textures"
    floor = _tiled(S, "carpet", carpet_tiles(tex), 1.0, .95)
    wall = d.pbr("wall_paint", "plastered_wall_04", 3.0, .9, tint=(1.0, 0.96, 0.86))
    accent = d.pbr("wall_accent", "plastered_wall_04", 3.0, .9, tint=(0.62, 0.78, 0.74))
    ceiling = _tiled(S, "ceiling", ceiling_tiles(tex), 0.6, .9)
    d.quad_uv("/Dressing/room/floor", [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)], floor)
    d.quad_uv("/Dressing/room/back", [(x1, y0, 0.), (x0, y0, 0.), (x0, y0, h), (x1, y0, h)], wall)
    d.quad_uv("/Dressing/room/front", [(x0, y1, 0.), (x0, y1, h), (x1, y1, h), (x1, y1, 0.)], accent)
    d.quad_uv("/Dressing/room/left", [(x0, y0, 0.), (x0, y1, 0.), (x0, y1, h), (x0, y0, h)], wall)
    W = WINDOW
    # right wall with a window opening (four quads around it)
    for c in ([(x1, y0, 0.), (x1, y0, h), (x1, W["y0"], h), (x1, W["y0"], 0.)],
              [(x1, W["y1"], 0.), (x1, W["y1"], h), (x1, y1, h), (x1, y1, 0.)],
              [(x1, W["y0"], 0.), (x1, W["y0"], W["z0"]), (x1, W["y1"], W["z0"]), (x1, W["y1"], 0.)],
              [(x1, W["y0"], W["z1"]), (x1, W["y0"], h), (x1, W["y1"], h), (x1, W["y1"], W["z1"])]):
        d.quad_uv(f"/Dressing/room/right_{len(c)}_{c[0][1]:.1f}_{c[0][2]:.1f}".replace(".", "p").replace("-", "m"), c, wall)
    d.quad_uv("/Dressing/room/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], ceiling)
    white = S.color("window_frame", (.90, .90, .88), .4, 0.)
    sill = S.color("sill", (.85, .84, .80), .3, 0.)
    for k, (cy, cz, sy, sz) in enumerate((((W["y0"] + W["y1"])/2, W["z0"], W["y1"] - W["y0"], 0.06),
                                         ((W["y0"] + W["y1"])/2, W["z1"], W["y1"] - W["y0"], 0.06),
                                         (W["y0"], (W["z0"] + W["z1"])/2, 0.06, W["z1"] - W["z0"]),
                                         (W["y1"], (W["z0"] + W["z1"])/2, 0.06, W["z1"] - W["z0"]),
                                         ((W["y0"] + W["y1"])/2, (W["z0"] + W["z1"])/2, 0.05, W["z1"] - W["z0"]))):
        S.box(f"/Dressing/window/frame_{k}", (0.08, sy, sz), white, (x1 - 0.02, cy, cz), (0, 0, 0, 1), bevel=0.005)
    S.box("/Dressing/window/sill", (0.22, W["y1"] - W["y0"] + 0.1, 0.03), sill, (x1 - 0.1, (W["y0"] + W["y1"])/2, W["z0"] - 0.02),
          (0, 0, 0, 1), bevel=0.004)
    sky = S.color("outside", (.75, .82, .90), .9, 0., emissive=(.85, .90, 1.0), emissive_intensity=900.)
    d.quad_uv("/Dressing/window/outside", [(x1 + 0.6, W["y0"] - 0.5, W["z0"] - 0.5), (x1 + 0.6, W["y0"] - 0.5, W["z1"] + 0.5),
                                          (x1 + 0.6, W["y1"] + 0.5, W["z1"] + 0.5), (x1 + 0.6, W["y1"] + 0.5, W["z0"] - 0.5)], sky)
    daylight = S.UsdLux.RectLight.Define(S.stage, "/Dressing/lights/window_daylight")
    daylight.CreateWidthAttr(W["y1"] - W["y0"]); daylight.CreateHeightAttr(W["z1"] - W["z0"])
    daylight.CreateIntensityAttr(1200.); daylight.CreateEnableColorTemperatureAttr(True); daylight.CreateColorTemperatureAttr(6500.)
    xf = S.UsdGeom.Xformable(daylight)
    xf.AddTranslateOp().Set(S.Gf.Vec3d(x1 + 0.05, (W["y0"] + W["y1"])/2, (W["z0"] + W["z1"])/2))
    xf.AddRotateYOp().Set(-90.)
    # skirting
    skirt = S.color("skirting", (.88, .87, .84), .45, 0.)
    for k, (c, sz) in enumerate((((0, y0 + 0.01, 0.04), (x1 - x0, 0.015, 0.08)), ((0, y1 - 0.01, 0.04), (x1 - x0, 0.015, 0.08)),
                                 ((x0 + 0.01, (y0 + y1)/2, 0.04), (0.015, y1 - y0, 0.08)))):
        S.box(f"/Dressing/room/skirting_{k}", sz, skirt, c, (0, 0, 0, 1), bevel=0.002)
    # LED panels (emissive face + rect light each)
    target = lay.get("lighting_lux", 400.)
    ref = (0.0, 0.67, P.TABLE["top_z"])
    L = panel_luminance(target, ref)
    lens = S.color("panel_lens", (.95, .95, .95), .5, 0., emissive=(1.0, 0.96, 0.90), emissive_intensity=L*0.25)
    frame = S.color("panel_frame", (.85, .85, .85), .4, .2)
    for k, (x, y) in enumerate(PANELS):
        S.box(f"/Dressing/lights/panel_frame_{k}", (0.62, 0.62, 0.02), frame, (x, y, h - 0.01), (0, 0, 0, 1), bevel=0.003)
        d.quad_uv(f"/Dressing/lights/panel_lens_{k}", [(x - PANEL_HALF, y - PANEL_HALF, h - 0.021),
                                                      (x - PANEL_HALF, y + PANEL_HALF, h - 0.021),
                                                      (x + PANEL_HALF, y + PANEL_HALF, h - 0.021),
                                                      (x + PANEL_HALF, y - PANEL_HALF, h - 0.021)], lens)
        lt = S.UsdLux.RectLight.Define(S.stage, f"/Dressing/lights/panel_light_{k}")
        lt.CreateWidthAttr(2*PANEL_HALF); lt.CreateHeightAttr(2*PANEL_HALF); lt.CreateIntensityAttr(float(L))
        lt.CreateEnableColorTemperatureAttr(True); lt.CreateColorTemperatureAttr(4000.)
        S.UsdGeom.Xformable(lt).AddTranslateOp().Set(S.Gf.Vec3d(x, y, h - 0.03))
    dome = S.UsdLux.DomeLight.Define(S.stage, "/Dressing/lights/ambient")
    dome.CreateIntensityAttr(25.)
    # client area: rug, the client's chair facing the robot, the therapist's armchair by the window
    rugm = _tex_mat(S, "rug", rug(tex), .95)
    cx, cy = P.CLIENT_CHAIR_XY
    S.mesh("/Dressing/client/rug", [(cx - 1.0, cy - 0.7, 0.004), (cx + 1.0, cy - 0.7, 0.004), (cx + 1.0, cy + 0.7, 0.004),
                                    (cx - 1.0, cy + 0.7, 0.004)], [0, 1, 2, 0, 2, 3],
           uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[(0, 0, 1)]*4, mat=rugm)
    d.ref("/Dressing/client/chair", "SchoolChair_01", (cx, cy + 0.05, 0.0), yaw=math.pi)
    d.ref("/Dressing/room/armchair", "modern_arm_chair_01", (2.2, 2.1, 0.0), yaw=math.radians(-130))
    # back wall (behind the client): shelves of small percussion (render-only)
    d.ref("/Dressing/room/shelves", "steel_frame_shelves_03", (x0 + 0.25, 1.0, 0.0), yaw=-math.pi/2)
    _shelf_instruments(S, d, (x0 + 0.25, 1.0))
    d.ref("/Dressing/room/basket", "wicker_basket_02", (x0 + 0.35, 2.0, 0.0))
    d.ref("/Dressing/room/plant", "potted_plant_02", (2.55, 3.0, 0.0))
    d.ref("/Dressing/room/succulent", "potted_plant_04", (x1 - 0.12, 1.4, W["z0"]))
    d.ref("/Dressing/room/clock", "wall_clock", (0.9, y1 - 0.02, 2.05), yaw=math.pi)
    d.ref("/Dressing/room/picture", "hanging_picture_frame_01", (x0 + 0.02, 1.2, 1.6), yaw=math.pi/2)
    p1 = poster(tex, "poster_rhythm", "Rhythm & Movement", ["Walk with the beat", "Stop with the drum", "Tap - Tap - Rest"],
                staff=True)
    p2 = poster(tex, "poster_session", "Today's Session", ["Hello song", "Drum circle", "Xylophone echo", "Goodbye song"],
                bg=(0.88, 0.94, 0.97), fg=(0.10, 0.25, 0.30))
    for k, (pth, x) in enumerate(((p1, 0.9), (p2, 1.7))):
        m = _tex_mat(S, f"poster_{k}", pth, .6)
        S.mesh(f"/Dressing/room/poster_{k}", [(x - 0.3, y1 - 0.01, 1.2), (x + 0.3, y1 - 0.01, 1.2), (x + 0.3, y1 - 0.01, 1.8),
                                               (x - 0.3, y1 - 0.01, 1.8)][::-1], [0, 1, 2, 0, 2, 3],
               uv=[(0, 1), (1, 1), (1, 0), (0, 0)][::-1], normals=[(0, -1, 0)]*4, mat=m)
    # the wall behind the robot station: a whiteboard with the session's song sheet, and the clock
    wb = poster(tex, "whiteboard_session", "Group 2 - Tuesday", ["Hello song (C)", "Steady beat: walk", "Echo game",
                                                                 "Pentatonic jam"], bg=(0.97, 0.97, 0.96), fg=(0.1, 0.2, 0.55))
    m = _tex_mat(S, "whiteboard", wb, .25)
    S.box("/Dressing/room/whiteboard_frame", (1.24, 0.02, 0.84), S.mats["aluminum"], (-1.3, y0 + 0.012, 1.45), (0, 0, 0, 1),
          bevel=0.004)
    # the board faces +y: a viewer facing it sees +x to the LEFT, so u runs from x = -0.7 (u 0) to x = -1.9 (u 1)
    # (u increasing with +x draws the text mirrored)
    S.mesh("/Dressing/room/whiteboard", [(-1.9, y0 + 0.024, 1.07), (-0.7, y0 + 0.024, 1.07), (-0.7, y0 + 0.024, 1.83),
                                         (-1.9, y0 + 0.024, 1.83)], [0, 1, 2, 0, 2, 3], uv=[(1, 0), (0, 0), (0, 1), (1, 1)],
           normals=[(0, 1, 0)]*4, mat=m)
    d.ref("/Dressing/room/clock_back", "wall_clock", (1.2, y0 + 0.02, 2.0))
    if lay["preset"] == "gait":
        tape = S.color("floor_tape", (.95, .80, .10), .6, 0.)
        for k in range(8):
            S.box(f"/Dressing/gait/step_{k}", (0.30, 0.05, 0.002), tape, (-1.6 + 0.4*k, 1.25, 0.002), (0, 0, 0, 1), bevel=0.0005)
        T = P.TABLE
        S.box("/Dressing/gait/table_top", (2*T["half"][1], 2*T["half"][0], 0.025), S.mats["oak_table"],
              (x0 + 0.35, 2.0, T["top_z"] - 0.0125), (0, 0, 0, 1), uv_per_m=1.)
        for sx in (-1, 1):
            for sy in (-1, 1):
                S.box(f"/Dressing/gait/table_leg_{sx + 1}{sy + 1}", (0.04, 0.04, T["top_z"] - 0.025), S.mats["table_frame"],
                      (x0 + 0.35 + sx*0.27, 2.0 + sy*0.57, (T["top_z"] - 0.025)/2), (0, 0, 0, 1))
    _lounge(S, d, tex)
    _station(S, world)
    if world.scene.fixtures.get("speaker"):
        _speaker(S, world.scene.fixtures["speaker"])
    return d.credits


def keyboard_texture(folder):
    """Top view of a 61-key keyboard (white keys with black keys), for the digital piano's key bed."""
    W, H = 2440, 300
    img = np.ones((H, W, 3))*0.95
    n_white = 36
    kw = W/n_white
    for k in range(n_white + 1):
        x = int(k*kw)
        img[:, max(0, x - 1):x + 1] = 0.55
    for k in range(n_white):
        if k % 7 in (0, 1, 3, 4, 5):
            x0 = int((k + 0.68)*kw)
            img[:int(H*0.62), x0:int(x0 + 0.62*kw)] = 0.04
    return _img(folder, "keyboard_keys.png", img)


def _lounge(S, d, tex):
    """The rest of a music-therapy room (visual only): a sofa with cushions and an ottoman on the front wall, a low
    instrument cabinet on the back wall with frame drums, song books and a photo, a digital piano on an X-stand, a crate
    of boomwhackers by the shelves, pictures."""
    import trimesh
    from .render import cylinder_z, disc_z
    R = P.ROOM
    x0, x1, y0, y1 = R["x0"], R["x1"], R["y0"], R["y1"]
    d.ref("/Dressing/lounge/sofa", "sofa_02", (-1.55, y1 - 0.50, 0.0), yaw=math.pi)
    d.ref("/Dressing/lounge/pillows", "throw_pillows_01", (-1.55, y1 - 0.42, 0.02), yaw=math.pi)
    d.ref("/Dressing/lounge/ottoman", "Ottoman_01", (-0.55, y1 - 0.85, 0.0), yaw=0.3, scale=0.8)
    d.ref("/Dressing/lounge/picture_front", "hanging_picture_frame_02", (-1.55, y1 - 0.01, 1.65), yaw=math.pi)
    # back wall: low cabinet with instruments on it, right of the whiteboard
    cab_x = 1.75
    d.ref("/Dressing/lounge/cabinet", "modern_wooden_cabinet", (cab_x, y0 + 0.28, 0.0), yaw=0.0, scale=0.75)
    top = 0.68*0.75
    skin = S.color("frame_drum_skin", (.88, .82, .68), .7, 0.)
    rim = S.color("frame_drum_rim", (.62, .40, .20), .5, 0.)
    for k, (dx, r) in enumerate(((-0.55, 0.20), (-0.15, 0.16))):
        base = f"/Dressing/lounge/frame_drum_{k}"
        S.xform(base, (cab_x + dx, y0 + 0.14, top + r + 0.01), (0.7071, 0, 0, 0.7071))
        v, f, n, uv = cylinder_z(r, -0.03, 0.03, n=64)
        S.mesh(base + "/rim", v, f, uv=uv, normals=n, mat=rim)
        v, f, n, uv = disc_z(r - 0.002, 0.03, n=64)
        S.mesh(base + "/skin", v, f, normals=n, mat=skin)
    d.ref("/Dressing/lounge/books", "book_encyclopedia_set_01", (cab_x + 0.15, y0 + 0.20, top), yaw=0.0)
    d.ref("/Dressing/lounge/photo", "standing_picture_frame_01", (cab_x + 0.75, y0 + 0.22, top), yaw=math.radians(15))
    d.ref("/Dressing/lounge/picture_back", "hanging_picture_frame_03", (cab_x + 0.2, y0 + 0.01, 1.55), yaw=0.0)
    # digital piano on an X-stand, back-left corner, facing +y
    px, py = -2.35, y0 + 0.55
    body = S.color("piano_body", (.05, .05, .055), .35, 0.)
    keys = S.textured("piano_keys", {"diff": keyboard_texture(tex)}, 1.0, .35) if hasattr(S, "textured") else body
    S.box("/Dressing/lounge/piano_body", (0.98, 0.30, 0.10), body, (px, py, 0.78), (0, 0, 0, 1), bevel=0.012)
    S.mesh("/Dressing/lounge/piano_keys", [(px - 0.45, py + 0.02, 0.8305), (px + 0.45, py + 0.02, 0.8305),
                                           (px + 0.45, py + 0.15, 0.8305), (px - 0.45, py + 0.15, 0.8305)], [0, 1, 2, 0, 2, 3],
           uv=[(0, 1), (1, 1), (1, 0), (0, 0)], normals=[(0, 0, 1)]*4, mat=keys)
    for sgn in (-1, 1):
        for k in (-1, 1):
            leg = trimesh.creation.cylinder(radius=0.012, height=0.95, sections=12)
            leg.apply_transform(trimesh.transformations.rotation_matrix(sgn*0.62, [0, 1, 0]))
            S.mesh(f"/Dressing/lounge/piano_leg_{sgn + 1}{k + 1}", leg.vertices + np.array([px + k*0.30, py, 0.36]), leg.faces,
                   smooth=True, mat=S.mats["cart_frame"])
    # crate of boomwhackers beside the shelves
    d.ref("/Dressing/lounge/crate", "plastic_crate_01", (x0 + 0.45, 2.75, 0.0), yaw=0.2)
    for k, c in enumerate([(.85, .1, .1), (.95, .55, .05), (.95, .85, .1), (.15, .65, .2), (.1, .4, .85)]):
        tube = trimesh.creation.cylinder(radius=0.025, height=0.5 - 0.04*k, sections=16)
        tube.apply_transform(trimesh.transformations.rotation_matrix(0.25 - 0.12*k, [1, 0, 0]))
        S.mesh(f"/Dressing/lounge/crate_whacker_{k}", tube.vertices + np.array([x0 + 0.40 + 0.03*k, 2.70 + 0.03*k, 0.30]),
               tube.faces, smooth=True, mat=S.color(f"boomwhacker_{k}", c, .35, 0.))


def _shelf_instruments(S, d, xy):
    """A therapy room's instrument shelving, stocked (render-only): the pinned steel shelf unit's boards (measured from
    its USD: right bay - world y -0.29..1.0 - boards at 0.66, 1.09, 1.75 m and the 2.30 m top; left bay - y 1.0..2.04 -
    a drawer unit to 0.64 m, a board at 1.32 m) carry hand drums, a djembe, tambourines, frame drums on edge,
    boomwhackers, egg shakers in a bowl, maracas, claves, a tone block, chime bars, song binders and storage boxes.
    Local coordinates: u = world y, v = distance out from the wall-side back of the shelf (world x - x0)."""
    import trimesh
    from .render import cylinder_z, disc_z
    x, y = xy
    back = x - 0.33                                   # shelf back (world x); items stand 0.10-0.55 m in front of it
    W = lambda u, v, z: (back + v, u, z)
    skin = S.color("hand_drum_skin", (.86, .80, .66), .7, 0.)
    wood = S.mats.get("walnut_shell") or S.color("hd_wood", (.4, .25, .12), .5, 0.)
    light_wood = S.color("shelf_light_wood", (.72, .55, .35), .55, 0.)
    red = S.color("shelf_red", (.70, .12, .10), .5, 0.)
    metal = S.color("jingle_metal", (.78, .76, .70), .3, .9)

    def drum(name, u, v, z, r, h, shell=wood, skin_m=skin, q=(0, 0, 0, 1)):
        S.xform(name, W(u, v, z), q)
        vv, f, n, uv = cylinder_z(r, 0.0, h, n=48)
        S.mesh(name + "/shell", vv, f, uv=uv, normals=n, mat=shell)
        vv, f, n, uv = disc_z(r, h, n=48)
        S.mesh(name + "/skin", vv, f, normals=n, mat=skin_m)
        vv, f, n, uv = disc_z(r, 0.0, n=48, up=False)
        S.mesh(name + "/bottom", vv, f, normals=n, mat=shell)

    def tube(name, p, axis_q, r, L, mat):
        t = trimesh.creation.cylinder(radius=r, height=L, sections=20)
        t.apply_transform(axis_q)
        S.mesh(name, t.vertices + np.array(p), t.faces, smooth=True, mat=mat)

    # ---- right bay, board 0.66 m: three hand drums and a djembe
    for k, (u, r, h) in enumerate(((-0.12, 0.13, 0.20), (0.22, 0.10, 0.07), (0.50, 0.11, 0.22))):
        drum(f"/Dressing/shelf/hand_drum_{k}", u, 0.30, 0.66, r, h)
    drum("/Dressing/shelf/djembe", 0.82, 0.30, 0.66, 0.15, 0.33, shell=light_wood)
    # ---- right bay, board 1.09 m: boomwhackers lying in a bundle, egg shakers in a bowl, two tambourines
    colors = [(.85, .1, .1), (.95, .55, .05), (.95, .85, .1), (.15, .65, .2), (.1, .4, .85), (.45, .15, .7),
              (.9, .3, .55), (.2, .75, .75)]
    for k, c in enumerate(colors):
        L = 0.63 - 0.05*k
        rot = trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0])     # along world x (into the shelf)
        tube(f"/Dressing/shelf/boomwhacker_{k}", W(-0.22 + 0.052*(k % 4), 0.06 + L/2, 1.09 + 0.026 + 0.05*(k // 4)), rot,
             0.025, L, S.color(f"boomwhacker_{k}", c, .35, 0.))
    bowl = trimesh.creation.icosphere(subdivisions=3, radius=0.11)
    bowl.vertices[:, 2] = np.minimum(bowl.vertices[:, 2], 0.0)
    S.mesh("/Dressing/shelf/egg_bowl", bowl.vertices*np.array([1, 1, 0.55]) + np.array(W(0.18, 0.30, 1.09 + 0.061)),
           bowl.faces, smooth=True, mat=light_wood)
    for k, c in enumerate(((.9, .2, .2), (.2, .5, .9), (.95, .8, .1), (.2, .7, .3), (.95, .45, .1), (.6, .2, .7))):
        e = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        e.vertices *= np.array([0.021, 0.021, 0.028])
        a = 2*math.pi*k/6
        S.mesh(f"/Dressing/shelf/egg_{k}", e.vertices + np.array(W(0.18 + 0.045*math.cos(a), 0.30 + 0.045*math.sin(a),
                                                                     1.09 + 0.045)), e.faces, smooth=True,
               mat=S.color(f"egg_{k}", c, .3, 0.))
    for k, u in enumerate((0.48, 0.66)):           # tambourines standing on edge, leaning back
        base = f"/Dressing/shelf/tambourine_{k}"
        S.xform(base, W(u, 0.16, 1.09 + 0.125), (0.0, 0.62, 0.0, 0.785))
        vv, f, n, uv = cylinder_z(0.12, 0.0, 0.05, n=48)
        S.mesh(base + "/rim", vv, f, uv=uv, normals=n, mat=light_wood if k else wood)
        vv, f, n, uv = disc_z(0.12, 0.05, n=48)
        S.mesh(base + "/skin", vv, f, normals=n, mat=skin)
        for j in range(5):
            a = 2*math.pi*j/5
            jv, jf, jn, _ = cylinder_z(0.022, 0.012, 0.038, n=16)
            S.mesh(f"{base}/jingle_{j}", jv + np.array([0.12*math.cos(a), 0.12*math.sin(a), 0.0]), jf, normals=jn, mat=metal)
    # ---- right bay, board 1.75 m: frame drums on edge, maracas, claves, a tone block
    for k, (v, r) in enumerate(((0.12, 0.20), (0.19, 0.17), (0.26, 0.14))):     # face to face, facing the room
        base = f"/Dressing/shelf/frame_drum_{k}"
        S.xform(base, W(0.10, v, 1.75 + r + 0.005), (0.0, 0.7071, 0.0, 0.7071))
        vv, f, n, uv = cylinder_z(r, -0.025, 0.025, n=64)
        S.mesh(base + "/rim", vv, f, uv=uv, normals=n, mat=light_wood)
        vv, f, n, uv = disc_z(r - 0.002, 0.025, n=64)
        S.mesh(base + "/skin", vv, f, normals=n, mat=skin)
    for k in range(2):
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.045)
        S.mesh(f"/Dressing/shelf/maraca_{k}", head.vertices + np.array(W(0.55 + 0.11*k, 0.35, 1.75 + 0.045)), head.faces,
               smooth=True, mat=S.color(f"maraca_{k}", ((.85, .25, .1), (.1, .55, .3))[k], .45, 0.))
        tube(f"/Dressing/shelf/maraca_handle_{k}", W(0.55 + 0.11*k, 0.35 - 0.11, 1.75 + 0.012),
             trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0]), 0.011, 0.14, wood)
        tube(f"/Dressing/shelf/clave_{k}", W(0.80 + 0.03*k, 0.30, 1.75 + 0.012),
             trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0]), 0.012, 0.20, S.color("rosewood_clave", (.38, .15, .08), .4, 0.))
    S.box("/Dressing/shelf/tone_block", (0.20, 0.06, 0.06), red, W(0.93, 0.30, 1.75 + 0.03), (0, 0, 0, 1), bevel=0.01)
    # ---- right bay top (2.30 m) and left bay: storage boxes, song binders, chime bars on the drawer unit
    for k, (u, c) in enumerate(((0.05, (.85, .80, .70)), (0.55, (.30, .45, .65)))):
        S.box(f"/Dressing/shelf/storage_box_{k}", (0.42, 0.36, 0.26), S.color(f"storage_box_{k}", c, .7, 0.),
              W(u, 0.33, 2.30 + 0.13), (0, 0, 0, 1), bevel=0.01)
    for k in range(5):                               # song binders standing on the 1.32 m board, left bay
        S.box(f"/Dressing/shelf/binder_{k}", (0.29, 0.05, 0.31), S.color(f"binder_{k}", ((.1, .3, .6), (.7, .1, .1),
                                                                                      (.1, .5, .2), (.9, .6, .1),
                                                                                      (.3, .3, .3))[k], .6, 0.),
              W(1.15 + 0.055*k, 0.25, 1.32 + 0.155), (0, 0, 0, 1), bevel=0.004)
    drum("/Dressing/shelf/bongo_0", 1.62, 0.30, 1.32, 0.09, 0.14, shell=light_wood)
    drum("/Dressing/shelf/bongo_1", 1.80, 0.30, 1.32, 0.11, 0.16, shell=light_wood)
    bar_colors = ((.85, .1, .1), (.95, .55, .05), (.95, .85, .1), (.15, .65, .2), (.1, .4, .85))
    for k, c in enumerate(bar_colors):               # a set of single chime bars (resonator boxes) on the drawer unit
        L = 0.24 - 0.02*k
        S.box(f"/Dressing/shelf/chime_box_{k}", (L, 0.05, 0.05), S.color(f"chime_box_{k}", c, .45, 0.),
              W(1.12 + 0.07*k, 0.30, 0.64 + 0.025), (0, 0, 0.7071, 0.7071), bevel=0.004)
        S.box(f"/Dressing/shelf/chime_bar_{k}", (L - 0.02, 0.03, 0.008), S.mats.get("aluminum") or metal,
              W(1.12 + 0.07*k, 0.30, 0.64 + 0.054), (0, 0, 0.7071, 0.7071), bevel=0.002)


def _station(S, world):
    """Sensors on the cart mast: the station camera (D435-class), the measurement microphone (UMIK-1 class) and the
    cue light (unlit / lit lens, switched per render request by the task clock)."""
    import trimesh
    sc = world.scene
    mx, my = sc.fixtures["mast"]["x"], sc.fixtures["mast"]["y"]
    cam = sc.cameras["station_rgb"]
    eye, tgt = np.asarray(cam["eye"], float), np.asarray(cam["target"], float)
    fwd = (tgt - eye)/np.linalg.norm(tgt - eye)
    yaw = math.atan2(fwd[1], fwd[0])
    pitch = math.asin(-fwd[2])
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_euler("ZY", [yaw, pitch]).as_quat()
    S.box("/Dressing/station/camera", (0.025, 0.09, 0.025), S.mats["camera_dark"], tuple(eye - 0.02*fwd), tuple(q), bevel=0.004)
    S.box("/Dressing/station/camera_mount", (0.02, 0.02, 0.04), S.mats["cart_frame"], tuple(eye - 0.02*fwd + np.array([0, 0, 0.03])),
          (0, 0, 0, 1))
    if not getattr(sc, "g1", False):       # the G1 hears through its head array: no measurement microphone on a mast
        mic = trimesh.creation.cylinder(radius=0.0095, height=0.175, sections=20)
        mp = np.asarray(P.STATION_MIC, float)
        dirv = np.array([0.0, 0.85, -0.5]); dirv /= np.linalg.norm(dirv)
        Rm = Rot.align_vectors([dirv], [[0, 0, 1]])[0]
        S.mesh("/Dressing/station/mic", Rm.apply(mic.vertices) + mp - dirv*0.0875, mic.faces, smooth=True, mat=S.mats["aluminum"])
        cap = trimesh.creation.icosphere(subdivisions=2, radius=0.0097)
        S.mesh("/Dressing/station/mic_capsule", cap.vertices + mp, cap.faces, smooth=True, mat=S.mats["camera_dark"])
        S.box("/Dressing/station/mic_clip", (0.02, 0.06, 0.02), S.mats["cart_frame"], tuple(mp - dirv*0.12), (0, 0, 0, 1))
    lit = S.color("cue_lit", (.2, 1., .3), .3, 0., emissive=(.2, 1., .3), emissive_intensity=6000.)
    unlit = S.color("cue_unlit", (.05, .12, .06), .3, 0.)
    pos = (mx + 0.06, my + 0.02, 1.47)
    S.box("/Dressing/station/cue_housing", (0.07, 0.03, 0.05), S.mats["cart_frame"], (pos[0], pos[1] - 0.01, pos[2]), (0, 0, 0, 1))
    S.box("/Dressing/station/cue_lit", (0.05, 0.004, 0.035), lit, (pos[0], pos[1] + 0.006, pos[2]), (0, 0, 0, 1), bevel=0.001)
    S.box("/Dressing/station/cue_unlit", (0.05, 0.004, 0.035), unlit, (pos[0], pos[1] + 0.006, pos[2]), (0, 0, 0, 1), bevel=0.001)
    S.UsdGeom.Imageable(S.stage.GetPrimAtPath("/Dressing/station/cue_lit")).CreateVisibilityAttr("invisible")
    S.UsdGeom.Imageable(S.stage.GetPrimAtPath("/Dressing/station/cue_unlit")).CreateVisibilityAttr("inherited")
    if getattr(sc, "g1", False):
        _tripod(S, sc.fixtures["mast"])
    else:
        _platform(S)


def _tripod(S, mast):
    """G1 configuration: the station camera and cue light on a tripod light stand (scene.sensor_stand)."""
    import trimesh
    beam = S.color("platform_beam", (.62, .64, .66), .32, .85)
    trim = S.color("platform_trim", (.16, .17, .18), .55, 0.)
    mx, my = mast["x"], mast["y"]
    S.box("/Dressing/tripod/pole", (0.03, 0.03, 1.56), trim, (mx, my, 0.80), (0, 0, 0, 1), bevel=0.005)
    S.box("/Dressing/tripod/arm", (0.03, 0.24, 0.024), beam, (mx, my + 0.10, 1.56), (0, 0, 0, 1), bevel=0.004)
    for k in range(3):
        a = 2*math.pi*k/3 + math.pi/6
        leg = trimesh.creation.cylinder(radius=0.008, height=0.42, sections=10)
        d = np.array([math.cos(a), math.sin(a), 0.])
        Rm = trimesh.geometry.align_vectors([0, 0, 1], (d*0.14 + np.array([0, 0, -0.38]))/np.linalg.norm(d*0.14 + np.array([0, 0, -0.38])))
        leg.apply_transform(Rm)
        S.mesh(f"/Dressing/tripod/leg_{k}", leg.vertices + np.array([mx, my, 0.]) + d*0.07 + np.array([0, 0, 0.20]), leg.faces,
               smooth=True, mat=trim)


def _platform(S):
    """The parked mobile dual-arm platform carrying both FR3 bases (visual shell over params.STATION's colliders): a
    white moulded body over a dark bumper skirt on four braked castors, a status light strip, a column and the
    aluminium shoulder beam, an emergency stop and a small service display at the back, the sensor mast."""
    import trimesh
    St = P.STATION
    shell = S.color("platform_shell", (.86, .87, .88), .28, 0.)
    trim = S.color("platform_trim", (.16, .17, .18), .55, 0.)
    beam = S.color("platform_beam", (.62, .64, .66), .32, .85)
    led = S.color("platform_led", (.25, .8, 1.), .3, 0., emissive=(.25, .8, 1.), emissive_intensity=1800.)
    estop = S.color("estop_red", (.75, .05, .04), .35, 0.)
    estop_ring = S.color("estop_yellow", (.95, .78, .05), .4, 0.)
    glass = S.color("display_glass", (.02, .03, .04), .08, 0., emissive=(.10, .25, .35), emissive_intensity=60.)
    (bx, by, bz), (hx, hy, hz) = St["base_c"], St["base_half"]
    z0, z1 = bz - hz, bz + hz
    S.box("/Dressing/platform/skirt", (2*hx + 0.02, 2*hy + 0.02, 0.09), trim, (bx, by, z0 + 0.045), (0, 0, 0, 1), bevel=0.02)
    S.box("/Dressing/platform/body", (2*hx, 2*hy, z1 - z0 - 0.07), shell, (bx, by, (z0 + 0.08 + z1)/2), (0, 0, 0, 1),
          bevel=0.035)
    S.box("/Dressing/platform/top_cover", (2*hx - 0.06, 2*hy - 0.06, 0.012), trim, (bx, by, z1 + 0.004), (0, 0, 0, 1),
          bevel=0.004)
    S.box("/Dressing/platform/led_front", (2*hx - 0.12, 0.004, 0.012), led, (bx, by + hy + 0.001, z1 - 0.05), (0, 0, 0, 1),
          bevel=0.002)
    for sx in (-1, 1):
        S.box(f"/Dressing/platform/led_side_{sx + 1}", (0.004, 2*hy - 0.12, 0.012), led, (bx + sx*(hx + 0.001), by, z1 - 0.05),
              (0, 0, 0, 1), bevel=0.002)
        for sy in (-1, 1):
            w = trimesh.creation.cylinder(radius=0.034, height=0.028, sections=28)
            w.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
            S.mesh(f"/Dressing/platform/castor_{sx + 1}{sy + 1}", w.vertices + np.array([bx + sx*(hx - 0.07), by + sy*(hy - 0.07),
                                                                                          0.034]), w.faces, smooth=True,
                   mat=S.mats["rubber"])
            S.box(f"/Dressing/platform/castor_fork_{sx + 1}{sy + 1}", (0.04, 0.05, 0.03), beam,
                  (bx + sx*(hx - 0.07), by + sy*(hy - 0.07), 0.075), (0, 0, 0, 1), bevel=0.004)
    (cx, cy, cz), (chx, chy, chz) = St["column_c"], St["column_half"]
    S.box("/Dressing/platform/column", (2*chx, 2*chy, 2*chz), shell, (cx, cy, cz), (0, 0, 0, 1), bevel=0.02)
    S.box("/Dressing/platform/column_accent", (0.03, 0.004, 2*chz - 0.08), trim, (cx, cy + chy + 0.001, cz), (0, 0, 0, 1),
          bevel=0.001)
    (mx, my, mz), (mhx, mhy, mhz) = St["beam_c"], St["beam_half"]
    S.box("/Dressing/platform/beam", (2*mhx, 2*mhy, 2*mhz), beam, (mx, my, mz), (0, 0, 0, 1), bevel=0.008)
    for sx in (-1, 1):
        S.box(f"/Dressing/platform/beam_cap_{sx + 1}", (0.012, 2*mhy + 0.004, 2*mhz + 0.004), trim,
              (mx + sx*(mhx + 0.004), my, mz), (0, 0, 0, 1), bevel=0.003)
    # emergency stop and service display on the back of the column
    ring = trimesh.creation.cylinder(radius=0.028, height=0.012, sections=32)
    button = trimesh.creation.cylinder(radius=0.02, height=0.022, sections=32)
    Rx = trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0])
    for nm, m_, o in (("estop_ring", estop_ring, 0.006), ("estop", estop, 0.018)):
        g = (ring if nm == "estop_ring" else button).copy()
        g.apply_transform(Rx)
        S.mesh(f"/Dressing/platform/{nm}", g.vertices + np.array([cx + 0.05, cy - chy - o, cz + chz - 0.06]), g.faces,
               smooth=True, mat=m_)
    S.box("/Dressing/platform/display", (0.12, 0.006, 0.08), glass, (cx - 0.03, cy - chy - 0.003, cz + 0.02), (0, 0, 0, 1),
          bevel=0.003)
    # sensor mast (pole and top arm) at the back of the platform
    mxy = St["mast_xy"]
    top = mz + mhz
    S.box("/Dressing/platform/mast", (0.04, 0.04, 1.58 - top), beam, (mxy[0], mxy[1], (top + 1.58)/2), (0, 0, 0, 1),
          bevel=0.004)
    S.box("/Dressing/platform/mast_arm", (0.03, 0.24, 0.024), beam, (mxy[0], mxy[1] + 0.10, 1.56), (0, 0, 0, 1), bevel=0.004)
    S.box("/Dressing/platform/mast_foot", (0.10, 0.10, 0.012), trim, (mxy[0], mxy[1], top + 0.006), (0, 0, 0, 1), bevel=0.003)


def _speaker(S, sp):
    """Front of the powered monitor (assessment preset): grille and woofer, visual only."""
    from .render import disc_z
    x, y, z = sp["position"]
    w, dpt, h = sp["size"]
    grille = S.color("speaker_grille", (.08, .08, .085), .8, 0.)
    v, f, n, uv = disc_z(0.055, 0.0, n=48)
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_euler("x", 90, degrees=True)
    S.mesh("/Dressing/speaker/woofer", R.apply(v) + np.array([x, y - 0.001, z - 0.02]), f, normals=R.apply(n), mat=grille)
    v, f, n, uv = disc_z(0.018, 0.0, n=32)
    S.mesh("/Dressing/speaker/tweeter", R.apply(v) + np.array([x, y - 0.001, z + 0.075]), f, normals=R.apply(n), mat=grille)

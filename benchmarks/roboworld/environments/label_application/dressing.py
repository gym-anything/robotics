"""Visual-only scene dressing, authored straight into the render USD (no physics, no colliders).

Room shell with pinned CC0 textures (painted block walls, sealed concrete floor, suspended
ceiling), fluorescent fixtures, stock shelving, dock door and floor props from pinned Poly Haven
models (visual_assets.py), a detailed peel-mode desktop printer model laid over the printer's
physics colliders, printed carton markings, and the overhead policy-camera mount. Everything here
is appearance: geometry that a robot or label can touch is built in scene.py, never here.
"""
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import visual_assets as va
from .artwork import SPECIMEN_DRAW_ML, SPECIMEN_STOCK_SIZE

# derived appearance: paint / sealer colours applied to pinned scans
PAINTED_BLOCK = (0.80, 0.79, 0.74)     # off-white masonry paint
SEALED_FLOOR = (0.46, 0.46, 0.45)      # grey sealed concrete
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------- derived textures
def _derived(folder):
    """Write derived textures into folder/textures; returns {name: path}."""
    tex = folder/"textures"
    tex.mkdir(parents=True, exist_ok=True)
    out = {}
    # painted block wall: luminance detail of the pinned block scan under an off-white paint
    src = va.texture("concrete_block_wall")
    lum = np.asarray(Image.open(src["diff"]).convert("L"), np.float32)/255.
    lum = (lum - lum.mean())/max(lum.std(), 1e-6)
    paint = np.clip(np.asarray(PAINTED_BLOCK)[None, None, :]*(1. + .035*lum[..., None]), 0, 1)
    p = tex/"painted_block_diff.jpg"
    Image.fromarray((paint*255).astype(np.uint8)).save(p, quality=92)
    out["painted_block"] = dict(diff=p, rough=src["rough"], nor_gl=src["nor_gl"])
    # sealed floor: the hangar concrete scan re-toned to a mid grey, detail kept
    src = va.texture("hangar_concrete_floor")
    lum = np.asarray(Image.open(src["diff"]).convert("L"), np.float32)/255.
    lum = (lum - lum.mean())/max(lum.std(), 1e-6)
    fl = np.clip(np.asarray(SEALED_FLOOR)[None, None, :]*(1. + .10*lum[..., None]), 0, 1)
    p = tex/"sealed_floor_diff.jpg"
    Image.fromarray((fl*255).astype(np.uint8)).save(p, quality=92)
    out["sealed_floor"] = dict(diff=p, rough=src["rough"], nor_gl=src["nor_gl"])
    # plaster wall (lab room): detail of the pinned plaster scan under a clean off-white paint
    src = va.texture("beige_wall_002")
    lum = np.asarray(Image.open(src["diff"]).convert("L"), np.float32)/255.
    lum = (lum - lum.mean())/max(lum.std(), 1e-6)
    pl = np.clip(np.asarray((.86, .87, .85))[None, None, :]*(1. + .02*lum[..., None]), 0, 1)
    p = tex/"painted_plaster_diff.jpg"
    Image.fromarray((pl*255).astype(np.uint8)).save(p, quality=92)
    out["plaster"] = dict(diff=p, rough=src["rough"], nor_gl=src["nor_gl"])
    # suspended ceiling: 600 mm tiles, fissured mineral fibre, white T-bar grid (1 texture = 1.2 m)
    rng = np.random.default_rng(3)
    n = 1024
    base = .90 + .03*rng.standard_normal((n, n))
    img = Image.fromarray((np.clip(base, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(img)
    for _ in range(2600):
        x, y = rng.integers(0, n, 2)
        d.line([(x, y), (x + rng.integers(-6, 7), y + rng.integers(-6, 7))], fill=int(170 + rng.integers(0, 40)), width=1)
    for k in (0, n//2):
        d.rectangle([k - 6, 0, k + 6, n], fill=246); d.rectangle([0, k - 6, n, k + 6], fill=246)
    p = tex/"ceiling_tile_diff.png"
    img.convert("RGB").save(p)
    out["ceiling"] = dict(diff=p)
    # printer badge + key legends
    im = Image.new("RGBA", (512, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((10, 18), "PACKLINE", font=_font(58), fill=(235, 235, 235, 255))
    d.text((352, 40), "LP-4d", font=_font(40), fill=(200, 60, 40, 255))
    p = tex/"printer_badge.png"; im.save(p); out["printer_badge"] = dict(diff=p)
    for key, model in (("printer_badge_2in", "LP-2d"), ("printer_badge_tt", "LP-4t")):   # 2-inch DT, 4-inch TT class
        im = Image.new("RGBA", (512, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((10, 18), "PACKLINE", font=_font(58), fill=(235, 235, 235, 255))
        d.text((352, 40), model, font=_font(40), fill=(200, 60, 40, 255))
        p = tex/f"{key}.png"; im.save(p); out[key] = dict(diff=p)
    # carton markings: box maker's certificate, orientation arrows, recycling mark (dark ink)
    im = Image.new("RGBA", (1024, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (60, 38, 22, 235)
    d.ellipse([40, 140, 330, 430], outline=ink, width=8)
    d.text((80, 200), "BOX CERTIFICATE", font=_font(22), fill=ink)
    d.text((95, 245), "SINGLE WALL", font=_font(24), fill=ink)
    d.text((95, 285), "ECT 32 lb/in", font=_font(28), fill=ink)
    d.text((105, 330), "MAX 65 lb", font=_font(28), fill=ink)
    # the middle third stays unprinted: that is where shipping labels go
    for i, x in enumerate((790, 900)):
        d.polygon([(x, 120), (x - 50, 200), (x - 18, 200), (x - 18, 330), (x + 18, 330), (x + 18, 200), (x + 50, 200)], fill=ink)
    d.text((735, 360), "THIS SIDE UP", font=_font(34), fill=ink)
    d.text((735, 420), "\u267B PLEASE RECYCLE", font=_font(24), fill=ink)
    p = tex/"carton_print.png"; im.save(p); out["carton_print"] = dict(diff=p)
    # evacuated-tube manufacturer label (fictional brand), u = 0..240 deg around, v = along the tube
    im = Image.new("RGB", (1024, 512), (246, 246, 242))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1023, 70], fill=(150, 110, 200))
    d.text((30, 12), f"VACUTEST  K2EDTA  {SPECIMEN_DRAW_ML:.1f} mL", font=_font(40), fill=(255, 255, 255))
    d.text((30, 110), "REF VT-13100-6   LOT 4H21B   EXP 2027-08", font=_font(30), fill=(40, 40, 40))
    d.text((30, 170), "STERILE R  •  FOR IN VITRO DIAGNOSTIC USE", font=_font(24), fill=(60, 60, 60))
    # fill scale (ticks across the label) instead of a barcode, so no reader confuses it with the specimen ID
    for k in range(12):
        y = 250 + 14*k
        d.line([(560, y), (560 + (60 if k % 4 == 0 else 30), y)], fill=(40, 40, 40), width=3)
    d.polygon([(40, 300), (80, 270), (80, 330)], fill=(200, 30, 30))
    d.text((95, 280), "FILL TO LINE", font=_font(28), fill=(200, 30, 30))
    p = tex/"tube_mfr_label.png"; im.save(p); out["tube_label"] = dict(diff=p)
    # lab stock fronts (fictional brands): nitrile glove box, tube pack carton
    for size, col in (("M", (120, 80, 190)), ("L", (40, 90, 170))):
        im = Image.new("RGB", (768, 208), (246, 246, 244))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 767, 54], fill=col)
        d.text((24, 6), "NITRILE EXAM GLOVES", font=_font(40), fill=(255, 255, 255))
        d.text((24, 74), "POWDER-FREE  •  100 GLOVES", font=_font(30), fill=(50, 50, 50))
        d.rounded_rectangle([560, 70, 740, 196], radius=18, outline=col, width=8)
        d.text((620, 84), size, font=_font(84), fill=col)
        d.text((24, 130), "NON-STERILE  •  AMBIDEXTROUS", font=_font(24), fill=(80, 80, 80))
        p = tex/f"glove_box_{size}.png"; im.save(p); out[f"glove_box_{size}"] = dict(diff=p)
    im = Image.new("RGB", (768, 256), (238, 238, 236))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 767, 64], fill=(150, 110, 200))
    d.text((24, 10), f"VACUTEST K2EDTA {SPECIMEN_DRAW_ML:.1f} mL", font=_font(42), fill=(255, 255, 255))
    d.text((24, 90), f"{SPECIMEN_STOCK_SIZE}  •  100 TUBES", font=_font(34), fill=(40, 40, 40))
    d.text((24, 150), "REF VT-13100-6   LOT 4H21B", font=_font(28), fill=(70, 70, 70))
    p = tex/"tube_pack.png"; im.save(p); out["tube_pack"] = dict(diff=p)
    return out


# ----------------------------------------------------------------------------- USD helpers
class Dresser:
    def __init__(self, stage, folder):
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.stage, self.folder = stage, Path(folder)
        self.mats = {}
        self.tex = _derived(self.folder)
        self.credits = set()

    # materials
    def color(self, name, rgb, rough=.5, metal=0., emissive=None, opacity=None):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/DressMaterials/"+name)
        s = UsdShade.Shader.Define(self.stage, str(m.GetPath())+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if emissive is not None:
            s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
        if opacity is not None:
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def decal(self, name, path):
        """Texture with alpha cut-out (printed ink on a surface)."""
        if name in self.mats:
            return self.mats[name]
        Sdf, UsdShade = self.Sdf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/DressMaterials/"+name)
        s = UsdShade.Shader.Define(self.stage, str(m.GetPath())+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.7)
        rd = UsdShade.Shader.Define(self.stage, str(m.GetPath())+"/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        t = UsdShade.Shader.Define(self.stage, str(m.GetPath())+"/tex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(path).resolve())))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        s.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t.CreateOutput("a", Sdf.ValueTypeNames.Float))
        s.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(.5)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def pbr(self, name, maps, tile_m, roughness=.6):
        """OmniPBR with the maps copied into textures/ (UVs are 2 cycles per metre)."""
        if name in self.mats:
            return self.mats[name]
        from whiteboard.appearance import bind_pbr
        dst = self.folder/"textures"
        local = {}
        for k in ("diff", "rough", "nor_gl"):
            src = Path(maps.get(k, maps["diff"]))
            if src.parent != dst:
                shutil.copy2(src, dst/src.name)
            local[k] = dst/src.name
        m = self.UsdShade.Material.Define(self.stage, "/DressMaterials/"+name)
        bind_pbr(self.stage, m, local, tile_m, roughness=roughness)
        self.mats[name] = m
        return m

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    # geometry
    def xform(self, path, pos=(0, 0, 0), yaw=0., scale=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        if yaw:
            x.AddRotateZOp().Set(float(math.degrees(yaw)))
        if scale is not None:
            x.AddScaleOp().Set(self.Gf.Vec3f(*([float(scale)]*3 if np.isscalar(scale) else map(float, scale))))
        return x

    def box(self, path, center, size, mat, bevel=.002, yaw=0.):
        from whiteboard.usd_scene import bevel_box
        g = bevel_box(self.stage, path, np.asarray(size, float), bevel)
        xf = self.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, center)))
        if yaw:
            xf.AddRotateZOp().Set(float(math.degrees(yaw)))
        self.bind(g, mat)
        return g

    def cyl(self, path, center, r, h, mat, axis="Z", rot=None):
        """Smooth 64-segment cylinder (USD's primitive cylinder tessellates coarsely in RTX)."""
        from whiteboard.usd_scene import smooth_cylinder
        g = smooth_cylinder(self.stage, path, float(r), float(h))
        xf = self.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, center)))
        base = {"Z": (0., 0., 0.), "Y": (90., 0., 0.), "X": (0., 90., 0.)}[axis]
        extra = (0., 0., 0.) if rot is None else tuple(map(float, rot))
        if any(base) or any(extra):
            xf.AddRotateXYZOp().Set(self.Gf.Vec3f(base[0] + extra[0], base[1] + extra[1], base[2] + extra[2]))
        self.bind(g, mat)
        return g

    def quad(self, path, corners, mat, uv_per_m=2.):
        """Planar quad (corners counter-clockwise seen from the front); UVs = metres x 2."""
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        uv = np.array([[0, 0], [l1, 0], [l1, l2], [0, l2]])*uv_per_m
        from whiteboard.usd_scene import mesh
        g = mesh(self.stage, path, c, [np.array([0, 1, 2, 3])], uv)
        self.bind(g, mat)
        return g

    def decal_quad(self, path, corners, mat, flip_u=False):
        c = np.asarray(corners, float)
        from whiteboard.usd_scene import mesh
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        if flip_u:
            uv[:, 0] = 1 - uv[:, 0]
        g = mesh(self.stage, path, c, [np.array([0, 1, 2, 3])], uv)
        self.bind(g, mat)
        return g

    def ref(self, path, asset, pos, yaw=0., sub=None, center_sub=True):
        """Reference a pinned model; with `sub`, keep only that piece (others deactivated) and
        re-centre it on its footprint. Model textures are copied next to the stage because RTX
        resolves the models' ./textures/ paths against the root layer."""
        from pxr import Usd
        usd = va.model(asset)
        if asset not in self.credits:
            dst = self.folder/"textures"
            for f in (usd.parent/"textures").glob("*"):
                if not (dst/f.name).exists():
                    shutil.copy2(f, dst/f.name)
        self.credits.add(asset)
        outer = self.xform(path, pos, yaw)
        center = self.UsdGeom.Xform.Define(self.stage, path+"/center")
        inner = self.UsdGeom.Xform.Define(self.stage, path+"/center/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{asset}")
        if sub:
            for child in inner.GetPrim().GetChildren():
                if child.GetName() not in (sub, "_materials"):
                    child.SetActive(False)
            if center_sub:
                src = Usd.Stage.Open(str(usd))
                r = self.UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
                    src.GetPrimAtPath(f"/{asset}/{sub}")).ComputeAlignedRange()
                mn, mx = r.GetMin(), r.GetMax()
                center.AddTranslateOp().Set(self.Gf.Vec3d(-(mn[0] + mx[0])/2, -(mn[1] + mx[1])/2, -mn[2]))
        # the preview-surface network only: RTX mis-compiles these MaterialX graphs
        self._rebind_omnipbr(path, asset, usd)
        return outer


    def _rebind_omnipbr(self, path, asset, usd):
        """RTX renders neither the models' MaterialX graphs (missing ND_normalmap_float) nor their
        UsdTransform2d preview networks; author an OmniPBR per source material with the same
        texture files and bind it on every mesh that used that material."""
        from pxr import Usd
        from whiteboard.appearance import mdl_surface
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        root = self.stage.GetPrimAtPath(path)
        made = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(self.UsdGeom.Gprim):
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
                pick = lambda *keys: next((v for k, v in files.items() if any(t in k for t in keys)), None)
                m = UsdShade.Material.Define(self.stage, f"/DressMaterials/pbr_{asset}_{key}")
                sh = mdl_surface(self.stage, m, "OmniPBR")
                for tag, inp in ((("_diff",), "diffuse_texture"), (("_rough",), "reflectionroughness_texture"),
                                 (("_nor_gl",), "normalmap_texture"), (("_metal",), "metallic_texture")):
                    f = pick(*tag)
                    if f is not None and f.exists():
                        sh.CreateInput(inp, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(f.resolve())))
                if pick("_rough") is not None:
                    sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.)
                if pick("_metal") is not None:
                    sh.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(1.)
                if "glass" in key:
                    sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
                    sh.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(.25)
                made[key] = m
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[key])
        scopes = [prim.GetPath() for prim in Usd.PrimRange(root) if prim.GetName() == "_materials"]
        for sp in scopes:
            self.stage.GetPrimAtPath(sp).SetActive(False)


# ----------------------------------------------------------------------------- room
ROOM = dict(x0=-2.4, x1=2.6, y0=-3.4, y1=1.25, h=3.0)


def room(d, preset):
    R = ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    lab = preset == "specimen"
    wall = d.pbr("wall_plaster" if lab else "wall_block", d.tex["plaster"] if lab else d.tex["painted_block"],
                 3.0 if lab else 2.0, roughness=.85)
    floor = d.pbr("floor_sealed", d.tex["sealed_floor"], 4.0, roughness=.45)
    ceiling = d.color("ceiling_flat", (.9, .9, .88), .9)
    d.quad("/Dressing/room/floor", [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], floor)
    d.quad("/Dressing/room/wall_back", [(x0, y1, 0), (x1, y1, 0), (x1, y1, h), (x0, y1, h)][::-1], wall)
    d.quad("/Dressing/room/wall_front", [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)], wall)
    d.quad("/Dressing/room/wall_left", [(x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h)][::-1], wall)
    d.quad("/Dressing/room/wall_right", [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)], wall)
    # ceiling tiles: texture spans 1.2 m (2 x 2 tiles of 600 mm)
    ctex = d.tex["ceiling"]["diff"]
    d.quad("/Dressing/room/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)],
           _textured(d, "ceiling_tiles", ctex), uv_per_m=1/1.2)
    # cove base (100 mm rubber) and a painted safety band on the dock wall
    base = d.color("cove_base", (.10, .10, .10), .7)
    for nm, a, b in (("back", (x0, y1 - .004), (x1, y1 - .004)), ("left", (x0 + .004, y0), (x0 + .004, y1)),
                     ("right", (x1 - .004, y0), (x1 - .004, y1))):
        cx, cy = (a[0] + b[0])/2, (a[1] + b[1])/2
        sx, sy = abs(b[0] - a[0]) or .008, abs(b[1] - a[1]) or .008
        d.box(f"/Dressing/room/cove_{nm}", (cx, cy, .05), (sx, sy, .10), base, bevel=.001)
    # fluorescent fixtures (models) with the RectLights that actually light the room under them
    for i, (fx, fy) in enumerate(((-.9, .15), (.9, .15), (-.9, -1.35), (.9, -1.35))):
        for k, dy in enumerate((-.12, .12)):
            d.ref(f"/Dressing/lights/fixture_{i}_{k}", "mounted_fluorescent_lights", (fx, fy + dy, h - .005),
                  yaw=0., sub="mounted_fluorescent_lights_d")
        L = d.UsdLux.RectLight.Define(d.stage, f"/Dressing/lights/panel_{i}")
        L.CreateWidthAttr(1.1); L.CreateHeightAttr(.35); L.CreateIntensityAttr(5000.)
        L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(5000.)
        d.UsdGeom.Xformable(L).AddTranslateOp().Set(d.Gf.Vec3d(fx, fy, h - .06))
    dome = d.UsdLux.DomeLight.Define(d.stage, "/Dressing/lights/ambient")
    dome.CreateIntensityAttr(90.)
    # walls: clock, fire safety, security camera
    d.ref("/Dressing/wall/clock", "wall_clock", (.9, y1 - .005, 2.25), yaw=0.)
    d.ref("/Dressing/wall/alarm", "fire_alarm", (x1 - .03, .55, 1.40), yaw=math.pi/2)
    d.ref("/Dressing/floor/extinguisher", "korean_fire_extinguisher_01", (x1 - .25, .75, 0.), yaw=math.pi)
    d.ref("/Dressing/wall/security_camera", "security_camera_01", (x1 - .15, y1 - .30, h - .05), yaw=math.radians(-135))


def _textured(d, name, path):
    """Plain (opaque) textured material via UsdPreviewSurface."""
    if name in d.mats:
        return d.mats[name]
    Sdf, UsdShade = d.Sdf, d.UsdShade
    m = UsdShade.Material.Define(d.stage, "/DressMaterials/"+name)
    s = UsdShade.Shader.Define(d.stage, str(m.GetPath())+"/surface")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.9)
    rd = UsdShade.Shader.Define(d.stage, str(m.GetPath())+"/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    t = UsdShade.Shader.Define(d.stage, str(m.GetPath())+"/tex")
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(path).resolve())))
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    d.mats[name] = m
    return m


# ----------------------------------------------------------------------------- stock and props
SHELF_TOPS = (.125, .630, 1.140, 1.650)      # measured plate tops of steel_frame_shelves_01


def pack_props(d, floor_boxes=True):
    R = ROOM
    y_shelf = R["y1"] - .27
    for nm, x in (("a", -1.60), ("b", 1.62)):
        d.ref(f"/Dressing/stock/shelf_{nm}", "steel_frame_shelves_01", (x, y_shelf, 0.))
        for lvl, z in enumerate(SHELF_TOPS):
            if (nm, lvl) in (("a", 2), ("b", 1)):
                for k, dx in enumerate((-.34, 0., .34)):
                    d.ref(f"/Dressing/stock/crate_{nm}{lvl}_{k}", "plastic_crate_01", (x + dx, y_shelf, z), yaw=0.)
            elif (nm, lvl) != ("b", 3):
                for k, dx in enumerate((-.27, .27)):
                    d.ref(f"/Dressing/stock/box_{nm}{lvl}_{k}", "cardboard_box_01", (x + dx, y_shelf, z),
                          yaw=math.pi/2 + (.03 if k else -.02))
    # floor: carton stack, tote stack, hand truck by the dock door, liner/waste bin
    for k, (px, py, pz, yaw) in enumerate(((2.15, -.35, 0., .1), (2.18, -.33, .342, -.05), (2.12, -.85, 0., 1.62))):
        if floor_boxes:
            d.ref(f"/Dressing/floor/box_{k}", "cardboard_box_01", (px, py, pz), yaw=yaw)
    for k in range(3):
        d.ref(f"/Dressing/floor/tote_{k}", "plastic_crate_03", (-1.95, -1.05, .266*k), yaw=math.pi/2)
    d.ref("/Dressing/wall/dock_door", "rollershutter_door", (R["x0"] + .02, -2.0, 0.), yaw=-math.pi/2,
          sub="rollershutter_door")
    d.ref("/Dressing/floor/hand_truck", "hand_truck", (-2.05, -.25, 0.), yaw=math.radians(115))
    d.ref("/Dressing/floor/bin", "metal_trash_can", (-1.30, -.70, 0.), sub="metal_trash_can")
    # dock-door floor marking (yellow safety tape) and lane line
    tape = d.color("floor_tape_yellow", (.85, .68, .05), .5)
    for k, (cx, cy, sx, sy) in enumerate(((-1.75, -2.0, .03, 1.5), (-2.1, -1.24, .6, .03), (-2.1, -2.76, .6, .03))):
        d.box(f"/Dressing/floor/tape_{k}", (cx, cy, .0006), (sx, sy, .001), tape, bevel=.0002)


def bench_props(d, preset, scene):
    """Small items on and under the bench, placed clear of the task workspace."""
    z = scene.TABLE
    ink = d.color("pad_yellow", (.95, .86, .35), .8)
    if preset != "specimen":
        d.ref("/Dressing/bench/pad", "office_notepads", (.80, -.24, z), yaw=math.pi/2, sub="office_notepads_yellow_pad")
        # handheld tape dispenser (steel frame, red handle, 48 mm tape)
        red = d.color("tape_gun_red", (.70, .07, .05), .45)
        steel = d.color("tape_gun_steel", (.55, .56, .58), .3, .9)
        tape = d.color("tape_roll", (.63, .50, .33), .25)
        d.box("/Dressing/bench/tapegun_frame", (-.74, -.25, z + .045), (.20, .012, .09), steel, bevel=.004, yaw=.3)
        d.cyl("/Dressing/bench/tapegun_roll", (-.74, -.25, z + .075), .05, .048, tape, axis="Y", rot=(0, 0, math.degrees(.3)))
        d.box("/Dressing/bench/tapegun_handle", (-.83, -.28, z + .03), (.03, .035, .06), red, bevel=.008, yaw=.3)
        # flat carton blanks and spare cartons on the bench's lower shelf
        kraft = _textured(d, "kraft_blank", scene_kraft(d))
        for k in range(5):
            d.box(f"/Dressing/under/blank_{k}", (-.35, .02, .19 + .004 + .007*k), (.62, .52, .006), kraft, bevel=.001,
                  yaw=.02*(k - 2))
        d.ref("/Dressing/under/box", "cardboard_box_01", (.45, .05, .19), yaw=math.pi/2)
    if preset == "parcel":
        # 150 lb shipping scale with remote display (appearance only)
        sc = d.color("scale_body", (.18, .18, .20), .5)
        st = d.color("scale_platter", (.65, .66, .67), .22, .95)
        lcd = d.color("scale_lcd", (.20, .30, .18), .2, emissive=(.25, .45, .2))
        d.box("/Dressing/bench/scale_base", (.62, .24, z + .02), (.30, .24, .04), sc, bevel=.006)
        d.box("/Dressing/bench/scale_platter", (.62, .24, z + .043), (.29, .23, .004), st, bevel=.002)
        d.box("/Dressing/bench/scale_display", (.62, .37, z + .07), (.12, .03, .06), sc, bevel=.006)
        d.box("/Dressing/bench/scale_lcd", (.62, .354, z + .075), (.08, .002, .025), lcd, bevel=.0005)
    # label stock: spare rolls (4 x 6 in on 3 in cores, 300 labels) behind the printer
    white = d.color("label_roll", (.95, .95, .93), .6)
    core = d.color("roll_core", (.55, .42, .28), .8)
    for k, x in enumerate((-.80, -.66)):
        d.cyl(f"/Dressing/bench/roll_{k}", (x, .26, z + .054), .065, .104 if preset != "specimen" else .052, white)
        d.cyl(f"/Dressing/bench/roll_core_{k}", (x, .26, z + .055), .039, .106 if preset != "specimen" else .054, core)


def scene_kraft(d):
    return next((d.folder/"textures").glob("kraft*.png"), None) or d.tex["carton_print"]["diff"]


def lab_props(d, scene):
    """Specimen bench extras (visual; tube-wrap's text calls the bench an accessioning bench, tube-label's a blood
    collection station): tube racks, centrifuge, sharps bin, bags."""
    z = scene.TABLE
    rack = d.color("rack_white", (.93, .93, .92), .5)
    cap = {"lav": d.color("cap_lav", (.55, .40, .85), .4), "red": d.color("cap_red2", (.75, .08, .08), .4),
           "grn": d.color("cap_grn", (.10, .55, .25), .4)}
    tube = d.color("tube_clear", (.86, .88, .90), .08, opacity=.3)
    d.box("/Dressing/bench/rack", (.55, .22, z + .03), (.24, .07, .06), rack, bevel=.004)
    for i in range(10):
        c = list(cap.values())[i % 3]
        x = .45 + .022*i
        d.cyl(f"/Dressing/bench/rack_tube_{i}", (x, .22, z + .075), .0065, .10, tube)
        d.cyl(f"/Dressing/bench/rack_cap_{i}", (x, .22, z + .128), .0078, .016, c)
    body = d.color("centrifuge", (.88, .89, .9), .35)
    lid = d.color("centrifuge_lid", (.25, .45, .75), .2)
    d.cyl("/Dressing/bench/centrifuge", (-.72, .22, z + .11), .15, .22, body)
    d.cyl("/Dressing/bench/centrifuge_lid", (-.72, .22, z + .226), .135, .012, lid)
    red = d.color("sharps", (.72, .06, .05), .45)
    d.box("/Dressing/bench/sharps", (.80, -.25, z + .11), (.16, .12, .22), red, bevel=.01)
    bag = d.color("biobag", (.85, .9, .9), .15, opacity=.4)
    d.box("/Dressing/bench/biobag", (-.75, -.22, z + .006), (.16, .22, .012), bag, bevel=.003)


def lab_shelf_stock(d, nm, x, y):
    """Clinical consumables on one steel shelf unit (visual): glove boxes (240 x 125 x 65 mm), tube
    pack cartons, 1 L reagent bottles, a specimen transport cooler on the bottom shelf."""
    yf = y - .20                                    # front row line (shelf depth ~0.45 m, front toward -y)
    gl = {k: d.decal(f"glove_{k}", d.tex[f"glove_box_{k}"]["diff"]) for k in ("M", "L")}
    white = d.color("carton_white", (.93, .93, .91), .6)
    tp = d.decal("tube_pack_front", d.tex["tube_pack"]["diff"])
    hdpe = d.color("reagent_hdpe", (.92, .92, .90), .45)
    cap_b = d.color("reagent_cap", (.12, .28, .62), .4)
    for lvl in (1, 2):
        z = SHELF_TOPS[lvl]
        for k in range(3):                          # glove boxes lying flat, printed long face out
            for j, sz in enumerate(("M", "L")):
                cx, cz = x - .36 + .25*j, z + .0325 + .066*k
                d.box(f"/Dressing/stock/glove_{nm}{lvl}_{j}_{k}", (cx, yf + .0625, cz), (.24, .125, .065), white, bevel=.002)
                d.decal_quad(f"/Dressing/stock/glove_{nm}{lvl}_{j}_{k}_front",
                             [(cx - .118, yf - .0006, cz - .031), (cx + .118, yf - .0006, cz - .031),
                              (cx + .118, yf - .0006, cz + .031), (cx - .118, yf - .0006, cz + .031)], gl[sz])
        for k in range(4):                          # 1 L reagent bottles
            bx = x + .16 + .085*k
            d.cyl(f"/Dressing/stock/bottle_{nm}{lvl}_{k}", (bx, yf + .06, z + .09), .04, .18, hdpe)
            d.cyl(f"/Dressing/stock/bottle_cap_{nm}{lvl}_{k}", (bx, yf + .06, z + .197), .018, .034, cap_b)
    z = SHELF_TOPS[3]
    for k in range(3):                              # tube pack cartons
        cx = x - .30 + .21*k
        d.box(f"/Dressing/stock/tubepack_{nm}_{k}", (cx, yf + .08, z + .045), (.20, .16, .09), white, bevel=.002)
        d.decal_quad(f"/Dressing/stock/tubepack_{nm}_{k}_front",
                     [(cx - .098, yf - .0006, z + .002), (cx + .098, yf - .0006, z + .002),
                      (cx + .098, yf - .0006, z + .088), (cx - .098, yf - .0006, z + .088)], tp)
    z = SHELF_TOPS[0]
    cool = d.color("cooler_white", (.90, .91, .92), .35)
    lid = d.color("cooler_lid", (.10, .45, .75), .35)
    d.box(f"/Dressing/stock/cooler_{nm}", (x, yf + .15, z + .16), (.50, .30, .32), cool, bevel=.02)
    d.box(f"/Dressing/stock/cooler_lid_{nm}", (x, yf + .15, z + .33), (.52, .32, .03), lid, bevel=.01)


def part_props(d):
    d.ref("/Dressing/floor/cart", "industrial_storage_cart", (2.02, -.15, 0.), yaw=math.pi/2)


# ----------------------------------------------------------------------------- printer
def printer(d, fr, name):
    """Detailed desktop label printer (peel/present mode) over the physics colliders; housing size and class
    follow scene.printer (`kind`: direct_thermal | thermal_transfer, half-depth `hd`, height `ht`).
    A thermal-transfer printer shows its ribbon supply and take-up rolls through the media window.
    Local frame: u = label exit direction, v = lateral, z absolute; housing u in [-hd, hd]."""
    base, hw, exit_z = fr["base"], fr["hw"], fr["exit_z"]
    hd, ht, kind = fr.get("hd", .11), fr.get("ht", .17), fr.get("kind", "direct_thermal")
    root = f"/Dressing/{name}"
    d.xform(root, (fr["x"], fr["y"], 0.), fr["yaw"])
    shell = d.color("printer_shell", (.085, .088, .092), .45)
    lid = d.color("printer_lid", (.19, .195, .20), .55)
    dark = d.color("printer_trim", (.04, .04, .045), .5)
    window = d.color("printer_window", (.03, .035, .04), .04, opacity=.55)
    paper = d.color("roll_paper", (.95, .95, .93), .6)
    liner = d.color("liner_spent", (.93, .90, .72), .35)
    feet = d.color("rubber", (.03, .03, .03), .9)
    # base shell below the exit slot, clamshell lid above it (dark seam between), rounded front
    lo = exit_z - .0045 - base
    d.box(f"{root}/base", (-.003, 0, base + lo/2), (2*hd - .006, 2*hw, lo), shell, bevel=.008)
    hi = base + ht - .002 - (exit_z + .0045)
    d.box(f"{root}/lid", (-.005, 0, exit_z + .0045 + hi/2), (2*hd - .01, 2*hw - .004, hi), lid, bevel=.02)
    d.box(f"{root}/lid_crown", (-.02, 0, base + ht - .002), (2*hd - .05, 2*hw - .03, .012), lid, bevel=.006)
    d.box(f"{root}/seam", (0, 0, exit_z + .0045), (2*hd - .008, 2*hw + .001, .003), dark, bevel=.0005)
    # media window in the lid top with the label roll (and, thermal transfer, the ribbon) behind it
    d.box(f"{root}/window", (-.03, 0, base + ht + .0045), (2*hd - .12, 2*hw*.62, .002), window, bevel=.0008)
    rr = .055*ht/.17
    d.cyl(f"{root}/roll", (-.03*hd/.11, 0, base + ht*.62), rr, 2*(fr["head_width"]/2 + .004), paper, axis="Y")
    if kind == "thermal_transfer":
        ink = d.color("ribbon_ink", (.05, .05, .06), .3, .2)
        core = d.color("ribbon_core", (.85, .85, .83), .6)
        for k, (u, r_) in enumerate(((-.075, .022), (.045, .016))):          # ribbon supply, take-up
            d.cyl(f"{root}/ribbon_{k}", (u, 0, base + ht - .03), r_, fr["head_width"] + .006, ink, axis="Y")
            d.cyl(f"{root}/ribbon_core_{k}", (u, 0, base + ht - .03), .007, fr["head_width"] + .012, core, axis="Y")
    # front bezel around the exit: dark panel, printhead slot shadow
    d.box(f"{root}/bezel", (hd - .0035, 0, exit_z + .012), (.003, 2*hw - .03, .03), dark, bevel=.001)
    # keypad on the lid front edge: power, pause, feed, cancel, status LEDs
    key = d.color("key_grey", (.30, .31, .32), .5)
    led_g = d.color("led_green", (.05, .5, .1), .3, emissive=(0., 4., .6))
    led_o = d.color("led_off", (.2, .18, .1), .3)
    for k, v in enumerate((-.045, -.022, .001, .024)):
        d.cyl(f"{root}/key_{k}", (hd - .025, v - hw*.35, base + ht - .014), .007, .004, key, rot=(0, -25, 0))
    for k, v in enumerate((-.03, -.018, -.006)):
        d.cyl(f"{root}/led_{k}", (hd - .02, v + hw*.45, base + ht - .014), .0022, .003, led_g if k == 0 else led_o, rot=(0, -25, 0))
    # class badge on the lid front (above the keypad side)
    badge_key = {"thermal_transfer": "printer_badge_tt"}.get(kind, "printer_badge_2in" if fr["head_width"] < .07 else "printer_badge")
    badge = d.decal(badge_key, d.tex[badge_key]["diff"])
    u, zc = hd - .0075, base + ht - .035
    d.decal_quad(f"{root}/badge", [(u + .001, .06, zc - .010), (u + .001, -.02, zc - .010),
                                   (u + .001, -.02, zc + .010), (u + .001, .06, zc + .010)], badge, flip_u=True)
    # rubber feet
    for su in (-1, 1):
        for sv in (-1, 1):
            d.cyl(f"{root}/foot_{su+1}_{sv+1}", (su*(hd - .025), sv*(hw - .025), base + .002), .011, .004, feet)
    # spent liner: the backing web leaves under the peel bar and hangs down the front
    d.box(f"{root}/liner_tail", (hd + .008, 0, exit_z - .045), (.0003, 2*(fr["head_width"]/2) , .08), liner, bevel=.0001)
    # printer stand furniture (the grey riser top and side panels are the collider shapes): powder-coated
    # edge trims, a lower shelf between the side panels with spare label rolls, levelling feet
    if fr.get("riser"):
        bz, top = fr["bench_z"], fr["bench_z"] + fr["riser"]
        trim = d.color("stand_trim", (.16, .17, .18), .4, .3)
        shelf = d.color("stand_shelf", (.58, .59, .60), .45, .2)
        d.box(f"{root}/stand_front_trim", (.08, 0, top - .006), (.004, .322, .014), trim, bevel=.001)
        d.box(f"{root}/stand_back_trim", (-.16, 0, top - .006), (.004, .322, .014), trim, bevel=.001)
        sz = bz + .10
        d.box(f"{root}/stand_shelf", (-.04, 0, sz), (.20, .31, .006), shelf, bevel=.001)
        d.box(f"{root}/stand_shelf_lip", (.058, 0, sz + .012), (.004, .31, .02), trim, bevel=.001)
        rollw = d.color("spare_roll", (.95, .95, .93), .6)
        core = d.color("spare_core", (.55, .42, .28), .8)
        w = fr["head_width"] + .004
        for k, v in enumerate((-.08, .08)):
            d.cyl(f"{root}/stand_roll_{k}", (-.04, v, sz + .003 + .06), .06, w, rollw, axis="Y")
            d.cyl(f"{root}/stand_roll_core_{k}", (-.04, v, sz + .003 + .06), .038, w + .002, core, axis="Y")
        for a, su in enumerate((-.15, .07)):
            for b, sv in enumerate((-.15, .15)):
                d.cyl(f"{root}/stand_foot_{a}_{b}", (su, sv, bz + .003), .012, .006, feet)
    # power and USB cables to the back of the bench
    cable = d.color("cable", (.02, .02, .02), .6)
    for k, v in enumerate((-.03, .03)):
        pts = [(-.11, v, base + .03), (-.16, v, base + .02), (-.20, v + .02*k, base - .02), (-.22, v, fr["bench_z"] + .004)]
        _tube(d, f"{root}/cable_{k}", pts, .0028, cable)


def _tube(d, path, pts, r, mat, n=24, sides=10):
    """Swept circular tube through a Catmull-Rom spline of pts."""
    P = np.asarray(pts, float)
    P = np.vstack([P[0], P, P[-1]])
    curve = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t*t, t*t*t
            curve.append(.5*((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3))
    curve.append(P[-2])
    C = np.asarray(curve)
    verts, faces = [], []
    up = np.array([0., 0., 1.])
    for i, c in enumerate(C):
        tg = C[min(i + 1, len(C) - 1)] - C[max(i - 1, 0)]
        tg /= np.linalg.norm(tg) + 1e-12
        a = np.cross(tg, up if abs(tg @ up) < .9 else np.array([1., 0, 0])); a /= np.linalg.norm(a)
        b = np.cross(tg, a)
        for k in range(sides):
            th = 2*math.pi*k/sides
            verts.append(c + r*(math.cos(th)*a + math.sin(th)*b))
    for i in range(len(C) - 1):
        for k in range(sides):
            a0, a1 = i*sides + k, i*sides + (k + 1) % sides
            b0, b1 = a0 + sides, a1 + sides
            faces.append(np.array([a0, a1, b1, b0]))
    from whiteboard.usd_scene import mesh
    g = mesh(d.stage, path, np.asarray(verts), faces)
    d.bind(g, mat)
    return g


# ----------------------------------------------------------------------------- carton print
def old_label_print(d, scene, body_path, seed=0):
    """cover-old-label: the previous shipment's 4 x 3 in address label, printed, lying on the board (20 um) at its
    seeded position (layouts.old_label): the new label bonds over it. It has its own tracking barcode."""
    from . import artwork, layouts
    o = scene.task_geometry["old_label"]; g = scene.task_geometry["carton"]
    art = artwork.old_address_label(d.folder/"textures", layouts.old_label_payload(seed), seed=seed)
    m = _textured(d, "old_label_print", art["path"])
    (u0, u1), (v0, v1), z = o["u"], o["v"], g["H"]/2 + 2e-5
    from whiteboard.usd_scene import mesh
    pts = np.array([(u0, v0, z), (u1, v0, z), (u1, v1, z), (u0, v1, z)])
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)     # image x = the 4 in side (carton u), seen from above
    gq = mesh(d.stage, f"{body_path}/old_label", pts, [np.array([0, 1, 2, 3])], uv)
    d.bind(gq, m)


def plate_marking(d, scene, name, body_path):
    """part-id-match: the plate's laser-annealed part number on its upper face (visual only; flush with the steel)."""
    from . import artwork
    it = scene.items[name]
    mk = it["marking"]
    path = artwork.plate_marking(d.folder/"textures", it["part_number"], name=f"{name}_marking")
    m = d.decal(f"{name}_marking", path)
    (u0, u1), (v0, v1), z = mk["u"], mk["v"], it["size"][2]/2 + 2e-5
    flip = mk["v"][0] < 0                                 # text reads from the plate's nearer long edge
    pts = [(u0, v0, z), (u1, v0, z), (u1, v1, z), (u0, v1, z)] if not flip else \
          [(u1, v1, z), (u0, v1, z), (u0, v0, z), (u1, v0, z)]
    d.decal_quad(f"{body_path}/marking", pts, m)


def carton_markings(d, scene, body_path, seed=0):
    g = scene.task_geometry.get("carton")
    if not g:
        return
    if "old_label" in scene.task_geometry:
        old_label_print(d, scene, body_path, seed=seed)
    L, W, H = g["L"], g["W"], g["H"]
    m = d.decal("carton_print", d.tex["carton_print"]["diff"])
    y = -W/2 - .0001
    w = min(L*.8, 2*H*.8*2)
    hh = w/2
    zc = -H/2 + H*.45
    d.decal_quad(f"{body_path}/print_front", [(-w/2, y, zc - hh/2), (w/2, y, zc - hh/2), (w/2, y, zc + hh/2), (-w/2, y, zc + hh/2)], m)
    x = L/2 + .0001
    w2 = W*.8; h2 = w2/2
    d.decal_quad(f"{body_path}/print_end", [(x, -w2/2, zc - h2/2), (x, w2/2, zc - h2/2), (x, w2/2, zc + h2/2), (x, -w2/2, zc + h2/2)], m)


# ----------------------------------------------------------------------------- task items
def _cyl_shell(d, path, r, z0, z1, a0, a1, mat, n=48):
    """Partial cylindrical shell about local z (angles in rad), UV = (angle fraction, z fraction). Faces are wound so
    their normals point out of the cylinder whichever way a0 -> a1 runs (the manufacturer label runs
    2.1 -> -2.1 rad); the texture coordinates at each vertex are unchanged."""
    from whiteboard.usd_scene import mesh
    ang = np.linspace(a0, a1, n)
    verts, uv, faces = [], [], []
    for i, a in enumerate(ang):
        for j, zz in enumerate((z0, z1)):
            verts.append((r*math.cos(a), r*math.sin(a), zz))
    for i in range(n - 1):
        a, b, c, e = 2*i, 2*i + 2, 2*i + 3, 2*i + 1
        fu0, fu1 = i/(n - 1), (i + 1)/(n - 1)
        corners = [(a, (fu0, 1)), (b, (fu1, 1)), (c, (fu1, 0)), (e, (fu0, 0))]
        if a1 < a0:                                         # angle falling along the strip: reverse the winding
            corners = corners[:1] + corners[:0:-1]
        faces.append(np.array([k for k, _ in corners]))
        uv.extend(t for _, t in corners)
    g = mesh(d.stage, path, np.asarray(verts), faces, np.asarray(uv))
    d.bind(g, mat)
    return g


def tube_visual(d, sc, body_path, down_local=(-1., 0., 0.)):
    """Evacuated specimen tube: smooth PET wall, K2EDTA blood, manufacturer label over the collider's label
    patch (0..240 deg, 18..75 mm from the cap end). Local z = tube axis, cap at +z.

    The tube lies on its rest, so the blood pools along its length: a circular-segment prism under a
    horizontal free surface (`down_local`: world down in the tube frame at export), filling the tube's
    6.0 mL draw volume (its manufacturer label) out of the interior (0.7 mm PET wall, stopper 12 mm deep,
    rounded bottom 1.5 mm). Visual only; the tube's mass is set in scene.py."""
    from whiteboard.appearance import bind_glass
    from whiteboard.usd_scene import mesh
    g = sc.task_geometry["tube"]
    r, L = g["r"], g["L"]
    glass = d.UsdShade.Material.Define(d.stage, "/DressMaterials/tube_pet")
    bind_glass(d.stage, glass)
    d.mats["tube_pet"] = glass
    blood = d.color("blood", (.30, .02, .03), .15)
    d.cyl(f"{body_path}/tube_glass", (0, 0, 0), r, L, glass)
    ri = r - .0007
    z0, z1 = -L/2 + .0015, L/2 - .012
    frac = min(SPECIMEN_DRAW_ML*1e-6/(math.pi*ri*ri*(z1 - z0)), .95)
    lo, hi = 0., math.pi                               # segment half-angle a: (a - sin a cos a)/pi = frac
    for _ in range(60):
        a = .5*(lo + hi)
        lo, hi = (a, hi) if (a - math.sin(a)*math.cos(a))/math.pi < frac else (lo, a)
    dn = np.asarray(down_local, float); dn[2] = 0.
    dn = dn/np.linalg.norm(dn) if np.linalg.norm(dn) > 1e-9 else np.array([-1., 0., 0.])
    sd = np.array([-dn[1], dn[0], 0.])
    ring = [ri*(math.cos(th)*dn + math.sin(th)*sd) for th in np.linspace(-a, a, 25)]
    verts, faces = [], []
    for z in (z0, z1):
        verts += [p + np.array([0., 0., z]) for p in ring]
    m = len(ring)
    for k in range(m - 1):                              # curved wetted wall
        faces.append(np.array([k, k + 1, m + k + 1, m + k]))
    faces.append(np.array([m - 1, 0, m, 2*m - 1]))      # flat free surface (chord)
    faces.append(np.arange(m)[::-1]); faces.append(np.arange(m, 2*m))
    gm = mesh(d.stage, f"{body_path}/tube_blood", np.asarray(verts), faces)
    d.bind(gm, blood)
    lab = _textured(d, "tube_mfr_label", d.tex["tube_label"]["diff"])
    # drawn 10 um OUTSIDE the PET wall, where a paper label is: drawn 10 um under it, inside the solid refractive
    # glass, it renders black from above, the policy cameras' view. An applied patient label's back sheet lies at r and its
    # printed front at r + its 0.10 mm thickness, so the patient label still hides this shell where it covers it.
    # Visual only: the label's collider patch (scene.specimen) is unchanged.
    _cyl_shell(d, f"{body_path}/tube_mfr_label", r + .00001, L/2 - .075, L/2 - .018, 2.1, -2.1, lab)


def canister_visual(d, sc, body_path):
    """Round HDPE canister (the part collider is its cylinder): screw lid and a moulded base ring."""
    it = sc.items["part"]
    r, h = it["radius"], it["height"]
    lid = d.color("canister_lid", (.12, .30, .62), .45)
    ring = d.color("hdpe_ring", (.88, .88, .86), .5)
    d.cyl(f"{body_path}/lid", (0, 0, h/2 + .011), r + .002, .022, lid)
    d.cyl(f"{body_path}/lid_grip", (0, 0, h/2 + .011), r + .0035, .014, lid)
    d.cyl(f"{body_path}/shoulder_ring", (0, 0, h/2 - .012), r + .0012, .004, ring)
    d.cyl(f"{body_path}/base_ring", (0, 0, -h/2 + .006), r + .0012, .004, ring)


# ----------------------------------------------------------------------------- policy camera mount
def camera_mount(d, eye):
    """Ceiling drop rod + clamp + camera body for the overhead policy camera (external mount)."""
    steel = d.color("unistrut", (.55, .56, .58), .35, .8)
    cam = d.color("camera_body", (.06, .06, .07), .4)
    h = ROOM["h"]
    x, y, z = eye
    d.cyl("/Dressing/camera_mount/rod", (x, y, (z + h)/2 + .03), .012, h - z - .06, steel)
    d.box("/Dressing/camera_mount/plate", (x, y, h - .005), (.12, .12, .01), steel, bevel=.002)
    d.box("/Dressing/camera_mount/body", (x, y, z + .02), (.09, .03, .025), cam, bevel=.004)


def dress(stage, world, folder, eye_policy=None):
    """Author the dressing into `stage`. Returns the list of pinned model names used (credits)."""
    from . import scene as scene_mod
    d = Dresser(stage, folder)
    preset = world.config.preset
    sc = world.scene
    room(d, preset)
    if preset != "specimen":
        pack_props(d, floor_boxes=preset != "part")
    if preset == "specimen":
        lab_props(d, scene_mod)
        # lab consumables on the shelving (a clinical bench stocks gloves, tubes, reagents, transport
        # coolers; not shipping cartons)
        R = ROOM
        for nm, x in (("a", -1.60), ("b", 1.62)):
            d.ref(f"/Dressing/stock/shelf_{nm}", "steel_frame_shelves_01", (x, R["y1"] - .27, 0.))
            lab_shelf_stock(d, nm, x, R["y1"] - .27)
        # white step-on waste bin (appearance only)
        white = d.color("bin_white", (.88, .88, .86), .4)
        pedal = d.color("bin_pedal", (.2, .2, .2), .5)
        d.cyl("/Dressing/floor/bin_body", (-1.30, -.70, .33), .17, .66, white)
        d.cyl("/Dressing/floor/bin_lid", (-1.30, -.70, .67), .175, .025, white)
        d.box("/Dressing/floor/bin_pedal", (-1.30, -.90, .02), (.12, .06, .02), pedal, bevel=.004)
    if preset == "part":
        part_props(d)
    bench_props(d, preset, scene_mod)
    for name, fr in getattr(sc, "printers", {}).items():
        printer(d, dict(fr, bench_z=scene_mod.TABLE), name)
    if "carton" in sc.bodies:
        carton_markings(d, sc, f"/Lab/body_{sc.bodies['carton']}", seed=world.config.seed)
    for name, it in sc.items.items():
        if it.get("kind") == "steel_plate" and "marking" in it:
            plate_marking(d, sc, name, f"/Lab/body_{sc.bodies[name]}")
    if "tube" in sc.bodies:
        _, Rt = world.body_pose("tube")
        tube_visual(d, sc, f"/Lab/body_{sc.bodies['tube']}", down_local=Rt.T @ np.array([0., 0., -1.]))
    if "part" in sc.bodies and sc.items["part"]["kind"] == "hdpe_jug":
        canister_visual(d, sc, f"/Lab/body_{sc.bodies['part']}")
    if eye_policy is not None:
        camera_mount(d, eye_policy)
    return sorted(d.credits)

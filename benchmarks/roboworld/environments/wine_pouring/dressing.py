"""Visual-only dressing of the wine bar (no collision, no physics): room shell, back bar, stools, pendant lamps, props.

Everything a robot or the liquid can touch is built in scene.py. Models and textures are pinned CC0 Poly Haven assets
(visual_assets.py). Lighting: three pendant lamps over the bar (3500 K sphere lights) whose luminance is set so the
bar-top illuminance at the work area is 500 lx (EN 12464-1's food-preparation value; its bar/buffet value is 300 lx
[S40]: a robot work area is lit like a kitchen pass), a back-bar wash and a weak ambient dome.
The Dresser class (asset references re-bound to OmniPBR) follows barcode_scanning/dressing.py.
"""

import math
import re
import shutil
from pathlib import Path
import numpy as np

from . import params as P
from . import visual_assets as va
from .scene import BAR_Y0, BAR_Y1, BAR_TOP

ROOM = P.ROOM
TARGET_LUX = 500.
PENDANTS = ((-0.85, 0.25), (0.0, 0.25), (0.85, 0.25))
PENDANT_SCALE = 0.75                    # hanging_industrial_lamp: 1.34 m from ceiling hook to shade rim, 0.55 m shade
PENDANT_Z = 3.1 - PENDANT_SCALE*1.34 - 0.03   # lamp just below the shade rim (the model is one opaque mesh: a light
                                        # inside it was occluded), ~1.0 m above the bar top
PENDANT_RADIUS = 0.05


class Dresser:
    def __init__(self, S):
        self.S, self.stage = S, S.stage
        self.credits = set()
        self.folder = S.folder
        self.tex = self.folder/"textures"

    def pbr(self, name, tex_name, tile_m, roughness=.6):
        maps = va.texture(tex_name)
        self.credits.add(tex_name)
        return self.S.textured(name, maps, tile_m, roughness)

    def quad_uv(self, path, corners, mat):
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n = np.cross(e1, e2); n /= np.linalg.norm(n)
        uv = np.array([[0, 0], [l1, 0], [l1, l2], [0, l2]])
        return self.S.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    def ref(self, path, asset, pos, yaw=0., scale=None):
        """Reference a pinned model (visual only), re-bound to OmniPBR (RTX does not compile the models' MaterialX)."""
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


def pendant_luminance(reference=(0.0, 0.25, BAR_TOP)):
    """Sphere-light luminance (nit) giving TARGET_LUX on the bar top below the middle pendant: a Lambertian sphere of
    radius a and luminance L gives E = L pi a^2 cos / r^2 at distance r (summed over the three lamps)."""
    ref = np.asarray(reference, float)
    g = 0.
    for x, y in PENDANTS:
        v = np.array([x, y, PENDANT_Z]) - ref
        r = float(np.linalg.norm(v))
        g += math.pi*PENDANT_RADIUS**2*max(0., v[2]/r)/r**2
    return TARGET_LUX/g


def _dark_stone(d, S):
    """The bar top: engineered quartz, one continuous slab (quartz bar tops [S10]). A generated, tileable speckle
    texture (grey matrix with dark and light aggregate, as composite quartz looks): the marble scan showed its tile
    joints every 40 cm across the bar."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(11)
    n = 2048
    base = rng.normal(0., 1., (n, n)).astype(np.float32)
    low = np.asarray(Image.fromarray(((base - base.min())/np.ptp(base)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24)), np.float32)/255.
    img = 0.30 + 0.012*(low - low.mean())/max(low.std(), 1e-6)
    for frac, val, size in ((0.010, 0.10, 2), (0.006, 0.55, 1), (0.003, 0.05, 3)):
        k = int(frac*n*n)
        ys, xs = rng.integers(0, n, k), rng.integers(0, n, k)
        for dy in range(size):
            for dx in range(size):
                img[(ys + dy) % n, (xs + dx) % n] = val
    img = np.asarray(Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), np.float32)/255.
    rgb = np.stack([img*1.0, img*0.98, img*0.95], -1)
    p = d.tex/"bar_quartz_diff.png"
    Image.fromarray((rgb*255).astype(np.uint8)).save(p)
    # satin (honed) finish, roughness 0.30: with the polished 0.15 the centre pendant's bulb reflected as a saturated
    # highlight in the middle of the policy camera, where bottle and coaster stand
    return S.textured("bar_quartz", {"diff": p}, 0.35, .30)


def _back_bar(S, wood, y0):
    """Back-bar shelving with this scene's own bottles (LEONOR and BG CLASSIQUE 2 meshes, green and flint glass,
    labels) and inverted Bordeaux glasses: the same models as the physical glassware, instanced, visual only."""
    from . import geometry as GEO
    from .render import _label_texture
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    protos = {}
    for kind, color in (("bottle_leonor", "glass_green"), ("bottle_leonor", "glass_flint"), ("bottle_burgundy", "glass_green")):
        geo = GEO.get(kind)
        v, t, vn = geo.render_mesh_shaded(64)
        path = f"/Dressing/protos/{kind}_{color}"
        S.xform(path)
        S.mesh(path + "/wall", v, t, normals=vn, mat=S.mats[color])
        liq = "white_wine" if color == "glass_flint" else "red_wine"
        lv, lt = geo.interior_mesh(geo.lines["fill_750"], 48)
        S.mesh(path + "/wine", lv, lt, mat=S.mats[f"liquid_{liq}"], smooth=True)
        tex = _label_texture(S.folder/"textures", liq, len(protos))
        lab = S.mdl(f"backbar_label_{len(protos)}", "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))),
                                                              "reflection_roughness_constant": (T.Float, .75)})
        r = float(geo.solid[:, 0].max()) + 0.0004
        vv, uv, ff, nn = [], [], [], []
        for i, a in enumerate(np.linspace(-0.9, 0.9, 25)):
            dd = np.array([math.sin(a), -math.cos(a), 0.])
            for j, z in enumerate((0.06, 0.14)):
                vv.append(dd*r + np.array([0., 0., z])); uv.append((i/24, j)); nn.append(dd)
        for i in range(24):
            q0, q1, q2, q3 = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            ff += [q0, q2, q3, q0, q3, q1]
        S.mesh(path + "/label", vv, ff, uv=uv, normals=nn, mat=lab)
        S.UsdGeom.Imageable(S.stage.GetPrimAtPath(path)).MakeInvisible()
        protos[(kind, color)] = path
    gv, gt, gn = GEO.get("glass_pure").render_mesh_shaded(64)
    S.xform("/Dressing/protos/glass")
    S.mesh("/Dressing/protos/glass/wall", gv, gt, normals=gn, mat=S.mats["crystal"])
    S.UsdGeom.Imageable(S.stage.GetPrimAtPath("/Dressing/protos/glass")).MakeInvisible()
    rng = np.random.default_rng(7)
    keys = list(protos)
    shelves = (1.30, 1.72)
    for s_i, z in enumerate(shelves):
        S.box(f"/Dressing/backbar/shelf_{s_i}", (3.0, 0.24, 0.025), wood, (0., y0 + 0.14, z - 0.0125), (0, 0, 0, 1))
        for k, x in enumerate(np.arange(-1.40, 1.41, 0.095)):
            if abs(x) < 0.05:
                continue
            key = keys[int(rng.integers(0, len(keys)))]
            inst = S.xform(f"/Dressing/backbar/bottle_{s_i}_{k}", (float(x), y0 + 0.14, z), (0., 0., math.sin(rng.uniform(-0.4, 0.4)), 1.))
            inst.GetPrim().GetReferences().AddInternalReference(protos[key])
            S.UsdGeom.Imageable(inst.GetPrim()).MakeVisible()
    for k, x in enumerate(np.arange(-1.35, 1.36, 0.11)):
        for row, y in enumerate((y0 + 0.17, y0 + 0.30, y0 + 0.43)):
            inst = S.xform(f"/Dressing/backbar/glass_{row}_{k}", (float(x + 0.05*row), y, 0.95 + 0.267), (1., 0., 0., 0.))
            inst.GetPrim().GetReferences().AddInternalReference("/Dressing/protos/glass")
            S.UsdGeom.Imageable(inst.GetPrim()).MakeVisible()


def _delivery_spots(S, d):
    """The four numbered delivery spots a task names ("spot 2"): an illuminated ring inset in the bar top, 55 mm radius
    (the evaluator's tolerance), and its number engraved toward the robot side, as Makr Shakr's Toni marks its
    illuminated delivery positions [S20]. Visual only (0.3 mm proud of the bar top, no collision): the policy camera
    needs them to know which glass position a task means."""
    from PIL import Image, ImageDraw, ImageFont
    ring = S.color("spot_ring", (1.0, 0.86, 0.62), .3, 0., emissive=(1.0, 0.80, 0.55), emissive_intensity=400.)
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    r_out, r_in, n = P.SPOT_RADIUS_M, P.SPOT_RADIUS_M - 0.004, 64
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    for k, (x, y) in P.DELIVERY_SPOTS.items():
        z = BAR_TOP + 0.0003
        v = [(x + r*math.cos(t), y + r*math.sin(t), z) for r in (r_in, r_out) for t in th]
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [i, n + i, n + j, i, n + j, j]
        S.mesh(f"/Dressing/spots/ring_{k}", v, f, normals=[(0., 0., 1.)]*len(v), mat=ring)
        im = Image.new("RGB", (128, 128), (40, 38, 36))
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
        except OSError:
            font = ImageFont.load_default()
        dr.text((64, 64), str(k), font=font, fill=(235, 220, 190), anchor="mm")
        pth = d.tex/f"spot_{k}.png"
        im.save(pth)
        mat = S.mdl(f"spot_digit_{k}", "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(pth.resolve()))),
                                                   "reflection_roughness_constant": (T.Float, .5)})
        c = np.array([x, y - r_out - 0.016, z])
        hs = 0.011
        # the digit reads upright from the robot side (y < 0): its top edge toward the guests
        S.mesh(f"/Dressing/spots/digit_{k}", [c + (-hs, -hs, 0), c + (hs, -hs, 0), c + (hs, hs, 0), c + (-hs, hs, 0)],
               [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[(0., 0., 1.)]*4, mat=mat)


def dress(S, world):
    d = Dresser(S)
    R = ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    floor = d.pbr("floor_planks", "dark_wooden_planks", 2.0, .55)
    brick = d.pbr("brick_back", "brick_wall_10", 2.5, .85)
    plaster = d.pbr("plaster", "grey_plaster", 3.0, .9)
    marble = _dark_stone(d, S)
    walnut = d.pbr("bar_walnut", "natural_walnut_veneer", 1.0, .45)
    d.quad_uv("/Dressing/room/floor", [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)], floor)
    # first edge horizontal: the texture's U (brick length) runs along the wall (the earlier corner order stood the
    # bricks on end); normal +y, into the room
    d.quad_uv("/Dressing/room/back", [(x1, y0, 0.), (x0, y0, 0.), (x0, y0, h), (x1, y0, h)], brick)
    d.quad_uv("/Dressing/room/front", [(x0, y1, 0.), (x0, y1, h), (x1, y1, h), (x1, y1, 0.)], plaster)
    d.quad_uv("/Dressing/room/left", [(x0, y0, 0.), (x0, y1, 0.), (x0, y1, h), (x0, y0, h)], plaster)
    d.quad_uv("/Dressing/room/right", [(x1, y0, 0.), (x1, y0, h), (x1, y1, h), (x1, y1, 0.)], plaster)
    d.quad_uv("/Dressing/room/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)],
              S.color("ceiling_dark", (.10, .09, .085), .9, 0.))
    # re-bind the bar top and die (static boxes built in scene.py) to their real finishes
    for prim in S.stage.Traverse():
        name = prim.GetName()
        if not name.startswith("static_"):
            continue
        idx = int(name.split("_")[1])
        lab = world.scene.b.shape_label[idx]
        if lab == "bar_top":
            S.bind(S.UsdGeom.Mesh(prim), marble)
        elif lab in ("bar_die", "bar_back_panel"):
            S.bind(S.UsdGeom.Mesh(prim), walnut)
    # footrail on the guest side (brass tube look) and a back counter with shelving against the brick wall
    S.box("/Dressing/bar/footrail", (P.BAR_LENGTH_M - 0.1, 0.05, 0.05), S.color("brass", (.78, .58, .30), .25, 1.),
          (0., BAR_Y1 + 0.02, 0.23), (0, 0, 0, 1), bevel=.02)
    S.box("/Dressing/backbar/counter", (3.2, 0.55, 0.92), walnut, (0., y0 + 0.30, 0.46), (0, 0, 0, 1))
    S.box("/Dressing/backbar/top", (3.24, 0.58, 0.03), marble, (0., y0 + 0.30, 0.935), (0, 0, 0, 1), uv_per_m=1.)
    _back_bar(S, walnut, y0)
    if world.config.preset in ("service", "lined"):           # the tasting class does not use delivery spots
        _delivery_spots(S, d)
    # guest side: stools, candles; room: barrel, plant, picture
    # stools 0.65 m apart centre to centre (bar seating is laid out at 0.6-0.75 m per stool); the model is 0.48 x 0.49 m,
    # and the earlier 0.34 m pitch made neighbours interpenetrate by 15 cm
    for k, x in enumerate((-0.975, -0.325, 0.325, 0.975), start=1):
        d.ref(f"/Dressing/guests/stool_{k}", "bar_chair_round_01", (x, BAR_Y1 + 0.42, 0.), yaw=math.pi)
    d.ref("/Dressing/backbar/candles", "brass_candleholders", (1.95, y0 + 0.32, 0.95), scale=0.6)
    d.ref("/Dressing/room/barrel", "wine_barrel_01", (-2.55, 0.6, 0.), yaw=0.4)
    d.ref("/Dressing/room/plant", "potted_plant_02", (2.35, 0.9, 0.))
    d.ref("/Dressing/room/picture", "fancy_picture_frame_01", (2.35, y0 + 0.03, 1.75), yaw=math.pi)
    # lighting
    L = pendant_luminance()
    for k, (x, y) in enumerate(PENDANTS):
        d.ref(f"/Dressing/lights/pendant_{k}", "hanging_industrial_lamp", (x, y, R["h"]), scale=PENDANT_SCALE)
        lt = S.UsdLux.SphereLight.Define(S.stage, f"/Dressing/lights/pendant_light_{k}")
        lt.CreateRadiusAttr(PENDANT_RADIUS); lt.CreateIntensityAttr(float(L))
        lt.CreateEnableColorTemperatureAttr(True); lt.CreateColorTemperatureAttr(3500.)
        S.UsdGeom.Xformable(lt).AddTranslateOp().Set(S.Gf.Vec3d(x, y, PENDANT_Z))
    wash = S.UsdLux.RectLight.Define(S.stage, "/Dressing/lights/backbar_wash")
    wash.CreateWidthAttr(3.0); wash.CreateHeightAttr(0.05); wash.CreateIntensityAttr(float(L)*0.15)
    wash.CreateEnableColorTemperatureAttr(True); wash.CreateColorTemperatureAttr(3000.)
    S.UsdGeom.Xformable(wash).AddTranslateOp().Set(S.Gf.Vec3d(0., y0 + 0.12, 2.25))
    dome = S.UsdLux.DomeLight.Define(S.stage, "/Dressing/lights/ambient")
    dome.CreateIntensityAttr(40.)
    return d.credits

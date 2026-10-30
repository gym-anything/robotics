"""Visual-only dressing of the teaching kitchen (no collision, no physics).

Everything a robot, a tool or the mixture can touch is built in scene.py / tools.py. This file draws
  * the static collision boxes of scene.py in full: the quartz top, the base cabinets with door and drawer fronts,
    handles and toe kick, the tiled back wall, the wooden wall shelf on its brackets, the hob's glass with its printed
    zone ring and knob scale, and the cart as a welded frame (deck plate, posts on casters, rails, lower shelf with the
    two controllers);
  * the room fittings within reach, which scene.fittings gives static collision proxies: a sink unit to the left of the
    counter run, a wall rail with a towel, canisters, jars and a stack of bowls on the shelf, socket plates;
  * a fridge beyond the sink and a plant by the window (static proxies: scene.furniture), and what stands around them
    and has no physics: ceiling with six LED panels, an under-shelf LED strip, a recipe card clipped to the tiles, the
    station camera on its wall bracket and the overhead camera's housing under the shelf;
  * the room itself, its furniture and what stands on the counter for the task's recipe: kitchen.py (called here).

Where the dressing differs from the physics, on purpose and stated:
  * the sink unit with its tap and bottle, the towel rail with its towel, the canisters, jars and prep bowls on the
    shelf, the shelf's brackets and light strip and the socket plates have static collision proxies of their drawn outlines (scene.fittings, from the same numbers in params): boxes, so
    a round canister's proxy is the square around it, and nothing on the shelf can be knocked over;
  * fronts and handles stand 18 mm and 46 mm in front of the collision carcass, under the top's 25 mm overhang;
  * a post's caster is drawn where the collision post runs straight to the floor;
  * the fridge and the plant are out of reach of an arm and of a tool held at full stretch (1.26 m from an arm base);
    scene.furniture gives the fridge's cabinet, doors, handles and feet their boxes and the plant's pot a stack of
    cylinders (params.FRIDGE, PLANT_POT); the plant's leaves have no collider (a thrown thing passes through them);
    the recipe card lies flat on the tiles, which are its collider.

Lighting: six 4000 K LED panels in the ceiling whose luminance gives 400 lx on the counter top at the work zone and
an under-shelf strip that adds 100 lx, 500 lx together (EN 12464-1 names 500 lx for kitchens' working areas; the same
target as wine_pouring and drilling), plus a weak ambient dome. Textures and models are pinned CC0 Poly Haven assets,
borrowed from the sibling packages' pinned caches (this package has no visual_assets manifest of its own); the tile,
quartz, towel and recipe-card textures are generated here.
"""

import importlib
import math
import re
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import kitchen as KIT
from . import geometry as GEO

CZ = P.COUNTER_Z
TARGET_LUX = 500.
STRIP_SHARE = 0.20
PANELS = [(x, y) for x in (-1.3, 0.3, 1.9) for y in (-0.60, -2.35)]      # ceiling panels over the room (ROOM in params)
PANEL_SIZE = (0.86, 0.58)
KELVIN = 4000.
# the lamps' colour as a camera with auto white balance records it: 4000 K (linear sRGB 1.53 : 1 : 0.577) corrected to
# the power 0.85, i.e. the remaining 15 % of the cast, at unit luminance (render_worker.py: why it is put here)
LIGHT_RGB = (1.057, 0.992, 0.913)
TILE_TOP = 1.80                       # the wall is tiled to here; painted plaster above
SINK_X = P.SINK["x"]                  # the sink unit beside the counter run (collision proxies: scene.fittings)
FRIDGE_X = P.FRIDGE["x"]
PINNED = ("barcode_scanning", "drilling", "wine_pouring", "gift_bags")     # packages whose pinned Poly Haven caches are used
TEXTURES = dict(floor_tiles_06=2.0, beige_wall_002=2.0, white_oak_veneer=1.0)   # name -> tile size (m)
RECIPE_FOR = {"dry-whisk": ("dry_mix",), "muffin-fold": ("dry_mix", "muffin_wet"), "pancake-whisk": ("pancake_dry", "pancake_wet"),
              "pudding-whisk": ("pudding",), "cake-mixer": ("cake",), "sauce-blend": ("sauce",), "stir-sauce": ("stir_sauce",),
              "reheat-porridge": ("porridge",), "stir-drink": ("lemonade",)}
RECIPE_VARIANT = {"honey": ("sauce_honey",), "pet-meal": ("pet_meal",)}


def _pinned(kind, name):
    """A pinned Poly Haven asset from a sibling package's manifest (hash-checked there)."""
    for pkg in PINNED:
        va = importlib.import_module(f"{pkg}.visual_assets")
        if name in (va.TEXTURES if kind == "texture" else va.MODELS):
            return va.texture(name) if kind == "texture" else va.model(name)
    raise KeyError(f"{kind} {name} is not pinned in {PINNED}")


class Dresser:
    def __init__(self, S):
        self.S, self.stage = S, S.stage
        self.credits = set()
        self.folder = S.folder
        self.tex = self.folder/"textures"

    def pbr(self, name, tex_name, tile_m, roughness=.6, tint=None, bump=.4):
        self.credits.add(tex_name)
        return self.S.textured(name, _pinned("texture", tex_name), tile_m, roughness, tint=tint, bump=bump)

    def ref(self, path, asset, pos, yaw=0., scale=None, prims=None):
        """Reference a pinned model (visual only), re-bound to OmniPBR (RTX does not compile the models' MaterialX)."""
        from pxr import Usd
        usd = _pinned("model", asset)
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
        if prims is not None:
            for child in inner.GetPrim().GetChildren():
                if child.GetName() not in prims and child.GetName() != "_materials":
                    child.SetActive(False)
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


# ====================================================================================== generated textures
def _tile_maps(folder, tile=(0.200, 0.100), grout=0.0025, cover=(0.400, 0.400), px=1024, seed=5):
    """White glazed wall tiles 200 x 100 mm in running bond with 2.5 mm light-grey grout: diffuse, roughness and
    (OpenGL) normal maps covering `cover` metres, tileable. The glaze is glossy and each tile's face is slightly
    pillowed into the joint, so the joints catch the light as on a real wall."""
    from PIL import Image
    path = {k: Path(folder)/f"wall_tile_{k}.png" for k in ("diff", "rough", "nor_gl")}
    if all(p.exists() for p in path.values()):
        return path
    rng = np.random.default_rng(seed)
    ppm = px/cover[0]
    y, x = np.mgrid[0:px, 0:px]
    u, v = (x + 0.5)/ppm, (px - y - 0.5)/ppm                      # metres, v up
    row = np.floor(v/tile[1]).astype(int)
    uo = u + 0.5*tile[0]*(row % 2)
    col = np.floor(uo/tile[0]).astype(int) % int(round(cover[0]/tile[0]))
    fu, fv = uo - np.floor(uo/tile[0])*tile[0], v - row*tile[1]
    edge = np.minimum(np.minimum(fu, tile[0] - fu), np.minimum(fv, tile[1] - fv))      # distance to the tile's edge
    face = np.clip((edge - 0.5*grout)/0.0018, 0., 1.)
    face = face*face*(3 - 2*face)                                 # 0 in the joint, 1 on the glazed face
    tone = 1.0 + 0.012*rng.standard_normal((int(round(cover[1]/tile[1])), int(round(cover[0]/tile[0]))))
    tone = tone[row % tone.shape[0], col]
    glaze = np.array([0.905, 0.905, 0.885])                       # sRGB of the glaze; the joint a light cement grey
    joint = np.array([0.60, 0.60, 0.58])
    rgb = joint + (glaze*tone[..., None] - joint)*face[..., None]
    rgb += 0.004*rng.standard_normal(rgb.shape[:2])[..., None]
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(path["diff"])
    rough = 0.85 + (0.10 - 0.85)*face
    Image.fromarray((np.clip(rough, 0, 1)*255).astype(np.uint8)).save(path["rough"])
    height = 0.0009*face                                          # the face stands 0.9 mm above the joint
    gy, gx = np.gradient(height, 1.0/ppm)                         # per metre; rows run downward (v up = -row)
    nrm = np.dstack([-gx, gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray(((nrm*0.5 + 0.5)*255).astype(np.uint8)).save(path["nor_gl"])
    return path


def _quartz_map(folder, px=1536, seed=11):
    """Pale engineered quartz, one continuous slab: a warm white matrix with fine grey and translucent aggregate
    (tileable; the grains are 0.3-1.5 mm at the 0.40 m tile the material uses)."""
    from PIL import Image, ImageFilter
    p = Path(folder)/"quartz_pale_diff.png"
    if p.exists():
        return {"diff": p}
    rng = np.random.default_rng(seed)
    base = rng.normal(0., 1., (px, px)).astype(np.float32)
    low = np.asarray(Image.fromarray(((base - base.min())/np.ptp(base)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28)), np.float32)/255.
    img = 0.845 + 0.012*(low - low.mean())/max(low.std(), 1e-6)
    for frac, val, size in ((0.012, 0.70, 2), (0.006, 0.93, 2), (0.003, 0.55, 3), (0.0008, 0.42, 4)):
        k = int(frac*px*px)
        ys, xs = rng.integers(0, px, k), rng.integers(0, px, k)
        for dy in range(size):
            for dx in range(size):
                img[(ys + dy) % px, (xs + dx) % px] = val
    img = np.asarray(Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32)/255.
    rgb = np.stack([img*1.0, img*0.985, img*0.955], -1)
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(p)
    return {"diff": p}


def _towel_map(folder, px=512):
    """A cotton tea towel: off-white with two groups of blue stripes along its length and a fine woven grid."""
    from PIL import Image
    p = Path(folder)/"tea_towel_diff.png"
    if p.exists():
        return p
    y, x = np.mgrid[0:px, 0:px]
    u = x/px
    rgb = np.ones((px, px, 3))*np.array([0.90, 0.89, 0.85])
    weave = 0.03*(np.sin(2*math.pi*x/4.0)*np.sin(2*math.pi*y/4.0))
    rgb += weave[..., None]
    for c, w in ((0.12, 0.035), (0.19, 0.012), (0.81, 0.012), (0.88, 0.035)):
        m = np.abs(u - c) < 0.5*w
        rgb[m] = np.array([0.16, 0.30, 0.52]) + weave[m][:, None]
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(p)
    return p


def _recipe_map(folder, world):
    """The recipe card (the scene's only text): the task's title, its ingredients in grams from params.RECIPES and its
    instruction, set like a printed teaching-kitchen card."""
    from PIL import Image, ImageDraw, ImageFont
    from . import task as T
    cfg = world.config
    spec = T.task_spec(cfg)
    W, H = 1050, 1490
    im = Image.new("RGB", (W, H), (244, 241, 230))
    d = ImageDraw.Draw(im)

    def font(size, bold=False):
        f = f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{'-Bold' if bold else ''}.ttf"
        return ImageFont.truetype(f, size) if Path(f).exists() else ImageFont.load_default()

    def wrap(text, f, width):
        out, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if cur and d.textlength(trial, font=f) > width:
                out.append(cur)
                cur = word
            else:
                cur = trial
        return out + ([cur] if cur else [])

    d.rectangle([0, 0, W, 150], fill=(150, 52, 40))
    d.text((60, 75), "TEACHING KITCHEN  ·  STATION RECIPE", font=font(40, True), fill=(250, 244, 230), anchor="lm")
    y = 200
    for ln in wrap(spec["title"], font(74, True), W - 120):
        d.text((60, y), ln, font=font(74, True), fill=(40, 34, 30))
        y += 92
    d.line([60, y + 8, W - 60, y + 8], fill=(150, 52, 40), width=5)
    y += 44
    d.text((60, y), "Ingredients", font=font(48, True), fill=(150, 52, 40))
    y += 70
    scale = float(world.scene.layout.get("scale", 1.0))
    for key in RECIPE_VARIANT.get(cfg.variant, RECIPE_FOR.get(cfg.task, ())):
        for name, grams in P.RECIPES[key].items():
            unit = "ml" if name.endswith("_ml") else "g"
            label = name[:-3] if name.endswith("_ml") else name
            d.text((80, y), f"{grams*scale:.0f} {unit}", font=font(44, True), fill=(40, 34, 30))
            d.text((270, y), label.replace("_", " "), font=font(44), fill=(40, 34, 30))
            y += 58
    y += 26
    d.text((60, y), "Method", font=font(48, True), fill=(150, 52, 40))
    y += 70
    for ln in wrap(spec["instruction"], font(38), W - 120):
        if y > H - 90:
            break
        d.text((60, y), ln, font=font(38), fill=(40, 34, 30))
        y += 52
    d.rectangle([0, H - 26, W, H], fill=(150, 52, 40))
    p = Path(folder)/f"recipe_{cfg.task}_{cfg.variant or 'plain'}_{cfg.seed}.png"
    im.save(p)
    return p


# ====================================================================================== small builders
def _annulus(S, path, centre, r0, r1, z, mat, a0=0., a1=2*math.pi, n=96):
    """A flat printed ring (or arc) at height z, facing up."""
    a = np.linspace(a0, a1, n + 1)
    v = [(centre[0] + r*math.cos(x), centre[1] + r*math.sin(x), z) for r in (r0, r1) for x in a]
    f = []
    for i in range(n):
        f += [i, n + 1 + i, n + 2 + i, i, n + 2 + i, i + 1]
    return S.mesh(path, v, f, normals=[(0., 0., 1.)]*len(v), mat=mat)


def _disc(S, path, centre, r, z, mat, n=32):
    a = np.linspace(0, 2*math.pi, n, endpoint=False)
    v = [(centre[0], centre[1], z)] + [(centre[0] + r*math.cos(x), centre[1] + r*math.sin(x), z) for x in a]
    f = []
    for i in range(n):
        f += [0, 1 + i, 1 + (i + 1) % n]
    return S.mesh(path, v, f, normals=[(0., 0., 1.)]*len(v), mat=mat)


def _bar_handle(S, path, a, b, out, mat, r=0.005, stand=0.028):
    """A bar handle between points a and b on a front, standing `stand` off it along `out`, on two standoffs."""
    a, b, out = np.asarray(a, float), np.asarray(b, float), np.asarray(out, float)
    ax = (b - a)/np.linalg.norm(b - a)
    S.tube(path + "_bar", [a + out*stand - ax*0.012, a + out*stand, b + out*stand, b + out*stand + ax*0.012], r, mat, sides=12)
    for k, p in enumerate((a, b)):
        S.tube(f"{path}_post{k}", [p, p + out*stand], 0.8*r, mat, sides=10)


def _front(S, path, x0, x1, z0, z1, y_face, mat, handle_mat, handle="v", hinge="l", thick=0.018):
    """A door or drawer front on a carcass whose face is at y_face (fronts face -y), with its bar handle."""
    gap = 0.0015
    cx, cz = 0.5*(x0 + x1), 0.5*(z0 + z1)
    S.box(path, (x1 - x0 - 2*gap, thick, z1 - z0 - 2*gap), mat, p=(cx, y_face - 0.5*thick, cz), r=0.0012)
    yf = y_face - thick
    if handle == "v":
        xh = x1 - 0.035 if hinge == "l" else x0 + 0.035
        _bar_handle(S, path + "_handle", (xh, yf, z1 - 0.20), (xh, yf, z1 - 0.06), (0., -1., 0.), handle_mat)
    elif handle == "h":
        w = min(0.16, 0.5*(x1 - x0))
        _bar_handle(S, path + "_handle", (cx - 0.5*w, yf, z1 - 0.045), (cx + 0.5*w, yf, z1 - 0.045), (0., -1., 0.), handle_mat)


def _base_unit(S, path, x0, x1, kind, mat, handle_mat, y_face=0.025, z0=0.10, z1=CZ - P.COUNTER_TOP_THICK):
    """The fronts of one base unit: 'doors' (two), 'door-l' / 'door-r' (one, hinged on that side) or 'drawers'."""
    za, zb = z0 + 0.003, z1 - 0.003
    if kind == "doors":
        xm = 0.5*(x0 + x1)
        _front(S, path + "_l", x0, xm, za, zb, y_face, mat, handle_mat, "v", "l")
        _front(S, path + "_r", xm, x1, za, zb, y_face, mat, handle_mat, "v", "r")
    elif kind in ("door-l", "door-r"):
        _front(S, path, x0, x1, za, zb, y_face, mat, handle_mat, "v", kind[-1])
    else:
        hs = (0.30, 0.30)                                   # two deep drawers under a shallow one
        cuts = [za, za + hs[0], za + hs[0] + hs[1], zb]
        for k in range(3):
            _front(S, f"{path}_{k}", x0, x1, cuts[k], cuts[k + 1], y_face, mat, handle_mat, "h")


def _canister(S, path, p, r, h, body, lid):
    """A lidded storage canister (a closed solid; nothing is inside it)."""
    S.revolve(path + "_body", [(0., 0.), (r - 0.004, 0.), (r, 0.004), (r, h), (0., h)], body, segments=48, p=p)
    S.revolve(path + "_lid", [(0., h), (r + 0.003, h), (r + 0.003, h + 0.010), (r - 0.004, h + 0.014), (0., h + 0.014)], lid,
              segments=48, p=p)
    S.revolve(path + "_knob", [(0., h + 0.014), (0.010, h + 0.014), (0.014, h + 0.024), (0.011, h + 0.032), (0., h + 0.033)],
              lid, segments=24, p=p)


def _jar(S, path, p, r, h, fill, contents, lid):
    """A glass storage jar with dry goods in it and a steel screw lid."""
    prof = GEO.cylinder_profile(r - 0.003, h, corner=0.008)
    g = S.revolve(path + "_glass", GEO.wall_polygon(prof, 0.003), S.mats["glass"], segments=64, crease_deg=50.,
                  p=(p[0], p[1], p[2] + 0.003))
    S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    S.revolve(path + "_goods", [(0., 0.0005), (r - 0.0045, 0.0005), (r - 0.0035, 0.009), (r - 0.0035, fill*h),
                                (0., fill*h + 0.004)], contents, segments=48, p=(p[0], p[1], p[2] + 0.003))
    S.revolve(path + "_lid", [(0., h + 0.002), (r + 0.001, h + 0.002), (r + 0.001, h + 0.016), (r - 0.003, h + 0.019),
                              (0., h + 0.019)], lid, segments=48, p=p)


def _cloth_over_rail(S, path, x0, x1, rail_y, rail_z, rail_r, back, front, mat, nu=28):
    """A towel draped over a rail: a back flap, the turn over the rod and a longer front flap, with soft vertical
    folds that deepen toward the hems."""
    rr = rail_r + 0.0015
    path_yz = []
    for z in np.linspace(-back, 0., 14):
        path_yz.append((rr, z, abs(z)/back, 1.))
    for a in np.linspace(0, math.pi, 9)[1:-1]:
        path_yz.append((rr*math.cos(a), rr*math.sin(a), 0., 0.))
    for z in np.linspace(0., -front, 22):
        path_yz.append((-rr, z, abs(z)/front, -1.))
    v, uv = [], []
    us = np.linspace(0., 1., nu)
    for k, (dy, dz, hang, side) in enumerate(path_yz):
        for u in us:
            fold = 0.011*hang*(math.sin(2*math.pi*2.5*u + 0.7) + 0.4*math.sin(2*math.pi*6*u))
            pinch = 1.0 - 0.06*hang                                      # the hem gathers a little
            x = 0.5*(x0 + x1) + (u - 0.5)*(x1 - x0)*pinch
            y = rail_y + dy + (min(0.3*abs(fold), 0.004) if side > 0 else -0.8*abs(fold) if side < 0 else 0.)
            v.append((x, y, rail_z + dz))
            uv.append((u, k/(len(path_yz) - 1)))
    f = []
    for k in range(len(path_yz) - 1):
        for i in range(nu - 1):
            a, b = k*nu + i, k*nu + i + 1
            c, d = a + nu, b + nu
            f += [a, b, d, a, d, c]
    return S.mesh(path, v, f, uv=uv, mat=mat, smooth=True, double=True)


def _look_quat(forward, up=(0., 0., 1.)):
    """Quaternion (xyzw) of a frame whose z axis is `forward` and whose x axis is horizontal."""
    z = np.asarray(forward, float)/np.linalg.norm(forward)
    x = np.cross(up, z)
    x = x/np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1., 0., 0.])
    return Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()


# ====================================================================================== lighting
def panel_gain(ref):
    """Illuminance (lx) at `ref` per nit of the ceiling panels: E = sum L A cos_l cos_s / r^2."""
    ref = np.asarray(ref, float)
    A = PANEL_SIZE[0]*PANEL_SIZE[1]
    g = 0.
    for x, y in PANELS:
        d = np.array([x, y, P.ROOM["z1"] - 0.045]) - ref
        r = float(np.linalg.norm(d))
        g += A*(d[2]/r)**2/r**2
    return g


def strip_gain(ref, x0, x1, y, z, width):
    """The same for the under-shelf strip (a line of small patches facing down)."""
    ref = np.asarray(ref, float)
    xs = np.linspace(x0, x1, 81)
    xm, dA = 0.5*(xs[:-1] + xs[1:]), (xs[1] - xs[0])*width
    d = np.column_stack([xm, np.full(len(xm), y), np.full(len(xm), z)]) - ref
    r = np.linalg.norm(d, axis=1)
    return float((dA*(d[:, 2]/r)**2/r**2).sum())


# ====================================================================================== the room and the station
def dress(S, world):
    d = Dresser(S)
    sc = world.scene
    boxes = {name: (np.asarray(pos, float), np.asarray(half, float)) for name, pos, half, _, _ in sc.static_boxes}
    R = P.ROOM
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["z1"]
    tex = d.tex
    # ---- materials
    floor = d.pbr("floor_tiles", "floor_tiles_06", TEXTURES["floor_tiles_06"], .40)
    # washable warm-white paint on plaster: the pinned plaster's relief; its colour (linear mean 0.227, 0.158, 0.092,
    # nearly uniform) is brought to the paint's (0.61, 0.59, 0.52) by the tint
    plaster = d.pbr("wall_paint", "beige_wall_002", TEXTURES["beige_wall_002"], .80, tint=(2.69, 3.73, 5.65), bump=.20)
    oak = d.pbr("shelf_oak", "white_oak_veneer", TEXTURES["white_oak_veneer"], .50)
    tile = S.textured("wall_tile", _tile_maps(tex), 0.400, .10, bump=1.0, rough_influence=1.0)
    quartz = S.textured("quartz", _quartz_map(tex), 0.40, .22)
    steel, chrome, satin = S.mat("steel_brushed"), S.mat("chrome"), S.mat("steel_satin")
    cab, frame, plate = S.mat("cabinet"), S.mat("cart_frame"), S.mat("cart_plate")
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf

    # ---- room shell, door, furniture, what stands on the counter (kitchen.py)
    KIT.shell(S, d, floor, plaster, tile, TILE_TOP)
    KIT.door(S)
    KIT.furniture(S, d, world)
    KIT.props(S, d, world)
    KIT.tools(S, d, world)
    KIT.outlook(S, d)

    # ---- the counter run (the collision boxes of scene.room, drawn in full)
    cp, ch = boxes["counter_top"]
    S.box("/Dressing/counter/top", 2*ch, quartz, p=cp, r=0.0025, uv_per_m=1.)
    kp, kh = boxes["cabinet_carcass"]
    S.box("/Dressing/counter/carcass", 2*kh, cab, p=kp, r=0.001)
    tp, th = boxes["toe_kick"]
    S.box("/Dressing/counter/toe_kick", 2*th, S.mat("toe_kick"), p=tp, r=0.001)
    cx0, cx1 = cp[0] - ch[0], cp[0] + ch[0]
    y_face = kp[1] - kh[1]
    units = ("drawers", "doors", "doors", "drawers")
    w = (cx1 - cx0)/len(units)
    for k, kind in enumerate(units):
        _base_unit(S, f"/Dressing/counter/unit{k}", cx0 + k*w, cx0 + (k + 1)*w, kind, cab, satin, y_face=y_face)

    # ---- the wall shelf on its brackets, the strip light under it, and what stands on it
    sp, sh = boxes["wall_shelf"]
    S.box("/Dressing/shelf/board", 2*sh, oak, p=sp, r=0.002, uv_per_m=1., grain=0)
    z_under, z_on = sp[2] - sh[2], sp[2] + sh[2]
    for k, bx in enumerate(P.SHELF_BRACKET_X):
        S.box(f"/Dressing/shelf/bracket{k}_wall", (0.030, 0.004, 0.20), frame, p=(bx, y1 - 0.002, z_under - 0.10), r=0.001)
        S.box(f"/Dressing/shelf/bracket{k}_arm", (0.030, 2*sh[1] - 0.03, 0.004), frame, p=(bx, y1 - sh[1] + 0.015, z_under - 0.002), r=0.001)
        S.tube(f"/Dressing/shelf/bracket{k}_brace", [(bx, y1 - 0.006, z_under - 0.175), (bx, y1 - 0.205, z_under - 0.006)],
               0.005, frame, sides=8)
    strip_y, strip_z = y1 - P.SHELF_STRIP["off_wall"], z_under - 0.0085
    S.box("/Dressing/shelf/strip_profile", (2*sh[0] - 0.10, 0.018, 0.008), S.mat("aluminum"), p=(sp[0], strip_y, z_under - 0.004), r=0.001)
    cer_w, cer_b, cork = S.mat("ceramic_white"), S.mat("ceramic_blue"), S.mat("cork")
    it = P.SHELF_ITEMS                                    # positions and sizes shared with scene.fittings
    for k, ((cxk, r, hh), body) in enumerate(zip(it["canisters"], (cer_w, cer_b, cer_w))):
        _canister(S, f"/Dressing/shelf/canister{k}", (cxk, y1 - it["canister_off_wall"], z_on), r, hh, body, cork)
    for k, ((cxk, r, hh), (fill, goods)) in enumerate(zip(it["jars"], ((0.72, "dry_goods"), (0.55, "dry_goods_pale")))):
        _jar(S, f"/Dressing/shelf/jar{k}", (cxk, y1 - it["jar_off_wall"], z_on), r, hh, fill, S.mat(goods), satin)
    pbx, pbr, pbh = it["prep_bowls"]
    for k in range(4):                                    # a nest of small steel prep bowls
        prof = GEO.bowl_profile(pbr - 0.004*k, pbh, 0.030)
        S.revolve(f"/Dressing/shelf/prep_bowl{k}", GEO.wall_polygon(prof, 0.0010), steel, segments=64, crease_deg=50.,
                  p=(pbx, y1 - it["prep_off_wall"], z_on + 0.001 + 0.009*k))

    # ---- wall fittings over the counter: a rail with a towel, two socket plates
    tr = P.TOWEL_RAIL
    rail_y, rail_z = y1 - tr["off_wall"], tr["z"]
    S.cylinder("/Dressing/rail/rod", tr["rod_r"], tr["x"][1] - tr["x"][0], satin, p=(0.5*(tr["x"][0] + tr["x"][1]), rail_y, rail_z),
               axis="X")
    for k, rx in enumerate(tr["posts"]):
        S.tube(f"/Dressing/rail/post{k}", [(rx, y1, rail_z), (rx, rail_y, rail_z)], 0.005, satin, sides=10)
        S.cylinder(f"/Dressing/rail/rose{k}", 0.013, 0.004, satin, p=(rx, y1 - 0.002, rail_z), axis="Y")
    towel = S.mdl("tea_towel", "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(_towel_map(tex).resolve()))),
                                           "reflection_roughness_constant": (T.Float, .92)})
    _cloth_over_rail(S, "/Dressing/rail/towel", tr["towel_x"][0], tr["towel_x"][1], rail_y, rail_z, tr["rod_r"],
                     tr["towel_drop"][0], tr["towel_drop"][1], towel)
    white, dark = S.mat("socket_white"), S.mat("vent_dark")
    for k, sx_ in enumerate(P.SOCKETS["x"]):
        S.box(f"/Dressing/socket{k}/plate", P.SOCKETS["size"], white, p=(sx_, y1 - 0.5*P.SOCKETS["size"][1], P.SOCKETS["z"]), r=0.003)
        for j, off in enumerate((-0.036, 0.036)):
            S.box(f"/Dressing/socket{k}/outlet{j}", (0.040, 0.002, 0.040), dark, p=(sx_ + off, y1 - 0.0075, 1.112), r=0.004)
            S.box(f"/Dressing/socket{k}/rocker{j}", (0.012, 0.004, 0.020), white, p=(sx_ + off, y1 - 0.0085, 1.148), r=0.0015)

    # ---- the recipe card, clipped flat to the tiles (the wall is its collider; a card on a stand stood where a hand
    #      comes and had none)
    rc = P.RECIPE_CARD
    rx, (cw, chh) = rc["x"], rc["size"]
    yb, zb = y1 - 0.0012, rc["z_bottom"]
    up = np.array([0., 0., 1.])
    b0, b1 = np.array([rx - 0.5*cw, yb, zb]), np.array([rx + 0.5*cw, yb, zb])
    card = S.mdl("recipe_card", "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(_recipe_map(tex, world).resolve()))),
                                            "reflection_roughness_constant": (T.Float, .7)})
    nrm = np.cross(b1 - b0, up)
    nrm /= np.linalg.norm(nrm)
    S.mesh("/Dressing/recipe/card", [b0, b1, b1 + up*chh, b0 + up*chh], [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)],
           normals=[nrm]*4, mat=card)
    S.box("/Dressing/recipe/clip", (0.034, 0.004, 0.014), satin, p=(rx, y1 - 0.002, zb + chh - 0.004), r=0.001)

    # ---- the sink unit (dressing only): a base unit, the top around an inset steel bowl, a mixer tap
    sx0, sx1 = SINK_X
    # the carcass as panels (sides, back, floor): a solid box would close the bowl 30 mm under its rim
    ky0, ky1, kz0, kz1 = kp[1] - kh[1], kp[1] + kh[1], kp[2] - kh[2], kp[2] + kh[2]
    for name, (size, pos) in dict(side_l=((0.018, ky1 - ky0, kz1 - kz0), (sx0 + 0.009, kp[1], kp[2])),
                                  side_r=((0.018, ky1 - ky0, kz1 - kz0), (sx1 - 0.009, kp[1], kp[2])),
                                  back=((sx1 - sx0, 0.018, kz1 - kz0), (0.5*(sx0 + sx1), ky1 - 0.009, kp[2])),
                                  floor=((sx1 - sx0, ky1 - ky0, 0.018), (0.5*(sx0 + sx1), kp[1], kz0 + 0.009)),
                                  rail=((sx1 - sx0, 0.018, 0.06), (0.5*(sx0 + sx1), ky0 + 0.009, kz1 - 0.03))).items():
        S.box(f"/Dressing/sink/carcass_{name}", size, cab, p=pos, r=0.001)
    S.box("/Dressing/sink/toe_kick", (sx1 - sx0, 2*th[1], 2*th[2]), S.mat("toe_kick"), p=(0.5*(sx0 + sx1), tp[1], tp[2]), r=0.001)
    _base_unit(S, "/Dressing/sink/unit", sx0, sx1, "doors", cab, satin, y_face=y_face)
    bx0, bx1 = sx0 + P.SINK["bowl_inset"][0], sx1 - P.SINK["bowl_inset"][1]
    (by0, by1), depth = P.SINK["bowl_y"], P.SINK["bowl_depth"]
    zt, tt = CZ, P.COUNTER_TOP_THICK
    cd = P.COUNTER_DEPTH
    for name, (ax0, ax1, ay0, ay1) in dict(left=(sx0, bx0, 0., cd), right=(bx1, sx1, 0., cd), front=(bx0, bx1, 0., by0),
                                           back=(bx0, bx1, by1, cd)).items():
        S.box(f"/Dressing/sink/top_{name}", (ax1 - ax0, ay1 - ay0, tt), quartz, p=(0.5*(ax0 + ax1), 0.5*(ay0 + ay1), zt - 0.5*tt),
              r=0.0004, uv_per_m=1.)
    bcx, bcy = 0.5*(bx0 + bx1), 0.5*(by0 + by1)
    S.box("/Dressing/sink/bowl_floor", (bx1 - bx0, by1 - by0, 0.002), steel, p=(bcx, bcy, zt - depth), r=0.0005)
    for name, (size, p) in dict(l=((0.002, by1 - by0, depth), (bx0 + 0.001, bcy, zt - 0.5*depth)),
                                r=((0.002, by1 - by0, depth), (bx1 - 0.001, bcy, zt - 0.5*depth)),
                                f=((bx1 - bx0, 0.002, depth), (bcx, by0 + 0.001, zt - 0.5*depth)),
                                b=((bx1 - bx0, 0.002, depth), (bcx, by1 - 0.001, zt - 0.5*depth))).items():
        S.box(f"/Dressing/sink/bowl_{name}", size, steel, p=p, r=0.0005)
    rim = 0.014
    for name, (size, p) in dict(l=((rim, by1 - by0 + 2*rim, 0.0012), (bx0 - 0.5*rim, bcy, zt + 0.0006)),
                                r=((rim, by1 - by0 + 2*rim, 0.0012), (bx1 + 0.5*rim, bcy, zt + 0.0006)),
                                f=((bx1 - bx0, rim, 0.0012), (bcx, by0 - 0.5*rim, zt + 0.0006)),
                                b=((bx1 - bx0, rim, 0.0012), (bcx, by1 + 0.5*rim, zt + 0.0006))).items():
        S.box(f"/Dressing/sink/rim_{name}", size, steel, p=p, r=0.0005)
    S.cylinder("/Dressing/sink/drain", 0.030, 0.003, satin, p=(bcx, bcy, zt - depth + 0.0025))
    S.cylinder("/Dressing/sink/drain_hole", 0.019, 0.0034, dark, p=(bcx, bcy, zt - depth + 0.0026))
    tx, ty = bcx, by1 + P.SINK_TAP["behind_bowl"]
    S.cylinder("/Dressing/sink/tap_base", 0.024, 0.050, chrome, p=(tx, ty, zt + 0.025))
    neck = [(tx, ty, zt + 0.05), (tx, ty, zt + 0.28)]
    neck += [(tx, ty - 0.085 + 0.085*math.cos(a), zt + 0.28 + 0.085*math.sin(a)) for a in np.linspace(0, math.pi, 18)[1:]]
    neck += [(tx, ty - 0.17, zt + 0.245)]
    S.tube("/Dressing/sink/tap_neck", neck, 0.011, chrome, sides=16)
    S.cylinder("/Dressing/sink/tap_aerator", 0.0125, 0.012, satin, p=(tx, ty - 0.17, zt + 0.243))
    S.cylinder("/Dressing/sink/tap_valve", 0.014, 0.030, chrome, p=(tx + 0.036, ty, zt + 0.032), axis="X")
    S.tube("/Dressing/sink/tap_lever", [(tx + 0.050, ty, zt + 0.032), (tx + 0.056, ty - 0.012, zt + 0.085)], 0.0045, chrome, sides=10)
    d.ref("/Dressing/sink/cleaner", "multi_cleaner_bottle", (sx1 - P.SINK_BOTTLE["from_right"], P.SINK_BOTTLE["y"], zt), yaw=0.5)

    # ---- the fridge at the end of the run (collision proxies: scene.furniture, from the same numbers)
    F = P.FRIDGE
    (fx0, fx1), (fy0, fy1), (fz0, fz1), (dy0, dy1) = F["x"], F["y"], F["z"], F["door_y"]
    fcx = 0.5*(fx0 + fx1)
    fr = S.mat("fridge_steel")
    S.box("/Dressing/fridge/body", (fx1 - fx0, fy1 - fy0, fz1 - fz0), S.mat("mixer_grey"), p=(fcx, 0.5*(fy0 + fy1), 0.5*(fz0 + fz1)),
          r=0.004)
    S.box("/Dressing/fridge/gasket", (fx1 - fx0 - 0.02, 0.008, 1.74), S.mat("gasket"), p=(fcx, 0.027, 0.91), r=0.001)
    for name, (za, zb_) in zip(("lower", "upper"), F["doors"]):
        S.box(f"/Dressing/fridge/door_{name}", (fx1 - fx0, dy1 - dy0, zb_ - za), fr, p=(fcx, 0.5*(dy0 + dy1), 0.5*(za + zb_)), r=0.006)
    hd, ft = F["handle"], F["foot"]
    for name, (za, zb_) in zip(("lower", "upper"), hd["z"]):
        _bar_handle(S, f"/Dressing/fridge/handle_{name}", (fx1 - hd["from_right"], dy0, za), (fx1 - hd["from_right"], dy0, zb_),
                    (0., -1., 0.), satin, r=hd["r"], stand=hd["stand"])
    for k, (px, py) in enumerate((x, y) for y in (fy0 + ft["inset"], fy1 - ft["inset"]) for x in (fx0 + ft["inset"], fx1 - ft["inset"])):
        S.cylinder(f"/Dressing/fridge/foot{k}", ft["r"], ft["h"], S.mat("vent_dark"), p=(px, py, 0.5*ft["h"]))

    # ---- right of the run: a plant on the floor (beyond both presets' reach; its pot's proxy: scene.furniture)
    d.ref("/Dressing/room/plant", "potted_plant_02", (P.PLANT_XY[0], P.PLANT_XY[1], 0.), yaw=0.8)

    # ---- the hob's glass with its print (hob preset)
    if "hob_glass" in boxes:
        hp, hh = boxes["hob_glass"]
        S.box("/Dressing/hob/glass", 2*hh, S.mat("hob_glass"), p=hp, r=0.0008)
        zp = hp[2] + hh[2] + 4e-5
        pr = S.mat("hob_print")
        zone = P.HOB_ZONE
        _annulus(S, "/Dressing/hob/zone_ring", zone, 0.0885, 0.0905, zp, pr)
        _annulus(S, "/Dressing/hob/zone_core", zone, 0.0440, 0.0450, zp, pr)
        for k, a in enumerate(np.arange(4)*math.pi/2):
            c, s_ = math.cos(a), math.sin(a)
            p0, p1 = np.array(zone) + 0.096*np.array([c, s_]), np.array(zone) + 0.112*np.array([c, s_])
            n2 = 0.0008*np.array([-s_, c])
            S.mesh(f"/Dressing/hob/zone_tick{k}", [(*(p0 - n2), zp), (*(p1 - n2), zp), (*(p1 + n2), zp), (*(p0 + n2), zp)],
                   [0, 1, 2, 0, 2, 3], normals=[(0., 0., 1.)]*4, mat=pr)
        knob = sc.tools.get("knob")
        if knob is not None:
            kx, ky = sc.layout["knob_xy"]
            # the knob's pointer (the grip's -x end, render._knob) turns clockwise 30 degrees a level from 9 o'clock:
            # a dot per level, growing with the power, and a ring at 0
            for lv, ang in enumerate(knob.detents):
                a = math.pi - ang
                c = (kx + 0.034*math.cos(a), ky + 0.034*math.sin(a))
                if lv == 0:
                    _annulus(S, "/Dressing/hob/level0", c, 0.0020, 0.0030, zp, pr, n=24)
                else:
                    _disc(S, f"/Dressing/hob/level{lv}", c, 0.0010 + 0.00022*lv, zp, pr)

    # ---- the cart: a welded frame on casters (collision boxes cart_*), controllers
    dp, dh = boxes["cart_deck"]
    S.box("/Dressing/cart/deck", 2*dh, plate, p=dp, r=0.002)
    posts = [(n, b) for n, b in boxes.items() if n.startswith("cart_post")]
    px_ = sorted({round(b[0][0], 4) for _, b in posts})
    py_ = sorted({round(b[0][1], 4) for _, b in posts})
    z_rail = dp[2] - dh[2] - 0.020
    zs, hs_ = boxes["cart_shelf"]
    for lvl, zr in (("top", z_rail), ("low", zs[2] - hs_[2] - 0.020)):
        for k, yy in enumerate(py_):
            S.box(f"/Dressing/cart/rail_{lvl}_x{k}", (px_[1] - px_[0] - 0.04, 0.04, 0.04), frame, p=(0.5*(px_[0] + px_[1]), yy, zr), r=0.003)
        for k, xx in enumerate(px_):
            S.box(f"/Dressing/cart/rail_{lvl}_y{k}", (0.04, py_[1] - py_[0] - 0.04, 0.04), frame, p=(xx, 0.5*(py_[0] + py_[1]), zr), r=0.003)
    tread, hub = S.mat("caster_tread"), S.mat("caster_hub")
    for k, (name, (pp, ph)) in enumerate(posts):
        top = pp[2] + ph[2]
        S.box(f"/Dressing/cart/post{k}", (2*ph[0], 2*ph[1], top - 0.118), frame, p=(pp[0], pp[1], 0.5*(top + 0.118)), r=0.003)
        S.box(f"/Dressing/cart/caster{k}_plate", (0.060, 0.060, 0.006), hub, p=(pp[0], pp[1], 0.115), r=0.001)
        for j, sx_ in enumerate((-0.019, 0.019)):
            S.box(f"/Dressing/cart/caster{k}_fork{j}", (0.004, 0.050, 0.070), hub, p=(pp[0] + sx_, pp[1] - 0.012, 0.079), r=0.001)
        S.cylinder(f"/Dressing/cart/caster{k}_wheel", 0.045, 0.030, tread, p=(pp[0], pp[1] - 0.020, 0.045), axis="X")
        S.cylinder(f"/Dressing/cart/caster{k}_hub", 0.018, 0.034, hub, p=(pp[0], pp[1] - 0.020, 0.045), axis="X")
    S.box("/Dressing/cart/shelf", 2*hs_, plate, p=zs, r=0.002)
    green = S.color("led_green", (.05, .5, .1), .35, 0., emissive=(.15, 1., .25), emissive_intensity=800.)
    cfront, vent = S.mat("controller_front"), S.mat("vent_dark")
    for k, name in enumerate(n for n in boxes if n.startswith("cart_controller")):
        bp, bh = boxes[name]
        base = f"/Dressing/cart/controller{k}"
        S.box(base + "/case", 2*bh, S.mat("controller"), p=bp, r=0.004)
        yf = bp[1] - bh[1]
        S.box(base + "/front", (2*bh[0] - 0.02, 0.004, 2*bh[2] - 0.02), cfront, p=(bp[0], yf - 0.002, bp[2]), r=0.001)
        for j in range(9):
            S.box(f"{base}/vent{j}", (0.004, 0.002, 0.10), vent, p=(bp[0] - bh[0] + 0.035 + 0.011*j, yf - 0.0045, bp[2] - 0.01), r=0.0005)
        S.cylinder(base + "/led", 0.004, 0.003, green, p=(bp[0] + bh[0] - 0.040, yf - 0.005, bp[2] + bh[2] - 0.035), axis="Y")
        S.box(base + "/switch", (0.022, 0.006, 0.030), vent, p=(bp[0] + bh[0] - 0.045, yf - 0.006, bp[2] - 0.03), r=0.002)
        for j in range(3):
            S.cylinder(f"{base}/socket{j}", 0.009, 0.006, S.mat("steel_satin"),
                       p=(bp[0] + 0.02 + 0.03*j, yf - 0.006, bp[2] - bh[2] + 0.035), axis="Y")
        for j, hx_ in enumerate((-1, 1)):
            S.tube(f"{base}/handle{j}", [(bp[0] + hx_*(bh[0] - 0.018), yf - 0.004, bp[2] - 0.045),
                                         (bp[0] + hx_*(bh[0] - 0.018), yf - 0.030, bp[2] - 0.045),
                                         (bp[0] + hx_*(bh[0] - 0.018), yf - 0.030, bp[2] + 0.045),
                                         (bp[0] + hx_*(bh[0] - 0.018), yf - 0.004, bp[2] + 0.045)], 0.004, S.mat("steel_satin"), sides=8)
    # emergency stop on the cart's rear rail (the operator's side)
    ex, ey = dp[0] - 0.30, py_[0] - 0.020
    S.box("/Dressing/cart/estop_box", (0.072, 0.040, 0.072), S.mat("estop_yellow"), p=(ex, ey - 0.020, z_rail - 0.02), r=0.004)
    S.cylinder("/Dressing/cart/estop_button", 0.020, 0.018, S.mat("estop_red"), p=(ex, ey - 0.049, z_rail - 0.02), axis="Y")
    cable = S.mat("cable_black")
    cam = sc.cameras.get("station_rgb")
    if cam is not None:                                    # the station camera on its bracket on the back wall
        eye, fwd = np.asarray(cam["eye"], float), np.asarray(cam["target"], float) - np.asarray(cam["eye"], float)
        fwd /= np.linalg.norm(fwd)
        q = _look_quat(fwd)
        S.box("/Dressing/wall/station_plate", (0.070, 0.006, 0.090), frame, p=(eye[0], y1 - 0.003, eye[2] + 0.02), r=0.002)
        S.tube("/Dressing/wall/station_arm", [(eye[0], y1 - 0.006, eye[2] + 0.02), (eye[0], y1 - 0.030, eye[2] + 0.02),
                                              eye - fwd*0.040], 0.007, frame, sides=10)
        S.box("/Dressing/wall/station_camera", (0.052, 0.036, 0.034), S.mat("camera_dark"), p=eye - fwd*0.0285, q=q, r=0.004)
        S.cylinder("/Dressing/wall/station_lens", 0.0125, 0.010, S.mat("vent_dark"), p=eye - fwd*0.0062, q=q)
        S.tube("/Dressing/wall/station_cable", [(eye[0] + 0.020, y1 - 0.004, eye[2] + 0.05), (eye[0] + 0.020, y1 - 0.004, z_under - 0.006),
                                                (eye[0] + 0.07, y1 - 0.004, z_under - 0.006)], 0.0025, cable, sides=8)
    cam = sc.cameras.get("overhead_rgb")
    if cam is not None:
        e = np.asarray(cam["eye"], float)
        top = min(z_under, e[2] + 0.034)
        S.box("/Dressing/shelf/overhead_camera", (0.050, 0.044, top - (e[2] + 0.006)), S.mat("camera_dark"),
              p=(e[0], e[1] + 0.004, 0.5*(top + e[2] + 0.006)), r=0.003)
        S.cylinder("/Dressing/shelf/overhead_lens", 0.011, 0.005, S.mat("vent_dark"), p=(e[0], e[1], e[2] + 0.0035))
        S.tube("/Dressing/shelf/overhead_cable", [(e[0], e[1] + 0.026, top - 0.004), (e[0], y1 - 0.004, z_under - 0.006),
                                                  (e[0] + 0.05, y1 - 0.004, z_under - 0.006)], 0.0025, cable, sides=8)

    # ---- light: six ceiling panels (pinned model + a rect light each), the under-shelf strip, a weak ambient dome
    ref = (float(sc.layout["vessel_xy"][0]), float(sc.layout["vessel_xy"][1]), CZ)
    sx0_, sx1_ = sp[0] - sh[0] + 0.06, sp[0] + sh[0] - 0.06
    lum_panel = (1. - STRIP_SHARE)*TARGET_LUX/panel_gain(ref)
    lum_strip = STRIP_SHARE*TARGET_LUX/strip_gain(ref, sx0_, sx1_, strip_y, strip_z, 0.012)
    for k, (lx, ly) in enumerate(PANELS):
        d.ref(f"/Dressing/lights/panel{k}", "mounted_fluorescent_lights", (lx, ly, h))
        lt = S.UsdLux.RectLight.Define(S.stage, f"/Lights/panel{k}")
        lt.CreateWidthAttr(PANEL_SIZE[0])
        lt.CreateHeightAttr(PANEL_SIZE[1])
        lt.CreateIntensityAttr(float(lum_panel))
        lt.CreateColorAttr(S.Gf.Vec3f(*LIGHT_RGB))
        lt.GetPrim().CreateAttribute("visibleInPrimaryRay", T.Bool).Set(True)
        S.place(lt, (lx, ly, h - 0.045))                         # a RectLight emits along its -z: downwards
    lt = S.UsdLux.RectLight.Define(S.stage, "/Lights/shelf_strip")
    lt.CreateWidthAttr(float(sx1_ - sx0_))
    lt.CreateHeightAttr(0.012)
    lt.CreateIntensityAttr(float(lum_strip))
    lt.CreateColorAttr(S.Gf.Vec3f(*LIGHT_RGB))
    lt.GetPrim().CreateAttribute("visibleInPrimaryRay", T.Bool).Set(True)
    S.place(lt, (sp[0], strip_y, strip_z))
    d.credits.add(KIT.daylight(S))                           # the sky outside the window
    S.stage.GetPrimAtPath("/Dressing").SetCustomDataByKey(
        "lighting", dict(target_lux=TARGET_LUX, reference=list(ref), panel_nit=float(lum_panel), strip_nit=float(lum_strip)))
    return d.credits

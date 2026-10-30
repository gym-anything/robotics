"""Visual-only dressing of each preset's room (no collision, no physics): walls, floor, ceiling, luminaires
and background props from pinned CC0 Poly Haven models (visual_assets.py, SHA-256 pinned).

Everything a robot can touch is built in scene.py. The luminaires are RectLights sized like 2 x 4 ft troffers;
their intensity is set so the illuminance at the preset's work face is the EN 12464-1 value in params.py
(analytic point-source sum over the panels; verified by a rendered lux probe).
"""

import math
import re
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from . import params as P
from . import visual_assets as va

ROOMS = {
    "stockroom": dict(x0=-2.9, x1=2.9, y0=-3.2, y1=0.95, h=3.0),
    "counter": dict(x0=-3.4, x1=3.4, y0=-1.6, y1=4.2, h=3.2),
    "library": dict(x0=-3.4, x1=3.4, y0=-1.6, y1=4.2, h=3.0),
}
HDR_PER_NIT = 0.0042          # measured (render_calibration: dome of 100 nit on white paper, fixed tonemap exposure)


class Dresser:
    def __init__(self, S):
        self.S, self.stage = S, S.stage
        self.credits = set()
        self.folder = S.folder
        self.tex = self.folder/"textures"

    def pbr(self, name, maps, tile_m, roughness=.7, tint=None):
        """OmniPBR from pinned maps; `tint` re-tones the diffuse map (a painted/sealed surface over the scan's detail)."""
        if name in self.S.mats:
            return self.S.mats[name]
        T, Sdf = self.S.Sdf.ValueTypeNames, self.S.Sdf
        diff = maps["diff"]
        if tint is not None:
            lum = np.asarray(Image.open(diff).convert("L"), np.float32)/255.
            lum = (lum - lum.mean())/max(lum.std(), 1e-6)
            img = np.clip(np.asarray(tint)[None, None, :]*(1. + .05*lum[..., None]), 0, 1)
            diff = self.tex/f"{name}_diff.jpg"
            Image.fromarray((img*255).astype(np.uint8)).save(diff, quality=92)
        inputs = {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(diff).resolve()))),
                  "reflection_roughness_constant": (T.Float, roughness),
                  "texture_scale": (T.Float2, self.S.Gf.Vec2f(1/tile_m, 1/tile_m))}
        if "rough" in maps:
            inputs["reflectionroughness_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["rough"]).resolve())))
            inputs["reflection_roughness_texture_influence"] = (T.Float, .6)
        if "nor_gl" in maps:
            inputs["normalmap_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["nor_gl"]).resolve())))
            inputs["bump_factor"] = (T.Float, .5)
        return self.S.mdl(name, "OmniPBR", inputs)

    def quad_uv(self, path, corners, mat):
        """Planar quad (corners CCW from the front); UV in metres."""
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n = np.cross(e1, e2); n /= np.linalg.norm(n)
        uv = np.array([[0, 0], [l1, 0], [l1, l2], [0, l2]])
        return self.S.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    def ref(self, path, asset, pos, yaw=0., sub=None, scale=None, keep_x=None):
        """Reference a pinned model (visual only), re-bound to OmniPBR (RTX does not compile the models' MaterialX).
        keep_x=(x0, x1): keep only the model's parts lying within x0..x1 of its own frame, shifted so x0 sits at the
        origin (a slice of a long row of books)."""
        from pxr import Usd
        usd = va.model(asset)
        if asset not in self.credits:
            dst = self.tex
            for f in (usd.parent/"textures").glob("*"):
                if not (dst/f.name).exists():
                    shutil.copy2(f, dst/f.name)
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
        # drop curve prims: the desk lamp's cord curves carry Blender's default widths of 2 (m), which RTX drew as a
        # 2 m-thick dark tube arching over the counter (their extents omit the width, so bounds looked right); the
        # models carry the same cords as meshes
        for p in Usd.PrimRange(inner.GetPrim()):
            if p.GetTypeName() == "BasisCurves":
                p.SetActive(False)
        if sub:
            for child in inner.GetPrim().GetChildren():
                if child.GetName() not in (sub, "_materials"):
                    child.SetActive(False)
        if keep_x:
            cache = S.UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            to_asset = S.UsdGeom.Xformable(inner.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            for child in inner.GetPrim().GetChildren():
                if child.GetName() == "_materials":
                    continue
                r = cache.ComputeWorldBound(child)
                r.Transform(to_asset)
                b = r.ComputeAlignedRange()
                if b.IsEmpty() or b.GetMin()[0] < keep_x[0] or b.GetMax()[0] > keep_x[1]:
                    child.SetActive(False)
            inner.AddTranslateOp().Set(S.Gf.Vec3d(-float(keep_x[0]), 0., 0.))
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
                # a book's own cover art (decorative_book_set_01: "..._cover02_1k.png") before the shared base colour
                for tag, inp in (("_cover[0-9]+_|_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"),
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
        scopes = [prim.GetPath() for prim in Usd.PrimRange(root) if prim.GetName() == "_materials"]
        for sp in scopes:
            prim = self.stage.GetPrimAtPath(sp)
            if prim:
                prim.SetActive(False)


def _ceiling_texture(folder):
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
    p = Path(folder)/"ceiling_tile_diff.png"
    img.convert("RGB").save(p)
    return p


def _sign(folder, name, text, sub="", bg=(20, 60, 120), fg=(255, 255, 255)):
    from .artwork import _font, FONT_BOLD, FONT_SANS
    im = Image.new("RGB", (1200, 300), bg)
    d = ImageDraw.Draw(im)
    d.text((600, 120 if sub else 150), text, font=_font(FONT_BOLD, 120), fill=fg, anchor="mm")
    if sub:
        d.text((600, 235), sub, font=_font(FONT_SANS, 52), fill=fg, anchor="mm")
    p = Path(folder)/f"{name}.png"
    im.save(p)
    return p


def room_shell(d, R, floor, wall):
    S = d.S
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    d.quad_uv("/Dressing/room/floor", [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], floor)
    d.quad_uv("/Dressing/room/wall_back", [(x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h)], wall)
    d.quad_uv("/Dressing/room/wall_front", [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)], wall)
    d.quad_uv("/Dressing/room/wall_left", [(x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h)], wall)
    d.quad_uv("/Dressing/room/wall_right", [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)], wall)
    ceil = S.printed("ceiling_tiles", _ceiling_texture(d.tex), "matte")
    c = np.asarray([(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], float)
    uv = np.array([[0, 0], [0, (y1 - y0)/1.2], [(x1 - x0)/1.2, (y1 - y0)/1.2], [(x1 - x0)/1.2, 0]])
    S.mesh("/Dressing/room/ceiling", c, [0, 1, 2, 0, 2, 3], uv=uv, normals=[(0, 0, -1)]*4, mat=ceil)
    base = S.color("cove_base", (.10, .10, .10), .7)
    for nm, a, b_ in (("back", (x0, y1 - .004), (x1, y1 - .004)), ("left", (x0 + .004, y0), (x0 + .004, y1)),
                      ("right", (x1 - .004, y0), (x1 - .004, y1)), ("front", (x0, y0 + .004), (x1, y0 + .004))):
        cx, cy = (a[0] + b_[0])/2, (a[1] + b_[1])/2
        sx, sy = abs(b_[0] - a[0]) or .008, abs(b_[1] - a[1]) or .008
        S.box(f"/Dressing/room/cove_{nm}", (sx, sy, .10), base, (cx, cy, .05), (0, 0, 0, 1), bevel=.001)


def troffers(d, R, positions, target_lux, reference, normal):
    """2 x 4 ft troffer panels (RectLights 1.2 x 0.6 m) with fixture models; intensity from the lux target at the
    reference point (analytic, Lambertian panel as a point source with cos factors)."""
    S = d.S
    h = R["h"]
    area = 1.2*.6
    ref, n = np.asarray(reference, float), np.asarray(normal, float)
    # E = sum L * A * cos_s * cos_r / r^2 (panel luminance L in nit, E in lux)
    g = 0.
    for (x, y) in positions:
        p = np.array([x, y, h - .02])
        v = p - ref
        r2 = float(v @ v); r = math.sqrt(r2)
        cos_s = max(0., v[2]/r)          # panel faces down
        cos_r = max(0., float(v @ n)/r)
        g += area*cos_s*cos_r/r2
    L = target_lux/max(g, 1e-9)
    for i, (x, y) in enumerate(positions):
        light = S.UsdLux.RectLight.Define(S.stage, f"/Dressing/lights/panel_{i}")
        light.CreateWidthAttr(1.2); light.CreateHeightAttr(.6)
        light.CreateIntensityAttr(float(L)); light.CreateEnableColorTemperatureAttr(True); light.CreateColorTemperatureAttr(5000.)
        S.UsdGeom.Xformable(light).AddTranslateOp().Set(S.Gf.Vec3d(x, y, h - .02))
        S.box(f"/Dressing/lights/troffer_{i}", (1.22, .62, .012), S.color("troffer_lens", (.95, .95, .95), .3, 0., emissive=(1, 1, 1)),
              (x, y, h - .006), (0, 0, 0, 1))
    dome = S.UsdLux.DomeLight.Define(S.stage, "/Dressing/lights/ambient")
    dome.CreateIntensityAttr(20.)
    return dict(panel_luminance_nit=L, target_lux=target_lux, reference=list(ref), normal=list(n), panels=len(positions))


def _checkout_hardware(S, d, world):
    """What a checkout lane has around its scanner: a customer pole display and a card terminal at the customer edge of
    the counter's free left end, a kick plate and an edge trim on the customer side, and a bag stand on the floor past
    the right end. Visual only (no colliders), placed where no task's arm goes: the basket, the scanner and the tote
    take x = -0.6 ... +0.6 m on the robot side, the water case x = 0.7 m."""
    from scipy.spatial.transform import Rotation
    z = P.COUNTER_TOP_M
    face_customer = lambda tilt: Rotation.from_euler("x", tilt, degrees=True).as_quat()
    black, frame, steel = S.mats["tablet_black"], S.mats["cart_frame"], S.mats["aluminum"]
    # pole display facing the customer (+y): foot, pole, 220 x 100 mm head
    S.box("/Dressing/pos/display_foot", (.11, .09, .012), black, (-.80, .50, z + .006), (0, 0, 0, 1))
    S.box("/Dressing/pos/display_pole", (.022, .022, .30), frame, (-.80, .50, z + .16), (0, 0, 0, 1))
    S.box("/Dressing/pos/display_head", (.23, .035, .11), black, (-.80, .50, z + .34), face_customer(-10.))
    scr = S.printed("pos_display", _sign(d.tex, "pos_display", "TOTAL  $0.00", "THANK YOU", (10, 30, 20), (120, 230, 150)), "matte")
    # the head turned -10 deg about x: its front normal (0, .985, -.174), its up (0, .174, .985); the screen 1 mm proud
    S.quad("/Dressing/pos/display_screen", (-.80, .50 + .985*.0185, z + .34 - .174*.0185), (-1, 0, 0), (0, .174, .985),
           (.20, .085), scr)
    # card terminal on a tilted stand, keypad and screen towards the customer
    S.box("/Dressing/pos/pinpad_stand", (.09, .10, .03), frame, (-.60, .50, z + .015), (0, 0, 0, 1))
    S.box("/Dressing/pos/pinpad", (.08, .17, .03), black, (-.60, .51, z + .075), face_customer(-55.))
    pin = S.printed("pos_pinpad", _sign(d.tex, "pos_pinpad", "TAP OR", "INSERT CARD", (30, 30, 34), (235, 235, 235)), "matte")
    # slab turned -55 deg about x: its top face looks at the customer and up; screen at the slab's upper end (local y -50 mm)
    S.quad("/Dressing/pos/pinpad_screen", (-.60, .51 - .0156, z + .075 + .0502), (-1, 0, 0), (0, -.574, .819), (.06, .045), pin)
    # customer side of the counter: dark kick plate at the floor, aluminium edge trim under the top
    S.box("/Dressing/counter/kick", (1.83, .012, .10), S.mats["robot_dark"], (0., .617, .05), (0, 0, 0, 1))
    S.box("/Dressing/counter/trim", (1.83, .008, .02), steel, (0., .614, z - .048), (0, 0, 0, 1))
    # bag stand past the right end: two uprights, a top bar, three hanging carrier bags
    for k, y in enumerate((.18, .42)):
        S.box(f"/Dressing/bags/upright_{k}", (.016, .016, .95), frame, (1.20, y, .475), (0, 0, 0, 1))
    S.box("/Dressing/bags/foot", (.30, .34, .015), frame, (1.20, .30, .0075), (0, 0, 0, 1))
    S.box("/Dressing/bags/bar", (.016, .30, .016), steel, (1.20, .30, .95), (0, 0, 0, 1))
    for k, (y, yaw) in enumerate(((.22, .1), (.30, -.05), (.38, .08))):
        S.box(f"/Dressing/bags/bag_{k}", (.012, .26, .40), S.mats["bag_white"], (1.20, y, .74),
              Rotation.from_euler("z", yaw).as_quat())


def _cart_casters(S, world):
    """Casters under the cart's four posts (the cart was "a plain grey table": its posts stopped 60 mm above the floor):
    a 70 mm rubber wheel in a steel fork under each, visual only."""
    import trimesh
    sc = world.scene
    if not hasattr(sc, "cart_point"):
        return
    from .scene import CART_FRONT_Y, CART_BACK_Y
    cy = (CART_FRONT_Y + CART_BACK_Y)/2
    wheel = trimesh.creation.cylinder(radius=.035, height=.024, sections=32)
    wheel.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, (1, 0, 0)))
    for sx in (-.47, .47):
        for sy in (-.27, .27):
            p, q = sc.cart_point((sx, cy + sy, .035), (0., 0., 0., 1.))
            m = wheel.copy()
            T = np.eye(4)
            from scipy.spatial.transform import Rotation
            T[:3, :3] = Rotation.from_quat(q).as_matrix(); T[:3, 3] = p
            m.apply_transform(T)
            S.mesh(f"/Dressing/cart/caster_{sx}_{sy}".replace(".", "p").replace("-", "m"), m.vertices, m.faces,
                   mat=S.mats["caster_rubber"])
            pf, qf = sc.cart_point((sx, cy + sy, .062), (0., 0., 0., 1.))
            S.box(f"/Dressing/cart/fork_{sx}_{sy}".replace(".", "p").replace("-", "m"), (.04, .036, .012), S.mats["cart_steel"], pf, qf)


def dress(S, world):
    preset = world.config.preset
    d = Dresser(S)
    R = ROOMS[preset]
    meta = {}
    if preset == "stockroom":
        floor = d.pbr("floor_sealed", va.texture("hangar_concrete_floor"), 4., .45, tint=(.46, .46, .45))
        wall = d.pbr("wall_painted_block", va.texture("concrete_block_wall"), 2., .85, tint=(.80, .79, .74))
        room_shell(d, R, floor, wall)
        meta["lights"] = troffers(d, R, [(-1.2, .0), (1.2, .0), (-1.2, -1.8), (1.2, -1.8)], P.ROOM_LUX_STOCKROOM,
                                  (0., -.02, 1.03), (0., -1., 0.))
        # background wall shelving (left and right walls) with cases, crates and supplies
        for k, (x, yaw) in enumerate(((R["x0"] + .33, math.pi/2), (R["x1"] - .33, -math.pi/2))):
            for j, y in enumerate((-.6, -1.95)):
                d.ref(f"/Dressing/bg/shelf_{k}_{j}", "steel_frame_shelves_01" if j == 0 else "steel_frame_shelves_02", (x, y, 0.), yaw)
        for k, (x, y, z, yaw) in enumerate(((R["x0"] + .35, -.4, .13, 0.3), (R["x0"] + .35, -.85, .64, -.2), (R["x1"] - .35, -.55, .13, .1),
                                           (R["x1"] - .35, -2.1, .64, .4), (R["x0"] + .35, -2.2, .13, 0.))):
            d.ref(f"/Dressing/bg/box_{k}", "cardboard_box_01", (x, y, z), yaw)
        for k, (x, y, z) in enumerate(((R["x0"] + .35, -1.85, 1.15), (R["x1"] - .35, -.7, 1.15), (R["x1"] - .35, -1.9, .13))):
            d.ref(f"/Dressing/bg/crate_{k}", "plastic_crate_01", (x, y, z), math.pi/2)
        for k, (asset, x, y, z) in enumerate((("bleach_bottle", R["x1"] - .35, -.35, .64), ("multi_cleaner_bottle", R["x1"] - .35, -.2, .64),
                                              ("plastic_bottle_gallon", R["x0"] + .35, -1.6, .13), ("russian_food_cans_01", R["x0"] + .35, -2.3, .64),
                                              ("russian_food_cans_01", R["x1"] - .35, -2.35, 1.15))):
            d.ref(f"/Dressing/bg/supply_{k}", asset, (x, y, z), 0.)
        d.ref("/Dressing/floor/hand_truck", "hand_truck", (R["x0"] + 1.0, R["y0"] + .5, 0.), math.radians(20))
        d.ref("/Dressing/floor/bin", "metal_trash_can", (R["x1"] - .6, R["y0"] + .5, 0.))
        d.ref("/Dressing/floor/crate", "plastic_crate_03", (R["x1"] - 1.3, R["y0"] + .45, 0.), .2)
        d.ref("/Dressing/floor/ladder", "ladder_sectioned_01", (R["x1"] - .12, .45, 0.), -math.pi/2)
        d.ref("/Dressing/floor/broom", "plastic_broom", (R["x0"] + .15, .6, 0.), 0.)
        d.ref("/Dressing/wall/door", "rollershutter_door", (-1.0, R["y0"] + .02, 0.), 0.)
        d.ref("/Dressing/wall/clock", "wall_clock", (1.6, R["y1"] - .01, 2.35), 0.)
        d.ref("/Dressing/wall/extinguisher", "korean_fire_extinguisher_01", (R["x1"] - .25, -2.7, 0.), math.pi)
        d.ref("/Dressing/wall/alarm", "fire_alarm", (R["x1"] - .01, -2.9, 1.4), -math.pi/2)
        d.ref("/Dressing/wall/camera", "security_camera_01", (R["x1"] - .15, R["y1"] - .25, R["h"] - .05), math.radians(-135))
        cp, cq = world.scene.cart_point((.36, -.62, P.CART_TOP_M + .001), (0., 0., 0., 1.)) if hasattr(world.scene, "cart_point") \
            else ((.36, -.62, P.CART_TOP_M + .001), None)
        yaw = .3 + (world.scene.cart_pose[1] if getattr(world.scene, "cart_pose", None) else 0.)
        d.ref("/Dressing/cart/clipboard", "clipboard", cp, yaw)
        sign = S.printed("aisle_sign", _sign(d.tex, "aisle_sign", "AISLE A", "BAY 02  03  04", (230, 190, 30), (20, 20, 20)), "matte")
        S.quad("/Dressing/wall/aisle_sign", (0., .455, 2.05), (1, 0, 0), (0, 0, 1), (.8, .2), sign)
    else:
        lib = preset == "library"
        floor = d.pbr("floor_lib" if lib else "floor_store", va.texture("laminate_floor_02" if lib else "large_grey_tiles"),
                      2., .5 if lib else .35)
        wall = d.pbr("wall_plaster", va.texture("beige_wall_002"), 3., .85, tint=(.86, .85, .80) if lib else (.90, .90, .88))
        room_shell(d, R, floor, wall)
        meta["lights"] = troffers(d, R, [(-1.3, .3), (1.3, .3), (-1.3, 2.3), (1.3, 2.3), (0., -1.0)],
                                  P.ROOM_LUX_COUNTER, (0., .30, P.COUNTER_TOP_M), (0., 0., 1.))
        if lib:
            # four 1.374 m bookcases side by side (1.40 m pitch) along the back wall; books stand on the measured
            # board tops (0.108 ... 1.663 m) with their spines near the front edge; the 2.5 m decorative row is cut
            # into shelf-width slices, the 0.55 m encyclopedia sets go two to a board
            ys = R["y1"] - .35
            rng = np.random.default_rng(7)
            boards = (.108, .392, .672, .960, 1.277, 1.663)
            n = 0
            for k, x in enumerate((-2.1, -.7, .7, 2.1)):
                d.ref(f"/Dressing/bg/bookshelf_{k}", "wooden_bookshelf_worn", (x, ys, 0.), math.pi)
                for j, zb in enumerate(boards):
                    kind = rng.choice(["deco", "deco", "ency"] if j < 3 else ["deco", "deco", "deco", "gap"])
                    if kind == "ency":                                  # spines on the set's -y side: unrotated
                        for dx in (-.28, .28):
                            d.ref(f"/Dressing/bg/books_{n}", "book_encyclopedia_set_01", (x + dx - .256, ys - .17, zb), 0.)
                            n += 1
                    elif kind == "deco":
                        x0 = float(rng.choice([0., .6, 1.2]))
                        L = float(rng.uniform(.8, 1.12))
                        d.ref(f"/Dressing/bg/books_{n}", "decorative_book_set_01", (x + L/2, ys - .17, zb), math.pi,
                              keep_x=(x0 - .03, x0 + L))
                        n += 1
            d.ref("/Dressing/desk/lamp", "desk_lamp_arm_01", (-.78, .5, P.COUNTER_TOP_M), math.radians(200))
            sign = S.printed("desk_sign", _sign(d.tex, "desk_sign", "RETURNS", "CHECK-IN / CIRCULATION"), "matte")
        else:
            # store display units (1.08 m, four open cubbies on boards measured at 0.414 / 0.790 / 1.165 m) stocked with
            # canned goods and cleaning bottles; steel back-room racks (boards at 0.133 / 0.639 / 1.145 / 1.651 m) behind
            # the robots with cartons and jugs.
            rng = np.random.default_rng(11)
            ys = R["y1"] - .4
            n = 0
            for k, x in enumerate((-2.3, -1.2, 1.2, 2.3)):
                d.ref(f"/Dressing/bg/display_{k}", "wooden_display_shelves_01", (x, ys, 0.), math.pi)
                for zb in (.414, .790, 1.165):
                    for dx in (-.405, -.135, .135, .405):
                        kind = rng.choice(["cans", "cans", "bleach", "cleaner", "gap"])
                        if kind == "cans":
                            d.ref(f"/Dressing/bg/goods_{n}", "russian_food_cans_01", (x + dx, ys - .04, zb), math.pi)
                        elif kind in ("bleach", "cleaner"):
                            for bx in (-.05, .05):
                                d.ref(f"/Dressing/bg/goods_{n}", "bleach_bottle" if kind == "bleach" else "multi_cleaner_bottle",
                                      (x + dx + bx, ys - .05, zb), math.pi + float(rng.normal(0, .08)))
                                n += 1
                        n += 1
            for k, x in enumerate((-1.0, 1.0)):
                d.ref(f"/Dressing/back/shelf_{k}", "steel_frame_shelves_01", (x, R["y0"] + .35, 0.), 0.)
                for zb in (.133, .639, 1.145, 1.651):
                    for dx in (-.28, .21):
                        if rng.random() < .75:
                            d.ref(f"/Dressing/back/goods_{n}", "cardboard_box_01" if rng.random() < .6 else "plastic_bottle_gallon",
                                  (x + dx, R["y0"] + .37, zb + .008), float(rng.normal(0, .05)))
                            n += 1
            sign = S.printed("desk_sign", _sign(d.tex, "desk_sign", "CHECKOUT", "LANE 1", (180, 30, 30)), "matte")
            _checkout_hardware(S, d, world)
        S.quad("/Dressing/wall/sign", (0., 1.4, 2.35), (1, 0, 0), (0, 0, 1), (1.2, .3), sign)
        S.box("/Dressing/wall/sign_rod_l", (.006, .006, .7), S.mats["cart_frame"], (-.5, 1.4, 2.85), (0, 0, 0, 1))
        S.box("/Dressing/wall/sign_rod_r", (.006, .006, .7), S.mats["cart_frame"], (.5, 1.4, 2.85), (0, 0, 0, 1))
    _cart_casters(S, world)
    S.stage.GetRootLayer().customLayerData = dict(lights=str(meta.get("lights")))
    d.meta = meta
    return d.credits

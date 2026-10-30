"""Render-only dressing: room shells, furniture bodies below the physical work surfaces, props and lights.

Nothing here has physics. Every surface a robot or a packaging item can touch is an authored collision box
in scene.py; this module only adds what makes the place recognisable (boutique cash-wrap, dining room,
back room). Models and scanned textures are pinned CC0 Poly Haven assets (visual_assets.py, SHA-256 in
visual_assets.json); authored artwork (bag print, wrap pattern, signage) is generated in artwork.py.
"""

import math
import shutil
from pathlib import Path

import numpy as np

from label_application.dressing import Dresser as _LabelDresser, _textured
from . import visual_assets as va
from . import artwork


class Dresser(_LabelDresser):
    """The label environment's dresser (materials, boxes, cylinders, quads, model references with OmniPBR
    rebinding) resolving models and textures from this package's pinned manifest."""

    def __init__(self, stage, folder):
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        self.stage, self.folder = stage, Path(folder)
        (self.folder/"textures").mkdir(parents=True, exist_ok=True)
        self.mats = {}
        self.tex = {}
        self.credits = set()

    def scan(self, name, tile_m, roughness=.6):
        """OmniPBR from a pinned scanned texture (diff/rough/normal)."""
        self.credits.add(name)
        return self.pbr(name, va.texture(name), tile_m, roughness=roughness)

    def painted(self, name, scan, rgb, tile_m, roughness=.85, detail=.012):
        """Paint over a scanned surface: the scan's luminance detail under a flat paint colour (the plaster's
        relief stays, its own tint does not)."""
        from PIL import Image
        src = va.texture(scan)
        self.credits.add(scan)
        lum = np.asarray(Image.open(src["diff"]).convert("L"), np.float32)/255.
        lum = (lum - lum.mean())/max(lum.std(), 1e-6)
        img = np.clip(np.asarray(rgb, np.float32)[None, None, :]*(1. + detail*lum[..., None]), 0, 1)
        p = self.folder/"textures"/f"{name}_diff.jpg"
        Image.fromarray((img*255).astype(np.uint8)).save(p, quality=92)
        return self.pbr(name, dict(diff=p, rough=src["rough"], nor_gl=src["nor_gl"]), tile_m, roughness=roughness)

    def ref(self, path, asset, pos, yaw=0., sub=None, center_sub=True, scale=None):
        from pxr import Usd
        usd = va.model(asset)
        if asset not in self.credits:
            dst = self.folder/"textures"
            for f in (usd.parent/"textures").glob("*"):
                if not (dst/f.name).exists():
                    shutil.copy2(f, dst/f.name)
        self.credits.add(asset)
        outer = self.xform(path, pos, yaw, scale=scale)
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
        self._rebind_omnipbr(path, asset, usd)
        return outer

    # ------------------------------------------------------------- authored solids (tagged for the light guard)
    def box(self, path, center, size, mat, bevel=.002, yaw=0.):
        g = super().box(path, center, size, mat, bevel=bevel, yaw=yaw)
        g.GetPrim().SetCustomDataByKey("opaque_dressing", True)
        return g

    def cyl(self, path, center, r, h, mat, axis="Z", rot=None):
        g = super().cyl(path, center, r, h, mat, axis=axis, rot=rot)
        g.GetPrim().SetCustomDataByKey("opaque_dressing", True)
        return g

    def _place(self, g, center, R):
        """Pose a prim with rotation matrix R (columns = local axes in world) at `center`."""
        R = np.asarray(R, float)
        rows = np.r_[np.c_[R.T, np.zeros(3)].ravel(), [*map(float, center), 1.]]
        self.UsdGeom.Xformable(g).AddTransformOp().Set(self.Gf.Matrix4d(*map(float, rows)))
        return g

    def rbox(self, path, center, size, mat, R, bevel=.002):
        """Bevelled box with an arbitrary orientation R."""
        from whiteboard.usd_scene import bevel_box
        g = self._place(bevel_box(self.stage, path, np.asarray(size, float), bevel), center, R)
        g.GetPrim().SetCustomDataByKey("opaque_dressing", True)
        self.bind(g, mat)
        return g

    def rcyl(self, path, a, b, r, mat):
        """Smooth cylinder of radius r from point a to point b."""
        from whiteboard.usd_scene import smooth_cylinder
        a, b = np.asarray(a, float), np.asarray(b, float)
        g = smooth_cylinder(self.stage, path, float(r), float(np.linalg.norm(b - a)))
        self._place(g, (a + b)/2, _frame_z(b - a))
        g.GetPrim().SetCustomDataByKey("opaque_dressing", True)
        self.bind(g, mat)
        return g

    def strip(self, path, pts, side, width, mat, length_m=.2, along_v=False, notch=0.):
        """A thin flexible strip (ribbon tail, receipt) along centreline `pts` (n x 3); `side` is the unit width
        direction (3,) or per point (n x 3). UVs: the strip length runs along u (or v with along_v) at
        1/length_m cycles per metre; the other coordinate spans the width 0..1. `notch` pulls the centre of the
        last row back (a V-cut ribbon end). Double sided."""
        from whiteboard.usd_scene import mesh
        P = np.asarray(pts, float)
        S = np.broadcast_to(np.asarray(side, float), P.shape)
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        V, uv = [], []
        for k in range(len(P)):
            for j, f in enumerate((-.5, 0., .5)):
                q = P[k] + S[k]*f*width
                sk = s[k]
                if notch and k == len(P) - 1 and j == 1:
                    q = q - (P[k] - P[k - 1])/max(np.linalg.norm(P[k] - P[k - 1]), 1e-9)*notch
                    sk -= notch
                V.append(q)
                uv.append((f + .5, sk/length_m) if along_v else (sk/length_m, f + .5))
        V, uv = np.asarray(V), np.asarray(uv)
        F, UV = [], []
        for k in range(len(P) - 1):
            for j in range(2):
                a, b = 3*k + j, 3*k + j + 1
                q = [a, b, b + 3, a + 3]
                F.append(np.array(q)); UV.extend(uv[q])
        g = mesh(self.stage, path, V, F, np.asarray(UV))
        g.CreateDoubleSidedAttr(True)
        self.bind(g, mat)
        return g

    def flat_decal(self, path, center, w, h, yaw, image, lift=.0003):
        """Horizontal printed quad (image top toward +y rotated by yaw) just above a surface."""
        c, ca, sa = np.asarray(center, float) + (0, 0, lift), math.cos(yaw), math.sin(yaw)
        ex, ey = np.array([ca, sa, 0.]), np.array([-sa, ca, 0.])
        corners = [c - ex*w/2 - ey*h/2, c + ex*w/2 - ey*h/2, c + ex*w/2 + ey*h/2, c - ex*w/2 + ey*h/2]
        return self.decal_quad(path, corners, _textured(self, Path(image).stem, image))

    def screen(self, name, image, nits):
        """A lit display: OmniPBR, black diffuse, the image as emission at `nits` (cd/m2), glossy cover glass."""
        if name in self.mats:
            return self.mats[name]
        from whiteboard.appearance import mdl_surface
        Sdf, Gf = self.Sdf, self.Gf
        m = self.UsdShade.Material.Define(self.stage, "/DressMaterials/"+name)
        sh = mdl_surface(self.stage, m, "OmniPBR")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.01, .01, .012))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(.08)
        sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(image).resolve())))
        sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(nits))
        self.mats[name] = m
        return m

    def textured(self, name, image, rough=.6):
        """Opaque UsdPreviewSurface with a diffuse texture and a chosen roughness."""
        m = _textured(self, name, image)
        self.UsdShade.Shader.Get(self.stage, f"/DressMaterials/{name}/surface").GetInput("roughness").Set(float(rough))
        return m

    def _rebind_omnipbr(self, path, asset, usd):
        """As the label dresser's rebinder (OmniPBR per source material, same texture files), with two fixes.
        (1) A texture the model references but that is not in the pinned set resolves to the pinned file with the
        same stem: the Poly Haven USDs reference `*_rough*.exr` while the 1k pins hold the `.jpg` of that map; the
        old rebinder skipped the missing file yet still set the roughness-texture influence to 1, so every such
        surface rendered as if its roughness map were black (mirror gloss: the 'wet plastic' chairs).
        (2) Data maps (roughness, metallic, normal) are read raw, not as sRGB colour."""
        from pxr import Usd
        from whiteboard.appearance import mdl_surface
        Sdf, UsdShade = self.Sdf, self.UsdShade
        tex_dir = usd.parent/"textures"
        root = self.stage.GetPrimAtPath(path)
        made = {}
        self.texture_substitutions = getattr(self, "texture_substitutions", {})
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
                            name = Path(f.path).name
                            local = tex_dir/name
                            if not local.exists():
                                alt = sorted(tex_dir.glob(Path(name).stem + ".*"))
                                if alt:
                                    local = alt[0]
                                    self.texture_substitutions[f"{asset}/{name}"] = alt[0].name
                            files[name] = local
                pick = lambda *keys: next((v for k, v in files.items() if any(t in k for t in keys)), None)
                m = UsdShade.Material.Define(self.stage, f"/DressMaterials/pbr_{asset}_{key}")
                sh = mdl_surface(self.stage, m, "OmniPBR")
                bound = set()
                for tag, inp in ((("_diff",), "diffuse_texture"), (("_rough",), "reflectionroughness_texture"),
                                 (("_nor_gl",), "normalmap_texture"), (("_metal",), "metallic_texture")):
                    f = pick(*tag)
                    if f is not None and f.exists():
                        a = sh.CreateInput(inp, Sdf.ValueTypeNames.Asset)
                        a.Set(Sdf.AssetPath(str(f.resolve())))
                        if inp != "diffuse_texture":
                            a.GetAttr().SetColorSpace("raw")
                        bound.add(inp)
                if "reflectionroughness_texture" in bound:
                    sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.)
                if "metallic_texture" in bound:
                    sh.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(1.)
                if "glass" in key:
                    sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
                    sh.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(.25)
                made[key] = m
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[key])
        for sp in [prim.GetPath() for prim in Usd.PrimRange(root) if prim.GetName() == "_materials"]:
            self.stage.GetPrimAtPath(sp).SetActive(False)

    def text_quad(self, path, center, size, normal_yaw, image, tilt=0.):
        """A printed card/sign: a quad of `size` (w, h) facing yaw `normal_yaw`, textured with `image`."""
        w, h = size
        c = np.asarray(center, float)
        n = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.])
        right = np.array([-n[1], n[0], 0.])                     # left-to-right as seen from the front
        up = np.array([0., 0., 1.])*math.cos(tilt) - n*math.sin(tilt)
        corners = [c - right*w/2 - up*h/2, c + right*w/2 - up*h/2, c + right*w/2 + up*h/2, c - right*w/2 + up*h/2]
        return self.decal_quad(path, corners, _textured(self, Path(image).stem, image))


def _frame_z(z, up=(0., 0., 1.)):
    """Rotation whose local +z is `z` (columns: local x, y, z in world)."""
    z = np.asarray(z, float); z = z/np.linalg.norm(z)
    ref = np.asarray(up, float) if abs(np.dot(z, up)) < .95 else np.array([1., 0., 0.])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _prism_x(d, path, profile_yz, length, center, mat, yaw=0.):
    """Convex (y, z) profile extruded along local x by `length`, placed at `center` with a yaw (the label
    environment's prism helper, built along x)."""
    from whiteboard.usd_scene import mesh
    P = np.asarray(profile_yz, float)
    n = len(P)
    V = np.r_[np.c_[np.full(n, -length/2), P], np.c_[np.full(n, length/2), P]]
    F = [np.arange(n)[::-1], np.arange(n, 2*n)]
    F += [np.array([k, (k + 1) % n, n + (k + 1) % n, n + k]) for k in range(n)]
    g = mesh(d.stage, path, V, F)
    g.CreateDoubleSidedAttr(True)
    d._place(g, center, _rz(yaw))
    g.GetPrim().SetCustomDataByKey("opaque_dressing", True)
    d.bind(g, mat)
    return g


ROOMS = {
    "counter": dict(x0=-2.6, x1=2.6, y0=-3.2, y1=1.35, h=3.0, wall="grey_plaster", wall_tile=3.0, paint=(.86, .84, .80),
                    floor="diagonal_parquet", floor_tile=2.0),
    "home": dict(x0=-2.8, x1=2.8, y0=-3.0, y1=1.45, h=2.7, wall="grey_plaster", wall_tile=3.0, paint=(.80, .78, .72),
                 floor="diagonal_parquet", floor_tile=2.0),
    "receiving": dict(x0=-2.6, x1=2.8, y0=-3.0, y1=1.30, h=3.2, wall="grey_plaster", wall_tile=3.0, paint=(.78, .79, .76),
                      floor="concrete_floor_worn_001", floor_tile=4.0),      # painted like the other rooms (the bare scan
                                                                             # reads as cork at this tile size)
    "dev": dict(x0=-2.0, x1=2.0, y0=-2.5, y1=1.3, h=2.8, wall="grey_plaster", wall_tile=3.0,
                floor="concrete_floor_worn_001", floor_tile=4.0),
}


def room(d, preset):
    R = ROOMS[preset]
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    wall = d.painted("wall_paint", R["wall"], R["paint"], R["wall_tile"]) if "paint" in R else d.scan(R["wall"], R["wall_tile"], roughness=.85)
    floor = d.scan(R["floor"], R["floor_tile"], roughness=.5)
    ceiling = d.color("ceiling_paint", (.90, .90, .88), .9)
    d.quad("/Dressing/room/floor", [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], floor, uv_per_m=1/R["floor_tile"])
    for nm, c in (("back", [(x0, y1, 0), (x1, y1, 0), (x1, y1, h), (x0, y1, h)][::-1]),
                  ("front", [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)]),
                  ("left", [(x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h)][::-1]),
                  ("right", [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)])):
        d.quad(f"/Dressing/room/wall_{nm}", c, wall, uv_per_m=1/R["wall_tile"])
    d.quad("/Dressing/room/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], ceiling)
    skirting = d.color("skirting", (.93, .92, .89) if preset != "receiving" else (.12, .12, .12), .5)
    for nm, a, b in (("back", (x0, y1 - .006), (x1, y1 - .006)), ("left", (x0 + .006, y0), (x0 + .006, y1)),
                     ("right", (x1 - .006, y0), (x1 - .006, y1))):
        cx, cy = (a[0] + b[0])/2, (a[1] + b[1])/2
        sx, sy = abs(b[0] - a[0]) or .012, abs(b[1] - a[1]) or .012
        d.box(f"/Dressing/room/skirting_{nm}", (cx, cy, .05), (sx, sy, .10), skirting, bevel=.002)
    dome = d.UsdLux.DomeLight.Define(d.stage, "/Dressing/lights/ambient")
    dome.CreateIntensityAttr(60.)
    return R


# ----------------------------------------------------------------------------- lights
# Photometric convention (RTX, as in the filleting track): an area light's `intensity` is its luminance in cd/m2
# (nits) with `normalize` off, so a Lambertian emitter of area A and luminous flux F has L = F/(pi A).
# The camera exposure is fixed by the render worker (wiping/render_worker.py: histogram off, ISO 400, f/2.8,
# 1/120 s, ACES tonemap) and is recorded with each render (render.py export.json "camera_exposure").
LUM = dict(
    # 600 x 600 recessed LED panel, 36 W, 3600 lm, 4000 K [web: LEDVANCE panel 600 datasheet]; emitting area = the
    # 580 x 580 mm diffuser
    panel=dict(flux=3600., kelvin=4000., size=.58),
    # retail LED track spot, 36 deg beam (18 deg half-angle), centre-beam intensity 3000 cd, 3000 K [E: 1200 lm
    # class]; a Lambertian disk of radius r has on-axis intensity L*pi*r^2
    spot=dict(cd=3000., half_angle=18., softness=.25, kelvin=3000., r=.03),
    # 100 W-equivalent LED lamp in the dining pendant, 1521 lm, 2700 K [remembered: EU 1194/2012 equivalence
    # table]; a sphere of radius r has F = L*pi*4*pi*r^2
    bulb=dict(flux=1521., kelvin=2700.),
    # daylight through the dining-room window: CIE overcast sky at ~20 klx outdoor horizontal illuminance gives a
    # zenith luminance of 9E/(7 pi) ~ 8.2 kcd/m2; the near-horizon band a window sees averages ~0.55 of it, times
    # ~0.8 glazing transmittance ~ 3.6 kcd/m2, less where ground and buildings fill the lower view: 3000 cd/m2 [D]
    window=dict(nits=3000., kelvin=6500.),
)


def _light_xf(d, L, pos, emit=(0., 0., -1.)):
    """Place a UsdLux light at `pos` emitting along `emit` (lights emit along their local -z)."""
    d._place(L, pos, _frame_z(-np.asarray(emit, float)))
    return L


def _rect_light(d, path, pos, size, nits, kelvin, emit=(0., 0., -1.)):
    L = d.UsdLux.RectLight.Define(d.stage, path)
    L.CreateWidthAttr(float(size[0])); L.CreateHeightAttr(float(size[1])); L.CreateIntensityAttr(float(nits))
    L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(float(kelvin))
    return _light_xf(d, L, pos, emit)


def _sphere_light(d, path, pos, radius, nits, kelvin):
    L = d.UsdLux.SphereLight.Define(d.stage, path)
    L.CreateRadiusAttr(float(radius)); L.CreateIntensityAttr(float(nits))
    L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(float(kelvin))
    d.UsdGeom.Xformable(L).AddTranslateOp().Set(d.Gf.Vec3d(*map(float, pos)))
    return L


def _panel_light(d, path, x, y, h):
    """Recessed 600 x 600 panel: housing and diffuser (architecture()) with the emitter just below the diffuser's
    face."""
    P = LUM["panel"]
    return _rect_light(d, path, (x, y, h - PANEL_DROP - EMITTER_GAP), (P["size"], P["size"]), P["flux"]/(math.pi*P["size"]**2), P["kelvin"])


def _track_spot(d, k, mount, target, black):
    """A track spot: stem, can aimed at `target`, and the disk emitter at the can mouth with a 36 deg beam."""
    S = LUM["spot"]
    mount, target = np.asarray(mount, float), np.asarray(target, float)
    aim = (target - mount)/np.linalg.norm(target - mount)
    pivot = mount - (0., 0., .07)
    d.cyl(f"/Dressing/lights/spot_{k}_stem", mount - (0., 0., .035), .006, .07, black)
    d.cyl(f"/Dressing/lights/spot_{k}_knuckle", pivot, .012, .02, black, axis="X")
    d.rcyl(f"/Dressing/lights/spot_{k}_can", pivot - aim*.02, pivot + aim*.10, .035, black)
    L = d.UsdLux.DiskLight.Define(d.stage, f"/Dressing/lights/spot_{k}")
    L.CreateRadiusAttr(S["r"]); L.CreateIntensityAttr(S["cd"]/(math.pi*S["r"]**2))
    L.CreateEnableColorTemperatureAttr(True); L.CreateColorTemperatureAttr(S["kelvin"])
    sh = d.UsdLux.ShapingAPI.Apply(L.GetPrim())
    sh.CreateShapingConeAngleAttr(S["half_angle"]); sh.CreateShapingConeSoftnessAttr(S["softness"])
    return _light_xf(d, L, pivot + aim*(.10 + EMITTER_GAP), aim)


LIGHT_TYPES = ("RectLight", "DiskLight", "SphereLight", "CylinderLight", "DomeLight", "DistantLight")


def lighting_record(stage):
    """Every light in the stage as authored (world position, emission direction, all inputs) plus its luminous
    flux where defined: written into each render's export.json so a change in lighting is visible in the output."""
    from pxr import UsdGeom
    out = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t not in LIGHT_TYPES:
            continue
        M = cache.GetLocalToWorldTransform(prim)
        pos = list(map(float, M.ExtractTranslation()))
        emit = M.TransformDir((0., 0., -1.))
        inputs = {}
        for a in prim.GetAttributes():
            if a.GetName().startswith(("inputs:", "shaping:")) and a.HasAuthoredValue():
                v = a.Get()
                inputs[a.GetName().split(":", 1)[1]] = v if isinstance(v, (bool, int, float, str)) else list(map(float, v))
        rec = dict(path=str(prim.GetPath()), type=t, position=[round(v, 5) for v in pos],
                   emit_dir=[round(v/max(emit.GetLength(), 1e-12), 5) for v in emit], inputs=inputs)
        L = float(inputs.get("intensity", 0.))*2**float(inputs.get("exposure", 0.))
        if t == "RectLight":
            rec["flux_lm"] = round(math.pi*L*inputs["width"]*inputs["height"], 1)
        elif t == "SphereLight":
            rec["flux_lm"] = round(math.pi*L*4*math.pi*inputs["radius"]**2, 1)
        elif t == "DiskLight":
            rec["on_axis_cd"] = round(L*math.pi*inputs["radius"]**2, 1)
        out.append(rec)
    return out


def check_lights(stage, clearance=None):
    """Refuse a stage in which a light's emitting centre lies inside, or within `clearance` (default half of
    EMITTER_GAP) of, an authored opaque solid (box, cylinder, prism: customData opaque_dressing). Inside, the light
    is sealed in (the luminaire housings); 0.5 mm off a solid it was still measured to light ~nothing."""
    margin = -(EMITTER_GAP/2 if clearance is None else clearance)
    from pxr import Gf, Usd, UsdGeom
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    solids = []
    for prim in stage.Traverse():
        if prim.GetCustomDataByKey("opaque_dressing"):
            r = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
            solids.append((prim, cache.GetLocalToWorldTransform(prim).GetInverse(), np.asarray(r.GetMin()), np.asarray(r.GetMax())))
    bad = []
    for prim in stage.Traverse():
        if prim.GetTypeName() not in LIGHT_TYPES[:4]:
            continue
        c = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        for sp, inv, mn, mx in solids:
            q = np.asarray(inv.Transform(Gf.Vec3d(c)))
            if np.all(q > mn + margin) and np.all(q < mx - margin):
                bad.append(f"{prim.GetPath()} inside or within {-margin*1e3:.0f} mm of {sp.GetPath()}")
    if bad:
        raise RuntimeError("lights sealed in or against opaque dressing: " + "; ".join(bad))


def pedestals(d, scene, preset=None):
    """Robot stands from the floor to each arm's mount plate (visual; the robot base is fixed). At the counter and in
    the stock room: a steel column on a foot plate anchored to the floor. At home nothing is bolted through the rug:
    a ballasted stand, a steel disc on the floor with a column clad in the dining table's veneer."""
    steel = d.color("pedestal_steel", (.20, .21, .22), .45, .3)
    plate = d.color("pedestal_plate", (.55, .56, .58), .3, .8)
    bolt = d.color("anchor_bolt", (.62, .62, .60), .3, .9)
    for name, ((x, y), yaw) in scene.mounts.items():
        z = scene.mount_z
        if preset == "home":
            clad = d.scan("black_oak_veneer", 1.5, roughness=.45)
            d.cyl(f"/Dressing/pedestal/{name}_ballast", (x, y, .017), .23, .030, steel)
            d.box(f"/Dressing/pedestal/{name}_column", (x, y, .032 + (z - .044)/2), (.20, .20, z - .044), clad, bevel=.006)
        else:
            d.box(f"/Dressing/pedestal/{name}_column", (x, y, (z - .012)/2), (.16, .16, z - .012), steel, bevel=.004)
            d.box(f"/Dressing/pedestal/{name}_foot", (x, y, .006), (.42, .42, .012), steel, bevel=.002)
            for i, (bx, by) in enumerate(((-.17, -.17), (-.17, .17), (.17, -.17), (.17, .17))):
                d.cyl(f"/Dressing/pedestal/{name}_anchor_{i}", (x + bx, y + by, .018), .011, .012, bolt)
        d.box(f"/Dressing/pedestal/{name}_plate", (x, y, z - .006), (.20, .20, .012), plate, bevel=.002)
        for i, (bx, by) in enumerate(((-.08, -.08), (-.08, .08), (.08, -.08), (.08, .08))):
            d.cyl(f"/Dressing/pedestal/{name}_mount_bolt_{i}", (x + bx, y + by, z - .002), .006, .006, bolt)


# ----------------------------------------------------------------------------- preset A: boutique
# Dimension labels: [W] web-sourced product data, [D] derived, [E] estimate, [M] measured from a pinned asset.
PANEL_XY = ((-.9, .1), (.9, .1))     # recessed panels over the cash-wrap (housings in architecture())
PANEL_DROP = .045                    # diffuser face below the ceiling (40 mm housing + 6 mm diffuser, 1 mm overlap)
EMITTER_GAP = .010                   # clearance between an emitter and the solid behind it: at 0.5 mm the path tracer's
                                     # shadow rays to the panel emitters hit the diffuser and the panels lit ~nothing
TRACK_Y = -.25                       # ceiling track over the customer side of the counter
# Shop palette in sRGB (0..1): textures take it as is, flat materials through _lin (UsdPreviewSurface is linear)
SRGB = dict(burgundy=(.486, .086, .149), gold=(.769, .627, .322), ivory=(.93, .90, .82), navy=(.12, .17, .36),
            white=(.96, .95, .93), blush=(.93, .80, .80), kraft=(.66, .50, .34))


def _lin(c):
    return tuple(float(v)**2.2 for v in c)


# Ribbon on 25 yd (22.9 m) spools: satin caliper ~0.15 mm [E] on a 38 mm core [E]; wound diameter
# OD = 2 sqrt(L t / pi + r_core^2) = 76 mm [D]; flanges 85 mm [E]. (name, sRGB-linear colour, width, current OD)
RIBBONS = (("burgundy", .038, .076), ("gold", .038, .062), ("ivory", .025, .072), ("navy", .038, .070))
SPOOL_FLANGE_D, SPOOL_FLANGE_T = .085, .0015
TERMINAL = (.082, .187, .068)        # countertop card terminal W x L x H [W: Ingenico Desk/5000 datasheet]
PRINTER = (.128, .127, .130)         # thermal receipt printer W x D x H, 80 mm paper [W: Epson TM-m30III]
TABLET = (.262, .192, .016)          # 10.9 in tablet (248.6 x 179.5 x 6.1 mm) in a stand enclosure [E]; stand
                                     # height ~240 mm [W: Square Stand 2nd generation]
DRAWER = (.41, .10)                  # under-counter cash drawer front, 16 x 4 in [E]
SHELF_COLS = (-.351, 0., .351)       # cubby centres of wooden_display_shelves_01 [M]
# Where the props stand (m, room frame; one place for the picture and for the collision shape)
COUNTER_PROPS = dict(ribbon_rack=(-.655, .345), tissue_stack=(-.645, -.012), card_terminal=(.690, -.068),
                     receipt_printer=(.688, .098), pos_stand=(.678, .330, math.radians(145.)))
COUNTER_SHELVES_X = (-1.45, 1.45)
HOME_CHAIRS = ((-.45, -.55, math.pi/2), (.45, -.55, math.pi/2), (-1.0, .1, 0.))
HOME_GIFTS = ((-.62, (.20, .14, .10)), (.62, (.14, .14, .14)))      # other presents on the table: x, size
RECEIVING_SHELVES_X = (-1.6, 1.7)
# bounding boxes of the referenced furniture models in their own frame (min, max; m) [M: read from the USD files with
# UsdGeom.BBoxCache]
MODEL_BOUNDS = {"wooden_display_shelves_01": ((-.539, -.186, 0.), (.539, .186, 1.557)),
                "dining_chair_02": ((-.217, -.245, .001), (.217, .331, .975)),
                "steel_frame_shelves_01": ((-.549, -.251, -.001), (.549, .251, 2.139))}


def _model_box(name, asset, pos, yaw=0.):
    """Axis-aligned collision box of a referenced model placed at pos with a yaw of 0 or +-90 degrees."""
    (x0, y0, z0), (x1, y1, z1) = MODEL_BOUNDS[asset]
    q = int(round(yaw/(math.pi/2))) % 4
    if abs(yaw - q*math.pi/2) > 1e-9 and abs(yaw - (q - 4)*math.pi/2) > 1e-9:
        raise ValueError("model boxes are placed at multiples of 90 degrees")
    for _ in range(q):                                  # (x, y) -> (-y, x)
        x0, x1, y0, y1 = -y1, -y0, x0, x1
    return dict(name=name, center=(pos[0] + (x0 + x1)/2, pos[1] + (y0 + y1)/2, pos[2] + (z0 + z1)/2),
                half=((x1 - x0)/2, (y1 - y0)/2, (z1 - z0)/2))


def physical_props(preset, z, top):
    """Collision boxes of everything that is drawn standing on a work surface or within an arm's reach: the till and
    the wrapping supplies on the counter, the cabinet under it, the display shelves behind the robots; the other
    presents on the dining table, its legs and the chairs; the receiving table's legs and lower shelf and the steel
    shelving. They are fixed (a pushed terminal does not slide: the pictures cannot move). Each is dict(name, center,
    half[, yaw]); the scene gives them the work surface's friction class. `top`: the work surface's furniture dict."""
    cx, cy, _ = top["center"]
    hx, hy, _ = top["half"]
    out = []
    if preset == "counter":
        P = COUNTER_PROPS
        t_base, flange = .014, SPOOL_FLANGE_D
        h_rack = t_base + .006 + flange
        out += [dict(name="prop_ribbon_rack", center=(*P["ribbon_rack"], z + h_rack/2), half=(.1025, .0425, h_rack/2)),
                dict(name="prop_tissue_stack", center=(P["tissue_stack"][0] + .001, P["tissue_stack"][1] + .007, z + .0095),
                     half=(.102, .135, .0095)),
                dict(name="prop_card_terminal", center=(*P["card_terminal"], z + TERMINAL[2]/2),
                     half=(TERMINAL[0]/2, TERMINAL[1]/2, TERMINAL[2]/2)),
                dict(name="prop_receipt_printer", center=(*P["receipt_printer"], z + PRINTER[2]/2),
                     half=(PRINTER[0]/2, PRINTER[1]/2, PRINTER[2]/2))]
        # the tablet stand: a box across the screen (width along the tablet, depth the round base), turned with it
        x0, y0, facing = P["pos_stand"]
        h_stand = .052 + TABLET[1]*math.cos(math.radians(28.)) + .004
        out.append(dict(name="prop_pos_stand", center=(x0, y0, z + h_stand/2), half=(TABLET[0]/2, .072, h_stand/2),
                        yaw=facing - math.pi/2))
        out.append(dict(name="prop_counter_cabinet", center=(cx, cy + .02, (z - .04)/2), half=(hx - .02, hy - .04, (z - .04)/2)))
        Y = ROOMS["counter"]["y1"] - .25
        out += [_model_box(f"prop_display_shelves_{'lr'[x > 0]}", "wooden_display_shelves_01", (x, Y, 0.)) for x in COUNTER_SHELVES_X]
    elif preset == "home":
        for k, (x, size) in enumerate(HOME_GIFTS):
            out.append(dict(name=f"prop_other_present_{k}", center=(x, cy - hy + .14, z + size[2]/2 + .001),
                            half=(size[0]/2, size[1]/2, size[2]/2)))
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(dict(name=f"prop_table_leg_{'np'[sx > 0]}{'np'[sy > 0]}",
                                center=(cx + sx*(hx - .06), cy + sy*(hy - .06), (z - .04)/2), half=(.03, .03, (z - .04)/2)))
        for i, (x, y, yaw) in enumerate(HOME_CHAIRS):
            b = _model_box(f"prop_chair_{i}", "dining_chair_02", (x, y, 0.), yaw)
            # a chair pushed in at the table: the part under the table top is out of reach and would cut through the
            # top as a bounding box, so the box ends at the table's edge
            lo, hi = b["center"][1] - b["half"][1], b["center"][1] + b["half"][1]
            if abs(b["center"][0] - cx) < hx and lo < cy - hy < hi:
                hi = cy - hy - .005
                b["center"] = (b["center"][0], (lo + hi)/2, b["center"][2])
                b["half"] = (b["half"][0], (hi - lo)/2, b["half"][2])
            out.append(b)
    elif preset == "receiving":
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(dict(name=f"prop_table_leg_{'np'[sx > 0]}{'np'[sy > 0]}",
                                center=(cx + sx*(hx - .04), cy + sy*(hy - .04), (z - .04)/2), half=(.019, .019, (z - .04)/2)))
        out.append(dict(name="prop_table_undershelf", center=(cx, cy, .18), half=(hx - .03, hy - .03, .006)))
        Y = ROOMS["receiving"]["y1"] - .27
        out += [_model_box(f"prop_steel_shelves_{'lr'[x > 0]}", "steel_frame_shelves_01", (x, Y, 0.)) for x in RECEIVING_SHELVES_X]
    return out
SHELF_LEVELS = (.414, .790, 1.165)   # board top surfaces [M]; clear height ~0.35 m; top board 1.542 [M]
SHELF_TOP = 1.542


def ribbon_rack(d, c):
    """Counter ribbon dispenser on the staff side: oak base and cheeks, steel rod, four flanged spools with their
    tails drawn over the front edge onto the counter."""
    x0, y0, z0 = c
    tex = d.folder/"textures"
    oak = d.scan("grey_oak_veneer_01", 1.5, roughness=.55)
    steel = d.color("rod_steel", (.70, .70, .68), .25, .95)
    flange = d.color("spool_flange", (.94, .94, .92), .35)
    L, depth, t_base = .205, .085, .014
    d.box("/Dressing/counter/ribbon_rack/base", (x0, y0, z0 + t_base/2), (L, depth, t_base), oak, bevel=.002)
    axis_z = z0 + t_base + SPOOL_FLANGE_D/2 + .006
    cheek_h = axis_z - z0 - t_base + .022
    for sx in (-1, 1):
        d.box(f"/Dressing/counter/ribbon_rack/cheek_{'lr'[sx > 0]}", (x0 + sx*(L/2 - .006), y0, z0 + t_base + cheek_h/2),
              (.012, .060, cheek_h), oak, bevel=.003)
    d.cyl("/Dressing/counter/ribbon_rack/rod", (x0, y0, axis_z), .0045, L - .004, steel, axis="X")
    x = x0 - L/2 + .012 + .007
    rng = np.random.default_rng(5)
    for name, w, od in RIBBONS:
        rgb = SRGB[name]
        xc = x + w/2 + SPOOL_FLANGE_T
        for sx in (-1, 1):
            d.cyl(f"/Dressing/counter/ribbon_rack/{name}_flange_{'lr'[sx > 0]}", (xc + sx*(w/2 + SPOOL_FLANGE_T/2), y0, axis_z),
                  SPOOL_FLANGE_D/2, SPOOL_FLANGE_T, flange, axis="X")
        mat = d.textured(f"ribbon_{name}", artwork.ribbon_satin(tex, name, rgb), rough=.32)
        d.cyl(f"/Dressing/counter/ribbon_rack/{name}_wound", (xc, y0, axis_z), od/2, w - .0004, mat, axis="X")
        zb = axis_z - od/2 - .0004
        reach = .055 + .03*rng.random()
        pts = [(xc, y0, zb), (xc, y0 - .018, zb - .0015), (xc, y0 - depth/2 + .002, z0 + t_base + .0006),
               (xc, y0 - depth/2 - .0015, z0 + t_base - .004), (xc, y0 - depth/2 - .003, z0 + .003),
               (xc, y0 - depth/2 - .008, z0 + .0007), (xc, y0 - depth/2 - reach, z0 + .0007)]
        d.strip(f"/Dressing/counter/ribbon_rack/{name}_tail", pts, (1., 0., 0.), w - .001, mat, length_m=math.pi*od,
                notch=.009)
        x = xc + w/2 + SPOOL_FLANGE_T + .007


def tissue_stack(d, c):
    """Folded tissue on the cash-wrap: 50 x 75 cm sheets folded to 190 x 254 mm [D]; burgundy and gold under the
    white in use, each colour fanned out a little so its edge shows, the loose sheets slightly misaligned."""
    x0, y0, z0 = c
    tex = d.folder/"textures"
    rng = np.random.default_rng(3)
    z = z0
    for name, n, (ox, oy, oyaw) in (("burgundy", 4, (.012, .014, .06)), ("gold", 4, (-.010, .006, -.05)),
                                    ("white", 8, (0., 0., 0.))):
        rgb = SRGB[name]
        m = d.color(f"tissue_{name}", _lin(rgb), .9)
        for k in range(n):
            jx, jy, jyaw = rng.uniform(-.002, .002, 2).tolist() + [rng.uniform(-.012, .012)]
            d.box(f"/Dressing/counter/tissue/{name}_{k}", (x0 + ox + jx, y0 + oy + jy, z + .0006), (.190, .254, .0011), m,
                  bevel=.0004, yaw=oyaw + jyaw)
            z += .0011
        d.flat_decal(f"/Dressing/counter/tissue/{name}_top", (x0 + ox + jx, y0 + oy + jy, z), .189, .253, oyaw + jyaw,
                     artwork.tissue(tex, name, rgb, fold_lines=(.5,)), lift=.0002)


def card_terminal(d, c):
    """Countertop card terminal at the customer edge, keypad toward the customer (-y), display above it."""
    x0, y0, z0 = c
    tex = d.folder/"textures"
    W, Lg, H = TERMINAL
    body = d.color("terminal_body", (.055, .055, .06), .42)
    yA, yB, yC, yD = -Lg/2, .005, .055, Lg/2
    zA, zB, zC = .026, .046, .066
    _prism_x(d, "/Dressing/counter/terminal/keypad_wedge", [(yA, 0.), (yB, 0.), (yB, zB), (yA, zA)], W, (x0, y0, z0), body)
    _prism_x(d, "/Dressing/counter/terminal/head", [(yB, 0.), (yD, 0.), (yD, H), (yC, zC), (yB, zB)], W, (x0, y0, z0), body)

    def slope_quad(path, p0, p1, half_w, t0, t1, mat):
        p0, p1 = np.asarray(p0), np.asarray(p1)
        tdir = (p1 - p0)/np.linalg.norm(p1 - p0)
        nrm = np.array([0., -tdir[1], tdir[0]])*np.sign(tdir[0])
        a, b = p0 + (p1 - p0)*t0, p0 + (p1 - p0)*t1
        P = lambda q, sx: (x0 + sx*half_w, y0 + q[0] + nrm[1]*.0004, z0 + q[1] + nrm[2]*.0004)
        return d.decal_quad(path, [P(a, -1), P(a, 1), P(b, 1), P(b, -1)], mat)
    slope_quad("/Dressing/counter/terminal/keypad", (yA, zA), (yB, zB), .036, .06, .94,
               _textured(d, "terminal_keypad", artwork.terminal_keypad(tex)))
    slope_quad("/Dressing/counter/terminal/screen", (yB, zB), (yC, zC), .031, .10, .90,
               d.screen("terminal_screen", artwork.terminal_screen(tex), 300.))
    slot = d.color("slot_black", (.01, .01, .012), .6)
    d.box("/Dressing/counter/terminal/chip_slot", (x0, y0 + yA - .0008, z0 + .010), (.060, .002, .0035), slot, bevel=.0005)
    d.box("/Dressing/counter/terminal/lid_seam", (x0, y0 + .075, z0 + H + .0002), (W - .012, .0016, .0006), slot, bevel=.0002)


def receipt_printer(d, c):
    """Thermal receipt printer on the staff side, paper exit toward the staff (+y), the last receipt curling out."""
    x0, y0, z0 = c
    tex = d.folder/"textures"
    W, D, H = PRINTER
    body = d.color("printer_body", (.085, .085, .09), .5)
    dark = d.color("slot_black", (.01, .01, .012), .6)
    d.box("/Dressing/counter/printer/body", (x0, y0, z0 + H/2), (W, D, H), body, bevel=.012)
    d.box("/Dressing/counter/printer/cover_seam", (x0, y0 - .010, z0 + H + .0002), (W - .02, .0014, .0006), dark, bevel=.0002)
    for sx in (-1, 1):
        d.box(f"/Dressing/counter/printer/side_seam_{'lr'[sx > 0]}", (x0 + sx*(W/2 + .0002), y0 + .016, z0 + H - .03),
              (.0006, .055, .0014), dark, bevel=.0002)
    slot_y = y0 + D/2 - .013
    d.box("/Dressing/counter/printer/paper_exit", (x0, slot_y, z0 + H + .0002), (.086, .0028, .0008), dark, bevel=.0002)
    d.box("/Dressing/counter/printer/feed_button", (x0 - .035, y0 + D/2 + .0006, z0 + .100), (.014, .002, .007),
          d.color("button_grey", (.35, .35, .37), .4), bevel=.0008)
    d.box("/Dressing/counter/printer/power_led", (x0 - .016, y0 + D/2 + .0006, z0 + .100), (.004, .0015, .003),
          d.color("led_green", (.05, .5, .1), .3, emissive=(.2, 3., .4)), bevel=.0004)
    curl = [(0., 0.), (.0008, .014), (.003, .027), (.009, .038), (.019, .045), (.031, .046), (.042, .040), (.049, .029),
            (.052, .016)]
    pts = [(x0, slot_y + dy, z0 + H + .0004 + dz) for dy, dz in curl]
    d.strip("/Dressing/counter/printer/receipt", pts, (1., 0., 0.), .079, d.textured("receipt_paper", artwork.receipt(tex), .55),
            length_m=.15, along_v=True)


def pos_stand(d, c, facing):
    """Point-of-sale tablet in a swivel stand on the staff side, screen toward the staff (`facing`, azimuth)."""
    x0, y0, z0 = c
    tex = d.folder/"textures"
    e = math.radians(28.)                                       # screen normal raised 28 deg above horizontal [E]
    n = np.array([math.cos(facing)*math.cos(e), math.sin(facing)*math.cos(e), math.sin(e)])
    up = np.array([-math.cos(facing)*math.sin(e), -math.sin(facing)*math.sin(e), math.cos(e)])
    w = np.cross(up, n)
    R = np.c_[w, up, n]
    white = d.color("stand_white", (.90, .90, .89), .32)
    W, Hh, T = TABLET
    centre = np.array([x0, y0, z0 + .052 + Hh/2*math.cos(e)])
    d.cyl("/Dressing/counter/pos/base", (x0, y0, z0 + .006), .072, .012, white)
    back = centre - n*(T/2) - up*.02
    d.rcyl("/Dressing/counter/pos/neck", (x0 - n[0]*.015, y0 - n[1]*.015, z0 + .010), back - n*.004, .016, white)
    d.rbox("/Dressing/counter/pos/enclosure", centre, (W, Hh, T), white, R, bevel=.007)
    black = d.color("tablet_bezel", (.02, .02, .022), .2)
    front = centre + n*(T/2)
    d.rbox("/Dressing/counter/pos/bezel", front + n*.0004, (.250, .180, .0008), black, R, bevel=.0003)
    sw, sh = .2356, .1640                                       # 10.9 in display area [E]
    q = front + n*.0010
    d.decal_quad("/Dressing/counter/pos/screen", [q - w*sw/2 - up*sh/2, q + w*sw/2 - up*sh/2, q + w*sw/2 + up*sh/2,
                                                  q - w*sw/2 + up*sh/2], d.screen("pos_screen", artwork.pos_screen(tex), 350.))


def cash_drawer(d, x, y_face, z_top):
    """Cash drawer under the counter top on the staff face of the cabinet."""
    W, H = DRAWER
    steel = d.color("drawer_black", (.035, .035, .04), .4, .3)
    chrome = d.color("lock_chrome", (.75, .75, .74), .2, .95)
    d.box("/Dressing/counter/cash_drawer/front", (x, y_face + .006, z_top - .004 - H/2), (W, .012, H), steel, bevel=.003)
    d.box("/Dressing/counter/cash_drawer/lip", (x, y_face + .013, z_top - .004 - H + .006), (W*.7, .008, .006), steel, bevel=.002)
    d.cyl("/Dressing/counter/cash_drawer/lock", (x + W/2 - .045, y_face + .0135, z_top - .004 - .028), .008, .005, chrome, axis="Y")


# ---------------------------------------------------------------- shelf stock (each item rests on a measured board)
def tote_stack(d, path, base, n, size, ground, ink, yaw=0.):
    """Flat-packed totes (2.5 mm each, see scene.BAG_PLATE_T) with the shop print on the top one."""
    x, y, z = base
    W, H = size
    tag = f"{ground[0]}_{ground[1]}_{ground[2]}_{int(W*1000)}"
    m = d.color(f"tote_{tag}", tuple((np.asarray(ground)/255.)**2.2), .55)
    rng = np.random.default_rng(len(path))
    for k in range(n):
        jx, jy = rng.uniform(-.0015, .0015, 2)
        jyaw = yaw + rng.uniform(-.012, .012)
        d.box(f"{path}/bag_{k}", (x + jx, y + jy, z + .00125 + .0025*k), (W, H, .0024), m, bevel=.0008, yaw=jyaw)
    img = artwork.flat_tote(d.folder/"textures", tag, ground, ink, W, H, .040*H/.254)
    d.flat_decal(f"{path}/print", (x + jx, y + jy, z + .0025*n), W*.995, H*.995, jyaw, img, lift=.0001)


def tissue_packs(d, path, base, colours, yaw=0.):
    x, y, z = base
    tex = d.folder/"textures"
    rng = np.random.default_rng(len(path) + 1)
    for k, name in enumerate(colours):
        rgb = SRGB[name]
        jx, jy = rng.uniform(-.003, .003, 2)
        jyaw = yaw + rng.uniform(-.02, .02)
        d.box(f"{path}/pack_{k}", (x + jx, y + jy, z + .006 + .012*k), (.19, .25, .0118), d.color(f"tissue_{name}", _lin(rgb), .9),
              bevel=.002, yaw=jyaw)
    d.flat_decal(f"{path}/header", (x + jx, y + jy, z + .012*len(colours)), .188, .248, jyaw,
                 artwork.tissue_pack(tex, name, rgb), lift=.0002)


def boxed_candles(d, path, base, n, yaw=0.):
    """The boxed candle sold here (scene.CANDLE_HALF: 90 x 90 x 110 mm SBS carton), print toward the room."""
    x, y, z = base
    carton = d.color("carton_white", (.80, .76, .68), .45)
    for k in range(n):
        cx = x + (k - (n - 1)/2)*.098
        d.box(f"{path}/carton_{k}", (cx, y, z + .055), (.09, .09, .11), carton, bevel=.0015)
        d.text_quad(f"{path}/print_{k}", (cx, y - .045 - .0003, z + .055), (.087, .107), -math.pi/2,
                    artwork.candle_carton(d.folder/"textures"))


def jar_candles(d, path, base, n, lid_first=True):
    """Candles poured in amber glass jars, 80 mm x 85 mm [E], cream paper label."""
    x, y, z = base
    glass = d.textured("jar_amber", artwork.jar_label(d.folder/"textures"), rough=.12)
    wax = d.color("candle_wax", (.90, .84, .70), .6)
    oak = d.color("jar_lid_oak", (.55, .40, .25), .55)
    wick = d.color("wick", (.05, .04, .03), .8)
    for k in range(n):
        cx = x + (k - (n - 1)/2)*.092
        d.cyl(f"{path}/jar_{k}", (cx, y, z + .0425), .040, .085, glass)
        d.cyl(f"{path}/wax_{k}", (cx, y, z + .0855), .0355, .001, wax)          # wax face inside a 4.5 mm glass rim
        d.cyl(f"{path}/wick_{k}", (cx, y, z + .089), .0012, .007, wick)
        if lid_first and k == 0:
            d.cyl(f"{path}/lid_{k}", (cx, y, z + .091), .0415, .012, oak)


def gift_boxes(d, path, base, specs):
    """Pre-wrapped display boxes stacked largest first, satin ribbon cross and a bow on the top one."""
    x, y, z = base
    for k, (size, rgb, ribbon_rgb, yaw) in enumerate(specs):
        m = d.color(f"giftbox_{rgb}", _lin(SRGB[rgb]), .5)
        rb = d.color(f"giftribbon_{ribbon_rgb}", _lin(SRGB[ribbon_rgb]), .3)
        c = (x, y, z + size[2]/2)
        d.box(f"{path}/box_{k}", c, size, m, bevel=.002, yaw=yaw)
        d.box(f"{path}/box_{k}_ribbon_a", c, (size[0] + .003, .016, size[2] + .003), rb, bevel=.001, yaw=yaw)
        d.box(f"{path}/box_{k}_ribbon_b", c, (.016, size[1] + .003, size[2] + .003), rb, bevel=.001, yaw=yaw)
        z += size[2]
    for sx in (-1, 1):                                      # two loops raised 20 deg, splayed 30 deg
        d.rbox(f"{path}/bow_{'lr'[sx > 0]}", (x + sx*.02*math.cos(yaw), y + sx*.02*math.sin(yaw), z + .008),
               (.045, .018, .014), rb, _rz(yaw + sx*.5) @ _ry(-sx*.35), bevel=.006)
    d.box(f"{path}/bow_knot", (x, y, z + .006), (.016, .018, .012), rb, bevel=.005, yaw=yaw)


BOOKS = (("book_hardcover_01_cover87", .045), ("book_softcover_01_cover86", .032), ("book_hardcover_01_cover85", .045),
         ("book_hardcover_01_cover83", .038), ("book_softcover_01_cover82", .027), ("book_hardcover_01_cover80", .038))


def books(d, path, base, k0=0, n=5):
    """Gift books standing in a cubby, picked one by one from the pinned book set [M: 154 x 228 mm, upright]."""
    x, y, z = base
    picks = [BOOKS[(k0 + i) % len(BOOKS)] for i in range(n)]
    total = sum(t for _, t in picks) + .002*(n - 1)
    cx = x - total/2
    for i, (name, t) in enumerate(picks):
        d.ref(f"{path}/book_{i}", "decorative_book_set_01", (cx + t/2, y, z), sub=name)
        cx += t + .002


def stock_shelves(d, X, Y, plan):
    """Place stock on one shelf unit: plan entries (col index, level index or 'top', item function, kwargs)."""
    for k, (col, lvl, fn, kw) in enumerate(plan):
        z = SHELF_TOP if lvl == "top" else SHELF_LEVELS[lvl]
        fn(d, f"/Dressing/shop/stock_{'lr'[X > 0]}_{k}", (X + SHELF_COLS[col] + kw.pop("dx", 0.), Y + kw.pop("dy", 0.), z + .0005), **kw)


def _vase(d, path, base, asset="ceramic_vase_02", yaw=0.):
    d.ref(path, asset, base, yaw=yaw)


def boutique(d, scene):
    z = scene.surface_z
    top = [f for f in scene.furniture if f["name"] == "counter_top"][0]
    cx, cy, _ = top["center"]
    hx, hy, _ = top["half"]
    R = ROOMS["counter"]
    # cash-wrap cabinet under the laminate top: front panel, kick plate, side panels (visual)
    oak = d.scan("grey_oak_veneer_01", 1.5, roughness=.55)
    dark = d.color("cabinet_dark", (.07, .07, .075), .5)
    d.box("/Dressing/counter/cabinet", (cx, cy + .02, (z - .04)/2), (2*hx - .04, 2*hy - .08, z - .04), oak, bevel=.003)
    d.box("/Dressing/counter/kick", (cx, cy - hy + .07, .05), (2*hx - .06, .02, .10), dark, bevel=.002)
    # the counter's two ends, clear of the work area (|x| <= 0.48 m: boards, candle, mat, bag stack), and low next to
    # it: left end = wrapping supplies (ribbon dispenser on the staff side, tissue on the customer side); right end =
    # the till: card terminal at the customer edge, receipt printer and POS tablet on the staff side, cash drawer
    # under the top on the staff face
    ribbon_rack(d, (*COUNTER_PROPS["ribbon_rack"], z))
    tissue_stack(d, (*COUNTER_PROPS["tissue_stack"], z))
    card_terminal(d, (*COUNTER_PROPS["card_terminal"], z))
    receipt_printer(d, (*COUNTER_PROPS["receipt_printer"], z))
    pos_stand(d, (*COUNTER_PROPS["pos_stand"][:2], z), COUNTER_PROPS["pos_stand"][2])
    cash_drawer(d, .50, cy + .02 + (2*hy - .08)/2, z - .04)
    # back wall: merchandise shelves stocked on their boards
    Y = R["y1"] - .25
    d.ref("/Dressing/shop/shelves_l", "wooden_display_shelves_01", (COUNTER_SHELVES_X[0], Y, 0.))
    d.ref("/Dressing/shop/shelves_r", "wooden_display_shelves_01", (COUNTER_SHELVES_X[1], Y, 0.))
    tote_m, tote_s = (.203, .254), (.152, .203)          # 8x4x10 in (the task's tote) and 6x3x8 in [W/E]
    gold_rgb = tuple(int(c) for c in artwork.GOLD)
    stock_shelves(d, -1.45, Y, [
        (0, 0, tote_stack, dict(n=14, size=tote_m, ground=artwork.BURGUNDY, ink=artwork.GOLD)),
        (1, 0, tote_stack, dict(n=12, size=tote_s, ground=gold_rgb, ink=artwork.BURGUNDY)),
        (2, 0, tissue_packs, dict(colours=("white", "blush", "gold", "burgundy", "white"))),
        (0, 1, boxed_candles, dict(n=3)),
        (1, 1, _vase, dict()),
        (2, 1, jar_candles, dict(n=3)),
        (0, 2, books, dict(k0=0, n=6)),
        (1, 2, gift_boxes, dict(specs=(((.22, .16, .09), "burgundy", "gold", .02),
                                       ((.17, .12, .07), "navy", "gold", -.05),
                                       ((.12, .09, .06), "kraft", "burgundy", .08)))),
        (2, 2, tote_stack, dict(n=8, size=tote_m, ground=artwork.BURGUNDY, ink=artwork.GOLD)),
    ])
    stock_shelves(d, 1.45, Y, [
        (0, 0, gift_boxes, dict(specs=(((.24, .18, .10), "kraft", "burgundy", -.03),
                                       ((.18, .13, .08), "burgundy", "gold", .04)))),
        (1, 0, tote_stack, dict(n=14, size=tote_m, ground=artwork.BURGUNDY, ink=artwork.GOLD)),
        (2, 0, tote_stack, dict(n=10, size=tote_s, ground=gold_rgb, ink=artwork.BURGUNDY)),
        (0, 1, jar_candles, dict(n=3, lid_first=False)),
        (1, 1, boxed_candles, dict(n=3)),
        (2, 1, tissue_packs, dict(colours=("burgundy", "white", "gold", "blush"))),
        (0, 2, tissue_packs, dict(colours=("gold", "burgundy", "white"))),
        (1, 2, _vase, dict()),
        (2, 2, books, dict(k0=2, n=5)),
        (2, "top", _vase, dict(asset="brass_vase_01", dx=-.05)),
    ])
    d.ref("/Dressing/shop/plant", "potted_plant_02", (-2.2, -.6, 0.))
    # lights: two recessed panels, four track spots accenting the shelves (none aimed at the cash-wrap: its
    # light comes from the panels)
    h = R["h"]
    for i, (x, y) in enumerate(PANEL_XY):
        _panel_light(d, f"/Dressing/lights/panel_{i}", x, y, h)
    black = d.color("track_black", (.05, .05, .055), .35, .6)
    for k, (x, target) in enumerate(((-.9, (-1.45, Y - .17, 1.00)), (-.3, (-1.10, Y - .17, .62)),
                                     (.3, (1.10, Y - .17, .98)), (.9, (1.48, Y - .17, 1.36)))):
        _track_spot(d, k, (x, TRACK_Y, h - .035), target, black)
    d.text_quad("/Dressing/shop/sign", (-.45, R["y1"] - .01, 2.45), (1.0, .25), -math.pi/2, artwork.shop_sign(d.folder/"textures"))


# ----------------------------------------------------------------------------- preset B: dining room
PENDANT = dict(top=1.1726, bulb_z=.420, bulb_r=.046,   # [M] modern_ceiling_lamp_01: canopy top 1.1726 m above the
              x=(-.45, .45))                           # origin, bulb mesh r 36.5 mm centred 0.420 m; the light sphere
                                                       # encloses it with ~1 cm clearance (EMITTER_GAP)
WINDOW = dict(yc=-.4, wy=1.6, zs=.85, zh=2.15)         # cased window in the +x wall (architecture())
def dining(d, scene):
    z = scene.surface_z
    top = [f for f in scene.furniture if f["name"] == "table_top"][0]
    cx, cy, _ = top["center"]
    hx, hy, _ = top["half"]
    oak = d.scan("black_oak_veneer", 1.5, roughness=.45)
    for sx in (-1, 1):
        for sy in (-1, 1):
            d.box(f"/Dressing/table/leg_{'np'[sx > 0]}{'np'[sy > 0]}", (cx + sx*(hx - .06), cy + sy*(hy - .06), (z - .04)/2), (.06, .06, z - .04), oak, bevel=.004)
    d.box("/Dressing/table/apron_f", (cx, cy - hy + .06, z - .07), (2*hx - .16, .02, .06), oak, bevel=.002)
    R = ROOMS["home"]
    for i, (x, y, yaw) in enumerate(HOME_CHAIRS):
        d.ref(f"/Dressing/room/chair_{i}", "dining_chair_02", (x, y, 0.), yaw=yaw)
    d.ref("/Dressing/room/sofa", "Sofa_01", (.3, -2.3, 0.), yaw=math.pi/2)
    d.ref("/Dressing/room/console", "ClassicConsole_01", (-1.9, R["y1"] - .25, 0.))
    d.ref("/Dressing/room/vase", "ceramic_vase_01", (-2.1, R["y1"] - .25, .80))
    d.ref("/Dressing/room/frame", "standing_picture_frame_01", (-1.7, R["y1"] - .25, .80), yaw=-math.pi/2)
    d.ref("/Dressing/room/plant", "potted_plant_01", (2.3, -1.6, 0.))
    d.ref("/Dressing/room/picture", "hanging_picture_frame_02", (1.6, R["y1"] - .02, 1.55), yaw=-math.pi/2)
    # pendant hung from the ceiling by its canopy; the light is the lamp's bulb
    # A pair over the table's two halves: one central pendant (0.43 m shade, hung to 1.75 m) would sit 6 cm from the
    # overhead camera's drop rod and fill the top of the policy view; at x = +-0.45 m both shades stay outside it.
    zl = R["h"] - PENDANT["top"]
    B = LUM["bulb"]
    for i, x in enumerate(PENDANT["x"]):
        d.ref(f"/Dressing/room/pendant_{i}", "modern_ceiling_lamp_01", (x, .1, zl))
        _sphere_light(d, f"/Dressing/lights/pendant_{i}", (x, .1, zl + PENDANT["bulb_z"]), PENDANT["bulb_r"],
                      B["flux"]/(math.pi*4*math.pi*PENDANT["bulb_r"]**2), B["kelvin"])
    # daylight through the window's four panes, each emitter filling its pane on the room side of the glass and
    # facing into the room
    Wn = LUM["window"]
    xw = R["x1"] - .013 - EMITTER_GAP                  # room face of the glass (box at x1 - 0.010 +- 0.003) - gap
    ys = (WINDOW["yc"] - WINDOW["wy"]/2 + .025, WINDOW["yc"] - .0175, WINDOW["yc"] + .0175, WINDOW["yc"] + WINDOW["wy"]/2 - .025)
    zm = (WINDOW["zs"] + WINDOW["zh"])/2
    zz = (WINDOW["zs"] + .025, zm - .0175, zm + .0175, WINDOW["zh"] - .025)
    for i in range(2):
        for j in range(2):
            y0, y1, z0, z1 = ys[2*i], ys[2*i + 1], zz[2*j], zz[2*j + 1]
            _rect_light(d, f"/Dressing/lights/window_{i}{j}", (xw, (y0 + y1)/2, (z0 + z1)/2), (y1 - y0 - .002, z1 - z0 - .002),
                        Wn["nits"], Wn["kelvin"], emit=(-1., 0., 0.))
    # other presents waiting at the table's far corners (visual only, beyond the work area): wrapped boxes
    # with a satin ribbon cross
    wrap_m = _textured(d, "other_gift_wrap", artwork.wrap_pattern(d.folder/"textures"))
    gold = _textured(d, "other_gift_gold", artwork.plain(d.folder/"textures", "gold_paper", (.80, .62, .30), amp=.06))
    ribbon = d.color("satin_ribbon", (.55, .02, .04), .25)
    for k, ((x, size), m) in enumerate(zip(HOME_GIFTS, (wrap_m, gold))):
        c = (x, cy - hy + .14, z + size[2]/2 + .001)
        d.box(f"/Dressing/table/gift_{k}", c, size, m, bevel=.002)
        d.box(f"/Dressing/table/gift_{k}_ribbon_a", c, (size[0] + .004, .018, size[2] + .004), ribbon, bevel=.001)
        d.box(f"/Dressing/table/gift_{k}_ribbon_b", c, (.018, size[1] + .004, size[2] + .004), ribbon, bevel=.001)


# ----------------------------------------------------------------------------- preset C: back room
def back_room(d, scene):
    z = scene.surface_z
    top = [f for f in scene.furniture if f["name"].endswith("_top")][0]
    cx, cy, _ = top["center"]
    hx, hy, _ = top["half"]
    steel = d.color("stainless", (.66, .67, .68), .22, .95)
    for sx in (-1, 1):
        for sy in (-1, 1):
            d.cyl(f"/Dressing/table/leg_{'np'[sx > 0]}{'np'[sy > 0]}", (cx + sx*(hx - .04), cy + sy*(hy - .04), (z - .04)/2), .019, z - .04, steel)
    d.box("/Dressing/table/undershelf", (cx, cy, .18), (2*hx - .06, 2*hy - .06, .012), steel, bevel=.002)
    R = ROOMS["receiving"]
    for i, x in enumerate(RECEIVING_SHELVES_X):
        d.ref(f"/Dressing/stock/shelf_{i}", "steel_frame_shelves_01", (x, R["y1"] - .27, 0.))
        for k, zz in enumerate((.13, .64, 1.15)):
            d.ref(f"/Dressing/stock/box_{i}_{k}", "cardboard_box_01", (x - .2 + .25*(k % 2), R["y1"] - .27, zz))
    d.ref("/Dressing/stock/crate_a", "plastic_crate_01", (-2.1, -.9, 0.))
    d.ref("/Dressing/stock/crate_b", "plastic_crate_01", (-2.1, -.9, .30))
    d.ref("/Dressing/stock/hand_truck", "hand_truck", (2.3, -1.2, 0.), yaw=math.pi)
    # waste bin for packaging by the right wall: an authored grey plastic bin with a liner.
    bin_grey = d.color("waste_bin", (.17, .18, .19), .6)
    liner = d.color("bin_liner", (.03, .03, .035), .35)
    d.cyl("/Dressing/stock/bin/body", (2.42, .35, .30), .18, .60, bin_grey)
    d.cyl("/Dressing/stock/bin/rim", (2.42, .35, .605), .195, .03, bin_grey)
    d.cyl("/Dressing/stock/bin/liner", (2.42, .35, .622), .175, .004, liner)
    for i, (fx, fy) in enumerate(((-.9, .1), (.9, .1))):
        d.ref(f"/Dressing/lights/fixture_{i}", "mounted_fluorescent_lights", (fx, fy, R["h"] - .005), sub="mounted_fluorescent_lights_d")
        _rect_light(d, f"/Dressing/lights/panel_{i}", (fx, fy, R["h"] - .06), (1.1, .35), 6000., 4500.)


def _door(d, path, wall, along, width, h_door, R, leaf_rgb, frame_rgb):
    """A hinged interior door in a wall: casing, leaf with two raised panels, lever handle, hinges, kick plate.
    `wall` is 'back' (y = y1), 'left' (x = x0) or 'right' (x = x1); `along` is the door centre along that wall."""
    frame = d.color(f"{path.split('/')[-1]}_casing", frame_rgb, .45)
    leaf = d.color(f"{path.split('/')[-1]}_leaf", leaf_rgb, .5)
    steel = d.color("door_hardware", (.72, .72, .70), .25, .95)
    t, cw = .045, .07                                       # leaf thickness, casing width
    if wall == "back":
        y = R["y1"] - .012
        place = lambda u, z: (along + u, y, z)
        size = lambda su, sz, sd=.024: (su, sd, sz)
    else:
        x = R["x0"] + .012 if wall == "left" else R["x1"] - .012
        place = lambda u, z: (x, along + u, z)
        size = lambda su, sz, sd=.024: (sd, su, sz)
    d.box(f"{path}/leaf", place(0., h_door/2), size(width, h_door, t), leaf, bevel=.004)
    for k, zc in enumerate((.62, 1.52)):
        d.box(f"{path}/panel_{k}", place(0., zc), size(width - .22, .7 if k == 0 else .9, t + .012), leaf, bevel=.006)
    for side in (-1, 1):
        d.box(f"{path}/casing_{'lr'[side > 0]}", place(side*(width/2 + cw/2), (h_door + cw)/2), size(cw, h_door + cw, .03), frame, bevel=.004)
        for zc in (.25, h_door - .25):
            d.box(f"{path}/hinge_{'lr'[side > 0]}_{int(zc*100)}", place(-width/2 + .01, zc), size(.02, .10, t + .006), steel, bevel=.002) if side < 0 else None
    d.box(f"{path}/casing_head", place(0., h_door + cw/2), size(width + 2*cw, cw, .03), frame, bevel=.004)
    d.box(f"{path}/kick_plate", place(0., .12), size(width - .06, .22, t + .004), steel, bevel=.001)
    d.box(f"{path}/handle_rose", place(width/2 - .08, 1.02), size(.06, .06, t + .03), steel, bevel=.008)
    d.box(f"{path}/handle_lever", place(width/2 - .13, 1.02), size(.12, .02, t + .07), steel, bevel=.006)


def architecture(d, preset):
    """Room fabric beyond the shell (render only): a lower wall band with a chair rail, a staff door, ceiling
    luminaire housings at the authored lights; the dining room adds a cased window with curtains and a rug."""
    R = ROOMS[preset]
    x0, x1, y0, y1, h = R["x0"], R["x1"], R["y0"], R["y1"], R["h"]
    band = d.color("wall_lower_band", (.62, .58, .52) if preset == "counter" else (.74, .70, .62), .7)
    rail = d.color("chair_rail", (.93, .92, .89), .45)
    for nm, a, b in (("back", (x0, y1 - .004), (x1, y1 - .004)), ("left", (x0 + .004, y0), (x0 + .004, y1)),
                     ("right", (x1 - .004, y0), (x1 - .004, y1))):
        cx, cy = (a[0] + b[0])/2, (a[1] + b[1])/2
        sx, sy = abs(b[0] - a[0]) or .006, abs(b[1] - a[1]) or .006
        d.box(f"/Dressing/room/band_{nm}", (cx, cy, .55), (sx, sy, .90), band, bevel=.0005)
        rx, ry = (sx, .03) if nm == "back" else (.03, sy)
        d.box(f"/Dressing/room/rail_{nm}", (cx + (0 if nm == "back" else (.012 if nm == "left" else -.012)),
                                            cy - (.012 if nm == "back" else 0), 1.01), (rx, ry, .045), rail, bevel=.006)
    housing = d.color("luminaire_housing", (.92, .92, .90), .35, .2)
    diffuser = d.color("luminaire_diffuser", (.98, .98, .96), .9, emissive=(4., 3.8, 3.4))
    if preset == "counter":
        _door(d, "/Dressing/room/staff_door", "back", .45, .90, 2.10, R, (.92, .91, .88), (.95, .94, .92))
        for i, (x, y) in enumerate(PANEL_XY):
            d.box(f"/Dressing/lights/panel_{i}_housing", (x, y, h - .02), (.64, .64, .04), housing, bevel=.004)
            d.box(f"/Dressing/lights/panel_{i}_diffuser", (x, y, h - PANEL_DROP + .003), (.58, .58, .006), diffuser, bevel=.001)
        # ceiling track over the cash-wrap (its four aimed spots are authored with their lights in boutique())
        black = d.color("track_black", (.05, .05, .055), .35, .6)
        d.box("/Dressing/lights/track", (0., TRACK_Y, h - .02), (2.4, .035, .03), black, bevel=.003)
    if preset == "home":
        _door(d, "/Dressing/room/hall_door", "left", -1.55, .85, 2.05, R, (.88, .86, .80), (.95, .94, .92))
        frame = d.color("window_frame", (.95, .95, .93), .4)
        glass = d.color("window_glass", (.75, .85, .95), .05, 0., emissive=(1.6, 1.8, 2.0))
        curtain = d.color("curtain_linen", (.82, .78, .70), .9)
        xw, yc, wy, zs, zh = x1 - .02, WINDOW["yc"], WINDOW["wy"], WINDOW["zs"], WINDOW["zh"]
        d.box("/Dressing/room/window_glass", (xw + .01, yc, (zs + zh)/2), (.006, wy, zh - zs), glass, bevel=.0005)
        for k, yy in enumerate((yc - wy/2, yc, yc + wy/2)):
            d.box(f"/Dressing/room/window_stile_{k}", (xw - .01, yy, (zs + zh)/2), (.05, .05 if k != 1 else .035, zh - zs + .05), frame, bevel=.004)
        for k, zz in enumerate((zs, (zs + zh)/2, zh)):
            d.box(f"/Dressing/room/window_rail_{k}", (xw - .01, yc, zz), (.05, wy + .05, .05 if k != 1 else .035), frame, bevel=.004)
        d.box("/Dressing/room/window_sill", (xw - .06, yc, zs - .04), (.14, wy + .2, .03), frame, bevel=.004)
        for k, yy in enumerate((yc - wy/2 - .22, yc + wy/2 + .22)):
            d.box(f"/Dressing/room/curtain_{k}", (xw - .10, yy, (zh + .25)/2 + .05), (.06, .40, zh + .15), curtain, bevel=.02)
        d.cyl("/Dressing/room/curtain_rod", (xw - .12, yc, zh + .22), .012, wy + 1.1, frame, axis="Y")
        rug = d.color("rug_wool", (.42, .14, .12), .95)
        border = d.color("rug_border", (.80, .68, .48), .95)
        d.box("/Dressing/room/rug_border", (0., -.05, .003), (2.6, 1.9, .006), border, bevel=.001)
        d.box("/Dressing/room/rug", (0., -.05, .005), (2.4, 1.7, .006), rug, bevel=.001)


PRESET_DRESSING = {"counter": boutique, "home": dining, "receiving": back_room, "dev": lambda d, s: None}


def camera_mount(d, eye, h):
    """Ceiling drop rod + plate + camera body for the overhead policy camera."""
    steel = d.color("unistrut", (.55, .56, .58), .35, .8)
    cam = d.color("camera_body", (.06, .06, .07), .4)
    x, y, z = eye
    d.cyl("/Dressing/camera_mount/rod", (x, y, (z + h)/2 + .03), .012, h - z - .06, steel)
    d.box("/Dressing/camera_mount/plate", (x, y, h - .005), (.12, .12, .01), steel, bevel=.002)
    d.box("/Dressing/camera_mount/body", (x, y, z + .02), (.09, .03, .025), cam, bevel=.004)


def dress(stage, world, folder, eye_policy=None, d=None):
    """Author the dressing into `stage`; returns the pinned asset names used (credits)."""
    d = Dresser(stage, folder) if d is None else d
    preset = world.c.preset
    R = room(d, preset)
    architecture(d, preset)
    pedestals(d, world.scene, preset)
    PRESET_DRESSING[preset](d, world.scene)
    if eye_policy is not None:
        camera_mount(d, eye_policy, R["h"])
    check_lights(stage)
    return sorted(d.credits)

"""USD export of the bar scene and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses and the liquid's surface meshes (one
per liquid, reconstructed from the particles by surface.py), so one exported stage serves a whole session.
Appearance rules:
  * collision hulls are never drawn: robots use their visual meshes in their model's own per-part colours; glassware is
    its exact revolved wall polygon (the
    same polygon the liquid and the rigid pieces come from), so RTX refracts through the real wall thickness;
  * glass and liquids are dielectrics with Beer-Lambert volume absorption (WinePouring.mdl): wine colour is its
    absorbance spectrum (params.LIQUIDS, A420/A520/A620 per cm -> natural-log coefficients per metre), so a thin stream
    is lighter than the glass's depth, as in life; bottle glass green, flint or black, stemware clear crystal (IOR 1.52);
  * the room is visual-only dressing from pinned CC0 Poly Haven assets (dressing.py).
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import select
import shutil
import socket
import subprocess
import threading
import time
import numpy as np

from . import params as P
from . import geometry as GEO

MDL_FILE = Path(__file__).with_name("WinePouring.mdl")
# diffuse colour, roughness, metallic
COLORS = dict(
    floor=((.20, .15, .11), .7, 0.), robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.),
    camera_dark=((.03, .03, .035), .45, 0.), aluminum=((.62, .64, .66), .25, .9),
    pedestal_steel=((.18, .19, .20), .45, .6), pedestal_plate=((.55, .56, .57), .3, .9),
    stainless=((.80, .80, .79), .18, 1.), foil=((.34, .05, .07), .35, .7), label_paper=((.92, .89, .82), .8, 0.),
    bar_wood=((.30, .17, .09), .45, 0.), bar_wood_dark=((.16, .09, .05), .5, 0.), quartz=((.86, .85, .82), .18, 0.),
    rubber=((.03, .03, .03), .8, 0.), led_strip=((1., .85, .6), .4, 0.),
)
# glass absorption (natural-log coefficients per metre, R G B) - estimates: green 'feuille morte' bottle glass passes
# about 15/45/12 % through its 3 mm wall; flint passes ~99 %; crystal stemware ~99.7 % per cm; black bottle glass (the
# kind sold so dark that the fill level cannot be inspected; split-bottle's bottle) about 1 % through its 3 mm wall
GLASS = {"glass_green": ((632., 266., 707.), 1.52), "glass_flint": ((3., 2., 4.), 1.52), "crystal": ((0.3, 0.3, 0.3), 1.52),
         "glass_clear": ((2., 1.5, 2.5), 1.52), "glass_black": ((1535., 1535., 1535.), 1.52)}


CLEAR_GLASS = ("crystal", "glass_clear", "glass_flint")


def absorption_per_m(liquid):
    """Decadic absorbance per cm (R <- A620, G <- A520, B <- A420) -> natural-log absorption coefficient per metre."""
    a = np.asarray(P.LIQUIDS[liquid]["absorbance_rgb_per_cm"], float)
    return tuple(float(x) for x in a*math.log(10.)*100.)


class Stage:
    """Thin USD authoring helper (the subset of barcode_scanning.render.Stage this scene uses)."""

    def __init__(self, folder):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.Usd, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder/"textures").mkdir(exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(self.folder/"scene.usdc"))
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/Lab").GetPrim())
        self.mats = {}
        shutil.copyfile(MDL_FILE, self.folder/MDL_FILE.name)

    def color(self, name, rgb, rough=.5, metal=0., emissive=None, emissive_intensity=2000.):
        if name in self.mats:
            return self.mats[name]
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        inputs = {"diffuse_color_constant": (T.Color3f, Gf.Vec3f(*map(float, rgb))),
                  "reflection_roughness_constant": (T.Float, float(rough)), "metallic_constant": (T.Float, float(metal))}
        if emissive is not None:
            inputs.update({"enable_emission": (T.Bool, True), "emissive_color": (T.Color3f, Gf.Vec3f(*map(float, emissive))),
                           "emissive_intensity": (T.Float, float(emissive_intensity))})
        return self.mdl(name, "OmniPBR", inputs)

    def mdl(self, name, module, inputs, path=None):
        if name in self.mats:
            return self.mats[name]
        UsdShade = self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(self.Sdf.AssetPath(path or (module + ".mdl")), "mdl")
        sh.SetSourceAssetSubIdentifier(module, "mdl")
        for k, (typ, v) in inputs.items():
            sh.CreateInput(k, typ).Set(v)
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        self.mats[name] = m
        return m

    def dielectric(self, name, absorption, ior):
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        return self.mdl(name, "WinePouringDielectric", {"absorption": (T.Color3f, Gf.Vec3f(*map(float, absorption))),
                                                         "dielectric_ior": (T.Float, float(ior))},
                        path=str((self.folder/MDL_FILE.name).resolve()))

    def textured(self, name, maps, tile_m, roughness=.6, metal=0.):
        T, Sdf = self.Sdf.ValueTypeNames, self.Sdf
        inputs = {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(maps["diff"]).resolve()))),
                  "reflection_roughness_constant": (T.Float, roughness), "metallic_constant": (T.Float, metal),
                  "texture_scale": (T.Float2, self.Gf.Vec2f(1/tile_m, 1/tile_m))}
        if maps.get("rough"):
            inputs["reflectionroughness_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["rough"]).resolve())))
            inputs["reflection_roughness_texture_influence"] = (T.Float, .6)
        if maps.get("nor_gl"):
            inputs["normalmap_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["nor_gl"]).resolve())))
            inputs["bump_factor"] = (T.Float, .4)
        return self.mdl(name, "OmniPBR", inputs)

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    def xform(self, path, p=None, q=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        if p is not None:
            x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return x

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False):
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        v = np.asarray(v, np.float32); f = np.asarray(f, np.int32).reshape(-1, 3)
        g = UsdGeom.Mesh.Define(self.stage, path)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
        g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
        g.CreateSubdivisionSchemeAttr("none")
        g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([v.min(0), v.max(0)], np.float32)))
        if normals is not None:
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)))
            g.SetNormalsInterpolation("vertex")
        elif smooth:
            import trimesh
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)))
            g.SetNormalsInterpolation("vertex")
        if uv is not None:
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        if mat is not None:
            self.bind(g, mat)
        return g

    def box(self, path, size, mat, p=None, q=None, bevel=.0015, uv_per_m=None):
        from whiteboard.usd_scene import bevel_box
        g = bevel_box(self.stage, path, np.asarray(size, float), min(bevel, float(min(size))*.2))
        if p is not None:
            xf = self.UsdGeom.Xformable(g)
            xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            xf.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        if uv_per_m:
            self.box_uv(g, uv_per_m)
        if mat is not None:
            self.bind(g, mat)
        return g

    def box_uv(self, g, per_m):
        Sdf, Vt, UsdGeom = self.Sdf, self.Vt, self.UsdGeom
        pts = np.asarray(g.GetPointsAttr().Get()); idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(g.GetFaceVertexCountsAttr().Get())
        uv, k = [], 0
        for c in counts:
            face = pts[idx[k:k + c]]
            n = np.cross(face[1] - face[0], face[2] - face[0])
            ax = int(np.argmax(np.abs(n)))
            a, b_ = [i for i in range(3) if i != ax]
            uv.extend(face[:, [a, b_]]*per_m)
            k += c
        UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
            Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), role="diagnostic", aperture=36.):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*target),
                                                            self.Gf.Vec3d(*up)).GetInverse())
        c.CreateHorizontalApertureAttr(float(aperture)); c.CreateVerticalApertureAttr(float(aperture*9/16))
        c.CreateFocalLengthAttr(float(focal_mm)); c.CreateClippingRangeAttr(self.Gf.Vec2f(.005, 40.))
        c.GetPrim().SetCustomDataByKey("observation_role", role)
        return c

    def alias(self, name, mounted):
        a = self.UsdGeom.Xform.Define(self.stage, "/Cameras/" + name)
        a.GetPrim().SetCustomDataByKey("mounted_camera_path", str(mounted))

    def save(self):
        self.stage.GetRootLayer().Save()


def _label_texture(folder, liquid, seed=0):
    """A front label for the bottle (visual only): estate name, appellation and vintage on cream paper."""
    from PIL import Image, ImageDraw, ImageFont
    names = {"red_wine": ("CHÂTEAU LA GARENNE", "Bordeaux Supérieur", "Appellation Bordeaux Supérieur Contrôlée"),
             "white_wine": ("DOMAINE DES CRAIES", "Entre-deux-Mers", "Appellation Entre-deux-Mers Contrôlée"),
             "rose_wine": ("CLOS SAINTE-ANNE", "Bordeaux Rosé", "Appellation Bordeaux Rosé Contrôlée")}
    title, sub, aoc = names.get(liquid, names["red_wine"])
    W, H = 1400, 1000
    im = Image.new("RGB", (W, H), (236, 228, 208))
    d = ImageDraw.Draw(im)
    def font(size, bold=False):
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",):
            if Path(f).exists():
                return ImageFont.truetype(f, size)
        return ImageFont.load_default()
    d.rectangle([30, 30, W - 30, H - 30], outline=(120, 30, 40), width=8)
    d.rectangle([52, 52, W - 52, H - 52], outline=(160, 130, 70), width=3)
    d.text((W/2, 300), title, font=font(92, True), fill=(90, 20, 30), anchor="mm")
    d.text((W/2, 430), sub, font=font(64), fill=(60, 40, 30), anchor="mm")
    d.text((W/2, 560), str(2019 + seed % 5), font=font(110, True), fill=(120, 30, 40), anchor="mm")
    d.text((W/2, 760), aoc, font=font(38), fill=(60, 40, 30), anchor="mm")
    d.text((W/2, 850), "75 cl          13,5 % vol", font=font(40), fill=(60, 40, 30), anchor="mm")
    p = Path(folder)/f"label_{liquid}.png"
    im.save(p)
    return p


def _bottle_dressing(S, base, geo, liquid, folder, seed):
    """Front label (cylindrical patch on the body) and the cut foil capsule on the neck; visual only."""
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    tex = _label_texture(folder/"textures", liquid or "red_wine", seed)
    mat = S.mdl(f"label_{liquid}", "OmniPBR", {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))),
                                               "reflection_roughness_constant": (T.Float, .75)})
    r = float(np.interp(0.09, *np.asarray(sorted(((z, rr) for rr, z in geo.solid if 0.02 < z < 0.19 and rr > 0.03))).T)) + 0.0004
    z0, z1, a0, a1, n = 0.055, 0.135, -0.9, 0.9, 40
    v, uv, f, nr = [], [], [], []
    for i, a in enumerate(np.linspace(a0, a1, n + 1)):
        d = np.array([math.cos(a), math.sin(a), 0.])
        for j, z in enumerate((z0, z1)):
            v.append(d*r + np.array([0., 0., z])); uv.append((i/n, j)); nr.append(d)
    for i in range(n):
        a, b, c, d_ = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
        f += [a, c, d_, a, d_, b]
    S.mesh(f"{base}/label", v, f, uv=uv, normals=nr, mat=mat)
    # foil capsule: a thin shell over the neck from 40 mm below the lip to 4 mm below it (cut for pouring)
    top = geo.rim_z
    prof = [(rr + 0.0004, z) for rr, z in geo.solid if top - 0.045 < z < top - 0.004 and rr > 0.012]
    if len(prof) >= 2:
        prof = sorted(prof, key=lambda p: p[1])
        cap = GEO.Container("capsule", np.array(prof + [(p[0] - 0.0003, p[1]) for p in prof[::-1]]), np.zeros((3, 2)),
                            0., 0., 1., "foil", [])
        vv, tt, vn = cap.render_mesh_shaded(64)
        S.mesh(f"{base}/capsule", vv, tt, normals=vn, mat=S.mats["foil"])


def _line_texture(folder, text):
    """The printed measure mark next to a fill line ("175ml" and the crowned-M style verification mark, white enamel)."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 512, 160
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(f, 96) if Path(f).exists() else ImageFont.load_default()
    d.text((W/2, H/2), text, font=font, fill=(245, 245, 240, 255), anchor="mm")
    p = Path(folder)/f"line_{text}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p)
    return p


def _fill_lines(S, base, geo, folder):
    """Each fill line of a lined glass as printed white enamel on the outside of the bowl: a full ring 0.8 mm tall at
    the line's height and the "175ml" mark above it, front and back (UK lined measures, SI 1988/120). Visual only."""
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    enamel = S.color("line_enamel", (.86, .86, .83), .45, 0.)
    for name, z in geo.lines.items():
        rs = [rr for rr, zz in geo.solid if abs(zz - z) < 0.004]
        r = (max(rs) if rs else float(geo.solid[:, 0].max())) + 0.0003
        n = 128
        v, f, nr = [], [], []
        for i in range(n + 1):
            a = 2*math.pi*i/n
            d = np.array([math.cos(a), math.sin(a), 0.])
            for zz in (z - 0.0004, z + 0.0004):
                v.append(d*r + np.array([0., 0., zz])); nr.append(d)
        for i in range(n):
            a_, b_, c_, d_ = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            f += [a_, c_, d_, a_, d_, b_]
        S.mesh(f"{base}/{name}_ring", v, f, normals=nr, mat=enamel)
        label = name.split("_")[-1] + "ml"
        tex = _line_texture(folder/"textures", label)
        mat = S.mdl(f"line_mark_{label}", "OmniPBR", {
            "diffuse_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))),
            "enable_opacity": (T.Bool, True), "enable_opacity_texture": (T.Bool, True),
            "opacity_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))), "opacity_mode": (T.Int, 0),
            "reflection_roughness_constant": (T.Float, .5)})
        rs2 = [rr for rr, zz in geo.solid if z + 0.002 < zz < z + 0.010]
        r2 = (max(rs2) if rs2 else r) + 0.0003
        for k, a_mid in enumerate((-math.pi/2, math.pi/2)):
            v, uv, f, nr = [], [], [], []
            m = 12
            for i, a in enumerate(np.linspace(a_mid - 0.30, a_mid + 0.30, m + 1)):
                d = np.array([math.cos(a), math.sin(a), 0.])
                for j, zz in enumerate((z + 0.0015, z + 0.0075)):
                    v.append(d*r2 + np.array([0., 0., zz])); uv.append((i/m, j)); nr.append(d)
            for i in range(m):
                a_, b_, c_, d_ = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
                f += [a_, c_, d_, a_, d_, b_]
            S.mesh(f"{base}/{name}_mark_{k}", v, f, uv=uv, normals=nr, mat=mat)


def export(world, folder, seed=0):
    """Export the world's static appearance + posable bodies to folder/scene.usdc; returns (path, meta)."""
    import newton
    from . import dressing
    sc = world.scene
    b = sc.b
    folder = Path(folder)
    S = Stage(folder)
    for name, (rgb, rough, metal) in COLORS.items():
        S.color(name, rgb, rough, metal)
    for name, (absorb, ior) in GLASS.items():
        S.dielectric(name, absorb, ior)
    for liq in P.LIQUIDS:
        S.dielectric(f"liquid_{liq}", absorption_per_m(liq), P.LIQUIDS[liq]["ior"])
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    V = int(newton.ShapeFlags.VISIBLE)
    # ---- robots (visual meshes) and static solids
    for i in range(b.shape_count):
        body, lab = b.shape_body[i], b.shape_label[i]
        mname = sc.materials.get(i, "robot")
        if mname == "collision_hidden" or lab == "floor":
            continue
        flags = int(b.shape_flags[i])
        if mname == "robot" and not flags & V:
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        tf = np.asarray(b.shape_transform[i], float)
        if src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True)
        elif b.shape_type[i] == newton.GeoType.BOX:
            g = S.box(path, 2*scale, None, uv_per_m=1. if body < 0 else None)
        else:
            continue
        xf = S.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(S.Gf.Vec3d(*tf[:3]))
        xf.AddOrientOp().Set(S.Gf.Quatf(float(tf[6]), S.Gf.Vec3f(*tf[3:6])))
        if mname == "robot":
            # each visual sub-mesh keeps the colour its robot model gives it (FR3/Panda: white shells, graphite joint
            # rings and hand, status-light lens), linearised from sRGB (deposit_cans/render.py convention)
            col = np.clip(np.asarray(b.shape_color[i], float), 0., 1.)**2.2
            key = "robot_%02x%02x%02x" % tuple(int(round(c*255)) for c in np.asarray(b.shape_color[i], float).clip(0, 1))
            bright = float(col.mean()) > 0.5
            mat = S.color(key, col, .28 if bright else .38, 0.)
        else:
            mat = S.mats.get(mname) or S.mats["robot_dark"]
        S.bind(g, mat)
    # ---- glassware: the revolved wall polygons
    meta = dict(containers={}, liquids={})
    for body, geo, name in sc.containers:
        it = sc.items[name]
        mat_key = it.get("color") or geo.material
        mat = S.mats.get(mat_key) or S.mats["crystal"]
        if geo.material == "stainless":
            mat = S.mats["stainless"]
        v, t, nrm = geo.render_mesh_shaded(128 if geo.kind.startswith("glass") else 96)
        g = S.mesh(f"/Lab/body_{body}/{name}_wall", v, t, normals=nrm, mat=mat)
        if mat_key in CLEAR_GLASS:
            # clear glass casts almost no shadow in reality (its light is refracted into caustics); the path tracer has
            # no caustics, so the bar under every glass foot rendered black, seen through the foot from above
            S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
        if geo.kind.startswith("bottle"):
            _bottle_dressing(S, f"/Lab/body_{body}", geo, it.get("liquid"), folder, seed)
        if geo.lines:
            _fill_lines(S, f"/Lab/body_{body}", geo, folder)
        meta["containers"][name] = dict(body=body, kind=geo.kind)
    # ---- liquid surfaces (one mesh per liquid; geometry replaced per request)
    for liq in P.LIQUIDS:
        g = S.mesh(f"/Liquids/{liq}", [(0., 0., -50.), (.001, 0., -50.), (0., .001, -50.)], [0, 1, 2],
                   mat=S.mats[f"liquid_{liq}"])
        meta["liquids"][liq] = str(g.GetPath())
    credits = dressing.dress(S, world)
    meta["cameras"] = _cameras(S, sc)
    meta["visual_assets"] = sorted(credits)
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _cameras(S, sc):
    out = {}
    for name, m in sc.cameras.items():
        focal = 18/math.tan(math.radians(m["fov"]/2))
        if m["body"] >= 0:
            g = S.UsdGeom.Camera.Define(S.stage, f"/Lab/body_{m['body']}/{name}")
            g.AddTransformOp().Set(S.Gf.Matrix4d().SetLookAt(S.Gf.Vec3d(*m["eye"]), S.Gf.Vec3d(*m["target"]),
                                                              S.Gf.Vec3d(*m["up"])).GetInverse())
            g.CreateHorizontalApertureAttr(36.); g.CreateVerticalApertureAttr(20.25)
            g.CreateFocalLengthAttr(focal); g.CreateClippingRangeAttr(S.Gf.Vec2f(.005, 30.))
            g.GetPrim().SetCustomDataByKey("observation_role", m["role"])
        else:
            g = S.camera(f"/Cameras/{name}_cam", m["eye"], m["target"], focal, up=m.get("up", (0, 0, 1)), role=m["role"])
        S.alias(name, g.GetPath())
        out[name] = m["role"]
    return out


# ------------------------------------------------------------------------------ liquid meshes for a request
def liquid_meshes(world):
    """{liquid name: (vertices, triangles)} reconstructed from the current particles (surface.py)."""
    from .surface import reconstruct
    L = world.liquid
    out = {}
    if L is None or not L.n:
        return out
    x, _, lid = L.read()
    for k, name in enumerate(P.LIQUIDS):
        sel = lid == k
        if sel.any():
            out[name] = reconstruct(x[sel], L.dx, world.config.device, with_normals=True)
    return out


def encode_meshes(meshes):
    out = {}
    for name, m in meshes.items():
        v, t = m[0], m[1]
        if not (len(v) and len(t)):
            continue
        out[name] = dict(v=base64.b64encode(np.ascontiguousarray(v, np.float32).tobytes()).decode(),
                         t=base64.b64encode(np.ascontiguousarray(t, np.int32).tobytes()).decode())
        if len(m) > 2:
            out[name]["n"] = base64.b64encode(np.ascontiguousarray(m[2], np.float32).tobytes()).decode()
    return out


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, world, folder, scene, *, width=1280, height=720, samples=24):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, WPR_RENDER_PORT=str(listener.getsockname()[1]), WPR_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "wine_pouring.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept(); self.connection.settimeout(600)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    def _call(self, message):
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            send(self.connection, message)
            deadline = time.monotonic() + 900
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
        if not result.get("ok") or result.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(result)[:400])
        return result

    def observe(self, cameras, body_q, tick, meshes=None, resolution=None, visibility=None):
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras),
                   meshes=encode_meshes(meshes or {}), visibility=visibility or {})
        if resolution:
            msg["resolution"] = list(resolution)
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

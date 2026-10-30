"""USD export of the therapy room and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses (and the cue light's state), so one
exported stage serves a whole session. Appearance rules:
  * collision proxies are never drawn: robots use their visual meshes in their model's own per-part colours; every
    instrument and tool is an authored visual mesh built from the same geometry the physics uses (the undercut bar
    profiles of modes.py, the 5A stick's turned profile, the drum's 14 x 5 in shell, hoop, lugs and head);
  * real materials: rosewood bars, maple resonator boxes, green felt, steel pins, a walnut-veneer shell with a chrome
    hoop and a white coated head, hickory sticks, rubber mallet heads, a felt beater;
  * the room is visual-only dressing (dressing.py) from pinned CC0 Poly Haven assets and authored meshes.
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import threading
import time
import numpy as np

from . import params as P

# diffuse colour (sRGB-ish linear), roughness, metallic
COLORS = dict(
    floor=((.20, .19, .18), .9, 0.), robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.),
    camera_dark=((.03, .03, .035), .45, 0.), aluminum=((.62, .64, .66), .25, .9),
    cart_steel=((.55, .57, .58), .35, .7), cart_frame=((.12, .12, .13), .45, .5), beech=((.70, .52, .34), .5, 0.),
    table_laminate=((.62, .50, .36), .45, 0.), table_frame=((.15, .15, .16), .4, .6), felt=((.08, .25, .12), .95, 0.),
    pin_metal=((.72, .72, .70), .25, 1.), rubber=((.03, .03, .03), .8, 0.), xylo_box=((.72, .58, .40), .5, 0.),
    maple=((.70, .54, .36), .45, 0.), chrome=((.92, .92, .92), .08, 1.), head_coated=((.74, .73, .69), .72, 0.),
    shell=((.36, .20, .10), .35, 0.), hickory=((.58, .43, .27), .45, 0.), mallet_rubber=((.10, .18, .45), .55, 0.),
    beater_felt=((.86, .84, .78), .95, 0.), speaker_black=((.04, .04, .045), .6, 0.), trap_felt=((.03, .03, .03), .95, 0.),
    hole_dark=((.02, .015, .01), .9, 0.), label_ink=((.95, .93, .88), .6, 0.),
)


class Stage:
    """Thin USD authoring helper (the subset of wine_pouring.render.Stage this scene uses)."""

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

    def textured(self, name, maps, tile_m, roughness=.6, metal=0., tint=None):
        T, Sdf = self.Sdf.ValueTypeNames, self.Sdf
        inputs = {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(maps["diff"]).resolve()))),
                  "reflection_roughness_constant": (T.Float, roughness), "metallic_constant": (T.Float, metal),
                  "texture_scale": (T.Float2, self.Gf.Vec2f(1/tile_m, 1/tile_m))}
        if tint is not None:
            inputs["diffuse_tint"] = (T.Color3f, self.Gf.Vec3f(*map(float, tint)))
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

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, uv_face=False):
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        v = np.asarray(v, np.float32)
        f = np.asarray(f, np.int32).reshape(-1, 3)
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
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 "faceVarying" if uv_face else "vertex").Set(
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
        pts = np.asarray(g.GetPointsAttr().Get())
        idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
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


# ====================================================================================================== mesh builders
def lathe(xs, rs, n=40, cap0=True, cap1=True):
    """Surface of revolution about the x axis: radius rs(x). Returns (v, f, normals)."""
    xs, rs = np.asarray(xs, float), np.asarray(rs, float)
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    V, N, F = [], [], []
    dr = np.gradient(rs, xs)
    for i, (x, r) in enumerate(zip(xs, rs)):
        nx = -dr[i]
        for t in th:
            V.append((x, r*math.cos(t), r*math.sin(t)))
            nn = np.array([nx, math.cos(t), math.sin(t)])
            N.append(nn/np.linalg.norm(nn))
    m = len(xs)
    for i in range(m - 1):
        for j in range(n):
            a, b, c, d = i*n + j, i*n + (j + 1) % n, (i + 1)*n + j, (i + 1)*n + (j + 1) % n
            F += [a, c, d, a, d, b]
    for cap, i, sgn in ((cap0, 0, -1), (cap1, m - 1, 1)):
        if cap and rs[i] > 1e-6:
            ci = len(V)
            V.append((xs[i], 0., 0.)); N.append((sgn, 0., 0.))
            for j in range(n):
                a, b = i*n + j, i*n + (j + 1) % n
                F += ([ci, b, a] if sgn < 0 else [ci, a, b])
    return np.array(V), np.array(F), np.array(N)


def cylinder_z(r, z0, z1, n=48, outward=True):
    th = np.linspace(0, 2*math.pi, n + 1)
    V, N, F, UV = [], [], [], []
    for k, t in enumerate(th):
        for z in (z0, z1):
            V.append((r*math.cos(t), r*math.sin(t), z))
            s = 1 if outward else -1
            N.append((s*math.cos(t), s*math.sin(t), 0.))
            UV.append((k/n*2*math.pi*r, z - z0))
    for k in range(n):
        a, b, c, d = 2*k, 2*k + 1, 2*k + 2, 2*k + 3
        F += ([a, c, d, a, d, b] if outward else [a, d, c, a, b, d])
    return np.array(V), np.array(F), np.array(N), np.array(UV)


def disc_z(r, z, n=64, up=True):
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    V = [(0., 0., z)] + [(r*math.cos(t), r*math.sin(t), z) for t in th]
    F = []
    for k in range(n):
        a, b = 1 + k, 1 + (k + 1) % n
        F += ([0, a, b] if up else [0, b, a])
    N = [(0., 0., 1. if up else -1.)]*len(V)
    UV = [(v[0], v[1]) for v in V]
    return np.array(V), np.array(F), np.array(N), np.array(UV)


def ring_z(r0, r1, z0, z1, n=64):
    """Closed annular ring (hoop): outer and inner walls, top and bottom."""
    out = []
    for args in ((r1, z0, z1, n, True), (r0, z0, z1, n, False)):
        out.append(cylinder_z(*args)[:3])
    th = np.linspace(0, 2*math.pi, n + 1)
    for z, up in ((z1, True), (z0, False)):
        V, F, N = [], [], []
        for t in th:
            V += [(r0*math.cos(t), r0*math.sin(t), z), (r1*math.cos(t), r1*math.sin(t), z)]
            N += [(0, 0, 1 if up else -1)]*2
        for k in range(n):
            a, b, c, d = 2*k, 2*k + 1, 2*k + 2, 2*k + 3
            F += ([a, b, d, a, d, c] if up else [a, d, b, a, c, d])
        out.append((np.array(V), np.array(F), np.array(N)))
    return _merge(out)


def _merge(parts):
    V, F, N, off = [], [], [], 0
    for v, f, nn in parts:
        V.append(v); N.append(nn); F.append(np.asarray(f).reshape(-1) + off)
        off += len(v)
    return np.vstack(V), np.concatenate(F), np.vstack(N)


def bar_mesh(bar, nseg=60):
    """The undercut bar in its body frame (x along the bar, top face z = +H0/2): flat top, the profile h(x) underneath.
    Returns (v, f, uv) with face-varying UVs in metres (top: x, y; sides: x, z)."""
    from .modes import bar_profile
    L = float(bar["length"])
    xs, h = bar_profile(L, float(bar["flat_half"]), float(bar["h_min"]), n=nseg)
    hx = np.r_[h[0], 0.5*(h[1:] + h[:-1]), h[-1]]
    xs = xs - L/2
    b = P.BAR_WIDTH_M/2
    zt = P.BAR_THICK_M/2
    zb = zt - hx
    V, F, UV = [], [], []

    def quad(p0, p1, p2, p3, uv4):
        i = len(V)
        V.extend([p0, p1, p2, p3]); UV.extend(uv4)
        F.extend([i, i + 1, i + 2, i, i + 2, i + 3])
    for k in range(len(xs) - 1):
        x0, x1 = xs[k], xs[k + 1]
        quad((x0, -b, zt), (x1, -b, zt), (x1, b, zt), (x0, b, zt), [(x0, -b), (x1, -b), (x1, b), (x0, b)])          # top
        quad((x0, b, zb[k]), (x1, b, zb[k + 1]), (x1, -b, zb[k + 1]), (x0, -b, zb[k]),
             [(x0, b), (x1, b), (x1, -b), (x0, -b)])                                                              # bottom
        quad((x0, b, zt), (x1, b, zt), (x1, b, zb[k + 1]), (x0, b, zb[k]), [(x0, zt), (x1, zt), (x1, zb[k + 1]), (x0, zb[k])])
        quad((x0, -b, zb[k]), (x1, -b, zb[k + 1]), (x1, -b, zt), (x0, -b, zt),
             [(x0, zb[k]), (x1, zb[k + 1]), (x1, zt), (x0, zt)])
    for x, k, s in ((xs[0], 0, -1), (xs[-1], len(xs) - 1, 1)):
        p = [(x, -b, zb[k]), (x, b, zb[k]), (x, b, zt), (x, -b, zt)]
        if s < 0:
            p = p[::-1]
        quad(*p, [(q[1], q[2]) for q in p])
    return np.array(V), np.array(F), np.array(UV)


def _label_texture(folder, text):
    from PIL import Image, ImageDraw, ImageFont
    W, H = 256, 128
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    font = ImageFont.truetype(f, 92) if Path(f).exists() else ImageFont.load_default()
    d.text((W/2, H/2), text, font=font, fill=(240, 236, 226, 255), anchor="mm")
    p = Path(folder)/f"label_{text}.png"
    im.save(p)
    return p


ORFF_NAMES = {"C4": "c'", "D4": "d'", "E4": "e'", "F4": "f'", "G4": "g'", "A4": "a'", "B4": "h'", "C5": "c''",
              "D5": "d''", "E5": "e''", "F5": "f''", "G5": "g''", "A5": "a''", "F#4": "fis'", "Bb4": "b'", "F#5": "fis''"}


# ====================================================================================================== export
def export(world, folder, seed=0):
    """Export the world's static appearance + posable bodies to folder/scene.usdc; returns (path, meta)."""
    import newton
    from . import dressing, visual_assets as va
    sc = world.scene
    b = sc.b
    folder = Path(folder)
    S = Stage(folder)
    for name, (rgb, rough, metal) in COLORS.items():
        S.color(name, rgb, rough, metal)
    S.textured("rosewood", va.texture("rosewood_veneer1"), 0.25, .38)
    S.textured("maple_tex", va.texture("white_maple_veneer"), 0.5, .45, tint=(0.88, 0.72, 0.52))
    S.textured("walnut_shell", va.texture("natural_walnut_veneer"), 0.5, .25, tint=(0.62, 0.40, 0.24))
    S.textured("oak_table", va.texture("oak_veneer_01"), 0.9, .45)
    S.textured("beech_tex", va.texture("white_oak_veneer"), 0.4, .5)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    V = int(newton.ShapeFlags.VISIBLE)
    special = {"xylo_box": "maple_tex", "beech": "beech_tex", "table_laminate": "oak_table", "maple": "maple_tex"}
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
        t = b.shape_type[i]
        if src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True)
        elif t == newton.GeoType.BOX:
            g = S.box(path, 2*scale, None, uv_per_m=1.)
        elif t == newton.GeoType.SPHERE:
            import trimesh
            m = trimesh.creation.icosphere(subdivisions=2, radius=float(scale[0]))
            g = S.mesh(path, m.vertices, m.faces, smooth=True)
        elif t in (newton.GeoType.CAPSULE, newton.GeoType.CYLINDER):
            import trimesh
            r, hh = float(scale[0]), float(scale[1])
            m = trimesh.creation.capsule(height=2*hh, radius=r, count=[16, 16]) if t == newton.GeoType.CAPSULE else \
                trimesh.creation.cylinder(radius=r, height=2*hh, sections=24)
            g = S.mesh(path, m.vertices, m.faces, smooth=True)
        else:
            continue
        xf = S.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(S.Gf.Vec3d(*tf[:3]))
        xf.AddOrientOp().Set(S.Gf.Quatf(float(tf[6]), S.Gf.Vec3f(*tf[3:6])))
        if mname == "robot":
            col = np.clip(np.asarray(b.shape_color[i], float), 0., 1.)**2.2
            key = "robot_%02x%02x%02x" % tuple(int(round(c*255)) for c in np.asarray(b.shape_color[i], float).clip(0, 1))
            mat = S.color(key, col, .28 if float(col.mean()) > 0.5 else .38, 0.)
        else:
            mat = S.mats.get(special.get(mname, mname)) or S.mats["robot_dark"]
        S.bind(g, mat)
    meta = dict(instruments=_instruments(S, world, folder))
    credits = dressing.dress(S, world)
    meta["cameras"] = _cameras(S, sc)
    meta["visual_assets"] = sorted(credits)
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1, default=str))
    return S.folder/"scene.usdc", meta


def _instruments(S, world, folder):
    """Authored visual meshes of the instruments and tools on their bodies."""
    from . import modes
    sc = world.scene
    out = {}
    for name, it in sc.items.items():
        body = it.get("body")
        base = f"/Lab/body_{body}"
        k = it["kind"]
        if k == "bar":
            bd = modes.bar(it["note"])
            v, f, uv = bar_mesh(bd)
            S.mesh(f"{base}/{name}_wood", v, f, uv=uv, mat=S.mats["rosewood"], uv_face=False)
            for j, xn in enumerate(sorted(float(x) for x in bd["nodes"])):
                hv, hf, hn, _ = disc_z(P.BAR_HOLE_R_M, P.BAR_THICK_M/2 + 0.0002, n=20)
                S.mesh(f"{base}/{name}_hole{j}", hv + np.array([xn, 0, 0]), hf, normals=hn, mat=S.mats["hole_dark"])
            tex = _label_texture(Path(folder)/"textures", ORFF_NAMES.get(it["note"], it["note"]))
            T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
            mat = S.mdl(f"bar_label_{it['note'].replace('#', 's')}", "OmniPBR", {
                "diffuse_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))), "enable_opacity": (T.Bool, True),
                "enable_opacity_texture": (T.Bool, True), "opacity_texture": (T.Asset, Sdf.AssetPath(str(tex.resolve()))),
                "opacity_mode": (T.Int, 0), "reflection_roughness_constant": (T.Float, .6)})
            L = float(bd["length"])
            x0 = -L/2 + 0.022                                   # near the player's end (bar -x end faces the robot)
            z = P.BAR_THICK_M/2 + 0.0003
            hw, hh = 0.012, 0.006
            S.mesh(f"{base}/{name}_label", [(x0 - hh, hw, z), (x0 - hh, -hw, z), (x0 + hh, -hw, z), (x0 + hh, hw, z)],
                   [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[(0, 0, 1)]*4, mat=mat)
            out[name] = "bar"
        elif k == "xylophone_frame":
            _xylophone_visual(S, base, sc)
            out[name] = "xylophone"
        elif k == "resonator_box":
            from .modal import resonator_bar_box
            Bx = resonator_bar_box(it["note"])
            S.box(f"{base}/box", (Bx["width"], Bx["length"], Bx["height"]), S.mats["maple_tex"], p=(0, 0, Bx["height"]/2),
                  q=(0, 0, 0, 1), bevel=0.004, uv_per_m=1.)
            S.box(f"{base}/slot", (Bx["slot_w"], Bx["slot_l"], 0.0006), S.mats["hole_dark"], p=(0, 0, Bx["height"] + 0.0003),
                  q=(0, 0, 0, 1), bevel=0.0002)
        elif k == "drum":
            _drum_visual(S, base, it)
            out[name] = "drum"
        elif k == "stick":
            from .modes import stick_geometry
            x, xm, d = stick_geometry(120)
            r = np.r_[d[0], 0.5*(d[1:] + d[:-1]), d[-1]]/2
            r[0] = r[1]*0.92
            r[-1] = 0.0015
            v, f, nn = lathe(x, r, n=32)
            S.mesh(f"{base}/{name}_wood", v, f, normals=nn, mat=S.mats["hickory"])
            out[name] = "stick"
        elif k in ("mallet", "beater"):
            M = P.MALLET if k == "mallet" else P.BEATER
            L, rh, rr = M["L"], M["handle_r"], M["head_r"]
            v, f, nn = lathe([0.0, 0.004, L - 2*rr + 0.004], [rh*0.9, rh, rh], n=24)
            S.mesh(f"{base}/{name}_handle", v, f, normals=nn, mat=S.mats["maple"])
            import trimesh
            m = trimesh.creation.icosphere(subdivisions=3, radius=rr)
            S.mesh(f"{base}/{name}_head", m.vertices + np.array([L - rr, 0, 0]), m.faces, smooth=True,
                   mat=S.mats["mallet_rubber" if k == "mallet" else "beater_felt"])
            out[name] = k
        elif k == "tray":
            pass
    return out


def _xylophone_visual(S, base, sc):
    """Orff alto xylophone box (frame coordinates: x along the instrument, low notes at -x): two side rails following
    the bars' node lines with felt strips on top, end blocks, a dark top deck with the resonator openings, a bottom
    plate and rubber feet. Visual only: the physics hull is three boxes (scene.xylophone)."""
    X = P.XYLO_BOX
    L, H = X["length"], X["height"]
    wl, wh = X["width_low"]/2, X["width_high"]/2
    rail_t, end_t = 0.016, 0.022
    wood = S.mats["maple_tex"]
    dark = S.color("xylo_deck", (.10, .07, .05), .8, 0.)
    felt = S.mats["felt"]
    for side in (-1, 1):
        # a side panel: slanted from the low end (width wl) to the high end (wh)
        dy = side*(wh - wl)
        ang = math.atan2(dy, L)
        yc = side*(wl + wh)/2
        q = (0., 0., math.sin(ang/2), math.cos(ang/2))
        Lr = math.hypot(L - 2*end_t, dy)
        S.box(f"{base}/xylo_rail_{side + 1}", (Lr, rail_t, H - 0.012), wood, p=(0., yc - side*rail_t/2, H/2 + 0.002), q=q,
              bevel=0.004, uv_per_m=1.)
        S.box(f"{base}/xylo_felt_{side + 1}", (Lr, 0.010, 0.004), felt, p=(0., yc - side*rail_t/2, H + 0.0005), q=q, bevel=0.0015)
    for end, w in ((-1, wl), (1, wh)):
        S.box(f"{base}/xylo_end_{end + 1}", (end_t, 2*w + 0.006, H + 0.006), wood, p=(end*(L/2 - end_t/2), 0., (H + 0.006)/2),
              q=(0, 0, 0, 1), bevel=0.006, uv_per_m=1.)
    pts = np.array([(-L/2 + end_t, -wl + rail_t, H - 0.006), (L/2 - end_t, -wh + rail_t, H - 0.006),
                    (L/2 - end_t, wh - rail_t, H - 0.006), (-L/2 + end_t, wl - rail_t, H - 0.006)])
    S.mesh(f"{base}/xylo_deck", pts, [0, 1, 2, 0, 2, 3], normals=[(0, 0, 1)]*4, mat=dark)
    bottom = np.array([(-L/2, -wl, 0.004), (L/2, -wh, 0.004), (L/2, wh, 0.004), (-L/2, wl, 0.004)])
    S.mesh(f"{base}/xylo_bottom", bottom[::-1], [0, 1, 2, 0, 2, 3], normals=[(0, 0, -1)]*4, mat=wood)
    for r_i, po in enumerate(sc.resonator_openings):
        Sres = P.RESONATORS[r_i][1]*1e-4
        w_ = 0.045
        l_ = Sres/w_
        S.box(f"{base}/opening_{r_i}", (l_, w_, 0.0006), S.mats["hole_dark"], p=(po[0], 0., H - 0.0055), q=(0, 0, 0, 1),
              bevel=0.0002)


def _drum_visual(S, base, it):
    """14 x 5 in shell (walnut veneer), white coated batter head, chrome triple-flanged hoop, eight lugs and rods."""
    a = P.DRUM_RADIUS_M
    depth = P.DRUM_SHELL_DEPTH_M
    v, f, n, uv = cylinder_z(a + 0.002, -depth, -0.004, n=96)
    S.mesh(f"{base}/shell_out", v, f, uv=uv, normals=n, mat=S.mats["walnut_shell"])
    v, f, n, uv = cylinder_z(a - 0.004, -depth, -0.004, n=96, outward=False)
    S.mesh(f"{base}/shell_in", v, f, normals=n, mat=S.mats["maple"])
    v, f, n = ring_z(a - 0.004, a + 0.002, -depth - 0.0005, -depth + 0.003, n=96)
    S.mesh(f"{base}/shell_edge", v, f, normals=n, mat=S.mats["maple"])
    v, f, n, uv = disc_z(a, 0.0, n=128)
    S.mesh(f"{base}/head", v, f, normals=n, mat=S.mats["head_coated"])
    v, f, n, uv = cylinder_z(a + 0.0005, -0.010, 0.0, n=96)
    S.mesh(f"{base}/head_collar", v, f, normals=n, mat=S.mats["head_coated"])
    r0, r1 = P.DRUM_HOOP_R_M
    v, f, n = ring_z(r0, r1, -0.012, P.DRUM_HOOP_H_M, n=128)
    S.mesh(f"{base}/hoop", v, f, normals=n, mat=S.mats["chrome"])
    for k in range(8):
        t = 2*math.pi*(k + 0.5)/8
        c, s = math.cos(t), math.sin(t)
        q = (0., 0., math.sin(t/2), math.cos(t/2))
        S.box(f"{base}/lug_{k}", (0.014, 0.022, 0.05), S.mats["chrome"], p=((a + 0.009)*c, (a + 0.009)*s, -0.055), q=q,
              bevel=0.005)
        S.box(f"{base}/claw_{k}", (0.010, 0.012, 0.010), S.mats["chrome"], p=(r1*c, r1*s, -0.006), q=q, bevel=0.002)
        import trimesh
        rod = trimesh.creation.cylinder(radius=0.0026, height=0.040, sections=12)
        S.mesh(f"{base}/rod_{k}", rod.vertices + np.array([(r1 + 0.004)*c, (r1 + 0.004)*s, -0.030]), rod.faces, smooth=True,
               mat=S.mats["chrome"])


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


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, world, folder, scene, *, width=1280, height=720, samples=16):
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
        env = dict(os.environ, PRC_RENDER_PORT=str(listener.getsockname()[1]), PRC_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "percussion.render_worker", "--scene", str(Path(scene).resolve()),
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

    def observe(self, cameras, body_q, tick, resolution=None, visibility=None):
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras),
                   visibility=visibility or {})
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


def cue_visibility(env):
    """Request visibility of the cue light's lit / unlit lens for the current state."""
    on = bool(getattr(env, "cue", {}).get("on"))
    return {"/Dressing/station/cue_lit": on, "/Dressing/station/cue_unlit": not on}

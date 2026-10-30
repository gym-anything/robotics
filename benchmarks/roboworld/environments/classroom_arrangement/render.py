"""USD export of room 114 and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses, so one exported stage serves a
whole session or a whole video. Appearance rules:
  * collision primitives of furniture are drawn as the parts they stand for (bevelled top, steel book box, round
    tubes, glides); the chair's seat and back are one moulded shell surface (the collision boxes are not drawn);
  * robots use the franka_description visual meshes with their own material colours; the TMR chassis is two-tone
    (black skirt, light grey deck) as in Franka's product images;
  * the floor is 12 x 12 in vinyl composition tile with seams; floor dots are 19 mm vinyl dots at the task's marks;
  * room: painted block walls, cove base, acoustic ceiling grid, six 2 x 4 ft troffers (rectangle lights), daylight
    through three window openings, the whiteboard, door, bookcase, cubbies and wall dressing (render-only).
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
from scipy.spatial.transform import Rotation

from . import params as P
from . import furniture as F
from . import layout as L

COLORS = dict(
    wall=((.80, .78, .72), .85, 0.), wall_front=((.78, .80, .76), .85, 0.), ceiling=((.90, .90, .88), .9, 0.),
    skirting=((.12, .12, .13), .6, 0.), board=((.95, .96, .96), .12, 0.), aluminum=((.70, .72, .74), .3, .85),
    door=((.50, .34, .20), .5, 0.), frame_white=((.88, .88, .86), .5, 0.), glass_dark=((.08, .10, .12), .05, 0.),
    bookcase=((.55, .40, .26), .55, 0.), cubby=((.62, .48, .32), .55, 0.),
    desk_steel=((.20, .21, .22), .45, .5), desk_frame=((.62, .63, .65), .35, .7), chrome=((.86, .87, .88), .12, 1.),
    chair_shell=((.11, .18, .40), .42, 0.), glide_nylon=((.06, .06, .06), .6, 0.), glide_felt=((.22, .20, .18), .95, 0.),
    glide_steel=((.70, .70, .72), .3, 1.), teacher_steel=((.30, .31, .33), .45, .4), tote=((.12, .32, .62), .5, 0.),
    basket=((.25, .27, .28), .6, 0.), book_0=((.62, .10, .10), .6, 0.), book_1=((.10, .35, .20), .6, 0.),
    book_2=((.85, .65, .15), .6, 0.), book_3=((.15, .25, .55), .6, 0.), robot=((.88, .88, .87), .35, 0.),
    robot_dark=((.07, .075, .08), .45, 0.), camera_dark=((.03, .03, .035), .45, 0.), tmr_black=((.05, .05, .055), .5, 0.),
    tmr_deck=((.80, .81, .80), .4, 0.), sill=((.88, .88, .86), .4, 0.), cork=((.60, .44, .28), .95, 0.),
    paper=((.93, .93, .90), .9, 0.), red=((.75, .12, .12), .6, 0.), blue=((.14, .30, .65), .6, 0.),
    page_edge=((.90, .88, .82), .85, 0.), book_label=((.94, .92, .84), .55, 0.), basket_inside=((.10, .105, .11), .75, 0.),
    yellow=((.92, .78, .22), .6, 0.), green=((.18, .52, .25), .6, 0.), black=((.03, .03, .03), .5, 0.),
    projector=((.90, .90, .88), .5, 0.), dot_green=((.10, .72, .25), .5, 0.), dot_orange=((.95, .45, .05), .5, 0.),
)


def _q_wxyz(q):
    return float(q[3]), (float(q[0]), float(q[1]), float(q[2]))


class Stage:
    """Thin USD authoring helper (barcode_scanning.render.Stage, trimmed)."""

    def __init__(self, folder):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.Gf, self.Sdf, self.Usd, self.UsdGeom, self.UsdLux, self.UsdShade, self.Vt = Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        self.folder = Path(folder)
        (self.folder/"textures").mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(self.folder/"scene.usdc"))
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/Lab").GetPrim())
        self.mats = {}
        self._n = 0

    def color(self, name, rgb, rough=.5, metal=0., emissive=None):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        s = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if emissive is not None:
            s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, emissive)))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def textured(self, name, texture, rough=.6, metal=0., normal=None, rough_tex=None, tint=None, normal_strength=1.):
        """OmniPBR with a diffuse texture and optional tangent-space normal map, roughness map and colour tint."""
        if name in self.mats:
            return self.mats[name]
        T, Sdf, UsdShade = self.Sdf.ValueTypeNames, self.Sdf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_texture", T.Asset).Set(Sdf.AssetPath(str(Path(texture).resolve())))
        sh.CreateInput("reflection_roughness_constant", T.Float).Set(float(rough))
        sh.CreateInput("metallic_constant", T.Float).Set(float(metal))
        if normal is not None:
            sh.CreateInput("normalmap_texture", T.Asset).Set(Sdf.AssetPath(str(Path(normal).resolve())))
            sh.CreateInput("bump_factor", T.Float).Set(float(normal_strength))
        if rough_tex is not None:
            sh.CreateInput("reflectionroughness_texture", T.Asset).Set(Sdf.AssetPath(str(Path(rough_tex).resolve())))
            sh.CreateInput("reflection_roughness_texture_influence", T.Float).Set(1.)
        if tint is not None:
            sh.CreateInput("diffuse_tint", T.Color3f).Set(self.Gf.Vec3f(*map(float, tint)))
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        self.mats[name] = m
        return m

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    def xform(self, path, p=None, q=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        if p is not None:
            x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            w, v = _q_wxyz(q)
            x.AddOrientOp().Set(self.Gf.Quatf(w, self.Gf.Vec3f(*v)))
        return x

    def place(self, g, p=None, q=None):
        if p is not None:
            xf = self.UsdGeom.Xformable(g)
            xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            w, v = _q_wxyz(q if q is not None else (0, 0, 0, 1))
            xf.AddOrientOp().Set(self.Gf.Quatf(w, self.Gf.Vec3f(*v)))
        return g

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, p=None, q=None):
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        v = np.asarray(v, np.float32); f = np.asarray(f, np.int32).reshape(-1, 3)
        g = UsdGeom.Mesh.Define(self.stage, path)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
        g.CreateFaceVertexCountsAttr([3]*len(f))
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
        return self.place(g, p, q)

    def box(self, path, size, mat, p=None, q=None, bevel=.002, uv_per_m=None):
        from whiteboard.usd_scene import bevel_box
        g = bevel_box(self.stage, path, np.asarray(size, float), min(bevel, float(min(size))*.2))
        if uv_per_m:
            self.box_uv(g, uv_per_m)
        if mat is not None:
            self.bind(g, mat)
        return self.place(g, p, q)

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

    def cylinder(self, path, r, h, mat, p=None, q=None, n=32, caps=True):
        v, nr, f = [], [], []
        for a in np.linspace(0, 2*math.pi, n + 1):
            d = np.array([math.cos(a), math.sin(a), 0.])
            for z in (-h/2, h/2):
                v.append(d*r + (0, 0, z)); nr.append(d)
        for i in range(n):
            a, b, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            f += [a, c, d, a, d, b]
        if caps:
            for z, up in ((-h/2, False), (h/2, True)):
                c0 = len(v); v.append((0, 0, z)); nr.append((0, 0, 1 if up else -1))
                ring = []
                for a in np.linspace(0, 2*math.pi, n, endpoint=False):
                    ring.append(len(v)); v.append((r*math.cos(a), r*math.sin(a), z)); nr.append((0, 0, 1 if up else -1))
                for i in range(n):
                    a_, b_ = ring[i], ring[(i + 1) % n]
                    f += [c0, a_, b_] if up else [c0, b_, a_]
        return self.mesh(path, v, f, normals=nr, mat=mat, p=p, q=q)

    def tube(self, path, pts, radius, mat, n=14, p=None, q=None):
        pts = np.asarray(pts, float)
        v, nr, f = [], [], []
        for i, pt in enumerate(pts):
            t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]; t /= np.linalg.norm(t)
            a = np.array([0., 0., 1.]) if abs(t[2]) < .9 else np.array([1., 0., 0.])
            u = np.cross(t, a); u /= np.linalg.norm(u); w_ = np.cross(t, u)
            for k in range(n):
                ang = 2*math.pi*k/n
                d = math.cos(ang)*u + math.sin(ang)*w_
                v.append(pt + radius*d); nr.append(d)
        for i in range(len(pts) - 1):
            for k in range(n):
                a, b = i*n + k, i*n + (k + 1) % n
                c, d = a + n, b + n
                f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, normals=nr, mat=mat, p=p, q=q)

    def disc(self, path, r, mat, p, n=24):
        v = [(0., 0., 0.)] + [(r*math.cos(a), r*math.sin(a), 0.) for a in np.linspace(0, 2*math.pi, n, endpoint=False)]
        f = []
        for i in range(n):
            f += [0, 1 + i, 1 + (i + 1) % n]
        return self.mesh(path, v, f, normals=[(0, 0, 1)]*len(v), mat=mat, p=p, q=(0, 0, 0, 1))

    def quad(self, path, centre, u, v, size, mat):
        c, u, v = (np.asarray(x, float) for x in (centre, u, v))
        a, b = size[0]/2, size[1]/2
        pts = [c - a*u - b*v, c + a*u - b*v, c + a*u + b*v, c - a*u + b*v]
        n = np.cross(u, v)
        return self.mesh(path, pts, [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], normals=[n]*4, mat=mat)

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), aperture=36., role="diagnostic"):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*map(float, eye)), self.Gf.Vec3d(*map(float, target)),
                                                            self.Gf.Vec3d(*map(float, up))).GetInverse())
        c.CreateHorizontalApertureAttr(float(aperture)); c.CreateVerticalApertureAttr(float(aperture*9/16))
        c.CreateFocalLengthAttr(float(focal_mm)); c.CreateClippingRangeAttr(self.Gf.Vec2f(.01, 40.))
        c.GetPrim().SetCustomDataByKey("observation_role", role)
        return c

    def alias(self, name, mounted):
        a = self.UsdGeom.Xform.Define(self.stage, "/Cameras/" + name)
        a.GetPrim().SetCustomDataByKey("mounted_camera_path", str(mounted))

    def save(self):
        self.stage.GetRootLayer().Save()


# ------------------------------------------------------------------------------ procedural textures
EXTERIOR_HDRI = "suburban_football_field"
HDRIS = {"suburban_football_field": dict(author="Grzegorz Wronkowski", license="CC0 (https://polyhaven.com/license)",
                                         sha256="38d31700e0d22b36ac3b74ee15c67cdc2c793e3a88bcafa7c69b67abf58e743f")}


def hdri(name):
    """Pinned CC0 Poly Haven HDR (2k), downloaded once into classroom_arrangement/.cache and hash-checked."""
    import hashlib
    from urllib.request import Request, urlopen
    path = Path(__file__).resolve().parent/".cache"/"visual_assets"/f"{name}_2k.hdr"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/{name}_2k.hdr"
        with urlopen(Request(url, headers={"User-Agent": "GymAnythingClassroom/1.0"}), timeout=60) as r:
            path.write_bytes(r.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != HDRIS[name]["sha256"]:
        raise ValueError(f"exterior HDR hash mismatch: {path}")
    return path


def clip_mesh(v, f, z0):
    """Split a triangle mesh at the plane z = z0: (vertices, faces below, faces above). Crossing triangles are cut
    along the plane (lone vertex: one triangle; the other side: a quad as two triangles), winding preserved."""
    s = v[f][:, :, 2] - z0
    side = s >= 0
    n_up = side.sum(axis=1)
    below, above = [f[n_up == 0]], [f[n_up == 3]]
    verts = [v]
    nv = len(v)
    new_b, new_a = [], []
    for tri, sd in zip(f[(n_up == 1) | (n_up == 2)], side[(n_up == 1) | (n_up == 2)]):
        lone = int(np.flatnonzero(sd != (sd.sum() >= 2))[0])
        a, b, c = tri[lone], tri[(lone + 1) % 3], tri[(lone + 2) % 3]
        pa, pb, pc = v[a], v[b], v[c]
        pab = pa + (pb - pa)*(z0 - pa[2])/(pb[2] - pa[2])
        pac = pa + (pc - pa)*(z0 - pa[2])/(pc[2] - pa[2])
        verts.append(np.array([pab, pac]))
        iab, iac = nv, nv + 1
        nv += 2
        lone_tris, other = [(a, iab, iac)], [(iab, b, c), (iab, c, iac)]
        if sd[lone]:
            new_a += lone_tris; new_b += other
        else:
            new_b += lone_tris; new_a += other
    if new_b:
        below.append(np.array(new_b))
    if new_a:
        above.append(np.array(new_a))
    return np.vstack(verts), np.vstack(below), np.vstack(above)


def textures(folder):
    """Floor tile, desk top, laminates, cork, ceiling: generated once, deterministic."""
    from PIL import Image, ImageDraw, ImageFilter
    folder = Path(folder)
    rng = np.random.default_rng(114)
    out = {}
    # VCT: 8 x 8 tiles of 12 in (304.8 mm) at 128 px/tile, off-white with a darker fleck, per-tile shade, 1 px seams
    n, px = 8, 128
    img = np.zeros((n*px, n*px, 3), np.float32)
    base = np.array([.74, .73, .69])
    for i in range(n):
        for j in range(n):
            shade = base*(1 + rng.normal(0, .025)) + (rng.normal(0, .008, 3))
            tile = np.ones((px, px, 3))*shade
            fleck = rng.random((px, px)) < .05
            tile[fleck] *= rng.uniform(.65, .9, (fleck.sum(), 1))
            img[i*px:(i + 1)*px, j*px:(j + 1)*px] = tile
    img = np.array(Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6)), np.float32)/255
    for k in range(n):
        img[k*px, :] *= .78; img[:, k*px] *= .78
    out["vct"] = folder/"vct.png"
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(out["vct"])
    # Virco "gray nebula" hard-plastic top: light grey with fine dark and white speckle
    t = np.ones((512, 512, 3))*np.array([.62, .62, .60])
    s = rng.random((512, 512))
    t[s < .06] *= .55; t[s > .96] = .85
    out["desk_top"] = folder/"desk_top.png"
    Image.fromarray((np.clip(t, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.5)).save(out["desk_top"])
    # wood laminates (teacher's desk walnut, tutoring table maple): grain along u
    for name, col, contrast in (("walnut", (.42, .27, .16), .18), ("maple", (.78, .62, .42), .10)):
        x = np.linspace(0, 1, 1024)
        grain = np.sin(x[None, :]*90 + 3*np.sin(np.linspace(0, 6, 256))[:, None] + rng.normal(0, .3, (256, 1)))
        noise = rng.normal(0, 1, (256, 1024))
        from scipy.ndimage import gaussian_filter
        v = 1 + contrast*(.6*grain + .25*gaussian_filter(noise, 2))
        tex = np.clip(np.asarray(col)[None, None, :]*v[:, :, None], 0, 1)
        out[name] = folder/f"{name}.png"
        Image.fromarray((tex*255).astype(np.uint8)).save(out[name])
    # cork
    c = np.ones((256, 256, 3))*np.array([.60, .45, .29])*(1 + .25*rng.normal(0, 1, (256, 256, 1)))
    out["cork"] = folder/"cork.png"
    Image.fromarray((np.clip(c, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.8)).save(out["cork"])
    # acoustic ceiling tile (2 x 2 ft) with fissures
    ct = np.ones((256, 256, 3))*.90
    ct *= (1 - .08*(rng.random((256, 256, 1)) < .04))
    out["ceiling"] = folder/"ceiling.png"
    Image.fromarray((np.clip(ct, 0, 1)*255).astype(np.uint8)).save(out["ceiling"])
    # posters and the alphabet strip: simple printed sheets (render-only dressing)
    for k, (bg, fg) in enumerate((((.95, .93, .85), (.15, .30, .60)), ((.20, .45, .70), (.98, .95, .80)),
                                  ((.98, .85, .30), (.55, .15, .10)), ((.30, .60, .35), (.97, .97, .95)))):
        im = Image.new("RGB", (400, 560), tuple(int(255*x) for x in bg))
        d = ImageDraw.Draw(im)
        for r in range(6):
            d.rectangle((40, 60 + r*80, 360, 90 + r*80), fill=tuple(int(255*x) for x in fg))
        d.ellipse((120, 20, 280, 50), fill=tuple(int(255*x) for x in fg))
        out[f"poster_{k}"] = folder/f"poster_{k}.png"
        im.save(out[f"poster_{k}"])
    im = Image.new("RGB", (2600, 200), (250, 250, 245))
    d = ImageDraw.Draw(im)
    for k, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        col = [(200, 40, 40), (40, 90, 190), (40, 150, 70), (220, 160, 30)][k % 4]
        d.rectangle((k*100 + 8, 20, k*100 + 92, 180), outline=col, width=6)
        d.text((k*100 + 38, 80), ch, fill=col)
    out["alphabet"] = folder/"alphabet.png"
    im.save(out["alphabet"])
    clock = Image.new("RGB", (256, 256), (245, 245, 240))
    d = ImageDraw.Draw(clock)
    d.ellipse((4, 4, 252, 252), outline=(20, 20, 20), width=10)
    for k in range(12):
        a = k*math.pi/6
        d.line((128 + 100*math.sin(a), 128 - 100*math.cos(a), 128 + 115*math.sin(a), 128 - 115*math.cos(a)), fill=(20, 20, 20), width=6)
    d.line((128, 128, 128 + 60*math.sin(1.9), 128 - 60*math.cos(1.9)), fill=(10, 10, 10), width=8)
    d.line((128, 128, 128 + 95*math.sin(4.3), 128 - 95*math.cos(4.3)), fill=(10, 10, 10), width=5)
    out["clock"] = folder/"clock.png"
    clock.save(out["clock"])
    out.update(textures_v2(folder, rng))
    return out


def _normal_map(h, strength, path):
    """Tangent-space normal map (OpenGL convention) from a height field in metres per pixel units."""
    from PIL import Image
    gy, gx = np.gradient(h)
    n = np.dstack([-gx*strength, gy*strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n*0.5 + 0.5)*255).astype(np.uint8)).save(path)
    return path


def textures_v2(folder, rng):
    """Painted concrete-block walls, VCT relief, speckled tops, ceiling fissures, printed posters (deterministic)."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import gaussian_filter
    out = {}
    font = lambda size, bold=True: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", size)
    # painted CMU: 8 x 16 in faces (203 x 406 mm), running bond, 10 mm mortar joints tooled 3 mm deep, eggshell paint;
    # one texture = 4 blocks x 8 courses = 1.6256 m square at 640 px/m
    W = 1040
    ppm = W/1.6256
    h = gaussian_filter(rng.normal(0, 1, (W, W)), 1.2)*0.25 + gaussian_filter(rng.normal(0, 1, (W, W)), 6)*0.5
    course, block, joint = 0.2032*ppm, 0.4064*ppm, 0.010*ppm
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    row = np.floor(yy/course)
    xs = (xx + (row % 2)*block/2) % block
    ys = yy % course
    mortar = (np.minimum(xs, block - xs) < joint/2) | (np.minimum(ys, course - ys) < joint/2)
    depth = gaussian_filter(mortar.astype(np.float32), 1.5)
    h = h*0.6 - depth*3.0
    col = np.array([.84, .81, .74])*(1 + 0.012*gaussian_filter(rng.normal(0, 1, (W, W)), 3))[..., None]
    col = col*(1 - 0.06*depth[..., None])
    out["cmu"] = folder/"cmu.png"
    Image.fromarray((np.clip(col, 0, 1)*255).astype(np.uint8)).save(out["cmu"])
    out["cmu_n"] = _normal_map(h, 0.9, folder/"cmu_n.png")
    # VCT relief: recessed 1 px seams and faint wax swirl (same 8 x 8 tile sheet as the diffuse)
    Wv = 1024
    hv = gaussian_filter(rng.normal(0, 1, (Wv, Wv)), 8)*0.3
    for k in range(8):
        hv[k*128:k*128 + 2, :] -= 2.0; hv[:, k*128:k*128 + 2] -= 2.0
    out["vct_n"] = _normal_map(gaussian_filter(hv, .8), 0.6, folder/"vct_n.png")
    r = np.clip(0.32 + 0.10*gaussian_filter(rng.normal(0, 1, (512, 512)), 20) + 0.05*rng.random((512, 512)), 0, 1)
    out["vct_r"] = folder/"vct_r.png"
    Image.fromarray((r*255).astype(np.uint8)).save(out["vct_r"])
    # Virco "gray nebula" top at 1 texture per 0.30 m: fine dark and light flecks on a warm light grey, slight mottling
    Wt = 768
    base = np.array([.70, .70, .68])*(1 + 0.03*gaussian_filter(rng.normal(0, 1, (Wt, Wt)), 12))[..., None]
    s = rng.random((Wt, Wt))
    base[s < .05] *= .45; base[(s > .05) & (s < .08)] *= .7; base[s > .975] = .92
    out["desk_top2"] = folder/"desk_top2.png"
    Image.fromarray((np.clip(gaussian_filter(base, (0.6, 0.6, 0)), 0, 1)*255).astype(np.uint8)).save(out["desk_top2"])
    out["desk_top2_n"] = _normal_map(gaussian_filter(rng.normal(0, 1, (Wt, Wt)), 1.0)*0.4, 0.5, folder/"desk_top2_n.png")
    # acoustic ceiling tile, 2 x 2 ft: pinholes and fissures, bevelled edge shading
    Wc = 512
    c = np.ones((Wc, Wc))*.92
    c -= 0.10*(rng.random((Wc, Wc)) < .03)
    fiss = gaussian_filter(rng.normal(0, 1, (Wc, Wc)), 3) > 1.6
    c -= 0.08*fiss
    edge = np.minimum.reduce([np.arange(Wc)[None, :].repeat(Wc, 0), np.arange(Wc)[::-1][None, :].repeat(Wc, 0),
                              np.arange(Wc)[:, None].repeat(Wc, 1), np.arange(Wc)[::-1][:, None].repeat(Wc, 1)])
    c -= 0.10*np.exp(-edge/6.)
    out["ceiling2"] = folder/"ceiling2.png"
    Image.fromarray((np.clip(np.dstack([c, c, c*0.985]), 0, 1)*255).astype(np.uint8)).save(out["ceiling2"])
    out["ceiling2_n"] = _normal_map(gaussian_filter(-(rng.random((Wc, Wc)) < .03).astype(float) - fiss*0.8, 1.0), 1.5, folder/"ceiling2_n.png")
    # printed posters with real content (render-only dressing)
    posters = []
    im = Image.new("RGB", (600, 840), (250, 248, 240)); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 600, 120), fill=(30, 70, 140)); d.text((30, 30), "CLASSROOM", font=font(44), fill="white")
    d.text((30, 78), "EXPECTATIONS", font=font(30), fill=(255, 220, 90))
    for k, line in enumerate(("Be on time", "Be prepared", "Be respectful", "Raise your hand", "Do your best",
                              "Push in your chair")):
        d.ellipse((40, 170 + k*105, 90, 220 + k*105), fill=(230, 90, 40)); d.text((54, 178 + k*105), str(k + 1), font=font(30), fill="white")
        d.text((110, 175 + k*105), line, font=font(38, False), fill=(30, 30, 40))
    posters.append(im)
    im = Image.new("RGB", (600, 840), (235, 240, 248)); d = ImageDraw.Draw(im)
    d.text((40, 30), "PERIODIC TABLE", font=font(40), fill=(20, 40, 80))
    cols = [(240, 120, 100), (250, 200, 90), (140, 200, 120), (120, 170, 230), (190, 150, 220)]
    sym = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()
    for k, s_ in enumerate(sym):
        x, y = 40 + (k % 6)*88, 120 + (k // 6)*110
        d.rectangle((x, y, x + 80, y + 100), fill=cols[k % 5], outline=(40, 40, 60), width=2)
        d.text((x + 12, y + 30), s_, font=font(34), fill=(20, 20, 30)); d.text((x + 6, y + 4), str(k + 1), font=font(16, False), fill=(20, 20, 30))
    posters.append(im)
    im = Image.new("RGB", (600, 840), (250, 214, 80)); d = ImageDraw.Draw(im)
    d.text((40, 40), "GROWTH", font=font(70), fill=(140, 30, 20)); d.text((40, 120), "MINDSET", font=font(70), fill=(140, 30, 20))
    for k, line in enumerate(("Mistakes help me learn.", "I can't do it... yet.", "Effort makes me stronger.",
                              "I like a challenge.")):
        d.text((40, 290 + k*120), line, font=font(34, False), fill=(40, 30, 20))
    posters.append(im)
    im = Image.new("RGB", (600, 840), (60, 120, 170)); d = ImageDraw.Draw(im)
    d.text((40, 30), "THE WORLD", font=font(56), fill="white")
    for cx, cy, rx, ry in ((150, 300, 90, 120), (190, 520, 60, 110), (330, 280, 60, 70), (350, 450, 70, 130),
                           (480, 300, 100, 80), (500, 560, 50, 40)):
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(110, 170, 90))
    d.text((40, 760), "Continents and Oceans", font=font(30, False), fill="white")
    posters.append(im)
    for k, im in enumerate(posters):
        out[f"poster2_{k}"] = folder/f"poster2_{k}.png"
        im.save(out[f"poster2_{k}"])
    im = Image.new("RGB", (2600, 200), (250, 250, 245)); d = ImageDraw.Draw(im)
    for k, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        col = [(200, 40, 40), (40, 90, 190), (40, 150, 70), (220, 160, 30)][k % 4]
        d.rectangle((k*100 + 6, 14, k*100 + 94, 186), fill=col)
        d.text((k*100 + 22, 30), ch + ch.lower(), font=font(62), fill="white")
    out["alphabet2"] = folder/"alphabet2.png"
    im.save(out["alphabet2"])
    return out


# ------------------------------------------------------------------------------ furniture appearance
def chair_shell(S, path, mat):
    """Virco 9018 one-piece shell: a seat panel with a waterfall front, curving up into a contoured backrest."""
    seat_z = P.CHAIR_SEAT_H
    prof = []                                   # (y, z) along the shell's centre line, front to top of the back
    for a in np.linspace(math.pi, 1.5*math.pi, 6):                 # waterfall front edge
        prof.append((-0.185 + 0.03*math.cos(a), seat_z - 0.03 - 0.03*math.sin(a) - 0.03))
    for y in np.linspace(-0.185, 0.15, 8)[1:]:
        prof.append((y, seat_z - 0.006 - 0.012*(1 - ((y + 0.02)/0.17)**2)))
    for a in np.linspace(0, math.pi/2 - math.radians(12), 7)[1:]:   # bend up into the back
        prof.append((0.15 + 0.07*math.sin(a), seat_z - 0.006 + 0.07*(1 - math.cos(a))))
    y0, z0 = prof[-1]
    for t in np.linspace(0, 1, 8)[1:]:
        prof.append((y0 + (0.775 - z0)*t*math.tan(math.radians(12)), z0 + (0.775 - z0)*t))
    prof = np.array(prof)
    xs = np.linspace(-1, 1, 15)
    v, uv = [], []
    for i, (y, z) in enumerate(prof):
        back = max(0., (z - seat_z)/(0.775 - seat_z))
        half = P.CHAIR_SEAT_HALF_W - 0.02*back + 0.005*math.sin(math.pi*back)
        for x in xs:
            curve = 0.035*back*x*x                      # the back wraps around the sitter
            v.append((x*half, y + curve, z)); uv.append((x, i/len(prof)))
    m = len(xs)
    f = []
    for i in range(len(prof) - 1):
        for j in range(m - 1):
            a, b, c, d = i*m + j, i*m + j + 1, (i + 1)*m + j, (i + 1)*m + j + 1
            f += [a, c, d, a, d, b]
    v = np.array(v)
    nrm = np.zeros_like(v)
    import trimesh
    tm = trimesh.Trimesh(v, np.array(f).reshape(-1, 3), process=False)
    nrm = tm.vertex_normals
    off = 0.006
    v2 = np.vstack([v + nrm*off/2, v - nrm*off/2])
    f2 = np.array(f).reshape(-1, 3)
    faces = np.vstack([f2, f2[:, ::-1] + len(v)])
    # close the shell's rim
    rim = [i*m for i in range(len(prof))] + [(len(prof) - 1)*m + j for j in range(m)] + \
          [i*m + m - 1 for i in reversed(range(len(prof)))] + [j for j in reversed(range(m))]
    side = []
    for a, b in zip(rim, rim[1:]):
        side += [[a, b, b + len(v)], [a, b + len(v), a + len(v)]]
    faces = np.vstack([faces, np.array(side)])
    return S.mesh(path, v2, faces, mat=mat, smooth=True)


def basket_visual(S, path, p, mat, mats):
    """Render-only open wastebasket: wall, floor, a dark inside and a rolled rim (the collision body is the closed
    cylinder; nothing is thrown into it in these tasks)."""
    r, h = p.size
    x, y, z = p.pos
    S.cylinder(path + "/wall", r, 2*h, mat, (x, y, z), caps=False)
    S.cylinder(path + "/floor", r, 0.006, mat, (x, y, z - h + 0.003))
    S.cylinder(path + "/inside", r - 0.004, 2*h - 0.012, mats["basket_inside"], (x, y, z + 0.006), caps=False)
    S.cylinder(path + "/inside_floor", r - 0.004, 0.002, mats["basket_inside"], (x, y, z - h + 0.012))
    S.tube(path + "/rim", [(x + (r + 0.002)*math.cos(a), y + (r + 0.002)*math.sin(a), z + h - 0.004)
                            for a in np.linspace(0, 2*math.pi, 49)], 0.005, mat)


def book_visual(S, path, p, mat, mats):
    """Render-only hardcover book: coloured boards and spine, cream page edges on three sides, a printed title band."""
    sx, sy, sz = p.size
    x, y, z = p.pos
    S.box(path + "/cover", (2*sx, 2*sy, 2*sz), mat, p.pos, p.quat, bevel=0.002)
    e = 0.0006
    S.box(path + "/fore", (0.002, 2*sy - 0.008, 2*sz - 0.006), mats["page_edge"], (x + sx + e - 0.001, y, z), p.quat)
    for s in (-1, 1):
        S.box(path + f"/edge{'ab'[s > 0]}", (2*sx - 0.010, 0.002, 2*sz - 0.006), mats["page_edge"],
              (x + 0.003, y + s*(sy + e - 0.001), z), p.quat)
    S.box(path + "/title", (1.2*sx, 0.35*sy, 0.0008), mats["book_label"], (x + 0.1*sx, y + 0.35*sy, z + sz + 0.0004), p.quat)
    S.box(path + "/spine_band", (0.0008, 1.4*sy, 0.4*sz), mats["book_label"], (x - sx - 0.0004, y, z), p.quat)


def furniture_visual(S, base, kind, parts, mats, glide):
    for p in parts:
        mat = mats.get(p.material) or mats["desk_frame"]
        path = f"{base}/{p.name}"
        if kind == "chair" and p.role in ("seat", "back"):
            continue
        if p.role == "basket":
            basket_visual(S, path, p, mat, mats)
            continue
        if p.role == "contents" and p.kind == "box":
            book_visual(S, path, p, mat, mats)
            continue
        if p.kind == "box":
            size = 2*np.asarray(p.size)
            bevel = .006 if p.role == "top" else .002
            S.box(path, size, mat, p.pos, p.quat, bevel=bevel, uv_per_m=(1.6 if p.role == "top" else None))
        elif p.kind == "capsule":
            r, h = p.size
            R = Rotation.from_quat(p.quat)
            a = np.asarray(p.pos) - R.apply((0, 0, h + r*.6)); b = np.asarray(p.pos) + R.apply((0, 0, h + r*.6))
            S.tube(path, [a, b], r, mat)
        elif p.kind == "cylinder":
            r, h = p.size
            S.cylinder(path, r, 2*h, mat, p.pos, p.quat)
    if kind == "chair":
        chair_shell(S, f"{base}/shell", mats["chair_shell"])


# ------------------------------------------------------------------------------ export
def export(world, folder, extra_cameras=None):
    """Export the world's static appearance and posable bodies to folder/scene.usdc; returns (path, meta)."""
    import newton
    sc = world.scene
    b = sc.b
    S = Stage(folder)
    tex = textures(S.folder/"textures")
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    mats["vct"] = S.textured("vct", tex["vct"], rough=.35, normal=tex["vct_n"], rough_tex=tex["vct_r"], normal_strength=.6)
    mats["desk_top"] = S.textured("desk_top_tex", tex["desk_top2"], rough=.45, normal=tex["desk_top2_n"], normal_strength=.4)
    # pinned CC0 oak veneer (Poly Haven, the whiteboard classroom's asset set): maple-tinted on the tutoring table,
    # walnut-tinted on the teacher's desk laminate
    from whiteboard.visual_assets import asset
    oak = {k: asset("oak_veneer_01", k) for k in ("diff", "rough", "nor_gl")}
    mats["teacher_top"] = S.textured("walnut_pbr", oak["diff"], rough=.45, normal=oak["nor_gl"], rough_tex=oak["rough"],
                                     tint=(.38, .26, .18))
    mats["table_top"] = S.textured("maple_pbr", oak["diff"], rough=.45, normal=oak["nor_gl"], rough_tex=oak["rough"],
                                   tint=(1.05, 0.98, 0.88))
    mats["cork_tex"] = S.textured("cork_tex", tex["cork"], rough=.95)
    mats["ceiling_tex"] = S.textured("ceiling_tex", tex["ceiling2"], rough=.95, normal=tex["ceiling2_n"], normal_strength=.8)
    mats["wall"] = S.textured("cmu", tex["cmu"], rough=.75, normal=tex["cmu_n"], normal_strength=1.)
    mats["wall_front"] = S.textured("cmu_front", tex["cmu"], rough=.75, normal=tex["cmu_n"], tint=(.98, 1.0, 1.02))
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    # ---- furniture (drawn from the same part list as the physics)
    for name, it in sc.items.items():
        furniture_visual(S, f"/Lab/body_{it['body']}/{name}", it["kind"], sc.parts[name], mats, world.spec["glides"])
    # ---- robot: visual meshes with their material colours (collision hulls never drawn)
    V, C = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    robot_shapes = set(sc.robot_shapes)
    for i in range(b.shape_count):
        if i not in robot_shapes:
            continue
        flags = int(b.shape_flags[i])
        body = b.shape_body[i]
        label = b.body_label[body].split("/")[-1] if body >= 0 else ""
        mname = sc.materials.get(i)
        if mname in ("camera_dark", "aluminum"):
            tf = np.asarray(b.shape_transform[i], float)
            S.box(f"/Lab/body_{body}/shape_{i}", 2*np.asarray(b.shape_scale[i], float), mats[mname], tf[:3], tf[3:])
            continue
        if flags & C or not flags & V:
            continue
        src = b.shape_source[i]
        tf = np.asarray(b.shape_transform[i], float)
        scale = np.asarray(b.shape_scale[i], float)
        path = f"/Lab/body_{body}/shape_{i}"
        col = tuple(float(x) for x in b.shape_color[i])
        if src is not None and hasattr(src, "vertices"):
            v = np.asarray(src.vertices)*scale
            if np.abs(v).max() > 5.:
                v = v*1e-3        # the TMR chassis and Spine column DAEs carry millimetre coordinates without a unit tag
            f = np.asarray(src.indices).reshape(-1, 3)
            if label == "base_link" and len(v) > 100000:              # TMR chassis: black skirt, light deck
                # triangles crossing z = 0.26 m are clipped at that height (assigning whole triangles by centroid
                # gives a sawtooth boundary)
                v2, below, above = clip_mesh(v, f, 0.26)
                for part, faces, m in (("skirt", below, mats["tmr_black"]), ("deck", above, mats["tmr_deck"])):
                    S.mesh(f"{path}_{part}", v2, faces, mat=m, smooth=True, p=tf[:3], q=tf[3:])
                continue
            if label == "base_link":                                    # Spine column (its DAE colours are placeholders)
                m = mats["robot"]
            else:
                key = f"robot_{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
                m = S.color(key, col, .35, 0.)
            S.mesh(path, v, f, mat=m, smooth=True, p=tf[:3], q=tf[3:])
        elif b.shape_type[i] == newton.GeoType.BOX:
            S.box(path, 2*scale, mats["tmr_black"], tf[:3], tf[3:])
        elif b.shape_type[i] == newton.GeoType.CYLINDER:
            S.cylinder(path, scale[0], 2*scale[1], mats["robot_dark"], tf[:3], tf[3:])
    # ---- room
    room(S, mats, tex)
    dots(S, world.spec, mats)
    meta = dict(cameras=cameras(S, world))
    for name, (eye, target, f) in (extra_cameras or {}).items():
        if name in meta["cameras"]:
            raise ValueError(f"extra camera {name!r} would replace the built-in camera of that name; rename it")
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); meta["cameras"][name] = "diagnostic"
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def room(S, mats, tex):
    X, Y, Z = P.ROOM_X, P.ROOM_Y, P.ROOM_Z
    UsdLux, Gf = S.UsdLux, S.Gf
    # floor: VCT quad, UVs in tile-sheet units (8 tiles = 2.4384 m)
    sheet = 8*0.3048
    pts = [(-X/2, 0, 0), (X/2, 0, 0), (X/2, Y, 0), (-X/2, Y, 0)]
    S.mesh("/Room/floor", pts, [0, 1, 2, 0, 2, 3], uv=[(p[0]/sheet, p[1]/sheet) for p in pts], normals=[(0, 0, 1)]*4,
           mat=mats["vct"])
    T = L.WALL_T
    cmu = 1/1.6256
    S.box("/Room/wall_front", (X + 2*T, T, Z), mats["wall_front"], (0, -T/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/wall_back", (X + 2*T, T, Z), mats["wall"], (0, Y + T/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/wall_right", (T, Y, Z), mats["wall"], (X/2 + T/2, Y/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/ceiling", (X + 2*T, Y + 2*T, 0.05), mats["ceiling_tex"], (0, Y/2, Z + 0.025), (0, 0, 0, 1), uv_per_m=1/0.6096)
    for k in range(13):
        S.box(f"/Room/grid_x_{k}", (0.015, Y, 0.01), mats["frame_white"], (-X/2 + 0.1 + k*0.6096, Y/2, Z - 0.005), (0, 0, 0, 1))
    for k in range(16):
        S.box(f"/Room/grid_y_{k}", (X, 0.015, 0.01), mats["frame_white"], (0, 0.3 + k*0.6096, Z - 0.005), (0, 0, 0, 1))
    # left wall with three window openings (sill 0.90 m, head 2.60 m)
    wx = -X/2 - T/2
    S.box("/Room/wall_left_low", (T, Y, 0.90), mats["wall"], (wx, Y/2, 0.45), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/wall_left_head", (T, Y, Z - 2.60), mats["wall"], (wx, Y/2, (Z + 2.60)/2), (0, 0, 0, 1), uv_per_m=cmu)
    wins = (1.70, 4.80, 7.90)
    edges = [0.] + [e for c in wins for e in (c - 1.15, c + 1.15)] + [Y]
    for k in range(0, len(edges), 2):
        a, b_ = edges[k], edges[k + 1]
        if b_ - a > 1e-3:
            S.box(f"/Room/pier_{k}", (T, b_ - a, 1.70), mats["wall"], (wx, (a + b_)/2, 1.75), (0, 0, 0, 1), uv_per_m=cmu)
    for k, c in enumerate(wins):
        S.box(f"/Room/sill_{k}", (0.22, 2.36, 0.04), mats["sill"], (-X/2 + 0.04, c, 0.90), (0, 0, 0, 1))
        for j, yy in enumerate((c - 1.15, c - 0.385, c + 0.385, c + 1.15)):
            S.box(f"/Room/mullion_{k}_{j}", (0.06, 0.05, 1.70), mats["frame_white"], (wx, yy, 1.75), (0, 0, 0, 1))
        S.box(f"/Room/transom_{k}", (0.06, 2.30, 0.05), mats["frame_white"], (wx, c, 2.10), (0, 0, 0, 1))
    # outside: the sky dome carries a pinned CC0 HDR of a suburban sports field (suburban_football_field, Poly Haven),
    # seen through the windows
    # cove base
    S.box("/Room/base_front", (X, 0.012, 0.10), mats["skirting"], (0, 0.006, 0.05), (0, 0, 0, 1))
    S.box("/Room/base_back", (X, 0.012, 0.10), mats["skirting"], (0, Y - 0.006, 0.05), (0, 0, 0, 1))
    S.box("/Room/base_right", (0.012, Y, 0.10), mats["skirting"], (X/2 - 0.006, Y/2, 0.05), (0, 0, 0, 1))
    S.box("/Room/base_left", (0.012, Y, 0.10), mats["skirting"], (-X/2 + 0.006, Y/2, 0.05), (0, 0, 0, 1))
    # installed fixtures (physics: layout.STATIC)
    for name, c, h, mat in L.STATIC:
        if name.startswith("wall"):
            continue
        m = mats.get(mat) or mats["frame_white"]
        S.box(f"/Room/{name}", 2*np.asarray(h), m, c, (0, 0, 0, 1), bevel=0.004)
    for k, x in enumerate((-1.22, 1.22)):
        S.box(f"/Room/board_trim_{k}", (0.02, 0.03, 1.24), mats["aluminum"], (x, 0.02, 1.524), (0, 0, 0, 1))
    for k, z in enumerate((0.914, 2.134)):
        S.box(f"/Room/board_trim_h_{k}", (2.46, 0.03, 0.02), mats["aluminum"], (0, 0.02, z), (0, 0, 0, 1))
    S.box("/Room/door_frame_l", (0.06, 0.08, 2.15), mats["frame_white"], (P.DOOR_X - P.DOOR_LEAF/2 - 0.03, 0.03, 1.075), (0, 0, 0, 1))
    S.box("/Room/door_frame_r", (0.06, 0.08, 2.15), mats["frame_white"], (P.DOOR_X + P.DOOR_LEAF/2 + 0.03, 0.03, 1.075), (0, 0, 0, 1))
    S.box("/Room/door_frame_t", (P.DOOR_LEAF + 0.12, 0.08, 0.06), mats["frame_white"], (P.DOOR_X, 0.03, 2.11), (0, 0, 0, 1))
    S.box("/Room/door_window", (0.22, 0.01, 0.55), mats["glass_dark"], (P.DOOR_X - 0.25, 0.045, 1.55), (0, 0, 0, 1))
    S.tube("/Room/door_handle", [(P.DOOR_X + 0.30, 0.10, 1.00), (P.DOOR_X + 0.42, 0.10, 1.00)], 0.012, mats["chrome"])
    # bookcase shelves with books, cubby bins
    for k, z in enumerate((0.35, 0.75, 1.15, 1.55)):
        rng = np.random.default_rng(k)
        x = 2.55
        while x < 3.35:
            w = rng.uniform(0.025, 0.05); h = rng.uniform(0.20, 0.30)
            S.box(f"/Room/books_{k}_{int(x*1000)}", (w, 0.20, h), mats[f"book_{rng.integers(4)}"], (x + w/2, 9.24, z + h/2 + 0.01), (0, 0, 0, 1))
            x += w + 0.003
    for k in range(4):
        for j in range(2):
            S.box(f"/Room/bin_{k}_{j}", (0.40, 0.32, 0.30), mats[("red", "blue", "yellow", "green")[(k + j) % 4]],
                  (0.85 + k*0.5, 9.20, 0.18 + j*0.44), (0, 0, 0, 1))
    # wall dressing: bulletin board, posters, alphabet strip, clock, flag, projector, cart-free front
    S.box("/Room/bulletin", (0.02, 1.80, 1.10), mats["cork_tex"], (X/2 - 0.01, 5.20, 1.55), (0, 0, 0, 1), uv_per_m=1.)
    S.box("/Room/bulletin_trim", (0.025, 1.84, 1.14), mats["aluminum"], (X/2 - 0.005, 5.20, 1.55), (0, 0, 0, 1))
    rng = np.random.default_rng(7)
    for k in range(9):
        S.box(f"/Room/note_{k}", (0.003, 0.215, 0.28), mats[("paper", "yellow", "paper", "blue")[k % 4]],
              (X/2 - 0.022, 4.50 + (k % 5)*0.33 + rng.uniform(-.03, .03), 1.25 + (k // 5)*0.48 + rng.uniform(-.04, .04)), (0, 0, 0, 1))
    for k in range(4):
        m = S.textured(f"poster2_{k}", tex[f"poster2_{k}"], rough=.6)
        S.quad(f"/Room/poster_{k}", (-1.2 + k*1.1, Y - 0.005, 1.75), (-1, 0, 0), (0, 0, 1), (0.60, 0.84), m)
    S.quad("/Room/alphabet", (0, 0.004, 2.45), (1, 0, 0), (0, 0, 1), (3.6, 0.28), S.textured("alphabet2", tex["alphabet2"], .7))
    S.quad("/Room/clock", (1.75, 0.006, 2.45), (1, 0, 0), (0, 0, 1), (0.32, 0.32), S.textured("clock", tex["clock"], .4))
    S.box("/Room/projector", (0.35, 0.30, 0.12), mats["projector"], (0., 4.2, Z - 0.35), (0, 0, 0, 1))
    S.box("/Room/projector_pole", (0.04, 0.04, 0.30), mats["aluminum"], (0., 4.2, Z - 0.15), (0, 0, 0, 1))
    # lights: six 2 x 4 ft troffers (diffuser + rectangle light), sky dome and sun through the windows
    for i, y in enumerate((1.9, 4.8, 7.7)):
        for j, x in enumerate((-1.8, 1.8)):
            S.box(f"/Room/troffer_{i}_{j}", (0.61, 1.22, 0.04), mats["frame_white"], (x, y, Z - 0.02), (0, 0, 0, 1))
            S.box(f"/Room/diffuser_{i}_{j}", (0.57, 1.18, 0.005), S.color("diffuser", (.95, .95, .95), .3, 0., emissive=(4., 4., 3.8)),
                  (x, y, Z - 0.043), (0, 0, 0, 1))
            lt = UsdLux.RectLight.Define(S.stage, f"/Room/light_{i}_{j}")
            lt.CreateWidthAttr(0.57); lt.CreateHeightAttr(1.18); lt.CreateIntensityAttr(2000.); lt.CreateExposureAttr(0.)
            lt.CreateColorAttr(Gf.Vec3f(1., .97, .92))
            xf = S.UsdGeom.Xformable(lt.GetPrim()); xf.AddTranslateOp().Set(Gf.Vec3d(x, y, Z - 0.05))
    dome = UsdLux.DomeLight.Define(S.stage, "/Room/sky")
    dome.CreateTextureFileAttr(S.Sdf.AssetPath(str(Path(hdri(EXTERIOR_HDRI)).resolve())))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(500.)
    dome.AddRotateZOp().Set(90.)
    dressing(S, mats)
    sun = UsdLux.DistantLight.Define(S.stage, "/Room/sun")
    # 1100: at 1500 sunlit floor patches clip to white and hide glide detail
    sun.CreateIntensityAttr(1100.); sun.CreateAngleAttr(0.53); sun.CreateColorAttr(Gf.Vec3f(1., .95, .86))
    xf = S.UsdGeom.Xformable(sun.GetPrim())
    xf.AddOrientOp().Set(Gf.Quatf(*_rot_to_wxyz(Rotation.from_euler("xyz", (0., -55., -15.), degrees=True))))


def dressing(S, mats, rng_seed=11):
    """Render-only classroom dressing outside every task's working area (no physics): backpacks on the back-wall
    hooks, plants on the sills, a globe and books on the bookcase, a recycling bin in the front corner,
    jackets on the coat hooks, papers on the bulletin board."""
    X, Y = P.ROOM_X, P.ROOM_Y
    rng = np.random.default_rng(rng_seed)
    fabric = [S.color(f"fabric_{k}", c, .85) for k, c in enumerate(((.55, .10, .12), (.10, .22, .45), (.12, .12, .14),
                                                                      (.18, .40, .25), (.45, .30, .55), (.85, .45, .10)))]
    # coat/backpack hooks on the back wall between the cubbies and the bookcase
    S.box("/Dress/hook_rail", (2.6, 0.03, 0.08), mats["frame_white"], (-2.3, Y - 0.02, 1.62), (0, 0, 0, 1))
    for k in range(8):
        x = -3.45 + k*0.33
        S.tube(f"/Dress/hook_{k}", [(x, Y - 0.03, 1.62), (x, Y - 0.10, 1.60), (x, Y - 0.10, 1.65)], 0.006, mats["chrome"])
        if k % 3 != 2:      # a backpack: rounded body, front pocket, straps
            m = fabric[rng.integers(len(fabric))]
            z = 1.30 + rng.uniform(-.03, .03)
            S.box(f"/Dress/pack_{k}", (0.30, 0.16, 0.42), m, (x, Y - 0.12, z), (0, 0, 0, 1), bevel=0.05)
            S.box(f"/Dress/pack_pocket_{k}", (0.22, 0.05, 0.16), m, (x, Y - 0.215, z - 0.08), (0, 0, 0, 1), bevel=0.02)
            for j, sx in enumerate((-0.08, 0.08)):
                S.tube(f"/Dress/pack_strap_{k}_{j}", [(x + sx, Y - 0.05, z + 0.20), (x + sx, Y - 0.04, z - 0.15)], 0.012, mats["black"])
        else:               # a jacket
            S.box(f"/Dress/jacket_{k}", (0.40, 0.10, 0.65), fabric[rng.integers(len(fabric))], (x, Y - 0.10, 1.25), (0, 0, 0, 1), bevel=0.04)
    # plants on the window sills
    pot = S.color("pot", (.62, .32, .20), .8); leaf = S.color("leaf", (.16, .42, .14), .7)
    for k, y in enumerate((1.2, 4.3, 8.4)):
        x = -X/2 + 0.10
        S.cylinder(f"/Dress/pot_{k}", 0.07, 0.13, pot, (x, y, 0.985))
        for j in range(7):
            a = j*2*math.pi/7 + rng.uniform(-.2, .2)
            S.tube(f"/Dress/leaf_{k}_{j}", [(x, y, 1.04), (x + 0.12*math.cos(a), y + 0.12*math.sin(a), 1.20 + rng.uniform(-.03, .05))], 0.018, leaf)
    # globe and stacked books on the bookcase top, a flag by the board, recycling bin by the door
    S.cylinder("/Dress/globe_base", 0.08, 0.02, mats["black"], (2.75, 9.30, 1.81))
    S.tube("/Dress/globe_stand", [(2.75, 9.30, 1.82), (2.75, 9.30, 1.92)], 0.008, mats["chrome"])
    glob = S.UsdGeom.Sphere.Define(S.stage, "/Dress/globe"); glob.CreateRadiusAttr(0.13); S.place(glob, (2.75, 9.30, 2.05))
    S.bind(glob, S.color("ocean", (.15, .35, .65), .35))
    for k in range(5):
        S.box(f"/Dress/book_stack_{k}", (0.26, 0.20, 0.035), mats[f"book_{k % 4}"], (3.15, 9.30, 1.825 + k*0.036),
              Rotation.from_euler("z", rng.uniform(-.2, .2)).as_quat())
    S.cylinder("/Dress/recycle", 0.17, 0.55, S.color("recycle_blue", (.10, .30, .68), .55), (3.48, 0.24, 0.28))
    # papers pinned to the bulletin board
    for k in range(10):
        S.box(f"/Dress/paper_{k}", (0.003, 0.215, 0.279), mats["paper"],
              (X/2 - 0.024, 4.45 + (k % 5)*0.33 + rng.uniform(-.03, .03), 1.80 + (k // 5)*0.30 + rng.uniform(-.03, .03)),
              Rotation.from_euler("x", rng.uniform(-.08, .08)).as_quat())


def _rot_to_wxyz(R):
    x, y, z, w = R.as_quat()
    return float(w), float(x), float(y), float(z)


def dots(S, spec, mats):
    for n, pts in spec["dots"].items():
        for k, (x, y) in enumerate(pts):
            S.disc(f"/Room/dot_{n}_{k}", 0.0095, mats["dot_green" if n.startswith("desk") else "dot_orange"], (x, y, 0.0004))


def cameras(S, world):
    """Policy cameras (head, wrists) and diagnostic cameras; returns {name: role}."""
    sc = world.scene
    out = {}
    rig = world.rig
    inst = sc.robot_instance
    # head camera (ZED Mini class) on Franka's head bracket: the frame's +x is the optical axis, +z up
    fr = rig.frame_pose(f"{inst}/head_camera", world.state)
    carriage = sc.robot_body["franka_spine_mounting_point"]
    pb, Rb = world.body_pose(carriage)
    qw = fr["orientation_wxyz"]
    Rf = Rotation.from_quat((qw[1], qw[2], qw[3], qw[0]))
    pf = np.asarray(fr["position_m"])
    Rl = Rotation.from_matrix(Rb).inv()*Rf
    pl = Rotation.from_matrix(Rb).inv().apply(pf - pb)
    eye, target, up = pl, pl + Rl.apply((1, 0, 0)), Rl.apply((0, 0, 1))
    g = S.camera(f"/Lab/body_{carriage}/head_rgb", eye, target, 18/math.tan(math.radians(P.HEAD_HFOV/2)), up=up, role="policy")
    S.alias("head_rgb", g.GetPath()); out["head_rgb"] = "policy"
    for name, m in sc.cameras.items():
        g = S.camera(f"/Lab/body_{m['body']}/{name}", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)), up=m["up"],
                     role="policy")
        S.alias(name, g.GetPath()); out[name] = "policy"
    base = sc.robot_body["base_link"]
    g = S.camera(f"/Lab/body_{base}/chase", (-2.3, 0.6, 2.3), (0.9, 0., 0.55), 16., role="diagnostic")
    S.alias("chase", g.GetPath()); out["chase"] = "diagnostic"
    g = S.camera(f"/Lab/body_{base}/side", (0.5, 2.4, 1.6), (0.6, 0., 0.6), 18., role="diagnostic")
    S.alias("side", g.GetPath()); out["side"] = "diagnostic"
    diag = dict(room=((3.3, 9.3, 2.6), (-0.8, 3.2, 0.6), 17.), front=((-3.2, 0.35, 2.2), (0.6, 5.0, 0.5), 16.),
                plan=((0.0, 4.8, 3.15), (0.0, 4.81, 0.0), 6.))
    for name, (eye, target, f) in diag.items():
        up = (0, 1, 0) if name == "plan" else (0, 0, 1)
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f, up=up)
        S.alias(name, c.GetPath()); out[name] = "diagnostic"
    return out


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, folder, scene, *, width=640, height=360, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CLS_RENDER_PORT=str(listener.getsockname()[1]), CLS_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "classroom_arrangement.render_worker", "--scene", str(Path(scene).resolve()),
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
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
        if not result.get("ok") or result.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(result)[:400])
        return result

    def observe(self, cameras, body_q, tick, resolution=None, samples=None):
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras))
        if resolution:
            msg["resolution"] = list(resolution)
        if samples:
            msg["spp"] = int(samples)
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

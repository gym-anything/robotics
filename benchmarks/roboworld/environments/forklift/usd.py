"""Thin USD authoring helper for the forklift exporter."""

import math
from pathlib import Path

import numpy as np


import re


def clean(path):
    """A valid USD prim path: each component [A-Za-z_][A-Za-z0-9_]* ('-' -> 'm', other characters -> '_')."""
    parts = [re.sub(r"[^A-Za-z0-9_]", "_", c.replace("-", "m")) for c in str(path).split("/")]
    parts = [("_" + c) if c and c[0].isdigit() else c for c in parts]
    return "/".join(parts)


class Stage:
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

    # ---------------------------------------------------------------- materials
    def color(self, name, rgb, rough=.5, metal=0., emissive=None, opacity=None):
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
        if opacity is not None:
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        self.mats[name] = m
        return m

    def mdl(self, name, module, inputs):
        if name in self.mats:
            return self.mats[name]
        UsdShade = self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, f"/Looks/{name}/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(self.Sdf.AssetPath(module + ".mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier(module, "mdl")
        for k, (typ, v) in inputs.items():
            sh.CreateInput(k, typ).Set(v)
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        self.mats[name] = m
        return m

    def textured(self, name, diffuse, rough=None, normal=None, rough_const=.8, tint=(1., 1., 1.), metal=0.,
                 opacity_cutout=None):
        """OmniPBR with texture maps (UV in metres scaled by the mesh's st primvar)."""
        T = self.Sdf.ValueTypeNames
        inp = {"diffuse_texture": (T.Asset, self.Sdf.AssetPath(str(Path(diffuse).resolve()))),
               "diffuse_tint": (T.Color3f, self.Gf.Vec3f(*tint)), "metallic_constant": (T.Float, float(metal))}
        if rough:
            inp["reflectionroughness_texture"] = (T.Asset, self.Sdf.AssetPath(str(Path(rough).resolve())))
            inp["reflection_roughness_texture_influence"] = (T.Float, 1.)
        else:
            inp["reflection_roughness_constant"] = (T.Float, float(rough_const))
        if normal:
            inp["normalmap_texture"] = (T.Asset, self.Sdf.AssetPath(str(Path(normal).resolve())))
        if opacity_cutout:
            inp["enable_opacity"] = (T.Bool, True)
            inp["enable_opacity_texture"] = (T.Bool, True)
            inp["opacity_texture"] = (T.Asset, self.Sdf.AssetPath(str(Path(opacity_cutout).resolve())))
            inp["opacity_mode"] = (T.Int, 1)              # mono_average of the (greyscale) opacity texture
            inp["opacity_threshold"] = (T.Float, 0.)        # fractional: distant cords average to a haze
        return self.mdl(name, "OmniPBR", inp)

    def emissive(self, name, rgb, intensity):
        T = self.Sdf.ValueTypeNames
        return self.mdl(name, "OmniPBR", {"diffuse_color_constant": (T.Color3f, self.Gf.Vec3f(.9, .9, .9)),
                                          "enable_emission": (T.Bool, True),
                                          "emissive_color": (T.Color3f, self.Gf.Vec3f(*rgb)),
                                          "emissive_intensity": (T.Float, float(intensity))})

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    # ---------------------------------------------------------------- geometry
    def xform(self, path, p=None, q=None, scale=None):
        x = self.UsdGeom.Xform.Define(self.stage, clean(path))
        if p is not None:
            x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        if scale is not None:
            x.AddScaleOp().Set(self.Gf.Vec3f(*map(float, scale)))
        return x

    def place(self, prim, p=None, q=None):
        xf = self.UsdGeom.Xformable(prim)
        if p is not None:
            xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
        if q is not None:
            xf.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return prim

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, uv_face_varying=False):
        UsdGeom, Vt, Sdf = self.UsdGeom, self.Vt, self.Sdf
        v = np.asarray(v, np.float32); f = np.asarray(f, np.int32).reshape(-1, 3)
        g = UsdGeom.Mesh.Define(self.stage, clean(path))
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
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 "faceVarying" if uv_face_varying else "vertex").Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        if mat is not None:
            self.bind(g, mat)
        return g

    def box(self, path, size, mat, p=None, q=None, uv_per_m=None):
        """Axis-aligned box (full size), optional planar UVs at uv_per_m cycles per metre."""
        sx, sy, sz = (float(x)/2 for x in size)
        corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        faces_q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        normals_f = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        v, n, uv, f = [], [], [], []
        for qd, nn in zip(faces_q, normals_f):
            k = len(v)
            ax = int(np.argmax(np.abs(nn)))
            a, b = [i for i in range(3) if i != ax]
            for i in qd:
                v.append(corners[i]); n.append(nn)
                uv.append((corners[i][a]*(uv_per_m or 1.), corners[i][b]*(uv_per_m or 1.)))
            f += [k, k + 1, k + 2, k, k + 2, k + 3]
        g = self.mesh(path, v, f, uv=uv if uv_per_m else None, normals=n, mat=mat)
        if p is not None or q is not None:
            self.place(g, p, q if q is not None else (0., 0., 0., 1.))
        return g

    def cylinder(self, path, r, h, mat, n=32, caps=True, p=None, q=None, axis="z"):
        ks = np.linspace(0, 2*math.pi, n + 1)
        v, nr, f = [], [], []
        for a in ks:
            d = np.array([math.cos(a), math.sin(a), 0.])
            for z in (-h/2, h/2):
                v.append(d*r + np.array([0, 0, z])); nr.append(d)
        for i in range(n):
            a, b, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
            f += [a, c, d, a, d, b]
        if caps:
            for z, s in ((-h/2, -1), (h/2, 1)):
                c0 = len(v); v.append((0, 0, z)); nr.append((0, 0, s))
                for a in ks[:-1]:
                    v.append((r*math.cos(a), r*math.sin(a), z)); nr.append((0, 0, s))
                for i in range(n):
                    a_, b_ = c0 + 1 + i, c0 + 1 + (i + 1) % n
                    f += [c0, a_, b_] if s > 0 else [c0, b_, a_]
        v = np.asarray(v, float); nr = np.asarray(nr, float)
        if axis == "x":
            v = v[:, [2, 0, 1]]; nr = nr[:, [2, 0, 1]]
        elif axis == "y":
            v = v[:, [1, 2, 0]]; nr = nr[:, [1, 2, 0]]
        g = self.mesh(path, v, f, normals=nr, mat=mat)
        if p is not None:
            self.place(g, p, q if q is not None else (0., 0., 0., 1.))
        return g

    def tube(self, path, pts, radius, mat, n=12):
        pts = np.asarray(pts, float)
        v, nr, f = [], [], []
        for i, p in enumerate(pts):
            t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]; t /= np.linalg.norm(t)
            a = np.array([0., 0., 1.]) if abs(t[2]) < .9 else np.array([1., 0., 0.])
            u = np.cross(t, a); u /= np.linalg.norm(u); w_ = np.cross(t, u)
            for k in range(n):
                ang = 2*math.pi*k/n
                d = math.cos(ang)*u + math.sin(ang)*w_
                v.append(p + radius*d); nr.append(d)
        for i in range(len(pts) - 1):
            for k in range(n):
                a, b = i*n + k, i*n + (k + 1) % n
                c, d = a + n, b + n
                f += [a, c, d, a, d, b]
        return self.mesh(path, v, f, normals=nr, mat=mat)

    def revolve(self, path, profile, mat, n=48, axis="z", uv=False):
        """Closed surface of revolution through the (radius, height) profile; axis z (or x)."""
        prof = np.asarray(profile, float)
        v, nr, f, st = [], [], [], []
        m = len(prof)
        L = np.r_[0., np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
        for j, a in enumerate(np.linspace(0, 2*math.pi, n, endpoint=False)):
            d = np.array([math.cos(a), math.sin(a), 0.])
            for i, (rad, z) in enumerate(prof):
                t = prof[min(i + 1, m - 1)] - prof[max(i - 1, 0)]
                nrm = d*t[1] - np.array([0., 0., t[0]])
                nrm = nrm/np.linalg.norm(nrm) if np.linalg.norm(nrm) > 1e-12 else np.array([0., 0., np.sign(z) or 1.])
                v.append(d*rad + np.array([0., 0., z])); nr.append(nrm); st.append((j/n, L[i]/max(L[-1], 1e-9)))
        for k in range(n):
            k2 = (k + 1) % n
            for i in range(m - 1):
                a, b, c, d = k*m + i, k*m + i + 1, k2*m + i, k2*m + i + 1
                f += [a, c, d, a, d, b]
        v = np.asarray(v); nr = np.asarray(nr)
        if axis == "x":
            v = v[:, [2, 0, 1]]; nr = nr[:, [2, 0, 1]]
        return self.mesh(path, v, f, normals=nr, mat=mat, uv=st if uv else None)

    def sphere(self, path, r, mat, n=24, uv=True):
        prof = [(r*math.sin(t), -r*math.cos(t)) for t in np.linspace(0, math.pi, n)]
        return self.revolve(path, prof, mat, n=2*n, uv=uv)

    def quad(self, path, centre, u, v, size, mat, uv_scale=(1., 1.)):
        c, u, v = (np.asarray(x, float) for x in (centre, u, v))
        a, b = size[0]/2, size[1]/2
        pts = [c - a*u - b*v, c + a*u - b*v, c + a*u + b*v, c - a*u + b*v]
        n = np.cross(u, v)
        uv = [(0, 0), (uv_scale[0], 0), (uv_scale[0], uv_scale[1]), (0, uv_scale[1])]
        return self.mesh(path, pts, [0, 1, 2, 0, 2, 3], uv=uv, normals=[n]*4, mat=mat)

    def grid(self, path, centre, u, v, size, cells, mat, uv_per_m=1.):
        """A flat grid mesh (cells = (nu, nv)) whose points the render worker can displace (net dimples)."""
        c, u, v = (np.asarray(x, float) for x in (centre, u, v))
        nu, nv = cells
        A = np.linspace(-size[0]/2, size[0]/2, nu + 1); B = np.linspace(-size[1]/2, size[1]/2, nv + 1)
        pts, uv = [], []
        for b in B:
            for a in A:
                pts.append(c + a*u + b*v); uv.append(((a + size[0]/2)*uv_per_m, (b + size[1]/2)*uv_per_m))
        f = []
        for j in range(nv):
            for i in range(nu):
                p0 = j*(nu + 1) + i
                f += [p0, p0 + 1, p0 + nu + 2, p0, p0 + nu + 2, p0 + nu + 1]
        n = np.cross(u, v)
        g = self.mesh(path, pts, f, uv=uv, normals=[n]*len(pts), mat=mat)
        g.GetPrim().SetCustomDataByKey("grid", dict(nu=nu, nv=nv))
        return g, np.asarray(pts), (A, B)

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), role="diagnostic", aperture=36., aspect=16/9):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*target),
                                                            self.Gf.Vec3d(*up)).GetInverse())
        c.CreateHorizontalApertureAttr(float(aperture)); c.CreateVerticalApertureAttr(float(aperture/aspect))
        c.CreateFocalLengthAttr(float(focal_mm)); c.CreateClippingRangeAttr(self.Gf.Vec2f(.01, 80.))
        c.GetPrim().SetCustomDataByKey("observation_role", role)
        return c

    def alias(self, name, mounted):
        a = self.UsdGeom.Xform.Define(self.stage, "/Cameras/" + name)
        a.GetPrim().SetCustomDataByKey("mounted_camera_path", str(mounted))

    def save(self):
        self.stage.GetRootLayer().Save()

"""Render-only view of the sewing cell (Isaac Sim 5.1 RTX in a separate worker; sewing/render_worker.py).

The backend process records a FRAME of physical state -- every robot body's pose, the machine's moving parts from the
shaft angle (needle bar, feed dog, take-up lever, handwheel) and the presser-foot body's height, every ply's vertex
positions, and the seam's threads at the holes' current material points -- and the worker draws it. Rendering never
steps physics; nothing in an image is animated or authored to look like sewing.

Visual model:
  plies    each ply's simulated surface (vertex normals per frame; the stack's outer faces: the top ply at its top face,
           the bottom ply at its bottom face), double-sided, unbleached muslin: a plain-weave
           close-up texture (CC0 'rough_linen') tinted ecru, tiled in the ply's material coordinates;
  threads  the needle thread on the top ply's surface and the bobbin thread under the bottom ply, between consecutive
           formed stitches (the holes' material points on the deformed plies), each stitch a dash that goes into the
           cloth at its needle holes, 0.40 mm wide (Tex 30 spun polyester); before trimming, the needle thread from the last stitch up to the needle's eye and its fixed
           path over the take-up lever, the tension unit and the stand to the cone (path authored);
  machine  the authored head (sewing/assets/build_machine.py) with its moving parts posed from the simulation.
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
import secrets
import select
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import layout as LY
from . import params as P

ROOT = Path(__file__).resolve().parent
MACHINE = ROOT/"assets"/"machine"
THREAD_W = P.THREAD_DIAM              # Tex 30 spun polyester (params: thread.diameter)
MUSLIN_SRGB = (238, 226, 198)        # unbleached cotton muslin, ecru [estimate from photographs]
THREAD_SRGB = (32, 42, 78)           # navy spun polyester (the cones in machine_static.usd)
TAKE_UP_PIVOT = (-0.047, -0.0545, 0.243)
CONE_TOP = (0.255, 0.230, 0.158)
EXPOSURE = {"default": dict(exposureTime=1/125., fNumber=2.8, filmIso=400.)}
DOME_NITS = 400.




def weave_texture():
    """A neutral (grey) plain-weave map from the CC0 'rough_linen' close-up: its luminance, normalised to a mean of
    0.85, so the material's tint gives the cloth's colour (the source map is a blue linen, which the tint multiplied to
    grey)."""
    from PIL import Image
    from . import visual_assets as VA
    out = ROOT/".cache"/"textures"/"weave_neutral.png"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    im = np.asarray(Image.open(VA.texture("rough_linen")["diff"]).convert("RGB"), float)/255.
    lum = im @ np.array([0.2126, 0.7152, 0.0722])
    lum = np.clip(lum/lum.mean()*0.85, 0, 1)
    Image.fromarray((lum*255).astype(np.uint8)).convert("RGB").save(out)
    return out


MUSLIN_TILE_M = 0.20          # m of cloth one muslin texture tile covers


def muslin_textures():
    """Unbleached muslin at true scale, as maps of one MUSLIN_TILE_M tile (2048 px, 0.1 mm a pixel): the CC0
    'rough_linen' weave (a 270.7 mm photograph whose threads measure 1.09 x 1.47 mm by FFT) scaled to muslin's
    ~0.4 mm thread pitch (60-70 threads per inch [read]; 2 x 2 linen tiles across the muslin tile, each at the source's
    full resolution), with what makes unbleached cotton read as cloth rather than card [authored, from photographs]:
    slub yarns (thick places along the warp and the weft, 5-40 mm long, +-6 % brightness), seed-coat flecks (0.2-0.9 mm,
    brown, ~0.8 per cm2) and a +-2 % mottle. Returns (diffuse, normal); the diffuse is near neutral (mean 0.85) and the
    material's tint gives the colour. Deterministic (seeded) and cached."""
    from PIL import Image, ImageDraw, ImageFilter
    from . import visual_assets as VA
    d = ROOT/".cache"/"textures"
    diff, nrm = d/"muslin_diff_v3.png", d/"muslin_nor_gl_v3.png"
    if diff.is_file() and nrm.is_file():
        return diff, nrm
    d.mkdir(parents=True, exist_ok=True)
    tex = VA.texture("rough_linen")
    N = 2048
    lin = np.asarray(Image.open(tex["diff"]).convert("RGB"), float)[:1024, :1024]/255.
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    lum = lum/lum.mean()
    base = np.tile(lum, (2, 2))
    rng = np.random.default_rng(20261009)
    # mottle: low-frequency variation of the yarns' whiteness
    mot = np.asarray(Image.fromarray((rng.random((64, 64))*255).astype(np.uint8)).resize((N, N), Image.BICUBIC),
                     float)/255.
    base *= 1.0 + 0.04*(mot - 0.5)
    # slubs: thick places in single yarns, along v (rows of the image run along the warp) and along u
    sl = Image.new("F", (N, N), 0.0)
    dr = ImageDraw.Draw(sl)
    px = N/(MUSLIN_TILE_M*1e3)                     # pixels per mm
    for _ in range(260):
        x, y = rng.random(2)*N
        length = rng.uniform(5, 40)*px
        w = rng.uniform(0.2, 0.5)*px
        amp = rng.choice((-1.0, 1.0))*rng.uniform(0.02, 0.06)
        if rng.random() < 0.6:
            dr.line([(x, y), (x, y + length)], fill=float(amp), width=max(1, int(round(w))))
        else:
            dr.line([(x, y), (x + length, y)], fill=float(amp), width=max(1, int(round(w))))
    from scipy.ndimage import gaussian_filter
    sl = gaussian_filter(np.asarray(sl, float), 1.2)
    base *= 1.0 + sl
    rgb = np.repeat((np.clip(base*0.85, 0, 1))[..., None], 3, axis=2)
    # seed-coat flecks: brown, small, soft-edged (relative to the ecru tint the material multiplies by)
    fl = Image.new("L", (N, N), 0)
    df = ImageDraw.Draw(fl)
    n = int(0.8*(MUSLIN_TILE_M*100)**2)
    for _ in range(n):
        x, y = rng.random(2)*N
        ln = rng.uniform(0.15, 0.9)*px/2
        ang = rng.uniform(0, np.pi)
        dx, dy = ln*np.cos(ang), ln*np.sin(ang)
        df.line([(x - dx, y - dy), (x + dx, y + dy)], fill=int(rng.uniform(70, 200)),
                width=max(1, int(round(rng.uniform(0.1, 0.3)*px))))
    a = np.asarray(fl.filter(ImageFilter.GaussianBlur(0.8)), float)[..., None]/255.
    fleck = np.array([0.42, 0.33, 0.24])
    rgb = rgb*(1 - a) + fleck*a
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(diff)
    nor = np.asarray(Image.open(tex["nor_gl"]).convert("RGB"))[:1024, :1024]
    Image.fromarray(np.tile(nor, (2, 2, 1))).save(nrm)
    return diff, nrm

def srgb_to_linear(c):
    c = np.asarray(c, float)/255.
    return np.where(c <= .04045, c/12.92, ((c + .055)/1.055)**2.4)


# ------------------------------------------------------------------------------------------------ robot visuals
def _origin(el):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


@lru_cache(maxsize=4)
def _urdf(model):
    from robots.assets import resolve
    p = Path(resolve(model))
    if model == "fr3":
        p = p.with_name("fr3_franka_hand.urdf")
    return p, ET.parse(p).getroot()


def _mesh_path(urdf_path, fn):
    if fn.startswith("package://"):
        pkg, rel = fn[len("package://"):].split("/", 1)
        for base in (urdf_path.parent.parent, urdf_path.parent.parent.parent):
            if (base/rel).exists():
                return base/rel
            if (base/pkg/rel).exists():
                return base/pkg/rel
        return urdf_path.parent.parent/rel
    return (urdf_path.parent/fn).resolve()


def _load_mesh(path, T):
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    parts = []
    for node in sc.graph.nodes_geometry:
        M, gname = sc.graph[node]
        g = sc.geometry[gname]
        V = np.asarray(g.vertices, float) @ M[:3, :3].T + M[:3, 3]
        V = V @ T[:3, :3].T + T[:3, 3]
        col = (200, 200, 200)
        mat = getattr(g.visual, "material", None)
        for attr in ("baseColorFactor", "main_color", "diffuse"):
            cc = getattr(mat, attr, None) if mat is not None else None
            if cc is not None:
                col = tuple(int(v) for v in np.asarray(cc)[:3])
                break
        parts.append((V.astype(np.float32), np.asarray(g.faces, np.int32), col))
    return parts


@lru_cache(maxsize=64)
def merged_link_visuals(model, link):
    """Visual parts [(V, F, srgb)] of a URDF link and every link fixed to it (Newton collapses fixed joints), in the
    link's frame."""
    path, root = _urdf(model)
    links = {l.get("name"): l for l in root.findall("link")}
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    out = []

    def walk(name, T):
        el = links.get(name)
        if el is None:
            return
        for vis in el.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            p = _mesh_path(path, m.get("filename"))
            if p.exists():
                out.extend(_load_mesh(p, T @ _origin(vis.find("origin"))))
        for j in children.get(name, []):
            if j.get("type") == "fixed":
                walk(j.find("child").get("link"), T @ _origin(j.find("origin")))
    walk(link, np.eye(4))
    return out


ROBOT_ROOTS = {"fr3": "fr3_link0", "panda": "panda_link0"}
# body-label model names -> catalog asset ids (the G1's links are all posed bodies: its pelvis floats)
ROBOT_ASSETS = {"g1_29dof_with_hand_rev_1_0": "g1"}


# ------------------------------------------------------------------------------------------------ frames
def _vertex_normals(x, tri):
    n = np.zeros_like(x)
    fn = np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]])
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-20)[:, None]
    return n


def _hole_point(cl, x, h, side):
    """A hole's material point on the deformed ply, moved to the ply's top (+1) or bottom (-1) surface."""
    T = cl.tri_host[h.face]
    p = np.dot(h.bary, x[T])
    n = np.cross(x[T[1]] - x[T[0]], x[T[2]] - x[T[0]])
    n /= max(np.linalg.norm(n), 1e-20)
    if n[2] < 0:
        n = -n
    return p + side*n*(0.5*cl.plies[h.ply]["thickness"] + 0.6*THREAD_W)


def _hole_frame(cl, x, h):
    """A hole's material point on the deformed ply and the ply's upward unit normal there."""
    T = cl.tri_host[h.face]
    p = np.dot(h.bary, x[T])
    n = np.cross(x[T[1]] - x[T[0]], x[T[2]] - x[T[0]])
    n /= max(np.linalg.norm(n), 1e-20)
    return p, (n if n[2] >= 0 else -n)


def _stitched(cl, x, holes, side):
    """One side's thread as a lockstitch shows it: on the surface between consecutive needle holes and, at each hole,
    down into the cloth (the thread passes through the hole to the interlock inside the stack), so that each stitch is a
    separate dash, as a seam looks. A polyline drawn through the holes on the surface reads as a drawn line."""
    pts = []
    for k, h in enumerate(holes):
        p, n = _hole_frame(cl, x, h)
        th = cl.plies[h.ply]["thickness"]
        surf = p + side*n*(0.5*th + 0.6*THREAD_W)
        sunk = p - side*n*(0.5*th + 0.8*THREAD_W)          # under the ply's surface: hidden by the cloth
        if k:
            q, _ = _hole_frame(cl, x, holes[k - 1])
            d = p - q
            L = float(np.linalg.norm(d))
            u = d/max(L, 1e-9)
            r = min(0.35e-3, 0.3*L)                           # the hole's half width and the thread's ramp into it
            pts.append(surf - u*r)
        pts.append(sunk)
        if k + 1 < len(holes):
            q, _ = _hole_frame(cl, x, holes[k + 1])
            d = q - p
            L = float(np.linalg.norm(d))
            u = d/max(L, 1e-9)
            pts.append(surf + u*min(0.35e-3, 0.3*L))
    return np.array(pts)


def thread_paths(world):
    """Polylines of the seam's threads: [needle side, bobbin side, needle thread to the eye and up to the cone]; each
    stitch a dash on its side's surface, the thread going into the cloth at every needle hole."""
    M = world.machine
    cl = world.cloth
    x = cl.positions()
    paths = []
    top_h, bot_h = [], []
    seam = None
    for il in M.interlocks:
        holes = [M.holes[i] for i in il["holes"]]
        if not holes:
            continue
        if seam is not None and il.get("seam") != seam and top_h:
            if len(top_h) >= 2:
                paths += [_stitched(cl, x, top_h, +1), _stitched(cl, x, bot_h, -1)]
            top_h, bot_h = [], []
        seam = il.get("seam")
        top_h.append(max(holes, key=lambda h: h.z0))
        bot_h.append(min(holes, key=lambda h: h.z0))
    if len(top_h) >= 2:
        paths += [_stitched(cl, x, top_h, +1), _stitched(cl, x, bot_h, -1)]
    # needle thread: from the last stitch (if not trimmed) to the eye, over the take-up lever, the tension discs, the
    # head and stand guides, to the cone (route authored; the eye and the lever follow the shaft)
    tip = float(M.needle_tip(M.theta))
    eye = np.array([0.0, 0.0, tip + 0.0025])
    lever = take_up_eye(M.theta)
    route = [eye, np.array([0.0, 0.0, tip + 0.050]), np.array([-0.012, -0.050, 0.120]), np.array([-0.040, -0.058, 0.225]),
             lever, np.array([-0.020, -0.0565, 0.205]), np.array([-0.045, -0.060, 0.262]), np.array([0.005, -0.060, 0.262]),
             np.array([0.255, 0.23, 0.56]), np.array(CONE_TOP)]
    if M.interlocks and M.tether != "trimmed":
        il = M.interlocks[-1]
        top = max((M.holes[i] for i in il["holes"]), key=lambda h: h.z0)
        route = [_hole_point(cl, x, top, +1)] + route
    else:
        route = [eye + np.array([0.0, -0.002, -0.003])] + route
    paths.append(np.array(route))
    return paths


def take_up_eye(theta):
    """The take-up lever's eye: the lever swings 30 deg about its pivot with the shaft (visual kinematics, phase
    authored: highest just after the needle leaves the cloth)."""
    a = math.radians(15.0)*math.cos(theta - 5.2)
    piv = np.array(TAKE_UP_PIVOT)
    return piv + 0.034*np.array([-math.sin(a), 0.0, math.cos(a)])


def take_up_angle(theta):
    return 15.0*math.cos(theta - 5.2)


def world_frame(world, label="UNASSISTED"):
    """A recorded frame of the world's physical state for the renderer."""
    w = world
    q = w.state.body_q.numpy().astype(float)
    labels = list(w.model.body_label)
    frame = dict(tick=int(w.tick), time_s=float(w.time), label=label, bodies=labels, body_q=q.tolist())
    M = w.machine
    if M is not None:
        zd, yd = M.dog(M.theta)
        frame["machine"] = dict(theta=float(M.theta), needle_tip=float(M.needle_tip(M.theta)), foot_z=float(M.foot_z),
                                dog=(float(yd), float(zd)), take_up_deg=take_up_angle(M.theta))
        cl = w.cloth
        x = cl.positions()
        plies = []
        n_plies = len(cl.plies)
        for k, p_ in enumerate(cl.plies):
            v0, nv = p_["v0"], p_["nv"]
            xs = x[v0:v0 + nv]
            tri = cl.tri_host[p_["t0"]:p_["t0"] + p_["nt"]] - v0
            nrm = _vertex_normals(xs, tri)
            # the stack's outer faces are drawn, not the plies' mid-surfaces: the top ply at its top face, the bottom
            # ply at its bottom face (the material normal turned to face up where the ply lies on the table). Drawn at
            # mid-surfaces 0.3 mm apart, a ply dipping into the other within the contact's compliance showed through
            # it as dark slits in the cloth, and the thread laid on the top face floated over the drawn cloth
            side = 1.0 if k == n_plies - 1 else (-1.0 if k == 0 else 0.0)
            if side and n_plies > 1:
                flat = np.abs(nrm[:, 2]) > 0.9
                up = 1.0 if not flat.any() or nrm[flat, 2].mean() >= 0 else -1.0
                xs = xs + (side*up*0.5*p_["thickness"])*nrm
            plies.append(dict(points=xs.astype(np.float32), normals=nrm.astype(np.float32)))
        frame["plies"] = plies
        frame["threads"] = [p.astype(np.float32) for p in thread_paths(w)]
    return frame


# ------------------------------------------------------------------------------------------------ USD export
def _camera(stage, path, eye, target, hfov, role, up=(0., 0., 1.)):
    from pxr import Gf, UsdGeom
    c = UsdGeom.Camera.Define(stage, path)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up)))
    c.AddTransformOp().Set(m.GetInverse())
    c.CreateHorizontalApertureAttr(36.)
    c.CreateVerticalApertureAttr(20.25)
    c.CreateFocalLengthAttr(18./math.tan(math.radians(hfov)/2))
    c.CreateClippingRangeAttr(Gf.Vec2f(.004, 60.))
    c.GetPrim().SetCustomDataByKey("observation_role", role)
    c.GetPrim().SetCustomDataByKey("horizontal_fov_deg", float(hfov))
    return c


class _Author:
    """Small USD authoring helper (materials cached by name); as welding/render.py."""

    def __init__(self, stage, folder):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        self.stage, self.folder = stage, Path(folder)
        self.Gf, self.Sdf, self.UsdGeom, self.UsdShade, self.Vt = Gf, Sdf, UsdGeom, UsdShade, Vt
        self.mats = {}
        UsdGeom.Scope.Define(stage, "/Looks")

    def pbr(self, name, srgb, rough=.5, metal=0., diff=None, rough_tex=None, normal=None, tile_m=1., tint=None,
            emissive=None, emissive_nits=0.):
        if name in self.mats:
            return self.mats[name]
        Sdf, Gf, UsdShade = self.Sdf, self.Gf, self.UsdShade
        m = UsdShade.Material.Define(self.stage, "/Looks/" + name)
        sh = UsdShade.Shader.Define(self.stage, str(m.GetPath()) + "/Shader")
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        inp = lambda k, t, v: sh.CreateInput(k, t).Set(v)
        inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, srgb_to_linear(srgb))))
        inp("reflection_roughness_constant", Sdf.ValueTypeNames.Float, float(rough))
        inp("metallic_constant", Sdf.ValueTypeNames.Float, float(metal))
        inp("texture_scale", Sdf.ValueTypeNames.Float2, Gf.Vec2f(1./tile_m, 1./tile_m))
        if diff is not None:
            inp("diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(diff).resolve())))
        if tint is not None:
            inp("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, srgb_to_linear(tint))))
        if rough_tex is not None:
            inp("reflectionroughness_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(rough_tex).resolve())))
            inp("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 1.)
        if normal is not None:
            inp("normalmap_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(Path(normal).resolve())))
        if emissive is not None:
            inp("enable_emission", Sdf.ValueTypeNames.Bool, True)
            inp("emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, emissive)))
            inp("emissive_intensity", Sdf.ValueTypeNames.Float, float(emissive_nits))
        self.mats[name] = m
        return m

    def bind(self, prim, mat):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(mat)

    def xform(self, path, pos=(0, 0, 0), quat_xyzw=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        q = (0., 0., 0., 1.) if quat_xyzw is None else quat_xyzw
        x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return x

    def box(self, path, lo, hi, mat, uv=False, uv_world=True):
        from pxr import UsdGeom
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        sgn = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        V = lo + sgn*(hi - lo)
        F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
        P, idx, N, ST = [], [], [], []
        for f in F:
            a, b, cc = V[f[0]], V[f[1]], V[f[2]]
            n = np.cross(b - a, cc - a)
            n /= np.linalg.norm(n)
            for k in f:
                idx.append(len(P))
                P.append(V[k])
                N.append(n)
                ax = np.argmax(np.abs(n))
                u, v = [d for d in range(3) if d != ax]
                ST.append((V[k][u], V[k][v]))
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
        m.CreateFaceVertexCountsAttr([4]*6)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateNormalsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(N, np.float32)))
        m.SetNormalsInterpolation("faceVarying")
        m.CreateSubdivisionSchemeAttr("none")
        if uv:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", self.Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.faceVarying)
            pv.Set(self.Vt.Vec2fArray.FromNumpy(np.asarray(ST, np.float32)[idx]))
        self.bind(m, mat)
        return m

    def cyl(self, path, p0, p1, r, mat, sides=24):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        z = d/np.linalg.norm(d)
        x = np.cross(z, [1, 0, 0] if abs(z[0]) < .9 else [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        a = np.linspace(0, 2*math.pi, sides, endpoint=False)
        ring = np.cos(a)[:, None]*x + np.sin(a)[:, None]*y
        V = np.concatenate([p0 + r*ring, p1 + r*ring, [p0, p1]])
        F = []
        for k in range(sides):
            k1 = (k + 1) % sides
            F.append([k, k1, sides + k1, sides + k])
            F.append([2*sides, k1, k])
            F.append([2*sides + 1, sides + k, sides + k1])
        return self.mesh(path, V, F, mat, smooth=True)

    def mesh(self, path, V, F, mat, smooth=False):
        from pxr import UsdGeom
        m = UsdGeom.Mesh.Define(self.stage, path)
        V = np.asarray(V, np.float32)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(V))
        m.CreateFaceVertexCountsAttr([len(f) for f in F])
        m.CreateFaceVertexIndicesAttr([int(i) for f in F for i in f])
        m.CreateSubdivisionSchemeAttr("none")
        if smooth:
            N = np.zeros_like(V)
            for f in F:
                n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
                for i in f:
                    N[i] += n
            N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
            m.CreateNormalsAttr(self.Vt.Vec3fArray.FromNumpy(N))
            m.SetNormalsInterpolation("vertex")
        self.bind(m, mat)
        return m

    def tri_mesh(self, path, V, F, mat):
        from pxr import UsdGeom
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(self.Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
        m.CreateFaceVertexCountsAttr(self.Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(self.Vt.IntArray.FromNumpy(np.asarray(F, np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr("none")
        self.bind(m, mat)
        return m

    def usd(self, path, usd_file, pos=(0, 0, 0), quat_xyzw=None):
        x = self.xform(path, pos, quat_xyzw)
        inner = self.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(Path(usd_file).resolve()))
        return x

    def reference(self, path, usd, pos, yaw_deg=0., scale=1., ground=True):
        """Reference a pinned Poly Haven model with its materials rebound to OmniPBR (label_application recipe)."""
        from pxr import Usd
        from label_application.dressing import Dresser as LAD
        x = self.UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        x.AddRotateZOp().Set(float(yaw_deg))
        if scale != 1.:
            x.AddScaleOp().Set(self.Gf.Vec3f(scale, scale, scale))
        inner = self.UsdGeom.Xform.Define(self.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(Path(usd).resolve()))
        LAD._rebind_omnipbr(self, path, Path(usd).stem.replace("_1k", ""), Path(usd))
        if ground:
            bb = self.UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
                inner.GetPrim()).ComputeAlignedRange()
            if not bb.IsEmpty():
                ops = x.GetOrderedXformOps()
                t = ops[0].Get()
                ops[0].Set(self.Gf.Vec3d(t[0], t[1], t[2] + (pos[2] - bb.GetMin()[2])))
        return x


def _curve(stage, path, P, width, mat):
    from pxr import UsdGeom, Vt, UsdShade
    c = UsdGeom.BasisCurves.Define(stage, path)
    c.CreateTypeAttr("linear")
    c.CreateWrapAttr("nonperiodic")
    c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
    c.CreateCurveVertexCountsAttr([len(P)])
    c.CreateWidthsAttr([float(width)])
    c.SetWidthsInterpolation("constant")
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def scene_meta(world):
    """What export_scene needs from the world (static facts)."""
    sc = world.scene
    arms = {}
    for arm, hw in sc.arm_hw.items():
        if arm not in LY.ROBOT_BASES:          # the G1 floats: its pelvis is a posed body, no fixed base to draw
            continue
        base = LY.ROBOT_BASES[arm]
        arms[arm] = dict(model=hw["model"], mount_pos=list(base["pos"]), mount_yaw_deg=base["yaw_deg"])
    labels = list(world.model.body_label)
    mounted = {name: dict(c, body=labels[c["body"]]) for name, c in getattr(sc, "cameras", {}).items()}
    cl = world.cloth
    plies = []
    if cl is not None:
        for p_ in cl.plies:
            v0 = p_["v0"]
            plies.append(dict(tris=(cl.tri_host[p_["t0"]:p_["t0"] + p_["nt"]] - v0).tolist(),
                              uv=cl.X[v0:v0 + p_["nv"]].tolist()))
    # the physics builds the arms' pedestal only for the arm rigs (scene.py: the humanoid stands there instead); the
    # render draws what the physics has (no steel column at the G1's waist and no pedal under its foot)
    return dict(arms=arms, plies=plies, mounted_cameras=mounted, pedestal=not getattr(sc, "g1", False),
                humanoid=bool(getattr(sc, "g1", False)))


def export_scene(folder, frame, world_meta, extra_cameras=None):
    """Static render stage posed at `frame`: room, table, machine, robots, plies, threads and cameras. Writes
    folder/scene.usdc and export.json."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, Sdf, Vt
    from . import visual_assets as VA
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    A = _Author(stage, folder)
    zf = -LY.TABLE_H
    # ---- the sample room (walls, windows, door, the other workstations, cutting table, shelving, dress form,
    # ironing station, signage): sewing/room.py, render-only
    from . import room as RM
    RM.build(A, zf, VA, weave_texture, MACHINE, folder, P)
    # ---- the K-leg sewing table: wood-grain melamine top with the head cut-out filled by the bed, steel legs, knee
    # lifter, the motor's control box under the top, the pedal on the floor
    tt = VA.texture("white_oak_veneer")
    top = A.pbr("table_top", (200, 180, 150), rough=.45, diff=tt["diff"], rough_tex=tt["rough"], tile_m=.6)
    edge = A.pbr("table_edge", (60, 60, 62), rough=.5)
    steel = A.pbr("k_leg_steel", (70, 72, 76), rough=.45, metal=.6)
    T = LY.TABLE
    (x0, x1), (y0, y1), th = T["x"], T["y"], T["thickness"]
    bx, by = LY.BED["x"], LY.BED["y"]
    # the front edge rounded as the cloth meets it (cell.table_edge_radius: the drape wraps a 6 mm round in the physics;
    # with a square edge drawn the drape seems to fold over an edge it does not rest on)
    re = P.TABLE_EDGE_RADIUS
    for nm, lo, hi in (("front", (x0, y0 + re, -th), (x1, by[0], 0.)), ("back", (x0, by[1], -th), (x1, y1, 0.)),
                       ("left", (x0, by[0], -th), (bx[0], by[1], 0.)), ("right", (bx[1], by[0], -th), (x1, by[1], 0.))):
        A.box(f"/World/Table/top_{nm}", lo, hi, top, uv=True)
    A.cyl("/World/Table/edge_round", (x0, y0 + re, -re), (x1, y0 + re, -re), re, edge, sides=32)
    A.box("/World/Table/edge_front", (x0, y0, -th), (x1, y0 + re, -re), edge)
    for k, x in enumerate((x0 + .06, x1 - .06)):
        A.box(f"/World/Table/leg_post_{k}", (x - .02, -.02, zf + .03), (x + .02, .02, -th), steel)
        A.box(f"/World/Table/leg_foot_{k}", (x - .02, y0 + .02, zf), (x + .02, y1 - .02, zf + .03), steel)
        A.box(f"/World/Table/leg_arm_{k}", (x - .015, y0 + .03, -th - .03), (x + .015, y1 - .03, -th), steel)
    A.box("/World/Table/crossbar", (x0 + .06, -.02, zf + .25), (x1 - .06, .02, zf + .29), steel)
    A.box("/World/Table/control_box", (.18, -.02, -th - .20), (.42, .17, -th - .02), A.pbr("control_box", (215, 215, 210),
                                                                                           rough=.4))
    A.box("/World/Table/knee_lifter", (.05, -.20, -th - .16), (.14, -.19, -th - .08), steel)
    # the treadle pedal on the floor: the arm cells' machines are wired through the control box and nothing stands
    # there; in the humanoid deployment the G1 stands where the pedal is, and the physics has no pedal, so it is not
    # drawn (a cell wiring the head to its controller has none)
    humanoid = bool(world_meta.get("humanoid"))
    if not humanoid:
        A.box("/World/Table/pedal", (-.05, -.35, zf), (.10, -.10, zf + .04), A.pbr("pedal", (40, 40, 42), rough=.6))
    A.box("/World/Table/drawer", (x0 + .05, y0 + .02, -th - .12), (x0 + .40, y0 + .40, -th),
          A.pbr("drawer", (190, 180, 160), rough=.5))
    A.box("/World/Table/drawer_front", (x0 + .045, y0 + .012, -th - .125), (x0 + .405, y0 + .02, -th - .002),
          A.pbr("drawer_front", (205, 195, 175), rough=.4))
    A.box("/World/Table/drawer_handle", (x0 + .19, y0 + .002, -th - .07), (x0 + .26, y0 + .012, -th - .055),
          A.pbr("handle_steel", (200, 202, 206), rough=.25, metal=1.))
    # the servo motor's control box under the top: a cast housing with its power switch and cable to the pedal
    A.box("/World/Table/control_box_switch", (.40, -.022, -th - .12), (.42, -.018, -th - .08),
          A.pbr("switch_red", (190, 35, 30), rough=.4))
    if not humanoid:
        A.cyl("/World/Table/pedal_rod", (.025, -.22, zf + .04), (.25, .05, -th - .20), .004, steel)
    # ---- the robots' pedestal (steel column and plate) in the operator's seat: only where the physics builds it
    Pd = LY.PEDESTAL
    ped = A.pbr("pedestal", (80, 84, 90), rough=.4, metal=.5)
    draw_pedestal = world_meta.get("pedestal", True)
    if draw_pedestal:
        A.box("/World/Pedestal/plate", (Pd["x"][0], Pd["y"][0], Pd["z"][1] - .02),
              (Pd["x"][1], Pd["y"][1], Pd["z"][1] - 1e-3), ped)
        A.box("/World/Pedestal/column", ((Pd["x"][0] + Pd["x"][1])/2 - .08, (Pd["y"][0] + Pd["y"][1])/2 - .08, zf),
              ((Pd["x"][0] + Pd["x"][1])/2 + .08, (Pd["y"][0] + Pd["y"][1])/2 + .08, Pd["z"][1] - .02), ped)
        A.box("/World/Pedestal/base", (Pd["x"][0], Pd["y"][0] - .05, zf), (Pd["x"][1], Pd["y"][1] + .05, zf + .02), ped)
    # ---- the machine: static head, and the moving parts posed per frame
    A.usd("/World/Machine/static", MACHINE/"machine_static.usd")
    m = frame.get("machine") or dict(theta=0., needle_tip=float(P.POINT_BDC + P.STROKE), foot_z=0., dog=(0., 0.),
                                     take_up_deg=0.)
    A.usd("/World/Machine/needle_bar", MACHINE/"needle_bar.usd", (0., 0., m["needle_tip"]))
    A.usd("/World/Machine/presser_foot", MACHINE/"presser_foot.usd", (0., 0., m["foot_z"]))
    A.usd("/World/Machine/feed_dog", MACHINE/"feed_dog.usd", (0., m["dog"][0], m["dog"][1]))
    A.usd("/World/Machine/handwheel", MACHINE/"handwheel.usd", LY.HANDWHEEL["center"])
    A.usd("/World/Machine/take_up", MACHINE/"take_up.usd", TAKE_UP_PIVOT)
    FIXTURES = [(x, y) for x in (-2.0, 0.2, 2.4) for y in (-1.2, 0.9)]
    for k, (x, y) in enumerate(FIXTURES):
        try:
            A.reference(f"/World/Props/ceiling_light_{k}", VA.model("mounted_fluorescent_lights"), (x, y, zf + 2.97), 90.,
                        ground=False)
        except Exception:
            pass
    # ---- lights: the HDRI dome (low), six 4000 K ceiling fixtures (~500 lux on the table), the machine's LED
    # needle lamp (~1000 lux at the needle) [read: BG ETEM / Apparel Resources workplace lighting]
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    # the sky outside the windows (the closed room lets daylight in only through them) [authored intensity, checked
    # in renders: windows bright, the room lit mainly by the ceiling fixtures]
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(VA.hdri("kloofendal_48d_partly_cloudy_puresky").resolve())))
    dome.CreateIntensityAttr(DOME_NITS)
    for k, (x, y) in enumerate(FIXTURES):
        r = UsdLux.RectLight.Define(stage, f"/World/Lights/fixture_{k}")
        UsdGeom.Xformable(r).AddTranslateOp().Set(Gf.Vec3d(x, y, zf + 2.93))
        r.CreateWidthAttr(1.2)
        r.CreateHeightAttr(.25)
        r.CreateIntensityAttr(9000.)
        r.CreateColorTemperatureAttr(4000.)
        r.CreateEnableColorTemperatureAttr(True)
    lamp = UsdLux.RectLight.Define(stage, "/World/Lights/needle_lamp")
    UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(-.002, -.047, .089))
    lamp.CreateWidthAttr(.06)
    lamp.CreateHeightAttr(.012)
    lamp.CreateIntensityAttr(15000.)
    lamp.CreateColorTemperatureAttr(5500.)
    lamp.CreateEnableColorTemperatureAttr(True)
    # ---- robot bodies -> /World/Bodies/b_<i> (posed per frame) and the roots at their mounts
    labels = frame["bodies"]
    q = np.asarray(frame["body_q"], float)
    robot_mats = {}

    def link_vis(path, model, link):
        for k, (V, F, col) in enumerate(merged_link_visuals(model, link)):
            key = f"{model}_{col[0]}_{col[1]}_{col[2]}"
            if key not in robot_mats:
                robot_mats[key] = A.pbr("robot_" + key, col, rough=.35, metal=.1)
            A.tri_mesh(f"{path}/vis_{k}", V, F, robot_mats[key])
    body_paths = {}
    for i, lab in enumerate(labels):
        path = f"/World/Bodies/b_{i}"
        A.xform(path, q[i, :3], q[i, 3:7]).GetPrim().SetCustomDataByKey("gar_name", lab)
        body_paths[lab] = path
        bits = lab.split("/")
        if len(bits) >= 3 and bits[-2] in ROBOT_ROOTS:
            link_vis(path, bits[-2], bits[-1])
        elif len(bits) >= 3 and bits[-2] in ROBOT_ASSETS:
            link_vis(path, ROBOT_ASSETS[bits[-2]], bits[-1])
    for arm, info in world_meta["arms"].items():
        path = f"/World/Robots/{arm}_base"
        a = math.radians(info["mount_yaw_deg"])
        A.xform(path, info["mount_pos"], (0., 0., math.sin(a/2), math.cos(a/2)))
        link_vis(path, info["model"], ROBOT_ROOTS[info["model"]])
    # ---- plies (points streamed per frame)
    m_diff, m_nrm = muslin_textures()
    muslin = A.pbr("muslin", MUSLIN_SRGB, rough=.85, diff=m_diff, tint=MUSLIN_SRGB, normal=m_nrm, tile_m=1.)
    ply_paths = []
    for k, pm in enumerate(world_meta["plies"]):
        path = f"/World/Cloth/ply{k}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        tri = np.asarray(pm["tris"], np.int32)
        pts = frame["plies"][k]["points"] if frame.get("plies") else np.zeros((len(pm["uv"]), 3))
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(pm["uv"], np.float32)/MUSLIN_TILE_M))
        A.bind(mesh, muslin)
        ply_paths.append(path)
    # ---- threads: a pool of curves (points and counts set per frame)
    # navy Tex 30 spun polyester, as the cones on the stand: a contrasting thread, as sample rooms use to inspect a
    # seam (and the stitches the policy's cameras must see) [authored]
    thread_m = A.pbr("thread_polyester", THREAD_SRGB, rough=.55)
    UsdGeom.Scope.Define(stage, "/World/Threads")
    n_curves = 8
    for k in range(n_curves):
        _curve(stage, f"/World/Threads/c{k}", np.zeros((2, 3)), THREAD_W, thread_m)
    # ---- cameras
    cams = {**{k: dict(v, role="policy") for k, v in LY.CAMERAS.items()},
            **{k: dict(v, role="diagnostic") for k, v in LY.DIAGNOSTIC_CAMERAS.items()}, **(extra_cameras or {})}
    for name, c in cams.items():
        _camera(stage, "/Cameras/" + name, c["eye"], c["target"], c["hfov"], c.get("role", "diagnostic"))
    # robot-mounted cameras (the G1's head D435): a camera under the body's prim, which the worker poses every frame,
    # and an alias under /Cameras that names it (render_worker: mounted_camera_path)
    for name, c in world_meta.get("mounted_cameras", {}).items():
        mounted = f"{body_paths[c['body']]}/{name}_cam"
        _camera(stage, mounted, c["eye"], c["target"], c["fov"], c.get("role", "policy"), up=tuple(c["up"]))
        UsdGeom.Xform.Define(stage, "/Cameras/" + name).GetPrim().SetCustomDataByKey("mounted_camera_path", mounted)
        cams[name] = c
    stage.GetRootLayer().Save()
    meta = dict(bodies=labels, body_paths=body_paths, plies=ply_paths, threads=[f"/World/Threads/c{k}"
                                                                                for k in range(n_curves)],
                cameras=list(cams), exposure=EXPOSURE, handwheel_center=list(LY.HANDWHEEL["center"]),
                take_up_pivot=list(TAKE_UP_PIVOT))
    (folder/"export.json").write_text(json.dumps(meta, indent=1, default=str))
    return folder/"scene.usdc"


# ------------------------------------------------------------------------------------------------ worker client
def _enc(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def build_request(frame, meta, cameras, sequence):
    req = dict(op="observe", sequence=sequence, tick=int(frame["tick"]), cameras=list(cameras),
               exposure={c: meta["exposure"].get(c, meta["exposure"]["default"]) for c in cameras},
               body_q={meta["body_paths"][lab]: list(map(float, q)) for lab, q in zip(frame["bodies"], frame["body_q"])
                       if lab in meta["body_paths"]})
    m = frame.get("machine")
    if m is not None:
        th = m["theta"]
        hw = meta["handwheel_center"]
        a = math.radians(m["take_up_deg"])
        req["xforms"] = {"/World/Machine/needle_bar": [0., 0., m["needle_tip"], 0., 0., 0., 1.],
                         "/World/Machine/presser_foot": [0., 0., m["foot_z"], 0., 0., 0., 1.],
                         "/World/Machine/feed_dog": [0., m["dog"][0], m["dog"][1], 0., 0., 0., 1.],
                         "/World/Machine/handwheel": [*hw, math.sin(-th/2), 0., 0., math.cos(-th/2)],
                         "/World/Machine/take_up": [*meta["take_up_pivot"], 0., math.sin(a/2), 0., math.cos(a/2)]}
        req["plies"] = [dict(path=p, n=int(len(pl["points"])), points=_enc(pl["points"], np.float32),
                             normals=_enc(pl["normals"], np.float32)) for p, pl in zip(meta["plies"], frame["plies"])]
        th_ = frame.get("threads") or []
        req["threads"] = [dict(path=p, points=_enc(th_[k], np.float32) if k < len(th_) else None)
                          for k, p in enumerate(meta["threads"])]
    return req


class Renderer:
    """Owns one render-only Isaac worker (sewing.render_worker)."""

    def __init__(self, folder, scene, *, width=1280, height=720, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.closed, self.lock, self.sequence = False, threading.Lock(), 0
        self.scene = Path(scene)
        self.meta = json.loads((self.scene.parent/"export.json").read_text())
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        env = dict(os.environ, WIPE_RENDER_PORT=str(listener.getsockname()[1]), WIPE_RENDER_SECRET=secret,
                   WIPE_MINIMAL_RENDER="1")
        self.log = (self.folder/"worker.log").open("wb")
        self.process = None
        self.timings = []
        try:
            self.process = subprocess.Popen([executable, "-m", "sewing.render_worker", "--scene", str(self.scene.resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=ROOT.parent, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept()
            self.connection.settimeout(600)
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

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def observe(self, frame, cameras=("operator",)):
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            self.sequence += 1
            t0 = time.perf_counter()
            req = build_request(frame, self.meta, cameras, self.sequence)
            send(self.connection, req)
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
            if not result.get("ok"):
                raise RuntimeError("Renderer error: " + str(result.get("error")))
            if result.get("physics_steps") != 0:
                raise RuntimeError("Renderer stepped physics")
            self.timings.append(dict(result.get("timings", {}), total_s=time.perf_counter() - t0))
            return {k: base64.b64decode(v) for k, v in result["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

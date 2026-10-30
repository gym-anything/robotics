"""What the truck handles: GMA pallets, stretch-wrapped unit loads, loose cartons, steel coils on chocks, OCC bales and
their dunnage strips, each drawn on its simulated body from its collision shapes (physics dimensions are authoritative:
every drawn part fills its collider and stays within its faces, apart from a millimetre of film, strap or label).

This module is also the drawing kit the interior shares (interior.py imports it): a mesh buffer with bevelled boxes,
OmniPBR materials over the cached texture maps, generated textures (prints, labels, barcodes), prototypes and
instances, and the unit-load builder (a pallet's interlocked carton stack, optionally stretch-wrapped) that draws both
the loads on the forks and the loads stored in the racks.

Sources (forklift/assets/extra_sources_interior.json): cartons, the wrinkled stretch-film mesh, pallet wood, painted and
galvanised steel maps from NVIDIA's DigitalTwin warehouse pack; the Simple_Warehouse kit's cartons. Brand names on
generated prints are fictional.
"""

import functools
import math
import zlib

import numpy as np

from .common import KIT, box_half

DT = "nvidia/Assets/DigitalTwin/"
DT_BASE = DT + "Materials/Base/"
DT_WH = DT + "Assets/Warehouse/"
KIT_TEX = "Isaac/Environments/Simple_Warehouse/Materials/Textures/"
PALLET_ASM = DT_WH + "Shipping/Cardboard_Boxes_on_Pallet/Pallet_Asm_A/Pallet_Asm_A05_102x102x103cm_PR_V_NVD_01.usd"
ASM_TEX = DT_WH + "Shipping/Cardboard_Boxes_on_Pallet/Materials/textures/"
_BOXES = DT_WH + "Shipping/Cardboard_Boxes/"
CARTONS = {          # carton props by short name (native sizes are read from the assets)
    "flat15": _BOXES + "Flat_A/FlatBox_A15_51x63x33cm_PR_NVD_01.usd",
    "md10": _BOXES + "Multi-Depth_A/MultiDepthBox_A10_31x46x31cm_PR_NVD_01.usd",
    "md12": _BOXES + "Multi-Depth_A/MultiDepthBox_A12_36x51x36cm_PR_NVD_01.usd",
    "md05": _BOXES + "Multi-Depth_A/MultiDepthBox_A05_31x31x31cm_PR_NVD_01.usd",
    "white12": _BOXES + "White_A/WhiteCorrugatedBox_A12_51x52x52cm_PR_NVD_01.usd",
    "white11": _BOXES + "White_A/WhiteCorrugatedBox_A11_46x47x47cm_PR_NVD_01.usd",
    "kitA": KIT + "SM_CardBoxA_01.usd", "kitB": KIT + "SM_CardBoxB_01.usd", "kitC": KIT + "SM_CardBoxC_01.usd",
}
FILM = 0.008          # the shell between a wrapped load's cartons and its collider faces that the film's wrinkles occupy


# ====================================================================================================== cached assets
def tex(key):
    """Absolute path of a cached texture or asset, or None when it is not installed (the caller falls back to a flat
    colour or a plain box, so an export never fails on a missing visual asset)."""
    from ..assets.install import cache_dir
    p = cache_dir() / key
    return str(p) if p.is_file() else None


# ====================================================================================================== materials
RAW = ("normalmap_texture", "ORM_texture", "reflectionroughness_texture", "opacity_texture", "metallic_texture",
       "detail_normalmap_texture", "ao_texture")


def pbr(S, name, diffuse=None, normal=None, orm=None, rough=None, color=(0.5, 0.5, 0.5), tint=(1.0, 1.0, 1.0),
        rough_const=0.5, rough_infl=1.0, metal=0.0, metal_infl=None, scale=1.0, rotate=0.0, add=0.0, brightness=1.0,
        desat=0.0, bump=1.0, opacity=None, opacity_tex=None, emissive=None, spec=0.5):
    """OmniPBR material. Texture maps repeat every 1/scale metres on meshes whose st is in metres. A missing map (None)
    leaves the constant in charge. Normal, ORM, roughness and opacity maps are read as raw data, the diffuse as sRGB."""
    if name in S.mats:
        return S.mats[name]
    T, Gf, Sdf, UsdShade = S.Sdf.ValueTypeNames, S.Gf, S.Sdf, S.UsdShade
    v = dict(diffuse_color_constant=(T.Color3f, Gf.Vec3f(*map(float, color))), diffuse_tint=(T.Color3f, Gf.Vec3f(*map(float, tint))),
             reflection_roughness_constant=(T.Float, float(rough_const)), metallic_constant=(T.Float, float(metal)),
             specular_level=(T.Float, float(spec)), albedo_add=(T.Float, float(add)),
             albedo_brightness=(T.Float, float(brightness)), albedo_desaturation=(T.Float, float(desat)))
    sx, sy = (scale, scale) if np.isscalar(scale) else scale
    if (sx, sy) != (1.0, 1.0):
        v["texture_scale"] = (T.Float2, Gf.Vec2f(float(sx), float(sy)))
    if rotate:
        v["texture_rotate"] = (T.Float, float(rotate))
    if diffuse:
        v["diffuse_texture"] = (T.Asset, Sdf.AssetPath(str(diffuse)))
    if normal:
        v["normalmap_texture"] = (T.Asset, Sdf.AssetPath(str(normal)))
        v["bump_factor"] = (T.Float, float(bump))
    if orm:
        v["ORM_texture"] = (T.Asset, Sdf.AssetPath(str(orm)))
        v["enable_ORM_texture"] = (T.Bool, True)
        v["reflection_roughness_texture_influence"] = (T.Float, float(rough_infl))
        v["metallic_texture_influence"] = (T.Float, 1.0 if metal_infl is None else float(metal_infl))
    elif rough:
        v["reflectionroughness_texture"] = (T.Asset, Sdf.AssetPath(str(rough)))
        v["reflection_roughness_texture_influence"] = (T.Float, float(rough_infl))
    if opacity is not None or opacity_tex:
        v["enable_opacity"] = (T.Bool, True)
        v["opacity_constant"] = (T.Float, 1.0 if opacity is None else float(opacity))
        v["opacity_threshold"] = (T.Float, 0.0)
        if opacity_tex:
            v["enable_opacity_texture"] = (T.Bool, True)
            v["opacity_texture"] = (T.Asset, Sdf.AssetPath(str(opacity_tex)))
            v["opacity_mode"] = (T.Int, 1)
    if emissive is not None:
        v["enable_emission"] = (T.Bool, True)
        v["emissive_color"] = (T.Color3f, Gf.Vec3f(*map(float, emissive[:3])))
        v["emissive_intensity"] = (T.Float, float(emissive[3]))
    m = UsdShade.Material.Define(S.stage, "/Looks/" + name)
    sh = UsdShade.Shader.Define(S.stage, f"/Looks/{name}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    for k, (typ, val) in v.items():
        inp = sh.CreateInput(k, typ)
        inp.Set(val)
        if typ == T.Asset:
            inp.GetAttr().SetColorSpace("raw" if k in RAW else "sRGB")
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    S.mats[name] = m
    return m


# painted steel: NVIDIA's own recipe (Metal_Painted_*_A.mdl): the neutral painted albedo lifted by 0.35 and tinted
PAINT = dict(blue=(0.020, 0.085, 0.40), orange=(0.80, 0.21, 0.02), yellow=(0.86, 0.60, 0.02), grey=(0.33, 0.35, 0.37),
             white=(0.80, 0.81, 0.80), green=(0.035, 0.14, 0.075), red=(0.55, 0.03, 0.02), black=(0.025, 0.025, 0.028),
             dark=(0.10, 0.11, 0.12), cream=(0.74, 0.72, 0.64), ltblue=(0.25, 0.40, 0.55))
WOODS = dict(wood_new=((1.0, 0.98, 0.92), 1.05, 0.30), wood_mid=((0.94, 0.90, 0.82), 1.0, 0.45),
             wood_old=((0.84, 0.84, 0.82), 0.92, 0.65), wood_dark=((0.74, 0.68, 0.60), 0.80, 0.40))


def mat(S, key):
    """The shared material library by key (made on first use)."""
    name = "v_" + key
    if name in S.mats:
        return S.mats[name]
    M = DT_BASE + "Metals/Textures/"
    if key.startswith("paint_"):
        return pbr(S, name, tex(M + "T_Metal_Painted_Gray_A_Albedo.png"), tex(M + "T_Metal_Painted_Green_A_Normal.png"),
                   tex(M + "T_Metal_Painted_Green_A_ORM.png"), color=PAINT[key[6:]], tint=PAINT[key[6:]], add=0.35,
                   rough_const=0.45, spec=0.35, scale=2.0)
    if key in WOODS:
        tint, bright, desat = WOODS[key]
        W = DT_BASE + "Wood/Textures/T_Wood_Recycled_A1_"
        return pbr(S, name, tex(W + "Albedo.png"), tex(W + "Normal.png"), tex(W + "ORM.png"), color=(0.42, 0.30, 0.17),
                   tint=tint, brightness=bright, desat=desat, rough_const=0.85, scale=1.6)
    if key == "wood_blue":           # pooled pallets are painted (blue); the grain still shows through the paint
        W = DT_BASE + "Wood/Textures/T_Wood_Recycled_A1_"
        return pbr(S, name, tex(W + "Albedo.png"), tex(W + "Normal.png"), tex(W + "ORM.png"), color=(0.03, 0.12, 0.40),
                   tint=(0.10, 0.30, 0.95), desat=0.85, brightness=0.75, rough_const=0.7, scale=1.6)
    if key == "galv":                # galvanised steel
        return pbr(S, name, tex(M + "T_Steel_A1_Albedo.png"), tex(M + "T_Steel_A1_Normal.png"), tex(M + "T_Steel_A1_ORM.png"),
                   color=(0.55, 0.56, 0.58), metal=1.0, rough_const=0.4, spec=0.35, tint=(0.80, 0.82, 0.84), scale=2.0)
    if key == "steel":               # bare rolled steel, darker and rougher
        return pbr(S, name, tex(M + "T_Metal_Rough_A_Albedo.png"), tex(M + "T_Metal_Rough_A_Normal.png"),
                   tex(M + "T_Metal_Rough_A_ORM.png"), color=(0.30, 0.31, 0.33), metal=1.0, rough_const=0.5, scale=2.0)
    if key == "brushed":             # cold-rolled strip: brushed along u
        return pbr(S, name, tex(M + "T_Aluminium_Brushed_A1_Albedo.png"), tex(M + "T_Aluminium_Brushed_A1_Normal.png"),
                   tex(M + "T_Aluminium_Brushed_A1_ORM.png"), color=(0.60, 0.62, 0.66), tint=(0.86, 0.89, 0.95), metal=1.0,
                   rough_const=0.3, scale=(0.8, 3.0))
    if key == "rubber":
        R = DT_BASE + "Rubber/Textures/T_Rubber_New_A1_"
        return pbr(S, name, tex(R + "Albedo.png"), tex(R + "Normal.png"), tex(R + "ORM.png"), color=(0.02, 0.02, 0.022),
                   rough_const=0.85, scale=3.0)
    if key == "plastic_black":
        Pp = DT_BASE + "Plastics/Textures/T_Plastic_Black_A_"
        return pbr(S, name, tex(Pp + "Albedo.png"), tex(Pp + "Normal.png"), tex(Pp + "ORM.png"), color=(0.03, 0.03, 0.03),
                   rough_const=0.5, scale=3.0)
    if key in ("card", "card_white", "card_dark"):
        C = ASM_TEX + "T_Pallet_Assembly_Tile_"
        tint, bright, desat = {"card": ((1.0, 1.0, 1.0), 1.0, 0.0), "card_white": ((1.25, 1.3, 1.45), 1.25, 0.8),
                               "card_dark": ((0.85, 0.80, 0.72), 0.8, 0.1)}[key]
        return pbr(S, name, tex(C + "Albedo.png"), tex(C + "Normal.png"), tex(C + "ORM.png"), color=(0.52, 0.38, 0.22),
                   tint=tint, brightness=bright, desat=desat, rough_const=0.85, scale=1.2)
    if key == "film":                # stretch film: NVIDIA's Pallet_Assembly_A_Covering values, a little more body
        return pbr(S, name, color=(0.62, 0.62, 0.62), rough_const=0.06, metal=0.39, opacity=0.08)
    if key == "strap":               # polyester strapping / blued steel strap
        return pbr(S, name, color=(0.03, 0.04, 0.06), rough_const=0.35, metal=0.6)
    if key == "black":
        return pbr(S, name, color=(0.012, 0.012, 0.013), rough_const=0.6)
    raise KeyError(key)


# ====================================================================================================== mesh buffer
def _box_template():
    """Bevelled box as index data: 24 vertices on the six faces (inset by the bevel from the edges), 12 edge chamfers,
    8 corner triangles; flat normals. Returns per emitted vertex its corner signs, inset mask, normal and the axis its
    texture is projected along, and the triangles (the first 24 vertices and 12 triangles are the sharp box)."""
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    Sg, Mk, Nr, Ax, F = [], [], [], [], []

    def emit(corners, types, normal, axis):
        k = len(Sg)
        for c, t in zip(corners, types):
            m = np.ones(3); m[t] = 0.0
            Sg.append(signs[c]); Mk.append(m); Nr.append(normal); Ax.append(axis)
        F.extend((k, k + i, k + i + 1) for i in range(1, len(corners) - 1))
    ring = {(-1, -1): 0, (1, -1): 1, (1, 1): 2, (-1, 1): 3}
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (-1, 1):
            cs = sorted((i for i in range(8) if signs[i][a] == s), key=lambda i: ring[(signs[i][b], signs[i][c])])
            n = np.zeros(3); n[a] = s
            emit(cs, [a] * 4, n, a)
    for a in range(3):
        for b in range(a + 1, 3):
            c = 3 - a - b
            for sa in (-1, 1):
                for sb in (-1, 1):
                    lo = next(i for i in range(8) if signs[i][a] == sa and signs[i][b] == sb and signs[i][c] == -1)
                    hi = next(i for i in range(8) if signs[i][a] == sa and signs[i][b] == sb and signs[i][c] == 1)
                    n = np.zeros(3); n[a] = sa; n[b] = sb
                    emit([lo, hi, hi, lo], [a, a, b, b], n / math.sqrt(2.0), a)
    for i in range(8):
        emit([i, i, i], [0, 1, 2], signs[i] / math.sqrt(3.0), 2)
    Sg, Mk, Nr, Ax, F = np.array(Sg), np.array(Mk), np.array(Nr), np.array(Ax), np.array(F)
    Pt = Sg * (1.0 - 0.1 * Mk)                       # winding from a sample box
    g = np.cross(Pt[F[:, 1]] - Pt[F[:, 0]], Pt[F[:, 2]] - Pt[F[:, 0]])
    flip = (g * Nr[F[:, 0]]).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    return Sg, Mk, Nr, Ax, F


_BT = _box_template()
_PROJ = {None: (np.array([1, 0, 0]), np.array([2, 2, 1]))}          # texture axes (u, v) by projection axis
for _g in range(3):                                                  # with a grain axis: v runs along the grain
    _u, _v = [], []
    for _a in range(3):
        if _a == _g:
            o = [i for i in range(3) if i != _a]
            _u.append(o[0]); _v.append(o[1])
        else:
            _u.append(3 - _a - _g); _v.append(_g)
    _PROJ[_g] = (np.array(_u), np.array(_v))


class Buf:
    """Triangles of one mesh (positions, flat or smooth normals, st in metres), gathered part by part."""

    def __init__(self):
        self.V, self.N, self.UV, self.F, self.n = [], [], [], [], 0

    def add(self, V, N, UV, F, fix=False):
        """Append triangles. fix: turn every triangle to face the way its vertices' normals point."""
        V = np.asarray(V, np.float32)
        N = np.asarray(N, np.float32)
        F = np.asarray(F, np.int32).reshape(-1, 3)
        if fix and len(F):
            g = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            back = (g * (N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]])).sum(1) < 0
            F = np.where(back[:, None], F[:, ::-1], F)
        self.V.append(V); self.N.append(N); self.UV.append(np.asarray(UV, np.float32))
        self.F.append(F + self.n)
        self.n += len(V)

    def box(self, c, size, R=None, bevel=0.002, grain=None, uvs=1.0, uvo=(0.0, 0.0)):
        """Box of full `size` centred at c (rotated by R), chamfered by `bevel` (0: sharp). grain: the axis the texture's
        v runs along (boards). uvs: st per metre; uvo: st offset (to decorrelate neighbours)."""
        Sg, Mk, Nr, Ax, F = _BT
        h = np.asarray(size, float) / 2
        b = min(float(bevel), 0.45 * float(h.min()))
        if b <= 0.0:
            Sg, Mk, Nr, Ax, F = Sg[:24], Mk[:24], Nr[:24], Ax[:24], F[:12]
        Pt = Sg * (h - b * Mk)
        ua, va = _PROJ[grain]
        k = np.arange(len(Pt))
        uv = np.stack([Pt[k, ua[Ax]], Pt[k, va[Ax]]], 1) * uvs + np.asarray(uvo, float)
        if R is not None:
            R = np.asarray(R, float)
            Pt, Nr = Pt @ R.T, Nr @ R.T
        self.add(Pt + np.asarray(c, float), Nr, uv, F)

    def cyl(self, c, r, h, axis=2, n=20, caps=True, uvs=1.0, R=None, r2=None, a0=0.0, a1=2 * math.pi, inward=False):
        """Cylinder (or cone frustum with r2 at the top) of height h about `axis` through c; smooth sides, flat caps.
        a0..a1: an arc instead of the full turn. inward: normals face the axis (a bore)."""
        r2 = r if r2 is None else r2
        full = abs((a1 - a0) - 2 * math.pi) < 1e-9
        ang = np.linspace(a0, a1, n + 1)
        ca, sa = np.cos(ang), np.sin(ang)
        z = np.array([-h / 2, h / 2])
        rr = np.array([r, r2])
        V = np.stack([np.outer(ca, rr).ravel(), np.outer(sa, rr).ravel(), np.tile(z, n + 1)], 1)
        sl = (r - r2) / h if h else 0.0
        Nn = np.stack([np.repeat(ca, 2), np.repeat(sa, 2), np.full(2 * (n + 1), sl)], 1)
        Nn /= np.linalg.norm(Nn, axis=1)[:, None]
        UV = np.stack([np.repeat(ang * r, 2), np.tile(z, n + 1)], 1) * uvs
        i = np.arange(n) * 2
        F = np.concatenate([np.stack([i, i + 2, i + 3], 1), np.stack([i, i + 3, i + 1], 1)])
        if inward:
            Nn, F = -Nn, F[:, ::-1]
        parts = [(V, Nn, UV, F)]
        if caps and full:
            for zz, s, rad in ((-h / 2, -1.0, r), (h / 2, 1.0, r2)):
                if rad <= 0:
                    continue
                Vc = np.vstack([[0.0, 0.0, zz], np.stack([ca[:-1] * rad, sa[:-1] * rad, np.full(n, zz)], 1)])
                Nc = np.tile([0.0, 0.0, s], (n + 1, 1))
                j = np.arange(n)
                Fc = np.stack([np.zeros(n, int), 1 + j, 1 + (j + 1) % n], 1)
                parts.append((Vc, Nc, Vc[:, :2] * uvs, Fc if s > 0 else Fc[:, ::-1]))
        perm = {2: [0, 1, 2], 0: [2, 0, 1], 1: [1, 2, 0]}[axis]
        for V, Nn, UV, F in parts:
            V, Nn = V[:, perm], Nn[:, perm]
            if R is not None:
                V, Nn = V @ np.asarray(R, float).T, Nn @ np.asarray(R, float).T
            self.add(V + np.asarray(c, float), Nn, UV, F)

    def quad(self, c, u, v, size, st=(0.0, 0.0, 1.0, 1.0), both=False):
        """Flat rectangle centred at c spanned by unit vectors u, v (normal u x v); st = (u0, v0, u1, v1)."""
        c, u, v = (np.asarray(x, float) for x in (c, u, v))
        a, b = size[0] / 2, size[1] / 2
        Pt = np.array([c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v])
        n = np.cross(u, v)
        UV = np.array([(st[0], st[1]), (st[2], st[1]), (st[2], st[3]), (st[0], st[3])])
        self.add(Pt, np.tile(n, (4, 1)), UV, [(0, 1, 2), (0, 2, 3)])
        if both:
            self.add(Pt, np.tile(-n, (4, 1)), UV, [(0, 2, 1), (0, 3, 2)])

    def tube(self, pts, r, n=8, uvs=1.0, closed=False):
        """Round tube along a polyline (pipes, wires, conduit); smooth normals."""
        pts = np.asarray(pts, float)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        m = len(pts)
        d = np.gradient(pts, axis=0)
        if closed:
            d[0] = d[-1] = pts[1] - pts[-2]
        d /= np.linalg.norm(d, axis=1)[:, None]
        ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(d, e1)
        ang = np.linspace(0, 2 * math.pi, n + 1)
        Nn = (np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :])
        V = pts[:, None, :] + r * Nn
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        UV = np.stack([np.broadcast_to(ang * r, (m, n + 1)), np.broadcast_to(s[:, None], (m, n + 1))], 2) * uvs
        i, j = np.meshgrid(np.arange(m - 1), np.arange(n), indexing="ij")
        a = (i * (n + 1) + j).ravel(); b = a + 1; c_ = a + n + 1; d_ = c_ + 1
        self.add(V.reshape(-1, 3), Nn.reshape(-1, 3), UV.reshape(-1, 2), np.concatenate([np.stack([a, c_, d_], 1), np.stack([a, d_, b], 1)]))

    def prism(self, poly, x0, x1, axis=0, uvs=1.0):
        """Convex polygon (in the two axes after `axis`, counter-clockwise) extruded from x0 to x1 along `axis`."""
        poly = np.asarray(poly, float)
        k = len(poly)
        perm = {0: [2, 0, 1], 1: [1, 2, 0], 2: [0, 1, 2]}[axis]
        for i in range(k):
            p, q = poly[i], poly[(i + 1) % k]
            e = q - p
            nrm = np.array([e[1], -e[0], 0.0]); nrm /= np.linalg.norm(nrm)
            V = np.array([[p[0], p[1], x0], [q[0], q[1], x0], [q[0], q[1], x1], [p[0], p[1], x1]])
            L = float(np.linalg.norm(e))
            UV = np.array([[0, x0], [L, x0], [L, x1], [0, x1]]) * uvs
            self.add(V[:, perm], np.tile(nrm[perm], (4, 1)), UV, [(0, 1, 2), (0, 2, 3)])
        for xx, s in ((x0, -1.0), (x1, 1.0)):
            V = np.c_[poly, np.full(k, xx)]
            Fc = [(0, i, i + 1) for i in range(1, k - 1)]
            n = np.array([0.0, 0.0, s])
            self.add(V[:, perm], np.tile(n[perm], (k, 1)), poly * uvs, Fc if s > 0 else [f[::-1] for f in Fc])

    def write(self, S, path, material):
        if not self.n:
            return None
        return S.mesh(path, np.concatenate(self.V), np.concatenate(self.F), uv=np.concatenate(self.UV),
                      normals=np.concatenate(self.N), mat=material)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_z(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def quat_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, float)).as_matrix()


def seed_of(*parts):
    """Deterministic 32-bit seed from names and numbers."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ====================================================================================================== prototypes
def proto(S, name):
    """Path of a prototype under /Protos (invisible there; drawn only through instances)."""
    if not S.stage.GetPrimAtPath("/Protos"):
        S.UsdGeom.Imageable(S.UsdGeom.Xform.Define(S.stage, "/Protos")).MakeInvisible()
    return "/Protos/" + name


def has(S, path):
    return bool(S.stage.GetPrimAtPath(path))


def inst(S, path, proto_path, p=(0.0, 0.0, 0.0), q=None, scale=None):
    """An instance of a prototype at p (orientation q xyzw, scale). Queued: flush(S) authors the queue (a draw() calls
    it before it returns)."""
    from ..usd import clean
    S.__dict__.setdefault("_instances", []).append(
        (clean(path), proto_path, (float(p[0]), float(p[1]), float(p[2])), None if q is None else tuple(float(v) for v in q),
         None if scale is None else tuple(float(v) for v in scale)))


def flush(S):
    """Author the queued instances through the layer in one change block. (Through the stage each instanceable
    reference recomposed the stage: 7 ms apiece, 32 s for the warehouse's 4,300.)"""
    queue = S.__dict__.pop("_instances", [])
    if not queue:
        return
    Sdf, Gf, Vt = S.Sdf, S.Gf, S.Vt
    layer = S.stage.GetRootLayer()
    T = Sdf.ValueTypeNames
    with Sdf.ChangeBlock():
        for path, proto_path, p, q, scale in queue:
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.specifier, spec.typeName = Sdf.SpecifierDef, "Xform"
            par = spec.nameParent
            while par is not None and par.path != Sdf.Path.absoluteRootPath and par.specifier == Sdf.SpecifierOver:
                par.specifier = Sdf.SpecifierDef                     # ancestors the layer made on the way are groups
                if not par.typeName:
                    par.typeName = "Xform"
                par = par.nameParent
            spec.referenceList.Prepend(Sdf.Reference(primPath=proto_path))
            spec.instanceable = True
            order = ["xformOp:translate"]
            Sdf.AttributeSpec(spec, "xformOp:translate", T.Double3).default = Gf.Vec3d(*p)
            if q is not None:
                Sdf.AttributeSpec(spec, "xformOp:orient", T.Quatf).default = Gf.Quatf(q[3], Gf.Vec3f(q[0], q[1], q[2]))
                order.append("xformOp:orient")
            if scale is not None:
                Sdf.AttributeSpec(spec, "xformOp:scale", T.Float3).default = Gf.Vec3f(*scale)
                order.append("xformOp:scale")
            Sdf.AttributeSpec(spec, "xformOpOrder", T.TokenArray, Sdf.VariabilityUniform).default = Vt.TokenArray(order)


@functools.lru_cache(maxsize=None)
def asset_bounds(path):
    """(min, max) in metres of a USD asset's meshes in its own frame (z up), from the mesh points (authored extents of
    some props are stale), and its metres per unit."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(st.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        M = np.array(cache.GetLocalToWorldTransform(prim))
        W = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M
        lo, hi = np.minimum(lo, W[:, :3].min(0)), np.maximum(hi, W[:, :3].max(0))
    if UsdGeom.GetStageUpAxis(st) == "Y":
        lo, hi = lo[[0, 2, 1]] * [1, -1, 1], hi[[0, 2, 1]] * [1, -1, 1]
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return lo * mpu, hi * mpu, mpu, UsdGeom.GetStageUpAxis(st)


def asset_proto(S, name, key, unit=False, fallback=(0.5, 0.5, 0.5)):
    """Prototype of a cached USD prop with its base centre at the origin, in metres (or, with unit=True, squeezed into
    the unit box [-0.5, 0.5]^2 x [0, 1] so that an instance's scale is the size it is drawn at). Returns (prototype
    path, native size in metres)."""
    P = proto(S, name)
    path = tex(key)
    if path is None:                               # prop not installed: a plain box of the fallback size
        if not has(S, P):
            S.UsdGeom.Xform.Define(S.stage, P)
            b = Buf()
            sz = (1.0, 1.0, 1.0) if unit else fallback
            b.box((0, 0, sz[2] / 2), sz, bevel=0.004 / max(fallback), uvs=1.0)
            b.write(S, P + "/box", mat(S, "card"))
        return P, np.asarray(fallback, float)
    lo, hi, mpu, up = asset_bounds(path)
    size = hi - lo
    if has(S, P):
        return P, size
    S.UsdGeom.Xform.Define(S.stage, P)
    geo = S.UsdGeom.Xform.Define(S.stage, P + "/geo")
    c = (lo + hi) / 2
    k = (1.0 / size) if unit else np.ones(3)
    # ops apply right to left: the asset's units to metres, (y-up to z-up), base centre to the origin, then the squeeze
    geo.AddScaleOp(opSuffix="fit").Set(S.Gf.Vec3f(*map(float, k)))
    geo.AddTranslateOp().Set(S.Gf.Vec3d(float(-c[0]), float(-c[1]), float(-lo[2])))
    if up == "Y":
        geo.AddRotateXOp().Set(90.0)
    geo.AddScaleOp().Set(S.Gf.Vec3f(mpu, mpu, mpu))
    S.stage.DefinePrim(P + "/geo/asset").GetReferences().AddReference(str(path))
    return P, size


# ====================================================================================================== generated textures
def font(size, mono=False):
    from PIL import ImageFont
    for nm in (("DejaVuSansMono-Bold.ttf",) if mono else ("DejaVuSans-Bold.ttf",)):
        try:
            return ImageFont.truetype(nm, int(size))
        except OSError:
            pass
    return ImageFont.load_default()


def save_tex(S, name, img):
    """Write a generated image into the stage's textures folder; returns its path."""
    p = S.folder / "textures" / name
    img.save(p, compress_level=1)          # always rewritten: a stage folder is reused across exports
    return str(p)


_C128 = ("212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 221312 231212 112232 122132 122231 113222 "
         "123122 123221 223211 221132 221231 213212 223112 312131 311222 321122 321221 312212 322112 322211 212123 212321 "
         "232121 111323 131123 131321 112313 132113 132311 211313 231113 231311 112133 112331 132131 113123 113321 133121 "
         "313121 211331 231131 213113 213311 213131 311123 311321 331121 312113 312311 332111 314111 221411 431111 111224 "
         "111422 121124 121421 141122 141221 112214 112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 "
         "111242 121142 121241 114212 124112 124211 411212 421112 421211 212141 214121 412121 111143 111341 131141 114113 "
         "114311 411113 411311 113141 114131 311141 411131 211412 211214 211232 2331112").split()


def code128(text):
    """Module widths (bar, space, bar, ...) of a Code 128 set-B barcode of an ASCII text."""
    vals = [ord(ch) - 32 for ch in text]
    chk = (104 + sum((i + 1) * v for i, v in enumerate(vals))) % 103
    return [int(d) for v in [104] + vals + [chk, 106] for d in _C128[v]]


def draw_barcode(d, box, text, fill=(0, 0, 0)):
    """Draw a Code 128 barcode into the rectangle box = (x0, y0, x1, y1) of an ImageDraw."""
    w = code128(text)
    x0, y0, x1, y1 = box
    unit = (x1 - x0) / (sum(w) + 20.0)
    x = x0 + 10 * unit
    for i, m in enumerate(w):
        if i % 2 == 0:
            d.rectangle([x, y0, x + m * unit - 0.5, y1], fill=fill)
        x += m * unit


BRANDS = (("NORTHFIELD", "PANTRY SUPPLY", (20, 60, 120)), ("KESTREL", "HARDWARE CO.", (25, 25, 25)),
          ("BLUE HERON", "HOME GOODS", (15, 70, 110)), ("ALDERWOOD", "PAPER PRODUCTS", (30, 90, 45)),
          ("SUMMIT", "AUTO PARTS", (150, 25, 20)), ("TALLGRASS", "FOODS", (120, 70, 10)))


def print_material(S, k):
    """One-colour flexo print for a carton's side face (brand, item line, handling marks, case barcode), as a decal:
    ink where the mask is opaque, bare board elsewhere."""
    name = f"v_print{k}"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw
    brand, sub, ink = BRANDS[k % len(BRANDS)]
    Wd, Ht = 1024, 512
    m = Image.new("L", (Wd, Ht), 0)
    d = ImageDraw.Draw(m)
    rng = np.random.default_rng(100 + k)
    style = k % 3
    if style == 0:                                   # logo block left, text right
        d.rounded_rectangle([40, 60, 300, 320], 30, outline=255, width=14)
        d.polygon([(90, 270), (170, 110), (250, 270)], fill=255)
        d.text((340, 70), brand, font=font(92), fill=255)
        d.text((344, 180), sub, font=font(46), fill=255)
    elif style == 1:                                 # banner
        d.rectangle([30, 60, Wd - 30, 230], fill=255)
        d.text((60, 85), brand, font=font(104), fill=0)
        d.text((60, 250), sub, font=font(52), fill=255)
    else:                                            # ring logo centre
        d.ellipse([60, 50, 330, 320], outline=255, width=16)
        d.text((122, 130), brand[:2], font=font(120), fill=255)
        d.text((370, 90), brand, font=font(84), fill=255)
        d.line([370, 196, 960, 196], fill=255, width=8)
        d.text((372, 212), sub, font=font(44), fill=255)
    item = f"ITEM {int(rng.integers(10000, 99999))}-{'ABCDEF'[k % 6]}   QTY {int(rng.choice([6, 12, 24, 48]))}"
    d.text((44, 372), item, font=font(40, mono=True), fill=255)
    d.text((44, 426), f"LOT {int(rng.integers(100, 999))}{'KLMNPR'[k % 6]}{int(rng.integers(10, 99))}   MADE FOR {brand}",
           font=font(30, mono=True), fill=255)
    # this-side-up arrows and a case code in a quiet zone
    for i in range(2):
        x = 820 + i * 80
        d.polygon([(x, 330), (x + 30, 290), (x + 60, 330), (x + 40, 330), (x + 40, 400), (x + 20, 400), (x + 20, 330)], fill=255)
    d.rectangle([800, 404, 980, 412], fill=255)
    bars = Image.new("L", (300, 90), 0)
    draw_barcode(ImageDraw.Draw(bars), (0, 0, 300, 90), f"{int(rng.integers(10 ** 7, 10 ** 8))}", fill=255)
    m.paste(bars, (500, 400))
    # flexo ink is uneven: speckle the mask
    a = np.asarray(m, np.float32) / 255.0
    a *= 0.80 + 0.20 * rng.random(a.shape, dtype=np.float32)
    mask = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255), "L")
    rgb = Image.new("RGB", (Wd, Ht), ink)
    return pbr(S, name, save_tex(S, f"print{k}_rgb.png", rgb), color=tuple(c / 255 for c in ink), rough_const=0.8,
               opacity_tex=save_tex(S, f"print{k}_mask.png", mask))


def pallet_label_material(S):
    """Pallet labels (four on one sheet): white A5 logistics labels with ship-to block, SSCC and case barcodes."""
    name = "v_pallet_label"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image, ImageDraw
    cw, ch = 592, 420
    img = Image.new("RGB", (cw * 2, ch * 2), (236, 236, 232))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(31)
    for k in range(4):
        x0, y0 = (k % 2) * cw, (k // 2) * ch
        brand = BRANDS[k % len(BRANDS)][0]
        d.rectangle([x0 + 6, y0 + 6, x0 + cw - 7, y0 + ch - 7], outline=(20, 20, 20), width=3)
        d.text((x0 + 20, y0 + 16), "FROM", font=font(16), fill=(0, 0, 0))
        d.text((x0 + 20, y0 + 36), f"{brand} DC {int(rng.integers(10, 99))}", font=font(24), fill=(0, 0, 0))
        d.line([x0 + 6, y0 + 74, x0 + cw - 7, y0 + 74], fill=(20, 20, 20), width=3)
        d.text((x0 + 20, y0 + 82), f"SSCC  0 0{int(rng.integers(10 ** 6, 10 ** 7))} {int(rng.integers(10 ** 8, 10 ** 9))} {int(rng.integers(0, 9))}",
               font=font(22, mono=True), fill=(0, 0, 0))
        d.text((x0 + 20, y0 + 114), f"ITEM {int(rng.integers(10 ** 4, 10 ** 5))}   QTY {int(rng.choice([24, 36, 48, 60]))} CS   LOT {int(rng.integers(100, 999))}",
               font=font(20, mono=True), fill=(0, 0, 0))
        d.line([x0 + 6, y0 + 148, x0 + cw - 7, y0 + 148], fill=(20, 20, 20), width=3)
        draw_barcode(d, (x0 + 24, y0 + 162, x0 + cw - 24, y0 + 268), f"00{int(rng.integers(10 ** 9, 10 ** 10))}")
        draw_barcode(d, (x0 + 60, y0 + 290, x0 + cw - 60, y0 + 372), f"02{int(rng.integers(10 ** 7, 10 ** 8))}37")
        d.text((x0 + 150, y0 + 378), f"(00) {int(rng.integers(10 ** 9, 10 ** 10))}", font=font(18, mono=True), fill=(0, 0, 0))
    return pbr(S, name, save_tex(S, "pallet_labels.png", img), rough_const=0.55)


# ====================================================================================================== pallets
def gma_parts():
    """[(centre, size, kind)] of the GMA pallet's boards in its own frame (x along the stringers, z up from the floor):
    the collision boxes of forklift.loads.gma_boxes, which are the boards themselves."""
    from .. import loads as LD
    return [(np.asarray(c, float), 2 * np.asarray(h, float), kind) for c, h, kind in LD.gma_boxes()]


def _board(bufs, k, c, size, rng, R=None, woods=3):
    """One board into the buffer of the wood its tint falls in: bevelled, grain along its length, its own offset in the
    grain texture so no two boards repeat."""
    long_axis = int(np.argmax(size))
    bufs[int(rng.integers(0, woods)) if k is None else k].box(c, size, R=R, bevel=0.0025, grain=long_axis,
                                                              uvo=rng.uniform(0.0, 4.0, 2))


PALLET_WOODS = (("wood_new", "wood_mid", "wood_old"), ("wood_mid", "wood_old", "wood_dark"), ("wood_new", "wood_new", "wood_mid"),
                ("wood_blue", "wood_blue", "wood_blue"))


def pallet_proto(S, variant):
    """GMA stringer pallet prototype (origin at the floor under its centre). Variants differ in which board takes which
    of three wood tones; variant 3 is a painted pooled pallet. The stringers are solid, as the colliders are: this is a
    two-way pallet (no notches)."""
    P = proto(S, f"pallet{variant}")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    rng = np.random.default_rng(40 + variant)
    bufs = [Buf(), Buf(), Buf()]
    for c, size, kind in gma_parts():
        _board(bufs, None, c, size, rng)
    nails = Buf()
    for c, size, kind in gma_parts():                 # nail heads where the top boards cross the stringers
        if kind == "top_deck":
            for y in (-0.49, 0.0, 0.49):
                for dx in (-0.012, 0.014):
                    if abs(dx) < size[0] / 2:
                        nails.cyl((c[0] + dx, y + rng.uniform(-0.004, 0.004), c[2] + size[2] / 2 + 0.0002), 0.0028, 0.0006, n=6)
    for b, key in zip(bufs, PALLET_WOODS[variant % len(PALLET_WOODS)]):
        b.write(S, f"{P}/{key}_{bufs.index(b)}", mat(S, key))
    nails.write(S, P + "/nails", mat(S, "steel"))
    return P


def draw_pallet_shapes(S, path, parts, variant, rng):
    """A pallet from the collision boxes of its body: [(position, quaternion, half extents)] in the body frame."""
    bufs = [Buf(), Buf(), Buf()]
    for p, q, half in parts:
        _board(bufs, None, p, 2 * np.asarray(half, float), rng, R=quat_R(q))
    for i, (b, key) in enumerate(zip(bufs, PALLET_WOODS[variant % len(PALLET_WOODS)])):
        b.write(S, f"{path}/board_{key}_{i}", mat(S, key))


# ====================================================================================================== cartons
def carton_proto(S, kind, design=None):
    """A carton squeezed into the unit box (an instance's scale is its size); design: index of a printed side decal.
    Returns (prototype path, native size)."""
    base, size = asset_proto(S, "carton_" + kind, CARTONS[kind], unit=True, fallback=(0.5, 0.4, 0.3))
    if design is None:
        return base, size
    P = proto(S, f"carton_{kind}_p{design}")
    if not has(S, P):
        S.UsdGeom.Xform.Define(S.stage, P)
        S.stage.DefinePrim(P + "/box", "Xform").GetReferences().AddInternalReference(base)
        b = Buf()
        e = 0.5 + 0.0012 / float(min(size[:2]))
        for n, u in (((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, 1, 0), (-1, 0, 0)), ((0, -1, 0), (1, 0, 0))):
            b.quad((n[0] * e, n[1] * e, 0.47), u, (0, 0, 1), (0.80, 0.56))
        b.write(S, P + "/print", print_material(S, design))
    return P, size


# ====================================================================================================== stretch wrap
@functools.lru_cache(maxsize=None)
def _film_source():
    """The wrinkled film mesh of NVIDIA's wrapped pallet and the box of cartons it was modelled around (cm)."""
    path = tex(PALLET_ASM)
    if path is None:
        return None
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(path))

    def world(p):
        prim = st.GetPrimAtPath(p)
        m = UsdGeom.Mesh(prim)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = np.asarray(m.GetPointsAttr().Get(), float)
        return (np.c_[pts, np.ones(len(pts))] @ M)[:, :3], m, M[:3, :3]
    C, m, M3 = world("/World/Covering/SM_Pallet_Assembly_A05_Covering_01")
    Bx = world("/World/SM_Pallet_Assembly_A05_Boxes_01")[0]
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int32)
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int32)
    Nf = np.asarray(m.GetNormalsAttr().Get(), float) @ np.linalg.inv(M3).T          # face-varying
    Nf /= np.linalg.norm(Nf, axis=1)[:, None]
    return C, counts, idx, Nf, Bx.min(0), Bx.max(0)


def film_proto(S, mirror=False):
    """Stretch film prototype in the unit box (an instance's scale is the load's collider box): NVIDIA's wrinkled film
    mesh pressed into the shell between the cartons and the collider faces. The original stands up to 5 cm off its
    cartons; here every point's stand-off is compressed into the FILM shell, and the original normals are kept, so the
    wrinkles shade as modelled while the film stays within the collider. Falls back to a plain shell without the asset."""
    P = proto(S, "film_m" if mirror else "film")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    src = _film_source()
    ref = np.array([1.18, 0.98, 1.15])                       # the load the shell fractions are sized for
    g = FILM / ref
    if src is None:
        b = Buf()
        b.box((0, 0, 0.5), (1.0, 1.0, 1.0), bevel=0.01)
        b.write(S, P + "/film", mat(S, "film"))
        return P
    C, counts, idx, Nf, lo, hi = src
    c, hb = (lo + hi) / 2, (hi - lo) / 2
    # every point keeps its place over the carton box and its stand-off from the box's faces, the stand-off compressed
    # from the original's wrinkle depth into the shell (decided by position: wrinkle normals point every way)
    out = np.empty_like(C)
    smax = 4.5                                               # cm: the deepest wrinkle of the original
    for a in range(2):
        r = np.abs(C[:, a] - c[a])
        out[:, a] = np.sign(C[:, a] - c[a]) * np.where(r <= hb[a], r / hb[a] * (0.5 - g[a]),
                                                       (0.5 - g[a]) + g[a] * np.clip((r - hb[a]) / smax, 0.0, 1.0))
    # height: the cartons stand on the pallet (no film under them) and the film's top sheet lies in the shell above
    # them; the skirt the original hangs over its pallet folds onto the bottom edge
    t = np.clip((C[:, 2] - lo[2]) / (hi[2] - lo[2]), 0.0, 1.0)
    out[:, 2] = np.where(C[:, 2] > hi[2], (1.0 - g[2]) + g[2] * np.clip((C[:, 2] - hi[2]) / smax, 0.0, 1.0), t * (1.0 - g[2]))
    Nn = Nf * (2 * hb)                                       # so that an instance's scale reproduces the modelled normals
    Nn /= np.linalg.norm(Nn, axis=1)[:, None]
    if mirror:
        out[:, 0] *= -1.0
        Nn[:, 0] *= -1.0
        ends = np.cumsum(counts)                             # reverse each face's winding
        face = np.repeat(np.arange(len(counts)), counts)
        order = ends[face] - 1 - (np.arange(len(idx)) - (ends - counts)[face])
        idx, Nn = idx[order], Nn[order]
    g_ = S.UsdGeom.Mesh.Define(S.stage, P + "/film")
    g_.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(out.astype(np.float32)))
    g_.CreateFaceVertexCountsAttr(S.Vt.IntArray.FromNumpy(counts))
    g_.CreateFaceVertexIndicesAttr(S.Vt.IntArray.FromNumpy(idx))
    g_.CreateSubdivisionSchemeAttr("none")
    g_.CreateNormalsAttr(S.Vt.Vec3fArray.FromNumpy(Nn.astype(np.float32)))
    g_.SetNormalsInterpolation("faceVarying")
    g_.CreateExtentAttr(S.Vt.Vec3fArray.FromNumpy(np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]], np.float32)))
    S.UsdGeom.PrimvarsAPI(g_).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    S.bind(g_, mat(S, "film"))
    # corner boards under the film: cardboard angles on the four vertical edges
    b = Buf()
    t, w = 0.004 / ref[:2], 0.045 / ref[:2]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ex, ey = sx * (0.5 - g[0] * 0.9), sy * (0.5 - g[1] * 0.9)
            b.box((ex - sx * t[0] / 2, ey - sy * w[1] / 2, 0.5), (t[0], w[1], 0.96), bevel=0.0)
            b.box((ex - sx * w[0] / 2, ey - sy * t[1] / 2, 0.5), (w[0], t[1], 0.96), bevel=0.0)
    b.write(S, P + "/corner_boards", mat(S, "card_white"))
    return P


def label_proto(S, k):
    """A pallet label (A5, 210 x 148 mm) lying in the x-z plane facing -y, origin at its centre."""
    P = proto(S, f"pallet_label{k % 4}")
    if not has(S, P):
        S.UsdGeom.Xform.Define(S.stage, P)
        b = Buf()
        u0, v0 = (k % 2) * 0.5, 1.0 - ((k % 4) // 2) * 0.5
        b.quad((0, 0, 0), (1, 0, 0), (0, 0, 1), (0.210, 0.148), st=(u0, v0 - 0.5, u0 + 0.5, v0))
        b.write(S, P + "/label", pallet_label_material(S))
    return P


# ====================================================================================================== unit loads
def _grid(nx, ny):
    """Column pattern: nx by ny equal cells of the unit square, [(cx, cy, sx, sy)]."""
    return [((i + 0.5) / nx - 0.5, (j + 0.5) / ny - 0.5, 1.0 / nx, 1.0 / ny) for i in range(nx) for j in range(ny)]


def _interlock(ax, ay, flip):
    """Five-per-layer interlock of 3:2 cartons on a 48 x 40 in footprint: two lengthways side by side and three
    crossways beside them; alternate layers are mirrored so no seam runs through the stack."""
    w = (ax / 3) / (ax / 3 + ax / 2)                      # share of the depth the lengthways pair takes
    s = -1.0 if flip else 1.0
    cells = [((i + 0.5) / 2 - 0.5, s * (-0.5 + w / 2), 0.5, w) for i in range(2)]
    cells += [((i + 0.5) / 3 - 0.5, s * (-0.5 + w + (1 - w) / 2), 1.0 / 3, 1 - w) for i in range(3)]
    return cells


# pattern name, the cartons that suit its cell shape, weight
PATTERNS = (("3x2", ("flat15", "kitA", "white11"), 3.0), ("interlock", ("md10", "md12", "kitA"), 3.0),
            ("2x2", ("white12", "kitB", "flat15"), 2.0), ("4x3", ("md05", "white11", "kitB"), 2.0),
            ("2x3", ("md12", "md10"), 1.5), ("3x3", ("md05", "kitC", "white11"), 1.5))


def unit_load(S, path, size, seed, p=(0.0, 0.0, 0.0), q=None, wrapped=None, label_side=None):
    """A unit load filling the box of `size` (lx, ly, h) whose base centre is at p (frame of the parent prim): layers of
    one carton type in a column or interlocked pattern, optionally stretch-wrapped with corner boards and a pallet
    label on the side `label_side` ('+x', '-x', '+y', '-y'). Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    lx, ly, h = (float(v) for v in size)
    if wrapped is None:
        wrapped = rng.random() < 0.7
    g = FILM if wrapped else 0.0015
    ax, ay, az = lx - 2 * g, ly - 2 * g, h - (g if wrapped else 0.0)
    wts = np.array([w for _, _, w in PATTERNS])
    name, kinds, _ = PATTERNS[int(rng.choice(len(PATTERNS), p=wts / wts.sum()))]
    kind = kinds[int(rng.integers(0, len(kinds)))]
    design = int(rng.integers(0, len(BRANDS))) if (rng.random() < 0.55 and not kind.startswith("kit")) else None
    cp, native = carton_proto(S, kind, design)
    layers = [_interlock(ax, ay, False), _interlock(ax, ay, True)] if name == "interlock" else \
        [_grid(*(int(v) for v in name.split("x")))] * 2
    sx0, sy0 = layers[0][0][2] * ax, layers[0][0][3] * ay
    k = math.sqrt((sx0 * sy0) / (native[0] * native[1]))            # the carton's mean horizontal scale
    n = max(1, int(round(az / (native[2] * k))))
    lh = az / n
    S.xform(path, p, q if q is not None else (0.0, 0.0, 0.0, 1.0))
    gap = 0.004
    i = 0
    for layer in range(n):
        for cx, cy, sx, sy in layers[layer % 2]:
            sx, sy = sx * ax, sy * ay
            turn = (native[0] >= native[1]) != (sx >= sy)
            yaw = (math.pi / 2 if turn else 0.0) + (math.pi if rng.random() < 0.5 else 0.0) + float(rng.uniform(-0.006, 0.006))
            sc = (sy - gap, sx - gap, lh - 0.0015) if turn else (sx - gap, sy - gap, lh - 0.0015)
            j = rng.uniform(-0.0012, 0.0012, 2)
            inst(S, f"{path}/c{i}", cp, (cx * ax + j[0], cy * ay + j[1], layer * lh), quat_z(yaw), sc)
            i += 1
    if wrapped:
        inst(S, path + "/film", film_proto(S, mirror=bool(rng.integers(0, 2))), (0.0, 0.0, 0.0),
             quat_z(math.pi * int(rng.integers(0, 2))), (lx, ly, h))
        for k, side in enumerate((label_side,) if isinstance(label_side, str) else (label_side or ())):
            zl = min(h - 0.16, 0.42 + float(rng.uniform(0.0, 0.25)))
            off = float(rng.uniform(-0.18, 0.18))
            pos, yaw = {"-y": ((off, -ly / 2 - 0.0008, zl), 0.0), "+y": ((off, ly / 2 + 0.0008, zl), math.pi),
                        "+x": ((lx / 2 + 0.0008, off, zl), math.pi / 2), "-x": ((-lx / 2 - 0.0008, off, zl), -math.pi / 2)}[side]
            inst(S, f"{path}/label{k}", label_proto(S, int(rng.integers(0, 4))), pos, quat_z(yaw + float(rng.uniform(-0.04, 0.04))))
    return dict(pattern=name, carton=kind, layers=n, wrapped=wrapped)


# ====================================================================================================== bales
def o_vec(i):
    v = np.zeros(3); v[i] = 1.0
    return v


def _bale_textures(S):
    """Albedo and normal maps of a compressed cardboard (OCC) bale. Side: the flakes seen edge-on (fine strata across
    u, the bale's compression axis, in flakes a hand wide). End: the last flake seen flat (crushed sheets, creases).
    Mostly one kraft tone, as a bale of sorted corrugated is: variation is in the shadows, with the odd white-lined or
    printed sheet."""
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(77)
    kraft = np.array([166.0, 141.0, 110.0])

    def tone(k=0.07):
        t = kraft * rng.normal(1.0, k)
        return t * np.array([1.0, rng.normal(1.0, 0.012), rng.normal(1.0, 0.025)])
    out = {}
    # --- side: strata. Each flake (4-11 cm of the long axis) is a pack of crushed sheets seen edge-on: thin layers that
    # run a hand or two along the face before another sheet's edge takes over
    Wd, Ht = 2048, 1024
    col = np.zeros((Ht, Wd, 3), np.float32)
    hgt = np.zeros((Ht, Wd), np.float32)
    x = 0
    while x < Wd:
        flake = int(rng.integers(60, 150))
        x1 = min(Wd, x + flake)
        y = 0
        while y < Ht:
            seg = int(rng.integers(70, 320))
            y1 = min(Ht, y + seg)
            ft = rng.normal(1.0, 0.035)
            xx = x + int(rng.integers(-2, 3))
            while xx < x1:
                w = int(rng.choice([1, 2, 3, 4, 6], p=[0.25, 0.35, 0.2, 0.12, 0.08]))
                r = rng.random()
                c = tone(0.06) * ft
                if r < 0.02:
                    c = np.array([206.0, 200.0, 188.0]) * rng.uniform(0.85, 1.0)     # white-lined board
                elif r < 0.03:
                    c = np.array([[120, 74, 66], [78, 92, 112], [76, 104, 84]], float)[int(rng.integers(0, 3))]
                a0, a1 = max(xx, x), min(xx + w, x1)
                if a1 > a0:
                    col[y:y1, a0:a1] = c
                    hgt[y:y1, a0:a1] = rng.uniform(0.5, 1.0)
                xx += w
            if y1 < Ht:                                                              # the sheet's edge: a ragged thin shadow
                for a0 in range(x, x1, 12):
                    yj = y1 + int(rng.integers(-9, 10))
                    col[max(yj - 2, 0):max(yj, 0), a0:min(a0 + 12, x1)] *= 0.78
                    hgt[max(yj - 2, 0):max(yj, 0), a0:min(a0 + 12, x1)] = 0.3
            y = y1
        g = int(rng.integers(2, 5))
        col[:, x1:x1 + g] = kraft * 0.40
        hgt[:, x1:x1 + g] = 0.0
        x = x1 + g
    for y in range(0, Ht, 4):                                                        # ragged: rows drift a pixel or two
        sh = int(rng.integers(-1, 2))
        col[y:y + 4] = np.roll(col[y:y + 4], sh, axis=1)
        hgt[y:y + 4] = np.roll(hgt[y:y + 4], sh, axis=1)
    for _ in range(60):                                                              # torn edges standing proud
        fx, fy = int(rng.integers(0, Wd - 30)), int(rng.integers(0, Ht - 50))
        fw, fh = int(rng.integers(4, 14)), int(rng.integers(14, 46))
        col[fy:fy + fh, fx:fx + fw] = tone(0.08)
        hgt[fy:fy + fh, fx:fx + fw] = 1.0
    # crumple: nothing in a bale runs straight; warp the sheet by a smooth field of a few millimetres
    def smooth(shape, sigma, amp):
        f = np.asarray(Image.fromarray(np.uint8(rng.random(shape) * 255)).filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0
        return (f - f.mean()) / max(float(f.std()), 1e-6) * amp
    yy_, xx_ = np.meshgrid(np.arange(Ht), np.arange(Wd), indexing="ij")
    wx = np.clip(xx_ + smooth((Ht, Wd), 26, 5.0), 0, Wd - 1).astype(int)
    wy = np.clip(yy_ + smooth((Ht, Wd), 18, 7.0), 0, Ht - 1).astype(int)
    col, hgt = col[wy, wx], hgt[wy, wx]
    col *= (0.93 + 0.07 * rng.random((Ht, Wd, 1), dtype=np.float32))
    out["side"] = (col, hgt)
    # --- end: the flat of the last flake: crushed case walls, their folds, the odd print, label or strip of tape
    Wd2 = Ht2 = 1024
    im = Image.new("RGB", (Wd2, Ht2), tuple(int(v) for v in kraft)); d = ImageDraw.Draw(im)
    hm = Image.new("L", (Wd2, Ht2), 128); dh = ImageDraw.Draw(hm)
    for _ in range(46):
        cx, cy = rng.uniform(0, Wd2), rng.uniform(0, Ht2)
        w, h_, a = rng.uniform(200, 560), rng.uniform(140, 420), rng.uniform(-0.35, 0.35) + (math.pi / 2 if rng.random() < 0.3 else 0.0)
        pts = [(cx + ca * w / 2 * math.cos(a) - sa_ * h_ / 2 * math.sin(a), cy + ca * w / 2 * math.sin(a) + sa_ * h_ / 2 * math.cos(a))
               for ca, sa_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        c = tone(0.05)
        d.polygon(pts, fill=tuple(int(v) for v in np.clip(c, 0, 255)), outline=tuple(int(v * 0.80) for v in c))
        dh.polygon(pts, fill=int(rng.uniform(100, 190)))
        r = rng.random()
        if r < 0.10:                                                                 # packing tape
            d.line([pts[0], pts[2]], fill=(148, 128, 96), width=14)
        elif r < 0.18:                                                               # a shipping label
            d.rectangle([cx - 30, cy - 20, cx + 30, cy + 20], fill=(206, 202, 192))
            for k in range(4):
                d.line([cx - 24, cy - 12 + 8 * k, cx + 24 - 6 * k, cy - 12 + 8 * k], fill=(70, 70, 70), width=2)
        elif r < 0.30:                                                               # a line of flexo print
            ink = [(52, 52, 54), (60, 78, 104), (104, 60, 54)][int(rng.integers(0, 3))]
            for k in range(int(rng.integers(1, 3))):
                d.rectangle([cx - 60, cy - 10 + 26 * k, cx + int(rng.integers(0, 60)), cy + 4 + 26 * k], fill=ink)
    for _ in range(36):                                                              # folds
        x0, y0 = rng.uniform(0, Wd2), rng.uniform(0, Ht2)
        a, ln = rng.uniform(-0.3, 0.3) + (math.pi / 2 if rng.random() < 0.5 else 0.0), rng.uniform(80, 320)
        d.line([x0, y0, x0 + ln * math.cos(a), y0 + ln * math.sin(a)], fill=tuple(int(v * 0.84) for v in kraft), width=3)
        dh.line([x0, y0, x0 + ln * math.cos(a), y0 + ln * math.sin(a)], fill=70, width=4)
    col2 = np.asarray(im.filter(ImageFilter.GaussianBlur(0.8)), np.float32) * (0.93 + 0.07 * rng.random((Ht2, Wd2, 1), dtype=np.float32))
    out["end"] = (col2, np.asarray(hm, np.float32) / 255.0)
    paths = {}
    for nm, (c, hmap) in out.items():
        hs = np.asarray(Image.fromarray(np.uint8(hmap * 255)).filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255.0
        gx = np.gradient(hs, axis=1) * 5.0
        gy = np.gradient(hs, axis=0) * 5.0
        nrm = np.stack([-gx, gy, np.ones_like(gx)], 2)
        nrm /= np.linalg.norm(nrm, axis=2)[:, :, None]
        occ = 0.72 + 0.28 * hs[:, :, None]                                           # the gaps are in shadow
        paths[nm] = (save_tex(S, f"bale_{nm}_d.png", Image.fromarray(np.uint8(np.clip(c * occ, 0, 255)))),
                     save_tex(S, f"bale_{nm}_n.png", Image.fromarray(np.uint8((nrm * 0.5 + 0.5) * 255))))
    return paths


def bale_proto(S, size):
    """OCC bale prototype (origin at its centre; x the long, compression axis). The collider's faces are its outermost
    surface: the cardboard bulges to them between the baling wires and is pinched in ~2 cm under each wire, its edges
    are rounded, and the wires lie in the grooves (loops around the long axis, as a baler ties them)."""
    P = proto(S, "bale")
    if has(S, P):
        return P
    S.UsdGeom.Xform.Define(S.stage, P)
    tx = _bale_textures(S)
    side = pbr(S, "v_bale_side", tx["side"][0], tx["side"][1], rough_const=0.95, bump=1.6)
    end = pbr(S, "v_bale_end", tx["end"][0], tx["end"][1], rough_const=0.95, bump=1.2)
    h = np.asarray(size, float) / 2
    hx, hy, hz = h
    rng = np.random.default_rng(5)
    wires_y = np.linspace(-hy, hy, 6)[1:-1] + rng.uniform(-0.015, 0.015, 4)           # four wires across the 30 in width
    pinch, rad = 0.020, 0.045
    Pt, UV, Fs, kinds = [], [], [], []
    n0 = 0
    for axis in range(3):
        o = [i for i in range(3) if i != axis]
        na, nb = int(2 * h[o[0]] / 0.02) + 1, int(2 * h[o[1]] / 0.02) + 1
        A, B = np.meshgrid(np.linspace(-h[o[0]], h[o[0]], na), np.linspace(-h[o[1]], h[o[1]], nb), indexing="ij")
        for sgn in (-1.0, 1.0):
            Q = np.zeros(A.shape + (3,))
            Q[..., o[0]], Q[..., o[1]], Q[..., axis] = A, B, sgn * h[axis]
            # texture: the flakes' strata run across the long axis x on the four long faces; the ends show a flat flake
            uv = np.stack([Q[..., 0] / 1.6 + 0.5, Q[..., 2] / 1.25 + 0.5], 2) if axis == 1 else \
                np.stack([Q[..., 0] / 1.6 + 0.5, Q[..., 1] / 0.8 + 0.5], 2) if axis == 2 else \
                np.stack([Q[..., 1] / 0.8 + 0.5, Q[..., 2] / 1.25 + 0.5], 2)
            i, j = np.meshgrid(np.arange(na - 1), np.arange(nb - 1), indexing="ij")
            a = (i * nb + j).ravel(); b = a + 1; c = a + nb; d = c + 1
            F = np.concatenate([np.stack([a, c, d], 1), np.stack([a, d, b], 1)])
            if (np.cross(o_vec(o[0]), o_vec(o[1]))[axis] * sgn) < 0:
                F = F[:, ::-1]
            Pt.append(Q.reshape(-1, 3)); UV.append(uv.reshape(-1, 2)); Fs.append(F + n0); kinds.append((axis, len(F)))
            n0 += na * nb
    Pt, UV, F = np.concatenate(Pt), np.concatenate(UV), np.concatenate(Fs)
    # rounded box: every point of the sharp box moves onto the box rounded by `rad` (edges and corners stay watertight)
    core = np.clip(Pt, -(h - rad), h - rad)
    nrm = Pt - core
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1)[:, None], 1e-9)
    Pr = core + rad * nrm
    # inward relief (functions of position only, so the faces agree where they meet): the wire grooves across the top,
    # bottom and ends; the flakes' ridges across the long axis; shallow lumps
    dip = pinch * np.exp(-((Pr[:, 1:2] - wires_y) / 0.035) ** 2).sum(1) * (1.0 - nrm[:, 1] ** 2)
    flakes = 0.004 * (0.5 + 0.5 * np.sin(Pr[:, 0] * 95.0 + 3.0 * np.sin(Pr[:, 0] * 13.0))) * (1.0 - nrm[:, 0] ** 2)
    lumps = 0.006 * (0.5 + 0.25 * np.sin(Pr[:, 0] * 5.1 + Pr[:, 2] * 3.0) * np.cos(Pr[:, 1] * 4.3 + 2.0)
                     + 0.25 * np.sin(Pr[:, 0] * 11.0 + Pr[:, 1] * 7.0 + Pr[:, 2] * 9.0))
    Pr = Pr - (dip + flakes + lumps)[:, None] * nrm
    # normals from the welded surface
    _, weld = np.unique(np.round(Pt, 5), axis=0, return_inverse=True)
    weld = weld.ravel()
    fn = np.cross(Pr[F[:, 1]] - Pr[F[:, 0]], Pr[F[:, 2]] - Pr[F[:, 0]])
    acc = np.zeros((weld.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, weld[F[:, k]], fn)
    Nn = acc[weld]
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1)[:, None], 1e-12)
    n_end = sum(n for a, n in kinds if a == 0)
    bs, be = Buf(), Buf()
    be.add(Pr, Nn, UV, F[:n_end])
    bs.add(Pr, Nn, UV, F[n_end:])
    bs.write(S, P + "/sides", side)
    be.write(S, P + "/ends", end)
    wires = Buf()
    wr = 0.0017                                           # 12 gauge baling wire, lying in its groove
    for y in wires_y:
        ex, ez = hx - pinch + wr, hz - pinch + wr
        r = rad
        loop = []
        for cx, cz, a0 in ((ex - r, ez - r, 0.0), (-(ex - r), ez - r, math.pi / 2), (-(ex - r), -(ez - r), math.pi),
                           (ex - r, -(ez - r), 1.5 * math.pi)):
            for a in np.linspace(a0, a0 + math.pi / 2, 6):
                loop.append((cx + r * math.cos(a), y, cz + r * math.sin(a)))
        wires.tube(loop, wr, n=6, closed=True)
        wires.box((0.2 + 0.1 * math.sin(y * 9), y, ez + 0.002), (0.05, 0.006, 0.005), bevel=0.001)   # the twisted tie
    wires.write(S, P + "/wires", mat(S, "steel"))
    return P


# ====================================================================================================== coil
def _coil_end_material(S):
    """End face of a wound coil: the laps seen edge-on (hairlines across u, which the mesh maps to the radius)."""
    name = "v_coil_end"
    if name in S.mats:
        return S.mats[name]
    from PIL import Image
    rng = np.random.default_rng(9)
    Wd = 2048
    base = 0.52 + 0.10 * rng.random(Wd)
    base[rng.random(Wd) < 0.10] *= 0.55                    # a slightly open lap now and then
    base = np.convolve(base, [0.25, 0.5, 0.25], mode="same")
    col = np.stack([base * 0.96, base * 0.98, base * 1.03], 1)
    img = np.tile(np.uint8(np.clip(col, 0, 1) * 255)[None, :, :], (16, 1, 1))
    hgt = np.tile(base[None, :], (16, 1))
    gx = np.gradient(hgt, axis=1) * 3.0
    nrm = np.stack([-gx, np.zeros_like(gx), np.ones_like(gx)], 2)
    nrm /= np.linalg.norm(nrm, axis=2)[:, :, None]
    return pbr(S, name, save_tex(S, "coil_end_d.png", Image.fromarray(img)), save_tex(S, "coil_end_n.png", Image.fromarray(np.uint8((nrm * 0.5 + 0.5) * 255))),
               metal=1.0, rough_const=0.38)


def draw_coil(S, path, r, length, p, q, rng):
    """A slit steel coil lying eye-horizontal: wound strip with its eye, three circumferential straps and three eye
    bands. Drawn in the cylinder collider's frame (axis z), 2 mm inside it so the straps stay within the collider."""
    S.xform(path, p, q)
    ro, ri, hl = r - 0.002, 0.12, length / 2 - 0.001
    ch = 0.004
    body, ends = Buf(), Buf()
    body.cyl((0, 0, 0), ro, 2 * (hl - ch), n=64, caps=False, uvs=1.0)
    body.cyl((0, 0, 0), ri, 2 * hl, n=40, caps=False, inward=True)
    ang = np.linspace(0, 2 * math.pi, 65)
    for s in (-1.0, 1.0):
        # chamfer ring and the end annulus (st: u along the radius, so the lap lines run around the eye)
        body.cyl((0, 0, s * (hl - ch / 2)), ro if s > 0 else ro - ch, ch, n=64, caps=False, r2=ro - ch if s > 0 else ro)
        rr = np.array([ri, ro - ch])
        V = np.stack([np.outer(np.cos(ang), rr).ravel(), np.outer(np.sin(ang), rr).ravel(), np.full(130, s * hl)], 1)
        UV = np.stack([np.tile((rr - ri) / (ro - ri) * 3.0, 65), np.repeat(ang / (2 * math.pi), 2)], 1)
        i = np.arange(64) * 2
        F = np.concatenate([np.stack([i, i + 2, i + 3], 1), np.stack([i, i + 3, i + 1], 1)])
        ends.add(V, np.tile([0.0, 0.0, s], (130, 1)), UV, F if s > 0 else F[:, ::-1])
    body.write(S, path + "/strip", mat(S, "brushed"))
    ends.write(S, path + "/laps", _coil_end_material(S))
    st = Buf()
    for z in (-0.30, 0.0, 0.30):                           # circumferential straps with their seals
        st.cyl((0, 0, z * length / 0.9), ro + 0.0009, 0.019, n=64, caps=False)
        a = float(rng.uniform(0.3, 1.2))
        st.box(((ro + 0.002) * math.cos(a), (ro + 0.002) * math.sin(a), z * length / 0.9), (0.004, 0.03, 0.03), R=rot_z(a), bevel=0.001)
    for a in (math.radians(75), math.radians(195), math.radians(315)):            # eye bands through the bore
        R = rot_z(a)
        w, t = 0.019, 0.0009
        for rad_, sg in ((ro + t / 2, 1), (ri - t / 2, -1)):
            st.box(R @ np.array([rad_, 0.0, 0.0]), (t, w, 2 * hl), R=R, bevel=0.0)
        for s in (-1.0, 1.0):
            st.box(R @ np.array([(ro + ri) / 2, 0.0, s * (hl + t / 2)]), (ro - ri + 2 * t, w, t), R=R, bevel=0.0)
    st.write(S, path + "/straps", mat(S, "strap"))


# ====================================================================================================== draw
def _shape_tf(tf, s):
    return np.asarray(tf[s, :3], float), np.asarray(tf[s, 3:], float)


def draw(S, world):
    import newton
    sc = world.scene
    st = world.model.shape_type.numpy()
    scale = world.model.shape_scale.numpy()
    tf = world.model.shape_transform.numpy()
    labels = world.model.shape_label
    from .. import params as P
    bale_size = np.asarray(P.BALE["size"], float)
    for name in sorted(sc.loads):
        info = sc.loads[name]
        b = int(info["body"])
        root = f"/Lab/body_{b}"
        seed = seed_of("load", name, sc.c.seed)
        rng = np.random.default_rng(seed)
        boards, rest = [], []
        for s in info["shapes"]:
            kind = str(labels[s]).split(":")[-1]
            (boards if kind in ("stringer", "top_deck", "bottom_deck") else rest).append((s, kind))
        if boards:
            parts = [(*_shape_tf(tf, s), box_half(world, s, st, scale)) for s, _ in boards]
            draw_pallet_shapes(S, root, parts, int(rng.integers(0, len(PALLET_WOODS))), rng)
        for s, kind in rest:
            p, q = _shape_tf(tf, s)
            half = box_half(world, s, st, scale)
            material = sc.materials.get(s, "carton")
            if kind == "wrapped" and half is not None:
                base = p - quat_R(q) @ np.array([0.0, 0.0, half[2]])
                unit_load(S, f"{root}/unit_load", 2 * half, seed, p=base, q=q, wrapped=True, label_side=("+x", "-y"))
            elif kind == "coil" and int(st[s]) == int(newton.GeoType.CYLINDER):
                draw_coil(S, f"{root}/coil", float(scale[s, 0]), 2 * float(scale[s, 1]), p, q, rng)
                # timber chocks nailed to the deck either side of the coil (the composite is rigid; they have no
                # collider of their own and sit under the coil's overhang)
                r = float(scale[s, 0])
                th = math.radians(30.0)                    # wedge angle: its face is tangent to the coil 30 deg up
                yt, zt = r * math.sin(th), r * (1.0 - math.cos(th))
                y1, y2 = yt - zt / math.tan(th), 0.775 * r
                z0 = float(p[2]) - r                       # the deck the coil lies on
                ck = Buf()
                for sy in (-1.0, 1.0):
                    poly = [(sy * y1, z0), (sy * y2, z0), (sy * y2, z0 + zt + math.tan(th) * (y2 - yt) - 0.002)]
                    ck.prism(poly if sy > 0 else poly[::-1], -0.40, 0.40, axis=0, uvs=1.0)
                ck.write(S, f"{root}/chocks", mat(S, "wood_mid"))
            elif half is not None and material == "bale":
                if np.allclose(np.sort(2 * half), np.sort(bale_size), atol=1e-3):
                    inst(S, f"{root}/bale", bale_proto(S, 2 * half), p,
                         tuple((np.asarray(q) if rng.random() < 0.5 else _mul_q(q, quat_z(math.pi)))))
                else:
                    bf = Buf(); bf.box(p, 2 * half, R=quat_R(q), bevel=0.02)
                    bf.write(S, f"{root}/bale", mat(S, "card_dark"))
            elif half is not None and material == "dunnage":
                bf = Buf()
                bf.box(p, 2 * half, R=quat_R(q), bevel=0.005, grain=int(np.argmax(half)), uvo=rng.uniform(0, 3, 2))
                bf.write(S, f"{root}/dunnage", mat(S, ("wood_mid", "wood_old", "wood_new")[int(rng.integers(0, 3))]))
            elif half is not None:                        # a loose carton: one carton filling its collider
                kindc = "flat15"
                cp, native = carton_proto(S, kindc, int(seed_of(name.rsplit("_c", 1)[0]) % len(BRANDS)))
                size = 2 * half
                turn = (native[0] >= native[1]) != (size[0] >= size[1])
                qq = _mul_q(q, quat_z(math.pi / 2 + (math.pi if rng.random() < 0.5 else 0.0))) if turn else \
                    _mul_q(q, quat_z(math.pi if rng.random() < 0.5 else 0.0))
                sc_ = (size[1], size[0], size[2]) if turn else tuple(size)
                base = p - quat_R(q) @ np.array([0.0, 0.0, half[2]])
                inst(S, f"{root}/carton", cp, base, qq, np.asarray(sc_) - 0.001)
            elif int(st[s]) == int(newton.GeoType.CYLINDER):
                bf = Buf(); bf.cyl(p, float(scale[s, 0]), 2 * float(scale[s, 1]), R=quat_R(q))
                bf.write(S, f"{root}/load_{s}", mat(S, "steel"))
    flush(S)


def _mul_q(a, b):
    from scipy.spatial.transform import Rotation
    return tuple((Rotation.from_quat(np.asarray(a, float)) * Rotation.from_quat(np.asarray(b, float))).as_quat())

"""USD export of the teaching-kitchen station and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses and the mixture's surface meshes
(reconstructed from the material points by surface.py, with per-vertex colour from each point's composition), so one
exported stage serves a whole session. Interface and lifecycle follow wine_pouring/render.py. Appearance rules:

  * collision pieces are never drawn as such: a vessel's ~150 wall boxes and a tool's cage spheres are hidden
    (material "collision_hidden"); each vessel is its revolved wall polygon (geometry.wall_polygon of the same inner
    profile the physics and the continuum use, at the vessel's true wall thickness `render_wall`), so RTX refracts
    through the real thickness of the glass bowl and reflects in the real curve of the steel one;
  * the whisk's ten wire legs are tubes of the wire's own diameter through the marker points the continuum drags
    (tools.Whisk.markers), closed into five loops over the tip; the mixer's beater strips are flat ribbons through
    theirs; handles, blades and housings are their collision boxes with rounded edges (the radius is appearance
    only: at most 10 mm on the mixer's housing, 1-5 mm elsewhere);
  * robots use the visual meshes and colours of their catalog models (wine_pouring / drilling convention);
  * the mixture is one surface whose colour, gloss and translucency are per-vertex primvars computed from the
    species masses of the points under it (point_look): dry flour matte warm white, sugar brighter, the minor powder
    pale grey, milk/egg/oil off-white to yellow by fat share, batter beige by its wetted-solid fraction with lumps as
    paler specks, cooked starch glossier and clearer, finished pastes their recipe's own colours blended by mass.
    Spilt material (flags 2 on the counter, 3 on the floor) is a volume-true height field where it lies; airborne
    points (flag 1) are droplets of their own volume. Burnt-on patches of the pan bottom (heat.Hob.burnt) are drawn;
  * the room is visual-only dressing (dressing.py): tiled backsplash, cabinet fronts, hob glass print, shelf, sink,
    recipe card, luminaires, the cart's frame and cameras.
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
from scipy.spatial.transform import Rotation

from . import params as P
from . import geometry as GEO
from .materials import AQ, FAT, SUGAR, MINOR, SOLID_D, SOLID_L, SOLID_S, SOLID_C, PASTE_A, PASTE_C

MDL_FILE = Path(__file__).with_name("IngredientMixing.mdl")
# diffuse colour (linear), roughness, metallic. Appearance only (authored); no physics reads these.
COLORS = dict(
    robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    aluminum=((.62, .64, .66), .25, .9),
    frame_white=((.84, .84, .82), .35, 0.), door_leaf=((.50, .58, .60), .45, 0.), vision_glass=((.05, .07, .08), .05, 0.),
    cabinet=((.115, .185, .215), .42, 0.), toe_kick=((.02, .02, .022), .6, 0.),
    cart_plate=((.60, .62, .64), .32, .9), cart_frame=((.035, .037, .040), .42, .35),
    controller=((.030, .031, .034), .45, .30), controller_front=((.055, .057, .060), .35, .55),
    vent_dark=((.008, .008, .009), .8, 0.), caster_tread=((.045, .048, .052), .65, 0.), caster_hub=((.42, .43, .45), .45, .6),
    estop_yellow=((.82, .58, .015), .40, 0.), estop_red=((.62, .015, .012), .32, 0.), cable_black=((.018, .018, .02), .55, 0.),
    steel_brushed=((.74, .74, .73), .26, 1.), bowl_steel=((.72, .72, .71), .40, 1.), steel_wire=((.78, .78, .79), .20, 1.), steel_satin=((.62, .63, .64), .34, 1.),
    chrome=((.86, .86, .87), .08, 1.), stand_steel=((.66, .67, .68), .30, 1.),
    ceramic=((.80, .76, .66), .10, 0.), ceramic_white=((.84, .84, .82), .12, 0.), ceramic_blue=((.10, .19, .30), .12, 0.),
    silicone_mat=((.09, .10, .11), .75, 0.), tool_handle=((.022, .022, .024), .62, 0.), thermo_body=((.72, .08, .05), .45, 0.),
    silicone_red=((.55, .028, .025), .50, 0.), wood=((.60, .43, .25), .62, 0.), cork=((.45, .32, .19), .85, 0.),
    mixer_white=((.83, .83, .81), .32, 0.), mixer_grey=((.20, .21, .22), .45, 0.), knob_black=((.02, .02, .021), .38, 0.),
    knob_mark=((.85, .85, .83), .5, 0.), hob_glass=((.010, .010, .012), .045, 0.), hob_print=((.42, .43, .44), .35, 0.),
    burnt=((.020, .013, .009), .85, 0.), ceiling=((.82, .82, .80), .9, 0.), skirting=((.74, .74, .72), .5, 0.),
    fridge_steel=((.56, .57, .58), .30, 1.), gasket=((.03, .03, .03), .7, 0.), socket_white=((.80, .80, .78), .35, 0.),
    paper=((.88, .87, .82), .75, 0.), dry_goods=((.62, .46, .22), .8, 0.), dry_goods_pale=((.78, .70, .52), .8, 0.),
)
# glass absorption (natural-log coefficients per metre, R G B) and index: soda-lime kitchen glass with its faint green
# cast (about 98.5 / 99.2 / 98.8 % through a 4 mm wall: an estimate from the look of pressed mixing bowls)
GLASS = {"glass": ((3.8, 2.0, 3.0), 1.52)}
HIDDEN = {"collision_hidden"}
# static collision boxes that dressing.py draws in full (fronts, frame, print): the export loop leaves them out
DRESSED = ("counter_top", "cabinet_carcass", "toe_kick", "back_wall", "wall_shelf", "cart_", "hob_glass", "fit_", "fur_", "prop_", "wall_")
# edge radius (m) of a collision box when drawn, by material (appearance only; limited to 45 % of the thinnest side)
EDGE_RADIUS = dict(mixer_white=0.010, tool_handle=0.005, silicone_red=0.004, wood=0.003, mixer_grey=0.0015,
                   camera_dark=0.003, aluminum=0.001, ceramic=0.0006, silicone_mat=0.0008, knob_black=0.002,
                   robot_dark=0.001, stand_steel=0.002)
UNKNOWN = ((.40, .40, .40), .5, 0.)        # a material name this file does not know: neutral grey, listed in export.json
# the mixture's base colours (linear albedo; authored from the look of the ingredients, not measured)
# (Flour is a warm white, sugar and baking powder white, a pancake batter cream.)
# The three white powders are told apart as they are on a bench, by grain and by tone, not by a false colour:
# flour a warm cream, matte; granulated sugar crystals half a millimetre across, a neutral white a shade greyer in
# bulk because light goes into the crystals, with a share of the points catching the light (sugar_glint, glossy);
# baking powder and salt (minor) the whitest and finest, dead matte.
LOOK = dict(flour=(.86, .82, .72), lump=(.90, .87, .78), sugar=(.80, .81, .82), sugar_glint=(1., 1., .99), minor=(.97, .97, .96),
            milk=(.93, .91, .84), oil=(.80, .56, .09), batter=(.82, .69, .44), cooked=(.70, .64, .46),
            char=(.10, .05, .02))
PASTE_GLOSS = {"oat porridge": 0.42, "canned pet food": 0.45, "relish": 0.55}
# How strongly a paste scatters light, against mayonnaise (kubelka_munk). Mayonnaise is a dense emulsion of oil
# droplets a few micrometres across and scatters strongly; ketchup and relish are fruit pulp in a clear serum, honey
# is a clear syrup. 0.1 is an ESTIMATE: arithmetic from droplet and particle sizes gives 0.02 to 0.1 for
# ketchup, and no measurement of either paste's scattering was found. What it is checked
# against is one observation: two parts of mayonnaise with one of ketchup are "pale coral rather than tomato-red"
# (Kelvin's Kitchen, Salsa Rosada, opened) and "an even pink" (The Latin Spoon, Salsa Rosada, opened). With every
# paste scattering alike that blend comes out sRGB (0.82, 0.27, 0.27),
# tomato red; with 0.1 it is (0.92, 0.53, 0.52); with 0.05, (0.94, 0.62, 0.60); with 0.2, (0.90, 0.44, 0.43).
# A paste not named here scatters like mayonnaise: nothing is known of it.
PASTE_SCATTER = {"ketchup": 0.1, "relish": 0.1, "honey": 0.1}


# ====================================================================================== mesh helpers (appearance)
def rounded_box_mesh(size, radius=0.002, n=3):
    """A box with rounded edges and corners: (vertices, triangles, vertex normals). The normals are the exact ones of
    a rounded box (direction from the nearest point of the inner box), so flat faces shade flat and fillets smooth."""
    from scipy.spatial import ConvexHull
    h = np.asarray(size, float)/2
    r = float(min(radius, 0.45*h.min()))
    dirs = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            d = np.array([i, j, n - i - j], float)
            dirs.append(d/np.linalg.norm(d))
    dirs = np.asarray(dirs)
    pts, nrm = [], []
    for sx in (-1., 1.):
        for sy in (-1., 1.):
            for sz in (-1., 1.):
                s = np.array([sx, sy, sz])
                pts.append((h - r)*s + r*dirs*s)
                nrm.append(dirs*s)
    pts, nrm = np.vstack(pts), np.vstack(nrm)
    hull = ConvexHull(pts)
    tri = hull.simplices.copy()
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    tri[flip] = tri[flip][:, ::-1]
    return pts.astype(np.float32), tri.astype(np.int32), nrm.astype(np.float32)


def box_uv(v, t, per_m=1., grain=None):
    """Face-varying box-projection UVs (metres x per_m) for a triangle mesh whose faces are mostly axis-aligned.
    grain: the box axis (0, 1, 2) a texture's V direction (a veneer's grain) must follow on every face that has it."""
    v = np.asarray(v, float)
    t = np.asarray(t, int).reshape(-1, 3)
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    ax = np.argmax(np.abs(fn), axis=1)
    uv = np.zeros((len(t), 3, 2))
    for k, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
        if grain is not None and a == grain:
            a, b = b, a
        m = ax == k
        uv[m, :, 0] = v[t[m]][:, :, a]
        uv[m, :, 1] = v[t[m]][:, :, b]
    return (uv.reshape(-1, 2)*per_m).astype(np.float32)


def tube_mesh(points, radius, sides=8, squash=1.0, normal0=None, cap=True):
    """A tube swept along a polyline (parallel-transport frame): (vertices, triangles, normals). radius: scalar or one
    per point; squash < 1 flattens the section along the frame's second axis (a ribbon); normal0: the first axis at
    the start (the direction the wide side of a ribbon faces is perpendicular to it)."""
    Pp = np.asarray(points, float)
    m = len(Pp)
    rad = np.broadcast_to(np.asarray(radius, float), (m,))
    T = np.gradient(Pp, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    n0 = np.asarray(normal0, float) if normal0 is not None else np.cross(T[0], [0., 0., 1.])
    if np.linalg.norm(n0 - (n0 @ T[0])*T[0]) < 1e-6:
        n0 = np.cross(T[0], [1., 0., 0.])
    N = np.zeros_like(Pp)
    prev = n0
    for i in range(m):
        prev = prev - (prev @ T[i])*T[i]
        prev /= max(np.linalg.norm(prev), 1e-12)
        N[i] = prev
    B = np.cross(T, N)
    ang = np.linspace(0, 2*math.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    v = Pp[:, None, :] + rad[:, None, None]*(ca[None, :, None]*N[:, None, :] + squash*sa[None, :, None]*B[:, None, :])
    nr = ca[None, :, None]*N[:, None, :]*squash + sa[None, :, None]*B[:, None, :]
    nr /= np.maximum(np.linalg.norm(nr, axis=2, keepdims=True), 1e-12)
    v, nr = v.reshape(-1, 3), nr.reshape(-1, 3)
    tri = []
    for i in range(m - 1):
        for s in range(sides):
            a, b_ = i*sides + s, i*sides + (s + 1) % sides
            c, d = a + sides, b_ + sides
            tri += [(a, b_, d), (a, d, c)]
    v, nr = list(v), list(nr)
    if cap:
        for end, sign in ((0, -1.), (m - 1, 1.)):
            base = len(v)
            ring = [np.asarray(v[end*sides + s]) for s in range(sides)]
            v += ring + [Pp[end]]
            nr += [sign*T[end]]*(sides + 1)
            for s in range(sides):
                a, b_ = base + s, base + (s + 1) % sides
                tri.append((a, b_, base + sides) if sign > 0 else (b_, a, base + sides))
    return np.asarray(v, np.float32), np.asarray(tri, np.int32), np.asarray(nr, np.float32)


def revolve_shaded(poly_rz, segments=96, crease_deg=35.):
    """A closed (r, z) polygon revolved about z with crease-aware normals: (vertices, triangles, normals). Edges on
    the axis are skipped; corners sharper than `crease_deg` stay sharp, the rest shade smooth."""
    p = np.asarray(poly_rz, float)
    k = len(p)
    area = 0.5*sum(p[i, 0]*p[(i + 1) % k, 1] - p[(i + 1) % k, 0]*p[i, 1] for i in range(k))
    sgn = 1. if area > 0 else -1.
    d = np.roll(p, -1, axis=0) - p
    ln = np.linalg.norm(d, axis=1)
    ok = (ln > 1e-9) & ~((p[:, 0] < 1e-9) & (np.roll(p, -1, axis=0)[:, 0] < 1e-9))
    sn = np.zeros_like(d)
    sn[ln > 1e-9] = (np.column_stack([d[:, 1], -d[:, 0]])*sgn)[ln > 1e-9]/ln[ln > 1e-9, None]
    cosc = math.cos(math.radians(crease_deg))
    ang = np.linspace(0, 2*math.pi, segments, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    V, Nn, Tt = [], [], []

    def ring(pt, n2):
        V.append(np.column_stack([pt[0]*ca, pt[0]*sa, np.full(segments, pt[1])]))
        Nn.append(np.column_stack([n2[0]*ca, n2[0]*sa, np.full(segments, n2[1])]))

    for i in range(k):
        if not ok[i]:
            continue
        j, h = (i + 1) % k, (i - 1) % k
        na = sn[i] + sn[h] if ok[h] and sn[i] @ sn[h] > cosc else sn[i]
        nb = sn[i] + sn[j] if ok[j] and sn[i] @ sn[j] > cosc else sn[i]
        na, nb = na/np.linalg.norm(na), nb/np.linalg.norm(nb)
        base = sum(len(x) for x in V)
        ring(p[i], na)
        ring(p[j], nb)
        tris = []
        for s in range(segments):
            a0, a1 = base + s, base + (s + 1) % segments
            b0, b1 = a0 + segments, a1 + segments
            if p[i, 0] > 1e-9 and p[j, 0] > 1e-9:
                tris += [(a0, a1, b1), (a0, b1, b0)]
            elif p[i, 0] > 1e-9:                        # the segment ends on the axis: a fan to its end point
                tris.append((a0, a1, b0))
            else:                                       # it starts on the axis: a fan from its start point
                tris.append((a0, b1, b0))
        tris = np.asarray(tris, np.int64)
        vv, nn = np.vstack(V), np.vstack(Nn)
        t0 = tris[0]
        g = np.cross(vv[t0[1]] - vv[t0[0]], vv[t0[2]] - vv[t0[0]])
        if g @ (nn[t0[0]] + nn[t0[1]] + nn[t0[2]]) < 0:
            tris = tris[:, ::-1]
        Tt.append(tris)
    return np.vstack(V).astype(np.float32), np.vstack(Tt).astype(np.int32), np.vstack(Nn).astype(np.float32)


# ====================================================================================== USD authoring
class Stage:
    """Thin USD authoring helper (wine_pouring / drilling Stage plus rounded boxes, tubes and revolved solids)."""

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
        self.unknown = set()
        shutil.copyfile(MDL_FILE, self.folder/MDL_FILE.name)

    # ------------------------------------------------------------------ materials
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

    def mat(self, name):
        """A named constant material of COLORS (made on first use)."""
        if name not in self.mats:
            if name not in COLORS:
                self.unknown.add(name)
            rgb, rough, metal = COLORS.get(name, UNKNOWN)
            self.color(name, rgb, rough, metal)
        return self.mats[name]

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

    def own_mdl(self, name, module, inputs):
        return self.mdl(name, module, inputs, path=str((self.folder/MDL_FILE.name).resolve()))

    def dielectric(self, name, absorption, ior):
        T, Gf = self.Sdf.ValueTypeNames, self.Gf
        return self.own_mdl(name, "IngredientMixingDielectric", {"absorption": (T.Color3f, Gf.Vec3f(*map(float, absorption))),
                                                                  "dielectric_ior": (T.Float, float(ior))})

    def textured(self, name, maps, tile_m, roughness=.6, metal=0., tint=None, bump=.4, rough_influence=.6):
        if name in self.mats:
            return self.mats[name]
        T, Sdf = self.Sdf.ValueTypeNames, self.Sdf
        inputs = {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(maps["diff"]).resolve()))),
                  "reflection_roughness_constant": (T.Float, roughness), "metallic_constant": (T.Float, metal),
                  "texture_scale": (T.Float2, self.Gf.Vec2f(1/tile_m, 1/tile_m))}
        if tint is not None:
            inputs["diffuse_tint"] = (T.Color3f, self.Gf.Vec3f(*map(float, tint)))
        if maps.get("rough"):
            inputs["reflectionroughness_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["rough"]).resolve())))
            inputs["reflection_roughness_texture_influence"] = (T.Float, float(rough_influence))
        if maps.get("nor_gl"):
            inputs["normalmap_texture"] = (T.Asset, Sdf.AssetPath(str(Path(maps["nor_gl"]).resolve())))
            inputs["bump_factor"] = (T.Float, float(bump))
        return self.mdl(name, "OmniPBR", inputs)

    def bind(self, prim, mat):
        self.UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    # ------------------------------------------------------------------ transforms and geometry
    def xform(self, path, p=None, q=None):
        x = self.UsdGeom.Xform.Define(self.stage, path)
        if p is not None:
            x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
            x.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        return x

    def place(self, g, p, q=(0., 0., 0., 1.), scale=None):
        xf = self.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
        xf.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        if scale is not None:
            xf.AddScaleOp().Set(self.Gf.Vec3f(*map(float, scale)))

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, uv_interp="vertex", double=False):
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
            normals = np.asarray(normals, np.float32)
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            g.SetNormalsInterpolation("vertex" if len(normals) == len(v) else "faceVarying")
        elif smooth:
            import trimesh
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)))
            g.SetNormalsInterpolation("vertex")
        if uv is not None:
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, uv_interp).Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        if double:
            g.CreateDoubleSidedAttr(True)
        if mat is not None:
            self.bind(g, mat)
        return g

    def box(self, path, size, mat, p=None, q=(0., 0., 0., 1.), r=0.0015, uv_per_m=None, n=3, grain=None):
        """A box with rounded edges (radius r, appearance only), optionally with box-projection UVs in metres
        (grain: the box axis a wood texture's grain runs along)."""
        v, t, nr = rounded_box_mesh(size, r, n)
        g = self.mesh(path, v, t, normals=nr, mat=mat, uv=box_uv(v, t, uv_per_m, grain) if uv_per_m else None,
                      uv_interp="faceVarying")
        if p is not None:
            self.place(g, p, q)
        return g

    def cylinder(self, path, radius, height, mat, p=None, q=(0., 0., 0., 1.), axis="Z", capsule=False):
        UsdGeom = self.UsdGeom
        g = (UsdGeom.Capsule if capsule else UsdGeom.Cylinder).Define(self.stage, path)
        g.CreateRadiusAttr(float(radius))
        g.CreateHeightAttr(float(height))
        g.CreateAxisAttr(axis)
        if p is not None:
            self.place(g, p, q)
        if mat is not None:
            self.bind(g, mat)
        return g

    def sphere(self, path, radius, mat, p=None):
        g = self.UsdGeom.Sphere.Define(self.stage, path)
        g.CreateRadiusAttr(float(radius))
        if p is not None:
            self.place(g, p)
        if mat is not None:
            self.bind(g, mat)
        return g

    def tube(self, path, points, radius, mat, sides=10, squash=1.0, normal0=None, cap=True):
        v, t, nr = tube_mesh(points, radius, sides, squash, normal0, cap)
        return self.mesh(path, v, t, normals=nr, mat=mat)

    def revolve(self, path, poly_rz, mat, segments=96, crease_deg=35., p=None, q=(0., 0., 0., 1.)):
        v, t, nr = revolve_shaded(poly_rz, segments, crease_deg)
        g = self.mesh(path, v, t, normals=nr, mat=mat)
        if p is not None:
            self.place(g, p, q)
        return g

    def quad(self, path, corners, mat, uv_per_m=1., double=False):
        """A flat quad (corners counter-clockwise seen from its front) with UVs in metres x uv_per_m along its edges."""
        c = np.asarray(corners, float)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        nrm = np.cross(e1, e2)
        nrm /= np.linalg.norm(nrm)
        uv = np.array([[0, 0], [np.linalg.norm(e1), 0], [np.linalg.norm(e1), np.linalg.norm(e2)], [0, np.linalg.norm(e2)]])
        return self.mesh(path, c, [0, 1, 2, 0, 2, 3], uv=uv*uv_per_m, normals=[nrm]*4, mat=mat, double=double)

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), role="diagnostic", aperture=36., aspect=16/9):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*map(float, eye)), self.Gf.Vec3d(*map(float, target)),
                                                            self.Gf.Vec3d(*map(float, up))).GetInverse())
        c.CreateHorizontalApertureAttr(float(aperture))
        c.CreateVerticalApertureAttr(float(aperture/aspect))
        c.CreateFocalLengthAttr(float(focal_mm))
        c.CreateClippingRangeAttr(self.Gf.Vec2f(.005, 40.))
        c.GetPrim().SetCustomDataByKey("observation_role", role)
        return c

    def alias(self, name, mounted):
        a = self.UsdGeom.Xform.Define(self.stage, "/Cameras/" + name)
        a.GetPrim().SetCustomDataByKey("mounted_camera_path", str(mounted))

    def save(self):
        self.stage.GetRootLayer().Save()


# ====================================================================================== export
def _vessel(S, v, name):
    """A vessel as its revolved wall polygon at its true wall thickness; a steel bowl gets its foot (the collision
    base's thickness) and its rolled rim bead."""
    base = f"/Lab/body_{v.body}"
    glass = v.kind == "glass"
    # a mixing bowl or a pan in satin stainless (at the mirror polish of the fittings an empty bowl seen from above read
    # as a chrome dome, its hollow lost in the reflections of the ceiling panels)
    mat = S.mats["glass"] if glass else S.mat("ceramic" if v.kind == "ceramic" else "bowl_steel")
    poly = GEO.wall_polygon(v.profile, v.render_wall)
    g = S.revolve(f"{base}/{name}_wall", poly, mat, segments=144 if glass else 112, crease_deg=50.)
    if glass:
        # clear glass casts almost no shadow in reality (its light is refracted into caustics, which the path tracer
        # does not form): without this the counter under the bowl's foot rendered black (wine_pouring)
        S.UsdGeom.PrimvarsAPI(g).CreatePrimvar("doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    if v.render_wall < v.wall_rigid - 1e-5:
        r0 = float(v.profile[1, 0]) if abs(v.profile[1, 1]) < 1e-6 else float(v.profile[2, 0])
        foot = [(0., -v.wall_rigid), (r0, -v.wall_rigid), (r0 + 0.0015, -v.render_wall + 1e-5), (0., -v.render_wall + 1e-5)]
        S.revolve(f"{base}/{name}_foot", foot, mat, segments=96)
        rb = 0.0022                                           # rolled rim of a spun steel bowl (bead 5 mm in the physics)
        cr, cz = v.rim_r + rb - 0.0002, v.depth - rb
        ring = [(cr + rb*math.cos(a), cz + rb*math.sin(a)) for a in np.linspace(0, 2*math.pi, 16, endpoint=False)]
        S.revolve(f"{base}/{name}_bead", ring, mat, segments=112, crease_deg=80.)
    return g


def _whisk(S, sc, t):
    """Wire loops (tubes of the wire's diameter through the continuum's marker points) and the ferrule."""
    b = sc.b
    mk = t.markers[0]
    x = np.asarray(mk["x"], float)
    legs = 2*int(t.spec["loops"])
    n = len(x)//legs
    wire = S.mat("steel_wire")
    r0, tip = float(t.cage_profile[0, 0]), np.array([0., 0., float(t.cage_profile[-1, 1])])
    for k in range(legs//2):
        a, c = x[k*n:(k + 1)*n], x[(k + legs//2)*n:(k + legs//2 + 1)*n]
        da, dc = a[n//2, :2]/np.linalg.norm(a[n//2, :2]), c[n//2, :2]/np.linalg.norm(c[n//2, :2])
        loop = np.vstack([[r0*da[0], r0*da[1], 0.], a, tip + [0., 0., 0.0007*(k - 2)], c[::-1], [r0*dc[0], r0*dc[1], 0.]])
        S.tube(f"/Lab/body_{mk['body']}/wire_{k}", loop, 0.5*float(mk["dia"][0]), wire, sides=8)
    lab = f"{t.name}_cage_neck"
    if lab in b.shape_label:
        i = list(b.shape_label).index(lab)
        tf, sc_ = np.asarray(b.shape_transform[i], float), np.asarray(b.shape_scale[i], float)
        S.cylinder(f"/Lab/body_{b.shape_body[i]}/ferrule", sc_[0], 2*sc_[1], S.mat("steel_brushed"), p=tf[:3], q=tf[3:],
                   capsule=True)


def _mixer(S, sc, t):
    """Beater strips (flat ribbons through the marker points), their hubs and collars, and the housing's vent slots."""
    s = t.spec
    strip = S.mat("steel_wire")
    z0, z1 = -s["shaft_len"], -s["shaft_len"] - s["beater_len"]
    count = {}
    for mk in t.markers:
        body = mk["body"]
        k = count.get(body, 0)
        count[body] = k + 1
        x = np.asarray(mk["x"], float)
        rad = x[len(x)//2, :2]/np.linalg.norm(x[len(x)//2, :2])
        line = np.vstack([[0.0035*rad[0], 0.0035*rad[1], z0 + 0.001], x, [0.003*rad[0], 0.003*rad[1], z1]])
        # the flat side of a strip faces the beater's axis: the ribbon's thin direction is radial
        S.tube(f"/Lab/body_{body}/strip_{k}", line, 0.5*float(mk["dia"][0]), strip, sides=8, squash=0.22,
               normal0=np.cross([0., 0., 1.], [rad[0], rad[1], 0.]))
    for body in t.beaters:
        S.cylinder(f"/Lab/body_{body}/hub_top", 0.0045, 0.006, strip, p=(0., 0., z0))
        S.cylinder(f"/Lab/body_{body}/hub_tip", 0.004, 0.005, strip, p=(0., 0., z1 + 0.0015))
        S.cylinder(f"/Lab/body_{body}/collar", 0.0065, 0.008, S.mat("mixer_grey"), p=(0., 0., -0.004))
    hx, hy, hz = (0.5*v for v in s["body"])
    vent = S.mat("vent_dark")
    for side in (-1, 1):
        for k in range(6):
            S.box(f"/Lab/body_{t.body}/vent_{'l' if side < 0 else 'r'}{k}", (0.003, 0.0008, 0.034), vent,
                  p=(-hx + 0.022 + 0.008*k, side*(hy + 0.0001), 0.004), r=0.0002)


def _knob(S, t):
    """The hob knob: its cap (hidden collision dodecagon) as a cylinder and a white pointer line on the grip's end."""
    h = P.HOB
    base = f"/Lab/body_{t.body}"
    S.cylinder(f"{base}/cap", h["knob_r"], h["knob_h"], S.mat("knob_black"), p=(0., 0., 0.5*h["knob_h"]))
    S.box(f"{base}/pointer", (0.011, 0.0035, 0.0006), S.mat("knob_mark"), p=(-0.012, 0., h["knob_h"] + 0.0142), r=0.0002)


def authored(sc):
    """Labels of the collision shapes that kitchen.tools draws as one authored shape instead (the wooden spoon's dish
    boxes and its round handle): the export loop leaves them out."""
    out = set()
    for t in sc.tools.values():
        if t.kind == "spoon" and "bowl_depth" in P.SPOON:
            keep = (f"{t.name}_blade", f"{t.name}_rim") + (() if sc.sleeves else (f"{t.name}_handle",))
            out.update(lab for lab in sc.b.shape_label if lab.startswith(keep))
    return out


def export(world, folder, seed=0):
    """Export the world's static appearance + posable bodies to folder/scene.usdc; returns (path, meta)."""
    import newton
    from . import dressing
    sc = world.scene
    b = sc.b
    drawn = authored(sc)
    folder = Path(folder)
    S = Stage(folder)
    for name, (rgb, rough, metal) in COLORS.items():
        S.color(name, rgb, rough, metal)
    for name, (absorb, ior) in GLASS.items():
        S.dielectric(name, absorb, ior)
    q = world.bq
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    V = int(newton.ShapeFlags.VISIBLE)
    labels = list(b.shape_label)
    # ---- robots (visual meshes) and every drawn collision primitive
    for i in range(b.shape_count):
        body, lab = b.shape_body[i], labels[i]
        mname = sc.materials.get(i, "robot")
        if mname in HIDDEN or lab == "floor" or (body < 0 and lab.startswith(DRESSED)) or lab in drawn:
            continue
        flags = int(b.shape_flags[i])
        if mname == "robot" and not flags & V:
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        tf = np.asarray(b.shape_transform[i], float)
        st = b.shape_type[i]
        if mname == "robot":
            # each visual sub-mesh keeps the colour its robot model gives it, linearised from sRGB (wine_pouring)
            col = np.clip(np.asarray(b.shape_color[i], float), 0., 1.)**2.2
            key = "robot_%02x%02x%02x" % tuple(int(round(c*255)) for c in np.asarray(b.shape_color[i], float).clip(0, 1))
            mat = S.color(key, col, .28 if float(col.mean()) > 0.5 else .38, 0.)
        else:
            mat = S.mat(mname)
        if src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True, mat=mat)
            S.place(g, tf[:3], tf[3:])
        elif st == newton.GeoType.BOX:
            S.box(path, 2*scale, mat, p=tf[:3], q=tf[3:], r=EDGE_RADIUS.get(mname, 0.0015))
        elif st in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            S.cylinder(path, scale[0], 2*scale[1], mat, p=tf[:3], q=tf[3:], capsule=st == newton.GeoType.CAPSULE)
        elif st == newton.GeoType.SPHERE:
            S.sphere(path, scale[0], mat, p=tf[:3])
    # ---- vessels, wires, strips, the knob
    meta = dict(vessels={}, mixture={})
    for name, v in sc.vessels.items():
        _vessel(S, v, name)
        meta["vessels"][name] = dict(body=int(v.body), kind=v.kind)
    for name, t in sc.tools.items():
        if t.kind == "whisk":
            _whisk(S, sc, t)
        elif t.kind == "mixer":
            _mixer(S, sc, t)
        elif t.kind == "knob":
            _knob(S, t)
    # ---- the mixture's surfaces (geometry and per-vertex look replaced per request)
    T = S.Sdf.ValueTypeNames
    food = S.own_mdl("food", "IngredientMixingFood", {})
    S.UsdGeom.Scope.Define(S.stage, "/Mixture")
    T_ = S.Sdf.ValueTypeNames
    # vapour: faint shells that give light of their own (render.puffs). Set by looking at the simmering custard from
    # the overhead and the close-up cameras: without light of its own a puff reads as grey smoke against the tiles; at
    # more than 0.06 a shell the food under a simmer's plume is hidden from above. Drawn, not measured.
    steam = S.mdl("steam", "OmniPBR", {"diffuse_color_constant": (T_.Color3f, S.Gf.Vec3f(.95, .95, .95)),
                                       "reflection_roughness_constant": (T_.Float, 1.0),
                                       "enable_opacity": (T_.Bool, True), "opacity_constant": (T_.Float, 0.06),
                                       "enable_emission": (T_.Bool, True), "emissive_color": (T_.Color3f, S.Gf.Vec3f(1., 1., 1.)),
                                       "emissive_intensity": (T_.Float, 150.0)})
    for name, mat in (("food", food), ("spill", food), ("burnt", S.mat("burnt")), ("steam", steam)):
        g = S.mesh(f"/Mixture/{name}", [(0., 0., -50.), (.001, 0., -50.), (0., .001, -50.)], [0, 1, 2], mat=mat)
        pv = S.UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar("displayColor", T.Color3fArray, "vertex").Set(S.Vt.Vec3fArray([S.Gf.Vec3f(.8, .75, .6)]*3))
        pv.CreatePrimvar("gloss", T.FloatArray, "vertex").Set(S.Vt.FloatArray([0.]*3))
        pv.CreatePrimvar("trans", T.FloatArray, "vertex").Set(S.Vt.FloatArray([0.]*3))
        S.UsdGeom.Imageable(g).MakeInvisible()
        meta["mixture"][name] = str(g.GetPath())
    credits = dressing.dress(S, world)
    meta["cameras"] = _cameras(S, sc)
    meta["visual_assets"] = sorted(credits)
    meta["unknown_materials"] = sorted(S.unknown)       # scene materials drawn neutral grey: add them to COLORS
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _cameras(S, sc):
    out = {}
    for name, m in sc.cameras.items():
        aspect = m.get("width", 1280)/m.get("height", 720)
        focal = 18/math.tan(math.radians(m["fov"]/2))
        if m["body"] >= 0:
            g = S.UsdGeom.Camera.Define(S.stage, f"/Lab/body_{m['body']}/{name}")
            g.AddTransformOp().Set(S.Gf.Matrix4d().SetLookAt(S.Gf.Vec3d(*m["eye"]), S.Gf.Vec3d(*m["target"]),
                                                              S.Gf.Vec3d(*m["up"])).GetInverse())
            g.CreateHorizontalApertureAttr(36.)
            g.CreateVerticalApertureAttr(36./aspect)
            g.CreateFocalLengthAttr(focal)
            g.CreateClippingRangeAttr(S.Gf.Vec2f(.005, 30.))
            g.GetPrim().SetCustomDataByKey("observation_role", m["role"])
        else:
            g = S.camera(f"/Cameras/{name}_cam", m["eye"], m["target"], focal, up=m.get("up", (0, 0, 1)), role=m["role"],
                         aspect=aspect)
        S.alias(name, g.GetPath())
        out[name] = dict(role=m["role"], width=m.get("width", 1280), height=m.get("height", 720))
    return out


# ====================================================================================== the mixture's look
def _smooth(a, b, x):
    t = np.clip((x - np.float32(a))*np.float32(1./(b - a)), 0., 1.)
    return t*t*(np.float32(3.) - np.float32(2.)*t)


def _lerp(a, b, t):
    return a + (b - a)*t[:, None]


def paste_colors(recipe):
    """Linear RGB and gloss of the recipe's three finished pastes. Their `color` is read as an sRGB display colour
    (ketchup (0.72, 0.08, 0.05) is ketchup red as sRGB; as a linear albedo it would be a pale salmon)."""
    pastes = recipe["pastes"] if isinstance(recipe, dict) else recipe.pastes
    get = (lambda p, k: p[k]) if isinstance(pastes[0], dict) else (lambda p, k: getattr(p, k))
    rgb = np.array([np.asarray(get(p, "color"), float)**2.2 for p in pastes], np.float32)
    gloss = np.array([PASTE_GLOSS.get(get(p, "name"), 0.75) for p in pastes], np.float32)
    return rgb, gloss


def paste_scatter(recipe):
    """Each of the recipe's three pastes' scattering against mayonnaise (PASTE_SCATTER)."""
    pastes = recipe["pastes"] if isinstance(recipe, dict) else recipe.pastes
    get = (lambda p, k: p[k]) if isinstance(pastes[0], dict) else (lambda p, k: getattr(p, k))
    return np.array([PASTE_SCATTER.get(get(p, "name"), 1.0) for p in pastes], np.float64)


def kubelka_munk(parts, weights, scatter=None):
    """Colour of a blend of opaque pastes: the Kubelka-Munk rule with two constants a paste. Each part has its
    absorption over scattering from its own colour, K/S = (1 - R)^2 / (2 R) per channel, and a scattering strength
    s (`scatter`, one number a part; all 1 when not given). The blend's absorption is the sum of s K/S by share, its
    scattering the sum of s by share, and its reflectance R = 1 + K/S - sqrt((K/S)^2 + 2 K/S) of their ratio.
    (Reflectances averaged by mass make mayonnaise with a third of ketchup beige. With every part scattering alike
    the ketchup tints it tomato red. A real one is pink: PASTE_SCATTER.)
    parts: (m, 3) reflectances; weights: (n, m). Returns (n, 3)."""
    R = np.clip(np.asarray(parts, np.float64), 0.02, 0.995)
    ks = (1.0 - R)**2/(2.0*R)
    w = np.asarray(weights, np.float64)
    if scatter is not None:
        w = w*np.asarray(scatter, np.float64)[None, :]
    mix = (w @ ks)/np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (1.0 + mix - np.sqrt(mix*mix + 2.0*mix)).astype(np.float32)


def point_look(spec, wet, recipe, char=None):
    """Per point: linear RGB (n, 3), gloss (n,) and translucency (n,) from its species masses and saturation.

    Dry look (saturation below ~0.35): the powders by mass. Wet look (above ~0.8): the liquid's colour by its fat
    share, shifted to batter beige by the wetted-solid fraction, to paler specks where the solid is still in lumps
    and to a clearer gel colour by the cooked share. Finished pastes take their recipe colours, blended by mass."""
    spec = np.asarray(spec, np.float32)
    n = len(spec)
    L = {k: np.asarray(v, np.float32) for k, v in LOOK.items()}
    eps = np.float32(1e-30)
    tot = spec.sum(1) + eps
    aq, fat, sugar, minor = spec[:, AQ], spec[:, FAT], spec[:, SUGAR], spec[:, MINOR]
    sd, sl, ss, sk = spec[:, SOLID_D], spec[:, SOLID_L], spec[:, SOLID_S], spec[:, SOLID_C]
    pastes = spec[:, PASTE_A:PASTE_C + 1]
    m_p = pastes.sum(1)
    m_rest = np.maximum(tot - m_p, 0.)
    solids = sd + sl + ss + sk
    powder = solids + sugar + minor
    # sugar: three points in ten are crystals turned to the light (the surface's 1.6-spacing average leaves a coarse sparkle)
    glint = (np.arange(n, dtype=np.uint64)*np.uint64(2246822519) % np.uint64(1009)) < np.uint64(303)
    c_sugar = np.where(glint[:, None], L["sugar_glint"], L["sugar"])
    c_dry = ((sd + ss)[:, None]*L["flour"] + sl[:, None]*L["lump"] + sk[:, None]*L["cooked"] + sugar[:, None]*c_sugar
             + minor[:, None]*L["minor"])/(powder[:, None] + eps)
    liq = aq + fat
    # milk (3.5 % fat) stays white; a beaten blend of milk, egg and oil (about a fifth fat) is custard yellow
    c_liq = _lerp(L["milk"], L["oil"], _smooth(0.0, 0.35, fat/(liq + eps)))
    x_s = solids/(m_rest + eps)
    c_wet = _lerp(c_liq, L["batter"], _smooth(0.02, 0.30, x_s))
    # lumps: every other point of a lumpy region is drawn pale, so lumps read as specks, not as a paler batter
    h = (np.arange(n, dtype=np.uint64)*np.uint64(2654435761) % np.uint64(1000)).astype(np.float32)/np.float32(1000.)
    lump = np.clip(sl/(solids + eps), 0., 1.)*np.where(h > 0.55, 1.0, 0.25)
    c_wet = _lerp(c_wet, L["lump"], np.clip(0.85*lump, 0., 1.))
    cook = np.clip(sk/(solids + eps), 0., 1.)
    c_wet = _lerp(c_wet, L["cooked"], 0.75*cook*_smooth(0.0, 0.03, x_s))
    s = np.where(powder > eps, _smooth(0.35, 0.80, np.asarray(wet, np.float32)), np.float32(1.0))
    c_rest = _lerp(c_dry, c_wet, s)
    p_rgb, p_gloss = paste_colors(recipe)
    if float(m_p.max(initial=0.)) > 0.:
        # the pastes among themselves by the pigment rule; with whatever else is in the point, by mass
        c_paste = kubelka_munk(p_rgb, pastes, paste_scatter(recipe))
        rgb = (m_rest[:, None]*c_rest + m_p[:, None]*c_paste)/tot[:, None]
    else:
        rgb = c_rest*(m_rest/tot)[:, None]
    if char is not None:
        # flecks of burnt-on film scraped off the pan (heat.py): that share of the points is drawn dark brown
        h2 = (np.arange(n, dtype=np.uint64)*np.uint64(40503) % np.uint64(997)).astype(np.float32)/np.float32(997.)
        fleck = (h2 < np.asarray(char, np.float32))[:, None]
        rgb = np.where(fleck, np.asarray(LOOK["char"], np.float32)[None, :], rgb)
    # a liquid without solids is mirror-smooth (1.0); a batter is satin (0.70); cooked starch is glossy again
    g_wet = 1.0 - 0.30*_smooth(0.02, 0.30, x_s)*(1. - cook) - 0.08*cook
    dry_sugar = (1. - s)*sugar/(powder + eps)
    g_rest = s*g_wet + dry_sugar*np.where(glint, np.float32(0.75), np.float32(0.10))   # dry sugar: glinting crystals
    gloss = (m_rest*g_rest + pastes @ p_gloss)/tot
    trans = m_rest*(s*cook*_smooth(0.0, 0.03, x_s) + 0.5*dry_sugar)/tot              # ... and glassy, not chalky
    return np.clip(rgb, 0., 1.).astype(np.float32), np.clip(gloss, 0., 1.).astype(np.float32), \
        np.clip(trans, 0., 1.).astype(np.float32)


def point_spacing(config):
    return float(config.cell_m)/(2 if config.points_per_cell == 8 else 1)


from .materials import ST_CHAR as M_ST_CHAR


def mixture_state(world):
    """The mixture as drawn, for one frame: point positions, flags, look and rest volumes, and the hob's burnt
    patches. None when the world has no mixture. video.Recorder stores exactly this."""
    c = world.cont
    if c is None or not c.n:
        return None
    x, spec, flags, wet, vol = c.read("x", "spec", "flags", "wet", "vol0")
    heat = getattr(world, "heat", None)
    rgb, gloss, trans = point_look(spec, wet, world.recipe, char=None if heat is None else c.read("state")[:, M_ST_CHAR])
    out = dict(x=x.astype(np.float32), flags=flags.astype(np.uint8), rgb=rgb, gloss=gloss, trans=trans,
               vol=vol.astype(np.float32), burnt=None if heat is None else np.asarray(heat.burnt, bool).copy())
    if heat is not None:
        # what leaves the pan as vapour (heat.py): drawn as steam above the food and, at the boil, bubbles on it
        out["vapour"] = np.array([heat.steam_g_s, heat.boiling_w, world.sim_time, heat.z_top + 2.0*c.dx], np.float32)
    return out


_ICO = None


def _icosahedron():
    global _ICO
    if _ICO is None:
        f = (1 + 5**0.5)/2
        v = np.array([(-1, f, 0), (1, f, 0), (-1, -f, 0), (1, -f, 0), (0, -1, f), (0, 1, f), (0, -1, -f), (0, 1, -f),
                      (f, 0, -1), (f, 0, 1), (-f, 0, -1), (-f, 0, 1)], float)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        t = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                      (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
                      (6, 2, 10), (8, 6, 7), (9, 8, 1)], int)
        _ICO = (v, t)
    return _ICO


MESSAGE_LIMIT = 60*1024*1024
DROPLET_BYTES = (12*11 + 20*3)*4*4//3


def _encoded_bytes(m):
    if m is None or not (len(m["v"]) and len(m["t"])):
        return 0
    b64 = lambda n: 4*((n + 2)//3)
    n = b64(12*len(m["v"])) + b64(4*np.size(m["t"]))
    if m.get("n") is not None:
        n += b64(4*np.size(m["n"]))
    if m.get("a") is not None and np.shape(m["a"])[1] >= 5:
        n += b64(12*len(m["a"])) + 2*b64(4*len(m["a"]))
    return n


def _merge_drops(x, vol, attr, max_drops, cell=0.00125):
    while True:
        _, inv = np.unique(np.floor(x/cell).astype(np.int64), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        n = int(inv.max()) + 1
        if n <= max_drops:
            break
        cell *= 2
    w = np.bincount(inv, weights=vol, minlength=n)
    mean = lambda a: np.column_stack([np.bincount(inv, weights=vol*a[:, j], minlength=n) for j in range(a.shape[1])])/w[:, None]
    return mean(x), w, mean(np.asarray(attr, np.float64))


def _droplets(x, vol, attr, max_drops=None):
    """Airborne points as droplets of their own volume (an icosahedron of circumradius r encloses 2.536 r^3). More than
    `max_drops` points are drawn as fewer droplets of the same total volume, merged on a grid made coarser until they
    fit, as a spill too wide is drawn on a coarser grid (_spill_layer)."""
    if max_drops is not None and len(x) > max_drops:
        x, vol, attr = _merge_drops(x, vol, attr, max_drops)
    iv, it = _icosahedron()
    r = np.cbrt(np.maximum(vol, 1e-12)/2.5362)
    n = len(x)
    v = (x[:, None, :] + r[:, None, None]*iv[None]).reshape(-1, 3)
    nr = np.broadcast_to(iv[None], (n, 12, 3)).reshape(-1, 3)
    t = (it[None] + 12*np.arange(n)[:, None, None]).reshape(-1, 3)
    return dict(v=v.astype(np.float32), t=t.astype(np.int32), n=nr.astype(np.float32),
                a=np.repeat(attr, 12, axis=0).astype(np.float32))


def _spill_layer(x, vol, attr, z0, cell=0.000625, lift=4e-5, max_vertices=250_000):
    """Material lying on a flat surface (the continuum freezes landed points at the surface's height) as a volume-true
    height field: each point's rest volume is binned on a `cell` grid and spread over its neighbourhood (a binomial
    blur of 1.4 cells' standard deviation, 0.9 mm), so a single 2 mm3 point is a round speck 0.4 mm high and 4 mm
    across and a thousand in one place are a mound of their volume. A spill too wide for `max_vertices` at that cell
    is drawn on a coarser grid (cell doubled until it fits)."""
    while True:
        m = _spill_grid(x, vol, attr, z0, cell, lift)
        if m is None or len(m["v"]) <= max_vertices:
            return m
        cell *= 2


SPILL_DEPTH = 0.006      # m: deepest a spill is drawn; what is piled higher is spread out round it (see _spill_grid)


def _spill_grid(x, vol, attr, z0, cell, lift):
    # room round the points for a pile to be spread into: half the side of a square SPILL_DEPTH deep of all of it
    pad = 10 + int(0.5*math.sqrt(float(vol.sum())/SPILL_DEPTH)/cell)
    while True:
        ij = np.floor(x[:, :2]/cell).astype(np.int64)
        lo = ij.min(0) - pad
        shape = ij.max(0) - lo + pad + 1
        if shape[0]*shape[1] <= 8_000_000:
            break
        cell *= 2
    flat = (ij[:, 0] - lo[0])*shape[1] + (ij[:, 1] - lo[1])
    size = int(shape[0]*shape[1])

    def blur(a, passes=2):
        # a round kernel (a Gaussian of 1.4 cells for the two passes, 1.0 for one, cut at four deviations). A
        # binomial 1-4-6-4-1 along each axis ends on a square: a single speck's outline, drawn where its height falls
        # under 12 micrometres, follows that square, and spilt flour lies on the counter as pale square flecks
        from scipy.ndimage import gaussian_filter as _gf
        return _gf(a.reshape(shape).astype(np.float64), 1.4 if passes >= 2 else 1.0, mode="constant", truncate=4.0)

    V = blur(np.bincount(flat, weights=vol, minlength=size))
    A = [blur(np.bincount(flat, weights=vol*attr[:, k], minlength=size)) for k in range(attr.shape[1])]
    # The continuum freezes a spilt point where it lands, and what flows out along the counter lands where it leaves
    # the grid: half a bowl of milk poured out is a line of points along the grid's edge, and drawn as the mound of
    # its volume it would stand there as a wall 265 mm high. A spill is
    # therefore drawn no deeper than SPILL_DEPTH: what is piled higher is spread out round the pile, volume and
    # colour kept. This is a drawing of where a puddle of that volume would lie on a bare flat counter, not a
    # simulation of it: the spilt points themselves stay where they froze and nothing touches them.
    v_cap = SPILL_DEPTH*cell**2
    if V.max() > v_cap:
        # spread on a grid four times coarser (a pour of 0.3 l needs a blur 126 mm wide: on the fine grid that cost
        # tens of seconds a frame and all but stopped the recording of the first tilt run that used it); the spread
        # part takes the mean colour of what was taken off the piles
        from scipy.ndimage import gaussian_filter
        f_ = 4
        s0, s1 = (-shape[0]) % f_, (-shape[1]) % f_
        low = np.minimum(V, v_cap)
        ex = np.pad(V - low, ((0, s0), (0, s1)))
        room = np.pad(v_cap - low, ((0, s0), (0, s1)))
        blk = lambda a_: a_.reshape(a_.shape[0]//f_, f_, a_.shape[1]//f_, f_).sum((1, 3))
        ec, rc = blk(ex), blk(room)
        placed = np.zeros_like(ec)
        sigma = max(1.0, 0.5*math.sqrt(float(ec.sum())/(math.pi*v_cap*f_*f_)))
        for _ in range(12):
            if ec.sum() <= 1e-3*float(V.sum()):
                break
            g_ = gaussian_filter(ec, sigma, mode="constant", truncate=2.5)
            g_ *= ec.sum()/max(float(g_.sum()), 1e-30)            # what the border cut off is not lost
            take = np.minimum(g_, np.maximum(rc - placed, 0.))
            placed += take
            ec = g_ - take
            sigma *= 1.3
        # within a coarse cell, to the fine cells in proportion to the room each has; what found no room after twelve
        # passes stays on top, evenly
        big = lambda a_: np.repeat(np.repeat(a_, f_, 0), f_, 1)
        up = (big(placed/np.maximum(rc, 1e-30))*room + big(ec/(f_*f_)))[:shape[0], :shape[1]]
        taken = V - low
        tot = max(float(taken.sum()), 1e-30)
        A = [Ak*(low/np.maximum(V, 1e-30)) + up*float((Ak*(taken/np.maximum(V, 1e-30))).sum())/tot for Ak in A]
        V = low + up
    H = V/cell**2
    live = H > 1.2e-5
    if not live.any():
        return None
    grow = live.copy()
    for sx_, sy_ in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        grow |= np.roll(np.roll(live, sx_, 0), sy_, 1)
    idx = -np.ones(shape, np.int64)
    ii, jj = np.nonzero(grow)
    idx[ii, jj] = np.arange(len(ii))
    Hs = np.where(live, H, 0.)
    v = np.column_stack([(ii + lo[0] + 0.5)*cell, (jj + lo[1] + 0.5)*cell, z0 + lift + Hs[ii, jj]])
    gx, gy = np.gradient(Hs, cell)
    nr = np.column_stack([-gx[ii, jj], -gy[ii, jj], np.ones(len(ii))])
    nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    # attributes: where a skirt vertex holds no volume of its own, the mean of its neighbours that do
    V2 = np.maximum(blur(V, 1), 1e-30)
    a = np.column_stack([np.where(V > 1e-18, Ak/np.maximum(V, 1e-30), blur(Ak, 1)/V2)[ii, jj] for Ak in A])
    q00, q10, q11, q01 = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    okq = (q00 >= 0) & (q10 >= 0) & (q11 >= 0) & (q01 >= 0) & (live[:-1, :-1] | live[1:, :-1] | live[1:, 1:] | live[:-1, 1:])
    a0, a1, a2, a3 = q00[okq], q10[okq], q11[okq], q01[okq]
    t = np.concatenate([np.column_stack([a0, a1, a2]), np.column_stack([a0, a2, a3])])
    return dict(v=v.astype(np.float32), t=t.astype(np.int32), n=nr.astype(np.float32), a=a.astype(np.float32))


def _merge(parts):
    parts = [p for p in parts if p is not None and len(p["t"])]
    if not parts:
        return None
    off, ts = 0, []
    for p in parts:
        ts.append(p["t"] + off)
        off += len(p["v"])
    return dict(v=np.concatenate([p["v"] for p in parts]), t=np.concatenate(ts).astype(np.int32),
                n=np.concatenate([p["n"] for p in parts]), a=np.concatenate([p["a"] for p in parts]))


def burnt_mesh(burnt, pan_q, radius, lift=0.0004):
    """The hob's burnt-on patches (heat.Hob: N_RING rings x N_SECTOR sectors of the pan bottom) as flat annular
    sectors just above the pan's inside bottom, in world coordinates."""
    from .heat import N_RING, N_SECTOR
    burnt = np.asarray(burnt, bool).reshape(N_RING, N_SECTOR)
    if not burnt.any():
        return None
    edges = np.linspace(0, radius, N_RING + 1)
    sub = 4
    v, t = [], []
    for i, j in zip(*np.nonzero(burnt)):
        a = 2*math.pi*(j + np.linspace(0, 1, sub + 1))/N_SECTOR
        base = len(v)
        for r in (edges[i], edges[i + 1]):
            v += [(r*math.cos(x), r*math.sin(x), lift) for x in a]
        for k in range(sub):
            p0, p1, p2, p3 = base + k, base + k + 1, base + sub + 1 + k + 1, base + sub + 1 + k
            t += [(p0, p3, p2), (p0, p2, p1)]
    R = Rotation.from_quat(pan_q[3:])
    vw = pan_q[:3] + R.apply(np.asarray(v))
    nr = np.tile(R.apply([0., 0., 1.]), (len(vw), 1))
    return dict(v=vw.astype(np.float32), t=np.asarray(t, np.int32), n=nr.astype(np.float32), a=np.zeros((len(vw), 5), np.float32))


def _unit(k, salt):
    """A fixed pseudo-random number in [0, 1) for index k (the same in every frame)."""
    return ((np.asarray(k, np.uint64) + np.uint64(salt))*np.uint64(2654435761) % np.uint64(100003)).astype(np.float64)/100003.0


_BALL = None


def _ball(levels=2):
    """A sphere of unit radius: an icosahedron with every face divided twice (162 vertices, 320 triangles)."""
    global _BALL
    if _BALL is None:
        v, t = _icosahedron()
        v, t = [tuple(p) for p in v], [tuple(f) for f in t]
        for _ in range(levels):
            mid, nt = {}, []

            def m(i, j):
                key = (min(i, j), max(i, j))
                if key not in mid:
                    q = np.add(v[i], v[j])
                    mid[key] = len(v)
                    v.append(tuple(q/np.linalg.norm(q)))
                return mid[key]
            for i, j, k in t:
                a_, b_, c_ = m(i, j), m(j, k), m(k, i)
                nt += [(i, a_, c_), (j, b_, a_), (k, c_, b_), (a_, b_, c_)]
            t = nt
        _BALL = (np.array(v, float), np.array(t, int))
    return _BALL


SHELLS = (0.35, 0.55, 0.78, 1.0)             # a puff's shells, as shares of its radius


def puffs(centre, size):
    """Puffs of vapour as nested faint shells: a line of sight through the middle of a puff crosses eight surfaces and
    one near its edge two, so a puff is densest in the middle and has no outline."""
    bv, bt = _ball()
    m = len(bv)
    rad = (np.asarray(size, float)[:, None]*np.asarray(SHELLS)[None, :]).reshape(-1)
    cen = np.repeat(np.asarray(centre, float), len(SHELLS), axis=0)
    v = (cen[:, None, :] + rad[:, None, None]*bv[None]).reshape(-1, 3)
    nr = np.broadcast_to(bv[None], (len(cen), m, 3)).reshape(-1, 3)
    tr = (bt[None] + m*np.arange(len(cen))[:, None, None]).reshape(-1, 3)
    return dict(v=v.astype(np.float32), t=tr.astype(np.int32), n=nr.astype(np.float32), a=np.ones((len(v), 5), np.float32))


def puffs_of_a_recording(mesh):
    """A recording made before the puffs were drawn as shells holds each as a twenty-sided solid (12 vertices, 20
    triangles): the same puffs, at the same places and of the same sizes, as shells. Any other mesh is returned as
    it is."""
    v = np.asarray(mesh["v"], float)
    if len(v) == 0 or len(v) % 12 or len(mesh["t"]) != 20*(len(v)//12):
        return mesh
    g = v.reshape(-1, 12, 3)
    centre = g.mean(1)
    return puffs(centre, np.linalg.norm(g - centre[:, None, :], axis=2).mean(1))


def steam_mesh(vapour, pan_q, radius):
    """Steam over a pan: puffs rising from the food's surface, as many as the vapour leaving it says (one for every
    4 mg/s, up to 36: a few wisps from a sauce at 60 C, a steady plume at the boil), each growing and thinning as
    it rises 0.25 m in two seconds. Drawn, not simulated: its amount is the hob model's (heat.Hob.steam_g_s)."""
    rate, _, t, z_top = (float(v) for v in vapour)
    n = int(min(36, rate/0.004))
    if n <= 0:
        return None
    k = np.arange(n)
    age = (t/2.0 + _unit(k, 11)) % 1.0                         # 0 at the surface, 1 where it has thinned away
    r0 = 0.7*radius*np.sqrt(_unit(k, 23))
    a0 = 2*math.pi*_unit(k, 37)
    sway = 0.012*np.sin(2*math.pi*(age*1.5 + _unit(k, 53)))
    local = np.column_stack([r0*np.cos(a0) + sway, r0*np.sin(a0) + 0.5*sway, z_top + 0.01 + 0.25*age])
    size = (0.010 + 0.030*age)*(1.0 - 0.6*age**3)
    R = Rotation.from_quat(pan_q[3:])
    centre = pan_q[:3] + R.apply(local)
    centre[:, 2] = pan_q[2] + local[:, 2]                       # it rises plumb, whatever the pan's tilt
    return puffs(centre, size)


def bubble_mesh(vapour, pan_q, radius, attr):
    """A simmer: bubbles breaking the food's surface, as many as the heat going into vapour inside the food says
    (one for every 25 W, up to 40), each lasting a third of a second. Drawn on the surface, not simulated."""
    _, boil, t, z_top = (float(v) for v in vapour)
    n = int(min(40, boil/25.0))
    if n <= 0:
        return None
    iv, it = _icosahedron()
    k = np.arange(n)
    life = (t*3.0 + _unit(k, 71)) % 1.0
    gen = np.floor(t*3.0 + _unit(k, 71)).astype(np.uint64)      # a new place each time a bubble is born
    r0 = 0.85*radius*np.sqrt(_unit(k + 97*gen, 5))
    a0 = 2*math.pi*_unit(k + 131*gen, 17)
    size = 0.0035 + 0.006*np.sin(math.pi*life)
    R = Rotation.from_quat(pan_q[3:])
    centre = pan_q[:3] + R.apply(np.column_stack([r0*np.cos(a0), r0*np.sin(a0), np.full(n, z_top - 0.002)]))
    v = (centre[:, None, :] + size[:, None, None]*iv[None]).reshape(-1, 3)
    nr = np.broadcast_to(iv[None], (n, 12, 3)).reshape(-1, 3)
    tr = (it[None] + 12*np.arange(n)[:, None, None]).reshape(-1, 3)
    a = np.tile(np.asarray(attr, np.float32)[None, :], (len(v), 1))
    return dict(v=v.astype(np.float32), t=tr.astype(np.int32), n=nr.astype(np.float32), a=a)


def meshes_from_state(state, world, body_q=None, cell=None):
    """{name: dict(v, t, n, a)} for the worker from a mixture_state: 'food' (the surface of the points in the grid),
    'spill' (landed and airborne points), 'burnt' (the pan's burnt-on patches). a = (r, g, b, gloss, trans) per
    vertex. Timings of the last call are in material_meshes.timing."""
    from . import surface
    out = {}
    timing = material_meshes.timing = {}
    if state is None:
        return out
    t0 = time.perf_counter()
    sc = world.scene
    spacing = point_spacing(world.config)
    x, flags = state["x"], state["flags"]
    attr = np.column_stack([state["rgb"], state["gloss"], state["trans"]]).astype(np.float32)
    vol = state["vol"] if state.get("vol") is not None else np.full(len(x), spacing**3, np.float32)
    m = flags == 0
    if m.any():
        # how far each point is loose dry powder, from its look: a dry powder has next to no gloss (point_look; a dry
        # mix with a tenth of sugar in it reads 0.01, a batter 0.7, a liquid 1)
        powder = np.clip(1.0 - attr[m][:, 3]/0.15, 0.0, 1.0)
        mesh = surface.reconstruct(x[m], spacing, world.config.device, cell=cell, attrs=attr[m], powder=powder)
        timing["surface"] = dict(surface.TIMING)
        if len(mesh["t"]):
            out["food"] = mesh
    if state.get("burnt") is not None and np.any(state["burnt"]):
        bq = world.bq if body_q is None else np.asarray(body_q, float)
        b = burnt_mesh(state["burnt"], bq[sc.vessel.body], P.PAN["radius"])
        if b is not None:
            out["burnt"] = b
    if state.get("vapour") is not None:
        bq = world.bq if body_q is None else np.asarray(body_q, float)
        st = steam_mesh(state["vapour"], bq[sc.vessel.body], P.PAN["radius"])
        if st is not None:
            out["steam"] = st
        if "food" in out and m.any():
            top = attr[m][np.argsort(x[m][:, 2])[-200:]].mean(0)          # the look of the food's top
            top[3], top[4] = 1.0, max(float(top[4]), 0.25)                # a bubble's skin is wet and a little clear
            bub = bubble_mesh(state["vapour"], bq[sc.vessel.body], P.PAN["radius"], top)
            if bub is not None:
                out["food"] = _merge([out["food"], bub])
    parts = []
    plane = sc.counter_plane
    for flag, z in ((2, plane["z"]), (3, plane.get("floor", 0.))):
        k = flags == flag
        if k.any():
            parts.append(_spill_layer(x[k].astype(np.float64), vol[k].astype(np.float64), attr[k].astype(np.float64), z))
    k = (flags == 1) | (flags == 5)                 # drops in flight, and the film on wires and beater strips
    if k.any():
        room = MESSAGE_LIMIT - sum(_encoded_bytes(p) for p in (*out.values(), *parts))
        parts.append(_droplets(x[k].astype(np.float64), vol[k].astype(np.float64), attr[k],
                               max_drops=max(1, room//DROPLET_BYTES)))
    spill = _merge(parts)
    if spill is not None:
        out["spill"] = spill
    out = {name: out[name] for name in ("food", "spill", "burnt", "steam") if name in out}
    timing["total_s"] = time.perf_counter() - t0
    timing["points"] = int(len(x))
    timing["triangles"] = {k_: int(len(v_["t"])) for k_, v_ in out.items()}
    return out


def material_meshes(world, cell=None):
    """The mixture's surface meshes with per-vertex look for the world's present state (wine's liquid_meshes)."""
    return meshes_from_state(mixture_state(world), world, cell=cell)


material_meshes.timing = {}


def encode_meshes(meshes):
    out = {}
    b64 = lambda a, dt: base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()
    for name, m in meshes.items():
        if m is None:
            continue
        if not isinstance(m, dict):
            m = dict(zip("vtna", m))
        if not (len(m["v"]) and len(m["t"])):
            continue
        e = dict(v=b64(m["v"], np.float32), t=b64(m["t"], np.int32))
        if m.get("n") is not None:
            e["n"] = b64(m["n"], np.float32)
        if m.get("a") is not None and np.shape(m["a"])[1] >= 5:
            a = np.asarray(m["a"], np.float32)
            e["c"] = b64(a[:, :3], np.float32)
            e["g"] = b64(a[:, 3], np.float32)
            e["x"] = b64(a[:, 4], np.float32)
        out[name] = e
    size = sum(len(v) for e in out.values() for v in e.values())
    if size > MESSAGE_LIMIT:
        raise ValueError(f"mixture meshes of {size >> 20} MiB exceed the renderer's 64 MiB message: coarsen the field "
                         "(material_meshes(world, cell=0.0025))")
    return out


# ====================================================================================== worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py). width/height None: every camera renders at its
    own resolution (scene.cameras[name]['width'/'height'])."""

    def __init__(self, world, folder, scene, *, width=None, height=None, samples=24):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        self.last_timings = {}
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, MIX_RENDER_PORT=str(listener.getsockname()[1]), MIX_RENDER_SECRET=secret)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "ingredient_mixing.render_worker", "--scene",
                                             str(Path(scene).resolve()), "--width", str(width or 1280),
                                             "--height", str(height or 720), "--samples", str(samples)],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
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

    def observe(self, cameras, body_q, tick, meshes=None, resolution=None, visibility=None, resolutions=None,
                subframes=None):
        """Render `cameras` for the body poses `body_q` and the mixture `meshes` (material_meshes; None: no mixture
        drawn). Returns {camera: PNG bytes}. resolution: one (w, h) for every camera of this request; resolutions:
        {camera: (w, h)}; otherwise the renderer's width/height, otherwise each camera's own."""
        cams = self.world.scene.cameras
        res = {}
        for c in cameras:
            cam = cams.get(c, {})
            res[c] = list((resolutions or {}).get(c) or resolution
                          or (self.width or cam.get("width", 1280), self.height or cam.get("height", 720)))
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras),
                   meshes=encode_meshes(meshes or {}), visibility=visibility or {}, resolutions=res)
        if subframes:
            msg["subframes"] = int(subframes)
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def close(self, graceful=True):
        """End the worker and reap it. graceful=False is for a worker that cannot be asked (its process has exited, or
        it failed a frame and the state of its socket is unknown): no close request is sent and nothing waits for one,
        the process is sent TERM, then KILL (render_lifecycle.stop_worker)."""
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self, graceful=graceful)
            self.closed = True


class Persistent:
    """A Renderer for long offline jobs (films, stills) on a GPU shared with other sessions: when the worker dies under
    a request (the card ran out of memory because another session started a worker of its own), a new worker is
    started after a wait and the request is made again. `restarts` counts them; after `tries` failures in a row the
    error is raised. Policy observations do not use this: a session's backend reports the failure."""

    def __init__(self, world, folder, scene, tries=8, wait_s=45.0, **kw):
        self._args, self._kw = (world, folder, scene), kw
        self.tries, self.wait_s, self.restarts = tries, wait_s, 0
        self._r = self._start()

    def _start(self):
        last = None
        for k in range(self.tries):
            try:
                return Renderer(*self._args, **self._kw)
            except (ConnectionError, RuntimeError, TimeoutError, OSError) as e:
                last = e
                print(f"render worker did not start ({type(e).__name__}: {e}); waiting {self.wait_s:g} s", flush=True)
                time.sleep(self.wait_s)
        raise last

    def observe(self, *a, **kw):
        for k in range(self.tries):
            try:
                return self._r.observe(*a, **kw)
            except (ConnectionError, RuntimeError, TimeoutError, OSError) as e:
                self.restarts += 1
                print(f"render worker lost ({type(e).__name__}: {e}); restart {self.restarts} after {self.wait_s:g} s", flush=True)
                try:
                    self._r.close()
                except Exception:
                    pass
                time.sleep(self.wait_s)
                self._r = self._start()
        raise RuntimeError(f"the render worker was lost {self.tries} times on one request")

    @property
    def last_timings(self):
        return self._r.last_timings

    def close(self):
        self._r.close()

"""USD export of the skills-laboratory bay and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses and the liquid's meshes built from
the simulator's state (liquid_meshes), so one exported stage serves a whole session. Interface and lifecycle follow
ingredient_mixing/render.py (which follows wine_pouring's). Appearance rules:

  * a syringe's barrel is its revolved wall (the bore and the outside diameter of the ledger) in a clear dielectric,
    with its printed scale at the derived pitch; the stopper, the cross-shaped rod and the thumb rest are drawn on the
    plunger's body, so where the stopper stands against the scale is where the physics has it;
  * the liquid is drawn from the state and nothing else: in each barrel the column between the nozzle and the stopper,
    cut by a level surface so that the gas is where gravity puts it (in the 1 mL barrel, where the model's bubble does
    not rise by itself, a slug where the model has it); in the vial the same cut of its inside; in the extension tube
    the parcels of its queue in order, each in its liquid's colour; drops in flight as spheres of their volume, a
    hanging drop at the outlet it hangs from, and what has landed as puddles on the body it landed on;
  * every liquid is a dielectric with its own absorption: saline clear, the practice medication blue, the contrast
    simulant amber, the trainer's blood red;
  * collision pieces that are drawn as something truer are hidden (the barrel's solid cylinder, a sheath's two boxes,
    the vial's solid glass, the pad's block, the training arm's hand block);
  * robots use the visual meshes and colours of their catalogue models;
  * the room is visual-only dressing (dressing.py) over the collision boxes of scene.room.
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
from . import hydraulics as H

MDL_FILE = Path(__file__).with_name("SyringeHandling.mdl")
# diffuse colour (linear), roughness, metallic. Appearance only (authored); no physics reads these.
COLORS = dict(
    robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    aluminum=((.62, .64, .66), .25, .9), steel_brushed=((.74, .74, .73), .26, 1.), chrome=((.86, .86, .87), .08, 1.),
    steel_needle=((.80, .80, .81), .18, 1.), cart_plate=((.60, .62, .64), .32, .9), cart_frame=((.035, .037, .040), .42, .35),
    controller=((.030, .031, .034), .45, .30), toe_kick=((.02, .02, .022), .6, 0.),
    solid_surface=((.80, .80, .77), .28, 0.), casework=((.62, .64, .64), .40, 0.), casework_front=((.20, .33, .41), .36, 0.),
    wall_panel=((.33, .47, .43), .32, 0.), wall_paint=((.66, .62, .54), .85, 0.), ceiling=((.82, .82, .80), .9, 0.),
    floor_vinyl=((.33, .40, .42), .42, 0.), skirting=((.27, .33, .35), .45, 0.),
    trunking=((.80, .81, .80), .35, 0.), outlet_white=((.85, .85, .83), .3, 0.), outlet_black=((.03, .03, .03), .35, 0.),
    outlet_yellow=((.80, .62, .05), .35, 0.), socket_white=((.80, .80, .78), .35, 0.),
    ceramic=((.82, .82, .80), .08, 0.), sink_dark=((.25, .26, .27), .2, 0.),
    tray=((.025, .12, .36), .34, 0.), basin=((.72, .73, .73), .22, 1.), wipes_box=((.80, .82, .84), .5, 0.), gauze=((.86, .86, .83), .95, 0.),
    sharps_red=((.62, .03, .025), .38, 0.), sharps_flap=((.70, .70, .68), .35, 0.), mat=((.035, .04, .05), .8, 0.), castor=((.05, .05, .055), .5, 0.), led_green=((.05, .6, .1), .3, 0.), cable=((.02, .02, .022), .55, 0.), bolt=((.55, .56, .57), .3, 1.),
    pad_skin=((.76, .52, .40), .55, 0.), pad_fat=((.86, .72, .36), .6, 0.), pad_muscle=((.52, .10, .09), .5, 0.), pad_board=((.20, .20, .21), .6, 0.),
    pad_ring=((.03, .05, .25), .5, 0.), arm_skin=((.74, .50, .38), .50, 0.), arm_board=((.10, .22, .42), .85, 0.), arm_model=((.72, .50, .38), .5, 0.),
    syringe_rod=((.84, .85, .84), .35, 0.), stopper_rubber=((.025, .025, .028), .55, 0.), stopper_grey=((.30, .31, .33), .5, 0.),
    tip_cap=((.80, .81, .80), .4, 0.), vial_cap=((.72, .73, .74), .30, 1.), vial_target=((.35, .36, .38), .6, 0.),
    adapter=((.82, .84, .86), .35, 0.), connector=((.84, .86, .88), .30, 0.), connector_ring=((.04, .18, .62), .35, 0.),
    clamp=((.86, .86, .84), .35, 0.), tape=((.85, .82, .74), .7, 0.), dressing_film=((.80, .82, .82), .15, 0.),
    stand_base=((.08, .08, .09), .5, 0.), bed=((.70, .72, .74), .5, 0.), bed_frame=((.55, .57, .58), .35, .6), linen=((.80, .82, .84), .9, 0.),
    blanket=((.28, .45, .55), .95, 0.), manikin=((.70, .50, .40), .55, 0.), curtain=((.42, .55, .58), .9, 0.), curtain_track=((.75, .75, .74), .35, .5),
    paper=((.88, .87, .82), .75, 0.), glove_blue=((.20, .32, .62), .6, 0.), glove_box=((.82, .83, .84), .6, 0.),
    dispenser=((.84, .84, .83), .30, 0.), handrub=((.55, .68, .78), .25, 0.), led_panel=((.9, .9, .9), .6, 0.),
    lab_far=((.62, .64, .62), .7, 0.), window_frame=((.80, .80, .78), .4, 0.), chair=((.10, .17, .30), .6, 0.),
    saline_bag=((.80, .84, .86), .2, 0.), print_dark=((.02, .02, .03), .5, 0.), clock_red=((.72, .05, .04), .45, 0.),
)
for _k, _n in P.NEEDLES.items():                       # hub colours by gauge (ISO 6009) and their sheaths
    COLORS["hub_" + _k] = (tuple(c**2.2 for c in _n["colour"]), .40, 0.)
# dielectrics: absorption (natural-log coefficients per metre, R G B) and index
GLASS = {
    "barrel_clear": ((6., 5., 5.), 1.49),               # polypropylene barrel: clear, a trace of haze taken as absorption
    "glass": ((3.0, 1.6, 2.4), 1.52),                   # type I vial glass
    "tube_clear": ((4., 3., 3.), 1.47),                 # PVC extension tubing
    "liquid_saline": ((0.6, 0.3, 0.2), 1.333),
    "liquid_medication": ((320., 110., 12.), 1.333),    # the practice medication: water with blue dye (authored so cameras see it)
    "liquid_contrast": ((14., 60., 300.), 1.44),        # iohexol: pale amber (the dye is authored; the index is the real solution's order)
    "liquid_blood": ((90., 1300., 1500.), 1.35),        # trainer's blood: a red dye
}
LIQUID_NAMES = P.SPECIES                                # one dynamic mesh per liquid, in /Liquid
CLOCK_NAMES = (("clock_hands", "print_dark"), ("clock_second", "clock_red"))      # the wall clock's hands: /Liquid too
CLOCK_START_S = 10*3600 + 10*60                         # the wall clock reads 10:10:00 at simulated time 0
HIDDEN = {"collision_hidden"}
# static collision boxes that dressing.py draws in full: the export loop leaves them out
DRESSED = ("counter_top", "counter_upstand", "casework", "toe_kick", "back_wall", "wall_", "trunking", "fit_", "sink_", "stand_", "bed",
           "arm_hand", "arm_forearm", "pad_mat")
EDGE_RADIUS = dict(tray=0.0012, basin=0.0012, sharps_red=0.004, sharps_lid=0.003, camera_dark=0.003, aluminum=0.001, robot_dark=0.001,
                   wipes_box=0.002, gauze=0.001, clamp=0.0012, tip_cap=0.0006, arm_board=0.004, sharps_flap=0.0006, syringe_rod=0.0003)
UNKNOWN = ((.40, .40, .40), .5, 0.)        # a material name this file does not know: neutral grey, listed in export.json


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
        return self.own_mdl(name, "SyringeHandlingDielectric", {"absorption": (T.Color3f, Gf.Vec3f(*map(float, absorption))),
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
        if not capsule:
            # a smooth cylinder as a mesh (the renderer facets USD's own cylinder into ten sides at this size)
            v, t, nr = revolve_shaded([(0., -0.5*height), (radius, -0.5*height), (radius, 0.5*height), (0., 0.5*height)], 48, 35.)
            turn = dict(Z=np.eye(3), X=np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]), Y=np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]))[axis]
            g = self.mesh(path, np.asarray(v) @ turn.T, t, normals=np.asarray(nr) @ turn.T, mat=mat)
            if p is not None:
                self.place(g, p, q)
            return g
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



# ====================================================================================== printed matter
def _font(size, bold=False):
    from PIL import ImageFont
    f = f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
    return ImageFont.truetype(f, size) if Path(f).exists() else ImageFont.load_default()


def scale_map(folder, kind, px_per_mm=40):
    """The barrel's printed scale as an image with a clear ground: lines at the syringe's divisions at their true pitch
    (area of the bore), long lines and numerals at the whole and half millilitres. u runs round the barrel, v along it
    from the zero line. Returns (path, width_m, length_m)."""
    from PIL import Image, ImageDraw
    s = P.SYRINGES[kind]
    A = math.pi*s["bore"]**2/4
    per_ml = H.ML/A
    n = int(round(s["nominal_ml"]/s["div_ml"]))
    width = 0.55*math.pi*s["od"]
    margin = 0.003
    length = s["nominal_ml"]*per_ml + 2*margin
    W, Hh = int(width*1e3*px_per_mm), int(length*1e3*px_per_mm)
    im = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    big = 10 if kind == "s1" else 5
    step = {"s3": 0.5, "s10": 1.0, "flush": 1.0, "s1": 0.1}[kind]
    lw = max(2, int(0.16*px_per_mm))
    f = _font(int((1.9 if kind != "s1" else 1.5)*px_per_mm), bold=True)
    for k in range(n + 1):
        y = Hh - int((margin + k*s["div_ml"]*per_ml)*1e3*px_per_mm)
        major = (k % big == 0)
        x1 = int(W*(0.62 if major else 0.40))
        d.rectangle([0, y - lw//2, x1, y + lw//2], fill=(8, 8, 12, 255))
        ml = k*s["div_ml"]
        if abs(ml/step - round(ml/step)) < 1e-6 and ml > 0:
            txt = f"{ml:g}"
            tw = d.textlength(txt, font=f)
            lab = Image.new("RGBA", (int(tw) + 4, f.size + 6), (0, 0, 0, 0))
            ImageDraw.Draw(lab).text((2, 0), txt, font=f, fill=(8, 8, 12, 255))
            lab = lab.rotate(180, expand=True)             # upright when the syringe is held needle up, as makers print them
            im.alpha_composite(lab, (min(int(W*0.66), W - lab.width), min(max(y - lab.height//2, 0), Hh - lab.height)))
    d.text((int(W*0.05), 4), "mL", font=_font(int(1.4*px_per_mm)), fill=(8, 8, 12, 255))
    path = Path(folder)/f"scale_{kind}.png"
    im.save(path)
    return path, width, length


def label_map(folder, name, size_m, lines, ground=(245, 245, 240), ink=(15, 20, 40), band=None, px_per_mm=12):
    """A printed label: lines of (text, height in mm, bold). Returns the image path."""
    from PIL import Image, ImageDraw
    W, Hh = int(size_m[0]*1e3*px_per_mm), int(size_m[1]*1e3*px_per_mm)
    im = Image.new("RGB", (W, Hh), ground)
    d = ImageDraw.Draw(im)
    y = int(0.06*Hh)
    if band:
        d.rectangle([0, 0, W, int(0.16*Hh)], fill=band)
        y = int(0.20*Hh)
    for text, mm, bold in lines:
        f = _font(max(6, int(mm*px_per_mm)), bold)
        d.text((int(0.05*W), y), text, font=f, fill=ink)
        y += int(mm*px_per_mm*1.35)
    path = Path(folder)/f"{name}.png"
    im.save(path)
    return path


def wrap_patch(radius, length, arc, n=24, x0=0.0, centre=math.pi/2):
    """A patch of a cylinder about the x axis: from x0 to x0 + length, `arc` radians wide about the angle `centre`
    (measured from +y toward +z). UV: u round, v along. Returns (v, t, uv, normals)."""
    v, uv, nr, t = [], [], [], []
    for i in range(n + 1):
        a = centre - 0.5*arc + arc*i/n
        c, s_ = math.cos(a), math.sin(a)
        for j, x in enumerate((x0, x0 + length)):
            v.append((x, radius*c, radius*s_))
            uv.append((i/n, j))
            nr.append((0., c, s_))
    for i in range(n):
        a, b, c, d = 2*i, 2*i + 1, 2*i + 2, 2*i + 3
        t += [(a, c, b), (b, c, d)]
    return np.array(v), np.array(t), np.array(uv), np.array(nr)


# ====================================================================================== the equipment's own shapes
RY90 = tuple(Rotation.from_euler("y", 90., degrees=True).as_quat())           # a revolve's z axis onto a body's +x


def _decal(S, name, image, alpha=False):
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    inputs = {"diffuse_texture": (T.Asset, Sdf.AssetPath(str(Path(image).resolve()))), "reflection_roughness_constant": (T.Float, 0.6)}
    if alpha:
        inputs.update({"enable_opacity": (T.Bool, True), "enable_opacity_texture": (T.Bool, True),
                       "opacity_texture": (T.Asset, Sdf.AssetPath(str(Path(image).resolve()))), "opacity_mode": (T.Int, 0),
                       "opacity_threshold": (T.Float, 0.5)})
    return S.mdl(name, "OmniPBR", inputs)


def _translucent(S, name, rgb, opacity, rough=0.35):
    T, Gf = S.Sdf.ValueTypeNames, S.Gf
    return S.mdl(name, "OmniPBR", {"diffuse_color_constant": (T.Color3f, Gf.Vec3f(*map(float, rgb))),
                                   "reflection_roughness_constant": (T.Float, float(rough)), "enable_opacity": (T.Bool, True),
                                   "opacity_constant": (T.Float, float(opacity))})


def _syringe(S, sc, s):
    sp, lu = s.s, P.LUER
    base = f"/Lab/body_{s.barrel}/look"
    ro, ri, L = 0.5*sp["od"], 0.5*sp["bore"], sp["barrel_len"]
    NL = lu["nozzle_len"]
    clear = S.mats["barrel_clear"]
    wall = [(ri, L), (ro, L), (ro, -0.001), (0.0022, -0.001), (0.5*lu["tip_d"], -NL), (0.0010, -NL), (0.0010, 0.0), (ri, 0.0)]
    S.revolve(base + "_barrel", wall, clear, segments=64, p=(0., 0., 0.), q=RY90)
    cr = min(0.5*lu["collar_od"], ro)
    S.revolve(base + "_collar", [(cr - 0.0009, -0.001), (cr, -0.001), (cr, -(NL - lu["nozzle_proj"])), (cr - 0.0009, -(NL - lu["nozzle_proj"]))],
              clear, segments=48, p=(0., 0., 0.), q=RY90)
    path, width, length = scale_map(S.folder/"textures", s.kind)
    v, t, uv, nr = wrap_patch(ro + 0.00004, length, width/ro, x0=-0.003)
    S.mesh(base + "_scale", v, t, uv=uv, normals=nr, mat=_decal(S, f"scale_{s.kind}", path, alpha=True), double=True)
    if s.kind == "flush":
        img = label_map(S.folder/"textures", "flush_label", (0.030, 0.016), [("0.9% Sodium Chloride", 2.6, True), ("Flush  10 mL", 2.6, False),
                                                                             ("training use only", 1.6, False)], band=(30, 90, 170))
        v, t, uv, nr = wrap_patch(ro + 0.00006, 0.016, 0.030/ro, x0=L - 0.019, centre=-math.pi/2)
        S.mesh(base + "_label", v, t, uv=uv, normals=nr, mat=_decal(S, "flush_label", img), double=True)
    # the plunger: a cross-shaped rod, the thumb rest's disc (the stopper is the body's own visual cylinder)
    pb = f"/Lab/body_{s.plunger}/look"
    rod = S.mat("syringe_rod")
    x0, x1 = sp["stopper_len"], s.rod_len
    wd = min(sp["rod_w"], 2*ri - 0.0006)
    S.box(pb + "_rod_a", (x1 - x0, wd, 0.0011), rod, p=(0.5*(x0 + x1), 0., 0.), r=0.0003)
    S.box(pb + "_rod_b", (x1 - x0, 0.0011, wd), rod, p=(0.5*(x0 + x1), 0., 0.), r=0.0003)
    S.cylinder(pb + "_disc", ri - 0.0004, 0.0012, rod, p=(x0 + 0.001, 0., 0.), axis="X")
    S.cylinder(pb + "_thumb", 0.5*sp["thumb_d"], sp["thumb_t"], rod, p=(s.rod_len + 0.5*sp["thumb_t"], 0., 0.), axis="X")


def _needle(S, n):
    if n.sheath is None:
        return
    rs = 0.5*P.NEEDLE_SHEATH["od"]
    rgb = COLORS["hub_" + n.kind][0] if n.n["blunt"] else (.80, .82, .83)
    mat = _translucent(S, f"sheath_{n.kind}", rgb, 0.55)
    base = f"/Lab/body_{n.sheath}/look"
    S.cylinder(base, rs, n.sheath_len, mat, p=(-0.5*n.sheath_len, 0., 0.), axis="X")
    for k in range(4):                                  # the four grip ribs
        a = math.pi/4 + k*math.pi/2
        S.box(f"{base}_rib{k}", (0.6*n.sheath_len, 0.0008, 0.0009), mat, p=(-0.32*n.sheath_len, (rs + 0.0003)*math.cos(a), (rs + 0.0003)*math.sin(a)),
              q=tuple(Rotation.from_euler("x", a + math.pi/2).as_quat()), r=0.0002)


def _vial(S, v):
    d = P.VIAL
    r, Hh = 0.5*d["od"], d["height"]
    body_h = Hh - d["neck_h"] - d["cap_h"]
    ri, z0 = v.r_in, v.z0
    top = Hh - d["cap_h"]
    wall = [(0., 0.), (r - 0.002, 0.), (r, 0.002), (r, body_h - 0.003), (0.0083, body_h + 0.003), (0.0083, top - 0.0035), (0.010, top - 0.0035),
            (0.010, top), (0.0063, top), (0.0063, body_h + 0.002), (ri, body_h - 0.005), (ri, z0 + 0.001), (ri - 0.001, z0), (0., z0)]
    base = f"/Lab/body_{v.body}/look"
    S.revolve(base + "_glass", wall, S.mats["glass"], segments=72)
    rc = 0.5*d["flange_d"] + 0.0003
    S.revolve(base + "_crimp", [(0.0036, Hh), (rc - 0.0006, Hh), (rc, Hh - 0.0006), (rc, top - 0.0030), (rc - 0.0012, top - 0.0042), (0.0086, top - 0.0042),
                                (0.0086, top - 0.0036), (rc - 0.0008, top - 0.0030), (rc - 0.0008, Hh - 0.0008), (0.0036, Hh - 0.0008)], S.mat("vial_cap"), segments=64)
    S.cylinder(base + "_target", 0.5*d["target_d"] + 0.0002, 0.0012, S.mat("vial_target"), p=(0., 0., Hh - 0.0007))
    S.cylinder(base + "_plug", 0.0062, 0.006, S.mat("vial_target"), p=(0., 0., top - 0.003))
    img = label_map(S.folder/"textures", "vial_label", (0.050, 0.022), [("Simulated medication", 3.0, True), ("for training only", 2.4, False),
                                                                        ("10 mL  multi-dose", 2.4, False)], band=(25, 70, 160))
    vv, t, uv, nr = wrap_patch(r + 0.00008, 0.022, 0.050/r, x0=0.005, centre=-math.pi/2)
    # the patch is about the x axis: turn it onto the vial's z (x -> z, y -> x, z -> y)
    turn = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    S.mesh(base + "_label", vv @ turn.T, t, uv=uv, normals=nr @ turn.T, mat=_decal(S, "vial_label", img), double=True)


def _pad(S, sc):
    b = sc.bodies["pad"]
    hx, hy = 0.5*P.PAD["size"][0], 0.5*P.PAD["size"][1]
    base = f"/Lab/body_{b}/look"
    z = 0.0
    for name, th in (("board", P.PAD["board"]), ("muscle", P.LAYERS["muscle"]["thick"]), ("fat", P.LAYERS["fat"]["thick"]),
                     ("skin", P.LAYERS["skin"]["thick"])):
        S.box(f"{base}_{name}", (2*hx, 2*hy, th), S.mat("pad_" + name), p=(0., 0., z + 0.5*th), r=min(0.002, 0.4*th))
        z += th
    r1 = 0.5*P.PAD["ring_d"]
    ang = np.linspace(0, 2*math.pi, 49)
    v = np.array([(rr*math.cos(a), rr*math.sin(a), z + 0.00015) for a in ang for rr in (r1 - 0.0007, r1)])
    t = [(2*i, 2*i + 2, 2*i + 1) for i in range(48)] + [(2*i + 1, 2*i + 2, 2*i + 3) for i in range(48)]
    S.mesh(base + "_ring", v, t, normals=[(0., 0., 1.)]*len(v), mat=S.mat("pad_ring"), double=True)


def _lock(S, sc):
    g = sc.lock_geometry
    lk = sc.fluid.locks["lock"]
    c = P.CONNECTOR
    S.cylinder(f"/Lab/body_{lk.body}/look_ring", 0.5*c["od"] + 0.0005, 0.005, S.mat("connector_ring"), p=(-0.008, 0., 0.), axis="X")
    S.box(f"/Lab/body_{lk.body}/look_tape", (0.012, 0.045, 0.0004), S.mat("tape"), p=(-0.022, 0., 0.5*c["od"] + 0.0003), r=0.0001)
    S.tube("/Dressing/lock/extension", [tuple(p) for p in g["tube"]], 0.5*P.EXTENSION["od"], S.mats["tube_clear"], sides=14)
    cx, cy, cz = g["cannula"]
    S.cylinder("/Dressing/lock/cannula_hub", 0.003, 0.016, S.mat("clamp"), p=(cx, cy, cz + 0.001), axis="X")
    S.box("/Dressing/lock/dressing", (0.060, 0.050, 0.0006), _translucent(S, "dressing_film", (.82, .84, .84), 0.35, 0.12), p=(cx + 0.012, cy, cz + 0.004), r=0.0002)
    # the hand of the training arm: palm, four fingers and a thumb (the collision block is hidden)
    x0, x1, th = g["hand"]
    skin = S.mat("arm_skin")
    zb = g["board_z"]
    yc = g["y"]
    # the forearm and the hand as one skin each (the collision capsules and the hand's block are hidden): the forearm
    # tapers from the elbow to the wrist and lies on the board; the palm widens from the wrist to the knuckles
    rr, seg = g["radii"], g["seg"]
    xs = np.linspace(x1 + 0.004, x1 + 4*seg, 17)
    rad = np.interp(xs, [x1] + [x1 + seg*(k + 0.5) for k in range(4)] + [x1 + 4*seg], [rr[0] - 0.003] + list(rr) + [rr[3] + 0.001])
    bulge = 0.004*np.exp(-((xs - (x1 + 2.6*seg))/(0.9*seg))**2)            # the forearm's belly
    st = [(float(x), yc, zb + float(r + b_)*0.93, float(r + b_)*1.06, float(r + b_)*0.93) for x, r, b_ in zip(xs, rad, bulge)]
    st = [(st[0][0] - 0.004, yc, st[0][2], st[0][3]*0.6, st[0][4]*0.6)] + st + [(st[-1][0] + 0.012, yc, st[-1][2], st[-1][3]*0.75, st[-1][4]*0.75)]
    v_, f_ = loft_mesh(st, 32)
    S.mesh("/Dressing/arm/forearm", v_, f_, mat=skin, smooth=True)
    palm = [(x1 + 0.006, yc, zb + 0.023, 0.024, 0.019), (x1 - 0.010, yc, zb + 0.022, 0.030, 0.018), (x1 - 0.035, yc - 0.002, zb + 0.020, 0.040, 0.016),
            (x1 - 0.065, yc - 0.003, zb + 0.019, 0.043, 0.014), (x1 - 0.086, yc - 0.003, zb + 0.018, 0.041, 0.011), (x1 - 0.094, yc - 0.003, zb + 0.017, 0.036, 0.007)]
    v_, f_ = loft_mesh(palm, 32)
    S.mesh("/Dressing/arm/palm", v_, f_, mat=skin, smooth=True)
    nail = S.mat("paper")
    for k in range(4):                                   # index to little finger: three phalanges, slightly curled onto the board
        yy = yc - 0.031 + 0.0205*k
        ln = (0.074, 0.083, 0.077, 0.060)[k]
        r0 = (0.0088, 0.0090, 0.0086, 0.0076)[k]
        xk = x1 - 0.088 + 0.004*abs(k - 1.5)
        pts = [(xk + 0.010, yy, zb + 0.018), (xk - 0.42*ln, yy, zb + 0.021), (xk - 0.72*ln, yy, zb + 0.016), (xk - ln, yy, zb + r0*0.9)]
        S.tube(f"/Dressing/arm/finger{k}", pts, r0, skin, sides=14)
        for jn, f_ in enumerate((0.42, 0.72)):           # the knuckles' swell
            S.sphere(f"/Dressing/arm/knuckle{k}_{jn}", r0*1.10, skin, p=(xk - f_*ln, yy, zb + (0.021, 0.016)[jn]))
        S.box(f"/Dressing/arm/nail{k}", (0.010, 0.0105, 0.0012), nail, p=(xk - ln + 0.004, yy, zb + r0*1.75), r=0.0005)
    S.tube("/Dressing/arm/thumb", [(x1 - 0.020, yc - 0.034, zb + 0.016), (x1 - 0.050, yc - 0.055, zb + 0.016), (x1 - 0.080, yc - 0.064, zb + 0.014),
                                   (x1 - 0.104, yc - 0.064, zb + 0.010)], 0.0108, skin, sides=14)
    S.sphere("/Dressing/arm/thumb_base", 0.0165, skin, p=(x1 - 0.024, yc - 0.036, zb + 0.017))


def loft_mesh(stations, sides=28):
    """A closed skin over elliptical cross-sections along x: stations are (x, yc, zc, ry, rz); the ends are capped with
    a fan. Returns (vertices, triangles) for a smooth-shaded mesh."""
    v, f = [], []
    ang = np.linspace(0., 2*np.pi, sides, endpoint=False)
    for (x, yc, zc, ry, rz) in stations:
        v += [(x, yc + ry*np.cos(a_), zc + rz*np.sin(a_)) for a_ in ang]
    n = len(stations)
    for i in range(n - 1):
        for k in range(sides):
            a0, a1 = i*sides + k, i*sides + (k + 1) % sides
            b0, b1 = a0 + sides, a1 + sides
            f += [(a0, b0, a1), (a1, b0, b1)]
    for end, i in ((0, 0), (1, n - 1)):
        x, yc, zc, _, _ = stations[i]
        c = len(v)
        v.append((x, yc, zc))
        for k in range(sides):
            a0, a1 = i*sides + k, i*sides + (k + 1) % sides
            f.append((c, a0, a1) if end == 0 else (c, a1, a0))
    return np.asarray(v, float), np.asarray(f, int)


# shapes whose collision piece is replaced by one of the shapes above (suffix of the shape's label)
REPLACED = ("vial_cap", "_barrel_tube", "_collar", "_nozzle_tip", "_rod", "_thumb0", "_thumb1", "_thumb_fill", "_sheath0", "_sheath1", "_glass", "_neck",
            "_foot0", "_foot1", "_foot2", "pad_block")


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
    S.mats["syringe_barrel"] = S.mats["barrel_clear"]              # the flange is the barrel's own plastic
    S.mats["sharps_lid"] = _translucent(S, "sharps_lid", (.86, .86, .83), 0.88, 0.25)
    q = world.bq
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    # one more carrier after the bodies: the following camera rides it (video.with_follow gives its pose per frame)
    tx, ty, _ = sc.layout["tray"]
    S.xform(f"/Lab/body_{len(q)}", (tx, ty, P.COUNTER_Z + 0.02), (0., 0., 0., 1.))
    V = int(newton.ShapeFlags.VISIBLE)
    labels = list(b.shape_label)
    for i in range(b.shape_count):
        body, lab = b.shape_body[i], labels[i]
        mname = sc.materials.get(i, "robot")
        if mname in HIDDEN or lab == "floor" or (body < 0 and lab.startswith(DRESSED)) or lab.endswith(REPLACED):
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
            col = np.clip(np.asarray(b.shape_color[i], float), 0., 1.)**2.2
            key = "robot_%02x%02x%02x" % tuple(int(round(c*255)) for c in np.asarray(b.shape_color[i], float).clip(0, 1))
            mat = S.color(key, col, .28 if float(col.mean()) > 0.5 else .38, 0.)
        elif mname == "stopper_rubber" and "flush" in lab:
            mat = S.mat("stopper_grey")
        else:
            mat = S.mat(mname)
        if src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True, mat=mat)
            S.place(g, tf[:3], tf[3:])
        elif st == newton.GeoType.BOX:
            S.box(path, 2*scale, mat, p=tf[:3], q=tf[3:], r=min(EDGE_RADIUS.get(mname, 0.0006), 0.45*float(scale.min())))
        elif st in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            S.cylinder(path, scale[0], 2*scale[1], mat, p=tf[:3], q=tf[3:], capsule=st == newton.GeoType.CAPSULE)
        elif st == newton.GeoType.SPHERE:
            S.sphere(path, scale[0], mat, p=tf[:3])
    fl = sc.fluid
    for s in fl.syringes.values():
        _syringe(S, sc, s)
    for n in fl.needles.values():
        _needle(S, n)
    for v in fl.vials.values():
        _vial(S, v)
    if "pad" in sc.bodies:
        _pad(S, sc)
    if fl.locks:
        _lock(S, sc)
    # ---- the liquid's meshes (geometry replaced per request)
    T = S.Sdf.ValueTypeNames
    S.UsdGeom.Scope.Define(S.stage, "/Liquid")
    meta = dict(liquid={})
    for name, mat in [(n, "liquid_" + n) for n in LIQUID_NAMES] + list(CLOCK_NAMES):
        g = S.mesh(f"/Liquid/{name}", [(0., 0., -50.), (.001, 0., -50.), (0., .001, -50.)], [0, 1, 2], mat=S.mats[mat])
        pv = S.UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar("displayColor", T.Color3fArray, "vertex").Set(S.Vt.Vec3fArray([S.Gf.Vec3f(.8, .8, .8)]*3))
        pv.CreatePrimvar("gloss", T.FloatArray, "vertex").Set(S.Vt.FloatArray([0.]*3))
        pv.CreatePrimvar("trans", T.FloatArray, "vertex").Set(S.Vt.FloatArray([0.]*3))
        S.UsdGeom.Imageable(g).MakeInvisible()
        meta["liquid"][name] = str(g.GetPath())
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
        if m["body"] >= 0 or m["body"] == -2:
            body = m["body"] if m["body"] >= 0 else sc.b.body_count          # -2: the following camera's carrier
            g = S.UsdGeom.Camera.Define(S.stage, f"/Lab/body_{body}/{name}")
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


# ====================================================================================== the liquid, from the state
def _cylinder_soup(radius, x0, x1, n=20):
    """A closed cylinder about the x axis as triangles (vertices, faces), for clipping."""
    ang = np.linspace(0, 2*math.pi, n, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)])*radius
    v = np.array([(x, y, z) for x in (x0, x1) for y, z in ring] + [(x0, 0., 0.), (x1, 0., 0.)])
    t = []
    for i in range(n):
        j = (i + 1) % n
        t += [(i, j, n + i), (j, n + j, n + i), (2*n, j, i), (2*n + 1, n + i, n + j)]
    return v, np.array(t)


def _clip(v, t, normal, offset):
    """The part of a closed convex triangle mesh with normal . x <= offset, closed again by a cap. Returns (v, t)."""
    d = v @ normal - offset
    if (d <= 0).all():
        return v, t
    if (d >= 0).all():
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    out_v = [tuple(p) for p in v]
    out_t, cut = [], []

    def mid(i, j):
        s = d[i]/(d[i] - d[j])
        out_v.append(tuple(v[i] + s*(v[j] - v[i])))
        return len(out_v) - 1
    for a, b, c in t:
        ins = [d[a] <= 0, d[b] <= 0, d[c] <= 0]
        k = sum(ins)
        if k == 3:
            out_t.append((a, b, c))
        elif k == 0:
            continue
        else:
            tri = [a, b, c]
            while not (ins[0] and not ins[1]) and not (k == 1 and ins[0]):      # rotate to a canonical order
                tri = tri[1:] + tri[:1]
                ins = ins[1:] + ins[:1]
                if k == 2 and ins[0] and ins[1] and not ins[2]:
                    break
            if k == 1:
                i0 = ins.index(True)
                a_, b_, c_ = tri[i0], tri[(i0 + 1) % 3], tri[(i0 + 2) % 3]
                p, q_ = mid(a_, b_), mid(a_, c_)
                out_t.append((a_, p, q_))
                cut.append((p, q_))
            else:
                i0 = ins.index(False)
                c_, a_, b_ = tri[i0], tri[(i0 + 1) % 3], tri[(i0 + 2) % 3]
                p, q_ = mid(b_, c_), mid(a_, c_)
                out_t += [(a_, b_, p), (a_, p, q_)]
                cut.append((p, q_))
    V = np.array(out_v)
    if cut:
        idx = sorted({i for e in cut for i in e})
        c0 = V[idx].mean(0)
        V = np.vstack([V, c0])
        ci = len(V) - 1
        for p, q_ in cut:
            tri = (ci, q_, p)
            n_ = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
            out_t.append(tri if n_ @ normal > 0 else (ci, p, q_))
    return V, np.array(out_t)


def _volume(v, t):
    if not len(t):
        return 0.0
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())/6.0)


def _level_fill(v, t, up, volume):
    """Clip a closed convex mesh by a level surface (normal `up`) so that what is under it has `volume`."""
    full = _volume(v, t)
    if volume >= full*0.999:
        return v, t
    if volume <= full*0.001:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    h = v @ up
    lo, hi = float(h.min()), float(h.max())
    for _ in range(22):
        m = 0.5*(lo + hi)
        if _volume(*_clip(v, t, up, m)) < volume:
            lo = m
        else:
            hi = m
    return _clip(v, t, up, 0.5*(lo + hi))


def _species(mix):
    return P.SPECIES[int(np.argmax(mix[:H.AIR]))]


def _ball(n=1):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=n, radius=1.0)
    return np.asarray(m.vertices), np.asarray(m.faces)


_BALL = None


def _blob(centre, radius, flatten=1.0, up=(0., 0., 1.)):
    global _BALL
    if _BALL is None:
        _BALL = _ball()
    v, t = _BALL
    up = np.asarray(up, float)
    w = v*radius
    if flatten != 1.0:
        w = w - np.outer(w @ up, up)*(1.0 - flatten)
    return w + np.asarray(centre, float), t


def liquid_meshes(world, body_q=None, state=None):
    """{liquid name: (vertices, triangles)} in world coordinates for the simulator's present state (or for a recorded
    `state` from liquid_state with the recorded body poses `body_q`)."""
    st = state if state is not None else liquid_state(world)
    bq = world.bq if body_q is None else np.asarray(body_q, float)
    parts = {k: [] for k in LIQUID_NAMES}
    up = np.array([0., 0., 1.])

    def add(name, v, t, body=None):
        if not len(t):
            return
        if body is not None and body >= 0:
            q = bq[body]
            v = q[:3] + Rotation.from_quat(q[3:]).apply(v)
        parts[name].append((v, t))

    for s in st["syringes"]:
        q = bq[s["body"]]
        R = Rotation.from_quat(q[3:])
        up_l = R.inv().apply(up)
        r = 0.5*s["bore"] - 0.00012
        x = max(s["x"], 1e-5)
        liq = sum(v for _, v in s["parcels"])
        if liq <= 1e-11:
            continue
        v0, t0 = _cylinder_soup(r, 0.0, x)
        full = _volume(v0, t0)
        scale = min(1.0, full/max(liq + s["gas"], 1e-12))              # the drawn column never overfills the bore
        if s["slug"]:
            # a small bore: the gas is a slug across the bore where the model has it
            lg = min(s["gas"]/(math.pi*r*r), x)
            a = s["gas_pos"]*(x - lg)
            pieces = [(0.0, a), (a + lg, x)]
        else:
            pieces = None
        x_at = 0.0
        tot = max(liq, 1e-12)
        for sp, vol in s["parcels"]:
            share = vol/tot
            if pieces is None:
                xa, xb = x_at, x_at + share*x
                v, t = _cylinder_soup(r, xa, xb)
                v, t = _level_fill(v, t, up_l, min(vol*scale, _volume(v, t)))
                add(sp, v, t, s["body"])
            else:
                length = sum(b - a for a, b in pieces)
                la, lb = x_at/x*length, (x_at + share*x)/x*length
                run = 0.0
                for a, b in pieces:
                    sa, sb = max(la, run), min(lb, run + (b - a))
                    if sb > sa + 1e-6:
                        v, t = _cylinder_soup(r, a + (sa - run), a + (sb - run))
                        add(sp, v, t, s["body"])
                    run += b - a
            x_at += share*x
    for vi in st["vials"]:
        q = bq[vi["body"]]
        R = Rotation.from_quat(q[3:])
        if vi["volume"] > 1e-10:
            v0, t0 = _cylinder_soup(vi["r"] - 0.0002, vi["z0"] + 0.0003, vi["z0"] + vi["h"], n=28)
            turn = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])          # the soup's x axis onto the vial's z
            v0 = v0 @ turn.T
            v, t = _level_fill(v0, t0, R.inv().apply(up), vi["volume"])
            add(vi["species"], v, t, vi["body"])
    for ln in st["lines"]:
        pts = np.asarray(ln["path"], float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        vol_total = sum(v for _, v in ln["parcels"])
        at = 0.0
        for sp, vol in ln["parcels"]:
            a, b = at/vol_total*total, (at + vol)/vol_total*total
            at += vol
            if sp == "air" or b - a < 2e-4:
                continue
            ss = np.unique(np.clip(np.concatenate([[a, b], cum[(cum > a) & (cum < b)]]), a, b))
            p = np.column_stack([np.interp(ss, cum, pts[:, k]) for k in range(3)])
            v, t, _ = tube_mesh([tuple(x) for x in p], ln["r"], 8)
            add(sp, np.asarray(v), np.asarray(t).reshape(-1, 3))
    for dr in st["drops"]:
        v, t = _blob(dr["p"], dr["r"])
        add(dr["species"], v, t)
    for pd in st["puddles"]:
        q = bq[pd["body"]] if pd["body"] >= 0 else None
        nrm = np.array([0., 0., 1.]) if q is None else Rotation.from_quat(q[3:]).inv().apply(up)
        v, t = _blob(np.asarray(pd["p"]) + 0.0002*nrm, pd["r"], flatten=min(1.0, 0.0009/pd["r"]), up=nrm)
        add(pd["species"], v, t, pd["body"])
    out = {}
    if "t" in st:                                    # the wall clock shows the simulated time
        out.update(clock_meshes(st["t"]))
    for name, items in parts.items():
        if not items:
            continue
        vs, ts, n = [], [], 0
        for v, t in items:
            vs.append(v)
            ts.append(np.asarray(t) + n)
            n += len(v)
        out[name] = dict(v=np.vstack(vs), t=np.vstack(ts))
    return out


def clock_meshes(t):
    """The wall clock's hour and minute hands and its ticking second hand at simulated time t (s), as flat bars on the
    clock's face (dressing.py draws the case, the face and the marks)."""
    kx, kw, kh, kd, kz = P.WALL_FITTINGS["clock"]
    y_face = P.ROOM["y1"] - kd - 0.002
    c = np.array([kx, 0., kz + 0.5*kh])
    now = CLOCK_START_S + float(t)
    sec = math.floor(now) % 60                           # a quartz movement: the second hand steps once a second
    minute = (now/60.0) % 60.0
    hour = (now/3600.0) % 12.0

    def bar(a, back, length, width, y, th=0.0015):
        u, v = np.array([math.sin(a), 0., math.cos(a)]), np.array([math.cos(a), 0., -math.sin(a)])
        pts = []
        for dy in (0.0, -th):
            for al, ac in ((-back, -0.5), (length, -0.5), (length, 0.5), (-back, 0.5)):
                pts.append(c + u*al + v*ac*width + np.array([0., y + dy, 0.]))
        f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        return np.array(pts), np.array(f)
    hands = [bar(2*math.pi*hour/12.0, 0.012, 0.070, 0.009, y_face), bar(2*math.pi*minute/60.0, 0.014, 0.105, 0.006, y_face - 0.0016)]
    v = np.vstack([h[0] for h in hands])
    tri = np.vstack([hands[0][1], hands[1][1] + 8])
    sv, st_ = bar(2*math.pi*sec/60.0, 0.028, 0.118, 0.0022, y_face - 0.0032, th=0.001)
    return {"clock_hands": dict(v=v, t=tri), "clock_second": dict(v=sv, t=st_)}


def liquid_state(world):
    """What liquid_meshes draws, as plain numbers (recorded per frame for films)."""
    fl = world.fluid
    d = world.solver.mj_data
    st = dict(syringes=[], vials=[], lines=[], drops=[], puddles=[], t=float(world.sim_time))
    for s in fl.syringes.values():
        ch = s.chamber
        label = "SALINE FLUSH" if s.kind == "flush" else ("CONTRAST" if s.name == "dose" and world.config.task == "fast-bolus" else "MEDICATION" if s.name == "dose" else f"{s.s['nominal_ml']:g} mL")
        st["syringes"].append(dict(label=label, body=int(s.barrel), bore=s.s["bore"], nominal_ml=s.s["nominal_ml"], div_ml=s.s["div_ml"], od=s.s["od"], x=float(d.qpos[s.a_pl]), gas=float(ch.gas_volume()),
                                   gas_pos=float(ch.gas_pos), slug=bool(s.s["bore"] < 0.006),
                                   parcels=[(_species(p), float(p[:H.AIR].sum())) for p in ch.stack.items if float(p[:H.AIR].sum()) > 1e-12]))
    for v in fl.vials.values():
        if v.vessel.liquid_volume > 0:
            up, level = v.plane(world)
            st["vials"].append(dict(up=up.tolist(), level=level, body=int(v.body), r=v.r_in, z0=v.z0, h=v.h_in, body_h=v.body_h, neck_r=P.VIAL["bore"]/2, volume=float(v.vessel.liquid_volume), species=_species(v.vessel.liquid)))
    for lk in fl.locks.values():
        st["lines"].append(dict(path=[list(map(float, p)) for p in world.scene.lock_geometry["tube"]], r=0.5*P.EXTENSION["bore"]*0.92,
                                parcels=[("air" if p[H.AIR] > 0.5*p.sum() else _species(p), float(p.sum())) for p in lk.line.items]))
    sp = fl.spray
    for i in range(len(sp.pos)):
        vol = float(sp.mix[i, :H.AIR].sum())
        st["drops"].append(dict(p=[float(x) for x in sp.pos[i]], r=(3*vol/(4*math.pi))**(1/3), species=_species(sp.mix[i])))
    for acc, where in sp.pending.values():
        vol = float(acc[:H.AIR].sum())
        if vol > 2e-10 and where is not None:
            origin = np.asarray(where[0], float)
            r = (3*vol/(4*math.pi))**(1/3)
            st["drops"].append(dict(p=[float(origin[0]), float(origin[1]), float(origin[2] - 0.7*r)], r=r, species=_species(acc)))
    # what has landed: merged on a 4 mm grid of the body it lies on
    grid = {}
    for place, body, local, mix in sp.deposits:
        key = (int(body), _species(mix), int(round(local[0]/0.004)), int(round(local[1]/0.004)), int(round(local[2]/0.004)))
        g = grid.setdefault(key, [np.zeros(3), 0.0])
        vol = float(mix[:H.AIR].sum())
        g[0] += np.asarray(local, float)*vol
        g[1] += vol
    for (body, species, *_), (psum, vol) in grid.items():
        if vol > 1e-10:
            st["puddles"].append(dict(body=body, species=species, p=[float(x) for x in psum/vol], r=float(math.sqrt(vol/(math.pi*0.0008)))))
    return st


# ====================================================================================== worker client
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
    if size > 60*1024*1024:
        raise ValueError(f"liquid meshes of {size >> 20} MiB exceed the renderer's 64 MiB message: merge deposits "
                         "(fewer deposits)")
    return out


# ====================================================================================== worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py). width/height None: every camera renders at its
    own resolution (scene.cameras[name]['width'/'height'])."""

    def __init__(self, world, folder, scene, *, width=None, height=None, samples=24, engine="isaac"):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        self.engine=engine
        self.last_timings = {}
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        module="syringe_handling.render_worker"
        if engine=='mujoco':
            import sys,mujoco
            executable=sys.executable;module='syringe_handling.sensor_worker'
            self.folder.mkdir(parents=True,exist_ok=True);scene=self.folder/'sensor.mjb'
            m=world.solver.mj_model;mujoco.mj_saveModel(m,str(scene),None)
            colours={}
            for shape,material in world.scene.materials.items():
                try:g=world.geom_of_shape(shape)
                except (KeyError,IndexError):continue
                if g<0:continue
                rgba=[*np.power(COLORS.get(material,((.6,.64,.68),0,0))[0],1/2.2),1.]
                if material=='collision_hidden':rgba[3]=0.
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
                if 'barrel_tube' in name or 'vial_glass' in name:rgba=[.65,.8,.88,.22]
                colours[g]=rgba
            scene.with_suffix('.json').write_text(json.dumps(dict(colours=colours,cameras=world.scene.cameras,body_map={v:k for k,v in world._newton_to_mj.items()})))

        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        start_s = float(os.environ.get("SYRINGE_RENDER_START_S", "240"))   # a worker started on a full card hangs: give up
        listener.settimeout(start_s)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, SYRINGE_RENDER_PORT=str(listener.getsockname()[1]), SYRINGE_RENDER_SECRET=secret)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", module, "--scene",
                                             str(Path(scene).resolve()), "--width", str(width or 1280),
                                             "--height", str(height or 720), "--samples", str(samples)],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept()
            self.connection.settimeout(start_s)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed")
            self.connection.settimeout(600)
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
        """Render `cameras` for the body poses `body_q` and the liquid `meshes` (liquid_meshes; None: no liquid
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
        if self.engine=="mujoco":msg["liquid"]=liquid_state(self.world)
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
        empty = 0
        for k in range(self.tries):
            try:
                return self._r.observe(*a, **kw)
            except (ConnectionError, RuntimeError, TimeoutError, OSError) as e:
                # a frame that comes back empty from three fresh workers is not the card being busy: a camera stands
                # inside a solid (a close view moved to the sharps slot put the wall camera inside the wall, and a
                # splice restarted its worker for 25 minutes on that one frame)
                empty += int("No valid readback" in str(e))
                if empty >= 3:
                    raise RuntimeError(f"the frame came back empty from {empty} workers in a row (cameras {a[0] if a else kw.get('cameras')}): "
                                       "a camera inside a solid?") from e
                self.restarts += 1
                print(f"render worker lost ({type(e).__name__}: {e}); restart {self.restarts} after {self.wait_s:g} s", flush=True)
                try:
                    self._r.close()
                except Exception:
                    pass
                time.sleep(self.wait_s)
                self._r = self._start()
        raise RuntimeError(f"the render worker was lost {self.tries} times on one request")

    def renew(self, why=""):
        """A new worker for the one in use (a worker that has lost readbacks on a busy card went on at 30 to 120 s a
        frame; a new one rendered the same frames at 3 s)."""
        self.restarts += 1
        print(f"render worker renewed ({why}); restart {self.restarts}", flush=True)
        try:
            self._r.close()
        except Exception:
            pass
        self._r = self._start()

    @property
    def last_timings(self):
        return self._r.last_timings

    def close(self):
        self._r.close()
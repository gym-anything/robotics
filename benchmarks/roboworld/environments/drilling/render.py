"""USD export of the workshop and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses and the current workpiece meshes, so
one exported stage serves a whole session. Appearance rules:
  * collision proxies are never drawn; the robot uses its model's visual meshes and colours; the cordless drill and
    the drill press are drawn by the Poly Haven models their colliders were fitted to (tools.py); the rotary hammer's
    housing and every shape listed in visual_models.shape_visual (the robot cart, kit trolley, workbench, press
    cabinet, the hammer's moving parts, bits, fasteners, cleaning and hand tools, hold-down clamps) by procedural
    models fitted to their colliders; the rest by their collision primitives with physical materials;
  * every drillable workpiece is drawn from its hole records: the box minus every hole (stages, drill-point cones,
    dimples, entry spall), plus exit burr rings and tear-out craters, rebuilt with manifold3d when its holes change;
  * layout marks are pencil crosses on the faces; dust in a concrete hole is drawn as its column;
  * drilling debris is DRAWN ONLY - no collider, no mass, no effect on the physics - from the same hole records
    (DEBRIS_LEDGER, hole_debris): a concrete hole's dust that left the hole clings to the face below the mouth and
    lies in a heap with a scatter around it where it fell (the vacuum takes the hole's own column, never this);
    steel and aluminium holes leave curled chips, timber holes curled shavings, on the workpiece and on whatever is
    below; seeded by (workpiece, hole id), so a re-render of a recording gives the same frames;
  * the vacuum unit and its switch are procedural models on their bodies (visual_models.shop_vacuum); the hose is
    drawn only, recomputed for every pose from the unit's inlet to the wand (Hose); a broken bit is drawn cut back
    as the physics shortened it (broken_bit_updates);
  * the room dressing is visual only (dressing.py).
"""
from gym_anything_robotics.paths import worker_env

import base64
import hashlib
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

# diffuse colour (linear), roughness, metallic
COLORS = dict(
    robot=((.86, .87, .86), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    aluminum=((.62, .64, .66), .25, .9), steel=((.55, .56, .58), .35, .9), steel_dark=((.12, .12, .13), .45, .7),
    steel_bright=((.70, .71, .72), .22, 1.), steel_galv=((.52, .54, .56), .40, .8), zinc=((.62, .63, .60), .35, .85),
    hss_bit=((.38, .38, .40), .30, .9), sds_bit=((.20, .20, .21), .35, .8), chuck_black=((.04, .04, .045), .35, .3),
    aluminium=((.74, .75, .77), .32, .9), douglas_fir=((.62, .42, .24), .65, 0.), plywood=((.66, .52, .34), .7, 0.),
    concrete=((.36, .35, .33), .9, 0.), concrete_wall=((.40, .39, .37), .92, 0.),
    drill_yellow=((.78, .52, .02), .45, 0.), drill_grip=((.02, .02, .02), .7, 0.), drill_battery=((.04, .04, .04), .5, 0.),
    drill_trigger=((.05, .05, .05), .4, 0.), drill_collar=((.03, .03, .03), .5, 0.),
    hammer_yellow=((.78, .52, .02), .45, 0.), hammer_black=((.03, .03, .03), .55, 0.), hammer_grip=((.02, .02, .02), .75, 0.),
    press_green=((.10, .22, .12), .45, .2), stand_grey=((.22, .23, .24), .5, .6), vise_blue=((.05, .16, .36), .45, .2),
    knob_black=((.02, .02, .02), .5, 0.), switch_red=((.5, .02, .02), .4, 0.), punch_knurl=((.30, .30, .31), .45, .9),
    cart_grey=((.18, .19, .20), .5, .5), tray_black=((.02, .02, .02), .85, 0.), bench_steel=((.10, .20, .30), .45, .4),
    hardwood=((.45, .28, .14), .55, 0.), hose_black=((.03, .03, .03), .7, 0.), brush_handle=((.6, .1, .05), .5, 0.),
    brush_wire=((.55, .55, .55), .6, .9), hickory=((.70, .52, .30), .55, 0.), clamp_orange=((.80, .25, .02), .45, 0.),
    block_paint=((.70, .70, .66), .85, 0.), ceiling=((.85, .85, .83), .9, 0.), pencil=((.05, .05, .06), .6, 0.),
    burr=((.75, .75, .76), .25, 1.), scorch=((.08, .05, .03), .9, 0.), dust=((.33, .32, .30), .95, 0.),
    floor_concrete=((.33, .33, .32), .8, 0.),
    # procedural models (visual_models.py) and room dressing
    cart_frame=((.035, .037, .040), .42, .35), cart_panel=((.42, .44, .45), .45, .25), alu_machined=((.70, .71, .73), .24, 1.),
    anodized_black=((.025, .025, .028), .35, .55), black_oxide=((.035, .035, .038), .38, .75),
    caster_tread=((.045, .048, .052), .65, 0.), caster_hub=((.42, .43, .45), .45, 0.),
    estop_yellow=((.82, .58, .015), .40, 0.), estop_red=((.62, .015, .012), .32, 0.),
    controller_black=((.030, .031, .034), .45, .30), controller_front=((.055, .057, .060), .35, .55),
    vent_dark=((.008, .008, .009), .8, 0.), conduit_black=((.018, .018, .02), .55, 0.), rubber_grey=((.10, .10, .105), .7, 0.),
    rubber_dust=((.03, .03, .032), .75, 0.), decal_white=((.78, .78, .76), .5, 0.), trolley_grey=((.30, .31, .32), .42, .35),
    anchor_box_red=((.55, .03, .02), .55, 0.), cardboard=((.47, .34, .20), .85, 0.), bag_clear=((.55, .56, .58), .30, 0.),
    bench_top_steel=((.115, .12, .125), .50, .60), drawer_grey=((.30, .33, .36), .40, .35), heel_blue=((.05, .12, .26), .45, .3),
    hammer_grey=((.13, .13, .14), .45, .1), hose_grey=((.20, .21, .22), .6, 0.), pegboard=((.24, .15, .08), .62, 0.),
    peg_hole=((.012, .010, .008), .9, 0.), emt_galv=((.58, .59, .60), .32, .9), box_grey=((.50, .51, .52), .38, .8),
    door_grey=((.40, .42, .44), .40, .35), safety_yellow=((.80, .60, .02), .45, 0.), plastic_grey=((.36, .37, .38), .5, 0.),
    cement_paper=((.62, .58, .50), .9, 0.), copper=((.80, .45, .28), .30, 1.), pipe_red=((.45, .04, .03), .45, .2),
    duct_galv=((.62, .63, .64), .36, .85), wood_frame=((.55, .40, .24), .65, 0.), white_paint=((.80, .80, .78), .55, 0.),
    chest_red=((.40, .025, .018), .38, .35), punch_tip=((.20, .20, .21), .50, .60), aid_green=((.02, .35, .12), .45, 0.),
    # drawn-only debris (DEBRIS_LEDGER): concrete dust ("dust" above) is a light grey, linear albedo 0.33: the unweathered
    # cement paste inside the wall, as powder (AUTHORED from memory of grey cement powder's reflectance, 0.3 to 0.45;
    # not measured here. The wall texture's own measured mean, the weathered tan surface, is 0.174, 0.150, 0.108: the
    # first value, 1.5 x that, could not be told from the wall's lighter bands);
    # fresh-cut chips are bright; fresh fir shavings show the earlywood colour
    # (measured on The Wood Database's flat-sawn Douglas-fir photograph: earlywood linear (.55, .35, .20))
    chip_steel=((.60, .61, .63), .28, 1.), chip_alu=((.80, .81, .83), .24, 1.),
    shaving_fir=((.58, .38, .21), .80, 0.), shaving_ply=((.60, .45, .28), .80, 0.),
    vac_tank=((.035, .036, .040), .50, 0.),
    # Hilti red (the impact wrench's housing): sRGB about (210, 5, 30), linear
    hilti_red=((.64, .0015, .013), .42, 0.),
)
# emissive appearance (indicator LEDs): name -> (diffuse, emission colour, emissive intensity)
EMISSIVE = dict(led_green=((.05, .5, .1), (.15, 1., .25), 800.), led_amber=((.5, .3, .02), (1., .55, .05), 800.),
                led_red=((.5, .03, .02), (1., .06, .04), 800.))
# normal-mapped constant materials: name -> (diffuse, roughness, metallic, Poly Haven texture whose normal map is used,
# tile in metres, bump factor): aluminium tread plate takes the diamond relief of the pinned metal_plate texture
NORMAL_MAPPED = dict(tread_alu=((.55, .56, .58), .34, .9, "metal_plate", 0.42, 1.2))
HIDDEN = {"proxy_hidden", "press_hidden", "collision_hidden"}
WOOD_LIKE = {"douglas_fir", "plywood"}


class Stage:
    """Thin USD authoring helper (barcode_scanning / wine_pouring Stage subset plus implicit primitives)."""

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

    def place(self, g, p, q, scale=None):
        xf = self.UsdGeom.Xformable(g)
        xf.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, p)))
        xf.AddOrientOp().Set(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*map(float, q[:3]))))
        if scale is not None:
            xf.AddScaleOp().Set(self.Gf.Vec3f(*map(float, scale)))

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, uv_interp="vertex"):
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
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, uv_interp).Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
        if mat is not None:
            self.bind(g, mat)
        return g

    def box(self, path, size, mat, p=None, q=None, bevel=.0015, uv_per_m=None):
        from whiteboard.usd_scene import bevel_box
        g = bevel_box(self.stage, path, np.asarray(size, float), min(bevel, float(min(size))*.2))
        if p is not None:
            self.place(g, p, q)
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

    def cylinder(self, path, radius, height, mat, p=None, q=None, axis="Z", capsule=False):
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
            self.place(g, p, (0., 0., 0., 1.))
        if mat is not None:
            self.bind(g, mat)
        return g

    def camera(self, path, eye, target, focal_mm, up=(0, 0, 1), role="diagnostic", aperture=36., aspect=16/9):
        c = self.UsdGeom.Camera.Define(self.stage, path)
        c.AddTransformOp().Set(self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*eye), self.Gf.Vec3d(*target),
                                                            self.Gf.Vec3d(*up)).GetInverse())
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


# ------------------------------------------------------------------------------ workpieces from hole records
def _planar_uv(v, f, per_m=1.):
    """Face-varying UVs by projecting each triangle on the plane of its dominant normal axis."""
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.zeros((len(f), 3, 2), np.float32)
    for a in range(3):
        sel = ax == a
        i, j = [k for k in range(3) if k != a]
        uv[sel] = tri[sel][:, :, [i, j]]*per_m
    return uv.reshape(-1, 2)


def workpiece_uv(d, v, f):
    """Face-varying UVs (metres) of a workpiece mesh: planar per dominant axis; on wood the u coordinate runs along the
    board's longest axis wherever that axis lies in the face, so the grain follows the board's length, and the texture
    tile is centred on the board (the 1.2 m fir tile covers the whole ledger face without a wrap)."""
    if d.material not in ("douglas_fir", "plywood"):
        return _planar_uv(v, f, 1.)
    off = 0.5*TEXTURED.get(getattr(d, "render_material", d.material), (None, 1.))[1]
    # two boards cut from different stock: each board's grain is taken from its own place on the tile (a board shorter
    # than the tile is moved along it by up to the spare length, and across it by a few rings)
    tile = 2*off
    spare = max(tile - 2*float(max(d.half)), 0.)
    hsh = hashlib.sha256(str(d.name).encode()).digest()
    shift = np.array([(hsh[0]/255. - 0.5)*spare, (hsh[1]/255.)*0.37], np.float32) if d.material == "douglas_fir" \
        else np.zeros(2, np.float32)
    v = np.asarray(v, np.float32)
    f = np.asarray(f, np.int32).reshape(-1, 3)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ax = np.argmax(np.abs(n), axis=1)
    order = list(np.argsort(-np.asarray(d.half, float)))
    thin = order[-1]
    uv = np.zeros((len(f), 3, 2), np.float32)
    for a in range(3):
        sel = ax == a
        inplane = [k for k in order if k != a]          # longest first
        uv[sel] = tri[sel][:, :, inplane] + off + shift
        if inplane[1] == thin:                          # a board's edge: the tile's edge-grain band (straight rings)
            uv[sel, :, 1] += 0.3*2*off
    return uv.reshape(-1, 2)


def _cyl(m3, r, z0, z1, n=48):
    """manifold3d cylinder along +z from z0 to z1 (z0 < z1)."""
    return m3.Manifold.cylinder(float(z1 - z0), float(r), float(r), n).translate((0., 0., float(z0)))


def _cone(m3, r0, r1, z0, z1, n=48):
    return m3.Manifold.cylinder(float(z1 - z0), float(r0), float(r1), n).translate((0., 0., float(z0)))


def _frame(a):
    """4x3 affine taking +z onto the hole axis a (columns: t1, t2, a)."""
    from .holes import perp_basis
    t1, t2 = perp_basis(a)
    return np.column_stack([t1, t2, a])


def workpiece_mesh(d):
    """(vertices, triangles, uv, extras) of drillable d: its box minus every hole; extras = [(name, v, f, material)]
    for burr rings, tear-out craters and dust columns, in the body frame."""
    import manifold3d as m3
    h = np.asarray(d.half, float)
    solid = m3.Manifold.cube((2*h).tolist(), True)
    if d.material == "douglas_fir":
        # surfaced timber has eased edges (drawn only: the collider is the plain box). Without them two boards
        # clamped face to face read as one block
        c = _D["TIMBER_EASE_M"]
        ax = int(np.argmax(h))
        a_, b_ = [h[k] for k in range(3) if k != ax]
        side = (a_ + b_ - c)*math.sqrt(2.)
        size = [side, side, side]
        size[ax] = 2*h[ax] + 0.01
        rot = [0., 0., 0.]
        rot[ax] = 45.
        solid = solid ^ m3.Manifold.cube(size, True).rotate(rot)
    cuts = []
    extras = []
    from . import materials as M
    for hole in d.holes:
        a = hole.a/np.linalg.norm(hole.a)
        Rm = _frame(a)
        tool = None
        th = hole.thickness if math.isfinite(hole.thickness) else 0.3
        if hole.kind == "dimple":
            s = hole.stages[0]["s"]
            r = 0.5e-3*hole.stages[0]["d_mm"]
            tool = _cone(m3, r + 2e-4, 1e-6, -2e-4, s)            # 90-degree cone, apex at the dimple's depth
        else:
            for st in hole.stages:
                r = 0.5e-3*st["d_mm"]
                ph = M.point_height(st["d_mm"])*1e-3
                end = st["s"]
                if st["through"]:
                    part = _cyl(m3, r, -0.002, th + 0.002)
                else:
                    cyl_end = max(end - ph, 0.)
                    part = _cyl(m3, r, -0.002, max(cyl_end, 1e-4))
                    if end > 1e-5:
                        part = part + _cone(m3, r, 1e-6, cyl_end, end)
                tool = part if tool is None else tool + part
            if hole.entry_spall_d_m > 0:
                rs = 0.5*hole.entry_spall_d_m
                r0 = 0.5e-3*hole.d_mm
                tool = tool + _cone(m3, rs, r0, -0.002, max(rs - r0, 0.001))
            if hole.exit_fd > 1.01 and math.isfinite(hole.thickness):
                r_out = 0.5e-3*hole.d_mm*hole.exit_fd
                depth = min(0.6*(r_out - 0.5e-3*hole.d_mm) + 0.0015, 0.5*th)
                tool = tool + _cone(m3, 0.5e-3*hole.d_mm, r_out, th - depth, th + 0.002)
            if hole.burr_m > 2e-5 and math.isfinite(hole.thickness) and hole.through:
                r = 0.5e-3*hole.d_mm
                ring = _cyl(m3, r + 0.0006, th, th + hole.burr_m) - _cyl(m3, r, th - 0.001, th + hole.burr_m + 0.001)
                extras.append(("burr", ring, Rm, hole.p0, "burr"))
            if hole.dust_column_m() > 1e-4 and not hole.through:
                fl = hole.full_depth()
                r = 0.5e-3*hole.d_mm*0.98
                col = _cyl(m3, r, max(fl - hole.dust_column_m(), 0.), fl)
                extras.append(("dust", col, Rm, hole.p0, "dust"))
            if hole.scorch_max_c >= P.WOOD_SCORCH_C[0] and d.kind == "wood":
                r = 0.5e-3*hole.d_mm
                sleeve = _cyl(m3, r + 0.0004, 0.002, min(hole.depth, th) - 0.002) - _cyl(m3, r, 0., th)
                extras.append(("scorch", sleeve, Rm, hole.p0, "scorch"))
        if tool is None:
            continue
        T4 = np.zeros((3, 4))
        T4[:, :3] = Rm
        T4[:, 3] = hole.p0
        cuts.append(tool.transform(T4.tolist()))
    for c in cuts:
        solid = solid - c
    mesh = solid.to_mesh()
    v = np.asarray(mesh.vert_properties, np.float32)[:, :3]
    f = np.asarray(mesh.tri_verts, np.int32)
    out_extras = []
    for name, man, Rm, p0, mat in extras:
        T4 = np.zeros((3, 4))
        T4[:, :3] = Rm
        T4[:, 3] = p0
        mm = man.transform(T4.tolist()).to_mesh()
        out_extras.append((name, np.asarray(mm.vert_properties, np.float32)[:, :3], np.asarray(mm.tri_verts, np.int32), mat))
    return v, f, out_extras


def mark_meshes(sc, d):
    """{prim path: (vertices, triangles) or None} for the layout marks on drillable d (export records them in
    sc.mark_prims, body frame): each dye patch and stroke is a thin box with every hole and dimple of d that reaches it
    cut out, as the drill or the punch removes the dye or pencil it goes through."""
    import manifold3d as m3
    out = {}
    for path, centre, size, _ in getattr(sc, "mark_prims", {}).get(d.name, []):
        size = np.asarray(size, float)
        if "/dye_" in path:
            # layout dye is brushed on: overlapping strokes of uneven length, not a printed square
            rng = _debris_rng("dye", d.name, path)
            n_ax = int(np.argmin(size))
            u_ax, v_ax = [k for k in range(3) if k != n_ax]
            # strokes run across the part's narrower in-plane direction, each a band with round ends, a little askew
            if d.half[u_ax] > d.half[v_ax]:
                u_ax, v_ax = v_ax, u_ax
            n = int(_D["DYE_STROKES"])
            w_ = size[v_ax]/n
            reach = min(size[u_ax]*1.5, 2*float(d.half[u_ax]) - 0.003)
            m = None
            for k in range(n):
                L_ = min(size[u_ax]*rng.uniform(0.95, 1.5), reach)
                wd = w_*rng.uniform(1.3, 1.9)
                sz = size.copy()
                sz[u_ax], sz[v_ax] = L_ - wd, wd
                st_ = m3.Manifold.cube(sz.tolist(), True)
                for e in (-1., 1.):
                    cyl = m3.Manifold.cylinder(float(size[n_ax]), 0.5*wd, 0.5*wd, 24, True)
                    if n_ax != 2:
                        r_ = [0., 0., 0.]
                        r_[1 if n_ax == 0 else 0] = 90.
                        cyl = cyl.rotate(r_)
                    off_ = np.zeros(3)
                    off_[u_ax] = e*0.5*(L_ - wd)
                    st_ = st_ + cyl.translate(tuple(float(x) for x in off_))
                ang = [0., 0., 0.]
                ang[n_ax] = float(rng.uniform(-7., 7.))
                ct = np.zeros(3)
                ct[u_ax] = float(np.clip(size[u_ax]*rng.uniform(-0.10, 0.10), -0.5*(reach - L_), 0.5*(reach - L_)))
                ct[v_ax] = (k + 0.5)*w_ - 0.5*size[v_ax] + w_*rng.uniform(-0.2, 0.2)
                st_ = st_.rotate(ang).translate(tuple(float(x) for x in ct))
                m = st_ if m is None else m + st_
            m = m.translate(tuple(float(x) for x in centre))
        else:
            m = m3.Manifold.cube(size.tolist(), True).translate(tuple(float(x) for x in centre))
        for hole in d.holes:
            if hole.kind == "dimple":
                r = 0.5e-3*hole.stages[0]["d_mm"] + 1e-4
            else:
                r = max(0.5e-3*hole.d_mm, 0.5*hole.entry_spall_d_m) + 1e-4
            a = hole.a/np.linalg.norm(hole.a)
            T4 = np.zeros((3, 4))
            T4[:, :3] = _frame(a)
            T4[:, 3] = hole.p0
            m = m - _cyl(m3, r, -0.01, 0.01).transform(T4.tolist())
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties, np.float32)[:, :3]
        f = np.asarray(mesh.tri_verts, np.int32)
        out[path] = (v, f) if len(f) else None
    return out


def hole_state_key(d):
    s = json.dumps([[h.kind, [[round(x, 5) for x in h.p0], [round(x, 5) for x in h.a]],
                     [[float(st["d_mm"]), round(float(st["s"]), 5), bool(st["through"])] for st in h.stages],
                     round(float(h.burr_m), 5), round(float(h.exit_fd), 3), round(float(h.entry_spall_d_m), 5),
                     round(float(h.dust_column_m()), 4), bool(h.scorch_max_c >= P.WOOD_SCORCH_C[0]),
                     round(float(h.removed_mm3)), round(float(dust_out_mm3(h)))]
                    for h in d.holes], sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()


# ------------------------------------------------------------------------------ debris (drawn only)
# Drilling debris is DRAWN ONLY: it has no collider, no mass and no effect on the physics, and nothing here is read by
# the simulation. Every particle comes from the hole records the physics keeps (holes.Hole: removed volume, dust
# produced / evacuated / loose / packed, stages, the workpiece material), so the amount on screen is the amount the
# physics removed. Each hole's particles are drawn from a fixed random sequence seeded by (workpiece, hole id): a
# re-render of a recording gives the same frames, and a growing hole adds particles to the end of the same sequence
# (nothing already drawn moves). Particles come to rest on the colliders below them (a vertical ray cast against the
# static shapes, the drillables and the vise at that frame's poses) and are drawn in the frame of the body they rest
# on, so chips on a bar travel with the bar.
#
# Appearance constants (visual only; they are not physics parameters, so they live here and not in params.py).
# name: (value, unit, label, basis).
DEBRIS_LEDGER = dict(
    TIMBER_EASE_M=(0.0025, "m", "authored", "eased edge drawn on solid timber (surfaced lumber's edges are rounded "
                                            "by about 1/8 in; drawn as a chamfer on the four long edges)"),
    DYE_STROKES=(6, "-", "authored", "brush strokes a patch of layout dye is drawn with (ragged ends, uneven sides)"),
    DUST_TRAIL_SHARE=(0.03, "-", "authored", "share of the dust that left a hole clinging to the face below its mouth"),
    DUST_TRAIL_LENGTH_M=(0.09, "m", "authored", "mean length of the grey streak below the mouth (exponential)"),
    DUST_HALO_SHARE=(0.2, "-", "authored", "share of the clinging dust in a ring around the mouth"),
    DUST_FLECK_R_M=((0.10e-3, 0.35e-3), "m", "authored",
                    "radius range of a drawn dust grain cluster: below a pixel at policy-camera range, so the deposit "
                    "renders as a soft haze whose density is the deposit's"),
    DUST_FLECK_T_M=(0.04e-3, "m", "authored", "deposit thickness a wall fleck stands for (sets the drawn area)"),
    DUST_FLECK_R_MAX_M=(2.0e-3, "m", "authored", "largest radius a fleck grows to once the drawing budget is used up"),
    DUST_TRAIL_WIDEN=(0.12, "m/m", "authored", "the streak's half-width grows by this per metre below the mouth"),
    DUST_PILE_CORE=(0.7, "-", "authored", "share of the fallen dust in the heap; the rest is scattered around it"),
    DUST_PILE_SIGMA_M=((0.010, 0.008), "m, m/m", "authored", "heap radius (Gaussian sigma) = a + b x fall height"),
    DUST_SPRAY_SIGMA_M=((0.018, 0.022), "m, m/m", "authored", "scatter radius around the heap = a + b x fall height"),
    DUST_SPRAY_T_M=(0.08e-3, "m", "authored", "deposit thickness a floor fleck stands for"),
    CHIP_T_M=(0.15e-3, "m", "estimate", "metal chip thickness ~ feed per lip (0.05 mm at 0.1 mm/rev) x chip "
                                         "compression ratio ~3 (Machinery's Handbook feeds; cutting-ratio literature)"),
    CHIP_W_D=(0.55, "x d", "derived", "chip width ~ lip length (d/2)/sin 59 deg = 0.58 d for a 118 deg point"),
    CHIP_TURNS=((1.2, 4.5), "-", "authored", "turns of a broken helical chip"),
    CHIP_COIL_R_D=((0.22, 0.45), "x d", "authored", "coil radius of a helical chip"),
    CHIP_SPREAD_M=(0.022, "m", "authored", "mean distance a chip lands beyond the hole's rim (exponential)"),
    SHAVING_T_M=(0.3e-3, "m", "authored", "wood shaving thickness"),
    SHAVING_W_D=(0.45, "x d", "authored", "shaving width"),
    SHAVING_LEN_M=((0.006, 0.026), "m", "authored", "shaving length range"),
    SHAVING_CURL_R_M=((0.0015, 0.0045), "m", "authored", "shaving curl radius range"),
    SHAVING_SPREAD_M=((0.022, 0.07, 0.3), "m, m, -", "authored",
                      "mean distance beyond the rim: near heap (exponential), far throw, share thrown far"),
    PILE_BULK=(dict(chip=8., shaving=4.), "x solid", "authored",
               "bulk volume of a loose pile per solid volume (loose helical chips, loose curled shavings): each "
               "particle raises the pile under its footprint by its bulk volume / footprint area"),
    MAX_PARTICLES=(dict(fleck=20000, spray=20000, chip=400, shaving=900), "-", "authored", "drawing budget per hole"),
)
_D = {k: v[0] for k, v in DEBRIS_LEDGER.items()}
DEBRIS_MATERIAL = dict(concrete="dust", steel="chip_steel", aluminium="chip_alu", douglas_fir="shaving_fir",
                       plywood="shaving_ply")


def dust_vacuumed_mm3(h):
    """Dust the vacuum nozzle took out of hole h (a vacuum pass empties the hole's own column into the vacuum)."""
    v = getattr(h, "dust_vacuumed_mm3", None)
    if v is not None:
        return float(v)
    return float(sum(e.get("removed_mm3", 0.) for e in h.events if e.get("event") == "vacuum_pass"))


def dust_out_mm3(h):
    """Bulk dust that left concrete hole h at its mouth and fell (not into the vacuum): the evacuated dust less what
    the vacuum took. The removal law (dust produced = DUST_BULKING x removed) bounds it from below, which covers
    authored start-state holes whose records carry the removed volume and the column but no produced dust."""
    inside = h.dust_loose_mm3 + h.dust_packed_mm3 + h.dust_transit_mm3
    out = max(h.dust_evacuated_mm3, P.DUST_BULKING*h.removed_mm3 - inside) - dust_vacuumed_mm3(h)
    return max(out, 0.)


def _debris_rng(*key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256("/".join(map(str, key)).encode()).digest()[:8], "little"))


def _q7(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[3:7]).as_matrix(), q[:3]


def _body_tf(bq, body):
    if body < 0:
        return np.eye(3), np.zeros(3)
    return _q7(bq[body])


class Supports:
    """Vertical ray casts against the colliders a falling chip can rest on: static shapes, the drillables and the
    vises (boxes, cylinders, capsules; the floor plane at z = 0). Robot links, tools, items and proxies are skipped."""

    def __init__(self, world):
        import newton
        sc = world.scene
        b = sc.b
        labels = list(b.body_label)
        allowed = {-1}
        for d in sc.drillables.values():
            allowed.add(d.body)
            if getattr(d, "extra_body", None) is not None:
                allowed.add(d.extra_body)
        vises = [n for n, t in sc.tools.items() if type(t).__name__ == "Vise"]
        for k, lab in enumerate(labels):
            if any(lab.startswith(n + "_") for n in vises) or lab == "vacuum_unit":
                allowed.add(k)
        G = newton.GeoType
        types = {int(G.BOX): "box", int(G.CYLINDER): "cyl", int(G.CAPSULE): "cap"}
        visual = getattr(sc, "visual", set())
        self.shapes = []
        for i in range(b.shape_count):
            body = int(b.shape_body[i])
            t = types.get(int(b.shape_type[i]))
            mname = sc.materials.get(i, "robot")
            if t is None or body not in allowed or i in visual or mname in ("robot", "proxy_hidden"):
                continue
            self.shapes.append((i, body, t, np.asarray(b.shape_scale[i], float), np.asarray(b.shape_transform[i], float)))

    def down(self, origins, bq):
        """origins (M, 3) world -> (z of the first collider surface below each, body it belongs to); the floor (z 0,
        body -1) when nothing else is hit. Shapes containing an origin are ignored for that ray."""
        O = np.atleast_2d(np.asarray(origins, float))
        M = len(O)
        best_t = O[:, 2].copy()
        best_b = np.full(M, -1, int)
        for i, body, t, s, tf in self.shapes:
            Rb, tb = _body_tf(bq, body)
            Rs, ts = _q7(tf)
            R = Rb @ Rs
            c = Rb @ ts + tb
            o = (O - c) @ R
            dl = -R[2, :]                                            # world -z in the shape frame
            if t == "box":
                h = s[:3]
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv = np.where(np.abs(dl) > 1e-12, 1./np.where(np.abs(dl) > 1e-12, dl, 1.), 0.)
                    t1 = (-h - o)*inv
                    t2 = (h - o)*inv
                lo = np.where(np.abs(dl) > 1e-12, np.minimum(t1, t2), np.where(np.abs(o) <= h, -np.inf, np.inf))
                hi = np.where(np.abs(dl) > 1e-12, np.maximum(t1, t2), np.where(np.abs(o) <= h, np.inf, -np.inf))
                te, tx = lo.max(1), hi.min(1)
                hit = (te <= tx) & (te >= 0.)
                tt = np.where(hit, te, np.inf)
            else:
                r, hh = float(s[0]), float(s[1]) + (float(s[0]) if t == "cap" else 0.)
                inside = (o[:, 0]**2 + o[:, 1]**2 <= r*r) & (np.abs(o[:, 2]) <= hh)
                cand = np.full(M, np.inf)
                a = dl[0]**2 + dl[1]**2
                if a > 1e-12:
                    bb = 2*(o[:, 0]*dl[0] + o[:, 1]*dl[1])
                    cc = o[:, 0]**2 + o[:, 1]**2 - r*r
                    disc = bb*bb - 4*a*cc
                    ok = disc >= 0
                    tr = (-bb - np.sqrt(np.where(ok, disc, 0.)))/(2*a)
                    zr = o[:, 2] + tr*dl[2]
                    good = ok & (tr >= 0) & (np.abs(zr) <= hh)
                    cand = np.where(good, tr, cand)
                if abs(dl[2]) > 1e-12:
                    for zc in (hh, -hh):
                        tc = (zc - o[:, 2])/dl[2]
                        xy = (o[:, 0] + tc*dl[0])**2 + (o[:, 1] + tc*dl[1])**2
                        good = (tc >= 0) & (xy <= r*r)
                        cand = np.where(good, np.minimum(cand, tc), cand)
                tt = np.where(inside, np.inf, cand)
            better = tt < best_t
            best_t = np.where(better, tt, best_t)
            best_b = np.where(better, body, best_b)
        return O[:, 2] - best_t, best_b


def _supports(world):
    s = getattr(world, "_render_supports", None)
    if s is None:
        s = Supports(world)
        world._render_supports = s
    return s


def visible_mouth(world, d, h, bq):
    """(mouth point, outward face normal, drillable whose face it is), world frame: the hole's entry point, carried
    back along the hole axis through any drillable that covers it (a hole in a lower board opens at the top of the
    board above it; a wall hole behind a ledger opens on the ledger's face)."""
    R, t = _body_tf(bq, d.body)
    p = R @ np.asarray(h.p0, float) + t
    a = R @ (np.asarray(h.a, float)/np.linalg.norm(h.a))
    n = R @ np.asarray(h.face_normal, float)
    face = d
    for _ in range(4):
        q = p - a*1e-3
        nxt = None
        for e in world.hm.drillables.values():
            if e is face:
                continue
            Re, te = _body_tf(bq, e.body)
            ql = Re.T @ (q - te)
            if np.all(np.abs(ql) <= e.half + 1e-6):
                al = Re.T @ (-a)
                with np.errstate(divide="ignore"):
                    ts = np.where(np.abs(al) > 1e-12, (np.sign(al)*e.half - ql)/np.where(np.abs(al) > 1e-12, al, 1.), np.inf)
                k = int(np.argmin(ts))
                nl = np.zeros(3)
                nl[k] = np.sign(al[k])
                nxt = (q - a*float(ts[k]), Re @ nl, e)
                break
        if nxt is None:
            break
        p, n, face = nxt
    return p, n, face


def _face_frame(n):
    """Two unit vectors spanning the plane with normal n: `down` (steepest descent in it, or any for a level face)
    and `side`."""
    n = n/np.linalg.norm(n)
    g = np.array([0., 0., -1.]) - n*(-n[2])
    if np.linalg.norm(g) < 1e-6:
        g = np.cross(n, [1., 0., 0.]) if abs(n[0]) < 0.9 else np.cross(n, [0., 1., 0.])
    g /= np.linalg.norm(g)
    return g, np.cross(n, g)


def _on_face(e, bq, pts, margin=0.):
    """Mask of world points lying within drillable e's box (+margin) - i.e. on its face when they sit on it."""
    R, t = _body_tf(bq, e.body)
    loc = (pts - t) @ R
    return np.all(np.abs(loc) <= e.half + margin, axis=1)


def _grains(c, n, r, ang):
    """Fine dust: one triangle per grain cluster (centres c (M, 3), plane normal n, circumradius r (M,)) -> (v, f)."""
    M = len(c)
    if M == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    n = np.asarray(n, float)
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    a = ang[:, None] + np.arange(3)[None, :]*2*math.pi/3
    # each corner at its own distance (0.7 to 1.3 r, a fixed function of the grain's angle): scalene flakes, so that a
    # deposit seen from close does not read as a field of equal triangles
    k = 0.7 + 0.6*np.mod(np.sin(ang[:, None]*(np.arange(3)[None, :] + 1.)*12.9898)*43758.5453, 1.)
    v = c[:, None, :] + (r[:, None]*k)[..., None]*(np.cos(a)[..., None]*u + np.sin(a)[..., None]*w)
    return v.reshape(-1, 3).astype(np.float32), np.arange(3*M, dtype=np.int32).reshape(-1, 3)


def _fleck_radii(r, volume, thickness):
    """(count k, radii) of the flecks that draw a deposit of `volume` at `thickness`: the flecks' drawn area is the
    deposit's area. Up to the drawing budget the deposit grows by adding flecks of the base radii r; beyond it every
    fleck grows instead (to at most DUST_FLECK_R_MAX_M), so that the drawn area still follows the dust."""
    area = 0.75*math.sqrt(3.)*r**2                                 # a triangle of circumradius r
    need = volume/thickness
    cum = np.cumsum(area)
    k = int(np.searchsorted(cum, need))
    if k < len(r):
        return k, r[:k]
    scale = min(math.sqrt(need/float(cum[-1])), _D["DUST_FLECK_R_MAX_M"]/float(r.max()))
    return len(r), r*scale


def _ribbon(centre, across, width):
    """Strip mesh along a centre polyline (S, 3) with an across-direction per point (S, 3) -> (v, f)."""
    S = len(centre)
    a = across/np.linalg.norm(across, axis=1, keepdims=True)
    v = np.concatenate([centre - 0.5*width*a, centre + 0.5*width*a], 0)
    i = np.arange(S - 1)
    f = np.concatenate([np.stack([i, i + 1, S + i + 1], 1), np.stack([i, S + i + 1, S + i], 1)], 0)
    return v, f


def _chip_shape(rng_row, d_m, kind):
    """One particle's mesh in its own frame (lowest point at z 0, centred in x, y) and its solid volume (m^3).
    kind: chip (a broken helical metal chip) or shaving (a curled wood shaving)."""
    u = rng_row
    if kind == "chip":
        w = _D["CHIP_W_D"]*d_m
        turns = _D["CHIP_TURNS"][0] + u[0]*(_D["CHIP_TURNS"][1] - _D["CHIP_TURNS"][0])
        R0 = d_m*(_D["CHIP_COIL_R_D"][0] + u[1]*(_D["CHIP_COIL_R_D"][1] - _D["CHIP_COIL_R_D"][0]))
        cone = 0.15 + 0.6*u[2]                       # a drill's chip coils as a cone: its radius grows along it
        pitch = w*(0.9 + 1.6*u[3])
        S = max(8, int(turns*14))
        s = np.linspace(0., 2*math.pi*turns, S)
        Rr = R0*(1. + cone*s/s[-1])
        centre = np.stack([pitch*s/(2*math.pi), Rr*np.cos(s), Rr*np.sin(s)], 1)
        radial = np.stack([np.zeros(S), np.cos(s), np.sin(s)], 1)
        axial = np.tile([1., 0., 0.], (S, 1))
        phi = 0.9 + 0.4*u[4]
        across = math.cos(phi)*axial + math.sin(phi)*radial
        length = float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))
        vol = w*_D["CHIP_T_M"]*length
    else:
        w = _D["SHAVING_W_D"]*d_m*(0.7 + 0.6*u[0])
        L = _D["SHAVING_LEN_M"][0] + u[1]*(_D["SHAVING_LEN_M"][1] - _D["SHAVING_LEN_M"][0])
        rc = _D["SHAVING_CURL_R_M"][0] + u[2]*(_D["SHAVING_CURL_R_M"][1] - _D["SHAVING_CURL_R_M"][0])
        ang = min(L/rc, 2*math.pi*(0.6 + 0.9*u[3]))  # a curl of up to ~1.5 turns; a long shaving stays an open arc
        S = max(6, int(ang*5))
        s = np.linspace(0., ang, S)
        lead = w*0.6*u[4]*s/(2*math.pi)              # a slight helix: a shaving rarely closes on itself
        centre = np.stack([lead, rc*np.sin(s), rc*(1. - np.cos(s))], 1)
        across = np.tile([1., 0., 0.], (S, 1)) + np.stack([np.zeros(S), 0.25*(u[5] - 0.5)*np.ones(S), np.zeros(S)], 1)
        length = rc*ang
        vol = w*_D["SHAVING_T_M"]*length
    v, f = _ribbon(centre, across, w)
    # a random attitude: lying with its long axis roughly level, at any heading
    Rm = Rotation.from_euler("zyx", [2*math.pi*u[6], 0.2*(u[7] - 0.5), 2*math.pi*u[8]]).as_matrix()
    v = v @ Rm.T
    v -= np.array([0.5*(v[:, 0].min() + v[:, 0].max()), 0.5*(v[:, 1].min() + v[:, 1].max()), v[:, 2].min()])
    return v, f, vol, float(v[:, 2].max())


def _heap(centre, z0, sigma, volume, rng, rings=22, sectors=44):
    """Lumpy Gaussian dust heap of `volume` (m^3) on a level support at z0 -> (v, f, vertex normals)."""
    rmax = 2.6*sigma
    rr = np.linspace(0., rmax, rings + 1)[1:]
    th = 2*math.pi*np.arange(sectors)/sectors
    wob = (1. + 0.22*np.sin(2*th + rng.uniform(0, 6.3)) + 0.14*np.sin(5*th + rng.uniform(0, 6.3))
           + 0.08*np.sin(9*th + rng.uniform(0, 6.3)))
    R_, T_ = np.meshgrid(rr, th, indexing="ij")
    rs = R_/np.interp(T_, th, wob, period=2*math.pi)
    X = centre[0] + R_*np.cos(T_)
    Y = centre[1] + R_*np.sin(T_)
    lump = 1. + sum(a*np.sin((X - centre[0])*k*math.cos(q)/sigma + (Y - centre[1])*k*math.sin(q)/sigma + ph)
                    for a, k, q, ph in ((0.22, 1.7, rng.uniform(0, 6.3), rng.uniform(0, 6.3)),
                                        (0.14, 3.1, rng.uniform(0, 6.3), rng.uniform(0, 6.3)),
                                        (0.08, 6.3, rng.uniform(0, 6.3), rng.uniform(0, 6.3))))
    shape = np.maximum(np.exp(-0.5*(rs/sigma)**2) - math.exp(-0.5*2.6**2), 0.)*lump
    shape[-1] = 0.
    # scale to the volume: integrate the profile over the polar grid (ring areas)
    dr = rr[1] - rr[0]
    area = 2*math.pi*rr*dr/sectors
    h0 = volume/float(np.sum(shape*area[:, None]) + 1e-18)
    centre_z = z0 + h0*(1 - math.exp(-0.5*2.6**2))
    v = np.concatenate([[[centre[0], centre[1], centre_z]], np.stack([X, Y, z0 + h0*shape], -1).reshape(-1, 3)], 0)
    f = []
    for j in range(sectors):
        f.append([0, 1 + j, 1 + (j + 1) % sectors])
    for i in range(rings - 1):
        for j in range(sectors):
            a, b = 1 + i*sectors + j, 1 + i*sectors + (j + 1) % sectors
            f += [[a, a + sectors, b + sectors], [a, b + sectors, b]]
    v = v.astype(np.float32)
    f = np.asarray(f, np.int32)
    import trimesh
    nrm = trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)
    return v, f, nrm


def _pile(grid, lo, hi, z_support, bulk_m3, cell=0.003):
    """Rest a particle whose footprint is the xy box [lo, hi] on its support, or on the loose pile already under its
    footprint (its mean height there: particles nest into a loose pile); then add its bulk volume to the pile under
    the footprint. `grid`: a coarse heightfield (absolute z) shared by a drillable's particles."""
    i0, j0 = int(math.floor(lo[0]/cell)), int(math.floor(lo[1]/cell))
    i1, j1 = int(math.floor(hi[0]/cell)), int(math.floor(hi[1]/cell))
    cells = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    base = max(z_support, float(np.mean([max(grid.get(c, z_support), z_support) for c in cells])))
    rise = bulk_m3/(len(cells)*cell*cell)
    for c in cells:
        grid[c] = max(grid.get(c, z_support), z_support) + rise
    return base


def hole_debris(world, d, h, bq, grid):
    """Drawn debris of hole h of drillable d at body poses bq: [(body it rests on, kind, v (world), f, material,
    vertex normals or None)]. concrete: grey dust clinging below the mouth and a heap with scattered clumps where the
    dust that left the hole fell; steel / aluminium: curled chips; timber / plywood: curled shavings."""
    if h.kind != "drilled" or h.d_mm <= 0:
        return []
    out = []
    mat = DEBRIS_MATERIAL.get(d.material)
    if mat is None:
        return out
    r_hole = 0.5e-3*h.d_mm
    mouth, n, face = visible_mouth(world, d, h, bq)
    sup = _supports(world)
    level = n[2] > 0.7
    if d.material == "concrete":
        V = dust_out_mm3(h)*1e-9                                   # m^3 (bulk)
        if V <= 0.:
            return out
        # one fixed-length random sequence per component, so a growing amount only appends particles
        rng, rng_h, rng_c = (_debris_rng(d.name, h.id, k) for k in ("flecks", "heap", "spray"))
        g, side = _face_frame(n)
        rlo, rhi = _D["DUST_FLECK_R_M"]
        # clinging dust: a deposit on the face below the mouth (a ring at the mouth, a streak widening downwards)
        NF = _D["MAX_PARTICLES"]["fleck"]
        U = rng.random((NF, 5))
        Z = np.clip(rng.standard_normal(NF), -2.5, 2.5)
        k, rf = _fleck_radii(rlo + U[:, 0]*(rhi - rlo), _D["DUST_TRAIL_SHARE"]*V, _D["DUST_FLECK_T_M"])
        if k:
            U, Z = U[:k], Z[:k]
            halo = U[:, 1] < _D["DUST_HALO_SHARE"]
            dz = -_D["DUST_TRAIL_LENGTH_M"]*np.log(1. - 0.95*U[:, 2])                  # at most 3 lengths
            dx = (0.6*r_hole + _D["DUST_TRAIL_WIDEN"]*dz)*Z
            ra = r_hole + 0.2e-3 + 0.006*U[:, 2]**2
            th = 2*math.pi*U[:, 3]
            off_g = np.where(halo, ra*np.sin(th), dz + r_hole*0.6)
            off_s = np.where(halo, ra*np.cos(th), dx)
            pts = mouth[None, :] + off_g[:, None]*g[None, :] + off_s[:, None]*side[None, :] + n[None, :]*1.0e-4
            keep = (np.hypot(off_g, off_s) > r_hole + 0.3e-3) & _on_face(face, bq, pts - n[None, :]*1.5e-4, 1e-4)
            fv, ff = _grains(pts[keep], n, rf[keep], 2*math.pi*U[keep, 4])
            out.append((face.body, "dust", fv, ff, "dust", None))
        # what fell: a lumpy heap with a scatter of fine dust around it, on the first support below the mouth
        drop = mouth + n*0.012
        zs, bs = sup.down(drop[None, :] - np.array([[0., 0., 0.004]]), bq)
        fall = max(float(mouth[2] - zs[0]), 0.)
        sig = _D["DUST_PILE_SIGMA_M"][0] + _D["DUST_PILE_SIGMA_M"][1]*fall
        heap_c, heap_v = np.array([drop[0], drop[1]]), _D["DUST_PILE_CORE"]*V
        if abs(n[2]) < 0.3:
            # under a mouth in a wall the heap lies against the wall's foot: a half heap. It is drawn as a whole one
            # centred on the face with twice the volume, of which the wall hides the half behind the face (centred
            # 12 mm out, as it was, the wall hid nearly half of the dust)
            heap_c, heap_v = np.array([mouth[0], mouth[1]]), 2.*heap_v
        hv, hf, hn = _heap(heap_c, float(zs[0]) + 2e-4, sig, heap_v, rng_h)
        out.append((int(bs[0]), "dustpile", hv, hf, "dust", hn))
        NC = _D["MAX_PARTICLES"]["spray"]
        C = rng_c.random((NC, 4))
        k, rc = _fleck_radii(rlo + C[:, 0]*(rhi - rlo), (1. - _D["DUST_PILE_CORE"])*V, _D["DUST_SPRAY_T_M"])
        if k:
            spr = _D["DUST_SPRAY_SIGMA_M"][0] + _D["DUST_SPRAY_SIGMA_M"][1]*fall
            rim = C[:k, 1] < 0.45                                   # nearly half the scatter frays the heap's rim
            rr = np.where(rim, sig*(1.4 + 1.6*C[:k, 3]), 0.6*sig + spr*np.sqrt(-2*np.log(1. - 0.999*C[:k, 1])))
            th = 2*math.pi*C[:k, 2]
            xy = np.stack([drop[0] + rr*np.cos(th), drop[1] + rr*np.sin(th)], 1)
            nh = np.array([n[0], n[1]])
            if np.linalg.norm(nh) > 0.3:                           # dust falls in front of the face, never behind it
                nh = nh/np.linalg.norm(nh)
                behind = (xy - drop[None, :2]) @ nh
                xy = xy - 2*np.minimum(behind, 0.)[:, None]*nh[None, :]
            org = np.column_stack([xy, np.full(k, drop[2] - 0.004)])
            z, bb = sup.down(org, bq)
            pts = np.column_stack([xy, z + 1.5e-4])
            for body in np.unique(bb):
                m_ = bb == body
                fv, ff = _grains(pts[m_], np.array([0., 0., 1.]), rc[:k][m_], 2*math.pi*C[:k, 3][m_])
                out.append((int(body), "spray", fv, ff, "dust", None))
        return out
    # metal chips and wood shavings: curled ribbons around the mouth
    kind = "chip" if d.kind == "metal" else "shaving"
    Vs = h.removed_mm3*1e-9
    if Vs <= 0.:
        return out
    rng = _debris_rng(d.name, h.id, kind)
    N = _D["MAX_PARTICLES"][kind]
    U = rng.random((N, 13))
    shapes, cum = [], 0.
    for row in U:
        v, f, vol, ht = _chip_shape(row[:9], 2*r_hole, kind)
        if cum + vol > Vs and shapes:
            break
        cum += vol
        shapes.append((v, f, vol))
        if cum >= Vs:
            break
    M = len(shapes)
    U = U[:M]
    g, side = _face_frame(n)
    if kind == "chip":
        dist = r_hole + 0.002 + _D["CHIP_SPREAD_M"]*(-np.log(1. - 0.999*U[:, 9]))
    else:
        near, far, share = _D["SHAVING_SPREAD_M"]
        dist = r_hole + 0.001 + np.where(U[:, 11] < share, far, near)*(-np.log(1. - 0.999*U[:, 9]))
    th = 2*math.pi*U[:, 10]
    if level:
        pts = mouth[None, :] + dist[:, None]*(np.cos(th)[:, None]*g[None, :] + np.sin(th)[:, None]*side[None, :])
    else:
        # a mouth in a vertical (or downward) face: they fall from in front of it
        out_d = 0.005 + 0.04*U[:, 12]
        pts = mouth[None, :] + n[None, :]*out_d[:, None] + (dist - r_hole)[:, None]*np.cos(th)[:, None]*side[None, :]
    on = _on_face(face, bq, pts - np.array([[0., 0., 1e-4]]), 0.) if level else np.zeros(M, bool)
    z = np.full(M, mouth[2])
    bb = np.full(M, face.body)
    if (~on).any():
        zz, bo = sup.down(pts[~on] + np.array([[0., 0., 0.05 if level else -0.002]]), bq)
        z[~on], bb[~on] = zz, bo
    bulk = _D["PILE_BULK"][kind]
    parts = {}
    for k_, (v, f, vol) in enumerate(shapes):
        lo_, hi_ = v[:, :2].min(0) + pts[k_, :2], v[:, :2].max(0) + pts[k_, :2]
        base = _pile(grid, lo_, hi_, z[k_], bulk*vol)
        parts.setdefault(int(bb[k_]), []).append((v + np.array([pts[k_, 0], pts[k_, 1], base + 5e-5]), f))
    for body, lst in parts.items():
        vs, fs, off = [], [], 0
        for v, f in lst:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        out.append((body, kind, np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32), mat, None))
    return out


def drillable_debris(world, d, bq):
    """{prim path: mesh entry} of every drawn debris particle of drillable d, each group in the frame of the body it
    rests on (paths /Lab/body_<b>/debris_<d>_<kind>, or /Lab/debris_<d>_<kind> on static shapes)."""
    grid = {}
    groups = {}
    for h in d.holes:
        for body, kind, v, f, mat, nrm in hole_debris(world, d, h, bq, grid):
            if not len(f):
                continue
            groups.setdefault((body, kind, mat), []).append((v, f, nrm))
    out = {}
    for (body, kind, mat), lst in groups.items():
        R, t = _body_tf(bq, body)
        vs, fs, ns, off = [], [], [], 0
        for v, f, nrm in lst:
            vs.append((np.asarray(v, float) - t) @ R)
            fs.append(np.asarray(f) + off)
            ns.append(None if nrm is None else np.asarray(nrm, float) @ R)
            off += len(v)
        path = f"/Lab/body_{body}/debris_{d.name}_{kind}" if body >= 0 else f"/Lab/debris_{d.name}_{kind}"
        entry = dict(v=np.concatenate(vs).astype(np.float32), t=np.concatenate(fs).astype(np.int32), mat=mat,
                     double=True)
        if all(x is not None for x in ns):
            entry["n"] = np.concatenate(ns).astype(np.float32)
        out[path] = entry
    return out


def drillable_meshes(world, d, bq):
    """{prim path: mesh} drawn from drillable d's current hole records: its workpiece (box minus holes), the layout
    marks on it, burr rings, dust columns, scorch sleeves and the debris (drillable_debris)."""
    out = {}
    v, f, extras = workpiece_mesh(d)
    out[f"/Lab/body_{d.body}/workpiece"] = dict(v=v, t=f, uv=workpiece_uv(d, v, f))
    out.update(mark_meshes(world.scene, d))
    for kind in ("burr", "dust", "scorch"):
        parts = [(ev, ef) for (nm, ev, ef, mat) in extras if nm == kind]
        if parts:
            vs, fs, off = [], [], 0
            for ev, ef in parts:
                vs.append(ev)
                fs.append(ef + off)
                off += len(ev)
            out[f"/Lab/body_{d.body}/{kind}"] = (np.concatenate(vs), np.concatenate(fs))
        else:
            out[f"/Lab/body_{d.body}/{kind}"] = None
    out.update(drillable_debris(world, d, bq))
    return out


class WorkpieceCache:
    """Per-drillable mesh cache keyed by the hole state; returns the meshes to send with a render request. `holes`
    ({drillable: hole records}, a recording's frame) replaces the world's own records for the call; debris rests on
    the colliders at the poses `bq` (default: the world's). Debris prims no longer present are sent as None."""

    def __init__(self, world):
        self.world = world
        self.keys = {}
        self.debris_paths = {}

    def changed(self, force=False, holes=None, bq=None):
        out = {}
        bq = self.world.bq if bq is None else np.asarray(bq, float)
        for name, d in self.world.hm.drillables.items():
            saved = d.holes
            if holes is not None:
                d.holes = holes[name]
            try:
                k = hole_state_key(d)
                if not (force or self.keys.get(name) != k):
                    continue
                self.keys[name] = k
                m = drillable_meshes(self.world, d, bq)
            finally:
                d.holes = saved
            paths = {p for p in m if "/debris_" in p}
            for p in self.debris_paths.get(name, set()) - paths:
                m[p] = None
            self.debris_paths[name] = paths
            out.update(m)
        return out


def encode_meshes(meshes):
    """Mesh entries for a render request: (v, t) tuples, or dicts {v, t[, n vertex normals][, mat: material created
    on demand under /Looks][, double: double-sided]} for prims the worker creates when they are missing (debris, the
    vacuum hose); None hides a prim."""
    def b64(a, dt):
        return base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()
    out = {}
    for path, m in meshes.items():
        if isinstance(m, dict):
            v, t = m["v"], m["t"]
        elif m is not None:
            v, t = m
        if m is None or not len(v) or not len(t):
            out[path] = None
            continue
        entry = dict(v=b64(v, np.float32), t=b64(t, np.int32))
        if isinstance(m, dict):
            if m.get("n") is not None:
                entry["n"] = b64(m["n"], np.float32)
            if m.get("uv") is not None:
                entry["uv"] = b64(m["uv"], np.float32)
            for k in ("mat", "double"):
                if m.get(k) is not None:
                    entry[k] = m[k]
        if path.endswith("/workpiece") and "uv" not in entry:
            uv = _planar_uv(np.asarray(v, np.float32), np.asarray(t, np.int32), 1.)
            entry["uv"] = b64(uv, np.float32)
        out[path] = entry
    return out


# ------------------------------------------------------------------------------ export
# name -> (texture, tile in metres, roughness). Douglas fir (the ledger and the timber boards) uses a flat-sawn softwood
# texture generated from measured Douglas-fir colours and ring widths (visual_assets.PROCEDURAL; a plywood veneer
# reads as a reddish hardwood on the 1.2 m ledger). "concrete" is a concrete test block of the component benches
# (drawn untextured it was a flat grey plane in the E9 film).
TEXTURED = {"concrete_wall": ("concrete", 2.4, .9), "floor_concrete": ("concrete_floor_worn_001", 3.0, .7),
            "block_paint": ("painted_plaster_wall", 2.5, .85), "plywood": ("plywood", 1.2, .7),
            "douglas_fir": ("douglas_fir_flatsawn", 0.6, .62), "bench_blue": ("blue_metal_plate", 1.0, .5),
            "ceiling": ("painted_plaster_wall", 3.0, .9), "hickory": ("plywood", 0.6, .55),
            "concrete": ("concrete", 2.4, .9)}
TINT = {"hickory": (0.95, 0.72, 0.48)}


def _material_for(S, name, dresser):
    if name in S.mats:
        return S.mats[name]
    tex = TEXTURED
    if name in tex and dresser is not None:
        t, tile, rough = tex[name]
        tint = TINT.get(name)
        return dresser.pbr(name, t, tile, rough, tint=tint)
    if name in EMISSIVE:
        rgb, em, inten = EMISSIVE[name]
        return S.color(name, rgb, .35, 0., emissive=em, emissive_intensity=inten)
    if name in NORMAL_MAPPED and dresser is not None:
        from . import visual_assets as va
        rgb, rough, metal, tex, tile, bump = NORMAL_MAPPED[name]
        T, Sdf, Gf = S.Sdf.ValueTypeNames, S.Sdf, S.Gf
        maps = va.texture(tex)
        dresser.credits.add(tex)
        return S.mdl(name, "OmniPBR", {"diffuse_color_constant": (T.Color3f, Gf.Vec3f(*map(float, rgb))),
                                       "reflection_roughness_constant": (T.Float, float(rough)),
                                       "metallic_constant": (T.Float, float(metal)),
                                       "normalmap_texture": (T.Asset, Sdf.AssetPath(str(Path(maps["nor_gl"]).resolve()))),
                                       "bump_factor": (T.Float, float(bump)),
                                       "texture_scale": (T.Float2, Gf.Vec2f(1/tile, 1/tile))})
    rgb, rough, metal = COLORS.get(name, COLORS["robot_dark"])
    return S.color(name, rgb, rough, metal)


def crease_normals(v, f, angle_deg=35.):
    """Per-corner normals (face-varying, in f.ravel() order): the area-weighted mean of the normals of the faces around
    each vertex that lie within `angle_deg` of the corner's own face (flat faces stay flat, curved ones smooth)."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64).reshape(-1, 3)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    fu = fn/np.maximum(ln, 1e-30)[:, None]
    cv = f.ravel()
    cf = np.repeat(np.arange(len(f)), 3)
    order = np.argsort(cv, kind="stable")
    cvs, cfs = cv[order], cf[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cvs)) + 1]
    counts = np.diff(np.r_[starts, len(cvs)])
    grp = np.repeat(np.arange(len(starts)), counts)
    gstart, gcount = starts[grp], counts[grp]
    rep_i = np.repeat(np.arange(len(cvs)), gcount)
    offs = np.arange(rep_i.size) - np.repeat(np.cumsum(gcount) - gcount, gcount)
    rep_j = gstart[rep_i] + offs
    fi, fj = cfs[rep_i], cfs[rep_j]
    keep = np.einsum("ij,ij->i", fu[fi], fu[fj]) >= math.cos(math.radians(angle_deg))
    acc = np.zeros((len(cvs), 3))
    np.add.at(acc, rep_i[keep], fn[fj[keep]])
    nl = np.linalg.norm(acc, axis=1)
    bad = nl < 1e-30
    acc[bad] = fu[cfs[bad]]
    acc /= np.maximum(np.linalg.norm(acc, axis=1), 1e-30)[:, None]
    out = np.empty_like(acc)
    out[order] = acc
    return out.astype(np.float32)


def draw_parts(S, path, parts, dresser, p=None, q=None):
    """Author procedural parts [(name, v, f, material[, opts])] as meshes under an Xform at `path` (placed at (p, q)
    when given): crease-split face-varying normals (or smooth vertex normals), face-varying planar UVs for textured
    materials, double-sided."""
    x = S.UsdGeom.Xform.Define(S.stage, path)
    if p is not None:
        S.place(x, p, q if q is not None else (0., 0., 0., 1.))
    used = {}
    for part in parts:
        name, v, f, mat = part[:4]
        opts = part[4] if len(part) > 4 else {}
        if v is None or not len(f):
            continue
        k = used.get(name, 0)
        used[name] = k + 1
        sub = f"{path}/{name}" + (f"_{k}" if k else "")
        v = np.asarray(v, np.float32)
        f = np.asarray(f, np.int32).reshape(-1, 3)
        textured = mat in TEXTURED or mat in NORMAL_MAPPED or opts.get("uv")
        uv = _planar_uv(v, f, float(opts.get("uv", 1.))) if textured else None
        g = S.mesh(sub, v, f, uv=uv, mat=_material_for(S, mat, dresser), uv_interp="faceVarying")
        if opts.get("smooth"):
            import trimesh
            g.CreateNormalsAttr(S.Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(v, f, process=False).vertex_normals
                                                          .astype(np.float32)))
            g.SetNormalsInterpolation("vertex")
        else:
            g.CreateNormalsAttr(S.Vt.Vec3fArray.FromNumpy(crease_normals(v, f, float(opts.get("crease", 35.)))))
            g.SetNormalsInterpolation("faceVarying")
        g.CreateDoubleSidedAttr(True)
    return x


def export(world, folder, seed=0):
    """Export the world's static appearance + posable bodies to folder/scene.usdc; returns (path, meta)."""
    import newton
    from . import dressing
    sc = world.scene
    b = sc.b
    folder = Path(folder)
    S = Stage(folder)
    dresser = dressing.Dresser(S)
    for name, (rgb, rough, metal) in COLORS.items():
        if name not in TEXTURED:
            S.color(name, rgb, rough, metal)
    q = world.bq
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    hidden_bodies = set()
    for t in sc.tools.values():
        for v in getattr(t, "visual", []):
            hidden_bodies.update(v.get("hide_shapes_of", []))
    drillable_bodies = {d.body for d in sc.drillables.values()}
    labels = list(b.body_label)
    hidden_bodies.update(k for k, lab in enumerate(labels) if lab in ("vacuum_unit", "vacuum_switch"))   # drawn as models
    V = int(newton.ShapeFlags.VISIBLE)
    for i in range(b.shape_count):
        body, lab = b.shape_body[i], b.shape_label[i]
        mname = sc.materials.get(i, "robot")
        if mname in HIDDEN or lab == "floor" or body in hidden_bodies:
            continue
        if body in drillable_bodies and lab.endswith("_solid"):
            continue                                      # drawn from its hole records (workpiece mesh)
        flags = int(b.shape_flags[i])
        if mname == "robot" and not flags & V:
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        tf = np.asarray(b.shape_transform[i], float)
        st = b.shape_type[i]
        if mname != "robot":
            from . import visual_models
            rep = visual_models.shape_visual(sc, i, lab, mname, np.asarray(b.shape_scale[i], float),
                                             np.asarray(b.shape_transform[i], float))
            if rep is not None:
                if rep:
                    tf_ = np.asarray(b.shape_transform[i], float)
                    draw_parts(S, path, rep, dresser, tf_[:3], tf_[3:])
                continue
        if mname == "robot":
            col = np.clip(np.asarray(b.shape_color[i], float), 0., 1.)**2.2
            key = "robot_%02x%02x%02x" % tuple(int(round(c*255)) for c in np.asarray(b.shape_color[i], float).clip(0, 1))
            mat = S.color(key, col, .28 if float(col.mean()) > 0.5 else .38, 0.)
        else:
            mat = _material_for(S, mname, dresser)
        if src is not None and hasattr(src, "vertices"):
            g = S.mesh(path, np.asarray(src.vertices)*scale, src.indices, smooth=True, mat=mat)
            S.place(g, tf[:3], tf[3:])
        elif st == newton.GeoType.BOX:
            g = S.box(path, 2*scale, mat, p=tf[:3], q=tf[3:], uv_per_m=1. if mname in ("concrete_wall", "block_paint",
                                                                                          "floor_concrete", "ceiling") else None)
        elif st in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            g = S.cylinder(path, scale[0], 2*scale[1], mat, p=tf[:3], q=tf[3:], capsule=st == newton.GeoType.CAPSULE)
        elif st == newton.GeoType.SPHERE:
            g = S.sphere(path, scale[0], mat, p=tf[:3])
        else:
            continue
    # workpieces: placeholder meshes, replaced per request from the hole records
    meta = dict(workpieces={}, assets=[])
    for name, d in sc.drillables.items():
        mat = _material_for(S, d.render_material, dresser)
        v, f, _ = workpiece_mesh(d)
        S.mesh(f"/Lab/body_{d.body}/workpiece", v, f, uv=workpiece_uv(d, v, f), mat=mat, uv_interp="faceVarying")
        for kind in ("burr", "dust", "scorch"):
            g = S.mesh(f"/Lab/body_{d.body}/{kind}", [(0., 0., -50.), (.001, 0., -50.), (0., .001, -50.)], [0, 1, 2],
                       mat=S.mats[kind])
            S.UsdGeom.Imageable(g).MakeInvisible()
        meta["workpieces"][name] = d.body
    # layout marks: carpenter's crosses on concrete and timber (2 mm x 40 mm marker strokes); on metal parts a patch of
    # blue layout dye with scribed crosses (0.4 mm x 14 mm), as machinists lay out
    pencil = S.mats["pencil"]
    dye = S.color("layout_dye", (.03, .10, .45), .35, .2)
    scribe = S.color("scribe", (.80, .82, .85), .3, .8)
    sc.mark_prims = {}
    for k, mk in enumerate(sc.layout.get("marks", [])):
        if "part" in mk:
            body = sc.parts[mk["part"]]["body"]
            p = np.asarray(mk["local"], float)
            n_ax = 2
            metal = sc.parts[mk["part"]]["material"] in ("steel", "aluminium")
        else:
            # a mark over the ledger is drawn on the ledger's face, where a carpenter marks it (drawn on the wall behind
            # it, the ledger task's four marks were hidden from every camera)
            on = sc.drillables["ledger"] if "ledger" in sc.layout and "ledger" in sc.drillables else sc.drillables["wall"]
            body = on.body
            qw = world.bq[body]
            p = Rotation.from_quat(qw[3:]).inv().apply(np.array([mk["x"], P.WALL_FACE_Y, mk["z"]]) - qw[:3])
            if on is not sc.drillables["wall"]:
                p[1] = -float(on.half[1])         # the ledger's own front face (it stands 3.5 mm off the wall on its brackets)
            n_ax = 1
            metal = False
        sgn = 1. if n_ax == 2 else -1.
        lift = np.zeros(3)
        dn = mk["part"] if "part" in mk else on.name
        specs = sc.mark_prims.setdefault(dn, [])
        if metal:
            size = [0.024, 0.024, 0.024]
            size[n_ax] = 0.00005
            lift[n_ax] = sgn*0.00003
            specs.append((f"/Lab/body_{body}/dye_{k}", p + lift, np.asarray(size), dye))
        for j in range(2):
            size = [0.040, 0.040, 0.040] if not metal else [0.014, 0.014, 0.014]
            size[n_ax] = 0.00005
            size[(n_ax + 1 + j) % 3] = 0.002 if not metal else 0.0004
            lift[n_ax] = sgn*(0.00009 if metal else 0.00003)
            specs.append((f"/Lab/body_{body}/mark_{k}_{j}", p + lift, np.asarray(size), scribe if metal else pencil))
    # marks are meshes with every hole and dimple at the mark cut out (drawn as flat plates, they hid the 9 mm hole drilled
    # through a dyed mark and every punched dimple); rebuilt with the workpiece meshes
    for dn, specs in getattr(sc, "mark_prims", {}).items():
        meshes = mark_meshes(sc, sc.drillables[dn])
        for path, centre, size, mat in specs:
            m = meshes[path]
            g = S.mesh(path, *(m if m is not None else ([(0., 0., -50.), (.001, 0., -50.), (0., .001, -50.)], [0, 1, 2])),
                       mat=mat)
            if m is None:
                S.UsdGeom.Imageable(g).MakeInvisible()
    # tool appearance models (fitted, posed with their bodies)
    for t in sc.tools.values():
        for vi, v in enumerate(getattr(t, "visual", [])):
            parent = f"/Lab/body_{v['body']}" if v["body"] >= 0 else "/Lab"
            path = f"{parent}/{t.name}_model{vi}"
            if "procedural" in v:
                from . import visual_models
                draw_parts(S, path, getattr(visual_models, v["procedural"])(), dresser)
                continue
            dresser.ref_parts(path, v["asset"], v["pos"], v["quat"], v.get("scale", 1.), v.get("prims"))
            meta["assets"].append(v["asset"])
    _draw_vacuum(S, world, dresser)
    # a broken bit hides its point per request (broken_bit_updates): author the attribute the worker sets
    for shank, tf, h, a, point, cone in broken_bit_shapes(world).values():
        prim = S.stage.GetPrimAtPath(point) if point else None
        if prim:
            S.UsdGeom.Imageable(prim).CreateVisibilityAttr(S.UsdGeom.Tokens.inherited)
    # the impact wrench's status LEDs start dark and are shown by the tool's light (wrench_light_updates)
    for leds in wrench_light_prims(world).values():
        for path in leds["green"] + leds["red"]:
            prim = S.stage.GetPrimAtPath(path)
            if prim:
                S.UsdGeom.Imageable(prim).CreateVisibilityAttr(S.UsdGeom.Tokens.invisible)
    credits = dresser.dress(world)
    meta["cameras"] = _cameras(S, sc)
    meta["visual_assets"] = sorted(set(credits) | set(meta["assets"]))
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _cameras(S, sc):
    out = {}
    for name, m in sc.cameras.items():
        aspect = m.get("width", 960)/m.get("height", 540)
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
        out[name] = dict(role=m["role"], width=m.get("width", 960), height=m.get("height", 540))
    return out


# ------------------------------------------------------------------------------ the vacuum unit and its hose
def _body_aabb(sc, body):
    """(lo, hi) of every collider on `body`, in its frame (None if it has none)."""
    b = sc.b
    import newton
    G = newton.GeoType
    pts = []
    for i in range(b.shape_count):
        if int(b.shape_body[i]) != body:
            continue
        s = np.asarray(b.shape_scale[i], float)
        R, t = _q7(b.shape_transform[i])
        st = int(b.shape_type[i])
        if st == int(G.BOX):
            h = s[:3]
        elif st in (int(G.CYLINDER), int(G.CAPSULE)):
            h = np.array([s[0], s[0], s[1] + (s[0] if st == int(G.CAPSULE) else 0.)])
        elif st == int(G.SPHERE):
            h = np.full(3, s[0])
        else:
            src = b.shape_source[i]
            if src is None or not hasattr(src, "vertices"):
                continue
            v = np.asarray(src.vertices, float)*s
            pts.append(v @ R.T + t)
            continue
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])*h
        pts.append(c @ R.T + t)
    if not pts:
        return None
    p = np.concatenate(pts)
    return p.min(0), p.max(0)


def vacuum_setup(world):
    """The layout's vacuum (layout["vacuum"] = {xyz, yaw, port_local, switch_local}; scene bodies "vacuum_unit" and
    "vacuum_switch" when the task builds them) resolved for drawing: unit and switch body indices (-1: none), the
    unit's envelope (lo, hi) in its frame (its colliders, else a DWV012-sized box standing on the floor), the layout
    pose of the unit frame, and the wand's rear end on the nozzle part. None when the layout has no vacuum."""
    from . import visual_models as VM
    sc = world.scene
    spec = sc.layout.get("vacuum")
    if not spec:
        return None
    labels = list(sc.b.body_label)
    unit = labels.index("vacuum_unit") if "vacuum_unit" in labels else -1
    switch = labels.index("vacuum_switch") if "vacuum_switch" in labels else -1
    xyz = np.asarray(spec["xyz"], float)
    R0 = Rotation.from_euler("z", float(spec.get("yaw", 0.))).as_matrix()
    port = np.asarray(spec.get("port_local", (0.30, 0., 0.40)), float)
    box = _body_aabb(sc, unit) if unit >= 0 else None
    if box is None:
        L, W, H = VM.VACUUM_SIZE_M
        if abs(port[1]) > abs(port[0]):
            L, W = W, L
        lo, hi = np.array([-0.5*L, -0.5*W, -xyz[2]]), np.array([0.5*L, 0.5*W, H - xyz[2]])
    else:
        lo, hi = box
    noz = next((n for n, p in sc.parts.items() if p.get("kind") == "nozzle"), None)
    rear = None
    if noz is not None:
        b = sc.b
        hb = sc.parts[noz]["body"]
        rear = np.array([-0.18, 0., 0.])
        for i in range(b.shape_count):
            if b.shape_label[i] == f"{noz}_hose_stub" and int(b.shape_body[i]) == hb:
                R, t = _q7(b.shape_transform[i])
                h = float(np.asarray(b.shape_scale[i], float)[1])
                ends = [t + R[:, 2]*h, t - R[:, 2]*h]
                rear = min(ends, key=lambda e: float(e[0]))          # the end away from the tip (+x)
        rear = (hb, rear)
    # the unit's named colliders (vacuum.py): drum tank, power head, hose inlet (unit frame)
    named = {}
    if unit >= 0:
        b = sc.b
        for i in range(b.shape_count):
            if int(b.shape_body[i]) == unit:
                Rs, ts = _q7(b.shape_transform[i])
                named[str(b.shape_label[i])] = (np.asarray(b.shape_scale[i], float), Rs, ts)
    drum = None
    if "vacuum_tank" in named and "vacuum_head" in named:
        (st, _, tt), (sh, _, th) = named["vacuum_tank"], named["vacuum_head"]
        drum = dict(tank=(float(st[0]), float(tt[2] - st[1]), float(tt[2] + st[1])),
                    head=(float(sh[0]), float(th[2] - sh[1]), float(th[2] + sh[1])))
    n_p, _ = VM._face_axis(lo, hi, port)
    port_end, port_dir = port + n_p*0.035, n_p
    if "vacuum_inlet" in named:
        si, Ri, ti = named["vacuum_inlet"]
        ax = Ri[:, 2]
        ends = [ti + ax*si[1], ti - ax*si[1]]
        e = max(ends, key=lambda q: float(np.linalg.norm(q[:2])))       # the end away from the tank axis
        port_dir = (e - ti)/np.linalg.norm(e - ti)
        port_end = e
        if drum is not None:
            drum["inlet"] = (e, port_dir, float(si[0]))
    return dict(spec=spec, unit=unit, switch=switch, lo=lo, hi=hi, R0=R0, t0=xyz, port=port, drum=drum,
                port_end=port_end, port_dir=port_dir,
                switch_local=np.asarray(spec["switch_local"], float) if spec.get("switch_local") is not None else None,
                switch_normal=np.asarray(spec.get("switch_normal_local") or (0., 0., 1.), float),
                nozzle=rear, hose=dict(VM.VACUUM_HOSE, **{k: spec[k] for k in ("hose_length_m",) if k in spec}))


def _draw_vacuum(S, world, dresser):
    """The vacuum unit (and its switch) as procedural models on their bodies, or at the layout pose when the scene
    has no such body; the hose prim is sent with every render request (Hose)."""
    from . import visual_models as VM
    v = vacuum_setup(world)
    if v is None:
        return
    if v["drum"] is not None:
        pivot = size = None
        if v["switch"] >= 0:
            box = _body_aabb(world.scene, v["switch"])
            Ru, tu = _body_tf(world.bq, v["unit"])
            Rs, ts = _body_tf(world.bq, v["switch"])
            pivot = Ru.T @ (ts - tu)
            size = tuple(float(x) for x in (box[1] - box[0])) if box is not None else None
        parts = VM.drum_vacuum(v["drum"]["tank"], v["drum"]["head"], v["drum"].get("inlet"), pivot, size)
    else:
        parts = VM.shop_vacuum(v["lo"], v["hi"], v["port"], v["switch_local"], rocker=v["switch"] < 0)
    if v["unit"] >= 0:
        draw_parts(S, f"/Lab/body_{v['unit']}/vacuum_model", parts, dresser)
    else:
        draw_parts(S, "/Lab/vacuum_unit_model", parts, dresser, v["t0"], Rotation.from_matrix(v["R0"]).as_quat())
    if v["switch"] >= 0:
        box = _body_aabb(world.scene, v["switch"])
        if box is not None:
            # the rocker's outward normal (layout switch_normal_local), carried into the switch body's frame
            n_u = v["switch_normal"]
            Ru = _body_tf(world.bq, v["unit"])[0] if v["unit"] >= 0 else v["R0"]
            Rs = _body_tf(world.bq, v["switch"])[0]
            draw_parts(S, f"/Lab/body_{v['switch']}/vacuum_switch_model", VM.rocker_switch(box[0], box[1],
                                                                                             Rs.T @ (Ru @ n_u)), dresser)


class Hose:
    """The vacuum hose, drawn only (no collider, no mass): a corrugated tube of the hose's real length (15 ft of 1-1/4
    in hose, VACUUM_HOSE) from the unit's inlet cuff to the rear end of the nozzle wand, recomputed for every pose.
    It leaves the cuff and the wand along their axes and hangs from each down to the floor (draping over a collider
    on the way, e.g. the trolley top); the length left over lies on the floor between the two touch-downs in a loose
    meander, on the side with fewer obstacles. Pulled further than its length it is drawn as a taut curve between the
    ends. The robot does not feel it."""

    def __init__(self, world):
        self.world = world
        self.v = vacuum_setup(world)
        self.key = None

    def ends(self, bq):
        from . import visual_models as VM
        v = self.v
        if v is None or v["nozzle"] is None:
            return None
        Ru, tu = _body_tf(bq, v["unit"]) if v["unit"] >= 0 else (v["R0"], v["t0"])
        d0 = Ru @ v["port_dir"]
        p0 = Ru @ v["port_end"] + tu
        hb, rear = v["nozzle"]
        Rh, th = _body_tf(bq, hb)
        d1 = Rh @ np.array([-1., 0., 0.])
        p1 = Rh @ rear + th
        return p0, d0, p1, d1

    @staticmethod
    def _len(P_):
        return float(np.sum(np.linalg.norm(np.diff(P_, axis=0), axis=1)))

    def _drape(self, P_, bq, r, top):
        z, _ = _supports(self.world).down(np.column_stack([P_[:, :2], np.full(len(P_), top)]), bq)
        P_[:, 2] = np.maximum(P_[:, 2], z + r)
        return P_

    def _grid(self, bq, cell=0.04):
        """Collider heightfield of the floor area (top z of the highest collider under each cell), cached per world:
        (xs, ys, occupied = top above 6 cm, top z, cell). Cells outside the room or under tall furniture (> 1.2 m)
        are impassable for the hose's floor run."""
        g = getattr(self, "_occ", None)
        if g is not None:
            return g
        R = P.ROOM
        xs = np.arange(R["x0"] + cell/2, R["x1"], cell)
        ys = np.arange(R["y0"] + cell/2, P.WALL_FACE_Y, cell)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        z, _ = _supports(self.world).down(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 2.9)]), bq)
        top = np.maximum(z.reshape(X.shape), 0.)
        top[[0, -1], :] = 9.
        top[:, [0, -1]] = 9.
        occ = top > 0.06
        grown = occ.copy()                                   # keep a hose radius off obstacles on the floor
        grown[1:, :] |= occ[:-1, :]
        grown[:-1, :] |= occ[1:, :]
        grown[:, 1:] |= occ[:, :-1]
        grown[:, :-1] |= occ[:, 1:]
        self._occ = (xs, ys, grown, top, cell)
        return self._occ

    def _cell(self, p, bq):
        xs, ys, occ, top, cell = self._grid(bq)
        i = int(np.clip(round((p[0] - xs[0])/cell), 0, len(xs) - 1))
        j = int(np.clip(round((p[1] - ys[0])/cell), 0, len(ys) - 1))
        return i, j

    def _astar(self, a, b, bq):
        """Shortest path between points a and b over the collider heightfield (8-connected; a step costs its 3D length,
        so a route over a low trolley top competes with one around it; tall furniture and walls are impassable)."""
        import heapq
        xs, ys, occ, top, cell = self._grid(bq)
        s0, s1 = self._cell(a, bq), self._cell(b, bq)
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        hfun = lambda c: cell*math.hypot(c[0] - s1[0], c[1] - s1[1])
        pad = int(1.2/cell)                                  # search a window around the two ends
        lo_i, hi_i = max(0, min(s0[0], s1[0]) - pad), min(top.shape[0] - 1, max(s0[0], s1[0]) + pad)
        lo_j, hi_j = max(0, min(s0[1], s1[1]) - pad), min(top.shape[1] - 1, max(s0[1], s1[1]) + pad)
        openq = [(hfun(s0), 0., s0)]
        came, cost = {s0: None}, {s0: 0.}
        while openq:
            _, g, c = heapq.heappop(openq)
            if c == s1:
                break
            if g > cost[c] + 1e-9:
                continue
            for di, dj in nb:
                q = (c[0] + di, c[1] + dj)
                if not (lo_i <= q[0] <= hi_i and lo_j <= q[1] <= hi_j) or top[q] > 1.2:
                    continue
                # furniture tops cost extra (a hose lies on the floor unless there is no way round)
                ng = g + math.sqrt((cell*cell)*(di*di + dj*dj) + (top[q] - top[c])**2) + (0.15 if occ[q] else 0.)
                if ng < cost.get(q, 1e18):
                    cost[q], came[q] = ng, c
                    heapq.heappush(openq, (ng + hfun(q), ng, q))
        if s1 not in came:
            return None
        path, c = [], s1
        while c is not None:
            path.append((xs[c[0]], ys[c[1]]))
            c = came[c]
        return np.asarray(path[::-1])

    def _free_floor(self, p, bq, clear):
        """Centre of the nearest free floor cell to p with `clear` metres of free floor around it."""
        xs, ys, occ, top, cell = self._grid(bq)
        i0, j0 = self._cell(p, bq)
        k = int(math.ceil(clear/cell))
        best = None
        for rad in range(1, int(2.0/cell)):
            for i in range(i0 - rad, i0 + rad + 1):
                for j in (j0 - rad, j0 + rad) if abs(i - i0) < rad else range(j0 - rad, j0 + rad + 1):
                    if k <= i < occ.shape[0] - k and k <= j < occ.shape[1] - k and not occ[i - k:i + k + 1, j - k:j + k + 1].any():
                        d = math.hypot(i - i0, j - j0)
                        if best is None or d < best[0]:
                            best = (d, np.array([xs[i], ys[j], 0.]))
            if best is not None:
                return best[1]
        return None

    def _height(self, pts, bq, r, exact=True):
        """(hose centre height on the colliders under each point, occupied flag); exact: a fresh ray cast, else the
        cached heightfield (used inside the length search)."""
        xs, ys, occ, top, cell = self._grid(bq)
        ii = np.clip(np.round((pts[:, 0] - xs[0])/cell).astype(int), 0, len(xs) - 1)
        jj = np.clip(np.round((pts[:, 1] - ys[0])/cell).astype(int), 0, len(ys) - 1)
        if not exact:
            return np.minimum(top[ii, jj], 1.2) + r, occ[ii, jj]
        z, _ = _supports(self.world).down(np.column_stack([pts[:, :2], np.full(len(pts), 2.9)]), bq)
        return np.maximum(z, 0.) + r, occ[ii, jj]

    @staticmethod
    def _coil(F, extra, r, occ, xs, ys, cell):
        """Lay `extra` metres more of hose as a loose flat coil on free floor beside the floor run F (a hose's slack
        lies in loops); the coil leaves and rejoins the run at one point."""
        def free(pts):
            ii = np.round((pts[:, 0] - xs[0])/cell).astype(int)
            jj = np.round((pts[:, 1] - ys[0])/cell).astype(int)
            ok = (ii >= 0) & (ii < len(xs)) & (jj >= 0) & (jj < len(ys))
            return ok.all() and not occ[ii[ok], jj[ok]].any()
        tang = np.gradient(F[:, :2], axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        def loop_pts(c, a0, sgn, rho, n, start):
            th = np.linspace(0., 2*math.pi*n, max(16, int(n*48)))
            rad = rho + 2.3*r*th/(2*math.pi)
            pts = c[None, :] + rad[:, None]*np.column_stack([np.cos(a0 + sgn*th), np.sin(a0 + sgn*th)])
            return np.vstack([pts, np.linspace(pts[-1], start, 10)[1:]])
        for rho in sorted({float(np.clip(extra/(2*math.pi), 0.05, 0.24)), 0.12, 0.08, 0.05}, reverse=True):
            order = [i for i in sorted(range(len(F)), key=lambda i: abs(i - len(F)//2)) if F[i, 2] < 0.05]
            for i in order[::3]:
                for sgn in (1., -1.):
                    nv = sgn*np.array([-tang[i, 1], tang[i, 0]])
                    c = F[i, :2] + nv*rho
                    a0 = math.atan2(-nv[1], -nv[0])
                    n = 0.3
                    while n < 8:
                        lp = loop_pts(c, a0, sgn, rho, n, F[i, :2])
                        if float(np.sum(np.linalg.norm(np.diff(np.vstack([F[i, :2], lp]), axis=0), axis=1))) >= extra:
                            break
                        n += 0.02
                    if free(lp):
                        loop = np.column_stack([lp, np.full(len(lp), r)])
                        return np.vstack([F[:i + 1], loop, F[i + 1:]])
        return F

    def _hang(self, P_end, d, r, bq, n=40):
        """From end P_end (exit direction d) down onto whatever is below a short way out: (points, touch-down)."""
        h = max(float(P_end[2]) - r, 0.)
        dh = np.array([d[0], d[1], 0.])
        dh = dh/np.linalg.norm(dh) if np.linalg.norm(dh) > 0.2 else np.array([0., -1., 0.])
        T = P_end + dh*(0.08 + 0.12*h)
        T[2] = float(self._height(T[None, :], bq, r)[0][0])
        if T[2] > P_end[2] - 0.02:
            # the end points at something taller than itself (an inlet facing furniture): the hose bends down to the
            # nearest free floor beside it instead of climbing onto it
            xs, ys, occ, top, cell = self._grid(bq)
            i0, j0 = self._cell(P_end, bq)
            best = None
            for di in range(-15, 16):
                for dj in range(-15, 16):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and not occ[i, j]:
                        dd = math.hypot(di, dj)
                        if best is None or dd < best[0]:
                            best = (dd, i, j)
            if best is not None:
                T = np.array([xs[best[1]], ys[best[2]], r])
        a = 0.06 + 0.2*max(float(P_end[2] - T[2]), 0.)
        s = np.linspace(0., 1., n)[:, None]
        B = (1 - s)**3*P_end + 3*(1 - s)**2*s*(P_end + a*d) + 3*(1 - s)*s**2*(T + np.array([0., 0., 0.3*a])) + s**3*T
        B = self._drape(B, bq, r, max(float(P_end[2]), 0.) + 0.05)
        B[0] = P_end
        return B, T

    def path(self, bq):
        p0, d0, p1, d1 = self.ends(bq)
        L, r = float(self.v["hose"]["length_m"]), float(self.v["hose"]["r_m"])
        h0, T0 = self._hang(p0, d0, r, bq)
        h1, T1 = self._hang(p1, d1, r, bq)
        Lf = L - self._len(h0) - self._len(h1)
        xs, ys, occ, top, cell = self._grid(bq)
        route = self._astar(T0, T1, bq)
        if route is not None:
            fl = ~self._height(np.column_stack([route, np.zeros(len(route))]), bq, r)[1]
            if fl.sum() < 12:
                # the run between the touch-downs lies on furniture: lead it over the nearest free floor beside the
                # inlet, where the slack can lie
                Fc = self._free_floor(T0, bq, 0.25)
                if Fc is not None:
                    ra, rb = self._astar(T0, Fc, bq), self._astar(Fc, T1, bq)
                    if ra is not None and rb is not None:
                        route = np.vstack([ra, rb[1:]])
        if route is not None:
            R2 = np.vstack([T0[:2], route[1:-1], T1[:2]]) if len(route) > 2 else np.vstack([T0[:2], T1[:2]])
            for _ in range(6):
                if len(R2) > 2:
                    R2[1:-1] = 0.25*R2[:-2] + 0.5*R2[1:-1] + 0.25*R2[2:]
            seg = np.linalg.norm(np.diff(R2, axis=0), axis=1)
            sc = np.concatenate([[0.], np.cumsum(seg)])
            tq = np.linspace(0., sc[-1], 200)
            base = np.column_stack([np.interp(tq, sc, R2[:, 0]), np.interp(tq, sc, R2[:, 1])])
            route3 = np.column_stack([base, self._height(base, bq, r)[0]])
        if route is None or self._len(route3) > Lf:
            # not enough hose to lie along the floor between the touch-downs: a taut curve between the ends
            Dd = float(np.linalg.norm(p1 - p0))
            a = float(np.clip(0.3*Dd, 0.1, 0.6))
            s = np.linspace(0., 1., 140)[:, None]
            B = (1 - s)**3*p0 + 3*(1 - s)**2*s*(p0 + a*d0) + 3*(1 - s)*s**2*(p1 + a*d1) + s**3*p1
            B = self._drape(B, bq, r, max(float(B[:, 2].max()), 0.) + 0.05)
            B[0], B[-1] = p0, p1
            return B, self._len(B)
        tang = np.gradient(base, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
        u = tq/max(sc[-1], 1e-9)
        on_floor = ~self._height(base, bq, r)[1]

        def floor(A, k, sign):
            F = base + (sign*A*np.sin(math.pi*k*u)*on_floor)[:, None]*nrm   # meander only where it lies on the floor
            z, bad = self._height(F, bq, r, exact=False)
            F[bad & on_floor] = base[bad & on_floor]                           # never into an obstacle
            z, _ = self._height(F, bq, r, exact=False)
            return np.column_stack([F, z]), int((bad & on_floor).sum())
        target, best_P = Lf, None
        for attempt in range(6):                       # smoothing the joins shortens the hose: aim longer, re-measure
            P_ = self._lay(floor, target, base, r, occ, xs, ys, cell, h0, h1, p0, p1)
            P_[1:-1, 2] = np.maximum(P_[1:-1, 2], self._height(P_[1:-1], bq, r)[0])     # smoothing never sinks it
            miss = L - self._len(P_)
            if best_P is None or abs(miss) < abs(L - self._len(best_P)):
                best_P = P_
            if abs(miss) < 0.02:
                break
            target += 0.7*miss
        return best_P, self._len(best_P)

    def _lay(self, floor, Lf, base, r, occ, xs, ys, cell, h0, h1, p0, p1):
        best = None
        for k in (1, 2, 3, 4, 5, 6):
            for sign in (1., -1.):
                lo_, hi_ = 0., 3.0
                if self._len(floor(hi_, k, sign)[0]) < Lf:
                    continue
                for _ in range(30):
                    mid = 0.5*(lo_ + hi_)
                    if self._len(floor(mid, k, sign)[0]) < Lf:
                        lo_ = mid
                    else:
                        hi_ = mid
                F, nbad = floor(0.5*(lo_ + hi_), k, sign)
                score = (nbad, 0.5*(lo_ + hi_))
                if best is None or score < best[0]:
                    best = (score, F)
        F = best[1] if best is not None else floor(0., 1, 1.)[0]
        short = Lf - self._len(F)
        if short > 0.05:
            F = self._coil(F, short, r, occ, xs, ys, cell)
        P_ = np.concatenate([h0, F[1:-1], h1[::-1]], 0)
        for _ in range(3):
            P_[1:-1] = 0.25*P_[:-2] + 0.5*P_[1:-1] + 0.25*P_[2:]
            P_[:, 2] = np.maximum(P_[:, 2], r)
        P_[0], P_[-1] = p0, p1
        return P_

    def meshes(self, bq):
        if self.v is None or self.v["nozzle"] is None:
            return {}
        ends = self.ends(bq)
        key = tuple(np.round(np.concatenate(ends), 3))
        if key == self.key:
            return {}
        self.key = key
        from . import visual_models as VM
        import trimesh
        P_, _ = self.path(bq)
        r = float(self.v["hose"]["r_m"])
        v, f = VM.sweep(P_, r, bend=0., n=12, corrugate=(0.0011, 0.0075), cap=True, step=0.0019)
        nrm = trimesh.Trimesh(v, f, process=False).vertex_normals.astype(np.float32)
        return {"/Lab/vacuum_hose": dict(v=v, t=f, n=nrm, mat="hose_grey")}


# ------------------------------------------------------------------------------ broken bits
def broken_bit_shapes(world):
    """{bit item: (shank prim, its pose (p, q) and half length in the body frame, point prim, cone length)} for every
    bit drawn as a shank + point (render.export paths)."""
    sc = world.scene
    b = sc.b
    out = {}
    labels = list(b.shape_label)
    for name, it in getattr(sc, "items", {}).items():
        if it.kind != "bit":
            continue
        ib = next((i for i, l in enumerate(labels) if l == f"{name}_body" and int(b.shape_body[i]) == it.body), None)
        ip = next((i for i, l in enumerate(labels) if l == f"{name}_point" and int(b.shape_body[i]) == it.body), None)
        if ib is None:
            continue
        tf = np.asarray(b.shape_transform[ib], float)
        h = float(np.asarray(b.shape_scale[ib], float)[1])
        a = np.asarray(it.axis, float)/np.linalg.norm(it.axis)
        cone = float(np.linalg.norm(np.asarray(it.tip, float) - (tf[:3] + a*h)))
        out[name] = (f"/Lab/body_{it.body}/shape_{ib}", tf, h, a,
                     f"/Lab/body_{it.body}/shape_{ip}" if ip is not None else None, cone)
    return out


def broken_bit_updates(bits, shortened, sent):
    """Visibility and xform updates that draw a broken bit as the physics keeps it (world.on_bit_broken): its point
    hidden and its shank cut back at the tip end by (piece left in the hole - point length), the drawn shank scaled
    along its axis. `sent` remembers what the worker already has."""
    vis, xf = {}, {}
    for name, (shank, tf, h, a, point, cone) in bits.items():
        s = float(shortened.get(name) or 0.)
        if sent.get(name, 0.) == s:
            continue
        sent[name] = s
        cut = min(max(0., s - cone), 2*h - 1e-3) if s > 0 else 0.
        R = Rotation.from_quat(tf[3:]).as_matrix()
        p = tf[:3] - a*0.5*cut
        xf[shank] = dict(t=[float(x) for x in p], s=[1., 1., float((2*h - cut)/(2*h))])
        if point is not None:
            vis[point] = s <= 0.
    return vis, xf


# ------------------------------------------------------------------------------ impact wrench status LEDs
def wrench_light_prims(world):
    """{impact wrench: {"green": [prims], "red": [prims]}}: the LEDs on both sides of its Adaptive Torque module
    (visual_models.impact_wrench, drawn under the wrench's head collider, render.export paths)."""
    sc = world.scene
    labels = list(sc.b.shape_label)
    out = {}
    for name, t in sc.tools.items():
        if type(t).__name__ != "ImpactWrench":
            continue
        i = next((i for i, l in enumerate(labels) if l == f"{name}_head"), None)
        if i is None:
            continue
        base = f"/Lab/body_{int(sc.b.shape_body[i])}/shape_{i}"
        out[name] = {c: [f"{base}/at_led_{c}"] for c in ("green", "red")}       # one mesh per colour, both sides
    return out


def wrench_light_updates(leds, lights, sent):
    """Visibility updates that light a wrench's LEDs as its simulated state says (tools.ImpactWrench s_light: off,
    green after the module's automatic stop, red after a timeout); `sent` remembers what the worker already has."""
    vis = {}
    for name, paths in leds.items():
        light = (lights or {}).get(name, "off")
        if sent.get(name) == light:
            continue
        sent[name] = light
        for c in ("green", "red"):
            for path in paths[c]:
                vis[path] = light == c
    return vis


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, world, folder, scene, *, width=None, height=None, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        self.cache = WorkpieceCache(world)
        self.hose = Hose(world)
        self.bits = broken_bit_shapes(world)
        self._short_sent = {}
        self.leds = wrench_light_prims(world)
        self._light_sent = {}
        self._first = True
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, DRL_RENDER_PORT=str(listener.getsockname()[1]), DRL_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "drilling.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width or 960), "--height", str(height or 540),
                                             "--samples", str(samples)],
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

    def observe(self, cameras, world=None, *, body_q=None, tick=None, meshes=None, resolution=None, visibility=None,
                resolutions=None, holes=None, shortened=None, lights=None):
        """Render `cameras` for the world's current state, or for recorded body_q with the recorded hole records
        `holes` ({drillable: [Hole]}, from which the workpieces and debris are drawn) and/or explicit `meshes`."""
        w = world or self.world
        t0 = time.perf_counter()
        if body_q is None:
            body_q = w.bq
            tick = w.tick
            meshes = self.cache.changed(force=self._first)
            self._first = False
        else:
            extra = meshes or {}
            meshes = self.cache.changed(force=self._first, holes=holes, bq=body_q) if holes is not None else {}
            if holes is not None:
                self._first = False
            meshes.update(extra)
        meshes.update(self.hose.meshes(body_q))
        if shortened is None and holes is None and tick == w.tick:            # the live world
            shortened = {n: it.state.get("shortened_m") for n, it in w.hm.items.items()}
        vis, xforms = broken_bit_updates(self.bits, shortened or {}, self._short_sent)
        if lights is None and holes is None and tick == w.tick:              # the live world: the tools' own lights
            lights = {n: getattr(t, "s_light", "off") for n, t in w.scene.tools.items()}
        vis.update(wrench_light_updates(self.leds, lights, self._light_sent))
        self.last_mesh_s = time.perf_counter() - t0
        msg = dict(op="observe", tick=int(tick or 0), body_poses=np.asarray(body_q, float).tolist(), cameras=list(cameras),
                   meshes=encode_meshes(meshes or {}), visibility=dict(vis, **(visibility or {})), xforms=xforms)
        res = {}
        for c in cameras:
            cam = w.scene.cameras.get(c, {})
            res[c] = list((resolutions or {}).get(c) or resolution or
                          (self.width or cam.get("width", 960), self.height or cam.get("height", 540)))
        msg["resolutions"] = res
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

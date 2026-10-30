"""USD export of the simulated state and the shared render-only Isaac RTX worker (wiping.render_worker).

The worker never steps physics. Each request carries the recorded poses of every posed body (robot links from
the robot simulator, packaging items from the contact world, ASSISTED phantom tools) and the deformed sheet
surfaces. A sheet is drawn as two single-sided skins offset by its real caliper: the printed outside and the
inside/back, one vertex per triangle corner (fixed topology); a fully torn (damaged) triangle collapses to its
centroid so tears show without changing the mesh. A tear that splits vertices (IPCWorld._split) shows directly: the
corners of the triangles on each side of the crack follow their own copy of the split vertex.
"""
from gym_anything_robotics.paths import worker_env

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import socket
import subprocess
import threading
import time

import numpy as np

from . import artwork

COLORS = dict(
    counter_laminate=((.86, .84, .80), .35, 0.), counter_mat=((.10, .10, .11), .8, 0.), table_wood=((.33, .22, .14), .45, 0.),
    steel_top=((.66, .67, .68), .22, .95), wood=((.55, .40, .26), .5, 0.), furniture=((.5, .5, .5), .6, 0.),
    robot=((.86, .87, .86), .3, 0.), robot_dark=((.07, .075, .08), .4, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    aluminum=((.62, .64, .66), .25, .9), bag_plate=((.20, .008, .019), .5, 0.), chipboard=((.55, .52, .47), .85, 0.),
    giftbox_base=((.08, .13, .26), .45, 0.), giftbox_lid=((.08, .13, .26), .45, 0.), photo_frame=((.12, .10, .09), .35, 0.),
    tape_clear=((.80, .81, .80), .85, 0.), rope=((.95, .93, .88), .7, 0.), kraft=((.62, .45, .28), .85, 0.),
    phantom_tool=((1., .45, .02), .3, 0.), wrap_back=((.95, .95, .93), .6, 0.),
    # receiving preset (carton.py): kraft-liner corrugated board, SBS tea cartons, the glass jar, tan 3M 371 tape
    corrugated=((.60, .44, .28), .88, 0.), tea_carton=((.16, .30, .22), .5, 0.), glass_jar=((.80, .86, .84), .06, 0.),
    packing_tape=((.52, .36, .19), .38, 0.))     # tan (buff) film: 3M 371 tan. Under the stock room's panel lights a
    # lighter, glossier tan (.84/.70/.46 at roughness .16, then .70/.52/.30 at .22) rendered as cream


def _normals(x, tris):
    fn = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


_CORNER_FACES = {}


def _corner_faces(T):
    """(corner index, incident face index) pairs: every face sharing the corner's vertex. Cached per triangle array and
    checked against its content: a tear splits vertices (IPCWorld._split), after which corners on the two sides of the
    crack no longer share faces (the skins then open along the crack)."""
    key = (id(T), T.shape)
    hit = _CORNER_FACES.get(key)
    if hit is None or not np.array_equal(hit[0], T):
        nv = int(T.max()) + 1
        inc = [[] for _ in range(nv)]
        for f, tri in enumerate(T):
            for v in tri:
                inc[v].append(f)
        ci, fi = [], []
        for c, v in enumerate(T.ravel()):
            ci.extend([c]*len(inc[v])); fi.extend(inc[v])
        while len(_CORNER_FACES) >= 16:
            _CORNER_FACES.pop(next(iter(_CORNER_FACES)))
        hit = _CORNER_FACES[key] = (T.copy(), np.asarray(ci), np.asarray(fi))
    return hit[1:]


def corner_normals(x, T, cos_max=.5):
    """Per-corner normals smoothed only over incident faces within 60 degrees of the corner's own face: a
    vertex on a 180-degree fold (stacked layers) keeps each layer's own normal instead of their zero mean."""
    fn = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
    fu = fn/np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    ci, fi = _corner_faces(T)
    own = np.repeat(np.arange(len(T)), 3)[ci]
    w = (np.einsum("ij,ij->i", fu[own], fu[fi]) > cos_max)[:, None]*fn[fi]
    acc = np.zeros((3*len(T), 3))
    np.add.at(acc, ci, w)
    return acc/np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)


def sheet_skins(ipc, k):
    """(front, back, normals) per triangle corner for sheet k of the contact world."""
    sl = ipc.sheet_slices[k]
    sheet = ipc.sheets[k]
    x = ipc.x_free()[sl]
    T = np.asarray(sheet.tris)
    nc = corner_normals(x, T)
    half = .5*max(m.t for m in sheet.mats)
    C = T.ravel()
    front, back = x[C] + nc*half, x[C] - nc*half
    tri_off = int(np.searchsorted(ipc.sheet_of_tri, k))
    dmg = ipc.damage[tri_off:tri_off + len(T)] >= 1. if len(getattr(ipc, "damage", [])) else np.zeros(len(T), bool)
    if dmg.any():
        dead = np.repeat(dmg, 3)
        cen = np.repeat(x[T].mean(1), 3, axis=0)
        front[dead] = cen[dead]; back[dead] = cen[dead]
    return front.astype(np.float32), back.astype(np.float32), nc.astype(np.float32)


def _print_side(ipc, k):
    """+1 if the printed side of sheet k is its +normal side at tick zero (outward from the item it wraps)."""
    sheet = ipc.sheets[k]
    kind = sheet.render.get("kind")
    if kind == "tote":
        return 1.                                   # bagnet orients every triangle outward
    sl = ipc.sheet_slices[k]
    x = np.asarray(sheet.x0)
    T = np.asarray(sheet.tris)
    n = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
    c = x[T].mean(1) - x.mean(0)
    return 1. if (n*c).sum() >= 0 else -1.


def _sheet_uv(sheet):
    """Texture coordinates per triangle corner."""
    T = np.asarray(sheet.tris)
    kind = sheet.render.get("kind")
    if kind == "tote":
        from . import bagnet as B
        X, kinds = np.asarray(sheet.X2), sheet.render["kinds"]
        Wd, G, H = B.W_BAG, B.G_BAG, B.H_BAG
        _, (us, vs, voff) = sheet.render["atlas"]
        uv = np.zeros((len(T), 3, 2))
        for t, tri in enumerate(T):
            P = X[tri]
            k = kinds[t]
            if k == "front":
                u, v = P[:, 0] + Wd/2, P[:, 2]
            elif k == "right":
                u, v = Wd + (P[:, 1] + G/2), P[:, 2]
            elif k == "back":
                u, v = Wd + G + (Wd/2 - P[:, 0]), P[:, 2]
            elif k == "left":
                u, v = 2*Wd + G + (G/2 - P[:, 1]), P[:, 2]
            else:                                   # bottom, under the front
                u, v = P[:, 0] + Wd/2, -(P[:, 1] + G/2)
            uv[t] = np.c_[u/us, (v + voff)/vs]
        return uv.reshape(-1, 2).astype(np.float32)
    X2 = np.asarray(sheet.X2)
    if X2.shape[1] == 3:
        X2 = X2[:, :2]
    span = .20 if kind == "wrap" else .05
    return (X2[T.ravel()]/span).astype(np.float32)


def _body_visual(stage, path, body, mats):
    """Visual geometry of a contact-world rigid item in its COM frame (its pose is the posed body prim)."""
    from whiteboard.usd_scene import bevel_box
    from pxr import Gf, UsdGeom, UsdShade
    r = body.render
    mat = mats.get(r.get("material"), mats["furniture"])
    if "outline" in r:
        from .scene import prism_mesh
        from pencap.newton15.render import mesh
        V, F = prism_mesh(np.asarray(r["outline"]), r["thickness"])
        g = mesh(stage, f"{path}/plate", V, F.ravel(), smooth=False)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
        return
    if r.get("material") == "photo_frame":
        _photo_frame(stage, path, np.asarray(r["half"]), mats)
        return
    if r.get("material") == "candle_carton":
        half = np.asarray(r["half"])
        g = bevel_box(stage, f"{path}/box", 2*half, .0015)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats["carton_white"])
        from whiteboard.usd_scene import mesh as qmesh
        for side, sy in (("front", -1.), ("back", 1.)):
            y = sy*(half[1] + 2e-4)
            c = np.array([[-half[0]*.97, y, -half[2]*.97], [half[0]*.97, y, -half[2]*.97],
                          [half[0]*.97, y, half[2]*.97], [-half[0]*.97, y, half[2]*.97]])
            if sy > 0:
                c = c[[1, 0, 3, 2]]
            q = qmesh(stage, f"{path}/print_{side}", c, [np.array([0, 1, 2, 3])], np.array([[0, 0], [1, 0], [1, 1], [0, 1.]]))
            UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(mats["candle_print"])
        return
    if r.get("material") == "corrugated" and "carton_face" in mats:     # liner skins, cut edges, print (carton.draw)
        from .carton import draw as draw_carton
        draw_carton(stage, path, body, mats)
        return
    if "parts" in r:
        com = np.asarray(r["com_offset"])
        for j, (half, c) in enumerate(r["parts"]):
            g = bevel_box(stage, f"{path}/part_{j}", 2*np.asarray(half), min(.0006, .4*min(half)))
            UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, np.asarray(c) - com)))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    elif "half" in r:
        half = np.asarray(r["half"])
        g = bevel_box(stage, f"{path}/box", 2*half, min(.0015, .3*float(half.min())))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)


FRAME_MOULDING = .013      # m, face width of the frame's moulding [E]; the frame's outer size is the physical body's


def _photo_frame(stage, path, half, mats):
    """How the photo frame is drawn (its physics is the box body, unchanged): a walnut moulding with a gilt inner
    fillet around a recessed mat board and print under glass, a hardboard back; all inside the body's box."""
    from whiteboard.usd_scene import bevel_box, mesh
    from pxr import Gf, UsdGeom, UsdShade
    hx, hy, hz = map(float, half)
    mw = FRAME_MOULDING

    def part(name, size, c, mat, bevel=.0008):
        g = bevel_box(stage, f"{path}/{name}", np.asarray(size, float), bevel)
        UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, c)))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats[mat])
    part("back", (2*hx - .004, 2*hy - .004, .004), (0., 0., -hz + .002), "frame_back")
    for sy in (-1, 1):
        part(f"rail_{'np'[sy > 0]}", (2*hx, mw, 2*hz), (0., sy*(hy - mw/2), 0.), "frame_moulding", bevel=.002)
        part(f"fillet_{'np'[sy > 0]}", (2*hx - 2*mw, .0018, 2*hz - .003), (0., sy*(hy - mw - .0009), -.0015), "frame_fillet", bevel=.0004)
    for sx in (-1, 1):
        part(f"stile_{'np'[sx > 0]}", (mw, 2*hy - 2*mw + .0004, 2*hz), (sx*(hx - mw/2), 0., 0.), "frame_moulding", bevel=.002)
        part(f"fillet_x{'np'[sx > 0]}", (.0018, 2*hy - 2*mw, 2*hz - .003), (sx*(hx - mw - .0009), 0., -.0015), "frame_fillet", bevel=.0004)
    ox, oy = hx - mw - .0018, hy - mw - .0018
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
    for name, zc, mat in (("print", hz - .0045, "frame_print"), ("glass", hz - .0032, "frame_glass")):
        q = mesh(stage, f"{path}/{name}", np.array([[-ox, -oy, zc], [ox, -oy, zc], [ox, oy, zc], [-ox, oy, zc]]),
                 [np.array([0, 1, 2, 3])], uv)
        UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(mats[mat])


def _worker_exposure():
    """The render worker's fixed camera exposure (it records size and samples but not these), read from its source
    and pinned by the source's SHA-256."""
    src = Path(__file__).resolve().parents[1]/"wiping"/"render_worker.py"
    try:
        text = src.read_text()
    except OSError:
        return dict(source=str(src), error="unreadable")
    kit = src.with_name("render_only.kit")
    return dict(source="wiping/render_worker.py", sha256=hashlib.sha256(text.encode()).hexdigest(),
                render_only_kit_sha256=hashlib.sha256(kit.read_bytes()).hexdigest() if kit.exists() else None,
                renderer=(re.findall(r'renderer="(\w+)"', text) or [None])[0],
                settings={k: v.strip() for k, v in re.findall(r'"(/rtx/post/[^"]+)":\s*([^,}]+)', text)})


def _material(stage, mats, name, color, rough, metal, texture=None, opacity=None):
    from pxr import Gf, Sdf, UsdShade
    m = UsdShade.Material.Define(stage, "/Materials/"+name)
    s = UsdShade.Shader.Define(stage, str(m.GetPath())+"/surface")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    if opacity is not None:
        s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if texture:
        rd = UsdShade.Shader.Define(stage, str(m.GetPath())+"/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, str(m.GetPath())+"/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(texture).resolve())))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    mats[name] = m
    return m


TOOL_SLOTS = 4      # ASSISTED phantom proxies (kind 0 none, 1 cylinder, 2 box), see phantom rows in snapshot()


def layout(world):
    """Posed-body index layout shared by export() and snapshot(): robot bodies, then dynamic items, then tools."""
    nr = world.rig.model.body_count
    items = [i for i, b in enumerate(world.ipc.bodies) if b.kind == "dynamic"]
    return nr, items


def cameras(world):
    """Fixed cameras: name -> (eye, target, focal mm on a 36 mm aperture, role)."""
    z = world.scene.surface_z
    p = world.c.preset
    cams = dict(overhead_rgb=((0., -.18, z + 1.15), (0., .16, z), 16., "policy"),
                room=((1.45, -1.75, z + .75), (0., .25, z + .05), 24., "diagnostic"),
                front=((0., -.85, z + .45), (0., .18, z + .02), 28., "diagnostic"))
    if p == "counter":
        cams.update(counter_wide=((.18, -.62, z + .52), (.18, .18, z + .14), 22., "diagnostic"),
                    bag=((.62, -.20, z + .32), (.36, .16, z + .01), 32., "diagnostic"),
                    mat=((.25, -.40, z + .40), (0., .17, z + .08), 30., "diagnostic"),
                    # the opened bag on the mat (centre (0, 0.17), 203 x 102 mm): straight down into it (the bottom, the
                    # board, the candle), and low from the side at the foot of its right gusset, where it tore in
                    # rounds 2 and 3 with no camera on it
                    bag_inside=((.015, .12, z + .66), (0., .17, z), 35., "diagnostic"),
                    gusset_foot=((.34, .13, z + .075), (.10, .17, z + .020), 50., "diagnostic"))
    elif p == "receiving":
        cams.update(carton=((.48, -.30, z + .50), (0., .15, z + .14), 30., "diagnostic"),
                    carton_end=((.62, .02, z + .24), (.18, .15, z + .16), 35., "diagnostic"))   # the +x end: tape tab
    elif p == "home":
        cams.update(gift=((.32, -.25, z + .26), (0., .17, z + .04), 32., "diagnostic"),
                    side=((.55, .17, z + .12), (0., .17, z + .04), 32., "diagnostic"),
                    gift_end=((.36, .02, z + .06), (.09, .17, z + .035), 45., "diagnostic"),   # the +x taped end, 30 deg off its normal
                    # the same end in profile from the front of the table, low: a flap swinging out toward +x is seen
                    # edge-on and a hand working on the end face stands beside it, not in front of it (the robot
                    # unwrap films of rounds 1-2 were shot from `side` and `gift`: the arm filled or blocked both)
                    gift_end_profile=((.13, -.30, z + .07), (.11, .17, z + .035), 38., "diagnostic"))
    return cams


def phantom_specs(tools):
    """Distinct (slot, half extents) of ASSISTED box tools appearing in a recording (tools: (frames, slots, 11))."""
    out = []
    T = np.asarray(tools)
    for s in range(T.shape[1]):
        act = T[:, s, 0] > 0
        for row in (np.unique(np.round(T[act, s, 1:4], 5), axis=0) if act.any() else []):
            out.append((s, tuple(map(float, row))))
    return out


def export(world, folder, phantoms=()):
    """Write scene.usdc for this world (static scene + posed bodies + sheet skins). `phantoms`: ASSISTED tool
    proxies [(slot, half extents)], one posed body each after the robot and item bodies. Returns the path."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    import newton
    from pencap.newton15.render import pose, mesh, camera
    from whiteboard.usd_scene import bevel_box
    folder = Path(folder)
    (folder/"textures").mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    mats = {}
    for name, v in COLORS.items():
        _material(stage, mats, name, *v)
    tex = folder/"textures"
    # matte office tape (3M 810): on the paper it is all but invisible, a faint matte strip. A film 12 % opaque
    # with a full dielectric highlight reads as a white label under the pendant lamp, so the highlight is that of a
    # matte film (specular workflow, 1 %), 6 % opaque.
    sh = UsdShade.Shader.Get(stage, "/Materials/tape_clear/surface")
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(.06)
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.01, .01, .01))
    sh = UsdShade.Shader.Get(stage, "/Materials/phantom_tool/surface")
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(.45)
    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.35, .12, 0.))
    _material(stage, mats, "candle_print", None, .45, 0., texture=artwork.candle_carton(tex))
    _material(stage, mats, "carton_white", (.80, .76, .68), .45, 0.)
    if world.c.preset == "receiving":                                    # corrugated board: liner, cut edge, print
        from .carton import textures as carton_textures
        for k_, p_ in carton_textures(tex).items():
            _material(stage, mats, f"carton_{k_}", None, .9, 0., texture=p_)
    _material(stage, mats, "chipboard_tex", None, .85, 0., texture=artwork.plain(tex, "chipboard", (.56, .53, .48)))
    # photo frame (drawn by _photo_frame): walnut moulding, gilt fillet, hardboard back, mat + print, glass
    _material(stage, mats, "frame_moulding", (.13, .065, .035), .38, 0.)
    _material(stage, mats, "frame_fillet", (.72, .55, .24), .30, .85)
    _material(stage, mats, "frame_back", (.30, .22, .15), .85, 0.)
    fh = world.scene.geometry.get("frame", {}).get("half", (.085, .060, .0075))
    _material(stage, mats, "frame_print", None, .75, 0., texture=artwork.framed_print(
        tex, 2*(fh[0] - FRAME_MOULDING - .0018), 2*(fh[1] - FRAME_MOULDING - .0018),
        (2*(fh[0] - FRAME_MOULDING) - .026, 2*(fh[1] - FRAME_MOULDING) - .026)))
    _material(stage, mats, "frame_glass", (.95, .97, .97), .03, 0., opacity=.05)
    UsdShade.Shader.Get(stage, "/Materials/frame_glass/surface").CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.)
    mats["chipboard"] = mats["chipboard_tex"]
    nr, items = layout(world)
    rig, ipc = world.rig, world.ipc
    b = rig.b
    q = rig.state.body_q.numpy()
    for i in range(nr):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[i, :3], q[i, 3:])
    # robot: the visible meshes (flag VISIBLE without collision) in their asset colours (sRGB -> linear)
    for i in range(b.shape_count):
        body, flags = b.shape_body[i], int(b.shape_flags[i])
        if body < 0 or not (flags & 1):
            continue
        src, scale = b.shape_source[i], np.asarray(b.shape_scale[i])
        path = f"/Lab/body_{body}/shape_{i}"
        if src is not None and hasattr(src, "vertices"):
            g = mesh(stage, path, np.asarray(src.vertices)*scale, src.indices)
        elif b.shape_type[i] == newton.GeoType.BOX:
            g = bevel_box(stage, path, 2*scale, min(.0015, float(scale.min())*.2))
        else:
            continue
        tf = np.asarray(b.shape_transform[i])
        pose(g, tf[:3], tf[3:])
        label = b.shape_label[i]
        if label.endswith(("_camera_case", "_camera_bracket")):
            m = mats["camera_dark" if label.endswith("_case") else "aluminum"]
        else:
            col = np.asarray(b.shape_color[i], float)**2.2
            key = "robot_c_" + "_".join(f"{int(round(c*255)):03d}" for c in col)
            m = mats.get(key) or _material(stage, mats, key, col, .32, 0.)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
    # static furniture (authored collision boxes, drawn as they are; wood tops with the pinned veneer scans)
    from . import dressing
    dresser = dressing.Dresser(stage, folder)
    scans = dict(table_wood=("black_oak_veneer", 1.2, .45))
    for f in world.scene.furniture:
        if f.get("visual") is False:              # a prop's collision box: the dressing draws the prop itself
            continue
        g = bevel_box(stage, f"/Lab/furniture_{f['name']}", 2*np.asarray(f["half"]), .002)
        UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(*map(float, f["center"])))
        mname = f.get("material")
        m = dresser.scan(scans[mname][0], scans[mname][1], roughness=scans[mname][2]) if mname in scans else \
            mats.get(mname, mats["furniture"])
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
    # contact-world items
    for j, bi in enumerate(items):
        body = ipc.bodies[bi]
        p, R = ipc.body_pose(bi)
        from scipy.spatial.transform import Rotation
        prim = UsdGeom.Xform.Define(stage, f"/Lab/body_{nr + j}")
        pose(prim, p, Rotation.from_matrix(R).as_quat())
        _body_visual(stage, f"/Lab/body_{nr + j}", body, mats)
    # ASSISTED phantom tool proxies (translucent amber; posed per frame, parked below the floor when unused)
    for j, (slot, half) in enumerate(phantoms):
        base = nr + len(items) + j
        prim = UsdGeom.Xform.Define(stage, f"/Lab/body_{base}")
        pose(prim, (0., 0., -5.), (0., 0., 0., 1.))
        g = bevel_box(stage, f"/Lab/body_{base}/phantom", 2*np.asarray(half), min(.002, .3*min(half)))
        g.GetPrim().SetCustomDataByKey("assisted_phantom", True)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats["phantom_tool"])
    # sheets: printed outside + inside skins
    for k, sheet in enumerate(ipc.sheets):
        kind = sheet.render.get("kind")
        if kind == "tote":
            from . import bagnet as B
            sheet.render["atlas"] = artwork.tote_print(tex, B.W_BAG, B.G_BAG, B.H_BAG, B.TURN_TOP)
            m_out = _material(stage, mats, f"sheet_{k}_out", None, .55, 0., texture=sheet.render["atlas"][0])
            m_in = _material(stage, mats, f"sheet_{k}_in", None, .8, 0., texture=artwork.tote_inside(tex))
        elif kind == "wrap":
            m_out = _material(stage, mats, f"sheet_{k}_out", None, .25, 0., texture=artwork.wrap_pattern(tex))
            m_in = mats["wrap_back"]
        elif kind == "packing_tape":
            m_out = m_in = mats["packing_tape"]
        elif kind == "kraft":
            m_out = m_in = _material(stage, mats, f"sheet_{k}_kraft", None, .85, 0., texture=artwork.plain(tex, "kraft", (.62, .45, .28)))
        else:
            m_out = m_in = mats["tape_clear"]
        side = _print_side(ipc, k)
        sheet.render["print_side"] = side
        front, back, n = sheet_skins(ipc, k)
        out, inn, n_out = (front, back, n) if side > 0 else (back, front, -n)
        f = np.arange(len(front)).reshape(-1, 3)
        uv = _sheet_uv(sheet)
        for nm, pts, faces, nn, mtl in (("out", out, f if side > 0 else f[:, ::-1], n_out, m_out),
                                       ("in", inn, f[:, ::-1] if side > 0 else f, -n_out, m_in)):
            g = mesh(stage, f"/Lab/sheet_{k}_{nm}", pts, faces.ravel(), smooth=False)
            g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nn.astype(np.float32))); g.SetNormalsInterpolation("vertex")
            g.CreateDoubleSidedAttr(False)
            g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([pts.min(0), pts.max(0)], np.float32)))
            UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(Vt.Vec2fArray.FromNumpy(uv))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mtl)
    # rope handles: tubes of the rope's radius swept along each rod's centre line (rope_render.py)
    from . import rope_render as RR
    mats.setdefault("rope_knot", mats["rope"])
    for k, P in RR.rope_polylines(ipc).items():
        rod = ipc.rods[k]
        pts, nn = RR.tube_points(P, rod.mat.radius)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        g = mesh(stage, f"/Lab/rope_{k}", pts, RR.tube_faces(len(P)).ravel(), smooth=True)
        g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nn)); g.SetNormalsInterpolation("vertex")
        g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([pts.min(0) - .01, pts.max(0) + .01], np.float32)))
        UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            Vt.Vec2fArray.FromNumpy(RR.tube_uv(len(P), length=L)))
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats["rope"])
    cams = cameras(world)
    credits = dressing.dress(stage, world, folder, eye_policy=cams["overhead_rgb"][0], d=dresser)
    for name, (eye, target, focal, role) in cams.items():
        camera(stage, name, eye, target, focal, role=role)
    for name, m in rig.cameras.items():
        c = UsdGeom.Camera.Define(stage, f"/Lab/body_{m['body']}/{name}")
        c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*m["eye"]), Gf.Vec3d(*m["target"]), Gf.Vec3d(*m["up"])).GetInverse())
        c.CreateHorizontalApertureAttr(36.); c.CreateVerticalApertureAttr(20.25)
        c.CreateFocalLengthAttr(18/math.tan(math.radians(m["fov"]/2)))
        c.CreateClippingRangeAttr(Gf.Vec2f(.005, 30.))
        c.GetPrim().SetCustomDataByKey("observation_role", "policy")
        alias = UsdGeom.Xform.Define(stage, "/Cameras/"+name)
        alias.GetPrim().SetCustomDataByKey("mounted_camera_path", str(c.GetPath()))
    stage.GetRootLayer().Save()
    lights = dressing.lighting_record(stage)
    (folder/"export.json").write_text(json.dumps(dict(
        preset=world.c.preset, task=world.c.task, tick=world.steps, phantoms=[list(p) for p in phantoms], visual_assets=credits,
        texture_substitutions=getattr(dresser, "texture_substitutions", {}), lighting=lights,
        lighting_sha256=hashlib.sha256(json.dumps(lights, sort_keys=True).encode()).hexdigest(),
        camera_exposure=_worker_exposure()), indent=1))
    return folder/"scene.usdc"


def snapshot(world, tools=None):
    """Recordable render state of the current step: posed bodies (7-vectors) and sheet skins."""
    from scipy.spatial.transform import Rotation
    nr, items = layout(world)
    q = world.rig.state.body_q.numpy()
    poses = [list(map(float, q[i])) for i in range(nr)]
    for bi in items:
        p, R = world.ipc.body_pose(bi)
        poses.append(list(map(float, p)) + list(map(float, Rotation.from_matrix(R).as_quat())))
    sheets = {}
    for k, sheet in enumerate(world.ipc.sheets):
        front, back, n = sheet_skins(world.ipc, k)
        sheets[k] = (front, back, n)
    from .rope_render import rope_polylines
    return dict(tick=world.steps, time=world.sim_time, poses=np.asarray(poses, np.float32), sheets=sheets,
                rods={k: v.astype(np.float32) for k, v in rope_polylines(world.ipc).items()},
                tools=np.zeros((TOOL_SLOTS, 11), np.float32) if tools is None else np.array(tools, np.float32, copy=True))


class Renderer:
    """Owns exactly one render-only Isaac worker (wiping.render_worker protocol)."""

    def __init__(self, world, folder, *, width=1280, height=720, samples=32, phantoms=(), scene=None):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.closed, self.lock, self.sequence, self.log_position = False, threading.Lock(), 0, 0
        self.phantoms = list(phantoms)
        if scene is None or not Path(scene).is_file() or self.phantoms:
            scene = export(world, self.folder, phantoms=phantoms)
        self.side = {k: s.render.get("print_side", 1.) for k, s in enumerate(world.ipc.sheets)}
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        # how long the worker may take to start and to answer one request (s). 300 s is ample on a free machine; with
        # the container's CPU quota exhausted a worker can take longer: GIFT_BAGS_RENDER_TIMEOUT_S raises it
        self.timeout_s = float(os.environ.get("GIFT_BAGS_RENDER_TIMEOUT_S", "300"))
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(self.timeout_s)
        secret = secrets.token_urlsafe(32)
        env = dict(os.environ, WIPE_RENDER_PORT=str(listener.getsockname()[1]), WIPE_RENDER_SECRET=secret,
                   WIPE_MINIMAL_RENDER="1", WIPE_TEXTURE_STREAMING="1", WIPE_TEXTURE_BUDGET=".05")
        self.log = (self.folder/"worker.log").open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "wiping.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=Path(__file__).resolve().parents[1], env=worker_env(env),
                                            stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept(); self.connection.settimeout(self.timeout_s)
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
        # one record per render: the worker's effective settings next to the scene's lighting and exposure
        here = Path(scene).resolve().parent
        try:
            record = json.loads((here/"export.json").read_text())
            record["renderer_settings"] = json.loads((here/"renderer_settings.json").read_text())
            (here/"export.json").write_text(json.dumps(record, indent=1))
        except (OSError, ValueError):
            pass

    def observe(self, cameras=("room",), state=None):
        """Render a recorded state (default: the live world). Returns {camera: png bytes}."""
        from whiteboard.ipc import send, receive
        from wiping.render_payload import encode
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            state = snapshot(self.world) if state is None else state
            meshes = {}
            for k, (front, back, n) in state["sheets"].items():
                k = int(k)
                if self.side.get(k, 1.) > 0:
                    out, inn, n_out = front, back, n
                else:
                    out, inn, n_out = back, front, -n
                meshes[f"/Lab/sheet_{k}_out"] = dict(points=encode(out, "float32"), normals=encode(n_out, "float32"))
                meshes[f"/Lab/sheet_{k}_in"] = dict(points=encode(inn, "float32"), normals=encode(-n_out, "float32"))
            from .rope_render import tube_points
            for k, P in (state.get("rods") or {}).items():
                pts, nn = tube_points(P, self.world.ipc.rods[int(k)].mat.radius)
                meshes[f"/Lab/rope_{int(k)}"] = dict(points=encode(pts, "float32"), normals=encode(nn, "float32"))
            poses = np.asarray(state["poses"]).tolist()
            tools = np.asarray(state.get("tools", np.zeros((TOOL_SLOTS, 11))))
            for slot, half in self.phantoms:
                row = tools[slot]
                live = row[0] > 0 and np.allclose(np.round(row[1:4], 5), half)
                poses.append(list(map(float, row[4:11])) if live else [0., 0., -5., 0., 0., 0., 1.])
            self.sequence += 1
            send(self.connection, dict(op="observe", sequence=self.sequence, tick=int(state["tick"]), body_poses=poses,
                                       cameras=list(cameras), meshes=meshes, material_fields={}, details={}))
            deadline = time.monotonic() + self.timeout_s
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
            if not result.get("ok") or result.get("physics_steps") != 0:
                raise RuntimeError("Invalid render response: " + str(result)[:400])
            return {k: base64.b64decode(v) for k, v in result["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

"""USD export of the dish room and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; each request carries the body poses and what the water did: per-chart overlay
textures (soil, foam, water sheen and dirty water drawn from the cells' state; alpha = how much of it there is, so a clean
dry glaze is the bare surface underneath), the airborne packets as short motion segments (jets and drops) in a water
dielectric, and the pools as closed water volumes (sink bowls: their level; cavities: the free surface across the tilted
cavity). Nothing is painted on by the renderer: every pixel of water, soap or soil is a field of the physics state.

Appearance rules: collision primitives are drawn as the parts they stand for; revolved ware is drawn from its (r, z)
polygon (the colliders are hidden hulls); the robot uses the franka_description visual meshes (classroom/painting
drawing); room dressing outside the working volumes comes from pinned CC0 Poly Haven assets (visual_assets.py).
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.render import Stage, clip_mesh
from wine_pouring import geometry as G
from . import params as P
from . import layout as L
from . import ware as W

MDL_FILE = Path(__file__).with_name("SurfaceRinsing.mdl")
COLORS = dict(
    steel=((.62, .63, .64), .30, 1.), steel_brushed=((.58, .59, .60), .38, 1.), chrome=((.92, .92, .93), .07, 1.),
    ceiling=((.88, .88, .86), .9, 0.), door=((.45, .47, .50), .45, .6), plastic_white=((.88, .88, .86), .4, 0.),
    rack_grey=((.30, .34, .40), .55, 0.), hose_steel=((.70, .71, .72), .35, .9), hose_green=((.08, .32, .12), .5, 0.),
    reel_red=((.55, .05, .04), .4, .2), rubber_grey=((.20, .20, .21), .7, 0.), glaze=((.93, .93, .91), .08, 0.),
    polymer=((.42, .44, .44), .55, 0.), hdpe=((.90, .90, .88), .65, 0.), tomato=((.62, .045, .02), .22, 0.),
    calyx=((.18, .32, .08), .6, 0.), shelf_wire=((.80, .80, .80), .25, .9), robot=((.88, .88, .87), .35, 0.),
    robot_dark=((.07, .075, .08), .45, 0.), camera_dark=((.03, .03, .035), .45, 0.), aluminum=((.70, .72, .74), .3, .85),
    tmr_black=((.05, .05, .055), .5, 0.), tmr_deck=((.80, .81, .80), .4, 0.), grout=((.55, .55, .53), .9, 0.),
    hole_dark=((.03, .03, .032), .8, 0.), foam=((.62, .63, .62), .55, 0.), scour_green=((.05, .30, .10), .9, 0.), sponge_yellow=((.85, .70, .18), .95, 0.), led=((1., 1., 1.), .3, 0.), frame_white=((.92, .92, .91), .5, 0.),
)
WATER_ABS = (0.45, 0.06, 0.02)                        # 1/m: pure water, red-absorbing
WASH_ABS = (6.0, 6.5, 7.5)                            # soapy wash water: cloudy (scatter approximated as absorption)


def renderer_provenance():
    """Which Isaac Sim renders the RGB (recorded with every still, video and camera timing): the interpreter in
    ISAAC_RENDER_PYTHON and the isaacsim version installed beside it. Isaac Sim 6.0.1 renders it: the 5.1 worker does
    not start on driver 595.91.07."""
    exe = os.environ.get("ISAAC_RENDER_PYTHON", "")
    version = None
    if exe:
        for d in sorted(Path(exe).parent.parent.glob("lib/python3*/site-packages/isaacsim-*.dist-info")):
            version = d.name[len("isaacsim-"):-len(".dist-info")]
    return dict(isaac_render_python=exe, isaacsim_version=version)


def _wxyz(R):
    x, y, z, w = R.as_quat()
    return float(w), float(x), float(y), float(z)


def _mdl(S, name, module, inputs, path=None):
    if name in S.mats:
        return S.mats[name]
    UsdShade = S.UsdShade
    m = UsdShade.Material.Define(S.stage, "/Looks/" + name)
    sh = UsdShade.Shader.Define(S.stage, f"/Looks/{name}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(S.Sdf.AssetPath(path or (module + ".mdl")), "mdl")
    sh.SetSourceAssetSubIdentifier(module, "mdl")
    for k, (typ, v) in inputs.items():
        sh.CreateInput(k, typ).Set(v)
    m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    S.mats[name] = m
    return m


def _dielectric(S, name, absorption, ior):
    T, Gf = S.Sdf.ValueTypeNames, S.Gf
    return _mdl(S, name, "RinsingDielectric", {"absorption": (T.Color3f, Gf.Vec3f(*map(float, absorption))),
                                                "dielectric_ior": (T.Float, float(ior))},
                path=str((S.folder/MDL_FILE.name).resolve()))


def _overlay_mat(S, name):
    """OmniPBR whose diffuse colour and opacity (alpha) come from one dynamic texture, roughness from another."""
    T, Sdf = S.Sdf.ValueTypeNames, S.Sdf
    return _mdl(S, "ov_" + name, "OmniPBR", {
        "diffuse_texture": (T.Asset, Sdf.AssetPath(f"dynamic://ov_{name}")),
        "enable_opacity": (T.Bool, True), "enable_opacity_texture": (T.Bool, True),
        "opacity_texture": (T.Asset, Sdf.AssetPath(f"dynamic://ov_{name}")), "opacity_mode": (T.Int, 0),
        "reflectionroughness_texture": (T.Asset, Sdf.AssetPath(f"dynamic://rg_{name}")),
        "reflection_roughness_texture_influence": (T.Float, 1.0), "reflection_roughness_constant": (T.Float, 0.5)})


# ------------------------------------------------------------------------------ export
def export(env, folder, extra_cameras=None):
    """Export the episode's static appearance to folder/scene.usdc; returns (path, meta)."""
    world, sc = env.world, env.world.scene
    S = Stage(folder)
    shutil.copyfile(MDL_FILE, S.folder/MDL_FILE.name)
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    _textured(S, mats)
    mats["glass"] = _dielectric(S, "glass_dielectric", (1.5, 1.0, 1.8), 1.52)
    mats["water"] = _dielectric(S, "water_dielectric", WATER_ABS, 1.333)
    mats["wash_water"] = _dielectric(S, "wash_dielectric", WASH_ABS, 1.333)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    meta = dict(textures={}, charts={}, curves={}, pools={}, points={}, foams={})
    _robot(S, world, mats)
    _room(S, mats, env.spec)
    _fixtures(S, world, mats)
    _items(S, world, mats)
    if env.water is not None:
        _water(S, env, mats, meta)
    _lights(S, mats, env.spec)
    _dressing(S, mats, env.spec)
    meta["cameras"] = _cameras(S, world, env.spec)
    for name, (eye, target, f) in (extra_cameras or {}).items():
        if name in meta["cameras"]:
            raise ValueError(f"extra camera {name!r} would replace a built-in camera")
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); meta["cameras"][name] = "diagnostic"
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta, indent=1))
    return S.folder/"scene.usdc", meta


def _textured(S, mats):
    from .visual_assets import texture
    fl = texture("brown_floor_tiles")
    mats["quarry_tile"] = S.textured("quarry_tile", fl["diff"], rough=.55, normal=fl["nor_gl"], rough_tex=fl["rough"],
                                     tint=(0.95, 0.80, 0.70))
    wt = texture("long_white_tiles")
    mats["tile"] = S.textured("wall_tile", wt["diff"], rough=.25, normal=wt["nor_gl"], tint=(1.0, 1.0, 1.0), normal_strength=0.6)
    bl = texture("concrete_block_wall")
    mats["block"] = S.textured("painted_block", bl["diff"], rough=.8, normal=bl["nor_gl"], tint=(1.25, 1.22, 1.15),
                               normal_strength=0.5)
    mats["sign"] = S.textured("sink_sign", _sign_texture(S.folder/"textures"), rough=.5)


def _sign_texture(folder):
    from PIL import Image, ImageDraw, ImageFont
    W_, H_ = 1600, 140
    im = Image.new("RGB", (W_, H_), (250, 250, 248))
    d = ImageDraw.Draw(im)
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(f, 72) if Path(f).exists() else ImageFont.load_default()
    for k, (word, col) in enumerate((("WASH", (200, 30, 30)), ("RINSE", (20, 70, 170)), ("SANITIZE", (20, 120, 50)))):
        x0 = k*W_//3
        d.rectangle([x0 + 8, 8, x0 + W_//3 - 8, H_ - 8], outline=col, width=6)
        d.text((x0 + W_//6, H_//2), word, font=font, fill=col, anchor="mm")
    p = Path(folder)/"sink_sign.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p)
    return p


def _robot(S, world, mats):
    """franka_description visual meshes with their material colours (classroom_arrangement's drawing rules)."""
    import newton
    sc, b = world.scene, world.scene.b
    V, C = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    for i in sc.robot_shapes:
        flags = int(b.shape_flags[i])
        body = b.shape_body[i]
        label = b.body_label[body].split("/")[-1] if body >= 0 else ""
        mname = sc.materials.get(i)
        tf = np.asarray(b.shape_transform[i], float)
        if mname in ("camera_dark", "aluminum"):
            S.box(f"/Lab/body_{body}/shape_{i}", 2*np.asarray(b.shape_scale[i], float), mats[mname], tf[:3], tf[3:])
            continue
        if flags & C or not flags & V:
            continue
        src = b.shape_source[i]
        scale = np.asarray(b.shape_scale[i], float)
        path = f"/Lab/body_{body}/shape_{i}"
        col = tuple(float(x) for x in b.shape_color[i])
        if src is not None and hasattr(src, "vertices"):
            v = np.asarray(src.vertices)*scale
            if np.abs(v).max() > 5.:
                v = v*1e-3
            f = np.asarray(src.indices).reshape(-1, 3)
            if label == "base_link" and len(v) > 100000:
                v2, below, above = clip_mesh(v, f, 0.26)
                for part, faces, m in (("skirt", below, mats["tmr_black"]), ("deck", above, mats["tmr_deck"])):
                    S.mesh(f"{path}_{part}", v2, faces, mat=m, smooth=True, p=tf[:3], q=tf[3:])
                continue
            m = mats["robot"] if label == "base_link" else S.color(
                f"robot_{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}", col, .35, 0.)
            S.mesh(path, v, f, mat=m, smooth=True, p=tf[:3], q=tf[3:])
        elif b.shape_type[i] == newton.GeoType.BOX:
            S.box(path, 2*scale, mats["tmr_black"], tf[:3], tf[3:])
        elif b.shape_type[i] == newton.GeoType.CYLINDER:
            S.cylinder(path, scale[0], 2*scale[1], mats["robot_dark"], tf[:3], tf[3:])


def _part(S, path, p, mats, uv_per_m=None):
    path = path.replace(".", "p").replace("-", "m")
    m = mats.get(p.material, mats["steel"])
    if p.kind == "box":
        S.box(path, 2*np.asarray(p.size), m, p.pos, p.quat, bevel=min(0.002, float(min(p.size))*0.4), uv_per_m=uv_per_m)
    elif p.kind == "cylinder":
        S.cylinder(path, p.size[0], 2*p.size[1], m, p.pos, p.quat, n=28)
    elif p.kind == "capsule":
        r, hh = p.size
        R = Rotation.from_quat(p.quat)
        a, b_ = np.asarray(p.pos) - R.apply((0, 0, hh)), np.asarray(p.pos) + R.apply((0, 0, hh))
        S.tube(path, [a - R.apply((0, 0, r*0.9)), a, b_, b_ + R.apply((0, 0, r*0.9))], r, m, n=16)
    elif p.kind == "sphere":
        S.mesh(path, *_ellipsoid(p.size[0], p.size[0], p.size[0]), mat=m, p=p.pos, q=p.quat)


def _ellipsoid(a, b, c, n=24):
    v, f, nr = [], [], []
    th = np.linspace(0, math.pi, n//2 + 1)
    ph = np.linspace(0, 2*math.pi, n + 1)
    for t in th:
        for p in ph:
            d = np.array([math.sin(t)*math.cos(p), math.sin(t)*math.sin(p), math.cos(t)])
            v.append(d*(a, b, c)); nr.append(d/np.array([a, b, c])**2)
    for i in range(len(th) - 1):
        for j in range(n):
            k0, k1 = i*(n + 1) + j, (i + 1)*(n + 1) + j
            f += [k0, k1, k1 + 1, k0, k1 + 1, k0 + 1]
    nr = np.array(nr); nr /= np.linalg.norm(nr, axis=1, keepdims=True)
    return np.array(v), np.array(f), None, nr


def _room(S, mats, spec):
    X, Y, Z = L.X, L.Y, L.Z
    pts = [(0, 0, 0), (X, 0, 0), (X, Y, 0), (0, Y, 0)]
    S.mesh("/Room/floor", pts, [0, 1, 2, 0, 2, 3], uv=[(p[0]/1.7, p[1]/1.7) for p in pts], normals=[(0, 0, 1)]*4,
           mat=mats["quarry_tile"])
    tile_top = 2.40
    for p in L.static_parts():
        if p.name.startswith("wall_") and p.kind == "box":
            c, h = np.asarray(p.pos), np.asarray(p.size)
            z0, z1 = c[2] - h[2], c[2] + h[2]
            if z0 < tile_top:
                zz = min(z1, tile_top)
                S.box(f"/Room/{p.name}_tile", (2*h[0], 2*h[1], zz - z0), mats["tile"], (c[0], c[1], (z0 + zz)/2), (0, 0, 0, 1),
                      bevel=0.0005, uv_per_m=1/1.27)
            if z1 > tile_top:
                zz = max(z0, tile_top)
                S.box(f"/Room/{p.name}_block", (2*h[0], 2*h[1], z1 - zz), mats["block"], (c[0], c[1], (z1 + zz)/2), (0, 0, 0, 1),
                      bevel=0.0005, uv_per_m=0.5)
            continue
        if p.name == "ceiling":
            S.box("/Room/ceiling", 2*np.asarray(p.size), mats["ceiling"], p.pos, (0, 0, 0, 1), bevel=0.001)
            continue
        if p.name == "sink_sign":
            c, h = np.asarray(p.pos), np.asarray(p.size)
            S.quad("/Room/sink_sign", (c[0], c[1] - h[1] - 0.0005, c[2]), (1, 0, 0), (0, 0, 1), (2*h[0], 2*h[2]), mats["sign"])
            S.box("/Room/sink_sign_back", 2*h, mats["plastic_white"], c, (0, 0, 0, 1), bevel=0.001)
            continue
        if p.name == "drain_strainer":
            S.cylinder("/Room/drain_strainer", p.size[0], 0.002, mats["chrome"], (p.pos[0], p.pos[1], 0.001), n=40)
            for k in range(8):
                S.box(f"/Room/drain_slot_{k}", (2*p.size[0]*0.8, 0.004, 0.0006), S.color("drain_dark", (.03, .03, .03), .8, 0.),
                      (p.pos[0], p.pos[1] - p.size[0]*0.7 + k*p.size[0]*0.2, 0.0022), (0, 0, 0, 1), bevel=0.0002)
            continue
        _part(S, f"/Room/{p.name}", p, mats)
    # basket strainers in the sink bowls' floors (drawn; the drain law is in water.pools)
    for st in L.STATIONS:
        for name, d in L.static_pools(st).items():
            x, y = d["drain_xy"]
            S.cylinder(f"/Room/strainer_{name}", 0.045, 0.002, mats["chrome"], (x, y, d["lo"][2] + 0.001), n=36)


def _fixtures(S, world, mats):
    sc = world.scene
    for name, parts in sc.parts.items():
        if not name.startswith("fx_"):
            continue
        body = sc.bodies[name]
        for p in parts:
            _part(S, f"/Lab/body_{body}/{p.name}", p, mats)
    # hose links: their capsule parts were added straight to the builder; draw them from the builder shapes
    b = sc.b
    import newton
    for i in range(b.shape_count):
        body = b.shape_body[i]
        lab = b.shape_label[i]
        if body < 0 or "hose" not in lab or b.shape_type[i] != newton.GeoType.CAPSULE:
            continue
        r, hh = np.asarray(b.shape_scale[i], float)[:2]
        tf = np.asarray(b.shape_transform[i], float)
        mat = mats["hose_green"] if "reel" in lab else mats["hose_steel"]
        S.tube(f"/Lab/body_{body}/hose_vis", [(0, 0, -hh - r), (0, 0, -hh), (0, 0, hh), (0, 0, hh + r)], r, mat, n=12,
               p=tf[:3], q=tf[3:])


def _items(S, world, mats):
    sc = world.scene
    for name, w in sc.ware.items():
        body = sc.items[name]["body"]
        base = f"/Lab/body_{body}"
        if w.solid is not None:
            c = G.Container(w.kind, w.solid, w.solid, 0., 0., 1., w.material, [])
            v, t, n = c.render_mesh_shaded(96 if w.kind != "can" else 128)
            mat = {"glass": mats["glass"], "glaze": mats["glaze"], "steel": mats["steel"], "polymer": mats["polymer"],
                   "aluminum": mats["aluminum"]}[w.material]
            S.mesh(f"{base}/{w.kind}_vis", v, t, normals=n, mat=mat)
            if w.kind == "can":
                # moulded lip and the two base rails (drawn; the lip is also a collider)
                lip = G.Container("lip", np.array([(P.ITEMS["can"]["R"] - 0.002, P.ITEMS["can"]["H"] - 0.022),
                                                   (0.2795, P.ITEMS["can"]["H"] - 0.012), (0.2795, P.ITEMS["can"]["H"]),
                                                   (P.ITEMS["can"]["R"] - 0.002, P.ITEMS["can"]["H"])]), np.zeros((3, 2)), 0, 0, 1,
                                  "polymer", [])
                lv, lt, ln = lip.render_mesh_shaded(128)
                S.mesh(f"{base}/lip_vis", lv, lt, normals=ln, mat=mats["polymer"])
            for p in w.prims:
                _part(S, f"{base}/{p.name}", p, mats)
            if w.kind == "colander":
                # the perforations (drawn; the water model drains through the same hole centres, pools.outflows): a dark
                # plug through the 0.8 mm wall at every hole, seen from inside and outside
                import trimesh
                holes = np.asarray(w.cavity["holes"], float)
                unit = trimesh.creation.cylinder(radius=0.5*P.ITEMS["colander"]["hole_d"], height=0.0016, sections=10)
                V, F, off = [], [], 0
                for hx, hy, hz, nx, ny, nz in holes:
                    Rh = Rotation.align_vectors([[nx, ny, nz]], [[0., 0., 1.]])[0].as_matrix()
                    V.append(unit.vertices @ Rh.T + (hx, hy, hz)); F.append(unit.faces + off); off += len(unit.vertices)
                S.mesh(f"{base}/perforations", np.concatenate(V), np.concatenate(F), mat=mats["hole_dark"])
            continue
        if w.kind == "sponge":
            # drawn at its full 111 x 66 mm (the colliders are its firm core, ware.sponge)
            from dataclasses import replace
            cf = P.SPONGE["conform"]
            full = lambda p: replace(p, size=(p.size[0] + cf, p.size[1] + cf, p.size[2]))
            for p in w.prims:
                _part(S, f"{base}/{p.name}", full(p), mats)
            top = sc.items[name]["top"]
            for p in w.top:
                _part(S, f"/Lab/body_{top}/{p.name}", full(p), mats)
            continue
        if w.kind == "tomato":
            d = P.ITEMS["tomato"]
            v, f, _, n = _ellipsoid(d["r"], d["r"], d["c"])
            S.mesh(f"{base}/tomato_vis", v, f, normals=n, mat=mats["tomato"])
            S.mesh(f"{base}/calyx", *_ellipsoid(0.005, 0.005, 0.0015, 12)[:2], normals=_ellipsoid(0.005, 0.005, 0.0015, 12)[3],
                   mat=mats["calyx"], p=(0, 0, d["c"] - 0.0003), q=(0, 0, 0, 1))
            continue
        for p in w.prims:
            _part(S, f"{base}/{p.name}", p, mats)


def _water(S, env, mats, meta):
    """Overlay meshes of every water chart (dynamic textures), the airborne packets as curves, pools as volumes."""
    ws = env.water
    surf = ws.surf
    for ch in surf.charts:
        m = surf.meshes[ch.name]
        quads = np.asarray(m["quads"], float)
        uv = np.asarray(m["uv"], float)
        cells = np.arange(ch.first, ch.first + ch.count)
        nrm = surf.nrm[cells]
        off = 0.0003 if ch.body >= 0 else 0.0008
        v = (quads + nrm[:, None, :]*off).reshape(-1, 3)
        f = (np.arange(len(quads))[:, None]*4 + np.array([0, 1, 2, 0, 2, 3])).ravel()
        uvv = uv.reshape(-1, 2).copy()
        if ch.kind == "rect":
            uvv = uvv
        else:
            uvv[:, 1] = uvv[:, 1]                              # ring index from the bottom of the image (tex rows flipped)
        parent = f"/Lab/body_{ch.body}" if ch.body >= 0 else "/Room"
        S.mesh(f"{parent}/water_{ch.name}", v, f, uv=uvv, normals=np.repeat(nrm, 4, axis=0), mat=_overlay_mat(S, ch.name))
        H, Wd = ch.tex.shape
        meta["charts"][ch.name] = dict(diffuse=f"ov_{ch.name}", rough=f"rg_{ch.name}", w=int(Wd), h=int(H))
    # airborne water: linear curves (a segment per packet: its path over the last tick)
    g = S.UsdGeom.BasisCurves.Define(S.stage, "/Air/water")
    g.CreateTypeAttr("linear")
    g.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(np.array([[0, 0, -50], [0, 0, -50.01]], np.float32)))
    g.CreateCurveVertexCountsAttr([2])
    g.CreateWidthsAttr(S.Vt.FloatArray.FromNumpy(np.full(2, 0.001, np.float32)))
    g.SetWidthsInterpolation("vertex")
    S.bind(g, mats["water"])
    meta["curves"]["water"] = "/Air/water"
    # pools: closed meshes with fixed topology, points rewritten per request
    for k, p in enumerate(ws.pools.pools):
        path = (f"/Lab/body_{p.body}/pool_{k}" if p.kind in ("cavity", "sponge") else f"/Room/pool_{p.name}")
        v, f = pool_topology(p)
        mesh = S.UsdGeom.Mesh.Define(S.stage, path)
        mesh.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(np.zeros((len(v), 3), np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*(len(f)//3))
        mesh.CreateFaceVertexIndicesAttr(S.Vt.IntArray.FromNumpy(np.asarray(f, np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        S.bind(mesh, mats["foam"] if p.kind == "sponge" else mats["wash_water"] if p.name == "wash" else mats["water"])
        meta["pools"][p.name] = path
        if p.kind == "static":
            fp = f"/Room/foam_{p.name}"
            fm = S.UsdGeom.Mesh.Define(S.stage, fp)
            fm.CreatePointsAttr(S.Vt.Vec3fArray.FromNumpy(np.zeros((FNX*FNY, 3), np.float32)))
            fm.CreateFaceVertexCountsAttr([4]*(len(FOAM_FACES)//4))
            fm.CreateFaceVertexIndicesAttr(S.Vt.IntArray.FromNumpy(np.asarray(FOAM_FACES, np.int32)))
            fm.CreateSubdivisionSchemeAttr("catmullClark")
            S.bind(fm, mats["foam"])
            meta["foams"][p.name] = fp


BOX_FACES = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
FNX, FNY = 72, 96                                   # suds sheet on a bowl's water: 6 mm grid
FOAM_FACES = [k for i in range(FNX - 1) for j in range(FNY - 1)
              for k in (i*FNY + j, (i + 1)*FNY + j, (i + 1)*FNY + j + 1, i*FNY + j + 1)]        # quads (Catmull-Clark)


def _suds_field(seed=5):
    """A fixed, smooth random field (0..1) over the suds grid: where the bowl's foam gathers in clumps."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((FNX, FNY)), 6.0) + 0.25*gaussian_filter(rng.standard_normal((FNX, FNY)), 2.5)
    f = (f - f.min())/(f.max() - f.min())
    return f


SUDS = _suds_field()


def _suds_fine(seed=11):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(np.random.default_rng(seed).standard_normal((FNX, FNY)), 1.1)
    return (f - f.min())/(f.max() - f.min())


SUDS_FINE = _suds_fine()


def suds_points(p):
    """Suds sheet of a static pool from its foam liquid: thickness th = foam liquid / (1 - phi) / area spread as clumps;
    the covered fraction grows with th (a 20 mm mean layer covers ~80 %), uncovered vertices sit just under the water."""
    th = p.foam_thickness()
    if th <= 1e-4 or p.vol <= 0:
        return np.zeros((FNX*FNY, 3)) + (0, 0, -50)
    lo, hi = np.array(p.lo) + (0.004, 0.004, 0), np.array(p.hi) - (0.004, 0.004, 0)
    x = np.linspace(lo[0], hi[0], FNX); y = np.linspace(lo[1], hi[1], FNY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cover = min(0.95, 1.0 - math.exp(-th/0.012))
    thr = np.quantile(SUDS, 1.0 - cover)
    # rounded mounds: zero slope at their edge and on top (the first sheet rose as hgt^0.6, a cliff at every clump's
    # edge, and read as crumpled paper), 2 mm under the water where there is no foam
    u = np.clip((SUDS - thr)/(0.6*(1.0 - thr) + 1e-9), 0, 1)
    hgt = u*u*(3 - 2*u)
    z = p.level - 0.002 + (0.004 + 1.3*th)*hgt*(0.6 + 0.4*SUDS_FINE) + 0.007*(SUDS_FINE - 0.5)*np.minimum(1.0, 4*hgt)   # bubble clusters
    return np.stack([X, Y, z], -1).reshape(-1, 3)
NT, NZ = 48, 12


def pool_topology(p):
    """Vertices count and triangles of a pool's water volume: static bowls a box (8 corners); cavities NT x (NZ + 1)
    wall points plus a bottom and a top centre."""
    if p.kind in ("static", "sponge"):
        return np.zeros((8, 3)), BOX_FACES
    f = []
    for i in range(NZ):
        for j in range(NT):
            a, b = i*NT + j, i*NT + (j + 1) % NT
            c, d = a + NT, b + NT
            f += [a, c, d, a, d, b]
    bot, top = (NZ + 1)*NT, (NZ + 1)*NT + 1
    for j in range(NT):
        f += [bot, (j + 1) % NT, j]
        f += [top, NZ*NT + j, NZ*NT + (j + 1) % NT]
    return np.zeros(((NZ + 1)*NT + 2, 3)), f


def pool_points(p, body_q):
    """Current water volume of a pool: static bowls a box up to the level; a cavity its inner wall from the floor up to
    the free surface, per azimuth (the surface across a tilted cavity), in the cavity's body frame. Empty: collapsed."""
    if p.kind == "static":
        lo, hi = np.array(p.lo), np.array(p.hi)
        z = min(p.level, hi[2])
        if p.vol <= 1e-9 or z - lo[2] < 0.001:
            # (a drained bowl's last fraction of a millimetre drawn as a box is a zero-thickness dielectric, which the
            # path tracer shows black, so an emptying pre-rinse sink would turn black)
            return np.zeros((8, 3)) + (0, 0, -50)
        lo = lo + (0.001, 0.001, 0.0005); hi = np.array([hi[0] - 0.001, hi[1] - 0.001, z])
        return np.array([(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
                         (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])])
    if p.kind == "sponge":
        # lather on the sponge's top face: drawn while its pore water is soapy (above ~0.1 g/L), thicker when fuller
        c = p.sp[0]/p.vol if p.vol > 1e-9 else 0.0
        if c < 0.1e0*1.0 or p.vol <= 1e-9:
            return np.zeros((8, 3)) + (0, 0, -50)
        d = P.ITEMS["sponge"]
        hgt = 0.002 + 0.004*min(1.0, c/0.75)*min(1.0, p.vol/max(p.cap, 1e-9))
        x0, x1, y0, y1 = -0.45*d["L"], 0.45*d["L"], -0.42*d["W"], 0.42*d["W"]
        z0 = d["H"] - 0.0005
        return np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0 + hgt), (x1, y0, z0 + hgt),
                         (x1, y1, z0 + hgt), (x0, y1, z0 + hgt)])
    n_pts = (NZ + 1)*NT + 2
    if p.vol <= 1e-9:
        return np.zeros((n_pts, 3)) + (0, 0, -50)
    zs, rs = W._r_of_z(p.cav["polygon"])
    z0, z1 = float(zs.min()), float(zs.max())
    n, d = p.n, p.d
    th = 2*math.pi*np.arange(NT)/NT
    out = np.zeros((n_pts, 3))
    tops = []
    for j, t in enumerate(th):
        c, s = math.cos(t), math.sin(t)
        zz = np.linspace(z0, z1, 200)
        rr = np.interp(zz, zs, rs)*0.995
        f = n[0]*rr*c + n[1]*rr*s + n[2]*zz - d
        k = np.flatnonzero(f > 0)
        ztop = z1 if not len(k) else zz[max(k[0] - 1, 0)] if k[0] > 0 else z0
        if len(k) and k[0] > 0:
            a, b = k[0] - 1, k[0]
            ztop = zz[a] + (zz[b] - zz[a])*(0 - f[a])/(f[b] - f[a])
        for i, z in enumerate(np.linspace(z0 + 0.0005, max(ztop, z0 + 0.0006), NZ + 1)):
            r = float(np.interp(z, zs, rs))*0.995
            out[i*NT + j] = (r*c, r*s, z)
        tops.append(out[NZ*NT + j])
    out[(NZ + 1)*NT] = (0, 0, z0 + 0.0005)
    out[(NZ + 1)*NT + 1] = np.mean(tops, axis=0)
    return out


def _lights(S, mats, spec):
    UsdLux, Gf, Z = S.UsdLux, S.Gf, L.Z
    # 2 x 4 ft LED troffers over the stations and the aisle (the room's only light: an interior dish room)
    spots = [(3.3, 4.6), (1.2, 2.8), (1.2, 4.6), (6.6, 3.2), (6.2, 1.0), (3.6, 2.6), (5.0, 4.6)]
    for k, (x, y) in enumerate(spots):
        S.box(f"/Room/troffer_{k}", (0.61, 1.22, 0.04), mats["frame_white"], (x, y, Z - 0.02), (0, 0, 0, 1))
        S.box(f"/Room/diffuser_{k}", (0.57, 1.18, 0.005), S.color("diffuser", (.95, .95, .95), .3, 0., emissive=(4., 4., 3.9)),
              (x, y, Z - 0.043), (0, 0, 0, 1))
        lt = UsdLux.RectLight.Define(S.stage, f"/Room/light_{k}")
        lt.CreateWidthAttr(0.57); lt.CreateHeightAttr(1.18); lt.CreateIntensityAttr(2600.); lt.CreateColorAttr(Gf.Vec3f(1., .98, .95))
        xf = S.UsdGeom.Xformable(lt.GetPrim()); xf.AddTranslateOp().Set(Gf.Vec3d(x, y, Z - 0.05))
    dome = UsdLux.DomeLight.Define(S.stage, "/Room/ambient")
    dome.CreateIntensityAttr(60.); dome.CreateColorAttr(Gf.Vec3f(.9, .92, .95))


def _dressing(S, mats, spec):
    """Render-only dressing outside every station's working volume: chemical jugs, crates on the dunnage shelf, broom
    and dustpan in the can-wash corner, a trash bag by the back door, a utility cart, ceiling light fittings."""
    from .visual_assets import model
    from pxr import Usd

    def ref(name, path, p, yaw=0.0, scale=1.0):
        import re
        usd = model(name)
        outer = S.UsdGeom.Xform.Define(S.stage, path)
        outer.AddTranslateOp().Set(S.Gf.Vec3d(*map(float, p)))
        if yaw:
            outer.AddRotateZOp().Set(float(yaw))
        if scale != 1.0:
            outer.AddScaleOp().Set(S.Gf.Vec3f(scale, scale, scale))
        inner = S.UsdGeom.Xform.Define(S.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{name}")
        for pr in Usd.PrimRange(inner.GetPrim()):
            if pr.GetTypeName() == "BasisCurves":
                pr.SetActive(False)
        UsdShade, Tn, Sdf = S.UsdShade, S.Sdf.ValueTypeNames, S.Sdf
        made = {}
        for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)):
            if not pr.IsA(S.UsdGeom.Gprim):
                continue
            src, _ = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()
            if not src:
                continue
            key = src.GetPrim().GetName()
            if key not in made:
                files = {}
                for sh in Usd.PrimRange(src.GetPrim()):
                    if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetIdAttr().Get() == "UsdUVTexture":
                        f = UsdShade.Shader(sh).GetInput("file").Get()
                        if f:
                            files[Path(f.path).name] = usd.parent/"textures"/Path(f.path).name
                pick = lambda tag: next((v for k, v in files.items() if re.search(tag, k)), None)
                m = UsdShade.Material.Define(S.stage, f"/Looks/pbr_{name}_{key}")
                sh = UsdShade.Shader.Define(S.stage, f"/Looks/pbr_{name}_{key}/Shader")
                sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
                for tag, inp in (("_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"), ("_nor_gl", "normalmap_texture")):
                    f = pick(tag)
                    if f is not None and f.exists() and f.suffix.lower() in (".jpg", ".png"):
                        sh.CreateInput(inp, Tn.Asset).Set(Sdf.AssetPath(str(f.resolve())))
                        if inp == "reflectionroughness_texture":
                            sh.CreateInput("reflection_roughness_texture_influence", Tn.Float).Set(1.0)
                m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
                made[key] = m
            UsdShade.MaterialBindingAPI.Apply(pr).Bind(made[key])
        for sp in [pr.GetPath() for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)) if pr.GetName() == "_materials"]:
            q = S.stage.GetPrimAtPath(sp)
            if q:
                q.SetActive(False)
        return outer

    sh = L.SHELF
    if spec.get("task") != "scrub-rinse-pan":
        # the wall shelf is that task's destination: an accepted pan (centre over the clean board, handle any way) can
        # reach the shelf's far end, and these drawn-only jugs have no colliders
        ref("bleach_bottle", "/Dress/sanitizer_jug", (L.DRAINBOARDS["clean"][1] + 0.12, L.Y - 0.15, 1.52), yaw=-20.0)
        ref("multi_cleaner_bottle", "/Dress/spray_cleaner", (L.DRAINBOARDS["clean"][1] + 0.25, L.Y - 0.12, 1.52), yaw=15.0)
    ref("plastic_crate_01", "/Dress/crate_0", (0.5*(sh["x0"] + sh["x1"]), 0.5*(sh["y0"] + sh["y1"]), sh["levels"][1]), yaw=0.0)
    ref("plastic_crate_02", "/Dress/crate_1", (0.5*(sh["x0"] + sh["x1"]), 0.5*(sh["y0"] + sh["y1"]), sh["levels"][0]), yaw=90.0)
    ref("plastic_broom", "/Dress/broom", (L.X - 0.08, 1.95, 0.0), yaw=180.0)
    ref("dustpan", "/Dress/dustpan", (L.X - 0.20, 1.92, 0.0), yaw=160.0)
    ref("trashbag", "/Dress/trashbag", (2.55, 0.30, 0.0), yaw=30.0)
    ref("plastic_container", "/Dress/container", (0.35, L.MACHINE["y1"] + 0.30, L.TABLE_Z), yaw=10.0)
    ref("industrial_storage_cart", "/Dress/cart", (1.6, 0.45, 0.0), yaw=0.0)


# diagnostic views corrected here (render only): the can bay's "side" eye (scene.views) sits inside the dunnage shelf
VIEW_FIX = {("can-wash", "side"): ([6.95, 2.55, 1.55], [6.2, 1.0, 0.4])}


def _cameras(S, world, spec):
    from classroom_arrangement import params as CP
    out = {}
    sc = world.scene
    rig, inst = world.rig, sc.robot_instance
    fr = rig.frame_pose(f"{inst}/head_camera", world.state)
    carriage = sc.robot_body["franka_spine_mounting_point"]
    pb, Rb = world.body_pose(carriage)
    qw = fr["orientation_wxyz"]
    Rf = Rotation.from_quat((qw[1], qw[2], qw[3], qw[0]))
    pf = np.asarray(fr["position_m"])
    Rl = Rotation.from_matrix(Rb).inv()*Rf
    pl = Rotation.from_matrix(Rb).inv().apply(pf - pb)
    g = S.camera(f"/Lab/body_{carriage}/head_rgb", pl, pl + Rl.apply((1, 0, 0)), 18/math.tan(math.radians(CP.HEAD_HFOV/2)),
                 up=Rl.apply((0, 0, 1)), role="policy")
    S.alias("head_rgb", g.GetPath()); out["head_rgb"] = "policy"
    for name, m in sc.cameras.items():
        role = m.get("role", "policy")
        fix = VIEW_FIX.get((spec.get("station"), name))
        if fix:
            m = dict(m, eye=fix[0], target=fix[1])
        if m["body"] >= 0:
            g = S.camera(f"/Lab/body_{m['body']}/{name}", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)), up=m["up"],
                         role=role)
        else:
            g = S.camera(f"/Cameras/{name}_cam", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)), up=m["up"],
                         role=role)
        S.alias(name, g.GetPath()); out[name] = role
    base = sc.robot_body["base_link"]
    g = S.camera(f"/Lab/body_{base}/chase", (-1.5, 0.7, 1.9), (1.0, 0.0, 0.9), 14., role="diagnostic")
    S.alias("chase", g.GetPath()); out["chase"] = "diagnostic"
    return out


# ------------------------------------------------------------------------------ dynamic state
SOIL_RGB = np.array([0.105, 0.011, 0.006])         # dried tomato paste, dark brick red (linear)
DUST_RGB = np.array([0.16, 0.11, 0.065])           # silt, debris film (linear)
EGG_RGB = np.array([0.50, 0.33, 0.07])             # dried egg on the pan: yellow-brown (linear; the paste's red was wrong)
WATER_RGB = np.array([0.80, 0.84, 0.88])           # what a wet sheen reflects in this room
FOAM_RGB = np.array([0.90, 0.90, 0.89])


def overlay_images(ws, fields):
    """{chart: (rgba uint8 H x W x 4, rough uint8 H x W x 4)} from the cell fields."""
    h, s0, sus, foam, soil, tr = (fields[k] for k in ("h", "surf", "sus", "foam", "soil", "soil_tr"))
    phi = P.FOAM["phi"]
    # paste hides the plate from ~0.1 mm (3e-4 kg/m2 e-fold); dust and debris are a haze (1 g/m2 is a ~1 um layer)
    a_soil = np.where(tr > 0.5, 0.75*(1.0 - np.exp(-soil/1.5e-3)), 1.0 - np.exp(-soil/3.0e-4))
    a_foam = 1.0 - np.exp(-(foam/(1 - phi))/1.2e-3)
    a_water = np.clip(h/1.5e-3, 0, 1)*0.30
    c_sus = np.where(h > 1e-6, sus/np.maximum(h, 1e-9), 0.0)            # kg/m3 of suspended solids
    dirty = 1.0 - np.exp(-c_sus/2.0)
    srgb = np.where(tr[:, None] > 0.5, DUST_RGB[None], SOIL_RGB[None])
    egg = ws.item_cells.get("pan") if hasattr(ws, "item_cells") else None
    if egg is not None and len(egg):
        srgb = srgb.copy()
        srgb[egg] = EGG_RGB
    wrgb = WATER_RGB[None]*(1 - dirty[:, None]) + srgb*dirty[:, None]
    a_w = np.maximum(a_water, dirty*np.clip(h/2e-4, 0, 1)*0.8)
    a = 1.0 - (1 - a_soil)*(1 - a_foam)*(1 - a_w)
    num = srgb*a_soil[:, None] + FOAM_RGB[None]*a_foam[:, None]*(1 - a_soil[:, None]) + \
        wrgb*a_w[:, None]*(1 - a_soil[:, None])*(1 - a_foam[:, None])
    rgb = num/np.maximum(a[:, None], 1e-6)
    # written linear: the renderer reads these dynamic textures as raw data (gamma-encoded, the paste came out salmon)
    rgb8 = (np.clip(rgb, 0, 1)*255 + 0.5).astype(np.uint8)
    a8 = (np.clip(a, 0, 1)*255 + 0.5).astype(np.uint8)
    rough = np.where(h > 2e-5, 0.04, 0.55)
    rough = np.where(a_foam > 0.5, 0.7, rough)
    r8 = (rough*255).astype(np.uint8)
    out = {}
    for ch in ws.surf.charts:
        idx = ch.tex
        img = np.empty(idx.shape + (4,), np.uint8)
        img[..., :3] = rgb8[idx]; img[..., 3] = a8[idx]
        rg = np.empty(idx.shape + (4,), np.uint8)
        rg[..., :3] = r8[idx][..., None]; rg[..., 3] = 255
        out[ch.name] = (img, rg)
    return out


def cell_fields(ws):
    E = ws.eng
    sp = E.sp.numpy()
    return dict(h=E.h.numpy(), surf=sp[0], sus=sp[1], foam=sp[4], soil=E.soil.numpy(), soil_tr=E.soil_tr.numpy())


def packet_segments(ws, max_n=60000):
    E = ws.eng
    alive = E.alive.numpy().astype(bool)
    if not alive.any():
        return np.array([[0, 0, -50], [0, 0, -50.01]], np.float32), np.full(2, 0.001, np.float32)
    x = E.px.numpy()[alive]; v = E.pv.numpy()[alive]; d = E.pd.numpy()[alive]
    if len(x) > max_n:
        k = np.linspace(0, len(x) - 1, max_n).astype(int)
        x, v, d = x[k], v[k], d[k]
    a = x - v*P.DT*1.0
    pts = np.stack([a, x], 1).reshape(-1, 3).astype(np.float32)
    wid = np.repeat(np.maximum(d, 0.0008), 2).astype(np.float32)
    return pts, wid


def dynamic_payload(env, cache, full=False, fields=None, packets=None, pools=None):
    """Per-request dynamic state from the live environment (or from a recording when fields/packets/pools are given)."""
    ws = env.water
    out = dict(textures={}, curves={}, pools={}, foams={})
    if ws is None:
        return out
    fields = fields if fields is not None else cell_fields(ws)
    imgs = overlay_images(ws, fields)
    for name, (img, rg) in imgs.items():
        key = hash(img.tobytes())
        if full or cache.get(name) != key:
            cache[name] = key
            out["textures"][f"ov_{name}"] = dict(w=img.shape[1], h=img.shape[0], data=base64.b64encode(img.tobytes()).decode())
            out["textures"][f"rg_{name}"] = dict(w=rg.shape[1], h=rg.shape[0], data=base64.b64encode(rg.tobytes()).decode())
    pts, wid = packets if packets is not None else packet_segments(ws)
    out["curves"]["water"] = dict(points=base64.b64encode(np.ascontiguousarray(pts, np.float32).tobytes()).decode(),
                                  widths=base64.b64encode(np.ascontiguousarray(wid, np.float32).tobytes()).decode(),
                                  n=int(len(pts)//2))
    bq = env.world.state.body_q.numpy()
    pl = pools if pools is not None else {p.name: pool_points(p, bq) for p in ws.pools.pools}
    for name, pts_ in pl.items():
        out["pools"][name] = base64.b64encode(np.ascontiguousarray(pts_, np.float32).tobytes()).decode()
    for p in ws.pools.pools:
        if p.kind != "static":
            continue
        out["foams"][p.name] = base64.b64encode(np.ascontiguousarray(suds_points(p), np.float32).tobytes()).decode()
    return out


# ------------------------------------------------------------------------------ worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (surface_rinsing/render_worker.py)."""

    def __init__(self, folder, scene, meta, *, width=640, height=360, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.meta = meta
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        self.cache = {}
        self.first = True
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PNT_RENDER_PORT=str(listener.getsockname()[1]), PNT_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "surface_rinsing.render_worker", "--scene",
                                             str(Path(scene).resolve()), "--width", str(width), "--height", str(height),
                                             "--samples", str(samples)],
                                            stdout=self.log, stderr=subprocess.STDOUT, env=worker_env(env), cwd=str(root))
            self.connection, _ = listener.accept()
            self.connection.settimeout(600)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed; see " + str(self.folder/"worker.log"))
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    def _call(self, message):
        import select
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            send(self.connection, message)
            deadline = time.monotonic() + 1800
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            reply = receive(self.connection)
        if not reply.get("ok") or reply.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(reply)[:400])
        return reply

    def observe(self, env, cameras, resolution=None, samples=None, body_q=None, dynamic=None, tick=None):
        """Render the named cameras from the environment's current state (or a recorded frame); {camera: PNG bytes}."""
        dyn = dynamic if dynamic is not None else dynamic_payload(env, self.cache, full=self.first)
        self.first = False
        bq = body_q if body_q is not None else env.world.state.body_q.numpy()
        msg = dict(op="observe", cameras=list(cameras), body_poses=np.round(bq, 7).tolist(),
                   tick=int(tick if tick is not None else env.world.tick), dynamic=dyn)
        if resolution is not None:
            msg["resolution"] = list(resolution)
        if samples is not None:
            msg["spp"] = int(samples)
        reply = self._call(msg)
        self.last_timings = reply.get("timings", {})
        return {k: base64.b64decode(v) for k, v in reply["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

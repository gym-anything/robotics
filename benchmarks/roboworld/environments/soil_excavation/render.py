"""USD export of the front yard being dug and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance: the broom-finished walkway, lawn and mulched bed at grade around the soil block, the
house front, the robot (franka_description visual meshes, classroom/painting drawing), the tools as the parts they are
built from, props and pinned CC0 dressing (visual_assets.py); a Poly Haven residential-garden HDRI lights it. Each
request carries the body poses and the soil surface reconstructed from the particles (surface.py), written to an .npz
next to the stage and re-authored on one mesh: nothing about the soil is painted on by the renderer.
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation

from classroom_arrangement.render import Stage
from . import params as P
from . import layout as L
from . import surface as SF

COLORS = dict(
    robot=((.88, .88, .87), .35, 0.), robot_dark=((.07, .075, .08), .45, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    tmr_black=((.05, .05, .055), .5, 0.), tmr_deck=((.80, .81, .80), .4, 0.), aluminum=((.70, .72, .74), .3, .85),
    stainless=((.80, .80, .81), .18, 1.), ash=((.62, .46, .30), .55, 0.), steel_red=((.55, .06, .05), .45, .3),
    steel_black=((.06, .06, .065), .45, .6), aluminium=((.72, .73, .74), .35, .9), rubber_green=((.10, .40, .12), .7, 0.),
    rubber_black=((.04, .04, .045), .75, 0.), fiberglass_yellow=((.85, .62, .05), .45, 0.), dewalt_yellow=((.95, .70, .02), .45, 0.),
    dewalt_black=((.05, .05, .05), .5, 0.), spf=((.80, .68, .48), .7, 0.), cedar=((.55, .33, .20), .7, 0.),
    rootball=((.10, .07, .045), .9, 0.), bark=((.30, .20, .12), .85, 0.), foliage=((.12, .30, .08), .7, 0.),
    stone=((.48, .45, .42), .8, 0.), tarp=((.05, .22, .62), .55, 0.), pvc=((.86, .86, .84), .35, 0.),
    path_light=((.20, .17, .12), .4, .7), stake=((.75, .62, .42), .8, 0.), valve_box=((.05, .25, .10), .6, 0.),
    flag_purple=((.40, .12, .55), .6, 0.), flag_wire=((.6, .6, .6), .4, .9), glass=((.05, .06, .07), .05, 0.),
    trim=((.93, .93, .90), .4, 0.), downspout=((.90, .90, .88), .4, 0.), pail=((.85, .85, .83), .5, 0.),
    door=((.16, .22, .30), .35, 0.), fence=((.58, .45, .33), .8, 0.), shingle=((.20, .20, .22), .85, 0.),
    edging=((.06, .06, .06), .5, .6), grommet=((.72, .58, .30), .3, .9),
)


GRASS_TINT = (0.62, 0.86, 0.50)
MULCH_TINT = (0.50, 0.42, 0.36)          # weathered shredded hardwood bark: dark brown, not dyed red


def _rot_to_wxyz(R):
    x, y, z, w = R.as_quat()
    return float(w), float(x), float(y), float(z)


# ------------------------------------------------------------------------------ export
def export(env, folder, extra_cameras=None):
    """Export the episode's static appearance to folder/scene.usdc; returns (path, meta)."""
    world, sc = env.world, env.world.scene
    S = Stage(folder)
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    _textured(S, mats)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    from painting.render import _robot
    _robot(S, world, mats)
    _yard(S, mats, env.spec)
    _braces(S, mats, world)
    _tools(S, world, mats)
    meta = dict(soil="/Lab/soil")
    _soil_prim(S, mats)
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


def _braces(S, mats, world):
    """Set-post's temporary braces, drawn along the spring-dampers that hold the post (world.set_braces): a 1x2 board
    from its nail point on the post face to a 2x2 stake driven into the lawn beside its other end. Render only -- the
    arms and tools pass through them; the braces' physics is the two spring-dampers alone."""
    if not getattr(world, "braces", None):
        return
    p, R = world.body_pose(world.scene.bodies["post"])
    for k, br in enumerate(world.braces):
        top = p + R @ np.asarray(br["local"], float)
        foot = np.asarray(br["anchor"], float)
        u = top - foot
        length = float(np.linalg.norm(u))
        u /= length
        side = np.cross((0., 0., 1.), u)
        side /= np.linalg.norm(side)
        q = Rotation.from_matrix(np.stack([u, side, np.cross(u, side)], 1)).as_quat()
        # 19 mm thick across the brace's vertical plane, 38 mm wide in it; 2 cm past each nail
        S.box(f"/Yard/brace_{k}", (length + 0.04, 0.019, 0.038), mats["spf"], 0.5*(top + foot), q, bevel=0.002)
        stake = foot + side*(0.019 + 0.038)/2
        S.box(f"/Yard/brace_stake_{k}", (0.038, 0.038, 0.40), mats["stake"], (stake[0], stake[1], foot[2] + 0.05 - 0.20),
              Rotation.from_euler("z", math.atan2(u[1], u[0])).as_quat(), bevel=0.003)


def _textured(S, mats):
    from .visual_assets import texture
    c = texture("brushed_concrete")
    # broom-finished walkway concrete is a light grey (albedo ~0.35-0.45); the CC0 brushed-concrete albedo alone renders
    # it nearly asphalt-dark
    from painting.render import _pbr_const
    mats["concrete"] = _pbr_const(S, "concrete_broom", SF.srgb_to_linear((174, 177, 174)), rough=.85,
                                  normal=c["nor_gl"], normal_strength=.7)
    g = texture("leafy_grass")
    # the CC0 leafy-grass albedo reads as straw under a full sun: tinted toward a watered lawn's green (render only)
    mats["lawn"] = S.textured("lawn_tex", g["diff"], rough=.9, normal=g["nor_gl"], tint=GRASS_TINT)
    m = texture("wood_chip_path")
    # the CC0 wood-chip albedo is a fresh orange cedar chip; garden-centre shredded bark mulch weathers to a dark brown
    mats["mulch_ground"] = S.textured("mulch_tex", m["diff"], rough=.9, normal=m["nor_gl"], tint=MULCH_TINT)
    sd = texture("weathered_plank_siding")
    from painting.render import _pbr_const
    mats["siding"] = _pbr_const(S, "siding_paint", SF.srgb_to_linear((196, 204, 206)), rough=.55, normal=sd["nor_gl"],
                                normal_strength=0.8)
    # soil material: the custom MDL next to the stage (soil_surfaces.mdl)
    import shutil
    shutil.copy2(Path(__file__).with_name("soil_surfaces.mdl"), S.folder/"soil_surfaces.mdl")
    s = texture("brown_mud_02")
    mats["rootball"] = S.textured("rootball_soil", s["diff"], rough=.96, normal=s["nor_gl"],
                                  tint=(.8, .7, .55))
    mats["fine_roots"] = S.color("fine_roots", (.42, .28, .14), .95, 0.)
    Sdf, UsdShade, T_ = S.Sdf, S.UsdShade, S.Sdf.ValueTypeNames
    mm = UsdShade.Material.Define(S.stage, "/Looks/soil")
    sh = UsdShade.Shader.Define(S.stage, "/Looks/soil/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath(str((S.folder/"soil_surfaces.mdl").resolve())), "mdl")
    sh.SetSourceAssetSubIdentifier("soil", "mdl")
    sh.CreateInput("grass_tint", T_.Color3f).Set(S.Gf.Vec3f(*GRASS_TINT))
    sh.CreateInput("mulch_tint", T_.Color3f).Set(S.Gf.Vec3f(*MULCH_TINT))
    for k, f in (("soil_normal", s["nor_gl"]), ("soil_diffuse", s["diff"]), ("grass_diffuse", g["diff"]),
                 ("grass_normal", g["nor_gl"]), ("mulch_diffuse", m["diff"]), ("mulch_normal", m["nor_gl"])):
        sh.CreateInput(k, T_.Asset).Set(Sdf.AssetPath(str(Path(f).resolve())))
    mm.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mats["soil"] = mm


def _quad_xy(S, path, x0, y0, x1, y1, z, mat, tile=1.0):
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return S.mesh(path, pts, [0, 1, 2, 0, 2, 3], uv=[(p[0]/tile, p[1]/tile) for p in pts], normals=[(0, 0, 1)]*4, mat=mat)


def _ground_quad(S, mats, path, x0, y0, x1, y1, z, kind):
    """Lawn or mulch ground drawn with the soil surface's own material (the same textures, tiling, tint and shading as
    the turf / mulch particles at the block's top): with separate OmniPBR materials the soil block reads as a darker
    rectangle in the bed."""
    from pxr import UsdGeom, Sdf, Vt
    import numpy as _np
    n = 2
    xs, ys = _np.linspace(x0, x1, n), _np.linspace(y0, y1, n)
    X, Y = _np.meshgrid(xs, ys, indexing="ij")
    v = _np.stack([X, Y, _np.full_like(X, z)], -1).reshape(-1, 3)
    f = [0, 2, 3, 0, 3, 1]
    g = S.mesh(path, v.tolist(), f, normals=[(0, 0, 1)]*4, mat=mats["soil"])
    pv = UsdGeom.PrimvarsAPI(g)
    one = _np.ones(4, _np.float32)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(_np.tile(_np.array([[0.09, 0.06, 0.04]], _np.float32), (4, 1))))
    pv.CreatePrimvar("turf", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(one if kind == "lawn" else 0*one))
    pv.CreatePrimvar("mulch", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(one if kind == "mulch" else 0*one))
    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(v[:, :2].astype(_np.float32)))
    return g


def _path_corner(S, mats, spec):
    """The hardpan preset's setting (render only, flush with grade like the lawn it lies on): a worn dirt path from the
    walkway south across the lawn that turns west at the compacted corner the soil block is, with ragged edges where
    the grass thins into it, not a ruler-edged 1 m square of dirt in the lawn. Drawn with the soil's own material and the hardpan crust's colour, so the block's
    surface continues into it."""
    from pxr import UsdGeom, Sdf, Vt
    b = spec["block"]
    bx0, by0, bx1, by1 = b["lo"][0], b["lo"][1], b["hi"][0], b["hi"][1]
    rng = np.random.default_rng(int(spec.get("seed", 0)) + 41)
    ph = rng.uniform(0, 2*math.pi, 8)

    def wobble(u, k):
        return 0.045*math.sin(7.1*u + ph[k]) + 0.03*math.sin(15.3*u + ph[k + 1]) + 0.015*math.sin(31.0*u + ph[k + 2])
    # Resolve the same 2.5 mm cracks as the 5 mm particle-surface mesh. A 25 mm
    # context grid interpolated them into large dark triangles at the seam.
    cell = 0.005
    xs = np.arange(L.YARD["x"][0], bx1 + 0.25, cell)
    ys = np.arange(by0 - 0.30, L.SLAB["y"][0] + 1e-6, cell)
    X, Y = np.meshgrid(xs + cell/2, ys + cell/2, indexing="ij")
    vx, vy = np.vectorize(lambda y: wobble(y, 0)), np.vectorize(lambda x: wobble(x, 3))
    half = 0.5*(bx1 - bx0) + 0.14
    legA = (np.abs(X - 0.5*(bx0 + bx1)) < half + vx(Y)) & (Y > by0 - 0.16 + vy(X))
    legB = (X < 0.5*(bx0 + bx1)) & (Y > by0 - 0.16 + vy(X)) & (Y < by0 + 0.62 + np.vectorize(lambda x: wobble(x, 5))(X))
    # The render-only overlap stays within the original 6 cm perimeter band, with a ragged
    # inner edge. A rectangular cutout plus an uncracked path exposed the solver block in stills.
    edge_x = 0.015*np.sin(11*Y) + 0.01*np.sin(23*Y)
    edge_y = 0.015*np.sin(13*X) + 0.01*np.sin(27*X)
    inner = ((X > bx0 + 0.035 + edge_x) & (X < bx1 - 0.035 + edge_x)
             & (Y > by0 + 0.035 + edge_y) & (Y < by1 - 0.035 + edge_y))
    keep = (legA | legB) & ~inner
    # distance to the path's edge (cells), for the grass thinning into it
    from scipy import ndimage
    # The cutout is filled by physical soil, not lawn: grass belongs at the outer path edge.
    dist = ndimage.distance_transform_edt(legA | legB)*cell
    nx, ny = X.shape
    vid = -np.ones((nx + 1, ny + 1), int)
    verts, faces, cols, turf = [], [], [], []
    base = SF.srgb_to_linear(P.MATERIALS["hardpan"]["color"])*1.12*dict(dry=1.15, moist=1.0, wet=0.85)[spec.get("moisture", "moist")]

    def v(i, j):
        if vid[i, j] < 0:
            vid[i, j] = len(verts)
            x, y = xs[0] + i*cell, ys[0] + j*cell
            verts.append((x, y, L.GRADE + 0.0006))
            dd = dist[min(i, nx - 1), min(j, ny - 1)]
            cols.append(base*(0.92 + 0.12*math.sin(9.0*x + 4.0*y)*math.sin(5.0*y - 3.0*x)))
            turf.append(float(np.clip(1.0 - dd/0.08, 0.0, 0.7)))
        return vid[i, j]
    for i, j in zip(*np.nonzero(keep)):
        a, b_, c, d = v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)
        faces += [a, b_, c, a, c, d]
    if not faces:
        return
    vv = np.asarray(verts, np.float32)
    # Match the particle surface's world-coordinate crust pattern across the boundary.
    # This is appearance only; it does not fracture soil.
    colors = np.asarray(cols, np.float32)*SF.crack_factor(vv[:, :2])[:, None]
    g = S.mesh("/Yard/path", vv.tolist(), faces, normals=[(0, 0, 1)]*len(vv), mat=mats["soil"])
    pv = UsdGeom.PrimvarsAPI(g)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    pv.CreatePrimvar("turf", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.asarray(turf, np.float32)))
    pv.CreatePrimvar("mulch", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.zeros(len(vv), np.float32)))
    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(vv[:, :2].copy()))
    g.GetPrim().CreateAttribute("primvars:doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)


def _yard(S, mats, spec):
    preset = spec["preset"]
    for g in L.ground_boxes(preset):
        lo, hi = g["lo"], g["hi"]
        if g["name"] in ("base_course", "block_floor"):
            continue
        if g["name"] == "slab":
            S.box("/Yard/slab", (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), mats["concrete"],
                  ((lo[0] + hi[0])/2, (lo[1] + hi[1])/2, (lo[2] + hi[2])/2), bevel=0.006, uv_per_m=1/1.5)
        elif g["name"] == "foundation":
            S.box("/Yard/foundation", (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), mats["concrete"],
                  ((lo[0] + hi[0])/2, (lo[1] + hi[1])/2, (lo[2] + hi[2])/2), bevel=0.004, uv_per_m=1/1.5)
        else:
            _ground_quad(S, mats, f"/Yard/{g['name']}", lo[0], lo[1], hi[0], hi[1], hi[2], "mulch" if g["surface"] == "bed" else "lawn")
    # the soil surface rounds off along the block's edge (no particles beyond it): a 25 mm strip of the surrounding
    # ground's material over the block perimeter hides that groove (render only)
    b = spec["block"]
    edge_mat = mats["mulch_ground"] if preset in ("bed", "irrigation", "bulbs") else mats["lawn"]
    x0b, y0b, x1b, y1b = b["lo"][0], b["lo"][1], b["hi"][0], b["hi"][1]
    wdt = 0.06
    if preset == "hardpan":
        _path_corner(S, mats, spec)                     # the path covers the block's perimeter itself
    for k, (a0, a1, c0, c1) in enumerate(((x0b, x1b, y0b, y0b + wdt), (x0b, x1b, y1b - wdt, y1b), (x0b, x0b + wdt, y0b, y1b),
                                          (x1b - wdt, x1b, y0b, y1b))):
        if preset in ("bed", "irrigation", "bulbs") and k == 0 and abs(y0b) < 1e-6:
            continue                                    # that side is the slab's edge
        if preset == "hardpan":
            continue
        g = _ground_quad(S, mats, f"/Yard/block_edge_{k}", a0, c0, a1, c1, L.GRADE + 0.0008,
                         "mulch" if preset in ("bed", "irrigation", "bulbs") else "lawn")
        # a raised strip's edge drew a thin black shadow line across the bed under the low sun (D9 video frames)
        g.GetPrim().CreateAttribute("primvars:doNotCastShadows", S.Sdf.ValueTypeNames.Bool).Set(True)
    # house front: siding above the foundation, a window and a downspout
    y = L.HOUSE_Y
    x0, x1 = L.YARD["x"]
    S.mesh("/Yard/siding", [(x0, y, 0.30), (x1, y, 0.30), (x1, y, 2.9), (x0, y, 2.9)], [0, 2, 1, 0, 3, 2],
           uv=[(x0/2, .15), (x1/2, .15), (x1/2, 1.45), (x0/2, 1.45)], normals=[(0, -1, 0)]*4, mat=mats["siding"])
    S.box("/Yard/window_glass", (1.2, 0.02, 1.1), mats["glass"], (-1.6, y - 0.01, 1.55), bevel=0.002)
    S.box("/Yard/window_trim", (1.32, 0.03, 1.22), mats["trim"], (-1.6, y - 0.002, 1.55), bevel=0.004)
    S.box("/Yard/sill", (1.40, 0.08, 0.04), mats["trim"], (-1.6, y - 0.04, 0.95), bevel=0.004)
    S.cylinder("/Yard/downspout", 0.035, 2.8, mats["downspout"], (2.9, y - 0.06, 1.4), n=16)
    _surroundings(S, mats, spec)
    # tarp
    (tx0, ty0), (tx1, ty1) = spec["tarp"]
    z = 0.0012 if ty0 >= L.SLAB["y"][0] and ty1 <= 0.0 else L.GRADE + 0.0012
    _tarp(S, mats, tx0, ty0, tx1, ty1, z)
    # props
    for name, pr in spec["props"].items():
        x, yy = pr["pos"]
        if pr["kind"] == "path_light":
            S.cylinder(f"/Yard/{name}_post", 0.025, 0.42, mats["path_light"], (x, yy - 0.10, L.GRADE + 0.21), n=16)
            S.cylinder(f"/Yard/{name}_cap", 0.09, 0.03, mats["path_light"], (x, yy - 0.10, L.GRADE + 0.43), n=24)
        elif pr["kind"] == "stake":
            S.box(f"/Yard/{name}", (0.04, 0.03, 0.55), mats["stake"], (x, yy - 0.285, L.GRADE + 0.02), bevel=0.002)
        elif pr["kind"] == "valve_box":
            S.cylinder(f"/Yard/{name}_lid", 0.13, 0.02, mats["valve_box"], (x, yy, L.GRADE + 0.005), n=32)
        elif pr["kind"] == "pail":
            Pa = P.PAIL
            n = 40
            ang = np.linspace(0, 2*np.pi, n, endpoint=False)
            v, f = [], []
            for k, a in enumerate(ang):
                for rr, z in ((Pa["r"] + Pa["wall"], 0.0), (Pa["r"] + Pa["wall"], Pa["height"]), (Pa["r"], Pa["height"]), (Pa["r"], 0.006)):
                    v.append((x + rr*math.cos(a), yy + rr*math.sin(a), z))
            for k in range(n):
                a, b2 = 4*k, 4*((k + 1) % n)
                for j in range(3):
                    f += [a + j, b2 + j, b2 + j + 1, a + j, b2 + j + 1, a + j + 1]
            S.mesh(f"/Yard/{name}", v, f, mat=mats["pail"], smooth=True)
            S.cylinder(f"/Yard/{name}_bottom", Pa["r"], 0.006, mats["pail"], (x, yy, 0.003), n=40)
    for k, m in enumerate(spec["marks"]):
        if m["kind"] == "flag":
            x, yy = m["pos"]
            S.cylinder(f"/Yard/flag_wire_{k}", 0.0015, 0.40, mats["flag_wire"], (x, yy, L.GRADE + 0.18), n=6)
            S.box(f"/Yard/flag_{k}", (0.10, 0.002, 0.07), mats["flag_purple"], (x + 0.05, yy, L.GRADE + 0.34), bevel=0.0005)
    if "pipe" in spec:
        pp = spec["pipe"]
        q = Rotation.from_euler("y", 90, degrees=True).as_quat()
        b = spec["block"]
        S.cylinder("/Yard/pipe", pp["r"], b["hi"][0] - b["lo"][0] + 0.04, mats["pvc"], ((b["lo"][0] + b["hi"][0])/2, pp["y"], pp["z"]),
                   q=q, n=32)


def _tarp(S, mats, x0, y0, x1, y1, z):
    """The poly tarp as a crinkled sheet (render only; the soil and MuJoCo see a flat sheet): smooth +-2.5 mm creases
    along the factory folds and random crinkle, a doubled hem and four brass grommets (a flat box read as plastic film)."""
    nx, ny = 92, 122
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rng = np.random.default_rng(7)
    Zs = np.zeros_like(X)
    for k in range(6):                                          # broad random crinkle
        kx, ky, ph = rng.uniform(3, 14), rng.uniform(3, 14), rng.uniform(0, 2*np.pi)
        Zs += 0.0009*np.sin(kx*X + ky*Y + ph)
    for f in (1/3, 2/3):                                        # factory fold lines
        Zs += 0.0016*np.exp(-((X - (x0 + f*(x1 - x0)))/0.012)**2)
        Zs += 0.0012*np.exp(-((Y - (y0 + f*(y1 - y0)))/0.012)**2)
    hem = (np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y]) < 0.025)
    Zs = np.where(hem, 0.0016, Zs - Zs.min() + 0.0004)
    v = np.stack([X, Y, z + Zs], -1).reshape(-1, 3)
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i*ny + j
            f += [a, a + ny, a + ny + 1, a, a + ny + 1, a + 1]
    uv = np.stack([(X - x0)/(x1 - x0), (Y - y0)/(y1 - y0)], -1).reshape(-1, 2)
    S.mesh("/Yard/tarp", v.tolist(), f, uv=uv.tolist(), mat=mats["tarp"], smooth=True)
    for k, (gx, gy) in enumerate(((x0 + 0.03, y0 + 0.03), (x1 - 0.03, y0 + 0.03), (x0 + 0.03, y1 - 0.03), (x1 - 0.03, y1 - 0.03))):
        S.cylinder(f"/Yard/tarp_grommet_{k}", 0.009, 0.002, mats["grommet"], (gx, gy, z + 0.0022), n=16)


def _surroundings(S, mats, spec):
    """Render-only context beyond the simulated yard: the rest of the front lawn and foundation bed, the whole house
    front (second window, front door and stoop, eave, gutter), wooden side fences and a hedge along the street, so
    renders do not end at the yard's 7 m x 5.4 m ground and show the sky image's own garden beyond its edge."""
    y = L.HOUSE_Y
    X0, X1 = L.YARD["x"]
    Y0 = L.YARD["y"][0]
    FX, FY = L.FENCE_X, L.STREET_Y               # side fences, street hedge (physical too: layout.context_boxes)
    z = L.GRADE - 0.0005
    # lawn around the yard (three strips) and the foundation bed's continuation along the house
    for k, (a0, b0, a1, b1) in enumerate(((-FX - 6, FY - 8, FX + 6, Y0), (-FX - 6, Y0, X0, 0.0), (X1, Y0, FX + 6, 0.0))):
        _ground_quad(S, mats, f"/Context/lawn_{k}", a0, b0, a1, b1, z, "lawn")
    for k, (a0, a1) in enumerate(((-FX - 6, X0), (X1, FX + 6))):
        _ground_quad(S, mats, f"/Context/bed_{k}", a0, 0.0, a1, y, z, "mulch")
        # black steel landscape edging between bed and lawn
        S.box(f"/Context/edging_{k}", (a1 - a0, 0.004, 0.05), mats["edging"], ((a0 + a1)/2, 0.0, L.GRADE + 0.005), bevel=0.001)
    # house front beyond the yard, foundation, eave and gutter
    for k, (a0, a1) in enumerate(((-FX - 6, X0), (X1, FX + 6))):
        S.mesh(f"/Context/siding_{k}", [(a0, y, 0.30), (a1, y, 0.30), (a1, y, 2.9), (a0, y, 2.9)], [0, 2, 1, 0, 3, 2],
               uv=[(a0/2, .15), (a1/2, .15), (a1/2, 1.45), (a0/2, 1.45)], normals=[(0, -1, 0)]*4, mat=mats["siding"])
        S.box(f"/Context/foundation_{k}", (a1 - a0, 0.25, 0.62), mats["concrete"], ((a0 + a1)/2, y + 0.125, -0.01),
              bevel=0.004, uv_per_m=1/1.5)
    S.box("/Context/eave", (2*FX + 12, 0.45, 0.20), mats["trim"], (0.0, y - 0.20, 3.0), bevel=0.01)
    S.box("/Context/roof", (2*FX + 12, 2.2, 0.06), mats["shingle"], (0.0, y + 0.55, 3.45),
          q=tuple(Rotation.from_euler("x", -26, degrees=True).as_quat()), bevel=0.005)
    S.cylinder("/Context/gutter", 0.06, 2*FX + 12, mats["downspout"], (0.0, y - 0.45, 2.95),
               q=tuple(Rotation.from_euler("y", 90, degrees=True).as_quat()), n=16)
    for k, xw in enumerate((4.9, -6.4)):
        S.box(f"/Context/window_glass_{k}", (1.2, 0.02, 1.1), mats["glass"], (xw, y - 0.01, 1.55), bevel=0.002)
        S.box(f"/Context/window_trim_{k}", (1.32, 0.03, 1.22), mats["trim"], (xw, y - 0.002, 1.55), bevel=0.004)
        S.box(f"/Context/sill_{k}", (1.40, 0.08, 0.04), mats["trim"], (xw, y - 0.04, 0.95), bevel=0.004)
    # front door at the walkway's east end, with its frame and a two-step concrete stoop
    xd = L.DOOR_X
    S.box("/Context/door", (0.92, 0.05, 2.05), mats["door"], (xd, y - 0.02, 0.30 + 1.025), bevel=0.004)
    S.box("/Context/door_frame", (1.08, 0.04, 2.15), mats["trim"], (xd, y - 0.005, 0.30 + 1.075), bevel=0.004)
    S.cylinder("/Context/door_knob", 0.03, 0.06, mats["path_light"], (xd + 0.36, y - 0.07, 1.30),
               q=tuple(Rotation.from_euler("x", 90, degrees=True).as_quat()), n=16)
    S.box("/Context/stoop_top", (1.5, 0.9, 0.30), mats["concrete"], (xd, y - 0.45, 0.15), bevel=0.006, uv_per_m=1/1.5)
    S.box("/Context/stoop_step", (1.5, 0.35, 0.15), mats["concrete"], (xd, y - 1.08, 0.075), bevel=0.006, uv_per_m=1/1.5)
    # wooden privacy fences on both sides (posts every 2.4 m, 1x6 boards) and a hedge along the street
    for k, xf in enumerate((-FX, FX)):
        S.box(f"/Context/fence_{k}", (0.02, y - FY, 1.8), mats["fence"], (xf, (y + FY)/2, 0.9), bevel=0.002)
        for j, yp in enumerate(np.arange(FY, y, 2.4)):
            S.box(f"/Context/fence_{k}_post_{j}", (0.09, 0.09, 1.9), mats["fence"], (xf - math.copysign(0.05, xf), yp, 0.95),
                  bevel=0.003)
        S.box(f"/Context/fence_{k}_rail", (0.04, y - FY, 0.09), mats["fence"], (xf - math.copysign(0.04, xf), (y + FY)/2, 1.5),
              bevel=0.002)
    return FX, FY


def _part_visual(S, path, p, mat):
    q = p.quat
    if p.kind == "box":
        S.box(path, 2*np.asarray(p.size), mat, p.pos, q, bevel=min(0.002, 0.3*min(p.size)))
    elif p.kind == "cylinder":
        S.cylinder(path, p.size[0], 2*p.size[1], mat, p.pos, q, n=28)
    elif p.kind == "capsule":
        r, hh = p.size
        S.cylinder(path, r, 2*hh, mat, p.pos, q, n=24)
        R = Rotation.from_quat(q)
        from soil_excavation.soil import unit_sphere
        v, f = unit_sphere(8, 16)
        for k, s in enumerate((-1., 1.)):
            c = np.asarray(p.pos) + R.apply((0., 0., s*hh))
            S.mesh(f"{path}_end{k}", v*r + c, f, mat=mat, smooth=True)
    elif p.kind == "ellipsoid":
        from soil_excavation.soil import unit_sphere
        v, f = unit_sphere(16, 32)
        rng = np.random.default_rng(abs(hash(path)) % 9973)
        bump = 1.0 + 0.06*np.sin(v @ rng.normal(size=3)*5) + 0.04*np.sin(v @ rng.normal(size=3)*9)
        vv = Rotation.from_quat(q).apply(v*np.asarray(p.size)*bump[:, None]) + np.asarray(p.pos)
        S.mesh(path, vv, f, mat=mat, smooth=True)
    elif p.kind == "mesh":
        v, f = p.mesh
        S.mesh(path, v, f, mat=mat, smooth=True, p=p.pos, q=q)


def _tools(S, world, mats):
    sc = world.scene
    for name, parts in sc.parts.items():
        body = sc.bodies[name]
        for k, p in enumerate(parts):
            if p.material == "mass_only" or "no_render" in p.role:
                continue
            if p.name == "flight_envelope":
                continue
            mat = mats.get(p.material) or mats["steel_black"]
            if p.name == "shrub":
                _shrub(S, f"/Lab/body_{body}/shrub", p)
                continue
            if p.material == "rootball":
                _rootball_visual(S, f"/Lab/body_{body}/rootball_soil", p, mats)
                continue
            _part_visual(S, f"/Lab/body_{body}/{name}_{k}", p, mat)


def _rootball_visual(S, path, part, mats):
    """Exposed nursery substrate with fine roots; millimetre dressing on the collider."""
    r, half = part.size
    count, rings = 64, 13
    vertices, uv, faces = [], [], []
    for j in range(rings):
        z = -half+2*half*j/(rings-1)
        for i in range(count):
            a = 2*math.pi*i/count
            rr = r+.0015*math.sin(7*a+39*z)+.0008*math.cos(19*a-55*z)
            vertices.append((rr*math.cos(a), rr*math.sin(a), z))
            uv.append((a*r/.08, (z+half)/.08))
    for j in range(rings-1):
        for i in range(count):
            a, b = j*count+i, j*count+(i+1)%count
            faces.extend((a, b, b+count, a, b+count, a+count))
    for j in (0, rings-1):
        centre = len(vertices)
        vertices.append((0, 0, -half if j == 0 else half))
        uv.append((.5, .5))
        for i in range(count):
            ring = (j*count+i, j*count+(i+1)%count)
            faces.extend((centre, *ring) if j else (centre, *reversed(ring)))
    R = Rotation.from_quat(part.quat)
    v = R.apply(vertices)+part.pos
    S.mesh(path, v, faces, uv=uv, mat=mats["rootball"], smooth=True)
    # Render-only root filaments hug the substrate; they do not brace or support it.
    for root in range(16):
        a0 = root*2*math.pi/16
        zz = np.linspace(-half+.004, half-.003, 9)
        aa = a0+.18*np.sin(35*zz+root)
        radius = r+.0018
        pts = R.apply(np.stack([radius*np.cos(aa), radius*np.sin(aa), zz], axis=1))+part.pos
        for j, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            axis = b-a
            rot, _ = Rotation.align_vectors([axis/np.linalg.norm(axis)], [[0., 0., 1.]])
            S.cylinder(f"{path}_root_{root}_{j}", .00055, float(np.linalg.norm(axis)),
                       mats["fine_roots"], (a+b)/2, rot.as_quat(), n=6)


def _shrub(S, path, p):
    """The #1 shrub's foliage on the root ball: the pinned shrub_04 model scaled to the part's 0.32 m envelope."""
    from .visual_assets import model
    try:
        usd = model("shrub_04")
    except Exception:
        return
    outer = S.UsdGeom.Xform.Define(S.stage, path)
    outer.AddTranslateOp().Set(S.Gf.Vec3d(p.pos[0], p.pos[1], p.pos[2] - p.size[1]))
    outer.AddScaleOp().Set(S.Gf.Vec3f(0.6, 0.6, 0.6))           # 0.35 reads as a bare twig
    inner = S.UsdGeom.Xform.Define(S.stage, path + "/asset")
    inner.GetPrim().GetReferences().AddReference(str(usd), "/shrub_04")
    _rebind(S, path, "shrub_04", usd)


def _rebind(S, path, name, usd):
    """Rebind a Poly Haven model's MaterialX graphs to OmniPBR with the same texture files (painting/percussion rule:
    MaterialX normal-map nodes crash the render-only worker)."""
    import re
    from pxr import Usd
    UsdShade, Tn, Sdf = S.UsdShade, S.Sdf.ValueTypeNames, S.Sdf
    made = {}
    for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)):
        if pr.GetTypeName() == "BasisCurves":
            pr.SetActive(False)
            continue
        if not pr.IsA(S.UsdGeom.Gprim):
            continue
        src, _ = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()
        if not src:
            continue
        key = src.GetPrim().GetName()
        if key not in made:
            files = {}
            for sh in Usd.PrimRange(src.GetPrim()):
                sid = UsdShade.Shader(sh).GetIdAttr().Get() if sh.IsA(UsdShade.Shader) else None
                if sid and (sid == "UsdUVTexture" or sid.startswith("ND_image")):    # Poly Haven: MaterialX image nodes
                    f = UsdShade.Shader(sh).GetInput("file").Get()
                    if f:
                        files[Path(f.path).name] = usd.parent/"textures"/Path(f.path).name
            pick = lambda tag: next((v for k, v in files.items() if re.search(tag, k)), None)
            m = UsdShade.Material.Define(S.stage, f"/Looks/pbr_{name}_{key}")
            sh = UsdShade.Shader.Define(S.stage, f"/Looks/pbr_{name}_{key}/Shader")
            sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            for tag, inp in (("_diff", "diffuse_texture"), ("_rough", "reflectionroughness_texture"), ("_nor_gl", "normalmap_texture"),
                             ("_alpha|_opacity", "opacity_texture")):
                f = pick(tag)
                if f is not None and f.exists() and f.suffix.lower() in (".jpg", ".png"):
                    sh.CreateInput(inp, Tn.Asset).Set(Sdf.AssetPath(str(f.resolve())))
                    if inp == "reflectionroughness_texture":
                        sh.CreateInput("reflection_roughness_texture_influence", Tn.Float).Set(1.0)
                    if inp == "opacity_texture":
                        sh.CreateInput("enable_opacity", Tn.Bool).Set(True)
                        sh.CreateInput("enable_opacity_texture", Tn.Bool).Set(True)
                        sh.CreateInput("opacity_mode", Tn.Int).Set(0)
                        sh.CreateInput("opacity_threshold", Tn.Float).Set(0.5)
            m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
            made[key] = m
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(made[key])
    for sp in [pr.GetPath() for pr in Usd.PrimRange(S.stage.GetPrimAtPath(path)) if pr.GetName() == "_materials"]:
        q = S.stage.GetPrimAtPath(sp)
        if q:
            q.SetActive(False)


def _soil_prim(S, mats):
    from pxr import Sdf, UsdGeom
    g = UsdGeom.Mesh.Define(S.stage, "/Lab/soil")
    g.CreateSubdivisionSchemeAttr("none")
    g.CreatePointsAttr()
    g.CreateNormalsAttr()
    g.SetNormalsInterpolation("vertex")
    g.CreateExtentAttr()
    g.CreateFaceVertexCountsAttr()
    g.CreateFaceVertexIndicesAttr()
    pv = UsdGeom.PrimvarsAPI(g)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    pv.CreatePrimvar("turf", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    pv.CreatePrimvar("mulch", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    S.bind(g, mats["soil"])


def _dressing(S, mats, spec):
    from .visual_assets import model
    y = L.HOUSE_Y

    def ref(name, path, p, yaw=0.0, scale=1.0):
        usd = model(name)
        outer = S.UsdGeom.Xform.Define(S.stage, path)
        outer.AddTranslateOp().Set(S.Gf.Vec3d(*map(float, p)))
        if yaw:
            outer.AddRotateZOp().Set(float(yaw))
        if scale != 1.0:
            outer.AddScaleOp().Set(S.Gf.Vec3f(scale, scale, scale))
        inner = S.UsdGeom.Xform.Define(S.stage, path + "/asset")
        inner.GetPrim().GetReferences().AddReference(str(usd), f"/{name}")
        _rebind(S, path, name, usd)
        return outer

    # foundation shrubs set back from the bed's work area (never inside a soil block)
    b = spec["block"]
    for k, (x, yy, nm, sc) in enumerate(((-2.3, 1.45, "shrub_01", 1.0), (2.2, 1.45, "shrub_02", 1.0), (-1.2, 1.55, "shrub_02", 0.8),
                                         (1.25, 1.55, "shrub_01", 0.85))):
        if b["lo"][0] - 0.3 < x < b["hi"][0] + 0.3 and b["lo"][1] - 0.3 < yy < b["hi"][1] + 0.3:
            continue
        ref(nm, f"/Dress/shrub_{k}", (x, yy, L.GRADE), yaw=37.0*k, scale=sc)
    ref("garden_hose_wall_mounted_01", "/Dress/hose", (2.5, y - 0.02, 0.55), yaw=0.0)
    # a hedge of the same shrubs along the street and a few foundation shrubs beyond the yard (render only)
    for k, xh in enumerate(np.arange(-8.4, 8.6, 1.2)):
        ref("shrub_02" if k % 2 else "shrub_01", f"/Dress/hedge_{k}", (float(xh), L.STREET_Y, L.GRADE), yaw=53.0*k, scale=1.5)
    for k, (xh, nm) in enumerate(((-5.0, "shrub_01"), (-7.6, "shrub_02"), (6.4, "shrub_02"), (8.0, "shrub_01"), (5.6, "shrub_04"))):
        ref(nm, f"/Dress/foundation_shrub_{k}", (xh, 1.25, L.GRADE), yaw=71.0*k, scale=1.1)
    ref("planter_pot_clay", "/Dress/pot", (-2.7, -0.35, 0.0), yaw=10.0)
    ref("garden_gloves_01", "/Dress/gloves", (-2.2, -0.9, 0.0), yaw=60.0)
    from .visual_assets import hdri
    dome = S.UsdLux.DomeLight.Define(S.stage, "/Lights/sky")
    dome.CreateTextureFileAttr().Set(S.Sdf.AssetPath(str(hdri().resolve())))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(3000.0)
    sun = S.UsdLux.DistantLight.Define(S.stage, "/Lights/sun")
    # Bright sky fill keeps the physical tool/soil interface legible in robot shadows. Diffuse overcast daylight
    # plus a weaker sun, rather than exposure changes per camera; policy cameras see the same lighting.
    sun.CreateIntensityAttr(1200.0); sun.CreateAngleAttr(2.0); sun.CreateColorAttr(S.Gf.Vec3f(1., .96, .90))
    # mid-morning sun from the east-south-east (the house faces south: the bed is in front of it). A DistantLight shines
    # along its local -z: rotateXYZ (a, 0, c) sends it toward (-sin a sin c, sin a cos c, -cos a); (39.6, 0, 70) gives
    # (-0.60, +0.22, -0.77), from azimuth ~110 deg at 50 deg elevation, so the robot's shadow falls to its side, not over
    # the soil in front of it. (35, 0, 150) would light the yard from the north-west (the house's eave shading the
    # whole bed); (35, 0, 30), from the south-east, would lay the robot's shadow across its own work area
    S.UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(S.Gf.Vec3f(39.6, 0.0, 70.0))


def _cameras(S, world, spec):
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
    from classroom_arrangement import params as CP
    g = S.camera(f"/Lab/body_{carriage}/head_rgb", pl, pl + Rl.apply((1, 0, 0)), 18/math.tan(math.radians(CP.HEAD_HFOV/2)),
                 up=Rl.apply((0, 0, 1)), role="policy")
    S.alias("head_rgb", g.GetPath()); out["head_rgb"] = "policy"
    for name, m in sc.cameras.items():
        g = S.camera(f"/Lab/body_{m['body']}/{name}", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)), up=m["up"],
                     role="policy")
        S.alias(name, g.GetPath()); out[name] = "policy"
    base = sc.robot_body["base_link"]
    g = S.camera(f"/Lab/body_{base}/chase", (-1.7, 1.0, 1.9), (1.0, 0.0, 0.3), 14., role="diagnostic")
    S.alias("chase", g.GetPath()); out["chase"] = "diagnostic"
    for name, (eye, target, f) in diagnostic_views(spec).items():
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); out[name] = "diagnostic"
    return out


def diagnostic_views(spec):
    """Fixed diagnostic cameras around the work (eye, target, focal mm on a 36 mm aperture)."""
    rx, ry, yaw = spec["robot"]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.])
    side = np.array([-fwd[1], fwd[0], 0.])
    b = spec["block"]
    tgt = np.array(spec.get("target") or ((b["lo"][0] + b["hi"][0])/2, (b["lo"][1] + b["hi"][1])/2)) if not spec.get("targets") \
        else np.mean(np.asarray(spec["targets"]), 0)
    if spec.get("target_line"):
        tgt = np.mean(np.asarray(spec["target_line"]), 0)
    if spec.get("target_square"):
        tgt = np.mean(np.asarray(spec["target_square"]), 0)
    T = np.array([tgt[0], tgt[1], L.GRADE])
    base = np.array([rx, ry, 0.])
    (tx0, ty0), (tx1, ty1) = spec["tarp"]
    tarp = np.array([(tx0 + tx1)/2, (ty0 + ty1)/2, 0.])
    views = dict(
        yard=(T - 1.8*fwd + 2.6*side + np.array([0, 0, 2.3]), T - 0.25*fwd + np.array([0, 0, 0.7]), 16.),
        work=(T + 0.95*side + 0.15*fwd + np.array([0, 0, 0.42]), T + np.array([0, 0, 0.02]), 20.),
        work_low=(T + 0.70*side + 0.20*fwd + np.array([0, 0, 0.16]), T + np.array([0, 0, -0.04]), 22.),
        hole=(T + 0.55*fwd + 0.20*side + np.array([0, 0, 0.85]), T + np.array([0, 0, -0.08]), 22.),
        top=(T + np.array([0.001, 0.001, 1.8]), T, 22.),
        tarp=(tarp - 0.9*side + np.array([0, 0, 1.0]) if abs(fwd[1]) > 0.5 else tarp + np.array([0, 0.9, 1.0]), tarp, 18.),
        front=(T + 0.9*fwd + 0.5*side + np.array([0, 0, 0.9]), base + 0.4*fwd + np.array([0, 0, 0.6]), 15.),
    )
    return {k: (tuple(map(float, e)), tuple(map(float, t)), f) for k, (e, t, f) in views.items()}


# ------------------------------------------------------------------------------ dynamic state
POLICY_CELL = 1.0         # marching-cubes cell for policy images, in particle spacings (diagnostic videos use 0.5)
_CACHE = {}


def soil_payload(env, folder, cell=None):
    """Reconstruct the soil surface from the current particles and write it next to the stage; returns the path, or
    None when no particle moved more than 0.5 mm since the last reconstruction for this environment (the worker keeps
    the mesh it has)."""
    w = env.world
    pos = w.soil.state.particle_q.numpy()
    key = id(env)
    last = _CACHE.get(key)
    cell = cell or POLICY_CELL*w.soil.h
    if last is not None and last["cell"] == cell and len(last["pos"]) == len(pos) and \
            float(np.abs(pos - last["pos"]).max()) < 5e-4:
        return None
    parts = w.soil.particles()
    parts["paint"] = w.particle_spec.get("paint")
    m = SF.soil_mesh(parts, w.soil.h, L.GRADE, moisture=env.spec["moisture"], cell=cell, block=env.spec["block"])
    path = Path(folder)/f"soil_{uuid.uuid4().hex[:8]}.npz"
    np.savez(path, **m)
    _CACHE[key] = dict(pos=pos.copy(), cell=cell)
    return path


class Renderer:
    """Owns exactly one render-only Isaac worker (soil_excavation/render_worker.py)."""

    def __init__(self, folder, scene, meta, *, width=640, height=360, samples=16):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.meta = meta
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, SOIL_RENDER_PORT=str(listener.getsockname()[1]), SOIL_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "soil_excavation.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
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
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            reply = receive(self.connection)
        if not reply.get("ok") or reply.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(reply)[:400])
        return reply

    def render(self, cameras, body_q, tick, soil_npz, resolution=None, samples=None, settings=None):
        from .render_worker import EXPOSURE
        msg = dict(op="observe", cameras=list(cameras), body_poses=np.round(np.asarray(body_q), 7).tolist(), tick=int(tick),
                   soil=str(soil_npz) if soil_npz else None, settings=dict(EXPOSURE, **(settings or {})))
        if resolution is not None:
            msg["resolution"] = list(resolution)
        if samples is not None:
            msg["spp"] = int(samples)
        reply = self._call(msg)
        self.last_timings = reply.get("timings", {})
        return {k: base64.b64decode(v) for k, v in reply["images"].items()}

    def observe(self, env, cameras, resolution=None, samples=None, cell=None):
        """Render the named cameras from the environment's current state; returns {camera: PNG bytes}."""
        t0 = time.perf_counter()
        npz = soil_payload(env, self.folder, cell=cell)
        t1 = time.perf_counter()
        if npz is None and getattr(self, "_soil_sent", None) != id(env):
            _CACHE.pop(id(env), None)                       # a new worker / env: send the surface once
            npz = soil_payload(env, self.folder, cell=cell)
        try:
            out = self.render(cameras, env.world.state.body_q.numpy(), env.world.tick, npz, resolution, samples)
            self._soil_sent = id(env)
        finally:
            if npz is not None:
                npz.unlink(missing_ok=True)
        self.last_timings = dict(self.last_timings, soil_surface_s=t1 - t0)
        return out

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

"""USD export of room 112 and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses, the pose of every loose piece and
the node positions of every bristle cluster, so one exported stage serves a whole session or a whole film.
Appearance rules:
  * furniture and the robot are drawn as in classroom_arrangement (same part lists, same robot meshes);
  * tools, containers, the wall rail and the table supplies are drawn as the products they are (moulded heads and
    pans, sockets, grips, hanger caps, rims, ribs, labels of a made-up maker) by sweeping.visuals, from the part lists
    the physics uses and in the same body frames: what can be touched lies on its collision part, what is added for
    the look stays within a few millimetres of it;
  * every bristle cluster of the solver is drawn as five filament strands that follow its nodes, their tips spread
    over a third of the distance between the rows of clusters (the collision model is the cluster's one rod);
  * every loose piece is drawn where the solver has it: grains, rings, beads and crumbs as instances of one mesh per
    class; wads as creased paper, scraps as curled snips, pencils with ferrule and eraser, mug shards glazed with a
    matt broken edge, each built from the piece's spheres;
  * the room shell (walls, windows, ceiling grid, lights, sky) is that of room 114; casework, cubbies, tool rail,
    mat, sand table and taped square are this room's.
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

from classroom_arrangement import params as CP
from classroom_arrangement import render as CR
from classroom_arrangement.render import Stage, textures, furniture_visual, hdri, EXTERIOR_HDRI, clip_mesh, _rot_to_wxyz
from . import params as P
from . import layout as L
from . import visuals as V
from .solver import principal

COLORS = dict(CR.COLORS)
COLORS.update(
    broom_red=((.62, .08, .08), .45, 0.), broom_black=((.05, .05, .055), .5, 0.), broom_block=((.10, .16, .36), .5, 0.),
    handle_steel=((.72, .10, .10), .35, .2), handle_wood=((.72, .56, .34), .55, 0.), cap_black=((.04, .04, .045), .5, 0.),
    brush_wood=((.60, .44, .26), .5, 0.), pan_black=((.06, .06, .065), .45, 0.), pan_grey=((.42, .44, .46), .5, 0.),
    barrel_grey=((.46, .47, .48), .6, 0.), pail_blue=((.10, .28, .60), .45, 0.), basket=((.25, .27, .28), .6, 0.),
    liner=((.07, .07, .075), .35, 0.), casework=((.86, .84, .78), .5, 0.), kick=((.10, .10, .11), .6, 0.),
    counter=((.30, .32, .34), .35, 0.), cove=((.10, .10, .11), .6, 0.), mat=((.16, .17, .19), .95, 0.),
    mat_border=((.05, .05, .055), .7, 0.), rail_steel=((.55, .57, .60), .3, .9), sand_frame=((.80, .62, .36), .5, 0.),
    sand_tub=((.75, .86, .92), .15, 0.), tape=((.10, .30, .78), .55, 0.), sink=((.70, .72, .74), .25, .9),
    cup=((.85, .30, .12), .5, 0.), ream=((.94, .94, .92), .7, 0.), ream_band=((.15, .35, .65), .6, 0.),
    scissors=((.80, .15, .15), .4, 0.), glue=((.95, .95, .93), .4, 0.), glue_cap=((.95, .55, .10), .4, 0.),
    bristle_angle=((.10, .10, .11), .65, 0.), bristle_lobby=((.12, .11, .10), .65, 0.), bristle_push=((.40, .16, .10), .65, 0.),
    bristle_counter=((.72, .62, .42), .65, 0.),
    grit=((.48, .42, .34), .9, 0.), rice=((.93, .91, .84), .55, 0.), cereal=((.78, .60, .32), .8, 0.),
    crumb=((.86, .52, .56), .9, 0.), wad=((.93, .93, .91), .85, 0.), shard=((.88, .90, .93), .2, 0.),
    shard_edge=((.80, .76, .70), .8, 0.), pencil=((.93, .72, .10), .5, 0.), bead_0=((.85, .12, .15), .3, 0.),
    bead_1=((.10, .35, .80), .3, 0.), bead_2=((.95, .78, .10), .3, 0.), bead_3=((.15, .62, .28), .3, 0.),
    scrap_0=((.92, .30, .30), .8, 0.), scrap_1=((.30, .55, .90), .8, 0.), scrap_2=((.95, .85, .30), .8, 0.),
    scrap_3=((.93, .93, .90), .8, 0.), towel=((.90, .90, .88), .6, 0.), soap=((.30, .60, .45), .3, 0.),
)
BRISTLE_LOOK = dict(angle_broom="bristle_angle", lobby_broom="bristle_lobby", push_broom="bristle_push",
                    counter_brush="bristle_counter")
STRANDS = 5                    # drawn filament strands per simulated cluster


# ------------------------------------------------------------------------------ small mesh makers
def _icosphere(sub=1):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _torus(R, r, n=20, m=10):
    v, f = [], []
    for i in range(n):
        a = 2*math.pi*i/n
        for j in range(m):
            b = 2*math.pi*j/m
            v.append(((R + r*math.cos(b))*math.cos(a), (R + r*math.cos(b))*math.sin(a), r*math.sin(b)))
    for i in range(n):
        for j in range(m):
            a, b = i*m + j, i*m + (j + 1) % m
            c, d = ((i + 1) % n)*m + j, ((i + 1) % n)*m + (j + 1) % m
            f += [(a, c, d), (a, d, b)]
    return np.array(v), np.array(f)


def _bead(r_out, r_in, length, n=18):
    """A barrel with a hole along x."""
    prof = [(-length/2, r_in), (-length/2, r_out*0.86), (0., r_out), (length/2, r_out*0.86), (length/2, r_in)]
    v, f = [], []
    for i in range(n):
        a = 2*math.pi*i/n
        for x, r in prof:
            v.append((x, r*math.cos(a), r*math.sin(a)))
    k = len(prof)
    for i in range(n):
        for j in range(k):
            a, b = i*k + j, i*k + (j + 1) % k
            c, d = ((i + 1) % n)*k + j, ((i + 1) % n)*k + (j + 1) % k
            f += [(a, c, d), (a, d, b)]
    return np.array(v), np.array(f)


def _hull(centres, radii):
    from scipy.spatial import ConvexHull
    d, _ = _icosphere(0)
    pts = np.concatenate([c + r*d for c, r in zip(centres, radii)])
    h = ConvexHull(pts)
    v = pts[h.vertices]
    remap = -np.ones(len(pts), int); remap[h.vertices] = np.arange(len(h.vertices))
    f = remap[h.simplices]
    cen = v.mean(0)
    for k, tri in enumerate(f):
        a, b, c = v[tri]
        if np.dot(np.cross(b - a, c - a), a - cen) < 0:
            f[k] = tri[::-1]
    return v, f


# ------------------------------------------------------------------------------ tools, supplies, debris
def tool_visual(S, base, tool, mats, handle_base=None):
    """The tool's picture in its body frame (sweeping.visuals builds it from the tool's collision parts); the lobby
    pan's yoke and handle go under handle_base, the frame of its second body (origin at the pivot)."""
    body, handle = V.tool(tool["kind"], tool["spec"])
    body.emit(S, base, mats)
    if handle is not None:
        handle.emit(S, handle_base, mats)


def supply_visual(S, base, kind, parts, mats):
    if kind == "scissors":       # unchanged: closed scissors do not fit their 24 mm wide collision box (visuals.NOTES)
        S.box(f"{base}/blade_a", (0.016, 0.11, 0.003), mats["chrome"], (0.004, 0.04, 0.004), Rotation.from_euler("z", 0.12).as_quat(), bevel=0.0005)
        S.box(f"{base}/blade_b", (0.016, 0.11, 0.003), mats["chrome"], (-0.004, 0.04, 0.006), Rotation.from_euler("z", -0.12).as_quat(), bevel=0.0005)
        for k, sx in enumerate((-1, 1)):
            v, f = _torus(0.018, 0.005)
            S.mesh(f"{base}/loop_{k}", v, f, mat=mats["scissors"], smooth=True, p=(sx*0.022, -0.055, 0.005), q=(0, 0, 0, 1))
        return
    V.supply(kind, parts).emit(S, base, mats)


def debris_visual(S, world, mats):
    """Instanced and per-piece geometry of the loose pieces, each in its piece's solver frame. Returns the meta the
    worker needs: {instancers: {path: piece indices}, prims: [(path, piece index)]}."""
    pieces = world.spec["debris"]
    UsdGeom, Vt, Gf = S.UsdGeom, S.Vt, S.Gf
    S.xform("/Debris")
    by_cls = {}
    for k, p in enumerate(pieces):
        by_cls.setdefault(p.cls, []).append(k)
    inst, prims = {}, []
    start = world.debris()

    def instancer(name, idx, protos, proto_of, scales=None):
        path = f"/Debris/{name}"
        pi = UsdGeom.PointInstancer.Define(S.stage, path)
        targets = []
        for j, (v, f, mat) in enumerate(protos):
            g = S.mesh(f"{path}/proto_{j}", v, f, mat=mat, smooth=True)
            targets.append(g.GetPath())
        pi.CreatePrototypesRel().SetTargets(targets)
        pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(proto_of, np.int32)))
        pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(start["pos"][idx].astype(np.float32)))
        pi.CreateOrientationsAttr(Vt.QuathArray.FromNumpy(start["quat"][idx].astype(np.float16)))
        if scales is not None:
            pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.asarray(scales, np.float32)))
        inst[path] = [int(i) for i in idx]

    def body_frame(p, v):
        com, _, V = principal(p.centres, p.radii, p.mass)
        return (np.asarray(v, float) - com) @ V

    sv, sf = _icosphere(1)
    rough = sv*(1 + 0.18*np.random.default_rng(2).normal(size=(len(sv), 1)))
    for cls, idx in by_cls.items():
        idx = np.array(idx, int)
        p0 = pieces[idx[0]]
        if cls in ("grit", "crumb"):
            rad = np.array([pieces[k].radii[0] for k in idx])
            instancer(cls, idx, [(rough, sf, mats[cls])], np.zeros(len(idx), int), scales=np.repeat(rad[:, None], 3, 1))
        elif cls == "rice":
            r = P.RICE
            v = sv*np.array([r["length"]/2, r["width"]/2, r["thick"]/2])
            instancer(cls, idx, [(body_frame(p0, v), sf, mats["rice"])], np.zeros(len(idx), int))
        elif cls == "cereal":
            c = P.CEREAL
            v, f = _torus((c["d"] - c["tube"])/2, c["tube"]/2)
            instancer(cls, idx, [(body_frame(p0, v), f, mats["cereal"])], np.zeros(len(idx), int))
        elif cls == "bead":
            b = P.BEAD
            v, f = _bead(b["d"]/2, 0.0021, b["length"])
            instancer(cls, idx, [(body_frame(p0, v), f, mats[f"bead_{j}"]) for j in range(4)], idx % 4)
        else:
            for k in idx:
                p = pieces[k]
                path = f"/Debris/{cls}_{k}"
                S.xform(path, start["pos"][k], start["quat"][k])
                A = V.Asm()                                          # in the piece's own frame (that of its spheres)
                if cls == "scrap":
                    mid = (p.centres.max(0) + p.centres.min(0))/2
                    v, f = V.scrap(float(np.ptp(p.centres[:, 0])), float(np.ptp(p.centres[:, 1])), seed=int(k))
                    A.add(f"snip_{k % 8}", (v + mid, f), crease=30.)
                elif cls == "pencil":
                    A.merge(V.pencil(P.PENCIL["length"], P.PENCIL["d"]))
                elif cls == "wad":
                    A.add(("paper_ruled", "paper_plain", "paper_ruled", "paper_yellow")[k % 4], V.wad(p.centres, p.radii, seed=int(k)),
                          uv=("box", 17.9), crease=0.)
                elif getattr(p, "grid", None) is not None:           # a piece of the mug's wall
                    na, nz, normals = p.grid
                    for m, vf in V.shard_plate(p.centres, normals, na, nz, P.MUG["wall"], seed=int(k)).items():
                        A.add(m, vf, uv=("box", 30.), crease=0. if m == "biscuit" else 50.)
                elif p.tag == "handle":                              # the mug's handle: a bent rod broken at both ends
                    r = float(p.radii[0])*0.97
                    A.add("glaze_out", V.tube(p.centres, r, 14, caps=False), uv=("box", 30.))
                    for end, nxt in ((p.centres[0], p.centres[1]), (p.centres[-1], p.centres[-2])):
                        A.add("biscuit", V.revolve([(r, 0.), (r*0.6, 0.0005), (0., 0.0003)], 14), R=V.basis(end - nxt, (0., 1., 0.)), p=end, crease=0.)
                else:                                                # pieces of the base and chips
                    v, f = _hull(p.centres, p.radii*0.95)
                    if p.tag == "chip":
                        A.add("glaze_out" if k % 3 == 0 else "biscuit", (v, f), uv=("box", 30.), crease=0.)
                    else:                                            # glazed above, the unglazed foot and the break below and around
                        nz_ = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
                        up = nz_[:, 2]/np.maximum(np.linalg.norm(nz_, axis=1), 1e-30) > 0.55
                        A.add("glaze_in", (v, f[up]), crease=0.)
                        A.add("biscuit", (v, f[~up]), crease=0.)
                com, _, Vp = principal(p.centres, p.radii, p.mass)   # the solver's frame of the piece: (x - com) @ Vp
                V.Asm().merge(A, R=Vp.T, p=-(com @ Vp)).emit(S, path, mats)
                prims.append((path, int(k)))
    return dict(instancers=inst, prims=prims)


def bristle_visual(S, world, mats):
    """One BasisCurves prim per bank: STRANDS strands a cluster, each following the cluster's nodes with a fixed
    sideways offset that grows from zero at the block to about the cluster's width at the tip."""
    UsdGeom, Vt = S.UsdGeom, S.Vt
    rods, slices = world.bristles()
    sw = world.sweeper
    S.xform("/Bristles")
    meta = {}
    rng = np.random.default_rng(9)
    for t, name in enumerate(world.tool_names):
        bank = sw.banks[t]
        sl = slices[name]
        n = bank.nodes
        R = sl.stop - sl.start
        # how far a strand's tip stands from its cluster's: a third of the distance between the rows of clusters (at
        # most 2.5 mm), so that a bank reads as a brush and not as the rows of its clusters (with 1.6 radii, about a
        # millimetre, the lobby broom's five rows read as five flat slats). At the block every
        # strand starts on its cluster. The first strand of a cluster is the cluster itself.
        ys = np.unique(np.round(np.asarray(bank.roots)[:, 1], 5))
        pitch = float(np.diff(ys).mean()) if len(ys) > 1 else 4.0*bank.radius
        off = rng.normal(size=(R, STRANDS, 3))*min(max(pitch/3.0, bank.radius*1.6), 2.5e-3)
        off[:, :, 2] *= 0.3
        off[:, 0] = 0.0
        pts = strands(rods[sl][:, :n], off)
        path = f"/Bristles/{name}"
        c = UsdGeom.BasisCurves.Define(S.stage, path)
        c.CreateTypeAttr("linear")
        c.CreateCurveVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(R*STRANDS, n, np.int32)))
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        w = 2*bank.radius*0.62*math.sqrt(3.0/STRANDS)
        c.CreateWidthsAttr(Vt.FloatArray.FromNumpy(np.full(R*STRANDS*n, w, np.float32)))
        c.SetWidthsInterpolation("vertex")
        c.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([[-5, -1, -1], [5, 11, 4]], np.float32)))
        S.bind(c, mats[BRISTLE_LOOK.get(world.scene.tools[name]["kind"], "bristle_angle")])
        meta[path] = dict(start=int(sl.start), stop=int(sl.stop), nodes=int(n), offsets=off.astype(np.float32).tolist())
    return meta


def strands(nodes, off):
    """nodes (R, n, 3), off (R, STRANDS, 3) -> points (R*STRANDS*n, 3)."""
    R, n, _ = nodes.shape
    grow = (np.arange(n)/(n - 1))[None, None, :, None]
    return (nodes[:, None, :, :] + off[:, :, None, :]*grow).reshape(-1, 3)


# ------------------------------------------------------------------------------ robot (as classroom_arrangement.render.export)
def robot_visual(S, world, mats):
    import newton
    sc = world.scene
    b = sc.b
    V, C = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    for i in sc.robot_shapes:
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
                v = v*1e-3
            f = np.asarray(src.indices).reshape(-1, 3)
            if label == "base_link" and len(v) > 100000:
                v2, below, above = clip_mesh(v, f, 0.26)
                for part, faces, m in (("skirt", below, mats["tmr_black"]), ("deck", above, mats["tmr_deck"])):
                    S.mesh(f"{path}_{part}", v2, faces, mat=m, smooth=True, p=tf[:3], q=tf[3:])
                continue
            if label == "base_link":
                m = mats["robot"]
            else:
                key = f"robot_{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
                m = S.color(key, col, .35, 0.)
            S.mesh(path, v, f, mat=m, smooth=True, p=tf[:3], q=tf[3:])
        elif b.shape_type[i] == newton.GeoType.BOX:
            S.box(path, 2*scale, mats["tmr_black"], tf[:3], tf[3:])
        elif b.shape_type[i] == newton.GeoType.CYLINDER:
            S.cylinder(path, scale[0], 2*scale[1], mats["robot_dark"], tf[:3], tf[3:])


# ------------------------------------------------------------------------------ room
def room(S, mats, tex, spec):
    X, Y, Z = CP.ROOM_X, CP.ROOM_Y, CP.ROOM_Z
    UsdLux, Gf = S.UsdLux, S.Gf
    sheet = 8*0.3048
    pts = [(-X/2, 0, 0), (X/2, 0, 0), (X/2, Y, 0), (-X/2, Y, 0)]
    S.mesh("/Room/floor", pts, [0, 1, 2, 0, 2, 3], uv=[(p[0]/sheet, p[1]/sheet) for p in pts], normals=[(0, 0, 1)]*4, mat=mats["vct"])
    T = 0.15
    cmu = 1/1.6256
    S.box("/Room/wall_front", (X + 2*T, T, Z), mats["wall_front"], (0, -T/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/wall_back", (X + 2*T, T, Z), mats["wall"], (0, Y + T/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/wall_right", (T, Y, Z), mats["wall"], (X/2 + T/2, Y/2, Z/2), (0, 0, 0, 1), uv_per_m=cmu)
    S.box("/Room/ceiling", (X + 2*T, Y + 2*T, 0.05), mats["ceiling_tex"], (0, Y/2, Z + 0.025), (0, 0, 0, 1), uv_per_m=1/0.6096)
    for k in range(13):
        S.box(f"/Room/grid_x_{k}", (0.015, Y, 0.01), mats["frame_white"], (-X/2 + 0.1 + k*0.6096, Y/2, Z - 0.005), (0, 0, 0, 1))
    for k in range(16):
        S.box(f"/Room/grid_y_{k}", (X, 0.015, 0.01), mats["frame_white"], (0, 0.3 + k*0.6096, Z - 0.005), (0, 0, 0, 1))
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
    # installed parts (physics: layout.STATIC and STATIC_CAPSULES), drawn from the same lists
    for name, c, h, mat, _ in L.STATIC:
        if name.startswith("wall") or name == "mat" or name.startswith("rail"):
            continue                                                 # the tool rail is drawn below, with its prongs
        m = mats.get(mat) or mats["frame_white"]
        S.box(f"/Room/{name}", 2*np.asarray(h), m, c, (0, 0, 0, 1), bevel=0.003)
    for name, a, b_, r, mat, _ in L.STATIC_CAPSULES:
        if not name.startswith("prong"):
            S.tube(f"/Room/{name}", [a, b_], r, mats[mat], n=10)
    V.rail([(n, np.asarray(c, float), np.asarray(h, float)) for n, c, h, *_ in L.STATIC if n.startswith("rail")],
           [(n, a, b_, r) for n, a, b_, r, *_ in L.STATIC_CAPSULES if n.startswith("prong")]).emit(S, "/Room/tool_rail", mats)
    for k, x in enumerate((-1.22, 1.22)):
        S.box(f"/Room/board_trim_{k}", (0.02, 0.03, 1.24), mats["aluminum"], (x, 0.02, 1.524), (0, 0, 0, 1))
    for k, z in enumerate((0.914, 2.134)):
        S.box(f"/Room/board_trim_h_{k}", (2.46, 0.03, 0.02), mats["aluminum"], (0, 0.02, z), (0, 0, 0, 1))
    dx, dl = CP.DOOR_X, CP.DOOR_LEAF
    S.box("/Room/door_frame_l", (0.06, 0.08, 2.15), mats["frame_white"], (dx - dl/2 - 0.03, 0.03, 1.075), (0, 0, 0, 1))
    S.box("/Room/door_frame_r", (0.06, 0.08, 2.15), mats["frame_white"], (dx + dl/2 + 0.03, 0.03, 1.075), (0, 0, 0, 1))
    S.box("/Room/door_frame_t", (dl + 0.12, 0.08, 0.06), mats["frame_white"], (dx, 0.03, 2.11), (0, 0, 0, 1))
    S.box("/Room/door_window", (0.22, 0.01, 0.55), mats["glass_dark"], (dx - 0.25, 0.045, 1.55), (0, 0, 0, 1))
    S.tube("/Room/door_handle", [(dx + 0.30, 0.10, 1.00), (dx + 0.42, 0.10, 1.00)], 0.012, mats["chrome"])
    # entrance mat: carpet with a bevelled rubber border
    mx, my = P.MAT["centre"]; sx, sy = P.MAT["size"]
    S.box("/Room/mat", (sx - 0.05, sy - 0.05, P.MAT["pile"]), mats["mat"], (mx, my, P.MAT["pile"]/2), (0, 0, 0, 1), bevel=0.002)
    S.box("/Room/mat_border", (sx, sy, P.MAT["pile"]*0.7), mats["mat_border"], (mx, my, P.MAT["pile"]*0.35), (0, 0, 0, 1), bevel=0.003)
    # taped square: four strips of 19 mm blue tape
    cx, cy = spec["square"]["centre"]; h = spec["square"]["side"]/2
    for k, (c, size) in enumerate((((cx, cy - h), (2*h + 0.019, 0.019)), ((cx, cy + h), (2*h + 0.019, 0.019)),
                                   ((cx - h, cy), (0.019, 2*h)), ((cx + h, cy), (0.019, 2*h)))):
        S.box(f"/Room/tape_{k}", (size[0], size[1], 0.0003), mats["tape"], (c[0], c[1], 0.00015), (0, 0, 0, 1), bevel=0.0)
    # casework: door lines, pulls, sink and faucet, towel dispenser, soap
    cw = P.CASEWORK
    yf = Y - cw["depth"]
    for k, x in enumerate(np.arange(cw["x"][0] + 0.6, cw["x"][1] - 0.01, 0.6)):
        S.box(f"/Room/cab_gap_{k}", (0.004, 0.004, cw["top"] - 0.16), mats["kick"], (x, yf - 0.001, 0.10 + (cw["top"] - 0.16)/2), (0, 0, 0, 1), bevel=0.)
        S.tube(f"/Room/cab_pull_{k}", [(x - 0.05, yf - 0.02, 0.78), (x - 0.05, yf - 0.02, 0.66)], 0.005, mats["chrome"])
    S.box("/Room/sink", (0.50, 0.38, 0.012), mats["sink"], (0., Y - 0.30, cw["top"] + 0.001), (0, 0, 0, 1), bevel=0.004)
    S.box("/Room/sink_bowl", (0.44, 0.32, 0.004), S.color("sink_dark", (.35, .36, .37), .3, .8), (0., Y - 0.30, cw["top"] + 0.008), (0, 0, 0, 1), bevel=0.)
    S.tube("/Room/faucet", [(0., Y - 0.07, cw["top"]), (0., Y - 0.07, cw["top"] + 0.24), (0., Y - 0.20, cw["top"] + 0.27), (0., Y - 0.22, cw["top"] + 0.20)],
           0.011, mats["chrome"])
    S.box("/Room/towels", (0.28, 0.10, 0.36), mats["towel"], (0.75, Y - 0.05, 1.50), (0, 0, 0, 1), bevel=0.01)
    S.cylinder("/Room/soap", 0.035, 0.16, mats["soap"], (0.38, Y - 0.10, cw["top"] + 0.08))
    rng = np.random.default_rng(12)
    for k in range(6):       # craft boxes and bins on the counter
        x = -2.2 + k*0.42 + rng.uniform(-.05, .05)
        if abs(x) < 0.45:
            continue
        S.box(f"/Room/craft_{k}", (0.30, 0.22, rng.uniform(0.10, 0.22)), mats[("red", "blue", "yellow", "green")[k % 4]],
              (x, Y - 0.22, cw["top"] + 0.11), Rotation.from_euler("z", rng.uniform(-.1, .1)).as_quat(), bevel=0.01)
    # cubbies: bins in three rows
    cb = P.CUBBIES
    for i in range(3):
        for j in range(5):
            S.box(f"/Room/cubby_gap_{i}_{j}", (0.005, 0.40, 0.32), mats["kick"], (X/2 - cb["depth"] - 0.001, cb["y"][0] + 0.25 + j*0.45, 0.30 + i*0.36), (0, 0, 0, 1), bevel=0.)
            S.box(f"/Room/cubby_bin_{i}_{j}", (0.02, 0.36, 0.26), mats[("red", "blue", "yellow", "green")[(i + j) % 4]],
                  (X/2 - cb["depth"] - 0.008, cb["y"][0] + 0.25 + j*0.45, 0.29 + i*0.36), (0, 0, 0, 1), bevel=0.004)
    # sand table: sand in the tub
    sx_, sy_ = P.SAND_TABLE["centre"]
    S.box("/Room/sand", (P.SAND_TABLE["size"][0] - 0.04, P.SAND_TABLE["size"][1] - 0.04, 0.01), mats["rice"],
          (sx_, sy_, P.SAND_TABLE["height"] + 0.001), (0, 0, 0, 1), bevel=0.)
    # wall dressing (render only): bulletin board with children's work, posters above the counter, alphabet, clock
    S.box("/Room/bulletin", (0.02, 1.80, 1.10), mats["cork_tex"], (X/2 - 0.01, 4.60, 1.55), (0, 0, 0, 1), uv_per_m=1.)
    S.box("/Room/bulletin_trim", (0.025, 1.84, 1.14), mats["aluminum"], (X/2 - 0.005, 4.60, 1.55), (0, 0, 0, 1))
    rng = np.random.default_rng(7)
    for k in range(10):
        S.box(f"/Room/note_{k}", (0.003, 0.215, 0.28), mats[("paper", "yellow", "paper", "blue")[k % 4]],
              (X/2 - 0.022, 3.90 + (k % 5)*0.33 + rng.uniform(-.03, .03), 1.25 + (k // 5)*0.48 + rng.uniform(-.04, .04)),
              Rotation.from_euler("x", rng.uniform(-.08, .08)).as_quat())
    for k in range(4):
        m = S.textured(f"poster2_{k}", tex[f"poster2_{k}"], rough=.6)
        # the image's left is on the reader's left: +x on the back wall, -x on the front wall (they were mirrored)
        S.quad(f"/Room/poster_{k}", (-1.9 + k*1.2, Y - 0.005, 1.95), (1, 0, 0), (0, 0, 1), (0.60, 0.84), m)
    S.quad("/Room/alphabet", (0, 0.004, 2.45), (-1, 0, 0), (0, 0, 1), (3.6, 0.28), S.textured("alphabet2", tex["alphabet2"], .7))
    S.quad("/Room/clock", (2.15, 0.006, 2.45), (-1, 0, 0), (0, 0, 1), (0.32, 0.32), S.textured("clock", tex["clock"], .4))   # clear of the strip's end
    pot = S.color("pot", (.62, .32, .20), .8); leaf = S.color("leaf", (.16, .42, .14), .7)
    for k, y in enumerate((1.2, 4.3, 7.4)):
        x = -X/2 + 0.10
        S.cylinder(f"/Room/pot_{k}", 0.07, 0.13, pot, (x, y, 0.985))
        for j in range(7):
            a = j*2*math.pi/7 + rng.uniform(-.2, .2)
            S.tube(f"/Room/leaf_{k}_{j}", [(x, y, 1.04), (x + 0.12*math.cos(a), y + 0.12*math.sin(a), 1.20 + rng.uniform(-.03, .05))], 0.018, leaf)
    # lights: six troffers, sky dome and sun through the windows (room 114's)
    for i, y in enumerate((1.9, 4.8, 7.7)):
        for j, x in enumerate((-1.8, 1.8)):
            S.box(f"/Room/troffer_{i}_{j}", (0.61, 1.22, 0.04), mats["frame_white"], (x, y, Z - 0.02), (0, 0, 0, 1))
            S.box(f"/Room/diffuser_{i}_{j}", (0.57, 1.18, 0.005), S.color("diffuser", (.95, .95, .95), .3, 0., emissive=(4., 4., 3.8)),
                  (x, y, Z - 0.043), (0, 0, 0, 1))
            lt = UsdLux.RectLight.Define(S.stage, f"/Room/light_{i}_{j}")
            lt.CreateWidthAttr(0.57); lt.CreateHeightAttr(1.18); lt.CreateIntensityAttr(2000.); lt.CreateExposureAttr(0.)
            lt.CreateColorAttr(Gf.Vec3f(1., .97, .92))
            S.UsdGeom.Xformable(lt.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, Z - 0.05))
    dome = UsdLux.DomeLight.Define(S.stage, "/Room/sky")
    dome.CreateTextureFileAttr(S.Sdf.AssetPath(str(Path(hdri(EXTERIOR_HDRI)).resolve())))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(500.)
    dome.AddRotateZOp().Set(90.)
    sun = UsdLux.DistantLight.Define(S.stage, "/Room/sun")
    # a hazy afternoon sun: at 1100 the pale floor in the sun's patches is burnt out to white and pale pieces lying
    # there cannot be seen; at 360 the patches are still white and the mug's white pieces cannot be found in them
    sun.CreateIntensityAttr(150.); sun.CreateAngleAttr(0.53); sun.CreateColorAttr(Gf.Vec3f(1., .95, .86))
    S.UsdGeom.Xformable(sun.GetPrim()).AddOrientOp().Set(Gf.Quatf(*_rot_to_wxyz(Rotation.from_euler("xyz", (0., -55., -15.), degrees=True))))


VIEWS = dict(      # fixed diagnostic cameras of this room: (eye, target, focal length mm)
    rail=((1.7, 0.9, 1.5), (3.6, 2.2, 0.9), 20.), open=((0.2, 3.0, 2.0), (2.6, 4.6, 0.0), 20.),
    square=((1.4, 1.9, 1.3), (2.35, 2.95, 0.05), 24.), alcove=((1.6, 7.3, 1.7), (3.2, 9.2, 0.0), 20.),
    table_b1=((2.5, 1.2, 1.5), (0.6, 2.9, 0.2), 20.), desk=((-0.4, 3.0, 1.7), (-2.0, 1.4, 0.1), 20.),
    craft=((-0.5, 5.6, 1.8), (-1.95, 7.3, 0.75), 22.), spill=((-1.0, 6.9, 1.7), (-2.6, 8.4, 0.0), 20.))


# ------------------------------------------------------------------------------ export
def export(world, folder, extra_cameras=None):
    """Export the world's static appearance and posable things to folder/scene.usdc; returns (path, meta)."""
    sc = world.scene
    S = Stage(folder)
    tex = textures(S.folder/"textures")
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    mats["vct"] = S.textured("vct", tex["vct"], rough=.35, normal=tex["vct_n"], rough_tex=tex["vct_r"], normal_strength=.6)
    from whiteboard.visual_assets import asset
    oak = {k: asset("oak_veneer_01", k) for k in ("diff", "rough", "nor_gl")}
    mats["teacher_top"] = S.textured("walnut_pbr", oak["diff"], rough=.45, normal=oak["nor_gl"], rough_tex=oak["rough"], tint=(.38, .26, .18))
    mats["table_top"] = S.textured("maple_pbr", oak["diff"], rough=.45, normal=oak["nor_gl"], rough_tex=oak["rough"], tint=(1.05, 0.98, 0.88))
    mats["brush_wood"] = S.textured("beech_pbr", oak["diff"], rough=.5, normal=oak["nor_gl"], rough_tex=oak["rough"], tint=(1.0, 0.86, 0.66))
    mats["handle_wood"] = mats["brush_wood"]
    V.Looks(S, mats).build()                                         # plastics, steel, paper, glaze and the printed labels
    mats["cork_tex"] = S.textured("cork_tex", tex["cork"], rough=.95)
    mats["ceiling_tex"] = S.textured("ceiling_tex", tex["ceiling2"], rough=.95, normal=tex["ceiling2_n"], normal_strength=.8)
    mats["wall"] = S.textured("cmu", tex["cmu"], rough=.75, normal=tex["cmu_n"], normal_strength=1.)
    mats["wall_front"] = S.textured("cmu_front", tex["cmu"], rough=.75, normal=tex["cmu_n"], tint=(.98, 1.0, 1.02))
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    sup = L.supplies()
    for name, it in sc.items.items():
        base = f"/Lab/body_{it['body']}/{name}"
        if it["kind"] == "supply":
            supply_visual(S, base, world.spec["supplies"][name]["kind"], sc.parts[name], mats)
        elif it["kind"] != "tool":
            furniture_visual(S, base, it["kind"], sc.parts[name], mats, world.spec["glides"])
    for name, t in sc.tools.items():
        hb = f"/Lab/body_{t['handle_body']}/{name}_handle" if "handle_body" in t else None
        tool_visual(S, f"/Lab/body_{t['body']}/{name}", t, mats, hb)
    robot_visual(S, world, mats)
    room(S, mats, tex, world.spec)
    meta = dict(cameras=CR.cameras(S, world))
    views = dict(VIEWS)
    views.update(extra_cameras or {})
    for name, (eye, target, f) in views.items():
        if name in meta["cameras"]:
            raise ValueError(f"extra camera {name!r} would replace the built-in camera of that name; rename it")
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); meta["cameras"][name] = "diagnostic"
    meta["debris"] = debris_visual(S, world, mats)
    meta["bristles"] = bristle_visual(S, world, mats)
    meta["dust"] = dust_visual(S, world)
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta))
    return S.folder/"scene.usdc", meta


def dust_visual(S, world):
    """The fine-dust film (sweeping/dust.py) as a stippled cut-out: a sheet 0.3 mm over the floor, and one on each
    dustpan's floor, of dust-coloured specks 1 mm across. A speck is there where a fixed random number is under
    the cell's cover (dust.picture), so the stipple does not flicker and thins out where a broom has passed. The
    mask is replaced frame by frame (Renderer.observe). A drawing of a reduced-order model: the specks are not
    particles and nothing collides with them."""
    dust = getattr(world, "dust", None)
    if dust is None or dust.view_box is None:
        return None
    from PIL import Image
    T, Sdf, UsdShade = S.Sdf.ValueTypeNames, S.Sdf, S.UsdShade
    folder = S.folder/"dust"
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (150, 140, 124)).save(folder/"colour.png")
    Image.new("L", (8, 8), 0).save(folder/"none.png")

    def look(name):
        m = S.textured(name, folder/"colour.png", rough=.95)
        sh = UsdShade.Shader.Get(S.stage, f"/Looks/{name}/Shader")
        sh.CreateInput("enable_opacity", T.Bool).Set(True)
        sh.CreateInput("enable_opacity_texture", T.Bool).Set(True)
        sh.CreateInput("opacity_texture", T.Asset).Set(Sdf.AssetPath(str((folder/"none.png").resolve())))
        sh.CreateInput("opacity_threshold", T.Float).Set(0.5)
        return m, f"/Looks/{name}/Shader"

    def sheet(path, x0, y0, x1, y1, z, mat):
        S.mesh(path, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)],
               normals=[(0, 0, 1)]*4, mat=mat)
    meta = dict(folder=str(folder), floor=None, pans={})
    m, shader = look("dust_floor")
    x0, y0, x1, y1 = dust.view_box
    sheet("/Room/dust_floor", x0, y0, x1, y1, 0.0003, m)
    meta["floor"] = dict(shader=shader)
    for name, p in dust.pans.items():
        t = world.scene.tools[name]
        m, shader = look(f"dust_{name}")
        sheet(f"/Lab/body_{t['body']}/{name}_dust", -p["width"]/2, -p["depth"]/2, p["width"]/2, p["depth"]/2, p["tray"] + 0.0004, m)
        meta["pans"][name] = dict(shader=shader)
    return meta


def frame(world):
    """What one rendered frame needs from the world: body poses, piece poses, bristle nodes (float32), and the
    fine-dust film's cover if the start state has any."""
    d = world.debris()
    rods, _ = world.bristles()
    dust = getattr(world, "dust", None)
    return dict(body_q=world.state.body_q.numpy().astype(np.float32),
                debris=np.concatenate([d["pos"], d["quat"]], 1).astype(np.float32) if len(d["pos"]) else np.zeros((0, 7), np.float32),
                bristles=rods.astype(np.float32), dust=None if dust is None else dust.picture())


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
        env = dict(os.environ, SWP_RENDER_PORT=str(listener.getsockname()[1]), SWP_RENDER_SECRET=secret)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "sweeping.render_worker", "--scene", str(Path(scene).resolve()),
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

    def observe(self, cameras, fr, tick, resolution=None, samples=None):
        """fr: render.frame(world) or a recorded frame of the same shape."""
        enc = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(fr["body_q"], float).tolist(), cameras=list(cameras),
                   debris=enc(fr["debris"]), bristles=enc(fr["bristles"]))
        if resolution:
            msg["resolution"] = list(resolution)
        if samples:
            msg["spp"] = int(samples)
        masks = self._dust_masks(fr.get("dust"))
        if masks:
            msg["dust"] = masks
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def _dust_masks(self, picture):
        """Stipple masks for the dust sheets whose cover has changed since the last frame: [[shader path, file]]."""
        meta_file = self.folder/"export.json"
        if picture is None or not meta_file.is_file():
            return []
        if not hasattr(self, "_dust"):
            meta = json.loads(meta_file.read_text()).get("dust")
            self._dust = dict(meta=meta, last={}, noise={}, count=0, files={})
        st = self._dust
        if st["meta"] is None:
            return []
        from PIL import Image
        jobs = []
        if picture.get("floor") is not None and st["meta"]["floor"]:
            jobs.append(("floor", st["meta"]["floor"]["shader"], np.asarray(picture["floor"]), 4))
        for name, cover in (picture.get("pans") or {}).items():
            if name in st["meta"]["pans"]:
                jobs.append((name, st["meta"]["pans"][name]["shader"], np.asarray(cover), 2))
        out = []
        for key, shader, cover, up in jobs:
            tag = hash(cover.tobytes())
            if st["last"].get(key) == tag:
                continue
            st["last"][key] = tag
            big = np.repeat(np.repeat(cover, up, 0), up, 1)                   # 1 mm specks
            if key not in st["noise"]:
                st["noise"][key] = np.random.default_rng(len(st["noise"]) + 11).integers(1, 256, big.shape, dtype=np.uint16)
            mask = (st["noise"][key] <= big).astype(np.uint8)*255             # cover 0: no speck; 255: every speck
            st["count"] += 1
            f = Path(st["meta"]["folder"])/f"{key}_{st['count']:06d}.png"
            Image.fromarray(np.ascontiguousarray(mask.T[::-1])).save(f)        # image rows run down from the far (+y) edge
            old = st["files"].get(key, [])
            for stale in old[:-2]:
                stale.unlink(missing_ok=True)
            st["files"][key] = old[-2:] + [f]
            out.append([shader, str(f.resolve())])
        return out

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

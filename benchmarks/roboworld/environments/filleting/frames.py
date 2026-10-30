"""Physics world state -> render frame (the format documented in filleting/render.py). Read-only: nothing here writes
to the world except the render cache attribute `world._render_cache` (static data computed once).

What is drawn from where (world.py / workpiece.py / assembly.py contracts):
  robot links     Newton bodies (world.state.body_q) + link0 and hand from the locked URDF (render.fr3_links_from_newton)
  knife           MuJoCo handle body "knife_handle" (origin at the handle centre, x toward the blade, z spine side);
                  the BladeSpec frame is handle * T(HANDLE_L/2, 0, -H); the five flexible blade segments from the
                  tissue solver (workpiece.blade_segment_poses), drawn as the knife's ":seg<k>" parts
  inserts         the right hand's knife-handle inserts (scene.finger_boxes, role "insert") on their finger bodies
  bones           tissue-solver rigid bodies n_knife_bodies + j (j = 0 head, then backbone segments); the body frame at
                  rest is world-aligned at the body COM (assembly: com = to_world(rigid_bodies[j]["com"])); backbone
                  segments are bone capsules (assembly["capsules"][j], world rest coordinates)
  head            skin-textured surface of rigid_bodies[0]["sdf"] (marching tets, 0.8 mm), UV from the anatomy's
                  skin_uv (the eye is painted in the anatomy's skin texture); the posterior face bonded to the flesh
                  (within 0.8 mm of the tissue sdf) gets a flesh-coloured material
  tissue          boundary faces of the active tets (faces used by exactly one active tet), wound outward; kinds:
                  0 skin (a skin-membrane (tet, face)), 2 original-mesh boundary that is not skin (cavity lining,
                  backbone channel, head face), 1 every other boundary face (knife cuts, interface splits)
  rods            assembly["rods"] node chains: current positions, radius ts.hradius
  fins            anatomy fin outlines (workpiece coordinates), triangulated with interior points (the frame also
                  carries the rest mesh, each fin's vertex range and name for render.fin_fields); each fin moves
                  rigidly with the best-fit rotation + translation (Kabsch) of the tissue nodes nearest its attachment
                  (median fins) or its whole area (paired fins lying on the flank). RENDER-ONLY DRAPE (disclosed): fins are not
                  in the physics; a median fin standing out from a fish lying on its side is lowered so its free edge
                  rests on the board (a dead fish's fins droop), smoothly from its base.

frame = world_frame(world, badge="ASSISTED")      # < 150 ms after the first call (measured in video.py __main__)
"""
import math
import time
import zlib

import numpy as np

from . import layout as LY

FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])     # local face j is opposite vertex j (tissue.py)
HEAD_H = .8e-3            # m, head surface grid
FIN_H = 2.5e-3            # m, fin interior point spacing
FIN_DRAPE_CLEAR = .6e-3   # m, draped fin edge height above the board
FIN_ANCHOR_R = 10e-3      # m, tissue nodes within this distance of a fin's attachment carry the fin rigidly
HEAD_RIM = 2.5e-3         # m, band of the head's bonded face next to its exterior skin drawn as skin (render-only)


def _face_keys(tet_ids, face_j):
    return np.asarray(tet_ids, np.int64)*4 + np.asarray(face_j, np.int64)


def boundary_faces(tets, active):
    """(tet ids, local face j) of the faces used by exactly one active tet."""
    ids = np.nonzero(active)[0]
    T = tets[ids]
    F = T[:, FACES].reshape(-1, 3).astype(np.int64)
    s = np.sort(F, 1)
    key = (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    grp = np.cumsum(first) - 1
    cnt = np.bincount(grp)
    single = order[cnt[grp] == 1]
    return ids[single//4], single % 4


# ------------------------------------------------------------------------------------------------ static cache
def _triangulate_polygon_with_interior(P2, h):
    """Delaunay triangulation of a simple polygon (2D) with interior grid points, triangles outside dropped."""
    from matplotlib.path import Path as MPath
    from scipy.spatial import Delaunay
    P2 = np.asarray(P2, float)
    # densify the boundary to spacing h
    pts = []
    for a, b in zip(P2, np.roll(P2, -1, 0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a)/h)))
        pts += [a + (b - a)*t for t in np.arange(n)/n]
    B = np.array(pts)
    lo, hi = B.min(0), B.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0] + h/2, hi[0], h), np.arange(lo[1] + h/2, hi[1], h))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    path = MPath(B)
    G = G[path.contains_points(G, radius=-.35*h)] if len(G) else G
    V = np.vstack([B, G]) if len(G) else B
    tri = Delaunay(V).simplices
    cen = V[tri].mean(1)
    tri = tri[path.contains_points(cen)]
    return V, tri


def _fin_meshes(world, cache):
    """Rest geometry of the fins in world coordinates + nearest rest node per vertex + UV."""
    from scipy.spatial import cKDTree
    wpc, asm, ts = world.workpiece, world.assembly, world.tissue
    to_w = asm["to_world"]
    fins = getattr(wpc, "fins", None) or []
    if not fins:
        return None
    Vs, Fs, UVs, off, spans = [], [], [], 0, []
    board = LY.BOARD_TOP_Z + FIN_DRAPE_CLEAR
    for f in fins:
        O = np.asarray(f["outline"], float)                       # workpiece coordinates
        A = np.asarray(f["attach"], float)
        c = O.mean(0)
        _, _, Vt = np.linalg.svd(O - c)
        e1, e2 = Vt[0], Vt[1]
        P2 = np.column_stack([(O - c) @ e1, (O - c) @ e2])
        V2, tri = _triangulate_polygon_with_interior(P2, FIN_H)
        V = c + V2[:, :1]*e1 + V2[:, 1:2]*e2
        # UV: u along the attachment line, v = distance from it (rays run along v in the fin texture)
        a0, a1 = A[0], A[-1]
        base = a1 - a0
        bl = max(np.linalg.norm(base), 1e-9)
        bdir = base/bl
        rel = V - a0
        u = np.clip((rel @ bdir)/bl, -.2, 1.2)
        dist = np.linalg.norm(rel - np.outer(rel @ bdir, bdir), axis=1)
        v = dist/max(dist.max(), 1e-9)
        W = to_w(V)
        # render-only drape of fins standing out in the median plane (see module docstring)
        if f.get("plane", "median") == "median":
            Wa = to_w(A)
            zb = float(Wa[:, 2].mean())
            dmax = max(dist.max(), 1e-9)
            t = np.clip(dist/dmax, 0, 1)
            drop = (zb - board)*(3*t**2 - 2*t**3)*np.clip((zb - board)/1e-3, 0, 1)
            W[:, 2] = np.maximum(W[:, 2] - drop, board)
        anchor_pts = to_w(A) if f.get("plane", "median") == "median" else W
        Vs.append(W); Fs.append(tri + off); UVs.append(np.column_stack([u, v]))
        spans.append((off, off + len(W), anchor_pts)); off += len(W)
    W = np.vstack(Vs)
    Xr = ts.hX[:ts.n]
    kinds = getattr(ts, "hkind", None)
    cand = np.nonzero(kinds[:ts.n] != 2)[0] if kinds is not None else np.arange(ts.n)   # tissue nodes, not rods
    tree = cKDTree(Xr[cand])
    groups = []
    for a, b, P in spans:
        # all tissue nodes within FIN_ANCHOR_R of the attachment (a well-conditioned 3D cloud: nodes nearest a straight
        # attachment line alone are nearly collinear and leave the rotation about that line undetermined)
        near = tree.query_ball_point(P, FIN_ANCHOR_R)
        nodes = np.unique(np.concatenate([np.asarray(n_, np.int64) for n_ in near] + [np.zeros(0, np.int64)]))
        if len(nodes) < 12:
            _, nn = tree.query(P, k=12)
            nodes = np.unique(np.asarray(nn).ravel())
        groups.append(np.array([a, b] + cand[nodes].tolist(), np.int64))
    return dict(rest=W, tris=np.vstack(Fs).astype(np.int32), uv=np.vstack(UVs).astype(np.float32), groups=groups,
                names=[str(f.get("name", "fin")) for f in fins])


def fin_points(fins, x, X_rest):
    """Current fin vertices: each fin (group [v0, v1, anchor nodes...]) moved by the Kabsch fit of its anchors."""
    P = np.asarray(fins["rest"], np.float64).copy()
    for g in fins["groups"]:
        a, b, nodes = int(g[0]), int(g[1]), np.asarray(g[2:], np.int64)
        A0 = np.asarray(X_rest[nodes], np.float64)
        A1 = np.asarray(x[nodes], np.float64)
        c0, c1 = A0.mean(0), A1.mean(0)
        H = (A0 - c0).T @ (A1 - c1)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        P[a:b] = (P[a:b] - c0) @ R.T + c1
    # fins are render-only membranes resting on whatever supports them: never below the board surface (a draped fin
    # tilted by a few degrees with its body otherwise sinks its lobes into the board)
    P[:, 2] = np.maximum(P[:, 2], LY.BOARD_TOP_Z + FIN_DRAPE_CLEAR)
    return P.astype(np.float32)


def _head_mesh(world):
    """Skin-textured head surface in the head body's rest frame (world-aligned at its COM)."""
    from .anatomy.common import marching_tets
    wpc, asm = world.workpiece, world.assembly
    if not wpc.rigid_bodies:
        return None                     # a workpiece without a head (fillet, salmon side, tenderloin)
    rb = wpc.rigid_bodies[0]
    sdf = rb.get("sdf")
    if sdf is None:
        return None
    caps = np.array([np.concatenate([p0, p1]) for p0, p1, _ in rb["capsules"]]).reshape(-1, 3)
    lo_b, hi_b = (np.asarray(b, float) for b in wpc.bounds)
    lo = np.minimum(caps.min(0) - .02, lo_b)
    hi = np.maximum(caps.max(0) + .02, hi_b)
    lo[0] = caps[:, 0].min() - .03
    hi[0] = caps[:, 0].max() + .02
    V, F = marching_tets(sdf, lo, hi, HEAD_H)
    if not len(F):
        return None
    uv = np.asarray(wpc.appearance["skin_uv"](V), np.float32)
    # posterior face bonded to the flesh: on the tissue boundary AND facing into the tissue (the head's outward
    # normal opposite to the tissue's outward normal); exterior head skin next to the seam faces the same way as the
    # body skin and stays skin
    cen = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    eps = 2e-4
    g = np.stack([(np.asarray(wpc.sdf(cen + e)) - np.asarray(wpc.sdf(cen - e)))/(2*eps) for e in np.eye(3)*eps], 1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-20)
    face = (np.abs(np.asarray(wpc.sdf(cen))) < HEAD_H) & (np.einsum("ij,ij->i", fn, g) < -.3)
    # the coarse tissue surface (3.5 mm tets) does not follow the head's 0.8 mm rim exactly: a band of the bonded face
    # next to the exterior skin would show as a red seam between skin and skin; render that band as skin
    from scipy.spatial import cKDTree
    skin_v = np.unique(F[~face].ravel())
    if len(skin_v) and face.any():
        dd, _ = cKDTree(V[skin_v]).query(cen[face])
        fi = np.nonzero(face)[0]
        face[fi[dd < HEAD_RIM]] = False
    com_w = asm["to_world"](np.asarray(rb["com"])[None])[0]
    Vw = asm["to_world"](V) - com_w
    # per-corner UV with the theta seam (v) unwrapped per triangle
    uvc = uv[F].copy()
    span = uvc[:, :, 1].max(1) - uvc[:, :, 1].min(1)
    wrap = span > .5
    uvc[wrap, :, 1] = np.where(uvc[wrap, :, 1] < .5, uvc[wrap, :, 1] + 1., uvc[wrap, :, 1])
    parts = [dict(points=Vw, tris=F[~face], uv=uvc[~face].reshape(-1, 2), material="skin"),
             dict(points=Vw, tris=F[face], material="flesh_face")]
    return dict(parts=parts, com_rest=com_w, triangles=int(len(F)), face_triangles=int(face.sum()))


def static_cache(world):
    cache = getattr(world, "_render_cache", None)
    if cache is not None:
        return cache
    t0 = time.perf_counter()
    ts, asm, wpc, sc = world.tissue, world.assembly, world.workpiece, world.scene
    c = dict()
    # tissue kinds
    orig = np.asarray(world.cut.orig)
    ob_t, ob_j = boundary_faces(orig, np.ones(len(orig), bool))
    c["orig_bnd"] = np.sort(_face_keys(ob_t, ob_j))
    if getattr(ts, "n_tris", 0):
        c["skin_keys"] = np.sort(_face_keys(ts.mem_tet, ts.mem_face))
    else:
        c["skin_keys"] = np.zeros(0, np.int64)
    c["topo_sig"] = None
    # node appearance (grown as nodes are added by cuts; a copy has its twin's rest position)
    c["n_app"] = 0
    c["rest_local"] = np.zeros((0, 3), np.float32)
    c["uv"] = np.zeros((0, 2), np.float32)
    c["color"] = np.zeros((0, 3), np.float32)
    tex = wpc.appearance.get("skin_texture") if wpc is not None else None
    if isinstance(tex, dict) and not tex.get("path") and callable(tex.get("make")):
        # the anatomy generates its texture on demand: write it once next to the render cache
        import os
        from .anatomy.common import save_png
        path = f"/tmp/filleting_render_cache/{wpc.name}_skin_seed{world.config.seed}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            save_png(path, tex["make"]())
        tex = dict(tex, path=path)
    c["skin_texture"] = tex.get("path") if isinstance(tex, dict) else None
    # rigid bones (none for the coupon, which has no anatomy)
    nk = asm["n_knife_bodies"] if asm is not None else 0
    c["bones"] = []
    for j, rb in enumerate(wpc.rigid_bodies if asm is not None else []):
        com_w = asm["to_world"](np.asarray(rb["com"])[None])[0]
        caps = [(np.asarray(p0) - com_w, np.asarray(p1) - com_w, float(r)) for p0, p1, r in asm["capsules"][j]]
        c["bones"].append(dict(name=rb["name"], body=nk + j, com_rest=com_w, capsules=caps))
    st0 = ts.body_states()
    c["bone_rest_check_mm"] = float(max(np.linalg.norm(st0["p"][b["body"]] - b["com_rest"]) for b in c["bones"]))*1e3 \
        if world.tick == 0 and c["bones"] else None
    c["head"] = _head_mesh(world) if asm is not None else None
    # rods
    c["rods"] = [dict(name=r["name"], kind=r["kind"], nodes=np.asarray(r["nodes"], np.int64))
                 for r in (asm["rods"] if asm is not None else [])]
    # fins
    c["fins"] = _fin_meshes(world, c) if asm is not None else None
    # right-hand inserts (MuJoCo boxes on the finger bodies)
    ins = {}
    for arm, body, name, p, q, half, role in sc.finger_boxes:
        if role == "insert":
            ins.setdefault(int(body), []).append((np.asarray(p, float), np.asarray(q, float), np.asarray(half, float)))
    c["inserts"] = ins
    c["body_label"] = list(world.model.body_label)
    c["setup_s"] = time.perf_counter() - t0
    world._render_cache = c
    return c


# ------------------------------------------------------------------------------------------------ per frame
def _appearance(world, c, n):
    """Extend the per-node appearance arrays to n nodes (only new nodes are evaluated)."""
    if n <= c["n_app"]:
        return
    from .render import tissue_appearance
    ts, asm = world.tissue, world.assembly
    new = np.arange(c["n_app"], n)
    if asm is None:
        # the coupon (a plain fish-muscle block, no anatomy): rest positions as they are, no skin texture, one flesh
        # colour (the anatomies' raw white-muscle tone)
        rest_l = np.asarray(ts.hX[new], np.float32)
        c["rest_local"] = np.vstack([c["rest_local"], rest_l])
        c["uv"] = np.vstack([c["uv"], np.zeros((len(new), 2), np.float32)])
        c["color"] = np.vstack([c["color"], np.tile(np.array([[.86, .62, .55]], np.float32), (len(new), 1))])
        c["n_app"] = n
        return
    rest_l = np.asarray(asm["to_local"](ts.hX[new]), np.float32)
    app = tissue_appearance(world.workpiece, rest_l)
    c["rest_local"] = np.vstack([c["rest_local"], rest_l])
    c["uv"] = np.vstack([c["uv"], app["uv"]])
    c["color"] = np.vstack([c["color"], np.asarray(app["flesh_color"], np.float32)])
    c["n_app"] = n


def _topology(world, c):
    """Boundary faces (tet, face) and their kinds, recomputed only when the mesh connectivity changed."""
    ts = world.tissue
    nt = ts.nt
    sig = (ts.n, nt, zlib.crc32(ts.htets[:nt].tobytes()), zlib.crc32(ts.hactive[:nt].tobytes()))
    if sig == c["topo_sig"]:
        return c["topo"]
    bt, bj = boundary_faces(ts.htets[:nt], ts.hactive[:nt] > 0)
    keys = _face_keys(bt, bj)
    kind = np.ones(len(keys), np.uint8)
    sk, ob = c["skin_keys"], c["orig_bnd"]
    if len(sk):
        i = np.clip(np.searchsorted(sk, keys), 0, len(sk) - 1)
        is_skin = sk[i] == keys
    else:
        is_skin = np.zeros(len(keys), bool)
    i = np.clip(np.searchsorted(ob, keys), 0, len(ob) - 1)
    is_orig = ob[i] == keys
    kind[is_orig] = 2
    kind[is_skin] = 0
    tris = ts.htets[bt[:, None], FACES[bj]].astype(np.int64)
    opp = ts.htets[bt, bj].astype(np.int64)
    c["topo_sig"] = sig
    c["topo"] = dict(tris=tris, opp=opp, kind=kind, id=(c.get("topo", {}).get("id", -1) + 1))
    return c["topo"]


def _pose(p, R):
    from scipy.spatial.transform import Rotation
    return tuple(map(float, p)), tuple(map(float, Rotation.from_matrix(R).as_quat()))


def world_frame(world, badge="ASSISTED", timings=None):
    """Render frame of the world's current state (see module docstring)."""
    from .render import fr3_links_from_newton
    from .scene import HANDLE_L
    from .workpiece import blade_segment_poses
    from scipy.spatial.transform import Rotation
    t0 = time.perf_counter()
    c = static_cache(world)
    t1 = time.perf_counter()
    ts = world.tissue
    body_q = world.state.body_q.numpy().astype(float)
    links = fr3_links_from_newton(c["body_label"], body_q, LY.ROBOT_BASES)
    bodies = {}
    # knife: handle (BladeSpec frame) + flexible blade segments
    ph, Rh = world.body_pose("knife_handle")
    blade = world.scene.blade
    pos, q = _pose(ph + Rh @ np.array([HANDLE_L/2, 0., -blade.H]), Rh)
    bodies["knife_fillet"] = dict(mesh="knife:fillet:handle", pos=pos, quat_xyzw=q)
    for k, (R, p) in enumerate(blade_segment_poses(world)):
        pos, q = _pose(p, R)
        bodies[f"knife_fillet_seg{k}"] = dict(mesh=f"knife:fillet:seg{k}", pos=pos, quat_xyzw=q)
    # inserts on the right fingers
    for body, boxes in c["inserts"].items():
        bodies[f"insert_{c['body_label'][body].split('/')[-1]}"] = dict(
            mesh=dict(boxes=boxes, material="insert"), pos=tuple(body_q[body, :3]), quat_xyzw=tuple(body_q[body, 3:]))
    # anatomical rigid bodies
    st = ts.body_states()
    for j, b in enumerate(c["bones"]):
        p, qb = st["p"][b["body"]], st["q"][b["body"]]
        if j == 0 and c["head"] is not None:
            bodies["head"] = dict(mesh=dict(parts=c["head"]["parts"]), pos=tuple(p), quat_xyzw=tuple(qb))
        else:
            bodies[f"bone_{b['name']}"] = dict(mesh=dict(capsules=b["capsules"], material="bone"), pos=tuple(p),
                                               quat_xyzw=tuple(qb))
    t2 = time.perf_counter()
    # tissue
    n = ts.n
    x = ts.x.numpy()[:n].astype(np.float32)
    _appearance(world, c, n)
    topo = _topology(world, c)
    tris = topo["tris"]
    # wind every face outward (away from its tet's opposite vertex) at the current positions
    a, b_, cc = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    nrm = np.cross(b_ - a, cc - a)
    flip = np.einsum("ij,ij->i", nrm, a - x[topo["opp"]]) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    tissue = dict(points=x, tris=tris.astype(np.int32), tri_kind=topo["kind"], rest=c["rest_local"][:n],
                  uv=c["uv"][:n], flesh_color=c["color"][:n], skin_texture=c["skin_texture"], topology_id=topo["id"])
    t3 = time.perf_counter()
    rods = [dict(points=x[r["nodes"]], radius=ts.hradius[r["nodes"]]) for r in c["rods"]]
    fins = None
    if c["fins"] is not None:
        f = c["fins"]
        fins = dict(points=fin_points(f, x, ts.hX), tris=f["tris"], uv=f["uv"], rest=f["rest"],
                    groups=[(int(g[0]), int(g[1])) for g in f["groups"]], names=f.get("names"))
    t4 = time.perf_counter()
    frame = dict(tick=int(world.tick), t=float(world.sim_time), badge=str(badge), robot_links=links, bodies=bodies,
                 tissue=tissue, rods=rods, fins=fins, task=world.config.task)
    if timings is not None:
        timings.append(dict(cache_s=t1 - t0, rigid_s=t2 - t1, tissue_s=t3 - t2, rods_fins_s=t4 - t3, total_s=t4 - t0))
    return frame

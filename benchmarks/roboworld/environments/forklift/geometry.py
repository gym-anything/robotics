"""Forklift C asset geometry in the truck frame, and everything derived from it.

The NVIDIA Isaac Sim 5.1 "Forklift C" USD (forklift/assets) is the truck's visual and the source of its collision
geometry ("collision = visual"). This module reads every mesh with its world transform, moves it into
the TRUCK FRAME (x forward, y left, z up; origin on the ground under the centre of the front axle; the asset's front
wheel centres are at x = +0.266 m in its own frame) and caches the result keyed by the USD's SHA-256.

Derived here (and checked by E0 and the collision-overlay renders):
- part meshes for the moving bodies (wheels, rear knuckles, mast stages, carriage, forks, steering wheel, levers);
- regions cut out of the merged body mesh (Body01): pedals, seat cushion and backrest, so they can move;
- convex hulls of authored regions of the chassis for MuJoCo collision.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .assets.install import resolve, cache_dir
from . import params as P

USD_KEY = "Isaac/Robots/IsaacSim/ForkliftC/forklift_c.usd"
ROOT = "/SM_Forklift_C01_01"
FRONT_AXLE_X_ASSET = 0.266         # measured: front wheel mesh centres x = 0.263 / 0.269 m


def usd_path():
    return resolve(USD_KEY)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(path):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(path))
    out = {}
    for p in st.Traverse():
        if p.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
        X = np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64)
        w = (np.c_[pts, np.ones(len(pts))] @ X)[:, :3]
        w[:, 0] -= FRONT_AXLE_X_ASSET
        # triangulate fans
        tris = []
        o = 0
        for c in counts:
            for k in range(1, c - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += c
        name = str(p.GetPath())[len(ROOT) + 1:]
        out[name] = (w.astype(np.float32), np.asarray(tris, np.int32))
    return out


def parts():
    """{mesh path below the root: (vertices (n,3) truck frame, triangles (m,3))}, cached."""
    path = usd_path()
    sha = _sha(path)
    cache = cache_dir() / "derived" / f"forklift_c_parts_{sha[:16]}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        names = json.loads(str(z["names"]))
        return {n: (z[f"v{i}"], z[f"f{i}"]) for i, n in enumerate(names)}
    data = _extract(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"names": json.dumps(list(data))}
    for i, (n, (v, f)) in enumerate(data.items()):
        arrays[f"v{i}"], arrays[f"f{i}"] = v, f
    np.savez_compressed(cache, **arrays)
    return data


def bounds(v):
    return np.asarray(v).min(0), np.asarray(v).max(0)



# ------------------------------------------------------------------------------------------------ parts of the truck
BODY01 = "body/SM_Forklift_C01_Body01_01"

# Regions cut out of the merged body mesh: every connected component (vertices welded at 0.1 mm) whose bounding box lies
# inside the box (truck frame, metres, padded 10 mm) belongs to the region. Boxes chosen from component listings and
# identification renders. A region given as dict(parts={name: box}, pad=...) is
# the union of tight per-part boxes (each part becomes its own mesh entry "Body01:<region>/<part>").
#
# Parking-brake lever. The lever that moves is the handle the operator holds: the rubber grip
# (component of 1,194 triangles, x -0.410..-0.321, y 0.296..0.321, z 1.076..1.189), its end cap (168, at the grip's top)
# and the 10 mm flat steel bar that carries the grip down through the slot in the cowl's roof (40, x -0.357..-0.284,
# z 1.025..1.105). The strip of the cowl's own lower skin below it (1,430 triangles; in x-z sections it continues the
# cowl's contour, y 0.245..0.344, from (-0.36, 1.00) to (-0.17, 0.87)) stays on the chassis.
# The handle's pivot is the 9 mm cross pin through the bar's lower end (its own component, x -0.303..-0.294, y
# 0.297..0.320, z 1.037..1.046; centre below). The slot in the roof decides the travel: swept about this pin against the
# chassis mesh, the handle is free from the asset's pose to +60 deg (forward, crossing only the hidden linkage's pins) and
# meets the slot's rear end between -20 and -30 deg. The asset is therefore drawn with the lever PULLED BACK (applied:
# pull to apply, push forward to release, as on counterbalance trucks' over-centre hand levers), and the released pose
# lies forward. Joint coordinate unchanged: released 0, applied -PARK_LEVER travel; the handle's meshes are turned to
# the released pose (+travel about the pin) so that at the applied stop they sit exactly where the asset draws them.
PARK_PIN = (-0.2984, 0.3082, 1.0418)
PARK_GRIP_BOX = ((-0.411, -0.320), (0.295, 0.322), (1.075, 1.190))       # grip + end cap
PARK_BAR_BOX = ((-0.358, -0.283), (0.302, 0.314), (1.024, 1.106))        # flat bar into the slot
REGIONS = {
    "pedal_inch": ((-0.32, -0.09), (0.10, 0.18), (0.73, 0.88)),
    "pedal_brake": ((-0.32, -0.09), (-0.03, 0.05), (0.75, 0.88)),
    "pedal_accel": ((-0.31, -0.14), (-0.175, -0.095), (0.67, 0.85)),
    "dir_lever": ((-0.45, -0.36), (0.17, 0.30), (1.26, 1.34)),
    "park_lever": dict(parts=dict(grip=PARK_GRIP_BOX, bar=PARK_BAR_BOX), pad=0.001),
    "key": ((-0.345, -0.325), (-0.068, -0.040), (1.018, 1.048)),
    "seat": ((-1.33, -0.70), (-0.32, 0.38), (1.025, 1.70)),
    "axle": ((-1.80, -1.61), (-0.45, 0.45), (0.15, 0.37)),
}

# Separate asset meshes -> simulated body (prefix match on the mesh name below the asset root).
MESH_BODY = {
    "left_front_wheel/": "wheel_fl", "right_front_wheel/": "wheel_fr",
    "left_back_wheel/": "wheel_rl", "right_back_wheel/": "wheel_rr",
    "left_rotator/": "knuckle_l", "right_rotator/": "knuckle_r",
    "body/SM_Forklift_C01_RearAxle": "axle",
    "body/SM_Forklift_C01_MastStage02": "mast_outer", "body/SM_Forklift_C01_MastStageWheel01": "mast_outer",
    "body/SM_Forklift_C01_MastStageWheel02": "mast_outer", "body/SM_Forklift_C01_Hose": "mast_outer",
    "body/SM_Forkliftt_C01_Forcer": "mast_outer", "body/SM_Forklift_C01_MastChain01": "mast_int",
    "body/SM_Forklift_C01_Decals02": "mast_outer",
    "body/SM_Forklift_C01_MastStage01": "mast_int", "body/SM_Forklift_C01_MastStageWheel03": "mast_int",
    "body/SM_Forklift_C01_MastStageWheel04": "mast_int", "body/SM_Forklift_C01_MastStageWheel07": "mast_int",
    "body/SM_Forklift_C01_MastStageWheel09": "mast_int",
    "body/SM_Forklift_C01_MastStage03": "mast_inner", "body/SM_Forklift_C01_MastForcerBig": "mast_inner",
    "body/SM_Forklift_C01_MastChain02": "mast_inner", "body/SM_Forklift_C01_MastStageWheel05": "mast_inner",
    "body/SM_Forklift_C01_MastStageWheel06": "mast_inner", "body/SM_Forklift_C01_MastStageWheel08": "mast_inner",
    "body/SM_Forklift_C01_MastStageWheel10": "mast_inner", "body/SM_Forklift_C01_MastStageWheel11": "mast_inner",
    "lift/SM_Forklift_C01_MastStage04": "carriage",
    "lift/SM_Forklift_C01_MastStage05": "sideshift", "lift/SM_Forklift_C01_Fork": "sideshift",
    "body/SM_Forklift_C01_GearStick01": "lever_lift", "body/SM_Forklift_C01_GearStick02": "lever_tilt",
    "body/SM_Forklift_C01_GearStick03": "lever_shift",
    "body/SM_Forklift_C01_OperatorChair": "seat",
}


def _components(v, f):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    q = np.round(np.asarray(v, np.float64) * 1e4).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    ff = inv[f]
    n = int(inv.max()) + 1
    rows = np.r_[ff[:, 0], ff[:, 1], ff[:, 2]]
    cols = np.r_[ff[:, 1], ff[:, 2], ff[:, 0]]
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    return lab[ff[:, 0]]


def _inside(lo, hi, box, pad=0.01):
    blo = np.array([b[0] for b in box]) - pad
    bhi = np.array([b[1] for b in box]) + pad
    return bool((lo >= blo).all() and (hi <= bhi).all())


def _region_parts():
    """[(region, part or None, box, pad)] in REGIONS order: the ownership tests of the Body01 split."""
    out = []
    for n, spec in REGIONS.items():
        if isinstance(spec, dict):
            out += [(n, part, box, spec.get("pad", 0.01)) for part, box in spec["parts"].items()]
        else:
            out.append((n, None, spec, 0.01))
    return out


def _split_body01(v, f):
    """Owner label per Body01 triangle: "<region>" or "<region>/<part>" for the region it belongs to, "" for the chassis."""
    tl = _components(v, f)
    owner = np.full(len(f), "", dtype=object)
    tests = _region_parts()
    order = np.argsort(tl, kind="stable")
    bounds = np.searchsorted(tl[order], np.arange(int(tl.max()) + 2))
    for c in range(int(tl.max()) + 1):
        t = order[bounds[c]:bounds[c + 1]]
        if not len(t):
            continue
        pts = v[f[t]].reshape(-1, 3)
        lo, hi = pts.min(0), pts.max(0)
        for n, part, box, pad in tests:
            if _inside(lo, hi, box, pad):
                owner[t] = n if part is None else f"{n}/{part}"
                break
    return owner


def park_released(V):
    """Asset-pose (applied) parking-lever vertices turned to the joint's zero, the released pose: +travel about PARK_PIN
    (the joint axis is +y; positive rotation carries the handle forward)."""
    a = P.PARK_LEVER["travel"]
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    piv = np.asarray(PARK_PIN)
    return ((np.asarray(V, np.float64) - piv) @ R.T + piv).astype(np.float32)


def body_meshes():
    """{body name: [(mesh label, V, F)]} for every simulated body of the truck, in the truck frame, at every joint's zero
    (the parking-brake handle turned to its released pose, see PARK_PIN), with the truck's configuration applied
    (overhead guard raised by params.GUARD_RAISE)."""
    out = dict(_body_meshes_raw())
    out["park_lever"] = [(label, park_released(V), F) for label, V, F in out["park_lever"]]
    if P.GUARD_RAISE:
        cfg = {}
        for body, entries in out.items():
            if body != "chassis":
                cfg[body] = entries
                continue
            new = []
            for label, V, F in entries:
                V = V.copy()
                V[V[:, 2] > P.GUARD_SPLIT, 2] += P.GUARD_RAISE
                new.append((label, V, F))
            cfg[body] = new
        out = cfg
    return out


def _body_meshes_raw():
    """{body name: [(mesh label, V, F)]} for every simulated body of the truck, in the truck frame, as in the asset."""
    path = usd_path()
    cache = cache_dir() / "derived" / f"forklift_c_bodies_{_sha(path)[:16]}_{_regions_sig()}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        index = json.loads(str(z["index"]))
        out = {}
        for i, (body, label) in enumerate(index):
            out.setdefault(body, []).append((label, z[f"v{i}"], z[f"f{i}"]))
        return out
    P = parts()
    out = {}
    v, f = P[BODY01]
    owner = _split_body01(v, f)
    for lab in (o for o in np.unique(owner) if o):
        out.setdefault(lab.split("/")[0], []).append(("Body01:" + lab, v, f[owner == lab]))
    out.setdefault("chassis", []).append(("Body01:rest", v, f[owner == ""]))
    for name, (vv, ff) in P.items():
        if name == BODY01:
            continue
        body = next((b for prefix, b in MESH_BODY.items() if name.startswith(prefix)), "chassis")
        if name == "body/SM_Forklift_C01_SteeringWheel_01":
            st = _components(vv, ff)
            sizes = np.bincount(st)
            order = np.argsort(-sizes)           # rim+spokes (largest), hub, knob (measured: 4500 / 2038 / 792 triangles)
            out.setdefault("steering_wheel", []).append(("SteeringWheel:rim", vv, ff[st == order[0]]))
            out.setdefault("steering_wheel", []).append(("SteeringWheel:hub", vv, ff[st == order[1]]))
            out.setdefault("knob", []).append(("SteeringWheel:knob", vv, ff[st == order[2]]))
            continue
        out.setdefault(body, []).append((name, vv, ff))
    # compact: keep only referenced vertices per entry
    compact = {}
    index, arrays = [], {}
    for body, entries in out.items():
        for label, vv, ff in entries:
            used, inv = np.unique(ff.ravel(), return_inverse=True)
            V, F = vv[used].astype(np.float32), inv.reshape(-1, 3).astype(np.int32)
            compact.setdefault(body, []).append((label, V, F))
            arrays[f"v{len(index)}"], arrays[f"f{len(index)}"] = V, F
            index.append((body, label))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, index=json.dumps(index), **arrays)
    return compact


def _regions_sig():
    return hashlib.sha256(json.dumps([REGIONS, MESH_BODY], sort_keys=True).encode()).hexdigest()[:8]


def body_vertices(meshes, body):
    return np.concatenate([V for _, V, _ in meshes[body]], 0)


# ------------------------------------------------------------------------------------------------ collision geometry
# Chassis collision = convex hulls of the chassis vertices inside authored boxes (truck frame). The boxes leave the cab
# open (operator's legs, footwell, arm space), checked in collision-overlay renders.
GR = P.GUARD_RAISE
CHASSIS_HULLS = {
    "counterweight": ((-2.14, -1.44), (-0.58, 0.58), (0.13, 1.085)),
    "lpg_tank": ((-1.95, -1.54), (-0.40, 0.40), (1.06, 1.44)),
    "hood": ((-1.45, -0.66), (-0.52, 0.52), (0.13, 1.075)),
    "side_l": ((-0.70, -0.30), (0.41, 0.58), (0.13, 0.60)),
    "side_r": ((-0.70, -0.30), (-0.58, -0.41), (0.13, 0.60)),
    "floor": ((-0.70, -0.30), (-0.42, 0.42), (0.13, 0.60)),
    "cowl_low": ((-0.30, 0.21), (-0.58, 0.58), (0.13, 0.60)),
    "post_fl": ((-0.56, -0.10), (0.40, 0.58), (1.08, 2.13 + GR)),
    "post_fr": ((-0.56, -0.10), (-0.58, -0.40), (1.08, 2.13 + GR)),
    "post_rl": ((-1.52, -1.28), (0.36, 0.58), (1.05, 2.13 + GR)),
    "post_rr": ((-1.52, -1.28), (-0.58, -0.36), (1.05, 2.13 + GR)),
    # overhead guard as its parts (collision = visual): three rear slats, the front bar, the two side rails; the asset's
    # guard is open between the third slat (x -0.94) and the front bar (x -0.45)
    "guard_slat_a": ((-1.42, -1.30), (-0.43, 0.43), (2.00 + GR, 2.14 + GR)),
    "guard_slat_b": ((-1.22, -1.10), (-0.43, 0.43), (2.00 + GR, 2.14 + GR)),
    "guard_slat_c": ((-1.02, -0.90), (-0.43, 0.43), (2.00 + GR, 2.14 + GR)),
    "guard_front": ((-0.56, -0.40), (-0.43, 0.43), (2.00 + GR, 2.14 + GR)),
    "guard_rail_l": ((-1.46, -0.36), (0.40, 0.57), (1.95 + GR, 2.14 + GR)),
    "guard_rail_r": ((-1.46, -0.36), (-0.57, -0.40), (1.95 + GR, 2.14 + GR)),
}
SEAT_HULLS = {
    "cushion": ((-1.16, -0.73), (-0.19, 0.28), (1.02, 1.15)),
    "backrest": ((-1.29, -1.08), (-0.19, 0.28), (1.14, 1.56)),
}
NON_COLLIDING = ("Glass", "Lights", "Decals", "Mirror")
# The cab interior (footwell, toe board, cowl, dash, column, lever console, fender walls) by convex decomposition of the
# chassis mesh inside CAB_CROP (CoACD 1.0.5, pinned in assets/install.py TOOLS; 60 hulls), cached by mesh hash
CAB_CROP = ((-0.76, 0.22), (-0.60, 0.60), (0.60, 1.40))
# The footwell below the pedal pivots is NOT decomposed: a convex decomposition thickens and fills it around the pedals
# (E14: the robot's feet were held by hulls flush with the pedal pads). Its surfaces are flat plates measured
# from the mesh sections (floor plate z 0.665 m to x -0.40, toe board rising 0.321 per m to (-0.12, 0.76), front wall):
FOOTWELL = ((-0.72, -0.105), (-0.45, 0.45), (0.60, 0.80))
FOOTWELL_PLATES = {
    "fw_floor": [(-0.72, -0.42, 0.635), (-0.40, 0.42, 0.665)],
    "fw_front": [(-0.125, -0.42, 0.74), (-0.095, 0.42, 0.80)],
}
TOEBOARD = dict(x=(-0.40, -0.12), z=(0.67, 0.76), y=(-0.42, 0.42), t=0.03)
CAB_COACD = dict(threshold=0.03, max_convex_hull=60, preprocess_mode="on", preprocess_resolution=100, resolution=2000,
                 mcts_nodes=20, mcts_iterations=100, mcts_max_depth=2, seed=1)
# Phantom volume carved out of the decomposition (E14b). CoACD fills spaces the real cab leaves open: the key sits entirely
# inside hull cab50 (its pocket in front of the dash, 4.7 cm deep, has no surface in the mesh), and the
# parking-brake lever's slot in the column cowl and the open space below the cowl's left housing are filled by
# cab12/20/26/36/37/42/48. Each carve is a convex solid in the truck frame; a hull H minus a carve C is the union over C's
# faces of H clipped to the face's outside half-space (exact, every piece convex; pieces overlap, all on the chassis).
# Bounds measured from the chassis visual mesh: the dash face round the key at x -0.3376 to
# -0.3393 within 35 mm of the key's axis, a console feature at x -0.382 beyond 35 mm; the lever slot between the cowl's
# side plates at y 0.245 and 0.340, under its roof (lowest z 1.075-1.095 for x -0.34 to -0.20), in front of its rear wall
# (face x -0.346, bottom edge z 1.00); the left housing's lower edge runs from (x -0.36, z 1.05) to (-0.17, 0.87), its inner
# skin (y 0.34-0.37) from (-0.37, 1.00) to (-0.23, 0.88). Checked: no chassis-mesh sample point lies more than 3 mm inside
# any carve.
CAB_CARVE = {
    # in front of the key: a 12-sided prism round the key's axis (truck x), radius 33 mm, from 1.5 mm proud of the dash face
    "key_pocket": dict(prism_x=dict(y=-0.0535, z=1.033, r=0.033, sides=12, x=(-0.43, -0.3380))),
    # the open space under the handle's housing, below the cowl strip (its underside measured from the mesh in y
    # 0.247-0.338: z 0.989 at x -0.345 falling to 0.873 at -0.195, nothing below it down to 0.86), as two prisms whose
    # tops are chords of that underside lowered 1 mm (at most 1.3 mm above it); the housing's
    # interior keeps its hulls
    "park_under_rear": dict(prism_y=dict(xz=((-0.345, 0.988), (-0.270, 0.9226), (-0.270, 0.86), (-0.345, 0.86)),
                                         y=(0.247, 0.338))),
    "park_under_front": dict(prism_y=dict(xz=((-0.270, 0.9226), (-0.195, 0.872), (-0.195, 0.86), (-0.270, 0.86)),
                                          y=(0.247, 0.338))),
    # the open space below the left housing's lower edge, beside the lever (the hand's way to the handle from the left);
    # the housing's inner skin (y 0.34-0.37) comes lower than its outer plate
    "park_left_inner": dict(prism_y=dict(xz=((-0.36, 0.966), (-0.36, 0.86), (-0.239, 0.86)), y=(0.338, 0.372))),
    "park_left": dict(prism_y=dict(xz=((-0.36, 1.03), (-0.36, 0.86), (-0.18, 0.86)), y=(0.372, 0.47))),
    # the open air round the parking-brake handle's whole sweep: every hull volume within ~30 mm of the grip
    # at any lever angle, whatever the decomposition put there; the real surfaces inside it are given back as skin
    # hulls (PARK_SKIN, skin_hulls below)
    "park_handle_air": dict(box=((-0.445, 0.262, 1.000), (-0.250, 0.356, 1.250))),
}

# Skin hulls: the chassis mesh's own surfaces inside a carved region, rebuilt deterministically (no CoACD) as thin
# convex slabs: triangles binned by centroid into SKIN["cell"] cubes, split by normal (cosine >= SKIN["cos"]) and by plane
# offset (SKIN["plane_tol"]), each group's vertices offset +-SKIN["half"] along its mean normal and hulled. Regions:
# - the park handle's open-air carve (CAB_CARVE["park_handle_air"]), less the hidden linkage inside the slot (its pins
#   and sector plate stay drawn on the chassis, inside the slot liner, and are not given collision: in a real truck they
#   move with the lever);
# - the cowl strip below the handle's housing (the 1,430-triangle skin; the shipped decomposition has no
#   hull over it).
PARK_SKIN = dict(cell=0.03, cos=0.85, plane_tol=0.004, half=0.003, flat=0.002, fill=1.10)
PARK_LINKAGE_BOX = ((-0.340, -0.174), (0.287, 0.329), (0.970, 1.090))
PARK_COWL_STRIP_BOX = ((-0.361, -0.167), (0.244, 0.345), (0.862, 1.018))


def _skin_triangles(meshes):
    """[(A, B, C)] chassis triangles to skin (truck frame): those that reach into the park carve (3 mm margin) and the
    cowl strip, without the hidden linkage."""
    lo = np.array(CAB_CARVE["park_handle_air"]["box"][0]) - 0.003
    hi = np.array(CAB_CARVE["park_handle_air"]["box"][1]) + 0.003
    tris = []
    for l, v, f in meshes["chassis"]:
        if any(k in l for k in NON_COLLIDING):
            continue
        v = np.asarray(v, np.float64)
        f = np.asarray(f)
        T = v[f]
        keep = (T.max(1) >= lo).all(1) & (T.min(1) <= hi).all(1)        # every triangle that reaches into the carve
        if l == "Body01:rest":
            tl = _components(v, f)
            order = np.argsort(tl, kind="stable")
            bounds = np.searchsorted(tl[order], np.arange(int(tl.max()) + 2))
            for k in range(int(tl.max()) + 1):
                t = order[bounds[k]:bounds[k + 1]]
                if not len(t):
                    continue
                pts = v[f[t]].reshape(-1, 3)
                if _inside(pts.min(0), pts.max(0), PARK_LINKAGE_BOX, 0.001):
                    keep[t] = False
                elif _inside(pts.min(0), pts.max(0), PARK_COWL_STRIP_BOX, 0.001):
                    keep[t] = True
        if keep.any():
            tris.append(v[f[keep]])
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def _patch_ok(T, area):
    """A patch (triangles T, areas) may become one slab: flat within PARK_SKIN["flat"] along its mean normal, and its
    in-plane convex hull no more than PARK_SKIN["fill"] times its own area (so it never bridges a notch or a slot)."""
    from scipy.spatial import ConvexHull, QhullError
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    m = n.sum(0)
    if np.linalg.norm(m) < 1e-12:
        return False, None
    m = m / np.linalg.norm(m)
    V = T.reshape(-1, 3)
    h = V @ m
    if h.max() - h.min() > PARK_SKIN["flat"]:
        return False, m
    u = np.cross(m, [1.0, 0.0, 0.0] if abs(m[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
    w = np.cross(m, u)
    uv = np.stack([V @ u, V @ w], 1)
    proj = 0.5 * np.abs(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]) @ m).sum()
    try:
        hull_area = ConvexHull(uv).volume
    except QhullError:
        return True, m
    return hull_area <= PARK_SKIN["fill"] * max(proj, 1e-12), m


def skin_hulls(meshes=None):
    """[(V, F, label)] the chassis skin slabs (see PARK_SKIN), deterministic; cached by the mesh split and PARK_SKIN.
    Seeds: triangles in SKIN["cell"] cubes, split by normal and plane offset and into vertex-connected pieces; then
    neighbouring patches merge greedily while the union stays a flat, notch-free patch (_patch_ok)."""
    M = meshes or body_meshes()
    key = hashlib.sha256(json.dumps([CAB_CARVE["park_handle_air"], PARK_SKIN, PARK_LINKAGE_BOX, PARK_COWL_STRIP_BOX,
                                     P.GUARD_RAISE, "merge+dedup", "reach"], sort_keys=True).encode()).hexdigest()[:8]
    cache = cache_dir() / "derived" / f"park_skin_{_sha(usd_path())[:16]}_{_regions_sig()}_{key}.npz"
    if cache.is_file():
        z = np.load(cache)
        return [(z[f"v{i}"], z[f"f{i}"], f"skin{i:02d}") for i in range(int(z["n"]))]
    from scipy.spatial import ConvexHull, QhullError, cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    T = _skin_triangles(M)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    ok = area > 1e-10
    T, n, area = T[ok], n[ok] / (2 * area[ok, None]), area[ok]
    # triangle adjacency: shared (welded at 0.1 mm) vertex
    q = np.round(T.reshape(-1, 3) * 1e4).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel().reshape(-1, 3)
    tri_of_v = {}
    for t, vs in enumerate(inv):
        for vv in vs:
            tri_of_v.setdefault(int(vv), []).append(t)
    cell = np.floor(T.mean(1) / PARK_SKIN["cell"]).astype(np.int64)
    patches = []
    for ck in sorted({tuple(c) for c in cell}):
        idx = np.where((cell == np.array(ck)).all(1))[0]
        groups = []
        for t in idx[np.argsort(-area[idx], kind="stable")]:
            for g in groups:
                if (g[0] / np.linalg.norm(g[0])) @ n[t] >= PARK_SKIN["cos"]:
                    g[0] += area[t] * n[t]; g[1].append(t)
                    break
            else:
                groups.append([area[t] * n[t], [t]])
        for g0, members in groups:
            m = g0 / np.linalg.norm(g0)
            d = T[members].mean(1) @ m
            order = np.argsort(d, kind="stable")
            runs, run = [], [members[order[0]]]
            for a, b in zip(order[:-1], order[1:]):
                if d[b] - d[a] > PARK_SKIN["plane_tol"]:
                    runs.append(run); run = []
                run.append(members[b])
            runs.append(run)
            for r in runs:                                   # vertex-connected pieces
                r = np.asarray(r); pos = {int(t): k for k, t in enumerate(r)}
                rows, cols = [], []
                for k, t in enumerate(r):
                    for vv in inv[t]:
                        for t2 in tri_of_v[int(vv)]:
                            if t2 in pos:
                                rows.append(k); cols.append(pos[t2])
                _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(r), len(r))))
                for c in range(lab.max() + 1):
                    patches.append(set(int(t) for t in r[lab == c]))
    # greedy merge of touching patches
    owner = {}
    for k, pset in enumerate(patches):
        for t in pset:
            owner[t] = k
    alive = {k: pset for k, pset in enumerate(patches)}

    def neighbours(k):
        out = set()
        for t in alive[k]:
            for vv in inv[t]:
                for t2 in tri_of_v[int(vv)]:
                    o = owner[t2]
                    if o != k:
                        out.add(o)
        return out

    changed = True
    while changed:
        changed = False
        for k in sorted(alive, key=lambda k: (-area[list(alive[k])].sum(), k)):
            if k not in alive:
                continue
            for o in sorted(neighbours(k)):
                if o not in alive:
                    continue
                u = sorted(alive[k] | alive[o])
                good, _ = _patch_ok(T[u], area[u])
                if good:
                    alive[k] = set(u)
                    for t in alive.pop(o):
                        owner[t] = k
                    changed = True
    out = []
    for k in sorted(alive):
        u = sorted(alive[k])
        _, m = _patch_ok(T[u], area[u])
        if m is None:
            continue
        V = T[u].reshape(-1, 3)
        V = np.concatenate([V + PARK_SKIN["half"] * m, V - PARK_SKIN["half"] * m])
        try:
            h = ConvexHull(V)
        except QhullError:
            continue
        if h.volume < 1e-10:
            continue
        out.append((V[h.vertices].astype(np.float32), _reindex(V, h), h.volume, h.equations))
    # drop slabs that lie within 1 mm inside a larger one (small features already covered)
    order = sorted(range(len(out)), key=lambda i: (-out[i][2], i))
    kept = []
    for i in order:
        Vi = out[i][0].astype(np.float64)
        if not any(((Vi @ out[j][3][:, :3].T + out[j][3][:, 3]).max(1) <= 0.001).all() for j in kept):
            kept.append(i)
    out = [out[i][:2] for i in sorted(kept)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, n=len(out), **{f"v{i}": v for i, (v, f) in enumerate(out)},
                        **{f"f{i}": f for i, (v, f) in enumerate(out)})
    return [(v, f, f"skin{i:02d}") for i, (v, f) in enumerate(out)]


def carve_planes(spec):
    """Outward face planes [(n, d)] (n.x <= d inside) of a CAB_CARVE solid, and its vertices."""
    from scipy.spatial import ConvexHull
    if "box" in spec:
        lo, hi = (np.asarray(v, float) for v in spec["box"])
        V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    elif "prism_x" in spec:
        q = spec["prism_x"]
        a = np.linspace(0.0, 2 * np.pi, q["sides"], endpoint=False)
        ring = np.stack([q["y"] + q["r"] * np.cos(a), q["z"] + q["r"] * np.sin(a)], 1)
        V = np.array([[x, yy, zz] for x in q["x"] for yy, zz in ring])
    elif "prism_y" in spec:
        q = spec["prism_y"]
        V = np.array([[x, y, z] for y in q["y"] for x, z in q["xz"]])
    else:
        raise ValueError(spec)
    h = ConvexHull(V)
    planes = []
    for eq in h.equations:                                   # n.x + c <= 0 inside
        n, d = eq[:3], -eq[3]
        if not any(np.allclose(n, p[0], atol=1e-9) and abs(d - p[1]) < 1e-9 for p in planes):
            planes.append((n, d))
    return planes, V


def _clip_outside(V, F, n, d):
    """The part of convex hull (V, F) on the outside of plane n.x = d (n.x >= d), exactly, or None."""
    s = V @ n - d
    keep = V[s >= -1e-12]
    if len(keep) == 0:
        return None
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    a, b = s[e[:, 0]], s[e[:, 1]]
    cross = (a * b) < 0
    t = a[cross] / (a[cross] - b[cross])
    P = V[e[cross, 0]] + t[:, None] * (V[e[cross, 1]] - V[e[cross, 0]])
    return np.concatenate([keep, P])


def carve_hulls(parts, carve=None, min_volume=1e-7):
    """Each convex part minus every carve solid (see CAB_CARVE). Returns (parts, record)."""
    from scipy.spatial import ConvexHull, QhullError
    carve = CAB_CARVE if carve is None else carve
    rec = dict(parts_before=len(parts), carved={})
    out = [(np.asarray(V, float), np.asarray(F, int)) for V, F in parts]
    for name, spec in carve.items():
        planes, _ = carve_planes(spec)
        nxt, touched = [], 0
        for V, F in out:
            # separated if some carve plane has the whole hull outside it
            if any((V @ n - d).min() >= -1e-9 for n, d in planes):
                nxt.append((V, F))
                continue
            touched += 1
            for n, d in planes:
                P = _clip_outside(V, F, n, d)
                if P is None or len(P) < 4:
                    continue
                try:
                    h = ConvexHull(P)
                except QhullError:
                    continue
                if h.volume < min_volume:
                    continue
                nxt.append((P[h.vertices], _reindex(P, h)))
        rec["carved"][name] = dict(hulls_cut=touched)
        out = nxt
    rec["parts_after"] = len(out)
    return [(np.asarray(V, np.float32), np.asarray(F, np.int32)) for V, F in out], rec


def _reindex(P, h):
    idx = {int(v): k for k, v in enumerate(h.vertices)}
    return np.array([[idx[int(i)] for i in tri] for tri in h.simplices], np.int32)


cab_hulls_override = None              # probes only (E14b carve check): a list of hulls used in place of the cached ones

# The cab's convex decomposition is SHIPPED, not regenerated. CoACD 1.0.5 is not deterministic here despite its fixed
# seed (it runs multithreaded): two fresh regenerations on one host shared 21 of 60 hulls, and after the carve gave 127
# and 131 parts against the shipped 138. The shipped decomposition (60 hulls, from the chassis mesh inside CAB_CROP minus
# FOOTWELL with CAB_COACD) is in Git, locked by its SHA-256 below; a world build refuses any other file, and the doctor
# checks it. It was decomposed under a Body01 split with REGIONS signature 004ec916, in which the parking-brake handle's grip, end cap and bar were chassis (so hulls wrap them: the park carves
# below remove the handle's sweep) and a strip of the cowl's lower skin, the slot liner and the linkage pins were the
# lever (so no hull covers them: PARK_COWL_STRIP below adds the strip). A deliberate new decomposition is
# `python -m forklift.geometry --decompose-cab OUT.npz`, then a new CAB_HULLS_SHA256 and every result re-run.
CAB_HULLS_FILE = Path(__file__).parent / "data" / "cab_hulls_coacd.npz"
CAB_HULLS_SHA256 = "861c70f30ed4f62cda7383ea4065eb0cddb6da895b02f2d66366ce320cd0b619"
_cab_checked = []


def cab_hulls_status():
    """(ok, detail) of the shipped cab decomposition against its lock."""
    if not CAB_HULLS_FILE.is_file():
        return False, f"{CAB_HULLS_FILE} missing (it ships with the forklift package)"
    sha = _sha(CAB_HULLS_FILE)
    if sha != CAB_HULLS_SHA256:
        return False, f"{CAB_HULLS_FILE.name} SHA-256 {sha[:16]}... differs from the lock {CAB_HULLS_SHA256[:16]}..."
    return True, f"{CAB_HULLS_FILE.name}: 60 convex hulls, SHA-256 {sha[:16]}... as locked"


def _shipped_cab_hulls():
    if not _cab_checked:
        ok, detail = cab_hulls_status()
        if not ok:
            raise RuntimeError(f"refusing to build the cab from a different decomposition: {detail}")
        _cab_checked.append(True)
    z = np.load(CAB_HULLS_FILE)
    return [(z[f"v{i}"], z[f"f{i}"]) for i in range(int(z["n"]))]


def cab_hulls(meshes=None, carved=True):
    """[(V, F)] convex parts of the cab interior (truck frame): the shipped, hash-locked decomposition; carved=True
    removes the CAB_CARVE solids (exact half-space clipping; cached by the lock and the carve)."""
    if cab_hulls_override is not None:
        return cab_hulls_override
    if not carved:
        return _shipped_cab_hulls()
    ckey = hashlib.sha256(json.dumps([CAB_HULLS_SHA256, CAB_CARVE], sort_keys=True).encode()).hexdigest()[:8]
    ccache = cache_dir() / "derived" / f"cab_carved_{CAB_HULLS_SHA256[:16]}_{ckey}.npz"
    if ccache.is_file():
        _shipped_cab_hulls()                     # the lock is checked on every build, cache or not
        z = np.load(ccache)
        return [(z[f"v{i}"], z[f"f{i}"]) for i in range(int(z["n"]))]
    out, _ = carve_hulls(_shipped_cab_hulls())
    ccache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ccache, n=len(out), **{f"v{i}": v for i, (v, f) in enumerate(out)},
                        **{f"f{i}": f for i, (v, f) in enumerate(out)})
    return out


def decompose_cab(meshes, out_path):
    """A NEW CoACD decomposition of the cab interior (not reproducible run to run; see CAB_HULLS_FILE). Writes out_path
    and returns its SHA-256. Never called by a world build."""
    import sys
    from .assets.install import install_tools
    sys.path.insert(0, str(install_tools()))
    import coacd
    Vs, Fs, off = [], [], 0
    (x0, x1), (y0, y1), (z0, z1) = CAB_CROP
    for l, v, f in meshes["chassis"]:
        if any(k in l for k in NON_COLLIDING):
            continue
        c = v[f].mean(1)
        box = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0) & (c[:, 2] < z1)
        (fx0, fx1), (fy0, fy1), (fz0, fz1) = FOOTWELL
        fw = (c[:, 0] > fx0) & (c[:, 0] < fx1) & (c[:, 1] > fy0) & (c[:, 1] < fy1) & (c[:, 2] > fz0) & (c[:, 2] < fz1)
        ff = f[box & ~fw]
        if not len(ff):
            continue
        used, inv = np.unique(ff.ravel(), return_inverse=True)
        Vs.append(v[used]); Fs.append(inv.reshape(-1, 3) + off); off += len(used)
    parts = coacd.run_coacd(coacd.Mesh(np.concatenate(Vs).astype(np.float64), np.concatenate(Fs)), **CAB_COACD)
    out = [(np.asarray(p[0], np.float32), np.asarray(p[1], np.int32)) for p in parts]
    np.savez_compressed(out_path, n=len(out), **{f"v{i}": v for i, (v, f) in enumerate(out)},
                        **{f"f{i}": f for i, (v, f) in enumerate(out)})
    return _sha(out_path)


def _box_vertices(V, box):
    lo = np.array([b[0] for b in box]); hi = np.array([b[1] for b in box])
    m = (V >= lo).all(1) & (V <= hi).all(1)
    return V[m]


BLADE_SEGMENTS = 8
BLADE_TOP = 0.1402           # m, the blades' flat top in the truck build frame (asset mesh, measured)
BLADE_TIP_X = 1.546          # m, where the tip taper begins


def _slabs(hull, n):
    """Cut a convex hull (V, F) into n convex slabs of equal length along x (each capped, re-hulled)."""
    import trimesh
    V, F = hull
    m = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=True)
    x0, x1 = float(m.bounds[0][0]), float(m.bounds[1][0])
    out = []
    for k in range(n):
        a, b = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
        piece = m.slice_plane([a, 0.0, 0.0], [1.0, 0.0, 0.0], cap=True).slice_plane([b, 0.0, 0.0], [-1.0, 0.0, 0.0], cap=True)
        h = _hull(piece.vertices)
        if h is not None:
            out.append(h)
    return out


def _hull(points):
    import trimesh
    if len(points) < 8:
        return None
    try:
        h = trimesh.convex.convex_hull(np.asarray(points, np.float64))
    except Exception:
        return None
    if h.volume < 1e-9:
        return None
    return np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.int32)


def rim_geometry(meshes):
    """Steering-wheel rim: centre, unit normal (column axis toward the operator), centreline radius, tube radius."""
    V = np.concatenate([v[np.unique(f)] for label, v, f in meshes["steering_wheel"] if label.endswith("rim")])
    c = V.mean(0)
    _, _, vt = np.linalg.svd(V - c)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    d = V - c
    r = np.linalg.norm(d - np.outer(d @ n, n), axis=1)
    ring = V[r > 0.75 * r.max()]
    rr = r[r > 0.75 * r.max()]
    h = (ring - c) @ n
    rc = 0.5 * (rr.min() + rr.max())
    rt = 0.5 * max(rr.max() - rr.min(), h.max() - h.min())
    c = c + n * float(np.median(h))
    return c.astype(float), n.astype(float), float(rc), float(rt)


def collision_shapes(meshes=None):
    """{body: [shape]} in the truck frame. shape = ("hull", V, F) | ("capsule", p0, p1, r) | ("cylinder_y", centre, r, hw)."""
    M = meshes or body_meshes()
    out = {}

    def entries(body, skip=()):
        return [(l, v, f) for l, v, f in M[body] if not any(k in l for k in skip)]

    chassis_V = np.concatenate([v[np.unique(f)] for l, v, f in entries("chassis", NON_COLLIDING)])
    for name, box in CHASSIS_HULLS.items():
        h = _hull(_box_vertices(chassis_V, box))
        if h is not None:
            out.setdefault("chassis", []).append(("hull", *h, name))
    for i, (V, F) in enumerate(cab_hulls(M)):
        h = _hull(V)
        if h is not None:
            out.setdefault("chassis", []).append(("hull", *h, f"cab{i:02d}"))
    for V, F, label in skin_hulls(M):
        out["chassis"].append(("hull", V, F, label))
    for name, (lo, hi) in FOOTWELL_PLATES.items():
        V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        out["chassis"].append(("hull", *_hull(V), name))
    tb = TOEBOARD
    V = np.array([[x, y, z - dz] for (x, z) in zip(tb["x"], tb["z"]) for y in tb["y"] for dz in (0.0, tb["t"])])
    out["chassis"].append(("hull", *_hull(V), "fw_toeboard"))
    seat_V = np.concatenate([v[np.unique(f)] for l, v, f in entries("seat") if "OperatorChair" not in l])
    for name, box in SEAT_HULLS.items():
        V = _box_vertices(seat_V, box)
        if name == "cushion":
            # rigid cushion at its loaded thickness: the foam's top surface lowered by its static set under the operator
            V = V.copy()
            top = V[:, 2] > V[:, 2].max() - 0.04
            V[top, 2] -= P.SEAT_FOAM_SET
        h = _hull(V)
        out.setdefault("seat", []).append(("hull", *h, name))
    for l, v, f in entries("seat"):
        if "OperatorChair" in l:
            out["seat"].append(("hull", *_hull(v[np.unique(f)]), l.split("/")[-1]))
    for pedal in ("pedal_inch", "pedal_brake"):
        V = body_vertices(M, pedal)
        out[pedal] = [("hull", *_hull(V[V[:, 0] < -0.255]), "pad"), ("hull", *_hull(V[V[:, 0] >= -0.265]), "arm")]
    # the parking-brake handle: its grip (with the end cap) and its flat bar, one hull each
    out["park_lever"] = [("hull", *_hull(v[np.unique(f)]), l.split("/")[-1]) for l, v, f in M["park_lever"]]
    for name in ("pedal_accel", "dir_lever", "key", "knob", "knuckle_l", "knuckle_r", "axle", "carriage"):
        out[name] = [("hull", *_hull(body_vertices(M, name)), name)]
    for lever in ("lever_lift", "lever_tilt", "lever_shift"):
        V = body_vertices(M, lever)
        out[lever] = [("hull", *_hull(V[V[:, 0] > -0.452]), "shaft"), ("hull", *_hull(V[V[:, 0] <= -0.452]), "knob")]
    c, n, rc, rt = rim_geometry(M)
    a = np.cross(n, [1.0, 0.0, 0.0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    ring = [c + rc * (np.cos(t) * a + np.sin(t) * b) for t in np.linspace(0, 2 * np.pi, 17)]
    out["steering_wheel"] = [("capsule", ring[i], ring[i + 1], rt, f"rim{i}") for i in range(16)]
    hub = np.concatenate([v[np.unique(f)] for l, v, f in M["steering_wheel"] if l.endswith("hub")])
    out["steering_wheel"].append(("hull", *_hull(hub), "hub"))
    for stage in ("mast_outer", "mast_int", "mast_inner"):
        # without the decal quads: the outer stage's sticker mesh reaches 0.155 m ahead of its rails (x 0.547, flush
        # with the fork face, z 0.30-0.34), and upright hulls that include it carry a phantom lug (it struck a
        # floor-stored pallet under a put-away slot at 425 kN)
        V = np.concatenate([v[np.unique(f)] for l, v, f in entries(stage, NON_COLLIDING)])
        for side, m in (("l", V[:, 1] > 0.06), ("r", V[:, 1] < -0.06)):
            out.setdefault(stage, []).append(("hull", *_hull(V[m]), f"upright_{side}"))
    ss = []
    for l, v, f in M["sideshift"]:
        V = v[np.unique(f)]
        if "_Fork0" in l:
            # blade: in front of the shank face and its heel radius only. Taken as every vertex below 0.160 m it swept in the heel radius
            # (19 mm above the 0.140 m blade top at x = 0.54), and its hull's top became a 1.1 deg ramp falling to the
            # tips: a bale rested on one line near its centre of mass and tipped off when the truck reversed
            # The flat run is BLADE_SEGMENTS box primitives (top at the blade's measured 0.1402 m), the tapered tip a
            # hull. As one convex mesh (and as mesh slabs) a load got contacts only along one cross-line per piece
            # near the deepest point (MuJoCo mesh-mesh), so a bale balanced on a line 45 mm ahead of its centre of
            # mass and tipped off the forks at under 1 m/s^2; box faces give the contact polygon
            B = V[(V[:, 2] < 0.160) & (V[:, 0] >= 0.549)]
            x0, x1 = 0.549, BLADE_TIP_X
            y0, y1, z0, z1 = B[:, 1].min(), B[:, 1].max(), B[:, 2].min(), BLADE_TOP
            hl = (x1 - x0) / (2 * BLADE_SEGMENTS)
            for k in range(BLADE_SEGMENTS):
                c = (x0 + hl * (2 * k + 1), (y0 + y1) / 2, (z0 + z1) / 2)
                ss.append(("box", c, (hl, (y1 - y0) / 2, (z1 - z0) / 2), l.split("_")[-2] + f"_blade{k}"))
            ss.append(("hull", *_hull(B[B[:, 0] >= x1 - 0.002]), l.split("_")[-2] + "_bladetip"))
            ss.append(("hull", *_hull(V[V[:, 0] < 0.549]), l.split("_")[-2] + "_shank"))
        else:
            ss.append(("hull", *_hull(V), "backrest"))
    out["sideshift"] = ss
    for w in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        V = body_vertices(M, w)
        lo, hi = V.min(0), V.max(0)
        out[w] = [("cylinder_y", (lo + hi) / 2, float((hi[2] - lo[2]) / 2), float((hi[1] - lo[1]) / 2), "tyre")]
    return out


# ------------------------------------------------------------------------------------------------ visual meshes (render)
def _extract_visual(path):
    """Every mesh of the asset with triangle corners' UVs (primvars:st, faceVarying) and its bound material path."""
    from pxr import Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(str(path))
    out = {}
    for p in st.Traverse():
        if p.GetTypeName() != "Mesh":
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
        X = np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64)
        w = (np.c_[pts, np.ones(len(pts))] @ X)[:, :3]
        w[:, 0] -= FRONT_AXLE_X_ASSET
        pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
        uv = np.asarray(pv.Get(), np.float32) if pv and pv.Get() is not None else None
        uv_idx = np.asarray(pv.GetIndices(), np.int64) if pv and pv.IsIndexed() else None
        interp = pv.GetInterpolation() if pv else None
        tris, cor = [], []
        o = 0
        for c in counts:
            for k in range(1, c - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
                cor.append((o, o + k, o + k + 1))
            o += c
        tris = np.asarray(tris, np.int32); cor = np.asarray(cor, np.int64)
        if uv is not None:
            if interp == "faceVarying":
                sel = uv_idx[cor] if uv_idx is not None else cor
            else:
                sel = uv_idx[tris] if uv_idx is not None else tris
            UV = uv[sel]
        else:
            UV = np.zeros((len(tris), 3, 2), np.float32)
        mat = str(UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterialPath())
        name = str(p.GetPath())[len(ROOT) + 1:]
        out[name] = (w.astype(np.float32), tris, UV.astype(np.float32), mat)
    return out


def visual_parts():
    """{body: [(label, V (n,3) truck frame incl. the truck's configuration, F (m,3), UV (m,3,2), material path)]} with the
    same split into simulated bodies as body_meshes (regions cut from Body01 by connected components). Cached."""
    path = usd_path()
    cache = cache_dir() / "derived" / f"forklift_c_visual_{_sha(path)[:16]}_{_regions_sig()}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        index = json.loads(str(z["index"]))
        out = {}
        for i, (body, label, mat) in enumerate(index):
            out.setdefault(body, []).append((label, z[f"v{i}"], z[f"f{i}"], z[f"t{i}"], mat))
    else:
        data = _extract_visual(path)
        out = {}
        v, f, uv, mat = data[BODY01]
        owner = _split_body01(v, f)
        for lab in (o for o in np.unique(owner) if o):
            m = owner == lab
            out.setdefault(lab.split("/")[0], []).append(("Body01:" + lab, v, f[m], uv[m], mat))
        m = owner == ""
        out.setdefault("chassis", []).append(("Body01:rest", v, f[m], uv[m], mat))
        for name, (vv, ff, uu, mm) in data.items():
            if name == BODY01:
                continue
            body = next((b for prefix, b in MESH_BODY.items() if name.startswith(prefix)), "chassis")
            if name == "body/SM_Forklift_C01_SteeringWheel_01":
                stc = _components(vv, ff)
                sizes = np.bincount(stc)
                order = np.argsort(-sizes)
                for part, ci in (("steering_wheel", order[0]), ("steering_wheel", order[1]), ("knob", order[2])):
                    sel = stc == ci
                    out.setdefault(part, []).append((f"SteeringWheel:{part}{ci}", vv, ff[sel], uu[sel], mm))
                continue
            out.setdefault(body, []).append((name, vv, ff, uu, mm))
        index, arrays = [], {}
        compact = {}
        for body, entries in out.items():
            for label, vv, ff, uu, mm in entries:
                used, inv = np.unique(ff.ravel(), return_inverse=True)
                V, F = vv[used].astype(np.float32), inv.reshape(-1, 3).astype(np.int32)
                compact.setdefault(body, []).append((label, V, F, uu.astype(np.float32), mm))
                i = len(index)
                arrays[f"v{i}"], arrays[f"f{i}"], arrays[f"t{i}"] = V, F, uu.astype(np.float32)
                index.append((body, label, mm))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, index=json.dumps(index), **arrays)
        out = compact
    out = dict(out)
    out["park_lever"] = [(label, park_released(V), F, UV, mat) for label, V, F, UV, mat in out["park_lever"]]
    if P.GUARD_RAISE and "chassis" in out:
        new = []
        for label, V, F, UV, mat in out["chassis"]:
            V = V.copy()
            V[V[:, 2] > P.GUARD_SPLIT, 2] += P.GUARD_RAISE
            new.append((label, V, F, UV, mat))
        out["chassis"] = new
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m forklift.geometry")
    ap.add_argument("--decompose-cab", metavar="OUT.npz", help="write a NEW CoACD decomposition of the cab (see "
                    "CAB_HULLS_FILE: not reproducible run to run; adopting it means a new lock and re-running every result)")
    ap.add_argument("--check", action="store_true", help="check the shipped cab decomposition against its lock")
    a = ap.parse_args()
    if a.decompose_cab:
        print(decompose_cab(body_meshes(), a.decompose_cab))
    else:
        print(*cab_hulls_status())

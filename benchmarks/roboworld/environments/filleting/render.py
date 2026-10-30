"""Render path for filleting: USD export of recorded states + the shared render-only Isaac RTX worker.

Frames are rendered from recorded states, never by re-running physics. The worker is
`wiping.render_worker` (path tracing, OptiX denoiser, fixed exposure ISO 400 f/2.8 1/120 s) launched through
filleting/render_worker.py, which adds changing-topology tissue surfaces to the shared state update.

Frame format (from the physics world):
  frame = {"tick": int, "t": float, "badge": str,                   # "ASSISTED" / "UNASSISTED"
           "robot_links": [{"name": str, "mesh": id, "pos": (3,), "quat_xyzw": (4,)}, ...],
           "bodies": {name: {"mesh": id, "pos": (3,), "quat_xyzw": (4,)}},
           "tissue": {"points": (N,3) world, "tris": (M,3) int, "tri_kind": (M,) uint8
                        (0 skin membrane (fish skin, pork silver skin), 1 cut flesh face, 2 original non-skin boundary
                         (trout cavity, slit and head faces; salmon flesh face; pork exterior), 3 fat),
                      "rest": (N,3) rest coordinates (the anatomy's workpiece frame),
                      "uv": (N,2) skin UV, "skin_texture": png,
                      optional "flesh_color": (N,3) sRGB 0..1 (anatomy flesh_color(rest)),
                      optional "flesh_phase": (N,) myomere phase (sheet lines where frac = 0)},
           "rods": [{"points": (k,3), "radius": (k,)}, ...],               # ribs, pin bones (bone tubes)
           optional "fins": {"points": (K,3), "tris": (F,3), "uv": (K,2),  # fin membranes (render-only)
                             optional "rest": (K,3), "groups": [(v0, v1)], "names": [str]},  # per-fin ranges
           optional "scale_g": float}                                       # bench scale display reading

Mesh ids (local frames; a physics body's pose is the pose of this frame):
  "fr3/<urdf link>"          FR3 + Franka Hand visual meshes from the locked robot bundle (robots/models/fr3),
                             URDF link frame (e.g. fr3/fr3_link3, fr3/fr3_hand, fr3/fr3_leftfinger)
  "panda/<urdf link>"        Panda + Franka Hand visual meshes (robots/models/panda), e.g. panda/panda_link3
  "knife:fillet" | "knife:boning"   whole knife in the BladeSpec frame (origin at the heel on the edge, x to the
                             tip, y through the thickness, z edge -> spine); ":blade", ":handle", ":seg<k>" parts
                             (segment k: origin at its edge centre, as BladeSpec.segment_shapes)
  "board:blue" | "board:red" board, origin at its centre
  "tweezers"                 origin at the joined end, x toward the tips, arms closing along z
  "bowl" | "fillet_tray" | "knife_block" | "scale" | "tweezer_tray"   origin at the bottom-face centre
  a path to a .dae/.stl/.obj file, or a dict {"points", "tris", "material": "bone"|"skin"|"steel"|...}
  or {"capsules": [(p0, p1, r), ...], "material": ...} (e.g. anatomy rigid bones, the trout head)
Rigid-body meshes are authored ONCE, from the frame passed to export_scene / Renderer (normally the first recorded
frame): every body that can ever appear must be present in that frame. A body missing from a later frame is parked
under the floor (hidden); a body that first appears after the export frame is not drawn (build_request reports it
in Renderer.unknown_bodies).
A body whose name matches a dressing item (board, bowl, fillet_tray, knife_block, scale, tweezer_tray,
tweezers, knife_fillet, knife_boning, knife) replaces that item: the static dressing copy is not drawn.

Anatomy hook: tissue_appearance(workpiece, rest) fills uv / skin_texture / flesh_color (/ flesh_phase) from a
filleting/anatomy Workpiece. When flesh_color is given without flesh_phase, the myomere phase, the per-pixel line
widths and the trout's cavity classes are derived from the anatomy generator rebuilt from the texture's seed, only if
it reproduces the recorded flesh_color (node_fields; ANATOMY_CHECK holds the check); otherwise no lines are drawn.
Per-corner primvars of the tissue surfaces: st, displayColor, myophase, restpos, aux (fish_surfaces.mdl).

Changing topology: fixed-size pages (PAGE_TRIANGLES unindexed triangles per page, unused slots collapsed to one
point, empty pages invisible), the repo's approach (wiping/film_render.py) extended with UV, colour and myomere
phase primvars. `tissue_mode="reauthor"` rewrites one mesh per kind instead.
"""
from gym_anything_robotics.paths import worker_env
import base64
import io
import json
import math
import os
import re
import secrets
import select
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import layout as L

ROOT = Path(__file__).resolve().parent


PAGE_TRIANGLES = 4096
KINDS = {0: "skin", 1: "flesh", 2: "cavity", 3: "fat"}
DRESSING_ITEMS = ("board", "bowl", "fillet_tray", "knife_block", "scale", "tweezer_tray", "tweezers",
                  "knife_fillet", "knife_boning", "knife", "ice_pan")

# ------------------------------------------------------------------------------------------------ cameras
# The policy overhead camera's mount, re-framed. layout.CAMERAS["overhead"] (1.00 m over the board centre, y 0.28)
# frames y -0.05..0.61 at the bench and only up to y 0.54 at the knife block's top (z 1.10): the block's top face, slot
# mouths and both knife handles (y 0.56-0.61, z 1.10-1.24) are 20-185 px above the image (handles 91-99 px), so
# take-knife and put-away-knife would show no knife; it also cuts the bench at x -0.585 (half of the B5 scale, the A3
# tray and frame pan) and shows a quarter of the bowl. Looking straight down from
# 1.25 m over (0, 0.39) frames y -0.02..0.80 and x +-0.72 at the board, the block, handles and a knife drawn out above
# it (y <= 0.64 at z 1.40). Same lens (60 deg: the same advertised intrinsics) and image axes (+x right, +y up); the
# board is 0.8x as wide in pixels. Set here, not in layout.py, because layout.py is part of the physics world's
# identity (provenance.WORLD) and a camera pose changes no physics; dressing.build_camera_mounts hangs the rod here.
OVERHEAD_EYE = (0.0, 0.39, L.BOARD_TOP_Z + 1.25)
OVERHEAD_TARGET = (0.0, 0.39, L.BOARD_TOP_Z)
# eye, target, horizontal FOV (deg), role. Fixed policy cameras come from layout.CAMERAS (the overhead's pose from
# OVERHEAD_EYE/OVERHEAD_TARGET above); the wrist cameras are mounted on each FR3 link7 (the shared D405 mounting used
# by label_application); hero and bench are diagnostic.
CAMERAS = {
    "overhead": dict(eye=OVERHEAD_EYE, target=OVERHEAD_TARGET,
                     hfov=L.CAMERAS["overhead"]["hfov_deg"], role="policy", up=(0., 1., 0.)),
    "supervisor": dict(eye=L.CAMERAS["supervisor"]["pos"], target=L.CAMERAS["supervisor"]["look_at"],
                       hfov=L.CAMERAS["supervisor"]["hfov_deg"], role="policy"),
    "hero": dict(eye=(2.50, -2.05, 2.10), target=(-0.25, 0.55, 1.00), hfov=78., role="diagnostic"),
    "hero_reverse": dict(eye=(-1.95, 0.95, 2.40), target=(1.30, -0.55, 0.85), hfov=80., role="diagnostic"),
    "bench": dict(eye=(1.25, 1.10, 1.70), target=(-0.05, 0.22, 0.97), hfov=50., role="diagnostic"),
    "workpiece": dict(eye=(0.10, 0.66, 1.33), target=(0.0, 0.27, 0.925), hfov=42., role="diagnostic"),
    # close-up of the knife block and the right hand taking the knife (task A0), from the bench's right front
    "knife_block": dict(eye=(0.62, 0.32, 1.32), target=(0.32, 0.58, 1.14), hfov=38., role="diagnostic"),
}
# D405 on the FR3 wrist link (link7 frame): case, bracket, optical centre, looking along +z (label_application)
D405 = dict(case=((.075, 0., .07), (.042, .042, .023)), bracket=((.05, 0., .046), (.05, .026, .006)),
            eye=(.075, -.012, .083), target=(.075, -.012, .283), up=(0., 1., 0.), hfov=87.,
            source="robots/attachments/d405 (60 g, 42 x 42 x 23 mm envelope; ideal pinhole 87 deg, 1280 x 720)")


def _sanitize(name):
    return "".join(c if c.isalnum() else "_" for c in name)


# ------------------------------------------------------------------------------------------------ robot meshes
@lru_cache(maxsize=1)
def _fr3_urdf():
    from robots.assets import resolve
    urdf = Path(resolve("fr3")).parent/"fr3_franka_hand.urdf"
    return urdf, ET.parse(urdf).getroot()


@lru_cache(maxsize=1)
def _panda_urdf():
    from robots.assets import resolve
    urdf = Path(resolve("panda"))
    return urdf, ET.parse(urdf).getroot()


# the arm models the frame builder draws: label segment -> (URDF loader, link name prefix, wrist link, hand link)
ROBOT_MODELS = {"fr3": (_fr3_urdf, "fr3_", "fr3_link7", "fr3_hand"),
                "panda": (_panda_urdf, "panda_", "panda_link7", "panda_hand")}


def _model_of(link):
    for model, (_, prefix, _, _) in ROBOT_MODELS.items():
        if link.startswith(prefix):
            return model
    raise ValueError("No robot model for link " + link)


def _origin(el):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    if el is not None:
        xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
        T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        T[:3, 3] = xyz
    return T


def fixed_transform(parent, child):
    """Transform parent-link -> child-link through fixed URDF joints (e.g. fr3_link7 -> fr3_hand,
    panda_link7 -> panda_link8 -> panda_hand)."""
    _, root = ROBOT_MODELS[_model_of(child)][0]()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    T = np.eye(4)
    link = child
    while link != parent:
        j = by_child[link]
        if j.get("type") != "fixed":
            raise ValueError(f"{link}: joint {j.get('name')} is not fixed")
        T = _origin(j.find("origin")) @ T
        link = j.find("parent").get("link")
    return T


def _load_mesh_file(path, T=np.eye(4)):
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    parts = []
    for node in sc.graph.nodes_geometry:
        M, gname = sc.graph[node]
        g = sc.geometry[gname]
        V = np.asarray(g.vertices, float) @ M[:3, :3].T + M[:3, 3]
        V = V @ T[:3, :3].T + T[:3, 3]
        col = (200, 200, 200)
        mat = getattr(g.visual, "material", None)
        for attr in ("baseColorFactor", "main_color", "diffuse"):
            c = getattr(mat, attr, None) if mat is not None else None
            if c is not None:
                col = tuple(int(v) for v in np.asarray(c)[:3])
                break
        parts.append((V, np.asarray(g.faces, np.int64), col))
    return parts


@lru_cache(maxsize=64)
def robot_link_parts(ref):
    """[(V, F, srgb255)] of a robot link visual in its link frame ('fr3/fr3_link3', 'panda/panda_link3')."""
    model = ref.split("/", 1)[0]
    if model in ROBOT_MODELS:
        link = ref.split("/", 1)[1]
        urdf, root = ROBOT_MODELS[model][0]()
        bundle = urdf.parent.parent
        el = next((l for l in root.findall("link") if l.get("name") == link), None)
        if el is None:
            raise ValueError(f"Unknown {model} link " + link)
        parts = []
        for vis in el.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            fn = m.get("filename")
            if fn.startswith("package://"):
                fn = fn[len("package://"):].split("/", 1)[1]        # package-relative: the bundle is the package root
            parts += _load_mesh_file(bundle/fn, _origin(vis.find("origin")))
        return parts
    if Path(ref).is_file():
        return _load_mesh_file(ref)
    raise ValueError("Unresolvable robot mesh id " + ref)


def fr3_links_from_newton(body_labels, body_q, bases):
    """Frame robot_links from Newton arm bodies (labels like 'left/fr3/fr3_link3' or 'left/panda/panda_link3'),
    adding the two links Newton merges into their parents: link0 (the fixed base, at the mount pose) and the hand
    (link7 * fixed joints). A base is drawn only at a mount that carries an arm."""
    from scipy.spatial.transform import Rotation
    out = []
    deployed = {}
    for label in body_labels:
        bits = label.split("/")
        if len(bits) >= 3 and bits[-2] in ROBOT_MODELS and bits[-1].startswith(ROBOT_MODELS[bits[-2]][1]):
            deployed.setdefault(bits[0], bits[-2])
    for arm, base in bases.items():
        model = deployed.get(arm)
        if model is None:
            continue
        pre = ROBOT_MODELS[model][1]
        yaw = math.radians(base["yaw_deg"])
        out.append(dict(name=f"{arm}/{model}/{pre}link0", mesh=f"{model}/{pre}link0", pos=tuple(base["pos"]),
                        quat_xyzw=(0., 0., math.sin(yaw/2), math.cos(yaw/2))))
    for label, q in zip(body_labels, np.asarray(body_q, float)):
        bits = label.split("/")
        if len(bits) < 3 or bits[-2] not in ROBOT_MODELS:
            continue                                           # tissue, knife, other world bodies
        model, link = bits[-2], bits[-1]
        _, pre, wrist, hand = ROBOT_MODELS[model]
        if not link.startswith(pre):
            continue
        out.append(dict(name=label, mesh=f"{model}/{link}", pos=tuple(q[:3]), quat_xyzw=tuple(q[3:7])))
        if link == wrist:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(q[3:7]).as_matrix()
            T[:3, 3] = q[:3]
            Th = T @ fixed_transform(wrist, hand)
            out.append(dict(name=label.replace(wrist, hand), mesh=f"{model}/{hand}", pos=tuple(Th[:3, 3]),
                            quat_xyzw=tuple(Rotation.from_matrix(Th[:3, :3]).as_quat())))
    return out


# ------------------------------------------------------------------------------------------------ colour / tissue
def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    if c.size and c.max() > 1.5:
        c = c/255.
    return np.where(c <= .04045, c/12.92, ((c + .055)/1.055)**2.4).astype(np.float32)


PRESET_COLOURS = {
    # sRGB, labelled STAND-IN defaults used only when the anatomy does not supply flesh_color. boundary_is_flesh: the
    # workpiece's original non-skin surface is meat (the salmon side's flesh face, the tenderloin's exterior) and is
    # drawn in the anatomy's flesh colour (fat lines, melanin spot, chain, fat); on the gutted trout it is the cavity
    # lining.
    # cavity_lining: the trout's original non-skin boundary is drawn as cut flesh (slit and head faces) or as the wet
    # cavity lining / backbone channel (anatomy classes, render.node_fields); membrane: which skin-membrane faces are a
    # thin membrane drawn over the flesh (render.membrane_faces).
    "A": dict(flesh=(232, 128, 100), dark=(150, 58, 52), fat=(238, 226, 206), cavity=(196, 180, 176),
              septum=(236, 222, 208), phase_per_len=60., septum_width=.07, cavity_lining=True),
    "B": dict(flesh=(242, 118, 72), dark=(165, 70, 50), fat=(245, 232, 212), cavity=(205, 190, 180),
              septum=(246, 236, 222), phase_per_len=28., septum_width=.12, boundary_is_flesh=True),
    "C": dict(flesh=(206, 112, 110), dark=(170, 80, 80), fat=(240, 232, 222), cavity=(200, 180, 180),
              septum=(230, 220, 215), phase_per_len=0., septum_width=0., boundary_is_flesh=True, membrane="skin"),
}

# Material inputs of fish_surfaces.mdl "flesh" per preset and surface (render-only; sRGB, converted to linear).
# fish: wet glossy muscle, per-pixel myosepta (anatomy phase) and flake relief; trout cavity: wet pinkish-grey
# peritoneal lining; pork: fibre grain, fine marbling along the fibres, creamy lobular fat, the silver skin (epimysium)
# as a thin glossy bluish-white translucent sheet with fibres along the muscle, through which the red muscle shows.
TISSUE_LOOK = {
    "A": dict(flesh=dict(septum=(248, 238, 230), septum_width=0., roughness=.30, wet=.75, flake=.10, mottle=.08),
              cavity=dict(septum=(248, 238, 230), septum_width=0., roughness=.26, wet=.8, flake=.08, mottle=.08,
                          aux_mode=2, lining=(168, 124, 120))),
    "B": dict(flesh=dict(septum=(250, 230, 214), septum_width=0., roughness=.30, wet=.72, flake=.10, mottle=.08,
                         septum_strength=.85, septum_wobble=.10),
              cavity=dict(septum=(250, 230, 214), septum_width=0., roughness=.30, wet=.72, flake=.10, mottle=.08,
                          septum_strength=.85, septum_wobble=.10)),
    "C": dict(flesh=dict(septum_width=0., roughness=.30, wet=.75, grain=.26, marbling=.25, fat_cream=1., mottle=.14),
              cavity=dict(septum_width=0., roughness=.32, wet=.7, grain=.20, marbling=.18, fat_cream=1., mottle=.14,
                          aux_mode=1, membrane=(216, 224, 238), membrane_cover=.60, membrane_fibre=.40)),
}


def tissue_look(preset, surface):
    """fish_surfaces.mdl flesh inputs (colours linear) for a preset's surface ('flesh' or 'cavity')."""
    look = dict(TISSUE_LOOK.get(preset, TISSUE_LOOK["A"])[surface])
    for k in ("septum", "lining", "membrane", "fat_colour"):
        if k in look:
            look[k] = tuple(float(v) for v in srgb_to_linear(look[k]))
    return look


def standin_flesh(rest, preset="A"):
    """STAND-IN flesh colour and myomere phase from rest coordinates (used when the anatomy supplies neither):
    orange-pink white muscle, a dark-muscle band at the horizontal septum near the skin, W-shaped myosepta
    (anterior flexure at the horizontal septum, posterior flexures dorsally and ventrally), n myomeres along
    the length (trout ~60), in the anatomy's workpiece frame."""
    c = PRESET_COLOURS[preset]
    R = np.asarray(rest, float)
    lo, hi = R.min(0), R.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    s = (R[:, 1] - (lo[1] + hi[1])/2)/(ext[1]/2)                   # -1 dorsal .. +1 ventral
    zt = (R[:, 2] - lo[2])/ext[2]                                    # 0 board side .. 1 upper flank
    base = np.tile(np.asarray(c["flesh"], float), (len(R), 1))
    dark = np.exp(-(s/.10)**2)*np.clip((np.abs(zt - .5) - .30)/.2, 0, 1)
    base = base*(1 - dark[:, None]) + np.asarray(c["dark"], float)*dark[:, None]
    belly = np.clip((s - .72)/.2, 0, 1)
    base = base*(1 - .6*belly[:, None]) + np.asarray(c["fat"], float)*.6*belly[:, None]
    w = 1. - np.abs(2*np.abs(np.clip(s, -1, 1)) - 1.)               # 0 at s=0, 1 at |s|=.5, 0 at |s|=1
    phase = (R[:, 0] - lo[0])/ext[0]*c["phase_per_len"] + 2.2*w if c["phase_per_len"] else np.zeros(len(R))
    return base/255., phase.astype(np.float32)


def tissue_appearance(workpiece, rest):
    """Appearance fields for frame["tissue"] from an anatomy Workpiece (filleting/anatomy),
    evaluated at the rest coordinates: uv, skin_texture (png path), flesh_color and, if the anatomy exposes it
    as appearance["myomere_phase"], flesh_phase (then the flesh shader draws the sheet lines per pixel; the
    anatomy's flesh_color should then omit them). Merge the result into the tissue dict of each frame."""
    app = workpiece.appearance
    rest = np.asarray(rest, float)
    out = dict(uv=np.asarray(app["skin_uv"](rest), np.float32), flesh_color=np.asarray(app["flesh_color"](rest), np.float32))
    tex = app.get("skin_texture")
    if isinstance(tex, dict) and tex.get("path"):
        out["skin_texture"] = str(tex["path"])
    if callable(app.get("myomere_phase")):
        out["flesh_phase"] = np.asarray(app["myomere_phase"](rest), np.float32)
    return out


def weld_key(P):
    """Index of each point's position class (1 um grid; 21 bits per axis = 2.1 m range)."""
    P = np.asarray(P, np.float64)
    if not len(P):
        return np.zeros(0, np.int64)
    q = np.round(P/1e-6).astype(np.int64)
    k = (q[:, 0] - q[:, 0].min())*(1 << 42) + (q[:, 1] - q[:, 1].min())*(1 << 21) + (q[:, 2] - q[:, 2].min())
    _, key = np.unique(k, return_inverse=True)
    return key.ravel()


def vertex_normals(P, T, weld=True, key=None):
    """Area-weighted vertex normals over triangles T; with `weld`, vertices at the same position (UV seams,
    freshly split cut nodes) share one normal so seams do not shade."""
    P = np.asarray(P, np.float64)
    if key is None:
        key = weld_key(P) if (weld and len(P)) else np.arange(len(P))
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    vn = np.zeros((key.max() + 1 if len(key) else 0, 3))
    for k in range(3):
        np.add.at(vn, key[T[:, k]], fn)
    vn = vn[key]
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)


# ------------------------------------------------------------------------------------------------ anatomy-derived looks
# RENDER-ONLY appearance fields derived from the anatomy generator (read-only use of filleting/anatomy; nothing here
# touches the world). A frame carries each node's rest position and the anatomy's flesh_color(rest) but not the
# generator, so the generator is rebuilt from the seed in the skin-texture name (frames.static_cache writes
# "<workpiece>_skin_seed<k>.png") and used ONLY if it reproduces the recorded flesh_color at sampled nodes; otherwise
# the anatomy's colours are drawn as recorded (per vertex) and no lines are added.
#   phase     the anatomy's own myomere phase (trout Myomeres.phase, salmon _phase_phys: the same W-myomere field the
#             tissue's sheet normals come from), so the per-pixel sheet lines lie where the simulated sheets are
#   colour    the anatomy draws its 0.2-0.9 mm septum lines and 1-vertex fat flecks INTO flesh_color; sampled at 3.5 mm
#             vertices they become scattered white dots. Nodes on a septum line take the
#             anatomy's colour just off the line (the line is then drawn per pixel); isolated pork fat flecks are
#             removed by a 5-point median of the anatomy's colour field (fine marbling is drawn per pixel instead).
#             Tenderloin surface nodes sit exactly on region boundaries (the flake-fat lumps' surface has d_fat = 0, so
#             the anatomy's d_fat < 0 test flips from vertex to vertex: red triangles on white fat); their colour is
#             sampled 0.35 mm under the surface, i.e. the material the surface belongs to
#   classes   trout cavity surfaces: the cavity lining and backbone channel are drawn as the wet lining; the slit faces
#             and the head face are cut flesh
SEED_IN_TEXTURE = re.compile(r"_seed(\d+)\.png$")
SEPTUM_HALF_WIDTH = {"A": (.20e-3, 0.), "B": (.28e-3, .32e-3)}  # m, render-only: base + extra x belly paleness
TAINT_DISTANCE = {"A": .20e-3, "B": .90e-3}                       # m: covers the anatomy's per-vertex line widths
_NODE_CACHE = {}
ANATOMY_CHECK = {}       # (preset, seed) -> max |anatomy flesh_color - recorded| at the sampled nodes, or the error


@lru_cache(maxsize=6)
def _anatomy(preset, seed):
    """Functions of rest positions (workpiece frame) from the preset's anatomy generator, or None."""
    if preset == "A":
        from .anatomy import trout as T
        tr = T.Trout(seed)
        loc = T._frame_maps(tr)[0]

        def classes(P):
            s, v, w = loc(P)
            d_out = tr.outer(s, v, w)
            lining = (np.abs(tr.cavity(s, v, w, d_out)) < 1e-3) & (np.abs(tr.slit(s, v, w)) >= 1e-3)
            return (lining | (np.abs(tr.channel(s, v, w)) < 1e-3)).astype(np.float32)
        return dict(colour=lambda P: T._trout_flesh(tr, *loc(P)), phase=lambda P: tr.myo.phase(*loc(P)),
                    pale=lambda P: np.zeros(len(P)), classes=classes, despeckle=False, model=tr)
    if preset == "B":
        from .anatomy import salmon as S
        sm = S.Salmon(seed)
        return dict(colour=lambda P: S._salmon_flesh(sm, *np.asarray(P, float).T),
                    phase=lambda P: sm._phase_phys(*np.asarray(P, float).T),
                    pale=lambda P: np.clip((sm.u_of(P[:, 0], P[:, 1]) - .45)/.5, 0, 1), classes=None, despeckle=False)
    if preset == "C":
        from .anatomy import tenderloin as TL
        tl = TL.Tenderloin(seed)
        return dict(colour=lambda P: TL._pork_flesh(tl, *np.asarray(P, float).T), phase=None, pale=None, classes=None,
                    despeckle=True, sdf=lambda P: tl.tissue(*np.asarray(P, float).T), under=.35e-3)
    return None


def _validated_anatomy(preset, seed, rest, srgb):
    if seed is None:
        return None
    try:
        an = _anatomy(preset, seed)
        if an is None or not len(rest):
            return None
        i = np.unique(np.linspace(0, len(rest) - 1, 192).astype(np.int64))
        err = float(np.abs(np.asarray(an["colour"](rest[i]), float) - srgb[i]).max())
    except Exception as exc:                        # no generator for this workpiece: draw the recorded colours
        ANATOMY_CHECK[(preset, seed)] = repr(exc)[:200]
        return None
    ANATOMY_CHECK[(preset, seed)] = err
    return an if err < 2e-3 else None


def _node_looks(an, preset, R, C):
    """(srgb, phase, half-width in phase units, classes) for nodes at rest positions R with recorded colours C."""
    n = len(R)
    C = C.copy()
    phase, hw, cls = np.full(n, .5), np.zeros(n), np.zeros(n, np.float32)
    if an is None:
        return C, phase, hw, cls
    if an["phase"] is not None and n:
        f = an["phase"]
        phase = np.asarray(f(R), float)
        e = 1e-4
        g = np.stack([(f(R + d) - f(R - d))/(2*e) for d in np.eye(3)*e], 1)
        gn = np.maximum(np.linalg.norm(g, axis=1), 1e-9)
        base, extra = SEPTUM_HALF_WIDTH[preset]
        hw = (base + extra*an["pale"](R))*gn                         # m -> phase units
        off = phase - np.round(phase)
        dist = np.abs(off)/gn
        t = dist < TAINT_DISTANCE[preset]
        if t.any():
            step = (np.where(off[t] >= 0, 1., -1.)*(TAINT_DISTANCE[preset] + .15e-3 - dist[t]))[:, None]   # m
            C[t] = an["colour"](R[t] + g[t]/gn[t][:, None]*step)
    Rs = R
    if an.get("under") and n:                                    # surface nodes: the material just under the surface
        f = an["sdf"]
        d = np.asarray(f(R), float)
        near = np.abs(d) < 1e-3
        if near.any():
            e = 1e-4
            g = np.stack([(f(R[near] + dd) - f(R[near] - dd))/(2*e) for dd in np.eye(3)*e], 1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            Rs = R.copy()
            Rs[near] = R[near] - g*(d[near] + an["under"])[:, None]
            C[near] = an["colour"](Rs[near])
    if an["despeckle"] and n:
        o = .35e-3
        samples = [C] + [np.asarray(an["colour"](Rs + d), float) for d in ((o, 0, 0), (-o, 0, 0), (0, o, 0), (0, -o, 0))]
        C = np.median(np.stack(samples), 0)
    if an["classes"] is not None and n:
        cls = an["classes"](R)
    return C, phase, hw, cls


def node_fields(tissue, preset="A"):
    """Per-node render fields of a frame's tissue: srgb (N,3), phase (N,), hw (N,) septum half-width in phase units
    (0: the material's septum_width), cls (N,) cavity-lining weight, anatomy (validated or not).
    Cached per workpiece: rest positions never change and nodes added by cuts are evaluated once."""
    rest = np.asarray(tissue.get("rest", tissue["points"]), np.float64)
    n = len(rest)
    if tissue.get("flesh_color") is None:
        col, phase = standin_flesh(rest, preset)
        return dict(srgb=col, phase=phase, hw=np.zeros(n), cls=np.zeros(n, np.float32), anatomy=False)
    col = np.asarray(tissue["flesh_color"], np.float64)
    col = col/255. if col.max() > 1.5 else col
    m = SEED_IN_TEXTURE.search(str(tissue.get("skin_texture") or ""))
    key = (preset, int(m.group(1)) if m else None, str(tissue.get("skin_texture")))
    ent = _NODE_CACHE.get(key)
    if ent is not None:
        k = min(ent["n"], n)
        probe = np.unique(np.linspace(0, k - 1, 16).astype(np.int64)) if k else np.zeros(0, np.int64)
        if not (np.array_equal(ent["rest"][probe], rest[probe]) and np.array_equal(ent["col"][probe], col[probe])):
            ent = None
    if ent is None:
        an = _validated_anatomy(preset, key[1], rest, col)
        ent = dict(n=0, an=an, rest=np.zeros((0, 3)), col=np.zeros((0, 3)), srgb=np.zeros((0, 3)), phase=np.zeros(0),
                   hw=np.zeros(0), cls=np.zeros(0, np.float32))
        _NODE_CACHE.clear()
        _NODE_CACHE[key] = ent
    if n > ent["n"]:
        R, C = rest[ent["n"]:n], col[ent["n"]:n]
        s, ph, hw, cl = _node_looks(ent["an"], preset, R, C)
        for k_, v in (("rest", R), ("col", C), ("srgb", s), ("phase", ph), ("hw", hw), ("cls", cl)):
            ent[k_] = np.concatenate([ent[k_], v])
        ent["n"] = n
    out = dict(srgb=ent["srgb"][:n], phase=ent["phase"][:n], hw=ent["hw"][:n], cls=ent["cls"][:n],
               anatomy=ent["an"] is not None)
    if tissue.get("flesh_phase") is not None:                    # the anatomy's own hook wins (material line width)
        out["phase"], out["hw"] = np.asarray(tissue["flesh_phase"], np.float64), np.zeros(n)
    return out


def trout_head_texture(skin_texture, seed, out_dir):
    """RENDER-ONLY head shading of the trout (without it the head reads as a smooth blob without gill cover or jaw):
    the gill-cover (opercular) margin as a fine dark line with a soft shadow on the body behind it, a slightly darker,
    warmer operculum, the preopercular groove and the maxilla (upper-jaw bone) outline above the gape,
    painted into a copy of the anatomy's own skin texture at the anatomy's landmarks (gill cover 25-57 mm and its rosy
    ellipse centred at 44/-6 mm, eye at 16/1.5 mm; photo mm, anatomy/trout.py). Shading only: the head's geometry (the
    head body's SDF surface) is unchanged. Returns the new texture path, or the original when unavailable."""
    from PIL import Image
    an = _anatomy("A", seed) if seed is not None else None
    src = Path(str(skin_texture))
    if an is None or not src.is_file():
        return skin_texture
    tr = an["model"]
    out = Path(out_dir)/(src.stem + "_head_marks.png")
    img = np.asarray(Image.open(src).convert("RGB"), np.float64)/255.
    H, W = img.shape[:2]
    k = tr.k
    ncol = min(W, int(np.ceil(W*90*k/tr.s_end)))                   # the head and gill region, s < 90 photo mm
    th = (np.arange(H) + .5)/H*2*np.pi
    S = (np.arange(ncol) + .5)/W*tr.s_end
    TH, SS = np.meshgrid(th, S, indexing="ij")
    v, _ = tr.loft.surface(SS, TH)
    S_, V_ = SS/k, v/k
    col = img[:, :ncol]

    def arc(sc, vc, a, b, vmin, vmax):          # distance along s to the posterior half of an ellipse
        d = S_ - (sc + a*np.sqrt(np.clip(1 - ((V_ - vc)/b)**2, 0, 1)))
        fade = np.clip(np.minimum(V_ - vmin, vmax - V_)/3., 0, 1)
        return d, fade
    d, fade = arc(44., -6., 13., 16., -21.5, 9.5)                    # opercular margin
    col *= (1 - .42*np.clip(1 - np.abs(d)/.45, 0, 1)*fade)[..., None]
    col *= (1 - .14*np.clip(1 - np.clip(d, 0, None)/2.2, 0, 1)*(d > 0)*fade)[..., None]
    op = np.clip((1 - ((S_ - 41.)/14.)**2 - ((V_ + 6.)/16.)**2)*3, 0, 1)*(d < 0)*(fade > 0)
    col[:] = col*(1 - .10*op)[..., None] + np.array([.62, .42, .40])*(.08*op)[..., None]
    d2, fade2 = arc(30., -9., 6., 12.5, -20., 2.5)                   # preopercular groove
    col *= (1 - .20*np.clip(1 - np.abs(d2)/.40, 0, 1)*fade2)[..., None]
    t = (S_ - 2.)/25.                                                # maxilla, snout to below the back of the eye
    inside = (t > 0) & (t < 1)
    dm = np.abs(V_ - (-4.2 - 4.8*np.clip(t, 0, 1))) - 2.3*np.sin(np.pi*np.clip(t, 0, 1))**.7
    col *= (1 - .22*np.clip(1 - np.abs(dm)/.40, 0, 1)*inside)[..., None]
    col *= (1 - .08*np.clip(-dm/.8, 0, 1)*inside)[..., None]
    img[:, :ncol] = col
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1)*255 + .5).astype(np.uint8)).save(out)
    return str(out)


def membrane_faces(T, K, preset):
    """Boundary faces drawn as a thin membrane over the flesh: the tenderloin's silver skin (its skin-membrane faces,
    the physics' silver-skin layer). The salmon side's belly peritoneum (anatomy membrane_mask) is not built by the
    world, so none is drawn."""
    if PRESET_COLOURS[preset].get("membrane") == "skin":
        return K == 0
    return np.zeros(len(T), bool)


def cover_mask(T, K, top, rest, n, passes=2):
    """Per-node cover 0..1 of the faces `top` (a membrane: skin or silver skin) over the original non-skin boundary:
    the area fraction of `top` faces among the skin + original-boundary faces (kinds 0 and 2) around a node, then
    smoothed over that surface graph. The shader cuts it at ~0.5, so the edge between a membrane and the surface next to
    it is a smooth curve through the mesh instead of the per-triangle saw-tooth of the face kinds. Knife cuts (kind 1)
    take no part: the edge of a cut face stays exactly where the cut is."""
    m = np.zeros(n)
    surf = (K == 0) | (K == 2)
    if not top.any():
        return m
    R = rest[T]
    area = .5*np.linalg.norm(np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]), axis=1)
    a_top = np.bincount(T[top].ravel(), weights=np.repeat(area[top], 3), minlength=n)
    a_all = np.bincount(T[surf].ravel(), weights=np.repeat(area[surf], 3), minlength=n)
    m = np.where(a_all > 0, a_top/np.maximum(a_all, 1e-30), 0.)
    Ts = T[surf]
    E = np.concatenate([Ts[:, [0, 1]], Ts[:, [1, 2]], Ts[:, [2, 0]]])
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    for _ in range(passes):
        nb = np.bincount(E[:, 0], weights=m[E[:, 1]], minlength=n) + np.bincount(E[:, 1], weights=m[E[:, 0]], minlength=n)
        m = np.where(deg > 0, .5*m + .5*nb/np.maximum(deg, 1), m)
    return m


def tissue_fields(tissue, preset="A"):
    """{surface name: per-corner fields}. skin: the skin membrane (st); flesh: knife cuts and interface splits; cavity:
    the original non-skin boundary (trout cavity lining, slit and head faces; salmon flesh face; tenderloin exterior),
    the tenderloin's silver skin, and the band of skin faces along the skin's edge with that boundary (drawn from the
    same skin texture, blended by cover_mask); fat: kind 3. Knife cuts (and kind 3) keep normals smoothed within their
    own kind, so the edge between skin and a cut face stays sharp; the skin and the original boundary (the silver skin
    and the meat it lies on) share smooth normals."""
    P = np.asarray(tissue["points"], np.float64)
    T = np.asarray(tissue["tris"], np.int64).reshape(-1, 3)
    K = np.asarray(tissue["tri_kind"], np.int64).ravel()
    if len(K) != len(T):
        raise ValueError("tri_kind must have one entry per triangle")
    n = len(P)
    rest = np.asarray(tissue.get("rest", P), np.float64)
    uv = np.asarray(tissue.get("uv", np.zeros((n, 2))), np.float32)
    nf = node_fields(tissue, preset)
    lin = srgb_to_linear(nf["srgb"])
    pc = PRESET_COLOURS[preset]
    mem = membrane_faces(T, K, preset)
    if mem.any():                                   # silver skin over the meat
        cover, edge = cover_mask(T, K, mem, rest, n), mem
    else:                                           # skin next to the original boundary (trout slit, salmon sides)
        cover = cover_mask(T, K, K == 0, rest, n)
        edge = (K == 0) & (cover[T].min(1) < .995)
    # normals: smooth over the skin and the original boundary together (the tet faces along e.g. the trout's slit rim
    # zig-zag at mesh scale, and per-kind normals lit every tooth separately); knife cuts keep their own (sharp) edge
    groups = ((0, 2), (1,), (3,))
    cls = nf["cls"]
    if pc.get("cavity_lining") and not nf["anatomy"] and tissue.get("skin_texture"):
        cls = np.ones(n, np.float32)                # a fish whose anatomy was not validated: all of it as lining
                                                    # (the coupon, a plain muscle block without skin, stays flesh)
    key = weld_key(P)
    CN = np.zeros((len(T), 3, 3))
    for g in groups:
        sel = np.isin(K, g)
        if sel.any():
            CN[sel] = vertex_normals(P, T[sel], key=key)[T[sel]]
    # skin UV per corner with the around-body seam (v) unwrapped per triangle
    ST = uv[T].copy()
    span = ST[:, :, 1].max(1) - ST[:, :, 1].min(1)
    w = span > .5
    ST[w, :, 1] = np.where(ST[w, :, 1] < .5, ST[w, :, 1] + 1., ST[w, :, 1])
    roots = {"skin": (K == 0) & ~edge, "flesh": K == 1, "cavity": (K == 2) | edge, "fat": K == 3}
    out = {}
    for name, sel in roots.items():
        Tk = T[sel]
        idx = Tk.ravel()
        f = dict(points=P[idx], normals=CN[sel].reshape(-1, 3))
        if name == "skin":
            f["st"] = ST[sel].reshape(-1, 2)
        else:
            f["color"] = lin[idx]
            f["myophase"] = nf["phase"][idx]
            aux = np.zeros((len(idx), 3))
            aux[:, 0] = nf["hw"][idx]
            if name == "cavity":
                if mem.any():
                    aux[:, 1] = cover[idx]              # silver-skin cover
                else:
                    aux[:, 1] = cls[idx] if pc.get("cavity_lining") else 0.
                    aux[:, 2] = cover[idx]              # skin cover (the skin texture at st)
                f["st"] = ST[sel].reshape(-1, 2)
            f["aux"] = aux
            f["restpos"] = rest[idx]
        width = dict(points=3, normals=3, st=2, color=3, aux=3, restpos=3)
        out[name] = {k: np.asarray(v, np.float32).reshape(len(idx), width[k]) if k in width else
                     np.asarray(v, np.float32).ravel() for k, v in f.items()}
    return out


# ------------------------------------------------------------------------------------------------ fins
# Render-only fin membranes (frames.py): per vertex, a ray coordinate (bony rays fan out from the fin base; ray counts
# from anatomy/trout.py), the distance to the free margin (the membrane recedes between the ray tips and fades out over
# the last millimetre: soft, irregular margins instead of a hard cut polygon) and the distance from the base (pigment
# and opacity gradient). Recordings made before the fin names were recorded carry the trout's eight fins in the
# anatomy's order.
TROUT_FINS = ("dorsal", "adipose", "anal", "caudal", "pectoral_upper", "pectoral_lower", "pelvic_upper", "pelvic_lower")
FIN_RAYS = {"dorsal": 16, "anal": 15, "caudal": 19, "pectoral": 13, "pelvic": 9, "adipose": 0}   # anatomy/trout.py
FIN_KIND = {"dorsal": 0, "caudal": 0, "adipose": 1, "anal": 2, "pectoral": 2, "pelvic": 2}      # fish_surfaces.mdl fin
_FIN_CACHE = {}


def _fin_coords(R, T, uv, name):
    """(ray coordinate, distance to the free margin (m), distance from the base (m)) per vertex of one fin."""
    base_name = str(name).split("_")[0]
    n_rays = FIN_RAYS.get(base_name, 14)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    B = Eu[cnt == 1]
    bv = np.unique(B)
    if base_name in ("pectoral", "pelvic"):
        base = bv[np.linalg.norm(R[bv] - R[0], axis=1) < 4.5e-3]       # paired fins: outline point 0 is the base
    else:
        base = bv[uv[bv, 1] < .05]                                    # median fins: v = 0 on the attachment line
    if len(base) < 2:
        base = bv[np.argsort(np.linalg.norm(R[bv] - R[0], axis=1))[:2]]
    free = B[~(np.isin(B[:, 0], base) & np.isin(B[:, 1], base))]
    A0, D = R[free[:, 0]], R[free[:, 1]] - R[free[:, 0]]
    t = np.clip(np.einsum("vej,ej->ve", R[:, None] - A0[None], D)/np.maximum((D*D).sum(1), 1e-18)[None], 0, 1)
    dm = np.linalg.norm(R[:, None] - (A0[None] + t[..., None]*D[None]), axis=-1).min(1)
    db = np.linalg.norm(R[:, None] - R[base][None], axis=-1).min(1)
    c = R.mean(0)
    e = np.linalg.svd(R - c)[2][:2]
    bm = R[base].mean(0)
    blen = max(float(np.linalg.norm(R[base].max(0) - R[base].min(0))), 1e-3)
    out = (bm - c)/max(float(np.linalg.norm(bm - c)), 1e-9)
    O = bm + out*{"caudal": .8*blen, "pectoral": 3e-3, "pelvic": 3e-3}.get(base_name, 2.2*blen)
    q = (R - O) @ e.T
    cd = (c - O) @ e.T
    cd /= max(float(np.linalg.norm(cd)), 1e-12)
    ang = np.arctan2(cd[0]*q[:, 1] - cd[1]*q[:, 0], q @ cd)
    span = max(float(np.ptp(ang)), 1e-6)
    ray = (ang - ang.min())/span*max(n_rays - 1, 1) if n_rays else np.full(len(R), .5)
    return ray, dm, db, FIN_KIND.get(base_name, 0)


def fin_fields(fins):
    """Fin membranes (frame["fins"]: points, tris, uv; optional rest, groups [(v0, v1)], names) per corner: points,
    normals, st = (ray coordinate, distance to the free margin in mm), aux = (distance from the base 0..1, fin kind,
    distance from the base in mm)."""
    if not fins or not len(fins.get("tris", ())):
        z3 = np.zeros((0, 3), np.float32)
        return dict(points=z3, normals=z3, st=np.zeros((0, 2), np.float32), aux=z3)
    P = np.asarray(fins["points"], np.float64)
    T = np.asarray(fins["tris"], np.int64)
    uv = np.asarray(fins["uv"], np.float64)
    R = np.asarray(fins.get("rest", P), np.float64)
    groups = fins.get("groups")
    if groups is None:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]]])
        _, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(P), len(P))), False)
        groups = [(int(np.nonzero(lab == k)[0].min()), int(np.nonzero(lab == k)[0].max()) + 1) for k in np.unique(lab)]
    groups = [(int(g[0]), int(g[1])) for g in groups]
    names = fins.get("names") or (TROUT_FINS if len(groups) == len(TROUT_FINS) else ("fin",)*len(groups))
    key = (len(R), R[0].tobytes(), R[-1].tobytes(), tuple(groups))
    if key not in _FIN_CACHE:
        st, aux = np.zeros((len(P), 2)), np.zeros((len(P), 3))
        for (a, b), name in zip(groups, names):
            sel = (T[:, 0] >= a) & (T[:, 0] < b)
            ray, dm, db, kind = _fin_coords(R[a:b], T[sel] - a, uv[a:b], name)
            st[a:b] = np.column_stack([ray, dm*1e3])
            aux[a:b] = np.column_stack([db/max(db.max(), 1e-9), np.full(b - a, kind), db*1e3])
        _FIN_CACHE.clear()
        _FIN_CACHE[key] = (st, aux)
    st, aux = _FIN_CACHE[key]
    vn = vertex_normals(P, T, weld=False)
    idx = T.ravel()
    return dict(points=P[idx].astype(np.float32), normals=vn[idx].astype(np.float32), st=st[idx].astype(np.float32),
                aux=aux[idx].astype(np.float32))


def rod_fields(rods, sides=8):
    """Bone rods (ribs, pin bones) as tube triangles, per corner."""
    from .dressing import tube
    Ps, Ns = [], []
    for r in rods or []:
        pts = np.asarray(r["points"], float)
        if len(pts) < 2:
            continue
        V, F = tube(pts, np.asarray(r["radius"], float), n=sides, caps=True)
        vn = vertex_normals(V, F)
        Ps.append(V[F.ravel()])
        Ns.append(vn[F.ravel()])
    if not Ps:
        return dict(points=np.zeros((0, 3), np.float32), normals=np.zeros((0, 3), np.float32))
    return dict(points=np.concatenate(Ps).astype(np.float32), normals=np.concatenate(Ns).astype(np.float32))


# ------------------------------------------------------------------------------------------------ export
def _camera(stage, path, eye, target, hfov, role, up=(0., 0., 1.)):
    from pxr import Gf, UsdGeom
    c = UsdGeom.Camera.Define(stage, path)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up)))
    c.AddTransformOp().Set(m.GetInverse())
    c.CreateHorizontalApertureAttr(36.)
    c.CreateVerticalApertureAttr(20.25)
    c.CreateFocalLengthAttr(18./math.tan(math.radians(hfov)/2))
    c.CreateClippingRangeAttr(Gf.Vec2f(.005, 40.))
    c.GetPrim().SetCustomDataByKey("observation_role", role)
    c.GetPrim().SetCustomDataByKey("horizontal_fov_deg", float(hfov))
    return c


def _pose(xf, pos, quat):
    from pxr import Gf, UsdGeom
    x = UsdGeom.Xformable(xf)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))


def _author_mesh_id(d, path, mesh, preset):
    """Author the visual geometry of a body mesh id under `path` (local frame)."""
    from . import dressing as D
    if isinstance(mesh, dict) and "parts" in mesh:
        for k, part in enumerate(mesh["parts"]):
            if len(part.get("tris", ())):
                _author_mesh_id(d, f"{path}/part_{k}", part, preset)
        return
    if isinstance(mesh, dict) and "boxes" in mesh:
        from scipy.spatial.transform import Rotation
        parts = []
        for p_, q_, h_ in mesh["boxes"]:
            V, F = D.rbox_mesh(2*np.asarray(h_, float), min(.0015, float(np.min(h_))*.4))
            parts.append((V @ Rotation.from_quat(q_).as_matrix().T + np.asarray(p_, float), F))
        V, F = D.merge(*parts)
        mesh = dict(points=V, tris=F, material=mesh.get("material", "insert"))
    if isinstance(mesh, dict):
        mats = {"bone": lambda: d.preview("bone", (232, 226, 206), .45), "steel": lambda: d.steel(),
                "skin": lambda: d.mats.get("tissue_skin"), "fin": lambda: d.preview("trout_fin", texture=d.tex["trout_fin"], rough=.35, coat=.6, opacity=.8),
                "eye": lambda: d.preview("trout_eye", (190, 170, 120), .15, metal=.6),
                "pupil": lambda: d.preview("trout_pupil", (5, 5, 6), .05, coat=1.),
                "insert": lambda: d.preview("insert_tpu", (48, 50, 54), .6),
                "flesh_face": lambda: d.preview("head_flesh_face", (196, 128, 112), .35, coat=.5)}
        mat = mats.get(mesh.get("material", "bone"), mats["bone"])()
        if "capsules" in mesh:
            parts = [D.tube([p0, p1], r, n=16) for p0, p1, r in mesh["capsules"]]
            V, F = D.merge(*parts)
        else:
            V, F = np.asarray(mesh["points"], float), np.asarray(mesh["tris"], np.int64)
        uv = mesh.get("uv")
        d.mesh(path + "/geom", V, F, mat, angle=50., double=mesh.get("material") in ("fin", "skin", "flesh_face"),
               uv=None if uv is None else np.asarray(uv, float))
        return
    kind = str(mesh)
    if kind.startswith("knife:"):
        bits = kind.split(":")
        D.build_knife(d, path, bits[1], bits[2] if len(bits) > 2 else "whole")
    elif kind.startswith("board:"):
        D.build_board(d, path, kind.split(":")[1])
    elif kind == "tweezers":
        D.build_tweezers(d, path)
    elif kind == "bowl":
        V, F = D.bowl_geometry()
        d.mesh(path + "/bowl", V, F, d.steel("bowl_steel", .12, srgb=(196, 198, 200)), angle=40.)
    elif kind == "fillet_tray":
        V, F, floor = D.gn_pan_geometry(L.FILLET_TRAY["size"][:2], L.FILLET_TRAY["size"][2], t=L.FILLET_TRAY["wall"])
        d.mesh(path + "/walls", V, F[~floor], d.steel("pan_steel", .22), angle=35.)
        perf = d.preview("perforated_steel", texture=d.tex["perforation"], alpha_cut=True, rough=.25, metal=1.)
        d.mesh(path + "/floor", V, F[floor], perf, uv=D.box_uv(V, F[floor])/.09, normals=None, double=True)
    elif kind == "knife_block":
        D.build_knife_block(d, path, kinds=())
    elif kind == "scale":
        D.build_scale(d, path)
    elif kind == "tweezer_tray":
        D.build_tweezer_tray(d, path)
    elif kind.split("/", 1)[0] in ROBOT_MODELS or Path(kind).is_file():
        _author_robot_link(d, path, kind)
    else:
        raise ValueError("Unknown mesh id " + kind)


def _author_robot_link(d, path, ref):
    from . import dressing as D
    for k, (V, F, col) in enumerate(robot_link_parts(ref)):
        lum = np.mean(col)
        if max(col) - min(col) > 60:                       # coloured part (status LEDs, red base ring)
            m = d.preview("robot_" + "_".join(map(str, col)), col, .35)
        elif lum > 200:
            m = d.preview("robot_white", (228, 230, 229), .33)
        elif lum > 40:
            m = d.preview("robot_grey", (70, 72, 74), .4)
        else:
            m = d.preview("robot_dark", (30, 33, 36), .42)
        d.mesh(f"{path}/part_{k}", V, F, m, angle=38.)


def _declare_paged(stage, root, fields, material, page_tris=PAGE_TRIANGLES):
    from pxr import UsdGeom, UsdShade
    x = UsdGeom.Xform.Define(stage, root)
    x.GetPrim().SetCustomDataByKey("page_triangles", int(page_tris))
    x.GetPrim().SetCustomDataByKey("fields", ",".join(fields))
    UsdShade.MaterialBindingAPI.Apply(x.GetPrim()).Bind(material)
    return x


def _declare_reauthor(stage, path, fields, material):
    from pxr import Sdf, UsdGeom, UsdShade
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDoubleSidedAttr(True)
    g.CreatePointsAttr()
    g.CreateNormalsAttr()
    g.SetNormalsInterpolation("vertex")
    g.CreateExtentAttr()
    g.CreateFaceVertexCountsAttr()
    g.CreateFaceVertexIndicesAttr()
    g.CreateVisibilityAttr("invisible")
    pv = UsdGeom.PrimvarsAPI(g)
    for k in fields:
        if k == "st":
            pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        elif k == "color":
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        elif k == "myophase":
            pv.CreatePrimvar("myophase", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
        elif k in ("aux", "restpos"):
            pv.CreatePrimvar(k, Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    return g


def _apply_local(stage, updates):
    from pxr import Sdf
    with Sdf.ChangeBlock():
        for spec, value in updates:
            spec.default = value


def export_scene(folder, frame, preset="A", *, page_triangles=PAGE_TRIANGLES, extra_cameras=None):
    """Static render stage (dressing + robot and body visuals + tissue/rod roots + cameras) posed at `frame`.
    Writes folder/scene.usdc and folder/export.json (the body-index mapping). Returns the scene path."""
    from pxr import Gf, Usd, UsdGeom
    from . import dressing as D
    from .render_worker import paged_updates
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    bodies = frame.get("bodies", {}) or {}
    skip = {n for n in bodies if n in DRESSING_ITEMS}
    info = D.dress(stage, folder, preset, skip=skip, task=frame.get("task"))
    d = info.pop("dresser")
    # rigid bodies: robot links first, then named bodies in sorted order -> /Lab/body_i
    links = frame.get("robot_links", [])
    names = [l["name"] for l in links] + sorted(bodies)
    for i, l in enumerate(links):
        p = f"/Lab/body_{i}"
        x = UsdGeom.Xform.Define(stage, p)
        _pose(x, l["pos"], l["quat_xyzw"])
        x.GetPrim().SetCustomDataByKey("gar_name", l["name"])
        _author_robot_link(d, p, l["mesh"])
    # tissue materials before bodies (bodies may use the skin material)
    tis = frame.get("tissue")
    skin_tex = (tis or {}).get("skin_texture") or d.tex["trout_skin"]
    if preset == "A" and tis is not None and tis.get("flesh_color") is not None and node_fields(tis, preset)["anatomy"]:
        m = SEED_IN_TEXTURE.search(str(tis.get("skin_texture") or ""))
        skin_tex = trout_head_texture(skin_tex, int(m.group(1)), folder/"textures")
    pc = PRESET_COLOURS[preset]
    flesh_look = tissue_look(preset, "flesh")
    if tis is not None and tis.get("flesh_color") is None:          # stand-in colours: lines of the stand-in phase
        flesh_look.update(septum=tuple(float(v) for v in srgb_to_linear(pc["septum"])), septum_width=pc["septum_width"])
    mats = dict(skin=d.tissue_material("tissue_skin", "skin", skin_texture=skin_tex),
                flesh=d.tissue_material("tissue_flesh", "flesh", **flesh_look),
                cavity=d.tissue_material("tissue_cavity", "flesh", **tissue_look(preset, "cavity"),
                                         **({} if pc.get("membrane") else dict(skin_texture=skin_tex, skin_blend=1.))),
                fat=d.tissue_material("tissue_fat", "flesh", **dict(flesh_look, roughness=.3, wet=.7)),
                fin=d.tissue_material("tissue_fin", "fin"),
                bone=d.preview("bone", (232, 226, 206), .45))
    for j, name in enumerate(sorted(bodies)):
        i = len(links) + j
        b = bodies[name]
        p = f"/Lab/body_{i}"
        x = UsdGeom.Xform.Define(stage, p)
        _pose(x, b["pos"], b["quat_xyzw"])
        x.GetPrim().SetCustomDataByKey("gar_name", name)
        _author_mesh_id(d, p, b["mesh"], preset)
    # wrist D405s on each arm's link7
    wrist = {}
    for i, l in enumerate(links):
        if any(l["name"].endswith(m[2]) for m in ROBOT_MODELS.values()):
            arm = l["name"].split("/")[0] if "/" in l["name"] else ("left" if not wrist else "right")
            p = f"/Lab/body_{i}/d405"
            UsdGeom.Xform.Define(stage, p)
            dark = d.preview("d405_dark", (28, 29, 31), .4)
            alu = d.steel("d405_bracket", .3, srgb=(160, 162, 165))
            d.box(p + "/case", D405["case"][0], D405["case"][1], dark, r=.004)
            d.box(p + "/bracket", D405["bracket"][0], D405["bracket"][1], alu, r=.001)
            for k, dy in enumerate((-.012, .012)):
                d.cyl(p + f"/lens_{k}", (.075, dy, .0815), (.075, dy, .0825), .004, d.preview("lens", (5, 5, 8), .05))
            cam = _camera(stage, p + "/camera", D405["eye"], D405["target"], D405["hfov"], "policy", up=D405["up"])
            alias = UsdGeom.Xform.Define(stage, f"/Cameras/{arm}_wrist_rgb")
            alias.GetPrim().SetCustomDataByKey("mounted_camera_path", str(cam.GetPath()))
            wrist[f"{arm}_wrist_rgb"] = str(cam.GetPath())
    # tissue and rods: paged roots (+ re-author meshes, invisible unless that mode is used)
    for parent in ("/Lab/tissue", "/Lab/tissue_re", "/Lab/rods", "/Lab/rods_re"):
        UsdGeom.Xform.Define(stage, parent)
    tissue_extra = ("color", "myophase", "aux", "restpos")
    kinds = {"skin": ("st",), "flesh": tissue_extra, "cavity": tissue_extra + ("st",), "fat": tissue_extra}
    for name, extra in kinds.items():
        _declare_paged(stage, f"/Lab/tissue/{name}", extra, mats[name], page_triangles)
        _declare_reauthor(stage, f"/Lab/tissue_re/{name}", extra, mats[name])
    _declare_paged(stage, "/Lab/rods/bone", (), mats["bone"], 2048)
    _declare_paged(stage, "/Lab/tissue/fin", ("st", "aux"), mats["fin"], 2048)
    _declare_reauthor(stage, "/Lab/tissue_re/fin", ("st", "aux"), mats["fin"])
    _declare_reauthor(stage, "/Lab/rods_re/bone", (), mats["bone"])
    if tis is not None:
        f = tissue_fields(tis, preset)
        ups = []
        for name, fields in f.items():
            ups += paged_updates(stage, f"/Lab/tissue/{name}", fields)
        _apply_local(stage, ups)
    _apply_local(stage, paged_updates(stage, "/Lab/rods/bone", rod_fields(frame.get("rods"))))
    _apply_local(stage, paged_updates(stage, "/Lab/tissue/fin", fin_fields(frame.get("fins"))))
    for name, c in {**CAMERAS, **(extra_cameras or {})}.items():
        _camera(stage, "/Cameras/" + name, c["eye"], c["target"], c["hfov"], c["role"], c.get("up", (0., 0., 1.)))
    stage.GetRootLayer().Save()
    lcd = info.pop("scale_lcd", None)
    meta = dict(preset=preset, bodies=names, n_links=len(links), wrist_cameras=wrist,
                cameras={k: {kk: vv for kk, vv in v.items()} for k, v in CAMERAS.items()},
                d405=D405, page_triangles=page_triangles,
                scale_display_attr=(str(lcd.GetPath()) + "/tex.inputs:file") if lcd is not None else None,
                skipped_dressing=sorted(skip), **{k: v for k, v in info.items() if k != "block_knife_poses"})
    (folder/"export.json").write_text(json.dumps(meta, indent=1, default=str))
    return folder/"scene.usdc"


def export_frame(frame, path, scene=None, preset="A"):
    """USD layer of ONE recorded frame's dynamic state. With `scene` (a scene.usdc written by export_scene), the
    layer sublayers it and overrides the body poses, so opening `path` shows the posed dressed room; the tissue
    and rods are authored as full meshes (/Lab/tissue_frame/<kind>, /Lab/rods_frame) and the paged copies are
    hidden. Without a scene the layer holds the tissue, rods and named body transforms only."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.GetRootLayer().customLayerData = dict(tick=int(frame.get("tick", 0)), t=float(frame.get("t", 0.)),
                                                badge=str(frame.get("badge", "")))
    bodies = frame.get("bodies", {}) or {}
    if scene is not None:
        scene = Path(scene)
        meta = json.loads((scene.parent/"export.json").read_text())
        stage.GetRootLayer().subLayerPaths.append(str(scene.resolve()))
        index = {n: i for i, n in enumerate(meta["bodies"])}
        poses = {l["name"]: l for l in frame.get("robot_links", [])}
        poses.update(bodies)
        for name, i in index.items():
            prim = stage.OverridePrim(f"/Lab/body_{i}")
            b = poses.get(name)
            pos, q = (b["pos"], b["quat_xyzw"]) if b else ((0., 0., -5.), (0., 0., 0., 1.))
            prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*map(float, pos)))
            prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf).Set(
                Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        for root in [f"/Lab/tissue/{k}" for k in KINDS.values()] + ["/Lab/tissue/fin", "/Lab/rods/bone"]:
            UsdGeom.Imageable(stage.OverridePrim(root)).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Lab"))
        preset = meta.get("preset", preset)
    else:
        for name, b in bodies.items():
            x = UsdGeom.Xform.Define(stage, "/Frame/bodies/" + _sanitize(name))
            _pose(x, b["pos"], b["quat_xyzw"])
            x.GetPrim().SetCustomDataByKey("mesh", str(b["mesh"]) if not isinstance(b["mesh"], dict) else "inline")
        for l in frame.get("robot_links", []):
            x = UsdGeom.Xform.Define(stage, "/Frame/robot/" + _sanitize(l["name"]))
            _pose(x, l["pos"], l["quat_xyzw"])
            x.GetPrim().SetCustomDataByKey("mesh", str(l["mesh"]))
    groups = {}
    if frame.get("tissue") is not None:
        groups.update({f"/Lab/tissue_frame/{k}": (v, f"/Materials/tissue_{k}") for k, v in tissue_fields(frame["tissue"], preset).items()})
    groups["/Lab/rods_frame/bone"] = (rod_fields(frame.get("rods")), "/DressMaterials/bone")
    if frame.get("fins"):
        groups["/Lab/tissue_frame/fin"] = (fin_fields(frame["fins"]), "/Materials/tissue_fin")
    from pxr import UsdShade
    for p, (f, mat) in groups.items():
        n = len(f["points"])
        g = UsdGeom.Mesh.Define(stage, p)
        g.CreateSubdivisionSchemeAttr("none")
        g.CreateDoubleSidedAttr(True)
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(f["points"]))
        g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(f["normals"]))
        g.SetNormalsInterpolation("vertex")
        g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n//3, 3, np.int32)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(n, dtype=np.int32)))
        pv = UsdGeom.PrimvarsAPI(g)
        if "st" in f:
            pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(Vt.Vec2fArray.FromNumpy(f["st"]))
        if "color" in f:
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex").Set(Vt.Vec3fArray.FromNumpy(f["color"]))
        if "myophase" in f:
            pv.CreatePrimvar("myophase", Sdf.ValueTypeNames.FloatArray, "vertex").Set(Vt.FloatArray.FromNumpy(f["myophase"]))
        for k in ("aux", "restpos"):
            if k in f:
                pv.CreatePrimvar(k, Sdf.ValueTypeNames.Float3Array, "vertex").Set(Vt.Vec3fArray.FromNumpy(f[k]))
        if scene is not None:
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(mat)))
    stage.GetRootLayer().Save()
    return path


# ------------------------------------------------------------------------------------------------ requests
def encode_fields(fields):
    from wiping.render_payload import encode
    width = {"points": 3, "normals": 3, "color": 3, "st": 2, "aux": 3, "restpos": 3}
    return {k: encode(np.asarray(v, np.float32).reshape(len(v), width[k]) if k in width else np.asarray(v, np.float32).ravel(),
                      "float32") for k, v in fields.items()}


def build_request(frame, meta, cameras, sequence, tissue_mode="pages", folder=None):
    """Request dict for the worker from a recorded frame (poses in the exported body order)."""
    index = {n: i for i, n in enumerate(meta["bodies"])}
    poses = [None]*len(index)
    for l in frame.get("robot_links", []):
        if l["name"] in index:
            poses[index[l["name"]]] = list(map(float, l["pos"])) + list(map(float, l["quat_xyzw"]))
    unknown = []
    for n, b in (frame.get("bodies") or {}).items():
        if n in index:
            poses[index[n]] = list(map(float, b["pos"])) + list(map(float, b["quat_xyzw"]))
        else:
            unknown.append(n)
    meta.setdefault("_unknown_bodies", set()).update(unknown)
    missing = [n for n, i in index.items() if poses[i] is None and i < meta["n_links"]]
    if missing:
        raise ValueError("Frame lacks robot links present in the scene: " + ", ".join(missing[:4]))
    poses = [p if p is not None else [0., 0., -5., 0., 0., 0., 1.] for p in poses]   # absent bodies park under the floor
    req = dict(op="observe", sequence=int(sequence), tick=int(frame.get("tick", 0)), body_poses=poses,
               cameras=list(cameras), meshes={}, material_fields={}, details={})
    preset = meta["preset"]
    rods = rod_fields(frame.get("rods"))
    if frame.get("tissue") is not None:
        tf = tissue_fields(frame["tissue"], preset)
        if tissue_mode == "pages":
            req["paged"] = {f"/Lab/tissue/{k}": encode_fields(v) for k, v in tf.items()}
            req["reauthor"] = {}
        else:
            req["paged"] = {f"/Lab/tissue/{k}": encode_fields({kk: vv[:0] for kk, vv in v.items()}) for k, v in tf.items()}
            req["reauthor"] = {f"/Lab/tissue_re/{k}": encode_fields(v) for k, v in tf.items()}
    fins = fin_fields(frame.get("fins"))
    if tissue_mode == "pages":
        req.setdefault("paged", {})["/Lab/tissue/fin"] = encode_fields(fins)
    else:
        req.setdefault("paged", {})["/Lab/tissue/fin"] = encode_fields({k: v[:0] for k, v in fins.items()})
        req.setdefault("reauthor", {})["/Lab/tissue_re/fin"] = encode_fields(fins)
    if tissue_mode == "pages":
        req.setdefault("paged", {})["/Lab/rods/bone"] = encode_fields(rods)
    else:
        req.setdefault("paged", {})["/Lab/rods/bone"] = encode_fields({k: v[:0] for k, v in rods.items()})
        req.setdefault("reauthor", {})["/Lab/rods_re/bone"] = encode_fields(rods)
    if frame.get("scale_g") is not None and meta.get("scale_display_attr") and folder is not None:
        from .dressing import scale_display_texture
        req["textures"] = {meta["scale_display_attr"]: str(scale_display_texture(Path(folder)/"textures", frame["scale_g"]))}
    return req


def badge_image(png, frame, camera):
    """Overlay the ASSISTED/UNASSISTED badge, sim time and tick (evidence films; never on policy observations)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(io.BytesIO(png)).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, im.width//64))
    except OSError:
        f = ImageFont.load_default()
    badge = str(frame.get("badge", ""))
    txt = f"{badge}  t = {float(frame.get('t', 0.)):.3f} s  tick {int(frame.get('tick', 0))}  [{camera}]"
    w = dr.textlength(txt, font=f)
    col = (190, 90, 10) if badge.upper().startswith("ASSIST") else (20, 110, 40) if badge else (40, 40, 40)
    dr.rectangle([8, 8, 24 + w, 16 + f.size + 8], fill=col)
    dr.text((16, 12), txt, font=f, fill=(255, 255, 255))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


class Renderer:
    """Owns exactly one render-only Isaac worker (wiping.render_worker protocol + filleting surfaces)."""

    def __init__(self, folder, first_frame, preset="A", *, width=1280, height=720, samples=32, scene=None,
                 tissue_mode="pages", extra_cameras=None):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.closed, self.lock, self.sequence, self.log_position = False, threading.Lock(), 0, 0
        self.tissue_mode = tissue_mode
        if scene is None or not Path(scene).is_file():
            scene = export_scene(self.folder, first_frame, preset, extra_cameras=extra_cameras)
        self.scene = Path(scene)
        self.meta = json.loads((self.scene.parent/"export.json").read_text())
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        env = dict(os.environ, WIPE_RENDER_PORT=str(listener.getsockname()[1]), WIPE_RENDER_SECRET=secret,
                   WIPE_MINIMAL_RENDER="1")
        self.log = (self.folder/"worker.log").open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "filleting.render_worker", "--scene", str(self.scene.resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=ROOT.parent, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
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
        self.timings = []

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def observe(self, frame, cameras=("overhead",), badge=False):
        """Render a recorded frame. Returns {camera: png bytes}; per-request timings are kept in self.timings."""
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            self.sequence += 1
            t0 = time.perf_counter()
            req = build_request(frame, self.meta, cameras, self.sequence, self.tissue_mode, self.folder)
            t1 = time.perf_counter()
            send(self.connection, req)
            deadline = time.monotonic() + 600
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
            t2 = time.perf_counter()
            if not result.get("ok") or result.get("physics_steps") != 0:
                raise RuntimeError("Invalid render response: " + str(result)[:600])
            self.unknown_bodies = sorted(self.meta.get("_unknown_bodies", ()))
            self.timings.append(dict(tick=int(frame.get("tick", 0)), encode_s=t1 - t0, round_trip_s=t2 - t1,
                                     worker=result.get("timings", {}), mode=self.tissue_mode))
            imgs = {k: base64.b64decode(v) for k, v in result["images"].items()}
            if badge:
                imgs = {k: badge_image(v, frame, k) for k, v in imgs.items()}
            return imgs

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True


def render_frames(frames, out_dir, folder, preset="A", cameras=("overhead", "supervisor"), *, width=1280, height=720,
                  samples=32, badge=True, tissue_mode="pages"):
    """Render a recorded sequence (list of frames) to out_dir/<camera>/<tick>.png through one worker.
    Returns (paths, timings)."""
    out_dir = Path(out_dir)
    r = Renderer(folder, frames[0], preset, width=width, height=height, samples=samples, tissue_mode=tissue_mode)
    paths = []
    try:
        for fr in frames:
            imgs = r.observe(fr, cameras, badge=badge)
            for cam, png in imgs.items():
                p = out_dir/cam/f"{int(fr.get('tick', 0)):08d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(png)
                paths.append(p)
    finally:
        r.close()
    return paths, r.timings

"""USD export of mechanical room B-012 and the client of the render-only Isaac worker (render_worker.py).

The stage holds static appearance; every request carries the recorded body poses and the paste surfaces of one
simulator state, so one exported stage serves a whole session or a whole film.
Appearance rules:
  * the installed room is drawn from the physics' static boxes (layout.statics, props, cart) with the materials of a
    basement pump room: painted CMU, a fresh grey CMU infill, a painted gypsum partition, the worn sealed slab, the
    2x4 forms in Douglas fir (drilling's generated texture), old pads, pumps and the crated new pump;
  * the tools are drawn on their collision parts as the products they are (magnesium, steel and blue-steel blades,
    soft-grip and wood handles; tools.py), in the same body frames;
  * the robot is drawn as in sweeping (same meshes, sweeping.render.robot_visual);
  * every wet region is one mesh of its height field (one vertex per cell, every `stride`-th cell for the big ones),
    shifted half a cell at its edges onto the rigid boundary, with per-vertex colour, gloss and tool texture from the
    cell's state (cell_look) and drawn by ConcreteFinishing.mdl: stone showing through a struck surface, bleed water as
    a mirror-like film, the closed dark face of a troweled slab, tooled mortar, gloss sealant, white compound;
  * visual-only dressing (no collider, nothing within the robot's working space): ducts and pipes along the walls,
    the pumps' piping, a disconnect switch, LED strip fixtures, a work light on a tripod raking across the pad, cement
    bags, a wet-floor sign, a broom, crates (drilling's pinned CC0 Poly Haven models and textures).
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

from classroom_arrangement import params as CP
from drilling.render import Stage as _Stage
from drilling import visual_assets as VA
from . import params as P
from . import layout as L

MDL_FILE = Path(__file__).with_name("ConcreteFinishing.mdl")
LIFT = 5.0e-5                  # paste meshes stand this far off their own surface (no shared faces with the rigid parts)
MAX_VERTICES = 140000          # per region: the stride is chosen to stay under this

COLORS = dict(
    magnesium=((.60, .61, .62), .38, .75), aluminum=((.74, .75, .77), .30, .90), trowel_steel=((.56, .58, .60), .22, 1.),
    edger_steel=((.52, .54, .56), .25, 1.), blue_steel=((.16, .20, .27), .30, .85), jointer_steel=((.30, .31, .32), .35, 1.),
    steel=((.60, .61, .62), .30, 1.), handle_soft=((.06, .12, .22), .65, 0.), handle_wood=((.55, .38, .22), .55, 0.),
    redwood=((.42, .17, .10), .75, 0.), nylon_white=((.88, .88, .86), .45, 0.), penetrometer_body=((.55, .06, .05), .40, .1),
    cart_grey=((.045, .045, .05), .55, 0.), rubber_black=((.03, .03, .035), .75, 0.), tub_black=((.03, .03, .032), .45, 0.),
    hawk_aluminum=((.72, .73, .75), .35, .9), pail_white=((.86, .86, .84), .45, 0.), rag_red=((.55, .09, .07), .95, 0.), board_plastic=((.78, .78, .76), .55, 0.),
    steel_black=((.04, .04, .045), .5, .3), pump_red=((.45, .05, .04), .38, .1), pump_motor=((.06, .14, .32), .42, .2),
    backer=((.10, .10, .11), .7, 0.), mortar_bed=((.30, .29, .27), .9, 0.), plate=((1., 0., 1.), .5, 0.),
    tape_blue=((.05, .22, .60), .6, 0.), safety_yellow=((.75, .55, .02), .6, 0.), light_housing=((.85, .85, .84), .4, .2),
    lamp=((.9, .9, .88), .3, 0.), tripod=((.05, .05, .055), .45, .4), robot_dark=((.07, .075, .08), .45, 0.),
    camera_dark=((.03, .03, .035), .45, 0.), tmr_black=((.05, .05, .055), .5, 0.), tmr_deck=((.80, .81, .80), .4, 0.),
    robot=((.86, .86, .86), .35, 0.), cured_concrete=((.36, .36, .35), .45, 0.),
)
TEXTURED = dict(
                gypsum=("painted_plaster_wall", 2.5, .85, (1.0, 0.99, 0.96)), deck=("concrete", 2.4, .9, (.85, .85, .85)),
                old_concrete=("brushed_concrete", 2.0, .85, (.80, .80, .78)), floor=("concrete_floor_worn_001", 3.0, .7, None),
                form_wood=("douglas_fir_flatsawn", 1.2, .7, None), pallet=("douglas_fir_flatsawn", 1.2, .8, (.85, .85, .85)),
                crate=("plywood", 1.2, .75, None))


class Stage(_Stage):
    """drilling's USD helper with placed meshes and this environment's own MDL module."""

    def mesh(self, path, v, f, uv=None, normals=None, mat=None, smooth=False, uv_interp="vertex", p=None, q=None):
        g = super().mesh(path, v, f, uv=uv, normals=normals, mat=mat, smooth=smooth, uv_interp=uv_interp)
        if p is not None:
            self.place(g, p, q)
        return g

    def box_world_uv(self, g, origin, per_m=1.0):
        """Box UVs in world metres (drilling's box_uv measures from the box's centre): textures with a real module, such
        as block coursing, then line up across the boxes of one wall."""
        Sdf, Vt, UsdGeom = self.Sdf, self.Vt, self.UsdGeom
        pts = np.asarray(g.GetPointsAttr().Get()) + np.asarray(origin, float)
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

    def own_mdl(self, name, module, inputs):
        dst = self.folder/MDL_FILE.name
        if not dst.exists() or dst.read_bytes() != MDL_FILE.read_bytes():
            shutil.copy2(MDL_FILE, dst)
        return self.mdl(name, module, inputs, path=str(dst.resolve()))


# ------------------------------------------------------------------------------ materials
def materials(S):
    mats = {name: S.color(name, *v) for name, v in COLORS.items()}
    for name, (tex, tile, rough, tint) in TEXTURED.items():
        mats[name] = S.textured(name, VA.texture(tex), tile, rough, tint=tint)
    mats["cmu"] = cmu_texture(S, "painted")
    mats["cmu_new"] = cmu_texture(S, "raw")
    mats["paste"] = S.own_mdl("paste", "ConcreteFinishingPaste", {})
    return mats


# ------------------------------------------------------------------------------ the wet surfaces
_STONE = {}


def _stones(reg):
    """A fixed stone pattern per region: (rank, tone) per cell. Coarse aggregate showing through a struck surface is
    drawn as individual stones 8 to 19 mm across (3/4 in aggregate), placed at random until they would cover 60 % of
    the surface; each stone has a rank in [0, 1) and a tone. A cell shows its stone when the stone's rank is below the
    cell's exposure E (E = 0.3: half of the stones show, about 30 % of the area), so working the surface buries
    stones one by one rather than fading a pattern."""
    key = (reg.name, reg.nu, reg.nv, reg.cell)
    if key not in _STONE:
        rng = np.random.default_rng(97)
        nu, nv, c = reg.nu, reg.nv, reg.cell
        rank = np.full((nu, nv), 2.0, np.float32)
        tone = np.ones((nu, nv), np.float32)
        area = nu*nv*c*c
        n = int(0.6*area/(math.pi*0.0065**2))
        cx, cy = rng.uniform(0, nu*c, n), rng.uniform(0, nv*c, n)
        rad = rng.uniform(0.004, 0.0095, n)
        rk, tn = rng.random(n), rng.uniform(0.75, 1.25, n)
        for x, y, r, k, t in zip(cx, cy, rad, rk, tn):
            i0, i1 = max(int((x - r)/c), 0), min(int((x + r)/c) + 1, nu)
            j0, j1 = max(int((y - r)/c), 0), min(int((y + r)/c) + 1, nv)
            if i0 >= i1 or j0 >= j1:
                continue
            ii = (np.arange(i0, i1) + 0.5)*c - x
            jj = (np.arange(j0, j1) + 0.5)*c - y
            el = rng.uniform(0.6, 1.0)
            inside = (ii[:, None]/r)**2 + (jj[None, :]/(r*el))**2 < 1.0
            sub = rank[i0:i1, j0:j1]
            put = inside & (sub > k)
            sub[put] = k
            tone[i0:i1, j0:j1][put] = t
        _STONE[key] = (rank, tone, rng.normal(0, 1, (nu, nv)).astype(np.float32))
    return _STONE[key]


def cell_look(reg, sl):
    """(rgb, gloss, tex) per cell of the slice, from the cell's state."""
    name = reg.material.name
    w, E, D, S = reg.w[sl], reg.E[sl], reg.D[sl], reg.S[sl]
    a = reg.a[sl]
    shape = w.shape
    if name in ("concrete", "repair_mortar"):
        # fresh concrete is a wet dark grey; a repair mortar is lighter; stone showing through is a little warmer
        base = np.array([.125, .125, .12]) if name == "concrete" else np.array([.24, .24, .235])
        stone = np.array([.150, .140, .126])
        rank, tone, grain = (x[sl] for x in _stones(reg))
        col = base*(1.0 + 0.05*grain)[..., None]                                       # sand mottling of the paste
        st = (rank < E)[..., None]
        col = np.where(st, stone*tone[..., None], col)
        dens = np.clip(D, 0, 1)[..., None]
        col = col*(1.0 - 0.32*dens) + np.array([.0, .004, .012])*dens                  # troweled: darker, cooler
        burn = (reg.late[sl] > 0)[..., None] & (D[..., None] > 0.95)
        col = np.where(burn, col*0.72, col)
        wet = np.clip(w/P.SHEEN_W, 0, 1)[..., None]
        col = col*(1.0 - 0.18*wet)
        moist = np.clip(1.0 - reg.age(sl)/max(reg.material.bleed_end, 1.0), 0, 1)*0.18     # a fresh face is moist
        gloss = np.where(w > P.SHEEN_W, np.clip(0.8 + 0.2*w/1e-4, 0, 1),
                         0.08 + moist + 0.45*np.clip(S, 0, 1) + 0.12*np.clip(D, 0, 1))
        tex = np.clip(a/1.0e-3, 0, 1)
    elif name == "mortar":
        col = np.broadcast_to(np.array([.26, .25, .235]), shape + (3,)).copy()
        tooled = (reg.compact[sl] > 0)[..., None]
        col = np.where(tooled, col*0.82, col)
        gloss = np.full(shape, 0.08)
        tex = np.clip(a/1.0e-3, 0, 1)
    elif name == "sealant":
        col = np.broadcast_to(np.array([.14, .145, .15]), shape + (3,)).copy()
        skinned = reg.age(sl) > reg.material.skin
        gloss = np.where(skinned, 0.25, 0.58)
        tex = np.clip(a/1.0e-3, 0, 1)*0.5
    else:                                                                              # compound
        col = np.broadcast_to(np.array([.80, .79, .76]), shape + (3,)).copy()
        gloss = np.full(shape, 0.06)
        tex = np.clip(a/1.0e-3, 0, 1)
    return col.astype(np.float32), gloss.astype(np.float32), tex.astype(np.float32)


def region_mesh(reg, stride=None):
    """Triangle mesh of the region's paste: vertices (world), triangles, vertex normals, attributes (rgb, gloss, tex)."""
    t = reg.t
    has = t > 2.0e-5
    if not has.any():
        return None
    ii, jj = np.nonzero(has)
    i0, i1, j0, j1 = max(int(ii.min()) - 1, 0), min(int(ii.max()) + 2, reg.nu), max(int(jj.min()) - 1, 0), min(int(jj.max()) + 2, reg.nv)
    if stride is None:
        stride = 1
        while ((i1 - i0)//stride)*((j1 - j0)//stride) > MAX_VERTICES:
            stride += 1
    sl = (slice(i0, i1, stride), slice(j0, j1, stride))
    m = has[sl]
    h = reg.h_at(sl).astype(float)
    nu, nv = m.shape
    step = reg.cell*stride
    u = reg.u[sl[0]][:, None] + 0*h
    v = reg.v[sl[1]][None, :] + 0*h
    # edge vertices move half a step toward the missing neighbour (the mesh ends on the rigid boundary)
    du = np.zeros_like(h); dv = np.zeros_like(h)
    du += np.where(m & ~np.roll(m, -1, 0), 0.5*step, 0.0); du -= np.where(m & ~np.roll(m, 1, 0), 0.5*step, 0.0)
    dv += np.where(m & ~np.roll(m, -1, 1), 0.5*step, 0.0); dv -= np.where(m & ~np.roll(m, 1, 1), 0.5*step, 0.0)
    pts = (reg.o[None, None, :] + (u + du)[..., None]*reg.U + (v + dv)[..., None]*reg.V + (h + LIFT)[..., None]*reg.N)
    # normals from the height gradient
    gu = np.gradient(h, step, axis=0) if nu > 1 else np.zeros_like(h)
    gv = np.gradient(h, step, axis=1) if nv > 1 else np.zeros_like(h)
    nrm = reg.N[None, None, :] - gu[..., None]*reg.U - gv[..., None]*reg.V
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    idx = -np.ones((nu, nv), np.int64)
    idx[m] = np.arange(int(m.sum()))
    q = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    a, b = np.nonzero(q)
    c00, c10, c01, c11 = idx[a, b], idx[a + 1, b], idx[a, b + 1], idx[a + 1, b + 1]
    flip = float(np.dot(np.cross(reg.U, reg.V), reg.N)) < 0
    tri = np.concatenate([np.stack([c00, c10, c11], 1), np.stack([c00, c11, c01], 1)]) if not flip else \
        np.concatenate([np.stack([c00, c11, c10], 1), np.stack([c00, c01, c11], 1)])
    if not len(tri):
        return None
    col, gloss, tex = cell_look(reg, sl)
    attrs = np.concatenate([col[m], gloss[m][:, None], tex[m][:, None]], 1)
    return dict(v=pts[m].astype(np.float32), t=tri.astype(np.int32), n=nrm[m].astype(np.float32), a=attrs.astype(np.float32))


def paste_meshes(world):
    from .transfer import all_regions
    out = {reg.name: region_mesh(reg) for reg in all_regions(world.regions)}
    q = world.state.body_q.numpy()
    for c in world.contactors:
        if c.roll > 1e-9:
            p, R = c.sole_pose(q[c.body])
            if c.tool is not None and c.tool.carry:
                centre = p+R[:, 2]*.001
            elif c.region is not None:
                lead = c.lead if c.lead is not None else np.array([0., 1.])
                centre = p+c.region.R[:, :2]@lead*(min(c.sole.hx, c.sole.hy)+.006)
            else:
                continue
            out['load_'+c.name.replace(':', '_')] = load_mesh(centre, R, c.roll, c.roll_mat,
                                                             min(c.sole.hx*.75, .06), min(c.sole.hy*.65, .03))
    for i, drop in enumerate(world.coupler.falling):
        r = (3*drop['packet'].volume/(4*np.pi))**(1/3)
        out[f'falling_{i}'] = load_mesh(drop['p'], np.eye(3), drop['packet'].volume, drop['packet'].material.name, r, r)
    packet = world.coupler.rag_packet
    if packet is not None and packet.volume > 1e-9:
        setup = L.TASK_SETUP[world.spec['task']]
        if 'rag' in setup:
            cx, cy, _ = setup['cart']; lx, ly = setup['rag']
            out['rag_deposit'] = load_mesh(np.array([cx+lx, cy+ly, P.CART[2]+P.RAG[2]]), np.eye(3),
                                            packet.volume, packet.material.name, .06, .025)
    return out


def load_mesh(centre, rotation, volume, material, a, b):
    """Volume-scaled closed ellipsoid, drawn from the carried/ballistic packet."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=2)
    height = volume/(4*np.pi*a*b/3)
    vertices = np.asarray(mesh.vertices)*[a, b, height]
    vertices[:, 2] += height
    normals = np.asarray(mesh.vertices)/[a, b, height]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colour = {'compound': (.80, .79, .76), 'sealant': (.14, .145, .15), 'mortar': (.26, .25, .235)}.get(material, (.18, .18, .17))
    attrs = np.tile([*colour, .15, .3], (len(vertices), 1)).astype(np.float32)
    return dict(v=(vertices@rotation.T+centre).astype(np.float32), t=np.asarray(mesh.faces, np.int32),
                n=(normals@rotation.T).astype(np.float32), a=attrs)


def encode_meshes(meshes, cache=None):
    """Base64 mesh arrays for the render worker. With `cache` (a dict kept by one film for one worker), a mesh equal to
    the one last sent goes as {"same": 1}: the worker leaves it as it is. Re-sending every paste mesh every frame (the
    pad alone about 280 000 vertices, some 25 MB) cost films one to three minutes a frame on a shared GPU."""
    b64 = lambda a, dt: base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()
    out = {}
    if cache is not None:
        for name in set(cache) - set(meshes):
            del cache[name]
    for name, m in meshes.items():
        if m is None or not len(m["t"]):
            if cache is not None:
                cache.pop(name, None)
            continue
        if cache is not None:
            import hashlib
            key = hashlib.blake2b(np.ascontiguousarray(m["v"], np.float32).tobytes() + np.ascontiguousarray(m["n"], np.float32).tobytes()
                                  + np.ascontiguousarray(m["a"], np.float32).tobytes()
                                  + np.ascontiguousarray(m["t"], np.int32).tobytes(), digest_size=16).hexdigest()
            if cache.get(name) == key:
                out[name] = dict(same=1)
                continue
            cache[name] = key
        out[name] = dict(v=b64(m["v"], np.float32), t=b64(m["t"], np.int32), n=b64(m["n"], np.float32),
                         c=b64(m["a"][:, :3], np.float32), g=b64(m["a"][:, 3], np.float32), x=b64(m["a"][:, 4], np.float32))
    size = sum(len(v) for e in out.values() for v in e.values() if isinstance(v, str))
    if size > 60*1024*1024:
        raise ValueError(f"paste meshes of {size >> 20} MiB exceed the renderer's 64 MiB message")
    return out


def frame(world, cache=None):
    """What one rendered frame needs from the world: body poses and the paste meshes (unchanged ones as markers when
    the caller keeps a `cache` for its worker)."""
    return dict(body_q=world.state.body_q.numpy().astype(np.float32), meshes=encode_meshes(paste_meshes(world), cache))


# ------------------------------------------------------------------------------ tools and statics
def _quat_xyzw(q):
    return tuple(float(x) for x in q)


def tool_visual(S, base, tool, mats):
    for p in tool.parts:
        mat = mats.get(p.material, mats["steel"])
        path = f"{base}/{p.name}"
        if p.kind == "box":
            S.box(path, 2*np.asarray(p.size), mat, p.pos, p.quat, bevel=min(0.0008, 0.3*min(p.size)))
        elif p.kind == "capsule":
            S.cylinder(path, p.size[0], 2*p.size[1], mat, p.pos, p.quat, capsule=True)
        elif p.kind == "cylinder":
            S.cylinder(path, p.size[0], 2*p.size[1], mat, p.pos, p.quat)


def statics_visual(S, spec, mats):
    for name, c, h, mat in spec["statics"]:
        if name == "door_leaf":
            continue                                       # drawn as the steel door (room_visual)
        m = mats.get(mat, mats["steel"])
        g = S.box(f"/Room/{name}", 2*np.asarray(h), m, c, (0, 0, 0, 1), bevel=0.0015)
        if mat in TEXTURED or mat in ("cmu", "cmu_new"):
            S.box_world_uv(g, c)
    for name, c, h, yaw, mat in spec["cart_boxes"]:
        q = (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))
        S.box(f"/Room/{name.replace('-', 'm')}", 2*np.asarray(h), mats.get(mat, mats["cart_grey"]), c, q, bevel=0.004)


TROFFERS = [(1.55, 1.45), (1.55, 3.45), (4.55, 1.45), (4.55, 3.45)]
TROFFER_SIZE = (1.2, 0.6)
TARGET_LUX = 350.            # at the pad's surface from the troffers (IES: 200-300 lx for mechanical rooms; a crew's own
                             # work light adds to it at the pad)


def troffer_luminance(reference, ceiling):
    """Rect-light luminance (nit) giving TARGET_LUX at `reference` from all troffers: E = sum L A cos_l cos_s / r^2
    (drilling.dressing.troffer_luminance with this room's fixtures)."""
    ref = np.asarray(reference, float)
    A = TROFFER_SIZE[0]*TROFFER_SIZE[1]
    g = 0.
    for x, y in TROFFERS:
        v = np.array([x, y, ceiling - 0.04]) - ref
        r = float(np.linalg.norm(v))
        c = max(0., v[2]/r)
        g += A*c*c/r**2
    return TARGET_LUX/g


def cmu_texture(S, kind="painted"):
    """CMU in running bond: 8 x 16 in nominal blocks (203 x 406 mm with their 3/8 in joints), coursing measured from
    the floor, the joints tooled concave (a 10 mm shadowed groove). The block faces carry the photographed block wall's
    grain in grey. `painted`: two coats of light grey latex (the grain at a third of its contrast, albedo 0.6);
    `raw`: new unpainted grey block (albedo 0.42, the grain at full contrast). One texture tile is 4 courses by 2
    blocks (0.8128 m square); the walls' UVs are world metres (Stage.box_world_uv), so the coursing starts at the floor
    and runs on across the boxes of one wall."""
    from PIL import Image
    maps = VA.texture("concrete_block_wall")
    out = S.folder/"textures"/f"cmu_{kind}.png"
    tile = 4*0.2032
    if not out.exists():
        N = 2048
        px = tile/N
        grain = np.asarray(Image.open(maps["diff"]).convert("L").resize((N, N)), np.float32)/255.
        grain = grain - grain.mean()
        base, contrast = (0.60, 0.30) if kind == "painted" else (0.42, 0.9)
        img = base + contrast*grain
        v = (np.arange(N) + 0.5)*px                              # image rows = texture v (bottom-up after the flip)
        u = (np.arange(N) + 0.5)*px
        off_v = 0.0                                              # world UVs (Stage.box_world_uv): courses from the floor
        course = np.floor((v - off_v)/0.2032).astype(int)
        dv = np.abs(((v - off_v) % 0.2032) - 0.2032) .clip(0, None)
        dv = np.minimum((v - off_v) % 0.2032, 0.2032 - (v - off_v) % 0.2032)
        joint_v = np.clip(1.0 - dv/0.0060, 0.0, 1.0)            # within 6 mm of a bed joint's centre line
        shade = np.zeros((N, N), np.float32)
        for r in range(N):
            uu = u + (0.2032 if course[r] % 2 else 0.0)
            du = np.minimum(uu % 0.4064, 0.4064 - uu % 0.4064)
            joint_u = np.clip(1.0 - du/0.0060, 0.0, 1.0)
            shade[r] = np.maximum(joint_v[r], joint_u)
        img = img*(1.0 - 0.35*shade**1.5)
        img = img[::-1]                                          # row 0 is the texture's top (v = 1)
        rgb = np.stack([img, img*0.995, img*0.975], -1)
        Image.fromarray((np.clip(rgb, 0, 1)**(1/2.2)*255).astype(np.uint8)).save(out)
    return S.textured(f"cmu_{kind}", dict(diff=out, rough=maps["rough"], nor_gl=maps["nor_gl"]), tile, .8 if kind == "painted" else .9)


def room_visual(S, D, spec, mats):
    """Floor, cove base, the door, the room's services under the deck, the old pumps' piping and motors, the
    expansion tank, the floor drain, the infill's old joints where no joint task is laid out, tape and lights."""
    from drilling import visual_models as VM
    rx, ry, rz = P.ROOM
    pad = 0.4
    pts = [(-pad, -pad, 0), (rx + pad, -pad, 0), (rx + pad, ry + pad, 0), (-pad, ry + pad, 0)]
    S.mesh("/Room/floor", pts, [0, 1, 2, 0, 2, 3], uv=[(p[0], p[1]) for p in pts], normals=[(0, 0, 1)]*4, mat=mats["floor"])
    segs = [((0., 0., 0.), (P.DOOR[0], 0., 0.), (0., 1., 0.)), ((P.DOOR[1], 0., 0.), (rx, 0., 0.), (0., 1., 0.)),
            ((0., 0., 0.), (0., ry, 0.), (1., 0., 0.)), ((rx, 0., 0.), (rx, P.CJ_Y - P.CJ_W/2 - 0.01, 0.), (-1., 0., 0.)),
            ((rx, P.CJ_Y + P.CJ_W/2 + 0.01, 0.), (rx, ry, 0.), (-1., 0., 0.)), ((0., ry, 0.), (rx, ry, 0.), (0., -1., 0.))]
    D.draw("/Dress/cove_base", VM.wall_base(segs))
    if spec.get("variant") != "door":
        D.draw("/Dress/door", VM.steel_door(P.DOOR[0], P.DOOR[1], 0.0, h=P.DOOR[2], normal=1.))
    else:                                    # open: the leaf swung out into the corridor, against its wall
        w = P.DOOR[1] - P.DOOR[0]
        S.box("/Dress/door_leaf", (0.044, w - 0.01, P.DOOR[2] - 0.02), D.S.color("door_grey", (.42, .45, .47), .45, .2),
              (P.DOOR[0] + 0.03, -P.PARTITION_T - w/2, P.DOOR[2]/2), (0, 0, 0, 1))
    D.draw("/Dress/switch", VM.junction_box((P.DOOR[0] - 0.25, 0.0, 1.20), (0., 1., 0.), size=0.075, depth=0.04, mat="white_paint"))
    # sprinkler main along the north wall and two branch lines across the room with pendent heads
    from drilling.dressing import _sprinkler_head
    zm = 2.78
    D.draw("/Dress/sprinkler_main", VM.pipe_run([(0.0, ry - 0.14, zm), (rx, ry - 0.14, zm)], 0.057, "pipe_red",
                                                 flanges=[(x, ry - 0.14, zm) for x in (1.2, 3.9)]))
    for k, x in enumerate((0.6, 2.4, 5.6)):
        D.draw(f"/Dress/main_bracket_{k}", VM.strut((x, ry - 0.0001, zm - 0.0775), (x, ry - 0.24, zm - 0.0775))
               + VM.pipe_clamp((x, ry - 0.14, zm), 0.057, (1., 0., 0.), (0., 0., -1.)))
    for k, x in enumerate((1.8, 4.3)):
        zb = 2.92
        D.draw(f"/Dress/branch_{k}", VM.pipe_run([(x, ry - 0.14, zm), (x, ry - 0.14, zb), (x, 0.45, zb)], 0.021, "pipe_red", bend=0.05))
        for j, y in enumerate((3.6, 2.2, 0.8)):
            D.draw(f"/Dress/head_{k}_{j}", _sprinkler_head((x, y, zb)))
    # a copper line and a conduit run along the north wall, a supply duct along the south side with two registers
    D.draw("/Dress/copper", VM.pipe_run([(0.0, ry - 0.06, 2.52), (rx, ry - 0.06, 2.52)], 0.0143, "copper"))
    D.draw("/Dress/emt_north", VM.pipe_run([(0.3, ry - 0.022, 2.96), (rx - 0.25, ry - 0.022, 2.96)], 0.0105, "emt_galv"))
    D.draw("/Dress/duct", VM.duct(0.3, rx - 0.3, 0.75, rz - 0.02, registers=[(1.6, 1), (4.2, 1)]))
    # the old pumps: motor, coupling guard, volute, suction and discharge rising to the deck with a valve on each
    for k, y in enumerate((1.30, 3.30)):
        S.cylinder(f"/Dress/pump{k}_motor", 0.125, 0.34, mats["pump_motor"], (0.86, y, 0.27), _rot_quat((1, 0, 0)))
        S.cylinder(f"/Dress/pump{k}_fan_cover", 0.118, 0.06, mats["pump_motor"], (1.05, y, 0.27), _rot_quat((1, 0, 0)))
        S.box(f"/Dress/pump{k}_guard", (0.12, 0.13, 0.13), D.S.color("guard_yellow", (.78, .58, .04), .5, .1), (0.62, y, 0.255), (0, 0, 0, 1))
        S.cylinder(f"/Dress/pump{k}_volute", 0.125, 0.20, mats["pump_red"], (0.40, y, 0.26), _rot_quat((1, 0, 0)))
        D.draw(f"/Dress/pump{k}_suction", VM.pipe_run([(0.30, y, 0.26), (0.12, y, 0.26), (0.12, y, rz)], 0.0445, "white_paint", bend=0.08))
        D.draw(f"/Dress/pump{k}_discharge", VM.pipe_run([(0.40, y, 0.385), (0.40, y, 0.70), (0.30, y, 0.70), (0.30, y, rz)], 0.0365,
                                                         "white_paint", bend=0.06))
        D.draw(f"/Dress/pump{k}_valve", VM.gate_valve((0.30, y, 1.30), (0., 0., 1.), 0.0365, (1., 0., 0.)))
    # expansion tank on the west wall between the pumps, and its line
    S.cylinder("/Dress/exp_tank", 0.17, 0.62, D.S.color("tank_blue", (.10, .22, .45), .4, .3), (0.22, 2.30, 1.55), (0, 0, 0, 1))
    D.draw("/Dress/exp_line", VM.pipe_run([(0.22, 2.30, 1.24), (0.22, 2.30, 0.95), (0.12, 2.30, 0.95), (0.12, 2.30, rz)], 0.0105, "copper"))
    # floor drain
    S.cylinder("/Dress/drain", 0.075, 0.002, D.S.color("drain_iron", (.08, .08, .085), .6, .6), (1.20, 1.00, 0.0006), (0, 0, 0, 1))
    # the infill's joints where no joint task is laid out: old mortar, tooled concave (2 mm behind the block faces)
    if "infill" not in spec["regions"]:
        x0, x1 = P.INFILL_X
        z0 = P.INFILL_Z0
        z1 = z0 + P.INFILL_COURSES*(P.BLOCK[2] + P.JOINT_W) + P.JOINT_W
        S.box("/Dress/infill_mortar", (x1 - x0, 0.010, z1 - z0), D.S.color("old_mortar", (.46, .45, .42), .9, 0.),
              ((x0 + x1)/2, ry + 0.0021 + 0.005, (z0 + z1)/2), (0, 0, 0, 1), bevel=0.0005)
    if "joint" not in spec["regions"]:
        S.box("/Dress/joint_sealant", (0.0040, P.CJ_W, rz - 0.10), D.S.color("cured_sealant", (.20, .205, .21), .55, 0.),
              (rx + 0.0021 + 0.0020, P.CJ_Y, rz/2), (0, 0, 0, 1), bevel=0.0005)
    if "wallboard" in spec["regions"]:
        # Visible open weave: contact uses its 0.5 mm envelope; the field's
        # base stores only fibre volume, leaving the openings for compound.
        cx, cz, _ = P.DW_PATCH
        fibre = D.S.color("fibreglass_mesh", (.62, .59, .48), .9, 0.)
        for k, offset in enumerate(np.arange(-.048, .05, .004)):
            S.box(f"/Room/mesh_x_{k}", (.10, .00025, .0012), fibre,
                  (cx, .000125, cz+offset), (0, 0, 0, 1))
            S.box(f"/Room/mesh_z_{k}", (.0012, .00025, .10), fibre,
                  (cx+offset, .000375, cz), (0, 0, 0, 1))
    # painter's tape marks the sealant segment's ends on the east wall
    for k, z in enumerate(P.CJ_SEGMENT):
        S.box(f"/Room/tape_{k}", (0.0004, 0.10, 0.024), mats["tape_blue"], (rx - 0.0002, P.CJ_Y, z + (0.012 if k == 0 else -0.012)), (0, 0, 0, 1))
    # safety line around the existing pumps
    for k, (a, b) in enumerate((((0.05, 0.75), (1.35, 0.75)), ((1.35, 0.75), (1.35, 3.85)), ((0.05, 3.85), (1.35, 3.85)))):
        c = ((a[0] + b[0])/2, (a[1] + b[1])/2, 0.0003)
        size = (abs(b[0] - a[0]) + 0.05, 0.05, 0.0006) if abs(b[1] - a[1]) < 1e-6 else (0.05, abs(b[1] - a[1]) + 0.05, 0.0006)
        S.box(f"/Room/safety_{k}", size, mats["safety_yellow"], c, (0, 0, 0, 1), bevel=0.0002)
    lights(S, D, mats)


def lights(S, D, mats):
    """Four 2 x 4 ft LED troffers (4000 K) under the deck, sized for TARGET_LUX at the pad, a work light on a tripod
    raking low across the pad from its north-west corner (what finishers use to see the sheen), and a weak dome for
    the bounce of a closed room."""
    UsdLux, Gf = S.UsdLux, S.Gf
    rz = P.ROOM[2]
    lum = troffer_luminance((P.PAD_CENTRE[0], P.PAD_CENTRE[1], 0.09), rz)
    for k, (x, y) in enumerate(TROFFERS):
        try:
            D.ref(f"/Dress/troffer_{k}", "mounted_fluorescent_lights", (x, y, rz), 0.)
        except Exception:
            S.box(f"/Room/troffer_{k}", (1.22, 0.62, 0.08), mats["light_housing"], (x, y, rz - 0.04), (0, 0, 0, 1))
        lt = UsdLux.RectLight.Define(S.stage, f"/Lights/troffer_{k}")
        lt.CreateWidthAttr(TROFFER_SIZE[0]); lt.CreateHeightAttr(TROFFER_SIZE[1])
        lt.CreateIntensityAttr(float(lum))
        lt.CreateEnableColorTemperatureAttr(True); lt.CreateColorTemperatureAttr(4000.)
        lt.GetPrim().CreateAttribute("visibleInPrimaryRay", S.Sdf.ValueTypeNames.Bool).Set(True)
        S.place(lt, (x, y, rz - 0.045), (0., 0., 0., 1.))
    # work light: a 50 W LED flood on a tripod 1.6 m up at the pad's north-west, aimed down across the pad
    wx, wy = L.WORKLIGHT
    for k in range(3):
        a = 2*math.pi*k/3 + 0.3
        foot = np.array([wx + 0.45*math.cos(a), wy + 0.45*math.sin(a), 0.0])
        top = np.array([wx, wy, 1.0])
        d = top - foot
        Lg = float(np.linalg.norm(d))
        S.cylinder(f"/Room/tripod_leg{k}", 0.011, Lg, mats["tripod"], tuple((foot + top)/2), _rot_quat(d/Lg))
    S.cylinder("/Room/tripod_mast", 0.013, 0.62, mats["tripod"], (wx, wy, 1.30), (0, 0, 0, 1))
    aim = np.array([P.PAD_CENTRE[0], P.PAD_CENTRE[1], 0.09]) - np.array([wx, wy, 1.62])
    aim /= np.linalg.norm(aim)
    R = _frame_from_z(-aim)
    S.box("/Room/worklight_head", (0.26, 0.20, 0.07), mats["tripod"], (wx, wy, 1.64), Rotation.from_matrix(R).as_quat())
    lt = UsdLux.RectLight.Define(S.stage, "/Lights/worklight")
    lt.CreateWidthAttr(0.22); lt.CreateHeightAttr(0.16)
    lt.CreateIntensityAttr(6000.); lt.CreateColorTemperatureAttr(4000.); lt.CreateEnableColorTemperatureAttr(True)
    S.place(lt, tuple(np.array([wx, wy, 1.64]) + 0.036*aim), Rotation.from_matrix(R).as_quat())
    dome = UsdLux.DomeLight.Define(S.stage, "/Lights/ambient")
    dome.CreateIntensityAttr(25.); dome.CreateColorTemperatureAttr(4000.); dome.CreateEnableColorTemperatureAttr(True)


def _rot_quat(z):
    """xyzw quaternion turning +z onto unit vector z."""
    z = np.asarray(z, float)/np.linalg.norm(z)
    c = float(z[2])
    if c > 1 - 1e-12:
        return (0., 0., 0., 1.)
    if c < -1 + 1e-12:
        return (1., 0., 0., 0.)
    axis = np.cross((0, 0, 1), z)
    axis /= np.linalg.norm(axis)
    return tuple(Rotation.from_rotvec(axis*math.acos(max(-1., min(1., c)))).as_quat())


def _frame_from_z(z):
    """A rotation with z (unit) as its third column (a USD light emits along its -z)."""
    z = np.asarray(z, float)/np.linalg.norm(z)
    x = np.cross((0, 0, 1), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1., 0., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def dressing(S, D, mats):
    """Visual-only dressing from drilling's pinned CC0 models; nothing within a station's working space."""
    placed = []

    def ref(path, asset, pos, yaw=0.0, scale=None):
        try:
            D.ref(path, asset, pos, yaw, scale)
            placed.append(asset)
        except Exception as exc:                            # a missing pin draws nothing rather than failing the export
            placed.append(f"{asset}: not drawn ({type(exc).__name__})")

    ref("/Dress/power_box", "power_box_01", (0.02, 0.45, 1.45), math.pi/2)
    ref("/Dress/cement_a", "cement_bag", (0.32, 2.12, 0.0), 1.75)
    ref("/Dress/cement_b", "cement_bag", (0.34, 2.52, 0.0), 1.45)
    ref("/Dress/crate", "plastic_crate_02", (0.40, 0.25, 0.0), 0.1)
    ref("/Dress/sign", "WetFloorSign_01", (5.25, 0.25, 0.0), 0.6)
    ref("/Dress/broom", "plastic_broom", (0.10, 4.10, 0.0), 0.0)
    ref("/Dress/extinguisher", "korean_fire_extinguisher_01", (4.40, 0.06, 0.0), math.pi)
    return D.credits, placed


# ------------------------------------------------------------------------------ cameras
VIEWS = dict(
    room=((5.70, 4.45, 2.75), (2.60, 1.70, 0.20), 15.),
    pad=((3.20, 4.45, 1.45), (3.20, 2.85, 0.08), 22.),
    pad_close=((2.75, 2.25, 0.55), (3.05, 2.80, 0.08), 26.),
    patch=((4.05, 0.40, 1.15), (4.62, 1.28, 0.02), 26.),      # from the south-west (from the east the crate hid it)
    infill=((3.70, 4.15, 1.45), (4.55, 4.88, 1.20), 20.),
    joint=((5.50, 2.32, 1.32), (6.10, 2.00, 1.22), 24.),       # close enough that the 13 mm bead reads
    wallboard=((1.20, 0.95, 1.45), (1.82, 0.0, 1.25), 22.),
    plan=((3.05, 2.44, 9.0), (3.05, 2.441, 0.0), 14.),
)


def cameras(S, world):
    """Policy cameras (head, wrists, base) and diagnostic views; returns {name: role}."""
    sc = world.scene
    out = {}
    inst = sc.robot_instance
    fr = world.rig.frame_pose(f"{inst}/head_camera", world.state)
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
        g = S.camera(f"/Lab/body_{m['body']}/{name}", m["eye"], m["target"], 18/math.tan(math.radians(m["fov"]/2)),
                     up=m["up"], role="policy")
        S.alias(name, g.GetPath()); out[name] = "policy"
    base = sc.robot_body["base_link"]
    g = S.camera(f"/Lab/body_{base}/chase", (-1.9, 0.9, 1.9), (0.9, 0., 0.35), 16.)
    S.alias("chase", g.GetPath()); out["chase"] = "diagnostic"
    for name, (eye, target, f) in VIEWS.items():
        up = (0, 1, 0) if name == "plan" else (0, 0, 1)
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f, up=up)
        S.alias(name, c.GetPath()); out[name] = "diagnostic"
    return out


# ------------------------------------------------------------------------------ export
def export(world, folder, extra_cameras=None):
    """Export the world's static appearance and posable things to folder/scene.usdc; returns (path, meta)."""
    from sweeping.render import robot_visual
    sc = world.scene
    S = Stage(folder)
    mats = materials(S)
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    for name, t in sc.tools.items():
        tool_visual(S, f"/Lab/body_{t['body']}/{name}", t["tool"], mats)
    robot_visual(S, world, mats)
    statics_visual(S, world.spec, mats)
    if "wallboard" in world.regions:
        from .wallboard_visual import replace
        replace(S.stage)
    from drilling.dressing import Dresser
    D = Dresser(S)
    room_visual(S, D, world.spec, mats)
    credits, placed = dressing(S, D, mats)
    # the paste: one mesh prim per region (the worker replaces its geometry and primvars every request)
    T = S.Sdf.ValueTypeNames
    S.UsdGeom.Xform.Define(S.stage, "/Paste")
    meshes = paste_meshes(world)
    for name in world.regions:
        m = meshes.get(name)
        if m is None:
            v = np.zeros((3, 3), np.float32); f = np.array([[0, 1, 2]]); a = np.zeros((3, 5), np.float32)
        else:
            v, f, a = m["v"], m["t"], m["a"]
        g = S.mesh(f"/Paste/{name}", v, f, normals=None if m is None else m["n"], mat=mats["paste"])
        pv = S.UsdGeom.PrimvarsAPI(g)
        pv.CreatePrimvar("displayColor", T.Color3fArray, "vertex").Set(S.Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a[:, :3])))
        pv.CreatePrimvar("gloss", T.FloatArray, "vertex").Set(S.Vt.FloatArray.FromNumpy(np.ascontiguousarray(a[:, 3])))
        pv.CreatePrimvar("tex", T.FloatArray, "vertex").Set(S.Vt.FloatArray.FromNumpy(np.ascontiguousarray(a[:, 4])))
    meta = dict(cameras=cameras(S, world), credits=sorted(credits), dressing=placed, regions=list(world.regions))
    for name, (eye, target, f) in (extra_cameras or {}).items():
        if name in meta["cameras"]:
            raise ValueError(f"extra camera {name!r} would replace a built-in camera")
        c = S.camera(f"/Cameras/{name}_cam", eye, target, f)
        S.alias(name, c.GetPath()); meta["cameras"][name] = "diagnostic"
    S.save()
    (S.folder/"export.json").write_text(json.dumps(meta))
    return S.folder/"scene.usdc", meta


# ------------------------------------------------------------------------------ worker client
# a render worker that met these while loading or rendering may draw a partial scene: assets left unloaded (the cart, the
# pump station gone from a film's room view) and fallback cyan where textures failed (a film on a GPU shared with
# four other render workers); such a worker is replaced, never filmed with
GPU_MEMORY_ERRORS = ("OUT_OF_DEVICE_MEMORY", "Out of GPU memory", "vkCreateDevice failed", "Unable to allocate buffer",
                     "cudaErrorMemoryAllocation", "subAllocate() failed")


def worker_log_size(folder):
    p = Path(folder)/"worker.log"
    return p.stat().st_size if p.is_file() else 0


def worker_memory_errors(folder, start=0):
    """Lines of the worker's log past byte `start` that report a GPU memory failure."""
    p = Path(folder)/"worker.log"
    if not p.is_file():
        return 0
    with p.open("rb") as f:
        f.seek(start)
        text = f.read().decode("utf-8", "replace")
    return sum(1 for line in text.splitlines() if any(k in line for k in GPU_MEMORY_ERRORS))


class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, folder, scene, *, width=640, height=360, samples=16, engine=None):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.folder = Path(folder)
        self.closed, self.lock = False, threading.RLock()
        self._last_meshes = {}
        self.width, self.height = width, height
        self.last_timings = {}
        self.engine = engine or os.environ.get("CF_RENDER_ENGINE", "isaac")
        if self.engine not in ("isaac", "software"):
            raise ValueError("render engine must be isaac or software")
        import sys
        executable = sys.executable if self.engine == "software" else os.environ.get("ISAAC_RENDER_PYTHON")
        module = "concrete_finishing.raster_worker" if self.engine == "software" else "concrete_finishing.render_worker"
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
        listener.settimeout(float(os.environ.get("CF_RENDER_START_S", "420")))
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, CF_RENDER_PORT=str(listener.getsockname()[1]), CF_RENDER_SECRET=secret)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder/"worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", module, "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept(); self.connection.settimeout(900)
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

    def observe(self, cameras, fr, tick, resolution=None, samples=None, resolutions=None):
        """fr: render.frame(world) or a recorded frame of the same shape. Returns {camera: PNG bytes}."""
        # Keep the cache with the actual worker, including HTTP callers that do
        # not supply frame(..., cache). Identical arrays must not dirty USD mesh
        # geometry on every robot-only action. Commit only after a valid response.
        # Serialize preparation, response and cache commit as one transaction.
        with self.lock:
            meshes, pending = {}, {}
            for name, m in fr.get("meshes", {}).items():
                if m.get("same"):
                    if name not in self._last_meshes:
                        raise ValueError(f"Mesh {name!r}: unchanged marker without confirmed worker geometry")
                    meshes[name] = m
                    pending[name] = self._last_meshes[name]
                else:
                    meshes[name] = {"same": 1} if self._last_meshes.get(name) == m else m
                    pending[name] = dict(m)
            msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(fr["body_q"], float).tolist(), cameras=list(cameras),
                       meshes=meshes)
            if resolution:
                msg["resolution"] = list(resolution)
            if resolutions:
                msg["resolutions"] = {k: list(v) for k, v in resolutions.items()}
            if samples:
                msg["spp"] = int(samples)
            try:
                r = self._call(msg)
            except BaseException:
                # The worker may have applied geometry before failing to render
                # or reply. Its state is unknown, including older cached meshes.
                self._last_meshes.clear()
                raise
            self._last_meshes = pending
            self.last_timings = r.get("timings", {})
            return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True

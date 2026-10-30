"""The dock wall with its doors, seals, bumpers and levelers, the trailers (docked, spotted at the yard ramp and parked),
and the whole yard preset: asphalt, apron, building wall, the mobile ramp, markings and surroundings. Draws every layout
part that common.owner(...) gives to 'dockyard', and the trailer and leveler bodies.

Rules kept here (visuals.common): nothing changes the simulation. What can be touched is drawn on its collider: the
trailer's floor, walls, nose, roof and rear frame, the leveler deck and lip, the ramp deck, lip and curbs, the wall and
bumper boxes are read from the shapes or the layout and drawn inside their faces (relief of a few millimetres; ribs,
rails, roof bows, doors and windows are sunk into the box, never proud of it). Everything else is dressing without a
collider: it comes from dressing(layout, preset, seed), which depends on the layout alone and lists each object with
its bounding box, and draw() draws exactly that list. Drawn without a collider and not in that list, because they
belong to a simulated body or to the task's state and not to the layout, or lie where there is no ground to drive on:
the running gear, landing gear and folded-back doors of the simulated trailers (they move with the trailer's body),
the stiffeners under a leveler deck (with the deck), the wheel chocks of trailers at dock doors (on the apron
outside), and the surroundings beyond the yard's and the test pad's fence (terrain, plants, road, neighbours).

Appearance comes from the layout and the built world only. Maps: Poly Haven and ambientCG textures from
assets/extra_sources_dockyard.json (a flat colour stands in for any that the cache does not hold); every other
graphic (liveries, signs, tape, wear, cracks) is generated into the stage's texture folder at export, the same every
time. Lenses, door lites and (if ROOF_DAYLIGHT_NITS is set) trailer roofs are emissive surfaces; no light source is
created here, and dock_light_source() follows render's dock lights rather than placing them.

draw_lab(S, world) draws the lab preset (no layout): the experiments' outdoor test pad from the world's tyre surfaces.
colliders(layout, preset, seed, doors_with_trailer) lists the boxes of the dressing items that should be given static
colliders (collide=True; a raised door leaf is placed above its opening).

Sections: geometry builder | materials and generated graphics | trailer | dock wall, doors, leveler | yard |
surroundings (terrain, plants, neighbours) | lab test pad | dressing, colliders and draw.
"""

import math
import zipfile
from pathlib import Path

import numpy as np

from .. import params as P
from .common import box_half, kit, owner

IN, FT = 0.0254, 0.3048


# =====================================================================================================================
# geometry builder: triangle soups per material in a local frame, instanced small parts
# =====================================================================================================================
def _rot(yaw=0.0, pitch=0.0, roll=0.0):
    """Rotation matrix Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_quat()


def _frame(axis):
    """Two unit vectors perpendicular to `axis` (u, w with u x w = axis)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, a)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


_UV_AXES = ((1, 2), (0, 2), (0, 1))


class _BoxTemplate:
    """Vertex signs, bevel masks, normals and triangles of a box with flat faces and (optionally) chamfered edges."""

    def __init__(self, bevel):
        faces = []
        for a in range(3):
            b, c = [k for k in range(3) if k != a]
            for sg in (-1, 1):
                ring = []
                for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    s = [0, 0, 0]
                    s[a], s[b], s[c] = sg, sb, sc
                    ring.append((tuple(s), a))
                n = [0.0, 0.0, 0.0]
                n[a] = sg
                faces.append((ring, n))
        if bevel:
            for a in range(3):
                for b in range(a + 1, 3):
                    c = 3 - a - b
                    for sa in (-1, 1):
                        for sb in (-1, 1):
                            s0, s1 = [0, 0, 0], [0, 0, 0]
                            s0[a], s0[b], s0[c] = sa, sb, -1
                            s1[a], s1[b], s1[c] = sa, sb, 1
                            ring = [(tuple(s0), a), (tuple(s0), b), (tuple(s1), b), (tuple(s1), a)]
                            n = [0.0, 0.0, 0.0]
                            n[a], n[b] = sa, sb
                            faces.append((ring, n))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        s = (sx, sy, sz)
                        faces.append(([(s, 0), (s, 1), (s, 2)], [sx, sy, sz]))
        S, M, N, F, UA, VA = [], [], [], [], [], []
        h, bb = np.array([1.0, 2.0, 3.0]), 0.1
        for ring, n in faces:
            n = np.asarray(n, float)
            n /= np.linalg.norm(n)
            pts = []
            for s, a in ring:
                m = np.ones(3)
                m[a] = 0.0
                pts.append(np.asarray(s, float) * (h - bb * m))
            if np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), n) < 0:
                ring = ring[::-1]
            k = len(S)
            d = int(np.argmax(np.abs(n)))
            for s, a in ring:
                m = np.ones(3)
                m[a] = 0.0
                S.append(s); M.append(m); N.append(n); UA.append(_UV_AXES[d][0]); VA.append(_UV_AXES[d][1])
            for j in range(1, len(ring) - 1):
                F.append((k, k + j, k + j + 1))
        self.S, self.M, self.N = np.asarray(S, float), np.asarray(M, float), np.asarray(N, float)
        self.F = np.asarray(F, np.int32)
        self.ua, self.va = np.asarray(UA), np.asarray(VA)
        self.idx = np.arange(len(S))


_BOX_PLAIN, _BOX_BEVEL = _BoxTemplate(False), _BoxTemplate(True)


class _Geo:
    __slots__ = ("v", "n", "uv", "f", "k")

    def __init__(self):
        self.v, self.n, self.uv, self.f, self.k = [], [], [], [], 0

    def add(self, V, N, UV, F):
        self.v.append(V); self.n.append(N); self.uv.append(UV); self.f.append(np.asarray(F, np.int32) + self.k)
        self.k += len(V)


class _Build:
    """Accumulates geometry per material name in one parent frame; at(...) gives a builder in a child frame that
    writes into the same meshes. flush() writes one mesh per material and one point instancer for the small repeated
    parts (rivets, bolts, wheels, fence posts)."""

    def __init__(self, geos=None, inst=None, M=None):
        self.g = {} if geos is None else geos
        self.i = {} if inst is None else inst
        self.M = np.eye(4) if M is None else M

    def at(self, p=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, roll=0.0, R=None):
        T = np.eye(4)
        T[:3, :3] = _rot(yaw, pitch, roll) if R is None else np.asarray(R, float)
        T[:3, 3] = p
        return _Build(self.g, self.i, self.M @ T)

    def _emit(self, mat, V, N, UV, F):
        R, t = self.M[:3, :3], self.M[:3, 3]
        g = self.g.get(mat)
        if g is None:
            g = self.g[mat] = _Geo()
        g.add(np.asarray(V, float) @ R.T + t, np.asarray(N, float) @ R.T, np.asarray(UV, float), F)

    # ---- primitives
    def box(self, mat, c, size, bevel=0.0, R=None, uvs=1.0, uvo=(0.0, 0.0), swap=False, only=None):
        """Box of full `size` centred at c (optionally rotated by R about c), edges chamfered by `bevel`. UVs are the
        face-plane coordinates in metres (x uvs); swap turns the texture a quarter turn. only: the faces to draw of an
        unbevelled box, a string of -x +x -y +y -z +z."""
        h = np.asarray(size, float) / 2.0
        c = np.asarray(c, float)
        T = _BOX_BEVEL if bevel > 0 else _BOX_PLAIN
        b = min(float(bevel), 0.49 * float(h.min())) if bevel > 0 else 0.0
        V = T.S * (h - b * T.M)
        P_ = V + c
        U = np.stack([P_[T.idx, T.ua], P_[T.idx, T.va]], 1)
        if swap:
            U = U[:, ::-1]
        U = U * uvs + np.asarray(uvo, float)
        N = T.N
        if R is not None:
            R = np.asarray(R, float)
            V, N = V @ R.T, N @ R.T
        if only is not None and bevel <= 0:
            ks = [("-x", "+x", "-y", "+y", "-z", "+z").index(only[i:i + 2]) for i in range(0, len(only), 2)]
            vi = np.concatenate([np.arange(4 * k, 4 * k + 4) for k in ks])
            F = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + 4 * j for j in range(len(ks))])
            self._emit(mat, V[vi] + c, N[vi], U[vi], F)
            return
        self._emit(mat, V + c, N, U, T.F)

    def poly(self, mat, pts, uv=None, uvs=1.0, swap=False):
        """Flat convex polygon (counter-clockwise seen from its front)."""
        Pn = np.asarray(pts, float)
        n = np.cross(Pn[1] - Pn[0], Pn[2] - Pn[0])
        ln = np.linalg.norm(n)
        if ln < 1e-14:
            return
        n /= ln
        if uv is None:
            d = int(np.argmax(np.abs(n)))
            uv = Pn[:, _UV_AXES[d]] * uvs
            if swap:
                uv = uv[:, ::-1]
        F = [(0, j, j + 1) for j in range(1, len(Pn) - 1)]
        self._emit(mat, Pn, np.tile(n, (len(Pn), 1)), uv, F)

    def rect(self, mat, c, u, v, size, uv01=False, uvs=1.0, tile=None, flip_u=False, swap=False, uvo=(0.0, 0.0)):
        """Rectangle centred at c spanning size[0] along u and size[1] along v (front = u x v). uv01: the texture once
        over the rectangle (signs, decals); tile=(nu, nv): repeated that many times; else UV in metres x uvs."""
        c, u, v = (np.asarray(x, float) for x in (c, u, v))
        u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
        a, b = size[0] / 2.0, size[1] / 2.0
        pts = [c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v]
        if uv01 or tile is not None:
            nu, nv = tile if tile is not None else (1.0, 1.0)
            uv = np.array([(0.0, 0.0), (nu, 0.0), (nu, nv), (0.0, nv)])
            if flip_u:
                uv[:, 0] = nu - uv[:, 0]
        else:
            uv = np.array([(-a, -b), (a, -b), (a, b), (-a, b)]) * uvs + np.array([c @ u, c @ v]) * uvs
            if swap:
                uv = uv[:, ::-1]
            uv = uv + np.asarray(uvo, float)
        self.poly(mat, pts, uv=uv)

    def cyl(self, mat, a, b, r, n=16, caps=True, r2=None, uvs=1.0):
        """Cylinder (or cone frustum r -> r2) from a to b, smooth sides."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        Lg = float(np.linalg.norm(b - a))
        if Lg < 1e-9:
            return
        ax, u, w = _frame(b - a)
        r2 = r if r2 is None else r2
        ang = np.linspace(0.0, 2 * math.pi, n + 1)
        D = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
        V = np.concatenate([a + r * D, b + r2 * D])
        Nn = D * Lg + ax * (r - r2)
        Nn /= np.linalg.norm(Nn, axis=1)[:, None]
        N = np.concatenate([Nn, Nn])
        UV = np.concatenate([np.stack([ang * r, np.zeros(n + 1)], 1), np.stack([ang * r, np.full(n + 1, Lg)], 1)]) * uvs
        i = np.arange(n)
        F = np.concatenate([np.stack([i, i + 1, n + 2 + i], 1), np.stack([i, n + 2 + i, n + 1 + i], 1)])
        self._emit(mat, V, N, UV, F)
        if caps:
            for cpt, rr, nn, sgn in ((a, r, -ax, -1), (b, r2, ax, 1)):
                if rr < 1e-9:
                    continue
                Vc = np.concatenate([[cpt], cpt + rr * D[:-1]])
                Fc = [(0, 1 + k, 1 + (k + 1) % n) if sgn > 0 else (0, 1 + (k + 1) % n, 1 + k) for k in range(n)]
                UVc = np.concatenate([[(0.0, 0.0)], np.stack([np.cos(ang[:-1]) * rr, np.sin(ang[:-1]) * rr], 1)]) * uvs
                self._emit(mat, Vc, np.tile(nn, (n + 1, 1)), UVc, Fc)

    def revolve(self, mat, profile, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), n=32, sharp=False, uvs=1.0):
        """Surface of revolution about `axis` through `origin`: profile = [(radius, height along the axis), ...],
        traversed so that the outside is on the left of the travel direction seen with the axis pointing up (a
        cylinder's wall bottom to top). sharp: flat per segment (creased); else smooth."""
        prof = np.asarray(profile, float)
        o = np.asarray(origin, float)
        ax, u, w = _frame(axis)
        ang = np.linspace(0.0, 2 * math.pi, n + 1)
        D = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
        segs = [(k, k + 1) for k in range(len(prof) - 1)] if sharp else [None]
        Ls = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
        for seg in segs:
            pr = prof if seg is None else prof[list(seg)]
            ls = Ls if seg is None else Ls[list(seg)]
            m = len(pr)
            tang = np.empty_like(pr)
            tang[1:-1] = pr[2:] - pr[:-2]
            tang[0], tang[-1] = pr[1] - pr[0], pr[-1] - pr[-2]
            if seg is not None:
                tang[:] = pr[1] - pr[0]
            tl = np.linalg.norm(tang, axis=1)
            tl[tl < 1e-12] = 1.0
            tang = tang / tl[:, None]
            V = o + pr[None, :, 1, None] * ax + pr[None, :, 0, None] * D[:, None, :]
            N = tang[None, :, 1, None] * D[:, None, :] - tang[None, :, 0, None] * ax
            UV = np.stack(np.broadcast_arrays(ang[:, None] * max(float(pr[:, 0].max()), 1e-6), ls[None, :]), -1) * uvs
            j, i = np.meshgrid(np.arange(n), np.arange(m - 1), indexing="ij")
            a_, b_, c_, d_ = j * m + i, (j + 1) * m + i, (j + 1) * m + i + 1, j * m + i + 1
            F = np.concatenate([np.stack([a_, b_, c_], -1).reshape(-1, 3), np.stack([a_, c_, d_], -1).reshape(-1, 3)])
            self._emit(mat, V.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), F)

    def tube(self, mat, pts, r, n=8, caps=True):
        """Round tube along a polyline (parallel-transported frames, smooth)."""
        Pn = np.asarray(pts, float)
        if len(Pn) < 2:
            return
        tang = np.empty_like(Pn)
        tang[1:-1] = Pn[2:] - Pn[:-2]
        tang[0], tang[-1] = Pn[1] - Pn[0], Pn[-1] - Pn[-2]
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        _, u, _ = _frame(tang[0])
        ang = np.linspace(0.0, 2 * math.pi, n + 1)
        V, N, UV = [], [], []
        s = 0.0
        for k in range(len(Pn)):
            t = tang[k]
            u = u - (u @ t) * t
            u /= np.linalg.norm(u)
            w = np.cross(t, u)
            D = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
            if k:
                s += float(np.linalg.norm(Pn[k] - Pn[k - 1]))
            V.append(Pn[k] + r * D); N.append(D); UV.append(np.stack([ang * r, np.full(n + 1, s)], 1))
        m = n + 1
        j, i = np.meshgrid(np.arange(len(Pn) - 1), np.arange(n), indexing="ij")
        a_, b_, c_, d_ = j * m + i, j * m + i + 1, (j + 1) * m + i + 1, (j + 1) * m + i
        F = np.concatenate([np.stack([a_, b_, c_], -1).reshape(-1, 3), np.stack([a_, c_, d_], -1).reshape(-1, 3)])
        self._emit(mat, np.concatenate(V), np.concatenate(N), np.concatenate(UV), F)
        if caps:
            for cpt, nn, ring, sgn in ((Pn[0], -tang[0], V[0], -1), (Pn[-1], tang[-1], V[-1], 1)):
                Vc = np.concatenate([[cpt], ring[:-1]])
                Fc = [(0, 1 + k, 1 + (k + 1) % n) if sgn > 0 else (0, 1 + (k + 1) % n, 1 + k) for k in range(n)]
                self._emit(mat, Vc, np.tile(nn, (n + 1, 1)), np.zeros((n + 1, 2)), Fc)

    def sheet(self, mat, A, B, uvs=1.0, flip=False):
        """Ruled surface between two polylines of equal length (flat quads; front = (A[i+1]-A[i]) x (B[i]-A[i]))."""
        A, B = np.asarray(A, float), np.asarray(B, float)
        if flip:
            A, B = A[::-1], B[::-1]
        m = len(A) - 1
        e1, e2 = A[1:] - A[:-1], B[:-1] - A[:-1]
        N = np.cross(e1, e2)
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(e1, axis=1))]
        hA = np.linalg.norm(B - A, axis=1)
        V = np.stack([A[:-1], A[1:], B[1:], B[:-1]], 1).reshape(-1, 3)
        UV = np.stack([np.stack([s[:-1], np.zeros(m)], 1), np.stack([s[1:], np.zeros(m)], 1),
                       np.stack([s[1:], hA[1:]], 1), np.stack([s[:-1], hA[:-1]], 1)], 1).reshape(-1, 2) * uvs
        k = 4 * np.arange(m)
        F = np.concatenate([np.stack([k, k + 1, k + 2], 1), np.stack([k, k + 2, k + 3], 1)])
        self._emit(mat, V, np.repeat(N, 4, axis=0), UV, F)

    def prism(self, mat, profile, a, b, up=(0.0, 0.0, 1.0), caps=True, uvs=1.0):
        """A closed 2-D profile [(p, q), ...] (counter-clockwise) extruded from a to b; the profile's q axis is `up`
        made perpendicular to the extrusion, its p axis = up x direction (so p, q, direction are right-handed)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        Lg = float(np.linalg.norm(d))
        d = d / Lg
        q = np.asarray(up, float) - (np.asarray(up, float) @ d) * d
        q /= np.linalg.norm(q)
        p = np.cross(q, d)
        pr = np.asarray(profile, float)
        ring = pr[:, 0, None] * p + pr[:, 1, None] * q
        ringc = np.concatenate([ring, ring[:1]])
        self.sheet(mat, a + ringc, b + ringc, uvs=uvs)
        if caps:
            import mapbox_earcut as earcut
            tri = np.asarray(earcut.triangulate_float64(pr, np.array([len(pr)], np.uint32)), np.int32).reshape(-1, 3)
            e1, e2 = pr[tri[:, 1]] - pr[tri[:, 0]], pr[tri[:, 2]] - pr[tri[:, 0]]
            cw = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 0
            tri[cw] = tri[cw][:, ::-1]                       # counter-clockwise in (p, q): front = +direction
            self._emit(mat, a + ring, np.tile(-d, (len(pr), 1)), pr * uvs, tri[:, ::-1])
            self._emit(mat, b + ring, np.tile(d, (len(pr), 1)), pr * uvs, tri)

    def beam(self, mat, a, b, size, up=(0.0, 0.0, 1.0), bevel=0.0, ext=0.0):
        """A bar of cross-section size = (width, depth along `up`) from a to b (extended by ext at both ends)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        Lg = float(np.linalg.norm(d))
        if Lg < 1e-9:
            return
        d = d / Lg
        upv = np.asarray(up, float)
        if abs(upv @ d) > 0.98:
            upv = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        side = np.cross(upv, d)
        side /= np.linalg.norm(side)
        R = np.stack([d, side, np.cross(d, side)], 1)
        self.box(mat, (a + b) / 2.0, (Lg + 2 * ext, size[0], size[1]), bevel=bevel, R=R)

    def inst(self, proto, p, R=None, s=1.0):
        """One instance of a registered prototype (rivet, bolt, wheel, post) at p in this frame."""
        T = np.eye(4)
        if R is not None:
            T[:3, :3] = R
        T[:3, 3] = p
        Mw = self.M @ T
        sc = np.full(3, float(s)) if np.isscalar(s) else np.asarray(s, float)
        self.i.setdefault(proto, []).append((Mw[:3, 3].copy(), Mw[:3, :3].copy(), sc))

    def flush(self, S, path):
        from ..usd import clean
        path = clean(path)
        S.UsdGeom.Xform.Define(S.stage, path)
        for mat, g in self.g.items():
            if g.k:
                S.mesh(f"{path}/{mat}", np.concatenate(g.v), np.concatenate(g.f), uv=np.concatenate(g.uv),
                       normals=np.concatenate(g.n), mat=_mat(S, mat))
        if self.i:
            Gf, Vt, UsdGeom = S.Gf, S.Vt, S.UsdGeom
            pi = UsdGeom.PointInstancer.Define(S.stage, f"{path}/parts")
            names = sorted(self.i)
            targets, idx, pos, quat, scl = [], [], [], [], []
            for k, name in enumerate(names):
                pp = f"{path}/parts/{name}"
                if name.startswith("plant_"):
                    _plant_proto(S, pp, name)
                else:
                    Bp = _Build()
                    _PROTOS[name](Bp)
                    Bp.flush(S, pp)
                targets.append(pp)
                Rs = np.array([it[1] for it in self.i[name]])
                qs = _quat(Rs)
                for (p_, _, s_), q_ in zip(self.i[name], qs):
                    idx.append(k); pos.append(p_); scl.append(s_)
                    quat.append(Gf.Quath(float(q_[3]), float(q_[0]), float(q_[1]), float(q_[2])))
            pi.CreatePrototypesRel().SetTargets(targets)
            pi.CreateProtoIndicesAttr(Vt.IntArray(idx))
            pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pos, np.float32)))
            pi.CreateOrientationsAttr(Vt.QuathArray(quat))
            pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.asarray(scl, np.float32)))
        self.g.clear(); self.i.clear()


# ---- prototypes of the small repeated parts (local +z out of the surface they sit on unless noted)
def _proto_rivet(B):
    """Round-head rivet, 9 mm across, 3 mm proud."""
    B.revolve("dy_alu", [(0.0045, 0.0), (0.004, 0.0018), (0.0022, 0.003), (0.0, 0.0031)], n=8)


def _proto_bolt(B):
    """Hex bolt head 24 mm across flats on a washer."""
    B.cyl("dy_steel_galv", (0, 0, 0), (0, 0, 0.003), 0.019, n=12)
    B.cyl("dy_steel_galv", (0, 0, 0.003), (0, 0, 0.015), 0.0135, n=6)


def _proto_screw(B):
    """Floor screw head flush in the boards (a dark countersunk disc)."""
    B.cyl("dy_steel_dark", (0, 0, 0), (0, 0, 0.0006), 0.006, n=8)


_PROTOS = {"rivet": _proto_rivet, "bolt": _proto_bolt, "screw": _proto_screw}

_RZ = {1: _rot(roll=-math.pi / 2), -1: _rot(roll=math.pi / 2)}        # local +z onto +y / -y
_RX = {1: _rot(pitch=math.pi / 2), -1: _rot(pitch=-math.pi / 2)}      # local +z onto +x / -x


# =====================================================================================================================
# materials (created on first use) and generated graphics
# =====================================================================================================================
def _asset(path):
    from ..assets.install import resolve
    try:
        return resolve(path)
    except FileNotFoundError:
        return None


def _ph(name):
    """Poly Haven maps of a texture (extra_sources_dockyard.json), or None when the cache does not hold them."""
    d = {k: _asset(f"polyhaven/{name}/{name}_{k}_2k.jpg") for k in ("diff", "rough", "nor_gl")}
    return dict(diff=d["diff"], rough=d["rough"], nor=d["nor_gl"]) if all(d.values()) else None


def _acg(S, name):
    """ambientCG maps of a material, extracted from its cached zip into the stage's texture folder."""
    z = _asset(f"ambientcg/{name}_2K-JPG.zip")
    if z is None:
        return None
    out = {}
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        for key, tag in (("diff", "Color"), ("rough", "Roughness"), ("nor", "NormalGL"), ("metal", "Metalness")):
            member = f"{name}_2K-JPG_{tag}.jpg"
            if member not in names:
                continue
            dst = S.folder / "textures" / f"acg_{name}_{tag}.jpg"
            if not dst.is_file() or dst.stat().st_size != zf.getinfo(member).file_size:
                dst.write_bytes(zf.read(member))
            out[key] = dst
    return out if "diff" in out else None


def _pbr(S, name, maps=None, tile=1.0, colour=(0.5, 0.5, 0.5), tint=(1.0, 1.0, 1.0), bright=1.0, rough=0.6,
         rough_infl=1.0, metal=0.0, bump=1.0, opacity_tex=None, opacity=None, glow=None, desat=0.0, diff=None,
         detail=None, cutout=None, alpha_channel=False):
    """OmniPBR. maps: dict(diff, rough, nor[, metal]) of image paths with mesh UVs in metres and `tile` metres per
    repeat (a pair for a non-square tile); diff alone: a generated image; glow = (rgb, intensity); detail = (normal
    map, its tile in metres, strength): fine relief on top of a coarse base. cutout: opacity threshold (leaf cards)
    instead of fractional opacity; alpha_channel: the opacity is the image's alpha rather than its grey value."""
    T, Gf, Sdf = S.Sdf.ValueTypeNames, S.Gf, S.Sdf

    def A(p):
        return (T.Asset, Sdf.AssetPath(str(Path(p).resolve())))
    tu, tv = (tile, tile) if np.isscalar(tile) else tile
    inp = {"diffuse_color_constant": (T.Color3f, Gf.Vec3f(*map(float, colour))),
           "diffuse_tint": (T.Color3f, Gf.Vec3f(*map(float, tint))),
           "albedo_brightness": (T.Float, float(bright)), "albedo_desaturation": (T.Float, float(desat)),
           "reflection_roughness_constant": (T.Float, float(rough)), "metallic_constant": (T.Float, float(metal)),
           "texture_scale": (T.Float2, Gf.Vec2f(1.0 / tu, 1.0 / tv))}
    maps = dict(maps or {})
    if diff is not None:
        maps["diff"] = diff
    if maps.get("diff"):
        inp["diffuse_texture"] = A(maps["diff"])
    if maps.get("rough"):
        inp["reflectionroughness_texture"] = A(maps["rough"])
        inp["reflection_roughness_texture_influence"] = (T.Float, float(rough_infl))
    if maps.get("nor"):
        inp["normalmap_texture"] = A(maps["nor"])
        inp["bump_factor"] = (T.Float, float(bump))
    if maps.get("metal"):
        inp["metallic_texture"] = A(maps["metal"])
        inp["metallic_texture_influence"] = (T.Float, 1.0)
    if opacity_tex is not None or opacity is not None:
        inp["enable_opacity"] = (T.Bool, True)
        inp["opacity_constant"] = (T.Float, float(1.0 if opacity is None else opacity))
        if opacity_tex is not None:
            inp["enable_opacity_texture"] = (T.Bool, True)
            inp["opacity_texture"] = A(opacity_tex)
            inp["opacity_mode"] = (T.Int, 0 if alpha_channel else 1)
            inp["opacity_threshold"] = (T.Float, float(cutout or 0.0))
    if detail is not None:
        inp["detail_normalmap_texture"] = A(detail[0])
        inp["detail_texture_scale"] = (T.Float2, Gf.Vec2f(1.0 / detail[1], 1.0 / detail[1]))
        inp["detail_bump_factor"] = (T.Float, float(detail[2]))
    if glow is not None:
        inp["enable_emission"] = (T.Bool, True)
        inp["emissive_color"] = (T.Color3f, Gf.Vec3f(*map(float, glow[0])))
        inp["emissive_intensity"] = (T.Float, float(glow[1]))
    return S.mdl(name, "OmniPBR", inp)


def _font(size, oblique=False):
    import matplotlib
    from PIL import ImageFont
    name = "DejaVuSans-BoldOblique.ttf" if oblique else "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), int(size))


def _noise(rng, shape, cells, octaves=3, persistence=0.5, tileable=False):
    """Value noise in 0..1, shape (h, w): random lattices of `cells` (rows, columns) doubled per octave, smoothed."""
    from PIL import Image
    h, w = shape
    cy, cx = (cells, cells) if np.isscalar(cells) else cells
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        ny, nx = max(2, int(round(cy * 2 ** o))), max(2, int(round(cx * 2 ** o)))
        g = rng.random((ny, nx)).astype(np.float32)
        if tileable:
            g = np.pad(g, 2, mode="wrap")
            big = np.asarray(Image.fromarray(g).resize((w * (nx + 4) // nx, h * (ny + 4) // ny), Image.BICUBIC))
            oy, ox = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
            layer = big[oy:oy + h, ox:ox + w]
        else:
            layer = np.asarray(Image.fromarray(g).resize((w, h), Image.BICUBIC))
        out += amp * layer
        tot += amp
        amp *= persistence
    return np.clip(out / tot, 0.0, 1.0)


def _save(S, name, arr):
    """Write an image array (uint8, or float 0..1) into the stage's texture folder."""
    from PIL import Image
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    p = S.folder / "textures" / name
    Image.fromarray(a).save(p, compress_level=1)
    return p


def _text_image(lines, size, fg, bg, border=None, pad=0.08, oblique=False):
    """A sign face: centred lines of text [(text, relative height), ...] on a plain ground. Returns a PIL image."""
    from PIL import Image, ImageDraw
    w, h = size
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    if border:
        bw = max(2, int(0.03 * min(w, h)))
        d.rectangle([bw, bw, w - 1 - bw, h - 1 - bw], outline=border, width=bw)
    tot = sum(r for _, r in lines)
    y = h * pad + (h * (1 - 2 * pad)) * 0.0
    for text, rel in lines:
        hh = h * (1 - 2 * pad) * rel / tot
        fs = int(hh * 0.9)
        f = _font(fs, oblique)
        while fs > 6 and d.textlength(text, font=f) > w * (1 - 2 * pad):
            fs = int(fs * 0.93)
            f = _font(fs, oblique)
        bb = d.textbbox((0, 0), text, font=f)
        d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], y + (hh - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=fg)
        y += hh
    return im


LIVERIES = (dict(name="ORBEK", sub="FREIGHT LINES", colour=(22, 58, 122), accent=(196, 40, 30), code="OBKU"),
            dict(name="TAMARO", sub="CARTAGE CO.", colour=(150, 30, 26), accent=(40, 42, 46), code="TMRZ"),
            dict(name="HALVEN", sub="LOGISTICS", colour=(18, 104, 70), accent=(226, 168, 22), code="HVNL"),
            dict(name="KESTRALWAY", sub="TRANSPORT", colour=(34, 34, 38), accent=(222, 120, 18), code="KSWT"))
BUILDING = dict(name="ORBEK", line="DISTRIBUTION CENTER 4", colour=(22, 58, 122))     # all fictional


def _skin_texture(S, k, unit, length, height):
    """A trailer's side sheet as one image (left to right as seen from outside): painted aluminium sheets with their
    seams, the carrier's name and unit number, road film rising from the bottom rail and streaks under the top rail."""
    from PIL import Image, ImageDraw
    lv = LIVERIES[k % len(LIVERIES)]
    name = f"dy_skin_{k}_{unit}.png"
    p = S.folder / "textures" / name
    if not _fresh(S, name):
        return p
    rng = np.random.default_rng(4100 + 17 * k + unit)
    W_, H_ = 4096, int(round(4096 * height / length / 8)) * 8
    ppm = W_ / length
    base = np.full((H_, W_, 3), 0.90, np.float32) * np.array([1.0, 1.0, 1.01], np.float32)
    # sheets 4 ft wide, each a shade of its own; a faint lighter band over every post (the sheet is drawn to it)
    xs = np.arange(W_) / ppm
    sheet = np.floor(xs / 1.2192).astype(int)
    shade = 1.0 + 0.018 * (np.random.default_rng(77 + k).random(sheet.max() + 2)[sheet] - 0.5)
    base *= shade[None, :, None]
    seam = (np.abs((xs / 1.2192) - np.round(xs / 1.2192)) * 1.2192 * ppm) < 0.8
    base[:, seam] *= 0.80
    post = np.exp(-(((xs / 0.6096) - np.round(xs / 0.6096)) * 0.6096 / 0.010) ** 2)
    base *= (1.0 - 0.05 * post)[None, :, None]
    im = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # carrier name: letters 0.95 m tall, a rule and the second line under it; unit numbers front and rear
    fs = int(0.95 * ppm / 0.73)
    f = _font(fs, oblique=True)
    tw = d.textlength(lv["name"], font=f)
    x0 = (W_ - tw) / 2
    y0 = H_ * 0.34
    bb = d.textbbox((x0, y0), lv["name"], font=f)
    d.text((x0, y0 - (bb[1] - y0) - 0.48 * ppm), lv["name"], font=f, fill=lv["colour"])
    yb = y0 + 0.62 * ppm
    d.rectangle([x0 - 0.5 * ppm, yb, x0 + tw + 0.5 * ppm, yb + 0.07 * ppm], fill=lv["accent"])
    f2 = _font(int(0.30 * ppm / 0.73))
    tw2 = d.textlength(lv["sub"], font=f2)
    d.text((x0 + tw - tw2, yb + 0.14 * ppm), lv["sub"], font=f2, fill=lv["colour"])
    f3 = _font(int(0.11 * ppm / 0.73))
    num = f"{lv['code']} {530000 + unit}"
    for xx in (0.35 * ppm, W_ - 0.35 * ppm - d.textlength(num, font=f3)):
        d.text((xx, 0.22 * ppm), num, font=f3, fill=(30, 30, 34))
    a = np.asarray(im, np.float32) / 255.0
    # road film: densest at the bottom rail, in drifts; rain streaks hanging from the top rail
    zz = (1.0 - (np.arange(H_) + 0.5) / H_)[:, None] * height
    drift = _noise(rng, (H_, W_), (2, 60), 4, 0.6)
    film = np.clip(np.exp(-zz / (0.16 + 0.22 * drift)) * (0.45 + 0.4 * drift), 0.0, 0.7)
    streak = _noise(rng, (H_, W_), (2, 500), 2) ** 2.5 * np.clip(1.0 - (height - zz) / 1.4, 0.0, 1.0)
    runs = _noise(rng, (H_, W_), (1, 700), 1) ** 3 * np.clip(zz / 0.9, 0.0, 1.0) * np.clip(1.2 - zz / 1.2, 0.0, 1.0)
    grime = _noise(rng, (H_, W_), (9, 50), 4)
    k_f = np.clip(film + 0.16 * streak + 0.10 * runs + 0.05 * (grime - 0.5), 0.0, 1.0)[..., None]
    dirt = np.array([0.40, 0.37, 0.33], np.float32)
    a = a * (1.0 - 0.6 * k_f) + dirt * 0.6 * k_f
    return _save(S, name, a)


def _closed_door_texture(S, k, unit, width, height):
    """The rear doors' outer sheets of a parked trailer (both leaves, as seen from behind): unit number and the
    carrier's mark at the top, film on the lower half."""
    from PIL import Image, ImageDraw
    lv = LIVERIES[k % len(LIVERIES)]
    name = f"dy_reardoor_{k}_{unit}.png"
    p = S.folder / "textures" / name
    if not _fresh(S, name):
        return p
    rng = np.random.default_rng(5200 + 13 * k + unit)
    W_, H_ = 1024, int(round(1024 * height / width / 8)) * 8
    ppm = W_ / width
    im = Image.new("RGB", (W_, H_), (228, 229, 231))
    d = ImageDraw.Draw(im)
    fs = int(0.20 * ppm / 0.73)
    f = _font(fs, oblique=True)
    while fs > 8 and d.textlength(lv["name"], font=f) > W_ * 0.36:
        fs = int(fs * 0.93)
        f = _font(fs, oblique=True)
    d.text((W_ * 0.56, 0.20 * ppm), lv["name"], font=f, fill=lv["colour"])
    f3 = _font(int(0.10 * ppm / 0.73))
    d.text((W_ * 0.56, 0.50 * ppm), f"{lv['code']} {530000 + unit}", font=f3, fill=(30, 30, 34))
    d.text((W_ * 0.08, 0.22 * ppm), "53", font=_font(int(0.16 * ppm / 0.73)), fill=(30, 30, 34))
    a = np.asarray(im, np.float32) / 255.0
    zz = (1.0 - (np.arange(H_) + 0.5) / H_)[:, None] * height
    drift = _noise(rng, (H_, W_), (6, 5), 3)
    film = np.clip(np.exp(-zz / (0.5 + 0.9 * drift)) * (0.5 + 0.6 * drift), 0.0, 0.8)[..., None]
    a = a * (1.0 - 0.7 * film) + np.array([0.33, 0.31, 0.28], np.float32) * 0.7 * film
    return _save(S, name, a)


def _fresh(S, name):
    """True the first time a generated image is asked for during this export (then it is made and reused)."""
    done = S.__dict__.setdefault("_dockyard_images", set())
    if name in done:
        return False
    done.add(name)
    return True


def _tex_once(S, name, make):
    p = S.folder / "textures" / name
    if _fresh(S, name):
        make(p)
    return p


def _tape_texture(S):
    """Conspicuity tape: 11 in red, 7 in white (49 CFR 571.108), one 18 in period."""
    a = np.zeros((16, 288, 3), np.uint8)
    a[:, :176] = (168, 16, 22)
    a[:, 176:] = (236, 236, 232)
    return _save(S, "dy_tape.png", a)


def _hazard_texture(S):
    """Yellow and black diagonal stripes, one seamless tile."""
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    k = ((xx + yy) % (n // 2)) < (n // 4)
    a = np.where(k[..., None], np.array([226, 172, 12], np.uint8), np.array([16, 16, 18], np.uint8)).astype(np.uint8)
    return _save(S, "dy_hazard.png", a)


def _etrack_texture(S):
    """Logistic track: galvanised strip with its slots (one 50 mm pitch)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (64, 160), (150, 154, 158))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 63, 14], fill=(128, 132, 136)); d.rectangle([0, 145, 63, 159], fill=(128, 132, 136))
    d.rounded_rectangle([22, 42, 42, 118], radius=5, fill=(18, 18, 20))
    return _save(S, "dy_etrack.png", np.asarray(im))


def _wear_alpha(S):
    """Opacity of worn paint: mostly there, scuffed away in patches (seamless tile)."""
    rng = np.random.default_rng(31)
    n = 512
    a = _noise(rng, (n, n), 6, 5, 0.6, tileable=True)
    b = _noise(rng, (n, n), 48, 2, 0.5, tileable=True)
    al = np.clip((a - 0.36) * 5.0, 0.0, 1.0) * np.clip(0.75 + 0.5 * (b - 0.3), 0.0, 1.0)
    return _save(S, "dy_wear_alpha.png", np.clip(0.12 + 0.88 * al, 0, 1))


def _stain_alpha(S, k):
    """Opacity of a ground stain (an irregular soft blob fading to nothing at the edge of the image)."""
    rng = np.random.default_rng(200 + k)
    n = 256
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0) * 2 - 1
    r = np.sqrt(xx ** 2 + yy ** 2)
    nz = _noise(rng, (n, n), 5, 4, 0.55)
    al = np.clip((1.0 - r - 0.55 * (nz - 0.35)) * 2.2, 0.0, 1.0) ** 1.3
    return _save(S, f"dy_stain{k}.png", 0.42 * al * (0.55 + 0.45 * _noise(rng, (n, n), 30, 2)))


def _tyremark_alpha(S):
    """Opacity of rubber laid down by braking and turning tyres: broken parallel streaks along the image's width,
    fading out toward both ends."""
    rng = np.random.default_rng(57)
    h, w = 128, 1024
    al = np.zeros((h, w), np.float32)
    x = np.linspace(0, 1, w)
    fade = np.clip(np.minimum(x, 1 - x) * 6.0, 0, 1)[None, :]
    for y0, wd in ((30, 22), (98, 22)):
        yy = np.arange(h)[:, None]
        band = np.clip(1.0 - np.abs(yy - y0) / (wd / 2.0), 0, 1) ** 0.6
        tread = 0.6 + 0.4 * (np.sin(yy * 1.9) > 0)
        al += band * tread
    brk = np.clip((_noise(rng, (h, w), (3, 24), 3) - 0.30) * 3.0, 0, 1)
    return _save(S, "dy_tyremark.png", np.clip(al * brk * fade, 0, 1) * 0.8)


def _chainlink_texture(S):
    """Chain-link fabric (50 mm mesh of 3 mm wire), one seamless tile of 4 x 4 diamonds: colour and opacity."""
    from PIL import Image, ImageDraw
    n, m = 512, 4
    al = Image.new("L", (n * 3, n * 3), 0)
    d = ImageDraw.Draw(al)
    step = n / m
    for k in range(-3 * m, 6 * m):
        d.line([(k * step, 0), (k * step + 3 * n, 3 * n)], fill=255, width=9)
        d.line([(k * step, 0), (k * step - 3 * n, 3 * n)], fill=255, width=9)
    al = np.asarray(al, np.float32)[n:2 * n, n:2 * n] / 255.0
    col = np.full((n, n, 3), 0.50, np.float32) * (0.85 + 0.3 * _noise(np.random.default_rng(9), (n, n), 40, 2,
                                                                       tileable=True))[..., None]
    return _save(S, "dy_chainlink.png", col), _save(S, "dy_chainlink_alpha.png", al)


def _sign(S, name, lines, size, fg, bg, border=None, mirror=False):
    def make(p):
        im = _text_image(lines, size, fg, bg, border)
        if mirror:
            from PIL import ImageOps
            im = ImageOps.mirror(im)
        im.save(p, compress_level=3)
    return _tex_once(S, name, make)


def _asphalt_overlays(S):
    """Opacity of two veils laid over the yard's asphalt, from a 30 m photograph of weathered paving: where it is
    darker than its mean (worn-in dirt, turning scuffs, old seams) and where it is lighter (bleached, dusty). They
    carry the large-scale variation; the close-up asphalt under them carries the grain."""
    from PIL import Image
    src = _ph("aerial_asphalt_01")
    if src is None:
        return None
    if _fresh(S, "dy_asphalt_veils"):
        lum = np.asarray(Image.open(src["diff"]).convert("L").resize((1024, 1024), Image.LANCZOS), np.float32) / 255.0
        nrm = (lum - lum.mean()) / max(float(lum.std()), 1e-6)
        _save(S, "dy_asphalt_dark.png", np.clip(-nrm * 0.24 + 0.04, 0.0, 0.62))
        _save(S, "dy_asphalt_light.png", np.clip(nrm * 0.16, 0.0, 0.40))
        _save(S, "dy_pad_veil.png", np.clip(-nrm * 0.16 + 0.03, 0.0, 0.40))           # the same wear, fainter, for concrete
    return tuple(S.folder / "textures" / f for f in ("dy_asphalt_dark.png", "dy_asphalt_light.png", "dy_pad_veil.png"))


def _crack_alpha(S):
    """Opacity of the cracks of a 3 m close-up of old asphalt (what is darker than its surroundings)."""
    from PIL import Image, ImageFilter
    src = _ph("asphalt_02")
    if src is None:
        return None
    if _fresh(S, "dy_cracks"):
        im = Image.open(src["diff"]).convert("L")
        lum = np.asarray(im, np.float32) / 255.0
        soft = np.asarray(im.filter(ImageFilter.GaussianBlur(22)), np.float32) / 255.0
        al = np.clip((soft - lum - 0.055) * 7.0, 0.0, 1.0)
        edge = np.minimum(np.minimum(*np.meshgrid(np.arange(al.shape[1]), np.arange(al.shape[0]))),
                          np.minimum(*np.meshgrid(np.arange(al.shape[1])[::-1], np.arange(al.shape[0])[::-1])))
        _save(S, "dy_cracks.png", al * np.clip(edge / 40.0, 0.0, 1.0))        # fade out at the tile's border
    return S.folder / "textures" / "dy_cracks.png"


def _yellow_paint(S, n):
    """Safety-yellow paint on steel, worked hard: the scanned paint's blotches toned down onto one yellow, its
    scratches down to dark metal, its roughness and relief kept."""
    from PIL import Image
    maps = _acg(S, "PaintedMetal001")
    if maps is None:
        return S.color(n, (0.74, 0.52, 0.025), 0.45, 0.2)
    if _fresh(S, "dy_yellow_paint"):
        col = np.asarray(Image.open(maps["diff"]).convert("RGB").resize((1024, 1024), Image.LANCZOS), np.float32) / 255.0
        lum = col.mean(-1)
        var = (lum - lum.mean()) / max(float(lum.mean()), 1e-6)
        out = np.array([0.90, 0.72, 0.10], np.float32) * np.clip(1.0 + 0.22 * var, 0.6, 1.25)[..., None]
        if "metal" in maps:
            chip = np.asarray(Image.open(maps["metal"]).convert("L").resize((1024, 1024), Image.LANCZOS), np.float32) / 255.0
            chip = np.clip((chip - 0.55) * 3.0, 0.0, 1.0)[..., None]
            out = out * (1.0 - 0.35 * chip) + np.array([0.42, 0.36, 0.28], np.float32) * 0.35 * chip
        grime = _noise(np.random.default_rng(12), (1024, 1024), 5, 4, 0.6, tileable=True)[..., None]
        out = out * (1.0 - 0.30 * np.clip((grime - 0.45) * 2.5, 0.0, 1.0))
        _save(S, "dy_yellow_paint.png", out)
    return _pbr(S, n, dict(diff=S.folder / "textures" / "dy_yellow_paint.png", rough=maps.get("rough"), nor=maps.get("nor")),
                tile=2.2, bump=0.5, rough_infl=0.8)


def _const(rgb, rough, metal=0.0):
    return lambda S, n: S.color(n, rgb, rough, metal)


def _tex(maps_of, tile, flat, **kw):
    """Material from downloaded maps; `flat` = (rgb, rough, metal) stands in when the cache does not hold them."""
    def make(S, n):
        maps = maps_of(S)
        if maps is None:
            return S.color(n, *flat)
        return _pbr(S, n, maps, tile=tile, **kw)
    return make


_MATS = {
    # bare and painted metals, plastics
    "dy_alu": _const((0.72, 0.73, 0.75), 0.32, 0.9),                # extruded aluminium rails
    "dy_steel_galv": _const((0.52, 0.54, 0.56), 0.42, 0.85),
    "dy_steel_frame": _const((0.58, 0.59, 0.61), 0.30, 0.9),        # stainless rear frame
    "dy_steel_dark": _const((0.035, 0.035, 0.04), 0.55, 0.4),       # chassis black
    "dy_black": _const((0.012, 0.012, 0.014), 0.6, 0.0),
    "dy_skin_white": _const((0.78, 0.79, 0.80), 0.36, 0.0),
    "dy_wheel_white": _const((0.72, 0.72, 0.71), 0.38, 0.0),
    "dy_hub": _const((0.16, 0.16, 0.17), 0.4, 0.7),
    "dy_red_lens": _const((0.42, 0.008, 0.01), 0.12, 0.0),
    "dy_amber_lens": _const((0.72, 0.27, 0.01), 0.12, 0.0),
    "dy_clear_lens": _const((0.80, 0.80, 0.78), 0.08, 0.0),
    "dy_glad_red": _const((0.50, 0.02, 0.02), 0.35, 0.3),
    "dy_glad_blue": _const((0.03, 0.10, 0.45), 0.35, 0.3),
    "dy_seal": _const((0.016, 0.016, 0.018), 0.72, 0.0),            # coated fabric of seals, curtains
    "dy_seal_yellow": _const((0.78, 0.58, 0.03), 0.6, 0.0),
    "dy_door_white": _const((0.74, 0.75, 0.74), 0.42, 0.0),
    "dy_door_ext": _const((0.63, 0.65, 0.66), 0.42, 0.0),
    "dy_box_grey": _const((0.30, 0.32, 0.34), 0.45, 0.2),
    "dy_frame": _const((0.17, 0.18, 0.20), 0.45, 0.4),               # painted steel door frames
    "dy_sign_plate": _const((0.75, 0.76, 0.76), 0.4, 0.5),
    "dy_spring": _const((0.03, 0.03, 0.035), 0.4, 0.6),
    "dy_cable": _const((0.02, 0.02, 0.022), 0.5, 0.0),
    "dy_facade": _const((0.66, 0.67, 0.65), 0.48, 0.0),
    "dy_facade_band": _const((0.035, 0.11, 0.30), 0.42, 0.0),
    "dy_trim": _const((0.11, 0.12, 0.13), 0.45, 0.3),
    "dy_glass": _const((0.02, 0.03, 0.035), 0.06, 0.0),
    "dy_plate_white": _const((0.80, 0.80, 0.78), 0.4, 0.0),
    "dy_tape_white": _const((0.85, 0.85, 0.83), 0.3, 0.0),
    # downloaded maps (UVs in metres)
    "dy_asphalt": _tex(lambda S: _ph("asphalt_pit_lane"), 2.0, ((0.09, 0.09, 0.09), 0.9, 0.0), bright=2.1, desat=0.75),
    "dy_asphalt_dark": lambda S, n: (lambda v: _pbr(S, n, colour=(0.012, 0.012, 0.013), rough=0.8, tile=30.0,
                                                    opacity_tex=v[0]) if v else S.color(n, (0.012, 0.012, 0.013), 0.8, 0.0,
                                                                                        opacity=0.0))(_asphalt_overlays(S)),
    "dy_asphalt_light": lambda S, n: (lambda v: _pbr(S, n, colour=(0.33, 0.33, 0.32), rough=0.9, tile=30.0,
                                                     opacity_tex=v[1]) if v else S.color(n, (0.33, 0.33, 0.32), 0.9, 0.0,
                                                                                         opacity=0.0))(_asphalt_overlays(S)),
    "dy_cracks": lambda S, n: (lambda a: _pbr(S, n, colour=(0.012, 0.011, 0.010), rough=0.9, opacity_tex=a) if a else
                               S.color(n, (0.012, 0.011, 0.010), 0.9, 0.0, opacity=0.0))(_crack_alpha(S)),
    "dy_asphalt_new": _tex(lambda S: _ph("asphalt_pit_lane"), 2.0, ((0.05, 0.05, 0.05), 0.9, 0.0), bright=1.25, desat=0.6),
    "dy_apron": _tex(lambda S: _ph("asphalt_04"), 4.0, ((0.42, 0.42, 0.40), 0.8, 0.0), tint=(1.0, 0.99, 0.95),
                     bright=1.08),
    "dy_concrete": _tex(lambda S: _ph("dirty_concrete"), 3.0, ((0.40, 0.40, 0.38), 0.85, 0.0)),
    "dy_wall_in": _tex(lambda S: {k: v for k, v in (_ph("painted_concrete_02") or {}).items() if k != "diff"} or None,
                       4.0, ((0.55, 0.56, 0.55), 0.75, 0.0), colour=(0.55, 0.56, 0.55), bump=0.6),
    "dy_plywood": _tex(lambda S: _ph("plywood"), 1.2, ((0.45, 0.38, 0.28), 0.7, 0.0), bright=2.6, desat=0.5),
    "dy_oak": _tex(lambda S: _ph("oak_wood_planks"), 1.2, ((0.36, 0.24, 0.13), 0.75, 0.0), tint=(0.95, 0.88, 0.80),
                   bright=1.0, desat=0.35, bump=1.2),
    "dy_gravel": _tex(lambda S: _ph("gravel_floor_02"), 2.0, ((0.18, 0.17, 0.15), 0.9, 0.0), tint=(0.46, 0.44, 0.41),
                      bump=0.5),
    # both grounds brought to the same dull green (their photographs average to tan and to moss yellow): half
    # desaturated, then tinted, so that stones and bare patches stay grey instead of turning blue
    "dy_grass": _tex(lambda S: _ph("leafy_grass"), 2.0, ((0.05, 0.065, 0.03), 0.95, 0.0), tint=(0.55, 0.80, 0.45),
                     bright=0.36, desat=0.55),
    "dy_field": _tex(lambda S: _ph("aerial_grass_rock"), 15.0, ((0.05, 0.065, 0.03), 0.95, 0.0), tint=(0.72, 1.0, 0.62),
                     bright=0.55, desat=0.55),
    "dy_tread": _tex(lambda S: {k: v for k, v in (_acg(S, "DiamondPlate008D") or {}).items() if k != "metal"} or None,
                     0.6, ((0.20, 0.21, 0.22), 0.45, 0.7), bright=1.7, metal=0.25, bump=1.5),
    "dy_tread_galv": _tex(lambda S: _acg(S, "DiamondPlate004"), 0.6, ((0.45, 0.46, 0.48), 0.4, 0.85), bump=1.5),
    "dy_yellow": _yellow_paint,
    "dy_rubber": _tex(lambda S: _acg(S, "Rubber004"), 0.5, ((0.02, 0.02, 0.022), 0.85, 0.0), bright=0.55),
    # generated graphics
    "dy_tape": lambda S, n: _pbr(S, n, diff=_tape_texture(S), tile=(0.4572, 0.05), rough=0.3),
    "dy_hazard": lambda S, n: _pbr(S, n, diff=_hazard_texture(S), tile=0.30, rough=0.5),
    "dy_etrack": lambda S, n: _pbr(S, n, diff=_etrack_texture(S), tile=(0.05, 0.125), rough=0.4, metal=0.6),
    "dy_paint_white": lambda S, n: _pbr(S, n, colour=(0.80, 0.80, 0.78), rough=0.55, opacity_tex=_wear_alpha(S),
                                        tile=1.7),
    "dy_paint_yellow": lambda S, n: _pbr(S, n, colour=(0.80, 0.58, 0.02), rough=0.55, opacity_tex=_wear_alpha(S),
                                         tile=1.7),
    "dy_tyremark": lambda S, n: _pbr(S, n, colour=(0.012, 0.012, 0.013), rough=0.7, opacity_tex=_tyremark_alpha(S)),
    "dy_seal_tar": _const((0.02, 0.02, 0.021), 0.75, 0.0),
    "dy_chainlink": lambda S, n: (lambda t: _pbr(S, n, diff=t[0], opacity_tex=t[1], tile=0.20, rough=0.4, metal=0.7))(
        _chainlink_texture(S)),
    # lit lenses and glazing (emissive surfaces only: no light sources are added here)
    "dy_led_green": lambda S, n: _pbr(S, n, colour=(0.02, 0.6, 0.08), rough=0.2, glow=((0.05, 1.0, 0.15), 1800.0)),
    "dy_led_red": lambda S, n: _pbr(S, n, colour=(0.6, 0.02, 0.02), rough=0.2, glow=((1.0, 0.04, 0.03), 1800.0)),
    "dy_led_green_off": _const((0.02, 0.10, 0.03), 0.2, 0.0),
    "dy_led_red_off": _const((0.12, 0.012, 0.012), 0.2, 0.0),
    "dy_daylight": lambda S, n: _pbr(S, n, colour=(0.8, 0.82, 0.85), rough=0.3, glow=((0.85, 0.92, 1.0), 1300.0)),
}
for _k in range(3):
    _MATS[f"dy_stain{_k}"] = (lambda k: lambda S, n: _pbr(S, n, colour=(0.035, 0.030, 0.024), rough=0.4,
                                                           opacity_tex=_stain_alpha(S, k)))(_k)


def _mat(S, name):
    if name not in S.mats:
        if name not in _MATS:
            raise KeyError(f"dockyard material {name!r} is not defined")
        _MATS[name](S, name)
    return S.mats[name]


def _image_mat(S, name, image, rough=0.4, opacity_tex=None, **kw):
    """A material showing one generated image once over a 0..1 UV rectangle."""
    if name not in S.mats:
        _pbr(S, name, diff=image, rough=rough, opacity_tex=opacity_tex, **kw)
    return name


_X, _Y, _Z = np.eye(3)


# =====================================================================================================================
# trailer: a 53 ft sheet-and-post dry van in its own frame (x from the rear sill toward the nose, y left, z up, origin
# on the floor top at the rear sill centre: dock.add_trailer)
# =====================================================================================================================
def _shapes(world):
    """{owner: {tag: (centre, quat, half, body)}} of the box shapes labelled '<owner>:<tag>' (trailers, levelers, ramp
    lips)."""
    sc = world.scene
    names = set(sc.trailers) | set(sc.levelers) | set(sc.ramp_lips)
    sb = world.model.shape_body.numpy()
    st = world.model.shape_type.numpy()
    scale = world.model.shape_scale.numpy()
    tf = world.model.shape_transform.numpy()
    out = {n: {} for n in names}
    for s in range(world.model.shape_count):
        n, _, tag = str(world.model.shape_label[s]).partition(":")
        if n in out and tag:
            half = box_half(world, s, st, scale)
            if half is not None:
                out[n][tag] = (np.asarray(tf[s, :3], float), np.asarray(tf[s, 3:], float), np.asarray(half, float),
                               int(sb[s]))
    return out


def _trailer_dims(sh):
    """The van's dimensions read from its collision shapes."""
    fl, wl, nose, roof, post, hdr, sill, icc = (sh[k] for k in ("floor", "wall_l", "nose", "roof", "rear_post_l",
                                                                "rear_header", "rear_sill", "icc_bar"))
    return dict(L=2 * fl[2][0], W=2 * fl[2][1], sub=2 * fl[2][2], H=2 * wl[2][2], wall=2 * wl[2][1], Wo=2 * nose[2][1],
                nose_t=2 * nose[2][0], roof_t=2 * roof[2][2], df=2 * post[2][0],
                post_y0=abs(post[0][1]) - post[2][1], post_y1=abs(post[0][1]) + post[2][1],
                hdr_h=2 * hdr[2][2], sill_h=2 * sill[2][2], icc_c=icc[0], icc_h=icc[2])


def _proto_wheel(B):
    """One wheel of a dual pair: a 275/70R22.5 tyre (0.96 m, four ribs) on a hub-piloted steel disc wheel with ten
    nuts and five hand holes. Axle along +y through the origin, outboard face at +y."""
    half = [(0.286, -0.118), (0.300, -0.128), (0.345, -0.137), (0.400, -0.138), (0.440, -0.132), (0.462, -0.122),
            (0.474, -0.112), (0.480, -0.100)]
    tread = []
    for g in (-0.066, -0.022, 0.022, 0.066):
        tread += [(0.480, g - 0.007), (0.467, g - 0.004), (0.467, g + 0.004), (0.480, g + 0.007)]
    B.revolve("dy_rubber", half + tread + [(r, -t) for r, t in half[::-1]], axis=_Y, n=40)
    B.revolve("dy_wheel_white", [(0.300, 0.105), (0.303, 0.118), (0.292, 0.124), (0.276, 0.112), (0.262, 0.085),
                                 (0.258, 0.020), (0.250, 0.012), (0.165, 0.012), (0.150, 0.030), (0.118, 0.045)],
              axis=_Y, n=40)
    B.revolve("dy_hub", [(0.118, 0.045), (0.108, 0.110), (0.085, 0.135), (0.0, 0.140)], axis=_Y, n=24)
    B.revolve("dy_hub", [(0.0, -0.112), (0.290, -0.112)], axis=_Y, n=24)
    for k in range(10):
        a = 2 * math.pi * k / 10
        c, s = 0.1429 * math.cos(a), 0.1429 * math.sin(a)
        B.cyl("dy_steel_galv", (c, 0.018, s), (c, 0.056, s), 0.015, n=6)
    for k in range(5):
        a = 2 * math.pi * (k + 0.5) / 5
        er = np.array([math.cos(a), 0.0, math.sin(a)])
        et = np.cross(er, _Y)
        c = 0.208 * er + 0.0128 * _Y
        B.poly("dy_black", [c + 0.045 * math.cos(f) * et + 0.024 * math.sin(f) * er
                            for f in np.linspace(0, 2 * math.pi, 12, endpoint=False)])


def _proto_chock(B):
    """Rubber wheel chock: a ribbed wedge 0.25 m long, 0.20 m wide, 0.18 m tall; the tyre side at +x."""
    prof = [(-0.125, 0.0), (0.125, 0.0), (0.125, 0.045), (0.085, 0.10), (0.03, 0.165), (-0.01, 0.18), (-0.04, 0.17),
            (-0.125, 0.03)]
    B.prism("dy_seal_yellow", prof, (0.0, 0.10, 0.0), (0.0, -0.10, 0.0), up=_Z)
    B.cyl("dy_steel_galv", (-0.10, -0.085, 0.045), (-0.10, 0.085, 0.045), 0.006, n=6)


_PROTOS.update(wheel=_proto_wheel, chock=_proto_chock)


def _trailer_body(S, B, D, doors, livery, unit, roof="dy_roof_translucent"):
    """Draw a van's body into B (its frame); doors 'open' (folded back along the sides) or 'closed'. The floor,
    liners, nose, roof and rear frame lie on the collision shapes' faces (D)."""
    L, W, H, sub, Wo = D["L"], D["W"], D["H"], D["sub"], D["Wo"]
    rt, nt, df = D["roof_t"], D["nose_t"], D["df"]
    yw, yo = W / 2, Wo / 2
    xn, zt = L + nt, H + rt
    rp = min(0.012, 0.45 * (yo - yw))          # the rails stand this proud of the side and nose sheets
    ys, xs = yo - rp, xn - rp
    lv = LIVERIES[livery % len(LIVERIES)]
    # ------------------------------------------------------------------ floor: laminated hardwood on steel cross-members
    B.rect("dy_oak", (L / 2, 0.0, 0.0), _X, _Y, (L, W))
    B.box("dy_tread_galv", (0.20, 0.0, 0.0015), (0.40, W - 0.004, 0.003))
    nb = max(1, int(round(W / 0.305)))
    bw = W / nb
    for x in np.arange(0.55, L - 0.1, 0.3048):
        for j in range(nb):
            for f in (0.25, 0.75):
                B.inst("screw", (x, -yw + (j + f) * bw, 0.0))
    B.rect("dy_steel_dark", (L / 2, 0.0, -0.04), _Y, _X, (W, L))
    for x in np.arange(0.20, L, 0.3048):
        B.box("dy_steel_dark", (x, 0.0, -(0.04 + sub) / 2), (0.05, W, sub - 0.04))
    # ------------------------------------------------------------------ side walls
    skin = _image_mat(S, f"dy_skin_{livery}_{unit}", _skin_texture(S, livery, unit, L, H - 0.02), rough=0.36)
    zl = (0.30 + H - 0.12) / 2
    for sg in (1, -1):
        yi = sg * yw
        B.rect("dy_plywood", (L / 2, yi, zl), sg * _X, _Z, (L, H - 0.42), swap=True)            # lining, grain upright
        B.box("dy_steel_galv", (L / 2, yi - sg * 0.001, 0.15), (L, 0.002, 0.30))               # scuff band
        B.box("dy_alu", (L / 2, yi - sg * 0.001, H - 0.06), (L, 0.002, 0.12))                  # roof rail, inside
        for x in np.arange(1.2192, L - 0.05, 1.2192):
            B.box("dy_black", (x, yi - sg * 0.0004, zl), (0.004, 0.0008, H - 0.42))
        for z in (1.0625, 1.9375):
            B.box("dy_etrack", (L / 2, yi - sg * 0.0015, z), (L - 0.4, 0.003, 0.125))
        B.rect(skin, (L / 2, sg * ys, (H - 0.02) / 2), -sg * _X, _Z, (L, H - 0.02), uv01=True)
        B.box("dy_alu", (xn / 2, sg * (ys + rp / 2), -0.07), (xn, rp, 0.34), bevel=0.003)      # bottom rail
        B.box("dy_alu", (xn / 2, sg * (ys + rp / 2), zt - 0.07), (xn, rp, 0.15), bevel=0.003)  # top rail
        B.rect("dy_tape", (L / 2, sg * (yo + 0.0006), -0.03), -sg * _X, _Z, (L - 0.5, 0.05))
        Rr = _RZ[sg]
        for k in range(1, int(L / 0.6096) + 1):
            for z in np.arange(0.20, H - 0.14, 0.10):
                B.inst("rivet", (k * 0.6096, sg * ys, z), Rr)
        for x in np.arange(0.05, L, 0.10):
            for z in (0.07, -0.19, H - 0.05):
                B.inst("rivet", (x, sg * yo, z), Rr)
        for x, m in ((L - 0.30, "dy_amber_lens"), (L / 2, "dy_amber_lens"), (0.22, "dy_red_lens")):
            B.box(m, (x, sg * (yo + 0.006), 0.05), (0.10, 0.012, 0.036), bevel=0.004)
    # ------------------------------------------------------------------ nose
    B.rect("dy_skin_white", (xs, 0.0, (H - 0.02) / 2), _Y, _Z, (Wo, H - 0.02))
    B.rect("dy_plywood", (L, 0.0, (0.5 + H) / 2), -_Y, _Z, (W, H - 0.5), swap=True)
    B.box("dy_steel_galv", (L - 0.001, 0.0, 0.25), (0.002, W, 0.5))
    for sg in (1, -1):
        B.box("dy_alu", (xn - 0.036, sg * (yo - 0.036), (zt - 0.24) / 2), (0.072, 0.072, zt + 0.24), bevel=0.012)
        B.box("dy_amber_lens", (xn + 0.006, sg * (yo - 0.14), zt - 0.07), (0.012, 0.10, 0.036), bevel=0.004)
    B.box("dy_alu", (xs + rp / 2, 0.0, -0.07), (rp, Wo, 0.34), bevel=0.003)
    B.box("dy_alu", (xs + rp / 2, 0.0, zt - 0.07), (rp, Wo, 0.15), bevel=0.003)
    for y in np.arange(-1.0, 1.01, 0.4):
        for z in np.arange(0.20, H - 0.14, 0.10):
            B.inst("rivet", (xs, y, z), _RX[1])
    B.box("dy_steel_dark", (xs + 0.012, 0.0, 0.95), (0.024, 0.42, 0.20), bevel=0.004)          # air and light connections
    B.cyl("dy_glad_red", (xs + 0.024, -0.13, 0.95), (xs + 0.085, -0.13, 0.93), 0.028, n=12)
    B.cyl("dy_glad_blue", (xs + 0.024, 0.13, 0.95), (xs + 0.085, 0.13, 0.93), 0.028, n=12)
    B.cyl("dy_black", (xs + 0.024, 0.0, 0.96), (xs + 0.07, 0.0, 0.96), 0.036, n=14)
    B.box("dy_alu", (xs + 0.02, -0.78, 1.25), (0.04, 0.30, 0.40), bevel=0.006)                 # document holder
    logo = _image_mat(S, f"dy_nose_{livery}", _sign(S, f"dy_nose_{livery}.png", [(lv["name"], 1.0)], (768, 192),
                                                    lv["colour"], (228, 229, 231)), rough=0.36)
    B.rect(logo, (xs + 0.0008, 0.0, H - 0.75), _Y, _Z, (1.6, 0.40), uv01=True)
    # ------------------------------------------------------------------ roof: sheet on bows, sunk into the roof box
    if roof:                                   # one glass-fibre sheet: daylight comes through it, the bows show against it
        B.rect("dy_roof_translucent", (L / 2, 0.0, H + 0.04), _X, _Y, (L, W))
        if roof == "dy_roof_daylight":         # the sheet's underside glowing with a daylight that the preset has not
            B.rect(roof, (L / 2, 0.0, H + 0.038), _Y, _X, (W, L))
    else:
        B.rect("dy_roof_out", (L / 2, 0.0, zt), _X, _Y, (L, W))
        B.rect("dy_roof_in", (L / 2, 0.0, H + 0.03), _Y, _X, (W, L))
    for x in np.arange(0.3048, L, 0.6096):
        B.box("dy_steel_galv", (x, 0.0, H + 0.015), (0.045, W, 0.03))
    for sg in (1, -1):
        B.box("dy_alu", (L / 2, sg * (yw + yo) / 2, H + rt / 2), (L, yo - yw, rt))
    B.box("dy_alu", (L + nt / 2, 0.0, H + rt / 2), (nt, Wo, rt))
    # ------------------------------------------------------------------ rear frame (posts, header, sill) and its lamps
    py, pw, hh, sh_ = (D["post_y0"] + D["post_y1"]) / 2, D["post_y1"] - D["post_y0"], D["hdr_h"], D["sill_h"]
    for sg in (1, -1):
        B.box("dy_steel_frame", (-df / 2, sg * py, H / 2), (df, pw, H), bevel=0.004)
    hz0 = D.get("hdr_z0", H)
    B.box("dy_steel_frame", (-df / 2, 0.0, hz0 + hh / 2), (df, Wo, hh), bevel=0.004)
    B.box("dy_steel_frame", (-df / 2, 0.0, -sh_ / 2), (df, Wo, sh_), bevel=0.003)
    xr = -df
    for y in (0.0, 0.16, -0.16, yo - 0.12, -(yo - 0.12)):
        B.box("dy_red_lens", (xr - 0.005, y, hz0 + hh / 2), (0.012, 0.10, 0.04), bevel=0.004)
    zl_ = -0.62 * sh_
    for y in (0.92, 1.07, -0.92, -1.07):
        B.cyl("dy_black", (xr, y, zl_), (xr - 0.006, y, zl_), 0.062, n=20)
        B.cyl("dy_red_lens", (xr - 0.006, y, zl_), (xr - 0.015, y, zl_), 0.052, n=20)
    B.rect("dy_tape", (xr - 0.0008, 0.0, -0.03), -_Y, _Z, (Wo - 0.08, 0.05))
    plate = _image_mat(S, f"dy_plate_{unit}", _sign(S, f"dy_plate_{unit}.png", [(f"{lv['code'][:3]} {unit % 10000:04d}", 1.0)],
                                                    (256, 128), (20, 30, 90), (236, 236, 230), border=(20, 30, 90)))
    B.box("dy_steel_galv", (xr - 0.002, 0.42, zl_), (0.004, 0.32, 0.165))
    B.rect(plate, (xr - 0.0045, 0.42, zl_), -_Y, _Z, (0.305, 0.152), uv01=True)
    for sg in (1, -1):
        B.rect("dy_tape_white", (xr - 0.0008, sg * (yo - 0.19), hz0 + hh - 0.035), -_Y, _Z, (0.30, 0.05))
        B.rect("dy_tape_white", (xr - 0.0008, sg * py, hz0 - 0.17), -_Y, _Z, (0.05, 0.30))
    if doors == "open":
        _rear_doors_open(B, D)
    else:
        _rear_doors_closed(S, B, D, livery, unit)


def _rear_guard(B, D):
    """Rear impact guard under the sill: the bar (the docked trailer's restraint hooks it), its uprights and braces."""
    sub = D["sub"]
    ic, ih = D["icc_c"], D["icc_h"]
    B.box("dy_steel_frame", ic, 2 * ih, bevel=0.004)
    B.rect("dy_tape", (ic[0] - ih[0] - 0.0008, ic[1], ic[2]), -_Y, _Z, (2 * ih[1] - 0.04, 0.05))
    for sg in (1, -1):
        zt_ = ic[2] + ih[2]
        B.box("dy_steel_frame", (ic[0], sg * 0.72, (zt_ - sub) / 2), (0.07, 0.07, -sub - zt_), bevel=0.003)
        B.beam("dy_steel_frame", (ic[0] + 0.03, sg * 0.72, ic[2]), (ic[0] + 0.66, sg * 0.72, -sub - 0.03), (0.05, 0.05))


def _kingpin(B, D):
    """Kingpin and its coupler plate under the nose."""
    L, sub = D["L"], D["sub"]
    B.box("dy_steel_dark", (L - 0.85, 0.0, -sub - 0.006), (1.6, 1.15, 0.012), bevel=0.003)
    for z0, z1, r in ((-0.012, -0.045, 0.036), (-0.045, -0.075, 0.0254), (-0.075, -0.095, 0.036)):
        B.cyl("dy_hub", (L - 0.914, 0.0, -sub + z0), (L - 0.914, 0.0, -sub + z1), r, n=16)


def _trailer_gear(S, B, D, h, livery, gear_x, tandem_x, track):
    """What hangs under a van: rear impact guard, sliding tandem, landing gear, kingpin. h: floor top above ground."""
    _rear_guard(B, D)
    _running_gear(S, B, D, h, tandem_x, track, LIVERIES[livery % len(LIVERIES)])
    _landing_gear(B, D, h, gear_x, track)
    _kingpin(B, D)


def _rear_doors_open(B, D):
    """Both leaves swung 270 degrees and held against the side sheets: what shows is the leaf's inner face with its
    gasket; the lock rods lie between leaf and side."""
    H, Wo, y1 = D["H"], D["Wo"], D["post_y1"]
    yo = Wo / 2
    lw, t = yo - 0.02, 0.022
    zc, zh = (H + 0.03) / 2, H + 0.02
    for sg in (1, -1):
        B.box("dy_skin_white", (0.012 + lw / 2, sg * (y1 - t / 2), zc), (lw, t, zh), bevel=0.003)
        yf = sg * (y1 + 0.0006)
        for cx, cz, sx, sz in ((0.012 + lw / 2, zc - zh / 2 + 0.03, lw - 0.02, 0.03), (0.012 + lw / 2, zc + zh / 2 - 0.03, lw - 0.02, 0.03),
                               (0.037, zc, 0.03, zh - 0.03), (lw - 0.013, zc, 0.03, zh - 0.03)):
            B.box("dy_black", (cx, yf, cz), (sx, 0.0012, sz))
        for fx in (0.28, 0.72):
            x = 0.012 + lw * fx
            B.cyl("dy_steel_galv", (x, sg * (y1 - t - 0.02), -0.02), (x, sg * (y1 - t - 0.02), H + 0.05), 0.0135, n=10)
        for z in np.linspace(0.22, H - 0.25, 5):
            B.cyl("dy_steel_frame", (-0.012, sg * (y1 + 0.004), z - 0.06), (-0.012, sg * (y1 + 0.004), z + 0.06), 0.014, n=10)
        B.box("dy_steel_galv", (lw - 0.05, sg * (yo + 0.02), 1.30), (0.05, 0.04, 0.02))            # hold-back hook


def _rear_doors_closed(S, B, D, livery, unit):
    """Both leaves shut on the rear frame: sheet with the carrier's mark, two lock rods per leaf with their guides,
    keepers and handles, five strap hinges per leaf."""
    H, Wo, df = D.get("hdr_z0", D["H"]), D["Wo"], D["df"]
    yo = Wo / 2
    xr, t = -df, 0.03
    lw = yo - 0.035
    zh = H + 0.02
    zc = zh / 2 + 0.01
    tot = lw + 0.004
    face = _image_mat(S, f"dy_reardoor_{livery}_{unit}", _closed_door_texture(S, livery, unit, 2 * tot, zh), rough=0.36)
    for sg in (1, -1):
        ya, yb = (tot, 0.004) if sg > 0 else (-0.004, -tot)
        B.box("dy_skin_white", (xr - t / 2, (ya + yb) / 2, zc), (t, lw, zh), bevel=0.003)
        x = xr - t - 0.0006
        B.poly(face, [(x, ya, zc - zh / 2), (x, yb, zc - zh / 2), (x, yb, zc + zh / 2), (x, ya, zc + zh / 2)],
               uv=[((tot - ya) / (2 * tot), 0.0), ((tot - yb) / (2 * tot), 0.0), ((tot - yb) / (2 * tot), 1.0),
                   ((tot - ya) / (2 * tot), 1.0)])
        for yy in (ya - sg * 0.012, yb + sg * 0.012):                                         # leaf edge frames
            B.box("dy_alu", (xr - t - 0.002, yy, zc), (0.004, 0.024, zh))
        xrod = xr - t - 0.024
        for fy in (0.27, 0.73):
            y = sg * (0.004 + lw * fy)
            B.cyl("dy_steel_galv", (xrod, y, -0.035), (xrod, y, H + 0.07), 0.0135, n=10)
            for z in (0.14, 0.75, 1.65, H - 0.14):
                B.box("dy_steel_galv", (xr - t - 0.016, y, z), (0.032, 0.062, 0.036), bevel=0.003)
            for z in (-0.03, H + 0.065):
                B.box("dy_steel_frame", (xr - 0.02, y, z), (0.04, 0.085, 0.05), bevel=0.004)
            B.box("dy_steel_galv", (xrod - 0.014, y + sg * 0.19, 1.08), (0.012, 0.40, 0.034), bevel=0.003)
            B.box("dy_steel_frame", (xr - t - 0.004, y + sg * 0.33, 1.08), (0.008, 0.075, 0.10), bevel=0.002)
        for z in np.linspace(0.22, H - 0.25, 5):
            B.cyl("dy_steel_frame", (xr - t - 0.008, sg * (yo - 0.012), z - 0.055), (xr - t - 0.008, sg * (yo - 0.012), z + 0.055),
                  0.013, n=10)
            B.box("dy_steel_frame", (xr - t - 0.003, sg * (yo - 0.17), z), (0.006, 0.30, 0.05), bevel=0.002)
    B.box("dy_black", (xr - t - 0.001, 0.0, zc), (0.004, 0.012, zh))                           # centre seal


def _running_gear(S, B, D, h, tandem_x, track, lv):
    """Sliding tandem under the rear: subframe rails, two axles on trailing beams and air springs, dual wheels, mud
    flaps, air tanks. The tyres stand on the ground h below the floor where dock.TrailerSupports carries the van."""
    sub = D["sub"]
    r = min(0.50, (h - sub - 0.03) / 2)
    s = r / 0.48
    za = -h + r
    zr0, zr1, ry = -sub, -sub - 0.20, 0.43
    for sg in (1, -1):
        B.box("dy_steel_dark", (tandem_x, sg * ry, (zr0 + zr1) / 2), (3.3, 0.075, 0.20), bevel=0.004)
    for dx in (-1.5, 0.0, 1.5):
        B.box("dy_steel_dark", (tandem_x + dx, 0.0, zr1 + 0.06), (0.07, 2 * ry, 0.12))
    for dx in (-0.62, 0.62):
        xa = tandem_x + dx
        B.cyl("dy_steel_dark", (xa, -(track / 2 + 0.02), za), (xa, track / 2 + 0.02, za), 0.062, n=14, caps=False)
        for sg in (1, -1):
            for off, flip in ((0.165, False), (-0.165, True)):
                B.inst("wheel", (xa, sg * (track / 2 + off * s), za), np.eye(3) if (sg > 0) != flip else _rot(yaw=math.pi), s)
            B.cyl("dy_hub", (xa, sg * (track / 2 - 0.42 * s), za), (xa, sg * (track / 2 - 0.30 * s), za), 0.20 * s, n=20)
            if zr1 - za > 0.06:
                B.box("dy_steel_dark", (xa + 0.48, sg * ry, (zr1 + za) / 2), (0.11, 0.07, zr1 - za + 0.04))
                B.cyl("dy_black", (xa - 0.22, sg * ry, za + 0.07), (xa - 0.22, sg * ry, zr1), 0.115, n=18)
            B.beam("dy_steel_dark", (xa + 0.50, sg * ry, za + 0.05), (xa - 0.30, sg * ry, za + 0.02), (0.07, 0.10))
    flap = _image_mat(S, f"dy_flap_{lv['code']}", _sign(S, f"dy_flap_{lv['code']}.png", [(lv["name"], 1.0)], (256, 96),
                                                        (228, 228, 224), (10, 10, 12)), rough=0.7)
    xf = tandem_x - 0.62 - r - 0.14
    fh = min(0.76, h - sub - 0.035 - 0.18)
    for sg in (1, -1):
        B.box("dy_steel_dark", (xf, sg * track / 2, -sub - 0.02), (0.03, 0.68, 0.03))
        B.box("dy_rubber", (xf, sg * track / 2, -sub - 0.035 - fh / 2), (0.008, 0.61, fh), bevel=0.002)
        B.rect(flap, (xf - 0.0045, sg * track / 2, -sub - 0.30), -_Y, _Z, (0.44, 0.165), uv01=True)
    for dx in (1.95, 2.30):
        B.cyl("dy_steel_dark", (tandem_x + dx, -0.36, -sub - 0.16), (tandem_x + dx, 0.36, -sub - 0.16), 0.14, n=18)


def _landing_gear(B, D, h, gear_x, track):
    """Two-speed landing gear: square legs with sand shoes on the ground, cross shaft, gearbox and crank on the left
    leg, braces to the floor and between the legs."""
    sub = D["sub"]
    gy, z0, zs = 0.9 * track / 2, -sub, -h
    hl = min(0.62, h - sub - 0.16)
    for sg in (1, -1):
        B.box("dy_steel_dark", (gear_x, sg * gy, z0 - hl / 2), (0.11, 0.11, hl), bevel=0.006)
        B.box("dy_steel_galv", (gear_x, sg * gy, (z0 - hl + 0.06 + zs + 0.03) / 2), (0.085, 0.085, (z0 - hl + 0.06) - (zs + 0.03)),
              bevel=0.004)
        B.box("dy_steel_dark", (gear_x, sg * gy, zs + 0.0125), (0.30, 0.26, 0.025), bevel=0.005)
        B.box("dy_steel_dark", (gear_x, sg * gy, zs + 0.05), (0.10, 0.12, 0.06), bevel=0.004)
        B.beam("dy_steel_dark", (gear_x - 0.05, sg * gy, z0 - hl + 0.06), (gear_x - 0.80, sg * gy, z0 - 0.03), (0.05, 0.035))
    zm = z0 - hl / 2
    B.cyl("dy_steel_galv", (gear_x - 0.075, -gy, zm), (gear_x - 0.075, gy, zm), 0.016, n=10)
    B.beam("dy_steel_dark", (gear_x - 0.062, gy, z0 - 0.05), (gear_x - 0.062, -gy, z0 - hl + 0.04), (0.006, 0.045))
    B.beam("dy_steel_dark", (gear_x - 0.069, -gy, z0 - 0.05), (gear_x - 0.069, gy, z0 - hl + 0.04), (0.006, 0.045))
    B.box("dy_steel_dark", (gear_x, gy + 0.10, zm), (0.14, 0.10, 0.17), bevel=0.008)
    B.tube("dy_steel_galv", [(gear_x, gy + 0.15, zm), (gear_x, gy + 0.27, zm), (gear_x - 0.02, gy + 0.29, zm - 0.02),
                             (gear_x - 0.33, gy + 0.29, zm - 0.26), (gear_x - 0.35, gy + 0.31, zm - 0.27),
                             (gear_x - 0.35, gy + 0.42, zm - 0.27)], 0.011, n=8)


def _draw_trailers(S, world, shapes):
    """The simulated vans (docked, or spotted at the yard ramp), each on its body."""
    sc = world.scene
    Tp = P.TRAILER
    for k, (name, tr) in enumerate(sorted(sc.trailers.items())):
        D = _trailer_dims(shapes[name])
        B = _Build()
        lit = ROOF_DAYLIGHT_NITS > 0 and sc.c.preset != "yard"
        _trailer_body(S, B, D, "open", k, 1047 + 31 * k, roof="dy_roof_daylight" if lit else "dy_roof_translucent")
        _trailer_gear(S, B, D, tr["bed_z"] - tr["ground_z"], k, Tp["gear_from_rear"], Tp["tandem_from_rear"], Tp["track"])
        B.flush(S, f"/Lab/body_{tr['body']}/van")


# Luminance (cd/m2) the translucent trailer roofs are given in the presets that have no sky (dock): 0 leaves a docked
# trailer as dark as the render's lights make it. A glass-fibre roof under an overcast sky shows a few hundred cd/m2
# from inside; a value here stands in for that daylight (it lights the trailer's inside: the lead's decision, off).
ROOF_DAYLIGHT_NITS = 0.0


def _translucent(S, n):
    """Glass-fibre roof sheet: thin-walled, about half of the light that is not reflected passes through diffusely."""
    T, Gf = S.Sdf.ValueTypeNames, S.Gf
    return S.mdl(n, "OmniSurface", {
        "diffuse_reflection_weight": (T.Float, 0.9), "diffuse_reflection_color": (T.Color3f, Gf.Vec3f(0.82, 0.82, 0.78)),
        "specular_reflection_roughness": (T.Float, 0.45), "specular_reflection_weight": (T.Float, 0.5),
        "enable_diffuse_transmission": (T.Bool, True), "subsurface_weight": (T.Float, 0.55),
        "subsurface_transmission_color": (T.Color3f, Gf.Vec3f(0.92, 0.90, 0.80)),
        "subsurface_scattering_color": (T.Color3f, Gf.Vec3f(0.92, 0.90, 0.80)), "thin_walled": (T.Bool, True)})


_MATS.update(dy_roof_out=_const((0.66, 0.67, 0.69), 0.42, 0.8), dy_roof_in=_const((0.70, 0.71, 0.70), 0.5, 0.0),
             dy_roof_translucent=_translucent,
             dy_roof_daylight=lambda S, n: _pbr(S, n, colour=(0.82, 0.82, 0.78), rough=0.5,
                                                glow=((1.0, 0.98, 0.92), math.pi * ROOF_DAYLIGHT_NITS)))


# =====================================================================================================================
# dock wall: the wall boxes, sectional doors, seals, bumpers, levelers, lights and signs
# =====================================================================================================================
def _cladding(B, mat, x0, x1, z0, z1, y, out, pitch=0.3048, depth=0.028):
    """Ribbed metal wall sheet on the plane y facing `out` (+1: +y, -1: -y) between x0..x1 and z0..z1. The ribs' crowns
    lie on the plane and the valleys are sunk `depth` behind it, so the sheet stays inside the wall's collision box."""
    pts = []
    x = math.floor(x0 / pitch) * pitch
    while x < x1 + 1e-9:
        for dx, dy in ((0.0, 0.0), (0.62 * pitch, 0.0), (0.72 * pitch, 1.0), (0.90 * pitch, 1.0)):
            pts.append((x + dx, dy))
        x += pitch
    pts.append((x, 0.0))
    Pp = np.array(pts)
    xi = np.unique(np.clip(np.r_[x0, Pp[:, 0], x1], x0, x1))
    di = np.interp(xi, Pp[:, 0], Pp[:, 1])
    A = np.stack([xi, y - out * depth * di, np.full(len(xi), float(z0))], 1)
    T = A.copy()
    T[:, 2] = z1
    B.sheet(mat, A, T, flip=(out > 0))


def _bumper(B, half):
    """Laminated dock bumper inside its collision box (centred on the origin; the wall behind at -y, the trailer meets
    +y): rubber plies clamped between two steel angles on through rods."""
    hx, hy, hz = half
    n = max(4, int(round((2 * hx - 0.012) / 0.0127)))
    w = (2 * hx - 0.012) / n
    for k in range(n):
        dep = 2 * hy - (0.003 if k % 3 == 0 else 0.0) - 0.002 * ((k * 7) % 3)      # worn unevenly, never proud of the box
        B.box("dy_rubber", (-hx + 0.006 + (k + 0.5) * w, -hy + dep / 2, 0.0), (w, dep, 2 * hz), bevel=0.0015)
    for sg in (-1, 1):
        B.box("dy_steel_dark", (sg * (hx - 0.003), -hy + 0.045, 0.0), (0.006, 0.09, 2 * hz))
        B.box("dy_steel_dark", (sg * (hx + 0.035), -hy + 0.003, 0.0), (0.07, 0.006, 2 * hz))
        for dz in (-0.6 * hz, 0.6 * hz):
            B.inst("bolt", (sg * (hx + 0.04), -hy + 0.006, dz), _RZ[1])
            B.cyl("dy_steel_galv", (sg * hx, -hy + 0.05, dz), (sg * (hx + 0.012), -hy + 0.05, dz), 0.012, n=6)


def _seal_side(B, w, d, h, inner):
    """Foam side pad of a dock seal in coated fabric (base centre on the origin, wall at y = 0, the pad out to y = d):
    yellow guide stripe on the edge toward the opening (`inner` = its x sign), wear pleats over the lower half."""
    B.box("dy_seal", (0.0, d / 2, h / 2), (w, d, h), bevel=0.02)
    for k in range(8):
        B.box("dy_seal", (0.0, d + 0.003, 0.21 + k * 0.17), (w - 0.03, 0.006, 0.19), bevel=0.002, R=_rot(roll=0.035))
    B.box("dy_seal_yellow", (inner * (w / 2 - 0.065), d + 0.0075, h / 2), (0.07, 0.0015, h - 0.12))


def _seal_head(B, w, d):
    """Head pad of a dock seal over the door (origin: the door head's centre in the wall face, the pad out to y = d)."""
    B.box("dy_seal", (0.0, d / 2, 0.15), (w, d, 0.30), bevel=0.02)


def _seal_curtain(B, w, drop):
    """The head pad's fabric curtain hanging `drop` below the door head (origin: the curtain's top centre)."""
    B.box("dy_seal", (0.0, 0.0, -drop / 2), (w, 0.006, drop), bevel=0.002)
    B.box("dy_seal_yellow", (0.0, 0.0038, -drop / 2), (0.09, 0.0015, drop * 0.85))


DOOR_OVERLAP = 0.045          # the door leaf covers the opening's edges by this much on each side


def _door_track_top(h):
    return 2 * h + 0.30


def _sectional_door(B, w, h, up, ext_only=False, S=None, number=None):
    """Insulated sectional overhead door on the room side of the wall (origin: opening centre on the floor in the
    wall's inner face, +y into the room): ribbed steel sections 24 in tall, a row of vision lites, hinges and
    rollers, bottom seal. up: raised above the opening (vertical lift) instead of closed in it. ext_only: the leaf
    of total width w as seen from outside only (its room side is never seen). number: the door's number as a decal
    on the room side of the bottom section (the same face as the wall placard above the opening, which the raised leaf
    covers), between its bottom bracket and the first hinges."""
    n = max(3, int(round(h / 0.6096)))
    hs = h / n
    z0 = h + 0.04 if up else 0.0
    ww = w if ext_only else w + 2 * DOOR_OVERLAP
    lites = [(k - (nl - 1) / 2) * 0.82 for nl in (3 if ww < 3.3 else 4,) for k in range(nl)]
    if ext_only:
        for k in range(n):
            zc = z0 + (k + 0.5) * hs
            B.box("dy_door_ext", (0.0, 0.0345, zc), (ww, 0.045, hs - 0.004), bevel=0.004)
            for f in (1 / 3, 2 / 3):
                B.box("dy_door_ext", (0.0, 0.0108, z0 + (k + f) * hs), (ww - 0.02, 0.002, 0.012), bevel=0.001)
        for x in lites:
            B.box("dy_black", (x, 0.0345, z0 + 2.5 * hs), (0.62, 0.060, 0.26), bevel=0.006)
            B.rect("dy_glass", (x, 0.0035, z0 + 2.5 * hs), _X, _Z, (0.54, 0.18))
        B.box("dy_black", (0.0, 0.0345, z0 + 0.012), (ww, 0.05, 0.03), bevel=0.006)
        return
    for k in range(n):
        zc = z0 + (k + 0.5) * hs
        B.box("dy_door_white", (0.0, 0.0345, zc), (ww, 0.045, hs - 0.004), bevel=0.004)
        B.rect("dy_door_ext", (0.0, 0.0116, zc), _X, _Z, (ww - 0.012, hs - 0.014))
        for f in (1 / 3, 2 / 3):
            z = z0 + (k + f) * hs
            B.box("dy_door_white", (0.0, 0.0578, z), (ww - 0.12, 0.002, 0.012), bevel=0.001)
            B.box("dy_door_ext", (0.0, 0.0108, z), (ww - 0.02, 0.002, 0.012), bevel=0.001)
        for x in (-(ww / 2 - 0.03), 0.0, ww / 2 - 0.03):
            B.box("dy_steel_galv", (x, 0.0585, zc), (0.05, 0.003, hs - 0.012))
        if k < n - 1:
            zj = z0 + (k + 1) * hs
            for x in (-(ww / 2 - 0.03), -ww / 4, 0.0, ww / 4, ww / 2 - 0.03):
                B.box("dy_steel_galv", (x, 0.062, zj), (0.07, 0.006, 0.10), bevel=0.002)
                B.cyl("dy_steel_galv", (x - 0.035, 0.068, zj), (x + 0.035, 0.068, zj), 0.007, n=8)
        for sg in (-1, 1):
            B.cyl("dy_steel_galv", (sg * (ww / 2 - 0.04), 0.074, z0 + (k + 0.92) * hs),
                  (sg * (ww / 2 + 0.02), 0.074, z0 + (k + 0.92) * hs), 0.006, n=6)
    zc = z0 + 2.5 * hs
    for x in lites:
        B.box("dy_black", (x, 0.0345, zc), (0.62, 0.060, 0.26), bevel=0.006)
        B.rect("dy_daylight", (x, 0.0655, zc), -_X, _Z, (0.54, 0.18))
        B.rect("dy_glass", (x, 0.0035, zc), _X, _Z, (0.54, 0.18))
    B.box("dy_black", (0.0, 0.0345, z0 + 0.012), (ww, 0.05, 0.03), bevel=0.006)
    B.box("dy_steel_galv", (0.0, 0.066, z0 + 0.10), (0.20, 0.028, 0.012), bevel=0.002)
    B.box("dy_steel_galv", (0.0, 0.066, z0 + 0.75), (0.16, 0.025, 0.02), bevel=0.004)
    B.box("dy_steel_galv", (0.0, 0.0705, z0 + (n - 0.25) * hs), (ww - 0.12, 0.019, 0.05), bevel=0.003)
    B.box("dy_steel_galv", (ww / 2 - 0.16, 0.066, z0 + 1.5 * hs), (0.20, 0.014, 0.05), bevel=0.003)
    if number is not None:
        text = f"number_{number}"
        img = _sign(S, f"dy_{text}.png", [(str(number), 1.0)], (256, 256), (245, 245, 240), (22, 58, 122),
                    border=(245, 245, 240))
        B.rect(_image_mat(S, f"dy_sign_{text}", img), (0.0, 0.0605, z0 + 0.34), -_X, _Z, (0.40, 0.40), uv01=True)


def _door_track(B, h, sg):
    """One vertical-lift track of a sectional door with its wall angle and brackets, up the jamb to twice the door
    height (origin: on the floor in the wall's inner face on the track's centreline, +y into the room; sg: the x side
    away from the opening)."""
    zt = _door_track_top(h)
    B.box("dy_steel_galv", (0.0, 0.050, zt / 2), (0.055, 0.060, zt), bevel=0.003)
    B.box("dy_steel_galv", (sg * 0.045, 0.002, zt / 2), (0.09, 0.004, zt))
    for z in np.arange(0.4, zt, 0.75):
        B.box("dy_steel_galv", (sg * 0.04, 0.03, z), (0.05, 0.06, 0.05), bevel=0.002)
        B.inst("bolt", (sg * 0.072, 0.004, z), _RZ[1])


def _door_shaft(B, w, h):
    """Spring shaft of a vertical-lift door above its tracks: cable drums, torsion springs with their winding cones,
    bearing plates, a jackshaft operator at one end (origin as _sectional_door)."""
    zt = _door_track_top(h)
    xt = w / 2 + DOOR_OVERLAP + 0.04
    zs = zt + 0.10
    for sg in (-1, 1):
        B.cyl("dy_alu", (sg * (xt - 0.02), 0.11, zs), (sg * (xt + 0.05), 0.11, zs), 0.075, n=20)
        B.box("dy_steel_galv", (sg * (xt + 0.07), 0.07, zs), (0.008, 0.14, 0.22))
        B.cyl("dy_spring", (sg * 0.08, 0.11, zs), (sg * 1.0, 0.11, zs), 0.042, n=16)
        B.cyl("dy_steel_galv", (sg * 1.0, 0.11, zs), (sg * 1.06, 0.11, zs), 0.048, n=12)
    B.cyl("dy_steel_galv", (-(xt + 0.07), 0.11, zs), (xt + 0.07, 0.11, zs), 0.0127, n=10)
    B.box("dy_steel_galv", (0.0, 0.06, zs), (0.008, 0.12, 0.20))
    B.box("dy_box_grey", (-(xt + 0.20), 0.10, zs), (0.20, 0.16, 0.24), bevel=0.01)
    B.cyl("dy_spring", (-(xt + 0.20), 0.10, zs - 0.12), (-(xt + 0.20), 0.10, zs - 0.30), 0.06, n=16)


def _track_guard(B):
    """Steel guard around the foot of a door track (track centre on the origin's x, wall at y = 0)."""
    B.box("dy_yellow", (0.0, 0.07, 0.5), (0.15, 0.14, 1.0), bevel=0.006)
    for z in (0.22, 0.52, 0.82):
        B.box("dy_black", (0.0, 0.07, z), (0.152, 0.142, 0.08))


def _control_box(B):
    """Leveler push-button station (origin: its centre on the wall, +y into the room)."""
    B.box("dy_box_grey", (0.0, 0.06, 0.0), (0.22, 0.12, 0.32), bevel=0.008)
    B.cyl("dy_black", (0.0, 0.12, 0.07), (0.0, 0.136, 0.07), 0.028, n=16)
    B.cyl("dy_glad_red", (0.0, 0.12, -0.06), (0.0, 0.146, -0.06), 0.024, n=16)
    B.box("dy_plate_white", (0.0, 0.1206, 0.0), (0.12, 0.001, 0.03))


def _signal_box(B, lit):
    """Restraint signal: red over green lenses in a visored housing; lit = 'red' or 'green' (origin on the wall)."""
    B.box("dy_seal_yellow", (0.0, 0.06, 0.0), (0.17, 0.12, 0.36), bevel=0.01)
    for z, col in ((0.085, "red"), (-0.085, "green")):
        B.cyl(f"dy_led_{col}" if lit == col else f"dy_led_{col}_off", (0.0, 0.12, z), (0.0, 0.128, z), 0.05, n=20)
        B.box("dy_black", (0.0, 0.155, z + 0.058), (0.13, 0.07, 0.006))


def dock_light_source(x_door):
    """Position and aim of a door's dock-light source as render._task_lights places it (DOCK_LIGHT; an 'offset' entry
    (dx from the jamb, dy from the dock face, z) is honoured if render grows one)."""
    from .. import render as RD
    from ..facility import DOOR_W, DOCK_FACE_Y
    off = RD.DOCK_LIGHT.get("offset", (0.35, -0.45, RD.DOCK_LIGHT["mount_h"]))
    pos = np.array([x_door + DOOR_W / 2 + off[0], DOCK_FACE_Y + off[1], off[2]])
    aim = np.array([x_door, DOCK_FACE_Y + 9.0, 1.5]) - pos
    return pos, aim / np.linalg.norm(aim)


def _dock_light(B, src, aim, y_wall):
    """Swing-arm dock light (world coordinates): the lamp head's rim sits just behind the light source so that the
    source shines out of it; two arm links fold from a wall bracket to the head."""
    back = src - 0.17 * aim
    B.cyl("dy_seal_yellow", back, src - 0.012 * aim, 0.05, r2=0.098, n=24, caps=False)
    B.cyl("dy_black", back - 0.03 * aim, back, 0.05, n=18)
    B.cyl("dy_black", src - 0.02 * aim, src - 0.008 * aim, 0.099, r2=0.102, n=24, caps=False)
    wrist = back - 0.03 * aim
    sh = np.array([src[0] + 0.42, y_wall - 0.04, src[2] + 0.03])
    l1 = l2 = 0.52
    d = wrist - sh
    dist = min(float(np.linalg.norm(d)), l1 + l2 - 0.01)
    dn = d / max(np.linalg.norm(d), 1e-9)
    a = math.acos(max(-1.0, min(1.0, dist / (2 * l1))))
    side = np.cross(dn, _Z)
    side /= max(np.linalg.norm(side), 1e-9)
    if side[1] > 0:                      # the elbow swings into the room
        side = -side
    elbow = sh + l1 * (math.cos(a) * dn + math.sin(a) * side)
    B.box("dy_steel_galv", (sh[0], y_wall - 0.004, sh[2]), (0.12, 0.008, 0.22), bevel=0.002)
    for dz in (-0.08, 0.08):
        B.inst("bolt", (sh[0], y_wall - 0.008, sh[2] + dz), _RZ[-1])
    B.cyl("dy_black", sh - 0.07 * _Z, sh + 0.07 * _Z, 0.022, n=12)
    B.cyl("dy_black", elbow - 0.05 * _Z, elbow + 0.05 * _Z, 0.02, n=12)
    B.beam("dy_seal_yellow", sh, elbow, (0.035, 0.035), bevel=0.004)
    B.beam("dy_seal_yellow", elbow, wrist, (0.035, 0.035), bevel=0.004)
    B.tube("dy_cable", [sh + (0.0, 0.0, -0.07), sh + (0.0, -0.02, -0.16), (sh + elbow) / 2 + (0.0, 0.0, -0.10), elbow + (0.0, 0.0, -0.06),
                        (elbow + wrist) / 2 + (0.0, 0.0, -0.08), wrist + (0.0, 0.0, -0.03)], 0.006, n=6)


def _leveler_bodies(S, world, shapes):
    """Each hydraulic dock leveler on its two bodies: the hinged deck (tread plate, toe-guard stripes on the faces that
    show when it rises, stiffeners beneath) and the hinged lip."""
    from scipy.spatial.transform import Rotation
    for name in world.scene.levelers:
        c, q, half, b = shapes[name]["deck"]
        B0 = _Build()
        B = B0.at(c, R=Rotation.from_quat(q).as_matrix())
        Wd, Ld, t = 2 * half
        B.box("dy_tread", (0.0, 0.0, 0.0), (Wd, Ld, t), bevel=0.003)
        for sg in (1, -1):
            B.rect("dy_hazard", (sg * (Wd / 2 + 0.0006), 0.0, 0.0), sg * _Y, _Z, (Ld - 0.01, t - 0.008))
            B.rect("dy_paint_yellow", (sg * (Wd / 2 - 0.05), 0.0, t / 2 + 0.0008), _X, _Y, (0.06, Ld - 0.06))
        B.rect("dy_hazard", (0.0, Ld / 2 + 0.0006, 0.0), -_X, _Z, (Wd - 0.01, t - 0.008))
        for x in np.linspace(-Wd / 2 + 0.15, Wd / 2 - 0.15, 6):
            B.box("dy_steel_dark", (x, 0.0, -t / 2 - 0.08), (0.05, Ld - 0.12, 0.16))
        B.box("dy_steel_dark", (0.0, Ld / 2 - 0.035, -t / 2 - 0.08), (Wd - 0.02, 0.06, 0.16))
        B0.flush(S, f"/Lab/body_{b}/deck")
        c, q, half, b = shapes[name]["lip"]
        B0 = _Build()
        B = B0.at(c, R=Rotation.from_quat(q).as_matrix())
        B.box("dy_tread", (0.0, 0.0, 0.0), 2 * half, bevel=0.004)
        seg = (2 * half[0] - 0.04) / 9
        for k in range(9):
            x0 = -half[0] + 0.02 + k * seg
            B.cyl("dy_steel_dark", (x0 + 0.015, -half[1], -half[2] - 0.016), (x0 + seg - 0.015, -half[1], -half[2] - 0.016), 0.022, n=10)
        B0.flush(S, f"/Lab/body_{b}/lip")


def _ramp_lip_bodies(S, world, shapes):
    """The yard ramp's hinged lip plates, each on its body: diamond tread plate, a bevelled leading edge, the pin's
    knuckles at the level-off edge and a hand slot."""
    from scipy.spatial.transform import Rotation
    for name in world.scene.ramp_lips:
        for tag, (c, q, half, b) in sorted(shapes[name].items()):
            B0 = _Build()
            B = B0.at(c, R=Rotation.from_quat(q).as_matrix())
            w_, l_, t = 2 * half
            B.box("dy_tread_galv", (0.0, 0.0, 0.0), (w_, l_ - 0.03, t), bevel=0.003)
            B.box("dy_tread_galv", (0.0, l_ / 2 - 0.015, -t / 4), (w_, 0.03, t / 2), bevel=0.002)          # bevelled toe
            for k in range(3):
                x0 = -w_ / 2 + 0.03 + k * (w_ - 0.06) / 3
                B.cyl("dy_steel_dark", (x0 + 0.01, -l_ / 2 - 0.006, 0.0), (x0 + (w_ - 0.06) / 3 - 0.01, -l_ / 2 - 0.006, 0.0), 0.013, n=10)
            B.box("dy_steel_dark", (0.0, -l_ / 2 + 0.09, t / 2 + 0.0005), (0.12, 0.035, 0.001))                # hand slot
            B0.flush(S, f"/Lab/body_{b}/{tag}")


def _dock_static(S, world, B):
    """The dock preset's wall, dock face, pits, bumpers and seal pads from the layout boxes, the pit curb angles and
    the apron."""
    from ..facility import DOOR_H
    sc = world.scene
    L = sc.layout
    x_lo, x_hi = L.extent[0], L.extent[1]
    lower, upper, trim = (m if m in S.mats else alt for m, alt in (("v_wall_lower", "dy_wall_in"), ("v_wall_upper", "dy_wall_in"),
                                                                    ("v_wall_trim", "dy_frame")))
    band = 2.40 if lower != "dy_wall_in" else 0.0            # visuals.interior.WALL_BAND
    for bx in L.boxes:
        n = bx["name"]
        if owner(world, "box", n) != "dockyard":
            continue
        (cx, cy, cz), (hx, hy, hz) = bx["centre"], bx["half"]
        if n.startswith(("dockwall", "dockheader")):
            pier = n.startswith("dockwall")
            z0, z1 = cz - hz, cz + hz
            # room face like the building's other walls (visuals.interior): block band, cap line, painted panels
            yi, zf = cy - hy, max(z0, 0.0)
            if zf < band:
                B.rect(lower, (cx, yi, (zf + band) / 2), _X, _Z, (2 * hx, band - zf))
                B.rect(trim, (cx, yi - 0.003, band), _X, _Z, (2 * hx, 0.06))
                B.rect(trim, (cx, yi - 0.002, 0.006), _X, _Z, (2 * hx, 0.012))
            B.rect(upper, (cx, yi, (max(z0, band) + z1) / 2), _X, _Z, (2 * hx, z1 - max(z0, band)))
            if z0 < 0.0:
                B.rect("dy_concrete", (cx, yi, z0 / 2), _X, _Z, (2 * hx, -z0))
            for xj in np.arange(math.floor((cx - hx) / 6.0) * 6.0 + 3.0, cx + hx, 6.0):       # caulked panel joints
                if cx - hx + 0.3 < xj < cx + hx - 0.3:
                    B.rect(trim, (xj, yi - 0.0015, (max(z0, band) + 0.03 + z1) / 2), _X, _Z, (0.018, z1 - max(z0, band) - 0.03))
            zb = 0.30 if pier else z0
            if pier:
                B.rect("dy_concrete", (cx, cy + hy, (z0 + zb) / 2), -_X, _Z, (2 * hx, zb - z0))
                for sg in (-1, 1):
                    xf = cx + sg * hx
                    if x_lo + 0.01 < xf < x_hi - 0.01:                     # a door jamb, not the building's corner
                        B.rect("dy_frame", (xf, cy, DOOR_H / 2), sg * _Y, _Z, (2 * hy, DOOR_H))
                        B.rect("dy_concrete", (xf, cy, (z0 + 0.0) / 2), sg * _Y, _Z, (2 * hy, -z0))
            else:
                B.rect("dy_frame", (cx, cy, z0), _Y, _X, (2 * hy, 2 * hx))
            _cladding(B, "dy_facade", cx - hx, cx + hx, zb, z1 - 0.9, cy + hy, +1)
            _cladding(B, "dy_facade_band", cx - hx, cx + hx, z1 - 0.9, z1, cy + hy, +1)
        elif n.startswith(("dockface", "pit")):
            B.box("dy_concrete", (cx, cy, cz), (2 * hx, 2 * hy, 2 * hz))
        elif n.startswith("bumper"):
            _bumper(B.at((cx, cy, cz)), (hx, hy, hz))
        elif n.startswith("seal"):
            door_x = min(sc.zones["doors"].values(), key=lambda d: abs(d["x"] - cx))["x"]
            _seal_side(B.at((cx, cy - hy, cz - hz)), 2 * hx, 2 * hy, 2 * hz, 1 if door_x > cx else -1)
        else:
            B.box("dy_concrete", (cx, cy, cz), (2 * hx, 2 * hy, 2 * hz))
    for name, lv in sc.levelers.items():                                  # pit curb angles, flush in the floor
        Wd, y0, y1 = lv["width"], lv["dock_y"] - lv["length"], lv["dock_y"]
        for sg in (-1, 1):
            B.box("dy_steel_dark", (lv["x"] + sg * (Wd / 2 + 0.045), (y0 - 0.08 + y1) / 2, 0.001), (0.06, y1 - y0 + 0.08, 0.002))
        B.box("dy_steel_dark", (lv["x"], y0 - 0.05, 0.001), (Wd + 0.15, 0.06, 0.002))
    for s in L.surfaces:
        if s["name"] == "apron":
            _apron_slab(B, s)


def _apron_slab(B, s):
    """A concrete apron: slab with sawn joints on a 4.5 m grid."""
    (cx, cy), (hx, hy), z = s["centre"], s["half"], s.get("z_top", 0.0)
    B.rect("dy_apron", (cx, cy, z), _X, _Y, (2 * hx, 2 * hy))
    for x in np.arange(cx - hx + 4.5, cx + hx - 0.1, 4.5):
        B.rect("dy_seal_tar", (x, cy, z + 0.0012), _X, _Y, (0.012, 2 * hy))
    for y in np.arange(cy - hy + 5.0, cy + hy - 0.1, 5.0):
        B.rect("dy_seal_tar", (cx, y, z + 0.0012), _X, _Y, (2 * hx, 0.012))


# =====================================================================================================================
# yard: ground and markings, the building's wall, the mobile ramp, parked trailers, fence and poles
# =====================================================================================================================
FACADE_RECESS = 0.10          # doors and windows stand this far behind the wall's face (inside its collision box)


def _facade_plan(L):
    """The yard building's wall and its openings, from the layout: dock-high doors west of the ramp, a personnel door,
    office windows, and a drive-in door at the head of the yard lane (where the truck comes out)."""
    from ..facility import DOOR_H, DOOR_W
    wall = next(b for b in L.boxes if b["name"] == "building_wall")
    x0, x1 = wall["centre"][0] - wall["half"][0], wall["centre"][0] + wall["half"][0]
    lane_x = L.zones["yard_lane"]["points"][0][0]
    ramp_x = L.zones["ramp"]["x"]
    sill = float(P.DOCK_HEIGHT)
    ops = []
    k = 0
    while x0 + 5.0 + k * 12 * FT + DOOR_W / 2 < ramp_x - 5.5 and k < 8:
        ops.append(dict(kind="dock", n=11 + k, x=x0 + 5.0 + k * 12 * FT, w=DOOR_W, h=DOOR_H, z=sill))
        k += 1
    ops.append(dict(kind="personnel", x=ramp_x - 2.5, w=0.95, h=2.15, z=0.0))
    xw = ramp_x + 2.5
    while xw + 1.0 < lane_x - 7.0:
        ops.append(dict(kind="window", x=xw, w=1.8, h=1.2, z=1.45))
        xw += 3.0
    ops.append(dict(kind="drive", x=lane_x, w=12 * FT, h=14 * FT, z=0.0))
    return dict(x0=x0, x1=x1, y=wall["centre"][1] - wall["half"][1], zt=wall["centre"][2] + wall["half"][2], base=sill,
                ops=sorted(ops, key=lambda o: o["x"]))


def _facade(S, B, L):
    """The building's wall as seen from the yard: cast-concrete base to dock height, ribbed insulated metal panels
    above with a coloured band under the coping, door and window openings recessed into the wall."""
    F = _facade_plan(L)
    x0, x1, y, zt, base, ops = F["x0"], F["x1"], F["y"], F["zt"], F["base"], F["ops"]
    band = zt - 0.9
    zs = sorted({0.0, base, band, zt} | {min(max(v, 0.0), zt) for o in ops for v in (o["z"], o["z"] + o["h"])})
    for za, zb in zip(zs[:-1], zs[1:]):
        if zb - za < 1e-6:
            continue
        cuts = sorted((o["x"] - o["w"] / 2, o["x"] + o["w"] / 2) for o in ops if o["z"] < zb - 1e-6 and o["z"] + o["h"] > za + 1e-6)
        spans, a = [], x0
        for c0, c1 in cuts:
            if c0 > a:
                spans.append((a, c0))
            a = max(a, c1)
        if a < x1:
            spans.append((a, x1))
        for a, b in spans:
            if zb <= base + 1e-6:
                for k in range(int(math.floor(a / 3.0)), int(math.ceil(b / 3.0))):          # cast in 3 m lifts
                    pa, pb = max(a, 3.0 * k), min(b, 3.0 * k + 3.0)
                    if pb - pa > 1e-6:
                        B.rect("dy_concrete", ((pa + pb) / 2, y, (za + zb) / 2), _X, _Z, (pb - pa, zb - za),
                               uvo=((k * 1.37) % 3.0, (k * 0.83) % 3.0))
                        B.rect("dy_seal_tar", (pa + 0.004, y - 0.0005, (za + zb) / 2), _X, _Z, (0.008, zb - za))
            else:
                _cladding(B, "dy_facade_band" if za >= band - 1e-6 else "dy_facade", a, b, za, zb, y, -1)
    B.box("dy_trim", ((x0 + x1) / 2, y + 0.16, zt + 0.03), (x1 - x0 + 0.04, 0.36, 0.06), bevel=0.01)       # coping
    B.rect("dy_trim", ((x0 + x1) / 2, y - 0.0005, base + 0.02), _X, _Z, (x1 - x0, 0.04))                  # base flashing
    r = FACADE_RECESS
    for o in ops:
        x, w, h, z = o["x"], o["w"], o["h"], o["z"]
        for sg in (-1, 1):                                                                              # reveals
            B.rect("dy_trim", (x + sg * w / 2, y + r / 2, z + h / 2), -sg * _Y, _Z, (r, h))
        B.rect("dy_trim", (x, y + r / 2, z + h), _Y, _X, (r, w))
        if z > 0.01:
            B.rect("dy_trim", (x, y + r / 2, z), _X, _Y, (w, r))
        if o["kind"] in ("dock", "drive"):
            _sectional_door(B.at((x, y + r - 0.0105, z)), w - 0.006, h, False, ext_only=True)
        elif o["kind"] == "window":
            B.rect("dy_glass", (x, y + r, z + h / 2), _X, _Z, (w, h))
            for xx in (x - w / 2 + 0.025, x, x + w / 2 - 0.025):
                B.box("dy_trim", (xx, y + r - 0.02, z + h / 2), (0.05, 0.04, h))
            for zz in (z + 0.025, z + h - 0.025):
                B.box("dy_trim", (x, y + r - 0.02, zz), (w, 0.04, 0.05))
        else:                                                                                           # personnel door
            B.rect("dy_facade_band", (x, y + r, z + h / 2), _X, _Z, (w, h))
            B.box("dy_trim", (x, y + r - 0.015, z + h - 0.03), (w, 0.03, 0.06))
            for sg in (-1, 1):
                B.box("dy_trim", (x + sg * (w / 2 - 0.03), y + r - 0.015, z + h / 2), (0.06, 0.03, h))
            B.box("dy_glass", (x + 0.15, y + r - 0.004, z + 1.55), (0.20, 0.008, 0.55))
            B.box("dy_steel_galv", (x - w / 2 + 0.14, y + r - 0.03, z + 1.02), (0.13, 0.05, 0.03), bevel=0.006)
            B.box("dy_steel_galv", (x, y + r - 0.004, z + 0.14), (w - 0.14, 0.004, 0.22))                 # kick plate


def _ramp_frame(s):
    """Origin on the deck's top at the surface centre and the rotation whose columns are: up the slope, across the
    deck (+x for a ramp climbing toward -y), the deck's normal (the frame scene.floor_rect builds the collider in)."""
    ax = np.array([*s.get("axis", (1.0, 0.0)), 0.0])
    ax /= np.linalg.norm(ax)
    ang = math.atan(s.get("grade", 0.0))
    u = ax * math.cos(ang) + _Z * math.sin(ang)
    side = np.cross(_Z, ax)
    n = np.cross(u, side)
    hl = s["half"][0] / math.cos(ang) if s.get("grade") else s["half"][0]
    return np.array([s["centre"][0], s["centre"][1], s.get("z_top", 0.0)]), np.stack([u, side, n], 1), hl, s["half"][1]


def _grating(B, hl, hw, u0=None):
    """Serrated bar grating in the deck frame (x up the slope, y across, z the normal; top at z = 0): bearing bars
    across the deck at 30 mm, cross rods at 100 mm, on channel cross-members."""
    u0 = -hl if u0 is None else u0
    for x in np.arange(u0 + 0.015, hl, 0.030):
        B.box("dy_steel_galv", (x, 0.0, -0.0125), (0.005, 2 * hw, 0.025))
    for y in np.arange(-hw + 0.05, hw, 0.10):
        B.box("dy_steel_galv", ((u0 + hl) / 2, y, -0.006), (hl - u0, 0.006, 0.006))
    for x in np.arange(u0 + 0.20, hl, 0.61):
        B.box("dy_yellow", (x, 0.0, -0.085), (0.06, 2 * hw, 0.12))


def _ramp(S, world, B):
    """The mobile yard ramp on its colliders: grating deck on the incline and the level-off, the approach plate at the
    foot, the curb channels along both edges (the hinged lip plates are bodies: _ramp_lip_bodies)."""
    L = world.scene.layout
    for s in L.surfaces:
        if not s["name"].startswith("ramp_"):
            continue
        o, R, hl, hw = _ramp_frame(s)
        Bl = B.at(o, R=R)
        if s["name"] == "ramp_incline":
            Bl.box("dy_tread_galv", (-hl + 0.26, 0.0, -0.008), (0.52, 2 * hw, 0.016), bevel=0.003)   # approach plate
            _grating(Bl, hl, hw, u0=-hl + 0.52)
        else:
            _grating(Bl, hl, hw)
    for bx in L.boxes:
        if bx["name"].startswith("curb_"):
            R = _rot(roll=bx.get("pitch") or 0.0)
            B.box("dy_yellow", bx["centre"], [2 * v for v in bx["half"]], bevel=0.008, R=R)
        elif bx["name"].startswith("ramp_bumper"):                  # faces the trailer at -y
            _bumper(B.at(bx["centre"], yaw=math.pi), tuple(bx["half"]))
            c, h = np.asarray(bx["centre"]), np.asarray(bx["half"])
            B.box("dy_steel_dark", (c[0], c[1] + h[1] + 0.004, c[2]), (2 * h[0] + 0.10, 0.008, 2 * h[2] + 0.02))    # mounting plate


def _parked_dims(bx):
    """A parked van inside its layout box: dimensions, the pose of its frame, the floor's height above the ground."""
    (cx, cy, cz), (hx, hy, hz) = bx["centre"], bx["half"]
    sub, rt, nt, wall, df, proud = 0.22, 0.06, 0.025, 0.025, 0.10, 0.075
    Wo = 2 * hx
    H = 2 * hz - sub - rt
    D = dict(L=2 * hy - proud - df - nt, W=Wo - 2 * wall, sub=sub, H=H, wall=wall, Wo=Wo, nose_t=nt, roof_t=rt, df=df,
             post_y0=Wo / 2 - wall - 0.01, post_y1=Wo / 2, hdr_h=0.16, hdr_z0=H - 0.10, sill_h=0.20,
             icc_c=np.array([0.25, 0.0, -0.55]), icc_h=np.array([0.04, 1.1, 0.05]))
    origin = (cx, cy - hy + proud + df, cz - hz + sub)
    return D, origin, math.pi / 2, cz - hz + sub


def _parked_trailers(L):
    """[(layout box, its landing-gear box or None, index)] of the parked vans."""
    out = []
    for bx in L.boxes:
        n = bx["name"]
        if n.startswith("parked_trailer") and n[-1].isdigit():
            gear = next((g for g in L.boxes if g["name"] == n + "_gear"), None)
            out.append((bx, gear, int(n[len("parked_trailer"):])))
    return out


def _line(B, a, b, width, mat, z=0.0034):
    a, b = np.array([a[0], a[1], z]), np.array([b[0], b[1], z])
    d = b - a
    Lg = float(np.linalg.norm(d))
    if Lg > 1e-6:
        u = d / Lg
        B.rect(mat, (a + b) / 2, u, np.cross(_Z, u), (Lg, width))


def _strip(B, mat, pts, width, z):
    """A flat ribbon of `width` along a ground polyline."""
    Pn = np.array([(x, y, z) for x, y in pts], float)
    t = np.gradient(Pn, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.cross(_Z, t)
    B.sheet(mat, Pn - nrm * width / 2, Pn + nrm * width / 2)


def _wet_material(S):
    if "dy_wet" not in S.mats:
        rng = np.random.default_rng(71)
        n = 256
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0) * 2 - 1
        al = np.clip((1.0 - np.sqrt(xx ** 2 + yy ** 2) - 0.35 * (_noise(rng, (n, n), 4, 3) - 0.4)) * 6.0, 0.0, 1.0)
        _pbr(S, "dy_wet", colour=(0.012, 0.012, 0.014), rough=0.04, opacity_tex=_save(S, "dy_wet_alpha.png", al * 0.86))
    return "dy_wet"


def _yard_static(S, world, B):
    """Everything in the yard that stands on a collider or is the ground itself."""
    sc = world.scene
    L = sc.layout
    rng = np.random.default_rng(600 + sc.c.seed)
    asp = next(s for s in L.surfaces if s["name"] == "asphalt")
    apr = next(s for s in L.surfaces if s["name"] == "apron")
    (ax, ay), (ahx, ahy) = asp["centre"], asp["half"]
    x0, x1, ys = ax - ahx, ax + ahx, ay - ahy
    ya = apr["centre"][1] - apr["half"][1]
    yn = apr["centre"][1] + apr["half"][1]
    # ---- the ground: far field, grass verge, gravel shoulder, asphalt, the concrete apron along the building
    B.rect("dy_gravel", (ax, (ys - 2.5 + yn) / 2, -0.015), _X, _Y, (2 * ahx + 5.0, yn - ys + 2.5))
    # beyond the fence: the building's body behind its wall, then rolling ground, trees, a road and neighbours
    wall = next(b for b in L.boxes if b["name"] == "building_wall")
    yb, depth, hb = wall["centre"][1] + wall["half"][1], 46.0, wall["centre"][2] + wall["half"][2]
    _shed(B.at((ax, yb + depth / 2, 0.0)), 2 * wall["half"][0], depth, hb, "dy_facade", "dy_facade_band", doors=0, sides=(1, 2, 3))
    _surroundings(S, B, (x0, x1, ys, yb + depth), sc.c.seed, road="S", trailers=True,
                  keep_out=[(x0 - 3.0, x1 + 3.0, ys - 3.0, yb + depth + 3.0)])
    B.rect("dy_asphalt", (ax, (ys + ya) / 2, 0.0), _X, _Y, (2 * ahx, ya - ys))
    B.rect("dy_asphalt_dark", (ax, (ys + ya) / 2, 0.0006), _X, _Y, (2 * ahx, ya - ys))
    B.rect("dy_asphalt_light", (ax, (ys + ya) / 2, 0.0012), _X, _Y, (2 * ahx, ya - ys))
    # cracked areas: 3 m blocks of crack pattern, each turned or mirrored at random, where a slow noise says so
    nx, ny = int((x1 - x0) // 3.0), int((ya - ys) // 3.0)
    where = _noise(rng, (ny, nx), 3, 2)
    quad = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    for j in range(ny):
        for i in range(nx):
            if where[j, i] < 0.50:
                continue
            cx, cy = x0 + 3.0 * i + 1.5, ys + 3.0 * j + 1.5
            uv = np.roll(quad, int(rng.integers(4)), axis=0)
            if rng.random() < 0.5:
                uv = uv[::-1]
                B.poly("dy_cracks", [(cx - 1.5, cy + 1.5, 0.0018), (cx - 1.5, cy - 1.5, 0.0018), (cx + 1.5, cy - 1.5, 0.0018),
                                     (cx + 1.5, cy + 1.5, 0.0018)][::-1], uv=uv)
            else:
                B.poly("dy_cracks", [(cx - 1.5, cy - 1.5, 0.0018), (cx + 1.5, cy - 1.5, 0.0018), (cx + 1.5, cy + 1.5, 0.0018),
                                     (cx - 1.5, cy + 1.5, 0.0018)], uv=uv)
    for sx in (x0, x1):
        B.rect("dy_asphalt", (sx, (ys + yn) / 2, -0.0075), (-_Y if sx == x0 else _Y), _Z, (yn - ys, 0.015))
    B.rect("dy_asphalt", (ax, ys, -0.0075), _X, _Z, (2 * ahx, 0.015))
    _apron_slab(B, apr)
    _strip(B, "dy_seal_tar", [(x, ya + 0.02 * math.sin(0.7 * x)) for x in np.arange(x0, x1 + 0.1, 0.5)], 0.035, 0.0022)
    # patches of newer asphalt with sealed edges, and sealed cracks wandering across the old surface
    for k in range(7):
        w, d = rng.uniform(1.6, 4.5), rng.uniform(1.2, 3.0)
        Bp = B.at((rng.uniform(x0 + 3, x1 - 3), rng.uniform(ys + 3, ya - 3), 0.0), yaw=rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.04))
        Bp.rect("dy_asphalt_new", (0.0, 0.0, 0.0026), _X, _Y, (w, d))
        for cx, cy, sx, sy in ((0, d / 2, w, 0.04), (0, -d / 2, w, 0.04), (w / 2, 0, 0.04, d), (-w / 2, 0, 0.04, d)):
            Bp.rect("dy_seal_tar", (cx, cy, 0.0030), _X, _Y, (sx + 0.02, sy + 0.02))
    for k in range(18):
        pt = np.array([rng.uniform(x0 + 2, x1 - 2), rng.uniform(ys + 2, ya - 2)])
        hd = rng.uniform(0, 2 * math.pi)
        pts = [pt.copy()]
        for _ in range(int(rng.uniform(14, 45))):
            hd += rng.normal(0, 0.28)
            pt = pt + 0.55 * np.array([math.cos(hd), math.sin(hd)])
            if not (x0 + 0.5 < pt[0] < x1 - 0.5 and ys + 0.5 < pt[1] < ya - 0.3):
                break
            pts.append(pt.copy())
        if len(pts) > 3:
            _strip(B, "dy_seal_tar", pts, rng.uniform(0.025, 0.045), 0.0022)
    # ---- painted markings (worn): the layout's lines, stalls for the parked trailers and their landing-gear pad
    for ln in L.lines:
        _line(B, ln["a"], ln["b"], ln["width"], "dy_paint_yellow" if ln["colour"] == "yellow" else "dy_paint_white")
    parked = _parked_trailers(L)
    if parked:
        xs_ = sorted(bx["centre"][0] for bx, _, _ in parked)
        pitch = (xs_[1] - xs_[0]) if len(xs_) > 1 else 4.0
        ya_, yb_ = min(bx["centre"][1] - bx["half"][1] for bx, _, _ in parked) - 0.5, max(bx["centre"][1] + bx["half"][1] for bx, _, _ in parked) + 0.5
        for x in [xs_[0] - pitch / 2] + [x + pitch / 2 for x in xs_]:
            _line(B, (x, ya_), (x, yb_), 0.10, "dy_paint_white")
        gears = [g for _, g, _ in parked if g is not None]
        if gears:
            gy = float(np.mean([g["centre"][1] for g in gears]))
            B.rect("dy_apron", ((xs_[0] + xs_[-1]) / 2, gy, 0.0028), _X, _Y, (xs_[-1] - xs_[0] + pitch - 0.3, 1.5))
            for dy in (-0.75, 0.75):
                B.rect("dy_seal_tar", ((xs_[0] + xs_[-1]) / 2, gy + dy, 0.0031), _X, _Y, (xs_[-1] - xs_[0] + pitch - 0.3, 0.03))
    # ---- rubber and oil: tyre marks where trucks brake and turn, stains where they stand
    rz = L.zones["ramp"]
    lane = L.zones["yard_lane"]["points"]
    marks = [(rz["x"] - 0.45, rz["y_foot"] + 2.6, -math.pi / 2, 4.6), (rz["x"] + 0.5, rz["y_foot"] + 3.4, -math.pi / 2, 3.8)]
    for (a0, a1), (b0, b1) in zip(lane[:-1], lane[1:]):
        hd = math.atan2(b1 - a1, b0 - a0)
        marks.append((b0 - 1.2 * math.cos(hd) + 0.5 * math.sin(hd), b1 - 1.2 * math.sin(hd) - 0.5 * math.cos(hd), hd + 0.12, 4.2))
    for o in _facade_plan(L)["ops"]:
        if o["kind"] == "dock":
            for dx in (-0.915, 0.915):
                marks.append((o["x"] + dx + rng.normal(0, 0.08), yn - rng.uniform(3.0, 6.0), math.pi / 2 + rng.normal(0, 0.02), rng.uniform(4.0, 7.0)))
    for x, y, hd, lg in marks:
        B.at((x, y, 0.0038), yaw=hd).rect("dy_tyremark", (0.0, 0.0, 0.0), _X, _Y, (lg, 0.60), uv01=True)
    spots = [(bx["centre"][0] + rng.normal(0, 0.4), bx["centre"][1] + bx["half"][1] - rng.uniform(1.0, 3.0)) for bx, _, _ in parked]
    spots += [(rng.uniform(x0 + 2, x1 - 2), rng.uniform(ya + 0.5, yn - 1.0)) for _ in range(9)]
    spots += [(rng.uniform(x0 + 2, x1 - 2), rng.uniform(ys + 2, ya - 1)) for _ in range(8)]
    for k, (x, y) in enumerate(spots):
        sz = rng.uniform(0.4, 1.1)
        B.at((x, y, 0.0042), yaw=rng.uniform(0, 6.28)).rect(f"dy_stain{k % 3}", (0.0, 0.0, 0.0), _X, _Y, (sz, sz * rng.uniform(0.6, 1.4)), uv01=True)
    for name, wp_ in ((sc.task_start or {}).get("floor", {}) or {}).items():                 # wet patch of the start state
        c, h = wp_["centre"], wp_["half"]
        B.rect(_wet_material(S), (c[0], c[1], 0.0046), _X, _Y, (2.3 * h[0], 2.3 * h[1]), uv01=True)
    # ---- the building, the ramp, the parked vans
    _facade(S, B, L)
    _ramp(S, world, B)
    for bx, gear, i in parked:
        D, origin, yaw, h = _parked_dims(bx)
        _trailer_body(S, B.at(origin, yaw=yaw), D, "closed", i + 1, 2210 + 97 * i)


# =====================================================================================================================
# surroundings beyond the pavement (yard and test pad): rolling ground, vegetation, a road, neighbouring buildings
# =====================================================================================================================
def _value_noise(seed):
    """Smooth value noise n(x, y, cell) in 0..1 over the plane (a 64 x 64 lattice, repeating every 64 cells)."""
    T = np.random.default_rng(seed).random((64, 64))

    def n(x, y, cell):
        gx, gy = np.asarray(x, float) / cell, np.asarray(y, float) / cell
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - ix, gy - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a, b = T[iy % 64, ix % 64], T[iy % 64, (ix + 1) % 64]
        c, d = T[(iy + 1) % 64, ix % 64], T[(iy + 1) % 64, (ix + 1) % 64]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return n


def _outside(rect, x, y):
    """Distance of (x, y) outside the rectangle (x0, x1, y0, y1); 0 inside."""
    x0, x1, y0, y1 = rect
    return np.hypot(np.maximum(np.maximum(x0 - x, x - x1), 0.0), np.maximum(np.maximum(y0 - y, y - y1), 0.0))


class _Terrain:
    """Height of the ground around a paved rectangle: level at its edge, an earth berm close outside the fence (two
    ridges, 13 and 27 m out, of uneven height), gentle swells, and land rising slowly into low hills several hundred
    metres away, so that the eye meets a ridge before the sky. flats = [(x0, x1, y0, y1, z, margin)]: level pads
    (roads, buildings)."""

    def __init__(self, rect, seed, flats=()):
        self.rect, self.n, self.flats = rect, _value_noise(seed), list(flats)

    def berm(self, x, y):
        return 0.55 + 0.45 * np.clip((self.n(x, y, 60.0) - 0.3) / 0.3, 0.0, 1.0)

    def __call__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d, n = _outside(self.rect, x, y), self.n
        z = (2.8 * np.exp(-((d - 13.0) / 5.0) ** 2) + 2.6 * np.exp(-((d - 27.0) / 5.5) ** 2)) * self.berm(x, y)
        z = z + (n(x, y, 33.0) - 0.5) * 2.2 * np.clip((d - 9.0) / 50.0, 0.0, 1.0)
        z = z + 0.017 * np.clip(d - 75.0, 0.0, None) + (n(x, y, 240.0) - 0.4) * 16.0 * np.clip((d - 110.0) / 260.0, 0.0, 1.0)
        z = -0.04 + z * np.clip((d - 3.5) / 7.0, 0.0, 1.0)
        for fx0, fx1, fy0, fy1, fz, m in self.flats:
            w = np.clip(1.0 - _outside((fx0, fx1, fy0, fy1), x, y) / m, 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            z = z * (1 - w) + fz * w
        return z


def _terrain_mesh(B, T, rect, reach=700.0, near=3.0):
    """The ground as a mesh: 3 m cells to 66 m around the rectangle, growing beyond; with two veils over it (dry and
    dark patches at field scale) so that it does not read as one colour."""
    x0, x1, y0, y1 = rect

    def axis(a0, a1):
        out, step, x = [], near, 66.0
        while x < reach:
            step *= 1.25
            x += step
            out.append(x)
        out = np.array(out)
        return np.concatenate([(a0 - out)[::-1], np.arange(a0 - 66.0, a1 + 66.0 + 1e-6, near), a1 + out])
    xs, ys = axis(x0, x1), axis(y0, y1)
    X, Y = np.meshgrid(xs, ys)
    Z = T(X, Y)
    N = np.stack([-np.gradient(Z, xs, axis=1), -np.gradient(Z, ys, axis=0), np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    nx = len(xs)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(len(ys) - 1))
    keep = ~((xs[i] >= x0 + 0.5) & (xs[i + 1] <= x1 - 0.5) & (ys[j] >= y0 + 0.5) & (ys[j + 1] <= y1 - 0.5))
    a = (j * nx + i)[keep]
    F = np.concatenate([np.stack([a, a + 1, a + nx + 1], 1), np.stack([a, a + nx + 1, a + nx], 1)])
    for mat, dz in (("dy_field", 0.0), ("dy_field_dry", 0.03), ("dy_field_dark", 0.05)):
        B._emit(mat, V + (0.0, 0.0, dz), N.reshape(-1, 3), V[:, :2], F)


def _field_veil(S, k):
    rng = np.random.default_rng(880 + k)
    a = _noise(rng, (512, 512), 5, 4, 0.55, tileable=True)
    return _save(S, f"dy_field_veil{k}.png", np.clip((a - 0.46) * 3.2, 0.0, 0.72))


_MATS.update(
    dy_field_dry=lambda S, n: _pbr(S, n, colour=(0.22, 0.19, 0.10), rough=0.95, opacity_tex=_field_veil(S, 0), tile=170.0),
    dy_field_dark=lambda S, n: _pbr(S, n, colour=(0.020, 0.034, 0.014), rough=0.95, opacity_tex=_field_veil(S, 1), tile=230.0),
    dy_shed_a=_const((0.60, 0.59, 0.55), 0.5, 0.0), dy_shed_b=_const((0.46, 0.49, 0.52), 0.5, 0.0),
    dy_shed_c=_const((0.66, 0.64, 0.58), 0.5, 0.0), dy_band_red=_const((0.30, 0.045, 0.035), 0.45, 0.0),
    dy_band_green=_const((0.04, 0.17, 0.09), 0.45, 0.0), dy_band_grey=_const((0.12, 0.13, 0.14), 0.45, 0.0),
    dy_roof_far=_const((0.30, 0.31, 0.32), 0.6, 0.2), dy_wood_pole=_const((0.13, 0.09, 0.06), 0.85, 0.0),
    dy_cont_blue=_const((0.03, 0.09, 0.26), 0.5, 0.2), dy_cont_red=_const((0.30, 0.055, 0.04), 0.5, 0.2),
    dy_cont_green=_const((0.04, 0.17, 0.11), 0.5, 0.2), dy_cont_grey=_const((0.33, 0.34, 0.35), 0.5, 0.2),
    dy_cont_orange=_const((0.52, 0.19, 0.03), 0.5, 0.2), dy_cont_white=_const((0.68, 0.68, 0.66), 0.5, 0.1))

# ---- plants: Poly Haven models referenced from the asset cache as point-instancer prototypes
_PLANTS = {}                 # model -> [(mesh prim path in its file, [material subset names], height m)]
_PLANT_PARTS = dict(island_tree_02=dict(bark="island_tree_02_diff_1k.jpg", branches="island_tree_02_branches_diff_1k.png",
                                        leaves=("island_tree_02_leaves_diff_1k.png", None)),
                    searsia_lucida=dict(bark="searsia_lucida_diff_1k.jpg",
                                        leaves=("searsia_lucida_diff_1k.jpg", "searsia_lucida_alpha_1k.png")),
                    wild_rooibos_bush=dict(bark="wild_rooibos_bush_diff_1k.jpg",
                                           leaves=("wild_rooibos_bush_diff_1k.jpg", "wild_rooibos_bush_alpha_1k.png")))


def _plant_variants(model):
    """The plants of a model file: [(mesh path, subsets, height)], or [] when the cache does not hold the model."""
    if model not in _PLANTS:
        out = []
        usd = _asset(f"polyhaven_models/{model}/{model}_1k.usdc")
        if usd is not None:
            from pxr import Usd, UsdGeom
            st = Usd.Stage.Open(str(usd))
            for prim in st.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    ext = UsdGeom.Mesh(prim).GetExtentAttr().Get()
                    subs = [c.GetName() for c in prim.GetChildren() if c.IsA(UsdGeom.Subset)]
                    out.append((str(prim.GetPath()), subs, float(ext[1][2] - ext[0][2]) if ext else 1.0))
        _PLANTS[model] = sorted(out)
    return _PLANTS[model]


def _plant_materials(S, model):
    """OmniPBR for a plant's bark, branches and leaves (leaf cards cut out by their alpha)."""
    T = _PLANT_PARTS[model]

    def tex(name):
        return _asset(f"polyhaven_models/{model}/textures/{name}")
    names = {}
    for part in ("bark", "branches", "leaves"):
        if part not in T:
            continue
        name = f"dy_plant_{model}_{part}"
        if name not in S.mats:
            if part == "leaves":
                diff, alpha = T[part]
                _pbr(S, name, diff=tex(diff), rough=0.65, opacity_tex=tex(alpha or diff), cutout=0.35, alpha_channel=alpha is None)
            else:
                _pbr(S, name, diff=tex(T[part]), rough=0.85)
        names[part] = S.mats[name]
    return names


def _plant_proto(S, path, name):
    """Prototype of one plant: a reference to its mesh in the model file, its material subsets bound to ours."""
    model, k = name[len("plant_"):].rsplit("_", 1)
    mesh, subs, _ = _plant_variants(model)[int(k)]
    S.UsdGeom.Xform.Define(S.stage, path)
    prim = S.stage.DefinePrim(path + "/m")
    prim.GetReferences().AddReference(str(_asset(f"polyhaven_models/{model}/{model}_1k.usdc")), mesh)
    S.UsdGeom.Mesh(prim).CreateSubdivisionSchemeAttr("none")
    mats = _plant_materials(S, model)
    S.bind(prim, mats["bark"])
    for sub in subs:
        part = "leaves" if sub.endswith(("_leaves", "_twigs")) else "branches" if sub.endswith("_branches") else "bark"
        sp = S.stage.OverridePrim(f"{path}/m/{sub}")
        S.UsdShade.MaterialBindingAPI.Apply(sp).Bind(mats.get(part, mats["bark"]))


def _scatter_plants(B, T, rng, rect, keep_out, amount=1.0, belt=None):
    """A belt of trees on the berm close outside the fence (belt(x, y) -> its inner distance from the rectangle; it
    is 11 m deep), trees in clumps across the fields and thick along the far ridges; shrubs and dry bushes in the
    rough ground outside the fence. Nothing inside the keep-out rectangles (roads, pads)."""
    trees, shrubs, tufts = (_plant_variants(m) for m in ("island_tree_02", "searsia_lucida", "wild_rooibos_bush"))
    x0, x1, y0, y1 = rect
    n = T.n

    def sample(count, dmin, dmax, accept):
        """Random points at dmin..dmax outside the rectangle that pass `accept` and the keep-outs."""
        pts = []
        for _ in range(40):
            m = 6 * count + 60
            x, y = rng.uniform(x0 - dmax, x1 + dmax, m), rng.uniform(y0 - dmax, y1 + dmax, m)
            d = _outside(rect, x, y)
            ok = (d >= dmin) & (d <= dmax) & accept(x, y, d)
            for kx0, kx1, ky0, ky1 in keep_out:
                ok &= ~((x > kx0) & (x < kx1) & (y > ky0) & (y < ky1))
            pts += list(zip(x[ok], y[ok]))
            if len(pts) >= count:
                break
        return pts[:count]

    def put(model, variants, pts, s0, s1, sink=0.06):
        for x, y in pts:
            k = int(rng.integers(len(variants)))
            s = float(rng.uniform(s0, s1))
            B.inst(f"plant_{model}_{k}", (x, y, float(T(x, y)) - sink * s), _rot(yaw=float(rng.uniform(0, 2 * math.pi))), s)
    if trees:
        per = 2 * (x1 - x0 + y1 - y0)
        inner = belt if belt is not None else (lambda x, y: np.full(np.shape(x), 8.0))
        put("island_tree_02", trees, sample(int(per / 3.3), 6.0, 34.0, lambda x, y, d: (d >= inner(x, y)) & (d <= inner(x, y) + 11.0)),
            2.5, 3.5)
        put("island_tree_02", trees, sample(int(70 * amount), 40.0, 200.0, lambda x, y, d: n(x + 300.0, y, 55.0) > 0.55), 2.2, 3.5)
        put("island_tree_02", trees, sample(int(240 * amount), 200.0, 640.0, lambda x, y, d: n(x, y + 500.0, 95.0) > 0.47), 3.0, 4.6)
    if shrubs:
        put("searsia_lucida", shrubs, sample(int(220 * amount), 3.4, 75.0,
                                             lambda x, y, d: (n(x, y, 17.0) > 0.42) & (rng.random(len(x)) < np.exp(-d / 40.0) + 0.15)),
            0.9, 1.9)
    if tufts:
        put("wild_rooibos_bush", tufts, sample(int(420 * amount), 2.7, 32.0, lambda x, y, d: n(x + 90.0, y, 9.0) > 0.35), 1.2, 2.6,
            sink=0.01)


def _shed(B, w, d, h, wall, band, doors=3, sides=(0, 1, 2, 3)):
    """A plain industrial building seen from afar (origin: the centre of its footprint on the ground, w along x, d
    along y): ribbed walls over a concrete plinth, a coloured band under the eaves, roller doors toward -y, roof
    units. sides: which walls to draw (0: the -y face, then counter-clockwise)."""
    for k, (lg, dp) in enumerate(((w, d), (d, w), (w, d), (d, w))):
        if k not in sides:
            continue
        Bw = B.at(yaw=k * math.pi / 2)
        Bw.rect("dy_concrete", (0.0, -dp / 2, 0.4), _X, _Z, (lg, 0.8))
        _cladding(Bw, wall, -lg / 2, lg / 2, 0.8, h - 1.1, -dp / 2, -1, pitch=0.4, depth=0.04)
        _cladding(Bw, band, -lg / 2, lg / 2, h - 1.1, h, -dp / 2, -1, pitch=0.4, depth=0.04)
    B.rect("dy_roof_far", (0.0, 0.0, h - 0.02), _X, _Y, (w - 0.2, d - 0.2))
    B.box("dy_trim", (0.0, 0.0, h + 0.05), (w + 0.2, d + 0.2, 0.10))
    for k in range(doors):
        x = -w / 2 + (k + 1) * w / (doors + 1)
        B.box("dy_door_ext", (x, -d / 2 - 0.01, 2.4), (4.2, 0.06, 4.8))
        B.box("dy_trim", (x, -d / 2 - 0.02, 4.86), (4.5, 0.08, 0.12))
    if doors:
        B.box("dy_facade_band", (-w / 2 + 2.0, -d / 2 - 0.01, 1.05), (0.95, 0.05, 2.1))
    for k in range(max(2, int(w / 14))):
        B.box("dy_steel_galv", (-w / 2 + 6.0 + k * 13.0, d * (0.15 if k % 2 else -0.2), h + 0.55), (2.0, 1.4, 1.1), bevel=0.05)


def _container(B, mat, length=12.19):
    """ISO shipping container (origin: the centre of its base, long axis x): corrugated sides and front, doors with
    lock rods at -x, corner posts and rails."""
    w, h = 2.438, 2.591
    for sg in (-1, 1):
        _cladding(B, mat, -length / 2 + 0.08, length / 2 - 0.08, 0.16, h - 0.12, sg * w / 2, sg, pitch=0.278, depth=0.036)
        B.box(mat, (0.0, sg * (w / 2 - 0.03), 0.08), (length, 0.06, 0.16))
        B.box(mat, (0.0, sg * (w / 2 - 0.03), h - 0.06), (length, 0.06, 0.12))
        for sx in (-1, 1):
            B.box(mat, (sx * (length / 2 - 0.08), sg * (w / 2 - 0.08), h / 2), (0.16, 0.16, h), bevel=0.01)
    B.rect(mat, (0.0, 0.0, h - 0.01), _X, _Y, (length - 0.1, w - 0.1))
    Bf = B.at(yaw=math.pi / 2)
    _cladding(Bf, mat, -w / 2 + 0.16, w / 2 - 0.16, 0.16, h - 0.12, -length / 2 + 0.02, -1, pitch=0.278, depth=0.036)   # front end (+x)
    B.rect(mat, (-length / 2 + 0.03, 0.0, h / 2), -_Y, _Z, (w - 0.3, h - 0.3))                                            # doors (-x)
    B.box("dy_black", (-length / 2 + 0.028, 0.0, h / 2), (0.004, 0.012, h - 0.3))
    for y in (-0.85, -0.32, 0.32, 0.85):
        B.cyl("dy_steel_galv", (-length / 2 + 0.005, y, 0.2), (-length / 2 + 0.005, y, h - 0.15), 0.017, n=6)


def _utility_line(B, T, pts):
    """Wooden poles carrying three wires along a road."""
    tops = []
    for x, y, yaw in pts:
        z = float(T(x, y))
        Bp = B.at((x, y, z), yaw=yaw)
        Bp.cyl("dy_wood_pole", (0, 0, -0.3), (0, 0, 9.6), 0.13, r2=0.09, n=10)
        Bp.box("dy_wood_pole", (0, 0, 8.9), (0.10, 2.3, 0.12))
        for dy in (-1.0, 0.0, 1.0):
            Bp.cyl("dy_plate_white", (0, dy, 8.96), (0, dy, 9.12 + (0.5 if dy == 0 else 0.0)), 0.035, n=8)
        R = _rot(yaw=yaw)
        tops.append([np.array([x, y, z]) + R @ np.array([0.0, dy, 9.14 + (0.5 if dy == 0 else 0.0)]) for dy in (-1.0, 0.0, 1.0)])
    for a, b in zip(tops[:-1], tops[1:]):
        for pa, pb in zip(a, b):
            f = np.linspace(0.0, 1.0, 9)[:, None]
            B.tube("dy_cable", pa + (pb - pa) * f - _Z * 1.1 * 4 * f * (1 - f), 0.012, n=4, caps=False)


def _surroundings(S, B, rect, seed, road="S", trailers=False, flats=(), keep_out=()):
    """What lies beyond the fence of the paved rectangle `rect` = (x0, x1, y0, y1): rolling ground with a berm, trees
    and scrub; on the `road` side ('S' or 'N') a public road with a pole line and two industrial neighbours (one with
    trailers in its yard); a container stack and a third building to the east, a fourth to the west. All of it is
    scenery where there is no ground to drive on (no colliders)."""
    rng = np.random.default_rng(7000 + seed)
    x0, x1, y0, y1 = rect
    sg = -1.0 if road == "S" else 1.0
    ye = y0 if road == "S" else y1

    def yo(d):                                     # the y at distance d outside the road-side edge
        return ye + sg * d
    T0 = _Terrain(rect, 31 + seed)
    face = math.pi if sg < 0 else 0.0              # the neighbours' doors look toward the road
    sheds = [dict(c=(x0 + 0.35 * (x1 - x0), yo(64.0)), yaw=face, size=(70.0, 30.0, 9.0), wall="dy_shed_a", band="dy_band_red", doors=4),
             dict(c=(x1 + 28.0, yo(84.0)), yaw=face, size=(40.0, 24.0, 7.5), wall="dy_shed_b", band="dy_band_grey", doors=2),
             dict(c=(x1 + 98.0, (y0 + y1) / 2), yaw=math.pi / 2, size=(62.0, 34.0, 11.0), wall="dy_shed_c", band="dy_band_green", doors=4),
             dict(c=(x0 - 80.0, (y0 + y1) / 2 - sg * 8.0), yaw=-math.pi / 2, size=(46.0, 28.0, 8.0), wall="dy_shed_b", band="dy_band_red",
                  doors=3)]
    fl, ko = list(flats), list(keep_out)
    road_y = (min(yo(8.0), yo(15.0)), max(yo(8.0), yo(15.0)))
    fl.append((x0 - 400.0, x1 + 400.0, road_y[0], road_y[1], -0.05, 7.0))
    ko.append((x0 - 400.0, x1 + 400.0, road_y[0] - 1.5, road_y[1] + 2.5))
    for sh in sheds:
        w, d, _ = sh["size"]
        ex, ey = (w / 2, d / 2) if abs(math.cos(sh["yaw"])) > 0.5 else (d / 2, w / 2)
        sh["z"] = float(T0(*sh["c"]))
        fl.append((sh["c"][0] - ex - 6.0, sh["c"][0] + ex + 6.0, sh["c"][1] - ey - 18.0, sh["c"][1] + ey + 18.0, sh["z"], 12.0))
        ko.append((sh["c"][0] - ex - 8.0, sh["c"][0] + ex + 8.0, sh["c"][1] - ey - 20.0, sh["c"][1] + ey + 20.0))
    cy0 = (y0 + y1) / 2 - 16.0
    stack = (x1 + 30.0, x1 + 46.0, cy0 - 2.0, cy0 + 30.0)
    zc = float(T0(x1 + 38.0, cy0 + 14.0))
    fl.append((*stack, zc, 9.0))
    ko.append((x1, stack[1] + 2.0, stack[2] - 2.0, stack[3] + 2.0))             # and no trees between it and the fence
    T = _Terrain(rect, 31 + seed, fl)
    _terrain_mesh(B, T, rect)
    # the road with its centre line, the pole line beyond it
    yr = (road_y[0] + road_y[1]) / 2
    B.rect("dy_asphalt", ((x0 + x1) / 2, yr, -0.035), _X, _Y, (x1 - x0 + 800.0, 7.0))
    for x in np.arange(x0 - 398.0, x1 + 398.0, 9.0):
        B.rect("dy_paint_white", (x, yr, -0.032), _X, _Y, (3.0, 0.12))
    for dy in (-3.35, 3.35):
        B.rect("dy_paint_white", ((x0 + x1) / 2, yr + dy, -0.032), _X, _Y, (x1 - x0 + 796.0, 0.10))
    _utility_line(B, T, [(x, yo(18.5), 0.0) for x in np.arange(x0 - 380.0, x1 + 381.0, 42.0)])
    for sh in sheds:
        _shed(B.at((sh["c"][0], sh["c"][1], sh["z"] - 0.02), yaw=sh["yaw"]), *sh["size"], sh["wall"], sh["band"], sh["doors"])
    # containers stacked east of the fence, long axis along y
    cols = ("dy_cont_blue", "dy_cont_red", "dy_cont_green", "dy_cont_grey", "dy_cont_orange", "dy_cont_white")
    for i in range(4):
        for j in range(2):
            for k in range(int(rng.integers(1, 4))):
                _container(B.at((x1 + 32.5 + 3.6 * i, cy0 + 6.3 + 12.6 * j, zc + 2.6 * k), yaw=math.pi / 2 * (1 if rng.random() < 0.5 else -1)),
                           cols[int(rng.integers(len(cols)))])
    if trailers:                                    # the first neighbour's trailers, backed toward its doors
        sh = sheds[0]
        for i in range(3):
            bx = dict(centre=(sh["c"][0] - 10.0 + 4.2 * i, sh["c"][1] - sg * (sh["size"][1] / 2 + 9.5), 2.2 + sh["z"]), half=(1.3, 8.0, 1.3))
            D, origin, yaw, h = _parked_dims(bx)
            Bt = B.at(origin, yaw=yaw) if sg < 0 else B.at((origin[0], bx["centre"][1] + 8.0 - 0.175, origin[2]), yaw=-math.pi / 2)
            _trailer_body(S, Bt, D, "closed", i + 1, 2210 + 97 * i)
            _trailer_gear(S, Bt, D, h - sh["z"], i + 1, D["L"] - 3.4, (2.6, 3.3, 2.9)[i], P.TRAILER["track"])
    def belt(x, y):                                # the tree belt stands behind the road on its side, else on the first berm
        on_road_side = (y < y0) if sg < 0 else (y > y1)
        return np.where(on_road_side, 18.5, 7.0)
    _scatter_plants(B, T, rng, rect, ko, amount=min(2.5, 0.6 + (x1 - x0 + y1 - y0) / 300.0), belt=belt)
    return T


# =====================================================================================================================
# lab preset: the experiments' outdoor test pad
# =====================================================================================================================
def _surface_quad(s):
    """Corners (counter-clockwise from above), slope direction, cross direction and normal of a tyre surface."""
    n = np.asarray(s.normal, float) / np.linalg.norm(s.normal)
    u = np.asarray(s.u, float) / np.linalg.norm(s.u)
    v = np.cross(n, u)
    o = np.asarray(s.origin, float)
    hu, hv = s.half
    return [o - hu * u - hv * v, o + hu * u - hv * v, o + hu * u + hv * v, o - hu * u + hv * v], u, v, n


def _digit_atlas(S):
    """Opacity of painted numerals 0-9 in a row (stencil face), for distance marks."""
    from PIL import Image, ImageDraw
    if _fresh(S, "dy_digits"):
        im = Image.new("L", (960, 144), 0)
        d = ImageDraw.Draw(im)
        f = _font(132)
        for g in range(10):
            bb = d.textbbox((0, 0), str(g), font=f)
            d.text((g * 96 + (96 - (bb[2] - bb[0])) / 2 - bb[0], (144 - (bb[3] - bb[1])) / 2 - bb[1]), str(g), font=f, fill=235)
        wear = _noise(np.random.default_rng(5), (144, 960), (6, 40), 3)
        _save(S, "dy_digits.png", np.asarray(im, np.float32) / 255.0 * np.clip(0.55 + 0.9 * wear, 0.0, 1.0))
    return S.folder / "textures" / "dy_digits.png"


def _wet_sheet(S):
    """Opacity of a film of water lying on a paved surface: thin everywhere, pooled in patches (seamless tile)."""
    a = _noise(np.random.default_rng(64), (512, 512), 5, 4, 0.5, tileable=True)
    return _save(S, "dy_wet_sheet.png", np.clip(0.62 + (a - 0.45) * 2.2, 0.50, 0.93))


_MATS.update(
    dy_digits=lambda S, n: _pbr(S, n, colour=(0.80, 0.80, 0.78), rough=0.55, opacity_tex=_digit_atlas(S)),
    dy_wet_sheet=lambda S, n: _pbr(S, n, colour=(0.012, 0.012, 0.014), rough=0.04, opacity_tex=_wet_sheet(S), tile=11.0),
    dy_pad_veil=lambda S, n: (lambda v: _pbr(S, n, colour=(0.10, 0.095, 0.085), rough=0.8, tile=30.0, opacity_tex=v[2]) if v else
                              S.color(n, (0.10, 0.095, 0.085), 0.8, 0.0, opacity=0.0))(_asphalt_overlays(S)))


def _numeral(B, value, c, u, n, height=0.9):
    """A painted number on the ground, centred at c, reading along u with its top toward n."""
    text = str(int(round(value)))
    wd = 0.62 * height
    for k, ch in enumerate(text):
        cc = np.asarray(c, float) + u * (k - (len(text) - 1) / 2) * wd
        g = int(ch)
        B.poly("dy_digits", [cc - u * wd / 2 - n * height / 2, cc + u * wd / 2 - n * height / 2, cc + u * wd / 2 + n * height / 2,
                             cc - u * wd / 2 + n * height / 2], uv=[(g / 10, 0.0), ((g + 1) / 10, 0.0), ((g + 1) / 10, 1.0), (g / 10, 1.0)])


def _lab_ramp(S, B, s, wet, label_side):
    """A test ramp surface drawn where it is, as part of a solid cast ramp: the driven face, painted edge lines,
    and walls from its edges down to the pad. (Only the face has a collider: a 0.3 m slab.)"""
    Q, u, v, n = _surface_quad(s)
    hu, hv = s.half
    o = np.asarray(s.origin, float)
    lift = 0.0012 * n
    B.poly("dy_asphalt" if str(s.material).startswith("asphalt") else "dy_apron", Q, uv=[(q @ u, q @ v) for q in Q])
    for sgn in (-1, 1):
        c = o + sgn * (hv - 0.16) * v + 2 * lift
        B.rect("dy_paint_yellow", c, u, v, (2 * hu, 0.10))
    for a in np.arange(-hu + 3.0, hu - 0.5, 3.0):
        B.rect("dy_seal_tar", o + a * u + lift, u, v, (0.012, 2 * hv - 0.4))
    if wet:
        B.rect("dy_wet_sheet", o + 4 * lift, u, v, (2 * hu, 2 * hv))
    ctr = o.copy()
    for k in range(4):
        a, b = Q[k], Q[(k + 1) % 4]
        if max(a[2], b[2]) < 0.02:
            continue
        ga, gb = np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])
        poly = [ga, gb, b, a] if b[2] >= a[2] else [gb, ga, a, b]
        nrm = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        mid = (a + b) / 2 - ctr
        if nrm @ np.array([mid[0], mid[1], 0.0]) < 0:
            poly = poly[::-1]
        B.poly("dy_concrete", poly)
    grade = abs(u[2]) / max(math.hypot(u[0], u[1]), 1e-9)
    top = min(q[2] for q in Q)
    if grade < 1e-4 and top > 0.7 and label_side is not None:           # the level top: the grade painted on its side wall
        ht = min(0.45, 0.45 * top)
        key = "".join(ch if ch.isdigit() else "_" for ch in label_side["text"]).strip("_")
        img = _tex_once(S, f"dy_grade_{key}.png",
                        lambda p: _text_image([(label_side["text"], 1.0)], (512, 128), (235, 235, 235), (0, 0, 0)).convert("L").save(p))
        mat = f"dy_grade_{key}"
        if mat not in S.mats:
            _pbr(S, mat, colour=(0.80, 0.80, 0.78), rough=0.55, opacity_tex=img)
        side = -v if (v @ label_side["toward"]) < 0 else v
        c = o + side * (hv + 0.002) - _Z * (top * 0.45)
        uu = np.cross(_Z, side)
        B.rect(mat, c, uu, _Z, (4 * ht, ht), uv01=True)


def draw_lab(S, world):
    """The lab preset (no layout): an outdoor vehicle test pad. Every drivable surface of the world is drawn where
    it is: the floor rectangle as a jointed concrete (or asphalt) pad with its edge line, a lane along the truck's
    heading from its start pose with a start line, metre ticks and numerals every 5 m, the experiment's ramps with
    their real slope, width and top, and a water film where the surface's material is wet. Around the pad, where
    there is no ground to drive on: an asphalt apron, a fence with light poles, a site cabin, and the surroundings."""
    sc = world.scene
    surfs = [s for s in sc.surfaces if s.body < 0]
    floor = next((s for s in surfs if s.name == "floor"), None)
    if floor is None:
        return
    B = _Build()
    Q, _, _, _ = _surface_quad(floor)
    x0, x1 = min(q[0] for q in Q), max(q[0] for q in Q)
    y0, y1 = min(q[1] for q in Q), max(q[1] for q in Q)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    asphalt = str(floor.material).startswith("asphalt")
    wet = str(floor.material).endswith("_wet")
    # ---- the pad
    B.rect("dy_asphalt" if asphalt else "dy_apron", (cx, cy, 0.0), _X, _Y, (x1 - x0, y1 - y0))
    B.rect("dy_pad_veil", (cx, cy, 0.0006), _X, _Y, (x1 - x0, y1 - y0))
    if not asphalt:
        for x in np.arange(x0 + 6.0, x1 - 0.1, 6.0):
            B.rect("dy_seal_tar", (x, cy, 0.0012), _X, _Y, (0.012, y1 - y0))
        for y in np.arange(y0 + 6.0, y1 - 0.1, 6.0):
            B.rect("dy_seal_tar", (cx, y, 0.0012), _X, _Y, (x1 - x0, 0.012))
    for a, b in (((x0 + 0.5, y0 + 0.5), (x1 - 0.5, y0 + 0.5)), ((x1 - 0.5, y0 + 0.5), (x1 - 0.5, y1 - 0.5)),
                 ((x1 - 0.5, y1 - 0.5), (x0 + 0.5, y1 - 0.5)), ((x0 + 0.5, y1 - 0.5), (x0 + 0.5, y0 + 0.5))):
        _line(B, a, b, 0.12, "dy_paint_yellow")
    # ---- the measuring lane from the truck's start pose along its heading
    xs, ys_, yaw = sc.truck_pose
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    n = np.array([-u[1], u[0], 0.0])
    p0 = np.array([xs, ys_, 0.0])

    def inside(p, m=0.7):
        return x0 + m <= p[0] <= x1 - m and y0 + m <= p[1] <= y1 - m

    def reach(sign):                                 # how far the lane runs from the start before leaving the pad
        s = 0.0
        while s < 2000.0 and all(inside(p0 + sign * (s + 1.0) * u + w * n) for w in (-2.0, 2.0)):
            s += 1.0
        return s
    if inside(p0 - 2.0 * n) and inside(p0 + 2.0 * n):
        s0, s1 = -min(reach(-1.0), 5.0), reach(1.0)
        for w in (-2.0, 2.0):
            _line(B, p0 + s0 * u + w * n, p0 + s1 * u + w * n, 0.10, "dy_paint_white")
        _line(B, p0 - 2.0 * n, p0 + 2.0 * n, 0.25, "dy_paint_white")
        for s in np.arange(5.0, s1 + 1e-6, 5.0):
            _line(B, p0 + s * u - 2.0 * n, p0 + s * u + 2.0 * n, 0.10, "dy_paint_white")
        for s in np.arange(1.0, min(s1, 30.0) + 1e-6, 1.0):
            if s % 5.0:
                _line(B, p0 + s * u - 2.0 * n, p0 + s * u - 1.55 * n, 0.06, "dy_paint_white")
        for s in np.arange(0.0, s1 + 1e-6, 5.0):
            if s <= 60.0 or s % 10.0 == 0.0:
                c = p0 + s * u - 2.85 * n
                if inside(c, 1.2):
                    _numeral(B, s, c + (0.0, 0.0, 0.0034), u, n)
    if wet:
        B.rect("dy_wet_sheet", (cx, cy, 0.0046), _X, _Y, (x1 - x0, y1 - y0))
    # ---- ramps and any other surface the experiment asked for
    for s in surfs:
        if s is floor:
            continue
        _, su, _, _ = _surface_quad(s)
        g = abs(su[2]) / max(math.hypot(su[0], su[1]), 1e-9)
        inc = next((t for t in surfs if t.name + "_top" == s.name), None)
        if inc is not None:
            _, tu, _, _ = _surface_quad(inc)
            g_inc = abs(tu[2]) / max(math.hypot(tu[0], tu[1]), 1e-9)
        label = dict(text=f"{100 * g_inc:.1f} %", toward=-n) if inc is not None and g < 1e-4 else None
        _lab_ramp(S, B, s, str(s.material).endswith("_wet"), label)
    # ---- off the pad (render only): apron, fence, poles, a site cabin; then the surroundings. The fence stands 14 m
    # outside everything that can be driven on (the pad and any ramp reaching beyond it)
    m = 14.0
    allq = np.array([q for s in surfs for q in _surface_quad(s)[0]])
    fx0, fx1 = min(x0, allq[:, 0].min()) - m, max(x1, allq[:, 0].max()) + m
    fy0, fy1 = min(y0, allq[:, 1].min()) - m, max(y1, allq[:, 1].max()) + m
    B.rect("dy_asphalt", (cx, cy, -0.006), _X, _Y, (fx1 - fx0 + 1.0, fy1 - fy0 + 1.0))
    B.rect("dy_asphalt_dark", (cx, cy, -0.004), _X, _Y, (fx1 - fx0 + 1.0, fy1 - fy0 + 1.0))
    B.rect("dy_asphalt_light", (cx, cy, -0.002), _X, _Y, (fx1 - fx0 + 1.0, fy1 - fy0 + 1.0))
    B.rect("dy_gravel", (cx, cy, -0.02), _X, _Y, (fx1 - fx0 + 6.0, fy1 - fy0 + 6.0))
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        _line(B, a, b, 0.035, "dy_seal_tar", z=0.0006)
    for a, b, o in (((fx0, fy0), (fx1, fy0), (0.0, -1.0)), ((fx1, fy0), (fx1, fy1), (1.0, 0.0)), ((fx1, fy1), (fx0, fy1), (0.0, 1.0)),
                    ((fx0, fy1), (fx0, fy0), (-1.0, 0.0))):
        _fence(B, a, b, o)
    for x, y, tw in ((fx0 + 0.9, fy0 + 0.9, math.pi / 4), (fx1 - 0.9, fy0 + 0.9, 3 * math.pi / 4), (fx1 - 0.9, fy1 - 0.9, -3 * math.pi / 4),
                     (fx0 + 0.9, fy1 - 0.9, -math.pi / 4)):
        _light_pole(B.at((x, y, 0.0), yaw=tw))
    cab = np.array([min(max(xs - 4.0, fx0 + 6.0), fx1 - 6.0), fy1 - 3.4, 0.0])       # the site cabin, north of the start
    Bc = B.at(cab)
    _container(Bc.at((0.0, 0.0, 0.12)), "dy_cont_white", length=6.06)
    for sx in (-2.6, 2.6):
        for sy in (-1.0, 1.0):
            Bc.box("dy_concrete", (sx, sy, 0.06), (0.4, 0.3, 0.12))
    Bc.box("dy_glass", (1.2, -1.222, 1.75), (1.1, 0.01, 0.8))
    Bc.box("dy_facade_band", (-1.3, -1.224, 1.2), (0.9, 0.012, 2.0))
    for k in range(4):
        kit(S, f"/World/dockyard/kit/lab_cone{k}", "S_TrafficCone.usd", (cab[0] + 3.9 + 0.45 * k, cab[1] - 1.7, 0.0))
    _surroundings(S, B, (fx0, fx1, fy0, fy1), 3, road="N", keep_out=[(fx0 - 3.0, fx1 + 3.0, fy0 - 3.0, fy1 + 3.0)])
    B.flush(S, "/World/dockyard/lab")


# ---------------------------------------------------------------------------------------------------- yard dressing
def _proto_fence_post(B):
    """Line post of a chain-link fence, 2.4 m of fabric: post, cap and the barbed-wire arm leaning out (+y)."""
    B.cyl("dy_steel_galv", (0, 0, 0), (0, 0, 2.47), 0.03, n=10, caps=False)
    B.revolve("dy_steel_galv", [(0.033, 2.45), (0.033, 2.48), (0.02, 2.50), (0.0, 2.505)], n=10)
    B.beam("dy_steel_galv", (0.0, 0.0, 2.44), (0.0, 0.12, 2.70), (0.022, 0.008))


def _proto_link(B):
    """One link of 8 mm proof-coil chain, 46 mm long, lying in the local x-y plane along x."""
    f = np.linspace(0, 2 * math.pi, 13)
    B.tube("dy_steel_dark", np.stack([0.019 * np.cos(f), 0.011 * np.sin(f), np.zeros(13)], 1), 0.004, n=6, caps=False)


def _proto_yard_wheel(B):
    """Pneumatic wheel of the ramp's undercarriage, 0.46 m across, axle along +y."""
    B.revolve("dy_rubber", [(0.13, -0.06), (0.19, -0.075), (0.222, -0.06), (0.23, -0.035), (0.23, 0.035), (0.222, 0.06),
                            (0.19, 0.075), (0.13, 0.06)], axis=_Y, n=28)
    B.revolve("dy_yellow", [(0.13, 0.06), (0.12, 0.045), (0.04, 0.05), (0.03, 0.08), (0.0, 0.08)], axis=_Y, n=20)
    B.revolve("dy_yellow", [(0.0, -0.05), (0.13, -0.06)], axis=_Y, n=20)


_PROTOS.update(fence_post=_proto_fence_post, link=_proto_link, yard_wheel=_proto_yard_wheel)


def _wall_frame(B, it):
    """Builder at the wall point behind a wall-mounted item, +y out of the wall (the item's yaw)."""
    p, half, yaw = np.array(it["p"]), it["half"], it["yaw"]
    return B.at(p - half[1] * np.array([-math.sin(yaw), math.cos(yaw), 0.0]), yaw=yaw)


def _fence(B, a, b, out, h=2.4):
    """Chain-link fence from a to b (ground points): posts every 3 m, top rail, tension wire, fabric, three strands of
    barbed wire on arms leaning toward `out`, a concrete mow strip."""
    a, b = np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])
    d = b - a
    Lg = float(np.linalg.norm(d))
    u = d / Lg
    out = np.array([out[0], out[1], 0.0])
    R = np.stack([np.cross(out, _Z), out, _Z], 1)
    n = max(1, int(round(Lg / 3.0)))
    for k in range(n + 1):
        B.inst("fence_post", a + u * (Lg * k / n), R)
    B.cyl("dy_steel_galv", a + h * _Z, b + h * _Z, 0.021, n=8)
    B.cyl("dy_steel_galv", a + 0.07 * _Z, b + 0.07 * _Z, 0.004, n=5)
    B.rect("dy_chainlink", (a + b) / 2 + (h / 2 + 0.02) * _Z, u, _Z, (Lg, h - 0.04))
    for j in range(3):
        o = out * (0.03 + 0.04 * j) + _Z * (h + 0.10 + 0.085 * j)
        B.cyl("dy_steel_galv", a + o, b + o, 0.003, n=4, caps=False)
    B.beam("dy_concrete", a + 0.04 * _Z, b + 0.04 * _Z, (0.2, 0.08))


def _draw_fence(S, world, B, it):
    _fence(B, it["a"], it["b"], it["out"], it["height"])


def _light_pole(B, head=True, pole=True):
    """Yard light pole (origin at its foot, the arm toward +x): concrete base, tapered steel pole; arm and LED head."""
    if pole:
        B.cyl("dy_concrete", (0, 0, 0), (0, 0, 0.75), 0.28, n=20)
        B.box("dy_steel_galv", (0, 0, 0.76), (0.30, 0.30, 0.02))
        for sx in (-0.11, 0.11):
            for sy in (-0.11, 0.11):
                B.inst("bolt", (sx, sy, 0.77))
        B.cyl("dy_steel_galv", (0, 0, 0.77), (0, 0, 9.0), 0.085, r2=0.055, n=14)
    if head:
        B.tube("dy_steel_galv", [(0, 0, 8.9), (0.25, 0, 9.15), (1.1, 0, 9.25)], 0.03, n=8)
        B.box("dy_trim", (1.30, 0, 9.22), (0.62, 0.30, 0.09), bevel=0.02)
        B.rect("dy_clear_lens", (1.30, 0, 9.174), _Y, _X, (0.24, 0.50))


def _draw_pole(S, world, B, it):
    _light_pole(B.at((it["foot"][0], it["foot"][1], 0.0), yaw=it["yaw"]), head=it["kind"] == "light_pole_head",
                pole=it["kind"] == "light_pole")


def _draw_bollard(S, world, B, it):
    Bp = B.at((it["p"][0], it["p"][1], 0.0))
    h, r = 2 * it["half"][2] - 0.004, 0.084
    Bp.cyl("dy_yellow", (0, 0, 0), (0, 0, 0.012), r + 0.07, n=20)
    Bp.cyl("dy_yellow", (0, 0, 0.012), (0, 0, h - 0.05), r, n=20, caps=False)
    Bp.revolve("dy_yellow", [(r, h - 0.05), (0.9 * r, h - 0.015), (0.55 * r, h), (0.0, h + 0.004)], n=20)
    Bp.cyl("dy_tape_white", (0, 0, h - 0.30), (0, 0, h - 0.20), r + 0.001, n=20, caps=False)


def _draw_downspout(S, world, B, it):
    Bw = _wall_frame(B, it)
    hz = it["half"][2]
    Bw.box("dy_facade", (0.0, 0.045, 0.05), (0.10, 0.075, 2 * hz - 0.60), bevel=0.008)
    Bw.box("dy_facade", (0.0, 0.07, hz - 0.125), (0.24, 0.14, 0.25), bevel=0.012)
    Bw.beam("dy_facade", (0.0, 0.045, -hz + 0.36), (0.0, 0.21, -hz + 0.12), (0.10, 0.075))
    for z in np.arange(-hz + 0.9, hz - 0.5, 1.8):
        Bw.box("dy_steel_galv", (0.0, 0.045, z), (0.16, 0.08, 0.03))


def _draw_wall_pack(S, world, B, it):
    Bw = _wall_frame(B, it)
    Bw.box("dy_trim", (0.0, 0.08, 0.0), (0.34, 0.16, 0.22), bevel=0.02)
    Bw.rect("dy_clear_lens", (0.0, 0.085, -0.111), -_X, _Y, (0.26, 0.11))


def _draw_seal_side(S, world, B, it):
    half = it["half"]
    _seal_side(_wall_frame(B, it).at((0.0, 0.0, -half[2])), 2 * half[0], 2 * half[1] - 0.009, 2 * half[2], it["inner"])


def _draw_bumper(S, world, B, it):
    half = it["half"]
    _bumper(_wall_frame(B, it).at((0.0, half[1], 0.0)), (half[0] - 0.07, half[1], half[2]))


def _draw_stowed_lip(S, world, B, it):
    """The lip of a stowed leveler hanging in its keepers on the dock face."""
    Bw = _wall_frame(B, it)
    hx, _, hz = it["half"]
    Bw.box("dy_tread", (0.0, 0.014, -0.02), (2 * hx, 0.016, 2 * hz - 0.04), bevel=0.003)
    Bw.cyl("dy_steel_dark", (-hx, 0.022, hz - 0.022), (hx, 0.022, hz - 0.022), 0.022, n=10)
    for sx in (-0.6 * hx, 0.6 * hx):
        Bw.box("dy_steel_dark", (sx, 0.02, -hz + 0.04), (0.10, 0.04, 0.08), bevel=0.004)


def _draw_building_sign(S, world, B, it):
    Bw = _wall_frame(B, it)
    hx, hy, hz = it["half"]
    img = _sign(S, "dy_building_sign.png", [(BUILDING["name"], 1.6), (BUILDING["line"], 0.7)], (1536, 384), (240, 240, 236),
                BUILDING["colour"])
    Bw.box("dy_trim", (0.0, hy, 0.0), (2 * hx, 2 * hy, 2 * hz), bevel=0.004)
    Bw.rect(_image_mat(S, "dy_building_sign", img), (0.0, 2 * hy + 0.0006, 0.0), -_X, _Z, (2 * hx - 0.06, 2 * hz - 0.06), uv01=True)


def _yard_trailer_frame(L):
    """Pose of the spotted trailer's frame from the layout: origin (rear sill centre, floor top), rotation, height."""
    yt = L.zones["yard_trailer"]
    return np.array([yt["rear"][0], yt["rear"][1], yt["bed_z"]]), _rot(yaw=yt["yaw"]), float(yt["bed_z"])


def _draw_chock(S, world, B, it):
    B.inst("chock", (it["p"][0], it["p"][1], 0.0), _rot(yaw=it["yaw"]))


def _draw_stand(S, world, B, it):
    """Trailer stabilising jack under the nose of a van standing without a tractor, run up to touch the floor box."""
    top = 2 * it["half"][2]
    Bp = B.at((it["p"][0], it["p"][1], 0.0))
    Bp.box("dy_steel_dark", (0, 0, 0.006), (0.44, 0.44, 0.012), bevel=0.003)
    for sx in (-1, 1):
        for sy in (-1, 1):
            Bp.beam("dy_yellow", (sx * 0.19, sy * 0.19, 0.012), (sx * 0.035, sy * 0.035, 0.55 * top), (0.04, 0.04))
    Bp.cyl("dy_yellow", (0, 0, 0.012), (0, 0, 0.62 * top), 0.045, n=12)
    Bp.cyl("dy_steel_galv", (0, 0, 0.62 * top), (0, 0, top - 0.02), 0.028, n=12)
    Bp.box("dy_steel_dark", (0, 0, top - 0.01), (0.22, 0.22, 0.02), bevel=0.003)
    Bp.tube("dy_steel_galv", [(0.0, 0.0, 0.60 * top), (0.16, 0.0, 0.60 * top), (0.18, 0.0, 0.57 * top)], 0.008, n=6)


def _draw_cone(S, world, B, it):
    kit(S, f"/World/dockyard/kit/{it['name']}", "S_TrafficCone.usd", (it["p"][0], it["p"][1], 0.0))


def _draw_parked_part(S, world, B, it):
    """One of the parts under a parked van: its tandem, rear impact guard, landing gear or kingpin."""
    L = world.scene.layout
    bx, gear, i = next(t for t in _parked_trailers(L) if t[0]["name"] == it["box"])
    D, origin, yaw, h = _parked_dims(bx)
    Bt = B.at(origin, yaw=yaw)
    k = it["kind"]
    if k == "parked_tandem":
        _running_gear(S, Bt, D, h, it["tandem"], P.TRAILER["track"], LIVERIES[(i + 1) % len(LIVERIES)])
    elif k == "parked_rear_guard":
        _rear_guard(Bt, D)
    elif k == "parked_landing_gear":
        _landing_gear(Bt, D, h, _parked_gear_x(bx, gear, D, origin), P.TRAILER["track"])
    else:
        _kingpin(Bt, D)


def _parked_gear_x(bx, gear, D, origin):
    """Where a parked van's landing gear stands along it (from the layout's gear box, else 3.4 m behind the nose)."""
    return (gear["centre"][1] - origin[1]) if gear is not None else D["L"] - 3.4


def _ramp_deck_z(rz, y):
    return rz["deck"] - max(0.0, y - rz["y_top"]) * rz["grade"]


def _draw_ramp_axle(S, world, B, it):
    """The ramp's undercarriage: axle, two pneumatic wheels and the legs up to the frame."""
    rz = world.scene.layout.zones["ramp"]
    x, yu = rz["x"], it["y_axle"]
    zd = _ramp_deck_z(rz, yu)
    for sg in (-1, 1):
        B.inst("yard_wheel", (x + sg * 0.88, yu, 0.23), _rot(yaw=-sg * math.pi / 2))       # the wheels roll along the ramp
        B.box("dy_yellow", (x + sg * 0.74, yu, (0.23 + zd - 0.28) / 2), (0.09, 0.09, zd - 0.28 - 0.23), bevel=0.006)
    B.cyl("dy_steel_dark", (x - 0.84, yu, 0.23), (x + 0.84, yu, 0.23), 0.03, n=12)


def _draw_ramp_pump(S, world, B, it):
    """Hand pump of the ramp's hydraulic leg on the outside of its west side channel, with its lever."""
    p, h = np.array(it["p"]), it["half"]
    B.box("dy_steel_dark", (p[0], p[1] - h[1] + 0.08, p[2] - h[2] + 0.13), (0.12, 0.16, 0.26), bevel=0.01)
    B.tube("dy_steel_galv", [(p[0], p[1] - h[1] + 0.08, p[2] - h[2] + 0.26), (p[0], p[1] - h[1] + 0.10, p[2] - h[2] + 0.45),
                             (p[0], p[1] + h[1] - 0.012, p[2] + h[2] - 0.012)], 0.012, n=8)


def _draw_ramp_frame(S, world, B, it):
    """What carries the ramp beneath its deck: the side channels below deck level and their skirt plates at the foot,
    the cross beam over the legs, brace frames, the hydraulic cylinder and its hose."""
    rz = world.scene.layout.zones["ramp"]
    x, w = rz["x"], rz["width"]
    yu = it["y_axle"]
    zd = _ramp_deck_z(rz, yu)
    ang = math.atan(rz["grade"])
    yk = rz["y_foot"] - 0.30 / rz["grade"]                           # below here the channel tapers out into a skirt plate
    for sg in (-1, 1):
        xs = x + sg * (w / 2 + 0.05)
        B.beam("dy_yellow", (xs, yk, _ramp_deck_z(rz, yk) - 0.14), (xs, rz["y_top"], rz["deck"] - 0.14),
               (0.10, 0.28), up=(0.0, math.sin(ang), math.cos(ang)), bevel=0.006)
        B.poly("dy_yellow", [(xs + sg * 0.05, rz["y_foot"], 0.0), (xs + sg * 0.05, yk, 0.0),
                             (xs + sg * 0.05, yk, _ramp_deck_z(rz, yk))])
        B.box("dy_yellow", (xs, (rz["y_top"] + rz["level_end"]) / 2, rz["deck"] - 0.14),
              (0.10, rz["y_top"] - rz["level_end"], 0.28), bevel=0.006)
        for dy in (1.25, -1.25):
            B.beam("dy_yellow", (x + sg * 0.74, yu, 0.50), (x + sg * 0.74, yu + dy, _ramp_deck_z(rz, yu + dy) - 0.20), (0.06, 0.06))
    B.box("dy_yellow", (x, yu, zd - 0.24), (1.60, 0.08, 0.08), bevel=0.005)
    B.cyl("dy_steel_dark", (x, yu - 0.10, 0.52), (x, yu - 0.55, _ramp_deck_z(rz, yu - 0.55) - 0.22), 0.045, n=14)
    B.cyl("dy_steel_galv", (x, yu - 0.55, _ramp_deck_z(rz, yu - 0.55) - 0.22), (x, yu - 0.85, _ramp_deck_z(rz, yu - 0.85) - 0.16),
          0.022, n=10)
    B.tube("dy_cable", [(x - w / 2 - 0.02, yu + 0.66, zd - 0.30), (x - 0.6, yu + 0.4, zd - 0.36), (x, yu - 0.12, 0.56)], 0.008, n=6)


def _draw_chain(S, world, B, it):
    """A safety chain from the ramp head to the trailer's rear frame, hanging slack."""
    a, b = np.array(it["a"]), np.array(it["b"])
    n = max(4, int(np.linalg.norm(b - a) * 1.06 / 0.036))
    for k in range(n):
        f0, f1 = k / n, (k + 1) / n
        p0 = a + (b - a) * f0 - _Z * it["sag"] * 4 * f0 * (1 - f0)
        p1 = a + (b - a) * f1 - _Z * it["sag"] * 4 * f1 * (1 - f1)
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        side = np.cross(_Z, t)
        side /= np.linalg.norm(side)
        up = np.cross(t, side)
        B.inst("link", (p0 + p1) / 2, np.stack([t, side, up], 1) if k % 2 else np.stack([t, up, -side], 1))
    for q in (a, b):
        B.box("dy_steel_dark", q, (0.04, 0.04, 0.05), bevel=0.006)


def _dressing_yard(L, seed):
    """Render-only objects of the yard preset (see dressing)."""
    from ..facility import BUMPER
    out = []
    F = _facade_plan(L)
    yf, zt = F["y"], F["zt"]
    pi = math.pi
    for o in F["ops"]:
        x, w, h, z = o["x"], o["w"], o["h"], o["z"]
        if o["kind"] == "dock":
            nm = f"dock{o['n']}"
            for sg in (-1, 1):
                out.append(_item(f"{nm}_seal{'W' if sg < 0 else 'E'}", "seal_side", (x + sg * (w / 2 + 0.15), yf - 0.155, z + h / 2),
                                 (0.15, 0.155, h / 2), yaw=pi, inner=sg))
                out.append(_item(f"{nm}_bumper{'W' if sg < 0 else 'E'}", "dock_bumper", (x + sg * 1.14, yf - BUMPER / 2, z - 0.22),
                                 (0.19, BUMPER / 2, 0.18), yaw=pi))
            out.append(_item(f"{nm}_seal_head", "seal_head", (x, yf - 0.15, z + h + 0.15), (w / 2 + 0.30, 0.15, 0.15), yaw=pi))
            out.append(_item(f"{nm}_seal_curtain", "seal_curtain", (x, yf - 0.296, z + h - 0.225), (w / 2 - 0.01, 0.005, 0.225), yaw=pi,
                             collide=False, why="hanging fabric"))
            out.append(_item(f"{nm}_lip", "stowed_lip", (x, yf - 0.022, z - 0.20), (0.97, 0.022, 0.20), yaw=pi, collide=False,
                             why="a plate 4 cm proud of the dock face, 0.8 m above the ground"))
            out.append(_item(f"{nm}_number", "sign", (x, yf - 0.004, z + h + 0.85), (0.36, 0.004, 0.36), yaw=pi, text=f"number_{o['n']}",
                             collide=False, why="flat sign"))
            out.append(_item(f"{nm}_signal", "restraint_signal", (x + w / 2 + 0.457, yf - 0.095, z + 2.55), (0.085, 0.095, 0.18), yaw=pi,
                             state="green: no trailer is held at this door"))
            out.append(_item(f"{nm}_sign", "sign", (x + w / 2 + 0.457, yf - 0.004, z + 3.05), (0.15, 0.004, 0.19), yaw=pi,
                             text="move_on_green", collide=False, why="flat sign"))
        elif o["kind"] == "drive":
            for sg in (-1, 1):
                for dy in (0.45, 1.65):
                    out.append(_item(f"bollard_{'W' if sg < 0 else 'E'}{int(dy > 1)}", "bollard", (x + sg * (w / 2 + 0.45), yf - dy, 0.537),
                                     (0.155, 0.155, 0.537)))
            out.append(_item("drive_number", "sign", (x, yf - 0.004, h + 0.75), (0.36, 0.004, 0.36), yaw=pi, text="number_20",
                             collide=False, why="flat sign"))
            out.append(_item("drive_light", "wall_pack", (x, yf - 0.08, h + 1.6), (0.17, 0.08, 0.11), yaw=pi, collide=False,
                             why="small fitting 5.8 m up the wall"))
        elif o["kind"] == "personnel":
            out.append(_item("door_light", "wall_pack", (x, yf - 0.08, h + 0.45), (0.17, 0.08, 0.11), yaw=pi, collide=False,
                             why="small fitting 2.5 m up the wall"))
    for k, x in enumerate(np.arange(F["x0"] + 8.6, F["x1"] - 3.0, 14.63)):
        if all(abs(x - o["x"]) > o["w"] / 2 + 0.5 for o in F["ops"]):
            out.append(_item(f"wall_light{k}", "wall_pack", (x, yf - 0.08, 6.72), (0.17, 0.08, 0.11), yaw=pi, collide=False,
                             why="small fitting 6.6 m up the wall"))
    for k, x in enumerate((F["x0"] + 1.2, L.zones["ramp"]["x"] - 4.2, L.zones["yard_lane"]["points"][0][0] - 4.5, F["x1"] - 1.2)):
        out.append(_item(f"downspout{k}", "downspout", (x, yf - 0.13, (zt - 0.15 + 0.10) / 2), (0.12, 0.13, (zt - 0.15 - 0.10) / 2), yaw=pi,
                         collide=False, why="a pipe 9 cm proud of the wall"))
    wins = [o for o in F["ops"] if o["kind"] == "window"]
    if wins:
        out.append(_item("building_sign", "building_sign", (float(np.mean([o["x"] for o in wins])), yf - 0.02, 5.6), (3.6, 0.02, 0.9), yaw=pi,
                         collide=False, why="flat sign 4.7 m up the wall"))
    # ---- the ramp's undercarriage and chains, chocks and stands at the spotted trailer, cones at the ramp foot
    rz = L.zones["ramp"]
    yu = rz["y_top"] + 2.9
    zdu = _ramp_deck_z(rz, yu)
    out.append(_item("ramp_axle", "ramp_axle", (rz["x"], yu, (zdu - 0.27) / 2), (0.965, 0.232, (zdu - 0.27) / 2), y_axle=yu))
    zb = _ramp_deck_z(rz, yu + 0.7) - 0.30
    out.append(_item("ramp_pump", "ramp_pump", (rz["x"] - rz["width"] / 2 - 0.17, yu + 0.767, zb + 0.40), (0.062, 0.147, 0.40)))
    out.append(_item("ramp_frame", "ramp_frame", (rz["x"], (rz["y_foot"] + rz["level_end"]) / 2, rz["deck"] / 2),
                     (rz["width"] / 2 + 0.105, (rz["y_foot"] - rz["level_end"]) / 2, rz["deck"] / 2), y_axle=yu, collide=False,
                     why="side channels, braces and cylinder under the ramp's sloping deck and curbs: no box fits them without "
                         "rising through the deck's surface"))
    o3, R3, bed = _yard_trailer_frame(L)
    Tp = P.TRAILER
    sub = Tp["subframe"]
    for sg in (-1, 1):
        a = np.array([rz["x"] + sg * (rz["width"] / 2 + 0.115), rz["y_rear"] + 0.50, bed - 0.10])     # on the side channel
        b = o3 + R3 @ np.array([-0.05, sg * (Tp["outside_W"] / 2 + 0.02), -0.12])                     # the sill's end
        out.append(_item(f"ramp_chain{'W' if sg < 0 else 'E'}", "chain", (a + b) / 2 - (0, 0, 0.05), np.abs(a - b) / 2 + (0.03, 0.03, 0.09),
                         a=tuple(a), b=tuple(b), sag=0.13, collide=False, why="a slack chain to the simulated trailer"))
        r = min(0.50, (bed - sub - 0.03) / 2)
        pc = o3 + R3 @ np.array([Tp["tandem_from_rear"] + 0.62 + 0.82 * r + 0.10, sg * (Tp["track"] / 2 + 0.165), 0.0])
        out.append(_item(f"chock{'W' if sg < 0 else 'E'}", "wheel_chock", (pc[0], pc[1], 0.09), (0.125, 0.10, 0.09),
                         yaw=L.zones["yard_trailer"]["yaw"] + pi))
        St = P.TRAILER_STAND
        ps = o3 + R3 @ np.array([P.TRAILER_INSIDE["L"] - St["from_nose"], sg * St["y"], 0.0])
        top = bed - sub                                          # the pad cranked up to the trailer's floor box
        out.append(_item(f"trailer_stand{'W' if sg < 0 else 'E'}", "trailer_stand", (ps[0], ps[1], top / 2), (0.22, 0.22, top / 2),
                         collide=False, why="a stabilizing jack under the nose: its support is dock.TrailerSupports' stand points "
                                            "(P.TRAILER_STAND), which push up on the trailer when it presses down on them"))
    for k, dx in enumerate((-2.3, 2.3)):                         # beside the ramp's foot, outside the lane's width
        out.append(_item(f"cone{k}", "cone", (rz["x"] + dx, rz["y_foot"] + 0.6, 0.232), (0.17, 0.17, 0.232)))
    # ---- what stands and hangs under the parked vans (their bodies and landing-gear legs are layout boxes)
    for bx, gear, i in _parked_trailers(L):
        D, origin, yaw, h = _parked_dims(bx)
        Rt = _rot(yaw=yaw)
        tx = (2.6, 3.3, 2.9)[i % 3]
        r = min(0.50, (h - D["sub"] - 0.03) / 2)
        gx = _parked_gear_x(bx, gear, D, origin)

        def part(tag, kind, x0, x1, hy, z0, z1, **extra):       # a box given in the van's frame
            c = np.asarray(origin) + Rt @ np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
            out.append(_item(f"{bx['name']}_{tag}", kind, c, ((x1 - x0) / 2, hy, (z1 - z0) / 2), yaw=yaw, box=bx["name"], **extra))
        part("tandem", "parked_tandem", min(tx - 1.655, tx - 0.62 - r - 0.16), tx + 2.445, Tp["track"] / 2 + 0.342, -h, -D["sub"],
             tandem=tx)
        part("rear_guard", "parked_rear_guard", D["icc_c"][0] - D["icc_h"][0] - 0.002, D["icc_c"][0] + 0.70,
             D["icc_h"][1], D["icc_c"][2] - D["icc_h"][2], -D["sub"])
        part("landing_gear", "parked_landing_gear", gx - 0.84, gx + 0.155, 0.9 * Tp["track"] / 2 + 0.435, -h, -D["sub"], collide=False,
             why="the legs stand inside the layout's %s_gear box; braces and crank are behind them under the body" % bx["name"])
        part("kingpin", "parked_kingpin", D["L"] - 1.65, D["L"] - 0.05, 0.575, -D["sub"] - 0.096, -D["sub"], collide=False,
             why="a plate and pin 10 cm under the body's box")
    # ---- perimeter: chain-link fence on the pavement's west, south and east edges, light poles inside it
    asp = next(s for s in L.surfaces if s["name"] == "asphalt")
    x0, x1 = asp["centre"][0] - asp["half"][0] + 0.25, asp["centre"][0] + asp["half"][0] - 0.25
    ys = asp["centre"][1] - asp["half"][1] + 0.25
    for nm, a, b, o_ in (("fence_west", (x0, yf), (x0, ys), (-1.0, 0.0)), ("fence_south", (x0, ys), (x1, ys), (0.0, -1.0)),
                         ("fence_east", (x1, ys), (x1, yf), (1.0, 0.0))):
        c = ((a[0] + b[0]) / 2 + 0.02 * o_[0], (a[1] + b[1]) / 2 + 0.02 * o_[1], 1.355)
        along = (abs(b[0] - a[0]) + abs(b[1] - a[1])) / 2 + 0.035
        out.append(_item(nm, "fence", c, (along, 0.125, 1.355) if o_[0] == 0 else (0.125, along, 1.355), a=a, b=b, out=o_, height=2.4))
    out.append(_item("sign_speed", "sign", (L.zones["ramp"]["x"] + 6.0, ys + 0.035, 1.55), (0.30, 0.004, 0.40), text="yard_speed",
                     collide=False, why="flat sign on the fence"))
    out.append(_item("sign_chock", "sign", (x1 - 0.035, ys + 14.0, 1.55), (0.40, 0.004, 0.30), yaw=pi / 2, text="chock_wheels",
                     collide=False, why="flat sign on the fence"))
    for k, (x, y, yaw) in enumerate(((x0 + 0.9, ys + 0.9, pi / 4), ((x0 + x1) / 2 + 3.0, ys + 0.9, pi / 2), (x1 - 0.9, ys + 0.9, 3 * pi / 4),
                                     (x1 - 0.9, (ys + yf) / 2 + 4.0, pi), (x0 + 0.9, (ys + yf) / 2 + 4.0, 0.0))):
        out.append(_item(f"light_pole{k}", "light_pole", (x, y, 4.5), (0.28, 0.28, 4.5), foot=(x, y)))
        c, s_ = math.cos(yaw), math.sin(yaw)
        out.append(_item(f"light_pole{k}_head", "light_pole_head", (x + 0.79 * c, y + 0.79 * s_, 9.08), (0.82, 0.15, 0.21), yaw=yaw,
                         foot=(x, y), collide=False, why="arm and lamp head 8.9 m up"))
    return out


_YARD_DRAW = dict(fence=_draw_fence, light_pole=_draw_pole, light_pole_head=_draw_pole, bollard=_draw_bollard,
                  downspout=_draw_downspout, wall_pack=_draw_wall_pack, seal_side=_draw_seal_side, dock_bumper=_draw_bumper,
                  stowed_lip=_draw_stowed_lip, building_sign=_draw_building_sign, wheel_chock=_draw_chock,
                  trailer_stand=_draw_stand, cone=_draw_cone, parked_tandem=_draw_parked_part,
                  parked_rear_guard=_draw_parked_part, parked_landing_gear=_draw_parked_part, parked_kingpin=_draw_parked_part,
                  ramp_axle=_draw_ramp_axle, ramp_pump=_draw_ramp_pump, ramp_frame=_draw_ramp_frame, chain=_draw_chain)


# =====================================================================================================================
# dressing and draw
# =====================================================================================================================
def _item(name, kind, p, half, yaw=0.0, collide=True, **extra):
    return dict(name=name, kind=kind, p=tuple(float(v) for v in p), yaw=float(yaw),
                half=tuple(float(v) for v in half), collide=bool(collide), **extra)


def _door_items(out, nm, num, x, y_in, w, h, with_trailer_state):
    """Dressing of one sectional door on the room side of a wall whose inner face is y_in (the room at -y)."""
    zt = _door_track_top(h)
    xt = w / 2 + DOOR_OVERLAP + 0.04
    pi = math.pi
    for sg in (-1, 1):
        side = "W" if sg < 0 else "E"
        out.append(_item(f"{nm}_track{side}", "door_track", (x + sg * (xt + 0.032), y_in - 0.04, zt / 2), (0.06, 0.04, zt / 2), yaw=pi,
                         x=x + sg * xt, y=y_in, h=h, side=sg, collide=False, why="8 cm proud of the wall, its foot inside the guard"))
        out.append(_item(f"{nm}_guard{side}", "track_guard", (x + sg * xt, y_in - 0.0715, 0.5), (0.0765, 0.0715, 0.5), yaw=pi,
                         x=x + sg * xt, y=y_in))
    out.append(_item(f"{nm}_shaft", "door_shaft", (x + 0.113, y_in - 0.0925, zt + 0.01), (xt + 0.188, 0.0925, 0.21), yaw=pi,
                     x=x, y=y_in, w=w, h=h, collide=False, why="spring shaft and operator 6.2 m up the wall"))
    leaf = dict(note="the box is the leaf without its rubber bottom seal (the lowest 3 cm), so that it does not rest on the floor "
                     "or on a stowed leveler's deck")
    if with_trailer_state:
        # the door's number on the leaf too: raised, the leaf covers the wall placard above the opening, and the work
        # orders name the door (load-bales has trailers at two doors)
        leaf.update(door=nm, state="closed in the opening; at a door where a trailer stands the leaf is raised flat against the "
                                   "wall above the opening: its box is then at p_raised, never in the doorway (see colliders())",
                    p_raised=(float(x), float(y_in - 0.04), float(h + 0.04 + (h + 0.03) / 2)), number=num)
    else:
        leaf.update(state="closed")
    out.append(_item(f"{nm}_panel", "door_panel", (x, y_in - 0.04, (h + 0.03) / 2), (w / 2 + DOOR_OVERLAP + 0.02, 0.04, (h - 0.03) / 2), yaw=pi,
                     x=x, y=y_in, w=w, h=h, **leaf))
    out.append(_item(f"{nm}_number", "sign", (x, y_in - 0.004, h + 0.50), (0.24, 0.004, 0.24), yaw=pi, text=f"number_{num}",
                     collide=False, why="flat sign"))


def dressing(layout, preset, seed):
    """Every render-only object (no collider) this module draws: dict(name, kind, p, yaw, half, collide[, ...]) with p
    the centre of its bounding box, half its half-extents in metres in its own frame and yaw that frame's rotation
    about z (a wall-mounted item's local +y points out of its wall). Depends on the layout alone; draw() draws
    exactly this list, each item inside its box. collide=False ('why' says so) marks what should not be given a
    collider: flat signs, fabric, chains, fittings high on a wall, and parts that a box cannot stand in for. Where an
    item's look, or its place, follows the task's state, 'state' says how (a door leaf is raised where a trailer
    stands: its box is the closed leaf)."""
    L = layout
    out = []
    if L is None:
        return out
    pi = math.pi
    if preset == "dock":
        from ..facility import DOOR_H, DOOR_W
        wall = next(b for b in L.boxes if b["name"].startswith("dockwall"))
        y_in, y_out = wall["centre"][1] - wall["half"][1], wall["centre"][1] + wall["half"][1]
        z_run = 7.3
        risers = []
        for d in sorted(L.zones["doors"].values(), key=lambda d: d["x"]):
            nm, x = d["name"], d["x"]
            num = int("".join(ch for ch in nm if ch.isdigit()) or 0)
            _door_items(out, nm, num, x, y_in, DOOR_W, DOOR_H, True)
            out.append(_item(f"{nm}_control", "leveler_control", (x + DOOR_W / 2 + 0.34, y_in - 0.073, 1.30), (0.11, 0.073, 0.16),
                             yaw=pi))
            out.append(_item(f"{nm}_signal", "restraint_signal", (x - DOOR_W / 2 - 0.30, y_in - 0.095, 1.85),
                             (0.085, 0.095, 0.18), yaw=pi, door=nm, side="in",
                             state="green when the door's trailer is held by the restraint, else red"))
            risers += [(x + DOOR_W / 2 + 0.34, 1.46), (x - DOOR_W / 2 - 0.30, 2.03)]
            out.append(_item(f"{nm}_sign", "sign", (x - DOOR_W / 2 - 0.30, y_in - 0.004, 2.26), (0.16, 0.004, 0.10),
                             yaw=pi, text="enter_on_green", collide=False, why="flat sign"))
            src, aim = dock_light_source(x)
            lo = np.minimum(src - 0.13, [src[0] - 0.16, src[1] - 0.22, src[2] - 0.17])
            hi = np.maximum(src + 0.13, [src[0] + 0.53, y_in, src[2] + 0.14])
            out.append(_item(f"{nm}_light", "dock_light", (lo + hi) / 2, (hi - lo) / 2, x=x, y=y_in, collide=False,
                             why="a swing arm that yields when struck, its head 2.5 m up"))
            out.append(_item(f"{nm}_seal_head", "seal_head", (x, y_out + 0.15, DOOR_H + 0.15), (DOOR_W / 2 + 0.30, 0.15, 0.15),
                             collide=False, why="foam over the rear of a docked trailer (9 cm above its header): rigid, it would "
                                                "clamp a trailer that rises on its springs as it is unloaded; the layout's own seal "
                                                "side pads are collide=False for the same reason"))
            out.append(_item(f"{nm}_seal_curtain", "seal_curtain", (x, y_out + 0.296, DOOR_H - 0.225), (DOOR_W / 2 - 0.01, 0.005, 0.225),
                             door=nm, collide=False, why="hanging fabric",
                             state="hangs 0.45 m; at a door with a trailer it is lifted onto the trailer's roof"))
            out.append(_item(f"{nm}_number_out", "sign", (x, y_out + 0.004, DOOR_H + 0.85), (0.36, 0.004, 0.36),
                             text=f"number_{num}", collide=False, why="flat sign"))
            out.append(_item(f"{nm}_signal_out", "restraint_signal", (x - DOOR_W / 2 - 0.457, y_out + 0.095, 2.55),
                             (0.085, 0.095, 0.18), door=nm, side="out",
                             state="red when the door's trailer is held by the restraint, else green"))
            out.append(_item(f"{nm}_sign_out", "sign", (x - DOOR_W / 2 - 0.457, y_out + 0.004, 3.05), (0.15, 0.004, 0.19),
                             text="move_on_green", collide=False, why="flat sign"))
        x0, x1 = L.extent[0], L.extent[1]
        za = min(z for _, z in risers)
        out.append(_item("conduit_run", "conduit", ((x0 + x1) / 2, y_in - 0.0225, (za + z_run + 0.0175) / 2),
                         ((x1 - x0) / 2 - 0.3, 0.0225, (z_run + 0.0175 - za) / 2), z_run=z_run, risers=risers, collide=False,
                         why="conduit 4 cm proud of the wall"))
        # the dock's posted speed limit (the evaluator's speed_limit: posted 8 km/h + 10 %), as the warehouse and the yard
        # post theirs: on the east wall facing the bale row and the route to the doors, above the bin and drums there
        out.append(_item("sign_speed_dock", "sign", (x1 - 0.004, 12.3, 2.2), (0.30, 0.004, 0.40), yaw=pi / 2,
                         text="dock_speed", collide=False, why="flat sign"))
    elif preset == "warehouse":
        from ..facility import DOOR_H, DOOR_W
        for k, pr in enumerate(pp for pp in L.props if pp.get("kind") == "dock_door_closed"):
            _door_items(out, pr["name"], 21 + k, pr["p"][0], pr["p"][1], DOOR_W, DOOR_H, False)
    elif preset == "yard":
        out += _dressing_yard(L, seed)
    return out


def colliders(layout, preset, seed, doors_with_trailer=()):
    """The boxes to give static colliders to: [dict(name, p, half, yaw)] of every dressing item with collide=True, in
    world coordinates (yaw about z). doors_with_trailer: names of the dock doors at which a trailer stands in this
    task (scene.trailers): their door leaves are raised, so their boxes are the raised leaves above the openings."""
    out = []
    for it in dressing(layout, preset, seed):
        if not it["collide"]:
            continue
        p = it["p_raised"] if it.get("p_raised") and it.get("door") in doors_with_trailer else it["p"]
        out.append(dict(name=it["name"], p=tuple(p), half=it["half"], yaw=it["yaw"]))
    return out


_SIGNS = {"enter_on_green": ([("ENTER ON", 1.0), ("GREEN ONLY", 1.0)], (512, 320), (16, 16, 18), (238, 238, 232), (20, 120, 50)),
          "move_on_green": ([("MOVE ON", 1.0), ("GREEN ONLY", 1.0)], (512, 320), (16, 16, 18), (238, 238, 232), (20, 120, 50)),
          "yard_speed": ([("YARD", 0.6), ("SPEED", 0.6), ("5", 1.7), ("MPH", 0.5)], (384, 512), (16, 16, 18), (238, 238, 232),
                         (16, 16, 18)),
          "dock_speed": ([("SPEED", 0.6), ("LIMIT", 0.6), ("8", 1.7), ("km/h", 0.5)], (384, 512), (16, 16, 18),
                         (238, 238, 232), (16, 16, 18)),
          "chock_wheels": ([("DRIVERS", 0.7), ("CHOCK", 1.0), ("WHEELS", 1.0)], (512, 384), (16, 16, 18), (226, 172, 12),
                           (16, 16, 18))}


def _draw_sign(S, B, it):
    """A flat sign on a wall: plate and printed face (the wall behind at local y = 0, the face toward +y)."""
    Bw = _wall_frame(B, it)
    hx, hy, hz = it["half"]
    text = it["text"]
    Bw.box("dy_sign_plate", (0.0, hy - 0.0004, 0.0), (2 * hx, 2 * hy - 0.0008, 2 * hz), bevel=0.001)
    yf = 2 * hy - 0.0002
    if text.startswith("number_"):
        img = _sign(S, f"dy_{text}.png", [(text.split("_")[1], 1.0)], (256, 256), (245, 245, 240), (22, 58, 122),
                    border=(245, 245, 240))
        Bw.rect(_image_mat(S, f"dy_sign_{text}", img), (0.0, yf, 0.0), -_X, _Z, (2 * hx - 0.01, 2 * hz - 0.01), uv01=True)
    elif text == "move_on_green":                     # read directly and, mirrored, in the driver's mirror
        lines, size, fg, bg, border = _SIGNS[text]
        for dz, mirror in ((hz / 2, True), (-hz / 2, False)):
            img = _sign(S, f"dy_{text}_{int(mirror)}.png", lines, size, fg, bg, border, mirror=mirror)
            Bw.rect(_image_mat(S, f"dy_sign_{text}_{int(mirror)}", img), (0.0, yf, dz), -_X, _Z, (2 * hx - 0.01, hz - 0.01),
                    uv01=True)
    else:
        lines, size, fg, bg, border = _SIGNS[text]
        img = _sign(S, f"dy_{text}.png", lines, size, fg, bg, border)
        Bw.rect(_image_mat(S, f"dy_sign_{text}", img), (0.0, yf, 0.0), -_X, _Z, (2 * hx - 0.01, 2 * hz - 0.01), uv01=True)


def _draw_item(S, world, B, it, shapes=None):
    """Draw one dressing item into B (world coordinates), inside its box."""
    sc = world.scene
    k = it["kind"]
    p, half = np.array(it["p"]), np.array(it["half"])
    tr = sc.trailers.get(it.get("door")) if it.get("door") else None
    if k == "door_track":
        _door_track(B.at((it["x"], it["y"], 0.0), yaw=it["yaw"]), it["h"], -it["side"])
    elif k == "door_shaft":
        _door_shaft(B.at((it["x"], it["y"], 0.0), yaw=it["yaw"]), it["w"], it["h"])
    elif k == "door_panel":
        _sectional_door(B.at((it["x"], it["y"], 0.0), yaw=it["yaw"]), it["w"], it["h"], up=tr is not None, S=S,
                        number=it.get("number"))
    elif k == "track_guard":
        _track_guard(B.at((it["x"], it["y"], 0.0), yaw=it["yaw"]))
    elif k == "leveler_control":
        _control_box(_wall_frame(B, it))
    elif k == "restraint_signal":
        held = tr is not None and bool(tr.get("restraint"))
        _signal_box(_wall_frame(B, it), ("green" if held else "red") if it.get("side") == "in" else ("red" if held else "green"))
    elif k == "sign":
        _draw_sign(S, B, it)
    elif k == "dock_light":
        src, aim = dock_light_source(it["x"])
        _dock_light(B, src, aim, it["y"])
    elif k == "seal_head":
        _seal_head(_wall_frame(B, it).at((0.0, 0.0, -half[2])), 2 * half[0], 2 * half[1])
    elif k == "seal_curtain":
        drop = 2 * half[2]
        if tr is not None:                             # lifted by the trailer: down to just above its rear header
            D = _trailer_dims((shapes or _shapes(world))[tr["name"]])
            drop = max(0.02, p[2] + half[2] - (tr["bed_z"] + D["H"] + D["hdr_h"]) - 0.015)
        _seal_curtain(B.at((p[0], p[1], p[2] + half[2]), yaw=it["yaw"]), 2 * half[0], drop)
    elif k == "conduit":
        y = p[1] + half[1] - 0.012
        B.cyl("dy_steel_galv", (p[0] - half[0], y, it["z_run"]), (p[0] + half[0], y, it["z_run"]), 0.011, n=8)
        for x in np.arange(p[0] - half[0] + 0.5, p[0] + half[0], 1.5):
            B.box("dy_steel_galv", (x, y - 0.003, it["z_run"]), (0.03, 0.03, 0.035))
        for x, z in it["risers"]:
            B.cyl("dy_steel_galv", (x, y, z), (x, y, it["z_run"]), 0.011, n=8)
    elif k in _YARD_DRAW:
        _YARD_DRAW[k](S, world, B, it)
    else:
        raise KeyError(f"no drawer for dressing kind {k!r}")


def _draw_dressing(S, world, B, items):
    """Draw the dressing list (static, world coordinates)."""
    shapes = _shapes(world)
    for it in items:
        _draw_item(S, world, B, it, shapes)


def _dock_chocks(world, B):
    """Wheel chocks at the tandems of the docked trailers (dock.TrailerSupports holds them there), on the apron."""
    Tp = P.TRAILER
    for tr in world.scene.trailers.values():
        if not tr.get("chocks") or world.scene.c.preset == "yard":
            continue
        h = tr["bed_z"] - tr["ground_z"]
        r = min(0.50, (h - Tp["subframe"] - 0.03) / 2)
        R = _rot(yaw=tr["yaw"])
        for sg in (-1, 1):
            pc = np.array([tr["rear"][0], tr["rear"][1], 0.0]) + R @ np.array([Tp["tandem_from_rear"] + 0.62 + 0.82 * r + 0.10,
                                                                             sg * (Tp["track"] / 2 + 0.165), 0.0])
            B.inst("chock", (pc[0], pc[1], tr["ground_z"]), _rot(yaw=tr["yaw"] + math.pi))


def draw(S, world):
    sc = world.scene
    L = sc.layout
    if L is None:
        return draw_lab(S, world)
    preset = sc.c.preset
    shapes = _shapes(world)
    B = _Build()
    if preset == "dock":
        _dock_static(S, world, B)
    elif preset == "yard":
        _yard_static(S, world, B)
    _dock_chocks(world, B)
    B.flush(S, "/World/dockyard/static")
    B = _Build()
    _draw_dressing(S, world, B, dressing(L, preset, sc.c.seed))
    B.flush(S, "/World/dockyard/dressing")
    _draw_trailers(S, world, shapes)
    _leveler_bodies(S, world, shapes)
    _ramp_lip_bodies(S, world, shapes)

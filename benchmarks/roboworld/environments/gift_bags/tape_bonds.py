"""Tape bonds at quadrature points inside the tape's triangles.

The contact world's `ipcworld.Bonds` ties each tape VERTEX to the paper with the cohesive law, so a 2.5 mm tape row
lets go at once when the peel front reaches it. The peel process zone of 19 mm Magic tape on gift wrap is ~0.6 mm long
(a bending-dominated zone, l ~ (8 EI Gamma / sigma^2)^(1/4) with EI = 1.3e-5 N m, Gamma = 150 J/m^2, sigma = 0.40 MPa),
shorter than a row, so each row's release is a jump. Lowering the strength until the zone spans three rows (Turon et
al. 2007) would need sigma ~ 2 kPa, 180x below the measured tack, and the tape would no longer hold in shear: not used.

Here the bond points are the centroids of an m x m subdivision of every tape triangle (barycentric on the tape
triangle), each tied to the closest material point of the paper (barycentric on a paper triangle) with the same
bilinear law per area (`params` CohesiveLaw: stiffness k, strength sigma, energy Gamma). The front then passes through
m points per row, each releasing 1/m^2 of a triangle's energy. Energy, gradient and Hessian are exact for the bond
energy; damage is irreversible and its dissipation booked (Gamma per area at full damage), as in `ipcworld.Bonds`,
whose interface (`D`, `dissipated`, `intact_fraction`, `begin`/`term`/`after`) this class keeps."""
import numpy as np
from scipy.spatial import cKDTree


def _closest_bary(p, a, b, c):
    from .ipcworld import _closest_bary as cb
    return cb(p, a, b, c)


def subdivision_points(m):
    """Barycentric centroids and area fractions of the m x m subdivision of a triangle."""
    pts = []
    for i in range(m):
        for j in range(m - i):
            pts.append(((i + 1/3)/m, (j + 1/3)/m))                          # upward sub-triangles
            if i + j < m - 1:
                pts.append(((i + 2/3)/m, (j + 2/3)/m))                      # downward ones
    uv = np.asarray(pts)
    bary = np.c_[1 - uv.sum(1), uv[:, 0], uv[:, 1]]
    return bary, np.full(len(bary), 1./m**2)


class QuadBonds:
    """Cohesive bonds between quadrature points of tape triangles and material points of paper triangles."""

    def __init__(self, world, ttri, tbary, ptri, pbary, area, law, d0):
        self.w = world
        self.ttri = np.asarray(ttri, np.int64)          # tape triangle (global triangle index)
        self.tbary = np.asarray(tbary, float)
        self.tri = np.asarray(ptri, np.int64)           # paper triangle
        self.bary = np.asarray(pbary, float)
        self.area = np.asarray(area, float)
        self.law = law
        n = len(self.ttri)
        self.d0 = np.broadcast_to(np.asarray(d0, float), (n,)).copy()   # rest offset along the paper normal, per bond
        self.D = np.zeros(n)
        self.dmax = np.zeros(n)
        self.n = np.zeros((n, 3))
        self.dissipated = 0.

    def _pts(self, x):
        Tt = self.w.tris[self.ttri]
        Tp = self.w.tris[self.tri]
        return (self.tbary[:, :, None]*x[Tt]).sum(1), (self.bary[:, :, None]*x[Tp]).sum(1), Tt, Tp

    def begin(self, w, dt):
        x = w.x_free()
        pt, pp, Tt, Tp = self._pts(x)
        nn = np.cross(x[Tp[:, 1]] - x[Tp[:, 0]], x[Tp[:, 2]] - x[Tp[:, 0]])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-30)
        s = np.sign(((pt - pp)*nn).sum(1)); s[s == 0] = 1.
        self.n = nn*s[:, None]

    def _delta(self, x):
        pt, pp, _, _ = self._pts(x)
        return pt - pp - self.d0[:, None]*self.n

    def term(self, w, q, xc_prev, dt, want):
        g = np.zeros(w.n_dof)
        coo = ([], [], [])
        live = self.D < 1.
        if not live.any():
            return 0., g, coo
        x = q[:3*w.n_free].reshape(-1, 3)
        idx = np.flatnonzero(live)
        Tt, Tp = w.tris[self.ttri[idx]], w.tris[self.tri[idx]]
        dl = (self.tbary[idx, :, None]*x[Tt]).sum(1) - (self.bary[idx, :, None]*x[Tp]).sum(1) - self.d0[idx, None]*self.n[idx]
        k = (1. - self.D[idx])*self.law.k*self.area[idx]
        E = .5*(k*(dl*dl).sum(1)).sum()
        f = k[:, None]*dl
        # scatter (tape corner c, paper corner c, c = 0..2), one bincount pass per component: each vertex sums its
        # contributions in a fixed order
        ii = np.concatenate([Tt[:, 0], Tp[:, 0], Tt[:, 1], Tp[:, 1], Tt[:, 2], Tp[:, 2]])
        ff = np.concatenate([p_ for c in range(3) for p_ in (self.tbary[idx, c:c+1]*f, -self.bary[idx, c:c+1]*f)])
        gg = g[:3*w.n_free].reshape(-1, 3)
        for d in range(3):
            gg[:, d] = np.bincount(ii, weights=ff[:, d], minlength=w.n_free)
        if want:
            ids = np.concatenate([Tt, Tp], 1)                                        # (B, 6)
            cw = np.concatenate([self.tbary[idx], -self.bary[idx]], 1)                # (B, 6)
            # one COO chunk ordered by (a, b, component); its index arrays are cached while
            # the bonded triangles are unchanged (the same array objects: the Newton assembly reuses its scatter map)
            c_ = getattr(self, "_coo", None)
            if c_ is None or not np.array_equal(c_[0], ids):
                sh = (6, 6, 3, len(ids))
                d3 = np.arange(3)[None, None, :, None]
                c_ = self._coo = [ids.copy(), np.broadcast_to(3*ids.T[:, None, None, :] + d3, sh).ravel(),
                                  np.broadcast_to(3*ids.T[None, :, None, :] + d3, sh).ravel(), None, None]
            if c_[3] is None or not np.array_equal(c_[3], k):
                # the values change only when damage grows (after a step): the same array object is handed out until
                # then (never modified in place), so the Newton assembly can reuse its scattered sum
                v = (k[None, None, :]*cw.T[:, None, :])*cw.T[None, :, :]              # (6, 6, B): k cw_a cw_b
                c_[3], c_[4] = k.copy(), np.broadcast_to(v[:, :, None, :], (6, 6, 3, len(ids))).ravel()
            coo[0].append(c_[1]); coo[1].append(c_[2]); coo[2].append(c_[4])
        return E, g, coo

    def after(self, w, dt):
        x = w.x_free()
        op = np.linalg.norm(self._delta(x), axis=1)
        self.dmax = np.maximum(self.dmax, op)
        d0, df = self.law.delta0, self.law.deltaf
        Dn = np.where(self.dmax <= d0, 0., np.where(self.dmax >= df, 1.,
                      df*(self.dmax - d0)/np.maximum(self.dmax*(df - d0), 1e-30)))
        newD = np.maximum(self.D, np.clip(Dn, 0., 1.))
        self.dissipated += float((self.area*self.law.Gamma*(newD - self.D)).sum())
        self.D = newD

    @property
    def intact_fraction(self):
        return float((self.D < 1.).mean()) if len(self.D) else 0.

    @property
    def vi(self):                                        # tape vertices touched (for code reading ipcworld.Bonds.vi)
        return np.unique(self.w.tris[self.ttri])


def bond_tape_to_sheet(world_ipc, tape_slice, target_sheet_slice, law, m=4, reach=6e-4, keep=None, d0=None, touch=None):
    """Quadrature bonds of every tape triangle (m x m subdivision centroids) to the closest point of the target sheet
    within `reach`. keep(point) -> bool restricts which quadrature points are bonded (e.g. a laid-on tail).
    With `d0` and `touch` (as laid): a point is bonded only where the tape touches the paper, i.e. its separation along
    the paper's normal is within `touch` of the contact rest offset d0 and it lies over the triangle (tangential offset
    <= touch); the sixth list returned is each bond's own separation, used as its rest offset, so every bond starts
    unloaded. Without them (the start-state relaxation): every point within `reach`, rest offset d0 for all."""
    x = world_ipc.x_free()
    T = world_ipc.tris
    tape_t = np.flatnonzero((T[:, 0] >= tape_slice.start) & (T[:, 0] < tape_slice.stop))
    targ_t = np.flatnonzero((T[:, 0] >= target_sheet_slice.start) & (T[:, 0] < target_sheet_slice.stop))
    cen = x[T[targ_t]].mean(1)
    kd = cKDTree(cen)
    sb, frac = subdivision_points(m)
    tt, tb, pt, pb, ar, rest = [], [], [], [], [], []
    for t in tape_t:
        P = x[T[t]]
        A = .5*np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        for b, f in zip(sb, frac):
            p = b @ P
            if keep is not None and not keep(p):
                continue
            best = None
            for c in kd.query_ball_point(p, 2e-2):
                a_, b_, c_ = x[T[targ_t[c]]]
                wb = _closest_bary(p, a_, b_, c_)
                d = np.linalg.norm(p - (wb[0]*a_ + wb[1]*b_ + wb[2]*c_))
                if best is None or d < best[0]:
                    best = (d, targ_t[c], wb)
            if not best or best[0] >= reach:
                continue
            if touch is not None:
                a_, b_, c_ = x[T[best[1]]]
                n = np.cross(b_ - a_, c_ - a_); n /= max(np.linalg.norm(n), 1e-30)
                r = p - (best[2][0]*a_ + best[2][1]*b_ + best[2][2]*c_)
                sn = float(r @ n)
                if abs(abs(sn) - d0) > touch or np.linalg.norm(r - sn*n) > touch:
                    continue
                rest.append(abs(sn))
            tt.append(t); tb.append(b); pt.append(best[1]); pb.append(best[2]); ar.append(A*f)
    return tt, tb, pt, pb, ar, rest


def bond_tapes(world_ipc, tape_slices, target_slice, law, d0, as_laid=False):
    """Bond each tape sheet to the target sheet with the configured scheme (params.TAPE_QUAD) and install the bonds.
    Returns the bond objects (ipcworld.Bonds or QuadBonds; same interface).
    as_laid (the task's start state; quadrature bonds only): a pressure-sensitive tape bonds where it touches and
    starts unloaded. A point is bonded only if it lies within the bond law's elastic range (delta0, 27 um for Magic
    tape on wrap) of the contact rest offset, and its own separation is its rest offset."""
    from . import ipcworld as W, params as Pm
    out = []
    for sl in tape_slices:
        if Pm.TAPE_QUAD and as_laid:
            tt, tb, pt, pb, ar, rest = bond_tape_to_sheet(world_ipc, sl, target_slice, law, m=Pm.TAPE_QUAD, d0=d0,
                                                          touch=law.delta0)
            b = QuadBonds(world_ipc, tt, tb, pt, pb, ar, law, d0=np.asarray(rest) if len(rest) else d0)
            T = world_ipc.tris
            n_q = int(((T[:, 0] >= sl.start) & (T[:, 0] < sl.stop)).sum())*Pm.TAPE_QUAD**2
            b.bonded_fraction = len(tt)/max(n_q, 1)       # share of the tape's quadrature points that touch the paper
        elif Pm.TAPE_QUAD:
            b = QuadBonds(world_ipc, *bond_tape_to_sheet(world_ipc, sl, target_slice, law, m=Pm.TAPE_QUAD)[:5], law, d0=d0)
        else:
            vi, ti, bw, ar = W.bond_sheet_to_sheet(world_ipc, sl, target_slice, law)
            b = W.Bonds(world_ipc, vi, ti, bw, ar, law, d0=d0)
        W.install_bonds(world_ipc, b)
        out.append(b)
    return out

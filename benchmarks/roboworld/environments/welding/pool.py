"""Filler deposition and the weld-pool free surface.

The bead is the liquid metal held by surface tension against gravity and the arc's pressure, frozen when the
solidification front passes. Per x slice (a y-z cross-section of the grid):

* toes: the contact line's two ends on the joint's original surface (below: contact line);
* free surface: a height function h(u) over the chord between the toes (u along the chord, h along its outward
  normal), solving Young-Laplace with hydrostatic and arc pressure:
      gamma kappa = P_k + rho g.r - p_arc(r),   kappa = -h'' / (1 + h'^2)^(3/2),   h(0) = h(L) = 0,
  by Newton iterations (batched over slices); h may not enter unmelted metal (active set);
* volume: all liquid slices share one pool pressure P_pool (P_k = P_pool + rho g_x x_k, the pool's own weight
  along the weld). P_pool is found by bisection so that the liquid slices hold the deposited filler not yet frozen
  (volume conservation); the liquid holds 1.03 x its metal (solidification shrinkage 3 %, TWI Job Knowledge 33), a slice
  that freezes keeps its profile and its metal leaves the inventory, and the remaining liquid feeds its shrinkage, so
  the last metal to freeze at the end of a weld is short of metal: the crater;
* contact line: the toes are the outer ends of the joint's original surface that has fused or that liquid metal
  connected to the pool lies on (a remelted tack's lap beside the fused toe joins the pool); the edge advances onto
  more metal only where the liquid would not freeze at the contact (_contact_freezes), and never recedes while liquid;
* overflow: filler the slice cannot hold under a face of at most 88 deg to the chord either spreads (where the edge can
  advance) or, where the edge is pinned on cold metal, stands as a convex bulge whose face overhangs its toe (a
  surface in polar form about a centre below the chord's midpoint, volume-controlled: _PolarSurface); where the bulge
  comes down on
  the metal beyond the toe the liquid rolls onto it (the contact line moves there, frozen at the contact: cold lap);
* undercut emerges where the equilibrium surface lies below the original surface in melted metal: those cells become
  air (a groove), their metal stays in the pool's volume.

Not represented: Marangoni-driven surface shape, humping, crater dynamics of short circuits, the
three-dimensional curvature along the weld (the slices are 2-D).
"""

import math

from scipy import ndimage

import numpy as np

from . import params as PR
from . import thermal as TH

N_NODES = 24            # chord intervals per slice profile [numerical]
# the pool's outcome-deciding constants live in the ledger (params.py, "weld pool"); the
# module names stay so a probe can set them
MAX_END_SLOPE = math.tan(math.radians(PR.v(PR.POOL_TOE_SLOPE_MAX)))
CAP_MODEL = PR.v(PR.POOL_CAP_MODEL)
SHRINK = PR.v(PR.POOL_SOLID_SHRINK)
BRIDGE_SLICES = PR.v(PR.POOL_BRIDGE_SLICES)
LONG_SIGMA = PR.v(PR.POOL_LONG_SIGMA)
TOE_SAMPLE = PR.v(PR.POOL_TOE_SAMPLE)
SPREAD_MAX_PER_UPDATE = PR.v(PR.POOL_SPREAD_MAX)
STACK_REACH = PR.v(PR.POOL_STACK_REACH)
# A slice's free surface moves only where the metal under it is liquid: where the outermost metal at a chord station has
# solidified, the surface is held there (an upper bound equal to the lower one). A slice whose whole surface stays free
# until every cell of it has frozen lets a solidifying tail keep trading metal hydrostatically with a 20-25 mm pool, and
# the metal drifts with it (E3).
FREEZE_SURFACE = True
FAN_REACH = PR.v(PR.POOL_FAN_REACH)


class Joint:
    """Joint cross-section in the weld frame (x along the weld). `surface`: the original free-surface polyline in the
    y-z plane, ordered from toe side A to toe side B, with `root_s` the arclength of the root. `box`: the y-z region
    (y0, y1, z0, z1) the pool model may change."""

    def __init__(self, kind, surface, root_s, box):
        self.kind = kind
        self.surface = np.asarray(surface, float)
        seg = np.diff(self.surface, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.root_s = root_s
        self.box = box
        # outward normal of each segment: the side away from metal is decided per joint kind below
        t = seg/self.seg_len[:, None]
        self.tangent = t
        self.normal = np.stack([-t[:, 1], t[:, 0]], 1)    # left normal; constructors order points so that it is outward

    def point(self, s):
        s = np.clip(s, 0., self.length)
        k = np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.seg_len) - 1)
        f = (s - self.cum[k])/self.seg_len[k]
        p = self.surface[k] + (self.surface[k + 1] - self.surface[k])*f[..., None]
        return p, self.normal[k]

    @staticmethod
    def tee_near_side(y_reach, z_reach, box):
        """Horizontal fillet T-joint, near side: base plate top z = 0 (metal below; the weld side is y > 0) and the
        bracket's face y = 0 (metal at y < 0). The surface runs from (0, z_reach) down the bracket face to the root
        (0, 0) and out along the base to (y_reach, 0); each segment's left normal then points into air."""
        return Joint("tee", [(0., z_reach), (0., 0.), (y_reach, 0.)], z_reach, box)

    @staticmethod
    def plate(y0, y1, box):
        """Bead on a flat plate: top surface z = 0 from y0 to y1 (metal below; left normal +z into air). The root is
        the arc root's y, given to Pool.update."""
        return Joint("plate", [(y0, 0.), (y1, 0.)], -y0, box)


# ---------------------------------------------------------------- surface geometry (both representations)
def polar_dirs(t, n, N=N_NODES, phi0=0., phi1=math.pi):
    """Node angles phi_i from phi0 (toward toe B) to phi1 (toward toe A), measured from the chord's direction t (toward
    toe B) through its outward normal n, and their unit directions [N+1, 2]."""
    phi = np.linspace(phi0, phi1, N + 1)
    return phi, np.cos(phi)[:, None]*np.asarray(t)[None, :] + np.sin(phi)[:, None]*np.asarray(n)[None, :]


def polar_frame(A, t, n, L, depth):
    """The polar surface's centre and angle range for a chord A + [0, L] t with outward normal n: the centre on the
    chord's perpendicular bisector, `depth` below the chord (inside the bead's core, below the deepest point of an arc
    crater), and the angles of toe B (phi0) and toe A (phi1) seen from it."""
    C = np.asarray(A, float) + .5*L*np.asarray(t, float) - depth*np.asarray(n, float)
    a = math.atan2(depth, .5*L)
    return C, a, math.pi - a


def surface_points(st):
    """A slice's free surface as (y, z) points from toe A to toe B (weld frame), for either representation; None
    without a profile. Overhanging profiles go back past their toes: neither y nor the chord coordinate is monotone."""
    if getattr(st, "pts", None) is not None:
        return st.pts
    if st.h is None or st.u is None or st.A is None or st.chord_t is None:
        return None
    return np.asarray(st.A)[None, :] + np.asarray(st.u)[:, None]*np.asarray(st.chord_t)[None, :] \
        + np.asarray(st.h)[:, None]*np.asarray(st.chord_n)[None, :]


def _polar_edge(r, phi_q, phi0=0., phi1=math.pi):
    """Distance from the centre, along the angle phi_q, to the polygon through the polar nodes r (equal angles over
    [phi0, phi1]): the straight edge between the two nodes around phi_q (not r interpolated in phi: the polygon is
    what the area constraint and the cells use)."""
    N = len(r) - 1
    dphi = (phi1 - phi0)/N
    q = phi_q - phi0
    i = np.clip((q/dphi).astype(int), 0, N - 1)
    f = np.clip(q - i*dphi, 0., dphi)
    ri, rj = r[i], r[i + 1]
    den = ri*np.sin(f) + rj*np.sin(dphi - f)
    return ri*rj*math.sin(dphi)/np.maximum(den, 1e-30)


def under_surface(kind, A, t, n, L, Y, Z, u=None, h=None, r=None, depth=0.):
    """Points (Y, Z arrays) under a slice's free surface: on the metal side of it, between the surface and what the
    liquid rests on (the region a profile makes weld metal where it is air). kind 'h': height function h(u) over the
    chord (u in [0, L] along t from toe A, h along the outward normal n); 'polar': radius r(phi) about the chord's
    midpoint over phi in [0, pi] (the chord's metal side within its span stays under). Returns (under, span, ww): the
    span mask (0 <= u <= L) and the height above the chord line."""
    A, t, n = (np.asarray(v, float) for v in (A, t, n))
    uu = (Y - A[0])*t[0] + (Z - A[1])*t[1]
    ww = (Y - A[0])*n[0] + (Z - A[1])*n[1]
    span = (uu >= 0) & (uu <= L)
    if kind == "polar":
        # about the centre `depth` below the chord's midpoint: inside the fan of angles between the toes, under the
        # polygon; below the chord within the span but outside the fan (the joint's core beside the centre), under
        _, phi0, phi1 = polar_frame(A, t, n, L, depth)
        du_, dw_ = uu - .5*L, ww + depth
        rho = np.hypot(du_, dw_)
        phi = np.arctan2(dw_, du_)
        phi = np.where(phi < -.5*math.pi, phi + 2*math.pi, phi)
        fan = (phi >= phi0) & (phi <= phi1)
        und = (fan & (rho <= _polar_edge(np.asarray(r, float), np.clip(phi, phi0, phi1), phi0, phi1))) \
            | (~fan & span & (ww < 0))
        return und, span, ww
    hu = np.interp(np.clip(uu, 0, L), u, h)
    return span & (ww <= hu), span, ww


def slice_under(st, Y, Z):
    """under_surface for a pool.SliceState (or a dict with the same fields)."""
    get = (lambda k: st.get(k)) if isinstance(st, dict) else (lambda k: getattr(st, k, None))
    kind = get("kind") or "h"
    A, t, n = get("A"), get("chord_t"), get("chord_n")
    L = float(np.linalg.norm(np.asarray(get("B"), float) - np.asarray(A, float)))
    return under_surface(kind, A, t, n, L, Y, Z, u=get("u"), h=get("h"), r=get("r"), depth=get("depth") or 0.)


class _PolarSurface:
    """A slice's free surface that may overhang its toes: r(phi) about the chord's midpoint M, phi in [0, pi] from the
    chord's direction t (toe B at phi = 0, r = L/2) through the outward normal to toe A (phi = pi). The surface is the
    polygon through the nodes M + r_i e(phi_i). Its energy per unit length of weld is
        E = gamma * perimeter + integral over the region of (rho_l (-g . x) + p_arc(x)) dA,
    minimised at a fixed area (the slice's share of the pool: volume control, which, unlike a fixed pressure, is stable
    past the half-circle, where a pinned drop's pressure falls as it grows) with each radius between the solid the
    liquid rests on and the metal beyond (lo <= r <= hi), stations of resolidified metal held. Projected Newton on the
    KKT system (tridiagonal + one border), feasibility restored along the area's gradient after each step. The
    region terms use the polygon's triangle fan: gravity exactly (linear), the arc's pressure with the three edge
    midpoints of each triangle (exact for quadratics)."""

    def __init__(self, A, t, n, L, depth, x, gs, rho, gamma, root=None, current=0.):
        self.t, self.n, self.L = np.asarray(t, float), np.asarray(n, float), float(L)
        self.M, self.phi0, self.phi1 = polar_frame(A, t, n, L, depth)
        self.r_toe = math.hypot(.5*L, depth)
        self.N = N_NODES
        self.phi, self.e = polar_dirs(self.t, self.n, self.N, self.phi0, self.phi1)
        self.dphi = (self.phi1 - self.phi0)/self.N
        self.c, self.s = math.cos(self.dphi), math.sin(self.dphi)
        self.gamma, self.rho = gamma, rho
        gs = np.asarray(gs, float)
        self.G0 = -rho*float(gs @ self.M)
        self.G1 = -rho*(self.e @ gs)                       # [N+1]
        self.F = self.sg = 0.
        if root is not None and current > 0.:
            self.F = PR.v(PR.ARC_FORCE)[0] + PR.v(PR.ARC_FORCE)[1]*current
            self.sg = PR.v(PR.ARC_PRESSURE_SIGMA)[0] + PR.v(PR.ARC_PRESSURE_SIGMA)[1]*current
            self.root_yz = np.asarray(root[1:], float)
            self.dx2 = (x - float(root[0]))**2

    def _p(self, m):
        """Arc pressure and its gradient (y, z) at points m [..., 2]."""
        d = m - self.root_yz
        p = self.F/(2*math.pi*self.sg**2)*np.exp(-(self.dx2 + np.sum(d*d, -1))/(2*self.sg**2))
        return p, -p[..., None]*d/self.sg**2

    def area(self, r):
        return .5*self.s*float(np.sum(r[:-1]*r[1:]))

    def parts(self, r):
        """(E, grad E [N+1], Hessian diag [N+1] / off-diagonal [N], area, grad area [N+1])."""
        ri, rj = r[:-1], r[1:]
        c, s, g = self.c, self.s, self.gamma
        l = np.sqrt(np.maximum(ri*ri + rj*rj - 2*ri*rj*c, 1e-30))
        dli, dlj = (ri - rj*c)/l, (rj - ri*c)/l
        a = .5*s*ri*rj
        dai, daj = .5*s*rj, .5*s*ri
        G0, Gi, Gj = self.G0, self.G1[:-1], self.G1[1:]
        Phi = .5*s*(G0*ri*rj + (Gi*ri*ri*rj + Gj*ri*rj*rj)/3.)
        E = g*float(np.sum(l)) + float(np.sum(Phi))
        gi = g*dli + .5*s*(G0*rj + (2*Gi*ri*rj + Gj*rj*rj)/3.)
        gj = g*dlj + .5*s*(G0*ri + (Gi*ri*ri + 2*Gj*ri*rj)/3.)
        hii = g*(1 - dli*dli)/l + .5*s*(2*Gi*rj/3.)
        hjj = g*(1 - dlj*dlj)/l + .5*s*(2*Gj*ri/3.)
        hij = g*(-c - dli*dlj)/l + .5*s*(G0 + 2*(Gi*ri + Gj*rj)/3.)
        if self.F > 0:
            Pi = self.M[None, :] + ri[:, None]*self.e[:-1]
            Pj = self.M[None, :] + rj[:, None]*self.e[1:]
            p1, q1 = self._p(.5*(self.M[None, :] + Pi))
            p2, q2 = self._p(.5*(Pi + Pj))
            p3, q3 = self._p(.5*(Pj + self.M[None, :]))
            pb = (p1 + p2 + p3)/3.
            dpi = (np.sum(q1*self.e[:-1], 1) + np.sum(q2*self.e[:-1], 1))/6.
            dpj = (np.sum(q2*self.e[1:], 1) + np.sum(q3*self.e[1:], 1))/6.
            E += float(np.sum(a*pb))
            gi = gi + dai*pb + a*dpi
            gj = gj + daj*pb + a*dpj
            hii = hii + 2*dai*dpi
            hjj = hjj + 2*daj*dpj
            hij = hij + .5*s*pb + dai*dpj + daj*dpi
        G = np.zeros(self.N + 1)
        G[:-1] += gi
        G[1:] += gj
        D = np.zeros(self.N + 1)
        D[:-1] += hii
        D[1:] += hjj
        GA = np.zeros(self.N + 1)
        GA[:-1] += dai
        GA[1:] += daj
        return E, G, D, hij, float(np.sum(a)), GA

    def _restore(self, r, V, free, lo, hi):
        """Move the free radii along the area's gradient until the polygon's area is V (the area is quadratic along a
        line: solved exactly, then clipped to the bounds; a few passes when clipping takes some of it back)."""
        for _ in range(6):
            Aw = self.area(r)
            if abs(Aw - V) <= 1e-12 + 1e-9*V:
                break
            _, _, _, _, _, GA = self.parts(r)
            w = np.where(free & (((Aw < V) & (r < hi - 1e-12)) | ((Aw > V) & (r > lo + 1e-12))), GA, 0.)
            if not w.any():
                break
            b = float(GA @ w)
            q = self.s*float(np.sum(w[:-1]*w[1:]))
            cst = Aw - V
            if abs(q) < 1e-30:
                tau = -cst/b
            else:
                disc = b*b - 2*q*cst
                if disc < 0:
                    tau = -b/q
                else:
                    r1, r2 = (-b + math.sqrt(disc))/q, (-b - math.sqrt(disc))/q
                    tau = r1 if abs(r1) < abs(r2) else r2
            r = np.clip(r + tau*w, lo, hi)
        return r

    def solve(self, V, r0, lo, hi, fixed=None, iters=80):
        """Minimise E at polygon area V from r0 (nodes 0 and N are the toes, held). Returns (r, pressure, converged).
        Newton on the KKT system of the free radii with a Levenberg shift: far from the minimum the reduced Hessian
        (on the area constraint's tangent space) can be indefinite and the plain Newton step then climbs; the shift
        grows until the step descends and the line search takes it, and shrinks again after each success."""
        N = self.N
        lo = np.maximum(np.asarray(lo, float), 0.)
        hi = np.maximum(np.asarray(hi, float), lo)
        lo[0] = hi[0] = lo[N] = hi[N] = self.r_toe
        fixed = np.zeros(N + 1, bool) if fixed is None else np.asarray(fixed, bool).copy()
        fixed[0] = fixed[N] = True
        r = np.clip(np.asarray(r0, float).copy(), lo, hi)
        free = ~fixed
        r = self._restore(r, V, free, lo, hi)
        if abs(self.area(r) - V) > 1e-6*max(V, 1e-12):
            return r, 0., False                         # the bounds cannot hold V (frozen stations, bodies)
        lam, ok, mu = 0., False, 0.
        ar = np.arange(N + 1)
        for _ in range(iters):
            E, G, D, O, A, GA = self.parts(r)
            act = free & (((r <= lo + 1e-12) & (G - lam*GA > 0)) | ((r >= hi - 1e-12) & (G - lam*GA < 0)))
            F_ = np.nonzero(free & ~act)[0]
            if len(F_) == 0:
                ok = True
                break
            ga = GA[F_]
            lam = float(G[F_] @ ga)/max(float(ga @ ga), 1e-30)
            gF = G[F_] - lam*ga
            gt = gF - (float(gF @ ga)/max(float(ga @ ga), 1e-30))*ga        # the gradient on the tangent space
            if float(np.max(np.abs(gt))) < 1e-9*self.gamma:
                ok = True
                break
            H = np.zeros((N + 1, N + 1))
            H[ar, ar] = D
            Oa = O - lam*.5*self.s
            H[ar[:-1], ar[1:]] = Oa
            H[ar[1:], ar[:-1]] = Oa
            HF = H[np.ix_(F_, F_)]
            m = len(F_)
            scale = max(float(np.mean(np.abs(np.diag(HF)))), 1e-9)
            mu = max(mu*.25, 0.)
            accepted = False
            for _shift in range(12):
                K = np.zeros((m + 1, m + 1))
                K[:m, :m] = HF + mu*np.eye(m)
                K[:m, m] = -ga
                K[m, :m] = -ga
                rhs = -np.concatenate([gF, [-(A - V)]])
                try:
                    sol = np.linalg.solve(K, rhs)
                except np.linalg.LinAlgError:
                    mu = max(4*mu, 1e-3*scale)
                    continue
                d = np.zeros(N + 1)
                d[F_] = sol[:m]
                if float(d[F_] @ gt) >= 0:
                    mu = max(4*mu, 1e-3*scale)
                    continue
                alpha = 1.
                for _ls in range(12):
                    rt = self._restore(np.clip(r + alpha*d, lo, hi), V, free, lo, hi)
                    Et = self.parts(rt)[0]
                    if Et < E - 1e-4*alpha*abs(float(d[F_] @ gt)):
                        accepted = True
                        break
                    alpha *= .5
                if accepted:
                    break
                mu = max(4*mu, 1e-3*scale)
            if not accepted:
                ok = float(np.max(np.abs(gt))) < 1e-6*self.gamma
                break
            step = float(np.max(np.abs(rt - r)))
            r = rt
            if step < 1e-10:
                ok = True
                break
        _, G, _, _, A, GA = self.parts(r)
        F_ = free & (r > lo + 1e-12) & (r < hi - 1e-12)
        if F_.any():
            lam = float(G[F_] @ GA[F_])/max(float(GA[F_] @ GA[F_]), 1e-30)
        self.converged = ok
        # a feasible surface holds the slice's metal exactly; one the iterations left short of the minimum is kept (its
        # energy is below the start's) and counted (pool.overhang_unconverged)
        return r, lam, abs(A - V) <= 1e-6*max(V, 1e-12)


class SliceState:
    __slots__ = ("liquid", "frozen", "A", "B", "u", "h", "horig", "area", "liq_area", "overlap", "toe_ext",
                 "undercut_cells", "chord_t", "chord_n", "x_air_max", "porosity", "freeze_t", "kind", "r", "pts",
                 "overhang", "depth", "contact")

    def __init__(self):
        self.liquid = False
        self.frozen = False
        self.A = self.B = None
        self.u = self.h = self.horig = None
        self.kind = "h"                 # "h": height over the chord; "polar": r(phi) about its midpoint (overhangs)
        self.r = None
        self.depth = None               # polar: the centre's depth below the chord's midpoint
        self.pts = None                 # the free surface as (y, z) points from toe A to toe B (both kinds)
        self.overhang = 0.              # m2: area of the profile held past the 88 deg height-function face (polar)
        self.area = 0.
        self.liq_area = 0.
        self.overlap = False
        self.toe_ext = [0, 0]           # samples (0.05 mm) the contact line lies past the fused edge at A, B (reported)
        self.contact = None             # [sA, sB]: the liquid's contact line on the joint's surface (arclength, m) while
                                        # the slice stays liquid; None once it has frozen
        self.undercut_cells = 0
        self.chord_t = self.chord_n = None
        self.x_air_max = 0.
        self.porosity = 0.
        self.freeze_t = None


class Pool:
    def __init__(self, solver, joint, original_metal):
        """solver: ThermalSolver; joint: Joint; original_metal: bool flat array (cells that were metal at the start)."""
        self.s = solver
        self.g = solver.g
        self.joint = joint
        g = self.g
        y0, y1, z0, z1 = joint.box
        (self.i_all, _), (self.j0, self.j1), (self.k0, self.k1) = (0, g.nx), *g.range_of((0, y0, z0), (0, y1, z1))[1:]
        self.orig = original_metal
        self.slices = [SliceState() for _ in range(g.nx)]
        self.gamma = PR.v(PR.STEEL_GAMMA)
        self.rho = PR.v(PR.STEEL_RHO_LIQ)
        self.frozen_volume = 0.          # informational: metal added to slices that have frozen (from their cells)
        # solidification shrinkage, mass-conserving: the pool's liquid takes SHRINK more volume than the same metal solid,
        # so its slices hold (1 + SHRINK) x their liquid metal; metal that freezes keeps the shape it froze in and the
        # pool feeds its shrinkage, and the last liquid freezes short (the crater). The frozen bead holds exactly the
        # deposit.
        self.liquid_expansion = 0.      # m3: SHRINK x the liquid metal of the slices the pool solves, now
        self.dV = np.zeros(self.g.nx)    # per slice: metal volume now minus at the start, kept from the cells changed
        self.overflow_events = 0
        self.cap_searches = 0            # slices whose exact cap was searched (_caps; a cost counter, not state)
        self.excess = 0.
        self.spilled_m3 = 0.
        self.stacked_m3 = 0.             # filler placed by _stack (no pool where the droplets landed)
        self.drop_axis = None            # wire axis (weld frame) of the arc that lands the droplets
        self.stack_cells = set()         # cells placed by _stack since the last pool formed (this stack's own metal)
        self.events = []
        self.dirty = set()               # x slices whose cells the pool changed since the last read (weld colliders)
        # rigid bodies over the joint (set by the world: world.World._pool_obstacles): callable(xs [S], pts [S, N+1, 2],
        # nrm [S, 2]) -> clearance [S, N+1] (m) from each chord point along the slice's outward normal to the first body
        # that is not the coupon's metal (the nozzle, a finger, a robot link), inf where none; None: no bodies
        self.obstacle = None
        self.obstacle_bounded = 0        # station-updates whose free surface a body bounded (reported)
        self.arrested_m3 = 0.            # always 0
        self.arrest_events = 0           # updates in which a full slice could advance neither toe (reported)
        self.overhang_m3 = 0.            # the largest volume held past the 88 deg face as overhanging bulges (reported)
        self.overhang_unconverged = 0    # overhang surfaces the iterations left short of their minimum (area exact)
        self.overhang_reasons = {}       # why a slice could not hold its share (reported: empty / buried / infeasible)
        self.overhang_events = 0
        self.overhang_failures = 0       # bulges the polar surface could not hold (their share waits, pool_unplaced)
        self.touchdown_events = 0        # a bulge came down on the metal beyond its toe (the contact line moved)
        self.p_pool = 0.
        self.root_y = None
        yc, zc = g.yc[self.j0:self.j1], g.zc[self.k0:self.k1]
        self.YY, self.ZZ = np.meshgrid(yc, zc, indexing="ij")             # [bj, bk]
        self.cell_area = (g.dy[self.j0:self.j1][:, None]*g.dz[self.k0:self.k1][None, :])
        self.h_new_weld = solver.h_of(TH.WELD, PR.v(PR.STEEL_LIQUIDUS) + 30.)

    # ---------------------------------------------------------------- data access
    def _box_indices(self, slices):
        g = self.g
        I = np.asarray(slices)[:, None, None]
        J = np.arange(self.j0, self.j1)[None, :, None]
        K = np.arange(self.k0, self.k1)[None, None, :]
        return ((I*g.ny + J)*g.nz + K).astype(np.int64)

    # ---------------------------------------------------------------- geometry per slice
    def _toes(self, peakf2d, i, mat2d=None, orig2d=None, fl2d=None):
        """Arclength interval [sA, sB] of the pool's contact with the joint's original surface around the root, or
        None. A surface cell counts as fused if it has been liquid (peak fraction >= 0.5) or if it was metal and is no
        longer original metal (it melted and flowed: now air in a groove, or weld metal refilling one). The contact
        reaches on past the fused interval over surface that liquid metal lies on (the cell 0.1 mm outside it is liquid
        metal): liquid connected to the pool along the surface is the pool's, whatever it rests on -- a tack's lap
        remelted by a fillet beside the fillet's fused toe, the droplets stacked at a cold start. (Ending the pool's
        chord at the fused toe would leave remelted metal beyond it to refreeze where it lay: a fillet over a tack would
        never take the tack's cold lap in.)"""
        J = self.joint
        s = np.arange(0., J.length + 1e-9, 5e-5)
        p, n = J.point(s)
        q = p - n*1.0e-4                                    # 0.1 mm inside the metal
        jj = np.clip(np.searchsorted(self.g.ye, q[:, 0]) - 1, self.j0, self.j1 - 1) - self.j0
        kk = np.clip(np.searchsorted(self.g.ze, q[:, 1]) - 1, self.k0, self.k1 - 1) - self.k0
        fused = peakf2d[jj, kk] >= .5
        if mat2d is not None:
            fused |= orig2d[jj, kk] & ((mat2d[jj, kk] == TH.AIR) | (mat2d[jj, kk] == TH.WELD))
        if not fused.any():
            return None
        wet = np.zeros_like(fused)
        if fl2d is not None and mat2d is not None:
            qo = p + n*1.0e-4                               # 0.1 mm outside the original surface
            jo = np.clip(np.searchsorted(self.g.ye, qo[:, 0]) - 1, self.j0, self.j1 - 1) - self.j0
            ko = np.clip(np.searchsorted(self.g.ze, qo[:, 1]) - 1, self.k0, self.k1 - 1) - self.k0
            wet = (mat2d[jo, ko] != TH.AIR) & (fl2d[jo, ko] >= .5) & ~orig2d[jo, ko]
        root_s = J.root_s if J.kind != "plate" or self.root_y is None else self.root_y - J.surface[0, 0]
        r = int(np.argmin(np.abs(s - root_s)))
        if not fused[r]:
            idx = np.nonzero(fused)[0]
            r = int(idx[np.argmin(np.abs(idx - r))])
        a = r
        while a > 0 and fused[a - 1]:
            a -= 1
        while a > 0 and wet[a - 1]:
            a -= 1
        b = r
        while b < len(s) - 1 and fused[b + 1]:
            b += 1
        while b < len(s) - 1 and wet[b + 1]:
            b += 1
        # the contact line does not recede while the slice is liquid (liquid lying on steel stays there): it is the
        # farther of the fused-and-wetted edge and where it already was. Kept as positions on the surface, not as
        # offsets from the fused edge: an offset carried the contact further out whenever fusion grew under it, and a
        # slice that had borrowed a neighbour's chord gave its offset to its own, far shorter fused interval (a start's
        # contact jumped 4.5 mm out onto cold plate in one update). Reset when the slice freezes, so a weld
        # over an earlier one starts from its own fusion (an offset left by a tack used to widen the fillet over it).
        st = self.slices[i]
        af, bf = a, b
        if st.contact is not None:
            a = min(a, int(round(st.contact[0]/5e-5)))
            b = max(b, int(round(st.contact[1]/5e-5)))
        a, b = max(0, a), min(len(s) - 1, b)
        st.toe_ext = [af - a, b - bf]
        if s[b] - s[a] < 3*min(self.g.dy.min(), self.g.dz.min()):
            return None
        return s[a], s[b]

    def _march(self, mask2d, A, t, n, u, sign, limit=8e-3, step=5e-5):
        """For each chord station u: distance along -n (sign=-1) from the chord to the first True cell of mask2d."""
        pts = A[None, :] + u[:, None]*t[None, :]
        d = np.arange(0., limit, step)
        P = pts[:, None, :] - d[None, :, None]*n[None, None, :]
        jj = np.clip(np.searchsorted(self.g.ye, P[..., 0]) - 1, self.j0, self.j1 - 1) - self.j0
        kk = np.clip(np.searchsorted(self.g.ze, P[..., 1]) - 1, self.k0, self.k1 - 1) - self.k0
        hit = mask2d[jj, kk]
        first = np.where(hit.any(1), hit.argmax(1), len(d) - 1)
        return -d[first]

    def _march_in(self, mask2d, A, t, n, u, top=8e-3, step=5e-5):
        """For each chord station u: height (along n, from the chord) of the outermost True cell of mask2d, marching
        inward from `top` above the chord; -top - step if none."""
        pts = A[None, :] + u[:, None]*t[None, :]
        d = np.arange(0., 2*top, step)
        P = pts[:, None, :] + (top - d)[None, :, None]*n[None, None, :]
        jj = np.clip(np.searchsorted(self.g.ye, P[..., 0]) - 1, self.j0, self.j1 - 1) - self.j0
        kk = np.clip(np.searchsorted(self.g.ze, P[..., 1]) - 1, self.k0, self.k1 - 1) - self.k0
        hit = mask2d[jj, kk]
        first = np.where(hit.any(1), hit.argmax(1), len(d) - 1)
        return top - d[first]

    # ---------------------------------------------------------------- Young-Laplace, batched
    def _toe_angles(self, h, L):
        """The surface's angle to the chord at each toe (rad), extrapolated from its first two segments' angles (the
        angle varies about linearly along a meniscus near its edge; the first segment alone reads a weightless 80 deg
        arc as 72 deg at 24 nodes)."""
        du = (L/N_NODES)[:, None]
        a1 = np.arctan(np.stack([h[:, 1], h[:, -2]], 1)/du)
        a2 = np.arctan(np.stack([h[:, 2] - h[:, 1], h[:, -3] - h[:, -2]], 1)/du)
        return np.max(1.5*a1 - .5*a2, 1)

    def _caps(self, prob, lim_deg=None):
        """Per slice, the pressure at which its equilibrium surface meets a toe at the limit slope (MAX_END_SLOPE): the
        most metal its chord holds before it spills over a toe. Bisection on each slice's own pressure, batched.
        The pressure of a weightless circular arc at the limit slope (2 gamma sin(theta) / L) is not the cap: with its
        weight, a 9.4 mm bead on a flat plate at that pressure can only reach a 52 deg toe (the first integral
        gamma (1 - cos theta) = P H - rho g H^2 / 2 is at most P^2 / (2 rho g)), so a pool would spill over its fused
        toes holding about two thirds of what it can: E3's bead on plate came out 33 % wider and 25 % lower than the
        published section, with weld metal past the fusion line on both sides."""
        ns = len(prob)
        if ns == 0:
            return np.zeros(0)
        lim = math.atan(MAX_END_SLOPE) if lim_deg is None else math.radians(lim_deg)
        L = np.array([p["L"] for p in prob])
        trial = [dict(p) for p in prob]

        def angle(Pv):
            for k in range(ns):
                trial[k]["pcap"] = float(Pv[k])
            h, _ = self._solve_profiles(trial, 1e30)
            return self._toe_angles(h, L)
        lo = np.zeros(ns)
        over = np.array([p.get("p_over", np.inf) for p in prob])
        hi = np.minimum(np.array([p["pcap"] for p in prob]), over)
        a = angle(hi)
        for _ in range(8):
            low = (a < lim) & (hi < over)
            if not low.any():
                break
            lo = np.where(low, hi, lo)
            hi = np.where(low, np.minimum(2*hi, over), hi)
            a = angle(hi)
        # slices whose surface does not reach the limit even at the overhang pressure (frozen stations at the toes):
        # their cap is the overhang pressure
        at_over = (a < lim) & (hi >= over)
        if at_over.all():
            return hi
        for _ in range(9):
            m = .5*(lo + hi)
            reach = angle(m) >= lim
            hi = np.where(reach | at_over, np.where(at_over, hi, m), hi)
            lo = np.where(reach | at_over, lo, m)
        return hi

    def _solve_profiles(self, prob, P):
        """Equilibrium free surfaces of all live slices at pool pressure P (P_k = min(P + rho g_x x_k, pcap_k)).

        Each slice's surface minimises its energy per unit length (h_0 = h_N = 0 at the toes, h >= hmin):
            F[h] = gamma sum du sqrt(1 + s^2) - rho sum du [(g.c) h + (g.n) h^2 / 2] + sum du (p_arc - P_k) h
        (surface tension, the liquid's weight, the arc pressure's work, the pressure-volume term), whose stationarity
        is Young-Laplace gamma kappa = P_k + rho g.r - p_arc. F is convex wherever gravity holds the liquid toward the
        chord, so the minimiser is unique and the held volume is monotone in P. Projected Newton with an Armijo line
        search, batched over slices (tridiagonal Hessians solved densely, 23 x 23). Returns (h [ns, N+1], u)."""
        ns = len(prob)
        N = N_NODES
        gamma, rho = self.gamma, self.rho
        L = np.array([p["L"] for p in prob])
        du = (L/N)[:, None]
        u = np.linspace(0, 1, N + 1)[None, :]*L[:, None]
        A = np.stack([p["A"] for p in prob])
        t = np.stack([p["t"] for p in prob])
        n = np.stack([p["n"] for p in prob])
        Pk = np.minimum(P + rho*self.gx*np.array([p["x"] for p in prob]), np.array([p["pcap"] for p in prob]))
        gn = rho*(n @ self.gs)                                                 # rho g.n per slice
        hmin = np.stack([p["hmin"] for p in prob])
        hmax = np.stack([p.get("hmax", np.full(N + 1, np.inf)) for p in prob])
        parc = np.stack([p["parc"] for p in prob])
        base_r = A[:, None, :] + u[..., None]*t[:, None, :]
        gc = rho*(base_r @ self.gs)                                            # rho g.c at chord points
        lin = (-gc - Pk[:, None] + parc)[:, 1:-1]                              # linear coefficient per interior node
        quad = -gn[:, None]                                                    # h^2/2 coefficient

        def energy(h):
            sl = np.diff(h, axis=1)/du
            return gamma*np.sum(du*np.sqrt(1 + sl*sl), 1) + np.sum(du*(lin*h[:, 1:-1] + .5*quad*h[:, 1:-1]**2), 1)

        h = np.stack([p.get("h0", np.zeros(N + 1)) for p in prob]).copy()
        h[:, 0] = h[:, -1] = 0.
        h[:, 1:-1] = np.minimum(np.maximum(h[:, 1:-1], hmin[:, 1:-1]), np.maximum(hmax[:, 1:-1], hmin[:, 1:-1]))
        M = N - 1
        ar = np.arange(M)
        lo = hmin[:, 1:-1]
        up = np.maximum(hmax[:, 1:-1], lo)
        F = energy(h)
        done = np.zeros(ns, bool)
        for _ in range(60):
            sl = np.diff(h, axis=1)/du
            q = 1/np.sqrt(1 + sl*sl)
            c = q**3
            grad = gamma*(sl[:, :-1]*q[:, :-1] - sl[:, 1:]*q[:, 1:]) + du*(lin + quad*h[:, 1:-1])
            H = np.zeros((ns, M, M))
            H[:, ar, ar] = gamma/du*(c[:, :-1] + c[:, 1:]) + du*quad
            H[:, ar[1:], ar[:-1]] = -gamma/du*c[:, 1:-1]
            H[:, ar[:-1], ar[1:]] = -gamma/du*c[:, 1:-1]
            act = ((h[:, 1:-1] <= lo + 1e-12) & (grad > 0)) | ((h[:, 1:-1] >= up - 1e-12) & (grad < 0))
            # nonconvex slices (liquid hanging below its chord): Levenberg shift keeps the step a descent direction
            dmin = np.min(H[:, ar, ar], 1)
            shift = np.where(dmin < 1e-9, -dmin + gamma/du[:, 0], 0.)
            H[:, ar, ar] += shift[:, None]
            Hf = np.where(act[:, :, None] | act[:, None, :], 0., H)
            Hf[:, ar, ar] = np.where(act, 1., Hf[:, ar, ar])
            gf = np.where(act, 0., grad)
            try:
                d = -np.linalg.solve(Hf, gf[..., None])[..., 0]
            except np.linalg.LinAlgError:
                d = -gf/np.maximum(np.abs(H[:, ar, ar]), 1e-12)
            if np.any(np.sum(d*gf, 1) >= 0):            # not a descent direction (indefinite): steepest descent there
                bad = np.sum(d*gf, 1) >= 0
                d[bad] = -gf[bad]/np.maximum(np.abs(H[bad][:, ar, ar]), 1e-12)
            d = np.where(done[:, None], 0., d)
            alpha = np.ones(ns)
            accepted = np.zeros(ns, bool) | done
            hn = h.copy()
            for _ls in range(30):
                trial = h.copy()
                trial[:, 1:-1] = np.minimum(np.maximum(lo, h[:, 1:-1] + alpha[:, None]*d), up)
                Ft = energy(trial)
                ok = (Ft <= F + 1e-4*np.sum(grad*(trial[:, 1:-1] - h[:, 1:-1]), 1)) & ~accepted
                hn[ok] = trial[ok]
                accepted |= ok
                if accepted.all():
                    break
                alpha = np.where(accepted, alpha, alpha*.5)
            step = np.max(np.abs(hn - h), 1)
            h = hn
            F = energy(h)
            done |= step < 1e-10
            if done.all():
                break
        return h, u

    def _longitudinal(self, prob, rows, h, u, areas, dx, h_hi, p_lo):
        g = self.g
        x = np.array([g.xc[i] for _, i in rows])
        cap = self._areas(prob, h_hi, u)
        W = np.exp(-.5*((x[:, None] - x[None, :])/LONG_SIGMA)**2)
        W[np.abs(x[:, None] - x[None, :]) > 3*LONG_SIGMA] = 0.
        # rows can be separated by a gap of slices that are not in the pool: no sharing across it
        idx = np.array([i for _, i in rows])
        seg = np.concatenate([[0], np.cumsum(np.diff(idx) > 1)])
        W[seg[:, None] != seg[None, :]] = 0.
        Wd = W*dx[None, :]
        tot = float(np.sum(areas*dx))
        sm = (Wd @ areas)/np.maximum(Wd.sum(1), 1e-30)
        for _ in range(4):                                     # conserve the pool's metal; respect each chord's cap
            sm *= tot/max(float(np.sum(sm*dx)), 1e-30) if tot > 0 else 1.
            over = sm > cap
            if not over.any():
                break
            sm = np.where(over, cap, sm)
        # per-slice pressures holding the shares (vectorised bisection)
        lo = np.full(len(rows), p_lo)
        hi = np.array([p["pcap"] for p in prob])
        hh = h
        for _ in range(28):
            mid = .5*(lo + hi)
            hm, _ = self._solve_profiles(prob, mid)
            am = self._areas(prob, hm, u)
            up = am < sm
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
            hh = hm
        return hh, self._areas(prob, hh, u)

    def _areas(self, prob, h, u):
        horig = np.stack([p["horig"] for p in prob])
        return np.trapezoid(h - horig, u, axis=1)

    # ---------------------------------------------------------------- update
    def update(self, deposited_m3, root=None, current=0., gravity=(0., 0., -9.81), root_y=None, drop=None):
        """Advance the pool's geometry to the thermal state. Returns dict(new, removed) energy (J) moved by cell
        changes (booked into the solver's ledger here)."""
        g, s = self.g, self.s
        self.root_y = root_y if root_y is not None else (None if root is None else float(root[1]))
        self.gx = float(gravity[0])
        self.gs = np.array([gravity[1], gravity[2]], float)
        flmax = s.slice_max_liquid(self.j0, self.j1, self.k0, self.k1)
        live = [i for i in range(g.nx) if flmax[i] >= .5]
        was = [i for i, st in enumerate(self.slices) if st.liquid]
        was_liquid = set(was)
        # freezing slices: fully solid again
        for i in was:
            if flmax[i] < .02:
                st = self.slices[i]
                st.liquid = False
                st.frozen = True
                st.contact = None
                self.frozen_volume += self.dV[i]
        # the pool that receives the filler is the contiguous run of liquid slices at the arc root (the droplets land
        # there); with the arc off, every liquid slice. Detached liquid (a cooling tail beyond a frozen gap) keeps its
        # metal as it is.
        arc_on = drop is not None           # the live arc root (the lagged pressure root decays, never to zero)
        self._arc_on = arc_on
        if not arc_on:
            self.stack_cells.clear()
        if live and arc_on:
            ir = int(np.clip(np.searchsorted(g.xe, drop[0]) - 1, 0, g.nx - 1))
            groups, cur = [], [live[0]]
            for i in live[1:]:
                if i == cur[-1] + 1:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            near = [gr for gr in groups if gr[0] - 2 <= ir <= gr[-1] + 2]
            for gr in groups:
                if not near or gr is not near[0]:
                    for i in gr:
                        self.slices[i].liquid = True
            live = near[0] if near else []
        if drop is not None:
            self.last_drop = np.asarray(drop, float).copy()
        if not live:
            unplaced = deposited_m3 - float(self.dV.sum())
            if arc_on and deposited_m3 > 0 and unplaced > 0:
                return self._stack(unplaced, drop)
            if not arc_on and unplaced > 1e-12 and getattr(self, "last_drop", None) is not None:
                # an overfull pool froze before it took all its metal: what it held back is laid where the arc landed
                self.drop_axis = None
                return self._stack(unplaced, self.last_drop)
            return dict(new_j=0., removed_j=0., slices=0)
        live = np.asarray(live)
        idx = self._box_indices(live)
        flat = idx.ravel()
        bj, bk = self.j1 - self.j0, self.k1 - self.k0
        fl = s.gather("fl", flat).reshape(len(live), bj, bk)
        pf = s.gather("peakf", flat).reshape(len(live), bj, bk)
        mat = s.mat[flat].reshape(len(live), bj, bk)
        tem = s.gather("T", flat).reshape(len(live), bj, bk)
        orig = self.orig[flat].reshape(len(live), bj, bk)
        prob, rows = [], []
        # toes of every liquid slice; a slice inside the pool whose own surface has not fused takes its neighbour's
        # chord: the liquid bridges it, lying on unfused metal there (cold lap). (Leaving it out lets single 0.5 mm
        # slices stay nearly empty between full ones -- fins -- in short-circuit tacks.)
        toes_all = [self._toes(pf[n_], i, mat[n_], orig[n_], fl[n_]) for n_, i in enumerate(live)]
        have = [n_ for n_, tt in enumerate(toes_all) if tt is not None]
        # only a gap with fused slices on both sides is bridged: at the pool's ends the edge of the liquid is where the
        # base melts (a moving pool's front is its fusion front), and a slice past the last fused one that took its
        # neighbour's chord would hold liquid on unfused metal, a cold-lap lip at both ends of every run (tacks' end
        # sections, a back-step's turnaround)
        bridged = set()
        for n_ in range(len(live)):
            if toes_all[n_] is None and have:
                lo_ = [m_ for m_ in have if m_ < n_ and n_ - m_ <= BRIDGE_SLICES]
                hi_ = [m_ for m_ in have if m_ > n_ and m_ - n_ <= BRIDGE_SLICES]
                if lo_ and hi_:
                    n2 = min(have, key=lambda m_: abs(m_ - n_))
                    toes_all[n_] = toes_all[n2]
                    bridged.add(n_)
        for n_, i in enumerate(live):
            st = self.slices[i]
            if st.frozen:                       # re-melted: its metal returns to the liquid inventory
                st.frozen = False
                self.frozen_volume -= self.dV[i]
            st.liquid = True
            toes = toes_all[n_]
            if toes is None:
                continue
            if n_ in bridged:
                st.overlap = True
            else:
                st.contact = [float(toes[0]), float(toes[1])]
            pa, _ = self.joint.point(np.array([toes[0]]))
            pb, _ = self.joint.point(np.array([toes[1]]))
            A, B = pa[0], pb[0]
            L = float(np.linalg.norm(B - A))
            t = (B - A)/L
            nrm = np.array([-t[1], t[0]])
            # outward: away from metal at the chord midpoint
            mid = .5*(A + B)
            jm = np.clip(np.searchsorted(g.ye, (mid - nrm*2e-4)[0]) - 1, self.j0, self.j1 - 1) - self.j0
            km = np.clip(np.searchsorted(g.ze, (mid - nrm*2e-4)[1]) - 1, self.k0, self.k1 - 1) - self.k0
            jo = np.clip(np.searchsorted(g.ye, (mid + nrm*2e-4)[0]) - 1, self.j0, self.j1 - 1) - self.j0
            ko = np.clip(np.searchsorted(g.ze, (mid + nrm*2e-4)[1]) - 1, self.k0, self.k1 - 1) - self.k0
            if orig[n_, jo, ko] and not orig[n_, jm, km]:
                nrm = -nrm
            u = np.linspace(0, L, N_NODES + 1)
            horig = self._march(orig[n_], A, t, nrm, u, -1)
            # the free surface cannot pass through metal that is solid now (unmelted base metal or weld metal that has
            # frozen again): the bound is the top of the solid seen from outside. (Bounding by never-melted metal only
            # lets the surface pass through refrozen weld metal it cannot remove: a period-2 slosh.)
            solid = (mat[n_] != TH.AIR) & (fl[n_] < .5)
            hmin = self._march_in(solid, A, t, nrm, u)
            hmin[0] = hmin[-1] = 0.
            hmax = np.full(N_NODES + 1, np.inf)
            if FREEZE_SURFACE:
                # where the outermost metal at a station is metal that melted and has solidified again (weld metal, or
                # parent metal the pool fused) the surface is frozen there; stations over liquid, over air, or over the
                # untouched original surface stay free: liquid may run onto cold metal (and freezes there, overlap).
                # (Pinning the untouched surface too makes a toe extended onto cold metal add no capacity: the pool
                # overflowed in 688 of 990 updates and spread 28 mm films, E3)
                metal = mat[n_] != TH.AIR
                htop = self._march_in(metal, A, t, nrm, u)
                resolid = metal & (fl[n_] < .5) & ((mat[n_] == TH.WELD) | (pf[n_] >= .5))
                hres = self._march_in(resolid, A, t, nrm, u)
                fixed = (np.abs(htop - hres) <= 1e-9) & (hres > -8e-3)
                fixed[0] = fixed[-1] = False
                hmax = np.where(fixed, np.maximum(hmin, hres), np.inf)
            parc = np.zeros(N_NODES + 1)
            if root is not None and current > 0.:
                F = PR.v(PR.ARC_FORCE)[0] + PR.v(PR.ARC_FORCE)[1]*current
                sg = PR.v(PR.ARC_PRESSURE_SIGMA)[0] + PR.v(PR.ARC_PRESSURE_SIGMA)[1]*current
                if F > 0:
                    pts = A[None, :] + u[:, None]*t[None, :]
                    r2 = (g.xc[i] - root[0])**2 + (pts[:, 0] - root[1])**2 + (pts[:, 1] - root[2])**2
                    parc = F/(2*math.pi*sg*sg)*np.exp(-r2/(2*sg*sg))
            h0 = None
            if st.kind == "h" and st.h is not None and st.u is not None and len(st.h) == N_NODES + 1 \
                    and abs(st.u[-1] - L) < 1e-9:
                h0 = st.h
            # first estimate of the most this chord can hold: a circular arc meeting the toes at the limit slope
            # (weightless; _caps then finds the pressure at which the slice's own surface, with its weight and the arc's
            # pressure, meets a toe at that slope)
            pcap = 2*self.gamma*math.sin(math.atan(MAX_END_SLOPE))/max(L, 1e-4)
            # the most pressure the slice's free surface can take before it would have to overhang: a semicircle over its
            # longest free span (stations not frozen), plus the hydrostatic head of that semicircle's height (the arc's
            # pressure is left out: it only lowers the surface's effective pressure where it acts). Above it the
            # height-function surface has no solution; with frozen stations at the toes the toe angle never reached the
            # limit, the cap search doubled the pressure to 256x and the surface came out 22-41 mm tall: 16 mm fins of
            # weld metal
            free = ~np.isfinite(hmax[1:-1])
            run, best = 0, 0
            for f_ in free:
                run = run + 1 if f_ else 0
                best = max(best, run)
            l_free = max(best + 1, 1)*L/N_NODES
            g_hold = max(0., -float(nrm @ self.gs))
            p_over = 2*self.gamma/max(l_free, 1e-5) + self.rho*g_hold*.5*l_free
            pcap = min(pcap, p_over)
            p = dict(L=L, A=A, t=t, n=nrm, horig=horig, hmin=hmin, hmax=hmax, x=float(g.xc[i]), parc=parc, pcap=pcap,
                     p_over=p_over)
            if h0 is not None:
                p["h0"] = h0
            prob.append(p)
            rows.append((n_, i))
            st.liq_area = float(np.sum((fl[n_] >= .5)*(mat[n_] != TH.AIR)*self.cell_area))
        if not prob:
            if arc_on and deposited_m3 > 0:
                unplaced = deposited_m3 - float(self.dV.sum())
                if unplaced > 0:
                    return self._stack(unplaced, drop)
            return dict(new_j=0., removed_j=0., slices=0)
        self.stack_cells.clear()            # a pool spans the root: the stacked droplets are now its metal
        dx = np.array([g.dx[i] for _, i in rows])
        # metal a pool slice holds outside its chord's span (beyond the toes: stacked droplets, an earlier pass) is
        # not part of its profile's area: it is counted from the cells, like the other slices' metal -- except liquid
        # weld metal connected, in the section, to the liquid over the chord's span: one body of liquid is the pool's (an
        # overhang from the last update, held again by this update's profile, whichever form it takes).
        outside = 0.
        for k, (n_, i) in enumerate(rows):
            A_, t_, L_, n_k = prob[k]["A"], prob[k]["t"], prob[k]["L"], prob[k]["n"]
            uu = (self.YY - A_[0])*t_[0] + (self.ZZ - A_[1])*t_[1]
            out_ = (uu < 0) | (uu > L_)
            o_ = orig[n_]
            liq = (fl[n_] >= .5) & (mat[n_] != TH.AIR)
            lab, _nl = ndimage.label(liq)
            own = np.unique(lab[liq & ~out_])
            pooled = liq & ~o_ & np.isin(lab, own[own > 0])
            prob[k]["pooled_out"] = out_ & pooled
            add_ = (mat[n_] != TH.AIR) & ~o_ & out_ & ~pooled
            rem_ = o_ & (mat[n_] == TH.AIR) & out_
            outside += float(np.sum(self.cell_area[add_]) - np.sum(self.cell_area[rem_]))*g.dx[i]
        # the live slices hold the filler not already present in other slices' cells (mass balance closed on the cells:
        # bookkeeping by frozen profile areas drifted 11 % from the deposit)
        in_rows = np.zeros(g.nx, bool)
        in_rows[[i for _, i in rows]] = True
        self.liquid_expansion = SHRINK*float(sum(self.slices[i].liq_area*g.dx[i] for _, i in rows))
        target = max(0., deposited_m3 + self.liquid_expansion - float(self.dV[~in_rows].sum()) - outside)
        self.target = target
        if self.obstacle is not None:
            # a body over the joint bounds the free surface as solid metal does: liquid cannot take the space a body
            # holds, so it spreads beneath it (liquid does not lift a body). Without the bound the pool rises into a
            # nozzle resting on the bead; when that metal freezes its collider box grows around the nozzle in one step
            # and throws it off at up to 146 mm/s and 1.1 kN. The bound is a
            # one-sided limit: the surface may stay below it, and the overhang pressure above (p_over) is still set by
            # the frozen stations only
            clr = self.obstacle(np.array([p_["x"] for p_ in prob]),
                                np.stack([p_["A"][None, :] + np.linspace(0, p_["L"], N_NODES + 1)[:, None]*p_["t"][None, :]
                                          for p_ in prob]),
                                np.stack([p_["n"] for p_ in prob]))
            if clr is not None:
                for p_, c_ in zip(prob, clr):
                    bound = np.isfinite(c_)
                    bound[0] = bound[-1] = False
                    if bound.any():
                        p_["hmax"] = np.where(bound, np.maximum(p_["hmin"], np.minimum(p_["hmax"], c_)), p_["hmax"])
                        self.obstacle_bounded += int(bound.sum())
        # pressure bracket: each slice's pressure is capped at what its chord can hold (a full slice spills its excess
        # to the rest of the pool); V(P) is then monotone and bounded, overflow = every slice full. The exact cap (the
        # pressure at which the slice's own surface meets a toe at the limit slope, _caps) costs a search per slice, so
        # the pool is first balanced under an upper bound of it (the weightless arc's pressure plus what the liquid's
        # weight and the arc's pressure can add); only the slices whose surface then passes the limit slope get their
        # exact cap and the pool is balanced again (the cap binds only there).
        lim = math.atan(MAX_END_SLOPE)
        Ls = np.array([p_["L"] for p_ in prob])
        g_hold = np.array([max(0., -float(p_["n"] @ self.gs)) for p_ in prob])
        weightless = CAP_MODEL == "weightless"
        if not weightless:
            for p_, gh in zip(prob, g_hold):
                p_["pcap"] = min(p_["pcap"] + math.sqrt(2*self.rho*gh*self.gamma*(1 - math.cos(lim)))
                                 + float(np.max(p_["parc"])), p_["p_over"])
        exact = np.full(len(prob), weightless)
        for _attempt in range(3):
            p_hi = max(p["pcap"] for p in prob)
            h_hi, u = self._solve_profiles(prob, p_hi)
            V_hi = float(np.sum(self._areas(prob, h_hi, u)*dx))
            # lower bracket: lower the pressure until the pool holds no more than the deposit (a small deposit between
            # wide fused toes is a concave meniscus that draws melted wall metal into the root; the active set keeps the
            # surface off unmelted metal). A fixed bracket here once returned a surface 1-4 mm too deep: a void inside
            # the joint.
            p_lo = -p_hi
            for _ in range(12):
                h_lo, _ = self._solve_profiles(prob, p_lo)
                V_lo = float(np.sum(self._areas(prob, h_lo, u)*dx))
                if V_lo <= target:
                    break
                p_lo *= 3.
            overflow = target > V_hi
            if target <= V_lo:
                P, h = p_lo, h_lo
            elif overflow:
                P, h = p_hi, h_hi
            else:
                a, b = p_lo, p_hi
                h = h_hi
                for _ in range(40):
                    m = .5*(a + b)
                    hm, _ = self._solve_profiles(prob, m)
                    Vm = float(np.sum(self._areas(prob, hm, u)*dx))
                    if Vm < target:
                        a = m
                    else:
                        b = m
                        h = hm
                    if abs(Vm - target) < 1e-3*max(target, 1e-9):
                        h = hm
                        break
                P = .5*(a + b)
            # surface tension along the weld: the slices are 2-D, the pool is not. Its longitudinal curvature evens out
            # features shorter than the bead (a fin of one 0.5 mm slice costs far more surface than the bead around
            # it), so the pool's metal is shared along x with a LONG_SIGMA kernel (shorter than the 1.8 mm arc-pressure
            # spot, so the crater under the arc stays) and each slice then takes its own pressure to hold its share.
            areas = self._areas(prob, h, u)
            if len(rows) > 2:
                h, areas = self._longitudinal(prob, rows, h, u, areas, dx, h_hi, p_lo)
            ang = self._toe_angles(h, Ls)
            caps = np.array([p_["pcap"] for p_ in prob])
            capped = np.minimum(P + self.rho*self.gx*np.array([p_["x"] for p_ in prob]), caps) >= caps*(1 - 1e-9)
            bad = ~exact & ((ang > lim + math.radians(.5)) | (capped & (ang < lim - math.radians(.5))))
            if not bad.any():
                break
            ks = np.nonzero(bad)[0]
            for k_, c_ in zip(ks, self._caps([prob[k_] for k_ in ks])):
                prob[k_]["pcap"] = float(c_)
                exact[k_] = True
            self.cap_searches += len(ks)
        self.p_pool = P
        self.dbg = dict(target=target, V_lo=V_lo, V_hi=V_hi, P=P, p_lo=p_lo, p_hi=p_hi, overflow=overflow, n=len(prob))
        if overflow:
            self.overflow_events += 1
        # overflow: filler a slice cannot hold under a face of POOL_TOE_SLOPE_MAX (88 deg) to its chord. A small pool is a
        # liquid drop (Bond number (L/l_c)^2 ~ 0.5 for 3 mm on steel, l_c = sqrt(gamma/rho g) = 4.2 mm). Its edge
        # advances only once the face would overhang, onto metal hot enough that the liquid does not freeze at the
        # contact (_contact_freezes), by what the overflow needs: a full chord's capacity grows as its length squared,
        # so each side moves L (sqrt(target / V_cap) - 1) / 2, at least 0.1 mm and at most SPREAD_MAX_PER_UPDATE
        # (liquid steel spilling at 0.2 m/s); it does not come back while liquid. (A fixed 0.1 mm per update cannot
        # keep up with 25 m/min on one spot; letting the edge retreat made advance and retreat flip-flop with the
        # gravity-consistent cap, and the arc dug a gap.) Toes whose continuation climbs against gravity (the
        # bracket's face in a horizontal fillet) do not advance: an overfull 2F fillet sags onto the plate leg.
        # What the slice cannot hold under that face it holds as a convex bulge past the face: liquid steel on cold
        # steel freezes at the contact and its base stays pinned while liquid keeps arriving, the bulk still liquid
        # (Tavakoli & Kavehpour, ISCST 2012; metal droplets on cold substrates freeze with solidification angles above
        # 90 deg, cold laps under them), so the face overhangs its toe (_overhang: a polar surface, volume-controlled).
        # Where the bulge comes down on the metal beyond the toe the liquid rolls onto it: the contact line moves there,
        # frozen at the contact (cold lap unless the metal under it fuses later).
        # the excess is what the profiles do not hold after the longitudinal sharing (which can leave the full slices'
        # sum a few mm3 under V_hi: a bisection that stops just short of each cap), not target - V_hi: that
        # difference would be dropped every update while the pool overflows (8-12 mm3 under a parked arc, all placed at
        # once when the arc goes off)
        held = float(np.sum(np.asarray(areas)*dx))
        self.excess = max(0., target - held) if overflow else 0.
        grow = math.sqrt(target/max(V_hi, 1e-15)) - 1.
        step_max = round(SPREAD_MAX_PER_UPDATE/TOE_SAMPLE)
        J = self.joint
        ghat = self.gs/max(float(np.linalg.norm(self.gs)), 1e-9)
        full_k = []
        for k, (n_, i) in enumerate(rows):
            st = self.slices[i]
            if overflow and areas[k] > 0 and prob[k]["pcap"] <= P + 1e-6:
                full_k.append(k)
                n_s = int(np.clip(math.ceil(.5*prob[k]["L"]*grow/TOE_SAMPLE), 1, step_max))
                moved, stopped = False, 0
                for side, sgn in ((0, -1.), (1, 1.)):
                    end = prob[k]["A"] if side == 0 else prob[k]["A"] + prob[k]["t"]*prob[k]["L"]
                    # the joint surface's direction beyond this toe (away from the root along the original surface)
                    seg = int(np.argmin(np.linalg.norm(J.surface[:-1] + .5*np.diff(J.surface, axis=0) - end, axis=1)))
                    out_dir = sgn*J.tangent[seg]
                    if float(out_dir @ ghat) < -.3:            # climbing more than ~17 deg against gravity
                        continue
                    froze, _where = self._contact_freezes(n_, side, toes_all[n_], tem, fl)
                    if froze:
                        stopped += 1
                        continue
                    if n_ not in bridged:
                        c = list(toes_all[n_]) if st.contact is None else list(st.contact)
                        c[side] = float(np.clip(toes_all[n_][side] + (-1 if side == 0 else 1)*n_s*TOE_SAMPLE, 0.,
                                                J.length))
                        st.contact = c
                    moved = True
                if not moved and stopped:
                    self.arrest_events += 1
                st.overlap = True
        sol = [None]*len(rows)
        if self.excess > 0 and full_k:
            # the full slices share the excess. Each takes what it held past its face last update (the liquid stays
            # where it is: with the arc off no metal arrives, and an overhang that a height-function face cannot hold
            # is still there), the rest in proportion to what each holds; a slice takes only what its bounds and frozen
            # stations allow (_overhang clips) and passes the rest on. (Shares in proportion to the face's capacity
            # alone move the parked pool's overhangs between slices every update after arc-off, fail where a slice
            # is mostly frozen, and swing its metal by +-40 mm3)
            prev = np.array([max(0., (self.slices[i].area - float(areas[k]))*dx[k])
                             if self.slices[i].kind == "polar" and self.slices[i].liquid else 0. for k, (_, i) in
                             enumerate(rows)])
            extra = np.zeros(len(rows))
            cand = list(full_k)
            remaining = self.excess
            for _pass in range(4):
                if remaining <= 1e-15 or not cand:
                    break
                w = np.array([prev[k] for k in cand])
                if _pass > 0 or w.sum() <= 0:
                    w = np.array([max(float(areas[k]), 1e-12)*dx[k] for k in cand])
                    shares = remaining*w/float(w.sum())
                elif w.sum() >= remaining:
                    shares = remaining*w/float(w.sum())
                else:
                    shares = w + (remaining - float(w.sum()))*np.array([max(float(areas[k]), 1e-12)*dx[k]
                                                                         for k in cand])/float(
                        sum(max(float(areas[k]), 1e-12)*dx[k] for k in cand))
                full_now = []
                for k, e in zip(cand, shares):
                    n_, i = rows[k]
                    res = self._overhang(prob[k], h[k], u[k], float(areas[k]) + (extra[k] + e)/dx[k], self.slices[i],
                                         mat[n_], fl[n_], pf[n_], root, current, float(g.xc[i]), cont=i in was_liquid)
                    if res is None:
                        self.overhang_failures += 1
                        continue
                    sol[k] = res
                    got = (res["area"] - float(areas[k]))*dx[k]
                    remaining -= got - extra[k]
                    extra[k] = got
                    if not res["saturated"]:
                        full_now.append(k)
                cand = full_now
            held_ov = float(extra.sum())
            self.excess = max(0., self.excess - held_ov)
            if held_ov > 0:
                self.overhang_m3 = max(self.overhang_m3, held_ov)
                self.overhang_events += 1
                self.overhang_unconverged += sum(1 for s_ in sol if s_ is not None and not s_["converged"])
        for k, (n_, i) in enumerate(rows):
            st = self.slices[i]
            A_, B_ = prob[k]["A"], prob[k]["A"] + prob[k]["t"]*prob[k]["L"]
            st.A, st.B, st.horig = A_, B_, prob[k]["horig"]
            st.chord_t, st.chord_n = prob[k]["t"], prob[k]["n"]
            if sol[k] is None:
                st.kind, st.r, st.u, st.h, st.depth = "h", None, u[k], h[k], None
                st.area, st.overhang = float(areas[k]), 0.
                st.pts = A_[None, :] + u[k][:, None]*prob[k]["t"][None, :] + h[k][:, None]*prob[k]["n"][None, :]
            else:
                rs = sol[k]
                st.kind, st.r, st.u, st.h, st.depth = "polar", rs["r"], u[k], h[k], rs["depth"]
                st.area, st.overhang = rs["area"], rs["area"] - float(areas[k])
                st.pts = rs["pts"]
                if n_ not in bridged:
                    for side, s_touch in rs["touch"]:
                        c = list(toes_all[n_]) if st.contact is None else list(st.contact)
                        c[side] = min(c[side], s_touch) if side == 0 else max(c[side], s_touch)
                        st.contact = c
                        self.touchdown_events += 1
        out = self._apply(rows, prob, sol, h, u, fl, mat, idx)
        # what the cells now hold against what the profiles were asked to hold: the cells' rounding of the surfaces
        # (reported; m3)
        self.raster_gap = target - (float(sum(self.dV[i] for _, i in rows)) - outside)
        return out

    def _overhang(self, p, h_cap, u_cap, area, st, mat2d, fl2d, pf2d, root, current, x, cont=False):
        """Hold `area` (m2, the slice's share measured from the original surface, as _areas) on the chord of problem p
        as a polar surface about the chord's midpoint (_PolarSurface), starting from the surface the slice's liquid had
        at the last update (cont: the slice was liquid then; any chord) or else from its height-function face at the
        cap (h_cap over u_cap). Bounds per node from the cells
        along its ray: the surface lies in the gap (no solid) that holds the start radius, from the solid it rests on to
        the first metal beyond (the joint's surface past a toe, a body); a node whose gap is air over resolidified metal
        is held there (a frozen station, as FREEZE_SURFACE). Returns dict(r, pts, area, touch=[(side, s)]) or None."""
        A, t, n, L = p["A"], p["t"], p["n"], p["L"]
        # the centre lies below the face's lowest point (an arc crater may dip below the chord: with a centre on the
        # chord the crater's nodes collapse onto it and the solve stalls), at most 0.3 L, or 0.8 of the
        # joint's depth below the chord (inside the joint's core: the toes and the overhang past them stay in view)
        same = st.kind == "polar" and st.r is not None and st.A is not None and np.allclose(st.A, A, atol=1e-9) \
            and np.allclose(st.B, A + L*t, atol=1e-9)
        lowest = min(float(np.min(h_cap)), float(np.min(st.h)) if same and st.h is not None else 0.)
        depth_max = max(.3*L, .8*float(-np.min(p["horig"])))
        depth = float(np.clip(max(0., -lowest) + 2.5e-4, 0., depth_max))
        if same and st.depth is not None and st.depth >= depth - 1e-12:
            depth = float(st.depth)
        S = _PolarSurface(A, t, n, L, depth, x, self.gs, self.rho, self.gamma, root, current)
        M = S.M
        corner = -float(np.trapezoid(p["horig"], u_cap))           # the joint's area between chord and original surface
        # the polygon's fan from the centre covers the triangle centre-B-A below the chord as well: what the profile
        # holds above the original surface is the fan less that triangle plus the corner
        tri = .5*L*depth
        V = area - corner + tri
        if V <= 0:
            self.overhang_reasons["empty"] = self.overhang_reasons.get("empty", 0) + 1
            return None
        phi, e = S.phi, S.e
        # the start: the surface the liquid had at the last update, whatever its chord. The liquid moves continuously
        # (5 ms against a millimetre drop's capillary time), so the solve follows its own equilibrium. Starting a moved
        # toe from the height-function face at the cap, another basin of this non-convex problem, made the bulge flip
        # at a strike to a sheet along the plate out to 8-10.6 mm and back in single updates, and the cells the sheet
        # laid froze on the cold plate, a lump of cold lap beside the tack's start
        src_pts = st.pts if st.pts is not None and (same or cont) else \
            A[None, :] + u_cap[:, None]*t[None, :] + h_cap[:, None]*n[None, :]
        d = np.asarray(src_pts, float) - M[None, :]
        ph = np.arctan2(d @ n, d @ t)
        ph = np.where(ph < -.5*math.pi, ph + 2*math.pi, ph)
        rr = np.hypot(d @ t, d @ n)
        o = np.argsort(ph)
        r0 = np.interp(phi, ph[o], rr[o])
        r0[0] = r0[-1] = S.r_toe
        # rays from the midpoint: cell classes along each node's direction, 0.05 mm steps
        R_MAX = .5*L + depth + FAN_REACH + 4e-3          # the rays' length: past the largest overhang seen
        rho_s = np.arange(0., R_MAX, 5e-5)
        P_ = M[None, None, :] + rho_s[None, :, None]*e[:, None, :]           # [N+1, S, 2]
        g = self.g
        jj = np.clip(np.searchsorted(g.ye, P_[..., 0]) - 1, self.j0, self.j1 - 1) - self.j0
        kk = np.clip(np.searchsorted(g.ze, P_[..., 1]) - 1, self.k0, self.k1 - 1) - self.k0
        metal = mat2d[jj, kk] != TH.AIR
        liquid = metal & (fl2d[jj, kk] >= .5)
        # the region the pool may change (Joint.box) is a wall: beyond it the cells cannot take metal, and a surface
        # past it claimed area the cells never held (a 6.5 s parked pool reaching 14 mm along the plate laid 20-44 mm2
        # per slice outside it: the metal fell 93 mm3 behind its deposit unseen). A pool that outgrows the
        # region leaves its excess waiting (pool_unplaced), in view
        y_lo, y_hi, z_lo, z_hi = g.ye[self.j0], g.ye[self.j1], g.ze[self.k0], g.ze[self.k1]
        out_box = (P_[..., 0] < y_lo) | (P_[..., 0] > y_hi - 1e-9) | (P_[..., 1] < z_lo) | (P_[..., 1] > z_hi - 1e-9)
        metal = metal | out_box
        liquid = liquid & ~out_box
        solid = metal & ~liquid
        resol = solid & ((mat2d[jj, kk] == TH.WELD) | (pf2d[jj, kk] >= .5))
        N = S.N
        lo, hi = np.zeros(N + 1), np.full(N + 1, R_MAX)
        fixed = np.zeros(N + 1, bool)
        nS = len(rho_s)
        for q in range(1, N):
            iref = int(np.clip(r0[q]/5e-5, 0, nS - 1))
            row = solid[q]
            if row[iref]:
                # the start radius lies in solid (rounding at a frozen station, a toe's member): the nearest gap,
                # above or below it (searching above only, a node beside a toe whose ray runs on into the member has
                # none, and the slice's share goes to the backlog)
                above = np.nonzero(~row[iref:])[0]
                below = np.nonzero(~row[:iref])[0]
                if not len(above) and not len(below):
                    self.overhang_reasons["buried"] = self.overhang_reasons.get("buried", 0) + 1
                    return None
                da = int(above[0]) if len(above) else nS
                db = iref - int(below[-1]) if len(below) else nS
                iref = iref + da if da <= db else int(below[-1])
                r0[q] = rho_s[iref]
            below = np.nonzero(row[:iref])[0]
            ilo = int(below[-1]) + 1 if len(below) else 0
            beyond = np.nonzero(row[iref:])[0]
            ihi = iref + int(beyond[0]) if len(beyond) else nS
            lo[q] = rho_s[ilo] if ilo > 0 else 0.
            hi[q] = rho_s[ihi] if ihi < nS else R_MAX
            if FREEZE_SURFACE and ilo > 0 and resol[q, ilo - 1] and not liquid[q, ilo:ihi].any():
                fixed[q] = True
                r0[q] = lo[q]
        # the most and least the bounds hold (free nodes at their limits, frozen stations held): a share beyond them is
        # clipped, the rest goes to the other slices
        rf = np.clip(r0, lo, hi)
        rf[0] = rf[-1] = S.r_toe
        a_max = S.area(np.where(fixed, rf, hi))
        a_min = S.area(np.where(fixed, rf, lo))
        saturated = V > a_max*(1 - 1e-6)
        V = float(np.clip(V, a_min*(1 + 1e-6), a_max*(1 - 1e-6)))
        r, lam, ok = S.solve(V, r0, lo, hi, fixed)
        if not ok or not np.all(np.isfinite(r)):
            self.overhang_reasons["infeasible"] = self.overhang_reasons.get("infeasible", 0) + 1
            return None
        pts = (M[None, :] + r[:, None]*e)[::-1]                            # from toe A to toe B
        # touch-down: a node resting on the metal beyond a toe (its upper bound the joint's surface outside the span)
        touch = []
        J = self.joint
        for side, nodes in ((1, range(1, N//2)), (0, range(N - 1, N//2, -1))):
            far = None
            for q in nodes:
                if hi[q] < R_MAX and r[q] >= hi[q] - 1e-7:
                    Pq = M + hi[q]*e[q]
                    uq = float((Pq - A) @ t)
                    if (side == 1 and uq > L) or (side == 0 and uq < 0):
                        far = Pq
            if far is not None:
                sj = np.arange(0., J.length + 1e-9, 5e-5)
                pj, _ = J.point(sj)
                touch.append((side, float(sj[int(np.argmin(np.linalg.norm(pj - far[None, :], axis=1)))])))
        return dict(r=r, pts=pts, area=S.area(r) - tri + corner, pressure=lam, touch=touch, depth=depth,
                    converged=S.converged, saturated=saturated)

    def _contact_freezes(self, n_, side, toes, tem, fl):
        """Can the pool's edge run past this toe onto the metal beyond it? Liquid meeting a solid takes, at the contact,
        the effusivity-weighted mean of the two temperatures, T_c = (e_l T_l + e_s T_s)/(e_l + e_s), e = sqrt(k rho c)
        (two semi-infinite bodies brought into contact); below the solidus the liquid freezes as it touches and the edge
        stays where it is. This criterion is authored for this model [authored]. It is consistent with Tavakoli &
        Kavehpour (ISCST 2012, read: at high Stefan number and low flow rate "instantaneous solidification ... at the
        moment of contact", the base then "frozen ... despite the continuing discharge of liquid"); it is not Schiaffino &
        Sonin's (1997) arrest at an apparent contact angle set by the Stefan number, whose wax and alkane data stop far
        below steel's S ~ 4 on 20 C plate. Past a
        frozen edge the liquid stands as an overhanging bulge (_overhang). The liquid's values are the ledger's
        molecular ones (31 W/m K,
        STEEL_RHO_LIQ, STEEL_CP_LIQ: not the enhanced conductivity that stands in for convection), the solid's
        EN 1993-1-2 k(T), c(T) at its temperature; T_l is the hottest liquid within 1 mm of the toe, T_s the metal's
        temperature 0.1 mm under the surface 0.1 mm beyond the toe. For steel on steel e_l / e_s is 0.93-1.12, so the
        edge runs only onto metal within about the liquid's superheat of the solidus: onto metal the arc has already
        heated near melting, never onto cold plate. An edge that runs onto any unfused metal lets a tack's cold start
        spread a 9-12 mm skirt of cold lap over the cold plate. Returns (frozen?, (y, z) of the landing point just
        beyond the toe, in the air)."""
        from .thermal import _steel_cp, _steel_k
        J, g = self.joint, self.g
        s_toe = toes[side]
        s_out = float(np.clip(s_toe + (-1. if side == 0 else 1.)*2*TOE_SAMPLE, 0., J.length))
        p, nrm = J.point(np.array([s_out]))
        q = p[0] - nrm[0]*1e-4
        jj = int(np.clip(np.searchsorted(g.ye, q[0]) - 1, self.j0, self.j1 - 1) - self.j0)
        kk = int(np.clip(np.searchsorted(g.ze, q[1]) - 1, self.k0, self.k1 - 1) - self.k0)
        t_s = float(tem[n_, jj, kk])
        pt, _ = J.point(np.array([s_toe]))
        liq = fl[n_] >= .5
        near = liq & ((self.YY - pt[0, 0])**2 + (self.ZZ - pt[0, 1])**2 <= 1e-6)
        if near.any():
            t_l = float(tem[n_][near].max())
        elif liq.any():
            t_l = float(tem[n_][liq].max())
        else:
            return True, tuple(p[0] + nrm[0]*2.5e-4)
        e_l = math.sqrt(float(_steel_k(np.array([PR.v(PR.STEEL_LIQUIDUS)]))[0])*PR.v(PR.STEEL_RHO_LIQ)*PR.v(PR.STEEL_CP_LIQ))
        e_s = math.sqrt(float(_steel_k(np.array([t_s]))[0])*PR.v(PR.STEEL_RHO)*float(_steel_cp(np.array([t_s]))[0]))
        t_c = (e_l*t_l + e_s*t_s)/(e_l + e_s)
        return t_c < PR.v(PR.STEEL_SOLIDUS), tuple(p[0] + nrm[0]*2.5e-4)

    def _apply(self, rows, prob, sol, h, u, fl, mat, idx):
        """Map the profiles to cells: air under the surface -> weld metal; liquid metal outside it -> air. sol[k]: the
        slice's polar surface (an overhang, _overhang) or None (its height-function face h[k] over u[k])."""
        s, g = self.s, self.g
        new_idx, rem_idx = [], []
        for k, (n_, i) in enumerate(rows):
            A, t, nrm, L = prob[k]["A"], prob[k]["t"], prob[k]["n"], prob[k]["L"]
            Y, Z = self.YY, self.ZZ
            if sol[k] is None:
                under, inside_span, ww = under_surface("h", A, t, nrm, L, Y, Z, u=u[k], h=h[k])
                uu = (Y - A[0])*t[0] + (Z - A[1])*t[1]
                hmin_u = np.interp(np.clip(uu, 0, L), u[k], prob[k]["hmin"])
                above_solid = ww >= hmin_u - 1e-4
                top = .5*float(np.max(h[k]))
            else:
                under, inside_span, ww = under_surface("polar", A, t, nrm, L, Y, Z, r=sol[k]["r"],
                                                       depth=sol[k]["depth"])
                # the polar surface lies on the solid it rests on by its bounds; air under it beneath solid metal (a
                # pocket inside a frozen tack) is not reachable: the connectivity test below removes it
                above_solid = np.ones_like(under)
                top = max(float(np.max(sol[k]["r"])) - sol[k]["depth"], 2e-4)*.5
            m = mat[n_]
            # liquid fills the air between the solid it rests on and its free surface (a groove refills), never air
            # beyond that solid: otherwise a chord along the bracket face can fill the air behind the bracket
            add = under & (m == TH.AIR) & above_solid
            if add.any():
                # and only air the liquid can reach: the air connected, in this section, to the pool's own surface
                # (the chord's middle, raised by half the profile's height). The span test alone lets 1-4 cells per
                # update land behind the bracket's back face at a weld's start (as in _stack)
                air = m == TH.AIR
                lab, nl = ndimage.label(air)
                if nl > 1:
                    P = A + t*L/2 + nrm*max(top, 1e-4)
                    J, K = np.nonzero(air)
                    q = int(np.argmin((Y[J, K] - P[0])**2 + (Z[J, K] - P[1])**2))
                    add &= lab == lab[J[q], K[q]]
            liquid = fl[n_] >= .5
            if sol[k] is None:
                hu = np.interp(np.clip((Y - A[0])*t[0] + (Z - A[1])*t[1], 0, L), u[k], h[k])
                remove = inside_span & (ww > hu + 1e-5) & (m != TH.AIR) & liquid
            else:
                remove = inside_span & (ww >= -sol[k]["depth"]) & ~under & (m != TH.AIR) & liquid
            # liquid weld metal of the pool beyond the toes (an overhang held last update) that this profile does not
            # hold rejoins it
            remove |= prob[k]["pooled_out"] & ~under
            if add.any():
                new_idx.append(idx[n_][add])
                self.dV[i] += float(np.sum(self.cell_area[add]))*g.dx[i]
            if remove.any():
                rem_idx.append(idx[n_][remove])
                self.slices[i].undercut_cells += int((remove & self.orig[idx[n_]]).sum())
                self.dV[i] -= float(np.sum(self.cell_area[remove]))*g.dx[i]
        # liquid that leaves one place (removed cells) reappears in the new cells with its enthalpy (the pool is mixed);
        # only the net growth of the bead is new filler, carried at h_new_weld (the rest of the droplets' enthalpy is
        # the arc's droplet source). New cells share one enthalpy per update.
        if not hasattr(self, "_vols"):
            self._vols = s.g.volumes()
        vols = self._vols
        ni = np.concatenate(new_idx) if new_idx else np.zeros(0, np.int64)
        ri = np.concatenate(rem_idx) if rem_idx else np.zeros(0, np.int64)
        v_new, v_rem = float(np.sum(vols[ni])), float(np.sum(vols[ri]))
        e_rem = 0.
        if len(ri):
            e_rem = float(np.sum(s.gather("H", ri)*vols[ri]))
            s.scatter("mat", ri, np.zeros(len(ri), np.uint8))
            self.dirty.update((np.asarray(ri)//(self.g.ny*self.g.nz)).tolist())
            s.scatter("body", ri, np.zeros(len(ri), np.uint8))
            s.scatter("H", ri, np.zeros(len(ri), np.float32))
        e_new = 0.
        if len(ni):
            moved = min(v_new, v_rem)
            e_moved = e_rem*moved/v_rem if v_rem > 0 else 0.
            # net growth is new filler (droplets at h_new_weld) only while the arc deposits it; with the arc off it is
            # the pool's own metal laid one cell differently (the cells' rounding of a reshaped surface), carrying the
            # pool's liquid enthalpy. (At h_new_weld, liquidus + 30 C, a cell or two of rounding mixed into a parked
            # pool cooling at ~1 500 C raises its hottest cell by up to 19 C with the arc off)
            h_fill = self.h_new_weld
            if not getattr(self, "_arc_on", True):
                if v_rem > 0:
                    h_fill = e_rem/v_rem
                else:
                    liq_idx = np.concatenate([idx[n_][(fl[n_] >= .5) & (mat[n_] != TH.AIR)] for n_, _ in rows]) \
                        if rows else np.zeros(0, np.int64)
                    if len(liq_idx):
                        h_fill = float(np.mean(s.gather("H", liq_idx)))
            e_new = e_moved + (v_new - moved)*h_fill
            h_cells = e_new/v_new
            s.scatter("mat", ni, np.full(len(ni), TH.WELD, np.uint8))
            self.dirty.update((np.asarray(ni)//(self.g.ny*self.g.nz)).tolist())
            s.scatter("body", ni, np.full(len(ni), TH.BODY_WELD, np.uint8))
            s.scatter("H", ni, np.full(len(ni), h_cells, np.float32))
            e_new = float(h_cells*v_new)            # float32 storage: book what was written
        if e_new or e_rem:
            led = s.w_ledger.numpy()
            led[4] += e_new
            led[5] += e_rem
            s.w_ledger.assign(led)
        return dict(new_j=e_new, removed_j=e_rem, slices=len(rows))

    def _stack_base(self, drop):
        """Where the droplets land under this stack: down the wire axis from the arc root, the first metal that is not
        the stack's own (the joint, a tack, an earlier bead). The arc root itself if there is no axis or no such
        metal in the grid."""
        base = np.asarray(drop, float)
        if self.drop_axis is None or not self.stack_cells:
            return base
        g = self.g
        a = np.asarray(self.drop_axis, float)
        a = a/max(float(np.linalg.norm(a)), 1e-12)
        P = base[None, :] + np.arange(0., 40e-3, 5e-5)[:, None]*a[None, :]
        ii = np.searchsorted(g.xe, P[:, 0]) - 1
        jj = np.searchsorted(g.ye, P[:, 1]) - 1
        kk = np.searchsorted(g.ze, P[:, 2]) - 1
        ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny) & (kk >= 0) & (kk < g.nz)
        n = int(np.argmin(ok)) if not ok.all() else len(P)
        if n == 0:
            return base
        flat = ((ii[:n]*g.ny + jj[:n])*g.nz + kk[:n]).astype(np.int64)
        hit = (self.s.mat[flat] != TH.AIR) & ~np.isin(flat, np.fromiter(self.stack_cells, np.int64))
        return P[int(np.argmax(hit))] if hit.any() else base

    def _stack(self, volume, drop, x_lim=None):
        """Filler with no pool to land in (the arc on metal it has not melted): droplets stack where they land, in
        the air next to metal nearest the arc root, as liquid weld metal; on cold metal they freeze there (cold lap).
        Their volume is counted from the cells (dV) like any slice's metal. x_lim: (x0, x1) the cells must lie in (a
        pool's overflow stays inside its liquid run)."""
        s, g = self.s, self.g
        base = self._stack_base(drop)
        # candidates within STACK_REACH along the weld: the stack's shape comes from its dome law below, not from the
        # window (a +-3 mm window gives the cold starts flat x faces: 5 mm box-shaped tack starts)
        lo, hi = base[0] - STACK_REACH, base[0] + STACK_REACH
        if x_lim is not None:
            lo, hi = max(lo, x_lim[0]), min(hi, x_lim[1])
        i0 = int(np.clip(np.searchsorted(g.xe, lo + 1e-9) - 1, 0, g.nx - 1))
        i1 = int(np.clip(np.searchsorted(g.xe, hi - 1e-9), i0 + 1, g.nx))
        sl = np.arange(i0, i1)
        idx = self._box_indices(sl)
        m = s.mat[idx.ravel()].reshape(idx.shape) != TH.AIR
        nb = np.zeros_like(m)
        nb[:, 1:, :] |= m[:, :-1, :]
        nb[:, :-1, :] |= m[:, 1:, :]
        nb[:, :, 1:] |= m[:, :, :-1]
        nb[:, :, :-1] |= m[:, :, 1:]
        c = ~m & nb
        # only air the liquid can reach: in each section, the air connected to where the droplets land (air behind the
        # bracket is never reachable through the 8 mm of steel)
        jb = int(np.argmin(np.abs(self.YY[:, 0] - base[1])))
        kb = int(np.argmin(np.abs(self.ZZ[0, :] - base[2])))
        for n_ in range(len(sl)):
            air = ~m[n_]
            if not air.any():
                continue
            lab, nl = ndimage.label(air)
            if nl <= 1:
                continue
            if air[jb, kb]:
                keep = lab[jb, kb]
            else:
                J, K = np.nonzero(air)
                keep = lab[J[np.argmin((J - jb)**2 + (K - kb)**2)], K[np.argmin((J - jb)**2 + (K - kb)**2)]]
            c[n_] &= lab == keep
        if not c.any():
            return dict(new_j=0., removed_j=0., slices=0)
        X = g.xc[sl][:, None, None]
        D = np.stack(np.broadcast_arrays(X - base[0], self.YY[None] - base[1], self.ZZ[None] - base[2]), -1)[c]
        d2 = np.sum(D*D, 1)
        # droplets landing on cold steel freeze where they land: the stack grows as a compact dome about the surface
        # they land on (base, not the arc root: the root sits on the stack's own top, and a dome measured from it climbs
        # the wire as a 20 mm liquid spike at a 30 mm stand-off), equally in every direction.
        vol = (self.cell_area[None]*g.dx[sl][:, None, None]*np.ones_like(m, float))[c]
        cells = idx[c]
        which = np.broadcast_to(sl[:, None, None], m.shape)[c]
        order = np.argsort(d2)
        take = order[:int(np.searchsorted(np.cumsum(vol[order]), volume)) + 1]
        ni = cells[take]
        s.scatter("mat", ni, np.full(len(ni), TH.WELD, np.uint8))
        self.dirty.update((np.asarray(ni)//(self.g.ny*self.g.nz)).tolist())
        s.scatter("body", ni, np.full(len(ni), TH.BODY_WELD, np.uint8))
        s.scatter("H", ni, np.full(len(ni), self.h_new_weld, np.float32))
        np.add.at(self.dV, which[take], vol[take])
        self.stack_cells.update(int(v) for v in ni)
        e = float(np.sum(vol[take])*self.h_new_weld)
        led = s.w_ledger.numpy()
        led[4] += e
        s.w_ledger.assign(led)
        self.stacked_m3 += float(np.sum(vol[take]))
        return dict(new_j=e, removed_j=0., slices=0, stacked=len(ni))

    # ---------------------------------------------------------------- checkpoints
    def snapshot(self):
        import copy
        keep = ("frozen_volume", "liquid_expansion", "overflow_events", "p_pool", "root_y", "excess", "stacked_m3", "spilled_m3", "obstacle_bounded", "arrested_m3", "arrest_events", "overhang_m3", "overhang_events", "overhang_failures", "touchdown_events", "overhang_unconverged", "overhang_reasons")
        return dict(slices=copy.deepcopy(self.slices), dV=self.dV.copy(), events=list(self.events),
                    stack_cells=sorted(self.stack_cells), last_drop=copy.deepcopy(getattr(self, "last_drop", None)),
                    **{k: getattr(self, k, None) for k in keep})

    def restore(self, snap):
        import copy
        self.slices = copy.deepcopy(snap["slices"])
        self.dV = snap["dV"].copy()
        self.events = list(snap["events"])
        self.stack_cells = set(snap.get("stack_cells", ()))
        self.last_drop = copy.deepcopy(snap.get("last_drop"))
        for k in ("frozen_volume", "liquid_expansion", "overflow_events", "p_pool", "root_y", "excess", "stacked_m3", "spilled_m3", "obstacle_bounded", "arrested_m3", "arrest_events", "overhang_m3", "overhang_events", "overhang_failures", "touchdown_events", "overhang_unconverged", "overhang_reasons"):
            setattr(self, k, snap.get(k, 0.))

    # ---------------------------------------------------------------- results
    def profiles(self):
        """Frozen and liquid slice profiles for the evaluator and the render: list of (i, state)."""
        return [(i, st) for i, st in enumerate(self.slices) if st.h is not None]

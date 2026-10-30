"""Implicit contact world for paper packaging: shells, rods and affine rigid bodies under IPC contact.

Everything that is not a robot lives here and is advanced together by one implicit-Euler step,
minimising the incremental potential

    E(q) = 1/2 |q - q~|_M^2 / dt^2  - f_g . q                        inertia, gravity
         + E_membrane + E_bend + E_crease                         paper shells (orthotropic StVK, hinges)
         + E_rod + E_body + E_joint + E_bond + E_film + E_drag    rods, affine bodies, pins/hinges, tape,
                                                                  lid air film, air drag
         + kappa_b B(x) + F(x)                                     IPC barrier contact + lagged friction
                                                                  (ipctk 1.6), x = P q + c

with Newton's method, a sparse direct solve (MKL Pardiso) and a backtracking line search limited by
additive continuous collision detection, so no sheet, rod or body can pass through another.

Degrees of freedom q: shell/rod vertex positions, then 12 affine coordinates (p, A) per non-static
body (x = A r + p, r relative to the body's centre of mass; Lan et al. 2022, "Affine Body Dynamics").
Static bodies (furniture) are fixed geometry. Driven bodies are robot links: they carry no inertia
and are pulled to the pose the robot simulator reports by a stiff spring (the link surface
compliance), so the contact forces on them are well defined and returned as reactions. A finger's
closing coordinate is instead governed by the hand: a force-capped spring toward the commanded finger
position (Franka Hand force/position semantics), so a pinch presses the sheets with the commanded force.

Paper tears by vertex splitting (IPCWorld._tear_post): a crack tip advances along a mesh edge when the virtual crack
closure energy release rate reaches the stock's tear energy (mode-mixed between in-plane Gamma_I and out-of-plane
Gamma_III); the two copies of the split vertex are held by an extrinsic cohesive bond that dissipates exactly Gamma per
crack length as it opens. Initiation where no crack exists is a strength test (membrane force against X; at authored
free edges also the out-of-plane pull against Gamma_III), interior holes nucleate in a crack band. Spare vertices are
preallocated per tearable sheet, so vertex ids and sheet slices never shift.

Nothing here moves a vertex except these forces: no attachment, teleport or kinematic grasp.
"""

from dataclasses import dataclass, field
import math
import os
import time

# Thread budget: this container's cgroup quota is 16 CPUs (cpu.max) although nproc reports 192. MKL and
# OpenBLAS default to one thread per visible core. Measured on the wrapped present at rest:
# 0.36 s of linear solves per contact step with no cap against 0.12-0.14 s at 2-8 threads.
SOLVER_THREADS = int(os.environ.get("GIFT_BAGS_THREADS", "4"))   # 4: the setting every runtime figure was measured at
for _k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_k, str(SOLVER_THREADS))
# Idle pool threads sleep instead of spinning. With the OpenMP default (active waiting) one counter session with a
# budget of 4 threads used 8.2 CPUs on average: each pool's idle workers spin between parallel regions,
# and several pools exist (MKL/PARDISO, OpenBLAS, the IPC Toolkit's TBB). With a passive wait policy a session uses
# about its budget (3.4-3.5 CPUs at 4 and at 3 threads).
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("KMP_BLOCKTIME", "0")
import numpy as np
try:
    # the variables above only reach BLAS if numpy was not imported yet; when a script imports numpy first, numpy's
    # OpenBLAS sizes itself to the machine (64 threads, ~9 CPUs busy-waiting on a 16-CPU quota)
    from threadpoolctl import threadpool_limits
    threadpool_limits(SOLVER_THREADS)
except ImportError:
    pass
import scipy.sparse as sp

G = 9.81
CCD_FRACTION = .5           # CCD separation as a fraction of dmin (barrier enforces dmin; see ContactSet.ccd)
STICK_FULL = .1             # a collision vertex keeps its whole stick displacement while it moves slower than this
                            # fraction of the friction transition velocity, none above the transition velocity itself,
                            # smoothly between (IPCWorld._stick_update) [A]
# A Newton step can end the iteration only if it changes no collision vertex's contact force (barrier + friction) by
# more than FORCE_RTOL x the largest contact force on any vertex, or FORCE_ATOL (N): IPCWorld._contact_forces_settled.
# 5 % [C]: what is bounded is the LAST update's change, and the error left after it is of
# second order. On the block bench the effective friction is 0.3500 for 0.35 at 1 % and at 5 %, 0.3500-0.3508 at
# 10 %, up to 2 % high at 20 %, and at 30 % the block no longer starts to slide at 20 degrees.
# DIAGNOSTIC switches: GIFT_BAGS_FORCE_RULE=0 is the step-size rule alone;
# GIFT_BAGS_FORCE_RTOL sets another fraction.
FORCE_RTOL = (None if os.environ.get("GIFT_BAGS_FORCE_RULE", "1") == "0"
              else float(os.environ.get("GIFT_BAGS_FORCE_RTOL", "0.05")))
FORCE_ATOL = 1e-6
CCD_MAX_ITERATIONS = 100000
GAP_FLOOR = 1e-12            # m, smallest accepted pair gap beyond dmin (see _gaps_ok)
AFFINE_LEVER = .1            # m, lever converting a driven body's force cap into its affine (rotation) cap
WARM_START = os.environ.get("GIFT_BAGS_WARM_START", "1") != "0"   # diagnostic switch (A/B timing); same solution
FINGER_MASS = .0291          # kg, FR3/Franka Hand finger link (URDF inertial; label_application PAD_SOLREF_REF_MASS)
AIR_RHO = 1.204          # kg/m^3, 20 C [handbook]
AIR_MU = 1.81e-5         # Pa s, 20 C [handbook]
# tearing by vertex splitting (IPCWorld._tear_post)
TEAR_D0 = 1e-6           # m, opening at which a fresh tear bond carries its initial traction (stiff extrinsic onset)
TEAR_CONE_DEG = 75.      # a running crack may continue along edges within this angle of its current direction (rest)
TEAR_SCREEN = .8         # strength-index screen for initiation candidates (exact edge test follows)
TEAR_EDGE_LC = True      # strength tests average the membrane force over the stock's length l_c (False: one element)
TEAR_EDGE_T3 = 1.05      # out-of-plane edge initiation: pull between the two sides over Gamma_III. The traction at which
                         # a running out-of-plane tear's cohesive bond is inserted, T0/Gamma_III = 1.02-1.12 on the trouser
                         # bench at 8 mm [C]; that an uncut edge starts at it is [E]
RELEASE_EVERY = 4        # steps between attempts to re-enable contact between separated sibling copies


# ----------------------------------------------------------------------------- materials
@dataclass(frozen=True)
class ShellMat:
    """Paper-like shell. Membrane and bending are per width (N/m, N m); strengths per width (N/m)."""
    name: str
    rho_A: float               # kg/m^2
    t: float                   # m, caliper (contact thickness and render thickness)
    Et_md: float               # N/m, membrane stiffness E t along MD
    Et_cd: float               # N/m
    nu12: float
    Gt12: float                # N/m, in-plane shear stiffness G12 t
    D_md: float                # N m, bending stiffness per width
    D_cd: float
    yield_strain: float = .005           # bending yield strain (permanent set of uncreased stock)
    X_md: float = 0.                     # N/m tensile strength; 0 = never damages
    X_cd: float = 0.
    Gamma: float = 0.                    # J/m, out-of-plane (mode III, trouser/Elmendorf) tear energy per crack length
                                         # for a crack running along MD; > 0 makes the stock tearable
    Gamma_I: float = 0.                  # J/m, in-plane (mode I) fracture energy per crack length; 0: use Gamma
    tear_cd: float = 1.                  # tear energy of a crack running along CD / along MD (both modes)
    drag_cd: float = 1.17                # flat plate broadside [handbook]

    @property
    def gamma_in_plane(self):
        return self.Gamma_I if self.Gamma_I > 0. else self.Gamma
    color: tuple = (.8, .8, .8)

    def membrane_C(self, angle=0.):
        """Plane-stress orthotropic stiffness (Voigt, per width, N/m) rotated so MD is at `angle`
        from the rest x axis."""
        nu21 = self.nu12*self.Et_cd/self.Et_md
        den = 1. - self.nu12*nu21
        Q = np.array([[self.Et_md/den, self.nu12*self.Et_cd/den, 0.],
                      [self.nu12*self.Et_cd/den, self.Et_cd/den, 0.],
                      [0., 0., self.Gt12]])
        if angle == 0.:
            return Q
        c, s = math.cos(angle), math.sin(angle)
        # strain transformation (engineering shear) from global to material axes
        T = np.array([[c*c, s*s, c*s], [s*s, c*c, -c*s], [-2*c*s, 2*c*s, c*c - s*s]])
        return T.T @ Q @ T


@dataclass(frozen=True)
class CreaseLaw:
    """Elasto-viscoplastic crease hinge (per unit crease length), two inelastic mechanisms in series:
    m = k (theta - theta_p),   theta_p = theta_1 + theta_r
    d theta_1/dt = omega0 sinh(m / m_a)                       permanent set (thermally activated flow)
    d theta_r/dt = omega2 sinh((m - k2 theta_r) / m2)         recoverable set held by a back spring k2
    Gives logarithmic relaxation at a held angle, a rate-dependent fold moment, the release angle growing with
    the hold, and the logarithmic creep recovery of a released fold, as measured on creased paperboard
    (Nagasawa, Kaneko & Adachi 2019). k2 = 0 switches the recoverable part off."""
    name: str
    k: float          # N m / m / rad, elastic crease stiffness
    m_a: float        # N m / m, activation moment of the permanent set
    omega0: float     # rad/s, reference flow rate of the permanent set
    k2: float = 0.    # N m / m / rad, back spring of the recoverable set
    m2: float = 1.    # N m / m, activation moment of the recoverable set
    omega2: float = 0.  # rad/s, reference flow rate of the recoverable set


@dataclass(frozen=True)
class RodMat:
    name: str
    radius: float     # m
    lin_rho: float    # kg/m
    EA: float         # N
    EI: float         # N m^2
    color: tuple = (.9, .9, .9)
    # the rod's own surface compliance in contact: it can be squeezed by up to contact_range (m) across its radius,
    # with barrier stiffness contact_kappa. None: the world's paper caliper range and stiffness (a rod as hard as the
    # paper's fibre core, compressible over tens of micrometres only)
    contact_range: float = None
    contact_kappa: float = None


@dataclass(frozen=True)
class CohesiveLaw:
    """Bilinear traction-separation law per unit area (tape on a substrate)."""
    name: str
    k: float          # Pa/m, initial stiffness (adhesive layer)
    Gamma: float      # J/m^2, fracture energy
    sigma: float      # Pa, strength

    def __post_init__(self):
        # a bilinear law needs damage onset before full separation (sigma^2 < 2 k Gamma); otherwise the bond is an
        # unbreakable spring up to delta0 that stores far more than Gamma and then snaps
        if self.sigma/self.k >= 2.*self.Gamma/self.sigma:
            raise ValueError(f"{self.name}: inconsistent cohesive law (delta0 {self.sigma/self.k:.3g} m >= deltaf "
                             f"{2*self.Gamma/self.sigma:.3g} m; need sigma < sqrt(2 k Gamma) = {(2*self.k*self.Gamma)**.5:.3g} Pa)")

    @property
    def delta0(self):
        return self.sigma/self.k

    @property
    def deltaf(self):
        return 2.*self.Gamma/self.sigma


# ----------------------------------------------------------------------------- geometry helpers
def edges_of(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(e, 1), axis=0)


def hinge_angles(x, H):
    """Signed dihedral angle of each hinge (x3,x4 edge; x1,x2 opposite) and its gradient (H,4,3)
    (label_application.labelsim formulation)."""
    x3, x4, x1, x2 = x[H[:, 0]], x[H[:, 1]], x[H[:, 2]], x[H[:, 3]]
    E = x4 - x3
    le = np.linalg.norm(E, axis=1)
    N1 = np.cross(x1 - x3, x1 - x4)
    N2 = np.cross(x2 - x4, x2 - x3)
    n1s, n2s = np.maximum((N1*N1).sum(1), 1e-30), np.maximum((N2*N2).sum(1), 1e-30)
    n1, n2 = N1/np.sqrt(n1s)[:, None], N2/np.sqrt(n2s)[:, None]
    theta = np.arctan2((np.cross(n1, n2)*E).sum(1)/np.maximum(le, 1e-30), (n1*n2).sum(1))
    u1 = le[:, None]*N1/n1s[:, None]
    u2 = le[:, None]*N2/n2s[:, None]
    u3 = (((x1-x4)*E).sum(1)/le)[:, None]*N1/n1s[:, None] + (((x2-x4)*E).sum(1)/le)[:, None]*N2/n2s[:, None]
    u4 = -(((x1-x3)*E).sum(1)/le)[:, None]*N1/n1s[:, None] - (((x2-x3)*E).sum(1)/le)[:, None]*N2/n2s[:, None]
    return theta, -np.stack([u3, u4, u1, u2], 1)


def crease_return(law, d, tr, dt):
    """Implicit return mapping of CreaseLaw over one step at a fixed hinge angle: d = theta - theta_p (elastic
    trial), tr = recoverable set. Returns the increments (s1 permanent, s2 recoverable); Newton on 2 unknowns."""
    k, k2 = law.k, law.k2
    two = law.omega2 > 0.
    s1 = s2 = 0.
    for _ in range(60):
        m = k*(d - s1 - s2)
        a1 = max(min(m/law.m_a, 60.), -60.)
        r1 = s1 - dt*law.omega0*math.sinh(a1)
        c1 = dt*law.omega0*math.cosh(a1)/law.m_a
        if two:
            a2 = max(min((m - k2*(tr + s2))/law.m2, 60.), -60.)
            r2 = s2 - dt*law.omega2*math.sinh(a2)
            c2 = dt*law.omega2*math.cosh(a2)/law.m2
            j11, j12, j21, j22 = 1. + c1*k, c1*k, c2*k, 1. + c2*(k + k2)
            det = j11*j22 - j12*j21
            d1, d2 = (-r1*j22 + r2*j12)/det, (-r2*j11 + r1*j21)/det
        else:
            d1, d2 = -r1/(1. + c1*k), 0.
        s1 += d1; s2 += d2
        if abs(d1) + abs(d2) < 1e-12:
            break
    return s1, s2


def _libm(f, a):
    """math.sinh/cosh elementwise: numpy's SIMD sinh/cosh differ from libm in the last bit on ~9 % of arguments, and
    the batched return mapping below must reproduce crease_return bit for bit."""
    return np.fromiter(map(f, a.tolist()), float, len(a))


def crease_return_batch(k, k2, m_a, omega0, m2, omega2, d, tr, dt):
    """crease_return for many hinges at once (one law per hinge, parameters as arrays): every hinge runs the same
    floating-point operations in the same order as the scalar function and stops at the same iteration, so the
    results are identical to calling crease_return hinge by hinge."""
    n = len(d)
    s1, s2 = np.zeros(n), np.zeros(n)
    two = omega2 > 0.
    dto0, dto2 = dt*omega0, dt*omega2
    act = np.arange(n)
    for _ in range(60):
        if not len(act):
            break
        kk = k[act]
        m = kk*(d[act] - s1[act] - s2[act])
        a1 = np.maximum(np.minimum(m/m_a[act], 60.), -60.)
        r1 = s1[act] - dto0[act]*_libm(math.sinh, a1)
        c1 = dto0[act]*_libm(math.cosh, a1)/m_a[act]
        d1 = -r1/(1. + c1*kk)
        d2 = np.zeros(len(act))
        tw = two[act]
        if tw.any():
            j, aj = np.flatnonzero(tw), act[tw]
            a2 = np.maximum(np.minimum((m[j] - k2[aj]*(tr[aj] + s2[aj]))/m2[aj], 60.), -60.)
            r2 = s2[aj] - dto2[aj]*_libm(math.sinh, a2)
            c2 = dto2[aj]*_libm(math.cosh, a2)/m2[aj]
            kj = kk[j]
            j11, j12, j21, j22 = 1. + c1[j]*kj, c1[j]*kj, c2*kj, 1. + c2*(kj + k2[aj])
            det = j11*j22 - j12*j21
            d1[j] = (-r1[j]*j22 + r2*j12)/det
            d2[j] = (-r2*j11 + r1[j]*j21)/det
        s1[act] += d1
        s2[act] += d2
        act = act[~(np.abs(d1) + np.abs(d2) < 1e-12)]
    return s1, s2


def hinge_theta(x, H):
    """The signed dihedral angle of hinge_angles without its gradient (same operations, identical values)."""
    x3, x4, x1, x2 = x[H[:, 0]], x[H[:, 1]], x[H[:, 2]], x[H[:, 3]]
    E = x4 - x3
    le = np.linalg.norm(E, axis=1)
    N1 = np.cross(x1 - x3, x1 - x4)
    N2 = np.cross(x2 - x4, x2 - x3)
    n1s, n2s = np.maximum((N1*N1).sum(1), 1e-30), np.maximum((N2*N2).sum(1), 1e-30)
    n1, n2 = N1/np.sqrt(n1s)[:, None], N2/np.sqrt(n2s)[:, None]
    return np.arctan2((np.cross(n1, n2)*E).sum(1)/np.maximum(le, 1e-30), (n1*n2).sum(1))


def _scatter3(idx, val, n):
    """Row-wise scatter-add of (m,3) values onto n rows: np.add.at(zeros, idx, val) with the same summation order (each
    row accumulates its contributions in input order from 0.0), at bincount speed."""
    out = np.empty((n, 3))
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=val[:, c], minlength=n)
    return out


def psd2(a, b, c):
    """PSD projection of symmetric 2x2 matrices [[a, c], [c, b]] (eigenvalue clamp) in closed form, (n,2,2): the
    projection is the matrix itself if both eigenvalues are >= 0, zero if both are <= 0, and l+ (A - l- I)/(l+ - l-)
    (l+ times the projector onto the positive eigenvector) otherwise."""
    h = .5*(a + b)
    r = np.sqrt((.5*(a - b))**2 + c*c)
    lp, lm = h + r, h - r
    out = np.empty((len(a), 2, 2))
    out[:, 0, 0], out[:, 1, 1], out[:, 0, 1] = a, b, c
    out[:, 1, 0] = c
    neg = lp <= 0.
    out[neg] = 0.
    mix = (lm < 0.) & (lp > 0.)
    if mix.any():
        s = (lp[mix]/(lp[mix] - lm[mix]))
        out[mix, 0, 0] = (a[mix] - lm[mix])*s
        out[mix, 1, 1] = (b[mix] - lm[mix])*s
        out[mix, 0, 1] = out[mix, 1, 0] = c[mix]*s
    return out


def wrap(a):
    """Angle difference wrapped to (-pi, pi]: a hinge folded flat sits at +-pi."""
    return (a + np.pi) % (2*np.pi) - np.pi


_E9 = np.eye(9).reshape(9, 3, 3)                    # unit 3x3 matrices E_c, c = 3 i + j
_E9T = _E9.transpose(0, 2, 1).copy()
_RR9, _CC9 = (np.meshgrid(np.arange(9) + 3, np.arange(9) + 3, indexing="ij")[k].ravel() for k in range(2))


def psd_project(H):
    """Batch symmetric PSD projection (eigenvalue clamp)."""
    w, V = np.linalg.eigh(.5*(H + np.swapaxes(H, -1, -2)))
    return np.matmul(V*np.maximum(w, 0.)[..., None, :], np.swapaxes(V, -1, -2))


def local_coords(sheet):
    """Per-triangle rest coordinates (T,3,2) in a frame whose +x is the paper's machine direction (MD).
    2D nets: X2 rotated by -md_angle. 3D rest shapes (stitched nets such as a bag): each triangle's own
    plane, with MD the projection of sheet.md_dir (per triangle (T,3) or one vector)."""
    X = np.asarray(sheet.X2, float)
    T = np.asarray(sheet.tris)
    if X.shape[1] == 2:
        c, s_ = math.cos(sheet.md_angle), math.sin(sheet.md_angle)
        Rm = np.array([[c, s_], [-s_, c]])
        return X[T] @ Rm.T
    P = X[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    md = np.broadcast_to(np.asarray(sheet.md_dir, float), n.shape).copy()
    e1 = md - (md*n).sum(1, keepdims=True)*n
    bad = np.linalg.norm(e1, axis=1) < 1e-6
    if bad.any():
        alt = np.cross(n[bad], [0., 0., 1.])
        alt[np.linalg.norm(alt, axis=1) < 1e-6] = [1., 0., 0.]
        e1[bad] = alt
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    d = P - P[:, :1]
    return np.stack([(d*e1[:, None]).sum(2), (d*e2[:, None]).sum(2)], 2)


CELLULOSE_DENSITY = 1500.      # kg/m3, cell-wall density: the limit a paper sheet compresses toward [T]
ZD_P_REF, ZD_EPS_REF = 1e6, .05  # paper through-thickness compression: 5 % of caliper at 1 MPa [E, REPORT §contact]


def _bprime(x, xh):
    return -2*(x - xh)*math.log(x/xh) - (x - xh)**2/x


def contact_caliper(stock):
    """Contact geometry of a paper stock, used by the whole contact world.

    The caliper t splits into an incompressible fibre core (fraction phi = sheet density / cellulose density)
    and a compressible remainder. Two sheets' midsurfaces can never come closer than the core (dmin = phi t);
    the barrier acts over the compressible part (dhat = (1 - phi) t), so an unloaded stack sits at exactly one
    caliper per layer, and its stiffness kappa is calibrated so a 1 MPa press compresses the caliper by 5 %
    (ZD_P_REF, ZD_EPS_REF). Rigid surfaces are deflated by phi t/2, the sheet's own core half.
    Rest distances (barrier just engaged): sheet-sheet t, sheet-rigid (1 - phi/2) t, rigid-rigid (1 - phi) t."""
    t = float(stock.t)
    phi = min(.95, max(.2, stock.rho_A/t/CELLULOSE_DENSITY))
    dmin, dhat = phi*t, (1. - phi)*t
    d_ref = (1. - ZD_EPS_REF)*t
    kappa = ZD_P_REF/(-_bprime(d_ref**2 - dmin**2, t*t - dmin**2)*2*d_ref)
    return dict(stock=stock.name, t=t, h_s=t/2, phi=phi, dmin=dmin, dhat=dhat, kappa=kappa, deflate=phi*t/2,
                rest_sheet_sheet=t, rest_sheet_rigid=(1. - phi/2)*t, rest_rigid_rigid=(1. - phi)*t)


def finger_axis_minimum(mf, s_pred, kf, s_cmd, F):
    """Minimiser of 1/2 mf (s - s_pred)^2 + Huber(kf, cap F)(s - s_cmd): a finger's closing coordinate with its axial
    inertia and its force-capped drive (the axial terms of IPCWorld._bodies_energy), in closed form."""
    s = (mf*s_pred + kf*s_cmd)/(mf + kf)
    if kf*(s - s_cmd) > F:
        s = s_pred - F/mf
    elif kf*(s - s_cmd) < -F:
        s = s_pred + F/mf
    return s


def huber3(d, k, cap):
    """Isotropic spring 1/2 k |d|^2 that turns linear (constant force `cap`) beyond |d| = cap/k.
    Returns energy, gradient and the Newton matrix k I in both branches: the exact Hessian of the linear
    branch is (cap/|d|)(I - u u^T), nearly singular, and Newton then proposes metre-scale moves of every
    capped robot link. Overestimating the curvature
    keeps the step at |g|/k <= cap/k: slower but monotone, and the energy is still convex."""
    n = float(np.linalg.norm(d))
    if n <= cap/k:
        return .5*k*n*n, k*d, k*np.eye(3)
    return cap*n - .5*cap*cap/k, cap*d/n, k*np.eye(3)


def deflate(V, F, h):
    """Offset a closed surface inward by h (each vertex moves so every incident face plane moves h inward;
    least squares over the vertex's distinct face normals; exact for boxes)."""
    if h <= 0 or not len(F):
        return V.copy()
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    out = V.copy()
    inc = [[] for _ in range(len(V))]
    for f, tri in enumerate(F):
        for v in tri:
            inc[v].append(f)
    for v, fs in enumerate(inc):
        if not fs:
            continue
        N = n[fs]
        U = [N[0]]
        for m in N[1:]:
            if all(abs(m @ u) < .999 for u in U):
                U.append(m)
        U = np.asarray(U)
        d, *_ = np.linalg.lstsq(U, -h*np.ones(len(U)), rcond=None)
        out[v] = V[v] + d
    return out


def box_mesh(half, center=(0., 0., 0.), n=(1, 1, 1)):
    """Closed box surface (triangles, outward), subdivided n per axis."""
    hx, hy, hz = half
    c = np.asarray(center, float)
    verts, faces = [], []

    def quad_grid(origin, u, v, nu, nv):
        base = len(verts)
        for i in range(nu + 1):
            for j in range(nv + 1):
                verts.append(origin + u*i/nu + v*j/nv)
        for i in range(nu):
            for j in range(nv):
                a = base + i*(nv + 1) + j
                b, cc, d = a + nv + 1, a + nv + 2, a + 1
                faces.extend([(a, b, cc), (a, cc, d)])
    X, Y, Z = np.array([2*hx, 0, 0.]), np.array([0, 2*hy, 0.]), np.array([0, 0, 2*hz])
    o = c - np.array([hx, hy, hz])
    quad_grid(o, Y, X, n[1], n[0])                       # -z
    quad_grid(o + Z, X, Y, n[0], n[1])                   # +z
    quad_grid(o, X, Z, n[0], n[2])                       # -y
    quad_grid(o + Y, Z, X, n[2], n[0])                   # +y
    quad_grid(o, Z, Y, n[2], n[1])                       # -x
    quad_grid(o + X, Y, Z, n[1], n[2])                   # +x
    V = np.asarray(verts)
    F = np.asarray(faces, np.int64)
    # weld duplicate corner/edge vertices
    key = np.round(V/1e-9).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.ravel()[F]


# ----------------------------------------------------------------------------- model parts

# ----------------------------------------------------------------------------- crumpled paper (slack)
SLACK_MIN_STRETCH = .5      # a crumpled patch hides at most half of its length [E: one pleat (a Z-fold) hides two thirds,
                            # creped papers 20-50 %]; beyond it the patch is solid again
CRUMPLE_RESIDUAL = .1       # a triangle that has begun to wrinkle goes on wrinkling at this fraction of the stock's yield
                            # strain [E]: a wrinkle, once formed, deepens by folding, at a small part of the load that
                            # started it. Without it three inner layers of the tote's fold resist the outer one
                            # together at their full crushing load (3 x 4.5 kN/m against its 7.9 kN/m strength) and
                            # the outer layer's first row tears under the press


def _strain_rot(phi):
    """Voigt strain transformation (engineering shear) to axes rotated by phi: e' = T e. T(-phi) is its inverse."""
    c, s = np.cos(phi), np.sin(phi)
    T = np.zeros((len(phi), 3, 3))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2] = c*c, s*s, c*s
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2] = s*s, c*c, -c*s
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2] = -2*c*s, 2*c*s, c*c - s*s
    return T


def slack_response(C, e, c, phi, tangent=True):
    """Membrane response of crumpled triangles.

    Where a fold was crushed flat the paper wrinkled: it kept its true length and hides part of it in wrinkles too
    small for the mesh. Such a triangle carries a slack c = (c1, c2) <= 0: the Green strain of its crushed length
    along two orthogonal rest directions (axis 1 at angle phi from the stock's machine direction). Along each of them
    it is stress-free anywhere between the crushed and the true length; shortened below the crushed length or
    stretched beyond the true length it answers with the stock's stiffness. Shear is elastic.
    The hidden strain k (c <= k <= 0, componentwise) is the minimiser of the stock's energy 1/2 (e - k)^T C (e - k):
    a convex two-variable box problem solved in closed form (the unconstrained minimiser, or the best point of the
    box's four edges). By the envelope theorem the force is S = C (e - k), with zero force along a slack direction,
    and the tangent is C condensed on the slack directions (uniaxial response across a slack direction).
    C (m,3,3), e (m,3) Voigt with engineering shear, c (m,2), phi (m,).
    Returns (elastic strain r (m,3), S (m,3), tangent (m,3,3) or None, hidden strain k (m,2)); r, S and the tangent in
    the stock's axes, k in the slack axes."""
    T = _strain_rot(phi)
    Ti = _strain_rot(-phi)
    ep = np.einsum("tij,tj->ti", T, e)
    Cp = np.matmul(np.matmul(Ti.transpose(0, 2, 1), C), Ti)
    a11, a12, a22 = Cp[:, 0, 0], Cp[:, 0, 1], Cp[:, 1, 1]
    b1, b2 = Cp[:, 0, 2]*ep[:, 2], Cp[:, 1, 2]*ep[:, 2]
    lo1, lo2 = ep[:, 0], ep[:, 1]                      # hidden strain 0: the true length
    hi1, hi2 = ep[:, 0] - c[:, 0], ep[:, 1] - c[:, 1]  # hidden strain c: the crushed length

    def q(r1, r2):
        return .5*(a11*r1*r1 + 2*a12*r1*r2 + a22*r2*r2) + b1*r1 + b2*r2
    det = a11*a22 - a12*a12
    u1, u2 = (-a22*b1 + a12*b2)/det, (a12*b1 - a11*b2)/det
    inside = (u1 >= lo1) & (u1 <= hi1) & (u2 >= lo2) & (u2 <= hi2)
    best_q = np.where(inside, q(u1, u2), np.inf)
    r1, r2 = u1.copy(), u2.copy()
    for fix, bound in ((0, lo1), (0, hi1), (1, lo2), (1, hi2)):
        if fix == 0:
            x1 = bound
            x2 = np.clip(-(b2 + a12*x1)/a22, lo2, hi2)
        else:
            x2 = bound
            x1 = np.clip(-(b1 + a12*x2)/a11, lo1, hi1)
        qq = q(x1, x2)
        take = ~inside & (qq < best_q)
        best_q = np.where(take, qq, best_q)
        r1, r2 = np.where(take, x1, r1), np.where(take, x2, r2)
    rp = np.stack([r1, r2, ep[:, 2]], 1)
    Sp = np.einsum("tij,tj->ti", Cp, rp)
    k = np.stack([ep[:, 0] - r1, ep[:, 1] - r2], 1)
    tol = 1e-12
    f1 = (r1 > lo1 + tol) & (r1 < hi1 - tol)          # slack (free) along axis 1 / 2
    f2 = (r2 > lo2 + tol) & (r2 < hi2 - tol)
    Sp[:, 0] = np.where(f1, 0., Sp[:, 0])             # exactly zero along a slack direction (it is, up to rounding)
    Sp[:, 1] = np.where(f2, 0., Sp[:, 1])
    r = np.einsum("tij,tj->ti", Ti, rp)
    S = np.einsum("tji,tj->ti", T, Sp)
    Ct = None
    if tangent:
        Ctp = Cp.copy()
        only1, only2, both = f1 & ~f2, f2 & ~f1, f1 & f2
        for sel, f in ((only1, 0), (only2, 1)):
            if sel.any():
                col = Cp[sel][:, :, f]
                Ctp[sel] = Cp[sel] - col[:, :, None]*col[:, None, :]/Cp[sel][:, f, f][:, None, None]
        if both.any():
            Cb = Cp[both]
            Ai = np.linalg.inv(Cb[:, :2, :2])
            Ctp[both] = Cb - np.matmul(np.matmul(Cb[:, :, :2], Ai), Cb[:, :2, :])
        Ct = np.matmul(np.matmul(T.transpose(0, 2, 1), Ctp), T)
    return r, S, Ct, k


@dataclass
class Sheet:
    """One shell piece. X2: rest coordinates in the flat net (m); x0: initial 3D positions."""
    name: str
    X2: np.ndarray
    tris: np.ndarray
    x0: np.ndarray
    mats: list                         # ShellMat per material id
    tri_mat: np.ndarray                # material id per triangle
    md_angle: float = 0.               # MD direction in the net (radians from +x), 2D nets
    md_dir: object = None              # MD vector(s) in 3D for 3D rest shapes (one or per triangle)
    creases: dict = field(default_factory=dict)        # (i, j) sorted -> CreaseLaw
    crease_rest: dict = field(default_factory=dict)    # (i, j) -> plastic rest angle (else from x0)
    render: dict = field(default_factory=dict)         # texture / colours (render.py)
    tearable: bool = False
    mu: float = None                   # friction coefficient of this sheet's surface (None: world default)
    fmat: str = None                   # friction material (params.FRICTION pairs); used instead of mu when set


@dataclass
class Rod:
    name: str
    x0: np.ndarray                     # (m,3)
    mat: RodMat
    fmat: str = None                   # friction material (params.FRICTION pairs)
    # ((local vertex index, body name), ...): rod nodes that are part of a rigid body (a rope's end inside its knot).
    # No contact is computed between such a node and that body; the node is tied to the body by the caller's weld
    attached: tuple = ()


@dataclass
class Body:
    """Affine rigid body. verts/faces are the collision surface in the body frame (origin at COM)."""
    name: str
    verts: np.ndarray
    faces: np.ndarray
    mass: float = 0.
    S: np.ndarray = None               # second moment of mass about the COM (kg m^2), body frame
    kind: str = "dynamic"              # dynamic | driven | static
    p: np.ndarray = None               # COM world position
    R: np.ndarray = None               # orientation
    mu: float = .4
    fmat: str = None                   # friction material (params.FRICTION pairs); used instead of mu when set
    k_lin: float = 2e5                 # driven: N/m translation pull to the target
    k_rot: float = 50.                 # driven: N m/rad rotation pull
    closing_axis: np.ndarray = None    # driven finger: world unit axis of closing
    f_cap: float = np.inf              # driven: N, force limit of the pull toward the target (Huber)
    render: dict = field(default_factory=dict)
    vertex_weld: float = 0.            # m, collision surface deflation (unused: kept 0)


@dataclass
class Pin:
    """Penalty pin joint between two body points (body b may be None = world point)."""
    a: int
    ra: np.ndarray
    b: int
    rb: np.ndarray
    k: float


@dataclass
class BodyHinge:
    """Crease hinge between two bodies: dihedral angle of 4 points (two on the axis, one on each plate)."""
    pts: tuple          # ((body, r_local), ...) x4 in hinge order (axis a, axis b, plate 1, plate 2)
    length: float
    law: CreaseLaw
    theta_p: float = None
    theta_r: float = 0.


# ----------------------------------------------------------------------------- Newton matrix pattern
PATTERN_SLACK = .10      # a grown pattern is rebuilt from the current need once it exceeds the last rebuild by this fraction
METIS_REFRESH = 100      # analyses with a reused fill-reducing ordering before a fresh METIS ordering
_OWN_PHASE = -7701       # marker in PyPardisoSolver.phase: the solver holds this module's analysis
CONST_CHUNK_MIN = 50000  # entries: Hessian chunks at least this large have their scattered sum cached while unchanged


class _NewtonPattern:
    """The upper triangle of the Newton matrix in a sorted CSR sparsity pattern kept across Newton iterations and steps.

    Entries come as COO chunks (rows, cols, values). The chunks of the elastic and coupling terms (membrane, hinges,
    drag, bodies, tape bonds, fixtures, mass/diagonal) keep their index arrays from one iteration to the next: each is
    recognised (the same array objects, or equal contents) and scattered through a cached map into the base pattern B.
    The contact chunk changes with the active pairs and is looked up in the full pattern A (a superset of B); a missing
    entry is merged into A, an entry that vanishes stays as an explicit zero, and A is rebuilt from the current need
    once it has grown by PATTERN_SLACK. While A is unchanged PARDISO's analysis (fill-reducing ordering and symbolic
    factorisation, about 60 % of a fresh solve) is reused; when A changes only the symbolic factorisation is redone
    with the METIS ordering of an earlier pattern (iparm[4] = 1), refreshed every METIS_REFRESH analyses or when the
    factor fill has grown by 10 %. Explicit zeros and the elimination ordering change the round-off of the solve only."""

    def __init__(self, n):
        self.n = n
        self.entries = {}                      # signature -> list of [rows, cols, pos in B (-1: lower entry), last values
                                               # object, its scattered sum (large chunks with unchanged values)]
        self.cur = []                          # entries of the current assembly's chunks, in order
        self.seq, self.pos_all = None, None    # the varying entries of the last assembly and their positions
        self.B = np.zeros(0, np.int64)         # sorted keys (row*n + col) of the base pattern
        self.B_ref = 0
        self.keys = None                       # sorted keys of the full pattern A
        self.A_ref = 0
        self.version = 0
        self.posBA = None
        self.reset_analysis()
        self.stats = dict(patterns=0, analyses=0, metis=0, rebuilds=0, registrations=0, base_s=0., contact_s=0., grow_s=0.,
                          fill_s=0., analyse_s=0., factor_solve_s=0.)

    def reset_analysis(self):
        self.analysed = None                   # (solver id, pattern version) PARDISO last analysed
        self.perm = None
        self.since_metis = 0
        self.metis_fill = None

    # ------------------------------------------------------------- assembly
    def _keys(self, r, c):
        r = np.asarray(r, np.int64).ravel()
        c = np.asarray(c, np.int64).ravel()
        return np.where(r <= c, r*self.n + c, -1)

    @staticmethod
    def _same(a, b):
        return a is b or (a.shape == b.shape and np.array_equal(a, b))

    def _base(self, chunks):
        """Positions in B of every entry of the base chunks (concatenated; len(B) for lower-triangle entries) and
        whether B changed. Chunks seen before reuse their map."""
        if self.B_ref and len(self.B) > (1. + PATTERN_SLACK)*self.B_ref:
            self.entries, self.seq, self.B, self.B_ref = {}, None, np.zeros(0, np.int64), 0     # shed stale keys
        seq, fresh = [], []
        for r, c, _ in chunks:
            sig = (r.size, int(r.flat[0]), int(r.flat[-1]), int(c.flat[0]), int(c.flat[-1]), int(r.flat[r.size//2]))
            ent = None
            for e in self.entries.get(sig, ()):
                if self._same(e[0], r) and self._same(e[1], c):
                    ent = e
                    e[0], e[1] = r, c                     # the newest objects: the identity test hits next time
                    break
            if ent is None:
                ent = [r, c, None, None, None]
                self.entries.setdefault(sig, []).append(ent)
                fresh.append(ent)
            seq.append(ent)
        changed = False
        if fresh:
            self.stats["registrations"] += len(fresh)
            ks = [self._keys(e[0], e[1]) for e in fresh]
            allk = np.concatenate(ks)
            new = np.setdiff1d(allk[allk >= 0], self.B)                     # sorted, unique
            if len(new):
                shift = np.searchsorted(new, self.B)                         # new keys below each old key
                for es in self.entries.values():
                    for e in es:
                        if e[2] is not None:
                            m = e[2] >= 0
                            e[2][m] += shift[e[2][m]]
                self.B = np.union1d(self.B, new)
                changed = True
                for es in self.entries.values():
                    for e in es:
                        e[4] = None                       # cached sums were in the old B
                self.seq = None
            for e, k in zip(fresh, ks):
                pos = np.full(len(k), -1, np.int64)
                up = k >= 0
                pos[up] = np.searchsorted(self.B, k[up])
                e[2] = pos
            if not self.B_ref:
                self.B_ref = len(self.B)
            if len(self.entries) > 4*len(seq) + 64:                          # forget chunks no longer produced
                live = {id(e) for e in seq}
                self.entries = {g: [e for e in es if id(e) in live] for g, es in self.entries.items()}
                self.entries = {g: es for g, es in self.entries.items() if es}
        self.cur = seq
        return changed

    def _base_data(self, chunks):
        """Values of the base pattern: the scatter-sum of all base chunks. A large chunk handed in with the same values
        object as in the previous assembly (a term whose Hessian is constant, e.g. tape bonds between damage updates)
        contributes its cached sum."""
        nb = len(self.B)
        dataB = np.zeros(nb + 1)
        var, vals = [], []
        for e, (_, _, v) in zip(self.cur, chunks):
            if e[3] is v and v.size >= CONST_CHUNK_MIN:
                if e[4] is None:
                    e[4] = np.bincount(np.where(e[2] < 0, nb, e[2]), weights=v.ravel(), minlength=nb + 1)
                dataB += e[4]
                continue
            e[3], e[4] = v, None
            var.append(e); vals.append(v.ravel())
        if var:
            if self.seq is None or len(var) != len(self.seq) or any(a is not b for a, b in zip(var, self.seq)):
                pa = np.concatenate([e[2] for e in var])
                self.pos_all = np.where(pa < 0, nb, pa)
                self.seq = var
            dataB += np.bincount(self.pos_all, weights=np.concatenate(vals), minlength=nb + 1)
        return dataB

    def _set_pattern(self, keys, posBA=None):
        n = self.n
        self.keys = keys
        rows = keys//n
        self.indptr = np.zeros(n + 1, np.int32)
        np.cumsum(np.bincount(rows, minlength=n), out=self.indptr[1:])
        self.indices = (keys - rows*n).astype(np.int32)
        self.ia1, self.ja1 = self.indptr + 1, self.indices + 1
        self.posBA = np.searchsorted(keys, self.B) if posBA is None else posBA
        self.version += 1
        self.stats["patterns"] += 1

    def assemble(self, chunks, contact):
        """chunks: the base COO chunks [(rows, cols, values)]; contact: one (rows, cols, values) or None."""
        n = self.n
        t0 = time.perf_counter()
        chunks = [(np.asarray(r), np.asarray(c), np.asarray(v, float)) for r, c, v in chunks if len(r)]
        b_changed = self._base(chunks)
        nb = len(self.B)
        dataB = self._base_data(chunks)
        t1 = time.perf_counter()
        if contact is not None:
            dk = self._keys(contact[0], contact[1])
            dv = np.asarray(contact[2], float).ravel()
            if (dk < 0).any():
                up = dk >= 0
                dk, dv = dk[up], dv[up]
        else:
            dk, dv = np.zeros(0, np.int64), np.zeros(0)
        if self.keys is None:
            self._set_pattern(np.union1d(self.B, dk))
            self.A_ref = len(self.keys)
        elif b_changed:
            self._set_pattern(np.union1d(self.keys, self.B))
        pos = np.searchsorted(self.keys, dk)
        hit = self.keys[np.minimum(pos, len(self.keys) - 1)] == dk
        t2 = time.perf_counter()
        if not hit.all():
            new = np.unique(dk[~hit])
            if len(self.keys) + len(new) > (1. + PATTERN_SLACK)*self.A_ref:
                self._set_pattern(np.union1d(self.B, dk))                     # the current need
                self.A_ref = len(self.keys)
                self.stats["rebuilds"] += 1
                pos = np.searchsorted(self.keys, dk)
            else:
                # merged index of a key = old keys below it + new keys below it
                self._set_pattern(np.insert(self.keys, np.searchsorted(self.keys, new), new),
                                  self.posBA + np.searchsorted(new, self.B))
                pos = pos + np.searchsorted(new, dk)
        t3 = time.perf_counter()
        data = np.zeros(len(self.keys))
        data[self.posBA] = dataB[:nb]
        if len(dk):
            data += np.bincount(pos, weights=dv, minlength=len(self.keys))
        H = sp.csr_matrix((data, self.indices, self.indptr), shape=(n, n), copy=False)
        H._gb_pattern = self
        H._gb_version = self.version
        st = self.stats
        st["base_s"] += t1 - t0; st["contact_s"] += t2 - t1; st["grow_s"] += t3 - t2; st["fill_s"] += time.perf_counter() - t3
        return H

    # ------------------------------------------------------------- PARDISO
    def _call(self, ps, phase, data, b):
        import ctypes
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        i32p, f64p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
        perm = self.perm if self.perm is not None else ps.perm
        ps._mkl_pardiso(ps.pt.ctypes.data_as(ctypes.POINTER(ps._pt_type[0])), ctypes.byref(ctypes.c_int32(1)),
                        ctypes.byref(ctypes.c_int32(1)), ctypes.byref(ctypes.c_int32(ps.mtype)),
                        ctypes.byref(ctypes.c_int32(phase)), ctypes.byref(ctypes.c_int32(self.n)), data.ctypes.data_as(f64p),
                        self.ia1.ctypes.data_as(i32p), self.ja1.ctypes.data_as(i32p), perm.ctypes.data_as(i32p),
                        ctypes.byref(ctypes.c_int32(1)), ps.iparm.ctypes.data_as(i32p), ctypes.byref(ctypes.c_int32(ps.msglvl)),
                        b.ctypes.data_as(f64p), x.ctypes.data_as(f64p), ctypes.byref(err))
        if err.value != 0:
            import pypardiso
            raise pypardiso.PyPardisoError(err.value)
        return x

    def _analyse(self, ps, data, b):
        if self.analysed is None or self.analysed[0] != id(ps):
            self.reset_analysis()
        ps.set_iparm(12, 0)                                     # A x = b (CSR), as pypardiso's _check_A sets it
        ps.iparm[17] = -1                                       # report the factor fill of this analysis
        if self.perm is None or self.since_metis >= METIS_REFRESH:
            if ps.iparm[0] == 0:
                self._call(ps, 11, data, b)                      # fills MKL's defaults (the first pypardiso call did)
            self.perm = np.zeros(self.n, np.int32)
            ps.set_iparm(5, 2)                                   # METIS ordering, returned in perm
            self._call(ps, 11, data, b)
            self.since_metis, self.metis_fill = 0, int(ps.iparm[17])
            self.stats["metis"] += 1
        else:
            ps.set_iparm(5, 1)                                   # reuse the ordering: symbolic factorisation only
            self._call(ps, 11, data, b)
            self.since_metis += 1
            if self.metis_fill and ps.iparm[17] > 1.1*self.metis_fill:
                self.since_metis = METIS_REFRESH                 # the ordering has aged: METIS at the next analysis
        ps.set_iparm(5, 0)
        ps.phase = _OWN_PHASE                                    # pypardiso's own solve()/factorize() overwrite this
        self.analysed = (id(ps), self.version)
        self.stats["analyses"] += 1

    def solve(self, ps, H, rhs):
        if H._gb_version != self.version:
            raise RuntimeError("Newton matrix assembled in an older pattern")
        b = np.ascontiguousarray(rhs, float)
        data = np.ascontiguousarray(H.data)
        t0 = time.perf_counter()
        if ps.mtype != 2:
            # a general solver (only after a failed SPD factorisation, see IPCWorld._solve): its analysis depends on the
            # values (matching, scaling), so it analyses every matrix, as pypardiso's solve did
            self.reset_analysis()
            ps.set_iparm(12, 0)
            x = self._call(ps, 13, data, b)
            self.stats["factor_solve_s"] += time.perf_counter() - t0
            return x
        if self.analysed != (id(ps), self.version) or ps.phase != _OWN_PHASE:
            # (the second test: someone used this solver through pypardiso's interface since our analysis)
            self._analyse(ps, data, b)
        t1 = time.perf_counter()
        x = self._call(ps, 23, data, b)
        self.stats["analyse_s"] += t1 - t0
        self.stats["factor_solve_s"] += time.perf_counter() - t1
        return x


# ----------------------------------------------------------------------------- the world
class ContactSet:
    """One IPC contact family on a collision mesh with its own minimum distance (dmin): barrier potential,
    lagged smooth Coulomb friction with per-pair coefficients, additive CCD."""

    def __init__(self, world, name, mesh, dmin, edges=None, dhat=None, kappa=None):
        import ipctk
        self.w, self.name, self.mesh, self.dmin = world, name, mesh, dmin
        self.E = world.Ec if edges is None else edges
        # barrier range and stiffness of this family: the world's (the paper caliper) unless given (a rope's own
        # surface compliance, RodMat.contact_range / contact_kappa)
        self.dhat = world.dhat if dhat is None else float(dhat)
        self._kappa = kappa
        self.barrier = ipctk.BarrierPotential(ipctk.ClampedLogBarrier(), self.dhat, 1.)
        self.ipctk = ipctk

    @property
    def kappa(self):
        return self.w.kappa if self._kappa is None else self._kappa

    def collisions(self, xc, cand=None):
        """Active pairs (d < dmin + dhat) at xc. With `cand` (a swept candidate set whose segment contains xc)
        only the narrow phase runs."""
        C = self.ipctk.NormalCollisions()
        C.use_area_weighting = True
        if cand is not None:
            C.build(cand, self.mesh, np.asfortranarray(xc), self.dhat, self.dmin)
        else:
            C.build(self.mesh, np.asfortranarray(xc), self.dhat, self.dmin, self.w.broad)
        return C

    def sweep(self, x0, x1):
        """Broad phase over the straight-line motion x0 -> x1, inflated to the barrier range: every pair that
        can be active anywhere on the segment, and every pair CCD must examine."""
        cand = self.ipctk.Candidates()
        cand.build(self.mesh, np.asfortranarray(x0), np.asfortranarray(x1), .5*(self.dmin + self.dhat), self.w.broad)
        return cand

    def tangential(self, xc0, dt):
        ipctk, w = self.ipctk, self.w
        self.friction = ipctk.FrictionPotential(w.eps_v*dt)
        C0 = self.collisions(xc0)
        self._C0 = C0                                 # the start point's pairs (IPCWorld.step reuses them)
        Tc = ipctk.TangentialCollisions()
        if len(C0):
            Bk = ipctk.BarrierPotential(ipctk.ClampedLogBarrier(), self.dhat, self.kappa)
            table = w._mu_table
            def blend(a, b):
                return table[(int(round(a)), int(round(b)))]
            Tc.build(self.mesh, np.asfortranarray(xc0), C0, Bk, w.vclass, w.vclass, blend)
        return Tc

    def energy(self, C, T, xcF, U):
        w = self.w
        e = self.kappa*self.barrier(C, self.mesh, xcF) if len(C) else 0.
        if T is not None and len(T):
            e += self.friction(T, self.mesh, U)
        return e

    def derivatives(self, C, T, xcF, U, hessian=True, parts=False):
        """Contact gradient and Hessian (collision-mesh coordinates). parts=True: the Hessian as a list of
        (matrix, scale) terms, barrier and friction unsummed (IPCWorld._hessian sums them while mapping to the DOFs)."""
        ipctk, w = self.ipctk, self.w
        g = np.zeros(3*w.n_coll)
        H = None
        Hp = []
        if len(C):
            g += self.kappa*self.barrier.gradient(C, self.mesh, xcF)
            if hessian:
                Hb = self.barrier.hessian(C, self.mesh, xcF, ipctk.PSDProjectionMethod.CLAMP)
                if parts:
                    Hp.append((Hb, self.kappa))
                else:
                    H = self.kappa*Hb
        if T is not None and len(T):
            g += self.friction.gradient(T, self.mesh, U)
            if hessian:
                Hf = self.friction.hessian(T, self.mesh, U, ipctk.PSDProjectionMethod.CLAMP)
                if parts:
                    Hp.append((Hf, None))
                else:
                    H = Hf if H is None else H + Hf
        return (g, Hp) if parts else (g, H)

    def ccd(self, x0, x1, cand=None):
        """Largest step fraction that keeps every pair beyond CCD_FRACTION * dmin. The full thickness dmin is
        enforced by the barrier energy itself (infinite at d <= dmin), which the line search evaluates at the
        end point; CCD only has to rule out tunnelling. Running CCD at dmin directly makes squeezed paper
        (a robot pressing it onto furniture: 10-100 N on a few vertices, gaps 1e-12 m) throttle every step to
        alpha ~ 1e-5 and the additive CCD iterate ~1e9 times."""
        w = self.w
        if cand is not None:
            return cand.compute_collision_free_stepsize(self.mesh, np.asfortranarray(x0), np.asfortranarray(x1),
                                                        CCD_FRACTION*self.dmin, w.ccd)
        return self.ipctk.compute_collision_free_stepsize(self.mesh, np.asfortranarray(x0), np.asfortranarray(x1),
                                                          CCD_FRACTION*self.dmin, w.broad, w.ccd)


class IPCWorld:
    def __init__(self, sheets=(), rods=(), bodies=(), *, dhat=None, mu_default=.4, eps_v=1e-3,
                 newton_max=24, tol=1e-6, kappa=None, gravity=(0., 0., -G), pair_mu=None, contact_stock=None,
                 mu_pairs=None, friction_strict=False, tear_pool=None, stick_friction=True, force_rtol=FORCE_RTOL):
        """`tear_pool`: spare vertices preallocated per tearable sheet (int, {sheet name: int}, or None for the default
        max(32, n/6)); 0 switches vertex splitting off (crack band only)."""
        import ipctk
        self.ipctk = ipctk
        ipctk.set_num_threads(SOLVER_THREADS)
        self._tear_pool_arg = tear_pool
        self.sheets, self.rods, self.bodies = list(sheets), list(rods), list(bodies)
        self.mu_default, self.eps_v = mu_default, eps_v
        self.stick_friction = bool(stick_friction)     # False: IPC's lagged friction alone (DIAGNOSTIC)
        self._stick = np.zeros((0, 3))                 # per collision vertex (IPCWorld._stick_update)
        self.newton_max, self.tol = newton_max, tol
        self.force_rtol = force_rtol                   # None: the step-size rule alone (DIAGNOSTIC)
        self.g = np.asarray(gravity, float)
        self.pair_mu = pair_mu or {}
        self.mu_pairs, self.friction_strict = mu_pairs or {}, friction_strict
        # One contact caliper per world (contact_caliper): the thickest sheet stock, or `contact_stock` for a
        # world without sheets (the paper covering of the rigid items). Thinner secondary sheets (tape) use it.
        stocks = [m for s in self.sheets for m in s.mats]
        stock = max(stocks, key=lambda m: m.t) if stocks else contact_stock
        if stock is None:
            raise ValueError("a contact world needs a sheet or a contact_stock for its caliper")
        self.caliper = contact_caliper(stock)
        self.h_s = self.caliper["h_s"]
        self.dmin = self.caliper["dmin"]
        self.dhat = float(dhat) if dhat is not None else self.caliper["dhat"]
        self.pins, self.hinges_b, self.bonds, self.films = [], [], [], []
        self.fixture_force = None            # ASSISTED only: explicit forces on free vertices (recorded)
        self.fixture_springs = []            # ASSISTED only: (vertex ids, targets, k, F_cap) phantom grips
        self.verbose = False                 # per-Newton-iteration diagnostics
        self.time = 0.
        self.steps = 0
        self.fault = None
        self.stats = dict(newton=0, steps=0, unconverged=0, solve_s=0., assemble_s=0., ipc_s=0., ccd_s=0.,
                          last_iters=0, max_iters=0, min_alpha=1., kappa=0.)
        self._build()
        # barrier stiffness (area-weighted barrier, J/m^4): IPC's minimum rule (ipctk.initial_barrier_stiffness)
        # gave ~3e5 here, which let paper settle deep into the near-singular barrier zone (20+ Newton iterations
        # per step); 1e10 carries ~1.2 MPa at dhat/2, far above any load in these scenes,
        # so contacts sit near dhat and Newton converges in 2-4 iterations (REPORT §6)
        self.kappa = kappa if kappa is not None else self.caliper["kappa"]
        self.check_initial()

    # ------------------------------------------------------------------ build
    def _build(self):
        ipctk = self.ipctk
        # free vertices: sheets then rods
        X2s, tris, tmat, mats, xs = [], [], [], [], []
        self.sheet_slices, off = [], 0
        mat_index = {}
        crease_list = []
        self.sheet_of_tri, tri_off = [], 0
        # A tearable sheet (a stock with Gamma > 0) gets a pool of inactive spare vertices right after its own vertices: a
        # split activates the next spare as the copy of the split vertex and the sheet's slice grows by one, so global
        # vertex ids never shift and a slice always holds exactly the sheet's real vertices. Inactive spares carry no
        # mass, energy or topology, are in no collision mesh, and get identity Newton rows.
        self.sheet_n0, self.sheet_pool, self._pool_start = [], [], []
        for s_i, s in enumerate(self.sheets):
            n = len(s.X2)                                  # rest coordinates define the vertex count (x0 may carry
            x_s = np.asarray(s.x0, float)[:n]              # parked spare rows copied back from an earlier world)
            pool = self._pool_size(s, n) if any(m.Gamma > 0. for m in s.mats) else 0
            ids = []
            for m in s.mats:
                if id(m) not in mat_index:
                    mat_index[id(m)] = len(mats)
                    mats.append(m)
                ids.append(mat_index[id(m)])
            xs.append(np.vstack([x_s, np.repeat(x_s[:1], pool, 0)]) if pool else x_s)
            tris.append(np.asarray(s.tris) + off)
            tmat.append(np.asarray(ids)[np.asarray(s.tri_mat)])
            X2s.append(local_coords(s))
            for (i, j), law in s.creases.items():
                crease_list.append((i + off, j + off, law, s.crease_rest.get((i, j))))
            self.sheet_slices.append(slice(off, off + n))
            self.sheet_n0.append(n)
            self.sheet_pool.append(pool)
            self._pool_start.append(off + n)
            self.sheet_of_tri.append(np.full(len(s.tris), s_i))
            off += n + pool
        self.n_sheet = off
        self.mats = mats
        self.rod_slices = []
        for r in self.rods:
            n = len(r.x0)
            xs.append(np.asarray(r.x0, float))
            self.rod_slices.append(slice(off, off + n))
            off += n
        self.n_free = off
        self.x_free0 = np.concatenate(xs) if xs else np.zeros((0, 3))
        self.tris = np.concatenate(tris).astype(np.int64) if tris else np.zeros((0, 3), np.int64)
        self.tri_mat = np.concatenate(tmat).astype(np.int64) if tmat else np.zeros(0, np.int64)
        self.sheet_of_tri = np.concatenate(self.sheet_of_tri) if self.sheet_of_tri else np.zeros(0, int)
        self.Lt = np.concatenate(X2s) if X2s else np.zeros((0, 3, 2))      # per-triangle rest coords (MD = +x)
        self._precompute_shells(crease_list)
        self._precompute_rods()
        self._tear_setup()
        # bodies
        self.body_dof = {}
        nd = 3*self.n_free
        for b_i, b in enumerate(self.bodies):
            b.p = np.asarray(b.p, float); b.R = np.asarray(b.R if b.R is not None else np.eye(3), float)
            if b.kind != "static":
                self.body_dof[b_i] = nd
                nd += 12
        self.n_dof = nd
        self._update_inactive_diag()
        self._build_collision()
        q = np.zeros(nd)
        q[:3*self.n_free] = self.x_free0.ravel()
        for b_i, o in self.body_dof.items():
            b = self.bodies[b_i]
            q[o:o+3] = b.p
            q[o+3:o+12] = b.R.ravel()
        self.q = q
        self.v = np.zeros(nd)
        self._mass_matrix()
        self.targets = {}            # driven body index -> (p_hat, A_hat)
        self.finger_cmd = {}         # driven finger body index -> (axis, s_cmd (along axis, world), F_cap, k)
        self.reactions = {}

    def _precompute_shells(self, crease_list):
        T, Lt = self.tris, self.Lt
        n = self.n_sheet
        self.area = np.zeros(0)
        self.vmass = np.zeros(self.n_free)
        if not len(T):
            self.hinge = np.zeros((0, 4), int); self.k_hinge = np.zeros(0); self.theta_p = np.zeros(0)
            self.theta_r = np.zeros(0)
            self.is_crease = np.zeros(0, bool); self.C = np.zeros((0, 3, 3)); self.thick = np.zeros(0)
            self.vthick = np.zeros(self.n_free); self.damage = np.zeros(0)
            self.slack_c = np.zeros((0, 2)); self.slack_phi = np.zeros(0); self._slack_idx = None
            return
        Dm = np.stack([Lt[:, 1] - Lt[:, 0], Lt[:, 2] - Lt[:, 0]], axis=2)
        det = np.linalg.det(Dm)
        if np.any(np.abs(det) < 1e-14):
            raise ValueError("Degenerate rest triangle")
        self.Dm_inv = np.linalg.inv(Dm)
        self.area = .5*np.abs(det)
        b = np.zeros((len(T), 3, 2))
        b[:, 1, :] = self.Dm_inv[:, 0, :]
        b[:, 2, :] = self.Dm_inv[:, 1, :]
        b[:, 0, :] = -(b[:, 1, :] + b[:, 2, :])
        self.bmat = b
        rho = np.array([self.mats[m].rho_A for m in self.tri_mat])
        np.add.at(self.vmass, T.ravel(), np.repeat(self.area*rho/3., 3))
        self.thick = np.array([self.mats[m].t for m in self.tri_mat])
        self.vthick = np.zeros(self.n_free)
        np.maximum.at(self.vthick, T.ravel(), np.repeat(self.thick, 3))
        self.C = np.stack([self.mats[m].membrane_C(0.) for m in self.tri_mat])
        self.md_angle_tri = np.zeros(len(T))          # local rest frames carry MD along +x
        self.damage = np.zeros(len(T))
        self.kappa_hist = np.zeros(len(T))     # max equivalent strain reached (crack band)
        # crumpled triangles (slack_response): hidden length as Green strain along two rest axes, and the axes' angle
        self.slack_c = np.zeros((len(T), 2))
        self.slack_phi = np.zeros(len(T))
        self._slack_idx = None                 # indices of the triangles with slack (None: none)
        # hinges
        edges = {}
        for t, tri in enumerate(T):
            for k in range(3):
                i, j, o = tri[k], tri[(k+1) % 3], tri[(k+2) % 3]
                edges.setdefault((min(i, j), max(i, j)), []).append((t, o, (i, j)))
        hinges, hk, hcrease, hlaw, hrest, htris = [], [], [], [], [], []
        claws = {(min(i, j), max(i, j)): (law, rest) for i, j, law, rest in crease_list}
        for (i, j), items in edges.items():
            if len(items) != 2:
                continue
            (t0, o0, (a0, b0)), (t1, o1, _) = items
            hinges.append((a0, b0, o0, o1)); htris.append((t0, t1))
            cl = claws.get((i, j))
            hcrease.append(cl is not None)
            hlaw.append(cl[0] if cl else None)
            hrest.append(cl[1] if cl else None)
        self.hinge = np.asarray(hinges, np.int64).reshape(-1, 4)
        self.hinge_tris = np.asarray(htris, np.int64).reshape(-1, 2)
        H = self.hinge
        t0 = self.hinge_tris[:, 0]
        k0 = np.argmax(T[t0] == H[:, :1], axis=1)
        k1 = np.argmax(T[t0] == H[:, 1:2], axis=1)
        e = Lt[t0, k1] - Lt[t0, k0]                    # hinge edge in the first triangle's rest frame
        le = np.linalg.norm(e, axis=1)
        Ae = self.area[self.hinge_tris[:, 0]] + self.area[self.hinge_tris[:, 1]]
        # bending direction perpendicular to the edge in the rest plane; D interpolated MD/CD
        perp = np.stack([-e[:, 1], e[:, 0]], 1)/le[:, None]
        cos2 = perp[:, 0]**2
        m0 = self.tri_mat[self.hinge_tris[:, 0]]
        Dmd = np.array([self.mats[m].D_md for m in m0]); Dcd = np.array([self.mats[m].D_cd for m in m0])
        D = Dmd*cos2 + Dcd*(1. - cos2)
        C_BEND = .5217     # label_application calibration: isometric cylinder wrap energy within 0.9 %
        self.hinge_len = le
        self.hinge_dual = 1.5*Ae/(3.*le)*2.
        self.is_crease = np.asarray(hcrease, bool)
        self.crease_law = hlaw
        k = C_BEND*D*le**2/Ae                      # panel hinge: E = k (theta - theta_p)^2
        for h in np.flatnonzero(self.is_crease):
            k[h] = .5*hlaw[h].k*le[h]               # crease line: E = 1/2 k_c |e| (theta - theta_p)^2
        self.k_hinge = k
        self.k_hinge0 = k.copy()
        ty = np.array([self.mats[m].yield_strain*2./self.mats[m].t for m in m0])
        self.theta_yield = 1.5*ty*self.hinge_dual   # panel plastic hinge limit (fully plastic moment)
        x0 = self.x_free0
        th0, _ = hinge_angles(x0, H) if len(H) else (np.zeros(0), None)
        self.theta_p = np.zeros(len(H))
        self.theta_r = np.zeros(len(H))        # recoverable part of a crease's set (0: aged, fully permanent)
        for h in range(len(H)):
            if hcrease[h]:
                self.theta_p[h] = th0[h] if hrest[h] is None else hrest[h]
        # vertex adjacency for rendering normals
        self.edges_sheet = edges_of(T)

    def _precompute_rods(self):
        segs, rest, k_s, kb, rad, bend, bend_k, bend_rest = [], [], [], [], [], [], [], []
        self.rod_radius_v = np.zeros(self.n_free)
        for r, sl in zip(self.rods, self.rod_slices):
            idx = np.arange(sl.start, sl.stop)
            x = np.asarray(r.x0, float)
            L = np.linalg.norm(np.diff(x, axis=0), axis=1)
            for a in range(len(idx) - 1):
                segs.append((idx[a], idx[a+1])); rest.append(L[a]); k_s.append(r.mat.EA/L[a])
            for a in range(1, len(idx) - 1):
                bend.append((idx[a-1], idx[a], idx[a+1]))
                lm = .5*(L[a-1] + L[a])
                bend_k.append(r.mat.EI/lm)
                bend_rest.append(0.)
            m = np.zeros(len(idx))
            m[:-1] += .5*L*r.mat.lin_rho; m[1:] += .5*L*r.mat.lin_rho
            self.vmass[idx] += m
            self.rod_radius_v[idx] = r.mat.radius
        self.rod_seg = np.asarray(segs, np.int64).reshape(-1, 2)
        self.rod_rest = np.asarray(rest)
        self.rod_k = np.asarray(k_s)
        self.rod_bend = np.asarray(bend, np.int64).reshape(-1, 3)
        self.rod_bend_k = np.asarray(bend_k)

    def _build_collision(self):
        """Collision mesh: free vertices (sheet + rod), then every body's surface vertices.
        x_coll = P q + c."""
        ipctk = self.ipctk
        V, F, E_extra, patch = [], [], [], []
        V.append(self.x_free0)
        F.append(self.tris)
        pid = np.arange(self.n_free, dtype=np.int64) + 1          # free vertices: own patch (self contact)
        patch.append(pid)
        rows, cols, vals = [], [], []
        nf = self.n_free
        r3 = np.arange(3*nf)
        rows.append(r3); cols.append(r3); vals.append(np.ones(3*nf))
        c = np.zeros(3*nf)
        cs = [c]
        self.body_vslice = []
        voff = nf
        external_patch = 0                                     # robots + furniture: never collide with each other
        next_patch = nf + 1
        for b_i, b in enumerate(self.bodies):
            r = deflate(np.asarray(b.verts, float), np.asarray(b.faces, np.int64), self.caliper["deflate"])
            b.verts_contact = r
            nv = len(r)
            Vw = r @ b.R.T + b.p
            V.append(Vw)
            F.append(np.asarray(b.faces, np.int64) + voff)
            if b.kind in ("static", "driven"):
                patch.append(np.full(nv, external_patch))
            else:
                patch.append(np.full(nv, next_patch)); next_patch += 1
            self.body_vslice.append(slice(voff, voff + nv))
            if b.kind == "static":
                cs.append(Vw.ravel())
            else:
                o = self.body_dof[b_i]
                cs.append(np.zeros(3*nv))
                vi = np.arange(voff, voff + nv)
                for k in range(3):
                    rows.append(3*vi + k); cols.append(np.full(nv, o + k)); vals.append(np.ones(nv))
                    for j in range(3):
                        rows.append(3*vi + k); cols.append(np.full(nv, o + 3 + 3*k + j)); vals.append(r[:, j])
            voff += nv
        self.n_coll = voff
        self.Vc0 = np.concatenate(V)
        self._F_body = [f.astype(np.int32) for f in F[1:]]
        self._E_extra = E_extra
        self.Fc = np.concatenate(F).astype(np.int32)
        self.P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(3*voff, self.n_dof))
        self.PT = self.P.T.tocsr()
        self.cvec = np.concatenate(cs)
        self.patch = np.concatenate(patch).astype(np.int32)
        # per-vertex contact half-thickness (sheets: t/2; rods: radius; bodies: 0)
        half = np.zeros(voff)
        half[:nf] = .5*self.vthick[:nf] + self.rod_radius_v
        self.vhalf = half
        self.vbody = np.full(voff, -1)
        for b_i, sl in enumerate(self.body_vslice):
            self.vbody[sl] = b_i
        mu_v = np.full(voff, self.mu_default)
        for k, sl in enumerate(self.sheet_slices):
            if self.sheets[k].mu is not None:
                mu_v[sl] = self.sheets[k].mu
        for b_i, sl in enumerate(self.body_vslice):
            mu_v[sl] = self.bodies[b_i].mu
        self.vmu = mu_v
        # friction classes: one integer code per (object kind, surface mu); a pair's coefficient comes from
        # pair_mu[(name_a, name_b)] when given, else the geometric mean of the two surfaces (vectorised in
        # ipctk's build through blend_mu)
        cls_names, cls_mu, code = [], [], np.zeros(voff)
        def _cls(name, mu):
            key = (name, float(mu))
            if key not in cls_names:
                cls_names.append(key); cls_mu.append(float(mu))
            return cls_names.index(key) + 1
        fm = {}
        for k, sl in enumerate(self.sheet_slices):
            f_ = self.sheets[k].fmat
            nm = ("mat:" + f_) if f_ else "sheet:" + self.sheets[k].name
            code[sl] = _cls(nm, mu_v[sl.start]); fm[nm] = f_
        for r, sl in zip(self.rods, getattr(self, "rod_slices", [])):
            nm = ("mat:" + r.fmat) if r.fmat else "rod"
            code[sl] = _cls(nm, mu_v[sl.start]); fm[nm] = r.fmat
        for b_i, sl in enumerate(self.body_vslice):
            f_ = self.bodies[b_i].fmat
            nm = ("mat:" + f_) if f_ else self.bodies[b_i].name.split(":")[0]
            code[sl] = _cls(nm, mu_v[sl.start]); fm[nm] = f_
        self.vclass = code
        table, missing = {}, []
        for i, (na, ma) in enumerate(cls_names):
            for j, (nb, mb) in enumerate(cls_names):
                v = float(np.sqrt(ma*mb))
                for key, val in self.pair_mu.items():
                    if tuple(sorted(key)) == tuple(sorted((na, nb))):
                        v = float(val)
                if fm.get(na) and fm.get(nb):
                    key = frozenset((fm[na], fm[nb]))
                    if key in self.mu_pairs:
                        v = float(self.mu_pairs[key])
                    else:
                        missing.append(tuple(sorted(key)))
                elif self.friction_strict:
                    missing.append((na, nb))
                table[(i + 1, j + 1)] = v
        if self.friction_strict and missing:
            raise ValueError(f"friction pairs missing from the ledger (params.FRICTION): {sorted(set(missing))}")
        self._mu_table = table
        self._friction_classes = [dict(name=n, mu=m) for n, m in cls_names]
        self.ccd = ipctk.AdditiveCCD(max_iterations=CCD_MAX_ITERATIONS)   # capped: returns a conservative (smaller) step
        self.broad = ipctk.LBVH()
        # full collision arrays (free vertex i = collision vertex i); _compact_collision drops the inactive spare
        # vertices of tearable sheets, so they are in no collision mesh and x_coll() always matches the mesh
        self._cfull = dict(n=voff, Vc0=self.Vc0, P=self.P, cvec=self.cvec, patch=self.patch, vhalf=self.vhalf,
                           vbody=self.vbody, vmu=self.vmu, vclass=self.vclass, body_vslice=list(self.body_vslice),
                           F_body=list(self._F_body))
        self._compact_collision()
        self._make_contact_meshes()

    def _compact_collision(self):
        f = self._cfull
        keep = np.ones(f["n"], bool)
        keep[:self.n_free] = self.active
        idx = np.flatnonzero(keep)
        cmap = np.full(f["n"], -1, np.int64)
        cmap[idx] = np.arange(len(idx))
        self._cmap = cmap
        drop = self.n_free - int(self.active.sum())
        self.n_coll = len(idx)
        if drop == 0:
            for k in ("Vc0", "P", "cvec", "patch", "vhalf", "vbody", "vmu", "vclass"):
                setattr(self, k, f[k])
        else:
            r3 = (3*idx[:, None] + np.arange(3)).ravel()
            self.Vc0, self.P, self.cvec = f["Vc0"][idx], f["P"][r3], f["cvec"][r3]
            self.patch, self.vhalf, self.vbody = f["patch"][idx], f["vhalf"][idx], f["vbody"][idx]
            self.vmu, self.vclass = f["vmu"][idx], f["vclass"][idx]
        self.PT = self.P.T.tocsr()
        self.body_vslice = [slice(sl.start - drop, sl.stop - drop) for sl in f["body_vslice"]]
        self._F_body = [(F - drop).astype(np.int32) for F in f["F_body"]]

    def _make_contact_meshes(self):
        """(Re)build the contact meshes from the live sheet triangles and the body surfaces. A fully torn
        triangle (damage 1) leaves the contact mesh: its zero-stiffness 'ghost' would otherwise keep both sides
        of a tear in contact and block objects passing through it. Called at build
        time, whenever the set of torn triangles changes, and on restore."""
        ipctk = self.ipctk
        alive = self.damage < 1. if len(self.damage) else np.ones(len(self.tris), bool)
        self._contact_alive = alive.copy()
        self.Fc = np.concatenate([self._cmap[self.tris[alive]].astype(np.int32)] + self._F_body).astype(np.int32)
        Ef = ipctk.edges(self.Fc) if len(self.Fc) else np.zeros((0, 2), np.int32)
        Ec = np.concatenate([Ef] + [e.astype(np.int32) for e in self._E_extra]) if self._E_extra else Ef
        self.Ec = np.ascontiguousarray(Ec, np.int32)
        self.mesh = ipctk.CollisionMesh(np.asfortranarray(self.Vc0), np.asfortranarray(self.Ec),
                                        np.asfortranarray(self.Fc))
        self.mesh.can_collide = self._contact_filter(self.patch, siblings=True)
        # contact sets: paper/rigid at the sheet caliper; rope against everything at its radius
        self.csets = [ContactSet(self, "paper", self.mesh, self.dmin)]
        if len(self.rod_seg):
            voff = self.n_coll
            rodv = np.zeros(voff, bool)
            for sl in self.rod_slices:
                rodv[self._cmap[sl.start:sl.stop]] = True
            pr = np.where(rodv, np.arange(voff) + 1, 0).astype(np.int32)
            Er = np.ascontiguousarray(np.concatenate([self._cmap[self.rod_seg].astype(np.int32), self.Ec]), np.int32)
            mr = ipctk.CollisionMesh(np.asfortranarray(self.Vc0), np.asfortranarray(Er), np.asfortranarray(self.Fc))
            flt = self._contact_filter(pr)
            names = [b.name for b in self.bodies]
            keys = {}
            for r, sl in zip(self.rods, self.rod_slices):
                for vi, bname in getattr(r, "attached", ()):
                    a_ = int(self._cmap[sl.start + vi])
                    bs = self.body_vslice[names.index(bname)]
                    for c_ in range(bs.start, bs.stop):
                        keys[(min(a_, c_), max(a_, c_))] = False
            if keys:
                flt = flt & ipctk.make_sparse_filter(keys, True)
            mr.can_collide = flt
            rr = float(self.rod_radius_v.max())
            # the rope touches (zero force) at its radius + the paper's core half + the
            # paper's range; a rope with its own contact range can be squeezed that much further in (dmin smaller)
            rng = max([r.mat.contact_range or 0. for r in self.rods] or [0.])
            if rng > 0.:
                kap = [r.mat.contact_kappa for r in self.rods if r.mat.contact_range]
                self.csets.append(ContactSet(self, "rope", mr, rr + self.caliper["deflate"] + self.dhat - rng, edges=Er,
                                             dhat=rng, kappa=kap[0] if kap and kap[0] else None))
            else:
                self.csets.append(ContactSet(self, "rope", mr, rr + self.caliper["deflate"], edges=Er))

    def _mass_matrix(self):
        nf = self.n_free
        rows, cols, vals = [np.arange(3*nf)], [np.arange(3*nf)], [np.repeat(self.vmass[:nf], 3)]
        for b_i, o in self.body_dof.items():
            b = self.bodies[b_i]
            if b.kind != "dynamic":
                continue
            rows.append(o + np.arange(3)); cols.append(o + np.arange(3)); vals.append(np.full(3, b.mass))
            S = np.asarray(b.S, float)
            for k in range(3):
                base = o + 3 + 3*k
                rr, cc = np.meshgrid(np.arange(3) + base, np.arange(3) + base, indexing="ij")
                rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(S.ravel())
        self.M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.n_dof, self.n_dof))
        # gravity generalized force
        fg = np.zeros(self.n_dof)
        fg[:3*nf] = (self.vmass[:nf, None]*self.g).ravel()
        for b_i, o in self.body_dof.items():
            b = self.bodies[b_i]
            if b.kind == "dynamic":
                fg[o:o+3] = b.mass*self.g
        self.fg = fg
        # inertia applies to free vertices and dynamic bodies only (driven bodies: springs)
        self.inertial = np.zeros(self.n_dof, bool)
        self.inertial[:3*nf] = True
        for b_i, o in self.body_dof.items():
            if self.bodies[b_i].kind == "dynamic":
                self.inertial[o:o+12] = True

    def _initial_kappa(self):
        """Barrier stiffness: IPC's rule (Li et al. 2020) via ipctk, floored so the barrier carries the
        heaviest supported object at a fraction of dhat."""
        m_avg = float(np.mean(self.vmass[:self.n_free])) if self.n_free else 1e-3
        diag = float(np.linalg.norm(self.Vc0.max(0) - self.Vc0.min(0))) if len(self.Vc0) else 1.
        k, self.kappa_max = self.ipctk.initial_barrier_stiffness(diag, self._barrier_fn(), self.dhat, m_avg,
                                                                 np.zeros(3*self.n_coll), np.zeros(3*self.n_coll))
        return max(k, 1e2)

    def check_initial(self):
        """Tick-zero audit: no intersections and every close pair farther apart than its minimum distance."""
        ipctk = self.ipctk
        xc = np.asfortranarray(self.x_coll())
        for cs in self.csets:
            if ipctk.has_intersections(cs.mesh, xc):
                raise ValueError(f"{cs.name}: initial state has intersecting surfaces")
            C = cs.collisions(xc)
            if len(C):
                dmin2 = C.compute_minimum_distance(cs.mesh, xc)
                if dmin2 <= cs.dmin**2:
                    raise ValueError(f"{cs.name}: initial pair at {math.sqrt(dmin2)*1e6:.1f} um <= dmin "
                                     f"{cs.dmin*1e6:.1f} um (place items with clearance)")
        self.initial_min_distance = {cs.name: (math.sqrt(cs.collisions(xc).compute_minimum_distance(cs.mesh, xc))
                                               if len(cs.collisions(xc)) else None) for cs in self.csets}

    def _barrier_fn(self):
        return self.ipctk.ClampedLogBarrier()

    # ------------------------------------------------------------------ state access
    def x_coll(self, q=None):
        q = self.q if q is None else q
        return (self.P @ q + self.cvec).reshape(-1, 3)

    def x_free(self, q=None):
        q = self.q if q is None else q
        return q[:3*self.n_free].reshape(-1, 3)

    def body_pose(self, b_i, q=None):
        """Body COM position and the polar (closest rotation) of its affine matrix."""
        b = self.bodies[b_i]
        if b.kind == "static":
            return b.p.copy(), b.R.copy()
        q = self.q if q is None else q
        o = self.body_dof[b_i]
        A = q[o+3:o+12].reshape(3, 3)
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        return q[o:o+3].copy(), R

    # ------------------------------------------------------------------ energies
    def _shell(self, x, want):
        """Membrane (orthotropic StVK, damage) + panel/crease hinges. Returns E, grad (n_free,3), COO."""
        T = self.tris
        E = 0.
        g = np.zeros_like(x)
        coo = ([], [], [])
        if not len(T):
            return E, g, coo
        b = self.bmat
        xt = x[T]
        F = np.matmul(xt.transpose(0, 2, 1), b)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1) - 1.), .5*((f2*f2).sum(1) - 1.), (f1*f2).sum(1)], 1)
        live = 1. - self.damage
        S = np.einsum("tij,tj->ti", self.C, e)*live[:, None]          # N/m membrane force (Voigt)
        sk = self._slack_idx
        Ck = None
        if sk is not None:                                            # crumpled triangles: slack_response
            rk, Sk, Ck, _ = slack_response(self.C[sk], e[sk], self.slack_c[sk], self.slack_phi[sk], tangent=want)
            e = e.copy()
            e[sk] = rk                                                # the elastic strain (what damage and tearing see)
            S[sk] = Sk*live[sk, None]
        w = self.area
        E += .5*(w*(e*S).sum(1)).sum()
        self._last_S, self._last_e = S, e
        J = np.zeros((len(T), 3, 3, 3))
        J[:, 0] = b[:, :, 0, None]*f1[:, None, :]
        J[:, 1] = b[:, :, 1, None]*f2[:, None, :]
        J[:, 2] = b[:, :, 0, None]*f2[:, None, :] + b[:, :, 1, None]*f1[:, None, :]
        J = J.reshape(len(T), 3, 9)
        gt = w[:, None]*np.einsum("ti,tij->tj", S, J)
        gidx, gval = [T.ravel()], [gt.reshape(-1, 3)]
        tc = self._topo_cache() if want else None
        if want:
            Cw = self.C*(w*live)[:, None, None]
            if Ck is not None:
                Cw[sk] = Ck*(w*live)[sk, None, None]
            Hm = np.matmul(np.matmul(J.transpose(0, 2, 1), Cw), J)
            # geometric stiffness w b S+ b^T (x) I3, S+ the PSD part of the 2x2 membrane force tensor (closed form),
            # added in place on the three (i, i) sub-blocks instead of through a (T,3,3,3,3) broadcast
            S2p = psd2(S[:, 0], S[:, 1], S[:, 2])
            Kg = w[:, None, None]*np.matmul(np.matmul(b, S2p), b.transpose(0, 2, 1))
            Hm4 = Hm.reshape(len(T), 3, 3, 3, 3)
            for i in range(3):
                Hm4[:, :, i, :, i] += Kg
            coo[0].append(tc["mem_rows"]); coo[1].append(tc["mem_cols"])
            coo[2].append(Hm.ravel())
        # hinges (panels and creases); a hinge next to a fully damaged triangle carries nothing
        H = self.hinge
        if len(H):
            theta, dth = hinge_angles(x, H)
            alive = np.minimum(live[self.hinge_tris[:, 0]], live[self.hinge_tris[:, 1]])
            d = wrap(theta - self.theta_p)
            k = self.k_hinge*alive
            E += (k*d*d).sum()
            gh = (2*k*d)[:, None, None]*dth
            gidx.append(H.ravel()); gval.append(gh.reshape(-1, 3))
            self._last_theta = theta
            if want:
                d12 = dth.reshape(-1, 12)
                Hb = (2*k)[:, None, None]*(d12[:, :, None]*d12[:, None, :])
                coo[0].append(tc["hin_rows"]); coo[1].append(tc["hin_cols"])
                coo[2].append(Hb.ravel())
        # membrane then hinge contributions, each vertex summed in that order
        g = _scatter3(np.concatenate(gidx), np.concatenate(gval), len(x))
        return E, g, coo

    def set_slack(self, c, phi, tri0=0):
        """Give triangles tri0 .. tri0 + len(c) the slack c (n, 2) <= 0 along axes at phi (n,) (slack_response)."""
        c = np.asarray(c, float)
        if (c > 0.).any() or (c < .5*(SLACK_MIN_STRETCH**2 - 1.) - 1e-12).any():
            raise ValueError("slack must lie between the Green strain of SLACK_MIN_STRETCH and 0")
        sl = slice(tri0, tri0 + len(c))
        self.slack_c[sl] = c
        self.slack_phi[sl] = np.asarray(phi, float)
        self._slack_refresh()

    def _slack_refresh(self):
        idx = np.flatnonzero((self.slack_c < 0.).any(1))
        self._slack_idx = idx if len(idx) else None

    def _slack_post(self):
        """After a step, the history of the crumpled triangles (and, while a start state is made, of the paper that
        crumples for the first time).
        1. Wrinkles pulled out do not come back by themselves: a triangle's slack shrinks to the hidden strain it has now
           (c <- k), so it is free to open further, up to its true length, and solid against being pushed back.
        2. It wrinkles again only under load: compressed beyond CRUMPLE_RESIDUAL x the stock's yield strain it hides the
           excess again (a wrinkled patch crushes at a small part of what fresh paper takes). The slack is a tensor:
           the excess is added along the most compressed direction and the triangle's slack axes are the principal axes
           of the sum (shear wrinkles a sheet along its compressed diagonal).
        3. With `crumple_virgin` (start-state generation only, bagnet.shipping_state) paper without slack crumples too,
           beyond its full yield strain. In a task the paper without slack has no in-plane plasticity.
        Returns the number of triangles whose slack grew."""
        virgin = getattr(self, "crumple_virgin", None)
        sk = self._slack_idx
        if sk is None and virgin is None:
            return 0
        if virgin is None and getattr(self, "_morph", None) is not None:
            # a start state that carries slack is being re-assembled (the tote with its handles): while its rest shape is
            # a blend the strains are fictitious (zero at the blend's start), and the ratchet would erase every slack
            return 0
        c, phi = self.slack_c, self.slack_phi
        sel = np.flatnonzero((c < 0.).any(1) | (virgin if virgin is not None else False))
        if not len(sel):
            return 0
        x = self.x_free()
        F = np.matmul(x[self.tris[sel]].transpose(0, 2, 1), self.bmat[sel])
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1) - 1.), .5*((f2*f2).sum(1) - 1.), (f1*f2).sum(1)], 1)
        r, _, _, k = slack_response(self.C[sel], e, c[sel], phi[sel], tangent=False)
        cs = np.maximum(c[sel], np.minimum(k, 0.))                 # 1. the ratchet
        cs[cs > -1e-9] = 0.
        has = (cs < 0.).any(1)
        ey = self._tri_prop("yield_strain")[sel]
        thr = np.where(has, CRUMPLE_RESIDUAL, 1.)*ey
        mean, rad = .5*(r[:, 0] + r[:, 1]), np.hypot(.5*(r[:, 0] - r[:, 1]), .5*r[:, 2])
        over = np.flatnonzero((mean - rad < -thr) & (self.damage[sel] < 1.))
        ph = phi[sel].copy()
        if len(over):                                              # 2. and 3. the slack tensor grows
            ca, sa = np.cos(ph[over]), np.sin(ph[over])
            K = np.zeros((len(over), 2, 2))
            for ci, (ux, uy) in ((cs[over, 0], (ca, sa)), (cs[over, 1], (-sa, ca))):
                K[:, 0, 0] += ci*ux*ux; K[:, 0, 1] += ci*ux*uy; K[:, 1, 0] += ci*ux*uy; K[:, 1, 1] += ci*uy*uy
            am = .5*np.arctan2(r[over, 2], r[over, 0] - r[over, 1]) + .5*np.pi
            ex = (mean - rad)[over] + thr[over]
            nx, ny = np.cos(am), np.sin(am)
            K[:, 0, 0] += ex*nx*nx; K[:, 0, 1] += ex*nx*ny; K[:, 1, 0] += ex*nx*ny; K[:, 1, 1] += ex*ny*ny
            lam, V = np.linalg.eigh(K)
            cs[over] = np.clip(lam, .5*(SLACK_MIN_STRETCH**2 - 1.), 0.)
            ph[over] = np.arctan2(V[:, 1, 0], V[:, 0, 0])
        self.slack_c[sel], self.slack_phi[sel] = cs, ph
        self._slack_refresh()
        return int(len(over))

    def membrane_state(self, x=None):
        """Per triangle at x (default: now): total Green strain e (n,3), elastic strain r (n,3), membrane force S
        (n,3), all Voigt in the stock's axes, and the hidden strain k (n,2) of crumpled triangles in their slack axes."""
        x = self.x_free() if x is None else x
        F = np.matmul(x[self.tris].transpose(0, 2, 1), self.bmat)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1) - 1.), .5*((f2*f2).sum(1) - 1.), (f1*f2).sum(1)], 1)
        live = 1. - self.damage
        S = np.einsum("tij,tj->ti", self.C, e)*live[:, None]
        r = e.copy()
        k = np.zeros((len(e), 2))
        sk = self._slack_idx
        if sk is not None:
            r[sk], Sk, _, k[sk] = slack_response(self.C[sk], e[sk], self.slack_c[sk], self.slack_phi[sk], tangent=False)
            S[sk] = Sk*live[sk, None]
        return e, r, S, k

    def _shell_energy(self, x):
        """Energy of _shell only (line search): the same operations for E and the same side effects (_last_S, _last_e,
        _last_theta), without gradient and Hessian."""
        T = self.tris
        E = 0.
        if not len(T):
            return E
        b = self.bmat
        xt = x[T]
        F = np.matmul(xt.transpose(0, 2, 1), b)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1) - 1.), .5*((f2*f2).sum(1) - 1.), (f1*f2).sum(1)], 1)
        live = 1. - self.damage
        S = np.einsum("tij,tj->ti", self.C, e)*live[:, None]
        sk = self._slack_idx
        if sk is not None:
            rk, Sk, _, _ = slack_response(self.C[sk], e[sk], self.slack_c[sk], self.slack_phi[sk], tangent=False)
            e = e.copy()
            e[sk] = rk
            S[sk] = Sk*live[sk, None]
        w = self.area
        E += .5*(w*(e*S).sum(1)).sum()
        self._last_S, self._last_e = S, e
        H = self.hinge
        if len(H):
            theta = hinge_theta(x, H)
            alive = np.minimum(live[self.hinge_tris[:, 0]], live[self.hinge_tris[:, 1]])
            d = wrap(theta - self.theta_p)
            k = self.k_hinge*alive
            E += (k*d*d).sum()
            self._last_theta = theta
        return E

    def _tri_prop(self, attr):
        """Per-triangle material property (array over tri_mat), cached while tri_mat and the materials are
        unchanged."""
        c = getattr(self, "_tprop", None)
        if c is None or c[0] is not self.tri_mat or not np.array_equal(c[1], self.tri_mat) or c[2] != [id(m) for m in self.mats]:
            c = self._tprop = (self.tri_mat, self.tri_mat.copy(), [id(m) for m in self.mats], {})
        d = c[3]
        if attr not in d:
            d[attr] = np.array([getattr(self.mats[m], attr) for m in self.tri_mat])
        return d[attr]

    def _topo_cache(self):
        """Index arrays that depend only on the mesh topology: the COO rows/cols of the membrane and hinge Hessian blocks
        (the same array objects while the topology is unchanged, so the Newton matrix assembly reuses their scatter
        maps). Rebuilt whenever tris or hinge differ from the cached copies (vertex splits edit both in place)."""
        c = getattr(self, "_tc", None)
        T, H = self.tris, self.hinge
        if c is None or c["n_free"] != self.n_free or not np.array_equal(c["tris"], T) or not np.array_equal(c["hinge"], H):
            idx = (3*T[:, :, None] + np.arange(3)).reshape(len(T), 9)
            hidx = (3*H[:, :, None] + np.arange(3)).reshape(len(H), 12)
            c = dict(n_free=self.n_free, tris=T.copy(), hinge=H.copy(),
                     mem_rows=np.repeat(idx, 9, axis=1).ravel(), mem_cols=np.tile(idx, 9).ravel(),
                     hin_rows=np.repeat(hidx, 12, axis=1).ravel(), hin_cols=np.tile(hidx, 12).ravel())
            nf = self.n_free
            v3 = 3*np.arange(nf)
            c["drag_rows"] = np.concatenate([v3 + r for r in range(3) for cc in range(3)])
            c["drag_cols"] = np.concatenate([v3 + cc for r in range(3) for cc in range(3)])
            self._tc = c
        return c

    def _rods_energy(self, x, want):
        E = 0.
        g = np.zeros_like(x)
        coo = ([], [], [])
        if len(self.rod_seg):
            s = self.rod_seg
            d = x[s[:, 1]] - x[s[:, 0]]
            L = np.linalg.norm(d, axis=1)
            u = d/L[:, None]
            ext = L - self.rod_rest
            E += .5*(self.rod_k*ext*ext).sum()
            f = (self.rod_k*ext)[:, None]*u
            np.add.at(g, s[:, 1], f); np.add.at(g, s[:, 0], -f)
            if want:
                # PSD spring Hessian: k u u^T + k max(0, 1 - L0/L)(I - u u^T)
                uu = u[:, :, None]*u[:, None, :]
                Kb = self.rod_k[:, None, None]*uu + (self.rod_k*np.maximum(0., 1. - self.rod_rest/L))[:, None, None]*(np.eye(3) - uu)
                blocks = np.zeros((len(s), 2, 3, 2, 3))
                blocks[:, 0, :, 0], blocks[:, 1, :, 1] = Kb, Kb
                blocks[:, 0, :, 1], blocks[:, 1, :, 0] = -Kb, -Kb
                idx = (3*s[:, :, None] + np.arange(3)).reshape(len(s), 6)
                coo[0].append(np.repeat(idx, 6, axis=1).ravel()); coo[1].append(np.tile(idx, 6).ravel())
                coo[2].append(blocks.reshape(len(s), 36).ravel())
        if len(self.rod_bend):
            # discrete bending: E = k/2 |x_{a-1} - 2 x_a + x_{a+1}|^2 / l^2 * l (curvature-squared, straight rest)
            bi = self.rod_bend
            lm = .5*(self.rod_rest[0] if len(self.rod_rest) else 1.)
            c = x[bi[:, 0]] - 2*x[bi[:, 1]] + x[bi[:, 2]]
            kk = self.rod_bend_k/np.maximum(lm*2, 1e-6)**2
            E += .5*(kk[:, None]*c*c).sum()
            gc = kk[:, None]*c
            np.add.at(g, bi[:, 0], gc); np.add.at(g, bi[:, 1], -2*gc); np.add.at(g, bi[:, 2], gc)
            if want:
                w_ = np.array([1., -2., 1.])
                blk = kk[:, None, None]*w_[None, :, None]*w_[None, None, :]
                for a in range(3):
                    for bb in range(3):
                        for k3 in range(3):
                            coo[0].append(3*bi[:, a] + k3); coo[1].append(3*bi[:, bb] + k3); coo[2].append(blk[:, a, bb])
        return E, g, coo

    def _bodies_energy(self, q, want):
        """Orthogonality (dynamic affine bodies) and target springs (driven bodies)."""
        E = 0.
        g = np.zeros(self.n_dof)
        coo = ([], [], [])
        dyn_H, dyn_o = [], []
        for b_i, o in self.body_dof.items():
            b = self.bodies[b_i]
            A = q[o+3:o+12].reshape(3, 3)
            if b.kind == "dynamic":
                vol = max(b.mass/500., 1e-6)
                kap = 1e9*vol                                       # ABD orthogonality stiffness
                Cm = A.T @ A - np.eye(3)
                E += kap*(Cm*Cm).sum()
                gA = 4*kap*(A @ Cm)
                g[o+3:o+12] += gA.ravel()
                if want:
                    # column c of the Hessian is its action on the unit matrix E_c (all nine at once); the PSD
                    # projection runs once for all dynamic bodies after the loop
                    dC = _E9T @ A + A.T @ _E9
                    dyn_H.append((4*kap*(_E9 @ Cm + A @ dC)).reshape(9, 9).T)
                    dyn_o.append(o)
            elif b.kind == "driven":
                p_hat, A_hat = self.targets.get(b_i, (b.p, b.R))
                p = q[o:o+3]
                dp = p - p_hat
                if b_i in self.finger_cmd:
                    a, s_cmd, F_cap, kf = self.finger_cmd[b_i]
                    P_perp = np.eye(3) - np.outer(a, a)
                    # across the closing axis the finger follows its hand through the same force-capped (Huber) spring as
                    # every other link
                    e_, g_, _ = huber3(P_perp @ dp, b.k_lin, b.f_cap)
                    E += e_
                    g[o:o+3] += P_perp @ g_
                    # finger mass along the closing axis (moves with the hand unless forced): keeps the
                    # force-capped drive well posed when the finger lags its command by more than F_cap/k
                    mf = FINGER_MASS/self._dt**2
                    du_in = float(a @ (p - self._finger_pred.get(b_i, p)))
                    E += .5*mf*du_in*du_in
                    g[o:o+3] += mf*du_in*a
                    u = float(a @ p) - s_cmd
                    sc = F_cap/kf
                    if abs(u) <= sc:
                        E += .5*kf*u*u; du, h = kf*u, kf
                    else:
                        # capped branch: near-zero curvature; the finger's axial inertia (mf) keeps the Newton
                        # matrix well posed (a full-stiffness matrix here advanced a lagging finger by only
                        # F_cap/kf = 0.12 mm per iteration: 24-iteration steps in free arm motion)
                        E += F_cap*abs(u) - .5*F_cap*sc; du, h = F_cap*np.sign(u), 1e-3*kf
                    g[o:o+3] += du*a
                    Hp = b.k_lin*P_perp + (h + mf)*np.outer(a, a)
                else:
                    # link surface compliance toward the robot simulator's pose, force-limited at the same cap
                    # as the reaction fed back to the robot (b.f_cap): both simulators agree on the largest
                    # load a link can put into the packaging; beyond it the obstacle lags its robot link
                    e_, g_, Hp = huber3(dp, b.k_lin, b.f_cap)
                    E += e_
                    g[o:o+3] += g_
                dA = A - A_hat
                kr = .5*b.k_rot
                nA = math.sqrt(float((dA*dA).sum()))
                cA = b.f_cap*AFFINE_LEVER if b_i not in self.finger_cmd else np.inf
                if nA <= cA/kr:
                    E += .5*kr*nA*nA
                    gA, hA = kr*dA.ravel(), np.full(9, kr)
                else:
                    E += cA*nA - .5*cA*cA/kr
                    gA, hA = cA*dA.ravel()/nA, np.full(9, kr)          # Newton matrix kr (see huber3)
                g[o+3:o+12] += gA
                if want:
                    rr, cc = np.meshgrid(np.arange(3) + o, np.arange(3) + o, indexing="ij")
                    coo[0].append(rr.ravel()); coo[1].append(cc.ravel()); coo[2].append(Hp.ravel())
                    d9 = np.arange(9) + o + 3
                    coo[0].append(d9); coo[1].append(d9); coo[2].append(hA)
        if dyn_H:
            Hk = psd_project(np.asarray(dyn_H))
            rr = (np.asarray(dyn_o)[:, None] + _RR9[None, :]).ravel()
            cc = (np.asarray(dyn_o)[:, None] + _CC9[None, :]).ravel()
            coo[0].append(rr); coo[1].append(cc); coo[2].append(Hk.ravel())
        return E, g, coo

    def _point(self, body, r, q):
        """World position of a body point and its (3, n_dof) sparse Jacobian rows as (cols, vals)."""
        b = self.bodies[body]
        if b.kind == "static":
            return b.R @ r + b.p, None
        o = self.body_dof[body]
        A = q[o+3:o+12].reshape(3, 3)
        cols = np.array([[o + k] + [o + 3 + 3*k + j for j in range(3)] for k in range(3)])
        vals = np.array([[1., r[0], r[1], r[2]]]*3)
        return A @ r + q[o:o+3], (cols, vals)

    def _pins_energy(self, q, want):
        E = 0.
        g = np.zeros(self.n_dof)
        coo = ([], [], [])
        for pin in self.pins:
            xa, Ja = self._point(pin.a, pin.ra, q)
            xb, Jb = (pin.rb, None) if pin.b is None else self._point(pin.b, pin.rb, q)
            d = xa - xb
            E += .5*pin.k*d @ d
            parts = [(Ja, 1.), (Jb, -1.)]
            for J, sgn in parts:
                if J is None:
                    continue
                cols, vals = J
                for k in range(3):
                    np.add.at(g, cols[k], sgn*pin.k*d[k]*vals[k])
            if want:
                for J1, s1 in parts:
                    for J2, s2 in parts:
                        if J1 is None or J2 is None:
                            continue
                        for k in range(3):
                            rr, cc = np.meshgrid(J1[0][k], J2[0][k], indexing="ij")
                            coo[0].append(rr.ravel()); coo[1].append(cc.ravel())
                            coo[2].append((s1*s2*pin.k*np.outer(J1[1][k], J2[1][k])).ravel())
        return E, g, coo

    def _body_hinges_energy(self, q, want):
        E = 0.
        g = np.zeros(self.n_dof)
        coo = ([], [], [])
        for hb in self.hinges_b:
            X, Js = [], []
            for body, r in hb.pts:
                x, J = self._point(body, np.asarray(r, float), q)
                X.append(x); Js.append(J)
            X = np.asarray(X)
            th, dth = hinge_angles(X, np.array([[0, 1, 2, 3]]))
            if hb.theta_p is None:
                hb.theta_p = float(th[0])
            d = float(wrap(th - hb.theta_p)[0])
            K = hb.law.k*hb.length
            E += .5*K*d*d
            for a in range(4):
                if Js[a] is None:
                    continue
                cols, vals = Js[a]
                for k in range(3):
                    np.add.at(g, cols[k], K*d*dth[0, a, k]*vals[k])
            if want:
                rows_, vals_ = [], []
                for a in range(4):
                    if Js[a] is None:
                        continue
                    for k in range(3):
                        rows_.append(Js[a][0][k]); vals_.append(Js[a][1][k]*dth[0, a, k])
                if rows_:
                    rr = np.concatenate(rows_); vv = np.concatenate(vals_)
                    R1, R2 = np.meshgrid(rr, rr, indexing="ij")
                    coo[0].append(R1.ravel()); coo[1].append(R2.ravel()); coo[2].append((K*np.outer(vv, vv)).ravel())
        return E, g, coo

    # ------------------------------------------------------------------ IPC
    def _collisions(self, xc):
        return self.csets[0].collisions(xc)

    # ------------------------------------------------------------------ assembly
    def _energy(self, q, Cn, Tc, xc_prev, q_tilde, dt, want=False):
        """Total incremental potential at q (and its gradient/Hessian if want)."""
        ipctk = self.ipctk
        t0 = time.perf_counter()
        nf3 = 3*self.n_free
        x = q[:nf3].reshape(-1, 3)
        dq = q - q_tilde
        Mdq = self.M @ dq
        E = .5*dq @ Mdq/dt**2 - self.fg @ q
        if not want:
            # line-search evaluation: the energy only, summed term by term in the same order as below (the gradients
            # of the large terms are not formed)
            E += self._shell_energy(x) + self._rods_energy(x, False)[0] + self._tear_bond_energy(x, False)[0]
            E += (self._bodies_energy(q, False)[0] + self._pins_energy(q, False)[0] + self._body_hinges_energy(q, False)[0]
                  + self._extras(q, xc_prev, dt, False)[0])
            t1 = time.perf_counter()
            xc = (self.P @ q + self.cvec).reshape(-1, 3)
            xcF = np.asfortranarray(xc)
            U = np.asfortranarray(xc - xc_prev + self._stick)
            for cs, Cn_k, Tc_k in zip(self.csets, Cn, Tc):
                E += cs.energy(Cn_k, Tc_k, xcF, U)
            self.stats["ipc_s"] += time.perf_counter() - t1
            self.stats["assemble_s"] += t1 - t0
            return E
        g = Mdq/dt**2 - self.fg
        coo = ([], [], [])
        Es, gs, c1 = self._shell(x, want)
        Er, gr, c2 = self._rods_energy(x, want)
        Et, gt_, ct = self._tear_bond_energy(x, want)
        E += Es + Er + Et
        g[:nf3] += (gs + gr + gt_).ravel()
        for k in range(3):
            c2[k].extend(ct[k])
        Eb, gb, c3 = self._bodies_energy(q, want)
        Ep, gp, c4 = self._pins_energy(q, want)
        Eh, gh, c5 = self._body_hinges_energy(q, want)
        Ex, gx, c6 = self._extras(q, xc_prev, dt, want)
        E += Eb + Ep + Eh + Ex
        g += gb + gp + gh + gx
        for c in (c1, c2, c3, c4, c5, c6):
            for k in range(3):
                coo[k].extend(c[k])
        t1 = time.perf_counter()
        xc = (self.P @ q + self.cvec).reshape(-1, 3)
        xcF = np.asfortranarray(xc)
        U = np.asfortranarray(xc - xc_prev + self._stick)
        Hc = None
        gc = np.zeros(3*self.n_coll) if want else None
        for cs, Cn_k, Tc_k in zip(self.csets, Cn, Tc):
            E += cs.energy(Cn_k, Tc_k, xcF, U)
            if want:
                g_k, H_k = cs.derivatives(Cn_k, Tc_k, xcF, U, parts=True)
                gc += g_k
                if H_k:
                    Hc = H_k if Hc is None else Hc + H_k       # a list of (matrix, scale) terms
        if want:
            g += self.PT @ gc
            self._last_gc = gc
        self.stats["ipc_s"] += time.perf_counter() - t1
        self.stats["assemble_s"] += t1 - t0
        if not want:
            return E
        return E, g, coo, Hc

    def _drag_begin(self, q0, dt):
        """Quadratic air drag normal to the sheet, f = -1/2 rho C_D A |v_n| v_n [handbook, C_D 1.17 broadside],
        integrated as an implicit damper with the coefficient taken at the step start."""
        self._drag = None
        if not len(self.tris):
            return
        x = q0[:3*self.n_free].reshape(-1, 3)
        T = self.tris
        fn = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
        vn = np.zeros_like(x)
        for k in range(3):
            np.add.at(vn, T[:, k], fn)
        nrm = vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
        v = self.v[:3*self.n_free].reshape(-1, 3)
        un = np.abs((v*nrm).sum(1))
        area = np.zeros(self.n_free)
        np.add.at(area, T.ravel(), np.repeat(self.area/3, 3))
        cd = self._tri_prop("drag_cd").mean() if len(self.tri_mat) else 1.17
        c = .5*AIR_RHO*cd*area*np.maximum(un, 1e-3)         # N s/m per vertex (floor: still air at 1 mm/s)
        self._drag = (nrm, c, x.copy())

    def _drag_term(self, q, dt, want):
        g = np.zeros(self.n_dof)
        coo = ([], [], [])
        if self._drag is None:
            return 0., g, coo
        nrm, c, x0 = self._drag
        x = q[:3*self.n_free].reshape(-1, 3)
        dn = ((x - x0)*nrm).sum(1)
        E = .5*(c/dt*dn*dn).sum()
        g[:3*self.n_free] = ((c/dt*dn)[:, None]*nrm).ravel()
        if want:
            blk = (c/dt)[:, None, None]*nrm[:, :, None]*nrm[:, None, :]
            tc = self._topo_cache()
            # the nine (r, cc) component chunks, concatenated (cached rows/cols)
            coo[0].append(tc["drag_rows"]); coo[1].append(tc["drag_cols"]); coo[2].append(blk.transpose(1, 2, 0).ravel())
        return E, g, coo

    def _extras(self, q, xc_prev, dt, want):
        """Bonds (tape), lid air films, drag, assisted fixture forces/springs."""
        E = 0.
        g = np.zeros(self.n_dof)
        coo = ([], [], [])
        nf3 = 3*self.n_free
        Ed, gd_, cd_ = self._drag_term(q, dt, want)
        E += Ed
        g += gd_
        for k in range(3):
            coo[k].extend(cd_[k])
        if self.fixture_force is not None:
            E -= (self.fixture_force.ravel()*q[:nf3]).sum()
            g[:nf3] -= self.fixture_force.ravel()
        for (ids, tgt, k, cap) in self.fixture_springs:
            ids = np.asarray(ids)
            x = q[:nf3].reshape(-1, 3)[ids]
            d = x - tgt
            n = np.linalg.norm(d, axis=1)
            sc = cap/k
            inside = n <= sc
            e_ = np.where(inside, .5*k*n*n, cap*n - .5*cap*sc)
            E += e_.sum()
            f = np.where(inside[:, None], k*d, cap*d/np.maximum(n, 1e-12)[:, None])
            gg = g[:nf3].reshape(-1, 3)
            np.add.at(gg, ids, f)
            if want:
                # one chunk for all held vertices
                kk = np.where(inside, float(k), cap/np.maximum(n, 1e-12)*.3)
                r3 = (3*ids[:, None] + np.arange(3)).ravel()
                coo[0].append(r3); coo[1].append(r3); coo[2].append(np.repeat(kk, 3))
        for fn in getattr(self, "extra_terms", []):
            e_, g_, c_ = fn(self, q, xc_prev, dt, want)
            E += e_
            g += g_
            for k in range(3):
                coo[k].extend(c_[k])
        return E, g, coo

    # ------------------------------------------------------------------ step
    def set_driven(self, b_i, p, R):
        self.targets[b_i] = (np.asarray(p, float), np.asarray(R, float))

    def step(self, dt):
        if self.fault:
            raise RuntimeError(self.fault)
        ipctk = self.ipctk
        q0 = self.q.copy()
        xc0 = (self.P @ q0 + self.cvec).reshape(-1, 3)
        q_tilde = q0 + dt*self.v*self.inertial
        self._dt = dt
        self._finger_pred = {}
        for b_i in self.finger_cmd:
            o = self.body_dof[b_i]
            prev = getattr(self, "_targets_prev", {}).get(b_i)
            tgt = self.targets.get(b_i)
            shift = (tgt[0] - prev[0]) if (prev is not None and tgt is not None) else 0.
            self._finger_pred[b_i] = q0[o:o+3] + shift
        self._targets_prev = {k: (v[0].copy(), v[1].copy()) for k, v in self.targets.items()}
        self._drag_begin(q0, dt)
        for fn in getattr(self, "pre_terms", []):
            fn(self, dt)
        # lagged friction: normal forces and tangent bases at the start of the step
        if len(self._stick) != len(xc0):             # the collision mesh changed (a tear): the stick memory starts again
            self._stick = np.zeros_like(xc0)
        Tc = [cs.tangential(xc0, dt) for cs in self.csets]
        C0 = [cs._C0 for cs in self.csets]
        q = q0.copy()
        self._ws_C = None
        if WARM_START:
            q = self._warm_start(q0, dt)
        fin_cands = None
        it, converged = 0, False
        alpha_min = 1.
        step_max = body_max = float("nan")      # (a Hessian failure on the first iteration leaves no update to report)
        Cn = None
        for it in range(1, self.newton_max + 1):
            xc = (self.P @ q + self.cvec).reshape(-1, 3)
            if Cn is None:
                # the pairs at the first iterate are already known: the warm start's (from its swept candidates) or,
                # at q0, the ones the friction bases were built from
                Cn = self._ws_C if self._ws_C is not None else C0
            try:
                E0, g, coo, Hc = self._energy(q, Cn, Tc, xc0, q_tilde, dt, want=True)
            except RuntimeError as exc:
                # ipctk could not PSD-project a contact Hessian (extreme squeeze). The current iterate is a
                # feasible, energy-decreasing state: end the step there, unconverged, and count it.
                self.stats["hessian_failures"] = self.stats.get("hessian_failures", 0) + 1
                self.stats["last_hessian_failure"] = str(exc)[:120]
                break
            ts = time.perf_counter()
            H = self._hessian(coo, Hc, dt)
            dq = self._solve(H, -g)
            self.stats["solve_s"] += time.perf_counter() - ts
            if not np.all(np.isfinite(dq)):
                self.stats["nonfinite_directions"] = self.stats.get("nonfinite_directions", 0) + 1
                break
            step_max = np.abs(dq[:3*self.n_free]).max() if self.n_free else 0.
            body_max = max([np.abs(dq[o:o+3]).max() + .05*np.abs(dq[o+3:o+12]).max()
                            for o in self.body_dof.values()] or [0.])
            tc = time.perf_counter()
            xc1 = (self.P @ (q + dq) + self.cvec).reshape(-1, 3)
            cands = [cs.sweep(xc, xc1) for cs in self.csets]
            alpha = min([1.] + [cs.ccd(xc, xc1, cd) for cs, cd in zip(self.csets, cands)])
            alpha_ccd = alpha
            self.stats["ccd_s"] += time.perf_counter() - tc
            if alpha <= 0. and it == 1:
                # the collision check allows no motion at all from the state this step starts in: some pair is already
                # closer than CCD_FRACTION x its minimum distance. Nothing can move from here. Such a state is a
                # fault of the start state or of a fixture's placement, and is reported as one.
                self.fault = ("a contact pair starts this step closer than the collision check's minimum separation "
                              f"({CCD_FRACTION:g} x its minimum distance): nothing can move; the state is invalid")
                self.stats["locked_start"] = self.stats.get("locked_start", 0) + 1
                raise RuntimeError(self.fault)
            if max(step_max, body_max) < self.tol and alpha >= 1.:
                if self._contact_forces_settled(Hc, dq):
                    q = q + dq
                    converged = True
                    fin_cands = cands
                    break
                self.stats["force_rule_iterations"] = self.stats.get("force_rule_iterations", 0) + 1
            gd = g @ dq
            accepted = False
            while alpha >= 1e-8:
                qa = q + alpha*dq
                xa = (self.P @ qa + self.cvec).reshape(-1, 3)
                Ca = [cs.collisions(xa, cd) for cs, cd in zip(self.csets, cands)]
                Ea = self._energy(qa, Ca, Tc, xc0, q_tilde, dt, want=False)
                if np.isfinite(Ea) and Ea <= E0 + 1e-4*alpha*gd + 1e-12*abs(E0) and self._gaps_ok(Ca, xa):
                    accepted = True
                    break
                alpha *= .5
            alpha_min = min(alpha_min, alpha)
            if self.verbose:
                ncol = sum(len(c) for c in Cn)
                nf3 = 3*self.n_free
                parts = [("free", np.linalg.norm(g[:nf3]))]
                for b_i, o in self.body_dof.items():
                    gt, ga = np.linalg.norm(g[o:o+3]), np.linalg.norm(g[o+3:o+12])
                    if gt + ga > 1.:
                        parts.append((self.bodies[b_i].name.split("/")[-1], round(float(gt), 2), round(float(ga), 3),
                                      "dq", float(np.abs(dq[o:o+3]).max()), float(np.abs(dq[o+3:o+12]).max())))
                print("      g-parts", parts, flush=True)
                print(f"    it {it}: E0 {E0:.6e} |g| {np.linalg.norm(g):.3e} g.dq {gd:.3e} step_max {step_max:.2e} "
                      f"body_max {body_max:.2e} ccd_alpha {alpha_ccd:.3e} ls_alpha {alpha:.3e} ncoll {ncol} "
                      f"accepted {accepted}", flush=True)
            if not accepted:
                # no feasible decrease along dq: keep the current (feasible) iterate and end the step unconverged
                self.stats["line_search_failures"] = self.stats.get("line_search_failures", 0) + 1
                break
            q = qa
            Cn = Ca                                       # the accepted point's pairs (from the swept candidates)
        self.stats["newton"] += it
        self.stats["last_iters"] = it
        self.stats["max_iters"] = max(self.stats["max_iters"], it)
        self.stats["min_alpha"] = min(self.stats["min_alpha"], alpha_min)
        if not converged:
            self.stats["unconverged"] += 1
            self.stats["last_unconverged_update_m"] = float(max(step_max, body_max))
        if not np.all(np.isfinite(q)):
            self.fault = "Non-finite state"
            raise FloatingPointError(self.fault)
        self.v = (q - q0)/dt
        self.v[~self.inertial] = 0.
        self.q = q
        # contact forces at the converged state (reactions on robot links, diagnostics)
        xc = (self.P @ q + self.cvec).reshape(-1, 3)
        xcF, U = np.asfortranarray(xc), np.asfortranarray(xc - xc0 + self._stick)
        gc = np.zeros(3*self.n_coll)
        ncol = 0
        # the final state's pairs: the narrow phase over the last sweep's candidates when Newton ended on a full step,
        # else the current iterate's pairs (accepted by the line search), instead of a fresh broad phase
        if fin_cands is not None:
            Cfin = [cs.collisions(xc, cd) for cs, cd in zip(self.csets, fin_cands)]
        elif Cn is not None:
            Cfin = Cn
        else:
            Cfin = [cs.collisions(xc) for cs in self.csets]
        for cs, Tc_k, Cf in zip(self.csets, Tc, Cfin):
            ncol += len(Cf)
            gc += cs.derivatives(Cf, Tc_k, xcF, U, hessian=False)[0]
        self._last_gc = gc
        self._last_collisions = ncol
        self._stick_update(xc - xc0, dt)
        self._step_converged = bool(converged)      # tear events record whether they were decided on a converged state
        self._reactions(q, xc0, dt)
        self._post_step(dt)
        self.time += dt
        self.steps += 1
        self.stats["steps"] += 1

    def _stick_update(self, dx, dt):
        """Static friction: what each collision vertex remembers of its own slow motion.

        The friction potential is a smooth function of the tangential displacement u of a contact pair, zero at u = 0 and
        saturated at the Coulomb limit beyond eps = eps_v dt. Measured from the start of each step (IPC's lagged
        friction), u starts every step at zero, so a pair can only carry a tangential load by slipping eps x load /
        (2 x limit) again in every step: any loaded contact creeps at eps_v x load / (2 x limit), 32-34 um/s for the flat
        tote lying on its back handle 0.9 degrees off level. A smaller eps_v removes the creep
        only in proportion and stalls the contact solver: on the tote, touching steps at the iteration cap rise from 37 %
        at 1e-3 m/s to 77 % at 1e-4 and 94 % at 1e-5 (measured).
        Here each collision vertex keeps a stick displacement b (|b| <= eps): the displacement it has accumulated while
        moving slower than the transition velocity. A pair's u is then the difference of its vertices' (step displacement
        + b), so a resting contact holds its load with the displacement it already has and does not move again: an
        elastic bristle of length eps (4 um) that saturates at the Coulomb limit, the usual stick model (Dahl). A vertex
        moving faster than eps_v forgets b (weight w below), and its pairs see IPC's lagged friction alone; the
        potential, its limit and its stiffness are unchanged. Static bodies never move and keep b = 0."""
        eps = self.eps_v*dt
        if eps <= 0. or not self.stick_friction:
            return
        d = np.linalg.norm(dx, axis=1)
        b = self._stick + dx
        nb = np.linalg.norm(b, axis=1)
        b *= np.minimum(1., eps/np.maximum(nb, 1e-300))[:, None]
        y = np.clip((d/eps - STICK_FULL)/(1. - STICK_FULL), 0., 1.)
        self._stick = b*(1. - y*y*(3. - 2.*y))[:, None]

    def _warm_start(self, q0, dt):
        """Start Newton with every robot obstacle at the minimum of its own coupling energy, when that move is
        collision-free. Driven bodies carry no inertia: a link's minimum is its target pose (Huber spring), and a
        finger's closing coordinate solves mf (s - s_pred) + clip(kf (s - s_cmd), -F, F) = 0 in closed form. This only
        changes the starting guess, not the solution; without it a finger driven at its force cap in free air sits
        on the cap's nearly flat branch and Newton spends two extra iterations per step placing it (4.0 instead of 1.8
        on the counter)."""
        q = q0.copy()
        moved = False
        for b_i, o in self.body_dof.items():
            b = self.bodies[b_i]
            if b.kind != "driven":
                continue
            p_hat, A_hat = self.targets.get(b_i, (b.p, b.R))
            p_hat = np.asarray(p_hat, float)
            if b_i in self.finger_cmd:
                a, s_cmd, F, kf = self.finger_cmd[b_i]
                s_ = finger_axis_minimum(FINGER_MASS/dt**2, float(a @ self._finger_pred.get(b_i, q0[o:o+3])), kf, s_cmd, F)
                p_new = p_hat - a*float(a @ p_hat) + a*s_
            else:
                p_new = p_hat
            q[o:o+3] = p_new
            q[o+3:o+12] = np.asarray(A_hat, float).ravel()
            moved = True
        if not moved:
            return q0
        xa = (self.P @ q0 + self.cvec).reshape(-1, 3)
        xb = (self.P @ q + self.cvec).reshape(-1, 3)
        cands = [cs.sweep(xa, xb) for cs in self.csets]
        alpha = min([1.] + [cs.ccd(xa, xb, cd) for cs, cd in zip(self.csets, cands)])
        if alpha < 1.:
            self.stats["warm_start_rejected"] = self.stats.get("warm_start_rejected", 0) + 1
            return q0
        C = [cs.collisions(xb, cd) for cs, cd in zip(self.csets, cands)]
        if not self._gaps_ok(C, xb):
            self.stats["warm_start_rejected"] = self.stats.get("warm_start_rejected", 0) + 1
            return q0
        self.stats["warm_start_used"] = self.stats.get("warm_start_used", 0) + 1
        self._ws_C = C
        return q

    def _contact_forces_settled(self, Hc, dq):
        """Does the Newton update dq leave the contact forces as they are?

        The size of the update alone (`tol`, 10 um in the task worlds) cannot end the iteration. Two parts of the
        contact model change over far less than that, so a small update says nothing about them:
          * friction goes from zero to its Coulomb limit over eps = eps_v dt = 4 um of tangential displacement. From a
            state at rest the first update is (load) / (limit / eps): under 10 um for any load below 2.5 x the limit.
            Accepting it, the next step starts from zero again, and the contact carries up to 2.5 x its Coulomb limit
            while creeping at most 2.5 mm/s: a block with friction 0.35 stands on a 35 degree incline;
          * the barrier's force doubles over a fraction of a micrometre at these loads, so after a small update the
            normal force is off by tens of percent, always on the high side (a Newton step on a convex force law lands
            on the stiff side), and the next step's friction, which uses that normal force, is too high: 0.38 to
            1.0 for a coefficient of 0.35 on a sliding block.
        The update's effect on the contact forces is predicted with the contact Hessian the iteration already has
        (barrier and friction, per collision vertex). The iteration may end only if no vertex's contact force changes
        by more than FORCE_RTOL (5 %) of the largest contact force on any vertex. Otherwise the update is taken through the
        line search and the iteration goes on; a step that reaches the iteration cap this way is counted unconverged."""
        if self.force_rtol is None or not Hc:
            return True
        dx = self.P @ dq
        dg = np.zeros(3*self.n_coll)
        for H, scale in Hc:
            dg += (H @ dx) if scale is None else scale*(H @ dx)
        change = np.linalg.norm(dg.reshape(-1, 3), axis=1).max()
        force = np.linalg.norm(self._last_gc.reshape(-1, 3), axis=1).max()
        return bool(change <= max(self.force_rtol*force, FORCE_ATOL))

    def _gaps_ok(self, C, xc):
        """Every active pair keeps at least GAP_FLOOR beyond its thickness: no accepted state sits so close to
        the barrier singularity that its Hessian stops being representable (states at 1e-20 m made the PSD
        projection fail)."""
        X = np.asfortranarray(xc)
        for cs, Ck in zip(self.csets, C):
            if len(Ck) and math.sqrt(max(Ck.compute_minimum_distance(cs.mesh, X), 0.)) - cs.dmin < GAP_FLOOR:
                return False
        return True

    def _ccd_min(self):
        return 0.

    def _pair_mu(self, ids):
        mus = self.vmu[ids]
        bodies = set(int(b) for b in self.vbody[ids] if b >= 0)
        key = None
        names = sorted(self.bodies[b].name.split(":")[0] for b in bodies) if bodies else []
        for k, v in self.pair_mu.items():
            if all(n in names for n in k) and len(k) == len(names):
                return v
        return float(np.sqrt(mus.min()*mus.max()))

    def _hessian(self, coo, Hc, dt):
        """The Newton matrix: elastic and coupling blocks (coo), M/dt^2, the contact Hessian mapped to the DOFs
        (P^T Hc P), 1e-9 on the diagonal (driven-body rotation DOFs, empty rows) and identity rows for inactive spare
        vertices. Returned as its UPPER TRIANGLE (the half PARDISO reads for a symmetric matrix), a sorted CSR matrix
        in a sparsity pattern that persists across Newton iterations and steps (_NewtonPattern): entries that are
        zero in this iteration are stored as explicit zeros, so PARDISO's analysis of the pattern can be reused."""
        t0 = time.perf_counter()
        n = self.n_dof
        pat = getattr(self, "_npat", None)
        if pat is None or pat.n != n:
            pat = self._npat = _NewtonPattern(n)
        chunks = list(zip(*coo)) + [self._base_chunk(dt)]
        H = pat.assemble(chunks, self._contact_chunk(Hc) if Hc is not None else None)
        self.stats["hessian_s"] = self.stats.get("hessian_s", 0.) + time.perf_counter() - t0
        return H

    def _base_chunk(self, dt):
        """(rows, cols, values) of M/dt^2, the 1e-9 diagonal and the inactive identity rows, upper entries only; the
        same array objects while M, dt and the inactive set are unchanged."""
        key = (id(self.M), dt, id(self._inactive_diag))
        bc = getattr(self, "_bchunk", None)
        if bc is None or bc[0] != key:
            n = self.n_dof
            Mu = sp.triu(self.M/dt**2, format="coo")
            parts = [(Mu.row, Mu.col, Mu.data), (np.arange(n), np.arange(n), np.full(n, 1e-9))]
            if self._inactive_diag is not None:
                Di = self._inactive_diag.tocoo()
                parts.append((Di.row, Di.col, Di.data))
            r, c, v = (np.concatenate([np.asarray(p[i], np.int64 if i < 2 else float) for p in parts]) for i in range(3))
            bc = self._bchunk = (key, (r, c, v), (self.M, self._inactive_diag))     # refs keep the ids unique
        return bc[1]

    def _contact_chunk(self, Hc):
        """P^T Hc P as COO (rows, cols, values), upper entries only. Hc: a sparse matrix or a list of (matrix, scale)
        terms (ContactSet.derivatives(parts=True)). A collision coordinate of a free vertex maps to exactly one DOF with
        coefficient 1 (x_coll = P q + c), so the free-free entries are an index map (the value Hc_ij); entries
        touching a body vertex go through the sparse product P^T H P; static vertices have empty rows in P and drop
        out."""
        terms = Hc if isinstance(Hc, list) else [(Hc, None)]
        rs, cs_, vs = [], [], []
        for M, scale in terms:
            M = M.tocoo()
            rs.append(M.row); cs_.append(M.col)
            vs.append(M.data*scale if scale is not None else M.data)
        r, c, v = np.concatenate(rs), np.concatenate(cs_), np.concatenate(vs)
        direct, cls = self._pmap()
        cr, cc = cls[r], cls[c]
        # free vertices keep their order in the DOFs, so an upper entry stays upper; body DOFs follow all free DOFs
        ff = (cr == 1) & (cc == 1) & (r <= c)
        out = [(direct[r[ff]], direct[c[ff]], v[ff])]
        other = ((cr == 2) | (cc == 2)) & (cr > 0) & (cc > 0) & ((r <= c) | ((cr == 2) & (cc == 2)))
        if other.any():
            sub = sp.csr_matrix((v[other], (r[other], c[other])), shape=(3*self.n_coll, 3*self.n_coll))
            Hd = (self.PT @ sub @ self.P).tocoo()
            m = Hd.row <= Hd.col
            out.append((Hd.row[m].astype(np.int64), Hd.col[m].astype(np.int64), Hd.data[m]))
        return tuple(np.concatenate([o[k] for o in out]) for k in range(3))

    def _pmap(self):
        """Per collision coordinate: its DOF when P's row is a single 1 (a free vertex), else -1; and its class (0 static:
        empty row, 1 free, 2 body). The free DOFs increase with the collision coordinates (checked here)."""
        pm = getattr(self, "_pm", None)
        if pm is None or pm[0] is not self.P:
            P = self.P.tocsr()
            cnt = np.diff(P.indptr)
            direct = np.full(P.shape[0], -1, np.int64)
            one = cnt == 1
            first = P.indptr[:-1][one]
            ok = P.data[first] == 1.
            direct[np.flatnonzero(one)[ok]] = P.indices[first[ok]]
            cls = np.where(direct >= 0, 1, np.where(cnt > 0, 2, 0)).astype(np.int8)
            d = direct[direct >= 0]
            nf3 = 3*self.n_free
            if np.any(np.diff(d) <= 0) or (len(d) and d.max() >= nf3) or np.any(P.indices[P.indptr[:-1][cls == 2]] < nf3):
                cls[cls == 1] = 2                       # not the expected layout: everything through the sparse product
            pm = self._pm = (self.P, direct, cls)
        return pm[1], pm[2]

    def _solve(self, H, rhs):
        import pypardiso
        if not hasattr(self, "_pardiso"):
            self._pardiso = pypardiso.PyPardisoSolver()
            self._pardiso.set_matrix_type(2)
            try:
                self._pardiso.libmkl.MKL_Set_Num_Threads(SOLVER_THREADS)
            except AttributeError:
                pass
        pat = getattr(H, "_gb_pattern", None)
        if pat is not None:
            try:
                return pat.solve(self._pardiso, H, rhs)
            except Exception as exc:
                # a failed factorisation falls back to a fresh general (LU, mtype 11) solver on the full
                # matrix, and that solver stays in use for later calls (which then receive the upper triangle)
                self.stats["solver_fallbacks"] = self.stats.get("solver_fallbacks", 0) + 1
                self.stats["last_solver_fallback"] = str(exc)[:120]
                pat.reset_analysis()
                Hf = (H + H.T - sp.diags(H.diagonal())).tocsr()
                self._pardiso = pypardiso.PyPardisoSolver()
                return self._pardiso.solve(Hf, rhs)
        Hu = sp.triu(H, format="csr")
        Hu.sort_indices()
        try:
            return self._pardiso.solve(Hu, rhs)
        except Exception:
            self._pardiso = pypardiso.PyPardisoSolver()
            return self._pardiso.solve(H, rhs)

    # ------------------------------------------------------------------ after the step
    def _reactions(self, q, xc_prev, dt):
        """Contact wrench on every driven body that touches something, about its frame origin, as the robot must feel
        it: the force and moment its coupling (link spring; finger cross-axis spring, closing drive and axial inertia)
        transmits at the accepted state. At equilibrium this equals the barrier-plus-friction wrench on the body. Read
        from the barrier side instead, a state accepted by the Newton step-size test right next to
        the stiff barrier could report kilonewtons (2.9 kN on a finger) that a 10 um step would have relieved. The
        barrier-side wrench is kept in `reactions_barrier` for diagnostics."""
        self.reactions, self.reactions_barrier = {}, {}
        gc = getattr(self, "_last_gc", None)
        if gc is None:
            return
        G3 = gc.reshape(-1, 3)
        xc = (self.P @ q + self.cvec).reshape(-1, 3)
        gb = None
        for b_i, b in enumerate(self.bodies):
            if b.kind != "driven":
                continue
            sl = self.body_vslice[b_i]
            f = -G3[sl]
            if not np.any(f):
                continue
            p, _ = self.body_pose(b_i, q)
            self.reactions_barrier[b_i] = (f.sum(0), np.cross(xc[sl] - p, f).sum(0))
            if gb is None:
                _, gb, _ = self._bodies_energy(q, want=False)
            o = self.body_dof[b_i]
            M = gb[o+3:o+12].reshape(3, 3)                  # sum_k f_k rbar_k^T of the contact forces (equilibrium)
            X = q[o+3:o+12].reshape(3, 3) @ M.T
            tau = np.array([X[1, 2] - X[2, 1], X[2, 0] - X[0, 2], X[0, 1] - X[1, 0]])
            self.reactions[b_i] = (gb[o:o+3].copy(), tau)

    def _post_step(self, dt):
        x = self.x_free()
        if len(self.hinge):
            self._plasticity(x, dt)
        if getattr(self, "suspend_tearing", False):
            pass          # start-state generation while the rest shape is still a blend (prepare_rest_morph): the strains
                          # of a fictitious rest shape must not damage or split the paper
        elif self._tear_sheets:
            self._tear_post(dt)
        elif len(self.tris) and any(self.mats[m].Gamma > 0 for m in np.unique(self.tri_mat)):
            self._damage(dt)
            if (self.damage >= 1.).sum() != (~self._contact_alive).sum():
                self._make_contact_meshes()
                self.stats["contact_rebuilds"] = self.stats.get("contact_rebuilds", 0) + 1
        for hb in self.hinges_b:
            self._body_hinge_flow(hb, dt)
        if len(self.tris):
            self.stats["crumple_updates"] = self.stats.get("crumple_updates", 0) + self._slack_post()
        for fn in getattr(self, "post_terms", []):
            fn(self, dt)

    def _plasticity(self, x, dt):
        theta = hinge_theta(x, self.hinge)
        d = wrap(theta - self.theta_p)
        # panels: rate-independent plastic hinge at the fully plastic moment
        pan = ~self.is_crease
        over = pan & (np.abs(d) > self.theta_yield)
        tp0 = self.theta_p[over].copy()
        self.theta_p[over] = wrap(theta[over] - np.sign(d[over])*self.theta_yield[over])
        # plastic work booked (diagnostic energy ledger): moment at the hinge limit 2 k theta_y times the plastic rotation
        self.plastic_dissipated = getattr(self, "plastic_dissipated", 0.) + float(
            (2.*self.k_hinge[over]*self.theta_yield[over]*np.abs(wrap(self.theta_p[over] - tp0))).sum())
        # creases: implicit return mapping of the two-mechanism law (CreaseLaw), all crease hinges at once (the same
        # per-hinge arithmetic as crease_return)
        hs = [h for h in np.flatnonzero(self.is_crease).tolist()
              if self.crease_law[h].omega0 > 0. or self.crease_law[h].omega2 > 0.]
        if hs:
            laws = [self.crease_law[h] for h in hs]
            par = [np.array([getattr(l, a) for l in laws], float) for a in ("k", "k2", "m_a", "omega0", "m2", "omega2")]
            hs = np.asarray(hs)
            s1, s2 = crease_return_batch(*par, d[hs], self.theta_r[hs], dt)
            self.theta_p[hs] = wrap(self.theta_p[hs] + s1 + s2)
            self.theta_r[hs] += s2

    def _body_hinge_flow(self, hb, dt):
        X = []
        for body, r in hb.pts:
            x, _ = self._point(body, np.asarray(r, float), self.q)
            X.append(x)
        th, _ = hinge_angles(np.asarray(X), np.array([[0, 1, 2, 3]]))
        dh = float(wrap(th - hb.theta_p)[0])
        s1, s2 = crease_return(hb.law, dh, hb.theta_r, dt)
        hb.theta_p = float(wrap(hb.theta_p + s1 + s2))
        hb.theta_r += s2

    def _damage(self, dt, eligible=None, in_plane=False):
        """Crack-band damage (Bazant & Oh 1983) on the tensile principal membrane strain: linear softening
        from the strength strain eps0 = X/Et to eps_f = 2 Gamma_I/(X h_b), so a fully damaged band of width
        h_b dissipates the in-plane fracture energy Gamma_I per unit crack length. Irreversible; updated after each
        converged step. With vertex splitting on (_tear_post) the band only acts in `eligible` triangles (no vertex on
        a free edge or crack face) with the in-plane Gamma_I: it nucleates interior holes in tension, edges and running
        cracks split instead. Without splitting (tear_pool=0) it runs everywhere with Gamma."""
        e = self._last_e
        exx, eyy, gxy = e[:, 0], e[:, 1], e[:, 2]
        c = .5*(exx + eyy)
        r = np.sqrt((.5*(exx - eyy))**2 + (.5*gxy)**2)
        e1 = c + r                                                # max principal Green strain (rest axes)
        ang = .5*np.arctan2(gxy, exx - eyy) - self.md_angle_tri  # principal direction vs MD
        cos2 = np.cos(ang)**2
        Xmd, Xcd, Emd, Ecd = (self._tri_prop(a) for a in ("X_md", "X_cd", "Et_md", "Et_cd"))
        Gam = self._tri_prop("gamma_in_plane" if in_plane else "Gamma")
        X = Xmd*cos2 + Xcd*(1 - cos2)
        Et = Emd*cos2 + Ecd*(1 - cos2)
        ok = (Gam > 0) & (X > 0)                                  # tearable triangles (tape and robot pads: no)
        if eligible is not None:
            ok &= eligible
        self.kappa_hist = np.maximum(self.kappa_hist, np.where(ok, e1, 0.))
        dnew = np.zeros(len(e1))
        eps0 = X[ok]/np.maximum(Et[ok], 1.)
        epsf = np.maximum(2*Gam[ok]/np.maximum(X[ok]*np.sqrt(2*self.area[ok]), 1e-12), 1.01*eps0)
        k = self.kappa_hist[ok]
        dnew[ok] = np.clip(np.where(k <= eps0, 0., np.where(k >= epsf, 1., 1. - (eps0/np.maximum(k, 1e-12))*(epsf - k)/(epsf - eps0))), 0., 1.)
        # viscous regularization: at most 0.25 per step (crack advance limited to ~1 band per 4 steps)
        self.damage = np.maximum(self.damage, np.minimum(dnew, self.damage + .25))
        self.damage[self.damage > .995] = 1.

    # ------------------------------------------------------------------ tearing by vertex splitting
    # A crack is a line of split vertices: the triangles on one side of the crack take a copy of each vertex on it.
    # Propagation from a crack tip (a vertex whose open fan ends at two copies of one vertex, the crack behind) uses
    # the virtual crack closure technique: for each candidate edge (v, w) ahead, the force P that would have to hold the
    # two sides of v together and the opening D of the copies one edge behind give G = P.D/(2 l) per mode (out-of-plane
    # = III, in-plane normal = I, in-plane along the crack = II), and the edge splits when G >= Gamma(psi, angle to MD)
    # with psi = G_III/G. Initiation (no crack behind: a free edge, a hole) is a strength test on the membrane force
    # across the candidate edge (N_n >= X(n)). The new pair is bonded by an extrinsic cohesive bond that starts at the
    # current tip force T0 and softens linearly to zero at dc = 2 Gamma l/T0, so a fully opened bond has dissipated
    # exactly Gamma l (booked in tear_dissipated) and the split itself releases no energy. Nothing moves a vertex: a copy
    # starts at the position and velocity of the vertex it duplicates, with the mass of its triangles.
    def _pool_size(self, s, n):
        p = self._tear_pool_arg
        if isinstance(p, dict):
            p = p.get(s.name)
        return max(32, n//6) if p is None else int(p)

    def _tear_setup(self):
        nf = self.n_free
        self.active = np.ones(nf, bool)
        for p0, pool in zip(self._pool_start, self.sheet_pool):
            self.active[p0:p0 + pool] = False
        self.vorig = np.arange(nf)
        self.sheet_used = [0]*len(self.sheets)
        self._tear_sheets = [k for k, p in enumerate(self.sheet_pool) if p > 0]
        self._tri_off = [int(np.searchsorted(self.sheet_of_tri, k)) for k in range(len(self.sheets))]
        Xr = np.zeros((nf, 3))
        for s, sl, n0 in zip(self.sheets, self.sheet_slices, self.sheet_n0):
            X = np.asarray(s.X2, float)[:n0]
            Xr[sl.start:sl.start + n0, :X.shape[1]] = X
        self._Xr = Xr                                          # rest positions (net coordinates), copies included
        self._sib_count, self._sib_groups = {}, []
        self.tear_bonds = {k: np.zeros(0, np.int64 if k in ("a", "b") else float) for k in
                           ("a", "b", "T0", "d0", "dc", "h", "diss", "gl", "ell", "psi", "t0")}
        self.tear_events = []
        self.tear_dissipated = 0.
        self.plastic_dissipated = 0.
        self._tips = set()
        self._inactive_diag = None
        self._topology_dirty = False
        if not self._tear_sheets:
            return
        m = self.tri_mat
        self._tri_tear = np.isin(self.sheet_of_tri, self._tear_sheets)
        get = lambda f: np.array([f(self.mats[i]) for i in m]) if len(m) else np.zeros(0)
        self._tri_rho = get(lambda a: a.rho_A)
        self._tri_G3, self._tri_G1, self._tri_cd = get(lambda a: a.Gamma), get(lambda a: a.gamma_in_plane), get(lambda a: a.tear_cd)
        self._tri_Xmd, self._tri_Xcd = get(lambda a: a.X_md), get(lambda a: a.X_cd)
        self._tri_Emd, self._tri_Ecd = get(lambda a: a.Et_md), get(lambda a: a.Et_cd)
        # screen threshold per triangle: where the element is longer than the stock's shortest l_c the averaged index
        # can exceed the element's own (_strength_index_lc), so the screen is lowered by l_c/h there
        lc_min = (2./math.pi)*np.minimum(self._tri_Ecd*self._tri_G1/np.maximum(self._tri_Xcd, 1e-30)**2,
                                         self._tri_Emd*self._tri_G1*self._tri_cd/np.maximum(self._tri_Xmd, 1e-30)**2)
        self._tri_screen = TEAR_SCREEN*np.minimum(1., lc_min/np.sqrt(2.*self.area)) if len(m) else np.zeros(0)
        # authored boundary edges (pairs of original vertex ids) of each tearable sheet: a crack face is a boundary edge
        # that was interior in the authored net
        self._orig_bnd = {}
        for k in self._tear_sheets:
            T = self.tris[self.sheet_of_tri == k]
            E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
            u, c = np.unique(E, axis=0, return_counts=True)
            self._orig_bnd[k] = set(map(tuple, u[c == 1].tolist()))
        self._auth_v = np.zeros(self.n_free, bool)                 # authored free-edge vertices (original ids)
        for k in self._tear_sheets:
            if self._orig_bnd[k]:
                self._auth_v[np.array(sorted(self._orig_bnd[k])).ravel()] = True
        self._edge_init = True
        self._tear_index()

    def _tear_index(self):
        """Vertex -> triangle incidence of tearable sheets, free-edge flags and crack tips, from the current topology."""
        self._v2t = {}
        for t in np.flatnonzero(self._tri_tear):
            for v in self.tris[t]:
                self._v2t.setdefault(int(v), set()).add(int(t))
        self._refresh_boundary()
        self._tips = {v for v in np.flatnonzero(self._vbnd) if self._is_tip(int(v))}

    def _refresh_boundary(self):
        live = (self.damage < 1.) & self._tri_tear
        T = self.tris[live]
        flag = np.zeros(self.n_free, bool)
        if len(T):
            E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
            u, c = np.unique(E, axis=0, return_counts=True)
            flag[u[c == 1].ravel()] = True
        self._vbnd = flag
        self._band_ok = ~flag[self.tris].any(1) if len(self.tris) else np.zeros(0, bool)
        self._bowtie_dirty = True

    def _update_inactive_diag(self):
        ina = np.flatnonzero(~self.active)
        if not len(ina):
            self._inactive_diag = None
            return
        d = np.zeros(self.n_dof)
        d[(3*ina[:, None] + np.arange(3)).ravel()] = 1.
        self._inactive_diag = sp.diags(d, format="csr")

    def _contact_filter(self, patch, siblings=False):
        """Patch filter; for the sheet contact set, the primitives incident to two sibling copies (the two sides of a
        fresh split, coincident at first) are excluded from each other until they have separated (_release_siblings).
        ipctk keeps a primitive pair if any of its cross vertex pairs may collide, so every pair between the two copies'
        closed stars is listed."""
        ipctk = self.ipctk
        f = ipctk.make_vertex_patches_filter(np.asarray(patch, np.int32))
        if siblings and getattr(self, "_sib_count", None):
            cm = self._cmap
            keys = {}
            for i, j in self._sib_count:
                a, b = int(cm[i]), int(cm[j])
                keys[(min(a, b), max(a, b))] = False
            f = f & ipctk.make_sparse_filter(keys, True)
        return f

    def _sheet_of_v(self, v):
        starts = [sl.start for sl in self.sheet_slices]
        return int(np.searchsorted(starts, v, side="right") - 1)

    def _fan(self, v):
        """Live triangles around v as ordered open chains [(t, p, q), ...] with t = (v, p, q) counter-clockwise:
        [] for an interior vertex (closed fan), None for a non-manifold or inconsistently oriented star."""
        nxt = {}
        for t in self._v2t.get(v, ()):
            if self.damage[t] >= 1.:
                continue
            tri = self.tris[t]
            k = int(np.flatnonzero(tri == v)[0])
            p, q = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            if p in nxt:
                return None
            nxt[p] = (t, q)
        if not nxt:
            return None
        qs = {q for _, q in nxt.values()}
        chains, used = [], 0
        for p0 in sorted(p for p in nxt if p not in qs):
            ch, p = [], p0
            while p in nxt and len(ch) <= len(nxt):
                t, q = nxt[p]
                ch.append((t, p, q))
                p = q
            chains.append(ch)
            used += len(ch)
        if chains and used != len(nxt):
            return None
        return chains

    def _is_tip(self, v):
        ch = self._fan(v)
        if not ch or len(ch) != 1 or len(ch[0]) < 2:
            return False
        a, b = ch[0][0][1], ch[0][-1][2]
        return a != b and self.vorig[a] == self.vorig[b]

    def _shell_parts(self, x):
        """Membrane strain/force and the per-element elastic gradients at x: (e, S, g_tri (T,3,3), g_hinge (H,4,3))."""
        T, b = self.tris, self.bmat
        F = np.matmul(x[T].transpose(0, 2, 1), b)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1) - 1.), .5*((f2*f2).sum(1) - 1.), (f1*f2).sum(1)], 1)
        live = 1. - self.damage
        S = np.einsum("tij,tj->ti", self.C, e)*live[:, None]
        sk = self._slack_idx
        if sk is not None:
            rk, Sk, _, _ = slack_response(self.C[sk], e[sk], self.slack_c[sk], self.slack_phi[sk], tangent=False)
            e = e.copy()
            e[sk] = rk
            S[sk] = Sk*live[sk, None]
        J = np.zeros((len(T), 3, 3, 3))
        J[:, 0] = b[:, :, 0, None]*f1[:, None, :]
        J[:, 1] = b[:, :, 1, None]*f2[:, None, :]
        J[:, 2] = b[:, :, 0, None]*f2[:, None, :] + b[:, :, 1, None]*f1[:, None, :]
        gt = (self.area[:, None]*np.einsum("ti,tij->tj", S, J.reshape(len(T), 3, 9))).reshape(-1, 3, 3)
        H = self.hinge
        gh = np.zeros((len(H), 4, 3))
        if len(H):
            theta, dth = hinge_angles(x, H)
            alive = np.minimum(live[self.hinge_tris[:, 0]], live[self.hinge_tris[:, 1]])
            gh = (2*self.k_hinge*alive*wrap(theta - self.theta_p))[:, None, None]*dth
        return e, S, gt, gh

    def _tape_grad_at(self, v, x):
        """Gradient of the tape bonds' energy at vertex v, per sheet triangle the bond is anchored to."""
        out = []
        for bd in self.bonds:
            T = self.tris[bd.tri]
            rows, slots = np.nonzero(T == v)
            if not len(rows):
                continue
            dl = bd._delta(x)[rows]
            k = (1. - bd.D[rows])*bd.law.k*bd.area[rows]*(bd.D[rows] < 1.)
            g = -bd.bary[rows, slots][:, None]*(k[:, None]*dl)
            out += [(int(bd.tri[r]), g[i]) for i, r in enumerate(rows)]
        return out

    def _side_forces(self, v, chain, parts, x):
        """For each interior fan edge k of the open chain at v (between positions k-1 and k), the pull P_k on side
        A = chain[:k] away from side B = chain[k:]: the force a bond between the two copies would have to carry if v
        were split there, with every external load on v (inertia, gravity, contact, fixtures) shared by mass."""
        _, _, gt, gh = parts
        n = len(chain)
        tri_ids = np.array([c[0] for c in chain])
        pos = {int(t): i for i, t in enumerate(tri_ids)}
        slot = np.argmax(self.tris[tri_ids] == v, axis=1)
        gpos = gt[tri_ids, slot].copy()
        mt = self.area[tri_ids]*self._tri_rho[tri_ids]/3.
        strad = {}
        for h in np.flatnonzero((self.hinge == v).any(1)):
            if self.k_hinge[h] == 0.:
                continue
            sl_ = int(np.flatnonzero(self.hinge[h] == v)[0])
            g = gh[h, sl_]
            t0, t1 = int(self.hinge_tris[h, 0]), int(self.hinge_tris[h, 1])
            if sl_ >= 2:
                i = pos.get(t0 if sl_ == 2 else t1)
                if i is not None:
                    gpos[i] += g
                continue
            i0, i1 = pos.get(t0), pos.get(t1)
            if i0 is None and i1 is None:
                continue
            if i0 is None or i1 is None:
                gpos[i0 if i0 is not None else i1] += g
            elif abs(i0 - i1) == 1:
                strad[min(i0, i1)] = strad.get(min(i0, i1), 0.) + g
            else:
                gpos[i0] += .5*g; gpos[i1] += .5*g
        for t, g in self._tape_grad_at(v, x):
            i = pos.get(t)
            if i is not None:
                gpos[i] += g
        cg, cm = np.cumsum(gpos, 0), np.cumsum(mt)
        gall = cg[-1] + sum(strad.values(), np.zeros(3))
        mtot = cm[-1]
        out = []
        for k in range(1, n):
            gA = cg[k - 1] + sum((g for i, g in strad.items() if i <= k - 2), np.zeros(3)) + .5*strad.get(k - 1, np.zeros(3))
            gB = gall - gA
            mA, mB = cm[k - 1], mtot - cm[k - 1]
            out.append(((mA*gB - mB*gA)/max(mtot, 1e-30), mA, mB))
        return out

    def _edge_frame(self, v, w, t1, t2, x):
        tv = x[w] - x[v]
        tv /= max(np.linalg.norm(tv), 1e-30)
        nn = np.zeros(3)
        for t in (t1, t2):
            a, b_, c = x[self.tris[t]]
            nt = np.cross(b_ - a, c - a)
            nn += nt/max(np.linalg.norm(nt), 1e-30)
        nn /= max(np.linalg.norm(nn), 1e-30)
        return tv, nn, np.cross(nn, tv)

    def _mean_normal(self, t1, t2, x):
        """Mean of the unit normals of triangles t1 and t2 (length 1 on a flat sheet, 0 across a 180 degree fold)."""
        nn = np.zeros(3)
        for t in (t1, t2):
            a, b_, c = x[self.tris[t]]
            nt = np.cross(b_ - a, c - a)
            nn += .5*nt/max(np.linalg.norm(nt), 1e-30)
        return nn

    def _rest_edge(self, t, v, w):
        """Edge v->w in triangle t's rest frame (MD = +x)."""
        tri = self.tris[t]
        kv, kw = int(np.flatnonzero(tri == v)[0]), int(np.flatnonzero(tri == w)[0])
        return self.Lt[t, kw] - self.Lt[t, kv]

    def _gamma(self, t1, t2, v, w, psi):
        """Tear energy per crack length for a crack along edge (v, w) between triangles t1, t2 with mode mix psi
        (0 in-plane, 1 out-of-plane): linear in psi between Gamma_I and Gamma_III, scaled by the edge's angle to MD."""
        e = self._rest_edge(t1, v, w)
        c2 = e[0]**2/max(e @ e, 1e-30)
        g = 0.
        for t in (t1, t2):
            g += .5*((1. - psi)*self._tri_G1[t] + psi*self._tri_G3[t])*(c2 + self._tri_cd[t]*(1. - c2))
        return g

    def _edge_strength_index(self, t, v, w, S):
        """Membrane force normal to edge (v, w) in triangle t over the tensile strength in that direction."""
        e = self._rest_edge(t, v, w)
        nrm = np.array([-e[1], e[0]])/max(np.linalg.norm(e), 1e-30)
        s0, s1, s2 = S[t]
        Nn = s0*nrm[0]**2 + s1*nrm[1]**2 + 2*s2*nrm[0]*nrm[1]
        X = self._tri_Xmd[t]*nrm[0]**2 + self._tri_Xcd[t]*nrm[1]**2
        return Nn/X if X > 0 else 0.

    def _edge_lc(self, t, v, w):
        """Averaging length of the strength test for a crack along edge (v, w) of triangle t: the Whitney-Nuismer
        average-stress length l_c = (2/pi) Et(n) Gamma_I(crack direction)/X(n)^2 [D from the stock's ledger], n the
        crack normal. With it the averaged test N_bar >= X equals G >= Gamma_I at a long crack's tip and N >= X in
        uniform tension."""
        e = self._rest_edge(t, v, w)
        c2 = e[0]**2/max(e @ e, 1e-30)                    # crack direction: cos^2 to MD; its normal: sin^2
        Et = self._tri_Emd[t]*(1. - c2) + self._tri_Ecd[t]*c2
        X = self._tri_Xmd[t]*(1. - c2) + self._tri_Xcd[t]*c2
        return (2./math.pi)*Et*self._tri_G1[t]*(c2 + self._tri_cd[t]*(1. - c2))/max(X*X, 1e-30)

    def _edge_tris(self, a, b):
        return [t for t in self._v2t.get(a, ()) if self.damage[t] < 1. and b in self.tris[t]]

    def _strength_index_lc(self, v, w, t1, t2, S):
        """Membrane strength index of a crack starting at v along edge (v, w): the normal membrane force resultant over
        the length l_c of the prospective crack line (edge (v, w) and its straightest continuations), over X l_c.
        Elements shorter than l_c are averaged (a path that ends before l_c is averaged over its own length: net
        section). An element longer than l_c cannot show how its force is distributed: the part above the next element's
        level is taken as concentrated within l_c of v, the rest as uniform (uniform tension then still fails at X)."""
        f1 = .5*(self._edge_strength_index(t1, v, w, S) + self._edge_strength_index(t2, v, w, S))
        if not TEAR_EDGE_LC:
            return f1
        lc = .5*(self._edge_lc(t1, v, w) + self._edge_lc(t2, v, w))
        segs = [(f1, float(np.linalg.norm(self._Xr[w] - self._Xr[v])))]
        L, a, b = segs[0][1], v, w
        while (L < lc or len(segs) < 2) and len(segs) < 16:
            d = self._Xr[b] - self._Xr[a]
            d /= max(np.linalg.norm(d), 1e-30)
            best = None
            for c in {int(i) for t in self._v2t.get(b, ()) if self.damage[t] < 1. for i in self.tris[t]} - {a, b}:
                e = self._Xr[c] - self._Xr[b]
                le = float(np.linalg.norm(e))
                cs = float(e @ d)/max(le, 1e-30)
                if cs > .8 and (best is None or cs > best[0]):
                    best = (cs, c, le)
            if best is None:
                break
            _, c, le = best
            ts = self._edge_tris(b, c)
            if len(ts) != 2:
                break
            segs.append((.5*(self._edge_strength_index(ts[0], b, c, S) + self._edge_strength_index(ts[1], b, c, S)), le))
            L += le
            a, b = b, c
        h1 = segs[0][1]
        if h1 >= lc:
            f2 = segs[1][0] if len(segs) > 1 else f1
            return f2 + (f1 - f2)*h1/lc if f1 > f2 else f1
        acc, rem = 0., lc
        for f, h in segs:
            use = min(h, rem)
            acc += f*use
            rem -= use
            if rem <= 0.:
                break
        return acc/(lc - rem)

    def _tear_post(self, dt):
        """After a converged step: cohesive bond history, splits (crack tips, then initiation at free edges and holes,
        then bowties), the in-plane crack band in interior triangles, contact bookkeeping between sibling copies."""
        x = self.x_free()
        parts = self._shell_parts(x)
        self._last_e = parts[0]
        self._tear_bonds_after(x)
        dead0 = int((self.damage >= 1.).sum())
        self._tear_splits(parts, x)
        if self._topology_dirty:
            self._after_topology()
        self._damage(dt, eligible=self._band_ok, in_plane=True)
        if int((self.damage >= 1.).sum()) != dead0:
            self._refresh_boundary()
            self._make_contact_meshes()
            self.stats["contact_rebuilds"] = self.stats.get("contact_rebuilds", 0) + 1
        self._release_siblings()

    def _tear_splits(self, parts, x):
        _, S, _, _ = parts
        decisions, touched = [], set()
        cone = math.cos(math.radians(TEAR_CONE_DEG))
        # 1. crack tips: VCCT
        for v in sorted(self._tips):
            ch = self._fan(v)
            if not ch or len(ch) != 1 or len(ch[0]) < 2:
                self._tips.discard(v)
                continue
            chain = ch[0]
            a, b = chain[0][1], chain[-1][2]
            if a == b or self.vorig[a] != self.vorig[b]:
                self._tips.discard(v)
                continue
            D = x[a] - x[b]
            dcr = self._Xr[v] - .5*(self._Xr[a] + self._Xr[b])
            dcr /= max(np.linalg.norm(dcr), 1e-30)
            lb = .5*(np.linalg.norm(self._Xr[v] - self._Xr[a]) + np.linalg.norm(self._Xr[v] - self._Xr[b]))
            best = None
            for k, (P, mA, mB) in enumerate(self._side_forces(v, chain, parts, x), start=1):
                w = chain[k - 1][2]
                de = self._Xr[w] - self._Xr[v]
                la = float(np.linalg.norm(de))
                if de @ dcr < cone*la:
                    continue
                t1, t2 = chain[k - 1][0], chain[k][0]
                tv, nn, mm = self._edge_frame(v, w, t1, t2, x)
                ell = .5*(la + lb)
                G = np.array([max((P @ mm)*(D @ mm), 0.), max((P @ tv)*(D @ tv), 0.), max((P @ nn)*(D @ nn), 0.)])/(2*ell)
                Gs = G.sum()
                # strength at the tip as well (extrinsic cohesive insertion): a crack shorter than the material's cohesive
                # length (E Gamma_I/X^2, ~13 mm for wrap) does not reach G = Gamma at the strength, yet its ligament
                # fails there; the bond still charges Gamma l. Either criterion splits.
                f_in = self._strength_index_lc(v, w, t1, t2, S)
                psi = G[2]/Gs if Gs > 0. else float((P @ nn)**2/max(P @ P, 1e-30))
                gam = self._gamma(t1, t2, v, w, psi)
                if gam <= 0.:
                    continue
                r = max(Gs/gam, f_in)
                if best is None or r > best[0]:
                    best = (r, k, P, G, psi, gam, ell, w, f_in)
            if best is not None and best[0] >= 1.:
                r, k, P, G, psi, gam, ell, w, f_in = best
                decisions.append(dict(kind="vcct" if G.sum()/gam >= f_in else "tip-strength", v=v, chain=chain, k=k, P=P,
                                      G=G, psi=psi, gamma=gam, ell=ell, tip=w, ratio=r, f_in=f_in))
        # 2. initiation where no crack is behind (a free edge, a hole rim, a crack face). In plane: the normal membrane
        # force resultant over the stock's length l_c of the prospective crack line against X l_c (_strength_index_lc).
        # At an authored free edge also out of plane:
        # the out-of-plane part of the pull P between the two sides against TEAR_EDGE_T3 Gamma_III, the traction at
        # which a running tear's bond is inserted on the trouser bench; mixed quadratically:
        # f = sqrt((N_bar(l_c)/X)^2 + (P_III/(c_III Gamma_III))^2) >= 1.
        e = parts[0]
        c = .5*(S[:, 0] + S[:, 1])
        rr = np.sqrt((.5*(S[:, 0] - S[:, 1]))**2 + S[:, 2]**2)
        th = .5*np.arctan2(2*S[:, 2], S[:, 0] - S[:, 1])
        Xp = self._tri_Xmd*np.cos(th)**2 + self._tri_Xcd*np.sin(th)**2
        idx = np.where(Xp > 0, (c + rr)/np.maximum(Xp, 1e-30), 0.)
        hot = np.flatnonzero((idx >= (self._tri_screen if TEAR_EDGE_LC else TEAR_SCREEN)) & self._tri_tear & (self.damage < 1.))
        cand = {int(v) for v in self.tris[hot].ravel() if self._vbnd[v]}
        if self._edge_init:
            # screen: per-vertex sum of the out-of-plane components of the elastic forces of its elements
            _, _, gt, gh = parts
            T = self.tris
            nt = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
            nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-30)
            oop = np.zeros(self.n_free)
            np.add.at(oop, T.ravel(), np.abs((gt*nt[:, None, :]).sum(2)).ravel()*np.repeat(self._tri_tear, 3))
            if len(gh):
                nh = nt[self.hinge_tris[:, 0]]
                np.add.at(oop, self.hinge.ravel(), np.abs((gh*nh[:, None, :]).sum(2)).ravel())
            g3 = np.full(self.n_free, np.inf)
            np.minimum.at(g3, T.ravel(), np.repeat(np.where(self._tri_tear, self._tri_G3, np.inf), 3))
            auth = self._vbnd & self._auth_v[self.vorig] & (oop >= TEAR_SCREEN*TEAR_EDGE_T3*g3)
            cand |= {int(v) for v in np.flatnonzero(auth)}
        inits = []
        for v in sorted(cand - self._tips):
            ch = self._fan(v)
            if not ch or len(ch) != 1 or len(ch[0]) < 2:
                continue
            chain = ch[0]
            a, b = chain[0][1], chain[-1][2]
            if a != b and self.vorig[a] == self.vorig[b]:
                continue
            authored = self._edge_init and not (self._crack_face(v, a) or self._crack_face(v, b))
            sides = self._side_forces(v, chain, parts, x) if authored else None
            best = None
            for k in range(1, len(chain)):
                w = chain[k - 1][2]
                t1, t2 = chain[k - 1][0], chain[k][0]
                f_in = self._strength_index_lc(v, w, t1, t2, S)
                f_out = 0.
                if authored:
                    # the out-of-plane (trouser) part of the pull: along the MEAN of the two sides' normals, not
                    # normalised. On a flat sheet that is the sheet's normal. Across a fold it shrinks with the fold
                    # angle and vanishes at 180 degrees: pulling the two leaves of a folded sheet apart opens the fold
                    # (in the unfolded sheet both flanks move to the same side), it does not tear along it.
                    g3k = TEAR_EDGE_T3*self._gamma(t1, t2, v, w, 1.)
                    f_out = abs(float(sides[k - 1][0] @ self._mean_normal(t1, t2, x)))/g3k if g3k > 0 else 0.
                f = math.hypot(max(f_in, 0.), f_out)
                if best is None or f > best[0]:
                    best = (f, k, w, t1, t2, f_in, f_out)
            if best is None or best[0] < 1.:
                continue
            f, k, w, t1, t2, f_in, f_out = best
            P = (sides if sides is not None else self._side_forces(v, chain, parts, x))[k - 1][0]
            tv, nn, mm = self._edge_frame(v, w, t1, t2, x)
            psi = float((P @ nn)**2/max(P @ P, 1e-30))
            ell = .5*float(np.linalg.norm(self._Xr[w] - self._Xr[v]))
            inits.append(dict(kind="init", v=v, chain=chain, k=k, P=P, G=None, psi=psi,
                              gamma=self._gamma(t1, t2, v, w, psi), ell=ell, tip=w, index=f, ratio=f,
                              f_in=f_in, f_out=f_out))
        # one initiation per sheet per step, at the most overstressed site (the weakest link): a uniformly loaded edge
        # otherwise starts a crack at every vertex in the same step; running tips (1.) are not limited
        best_init = {}
        for d in inits:
            k_ = self._sheet_of_v(d["v"])
            if k_ not in best_init or d["index"] > best_init[k_]["index"]:
                best_init[k_] = d
        decisions += [best_init[k_] for k_ in sorted(best_init)]
        # 3. bowties: a vertex joining separate fans (a crack ran out at a free edge, or two holes touch at a corner);
        # they can only appear where the topology or the set of torn triangles changed
        bow = sorted(int(u) for u in np.flatnonzero(self._vbnd)) if self._bowtie_dirty else []
        self._bowtie_dirty = False
        for v in bow:
            ch = self._fan(v)
            if ch and len(ch) > 1:
                decisions.append(dict(kind="bowtie", v=v, chains=ch))
        for d in decisions:
            v = d["v"]
            if v in touched:
                continue
            if d["kind"] == "bowtie":
                ch = d["chains"]
                main = max(range(len(ch)), key=lambda i: len(ch[i]))
                for i, chain in enumerate(ch):
                    if i == main or touched & {v}:
                        continue
                    copy = [c[0] for c in chain]
                    # fans split off earlier in this loop already reference their own copy, not v
                    rest = [c[0] for j, cc in enumerate(ch) if j != i for c in cc if v in self.tris[c[0]]]
                    P = self._pair_force(v, copy, rest, parts, x)
                    # the last ligament: half of the crack edge that reached v (the other half is the previous bond's)
                    near = [w for w in (chain[0][1], chain[-1][2]) if self._crack_face(v, w)
                            and not any(self.damage[t] >= 1. and w in self.tris[t] for t in self._v2t.get(v, ()))]
                    ell = .5*min([float(np.linalg.norm(self._Xr[w] - self._Xr[v])) for w in near] or [0.])
                    t1 = copy[0]
                    psi = .5
                    gam = .5*(self._tri_G1[t1] + self._tri_G3[t1])
                    bond = None
                    if ell > 0. and gam > 0.:
                        T0 = max(float(np.linalg.norm(P)), gam)
                        bond = dict(T0=T0, gl=gam*ell, ell=ell, psi=psi)
                    s = self._split(v, copy, bond, dict(kind="bowtie", P=P.tolist(), ell=ell))
                    if s is None:
                        break
                touched.add(v)
                continue
            chain, k = d["chain"], d["k"]
            A = [c[0] for c in chain[:k]]
            B = [c[0] for c in chain[k:]]
            star = {int(i) for t in self._v2t[v] for i in self.tris[t]}
            if star & touched:
                continue
            P = d["P"]
            # the criterion is checked after each step, so a stiff load path can carry the tip past it within one step
            # (ratio r > 1). The bond starts at the traction the tip had when it crossed the criterion, T0 = |P|/r (at
            # the step's opening G is linear in P), i.e. its full-separation opening dc = 2 Gamma l/T0 equals the
            # current opening one edge behind; the rest of the step's overshoot is released at the split
            r = max(float(d.get("ratio", 1.)), 1.)
            T0 = float(np.linalg.norm(P))/r
            gl = d["gamma"]*d["ell"]
            bond = dict(T0=T0, gl=gl, ell=d["ell"], psi=d["psi"]) if T0 > 0. and gl > 0. else None
            # the smaller side takes the copy, so held vertex ids (grips, fixtures) stay with the larger piece
            aA, aB = self.area[A].sum(), self.area[B].sum()
            copy = A if aA <= aB else B
            ev = dict(kind=d["kind"], tip=int(d["tip"]), P=[float(p) for p in P], psi=float(d["psi"]),
                      gamma=float(d["gamma"]), ell=float(d["ell"]))
            if d["G"] is not None:
                ev["G"] = [float(g) for g in d["G"]]
            if "index" in d:
                ev["strength_index"] = float(d["index"])
                ev["in_plane_index"], ev["out_of_plane_index"] = float(d["f_in"]), float(d["f_out"])
            elif "f_in" in d:
                ev["in_plane_index"] = float(d["f_in"])
            ev["ratio"] = r
            s = self._split(v, copy, bond, ev)
            if s is None:
                continue
            touched |= star | {s}
            self._tips.discard(v)
            self._tips.add(int(d["tip"]))

    def _crack_face(self, v, w):
        """Edge (v, w) is a crack face: interior in the authored net."""
        k = self._sheet_of_v(v)
        a, b = int(self.vorig[v]), int(self.vorig[w])
        return (min(a, b), max(a, b)) not in self._orig_bnd.get(k, ())

    def _pair_force(self, v, copy, rest, parts, x):
        """Pull on the triangles `copy` away from `rest` at v (as _side_forces, for an arbitrary partition)."""
        _, _, gt, gh = parts
        cs, rs = set(copy), set(rest)
        g = {0: np.zeros(3), 1: np.zeros(3)}
        m = {0: 0., 1: 0.}
        for t in list(cs) + list(rs):
            side = 0 if t in cs else 1
            g[side] += gt[t, int(np.flatnonzero(self.tris[t] == v)[0])]
            m[side] += self.area[t]*self._tri_rho[t]/3.
        for h in np.flatnonzero((self.hinge == v).any(1)):
            sl_ = int(np.flatnonzero(self.hinge[h] == v)[0])
            t0, t1 = int(self.hinge_tris[h, 0]), int(self.hinge_tris[h, 1])
            tt = [t0 if sl_ in (0, 1, 2) else t1]
            if sl_ < 2:
                tt = [t0, t1]
            for t in tt:
                if t in cs:
                    g[0] += gh[h, sl_]/len(tt)
                elif t in rs:
                    g[1] += gh[h, sl_]/len(tt)
        mt = max(m[0] + m[1], 1e-30)
        return (m[0]*g[1] - m[1]*g[0])/mt

    def _split(self, v, copy_tris, bond=None, event=None):
        """Duplicate vertex v: the triangles `copy_tris` take the next spare of v's sheet, starting at v's position and
        velocity with the mass of those triangles; the hinge across the separating edge(s) is removed; with `bond`
        (T0, gl = Gamma l, ell, psi) the pair gets an extrinsic cohesive bond. Returns the copy's id (None: pool empty)."""
        k = self._sheet_of_v(v)
        sl, n0 = self.sheet_slices[k], self.sheet_n0[k]
        used = self.sheet_used[k]
        if used >= self.sheet_pool[k]:
            self.stats["tear_pool_exhausted"] = self.stats.get("tear_pool_exhausted", 0) + 1
            return None
        s = self._pool_start[k] + used
        assert s == sl.stop
        self.sheet_used[k] = used + 1
        self.sheet_slices[k] = slice(sl.start, s + 1)
        sh = self.sheets[k]
        toff = self._tri_off[k]
        Tl = np.array(sh.tris)
        cs = set(int(t) for t in copy_tris)
        for t in cs:
            row = self.tris[t]
            row[row == v] = s
            rl = Tl[t - toff]
            rl[rl == v - sl.start] = s - sl.start
            self._v2t[v].discard(t)
            self._v2t.setdefault(s, set()).add(t)
        sh.tris = Tl
        vl = v - sl.start
        X2 = np.asarray(sh.X2)
        sh.X2 = np.vstack([X2[:n0 + used], X2[vl:vl + 1]])
        x0 = np.asarray(sh.x0, float)
        sh.x0 = np.vstack([x0[:n0 + used], x0[vl:vl + 1]])
        self._Xr[s] = self._Xr[v]
        for h in np.flatnonzero((self.hinge == v).any(1)):
            row = self.hinge[h]
            in0, in1 = int(self.hinge_tris[h, 0]) in cs, int(self.hinge_tris[h, 1]) in cs
            if row[0] == v or row[1] == v:
                if in0 and in1:
                    row[row == v] = s
                elif in0 != in1:
                    self.k_hinge[h] = self.k_hinge0[h] = 0.          # the hinge across the new crack face
                    self.is_crease[h] = False
                    self.crease_law[h] = None
            else:
                if row[2] == v and in0:
                    row[2] = s
                if row[3] == v and in1:
                    row[3] = s
        mA = float((self.area[list(cs)]*self._tri_rho[list(cs)]).sum()/3.)
        self.vmass[s], self.vmass[v] = mA, self.vmass[v] - mA
        self.vthick[s] = self.vthick[v]
        self.x_free0[s] = self.x_free0[v]
        f = self._cfull
        for key in ("Vc0", "vhalf", "vmu", "vclass"):
            f[key][s] = f[key][v]
        self.q[3*s:3*s + 3] = self.q[3*v:3*v + 3]
        self.v[3*s:3*s + 3] = self.v[3*v:3*v + 3]
        self.active[s] = True
        self.vorig[s] = self.vorig[v]
        # contact exclusion: every vertex pair between the closed stars of the two copies (group sides B and A); a vertex
        # of an earlier, still excluded group that is split now brings its copy into that group's side
        for grp in self._sib_groups:
            for side, other in (("A", "B"), ("B", "A")):
                if v in grp[side] and s not in grp[side]:
                    grp[side].add(s)
                    self._sib_add({s}, grp[other])
        sv = {int(v)} | {int(i) for t in self._v2t[v] for i in self.tris[t]}
        ss = {int(s)} | {int(i) for t in self._v2t[s] for i in self.tris[t]}
        self._sib_add(ss, sv)
        bid = None
        if bond is not None:
            bid = self._add_tear_bond(v, s, **bond)
        self._sib_groups.append(dict(v=int(v), s=int(s), A=ss, B=sv, bond=bid))
        ev = dict(event or {}, t=float(self.time), step=int(self.steps), sheet=k, v=int(v), copy=int(s),
                  rest=self._Xr[v].tolist(), bond=bid, converged=bool(getattr(self, "_step_converged", True)))
        if bid is not None:
            tb = self.tear_bonds
            ev.update(T0=float(tb["T0"][bid]), dc=float(tb["dc"][bid]), gl=float(tb["gl"][bid]))
        self.tear_events.append(ev)
        self._topology_dirty = True
        return s

    def _sib_add(self, A, B, sign=1):
        for i in A:
            for j in B:
                if i != j:
                    p = (min(i, j), max(i, j))
                    c = self._sib_count.get(p, 0) + sign
                    if c > 0:
                        self._sib_count[p] = c
                    else:
                        self._sib_count.pop(p, None)

    def _add_tear_bond(self, a, b, T0, gl, ell, psi):
        dc = 2.*gl/T0
        d0 = min(TEAR_D0, .01*dc)
        tb = self.tear_bonds
        row = dict(a=a, b=b, T0=T0, d0=d0, dc=dc, h=d0, diss=0., gl=gl, ell=ell, psi=psi, t0=self.time)
        for k_, v_ in row.items():
            tb[k_] = np.append(tb[k_], v_)
        return len(tb["a"]) - 1

    @staticmethod
    def _bond_env(r, T0, d0, dc):
        """Energy of the bilinear law along its envelope: elastic to T0 at d0, linear softening to zero at dc."""
        el = .5*T0/d0*r*r
        so = .5*T0*d0 + T0*(dc*(r - d0) - .5*(r*r - d0*d0))/np.maximum(dc - d0, 1e-30)
        return np.where(r <= d0, el, np.where(r < dc, so, .5*T0*dc))

    def _tear_bond_energy(self, x, want):
        """Cohesive bonds between sibling copies: secant unloading below the largest opening reached (h, fixed within
        a step), the softening envelope beyond it. Newton matrix: the isotropic secant (force/opening) block."""
        g = np.zeros_like(x)
        coo = ([], [], [])
        tb = getattr(self, "tear_bonds", None)
        if tb is None or not len(tb["a"]):
            return 0., g, coo
        act = tb["h"] < tb["dc"]
        if not act.any():
            return 0., g, coo
        a, b = tb["a"][act], tb["b"][act]
        T0, d0, dc, h = tb["T0"][act], tb["d0"][act], tb["dc"][act], tb["h"][act]
        d = x[a] - x[b]
        r = np.linalg.norm(d, axis=1)
        Th = T0*np.clip((dc - h)/np.maximum(dc - d0, 1e-30), 0., 1.)
        ks = Th/h
        inside = r <= h
        E = np.where(inside, self._bond_env(h, T0, d0, dc) - .5*Th*h + .5*ks*r*r, self._bond_env(r, T0, d0, dc))
        fm = np.where(inside, ks*r, np.where(r < dc, T0*(dc - r)/np.maximum(dc - d0, 1e-30), 0.))
        s = np.where(r > 1e-15, fm/np.maximum(r, 1e-15), ks)
        f = s[:, None]*d
        np.add.at(g, a, f)
        np.add.at(g, b, -f)
        if want:
            for c in range(3):
                for i, j, sg in ((a, a, 1.), (b, b, 1.), (a, b, -1.), (b, a, -1.)):
                    coo[0].append(3*i + c); coo[1].append(3*j + c); coo[2].append(sg*s)
        return float(E.sum()), g, coo

    def _tear_bonds_after(self, x):
        tb = self.tear_bonds
        if not len(tb["a"]):
            return
        r = np.linalg.norm(x[tb["a"]] - x[tb["b"]], axis=1)
        h = np.maximum(tb["h"], r)
        Th = tb["T0"]*np.clip((tb["dc"] - h)/np.maximum(tb["dc"] - tb["d0"], 1e-30), 0., 1.)
        diss = self._bond_env(h, tb["T0"], tb["d0"], tb["dc"]) - .5*Th*h
        self.tear_dissipated += float((diss - tb["diss"]).sum())
        tb["h"], tb["diss"] = h, diss

    def _after_topology(self):
        self._mass_matrix()
        self._update_inactive_diag()
        self._compact_collision()
        self.edges_sheet = edges_of(self.tris)
        self._refresh_boundary()
        self._make_contact_meshes()
        self.stats["contact_rebuilds"] = self.stats.get("contact_rebuilds", 0) + 1
        self.stats["splits"] = len(self.tear_events)
        self._topology_dirty = False

    def _release_siblings(self):
        """Re-enable contact between sibling copies once their bond (if any) is fully open and enabling the excluded
        pairs adds no active contact pair (every newly allowed pair is beyond the barrier's activation distance)."""
        if not self._sib_groups or self.steps % RELEASE_EVERY:
            return
        tb = self.tear_bonds
        ready = [g for g in self._sib_groups if g["bond"] is None or tb["h"][g["bond"]] >= tb["dc"][g["bond"]]]
        if not ready:
            return
        cs = self.csets[0]
        xc = np.asfortranarray(self.x_coll())
        n_old = len(cs.collisions(xc))
        released = 0
        for grp in ready[:8]:
            self._sib_add(grp["A"], grp["B"], -1)
            self.mesh.can_collide = self._contact_filter(self.patch, siblings=True)
            if len(cs.collisions(xc)) == n_old:
                self._sib_groups.remove(grp)
                released += 1
            else:
                self._sib_add(grp["A"], grp["B"])
                self.mesh.can_collide = self._contact_filter(self.patch, siblings=True)
        if released:
            self.stats["sibling_releases"] = self.stats.get("sibling_releases", 0) + released

    def cut(self, k, path):
        """Pre-cut sheet k along a path of its local vertex ids (p0 on a free edge, consecutive vertices joined by mesh
        edges): every vertex but the last is split, without cohesion (a knife cut made before the scene starts). The
        copies start coincident and are excluded from contact with each other until they separate."""
        sl = self.sheet_slices[k]
        path = [int(p) + sl.start for p in path]
        for p, w in zip(path[:-1], path[1:]):
            ch = self._fan(p)
            if not ch or len(ch) != 1:
                raise ValueError(f"cut: vertex {p - sl.start} has no single open fan")
            chain = ch[0]
            ks = [i for i in range(1, len(chain)) if chain[i - 1][2] == w]
            if not ks:
                raise ValueError(f"cut: no interior edge {p - sl.start}-{w - sl.start}")
            A = [c[0] for c in chain[:ks[0]]]
            if self._split(p, A, None, dict(kind="cut", tip=int(w))) is None:
                raise ValueError("cut: spare pool exhausted")
        self._after_topology()
        self._tips = {v for v in np.flatnonzero(self._vbnd) if self._is_tip(int(v))}

    def crack_length(self, k):
        """Torn length of sheet k (m, in the rest net): (split, band).
        Split: the rest length of every crack edge made by vertex splitting, each weighted by the fraction of its
        cohesive bond's fracture energy that has been dissipated (0: the bond never opened beyond its elastic range,
        the paper there still carries its full traction and is not torn; 1: fully separated). A cut (no bond) counts
        in full.
        Band: sqrt(A/2) per fully damaged triangle (its area over the band width sqrt(2A))."""
        sel = self.sheet_of_tri == k
        dead = float(np.sqrt(self.area[sel & (self.damage >= 1.)]/2.).sum()) if len(self.damage) else 0.
        if k not in getattr(self, "_orig_bnd", {}) or not self.sheet_used[k]:
            return 0., dead
        T = self.tris[sel]
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        u, c = np.unique(E, axis=0, return_counts=True)
        b = u[c == 1]
        ob = np.sort(self.vorig[b], 1)
        new = np.array([tuple(p) not in self._orig_bnd[k] for p in ob.tolist()], bool)
        L = np.linalg.norm(self._Xr[b[new, 0]] - self._Xr[b[new, 1]], axis=1)
        geometric = .5*float(L.sum())
        # the part of that length whose bond is still (partly) intact
        tb = self.tear_bonds
        intact = 0.
        for ev in self.tear_events:
            bid = ev.get("bond")
            if ev.get("sheet") != k or bid is None or "tip" not in ev or ev.get("kind") == "bowtie":
                continue
            frac = min(1., float(tb["diss"][bid])/max(float(tb["gl"][bid]), 1e-30))
            intact += (1. - frac)*float(np.linalg.norm(self._Xr[ev["v"]] - self._Xr[ev["tip"]]))
        return max(geometric - intact, 0.), dead

    def tear_summary(self):
        tb = self.tear_bonds
        out = dict(splits=len(self.tear_events), bonds=int(len(tb["a"])),
                   bonds_open=int((tb["h"] >= tb["dc"]).sum()) if len(tb["a"]) else 0,
                   tear_dissipated_j=float(self.tear_dissipated), plastic_dissipated_j=float(self.plastic_dissipated),
                   sibling_groups_excluded=len(self._sib_groups), pool_exhausted=int(self.stats.get("tear_pool_exhausted", 0)))
        out["crack_length_m"] = {self.sheets[k].name: self.crack_length(k) for k in range(len(self.sheets))
                                 if k in self._tear_sheets or (len(self.damage) and (self.damage[self.sheet_of_tri == k] >= 1.).any())}
        return out

    # ------------------------------------------------------------------ diagnostics
    STATE_FIELDS = ("q", "v", "time", "steps", "damage", "kappa_hist", "theta_p", "theta_r", "targets", "_targets_prev",
                    "finger_cmd", "reactions", "fault", "stats", "_last_gc", "_last_collisions", "fixture_force",
                    "fixture_springs", "slack_c", "slack_phi", "_stick")

    def get_state(self):
        """Deep copy of every mutable quantity: positions, velocities, damage and plastic state, robot
        targets, tape bond damage, hinge plastic angles, film logs (checkpoint/restore, start-state cache)."""
        import copy
        st = {k: copy.deepcopy(getattr(self, k, None)) for k in self.STATE_FIELDS}
        st["bonds"] = [dict(D=b.D.copy(), dmax=b.dmax.copy(), n=b.n.copy(), dissipated=b.dissipated) for b in self.bonds]
        st["hinges_b"] = [(h.theta_p, h.theta_r) for h in self.hinges_b]
        st["films"] = [len(f.log) for f in self.films]
        st["plastic_dissipated"] = float(getattr(self, "plastic_dissipated", 0.))
        if self._tear_sheets:
            # topology after splits: world and sheet triangles, the sheets' grown rest/initial rows, hinges, masses,
            # active spares, cohesive bonds, contact exclusions between siblings, tips and the event log
            st["tear"] = dict(
                tris=self.tris.copy(), hinge=self.hinge.copy(), k_hinge=self.k_hinge.copy(), k_hinge0=self.k_hinge0.copy(),
                is_crease=self.is_crease.copy(), crease_law=list(self.crease_law), vmass=self.vmass.copy(),
                vthick=self.vthick.copy(), active=self.active.copy(), vorig=self.vorig.copy(),
                Xr=self._Xr.copy(), sheet_used=list(self.sheet_used), x_free0=self.x_free0.copy(),
                cfull={k: self._cfull[k].copy() for k in ("Vc0", "vhalf", "vmu", "vclass")},
                sheets={k: (np.array(self.sheets[k].tris), np.array(self.sheets[k].X2), np.array(self.sheets[k].x0))
                        for k in self._tear_sheets},
                bonds={k: v.copy() for k, v in self.tear_bonds.items()}, sib_count=dict(self._sib_count),
                sib_groups=copy.deepcopy(self._sib_groups), events=len(self.tear_events),
                dissipated=float(self.tear_dissipated))
        return st

    def set_state(self, st):
        import copy
        for k in self.STATE_FIELDS:
            if k == "theta_r" and k not in st:             # a state without a recoverable set
                self.theta_r = np.zeros(len(st["theta_p"]))
                continue
            if k in ("slack_c", "slack_phi", "_stick") and k not in st:   # a state without these
                continue
            setattr(self, k, copy.deepcopy(st[k]))
        self._slack_refresh()
        for b, s in zip(self.bonds, st["bonds"]):
            b.D, b.dmax, b.n, b.dissipated = s["D"].copy(), s["dmax"].copy(), s["n"].copy(), s["dissipated"]
        for h, tp in zip(self.hinges_b, st["hinges_b"]):
            h.theta_p, h.theta_r = tp if isinstance(tp, tuple) else (tp, 0.)
        for f, n in zip(self.films, st["films"]):
            del f.log[n:]
            f.s0 = None
        self.plastic_dissipated = st.get("plastic_dissipated", 0.)
        tr = st.get("tear")
        if tr is not None and self._tear_sheets:
            self.tris[:] = tr["tris"]
            self.hinge[:] = tr["hinge"]
            self.k_hinge[:], self.k_hinge0[:], self.is_crease[:] = tr["k_hinge"], tr["k_hinge0"], tr["is_crease"]
            self.crease_law = list(tr["crease_law"])
            self.vmass[:], self.vthick[:] = tr["vmass"], tr["vthick"]
            self.active[:], self.vorig[:], self._Xr[:] = tr["active"], tr["vorig"], tr["Xr"]
            self.sheet_used = list(tr["sheet_used"])
            for k in self._tear_sheets:
                sl = self.sheet_slices[k]
                self.sheet_slices[k] = slice(sl.start, self._pool_start[k] + self.sheet_used[k])
            self.x_free0[:] = tr["x_free0"]
            for key in ("Vc0", "vhalf", "vmu", "vclass"):
                self._cfull[key][:] = tr["cfull"][key]
            for k, (T, X2, x0) in tr["sheets"].items():
                self.sheets[k].tris, self.sheets[k].X2, self.sheets[k].x0 = T.copy(), X2.copy(), x0.copy()
            self.tear_bonds = {k: v.copy() for k, v in tr["bonds"].items()}
            self._sib_count = dict(tr["sib_count"])
            self._sib_groups = copy.deepcopy(tr["sib_groups"])
            del self.tear_events[tr["events"]:]
            self.tear_dissipated = tr["dissipated"]
            self._tear_index()
            self._after_topology()
        elif len(self.damage) and not np.array_equal(self.damage < 1., self._contact_alive):
            self._make_contact_meshes()

    def kinetic_energy(self):
        return .5*self.v @ (self.M @ self.v)

    def summary(self):
        return dict(time=self.time, kappa_b=self.kappa, **{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                                          for k, v in self.stats.items()},
                    damaged_triangles=int((self.damage >= 1).sum()) if len(self.damage) else 0)


# ----------------------------------------------------------------------------- cohesive bonds (tape)
class Bonds:
    """Pressure-sensitive tape as cohesive bonds: each tape vertex i is tied to a material point on a sheet
    triangle (barycentric) by a bilinear traction-separation law per unit area (CohesiveLaw):
        delta = x_i - sum_k w_k x_tk - d0 n       (n: the target triangle's normal at the step start)
        traction = (1 - D) k delta;  D grows irreversibly with the largest opening (linear softening from
        delta0 = sigma/k to deltaf = 2 Gamma/sigma), so a fully opened bond has dissipated Gamma per area.
    Gamma is the release toughness the ledger gives (min of adhesive and paper-surface delamination)."""

    def __init__(self, world, vi, tri, bary, area, law, d0):
        self.w = world
        self.vi = np.asarray(vi, np.int64)
        self.tri = np.asarray(tri, np.int64)
        self.bary = np.asarray(bary, float)
        self.area = np.asarray(area, float)
        self.law = law
        self.d0 = float(d0)
        self.D = np.zeros(len(self.vi))
        self.dmax = np.zeros(len(self.vi))
        self.n = np.zeros((len(self.vi), 3))
        self.dissipated = 0.

    def begin(self, w, dt):
        x = w.x_free()
        T = w.tris[self.tri]
        nn = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-30)
        # orient toward the tape vertex
        d = x[self.vi] - (self.bary[:, :, None]*x[T]).sum(1)
        s = np.sign((d*nn).sum(1)); s[s == 0] = 1.
        self.n = nn*s[:, None]

    def _delta(self, x):
        T = self.w.tris[self.tri]
        return x[self.vi] - (self.bary[:, :, None]*x[T]).sum(1) - self.d0*self.n

    def term(self, w, q, xc_prev, dt, want):
        g = np.zeros(w.n_dof)
        coo = ([], [], [])
        live = self.D < 1.
        if not live.any():
            return 0., g, coo
        x = q[:3*w.n_free].reshape(-1, 3)
        dl = self._delta(x)
        k = (1. - self.D)*self.law.k*self.area*live
        E = .5*(k*(dl*dl).sum(1)).sum()
        f = k[:, None]*dl                                      # dE/d(delta)
        T = w.tris[self.tri]
        # tape vertex, then the three triangle corners: np.add.at's summation order, one bincount pass per component
        g[:3*w.n_free] = _scatter3(np.concatenate([self.vi, T[:, 0], T[:, 1], T[:, 2]]),
                                   np.concatenate([f] + [-self.bary[:, c:c+1]*f for c in range(3)]), w.n_free).ravel()
        if want:
            ids = np.concatenate([self.vi[:, None], T], 1)                  # (B,4)
            cw = np.concatenate([np.ones((len(self.vi), 1)), -self.bary], 1)  # (B,4)
            # the 48 (a, b, component) chunks as one chunk
            sh = (4, 4, 3, len(ids))
            d3 = np.arange(3)[None, None, :, None]
            coo[0].append(np.broadcast_to(3*ids.T[:, None, None, :] + d3, sh).ravel())
            coo[1].append(np.broadcast_to(3*ids.T[None, :, None, :] + d3, sh).ravel())
            v = k[None, None, :]*cw.T[:, None, :]*cw.T[None, :, :]
            coo[2].append(np.broadcast_to(v[:, :, None, :], sh).ravel())
        return E, g, coo

    def after(self, w, dt):
        x = w.x_free()
        op = np.linalg.norm(self._delta(x), axis=1)
        self.dmax = np.maximum(self.dmax, op)
        d0, df = self.law.delta0, self.law.deltaf
        Dn = np.where(self.dmax <= d0, 0., np.where(self.dmax >= df, 1.,
                      df*(self.dmax - d0)/np.maximum(self.dmax*(df - d0), 1e-30)))
        newD = np.maximum(self.D, np.clip(Dn, 0., 1.))
        # energy dissipated by damage growth (bilinear law: (Gamma) per area at full damage)
        self.dissipated += float((self.area*self.law.Gamma*(newD - self.D)).sum())
        self.D = newD

    @property
    def intact_fraction(self):
        return float((self.D < 1.).mean()) if len(self.D) else 0.


def install_bonds(world_ipc, bonds):
    world_ipc.bonds.append(bonds)
    world_ipc.extra_terms = list(getattr(world_ipc, "extra_terms", [])) + [bonds.term]
    world_ipc.pre_terms = list(getattr(world_ipc, "pre_terms", [])) + [bonds.begin]
    world_ipc.post_terms = list(getattr(world_ipc, "post_terms", [])) + [bonds.after]


def bond_sheet_to_sheet(world_ipc, tape_slice, target_sheet_slice, law, reach=6e-4):
    """Bond every tape vertex to the closest point of the target sheet's surface (within `reach`)."""
    x = world_ipc.x_free()
    tv = np.arange(tape_slice.start, tape_slice.stop)
    tmask = (world_ipc.tris[:, 0] >= target_sheet_slice.start) & (world_ipc.tris[:, 0] < target_sheet_slice.stop)
    tris = np.flatnonzero(tmask)
    T = world_ipc.tris[tris]
    cen = x[T].mean(1)
    from scipy.spatial import cKDTree
    kd = cKDTree(cen)
    vi, ti, bw, ar = [], [], [], []
    tape_area = np.zeros(world_ipc.n_free)
    tt = world_ipc.tris[(world_ipc.tris[:, 0] >= tape_slice.start) & (world_ipc.tris[:, 0] < tape_slice.stop)]
    A = .5*np.linalg.norm(np.cross(x[tt[:, 1]] - x[tt[:, 0]], x[tt[:, 2]] - x[tt[:, 0]]), axis=1)
    np.add.at(tape_area, tt.ravel(), np.repeat(A/3, 3))
    for v in tv:
        cand = kd.query_ball_point(x[v], 2e-2)
        best = None
        for c in cand:
            a, b, c3 = x[T[c]]
            w = _closest_bary(x[v], a, b, c3)
            p = w[0]*a + w[1]*b + w[2]*c3
            d = np.linalg.norm(x[v] - p)
            if best is None or d < best[0]:
                best = (d, tris[c], w)
        if best and best[0] < reach:
            vi.append(v); ti.append(best[1]); bw.append(best[2]); ar.append(tape_area[v])
    return vi, ti, bw, ar


def _closest_bary(p, a, b, c):
    """Barycentric weights of the closest point on triangle abc to p (Ericson)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.array([1., 0., 0.])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.array([0., 1., 0.])
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1/(d1 - d3)
        return np.array([1 - v, v, 0.])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.array([0., 0., 1.])
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2/(d2 - d6)
        return np.array([1 - w, 0., w])
    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3)/((d4 - d3) + (d5 - d6))
        return np.array([0., 1 - w, w])
    den = 1./(va + vb + vc)
    v, w = vb*den, vc*den
    return np.array([1 - v - w, v, w])


# ----------------------------------------------------------------------------- start-state relaxation
def prepare_rest_morph(w):
    """Record the authored start geometry's stretch per triangle (polar U of F = Ds Dm^-1) and the panel
    hinge angles, so the rest state can be blended from 'stress-free as authored' (lam = 0) to the true
    paper (lam = 1) while the start state settles. Only start-state generation uses this."""
    x = w.x_free()
    T = w.tris
    Ds = np.stack([x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]]], axis=2)
    F = np.matmul(Ds, w.Dm_inv)
    Cm = np.matmul(F.transpose(0, 2, 1), F)
    lam, V = np.linalg.eigh(Cm)
    U = np.matmul(V*np.sqrt(np.maximum(lam, 1e-12))[:, None, :], V.transpose(0, 2, 1))
    w._morph = dict(U=U, Dm=np.linalg.inv(w.Dm_inv), theta0=hinge_angles(x, w.hinge)[0] if len(w.hinge) else None,
                    theta_p_true=w.theta_p.copy())
    # no damage and no splits while the rest shape is a blend
    w.suspend_tearing = True
    set_rest_blend(w, 0.)


def set_rest_blend(w, lam):
    m = w._morph
    Ub = (1. - lam)*m["U"] + lam*np.eye(2)
    Dm = np.matmul(Ub, m["Dm"])
    w.Dm_inv = np.linalg.inv(Dm)
    b = np.zeros((len(w.tris), 3, 2))
    b[:, 1, :] = w.Dm_inv[:, 0, :]
    b[:, 2, :] = w.Dm_inv[:, 1, :]
    b[:, 0, :] = -(b[:, 1, :] + b[:, 2, :])
    w.bmat = b
    if m["theta0"] is not None:
        pan = ~w.is_crease
        w.theta_p[pan] = (1. - lam)*m["theta0"][pan] + lam*m["theta_p_true"][pan]


def end_rest_morph(w, keep_suspended=False):
    """The rest shape is the true paper from here on. Tearing resumes, unless the caller lets the state settle first
    (keep_suspended) and switches it back on itself (w.suspend_tearing = False)."""
    set_rest_blend(w, 1.)
    w._morph = None
    if not keep_suspended:
        w.suspend_tearing = False

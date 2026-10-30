"""Pressure-sensitive label as an orthotropic Kirchhoff-Love shell with an adhesive interface.

The label is solved by implicit Euler (incremental potential, Newton with line
search and a sparse direct solve). Rigid colliders (substrates, finger pads,
tools, liner) are analytic shapes posed by the rigid-body engine each step.

Physics:
  membrane   constant-strain triangles, orthotropic St Venant-Kirchhoff in Green strain
  bending    discrete hinges, E = k_e (theta - theta_rest)^2, k_e = c_b D(phi) |e|^2/(A1+A2)
  set        theta_rest follows theta beyond the bending yield curvature (paper creases/curl)
  contact    unilateral normal penalty with the physical adhesive-layer / pad stiffness,
             lagged smooth Coulomb friction on the non-adhesive face
  adhesion   contact -> bond (no sliding); contact fraction phi from pressure/roughness,
             dwell build-up, rate-dependent debonding of the bond front:
             v = v_ref ((G/(phi g_dwell) - w0)/(gamma90 - w0))^(1/n)   (Gent-Schultz / Maugis-Barquins)
             G from the virtual crack closure technique at front nodes
  air        enclosed unbonded pockets keep their gas (isothermal ideal gas, p V = const);
             pockets connected to the label edge vent
  drag       linear normal air drag
Nothing here moves a label vertex except these forces; there is no attachment,
teleport or kinematic grasp.
"""

from dataclasses import dataclass
import math
import time
import numpy as np
import scipy.sparse as sp
from . import materials as M

C_BEND = 0.5217       # hinge discretization constant: cylinder-wrap energy = continuum
C_BEND_REF = 0.5217   # the calibrated value; C_BEND / C_BEND_REF scales the label's effective bending stiffness
                      # (a limp-label control run divides C_BEND by 100)
PLASTIC_SHAPE_FACTOR = 1.5   # M_p / M_y for a rectangular section
C_CURV = 1.5         # hinge curvature estimator: angle/dual length = 1/R on a cylinder
# Newton step tolerance (m, largest vertex update). 1e-8 m removed the rest-state reaction spikes below but
# is unreachable during an active peel (1666 of 2100 steps hit the 32-iteration cap and the calibrated 90 deg
# point moved 9 %); the spikes are removed instead by momentum-consistent reactions (_reactions).
CONTACT_COLLECT = "carried"   # fingertip pads read the label in their own frame (see LabelSim.step)
CARRY_MAX_STEP = .02          # m per label step (10 m/s at 2 ms): larger moves are placements/parking, not motion
NEWTON_TOL = 1e-6
GAS_RTOL = 1e-3               # and each trapped-air pocket's last volume update below 0.1 % of its volume
GAS_VOLUME_AT_STEP = True     # pocket volumes enter the step measured against the substrate's current pose
GAS_CARRY = True              # ... at the start positions carried with the substrate since the last label step
GAP_FLOOR = 1e-7              # m, smallest pocket gap (regularizes V > 0; not air)
GAS_BOND_AT_FLOOR = True      # a pocket vertex in contact may bond; its air moves to its neighbours (a bubble)
GAS_SEALED_FILM = True        # a pocket enclosed at contact holds its floor film (sealed); a bonding vertex takes its film
GAS_REACTION = True           # the pocket's air pushes its substrate as hard as it pushes the label
# Tested and OFF: contact with the collider a vertex is bonded to taken from the bond's anchor plane instead of the
# nearest face. It left a held label's pad-edge membrane force unchanged (904 vs 925 N/m) and hid the pad-pair squeeze from the mid-plane set-back (a pinched vertex
# bonded to one pad then read its gap to the anchor plane, the set-back saw no squeeze, and the fast carry turn
# crushed it between the two pad penalties). Kept switchable for the record.
BONDED_CONTACT_PLANE = False


# ----------------------------------------------------------------------------- geometry
def quat_to_matrix(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass
class Collider:
    """Analytic rigid surface. Box half-extents or cylinder (radius, half_height) about local z."""
    name: str
    shape: str                      # "box" | "cylinder"
    size: tuple
    surface: M.Surface
    body: int = -1                  # rigid body index in the host engine (-1: static)
    local_pos: tuple = (0., 0., 0.)  # shape offset in body frame
    local_rot: tuple = (0., 0., 0., 1.)
    patches: tuple = ()             # ((surface, face_axis, sign, (u0,u1,v0,v1)),...) local rectangles
    role: str = "item"              # item | pad | tool | applicator | liner | plate | static
                                    # applicator: force-controlled phantom tool; plain penalty contact only
    rounding: float = 0.            # edge/corner radius of a box (m): rounded-box SDF, continuous normals
    pinch_axis: int = -1            # pads: local axis of the pinching face (set from the partner pad)
    pinch_sign: float = 1.
    pair_reach: float = 0.          # contact-pair collection reach beyond the default (fast-moving surfaces)
    partner: str = ""               # pads: the name of the partner pad of the same hand

    def __post_init__(self):
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.R_prev = np.eye(3)
        self.p_prev = np.zeros(3)
        self.Rl = quat_to_matrix(self.local_rot)
        self.pl = np.asarray(self.local_pos, float)
        self.pressure_force = 0.0   # rigid normal load reported by the host engine (sandwich rule)

    def set_body_pose(self, R_body, p_body, first=False):
        self.R_prev, self.p_prev = self.R.copy(), self.p.copy()
        self.R = R_body @ self.Rl
        self.p = R_body @ self.pl + p_body
        if first:
            self.R_prev, self.p_prev = self.R.copy(), self.p.copy()
            self.mark_step()

    def to_local(self, x):
        return (x - self.p) @ self.R

    def previous_world(self, x_local):
        return x_local @ self.R_prev.T + self.p_prev

    def mark_step(self):
        """Record the pose the label solver used in this step (the reference for Collider.carried). A host may
        pose a collider more than once per label step (a held item is re-posed after the rigid substeps), so
        p_prev is not the last label step's pose."""
        self.p_step, self.R_step = self.p.copy(), self.R.copy()

    def carried(self, x):
        """Points given at the start of the label step, carried rigidly with this collider's motion since the
        last label step (start-of-step geometry in the collider's frame; static colliders: unchanged)."""
        p0, R0 = getattr(self, "p_step", self.p), getattr(self, "R_step", self.R)
        if (np.array_equal(self.p, p0) and np.array_equal(self.R, R0)) \
                or np.linalg.norm(self.p - p0) > CARRY_MAX_STEP:     # static, or placed/parked (a jump)
            return x
        return (x - p0) @ R0 @ self.R.T + self.p

    def sdf(self, x, held=None):
        """Signed distance, outward unit normal (world) and closest surface point (local).
        `held` (pads only): points that were already in pinch-face contact last query (contact history)."""
        q = self.to_local(x)
        if self.shape == "box" and self.pinch_axis >= 0:
            dp, np_, cp = self._pad_sdf(q)
            db, nb, cb = self._box_sdf(q)
            # compressed-rubber zone just behind the pinching face: push along the face normal only;
            # elsewhere (fingertip end face when pressing, finger sides) the rounded box applies.
            # A point that entered through the pinching face keeps that face while it stays in the
            # pad's pinching half: the nearest-face rule flipped a sheet squeezed > 2 mm near the pad's
            # side to the side-face normal and pushed it in-plane at ~4 N per node (tore paper at pinch)
            # a NEW contact counts as pinched only if the pinch face is the node's nearest face: closed fingers
            # pressing straight down meet the label with their end faces, and nodes under the end face near the
            # inner pinch face lie inside the pinch layer too; classified as pinched, they were pushed apart
            # sideways along the finger gap and the pressed label tore
            # (both distances are negative inside; the pinch-face depth is never below the nearest-face depth,
            # so the pinch face is the nearest face when dp is within 0.1 mm of the rounded-box distance db)
            use = (dp < 0.) & (dp > -PAD_COMPRESSION_DEPTH) & (dp >= db - 1e-4)
            if held is not None:
                use |= held & (dp < 0.) & (dp > -float(self.size[self.pinch_axis]))
            self.last_pinch = use
            return (np.where(use, dp, db), np.where(use[:, None], np_, nb), np.where(use[:, None], cp, cb))
        if self.shape == "box":
            return self._box_sdf(q)
        return self._cyl_sdf(q)

    def _box_sdf(self, q):
        r = float(self.rounding)
        h = np.asarray(self.size, float) - r            # rounded box = inner box inflated by r
        d = np.abs(q) - h
        outside = np.linalg.norm(np.maximum(d, 0.), axis=1)
        inside = np.minimum(d.max(1), 0.)
        dist = outside + inside - r
        n = np.zeros_like(q)
        out = outside > 0
        n[out] = np.maximum(d[out], 0.) * np.sign(q[out])
        ins = ~out
        k = np.argmax(d[ins], axis=1)
        n_in = np.zeros((ins.sum(), 3))
        n_in[np.arange(ins.sum()), k] = np.sign(q[ins][np.arange(ins.sum()), k])
        n[ins] = n_in
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
        closest = q - dist[:, None]*n
        return dist, n @ self.R.T, closest

    def _cyl_sdf(self, q):
        r_c, hz = self.size
        rho = np.linalg.norm(q[:, :2], axis=1)
        radial = np.zeros_like(q)
        radial[:, :2] = q[:, :2]/np.maximum(rho, 1e-12)[:, None]
        radial[rho < 1e-12, 0] = 1.
        dr, dz = rho - r_c, np.abs(q[:, 2]) - hz
        outside = np.hypot(np.maximum(dr, 0.), np.maximum(dz, 0.))
        inside = np.minimum(np.maximum(dr, dz), 0.)
        dist = outside + inside
        axial = np.zeros_like(q)
        axial[:, 2] = np.sign(q[:, 2]) + (q[:, 2] == 0)
        n = np.where((np.maximum(dr, 0.) + np.maximum(dz, 0.) > 0)[:, None],
                     radial*np.maximum(dr, 0.)[:, None] + axial*np.maximum(dz, 0.)[:, None],
                     np.where((dr > dz)[:, None], radial, axial))
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
        closest = q - dist[:, None]*n
        return dist, n @ self.R.T, closest

    def _pad_sdf(self, q):
        """Rubber fingertip: the label touches it only through its pinching face, whose rim is a
        fillet of radius `rounding` curving away from the face. Behind the face (compressed rubber)
        the push is along the face normal, never in-plane; beyond the rim there is no contact
        (the finger's other colliders take over)."""
        a, sg, r = self.pinch_axis, self.pinch_sign, max(float(self.rounding), 1e-4)
        h = np.asarray(self.size, float)
        lat = [k for k in range(3) if k != a]
        e = np.maximum(np.abs(q[:, lat]) - (h[lat] - r), 0.)             # lateral excess into the fillet
        en = np.linalg.norm(e, axis=1)
        inside_rim = en < r
        drop = r - np.sqrt(np.maximum(r*r - en*en, 0.))                  # face falls away on the fillet
        dist = sg*q[:, a] - h[a] + drop
        # compressed rubber clamps a sheet: it pushes along its face normal only. The fillet lowers the
        # rim (penetration fades toward the edge) but does not tilt the push in-plane, which stretched
        # pinched paper between the two rims of the pad
        n = np.zeros_like(q)
        n[:, a] = sg
        dist = np.where(inside_rim, dist, 1e3)
        closest = q - dist[:, None]*n
        return dist, n @ self.R.T, closest

    def surface_at(self, closest_local, normal_world):
        """Per-point surface class index: 0 = base, k = patch k."""
        idx = np.zeros(len(closest_local), int)
        if not self.patches:
            return idx
        n_local = normal_world @ self.R
        for k, (_, axis, sign, (u0, u1, v0, v1)) in enumerate(self.patches, 1):
            if axis == "cyl":        # cylinder side: axial range u, angle range v (rad, atan2(y, x))
                ang = np.arctan2(closest_local[:, 1], closest_local[:, 0])
                hit = (np.abs(n_local[:, 2]) < .5) & (closest_local[:, 2] >= u0) & (closest_local[:, 2] <= u1) \
                    & (ang >= v0) & (ang <= v1)
                idx[hit] = k
                continue
            a, b = [i for i in range(3) if i != axis]
            hit = (n_local[:, axis]*sign > .9) & (closest_local[:, a] >= u0) & (closest_local[:, a] <= u1) \
                & (closest_local[:, b] >= v0) & (closest_local[:, b] <= v1)
            idx[hit] = k
        return idx

    def surface_list(self):
        return [self.surface] + [p[0] for p in self.patches]

    def anchor_world(self, anchor_local, anchor_n_local):
        """World position and normal of bonded material points (rigid by default)."""
        return anchor_local @ self.R.T + self.p, anchor_n_local @ self.R.T

    def substrate_curvature(self, anchor_local):
        """Curvature (1/m) of the surface under bonded material points, and its principal direction (world):
        a bonded strip is held to it. Flat faces: 0 (box edge fillets are not counted). Cylinders: 1/R round
        the circumference on the curved side."""
        k = np.zeros(len(anchor_local)); d = np.zeros((len(anchor_local), 3))
        if self.shape == "cylinder":
            r_c, hz = self.size
            q = np.asarray(anchor_local, float)
            rho = np.linalg.norm(q[:, :2], axis=1)
            side = (np.abs(q[:, 2]) < hz - 1e-6) & (rho > .5*r_c)
            k[side] = 1./r_c
            t = np.stack([-q[:, 1], q[:, 0], np.zeros(len(q))], 1)/np.maximum(rho, 1e-12)[:, None]
            d[side] = (t @ self.R.T)[side]
        return k, d


class LinerWeb(Collider):
    """Printer/dispenser release-liner web driven by the machine: bonded points travel along the web
    path (flat over the plate, then around the peel edge of radius r, then straight down).
    Anchor local x = arc length at t=0; the web advances by `feed` metres.

    The web is also a contact surface for unbonded label points: a released label cannot pass through
    the liner. The web path is the boundary of a quadrant with a rounded corner (flat top z = z_top for
    x <= edge_x, a quarter arc of radius r about (edge_x, z_top - r), then the vertical x = edge_x + r),
    so its signed distance is the rounded-corner SDF |max(k, 0)| + min(max(k_x, k_z), 0) - r with k
    measured from the arc centre; positive on the label's side. Contact is reported only within
    WEB_CONTACT_BAND of the web (a point far behind it, under the plate, is not pushed up through the plate)
    and over the web's width; pairs are collected 3 mm out (`pair_reach`) so a label snapping off the edge
    cannot tunnel through the web within one step. Its friction treats the web as static (the web's feed motion is not applied to sliding
    unbonded points). Without this contact the dispensed label threads through its own liner and jams under
    the peel plate."""

    WEB_CONTACT_BAND = 2.5e-3

    def __init__(self, name, surface, start, edge_x, radius, z_top, y_half):
        # box size only bounds the contact broad phase: cover the web from its start to beyond the edge
        span = abs(float(np.asarray(start, float)[0]) - edge_x) + .05
        super().__init__(name, "box", (span, y_half, span), surface, role="liner")
        self.start, self.edge_x, self.r, self.z_top = np.asarray(start, float), edge_x, radius, z_top
        self.y_half = y_half
        self.pair_reach = 3e-3        # a released label snaps off the peel edge at up to ~3 m/s (3 mm per 1 ms step)
        self.feed = 0.

    def sdf(self, x, held=None):
        x = np.asarray(x, float)
        cx, cz, r = self.edge_x, self.z_top - self.r, self.r
        k = np.stack([x[:, 0] - cx, x[:, 2] - cz], 1)
        kp = np.maximum(k, 0.)
        outside = np.linalg.norm(kp, axis=1)
        d = outside + np.minimum(k.max(1), 0.) - r
        g = np.zeros((len(x), 2))
        out = outside > 0
        g[out] = kp[out]/outside[out, None]
        ins = ~out
        ax = np.argmax(k[ins], axis=1)
        g[np.flatnonzero(ins), ax] = 1.
        n = np.stack([g[:, 0], np.zeros(len(x)), g[:, 1]], 1)
        valid = (np.abs(x[:, 1]) <= self.y_half) & (x[:, 0] >= self.start[0]) & (d > -self.WEB_CONTACT_BAND)
        d = np.where(valid, d, 1e3)
        closest = x - d[:, None]*n
        return d, n, closest

    def anchor_world(self, a, an):
        s = a[:, 0] + self.feed
        s_e = self.edge_x - self.start[0]
        y = a[:, 1]
        P = np.zeros((len(a), 3)); N = np.zeros((len(a), 3))
        flat = s <= s_e
        P[flat] = np.stack([self.start[0] + s[flat], y[flat], np.full(flat.sum(), self.z_top)], 1)
        N[flat] = (0, 0, 1)
        th = np.clip((s - s_e)/self.r, 0, np.pi/2)
        arc = (~flat) & (s - s_e <= self.r*np.pi/2)
        cx, cz = self.edge_x, self.z_top - self.r
        P[arc] = np.stack([cx + self.r*np.sin(th[arc]), y[arc], cz + self.r*np.cos(th[arc])], 1)
        N[arc] = np.stack([np.sin(th[arc]), 0*th[arc], np.cos(th[arc])], 1)
        down = (~flat) & ~arc
        extra = s[down] - s_e - self.r*np.pi/2
        P[down] = np.stack([np.full(down.sum(), cx + self.r), y[down], cz - extra], 1)
        N[down] = (1, 0, 0)
        # anchors sit half a label thickness off the web surface
        return P + N*self.half_offset, N

    half_offset = 0.

    def substrate_curvature(self, a):
        """1/r on the peel edge's arc (the web bends about the cross direction: principal direction x)."""
        s = a[:, 0] + self.feed
        s_e = self.edge_x - self.start[0]
        arc = (s > s_e) & (s - s_e <= self.r*np.pi/2)
        k = np.where(arc, 1./self.r, 0.)
        d = np.zeros((len(a), 3)); d[:, 0] = 1.
        return k, d


def rect_mesh(length_md, width_cd, h):
    """Rest mesh in the label plane: u along MD (length), v along CD (width)."""
    nu = max(2, int(round(length_md/h)) + 1)
    nv = max(2, int(round(width_cd/h)) + 1)
    u = np.linspace(0., length_md, nu)
    v = np.linspace(0., width_cd, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = np.stack([U.ravel(), V.ravel()], 1)
    idx = np.arange(nu*nv).reshape(nu, nv)
    tris = []
    for i in range(nu-1):
        for j in range(nv-1):
            a, b, c, d = idx[i, j], idx[i+1, j], idx[i+1, j+1], idx[i, j+1]
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    return X, np.asarray(tris, int), (nu, nv)


class LabelSim:
    """One label. Coordinates: world metres; label normal +n is the FACESTOCK side."""

    def __init__(self, stock, length_md, width_cd, h, origin, axis_md, axis_cd, *,
                 colliders=(), g=(0., 0., -M.G), newton_iters=8, tol=NEWTON_TOL, name="label",
                 newton_max=None):
        self.stock, self.name, self.h = stock, name, h
        self.X, self.tris, self.grid = rect_mesh(length_md, width_cd, h)
        self.length, self.width = length_md, width_cd
        a, b = np.asarray(axis_md, float), np.asarray(axis_cd, float)
        a /= np.linalg.norm(a)
        b -= a*(a @ b)
        b /= np.linalg.norm(b)
        self.x = np.asarray(origin, float) + self.X[:, :1]*a + self.X[:, 1:]*b
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        self.g = np.asarray(g, float)
        self.colliders = list(colliders)
        self.newton_iters, self.tol = newton_iters, tol
        # adaptive cap: a step not converged after newton_iters keeps iterating up to newton_max; one
        # still moving by more than UNRESOLVED_M after that is counted and flagged, never silently accepted
        self.newton_max = NEWTON_MAX if newton_max is None else newton_max
        self._precompute()
        self.time = 0.
        self.torn = False
        self.fault = None
        self.stats = dict(newton=0, solve_s=0., assemble_s=0., steps=0, max_residual=0., unconverged=0,
                          extended=0, unresolved=0, max_last_update_m=0., last_update_m=0.)
        # external prescribed forces (assisted fixtures only; recorded as such)
        self.fixture_force = np.zeros_like(self.x)
        self.grip_idx, self.grip_target, self.grip_k, self._grip_force = None, None, 2e4, np.zeros(3)
        self._pair_lam = np.zeros((self.n, max(len(self.colliders), 1)))
        self._pinch_held = np.zeros((self.n, max(len(self.colliders), 1)), bool)
        self._gas, self._gas_c, self.pockets = [], [], []
        self._S = dict(v=np.zeros(0, int), t=np.zeros(0, int), w=np.zeros((0, 3)), n=np.zeros((0, 3)))
        self.self_contact_force = 0.
        self._self_cand = None
        self.vertex_alive = np.ones(self.n, bool)
        self.rebond_ok = np.ones(self.n, bool)
        self._P = dict(v=np.zeros(0, int), c=np.zeros(0, int))
        # adhesive interface state
        self.bond = np.full(self.n, -1)         # collider index or -1
        self.bond_surface = np.zeros(self.n, int)
        self.anchor = np.zeros((self.n, 3))     # collider-local bonded point
        self.anchor_n = np.zeros((self.n, 3))   # collider-local surface normal
        self.phi = np.zeros(self.n)
        self.p_sustained = np.zeros(self.n)    # Pa, contact pressure filtered over M.WETOUT_TIME (wet-out drive)
        self.bond_time = np.zeros(self.n)
        self.damage = np.zeros(self.n)
        self.front_speed = np.zeros(self.n)
        self.G = np.zeros(self.n)
        self.pressure = np.zeros(self.n)
        self.gas = np.zeros(self.n)             # p*V share (J) per vertex; 0 = vented
        self.pocket_pressure = np.zeros(self.n)
        self.pocket_stiffness = np.zeros(self.n)
        self.contact_force = np.zeros((self.n, 3))
        self.reactions = {}
        self.events = []
        self._pardiso = None

    # ------------------------------------------------------------------ precompute
    def _precompute(self):
        X, T = self.X, self.tris
        s = self.stock
        Dm = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]], axis=2)   # (T,2,2) columns
        self.Dm_inv = np.linalg.inv(Dm)
        self.area = .5*np.abs(np.linalg.det(Dm))
        self.vertex_area = np.zeros(self.n)
        np.add.at(self.vertex_area, T.ravel(), np.repeat(self.area/3., 3))
        self.mass = self.vertex_area*s.rho_s
        nu12 = s.nu
        nu21 = nu12*s.E_cd/s.E_md
        den = 1. - nu12*nu21
        G12 = math.sqrt(s.E_md*s.E_cd)/(2.*(1.+s.nu))
        self.C = np.array([[s.E_md/den, nu12*s.E_cd/den, 0.],
                           [nu12*s.E_cd/den, s.E_cd/den, 0.],
                           [0., 0., G12]])
        # B matrices: f_j = sum_k b[k,j] x_k
        b = np.zeros((len(T), 3, 2))
        b[:, 1, :] = self.Dm_inv[:, 0, :]
        b[:, 2, :] = self.Dm_inv[:, 1, :]
        b[:, 0, :] = -(b[:, 1, :] + b[:, 2, :])
        self.b = b
        # hinges
        edges = {}
        for t, tri in enumerate(T):
            for k in range(3):
                i, j, o = tri[k], tri[(k+1) % 3], tri[(k+2) % 3]
                key = (min(i, j), max(i, j))
                edges.setdefault(key, []).append((t, o, (i, j)))
        hinges, boundary = [], set()
        self.edges = np.array(list(edges.keys()))
        for (i, j), items in edges.items():
            if len(items) == 2:
                (t0, o0, (a0, b0)), (t1, o1, _) = items
                hinges.append((a0, b0, o0, o1, t0, t1))
            else:
                boundary.update((i, j))
        H = np.asarray(hinges, int)
        self.hinge = H[:, :4]
        self.hinge_tris = H[:, 4:6]
        self.alive = np.ones(len(T), bool)       # triangles that still carry load (element erosion on tear)
        # Whitney-Nuismer average-stress tear criterion: the stress tested against the strength is the
        # area average over triangles within TEAR_LENGTH (reference configuration), the fracture
        # process scale of a fibre network; a single-element peak next to a nodal load is mesh noise
        cen = X[T].mean(1)
        from scipy.spatial import cKDTree
        pairs = cKDTree(cen).sparse_distance_matrix(cKDTree(cen), TEAR_LENGTH, output_type="coo_matrix")
        rows = np.r_[pairs.row, np.arange(len(T))]; cols = np.r_[pairs.col, np.arange(len(T))]
        Wt = sp.csr_matrix((self.area[cols], (rows, cols)), shape=(len(T), len(T)))
        Wt.sum_duplicates(); Wt.data[:] = self.area[Wt.indices]
        self.tear_avg = sp.diags(1./np.asarray(Wt.sum(1)).ravel()) @ Wt
        e = X[H[:, 1]] - X[H[:, 0]]
        le = np.linalg.norm(e, axis=1)
        # bending direction is perpendicular to the edge in the label plane
        perp = np.stack([-e[:, 1], e[:, 0]], 1)/le[:, None]
        cos2 = perp[:, 0]**2
        D = s.D_md*cos2 + s.D_cd*(1.-cos2)
        Ae = self.area[H[:, 4]] + self.area[H[:, 5]]
        self.k_hinge = C_BEND*D*le**2/Ae
        self.k_hinge0 = self.k_hinge.copy()
        self.hinge_len = le
        self.hinge_dual = C_CURV*Ae/(3.*le)*2.   # length over which the hinge angle is spread
        self.hinge_D = D*C_BEND/C_BEND_REF      # the effective bending stiffness the hinges carry
        self.theta_rest = np.zeros(len(H))
        # plastic hinge: the moment is capped at the fully plastic moment M_p = 1.5 M_y of a rectangular
        # section (shape factor 1.5), i.e. at 1.5 * kappa_y, kappa_y = 2 eps_y / t
        self.theta_yield = PLASTIC_SHAPE_FACTOR*s.yield_strain*2./s.t_face*self.hinge_dual
        self.boundary = np.zeros(self.n, bool)
        self.boundary[list(boundary)] = True
        # vertex adjacency (for fronts and pockets)
        E = self.edges
        self.adj = sp.coo_matrix((np.ones(2*len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                                 shape=(self.n, self.n)).tocsr()
        self.spacing = np.full(self.n, self.h)
        nt = len(self.tris)
        self.vt = sp.csr_matrix((np.ones(3*nt), (self.tris.ravel(), np.repeat(np.arange(nt), 3))),
                                shape=(self.n, nt))            # vertex -> incident triangles
        from scipy.spatial import cKDTree
        self._kdX = cKDTree(self.X[:, :2])
        self._cenX = self.X[self.tris, :2].mean(1)
        self._kdC = cKDTree(self._cenX)      # triangle centroids (material)
        self._build_pattern()

    def _build_pattern(self):
        """Fixed sparsity: triangle 3x3 blocks, hinge 4x4 blocks, diagonal blocks."""
        T, Hh = self.tris, self.hinge
        rows, cols = [], []

        def blocks(nodes):
            k = nodes.shape[1]
            I = (3*nodes[:, :, None, None] + np.arange(3)[None, None, :, None])  # (m,k,3,1)
            I = np.broadcast_to(I[:, :, :, None, None, 0], (len(nodes), k, 3, k, 3))
            J = (3*nodes[:, None, None, :, None] + np.arange(3)[None, None, None, None, :])
            J = np.broadcast_to(J, (len(nodes), k, 3, k, 3))
            return I.reshape(len(nodes), -1), J.reshape(len(nodes), -1)

        self._tri_rows, self._tri_cols = blocks(T)
        self._hin_rows, self._hin_cols = blocks(Hh)
        dn = np.arange(self.n)[:, None]
        self._dia_rows, self._dia_cols = blocks(dn)
        r = np.concatenate([self._tri_rows.ravel(), self._hin_rows.ravel(), self._dia_rows.ravel()])
        c = np.concatenate([self._tri_cols.ravel(), self._hin_cols.ravel(), self._dia_cols.ravel()])
        N = 3*self.n
        # map every COO entry to its CSR data slot (upper triangle only, Pardiso mtype 2)
        keep = c >= r
        A = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(N, N)).tocsr()
        A.sum_duplicates()
        A.sort_indices()
        self._csr = A
        lin = r.astype(np.int64)*N + c
        csr_rows = np.repeat(np.arange(N), np.diff(A.indptr)).astype(np.int64)
        csr_lin = csr_rows*N + A.indices
        order = np.argsort(csr_lin)
        pos = np.searchsorted(csr_lin[order], lin)
        slot = order[np.clip(pos, 0, len(order)-1)]
        slot[~keep] = -1
        self._slot = slot
        self._ntri, self._nhin = self._tri_rows.size, self._hin_rows.size

    # ------------------------------------------------------------------ energies
    def _membrane(self, x):
        T, b = self.tris, self.b
        xt = x[T]                                             # (T,3,3)
        F = np.matmul(xt.transpose(0, 2, 1), b)               # (T,3,2)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1)-1.), .5*((f2*f2).sum(1)-1.), (f1*f2).sum(1)], 1)
        S = e @ self.C.T                                      # Voigt stress (T,3)
        w = self.area*self.stock.t_face*self.alive
        E = .5*(w*(e*S).sum(1)).sum()
        # J (T,3,9): de/dx_k
        J = np.zeros((len(T), 3, 3, 3))
        J[:, 0] = b[:, :, 0, None]*f1[:, None, :]
        J[:, 1] = b[:, :, 1, None]*f2[:, None, :]
        J[:, 2] = b[:, :, 0, None]*f2[:, None, :] + b[:, :, 1, None]*f1[:, None, :]
        J = J.reshape(len(T), 3, 9)
        grad = w[:, None]*np.matmul(S[:, None, :], J)[:, 0, :]
        JT = J.transpose(0, 2, 1)
        Hm = w[:, None, None]*np.matmul(np.matmul(JT, self.C), J)
        # geometric stiffness with the PSD part of S (keeps Newton well posed in compression)
        S2 = np.stack([np.stack([S[:, 0], S[:, 2]], 1), np.stack([S[:, 2], S[:, 1]], 1)], 1)
        lam, V = np.linalg.eigh(S2)
        S2p = np.matmul(V*np.maximum(lam, 0.)[:, None, :], V.transpose(0, 2, 1))
        Kg = np.matmul(np.matmul(b, S2p), b.transpose(0, 2, 1))   # (T,3,3)
        Hg = w[:, None, None, None, None]*Kg[:, :, None, :, None]*np.eye(3)[None, None, :, None, :]
        Hm = Hm + Hg.reshape(len(T), 9, 9)
        self._membrane_force_per_width = S*self.stock.t_face   # N/m (Voigt)
        return E, grad, Hm

    def _hinge_angles(self, x):
        H = self.hinge
        x3, x4, x1, x2 = x[H[:, 0]], x[H[:, 1]], x[H[:, 2]], x[H[:, 3]]
        E = x4 - x3
        le = np.linalg.norm(E, axis=1)
        N1 = np.cross(x1 - x3, x1 - x4)
        N2 = np.cross(x2 - x4, x2 - x3)
        n1s, n2s = (N1*N1).sum(1), (N2*N2).sum(1)
        n1, n2 = N1/np.sqrt(n1s)[:, None], N2/np.sqrt(n2s)[:, None]
        theta = np.arctan2((np.cross(n1, n2)*E).sum(1)/le, (n1*n2).sum(1))
        u1 = le[:, None]*N1/n1s[:, None]
        u2 = le[:, None]*N2/n2s[:, None]
        u3 = (((x1-x4)*E).sum(1)/le)[:, None]*N1/n1s[:, None] + (((x2-x4)*E).sum(1)/le)[:, None]*N2/n2s[:, None]
        u4 = -(((x1-x3)*E).sum(1)/le)[:, None]*N1/n1s[:, None] - (((x2-x3)*E).sum(1)/le)[:, None]*N2/n2s[:, None]
        grad = np.stack([u3, u4, u1, u2], 1)                  # order matches hinge (x3,x4,x1,x2)
        return theta, -grad

    def _bending(self, x):
        theta, dth = self._hinge_angles(x)
        d = theta - self.theta_rest
        k = self.k_hinge
        E = (k*d*d).sum()
        g = (2*k*d)[:, None, None]*dth                      # (H,4,3)
        d12 = dth.reshape(-1, 12)
        Hb = (2*k)[:, None, None]*(d12[:, :, None]*d12[:, None, :])
        return E, g.reshape(-1, 12), Hb, theta

    # ------------------------------------------------------------------ contacts
    def _carry(self, c, x, idx=None):
        """Fingertip pads read the label at the start of the step in their own frame (the positions carried
        with the pad since the last label step); every other collider reads the start positions as they are."""
        col = self.colliders[c]
        return col.carried(x) if col.role == "pad" else x

    def _contact_pairs(self, x, reach=1.5e-3, update_history=True, extended=True, carried=False):
        """All (vertex, collider) pairs within `reach` of contact: a label can be sandwiched. `extended`: a
        collider's own `pair_reach` widens the collection (lagged pairs must exist before a fast node arrives,
        or it tunnels through); bond detection passes extended=False."""
        half = .5*self.stock.thickness
        pv, pc, pd, pn, pl = [], [], [], [], []
        for c, col in enumerate(self.colliders):
            xc = self._carry(c, x) if carried else x
            ext = np.max(col.size) + np.linalg.norm(col.size) + reach + half
            lo, hi = col.p - ext, col.p + ext
            cand = np.nonzero(((xc >= lo) & (xc <= hi)).all(1) & self.vertex_alive)[0]
            if not len(cand):
                continue
            if col.pinch_axis >= 0 and col.shape == "box":
                if self._pinch_held.shape[1] < len(self.colliders):
                    self._pinch_held = np.pad(self._pinch_held, ((0, 0), (0, len(self.colliders) - self._pinch_held.shape[1])))
                d, n, q = col.sdf(xc[cand], held=self._pinch_held[cand, c])
                if update_history:
                    self._pinch_held[:, c] = False
                    self._pinch_held[cand, c] = col.last_pinch
            else:
                d, n, q = col.sdf(xc[cand])
            keep = d < half + (max(reach, col.pair_reach) if extended else reach)
            if keep.any():
                pv.append(cand[keep]); pc.append(np.full(keep.sum(), c)); pd.append(d[keep])
                pn.append(n[keep]); pl.append(q[keep])
        if not pv:
            return dict(v=np.zeros(0, int), c=np.zeros(0, int), d=np.zeros(0), n=np.zeros((0, 3)), loc=np.zeros((0, 3)))
        return dict(v=np.concatenate(pv), c=np.concatenate(pc), d=np.concatenate(pd), n=np.concatenate(pn),
                    loc=np.concatenate(pl))

    def _single_pad_contact(self, P, x, carried=False):
        """A fingertip pad pushes along its pinching-face normal only where the sheet is squeezed between it
        and its partner pad (that rule stops squeezed paper being flipped sideways at the pad's side and torn
        at pinch onset). Where a vertex touches ONE pad only -- paper draped or pulled over the
        fingertip's edge, e.g. a hanging label as the hand turns fingers-down -- the pad acts with its true
        rounded-box geometry, so the sheet can wrap around the rounded edge. With the face-normal rule there,
        draped nodes sit 0.3-0.5 mm inside the pad with 8-15 N each, push the fingers apart (pinch 20 -> 3-9 N)
        and the carry turn tears the parcel label."""
        if not len(P["v"]):
            return
        half = .5*self.stock.thickness
        names = {col.name: c for c, col in enumerate(self.colliders)}
        close = P["d"] < half + 1.0e-3
        for c, col in enumerate(self.colliders):
            if col.pinch_axis < 0 or not col.partner or col.partner not in names:
                continue
            k = np.flatnonzero(P["c"] == c)
            if not len(k):
                continue
            partner = names[col.partner]
            squeezed = set(P["v"][(P["c"] == partner) & close].tolist())
            single = np.array([P["v"][i] not in squeezed for i in k])
            if not single.any():
                continue
            ks = k[single]
            xs = x[P["v"][ks]]
            d, n, q = col._box_sdf(col.to_local(self._carry(c, xs, P["v"][ks]) if carried else xs))
            P["d"][ks], P["n"][ks], P["loc"][ks] = d, n, q

    def _pad_on_adhesive_side(self, Q, x, ln):
        """Bond formation through a fingertip pad: the adhesive-side test (label normal . contact normal > 0) reads
        the pad face nearest to the node. A closed fingertip pressing the label from above sinks into the label
        region (rubber compression, PAD_COMPRESSION_DEPTH); a node that ends up nearest one of its vertical faces (the
        pinching face, or an outer side face) is read against that face's horizontal normal, and the sheet's small
        tilt toward it passes the test, which would bond the pad to the PRINTED side (in the task ladder's
        smooth-corner robot runs without this rule: every recorded pad bond 0.001-2.3 mm inside the pad, label normal
        z 0.95-1.00, component along the face normal 0.0001-0.29; 9 of 11 fingertip attempts on lifted areas tore).
        A pad can only bond the face of the sheet that faces it: the pad body must lie on the adhesive side of the
        sheet at that node, (pad centre - node) . label normal < 0 (the label normal points to the printed face). A
        pinched sheet bonds its adhesive-side pad, and a sheet wrapped over the fingertip or touching it with its
        adhesive still bonds; a pad on the printed side never bonds. Returns False for such pad pairs; True for every
        other pair. Contact forces are unchanged."""
        ok = np.ones(len(Q["v"]), bool)
        for c, col in enumerate(self.colliders):
            if col.role != "pad":
                continue
            k = np.flatnonzero(Q["c"] == c)
            if len(k):
                v = Q["v"][k]
                ok[k] = ((col.p - x[v])*ln[v]).sum(1) < 0.
        return ok

    def _bonded_contact_plane(self, P, x):
        """A vertex bonded to a collider touches it where it is bonded: its penalty contact with that collider
        uses the bond's own surface point and normal (the anchor plane), not the collider's nearest face.
        Near a fingertip pad's rim, or after the hand turns, the nearest face of the pad's rounded box can be a
        different face from the one the node is bonded to, so bond and contact pull the node toward two surfaces
        and strain the held sheet (925 N/m at the pad edge of a label held still;
        24-33 N on single nodes during the carry that tore the parcel label)."""
        if not len(P["v"]):
            return
        bonded = self.bond[P["v"]] == P["c"]
        if not bonded.any():
            return
        half = .5*self.stock.thickness
        for c in np.unique(P["c"][bonded]):
            k = np.flatnonzero(bonded & (P["c"] == c))
            v = P["v"][k]
            col = self.colliders[c]
            pa, na = col.anchor_world(self.anchor[v], self.anchor_n[v])
            surf = pa - na*half                                     # the bonded surface point (anchor is half t off)
            P["n"][k] = na
            P["d"][k] = ((x[v] - surf)*na).sum(1)
            P["loc"][k] = col.to_local(surf) if not isinstance(col, LinerWeb) else surf

    def _self_pairs(self, x):
        """Lagged self-contact pairs: (vertex, triangle) closer than SELF_CONTACT_REACH whose rest
        positions are more than 1.5 element lengths apart (neighbours are held apart by the sheet's own
        membrane/bending). Returns vertex, triangle, closest-point barycentrics and separation normal."""
        empty = dict(v=np.zeros(0, int), t=np.zeros(0, int), w=np.zeros((0, 3)), n=np.zeros((0, 3)))
        if not SELF_CONTACT:
            return empty
        live = np.flatnonzero(self.alive)
        if not len(live):
            return empty
        # candidate list (Verlet): rebuilt with a SELF_CONTACT_SKIN margin only when some vertex has
        # moved more than half the skin since the last build (the label mostly lies still)
        C = self._self_cand
        if C is None or C["alive"] != int(self.alive.sum()) or \
                np.abs(x - C["x"]).max() > .5*SELF_CONTACT_SKIN:
            from scipy.spatial import cKDTree
            cen = x[self.tris[live]].mean(1)
            reach = SELF_CONTACT_REACH + 1.2*self.h + SELF_CONTACT_SKIN   # centroid-to-vertex radius
            M_ = cKDTree(x).sparse_distance_matrix(cKDTree(cen), reach, output_type="coo_matrix")
            v, t, dcur = M_.row, live[M_.col], M_.data
            # a pair can touch only where the sheet has come back onto itself: rest distance well
            # beyond both the element size and the current distance (a flat or gently curved sheet
            # has rest ~ current distance and is skipped before any closest-point work)
            drest = np.linalg.norm(self.X[v] - self.X[self.tris[t]].mean(1), axis=1)
            keep = (drest > 1.5*self.h) & (drest > dcur + 1.5*self.h - SELF_CONTACT_SKIN) & \
                ~(self.tris[t] == v[:, None]).any(1) & self.vertex_alive[v]
            C = self._self_cand = dict(x=x.copy(), v=v[keep], t=t[keep], alive=int(self.alive.sum()))
        v, t = C["v"], C["t"]
        if not len(v):
            return empty
        a, b, c = (x[self.tris[t][:, k]] for k in range(3))
        w = _closest_on_triangle(x[v], a, b, c)
        q = w[:, :1]*a + w[:, 1:2]*b + w[:, 2:]*c
        dv = x[v] - q
        dist = np.linalg.norm(dv, axis=1)
        close = dist < SELF_CONTACT_REACH
        if not close.any():
            return empty
        v, t, w, dv, dist = v[close], t[close], w[close], dv[close], dist[close]
        tn = np.cross(b[close] - a[close], c[close] - a[close])
        tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-15)
        side = np.sign((dv*tn).sum(1)); side[side == 0] = 1.
        n = np.where((dist > 1e-6)[:, None], dv/np.maximum(dist, 1e-15)[:, None], tn*side[:, None])
        return dict(v=v, t=t, w=w, n=n)

    def _label_normals(self, x):
        T = self.tris
        fn = np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]])
        vn = np.zeros_like(x)
        for k in range(3):
            np.add.at(vn, T[:, k], fn)
        return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-15)

    # ------------------------------------------------------------------ step
    def step(self, dt):
        if self.fault:
            raise RuntimeError(self.fault)
        t0 = time.perf_counter()
        x0 = self.x.copy()
        m = self.mass
        half = .5*self.stock.thickness
        xt = x0 + dt*self.v + dt*dt*self.g
        # lagged per-step contact inputs (pairs, adhesive side, sandwich pressure, friction loads), read from the
        # start-of-step positions against the colliders' end-of-step pose, except that the robot's fingertip
        # pads read them in their own frame: the start positions carried with the pad since the last label step
        # (_carry, Collider.carried, mark_step), so a pad sweeping 1-2.5 mm per step meets the sheet one step
        # late instead of by a phantom penetration equal to its step. Traced per step:
        # (a) no carrying: a label held in a fingertip moving 1.6-2.5 mm per step was compared with the pads
        #     already moved that far: tangent planes 30-50 deg off, up to 1.4 mm of phantom penetration
        #     (5-13 N per node); the carry turn tore it;
        # (b) every vertex at its predicted position x0 + dt v: a released trailing edge was extrapolated across
        #     the printer's sharp box corners and 4 of 18 takes tore;
        # (c) carrying with every moving collider: the tube finger (a tool that slides round the tube) dragged
        #     the free sheet's geometry with it and the wrap tore at placement;
        # (d) carrying only the vertices a hand holds (bonded, pinched, touched last step): vertices newly met
        #     by the sweeping pad still saw its whole step; the carry turn ran at 0.6-1.6 kN/m (median);
        # (e) a collider re-posed twice in one label step (a held item) needs the pose of the last LABEL step
        #     as the reference (mark_step).
        carried = CONTACT_COLLECT == "carried"
        P = self._contact_pairs(x0, carried=carried)
        if BONDED_CONTACT_PLANE:
            self._bonded_contact_plane(P, x0)
        self._single_pad_contact(P, x0, carried=carried)
        # a pocket's start geometry carried with its substrate since the last label step (read before mark_step)
        gas_base = {}
        if GAS_CARRY:
            for k, pk in enumerate(self.pockets):
                col = next((c for c in self.colliders if c.name == pk["collider"]), None)
                if col is not None and self.gas[pk["vertices"]].sum() > 0:
                    gas_base[k] = col.carried(x0[pk["vertices"]])
        for col in self.colliders:
            col.mark_step()
        ln = self._label_normals(x0)
        P["adh"] = (ln[P["v"]]*P["n"]).sum(1) > 0.    # collider on the label's -n (adhesive) face
        self.pressure[:] = 0.
        count = np.bincount(P["v"], minlength=self.n)
        skip = np.zeros(len(P["v"]), bool)
        for c, col in enumerate(self.colliders):
            pf = col.pressure_force
            if col.role not in ("pad", "tool") or not pf:
                continue
            m = (P["c"] == c) & (P["d"] < half + 1.0e-3)
            under = np.unique(P["v"][m])
            under = under[count[under] >= 2]              # squeezed between this pad and something else
            if not len(under):
                continue
            if isinstance(pf, dict):
                # rigid load between this pad and each partner body goes to the label squeezed
                # between exactly those two bodies
                near = (P["d"] < half + 1.0e-3) & np.isin(P["v"], under) & (P["c"] != c)
                for c2 in np.unique(P["c"][near]):
                    F = pf.get(self.colliders[c2].body, 0.)
                    if F <= 0:
                        continue
                    v2 = np.unique(P["v"][near & (P["c"] == c2)])
                    area = self.vertex_area[v2].sum()
                    if area > 0:
                        self.pressure[v2] = np.maximum(self.pressure[v2], F/area)
                        if self.colliders[c2].role not in ("pad", "tool"):
                            # pressed onto an item: the rigid pad-item contact carries this load and the
                            # soft rigid contact may overlap pad and item by mm; the pad must not also
                            # crush the 0.1 mm label through its own penalty
                            skip |= (P["c"] == c) & np.isin(P["v"], v2)
            else:
                area = self.vertex_area[under].sum()
                if area > 0 and pf > 0:
                    self.pressure[under] = np.maximum(self.pressure[under], pf/area)
        # squeezed thinner than the label: a gripping surface (pad/tool) and an opposing surface (another
        # pad, an item, the printer plate) closer together than the label is thick -- soft rigid contact
        # lets closing fingers overlap by ~1 mm and the load report lags the geometry. Both penalties
        # cannot hold, so each surface is set back by half the deficit: the label sits at the mid-plane,
        # continuously (no per-node switching), and the squeeze load comes from the rigid solver above.
        P["off"] = np.zeros(len(P["v"]))
        grip = np.array([c.role in ("pad", "tool") for c in self.colliders])
        close = np.flatnonzero(P["d"] < half + 1.0e-3)
        if len(close) and grip[P["c"][close]].any():
            order = close[np.argsort(P["v"][close], kind="stable")]
            vs = P["v"][order]
            starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
            ends = np.r_[starts[1:], len(order)]
            for s0, s1 in zip(starts, ends):
                if s1 - s0 < 2:
                    continue
                ks = order[s0:s1]
                gks = ks[grip[P["c"][ks]]]
                if not len(gks):
                    continue
                kg = gks[np.argmin(P["d"][gks])]
                opp = ks[(P["c"][ks] != P["c"][kg]) & ((P["n"][ks]*P["n"][kg]).sum(1) < 0.)]
                if not len(opp):
                    continue
                ko = opp[np.argmin(P["d"][opp])]
                deficit = 2*half - (P["d"][kg] + P["d"][ko])
                if deficit > 0:
                    P["off"][kg] += .5*deficit
                    P["off"][ko] += .5*deficit
            P["d"] = P["d"] + P["off"]
        if self._pair_lam.shape[1] < len(self.colliders):
            self._pair_lam = np.pad(self._pair_lam, ((0, 0), (0, len(self.colliders) - self._pair_lam.shape[1])))
        P["lam"] = self._pair_lam[P["v"], P["c"]] if len(P["v"]) else np.zeros(0)
        if skip.any():
            P = {k: (v[~skip] if isinstance(v, np.ndarray) and len(v) == len(skip) else v) for k, v in P.items()}
        self._P = P
        self._S = self._self_pairs(x0)
        ln0 = self._label_normals(x0)
        # a pocket's volume at the start of the step is measured against its substrate's CURRENT pose: taken from
        # the previous label step, a carton springing back up after the roller left (~3 um per step) squeezed a
        # 3.5 um one-node pocket to the gap floor every step for 64 ms.
        # It is measured at the start positions CARRIED with the substrate since the last label step (as the
        # fingertip pads read contact), and the volume is linear about them: V = V0 + sum a.(x - x_carried).
        # Measured at the uncarried positions, a substrate that had moved into the pocket between label steps
        # left its vertices below the gap floor; the floor-clamped V0 then credited their climb back to the
        # surface as air, and the gas SUCKED the sheet into the substrate: a 6.7 um pocket on a substrate rising
        # 7 um per step sat 3.3 um inside it reading 5.9 MPa for as long as the rise lasted, as in the robot
        # chain's settle when the carton sprang back 230 um after the roller left. Carried, the substrate's motion compresses the air (V at the
        # uncarried x0 is the true, signed volume), and the Newton iterate starts at the carried positions (V > 0)
        self._gas = []
        for k, pk in enumerate(self.pockets):
            v = pk["vertices"]
            if self.gas[v].sum() <= 0:
                continue
            xb = gas_base.get(k, x0[v])
            self._gas.append(dict(v=v, a=self.vertex_area[v, None]*ln0[v], n=float(self.gas[v].sum()), xb=xb, col=pk["collider"],
                                  V0=self._pocket_volume(pk, xb) if GAS_VOLUME_AT_STEP else pk["volume_m3"]))
        self._gas_c = []
        x = x0.copy()
        for gp in self._gas:
            x[gp["v"]] = gp["xb"]
        converged = False
        last = 0.
        for it in range(max(self.newton_max, self.newton_iters)):
            if it == self.newton_iters:
                self.stats["extended"] += 1          # beyond the base cap: adaptive continuation
            E, g, Hd = self._assemble(x, xt, dt, x0, P)
            dx = self._solve(Hd, -g, self._gas, self._gas_c)
            # line search on total energy
            alpha, E0 = 1., E
            gd = g @ dx
            for _ in range(12):
                x_try = x + alpha*dx.reshape(-1, 3)
                E1 = self._energy(x_try, xt, dt, x0, P)
                if E1 <= E0 + 1e-4*alpha*gd or alpha < 1e-3:
                    break
                alpha *= .5
            x = x_try
            self.stats["newton"] += 1
            last = float(np.abs(alpha*dx).max())
            # a pocket's gas converges in its volume, not in metres: under the log barrier a Newton step can at
            # most about double a small pocket's volume, so a pocket squeezed to a micrometre-scale gap moves
            # its vertices by less than the displacement tolerance while its pressure is still orders of
            # magnitude out of balance. Tested on displacement alone, such a step was accepted: the pocket
            # collapsed onto its substrate, stayed pinned at the gap floor (MPa read, never applied) and
            # released its stored n ln(V/V_floor) seconds later as a kinetic kick that tore the parcel label
            # under the rub
            gas_rel = 0.
            if self._gas:
                d = (alpha*dx).reshape(-1, 3)
                for gp in self._gas:
                    v = gp["v"]
                    V = gp["V0"] + (gp["a"]*(x[v] - gp["xb"])).sum()
                    gas_rel = max(gas_rel, abs(float((gp["a"]*d[v]).sum()))/max(V, 1e-30))
            if last < self.tol and gas_rel < GAS_RTOL:
                converged = True
                break
        if not np.isfinite(x).all():
            self.fault = f"{self.name}: non-finite label state at t={self.time:.4f}s"
            raise FloatingPointError(self.fault)
        if not self.vertex_alive.all():
            # a vertex whose every triangle has failed is torn-off debris with no mechanics left: it is
            # frozen where it tore (it no longer belongs to the sheet; nothing draws or measures it)
            dv = ~self.vertex_alive
            x[dv] = x0[dv]
        self.v = (x - x0)/dt
        self.x = x
        self.time += dt
        self.stats["steps"] += 1
        self.stats["last_converged"] = converged
        self.stats["unconverged"] = self.stats.get("unconverged", 0) + (not converged)
        self.stats["last_update_m"] = last
        self.stats["max_last_update_m"] = max(self.stats.get("max_last_update_m", 0.), last if not converged else 0.)
        self.stats["unresolved"] = self.stats.get("unresolved", 0) + (not converged and last > UNRESOLVED_M)
        self._post_step(dt, P)
        self.stats["step_s"] = time.perf_counter() - t0
        return converged

    def _terms(self, x, xt, dt, x0, P, want_hessian):
        """Per-vertex external/interface terms: returns energy, grad (n,3), diag blocks (n,3,3)."""
        n = self.n
        m = self.mass
        half = .5*self.stock.thickness
        A = self.vertex_area
        grad = np.zeros((n, 3))
        Hd = np.zeros((n, 3, 3)) if want_hessian else None
        dxv = x - xt
        E = .5*(m*(dxv*dxv).sum(1)).sum()/dt**2
        grad += m[:, None]*dxv/dt**2
        if want_hessian:
            Hd += (m/dt**2)[:, None, None]*np.eye(3)
        # air drag along the label normal (linear, implicit)
        ln = self._label_normals(x0)
        c_d = self.stock.drag*A
        vn = ((x - x0)*ln).sum(1)/dt
        E += .5*(c_d*vn*vn).sum()*dt
        grad += (c_d*vn)[:, None]*ln
        if want_hessian:
            Hd += (c_d/dt)[:, None, None]*(ln[:, :, None]*ln[:, None, :])
        # assisted test-machine jaw (fixtures only; never exposed to a policy)
        if self.grip_idx is not None:
            gi = self.grip_idx
            r = x[gi] - self.grip_target
            E += .5*self.grip_k*(r*r).sum()
            grad[gi] += self.grip_k*r
            self._grip_force = -(self.grip_k*r).sum(0)
            if want_hessian:
                Hd[gi] += self.grip_k*np.eye(3)
        # prescribed loads (fixtures)
        f_ext = self.fixture_force
        E -= (f_ext*(x - x0)).sum()
        grad -= f_ext
        # trapped gas: each enclosed pocket is an isothermal ideal gas, E = -n ln V + p_atm V, with the
        # pocket volume linear in the label's normal motion over the step, V = V0 + sum_i A_i n_i.(x_i - x0_i).
        # The pocket couples all its vertices (one pressure): the Hessian is the rank-one n/V^2 a a^T,
        # applied exactly in _solve (Woodbury). A per-vertex diagonal stiffness let a 3744 mm^2 pocket
        # collapse 280x inside one 1 ms step (10^8 Pa, label blown off the plate); ln V is also a barrier
        # that keeps V > 0 in the line search.
        if want_hessian:
            self._gas_c = []
        for gp in self._gas:
            v = gp["v"]
            V = gp["V0"] + (gp["a"]*(x[v] - gp["xb"])).sum()
            if V <= 0.:
                return np.inf, grad, Hd
            E += -gp["n"]*np.log(V/gp["V0"]) + PATM*(V - gp["V0"])
            grad[v] += (PATM - gp["n"]/V)*gp["a"]
            if want_hessian:
                self._gas_c.append(gp["n"]/V**2)
        # collider contact pairs: unilateral normal penalty about the lagged tangent plane
        self._pf = np.zeros((len(P["v"]), 3))
        for c in np.unique(P["c"]):
            col = self.colliders[c]
            mk = np.nonzero(P["c"] == c)[0]
            sel = P["v"][mk]
            p_surf = P["loc"][mk] @ col.R.T + col.p
            nrm = P["n"][mk]
            d = ((x[sel] - p_surf)*nrm).sum(1) + P["off"][mk]
            k = col.surface.stiffness*A[sel]
            pen = np.maximum(half - d, 0.)
            E += .5*(k*pen*pen).sum()
            f = (k*pen)[:, None]*nrm
            np.add.at(grad, sel, -f)
            self._pf[mk] += f
            if want_hessian:
                np.add.at(Hd, sel, (k*(pen > 0))[:, None, None]*(nrm[:, :, None]*nrm[:, None, :]))
            # friction on non-bonded pairs (the facestock side, or adhesive against a non-bonding surface)
            fm = ~(P["adh"][mk] & (self.bond[sel] == c)) & (P["lam"][mk] > 0)
            if fm.any():
                fr = sel[fm]
                mu = col.surface.mu
                l = P["lam"][mk][fm]
                surf_prev = col.previous_world(P["loc"][mk][fm])
                u = (x[fr] - x0[fr]) - (p_surf[fm] - surf_prev)
                nn = nrm[fm]
                ut = u - (u*nn).sum(1, keepdims=True)*nn
                un = np.linalg.norm(ut, axis=1)
                eps = 1e-5*dt/1e-3 + 1e-7
                kf = mu*l/eps          # smooth (IPC f0) friction potential, stick zone eps
                stick = un < eps
                E += np.where(stick, .5*kf*un*un, mu*l*(un - .5*eps)).sum()
                ff = np.where(stick[:, None], kf[:, None]*ut, (mu*l/np.maximum(un, 1e-15))[:, None]*ut)
                np.add.at(grad, fr, ff)
                self._pf[mk[fm]] -= ff
                if want_hessian:
                    Pm = np.eye(3)[None] - nn[:, :, None]*nn[:, None, :]
                    kk = np.where(stick, kf, mu*l/np.maximum(un, eps))
                    np.add.at(Hd, fr, kk[:, None, None]*Pm)
        # self-contact: vertex vs. non-neighbouring triangle, unilateral penalty at one label thickness
        # along the lagged separation normal (frictionless, non-adhesive)
        S = self._S
        if len(S["v"]):
            w3 = S["w"]
            tri = self.tris[S["t"]]
            xc = (w3[:, :, None]*x[tri]).sum(1)
            d = ((x[S["v"]] - xc)*S["n"]).sum(1)
            pen = np.maximum(self.stock.thickness - d, 0.)
            k = SELF_CONTACT_STIFFNESS*A[S["v"]]
            E += .5*(k*pen*pen).sum()
            f = (k*pen)[:, None]*S["n"]
            np.add.at(grad, S["v"], -f)
            for c in range(3):
                np.add.at(grad, tri[:, c], w3[:, c, None]*f)
            if want_hessian:
                nn = (k*(pen > 0))[:, None, None]*(S["n"][:, :, None]*S["n"][:, None, :])
                np.add.at(Hd, S["v"], nn)
                for c in range(3):
                    np.add.at(Hd, tri[:, c], (w3[:, c]**2)[:, None, None]*nn)
            self.self_contact_force = float(np.linalg.norm(f, axis=1).sum())
        # bonds: adhesive-layer springs to the collider material point (tension + shear only)
        bd = np.nonzero(self.bond >= 0)[0]
        self._bf = np.zeros((n, 3))
        for c in np.unique(self.bond[bd]):
            col = self.colliders[c]
            sel = bd[self.bond[bd] == c]
            p, nrm = col.anchor_world(self.anchor[sel], self.anchor_n[sel])
            r = x[sel] - p
            rn = (r*nrm).sum(1)
            rt = r - rn[:, None]*nrm
            # adhesive layer in series with the surface's own compliance (rubber fingertip pads)
            surfs = col.surface_list()
            sn = np.array([surfs[k].stiffness for k in self.bond_surface[sel]])
            ss = np.array([surfs[k].shear_stiffness for k in self.bond_surface[sel]])
            body = np.isfinite(ss)          # compliant body (rubber pad): in series; rigid items: adhesive only
            # the front advancing through a node's tributary area leaves the fraction (1 - damage) of it
            # bonded: the bond softens as the front crosses the node instead of holding full stiffness
            # until the whole 3 mm row lets go at once (that released a row of slack per element and
            # turned a steady peel into a 0 -> 3x stick-slip sawtooth)
            Ab = A[sel]*np.clip(1. - self.damage[sel], 0., 1.)
            kn = Ab*np.where(body, 1./(1./M.ADHESIVE_NORMAL_STIFFNESS + 1./sn), M.ADHESIVE_NORMAL_STIFFNESS)
            kt = Ab*np.where(body, 1./(1./M.ADHESIVE_SHEAR_STIFFNESS + 1./ss), M.ADHESIVE_SHEAR_STIFFNESS)
            ten = np.maximum(rn, 0.)
            E += .5*(kn*ten*ten).sum() + .5*(kt*(rt*rt).sum(1)).sum()
            f = -(kn*ten)[:, None]*nrm - kt[:, None]*rt
            grad[sel] -= f
            self._bf[sel] += f
            if want_hessian:
                P = np.eye(3)[None] - np.einsum("ni,nj->nij", nrm, nrm)
                Hd[sel] += (kn*(rn > 0))[:, None, None]*np.einsum("ni,nj->nij", nrm, nrm) + kt[:, None, None]*P
        return E, grad, Hd

    def _energy(self, x, xt, dt, x0, P):
        Em, _, _ = self._membrane_energy_only(x)
        Eb = (self.k_hinge*(self._hinge_angles(x)[0] - self.theta_rest)**2).sum()
        Et, gt, _ = self._terms(x, xt, dt, x0, P, False)
        # net Newton residual force at the evaluated state: membrane and bending forces are internal (they sum
        # to zero), so the sum of the external/interface gradient IS the residual of the whole label's momentum
        # balance (0 at an exact solution). Kept for the accepted state (_reactions).
        self._residual_sum = np.asarray(gt).sum(0) if np.all(np.isfinite(gt)) else np.zeros(3)
        return Em + Eb + Et

    def _membrane_energy_only(self, x):
        T, b = self.tris, self.b
        F = np.matmul(x[T].transpose(0, 2, 1), b)
        f1, f2 = F[:, :, 0], F[:, :, 1]
        e = np.stack([.5*((f1*f1).sum(1)-1.), .5*((f2*f2).sum(1)-1.), (f1*f2).sum(1)], 1)
        return .5*((self.area*self.stock.t_face*self.alive)*(e*(e @ self.C.T)).sum(1)).sum(), None, None

    def _assemble(self, x, xt, dt, x0, P):
        t = time.perf_counter()
        Em, gm, Hm = self._membrane(x)
        Eb, gb, Hb, _ = self._bending(x)
        Et, gt, Hd = self._terms(x, xt, dt, x0, P, True)
        g = gt.ravel().copy()
        np.add.at(g, (3*self.tris[:, :, None] + np.arange(3)).reshape(len(self.tris), 9).ravel(), gm.ravel())
        np.add.at(g, (3*self.hinge[:, :, None] + np.arange(3)).reshape(len(self.hinge), 12).ravel(), gb.ravel())
        vals = np.concatenate([Hm.ravel(), Hb.ravel(), Hd.ravel()])
        slot = self._slot
        ok = slot >= 0
        data = np.bincount(slot[ok], weights=vals[ok], minlength=self._csr.nnz)
        A = self._csr.copy()
        A.data = data
        self.stats["assemble_s"] += time.perf_counter() - t
        return Em + Eb + Et, g, A

    def _solve(self, A, rhs, gas=(), gas_c=()):
        """Solve (A + sum_k c_k u_k u_k^T) x = rhs: sparse A by Pardiso, the pocket gas rank-one
        terms by the Woodbury identity (one extra right-hand side per pocket)."""
        if len(gas):
            U = np.zeros((len(rhs), len(gas)))
            for k, gp in enumerate(gas):
                U[(3*gp["v"][:, None] + np.arange(3)).ravel(), k] = gp["a"].ravel()
            Y = self._solve(A, np.asfortranarray(np.column_stack([rhs, U])))
            y, W = Y[:, 0], Y[:, 1:]
            Mk = np.diag(1./np.asarray(gas_c)) + U.T @ W
            return y - W @ np.linalg.solve(Mk, U.T @ y)
        t = time.perf_counter()
        import pypardiso
        if self._pardiso is None:
            # SPD (mtype 2), fixed sparsity: symbolic analysis once, numeric factorization per call
            self._pardiso = pypardiso.PyPardisoSolver(mtype=2)
            self._pardiso.set_statistical_info_off()
            self._pardiso.set_phase(11)
            self._pardiso._call_pardiso(A, rhs)
        s = self._pardiso
        s.set_phase(23)
        x = s._call_pardiso(A, rhs)
        self.stats["solve_s"] += time.perf_counter() - t
        return x

    # ------------------------------------------------------------------ interface update
    def _post_step(self, dt, P):
        x = self.x
        A = self.vertex_area
        half = .5*self.stock.thickness
        # remember per-pair normal loads for next step's friction
        self._pair_lam[:] = 0.
        if len(P["v"]):
            np.add.at(self._pair_lam, (P["v"], P["c"]), np.abs((self._pf*P["n"]).sum(1)))
        self.contact_force = np.zeros((self.n, 3))
        np.add.at(self.contact_force, P["v"], self._pf)
        # contact pressure: max(penalty pressure, rigid sandwich pressure). A label squeezed between two
        # surfaces carries the same normal force through both contacts: the interface pressure is the
        # largest single-pair pressure (the sum counted a sandwiched label twice)
        pen_p = self._pair_lam.max(1)/A
        p = np.maximum(pen_p, self.pressure)
        self.last_pressure = p
        self.p_sustained += (p - self.p_sustained)*(1. - np.exp(-dt/M.WETOUT_TIME))
        bonded = self.bond >= 0
        # 1) bond formation: adhesive face touching a collider (closest such pair per vertex)
        Q = self._contact_pairs(x, reach=5e-5, update_history=False, extended=False)
        ln = self._label_normals(x)
        adh = (ln[Q["v"]]*Q["n"]).sum(1) > 0.
        if PAD_BOND_ADHESIVE_SIDE:
            adh &= self._pad_on_adhesive_side(Q, x, ln)
        c_now = np.full(self.n, -1); d_best = np.full(self.n, np.inf)
        n_now = np.zeros((self.n, 3)); l_now = np.zeros((self.n, 3))
        for k in np.nonzero(adh & (Q["d"] < half + 5e-5))[0]:
            i = Q["v"][k]
            if Q["d"][k] < d_best[i]:
                d_best[i], c_now[i], n_now[i], l_now[i] = Q["d"][k], Q["c"][k], Q["n"][k], Q["loc"][k]
        # an interface that debonded cannot re-form while still in contact (a released front under shear
        # would otherwise heal every step and a lap joint could never slide off): a released vertex may
        # bond again only after it has separated from every surface (re-application after lifting)
        self.rebond_ok |= c_now < 0
        touch = (~bonded) & (c_now >= 0) & self.rebond_ok
        # trapped air keeps an enclosed pocket's surfaces apart -- where the sheet stands off. A pocket vertex within the
        # adhesive contact band (the solver's own contact criterion, as for every other vertex) is in contact: it may
        # bond, and its share of the pocket's air goes to its unbonded neighbours, so the air gathers where the sheet
        # stands off (a bubble); a pocket's last vertex keeps its air unless that is no more than a floor-gap film at
        # p_atm (the floor is the solver's regularization, not air). Barring every gas-holding vertex from bonding
        # fails: a flat 84 mm2 region enclosed at contact never bonded under a 5 N press; 0.014 mm3 of air enclosed
        # ahead of the wipe-on's roller spread as a floor-thick film over 125 mm2; and air films of 1.6-1.8 um (below
        # the kraft's roughness) held 345 and 196 mm2 of the robot chain's label off the carton through the whole hold
        # and rub. Barring only the vertices at the gap floor does not reach the last two
        if GAS_BOND_AT_FLOOR:
            free_nb = (self.adj @ (~bonded).astype(float)) > 0
            touch &= (self.gas <= PATM*GAP_FLOOR*self.vertex_area*(1. + 1e-6)) | free_nb   # re-checked per vertex below
        else:
            touch &= self.gas <= 0
        film = PATM*GAP_FLOOR*self.vertex_area*(1. + 1e-6)
        for i in np.nonzero(touch)[0]:
            if GAS_BOND_AT_FLOOR and self.gas[i] > film[i]:
                nb = self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i+1]]
                if not (self.bond[nb] < 0).any():
                    continue                 # the pocket's last vertex: its air has nowhere to go, it stays
            col = self.colliders[c_now[i]]
            sidx = col.surface_at(l_now[i:i+1], n_now[i:i+1])[0]
            surf = col.surface_list()[sidx]
            if surf.gamma90 <= 0:
                continue
            self.bond[i] = c_now[i]
            self.bond_surface[i] = sidx
            n_loc = n_now[i] @ col.R
            self.anchor[i] = l_now[i] + n_loc*half
            self.anchor_n[i] = n_loc
            self.bond_time[i] = 0.
            self.damage[i] = 0.
            if self.gas[i] > 0:          # gas squeezed out of this vertex stays in the pocket
                self._redistribute_gas(i)
        bonded = self.bond >= 0
        # 2) wet-out: contact fraction from the SUSTAINED pressure (first-order filter, M.WETOUT_TIME) and
        # roughness; the contact formed does not un-form on unloading (running maximum)
        for c in np.unique(self.bond[bonded]):
            col = self.colliders[c]
            sel = np.nonzero(self.bond == c)[0]
            surfs = col.surface_list()
            pr = np.array([surfs[k].p_rough for k in self.bond_surface[sel]])
            target = 1. - np.exp(-(self.p_sustained[sel] + M.P_TACK)/pr)
            self.phi[sel] = np.maximum(self.phi[sel], target)
        self.bond_time[bonded] += dt
        # 3) debonding: VCCT energy release at front nodes, rate law, damage
        self._last_P = P
        self._debond(dt)
        # 4) permanent set
        theta = self._hinge_angles(x)[0]
        over = theta - self.theta_rest
        y = self.theta_yield
        grow = np.abs(over) > y
        self.theta_rest[grow] = theta[grow] - np.sign(over[grow])*y[grow]
        # 5) tearing: orthotropic Tsai-Hill index on the membrane force per width in material axes
        # (MD, CD, shear), tensile normal components only (paper buckles, it does not crush); a triangle
        # reaching 1 fails and stops carrying membrane and bending load (element erosion), so a torn
        # sheet actually separates and its new free edges vent pockets and can peel like label edges
        Sn = self._membrane_force_per_width
        s1 = .5*(Sn[:, 0] + Sn[:, 1]) + np.sqrt((.5*(Sn[:, 0] - Sn[:, 1]))**2 + Sn[:, 2]**2)
        live = self.alive.astype(float)
        wsum = self.tear_avg @ live
        Savg = (self.tear_avg @ (Sn*live[:, None]))/np.maximum(wsum, 1e-12)[:, None]
        self.tear_index = tear_index(Savg, self.stock)
        fail = self.alive & (self.tear_index >= 1.)
        if fail.any():
            self.alive[fail] = False
            live = self.alive[self.hinge_tris].all(1)
            self.k_hinge = self.k_hinge0*live
            self.boundary[self.tris[fail].ravel()] = True
            va = np.zeros(self.n, bool); va[self.tris[self.alive].ravel()] = True
            self.vertex_alive = va
            self.bond[~va] = -1; self.gas[~va] = 0.
            self.events.append(dict(t=self.time, event="tear" if self.torn else "tear_onset",
                                    triangles=int(fail.sum()), total_failed=int((~self.alive).sum()),
                                    max_tear_index=float(self.tear_index[fail].max()),
                                    max_membrane_force_n_per_m=float(s1[fail].max())))
            self.torn = True
        self.max_membrane_force = float(s1[self.alive].max()) if self.alive.any() else 0.
        # 6) trapped air
        self._pockets()
        # 7) reactions on rigid bodies (sandwiched compressive contact is carried by the rigid engine)
        self._reactions(x)

    def _debond(self, dt):
        cv = vcct_factor(self.h)
        bonded = self.bond >= 0
        self.G[:] = 0.
        self.front_speed[:] = 0.
        self.front_read = {}           # front node -> (T N/m, cos theta, reach m) of the energy reading (diagnostics)
        if not bonded.any():
            return
        free = ~bonded
        # front nodes: bonded with at least one unbonded neighbour
        nfree = self.adj @ free.astype(float)
        front = bonded & (nfree > 0)
        x = self.x
        A = self.vertex_area
        F = self._bf
        rows = np.nonzero(front)[0]
        self.G_II = np.zeros(self.n)
        if len(rows) and FRONT_G == "energy":
            ce = energy_factor(self.h)
            ce180 = energy_factor_180(self.h)
            # Rivlin-Kendall energy balance at each front node: the free strip behind the front carries
            # tension T (membrane force per width along the edge to the free neighbour) at peel angle
            # theta to the substrate, and advancing the front by dA releases
            #     G = T (1 - cos theta) + T^2 / (2 E t)
            # (Kendall 1975: the load's work less the strip's stored stretch). Exact for steady elastic
            # peel at any angle (0 deg: the lap-shear / mode II limit, 90 deg: G = T), taken from the
            # membrane state behind the front, not from node forces at the tip, so it needs no mesh
            # calibration factor. Bending-driven release is the moment term below.
            St = self._membrane_force_per_width
            indptr, indices = self.adj.indptr, self.adj.indices
            vt_ptr, vt_idx = self.vt.indptr, self.vt.indices
            s_ = self.stock
            G = np.zeros(len(rows)); GII = np.zeros(len(rows))
            for k, i in enumerate(rows):
                nbs = indices[indptr[i]:indptr[i+1]]
                nf, nbd = nbs[free[nbs]], nbs[bonded[nbs]]
                col = self.colliders[self.bond[i]]
                _, na = col.anchor_world(self.anchor[i:i+1], self.anchor_n[i:i+1])
                nrm = na[0]
                # outward in-plane direction: away from the bonded neighbours (into the peeled region)
                ref = (x[nbd] - x[i]).mean(0) if len(nbd) else -(x[nf] - x[i]).mean(0)
                ref = ref - (ref @ nrm)*nrm
                rn = np.linalg.norm(ref)
                if rn < 1e-12:
                    continue
                d_out = -ref/rn
                # strip direction(s) (material). Tension and angle are read in the free strip outside the
                # peel's bending boundary layer (3 l, l = sqrt(D/T), clipped to 2..5 elements): Kendall's
                # balance holds for the far-field strip tension and angle, while triangles straddling the
                # front carry bending-induced membrane stretch.
                # FRONT_DIRECTION "loaded": every free 1-ring direction is read in its own corridor (one
                # element wide) and the front advances where the balance releases the most energy. "mean":
                # one direction, the mean of the free 1-ring, read over a disc. The disc and the mean direction mixed a peeled arm with unloaded sheet
                # lying flat beside a narrow bond patch (a fingertip pad): at the patch corners the reading
                # fell to 4-550 J/m^2 where the arm carried 1.6-2.7 kN/m at ~90 deg, the corners held, and
                # the paper tore there.
                if FRONT_DIRECTION == "loaded":
                    dirs = self.X[nf, :2] - self.X[i, :2]
                    dirs = dirs/np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
                    corridor = True
                else:
                    m = (self.X[nf, :2] - self.X[i, :2]).mean(0)
                    mn = np.linalg.norm(m)
                    if mn < 1e-12:
                        continue
                    dirs = (m/mn)[None]
                    corridor = False
                best = (0., 0., 0.)
                self.front_read[i] = (0., 1., 0.)
                for m in dirs:
                    Db = s_.D_md*m[0]**2 + s_.D_cd*m[1]**2
                    mp = np.array([-m[1], m[0]])
                    T = 0.
                    for _ in range(2):                  # reach depends on T: one fixed-point pass
                        reach = float(np.clip(3.*np.sqrt(Db/max(T, 1e-3)), 2.*self.h, 5.*self.h))
                        ct_idx = np.asarray(self._kdC.query_ball_point(self.X[i, :2], reach), int)
                        if corridor:
                            dv = self._cenX[ct_idx] - self.X[i, :2]
                            ct_idx = ct_idx[(dv @ m > 0.) & (np.abs(dv @ mp) <= self.h)]
                        ct_idx = ct_idx[self.alive[ct_idx] & free[self.tris[ct_idx]].all(1)]
                        if not len(ct_idx) and not corridor:
                            ct_idx = vt_idx[vt_ptr[i]:vt_ptr[i+1]]
                            ct_idx = ct_idx[self.alive[ct_idx] & free[self.tris[ct_idx]].any(1)]
                        if not len(ct_idx):
                            break
                        Sv = St[ct_idx].mean(0)
                        S2 = np.array([[Sv[0], Sv[2]], [Sv[2], Sv[1]]])
                        T = max(m @ S2 @ m, 0.)
                    if not len(ct_idx):
                        continue
                    zone = np.asarray(self._kdX.query_ball_point(self.X[i, :2], reach + 1e-9), int)
                    if corridor:
                        dv = self.X[zone, :2] - self.X[i, :2]
                        zone = zone[(dv @ m > 0.) & (np.abs(dv @ mp) <= self.h)]
                    zone = zone[free[zone]]
                    if not len(zone):
                        zone = nf
                    D = (x[zone] - x[i]).mean(0)
                    ct = float(np.cos(np.arctan2(D @ nrm, D @ d_out)))
                    Et = (s_.E_md*m[0]**2 + s_.E_cd*m[1]**2)*s_.t_face
                    # the discretisation artifact is the strip released per row, h (1 - cos theta): the
                    # correction scales with it, none at 0 deg, c90 at 90 deg; above 90 deg it runs linearly
                    # in (1 - cos theta) to its own factor c180, calibrated against the elastic Kendall
                    # solution at 180 deg (ENERGY_CALIBRATION_180)
                    u = 1. - ct
                    cth = 1. - (1. - ce)*u if u <= 1. else ce + (ce180 - ce)*(u - 1.)
                    gii = cth*T*T/(2.*Et)
                    g = cth*T*(1. - ct) + gii
                    key = g
                    if key > best[2]:
                        best = (g, gii, key)
                        self.front_read[i] = (T, ct, reach)     # diagnostics: the chosen direction's reading
                G[k], GII[k] = best[0], best[1]
            self.G[rows] = G
            self.G_II[rows] = GII
            if FRONT_PATCH_BALANCE:
                self._patch_balance(front, bonded, F, ce, ce180)
        elif len(rows):
            # (comparison path) virtual crack closure at the node: G_I = c_V F_n w / (2A) from the opening
            # of the free neighbours, G_II = c_V F_t s_t / (2A) from the node's tangential slip
            indptr, indices = self.adj.indptr, self.adj.indices
            G = np.zeros(len(rows))
            for k, i in enumerate(rows):
                nb = indices[indptr[i]:indptr[i+1]]
                nb = nb[free[nb]]
                col = self.colliders[self.bond[i]]
                pa, na = col.anchor_world(self.anchor[i:i+1], self.anchor_n[i:i+1])
                p, nrm = pa[0], na[0]
                opening = np.maximum(((x[nb] - p) @ nrm), 0.).mean()
                Fn = max(-(F[i] @ nrm), 0.)
                G[k] = cv*Fn*opening/(2.*A[i])
                dvec = x[i] - p
                st = dvec - (dvec @ nrm)*nrm
                Ft = F[i] - (F[i] @ nrm)*nrm
                self.G_II[i] = cv*np.linalg.norm(Ft)*np.linalg.norm(st)/(2.*A[i])
            self.G[rows] = G + self.G_II[rows]
        # crack initiation at a fully bonded label edge: bending-moment energy release rate
        # G = m^2/(2D), m = hinge moment per unit length (= D * kappa for a bent strip; for a label
        # wrapped on radius R this is the flagging driving force D/(2R^2)).
        # At fronts the larger of the strip-tension and bending-moment estimates is used: the energy
        # balance captures tension-driven peel, the moment term bending-driven release (peel plates, flags).
        edge = bonded & (self.boundary | front)
        if edge.any():
            theta = self._hinge_angles(x)[0]
            m = 2*self.k_hinge*(theta - self.theta_rest)/self.hinge_len
            gm = m*m/(2*self.hinge_D)
            Gv = np.zeros(self.n)
            for k in range(2):                       # hinge edge endpoints
                np.maximum.at(Gv, self.hinge[:, k], gm)
            if EDGE_WING_BENDING:
                # a bonded strip forced to curve (a label following a liner round a peel bar) stores D kappa^2/2
                # per area; advancing the front lets that strip spring back, which is the flagging release. That
                # energy sits in the first hinge BEHIND the front (all four of its vertices bonded), whose wing
                # vertices are the front's nodes -- at a label edge the edge row itself has no hinge parallel to
                # it (its in-row edges are boundary edges). An endpoints-only rule never reads it: the leading
                # edge of a label forced round a peel bar did not feel its bend (at 1 % yield strain 17 J/m^2
                # available against the liner's 1.5 J/m^2 released nothing), and once a front formed each row
                # was carried round the edge past yield before it let go, so the presented label curled. Bonded
                # hinges on a flat substrate
                # are flat and add nothing: ordinary peels are unchanged.
                hb = bonded[self.hinge].all(1)
                Gw = np.zeros(self.n)
                for k in (2, 3):                     # hinge wing vertices
                    np.maximum.at(Gw, self.hinge[hb, k], gm[hb])
                Gv = np.maximum(Gv, Gw)
            if SUBSTRATE_CURVATURE_RELEASE:
                # a bonded strip on a curved surface is held to that surface's curvature right up to the front
                # (the stiff adhesive's process zone, (4D/k)^(1/4) ~ 0.5 mm, is shorter than an element): the
                # front releases D kappa^2 / 2 per area, capped by the facestock's plastic moment
                # (kappa <= PLASTIC_SHAPE_FACTOR * 2 eps_y / t). Reading it from the hinges alone waited for a
                # 1-3 mm element to bend, and at a 1 mm peel bar rows were carried round past yield before they
                # released (the dispensed label curled)
                s_ = self.stock
                kcap = PLASTIC_SHAPE_FACTOR*2.*s_.yield_strain/s_.t_face
                ev = np.flatnonzero(edge)
                for c in np.unique(self.bond[ev]):
                    sel = ev[self.bond[ev] == c]
                    k, dw = self.colliders[c].substrate_curvature(self.anchor[sel])
                    on = k > 0
                    if not on.any():
                        continue
                    for i, kk, d in zip(sel[on], k[on], dw[on]):
                        # the curvature's direction in the label's material frame (mean deformation gradient)
                        tr = self.vt.indices[self.vt.indptr[i]:self.vt.indptr[i+1]]
                        F = np.matmul(x[self.tris[tr]].transpose(0, 2, 1), self.b[tr]).mean(0)   # (3,2)
                        m = np.linalg.lstsq(F, d, rcond=None)[0]
                        mn = np.linalg.norm(m)
                        m = m/mn if mn > 1e-12 else np.array([1., 0.])
                        Dd = (s_.D_md*m[0]**2 + s_.D_cd*m[1]**2)*C_BEND/C_BEND_REF
                        Gv[i] = max(Gv[i], .5*Dd*min(kk, kcap)**2)
            self.G[edge] = np.maximum(self.G[edge], Gv[edge])
        cand = np.nonzero(bonded & (self.G > 0))[0]
        for i in cand:
            col = self.colliders[self.bond[i]]
            surf = col.surface_list()[self.bond_surface[i]]
            gd = 1. + surf.alpha_dwell*math.log10(max(self.bond_time[i], 1.)/M.DWELL_REF)
            gd = max(gd, 0.2)
            scale = self.phi[i]*gd
            g_ref = surf.gamma90
            excess = self.G[i]/max(scale, 1e-9) - M.W0
            if excess <= 0 or g_ref <= M.W0:
                if g_ref <= M.W0 and self.G[i] > 0:
                    self.front_speed[i] = V_CAP
                else:
                    continue
            else:
                self.front_speed[i] = min(surf.v_ref*(excess/(g_ref - M.W0))**(1./M.RATE_EXPONENT), V_CAP)
            self.damage[i] += self.front_speed[i]*dt/self.spacing[i]
        release = bonded & (self.damage >= 1.)
        # a bond stretched beyond BOND_MAX_OPENING (10 mm) is gone: a nearly fully damaged bond left with ~zero
        # stiffness that the front law never released (one "liner-bonded" node stayed counted
        # through a 150 mm carry, carrying no load); set above the discrete front's legitimate opening
        for c in np.unique(self.bond[bonded]):
            sel = np.flatnonzero(self.bond == c)
            pa, _ = self.colliders[c].anchor_world(self.anchor[sel], self.anchor_n[sel])
            over = np.linalg.norm(self.x[sel] - pa, axis=1) > BOND_MAX_OPENING
            if over.any():
                release[sel[over]] = True
                self.stats["overstretched_releases"] = self.stats.get("overstretched_releases", 0) + int(over.sum())
        if release.any():
            self.bond[release] = -1
            self.rebond_ok[release] = False
            self.phi[release] = 0.
            self.damage[release] = 0.
            self.bond_time[release] = 0.

    def _patch_balance(self, front, bonded, F, ce, ce180):
        """Kendall's balance for each bonded patch as a whole: the pull the rest of the label exerts on the patch
        (minus every external force on it -- its bond forces and all contact forces on the patch and its 1-ring,
        which holds the fulcrum of a sub-element fold) per length of its loaded front, at the angle of that pull
        to the substrate. Exact for a patch of any shape and width. The front's local strip reading (corridor, 2-5
        elements out) assumes a uniform strip behind the front: at a bond patch narrower than the sheet -- a
        fingertip pad in a 101.6 mm label -- the load funnels into the patch within a few mm and the just-released
        row lies flat on the pad, so the strip read 0-60 N/m at ~0 deg where the patch front carried 100-700 N/m
        normal to the pad; the front stalled and the paper tore. Applied only where the loaded front is narrower
        than the sheet around it (PATCH_NARROW) or the patch has no interior node, as a lower bound on G, to the
        loaded front nodes only, with the same discretisation factor c_E(theta) as the strip reading. A uniform
        strip reads the same either way."""
        from scipy.sparse.csgraph import connected_components
        P = getattr(self, "_last_P", None)
        cf = np.zeros((self.n, 3))
        if P is not None and len(P["v"]) and len(self._pf) == len(P["v"]):
            np.add.at(cf, P["v"], self._pf)
        s_ = self.stock
        Et = s_.E_md*s_.t_face
        for c in np.unique(self.bond[bonded]):
            idx = np.flatnonzero(self.bond == c)
            ncomp, lab = connected_components(self.adj[idx][:, idx], directed=False)
            col = self.colliders[c]
            for comp in range(ncomp):
                isl = idx[lab == comp]
                fr = isl[front[isl]]
                if not len(fr):
                    continue
                ring = np.unique(np.r_[isl, self.adj[isl].indices])
                # the two extensions below read a patch loaded THROUGH THE SHEET; a presser on another rigid body
                # (a rub finger, a roller, the other hand) touching the patch or its rim loads the bond directly,
                # and read there the extensions tore the label in the assisted applications' rub. The patch's own
                # finger (pad and link
                # of one rigid body) is its substrate, not a presser.
                ext_ok = True
                if PATCH_EXT_UNTOUCHED and P is not None and len(P["v"]):
                    bc = self.colliders[c].body
                    foreign = np.array([k != c and self.colliders[k].role in ("pad", "tool", "applicator", "robot")
                                        and not (bc >= 0 and self.colliders[k].body == bc) for k in P["c"]], bool)
                    ext_ok = not np.any(np.isin(P["v"], ring) & foreign)
                pull = -(F[isl].sum(0) + cf[ring].sum(0))
                pn = float(np.linalg.norm(pull))
                if pn < 1e-6:
                    continue
                _, na = col.anchor_world(self.anchor[isl], self.anchor_n[isl])
                nrm = na.mean(0); nrm /= max(np.linalg.norm(nrm), 1e-12)
                if pull @ nrm <= 0. and not (PATCH_PRESSED_SHEAR and ext_ok):
                    continue
                if pull @ nrm <= 0.:
                    # pressed onto the substrate: no peel, but an in-plane pull still shears the patch off (Kendall
                    # at 0 deg, G = T^2/2Et). Skipping it left a re-tacked row on the liner under 3 kN/m of shear
                    pull = pull - (pull @ nrm)*nrm
                    pn = float(np.linalg.norm(pull))
                    if pn < 1e-6:
                        continue
                bfm = np.linalg.norm(F[fr], axis=1)
                load = fr[bfm >= .5*bfm.max()]              # the loaded part of the front
                out = self.x[load].mean(0) - self.x[isl].mean(0)
                out = out - (out @ nrm)*nrm
                pin = pull - (pull @ nrm)*nrm
                if np.linalg.norm(out) < 1e-9:
                    out = pin
                on = np.linalg.norm(out)
                if on < 1e-12:
                    continue
                out /= on
                across = np.cross(nrm, out)
                # the front line: every front node of the patch on its loaded side (within one element of the
                # loaded front's position along the peel direction), not only the most loaded ones -- a uniform
                # strip's front carries its load unevenly across the width
                line = fr[np.abs(self.x[fr] @ out - float(self.x[load].mean(0) @ out)) <= self.h]
                load = line if len(line) else load
                L = float(np.ptp(self.x[load] @ across)) + self.h
                # only where the front is narrower than the sheet around it: a front as wide as the sheet is a
                # uniform strip, which the strip reading describes (and the calibration was built on); read there
                # through the patch's instantaneous pull, the release followed the sawtooth's peaks and moved the
                # independent peel checks (0.5 mm/s -21 %, PET/PP -11 %)
                ahead = self.x @ out
                band = np.abs(ahead - float(self.x[load].mean(0) @ out)) <= self.h
                W = float(np.ptp(self.x[band] @ across)) + self.h if band.any() else L
                # ... or where the patch has no interior: every bonded node borders released label (a single
                # re-tacked row), so the strip reading has no bonded strip behind the front and reads ~0 whatever
                # the load (in a pinch test the released tab fell back on the liner, one full-width row re-tacked
                # and read G <= 0.05 J/m^2 under 11 N per node until the paper tore)
                # narrowness is the ISLAND's width across the pull, not the loaded line's: on a wide bond with an
                # oblique or curved front (a carton bond under a roller and the hand's tack pull) only ~20 mm of front
                # carried half the peak bond force, the wide bond was taken for a fingertip patch and its whole pull
                # was put on 20 mm (G 0.3 -> 40 J/m^2), releasing rows the roller had laid and trapping air in every
                # assisted wipe-on
                Li = float(np.ptp(self.x[isl] @ across)) + self.h
                if Li > PATCH_NARROW*W and not (PATCH_NO_INTERIOR and ext_ok and len(fr) == len(isl)):
                    continue
                T = pn/L
                ct = float(pin @ out)/pn                     # >0: pulled away over the front; <0: back over the patch
                u = 1. - ct
                cth = 1. - (1. - ce)*u if u <= 1. else ce + (ce180 - ce)*(u - 1.)
                g = cth*T*(1. - ct) + cth*T*T/(2.*Et)
                up = self.G[load] < g
                self.G[load[up]] = g
                for i in load[up]:
                    self.front_read[i] = (T, ct, -L)          # diagnostics: patch reading (negative reach = patch)

    def _redistribute_gas(self, i):
        g = self.gas[i]
        self.gas[i] = 0.
        if GAS_BOND_AT_FLOOR and GAS_SEALED_FILM:
            g = max(g - PATM*GAP_FLOOR*self.vertex_area[i], 0.)      # its floor film goes with the bonding vertex
        nb = self.adj.indices[self.adj.indptr[i]:self.adj.indptr[i+1]]
        nb = nb[self.bond[nb] < 0]
        if len(nb):
            self.gas[nb] += g/len(nb)

    def _pocket_volume(self, pk, xv):
        """Gas volume of pocket pk for its vertices' positions xv against its substrate's current pose (as _pockets)."""
        v = pk["vertices"]
        col = next((c for c in self.colliders if c.name == pk["collider"]), None)
        if col is None:
            return pk["volume_m3"]
        d, _, _ = col.sdf(xv)
        gap = np.maximum(d - .5*self.stock.thickness, GAP_FLOOR)
        return float((gap*self.vertex_area[v]).sum())

    def _pockets(self):
        """Connected unbonded regions: enclosed ones keep their gas amount (isothermal)."""
        free = self.bond < 0
        self.pocket_pressure[:] = 0.
        self.pocket_stiffness[:] = 0.
        self.pockets = []
        if free.all() or not free.any():
            self.gas[:] = 0.
            return
        sub = self.adj[free][:, free]
        ncomp, lab = sp.csgraph.connected_components(sub, directed=False)
        idx = np.nonzero(free)[0]
        x = self.x
        A = self.vertex_area
        patm = PATM
        for k in range(ncomp):
            v = idx[lab == k]
            if self.boundary[v].any():
                self.gas[v] = 0.       # vented to the label edge
                continue
            # substrate = the collider most of the surrounding bonded ring is bonded to
            ring = np.unique(self.adj[v].indices)
            ring = ring[self.bond[ring] >= 0]
            if not len(ring):
                continue
            c = np.bincount(self.bond[ring]).argmax()
            col = self.colliders[c]
            d, _, _ = col.sdf(x[v])
            gap = np.maximum(d - .5*self.stock.thickness, GAP_FLOOR)
            vol = (gap*A[v]).sum()
            n_gas = self.gas[v].sum()
            # a new pocket holds atmospheric air in its volume, the floor film included: sealed, it resists being
            # lifted (a vacuum opens), and it keeps the film as it is lifted. Given only the air ABOVE the floor (the
            # contact-band rule alone), a pocket enclosed at contact held none, had no gas term (nothing resisted a
            # lift) and, holding none, was taken for new again at every step: lifted by 30.7 um, a 3 x 3 pocket
            # gained 1.02 mm3 of air from nowhere and kept it as a 21 um bubble. The film is not air: a vertex that bonds takes its film with it
            # (_redistribute_gas), so a pocket's films never gather into a bubble
            if n_gas <= 0:
                n_gas = (patm*float((np.maximum(d - .5*self.stock.thickness - GAP_FLOOR, 0.)*A[v]).sum())
                         if GAS_BOND_AT_FLOOR and not GAS_SEALED_FILM else patm*vol)
            # pressure equalizes within a connected pocket far faster than a step: keep each vertex's
            # share of the gas proportional to its current local volume, so a later split divides
            # the gas by volume (per-vertex shares frozen at enclosure left split-off pockets empty)
            self.gas[v] = n_gas*(gap*A[v])/vol
            pk = n_gas/vol if n_gas > 0 else patm      # an air-free pocket reads 0, not vacuum
            self.pocket_pressure[v] = pk - patm
            self.pocket_stiffness[v] = pk*A[v]**2/vol      # reporting only (the solve uses the exact pocket energy)
            self.pockets.append(dict(vertices=v, area_m2=float(A[v].sum()), volume_m3=float(vol),
                                     overpressure_pa=float(pk - patm), collider=col.name))

    def _reactions(self, x):
        """Force/torque the label exerts on each collider body (about the collider origin).
        Compressive contact of sandwiched vertices is excluded (carried by the rigid contact)."""
        self.reactions = {}
        mags = {}
        P = self._P
        sandwich = self.pressure > 0
        for c, col in enumerate(self.colliders):
            mk = P["c"] == c
            keep = mk & ~sandwich[P["v"]]
            fc_pts = [x[P["v"][keep]]]
            fc = [-self._pf[keep]]
            b = self.bond == c
            fc_pts.append(x[b]); fc.append(-self._bf[b])
            if mk.any() and sandwich[P["v"][mk]].any():     # tangential (friction) part of sandwiched pairs
                sw = mk & sandwich[P["v"]]
                nrm = P["n"][sw]
                ft = self._pf[sw] - (self._pf[sw]*nrm).sum(1, keepdims=True)*nrm
                fc_pts.append(x[P["v"][sw]]); fc.append(-ft)
            pts, fs = np.concatenate(fc_pts), np.concatenate(fc)
            f = fs.sum(0)
            tau = np.cross(pts - col.p, fs).sum(0) if len(pts) else np.zeros(3)
            self.reactions[col.name] = (f, tau, col.body)
            mags[col.name] = float(np.linalg.norm(fs, axis=1).sum()) if len(fs) else 0.
        # the pocket's air pushes its substrate with the force it exerts on the label, reversed (the label's contact and
        # bond forces alone handed the substrate the label's pull and never the air's push: a pocket at +2.0 kPa over
        # 84 mm2 lifted its plate with 168 mN where the label weighs 3.3 mN; the pinned MPa pockets made the carton hop
        # by 1.1 mm through this)
        if GAS_REACTION:
            for gp in self._gas:
                if gp.get("col") not in self.reactions:
                    continue
                v = gp["v"]
                V = gp["V0"] + (gp["a"]*(x[v] - gp["xb"])).sum()
                if V <= 0.:
                    continue
                fs = -(gp["n"]/V - PATM)*gp["a"]
                col = next(c for c in self.colliders if c.name == gp["col"])
                f, tau, body = self.reactions[gp["col"]]
                self.reactions[gp["col"]] = (f + fs.sum(0), tau + np.cross(x[v] - col.p, fs).sum(0), body)
        # momentum-consistent reactions: the label cannot push its surroundings with a net force it does not have.
        # The Newton solve stops at a 1e-6 m update; with bonded adhesive-layer stiffness ~1.35e5 N/m per node the
        # residual left over hundreds of bonded nodes summed into 0.8-5.5 N one-step kicks on an item the label
        # rested on. That residual (the net gradient,
        # _residual_sum) is removed from the reported reactions in proportion to each collider's share of the
        # contact/bond load; the label's own motion is unchanged.
        R = getattr(self, "_residual_sum", None)
        tot = sum(mags.values())
        if R is not None and tot > 0 and np.all(np.isfinite(R)):
            for name, (f, tau, body) in self.reactions.items():
                self.reactions[name] = (f - R*mags[name]/tot, tau, body)
            self.last_residual_n = float(np.linalg.norm(R))

    # ------------------------------------------------------------------ diagnostics
    def summary(self):
        A = self.vertex_area
        bonded = self.bond >= 0
        by = {}
        for c, col in enumerate(self.colliders):
            s = self.bond == c
            if s.any():
                by[col.name] = dict(bonded_area_fraction=float((self.phi[s]*A[s]).sum()/A.sum()),
                                    vertices=int(s.sum()))
        return dict(time_s=self.time, bonded_vertices=int(bonded.sum()),
                    bonded_area_fraction=float((self.phi*A*bonded).sum()/A.sum()),
                    pockets=[{k: v for k, v in p.items() if k != "vertices"} for p in getattr(self, "pockets", [])],
                    by_collider=by, torn=self.torn,
                    max_membrane_force_n_per_m=getattr(self, "max_membrane_force", 0.),
                    kinetic_energy_j=float(.5*(self.mass*(self.v*self.v).sum(1)).sum()),
                    stats=dict(self.stats))


PATM = 101325.       # Pa
SELF_CONTACT = True
SELF_CONTACT_REACH = 1.5e-3          # m: pairs are collected within this gap (lagged, per step)
SELF_CONTACT_STIFFNESS = 1.5e10      # Pa/m: same normal stiffness as the rigid-surface contacts
SELF_CONTACT_SKIN = 2e-3             # m: Verlet margin of the self-contact candidate list
FRONT_G = "energy"   # front energy release rate: "energy" (Rivlin-Kendall strip balance) | "vcct" (comparison)
FRONT_DIRECTION = "mean"     # energy front: "mean" = mean free-neighbour direction read over a disc (calibrated,
                             # validated across rate and angle) | "loaded" = each free-neighbour direction in its own
                             # corridor, largest G (tested and rejected: a load-
                             # dependent bias, 0.5 mm/s peel -25 %; kept for comparison)
EDGE_WING_BENDING = True     # edge and front nodes read the stored bending of the first fully bonded hinge behind them
FRONT_PATCH_BALANCE = True   # each bonded patch also reads Kendall's balance for its resultant pull (narrow patches)
PATCH_NARROW = .8            # ... when the patch, measured across its pull, is narrower than this fraction of the sheet there
# Two extensions tested and NOT adopted (kept switchable): they release a patch's last, interior-less or
# pressed row (a pinch test's re-tacked tab; the last 3 nodes of the robot handoff's fingertip patch), but
# on, ungated, each made the assisted wipe-on's fingertip rub trap air or tear, and gated to patches no foreign
# presser touches, the helper-only wipe trapped 180 mm^2 and the robot-handoff wipe tore
PATCH_NO_INTERIOR = False    # the balance also for any patch with no interior node
PATCH_PRESSED_SHEAR = False  # a patch pressed on but pulled in-plane reads Kendall's 0 deg term (lap shear)
PATCH_EXT_UNTOUCHED = True   # (if on) the extensions only where no presser on another rigid body touches the patch
SUBSTRATE_CURVATURE_RELEASE = True   # a front on a curved surface (peel bar, tube) releases D kappa_s^2/2 (plastic cap)
# m: a bond stretched this far carries nothing any adhesive could: released. Not a peel law -- a discrete
# front node legitimately opens past 1 mm on the 2-3 mm meshes before the energy balance releases it (a 1 mm
# cap moved the calibrated 90 deg peel from 386 to 352 N/m), so the cap sits above 3x the largest element; it
# only removes nearly fully damaged bonds dragged far from any front (one carried 150 mm)
BOND_MAX_OPENING = 10e-3
PAD_COMPRESSION_DEPTH = 2e-3   # rubber fingertip: depth behind the pinching face treated as compression
PAD_BOND_ADHESIVE_SIDE = True  # a pad bonds the label only when the pad lies on its adhesive side (LabelSim._pad_on_adhesive_side)
TEAR_LENGTH = 2e-3             # average-stress characteristic length (m): fibre-network process scale,
                               # of the order of paper fibre length (1-3 mm); estimate


def tear_index(Sn, stock):
    """Tsai-Hill failure index of membrane forces per width (N/m, Voigt MD/CD/shear), tension only."""
    X1, X2, S12 = stock.strength_md, stock.strength_cd, stock.strength_shear
    s11, s22 = np.maximum(Sn[:, 0], 0.), np.maximum(Sn[:, 1], 0.)
    return (s11/X1)**2 - s11*s22/X1**2 + (s22/X2)**2 + (Sn[:, 2]/S12)**2


# Crack-closure constant c_V per label mesh size. The discrete front's energy-release estimate is
# mesh dependent (at fixed c_V the calibration peel rises +16 % at 2 mm and +29 % at 1.5 mm),
# so c_V is calibrated for every mesh size the environment uses, each
# against the same datasheet point (3M 7818, 90 deg on steel, 5 mm/s = 380 N/m):
#   3 mm -> 0.90 (389.5 N/m), 2 mm -> 1.044 (373.5 N/m), 1.5 mm -> 1.16 (388.9 N/m).
VCCT_CALIBRATION = ((1.5e-3, 1.16), (2e-3, 1.044), (3e-3, .90))
VCCT_FACTOR = None    # explicit override (peel bench --factor); None = calibration table


NEWTON_MAX = 32            # adaptive continuation limit for unconverged steps (base cap: newton_iters)
UNRESOLVED_M = 1e-5        # a step whose last Newton update still exceeds 10 um (0.1 label thickness)
                           # after NEWTON_MAX iterations is counted as unresolved (stats, step info)
ENERGY_FACTOR = None        # override (probes); None -> ENERGY_CALIBRATION
# Discretization factor of the energy-balance front at mesh size h: a node row releases as a whole,
# freeing h (1 - cos theta) of strip at once, which the continuum peel never does; the mean peel force
# under-reads Gamma by a mesh-dependent amount that vanishes as h -> 0 (measured 0.65 / 0.88 / 0.94 of
# the rate-law target at 3 / 2 / 1.5 mm, continuous peel kinematics). The factor c90 is calibrated per
# mesh on ONE point, 90 deg PET 7818 on steel at 5 mm/s = 380 N/m (3M datasheet), and applied in
# proportion to the released strip: c(theta) = 1 - (1 - c90)(1 - cos theta) (none at 0 deg, double
# the correction at 180 deg). Every other angle, substrate and speed is a check.
ENERGY_CALIBRATION = ((1.5e-3, .904), (2e-3, .9025), (3e-3, .664))
# (jointly with ENERGY_CALIBRATION_180: 90 deg front nodes read angles on both sides of 90 deg, so the two
# factors are coupled; c90 is calibrated after c180 and both are re-checked)


def energy_factor(h):
    if ENERGY_FACTOR is not None:
        return ENERGY_FACTOR
    hs, cs = zip(*ENERGY_CALIBRATION)
    return float(np.interp(h, hs, cs))


ENERGY_FACTOR_180 = None    # override (probes); None -> ENERGY_CALIBRATION_180
# The same discretisation factor at 180 deg, calibrated per mesh against the ELASTIC Kendall solution
# (F = Gamma/2 at the rate-law speed): numerics only, the adhesive law is not fitted. The extrapolation
# 2 c90 - 1 (used when ENERGY_CALIBRATION_180 is None) reads 85 % above the elastic answer at 3 mm and is
# mesh-dependent. The adhesive law itself stays elastic in angle: see
# materials.py (ANGLE DEPENDENCE) for why the datasheet's 90/180 ratio is bounded, not applied.
ENERGY_CALIBRATION_180 = ((1.5e-3, 1.93), (2e-3, .844), (3e-3, .76))


def energy_factor_180(h):
    if ENERGY_FACTOR_180 is not None:
        return ENERGY_FACTOR_180
    if ENERGY_CALIBRATION_180 is None:
        return 2.*energy_factor(h) - 1.
    hs, cs = zip(*ENERGY_CALIBRATION_180)
    return float(np.interp(h, hs, cs))


def vcct_factor(h):
    if VCCT_FACTOR is not None:
        return VCCT_FACTOR
    hs, cs = zip(*VCCT_CALIBRATION)
    return float(np.interp(h, hs, cs))
V_CAP = 2.0           # m/s cap on the front speed (reported when reached)


def _closest_on_triangle(p, a, b, c):
    """Barycentric weights (n,3) of the closest point on triangles abc to points p (Ericson 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab*ap).sum(1), (ac*ap).sum(1)
    bp = p - b
    d3, d4 = (ab*bp).sum(1), (ac*bp).sum(1)
    cp = p - c
    d5, d6 = (ab*cp).sum(1), (ac*cp).sum(1)
    va = d3*d6 - d5*d4
    vb = d5*d2 - d1*d6
    vc = d1*d4 - d3*d2
    n = len(p)
    w = np.zeros((n, 3))
    den = np.maximum(va + vb + vc, 1e-300)
    v_ = vb/den; w_ = vc/den
    w[:] = np.stack([1 - v_ - w_, v_, w_], 1)                       # interior
    def put(m, vals):
        w[m] = vals[m] if vals.ndim == 2 else vals
    # edges
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1/np.where(d1 - d3 == 0, 1, d1 - d3)
    put(m, np.stack([1 - t, t, 0*t], 1))
    m2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2/np.where(d2 - d6 == 0, 1, d2 - d6)
    put(m2, np.stack([1 - t, 0*t, t], 1))
    m3 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3)/np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6))
    put(m3, np.stack([0*t, 1 - t, t], 1))
    # vertices (take precedence)
    w[(d1 <= 0) & (d2 <= 0)] = (1, 0, 0)
    w[(d3 >= 0) & (d4 <= d3)] = (0, 1, 0)
    w[(d6 >= 0) & (d5 <= d6)] = (0, 0, 1)
    return w

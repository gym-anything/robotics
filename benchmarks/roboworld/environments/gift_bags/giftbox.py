"""Two-piece rigid (set-up) gift box: thin-walled base and telescoping lid as affine rigid bodies, and the
air film between them.

Air film (De Ruiter, Ostergaard, Marker & Jensen, Phys. Rev. Fluids 7, 044101, 2022: the lid's sliding is
controlled by flow in the thin air gap). Lumped, quasi-steady and incompressible (pressure-relaxation time
V/(p_atm C) ~ 0.8 ms << manipulation times):
    overlap L = skirt engagement; per side i, clearance c_i (from the lid's offset in the base frame)
    C = sum_i w_i c_i^3 / (12 mu L)                              laminar slot conductance (m^3/(s Pa))
    Q = A dz/dt                                                  rate of change of the enclosed volume
    dp = Q / C + K rho Q^2 / (2 (sum_i w_i c_i)^2)               viscous + entrance/exit loss
    lid feels -dp A, base +dp A along the base axis (equal and opposite)
In the incremental potential this is the dissipation  E = dt D(dz/dt),  D(v) = 1/2 c1 v^2 + 1/3 c2 |v|^3,
c1 = A^2/C, c2 = K rho A^3 / (2 (sum w c)^2), with the geometry taken at the start of the step.
"""

from dataclasses import dataclass
import math
import numpy as np
from . import ipcworld as W
from . import params as Pm


def tray_boxes(hx, hy, h, wall, bottom, open_up=True):
    """A thin-walled open box as five non-overlapping solid boxes (floor + four walls): list of
    (half extents, centre), origin at the outer bottom centre (opening up), or flipped (a lid, rim at z=0)."""
    wh = (h - bottom)/2
    parts = [((hx, hy, bottom/2), (0., 0., bottom/2)),
             ((hx, wall/2, wh), (0., -hy + wall/2, bottom + wh)), ((hx, wall/2, wh), (0., hy - wall/2, bottom + wh)),
             ((wall/2, hy - wall, wh), (-hx + wall/2, 0., bottom + wh)), ((wall/2, hy - wall, wh), (hx - wall/2, 0., bottom + wh))]
    if not open_up:
        parts = [(half, (c[0], c[1], h - c[2])) for half, c in parts]
    return parts


def boxes_mesh(parts, cell=.012):
    V, F = [], []
    for half, c in parts:
        n = tuple(max(1, int(math.ceil(2*hh/cell))) for hh in half)
        v, f = W.box_mesh(half, center=c, n=n)
        F.append(f + sum(len(x) for x in V)); V.append(v)
    return np.vstack(V), np.vstack(F)


def rigid_from_boxes(name, parts, mass, p, R=None, mu=.35, material="greyboard", cell=.012, **render):
    """Affine body from solid boxes of uniform density scaled to `mass`; exact COM and second moment."""
    vols = np.array([8*a*b*c for (a, b, c), _ in parts])
    rho = mass/vols.sum()
    ms = rho*vols
    cs = np.array([c for _, c in parts])
    com = (ms[:, None]*cs).sum(0)/mass
    S = np.zeros((3, 3))
    for m_i, ((a, b, c), ctr) in zip(ms, parts):
        d = np.asarray(ctr) - com
        S += m_i/3.*np.diag([a*a, b*b, c*c]) + m_i*np.outer(d, d)
    V, F = boxes_mesh(parts, cell)
    R = np.eye(3) if R is None else R
    return W.Body(name, V - com, F, mass=mass, S=S, kind="dynamic", p=np.asarray(p, float) + R @ com, R=R, mu=mu, fmat="greyboard",
                  render=dict(material=material, com_offset=com, parts=parts, **render))


# ----------------------------------------------------------------------------- the gift box of preset B
BOX_HALF = (.090, .065)          # base outer half extents (180 x 130 mm) [A]
BOX_H = .055                     # base height
LID_H = .030                     # lid skirt depth (partial telescope) [A]
WALL = 1.5e-3                    # greyboard 1.5 mm [V]
CLEARANCE = .5e-3                # lid-to-base clearance per side [V qualitative / E value]
BASE_MASS, LID_MASS = .063, .046 # kg [D: areas x (0.975 + 0.128 kg/m2)]
AUTHORED = "authored_180x130x55"  # the box above; the catalogue boxes of the home preset's seeds: params.GIFT_BOXES


@dataclass(frozen=True)
class BoxSpec:
    """One two-piece set-up box: base outer half extents and height, lid skirt depth, board caliper (walls, floor and
    lid top), lid-to-base clearance per side, base and lid masses."""
    name: str
    hx: float
    hy: float
    h: float
    lid_h: float
    wall: float
    clearance: float
    base_mass: float
    lid_mass: float


def box_spec(name=AUTHORED, clearance=CLEARANCE):
    """The authored box (module constants), or a catalogue box of params.GIFT_BOXES: base outer = the catalogue's inside
    size + one wall per side and the floor; lid skirt LID_H [A]; masses = board + covering-paper areal mass x area, the
    lid's acetate window (if any) replacing board over its area [D]."""
    if name == AUTHORED:
        return BoxSpec(AUTHORED, BOX_HALF[0], BOX_HALF[1], BOX_H, LID_H, WALL, clearance, BASE_MASS, LID_MASS)
    c = Pm.GIFT_BOXES[name]
    L, Wd, D = c["inside"]
    w = c["wall"]
    hx, hy, h = L/2 + w, Wd/2 + w, D + w
    lhx, lhy = hx + clearance + w, hy + clearance + w
    areal = Pm.GREYBOARD_RHO*w + Pm.BOX_COVER_RHO_A
    base_mass = areal*(4*hx*hy + 4*(hx + hy)*h)                    # floor + four walls
    lid_mass = areal*(4*lhx*lhy + 4*(lhx + lhy)*LID_H)               # top + skirt
    if c.get("window"):
        a, b, t = c["window"]
        lid_mass += (Pm.ACETATE_RHO*t - areal)*a*b
    return BoxSpec(name, hx, hy, h, LID_H, w, clearance, base_mass, lid_mass)


def make_box(origin, *, clearance=None, yaw=0., box=None):
    """Base resting on the surface at `origin` (outer bottom centre), lid seated (fully down). `box`: a BoxSpec (default:
    the authored box with `clearance`, default CLEARANCE)."""
    if box is None:
        box = box_spec(AUTHORED, CLEARANCE if clearance is None else clearance)
    elif clearance is not None and clearance != box.clearance:
        raise ValueError("give the clearance in the BoxSpec (box_spec(name, clearance))")
    hx, hy, h, lid_h, wall, clearance = box.hx, box.hy, box.h, box.lid_h, box.wall, box.clearance
    lhx, lhy = hx + clearance + wall, hy + clearance + wall
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    o = np.asarray(origin, float) + np.array([0., 0., 5e-5])        # 50 um initial clearance (IPC needs d > dmin)
    base = rigid_from_boxes("gift_base", tray_boxes(hx, hy, h, wall, wall), box.base_mass, o, R, mu=Pm.MU["box_box"],
                            material="giftbox_base", half=(hx, hy, h))
    lid_origin = o + R @ np.array([0., 0., h + wall - lid_h + 5e-5])    # lid top 50 um above the base rim
    lid = rigid_from_boxes("gift_lid", tray_boxes(lhx, lhy, lid_h, wall, wall, open_up=False), box.lid_mass, lid_origin, R,
                           mu=Pm.MU["box_box"], material="giftbox_lid", half=(lhx, lhy, lid_h))
    geom = dict(hx=hx, hy=hy, base_h=h, lid_h=lid_h, wall=wall, clearance=clearance, box=box.name,
                base_mass=box.base_mass, lid_mass=box.lid_mass)
    return base, lid, geom


class AirFilm:
    """Lid-base air film between affine bodies `lid` and `base` (indices in the contact world)."""

    def __init__(self, lid, base, geom, K=Pm.SLOT_LOSS_K, leak=0.):
        self.lid, self.base, self.g = lid, base, geom
        self.K, self.leak = K, leak
        self.log = []

    def state(self, w, q):
        """Overlap, side gaps, conductance and coefficients at the given state (base frame)."""
        g = self.g
        pb, Rb = w.body_pose(self.base, q)
        pl, Rl = w.body_pose(self.lid, q)
        ob = w.bodies[self.base].render["com_offset"]
        ol = w.bodies[self.lid].render["com_offset"]
        base_org = pb - Rb @ ob                                   # base outer-bottom origin
        lid_org = pl - Rl @ ol                                    # lid rim origin (opening down)
        rel = Rb.T @ (lid_org - base_org)
        overlap = g["base_h"] - rel[2]                            # skirt engaged length
        L = min(max(overlap, 0.), g["lid_h"] - g["wall"])
        c = g["clearance"]
        gaps = np.maximum(np.array([c - rel[0], c + rel[0], c - rel[1], c + rel[1]]), 5e-6)
        widths = np.array([2*g["hy"], 2*g["hy"], 2*g["hx"], 2*g["hx"]])
        A = 4*g["hx"]*g["hy"]
        if L <= 1e-5:
            return dict(L=L, active=False, A=A, n=Rb[:, 2])
        Cv = (widths*gaps**3).sum()/(12*Pm.AIR_MU*L) + self.leak
        c1 = A*A/Cv
        c2 = self.K*Pm.AIR_RHO*A**3/(2*(widths*gaps).sum()**2)
        return dict(L=L, active=True, A=A, n=Rb[:, 2], C=Cv, c1=c1, c2=c2, gaps=gaps)

    def begin(self, w, dt):
        self.s0 = self.state(w, w.q)
        ol, ob = w.body_dof[self.lid], w.body_dof[self.base]
        self.p0 = (w.q[ol:ol+3].copy(), w.q[ob:ob+3].copy())

    def term(self, w, q, xc_prev, dt, want):
        g = np.zeros(w.n_dof)
        coo = ([], [], [])
        s = getattr(self, "s0", None)
        if s is None or not s["active"]:
            return 0., g, coo
        ol, ob = w.body_dof[self.lid], w.body_dof[self.base]
        n = s["n"]
        dz = float(n @ ((q[ol:ol+3] - self.p0[0]) - (q[ob:ob+3] - self.p0[1])))
        c1, c2 = s["c1"], s["c2"]
        E = .5*c1/dt*dz*dz + c2/(3*dt*dt)*abs(dz)**3
        dE = c1/dt*dz + c2/(dt*dt)*dz*abs(dz)
        d2E = c1/dt + 2*c2/(dt*dt)*abs(dz)
        g[ol:ol+3] += dE*n
        g[ob:ob+3] -= dE*n
        if want:
            J = np.zeros(6); idx = np.r_[ol + np.arange(3), ob + np.arange(3)]
            J[:3], J[3:] = n, -n
            rr, cc = np.meshgrid(idx, idx, indexing="ij")
            coo[0].append(rr.ravel()); coo[1].append(cc.ravel()); coo[2].append((d2E*np.outer(J, J)).ravel())
        return E, g, coo

    def after(self, w, dt):
        s = getattr(self, "s0", None)
        ol, ob = w.body_dof[self.lid], w.body_dof[self.base]
        if s is None:
            return
        v = float(s["n"] @ (w.v[ol:ol+3] - w.v[ob:ob+3]))
        dp = 0.
        if s["active"]:
            Q = s["A"]*v
            dp = Q/s["C"] + self.K*Pm.AIR_RHO*Q*abs(Q)/(2*((np.array([2*self.g["hy"]]*2 + [2*self.g["hx"]]*2)*s["gaps"]).sum())**2)
        self.last = dict(t=w.time, overlap_m=float(s["L"]), rel_speed_m_s=v, dp_pa=float(dp),
                         force_n=float(dp*s["A"]), active=bool(s["active"]))
        self.log.append(self.last)


def install_film(world_ipc, film):
    """Register the air film's pre-step geometry, its implicit energy and its post-step log."""
    world_ipc.films.append(film)
    world_ipc.extra_terms = list(getattr(world_ipc, "extra_terms", [])) + [film.term]
    world_ipc.pre_terms = list(getattr(world_ipc, "pre_terms", [])) + [film.begin]
    world_ipc.post_terms = list(getattr(world_ipc, "post_terms", [])) + [film.after]

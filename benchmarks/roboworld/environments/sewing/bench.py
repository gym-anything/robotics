"""Component bench: cloth plies + the machine, no robot. Used by the feed, puncture, deflection and stitch probes
(E2-E9) and by the assisted demonstrations' phantom fixtures. The geometry is the cell's (machine origin at the needle,
bed flush with the table at z = 0)."""

import numpy as np

from . import params as P
from .cloth import ClothSolver, SHAPE_BOX, KIND_KINEMATIC
from .machine import Machine, MachineSettings
from .mesh import rect_panel, mesh_panel


def place_rect_ply(width, length, right_edge_x, near_end_y, z):
    """World placement of a rectangle ply [0,width] x [0,length] (u to +x, v to +y) whose right edge (u = width) is
    at x = right_edge_x and near end (v = 0) at y = near_end_y."""
    return np.array([right_edge_x - width, near_end_y, z]), np.eye(3)


def phantom_pull(mass, ids, force):
    """ASSISTED: a total force spread over the selected vertices in proportion to their mass (a uniform traction on
    the cloth they carry). An equal share per vertex would load the fine seam band (2.5 mm edges, ~1/9 of a bulk
    vertex's mass) nine times harder per area than the bulk and whip it."""
    m = mass[ids]
    return np.asarray(force, float)[None, :]*(m/np.sum(m))[:, None]


def phantom_torque(mass, ids, xy, tau, centre=None):
    """ASSISTED: a torque tau (N m about +z) as the in-plane force field m_i alpha (z x (r_i - r_c)) of a rigid rotation
    about r_c, alpha = tau / sum m_i |r_i - r_c|^2.
    centre=None: r_c is the selection's mass centre and the field is a pure couple (zero net force), as two hands
    turning a free stack apply; only a pin (the needle) then makes the work turn about another point (E9's control).
    centre=(x, y): hands pushing the work round that point (the needle): on a stack pinned there the field turns it
    rigidly, and its net force is reacted by the pin and the foot. A couple about the selection's own centre on a
    pinned stack instead compresses the cloth between that centre and the pin: at the 80 mN m a curve needs against
    the lowered foot it crumpled the work.
    A selection without rotational inertia (fewer than three vertices, or all at the centre) gets no force."""
    m = mass[ids]
    r = np.asarray(xy, float)
    if len(m) < 3:
        return np.zeros((len(m), 2))
    c = np.sum(m[:, None]*r, axis=0)/np.sum(m) if centre is None else np.asarray(centre, float)
    r = r - c
    inertia = float(np.sum(m*np.sum(r*r, axis=1)))
    if not inertia > 1e-15:
        return np.zeros((len(m), 2))
    alpha = tau/inertia
    return (m*alpha)[:, None]*np.column_stack([-r[:, 1], r[:, 0]])


class Bench:
    def __init__(self, fabric="muslin_125", plies=2, width=0.20, length=0.42, seam_allowance=0.010,
                 near_end_y=-0.030, settings=None, needle=True, solver_kw=None, band=22e-3, grain_deg=0.0,
                 pads=(), load=True):
        self.fabric_name = fabric
        f = dict(P.FABRICS[fabric])
        f["grain_deg"] = grain_deg
        self.fabric = f
        kw = P.solver_kwargs()
        kw.update(solver_kw or {})
        self.cloth = P.solver_attrs(ClothSolver(**kw))
        self.machine = Machine(settings or MachineSettings())
        self.machine.needle_installed = needle
        spec = rect_panel("ply", width, length, seam_offset=seam_allowance, band=band)
        self.mesh = mesh_panel(spec)
        th = max(f["thickness"], f.get("thickness_free", f["thickness"]))     # laid down uncompressed
        self.ply_ids = []
        # the operator's loading sequence: presser lifted, the plies laid down from just above the feed-dog teeth (at
        # the needle-up stop the teeth stand above the plate), the foot lowered once they have settled
        self.machine.presser = "lift2" if load else "down"
        z0 = max(0.0, self.machine.dog(self.machine.theta)[0]) + 0.05e-3 if load else 0.0
        for k in range(plies):
            origin, R = place_rect_ply(width, length, seam_allowance, near_end_y, z0 + 0.5*th + k*th + 1e-5)
            self.ply_ids.append(self.cloth.add_ply(f"ply{k}", self.mesh["X"], self.mesh["tri"], origin, R, f,
                                                   mu=1.0))
        self.machine.add_to_cloth(self.cloth, f)
        # phantom pads: kinematic flat pads that press and slide the cloth (ASSISTED fixtures)
        self.pads = []
        for name, half in pads:
            self.pads.append(self.cloth.add_shape(name, SHAPE_BOX, half, (0.0, 0.0, 0.5), kind=KIND_KINEMATIC,
                                                  mu=f["mu_pad"]/f["mu_steel"]*f["mu_steel"], on=False))
        self.cloth.build()
        self.machine.after_build()
        self.tick = 0
        self.width, self.length, self.seam_allowance, self.near_end_y = width, length, seam_allowance, near_end_y
        if load:
            self.settle(0.15)
            self.machine.command(presser="down")
            self.settle(0.05)

    @property
    def t(self):
        return self.tick*self.cloth.dt

    def step(self, n=1, fixture=None):
        for _ in range(n):
            if fixture is not None:
                fixture(self)
            self.machine.pre_tick(self.cloth.dt)
            self.cloth.step()
            self.tick += 1
            self.machine.post_tick(self.tick)

    def settle(self, seconds=0.2):
        self.step(int(round(seconds/self.cloth.dt)))

    def ply_vertices(self, k):
        p = self.cloth.plies[self.ply_ids[k]]
        return np.arange(p["v0"], p["v0"] + p["nv"])

    def marker(self, k, uv):
        """Vertex of ply k nearest to material point uv."""
        idx = self.ply_vertices(k)
        X = self.cloth.X[idx]
        return int(idx[np.argmin(np.linalg.norm(X - np.asarray(uv), axis=1))])

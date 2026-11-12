"""Outcome of break-hardpan: at least 80 % of the marked square loosened 80 mm deep or more and at least 4 kg of its
soil on the tarp (the task contracts: and at most 20 % of the square's lifted soil spilled elsewhere).

verify(e, lg, final=True, contract=None) reads the episode for the evaluator e from its particle ledger lg
(task.Ledger). Without a contract it returns the released judgement, (criteria, progress parts) as task.Evaluator.judge
takes it; with contract soil-tasks-v1 or soil-tasks-v2 it returns that contract's reading, (criteria, raw progress,
diagnostics) as task_set.py's and task_set_v2.py's TaskSetEvaluator take it."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from soil_excavation import layout as L, materials as M, params as P
from soil_excavation.catalogue import CONTRACT as TASKS_V1
from soil_excavation.catalogue_v2 import CONTRACT as TASKS_V2
from soil_excavation.task import ORIGIN_BLOCK, Ledger, crit, diag
from soil_excavation.task_set_v2 import TaskSetEvaluator, clip, ramp


def verify(e, lg, final=True, contract=None):
    if contract is None:
        return _break_hardpan(e, lg, final)
    return {TASKS_V1: read_v1, TASKS_V2: read_v2}[contract](e, lg)


def _break_hardpan(e, lg, final):
    (x0, y0), (x1, y1) = e.env.spec["target_square"]
    sq = (lg.pos0[:, 0] >= x0) & (lg.pos0[:, 0] <= x1) & (lg.pos0[:, 1] >= y0) & (lg.pos0[:, 1] <= y1)
    moved = np.linalg.norm(lg.pos - lg.pos0, axis=1) > 0.01
    broken = sq & (lg.loosened | moved)
    # loosened depth per 2 cm column of the square (from the bank positions)
    cell = 0.02
    nx, ny = int(round((x1 - x0)/cell)), int(round((y1 - y0)/cell))
    ix = np.clip(((lg.pos0[broken, 0] - x0)/cell).astype(int), 0, nx - 1)
    iy = np.clip(((lg.pos0[broken, 1] - y0)/cell).astype(int), 0, ny - 1)
    dmax = np.zeros((nx, ny))
    np.maximum.at(dmax, (ix, iy), L.GRADE - lg.pos0[broken, 2])
    cover = float(np.mean(dmax >= 0.08))
    m_tarp = float(lg.mass[sq & lg.on_tarp()].sum())
    c = [crit("loosened_80mm", cover, cover >= 0.8, ">= 80 % of the square loosened >= 80 mm deep", "-"),
         crit("fragments_on_tarp", m_tarp, m_tarp >= 4.0, ">= 4 kg of the square's soil on the tarp", "kg")]
    return c, [min(1., cover/0.8), min(1., m_tarp/4.)]


# soil-tasks-v1's reading (task_set.py)
def read_v1(e, lg):
    c, _ = _break_hardpan(e, lg, False)
    (x0, y0), (x1, y1) = e.env.spec["target_square"]
    sq = lg.in_rect(((x0, y0), (x1, y1)), pos=lg.pos0) & (lg.origin == ORIGIN_BLOCK)
    tarp = sq & lg.on_tarp()
    mass = float(lg.mass[tarp].sum())
    spill = float(lg.mass[sq & lg.out_of_block() & ~tarp].sum())
    frac = spill/max(mass + spill, 1e-9)
    c.append(crit("spill_fraction", frac, frac <= .20, "<= 20% of square soil lifted out"))
    # Smooth displacement/dilation credits at every depth, rather than a
    # count of columns whose deepest loosened particle crossed 80 mm.
    cell = .02
    nx, ny = round((x1 - x0)/cell), round((y1 - y0)/cell)
    initial = np.array([M.jp0(n) for n in M.NAMES])[lg.mat]
    motion = np.clip(np.linalg.norm(lg.pos - lg.pos0, axis=1)/.01, 0., 1.)
    dilation = np.clip((lg.jp/initial - 1.)/.05, 0., 1.)
    credit = np.maximum(motion, dilation)*np.clip((L.GRADE - lg.pos0[:, 2])/.08, 0., 1.)
    ix = np.clip(((lg.pos0[sq, 0] - x0)/cell).astype(int), 0, nx - 1)
    iy = np.clip(((lg.pos0[sq, 1] - y0)/cell).astype(int), 0, ny - 1)
    columns = np.zeros((nx, ny))
    np.maximum.at(columns, (ix, iy), credit[sq])
    raw = (.6*clip(float(columns.mean())/.8) + .4*clip(mass/4.))*ramp(frac, .2, .6)
    return c, raw, []


# soil-tasks-v2's reading (task_set_v2.py): only the soil the robot worked (WorkWindows)
def read_v2(e, lg):
    c, _ = _break_hardpan(e, lg, False)
    (x0, y0), (x1, y1) = e.env.spec["target_square"]
    sq = lg.in_rect(((x0, y0), (x1, y1)), pos=lg.pos0) & (lg.origin == ORIGIN_BLOCK)
    tarp = sq & lg.on_tarp()
    mass = float(lg.mass[tarp].sum())
    spill = float(lg.mass[sq & lg.out_of_block() & ~tarp].sum())
    frac = spill/max(mass + spill, 1e-9)
    c.append(crit("spill_fraction", frac, frac <= .20, "<= 20% of square soil lifted out"))
    # Smooth displacement/dilation credits at every depth, rather than a count of columns whose deepest loosened
    # particle crossed 80 mm; each particle's displacement and J_p rise are those of the windows in which the robot
    # worked its column, so soil creeping at rest earns nothing.
    ww = e.windows
    moved, dilated = ww.net(lg.pos[ww.idx], lg.jp[ww.idx])
    credit = np.maximum(np.clip(np.linalg.norm(moved, axis=1)/.01, 0., 1.), np.clip(dilated/.05, 0., 1.)) * \
        np.clip((L.GRADE - lg.pos0[ww.idx, 2])/.08, 0., 1.)
    columns = np.zeros(ww.shape[0]*ww.shape[1])
    np.maximum.at(columns, ww.column, credit)
    raw = (.6*clip(float(columns.mean())/.8) + .4*clip(mass/4.))*ramp(frac, .2, .6)
    return c, raw, [diag("worked_columns", float(ww.worked.mean()),
                         "share of the square's 2 cm columns a working body has touched; reported only")]


def _grid(lo, hi, step):
    return np.linspace(lo, hi, max(2, int(math.ceil((hi - lo)/step)) + 1))


def shape_points(shape, step=.01):
    """Points on one soil collider shape (scene.soil_shapes / World._robot_shapes_for_soil format), in its body's frame,
    at most `step` apart: a box's faces, a capsule's or cylinder's surface and axis, a mesh's triangles."""
    kind, size = shape["kind"], np.asarray(shape["size"], float)
    if kind == "box":
        g = np.stack(np.meshgrid(*[_grid(-h, h, step) for h in size], indexing="ij"), -1).reshape(-1, 3)
        pts = g[(np.abs(g) >= size - 1e-9).any(1)]
    elif kind in ("capsule", "cylinder"):
        r, half = size[0], size[1]
        end = half + (r if kind == "capsule" else 0.)
        z = _grid(-end, end, step)
        rad = np.sqrt(np.clip(r*r - np.clip(np.abs(z) - half, 0., None)**2, 0., None))
        a = np.linspace(0., 2*math.pi, max(8, int(math.ceil(2*math.pi*r/step))), endpoint=False)
        rings = [np.column_stack((q*np.cos(a), q*np.sin(a), np.full(len(a), h))) for h, q in zip(z, rad)]
        faces = [np.column_stack((f*r*np.cos(a), f*r*np.sin(a), np.full(len(a), h)))
                 for h in (-half, half) for f in _grid(0., 1., step/max(r, 1e-9))] if kind == "cylinder" else []
        pts = np.concatenate(rings + faces + [np.column_stack((0.*z, 0.*z, z))])
    elif kind in ("sphere", "ellipsoid"):
        radii = np.broadcast_to(size, (3,)) if kind == "ellipsoid" else np.full(3, size[0])
        n = max(32, int(4*math.pi*float(radii.max())**2/step**2))
        k = np.arange(n) + .5
        phi, theta = np.arccos(1. - 2.*k/n), math.pi*(1. + 5**.5)*k
        pts = np.column_stack((np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)))*radii
    elif kind == "mesh":
        v, f = shape["mesh"]
        v = np.asarray(v, float)*np.asarray(shape.get("scale", (1., 1., 1.)), float)
        tri = v[np.asarray(f, int).reshape(-1, 3)]
        div = np.ceil(np.linalg.norm(tri - np.roll(tri, 1, 1), axis=2).max(1)/step).astype(int)
        pts = [v]
        for m in np.unique(div[div > 1]):
            i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
            keep = i + j <= m
            bary = np.column_stack((i[keep], j[keep], m - i[keep] - j[keep]))/m
            pts.append(np.einsum("kb,tbd->tkd", bary, tri[div == m]).reshape(-1, 3))
        pts = np.unique(np.round(np.concatenate(pts), 4), axis=0)
    else:
        raise ValueError(f"Unsupported soil shape {kind}")
    rot = Rotation.from_quat(shape.get("quat", (0., 0., 0., 1.))).as_matrix()
    return pts @ rot.T + np.asarray(shape["pos"], float)


class WorkWindows:
    """break-hardpan's record of the soil the robot worked, kept by the verifier while the world steps.

    The square's soil creeps at rest: its J_p rises slowly under the idle robot (+1.5e-3 of the old reading in 15 s, not
    levelling off), so any reading of the state since construction eventually credits an idle robot. Here a particle's
    loosening is only the change in its position and J_p while its 2 cm column is open. A column opens when a point of
    a body the robot moves in the soil -- its own soil-coupled links and the tools; not the stones, root balls or posts
    lying in the ground -- moving at the blade-cut closure's v_min or faster, comes within BAND above the settled
    surface or below it, inside the column or within REACH of it; it closes HOLD_S after the last such touch, so the
    soil's own response to a blow (a clod falling back, the crater settling) still counts. Outside open columns the
    record does not change, so soil at rest earns nothing however long the robot idles. The record is history, like the
    legacy slab history: it is saved and restored with checkpoints and restarts with the episode.
    REACH and HOLD_S are sized from two scripted pick strikes with pry: of the particles that
    dilated past 5 %, all but 4 of 933 lay within 12 cm of a cell the mattock touched; 12 cm and 1 s keep 96 % of them."""
    CELL, REACH, BAND, HOLD_S = .02, .12, .02, 1.
    INCLUSIONS = ("stone", "rootball", "post")

    def __init__(self, env, lg):
        w = env.world
        (x0, y0), (x1, y1) = env.spec["target_square"]
        c = self.CELL
        self.shape = (round((x1 - x0)/c), round((y1 - y0)/c))
        self.pad = int(round(self.REACH/c)) + 1
        self.idx = np.flatnonzero(lg.in_rect(((x0, y0), (x1, y1)), pos=lg.pos0) & (lg.origin == ORIGIN_BLOCK))
        self.J0 = np.array([M.jp0(n) for n in M.NAMES])[lg.mat[self.idx]]
        ix = np.clip(((lg.pos0[self.idx, 0] - x0)/c).astype(int), 0, self.shape[0] - 1)
        iy = np.clip(((lg.pos0[self.idx, 1] - y0)/c).astype(int), 0, self.shape[1] - 1)
        self.column = ix*self.shape[1] + iy
        e = self.pad*c
        top, org = lg.heightfield(((x0 - e, y0 - e), (x1 + e, y1 + e)), cell=c)
        self.top = np.where(np.isnan(top), L.GRADE, top)      # the settled surface around the square
        self.origin = np.asarray(org[:2], float)
        self.v_min = float(P.CUT_CLOSURE["v_min"])
        self.bodies = self.working_bodies(w)
        self.restart()

    def working_bodies(self, w):
        """(coupled body row, body-frame points, corners of their body-frame box) of every body the robot moves in
        the soil."""
        couplers, rows = w._couplers()
        out = []
        for cp, row in zip(couplers, rows):
            if not cp["name"].startswith("robot/") and w.scene.items.get(cp["name"], {}).get("kind") in self.INCLUSIONS:
                continue
            pts = np.concatenate([shape_points(s) for s in cp["shapes"]])
            lo, hi = pts.min(0), pts.max(0)
            corners = np.array([[(lo, hi)[k >> i & 1][i] for i in range(3)] for k in range(8)])
            out.append((int(row), pts, corners))
        return out

    def restart(self):
        n = len(self.idx)
        self.moved, self.dilated = np.zeros((n, 3)), np.zeros(n)
        self.ref_pos = self.ref_jp = None
        self.open = np.zeros(self.shape, bool)
        self.last = np.full(self.shape, -1, np.int64)
        self.worked = np.zeros(self.shape, bool)
        self.prev = None

    KEYS = ("moved", "dilated", "ref_pos", "ref_jp", "open", "last", "worked")

    def state(self):
        st = {k: None if getattr(self, k) is None else getattr(self, k).copy() for k in self.KEYS}
        st["prev"] = None if self.prev is None else (self.prev[0], {r: p.copy() for r, p in self.prev[1].items()})
        return st

    def restore(self, st):
        for k in self.KEYS:
            setattr(self, k, None if st[k] is None else st[k].copy())
        self.prev = None if st["prev"] is None else (st["prev"][0], {r: p.copy() for r, p in st["prev"][1].items()})

    # ------------------------------------------------------------------ world reads (patched in the CPU tests)
    def body_points(self, w):
        """World points of the working bodies that can reach the square's surroundings: {row: (n, 3)}."""
        q = w.state.body_q.numpy()
        lo, hi = self.origin, self.origin + np.array(self.top.shape)*self.CELL
        zmax = float(self.top.max()) + self.BAND
        out = {}
        for row, pts, corners in self.bodies:
            rot, p = Rotation.from_quat(q[row, 3:7]).as_matrix(), q[row, :3]
            box = corners @ rot.T + p
            if (box[:, :2].max(0) >= lo).all() and (box[:, :2].min(0) <= hi).all() and box[:, 2].min() <= zmax:
                out[row] = pts @ rot.T + p
        return out

    def particle_state(self, w):
        return w.soil.state.particle_q.numpy()[self.idx].astype(float), \
            w.soil.state.mpm.particle_Jp.numpy()[self.idx].astype(float)

    # ------------------------------------------------------------------ recorder
    def touched(self, now, prev, dt):
        """Square columns within REACH of a working-body point that moved at v_min or faster over the last dt and is now
        at most BAND above the settled surface."""
        hit = np.zeros(self.top.shape, bool)
        for row, pts in now.items():
            if row not in prev:
                continue
            fast = pts[np.linalg.norm(pts - prev[row], axis=1) >= self.v_min*dt]
            ij = np.floor((fast[:, :2] - self.origin)/self.CELL).astype(int)
            ok = (ij >= 0).all(1) & (ij[:, 0] < hit.shape[0]) & (ij[:, 1] < hit.shape[1])
            ij, z = ij[ok], fast[ok, 2]
            low = z <= self.top[ij[:, 0], ij[:, 1]] + self.BAND
            hit[ij[low, 0], ij[low, 1]] = True
        r = int(round(self.REACH/self.CELL))
        near = np.zeros(self.shape, bool)
        for a, b in np.argwhere(hit) - self.pad:
            near[max(a - r, 0):max(a + r + 1, 0), max(b - r, 0):max(b + r + 1, 0)] = True
        return near

    def update(self, w):
        """Recorder, once per soil step: open the columns a working body touches (their reference state is read before
        the next soil step moves them) and close those untouched for HOLD_S; idle, nothing is read or changed."""
        ticks = int(w.soil_ticks)
        if w.tick % ticks:
            return
        tick = int(w.tick)
        now = self.body_points(w)
        near = np.zeros(self.shape, bool)
        if self.prev is not None and tick - self.prev[0] == ticks:
            near = self.touched(now, self.prev[1], ticks*P.SOIL_DT/P.SOIL_TICKS)
        self.prev = (tick, now)
        self.last[near] = tick
        closing = self.open & (tick - self.last > round(self.HOLD_S*P.SOIL_TICKS/P.SOIL_DT))
        opening = near & ~self.open
        if opening.any() or closing.any():
            self.sync(self.particle_state(w), opening, closing)

    def sync(self, now, opening, closing):
        pos, jp = now
        if self.ref_pos is not None:
            live = self.open.ravel()[self.column]
            self.moved[live] += pos[live] - self.ref_pos[live]
            self.dilated[live] += (jp[live] - self.ref_jp[live])/self.J0[live]
        self.ref_pos, self.ref_jp = pos.copy(), jp.copy()
        self.open = (self.open | opening) & ~closing
        self.worked |= opening

    def net(self, pos, jp):
        """Each tracked particle's (displacement, J_p/J_p bank change) accumulated while its column was open, including
        the open windows up to this state. Pure: the record is not changed."""
        moved, dilated = self.moved.copy(), self.dilated.copy()
        if self.ref_pos is not None:
            live = self.open.ravel()[self.column]
            moved[live] += pos[live] - self.ref_pos[live]
            dilated[live] += (jp[live] - self.ref_jp[live])/self.J0[live]
        return moved, dilated


# soil-tasks-v2's evaluator for the task (task_set_v2.install: V2_EVALUATOR); its reading is read_v2
class BreakHardpanEvaluator(TaskSetEvaluator):
    """break-hardpan: the same criteria; progress counts only soil the robot worked (WorkWindows)."""

    def __init__(self, env):
        self.windows = WorkWindows(env, Ledger(env))
        super().__init__(env)

    def _record(self, w):
        super()._record(w)
        self.windows.update(w)

    def history(self):
        return dict(super().history(), work_windows=self.windows.state())

    def restore_history(self, h):
        super().restore_history(h)
        self.windows.restore(h["work_windows"])

    def restart_history(self):
        super().restart_history()
        self.windows.restart()


V2_EVALUATOR = BreakHardpanEvaluator

"""Fine dust as a film: the part of floor dust that is too fine to be bodies.

REDUCED-ORDER MODEL. Everything else a broom moves in this environment is a rigid body in contact. Standard floor
dust is 60 % under 0.25 mm by mass, a gram of it is billions of particles, and it is what a broom leaves behind. It
is carried here as mass per area with a transport law whose closures are fitted to one published sweeping study.
It exerts no force on anything and nothing but bristle tips and a pan's tilt moves it.

State
    sigma       kg/m2 on a 4 mm grid over the room's floor (what a tip can pick up)
    fresh       kg/m2 on the same grid: what tips have just laid down; it joins sigma `hold` seconds later, when
                the bank that dropped it has passed (the leak length is a property of a whole bank's pass, see below)
    cover       the share of each cell's area that the tips of the passing bank have already been over
    load[k]     kg riding in front of the tip of bristle cluster k
    lodged[k]   kg caught in cluster k's bristles
    pans        kg per 2 mm cell on each dustpan's floor (in the pan's frame), and the wedge held in front of its lip
    tally       kg inside each container

Rules, for a cluster whose tip presses on a surface and has moved ds along it since the last update:
    1 pick-up   the tip takes a share `pickup` of what lies in its path above a residue: pickup sigma w ds, w the
                cluster's width (2 r); the rest stays where it lay (what a pass of a broom leaves behind).
                A cell is 4 mm and a cluster under 2 mm wide, so the cell remembers how much of its area has been
                passed over in this pass: later rows take only from the part no tip has passed yet, and the rows of
                a bank are taken to be staggered (the next row passes over what the first one's gaps left). Treating a cell as stirred instead left 46 % behind
                one row where its gaps are 22 % of its width.
    2 leak      the ridge a tip pushes trickles back: load (1 - exp(-ds/l_leak)) is laid under the tip (into `fresh`).
    3 lift      a tip that leaves the surface keeps a fraction beta of its load lodged in the bristles and drops the
                rest where it is: a broom lifted before the end of its stroke leaves a line
    4 lodged    lodged dust trickles out under a sliding tip with the time constant tau_lodged, and all at once when
                the tip is stopped harder than a_tap (the broom tapped on the floor)
    5 lip       a tip that comes onto a pan's floor from the room's floor first fills the wedge in front of the
                pan's lip, rho_b h^2/(2 tan phi) per length of lip; the rest of its load goes on with it. The wedge
                is the line a dustpan leaves: it stays on the floor when the pan is moved away
    6 in a pan  the pan's film slides toward whichever side the pan's floor tilts to by more than the repose angle
                phi, piles against a wall there, and falls from the open lip onto whatever is below: a container's
                tally, another pan, or the room's floor. A pan turned upside down drops all of it

Mass is conserved to rounding: `total()` is the same after every update (checked in the tests).
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import params as P

EVERY = 10                         # ticks between updates (5 ms): a tip at 1 m/s moves 5 mm, sub-sampled to 2 mm
DRAWN = 2.5                        # g/m2 at which a picture shows 63 % of a cell covered (authored drawing scale; 6.0 until
                                   # the loads were brought inside the published distribution, params.DUST_LOADS)
OFF, FLOOR = 0, 1                  # surface codes; pans are 2, 3, ...


def lay(grid, patch, rng, cell, origin):
    """Add one patch of film to `grid`. disc: grams within a radius, thinning toward its edge; rect: g/m2 in a box.
    A fifth of it is speckle (dust does not lie evenly), the total is exact."""
    nx, ny = grid.shape
    if patch["kind"] == "disc":
        cx, cy = patch["centre"]
        r = patch["radius"]
        box = (cx - r, cy - r, cx + r, cy + r)
    else:
        box = patch["box"]
    i0, j0 = max(0, int((box[0] - origin[0])/cell)), max(0, int((box[1] - origin[1])/cell))
    i1, j1 = min(nx, int(math.ceil((box[2] - origin[0])/cell))), min(ny, int(math.ceil((box[3] - origin[1])/cell)))
    x = origin[0] + (np.arange(i0, i1) + 0.5)*cell
    y = origin[1] + (np.arange(j0, j1) + 0.5)*cell
    X, Y = np.meshgrid(x, y, indexing="ij")
    if patch["kind"] == "disc":
        d = np.hypot(X - cx, Y - cy)/r
        shape = np.clip(1.0 - d*d, 0.0, None)
    else:
        shape = np.ones_like(X)
    for (ax, ay, bx, by) in patch.get("avoid", ()):
        shape[(X > ax) & (X < bx) & (Y > ay) & (Y < by)] = 0.0
    shape = shape*(0.8 + 0.4*rng.random(shape.shape))
    if shape.sum() <= 0:
        return 0.0
    kg = 1e-3*patch["grams"] if "grams" in patch else 1e-3*patch["g_per_m2"]*float((shape > 0).sum())*cell*cell
    grid[i0:i1, j0:j1] += shape*(kg/shape.sum())/(cell*cell)
    return kg


class DustFilm:
    def __init__(self, sweeper, tools, patches, seed=0, *, pose, active, clock, params=None):
        """sweeper: the bristle solver (cluster tips are read from it); tools: name -> the tool's builder dict (pans
        and containers are found in it); pose(name) -> (position, rotation matrix) of a tool's body; active() -> one
        flag per bristle tool (its bank is being simulated); clock() -> simulated time. `params` overrides entries of
        params.DUST."""
        D = self.p = dict(P.DUST, **(params or {}))
        self.sweeper, self.pose, self.active, self.clock = sweeper, pose, active, clock
        self.cell, self.pan_cell = D["cell"], D["pan_cell"]
        x0, y0, x1, y1 = P.ROOM
        self.origin = np.array((x0, y0))
        self.nx, self.ny = int(math.ceil((x1 - x0)/self.cell)), int(math.ceil((y1 - y0)/self.cell))
        self.sigma = np.zeros((self.nx, self.ny))
        self.fresh = np.zeros((self.nx, self.ny))
        self.cover = np.zeros((self.nx, self.ny), np.float32)
        self.cover_t = np.full((self.nx, self.ny), -1.0e9, np.float32)
        self.pending = []                                      # (time laid, flat cell indices)
        rng = np.random.default_rng(seed)
        self.patches = [dict(p) for p in patches]
        for p in self.patches:
            p["kg"] = lay(self.sigma, p, rng, self.cell, self.origin)
        sw = sweeper
        R = max((s.stop for s in sw.rod_slices), default=0)
        self.R = R
        self.tip = (sw.rod_n.numpy()[:R] - 1).astype(int) if R else np.zeros(0, int)
        self.width = 2.0*sw.rod_rad.numpy()[:R].astype(float) if R else np.zeros(0)
        self.rod_tool = sw.rod_tool.numpy()[:R].astype(int) if R else np.zeros(0, int)
        self.load = np.zeros(R)
        self.lodged = np.zeros(R)
        self.surf = np.zeros(R, int)
        self.prev = np.zeros((R, 3))                           # last contact point: world (floor) or pan frame
        self.speed = np.zeros(R)
        # pans: floor field in the pan's frame, x across, y from the back wall (-depth/2) to the lip (+depth/2)
        self.pans = {}
        for name, sp in tools.items():
            if sp.get("kind") != "pan":
                continue
            wdt, depth = sp["mouth"]["width"], sp["mouth"]["depth"]
            n = (int(math.ceil(wdt/self.pan_cell)), int(math.ceil(depth/self.pan_cell)))
            lip = float(sp["spec"]["lip"])
            self.pans[name] = dict(code=2 + len(self.pans), width=wdt, depth=depth, tray=float(sp.get("tray", 0.004)), n=n, field=np.zeros(n),
                                   wedge=0.0, wedge_cap=D["bulk_density"]*lip*lip/(2.0*math.tan(math.radians(D["repose_deg"])))*wdt,
                                   wedge_at=None, slide=np.zeros(2), speed=np.zeros(2), lip=lip)
        self.by_code = {p["code"]: (n, p) for n, p in self.pans.items()}
        self.containers = {n: sp["inner"] for n, sp in tools.items() if sp.get("kind") == "container"}
        self.tally = {n: 0.0 for n in self.containers}
        self.outside = 0.0                                     # fell outside the room's floor (never, in a closed room)
        self.total0 = self.total()
        self.sigma0 = self.sigma.copy()        # what lay in each place at the start: a broom never moves a share of it (rule 1)
        self.updates = 0
        # the piece of floor that pictures of the film cover: the patches and 0.6 m around them
        if self.patches:
            bx = [p["box"] if p["kind"] == "rect" else (p["centre"][0] - p["radius"], p["centre"][1] - p["radius"],
                                                        p["centre"][0] + p["radius"], p["centre"][1] + p["radius"]) for p in self.patches]
            m = 0.6
            self.view_box = (max(x0, min(b[0] for b in bx) - m), max(y0, min(b[1] for b in bx) - m),
                             min(x1, max(b[2] for b in bx) + m), min(y1, max(b[3] for b in bx) + m))
        else:
            self.view_box = None

    # ------------------------------------------------------------------ bookkeeping
    def total(self):
        a = self.cell*self.cell
        return float(self.sigma.sum()*a + self.fresh.sum()*a + self.load.sum() + self.lodged.sum() + self.outside
                     + sum(p["field"].sum() + p["wedge"] for p in self.pans.values()) + sum(self.tally.values()))

    def ridge_on_floor(self):
        """kg riding in front of bristle tips that are on the room's floor: a ridge of dust that lies on the floor in
        front of a broom. (What is lodged between the bristles after a lift is in the broom, not on the floor.)"""
        return float(self.load[self.surf == FLOOR].sum())

    def on_floor(self, box=None, disc=None):
        """kg on the room's floor (lines in front of pans included), in a box (x0, y0, x1, y1) or a disc (cx, cy, r).
        Not counted here: the ridge in front of a broom's tips (ridge_on_floor); the evaluator adds it."""
        s = self.sigma + self.fresh
        a = self.cell*self.cell
        if box is None and disc is None:
            return float(s.sum()*a) + sum(p["wedge"] for p in self.pans.values())
        if disc is not None:
            cx, cy, r = disc
            box = (cx - r, cy - r, cx + r, cy + r)
        i0, j0 = max(0, int((box[0] - self.origin[0])/self.cell)), max(0, int((box[1] - self.origin[1])/self.cell))
        i1, j1 = min(self.nx, int(math.ceil((box[2] - self.origin[0])/self.cell))), min(self.ny, int(math.ceil((box[3] - self.origin[1])/self.cell)))
        part = s[i0:i1, j0:j1]
        if disc is not None:
            x = self.origin[0] + (np.arange(i0, i1) + 0.5)*self.cell
            y = self.origin[1] + (np.arange(j0, j1) + 0.5)*self.cell
            part = part*((x[:, None] - cx)**2 + (y[None, :] - cy)**2 < r*r)
        kg = float(part.sum()*a)
        for p in self.pans.values():
            if p["wedge"] > 0 and p["wedge_at"] is not None:
                m = p["wedge_at"][0].mean(0)
                if box[0] <= m[0] <= box[2] and box[1] <= m[1] <= box[3]:
                    kg += p["wedge"]
        return kg

    def summary(self):
        """Where the dust is, in grams (privileged state)."""
        g = lambda v: round(1e3*float(v), 4)
        return dict(start=g(self.total0), floor=g(self.on_floor()), riding_on_tips=g(self.load.sum()), lodged_in_bristles=g(self.lodged.sum()),
                    pans={n: g(p["field"].sum()) for n, p in self.pans.items()},
                    lines_at_a_lip={n: g(p["wedge"]) for n, p in self.pans.items()},
                    containers={n: g(v) for n, v in self.tally.items()}, outside_the_room=g(self.outside),
                    unaccounted=round(1e3*(self.total() - self.total0), 9))

    def state(self):
        sparse = lambda g: (np.flatnonzero(g), g.reshape(-1)[np.flatnonzero(g)].copy())
        return dict(sigma=sparse(self.sigma), fresh=sparse(self.fresh), pending=[(t, c.copy()) for t, c in self.pending], load=self.load.copy(),
                    lodged=self.lodged.copy(), surf=self.surf.copy(), prev=self.prev.copy(), speed=self.speed.copy(),
                    pans={n: dict(field=p["field"].copy(), wedge=p["wedge"], wedge_at=None if p["wedge_at"] is None else tuple(a.copy() for a in p["wedge_at"]),
                                  slide=p["slide"].copy(), speed=p["speed"].copy()) for n, p in self.pans.items()},
                    tally=dict(self.tally), outside=self.outside, cover=(np.flatnonzero(self.cover), self.cover.reshape(-1)[np.flatnonzero(self.cover)].copy(),
                                                                       self.cover_t.reshape(-1)[np.flatnonzero(self.cover)].copy()))

    def restore(self, s):
        for g, (cells, values) in ((self.sigma, s["sigma"]), (self.fresh, s["fresh"])):
            g[:] = 0.0
            g.reshape(-1)[cells] = values
        self.pending = [(t, c.copy()) for t, c in s["pending"]]
        self.load[:], self.lodged[:], self.surf[:], self.prev[:], self.speed[:] = s["load"], s["lodged"], s["surf"], s["prev"], s["speed"]
        for n, p in self.pans.items():
            q = s["pans"][n]
            p["field"][:] = q["field"]
            p["wedge"], p["wedge_at"] = q["wedge"], None if q["wedge_at"] is None else tuple(a.copy() for a in q["wedge_at"])
            p["slide"][:], p["speed"][:] = q["slide"], q["speed"]
        self.tally = dict(s["tally"])
        self.outside = s["outside"]
        self.cover[:] = 0.0
        self.cover_t[:] = -1.0e9
        cells, c, t = s["cover"]
        self.cover.reshape(-1)[cells] = c
        self.cover_t.reshape(-1)[cells] = t

    # ------------------------------------------------------------------ where a falling or laid-down speck lands
    def _cells(self, xy):
        ij = np.floor((np.atleast_2d(xy) - self.origin)/self.cell).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        return ij[:, 0]*self.ny + ij[:, 1], inside

    def _lay_floor(self, xy, kg):
        """Onto the floor, not to be picked up again before the bank has passed."""
        flat, inside = self._cells(xy)
        kg = np.broadcast_to(np.asarray(kg, float), (len(flat),))
        if (~inside).any():
            self.outside += float(kg[~inside].sum())
        if inside.any():
            np.add.at(self.fresh.reshape(-1), flat[inside], kg[inside]/(self.cell*self.cell))
            self.pending.append((self.clock(), np.unique(flat[inside])))

    def _lay_pan(self, p, local_xy, kg):
        i = np.clip(np.floor((local_xy[:, 0] + p["width"]/2)/self.pan_cell).astype(int), 0, p["n"][0] - 1)
        j = np.clip(np.floor((local_xy[:, 1] + p["depth"]/2)/self.pan_cell).astype(int), 0, p["n"][1] - 1)
        np.add.at(p["field"], (i, j), kg)

    def drop(self, points, kg, poses):
        """Specks let go at world points: into a container if they are over its opening, onto a pan's floor if they
        are over it, else onto the room's floor."""
        points = np.atleast_2d(np.asarray(points, float))
        kg = np.broadcast_to(np.asarray(kg, float), (len(points),)).copy()
        left = kg > 0
        for name, inner in self.containers.items():
            if not left.any():
                break
            pc, Rc = poses[name]
            loc = (points - pc) @ Rc
            if inner["shape"] == "round":
                m = np.hypot(loc[:, 0], loc[:, 1]) < inner.get("r_corner", inner["r"])
            else:
                m = (np.abs(loc[:, 0]) < inner["hx"]) & (np.abs(loc[:, 1]) < inner["hy"])
            m &= left & (loc[:, 2] > 0.0) & (Rc[2, 2] > 0.5)           # above its bottom, and it stands
            self.tally[name] += float(kg[m].sum())
            left &= ~m
        for name, p in self.pans.items():
            if not left.any():
                break
            pp, Rp = poses[name]
            loc = (points - pp) @ Rp
            m = left & (np.abs(loc[:, 0]) < p["width"]/2) & (np.abs(loc[:, 1]) < p["depth"]/2) & (loc[:, 2] > 0.0) & (Rp[2, 2] > 0.5)
            if m.any():
                self._lay_pan(p, loc[m, :2], kg[m])
            left &= ~m
        if left.any():
            self._lay_floor(points[left, :2], kg[left])

    # ------------------------------------------------------------------ one update
    def busy(self):
        return bool(self.load.any() or any(p["field"].any() for p in self.pans.values()))

    def update(self, dt, tips=None):
        """One update over the last `dt` seconds. `tips` (R, 3): the cluster tips, if the caller has them already
        (every download from the solver waits for the queued kernels; two films on one brush share one)."""
        D = self.p
        self.updates += 1
        now = self.clock()
        # what was laid down `hold` seconds ago can be picked up again
        hold = D["hold"]
        while self.pending and now - self.pending[0][0] >= hold:
            _, cells = self.pending.pop(0)
            f, s = self.fresh.reshape(-1), self.sigma.reshape(-1)
            s[cells] += f[cells]
            f[cells] = 0.0
            self.cover.reshape(-1)[cells] = 0.0                # it lies anywhere in the cell
        poses = {n: self.pose(n) for n in list(self.pans) + list(self.containers)}
        self._wedges(poses)
        self._pans(poses, dt)
        if not self.R:
            return
        active = np.asarray(self.active(), bool)
        on = active[self.rod_tool] if len(active) else np.zeros(self.R, bool)
        if not on.any() and not self.surf.any():
            return
        tip = self.tips() if tips is None else tips
        # ---- which surface each tip presses on
        surf = np.zeros(self.R, int)
        local = np.zeros((self.R, 3))
        surf[on & (tip[:, 2] < 0.5*self.width + self.p["contact_gap"])] = FLOOR
        for name, p in self.pans.items():
            pp, Rp = poses[name]
            loc = (tip - pp) @ Rp
            m = on & (np.abs(loc[:, 0]) < p["width"]/2) & (np.abs(loc[:, 1]) < p["depth"]/2) & (loc[:, 2] > -0.002) & \
                (loc[:, 2] < p["tray"] + 0.5*self.width + self.p["contact_gap"])
            surf[m] = p["code"]
            local[m] = loc[m]
        point = np.where((surf >= 2)[:, None], local, tip)
        same = (surf == self.surf) & (surf != OFF)
        ds = np.where(same, np.linalg.norm((point - self.prev)[:, :2], axis=1), 0.0)
        speed = ds/dt
        a = self.cell*self.cell
        # ---- 5: a tip coming from the floor onto a pan fills the wedge at that pan's lip first
        for name, p in self.pans.items():
            m = (surf == p["code"]) & (self.surf == FLOOR) & (self.load > 0)
            if m.any():
                room = max(0.0, p["wedge_cap"] - p["wedge"])
                give = self.load[m]*min(1.0, room/float(self.load[m].sum()))
                self.load[m] -= give
                p["wedge"] += float(give.sum())
                pp, Rp = poses[name]
                ends = np.array([pp + Rp @ (-p["width"]/2, p["depth"]/2, 0.0), pp + Rp @ (p["width"]/2, p["depth"]/2, 0.0)])
                p["wedge_at"] = (ends[:, :2].copy(), pp.copy())
        # ---- 3: lift (and a surface change counts as leaving the first surface, except floor -> pan over the lip)
        left = (self.surf != OFF) & (surf != self.surf) & ~((self.surf == FLOOR) & (surf >= 2))
        if left.any():
            beta = self.p["lodging"]
            m = left & (self.load > 0)
            if m.any():
                gone = self.load[m]*(1.0 - beta)
                self.lodged[m] += self.load[m]*beta
                self.load[m] = 0.0
                self.drop(tip[m], gone, poses)
        # ---- 4: lodged dust trickles out under a sliding tip; all of it when the tip is stopped hard
        sliding = (surf != OFF) & (ds > 0)
        tap = (surf != OFF) & (self.speed - speed > self.p["a_tap"]*dt)
        trickle = np.where(tap, self.lodged, np.where(sliding, self.lodged*(1.0 - math.exp(-dt/self.p["tau_lodged"])), 0.0))
        self.lodged -= trickle
        # ---- 2: leak
        leak = np.where(same & (ds > 0), self.load*(1.0 - np.exp(-ds/self.p["leak_length"])), 0.0)
        self.load -= leak
        out = trickle + leak
        m = out > 0
        if m.any():
            f = m & (surf == FLOOR)
            if f.any():
                self._lay_floor(tip[f, :2], out[f])
            for name, p in self.pans.items():
                q = m & (surf == p["code"])
                if q.any():
                    self._lay_pan(p, local[q, :2], out[q])
        # ---- 1: pick-up along the path since the last update (sub-sampled to half a cell)
        f = same & (surf == FLOOR) & (ds > 0)
        if f.any():
            idx = np.flatnonzero(f)
            n = int(min(8, max(1, math.ceil(float(ds[idx].max())/(0.5*self.cell)))))
            u = (np.arange(n) + 0.5)/n
            pts = self.prev[idx, None, :2] + u[None, :, None]*(point[idx, None, :2] - self.prev[idx, None, :2])
            flat, inside = self._cells(pts.reshape(-1, 2))
            area = np.repeat(self.width[idx]*ds[idx]/n, n)
            area[~inside] = 0.0
            flat = np.where(inside, flat, 0)
            cells, inv = np.unique(flat, return_inverse=True)
            swept = np.bincount(inv, weights=area, minlength=len(cells))
            s = self.sigma.reshape(-1)
            cov, cov_t = self.cover.reshape(-1), self.cover_t.reshape(-1)
            c = np.where(now - cov_t[cells] > hold, 0.0, cov[cells])        # a pass that ended is forgotten
            avail = np.clip(s[cells] - self.p["residue_share"]*self.sigma0.reshape(-1)[cells], 0.0, None)*a
            # Of the cell's area a share c has been passed over in this pass, and what lies there now is what the pass
            # left (1 - pickup of what lay there); the rest, 1 - c, is untouched. So of what the cell holds now the
            # untouched part is (1 - c)/(1 - c pickup), and the tip takes `pickup` of as much of that as it passes
            # over. With every later row taking its share of the WHOLE cell again, five
            # rows clear 98 % in one sweep whatever `pickup` is.
            pk = self.p["pickup"]
            taken = pk*avail*np.minimum(1.0 - c, swept/a)/np.maximum(1.0 - c*pk, 1e-15)
            s[cells] -= taken/a
            cov[cells] = np.minimum(1.0, c + swept/a)
            cov_t[cells] = now
            share = np.where(swept[inv] > 0, area/np.maximum(swept[inv], 1e-30), 0.0)*taken[inv]
            self.load[idx] += share.reshape(len(idx), n).sum(1)
        for name, p in self.pans.items():
            q = same & (surf == p["code"]) & (ds > 0)
            if q.any() and p["field"].any():
                idx = np.flatnonzero(q)
                i = np.clip(np.floor((local[idx, 0] + p["width"]/2)/self.pan_cell).astype(int), 0, p["n"][0] - 1)
                j = np.clip(np.floor((local[idx, 1] + p["depth"]/2)/self.pan_cell).astype(int), 0, p["n"][1] - 1)
                flat = i*p["n"][1] + j
                cells, inv = np.unique(flat, return_inverse=True)
                pa = self.pan_cell*self.pan_cell
                swept = np.bincount(inv, weights=self.width[idx]*ds[idx], minlength=len(cells))
                fld = p["field"].reshape(-1)
                taken = fld[cells]*np.minimum(1.0, swept/pa)
                fld[cells] -= taken
                self.load[idx] += self.width[idx]*ds[idx]/np.maximum(swept[inv], 1e-30)*taken[inv]
        self.surf, self.speed = surf, speed
        self.prev = np.where((surf != OFF)[:, None], point, self.prev)

    def tips(self):
        X = self.sweeper.rod_x.numpy()[:self.R]
        return X[np.arange(self.R), self.tip].astype(float)

    def _wedges(self, poses):
        """A pan that has been moved away leaves the line that stood at its lip on the floor."""
        for name, p in self.pans.items():
            if p["wedge"] > 0 and p["wedge_at"] is not None:
                pp, Rp = poses[name]
                if np.linalg.norm(pp - p["wedge_at"][1]) > 0.003:
                    ends = p["wedge_at"][0]
                    n = max(2, int(np.linalg.norm(ends[1] - ends[0])/(0.5*self.cell)))
                    u = (np.arange(n) + 0.5)/n
                    self._lay_floor(ends[0] + u[:, None]*(ends[1] - ends[0]), np.full(n, p["wedge"]/n))
                    p["wedge"], p["wedge_at"] = 0.0, None

    def _pans(self, poses, dt):
        tan_phi = math.tan(math.radians(self.p["repose_deg"]))
        for name, p in self.pans.items():
            fld = p["field"]
            if not fld.any():
                p["speed"][:] = 0.0
                p["slide"][:] = 0.0
                continue
            pp, Rp = poses[name]
            down = Rp.T @ np.array((0.0, 0.0, -1.0))              # gravity's direction in the pan's frame
            if down[2] > 0.0:
                # upside down: everything falls from where it lies
                i, j = np.nonzero(fld)
                loc = np.stack([(i + 0.5)*self.pan_cell - p["width"]/2, (j + 0.5)*self.pan_cell - p["depth"]/2, np.full(len(i), p["tray"])], 1)
                kg = fld[i, j].copy()
                fld[:] = 0.0
                self._fall(name, pp + loc @ Rp.T, kg, poses)
                continue
            g_in = down[:2]*9.81
            pull = np.linalg.norm(g_in)
            hold = tan_phi*9.81*(-down[2])
            if pull > hold:
                p["speed"] += g_in/pull*(pull - hold)*dt
            else:
                v = np.linalg.norm(p["speed"])
                p["speed"] *= max(0.0, 1.0 - (hold - pull)*dt/v) if v > 0 else 0.0
            # the film cannot slide faster than something that has slid the whole length of the pan
            v_max = math.sqrt(2.0*9.81*max(p["width"], p["depth"]))
            p["speed"] = np.clip(p["speed"], -v_max, v_max)
            p["slide"] += p["speed"]*dt
            for ax in (0, 1):
                k = int(p["slide"][ax]/self.pan_cell)
                if k == 0:
                    continue
                p["slide"][ax] -= k*self.pan_cell
                # Film that already lies against the wall it slides to has stopped: its speed does not go on growing.
                edge = fld[-1, :] if (ax == 0 and k > 0) else fld[0, :] if ax == 0 else fld[:, 0]
                if not (ax == 1 and k > 0) and float(edge.sum()) >= float(fld.sum())*(1.0 - 1e-12):
                    p["speed"][ax] = 0.0
                    p["slide"][ax] = 0.0
                    continue
                moved = np.zeros_like(fld)
                n = fld.shape[ax]
                kk = min(abs(k), n)
                if ax == 0:
                    # side walls: what reaches one piles against it
                    if k > 0:
                        moved[kk:, :] = fld[:n - kk, :]; moved[-1, :] += fld[n - kk:, :].sum(0)
                    else:
                        moved[:n - kk, :] = fld[kk:, :]; moved[0, :] += fld[:kk, :].sum(0)
                elif k > 0:
                    # toward the lip: what passes it falls
                    moved[:, kk:] = fld[:, :n - kk]
                    over = fld[:, n - kk:].sum(1)
                    i = np.flatnonzero(over)
                    if len(i):
                        loc = np.stack([(i + 0.5)*self.pan_cell - p["width"]/2, np.full(len(i), p["depth"]/2 + 0.002), np.zeros(len(i))], 1)
                        self._fall(name, pp + loc @ Rp.T, over[i], poses)
                else:
                    # back wall
                    moved[:, :n - kk] = fld[:, kk:]; moved[:, 0] += fld[:, :kk].sum(1)
                fld[:] = moved

    def _fall(self, from_pan, points, kg, poses):
        other = {n: v for n, v in poses.items() if n != from_pan}
        pans = self.pans
        self.pans = {n: p for n, p in pans.items() if n != from_pan}
        try:
            self.drop(points, kg, other)
        finally:
            self.pans = pans

    # ------------------------------------------------------------------ for pictures
    def picture(self):
        """What a renderer needs to draw the film: how much of each cell is covered, 0 to 255, for the floor inside
        `view_box` (lines at pan lips and what rides in front of the tips included) and for each pan's floor.
        Cover is 1 - exp(-load/DRAWN): an AUTHORED drawing scale (2.5 g/m2 covers 63 %), not a measurement."""
        cover = lambda g_per_m2: np.round(255.0*(1.0 - np.exp(-np.asarray(g_per_m2)/DRAWN))).astype(np.uint8)
        out = dict(floor=None, box=self.view_box, pans={})
        if self.view_box is not None:
            img, box = self.image(self.view_box)
            m = (self.surf == FLOOR) & (self.load > 0)
            if m.any():
                ij = np.floor((self.prev[m, :2] - (box[0], box[1]))/self.cell).astype(int)
                ok = (ij[:, 0] >= 0) & (ij[:, 0] < img.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < img.shape[1])
                np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1e3*self.load[m][ok]/(self.cell*self.cell))
            out["floor"] = cover(img)
        for n, p in self.pans.items():
            out["pans"][n] = cover(1e3*p["field"]/(self.pan_cell*self.pan_cell))
        return out

    def image(self, box, full=None):
        """The floor film inside `box` as g/m2 on the 4 mm grid (rows along x), lines at pan lips included."""
        i0, j0 = max(0, int((box[0] - self.origin[0])/self.cell)), max(0, int((box[1] - self.origin[1])/self.cell))
        i1, j1 = min(self.nx, int(math.ceil((box[2] - self.origin[0])/self.cell))), min(self.ny, int(math.ceil((box[3] - self.origin[1])/self.cell)))
        img = 1e3*(self.sigma[i0:i1, j0:j1] + self.fresh[i0:i1, j0:j1])
        for p in self.pans.values():
            if p["wedge"] > 0 and p["wedge_at"] is not None:
                ends = p["wedge_at"][0]
                n = max(2, int(np.linalg.norm(ends[1] - ends[0])/(0.5*self.cell)))
                u = (np.arange(n) + 0.5)/n
                ij = np.floor((ends[0] + u[:, None]*(ends[1] - ends[0]) - self.origin)/self.cell).astype(int) - (i0, j0)
                ok = (ij[:, 0] >= 0) & (ij[:, 0] < img.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < img.shape[1])
                np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1e3*p["wedge"]/n/(self.cell*self.cell))
        return img, (self.origin[0] + i0*self.cell, self.origin[1] + j0*self.cell, self.origin[0] + i1*self.cell, self.origin[1] + j1*self.cell)

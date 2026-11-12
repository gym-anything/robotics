"""Task evaluator: physically meaningful criteria read from the world state.

Everything is measured from the simulated state, never from commands: the soil surface from the particles (column
tops on a 1 cm grid, a column counted where the top voxel holds at least 3 of its nominal 8 particles), masses from
particle masses and where they lie, loosening from each particle's plastic volume ratio J_p against its bank value,
packing from particle volumes per voxel, tool poses and grasps from MuJoCo bodies and contacts, protective stops from
the robot's reflex log, pipe strikes from the strike log. Tolerances are authored and stated with their basis
(params / layout). progress() is cheap (cached per tick); evaluate() is the verdict (`gar verify`): set-post runs its
brace-release-and-push test on a copy of the state and restores it (no history discontinuity).

The eight benchmark tasks (catalogue.EIGHT_TASKS) are judged by their folders, tasks/<task>/verifier.py
(Evaluator.judge; gym_anything_robotics.task_folders); this module keeps the evaluator, the measurements they share
(Ledger, tool_held, the hole and spoil readings, the common criteria) and the judgements of the other tasks.
"""

import math
import numpy as np

from . import params as P
from . import layout as L
from . import materials as M
from .soil import ORIGIN_BLOCK, ORIGIN_PILE

LOOSE_JP = 1.05            # a particle is loosened when its J_p rose 5 % over its bank value (dilated, cohesion lost)
CRUMB_MOVE = 0.02          # a loosened particle 2 cm or more from its bank position has been dug or knocked loose


class Ledger:
    """Particle ledger of one evaluation."""

    def __init__(self, env):
        w = env.world
        p = w.soil.particles()
        self.pos, self.pos0 = p["pos"].astype(float), p["pos0"].astype(float)
        self.mass, self.mat, self.jp, self.origin = p["mass"].astype(float), p["mat"], p["jp"].astype(float), p["origin"]
        J0 = np.array([M.jp0(n) for n in M.NAMES])[self.mat]
        self.loosened = (self.jp > LOOSE_JP*J0) | (self.origin == ORIGIN_PILE)
        # loose soil that is no longer where the bank held it: dug and dropped back, knocked off a wall, or spoil. The
        # blade cut closure remolds the soil a blade edge sweeps in place (a 2 cm grid cannot hold a 2.5 mm cut open);
        # that soil has not moved and stands for the cut face, not for crumbs
        self.displaced_loose = self.loosened & ((np.linalg.norm(self.pos - self.pos0, axis=1) > CRUMB_MOVE)
                                                | (self.origin == ORIGIN_PILE))
        self.h = w.soil.h
        self.voxel = w.soil.voxel
        self.spec = env.spec
        self.block = env.spec["block"]
        self.vol = (8*(0.5*self.h*np.array([M.bank_fraction(n) for n in M.NAMES])[self.mat]**(1/3))**3)

    # ------------------------------------------------------------------ regions
    def in_rect(self, rect, pad=0.0, pos=None):
        (x0, y0), (x1, y1) = rect
        x = self.pos if pos is None else pos
        return (x[:, 0] >= x0 - pad) & (x[:, 0] <= x1 + pad) & (x[:, 1] >= y0 - pad) & (x[:, 1] <= y1 + pad)

    def block_rect(self):
        return ((self.block["lo"][0], self.block["lo"][1]), (self.block["hi"][0], self.block["hi"][1]))

    def on_tarp(self, origin=None):
        (x0, y0), (x1, y1) = self.spec["tarp"]
        z0 = 0.0 if y1 <= 0.0 and y0 >= L.SLAB["y"][0] else L.GRADE
        sel = self.in_rect(self.spec["tarp"], 0.01) & (self.pos[:, 2] > z0 - 0.02) & (self.pos[:, 2] < z0 + 0.40)
        return sel if origin is None else sel & (self.origin == origin)

    def on_walkway(self):
        s = L.SLAB
        sel = self.in_rect(((s["x"][0], s["y"][0]), (s["x"][1], s["y"][1]))) & (self.pos[:, 2] > -0.02) & ~self.on_tarp()
        return sel

    def out_of_block(self):
        return ~self.in_rect(self.block_rect(), 0.0) | (self.pos[:, 2] > L.GRADE + 0.05)

    # ------------------------------------------------------------------ surface
    def heightfield(self, rect, cell=0.01, min_count=3, z_max=L.GRADE + 0.15):
        """Column tops (m) over rect on a `cell` grid: the top of the highest voxel (2 x 2 x 2 particle spacings) holding
        >= min_count particles, refined to the highest particle in it plus the surface offset 0.55 h; NaN where empty.
        Soil more than 0.15 m above grade is not ground (a clod on a tool held over the hole: the post-hole digger parked
        above the stake with soil on its blades read as a hole 1 cm *above* grade, phantom_digger r4)."""
        (x0, y0), (x1, y1) = rect
        sel = self.in_rect(rect, 0.02) & (self.pos[:, 2] < z_max)
        x = self.pos[sel]
        nx, ny = max(1, int(round((x1 - x0)/cell))), max(1, int(round((y1 - y0)/cell)))
        top = np.full((nx, ny), np.nan)
        if not len(x):
            return top, (x0, y0, cell)
        v = self.voxel
        zi = np.floor(x[:, 2]/v).astype(int)
        ix = np.floor((x[:, 0] - x0)/cell).astype(int); iy = np.floor((x[:, 1] - y0)/cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, zi, z = ix[ok], iy[ok], zi[ok], x[ok, 2]
        # voxel occupancy with the 2 cm voxel footprint around each column (column cells are 1 cm)
        vx = np.floor((x[ok, 0] - x0)/v).astype(int); vy = np.floor((x[ok, 1] - y0)/v).astype(int)
        key = (vx.astype(np.int64)*100000 + vy)*100000 + (zi + 50000)
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        dense = cnt[inv.ravel()] >= min_count
        flat = np.full(nx*ny, -np.inf)
        np.maximum.at(flat, ix[dense]*ny + iy[dense], z[dense])
        top = np.where(np.isfinite(flat), flat, np.nan).reshape(nx, ny)
        return top + 0.55*self.h, (x0, y0, cell)

    def depth_at(self, top, origin, xy, r):
        x0, y0, cell = origin
        nx, ny = top.shape
        cx = x0 + (np.arange(nx) + 0.5)*cell
        cy = y0 + (np.arange(ny) + 0.5)*cell
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        m = np.hypot(X - xy[0], Y - xy[1]) <= r
        vals = L.GRADE - top[m]
        vals = np.where(np.isnan(vals), 0.6, vals)            # an empty column inside the block: dug to the floor
        return vals, X[m], Y[m]

    def packing(self, lo, hi, cell=None):
        """Packing fraction (particle volume / cell volume) on a grid of `cell` over the box [lo, hi]."""
        cell = cell or self.voxel
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        n = np.maximum(np.round((hi - lo)/cell).astype(int), 1)
        sel = (self.pos >= lo).all(1) & (self.pos < hi).all(1)
        idx = np.floor((self.pos[sel] - lo)/cell).astype(int)
        idx = np.minimum(idx, n - 1)
        grid = np.zeros(n)
        np.add.at(grid, tuple(idx.T), self.vol[sel])
        return grid/cell**3, lo, cell


# ------------------------------------------------------------------ robot-side checks
def tool_held(env, tool, arms=None):
    """Each arm's two fingertip pads touch the tool (MuJoCo contacts) and its grip is in force mode."""
    w = env.world
    arms = arms or [a for a, rec in env.hold_record.items() if rec["tool"] == tool]
    bodies = {w.scene.bodies[n] for n in (tool, f"{tool}_a", f"{tool}_b", "drill_trigger") if n in w.scene.bodies}
    pads = {}
    for s, label in w.scene.pad_shapes.items():
        pads[int(w.scene.b.shape_body[s])] = label
    touching = {a: set() for a in arms}
    for b1, b2, fn, s1, s2 in w.contact_table():
        for pb, ob in ((b1, b2), (b2, b1)):
            if pb in pads and ob in bodies and fn > 0.5:
                for a in arms:
                    if pads[pb].split("/")[-1].startswith(f"{a}_"):
                        touching[a].add(pb)
    if tool == "drill":
        # the side handle is gripped (left); the pistol grip rests in the right hand, squeezed only to run the motor
        return {a: (len(touching[a]) >= 2 and w.grip_force.get(a, 0.) > 0) if a == "left" else len(touching[a]) >= 1
                for a in arms}
    return {a: len(touching[a]) >= 2 and w.grip_force.get(a, 0.) > 0 for a in arms}


def _pose(env, name):
    p, R = env.world.body_pose(env.world.scene.bodies[name])
    return p, R


def _tilt_deg(R, axis=2):
    return math.degrees(math.acos(max(-1., min(1., float(R[2, axis])))))


def crit(name, value, ok, target, unit=""):
    return dict(name=name, value=None if value is None else (round(float(value), 4) if isinstance(value, (int, float, np.floating)) else value),
                ok=bool(ok), target=target, unit=unit)


def diag(name, value, note, unit=""):
    """A reported reading that does not decide success."""
    if isinstance(value, (int, float, np.floating)):
        value = round(float(value), 4) if np.isfinite(value) else None
    return dict(name=name, value=value, note=note, unit=unit)


def _finite(v):
    return float(v) if v is not None and np.isfinite(v) else None


class VerdictUnavailable(RuntimeError):
    """The verdict could not be decided (a numerical fault inside a counterfactual test): an interruption, not a
    failed task."""


# ------------------------------------------------------------------ the evaluator
class Evaluator:
    def __init__(self, env):
        self.env = env
        self.task = env.spec["task"]
        self.initial_reflexes = len(env.world.reflexes)
        env.world.recorders.append(self._record)
        self.base_off_slab = 0.0
        self.max_tool_drop = {}

    def _record(self, w):
        if w.tick % 100:
            return
        x, y, yaw = w.base_pose()
        # TMR footprint corners (0.76 x 0.55 m about base_link; URDF base hull, rounded): any corner beyond the slab
        c, s = math.cos(yaw), math.sin(yaw)
        off = 0.0
        for dx, dy in ((0.38, 0.275), (0.38, -0.275), (-0.38, 0.275), (-0.38, -0.275)):
            px, py = x + c*dx - s*dy, y + s*dx + c*dy
            off = max(off, L.SLAB["y"][0] - py, py - L.SLAB["y"][1], L.SLAB["x"][0] - px, px - L.SLAB["x"][1])
        self.base_off_slab = max(self.base_off_slab, off)

    def common(self):
        w = self.env.world
        return [crit("no_fault", None, w.fault is None, "simulation finite and guarded"),
                crit("protective_stops", len(w.reflexes), len(w.reflexes) == 0, "0")]

    def progress(self):
        try:
            return self._evaluate(final=False)
        except Exception as exc:                  # progress is advisory; the verdict raises
            return dict(score=0.0, succeeded=False, error=f"{type(exc).__name__}: {exc}")

    def evaluate(self):
        return self._evaluate(final=True)

    # tasks whose success does not depend on protective stops (reported as a diagnostic): a benchmark task says so in
    # its verifier.py (STOPS_REPORTED_ONLY = True); none of the others. Tasks whose instruction states it keep the
    # requirement.
    STOPS_REPORTED_ONLY = ()

    def judge(self, lg, final):
        """The task's own reading of the episode from the ledger lg: (criteria, progress parts[, extra]). A benchmark
        task's is its verifier's verify (tasks/<task>/verifier.py), another task's the method named after it."""
        folder = L.task_verifier(self.task)
        if folder is not None:
            return folder.verify(self, lg, final)
        return getattr(self, "_" + self.task.replace("-", "_"))(lg, final)

    def _evaluate(self, final):
        # common criteria are read first: set-post's verdict runs a counterfactual test that restores the world
        # (read after the restore, no_fault would miss a fault flag the restore cleared)
        common = self.common()
        out = self.judge(Ledger(self.env), final)
        crits, parts = out[0], out[1]
        extra = out[2] if len(out) > 2 else {}
        diagnostics = list(extra.get("diagnostics", ()))
        if self.task in self.STOPS_REPORTED_ONLY or getattr(L.task_verifier(self.task), "STOPS_REPORTED_ONLY", False):
            diagnostics += [diag(c["name"], c["value"], "protective stops in the episode; reported, not a success "
                                 "requirement of this task") for c in common if c["name"] == "protective_stops"]
            common = [c for c in common if c["name"] != "protective_stops"]
        crits = common + crits
        ok = all(c["ok"] for c in crits) and not extra.get("pending", False)
        if "score" in extra:
            score = float(np.clip(extra["score"], 0, 1))
        else:
            score = float(np.clip(np.mean(parts), 0, 1)) if parts else float(ok)
        res = dict(task=self.task, succeeded=ok, score=round(score if not ok else 1.0, 4), criteria=crits,
                   tick=int(self.env.world.tick), time_s=round(self.env.world.sim_time, 4),
                   strikes=list(self.env.world.strikes), reflexes=list(self.env.world.reflexes)[-5:])
        if diagnostics:
            res["diagnostics"] = diagnostics
        res.update(extra.get("fields", {}))
        return res

    # ---------------------------------------------------------------- T3 / T6 (holes)
    def _hole(self, lg, mark, depth_target, tol, dia_range, mid_depth_dia=False, rect_pad=0.4):
        mark = np.asarray(mark)
        rect = ((mark[0] - rect_pad, mark[1] - rect_pad), (mark[0] + rect_pad, mark[1] + rect_pad))
        top, org = lg.heightfield(rect)
        d_c, _, _ = lg.depth_at(top, org, mark, 0.05)
        depth = float(np.mean(d_c))
        d_all, X, Y = lg.depth_at(top, org, mark, rect_pad)
        level = 0.5*depth_target
        area = float(np.sum(d_all >= level))*org[2]**2
        dia = 2*math.sqrt(area/math.pi)
        # firm floor: loose crumbs (displaced loosened soil or spoil) within 30 mm above the floor under the centre
        floor_z = L.GRADE - depth
        r = np.hypot(lg.pos[:, 0] - mark[0], lg.pos[:, 1] - mark[1])
        crumbs = (r < 0.10) & lg.displaced_loose & (lg.pos[:, 2] < floor_z + 0.03) & (lg.pos[:, 2] > floor_z - 0.10)
        m_crumbs = float(lg.mass[crumbs].sum())
        return depth, dia, m_crumbs

    def _spoil(self, lg):
        bed = lg.origin == ORIGIN_BLOCK
        dug = lg.out_of_block() & bed
        m_dug = float(lg.mass[dug].sum())
        m_tarp = float(lg.mass[dug & lg.on_tarp()].sum())
        m_walk = float(lg.mass[dug & lg.on_walkway()].sum())
        return m_dug, m_tarp, m_walk

    def _post_hole(self, lg, final):
        sp = self.env.spec
        mark = np.asarray(sp["target"])
        depth, dia_top, crumbs = self._hole(lg, mark, 0.60, 0.0, (0.2, 0.3), rect_pad=0.3)
        # diameter at mid-depth: columns deeper than half the target
        rect = ((mark[0] - 0.3, mark[1] - 0.3), (mark[0] + 0.3, mark[1] + 0.3))
        top, org = lg.heightfield(rect)
        d_all, _, _ = lg.depth_at(top, org, mark, 0.3)
        dia = 2*math.sqrt(float(np.sum(d_all >= 0.30))*org[2]**2/math.pi)
        m_dug, m_tarp, m_walk = self._spoil(lg)
        frac = m_tarp/max(m_dug, 1e-9)
        c = [crit("depth_at_stake", depth, depth >= 0.60, ">= 0.60 m (24 in)", "m"),
             crit("mid_depth_diameter", dia, 0.20 <= dia <= 0.30, "0.20-0.30 m at 0.30 m depth (~3 x post width)", "m"),
             crit("loose_on_floor", crumbs, crumbs <= 0.3, "<= 0.3 kg of loose crumbs on the floor", "kg"),
             crit("spoil_on_tarp", frac, frac >= 0.8 and m_dug > 0.5, ">= 80 % of the excavated mass", "-")]
        return c, [min(1., depth/0.6), min(1., dia/0.2), max(0., 1 - crumbs), min(1., frac/0.8)]

    # ---------------------------------------------------------------- T4
    def _plant_and_backfill(self, lg, final):
        p, R = _pose(self.env, "rootball")
        top_z = p[2] + P.ROOTBALL["height"]*R[2, 2]
        flare = top_z - L.GRADE
        tilt = _tilt_deg(R)
        rb = P.ROOTBALL["diameter"]/2
        rr = np.hypot(lg.pos[:, 0] - p[0], lg.pos[:, 1] - p[1])
        over = (rr < rb - 0.01) & (lg.pos[:, 2] > top_z + 0.5*lg.h)
        cover = float(lg.mass[over].sum())/(P.MATERIALS["topsoil"]["rho"]*math.pi*(rb - 0.01)**2)     # m of soil
        hole_r = self.env.spec["predug"]["radius"]
        rect = ((p[0] - hole_r - 0.05, p[1] - hole_r - 0.05), (p[0] + hole_r + 0.05, p[1] + hole_r + 0.05))
        top, org = lg.heightfield(rect)
        d_ann, X, Y = lg.depth_at(top, org, p[:2], hole_r)
        ann = np.hypot(X - p[0], Y - p[1]) > rb + 0.02
        surf = float(np.mean(-d_ann[ann])) if ann.any() else float("nan")
        # packing in the annulus between the hole floor and grade (2 cm cells): air pockets below the surface
        floor = L.GRADE - self.env.spec["predug"]["depth"]
        grid, lo, cell = lg.packing((p[0] - hole_r, p[1] - hole_r, floor), (p[0] + hole_r, p[1] + hole_r, L.GRADE - 0.02))
        n = grid.shape
        cx = lo[0] + (np.arange(n[0]) + 0.5)*cell; cy = lo[1] + (np.arange(n[1]) + 0.5)*cell
        X3, Y3 = np.meshgrid(cx, cy, indexing="ij")
        rad = np.hypot(X3 - p[0], Y3 - p[1])
        in_ann = (rad > rb + 0.02) & (rad < hole_r - 0.02)
        cells = grid[in_ann]
        voids = float(np.mean(cells < 0.6)) if cells.size else 1.0
        dens = float(np.mean(np.minimum(cells, 1.2))) if cells.size else 0.0
        c = [crit("rootball_flare", flare, 0.0 <= flare <= 0.05, "top 0-50 mm above grade (CSU / Ohio extension)", "m"),
             crit("rootball_plumb", tilt, tilt <= 5.0, "<= 5 deg", "deg"),
             crit("soil_over_ball", cover, cover <= 0.01, "<= 10 mm of soil over the root ball", "m"),
             crit("backfill_level", surf, abs(surf) <= 0.02 if np.isfinite(surf) else False, "annulus surface within +-20 mm of grade", "m"),
             crit("air_pockets", voids, voids <= 0.03, "<= 3 % of annulus cells below packing 0.6", "-"),
             crit("backfill_density", dens, 0.90 <= dens <= 1.05, "0.90-1.05 of bank packing (firmed, not over-compacted)", "-")]
        return c, [float(0 <= flare <= 0.05), max(0., 1 - tilt/15.), max(0., 1 - cover/0.05),
                   max(0., 1 - abs(surf)/0.1) if np.isfinite(surf) else 0., max(0., 1 - voids/0.3), min(1., dens/0.9)]

    # ---------------------------------------------------------------- T10 / T11 (irrigation lateral)
    def _backfill_pipe(self, lg, final):
        sp = self.env.spec
        pp = sp["pipe"]
        tr = sp["predug"]
        x0, x1 = tr["x"]
        r = pp["r"]
        sand = lg.mat == M.NAMES.index("sand")
        # haunch zone: beside the pipe below its springline, out to 0.08 m from its axis, along the trench
        grid, lo, cell = lg.packing((x0 + 0.05, pp["y"] - 0.10, pp["z"] - r), (x1 - 0.05, pp["y"] + 0.10, pp["z"]), cell=0.02)
        n = grid.shape
        cy = lo[1] + (np.arange(n[1]) + 0.5)*cell
        side = np.abs(cy - pp["y"]) > r + 0.005
        haunch = grid[:, side, :]
        voids = float(np.mean(haunch < 0.6)) if haunch.size else 1.0
        from .metrics import pipe_sand_cover
        cover = pipe_sand_cover(lg.pos[sand], lg.vol[sand], lg.h, (x0+.05, x1-.05),
                                pp['y'], pp['z']+r, r)
        coverage = cover['coverage_fraction']
        rect = ((x0 + 0.05, pp["y"] - tr["width"]/2 + 0.03), (x1 - 0.05, pp["y"] + tr["width"]/2 - 0.03))
        top, org = lg.heightfield(rect)
        surf = L.GRADE - top
        surf = float(np.nanmean(-surf)) if np.isfinite(surf).any() else float("nan")
        w = self.env.world
        c = [crit("haunch_voids", voids, voids <= 0.05, "<= 5 % of haunch cells below packing 0.6", "-"),
             crit("sand_cover", cover, coverage >= .95,
                  ">= 50 mm continuous sand depth across the crown at >=95% of stations; every cell packing >=0.55"),
             crit("surface_at_grade", surf, np.isfinite(surf) and abs(surf) <= 0.02, "within +-20 mm of grade", "m"),
             crit("no_strikes", len(w.strikes), len(w.strikes) == 0, "0 strikes on the lateral")]
        return c, [max(0., 1 - voids/0.5), min(1., coverage/.95), max(0., 1 - abs(surf)/0.2) if np.isfinite(surf) else 0.,
                   float(len(w.strikes) == 0)]

"""Task evaluator: physically meaningful criteria read from the world state.

Everything is measured from the simulated state, never from commands: the soil surface from the particles (column
tops on a 1 cm grid, a column counted where the top voxel holds at least 3 of its nominal 8 particles), masses from
particle masses and where they lie, loosening from each particle's plastic volume ratio J_p against its bank value,
packing from particle volumes per voxel, tool poses and grasps from MuJoCo bodies and contacts, protective stops from
the robot's reflex log, pipe strikes from the strike log. Tolerances are authored and stated with their basis
(params / layout). progress() is cheap (cached per tick); evaluate() is the verdict (`gar verify`): set-post runs its
brace-release-and-push test on a copy of the state and restores it (no history discontinuity).
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

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

    # tasks whose success does not depend on protective stops (reported as a diagnostic): a recovered stop during the
    # work does not change whether the post is set, and set-post's instruction does not ask for it. Tasks whose
    # instruction states it keep the requirement.
    STOPS_REPORTED_ONLY = ("set-post",)

    def _evaluate(self, final):
        # common criteria are read first: set-post's verdict runs a counterfactual test that restores the world
        # (read after the restore, no_fault would miss a fault flag the restore cleared)
        common = self.common()
        fn = getattr(self, "_" + self.task.replace("-", "_"))
        out = fn(Ledger(self.env), final)
        crits, parts = out[0], out[1]
        extra = out[2] if len(out) > 2 else {}
        diagnostics = list(extra.get("diagnostics", ()))
        if self.task in self.STOPS_REPORTED_ONLY:
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

    # ---------------------------------------------------------------- T1
    def _penetrate_marked(self, lg, final):
        sp = self.env.spec
        p, R = _pose(self.env, "spade")
        mark = np.asarray(sp["target"])
        depth = L.GRADE - p[2]
        off = float(np.linalg.norm(p[:2] - mark))
        tilt = _tilt_deg(R)
        held = tool_held(self.env, "spade")
        c = [crit("edge_depth", depth, depth >= 0.15, ">= 0.15 m below grade (2/3 of the 290 mm head)", "m"),
             crit("edge_on_mark", off, off <= 0.03, "<= 30 mm from the mark (1.5 voxels)", "m"),
             crit("blade_vertical", tilt, tilt <= 10.0, "<= 10 deg", "deg"),
             crit("both_hands", held, all(held.values()), "both hands on the spade")]
        return c, [max(0., min(1., depth/0.15)), max(0., 1 - off/0.1), max(0., 1 - tilt/30.), float(all(held.values()))]

    # ---------------------------------------------------------------- T2
    def _spit_to_tarp(self, lg, final):
        bed = lg.origin == ORIGIN_BLOCK
        tarp = lg.on_tarp() & bed
        m_tarp = float(lg.mass[tarp].sum())
        out = lg.out_of_block() & bed & ~tarp
        m_spill = float(lg.mass[out].sum())
        frac = m_spill/max(m_spill + m_tarp, 1e-9)
        held = tool_held(self.env, "spade")
        c = [crit("soil_on_tarp", m_tarp, m_tarp >= 1.5, ">= 1.5 kg of bed soil on the tarp (a half spit)", "kg"),
             crit("spill_fraction", frac, frac <= 0.15 or m_tarp + m_spill < 1e-6, "<= 15 % of the soil lifted out", "-"),
             crit("spade_held", held, all(held.values()), "both hands on the spade")]
        return c, [min(1., m_tarp/1.5), max(0., 1 - frac/0.5), float(all(held.values()))]

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

    def _planting_hole(self, lg, final):
        sp = self.env.spec
        depth, dia, crumbs = self._hole(lg, sp["target"], 0.16, 0.02, (0.33, 0.50))
        m_dug, m_tarp, m_walk = self._spoil(lg)
        frac = m_tarp/max(m_dug, 1e-9)
        c = [crit("depth_at_mark", depth, abs(depth - 0.16) <= 0.02, "0.16 +- 0.02 m (#1 root ball height)", "m"),
             crit("top_diameter", dia, 0.33 <= dia <= 0.50, "0.33-0.50 m where depth >= 80 mm (2-3 x 165 mm)", "m"),
             crit("loose_on_floor", crumbs, crumbs <= 0.3, "<= 0.3 kg of loose crumbs on the floor (ANSI A300: undug base)", "kg"),
             crit("spoil_on_tarp", frac, frac >= 0.8 and m_dug > 0.5, ">= 80 % of the excavated mass", "-"),
             crit("walkway_clean", m_walk, m_walk <= 0.2, "<= 0.2 kg on the walkway", "kg")]
        return c, [min(1., depth/0.16) if depth <= 0.16 else max(0., 1 - (depth - 0.16)/0.1), min(1., dia/0.33),
                   max(0., 1 - crumbs/1.0), min(1., frac/0.8), max(0., 1 - m_walk/1.0)]

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

    # ---------------------------------------------------------------- T5
    def _cable_trench(self, lg, final):
        (x0, y0), (x1, y1) = self.env.spec["target_line"]
        rect = ((x0 - 0.05, y0 - 0.25), (x1 + 0.05, y0 + 0.25))
        top, org = lg.heightfield(rect)
        depth = L.GRADE - top
        depth = np.where(np.isnan(depth), 0.5, depth)
        xs = org[0] + (np.arange(top.shape[0]) + 0.5)*org[2]
        ys = org[1] + (np.arange(top.shape[1]) + 0.5)*org[2]
        along = (xs >= x0) & (xs <= x1)
        prof = depth[along]
        deep = prof.max(axis=1)
        # centre of the deepest band (>= 80 % of the row's depth): the argmax of a flat-bottomed trench is its first
        # column, an edge (a correct 12 cm trench on the line would fail 'on_line')
        band = prof >= 0.8*deep[:, None]
        centre = (band*ys[None, :]).sum(1)/np.maximum(band.sum(1), 1)
        centre_off = np.abs(centre - y0)
        ok_depth = float(np.mean(deep >= 0.16))
        widths = []
        for k in range(prof.shape[0]):
            half = deep[k]/2
            row = prof[k] >= max(half, 0.02)
            widths.append(row.sum()*org[2])
        widths = np.asarray(widths)
        ok_width = float(np.mean((widths >= 0.06) & (widths <= 0.20)))
        ok_centre = float(np.mean(centre_off <= 0.05))
        m_dug, m_tarp, m_walk = self._spoil(lg)
        frac = m_tarp/max(m_dug, 1e-9)
        c = [crit("depth_coverage", ok_depth, ok_depth >= 0.9, ">= 90 % of the 1.5 m at >= 0.16 m (NEC 300.5: 150 mm cover)", "-"),
             crit("width_coverage", ok_width, ok_width >= 0.9, "6-20 cm wide at half depth over >= 90 %", "-"),
             crit("on_line", ok_centre, ok_centre >= 0.9, "deepest point within 5 cm of the line over >= 90 %", "-"),
             crit("spoil_on_tarp", frac, frac >= 0.9 and m_dug > 0.5, ">= 90 % of the spoil on the lawn-side tarp", "-"),
             crit("walkway_clean", m_walk, m_walk <= 0.2, "<= 0.2 kg on the walkway", "kg"),
             crit("base_on_slab", self.base_off_slab, self.base_off_slab <= 0.0, "base footprint never beyond the slab", "m")]
        return c, [ok_depth, ok_width, ok_centre, min(1., frac/0.9), max(0., 1 - m_walk), float(self.base_off_slab <= 0)]

    # ---------------------------------------------------------------- T7
    # set-post. The instruction states the job: backfill the braced post with the
    # tarp soil, tamp it firm, back to grade, so the post stands plumb and solid on its own once the braces come off. The
    # verdict reads that outcome and nothing about the method (no lift, tool or order check):
    #   backfill_to_grade  per 1 cm column over the dug footprint (the lip outline of the hand-dug wall at grade less
    #                      lip_inset, the post excluded): a share within [low, crown] of grade (POST_FINISH)
    #   surrounding_grade  the yard surface round the hole is not dug down
    #   fill_from_tarp     native soil moved out of the yard beyond the lip stays within a scuff allowance: the fill is
    #                      "the soil on the tarp", not the lawn
    #   post_free          the robot and the tools it holds are off the post and off the fill round it, and no tool is
    #                      buried in the hole: the post is judged standing on its own
    #   post_plumb         braces released: the post's tilt after standing unloaded, and again after each push is
    #                      taken off (a post a person's lean leaves leaning is neither plumb nor solid)
    #   post_support       from the released state, a person's lean toward +x, -x, +y, -y: the post axis at grade stays
    #                      within the limit during the push and over a further creep_s at its measured creep rate
    # progress (score below success): 1/2 placement + 1/2 support x plumb, all from end-state readings.
    def _post_site(self):
        """Fixed geometry of the pre-dug post hole, cached: its lip outline at grade from the hand-dug wall the hole was
        cut with (layout.dug_wall), and the cavity volume from the gravel top to grade around the post (nominal pose)."""
        if getattr(self, "_site", None) is None:
            sp = self.env.spec
            cut = next(c for c in sp["cuts"] if c["kind"] == "cylinder")
            mx, my = sp["target"]
            pd = sp["predug"]
            th = np.linspace(-math.pi, math.pi, 721)
            rr = np.arange(0.0, 0.30, 0.0005)
            pts = np.stack([mx + rr[None, :]*np.cos(th)[:, None], my + rr[None, :]*np.sin(th)[:, None],
                            np.full((len(th), len(rr)), L.GRADE - 0.005)], -1).reshape(-1, 3)
            lip = (L.dug_wall(pts, cut).reshape(len(th), len(rr))*rr[None, :]).max(1)
            g = np.arange(-0.25, 0.25, 0.005) + 0.0025
            z_bot = L.GRADE - pd["depth"] + pd["base"]
            zz = np.arange(z_bot, L.GRADE, 0.005) + 0.0025
            X, Y, Z = np.meshgrid(g + mx, g + my, zz, indexing="ij")
            G = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            a = P.POST["section"]/2
            cav = L.dug_wall(G, cut) & ((np.abs(G[:, 0] - mx) > a) | (np.abs(G[:, 1] - my) > a))
            self._site = dict(cut=cut, centre=np.array([mx, my], float), th=th, lip=lip,
                              cavity_m3=float(cav.sum())*0.005**3, gravel_top=z_bot)
        return self._site

    @staticmethod
    def _lip_at(site, x, y):
        """Radius from the hole centre and the lip radius in that direction."""
        dx, dy = np.asarray(x, float) - site["centre"][0], np.asarray(y, float) - site["centre"][1]
        return np.hypot(dx, dy), np.interp(np.arctan2(dy, dx), site["th"], site["lip"])

    def _post_surface(self, lg, site, p, R):
        """Column tops (1 cm) over the soil block: the dug footprint without the post, and the yard beyond the lip."""
        F = P.POST_FINISH
        sp = self.env.spec
        b = sp["block"]
        rect = ((b["lo"][0], b["lo"][1]), (b["hi"][0], b["hi"][1]))
        top, (x0, y0, cell) = lg.heightfield(rect)
        nx, ny = top.shape
        X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5)*cell, y0 + (np.arange(ny) + 0.5)*cell, indexing="ij")
        s = np.where(np.isnan(top), -0.6, top - L.GRADE)          # an empty column inside the block: dug to the floor
        self._last_top = (np.where(np.isnan(top), L.GRADE - 0.6, top), (x0, y0, cell))   # for _post_free's "buried"
        r, lip = self._lip_at(site, X, Y)
        g = p + R @ np.array([0., 0., (L.GRADE - p[2])/max(float(R[2, 2]), 1e-6)])   # the post's axis at grade
        yaw = math.atan2(R[1, 0], R[0, 0])
        u = (X - g[0])*math.cos(yaw) + (Y - g[1])*math.sin(yaw)
        v = -(X - g[0])*math.sin(yaw) + (Y - g[1])*math.cos(yaw)
        on_post = np.maximum(np.abs(u), np.abs(v)) < P.POST["section"]/2 + F["post_margin"]
        foot = (r < lip - F["lip_inset"]) & ~on_post
        sf = s[foot]
        yard = (r > lip + F["lip_margin"]) & (X > rect[0][0] + 0.02) & (X < rect[1][0] - 0.02) & \
            (Y > rect[0][1] + 0.02) & (Y < rect[1][1] - 0.02)
        st = sp.get("props", {}).get("stake")
        if st:                                                    # the survey stake's static box (scene._prop)
            sx, sy = st["pos"]
            yard &= ~((np.abs(X - sx) < 0.05) & (Y > sy - 0.33) & (Y < sy - 0.24))
        sy_ = s[yard]
        return dict(share=float(np.mean((sf >= F["low"]) & (sf <= F["crown"]))) if sf.size else 0.0,
                    below=float(np.mean(sf < F["low"])) if sf.size else 1.0,
                    above=float(np.mean(sf > F["crown"])) if sf.size else 0.0,
                    mean_m=float(np.mean(sf)) if sf.size else float("nan"),
                    p10_m=float(np.percentile(sf, 10)) if sf.size else float("nan"),
                    p90_m=float(np.percentile(sf, 90)) if sf.size else float("nan"), columns=int(foot.sum()),
                    yard_share=float(np.mean(sy_ >= F["yard_low"])) if sy_.size else 0.0,
                    yard_p5_m=float(np.percentile(sy_, 5)) if sy_.size else float("nan"), yard_columns=int(yard.sum()))

    def _post_placement(self, lg, site):
        """Soil put back: the bank-equivalent volume of tarp spoil (not the gravel base) inside the dug cavity up to
        grade, less native yard soil dug out from beyond the lip in excess of the scuff allowance, over PLACEMENT_FULL of
        the cavity's volume. Counting particles, not surface height, so tamping a lift in place neither costs nor earns
        placement; soil above grade (a crown or a heap) neither adds nor subtracts, so a flat and a crowned finish read
        the same and overfilling before levelling is not penalised on the way (a heap fails backfill_to_grade). Lawn soil
        used as fill earns nothing as tarp soil and its removal costs its volume, i.e. it counts twice against
        placement."""
        F = P.POST_FINISH
        h3 = float(lg.h)**3                                       # one particle's volume at bank packing
        pile = np.flatnonzero((lg.origin == ORIGIN_PILE) & (lg.mat != M.NAMES.index("gravel")))
        x = lg.pos[pile]
        v_in = float(np.count_nonzero(L.dug_wall(x, site["cut"]) & (x[:, 2] <= L.GRADE)))*h3
        blk = np.flatnonzero(lg.origin == ORIGIN_BLOCK)
        r0, lip0 = self._lip_at(site, lg.pos0[blk, 0], lg.pos0[blk, 1])
        xb = lg.pos[blk]
        # yard soil from beyond the lip that is now in the hole (used as fill); lawn damage elsewhere is read by
        # surrounding_grade
        dug = (r0 > lip0 + F["lip_margin"]) & L.dug_wall(xb, site["cut"]) & (xb[:, 2] <= L.GRADE + F["crown"])
        v_dug = float(np.count_nonzero(dug))*h3
        v_cost = max(0.0, v_dug - F["lawn_allowance_m3"])
        frac = float(np.clip((v_in - v_cost)/(F["placement_full"]*site["cavity_m3"]), 0., 1.))
        return frac, dict(tarp_soil_in_hole_L=round(v_in*1e3, 2), yard_soil_dug_L=round(v_dug*1e3, 2),
                          cavity_L=round(site["cavity_m3"]*1e3, 2))

    def _post_tools(self):
        """{newton body: tool name} of the episode's tools (held and kit), and the subset a finger pad touches now."""
        w, sp = self.env.world, self.env.spec
        tools = {}
        for t in list(sp.get("held", {}).values()) + list(sp.get("kit", [])):
            for n in (t, f"{t}_a", f"{t}_b") + (("auger", "drill_trigger") if t == "drill" else ()):
                if n in w.scene.bodies:
                    tools[int(w.scene.bodies[n])] = t
        mj_held = w.held_bodies()
        held = {i for i in tools if w._mj_body.get(i) in mj_held}
        return tools, held

    def _robot_post_force(self):
        """Largest MuJoCo normal force between the post and any robot body (N)."""
        w = self.env.world
        post = w.scene.bodies["post"]
        robot = getattr(self, "_robot_set", None)
        if robot is None:
            robot = self._robot_set = {int(i) for i in w.scene.robot_bodies}
        f = 0.0
        for b1, b2, fn, _, _ in w.contact_table():
            if post in (b1, b2) and (b2 if b1 == post else b1) in robot:
                f = max(f, fn)
        return f

    def _post_free(self, site):
        """Whether the post is judged on its own, read on the end state.
        - The robot and the tools it holds are off the post (MuJoCo contacts above contact_n).
        - The robot is off the soil that holds the post: no robot part below grade + clear_height on the soil block
          within lip + clear_margin (0.30 m: about the depth of the post's lateral pivot below grade, 0.31 m, so the
          soil wedge the lean loads near grade). The robot cannot
          be parked for the test, so a robot there would be part of the foundation measured.
        - No tool, held or not, is buried in the hole: a part more than in_hole_depth below grade inside the lip with
          soil over it (column top more than 10 mm above the part; a tool lying in a low spot of the finished surface is
          not buried). A buried tool is part of the fill (a shim).
        A tool the robot still holds resting on the fill, or a free tool against the post, is not part of the
        foundation: the load test parks every unburied tool away from the post, so it can neither prop the post nor
        load the fill, and it does not fail the post.
        Collision-geom boxes are sampled at 1 cm."""
        F = P.POST_FINISH
        w = self.env.world
        post = w.scene.bodies["post"]
        tools, held = self._post_tools()
        robot = {int(i) for i in w.scene.robot_bodies}
        touching = set()
        for b1, b2, fn, _, _ in w.contact_table():
            if fn > F["contact_n"] and post in (b1, b2):
                o = b2 if b1 == post else b1
                if o in robot:
                    touching.add("robot")
                elif o in held:
                    touching.add(tools[o] + " (held)")
        m, d = w.mjm, w.mjd
        owner = {int(w._mj_body[i]): "robot" for i in robot if i in w._mj_body}
        owner.update({int(w._mj_body[i]): t for i, t in tools.items() if i in w._mj_body})
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        z_hole, z_clear = L.GRADE - F["in_hole_depth"], L.GRADE + F["clear_height"]
        blk = self.env.spec["block"]
        top = getattr(self, "_last_top", None)
        in_hole, on_fill = set(), set()
        for gi in range(m.ngeom):
            name = owner.get(int(m.geom_bodyid[gi]))
            if name is None or (int(m.geom_contype[gi]) == 0 and int(m.geom_conaffinity[gi]) == 0):
                continue
            is_robot = name == "robot"
            c, hs = np.asarray(m.geom_aabb[gi, :3], float), np.asarray(m.geom_aabb[gi, 3:], float)
            Rg, pg = np.asarray(d.geom_xmat[gi], float).reshape(3, 3), np.asarray(d.geom_xpos[gi], float)
            z_cut = z_clear if is_robot else z_hole
            if float((pg + (c + corners*hs) @ Rg.T)[:, 2].min()) >= z_cut:
                continue
            n = np.clip(np.ceil(2*hs/0.01).astype(int) + 1, 2, 160)
            axes = [np.linspace(-hs[k], hs[k], n[k]) for k in range(3)]
            pts = pg + (np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + c) @ Rg.T
            pts = pts[pts[:, 2] < z_cut]
            r, lip = self._lip_at(site, pts[:, 0], pts[:, 1])
            deep = (r < lip) & (pts[:, 2] < z_hole)
            if top is not None and deep.any():
                grid, (x0, y0, cell) = top
                ix = np.clip(((pts[:, 0] - x0)/cell).astype(int), 0, grid.shape[0] - 1)
                iy = np.clip(((pts[:, 1] - y0)/cell).astype(int), 0, grid.shape[1] - 1)
                deep &= grid[ix, iy] > pts[:, 2] + 0.01                # soil over the part
            if deep.any():
                in_hole.add(name)
            if is_robot:
                on_block = (pts[:, 0] > blk["lo"][0]) & (pts[:, 0] < blk["hi"][0]) & \
                    (pts[:, 1] > blk["lo"][1]) & (pts[:, 1] < blk["hi"][1])
                if np.any(on_block & (r < lip + F["clear_margin"])):
                    on_fill.add(name)
        robot_in = "robot" in touching | in_hole | on_fill
        buried = sorted(n for n in in_hole if n != "robot")
        return dict(ok=not touching and not in_hole and not on_fill, touching=sorted(touching),
                    in_hole=sorted(in_hole), on_fill=sorted(on_fill - in_hole),
                    # whether the foundation can still be measured on a copy with the unburied tools parked
                    measurable=not robot_in and not buried)

    @staticmethod
    def post_push_impulse(t):
        """Load impulse (N s) of the test push up to time t: ramp over ramp_s, then held."""
        T = P.POST_TEST
        F, r = T["force"], T["ramp_s"]
        return F*t*t/(2*r) if t <= r else F*r/2 + F*(t - r)

    @classmethod
    def post_support_credit(cls, cross_t):
        """Support part of the progress for one push direction: the share of the notional lean's load impulse (the
        push of ramp_s + hold_s, continued for creep_s) carried before the post axis at grade reached the limit; 1 if
        it never did (measured or projected). Continuous and monotone in the time the post holds; ~0 for a post that
        runs away at once."""
        T = P.POST_TEST
        horizon = T["ramp_s"] + T["hold_s"] + T["creep_s"]
        if cross_t is None:
            return 1.0
        return float(np.clip(cls.post_push_impulse(min(cross_t, horizon))/cls.post_push_impulse(horizon), 0., 1.))

    @staticmethod
    def post_plumb_credit(tilt):
        """Plumb factor of the progress: 1 within plumb_deg, falling linearly to 0 at plumb_zero_deg."""
        T = P.POST_TEST
        if tilt is None or not np.isfinite(tilt):
            return 0.0
        return float(np.clip((T["plumb_zero_deg"] - tilt)/(T["plumb_zero_deg"] - T["plumb_deg"]), 0., 1.))

    def _set_post(self, lg, final):
        F, T = P.POST_FINISH, P.POST_TEST
        site = self._post_site()
        p, R = _pose(self.env, "post")
        surf = self._post_surface(lg, site, p, R)
        A, place = self._post_placement(lg, site)
        free = self._post_free(site)
        c = [crit("backfill_to_grade", surf["share"], surf["share"] >= F["share"],
                  f">= {F['share']:.0%} of the dug footprint's 1 cm columns within {F['low']*1e3:+.0f}/"
                  f"{F['crown']*1e3:+.0f} mm of grade (a slight crown allowed); the post section + "
                  f"{F['post_margin']*1e3:.0f} mm and the outer {F['lip_inset']*1e3:.0f} mm of the ragged lip are not read",
                  "-"),
             crit("surrounding_grade", surf["yard_share"], surf["yard_share"] >= F["yard_share"],
                  f">= {F['yard_share']:.0%} of the yard's columns {F['lip_margin']*1e3:.0f} mm or more beyond the lip "
                  f"no lower than {F['yard_low']*1e3:+.0f} mm: the lawn round the hole not dug down", "-"),
             crit("fill_from_tarp", place["yard_soil_dug_L"], place["yard_soil_dug_L"] <= F["lawn_allowance_m3"]*1e3,
                  f"<= {F['lawn_allowance_m3']*1e3:.1f} L of native yard soil from beyond the lip in the hole (edge "
                  "crumbs): the backfill is the soil on the tarp, not the lawn", "L"),
             crit("post_free", dict(touching=free["touching"], on_fill=free["on_fill"], in_hole=free["in_hole"]),
                  free["ok"], f"robot and held tools off the post; robot off the soil within "
                  f"{F['clear_margin']*1e3:.0f} mm of the lip; no tool buried (more than {F['in_hole_depth']*1e3:.0f} mm "
                  "below grade with soil over it) in the hole: the post is judged on its own")]
        diags = [diag("soil_placed", A, f"tarp soil in the hole up to grade, bank-equivalent, over "
                      f"{F['placement_full']:.0%} of the cavity, less yard soil in the hole beyond the "
                      f"{F['lawn_allowance_m3']*1e3:.1f} L allowance ({place['tarp_soil_in_hole_L']} L in, "
                      f"{place['yard_soil_dug_L']} L of yard soil in the hole, cavity {place['cavity_L']} L)", "-"),
                 diag("braced_tilt", _tilt_deg(R), "the post's tilt with the braces on (not scored: the braces, not the "
                      "backfill, hold it)", "deg"),
                 diag("surface", dict(mean_m=round(surf["mean_m"], 4), p10_m=round(surf["p10_m"], 4),
                                      p90_m=round(surf["p90_m"], 4), below=round(surf["below"], 3),
                                      above=round(surf["above"], 3), columns=surf["columns"],
                                      yard_p5_m=round(surf["yard_p5_m"], 4), yard_columns=surf["yard_columns"]),
                      "dug footprint and yard column statistics, relative to grade")]
        if not final:
            # advisory (owner-side state and per-step reward): placement only; support and plumb need the verdict's
            # counterfactual test, so the advisory never reports success
            return c, [A], dict(score=0.5*A, pending=True, diagnostics=diags,
                                fields=dict(progress=round(0.5*A, 4), verdict_pending=True,
                                            note="advisory: placement only; the post's plumb and support are measured "
                                                 "by the verdict (gar verify) on the released post"))
        res, tilt, d, support_ok = None, None, None, False
        self.last_load_test = None
        S = U = 0.0
        if free["measurable"]:
            res = self.last_load_test = self.post_load_test()
            if res["fault"]:
                raise VerdictUnavailable(f"set-post verdict unavailable: numerical fault in the counterfactual load test "
                                         f"before the post's reading was decided ({res['fault']})")
            rel = res.get("released") or {}
            tilts = [rel.get("tilt_deg")] + [v for v in res["plumb_reading_deg"].values() if v is not None]
            tilt = max(t for t in tilts if t is not None) if rel else None
            d = res["max_ground_deflection_m"]
            pushed = res["ground_deflection_m"]
            support_ok = (len(pushed) == 4 and not res["robot_contact"]
                          and all(v is None for v in res["cross_t"].values()) and d <= T["max_ground_deflection"])
            S = min((0.0 if k in res["robot_contact"] or "release" in res["robot_contact"] else
                     self.post_support_credit(res["cross_t"][k])) for k in pushed) if len(pushed) == 4 else 0.0
            U = self.post_plumb_credit(tilt)
        c.append(crit("post_plumb", _finite(tilt), tilt is not None and tilt <= T["plumb_deg"],
                      f"<= {T['plumb_deg']:.0f} deg from vertical with the braces released, after {T['release_s']} s "
                      f"standing unloaded, and in every push direction: the lean the push leaves, projected over the "
                      f"same {T['ramp_s'] + T['hold_s'] + T['creep_s']:.0f} s lean as the support reading (a push is "
                      f"unloaded only where its projected lean passes {T['plumb_deg']:.0f} deg; one that gives way is "
                      "read at its lean when it gave way); worst reading", "deg"))
        c.append(crit("post_support", _finite(d), support_ok,
                      f"<= {T['max_ground_deflection']*1e3:.1f} mm horizontal motion of the post at grade (IBC 1806.3.4 "
                      f"ground-surface motion) under {T['force']:.0f} N at {T['height']:.0f} m above grade toward +x, "
                      f"-x, +y and -y, braces released: peak over a {T['ramp_s'] + T['hold_s']:.1f} s push and the "
                      f"motion projected {T['creep_s']:.0f} s further at its last {T['creep_window_s']} s creep rate; "
                      "no robot contact with the post during the test", "m"))
        progress = 0.5*A + 0.5*S*U
        diags += [diag("support_credit", S, "share of the notional lean's load impulse carried within the limit, "
                       "worst direction (1 = never reached)", "-"),
                  diag("plumb_credit", U, f"1 within {T['plumb_deg']} deg, 0 at {T['plumb_zero_deg']} deg (worst of "
                       "released and after-push tilt)", "-")]
        fields = dict(progress=round(progress, 4))
        if res is not None:
            diags.append(diag("push_point_deflection", res["max_deflection_m"], f"peak horizontal motion {T['height']} m "
                              "above grade, worst direction (reported only)", "m"))
            fields["load_test"] = dict(
                released_tilt_deg=_finite((res.get("released") or {}).get("tilt_deg")),
                toppled=bool((res.get("released") or {}).get("toppled")), parked_tools=res["parked_tools"],
                ground_deflection_m={k: _finite(v) for k, v in res["ground_deflection_m"].items()},
                projected_ground_m={k: _finite(v) for k, v in res["projected_ground_m"].items()},
                cross_t={k: _finite(v) for k, v in res["cross_t"].items()},
                push_point_deflection_m={k: _finite(v) for k, v in res["deflection_m"].items()},
                creep_m_s={k: _finite(v) for k, v in res["creep_m_s"].items()},
                loaded_tilt_deg={k: _finite(v) for k, v in res["loaded_tilt_deg"].items()},
                residual_tilt_deg={k: _finite(v) for k, v in res["residual_tilt_deg"].items()},
                residual_measured_deg={k: _finite(v) for k, v in res["residual_measured_deg"].items()},
                plumb_reading_deg={k: _finite(v) for k, v in res["plumb_reading_deg"].items()},
                fault_after_fail=res.get("fault_after_fail"),
                robot_contact=res["robot_contact"], seconds=res["seconds"], stopped=res["stopped"],
                wall_s=res["wall_s"])
        else:
            fields["load_test"] = dict(skipped="not measurable: " + ("a tool is buried in the hole" if any(
                n != "robot" for n in free["in_hole"]) else "the robot is on the post or the fill round it"))
        # a non-success never ties a success
        return c, [A, S*U], dict(score=min(progress, 0.99), diagnostics=diags, fields=fields)

    def post_load_test(self):
        """Set-post's verdict test, on a checkpoint copy of the state that is restored afterwards (POST_TEST).

        0. Every tool that is not buried in the hole is parked on the lawn far from the post, held or not: a tool is not
           part of the foundation, so it can neither prop the post nor stand in for fill.
        1. Release: the braces come off and the post stands unloaded for release_s; its tilt then is its plumb on its
           own. A released post already past plumb_zero_deg is not pushed.
        2. Push, from that released state (checkpointed): a phantom horizontal force at `height` above grade, ramped to
           `force` over ramp_s and held hold_s, toward +x, -x, +y and -y, each restored from the same released state.
           Readings per branch: the peak horizontal motion of the post axis at grade from the released pose (and of the
           push point); the time it first exceeds the limit (the branch ends there); otherwise the creep rate over the
           last creep_window_s and the motion projected creep_s further at that rate, which stands for the rest of a
           person's few-second lean (a steady creep would satisfy a settle rule).
        3. After a branch that stays within the limit with the post leaning more than plumb_deg under the load, the
           load is ramped off over unload_s and the post rests rest_s; its tilt then is the lean the push left
           (nearly all of the loaded lean stays in this world). A branch that never leaned past
           plumb_deg cannot leave more, so it is not unloaded.
        The robot cannot be parked: post-robot contact is read every 10 ticks of the test; a branch where the robot
        touches the post above contact_n is not a measurement of the foundation and ends there.
        Counterfactual: recorders and fixtures are detached while it runs; the state, the world's fault flag, clocks and
        counters are restored afterwards. A fault before a branch is decided is returned in `fault`; the evaluator then
        reports the verdict as unavailable rather than as a failed foundation."""
        env, w = self.env, self.env.world
        T, Fn = P.POST_TEST, P.POST_FINISH
        DT = 0.0005
        b = w.scene.bodies["post"]
        com = np.asarray(w.model.body_com.numpy()[b], float)
        t_wall = time.perf_counter()
        site = self._post_site()
        free = self._post_free(site)
        tools, _ = self._post_tools()
        park = [i for i, t in tools.items() if t not in {n.replace(" (held)", "") for n in free["in_hole"]}]
        snap = env.checkpoint()
        recorders, fixture0, fault0 = list(w.recorders), w.fixture, w.fault
        timing0, soil_timing0 = dict(w.timing), dict(w.soil.timing)
        contacts0, speed0 = w.strike_contacts, w.soil.max_speed
        mj_time0 = float(w.mjd.time)
        res = dict(force_n=T["force"], height_m=T["height"], fault=None, released=None, deflection_m={},
                   ground_deflection_m={}, projected_ground_m={}, cross_t={}, creep_m_s={}, loaded_tilt_deg={},
                   residual_tilt_deg={}, residual_measured_deg={}, plumb_reading_deg={}, robot_contact={}, seconds={},
                   stopped=None, counterfactual_branches={},
                   parked_tools=sorted({tools[i] for i in park}),
                   branch_label="Tools parked, brace release, then independent push tests each restored from the same "
                                "released state; counterfactual, not episode time")
        lying = Rotation.from_euler("y", -90, degrees=True).as_matrix()
        w.recorders.clear()
        try:
            env.restore(snap)
            for k, i in enumerate(sorted(park)):
                name = next(n for n, j in w.scene.bodies.items() if j == i)
                env._place_body(name, np.array([-2.5 + 0.7*k, -3.3, L.GRADE + 0.15]), lying, at_rest=True)
            if park:
                env._fk()
            w.release_braces()
            w.fixture = None
            p0, R0 = w.body_pose(b)
            cz = max(float(R0[2, 2]), 1e-6)
            loc_g = np.array([0., 0., (L.GRADE - p0[2])/cz])                 # the post axis at grade (body frame)
            loc_h = np.array([0., 0., (L.GRADE + T["height"] - p0[2])/cz])   # ... at the push height
            g_start = p0 + R0 @ loc_g
            rows, toppled = [], False
            try:
                for k in range(1, int(round(T["release_s"]/DT)) + 1):
                    w.step()
                    if k % 20 == 0:
                        p1, R1 = w.body_pose(b)
                        rows.append(dict(t=round(k*DT, 4), tilt_deg=_tilt_deg(R1),
                                         ground_m=(p1 + R1 @ loc_g - g_start).tolist()))
                        if rows[-1]["tilt_deg"] > T["plumb_zero_deg"]:
                            toppled = True
                            break
                        f_robot = self._robot_post_force()
                        if f_robot > Fn["contact_n"] and "release" not in res["robot_contact"]:
                            res["robot_contact"]["release"] = dict(t=round(k*DT, 4), force_n=round(f_robot, 2))
            except Exception as exc:
                res["fault"] = f"release: {type(exc).__name__}: {exc}"
                return res
            p1, R1 = w.body_pose(b)
            res["released"] = dict(start_tilt_deg=_tilt_deg(R0), tilt_deg=_tilt_deg(R1), toppled=toppled,
                                   seconds=round(len(rows)*20*DT, 4), rows=rows)
            if toppled:
                res["stopped"] = f"the released post tilted past {T['plumb_zero_deg']} deg unloaded; not pushed"
                return res
            rel = env.checkpoint()
            g0, h0 = p1 + R1 @ loc_g, p1 + R1 @ loc_h
            n_push = int(round((T["ramp_s"] + T["hold_s"])/DT))
            win = int(round(T["creep_window_s"]/DT))//10                   # recorded rows in the creep window
            limit = T["max_ground_deflection"]
            for name, dvec in (("+x", (1., 0., 0.)), ("-x", (-1., 0., 0.)), ("+y", (0., 1., 0.)), ("-y", (0., -1., 0.))):
                env.restore(rel)
                t0 = w.tick
                dvec = np.asarray(dvec)
                phase = dict(off=None)

                def fixture(world, dvec=dvec, t0=t0, phase=phase):
                    t = (world.tick - t0)*DT
                    scale = min(1.0, t/T["ramp_s"]) if phase["off"] is None else \
                        max(0.0, 1.0 - (t - phase["off"])/T["unload_s"])
                    F = T["force"]*scale*dvec
                    _, R = world.body_pose(b)
                    world.body_wrench[b, :3] += F.astype(np.float32)
                    world.body_wrench[b, 3:] += np.cross(R @ (loc_h - com), F).astype(np.float32)
                    return world.body_wrench
                w.fixture = fixture
                rows = []
                res["counterfactual_branches"][name] = rows
                peak_g = peak_h = tilt_peak = 0.0
                cross, contact, k = None, None, 0
                gd = np.zeros(2)
                try:
                    while k < n_push:
                        w.step()
                        k += 1
                        p2, R2 = w.body_pose(b)
                        gd, hd = (p2 + R2 @ loc_g - g0)[:2], (p2 + R2 @ loc_h - h0)[:2]
                        peak_g, peak_h = max(peak_g, float(np.hypot(*gd))), max(peak_h, float(np.hypot(*hd)))
                        tilt_peak = max(tilt_peak, _tilt_deg(R2))
                        if peak_g > limit:
                            cross = k*DT
                            break
                        if k % 10:
                            continue
                        rows.append(dict(t=round(k*DT, 4), force_N=T["force"]*min(1., k*DT/T["ramp_s"]),
                                         ground_m=gd.tolist(), push_point_m=hd.tolist(), tilt_deg=_tilt_deg(R2)))
                        f_robot = self._robot_post_force()
                        if f_robot > Fn["contact_n"]:
                            contact = dict(t=round(k*DT, 4), force_n=round(f_robot, 2))
                            break
                    proj, creep, residual, measured, plumb_read = None, None, None, None, None
                    tilt_end = _tilt_deg(w.body_pose(b)[1])
                    if cross is not None:                     # crossed during the push: its lean when it gave way
                        plumb_read = tilt_end
                    if cross is None and contact is None:
                        g_hist = [np.asarray(r["ground_m"]) for r in rows]
                        creep = float(np.hypot(*(g_hist[-1] - g_hist[-1 - win])))/T["creep_window_s"]
                        proj = float(np.hypot(*gd)) + creep*T["creep_s"]
                        # the lean projected over the same few-second lean as the motion
                        rate = max(0.0, (rows[-1]["tilt_deg"] - rows[-1 - win]["tilt_deg"])/T["creep_window_s"])
                        tilt_proj = tilt_end + rate*T["creep_s"]
                        if proj > limit:
                            cross = k*DT + (limit - float(np.hypot(*gd)))/max(creep, 1e-12)
                            plumb_read = tilt_end + rate*(cross - k*DT)   # its lean when it would give way
                        elif tilt_proj <= T["plumb_deg"]:
                            plumb_read = tilt_proj                # it cannot leave more than it leans
                        else:
                            phase["off"] = k*DT
                            for _ in range(int(round((T["unload_s"] + T["rest_s"])/DT))):
                                w.step()
                                k += 1
                                if k % 10 == 0:
                                    p2, R2 = w.body_pose(b)
                                    gd2 = (p2 + R2 @ loc_g - g0)[:2]
                                    rows.append(dict(t=round(k*DT, 4), force_N=T["force"]*max(
                                        0., 1. - (k*DT - phase["off"])/T["unload_s"]), ground_m=gd2.tolist(),
                                        push_point_m=(p2 + R2 @ loc_h - h0)[:2].tolist(), tilt_deg=_tilt_deg(R2)))
                                    f_robot = self._robot_post_force()
                                    if f_robot > Fn["contact_n"]:
                                        contact = dict(t=round(k*DT, 4), force_n=round(f_robot, 2))
                                        break
                            if contact is None:
                                measured = _tilt_deg(w.body_pose(b)[1])
                                # the lean left after the 1 s push, scaled to the projected lean: the share left
                                # after unloading held at 0.86-0.88 for 1 s and 3 s pushes
                                residual = plumb_read = measured*tilt_proj/max(tilt_end, 1e-9)
                except Exception as exc:                     # e.g. the post thrown out of the world
                    if any(v is not None for v in res["cross_t"].values()):
                        res["fault_after_fail"] = f"{name}: {type(exc).__name__}: {exc}"   # already failed: a verdict
                    else:
                        res["fault"] = f"{name}: {type(exc).__name__}: {exc}"
                    break
                res["ground_deflection_m"][name], res["deflection_m"][name] = peak_g, peak_h
                res["projected_ground_m"][name], res["creep_m_s"][name] = proj, creep
                res["cross_t"][name], res["loaded_tilt_deg"][name] = cross, tilt_peak
                res["residual_tilt_deg"][name] = residual
                res["residual_measured_deg"][name] = measured
                res["plumb_reading_deg"][name] = plumb_read
                if contact:
                    res["robot_contact"][name] = contact
                res["seconds"][name] = round(k*DT, 4)
            return res
        finally:
            env.restore(snap)
            w.fault = fault0
            w.fixture = fixture0
            w.recorders[:] = recorders
            w.timing, w.soil.timing = timing0, soil_timing0
            w.strike_contacts, w.soil.max_speed = contacts0, speed0
            w.mjd.time = mj_time0
            ok = [max(res["ground_deflection_m"][k], res["projected_ground_m"][k] or 0.0)
                  for k in res["ground_deflection_m"]]
            res["max_ground_deflection_m"] = max(ok) if ok and res["released"] and not res["released"]["toppled"] \
                else float("inf")
            res["max_deflection_m"] = max(res["deflection_m"].values()) if res["deflection_m"] else float("inf")
            res["wall_s"] = round(time.perf_counter() - t_wall, 1)

    # ---------------------------------------------------------------- T8
    def _break_hardpan(self, lg, final):
        (x0, y0), (x1, y1) = self.env.spec["target_square"]
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

    # ---------------------------------------------------------------- T9
    def _auger_holes(self, lg, final):
        res = []
        for k, t in enumerate(self.env.spec["targets"]):
            t = np.asarray(t)
            rect = ((t[0] - 0.08, t[1] - 0.08), (t[0] + 0.08, t[1] + 0.08))
            top, org = lg.heightfield(rect, cell=0.005)
            d, X, Y = lg.depth_at(top, org, t, 0.08)
            deep = d >= 0.05
            dia = 2*math.sqrt(float(deep.sum())*org[2]**2/math.pi)
            if deep.any():
                cx, cy = float(np.mean(X[deep])), float(np.mean(Y[deep]))
                off = math.hypot(cx - t[0], cy - t[1])
                dc, _, _ = lg.depth_at(top, org, (cx, cy), 0.015)
                depth = float(np.mean(dc))
            else:
                off, depth = float("nan"), 0.0
            res.append(dict(depth=depth, dia=dia, off=off,
                            ok=depth >= 0.18 and 0.07 <= dia <= 0.095 and np.isfinite(off) and off <= 0.02))
        held = tool_held(self.env, "drill")
        n_ok = sum(r["ok"] for r in res)
        c = [crit(f"hole_{k + 1}", dict(depth=round(r["depth"], 3), diameter=round(r["dia"], 3),
                                        offset=None if not np.isfinite(r["off"]) else round(r["off"], 3)), r["ok"],
                  ">= 0.18 m deep, 70-95 mm wide at 5 cm, within 20 mm of the mark") for k, r in enumerate(res)]
        c.append(crit("drill_held", held, all(held.values()), "both hands on the drill"))
        return c, [n_ok/len(res), float(all(held.values()))]

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

    def _expose_line(self, lg, final):
        sp = self.env.spec
        pp = sp["pipe"]
        crown = pp["z"] + pp["r"]
        b = sp["block"]
        xs = np.arange(b["lo"][0] + 0.05, b["hi"][0] - 0.05, 0.01)
        above = (np.abs(lg.pos[:, 1] - pp["y"]) < pp["r"]) & (lg.pos[:, 2] > crown) & (lg.pos[:, 2] < crown + 0.01 + 0.5*lg.h)
        covered = np.zeros(len(xs), bool)
        ix = np.floor((lg.pos[above, 0] - xs[0])/0.01).astype(int)
        ix = ix[(ix >= 0) & (ix < len(xs))]
        covered[ix] = True
        run, best = 0, 0
        for cv in covered:
            run = 0 if cv else run + 1
            best = max(best, run)
        exposed = best*0.01
        zone = sp["tolerance_zone"]
        outside = (lg.origin == ORIGIN_BLOCK) & ((lg.pos0[:, 1] < zone[0] - 0.0) | (lg.pos0[:, 1] > zone[1])) & \
            (np.linalg.norm(lg.pos - lg.pos0, axis=1) > 0.03)
        m_out = float(lg.mass[outside].sum())
        w = self.env.world
        c = [crit("crown_exposed", exposed, exposed >= 0.15, ">= 150 mm of crown with no soil within 10 mm above it", "m"),
             crit("no_strikes", len(w.strikes), len(w.strikes) == 0, "0 strikes (> 30 N or > 0.2 m/s on the lateral)"),
             crit("outside_zone", m_out, m_out <= 0.5, "<= 0.5 kg disturbed outside the 18 in tolerance zone", "kg")]
        return c, [min(1., exposed/0.15), float(len(w.strikes) == 0), max(0., 1 - m_out/2.)]

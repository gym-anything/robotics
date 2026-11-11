"""soil-tasks-v2: state-only judgements for the explicitly selected render-checked eight-task set.

No physics edits or runtime construction. The legacy Ledger and outcome geometry are reused, as in
soil-tasks-v1 (task_set.py). Differences from v1: a failed run scores 0.9 x progress; progress is net of the
settled start plus an idle margin, so an untouched or idle start reads exactly 0; penetrate-marked's progress has
no hand factor (both hands stay a success criterion); break-hardpan credits only soil the robot worked
(WorkWindows). Under this contract set-post keeps its own verdict intact (criteria, counterfactual load test,
success); only its failed-run score is measured from the settled start. Its own contract (no contract selector)
is untouched.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import layout as L, materials as M, params as P
from .catalogue import POST_CONTRACT, REVISED
from .catalogue_v2 import CONTRACT, instruction
from .task import Evaluator as LegacyEvaluator, Ledger, crit, diag, _pose, _tilt_deg, tool_held, ORIGIN_BLOCK

# 1 on success, else 0.9 x progress. Explicit contract only; bare selections, soil-tasks-v1 and set-post's own
# contract (soil-set-post-v1) keep their own rules, including set-post's min(progress, 0.99).
FAILURE_FACTOR = 0.9
SCORE_RULE = "1 on success, else 0.9 x progress; progress net of the settled start plus an idle margin"
# Idle margins on the raw progress reading: credit starts only above the settled start's reading plus the margin, so an
# untouched or idle start reads exactly 0 in every run. Measured in the render-check round, robot idle
# holding every target:
#   set-post: the load-test reading differed by up to 4.4e-5 between runs (0.0024009-0.0024452 at the start), by up
#     to 7.0e-7 between the start's baseline and a later start reading, and rose 4.7e-6 in 2 s idle;
#   expose-line: cover creep raised the reading 4.2e-6 in 2 s;
#   break-hardpan: its loosening reading rose steadily under the idle robot (+1.5e-3 in 15 s, J_p creep of the soil at
#     rest); it now counts only soil the robot worked (WorkWindows) and reads exactly 0 however long the robot idles;
#   the other five read 0 untouched and after 2 s idle.
# 0.002 is 45x set-post's run-to-run spread and covers set-post's and expose-line's idle drift many times over.
IDLE_MARGIN = 0.002
IDLE_MARGINS = {}


def clip(x):
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else 0.0


def ramp(x, full, zero):
    return clip((zero - x)/(zero - full))


def band(x, low, high, zero_low, zero_high):
    return min(clip((x - zero_low)/(low - zero_low)), ramp(x, high, zero_high))


def idle_margin(task):
    return IDLE_MARGINS.get(task, IDLE_MARGIN)


def net_progress(raw, start, margin=0.0):
    """Progress above the zero point start + margin, normalized to full credit at 1."""
    zero = start + margin
    return clip((raw - zero)/(1.0 - zero)) if zero < 1.0 else 0.0


def failed_score(progress):
    return FAILURE_FACTOR*clip(progress)


def part_lowest_z(part, position, rotation):
    """Exact support height of an authored box/cylinder/capsule/mesh part.

    Use scene.parts (already transformed into the spindle frame), not an assumed
    vertical auger length: a tilted flight and the tip both have to clear grade.
    """
    r = rotation @ Rotation.from_quat(part.quat).as_matrix()
    p = np.asarray(position) + rotation @ np.asarray(part.pos)
    if part.kind == "mesh":
        return float((p + np.asarray(part.mesh[0]) @ r.T)[:, 2].min())
    if part.kind == "box":
        return float(p[2] - np.abs(r[2]) @ np.asarray(part.size))
    radius, half = part.size[:2]
    radial = radius if part.kind == "capsule" else radius*math.sqrt(max(0., 1. - r[2, 2]**2))
    if part.kind not in ("capsule", "cylinder"):
        raise ValueError(f"Unsupported auger geometry {part.kind}")
    return float(p[2] - abs(r[2, 2])*half - radial)


def crown_cover(lg, spec, positions):
    """Maximum soil thickness over each 1 cm station of the pipe's crown.

    Read the full overhead column. A void beside the crown beneath a floating
    roof of soil is not an exposed utility (the old 1 cm shell missed that roof).
    """
    pp, b = spec["pipe"], spec["block"]
    xs = np.arange(b["lo"][0] + .05, b["hi"][0] - .05, .01)
    crown = pp["z"] + pp["r"]
    cover = np.zeros(len(xs))
    sel = (np.abs(positions[:, 1] - pp["y"]) < pp["r"]) & (positions[:, 2] + .55*lg.h > crown)
    x = positions[sel]
    ix = np.floor((x[:, 0] - xs[0])/.01).astype(int)
    valid = (ix >= 0) & (ix < len(xs))
    np.maximum.at(cover, ix[valid], np.maximum(0., x[valid, 2] + .55*lg.h - crown))
    return cover


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


class History:
    """The verdict's recorded history (the legacy slab history; break-hardpan's work windows): saved and restored with
    the backend's checkpoints, restarted with each episode."""

    def history(self):
        return dict(base_off_slab=self.base_off_slab)

    def restore_history(self, h):
        self.base_off_slab = h["base_off_slab"]

    def restart_history(self):
        self.base_off_slab = 0.0


class TaskSetEvaluator(History, LegacyEvaluator):
    STOPS_REPORTED_ONLY = REVISED

    def __init__(self, env):
        super().__init__(env)
        lg = Ledger(env)
        self.start_cover = crown_cover(lg, env.spec, lg.pos) if self.task == "expose-line" else None
        self.start_raw = self._read(lg)[1]

    def _read(self, lg):
        return getattr(self, "read_" + self.task.replace("-", "_"))(lg)

    def progress(self):
        # Legacy's advisory error result has no contract; tag even this path.
        return dict(super().progress(), task_contract=CONTRACT)

    def _evaluate(self, final):
        common = self.common()
        criteria, raw, diagnostics = self._read(Ledger(self.env))
        criteria = [c for c in common if c["name"] != "protective_stops"] + criteria
        diagnostics += [diag(c["name"], c["value"], "reported only; recovery does not invalidate the job")
                        for c in common if c["name"] == "protective_stops"]
        success = all(c["ok"] for c in criteria)
        margin = idle_margin(self.task)
        progress = net_progress(raw, self.start_raw, margin)
        w = self.env.world
        return dict(task=self.task, task_contract=CONTRACT, succeeded=success,
                    score=1.0 if success else round(failed_score(progress), 4), progress=progress,
                    raw_progress=raw, start_raw_progress=self.start_raw, idle_margin=margin,
                    score_rule=SCORE_RULE,
                    criteria=criteria, diagnostics=diagnostics, tick=int(w.tick), time_s=round(w.sim_time, 4),
                    strikes=list(w.strikes), reflexes=list(w.reflexes)[-5:])

    def read_penetrate_marked(self, lg):
        p, r = _pose(self.env, "spade")
        depth, off, tilt = L.GRADE - p[2], float(np.linalg.norm(p[:2] - self.env.spec["target"])), _tilt_deg(r)
        held = tool_held(self.env, "spade")
        # An empty contact mapping is not a two-handed hold.
        both = len(held) == 2 and all(held.values())
        c = [crit("edge_depth", depth, depth >= .15, ">= 0.15 m below grade", "m"),
             crit("edge_on_mark", off, off <= .03, "<= 30 mm from the mark", "m"),
             crit("blade_vertical", tilt, tilt <= 10., "<= 10 degrees", "deg"),
             crit("both_hands", held, both, "both hands on the spade")]
        # Progress from the spade and soil alone: both hands are a success criterion, not credit.
        raw = clip(depth/.15)*ramp(off, .03, .10)*ramp(tilt, 10., 30.)
        return c, raw, []

    def read_spit_to_tarp(self, lg):
        c, _ = self._spit_to_tarp(lg, False)
        bed, tarp = lg.origin == ORIGIN_BLOCK, lg.on_tarp()
        mass = float(lg.mass[bed & tarp].sum())
        spill = float(lg.mass[bed & ~tarp & lg.out_of_block()].sum())
        frac = spill/max(mass + spill, 1e-9)
        held = c.pop()
        return c, clip(mass/1.5)*ramp(frac, .15, .50), [diag("spade_held", held["value"], "reported only")]

    def read_planting_hole(self, lg):
        depth, diameter, crumbs = self._hole(lg, self.env.spec["target"], .16, .02, (.33, .50))
        dug, tarp, walk = self._spoil(lg)
        frac = tarp/max(dug, 1e-9)
        c = [crit("depth_at_mark", depth, abs(depth - .16) <= .02, "0.16 +/- 0.02 m", "m"),
             crit("top_diameter", diameter, .33 <= diameter <= .50, "0.33-0.50 m at 80 mm depth", "m"),
             crit("loose_on_floor", crumbs, crumbs <= .3, "<= 0.3 kg displaced loose crumbs", "kg"),
             crit("spoil_on_tarp", frac, frac >= .8 and dug > .5, ">= 80% of excavated mass; > 0.5 kg excavated"),
             crit("walkway_clean", walk, walk <= .2, "<= 0.2 kg", "kg")]
        shape = min(band(depth, .14, .18, 0., .26), band(diameter, .33, .50, 0., .65))
        raw = clip(shape*(.8 + .2*ramp(crumbs, .3, 1.))*(.7 + .3*clip(frac/.8)) - .2*max(0., walk - .2)/.8)
        return c, raw, []

    def read_cable_trench(self, lg):
        c, _ = self._cable_trench(lg, False)  # unchanged success geometry and slab history
        (x0, y0), (x1, _) = self.env.spec["target_line"]
        top, org = lg.heightfield(((x0 - .05, y0 - .25), (x1 + .05, y0 + .25)))
        depth = np.where(np.isnan(top), .5, L.GRADE - top)
        xs = org[0] + (np.arange(top.shape[0]) + .5)*org[2]
        ys = org[1] + (np.arange(top.shape[1]) + .5)*org[2]
        rows = depth[(xs >= x0) & (xs <= x1)]
        deep = rows.max(axis=1)
        deepest = rows >= .8*deep[:, None]
        centre = (deepest*ys[None, :]).sum(1)/np.maximum(deepest.sum(1), 1)
        width = (rows >= np.maximum(deep/2, .02)[:, None]).sum(1)*org[2]
        # Weight each row's alignment by that row's depth. Work on another row
        # cannot cancel an off-line excavation; flat untouched rows earn zero.
        shape = np.mean([clip(d/.16)*ramp(abs(y - y0), .05, .15)*
                         (.8 + .2*band(w, .06, .20, 0., .30)) for d, y, w in zip(deep, centre, width)])
        dug, tarp, walk = self._spoil(lg)
        raw = clip(shape*(.7 + .3*clip(tarp/max(dug, 1e-9)/.9)) - .2*max(0., walk - .2))
        return c, raw, []

    def read_auger_holes(self, lg):
        c, credits = [], []
        for k, mark in enumerate(self.env.spec["targets"]):
            x, y = mark
            top, org = lg.heightfield(((x - .08, y - .08), (x + .08, y + .08)), cell=.005)
            depths, X, Y = lg.depth_at(top, org, mark, .08)
            deep = depths >= .05
            diameter = 2*math.sqrt(float(deep.sum())*org[2]**2/math.pi)
            if deep.any():
                cx, cy = float(X[deep].mean()), float(Y[deep].mean())
                off = math.hypot(cx - x, cy - y)
                dc, _, _ = lg.depth_at(top, org, (cx, cy), .015)
                depth = float(dc.mean())
            else:
                off, depth = float("inf"), 0.
            ok = depth >= .18 and .07 <= diameter <= .095 and off <= .02
            c.append(crit(f"hole_{k + 1}", dict(depth=depth, diameter=diameter, offset=off if np.isfinite(off) else None),
                          ok, ">= 0.18 m deep, 70-95 mm wide at 5 cm, within 20 mm of the mark"))
            credits.append(min(clip(depth/.18), band(diameter, .07, .095, .04, .14), ramp(off, .02, .06)))
        p, r = _pose(self.env, "auger")
        clearance = min(part_lowest_z(part, p, r) for part in self.env.world.scene.parts["auger"]) - L.GRADE
        c.append(crit("auger_clear", clearance, clearance >= 0., "every auger/spindle part at or above grade", "m"))
        raw = float(np.mean(credits))*(.9 + .1*clip(1. + clearance/.05))
        return c, raw, []

    def read_expose_line(self, lg):
        c, _ = self._expose_line(lg, False)
        cover = crown_cover(lg, self.env.spec, lg.pos)
        run = best = 0
        for thickness in cover:
            run = run + 1 if thickness == 0. else 0
            best = max(best, run)
        exposed = best*.01
        c[0] = crit("crown_exposed", exposed, exposed >= .15, ">= 150 mm contiguous crown with no overhead soil", "m")
        dug, tarp, _ = self._spoil(lg)
        frac = tarp/max(dug, 1e-9)
        c.append(crit("spoil_on_tarp", frac, frac >= .8 and dug > .5, ">= 80% of excavated mass; > 0.5 kg excavated"))
        # Compare each 15 cm window with its own settled start cover, then take
        # the best. No command, action order, hand or chosen digging method enters.
        start = self.start_cover
        uncover = max((clip(1. - cover[i:i + 15].mean()/start[i:i + 15].mean())
                       if start[i:i + 15].mean() > 0. else 0. for i in range(len(cover) - 14)), default=0.)
        zone = self.env.spec["tolerance_zone"]
        outside = (lg.origin == ORIGIN_BLOCK) & ((lg.pos0[:, 1] < zone[0]) | (lg.pos0[:, 1] > zone[1])) & \
                  (np.linalg.norm(lg.pos - lg.pos0, axis=1) > .03)
        m_out = float(lg.mass[outside].sum())
        raw = clip((.5*uncover + .5*clip(exposed/.15))*2.**(-len(self.env.world.strikes))*
                   (.8 + .2*clip(frac/.8)) - .25*max(0., m_out - .5)/2.)
        return c, raw, [diag("cover_removed", uncover, "best 15 cm window's reduction from settled start cover")]


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

    def read_break_hardpan(self, lg):
        c, _ = self._break_hardpan(lg, False)
        (x0, y0), (x1, y1) = self.env.spec["target_square"]
        sq = lg.in_rect(((x0, y0), (x1, y1)), pos=lg.pos0) & (lg.origin == ORIGIN_BLOCK)
        tarp = sq & lg.on_tarp()
        mass = float(lg.mass[tarp].sum())
        spill = float(lg.mass[sq & lg.out_of_block() & ~tarp].sum())
        frac = spill/max(mass + spill, 1e-9)
        c.append(crit("spill_fraction", frac, frac <= .20, "<= 20% of square soil lifted out"))
        # Smooth displacement/dilation credits at every depth, rather than a count of columns whose deepest loosened
        # particle crossed 80 mm; each particle's displacement and J_p rise are those of the windows in which the robot
        # worked its column, so soil creeping at rest earns nothing.
        ww = self.windows
        moved, dilated = ww.net(lg.pos[ww.idx], lg.jp[ww.idx])
        credit = np.maximum(np.clip(np.linalg.norm(moved, axis=1)/.01, 0., 1.), np.clip(dilated/.05, 0., 1.)) * \
            np.clip((L.GRADE - lg.pos0[ww.idx, 2])/.08, 0., 1.)
        columns = np.zeros(ww.shape[0]*ww.shape[1])
        np.maximum.at(columns, ww.column, credit)
        raw = (.6*clip(float(columns.mean())/.8) + .4*clip(mass/4.))*ramp(frac, .2, .6)
        return c, raw, [diag("worked_columns", float(ww.worked.mean()),
                             "share of the square's 2 cm columns a working body has touched; reported only")]


class SetPostEvaluator(History, LegacyEvaluator):
    """set-post under the explicit soil-tasks-v2 contract.

    set-post's own judgement runs unchanged: the same criteria, the same braces-off
    counterfactual load test and the same success. Only the failed-run score
    differs. Its own progress 0.5 x placement + 0.5 x support x plumb is not
    zero at the settled start: the bare post in its dug hole carries the test push
    for about 0.12 s before its foot passes 12.7 mm (support credit about 0.005,
    plumb credit about 0.9), so an untouched start read 0.0024. Here progress is
    that same reading net of its value at the settled start, measured
    once at install with the same load test (and the advisory placement reading
    net of its own start value): with z = p0 + IDLE_MARGIN (the load test's
    run-to-run spread with headroom), progress = clip((p - z)/(1 - z)); score =
    1 on success, else 0.9 x progress.
    """

    def __init__(self, env):
        super().__init__(env)
        self.start_raw = self.start_advisory = None
        lg = Ledger(env)
        self.start_advisory = self._raw(lg, final=False)
        try:
            self.start_raw = self._raw(lg, final=True)
        except Exception as exc:
            raise RuntimeError(f"set-post {CONTRACT}: the start's baseline load test failed ({exc})") from exc
        self.start_load_test = getattr(self, "last_load_test", None)

    def _raw(self, lg, final):
        return 0.5*float(sum(LegacyEvaluator._set_post(self, lg, final)[1]))

    def _set_post(self, lg, final):
        c, parts, extra = super()._set_post(lg, final)
        start = self.start_raw if final else self.start_advisory
        raw = 0.5*float(sum(parts))
        margin = idle_margin(self.task)
        progress = net_progress(raw, start, margin)
        fields = dict(extra.get("fields", {}), progress=progress, raw_progress=raw, start_raw_progress=start,
                      idle_margin=margin,
                      set_post_progress=round(raw, 4), set_post_contract=POST_CONTRACT, score_rule=SCORE_RULE)
        return c, parts, dict(extra, score=failed_score(progress), fields=fields)

    def progress(self):
        return dict(super().progress(), task_contract=CONTRACT)

    def evaluate(self):
        return dict(super().evaluate(), task_contract=CONTRACT)


def install(env, contract):
    """Install the explicit soil-tasks-v2 contract after the original environment has settled; do not move any body.
    Any other contract leaves the environment exactly as built: its evaluator, instruction and results unchanged."""
    if contract != CONTRACT:
        return
    old = env.evaluator
    env.spec["instruction"] = instruction(env.spec["task"], contract)
    env.evaluator = {"set-post": SetPostEvaluator, "break-hardpan": BreakHardpanEvaluator}.get(
        env.spec["task"], TaskSetEvaluator)(env)
    env.world.recorders.remove(old._record)  # exactly one slab-history recorder
    env._eval_key = None

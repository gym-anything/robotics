"""State-only judgements for the explicitly selected eight-task contract.

No physics edits or runtime construction. The legacy Ledger and outcome geometry
are reused; set-post is delegated intact, including its counterfactual verdict.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import layout as L, materials as M
from .catalogue import CONTRACT, REVISED, default_contract, instruction
from .task import Evaluator as LegacyEvaluator, Ledger, crit, diag, _pose, _tilt_deg, tool_held, ORIGIN_BLOCK

# set-post's rule is a cap, not a multiplicative discount. Keep that rule.
FAILURE_CAP = 0.99


def clip(x):
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else 0.0


def ramp(x, full, zero):
    return clip((zero - x)/(zero - full))


def band(x, low, high, zero_low, zero_high):
    return min(clip((x - zero_low)/(low - zero_low)), ramp(x, high, zero_high))


def net_progress(raw, start):
    return clip((raw - start)/(1.0 - start)) if start < 1.0 else 0.0


def failed_score(progress):
    return min(clip(progress), FAILURE_CAP)


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


class TaggedEvaluator:
    """Only add provenance to the legacy result; no criteria or scores change."""
    def __init__(self, evaluator):
        self.legacy = evaluator
        self.task_contract = default_contract(evaluator.task)

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def evaluate(self):
        return dict(self.legacy.evaluate(), task_contract=self.task_contract)

    def progress(self):
        return dict(self.legacy.progress(), task_contract=self.task_contract)


class TaskSetEvaluator(LegacyEvaluator):
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
        progress = net_progress(raw, self.start_raw)
        w = self.env.world
        return dict(task=self.task, task_contract=CONTRACT, succeeded=success,
                    score=1.0 if success else round(failed_score(progress), 4), progress=progress,
                    raw_progress=raw, start_raw_progress=self.start_raw,
                    score_rule="1 on success, else min(progress, 0.99) (set-post's rule)",
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
        raw = clip(depth/.15)*ramp(off, .03, .10)*ramp(tilt, 10., 30.)*(.8 + .2*both)
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

    def read_break_hardpan(self, lg):
        c, _ = self._break_hardpan(lg, False)
        (x0, y0), (x1, y1) = self.env.spec["target_square"]
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


def install(env, contract):
    """Install after the original environment has settled; do not move any body."""
    old = env.evaluator
    env.spec["instruction"] = instruction(env.spec["task"], contract)
    if contract == CONTRACT:
        evaluator = TaskSetEvaluator(env)
        env.world.recorders.remove(old._record)  # exactly one slab-history recorder
    else:
        evaluator = TaggedEvaluator(old)
    env.evaluator = evaluator
    env._eval_key = None

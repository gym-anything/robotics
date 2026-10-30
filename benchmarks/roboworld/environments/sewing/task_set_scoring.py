"""Opt-in eight-task verifiers. Pure NumPy state readers; no simulator imports.

The path ledger measurement below ports Evaluator._curve_seam's runs, joins,
thread links and density unchanged, with distances to a finite task polyline. The
original curve continues to use its original evaluator and scoring implementation.
"""

import numpy as np

from .task import (Evaluator, _check, _polyline_distance, _triangle_under, TOL_LINE, ENDS_MEET,
                   END_REACH, STEP_MAX, NOT_A_SKIP, THREAD_LINK_MAX, SAME_PLACE, JOIN_SECURE,
                   BACKTACK_MIN, INCH, SPI_BAND, STITCH_BOUNDS, CURVE_SOFT, CURVE_SOFT_REG,
                   CURVE_DAMAGE, CURVE_FAULT, CURVE_PARTIAL_MAX)
from .task_set import CONTRACT


def ramp(error, good, bad):
    """One inside good, zero at bad; continuous between them, zero for nonfinite data."""
    return float(np.clip((bad - error)/(bad - good), 0., 1.)) if np.isfinite(error) else 0.


def seam_path(task, width, length, allowance=.010):
    right = width - allowance
    if task == "three-sides":
        return np.array([[right, length], [right, allowance], [allowance, allowance], [allowance, length]])
    if task == "pivot":
        return np.array([[right, length], [right, allowance], [0., allowance]])
    end = length - .120 if task == "seam-to-notch" else 0.
    return np.array([[right, length], [right, end]])


def weighted_coverage(at, credit, lo, hi):
    """Exact integral of the best stitch credit over intervals +/- STEP_MAX/2.

    Overlapping/repeated stitches cannot add length. Unlike a sampling grid this
    changes continuously as a stitch moves. Also returns the stitches owning length.
    """
    intervals = [(max(lo, p - STEP_MAX/2), min(hi, p + STEP_MAX/2), float(c), i)
                 for i, (p, c) in enumerate(zip(at, credit)) if c > 0]
    intervals = [r for r in intervals if r[1] > r[0]]
    events = sorted({lo, hi, *(x for a, b, _, _ in intervals for x in (a, b))})
    area, owners = 0., set()
    for a, b in zip(events[:-1], events[1:]):
        covering = [(c, i) for x, y, c, i in intervals if x <= a and y >= b]
        if covering:
            c, i = max(covering)
            area += (b - a)*c
            owners.add(i)
    return area/max(hi - lo, 1e-12), owners


class TaskEvaluator(Evaluator):
    def __init__(self, env):
        super().__init__(env)
        self.square_start_error = None
        if env.config.task == "square-plies" and getattr(env.world, "cloth", True) is not None:
            self.square_start_error = max(self.ply_registration()[0], float(np.linalg.norm(
                env.world.ply_positions(0)[:, :2] - env.world.ply_positions(1)[:, :2], axis=1).max()))
            if self.square_start_error <= .0015:
                raise ValueError("square-plies must start with slipped plies")

    def _curve_seam(self):
        # Same state ledger as the curve task; only the target path differs.
        w = self.env.world
        A = .010
        line = seam_path(self.env.config.task, w.config.ply_width, w.config.ply_length, A)
        st = self.stitches()
        formed = [s for s in st if s["formed"] and s["holes"]]
        two = [s for s in formed if s["plies"] == [0, 1]]
        uv = np.array([s["holes"][k] for s in two for k in (0, 1)], float).reshape(-1, 2)
        def distance(points):
            if self.env.config.task == "seam-to-notch":
                mark = w.config.ply_length - .120
                lateral = abs(points[:, 0] - (w.config.ply_width - A))
                overrun = np.maximum(mark - .0032 - points[:, 1], 0.)
                return np.maximum(lateral, np.where(overrun > 0., TOL_LINE + overrun, 0.))
            return _polyline_distance(points, line)[0]
        d = (A + distance(uv)).reshape(-1, 2)
        at2 = _polyline_distance(uv, line)[1].reshape(-1, 2)
        allh = np.array([h for s in formed for h in s["holes"].values()], float).reshape(-1, 2)
        d_all = A + distance(allh)
        cut = np.cumsum([0] + [len(s["holes"]) for s in formed])
        dev_formed = np.array([np.abs(d_all[cut[i]:cut[i + 1]] - A).max() for i in range(len(formed))])
        skipped = [s for s in st if not s["formed"] and s["reason"] not in NOT_A_SKIP]
        skip_s = [np.sort(_polyline_distance(np.array([s["holes"][k] for s in skipped if k in s["holes"]],
                                                      float).reshape(-1, 2), line)[1]) for k in (0, 1)]
        # places and sequences: the stitches of the seam in sewing order, a skipped stitch's holes included
        # (skips judges the skip itself), collapsed into places while within SAME_PLACE of a place's first stitch (a pile,
        # a lock or a bar tack is one place; a slow creep is a place every SAME_PLACE); a sequence is consecutive places
        # of one seam piece and feed direction moving the same way along the line, a step up to THREAD_LINK_MAX (a
        # longer thread is judged by no_long_threads)
        two_id = {id(x): i for i, x in enumerate(two)}
        members = [x for x in st if (id(x) in two_id) or (not x["formed"] and x["reason"] not in NOT_A_SKIP
                                                         and sorted(x["holes"]) == [0, 1])]
        m_at = np.array([_polyline_distance(np.array([x["holes"][k] for k in (0, 1)], float), line)[1]
                         for x in members], float).reshape(-1, 2)
        seqs, places, sign = [], [], 0.0            # places: [members' indices]; seqs: [places]
        for i, x in enumerate(members):
            if places:
                first = members[places[-1][0]]
                d0 = float(m_at[i].mean() - m_at[places[-1][0]].mean())
                if x["reverse"] == first["reverse"] and x["seam"] == first["seam"] and abs(d0) < SAME_PLACE:
                    places[-1].append(i)
                    continue
                prev = members[places[-1][-1]]
                step = float(m_at[i].mean() - m_at[places[-1][0]].mean())
                if (x["reverse"] == prev["reverse"] and x["seam"] == prev["seam"] and abs(step) <= THREAD_LINK_MAX
                        and (sign == 0.0 or np.sign(step) == sign)):
                    sign = sign or float(np.sign(step))
                    places.append([i])
                    continue
                seqs.append(places)
            places, sign = [[i]], 0.0
        if places:
            seqs.append(places)
        density, step_local, runs = None, np.full(len(two), np.nan), []
        for q in seqs:
            pos = np.array([m_at[pl].mean(axis=0) for pl in q])        # (places, plies)
            runs.append([two_id[id(members[i])] for pl in q for i in pl if id(members[i]) in two_id])
            if len(q) < 2:
                continue
            for k in (0, 1):
                c = np.sort(np.abs(pos[:, k] - pos[0, k]))
                if c[-1] - c[0] >= INCH:
                    x0 = np.arange(c[0], c[-1] - INCH + 1e-12, 0.25e-3)
                    n = np.searchsorted(c, x0 + INCH) - np.searchsorted(c, x0)
                    density = (min(density[0], int(n.min())) if density else int(n.min()),
                               max(density[1], int(n.max())) if density else int(n.max()))
            c = np.abs(pos.mean(axis=1) - pos[0].mean())
            for m, pl in enumerate(q):
                lo, hi = np.searchsorted(c, c[m] - 0.5*INCH), np.searchsorted(c, c[m] + 0.5*INCH, side="right") - 1
                if hi > lo:
                    mean_step = (c[hi] - c[lo])/(hi - lo)
                else:                             # no other place in the inch about it: the step to its nearer neighbour
                    mean_step = min([c[m] - c[m - 1]]*(m > 0) + [c[m + 1] - c[m]]*(m < len(q) - 1))
                for i in pl:
                    if id(members[i]) in two_id:
                        step_local[two_id[id(members[i])]] = mean_step
        # joins: each end of a trimmed seam piece inside the seam (not within END_REACH of the far end or the
        # line's end) has the JOIN_SECURE of seam inside it sewn at least twice, by its own backtack or lock or by
        # another piece overlapping it; the smallest such length over every interior end (None without one)
        ell = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        spans_seq = []
        for j, q in enumerate(seqs):
            pts = [float(m_at[i].mean()) for pl in q for i in pl]
            if j:                                 # the thread from the previous stitch of the piece sews into it too
                prev = seqs[j - 1][-1][-1]
                if (members[prev]["seam"] == members[q[0][0]]["seam"]
                        and abs(float(m_at[prev].mean()) - pts[0]) <= THREAD_LINK_MAX):
                    pts.append(float(m_at[prev].mean()))
            spans_seq.append((min(pts), max(pts), members[q[0][0]]["seam"]))
        joins = None
        pieces = sorted({sq[2] for sq in spans_seq}, key=lambda v: (v is None, v))
        if len(pieces) > 1:
            grid = np.arange(0.0, ell, 0.1e-3)
            passes = np.zeros(len(grid), int)
            for lo_, hi_, _ in spans_seq:
                passes[(grid >= lo_ - 0.5*STEP_MAX) & (grid <= hi_ + 0.5*STEP_MAX)] += 1
            for piece in pieces:
                ends = [sq for sq in spans_seq if sq[2] == piece]
                for e, inward in ((min(sq[0] for sq in ends), 1.0), (max(sq[1] for sq in ends), -1.0)):
                    if e <= END_REACH or e >= ell - END_REACH:
                        continue
                    depth = (grid - e)*inward
                    sel = np.nonzero((depth >= 0) & (depth <= JOIN_SECURE))[0]
                    sel = sel[np.argsort(depth[sel])]
                    single = np.nonzero(passes[sel] < 2)[0]
                    secured = JOIN_SECURE if not len(single) else float(depth[sel[single[0]]])
                    joins = secured if joins is None else min(joins, secured)
        links = []
        for a, b in zip(formed[:-1], formed[1:]):
            if a["seam"] == b["seam"]:
                top = [np.asarray(x["holes"].get(1, x["holes"].get(0)), float) for x in (a, b)]
                bottom = [np.asarray(x["holes"].get(0, x["holes"].get(1)), float) for x in (a, b)]
                links.append(max(float(np.linalg.norm(top[1] - top[0])), float(np.linalg.norm(bottom[1] - bottom[0]))))
        return dict(stitches=st, formed=formed, two=two, d=d, s=at2, dev=np.abs(d - A).max(axis=1),
                    reg=np.linalg.norm(uv[0::2] - uv[1::2], axis=1), d_all=d_all, dev_formed=dev_formed,
                    is_two=np.array([s["plies"] == [0, 1] for s in formed], bool), skip_s=skip_s, runs=runs,
                    density=density, step_local=step_local, links=np.asarray(links, float), joins=joins,
                    ell=float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()))

    def _seam_checks(self):
        c = super()._eval_curve()
        g = self._curve
        task = self.env.config.task
        if len(g["two"]) >= 6:
            # Distances here are to the target line (encoded as A + error for the
            # inherited measurement contract), not signed edge allowances.
            self._info.pop("seam_mean", None)
            self._info["mean_line_error_mm"] = float((g["d"] - .010).mean()*1000)
            c["seam_line"] = _check(float(g["dev_formed"].max()) <= TOL_LINE + 1e-12,
                                    float(g["dev_formed"].max())*1000,
                                    "<= 1.6 mm from target line; notch longitudinal overrun <= 3.2 mm")
            far = float(g["s"].min(axis=0).max())
            end = float(g["ell"] - g["s"].max(axis=0).min())
            c["joined_far_end"] = _check(far <= END_REACH + 1e-12, far*1000,
                                         "<= 6 mm along the seam, each ply")
            c["reaches_left_edge"] = _check(end <= END_REACH + 1e-12, end*1000,
                                            "<= 6 mm along the seam, each ply")
        if "reaches_left_edge" in c:
            c["joined_end"] = c.pop("reaches_left_edge")
        if task != "pivot":
            # End-local backtacks, measured relative to the sewn seam's ends (which
            # themselves must reach the prescribed endpoints). Either sewing order.
            counts = [0, 0]
            if len(g["two"]):
                lo, hi = g["s"].min(axis=0), g["s"].max(axis=0)
                for s, at in zip(g["two"], g["s"]):
                    if s["reverse"]:
                        counts[0] += int(np.all(at <= lo + JOIN_SECURE + 1e-12))
                        counts[1] += int(np.all(at >= hi - JOIN_SECURE - 1e-12))
            self._secured = [min(1., n/BACKTACK_MIN) for n in counts]
            for key, n in zip(("backtack_start", "backtack_end"), counts):
                c[key] = _check(n >= BACKTACK_MIN, n, ">= 3 reverse stitches within 6.35 mm of this sewn end")
        if task in ("pivot", "three-sides"):
            line = seam_path(task, self.env.world.config.ply_width, self.env.world.config.ply_length)
            for i, corner in enumerate(line[1:-1]):
                # A matching corner in BOTH plies, not just a top-ply stitch.
                d = min((max(np.linalg.norm(np.asarray(s["holes"][k]) - corner) for k in (0, 1))
                         for s in g["two"]), default=float("inf"))
                c[f"corner_{i + 1}"] = _check(d <= TOL_LINE, None if not np.isfinite(d) else d*1e3,
                                              "<= 1.6 mm from the square corner, both plies")
        if task == "seam-to-notch":
            mark = self.env.world.config.ply_length - .120
            near = [min((s["holes"][k][1] for s in g["two"]), default=float("inf")) for k in (0, 1)]
            c["stops_at_mark"] = _check(all(abs(v - mark) <= .0032 for v in near),
                                        target="sewn end within 3.2 mm of 120 mm from far end, both plies")
            # All punctures count here, including unformed cycles and one-ply holes.
            c["opening_clear"] = _check(all(h.uv[1] >= mark - .0032 for h in self.env.world.machine.holes),
                                        target="no puncture more than 3.2 mm into the opening")
        return c

    def _seam_progress(self, checks):
        g, M = self._curve, self.env.world.machine
        broken = {t.get("cycle") for t in M.threads if t["broken"]}
        place = np.minimum(np.clip(1 - (g["dev"] - TOL_LINE)/CURVE_SOFT, 0., 1.),
                           np.clip(1 - (g["reg"] - ENDS_MEET)/CURVE_SOFT_REG, 0., 1.))
        m = g["step_local"]
        dens = np.ones(len(m))
        known = ~np.isnan(m)
        lo, hi = INCH/(SPI_BAND[1] + 1), INCH/SPI_BAND[0]
        dens[known] = np.clip(np.minimum((m[known] - STITCH_BOUNDS[0])/(lo - STITCH_BOUNDS[0]),
                                        (STITCH_BOUNDS[1] - m[known])/(STITCH_BOUNDS[1] - hi)), 0., 1.)
        credit = np.minimum(place, dens)
        credit[np.array([s["cycle"] in broken for s in g["two"]], bool)] = 0.
        coverage, owners = weighted_coverage(g["s"].mean(axis=1), credit, END_REACH, g["ell"] - END_REACH)
        dominated = np.array([i not in owners for i in range(len(place))], bool)
        one_ply = np.clip(1 - (g["dev_formed"][~g["is_two"]] - TOL_LINE)/CURVE_SOFT, 0., 1.)
        strays = (float(np.sum(1 - place[dominated])) + float(np.sum(1 - one_ply))
                  + float(np.sum(g["links"] > THREAD_LINK_MAX)))
        if self.env.config.task == "seam-to-notch":
            mark = self.env.world.config.ply_length - .120
            # Unformed punctures beyond the opening are rework too; formed holes
            # already lose line credit through the finite target path.
            strays += sum(1 - ramp(mark - h[1], .0032, .010)
                          for s in g["stitches"] if not s["formed"] for h in s["holes"].values())/2
        need = (g["ell"] - 2*END_REACH)/self.env.world.config.stitch_length
        rework = max(0., 1 - strays/need)
        damage = min(CURVE_DAMAGE.get(M.needle_state, 1.), CURVE_FAULT if broken else 1.)
        trimmed = M.tether == "trimmed"
        let_go = trimmed and checks["released"]["ok"]
        finish = (.90 if self.env.config.task == "pivot" else .80 + .05*sum(self._secured))
        finish += .05*trimmed + .05*let_go
        progress = coverage*rework*damage*finish
        self._info["progress"] = dict(coverage=coverage, strays=strays, rework=rework, damage=damage,
                                      finish=finish, value=progress, path_mm=g["ell"]*1000)
        return progress

    def load_progress(self):
        w = self.env.world
        W, L = w.config.ply_width, w.config.ply_length
        q = []
        for k in (0, 1):
            hits = self.under_needle(k)
            if len(hits) != 1:
                q.append(0.)
                continue
            u, v = hits[0]
            q.append(min(ramp(abs(W - u - .010), TOL_LINE, .010),
                         ramp(max(.003 - (L - v), L - v - .008, 0.), 0., .020)))
        off, n = self.ply_registration()
        reg = ramp(off, .0015, .010) if n else 0.
        # Pressure counts only where both plies have reached their intended place;
        # a lowered foot on the empty plate cannot earn anything.
        pressure = min(q)*float(np.clip(w.machine.foot_N, 0., 1.))*(w.machine.presser == "down")
        return (.8*float(np.mean(q)) + .2*pressure)*reg

    def _square(self):
        from .layout import TABLE, BED
        w = self.env.world
        X, tri = np.asarray(w.mesh["X"]), np.asarray(w.mesh["tri"])
        xs = [np.asarray(w.ply_positions(k)) for k in (0, 1)]
        off, n = self.ply_registration()
        # Material correspondence also covers the parts for which registration has
        # no resting triangles (lifting the mismatched portion must not hide it).
        off = max(off, float(np.linalg.norm(xs[0][:, :2] - xs[1][:, :2], axis=1).max()))
        samples = np.array([(u, v) for u in np.linspace(.1, .9, 4)*w.config.ply_width
                            for v in np.linspace(.1, .9, 5)*w.config.ply_length])
        t, *bc = _triangle_under(samples, X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]])
        points = sum(b[:, None]*xs[1][tri[t, j], :2] for j, b in enumerate(bc))
        layers = [layer_count(p, tri, x, X) for p in points for x in xs]
        p = np.vstack(xs)
        inside = ((p[:, 0] > TABLE["x"][0]) & (p[:, 0] < BED["x"][0])
                  & (p[:, 1] > TABLE["y"][0]) & (p[:, 1] < TABLE["y"][1]))
        height = float(np.maximum(-.001 - p[:, 2], p[:, 2] - .002).max())
        contacts = [self.hand_contact(a) for a in w.hand_arms]
        c = dict(edges_aligned=_check(n >= 50 and off <= .0015,
                                      dict(misregistration_mm=off*1000, edge_points=n), "<= 1.5 mm, >= 50 points"),
                 single_layer=_check(all(k == 1 for k in layers), layers, "one crossing per ply at 20 samples"),
                 on_left_table=_check(bool(inside.all()) and height <= 0., target="all vertices left of bed on table, z -1..2 mm"),
                 released=_check(all(f < .05 for f in contacts), contacts, "< 0.05 N per hand, pad magnitudes"),
                 no_reflex=_check(not w.reflexes, len(w.reflexes)))
        # Continuous geometric flatness: worst height, boundary violation and signed
        # projected triangle area. No reward for a fraction of Boolean checks.
        boundary = max(float((TABLE["x"][0] - p[:, 0]).max()), float((p[:, 0] - BED["x"][0]).max()),
                       float((TABLE["y"][0] - p[:, 1]).max()), float((p[:, 1] - TABLE["y"][1]).max()), 0.)
        def area(x):
            a, b = x[tri[:, 1], :2] - x[tri[:, 0], :2], x[tri[:, 2], :2] - x[tri[:, 0], :2]
            return a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
        ratios = np.concatenate([area(x)/area(X) for x in xs])
        flat = min(ramp(max(height, 0.), 0., .020), ramp(boundary, 0., .020),
                   float(np.clip(ratios.min()/.2, 0., 1.)))
        release = ramp(max(contacts, default=0.), .05, 1.)
        progress = ramp(off, .0015, self.square_start_error)*flat*(.9 + .1*release)
        self._info["progress"] = dict(value=progress, initial_error_m=self.square_start_error,
                                      misregistration_m=off, flatness=flat, release=release)
        return c, progress

    def evaluate(self):
        self._info = {}
        task = self.env.config.task
        if task == "load":
            checks, progress = self._eval_load(), self.load_progress()
        elif task == "square-plies":
            checks, progress = self._square()
        else:
            checks = self._seam_checks()
            progress = self._seam_progress(checks)
            if task == "load-and-seam":
                # Smoothly retain loading credit as a real joined span forms. A
                # first stitch never unlocks a flat 0.2 bonus; sequence is immaterial.
                c = self._info["progress"]["coverage"]
                progress = max(.2*self.load_progress(), progress + .2*(1 - progress)*min(1., c/.05))
        success = all(c["ok"] for c in checks.values())
        progress = float(np.clip(progress, 0., 1.))
        self._info["continuous_progress"] = progress
        return dict(task=task, task_contract=CONTRACT, success=success,
                    score=1. if success else CURVE_PARTIAL_MAX*progress,
                    primary_outcome=progress > 0., checks=checks, info=self._info)


def layer_count(point, tri, positions, material=None):
    """Vertical crossings, deduplicating adjacent triangles on a shared edge."""
    a = positions[tri[:, 0]]
    v0, v1 = positions[tri[:, 1]] - a, positions[tri[:, 2]] - a
    den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
    valid = abs(den) > 1e-18
    a, v0, v1, den = a[valid], v0[valid], v1[valid], den[valid]
    q = point - a[:, :2]
    b = (q[:, 0]*v1[:, 1] - q[:, 1]*v1[:, 0])/den
    c = (v0[:, 0]*q[:, 1] - v0[:, 1]*q[:, 0])/den
    hit = np.minimum(np.minimum(1 - b - c, b), c) >= -1e-9
    heights = (a[:, 2] + b*v0[:, 2] + c*v1[:, 2])[hit]
    if material is None:
        material = positions[:, :2]
    T = tri[valid][hit]
    uv = ((1 - b[hit] - c[hit])[:, None]*material[T[:, 0]]
          + b[hit, None]*material[T[:, 1]] + c[hit, None]*material[T[:, 2]])
    # Coincident folds still have different material points. Height alone would
    # merge them and incorrectly call a flat fold one layer.
    unique = []
    for crossing in np.column_stack([uv, heights]):
        if not any(np.max(abs(crossing - old)) < 1e-7 for old in unique):
            unique.append(crossing)
    return len(unique)

"""Opt-in eight-task verifiers. Pure NumPy state readers; no simulator imports.

The path ledger measurement below ports the curve's curve_seam (tasks/curve/verifier.py) runs, joins,
thread links and density unchanged, with distances to a finite task polyline. The
original curve continues to use its original evaluator and scoring implementation.

A benchmark task's own grading is in its folder's verifier.py (tasks/<task>/): verify(e) calls TaskEvaluator.verdict,
and the seam grading here reads the task's hooks from it (seam_path, distance, BACKTACKS, CORNERS, extra_checks,
extra_strays, FINISH; check_start when the evaluator is made). three-sides, outside the benchmark, stays here.
"""

import numpy as np

from .task import (Evaluator, _check, _polyline_distance, TOL_LINE, ENDS_MEET,
                   END_REACH, STEP_MAX, NOT_A_SKIP, THREAD_LINK_MAX, SAME_PLACE, JOIN_SECURE,
                   BACKTACK_MIN, INCH, SPI_BAND, STITCH_BOUNDS, CURVE_SOFT, CURVE_SOFT_REG,
                   CURVE_DAMAGE, CURVE_FAULT, CURVE_PARTIAL_MAX, TASKS, task_verifier)
from .task_set import CONTRACT


def ramp(error, good, bad):
    """One inside good, zero at bad; continuous between them, zero for nonfinite data."""
    return float(np.clip((bad - error)/(bad - good), 0., 1.)) if np.isfinite(error) else 0.


def _verifier(task):
    """The verifier.py of a benchmark task this contract defines (corner-seam is sewing-tasks-v2's), else None."""
    return task_verifier(task) if isinstance(task, str) and task in TASKS else None


def seam_path(task, width, length, allowance=.010):
    """The seam's target line in the plies' material frame: the task's own (its verifier.py seam_path), else the
    straight full-length line 10 mm from the right edges."""
    folder = _verifier(task)
    if hasattr(folder, "seam_path"):
        return folder.seam_path(width, length, allowance)
    right = width - allowance
    if task == "three-sides":
        return np.array([[right, length], [right, allowance], [allowance, allowance], [allowance, length]])
    return np.array([[right, length], [right, 0.]])


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
        check_start = getattr(_verifier(env.config.task), "check_start", None)
        if check_start is not None:           # a task whose start is checked (square-plies: the plies slipped)
            check_start(self)

    def _curve_seam(self):
        # Same state ledger as the curve task; only the target path differs.
        w = self.env.world
        A = .010
        line = seam_path(self.env.config.task, w.config.ply_width, w.config.ply_length, A)
        st = self.stitches()
        formed = [s for s in st if s["formed"] and s["holes"]]
        two = [s for s in formed if s["plies"] == [0, 1]]
        uv = np.array([s["holes"][k] for s in two for k in (0, 1)], float).reshape(-1, 2)
        task_distance = getattr(_verifier(self.env.config.task), "distance", None)
        def distance(points):
            if task_distance is not None:     # the task's own distance from its line (seam-to-notch)
                return task_distance(self, points, A)
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
        folder = _verifier(task)
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
        if getattr(folder, "BACKTACKS", True):
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
        if task == "three-sides" or getattr(folder, "CORNERS", False):
            line = seam_path(task, self.env.world.config.ply_width, self.env.world.config.ply_length)
            for i, corner in enumerate(line[1:-1]):
                # A matching corner in BOTH plies, not just a top-ply stitch.
                d = min((max(np.linalg.norm(np.asarray(s["holes"][k]) - corner) for k in (0, 1))
                         for s in g["two"]), default=float("inf"))
                c[f"corner_{i + 1}"] = _check(d <= TOL_LINE, None if not np.isfinite(d) else d*1e3,
                                              "<= 1.6 mm from the square corner, both plies")
        if hasattr(folder, "extra_checks"):   # checks of the task's own (seam-to-notch: the mark and the opening)
            folder.extra_checks(self, g, c)
        return c

    def _seam_progress(self, checks):
        g, M = self._curve, self.env.world.machine
        folder = _verifier(self.env.config.task)
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
        if hasattr(folder, "extra_strays"):   # rework of the task's own (seam-to-notch: punctures in the opening)
            strays += folder.extra_strays(self, g)
        need = (g["ell"] - 2*END_REACH)/self.env.world.config.stitch_length
        rework = max(0., 1 - strays/need)
        damage = min(CURVE_DAMAGE.get(M.needle_state, 1.), CURVE_FAULT if broken else 1.)
        trimmed = M.tether == "trimmed"
        let_go = trimmed and checks["released"]["ok"]
        finish = (folder.FINISH if hasattr(folder, "FINISH") else .80 + .05*sum(self._secured))
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

    def evaluate(self):
        """The episode's evaluation (verdict): a benchmark task's verifier.py (tasks/<task>/) decides, its verify(e)
        calling verdict; a task outside the benchmark (three-sides) is a seam graded here."""
        folder = _verifier(self.env.config.task)
        if folder is not None:
            return folder.verify(self)
        return self.verdict()

    def verdict(self, checks=None, *, primary=None, partial=None, outcome=None, blend=None):
        """success: every check passes; score: 1.0 on success, otherwise the failure factor times the continuous
        progress; primary_outcome: progress above 0. A seam is graded on its target line (_seam_checks,
        _seam_progress), with the task's hooks from its verifier.py; outcome(e), a task's own checks and progress
        instead of a seam (load, square-plies); blend(e, progress), the seam's progress adjusted by the task
        (load-and-seam). checks, primary and partial are the released contract's (task.Evaluator.verdict) and are not
        used here."""
        self._info = {}
        task = self.env.config.task
        if outcome is not None:
            checks, progress = outcome(self)
        else:
            checks = self._seam_checks()
            progress = self._seam_progress(checks)
            if blend is not None:
                progress = blend(self, progress)
        success = all(c["ok"] for c in checks.values())
        progress = float(np.clip(progress, 0., 1.))
        self._info["continuous_progress"] = progress
        return dict(task=task, task_contract=CONTRACT, success=success,
                    score=1. if success else CURVE_PARTIAL_MAX*progress,
                    primary_outcome=progress > 0., checks=checks, info=self._info)

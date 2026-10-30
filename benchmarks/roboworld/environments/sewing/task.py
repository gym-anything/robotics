"""Tasks of the sewing environment (preset sample-room) and the physical-state evaluator.

Every judgment reads the simulated world: the machine's stitch ledger (holes at material points of the plies, formed
interlocks, thread elements, faults), the plies' vertex positions in their material coordinates, the robots' reflex
log and the cloth's contacts with the hands. Thresholds come from published specifications (training-manual seam
tolerance 1/16 in, Army stitch density 11 +- 1 SPI applied about the set 10 SPI, FED-STD-751a balance, industry
ends-meet < 3 mm, Army tail 38 mm); the pucker gate is authored (AATCC 88B is photographic).

Material frame of a ply: u across the seam (the seam edge is u = width), v along it. The stack starts with its far
end (v = length) under the foot just past the needle, so a seam runs from v ~ length toward v = 0 as the machine feeds
the plies away from the operator.
"""

import math

import numpy as np

PRESETS = ("sample-room",)
TASKS = {"load": "A0", "plain-seam": "A1", "baste": "A2", "pivot": "A5", "curve": "A6"}
INSTRUCTIONS = {
    "load": "Lay the two muslin plies from the left of the table, edges aligned, under the raised presser foot so the "
            "needle will enter 10 mm from their right edges and 3-8 mm from their far ends, then lower the foot.",
    "plain-seam": "Join the two plies with a plain seam 10 mm from their right edges along the whole length. Backtack "
                  "three or four stitches at both ends, trim the threads and let go of the work.",
    "baste": "Baste the two plies together 6 mm from their right edges with the 5 mm stitch, without backtacks, along "
             "the whole length; trim the threads and let go of the work.",
    "pivot": "Sew 10 mm from the right edge, turn the corner with the needle down, and continue 10 mm from the near "
             "edge to the left edge; trim the threads and let go of the work.",
    # the instruction names what is sewn, not a method (the outcome is judged, not the technique; a curve is also taken
    # by small needle-down pivots)
    "curve": "Join the two plies with a seam 10 mm from their edge, from the far end of the right edge around the "
             "rounded corner and along the near edge to the left edge; trim the threads and let go of the work.",
}
# the G1 drives the machine as the arm cells do, through the head's control box (tool_io); a standing person would use
# the pedal, the knee lifter and the touch-back switch, which no robot here can reach
G1_MACHINE_NOTE = (" The machine is driven through its control box (tool_io: speed, needle stop, presser lift, reverse, "
                   "trim), as an automated cell wires it; the G1 does not work the pedal, the knee lifter or the "
                   "touch-back switch, which a person standing at this machine would use.")


def instruction(task, robot=None):
    """The task's instruction for a deployment: the G1's adds how it drives the machine (G1_MACHINE_NOTE)."""
    text = INSTRUCTIONS[task]
    return text + G1_MACHINE_NOTE if robot == "g1" else text


# start states and per-task machine/ply settings (the policy may change speed, presser, needle stop, reverse, trim)
TASK_SETUP = {
    "load": dict(start="tray", ply_width=0.200, ply_length=0.420, stitch_length=2.5e-3),
    "plain-seam": dict(start="under_foot", ply_width=0.200, ply_length=0.420, stitch_length=2.5e-3),
    "baste": dict(start="under_foot", ply_width=0.200, ply_length=0.420, stitch_length=5.0e-3,
                  seam_allowance=0.006, needle_tension=0.35),
    "pivot": dict(start="under_foot", ply_width=0.200, ply_length=0.200, stitch_length=2.5e-3),
    "curve": dict(start="under_foot", ply_width=0.200, ply_length=0.300, stitch_length=2.5e-3, corner_radius=0.100),
}
TIME_LIMIT_S = {"load": 60.0, "plain-seam": 120.0, "baste": 120.0, "pivot": 120.0, "curve": 120.0}

# Append only: A3/A4/A7-A9 remain reserved for the design's tasks that are not built. Task setup and
# evaluation live outside env.SOURCE_FILES; GAR constructs task_set_env.TaskEnvironment.
TASKS.update({"seam-to-notch": "A10", "square-plies": "A11", "three-sides": "A12", "load-and-seam": "A13"})
INSTRUCTIONS.update({
    "seam-to-notch": "Join the two plies with a seam 10 mm from their right edges, from the far end to a point "
                     "120 mm from that end, leaving the rest open; backtack three or four stitches at both ends, "
                     "trim the threads and let go of the work.",
    "square-plies": "The top ply has slipped on the stack at the left of the table. Lay it back square on the "
                    "bottom ply with the edges matching, and leave the stack flat on the table.",
    "three-sides": "Join the square plies with a seam 10 mm from their edges along three sides: the right edge, "
                   "the near edge and the left edge, with square corners and the far edge left open; backtack "
                   "both ends, trim the threads and let go of the work.",
    "load-and-seam": "Lay the two muslin plies from the left of the table under the presser foot, edges aligned, "
                     "and join them with a plain seam 10 mm from their right edges along the whole length; "
                     "backtack three or four stitches at both ends, trim the threads and let go of the work.",
})
for _task, _base in (("seam-to-notch", "plain-seam"), ("square-plies", "load"),
                     ("three-sides", "pivot"), ("load-and-seam", "load")):
    TASK_SETUP[_task] = dict(TASK_SETUP[_base])
    TIME_LIMIT_S[_task] = 180.0 if _task in ("three-sides", "load-and-seam") else TIME_LIMIT_S[_base]

TOL_LINE = 1.6e-3            # 1/16 in seam-line tolerance [read: training manual]
TOL_MEAN = 0.8e-3            # mean seam allowance [authored: half the line tolerance]
SPI_BAND = (9.0, 11.0)       # per-inch window about the set 10 SPI [read: Army 11 +- 1, applied about the setting]
STITCH_BOUNDS = (1.5e-3, 3.5e-3)
ENDS_MEET = 3.0e-3           # [read: industry < 3 mm]
BALANCE_MIN = 0.95           # [read: FED-STD-751a; fraction authored]
TAIL_MAX = 38e-3             # [read: Army major-defect boundary]
PUCKER_RMS = 0.5e-3          # [authored, no published numeric gate]
RAW_EDGE_STRIP = 4e-3        # the seam allowance's outer strip left out of the pucker band: there the cut edge's lift
                             # at each feed stroke is a first-order tick artifact (E17b; it halves at half the tick)
CONTRACTION = 0.02           # [authored]
BACKTACK_MIN = 3             # [literature: practice 3-4 stitches]
END_REACH = 6e-3             # interlocks within 6 mm of each end [authored]
INCH = 25.4e-3
STEP_MAX = 3.5e-3            # the curve: the longest step between neighbouring stitches along the seam, per stitch
                             # [authored, the pivot's `continuous`]
NOT_A_SKIP = ("no_fabric", "needle_bent", "needle_broken")   # the curve: unformed cycles that are not skipped stitches
                                                             # (no work under the needle; no needle: needle_intact)
THREAD_LINK_MAX = 10e-3      # the curve: the longest thread between consecutive stitches of one seam piece (between
                             # trims), in the work [authored: a skipped stitch's 5 mm and a restart up to four stitches
                             # back over the seam pass; a jump across the work left without a trim does not]
SAME_PLACE = 0.5e-3          # the curve: stitches within this of a place's first stitch along the seam are one place
JOIN_SECURE = 25.4e-3/4      # the curve: the seam inside each end of a trimmed piece, within the seam, sewn at least
                             # twice (backtacked, locked or overlapped) [read: GL-PD-20023 3.5.2, ends "backstitched
                             # or overstitched not less than 1/4 inch"; MIL-DTL-32184 asks 1/2 in for a restart]
# the curve's partial score (Evaluator._partial_curve) [authored, none measured]
CURVE_SOFT = 1.6e-3          # a stitch's credit falls linearly from 1 at TOL_LINE off the line to 0 this much further
CURVE_SOFT_REG = 3.0e-3      # and from 1 with the plies' material points ENDS_MEET apart to 0 this much further
CURVE_SEAM_SHARE = 0.90      # the seam's share; 0.05 more once trimmed and 0.05 more for letting go once trimmed
CURVE_DAMAGE = {"damaged": 0.75, "bent": 0.5, "broken": 0.5}  # factor by the needle's state
CURVE_FAULT = 0.75           # factor for a broken thread (a protective stop is judged by success only)
CURVE_PARTIAL_MAX = 0.95     # the most a seam that fails a check scores: 1.0 is success only


def _check(ok, value=None, target=None, note=None):
    d = dict(ok=bool(ok))
    if value is not None:
        d["value"] = value
    if target is not None:
        d["target"] = target
    if note:
        d["note"] = note
    return d


def _polyline_distance(points, poly):
    """Distance of each point to the polyline and the arc length of its nearest point along it."""
    P = np.asarray(points, float)
    best = np.full(len(P), np.inf)
    at = np.zeros(len(P))
    s0 = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab)/max(L2, 1e-18), 0.0, 1.0)
        q = a + t[:, None]*ab
        d = np.linalg.norm(P - q, axis=1)
        m = d < best
        best[m] = d[m]
        at[m] = s0 + t[m]*math.sqrt(L2)
        s0 += math.sqrt(L2)
    return best, at



_PAIRS_AT_ONCE = 500_000     # point-triangle pairs _triangle_under holds in memory at a time (a few tens of MB)


def _ranges(start, count):
    """The integer ranges start[i] .. start[i] + count[i] - 1, concatenated: (i of every element, the elements)."""
    owner = np.repeat(np.arange(len(count)), count)
    return owner, np.repeat(start - (np.cumsum(count) - count), count) + np.arange(len(owner))


def _triangle_under(q, p0, p1, p2):
    """For each plan point q[i] the triangle (p0, p1, p2)[t] under it, or the one reaching nearest to it where none
    is: the triangle whose smallest barycentric coordinate at the point is largest, the first of equals. Returns
    (t, l0, l1, l2), the triangle and the point's barycentric coordinates in it.

    The result is that of trying every triangle for every point (np.argmax over all of them: 1.4 s a call on the load
    task, paid by every request), and the coordinates are computed by the same arithmetic, so they are the same to the last bit. Only the search is
    shortened, by a bound that cannot drop the answer: a triangle's smallest coordinate is at least f only inside the
    triangle enlarged about its centroid by 1 - 3 f, so once some triangle gives a point the value f, the only others
    that can equal or beat it have the point inside their box about the centroid enlarged by that factor. The
    triangles are sorted by size into octaves, each with a grid over its centroids whose cell is the octave's largest
    box; a point looks at the cells its enlarged boxes can lie in, first to twice the box (every triangle the point is
    in, and the near ones), then as far as the value it has found requires (doubling while it has found none), the
    large triangles first because they reach furthest. A window covering more than an eighth of an octave's grid, and
    an octave of at most 256 triangles, is searched whole. The bound is exact for exact arithmetic and is applied
    with a margin of 1e-6 of the box, orders of magnitude above the rounding of a sound triangle's coordinates
    (about 1e-11 of their size); triangles too thin in plan for that (twice the area under 1e-4 of the longest edge
    squared: a fold seen edge-on) are not bounded at all and are tried for every point, and so is every triangle when
    a coordinate is not a number. A test holds the result to the exhaustive search's on the evaluator's fixtures and
    on randomised stacks (test_sewing_evaluator.py, LoadSearchEqualityTest)."""
    n, m = len(q), len(p0)
    v0, v1 = p1 - p0, p2 - p0
    raw = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
    den = np.where(np.abs(raw) < 1e-18, 1e-18, raw)
    qx, qy = np.ascontiguousarray(q[:, 0]), np.ascontiguousarray(q[:, 1])
    best = np.full(n, -np.inf)              # the largest smallest coordinate found for each point
    tri = np.full(n, m, dtype=np.int64)     # and its triangle

    def coordinates(qi, t):
        ax, ay = qx[qi] - p0[t, 0], qy[qi] - p0[t, 1]
        l1 = (ax*v1[t, 1] - v1[t, 0]*ay)/den[t]
        l2 = (v0[t, 0]*ay - ax*v0[t, 1])/den[t]
        return 1.0 - l1 - l2, l1, l2

    def offer(qi, f, t):
        better = (f > best[qi]) | ((f == best[qi]) & (t < tri[qi]))
        qi = qi[better]
        best[qi], tri[qi] = f[better], t[better]

    def whole(qi, idx, only=False):
        """Every triangle of idx (ascending) for the points qi; only: idx is every triangle there is."""
        a, b0, b1, d = p0[idx], v0[idx], v1[idx], den[idx]
        step = max(1, min(4096, 1_500_000//len(idx)))
        for s in range(0, len(qi), step):
            k = qi[s:s + step]
            ax, ay = qx[k][:, None] - a[:, 0], qy[k][:, None] - a[:, 1]
            l1 = (ax*b1[:, 1] - b1[:, 0]*ay)/d
            l2 = (b0[:, 0]*ay - ax*b0[:, 1])/d
            f = np.minimum(np.minimum(1.0 - l1 - l2, l1), l2)
            t = np.argmax(f, axis=1)
            if only:
                best[k], tri[k] = f[np.arange(len(k)), t], idx[t]
            else:
                offer(k, f[np.arange(len(k)), t], idx[t])

    if n == 0 or m == 0:
        return tri, np.zeros(n), np.zeros(n), np.zeros(n)
    # few pairs, or a coordinate that is not a number (the bound reasons about finite geometry): the plain search
    if n*m <= 200_000 or not all(np.isfinite(a).all() for a in (q, p0, p1, p2)):
        whole(np.arange(n), np.arange(m), only=True)
        return (tri, *coordinates(np.arange(n), tri))
    c = (p0 + p1 + p2)/3.0
    box = np.maximum(np.maximum(np.abs(p0 - c), np.abs(p1 - c)), np.abs(p2 - c))    # half extents about the centroid
    longest = np.maximum(np.maximum((v0**2).sum(axis=1), (v1**2).sum(axis=1)), ((v1 - v0)**2).sum(axis=1))
    sound = (np.abs(raw) >= 1e-18) & (np.abs(raw) >= 1e-4*longest)
    if not sound.all():
        whole(np.arange(n), np.nonzero(~sound)[0])
    size = box.max(axis=1)
    octaves = []
    if sound.any():
        octave = np.zeros(m, dtype=np.int64)
        octave[sound] = np.clip(np.floor(np.log2(size[sound].max()/size[sound])), 0, 12).astype(np.int64)
        for k in np.unique(octave[sound]):                      # ascending: the largest triangles first
            idx = np.nonzero(sound & (octave == k))[0]
            x0, y0 = c[idx, 0].min(), c[idx, 1].min()
            ex, ey = c[idx, 0].max() - x0, c[idx, 1].max() - y0
            h = 2.0*float(size[idx].max())
            while (int(ex/h) + 1)*(int(ey/h) + 1) > 4*len(idx) + 64:
                h *= 2.0
            nx, ny = int(ex/h) + 1, int(ey/h) + 1
            cell = np.floor((c[idx, 1] - y0)/h).astype(np.int64)*nx + np.floor((c[idx, 0] - x0)/h).astype(np.int64)
            octaves.append(dict(idx=idx, by_cell=idx[np.argsort(cell, kind="stable")], size=float(size[idx].max()),
                                first=np.concatenate([[0], np.cumsum(np.bincount(cell, minlength=nx*ny))]),
                                h=h, x0=x0, y0=y0, nx=nx, ny=ny))
    active = np.arange(n)
    reach = np.full(n, 2.0)                 # the enlargement each point's search goes to in this round
    while len(active) and octaves:
        searched = np.full(len(active), np.inf)
        for o in octaves:
            # the enlargement the value found so far requires (infinite while none is found), at most this round's
            s = np.minimum(reach[active], np.maximum(1.0, 1.0 - 3.0*best[active]))
            R = (s*(o["size"]*(1.0 + 1e-6)) + 1e-12)*(1.0 + 1e-9) + 1e-9
            nx, ny, h = o["nx"], o["ny"], o["h"]
            ix0 = np.clip(np.floor((qx[active] - R - o["x0"])/h), 0, nx).astype(np.int64)
            ix1 = np.clip(np.floor((qx[active] + R - o["x0"])/h), -1, nx - 1).astype(np.int64)
            iy0 = np.clip(np.floor((qy[active] - R - o["y0"])/h), 0, ny).astype(np.int64)
            iy1 = np.clip(np.floor((qy[active] + R - o["y0"])/h), -1, ny - 1).astype(np.int64)
            cols, rows = np.maximum(ix1 - ix0 + 1, 0), np.maximum(iy1 - iy0 + 1, 0)
            every = (cols*rows*8 >= nx*ny) | (len(o["idx"]) <= 256)
            if every.any():
                whole(active[every], o["idx"])
                s = np.where(every, np.inf, s)
            g = np.nonzero(~every & (cols > 0) & (rows > 0))[0]
            if len(g):
                # the triangles in a row of cells are consecutive in by_cell: one range per point and row
                row_of, iy = _ranges(iy0[g], rows[g])
                lo = o["first"][iy*nx + ix0[g][row_of]]
                count = o["first"][iy*nx + ix1[g][row_of] + 1] - lo
                per_point = np.bincount(row_of, weights=count, minlength=len(g))
                part = ((np.cumsum(per_point) - per_point)//_PAIRS_AT_ONCE).astype(np.int64)
                row_first = np.searchsorted(row_of, np.concatenate([[0], np.nonzero(np.diff(part))[0] + 1, [len(g)]]))
                for ra, rb in zip(row_first[:-1], row_first[1:]):
                    pair_row, pos = _ranges(lo[ra:rb], count[ra:rb])
                    gi = g[row_of[ra:rb][pair_row]]
                    qi, t = active[gi], o["by_cell"][pos]
                    # the bound: only a triangle whose enlarged box holds the point can reach the value found
                    sq = s[gi]*(1.0 + 1e-6)
                    keep = ((np.abs(qx[qi] - c[t, 0]) <= sq*box[t, 0] + 1e-12)
                            & (np.abs(qy[qi] - c[t, 1]) <= sq*box[t, 1] + 1e-12))
                    qi, t = qi[keep], t[keep]
                    if not len(t):
                        continue
                    l0, l1, l2 = coordinates(qi, t)
                    f = np.minimum(np.minimum(l0, l1), l2)
                    new = np.concatenate([[True], qi[1:] != qi[:-1]])           # a point's pairs are consecutive
                    start = np.nonzero(new)[0]
                    top = np.maximum.reduceat(f, start)
                    offer(qi[start], top, np.minimum.reduceat(np.where(f == top[np.cumsum(new) - 1], t, m), start))
            searched = np.minimum(searched, s)
        need = np.maximum(1.0, 1.0 - 3.0*best[active])
        more = searched < need
        active, need, searched = active[more], need[more], searched[more]
        reach[active] = np.where(np.isfinite(need), need, 2.0*searched)
    return (tri, *coordinates(np.arange(n), tri))


def _spans(seq, stitches):
    """For consecutive stitches of seq: how many stitch settings each step spans, 1 plus the skipped stitches (the
    hook missed the loop and the feed went on) whose cycle lies between them."""
    missed = [s["cycle"] for s in stitches if not s["formed"] and s["cycle"] is not None]
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        if a["cycle"] is None or b["cycle"] is None:
            out.append(1)
            continue
        lo, hi = sorted((a["cycle"], b["cycle"]))
        out.append(1 + sum(1 for k in missed if lo < k < hi))
    return np.asarray(out, float)


def _skips(stitches, ignore=()):
    """At most one isolated skipped stitch (FED-STD-751a): the count and the longest run. `ignore`: reasons
    of unformed cycles that are not skipped stitches, left out (the curve: NOT_A_SKIP)."""
    stitches = [s for s in stitches if s["formed"] or s["reason"] not in ignore]
    runs = run = 0
    for s in stitches:
        run = run + 1 if not s["formed"] else 0
        runs = max(runs, run)
    n = sum(1 for s in stitches if not s["formed"])
    return _check(runs < 2 and n <= 1, dict(isolated=n, longest_run=runs))


def _longest_step(at, skipped):
    """The longest step between neighbouring places `at` along a seam, each divided by the stitches it spans: 1 plus
    the skipped stitches whose holes lie between them (places `skipped`, sorted), as _spans counts them in time."""
    p = np.sort(np.asarray(at, float))
    if len(p) < 2:
        return 0.0
    inside = np.searchsorted(skipped, p[1:], side="left") - np.searchsorted(skipped, p[:-1], side="right")
    return float((np.diff(p)/(1 + np.maximum(inside, 0))).max())


class Evaluator:
    def __init__(self, env):
        self.env = env
        self._info = {}

    # ------------------------------------------------------------------ measurements
    def stitches(self):
        """Formed stitches in order, each with its holes per ply (material u, v), the reverse flag and the seam piece
        its interlock belongs to (the machine starts a new one at the first stitch after a trim; None when unknown)."""
        M = self.env.world.machine
        interlocks = getattr(M, "interlocks", None)
        out = []
        for s in M.stitches:
            holes = [M.holes[i] for i in s.get("holes", [])]
            seam = s.get("seam")
            if seam is None and interlocks and s.get("interlock") is not None:
                seam = interlocks[s["interlock"]].get("seam")
            out.append(dict(formed=bool(s.get("formed")), reverse=bool(s.get("reverse")), cycle=s.get("cycle"),
                            reason=s.get("reason"), balance=s.get("balance"),
                            holes={h.ply: (h.uv[0], h.uv[1]) for h in holes}, n_holes=len(holes),
                            plies=sorted(h.ply for h in holes), seam=seam))
        return out

    def ply_corners(self, k):
        """World xy of the ply's material corners (seam-edge near and far ends)."""
        w = self.env.world
        X = w.mesh["X"]
        x = w.ply_positions(k)
        W, L = w.config.ply_width, w.config.ply_length
        out = {}
        for name, uv in (("seam_near", (W, 0.0)), ("seam_far", (W, L)), ("left_near", (0.0, 0.0)),
                         ("left_far", (0.0, L))):
            i = int(np.argmin(np.linalg.norm(X - np.asarray(uv), axis=1)))
            out[name] = x[i]
        return out

    def ply_registration(self, band=3e-3, support=2e-3):
        """How far the plies' cut edges lie from each other in material coordinates where the two lie stacked on the
        table or the bed (distances are in the plies' material coordinates). For every vertex within
        `band` of a cut edge of one ply that rests on the support (within `support` of the table top, inside the
        table's footprint), the other ply's material point at the same place in plan (from its resting triangle there,
        barycentric, extrapolated linearly where that ply does not reach) is compared with the vertex's own: the
        largest difference, and how many edge points were judged. A part hanging over the table's edge is not judged:
        two plies draped over an edge hang apart by their thickness and swing (comparing the plies' corners in the
        world reads 3.6 mm between the hanging near corners of two plies registered to 0.2 mm on the plate)."""
        from .layout import TABLE
        w = self.env.world
        X = np.asarray(w.mesh["X"], float)
        tri = np.asarray(w.mesh["tri"], int)
        W, L = w.config.ply_width, w.config.ply_length
        xs = [np.asarray(w.ply_positions(k), float) for k in (0, 1)]

        def resting(x):
            return ((x[:, 2] >= -1e-3) & (x[:, 2] <= support) & (x[:, 0] > TABLE["x"][0]) & (x[:, 0] < TABLE["x"][1])
                    & (x[:, 1] > TABLE["y"][0]) & (x[:, 1] < TABLE["y"][1]))
        edge = np.minimum(np.minimum(X[:, 0], W - X[:, 0]), np.minimum(X[:, 1], L - X[:, 1])) <= band
        worst, n = 0.0, 0
        for a, b in ((0, 1), (1, 0)):
            q_idx = np.nonzero(edge & resting(xs[a]))[0]
            ok_b = resting(xs[b])
            tb = tri[ok_b[tri].all(axis=1)]
            if not len(q_idx) or not len(tb):
                continue
            P = xs[b][:, :2]
            # the triangle under each edge point (or nearest) and the point's place in it
            t, l0, l1, l2 = _triangle_under(xs[a][q_idx, :2], P[tb[:, 0]], P[tb[:, 1]], P[tb[:, 2]])
            Xb = l0[:, None]*X[tb[t, 0]] + l1[:, None]*X[tb[t, 1]] + l2[:, None]*X[tb[t, 2]]
            d = np.linalg.norm(Xb - X[q_idx], axis=1)
            worst = max(worst, float(d.max()))
            n += len(q_idx)
        return worst, n

    def under_needle(self, k):
        """Material (u, v) of ply k under the needle axis and how many times the axis crosses the ply (folds)."""
        w = self.env.world
        cl = w.cloth
        p = cl.plies[w.ply_ids[k]]
        x = cl.positions()
        T = cl.tri_host[p["t0"]:p["t0"] + p["nt"]]
        X = cl.X
        # every triangle of the ply at once, by the same arithmetic as a loop over them (which takes 0.08 s a ply on
        # the load task, in every request's state); a triangle edge-on in plan is skipped
        a = x[T[:, 0], :2]
        v0, v1 = x[T[:, 1], :2] - a, x[T[:, 2], :2] - a
        den = v0[:, 0]*v1[:, 1] - v1[:, 0]*v0[:, 1]
        ok = ~(np.abs(den) < 1e-18)
        T, a, v0, v1, den = T[ok], a[ok], v0[ok], v1[ok], den[ok]
        q0, q1 = -a[:, 0], -a[:, 1]
        l1 = (q0*v1[:, 1] - v1[:, 0]*q1)/den
        l2 = (v0[:, 0]*q1 - q0*v0[:, 1])/den
        l0 = 1 - l1 - l2
        hit = np.minimum(np.minimum(l0, l1), l2) >= -1e-9
        T, l0, l1, l2 = T[hit], l0[hit], l1[hit], l2[hit]
        return list(l0[:, None]*X[T[:, 0]] + l1[:, None]*X[T[:, 1]] + l2[:, None]*X[T[:, 2]])

    def released(self):
        """No hand touches the cloth (net cloth force on each arm's fingers and hand below 0.05 N)."""
        w = self.env.world
        return all(np.linalg.norm(w.finger_force(a)) < 0.05 for a in w.hand_arms)

    def hand_contact(self, arm):
        """The cloth's contact on an arm's hand (N): the sum of the magnitudes of its reactions on each of the hand's
        pads (World.pads: both fingers and the housing), read as World.finger_force reads them. finger_force adds them
        as vectors, so a pinch's two fingers, pressing the cloth from opposite sides, cancel in it: a Franka Hand
        closed on slack work at 1.15 N a finger reads 0.007 N in it and would pass `released`. A world without pad records (the evaluator's self-test stand-ins) reports the net force."""
        w = self.env.world
        pads = getattr(w, "pads", None)
        if pads is None:
            return float(np.linalg.norm(w.finger_force(arm)))
        return float(sum(np.linalg.norm(w.cloth.shape_reaction(i)[0]) for i, *_ in pads
                         if w.cloth.shapes[i]["name"].startswith(arm + "_")))

    def pucker(self, band=20e-3):
        """The seam's waviness and contraction. Waviness: RMS height about its mean of the top ply's seam band (20 mm
        from the cut edge, without the allowance's outer RAW_EDGE_STRIP, where the edge's lift at each feed stroke is a
        tick artifact, E17b) along the stitched length (between the first and last forward holes) where it lies over the
        table; the work hanging over the table's edges, unsewn ahead of the needle or sewn far behind it, is drape, not
        pucker (the whole band reads the unsewn plies' drape over the front edge as 8.8 mm of pucker). The thread
        elements relax nearly slack after the take-up sets them, so a tension pucker is
        not modelled: this check measures waves of the work and the feed's contraction, not the threads' pull.
        Contraction: material vs world length along the seam (the path through consecutive forward holes)."""
        from .layout import TABLE
        w = self.env.world
        X = w.mesh["X"]
        W = w.config.ply_width
        top = len(w.ply_ids) - 1
        x = w.ply_positions(top)
        M = w.machine
        fwd = [M.holes[i] for s in M.stitches if s.get("formed") and not s.get("reverse") for i in s.get("holes", [])
               if M.holes[i].ply == top]
        sel = (X[:, 0] > W - band) & (X[:, 0] < W - RAW_EDGE_STRIP)
        if fwd:
            v = [h.uv[1] for h in fwd]
            sel &= (X[:, 1] >= min(v)) & (X[:, 1] <= max(v))
        sel &= ((x[:, 0] > TABLE["x"][0]) & (x[:, 0] < TABLE["x"][1]) & (x[:, 1] > TABLE["y"][0])
                & (x[:, 1] < TABLE["y"][1]))
        z = x[sel, 2]
        rms = float(np.sqrt(np.mean((z - z.mean())**2))) if len(z) else 0.0
        contraction = None
        if len(fwd) >= 2:
            # along the seam: the world length of the path through consecutive forward holes against its material
            # length (a chord between the first and last holes read the sewn work's drape over the table's back edge as
            # 12 % contraction of a straight seam)
            cl = w.cloth
            xs = cl.positions()
            pts = np.array([np.dot(h.bary, xs[cl.tri_host[h.face]]) for h in fwd])
            uv = np.array([h.uv for h in fwd])
            order = np.argsort(uv[:, 1])
            pts, uv = pts[order], uv[order]
            mat = float(np.sum(np.linalg.norm(np.diff(uv, axis=0), axis=1)))
            world = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            contraction = float(1.0 - world/mat) if mat > 0 else None
        return rms, contraction

    # ------------------------------------------------------------------ checks
    def common(self, per_pad_release=False):
        """Every sewing task: needle intact, no robot reflex while the machine ran, threads trimmed, the work released
        (the tails, 3 mm after the trimmer, are recorded in the note). per_pad_release (the curve): no hand touches the
        work by the contact on each of its pads (hand_contact), which a pinch cannot cancel; the other tasks keep the
        net force (released)."""
        env = self.env
        M = env.world.machine
        c = {}
        c["needle_intact"] = _check(M.needle_state == "ok", M.needle_state,
                                    note="a needle that struck the plate or the hook ('damaged') is burred and is "
                                         "replaced in practice: it fails, as a bent or broken one does")
        c["no_reflex_while_sewing"] = _check(not env.history["reflex_while_sewing"],
                                             env.history["reflex_while_sewing"])
        trimmed = M.tether == "trimmed"
        c["threads_trimmed"] = _check(trimmed or not M.stitches, "trimmed" if trimmed else "attached",
                                      note=f"tails {M.tails['start']*1e3:.0f} / "
                                           f"{(M.tails['end'] or 0)*1e3:.0f} mm (<= {TAIL_MAX*1e3:.0f} mm)")
        if per_pad_release:
            contact = {a: self.hand_contact(a) for a in env.world.hand_arms}
            c["released"] = _check(all(f < 0.05 for f in contact.values()),
                                   {a: round(f, 3) for a, f in contact.items()},
                                   "< 0.05 N on each hand, its pads' contact forces added as magnitudes")
        else:
            c["released"] = _check(self.released())
        c["no_thread_break"] = _check(not any(t["broken"] for t in M.threads))
        return c

    def seam_checks(self, allowance, stitch_length, backtack=True, reverse_allowed=True):
        env = self.env
        w = env.world
        W, L = w.config.ply_width, w.config.ply_length
        st = self.stitches()
        formed = [s for s in st if s["formed"]]
        c = {}
        both = [s for s in formed if s["plies"] == [0, 1]]
        c["stitches_formed"] = _check(len(both) >= 3, len(both))
        if len(both) < 3:
            return c
        vs = sorted(s["holes"][1][1] for s in both)
        c["joined_far_end"] = _check(L - vs[-1] <= END_REACH, round((L - vs[-1])*1e3, 2), f"<= {END_REACH*1e3:g} mm")
        c["joined_near_end"] = _check(vs[0] <= END_REACH, round(vs[0]*1e3, 2), f"<= {END_REACH*1e3:g} mm")
        dists = [W - s["holes"][k][0] for s in both for k in (0, 1)]
        worst = max(abs(d - allowance) for d in dists)
        c["seam_line"] = _check(worst <= TOL_LINE, round(worst*1e3, 3), f"|d - {allowance*1e3:g}| <= 1.6 mm")
        c["seam_mean"] = _check(abs(np.mean(dists) - allowance) <= TOL_MEAN, round(float(np.mean(dists))*1e3, 3),
                                f"{allowance*1e3:g} +- 0.8 mm")
        fwd = sorted((s for s in both if not s["reverse"]), key=lambda s: s["holes"][1][1])
        fv = [s["holes"][1][1] for s in fwd]
        steps = np.diff(fv)
        if len(steps):
            # a skipped stitch (the hook missed the loop; the feed went on) leaves a span of two settings between its
            # neighbours: each step is judged against the stitches it spans (the skip itself is judged by `skips`)
            lo, hi = 0.6*stitch_length, 1.4*stitch_length
            per = steps/_spans(fwd, st)
            c["stitch_length"] = _check(bool(((per >= lo) & (per <= hi)).all()),
                                        [round(float(per.min())*1e3, 2), round(float(per.max())*1e3, 2)],
                                        f"{lo*1e3:.1f}-{hi*1e3:.1f} mm")
            if abs(stitch_length - 2.5e-3) < 1e-9:
                spi = []
                for v0 in np.arange(fv[0], fv[-1] - INCH, 2e-3):
                    n = int(np.sum((np.asarray(fv) >= v0) & (np.asarray(fv) < v0 + INCH)))
                    spi.append(n)
                if spi:
                    c["stitch_density"] = _check(min(spi) >= SPI_BAND[0] and max(spi) <= SPI_BAND[1] + 1,
                                                 [min(spi), max(spi)], "9-11 SPI per inch window")
        c["skips"] = _skips(st)
        rev = [s for s in both if s["reverse"]]
        if backtack:
            # the seam's middle (all stitches when none was sewn forward: a seam that is only a backtack)
            mid = (fv[0] + fv[-1])/2 if fv else (vs[0] + vs[-1])/2
            far = [s for s in rev if s["holes"][1][1] > mid]
            near = [s for s in rev if s["holes"][1][1] <= mid]
            c["backtack_start"] = _check(len(far) >= BACKTACK_MIN, len(far), ">= 3 reverse stitches")
            c["backtack_end"] = _check(len(near) >= BACKTACK_MIN, len(near), ">= 3 reverse stitches")
        elif not reverse_allowed:
            c["no_reverse"] = _check(not rev, len(rev))
        odd = sum(1 for s in formed if len(s["plies"]) != len(set(s["plies"])))   # a ply pierced twice: a fold
        c["no_tucks"] = _check(odd == 0 and max(dists) <= 20e-3,
                               dict(stitches_through_a_fold=odd, farthest_hole_mm=round(max(dists)*1e3, 2)),
                               note="no hole beyond 20 mm of the edge (a fold carried under the needle puts one there);"
                                    " no ply pierced twice by one stitch")
        self._balance([s for s in both])
        k0, k1 = self.ply_corners(0), self.ply_corners(1)
        off = max(abs(k0["seam_near"][1] - k1["seam_near"][1]), abs(k0["seam_far"][1] - k1["seam_far"][1]))
        c["ends_meet"] = _check(off <= ENDS_MEET, round(float(off)*1e3, 2), "<= 3 mm")
        rms, con = self.pucker()
        # recorded, not scored: one input sewed a flat seam or one curled at its start
        # (1 of 6 identical bench seams curled 16.5 mm), so the check passed or failed partly by chance whatever the
        # policy did; it returns to the score once repeated identical seams agree
        self._info["pucker"] = dict(value=dict(rms_mm=round(rms*1e3, 3),
                                               contraction=None if con is None else round(con, 4)),
                                    ok=bool(rms <= PUCKER_RMS and (con is None or con <= CONTRACTION)),
                                    target="RMS <= 0.5 mm, contraction <= 2 %",
                                    note="not scored while identical inputs give two seam shapes")
        return c

    def _eval_load(self):
        env = self.env
        w = env.world
        W, L = w.config.ply_width, w.config.ply_length
        c = {}
        c["foot_down"] = _check(w.machine.presser == "down" and w.machine.foot_N > 1.0, round(w.machine.foot_N, 2))
        for k in (0, 1):
            hits = self.under_needle(k)
            c[f"ply{k}_single_layer"] = _check(len(hits) == 1, len(hits))
            if hits:
                u, v = hits[0]
                c[f"ply{k}_seam_allowance"] = _check(abs((W - u) - 0.010) <= TOL_LINE, round((W - u)*1e3, 2),
                                                     "10 +- 1.6 mm")
                c[f"ply{k}_from_end"] = _check(3e-3 <= L - v <= 8e-3, round((L - v)*1e3, 2), "3-8 mm")
        off, n_edge = self.ply_registration()
        c["edges_aligned"] = _check(n_edge > 0 and off <= 1.5e-3,
                                    dict(misregistration_mm=round(off*1e3, 2), edge_points=n_edge),
                                    "<= 1.5 mm in material coordinates, where the plies lie stacked")
        c["no_reflex"] = _check(not w.reflexes, len(w.reflexes))
        return c

    def _eval_plain_seam(self):
        c = self.seam_checks(0.010, 2.5e-3, backtack=True)
        c.update(self.common())
        return c

    def _eval_baste(self):
        c = self.seam_checks(0.006, 5.0e-3, backtack=False, reverse_allowed=False)
        c.update(self.common())
        return c

    def _eval_pivot(self):
        env = self.env
        w = env.world
        W, L = w.config.ply_width, w.config.ply_length
        st = [s for s in self.stitches() if s["formed"] and s["plies"] == [0, 1]]
        c = {}
        c["stitches_formed"] = _check(len(st) >= 6, len(st))
        if len(st) >= 6:
            uv = np.array([s["holes"][1] for s in st])
            corner = np.array([W - 0.010, 0.010])
            d = np.linalg.norm(uv - corner, axis=1)
            k = int(np.argmin(d))
            c["corner"] = _check(d[k] <= TOL_LINE, round(float(d[k])*1e3, 2), "<= 1.6 mm from (10, 10) mm")
            leg1, leg2 = uv[:k + 1], uv[k:]
            e1 = np.abs((W - leg1[:, 0]) - 0.010).max() if len(leg1) else 0.
            e2 = np.abs(leg2[:, 1] - 0.010).max() if len(leg2) else 0.
            c["leg1_line"] = _check(e1 <= TOL_LINE, round(float(e1)*1e3, 3))
            c["leg2_line"] = _check(e2 <= TOL_LINE, round(float(e2)*1e3, 3))
            gaps = np.linalg.norm(np.diff(uv, axis=0), axis=1)/_spans(st, self.stitches())
            c["continuous"] = _check(gaps.max() <= 3.5e-3, round(float(gaps.max())*1e3, 2),
                                     "<= 3.5 mm per stitch (a skipped stitch's span counts two)")
            c["skips"] = _skips(self.stitches())
            c["leg2_reaches_left_edge"] = _check(leg2[:, 0].min() <= END_REACH if len(leg2) else False,
                                                 round(float(leg2[:, 0].min())*1e3, 2) if len(leg2) else None)
        c.update(self.common())
        return c

    PRIMARY = {"plain-seam": ("stitches_formed",), "baste": ("stitches_formed",), "pivot": ("stitches_formed",),
               "curve": ("stitches_formed",),
               "load": ("ply0_single_layer", "ply1_single_layer")}

    def _curve_seam(self):
        """The curve's stitches measured on the work. Every formed stitch through exactly one hole in each ply (two),
        forward or reverse, in any order: each hole's distance from the edge path (d) and its place along the line 10 mm
        inside it (s: arc length from the far end of the right edge), the worse of its two holes off the line (dev), the
        two plies' material points apart (reg). Every formed stitch, whatever plies it binds (formed): its holes'
        distances from the edge (d_all) and its worse hole off the line (dev_formed). The places of the skipped
        stitches' holes on each ply (skip_s). The sequences of sewing on places (runs, indices into two; see the comments
        below): their stitches per inch on each ply, smallest and largest over every inch window inside a
        sequence (density, None without one), and each stitch's mean step between places over the inch about it
        (step_local). The thread between consecutive formed stitches of one seam piece, in the work (links): the
        needle thread from top hole to top hole, the bobbin's from bottom to bottom, the longer. The shortest seam
        sewn twice inside an end of a trimmed piece within the seam (joins, None without one). The line's length (ell)."""
        from .mesh import seam_edge_path
        w = self.env.world
        W, L, R, A = w.config.ply_width, w.config.ply_length, w.config.corner_radius, w.config.seam_allowance
        edge = seam_edge_path(W, L, R, n_arc=96)
        line = seam_edge_path(W, L, R, A, n_arc=96)
        st = self.stitches()
        formed = [s for s in st if s["formed"] and s["holes"]]
        two = [s for s in formed if s["plies"] == [0, 1]]
        uv = np.array([s["holes"][k] for s in two for k in (0, 1)], float).reshape(-1, 2)
        d = _polyline_distance(uv, edge)[0].reshape(-1, 2)
        at2 = _polyline_distance(uv, line)[1].reshape(-1, 2)
        allh = np.array([h for s in formed for h in s["holes"].values()], float).reshape(-1, 2)
        d_all = _polyline_distance(allh, edge)[0]
        cut = np.cumsum([0] + [len(s["holes"]) for s in formed])
        dev_formed = np.array([np.abs(d_all[cut[i]:cut[i + 1]] - A).max() for i in range(len(formed))])
        skipped = [s for s in st if not s["formed"] and s["reason"] not in NOT_A_SKIP]
        skip_s = [np.sort(_polyline_distance(np.array([s["holes"][k] for s in skipped if k in s["holes"]],
                                                      float).reshape(-1, 2), line)[1]) for k in (0, 1)]
        at = at2.mean(axis=1)
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

    def _eval_curve(self):
        """The seam 10 mm inside the curved edge (task A6), judged on the finished work, not on how or in which
        order it was sewn: every hole of every formed stitch within
        1/16 in of 10 mm from the edge path (right edge, the rounded corner, the near edge); the two plies joined at the
        same material points (within ends_meet's 3 mm); along the line 10 mm inside the edge each ply's stitches leave
        no gap and run from within 6 mm of the far end of the right edge to within 6 mm of the left edge; 9-11
        stitches an inch (12 allowed) wherever the seam was sewn continuously (task A6: "Density 9-11 SPI measured
        along the curve"); no thread left strung across the work between consecutive stitches of one seam piece.
        Every formed stitch through both plies counts, forward or reverse: a backtack, a lock over earlier stitches, a
        restart over the seam, a seam in trimmed pieces and a seam sewn from the left edge are the same seam. Cycles
        with no fabric under the needle are not skipped stitches. The authored 0.8 mm mean allowance is
        recorded, not judged (it fails seams inside the sourced 1/16 in); one-ply stitches are judged by seam_line."""
        g = self._curve = self._curve_seam()
        w = self.env.world
        A = w.config.seam_allowance
        n = len(g["two"])
        c = {}
        c["stitches_formed"] = _check(n >= 6, n)
        if n >= 6:
            worst = float(np.abs(g["d_all"] - A).max())
            c["seam_line"] = _check(worst <= TOL_LINE, round(worst*1e3, 3),
                                    f"|d - {A*1e3:g}| <= 1.6 mm, every hole of every formed stitch")
            self._info["seam_mean"] = dict(value=round(float(g["d"].mean())*1e3, 3),
                                           note="the mean allowance, recorded, not judged (the sourced "
                                                "tolerance is every hole within 1/16 in, seam_line)")
            reg = float(g["reg"].max())
            c["plies_registered"] = _check(reg <= ENDS_MEET, round(reg*1e3, 2),
                                           "<= 3 mm between the two plies' material points at a stitch")
            step = max(_longest_step(g["s"][:, k], g["skip_s"][k]) for k in (0, 1))
            c["no_gaps"] = _check(step <= STEP_MAX, round(step*1e3, 2),
                                  "<= 3.5 mm per stitch between neighbours along the seam, each ply (a skipped "
                                  "stitch's span counts two)")
            if abs(w.config.stitch_length - 2.5e-3) < 1e-9:
                dens = g["density"]
                c["stitch_density"] = _check(dens is None or (dens[0] >= SPI_BAND[0] and dens[1] <= SPI_BAND[1] + 1),
                                             None if dens is None else list(dens),
                                             "9-11 (12 allowed) stitches in every inch of continuous sewing, each "
                                             "ply")
            link = float(g["links"].max()) if len(g["links"]) else 0.0
            c["no_long_threads"] = _check(link <= THREAD_LINK_MAX, round(link*1e3, 2),
                                          "<= 10 mm of thread between consecutive stitches of one seam piece, in the "
                                          "work (a jump left without a trim; the trimmer cuts only the threads to "
                                          "the needle and the bobbin)")
            c["joins_secured"] = _check(g["joins"] is None or g["joins"] >= JOIN_SECURE - 1e-9,
                                        None if g["joins"] is None else round(g["joins"]*1e3, 2),
                                        ">= 6.35 mm (1/4 in) of seam sewn twice inside each end of a trimmed piece that "
                                        "lies within the seam (a backtack, a lock or another piece's overlap)")
            far = float(g["s"].min(axis=0).max())
            c["joined_far_end"] = _check(far <= END_REACH, round(far*1e3, 2),
                                         f"<= {END_REACH*1e3:g} mm along the seam, each ply")
            left = float(g["ell"] - g["s"].max(axis=0).min())
            c["reaches_left_edge"] = _check(left <= END_REACH, round(left*1e3, 2),
                                            f"<= {END_REACH*1e3:g} mm along the seam, each ply")
            c["skips"] = _skips(g["stitches"], ignore=NOT_A_SKIP)
            odd = sum(1 for s in g["stitches"] if s["formed"] and len(s["plies"]) != len(set(s["plies"])))
            c["no_tucks"] = _check(odd == 0, dict(stitches_through_a_fold=odd),
                                   note="a guard: the machine pierces a ply once a stroke, so it cannot fail on a "
                                        "machine-made ledger; a fold along the seam shows as a hole off the line "
                                        "(seam_line), a tuck across it as a gap (no_gaps) only when large")
            self._balance(g["two"])
        c.update(self.common(per_pad_release=True))
        return c

    def _partial_curve(self, checks):
        """The curve's score short of success (constants authored, none measured):
        the share of the seam's required span, 6 mm from either end along the line, sewn through both plies, each
        stitch covering half the 3.5 mm step either side of its place with its credit, the best stitch at a place
        counting once (overlaps add nothing). A stitch's credit is the least of: 1 within 1/16 in of the line falling
        linearly to 0 at 3.2 mm; 1 with the plies' material points within 3 mm falling to 0 at 6 mm apart; 1 with the
        mean step of its run's inch about it in the 9-12 stitches-an-inch band falling to 0 at 1.5 and 3.5 mm
        (STITCH_BOUNDS); 0 where the thread to it broke. Times 1 - strays / the stitches the required span takes, a stray
        being a formed stitch off the line or with its plies apart (credit 0 there, whatever plies it binds) or a thread
        longer than THREAD_LINK_MAX: rework, which re-sewing over the seam neither adds nor dilutes. Times the needle's
        damage factor (CURVE_DAMAGE) and CURVE_FAULT for a broken thread. Times 0.90, + 0.05 once the threads are
        trimmed and + 0.05 for letting go once trimmed; at most 0.95. The order, direction, speed and method of sewing,
        pivots, presser lifts and backtacks earn nothing and cost nothing; a protective stop is judged by success
        only."""
        g, M = self._curve, self.env.world.machine
        broken = {t.get("cycle") for t in M.threads if t["broken"]}
        place = np.minimum(np.clip(1.0 - (g["dev"] - TOL_LINE)/CURVE_SOFT, 0.0, 1.0),
                           np.clip(1.0 - (g["reg"] - ENDS_MEET)/CURVE_SOFT_REG, 0.0, 1.0))
        lo_band, hi_band = INCH/(SPI_BAND[1] + 1), INCH/SPI_BAND[0]
        m = g["step_local"]
        dens = np.ones(len(m))
        known = ~np.isnan(m)
        dens[known] = np.clip(np.minimum((m[known] - STITCH_BOUNDS[0])/(lo_band - STITCH_BOUNDS[0]),
                                         (STITCH_BOUNDS[1] - m[known])/(STITCH_BOUNDS[1] - hi_band)), 0.0, 1.0)
        credit = np.minimum(place, dens)
        credit[np.array([s["cycle"] in broken for s in g["two"]], bool)] = 0.0
        grid = np.arange(END_REACH, g["ell"] - END_REACH, 0.05e-3) + 0.025e-3
        best, owner = np.zeros(len(grid)), np.full(len(grid), -1)
        at = g["s"].mean(axis=1)
        for k in np.argsort(credit, kind="stable"):          # ascending: the best stitch at a place is written last
            if credit[k] > 0:
                a, b = np.searchsorted(grid, [at[k] - 0.5*STEP_MAX, at[k] + 0.5*STEP_MAX])
                best[a:b], owner[a:b] = credit[k], k
        coverage = float(best.mean()) if len(grid) else 0.0
        # rework: a stitch that is the best nowhere counts 1 - its place credit (on the line: nothing; off it:
        # in proportion); a formed one-ply stitch 1 - its line credit; a thread over THREAD_LINK_MAX 1
        dominated = np.ones(len(place), bool)
        dominated[np.unique(owner[owner >= 0])] = False
        one_ply = np.clip(1.0 - (g["dev_formed"][~g["is_two"]] - TOL_LINE)/CURVE_SOFT, 0.0, 1.0)
        strays = (float(np.sum(1.0 - place[dominated])) + float(np.sum(1.0 - one_ply))
                  + float(np.sum(g["links"] > THREAD_LINK_MAX)))
        need = (g["ell"] - 2*END_REACH)/self.env.world.config.stitch_length
        rework = max(0.0, 1.0 - strays/need)
        damage = min(CURVE_DAMAGE.get(M.needle_state, 1.0), CURVE_FAULT if broken else 1.0)
        trimmed = M.tether == "trimmed"
        let_go = trimmed and checks["released"]["ok"]
        score = CURVE_PARTIAL_MAX*damage*rework*coverage*(CURVE_SEAM_SHARE + 0.05*trimmed + 0.05*let_go)
        self._info["partial_score"] = dict(value=round(score, 4), coverage=round(coverage, 4), strays=round(strays, 3),
                                           rework_factor=round(rework, 4), damage=damage, trimmed=trimmed,
                                           let_go=let_go, path_mm=round(g["ell"]*1e3, 1),
                                           note="the score short of success (1.0 on success)")
        return score

    def _balance(self, stitches):
        """The stitches' balance, recorded but not scored: balance follows from the two set tensions through the
        capstan law (machine.py), which no task lets the policy change, so it is the same for every stitch a policy can
        make and a check of it could not fail."""
        if stitches:
            bal = sum(1 for s in stitches if s["balance"] == "balanced")/len(stitches)
            self._info["balance"] = dict(value=round(bal, 3), target=">= 0.95 balanced (FED-STD-751a)",
                                         note="not scored: set by the tensions, which the tasks fix")

    def evaluate(self):
        """success: every check passes. score: for the curve, 1.0 on success and otherwise its continuous partial
        score (_partial_curve), 0 with nothing sewn; for the other tasks the fraction of checks passed, and 0 while the
        task's primary outcome is absent (no stitch through both plies; for load, the plies not lying under the
        needle): the common checks (needle intact, released, ...) hold for a policy that does nothing and must not
        score it. info: recorded quantities that are not checks (balance; the curve's partial score and its parts)."""
        task = self.env.config.task
        self._info = {}
        name = task.replace("-", "_")
        checks = getattr(self, "_eval_" + name)()
        passed = [v["ok"] for v in checks.values()]
        primary = all(checks.get(k, {"ok": False})["ok"] for k in self.PRIMARY.get(task, ()))
        success = bool(passed) and all(passed)
        partial = getattr(self, "_partial_" + name, None)
        if partial is not None:
            p = partial(checks)
            score = 1.0 if success else float(p)
        else:
            score = float(np.mean(passed)) if passed and primary else 0.0
        from .task_set import default_contract
        return dict(task=task, success=success, score=score, primary_outcome=primary, checks=checks, info=self._info,
                    task_contract=default_contract(task))

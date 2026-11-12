"""Outcome of curve: one seam 10 mm inside the plies' edge, from the far end of the right edge around the rounded
corner and along the near edge to the left edge, through both plies at the same material points, 9-11 stitches an
inch, with no gaps, no threads strung across the work and every join inside the seam sewn twice; the needle intact, no
thread broken, the threads trimmed and the work let go.

verify(e) evaluates an episode: e is its evaluator, task.Evaluator (the curve is graded under its own contract,
sewing-curve-v1); it returns e.verdict's evaluation (task, success, score, primary_outcome, checks, info,
task_contract). checks(e) are the curve's checks: task.Evaluator._eval_curve on curve_seam, the stitches measured
against the line 10 mm inside the rounded edge. PRIMARY is the check without which the score is 0 (no stitch through
both plies), and partial(e, checks) the score short of success."""
import numpy as np

from sewing.task import (CURVE_DAMAGE, CURVE_FAULT, CURVE_PARTIAL_MAX, CURVE_SOFT, CURVE_SOFT_REG, END_REACH, ENDS_MEET,
                         INCH, JOIN_SECURE, NOT_A_SKIP, SAME_PLACE, SPI_BAND, STEP_MAX, STITCH_BOUNDS, THREAD_LINK_MAX,
                         TOL_LINE, _polyline_distance)

CURVE_SEAM_SHARE = 0.90      # the seam's share; 0.05 more once trimmed and 0.05 more for letting go once trimmed
PRIMARY = ("stitches_formed",)


def verify(e):
    return e.verdict(checks, primary=PRIMARY, partial=partial)


def checks(e):
    return e._eval_curve(curve_seam)


def curve_seam(ev):
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
    from sewing.mesh import seam_edge_path
    w = ev.env.world
    W, L, R, A = w.config.ply_width, w.config.ply_length, w.config.corner_radius, w.config.seam_allowance
    edge = seam_edge_path(W, L, R, n_arc=96)
    line = seam_edge_path(W, L, R, A, n_arc=96)
    st = ev.stitches()
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


def partial(e, checks):
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
    g, M = e._curve, e.env.world.machine
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
    need = (g["ell"] - 2*END_REACH)/e.env.world.config.stitch_length
    rework = max(0.0, 1.0 - strays/need)
    damage = min(CURVE_DAMAGE.get(M.needle_state, 1.0), CURVE_FAULT if broken else 1.0)
    trimmed = M.tether == "trimmed"
    let_go = trimmed and checks["released"]["ok"]
    score = CURVE_PARTIAL_MAX*damage*rework*coverage*(CURVE_SEAM_SHARE + 0.05*trimmed + 0.05*let_go)
    e._info["partial_score"] = dict(value=round(score, 4), coverage=round(coverage, 4), strays=round(strays, 3),
                                    rework_factor=round(rework, 4), damage=damage, trimmed=trimmed,
                                    let_go=let_go, path_mm=round(g["ell"]*1e3, 1),
                                    note="the score short of success (1.0 on success)")
    return score

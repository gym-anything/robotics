"""Outcome of set-post: the hole backfilled to grade from the tarp, the lawn round it not dug down, and the post, with
the robot and its tools off it, standing plumb within 1 deg and moving at most 12.7 mm at grade under a person's lean
once the braces come off (post_load_test, on a copy of the state).

verify(e, lg, final=True) reads the episode for the evaluator e from its particle ledger lg (task.Ledger) and
returns the judgement, (criteria, progress parts, extra) as task.Evaluator.judge takes it; final=False is the advisory
reading (placement only, no load test). Every contract that selects the task keeps this judgement: soil-set-post-v1
as it is, soil-tasks-v2 through SetPostEvaluator, which rebases only its failed-run score."""
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from soil_excavation import layout as L, materials as M, params as P
from soil_excavation.catalogue import POST_CONTRACT
from soil_excavation.catalogue_v2 import CONTRACT
from soil_excavation.task import Evaluator as LegacyEvaluator, Ledger
from soil_excavation.task import ORIGIN_BLOCK, ORIGIN_PILE, VerdictUnavailable, _finite, _pose, _tilt_deg, crit, diag
from soil_excavation.task_set_v2 import SCORE_RULE, History, failed_score, idle_margin, net_progress

# set-post's success does not depend on protective stops (task.Evaluator reports them as a diagnostic): a recovered
# stop during the work does not change whether the post is set, and the instruction does not ask for it
STOPS_REPORTED_ONLY = True


def verify(e, lg, final=True):
    return _set_post(e, lg, final)


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
def _post_site(e):
    """Fixed geometry of the pre-dug post hole, cached: its lip outline at grade from the hand-dug wall the hole was
    cut with (layout.dug_wall), and the cavity volume from the gravel top to grade around the post (nominal pose)."""
    if getattr(e, "_site", None) is None:
        sp = e.env.spec
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
        e._site = dict(cut=cut, centre=np.array([mx, my], float), th=th, lip=lip,
                       cavity_m3=float(cav.sum())*0.005**3, gravel_top=z_bot)
    return e._site


def _lip_at(site, x, y):
    """Radius from the hole centre and the lip radius in that direction."""
    dx, dy = np.asarray(x, float) - site["centre"][0], np.asarray(y, float) - site["centre"][1]
    return np.hypot(dx, dy), np.interp(np.arctan2(dy, dx), site["th"], site["lip"])


def _post_surface(e, lg, site, p, R):
    """Column tops (1 cm) over the soil block: the dug footprint without the post, and the yard beyond the lip."""
    F = P.POST_FINISH
    sp = e.env.spec
    b = sp["block"]
    rect = ((b["lo"][0], b["lo"][1]), (b["hi"][0], b["hi"][1]))
    top, (x0, y0, cell) = lg.heightfield(rect)
    nx, ny = top.shape
    X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5)*cell, y0 + (np.arange(ny) + 0.5)*cell, indexing="ij")
    s = np.where(np.isnan(top), -0.6, top - L.GRADE)          # an empty column inside the block: dug to the floor
    e._last_top = (np.where(np.isnan(top), L.GRADE - 0.6, top), (x0, y0, cell))   # for _post_free's "buried"
    r, lip = _lip_at(site, X, Y)
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


def _post_placement(e, lg, site):
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
    r0, lip0 = _lip_at(site, lg.pos0[blk, 0], lg.pos0[blk, 1])
    xb = lg.pos[blk]
    # yard soil from beyond the lip that is now in the hole (used as fill); lawn damage elsewhere is read by
    # surrounding_grade
    dug = (r0 > lip0 + F["lip_margin"]) & L.dug_wall(xb, site["cut"]) & (xb[:, 2] <= L.GRADE + F["crown"])
    v_dug = float(np.count_nonzero(dug))*h3
    v_cost = max(0.0, v_dug - F["lawn_allowance_m3"])
    frac = float(np.clip((v_in - v_cost)/(F["placement_full"]*site["cavity_m3"]), 0., 1.))
    return frac, dict(tarp_soil_in_hole_L=round(v_in*1e3, 2), yard_soil_dug_L=round(v_dug*1e3, 2),
                      cavity_L=round(site["cavity_m3"]*1e3, 2))


def _post_tools(e):
    """{newton body: tool name} of the episode's tools (held and kit), and the subset a finger pad touches now."""
    w, sp = e.env.world, e.env.spec
    tools = {}
    for t in list(sp.get("held", {}).values()) + list(sp.get("kit", [])):
        for n in (t, f"{t}_a", f"{t}_b") + (("auger", "drill_trigger") if t == "drill" else ()):
            if n in w.scene.bodies:
                tools[int(w.scene.bodies[n])] = t
    mj_held = w.held_bodies()
    held = {i for i in tools if w._mj_body.get(i) in mj_held}
    return tools, held


def _robot_post_force(e):
    """Largest MuJoCo normal force between the post and any robot body (N)."""
    w = e.env.world
    post = w.scene.bodies["post"]
    robot = getattr(e, "_robot_set", None)
    if robot is None:
        robot = e._robot_set = {int(i) for i in w.scene.robot_bodies}
    f = 0.0
    for b1, b2, fn, _, _ in w.contact_table():
        if post in (b1, b2) and (b2 if b1 == post else b1) in robot:
            f = max(f, fn)
    return f


def _post_free(e, site):
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
    w = e.env.world
    post = w.scene.bodies["post"]
    tools, held = _post_tools(e)
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
    blk = e.env.spec["block"]
    top = getattr(e, "_last_top", None)
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
        r, lip = _lip_at(site, pts[:, 0], pts[:, 1])
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


def post_push_impulse(t):
    """Load impulse (N s) of the test push up to time t: ramp over ramp_s, then held."""
    T = P.POST_TEST
    F, r = T["force"], T["ramp_s"]
    return F*t*t/(2*r) if t <= r else F*r/2 + F*(t - r)


def post_support_credit(cross_t):
    """Support part of the progress for one push direction: the share of the notional lean's load impulse (the
    push of ramp_s + hold_s, continued for creep_s) carried before the post axis at grade reached the limit; 1 if
    it never did (measured or projected). Continuous and monotone in the time the post holds; ~0 for a post that
    runs away at once."""
    T = P.POST_TEST
    horizon = T["ramp_s"] + T["hold_s"] + T["creep_s"]
    if cross_t is None:
        return 1.0
    return float(np.clip(post_push_impulse(min(cross_t, horizon))/post_push_impulse(horizon), 0., 1.))


def post_plumb_credit(tilt):
    """Plumb factor of the progress: 1 within plumb_deg, falling linearly to 0 at plumb_zero_deg."""
    T = P.POST_TEST
    if tilt is None or not np.isfinite(tilt):
        return 0.0
    return float(np.clip((T["plumb_zero_deg"] - tilt)/(T["plumb_zero_deg"] - T["plumb_deg"]), 0., 1.))


def _set_post(e, lg, final):
    F, T = P.POST_FINISH, P.POST_TEST
    site = _post_site(e)
    p, R = _pose(e.env, "post")
    surf = _post_surface(e, lg, site, p, R)
    A, place = _post_placement(e, lg, site)
    free = _post_free(e, site)
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
    e.last_load_test = None
    S = U = 0.0
    if free["measurable"]:
        res = e.last_load_test = post_load_test(e)
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
                 post_support_credit(res["cross_t"][k])) for k in pushed) if len(pushed) == 4 else 0.0
        U = post_plumb_credit(tilt)
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


def post_load_test(e):
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
    env, w = e.env, e.env.world
    T, Fn = P.POST_TEST, P.POST_FINISH
    DT = 0.0005
    b = w.scene.bodies["post"]
    com = np.asarray(w.model.body_com.numpy()[b], float)
    t_wall = time.perf_counter()
    site = _post_site(e)
    free = _post_free(e, site)
    tools, _ = _post_tools(e)
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
                    f_robot = _robot_post_force(e)
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
                    f_robot = _robot_post_force(e)
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
                                f_robot = _robot_post_force(e)
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


# soil-tasks-v2's evaluator for the task (task_set_v2.install: V2_EVALUATOR): this verdict, its failed-run score
# rebased on the settled start; judge() is task.Evaluator's hook that reads the episode
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
        return 0.5*float(sum(_set_post(self, lg, final)[1]))

    def judge(self, lg, final):
        c, parts, extra = _set_post(self, lg, final)
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


V2_EVALUATOR = SetPostEvaluator

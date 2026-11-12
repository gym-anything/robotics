"""Outcome of pad-finish: the struck pad finished: level, the stone down, every side's edge tooled, a
smooth, hard steel-trowel finish over the whole top, and no blister, delamination risk, burn or water
worked in; the forms undisturbed, the tools released on a cart shelf and the robot off the pad.

verify(ev) is called by task.Evaluator.evaluate with the episode's evaluator and returns its evaluation (a
dict, reward_version pad-finish/progress-v3: score 1 on success, else 0.95 x the progress from the seed's own
start; no other contract applies to this task). prepare(ev), called when the evaluator is built, reads that
start. LIMIT_S is the advisory simulated-time limit reported with the evaluation.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from concrete_finishing import layout as L, params as P
from concrete_finishing.task import E, FAIL_TERM, DELAM_Q, LIP_W, blister_lift, _c

LIMIT_S = 5*3600             # s: three times an expert's robot time plus the waits the material needs (task.LIMITS_S)

# Authored tolerances:
TOP_D = P.TOOLS["edger"]["radius"] + 0.002   # m: the top outside the arris (the radius test lets the arc end one 2 mm cell late)
EDGER_W = P.TOOLS["edger"]["blade"][1] + 0.0015   # m: the edger's 76 mm flat plus its 1.5 mm lip on the form: its track counts as steel-finished by it
EDGER_TEX = 6.0e-5           # m: texture an edger-finished track may keep (tools.edger's own 0.05 mm, approached from above, +20 %)
SPOT_MM = 75.0               # mm: widest unfinished or stony spot (inscribed disc): narrower than any blade's missed stroke (edger 76, float 79 mm)
PATCH_MM2 = 4400.0           # mm2: largest connected unfinished or stony patch: the area of the 75 mm spot already allowed
CLEAR_MISS = 0.5             # the spot/patch caps read cells clearly short: steel-trowel credit below half
STONY_E = 0.05               # ... and aggregate exposure above 5 %, over twice the 2 % cell limit
SIDE_SHARE = 0.80            # each side: share of its lines tooled (executed edging left 0.83-0.96 per side, misses <= 48 mm)
SHORT_MISS_MM = 50.0         # mm: an edge miss up to this long is a scattered miss (a third of the edger's 152 mm blade) ...
LONG_MISS_MM = 100.0         # mm: ... longer misses may add up to at most this per side: less than one missed edger stroke
ROBOT_CLEAR = 0.06           # m: no robot link origin this close above the pad or its forms
FINGER_CLEAR = 0.01          # m: no fingertip pad face (55 mm out along its finger, scene.PAD_TIP_Z) this close above them
CONTACT_N = 0.05             # N: a fingertip pad bearing on the pad (Coupler/env.WAIT_CONTACT_N tolerance)
REWARD_VERSION = "pad-finish/progress-v3"

# The whole top outside the arris is the finished surface; the radius is cut relative to the surface the edger
# rode on; a share criterion also fails on one concentrated miss; trapped water is judged as committed by
# the end of bleeding, not as accumulated so far; tools may lie on any shelf of the cart; "arms clear" is
# replaced by the robot being left off the pad and forms.
FINISH_WORK = ("flat_plane", "flat_straightedge", "stone_down", "radius_profile", "lip_passed", "smooth_hard")


def _spot_mm(fail, cell):
    """Diameter (mm) of the widest disc of failing cells (cells outside the measured area count as not failing)."""
    return float(2.0*distance_transform_edt(fail).max()*cell*1e3) if fail.any() else 0.0


def _patch_mm2(fail, cell):
    """Area (mm2) of the largest 8-connected patch of failing cells."""
    if not fail.any():
        return 0.0
    from scipy.ndimage import label
    lab, n = label(fail, structure=np.ones((3, 3), bool))
    return float(np.bincount(lab.ravel())[1:].max()*(cell*1e3)**2)


def _rejected_deficit(miss, deficit, cell, n_top):
    """Progress only: the work left in the components the caps reject (8-connected, larger than PATCH_MM2 or wider
    than SPOT_MM), as the sum of each cell's deficit over the number of top cells."""
    if not miss.any():
        return 0.0
    from scipy.ndimage import label, maximum, sum as nsum
    lab, n = label(miss, structure=np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel())[1:]*(cell*1e3)**2
    width = 2.0*np.asarray(maximum(distance_transform_edt(miss), lab, idx))*cell*1e3
    rej = idx[(area > PATCH_MM2) | (width > SPOT_MM)]
    if not len(rej):
        return 0.0
    return float(np.asarray(nsum(deficit, lab, rej)).sum())/max(n_top, 1)


def _miss_runs(ok):
    """Lengths, in lines, of the stretches of consecutive failing lines."""
    runs, cur = [], 0
    for v in ok:
        if v:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    return runs + ([cur] if cur else [])


def prepare(ev):
    w = ev.env.world
    if "pad" in w.regions:
        # progress is measured from this seed's own start (its struck surface, read once at reset)
        ev._D0 = w.regions["pad"].D.copy()
        ev._E0 = w.regions["pad"].E.copy()
        crit = _finish_criteria(ev, w.regions["pad"])
        ev.finish_start = {k: crit[k]["value"] for k in FINISH_WORK}
        ev.finish_start_m = dict(ev._finish_m)


def _top(ev, r):
    return r.paste_mask & (ev.pad_d >= TOP_D)


def _finish_edges(ev, r):
    """Per form side, per perimeter line (one per 2 mm cell column/row, as _perimeter), whether the line has the
    edger's 3/8 in arc within 1.5 mm, the arc's top taken from that line's own surface just past the radius
    (median at 11.5-21.5 mm in; lines within that of a corner take the nearest line's), and whether the lip ran."""
    X, Y, _ = ev._pad_xyz
    on, h, dmin = r.paste_mask, r.h, ev.pad_d
    rad, tol = P.TOOLS["edger"]["radius"], E["radius_tol_mm"]*1e-3
    sides = {"s": [], "n": [], "w": [], "e": []}
    for i in np.flatnonzero((X[:, 0] > L.PX0) & (X[:, 0] < L.PX1)):
        for side, d in (("s", Y[i] - L.PY0), ("n", L.PY1 - Y[i])):
            sel = on[i] & (d >= 0) & (d < 0.10)
            sides[side].append((h[i][sel], dmin[i][sel], r.compact[i][sel]))
    for j in np.flatnonzero((Y[0] > L.PY0) & (Y[0] < L.PY1)):
        for side, d in (("w", X[:, j] - L.PX0), ("e", L.PX1 - X[:, j])):
            sel = on[:, j] & (d >= 0) & (d < 0.10)
            sides[side].append((h[:, j][sel], dmin[:, j][sel], r.compact[:, j][sel]))
    out = {}
    for side, lines in sides.items():
        lines = [ln for ln in lines if len(ln[0])]
        tops = np.array([float(np.median(hl[(d >= rad + 0.002) & (d < rad + 0.012)]))
                         if ((d >= rad + 0.002) & (d < rad + 0.012)).any() else np.nan for hl, d, _ in lines])
        good = np.flatnonzero(np.isfinite(tops))
        if len(good):
            near = good[np.abs(np.arange(len(tops))[:, None] - good[None, :]).argmin(axis=1)]
            tops = tops[near]
        else:
            tops = np.full(len(lines), P.FORM[1])
        prof, lip = [], []
        for (hl, d, comp), top in zip(lines, tops):
            ok = False
            for shift in (0.0, 0.5*r.cell, r.cell):
                ds = d - shift
                inr = (ds >= LIP_W) & (ds < rad)
                arc = top - (rad - np.sqrt(np.maximum(rad*rad - (rad - ds)**2, 0.0)))
                if inr.any() and np.abs(hl[inr] - arc[inr]).max() <= tol:
                    ok = True
                    break
            prof.append(ok)
            lip.append(bool(comp[d < 0.0025].any()))
        out[side] = (np.array(prof, bool), np.array(lip, bool))
    return out


def _finish_criteria(ev, r):
    """pad-finish's criteria read from the pad alone (the work and its defects)."""
    on = r.paste_mask
    top = _top(ev, r)
    where = "; over the whole top outside the arris (from 11.5 mm in from the forms)"
    out = {}
    # the finish: smooth and hard, the edger's own steel finish counting in its 76 mm track
    tex = np.where(ev.pad_d < EDGER_W, EDGER_TEX, E["trowel_tex"])
    ok = (r.D >= E["dens_min"]) & (r.a <= tex)
    share = float(ok[top].mean())
    # progress and the caps: graded steel-trowel credit per cell, none at float texture (0.15 mm) or coarser
    credit = np.minimum(np.clip((E["float_tex"][0] - r.a)/(E["float_tex"][0] - tex), 0.0, 1.0),
                        np.clip((r.D - ev._D0)/np.maximum(E["dens_min"] - ev._D0, 1e-6), 0.0, 1.0))
    miss = top & ~ok & (credit < CLEAR_MISS)            # clearly unfinished: the caps ignore near-threshold seams
    spot, patch = _spot_mm(miss, r.cell), _patch_mm2(miss, r.cell)
    out["smooth_hard"] = _c(share, share >= E["tex_share"] and spot <= SPOT_MM and patch <= PATCH_MM2, E["tex_share"],
                            "share of the top with densification >= 0.8 and texture <= 0.05 mm (<= 0.06 mm in the "
                            "edger's 77.7 mm track, its own steel finish), and no clearly unfinished patch (steel-trowel "
                            "credit below half) wider than 75 mm or larger than 4400 mm2 (CIP 14 'smooth', dense surface)" + where)
    out["smooth_hard"].update(largest_spot_mm=round(spot, 1), spot_limit_mm=SPOT_MM, largest_patch_mm2=round(patch, 1),
                              patch_limit_mm2=PATCH_MM2)
    stone = r.E < E["agg_max"]
    share = float(stone[top].mean())
    stony = top & (r.E > STONY_E)
    spot, patch = _spot_mm(stony, r.cell), _patch_mm2(stony, r.cell)
    out["stone_down"] = _c(share, share >= E["agg_share"] and spot <= SPOT_MM and patch <= PATCH_MM2, E["agg_share"],
                           "share of the top with coarse aggregate exposure below 2 %, and no clearly stony patch "
                           "(exposure above 5 %) wider than 75 mm or larger than 4400 mm2 (CIP 14 'embeds large aggregate')" + where)
    out["stone_down"].update(largest_spot_mm=round(spot, 1), spot_limit_mm=SPOT_MM, largest_patch_mm2=round(patch, 1),
                             patch_limit_mm2=PATCH_MM2)
    out.update(ev._flatness(r, top, where))
    gaps = ev._placement_gaps
    # the tooled edge, all of it: the perimeter's share, and on every side most of its length with no long misses
    edges = _finish_edges(ev, r)
    mm = r.cell*1e3
    for name, k, basis in (("radius_profile", 0, "share of the perimeter whose profile lies within 1.5 mm of the 3/8 "
                            "in arc of the edger 136D, the arc cut from the surface beside it"),
                           ("lip_passed", 1, "share of the perimeter where the edger's lip separated the concrete "
                            "from the form")):
        flags = np.concatenate([edges[s][k] for s in "snwe"])
        share = float(flags.mean()) if len(flags) else 0.0
        side = {s: float(edges[s][k].mean()) for s in "snwe"}
        # misses up to 50 mm are scattered imprecision only where the lip ran: lines without the lip count at any
        # length
        long = {s: (sum(n for n in _miss_runs(edges[s][k]) if n*mm > SHORT_MISS_MM) if k == 0 else
                    int((~edges[s][1]).sum()))*mm for s in "snwe"}
        out[name] = _c(share, share >= 0.90 and min(side.values()) >= SIDE_SHARE and max(long.values()) <= LONG_MISS_MM,
                       0.90, basis + ("; every side at least 80 % of its length, its misses longer than 50 mm adding "
                                     "up to at most 100 mm" if k == 0 else "; every side at least 80 % of its length, "
                                     "its lines without the mark adding up to at most 100 mm"))
        out[name].update(side_share={s: round(v, 4) for s, v in side.items()}, side_share_min=SIDE_SHARE,
                         side_long_miss_mm={s: round(v, 1) for s, v in long.items()}, long_miss_limit_mm=LONG_MISS_MM)
    n_top = int(top.sum())
    stone_credit = np.clip((ev._E0 - r.E)/np.maximum(ev._E0 - E["agg_max"], 1e-6), 0.0, 1.0)
    ev._finish_m = dict(placements=float((gaps*1e3 <= E["gap_mm"]).mean()), credit=float(credit[top].mean()),
                          finish_left=_rejected_deficit(miss, 1.0 - credit, r.cell, n_top),
                          stone_left=_rejected_deficit(stony, 1.0 - stone_credit, r.cell, n_top),
                          side_radius={s: float(edges[s][0].mean()) for s in "snwe"},
                          side_arc_lip={s: float((edges[s][0] & edges[s][1]).mean()) for s in "snwe"})
    # defects: blisters and delamination as the trapped water will stand when bleeding ends
    qf = ev._committed_water(r)
    blister = (blister_lift(qf, r.bl, r.cell) > 3e-4) & on
    out["no_blister"] = _c(int(blister.sum()), not blister.any(), 0, "cells raised by a blister now or by the "
                           "water their sealed top will trap by the end of bleeding (CIP 13)", "cells")
    dl = float((qf[on] > DELAM_Q).mean())
    out["delamination_risk"] = _c(dl, dl < E["delam_max"], E["delam_max"], "share of the area whose sealed top "
                                  "traps more than 0.01 mm of bleed water by the end of bleeding (CIP 20)")
    out["delamination_risk"].update(trapped_now=round(float((r.q[on] > DELAM_Q).mean()), 5))
    burn = float(((r.late[on] > 0) & (r.D[on] >= 0.98)).mean())
    out["no_trowel_burn"] = _c(burn, burn < E["burn_max"], E["burn_max"],
                               "share troweled to densification >= 0.98 after initial set (CIP 14 'trowel burns')")
    n = int(r.wet_worked[on].sum())
    out["no_water_worked"] = _c(n, n == 0, 0, "cells worked while bleed water stood on them (CIP 14)", "cells")
    dw = float(r.dwc[on].max())
    out["surface_wc"] = _c(dw, dw < P.DWC_DUST, P.DWC_DUST,
                           "largest rise of the top 6 mm's water-cement ratio from water worked in (CIP 1)")
    # the defect masks progress reads (computed once, so progress follows the criteria exactly)
    ev._finish_m.update(
        water_free=1.0 - float(((qf > DELAM_Q) | blister)[on].mean()),
        surface_free=1.0 - float(((r.wet_worked > 0) | (r.dwc >= P.DWC_DUST) | ((r.late > 0) & (r.D >= 0.98)))[on].mean()))
    return out


def _finish(ev, r):
    """pad-finish: (criteria, progress), reward_version progress-v2. `progress` in [0, 1] says how far the pad is
    from this seed's start (0) to finished (1), from the verifier's own quantities mapped linearly from their start
    values to their thresholds (a failing criterion earns at most FAIL_TERM of its term): level (the worst point and
    the share of straightedge placements within 3 mm), stone down, the tooled edge per side (the arc, and the lip
    where the arc is) and the finish (smooth-hard share and graded steel-trowel credit per cell). Stone and finish
    are scaled by the share of the pad free of the verifier's defects; the mean of the four is scaled by 0.9 + 0.1 x
    the share of tools put back. It reads no tool sequence, no milestone and no time; waiting changes nothing."""
    out = _finish_criteria(ev, r)
    out["forms_undisturbed"] = ev._forms()
    off, n_tools = ev._tools_off(shelves=True)
    out["tools_back"] = _c(len(off), not off, 0, "tools not on a shelf of the cart" + (f": {', '.join(off)}" if off else ""))
    out["robot_off_pad"] = _robot_off_pad(ev)

    def lin(v, v0, thr, lower=False):
        if lower:                                                             # lower is better (mm)
            v0 = v0 if v0 > thr else 2.0*thr
            return float(np.clip((v0 - v)/(v0 - thr), 0.0, 1.0))
        v0 = v0 if v0 < thr else 0.0
        return float(np.clip((v - v0)/(thr - v0), 0.0, 1.0))

    def term(name):
        c = out[name]
        t = lin(float(c["value"]), float(ev.finish_start[name]), float(c["threshold"]), c["unit"] == "mm")
        return 1.0 if c["passed"] else FAIL_TERM*t
    m, m0 = ev._finish_m, ev.finish_start_m
    # level: the worst point (as acceptance) and how much of the pad is flat (straightedge placements within 3 mm)
    flat = lin(m["placements"], m0["placements"], 1.0)
    level = 0.25*term("flat_plane") + 0.25*term("flat_straightedge") + \
        0.5*(1.0 if out["flat_straightedge"]["passed"] else FAIL_TERM*flat)
    # edge, per side: the arc, and the lip only where the arc is (the lip's mark alone is not a tooled edge)
    if out["radius_profile"]["passed"] and out["lip_passed"]["passed"]:
        edge = 1.0
    else:
        edge = FAIL_TERM*float(np.mean([0.5*lin(m["side_radius"][s], m0["side_radius"][s], 0.90) +
                                        0.5*lin(m["side_arc_lip"][s], m0["side_radius"][s], 0.90) for s in "snwe"]))
    # finish: the smooth-hard share and graded steel-trowel credit (none at float texture or coarser)
    sh, sd = out["smooth_hard"], out["stone_down"]
    # finish and stone: when failing, also no more than the work left in the patches the caps reject
    finish = 1.0 if sh["passed"] else \
        FAIL_TERM*min(0.5*lin(float(sh["value"]), float(ev.finish_start["smooth_hard"]), E["tex_share"]) +
                      0.5*lin(m["credit"], m0["credit"], E["tex_share"]), 1.0 - m["finish_left"])
    stone = 1.0 if sd["passed"] else \
        FAIL_TERM*min(lin(float(sd["value"]), float(ev.finish_start["stone_down"]), E["agg_share"]), 1.0 - m["stone_left"])
    terms = dict(level=level, stone=stone, edge=edge, finish=finish)
    sound, surface = m["water_free"], m["surface_free"]
    tidy = 0.9 + 0.1*(n_tools - len(off))/max(n_tools, 1)
    # trapped water lifts the troweled layer that carries the plane and the arc: it scales every term; water worked
    # in, w/c and burns take away surface only
    work = sound*(level + edge + surface*(stone + finish))/4.0
    prog = dict(value=round(work*tidy, 4), work=round(work, 4), **{k: round(v, 4) for k, v in terms.items()},
                defect_free=round(sound*surface, 5), water_free=round(sound, 5), surface_free=round(surface, 5), tools=round(tidy, 4),
                finish_left=round(m['finish_left'], 5), stone_left=round(m['stone_left'], 5), flat_placements=round(m["placements"], 4),
                finish_credit=round(m["credit"], 4), side_radius={s: round(v, 4) for s, v in m["side_radius"].items()},
                side_arc_lip={s: round(v, 4) for s, v in m["side_arc_lip"].items()},
                start=dict({k: ev.finish_start[k] for k in FINISH_WORK}, flat_placements=round(m0["placements"], 4),
                           finish_credit=round(m0["credit"], 4), side_radius={s: round(v, 4) for s, v in m0["side_radius"].items()}))
    return out, prog


def _robot_off_pad(ev):
    """Nothing of the robot left in or resting on the fresh pad or its forms: no arm link or hand origin over the pad
    and its forms lower than ROBOT_CLEAR above the form tops, no fingertip pad face (scene.PAD_TIP_Z out along its
    finger) lower than FINGER_CLEAR above them, and no fingertip pad bearing on the pad. A proxy from body poses: the
    paste meets only the nominated soles and fingertip pads, so a link inside the concrete leaves no mark."""
    w = ev.env.world
    q = w.state.body_q.numpy()
    fw = P.FORM[0]
    from concrete_finishing.scene import PAD_TIP_Z

    def over(p):
        return L.PX0 - fw < p[0] < L.PX1 + fw and L.PY0 - fw < p[1] < L.PY1 + fw
    low = []
    for n, i in w.scene.robot_body.items():
        if "finger" in n:
            # a finger by its fingertip pad's face, wherever the finger points
            p, R = w.body_pose(i)
            face = np.asarray(p, float) + np.asarray(R, float) @ np.array([0.0, 0.0, PAD_TIP_Z])
            if over(face) and face[2] < P.FORM[1] + FINGER_CLEAR:
                low.append(n)
        elif ("fr3_link" in n or "hand" in n) and over(q[i]) and q[i, 2] < P.FORM[1] + ROBOT_CLEAR:
            low.append(n)
    touch = [c.name for c in w.contactors if c.tool is None and c.region is not None and c.region.name == "pad"
             and float(w.plate_force.get(c.name, 0.0)) > CONTACT_N]
    bad = low + touch
    return _c(len(bad), not bad, 0, "robot links within 60 mm above the pad or its forms, fingertip faces within 10 "
                                    "mm, or fingertips bearing on "
                                    "the pad" + (f": {', '.join(bad)}" if bad else ""))


def verify(ev):
    w = ev.env.world
    t = ev.task
    crit, prog = _finish(ev, w.regions["pad"])
    passed = [c["passed"] for c in crit.values()]
    clock = ev.env.material_clock()
    success = all(passed)
    # score: 1 for a finished pad, else 0.95 x progress (the 0.95 a reporting convention: partial work stays
    # below success). `within_limit` is advisory, as for every task: the benchmark's own budget governs
    return dict(task=t, success=success, score=1.0 if success else round(0.95*prog["value"], 4), progress=prog,
                reward_version=REWARD_VERSION, criteria=crit,
                off_region_ml={n: round(float(r.off_m3)*1e6, 4) for n, r in w.regions.items()},
                material_clock_s=round(clock, 3), limit_s=ev.limit_s, within_limit=bool(clock <= ev.limit_s))

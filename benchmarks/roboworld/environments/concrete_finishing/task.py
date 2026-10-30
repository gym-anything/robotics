"""Task instructions and the evaluator.

Every criterion is read from the material's fields (heights, surface states, what each cell recorded when it was last
worked: its yield stress then, whether after initial set, whether under bleed water), the start state and the world's
contact history; never from tool poses along the way or from the clock, except through what the material recorded.
Values are authored with their basis in params.EVAL. `success` needs every criterion; `score` is the
share passed, except pad-finish's (Evaluator._finish): 1 on success, else
0.95 x a continuous progress from the seed's own start to the acceptance thresholds.
"""

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from . import params as P
from . import layout as L

E = P.EVAL

INSTRUCTIONS = {
    "pad-darby": "The pad was just struck off. Darby it: take out the high and low spots and get the stone down before "
                 "the bleed water comes up. There is concrete in the tub for low spots. Tools back on the cart when you "
                 "are done.",
    "pad-edge": "Edge the pad. Tools back on the cart when you are done.",
    "pad-float": "Float the pad. Tools back on the cart when you are done.",
    "pad-trowel": "Steel-trowel the pad smooth and hard. Tools back on the cart when you are done.",
    "pad-finish": "The pad was just struck off. Finish it with tooled edges and a smooth, hard steel-trowel finish. Tools "
                  "back on the cart when you are done.",
    "patch-flush": "Finish the floor patch flush with the slab and give it a float finish.",
    "tool-joints": "Tool the joints of the new block concave. Fill anything short first.",
    "tool-sealant": "Tool the sealant in the control joint between the tape marks, dry.",
    "patch-drywall": "Fill the anchor holes and knife the patch flush.",
}
VARIANT_TEXT = {"door": " The door to the corridor is open and there is a draught across the room.",
                "fan": " A pedestal fan is running and blowing across the work.",
                "hot": " It is a hot day: the concrete and the air are at about 27 C.",
                "cool": " It is a cool day: the concrete and the air are at about 15 C.",
                "fast-sealant": " The sealant is a fast-skinning grade."}


def instruction(selection, spec=None):
    """The task's text as a session serves it: the instruction, and what a crew would see of the site's variant.
    `selection`: a dict with task and variant (variants.resolve) or a Config."""
    task = selection["task"] if isinstance(selection, dict) else selection.task
    variant = selection.get("variant", "default") if isinstance(selection, dict) else selection.variant
    return INSTRUCTIONS[task] + VARIANT_TEXT.get(variant, "")


# simulated-time limits: three times an expert's robot time plus the waits the material needs
LIMITS_S = {"pad-darby": 25*60, "pad-edge": 110*60, "pad-float": 35*60, "pad-trowel": 90*60, "pad-finish": 5*3600,
            "patch-flush": 25*60, "tool-joints": 60*60, "tool-sealant": 30*60, "patch-drywall": 15*60}
SMEAR_T = 2.0e-4             # paste thinner than 0.2 mm on a finished face is not counted as smear
FIELD_D = 0.150              # m: the field, past the edger's 76 mm and the overflow ridge beside it
LIP_W = 0.0015               # m: the edger's lip, 1.5 mm thick (tools.edger; its sole's lip_w)
OFF_TOL = 2.0e-8             # m3: paste laid past a region's modelled edge counts once it exceeds 1 cm2 at SMEAR_T
DELAM_Q = 1.0e-5             # trapped bleed water above 0.01 mm marks a cell sealed while it bled
COMPACT_MIN = 0.5*P.MORTAR_COMPACT   # half the normalized surface fabric; authored outcome definition
# pad-finish only. Authored tolerances:
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
FAIL_TERM = 0.9              # a progress term whose criterion fails earns at most this share of it
REWARD_VERSION = "pad-finish/progress-v3"


def remaining_bleed(material, age):
    """m of bleed water still to rise at each `age` (s) until bleeding ends: the integral of materials.Concrete.
    bleed_rate (a triangle of mean bleed_mean over [0, bleed_end], peak at a third of it) in closed form."""
    a = np.asarray(age, float)
    if not getattr(material, "bleeds", False):
        return np.zeros_like(a)
    te = float(material.bleed_end)
    peak = 2.0*material.bleed_mean/3600.0
    a = np.clip(a, 0.0, te)
    return np.where(a < te/3.0, 0.5*peak*te - 1.5*peak*a*a/te, 0.75*peak*(te - a)**2/te)/1000.0


def blister_lift(q, bl, cell):
    """The blister paste.Region._blisters raises from trapped water q (it only grows, so the final q gives the final
    lift): 4 x the water above BLISTER_Q, box-blurred over +-7.5 mm, never lower than the present blister, at most 3 mm."""
    lift = np.where(q > P.BLISTER_Q, (q - P.BLISTER_Q)*4.0, 0.0)
    k = max(1, int(round(0.0075/cell)))
    for ax in (0, 1):
        acc = np.zeros_like(lift)
        for s in range(-k, k + 1):
            acc += np.roll(lift, s, axis=ax)
        lift = acc/(2*k + 1)
    return np.minimum(np.maximum(bl, lift), 0.003)


def _longest_false(ok):
    best = cur = 0
    for v in ok:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


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


def _c(value, ok, threshold, basis, unit=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 5)
    return dict(value=value, threshold=threshold, unit=unit, passed=bool(ok), basis=basis)


def _with_off(c, r, where):
    """Adjacent sparse catch deposits remain visible and collectible. This is the
    current deposited volume, not an irreversible lifetime spill counter."""
    off = float(getattr(r, "off_m3", 0.0))
    c["off_region_ml"] = round(off*1e6, 4)
    if off > OFF_TOL:
        c["passed"] = False
        c["basis"] += f"; {off*1e6:.2f} ml remains {where} on spatial catch surfaces"
    return c


def hull_gap(z, step):
    """Largest gap (m) under a straightedge of length EVAL straightedge_m laid on the profile z (sampled every `step` m):
    within each window the straightedge rests on the profile's upper hull."""
    n = len(z)
    if n < 3:
        return 0.0
    win = max(3, int(round(E["straightedge_m"]/step)))
    best = 0.0
    starts = [0] if n <= win else list(range(0, n - win + 1, max(1, win//4))) + [n - win]
    for s0 in starts:
        seg = np.asarray(z[s0:s0 + win], float)
        x = np.arange(len(seg))*step
        hull = []
        for k in range(len(seg)):
            while len(hull) >= 2:
                (x1, y1), (x2, y2) = hull[-2], hull[-1]
                if (x2 - x1)*(seg[k] - y1) - (y2 - y1)*(x[k] - x1) >= 0:
                    hull.pop()
                else:
                    break
            hull.append((x[k], seg[k]))
        hx, hy = np.array([p[0] for p in hull]), np.array([p[1] for p in hull])
        best = max(best, float((np.interp(x, hx, hy) - seg).max()))
    return best


class Evaluator:
    def __init__(self, env):
        self.env = env
        w = env.world
        self.task = env.config.task
        self.instruction = INSTRUCTIONS[self.task]
        self.limit_s = LIMITS_S[self.task]
        self.h0 = {n: r.h.copy() for n, r in w.regions.items()}
        for n, r in w.regions.items():
            X, Y, Z = (r.o[:, None, None] + r.U[:, None, None]*r.u[None, :, None] + r.V[:, None, None]*r.v[None, None, :])
            setattr(self, f"_{n}_xyz", (X, Y, Z))
        if "pad" in w.regions:
            X, Y, _ = self._pad_xyz
            self.pad_d = np.minimum.reduce([X - L.PX0, L.PX1 - X, Y - L.PY0, L.PY1 - Y])
            if self.task == "pad-finish":
                # progress is measured from this seed's own start (its struck surface, read once at reset)
                self._D0 = w.regions["pad"].D.copy()
                self._E0 = w.regions["pad"].E.copy()
                crit = self._finish_criteria(w.regions["pad"])
                self.finish_start = {k: crit[k]["value"] for k in self.FINISH_WORK}
                self.finish_start_m = dict(self._finish_m)
        if "infill" in w.regions:
            r = w.regions["infill"]
            self.joint_e = distance_transform_edt(r.paste_mask)*r.cell          # distance in from the block faces
        if "joint" in w.regions:
            r = w.regions["joint"]
            X, Y, Z = self._joint_xyz
            self.sj_seg = r.paste_mask & (Z >= P.CJ_SEGMENT[0]) & (Z <= P.CJ_SEGMENT[1])
            self.sj_across = Y - P.CJ_Y
        if "wallboard" in w.regions:
            r = w.regions["wallboard"]
            X, Y, Z = self._wallboard_xyz
            near = np.hypot(X - P.DW_PATCH[0], Z - P.DW_PATCH[1]) < 0.15
            for hx, hz in P.DW_HOLES:
                near |= np.hypot(X - hx, Z - hz) < 0.15
            self.dw_near = near

    # ------------------------------------------------------------------ the pad
    def _core(self, r):
        return r.paste_mask & (self.pad_d > E["band_m"])

    def _flatness(self, r, area=None, where=""):
        core = self._core(r) if area is None else area
        h = r.h
        idx = np.argwhere(core)
        i0, i1, j0, j1 = idx[:, 0].min(), idx[:, 0].max(), idx[:, 1].min(), idx[:, 1].max()
        every = max(1, int(round(0.025/r.cell)))
        gaps = [hull_gap(h[i0:i1 + 1, j], r.cell) for j in range(j0, j1 + 1, every)]
        gaps += [hull_gap(h[i, j0:j1 + 1], r.cell) for i in range(i0, i1 + 1, every)]
        ii = np.arange(i0, i1 + 1)
        for k in range(-(i1 - i0), j1 - j0 + 1, every):
            for jj in (j0 + k + (ii - i0), j1 - k - (ii - i0)):
                ok = (jj >= j0) & (jj <= j1)
                if ok.sum() > 10:
                    gaps.append(hull_gap(h[ii[ok], jj[ok]], r.cell*math.sqrt(2)))
        gap = max(gaps)
        self._placement_gaps = np.asarray(gaps)               # every straightedge placement's gap (pad-finish progress)
        plane = float(np.abs(h[core] - P.FORM[1]).max())
        return {"flat_straightedge": _c(gap*1e3, gap*1e3 <= E["gap_mm"], E["gap_mm"],
                                        "largest gap under a 1.2 m straightedge along x, y and both diagonals every 25 mm "
                                        "(ACI 117 'flat' 1/4 in in 10 ft; authored)" + where, "mm"),
                "flat_plane": _c(plane*1e3, plane*1e3 <= E["plane_mm"], E["plane_mm"],
                                 "farthest point from the plane of the form tops, the crew's reference (authored)" + where, "mm")}

    def _perimeter(self, r):
        """Profiles across the edge: one per cell column along the long forms and per cell row along the short ones,
        each (heights, distance in from the form, compaction) over the band within 0.10 m of its form."""
        X, Y, _ = self._pad_xyz
        on = r.paste_mask
        h = r.h
        h0 = self.h0["pad"]
        lines = []
        xs = np.flatnonzero((X[:, 0] > L.PX0) & (X[:, 0] < L.PX1))
        ys = np.flatnonzero((Y[0] > L.PY0) & (Y[0] < L.PY1))
        dmin = self.pad_d                                    # the arc is about the nearest form (corners round both ways)
        for i in xs:
            for side, d in (("s", Y[i] - L.PY0), ("n", L.PY1 - Y[i])):
                sel = on[i] & (d >= 0) & (d < 0.10)
                lines.append((h[i][sel], dmin[i][sel], r.compact[i][sel], r.wet_worked[i][sel], h0[i][sel]))
        for j in ys:
            for side, d in (("w", X[:, j] - L.PX0), ("e", L.PX1 - X[:, j])):
                sel = on[:, j] & (d >= 0) & (d < 0.10)
                lines.append((h[:, j][sel], dmin[:, j][sel], r.compact[:, j][sel], r.wet_worked[:, j][sel], h0[:, j][sel]))
        return lines

    def _edges(self, r, share):
        rad = P.TOOLS["edger"]["radius"]
        top = P.FORM[1]
        prof, lip, gouge, step_ok, n, n_track = 0, 0, 0.0, 0, 0, 0
        for h, d, comp, wet, h0 in self._perimeter(r):
            if not len(h):
                continue
            n += 1
            arc = np.where(d < rad, top - (rad - np.sqrt(np.maximum(rad*rad - (rad - d)**2, 0.0))), top)
            # the arc from the lip's slot to the radius. The slot the lip cuts down the form (1.5 mm) is the separation
            # the lip criterion reads, not part of the arc (counted, it failed every edger's radius); on the pad's 2 mm
            # grid it takes one or two cells (an edger turned 0.7 degrees on the form cut it over two), so the arc may
            # begin up to one cell further from the form: the profile passes if it matches the arc so shifted
            ok = False
            for shift in (0.0, 0.5*r.cell, r.cell):
                ds = d - shift
                arc_s = np.where((ds >= 0) & (ds < rad), top - (rad - np.sqrt(np.maximum(rad*rad - (rad - ds)**2, 0.0))), top)
                inr = (ds >= LIP_W) & (ds < rad)
                if inr.any() and np.abs(h[inr] - arc_s[inr]).max() <= E["radius_tol_mm"]*1e-3:
                    ok = True
                    break
            if ok:
                prof += 1
            if comp[d < 0.0025].any():
                lip += 1
            # a gouge: the surface more than gouge_mm below both the edge's arc and where it stood at the start, in the
            # 80 mm edge band past the lip's slot (the slot, up to one cell wider than the lip on this grid, is the
            # lip's work, not a hole)
            band = (d >= LIP_W + r.cell) & (d < 0.080)
            # measured against the arc begun one cell further from the form (as the radius allows): with the slot over
            # two cells the next cell is the arc's start, not a hole (A2: 2.2 mm "gouges" there at every corner)
            ds = d - r.cell
            arc1 = np.where((ds >= 0) & (ds < rad), top - (rad - np.sqrt(np.maximum(rad*rad - (rad - ds)**2, 0.0))),
                            np.where(ds < 0, top - rad, top))
            ref = np.minimum(np.minimum(arc, arc1), h0)
            gouge = max(gouge, float(np.clip(ref[band] - h[band], 0.0, None).max()) if band.any() else 0.0)
            # the track's inner edge, where a line reaches 92 mm from the form (lines within 92 mm of a corner do not:
            # counted in the share, they held a perfectly floated pad to 85 %)
            a, b = (d >= 0.060) & (d < 0.074), (d >= 0.078) & (d < 0.092)
            if a.any() and b.any():
                n_track += 1
                if abs(float(h[a].mean() - h[b].mean())) <= E["track_step_mm"]*1e-3:
                    step_ok += 1
        n = max(n, 1)
        return prof/n, lip/n, gouge, step_ok/max(n_track, 1)

    def _pad_edges(self, r, share, with_field=False, with_tracks=False):
        prof, lip, gouge, steps = self._edges(r, share)
        out = {"radius_profile": _c(prof, prof >= share, share, "share of the perimeter whose profile lies within 1.5 mm of "
                                                              "the 3/8 in arc of the edger 136D"),
               "lip_passed": _c(lip, lip >= share, share, "share of the perimeter where the edger's lip separated the "
                                                         "concrete from the form"),
               "no_gouge": _c(gouge*1e3, gouge*1e3 <= E["gouge_mm"], E["gouge_mm"], "deepest hole cut in the 80 mm edge band "
                                                                               "below both the arc and the start surface", "mm")}
        if with_tracks:
            out["tracks_floated_out"] = _c(steps, steps >= E["tex_share"], E["tex_share"],
                                           "share of the perimeter with no step above 1 mm at the edger track's inner edge")
        if with_field:
            # the field beyond the edger's track: the edger is 76 mm wide and leaves its overflow beside its inner edge
            # (A2: the surface changed between 80 and 120 mm from the forms and nowhere beyond)
            far = r.paste_mask & (self.pad_d > FIELD_D)
            ch = float(np.abs(r.h[far] - self.h0["pad"][far]).max())
            edge = r.paste_mask & (self.pad_d < 0.080)
            out["field_untouched"] = _c(ch*1e3, ch*1e3 < E["field_change_mm"], E["field_change_mm"],
                                        "largest change of the surface more than 150 mm from the forms (past the edger's track)", "mm")
            out["edge_not_worked_wet"] = _c(int(r.wet_worked[edge].sum()), not r.wet_worked[edge].any(), 0,
                                            "edge cells worked under bleed water (CIP 14)", "cells")
        return out

    def _pad(self, r, t):
        core = self._core(r)
        on = r.paste_mask
        out = {}
        if t in ("pad-darby",):
            out.update(self._flatness(r))
            share = float((r.E[core] < E["agg_max"]).mean())
            out["stone_down"] = _c(share, share >= E["agg_share"], E["agg_share"],
                                   "share of the area with coarse aggregate exposure below 2 % (CIP 14 'embeds large aggregate')")
            out["not_sealed"] = _c(float(r.S[on].max()), float(r.S[on].max()) < E["seal_max"], E["seal_max"],
                                   "largest sealing: a darby leaves the surface open (CIP 14)")
        if t == "pad-edge":
            out.update(self._pad_edges(r, E["edge_share"], with_field=True))
        if t == "pad-float":
            share = float(((r.a[core] >= E["float_tex"][0]) & (r.a[core] <= E["float_tex"][1]) & (r.E[core] < E["agg_max"])).mean())
            out["float_texture"] = _c(share, share >= E["tex_share"], E["tex_share"],
                                      "share of the area with a float texture of 0.15-0.6 mm and the stone down")
            out.update(self._flatness(r))
            e = self._pad_edges(r, 0.90, with_tracks=True)
            out["radius_profile"] = e["radius_profile"]
            out["tracks_floated_out"] = e["tracks_floated_out"]
        if t == "pad-trowel":
            share = float(((r.D[core] >= E["dens_min"]) & (r.a[core] <= E["trowel_tex"])).mean())
            out["smooth_hard"] = _c(share, share >= E["tex_share"], E["tex_share"],
                                    "share with densification >= 0.8 and texture <= 0.05 mm (CIP 14 'smooth', dense surface)")
            bl = int((r.bl[on] > 3e-4).sum())
            out["no_blister"] = _c(bl, bl == 0, 0, "cells raised by a blister (CIP 13)", "cells")
            dl = float((r.q[on] > DELAM_Q).mean())
            out["delamination_risk"] = _c(dl, dl < E["delam_max"], E["delam_max"],
                                          "share of the area sealed while bleed water still rose (CIP 20)")
            burn = float(((r.late[on] > 0) & (r.D[on] >= 0.98)).mean())
            out["no_trowel_burn"] = _c(burn, burn < E["burn_max"], E["burn_max"],
                                       "share troweled to densification >= 0.98 after initial set (CIP 14 'trowel burns')")
            out.update(self._flatness(r))
            e = self._pad_edges(r, 0.90)
            out["radius_profile"] = e["radius_profile"]
        if t in ("pad-darby", "pad-float", "pad-trowel"):
            n = int(r.wet_worked[on].sum())
            out["no_water_worked"] = _c(n, n == 0, 0, "cells worked while bleed water stood on them (CIP 14)", "cells")
        if t in ("pad-float", "pad-trowel"):
            dw = float(r.dwc[on].max())
            out["surface_wc"] = _c(dw, dw < P.DWC_DUST, P.DWC_DUST,
                                   "largest rise of the top 6 mm's water-cement ratio from water worked in (CIP 1)")
        out["forms_undisturbed"] = self._forms()
        out["tools_back"] = self._tools_back()
        out["arms_clear"] = self._arms_clear()
        return out

    # ------------------------------------------------------------------ pad-finish
    # The whole top outside the arris is the finished surface; the radius is cut relative to the surface the edger
    # rode on; a share criterion also fails on one concentrated miss; trapped water is judged as committed by
    # the end of bleeding, not as accumulated so far; tools may lie on any shelf of the cart; "arms clear" is
    # replaced by the robot being left off the pad and forms.
    FINISH_WORK = ("flat_plane", "flat_straightedge", "stone_down", "radius_profile", "lip_passed", "smooth_hard")

    def _top(self, r):
        return r.paste_mask & (self.pad_d >= TOP_D)

    def _committed_water(self, r):
        """Trapped water each cell will hold when bleeding ends: what it holds now plus its sealed share of the bleed
        water still to rise (paste.Region.slow_step: q grows by dt bleed_rate S / rho_w; sealing never decreases)."""
        return r.q + np.clip(r.S, 0.0, 1.0)*remaining_bleed(r.material, r.age())

    def _finish_edges(self, r):
        """Per form side, per perimeter line (one per 2 mm cell column/row, as _perimeter), whether the line has the
        edger's 3/8 in arc within 1.5 mm, the arc's top taken from that line's own surface just past the radius
        (median at 11.5-21.5 mm in; lines within that of a corner take the nearest line's), and whether the lip ran."""
        X, Y, _ = self._pad_xyz
        on, h, dmin = r.paste_mask, r.h, self.pad_d
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

    def _finish_criteria(self, r):
        """pad-finish's criteria read from the pad alone (the work and its defects)."""
        on = r.paste_mask
        top = self._top(r)
        where = "; over the whole top outside the arris (from 11.5 mm in from the forms)"
        out = {}
        # the finish: smooth and hard, the edger's own steel finish counting in its 76 mm track
        tex = np.where(self.pad_d < EDGER_W, EDGER_TEX, E["trowel_tex"])
        ok = (r.D >= E["dens_min"]) & (r.a <= tex)
        share = float(ok[top].mean())
        # progress and the caps: graded steel-trowel credit per cell, none at float texture (0.15 mm) or coarser
        credit = np.minimum(np.clip((E["float_tex"][0] - r.a)/(E["float_tex"][0] - tex), 0.0, 1.0),
                            np.clip((r.D - self._D0)/np.maximum(E["dens_min"] - self._D0, 1e-6), 0.0, 1.0))
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
        out.update(self._flatness(r, top, where))
        gaps = self._placement_gaps
        # the tooled edge, all of it: the perimeter's share, and on every side most of its length with no long misses
        edges = self._finish_edges(r)
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
        stone_credit = np.clip((self._E0 - r.E)/np.maximum(self._E0 - E["agg_max"], 1e-6), 0.0, 1.0)
        self._finish_m = dict(placements=float((gaps*1e3 <= E["gap_mm"]).mean()), credit=float(credit[top].mean()),
                              finish_left=_rejected_deficit(miss, 1.0 - credit, r.cell, n_top),
                              stone_left=_rejected_deficit(stony, 1.0 - stone_credit, r.cell, n_top),
                              side_radius={s: float(edges[s][0].mean()) for s in "snwe"},
                              side_arc_lip={s: float((edges[s][0] & edges[s][1]).mean()) for s in "snwe"})
        # defects: blisters and delamination as the trapped water will stand when bleeding ends
        qf = self._committed_water(r)
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
        self._finish_m.update(
            water_free=1.0 - float(((qf > DELAM_Q) | blister)[on].mean()),
            surface_free=1.0 - float(((r.wet_worked > 0) | (r.dwc >= P.DWC_DUST) | ((r.late > 0) & (r.D >= 0.98)))[on].mean()))
        return out

    def _finish(self, r):
        """pad-finish: (criteria, progress), reward_version progress-v2. `progress` in [0, 1] says how far the pad is
        from this seed's start (0) to finished (1), from the verifier's own quantities mapped linearly from their start
        values to their thresholds (a failing criterion earns at most FAIL_TERM of its term): level (the worst point and
        the share of straightedge placements within 3 mm), stone down, the tooled edge per side (the arc, and the lip
        where the arc is) and the finish (smooth-hard share and graded steel-trowel credit per cell). Stone and finish
        are scaled by the share of the pad free of the verifier's defects; the mean of the four is scaled by 0.9 + 0.1 x
        the share of tools put back. It reads no tool sequence, no milestone and no time; waiting changes nothing."""
        out = self._finish_criteria(r)
        out["forms_undisturbed"] = self._forms()
        off, n_tools = self._tools_off(shelves=True)
        out["tools_back"] = _c(len(off), not off, 0, "tools not on a shelf of the cart" + (f": {', '.join(off)}" if off else ""))
        out["robot_off_pad"] = self._robot_off_pad()

        def lin(v, v0, thr, lower=False):
            if lower:                                                             # lower is better (mm)
                v0 = v0 if v0 > thr else 2.0*thr
                return float(np.clip((v0 - v)/(v0 - thr), 0.0, 1.0))
            v0 = v0 if v0 < thr else 0.0
            return float(np.clip((v - v0)/(thr - v0), 0.0, 1.0))

        def term(name):
            c = out[name]
            t = lin(float(c["value"]), float(self.finish_start[name]), float(c["threshold"]), c["unit"] == "mm")
            return 1.0 if c["passed"] else FAIL_TERM*t
        m, m0 = self._finish_m, self.finish_start_m
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
            FAIL_TERM*min(0.5*lin(float(sh["value"]), float(self.finish_start["smooth_hard"]), E["tex_share"]) +
                          0.5*lin(m["credit"], m0["credit"], E["tex_share"]), 1.0 - m["finish_left"])
        stone = 1.0 if sd["passed"] else \
            FAIL_TERM*min(lin(float(sd["value"]), float(self.finish_start["stone_down"]), E["agg_share"]), 1.0 - m["stone_left"])
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
                    start=dict({k: self.finish_start[k] for k in self.FINISH_WORK}, flat_placements=round(m0["placements"], 4),
                               finish_credit=round(m0["credit"], 4), side_radius={s: round(v, 4) for s, v in m0["side_radius"].items()}))
        return out, prog

    # ------------------------------------------------------------------ the floor patch
    def _patch(self, r):
        rec = r.paste_mask
        h, t = r.h, r.t
        inner = rec & ~np.roll(rec, 1, 0) | rec & ~np.roll(rec, -1, 0) | rec & ~np.roll(rec, 1, 1) | rec & ~np.roll(rec, -1, 1)
        outer = ~rec & (np.roll(rec, 1, 0) | np.roll(rec, -1, 0) | np.roll(rec, 1, 1) | np.roll(rec, -1, 1))
        # the lip: at every inner perimeter cell, the step to the slab beside it
        lip = float(np.abs(h[inner]).max())
        mean = float(h[rec].mean())
        filled = float((h[inner] >= -5e-4).mean())
        dist = distance_transform_edt(~rec)*r.cell
        smear = int(((dist > 0.010) & (t > SMEAR_T) & ~rec).sum())
        worked = rec & (r.worked > 0)
        tex = float(((r.a[rec] >= E["float_tex"][0]) & (r.a[rec] <= E["float_tex"][1]) & worked[rec]).mean())
        late = float(r.tau_w[worked].max()) if worked.any() else None
        return {"no_lip": _c(lip*1e3, lip*1e3 <= E["lip_mm"], E["lip_mm"], "largest step between the patch and the slab "
                                                                       "at the saw cut (ADA 1/4 in is a hard limit)", "mm"),
                "no_hard_lip": _c(lip*1e3, lip*1e3 <= E["lip_hard_mm"], E["lip_hard_mm"], "ADA 1/4 in", "mm"),
                "flush_mean": _c(mean*1e3, abs(mean)*1e3 <= E["patch_mean_mm"], E["patch_mean_mm"],
                                 "mean height of the patch against the slab plane", "mm"),
                "against_the_cut": _c(filled, filled >= E["fill_share"], E["fill_share"],
                                      "share of the perimeter filled to within 0.5 mm of the slab top (Sika: force material "
                                      "against the edge)"),
                "no_feather_edge": _with_off(_c(smear*r.cell**2*1e4, smear == 0, 0.0, "mortar thicker than 0.2 mm on the slab "
                                                "beyond 10 mm of the cut (Sika: do not feather edge)", "cm2"), r,
                                             "on the slab more than 0.2 m from the cut"),
                "float_texture": _c(tex, tex >= E["tex_share"], E["tex_share"], "share of the patch floated to a 0.15-0.6 mm "
                                                                               "texture"),
                "finished_in_time": _c(None if late is None else late/1e3, late is not None and late <= 30e3, 30.0,
                                       "the stiffest the mortar was where it was last worked (Sika: finish within 30 min)", "kPa")}

    # ------------------------------------------------------------------ block joints
    def _joints(self, r):
        on = r.paste_mask
        e = self.joint_e
        centre = on & (e >= 0.75*P.JOINT_W/2)
        h = r.h
        depth = -h[centre]
        lo, hi = E["concave_mm"]
        conc = float(((depth >= lo*1e-3) & (depth <= hi*1e-3)).mean())
        comp = float((r.compact[centre] >= COMPACT_MIN).mean())
        tw = r.tau_w[centre]
        ripe = float(((tw >= E["mortar_window"][0]) & (tw <= E["mortar_window"][1])).mean())
        length = float(centre.sum())*r.cell/max(int(round(0.75*P.JOINT_W/r.cell)) - 1, 1)
        length = max(length, 1e-3)
        smear = float(((~on) & (r.t > SMEAR_T)).sum())*r.cell**2*1e4/length
        void = int((on & (h < -(hi*1e-3 + 1.5e-3))).sum())
        return {"concave": _c(conc, conc >= E["joint_share"], E["joint_share"], "share of the joint centres 1.4-2.8 mm deep "
                                                                               "(a 1/2 in concave jointer in a 3/8 in joint)"),
                "compacted": _c(comp, comp >= E["joint_share"], E["joint_share"], "share of joint centres with normalized surface consolidation >=0.5; "
                                                                                 "confined plastic-work model, inspired by BIA tooling practice, not measured porosity"),
                "tooled_thumbprint_hard": _c(ripe, ripe >= 0.90, 0.90, "share of the joint centres last tooled at 5-20 kPa "
                                                                       "(thumbprint hard, BIA TN 7B)"),
                "smear": _with_off(_c(smear, smear < E["smear_cm2_m"], E["smear_cm2_m"], "mortar thicker than 0.2 mm on the "
                                      "block faces per metre of joint", "cm2/m"), r, "on the wall beyond the infill"),
                "no_void": _c(void, void == 0, 0, "joint cells deeper than 4.3 mm below the face", "cells")}

    # ------------------------------------------------------------------ the sealant joint
    def _sealant(self, r):
        seg = self.sj_seg
        a = self.sj_across
        centre = seg & (np.abs(a) < 0.002)
        h, t = r.h, r.t
        lo, hi = E["sealant_centre_mm"]
        ok = (-h[centre] >= lo*1e-3) & (-h[centre] <= hi*1e-3) & (t[centre] >= E["sealant_min_mm"]*1e-3)
        conc = float(ok.mean())
        rows_l = (r.contact_l > 0) & seg & r.edge_l
        rows_r = (r.contact_r > 0) & seg & r.edge_r
        both = float(min((rows_l.sum(axis=0) > 0)[seg.any(axis=0)].mean(), (rows_r.sum(axis=0) > 0)[seg.any(axis=0)].mean()))
        void = int((seg & (t < 1e-3)).sum())
        X, Y, Z = self._joint_xyz
        inseg = (Z >= P.CJ_SEGMENT[0]) & (Z <= P.CJ_SEGMENT[1])
        smear = float((~r.paste_mask & inseg & (np.abs(a) > P.CJ_W/2 + 0.001) & (t > SMEAR_T)).sum())*r.cell**2*1e4 \
            / (P.CJ_SEGMENT[1] - P.CJ_SEGMENT[0])
        worked = seg & (r.worked > 0)
        work_age = np.maximum(r.seal_age, r.seal_t-r.t0)
        before = bool(worked.any()) and float(work_age[worked].max()) < r.material.skin and not r.skin_torn[seg].any()
        return {"concave_and_thick": _c(conc, conc >= 0.95, 0.95, "share of the centre line 0.5-2.5 mm below the wall faces "
                                                                   "with at least 6 mm of sealant (Sikaflex-1A minimum)"),
                "wall_contact": _c(both, both >= E["contact_share"], E["contact_share"], "share of the length wetted against "
                                                                                        "both joint walls ('full contact')"),
                "no_void": _c(void, void == 0, 0, "joint cells with less than 1 mm of sealant", "cells"),
                "smear": _with_off(_c(smear, smear < E["smear_cm2_m"], E["smear_cm2_m"], "sealant beyond 1 mm of the joint "
                                      "edges per metre", "cm2/m"), r, "on the wall beyond the joint's modelled strip"),
                "tooled_before_skin": _c(bool(before), before, True, "every tooled cell worked before the skin formed")}

    # ------------------------------------------------------------------ the wallboard patch
    def _drywall(self, r):
        h, t = r.h, r.t
        holes = r.holes
        filled = float(h[holes].min())
        comp = t > 5e-5
        mesh = r.mesh
        off = comp & ~mesh
        flush = float(np.abs(h[off]).max()) if off.any() else 0.0
        over = float(np.abs(h[comp & mesh] - P.MESH_T).max()) if (comp & mesh).any() else 0.0
        far = int((comp & ~self.dw_near).sum())
        # Filling openings is not covering fibres. Allow 10 micrometres for
        # field/interpolation resolution around the 0.5 mm contact envelope.
        bare = float((h[mesh] < r.support_height()[mesh] - 1e-5).mean()) if mesh.any() else 0.0
        return {"holes_filled": _c(filled*1e3, filled >= -0.3e-3, -0.3, "lowest point over the anchor holes against the wall "
                                                                         "plane", "mm"),
                "flush": _c(flush*1e3, flush <= E["flush_mm"]*1e-3, E["flush_mm"], "farthest compound surface from the wall "
                                                                                  "plane off the mesh (paint does not hide more)", "mm"),
                "over_mesh": _c(over*1e3, over <= E["flush_mm"]*1e-3, E["flush_mm"], "compound over the mesh against its own "
                                                                                    "plane", "mm"),
                "mesh_covered": _c(bare, bare < 0.02, 0.02, "share below the fibre-top contact envelope by more than 0.01 mm"),
                "no_stray_compound": _with_off(_c(far*r.cell**2*1e4, far == 0, 0.0, "compound beyond 150 mm of the holes and "
                                                  "the patch", "cm2"), r, "on the wall beyond the modelled board")}

    # ------------------------------------------------------------------ shared
    def _forms(self):
        f = float(self.env.world.form_peak_n)
        return _c(f, f <= P.FORM_DISTURB_N, P.FORM_DISTURB_N, "largest horizontal contact force on a form or kicker "
                                                             "(a braced form moves at about this; estimate)", "N")

    def _tools_off(self, shelves=False):
        """Tools whose origin is not within the cart's footprint and just above its top (or, with `shelves`, any of
        its three shelves: pad-finish), and the number of tools."""
        w = self.env.world
        cx, cy, cyaw = L.TASK_SETUP[self.task]["cart"]
        c, s = math.cos(cyaw), math.sin(cyaw)
        levels = [P.CART[2]]
        if shelves:
            levels = [ctr[2] + half[2] for name, ctr, half in L.cart_parts() if name in ("top", "shelf_mid", "shelf_low")]
        off = []
        for n in w.tool_names:
            p, R = w.tool_pose(n)
            dx, dy = p[0] - cx, p[1] - cy
            lx, ly = c*dx + s*dy, -s*dx + c*dy
            if not (abs(lx) < P.CART[0]/2 + 0.05 and abs(ly) < P.CART[1]/2 + 0.02
                    and any(z - 0.01 < p[2] < z + 0.12 for z in levels)):
                off.append(n)
        if shelves:                                          # pad-finish: put down on the cart and let go
            held = self._tools_held()
            off += [f"{n} (held)" for n in held if n not in off]
            off += [f"{n} (not resting on the cart)" for n in self._tools_unsupported() if n not in off and n not in held]
        return off, len(w.tool_names)

    def _real_contacts(self):
        """(mj body pairs, mj geom pairs) of the world's real contacts at the last physics tick: MuJoCo entries that are
        not excluded and not only within the detection margin (dist <= 0). None before the first tick (no contact data:
        a constructed state read without stepping), when the contact rules of tools_back are not applied."""
        w = self.env.world
        d = getattr(w, "mjd", None)
        if d is None or not getattr(w, "tick", 0) or not hasattr(w, "_geom_body"):
            return None
        n = int(d.ncon)
        g = np.asarray(d.contact.geom[:n]).reshape(-1, 2)
        real = (np.asarray(d.contact.exclude[:n]) == 0) & (np.asarray(d.contact.dist[:n]) <= 0.0)
        g = g[real]
        return w._geom_body[g], g

    def _tools_held(self):
        """Tools gripped at the end: both fingers of one hand in real contact with the tool. A single fingertip resting
        against a released tool, or a palm or link merely near it, is not a grip."""
        c = self._real_contacts()
        if c is None:
            return []
        w = self.env.world
        body, _ = c
        mj = getattr(w, "_mj_body", {})
        tool_of = {mj[w.scene.tools[n]["body"]]: n for n in w.tool_names if w.scene.tools[n]["body"] in mj}
        fingers = {}
        for name, i in w.scene.robot_body.items():
            if "finger" in name and i in mj:
                fingers[mj[i]] = name.split("_")[0]                       # the arm a finger belongs to
        touch = {}
        for a, b in body:
            for f, t in ((int(a), int(b)), (int(b), int(a))):
                if f in fingers and t in tool_of:
                    touch.setdefault((tool_of[t], fingers[f]), set()).add(f)
        return sorted({tool for (tool, arm), fs in touch.items() if len(fs) >= 2})

    def _tools_unsupported(self):
        """Tools not resting on the cart at the end: in real contact with the floor, or with neither a cart geom nor a
        tool that itself rests on the cart (a tool leaning on the cart from the floor)."""
        c = self._real_contacts()
        if c is None:
            return []
        w = self.env.world
        _, geoms = c
        labels = list(getattr(w.model, "shape_label", []))
        sg = getattr(w, "_shape_geom", {})
        cart = {sg[i] for i, lab in enumerate(labels) if lab.startswith("cart_") and i in sg}
        floor = {sg[i] for i, lab in enumerate(labels) if lab == "floor" and i in sg}
        mj = getattr(w, "_mj_body", {})
        tool_of = {mj[w.scene.tools[n]["body"]]: n for n in w.tool_names if w.scene.tools[n]["body"] in mj}
        gb = w._geom_body
        on_cart, on_floor, pairs = set(), set(), set()
        for g0, g1 in geoms:
            for ga, gb_ in ((int(g0), int(g1)), (int(g1), int(g0))):
                t = tool_of.get(int(gb[ga]))
                if t is None:
                    continue
                if gb_ in cart:
                    on_cart.add(t)
                elif gb_ in floor:
                    on_floor.add(t)
                elif int(gb[gb_]) in tool_of:
                    pairs.add((t, tool_of[int(gb[gb_])]))
        changed = True
        while changed:                                   # a tool lying on a tool that rests on the cart rests on the cart
            changed = False
            for a, b in pairs:
                if b in on_cart and a not in on_cart:
                    on_cart.add(a)
                    changed = True
        return sorted(t for t in w.tool_names if t in on_floor or t not in on_cart)

    def _tools_back(self):
        off, _ = self._tools_off()
        return _c(len(off), not off, 0, "tools not back on the cart" + (f": {', '.join(off)}" if off else ""))

    def _robot_off_pad(self):
        """Nothing of the robot left in or resting on the fresh pad or its forms: no arm link or hand origin over the pad
        and its forms lower than ROBOT_CLEAR above the form tops, no fingertip pad face (scene.PAD_TIP_Z out along its
        finger) lower than FINGER_CLEAR above them, and no fingertip pad bearing on the pad. A proxy from body poses: the
        paste meets only the nominated soles and fingertip pads, so a link inside the concrete leaves no mark."""
        w = self.env.world
        q = w.state.body_q.numpy()
        fw = P.FORM[0]
        from .scene import PAD_TIP_Z

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

    def _arms_clear(self):
        w = self.env.world
        q = w.state.body_q.numpy()
        over = [n for n, i in w.scene.robot_body.items() if ("fr3_link" in n or "finger" in n or "hand" in n)
                and L.PX0 - 0.02 < q[i, 0] < L.PX1 + 0.02 and L.PY0 - 0.02 < q[i, 1] < L.PY1 + 0.02 and q[i, 2] < 0.40]
        return _c(len(over), not over, 0, "robot links less than 0.31 m above the pad's surface")

    def evaluate(self):
        w = self.env.world
        t = self.task
        if t == "pad-finish":
            crit, prog = self._finish(w.regions["pad"])
            passed = [c["passed"] for c in crit.values()]
            clock = self.env.material_clock()
            success = all(passed)
            # score: 1 for a finished pad, else 0.95 x progress (the 0.95 a reporting convention: partial work stays
            # below success). `within_limit` is advisory, as for every task: the benchmark's own budget governs
            return dict(task=t, success=success, score=1.0 if success else round(0.95*prog["value"], 4), progress=prog,
                        reward_version=REWARD_VERSION, criteria=crit,
                        off_region_ml={n: round(float(r.off_m3)*1e6, 4) for n, r in w.regions.items()},
                        material_clock_s=round(clock, 3), limit_s=self.limit_s, within_limit=bool(clock <= self.limit_s))
        if t.startswith("pad"):
            crit = self._pad(w.regions["pad"], t)
        elif t == "patch-flush":
            crit = self._patch(w.regions["patch"])
        elif t == "tool-joints":
            crit = self._joints(w.regions["infill"])
        elif t == "tool-sealant":
            crit = self._sealant(w.regions["joint"])
        else:
            crit = self._drywall(w.regions["wallboard"])
        passed = [c["passed"] for c in crit.values()]
        clock = self.env.material_clock()
        return dict(task=t, success=bool(passed) and all(passed), score=round(sum(passed)/max(len(passed), 1), 4),
                    criteria=crit, off_region_ml={n: round(float(r.off_m3)*1e6, 4) for n, r in w.regions.items()},
                    material_clock_s=round(clock, 3), limit_s=self.limit_s,
                    within_limit=bool(clock <= self.limit_s))

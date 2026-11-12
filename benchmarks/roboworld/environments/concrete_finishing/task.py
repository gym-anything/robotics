"""Task instructions and the evaluator.

Every criterion is read from the material's fields (heights, surface states, what each cell recorded when it was last
worked: its yield stress then, whether after initial set, whether under bleed water), the start state and the world's
contact history; never from tool poses along the way or from the clock, except through what the material recorded.
Values are authored with their basis in params.EVAL. `success` needs every criterion; `score` is the
share passed, except pad-finish's (tasks/pad-finish/verifier.py): 1 on success, else
0.95 x a continuous progress from the seed's own start to the acceptance thresholds.

Each benchmark task's own criteria, time limit and graded progress are in its folder's verifier.py (tasks/<task>/):
Evaluator.evaluate calls its verify(ev), which judges them by the episode's contract (Evaluator.judge here, or the
concrete-tasks graders of task_set_progress.py). The checks the tasks share stay here, as do pad-float's, which is
not a benchmark task.
"""

import math
import numpy as np
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


# simulated-time limits: three times an expert's robot time plus the waits the material needs. A benchmark task's is
# LIMIT_S in its verifier.py; this table keeps pad-float's
LIMITS_S = {"pad-float": 35*60}
SMEAR_T = 2.0e-4             # paste thinner than 0.2 mm on a finished face is not counted as smear
FIELD_D = 0.150              # m: the field, past the edger's 76 mm and the overflow ridge beside it
LIP_W = 0.0015               # m: the edger's lip, 1.5 mm thick (tools.edger; its sole's lip_w)
OFF_TOL = 2.0e-8             # m3: paste laid past a region's modelled edge counts once it exceeds 1 cm2 at SMEAR_T
DELAM_Q = 1.0e-5             # trapped bleed water above 0.01 mm marks a cell sealed while it bled
COMPACT_MIN = 0.5*P.MORTAR_COMPACT   # half the normalized surface fabric; authored outcome definition
FAIL_TERM = 0.9              # a progress term whose criterion fails earns at most this share of it


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
        folder = L.task_verifier(self.task)
        self.limit_s = folder.LIMIT_S if folder is not None else LIMITS_S[self.task]
        self.h0 = {n: r.h.copy() for n, r in w.regions.items()}
        for n, r in w.regions.items():
            X, Y, Z = (r.o[:, None, None] + r.U[:, None, None]*r.u[None, :, None] + r.V[:, None, None]*r.v[None, None, :])
            setattr(self, f"_{n}_xyz", (X, Y, Z))
        if "pad" in w.regions:
            X, Y, _ = self._pad_xyz
            self.pad_d = np.minimum.reduce([X - L.PX0, L.PX1 - X, Y - L.PY0, L.PY1 - Y])
        if folder is not None and hasattr(folder, "prepare"):
            folder.prepare(self)            # what the task's verifier reads once, from the start state

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

    def _pad(self, r, t, own=None):
        """The pad's criteria: the task's own, `own(ev, r, core, on)` -> {name: criterion} from its verifier.py
        (pad-float's below), then the forms, the tools and the arms."""
        core = self._core(r)
        on = r.paste_mask
        out = {}
        if own is not None:
            out.update(own(self, r, core, on))
        if t == "pad-float":
            share = float(((r.a[core] >= E["float_tex"][0]) & (r.a[core] <= E["float_tex"][1]) & (r.E[core] < E["agg_max"])).mean())
            out["float_texture"] = _c(share, share >= E["tex_share"], E["tex_share"],
                                      "share of the area with a float texture of 0.15-0.6 mm and the stone down")
            out.update(self._flatness(r))
            e = self._pad_edges(r, 0.90, with_tracks=True)
            out["radius_profile"] = e["radius_profile"]
            out["tracks_floated_out"] = e["tracks_floated_out"]
            out["no_water_worked"] = self._no_water_worked(r)
            out["surface_wc"] = self._surface_wc(r)
        out["forms_undisturbed"] = self._forms()
        out["tools_back"] = self._tools_back()
        out["arms_clear"] = self._arms_clear()
        return out

    def _no_water_worked(self, r):
        """The pad cells worked while bleed water stood on them (pad-darby, pad-float, pad-trowel)."""
        on = r.paste_mask
        n = int(r.wet_worked[on].sum())
        return _c(n, n == 0, 0, "cells worked while bleed water stood on them (CIP 14)", "cells")

    def _surface_wc(self, r):
        """The largest rise of the pad surface's water-cement ratio (pad-float, pad-trowel)."""
        on = r.paste_mask
        dw = float(r.dwc[on].max())
        return _c(dw, dw < P.DWC_DUST, P.DWC_DUST,
                  "largest rise of the top 6 mm's water-cement ratio from water worked in (CIP 1)")

    def _committed_water(self, r):
        """Trapped water each cell will hold when bleeding ends: what it holds now plus its sealed share of the bleed
        water still to rise (paste.Region.slow_step: q grows by dt bleed_rate S / rho_w; sealing never decreases).
        pad-finish's criteria and pad-trowel's progress under the concrete-tasks contracts read it."""
        return r.q + np.clip(r.S, 0.0, 1.0)*remaining_bleed(r.material, r.age())

    # ------------------------------------------------------------------ shared
    def _forms(self):
        f = float(self.env.world.form_peak_n)
        return _c(f, f <= P.FORM_DISTURB_N, P.FORM_DISTURB_N, "largest horizontal contact force on a form or kicker "
                                                             "(a braced form moves at about this; estimate)", "N")

    def _tools_off(self, shelves=False):
        """Tools whose origin is not within the cart's footprint and just above its top (or, with `shelves`, any of
        its three shelves: pad-finish), and the number of tools."""
        w = self.env.world
        cx, cy, cyaw = L.station(self.task)["cart"]
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

    def _arms_clear(self):
        w = self.env.world
        q = w.state.body_q.numpy()
        over = [n for n, i in w.scene.robot_body.items() if ("fr3_link" in n or "finger" in n or "hand" in n)
                and L.PX0 - 0.02 < q[i, 0] < L.PX1 + 0.02 and L.PY0 - 0.02 < q[i, 1] < L.PY1 + 0.02 and q[i, 2] < 0.40]
        return _c(len(over), not over, 0, "robot links less than 0.31 m above the pad's surface")

    def evaluate(self):
        """The episode's outcome: a benchmark task's verifier.py (tasks/<task>/) decides, verify(ev); pad-float, a
        task without a folder, is judged on its pad criteria."""
        folder = L.task_verifier(self.task)
        if folder is not None:
            return folder.verify(self)
        return self.judge(lambda ev: ev._pad(ev.env.world.regions["pad"], ev.task))

    def judge(self, criteria, graded=None):
        """The released result (criteria-share-v1) of the task's criteria, `criteria(ev)` -> {name: criterion}:
        `success` needs every one and `score` is the share passed. `graded`, the task's progress, is read only by the
        concrete-tasks contracts (task_set_progress.Evaluator.judge)."""
        w = self.env.world
        t = self.task
        crit = criteria(self)
        passed = [c["passed"] for c in crit.values()]
        clock = self.env.material_clock()
        return dict(task=t, success=bool(passed) and all(passed), score=round(sum(passed)/max(len(passed), 1), 4),
                    criteria=crit, off_region_ml={n: round(float(r.off_m3)*1e6, 4) for n, r in w.regions.items()},
                    material_clock_s=round(clock, 3), limit_s=self.limit_s,
                    within_limit=bool(clock <= self.limit_s))

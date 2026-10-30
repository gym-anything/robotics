"""Seven opt-in evaluators, using the frozen task.py predicates and material state.

No stepping, scene construction or physics changes. The eighth task, pad-finish,
keeps its own evaluator: this one is not installed for it. Form and robot-clear
checks gate success; they are not points.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import layout as L, params as P
from .task_set import CONTRACT, TASKS
from .task import (Evaluator as OriginalEvaluator, E, FAIL_TERM, DELAM_Q, COMPACT_MIN,
                   LIP_W, SMEAR_T, OFF_TOL, blister_lift, _c)

FAIL_FACTOR = 0.95


def clip(value):
    return float(np.clip(value, 0.0, 1.0))


def falloff(value, threshold, zero):
    return clip((zero - value)/(zero - threshold))


def band(values, lo, hi, scale):
    """Continuous per-cell credit: one inside the band, linear outside it."""
    return np.clip(1.0 - np.maximum(lo - values, values - hi)/scale, 0.0, 1.0)


@dataclass(frozen=True)
class Quantity:
    value: float
    target: float
    passed: bool
    keep: float = 1.0
    group: str = ""
    weight: float = 1.0


class Evaluator(OriginalEvaluator):
    def __init__(self, env):
        if env.config.task not in TASKS[:-1]:
            raise ValueError("this evaluator is only for the seven tasks other than pad-finish")
        super().__init__(env)
        r = env.world.regions.get("pad")
        if r is not None:
            self._D0 = r.D.copy()
        _, self.start, _ = self._measure()

    def _tools_back(self):
        off, _ = self._tools_off(shelves=True)
        return _c(len(off), not off, 0, "tools not released and resting on a cart shelf"
                  + (f": {', '.join(off)}" if off else ""))

    def _measure(self):
        """Return the success criteria, graded work quantities, and defect factors."""
        w, t = self.env.world, self.task
        q, gates = {}, {}

        def add(name, value, target, *, criterion=None, keep=1.0, group=None, weight=1.0):
            c = crit[criterion or name]
            q[name] = Quantity(float(value), float(target), c["passed"], clip(keep), group or name, weight)

        def scalar(name, zero, *, absolute=False, group=None, weight=1.0):
            c = crit[name]
            v = abs(c["value"]) if absolute else c["value"]
            add(name, -v, -c["threshold"], keep=falloff(v, c["threshold"], zero), group=group, weight=weight)

        def residue(name, r, mask, zero_cm2, *, length=1.0):
            # Volume above the visible-smear thickness responds before a whole cell
            # crosses the binary acceptance boundary. Include recoverable catch tiles.
            excess = float(np.maximum(r.t[mask] - SMEAR_T, 0).sum())*r.cell**2
            excess += max(float(r.off_m3) - OFF_TOL, 0.0)
            area_equivalent = excess/SMEAR_T*1e4/length
            add(name, -area_equivalent, 0, keep=falloff(area_equivalent, 0, zero_cm2))

        if t.startswith("pad"):
            r = w.regions["pad"]
            crit = self._pad(r, t)
            on, core = r.paste_mask, self._core(r)
            off, n = self._tools_off(shelves=True)
            gates["tools"] = 0.9 + 0.1*(n - len(off))/max(n, 1)
            if t in ("pad-darby", "pad-trowel"):
                scalar("flat_plane", 2*E["plane_mm"], group="level", weight=0.25)
                scalar("flat_straightedge", 2*E["gap_mm"], group="level", weight=0.25)
                gaps = self._placement_gaps*1e3
                graded = np.clip(1.0 - np.maximum(gaps - E["gap_mm"], 0)/E["gap_mm"], 0, 1)
                add("placements", float(graded.mean()), 1, criterion="flat_straightedge", group="level", weight=0.5)
                gates["water_worked"] = 1.0 - float((r.wet_worked[on] > 0).mean())
            if t == "pad-darby":
                # Exposure itself gives feedback before a cell reaches the 2% limit.
                raw = np.clip((0.30 - r.E[core])/(0.30 - E["agg_max"]), 0, 1)
                share = crit["stone_down"]["value"]
                add("stone_down", 0.5*share + 0.5*float(raw.mean()), E["agg_share"])
                gates["open_surface"] = falloff(float(r.S[on].max()), E["seal_max"], 0.6)
            elif t == "pad-edge":
                for side, value in self._edge_credit(r).items():
                    add(f"edge_{side}", value, 1, criterion="radius_profile", group="edge")
                    # Both arc and lip must pass before an edge term can be complete.
                    v = q[f"edge_{side}"]
                    q[f"edge_{side}"] = Quantity(v.value, v.target,
                        v.passed and crit["lip_passed"]["passed"], v.keep, v.group, v.weight)
                gates["gouge"] = falloff(crit["no_gouge"]["value"], E["gouge_mm"], 6.0)
                gates["field"] = falloff(crit["field_untouched"]["value"], E["field_change_mm"], 5.0)
                edge = on & (self.pad_d < 0.080)
                gates["edge_dry"] = 1.0 - float((r.wet_worked[edge] > 0).mean())
            else:
                # Align early-stop water damage and all-shelf tool return with
                # pad-finish, without changing that task or any material law.
                qf = self._committed_water(r)
                blister = blister_lift(qf, r.bl, r.cell) > 3e-4
                dl = float((qf[on] > DELAM_Q).mean())
                crit["delamination_risk"] = _c(dl, dl < E["delam_max"], E["delam_max"],
                    "share trapping more than 0.01 mm bleed water by the end of bleeding")
                nb = int(blister[on].sum())
                crit["no_blister"] = _c(nb, nb == 0, 0, "cells blistered now or by committed bleed water", "cells")
                gates["water_free"] = 1.0 - float(((qf > DELAM_Q) | blister)[on].mean())
                gates["surface_free"] = 1.0 - float(((r.wet_worked > 0) | (r.dwc >= P.DWC_DUST)
                    | ((r.late > 0) & (r.D >= 0.98)))[on].mean())
                # water_worked is already part of surface_free: don't count it twice.
                del gates["water_worked"]
                tex = np.clip((E["float_tex"][0] - r.a)/(E["float_tex"][0] - E["trowel_tex"]), 0, 1)
                dens = np.clip((r.D - self._D0)/np.maximum(E["dens_min"] - self._D0, 1e-6), 0, 1)
                raw = float(np.minimum(tex, dens)[core].mean())
                add("smooth_hard", 0.5*crit["smooth_hard"]["value"] + 0.5*raw, E["tex_share"])
                radius = crit["radius_profile"]["value"]
                add("radius_profile", radius, 0.90, keep=clip((radius - 0.45)/0.45))

        elif t == "patch-flush":
            r = w.regions["patch"]
            crit = self._patch(r)
            on = r.paste_mask
            inner = on & (~np.roll(on, 1, 0) | ~np.roll(on, -1, 0)
                          | ~np.roll(on, 1, 1) | ~np.roll(on, -1, 1))
            scalar("no_lip", E["lip_hard_mm"])
            scalar("flush_mean", 2*E["patch_mean_mm"], absolute=True)
            fill = np.clip(1.0 - np.maximum(-5e-4 - r.h[inner], 0)/0.01, 0, 1)
            add("against_the_cut", 0.5*crit["against_the_cut"]["value"] + 0.5*float(fill.mean()), E["fill_share"])
            timely = (r.worked > 0) & (r.tau_w <= 30e3)
            lo, hi = E["float_tex"]
            tex = band(r.a, lo, hi, hi)*timely
            share = float(((r.a >= lo) & (r.a <= hi) & timely)[on].mean())
            add("float_texture", 0.5*share + 0.5*float(tex[on].mean()), E["tex_share"])
            # An arbitrarily late bad stroke still prevents success; it cannot earn
            # valid texture credit simply because another cell was worked in time.
            gates["timely"] = 1.0 - float(((r.worked > 0) & (r.tau_w > 30e3))[on].mean())
            mask = ~on & (distance_transform_edt(~on)*r.cell > 0.010)
            residue("no_feather_edge", r, mask, 20.0)

        elif t == "tool-joints":
            r = w.regions["infill"]
            crit = self._joints(r)
            on = r.paste_mask
            centre = on & (self.joint_e >= 0.75*P.JOINT_W/2)
            depth = -r.h[centre]*1e3
            lo, hi = E["concave_mm"]
            graded = float(band(depth, lo, hi, hi).mean())
            add("concave", 0.5*crit["concave"]["value"] + 0.5*graded, E["joint_share"], group="tooling")
            graded = float(np.clip(r.compact[centre]/COMPACT_MIN, 0, 1).mean())
            add("compacted", 0.5*crit["compacted"]["value"] + 0.5*graded, E["joint_share"], group="tooling")
            lo, hi = E["mortar_window"]
            tw = r.tau_w[centre]
            graded = float(np.minimum(np.clip(tw/lo, 0, 1), np.clip((2*hi - tw)/hi, 0, 1)).mean())
            add("tooled_thumbprint_hard", 0.5*crit["tooled_thumbprint_hard"]["value"] + 0.5*graded, 0.90, group="tooling")
            deficit = float(np.maximum(-r.h[on] - (E["concave_mm"][1]*1e-3 + 1.5e-3), 0).mean())
            add("no_void", -deficit, 0, keep=clip(1.0 - deficit/0.003))
            length = max(float(centre.sum())*r.cell/max(int(round(0.75*P.JOINT_W/r.cell)) - 1, 1), 1e-3)
            residue("smear", r, ~on, 6.0, length=length)

        elif t == "tool-sealant":
            r = w.regions["joint"]
            crit = self._sealant(r)
            seg, a = self.sj_seg, self.sj_across
            centre = seg & (np.abs(a) < 0.002)
            lo, hi = E["sealant_centre_mm"]
            grade = np.minimum(band(-r.h[centre]*1e3, lo, hi, 6.0),
                               np.clip(r.t[centre]/(E["sealant_min_mm"]*1e-3), 0, 1))
            add("concave_and_thick", 0.5*crit["concave_and_thick"]["value"] + 0.5*float(grade.mean()), 0.95)
            rows = seg.any(axis=0)
            left = np.max(np.where(seg & r.edge_l, np.clip(r.contact_l, 0, 1), 0), axis=0)[rows]
            right = np.max(np.where(seg & r.edge_r, np.clip(r.contact_r, 0, 1), 0), axis=0)[rows]
            add("wall_contact", min(float(left.mean()), float(right.mean())), E["contact_share"])
            void = float(np.maximum(1e-3 - r.t[seg], 0).mean())
            add("no_void", -void, 0, keep=clip(1.0 - void/(0.02*1e-3)))
            _, _, z = self._joint_xyz
            mask = ~r.paste_mask & (z >= P.CJ_SEGMENT[0]) & (z <= P.CJ_SEGMENT[1]) & (np.abs(a) > P.CJ_W/2 + 0.001)
            residue("smear", r, mask, 6.0, length=P.CJ_SEGMENT[1] - P.CJ_SEGMENT[0])
            worked = seg & (r.worked > 0)
            age = np.maximum(r.seal_age, r.seal_t - r.t0)
            bad = (r.skin_torn > 0) | (worked & (age >= r.material.skin))
            # Unworked cells are neutral, not a point for waiting. Damage reduces
            # progress even when a constructed bad cell is no longer marked worked.
            gates["skin_sound"] = 1.0 - float(bad[seg].mean())

        else:
            r = w.regions["wallboard"]
            crit = self._drywall(r)
            x, _, z = self._wallboard_xyz
            for i, (hx, hz) in enumerate(P.DW_HOLES):
                hole = r.holes & ((x - hx)**2 + (z - hz)**2 < (P.DW_HOLE_D/2)**2)
                value = float(r.h[hole].min())*1e3
                add(f"hole_{i}", value, -0.3, criterion="holes_filled", group="holes")
            comp = r.t > 5e-5
            for name, mask, plane in (("flush", comp & ~r.mesh, 0), ("over_mesh", comp & r.mesh, P.MESH_T)):
                scalar(name, 2*E["flush_mm"], group=name, weight=0.5)
                errors = np.abs(r.h[mask] - plane)*1e3
                graded = float(np.clip(1 - np.maximum(errors - E["flush_mm"], 0)/3.0, 0, 1).mean()) if len(errors) else 1.0
                add(name + "_area", graded, 1, criterion=name, group=name, weight=0.5)
            deficit = np.maximum(r.support_height()[r.mesh] - 1e-5 - r.h[r.mesh], 0)
            graded = float(np.clip(1 - deficit/P.MESH_T, 0, 1).mean())
            add("mesh_covered", 0.5*(1 - crit["mesh_covered"]["value"]) + 0.5*graded, 0.98,
                keep=clip(1 - float((deficit > 0).mean())))
            # Drywall stray compound uses its 0.05 mm visibility threshold, not SMEAR_T.
            excess = float(np.maximum(r.t[~self.dw_near] - 5e-5, 0).sum())*r.cell**2
            excess += max(float(r.off_m3) - OFF_TOL, 0)
            area_equivalent = excess/5e-5*1e4
            add("no_stray_compound", -area_equivalent, 0, keep=falloff(area_equivalent, 0, 10.0))
        return crit, q, gates

    def _edge_credit(self, r):
        """Per-side graded arc/lip credit against the legacy form-top reference.

        Uses the same lip exclusion, allowed shifts and 1.5 mm tolerance as _edges.
        Radius errors shrink continuously before an entire line passes.
        """
        x, y, _ = self._pad_xyz
        nx = int(((x[:, 0] > L.PX0) & (x[:, 0] < L.PX1)).sum())
        out = {s: [] for s in "snwe"}
        rad, tol = P.TOOLS["edger"]["radius"], E["radius_tol_mm"]*1e-3
        for i, (h, d, comp, _, _) in enumerate(self._perimeter(r)):
            side = "sn"[i % 2] if i < 2*nx else "we"[(i - 2*nx) % 2]
            if not len(h):
                continue
            error = math.inf
            for shift in (0.0, 0.5*r.cell, r.cell):
                ds = d - shift
                arc = P.FORM[1] - (rad - np.sqrt(np.maximum(rad*rad - (rad - ds)**2, 0)))
                sel = (ds >= LIP_W) & (ds < rad)
                if sel.any():
                    error = min(error, float(np.abs(h[sel] - arc[sel]).max()))
            radius = clip(1.0 - max(error - tol, 0)/rad)
            lip_cells = comp[d < 0.0025]
            lip = clip(float(lip_cells.max())/COMPACT_MIN) if len(lip_cells) else 0.0
            out[side].append(0.5*radius + 0.5*radius*lip)
        return {s: float(np.mean(values)) if values else 0.0 for s, values in out.items()}

    def evaluate(self):
        crit, quantities, gates = self._measure()
        groups, details = {}, {}
        for name, q in quantities.items():
            start = self.start[name]
            if not all(math.isfinite(x) for x in (q.value, q.target, q.keep, start.value)):
                raise ValueError(f"Nonfinite task quantity: {name}")
            if start.passed:
                gates[name] = 1.0 if q.passed else q.keep
                continue
            # A failing scalar already numerically at its target (e.g. a strict
            # boundary or an auxiliary acceptance gate) cannot divide by zero.
            span = q.target - start.value
            raw = clip((q.value - start.value)/span) if span > 1e-12 else 0.0
            value = 1.0 if q.passed else FAIL_TERM*raw
            details[name] = value
            groups.setdefault(q.group, []).append((value, q.weight))
        terms = {g: sum(v*w for v, w in items)/sum(w for _, w in items) for g, items in groups.items()}
        work = float(np.mean(list(terms.values()))) if terms else 0.0
        if not all(math.isfinite(v) for v in gates.values()):
            raise ValueError("Nonfinite task defect factor")
        progress = clip(work*math.prod(gates.values()))
        success = bool(crit) and all(c["passed"] for c in crit.values())
        clock = self.env.material_clock()
        return dict(task=self.task, success=success, score=1.0 if success else FAIL_FACTOR*progress,
                    reward_version=CONTRACT, criteria=crit,
                    progress=dict(value=progress, work=work, terms=terms, quantities=details, factors=gates,
                                  start={k: dict(value=v.value, target=v.target, work=not v.passed)
                                         for k, v in self.start.items()}),
                    off_region_ml={n: round(float(r.off_m3)*1e6, 4) for n, r in self.env.world.regions.items()},
                    material_clock_s=round(clock, 3), limit_s=self.limit_s, within_limit=bool(clock <= self.limit_s))

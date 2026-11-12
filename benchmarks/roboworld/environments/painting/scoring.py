"""Opt-in task progress from film/nap measurements, without stepping any physics.

Failed scores use behind-radiator's 0.9 factor. Units and tolerances are params.EVAL's
and task.BEHIND_RADIATOR's. A task's own parts are in its verifier.py (tasks/<task>/): NEED, COVER_DFT and ACRYLIC,
progress(e, checks) for a task with its own progress, quality(e, checks, defects) for its own quality terms.
"""
import numpy as np

from . import params as P
from .catalogue import CONTRACT
from .layout import task_verifier

FAILED = 0.9
E = P.EVAL
PIGMENT_SHARE = 0.95


def clip(x):
    return float(np.clip(x, 0.0, 1.0))


def band_credit(g, lo, hi):
    """Full credit in the coat band, linear below lo and from hi to 2*hi."""
    return np.clip(np.minimum(g/lo, (2*hi - g)/hi), 0.0, 1.0)


def remaining(credit, cell, need):
    return max(0.0, float(np.sum(1.0 - credit))*cell - (1.0 - need)*credit.size*cell)


def paper_colour(ch, task):
    """Requested acrylic's projected dry thickness and fraction of the film.

    The instruction names a paint (the task's ACRYLIC), so a coat from the other palette well cannot
    complete it. This reads species amounts already in the simulator's film.
    """
    colour = getattr(task_verifier(task), "ACRYLIC", "ultramarine")
    index = next((i for i, paint in enumerate(ch.paints) if paint.color == colour), None)
    wanted = np.zeros_like(ch.w) if index is None else ch.d[index] + ch.s[index]
    total = ch.projected_dft()
    share = np.divide(wanted, total, out=np.zeros_like(total), where=total > 0)
    return wanted, share


def wall_delta(e):
    from .paint import stack_reflectance, delta_e
    ch = e.paint.charts["wall"]
    sl = e._bbox("wall", e.targets["wall"])
    d = (ch.d + ch.s)[(slice(None),) + sl]
    rgb = stack_reflectance(ch.paints, d, np.zeros_like(d), np.zeros_like(ch.w[sl]), ch.substrate_rgb[sl])
    return delta_e(rgb, ch.paints[0].rinf[None, None])


class TaskProgress:
    def __init__(self, evaluator):
        self.e = evaluator
        self.task = evaluator.task
        self.declaration = {}
        if hasattr(evaluator.folder, "progress"):         # the task's own progress (load-roller): no target chart
            return
        self.key = next(n for n in ("wall", "paper", "crate", "print") if n in evaluator.targets)
        self.chart = evaluator.paint.charts[self.key]
        self.mask = evaluator.targets[self.key]
        self.need = getattr(evaluator.folder, "NEED", 0.95)
        self.start_delta = wall_delta(evaluator).copy() if self.key == "wall" else None
        # Includes authored pre-painted bands and any seed-specific first coat.
        self.r0 = max(self._remaining(), self.chart.area)

    def _credit(self):
        e, ch, m = self.e, self.chart, self.mask
        if self.key == "wall":
            sl = e._bbox("wall", m)
            # Continuous optical distance to full hide, relative to each cell's
            # own first-coat/substrate appearance. Already-hidden cells stay full
            # only while still hidden. Coverage is then net of the initial field.
            deficit = np.maximum(wall_delta(e) - E["hide_dE"], 0.0)
            scale = np.maximum(self.start_delta - E["hide_dE"], 1e-9)
            return np.clip(1.0 - deficit/scale, 0.0, 1.0)[m[sl]]
        if self.key == "paper":
            threshold = getattr(e.folder, "COVER_DFT", E["stencil_hide_dft"])
            wanted, share = paper_colour(ch, self.task)
            return np.clip(wanted[m]/threshold, 0.0, 1.0)*np.clip(share[m]/PIGMENT_SHARE, 0.0, 1.0)
        from .task import _gauge
        return band_credit(_gauge(ch.projected_dft(), m, ch.dx)[m], *E["spray_dft"])

    def _remaining(self):
        return remaining(self._credit(), self.chart.area, self.need)

    def capture_declaration(self):
        # Horizontal pooling is judged at declaration, before the film settle.
        self.declaration = {n: self.e.paint.charts[n].h().copy() for n in ("crate", "print") if n in self.e.targets}

    def reward(self, checks, settled):
        e = self.e
        own = getattr(e.folder, "progress", None)
        if own is not None:
            progress, detail = own(e, checks)
        else:
            ch, m = self.chart, self.mask
            rem = self._remaining()
            cover = clip(1.0 - rem/self.r0)
            defects = {}
            touched = float((ch.marred.astype(bool) & m).sum())*ch.area
            defects["marks"] = max(0.0, touched - E["protected_area"])
            quality = 1.0
            if self.key == "wall":
                from .task import BEHIND_RADIATOR
                C = BEHIND_RADIATOR
                defects["laps"] = max(0.0, float(((ch.lap > 0) & m).sum())*ch.area - E["lap_area"])
                defects["runs"] = max(0.0, float(((ch.flowed > C["run_flow"]) & m).sum())*ch.area - C["run_area"])
                protected, _ = e._protected(trim_sinks=True)
                defects["damage"] = C["damage_weight"]*max(0.0, protected - E["protected_area"])
                if hasattr(e.folder, "quality"):
                    quality = e.folder.quality(e, checks, defects)      # the task's own terms (roll-strip)
            elif self.key == "paper":
                over = float(((ch.projected_dft() > 3e-6) & e.targets["outside"]).sum())*ch.area
                defects["overrun"] = max(0.0, over - 0.5e-4)
            else:
                from .task import _gauge
                from scipy.ndimage import binary_dilation
                spray = e._overspray()
                emitted = spray.get("emitted_solids_mL", 0.0)
                outside = spray.get("outside_solids_mL", 0.0)
                fraction = outside/emitted if emitted > 0 else float(outside > 0)
                contain = clip((0.25 - fraction)/0.20)
                if self.key == "print":
                    g = _gauge(ch.projected_dft(), m, ch.dx)[m]
                    visible = g[g >= E["spray_dft"][0]/2]
                    ratio = float(visible.max()/visible.min()) if visible.size else 99.0
                    finish = clip((4.0 - ratio)/2.0)
                else:
                    grow = binary_dilation(m, iterations=max(1, int(round(E["bleed"]/ch.dx))))
                    bleed = float(((ch.projected_dft() > 3e-6) & ~grow).sum())/max(1, m.sum())
                    finish = clip((0.10 - bleed)/0.08)
                quality = (0.8 + 0.2*contain)*(0.8 + 0.2*finish)
                wet = self.declaration.get(self.key, ch.h()) if settled else ch.h()
                defects["pooling"] = float(np.clip((wet[m] - E["run_h"])/E["run_h"], 0.0, 1.0).sum())*ch.area
            net = cover - sum(defects.values())/self.r0
            # Spray's finish/containment factor acts on coverage before defects;
            # the strip's tool-return factor acts on the completed net work.
            progress = clip(cover*quality - sum(defects.values())/self.r0) if self.key in ("crate", "print") else clip(net)*quality
            detail = dict(coverage=cover, quality=quality, still_to_paint_cm2=rem*1e4,
                          start_to_paint_cm2=self.r0*1e4, defects_cm2={k: v*1e4 for k, v in defects.items()})
        return dict(contract=CONTRACT, score_rule=f"1 if success else {FAILED} x progress", progress=progress, **detail)

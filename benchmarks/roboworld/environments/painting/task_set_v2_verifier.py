"""painting-tasks-v2's verifier: painting.task.Evaluator (as a session without a contract builds it) with the contract's
checks and score.

Every success predicate is the legacy one (the spec keeps them unchanged: hide, runs by settle travel, laps, uniformity,
protected, every touched cell, roller release with all grip forces 0, coat, edges, overspray, load), with three
tightenings only, as in painting-tasks-v1:
- protected also counts the chartless trim sinks (stool nosing and underside, apron underside, baseboard ends), as the
  audited behind-radiator task does;
- a painted-wall verification first lands the drops already falling, as the audited task does;
- the craft coverage counts only the named acrylic (>= 95 % of the cell's solids), so the other palette well's colour
  cannot complete it.
painting-tasks-v1 instead read unrepaired marks (marred), the flowed-run area and finger contacts only; those are looser
than the spec, and v2 restores the legacy checks. Progress reads the same touched cells as the untouched check.
The score is 1 on success, else 0.9 x TaskSetV2Progress (0 at the untouched start). behind-radiator keeps its own
progress-v1 contract and is not built with this class.
"""
import numpy as np

from . import params as P
from .catalogue import INSTRUCTIONS
from .scoring import FAILED, PIGMENT_SHARE, TaskProgress, clip, paper_colour, uniformity_area
from .task import Evaluator
from .task_set_v2 import CONTRACT

E = P.EVAL


class TaskSetV2Progress(TaskProgress):
    """scoring.TaskProgress with v2's marks defect: every touched target cell, as the untouched success check reads it
    (spec: "touched beyond 0.5 cm2"), so progress reaches 1 only when that check passes. Otherwise unchanged."""

    def reward(self, checks, settled):
        e = self.e
        if self.task == "load-roller":
            roller = e.paint.applicators["roller"]
            load = clip(roller.load_mass()/0.050)
            volume = roller.s.sum(axis=0) + roller.v
            # A partially wetted nap cell earns partial credit before crossing
            # the spread verdict's 20%-of-capacity threshold.
            spread = clip(float(np.clip(volume/(0.2*P.ROLLER["sat"]*roller.C0), 0.0, 1.0).mean())/0.8)
            seconds = (e.env.world.tick - getattr(e, "_last_drip_tick", -10**9))*P.DT
            hold = float(checks["held_above_tray"]["ok"])*clip(seconds/10.0)
            progress = load*spread*(0.6 + 0.4*hold)
            detail = dict(load=load, spread=spread, hold=hold)
        else:
            ch, m = self.chart, self.mask
            rem = self._remaining()
            cover = clip(1.0 - rem/self.r0)
            defects = {}
            # The same touched cells as the untouched success check, so progress reaches 1 only when that check passes.
            touched = float((ch.touched.astype(bool) & m).sum())*ch.area
            defects["marks"] = max(0.0, touched - E["protected_area"])
            quality = 1.0
            if self.key == "wall":
                from .task import BEHIND_RADIATOR
                C = BEHIND_RADIATOR
                defects["laps"] = max(0.0, float(((ch.lap > 0) & m).sum())*ch.area - E["lap_area"])
                defects["runs"] = max(0.0, float(((ch.flowed > C["run_flow"]) & m).sum())*ch.area - C["run_area"])
                protected, _ = e._protected(trim_sinks=True)
                defects["damage"] = C["damage_weight"]*max(0.0, protected - E["protected_area"])
                if self.task == "roll-strip":
                    defects["uniformity"] = uniformity_area(e)
                    quality = 0.9 + 0.1*float(checks["roller_in_tray"]["ok"])
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


class TaskSetV2Evaluator(Evaluator):
    """Built on an environment whose config carries no contract, so the inherited checks are the legacy ones."""

    def __init__(self, env):
        super().__init__(env)
        if self.task not in INSTRUCTIONS:
            raise ValueError(f"{self.task} is not under {CONTRACT}")
        if self._task_progress is not None:
            raise ValueError(f"{CONTRACT} is built on the legacy evaluator, not on {self.contract}")
        self.contract = CONTRACT
        self._progress_v2 = TaskSetV2Progress(self)

    def measure(self, settled):
        base = super().measure(settled)
        base.pop("contract", None)                    # the legacy verdict's own contract name
        out = dict(task=base.pop("task"), settled=base.pop("settled"), contract=CONTRACT, **base)
        checks, t, ch = out["checks"], self.task, self.paint.charts
        if "paper" in self.targets:
            pch, m = ch["paper"], self.targets["paper"]
            thr = E["stroke_dft"] if t == "brush-stroke" else E["stencil_hide_dft"]
            wanted, share = paper_colour(pch, t)
            cov = float(((wanted[m] >= thr) & (share[m] >= PIGMENT_SHARE)).mean())
            checks["coverage"] = dict(value=round(cov, 4), limit=0.95, ok=cov >= 0.95,
                                      paint="cadmium_red" if t == "brush-stroke" else "ultramarine",
                                      min_pigment_share=PIGMENT_SHARE)
        if self.spec["preset"] == "wall" and t != "load-roller":
            a, det = self._protected(trim_sinks=True)
            checks["protected"] = dict(area_cm2=round(a*1e4, 3), limit_cm2=E["protected_area"]*1e4,
                                       detail_cm2={k: round(v*1e4, 3) for k, v in det.items()},
                                       ok=a <= E["protected_area"])
        oks = [c["ok"] for c in checks.values()]
        out["succeeded"] = bool(oks) and all(oks)
        out["reward"] = self._progress_v2.reward(checks, settled)
        out["score"] = 1.0 if out["succeeded"] else FAILED*out["reward"]["progress"]
        return out

    def evaluate(self, max_s=None, fast=None):
        """The legacy settle rule. Horizontal pooling is read at declaration, before the settle; on a painted wall the
        drops already falling land first (rigid world frozen, at most 2 s), as in the audited task, and the live paint is
        restored afterwards."""
        self._progress_v2.capture_declaration()
        paint = self.paint
        if "wall" not in self.targets or not len(paint.drips.pos):
            return super().evaluate(max_s, fast)
        psnap, t_drop = paint.snapshot(), 0.0
        bq = self.env.world.state.body_q.numpy()
        try:
            while len(paint.drips.pos) and t_drop < 2.0:
                paint.drips.step(bq, P.DT)
                t_drop += P.DT
            out = super().evaluate(max_s, fast)
            out.setdefault("settle", {})["drops_landed_s"] = round(t_drop, 4)
        finally:
            paint.restore(psnap)
        return out

    # ------------------------------------------------------------------ checkpoints (PaintingEnv.checkpoint/restore)
    def checkpoint_state(self):
        """Task timing only (the load-roller drop clock); no physics state."""
        return dict(drops0=self._drops0, last_drip_tick=getattr(self, "_last_drip_tick", None))

    def restore_state(self, state):
        self._drops0 = state["drops0"] if state else self.paint.drips.landed
        last = state.get("last_drip_tick") if state else self.env.world.tick
        if last is None:
            self.__dict__.pop("_last_drip_tick", None)
        else:
            self._last_drip_tick = last
        for key in ("_verdict_cache", "_hide_cache", "_prot_cache"):
            self.__dict__.pop(key, None)
        self._progress_v2.declaration.clear()
        self._decl_h.clear()
        self.env._eval_key = None

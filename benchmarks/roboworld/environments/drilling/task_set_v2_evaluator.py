"""drilling-tasks-v2 scoring: drilling.catalogue_evaluator.CatalogueEvaluator (the v1 task-set scoring, unchanged) under
the v2 contract name, plus rebar-relocate's continuous progress. State only; no simulator construction or action access.

- Every v1 term is reused as released (take, press, punch, blind, panel, timber and match scoring, the stray-hole and
  constraint rules); only the places where the released class names drilling-tasks-v1 are restated for v2: wrap,
  the panel stray-through-hole check, evaluate and the state hook.
- rebar-relocate (spec task 7; success is the legacy verdict, rated payload included): the three clear marks each meet
  the anchor-hole credit, the mark over the bar earns (S + R)/2 for the stopped hole and its replacement, less 0.25 per
  hole that cut a bar. The wall keeps the legacy nearest-hole rule for success and progress alike.
"""

from types import MethodType

import numpy as np

from . import params as P
from .catalogue_evaluator import CatalogueEvaluator, clip, mean
from .layout import WALL_AXIS_TOL_DEG, WALL_MARK_TOL_M
from .task import Evaluator, FAIL_SCORE_FACTOR, _angle_deg, _check, _ramp
from .task_set_v2 import CONTRACT, LEGACY, instruction, validate_contract

# rebar-relocate: an anchor hole's depth window (task._anchor_hole) and the far mat's front face, where the depth
# credit reaches 0 (spec: 149 mm; concrete.ConcreteFields puts the far bars at WALL_THICK - FAR_MAT_COVER - r)
ANCHOR_DEPTH_M = (0.108, 0.125)
FAR_MAT_FACE_M = P.WALL_THICK_M - P.FAR_MAT_COVER_M - P.REBAR_D_M
REBAR_CUT_M = 0.0005                     # task._anchor_hole / _rebar_relocate: more than 0.5 mm into a bar cuts it
REPLACEMENT_FAR_M = 0.200                # spec: the distance credit outside the window is 0 at 200 mm


class TaskSetV2Evaluator(CatalogueEvaluator):
    task_contract = LEGACY

    @classmethod
    def wrap(cls, old, contract):
        if not isinstance(old, cls):
            new = cls.__new__(cls)
            new.__dict__.update(old.__dict__)  # keep start poses, tick claims and checkpoint schema
            old = new
        old.task_contract = validate_contract(old.task, contract)
        old.instruction = instruction(old.task, old.task_contract)
        return old

    def nearest(self, name, p_local, kinds=("drilled",)):
        if name == "wall":
            # the wall task (rebar-relocate) keeps the legacy nearest-hole rule for success and progress alike
            return Evaluator.nearest(self, name, p_local, kinds)
        return super().nearest(name, p_local, kinds)

    def _panel_holes(self):
        out = super()._panel_holes()           # the legacy checks (the released class adds its own only under v1)
        if self.task_contract == CONTRACT:
            bad = [h.id for h in self._panel_damage() if h.through]
            out.append(_check("no stray through-hole", not bad, bad, kind="constraint"))
        return out

    def _wall_bore(self, h, e):
        """rebar-relocate anchor-hole credit (spec task 7): ramp(entry; 3, 6 mm) x ramp(axis; 3, 6 deg) x
        min(depth / 108 mm, 1) x ramp(depth; 125 mm, far-mat face). A cut bar or a broken bit's stub zeroes it."""
        if not self._intact(h) or h.rebar_damage_m > REBAR_CUT_M:
            return 0.
        fd = h.full_depth()
        return (_ramp(e, WALL_MARK_TOL_M, 2*WALL_MARK_TOL_M)
                * _ramp(_angle_deg(h.a, -h.face_normal), WALL_AXIS_TOL_DEG, 2*WALL_AXIS_TOL_DEG)
                * clip(fd/ANCHOR_DEPTH_M[0])*_ramp(fd, ANCHOR_DEPTH_M[1], FAR_MAT_FACE_M))

    def _stop_share(self, h):
        """S of the hole over the bar: 1 once it reached the bar uncut, else its depth / the depth at which it would
        reach the bar (read from the wall's bar geometry, as task._reached_bar does)."""
        if h.rebar_damage_m > REBAR_CUT_M or not self._intact(h):
            return 0.
        if self._reached_bar(h):
            return 1.
        d = self.w.hm.drillables["wall"]
        if d.fields is not None:
            gap = d.fields.bar_distance(d.to_fields(h.p0 + h.a*h.depth))[0] - 0.5e-3*h.d_mm
        else:
            gap = P.REBAR_COVER_M + P.REBAR_D_M - h.depth   # the vertical bar's face: cover + the horizontal layer
        gap = max(gap, 0.)
        return clip(h.depth/(h.depth + gap)) if h.depth + gap > 0. else 0.

    def _rebar_state(self):
        """Per-mark credit M_k (spec task 7), the number of holes that cut a bar, and the stopped hole."""
        L, kbar = self.L, self.L["rebar_mark"]
        marks = [self._wall_mark_local(mk) for mk in L["marks"]]
        credits = []
        for k, m in enumerate(marks):
            if k == kbar:
                credits.append(None)
                continue
            hit = self.nearest("wall", m)
            credits.append(0. if hit is None or hit[1] > 0.02 else self._wall_bore(*hit))
        hit = self.nearest("wall", marks[kbar])
        stopped = hit if hit is not None and hit[1] <= 2*WALL_MARK_TOL_M else None
        S = 0. if stopped is None else self._stop_share(stopped[0])*_ramp(stopped[1], WALL_MARK_TOL_M,
                                                                            2*WALL_MARK_TOL_M)
        # the replacement: on the stopped hole's channel line, 2 x its depth to 150 mm from it; without a stopped
        # hole, from the mark with the layout's nominal minimum (twice the bar's face depth plus 10 mm)
        if stopped is not None:
            ref, d_min = stopped[0].p0, P.ABORTED_HOLE_DISTANCE_FACTOR*stopped[0].depth
        else:
            ref, d_min = marks[kbar], L["replacement"]["min_m"]
        others = [m for k, m in enumerate(marks) if k != kbar]
        R = 0.
        for h in self.holes("wall"):
            if stopped is not None and h is stopped[0]:
                continue
            if not all(np.linalg.norm((h.p0 - o)[[0, 2]]) > 0.02 for o in others):
                continue
            dz = abs(float(h.p0[2] - ref[2]))
            dx = abs(float(h.p0[0] - ref[0]))
            near = clip((dx - 0.5*d_min)/(0.5*d_min)) if d_min > 0. else 1.
            window = min(near, _ramp(dx, P.RELOCATE_MAX_M, REPLACEMENT_FAR_M))
            R = max(R, self._wall_bore(h, dz)*window)
        cut = sum(h.rebar_damage_m > REBAR_CUT_M for h in self.holes("wall"))
        bar_cut = stopped is not None and stopped[0].rebar_damage_m > REBAR_CUT_M
        credits[kbar] = 0. if bar_cut else (S + R)/2.
        return credits, cut, S, R

    def _progress(self):
        if self.task == "rebar-relocate":
            credits, cut, _, _ = self._rebar_state()
            return clip(mean(credits) - 0.25*cut)
        return super()._progress()

    def evaluate(self):
        if self.task_contract == LEGACY:
            return Evaluator.evaluate(self)  # includes the ledger-anchor output, byte-for-byte semantics
        if self.w.fault or not np.isfinite(self.w.bq).all() or not np.isfinite(self.w.jq).all():
            return dict(task=self.task, contract=CONTRACT, success=False, score=None, evaluated=False,
                        reason=self.w.fault or "non-finite state", instruction=self.instruction)
        out = Evaluator.evaluate(self)
        progress = self._progress()
        # Explicit defect costs and damaged-bore terms already account for these constraints.
        penalize = [c for c in out["checks"] if c["kind"] == "constraint" and not c["passed"]
                    and c["name"] != "no bit broken" and "stray" not in c["name"]]
        progress = clip(progress*(0.5 if penalize else 1.))
        out.update(contract=CONTRACT, progress=progress, evaluated=True,
                   scoring="1 on success, else 0.9 x continuous work-product progress",
                   score=1. if out["success"] else FAIL_SCORE_FACTOR*progress)
        if self.task == "rebar-relocate":
            credits, cut, S, R = self._rebar_state()
            out["progress_terms"] = dict(marks=[round(c, 4) for c in credits], stopped_hole=round(S, 4),
                                         replacement=round(R, 4), holes_cutting_a_bar=int(cut))
        if self.task == "press-drill":
            out["evaluator_note"] = "The outcome is judged: a bar held without turning may pass without vise clamping."
        return out


def _get_state(env):
    state = env._task_set_v2_get_state()
    if env.evaluator.task_contract == CONTRACT:
        ev = state["evaluation"]
        if state.get("fault") or not state.get("finite", True) or "error" in ev:
            ev = dict(ev, success=False, score=None, evaluated=False,
                      reason=state.get("fault") or ev.get("error") or "non-finite state")
        state["evaluation"] = dict(ev, contract=CONTRACT)
    return state


def configure_environment(env, contract):
    """Install only evaluator/state-report hooks on this instance; no global monkeypatch or scene changes.

    Binding get_state here covers act/reset/checkpoint/trace callers inside DrillingEnv as well as the backend.
    The original initial checkpoint's measurement schema is retained.
    """
    env.evaluator = TaskSetV2Evaluator.wrap(env.evaluator, contract)
    if not hasattr(env, "_task_set_v2_get_state"):
        env._task_set_v2_get_state = env.get_state
        env.get_state = MethodType(_get_state, env)
    return env

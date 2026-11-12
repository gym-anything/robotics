"""State-only task scoring. No simulator construction or action access.

The old evaluator supplies tick-rate measurements and common constraints. The new contract is opt-in.
A benchmark task's own terms (its checks under the contract, its continuous progress, its rating of a mark's hole) are
in its verifier.py (tasks/<task>/), which passes them to CatalogueEvaluator.evaluate_checks or defines them for nearest.
"""

import math
from types import MethodType

import numpy as np

from .catalogue import CONTRACT, LEGACY, instruction, validate_contract
from .layout import task_verifier
from .task import Evaluator, FAIL_SCORE_FACTOR, _angle_deg, _check, _in_plane, _ramp


def clip(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite task measurement")
    return max(0., min(1., x))


def mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs) if xs else 0.


def depth_share(h, diameter, thickness):
    return 1. if h.through_at(diameter) else clip(h.full_depth(diameter)/thickness)


class CatalogueEvaluator(Evaluator):
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

    def _intact(self, h):
        broken = {n for n, it in self.w.hm.items.items() if it.kind == "bit" and it.state["broken"]}
        return h.blocked_at is None and not (broken & set(h.tools))

    def _bore(self, h, error, diameter, thickness, tol, max_d=None, axis_tol=1.):
        if not self._intact(h):
            return 0.
        value = depth_share(h, diameter, thickness)*_ramp(error, tol, 3*tol)
        value *= _ramp(_angle_deg(h.a, -h.face_normal), axis_tol, 3.)
        if max_d is not None:
            value *= _ramp(h.d_mm + 2e3*h.max_offset_m, max_d, max_d + 0.5)
        return value

    def nearest(self, name, p_local, kinds=("drilled",)):
        if self.task_contract == LEGACY or kinds != ("drilled",):
            return super().nearest(name, p_local, kinds)
        # A shallow abandoned start must not hide an acceptable finished opening.
        candidates = []
        # a task's own rating of a hole as the mark's, (quality, valid): candidate(ev, h, e) in its verifier.py
        candidate = getattr(task_verifier(self.task), "candidate", None)
        for h in self.holes(name):
            e = _in_plane(h, p_local)
            if candidate is not None:
                quality, valid = candidate(self, h, e)
            else:
                quality = self._bore(h, e, 9., 0.005 if name == "angle_bracket" else 0.006, 0.0005, 9.25)
                valid = h.through_at(9.) and e <= 0.0005 and h.d_mm + 2e3*h.max_offset_m <= 9.25
            valid = valid and _angle_deg(h.a, -h.face_normal) <= 1.
            candidates.append(((bool(valid), quality, h.full_depth(), -e, -h.id), h, e))
        return max(candidates, key=lambda c: c[0])[1:] if candidates else None

    def _finish(self):
        p = self.w.scene.tools["drill_press"]
        rpm = abs(p.spindle.speed())*60/(2*math.pi)
        stopped = (0. if p.s_on else _ramp(rpm, 1., 1000.))
        up = _ramp(-p.j["quill"].pos(self.w), 0.002, 0.080)
        # Continuous bit withdrawal from every current slot, read from its actual tip pose.
        tip, _ = self.w.hm.item_world(p.item)
        clear = 1.
        for name in p.item.slots:
            d = self.w.hm.drillables[name]
            pos, rot = self.w.body_pose(d.body)
            local = rot.T @ (tip - pos)
            above = local[2] - (d.center[2] + d.half[2])
            clear = min(clear, clip(1. + above/0.080))
        return mean((stopped, up, clear))

    def _slot(self, item, name):
        slot = item.slots.get(name)
        if not slot or slot[1] != "hole":
            return None
        return next((h for h in self.holes(name) if h.id == slot[2]), None)

    def _overlap(self, item, name, h):
        """Current shank length inside this hole, not the stale insertion history."""
        if h is None or self._slot(item, name) is not h:
            return 0.
        d = self.w.hm.drillables[name]
        tip, axis = self.w.hm.item_world(item)
        pos, rot = self.w.body_pose(d.body)
        tip_local, axis_local = rot.T @ (tip - pos), rot.T @ axis
        z = float((tip_local - h.p0) @ h.a)
        base = z - item.length*float(axis_local @ h.a)
        lo, hi = sorted((base, z))
        lo, hi = max(lo, 0.), min(hi, h.thickness)
        if hi <= lo:
            return 0.
        dot = float(axis_local @ h.a)
        if abs(dot) < 1e-9:
            return 0.
        # Slot ids are necessary but not enough: reject a stale slot beside the part or an
        # impossible shank across an undersize bore. 0.6 mm is holes.CAPTURE_TOL (chamfer capture).
        radius = 0.5e-3*h.d_mm + h.max_offset_m - item.radius + 0.0006
        radial = []
        for at in (lo, hi):
            offset = tip_local - h.p0 + axis_local*((at-z)/dot) - h.a*at
            radial.append(float(np.linalg.norm(offset)))
        if radius < 0. or max(radial) > radius:
            return 0.
        return clip((hi-lo)/h.thickness)

    def _match_state(self):
        upper = [self.nearest("angle_bracket", m["local"]) for m in self.L["marks"]]
        lower = sorted(self.holes("flat_bar"), key=lambda h: (
            self._bore(h, 0., 9., 0.006, 0.0005, 9.25), h.through_at(9.), -h.id), reverse=True)[:2]
        bores = [0. if h is None else self._bore(h[0], h[1], 9., 0.005, 0.0005, 9.25) for h in upper]
        bores += [self._bore(h, 0., 9., 0.006, 0.0005, 9.25) for h in lower] + [0.]*(2-len(lower))
        rows = []
        for k in range(2):
            it = self.w.hm.items[f"m8_bolt{k}"]
            ha, hb = self._slot(it, "angle_bracket"), self._slot(it, "flat_bar")
            mark = next((i for i, hit in enumerate(upper) if hit and hit[0] is ha), None)
            eligible = mark is not None and hb is not None and any(hb is h for h in lower)
            share = mean((self._overlap(it, "angle_bracket", ha), self._overlap(it, "flat_bar", hb))) if eligible else 0.
            rows.append((mark, None if hb is None else hb.id, share))
        # Two bolts in one pair of holes cannot stand for the two fastenings.
        distinct = rows[0][0] != rows[1][0] and rows[0][1] != rows[1][1]
        bolt = mean(r[2] for r in rows) if distinct else max(r[2] for r in rows)/2.
        return upper, lower, mean(bores), bolt, distinct and all(r[2] >= 1. - 1e-6 for r in rows)

    def _match_bolt(self):
        out = super()._match_bolt()
        if self.task_contract == LEGACY:
            return out
        _, lower, _, _, spans = self._match_state()
        out.append(_check("two 9 mm bores in the flat bar, square and within 9.25 mm diameter", len(lower) == 2 and all(
            h.through_at(9.) and _angle_deg(h.a, -h.face_normal) <= 1. and
            h.d_mm + 2e3*h.max_offset_m <= 9.25 for h in lower)))
        out.append(_check("two bolts span distinct pairs of matching through holes", spans))
        return out

    def _progress(self, progress=None):
        """Continuous work-product progress: the task's own progress(ev) (its verifier.py) when it passes one."""
        if progress is not None:
            return progress(self)
        if self.task == "match-bolt":
            upper, _, bore, bolt, _ = self._match_state()
            spacing = 0. if any(h is None for h in upper) else _ramp(
                abs(float(np.linalg.norm(upper[0][0].p0 - upper[1][0].p0)) - 0.060), 0.0005, 0.0015)
            return (2*bore + bolt*spacing)/3.
        raise ValueError(f"No continuous score for {self.task}")

    def evaluate_checks(self, checks, *, contract_checks=None, progress=None, note=None, terms=None):
        """The verdict under the task's contract. Under the contract the task's hooks apply: contract_checks(ev) in
        place of checks(ev), progress(ev) for the continuous score, and note for the report's evaluator_note."""
        if self.task_contract == LEGACY:
            return super().evaluate_checks(checks)  # includes the ledger-anchor output, byte-for-byte semantics
        if self.w.fault or not np.isfinite(self.w.bq).all() or not np.isfinite(self.w.jq).all():
            return dict(task=self.task, contract=CONTRACT, success=False, score=None, evaluated=False,
                        reason=self.w.fault or "non-finite state", instruction=self.instruction)
        out = super().evaluate_checks(checks if contract_checks is None else contract_checks)
        progress = self._progress(progress)
        # Explicit defect costs and damaged-bore terms already account for these constraints.
        penalize = [c for c in out["checks"] if c["kind"] == "constraint" and not c["passed"]
                    and c["name"] != "no bit broken" and "stray" not in c["name"]]
        progress = clip(progress*(0.5 if penalize else 1.))
        out.update(contract=CONTRACT, progress=progress, evaluated=True,
                   scoring="1 on success, else 0.9 x continuous work-product progress",
                   score=1. if out["success"] else FAIL_SCORE_FACTOR*progress)
        if note is not None:
            out["evaluator_note"] = note
        return out


def _get_state(env):
    state = env._catalogue_get_state()
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
    env.evaluator = CatalogueEvaluator.wrap(env.evaluator, contract)
    if not hasattr(env, "_catalogue_get_state"):
        env._catalogue_get_state = env.get_state
        env.get_state = MethodType(_get_state, env)
    return env

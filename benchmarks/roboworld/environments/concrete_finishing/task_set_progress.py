"""Seven opt-in evaluators, using the frozen task.py predicates and material state.

No stepping, scene construction or physics changes. The eighth task, pad-finish,
keeps its own evaluator: this one is not installed for it. Form and robot-clear
checks gate success; they are not points.

Each task's graded quantities and defect factors are its verifier.py's progress (tasks/<task>/): its verify(ev)
passes its criteria and progress to Evaluator.judge, which grades them; the start is measured with them too.
"""

from dataclasses import dataclass
import math
import numpy as np

from . import layout as L
from .task_set import CONTRACT, TASKS
from .task import (Evaluator as OriginalEvaluator, E, FAIL_TERM, SMEAR_T, OFF_TOL, _c)

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
        folder = L.task_verifier(self.task)
        _, self.start, _ = self._measure(folder.criteria, folder.progress)

    def _tools_back(self):
        off, _ = self._tools_off(shelves=True)
        return _c(len(off), not off, 0, "tools not released and resting on a cart shelf"
                  + (f": {', '.join(off)}" if off else ""))

    def _measure(self, criteria, graded):
        """Return the success criteria, graded work quantities, and defect factors: the task's criteria(ev), and
        graded(ev, crit, q, gates, add, scalar, residue), its progress, which adds its quantities and factors (both
        from its verifier.py)."""
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

        crit = criteria(self)
        graded(self, crit, q, gates, add, scalar, residue)
        return crit, q, gates

    def _pad_tools(self, gates):
        """The pad tasks' tool factor: 0.9 + 0.1 x the share of the tools released and resting on a cart shelf."""
        off, n = self._tools_off(shelves=True)
        gates["tools"] = 0.9 + 0.1*(n - len(off))/max(n, 1)

    def _pad_level(self, r, add, scalar, gates):
        """The level of pad-darby and pad-trowel: the plane, the worst straightedge gap and the graded share of the
        straightedge placements, and the share of the pad not worked under bleed water."""
        on = r.paste_mask
        scalar("flat_plane", 2*E["plane_mm"], group="level", weight=0.25)
        scalar("flat_straightedge", 2*E["gap_mm"], group="level", weight=0.25)
        gaps = self._placement_gaps*1e3
        graded = np.clip(1.0 - np.maximum(gaps - E["gap_mm"], 0)/E["gap_mm"], 0, 1)
        add("placements", float(graded.mean()), 1, criterion="flat_straightedge", group="level", weight=0.5)
        gates["water_worked"] = 1.0 - float((r.wet_worked[on] > 0).mean())

    def judge(self, criteria, graded):
        """The result under this contract of the task's criteria and progress (its verify(ev) passes them)."""
        crit, quantities, gates = self._measure(criteria, graded)
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

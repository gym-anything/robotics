"""drilling-tasks-v2 scoring: drilling.catalogue_evaluator.CatalogueEvaluator (the v1 task-set scoring, unchanged) under
the v2 contract name, plus rebar-relocate's continuous progress. State only; no simulator construction or action access.

- Every v1 term is reused as released (take, press, punch, blind, panel, timber and match scoring, the stray-hole and
  constraint rules); only the places where the released class names drilling-tasks-v1 are restated for v2: wrap,
  evaluate_checks and the state hook. A benchmark task's own terms are in its verifier.py (tasks/<task>/).
- rebar-relocate (spec task 7; success is the legacy verdict, rated payload included): the three clear marks each meet
  the anchor-hole credit, the mark over the bar earns (S + R)/2 for the stopped hole and its replacement, less 0.25 per
  hole that cut a bar (tasks/rebar-relocate/verifier.py: progress, terms). The wall keeps the legacy nearest-hole rule
  for success and progress alike.
"""

from types import MethodType

import numpy as np

from .catalogue_evaluator import CatalogueEvaluator, clip
from .task import Evaluator, FAIL_SCORE_FACTOR
from .task_set_v2 import CONTRACT, LEGACY, instruction, validate_contract


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

    def evaluate_checks(self, checks, *, contract_checks=None, progress=None, note=None, terms=None):
        """The verdict under the task's contract. Under the contract the task's hooks apply: contract_checks(ev) in
        place of checks(ev), progress(ev) for the continuous score, terms(ev) for the report's progress_terms and note
        for its evaluator_note."""
        if self.task_contract == LEGACY:
            return Evaluator.evaluate_checks(self, checks)  # includes the ledger-anchor output, byte-for-byte semantics
        if self.w.fault or not np.isfinite(self.w.bq).all() or not np.isfinite(self.w.jq).all():
            return dict(task=self.task, contract=CONTRACT, success=False, score=None, evaluated=False,
                        reason=self.w.fault or "non-finite state", instruction=self.instruction)
        out = Evaluator.evaluate_checks(self, checks if contract_checks is None else contract_checks)
        progress = self._progress(progress)
        # Explicit defect costs and damaged-bore terms already account for these constraints.
        penalize = [c for c in out["checks"] if c["kind"] == "constraint" and not c["passed"]
                    and c["name"] != "no bit broken" and "stray" not in c["name"]]
        progress = clip(progress*(0.5 if penalize else 1.))
        out.update(contract=CONTRACT, progress=progress, evaluated=True,
                   scoring="1 on success, else 0.9 x continuous work-product progress",
                   score=1. if out["success"] else FAIL_SCORE_FACTOR*progress)
        if terms is not None:
            out["progress_terms"] = terms(self)
        if note is not None:
            out["evaluator_note"] = note
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

"""State-only judgements for the explicitly selected eight-task contract.

No physics edits or runtime construction. The legacy Ledger and outcome geometry
are reused; set-post is delegated intact, including its counterfactual verdict.
Each task's reading is in its folder: tasks/<task>/verifier.py, verify(e, lg, contract=CONTRACT).
"""
import numpy as np

from . import layout as L
from .catalogue import CONTRACT, REVISED, default_contract, instruction
from .task import Evaluator as LegacyEvaluator, Ledger, diag

# set-post's rule is a cap, not a multiplicative discount. Keep that rule.
FAILURE_CAP = 0.99


def clip(x):
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else 0.0


def ramp(x, full, zero):
    return clip((zero - x)/(zero - full))


def band(x, low, high, zero_low, zero_high):
    return min(clip((x - zero_low)/(low - zero_low)), ramp(x, high, zero_high))


def net_progress(raw, start):
    return clip((raw - start)/(1.0 - start)) if start < 1.0 else 0.0


def failed_score(progress):
    return min(clip(progress), FAILURE_CAP)


class TaggedEvaluator:
    """Only add provenance to the legacy result; no criteria or scores change."""
    def __init__(self, evaluator):
        self.legacy = evaluator
        self.task_contract = default_contract(evaluator.task)

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def evaluate(self):
        return dict(self.legacy.evaluate(), task_contract=self.task_contract)

    def progress(self):
        return dict(self.legacy.progress(), task_contract=self.task_contract)


class TaskSetEvaluator(LegacyEvaluator):
    STOPS_REPORTED_ONLY = REVISED

    def __init__(self, env):
        super().__init__(env)
        lg = Ledger(env)
        start = getattr(L.task_verifier(self.task), "start_cover", None)
        self.start_cover = start(self, lg) if start is not None else None
        self.start_raw = self._read(lg)[1]

    def _read(self, lg):
        return L.task_verifier(self.task).verify(self, lg, contract=CONTRACT)

    def progress(self):
        # Legacy's advisory error result has no contract; tag even this path.
        return dict(super().progress(), task_contract=CONTRACT)

    def _evaluate(self, final):
        common = self.common()
        criteria, raw, diagnostics = self._read(Ledger(self.env))
        criteria = [c for c in common if c["name"] != "protective_stops"] + criteria
        diagnostics += [diag(c["name"], c["value"], "reported only; recovery does not invalidate the job")
                        for c in common if c["name"] == "protective_stops"]
        success = all(c["ok"] for c in criteria)
        progress = net_progress(raw, self.start_raw)
        w = self.env.world
        return dict(task=self.task, task_contract=CONTRACT, succeeded=success,
                    score=1.0 if success else round(failed_score(progress), 4), progress=progress,
                    raw_progress=raw, start_raw_progress=self.start_raw,
                    score_rule="1 on success, else min(progress, 0.99) (set-post's rule)",
                    criteria=criteria, diagnostics=diagnostics, tick=int(w.tick), time_s=round(w.sim_time, 4),
                    strikes=list(w.strikes), reflexes=list(w.reflexes)[-5:])


def install(env, contract):
    """Install after the original environment has settled; do not move any body."""
    old = env.evaluator
    env.spec["instruction"] = instruction(env.spec["task"], contract)
    if contract == CONTRACT:
        evaluator = TaskSetEvaluator(env)
        env.world.recorders.remove(old._record)  # exactly one slab-history recorder
    else:
        evaluator = TaggedEvaluator(old)
    env.evaluator = evaluator
    env._eval_key = None

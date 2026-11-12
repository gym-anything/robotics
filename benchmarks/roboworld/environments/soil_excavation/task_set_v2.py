"""soil-tasks-v2: state-only judgements for the explicitly selected render-checked eight-task set.

No physics edits or runtime construction. The legacy Ledger and outcome geometry are reused, as in
soil-tasks-v1 (task_set.py). Differences from v1: a failed run scores 0.9 x progress; progress is net of the
settled start plus an idle margin, so an untouched or idle start reads exactly 0; penetrate-marked's progress has
no hand factor (both hands stay a success criterion); break-hardpan credits only soil the robot worked
(WorkWindows). Under this contract set-post keeps its own verdict intact (criteria, counterfactual load test,
success); only its failed-run score is measured from the settled start. Its own contract (no contract selector)
is untouched.

Each task's reading is in its folder: tasks/<task>/verifier.py, verify(e, lg, contract=CONTRACT); a task judged
with its own evaluator names it there (V2_EVALUATOR: break-hardpan's work windows, set-post's rebased score).
"""
import numpy as np

from . import layout as L
from .catalogue import REVISED
from .catalogue_v2 import CONTRACT, instruction
from .task import Evaluator as LegacyEvaluator, Ledger, diag

# 1 on success, else 0.9 x progress. Explicit contract only; bare selections, soil-tasks-v1 and set-post's own
# contract (soil-set-post-v1) keep their own rules, including set-post's min(progress, 0.99).
FAILURE_FACTOR = 0.9
SCORE_RULE = "1 on success, else 0.9 x progress; progress net of the settled start plus an idle margin"
# Idle margins on the raw progress reading: credit starts only above the settled start's reading plus the margin, so an
# untouched or idle start reads exactly 0 in every run. Measured in the render-check round, robot idle
# holding every target:
#   set-post: the load-test reading differed by up to 4.4e-5 between runs (0.0024009-0.0024452 at the start), by up
#     to 7.0e-7 between the start's baseline and a later start reading, and rose 4.7e-6 in 2 s idle;
#   expose-line: cover creep raised the reading 4.2e-6 in 2 s;
#   break-hardpan: its loosening reading rose steadily under the idle robot (+1.5e-3 in 15 s, J_p creep of the soil at
#     rest); it now counts only soil the robot worked (WorkWindows) and reads exactly 0 however long the robot idles;
#   the other five read 0 untouched and after 2 s idle.
# 0.002 is 45x set-post's run-to-run spread and covers set-post's and expose-line's idle drift many times over.
IDLE_MARGIN = 0.002
IDLE_MARGINS = {}


def clip(x):
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else 0.0


def ramp(x, full, zero):
    return clip((zero - x)/(zero - full))


def band(x, low, high, zero_low, zero_high):
    return min(clip((x - zero_low)/(low - zero_low)), ramp(x, high, zero_high))


def idle_margin(task):
    return IDLE_MARGINS.get(task, IDLE_MARGIN)


def net_progress(raw, start, margin=0.0):
    """Progress above the zero point start + margin, normalized to full credit at 1."""
    zero = start + margin
    return clip((raw - zero)/(1.0 - zero)) if zero < 1.0 else 0.0


def failed_score(progress):
    return FAILURE_FACTOR*clip(progress)


class History:
    """The verdict's recorded history (the legacy slab history; break-hardpan's work windows): saved and restored with
    the backend's checkpoints, restarted with each episode."""

    def history(self):
        return dict(base_off_slab=self.base_off_slab)

    def restore_history(self, h):
        self.base_off_slab = h["base_off_slab"]

    def restart_history(self):
        self.base_off_slab = 0.0


class TaskSetEvaluator(History, LegacyEvaluator):
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
        margin = idle_margin(self.task)
        progress = net_progress(raw, self.start_raw, margin)
        w = self.env.world
        return dict(task=self.task, task_contract=CONTRACT, succeeded=success,
                    score=1.0 if success else round(failed_score(progress), 4), progress=progress,
                    raw_progress=raw, start_raw_progress=self.start_raw, idle_margin=margin,
                    score_rule=SCORE_RULE,
                    criteria=criteria, diagnostics=diagnostics, tick=int(w.tick), time_s=round(w.sim_time, 4),
                    strikes=list(w.strikes), reflexes=list(w.reflexes)[-5:])


def install(env, contract):
    """Install the explicit soil-tasks-v2 contract after the original environment has settled; do not move any body.
    Any other contract leaves the environment exactly as built: its evaluator, instruction and results unchanged."""
    if contract != CONTRACT:
        return
    old = env.evaluator
    env.spec["instruction"] = instruction(env.spec["task"], contract)
    env.evaluator = getattr(L.task_verifier(env.spec["task"]), "V2_EVALUATOR", TaskSetEvaluator)(env)
    env.world.recorders.remove(old._record)  # exactly one slab-history recorder
    env._eval_key = None

"""Outcome of fillet-2f: a continuous 1/4 in fillet along the near side of the tacked bracket, end to end with its
crater filled, that passes the AWS D1.1 visual and macro-section checks (Evaluator._eval_fillet_2f) and is solid.

verify(e, settle=False) receives the episode's task.Evaluator (e.env the WeldingEnv; settle=True, the on-demand
evaluation, judges a weld still freezing with the arc off as it freezes hands-off) and returns the evaluation:
success, score, checks, progress, reward, judged, flags and diagnostics (judge_fillet below on
Evaluator._final_rows). Score: 1 on success, else FAIL_SCORE_SCALE x fillet_progress net of the tacked start
(REWARD_CONTRACT). WELD: the task is judged on the weld's sections (the evaluator reads the start state's raw
progress, Evaluator.start_raw_progress)."""
import numpy as np

from welding.task import FAIL_SCORE_SCALE, fillet_progress

WELD = True
REWARD_CONTRACT = "progress-v1"


def verify(e, settle=False):
    return judge_fillet(e, *e._final_rows(settle))


def judge_fillet(e, rows, judged=None, flags=None):
    """fillet-2f's evaluation of one state's sections: success (every check), the progress reward and the score."""
    checks = {k: {("pass" if kk == "ok" else kk): vv for kk, vv in c.items()}
              for k, c in e._eval_fillet_2f(rows).items()}
    passed = [c["pass"] for c in checks.values()]
    success = bool(passed) and all(passed)
    reward = fillet_progress(rows, e.start_raw_progress, e._last_craters)
    reward.update(contract=REWARD_CONTRACT, score_rule="1 if success else 0.9 x progress",
                  mean_of_checks=float(np.mean(passed)))
    return dict(success=success, score=1. if success else FAIL_SCORE_SCALE*reward["progress"], checks=checks,
                progress=reward["progress"], reward=reward, judged=judged or {}, flags=flags or {},
                diagnostics=dict(e.diagnostics))

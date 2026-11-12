"""Outcome of unlock: the latch released with the near key seated in the keyway at that tick, and now the shackle's
short leg swung clear of its bore and the other key resting on its stand, not held by the robot.

verify(progress, env, plug_angle) is the native evaluator's update,
lockkey.newton15.evaluation.UnlockProgress.update: it receives the evaluator, the environment and the plug angle in
degrees and returns the evaluation (success, score, progress along the unlock chain, its stage, the checks and the
diagnostics). LockKeyEnv._evaluate calls it at every evaluation. The code stays in newton15/evaluation.py, which the
catalogue names as the evaluator: the seven other tasks' evaluators (task_set/scoring.py, scoring_v2.py) build on it
as their reading of the lock, and the native environment is left unchanged."""
from lockkey.newton15.evaluation import UnlockProgress

verify = UnlockProgress.update

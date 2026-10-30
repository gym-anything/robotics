"""Task-only integration around the frozen MixEnv/Hob implementations.

The GAR backend constructs TaskEnv. Original direct MixEnv users keep their old behavior. No class/module-wide
patch is installed: only an opt-in porridge instance wraps its own existing measurement method. That preserves
its exact 10 ms sampling point, including long actions, without changing any PHYSICS file or physical state.
"""
import copy
from functools import wraps

from .env import MixEnv
from . import scoring as S


class TaskEnv(MixEnv):
    def _build(self):
        super()._build()
        self.spec = S.task_spec(self.config)
        S.capture_start(self)
        if S.enabled(self.config) and self.config.task == "reheat-porridge" and self.info is not None:
            install_hot_recorder(self.world.heat)
        # The physical start snapshot already exists. Attach only judging metadata, without duplicating its
        # potentially large continuum buffers or advancing the world after hand-over.
        self._save_judging(self._initial)

    def _save_judging(self, snap):
        if S.enabled(self.config) and self.info is not None and "task_baseline" in self.info:
            snap["task_judging"] = copy.deepcopy(dict(
                contract=S.contract_for(self.config), baseline=self.info["task_baseline"],
                hot_record=getattr(self.world.heat, "hot_record", None)))
        return snap

    def checkpoint(self):
        return self._save_judging(super().checkpoint())

    def restore(self, snap):
        with self.lock:
            if S.enabled(self.config) and self.info is not None:
                judging = snap.get("task_judging")
                if judging is None or judging["contract"] != S.contract_for(self.config):
                    raise ValueError("Checkpoint lacks the selected task's judging record; nothing restored")
            super().restore(snap)
            if S.enabled(self.config) and self.info is not None:
                self.info["task_baseline"] = copy.deepcopy(judging["baseline"])
                if self.config.task == "reheat-porridge":
                    self.world.heat.task_baseline = copy.deepcopy(judging["baseline"])
                    self.world.heat.hot_record = copy.deepcopy(judging["hot_record"])

    def evaluate(self):
        with self.lock:
            if self.world.cont is None or self.info is None:
                return dict(success=False, score=0.0, contract=S.contract_for(self.config), criteria={},
                            note="no mixture in this build")
            return S.evaluate(self.world, self.info)


def install_hot_recorder(hob):
    """Wrap this instance's post-heat measurement; no change to Hob or World class definitions."""
    if getattr(hob, "_task_recorder_installed", False):
        raise ValueError("Task temperature recorder is already installed")
    original = hob.record_food_peak

    @wraps(original)
    def record(dt=0.0):
        result = original(dt)
        S.record_hot_state(hob)
        return result

    hob.record_food_peak = record
    hob._task_recorder_installed = True

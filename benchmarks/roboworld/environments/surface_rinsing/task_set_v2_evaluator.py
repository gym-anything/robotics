"""surface-rinsing-tasks-v2 evaluator: task_set.evaluator's, plus pre-rinse-plates' spray temperature.

For pre-rinse-plates it tracks the water that leaves the spray valve every tick (volume and volume x mixed temperature,
frozen during the settle branch, kept in checkpoints), adds the spray_temperature predicate and scores with the
temperature-graded progress (task_set_v2.score). It only reads the outlet's flow and temperature: no physics change.
Every other task is judged exactly as under surface-rinsing-tasks-v1; results report surface-rinsing-tasks-v2.
"""
from . import params as P
from .task_set.evaluator import Evaluator as TaskSetEvaluator
from .task_set_v2 import CONTRACT, SPRAY_BAND_C, SPRAY_TASK, score


class Evaluator(TaskSetEvaluator):
    spray_outlet = None

    def __init__(self, env):
        # pre-rinse-plates: water (L) that left the spray valve and its volume x temperature (L C), every tick
        self.spray = dict(volume_L=0.0, heat_L_C=0.0)
        super().__init__(env)
        if self.task == SPRAY_TASK and env.water is not None:
            self.spray_outlet = env.water.outlets["prsv"]          # the pre-rinse station's only outlet

    def _track(self, world):
        if self.frozen or self.env.water is None:
            return
        o = self.spray_outlet
        if o is not None and o.Q > 0.0:
            # the outlet's flow Q (m3/s) and mixed temperature T (C) of the last water step: what it emitted this tick
            v = o.Q*P.DT*1e3
            self.spray["volume_L"] += v
            self.spray["heat_L_C"] += v*o.T
        super()._track(world)

    def snapshot(self):
        out = super().snapshot()
        out["task_set"]["spray"] = dict(self.spray)
        return out

    def restore(self, state):
        super().restore(state)
        self.spray = dict(state["task_set"]["spray"])

    def measure(self):
        m = super().measure()
        if self.task == SPRAY_TASK:
            v = self.spray["volume_L"]
            m["task_set"]["spray"] = dict(volume_L=v, mean_C=self.spray["heat_L_C"]/v if v > 0.0 else None)
        return m

    def criteria(self, m):
        criteria = super().criteria(m)
        if self.task == SPRAY_TASK:
            spray = m["task_set"]["spray"]
            t = spray["mean_C"]
            criteria["spray_temperature"] = dict(
                passed=bool(t is not None and SPRAY_BAND_C[0] <= t <= SPRAY_BAND_C[1]),
                detail=dict(spray, band_C=list(SPRAY_BAND_C), rule="flow-weighted mean temperature of the water that "
                            "left the spray valve since reset"))
        return criteria

    def _score(self, criteria, m):
        # As task_set.evaluator: work quantities graded at the queried live state, final gates after the settle.
        quantities = getattr(self, "_progress_input", None)
        value, reward = score(self.task, self.start, m, criteria, quantities)
        reward["measurement"] = "live work quantities; settled terminal predicates" if quantities is not None else "live"
        return value, reward

    def progress(self):
        out = super().progress()
        out["contract"] = CONTRACT
        if self.env.water is None:
            out["reward"]["contract"] = CONTRACT
        return out

    def evaluate(self, settle_s=None):
        out = super().evaluate(settle_s)
        out["contract"] = CONTRACT
        return out

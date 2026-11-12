"""Task-only adapter over the immutable evaluator/physics modules.

A task's own parts come from its folder (tasks/<task>/verifier.py): its trackers (task_set_trackers), its task-set
measures (task_set_measure) and what it adds to the gates (task_set_criteria); its progress is task_set/scoring.py's.
"""
import copy

from ..layout import task_verifier
from ..task import Evaluator as LegacyEvaluator
from .catalogue import CONTRACT
from .scoring import score


class Evaluator(LegacyEvaluator):
    def __init__(self, env):
        self.glasses = {}
        self.frozen = False
        super().__init__(env)
        self.movable = {}
        w = env.world
        for n, item in w.scene.items.items():
            for key in ("body", "top"):
                if item.get(key) is not None and w._mj_body.get(item[key], -1) >= 0:
                    self.movable[int(w._mj_body[item[key]])] = n
        for n, fixture in w.scene.fixtures.items():
            if fixture["kind"] == "spray_valve":
                for key in ("valve", "lever"):
                    if fixture.get(key) is not None and w._mj_body.get(fixture[key], -1) >= 0:
                        self.movable[int(w._mj_body[fixture[key]])] = n
        trackers = getattr(task_verifier(self.task), "task_set_trackers", None)
        if trackers is not None:                    # rinse-glassware's GlassTrackers (its verifier.py)
            trackers(self, env)
        self.start = copy.deepcopy(self.measure())

    def _track(self, world):
        if self.frozen or self.env.water is None or world.tick % 20:
            return
        for tracker in self.glasses.values():
            tracker.sample(self)

    def snapshot(self):
        return dict(super().snapshot(), task_set=dict(start=copy.deepcopy(self.start),
                    glasses={n: t.snapshot() for n, t in self.glasses.items()}))

    def restore(self, state):
        super().restore(state)
        self.start = copy.deepcopy(state["task_set"]["start"])
        for n, st in state["task_set"]["glasses"].items():
            self.glasses[n].restore(st)

    def held(self):
        w = self.env.world
        out = set()
        for i in range(w.mjd.ncon):
            a, b = map(int, w.mjd.contact[i].geom[:2])
            for pad, other in ((a, b), (b, a)):
                if pad in w._pad_geoms:
                    name = self.movable.get(int(w.mjm.geom_bodyid[other]))
                    if name is not None:
                        out.add(name)
        return sorted(out)

    def _fixtures_closed(self):
        opened, parked, _ = super()._fixtures_closed()
        return opened, parked, bool(self.held())

    def measure(self):
        m = super().measure()
        data = dict(held=self.held())
        own = getattr(task_verifier(self.task), "task_set_measure", None)
        if own is not None:                         # the task's own task-set measures (its verifier.py)
            own(self, data)
        elif self.glasses:
            data["sanitizer"] = {n: t.snapshot() for n, t in self.glasses.items()}
        m["task_set"] = data
        return m

    def criteria(self, m):
        criteria = super().criteria(m)

        def gate(name, ok, detail):
            criteria[name] = dict(passed=bool(ok), detail=detail)

        own = getattr(task_verifier(self.task), "task_set_criteria", None)
        if own is not None:                         # what the task-set contracts add to the task's gates
            own(self, m, gate)
        return criteria

    def _score(self, criteria, m):
        # The hypothetical settle may passively drain films/flush floor water.
        # Grade work quantities at the queried live state, final gates after
        # settling. An untouched reset therefore has exactly zero progress,
        # without keying reward to tick, actions, hand, or an anticipated route.
        quantities = getattr(self, "_progress_input", None)
        value, reward = score(self.task, self.start, m, criteria, quantities)
        reward["measurement"] = "live work quantities; settled terminal predicates" if quantities is not None else "live"
        return value, reward

    def progress(self):
        out = super().progress()
        out["contract"] = CONTRACT
        if self.env.water is None:
            out.update(score=None, verifiable=False, reward=dict(contract=CONTRACT, progress=None,
                       verifiable=False, unverifiable_reason="water state unavailable"))
        if out.get("verifiable") is False:
            out["succeeded"] = False
        return out

    def evaluate(self, settle_s=None):
        # The settle branch cannot earn or erase sanitizer history. The inherited
        # evaluator still measures placement, cleanliness and drainage after it.
        self._progress_input = self.measure()
        self.frozen = True
        try:
            out = super().evaluate(settle_s)
            out["contract"] = CONTRACT
            if out.get("verifiable") is False:
                out["succeeded"] = False
            return out
        finally:
            self.frozen = False
            del self._progress_input

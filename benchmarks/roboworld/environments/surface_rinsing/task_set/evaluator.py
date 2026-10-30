"""Task-only adapter over the immutable evaluator/physics modules."""
import copy
import math
import numpy as np

from .. import params as P
from ..task import Evaluator as LegacyEvaluator, gauges
from .catalogue import CONTRACT, NEW_TASK
from .sanitizer import GlassTracker
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
        if self.task == "rinse-glassware":
            self.glasses = {n: GlassTracker(env, n) for n in self.items if n.startswith("glass_") and n != "glass_rack"}
            # Seed pre-contact carry-in before the first 10 ms sample.
            for tracker in self.glasses.values():
                tracker.sample(self)
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
        ws, w = self.env.water, self.env.world
        data = dict(held=self.held())
        if self.task == NEW_TASK:
            data["waste_rad"] = {n: float(w.joint_value(f["joint"])[0]) for n, f in ws.wastes.items()}
            data["drained_L"] = {n: float(ws.pools.pools[ws.pools.index[n]].drained)*1e3
                                 for n in ("wash", "rinse", "sanitize")}
        elif self.task == "flush-floor":
            ch = ws.static_charts["bay_floor"]
            cells = np.arange(ch.first, ch.first + ch.count)
            a, pos, nrm = ws.surf.area[cells], ws.surf.pos[cells], ws.surf.nrm[cells]
            f = self._fields()
            data["floor_gauges"] = (gauges(pos, a, f["soil"][cells] + f["sus"][cells], nrm)*1e5).tolist()
            _, inv = np.unique(np.floor((pos + 0.005*nrm)/0.01).astype(np.int64), axis=0, return_inverse=True)
            data["floor_areas"] = np.bincount(inv.ravel(), weights=a).tolist()
        elif self.task == "rinse-garbage-can":
            data["can_soap_limit_ug"] = float(ws.surf.area[ws.item_cells["can"]].sum())*P.CRITERIA["surf_per_item"]/0.085*1e9
        elif self.glasses:
            data["sanitizer"] = {n: t.snapshot() for n, t in self.glasses.items()}
        m["task_set"] = data
        return m

    def criteria(self, m):
        criteria = super().criteria(m)

        def gate(name, ok, detail):
            criteria[name] = dict(passed=bool(ok), detail=detail)

        if self.task == "rinse-sponge":
            gate("sponge_on_ledge", self._on_ledge("sponge", compartments=True), m["items"]["sponge"]["position"])
        elif self.task == NEW_TASK:
            gate("drained", m["pools"]["wash"]["volume_L"] <= 0.5, m["pools"]["wash"])
            q = m["task_set"]["waste_rad"]
            gate("waste_closed", all(abs(q[n]) <= math.radians(5) for n in ("wash", "rinse", "sanitize")), q)
            changes = {n: dict(volume_change_L=m["pools"][n]["volume_L"] - self.start["pools"][n]["volume_L"],
                               drained_L=m["task_set"]["drained_L"][n] - self.start["task_set"]["drained_L"][n])
                       for n in ("rinse", "sanitize")}
            gate("others_kept", all(abs(v["volume_change_L"]) <= 1.0 and v["drained_L"] <= 1.0 for v in changes.values()), changes)
            gate("floor_dry", m["contamination"]["floor_water_mL"] <= 50.0, m["contamination"])
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

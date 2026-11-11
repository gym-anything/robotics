"""GAR constructs this task-only adapter for contract `laundry-tasks-v2`; LaundryEnv stays intact.

The released adapter (`task_set_env`) serves every other selection unchanged. Differences
here: every start, authored or the unchanged catalogue lay, settles
before its zero point is taken, and only after it is steady (idle probes on the GPU showed
the catalogue lays still slipping up to 3 s after load, which gave an idle robot credit).
Import this module only when constructing a live backend. Tests stub LaundryEnv.
Authored presets are validated/settled at construction, never during an action.
"""
from collections import deque

import numpy as np

from .garment_task import SemanticGarmentIndex
from .newton15.env import LaundryEnv
from .task_set_v2 import CONTRACT, select, make_evaluator, task_id
from .task_set_v2_presets import start_positions, reach_report

# Settling before the episode (simulated seconds; per 0.5 s block, over the last
# SETTLE_STEADY_BLOCKS blocks: raw-quality change and largest particle displacement).
# Idle probes on o5: the tee's catalogue lay slips in bursts of 10-15 mm up to 3 s after load,
# each stepping its sleeve metric by ~.006; a start taken before a burst left an idle robot
# ~0.01-0.015 of score. After 3 s every lay moved <= ~4 mm per block.
SETTLE_MINIMUM_S = 3.
SETTLE_QUALITY_TOLERANCE = 1.e-3
SETTLE_MOTION_TOLERANCE_M = 5.e-3
SETTLE_STEADY_BLOCKS = 2
SETTLE_MAXIMUM_S = 30.   # fail closed after this; the jeans needed up to 12 s


class CatalogueLaundryEnv(LaundryEnv):
    def __init__(self, config=None, *, task=None, contract=CONTRACT):
        from .newton15.config import Config
        config = config or Config()
        self.entry, self.task_contract = select(config.garment, task, contract)
        super().__init__(config)
        self.start_report = None
        try:
            self._configure_task(config)
        except Exception:
            self.close()
            raise

    def _configure_task(self, config):
        index = SemanticGarmentIndex(self.sim.construction, self.sim.pattern_mesh, self.sim.topology)
        authored = self.entry.start in ("folded_towel", "bunched_ties")
        self.start_report = {}
        if authored:
            x = start_positions(self.entry, index, self.sim.reset_positions, config.table_top)
            self.start_report["authored_reach"] = reach_report(x, table_top=config.table_top,
                                                               robot_base_y=config.robot_base_y)
            # Uses the existing IPC/table guards; rejects crossings instead of hiding them.
            self.sim.author_state(x)
        # author_state stores float32; the reference must use those exact reset positions.
        self._bind_task(index)
        # Every start, authored or the catalogue lay, settles before the episode:
        # the zero point is the garment at rest, so holding still earns nothing.
        self._settle_start(index, maximum_s=SETTLE_MAXIMUM_S)

    def _bind_task(self, index):
        self.task = make_evaluator(self.entry, self.task_contract, index, self.state.particle_q.numpy())
        self.motion = deque()
        self.step_arrays = None
        self._record_motion(reset=True)

    def _settle_start(self, index, *, maximum_s=SETTLE_MAXIMUM_S):
        """Bounded ordinary hold, before the episode/trace. Fail closed on bad starts.

        Accept the first 0.5 s block, at least SETTLE_MINIMUM_S in, where every
        non-goal gate passes (at rest over 0.5 s, intact, released, on the table) and
        the task's raw quality moved at most SETTLE_QUALITY_TOLERANCE and no particle
        moved more than SETTLE_MOTION_TOLERANCE_M in each of the last
        SETTLE_STEADY_BLOCKS blocks.
        """
        stride = max(1, round(.5 / self.config.dt))
        minimum_ticks = round(SETTLE_MINIMUM_S / self.config.dt)
        maximum_ticks = round(maximum_s / self.config.dt)
        last = self.state.particle_q.numpy().copy()
        previous = self.task._quality(last)
        drift, moved = [], []
        while self.tick < maximum_ticks:
            self.act({}, ticks=min(stride, maximum_ticks - self.tick))
            result = super().evaluate()
            gates = result["gates"]
            now = self.state.particle_q.numpy().copy()
            quality = self.task._quality(now)
            drift.append(round(quality - previous, 6))
            moved.append(round(float(np.linalg.norm(now - last, axis=1).max()), 6))
            steady = (len(drift) >= SETTLE_STEADY_BLOCKS
                      and all(abs(d) <= SETTLE_QUALITY_TOLERANCE for d in drift[-SETTLE_STEADY_BLOCKS:])
                      and all(m <= SETTLE_MOTION_TOLERANCE_M for m in moved[-SETTLE_STEADY_BLOCKS:]))
            previous, last = quality, now
            healthy = all(ok for key, ok in gates.items() if not key.startswith("goal:"))
            if not healthy or not steady or self.tick < minimum_ticks:
                continue
            if all(ok for key, ok in gates.items() if key.startswith("goal:")):
                raise RuntimeError("Laundry task start solved itself while settling; refusing this start")
            x = self.state.particle_q.numpy()
            if self.entry.name == "tie_untangle":
                from .garment_task import GARMENT_TASK_SPECS
                old = GARMENT_TASK_SPECS[(self.sim.construction.id, self.entry.name)]
                fractions = [g.evaluate(index, x, np.array([0., 0., 1.])).metrics["end_to_end_reference_fraction"]
                             for g in old.goals]
                if not all(.30 <= f <= .50 for f in fractions):
                    raise RuntimeError(f"Settled ties are not bunched as specified: {fractions}")
            self.start_report.update(settling_ticks=self.tick, settling_seconds=self.tick*self.config.dt,
                                     settling_quality_drift=drift, settling_block_motion_m=moved,
                                     settled_reach=reach_report(x, table_top=self.config.table_top,
                                                               robot_base_y=self.config.robot_base_y))
            # Freeze the accepted episode-start snapshot, not the cloth's physical rest geometry.
            self.sim.initial.assign(self.state)
            self.sim.reset_positions = x.copy()
            self.sim.step_count = self.tick = 0
            self._bind_task(index)
            return
        failed = [key for key, ok in gates.items() if not ok and not key.startswith("goal:")]
        raise RuntimeError(f"Laundry task start did not rest with intact cloth within {maximum_s:g} seconds "
                           f"(failed {failed}; last quality drift {drift[-3:]}; last block motion {moved[-3:]} m)")

    def evaluate(self):
        value = super().evaluate()
        value.update(contract=self.task_contract, task_id=task_id(self.entry),
                     score_rule="1 if success else 0.9 * progress; crossings give zero progress",
                     start_quality=self.task.start_quality,
                     start_settling_s=(self.start_report or {}).get("settling_seconds"))
        return value

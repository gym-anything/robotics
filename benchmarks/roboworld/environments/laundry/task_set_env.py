"""GAR constructs this task-only adapter; the original LaundryEnv stays intact.

Import this module only when constructing a live backend. Tests stub LaundryEnv.
Authored presets are validated/settled at construction, never during an action.
"""
from collections import deque

import numpy as np

from .garment_task import SemanticGarmentIndex
from .newton15.env import LaundryEnv
from .task_set import CONTRACT, select, make_evaluator
from .task_set_presets import start_positions, reach_report


class CatalogueLaundryEnv(LaundryEnv):
    def __init__(self, config=None, *, task=None, contract=None):
        from .newton15.config import Config
        config = config or Config()
        self.entry, self.task_contract = select(config.garment, task, contract)
        super().__init__(config)
        # Preserve the original evaluator instance and all legacy reset paths exactly.
        self.start_report = None
        try:
            self._configure_task(config)
        except Exception:
            self.close()
            raise

    def _configure_task(self, config):
        if self.task_contract == CONTRACT:
            index = SemanticGarmentIndex(self.sim.construction, self.sim.pattern_mesh, self.sim.topology)
            authored = self.entry.start in ("folded_towel", "bunched_ties")
            if authored:
                x = start_positions(self.entry, index, self.sim.reset_positions, config.table_top)
                self.start_report = {"authored_reach": reach_report(x, table_top=config.table_top,
                                                    robot_base_y=config.robot_base_y)}
                # Uses the existing IPC/table guards; rejects crossings instead of hiding them.
                self.sim.author_state(x)
            # author_state stores float32; the reference must use those exact reset positions.
            self._bind_task(index)
            if authored:
                self._settle_start(index)

    def _bind_task(self, index):
        self.task = make_evaluator(self.entry, self.task_contract, index, self.state.particle_q.numpy())
        self.motion = deque()
        self.step_arrays = None
        self._record_motion(reset=True)

    def _settle_start(self, index):
        """Bounded ordinary hold, before the episode/trace. Fail closed on bad starts."""
        stride = max(1, round(.5 / self.config.dt))
        maximum_ticks = round(30. / self.config.dt)
        while self.tick < maximum_ticks:
            self.act({}, ticks=min(stride, maximum_ticks - self.tick))
            result = super().evaluate()
            gates = result["gates"]
            healthy = all(ok for key, ok in gates.items() if not key.startswith("goal:"))
            if not healthy:
                continue
            if all(ok for key, ok in gates.items() if key.startswith("goal:")):
                raise RuntimeError("Authored laundry task solved itself while settling; refusing this start")
            x = self.state.particle_q.numpy()
            if self.entry.name == "tie_untangle":
                from .garment_task import GARMENT_TASK_SPECS
                old = GARMENT_TASK_SPECS[(self.sim.construction.id, self.entry.name)]
                fractions = [g.evaluate(index, x, np.array([0., 0., 1.])).metrics["end_to_end_reference_fraction"]
                             for g in old.goals]
                if not all(.30 <= f <= .50 for f in fractions):
                    raise RuntimeError(f"Settled ties are not bunched as specified: {fractions}")
            self.start_report.update(settling_ticks=self.tick, settling_seconds=self.tick*self.config.dt,
                                     settled_reach=reach_report(x, table_top=self.config.table_top,
                                                               robot_base_y=self.config.robot_base_y))
            # Freeze the accepted episode-start snapshot, not the cloth's physical rest geometry.
            self.sim.initial.assign(self.state)
            self.sim.reset_positions = x.copy()
            self.sim.step_count = self.tick = 0
            self._bind_task(index)
            return
        raise RuntimeError("Authored laundry start did not rest with intact cloth within 30 seconds")

    def evaluate(self):
        value = super().evaluate()
        value.update(contract=self.task_contract, task_id=self.entry.id)
        if self.task_contract == CONTRACT:
            value.update(score_rule="1 if success else 0.9 * progress; crossings give zero progress",
                         start_quality=self.task.start_quality)
        return value

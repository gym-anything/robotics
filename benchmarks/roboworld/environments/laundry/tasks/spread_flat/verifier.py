"""Outcome of spread_flat: the towel unfolded and laid out flat: its plan hull at least 92 % of the flat towel's, its
long hems at least 0.405 m apart everywhere and the whole towel within 25 mm of height (FlatSpreadGoal).
Both task-set contracts add the shared gates (at rest over 0.5 s, released, on the table, stretch within the
garment's limit, no cloth crossings) and score 1 on success, else 0.9 x progress from the start.

goals(contract) gives the spec's goals under laundry-tasks-v1 and -v2 (task_set.task_spec, task_set_v2.task_spec);
verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator bound to the
selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest
evidence) and returns its garment_task.GarmentTaskResult."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from laundry.garment_task import GoalEvaluation, _convex_hull_area, _project
from laundry.task_set_goals import above, below, towel_hems


def goals(contract):
    return (FlatSpreadGoal(),)


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)


@dataclass(frozen=True)
class FlatSpreadGoal:
    id: str = "spread_flat"
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(("terry_body",))
        towel_hems(index, index.topology.physical_positions_m)

    def prepare(self, index, reference):
        return _convex_hull_area(index.topology.physical_positions_m[:, :2])

    def evaluate(self, index, positions_m, table_normal, context):
        p = _project(positions_m, table_normal)
        area = _convex_hull_area(p) / context.prepared
        first, second = towel_hems(index, positions_m)
        separation = float(np.linalg.norm(_project(first - second, table_normal), axis=1).min())
        thickness = float(np.ptp(positions_m @ table_normal))
        metrics = dict(area_fraction=area, minimum_hem_separation_m=separation, normal_extent_m=thickness)
        gates = dict(area=area >= .92, hems=separation >= .405, flat=thickness <= .025)
        quality = np.mean((above(area, .92), above(separation, .405), below(thickness, .025)))
        return GoalEvaluation(self.id, metrics, gates, float(quality), "flat single-sheet towel with separated long hems")

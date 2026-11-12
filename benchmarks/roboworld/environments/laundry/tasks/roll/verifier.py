"""Outcome of roll: the towel rolled up end to end from either short hem: both long hems wound the same way at least 1.5
turns about an axis along the short hems, the roll at most 0.22 m across that axis and the towel's width kept (at least
0.40 m) (RollGoal). laundry-tasks-v2 (the only contract that selects it) adds the shared gates (at rest over 0.5 s,
released, on the table, stretch within 20 %, no cloth crossings) and scores 1 on success, else 0.9 x progress from
the start.

goals(contract) gives the spec's goals under laundry-tasks-v2 (task_set_v2.task_spec), None under any other;
verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator bound to the
selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest
evidence) and returns its garment_task.GarmentTaskResult."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from laundry.garment_asset import PathRef
from laundry.garment_task import GoalEvaluation
from laundry.task_set import V2_CONTRACT
from laundry.task_set_goals import above, below, towel_hems


def goals(contract):
    return (RollGoal(),) if contract == V2_CONTRACT else None


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)


def signed_turning(points):
    """Total signed turning (rad) of a 2-D polyline's tangent: about 2 pi per roll turn,
    pi per U-bend; opposite bends (an accordion) cancel."""
    t = np.diff(points, axis=0)
    t = t[np.linalg.norm(t, axis=1) > 1.e-9]
    if len(t) < 2:
        return 0.
    d = np.diff(np.arctan2(t[:, 1], t[:, 0]))
    return float(np.sum((d + np.pi) % (2 * np.pi) - np.pi))


@dataclass(frozen=True)
class RollGoal:
    """Rolled end to end: both long hems wind the same way at least 1.5 turns about an axis
    along the short hems; the towel's width is kept and the roll is compact across its axis.

    State only: either end, either sense, any place or heading on the table. A half fold
    (pi) or a letter fold (2 pi) winds too little; an accordion cancels; a roll from a long
    edge bends the short hems out of line.
    """
    id: str = "roll"
    minimum_turning_rad: float = 3 * np.pi
    # An authored 2.2-turn roll (115 mm across) left to settle kept two
    # turns but flattened to 0.16-0.18 m across within 6 s; a roll half done is 0.25 m.
    maximum_across_m: float = .22
    minimum_width_m: float = .40
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(("terry_body",))
        towel_hems(index, index.topology.physical_positions_m)

    def prepare(self, index, reference):
        return None

    def evaluate(self, index, positions_m, table_normal, context):
        n = np.asarray(table_normal, dtype=np.float64)
        short = [index.path_samples(PathRef("terry_body", name), positions_m, 17)
                 for name in ("short_bottom_hem", "short_top_hem")]
        spans = [p[-1] - p[0] for p in short]
        spans = [v - (v @ n) * n for v in spans]
        if spans[0] @ spans[1] < 0:
            spans[1] = -spans[1]
        axis = spans[0] + spans[1]
        axis = axis / max(float(np.linalg.norm(axis)), 1.e-12)
        across = np.cross(n, axis)
        width = float(min(abs(v @ axis) for v in spans))
        # Both long hems run from the bottom short hem to the top one.
        turning = [signed_turning(np.stack((p @ across, p @ n), 1))
                   for p in (index.path_samples(ref, positions_m, 41)
                             for ref in (PathRef("terry_body", "long_left_hem", reverse=True),
                                         PathRef("terry_body", "long_right_hem")))]
        same_sense = turning[0] * turning[1] > 0
        wound = float(min(abs(t) for t in turning)) if same_sense else 0.
        extent = float(np.ptp(positions_m @ across))
        # Continuous quantities only; the winding term is zero for an accordion or a flat towel.
        quality = float(np.mean((above(wound, self.minimum_turning_rad), below(extent, self.maximum_across_m),
                                 above(width, self.minimum_width_m))))
        return GoalEvaluation(self.id, dict(long_hem_turning_rad=wound, left_hem_turning_rad=turning[0],
                                            right_hem_turning_rad=turning[1], across_extent_m=extent,
                                            width_m=width),
                              dict(wound=wound >= self.minimum_turning_rad, same_sense=bool(same_sense),
                                   compact=extent <= self.maximum_across_m, width=width >= self.minimum_width_m),
                              quality, "rolled end to end: long hems wound >= 1.5 turns, width kept")

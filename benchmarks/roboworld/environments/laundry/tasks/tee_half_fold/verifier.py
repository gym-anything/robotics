"""Outcome of tee_half_fold (the T-shirt's half_fold): the shirt folded in half lengthwise, one sleeve onto the other
(their panels overlapping in plan within 45 mm RMS, 90th-percentile height gap at most 120 mm), the side seams together
and the two halves of the bottom hem together (each within 25 mm RMS, 55 mm at most), and its plan footprint between
40 % and 62 % of the flat shirt's: folded once, not again (tee_half_fold_goals). Either half may go on top.
laundry-tasks-v2 (the only contract that selects it) adds the shared gates (at rest over 0.5 s, released, on the
table, stretch within 30 %, no cloth crossings) and scores 1 on success, else 0.9 x progress from the start.

goals(contract) gives the spec's goals under laundry-tasks-v2 (task_set_v2.task_spec), None under any other;
verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator bound to the
selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest
evidence) and returns its garment_task.GarmentTaskResult."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from laundry.garment_asset import PathRef
from laundry.garment_task import (GoalEvaluation, PanelCloudAlignmentGoal, PathAlignmentGoal, _convex_hull_area,
                                  _project)
from laundry.task_set import V2_CONTRACT
from laundry.task_set_goals import ContinuousGoal, above, below


def goals(contract):
    return tee_half_fold_goals() if contract == V2_CONTRACT else None


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)


@dataclass(frozen=True)
class HalfFootprintGoal:
    """The garment's plan footprint is about half its flat one: folded once, not again."""
    id: str = "half_footprint"
    panel_ids: tuple = ("front", "back", "left_sleeve", "right_sleeve")
    minimum_fraction: float = .40
    maximum_fraction: float = .62
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(self.panel_ids)

    def prepare(self, index, reference):
        ids = index.panel_vertices(self.panel_ids)
        return ids, _convex_hull_area(index.topology.physical_positions_m[ids, :2])

    def evaluate(self, index, positions_m, table_normal, context):
        ids, flat_area = context.prepared
        area = _convex_hull_area(_project(positions_m[ids], table_normal)) / flat_area
        return GoalEvaluation(self.id, dict(projected_area_fraction=area),
                              dict(folded=area <= self.maximum_fraction, not_refolded=area >= self.minimum_fraction),
                              float(np.mean((below(area, self.maximum_fraction), above(area, self.minimum_fraction)))),
                              "plan footprint about half the flat garment's")


def tee_half_fold_goals():
    """Lengthwise half fold of the tee: one sleeve on the other, side seams and the two
    halves of the bottom hem together. Symmetric: either half may go on top.

    The sleeve gap bound is the tee's, not the jeans' 30 mm: the settled tee's yoke and
    sewn sleeves stand up to 90 mm, and an authored half fold left to settle
    kept its fold with sleeve gaps of 85-94 mm at the 90th percentile."""
    return (
        ContinuousGoal(PanelCloudAlignmentGoal("sleeve_overlap", ("left_sleeve",), ("right_sleeve",),
                                               maximum_symmetric_projected_rms_m=.045, maximum_normal_gap_m=.120)),
        ContinuousGoal(PathAlignmentGoal("side_seam_alignment", PathRef("front", "right_side"),
                                         PathRef("front", "left_side", reverse=True),
                                         rms_threshold_m=.025, maximum_threshold_m=.055)),
        ContinuousGoal(PathAlignmentGoal("hem_alignment", PathRef("front", "bottom_hem", 0., .5),
                                         PathRef("front", "bottom_hem", .5, 1., reverse=True),
                                         rms_threshold_m=.025, maximum_threshold_m=.055)),
        HalfFootprintGoal(),
    )

"""State-only eight-task objectives. No runtime imports, forces, or pose writes.

Success predicates are separate from continuous geometric credit. Untouched
credit is subtracted once, when binding the task to its actual episode start.
The objectives written for one benchmark task (spread_flat, turn_face_down,
thirds_fold) are in its verifier.py (tasks/<task>/); the shared pieces are here.
"""
from dataclasses import dataclass, replace

import numpy as np

from .garment_asset import PathRef
from .garment_task import (
    GarmentTaskEvaluator, PathAlignmentGoal, PanelContainmentGoal,
    PanelStackGoal, PanelCloudAlignmentGoal, ProjectedAreaFractionGoal, PathExtensionGoal,
    _convex_hull, _project,
)


def below(value, threshold):
    """Continuous quality, one throughout the accepted interval, approaching zero."""
    return float(1 / (1 + max(float(value) - threshold, 0.) / max(threshold, 1.e-9)))


def above(value, threshold):
    return float(np.clip(value / threshold, 0., 1.))


def outside_distance(points, hull, margin=.015):
    """Continuous signed half-plane violation; a degenerate host gives no credit."""
    if len(hull) < 3:
        return np.full(len(points), 1.)
    edge = np.roll(hull, -1, axis=0) - hull
    delta = points[:, None] - hull
    signed = (edge[None, :, 0] * delta[:, :, 1] - edge[None, :, 1] * delta[:, :, 0])
    signed /= np.maximum(np.linalg.norm(edge, axis=1), 1.e-12)
    return np.maximum(-signed.min(axis=1) - margin, 0.)


@dataclass(frozen=True)
class ContinuousGoal:
    """Keep an existing goal's predicates; replace its score by metric credit."""
    original: object

    @property
    def id(self):
        return self.original.id

    @property
    def weight(self):
        return self.original.weight

    def validate(self, index):
        self.original.validate(index)

    def evaluate(self, index, positions_m, table_normal):
        g = self.original
        result = g.evaluate(index, positions_m, table_normal)
        m = result.metrics
        if isinstance(g, PathAlignmentGoal):
            q = (below(m["rms_m"], g.rms_threshold_m), below(m["maximum_m"], g.maximum_threshold_m))
        elif isinstance(g, PanelStackGoal):
            q = (below(m["projected_nearest_rms_m"], g.maximum_projected_nearest_rms_m),
                 below(m["normal_gap_quantile_m"], g.maximum_normal_gap_m))
        elif isinstance(g, PanelCloudAlignmentGoal):
            q = (below(m["symmetric_projected_rms_m"], g.maximum_symmetric_projected_rms_m),
                 below(m["normal_gap_quantile_m"], g.maximum_normal_gap_m))
        elif isinstance(g, PanelContainmentGoal):
            host = index.panel_vertices(g.host_panel_ids)
            moving = np.setdiff1d(index.panel_vertices(g.moving_panel_ids), host)
            hull = _convex_hull(_project(positions_m[host], table_normal))
            d = outside_distance(_project(positions_m[moving], table_normal), hull, g.hull_margin_m)
            # Distance changes before a particle crosses the hull, unlike the inside count.
            q = (float(1 / (1 + np.mean(d) / .045)),)
            result = replace(result, metrics={**m, "mean_outside_distance_m": float(np.mean(d))})
        elif isinstance(g, ProjectedAreaFractionGoal):
            q = (below(m["projected_area_fraction"], g.maximum_fraction),)
        elif isinstance(g, PathExtensionGoal):
            q = (above(m["end_to_end_reference_fraction"], g.minimum_reference_fraction),)
        else:
            raise TypeError(f"No continuous metric credit for {type(g).__name__}")
        return replace(result, score=float(np.mean(q)))


class StartRelativeEvaluator(GarmentTaskEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shared = {}
        starts = [self._goal_evaluation(g, self.reference_positions_m, np.array([0., 0., 1.]), None, shared)
                  for g in self.spec.goals]
        if all(all(result.gates.values()) for result in starts):
            raise ValueError("Eight-task start already meets its objectives; refusing a trivial episode")
        self.start_quality = self._quality(self.reference_positions_m)
        if self.start_quality >= 1 - 1.e-9:
            raise ValueError("Eight-task start already meets its objectives; refusing a trivial episode")

    def _quality(self, x):
        shared = {}
        return sum(g.weight * self._goal_evaluation(g, x, np.array([0., 0., 1.]), None, shared).score
                   for g in self.spec.goals) / sum(g.weight for g in self.spec.goals)

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        progress = float(np.clip((result.progress - self.start_quality) / (1 - self.start_quality), 0., 1.))
        if result.metrics.nonlocal_intersection_count:
            progress = 0.
        return replace(result, progress=progress,
                       potential=1. if result.success else self.spec.partial_credit_cap * progress)


def towel_hems(index, x):
    return (index.path_samples(PathRef("terry_body", "long_left_hem"), x, 17),
            index.path_samples(PathRef("terry_body", "long_right_hem", reverse=True), x, 17))

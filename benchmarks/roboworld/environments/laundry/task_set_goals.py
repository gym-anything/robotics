"""State-only eight-task objectives. No runtime imports, forces, or pose writes.

Success predicates are separate from continuous geometric credit. Untouched
credit is subtracted once, when binding the task to its actual episode start.
"""
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from .garment_asset import PathRef
from .garment_task import (
    GarmentTaskEvaluator, GoalEvaluation, PathAlignmentGoal, PanelContainmentGoal,
    PanelStackGoal, PanelCloudAlignmentGoal, ProjectedAreaFractionGoal, PathExtensionGoal,
    _convex_hull, _convex_hull_area, _inside_convex_hull, _project,
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


@dataclass(frozen=True)
class FaceDownGoal:
    id: str = "face_down"
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(("front", "back"))

    def prepare(self, index, reference):
        front, back = (index.panel_vertices((p,)) for p in ("front", "back"))
        return (np.setdiff1d(front, back), np.setdiff1d(back, front),
                _convex_hull_area(reference[np.union1d(front, back), :2]))

    def evaluate(self, index, positions_m, table_normal, context):
        front, back, reference_area = context.prepared
        fp, bp = positions_m[front], positions_m[back]
        squared = np.sum((_project(fp, table_normal)[:, None] - _project(bp, table_normal)[None]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        gap = (bp[nearest] - fp) @ table_normal
        # A nearby backing surface is necessary: an unrelated raised corner is not a flipped shirt.
        plan_distance = np.sqrt(squared[np.arange(len(front)), nearest])
        fraction = float(np.mean((gap > .0001) & (plan_distance <= .045)))
        area = _convex_hull_area(_project(np.concatenate((fp, bp)), table_normal)) / reference_area
        thickness = float(np.ptp(np.concatenate((fp, bp)) @ table_normal))
        flip = float(np.mean(np.clip((gap + .002) / .004, 0., 1.) / (1 + plan_distance / .045)))
        layout = above(area, .85) * below(thickness, .030)
        # Layout initially passes, so it modulates flip credit instead of awarding idle credit.
        quality = flip * (.5 + .5 * layout)
        return GoalEvaluation(self.id, dict(flipped_fraction=fraction, body_area_fraction=area,
                              body_normal_extent_m=thickness, flip_quality=flip),
                              dict(face_down=fraction >= .90, laid_out=area >= .85, flat=thickness <= .030),
                              quality, "front below the nearby back panel, body laid out flat")


def column_surfaces(points, triangles):
    """Barycentric heights and |normal.z| in columns; NaN means no triangle hit.

    Shared edges may produce duplicate hits; callers merge equal heights.
    Uses actual cloth triangles, never a vertex-count proxy for plies.
    """
    a, b, c = (triangles[:, k] for k in range(3))
    u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    d = points[:, None] - a[None, :, :2]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    safe = np.where(abs(det) > 1.e-12, det, 1.)
    beta = (d[:, :, 0] * v[:, 1] - d[:, :, 1] * v[:, 0]) / safe
    gamma = (u[:, 0] * d[:, :, 1] - u[:, 1] * d[:, :, 0]) / safe
    alpha = 1 - beta - gamma
    hit = (abs(det) > 1.e-12) & (alpha >= -1.e-8) & (beta >= -1.e-8) & (gamma >= -1.e-8)
    height = alpha * a[:, 2] + beta * b[:, 2] + gamma * c[:, 2]
    normal = np.cross(b - a, c - a)
    horizontal = abs(normal[:, 2]) / np.maximum(np.linalg.norm(normal, axis=1), 1.e-12)
    return np.where(hit, height, np.nan), horizontal


def three_ply_fraction(positions, triangles):
    """Area quadrature over the current hull: three separate, nearly horizontal plies.

    A roll fails the horizontal-sheet condition; a ball also fails extent/height checks.
    Grid axes follow the live towel's longest principal direction (rotation invariant).
    """
    xy = positions[:, :2]
    _, axes = np.linalg.eigh(np.cov(xy.T))
    local = positions.copy()
    local[:, :2] = xy @ axes
    hull = _convex_hull(local[:, :2])
    if len(hull) < 3:
        return 0.
    lo, hi = local[:, :2].min(0), local[:, :2].max(0)
    # Invalid exploded cloth must not allocate an unbounded quadrature grid.
    # A 0.70 m towel cannot span 2 m under the existing 20% stretch limit.
    if np.max(hi - lo) > 2.:
        return 0.
    count = np.maximum(2, np.ceil((hi - lo) / .010).astype(int))
    grid = [lo[i] + (np.arange(count[i]) + .5) * (hi[i] - lo[i]) / count[i] for i in (0, 1)]
    points = np.array(np.meshgrid(*grid)).reshape(2, -1).T
    points = points[_inside_convex_hull(points, hull, margin_m=0.)]
    if not len(points):
        return 0.
    good = 0
    for batch in np.array_split(points, max(1, int(np.ceil(len(points) / 256)))):
        heights, horizontal = column_surfaces(batch, local[triangles])
        for row in heights:
            order = np.argsort(row)
            order = order[np.isfinite(row[order])]
            levels = []
            flats = []
            for k in order:
                if not levels or row[k] - levels[-1] > .002:
                    levels.append(row[k])
                    flats.append(horizontal[k])
                else:
                    flats[-1] = max(flats[-1], horizontal[k])
            good += len(levels) == 3 and min(flats) >= .90 and levels[-1] - levels[0] <= .025
    return float(good / len(points))


@dataclass(frozen=True)
class ThirdsFoldGoal:
    id: str = "thirds_fold"
    weight: float = 1.
    uses_context: ClassVar[bool] = True

    def validate(self, index):
        index.panel_vertices(("terry_body",))

    def prepare(self, index, reference):
        # Material bands identify what is folded, not a world-space target or action sequence.
        flat = index.topology.physical_positions_m
        across = flat[:, 0]
        width = np.ptp(across)
        center = (across >= across.min() + width / 3 - 1.e-8) & (across <= across.min() + 2 * width / 3 + 1.e-8)
        return center, _convex_hull_area(flat[:, :2])

    def evaluate(self, index, positions_m, table_normal, context):
        center, flat_area = context.prepared
        plan = _project(positions_m, table_normal)
        hull = _convex_hull(plan[center])
        hems = np.concatenate(towel_hems(index, positions_m))
        outside = outside_distance(_project(hems, table_normal), hull, .015)
        area = _convex_hull_area(plan) / flat_area
        short = [index.path_samples(PathRef("terry_body", name), positions_m, 17)
                 for name in ("short_bottom_hem", "short_top_hem")]
        extent = max(float(np.linalg.norm(_project(p, table_normal)[:, None] - _project(p, table_normal)[None], axis=2).max())
                     for p in short)
        length = float(np.linalg.norm(_project(np.array([short[0].mean(0) - short[1].mean(0)]), table_normal)))
        thickness = float(np.ptp(positions_m @ table_normal))
        plies = three_ply_fraction(positions_m, index.topology.physical_triangles)
        # Continuous quantities only: ply counts are success predicates, not reward steps.
        quality = float(np.mean((below(area, .40), 1 / (1 + outside.mean() / .045),
                                 below(extent, .20), below(thickness, .025), above(length, .63))))
        return GoalEvaluation(self.id, dict(area_fraction=area, maximum_hem_outside_m=float(outside.max()),
                              short_hem_extent_m=extent, length_m=length, normal_extent_m=thickness,
                              three_ply_fraction=plies),
                              dict(area=area <= .40, hems_inside=bool(np.all(outside <= 1.e-8)),
                                   short_hems=extent <= .20, length=length >= .63,
                                   flat=thickness <= .025, three_plies=plies >= .80),
                              quality, "three flat plies, long hems in the middle-third footprint, length retained")

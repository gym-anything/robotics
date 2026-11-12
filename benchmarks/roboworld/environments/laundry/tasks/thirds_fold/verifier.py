"""Outcome of thirds_fold: the towel folded in thirds lengthwise (a letter fold, C or Z): three flat plies over at
least 80 % of its plan hull, both long hems within 15 mm of the plan hull of the towel's middle third, the footprint
at most 40 % of the flat towel's, each short hem folded to a plan extent of at most 0.20 m, the length kept (at least
0.63 m) and the towel within 25 mm of height. laundry-tasks-v2 counts plies as cloth crossings (ThirdsFoldGoal,
three_ply_fraction); laundry-tasks-v1 merged heights within 2 mm (ThirdsFoldGoalV1, three_ply_fraction_v1).
Both task-set contracts add the shared gates (at rest over 0.5 s, released, on the table, stretch within the
garment's limit, no cloth crossings) and score 1 on success, else 0.9 x progress from the start.

goals(contract) gives the spec's goals under laundry-tasks-v1 and -v2 (task_set.task_spec, task_set_v2.task_spec);
verify(task, positions_m, velocities_m_s, **evidence) judges an episode state with the evaluator bound to the
selected contract's spec (newton15/env.LaundryEnv.evaluate passes the strain, crossing, release, table and rest
evidence) and returns its garment_task.GarmentTaskResult."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from laundry.garment_asset import PathRef
from laundry.garment_task import GoalEvaluation, _convex_hull, _convex_hull_area, _inside_convex_hull, _project
from laundry.task_set import V2_CONTRACT
from laundry.task_set_goals import above, below, outside_distance, towel_hems


def goals(contract):
    return (ThirdsFoldGoal(),) if contract == V2_CONTRACT else (ThirdsFoldGoalV1(),)


def verify(task, positions_m, velocities_m_s, **evidence):
    return task.evaluate(positions_m, velocities_m_s, **evidence)


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


def sheet_crossings(row, horizontal, triangles, same_point_m=1.e-6):
    """The cloth sheets one column passes through, lowest first: (heights, |normal.z|).

    Every triangle hit is one crossing of the cloth. Hits at one height on triangles
    that share a vertex are the column passing through a shared edge or vertex of one
    sheet, and count once. No height tolerance merges anything else: at a fold the
    40 mm mesh joins two plies through one tilted cell, which is a crossing of its own
    and must not swallow the ply it meets (a merge tolerance did, failing correct folds).
    """
    hit = np.flatnonzero(np.isfinite(row))
    hit = hit[np.argsort(row[hit], kind="stable")]
    heights, flats, members = [], [], []
    for k in hit:
        corners = set(triangles[k].tolist())
        if heights and row[k] - heights[-1] <= same_point_m and corners & members[-1]:
            flats[-1] = max(flats[-1], horizontal[k])
            members[-1] |= corners
            continue
        heights.append(row[k])
        flats.append(horizontal[k])
        members.append(corners)
    return np.asarray(heights), np.asarray(flats)


def three_ply_fraction(positions, triangles):
    """Area quadrature over the current hull: three separate, nearly horizontal plies.

    A column counts when it crosses the cloth exactly three times, every crossing
    nearly horizontal, within 25 mm. A roll fails the horizontal-sheet condition; a
    ball also fails extent/height checks; a half fold has two and a quarter fold four.
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
            levels, flats = sheet_crossings(row, horizontal, triangles)
            good += len(levels) == 3 and flats.min() >= .90 and levels[-1] - levels[0] <= .025
    return float(good / len(points))


@dataclass(frozen=True)
class ThirdsFoldGoal:
    """As ThirdsFoldGoalV1, with the crossing-counting ply check (three_ply_fraction)."""
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


def three_ply_fraction_v1(positions, triangles):
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
class ThirdsFoldGoalV1:
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
        plies = three_ply_fraction_v1(positions_m, index.topology.physical_triangles)
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

"""State-only objectives of contract `laundry-tasks-v2`. No runtime imports, forces, or pose writes.

Shared pieces are imported unchanged from `task_set_goals` (the `laundry-tasks-v1` objectives,
which stay as released). New here: the thirds fold counts cloth
crossings instead of merging heights within 2 mm, and the swapped-in `towel/roll` and
`tee/half_fold` objectives.
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .garment_asset import PathRef
from .garment_task import (
    GoalEvaluation, PathAlignmentGoal, PanelCloudAlignmentGoal,
    _convex_hull, _convex_hull_area, _inside_convex_hull, _project,
)
from .task_set_goals import (  # noqa: F401  (re-exported for task_set_v2)
    ContinuousGoal, FaceDownGoal, FlatSpreadGoal, StartRelativeEvaluator,
    above, below, column_surfaces, outside_distance, towel_hems,
)


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
    """As `task_set_goals.ThirdsFoldGoal`, with this module's crossing-counting ply check."""
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

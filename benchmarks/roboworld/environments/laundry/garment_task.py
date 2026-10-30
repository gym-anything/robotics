"""Semantic, engine-independent objectives for sewn-garment manipulation.

The original towel task assumes one rectangular grid and hard-coded row/column
indices.  That representation cannot describe whether *both* sleeves are
inside a shirt body, whether two jeans legs overlap, or whether a waist tie is
actually extended.  This module binds task landmarks to authored panels and
pattern paths, then evaluates only particle state returned by the simulator.

No objective writes a particle pose or applies a force.  Thresholds are
explicit engineering priors until a real-fold dataset is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import ClassVar, Mapping, Protocol

import numpy as np

from .garment_assembly import SewnGarmentTopology
from .garment_asset import GarmentConstruction, PathRef
from .garment_meshing import GarmentPatternMesh


_EPSILON = 1.0e-12


def _unit_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        raise ValueError(f"{name} cannot be zero")
    return vector / norm


def _table_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.90:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second


def _project(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    first, second = _table_basis(normal)
    return np.stack((points @ first, points @ second), axis=1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if unique.ndim != 2 or unique.shape[1] != 2 or not np.isfinite(unique).all():
        raise ValueError("A projected panel hull needs finite 2D coordinates")
    # A vertical or edge-on garment has a zero-area tabletop projection. It is
    # valid physical state, not a failed action or a reason to fault the server.
    if len(unique) < 3 or np.linalg.matrix_rank(unique-unique.mean(0), tol=1.e-12) < 2:
        return np.empty((0, 2), dtype=np.float64)
    try:
        from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415
    except ImportError as exception:  # pragma: no cover - bootstrap dependency
        raise RuntimeError(
            "Semantic garment footprints require SciPy; run laundry/bootstrap_newton.ps1"
        ) from exception
    try:
        hull = ConvexHull(unique)
    except QhullError as exception:
        raise ValueError("A projected panel hull is collinear") from exception
    # SciPy returns counter-clockwise vertex order in two dimensions and is
    # robust to the sub-femtometre corner perturbations created by seam welding.
    return unique[hull.vertices]


def _convex_hull_area(points: np.ndarray) -> float:
    hull = _convex_hull(points)
    following = np.roll(hull, -1, axis=0)
    return 0.5 * abs(
        float(np.sum(hull[:, 0] * following[:, 1] - following[:, 0] * hull[:, 1]))
    )


def _inside_convex_hull(
    points: np.ndarray,
    hull: np.ndarray,
    *,
    margin_m: float,
) -> np.ndarray:
    if len(hull) < 3:
        return np.zeros(len(points), dtype=bool)
    inside = np.ones(len(points), dtype=bool)
    following = np.roll(hull, -1, axis=0)
    for first, second in zip(hull, following, strict=True):
        edge = second - first
        signed_double_area = edge[0] * (points[:, 1] - first[1]) - edge[1] * (
            points[:, 0] - first[0]
        )
        inside &= signed_double_area >= -margin_m * float(np.linalg.norm(edge))
    return inside


def _segment_distances(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    edge = second - first
    t = np.clip(
        np.sum((points - first) * edge, axis=-1) / np.maximum(np.sum(edge * edge, axis=-1), _EPSILON),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - (first + t[..., None] * edge), axis=-1)


def _lumped_material_area(topology, triangle_mask: np.ndarray) -> np.ndarray:
    """Rest (material) area per physical vertex, one third of each selected triangle."""
    material = topology.panel_material_coordinates_m[topology.panel_triangles[triangle_mask]]
    first, second = material[:, 1] - material[:, 0], material[:, 2] - material[:, 0]
    area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    lumped = np.zeros(topology.physical_vertex_count, dtype=np.float64)
    np.add.at(lumped, topology.physical_triangles[triangle_mask].reshape(-1), np.repeat(area / 3.0, 3))
    return lumped


@dataclass(frozen=True)
class TabletopFrame:
    """Axis-aligned rectangular tabletop (world metres, +z up) used by on-table checks."""

    top_z_m: float
    half_extent_x_m: float
    half_extent_y_m: float
    center_xy_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = (self.top_z_m, self.half_extent_x_m, self.half_extent_y_m, *self.center_xy_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Tabletop geometry must be finite")
        if min(self.half_extent_x_m, self.half_extent_y_m) <= 0.0:
            raise ValueError("Tabletop half extents must be positive")

    def supports(
        self, positions_m: np.ndarray, below_tolerance_m: float, above_limit_m: float = math.inf
    ) -> np.ndarray:
        """Particles over the tabletop rectangle, not dropped below its surface and not held
        higher above it than `above_limit_m`."""
        return (
            (np.abs(positions_m[:, 0] - self.center_xy_m[0]) <= self.half_extent_x_m)
            & (np.abs(positions_m[:, 1] - self.center_xy_m[1]) <= self.half_extent_y_m)
            & (positions_m[:, 2] >= self.top_z_m - below_tolerance_m)
            & (positions_m[:, 2] <= self.top_z_m + above_limit_m)
        )


@dataclass(frozen=True)
class GoalContext:
    """Workcell facts a context-aware goal may read; never written by a goal. `shared` is a
    scratch dict for one evaluation call, so goals of the same task can reuse a computation."""

    tabletop: TabletopFrame | None
    reference_positions_m: np.ndarray
    prepared: object
    shared: dict | None = None


def _score_below(value: float, threshold: float) -> float:
    return float(np.clip(1.0 - 0.5 * value / max(threshold, _EPSILON), 0.0, 1.0))


def _score_above(value: float, threshold: float) -> float:
    return float(np.clip(value / max(threshold, _EPSILON), 0.0, 1.0))


@dataclass(frozen=True)
class SemanticGarmentIndex:
    """Map construction-level panel/path semantics to welded particles."""

    construction: GarmentConstruction
    pattern_mesh: GarmentPatternMesh
    topology: SewnGarmentTopology

    def __post_init__(self) -> None:
        ids = {
            self.construction.id,
            self.pattern_mesh.construction_id,
            self.topology.construction_id,
        }
        if len(ids) != 1:
            raise ValueError("Construction, pattern mesh, and topology identifiers differ")
        if self.pattern_mesh.vertex_count != self.topology.material_vertex_count:
            raise ValueError("Pattern and welded material vertex counts differ")

    @property
    def particle_count(self) -> int:
        return self.topology.physical_vertex_count

    def panel_vertices(self, panel_ids: tuple[str, ...]) -> np.ndarray:
        if not panel_ids or len(set(panel_ids)) != len(panel_ids):
            raise ValueError("A panel group must contain unique panel identifiers")
        material: list[np.ndarray] = []
        for panel_id in panel_ids:
            if panel_id not in self.construction.panel_map:
                raise KeyError(f"Unknown panel {panel_id!r}")
            start, stop = self.pattern_mesh.panel_vertex_ranges[panel_id]
            material.append(np.arange(start, stop, dtype=np.int32))
        physical = self.topology.material_to_physical_vertex[np.concatenate(material)]
        return np.unique(physical).astype(np.int32, copy=False)

    def path_samples(
        self,
        path_ref: PathRef,
        positions_m: np.ndarray,
        sample_count: int,
    ) -> np.ndarray:
        if sample_count < 2:
            raise ValueError("A path landmark requires at least two samples")
        if path_ref.panel_id not in self.construction.panel_map:
            raise KeyError(f"Unknown panel {path_ref.panel_id!r}")
        key = (path_ref.panel_id, path_ref.path_id)
        try:
            material_indices = self.pattern_mesh.path_vertices[key]
            authored_fractions = self.pattern_mesh.path_fractions[key]
        except KeyError as exception:
            raise KeyError(f"Pattern mesh has no path landmark {key!r}") from exception
        physical_indices = self.topology.material_to_physical_vertex[material_indices]
        samples = np.linspace(
            path_ref.start_fraction,
            path_ref.end_fraction,
            sample_count,
            dtype=np.float64,
        )
        result = np.empty((sample_count, 3), dtype=np.float64)
        for output_index, fraction in enumerate(samples):
            upper = int(np.searchsorted(authored_fractions, fraction, side="right"))
            upper = min(max(upper, 1), len(authored_fractions) - 1)
            lower = upper - 1
            span = float(authored_fractions[upper] - authored_fractions[lower])
            alpha = float((fraction - authored_fractions[lower]) / max(span, _EPSILON))
            result[output_index] = (
                (1.0 - alpha) * positions_m[int(physical_indices[lower])]
                + alpha * positions_m[int(physical_indices[upper])]
            )
        return result[::-1].copy() if path_ref.reverse else result


@dataclass(frozen=True)
class GoalEvaluation:
    goal_id: str
    metrics: Mapping[str, float]
    gates: Mapping[str, bool]
    score: float
    note: str

    def __post_init__(self) -> None:
        if not self.goal_id or not self.note:
            raise ValueError("Goal evaluations require an identifier and note")
        if not 0.0 <= self.score <= 1.0 or not math.isfinite(self.score):
            raise ValueError("Goal scores must be finite and lie in [0, 1]")
        if not self.metrics or not self.gates:
            raise ValueError("Goal evaluations require metrics and gates")


class GarmentGoal(Protocol):
    id: str
    weight: float

    def validate(self, index: SemanticGarmentIndex) -> None: ...

    def evaluate(
        self,
        index: SemanticGarmentIndex,
        positions_m: np.ndarray,
        table_normal: np.ndarray,
    ) -> GoalEvaluation: ...


@dataclass(frozen=True)
class PathAlignmentGoal:
    id: str
    first: PathRef
    second: PathRef
    rms_threshold_m: float
    maximum_threshold_m: float
    sample_count: int = 17
    tangent_only: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(self.rms_threshold_m, self.maximum_threshold_m, self.weight) <= 0.0:
            raise ValueError("Path-alignment id, thresholds, and weight must be positive")
        if self.sample_count < 3 or self.maximum_threshold_m < self.rms_threshold_m:
            raise ValueError("Path-alignment sampling or thresholds are invalid")

    def validate(self, index: SemanticGarmentIndex) -> None:
        for reference in (self.first, self.second):
            panel = index.construction.panel_map[reference.panel_id]
            panel.path(reference.path_id)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        first = index.path_samples(self.first, positions_m, self.sample_count)
        second = index.path_samples(self.second, positions_m, self.sample_count)
        delta = first - second
        if self.tangent_only:
            delta = delta - (delta @ table_normal)[:, None] * table_normal[None, :]
        distance = np.linalg.norm(delta, axis=1)
        rms = float(np.sqrt(np.mean(distance * distance)))
        maximum = float(np.max(distance))
        gates = MappingProxyType(
            {
                "rms": rms <= self.rms_threshold_m,
                "maximum": maximum <= self.maximum_threshold_m,
            }
        )
        score = 0.5 * (
            _score_below(rms, self.rms_threshold_m)
            + _score_below(maximum, self.maximum_threshold_m)
        )
        return GoalEvaluation(
            self.id,
            MappingProxyType({"rms_m": rms, "maximum_m": maximum}),
            gates,
            score,
            "arclength-corresponded landmark separation",
        )


@dataclass(frozen=True)
class PanelContainmentGoal:
    id: str
    moving_panel_ids: tuple[str, ...]
    host_panel_ids: tuple[str, ...]
    required_fraction: float = 0.90
    hull_margin_m: float = 0.015
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(self.weight, self.required_fraction) <= 0.0:
            raise ValueError("Panel-containment id, fraction, and weight must be positive")
        if self.required_fraction > 1.0 or self.hull_margin_m < 0.0:
            raise ValueError("Panel-containment fraction/margin are invalid")

    def validate(self, index: SemanticGarmentIndex) -> None:
        index.panel_vertices(self.moving_panel_ids)
        index.panel_vertices(self.host_panel_ids)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        moving = index.panel_vertices(self.moving_panel_ids)
        host = index.panel_vertices(self.host_panel_ids)
        # Shared seam particles are not evidence that a sleeve has been tucked.
        moving = np.setdiff1d(moving, host, assume_unique=True)
        if len(moving) == 0:
            raise ValueError(f"Containment goal {self.id!r} has no independent particles")
        projected_host = _project(positions_m[host], table_normal)
        hull = _convex_hull(projected_host)
        inside = _inside_convex_hull(
            _project(positions_m[moving], table_normal),
            hull,
            margin_m=self.hull_margin_m,
        )
        fraction = float(np.mean(inside))
        return GoalEvaluation(
            self.id,
            MappingProxyType({"contained_fraction": fraction}),
            MappingProxyType({"contained_fraction": fraction >= self.required_fraction}),
            _score_above(fraction, self.required_fraction),
            "fraction of independent moving-panel particles inside the live host hull",
        )


@dataclass(frozen=True)
class PanelStackGoal:
    id: str
    moving_panel_ids: tuple[str, ...]
    host_panel_ids: tuple[str, ...]
    normal_gap_quantile: float = 0.90
    maximum_normal_gap_m: float = 0.025
    maximum_projected_nearest_rms_m: float = 0.045
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(
            self.normal_gap_quantile,
            self.maximum_normal_gap_m,
            self.maximum_projected_nearest_rms_m,
            self.weight,
        ) <= 0.0:
            raise ValueError("Panel-stack parameters must be positive")
        if self.normal_gap_quantile > 1.0:
            raise ValueError("Panel-stack quantile cannot exceed one")

    def validate(self, index: SemanticGarmentIndex) -> None:
        index.panel_vertices(self.moving_panel_ids)
        index.panel_vertices(self.host_panel_ids)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        moving = index.panel_vertices(self.moving_panel_ids)
        host = index.panel_vertices(self.host_panel_ids)
        moving = np.setdiff1d(moving, host, assume_unique=True)
        if len(moving) == 0:
            raise ValueError(f"Stack goal {self.id!r} has no independent particles")
        moving_xy = _project(positions_m[moving], table_normal)
        host_xy = _project(positions_m[host], table_normal)
        squared = np.sum((moving_xy[:, None, :] - host_xy[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        projected_distance = np.sqrt(squared[np.arange(len(moving)), nearest])
        normal_gap = np.abs(
            (positions_m[moving] - positions_m[host[nearest]]) @ table_normal
        )
        rms = float(np.sqrt(np.mean(projected_distance * projected_distance)))
        quantile_gap = float(np.quantile(normal_gap, self.normal_gap_quantile))
        gates = MappingProxyType(
            {
                "projected_nearest_rms": rms <= self.maximum_projected_nearest_rms_m,
                "normal_gap_quantile": quantile_gap <= self.maximum_normal_gap_m,
            }
        )
        score = 0.5 * (
            _score_below(rms, self.maximum_projected_nearest_rms_m)
            + _score_below(quantile_gap, self.maximum_normal_gap_m)
        )
        return GoalEvaluation(
            self.id,
            MappingProxyType(
                {
                    "projected_nearest_rms_m": rms,
                    "normal_gap_quantile_m": quantile_gap,
                }
            ),
            gates,
            score,
            "moving panels must form a nearby layer, not float above the host",
        )


@dataclass(frozen=True)
class PanelCloudAlignmentGoal:
    id: str
    first_panel_ids: tuple[str, ...]
    second_panel_ids: tuple[str, ...]
    maximum_symmetric_projected_rms_m: float
    maximum_normal_gap_m: float
    normal_gap_quantile: float = 0.90
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(
            self.maximum_symmetric_projected_rms_m,
            self.maximum_normal_gap_m,
            self.normal_gap_quantile,
            self.weight,
        ) <= 0.0:
            raise ValueError("Panel-cloud alignment parameters must be positive")
        if self.normal_gap_quantile > 1.0:
            raise ValueError("Panel-cloud quantile cannot exceed one")

    def validate(self, index: SemanticGarmentIndex) -> None:
        index.panel_vertices(self.first_panel_ids)
        index.panel_vertices(self.second_panel_ids)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        first = index.panel_vertices(self.first_panel_ids)
        second = index.panel_vertices(self.second_panel_ids)
        first_xy = _project(positions_m[first], table_normal)
        second_xy = _project(positions_m[second], table_normal)
        squared = np.sum((first_xy[:, None, :] - second_xy[None, :, :]) ** 2, axis=2)
        nearest_second = np.argmin(squared, axis=1)
        nearest_first = np.argmin(squared, axis=0)
        distances = np.concatenate(
            (
                np.sqrt(squared[np.arange(len(first)), nearest_second]),
                np.sqrt(squared[nearest_first, np.arange(len(second))]),
            )
        )
        rms = float(np.sqrt(np.mean(distances * distances)))
        normal_gaps = np.concatenate(
            (
                np.abs(
                    (positions_m[first] - positions_m[second[nearest_second]])
                    @ table_normal
                ),
                np.abs(
                    (positions_m[second] - positions_m[first[nearest_first]])
                    @ table_normal
                ),
            )
        )
        gap = float(np.quantile(normal_gaps, self.normal_gap_quantile))
        gates = MappingProxyType(
            {
                "symmetric_projected_rms": rms <= self.maximum_symmetric_projected_rms_m,
                "normal_gap_quantile": gap <= self.maximum_normal_gap_m,
            }
        )
        score = 0.5 * (
            _score_below(rms, self.maximum_symmetric_projected_rms_m)
            + _score_below(gap, self.maximum_normal_gap_m)
        )
        return GoalEvaluation(
            self.id,
            MappingProxyType(
                {"symmetric_projected_rms_m": rms, "normal_gap_quantile_m": gap}
            ),
            gates,
            score,
            "symmetric nearest-particle overlap of two semantic panel groups",
        )


@dataclass(frozen=True)
class ProjectedAreaFractionGoal:
    id: str
    maximum_fraction: float
    panel_ids: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(self.maximum_fraction, self.weight) <= 0.0:
            raise ValueError("Projected-area id, threshold, and weight must be positive")
        if self.maximum_fraction > 1.0:
            raise ValueError("Projected-area fraction cannot exceed one")

    def validate(self, index: SemanticGarmentIndex) -> None:
        if self.panel_ids:
            index.panel_vertices(self.panel_ids)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        indices = (
            index.panel_vertices(self.panel_ids)
            if self.panel_ids
            else np.arange(index.particle_count, dtype=np.int32)
        )
        reference_area = _convex_hull_area(
            _project(index.topology.physical_positions_m[indices], table_normal)
        )
        current_area = _convex_hull_area(_project(positions_m[indices], table_normal))
        fraction = current_area / max(reference_area, _EPSILON)
        return GoalEvaluation(
            self.id,
            MappingProxyType(
                {
                    "projected_area_fraction": float(fraction),
                    "reference_projected_area_m2": float(reference_area),
                    "current_projected_area_m2": float(current_area),
                }
            ),
            MappingProxyType({"projected_area_fraction": fraction <= self.maximum_fraction}),
            _score_below(float(fraction), self.maximum_fraction),
            "convex projected footprint relative to the compiled flat assembly",
        )


@dataclass(frozen=True)
class PathExtensionGoal:
    id: str
    anchor: PathRef
    tip: PathRef
    panel_id: str
    minimum_reference_fraction: float = 0.85
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or min(self.minimum_reference_fraction, self.weight) <= 0.0:
            raise ValueError("Path-extension id, fraction, and weight must be positive")
        if self.minimum_reference_fraction > 1.0:
            raise ValueError("Path-extension fraction cannot exceed one")

    def validate(self, index: SemanticGarmentIndex) -> None:
        if self.anchor.panel_id != self.panel_id or self.tip.panel_id != self.panel_id:
            raise ValueError("Path-extension landmarks must lie on the declared panel")
        panel = index.construction.panel_map[self.panel_id]
        panel.path(self.anchor.path_id)
        panel.path(self.tip.path_id)

    def evaluate(self, index, positions_m, table_normal) -> GoalEvaluation:
        anchor = np.mean(index.path_samples(self.anchor, positions_m, 5), axis=0)
        tip = np.mean(index.path_samples(self.tip, positions_m, 5), axis=0)
        reference = index.topology.physical_positions_m
        reference_anchor = np.mean(index.path_samples(self.anchor, reference, 5), axis=0)
        reference_tip = np.mean(index.path_samples(self.tip, reference, 5), axis=0)
        current_distance = float(np.linalg.norm(tip - anchor))
        reference_distance = float(np.linalg.norm(reference_tip - reference_anchor))
        fraction = current_distance / max(reference_distance, _EPSILON)
        return GoalEvaluation(
            self.id,
            MappingProxyType(
                {
                    "end_to_end_reference_fraction": float(fraction),
                    "end_to_end_distance_m": current_distance,
                    "reference_distance_m": reference_distance,
                }
            ),
            MappingProxyType(
                {"end_to_end_reference_fraction": fraction >= self.minimum_reference_fraction}
            ),
            _score_above(float(fraction), self.minimum_reference_fraction),
            "end-to-end extension of a semantic strip; intersections remain a separate gate",
        )


@dataclass(frozen=True)
class _SleevePreparation:
    sleeve: np.ndarray
    cuff: np.ndarray
    sleeve_weight: np.ndarray
    cuff_weight: np.ndarray
    body: np.ndarray
    loose: np.ndarray
    body_triangle: np.ndarray
    shell_triangle: np.ndarray
    own_triangle: np.ndarray
    reference_distance_m: np.ndarray
    reference_body_length_m: float
    reference_raw_credit: float


@dataclass(frozen=True)
class SleeveOnBodyGoal:
    """One sleeve and its cuff lying on a laid-out garment body, by any folding method.

    "On the body" is judged per unit of sleeve material. A unit is on the body when all hold:
    its plan position lies over the live body cloth (the union of projected body triangles, not a
    convex hull) and on the tabletop; it RESTS, i.e. it is within `rest_gap_m` above the table,
    body cloth, or loose (sleeve/cuff) cloth that itself rests, so support is grounded and a
    flattened sleeve's upper cloth layer held up by its own lower layer does not rest; it is not enclosed by
    the body shell (front/back cloth both above and below it: stuffed inside the open front, or
    a side or the hem folded over it); and no more than `maximum_own_layers` cloth layers (levels
    at least `layer_epsilon_m` apart) of its own sleeve and cuff are stacked in its column, whether
    bunched or neatly folded (the guide fold and a sleeve folded in thirds have six; a sleeve
    concertina'd into four or more full pleats has eight or more). Any face, crossing order, fold
    line, a doubled sleeve or a turned-back cuff is accepted; no height difference to one body
    particle is measured.

    The score is continuous progress from the task's start (0) to the sleeve and cuff resting on
    the laid-out body (1). Per unit: on the body 0.4 + 0.6 x layout (1 down to 75 % of the start's
    hem-to-neckline length, 0.4 at 50 %); resting over the body but enclosed or with too many own
    layers 0.4 (brought to the body, not in the asked state); anything else (in the air, or beside the body) at most 0.5, in proportion to how much
    its 3D distance to the body cloth has shrunk from the start; off the tabletop (more than
    10 mm below it or `above_table_limit_m` above it) 0. It rewards
    where the cloth is, not the grasp, order, fold line or intermediate shape.
    """

    id: str
    sleeve_panel_id: str
    cuff_panel_id: str
    body_panel_ids: tuple[str, ...]
    shell_panel_ids: tuple[str, ...]
    armhole_path_ids: tuple[str, ...]
    length_landmarks: tuple[PathRef, PathRef]
    sleeve_required_fraction: float = 0.85
    cuff_required_fraction: float = 0.80
    footprint_tolerance_m: float = 0.020
    armhole_band_m: float = 0.060
    cuff_share: float = 0.25
    layer_epsilon_m: float = 0.002
    rest_gap_m: float = 0.040
    maximum_own_layers: int = 7
    not_as_asked_credit: float = 0.4
    approach_ceiling: float = 0.5
    minimum_body_length_fraction: float = 0.75
    below_table_tolerance_m: float = 0.010
    above_table_limit_m: float = 0.30
    minimum_reference_distance_m: float = 0.050
    weight: float = 1.0
    uses_context: ClassVar[bool] = True

    def __post_init__(self) -> None:
        fractions = (self.sleeve_required_fraction, self.cuff_required_fraction, self.cuff_share,
                     self.minimum_body_length_fraction)
        if not self.id or self.weight <= 0.0 or not all(0.0 < value < 1.0 or value == 1.0 for value in fractions):
            raise ValueError("Sleeve-on-body id, weight and fractions must be positive and at most one")
        if self.minimum_body_length_fraction <= 0.5:
            raise ValueError("The laid-out body fraction must exceed one half")
        if min(self.footprint_tolerance_m, self.armhole_band_m, self.layer_epsilon_m, self.rest_gap_m,
               self.below_table_tolerance_m, self.above_table_limit_m, self.minimum_reference_distance_m) < 0.0:
            raise ValueError("Sleeve-on-body distances cannot be negative")
        if self.maximum_own_layers < 2 or not 0.0 < self.not_as_asked_credit < 1.0 or not 0.0 < self.approach_ceiling < 1.0:
            raise ValueError("Sleeve-on-body layer bound or partial credits are invalid")

    def validate(self, index: SemanticGarmentIndex) -> None:
        index.panel_vertices((self.sleeve_panel_id, self.cuff_panel_id))
        index.panel_vertices(self.body_panel_ids)
        if not set(self.shell_panel_ids) <= set(self.body_panel_ids):
            raise ValueError("The enclosing shell must be part of the body")
        for path_id in self.armhole_path_ids:
            index.construction.panel_map[self.sleeve_panel_id].path(path_id)
        for reference in self.length_landmarks:
            index.construction.panel_map[reference.panel_id].path(reference.path_id)

    def _body_length(self, index, positions_m) -> float:
        first, second = (
            np.mean(index.path_samples(reference, positions_m, 9), axis=0)[:2]
            for reference in self.length_landmarks
        )
        return float(np.linalg.norm(first - second))

    def prepare(self, index: SemanticGarmentIndex, reference_positions_m: np.ndarray) -> _SleevePreparation:
        topology, mesh = index.topology, index.pattern_mesh
        body = index.panel_vertices(self.body_panel_ids)
        cuff = np.setdiff1d(index.panel_vertices((self.cuff_panel_id,)), body)
        sleeve = np.setdiff1d(np.setdiff1d(index.panel_vertices((self.sleeve_panel_id,)), body), cuff)
        # Sleeve cloth within a pattern distance of the armhole seam is placed by the seam,
        # not by the fold; it lies past the side seam even after a perfect fold.
        start, stop = mesh.panel_vertex_ranges[self.sleeve_panel_id]
        material = topology.panel_material_coordinates_m
        band = np.full(topology.physical_vertex_count, np.inf)
        for path_id in self.armhole_path_ids:
            path = material[mesh.path_vertices[(self.sleeve_panel_id, path_id)]]
            local = material[start:stop]
            distance = _segment_distances(local[:, None, :], path[None, :-1], path[None, 1:]).min(axis=1)
            np.minimum.at(band, topology.material_to_physical_vertex[start:stop], distance)
        sleeve = sleeve[band[sleeve] > self.armhole_band_m]
        panel_index = {panel_id: number for number, panel_id in enumerate(topology.panel_ids)}
        own_triangle = np.isin(
            topology.triangle_panel_index,
            [panel_index[self.sleeve_panel_id], panel_index[self.cuff_panel_id]],
        )
        area = _lumped_material_area(topology, own_triangle)
        body_triangle = np.isin(topology.triangle_panel_index, [panel_index[p] for p in self.body_panel_ids])
        shell_triangle = np.isin(topology.triangle_panel_index, [panel_index[p] for p in self.shell_panel_ids])
        loose = np.setdiff1d(np.arange(topology.physical_vertex_count), body)
        reference = np.asarray(reference_positions_m, dtype=np.float64)
        from scipy.spatial import cKDTree  # noqa: PLC0415

        scored = np.concatenate((sleeve, cuff))
        reference_distance = np.maximum(
            cKDTree(reference[body]).query(reference[scored])[0], self.minimum_reference_distance_m
        )
        partial = _SleevePreparation(
            sleeve, cuff, area[sleeve], area[cuff], body, loose, body_triangle, shell_triangle, own_triangle,
            reference_distance, self._body_length(index, reference), 0.0,
        )
        raw = self._measure(index, reference, None, partial, 1.0, None)["raw_credit"]
        return _SleevePreparation(**{**partial.__dict__, "reference_raw_credit": raw})

    def _columns(self, index, positions, tabletop, prepared, shared):
        """Column facts for every loose (non-body) particle, shared by both sleeves' goals."""
        key = ("sleeve_on_body_columns", self.footprint_tolerance_m, self.layer_epsilon_m, self.rest_gap_m,
               self.below_table_tolerance_m, self.above_table_limit_m)
        if shared is not None and key in shared:
            return shared[key]
        from scipy.spatial import cKDTree  # noqa: PLC0415

        triangles = index.topology.physical_triangles
        loose = prepared.loose
        point = positions[loose]
        on_table = (
            np.ones(len(loose), dtype=bool)
            if tabletop is None
            else tabletop.supports(point, self.below_table_tolerance_m, self.above_table_limit_m)
        )
        corners = positions[triangles]
        centroid = corners[:, :, :2].mean(axis=1)
        radius = float(np.max(np.linalg.norm(corners[:, :, :2] - centroid[:, None, :], axis=2)))
        candidates = cKDTree(centroid).query_ball_point(point[:, :2], r=radius + self.footprint_tolerance_m)
        query = np.repeat(np.arange(len(loose)), [len(c) for c in candidates])
        triangle = np.fromiter((t for c in candidates for t in c), dtype=np.int64, count=len(query))
        a, b, c = (corners[triangle, k] for k in range(3))
        p = point[query]
        v0, v1, v2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], p[:, :2] - a[:, :2]
        denominator = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        valid = np.abs(denominator) > 1.0e-14
        safe = np.where(valid, denominator, 1.0)
        beta = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / safe
        gamma = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / safe
        alpha = 1.0 - beta - gamma
        contains = valid & (alpha >= -1.0e-9) & (beta >= -1.0e-9) & (gamma >= -1.0e-9)
        is_body = prepared.body_triangle[triangle]
        # Plan distance to the body cloth: zero inside a projected body triangle.
        edge_distance = np.minimum.reduce([
            _segment_distances(p[:, :2], a[:, :2], b[:, :2]),
            _segment_distances(p[:, :2], b[:, :2], c[:, :2]),
            _segment_distances(p[:, :2], c[:, :2], a[:, :2]),
        ])
        near_body = np.zeros(len(loose), dtype=bool)
        np.logical_or.at(near_body, query[is_body],
                         np.where(contains, 0.0, edge_distance)[is_body] <= self.footprint_tolerance_m)
        # Column through each loose point: cloth surfaces strictly above / below it.
        column = contains & ~np.any(triangles[triangle] == loose[query][:, None], axis=1)
        height = alpha * a[:, 2] + beta * b[:, 2] + gamma * c[:, 2]
        above = column & (height > p[:, 2] + self.layer_epsilon_m)
        below = column & (height < p[:, 2] - self.layer_epsilon_m)
        is_shell = prepared.shell_triangle[triangle]
        shell_above = np.zeros(len(loose), dtype=bool)
        shell_below = np.zeros(len(loose), dtype=bool)
        np.logical_or.at(shell_above, query[above & is_shell], True)
        np.logical_or.at(shell_below, query[below & is_shell], True)
        # Grounded rest: within the rest gap above the table or body cloth, or above loose cloth
        # whose own vertices rest. Propagated upward through any number of layers.
        floor = tabletop.top_z_m if tabletop is not None else float(np.min(positions[:, 2]))
        support = below & (p[:, 2] - height <= self.rest_gap_m)
        base = on_table & (point[:, 2] - floor <= self.rest_gap_m)
        np.logical_or.at(base, query[support & is_body], True)
        loose_support = support & ~is_body
        grounded_all = np.ones(index.topology.physical_vertex_count, dtype=bool)
        grounded = base.copy()
        for _ in range(64):
            grounded_all[loose] = grounded
            resting_triangle = grounded_all[triangles[triangle[loose_support]]].sum(axis=1) >= 2
            following = base.copy()
            np.logical_or.at(following, query[loose_support][resting_triangle], True)
            if np.array_equal(following, grounded):
                break
            grounded = following
        result = dict(on_table=on_table, near_body=near_body, enclosed=shell_above & shell_below,
                      grounded=grounded, query=query, triangle=triangle, column=column, height=height)
        if shared is not None:
            shared[key] = result
        return result

    def _measure(self, index, positions, tabletop, prepared, layout, shared) -> dict[str, float]:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        columns = self._columns(index, positions, tabletop, prepared, shared)
        scored = np.concatenate((prepared.sleeve, prepared.cuff))
        slot = np.searchsorted(prepared.loose, scored)
        point = positions[scored]
        on_table, near_body = columns["on_table"][slot], columns["near_body"][slot]
        enclosed, grounded = columns["enclosed"][slot], columns["grounded"][slot]
        # Distinct levels of this side's own sleeve and cuff in each scored unit's column.
        position_in_scored = np.full(len(prepared.loose), -1)
        position_in_scored[slot] = np.arange(len(scored))
        pick = columns["column"] & prepared.own_triangle[columns["triangle"]]
        owner = position_in_scored[columns["query"][pick]]
        keep = owner >= 0
        unit = np.concatenate((owner[keep], np.arange(len(scored))))
        level = np.concatenate((columns["height"][pick][keep], point[:, 2]))
        order = np.lexsort((level, unit))
        unit, level = unit[order], level[order]
        new_level = np.r_[True, (unit[1:] != unit[:-1]) | (np.diff(level) > self.layer_epsilon_m)]
        own_layers = np.bincount(unit[new_level], minlength=len(scored))
        bunched = own_layers > self.maximum_own_layers
        resting = on_table & near_body & grounded
        on_body = resting & ~enclosed & ~bunched
        distance = cKDTree(positions[prepared.body]).query(point)[0]
        approach = self.approach_ceiling * np.clip(1.0 - distance / prepared.reference_distance_m, 0.0, 1.0)
        reduced = self.not_as_asked_credit
        credit = np.where(on_body, reduced + (1.0 - reduced) * layout, np.where(resting, reduced, approach))
        credit = np.where(on_table, credit, 0.0)
        count = len(prepared.sleeve)
        weights = (prepared.sleeve_weight, prepared.cuff_weight)
        parts = (slice(0, count), slice(count, None))
        both = np.concatenate(weights)
        sleeve_credit, cuff_credit = (float(np.average(credit[s], weights=w)) for s, w in zip(parts, weights))
        sleeve_on, cuff_on = (float(np.average(on_body[s], weights=w)) for s, w in zip(parts, weights))
        return dict(
            sleeve_on_body_fraction=sleeve_on,
            cuff_on_body_fraction=cuff_on,
            enclosed_fraction=float(np.average(resting & enclosed, weights=both)),
            bunched_fraction=float(np.average(resting & bunched, weights=both)),
            over_body_not_resting_fraction=float(np.average(on_table & near_body & ~grounded, weights=both)),
            off_table_fraction=float(np.average(~on_table, weights=both)),
            maximum_own_layers=float(own_layers.max()),
            raw_credit=(1.0 - self.cuff_share) * sleeve_credit + self.cuff_share * cuff_credit,
        )

    def evaluate(self, index, positions_m, table_normal, context: GoalContext) -> GoalEvaluation:
        if not np.allclose(table_normal, (0.0, 0.0, 1.0)):
            raise ValueError("Sleeve-on-body evaluation assumes a +z tabletop normal")
        prepared = context.prepared
        length = self._body_length(index, positions_m) / max(prepared.reference_body_length_m, _EPSILON)
        # Sleeve cloth on a bunched, rolled or hem-folded body is not on a laid-out body.
        layout = float(np.clip((length - 0.5) / (self.minimum_body_length_fraction - 0.5), 0.0, 1.0))
        measured = self._measure(index, positions_m, context.tabletop, prepared, layout, context.shared)
        reference = prepared.reference_raw_credit
        progress = float(np.clip((measured["raw_credit"] - reference) / max(1.0 - reference, _EPSILON), 0.0, 1.0))
        gates = MappingProxyType(
            {
                "sleeve_on_body": measured["sleeve_on_body_fraction"] >= self.sleeve_required_fraction,
                "cuff_on_body": measured["cuff_on_body_fraction"] >= self.cuff_required_fraction,
                "body_laid_out": length >= self.minimum_body_length_fraction,
            }
        )
        return GoalEvaluation(
            self.id,
            MappingProxyType({**measured, "body_length_fraction": float(length), "progress": progress}),
            gates,
            progress,
            "area share of the sleeve and of its cuff resting on the laid-out body cloth; face- and method-agnostic",
        )


@dataclass(frozen=True)
class GarmentTaskSpec:
    id: str
    label: str
    construction_id: str
    goals: tuple[GarmentGoal, ...]
    maximum_speed_m_s: float = 0.020
    maximum_structural_strain: float = 0.30
    require_release: bool = True
    require_fixture_clearance: bool = True
    calibration_status: str = (
        "semantic objective and thresholds are engineering priors; no real-fold distribution registered"
    )
    # Optional task-scoped definitions; the defaults keep the original evaluation exactly.
    # rest_window_s: "resting" is no particle displaced more than maximum_speed_m_s * window
    # from its current position during the last window, instead of one instantaneous speed sample.
    rest_window_s: float | None = None
    # limit_compression=False: the strain gate bounds stretch only (in-plane compression of a
    # mesh triangle is how sub-mesh wrinkles and folds appear); requires the stretch range.
    limit_compression: bool = True
    # Minimum rest-area share of the garment over the tabletop, not more than 10 mm below its
    # surface and not held more than 0.30 m above it ("resting on the table").
    tabletop_support_fraction: float | None = None
    # Score 1.0 exactly when every gate passes, otherwise partial_credit_cap * progress.
    full_credit_requires_success: bool = False
    partial_credit_cap: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or not self.label or not self.construction_id or not self.goals:
            raise ValueError("Garment task specs require ids, labels, construction, and goals")
        if min(self.maximum_speed_m_s, self.maximum_structural_strain) <= 0.0:
            raise ValueError("Garment task state thresholds must be positive")
        goal_ids = tuple(goal.id for goal in self.goals)
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("Garment task goal identifiers must be unique")
        if any(goal.weight <= 0.0 for goal in self.goals):
            raise ValueError("Garment task goal weights must be positive")
        if self.rest_window_s is not None and not (math.isfinite(self.rest_window_s) and self.rest_window_s > 0.0):
            raise ValueError("A rest window must be finite and positive")
        if self.tabletop_support_fraction is not None and not 0.0 < self.tabletop_support_fraction <= 1.0:
            raise ValueError("Tabletop support fraction must lie in (0, 1]")
        if not 0.0 < self.partial_credit_cap <= 1.0 or (
            self.full_credit_requires_success and self.partial_credit_cap >= 1.0
        ):
            raise ValueError("Partial credit must stay below the full credit reserved for success")

    def bind(
        self,
        construction: GarmentConstruction,
        pattern_mesh: GarmentPatternMesh,
        topology: SewnGarmentTopology,
        *,
        reference_positions_m=None,
    ) -> "GarmentTaskEvaluator":
        if construction.id != self.construction_id:
            raise ValueError(
                f"Task {self.id!r} expects {self.construction_id!r}, got {construction.id!r}"
            )
        index = SemanticGarmentIndex(construction, pattern_mesh, topology)
        for goal in self.goals:
            goal.validate(index)
        return GarmentTaskEvaluator(self, index, reference_positions_m=reference_positions_m)


@dataclass(frozen=True)
class GarmentTaskMetrics:
    maximum_speed_m_s: float
    maximum_structural_strain: float
    nonlocal_intersection_count: int
    runtime_pose_writes: int
    external_particle_force_applications: int
    released: bool
    fixtures_clear: bool
    goal_metrics: Mapping[str, Mapping[str, float]]
    rest_window_displacement_m: float | None = None
    rest_window_span_s: float | None = None
    principal_stretch_range: tuple[float, float] | None = None
    tabletop_supported_fraction: float | None = None


@dataclass(frozen=True)
class GarmentTaskResult:
    success: bool
    potential: float
    metrics: GarmentTaskMetrics
    goal_scores: Mapping[str, float]
    gates: Mapping[str, bool]
    failed_gates: tuple[str, ...]
    progress: float | None = None


class GarmentTaskEvaluator:
    """Evaluate a bound task without mutating the simulation state."""

    def __init__(self, spec: GarmentTaskSpec, index: SemanticGarmentIndex, *, reference_positions_m=None):
        self.spec = spec
        self.index = index
        reference = (
            index.topology.physical_positions_m if reference_positions_m is None else reference_positions_m
        )
        self.reference_positions_m = np.array(reference, dtype=np.float64)
        if self.reference_positions_m.shape != (index.particle_count, 3):
            raise ValueError("Reference positions must match the garment particle count")
        self._prepared = {
            goal.id: goal.prepare(index, self.reference_positions_m)
            for goal in spec.goals
            if getattr(goal, "uses_context", False)
        }
        self._garment_area = _lumped_material_area(
            index.topology, np.ones(len(index.topology.physical_triangles), dtype=bool)
        )

    def _goal_evaluation(self, goal, positions, normal, tabletop, shared) -> GoalEvaluation:
        if not getattr(goal, "uses_context", False):
            return goal.evaluate(self.index, positions, normal)
        context = GoalContext(tabletop, self.reference_positions_m, self._prepared[goal.id], shared)
        return goal.evaluate(self.index, positions, normal, context)

    def _window_displacement(self, positions, recent_positions_m, recent_span_s):
        if recent_positions_m is None or recent_span_s is None:
            return None, None
        recent = np.asarray(recent_positions_m, dtype=np.float64)
        if recent.ndim != 3 or recent.shape[1:] != positions.shape or not np.all(np.isfinite(recent)):
            raise ValueError("Recent positions must be finite (samples, particles, 3)")
        if not math.isfinite(recent_span_s) or recent_span_s < 0.0:
            raise ValueError("Recent position span must be finite and nonnegative")
        if len(recent) == 0:
            return None, float(recent_span_s)
        return float(np.max(np.linalg.norm(recent - positions[None], axis=2))), float(recent_span_s)

    def evaluate(
        self,
        positions_m,
        velocities_m_s,
        *,
        maximum_structural_strain: float,
        nonlocal_intersection_count: int,
        runtime_pose_writes: int,
        external_particle_force_applications: int,
        released: bool,
        fixtures_clear: bool,
        table_normal=(0.0, 0.0, 1.0),
        tabletop: TabletopFrame | None = None,
        recent_positions_m=None,
        recent_span_s: float | None = None,
        principal_stretch_range: tuple[float, float] | None = None,
    ) -> GarmentTaskResult:
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.asarray(velocities_m_s, dtype=np.float64)
        expected = (self.index.particle_count, 3)
        if positions.shape != expected or velocities.shape != expected:
            raise ValueError(f"Task state arrays must have shape {expected}")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Task particle state must be finite")
        if not math.isfinite(maximum_structural_strain) or maximum_structural_strain < 0.0:
            raise ValueError("Maximum structural strain must be finite and nonnegative")
        counts = (
            nonlocal_intersection_count,
            runtime_pose_writes,
            external_particle_force_applications,
        )
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in counts):
            raise ValueError("Task integrity counts must be nonnegative integers")
        normal = _unit_vector(table_normal, "table normal")
        shared: dict = {}
        evaluations = tuple(
            self._goal_evaluation(goal, positions, normal, tabletop, shared) for goal in self.spec.goals
        )
        goal_metrics = MappingProxyType(
            {evaluation.goal_id: evaluation.metrics for evaluation in evaluations}
        )
        goal_scores = MappingProxyType(
            {evaluation.goal_id: evaluation.score for evaluation in evaluations}
        )
        gates: dict[str, bool] = {}
        for evaluation in evaluations:
            gates.update(
                {
                    f"goal:{evaluation.goal_id}:{name}": bool(passed)
                    for name, passed in evaluation.gates.items()
                }
            )
        maximum_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
        displacement, span = self._window_displacement(positions, recent_positions_m, recent_span_s)
        if self.spec.rest_window_s is None:
            gates["state:settled_speed"] = maximum_speed <= self.spec.maximum_speed_m_s
        else:
            # Rest is a property of motion over time; one instantaneous velocity sample also
            # reports solver velocity noise of particles that do not move.
            gates["state:at_rest"] = (
                displacement is not None
                and span >= self.spec.rest_window_s * (1.0 - 1.0e-9)
                and displacement <= self.spec.maximum_speed_m_s * self.spec.rest_window_s
            )
        stretch = None
        if principal_stretch_range is not None:
            stretch = tuple(float(value) for value in principal_stretch_range)
            if len(stretch) != 2 or not all(math.isfinite(value) and value >= 0.0 for value in stretch):
                raise ValueError("Principal stretch range must contain two finite nonnegative values")
        gated_strain = (
            max(stretch[1] - 1.0, 0.0)
            if not self.spec.limit_compression and stretch is not None
            else float(maximum_structural_strain)
        )
        gates.update(
            {
                "integrity:structural_strain": (
                    gated_strain <= self.spec.maximum_structural_strain
                ),
                "integrity:nonlocal_intersections": nonlocal_intersection_count == 0,
                "integrity:no_runtime_pose_writes": runtime_pose_writes == 0,
                "integrity:no_direct_particle_forces": (
                    external_particle_force_applications == 0
                ),
                "state:released": bool(released) or not self.spec.require_release,
                "state:fixtures_clear": (
                    bool(fixtures_clear) or not self.spec.require_fixture_clearance
                ),
            }
        )
        supported = None
        if self.spec.tabletop_support_fraction is not None:
            if tabletop is not None:
                supported = float(np.average(tabletop.supports(positions, 0.010, 0.30), weights=self._garment_area))
            gates["state:on_table"] = (
                supported is not None and supported >= self.spec.tabletop_support_fraction
            )
        total_weight = sum(goal.weight for goal in self.spec.goals)
        progress = sum(
            goal.weight * evaluation.score
            for goal, evaluation in zip(self.spec.goals, evaluations, strict=True)
        ) / total_weight
        failed = tuple(name for name, passed in gates.items() if not passed)
        potential = progress
        if self.spec.full_credit_requires_success:
            # Every valid completion scores exactly 1; nothing unfinished can tie it. A state
            # with crossing cloth is an integrity failure, not partial progress.
            potential = (
                1.0 if not failed
                else 0.0 if nonlocal_intersection_count > 0
                else self.spec.partial_credit_cap * progress
            )
        metrics = GarmentTaskMetrics(
            maximum_speed_m_s=maximum_speed,
            maximum_structural_strain=float(maximum_structural_strain),
            nonlocal_intersection_count=int(nonlocal_intersection_count),
            runtime_pose_writes=int(runtime_pose_writes),
            external_particle_force_applications=int(
                external_particle_force_applications
            ),
            released=bool(released),
            fixtures_clear=bool(fixtures_clear),
            goal_metrics=goal_metrics,
            rest_window_displacement_m=displacement,
            rest_window_span_s=span,
            principal_stretch_range=stretch,
            tabletop_supported_fraction=supported,
        )
        return GarmentTaskResult(
            success=not failed,
            potential=float(potential),
            metrics=metrics,
            goal_scores=goal_scores,
            gates=MappingProxyType(gates),
            failed_gates=failed,
            progress=float(progress) if self.spec.full_credit_requires_success else None,
        )


class GarmentEpisodePhase(str, Enum):
    READY = "ready"
    GRASPED = "grasped"
    MANIPULATING = "manipulating"
    RELEASED = "released"
    SETTLING = "settling"
    SUCCESS = "success"
    FAILED = "failed"
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class GarmentEpisodeState:
    """Minimal backend-neutral state consumed by a semantic task episode."""

    time_s: float
    particle_positions_m: np.ndarray
    particle_velocities_m_s: np.ndarray
    maximum_structural_strain: float
    nonlocal_intersection_count: int
    runtime_pose_writes: int
    external_particle_force_applications: int
    pinched_particle_count: int
    released: bool
    fixtures_clear: bool
    table_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("Episode time must be finite and nonnegative")
        if (
            isinstance(self.pinched_particle_count, bool)
            or int(self.pinched_particle_count) != self.pinched_particle_count
            or self.pinched_particle_count < 0
        ):
            raise ValueError("Pinched-particle count must be a nonnegative integer")


@dataclass(frozen=True)
class GarmentEpisodeStep:
    phase: GarmentEpisodePhase
    task_result: GarmentTaskResult
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: Mapping[str, object]


class GarmentEpisodeTask:
    """History, anti-cheat termination, and potential-difference reward.

    The geometry potential cannot be farmed by holding a static intermediate
    state.  Success additionally requires a real grasp/release history unless
    explicitly disabled for diagnostic evaluation.
    """

    def __init__(
        self,
        evaluator: GarmentTaskEvaluator,
        *,
        timeout_s: float = 60.0,
        require_grasp_history: bool = True,
    ) -> None:
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            raise ValueError("Garment task timeout must be finite and positive")
        self.evaluator = evaluator
        self.timeout_s = float(timeout_s)
        self.require_grasp_history = bool(require_grasp_history)
        self.reset()

    def reset(self) -> None:
        self.phase = GarmentEpisodePhase.READY
        self.saw_grasp = False
        self.saw_release_after_grasp = False
        self.peak_strain = 0.0
        self.maximum_intersections = 0
        self.maximum_runtime_pose_writes = 0
        self.maximum_external_force_applications = 0
        self.previous_potential: float | None = None
        self.failure_reason: str | None = None

    def update(
        self,
        state: GarmentEpisodeState,
        action: np.ndarray | None = None,
    ) -> GarmentEpisodeStep:
        self.peak_strain = max(
            self.peak_strain,
            float(state.maximum_structural_strain),
        )
        self.maximum_intersections = max(
            self.maximum_intersections,
            int(state.nonlocal_intersection_count),
        )
        self.maximum_runtime_pose_writes = max(
            self.maximum_runtime_pose_writes,
            int(state.runtime_pose_writes),
        )
        self.maximum_external_force_applications = max(
            self.maximum_external_force_applications,
            int(state.external_particle_force_applications),
        )
        grasped_now = state.pinched_particle_count > 0
        new_grasp = grasped_now and not self.saw_grasp
        self.saw_grasp = self.saw_grasp or grasped_now
        released_after_grasp = self.saw_grasp and state.released
        new_release = released_after_grasp and not self.saw_release_after_grasp
        self.saw_release_after_grasp = (
            self.saw_release_after_grasp or released_after_grasp
        )
        released_gate = (
            released_after_grasp
            if self.require_grasp_history
            else state.released
        )
        result = self.evaluator.evaluate(
            state.particle_positions_m,
            state.particle_velocities_m_s,
            maximum_structural_strain=self.peak_strain,
            nonlocal_intersection_count=self.maximum_intersections,
            runtime_pose_writes=self.maximum_runtime_pose_writes,
            external_particle_force_applications=(
                self.maximum_external_force_applications
            ),
            released=released_gate,
            fixtures_clear=state.fixtures_clear,
            table_normal=state.table_normal,
        )

        if self.maximum_intersections > 0:
            self.failure_reason = "nonlocal cloth intersection"
        elif self.maximum_runtime_pose_writes > 0:
            self.failure_reason = "runtime cloth-pose authoring"
        elif self.maximum_external_force_applications > 0:
            self.failure_reason = "direct external particle force application"
        elif self.peak_strain > self.evaluator.spec.maximum_structural_strain:
            self.failure_reason = "peak structural strain exceeded the task bound"

        history_ok = self.saw_grasp or not self.require_grasp_history
        success = bool(result.success and history_ok)
        hard_failure = self.failure_reason is not None
        timed_out = state.time_s >= self.timeout_s and not success and not hard_failure
        maximum_speed = result.metrics.maximum_speed_m_s
        settled = maximum_speed <= self.evaluator.spec.maximum_speed_m_s
        if success:
            self.phase = GarmentEpisodePhase.SUCCESS
        elif hard_failure:
            self.phase = GarmentEpisodePhase.FAILED
        elif timed_out:
            self.phase = GarmentEpisodePhase.TRUNCATED
        elif self.saw_release_after_grasp:
            self.phase = (
                GarmentEpisodePhase.RELEASED
                if settled
                else GarmentEpisodePhase.SETTLING
            )
        elif grasped_now and self.previous_potential is not None:
            self.phase = GarmentEpisodePhase.MANIPULATING
        elif grasped_now:
            self.phase = GarmentEpisodePhase.GRASPED
        else:
            self.phase = GarmentEpisodePhase.READY

        reward = self._reward(
            result,
            action=action,
            new_grasp=new_grasp,
            new_release=new_release,
            success=success,
            hard_failure=hard_failure,
            timed_out=timed_out,
        )
        self.previous_potential = result.potential
        info = MappingProxyType(
            {
                "phase": self.phase.value,
                "task_id": self.evaluator.spec.id,
                "construction_id": self.evaluator.spec.construction_id,
                "success": success,
                "failure_reason": self.failure_reason,
                "saw_grasp": self.saw_grasp,
                "saw_release_after_grasp": self.saw_release_after_grasp,
                "potential": result.potential,
                "failed_gates": result.failed_gates,
                "goal_scores": dict(result.goal_scores),
                "state_thresholds": {
                    "maximum_speed_m_s": self.evaluator.spec.maximum_speed_m_s,
                    "maximum_structural_strain": (
                        self.evaluator.spec.maximum_structural_strain
                    ),
                },
                "calibration_status": self.evaluator.spec.calibration_status,
            }
        )
        return GarmentEpisodeStep(
            phase=self.phase,
            task_result=result,
            reward=reward,
            terminated=success or hard_failure,
            truncated=timed_out,
            success=success,
            info=info,
        )

    def _reward(
        self,
        result: GarmentTaskResult,
        *,
        action: np.ndarray | None,
        new_grasp: bool,
        new_release: bool,
        success: bool,
        hard_failure: bool,
        timed_out: bool,
    ) -> float:
        previous = (
            result.potential
            if self.previous_potential is None
            else self.previous_potential
        )
        reward = 4.0 * (result.potential - previous) - 1.0e-3
        if action is not None:
            value = np.asarray(action, dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError("Reward action must be finite")
            reward -= 2.0e-4 * float(np.dot(value, value))
        if new_grasp:
            reward += 0.10
        if new_release:
            reward += 0.20 if result.potential >= 0.70 else -0.20
        if success:
            reward += 5.0
        elif hard_failure:
            reward -= 5.0
        elif timed_out:
            reward -= 1.0
        if not math.isfinite(reward):
            raise RuntimeError("Garment task reward became non-finite")
        return float(reward)


def _ref(panel: str, path: str, *, reverse: bool = False) -> PathRef:
    return PathRef(panel, path, reverse=reverse)


_SHIRT_BODY_PANELS = (
    "left_front", "right_front", "back", "back_yoke", "chest_pocket", "collar_stand", "collar",
)


GARMENT_TASK_SPECS: Mapping[tuple[str, str], GarmentTaskSpec] = MappingProxyType(
    {
        (spec.construction_id, spec.id): spec
        for spec in (
            GarmentTaskSpec(
                id="sleeve_tuck_fold",
                label="Tuck both short sleeves into the T-shirt body footprint",
                construction_id="cotton_jersey_half_sleeve_tee",
                goals=(
                    PanelContainmentGoal(
                        "left_sleeve_contained",
                        ("left_sleeve",),
                        ("front", "back"),
                        required_fraction=0.90,
                    ),
                    PanelStackGoal(
                        "left_sleeve_stacked",
                        ("left_sleeve",),
                        ("front", "back"),
                    ),
                    PanelContainmentGoal(
                        "right_sleeve_contained",
                        ("right_sleeve",),
                        ("front", "back"),
                        required_fraction=0.90,
                    ),
                    PanelStackGoal(
                        "right_sleeve_stacked",
                        ("right_sleeve",),
                        ("front", "back"),
                    ),
                ),
            ),
            GarmentTaskSpec(
                id="sleeve_fold",
                label="Fold both long sleeves and cuffs onto the shirt body",
                construction_id="woven_button_down_shirt",
                goals=tuple(
                    SleeveOnBodyGoal(
                        f"{side}_sleeve_and_cuff_on_body",
                        f"{side}_sleeve",
                        f"{side}_cuff",
                        _SHIRT_BODY_PANELS,
                        ("left_front", "right_front", "back"),
                        ("cap_front", "cap_back"),
                        (_ref("back", "bottom_hem"), _ref("back", "neckline")),
                    )
                    for side in ("left", "right")
                ),
                rest_window_s=0.5,
                limit_compression=False,
                tabletop_support_fraction=0.95,
                full_credit_requires_success=True,
                partial_credit_cap=0.9,
            ),
            GarmentTaskSpec(
                id="leg_align_fold",
                label="Fold one jeans leg shell onto the other",
                construction_id="denim_five_pocket_jeans",
                maximum_structural_strain=0.25,
                goals=(
                    PanelCloudAlignmentGoal(
                        "leg_shell_overlap",
                        ("front_left", "back_left"),
                        ("front_right", "back_right"),
                        maximum_symmetric_projected_rms_m=0.045,
                        maximum_normal_gap_m=0.030,
                    ),
                    PathAlignmentGoal(
                        "front_hem_alignment",
                        _ref("front_left", "hem"),
                        _ref("front_right", "hem"),
                        rms_threshold_m=0.025,
                        maximum_threshold_m=0.055,
                    ),
                    PathAlignmentGoal(
                        "back_hem_alignment",
                        _ref("back_left", "hem"),
                        _ref("back_right", "hem"),
                        rms_threshold_m=0.025,
                        maximum_threshold_m=0.055,
                    ),
                ),
            ),
            GarmentTaskSpec(
                id="half_fold",
                label="Fold the terry towel long edge to long edge",
                construction_id="terry_cotton_bath_towel",
                goals=(
                    PathAlignmentGoal(
                        "long_hem_alignment",
                        _ref("terry_body", "long_left_hem"),
                        _ref("terry_body", "long_right_hem", reverse=True),
                        rms_threshold_m=0.025,
                        maximum_threshold_m=0.060,
                    ),
                    ProjectedAreaFractionGoal(
                        "half_footprint",
                        maximum_fraction=0.62,
                        panel_ids=("terry_body",),
                    ),
                ),
                maximum_structural_strain=0.20,
            ),
            GarmentTaskSpec(
                id="tie_untangle",
                label="Extend both asymmetric waist ties without intersections",
                construction_id="asymmetric_draped_one_sleeve_top",
                goals=(
                    PathExtensionGoal(
                        "left_tie_extended",
                        _ref("left_tie", "anchor"),
                        _ref("left_tie", "tip"),
                        "left_tie",
                        minimum_reference_fraction=0.85,
                    ),
                    PathExtensionGoal(
                        "right_tie_extended",
                        _ref("right_tie", "anchor"),
                        _ref("right_tie", "tip"),
                        "right_tie",
                        minimum_reference_fraction=0.85,
                    ),
                ),
                maximum_structural_strain=0.35,
            ),
        )
    }
)


def get_garment_task_spec(construction_id: str, task_id: str) -> GarmentTaskSpec:
    try:
        return GARMENT_TASK_SPECS[(construction_id, task_id)]
    except KeyError as exception:
        raise KeyError(
            f"No semantic task {task_id!r} is registered for {construction_id!r}"
        ) from exception

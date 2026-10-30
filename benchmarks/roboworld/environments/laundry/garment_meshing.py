"""Triangulation and stitch compilation for arbitrary sewn garments.

The regular towel grid cannot represent armholes, crotch curves, pocket bags,
or overlapping panels.  This module triangulates every validated
``PanelPattern`` independently, retains material coordinates for anisotropic
cloth, and converts each construction seam into a mesh-resolution-invariant
batch of stitch pairs.

The returned layout is a *pattern atlas*: panels are separated so topology can
be audited without overlap.  It is not silently advertised as an assembled
task pose.  ``garment_assembly.py`` is responsible for placing those same
particles into a sewn three-dimensional rest/initial configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .garment_asset import GarmentConstruction, PanelPattern, PathRef, PatternPath, point_in_polygon


_AREA_EPSILON_M2 = 1.0e-12


@dataclass(frozen=True)
class PanelMesh:
    panel_id: str
    material_coordinates_m: np.ndarray
    triangles: np.ndarray
    vertex_area_m2: np.ndarray
    vertex_mass_kg: np.ndarray
    uv: np.ndarray
    path_vertices: Mapping[str, np.ndarray]
    path_fractions: Mapping[str, np.ndarray]
    target_edge_length_m: float
    source_area_m2: float

    def __post_init__(self) -> None:
        vertex_count = self.material_coordinates_m.shape[0]
        if self.material_coordinates_m.ndim != 2 or self.material_coordinates_m.shape[1] != 2:
            raise ValueError("Panel material coordinates must have shape (vertices, 2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("Panel triangles must have shape (triangles, 3)")
        if self.vertex_area_m2.shape != (vertex_count,) or self.vertex_mass_kg.shape != (vertex_count,):
            raise ValueError("Panel vertex scalar arrays have the wrong shape")
        if self.uv.shape != (vertex_count, 2):
            raise ValueError("Panel UV coordinates have the wrong shape")
        if self.target_edge_length_m <= 0.0 or self.source_area_m2 <= 0.0:
            raise ValueError("Panel mesh target length and source area must be positive")
        if len(self.triangles) == 0:
            raise ValueError("Panel mesh requires at least one triangle")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= vertex_count:
            raise ValueError("Panel triangle index lies outside the vertex array")
        if not (
            np.all(np.isfinite(self.material_coordinates_m))
            and np.all(np.isfinite(self.vertex_area_m2))
            and np.all(np.isfinite(self.vertex_mass_kg))
        ):
            raise ValueError("Panel mesh contains non-finite values")
        if np.min(self.triangle_signed_areas_m2) <= _AREA_EPSILON_M2:
            raise ValueError("Panel mesh contains an inverted or degenerate triangle")
        if set(self.path_vertices) != set(self.path_fractions):
            raise ValueError("Panel path index and arclength maps differ")
        for path_id, indices in self.path_vertices.items():
            fractions = self.path_fractions[path_id]
            if len(indices) < 2 or fractions.shape != indices.shape:
                raise ValueError(f"Path {path_id!r} has an invalid vertex/fraction map")
            if np.min(indices) < 0 or np.max(indices) >= vertex_count:
                raise ValueError(f"Path {path_id!r} references a missing mesh vertex")
            if not np.all(np.diff(fractions) > 0.0):
                raise ValueError(f"Path {path_id!r} fractions are not strictly increasing")
            if abs(float(fractions[0])) > 1.0e-12 or abs(float(fractions[-1]) - 1.0) > 1.0e-12:
                raise ValueError(f"Path {path_id!r} fraction map must include both endpoints")

    @property
    def vertex_count(self) -> int:
        return int(len(self.material_coordinates_m))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def triangle_signed_areas_m2(self) -> np.ndarray:
        a = self.material_coordinates_m[self.triangles[:, 0]]
        b = self.material_coordinates_m[self.triangles[:, 1]]
        c = self.material_coordinates_m[self.triangles[:, 2]]
        return 0.5 * (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    @property
    def triangulated_area_m2(self) -> float:
        return float(np.sum(self.triangle_signed_areas_m2))

    @property
    def area_error_fraction(self) -> float:
        return abs(self.triangulated_area_m2 - self.source_area_m2) / self.source_area_m2

    @property
    def mass_quadrature_scale(self) -> float:
        """Correction from sampled contour area to authored pattern area."""

        return self.source_area_m2 / self.triangulated_area_m2

    @property
    def triangle_aspect_ratios(self) -> np.ndarray:
        points = self.material_coordinates_m[self.triangles]
        edge_lengths = np.stack(
            (
                np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
                np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
                np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
            ),
            axis=1,
        )
        maximum_edge = np.max(edge_lengths, axis=1)
        minimum_altitude = 2.0 * self.triangle_signed_areas_m2 / maximum_edge
        return maximum_edge / minimum_altitude

    @property
    def maximum_aspect_ratio(self) -> float:
        return float(np.max(self.triangle_aspect_ratios))

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.vertex_mass_kg))


@dataclass(frozen=True)
class StitchConstraintBatch:
    seam_id: str
    vertex_pairs: np.ndarray
    physical_stitch_spacing_m: float
    numerical_spacing_m: float
    per_constraint_stiffness_n_m: float
    per_constraint_damping_n_s_m: float
    rest_length_m: float

    def __post_init__(self) -> None:
        if not self.seam_id:
            raise ValueError("Stitch batches require a seam identifier")
        if self.vertex_pairs.ndim != 2 or self.vertex_pairs.shape[1] != 2 or len(self.vertex_pairs) < 2:
            raise ValueError("Stitch pairs must have shape (count >= 2, 2)")
        if np.min(self.vertex_pairs) < 0:
            raise ValueError("Stitch pairs cannot contain negative vertex indices")
        if min(
            self.physical_stitch_spacing_m,
            self.numerical_spacing_m,
            self.per_constraint_stiffness_n_m,
        ) <= 0.0:
            raise ValueError("Stitch spacing and stiffness must be positive")
        if self.per_constraint_damping_n_s_m < 0.0 or self.rest_length_m < 0.0:
            raise ValueError("Stitch damping/rest length cannot be negative")
        if np.any(self.vertex_pairs[:, 0] == self.vertex_pairs[:, 1]):
            raise ValueError("A stitch cannot connect a particle to itself")


@dataclass(frozen=True)
class GarmentPatternMesh:
    construction_id: str
    positions_m: np.ndarray
    material_coordinates_m: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    vertex_area_m2: np.ndarray
    vertex_mass_kg: np.ndarray
    vertex_panel_index: np.ndarray
    triangle_panel_index: np.ndarray
    triangle_material_index: np.ndarray
    panel_ids: tuple[str, ...]
    material_profile_ids: tuple[str, ...]
    panel_vertex_ranges: Mapping[str, tuple[int, int]]
    path_vertices: Mapping[tuple[str, str], np.ndarray]
    path_fractions: Mapping[tuple[str, str], np.ndarray]
    stitches: tuple[StitchConstraintBatch, ...]
    panel_meshes: tuple[PanelMesh, ...]
    atlas_gap_m: float

    def __post_init__(self) -> None:
        vertex_count = len(self.positions_m)
        triangle_count = len(self.triangles)
        if self.positions_m.shape != (vertex_count, 3):
            raise ValueError("Garment positions must have shape (vertices, 3)")
        if self.material_coordinates_m.shape != (vertex_count, 2) or self.uv.shape != (vertex_count, 2):
            raise ValueError("Garment material/UV arrays have the wrong shape")
        if self.triangles.shape != (triangle_count, 3):
            raise ValueError("Garment triangle array has the wrong shape")
        if self.vertex_area_m2.shape != (vertex_count,) or self.vertex_mass_kg.shape != (vertex_count,):
            raise ValueError("Garment vertex scalar arrays have the wrong shape")
        if self.vertex_panel_index.shape != (vertex_count,):
            raise ValueError("Garment vertex panel index has the wrong shape")
        if self.triangle_panel_index.shape != (triangle_count,) or self.triangle_material_index.shape != (triangle_count,):
            raise ValueError("Garment triangle metadata arrays have the wrong shape")
        if len(self.panel_ids) != len(self.panel_meshes) or set(self.panel_ids) != set(self.panel_vertex_ranges):
            raise ValueError("Garment panel metadata is inconsistent")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= vertex_count:
            raise ValueError("Garment triangle index lies outside the vertex array")
        if self.atlas_gap_m <= 0.0:
            raise ValueError("Pattern atlas gap must be positive")
        for stitch in self.stitches:
            if np.max(stitch.vertex_pairs) >= vertex_count:
                raise ValueError(f"Stitch {stitch.seam_id!r} references a missing vertex")

    @property
    def vertex_count(self) -> int:
        return int(len(self.positions_m))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.vertex_mass_kg))

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.vertex_area_m2))


def _sample_path(path: PatternPath, spacing_m: float) -> tuple[tuple[float, float], ...]:
    """Subdivide every source segment so curve-control samples are retained."""

    samples: list[tuple[float, float]] = [path.points_m[0]]
    for start, end in zip(path.points_m[:-1], path.points_m[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        subdivisions = max(1, int(math.ceil(length / spacing_m)))
        for index in range(1, subdivisions + 1):
            fraction = index / subdivisions
            samples.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return tuple(samples)


def _path_sample_fractions(points: tuple[tuple[float, float], ...]) -> np.ndarray:
    lengths = np.asarray(
        [
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points[:-1], points[1:])
        ],
        dtype=np.float64,
    )
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(lengths)))
    cumulative /= cumulative[-1]
    cumulative[0] = 0.0
    cumulative[-1] = 1.0
    return cumulative


def _sample_path_with_required_fractions(
    path: PatternPath,
    spacing_m: float,
    required_fractions: tuple[float, ...] = (),
) -> tuple[tuple[tuple[float, float], ...], np.ndarray]:
    default_points = _sample_path(path, spacing_m)
    default_fractions = _path_sample_fractions(default_points)
    merged = {
        round(float(fraction), 12)
        for fraction in (
            required_fractions
            if required_fractions
            else default_fractions.tolist()
        )
    }
    merged.update((0.0, 1.0))
    retained_points: list[tuple[float, float]] = []
    retained_fractions: list[float] = []
    retained_keys: dict[tuple[int, int], int] = {}
    for fraction in sorted(merged):
        point = path.point_at_fraction(float(fraction))
        key = _quantized_key(point)
        existing = retained_keys.get(key)
        if existing is None:
            retained_keys[key] = len(retained_points)
            retained_points.append(point)
            retained_fractions.append(float(fraction))
        elif fraction == 1.0:
            # Endpoint identity is more important than a numerically adjacent
            # propagated fraction when coordinates quantize to one vertex.
            retained_points[existing] = path.points_m[-1]
            retained_fractions[existing] = 1.0
    order = np.argsort(retained_fractions)
    points = tuple(retained_points[int(index)] for index in order)
    fractions = np.asarray(
        [retained_fractions[int(index)] for index in order],
        dtype=np.float64,
    )
    return points, fractions


def _synchronized_path_fractions(
    construction: GarmentConstruction,
    target_edge_length_m: float,
    additional_fractions: Mapping[tuple[str, str], tuple[float, ...]] | None = None,
) -> Mapping[tuple[str, str], tuple[float, ...]]:
    """Propagate every seam-side discretization to its paired path.

    The physical sewn topology can only be watertight if both sides of a seam
    have identical correspondence samples.  Source curve-control vertices,
    regular mesh samples, partial seams, and overlapping construction seams
    are therefore propagated to a fixed point over the seam graph.
    """

    fractions: dict[tuple[str, str], set[float]] = {}
    for panel in construction.panels:
        for path in panel.outline_paths + panel.internal_paths:
            defaults = _path_sample_fractions(_sample_path(path, target_edge_length_m))
            fractions[(panel.id, path.id)] = {
                round(float(value), 12) for value in defaults
            }
        # A planar straight-line graph must be noded wherever two named paths
        # cross.  Without this shared sample, constrained edge recovery is
        # mathematically impossible (one protected edge would pass through
        # another), and stitched counterparts would discretize the crossing
        # differently.  Seed the intersection fractions before the seam-graph
        # fixed point so correspondence propagation includes them.
        paths = panel.outline_paths + panel.internal_paths
        for first_index, first_path in enumerate(paths):
            for second_path in paths[first_index + 1 :]:
                for first_start, first_end in zip(
                    first_path.points_m[:-1], first_path.points_m[1:]
                ):
                    for second_start, second_end in zip(
                        second_path.points_m[:-1], second_path.points_m[1:]
                    ):
                        intersection = _proper_segment_intersection_point(
                            np.asarray(first_start, dtype=np.float64),
                            np.asarray(first_end, dtype=np.float64),
                            np.asarray(second_start, dtype=np.float64),
                            np.asarray(second_end, dtype=np.float64),
                        )
                        if intersection is None:
                            continue
                        point = (float(intersection[0]), float(intersection[1]))
                        first_fraction = _point_fraction_on_path(point, first_path)
                        second_fraction = _point_fraction_on_path(point, second_path)
                        if first_fraction is None or second_fraction is None:
                            raise RuntimeError(
                                "Named-path intersection could not be mapped to arclength"
                            )
                        fractions[(panel.id, first_path.id)].add(
                            round(first_fraction, 12)
                        )
                        fractions[(panel.id, second_path.id)].add(
                            round(second_fraction, 12)
                        )
    if additional_fractions:
        unknown = set(additional_fractions) - set(fractions)
        if unknown:
            raise ValueError(f"Additional samples reference unknown paths: {sorted(unknown)!r}")
        for key, values in additional_fractions.items():
            fractions[key].update(round(float(value), 12) for value in values)

    def reference_t_values(path_ref: PathRef) -> set[float]:
        key = (path_ref.panel_id, path_ref.path_id)
        span = path_ref.end_fraction - path_ref.start_fraction
        result: set[float] = set()
        for fraction in fractions[key]:
            if path_ref.start_fraction - 1.0e-12 <= fraction <= path_ref.end_fraction + 1.0e-12:
                t = (fraction - path_ref.start_fraction) / span
                if path_ref.reverse:
                    t = 1.0 - t
                result.add(round(min(1.0, max(0.0, t)), 12))
        return result

    def fraction_from_t(path_ref: PathRef, t: float) -> float:
        if path_ref.reverse:
            t = 1.0 - t
        return path_ref.start_fraction + t * (
            path_ref.end_fraction - path_ref.start_fraction
        )

    def source_reference_t_values(path_ref: PathRef) -> set[float]:
        """Return authored polyline-vertex locations in seam coordinates."""

        path = construction.panel_map[path_ref.panel_id].path(path_ref.path_id)
        source_fractions = _path_sample_fractions(path.points_m)
        span = path_ref.end_fraction - path_ref.start_fraction
        result: set[float] = set()
        for fraction in source_fractions:
            if (
                path_ref.start_fraction - 1.0e-12
                <= fraction
                <= path_ref.end_fraction + 1.0e-12
            ):
                t = (float(fraction) - path_ref.start_fraction) / span
                if path_ref.reverse:
                    t = 1.0 - t
                result.add(round(min(1.0, max(0.0, t)), 12))
        return result

    def references_share_source_geometry(
        first_ref: PathRef,
        second_ref: PathRef,
    ) -> bool:
        """Whether both full seam refs trace the same authored polyline."""

        if not (
            first_ref.start_fraction == 0.0
            and first_ref.end_fraction == 1.0
            and second_ref.start_fraction == 0.0
            and second_ref.end_fraction == 1.0
        ):
            return False
        first = construction.panel_map[first_ref.panel_id].path(
            first_ref.path_id
        ).points_m
        second = construction.panel_map[second_ref.panel_id].path(
            second_ref.path_id
        ).points_m
        if first_ref.reverse:
            first = tuple(reversed(first))
        if second_ref.reverse:
            second = tuple(reversed(second))
        return len(first) == len(second) and all(
            math.hypot(a[0] - b[0], a[1] - b[1]) <= 1.0e-12
            for a, b in zip(first, second)
        )

    def quality_spaced_correspondence(
        values: set[float],
        minimum_t_spacing: float,
        protected: set[float],
    ) -> set[float]:
        retained = set(protected)
        for value in sorted(values - protected):
            if all(abs(value - existing) >= minimum_t_spacing for existing in retained):
                retained.add(value)
        return retained

    maximum_iterations = max(1, 2 * len(construction.seams) + 2)
    for _ in range(maximum_iterations):
        changed = False
        for seam in construction.seams:
            mean_length = 0.5 * (
                seam.side_a.length_m(construction.panel_map)
                + seam.side_b.length_m(construction.panel_map)
            )
            count = max(2, int(math.ceil(mean_length / target_edge_length_m)) + 1)
            correspondence = reference_t_values(seam.side_a) | reference_t_values(seam.side_b)
            uniform_correspondence = {
                round(index / (count - 1), 12) for index in range(count)
            }
            correspondence.update(uniform_correspondence)
            source_correspondence = set()
            if references_share_source_geometry(seam.side_a, seam.side_b):
                source_correspondence = (
                    source_reference_t_values(seam.side_a)
                    & source_reference_t_values(seam.side_b)
                )
            correspondence.update(source_correspondence)
            minimum_side_length = min(
                seam.side_a.length_m(construction.panel_map),
                seam.side_b.length_m(construction.panel_map),
            )
            correspondence = quality_spaced_correspondence(
                correspondence,
                min(0.20, 0.15 * target_edge_length_m / minimum_side_length),
                source_correspondence or uniform_correspondence,
            )
            for path_ref in (seam.side_a, seam.side_b):
                key = (path_ref.panel_id, path_ref.path_id)
                before = fractions[key]
                outside_interval = {
                    fraction
                    for fraction in before
                    if fraction < path_ref.start_fraction - 1.0e-12
                    or fraction > path_ref.end_fraction + 1.0e-12
                }
                replacement = {
                    round(fraction_from_t(path_ref, t), 12)
                    for t in correspondence
                }
                updated = outside_interval | replacement
                changed = changed or updated != before
                fractions[key] = updated
        if not changed:
            break
    else:
        raise ValueError("Seam discretization did not converge to a correspondence fixed point")
    return MappingProxyType(
        {key: tuple(sorted(values)) for key, values in fractions.items()}
    )


def _quantized_key(point: tuple[float, float], tolerance_m: float = 1.0e-10) -> tuple[int, int]:
    return round(point[0] / tolerance_m), round(point[1] / tolerance_m)


def _point_fraction_on_path(
    point: tuple[float, float],
    path: PatternPath,
    *,
    tolerance_m: float = 1.0e-9,
) -> float | None:
    traversed = 0.0
    for start, end in zip(path.points_m[:-1], path.points_m[1:]):
        delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length_squared = float(np.dot(delta, delta))
        relative = np.asarray(point, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        projection = float(np.dot(relative, delta) / length_squared)
        if -1.0e-10 <= projection <= 1.0 + 1.0e-10:
            projection = min(1.0, max(0.0, projection))
            closest = np.asarray(start, dtype=np.float64) + projection * delta
            if float(np.linalg.norm(closest - np.asarray(point))) <= tolerance_m:
                segment_length = math.sqrt(length_squared)
                return (traversed + projection * segment_length) / path.length_m
        traversed += math.sqrt(length_squared)
    return None


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    point_array = np.asarray(point, dtype=np.float64)
    start_array = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start_array
    projection = float(
        np.dot(point_array - start_array, delta) / np.dot(delta, delta)
    )
    projection = min(1.0, max(0.0, projection))
    closest = start_array + projection * delta
    return float(np.linalg.norm(point_array - closest))


def _oriented_triangle(
    a: int,
    b: int,
    c: int,
    coordinates: np.ndarray,
) -> tuple[int, int, int]:
    first, second, third = coordinates[[a, b, c]]
    twice_area = (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )
    if abs(twice_area) <= 2.0 * _AREA_EPSILON_M2:
        raise ValueError("Point insertion would create a degenerate garment triangle")
    return (a, b, c) if twice_area > 0.0 else (a, c, b)


def _triangle_aspect_ratio(
    triangle: tuple[int, int, int],
    coordinates: np.ndarray,
) -> float:
    """Return longest-edge over shortest-altitude for one material face."""

    points = coordinates[np.asarray(triangle, dtype=np.int32)]
    edge_lengths = np.asarray(
        (
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ),
        dtype=np.float64,
    )
    twice_area = abs(
        float(
            (points[1, 0] - points[0, 0])
            * (points[2, 1] - points[0, 1])
            - (points[1, 1] - points[0, 1])
            * (points[2, 0] - points[0, 0])
        )
    )
    if twice_area <= 2.0 * _AREA_EPSILON_M2:
        return math.inf
    maximum_edge = float(np.max(edge_lengths))
    return maximum_edge * maximum_edge / twice_area


def _improve_triangulation_quality(
    triangles: list[tuple[int, int, int]],
    coordinates: np.ndarray,
    protected_edges: set[tuple[int, int]],
) -> None:
    """Flip poor free diagonals while preserving the named path graph."""

    maximum_passes = max(8, 2 * len(triangles))
    for _pass in range(maximum_passes):
        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for triangle_index, triangle in enumerate(triangles):
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge_to_triangles.setdefault(
                    (min(first, second), max(first, second)), []
                ).append(triangle_index)
        candidates: list[tuple[float, tuple[int, int], int, int]] = []
        for edge, adjacent in edge_to_triangles.items():
            if edge in protected_edges or len(adjacent) != 2:
                continue
            first_triangle, second_triangle = adjacent
            current_worst = max(
                _triangle_aspect_ratio(triangles[first_triangle], coordinates),
                _triangle_aspect_ratio(triangles[second_triangle], coordinates),
            )
            candidates.append(
                (-current_worst, edge, first_triangle, second_triangle)
            )
        improved = False
        for _negative_worst, edge, first_triangle, second_triangle in sorted(
            candidates
        ):
            current_faces = (
                triangles[first_triangle], triangles[second_triangle]
            )
            if not all(edge[0] in face and edge[1] in face for face in current_faces):
                continue
            opposite_first = next(
                vertex for vertex in current_faces[0] if vertex not in edge
            )
            opposite_second = next(
                vertex for vertex in current_faces[1] if vertex not in edge
            )
            replacement = (
                min(opposite_first, opposite_second),
                max(opposite_first, opposite_second),
            )
            if replacement in edge_to_triangles or replacement in protected_edges:
                continue
            if not _segments_properly_intersect(
                coordinates[edge[0]],
                coordinates[edge[1]],
                coordinates[opposite_first],
                coordinates[opposite_second],
            ):
                continue
            try:
                proposed_first = _oriented_triangle(
                    opposite_first,
                    opposite_second,
                    edge[0],
                    coordinates,
                )
                proposed_second = _oriented_triangle(
                    opposite_second,
                    opposite_first,
                    edge[1],
                    coordinates,
                )
            except ValueError:
                continue
            current_worst = max(
                _triangle_aspect_ratio(current_faces[0], coordinates),
                _triangle_aspect_ratio(current_faces[1], coordinates),
            )
            proposed_worst = max(
                _triangle_aspect_ratio(proposed_first, coordinates),
                _triangle_aspect_ratio(proposed_second, coordinates),
            )
            if proposed_worst >= current_worst * (1.0 - 1.0e-12):
                continue
            triangles[first_triangle] = proposed_first
            triangles[second_triangle] = proposed_second
            improved = True
            break
        if not improved:
            return
    raise ValueError("Garment mesh quality flips did not converge")


def _insert_point_into_triangulation(
    triangles: list[tuple[int, int, int]],
    point_index: int,
    coordinates: np.ndarray,
) -> None:
    """Constrained Bowyer-Watson insertion into an existing panel CDT."""

    point = coordinates[point_index]
    containment_tolerance = 2.0e-12

    # Named internal paths commonly terminate on a sampled outline segment.
    # Shapely is free to omit that collinear point from the polygon ring, so
    # insert it by splitting the existing edge instead of asking a Delaunay
    # cavity routine to create a zero-area triangle on the edge.
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, (a, b, c) in enumerate(triangles):
        for first, second in ((a, b), (b, c), (c, a)):
            edge_to_triangles.setdefault(
                (min(first, second), max(first, second)), []
            ).append(triangle_index)
    for edge, adjacent in sorted(edge_to_triangles.items()):
        start, end = coordinates[list(edge)]
        delta = end - start
        length_squared = float(np.dot(delta, delta))
        projection = float(np.dot(point - start, delta) / length_squared)
        if not 1.0e-10 < projection < 1.0 - 1.0e-10:
            continue
        closest = start + projection * delta
        if float(np.linalg.norm(point - closest)) > 1.0e-10:
            continue
        retained = [
            triangle
            for triangle_index, triangle in enumerate(triangles)
            if triangle_index not in adjacent
        ]
        for triangle_index in adjacent:
            opposite = next(
                vertex
                for vertex in triangles[triangle_index]
                if vertex not in edge
            )
            retained.extend(
                (
                    _oriented_triangle(
                        edge[0], point_index, opposite, coordinates
                    ),
                    _oriented_triangle(
                        point_index, edge[1], opposite, coordinates
                    ),
                )
            )
        triangles[:] = retained
        return

    def triangle_crosses(triangle: tuple[int, int, int]) -> tuple[float, float, float]:
        a, b, c = coordinates[list(triangle)]
        return (
            (b[0] - a[0]) * (point[1] - a[1])
            - (b[1] - a[1]) * (point[0] - a[0]),
            (c[0] - b[0]) * (point[1] - b[1])
            - (c[1] - b[1]) * (point[0] - b[0]),
            (a[0] - c[0]) * (point[1] - c[1])
            - (a[1] - c[1]) * (point[0] - c[0]),
        )

    containing = [
        triangle_index
        for triangle_index, triangle in enumerate(triangles)
        if min(triangle_crosses(triangle)) >= -containment_tolerance
    ]
    if not containing:
        raise ValueError("A named/interior panel point could not be located in its polygon mesh")

    edge_to_triangles = {}
    for triangle_index, (a, b, c) in enumerate(triangles):
        for first, second in ((a, b), (b, c), (c, a)):
            edge_to_triangles.setdefault(
                (min(first, second), max(first, second)), []
            ).append(triangle_index)

    def in_circumcircle(triangle_index: int) -> bool:
        a, b, c = coordinates[list(triangles[triangle_index])] - point
        determinant = (
            (a[0] * a[0] + a[1] * a[1]) * (b[0] * c[1] - b[1] * c[0])
            - (b[0] * b[0] + b[1] * b[1]) * (a[0] * c[1] - a[1] * c[0])
            + (c[0] * c[0] + c[1] * c[1]) * (a[0] * b[1] - a[1] * b[0])
        )
        scale = max(
            float(np.max(np.abs((a, b, c)))),
            1.0e-6,
        )
        return determinant >= -1.0e-12 * scale**4

    # Traverse only the connected cavity.  This matters for a constrained
    # triangulation of a concave panel: a geometrically overlapping
    # circumcircle across a boundary indentation is not reachable through the
    # panel's triangle adjacency and must not enter the cavity.
    cavity: set[int] = set(containing)
    queue = list(containing)
    while queue:
        triangle_index = queue.pop()
        a, b, c = triangles[triangle_index]
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            for neighbour in edge_to_triangles[edge]:
                if neighbour in cavity:
                    continue
                if in_circumcircle(neighbour):
                    cavity.add(neighbour)
                    queue.append(neighbour)

    cavity_edge_count: dict[tuple[int, int], int] = {}
    for triangle_index in cavity:
        a, b, c = triangles[triangle_index]
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            cavity_edge_count[edge] = cavity_edge_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in cavity_edge_count.items() if count == 1]
    if len(boundary_edges) < 3:
        raise ValueError("Interior-point Delaunay cavity has fewer than three boundary edges")

    retained = [
        triangle
        for triangle_index, triangle in enumerate(triangles)
        if triangle_index not in cavity
    ]
    for first, second in boundary_edges:
        retained.append(
            _oriented_triangle(first, second, point_index, coordinates)
        )
    triangles[:] = retained


def _segments_properly_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float = 1.0e-13,
) -> bool:
    """Return whether two open segments cross away from their endpoints."""

    return _proper_segment_intersection_point(
        first_start,
        first_end,
        second_start,
        second_end,
        tolerance=tolerance,
    ) is not None


def _proper_segment_intersection_point(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float = 1.0e-13,
) -> np.ndarray | None:
    """Return one strict segment intersection, excluding shared endpoints."""

    first_delta = first_end - first_start
    second_delta = second_end - second_start

    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    denominator = cross(first_delta, second_delta)
    scale = max(
        float(np.linalg.norm(first_delta) * np.linalg.norm(second_delta)),
        1.0e-12,
    )
    if abs(denominator) <= tolerance * scale:
        return None
    relative = second_start - first_start
    first_fraction = cross(relative, second_delta) / denominator
    second_fraction = cross(relative, first_delta) / denominator
    fractional_tolerance = tolerance / max(
        min(float(np.linalg.norm(first_delta)), float(np.linalg.norm(second_delta))),
        1.0e-12,
    )
    if not (
        fractional_tolerance < first_fraction < 1.0 - fractional_tolerance
        and fractional_tolerance < second_fraction < 1.0 - fractional_tolerance
    ):
        return None
    return first_start + first_fraction * first_delta


def _recover_constraint_edge(
    triangles: list[tuple[int, int, int]],
    first: int,
    second: int,
    coordinates: np.ndarray,
    protected_edges: set[tuple[int, int]],
) -> None:
    """Recover one planar straight-line-graph edge by diagonal flips.

    All named path vertices have already been inserted.  A segment that is
    absent from the unconstrained point triangulation is crossed by a finite
    chain of interior diagonals.  Flipping a convex crossed diagonal to the
    other diagonal shortens that chain; previously recovered path edges are
    never touched.  This is the standard constrained-triangulation edge
    recovery operation, applied deterministically in sorted edge order.
    """

    target = (min(first, second), max(first, second))
    if target[0] == target[1]:
        raise ValueError("A constrained garment edge cannot collapse")
    maximum_flips = max(32, 8 * len(triangles))
    target_start = coordinates[first]
    target_end = coordinates[second]

    for _ in range(maximum_flips):
        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for triangle_index, (a, b, c) in enumerate(triangles):
            for edge_start, edge_end in ((a, b), (b, c), (c, a)):
                edge_to_triangles.setdefault(
                    (min(edge_start, edge_end), max(edge_start, edge_end)), []
                ).append(triangle_index)
        if target in edge_to_triangles:
            return

        crossing_edges = [
            edge
            for edge, adjacent in edge_to_triangles.items()
            if edge not in protected_edges
            and len(adjacent) == 2
            and first not in edge
            and second not in edge
            and _segments_properly_intersect(
                target_start,
                target_end,
                coordinates[edge[0]],
                coordinates[edge[1]],
            )
        ]
        flipped = False
        deferred: tuple[
            tuple[int, int], tuple[int, int], int, int
        ] | None = None
        for edge in sorted(crossing_edges):
            adjacent = edge_to_triangles[edge]
            first_triangle, second_triangle = adjacent
            opposite_first = next(
                vertex
                for vertex in triangles[first_triangle]
                if vertex not in edge
            )
            opposite_second = next(
                vertex
                for vertex in triangles[second_triangle]
                if vertex not in edge
            )
            replacement_edge = (
                min(opposite_first, opposite_second),
                max(opposite_first, opposite_second),
            )
            if replacement_edge in edge_to_triangles:
                continue
            # The old and new diagonals cross iff the two incident triangles
            # form a convex quadrilateral.  Concave flips would overlap faces.
            if not _segments_properly_intersect(
                coordinates[edge[0]],
                coordinates[edge[1]],
                coordinates[opposite_first],
                coordinates[opposite_second],
            ):
                continue
            # A sampled path can pass through a collinear grid vertex that is
            # not part of this particular crossed edge.  Flipping through it
            # would create a zero-area face; skip that diagonal and recover
            # the constraint through another convex edge instead.
            try:
                _oriented_triangle(
                    opposite_first,
                    opposite_second,
                    edge[0],
                    coordinates,
                )
                _oriented_triangle(
                    opposite_second,
                    opposite_first,
                    edge[1],
                    coordinates,
                )
            except ValueError:
                continue
            candidate = (
                edge,
                replacement_edge,
                first_triangle,
                second_triangle,
            )
            replacement_crosses_target = _segments_properly_intersect(
                target_start,
                target_end,
                coordinates[opposite_first],
                coordinates[opposite_second],
            )
            if replacement_crosses_target:
                deferred = deferred or candidate
                continue
            deferred = candidate
            break

        if deferred is not None:
            edge, replacement_edge, first_triangle, second_triangle = deferred
            opposite_first, opposite_second = replacement_edge
            triangles[first_triangle] = _oriented_triangle(
                opposite_first,
                opposite_second,
                edge[0],
                coordinates,
            )
            triangles[second_triangle] = _oriented_triangle(
                opposite_second,
                opposite_first,
                edge[1],
                coordinates,
            )
            flipped = True
        if not flipped:
            raise ValueError(
                "Constrained garment triangulation could not recover named "
                f"path edge ({first}, {second})"
            )
    raise ValueError(
        "Constrained garment triangulation exceeded its edge-recovery flip budget"
    )


def triangulate_panel(
    panel: PanelPattern,
    *,
    target_edge_length_m: float,
    required_path_fractions: Mapping[str, tuple[float, ...]] | None = None,
    additional_points_m: tuple[tuple[float, float], ...] = (),
) -> PanelMesh:
    """Triangulate one simple panel while preserving all named path samples."""

    if not math.isfinite(target_edge_length_m) or target_edge_length_m <= 0.0:
        raise ValueError("Target edge length must be finite and positive")
    try:
        import shapely
        from shapely.geometry import Polygon
    except ImportError as exception:  # pragma: no cover - exercised in the Newton venv bootstrap audit
        raise RuntimeError(
            "Boundary-conforming garment triangulation requires Shapely 2.1+; run "
            "laundry/bootstrap_newton.ps1 or install Shapely>=2.1,<3."
        ) from exception
    if not hasattr(shapely, "constrained_delaunay_triangles"):
        raise RuntimeError("Garment triangulation requires Shapely 2.1+ constrained Delaunay support")

    points: list[tuple[float, float]] = []
    point_lookup: dict[tuple[int, int], int] = {}

    def add_point(point: tuple[float, float]) -> int:
        key = _quantized_key(point)
        existing = point_lookup.get(key)
        if existing is not None:
            return existing
        index = len(points)
        points.append((float(point[0]), float(point[1])))
        point_lookup[key] = index
        return index

    required_path_fractions = required_path_fractions or {}
    unknown_required_paths = set(required_path_fractions) - {
        path.id for path in panel.outline_paths + panel.internal_paths
    }
    if unknown_required_paths:
        raise ValueError(
            f"Panel {panel.id!r} received samples for unknown paths {sorted(unknown_required_paths)!r}"
        )
    path_indices: dict[str, np.ndarray] = {}
    path_fractions: dict[str, np.ndarray] = {}
    sampled_paths: dict[str, tuple[tuple[float, float], ...]] = {}
    all_paths = panel.outline_paths + panel.internal_paths
    for path in all_paths:
        sampled, sampled_fractions = _sample_path_with_required_fractions(
            path,
            target_edge_length_m,
            required_path_fractions.get(path.id, ()),
        )
        indices = np.asarray([add_point(point) for point in sampled], dtype=np.int32)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Path {panel.id!r}:{path.id!r} folds onto the same sampled vertex")
        path_indices[path.id] = indices
        path_fractions[path.id] = sampled_fractions
        sampled_paths[path.id] = sampled

    for point in additional_points_m:
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError(f"Panel {panel.id!r} received a non-finite guard point")
        if not point_in_polygon(point, panel.outline_polygon):
            raise ValueError(f"Panel {panel.id!r} closure guard lies outside its outline")
        add_point(point)

    polygon = panel.outline_polygon
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    x_values = np.arange(min_x + 0.5 * target_edge_length_m, max_x, target_edge_length_m)
    y_values = np.arange(min_y + 0.5 * target_edge_length_m, max_y, target_edge_length_m)
    minimum_separation_squared = (0.20 * target_edge_length_m) ** 2
    protected_segments = tuple(
        (start, end)
        for path in all_paths
        for start, end in zip(path.points_m[:-1], path.points_m[1:])
    )
    for y in y_values:
        for x in x_values:
            candidate = (float(x), float(y))
            if not point_in_polygon(candidate, polygon, include_boundary=False):
                continue
            if any(
                _point_segment_distance(candidate, start, end)
                < 0.20 * target_edge_length_m
                for start, end in protected_segments
            ):
                continue
            if any(
                (candidate[0] - existing[0]) ** 2 + (candidate[1] - existing[1]) ** 2
                < minimum_separation_squared
                for existing in points
            ):
                continue
            add_point(candidate)

    coordinates = np.asarray(points, dtype=np.float64)
    boundary_ring: list[tuple[float, float]] = []
    for path in panel.outline_paths:
        boundary_ring.extend(sampled_paths[path.id][:-1])
    boundary_polygon = Polygon(boundary_ring)
    if not boundary_polygon.is_valid:
        raise ValueError(f"Panel {panel.id!r} produced an invalid sampled boundary polygon")
    boundary_area_error = abs(float(boundary_polygon.area) - panel.area_m2) / panel.area_m2
    if boundary_area_error > 5.0e-4:
        raise ValueError(
            f"Panel {panel.id!r} synchronized boundary area error "
            f"{boundary_area_error:.3e} exceeds the mesh audit tolerance"
        )
    triangle_geometry = shapely.constrained_delaunay_triangles(boundary_polygon)
    retained: list[tuple[int, int, int]] = []
    for triangle_polygon in triangle_geometry.geoms:
        triangle_coordinates = tuple(triangle_polygon.exterior.coords)[:-1]
        if len(triangle_coordinates) != 3:
            raise ValueError("Constrained Delaunay output contained a non-triangle face")
        try:
            indices = tuple(
                point_lookup[_quantized_key((float(x), float(y)))]
                for x, y in triangle_coordinates
            )
        except KeyError as exception:
            raise ValueError("Constrained triangulation introduced an unknown panel vertex") from exception
        retained.append(_oriented_triangle(*indices, coordinates))
    if not retained:
        raise ValueError(f"Panel {panel.id!r} produced no interior triangles")

    used_boundary_vertices = set(vertex for triangle in retained for vertex in triangle)
    expected_boundary_vertices = {
        int(index)
        for path in panel.outline_paths
        for index in path_indices[path.id]
    }
    if used_boundary_vertices != expected_boundary_vertices:
        missing = sorted(expected_boundary_vertices - used_boundary_vertices)
        raise ValueError(
            f"Panel {panel.id!r} constrained triangulation dropped {len(missing)} boundary vertices"
        )
    for point_index in range(len(coordinates)):
        if point_index in used_boundary_vertices:
            continue
        try:
            _insert_point_into_triangulation(retained, point_index, coordinates)
        except ValueError as exception:
            raise ValueError(
                f"Panel {panel.id!r} failed to insert mesh point {point_index} "
                f"at {coordinates[point_index].tolist()}: {exception}"
            ) from exception

    # Shapely constrains polygon rings, but open internal stitch and fold paths
    # are a planar straight-line graph inside that ring.  Merely inserting
    # their samples permits triangles to bridge across a seam (and previously
    # caused sewn overlays and sleeve folds to intersect in 3-D).  Recover all
    # named path segments as actual mesh edges before publishing the topology.
    edge_incidence: dict[tuple[int, int], int] = {}
    for triangle in retained:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = (min(first, second), max(first, second))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
    # Point-on-edge insertion can legitimately split one sampled outline
    # segment at an internal-path endpoint.  Protect the actual boundary chain
    # rather than requiring the now-subdivided original diagonal.
    protected_edges: set[tuple[int, int]] = {
        edge for edge, incidence in edge_incidence.items() if incidence == 1
    }
    recovered_internal_edges: set[tuple[int, int]] = set()
    for path in panel.internal_paths:
        indices = path_indices[path.id]
        for first_raw, second_raw in zip(indices[:-1], indices[1:]):
            first, second = int(first_raw), int(second_raw)
            start, end = coordinates[[first, second]]
            delta = end - start
            length_squared = float(np.dot(delta, delta))
            chain: list[tuple[float, int]] = [(0.0, first), (1.0, second)]
            for vertex, point in enumerate(coordinates):
                if vertex in (first, second):
                    continue
                fraction = float(np.dot(point - start, delta) / length_squared)
                if not 1.0e-10 < fraction < 1.0 - 1.0e-10:
                    continue
                closest = start + fraction * delta
                if float(np.linalg.norm(point - closest)) <= 1.0e-10:
                    chain.append((fraction, vertex))
            chain.sort()
            for (_, chain_first), (_, chain_second) in zip(chain[:-1], chain[1:]):
                try:
                    _recover_constraint_edge(
                        retained,
                        chain_first,
                        chain_second,
                        coordinates,
                        protected_edges,
                    )
                except ValueError as exception:
                    raise ValueError(
                        f"Panel {panel.id!r} could not constrain internal path "
                        f"{path.id!r} between {coordinates[chain_first].tolist()} "
                        f"and {coordinates[chain_second].tolist()}: {exception}"
                    ) from exception
                edge = (
                    min(chain_first, chain_second),
                    max(chain_first, chain_second),
                )
                protected_edges.add(edge)
                recovered_internal_edges.add(edge)

    _improve_triangulation_quality(
        retained,
        coordinates,
        protected_edges,
    )
    published_edges = {
        (min(first, second), max(first, second))
        for triangle in retained
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    }
    missing_constraints = recovered_internal_edges - published_edges
    if missing_constraints:
        raise ValueError(
            f"Panel {panel.id!r} lost {len(missing_constraints)} constrained path edges"
        )
    triangles = np.asarray(retained, dtype=np.int32)

    used = np.unique(triangles.reshape(-1))
    remap = np.full(len(coordinates), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    coordinates = coordinates[used]
    triangles = remap[triangles]
    for path_id, indices in tuple(path_indices.items()):
        remapped = remap[indices]
        if np.any(remapped < 0):
            raise ValueError(f"Triangulation discarded a named vertex on {panel.id!r}:{path_id!r}")
        path_indices[path_id] = remapped

    a = coordinates[triangles[:, 0]]
    b = coordinates[triangles[:, 1]]
    c = coordinates[triangles[:, 2]]
    triangle_areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    vertex_area = np.zeros(len(coordinates), dtype=np.float64)
    np.add.at(vertex_area, triangles[:, 0], triangle_areas / 3.0)
    np.add.at(vertex_area, triangles[:, 1], triangle_areas / 3.0)
    np.add.at(vertex_area, triangles[:, 2], triangle_areas / 3.0)
    # Synchronized seam samples can move a polyline corner by a sub-millimetre
    # to avoid pathological elements.  Preserve the authored cut-panel mass
    # exactly while publishing the geometric area error above.
    vertex_area *= panel.area_m2 / float(np.sum(vertex_area))
    vertex_mass = vertex_area * panel.areal_density_kg_m2
    span = np.ptp(coordinates, axis=0)
    if np.min(span) <= 0.0:
        raise ValueError(f"Panel {panel.id!r} has a collapsed material extent")
    uv = (coordinates - np.min(coordinates, axis=0)) / span

    result = PanelMesh(
        panel_id=panel.id,
        material_coordinates_m=coordinates,
        triangles=triangles,
        vertex_area_m2=vertex_area,
        vertex_mass_kg=vertex_mass,
        uv=uv,
        path_vertices=MappingProxyType(path_indices),
        path_fractions=MappingProxyType(path_fractions),
        target_edge_length_m=target_edge_length_m,
        source_area_m2=panel.area_m2,
    )
    if result.area_error_fraction > 5.0e-4:
        raise ValueError(
            f"Panel {panel.id!r} triangulation area error {result.area_error_fraction:.3e} "
            "exceeds the construction audit tolerance"
        )
    return result


def _triangulate_without_unintended_stitch_chords(
    panel: PanelPattern,
    *,
    target_edge_length_m: float,
    required_path_fractions: Mapping[str, tuple[float, ...]],
    stitch_references: tuple[PathRef, ...],
    additional_points_m: tuple[tuple[float, float], ...],
) -> PanelMesh:
    """Keep stitched particles from forming accidental interior cloth edges.

    Two nominally identical layers (for example, the two halves of a pocket
    bag) can choose the same triangulation chord between sewn boundary
    particles.  After seam welding that chord would have four incident faces,
    even though no four-face material hinge exists in the garment.  Boundary
    edges are intentional seam edges and remain untouched; interior chords are
    split through an unstitched material vertex.
    """

    guard_points = list(additional_points_m)
    for _iteration in range(6):
        mesh = triangulate_panel(
            panel,
            target_edge_length_m=target_edge_length_m,
            required_path_fractions=required_path_fractions,
            additional_points_m=tuple(guard_points),
        )
        stitched_vertices: set[int] = set()
        for path_ref in stitch_references:
            fractions = mesh.path_fractions[path_ref.path_id]
            indices = mesh.path_vertices[path_ref.path_id]
            mask = (
                (fractions >= path_ref.start_fraction - 1.0e-12)
                & (fractions <= path_ref.end_fraction + 1.0e-12)
            )
            stitched_vertices.update(int(index) for index in indices[mask])

        edge_incidence: dict[tuple[int, int], int] = {}
        for triangle in mesh.triangles:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = tuple(sorted((int(first), int(second))))
                edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
        unintended = tuple(
            edge
            for edge, incidence in edge_incidence.items()
            if incidence == 2
            and edge[0] in stitched_vertices
            and edge[1] in stitched_vertices
        )
        if not unintended:
            return mesh

        existing = {_quantized_key(point) for point in guard_points}
        inserted = 0
        for first, second in unintended:
            midpoint = 0.5 * (
                mesh.material_coordinates_m[first]
                + mesh.material_coordinates_m[second]
            )
            point = (float(midpoint[0]), float(midpoint[1]))
            key = _quantized_key(point)
            if key in existing:
                continue
            guard_points.append(point)
            existing.add(key)
            inserted += 1
        if inserted == 0:
            break
    raise ValueError(
        f"Panel {panel.id!r} retained an unintended interior edge between "
        "stitched particles after six topology-repair passes"
    )


def _indices_for_ref(
    path_ref: PathRef,
    *,
    global_path_vertices: Mapping[tuple[str, str], np.ndarray],
    global_path_fractions: Mapping[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (path_ref.panel_id, path_ref.path_id)
    indices = global_path_vertices[key]
    fractions = global_path_fractions[key]
    start_index = int(np.argmin(np.abs(fractions - path_ref.start_fraction)))
    end_index = int(np.argmin(np.abs(fractions - path_ref.end_fraction)))
    if end_index <= start_index:
        raise ValueError(f"Path reference {key!r} collapses at the selected mesh resolution")
    selected = indices[start_index : end_index + 1]
    return selected[::-1].copy() if path_ref.reverse else selected.copy()


def _evenly_subsample(indices: np.ndarray, count: int) -> np.ndarray:
    if count < 2 or count > len(indices):
        raise ValueError("Stitch subsampling count is incompatible with path vertices")
    selected = np.rint(np.linspace(0, len(indices) - 1, count)).astype(np.int32)
    if len(np.unique(selected)) != count:
        raise ValueError("Stitch subsampling generated duplicate path samples")
    return indices[selected]


def compile_garment_pattern_mesh(
    construction: GarmentConstruction,
    *,
    target_edge_length_m: float = 0.020,
    atlas_gap_m: float = 0.050,
    atlas_row_width_m: float = 2.4,
) -> GarmentPatternMesh:
    """Compile every panel and seam into one auditable pattern-atlas mesh."""

    if min(target_edge_length_m, atlas_gap_m, atlas_row_width_m) <= 0.0:
        raise ValueError("Mesh/atlas dimensions must be positive")
    synchronized_fractions = _synchronized_path_fractions(
        construction,
        target_edge_length_m,
    )
    closure_guard_points: dict[str, list[tuple[float, float]]] = {
        panel.id: [] for panel in construction.panels
    }
    for seam in construction.seams:
        if seam.side_a.panel_id != seam.side_b.panel_id:
            continue
        path_a = construction.panel_map[seam.side_a.panel_id].path(seam.side_a.path_id)
        path_b = construction.panel_map[seam.side_b.panel_id].path(seam.side_b.path_id)
        span_a = seam.side_a.end_fraction - seam.side_a.start_fraction
        correspondence: set[float] = set()
        for fraction in synchronized_fractions[(seam.side_a.panel_id, seam.side_a.path_id)]:
            if seam.side_a.start_fraction - 1.0e-12 <= fraction <= seam.side_a.end_fraction + 1.0e-12:
                t = (fraction - seam.side_a.start_fraction) / span_a
                if seam.side_a.reverse:
                    t = 1.0 - t
                correspondence.add(round(min(1.0, max(0.0, t)), 12))
        for t in sorted(correspondence):
            fraction_a = seam.side_a.start_fraction + (
                (1.0 - t) if seam.side_a.reverse else t
            ) * span_a
            span_b = seam.side_b.end_fraction - seam.side_b.start_fraction
            fraction_b = seam.side_b.start_fraction + (
                (1.0 - t) if seam.side_b.reverse else t
            ) * span_b
            point_a = path_a.point_at_fraction(fraction_a)
            point_b = path_b.point_at_fraction(fraction_b)
            closure_guard_points[seam.side_a.panel_id].append(
                (
                    0.5 * (point_a[0] + point_b[0]),
                    0.5 * (point_a[1] + point_b[1]),
                )
            )
    augmented_fractions = {
        key: set(values) for key, values in synchronized_fractions.items()
    }
    interior_guard_points: dict[str, list[tuple[float, float]]] = {
        panel.id: [] for panel in construction.panels
    }
    for panel in construction.panels:
        for point in closure_guard_points[panel.id]:
            boundary_match = False
            for path in panel.outline_paths:
                fraction = _point_fraction_on_path(point, path)
                if fraction is None:
                    continue
                augmented_fractions[(panel.id, path.id)].add(round(fraction, 12))
                boundary_match = True
                break
            if boundary_match:
                continue

            # A self-seam closure midpoint can land on an authored internal
            # landmark (the T-shirt sleeve fold axis is the canonical case).
            # Inserting it again as a free guard vertex beside an existing
            # path sample creates a sub-millimetre sliver even though the
            # topology already contains a suitable splitting vertex.  Snap to
            # that path discretization when it is within the same quality
            # radius used for interior sampling; otherwise add one explicit
            # path sample rather than a nearly coincident free point.
            internal_match = False
            for path in panel.internal_paths:
                fraction = _point_fraction_on_path(point, path)
                if fraction is None:
                    continue
                key = (panel.id, path.id)
                nearest_distance_m = path.length_m * min(
                    abs(fraction - existing)
                    for existing in augmented_fractions[key]
                )
                if nearest_distance_m >= 0.20 * target_edge_length_m:
                    augmented_fractions[key].add(round(fraction, 12))
                internal_match = True
                break
            if not internal_match:
                interior_guard_points[panel.id].append(point)

    synchronized_fractions = _synchronized_path_fractions(
        construction,
        target_edge_length_m,
        additional_fractions={
            key: tuple(sorted(values)) for key, values in augmented_fractions.items()
        },
    )

    # An internal topstitch shares discrete particles with its mounted layer,
    # but must not turn the shell midsurface into a three- or four-face edge.
    # Insert an *unstitched* midpoint on both material sides between discrete
    # stitch samples.  The host midpoint remains on its midsurface while an
    # attached pocket/overlay author can lift its counterpart away; only the
    # actual stitch particles coincide.  Splitting only the host side left the
    # mounted straight edge passing through that midpoint, an unavoidable
    # zero-thickness triangle overlap rather than a physical seam.
    for seam in construction.seams:
        side_panels = (
            construction.panel_map[seam.side_a.panel_id],
            construction.panel_map[seam.side_b.panel_id],
        )
        side_refs = (seam.side_a, seam.side_b)
        if not any(
            path_ref.path_id
            not in {path.id for path in panel.outline_paths}
            for panel, path_ref in zip(side_panels, side_refs)
        ):
            continue
        for path_ref in (seam.side_a, seam.side_b):
            panel = construction.panel_map[path_ref.panel_id]
            path = panel.path(path_ref.path_id)
            fractions = [
                fraction
                for fraction in synchronized_fractions[(panel.id, path.id)]
                if path_ref.start_fraction - 1.0e-12
                <= fraction
                <= path_ref.end_fraction + 1.0e-12
            ]
            for first, second in zip(fractions[:-1], fractions[1:]):
                midpoint_fraction = 0.5 * (first + second)
                interior_guard_points[panel.id].append(
                    path.point_at_fraction(midpoint_fraction)
                )

    stitch_references_by_panel: dict[str, list[PathRef]] = {
        panel.id: [] for panel in construction.panels
    }
    shared_outline_seam_count: dict[tuple[str, str], int] = {}
    for seam in construction.seams:
        stitch_references_by_panel[seam.side_a.panel_id].append(seam.side_a)
        stitch_references_by_panel[seam.side_b.panel_id].append(seam.side_b)
        if seam.side_a.panel_id == seam.side_b.panel_id:
            continue
        panel_a = construction.panel_map[seam.side_a.panel_id]
        panel_b = construction.panel_map[seam.side_b.panel_id]
        outline_a = {path.id for path in panel_a.outline_paths}
        outline_b = {path.id for path in panel_b.outline_paths}
        if seam.side_a.path_id in outline_a and seam.side_b.path_id in outline_b:
            pair = tuple(sorted((panel_a.id, panel_b.id)))
            shared_outline_seam_count[pair] = shared_outline_seam_count.get(pair, 0) + 1
    # Three or more shared cut edges describe a closed layered subassembly
    # such as a two-piece pocket bag.  Those are the cases where matching
    # triangulation chords can become duplicate four-face edges after welding.
    layered_subassembly_panels = {
        panel_id
        for pair, count in shared_outline_seam_count.items()
        if count >= 3
        for panel_id in pair
    }
    panel_meshes = tuple(
        _triangulate_without_unintended_stitch_chords(
            panel,
            target_edge_length_m=target_edge_length_m,
            required_path_fractions={
                path.id: synchronized_fractions[(panel.id, path.id)]
                for path in panel.outline_paths + panel.internal_paths
            },
            stitch_references=(
                tuple(stitch_references_by_panel[panel.id])
                if panel.id in layered_subassembly_panels
                else ()
            ),
            additional_points_m=tuple(interior_guard_points[panel.id]),
        )
        for panel in construction.panels
    )
    panel_ids = tuple(panel.id for panel in construction.panels)
    material_profile_ids = tuple(
        dict.fromkeys(panel.material_profile for panel in construction.panels)
    )
    material_to_index = {
        material_id: index for index, material_id in enumerate(material_profile_ids)
    }

    positions_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    vertex_panel_parts: list[np.ndarray] = []
    triangle_panel_parts: list[np.ndarray] = []
    triangle_material_parts: list[np.ndarray] = []
    panel_vertex_ranges: dict[str, tuple[int, int]] = {}
    global_path_vertices: dict[tuple[str, str], np.ndarray] = {}
    global_path_fractions: dict[tuple[str, str], np.ndarray] = {}

    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    vertex_offset = 0
    for panel_index, (panel, mesh) in enumerate(zip(construction.panels, panel_meshes)):
        minimum = np.min(mesh.material_coordinates_m, axis=0)
        maximum = np.max(mesh.material_coordinates_m, axis=0)
        span = maximum - minimum
        if cursor_x > 0.0 and cursor_x + span[0] > atlas_row_width_m:
            cursor_x = 0.0
            cursor_y += row_height + atlas_gap_m
            row_height = 0.0
        xy = mesh.material_coordinates_m - minimum + np.asarray((cursor_x, cursor_y))
        positions_parts.append(np.column_stack((xy, np.zeros(mesh.vertex_count))))
        material_parts.append(mesh.material_coordinates_m)
        uv_parts.append(mesh.uv)
        triangle_parts.append(mesh.triangles + vertex_offset)
        area_parts.append(mesh.vertex_area_m2)
        mass_parts.append(mesh.vertex_mass_kg)
        vertex_panel_parts.append(np.full(mesh.vertex_count, panel_index, dtype=np.int32))
        triangle_panel_parts.append(np.full(mesh.triangle_count, panel_index, dtype=np.int32))
        triangle_material_parts.append(
            np.full(
                mesh.triangle_count,
                material_to_index[panel.material_profile],
                dtype=np.int32,
            )
        )
        panel_vertex_ranges[panel.id] = (vertex_offset, vertex_offset + mesh.vertex_count)
        for path_id, indices in mesh.path_vertices.items():
            key = (panel.id, path_id)
            global_path_vertices[key] = indices + vertex_offset
            global_path_fractions[key] = mesh.path_fractions[path_id]
        vertex_offset += mesh.vertex_count
        cursor_x += span[0] + atlas_gap_m
        row_height = max(row_height, float(span[1]))

    stitches: list[StitchConstraintBatch] = []
    for seam in construction.seams:
        side_a = _indices_for_ref(
            seam.side_a,
            global_path_vertices=global_path_vertices,
            global_path_fractions=global_path_fractions,
        )
        side_b = _indices_for_ref(
            seam.side_b,
            global_path_vertices=global_path_vertices,
            global_path_fractions=global_path_fractions,
        )
        if len(side_a) != len(side_b):
            raise ValueError(
                f"Seam {seam.id!r} has {len(side_a)} versus {len(side_b)} mesh vertices "
                "after synchronized discretization"
            )
        count = len(side_a)
        paired_a = side_a
        paired_b = side_b
        pairs = np.column_stack((paired_a, paired_b)).astype(np.int32)
        seam_length = 0.5 * (
            seam.side_a.length_m(construction.panel_map)
            + seam.side_b.length_m(construction.panel_map)
        )
        numerical_spacing = seam_length / max(1, count - 1)
        # One numerical constraint represents several physical stitches.  The
        # scaling keeps the aggregate parallel stiffness/damping approximately
        # invariant under mesh refinement; it is still a prior until a sewn
        # seam coupon is tested.
        coarse_graining = max(1.0, numerical_spacing / seam.stitch_spacing_m)
        stitches.append(
            StitchConstraintBatch(
                seam_id=seam.id,
                vertex_pairs=pairs,
                physical_stitch_spacing_m=seam.stitch_spacing_m,
                numerical_spacing_m=numerical_spacing,
                per_constraint_stiffness_n_m=seam.line_stiffness_n_m * coarse_graining,
                per_constraint_damping_n_s_m=seam.line_damping_n_s_m * coarse_graining,
                rest_length_m=0.0,
            )
        )

    return GarmentPatternMesh(
        construction_id=construction.id,
        positions_m=np.concatenate(positions_parts, axis=0),
        material_coordinates_m=np.concatenate(material_parts, axis=0),
        uv=np.concatenate(uv_parts, axis=0),
        triangles=np.concatenate(triangle_parts, axis=0),
        vertex_area_m2=np.concatenate(area_parts, axis=0),
        vertex_mass_kg=np.concatenate(mass_parts, axis=0),
        vertex_panel_index=np.concatenate(vertex_panel_parts, axis=0),
        triangle_panel_index=np.concatenate(triangle_panel_parts, axis=0),
        triangle_material_index=np.concatenate(triangle_material_parts, axis=0),
        panel_ids=panel_ids,
        material_profile_ids=material_profile_ids,
        panel_vertex_ranges=MappingProxyType(panel_vertex_ranges),
        path_vertices=MappingProxyType(global_path_vertices),
        path_fractions=MappingProxyType(global_path_fractions),
        stitches=tuple(stitches),
        panel_meshes=panel_meshes,
        atlas_gap_m=atlas_gap_m,
    )

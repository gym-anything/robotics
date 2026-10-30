"""Render-only finite-thickness shells for arbitrary sewn garment panels.

The mechanical state remains the welded :class:`SewnGarmentTopology`.  This
module only barycentrically subdivides each real cut panel, offsets its two
render faces by the panel's declared unloaded thickness, and closes the cut
boundary with side walls.  Nothing produced here is fed back to Newton.

Keeping panels separate is intentional: a shirt pocket, denim yoke, collar,
or facing is a second physical cut piece with its own material coordinates.
The panel shells therefore retain construction layers and UV/grain frames
instead of collapsing a garment to one anonymous triangle soup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .garment_assembly import SewnGarmentTopology
from .garment_meshing import GarmentPatternMesh


@dataclass(frozen=True)
class GarmentRenderSource:
    """Minimal sewn indexing needed to render a committed artifact state."""

    construction_id: str
    material_to_physical_vertex: np.ndarray

    def __post_init__(self) -> None:
        if not self.construction_id:
            raise ValueError("A garment render source needs a construction id")
        mapping = np.asarray(self.material_to_physical_vertex)
        if mapping.ndim != 1 or len(mapping) == 0:
            raise ValueError("A garment render source needs a one-dimensional map")
        if np.min(mapping) < 0:
            raise ValueError("A garment render source cannot contain negative indices")


@dataclass(frozen=True)
class GarmentPanelRenderEmbedding:
    """Barycentric map from one render panel to the welded physical mesh."""

    panel_id: str
    coarse_vertex_indices: np.ndarray
    barycentric_weights: np.ndarray
    material_coordinates_m: np.ndarray
    surface_triangles: np.ndarray
    boundary_segments: np.ndarray
    subdivision_factor: int

    def __post_init__(self) -> None:
        count = int(self.coarse_vertex_indices.shape[0])
        if not self.panel_id:
            raise ValueError("A render embedding requires a panel identifier")
        if self.coarse_vertex_indices.shape != (count, 3):
            raise ValueError("Garment render indices must have shape (N, 3)")
        if self.barycentric_weights.shape != (count, 3):
            raise ValueError("Garment render weights must have shape (N, 3)")
        if self.material_coordinates_m.shape != (count, 2):
            raise ValueError("Garment render material coordinates must be 2-D")
        if self.surface_triangles.ndim != 2 or self.surface_triangles.shape[1] != 3:
            raise ValueError("Garment render surface triangles must have shape (T, 3)")
        if self.boundary_segments.ndim != 2 or self.boundary_segments.shape[1] != 2:
            raise ValueError("Garment render boundary segments must have shape (B, 2)")
        if self.subdivision_factor < 1:
            raise ValueError("Garment render subdivision must be positive")
        if count == 0 or len(self.surface_triangles) == 0:
            raise ValueError("A garment render panel cannot be empty")
        if np.min(self.coarse_vertex_indices) < 0:
            raise ValueError("Garment render embedding has a negative physical index")
        if np.min(self.surface_triangles) < 0 or np.max(self.surface_triangles) >= count:
            raise ValueError("Garment render triangles reference a missing vertex")
        if len(self.boundary_segments) and (
            np.min(self.boundary_segments) < 0
            or np.max(self.boundary_segments) >= count
        ):
            raise ValueError("Garment render boundary references a missing vertex")
        if not np.all(np.isfinite(self.barycentric_weights)):
            raise ValueError("Garment render weights must be finite")
        if not np.allclose(np.sum(self.barycentric_weights, axis=1), 1.0):
            raise ValueError("Garment render weights must sum to one")
        if float(np.min(self.barycentric_weights)) < -1.0e-12:
            raise ValueError("Garment render weights cannot be negative")

    @property
    def vertex_count(self) -> int:
        return int(self.coarse_vertex_indices.shape[0])

    def interpolate(self, physical_positions_m: np.ndarray) -> np.ndarray:
        positions = np.asarray(physical_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Physical garment positions must have shape (N, 3)")
        if int(np.max(self.coarse_vertex_indices)) >= positions.shape[0]:
            raise ValueError("Garment render embedding references a missing particle")
        sampled = positions[self.coarse_vertex_indices]
        return np.sum(sampled * self.barycentric_weights[..., None], axis=1)


@dataclass(frozen=True)
class ThickGarmentPanelTopology:
    """Closed, per-panel visual shell over a sewn mechanical garment."""

    embedding: GarmentPanelRenderEmbedding
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    uv: np.ndarray

    def __post_init__(self) -> None:
        surface_count = self.embedding.vertex_count
        if self.uv.shape != (2 * surface_count, 2):
            raise ValueError("Garment panel UV array has the wrong shape")
        if self.face_vertex_counts.ndim != 1 or self.face_vertex_indices.ndim != 1:
            raise ValueError("Garment panel face arrays must be one-dimensional")
        if int(np.sum(self.face_vertex_counts)) != len(self.face_vertex_indices):
            raise ValueError("Garment panel face counts and indices disagree")
        if np.min(self.face_vertex_indices) < 0 or np.max(
            self.face_vertex_indices
        ) >= 2 * surface_count:
            raise ValueError("Garment panel shell references a missing vertex")


def _panel_triangle_indices(
    pattern_mesh: GarmentPatternMesh,
    panel_id: str,
) -> np.ndarray:
    try:
        panel_index = pattern_mesh.panel_ids.index(panel_id)
    except ValueError as exception:
        raise KeyError(f"Unknown garment panel {panel_id!r}") from exception
    return np.flatnonzero(pattern_mesh.triangle_panel_index == panel_index).astype(
        np.int32
    )


def build_garment_panel_render_topology(
    pattern_mesh: GarmentPatternMesh,
    sewn_topology: SewnGarmentTopology | GarmentRenderSource,
    panel_id: str,
    *,
    subdivision_factor: int = 2,
    texture_repeats_per_m: float = 80.0,
) -> ThickGarmentPanelTopology:
    """Build a subdivided render shell for one actual construction panel.

    Vertices on a material edge are shared inside the panel, including across
    adjacent coarse triangles.  Material vertices are deliberately *not*
    shared between panels: that preserves a pocket's or collar's independent
    UV frame even where its stitch line is welded to a host panel.
    """

    if pattern_mesh.construction_id != sewn_topology.construction_id:
        raise ValueError("Pattern and sewn garment identifiers differ")
    if subdivision_factor < 1:
        raise ValueError("Garment render subdivision must be positive")
    if not np.isfinite(texture_repeats_per_m) or texture_repeats_per_m <= 0.0:
        raise ValueError("Garment texture repeat density must be finite and positive")
    factor = int(subdivision_factor)
    triangle_indices = _panel_triangle_indices(pattern_mesh, panel_id)
    if len(triangle_indices) == 0:
        raise ValueError(f"Panel {panel_id!r} has no compiled triangles")

    material_to_physical = np.asarray(
        sewn_topology.material_to_physical_vertex, dtype=np.int32
    )
    if material_to_physical.shape != (pattern_mesh.vertex_count,):
        raise ValueError("Garment render source has the wrong material vertex count")
    material_coordinates = pattern_mesh.material_coordinates_m
    registry: dict[tuple[object, ...], int] = {}
    coarse_indices: list[tuple[int, int, int]] = []
    weights: list[tuple[float, float, float]] = []
    coordinates: list[tuple[float, float]] = []

    def register_vertex(material_vertex: int) -> int:
        key = ("v", int(material_vertex))
        if key not in registry:
            physical = int(material_to_physical[material_vertex])
            registry[key] = len(coarse_indices)
            coarse_indices.append((physical, physical, physical))
            weights.append((1.0, 0.0, 0.0))
            coordinates.append(tuple(map(float, material_coordinates[material_vertex])))
        return registry[key]

    def register_edge(
        first_material: int,
        second_material: int,
        step_from_first: int,
    ) -> int:
        if step_from_first <= 0:
            return register_vertex(first_material)
        if step_from_first >= factor:
            return register_vertex(second_material)
        low, high = sorted((int(first_material), int(second_material)))
        step_from_low = (
            step_from_first if int(first_material) == low else factor - step_from_first
        )
        key = ("e", low, high, int(step_from_low))
        if key not in registry:
            fraction = step_from_low / factor
            low_physical = int(material_to_physical[low])
            high_physical = int(material_to_physical[high])
            registry[key] = len(coarse_indices)
            coarse_indices.append((low_physical, high_physical, high_physical))
            weights.append((1.0 - fraction, fraction, 0.0))
            point = (
                (1.0 - fraction) * material_coordinates[low]
                + fraction * material_coordinates[high]
            )
            coordinates.append(tuple(map(float, point)))
        return registry[key]

    def register_lattice_point(
        triangle_index: int,
        material_triangle: np.ndarray,
        i: int,
        j: int,
    ) -> int:
        # Barycentric integer weights over material vertices A, B, C.
        integer_weights = (factor - i - j, i, j)
        nonzero = [index for index, value in enumerate(integer_weights) if value]
        if len(nonzero) == 1:
            return register_vertex(int(material_triangle[nonzero[0]]))
        if len(nonzero) == 2:
            first_corner, second_corner = nonzero
            return register_edge(
                int(material_triangle[first_corner]),
                int(material_triangle[second_corner]),
                int(integer_weights[second_corner]),
            )
        key = ("f", int(triangle_index), int(i), int(j))
        if key not in registry:
            physical_triangle = material_to_physical[material_triangle]
            barycentric = np.asarray(integer_weights, dtype=np.float64) / factor
            registry[key] = len(coarse_indices)
            coarse_indices.append(tuple(map(int, physical_triangle)))
            weights.append(tuple(map(float, barycentric)))
            point = barycentric @ material_coordinates[material_triangle]
            coordinates.append(tuple(map(float, point)))
        return registry[key]

    surface_triangles: list[tuple[int, int, int]] = []
    edge_incidence: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for triangle_index_raw in triangle_indices:
        triangle_index = int(triangle_index_raw)
        material_triangle = np.asarray(
            pattern_mesh.triangles[triangle_index], dtype=np.int32
        )
        grid: dict[tuple[int, int], int] = {}
        for i in range(factor + 1):
            for j in range(factor + 1 - i):
                grid[(i, j)] = register_lattice_point(
                    triangle_index, material_triangle, i, j
                )
        for i in range(factor):
            for j in range(factor - i):
                a = grid[(i, j)]
                b = grid[(i + 1, j)]
                c = grid[(i, j + 1)]
                surface_triangles.append((a, b, c))
                if i + j < factor - 1:
                    d = grid[(i + 1, j + 1)]
                    surface_triangles.append((b, d, c))

        for first_raw, second_raw in zip(
            material_triangle,
            np.roll(material_triangle, -1),
        ):
            first, second = int(first_raw), int(second_raw)
            key = (min(first, second), max(first, second))
            count, _orientation = edge_incidence.get(key, (0, (first, second)))
            edge_incidence[key] = (count + 1, (first, second))

    boundary_segments: list[tuple[int, int]] = []
    for count, (first, second) in edge_incidence.values():
        if count != 1:
            continue
        sequence = [
            register_edge(first, second, step) for step in range(factor + 1)
        ]
        boundary_segments.extend(zip(sequence[:-1], sequence[1:]))

    embedding = GarmentPanelRenderEmbedding(
        panel_id=panel_id,
        coarse_vertex_indices=np.asarray(coarse_indices, dtype=np.int32),
        barycentric_weights=np.asarray(weights, dtype=np.float64),
        material_coordinates_m=np.asarray(coordinates, dtype=np.float64),
        surface_triangles=np.asarray(surface_triangles, dtype=np.int32),
        boundary_segments=np.asarray(boundary_segments, dtype=np.int32),
        subdivision_factor=factor,
    )
    count = embedding.vertex_count
    top = embedding.surface_triangles
    bottom = embedding.surface_triangles[:, ::-1] + count
    face_counts: list[int] = [3] * (len(top) + len(bottom))
    face_indices: list[int] = [int(value) for value in top.reshape(-1)]
    face_indices.extend(int(value) for value in bottom.reshape(-1))
    for first, second in embedding.boundary_segments:
        a, b = int(first), int(second)
        face_counts.append(4)
        face_indices.extend((a, a + count, b + count, b))

    uv_surface = embedding.material_coordinates_m * float(texture_repeats_per_m)
    return ThickGarmentPanelTopology(
        embedding=embedding,
        face_vertex_counts=np.asarray(face_counts, dtype=np.int32),
        face_vertex_indices=np.asarray(face_indices, dtype=np.int32),
        uv=np.concatenate((uv_surface, uv_surface), axis=0),
    )


def deform_thick_garment_panel(
    topology: ThickGarmentPanelTopology,
    physical_positions_m: np.ndarray,
    *,
    thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deform one visual panel from a committed welded physical state."""

    if not np.isfinite(thickness_m) or thickness_m <= 0.0:
        raise ValueError("Garment render thickness must be finite and positive")
    surface = topology.embedding.interpolate(physical_positions_m)
    normals = np.zeros_like(surface)
    triangles = topology.embedding.surface_triangles
    face = np.cross(surface[triangles[:, 1]]-surface[triangles[:, 0]],
                    surface[triangles[:, 2]]-surface[triangles[:, 0]])
    face_length = np.linalg.norm(face, axis=1)
    valid = face_length > 1.0e-16
    face[valid] /= face_length[valid, None]
    face[~valid] = 0.
    # Same unit-face weighting and face-order accumulation as the scalar
    # implementation; vectorize only the independent arithmetic/scatter.
    ids = triangles[valid].ravel()
    np.add.at(normals, ids, np.repeat(face[valid], 3, axis=0))
    first = np.full(len(surface), len(face), dtype=np.int64)
    np.minimum.at(first, ids, np.repeat(np.flatnonzero(valid), 3))
    first_face_normal = np.vstack((face, np.zeros((1, 3))))[first]
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1.0e-10
    normals[~fallback] /= lengths[~fallback, None]
    normals[fallback] = first_face_normal[fallback]
    still_zero = np.linalg.norm(normals, axis=1) <= 1.0e-10
    normals[still_zero] = (0.0, 0.0, 1.0)
    half = 0.5 * float(thickness_m)
    points = np.concatenate(
        (surface + half * normals, surface - half * normals), axis=0
    )
    vertex_normals = np.concatenate((normals, -normals), axis=0)
    return points, vertex_normals


__all__ = [
    "GarmentRenderSource",
    "GarmentPanelRenderEmbedding",
    "ThickGarmentPanelTopology",
    "build_garment_panel_render_topology",
    "deform_thick_garment_panel",
]

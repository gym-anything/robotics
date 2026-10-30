"""Material-space meshes and topology utilities for LaundryFold."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Hinge:
    edge_vertices: tuple[int, int]
    opposite_vertices: tuple[int, int]
    adjacent_triangles: tuple[int, int]
    material_edge_angle_rad: float
    rest_edge_length_m: float


@dataclass(frozen=True)
class RectangularMesh:
    positions_m: np.ndarray
    material_coordinates_m: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    vertex_area_m2: np.ndarray
    vertex_mass_kg: np.ndarray
    boundary: dict[str, np.ndarray]
    hinges: tuple[Hinge, ...]
    nx: int
    ny: int
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        n = (self.nx + 1) * (self.ny + 1)
        if self.positions_m.shape != (n, 3):
            raise ValueError("positions have the wrong size")
        if self.material_coordinates_m.shape != (n, 2) or self.uv.shape != (n, 2):
            raise ValueError("material coordinate arrays have the wrong size")
        if self.triangles.shape != (2 * self.nx * self.ny, 3):
            raise ValueError("triangle array has the wrong size")
        if self.vertex_area_m2.shape != (n,) or self.vertex_mass_kg.shape != (n,):
            raise ValueError("vertex scalar arrays have the wrong size")

    @property
    def vertex_count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.vertex_mass_kg))

    @property
    def total_area_m2(self) -> float:
        return float(np.sum(self.vertex_area_m2))


def _vertex(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


def _build_hinges(
    positions: np.ndarray,
    material: np.ndarray,
    triangles: np.ndarray,
) -> tuple[Hinge, ...]:
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle_id, tri in enumerate(triangles):
        a, b, c = (int(value) for value in tri)
        for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(u, v), max(u, v))
            edge_map.setdefault(key, []).append((triangle_id, opposite))

    hinges: list[Hinge] = []
    for (a, b), records in edge_map.items():
        if len(records) != 2:
            continue
        (t0, c), (t1, d) = records
        delta_material = material[b] - material[a]
        angle = math.atan2(float(delta_material[1]), float(delta_material[0]))
        length = float(np.linalg.norm(positions[b] - positions[a]))
        hinges.append(Hinge((a, b), (c, d), (t0, t1), angle, length))
    hinges.sort(key=lambda hinge: hinge.edge_vertices)
    return tuple(hinges)


def build_rectangular_mesh(
    *,
    length_m: float,
    width_m: float,
    nx: int,
    ny: int,
    areal_density_kg_m2: float,
    origin_m=(0.0, 0.0, 0.0),
    alternating_diagonals: bool = True,
) -> RectangularMesh:
    """Build a centred, +Z-facing triangle mesh with exact lumped mass.

    ``nx`` and ``ny`` count cells, not vertices.  Alternating diagonals reduce
    a systematic grid bias while preserving material coordinates.
    """

    if min(length_m, width_m, areal_density_kg_m2) <= 0.0:
        raise ValueError("length, width, and areal density must be positive")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    origin = np.asarray(origin_m, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_m must be a finite 3-vector")

    x_values = np.linspace(-0.5 * length_m, 0.5 * length_m, nx + 1)
    y_values = np.linspace(-0.5 * width_m, 0.5 * width_m, ny + 1)
    positions = np.empty(((nx + 1) * (ny + 1), 3), dtype=np.float64)
    material = np.empty((len(positions), 2), dtype=np.float64)
    uv = np.empty_like(material)
    for iy, y in enumerate(y_values):
        for ix, x in enumerate(x_values):
            index = _vertex(ix, iy, nx)
            positions[index] = origin + np.array([x, y, 0.0])
            material[index] = (x, y)
            uv[index] = (ix / nx, iy / ny)

    triangle_list: list[tuple[int, int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            bl = _vertex(ix, iy, nx)
            br = _vertex(ix + 1, iy, nx)
            tl = _vertex(ix, iy + 1, nx)
            tr = _vertex(ix + 1, iy + 1, nx)
            alternate = alternating_diagonals and ((ix + iy) % 2 == 1)
            if alternate:
                triangle_list.extend(((bl, br, tl), (br, tr, tl)))
            else:
                triangle_list.extend(((bl, br, tr), (bl, tr, tl)))
    triangles = np.asarray(triangle_list, dtype=np.int32)

    vertex_area = np.zeros(len(positions), dtype=np.float64)
    for tri in triangles:
        a, b, c = positions[tri]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        vertex_area[tri] += area / 3.0
    vertex_mass = vertex_area * areal_density_kg_m2

    boundary = {
        "left": np.asarray([_vertex(0, iy, nx) for iy in range(ny + 1)], dtype=np.int32),
        "right": np.asarray([_vertex(nx, iy, nx) for iy in range(ny + 1)], dtype=np.int32),
        "bottom": np.asarray([_vertex(ix, 0, nx) for ix in range(nx + 1)], dtype=np.int32),
        "top": np.asarray([_vertex(ix, ny, nx) for ix in range(nx + 1)], dtype=np.int32),
    }
    hinges = _build_hinges(positions, material, triangles)
    return RectangularMesh(
        positions,
        material,
        uv,
        triangles,
        vertex_area,
        vertex_mass,
        boundary,
        hinges,
        nx,
        ny,
        length_m,
        width_m,
    )


def triangle_signed_areas_xy(positions_m: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    a = positions[tri[:, 0], :2]
    b = positions[tri[:, 1], :2]
    c = positions[tri[:, 2], :2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

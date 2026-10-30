"""Render-only subdivision and finite-thickness geometry for cloth snapshots.

The interpolation is barycentric inside the accepted mechanical triangles. It
adds visual tessellation and a measured shell thickness without inventing a
second deformation state or feeding anything back to physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import RectangularMesh


@dataclass(frozen=True)
class ClothRenderEmbedding:
    coarse_vertex_indices: np.ndarray
    barycentric_weights: np.ndarray
    material_coordinates_m: np.ndarray
    surface_triangles: np.ndarray
    nx: int
    ny: int

    def __post_init__(self) -> None:
        count = (self.nx + 1) * (self.ny + 1)
        if self.coarse_vertex_indices.shape != (count, 3):
            raise ValueError("Render embedding indices have the wrong shape")
        if self.barycentric_weights.shape != (count, 3):
            raise ValueError("Render embedding weights have the wrong shape")
        if self.material_coordinates_m.shape != (count, 2):
            raise ValueError("Render material coordinates have the wrong shape")
        if self.surface_triangles.shape != (2 * self.nx * self.ny, 3):
            raise ValueError("Render surface triangles have the wrong shape")
        if not np.allclose(np.sum(self.barycentric_weights, axis=1), 1.0):
            raise ValueError("Render barycentric weights must sum to one")
        if np.min(self.barycentric_weights) < -1.0e-12:
            raise ValueError("Render barycentric weights cannot be negative")

    @property
    def vertex_count(self) -> int:
        return int(self.coarse_vertex_indices.shape[0])

    def interpolate(self, coarse_positions_m: np.ndarray) -> np.ndarray:
        positions = np.asarray(coarse_positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Coarse positions must have shape (N, 3)")
        if np.max(self.coarse_vertex_indices) >= positions.shape[0]:
            raise ValueError("Render embedding references a missing coarse vertex")
        sampled = positions[self.coarse_vertex_indices]
        return np.sum(sampled * self.barycentric_weights[..., None], axis=1)


@dataclass(frozen=True)
class ThickClothTopology:
    embedding: ClothRenderEmbedding
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    uv: np.ndarray
    boundary_loop: np.ndarray

    def __post_init__(self) -> None:
        surface_count = self.embedding.vertex_count
        if self.uv.shape != (2 * surface_count, 2):
            raise ValueError("Thick-cloth UV array has the wrong shape")
        if int(np.sum(self.face_vertex_counts)) != len(self.face_vertex_indices):
            raise ValueError("Thick-cloth face counts and indices disagree")
        if np.min(self.face_vertex_indices) < 0 or np.max(
            self.face_vertex_indices
        ) >= 2 * surface_count:
            raise ValueError("Thick-cloth topology references a missing vertex")


def build_render_embedding(
    mesh: RectangularMesh,
    *,
    subdivision_factor: int = 3,
) -> ClothRenderEmbedding:
    if subdivision_factor < 1:
        raise ValueError("subdivision_factor must be positive")
    factor = int(subdivision_factor)
    nx = mesh.nx * factor
    ny = mesh.ny * factor
    indices = np.empty(((nx + 1) * (ny + 1), 3), dtype=np.int32)
    weights = np.empty_like(indices, dtype=np.float64)
    material = np.empty(((nx + 1) * (ny + 1), 2), dtype=np.float64)

    def coarse_vertex(ix: int, iy: int) -> int:
        return iy * (mesh.nx + 1) + ix

    for high_y in range(ny + 1):
        coarse_y = min(high_y // factor, mesh.ny - 1)
        v = (high_y - coarse_y * factor) / factor
        for high_x in range(nx + 1):
            coarse_x = min(high_x // factor, mesh.nx - 1)
            u = (high_x - coarse_x * factor) / factor
            row = high_y * (nx + 1) + high_x
            bl = coarse_vertex(coarse_x, coarse_y)
            br = coarse_vertex(coarse_x + 1, coarse_y)
            tl = coarse_vertex(coarse_x, coarse_y + 1)
            tr = coarse_vertex(coarse_x + 1, coarse_y + 1)
            alternate = (coarse_x + coarse_y) % 2 == 1
            if alternate and u + v <= 1.0:
                indices[row] = (bl, br, tl)
                weights[row] = (1.0 - u - v, u, v)
            elif alternate:
                indices[row] = (br, tr, tl)
                weights[row] = (1.0 - v, u + v - 1.0, 1.0 - u)
            elif v <= u:
                indices[row] = (bl, br, tr)
                weights[row] = (1.0 - u, u - v, v)
            else:
                indices[row] = (bl, tr, tl)
                weights[row] = (1.0 - v, u, v - u)
            material[row] = (
                -0.5 * mesh.length_m + mesh.length_m * high_x / nx,
                -0.5 * mesh.width_m + mesh.width_m * high_y / ny,
            )

    triangles: list[tuple[int, int, int]] = []
    for high_y in range(ny):
        for high_x in range(nx):
            bl = high_y * (nx + 1) + high_x
            br = bl + 1
            tl = bl + nx + 1
            tr = tl + 1
            coarse_x = high_x // factor
            coarse_y = high_y // factor
            if (coarse_x + coarse_y) % 2 == 1:
                triangles.extend(((bl, br, tl), (br, tr, tl)))
            else:
                triangles.extend(((bl, br, tr), (bl, tr, tl)))
    return ClothRenderEmbedding(
        coarse_vertex_indices=indices,
        barycentric_weights=weights,
        material_coordinates_m=material,
        surface_triangles=np.asarray(triangles, dtype=np.int32),
        nx=nx,
        ny=ny,
    )


def build_thick_cloth_topology(
    mesh: RectangularMesh,
    *,
    subdivision_factor: int = 3,
    texture_repeats_per_m: float = 60.0,
) -> ThickClothTopology:
    if texture_repeats_per_m <= 0.0:
        raise ValueError("Texture repeat density must be positive")
    embedding = build_render_embedding(
        mesh, subdivision_factor=subdivision_factor
    )
    count = embedding.vertex_count
    top = embedding.surface_triangles
    bottom = embedding.surface_triangles[:, ::-1] + count
    counts: list[int] = [3] * (len(top) + len(bottom))
    faces: list[int] = [int(value) for value in top.reshape(-1)]
    faces.extend(int(value) for value in bottom.reshape(-1))

    nx, ny = embedding.nx, embedding.ny
    boundary: list[int] = []
    boundary.extend(range(0, nx + 1))
    boundary.extend(iy * (nx + 1) + nx for iy in range(1, ny + 1))
    boundary.extend(ny * (nx + 1) + ix for ix in range(nx - 1, -1, -1))
    boundary.extend(iy * (nx + 1) for iy in range(ny - 1, 0, -1))
    for index, a in enumerate(boundary):
        b = boundary[(index + 1) % len(boundary)]
        counts.append(4)
        faces.extend((a, a + count, b + count, b))

    uv_surface = (
        embedding.material_coordinates_m
        - np.min(embedding.material_coordinates_m, axis=0)
    ) * float(texture_repeats_per_m)
    return ThickClothTopology(
        embedding=embedding,
        face_vertex_counts=np.asarray(counts, dtype=np.int32),
        face_vertex_indices=np.asarray(faces, dtype=np.int32),
        uv=np.concatenate((uv_surface, uv_surface), axis=0),
        boundary_loop=np.asarray(boundary, dtype=np.int32),
    )


def deform_thick_cloth(
    topology: ThickClothTopology,
    coarse_positions_m: np.ndarray,
    *,
    thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if thickness_m <= 0.0:
        raise ValueError("Cloth render thickness must be positive")
    surface = topology.embedding.interpolate(coarse_positions_m)
    triangles = topology.embedding.surface_triangles
    normals = np.zeros_like(surface)
    first_face_normal = np.zeros_like(surface)
    for triangle in triangles:
        a, b, c = (int(value) for value in triangle)
        normal = np.cross(surface[b] - surface[a], surface[c] - surface[a])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-16:
            continue
        normal /= length
        normals[triangle] += normal
        for vertex in triangle:
            if not np.any(first_face_normal[vertex]):
                first_face_normal[vertex] = normal
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1.0e-10
    normals[~fallback] /= lengths[~fallback, None]
    normals[fallback] = first_face_normal[fallback]
    still_zero = np.linalg.norm(normals, axis=1) <= 1.0e-10
    normals[still_zero] = (0.0, 0.0, 1.0)
    half = 0.5 * float(thickness_m)
    points = np.concatenate((surface + half * normals, surface - half * normals), axis=0)
    vertex_normals = np.concatenate((normals, -normals), axis=0)
    return points, vertex_normals


__all__ = [
    "ClothRenderEmbedding",
    "ThickClothTopology",
    "build_render_embedding",
    "build_thick_cloth_topology",
    "deform_thick_cloth",
]

"""Topological self-contact filtering missing from Newton 1.4 Style3D.

Style3D's BVH kernels reject candidates that share a vertex, but nearby
triangles across one or more mesh edges can still be reported as self-contact.
Those local pairs are governed by the cloth bending model; treating them as
nonlocal collision pairs creates artificial impulses at smooth folds.

This adapter compacts all three Style3D broad-phase candidate buffers after
their official BVH query.  The rigid/soft contact path remains untouched.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.style3d.collision.collision import Collision

from .self_intersection import build_vertex_neighbourhoods


@wp.func
def _is_in_vertex_neighbourhood(
    source: int,
    candidate: int,
    offsets: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.int32),
) -> bool:
    begin = offsets[source]
    end = offsets[source + 1]
    for index in range(begin, end):
        if vertices[index] == candidate:
            return True
    return False


@wp.func
def _edge_is_topologically_local(
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    offsets: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.int32),
) -> bool:
    return (
        _is_in_vertex_neighbourhood(a0, b0, offsets, vertices)
        or _is_in_vertex_neighbourhood(a0, b1, offsets, vertices)
        or _is_in_vertex_neighbourhood(a1, b0, offsets, vertices)
        or _is_in_vertex_neighbourhood(a1, b1, offsets, vertices)
    )


@wp.kernel
def compact_nonlocal_vertex_face_candidates(
    tri_indices: wp.array2d(dtype=wp.int32),
    neighbourhood_offsets: wp.array(dtype=wp.int32),
    neighbourhood_vertices: wp.array(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
    diagnostics: wp.array(dtype=wp.int32),
):
    vertex = wp.tid()
    count = candidates[0, vertex]
    retained = wp.int32(0)
    for row in range(count):
        triangle = candidates[row + 1, vertex]
        local = (
            _is_in_vertex_neighbourhood(
                vertex,
                tri_indices[triangle, 0],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                vertex,
                tri_indices[triangle, 1],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                vertex,
                tri_indices[triangle, 2],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
        )
        if not local:
            candidates[retained + 1, vertex] = triangle
            retained += 1
    candidates[0, vertex] = retained
    wp.atomic_add(diagnostics, 0, count - retained)
    wp.atomic_add(diagnostics, 1, retained)


@wp.kernel
def compact_nonlocal_edge_edge_candidates(
    edge_indices: wp.array2d(dtype=wp.int32),
    neighbourhood_offsets: wp.array(dtype=wp.int32),
    neighbourhood_vertices: wp.array(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
    diagnostics: wp.array(dtype=wp.int32),
):
    edge = wp.tid()
    a0 = edge_indices[edge, 2]
    a1 = edge_indices[edge, 3]
    count = candidates[0, edge]
    retained = wp.int32(0)
    for row in range(count):
        other = candidates[row + 1, edge]
        local = _edge_is_topologically_local(
            a0,
            a1,
            edge_indices[other, 2],
            edge_indices[other, 3],
            neighbourhood_offsets,
            neighbourhood_vertices,
        )
        if not local:
            candidates[retained + 1, edge] = other
            retained += 1
    candidates[0, edge] = retained
    wp.atomic_add(diagnostics, 2, count - retained)
    wp.atomic_add(diagnostics, 3, retained)


@wp.kernel
def compact_nonlocal_edge_face_candidates(
    edge_indices: wp.array2d(dtype=wp.int32),
    tri_indices: wp.array2d(dtype=wp.int32),
    neighbourhood_offsets: wp.array(dtype=wp.int32),
    neighbourhood_vertices: wp.array(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
    diagnostics: wp.array(dtype=wp.int32),
):
    edge = wp.tid()
    a0 = edge_indices[edge, 2]
    a1 = edge_indices[edge, 3]
    count = candidates[0, edge]
    retained = wp.int32(0)
    for row in range(count):
        triangle = candidates[row + 1, edge]
        local = (
            _is_in_vertex_neighbourhood(
                a0,
                tri_indices[triangle, 0],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                a0,
                tri_indices[triangle, 1],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                a0,
                tri_indices[triangle, 2],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                a1,
                tri_indices[triangle, 0],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                a1,
                tri_indices[triangle, 1],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
            or _is_in_vertex_neighbourhood(
                a1,
                tri_indices[triangle, 2],
                neighbourhood_offsets,
                neighbourhood_vertices,
            )
        )
        if not local:
            candidates[retained + 1, edge] = triangle
            retained += 1
    candidates[0, edge] = retained
    wp.atomic_add(diagnostics, 4, count - retained)
    wp.atomic_add(diagnostics, 5, retained)


class TopologicallyFilteredStyle3DCollision(Collision):
    """Official Style3D collision handler plus welded-topology compaction."""

    def __init__(self, model, triangles: np.ndarray, *, graph_hops: int) -> None:
        super().__init__(model)
        neighborhoods = build_vertex_neighbourhoods(
            triangles,
            vertex_count=int(model.particle_count),
            graph_hops=graph_hops,
        )
        offsets = np.zeros(len(neighborhoods) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(
            np.asarray([len(values) for values in neighborhoods], dtype=np.int32)
        )
        flattened = np.asarray(
            [vertex for values in neighborhoods for vertex in sorted(values)],
            dtype=np.int32,
        )
        self.graph_hops = int(graph_hops)
        self.neighbourhood_offsets = wp.array(
            offsets,
            dtype=wp.int32,
            device=model.device,
        )
        self.neighbourhood_vertices = wp.array(
            flattened,
            dtype=wp.int32,
            device=model.device,
        )
        # removed/retained for VF, EE, and EF respectively, for the latest
        # frame.  These numbers make the adapter observable.
        self.topological_filter_diagnostics = wp.zeros(
            6,
            dtype=wp.int32,
            device=model.device,
        )

    def frame_begin(self, particle_q, particle_qd, dt: float) -> None:
        super().frame_begin(particle_q, particle_qd, dt)
        self.topological_filter_diagnostics.zero_()
        if self.stiff_vf > 0.0:
            wp.launch(
                compact_nonlocal_vertex_face_candidates,
                dim=self.model.particle_count,
                inputs=[
                    self.model.tri_indices,
                    self.neighbourhood_offsets,
                    self.neighbourhood_vertices,
                    self.broad_phase_vf,
                    self.topological_filter_diagnostics,
                ],
                device=self.model.device,
            )
        if self.stiff_ee > 0.0:
            wp.launch(
                compact_nonlocal_edge_edge_candidates,
                dim=self.model.edge_count,
                inputs=[
                    self.model.edge_indices,
                    self.neighbourhood_offsets,
                    self.neighbourhood_vertices,
                    self.broad_phase_ee,
                    self.topological_filter_diagnostics,
                ],
                device=self.model.device,
            )
        if self.stiff_ef > 0.0:
            wp.launch(
                compact_nonlocal_edge_face_candidates,
                dim=self.model.edge_count,
                inputs=[
                    self.model.edge_indices,
                    self.model.tri_indices,
                    self.neighbourhood_offsets,
                    self.neighbourhood_vertices,
                    self.broad_phase_ef,
                    self.topological_filter_diagnostics,
                ],
                device=self.model.device,
            )

    def filter_summary(self) -> dict[str, int]:
        values = self.topological_filter_diagnostics.numpy()
        return {
            "vertex_face_removed": int(values[0]),
            "vertex_face_retained": int(values[1]),
            "edge_edge_removed": int(values[2]),
            "edge_edge_retained": int(values[3]),
            "edge_face_removed": int(values[4]),
            "edge_face_retained": int(values[5]),
        }


__all__ = ["TopologicallyFilteredStyle3DCollision"]

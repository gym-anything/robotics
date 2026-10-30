"""Observational self-intersection audits for arbitrary cloth triangle meshes.

The runtime solvers use proximity contacts to prevent cloth layers from
crossing.  Contact candidate counts are useful diagnostics, but they do not
prove that the surface stayed embedded.  This module provides the stricter,
on-demand triangle--triangle query used by episode integrity gates.

The Newton adapter is deliberately observational: it owns a separate BVH and
never refits, clears, or resizes a live solver's collision buffers.  Newton's
triangle query is used only as a candidate generator because it can report
nearby, near-coplanar triangle pairs that do not actually intersect.  IPC
Toolkit performs the robust discrete confirmation and an independent whole-
surface check.  Both counts remain visible in the audit rather than silently
weakening the zero-intersection gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def separated_face_pairs(positions: np.ndarray, triangles: np.ndarray,
                         pairs: np.ndarray) -> np.ndarray:
    """Conservative vectorized plane-side rejection, never intersection proof.

    Reject only when all three vertices lie strictly on the same side of the
    other triangle's plane. Uncertain, degenerate, touching and coplanar cases
    still go to IPC's original robust predicates. The orient3d determinant's
    roundoff bound uses 64 eps times its absolute-product permanent (larger
    than Shewchuk's (7+56 eps) eps filter). Under/overflow is not certified.
    """
    a, b = positions[triangles[pairs[:, 0]]], positions[triangles[pairs[:, 1]]]

    def one_side(face, points):
        # Orient3d translated by each tested point, using explicit operation
        # order rather than a BLAS dot product with implementation-specific FMA.
        u, v, w = (face[:, i, None, :]-points for i in range(3))
        p = u[..., 0]*(v[..., 1]*w[..., 2]-v[..., 2]*w[..., 1])
        q = u[..., 1]*(v[..., 2]*w[..., 0]-v[..., 0]*w[..., 2])
        r = u[..., 2]*(v[..., 0]*w[..., 1]-v[..., 1]*w[..., 0])
        determinant = (p+q)+r
        au, av, aw = abs(u), abs(v), abs(w)
        permanent = (au[..., 0]*(av[..., 1]*aw[..., 2]+av[..., 2]*aw[..., 1])
            + au[..., 1]*(av[..., 2]*aw[..., 0]+av[..., 0]*aw[..., 2])
            + au[..., 2]*(av[..., 0]*aw[..., 1]+av[..., 1]*aw[..., 0]))
        error = 64*np.finfo(np.float64).eps*permanent
        safe = np.isfinite(permanent) & (permanent > np.finfo(np.float64).tiny/np.finfo(np.float64).eps)
        return np.all(safe & (determinant > error), axis=1) | np.all(safe & (determinant < -error), axis=1)

    return one_side(a, b) | one_side(b, a)


def build_vertex_neighbourhoods(
    triangles: np.ndarray,
    *,
    vertex_count: int,
    graph_hops: int,
) -> tuple[frozenset[int], ...]:
    """Return inclusive vertex neighborhoods on the welded cloth topology.

    ``graph_hops=0`` contains only each vertex.  A positive value also follows
    triangle edges by that many hops.  Because garment seams are welded before
    this function is called, stitched neighbors are classified exactly like
    neighbors inside a cut panel.
    """

    faces = np.asarray(triangles, dtype=np.int64)
    if vertex_count < 1:
        raise ValueError("A cloth topology requires at least one vertex")
    if graph_hops < 0:
        raise ValueError("Topological audit graph hops cannot be negative")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Cloth triangles must have shape (triangle_count, 3)")
    if faces.size and (int(np.min(faces)) < 0 or int(np.max(faces)) >= vertex_count):
        raise ValueError("A cloth triangle references a vertex outside the mesh")

    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for a_raw, b_raw, c_raw in faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    neighborhoods: list[frozenset[int]] = []
    for vertex in range(vertex_count):
        visited = {vertex}
        frontier = {vertex}
        for _ in range(graph_hops):
            frontier = {
                neighbor
                for current in frontier
                for neighbor in adjacency[current]
                if neighbor not in visited
            }
            visited.update(frontier)
        neighborhoods.append(frozenset(visited))
    return tuple(neighborhoods)


def projected_edge_separation(positions, triangles, pairs):
    """Certify disjoint 2-D projections; uncertain orientations stay in IPC.

    An edge separates two projected triangles if its own third vertex is
    strictly on one side and all three other vertices are strictly opposite.
    Disjoint projections imply disjoint 3-D triangles. The 64-eps orient2d
    roundoff bound deliberately retains near-degenerate/underflow cases.
    """
    faces = positions[triangles[pairs]]
    separated = np.zeros(len(pairs), dtype=bool)
    eps, tiny = np.finfo(float).eps, np.finfo(float).tiny
    def side(a, b, p):
        u, v = a-p, b-p
        left, right = u[...,0]*v[...,1], u[...,1]*v[...,0]
        permanent = abs(left)+abs(right)
        det, error = left-right, 64*eps*permanent
        safe = np.isfinite(permanent) & (permanent > tiny/eps)
        return np.where(safe & (det > error), 1, np.where(safe & (det < -error), -1, 0))
    for axes in ((0,1), (1,2), (0,2)):
        f = faces[...,list(axes)]
        for first, second in ((0,1), (1,0)):
            a = f[:,first]
            b, third = a[:,[1,2,0]], a[:,[2,0,1]]
            own = side(a, b, third)
            other = side(a[:,:,None,:], b[:,:,None,:], f[:,second,None,:,:])
            separated |= np.any((own != 0) & np.all(other == -own[:,:,None], axis=2), axis=1)
    return separated


@dataclass(frozen=True)
class SelfIntersectionAuditCounts:
    """Unique confirmed intersections plus conservative Newton candidates.

    ``raw_count`` retains the historical JSON key but now means all robustly
    confirmed pairs before the topology-local partition.  ``candidate_count``
    is the larger (or equal) conservative set returned by Newton's independent
    BVH query.
    """

    nonlocal_count: int
    topologically_local_count: int
    raw_count: int
    candidate_count: int | None = None

    def __post_init__(self) -> None:
        candidate_count = (
            self.raw_count
            if self.candidate_count is None
            else self.candidate_count
        )
        object.__setattr__(self, "candidate_count", candidate_count)
        if min(
            self.nonlocal_count,
            self.topologically_local_count,
            self.raw_count,
            candidate_count,
        ) < 0:
            raise ValueError("Self-intersection counts cannot be negative")
        if self.nonlocal_count + self.topologically_local_count != self.raw_count:
            raise ValueError("Self-intersection count partition is inconsistent")
        if candidate_count < self.raw_count:
            raise ValueError(
                "Confirmed intersections cannot outnumber Newton candidates"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "nonlocal": self.nonlocal_count,
            "topologically_local": self.topologically_local_count,
            "raw": self.raw_count,
            "newton_candidates": int(self.candidate_count),
            "newton_rejected_candidates": int(
                self.candidate_count - self.raw_count
            ),
        }


class IPCTrianglePairConfirmer:
    """Robust IPC confirmation for a candidate set and the complete surface.

    IPC Toolkit is kept as a runtime-only dependency, like Newton and Warp.
    The whole-mesh query prevents a bug in the Newton candidate generator from
    being mistaken for evidence that a state is intersection-free.  When an
    intersection exists, six directed edge--triangle predicates enumerate the
    responsible candidate pairs; these predicates also cover coplanar overlap.
    """

    def __init__(self, triangles: np.ndarray) -> None:
        faces = np.asarray(triangles, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("IPC audit triangles must have shape (count, 3)")
        self.triangles = np.ascontiguousarray(faces)
        self._ipctk = None
        self._collision_mesh = None
        self._edges = None

    def _require_ipctk(self):
        if self._ipctk is None:
            try:
                import ipctk  # noqa: PLC0415
            except ImportError as exception:  # pragma: no cover - runtime-only dependency
                raise RuntimeError(
                    "Robust sewn-garment topology audits require IPC Toolkit; "
                    "run laundry/bootstrap_newton.ps1 first."
                ) from exception
            self._ipctk = ipctk
        return self._ipctk

    def _positions_numpy(self, vertex_positions: Any) -> np.ndarray:
        if hasattr(vertex_positions, "numpy"):
            vertex_positions = vertex_positions.numpy()
        positions = np.ascontiguousarray(vertex_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("IPC audit positions must have shape (count, 3)")
        if self.triangles.size and int(np.max(self.triangles)) >= len(positions):
            raise ValueError("IPC audit topology references a missing position")
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("IPC audit positions contain non-finite values")
        return positions

    def _ensure_collision_mesh(self, positions: np.ndarray):
        ipctk = self._require_ipctk()
        if self._collision_mesh is None:
            edges = np.ascontiguousarray(
                ipctk.edges(self.triangles),
                dtype=np.int32,
            )
            self._edges = edges
            # No user filter is installed here.  IPC's native mesh adjacency
            # exclusions are sufficient for shared primitives; every other
            # triangle pair remains eligible for this fail-closed audit.
            self._collision_mesh = ipctk.CollisionMesh(
                positions,
                edges,
                self.triangles,
            )
        return self._collision_mesh

    def surface_has_intersections(self, vertex_positions: Any) -> bool:
        positions = self._positions_numpy(vertex_positions)
        ipctk = self._require_ipctk()
        mesh = self._ensure_collision_mesh(positions)
        return bool(ipctk.has_intersections(mesh, positions))

    def surface_intersection_pairs(
        self,
        vertex_positions: Any,
    ) -> np.ndarray:
        """Return every exact non-adjacent intersecting triangle pair.

        ``has_intersections`` is the inexpensive Boolean integrity gate.  This
        diagnostic method deliberately performs IPC's complete face--face
        broad phase and then confirms each candidate with the same robust
        directed edge--triangle predicates used for Newton candidates.  It is
        therefore suitable for artifact counts and panel-pair localization;
        it does not reduce an unknown number of crossings to ``int(bool)``.
        """

        positions = self._positions_numpy(vertex_positions)
        ipctk = self._require_ipctk()
        mesh = self._ensure_collision_mesh(positions)
        # IPC's compiled whole-surface predicate is conservative for sewn
        # touching seams. A false result proves there are no crossings; a true
        # result still requires the exact pair audit below (touch is not cross).
        if not ipctk.has_intersections(mesh, positions):
            return np.empty((0, 2), dtype=np.int32)
        broad_phase = ipctk.SweepAndPrune()
        broad_phase.can_vertices_collide = mesh.can_collide
        broad_phase.build(positions, self._edges, self.triangles, 0.0)
        pairs = np.asarray([(int(c.face0_id), int(c.face1_id))
            for c in broad_phase.detect_face_face_candidates()], dtype=np.int32).reshape(-1, 2)
        pairs = pairs[~separated_face_pairs(positions, self.triangles, pairs)]
        if len(pairs):
            pairs = pairs[~projected_edge_separation(positions, self.triangles, pairs)]
        confirmed: set[tuple[int, int]] = set()
        for first, second in pairs:
            pair = (min(int(first), int(second)), max(int(first), int(second)))
            if self.pair_intersects(positions, *pair):
                confirmed.add(pair)
        if not confirmed:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(sorted(confirmed), dtype=np.int32)

    def pair_intersects(
        self,
        positions: np.ndarray,
        triangle_a: int,
        triangle_b: int,
    ) -> bool:
        """Return IPC's exact discrete result for one triangle pair."""

        ipctk = self._require_ipctk()
        face_a = positions[self.triangles[triangle_a]]
        face_b = positions[self.triangles[triangle_b]]
        directed_pairs = ((face_a, face_b), (face_b, face_a))
        for edge_face, target_face in directed_pairs:
            for start, end in ((0, 1), (1, 2), (2, 0)):
                if ipctk.is_edge_intersecting_triangle(
                    edge_face[start],
                    edge_face[end],
                    target_face[0],
                    target_face[1],
                    target_face[2],
                ):
                    return True
        return False

    def confirmed_pairs(
        self,
        vertex_positions: Any,
        candidate_pairs: np.ndarray,
    ) -> np.ndarray:
        """Filter candidates while independently checking the whole surface."""

        positions = self._positions_numpy(vertex_positions)
        candidates = np.asarray(candidate_pairs, dtype=np.int32)
        if candidates.size == 0:
            candidates = np.empty((0, 2), dtype=np.int32)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("IPC candidate pairs must have shape (count, 2)")

        ipctk = self._require_ipctk()
        mesh = self._ensure_collision_mesh(positions)
        if not bool(ipctk.has_intersections(mesh, positions)):
            return np.empty((0, 2), dtype=np.int32)

        confirmed = [
            (int(triangle_a), int(triangle_b))
            for triangle_a, triangle_b in candidates
            if self.pair_intersects(
                positions,
                int(triangle_a),
                int(triangle_b),
            )
        ]
        if not confirmed:
            # IPC's closed-set Boolean also reports valid sewn seam touches.
            # Do not equate that positive flag with a proper surface crossing,
            # or assume that Newton's candidate list is exhaustive. Audit every
            # broad-phase face pair with the robust directed-edge predicate.
            return self.surface_intersection_pairs(positions)
        return np.asarray(confirmed, dtype=np.int32)


class NewtonTriangleIntersectionAuditor:
    """Newton candidate generation with robust IPC intersection confirmation."""

    def __init__(
        self,
        model: Any,
        triangles: np.ndarray,
        *,
        local_exclusion_graph_hops: int = 1,
        buffer_per_triangle: int = 256,
    ) -> None:
        faces = np.asarray(triangles, dtype=np.int32)
        if faces.shape != (int(model.tri_count), 3):
            raise ValueError("Audit topology must match the Newton model triangle count")
        if buffer_per_triangle < 1:
            raise ValueError("Intersection audit buffer size must be positive")
        self.model = model
        self.triangles = faces.copy()
        self.buffer_per_triangle = int(buffer_per_triangle)
        self.vertex_neighbourhoods = build_vertex_neighbourhoods(
            self.triangles,
            vertex_count=int(model.particle_count),
            graph_hops=local_exclusion_graph_hops,
        )
        self.local_exclusion_graph_hops = int(local_exclusion_graph_hops)
        self._detector = None
        self._ipc_confirmer = IPCTrianglePairConfirmer(self.triangles)

    def _ensure_detector(self, vertex_positions: Any):
        if self._detector is None:
            # Newton intentionally remains a runtime-only dependency.  The
            # repository's normal unit-test Python can import the pure topology
            # helpers above without installing Warp/Newton.
            from newton._src.solvers.vbd.tri_mesh_collision import (  # noqa: PLC0415
                TriMeshCollisionDetector,
            )

            self._detector = TriMeshCollisionDetector(
                self.model,
                vertex_positions=vertex_positions,
                triangle_triangle_collision_buffer_pre_alloc=(
                    self.buffer_per_triangle
                ),
                triangle_triangle_collision_buffer_max_alloc=(
                    self.buffer_per_triangle
                ),
            )
        return self._detector

    def raw_pairs(self, vertex_positions: Any) -> np.ndarray:
        """Return unique conservative Newton triangle-pair candidates."""

        detector = self._ensure_detector(vertex_positions)
        detector.vertex_positions = vertex_positions
        detector.resize_flags.zero_()
        detector.refit_triangles()
        detector.triangle_triangle_intersection_detection()
        # Newton 1.4 assigns resize flag slot three to the triangle--triangle
        # buffer.  An overflow must fail closed; under-counting is not evidence.
        if int(detector.resize_flags.numpy()[3]) != 0:
            raise RuntimeError("Triangle-intersection audit buffer overflowed")
        counts = detector.triangle_intersecting_triangles_count.numpy()
        offsets = detector.triangle_intersecting_triangles_offsets.numpy()
        hits = detector.triangle_intersecting_triangles.numpy()
        pairs: set[tuple[int, int]] = set()
        for triangle, count in enumerate(counts):
            begin = int(offsets[triangle])
            for other in hits[begin : begin + int(count)]:
                other_triangle = int(other)
                if other_triangle < 0 or other_triangle == triangle:
                    continue
                pairs.add(
                    (
                        min(triangle, other_triangle),
                        max(triangle, other_triangle),
                    )
                )
        if not pairs:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(sorted(pairs), dtype=np.int32)

    def confirmed_pairs(self, vertex_positions: Any) -> np.ndarray:
        """Return robust IPC intersections from Newton's candidate set."""

        candidates = self.raw_pairs(vertex_positions)
        return self._ipc_confirmer.confirmed_pairs(
            vertex_positions,
            candidates,
        )

    def pair_is_topologically_local(self, triangle_a: int, triangle_b: int) -> bool:
        vertices_b = set(int(value) for value in self.triangles[triangle_b])
        return any(
            not vertices_b.isdisjoint(self.vertex_neighbourhoods[int(vertex)])
            for vertex in self.triangles[triangle_a]
        )

    def nonlocal_pairs(self, vertex_positions: Any) -> np.ndarray:
        confirmed = self.confirmed_pairs(vertex_positions)
        retained = [
            (int(a), int(b))
            for a, b in confirmed
            if not self.pair_is_topologically_local(int(a), int(b))
        ]
        if not retained:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(retained, dtype=np.int32)

    def counts(self, vertex_positions: Any) -> SelfIntersectionAuditCounts:
        candidates = self.raw_pairs(vertex_positions)
        confirmed = self._ipc_confirmer.confirmed_pairs(
            vertex_positions,
            candidates,
        )
        nonlocal_count = sum(
            not self.pair_is_topologically_local(int(a), int(b))
            for a, b in confirmed
        )
        return SelfIntersectionAuditCounts(
            nonlocal_count=int(nonlocal_count),
            topologically_local_count=int(len(confirmed) - nonlocal_count),
            raw_count=int(len(confirmed)),
            candidate_count=int(len(candidates)),
        )


__all__ = [
    "IPCTrianglePairConfirmer",
    "NewtonTriangleIntersectionAuditor",
    "SelfIntersectionAuditCounts",
    "build_vertex_neighbourhoods",
]

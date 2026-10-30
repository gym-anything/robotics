"""Audited visible-shoe geometry for reduced-order ball contact.

The Unitree URDF represents each sole with four 5 mm spheres.  Those shapes do
not cover the instep used to kick a football.  Production contact therefore
uses a deterministic convex hull built from the pinned visible right-shoe STL.
The older oriented bounding box remains available as an explicit coupon and as
an audit reference, but it is not the production geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .football_math import (
    Quat,
    Vec3,
    add,
    clamp,
    norm,
    quat_inverse_rotate,
    quat_rotate,
    scale,
    sub,
    unit,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RIGHT_FOOT_PROXY_PATH = (
    HERE / "assets" / "g1_right_foot_convex_proxy.json"
)


@dataclass(frozen=True)
class FootPatchSpec:
    """Pinned right-shoe contact geometry and fallback mass prior."""

    minimum_local_m: Vec3 = (
        -0.06584136933088303,
        -0.03784090280532837,
        -0.035409145057201385,
    )
    maximum_local_m: Vec3 = (
        0.14236535131931305,
        0.03774179145693779,
        0.023583902046084404,
    )
    proxy_relative_path: str = "assets/g1_right_foot_convex_proxy.json"
    proxy_asset_sha256: str = (
        "232A588BEF5C4364DAA573C77F6F78874F17438F7A6F2CBF2EFBD6D1B46E80FD"
    )
    source_mesh_sha256: str = (
        "4B66222EA56653E627711B56D0A8949B4920DA5DF091DA0CEB343F54E884E3A5"
    )
    effective_mass_prior_kg: float = 2.0
    evidence_status: str = (
        "PINNED VISIBLE-STL CONVEX-HULL CONTACT PROXY / CONVEXITY FILLS "
        "MESH CONCAVITIES / ISOTROPIC MASS FALLBACK PRIOR / "
        "NOT HARDWARE-CALIBRATED"
    )

    def __post_init__(self) -> None:
        if any(a >= b for a, b in zip(self.minimum_local_m, self.maximum_local_m)):
            raise ValueError("foot patch bounds must have positive extent")
        if self.effective_mass_prior_kg <= 0.0:
            raise ValueError("foot effective mass must be positive")
        if not self.proxy_relative_path:
            raise ValueError("foot proxy path must not be empty")
        for label, digest in (
            ("proxy asset", self.proxy_asset_sha256),
            ("source mesh", self.source_mesh_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789ABCDEF" for character in digest
            ):
                raise ValueError(f"{label} SHA-256 must be uppercase hexadecimal")


@dataclass(frozen=True)
class SphereFootContactGeometry:
    signed_distance_m: float
    compression_m: float
    normal_foot_to_ball_world: Vec3
    closest_point_on_foot_world_m: Vec3
    ball_center_in_foot_local_m: Vec3
    closest_point_in_foot_local_m: Vec3
    feature: str
    proxy_kind: str


@dataclass(frozen=True)
class ConvexSurfaceQuery:
    signed_distance_m: float
    closest_point_local_m: Vec3
    outward_normal_local: Vec3
    feature: str
    triangle_index: int


class ConvexFootProxy:
    """Immutable, vectorized point-query representation of a convex shoe."""

    schema = "football-g1-right-foot-convex-proxy-v1"

    def __init__(
        self,
        *,
        path: Path,
        asset_sha256: str,
        payload: dict[str, Any],
    ) -> None:
        if payload.get("schema") != self.schema:
            raise ValueError(
                f"unsupported foot proxy schema: {payload.get('schema')!r}"
            )
        vertices = np.asarray(payload.get("vertices_m"), dtype=np.float64)
        faces = np.asarray(payload.get("triangles"), dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("foot proxy vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError("foot proxy triangles must have shape (M, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("foot proxy vertices contain non-finite values")
        if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
            raise ValueError("foot proxy contains an out-of-range triangle index")

        triangles = vertices[faces]
        edges_ab = triangles[:, 1] - triangles[:, 0]
        edges_ac = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edges_ab, edges_ac)
        double_areas = np.linalg.norm(normals, axis=1)
        if float(np.min(double_areas)) <= 1.0e-15:
            raise ValueError("foot proxy contains a degenerate triangle")
        normals /= double_areas[:, None]
        offsets = -np.einsum("ij,ij->i", normals, triangles[:, 0])
        interior = np.mean(vertices, axis=0)
        maximum_interior_residual = float(
            np.max(normals @ interior + offsets)
        )
        if maximum_interior_residual >= -1.0e-9:
            raise ValueError(
                "foot proxy facets are not consistently outward: "
                f"interior residual {maximum_interior_residual}"
            )

        geometry = payload.get("geometry", {})
        declared_minimum = np.asarray(
            geometry.get("minimum_local_m"), dtype=np.float64
        )
        declared_maximum = np.asarray(
            geometry.get("maximum_local_m"), dtype=np.float64
        )
        if (
            declared_minimum.shape != (3,)
            or declared_maximum.shape != (3,)
            or not np.allclose(declared_minimum, vertices.min(axis=0), atol=1e-12)
            or not np.allclose(declared_maximum, vertices.max(axis=0), atol=1e-12)
        ):
            raise ValueError("foot proxy declared bounds do not match its vertices")

        self.path = path
        self.asset_sha256 = asset_sha256
        self.payload = payload
        self.vertices_m = vertices
        self.faces = faces
        self.triangles = triangles
        self.a = triangles[:, 0]
        self.b = triangles[:, 1]
        self.c = triangles[:, 2]
        self.ab = edges_ab
        self.ac = edges_ac
        self.bc = triangles[:, 2] - triangles[:, 1]
        self.normals = normals
        self.offsets = offsets
        self.triangle_minimum_m = triangles.min(axis=1)
        self.triangle_maximum_m = triangles.max(axis=1)
        self.ab_dot_ab = np.einsum("ij,ij->i", self.ab, self.ab)
        self.ab_dot_ac = np.einsum("ij,ij->i", self.ab, self.ac)
        self.ac_dot_ac = np.einsum("ij,ij->i", self.ac, self.ac)
        self.barycentric_denominator = (
            self.ab_dot_ab * self.ac_dot_ac
            - self.ab_dot_ac * self.ab_dot_ac
        )
        self.minimum_local_m = vertices.min(axis=0)
        self.maximum_local_m = vertices.max(axis=0)
        self.interior_point_m = interior

        for array in (
            self.vertices_m,
            self.faces,
            self.triangles,
            self.a,
            self.b,
            self.c,
            self.ab,
            self.ac,
            self.bc,
            self.normals,
            self.offsets,
            self.triangle_minimum_m,
            self.triangle_maximum_m,
            self.ab_dot_ab,
            self.ab_dot_ac,
            self.ac_dot_ac,
            self.barycentric_denominator,
            self.minimum_local_m,
            self.maximum_local_m,
            self.interior_point_m,
        ):
            array.setflags(write=False)

    @staticmethod
    def _segment_closest(
        point_rows: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        direction = end - start
        fraction = np.einsum(
            "ij,ij->i", point_rows - start, direction
        ) / np.einsum("ij,ij->i", direction, direction)
        fraction = np.clip(fraction, 0.0, 1.0)
        return start + fraction[:, None] * direction

    def query_local(self, point_local_m: Vec3) -> ConvexSurfaceQuery:
        """Return the exact closest point on the triangulated convex proxy."""

        point = np.asarray(point_local_m, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("foot query point must be a finite 3-vector")
        # Exact branch-and-bound without a Python tree walk: the nearest hull
        # vertex supplies a valid upper distance bound, while each triangle
        # AABB supplies a lower bound.  A triangle whose lower bound exceeds
        # the upper bound cannot contain the closest point.  This typically
        # reduces the 2,492-facet hull to a few dozen candidates, without an
        # approximation or a contact-distance threshold.
        vertex_delta = self.vertices_m - point
        upper_distance_squared = float(
            np.min(np.einsum("ij,ij->i", vertex_delta, vertex_delta))
        )
        aabb_delta = np.maximum(
            np.maximum(
                self.triangle_minimum_m - point,
                point - self.triangle_maximum_m,
            ),
            0.0,
        )
        aabb_lower_distance_squared = np.einsum(
            "ij,ij->i", aabb_delta, aabb_delta
        )
        candidate_triangles = np.flatnonzero(
            aabb_lower_distance_squared <= upper_distance_squared + 1.0e-15
        )
        if len(candidate_triangles) == 0:
            raise RuntimeError("convex foot query pruned every triangle")

        a = self.a[candidate_triangles]
        b = self.b[candidate_triangles]
        c = self.c[candidate_triangles]
        ab = self.ab[candidate_triangles]
        ac = self.ac[candidate_triangles]
        normals = self.normals[candidate_triangles]
        point_rows = np.broadcast_to(point, a.shape)

        plane_distance = np.einsum(
            "ij,ij->i", point_rows - a, normals
        )
        face_projection = point_rows - plane_distance[:, None] * normals

        projection_offset = face_projection - a
        d00 = self.ab_dot_ab[candidate_triangles]
        d01 = self.ab_dot_ac[candidate_triangles]
        d11 = self.ac_dot_ac[candidate_triangles]
        denominator = self.barycentric_denominator[candidate_triangles]
        d20 = np.einsum("ij,ij->i", projection_offset, ab)
        d21 = np.einsum("ij,ij->i", projection_offset, ac)
        barycentric_b = (d11 * d20 - d01 * d21) / denominator
        barycentric_c = (d00 * d21 - d01 * d20) / denominator
        barycentric_a = 1.0 - barycentric_b - barycentric_c
        face_projection_is_inside = (
            (barycentric_a >= -1.0e-12)
            & (barycentric_b >= -1.0e-12)
            & (barycentric_c >= -1.0e-12)
        )

        closest_ab = self._segment_closest(point_rows, a, b)
        closest_bc = self._segment_closest(point_rows, b, c)
        closest_ca = self._segment_closest(point_rows, c, a)
        candidates = np.stack(
            (face_projection, closest_ab, closest_bc, closest_ca), axis=1
        )
        candidate_delta = candidates - point_rows[:, None, :]
        distance_squared = np.einsum(
            "ijk,ijk->ij", candidate_delta, candidate_delta
        )
        distance_squared[:, 0] = np.where(
            face_projection_is_inside,
            distance_squared[:, 0],
            np.inf,
        )
        flat_index = int(np.argmin(distance_squared))
        local_triangle_index, candidate_index = np.unravel_index(
            flat_index, distance_squared.shape
        )
        triangle_index = int(candidate_triangles[local_triangle_index])
        closest = candidates[local_triangle_index, candidate_index]
        distance = float(np.linalg.norm(point - closest))

        maximum_halfspace_value = float(
            np.max(self.normals @ point + self.offsets)
        )
        inside = maximum_halfspace_value <= 1.0e-10
        if distance > 1.0e-12:
            normal = (
                (closest - point) / distance
                if inside
                else (point - closest) / distance
            )
        else:
            normal = self.normals[triangle_index]
        location = "face" if candidate_index == 0 else "edge"
        state = "inside" if inside else ("surface" if distance <= 1e-12 else "outside")
        return ConvexSurfaceQuery(
            signed_distance_m=-distance if inside else distance,
            closest_point_local_m=tuple(float(value) for value in closest),
            outward_normal_local=tuple(float(value) for value in normal),
            feature=f"{state}_{location}",
            triangle_index=triangle_index,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "schema": self.payload["schema"],
            "model_status": self.payload["model_status"],
            "path": str(self.path),
            "asset_sha256": self.asset_sha256,
            "source": dict(self.payload["source"]),
            "builder": dict(self.payload["builder"]),
            "geometry": dict(self.payload["geometry"]),
            "runtime_query": (
                "exact closest point with nearest-vertex upper bound and "
                "triangle-AABB lower-bound pruning; no OBB narrow-phase "
                "substitution or distance threshold"
            ),
        }


@lru_cache(maxsize=4)
def load_convex_foot_proxy(path: str | Path = DEFAULT_RIGHT_FOOT_PROXY_PATH) -> ConvexFootProxy:
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    asset_sha256 = hashlib.sha256(raw).hexdigest().upper()
    payload = json.loads(raw.decode("utf-8"))
    return ConvexFootProxy(
        path=resolved,
        asset_sha256=asset_sha256,
        payload=payload,
    )


def validate_proxy_against_patch(
    proxy: ConvexFootProxy,
    patch: FootPatchSpec,
) -> None:
    if proxy.asset_sha256 != patch.proxy_asset_sha256:
        raise ValueError(
            "foot proxy asset hash mismatch: "
            f"expected {patch.proxy_asset_sha256}, got {proxy.asset_sha256}"
        )
    source_sha256 = str(proxy.payload["source"]["sha256"])
    if source_sha256 != patch.source_mesh_sha256:
        raise ValueError(
            "foot proxy source hash mismatch: "
            f"expected {patch.source_mesh_sha256}, got {source_sha256}"
        )
    if not np.allclose(
        proxy.minimum_local_m,
        np.asarray(patch.minimum_local_m),
        atol=1.0e-12,
    ) or not np.allclose(
        proxy.maximum_local_m,
        np.asarray(patch.maximum_local_m),
        atol=1.0e-12,
    ):
        raise ValueError("foot patch bounds do not match the pinned convex proxy")


def sphere_convex_contact(
    *,
    ball_center_world_m: Vec3,
    ball_radius_m: float,
    foot_center_world_m: Vec3,
    foot_orientation_world_wxyz: Quat,
    patch: FootPatchSpec = FootPatchSpec(),
    proxy: ConvexFootProxy | None = None,
) -> SphereFootContactGeometry:
    """Return sphere-vs-convex-shoe geometry with an outward shoe normal."""

    if ball_radius_m <= 0.0:
        raise ValueError("ball radius must be positive")
    if proxy is None:
        proxy = load_convex_foot_proxy(HERE / patch.proxy_relative_path)
    validate_proxy_against_patch(proxy, patch)
    center_local = quat_inverse_rotate(
        foot_orientation_world_wxyz,
        sub(ball_center_world_m, foot_center_world_m),
    )
    query = proxy.query_local(center_local)
    normal_world = unit(
        quat_rotate(foot_orientation_world_wxyz, query.outward_normal_local)
    )
    closest_world = add(
        foot_center_world_m,
        quat_rotate(
            foot_orientation_world_wxyz,
            query.closest_point_local_m,
        ),
    )
    return SphereFootContactGeometry(
        signed_distance_m=query.signed_distance_m,
        compression_m=max(0.0, ball_radius_m - query.signed_distance_m),
        normal_foot_to_ball_world=normal_world,
        closest_point_on_foot_world_m=closest_world,
        ball_center_in_foot_local_m=center_local,
        closest_point_in_foot_local_m=query.closest_point_local_m,
        feature=f"{query.feature}_triangle_{query.triangle_index}",
        proxy_kind="convex_hull_of_pinned_visible_right_shoe_stl",
    )


def sphere_obb_contact(
    *,
    ball_center_world_m: Vec3,
    ball_radius_m: float,
    foot_center_world_m: Vec3,
    foot_orientation_world_wxyz: Quat,
    patch: FootPatchSpec = FootPatchSpec(),
) -> SphereFootContactGeometry:
    """Return the legacy sphere-vs-visible-bounds OBB audit geometry."""

    if ball_radius_m <= 0.0:
        raise ValueError("ball radius must be positive")
    center_local = quat_inverse_rotate(
        foot_orientation_world_wxyz,
        sub(ball_center_world_m, foot_center_world_m),
    )
    closest_local = tuple(
        clamp(value, minimum, maximum)
        for value, minimum, maximum in zip(
            center_local, patch.minimum_local_m, patch.maximum_local_m
        )
    )
    outside_delta = sub(center_local, closest_local)
    outside_distance = norm(outside_delta)
    if outside_distance > 1.0e-12:
        normal_local = scale(1.0 / outside_distance, outside_delta)
        signed_distance = outside_distance
        feature = "outside"
    else:
        candidates: list[tuple[float, int, float]] = []
        for axis in range(3):
            candidates.append(
                (center_local[axis] - patch.minimum_local_m[axis], axis, -1.0)
            )
            candidates.append(
                (patch.maximum_local_m[axis] - center_local[axis], axis, 1.0)
            )
        face_distance, axis, sign = min(candidates, key=lambda item: item[0])
        mutable = list(center_local)
        mutable[axis] = (
            patch.maximum_local_m[axis]
            if sign > 0.0
            else patch.minimum_local_m[axis]
        )
        closest_local = tuple(mutable)  # type: ignore[assignment]
        normal_values = [0.0, 0.0, 0.0]
        normal_values[axis] = sign
        normal_local = tuple(normal_values)  # type: ignore[assignment]
        signed_distance = -face_distance
        feature = f"inside_{'xyz'[axis]}_{'max' if sign > 0.0 else 'min'}"
    normal_world = unit(quat_rotate(foot_orientation_world_wxyz, normal_local))
    closest_world = add(
        foot_center_world_m,
        quat_rotate(foot_orientation_world_wxyz, closest_local),
    )
    return SphereFootContactGeometry(
        signed_distance_m=signed_distance,
        compression_m=max(0.0, ball_radius_m - signed_distance),
        normal_foot_to_ball_world=normal_world,
        closest_point_on_foot_world_m=closest_world,
        ball_center_in_foot_local_m=center_local,
        closest_point_in_foot_local_m=closest_local,
        feature=feature,
        proxy_kind="legacy_oriented_visible_stl_bounds_box",
    )
